# WARP: An Efficient Engine for Multi-Vector Retrieval

# WARP：一种高效的多向量检索引擎

Jan Luca Scheerer*

扬·卢卡·舍雷尔（Jan Luca Scheerer）*

lscheerer@ethz.ch

ETH Zurich

苏黎世联邦理工学院（ETH Zurich）

Switzerland

瑞士

Matei Zaharia

马特·扎哈里亚（Matei Zaharia）

matei@berkeley.edu

UC Berkeley

加州大学伯克利分校（UC Berkeley）

United States

美国

Christopher Potts

克里斯托弗·波茨（Christopher Potts）

cgpotts@stanford.edu

Stanford University

斯坦福大学（Stanford University）

United States

美国

Gustavo Alonso

古斯塔沃·阿隆索（Gustavo Alonso）

alonso@inf.ethz.ch

ETH Zurich

苏黎世联邦理工学院（ETH Zurich）

Switzerland

瑞士

Omar Khattab

奥马尔·哈塔卜（Omar Khattab）

okhattab@cs.stanford.edu

Stanford University

斯坦福大学（Stanford University）

United States

美国

## Abstract

## 摘要

We study the efficiency of multi-vector retrieval methods like ColBERT and its recent variant XTR. We introduce WARP, a retrieval engine that drastically improves the efficiency of XTR-based ColBERT retrievers through three key innovations: (1) WARPSELECT for dynamic similarity imputation, (2) implicit decompression during retrieval, and (3) a two-stage reduction process for efficient scoring. Thanks also to highly-optimized C++ kernels and to the adoption of specialized inference runtimes, WARP can reduce end-to-end query latency relative to XTR's reference implementation by ${41}\mathrm{x}$ ,and thereby achieves a $3\mathrm{x}$ speedup over the official ColBERTv2 PLAID engine, while preserving retrieval quality.

我们研究了诸如ColBERT及其近期变体XTR等多向量检索方法的效率。我们推出了WARP，这是一款检索引擎，它通过三项关键创新大幅提升了基于XTR的ColBERT检索器的效率：（1）用于动态相似度估算的WARPSELECT；（2）检索过程中的隐式解压缩；（3）用于高效评分的两阶段缩减过程。得益于高度优化的C++内核以及采用专门的推理运行时，与XTR的参考实现相比，WARP可将端到端查询延迟降低${41}\mathrm{x}$，从而比官方的ColBERTv2 PLAID引擎实现了$3\mathrm{x}$的加速，同时保持了检索质量。

C https://github.com/jlscheerer/xtr-warp

C https://github.com/jlscheerer/xtr-warp

## Keywords

## 关键词

Dense Retrieval, Multi-Vector, Late Interaction, Efficiency

密集检索、多向量、后期交互、效率

## ACM Reference Format:

## ACM引用格式：

Jan Luca Scheerer, Matei Zaharia, Christopher Potts, Gustavo Alonso, and Omar Khattab. 2025. WARP: An Efficient Engine for Multi-Vector Retrieval. In Preprint. Preprint, 9 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnnn

扬·卢卡·舍雷尔（Jan Luca Scheerer）、马泰·扎哈里亚（Matei Zaharia）、克里斯托弗·波茨（Christopher Potts）、古斯塔沃·阿隆索（Gustavo Alonso）和奥马尔·哈塔卜（Omar Khattab）。2025年。WARP：用于多向量检索的高效引擎。预印本。预印本，9页。https://doi.org/10.1145/nnnnnnn.nnnnnnnn

## 1 Introduction

## 1 引言

Over the past several years, information retrieval (IR) research has introduced new neural paradigms for search based on pretrained Transformers. Central among these, the late interaction paradigm proposed in ColBERT [8] departs the bottlenecks of conventional single-vector representations. It instead encodes queries and documents into multi-vector representations on top of which it is able to scale gracefully to search massive collections.

在过去几年里，信息检索（IR）研究引入了基于预训练Transformer的新型神经搜索范式。其中，ColBERT [8]中提出的后期交互范式突破了传统单向量表示的瓶颈。它将查询和文档编码为多向量表示，在此基础上能够优雅地扩展以搜索大规模文档集合。

Since the original ColBERT was introduced, there has been substantial research in optimizing the latency of multi-vector retrieval models [2, 12, 13]. Perhaps most notably, PLAID [16] reduces late interaction search latency by ${45x}$ on a CPU compared to a vanilla ColBERTv2 [17] process, while continuing to deliver state-of-the-art retrieval quality. Orthogonally, Google DeepMind's ConteXtualized Token Retriever (XTR) [10] introduces a novel training objective that eliminates the need for a separate gathering stage and thereby significantly simplifies the subsequent scoring stage. While XTR lays extremely promising groundwork for more efficient multi-vector retrieval ${}^{1}$ ,we find that it relies naively on a general purpose vector similarity search library (ScANN) and combines that with native Python data structures and manual iteration, introducing substantial overhead.

自最初的ColBERT推出以来，在优化多向量检索模型的延迟方面已有大量研究[2, 12, 13]。最值得注意的是，与普通的ColBERTv2 [17]流程相比，PLAID [16]在CPU上可将后期交互搜索延迟降低${45x}$，同时仍能提供最先进的检索质量。另一方面，谷歌DeepMind的上下文Token检索器（ConteXtualized Token Retriever，XTR）[10]引入了一种新颖的训练目标，消除了单独的收集阶段的需求，从而显著简化了后续的评分阶段。虽然XTR为更高效的多向量检索奠定了极具前景的基础${}^{1}$，但我们发现它简单地依赖于通用向量相似度搜索库（ScANN），并将其与原生Python数据结构和手动迭代相结合，引入了大量开销。

<!-- Media -->

<!-- figureText: ${\mathrm{{XTR}}}_{\text{base }}/\mathrm{{ScaN}}\mathrm{A}$ 6862ms 2155ms Query Encoding Filtering Decompression Scoring Latency (ms) (k’ = 40000, opt=False) ${\mathrm{{XTR}}}_{\text{base }}/\mathrm{{ScaNN}}$ (k’ = 40000, opt=True) ColBERT ${}_{\mathrm{v}2}/$ PLAID 476ms ${\mathrm{{XTR}}}_{\mathrm{{base}}}/\mathrm{{WARP}}$ 171ms ( ${\mathrm{n}}_{\text{probe }} = {32}$ ) -->

<img src="https://cdn.noedgeai.com/0195afaa-ab63-7b7c-94ef-22219448cc3a_0.jpg?x=947&y=640&w=674&h=294&r=0"/>

Figure 1: Single-threaded CPU latency breakdown of the unoptimized reference implementation from (1) XTR, ${}^{2}\left( 2\right)$ a variant of XTR that we optimized, (3) the official PLAID system, and (4) our proposed WARP on LoTTE Pooled.

图1：在LoTTE Pooled数据集上，（1）XTR的未优化参考实现、（2）我们优化后的XTR变体、（3）官方PLAID系统以及（4）我们提出的WARP的单线程CPU延迟分解。

<!-- Media -->

The key insights from PLAID and XTR appear rather isolated. Whereas PLAID is concerned with aggressively and swiftly pruning away documents it finds unpromising, XTR tries to eliminate gathering complete document representations in the first place. We ask whether there are potential rich interactions between these two fundamentally distinct approaches to speeding up multi-vector search. To study this, we introduce a new engine for retrieval with XTR-based ColBERT models, called WARP, that combines techniques from ColBERTv2/PLAID with innovations tailored for the XTR architecture. Our contributions in WARP include: (1) the WARP method for imputing missing similarities, (2) a new method for implicit decompression of vectors during search, and (3) a novel two-stage reduction phase for efficient scoring.

PLAID和XTR的关键见解似乎相当孤立。PLAID致力于积极迅速地剔除它认为没有前景的文档，而XTR则试图从根本上消除收集完整文档表示的需求。我们不禁要问，这两种从根本上不同的加速多向量搜索的方法之间是否存在潜在的丰富交互。为了研究这一点，我们推出了一款新的基于XTR的ColBERT模型检索引擎，名为WARP，它将ColBERTv2/PLAID的技术与针对XTR架构量身定制的创新相结合。我们在WARP中的贡献包括：（1）用于估算缺失相似度的WARP方法；（2）一种在搜索过程中对向量进行隐式解压缩的新方法；（3）一种用于高效评分的新颖两阶段缩减阶段。

Experimental evaluation shows that WARP achieves a ${41}\mathrm{x}$ reduction in end-to-end latency compared to the XTR reference implementation on LoTTE Pooled, bringing query response times down from above 6 seconds to just 171 milliseconds in single-threaded execution,while also reducing index size by a factor of $2\mathrm{x} - 4\mathrm{x}$ compared to the ScaNN-based baseline. Furthermore, WARP demonstrates a $3\mathrm{x}$ speedup over the state-of-the-art ColBERTv2/PLAID system,as illustrated in Figure 1.

实验评估表明，与LoTTE Pooled数据集上的XTR参考实现相比，WARP可将端到端延迟降低${41}\mathrm{x}$，在单线程执行中，查询响应时间从6秒以上降至仅171毫秒，同时与基于ScaNN的基线相比，索引大小缩小了$2\mathrm{x} - 4\mathrm{x}$倍。此外，如图1所示，WARP比最先进的ColBERTv2/PLAID系统实现了$3\mathrm{x}$的加速。

---

<!-- Footnote -->

*Work completed as a visiting student researcher at Stanford University.

*本工作是作为斯坦福大学的访问学生研究员完成的。

${}^{1}$ https://github.com/google-deepmind/xtr

${}^{1}$ https://github.com/google-deepmind/xtr

<!-- Footnote -->

---

After briefly reviewing prior work on efficient neural information retrieval in Section 2, we analyze the latency bottlenecks in the ColBERT and XTR retrieval frameworks in Section 3, identifying key areas for optimization within the XTR framework. These findings form the foundation for our work on WARP, which we introduce and describe in detail in Section 4. In Section 5, we evaluate WARP's end-to-end latency and scalability using the BEIR [18] and LoTTE [17] benchmarks. Finally, we compare our implementation to existing state-of-the-art engines.

在第2节简要回顾了高效神经信息检索的先前工作之后，我们在第3节分析了ColBERT和XTR检索框架中的延迟瓶颈，确定了XTR框架内需要优化的关键领域。这些发现构成了我们对WARP研究的基础，我们将在第4节详细介绍和描述WARP。在第5节中，我们使用BEIR [18]和LoTTE [17]基准评估WARP的端到端延迟和可扩展性。最后，我们将我们的实现与现有的最先进引擎进行比较。

## 2 Related Work

## 2 相关工作

Dense retrieval models can be broadly categorized into single-vector and multi-vector approaches. Single-vector methods, exemplified by ANCE [19] and STAR/ADORE [20], encode a passage into a single dense vector [7]. While these techniques offer computational efficiency, their inherent limitation of representing complex documents with a single vector has been shown to constrain the model's ability to capture intricate information structures [8].

密集检索模型大致可分为单向量和多向量方法。以ANCE [19]和STAR/ADORE [20]为代表的单向量方法将段落编码为单个密集向量 [7]。虽然这些技术具有计算效率，但它们固有的用单个向量表示复杂文档的局限性已被证明会限制模型捕捉复杂信息结构的能力 [8]。

To address such limitations, ColBERT [8] introduces a multi-vector paradigm. In this approach, both queries and documents are independently encoded as multiple embeddings, allowing for a richer representation of document content and query intent. The multi-vector approach is further refined in ColBERTv2 [17], which improves supervision and incorporates residual compression to significantly reduce the space requirements associated with storing multiple vectors per indexed document. Building upon these innovations, PLAID [16] substantially accelerates ColBERTv2 by efficient pruning non-relevant passages using the residual representation and by employing optimized C++ kernels. EMVB [13] further optimizes PLAID's memory usage and single-threaded query latency using product quantization [6] and SIMD instructions.

为了解决这些局限性，ColBERT [8]引入了多向量范式。在这种方法中，查询和文档都被独立编码为多个嵌入，从而能够更丰富地表示文档内容和查询意图。多向量方法在ColBERTv2 [17]中得到了进一步改进，它改进了监督机制并引入了残差压缩，以显著减少每个索引文档存储多个向量所需的空间。在这些创新的基础上，PLAID [16]通过使用残差表示有效修剪不相关段落并采用优化的C++内核，大幅加速了ColBERTv2。EMVB [13]使用乘积量化 [6]和单指令多数据（SIMD）指令进一步优化了PLAID的内存使用和单线程查询延迟。

Separately, COIL [2] incorporates insights from conventional retrieval systems [15] by constraining token interactions to lexical matches between queries and documents. SPLATE [1] translates the embeddings produced by ColBERTv2 style pipelines to a sparse vocabulary, allowing the candidate generation step to be performed using traditional sparse retrieval techniques. CITADEL [12] introduces conditional token interaction through dynamic lexical routing, selectively considering tokens for relevance estimation. While CITADEL significantly reduces GPU execution time, it falls short of PLAID's CPU performance at comparable retrieval quality.

另外，COIL [2]通过将Token（标记）交互限制为查询和文档之间的词法匹配，融入了传统检索系统 [15]的见解。SPLATE [1]将ColBERTv2风格管道生成的嵌入转换为稀疏词汇表，从而允许使用传统的稀疏检索技术执行候选生成步骤。CITADEL [12]通过动态词法路由引入了条件Token交互，有选择地考虑Token进行相关性估计。虽然CITADEL显著减少了GPU执行时间，但在可比的检索质量下，其CPU性能不如PLAID。

The Contextualized Token Retriever (XTR) [10], introduced by Lee et al. [10], represents a notable conceptual advancement in dense retrieval. XTR simplifies the scoring process and eliminates the gathering stage entirely, theoretically enhancing retrieval efficiency. However, its current end-to-end latency limits its application in production environments where query response time is critical or GPU resources are constrained.

Lee等人 [10]提出的上下文Token检索器（Contextualized Token Retriever，XTR） [10]代表了密集检索领域的一个显著概念进步。XTR简化了评分过程并完全消除了收集阶段，理论上提高了检索效率。然而，其当前的端到端延迟限制了它在对查询响应时间要求严格或GPU资源受限的生产环境中的应用。

## 3 Latency of Current Neural Retrievers

## 3 当前神经检索器的延迟

We start by analyzing two state-of-the-art multi-vector retrieval methods to identify their bottlenecks, providing the foundation for our work on WARP. We evaluate the latency of PLAID and XTR across various configurations and datasets: BEIR NFCorpus, LoTTE Lifestyle, and LoTTE Pooled. In XTR, token retrieval emerges as a fundamental bottleneck: the need to retrieve a large number of candidates from the ANN backend significantly impacts performance. PLAID, while generally far more efficient, faces challenges in its decompression stage. Query encoding emerges as a shared limitation for both engines, particularly on smaller datasets. These insights inform the design of WARP, which we introduce in Section 4.

我们首先分析两种最先进的多向量检索方法，以确定它们的瓶颈，为我们对WARP的研究奠定基础。我们在各种配置和数据集（BEIR NFCorpus、LoTTE Lifestyle和LoTTE Pooled）上评估PLAID和XTR的延迟。在XTR中，Token检索成为一个基本瓶颈：从近似最近邻（ANN）后端检索大量候选对象的需求显著影响了性能。虽然PLAID通常效率要高得多，但它在解压缩阶段面临挑战。查询编码是这两种引擎的共同限制，特别是在较小的数据集上。这些见解为我们在第4节介绍的WARP的设计提供了依据。

<!-- Media -->

<!-- figureText: k = 10 284ms Candidate Generation Filtering Decompression Scoring Latency (ms) $\mathrm{k} = {100}$ 322ms 507ms -->

<img src="https://cdn.noedgeai.com/0195afaa-ab63-7b7c-94ef-22219448cc3a_1.jpg?x=945&y=244&w=675&h=243&r=0"/>

Figure 2: Breakdown of ColBERTv2/PLAID's avg. latency for varying $k$ on LoTTE Pooled (Dev.)

图2：在LoTTE Pooled（开发集）上，ColBERTv2/PLAID针对不同$k$的平均延迟分解

<!-- Media -->

### 3.1 ColBERTv2/PLAID

### 3.1 ColBERTv2/PLAID

As shown in Figure 2, we evaluate PLAID's performance using its optimized implementation [4] and the ColBERTv2 checkpoint from Hugging Face [3]. We configure PLAID's hyperparameters similar to the original paper [16]. Consistent with prior work [13], we observe single threaded CPU exceeding 500ms on LoTTE Pooled. Furthermore, we find that the decompression stage remains rather constant for fixed $k$ across all datasets,consuming approximately ${150} - {200}\mathrm{\;{ms}}$ for $k = {1000}$ . Notably,for smaller datasets like NFCorpus and large $k$ values,this stage constitutes a significant portion of the overall query latency. Thus, the decompression stage emerges as a critical bottleneck for small datasets. Query encoding contributes significantly to overall latency, particularly for smaller datasets and candidate generation consitutes a fixed cost based on the number of centroids. As anticipated, the filtering stage's execution time is proportional to the number of candidates,increasing for larger $k$ values and bigger datasets. Interestingly, the scoring stage appears to have a negligible impact on ColBERTv2/PLAID's overall latency across all measurements.

如图2所示，我们使用其优化实现 [4]和来自Hugging Face的ColBERTv2检查点 [3]评估PLAID的性能。我们将PLAID的超参数配置得与原论文 [16]相似。与先前的工作 [13]一致，我们观察到在LoTTE Pooled上，单线程CPU时间超过500毫秒。此外，我们发现对于固定的$k$，解压缩阶段在所有数据集上保持相当稳定，对于$k = {1000}$消耗大约${150} - {200}\mathrm{\;{ms}}$。值得注意的是，对于像NFCorpus这样的小数据集和较大的$k$值，这个阶段在总体查询延迟中占很大比例。因此，解压缩阶段成为小数据集的关键瓶颈。查询编码对总体延迟有显著贡献，特别是对于较小的数据集，而候选生成的成本基于质心数量是固定的。正如预期的那样，过滤阶段的执行时间与候选对象的数量成正比，对于较大的$k$值和较大的数据集会增加。有趣的是，在所有测量中，评分阶段对ColBERTv2/PLAID的总体延迟影响似乎可以忽略不计。

### 3.2 XTR/ScaNN

### 3.2 XTR/ScaNN

To enable benchmarking of the XTR framework, we developed a Python library based on Google DeepMind's published code [11]. The library's code, along with scripts to reproduce the benchmarks, will be made available on GitHub. Unless otherwise specified, all benchmarks utilize the XTR BASE_EN transformer model for encoding. This model was published and is available on Hugging Face [9]. In accordance with the paper [10], we evaluate the implementation for ${\mathrm{k}}^{\prime } = {1000}$ and ${\mathrm{k}}^{\prime } = {40000}$ .

为了对XTR框架进行基准测试，我们基于谷歌DeepMind发布的代码[11]开发了一个Python库。该库的代码以及重现基准测试的脚本将在GitHub上提供。除非另有说明，所有基准测试均使用XTR BASE_EN变压器模型（XTR BASE_EN transformer model）进行编码。该模型已发布，可在Hugging Face上获取[9]。根据论文[10]，我们评估了${\mathrm{k}}^{\prime } = {1000}$和${\mathrm{k}}^{\prime } = {40000}$的实现情况。

As our measurements show, the scoring stage constitutes a significant bottleneck in the end-to-end latency of the XTR framework, particularly when dealing with large ${k}^{\prime }$ values,as seen in Figure 3. We argue that this performance bottleneck is largely attributed to an unoptimized implementation in the released code, which relies on native Python data structures and manual iteration, introducing substantial overhead, especially for large numbers of token embed-dings. We refactored this implementation to leverage optimized data structures and vectorized operations. This helps us uncover hidden performance inefficiencies and establish a baseline for further optimization, with our improved implementation of XTR to be made publicly available.

正如我们的测量结果所示，评分阶段是XTR框架端到端延迟的一个重要瓶颈，尤其是在处理较大的${k}^{\prime }$值时，如图3所示。我们认为，这一性能瓶颈在很大程度上归因于发布代码中的未优化实现，该实现依赖于原生Python数据结构和手动迭代，引入了大量开销，特别是在处理大量令牌嵌入时。我们重构了这一实现，以利用优化的数据结构和向量化操作。这有助于我们发现隐藏的性能低效问题，并为进一步优化建立基线，我们改进后的XTR实现将公开提供。

<!-- Media -->

<!-- figureText: ${k}^{\prime } = {1000}$ 1139ms Token Retrieval Scoring 6000 Latency (ms) ${k}^{\prime } = {40000}$ -->

<img src="https://cdn.noedgeai.com/0195afaa-ab63-7b7c-94ef-22219448cc3a_2.jpg?x=172&y=243&w=673&h=196&r=0"/>

Figure 3: Breakdown of the Google DeepMind's reference implementation of ${\mathrm{{XTR}}}_{\text{base }}/$ ScaNN’s avg. latency for varying $k$ on LoTTE Pooled [17]

图3：谷歌DeepMind对${\mathrm{{XTR}}}_{\text{base }}/$ ScaNN在LoTTE Pooled上不同$k$值的平均延迟参考实现的分解情况[17]

<!-- figureText: ${k}^{\prime } = {1000}$ Query Encoding 1162ms Token Retrieval 1500 1750 Latency (ms) 500 -->

<img src="https://cdn.noedgeai.com/0195afaa-ab63-7b7c-94ef-22219448cc3a_2.jpg?x=172&y=604&w=673&h=197&r=0"/>

Figure 4: Figure 4 (c): Breakdown of XTR ${}_{\text{base }}$ /ScaNN’s avg. latency for varying ${k}^{\prime }$ on LoTTE Pooled [17]

图4：图4 (c)：XTR ${}_{\text{base }}$ /ScaNN在LoTTE Pooled上不同${k}^{\prime }$值的平均延迟分解情况[17]

<!-- Media -->

We present the evaluation of our optimized implementation in Figure 4. Notably, the optimized implementation's end-to-end latency is significantly lower than that of the reference implementation ranging from an end-to-end ${3.5}\mathrm{x}$ speed-up on LoTTE pooled to a ${6.3}\mathrm{x}$ speed-up on LoTTE Lifestyle for $k = {1000}$ . This latency reduction is owed in large parts to a more efficient scoring implementation - 14x speed-up on LoTTE Pooled, see ??. In particular, this reveals token retrieval as the fundamental bottlenecks of the XTR framework. Furthermore, we notice that query encoding constitutes a large fraction of the overall end-to-end CPU latency on smaller datasets. While the optimized scoring stage consitutes a small fraction of the overall end-to-end latency, it is still slow in absolute terms - ranging from ${33}\mathrm{\;{ms}}$ to ${281}\mathrm{\;{ms}}$ for $k = {1000}$ on BEIR NFCorpus and LoTTE Pooled, respectively.

我们在图4中展示了对优化实现的评估结果。值得注意的是，优化实现的端到端延迟明显低于参考实现，从LoTTE pooled上的端到端${3.5}\mathrm{x}$加速到LoTTE Lifestyle上的${6.3}\mathrm{x}$加速（针对$k = {1000}$）。这种延迟降低在很大程度上归功于更高效的评分实现——在LoTTE Pooled上实现了14倍的加速，详见??。特别是，这揭示了令牌检索是XTR框架的根本瓶颈。此外，我们注意到，在较小的数据集上，查询编码在整个端到端CPU延迟中占很大比例。虽然优化后的评分阶段在整个端到端延迟中占比很小，但绝对速度仍然较慢——在BEIR NFCorpus和LoTTE Pooled上，针对$k = {1000}$的延迟分别从${33}\mathrm{\;{ms}}$到${281}\mathrm{\;{ms}}$不等。

## 4 WARP

## 4 WARP

WARP optimizes retrieval for the refined late interaction architecture introduced in XTR. Seeking to find the best of the XTR and PLAID worlds, WARP introduces the novel WARPseLECT algorithm for candidate generation, which effectively avoids gathering token-level representations, and proposes an optimized two-stage reduction for faster scoring via a dedicated $\mathrm{C} +  +$ kernel combined with implicit decompression. WARP also uses specialized inference runtimes for faster query encoding.

WARP针对XTR中引入的改进型后期交互架构优化了检索。为了在XTR和PLAID之间找到最佳方案，WARP引入了新颖的WARPseLECT算法用于候选生成，该算法有效避免了收集令牌级表示，并通过专用的$\mathrm{C} +  +$内核结合隐式解压缩提出了一种优化的两阶段缩减方法，以实现更快的评分。WARP还使用专门的推理运行时来实现更快的查询编码。

As in XTR, queries and documents are encoded independently into embeddings at the token-level using a fine-tuned T5 transformer [14]. To scale to large datasets, document representations are computed in advance and constitute WARP's index. The similarity between a query $q$ and document $d$ is modeled using the XTR's adaptation of ColBERT's summation of MaxSim operations Equation (1): ${}^{3}$

与XTR一样，查询和文档使用经过微调的T5变压器（T5 transformer）[14]在令牌级别独立编码为嵌入。为了扩展到大型数据集，文档表示会提前计算，并构成WARP的索引。查询$q$和文档$d$之间的相似度使用XTR对ColBERT的MaxSim操作求和的改编公式（1）进行建模：${}^{3}$

$$
{S}_{d,q} = \mathop{\sum }\limits_{{i = 1}}^{n}\mathop{\max }\limits_{{1 \leq  j \leq  m}}\left\lbrack  {{\widehat{\mathrm{A}}}_{i,j}{q}_{i}^{T}{d}_{j} + \left( {1 - {\widehat{\mathrm{A}}}_{i,j}}\right) {m}_{i}}\right\rbrack   \tag{1}
$$

<!-- Media -->

<!-- figureText: Query Query Query Matrix Decompression Token Scores Scoring Document Scores ${\mathrm{{WARP}}}_{\text{SELECT }}$ Encoding -->

<img src="https://cdn.noedgeai.com/0195afaa-ab63-7b7c-94ef-22219448cc3a_2.jpg?x=950&y=233&w=675&h=216&r=0"/>

Figure 5: WARP Retrieval consisting of query encoding, ${\text{WARP}}_{\text{SELECT }}$ ,decompression,and scoring. Notably,centroid selection is combined with the computation of missing similarity estimates in WARPseLECT.

图5：WARP检索包括查询编码、${\text{WARP}}_{\text{SELECT }}$、解压缩和评分。值得注意的是，在WARPseLECT中，质心选择与缺失相似度估计的计算相结合。

<!-- Media -->

where $q$ and $d$ are the matrix representations of the query and passage embeddings respectively, ${m}_{i}$ denotes the missing similarity estimate for ${q}_{i}$ ,and $\widehat{\mathbf{A}}$ describes XTR’s alignment matrix. In particular, ${\widehat{\mathbf{A}}}_{i,j} = {\mathbb{1}}_{\left\lbrack  j \in  \operatorname{top} - {k}_{j}^{\prime }\left( {\mathbf{d}}_{i,{j}^{\prime }}\right) \right\rbrack  }$ captures whether a document token embedding of a candidate passage was retrieved for a specific query token ${q}_{i}$ as part of the token retrieval stage. We refer to $\left\lbrack  {8,{10}}\right\rbrack$ for an intuition behind this choice of scoring function.

其中 $q$ 和 $d$ 分别是查询和段落嵌入的矩阵表示，${m}_{i}$ 表示 ${q}_{i}$ 的缺失相似度估计，$\widehat{\mathbf{A}}$ 描述了 XTR 的对齐矩阵。特别地，${\widehat{\mathbf{A}}}_{i,j} = {\mathbb{1}}_{\left\lbrack  j \in  \operatorname{top} - {k}_{j}^{\prime }\left( {\mathbf{d}}_{i,{j}^{\prime }}\right) \right\rbrack  }$ 捕获了在词元检索阶段，候选段落的文档词元嵌入是否是针对特定查询词元 ${q}_{i}$ 检索得到的。关于这种评分函数选择背后的直觉，我们参考 $\left\lbrack  {8,{10}}\right\rbrack$。

### 4.1 Index Construction

### 4.1 索引构建

Akin to ColBERTv2 [17], WARP's compression strategy involves applying $k$ -means clustering to the produced document token em-beddings. As in ColBERTv2, we find that using a sample of all passages proportional to the square root of the collection size to generate this clustering performs well in practice. After having clustered the sample of passages, all token-level vectors are encoded and stored as quantized residual vectors to their nearest cluster centroid. Each dimension of the quantized residual vector is a $b$ -bit encoding of the delta between the centroid and the original uncompressed vector. In particular, these deltas are stored as a sequence of ${128} \cdot  \frac{b}{8}$ 8-bit values,wherein 128 represents the transformers token embedding dimension. Typically,we set $b = 4$ ,i.e.,compress each dimension of the residual into a single nibble,for an ${8x}$ compression ${}^{4}$ . In this case,each compressed 8 -bit value stores 2 indices in the range $\left\lbrack  {0,{2}^{b}}\right\rbrack$ . Instead of quantizing the residuals uniformly, WARP uses quantiles derived from the empirical distribution to determine bucket boundaries (bucket_cutoffs) and the corresponding representative values (bucket_weights). This process allows WARP to allocate more quantization levels to densely populated regions of the data distribution, thereby minimizing the overall quantization error for residual compression.

与 ColBERTv2 [17] 类似，WARP 的压缩策略包括对生成的文档词元嵌入应用 $k$ -均值聚类。与 ColBERTv2 一样，我们发现使用与集合大小的平方根成比例的所有段落样本进行聚类在实践中效果很好。对段落样本进行聚类后，所有词元级向量都被编码并存储为相对于其最近聚类中心的量化残差向量。量化残差向量的每个维度是质心与原始未压缩向量之间差值的 $b$ 位编码。特别地，这些差值存储为 ${128} \cdot  \frac{b}{8}$ 个 8 位值的序列，其中 128 表示变压器词元嵌入维度。通常，我们设置 $b = 4$，即把残差的每个维度压缩成一个半字节，以实现 ${8x}$ 压缩 ${}^{4}$。在这种情况下，每个压缩的 8 位值存储范围为 $\left\lbrack  {0,{2}^{b}}\right\rbrack$ 的 2 个索引。WARP 不是对残差进行均匀量化，而是使用从经验分布中得出的分位数来确定桶边界（bucket_cutoffs）和相应的代表值（bucket_weights）。这个过程使 WARP 能够为数据分布的密集区域分配更多的量化级别，从而最小化残差压缩的总体量化误差。

### 4.2 Retrieval

### 4.2 检索

Extending PLAID, the retrieval process in the WARP engine is divided into four distinct steps: query encoding, candidate generation, decompression, and scoring. Figure 5 illustrates the retrieval process in WARP. The process starts when the query text is encoded into $q$ , a (query_maxlen, 128)-dimensional tensor, using the Transformer model. ${}^{5}$ The most similar ${n}_{\text{probe }}$ centroids are identified for each of the query_maxlen query token embeddings.

在 PLAID 的基础上进行扩展，WARP 引擎中的检索过程分为四个不同的步骤：查询编码、候选生成、解压缩和评分。图 5 展示了 WARP 中的检索过程。当查询文本使用变压器模型编码为 $q$（一个 (query_maxlen, 128) 维的张量）时，该过程开始。${}^{5}$ 为每个 query_maxlen 查询词元嵌入确定最相似的 ${n}_{\text{probe }}$ 个质心。

---

<!-- Footnote -->

${}^{3}$ Note that we omit the normalization via $\frac{1}{Z}$ ,as we are only interested in the relative ranking between documents and the normalization constant is identical for any retrieved document.

${}^{3}$ 注意，我们省略了通过 $\frac{1}{Z}$ 进行的归一化，因为我们只对文档之间的相对排名感兴趣，并且对于任何检索到的文档，归一化常数都是相同的。

${}^{4}$ As compared to an uncompressed 32-bit floating point number per dimension.

${}^{4}$ 与每个维度未压缩的 32 位浮点数相比。

<!-- Footnote -->

---

Subsequently, WARP identifies all document token embeddings belonging to the clusters of the selected centroids and computes their individual relevance score. Computing this score involves decompressing residuals of these identified document token em-beddings and calculating the cosine similarity with the relevant query token embedding. Finally, WARP (implicitly) constructs a ${n}_{\text{candidates }} \times$ query_maxlen score matrix $S,{}^{6}$ wherein each entry ${S}_{{d}_{i},{q}_{j}}$ contains the maximum retrieved score for the $i$ -th candidate passage ${d}_{i}$ and the $j$ -th query token embedding ${q}_{j}$ :

随后，WARP 识别出属于所选质心聚类的所有文档词元嵌入，并计算它们各自的相关性得分。计算这个得分包括对这些识别出的文档词元嵌入的残差进行解压缩，并计算与相关查询词元嵌入的余弦相似度。最后，WARP（隐式地）构建一个 ${n}_{\text{candidates }} \times$ query_maxlen 得分矩阵 $S,{}^{6}$，其中每个条目 ${S}_{{d}_{i},{q}_{j}}$ 包含第 $i$ 个候选段落 ${d}_{i}$ 和第 $j$ 个查询词元嵌入 ${q}_{j}$ 的最大检索得分：

$$
\mathop{\max }\limits_{{1 \leq  j \leq  m}}{\widehat{\mathbf{A}}}_{i,j}{q}_{i}^{T}{d}_{j}
$$

Matrix entries not populated during token retrieval,i.e., ${\widehat{\mathbf{A}}}_{i,j} = 0$ ,are each imputed with a missing similarity estimate, as postulated in the XTR framework. To compute the relevance score of a document ${d}_{i}$ , the cummulative score over all query tokens is computed: $\mathop{\sum }\limits_{j}{S}_{{d}_{i},{q}_{j}}$ . To produce the ordered set of passages, the set of scores sorted and the top $k$ highest scoring passages are returned.

在词元检索期间未填充的矩阵条目，即 ${\widehat{\mathbf{A}}}_{i,j} = 0$，每个都用 XTR 框架中假设的缺失相似度估计进行插补。为了计算文档 ${d}_{i}$ 的相关性得分，计算所有查询词元的累积得分：$\mathop{\sum }\limits_{j}{S}_{{d}_{i},{q}_{j}}$。为了生成有序的段落集，对得分集进行排序并返回得分最高的前 $k$ 个段落。

### 4.3 WARPselect

### 4.3 WARPselect

In contrast to ColBERT, which populates the entire score matrix for the items retrieved, XTR only populates the score matrix with scores computed as part of the token retrieval stage. To account for the contribution of any missing tokens, XTR relies on missing similarity imputation, in which they set any missing similarity of the query token for a specific document as the lowest score obtained as part of the gathering stage. The authors argue that this approach is justified as it constitutes a natural upper bound for the true relevance score. In the case of WARP,this bound is no longer guaranteed to hold. ${}^{7}$

与ColBERT（一种为检索到的项目填充整个得分矩阵的方法）不同，XTR（一种检索方法）仅用在词元检索阶段计算出的得分来填充得分矩阵。为了考虑任何缺失词元的贡献，XTR依赖于缺失相似度插补，在这种方法中，他们将特定文档中查询词元的任何缺失相似度设置为收集阶段获得的最低得分。作者认为这种方法是合理的，因为它构成了真实相关性得分的自然上限。而对于WARP（一种检索方法），这个上限不再保证成立。 ${}^{7}$

Instead, WARP defines a novel strategy for candidate generation based on cumulative cluster sizes, ${\mathrm{{WARP}}}_{\text{SELECT }}$ . Given the query embedding matrix $q$ and the list of centroids $C$ (Section 4.1),WARP computes the token-level query-centroid relevance scores. As both the query embedding vectors and the set of centroids are normalized,the cosine similarity scores ${S}_{c,q}$ can be computed efficiently as a matrix multiplication:

相反，WARP基于累积簇大小定义了一种新颖的候选生成策略， ${\mathrm{{WARP}}}_{\text{SELECT }}$ 。给定查询嵌入矩阵 $q$ 和质心列表 $C$ （第4.1节），WARP计算词元级别的查询 - 质心相关性得分。由于查询嵌入向量和质心集都进行了归一化，余弦相似度得分 ${S}_{c,q}$ 可以作为矩阵乘法高效计算：

$$
{S}_{c,q} = C \cdot  {q}^{T}
$$

Once these relevance scores have been computed WARP identifies the ${n}_{\text{probe }}$ centroids with the largest similarity scores for decompression, as part of candidate generation.

一旦计算出这些相关性得分，WARP就会识别出具有最大相似度得分的 ${n}_{\text{probe }}$ 个质心进行解压缩，作为候选生成的一部分。

Using these query-centroid similarity scores,WARPseLECT folds the estimation of missing similarity scores into candidate generation. Specifically,for each query token ${q}_{i}$ ,it sets ${m}_{i}$ from Equation (1) as the first element in the sorted list of centroid scores for which the cumulative cluster size exceeds a threshold ${t}^{\prime }$ . This method is particularly attractive as all the centroid scores have already been computed and sufficiently sorted as part of candidate generation, so the cost of computing missing similarity imputation with this method is negligible.

利用这些查询 - 质心相似度得分，WARPseLECT将缺失相似度得分的估计融入到候选生成中。具体来说，对于每个查询词元 ${q}_{i}$ ，它将公式(1)中的 ${m}_{i}$ 设置为质心得分排序列表中的第一个元素，该排序列表的累积簇大小超过阈值 ${t}^{\prime }$ 。这种方法特别有吸引力，因为所有质心得分在候选生成过程中已经计算并充分排序，因此使用这种方法计算缺失相似度插补的成本可以忽略不计。

We find that ${t}^{\prime }$ is easy to configure (Section 4.6) without compromising the retrieval quality or efficiency. Unlike XTR, the missing similarity estimate of WARP is inherently tied to the number of retrieved tokens. Intuitively,increasing ${k}^{\prime }$ may only help refine the missing similarity estimate, but not significantly increase the density of the score matrix. ${}^{8}$

我们发现 ${t}^{\prime }$ 很容易配置（第4.6节），同时不会影响检索质量或效率。与XTR不同，WARP的缺失相似度估计本质上与检索到的词元数量相关。直观地说，增加 ${k}^{\prime }$ 可能只会有助于完善缺失相似度估计，但不会显著增加得分矩阵的密度。 ${}^{8}$

### 4.4 Decompression

### 4.4 解压缩

The input for the decompression phase is the set of ${n}_{\text{probe }}$ centroid indices for each of the query_maxlen query tokens. Its goal is to calculate relevance scores between each query token and the em-beddings within the identified clusters. For a query token ${q}_{i}$ ,let ${c}_{i,j}$ , where $j \in  \left\lbrack  {n}_{\text{probe }}\right\rbrack$ ,be the set of centroid indices identified during candidate generation. Let ${r}_{i,j,k}$ be the set of residuals associated with cluster ${c}_{i,j}$ . The decompression step computes:

解压缩阶段的输入是每个查询最大长度查询词元的 ${n}_{\text{probe }}$ 个质心索引集。其目标是计算每个查询词元与已识别簇内嵌入之间的相关性得分。对于查询词元 ${q}_{i}$ ，设 ${c}_{i,j}$ （其中 $j \in  \left\lbrack  {n}_{\text{probe }}\right\rbrack$ ）为候选生成期间识别出的质心索引集。设 ${r}_{i,j,k}$ 为与簇 ${c}_{i,j}$ 相关联的残差集。解压缩步骤计算：

$$
{s}_{i,j,k} = \operatorname{decompress}\left( {C\left\lbrack  {c}_{i,j}\right\rbrack  ,{r}_{i,j,k}}\right)  \times  {q}_{i}^{T}\forall i,j,k \tag{2}
$$

The decompress function converts residuals from their compact form from ColBERTv2 and PLAID into uncompressed vectors. Each residual ${r}_{i,j,k}$ is composed of 128 indices,each $b$ bits wide. These indices reference values in the bucket weights vector $\omega  \in  {\mathbb{R}}^{{2}^{b}}$ and are used to offset the centroid $C\left\lbrack  {c}_{i,j}\right\rbrack$ . The decompress function is defined as:

解压缩函数将残差从ColBERTv2和PLAID的紧凑形式转换为未压缩的向量。每个残差 ${r}_{i,j,k}$ 由128个索引组成，每个索引 $b$ 位宽。这些索引引用桶权重向量 $\omega  \in  {\mathbb{R}}^{{2}^{b}}$ 中的值，并用于偏移质心 $C\left\lbrack  {c}_{i,j}\right\rbrack$ 。解压缩函数定义为：

$$
\operatorname{decompress}\left( {C\left\lbrack  {c}_{i,j}\right\rbrack  ,{r}_{i,j,k}}\right)  = C\left\lbrack  {c}_{i,j}\right\rbrack   + \mathop{\sum }\limits_{{d = 1}}^{{128}}{\mathbf{e}}_{d} \cdot  \omega \left\lbrack  {\left( {r}_{i,j,k}\right) }_{d}\right\rbrack   \tag{3}
$$

Here, ${\mathbf{e}}_{d}$ is the unit vector for dimension $d$ ,and $\omega \left\lbrack  {\left( {r}_{i,j,k}\right) }_{d}\right\rbrack$ is the weight value at index ${\left( {r}_{i,j,k}\right) }_{d}$ for dimension $d$ . In other words,the indices are used to look up specific entries in $\omega$ for each dimension independently, adjusting the centroid accordingly.

这里， ${\mathbf{e}}_{d}$ 是维度 $d$ 的单位向量， $\omega \left\lbrack  {\left( {r}_{i,j,k}\right) }_{d}\right\rbrack$ 是维度 $d$ 中索引 ${\left( {r}_{i,j,k}\right) }_{d}$ 处的权重值。换句话说，这些索引用于独立地查找 $\omega$ 中每个维度的特定条目，并相应地调整质心。

Instead of explicitly decompressing residuals as in PLAID, WARP leverages the observation that the scoring function decomposes between centroids and residuals. As a result, WARP reuses the query-centroid relevance scores ${S}_{c,q}$ ,computed as part of candidate generation. That is, observe that:

与PLAID中显式解压缩残差不同，WARP利用了评分函数在质心和残差之间可分解的观察结果。因此，WARP重用了作为候选生成一部分计算出的查询 - 质心相关性得分 ${S}_{c,q}$ 。也就是说，观察到：

$$
{s}_{i,j,k} = \operatorname{decompress}\left( {C\left\lbrack  {c}_{i,j}\right\rbrack  ,{r}_{i,j,k}}\right)  \times  {q}_{i}^{T}
$$

$$
 = \left( {C\left\lbrack  {c}_{i,j}\right\rbrack   \times  {q}_{i}^{T}}\right)  + \left( {\mathop{\sum }\limits_{{d = 1}}^{{128}}\omega \left\lbrack  {\left( {r}_{i,j,m,k}\right) }_{d}\right\rbrack  {q}_{i,d}}\right)  \tag{4}
$$

To accelerate decompression,WARP computes $v = \widehat{q} \times  \widehat{\omega }$ ,wherein $\widehat{q} \in  {\mathbb{R}}^{\text{query_maxlen} \times  {128} \times  1}$ represents the query matrix that has been unsqueezed along the last dimension,and $\widehat{\omega } \in  {\mathbb{R}}^{1 \times  {2}^{b}}$ denotes the vector of bucket weights that has been unsqueezed along the first dimension. With these definitions, WARP can decompress and score candidate tokens via:

为了加速解压缩，WARP计算$v = \widehat{q} \times  \widehat{\omega }$，其中$\widehat{q} \in  {\mathbb{R}}^{\text{query_maxlen} \times  {128} \times  1}$表示已沿最后一个维度进行扩展的查询矩阵，$\widehat{\omega } \in  {\mathbb{R}}^{1 \times  {2}^{b}}$表示已沿第一个维度进行扩展的桶权重向量。根据这些定义，WARP可以通过以下方式对候选Token进行解压缩和评分：

$$
{s}_{i,j,k} = {S}_{{c}_{j},{q}_{i}} + \mathop{\sum }\limits_{{d = 1}}^{{128}}\left( {\omega  \cdot  {q}_{i,d}}\right) \left\lbrack  {\left( {r}_{i,{jk}}\right) }_{d}\right\rbrack   \tag{5}
$$

$$
 = {S}_{{c}_{j},{q}_{i}} + \mathop{\sum }\limits_{{i = 1}}^{{128}}{v}_{i,d}\left\lbrack  {\left( {r}_{i,j,k}\right) }_{d}\right\rbrack  
$$

---

<!-- Footnote -->

${}^{5}$ We set query_maxlen $= {32}$ in accordance with the XTR paper.

${}^{5}$ 我们根据XTR论文将查询最大长度（query_maxlen）设置为$= {32}$。

${}^{6}$ Note that our optimized implementation does not physically construct this matrix.

${}^{6}$ 请注意，我们的优化实现并未实际构建此矩阵。

${}^{6}$ For XTR baselines,’candidate generation’ refers to the token retrieval stage.

${}^{6}$ 对于XTR基线，“候选生成”指的是Token检索阶段。

${}^{7}$ Strictly speaking,the bound is also approximate in XTR’s case,as ScaNN does not return the exact nearest neighbors in general.

${}^{7}$ 严格来说，在XTR的情况下，该边界也是近似的，因为ScaNN通常不会返回精确的最近邻。

${}^{8}$ This is because tokens retrieved with a larger ${k}^{\prime }$ are often from new documents and, therefore, do not refine the scores of already retrieved ones.

${}^{8}$ 这是因为使用较大的${k}^{\prime }$检索到的Token通常来自新文档，因此不会优化已检索到的Token的分数。

<!-- Footnote -->

---

Note that candidate scoring can now be implemented as a simple selective sum. As the bucket weights are shared among centroids and the query-centroid relevance scores have already been computed during candidate generation, WARP can decompress and score arbitrarily many clusters using $O\left( 1\right)$ multiplications. This refined scoring function is far more efficient then the one outlined in PLAID, ${}^{9}$ as it never computes the decompressed embeddings explicitly and instead directly emits the resulting candidate scores. We provide an efficient implementation of the selective sum of Equation (4) and realize unpacking of the residual representation using low-complexity bitwise operations as part of WARP's $\mathrm{C} +  +$ kernel for decompression.

请注意，现在可以将候选评分实现为简单的选择性求和。由于质心之间共享桶权重，并且在候选生成期间已经计算了查询 - 质心相关性分数，因此WARP可以使用$O\left( 1\right)$次乘法对任意数量的聚类进行解压缩和评分。这种优化后的评分函数比PLAID中概述的函数效率高得多，${}^{9}$因为它从不显式计算解压缩后的嵌入，而是直接输出最终的候选分数。我们提供了方程（4）选择性求和的高效实现，并作为WARP解压缩$\mathrm{C} +  +$内核的一部分，使用低复杂度的按位运算实现残差表示的解包。

### 4.5 Scoring

### 4.5 评分

At the end of the decompression phase,we have query_maxlen $\times$ ${n}_{\text{probe }}$ strides of decompressed candidate document token-level scores and their corresponding document identifiers. Scoring combines these scores with the missing similarity estimates, computed during candidate generation, to produce document-level scores. This process corresponds to constructing the score matrix and taking the row-wise sum.

在解压缩阶段结束时，我们有查询最大长度（query_maxlen）为$\times$ ${n}_{\text{probe }}$的解压缩候选文档Token级分数的步长及其对应的文档标识符。评分将这些分数与候选生成期间计算的缺失相似度估计相结合，以生成文档级分数。此过程对应于构建分数矩阵并进行逐行求和。

Explicitly constructing the score matrix, as in the the reference XTR implementation, introduces a significant bottleneck, particularly for large values of ${n}_{\text{probe }}$ . To address this,WARP efficiently aggregates token-level scores using a two-stage reduction process:

如参考XTR实现中那样显式构建分数矩阵会引入显著的瓶颈，特别是对于较大的${n}_{\text{probe }}$值。为了解决这个问题，WARP使用两阶段归约过程高效地聚合Token级分数：

- Token-level reduction For each query token, reduce the corresponding set of ${n}_{\text{probe }}$ strides using the max operator. This step implicitly fills the score matrix with the maximum per-token score for each document. As a single cluster can contain multiple document token embeddings originating from the same document, WARP performs inner-cluster max-reduction directly during the decompression phase.

- Token级归约 对于每个查询Token，使用最大值运算符对相应的${n}_{\text{probe }}$个步长集合进行归约。此步骤隐式地用每个文档的每个Token的最大分数填充分数矩阵。由于单个聚类可以包含来自同一文档的多个文档Token嵌入，因此WARP在解压缩阶段直接执行聚类内的最大归约。

- Document-level reduction Reduce the resulting strides into document-level scores using a sum aggregation. It is essential to handle missing values properly at this stage - any missing per-token score must be replaced by the corresponding missing similarity estimate, ensuring compliance with the XTR scoring function described in Equation (1). This reduction step corresponds to the row-wise summation of the score matrix.

- 文档级归约 使用求和聚合将得到的步长归约为文档级分数。在此阶段正确处理缺失值至关重要 - 任何缺失的每个Token的分数必须用相应的缺失相似度估计值替换，以确保符合方程（1）中描述的XTR评分函数。此归约步骤对应于分数矩阵的逐行求和。

After performing both reduction phases, the final stride contains the document-level scores and the corresponding identifiers for all candidate documents. To retrieve the result set, we perform heap select to obtain the top- $k$ documents,similar to its use in the candidate generation phase. Formally,we consider a stride $S$ to be an list of key-value pairs:

在执行完两个归约阶段后，最终的步长包含所有候选文档的文档级分数和相应的标识符。为了检索结果集，我们执行堆选择以获取前$k$个文档，这与候选生成阶段的使用类似。形式上，我们将步长$S$视为键值对列表：

$$
S = \left\{  \left( {{k}_{i},{v}_{i}}\right) \right\}  ;\mathrm{K}\left( S\right)  = \left\{  {{k}_{i} \mid  \left( {{k}_{i},{v}_{i}}\right)  \in  S}\right\}  ;\mathrm{V}\left( S\right)  = \left\{  {{v}_{i} \mid  \left( {{k}_{i},{v}_{i}}\right)  \in  S}\right\}  
$$

Thus,strides implicitly define a partial function ${f}_{S} : K \rightharpoonup  V\left( S\right)$ :

因此，步长隐式地定义了一个部分函数${f}_{S} : K \rightharpoonup  V\left( S\right)$：

$$
{f}_{S}\left( k\right)  = \left\{  \begin{array}{ll} {v}_{i} & \text{ if }\exists {v}_{i}.\left( {k,{v}_{i}}\right)  \in  S \\   \bot  & \text{ otherwise } \end{array}\right. 
$$

We define a reduction as a combination of two strides ${S}_{1}$ and ${S}_{2}$ using a binary function $r$ into a single stride by applying $r$ to values of matching keys:

我们将归约定义为使用二元函数$r$将两个步长${S}_{1}$和${S}_{2}$组合成单个步长，方法是将$r$应用于匹配键的值：

$$
\operatorname{reduce}\left( {r,{S}_{1},{S}_{2}}\right)  = \left\{  {\left( {k,r\left( {{f}_{{S}_{1}}\left( k\right) ,{f}_{{S}_{2}}\left( k\right) }\right) }\right)  \mid  k \in  K\left( {S}_{1}\right)  \cup  K\left( {S}_{2}\right) }\right\}  
$$

With these definitions, token-level reduction can be written as:

根据这些定义，Token级归约可以写成：

$$
{r}_{\text{tok }}\left( {{v}_{1},{v}_{2}}\right)  = \left\{  \begin{array}{ll} \max \left( {{v}_{1},{v}_{2}}\right) & \text{ if }{v}_{1} \neq   \bot   \land  {v}_{2} \neq   \bot  \\  {v}_{1} & \text{ if }{v}_{1} \neq   \bot   \land  {v}_{2} =  \bot  \\  {v}_{2} & \text{ otherwise } \end{array}\right.  \tag{6}
$$

Defining the document-level reduction is slightly more complex as it involves incorporating the corresponding missing similarity estimates $m$ . After token-level reduction each of the query_maxlen strides ${S}_{1},\ldots ,{S}_{\text{query_maxlen }}$ covers scores for a single query token ${q}_{i}$ . We set ${S}_{i,i} = {S}_{i}$ and define:

定义文档级归约（document-level reduction）稍微复杂一些，因为它涉及纳入相应的缺失相似度估计值 $m$。在词元级归约（token-level reduction）之后，每个查询最大长度步长 ${S}_{1},\ldots ,{S}_{\text{query_maxlen }}$ 涵盖单个查询词元的得分 ${q}_{i}$。我们设置 ${S}_{i,i} = {S}_{i}$ 并定义：

$$
{S}_{i,j} = \operatorname{reduce}\left( {{r}_{\mathrm{{doc}},\left( {i,k,j}\right) },{S}_{i,k},{S}_{k + 1,j}}\right)  \tag{7}
$$

for any choice of $i \leq  k < j$ ,wherein ${r}_{\text{doc },\left( {i,k,j}\right) }$ merges two successive,non-overlapping strides ${S}_{i,k}$ and ${S}_{k + 1,j}$ . The resulting stride, ${S}_{i,j}$ ,now covers scores for query tokens ${q}_{i},\ldots ,{q}_{j}$ . Defining ${r}_{\text{doc,}\left( {i,k,j}\right) }$ is relatively straightforward:

对于 $i \leq  k < j$ 的任何选择，其中 ${r}_{\text{doc },\left( {i,k,j}\right) }$ 合并两个连续的、不重叠的步长 ${S}_{i,k}$ 和 ${S}_{k + 1,j}$。得到的步长 ${S}_{i,j}$ 现在涵盖查询词元 ${q}_{i},\ldots ,{q}_{j}$ 的得分。定义 ${r}_{\text{doc,}\left( {i,k,j}\right) }$ 相对简单：

$$
{r}_{\text{doc },\left( {i,k,j}\right) }\left( {{v}_{1},{v}_{2}}\right)  = \left\{  \begin{array}{ll} {v}_{1} + {v}_{2} & \text{ if }{v}_{1} \neq   \bot   \land  {v}_{2} \neq   \bot  \\  {v}_{1} + \left( {\mathop{\sum }\limits_{{t = k + 1}}^{j}{m}_{t}}\right) & \text{ if }{v}_{1} \neq   \bot   \land  {v}_{2} =  \bot  \\  \left( {\mathop{\sum }\limits_{{t = i}}^{k}{m}_{t}}\right)  + {v}_{2} & \text{ otherwise } \end{array}\right.  \tag{8}
$$

It is easy to verify that ${S}_{i,j}$ is well-defined,i.e.,independent of the choice of $k$ . The result of document-level reduction is ${S}_{1,\text{ query_maxlen }}$ and can be obtained by recursively applying Equation (7) to strides of increasing size.

很容易验证 ${S}_{i,j}$ 是明确定义的，即与 $k$ 的选择无关。文档级归约的结果是 ${S}_{1,\text{ query_maxlen }}$，可以通过对不断增大的步长递归应用公式 (7) 来获得。

WARP's two-stage reduction process, along with the final sorting step, is illustrated in Figure 6. In the token-level reduction stage, strides are merged by selecting the maximum value for matching keys. In the document-level reduction stage, values for matching keys are summed, with missing values being substituted by the corresponding missing similarity estimates.

WARP 的两阶段归约过程以及最后的排序步骤如图 6 所示。在词元级归约阶段，通过为匹配键选择最大值来合并步长。在文档级归约阶段，对匹配键的值求和，缺失值由相应的缺失相似度估计值替代。

In our implementation, we conceptually construct a binary tree of the required merges and alternate between two scratch buffers to avoid additional memory allocations. We realize Equation (8) using a prefix sum, which eliminates the need to compute the sum explicitly.

在我们的实现中，我们从概念上构建所需合并操作的二叉树，并在两个临时缓冲区之间交替使用，以避免额外的内存分配。我们使用前缀和来实现公式 (8)，这样就无需显式计算总和。

### 4.6 Hyperparameters

### 4.6 超参数

In this section, we aim to analyze the effects of WARP's three primary hyperparameters, namely:

在本节中，我们旨在分析 WARP 的三个主要超参数的影响，即：

- ${n}_{\text{probe }}$ - the #clusters to decompress per query token

- ${n}_{\text{probe }}$ - 每个查询词元要解压缩的聚类数量

- ${t}^{\prime }$ - the threshold on the cluster size used for WARPSELECT

- ${t}^{\prime }$ - 用于WARPSELECT的簇大小阈值

- $b$ - the number of bits per dimension of a residual vector

- $b$ - 残差向量每个维度的比特数

To study the effects of ${n}_{\text{probe }}$ and ${t}^{\prime }$ ,we analyze the normalized Recall $@{100}^{10}$ as a function of ${t}^{\prime }$ for ${n}_{\text{probe }} \in  \{ 1,2,4,8,{16},{32},{64}\}$ across four development datasets of increasing size: BEIR NFCor-pus, BEIR Quora, LoTTE Lifestyle, and LoTTE Pooled. For further details on the datasets, please refer to Table 1. Figure 7 visualizes the results of our analysis. We observe a consistent pattern across all evaluated datasets,namely substantial improvements as ${n}_{\text{probe }}$ increases from 1 to 16 (i.e., 1, 2, 4, 8, 16), followed by only marginal gains in Recall@100 beyond that. A notable exception is BEIR NFCorpus, where we still observe significant improvement when increasing from ${n}_{\text{probe }} = {16}$ to ${n}_{\text{probe }} = {32}$ . We hypothesize that this is due to the small number of embeddings per cluster in NF-Corpus, limiting the number of scores available for aggregation. Consequently,we conclude that setting ${n}_{\text{probe }} = {32}$ strikes a good balance between end-to-end latency and retrieval quality.

为了研究${n}_{\text{probe }}$和${t}^{\prime }$的影响，我们分析了归一化召回率$@{100}^{10}$随${t}^{\prime }$的变化情况，针对${n}_{\text{probe }} \in  \{ 1,2,4,8,{16},{32},{64}\}$在四个规模逐渐增大的开发数据集上进行分析：BEIR NFCorpus、BEIR Quora、LoTTE Lifestyle和LoTTE Pooled。有关数据集的更多详细信息，请参考表1。图7展示了我们的分析结果。我们在所有评估数据集中观察到了一致的模式，即当${n}_{\text{probe }}$从1增加到16（即1、2、4、8、16）时，召回率有显著提高，之后Recall@100的提升幅度很小。一个显著的例外是BEIR NFCorpus，当从${n}_{\text{probe }} = {16}$增加到${n}_{\text{probe }} = {32}$时，我们仍然观察到显著的改善。我们假设这是由于NFCorpus中每个簇的嵌入数量较少，限制了可用于聚合的分数数量。因此，我们得出结论，设置${n}_{\text{probe }} = {32}$可以在端到端延迟和检索质量之间取得良好的平衡。

---

<!-- Footnote -->

${}^{9}$ PLAID cannot adopt this approach directly,as it normalizes the vectors after decompression. Empirically, we find that that this normalization step has limited effect on the final embeddings as the residuals are already normalized prior to quantization.

${}^{9}$ PLAID不能直接采用这种方法，因为它在解压缩后对向量进行归一化。根据经验，我们发现这个归一化步骤对最终嵌入的影响有限，因为残差在量化之前已经进行了归一化。

${}^{10}$ The normalized Recall@k is calculated by dividing Recall@k by the dataset’s maximum, effectively scaling values between 0 and 1 to ensure comparability across datasets.

${}^{10}$ 归一化的Recall@k是通过将Recall@k除以数据集的最大值来计算的，有效地将值缩放到0到1之间，以确保跨数据集的可比性。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: (c) Sorting 36 107 636 164 294 346 486 ... 1.75 1.43 1.35 1.26 1.25 1.23 1.11 1.04 164 294 900 0.76 0.61 0.53 0.47 0.56 Missing Similarity Estimates (b) Document Level Reduction 294 ... (a) Token Level Reduction 294 492 195 0.78 0.87 0.54 0.80 0.72 0.88 0.37 0.70 0.85 -->

<img src="https://cdn.noedgeai.com/0195afaa-ab63-7b7c-94ef-22219448cc3a_5.jpg?x=180&y=236&w=668&h=699&r=0"/>

Figure 6: WARP's scoring phase: (a) In token-level reduction, strides are max-reduced. (b) In document-level reduction, values are summed, accounting for missing similarity estimates. (c) Scores are sorted,yielding the top- $k$ results.

图6：WARP的评分阶段：(a) 在词元级归约中，步长进行最大归约。(b) 在文档级归约中，对值进行求和，并考虑缺失的相似度估计。(c) 对分数进行排序，得到前$k$个结果。

<!-- Media -->

In general, we find that WARP is highly robust to variations in ${t}^{\prime }$ . However,smaller datasets,such as NFCorpus,appear to benefit from a smaller ${t}^{\prime }$ ,while larger datasets perform better with a larger ${t}^{\prime }$ . Empirically,we find that setting ${t}^{\prime }$ proportional to the square root of the dataset size consistently yields strong results across all datasets. Moreover,increasing ${t}^{\prime }$ beyond a certain point no longer improves Recall,leading us to bound ${t}^{\prime }$ by a maximum value, ${t}_{\max }^{\prime }$ .

一般来说，我们发现WARP对${t}^{\prime }$的变化具有很高的鲁棒性。然而，较小的数据集，如NFCorpus，似乎受益于较小的${t}^{\prime }$，而较大的数据集在较大的${t}^{\prime }$下表现更好。根据经验，我们发现将${t}^{\prime }$设置为与数据集大小的平方根成正比，在所有数据集上都能持续取得较好的结果。此外，将${t}^{\prime }$增加到一定程度后，召回率不再提高，因此我们将${t}^{\prime }$的最大值限制为${t}_{\max }^{\prime }$。

Next,we aim to quantify the effect of $b$ on the retrieval quality of WARP, as shown in Figure 8. To do this, we compute the nRecall@k for ${n}_{\text{probe }} = {32}$ and $k \in  \{ {10},{100}\}$ using two datasets: LoTTE Science and LoTTE Pooled. Our results show a significant improvement in retrieval performance when increasing $b$ from 2 to 4,particularly for smaller values of $k$ . For larger values of $k$ ,the difference in performance diminishes, particularly for the LoTTE Pooled dataset.

接下来，我们旨在量化$b$对WARP检索质量的影响，如图8所示。为此，我们使用两个数据集LoTTE Science和LoTTE Pooled计算了${n}_{\text{probe }} = {32}$和$k \in  \{ {10},{100}\}$的nRecall@k。我们的结果表明，当$b$从2增加到4时，检索性能有显著提高，特别是对于较小的$k$值。对于较大的$k$值，性能差异减小，特别是在LoTTE Pooled数据集上。

## 5 Evaluation

## 5 评估

We now evaluate WARP on six datasets from BEIR [18] and six datasets from LoTTE [17] listed in Table 1. We use servers with 28 Intel Xeon Gold 6132 @ 2.6 GHz CPU cores ${}^{11}$ and ${500}\mathrm{{GB}}$ RAM. The servers have two NUMA sockets with roughly ${92}\mathrm{{ns}}$ intra-socket memory latency, 142 ns inter-socket memory latency, 72 GBps intra-socket memory bandwidth, and 33 GBps inter-socket memory bandwidth.

我们现在在表1中列出的来自BEIR [18]的六个数据集和来自LoTTE [17]的六个数据集上评估WARP。我们使用配备28个英特尔至强金牌6132 @ 2.6 GHz CPU核心${}^{11}$和${500}\mathrm{{GB}}$内存的服务器。这些服务器有两个NUMA插槽，插槽内内存延迟约为${92}\mathrm{{ns}}$，插槽间内存延迟为142 ns，插槽内内存带宽为72 GBps，插槽间内存带宽为33 GBps。

<!-- Media -->

<!-- figureText: 0.90 1.00 0.90 0.80 ${n}_{\text{probe }} = 1$ ${n}_{\text{probe }} = 2$ 0.75 ${n}_{\text{probe }} = 4$ ${n}_{\text{probe }} = 8$ 0.65 Iprobe $= {32}$ ${\mathrm{n}}_{\text{probe }} = {64}$ 10000 (b) LoTTE Pooled (Dev Set) 0.80 ${\mathrm{n}}_{\text{probe }} = 1$ ${n}_{\text{probe }} = 2$ 0.75 ${n}_{\text{probe }} = 4$ ${n}_{\text{probe }} = 8$ 0.65 ${\mathrm{n}}_{\text{probe }} = {32}$ ${\mathrm{n}}_{\text{probe }} = {64}$ 100000 (a) LoTTE Science (Dev Set) -->

<img src="https://cdn.noedgeai.com/0195afaa-ab63-7b7c-94ef-22219448cc3a_5.jpg?x=927&y=266&w=690&h=314&r=0"/>

Figure 7: nRecall @100 as a function of ${t}^{\prime }$ and ${n}_{\text{probe }}$

图7：nRecall @100随${t}^{\prime }$和${n}_{\text{probe }}$的变化情况

<!-- figureText: S 0.95 2,0.95 0.90 0.75 0.70 ${n}_{\text{probe }} = {32},b = 4$ 2000 40000 60000 (b) LoTTE Pooled (Dev Set) 0.90 ${n}_{\text{probe }} = {32},b = 4$ (a) LoTTE Science (Dev Set) -->

<img src="https://cdn.noedgeai.com/0195afaa-ab63-7b7c-94ef-22219448cc3a_5.jpg?x=928&y=700&w=690&h=313&r=0"/>

Figure 8: nRecall @100 as a function of ${t}^{\prime }$ and $b$

图8：nRecall @100随${t}^{\prime }$和$b$的变化情况

<table><tr><td colspan="2" rowspan="2">Dataset</td><td colspan="2">Dev</td><td colspan="2">Test</td></tr><tr><td>#Queries</td><td>#Passages</td><td>#Queries</td><td>#Passages</td></tr><tr><td rowspan="6">BelR [18]</td><td>NFCORPUS</td><td>324</td><td>3.6K</td><td>323</td><td>3.6K</td></tr><tr><td>SciFact</td><td>-</td><td>-</td><td>300</td><td>5.2K</td></tr><tr><td>SCIDOCS</td><td>-</td><td>-</td><td>1,000</td><td>25.7K</td></tr><tr><td>Quora</td><td>5,000</td><td>${522.9}\mathrm{\;K}$</td><td>10,000</td><td>522.9K</td></tr><tr><td>FiQA-2018</td><td>500</td><td>57.6K</td><td>648</td><td>57.6K</td></tr><tr><td>Touché-2020</td><td>-</td><td>-</td><td>49</td><td>${382.5}\mathrm{\;K}$</td></tr><tr><td rowspan="7">LoTTE [17]</td><td>Lifestyle</td><td>417</td><td>${268.9}\mathrm{\;K}$</td><td>661</td><td>119.5K</td></tr><tr><td>Recreation</td><td>563</td><td>${263.0}\mathrm{\;K}$</td><td>924</td><td>167.0K</td></tr><tr><td>Writing</td><td>497</td><td>277.1K</td><td>1,071</td><td>200.0K</td></tr><tr><td>Technology</td><td>916</td><td>1.3M</td><td>596</td><td>638.5K</td></tr><tr><td>Science</td><td>538</td><td>343.6K</td><td>617</td><td>1.7M</td></tr><tr><td>Pooled</td><td>2,931</td><td>2.4M</td><td>3,869</td><td>2.8M</td></tr><tr><td/><td/><td/><td/><td/></tr></table>

<table><tbody><tr><td colspan="2" rowspan="2">数据集</td><td colspan="2">开发集</td><td colspan="2">测试集</td></tr><tr><td>查询数量</td><td>段落数量</td><td>查询数量</td><td>段落数量</td></tr><tr><td rowspan="6">BelR [18]</td><td>NFCORPUS（未找到通用中文译名，保留原文）</td><td>324</td><td>3.6K</td><td>323</td><td>3.6K</td></tr><tr><td>科学事实数据集（SciFact）</td><td>-</td><td>-</td><td>300</td><td>5.2K</td></tr><tr><td>科学文档数据集（SCIDOCS）</td><td>-</td><td>-</td><td>1,000</td><td>25.7K</td></tr><tr><td>Quora问答平台（Quora）</td><td>5,000</td><td>${522.9}\mathrm{\;K}$</td><td>10,000</td><td>522.9K</td></tr><tr><td>FiQA - 2018金融问答数据集（FiQA - 2018）</td><td>500</td><td>57.6K</td><td>648</td><td>57.6K</td></tr><tr><td>Touché - 2020评测任务（Touché - 2020）</td><td>-</td><td>-</td><td>49</td><td>${382.5}\mathrm{\;K}$</td></tr><tr><td rowspan="7">LoTTE [17]</td><td>生活方式</td><td>417</td><td>${268.9}\mathrm{\;K}$</td><td>661</td><td>119.5K</td></tr><tr><td>娱乐休闲</td><td>563</td><td>${263.0}\mathrm{\;K}$</td><td>924</td><td>167.0K</td></tr><tr><td>写作</td><td>497</td><td>277.1K</td><td>1,071</td><td>200.0K</td></tr><tr><td>技术</td><td>916</td><td>1.3M</td><td>596</td><td>638.5K</td></tr><tr><td>科学</td><td>538</td><td>343.6K</td><td>617</td><td>1.7M</td></tr><tr><td>合并的</td><td>2,931</td><td>2.4M</td><td>3,869</td><td>2.8M</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></tbody></table>

Table 1: Datasets used for evaluating ${\mathrm{{XTR}}}_{\text{base }}$ /WARP performance. The evaluation includes 6 datasets from BEIR [18] and 6 from LoTTE [17].

表1：用于评估${\mathrm{{XTR}}}_{\text{base }}$ /WARP性能的数据集。评估包括来自BEIR [18]的6个数据集和来自LoTTE [17]的6个数据集。

<!-- Media -->

When measuring latency for end-to-end results, we compute the average latency of all queries Table 1 and report the minimum average latency across three trials. For other results, we describe the specific measurement procedure in the relevant section. We measure latency on an otherwise idle machine. As XTR's token WARP: An Efficient Engine for Multi-Vector Retrieval retrieval stage does not benefit from GPU acceleration due to it's use of ScaNN [5],specifically designed for single-threaded ${}^{12}$ use on x86 processors with AVX2 support. Therefore, we perform CPU-only measurements and restrict the usage to a single thread unless otherwise stated.

在测量端到端结果的延迟时，我们计算表1中所有查询的平均延迟，并报告三次试验中的最小平均延迟。对于其他结果，我们将在相关部分描述具体的测量过程。我们在一台闲置的机器上测量延迟。由于XTR的令牌WARP：一种用于多向量检索的高效引擎检索阶段由于使用了ScaNN [5]而无法从GPU加速中受益，ScaNN [5]是专门为支持AVX2的x86处理器的单线程${}^{12}$使用而设计的。因此，除非另有说明，否则我们仅进行CPU测量，并将使用限制在单线程上。

---

<!-- Footnote -->

${}^{11}$ Each core has 2 threads for a total of 56 threads.

${}^{11}$ 每个核心有2个线程，总共56个线程。

<!-- Footnote -->

---

<!-- Media -->

<table><tr><td/><td>Lifestyle</td><td>Recreation</td><td>Writing</td><td>Technology</td><td>Science</td><td>Pooled</td><td>Avg.</td></tr><tr><td>BM25</td><td>63.8</td><td>56.5</td><td>60.3</td><td>41.8</td><td>32.7</td><td>48.3</td><td>50.6</td></tr><tr><td>ColBERT</td><td>80.2</td><td>68.5</td><td>74.7</td><td>61.9</td><td>53.6</td><td>67.3</td><td>67.7</td></tr><tr><td>${\mathrm{{GTR}}}_{\text{base }}$</td><td>82.0</td><td>65.7</td><td>74.1</td><td>58.1</td><td>49.8</td><td>65.0</td><td>65.8</td></tr><tr><td>XTR/ScaNN</td><td>83.5 (333.6)</td><td>69.6 (400.2)</td><td>78.0 (378.0)</td><td>63.9 (742.5)</td><td>55.3 (1827.6)</td><td>68.4 (2156.3)</td><td>69.8</td></tr><tr><td>WARP</td><td>83.5 (73.1)</td><td>69.5 (72.4)</td><td>78.6 (73.6)</td><td>64.6 (96.4)</td><td>56.1 (156.4)</td><td>69.3 (171.3)</td><td>70.3</td></tr><tr><td>${\text{Splade}}_{\mathrm{v}2} * \diamond$</td><td>82.3</td><td>69.0</td><td>77.1</td><td>62.4</td><td>55.4</td><td>68.9</td><td>69.2</td></tr><tr><td>${\mathrm{{ColBERT}}}_{\mathrm{v}2}{}^{\pm \diamond }$</td><td>84.7</td><td>72.3</td><td>80.1</td><td>66.1</td><td>56.7</td><td>71.6</td><td>71.9</td></tr><tr><td>${\mathrm{{GTR}}}_{\mathrm{{xxl}}}$</td><td>87.4</td><td>78.0</td><td>83.9</td><td>69.5</td><td>60.0</td><td>76.0</td><td>75.8</td></tr><tr><td>${\mathrm{{XTR}}}_{\mathrm{{xxl}}}$</td><td>89.1</td><td>79.3</td><td>83.3</td><td>73.7</td><td>60.8</td><td>77.3</td><td>77.3</td></tr></table>

<table><tbody><tr><td></td><td>生活方式</td><td>娱乐休闲</td><td>写作</td><td>技术（Technology）</td><td>科学（Science）</td><td>合并的；汇集的</td><td>平均值（Avg.）</td></tr><tr><td>BM25算法</td><td>63.8</td><td>56.5</td><td>60.3</td><td>41.8</td><td>32.7</td><td>48.3</td><td>50.6</td></tr><tr><td>ColBERT模型</td><td>80.2</td><td>68.5</td><td>74.7</td><td>61.9</td><td>53.6</td><td>67.3</td><td>67.7</td></tr><tr><td>${\mathrm{{GTR}}}_{\text{base }}$</td><td>82.0</td><td>65.7</td><td>74.1</td><td>58.1</td><td>49.8</td><td>65.0</td><td>65.8</td></tr><tr><td>XTR/ScaNN算法</td><td>83.5 (333.6)</td><td>69.6 (400.2)</td><td>78.0 (378.0)</td><td>63.9 (742.5)</td><td>55.3 (1827.6)</td><td>68.4 (2156.3)</td><td>69.8</td></tr><tr><td>WARP损失函数</td><td>83.5 (73.1)</td><td>69.5 (72.4)</td><td>78.6 (73.6)</td><td>64.6 (96.4)</td><td>56.1 (156.4)</td><td>69.3 (171.3)</td><td>70.3</td></tr><tr><td>${\text{Splade}}_{\mathrm{v}2} * \diamond$</td><td>82.3</td><td>69.0</td><td>77.1</td><td>62.4</td><td>55.4</td><td>68.9</td><td>69.2</td></tr><tr><td>${\mathrm{{ColBERT}}}_{\mathrm{v}2}{}^{\pm \diamond }$</td><td>84.7</td><td>72.3</td><td>80.1</td><td>66.1</td><td>56.7</td><td>71.6</td><td>71.9</td></tr><tr><td>${\mathrm{{GTR}}}_{\mathrm{{xxl}}}$</td><td>87.4</td><td>78.0</td><td>83.9</td><td>69.5</td><td>60.0</td><td>76.0</td><td>75.8</td></tr><tr><td>${\mathrm{{XTR}}}_{\mathrm{{xxl}}}$</td><td>89.1</td><td>79.3</td><td>83.3</td><td>73.7</td><td>60.8</td><td>77.3</td><td>77.3</td></tr></tbody></table>

-: cross-encoder distillation $\diamond$ : model-based hard negatives

-：交叉编码器蒸馏 $\diamond$ ：基于模型的硬负样本

Table 2: Success@5 on LoTTE. Numbers in parentheses show average latency (milliseconds), with the final column displaying the average score across the datasets. Both XTR/ScaNN and WARP use the model XTR_base. XTR/ScaNN uses ${k}^{\prime } = {40000}$ and WARP uses ${n}_{\text{nprobe }} = {32}$ .

表2：LoTTE数据集上的前5准确率（Success@5）。括号内的数字表示平均延迟（毫秒），最后一列显示各数据集的平均得分。XTR/ScaNN和WARP均使用XTR_base模型。XTR/ScaNN使用 ${k}^{\prime } = {40000}$ ，WARP使用 ${n}_{\text{nprobe }} = {32}$ 。

<table><tr><td/><td>NFCorpus</td><td>SciFact</td><td>SCIDOCS</td><td>FiQA-2018</td><td>Touché-2020</td><td>Quora</td><td>Avg.</td></tr><tr><td>BM25</td><td>25.0</td><td>90.8</td><td>35.6</td><td>53.9</td><td>53.8</td><td>97.3</td><td>59.4</td></tr><tr><td>ColBERT</td><td>25.4</td><td>87.8</td><td>34.4</td><td>60.3</td><td>43.9</td><td>98.9</td><td>58.5</td></tr><tr><td>${\mathrm{{GTR}}}_{\text{base }}$</td><td>27.5</td><td>87.2</td><td>34.0</td><td>67.0</td><td>44.3</td><td>99.6</td><td>59.9</td></tr><tr><td>T5-ColBERT ${}_{\text{base}}$</td><td>27.6</td><td>91.3</td><td>34.2</td><td>63.0</td><td>49.9</td><td>97.9</td><td>60.7</td></tr><tr><td>XTR/ScaNN</td><td>28.0 (158.1)</td><td>90.9 (309.7)</td><td>34.3 (297.3)</td><td>62.0 (338.2)</td><td>50.3 (560.2)</td><td>98.9 (411.2)</td><td>60.7</td></tr><tr><td>WARP</td><td>27.9 (58.0)</td><td>92.8 (64.3)</td><td>36.8 (66.1)</td><td>62.3 (70.7)</td><td>51.5 (94.8)</td><td>99.0 (67.6)</td><td>61.7</td></tr><tr><td>${\mathrm{{GTR}}}_{\mathrm{{xxl}}}$</td><td>30.0</td><td>90.0</td><td>36.6</td><td>78.0</td><td>46.6</td><td>99.7</td><td>63.5</td></tr><tr><td>T5-ColBERT ${}_{xx}$ l</td><td>29.0</td><td>94.6</td><td>38.5</td><td>72.5</td><td>50.1</td><td>99.1</td><td>64.0</td></tr><tr><td>${\mathrm{{XTR}}}_{\mathrm{{xxl}}}$</td><td>30.7</td><td>95.0</td><td>39.4</td><td>73.0</td><td>52.7</td><td>99.3</td><td>65.0</td></tr></table>

<table><tbody><tr><td></td><td>近场通信语料库（NFCorpus）</td><td>科学事实数据集（SciFact）</td><td>科学文档数据集（SCIDOCS）</td><td>金融问答2018数据集（FiQA - 2018）</td><td>Touché 2020任务数据集</td><td>Quora问答平台</td><td>平均值（Avg.）</td></tr><tr><td>二元独立模型25（BM25）</td><td>25.0</td><td>90.8</td><td>35.6</td><td>53.9</td><td>53.8</td><td>97.3</td><td>59.4</td></tr><tr><td>ColBERT模型</td><td>25.4</td><td>87.8</td><td>34.4</td><td>60.3</td><td>43.9</td><td>98.9</td><td>58.5</td></tr><tr><td>${\mathrm{{GTR}}}_{\text{base }}$</td><td>27.5</td><td>87.2</td><td>34.0</td><td>67.0</td><td>44.3</td><td>99.6</td><td>59.9</td></tr><tr><td>T5 - ColBERT ${}_{\text{base}}$</td><td>27.6</td><td>91.3</td><td>34.2</td><td>63.0</td><td>49.9</td><td>97.9</td><td>60.7</td></tr><tr><td>XTR/ScaNN算法</td><td>28.0 (158.1)</td><td>90.9 (309.7)</td><td>34.3 (297.3)</td><td>62.0 (338.2)</td><td>50.3 (560.2)</td><td>98.9 (411.2)</td><td>60.7</td></tr><tr><td>WARP损失函数</td><td>27.9 (58.0)</td><td>92.8 (64.3)</td><td>36.8 (66.1)</td><td>62.3 (70.7)</td><td>51.5 (94.8)</td><td>99.0 (67.6)</td><td>61.7</td></tr><tr><td>${\mathrm{{GTR}}}_{\mathrm{{xxl}}}$</td><td>30.0</td><td>90.0</td><td>36.6</td><td>78.0</td><td>46.6</td><td>99.7</td><td>63.5</td></tr><tr><td>T5 - ColBERT ${}_{xx}$ l</td><td>29.0</td><td>94.6</td><td>38.5</td><td>72.5</td><td>50.1</td><td>99.1</td><td>64.0</td></tr><tr><td>${\mathrm{{XTR}}}_{\mathrm{{xxl}}}$</td><td>30.7</td><td>95.0</td><td>39.4</td><td>73.0</td><td>52.7</td><td>99.3</td><td>65.0</td></tr></tbody></table>

Table 3: Recall@100 on BEIR. Numbers in parentheses show average latency (milliseconds), with the final column displaying the average score across the datasets.

表3：BEIR数据集上的Recall@100指标。括号内的数字表示平均延迟（毫秒），最后一列显示各数据集的平均得分。

<!-- Media -->

### 5.1 End-to-End Results

### 5.1 端到端结果

Table 2 presents results on LoTTE. Our WARP method over the XTR model outperforms the optimized ${\mathrm{{XTR}}}_{\text{base }}/\mathrm{{ScaNN}}$ implementation in terms of Success@5, while significantly reducing end-to-end latency,with speedups ranging from ${4.6}\mathrm{x}$ on LoTTE Lifestyle to ${12.8}\mathrm{x}$ on LoTTE Pooled. We observe a similar trend with the evaluation of nDCG@10 on the six BEIR [18] datasets, as shown in ??. ${\mathrm{{XTR}}}_{\text{base }}/$ WARP achieves speedups of ${2.7}\mathrm{x} - 6\mathrm{x}$ over ${\mathrm{{XTR}}}_{\text{base }}/\mathrm{{ScaNN}}$ with a slight gain in nDCG@10.Likewise, we find improvements of Recall@100 on BEIR with substantial gains in end-to-end latency, but we omit them for the sake of space.

表2展示了在LoTTE数据集上的实验结果。我们基于XTR模型的WARP方法在Success@5指标上优于优化后的${\mathrm{{XTR}}}_{\text{base }}/\mathrm{{ScaNN}}$实现，同时显著降低了端到端延迟，加速比从LoTTE Lifestyle数据集的${4.6}\mathrm{x}$到LoTTE Pooled数据集的${12.8}\mathrm{x}$不等。在对六个BEIR [18]数据集的nDCG@10指标评估中，我们也观察到了类似的趋势，如??所示。${\mathrm{{XTR}}}_{\text{base }}/$ WARP相较于${\mathrm{{XTR}}}_{\text{base }}/\mathrm{{ScaNN}}$实现了${2.7}\mathrm{x} - 6\mathrm{x}$的加速比，同时nDCG@10指标略有提升。同样，我们发现WARP在BEIR数据集上的Recall@100指标有所改善，并且端到端延迟大幅降低，但为节省篇幅，此处省略相关内容。

### 5.2 Scalability

### 5.2 可扩展性

We now assess WARP's scalability in relation to both dataset size and the degree of parallelism. To study the effect of the dataset size on WARP's performance, we evaluate its latency across development datasets of varying sizes: BEIR NFCorpus, BEIR Quora, LoTTE Science, LoTTE Technology, and LoTTE Pooled (Table 1). Figure 9a plots the recorded latency of different configurations versus the size of the dataset, measured in the number of document token embeddings. Our results confirm, like in PLAID, that WARP's latency scales in general with the square root of the dataset size-this is intuitive, as the number of clusters is by design proportional to the square root of the dataset size. Figure 9b illustrates WARP's performance on the LoTTE Pooled development set, showing how the number of CPU threads impacts performance for different values of ${n}_{\text{probe }}$ . Our results indicate that WARP effectively parallelizes across multiple threads,achieving a speedup of ${3.1}\mathrm{x}$ for ${n}_{\text{probe }} = {32}$ with 16 threads.

我们现在评估WARP在数据集大小和并行度方面的可扩展性。为了研究数据集大小对WARP性能的影响，我们在不同大小的开发数据集上评估其延迟：BEIR NFCorpus、BEIR Quora、LoTTE Science、LoTTE Technology和LoTTE Pooled（表1）。图9a绘制了不同配置下记录的延迟与数据集大小的关系，数据集大小以文档词嵌入的数量衡量。我们的结果证实，与PLAID类似，WARP的延迟总体上与数据集大小的平方根成正比 — 这是符合直觉的，因为聚类的数量在设计上与数据集大小的平方根成正比。图9b展示了WARP在LoTTE Pooled开发集上的性能，显示了CPU线程数量如何影响不同${n}_{\text{probe }}$值下的性能。我们的结果表明，WARP能够有效地在多个线程上并行化，对于${n}_{\text{probe }} = {32}$，使用16个线程可实现${3.1}\mathrm{x}$的加速比。

<!-- Media -->

<!-- figureText: ${n}_{\text{probe }} = 8$ ${n}_{\text{crobe }} = 8$ ${n}_{\text{crobo }} = {16}$ ${n}_{\text{probe }} = {32}$ 40 #Threads (b) End-to-end latency for varying ${n}_{\text{probe }}$ ${n}_{\text{probe }} = {24}$ ${n}_{\text{probe }} = {32}$ Dataset Size (#Embeddings) (a) End-to-end latency vs dataset size (measured in #embeddings) -->

<img src="https://cdn.noedgeai.com/0195afaa-ab63-7b7c-94ef-22219448cc3a_6.jpg?x=928&y=264&w=687&h=347&r=0"/>

Figure 9: WARP's scaling behaviour with respect to dataset size and the number of available CPU threads

图9：WARP在数据集大小和可用CPU线程数量方面的扩展行为

<!-- Media -->

### 5.3 Memory footprint

### 5.3 内存占用

<!-- Media -->

<table><tr><td colspan="2" rowspan="2">Dataset</td><td rowspan="2">#Tokens</td><td colspan="5">XTR Index Size (GiB)</td></tr><tr><td>BruteForce</td><td>FAISS</td><td>ScaNN</td><td>${\mathrm{{WARP}}}_{\left( b = 2\right) }$</td><td>${\mathrm{{WARP}}}_{\left( b = 4\right) }$</td></tr><tr><td rowspan="6">BelR [18]</td><td>NFCORPUS</td><td>1.35M</td><td>0.65</td><td>0.06</td><td>0.18</td><td>0.06</td><td>0.10</td></tr><tr><td>SciFact</td><td>1.87M</td><td>0.91</td><td>0.08</td><td>0.25</td><td>0.07</td><td>0.13</td></tr><tr><td>SCIDOCS</td><td>6.27M</td><td>3.04</td><td>0.28</td><td>0.82</td><td>0.24</td><td>0.43</td></tr><tr><td>Quora</td><td>9.12M</td><td>4.43</td><td>0.43</td><td>1.21</td><td>0.35</td><td>0.62</td></tr><tr><td>FiQA-2018</td><td>10.23M</td><td>4.95</td><td>0.46</td><td>1.34</td><td>0.38</td><td>0.69</td></tr><tr><td>Touché-2020</td><td>92.64M</td><td>45.01</td><td>4.28</td><td>12.22</td><td>3.40</td><td>6.16</td></tr><tr><td rowspan="6">LoTTE [17]</td><td>Lifestyle</td><td>23.71M</td><td>11.51</td><td>1.08</td><td>3.12</td><td>0.88</td><td>1.59</td></tr><tr><td>Recreation</td><td>30.04M</td><td>14.59</td><td>1.38</td><td>3.96</td><td>1.11</td><td>2.01</td></tr><tr><td>Writing</td><td>${32.21}\mathrm{M}$</td><td>15.64</td><td>1.48</td><td>4.25</td><td>1.19</td><td>2.15</td></tr><tr><td>Technology</td><td>131.92M</td><td>64.12</td><td>6.13</td><td>17.44</td><td>4.83</td><td>8.77</td></tr><tr><td>Science</td><td>442.15M</td><td>214.93</td><td>20.57</td><td>58.46</td><td>16.07</td><td>29.28</td></tr><tr><td>Pooled</td><td>660.04M</td><td>320.88</td><td>30.74</td><td>87.30</td><td>23.88</td><td>43.59</td></tr><tr><td>Total</td><td>-</td><td>1.44B</td><td>700.66</td><td>66.98</td><td>190.52</td><td>52.48</td><td>95.51</td></tr></table>

<table><tbody><tr><td colspan="2" rowspan="2">数据集</td><td rowspan="2">#Token数量</td><td colspan="5">XTR索引大小（吉字节）</td></tr><tr><td>暴力搜索法</td><td>FAISS（快速近似最近邻搜索库）</td><td>ScaNN（可扩展最近邻搜索）</td><td>${\mathrm{{WARP}}}_{\left( b = 2\right) }$</td><td>${\mathrm{{WARP}}}_{\left( b = 4\right) }$</td></tr><tr><td rowspan="6">BelR [18]</td><td>NFCORPUS</td><td>1.35M</td><td>0.65</td><td>0.06</td><td>0.18</td><td>0.06</td><td>0.10</td></tr><tr><td>SciFact</td><td>1.87M</td><td>0.91</td><td>0.08</td><td>0.25</td><td>0.07</td><td>0.13</td></tr><tr><td>SCIDOCS</td><td>6.27M</td><td>3.04</td><td>0.28</td><td>0.82</td><td>0.24</td><td>0.43</td></tr><tr><td>Quora</td><td>9.12M</td><td>4.43</td><td>0.43</td><td>1.21</td><td>0.35</td><td>0.62</td></tr><tr><td>FiQA - 2018</td><td>10.23M</td><td>4.95</td><td>0.46</td><td>1.34</td><td>0.38</td><td>0.69</td></tr><tr><td>Touché - 2020</td><td>92.64M</td><td>45.01</td><td>4.28</td><td>12.22</td><td>3.40</td><td>6.16</td></tr><tr><td rowspan="6">LoTTE [17]</td><td>生活方式</td><td>23.71M</td><td>11.51</td><td>1.08</td><td>3.12</td><td>0.88</td><td>1.59</td></tr><tr><td>休闲娱乐</td><td>30.04M</td><td>14.59</td><td>1.38</td><td>3.96</td><td>1.11</td><td>2.01</td></tr><tr><td>写作</td><td>${32.21}\mathrm{M}$</td><td>15.64</td><td>1.48</td><td>4.25</td><td>1.19</td><td>2.15</td></tr><tr><td>技术</td><td>131.92M</td><td>64.12</td><td>6.13</td><td>17.44</td><td>4.83</td><td>8.77</td></tr><tr><td>科学</td><td>442.15M</td><td>214.93</td><td>20.57</td><td>58.46</td><td>16.07</td><td>29.28</td></tr><tr><td>合并的</td><td>660.04M</td><td>320.88</td><td>30.74</td><td>87.30</td><td>23.88</td><td>43.59</td></tr><tr><td>总计</td><td>-</td><td>1.44B</td><td>700.66</td><td>66.98</td><td>190.52</td><td>52.48</td><td>95.51</td></tr></tbody></table>

Table 4: Comparison of index sizes for the datasets. Note that PLAID's usage is memory usage is effectively identical to WARP's, only slightly larger.

表4：各数据集的索引大小比较。请注意，PLAID的内存使用情况实际上与WARP的几乎相同，只是略大一点。

<!-- Media -->

Thanks to the adoption of a ColBERTv2- and PLAID-like approach for compression, WARP's advantage over XTR extends also to a reduction in index size for XTR-based methods, which decreases memory requirements and, thus, broadens deployment options. Table 4 compares index sizes across all evaluated test datasets. ${\mathrm{{WARP}}}_{\left( b = 4\right) }$ demonstrates a substantially smaller index size compared to both the BruteForce and ScaNN variants,providing a ${7.3}\mathrm{x}$ and $2\mathrm{x}$ reduction in memory footprint, respectively. While indexes generated by the FAISS implementation are marginally smaller, this comes at the cost of substantially reduced quality and latency. Notably, ${\text{WARP}}_{\left( b = 2\right) }$ outperforms the FAISS implementation in terms of quality with an even smaller index size. ${}^{13}$

由于采用了类似ColBERTv2和PLAID的压缩方法，WARP相对于XTR的优势还体现在基于XTR的方法的索引大小减小上，这降低了内存需求，从而拓宽了部署选项。表4比较了所有评估测试数据集的索引大小。${\mathrm{{WARP}}}_{\left( b = 4\right) }$与BruteForce和ScaNN变体相比，索引大小显著减小，分别减少了${7.3}\mathrm{x}$和$2\mathrm{x}$的内存占用。虽然FAISS实现生成的索引略小，但这是以显著降低质量和增加延迟为代价的。值得注意的是，${\text{WARP}}_{\left( b = 2\right) }$在质量方面优于FAISS实现，且索引大小更小。${}^{13}$

---

<!-- Footnote -->

${}^{12}$ As of the recently released version 1.3.0,ScaNN supports multi-threaded search via the search_batched_parallel function.

${}^{12}$截至最近发布的1.3.0版本，ScaNN通过search_batched_parallel函数支持多线程搜索。

<!-- Footnote -->

---

## 6 Conclusion

## 6 结论

We introduce WARP, a highly optimized engine for multi-vector retrieval based on ColBERTv2 PLAID and the XTR framework. WARP overcomes inefficiencies of existing engines by: (1) the acceleration of query encoding using specialized inference runtimes, reducing inference latency by $2\mathrm{x} - 3\mathrm{x}$ ,(2) WARP ${}_{\text{SELECT }}$ ,which dynamically adjusts to dataset characteristics while simultaneously decreasing computational overhead, and (3) an optimized two-stage reduction via a dedicated $\mathrm{C} +  +$ kernel combined with implicit decompression. These optimizations culminate in substantial performance gains, including a ${41}\mathrm{x}$ speedup over XTR on LoTTE Pooled,reducing latency from above 6s to just 171ms in single-threaded execution,and a 3x reduction in latency compared to ColBERTv2/PLAID, without negatively impacting retrieval quality. Beyond single-threaded performance, WARP shows significant speedup with increased thread count, and its reduced memory footprint enables deployment on resource-constrained devices.

我们介绍了WARP，这是一个基于ColBERTv2、PLAID和XTR框架的高度优化的多向量检索引擎。WARP通过以下方式克服了现有引擎的低效问题：（1）使用专门的推理运行时加速查询编码，将推理延迟降低了$2\mathrm{x} - 3\mathrm{x}$；（2）WARP ${}_{\text{SELECT }}$，它能动态适应数据集特征，同时降低计算开销；（3）通过专用的$\mathrm{C} +  +$内核结合隐式解压缩进行优化的两阶段缩减。这些优化最终带来了显著的性能提升，包括在LoTTE Pooled上比XTR快${41}\mathrm{x}$倍，在单线程执行中将延迟从6秒以上降低到仅171毫秒，并且与ColBERTv2/PLAID相比，延迟降低了3倍，同时不会对检索质量产生负面影响。除了单线程性能外，WARP在增加线程数时显示出显著的加速，并且其减少的内存占用使其能够在资源受限的设备上部署。

## References

## 参考文献

[1] Thibault Formal, Stéphane Clinchant, Hervé Déjean, and Carlos Lassance. 2024. Splate: sparse late interaction retrieval. (2024). arXiv: 2404.13950 [cs. IR].

[1] Thibault Formal、Stéphane Clinchant、Hervé Déjean和Carlos Lassance。2024年。Splate：稀疏后期交互检索。（2024年）。arXiv：2404.13950 [计算机科学.信息检索]。

[2] Luyu Gao, Zhuyun Dai, and Jamie Callan. 2021. Coil: revisit exact lexical match in information retrieval with contextualized inverted list. (2021). https://arxiv .org/abs/2104.07186 arXiv: 2104.07186 [cs.IR].

[2] Luyu Gao、Zhuyun Dai和Jamie Callan。2021年。Coil：通过上下文倒排列表重新审视信息检索中的精确词汇匹配。（2021年）。https://arxiv.org/abs/2104.07186 arXiv：2104.07186 [计算机科学.信息检索]。

[3] Stanford Future Data Systems Research Group. 2024. colbert-ir/colbertv2.0. https://huggingface.co/colbert-ir/colbertv2.0.(2024).

[3] 斯坦福未来数据系统研究小组。2024年。colbert - ir/colbertv2.0。https://huggingface.co/colbert - ir/colbertv2.0。（2024年）。

[4] Stanford Future Data Systems Research Group. 2024. ColBERTv2/PLAID (Code). https://github.com/stanford-futuredata/ColBERT.(2024).

[4] 斯坦福未来数据系统研究小组。2024年。ColBERTv2/PLAID（代码）。https://github.com/stanford - futuredata/ColBERT。（2024年）。

[5] Ruiqi Guo, Philip Sun, Erik Lindgren, Quan Geng, David Simcha, Felix Chern, and Sanjiv Kumar. 2020. Accelerating large-scale inference with anisotropic vector quantization. (2020). https://arxiv.org/abs/1908.10396 arXiv: 1908.10396 [cs.LG].

[5] Ruiqi Guo、Philip Sun、Erik Lindgren、Quan Geng、David Simcha、Felix Chern和Sanjiv Kumar。2020年。使用各向异性向量量化加速大规模推理。（2020年）。https://arxiv.org/abs/1908.10396 arXiv：1908.10396 [计算机科学.机器学习]。

[6] Herve Jégou, Matthijs Douze, and Cordelia Schmid. 2011. Product quantization for nearest neighbor search. IEEE Transactions on Pattern Analysis and Machine Intelligence, 33, 1, 117-128. DOI: 10.1109/TPAMI.2010.57.

[6] Herve Jégou、Matthijs Douze和Cordelia Schmid。2011年。用于最近邻搜索的乘积量化。《IEEE模式分析与机器智能汇刊》，33卷，第1期，117 - 128页。DOI：10.1109/TPAMI.2010.57。

[7] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu, (Eds.) Association for Computational Linguistics, Online, (Nov. 2020), 6769-6781. DOI: 10.18653/v1/2020.emnlp-main.550.

[7] 弗拉基米尔·卡尔普欣（Vladimir Karpukhin）、巴拉斯·奥古兹（Barlas Oguz）、闵世元（Sewon Min）、帕特里克·刘易斯（Patrick Lewis）、莱德尔·吴（Ledell Wu）、谢尔盖·叶杜诺夫（Sergey Edunov）、陈丹琦（Danqi Chen）和易文涛（Wen-tau Yih）。2020 年。用于开放域问答的密集段落检索。见《2020 年自然语言处理经验方法会议论文集》（EMNLP）。邦妮·韦伯（Bonnie Webber）、特雷弗·科恩（Trevor Cohn）、何玉兰（Yulan He）和刘洋（Yang Liu）编。计算语言学协会，线上会议（2020 年 11 月），6769 - 6781 页。DOI: 10.18653/v1/2020.emnlp-main.550。

[8] Omar Khattab and Matei Zaharia. 2020. Colbert: efficient and effective passage search via contextualized late interaction over BERT. CoRR, abs/2004.12832. https://arxiv.org/abs/2004.12832 arXiv: 2004.12832.

[8] 奥马尔·哈塔卜（Omar Khattab）和马特·扎哈里亚（Matei Zaharia）。2020 年。ColBERT：通过基于 BERT 的上下文后期交互实现高效有效的段落搜索。预印本库 CoRR，编号 abs/2004.12832。https://arxiv.org/abs/2004.12832 预印本编号：2004.12832。

[9] Jinhyuk Lee, Zhuyun Dai, Sai Meher Karthik Duddu, Tao Lei, Iftekhar Naim, Ming-Wei Chang, and Vincent Y. Zhao. 2024. google/xtr-base-en. https://huggi ngface.co/google/xtr-base-en. (2024).

[9] 李镇赫（Jinhyuk Lee）、戴竹云（Zhuyun Dai）、赛·梅赫尔·卡尔蒂克·杜杜（Sai Meher Karthik Duddu）、雷涛（Tao Lei）、伊夫特哈尔·奈姆（Iftekhar Naim）、张明伟（Ming-Wei Chang）和赵文森（Vincent Y. Zhao）。2024 年。google/xtr-base-en。https://huggingface.co/google/xtr-base-en。（2024 年）

[10] Jinhyuk Lee, Zhuyun Dai, Sai Meher Karthik Duddu, Tao Lei, Iftekhar Naim, Ming-Wei Chang, and Vincent Y. Zhao. 2024. Rethinking the role of token retrieval in multi-vector retrieval. (2024). arXiv: 2304.01982 [cs.CL].

[10] 李镇赫（Jinhyuk Lee）、戴竹云（Zhuyun Dai）、赛·梅赫尔·卡尔蒂克·杜杜（Sai Meher Karthik Duddu）、雷涛（Tao Lei）、伊夫特哈尔·奈姆（Iftekhar Naim）、张明伟（Ming-Wei Chang）和赵文森（Vincent Y. Zhao）。2024 年。重新思考 Token 检索在多向量检索中的作用。（2024 年）。预印本编号：2304.01982 [计算机科学 - 计算语言学]。

[11] Jinhyuk Lee, Zhuyun Dai, Sai Meher Karthik Duddu, Tao Lei, Iftekhar Naim, Ming-Wei Chang, and Vincent Y. Zhao. 2024. XTR: Rethinking the Role of Token Retrieval in Multi-Vector Retrieval (Code). https://github.com/google-d eepmind/xtr. (2024).

[11] 李镇赫（Jinhyuk Lee）、戴竹云（Zhuyun Dai）、赛·梅赫尔·卡尔蒂克·杜杜（Sai Meher Karthik Duddu）、雷涛（Tao Lei）、伊夫特哈尔·奈姆（Iftekhar Naim）、张明伟（Ming-Wei Chang）和赵文森（Vincent Y. Zhao）。2024 年。XTR：重新思考 Token 检索在多向量检索中的作用（代码）。https://github.com/google-deepmind/xtr。（2024 年）

[12] Minghan Li, Sheng-Chieh Lin, Barlas Oguz, Asish Ghoshal, Jimmy Lin, Yashar Mehdad, Wen-tau Yih, and Xilun Chen. 2022. Citadel: conditional token interaction via dynamic lexical routing for efficient and effective multi-vector retrieval. (2022). https://arxiv.org/abs/2211.10411 arXiv: 2211.10411 [cs. IR].

[12] 李明翰（Minghan Li）、林圣杰（Sheng-Chieh Lin）、巴拉斯·奥古兹（Barlas Oguz）、阿西什·戈沙尔（Asish Ghoshal）、林吉米（Jimmy Lin）、亚沙尔·梅赫达德（Yashar Mehdad）、易文涛（Wen-tau Yih）和陈希伦（Xilun Chen）。2022 年。Citadel：通过动态词法路由实现条件 Token 交互以进行高效有效的多向量检索。（2022 年）。https://arxiv.org/abs/2211.10411 预印本编号：2211.10411 [计算机科学 - 信息检索]。

[13] Franco Maria Nardini, Cosimo Rulli, and Rossano Venturini. 2024. Efficient multi-vector dense retrieval using bit vectors. (2024). arXiv: 2404.02805 [cs. IR].

[13] 佛朗哥·玛丽亚·纳尔迪尼（Franco Maria Nardini）、科西莫·鲁利（Cosimo Rulli）和罗萨诺·文图里尼（Rossano Venturini）。2024 年。使用位向量的高效多向量密集检索。（2024 年）。预印本编号：2404.02805 [计算机科学 - 信息检索]。

${}^{13}$ Additionally,WARP reduces memory requirements compared to PLAID as it no longer

${}^{13}$ 此外，与 PLAID 相比，WARP 降低了内存需求，因为它不再

requires storing a mapping from document ID to centroids/embeddings.

需要存储从文档 ID 到质心/嵌入的映射。

[14] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2023. Exploring the limits of transfer learning with a unified text-to-text transformer. (2023). https://arxi v.org/abs/1910.10683 arXiv: 1910.10683 [cs.LG].

[14] 科林·拉菲尔（Colin Raffel）、诺姆·沙泽尔（Noam Shazeer）、亚当·罗伯茨（Adam Roberts）、凯瑟琳·李（Katherine Lee）、沙兰·纳朗（Sharan Narang）、迈克尔·马泰纳（Michael Matena）、周燕琪（Yanqi Zhou）、李伟（Wei Li）和彼得·J·刘（Peter J. Liu）。2023 年。用统一的文本到文本转换器探索迁移学习的极限。（2023 年）。https://arxiv.org/abs/1910.10683 预印本编号：1910.10683 [计算机科学 - 机器学习]。

[15] Stephen Robertson and Hugo Zaragoza. 2009. The probabilistic relevance framework: bm25 and beyond. Foundations and Trends® in Information Retrieval, 3, 4, 333-389. DOI: 10.1561/1500000019.

[15] 斯蒂芬·罗伯逊（Stephen Robertson）和雨果·萨拉戈萨（Hugo Zaragoza）。2009 年。概率相关性框架：BM25 及其他。《信息检索基础与趋势》，第 3 卷，第 4 期，333 - 389 页。DOI: 10.1561/1500000019。

[16] Keshav Santhanam, Omar Khattab, Christopher Potts, and Matei Zaharia. 2022. Plaid: an efficient engine for late interaction retrieval. (2022). arXiv: 2205.09707 [cs.IR].

[16] 凯沙夫·桑塔南姆（Keshav Santhanam）、奥马尔·哈塔卜（Omar Khattab）、克里斯托弗·波茨（Christopher Potts）和马特·扎哈里亚（Matei Zaharia）。2022 年。Plaid：一种用于后期交互检索的高效引擎。（2022 年）。预印本编号：2205.09707 [计算机科学 - 信息检索]。

[17] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. 2021. Colbertv2: effective and efficient retrieval via lightweight late interaction. CoRR, abs/2112.01488. https://arxiv.org/abs/2112.01488 arXiv: 2112.01488.

[17] 凯沙夫·桑塔南姆（Keshav Santhanam）、奥马尔·哈塔卜（Omar Khattab）、乔恩·萨德 - 法尔孔（Jon Saad-Falcon）、克里斯托弗·波茨（Christopher Potts）和马特·扎哈里亚（Matei Zaharia）。2021 年。ColBERTv2：通过轻量级后期交互实现高效有效的检索。预印本库 CoRR，编号 abs/2112.01488。https://arxiv.org/abs/2112.01488 预印本编号：2112.01488。

[18] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021. Beir: a heterogenous benchmark for zero-shot evaluation of information retrieval models. (2021). https://arxiv.org/abs/2104.08663 arXiv: 2104.08663 [cs. IR].

[18] 南丹·塔库尔（Nandan Thakur）、尼尔斯·赖默斯（Nils Reimers）、安德里亚斯·吕克莱（Andreas Rücklé）、阿比谢克·斯里瓦斯塔瓦（Abhishek Srivastava）和伊琳娜·古列维奇（Iryna Gurevych）。2021 年。BEIR：一个用于信息检索模型零样本评估的异构基准。（2021 年）。https://arxiv.org/abs/2104.08663 预印本编号：2104.08663 [计算机科学 - 信息检索]。

[19] Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed, and Arnold Overwijk. 2020. Approximate nearest neighbor negative contrastive learning for dense text retrieval. (2020). https://arxiv.org /abs/2007.00808 arXiv: 2007.00808 [cs.IR].

[19] 李雄（Lee Xiong）、熊晨燕（Chenyan Xiong）、李烨（Ye Li）、邓国峰（Kwok-Fung Tang）、刘佳琳（Jialin Liu）、保罗·贝内特（Paul Bennett）、朱奈德·艾哈迈德（Junaid Ahmed）和阿诺德·奥弗维克（Arnold Overwijk）。2020 年。用于密集文本检索的近似最近邻负对比学习。(2020)。https://arxiv.org /abs/2007.00808 arXiv: 2007.00808 [计算机科学.信息检索（cs.IR）]。

[20] Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Jiafeng Guo, Min Zhang, and Shaoping Ma. 2021. Optimizing dense retrieval model training with hard negatives. (2021). https://arxiv.org/abs/2104.08051 arXiv: 2104.08051 [cs.IR].

[20] 詹景涛（Jingtao Zhan）、毛佳欣（Jiaxin Mao）、刘一群（Yiqun Liu）、郭佳峰（Jiafeng Guo）、张敏（Min Zhang）和马少平（Shaoping Ma）。2021 年。使用难负样本优化密集检索模型训练。(2021)。https://arxiv.org/abs/2104.08051 arXiv: 2104.08051 [计算机科学.信息检索（cs.IR）]。

<!-- Media -->

<!-- figureText: ${\mathrm{{XTR}}}_{\text{base }}/\mathrm{{WARP}}$ 58ms Candidate Generation Decompression Scoring 73ms 171 ms Latency (ms) BEIR NFCorpus (Test) ${\mathrm{{XTR}}}_{\text{base }}/\mathrm{{WARP}}$ ${\mathrm{{XTR}}}_{\text{base }}/\mathrm{{WARP}}$ LoTTE Pooled (Test) -->

<img src="https://cdn.noedgeai.com/0195afaa-ab63-7b7c-94ef-22219448cc3a_8.jpg?x=170&y=242&w=672&h=245&r=0"/>

Figure 10: Breakdown of ${\mathrm{{XTR}}}_{\text{base }}$ /WARP’s avg.

图 10：${\mathrm{{XTR}}}_{\text{base }}$ /WARP 的平均

single-threaded latency for ${n}_{\text{probe }} = {32}$ on the BEIR NFCorpus, LoTTE Lifestyle, and LoTTE Pooled datasets.

在 BEIR NFCorpus、LoTTE Lifestyle 和 LoTTE Pooled 数据集上 ${n}_{\text{probe }} = {32}$ 的单线程延迟。

<!-- figureText: ${\mathrm{{XTR}}}_{\text{base }}/\mathrm{{WARP}}$ 18ms Query Encoding Candidate Generation Decompression/Scoring(fused) 54ms 75 100 125 150 175 200 Latency (ms) ${\mathrm{{XTR}}}_{\text{base }}/\mathrm{{WARP}}$ LoTTE Lifestyle (Test) ${\mathrm{{XTR}}}_{\text{base }}/\mathrm{{WARP}}$ LoTTE Pooled (Test) 0 25 50 -->

<img src="https://cdn.noedgeai.com/0195afaa-ab63-7b7c-94ef-22219448cc3a_8.jpg?x=170&y=659&w=676&h=245&r=0"/>

Figure 11: Breakdown of ${\mathrm{{XTR}}}_{\text{base }}$ /WARP’s avg. latency for ${n}_{\text{probe }} = {32}$ and ${n}_{\text{threads }} = {16}$ on the BEIR NFCorpus,LoTTE Lifestyle, and LoTTE Pooled datasets

图 11：在 BEIR NFCorpus、LoTTE Lifestyle 和 LoTTE Pooled 数据集上 ${\mathrm{{XTR}}}_{\text{base }}$ /WARP 对 ${n}_{\text{probe }} = {32}$ 和 ${n}_{\text{threads }} = {16}$ 的平均延迟细分

<!-- Media -->

## A Additional Results

## A 附加结果

### A.1 Latency Breakdowns

### A.1 延迟细分

Next,we provide a more detailed breakdown of ${\mathrm{{XTR}}}_{\text{base }}$ /WARP’s performance on three datasets of varying sizes: BEIR NFCorpus [18], LoTTE Lifestyle [17], and LoTTE Pooled [17]. Figure 10 illustrates the latency breakdown across four key stages: query encoding, candidate generation, decompression, and scoring. For the smallest dataset, BEIR NFCorpus, the total latency is 58ms, with query encoding dominating the process. Moving to the larger LoTTE Lifestyle dataset,the total latency increases to ${73}\mathrm{\;{ms}}$ . Notably,on this dataset with over ${100}\mathrm{\;K}$ passages,WARP’s entire retrieval pipeline - comprising candidate generation, decompression, and scoring - constitutes only about ${25}\%$ of the end-to-end latency,with the remaining time spent on query encoding. Even for the largest dataset, LoTTE Pooled,where the total latency reaches ${171}\mathrm{\;{ms}}$ ,we observe that query encoding still consumes the majority of the processing time. While the other stages become more pronounced, query encoding remains the single most time-consuming stage of the retrieval process. Without the use of specialized inference runtimes, query encoding accounts for approximately half of the execution time, thus presenting the primary bottleneck for end-to-end retrieval using WARP.

接下来，我们将更详细地分析 ${\mathrm{{XTR}}}_{\text{base }}$ /WARP 在三个不同规模数据集上的性能：BEIR NFCorpus [18]、LoTTE Lifestyle [17] 和 LoTTE Pooled [17]。图 10 展示了四个关键阶段的延迟细分：查询编码、候选生成、解压缩和评分。对于最小的数据集 BEIR NFCorpus，总延迟为 58 毫秒，其中查询编码占主导地位。对于较大的 LoTTE Lifestyle 数据集，总延迟增加到 ${73}\mathrm{\;{ms}}$。值得注意的是，在这个包含超过 ${100}\mathrm{\;K}$ 个段落的数据集上，WARP 的整个检索流程（包括候选生成、解压缩和评分）仅占端到端延迟的约 ${25}\%$，其余时间用于查询编码。即使对于最大的数据集 LoTTE Pooled，总延迟达到 ${171}\mathrm{\;{ms}}$，我们也发现查询编码仍然占用了大部分处理时间。虽然其他阶段变得更加明显，但查询编码仍然是检索过程中最耗时的单个阶段。如果不使用专门的推理运行时，查询编码约占执行时间的一半，因此成为使用 WARP 进行端到端检索的主要瓶颈。

A key advantage of WARP over the reference implementation is its ability to leverage multi-threading, thereby significantly improving performance. Figure 11 illustrates the end-to-end latency breakdown for WARP using 16 threads. The decompression and scoring stages are fused in multi-threaded contexts. WARP demonstrates great scalability, achieving substantial latency reduction across all stages. In the 16-thread configuration, it notably surpasses the GPU-based implementation of PLAID on the LoTTE Pooled dataset.

WARP 相对于参考实现的一个关键优势是它能够利用多线程，从而显著提高性能。图 11 展示了使用 16 个线程时 WARP 的端到端延迟细分。在多线程环境中，解压缩和评分阶段是融合的。WARP 具有很强的可扩展性，在所有阶段都实现了显著的延迟降低。在 16 线程配置下，它在 LoTTE Pooled 数据集上明显超过了基于 GPU 的 PLAID 实现。

<!-- Media -->

<!-- figureText: ${\mathrm{{XTR}}}_{\text{base }}/\mathrm{{ScaNN}}$ 942ms 322ms Query Encoding Candidate Generation Filtering Scoring 400 600 800 1000 Latency (ms) (k’ = 40000, opt=False) ${\mathrm{{XTR}}}_{\text{base }}/\mathrm{{ScaNN}}$ 158ms (k’ = 40000, opt=True) ${\mathrm{{CoIBERT}}}_{\mathrm{v}2}/\mathrm{{PLAID}}$ (k = 1000) ${\mathrm{{XTR}}}_{\mathrm{{base}}}/\mathrm{{WARP}}$ 58ms $\left( {{\mathrm{n}}_{\text{probe }} = {32}}\right)$ 200 -->

<img src="https://cdn.noedgeai.com/0195afaa-ab63-7b7c-94ef-22219448cc3a_8.jpg?x=950&y=241&w=678&h=294&r=0"/>

Figure 12: Latency breakdown of the unoptimized reference implementation, optimized variant, ColBERTv2/PLAID, and ${\mathrm{{XTR}}}_{\text{base }}$ /WARP on BEIR NFCorpus Test

图 12：未优化的参考实现、优化变体、ColBERTv2/PLAID 和 ${\mathrm{{XTR}}}_{\text{base }}$ /WARP 在 BEIR NFCorpus 测试集上的延迟细分

<!-- figureText: ${\mathrm{{XTR}}}_{\text{base }}/\mathrm{{ScaNN}}$ 2018ms Query Encoding Candidate Generation Filtering Decompression Scoring 1000 1500 2000 Latency (ms) (k’ = 40000, opt=False) ${\mathrm{{XTR}}}_{\text{base }}/\mathrm{{ScaNN}}$ 333ms ColBERT ${}_{v2}$ /PLAID (k = 1 000) 73ms $\left( {{\mathrm{n}}_{\text{probe }} = {32}}\right)$ 0 500 -->

<img src="https://cdn.noedgeai.com/0195afaa-ab63-7b7c-94ef-22219448cc3a_8.jpg?x=946&y=697&w=679&h=292&r=0"/>

Figure 13: Latency breakdown of the unoptimized reference implementation, optimized variant, ColBERTv2/PLAID, and ${\mathrm{{XTR}}}_{\text{base }}$ /WARP on LoTTE Lifestyle Test

图 13：未优化的参考实现、优化变体、ColBERTv2/PLAID 和 ${\mathrm{{XTR}}}_{\text{base }}$ /WARP 在 LoTTE Lifestyle 测试集上的延迟细分

<!-- Media -->

### A.2 Performance Comparisons

### A.2 性能比较

Similar to Figure 1, we analyze the performance of WARP and contrast it with the performance of the baseline on the BEIR NFCorpus (Figure 12) and LoTTE Lifestyle (Figure 13) datasets. We find that WARP's single-threaded end-to-end latency is dominated by query encoding on BEIR NFCorpus and LoTTE Lifestyle, whereas the baselines introduce significant overhead via their retrieval pipeline.

与图 1 类似，我们分析了 WARP 的性能，并将其与 BEIR NFCorpus（图 12）和 LoTTE Lifestyle（图 13）数据集上的基线性能进行对比。我们发现，在 BEIR NFCorpus 和 LoTTE Lifestyle 上，WARP 的单线程端到端延迟主要由查询编码决定，而基线通过其检索流程引入了显著的开销。

### A.3 Evaluation of ColBERTv2/WARP

### A.3 ColBERTv2/WARP 评估

To assess WARP's ability to generalize beyond the XTR model, we conduct experiments using ColBERTv2 in place of ${\mathrm{{XTR}}}_{\text{base }}$ for query encoding. The results, presented in Table 5, show that WARP performs competitively with PLAID, despite not being specifically designed for retrieval with ColBERTv2. This suggests that WARP's approach may generalize effectively to retrieval models other than XTR. We leave more detailed analysis for future work.

为了评估WARP在XTR模型之外的泛化能力，我们使用ColBERTv2代替${\mathrm{{XTR}}}_{\text{base }}$进行查询编码并开展实验。表5中呈现的结果显示，尽管WARP并非专门为使用ColBERTv2进行检索而设计，但它与PLAID的表现相当。这表明WARP的方法可能能有效地泛化到XTR之外的其他检索模型。我们将更详细的分析留待未来工作。

<!-- Media -->

<table><tr><td/><td>NFCorpus</td><td>SciFact</td><td>SCIDOCS</td><td>FiQA-2018</td><td>Touché-2020</td><td>Quora</td><td>Avg.</td></tr><tr><td>${\mathrm{{ColBERT}}}_{\mathrm{v}2}$ /PLAID (k= 10)</td><td>33.3</td><td>69.0</td><td>15.3</td><td>34.5</td><td>25.6</td><td>85.1</td><td>43.8</td></tr><tr><td>ColBERTv2/PLAID (k= 100)</td><td>33.4</td><td>69.2</td><td>15.3</td><td>35.4</td><td>25.2</td><td>85.4</td><td>44.0</td></tr><tr><td>${\mathrm{{ColBERT}}}_{\mathrm{V}2}/\mathrm{{PLAID}}\left( {\mathrm{k} = {1000}}\right)$</td><td>33.5</td><td>69.2</td><td>15.3</td><td>$\underline{35.5}$</td><td>25.6</td><td>85.5</td><td>44.1</td></tr><tr><td>${\mathrm{{ColBERT}}}_{\mathrm{v}2}/\mathrm{{WARP}}\left( {{\mathrm{n}}_{\text{probe }} = {32}}\right)$</td><td>$\underline{34.6}$</td><td>70.6</td><td>16.2</td><td>33.6</td><td>26.4</td><td>84.5</td><td>44.3</td></tr></table>

<table><tbody><tr><td></td><td>近场通信语料库（NFCorpus）</td><td>科学事实数据集（SciFact）</td><td>科学文档数据集（SCIDOCS）</td><td>金融问答2018数据集（FiQA - 2018）</td><td>Touché 2020评测任务</td><td>Quora问答平台</td><td>平均值（Avg.）</td></tr><tr><td>${\mathrm{{ColBERT}}}_{\mathrm{v}2}$ /PLAID（k = 10）</td><td>33.3</td><td>69.0</td><td>15.3</td><td>34.5</td><td>25.6</td><td>85.1</td><td>43.8</td></tr><tr><td>ColBERTv2/PLAID（k = 100）</td><td>33.4</td><td>69.2</td><td>15.3</td><td>35.4</td><td>25.2</td><td>85.4</td><td>44.0</td></tr><tr><td>${\mathrm{{ColBERT}}}_{\mathrm{V}2}/\mathrm{{PLAID}}\left( {\mathrm{k} = {1000}}\right)$</td><td>33.5</td><td>69.2</td><td>15.3</td><td>$\underline{35.5}$</td><td>25.6</td><td>85.5</td><td>44.1</td></tr><tr><td>${\mathrm{{ColBERT}}}_{\mathrm{v}2}/\mathrm{{WARP}}\left( {{\mathrm{n}}_{\text{probe }} = {32}}\right)$</td><td>$\underline{34.6}$</td><td>70.6</td><td>16.2</td><td>33.6</td><td>26.4</td><td>84.5</td><td>44.3</td></tr></tbody></table>

Table 5: ColBERTv2/WARP nDCG@10 on BEIR. The last column shows the average over 6 BEIR datasets.

表5：ColBERTv2/WARP在BEIR数据集上的nDCG@10指标。最后一列显示了6个BEIR数据集的平均值。

<!-- Media -->

---

<!-- Footnote -->

Received 20 February 2007; revised 12 March 2009; accepted 5 June 2009

2007年2月20日收到；2009年3月12日修订；2009年6月5日接受

<!-- Footnote -->

---