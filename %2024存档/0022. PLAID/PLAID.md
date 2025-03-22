# PLAID: An Efficient Engine for Late Interaction Retrieval

# PLAID：一种高效的后期交互检索引擎

Keshav Santhanam*

凯沙夫·桑塔南（Keshav Santhanam）*

keshav2@stanford.edu

Stanford University

斯坦福大学

United States

美国

Omar Khattab*

奥马尔·哈塔卜（Omar Khattab）*

okhattab@stanford.edu

Stanford University

斯坦福大学

United States

美国

Christopher Potts

克里斯托弗·波茨（Christopher Potts）

Stanford University

斯坦福大学

United States

美国

Matei Zaharia

马特·扎哈里亚（Matei Zaharia）

Stanford University

斯坦福大学

United States

美国

## ABSTRACT

## 摘要

Pre-trained language models are increasingly important components across multiple information retrieval (IR) paradigms. Late interaction, introduced with the ColBERT model and recently refined in ColBERTv2, is a popular paradigm that holds state-of-the-art status across many benchmarks. To dramatically speed up the search latency of late interaction, we introduce the Performance-optimized Late Interaction Driver (PLAID). Without impacting quality, PLAID swiftly eliminates low-scoring passages using a novel centroid interaction mechanism that treats every passage as a lightweight bag of centroids. PLAID uses centroid interaction as well as centroid pruning, a mechanism for sparsifying the bag of centroids, within a highly-optimized engine to reduce late interaction search latency by up to $7 \times$ on a GPU and ${45} \times$ on a CPU against vanilla ColBERTv2, while continuing to deliver state-of-the-art retrieval quality. This allows the PLAID engine with ColBERTv2 to achieve latency of tens of milliseconds on a GPU and tens or just few hundreds of milliseconds on a CPU at large scale, even at the largest scales we evaluate with ${140}\mathrm{M}$ passages.

预训练语言模型在多种信息检索（IR）范式中日益成为重要组成部分。后期交互是一种流行的范式，由ColBERT模型引入，并在ColBERTv2中得到了最近的改进，在许多基准测试中处于领先地位。为了显著加快后期交互的搜索延迟，我们推出了性能优化的后期交互驱动程序（PLAID）。在不影响检索质量的情况下，PLAID通过一种新颖的质心交互机制迅速排除低得分段落，该机制将每个段落视为一个轻量级的质心集合。PLAID在一个高度优化的引擎中使用质心交互以及质心剪枝（一种使质心集合稀疏化的机制），与原始的ColBERTv2相比，在GPU上可将后期交互搜索延迟最多降低$7 \times$，在CPU上最多降低${45} \times$，同时仍能提供最先进的检索质量。这使得搭载ColBERTv2的PLAID引擎即使在我们以${140}\mathrm{M}$个段落进行评估的最大规模下，在GPU上也能实现几十毫秒的延迟，在CPU上实现几十或几百毫秒的延迟。

## 1 INTRODUCTION

## 1 引言

Recent advances in neural information retrieval (IR) have led to notable gains on retrieval benchmarks and retrieval-based NLP tasks. Late interaction, introduced in ColBERT [22], is a paradigm that delivers state-of-the-art quality in many of these settings, including passage ranking $\left\lbrack  {{14},{42},{48}}\right\rbrack$ ,open-domain question answering [21, 24], conversational tasks [35, 38], and beyond [20, 54]. ColBERT and its variants encode queries and documents into token-level vectors and conduct scoring via scalable yet fine-grained interactions at the level of tokens (Figure 1), alleviating the dot-product bottleneck of single-vector representations. The recent ColBERTv2 [42] model demonstrates that late interaction models often considerably outperform recent single-vector and sparse representations within and outside the training domain, a finding echoed in several recent studies $\left\lbrack  {{26},{29},{43},{44},{51},{53}}\right\rbrack$ .

神经信息检索（IR）的最新进展在检索基准测试和基于检索的自然语言处理（NLP）任务中取得了显著成果。由ColBERT [22]提出的后期交互（late interaction）是一种范式，在许多此类场景中都能提供最先进的性能，包括段落排序 $\left\lbrack  {{14},{42},{48}}\right\rbrack$、开放域问答 [21, 24]、对话任务 [35, 38] 等 [20, 54]。ColBERT及其变体将查询和文档编码为词元级别的向量，并通过在词元级别进行可扩展且细粒度的交互来进行评分（图1），缓解了单向量表示的点积瓶颈。最近的ColBERTv2 [42] 模型表明，后期交互模型在训练领域内外通常显著优于最近的单向量和稀疏表示，这一发现也在最近的几项研究中得到了印证 $\left\lbrack  {{26},{29},{43},{44},{51},{53}}\right\rbrack$。

Despite its strong retrieval quality, late interaction requires special infrastructure $\left\lbrack  {{22},{25}}\right\rbrack$ for low-latency retrieval as it encodes each query and each document as a full matrix. Most IR models represent documents as a single vector, either sparse (e.g., BM25 [41]; SPLADE [11]) or dense (e.g., DPR [18]; ANCE [49]), and thus mature sparse retrieval strategies like WAND [5] or dense kNN methods like HNSW [31] cannot be applied directly or optimally to late interaction. While recent work $\left\lbrack  {{28},{42},{45}}\right\rbrack$ has explored optimizing individual components of ColBERT's pipeline, an end-to-end optimized engine has never been studied to our knowledge.

尽管后期交互的检索质量很高，但由于它将每个查询和每个文档都编码为完整的矩阵，因此需要特殊的基础设施 $\left\lbrack  {{22},{25}}\right\rbrack$ 来实现低延迟检索。大多数信息检索（IR）模型将文档表示为单个向量，要么是稀疏的（例如BM25 [41]；SPLADE [11]），要么是密集的（例如DPR [18]；ANCE [49]），因此像WAND [5] 这样成熟的稀疏检索策略或像HNSW [31] 这样的密集最近邻（kNN）方法不能直接或最优地应用于后期交互。据我们所知，虽然最近的工作 $\left\lbrack  {{28},{42},{45}}\right\rbrack$ 已经探索了优化ColBERT流程的各个组件，但尚未对端到端优化的引擎进行研究。

We study how to optimize late-interaction search latency at a large scale, taking all steps of retrieval into account. We build on the state-of-the-art ColBERTv2 model. Besides improving quality with denoised supervision, ColBERTv2 aggressively compresses the storage footprint of late interaction. It reduces the index size by up to an order of magnitude using residual representations (§3.1). In those, each vector in a passage is encoded using the ID of its nearest centroid that approximates its token semantics-among tens or hundreds of thousands of centroids obtained through $k$ -means clustering-and a quantized residual vector.

我们研究如何在大规模情况下优化后期交互的搜索延迟，同时考虑检索的所有步骤。我们基于最先进的ColBERTv2模型进行研究。除了通过去噪监督提高性能外，ColBERTv2还大幅压缩了后期交互的存储占用。它使用残差表示（§3.1）将索引大小最多缩小了一个数量级。在这些表示中，段落中的每个向量都使用其最近质心的ID进行编码，该质心在通过 $k$ -均值聚类获得的数万或数十万个质心中近似表示其词元语义，同时还使用一个量化的残差向量。

We introduce the Performance-optimized Late Interaction Driver (PLAID), ${}^{1}$ an efficient retrieval engine that reduces late interaction search latency by ${2.5} - 7 \times$ on GPU and $9 - {45} \times$ on CPU against vanilla ColBERTv2 while retaining high quality. This allows the PLAID implementation of ColBERTv2, PLAID ColBERTv2, to achieve CPU-only latency of tens or just few hundreds of milliseconds and GPU latency of few tens of milliseconds at very large scale,even on ${140}\mathrm{M}$ passages. Crucially, PLAID ColBERTv2 does so while continuing to deliver state-of-the-art retrieval quality.

我们推出了性能优化的后期交互驱动程序（PLAID） ${}^{1}$，这是一种高效的检索引擎，与原始的ColBERTv2相比，它在GPU上可将后期交互搜索延迟降低 ${2.5} - 7 \times$，在CPU上可降低 $9 - {45} \times$，同时保持较高的性能。这使得ColBERTv2的PLAID实现（即PLAID ColBERTv2）能够在非常大规模的情况下，甚至在 ${140}\mathrm{M}$ 个段落上，实现仅使用CPU时几十或几百毫秒的延迟，以及使用GPU时几十毫秒的延迟。至关重要的是，PLAID ColBERTv2在实现这一目标的同时，仍能提供最先进的检索质量。

To dramatically speed up search, PLAID leverages the centroid component of the ColBERTv2 representations, which is a compact integer ID per token. Instead of exhaustively scoring all passages found with nearest-neighbor search, PLAID uses the centroids to identify high-scoring passages and eliminate weaker candidates without loading their larger residuals. We conduct this in a multistage pipeline and introduce centroid interaction, a scoring mechanism that treats every passage as a lightweight bag of centroid IDs. We show that this centroid-only multi-vector search exhibits high recall without using the vector residuals (§3.3), allowing us to reserve full scoring to a very small number of candidate passages. Because the centroids come from a fixed set (i.e., constitute a discrete vocabulary), the distance between the query vectors and all centroids can be computed once during search and re-used across all bag-of-centroids passage representations. This allows us to further leverage the centroid scores for centroid pruning, which sparsifies the bag of centroid representations in the earlier stages of retrieval by skipping centroid IDs that are distant from all query vectors.

为了显著加快搜索速度，PLAID利用了ColBERTv2表示中的质心组件，每个词元对应一个紧凑的整数ID。PLAID不是对通过最近邻搜索找到的所有段落进行详尽评分，而是使用质心来识别高分段落，并在不加载较大残差的情况下排除较弱的候选段落。我们在一个多阶段的流程中进行这一操作，并引入了质心交互，这是一种将每个段落视为轻量级质心ID集合的评分机制。我们表明，这种仅使用质心的多向量搜索在不使用向量残差的情况下也能表现出较高的召回率（§3.3），使我们能够将完整评分仅应用于极少数候选段落。由于质心来自一个固定的集合（即构成一个离散的词汇表），因此在搜索过程中可以一次性计算查询向量与所有质心之间的距离，并在所有质心集合的段落表示中重复使用。这使我们能够进一步利用质心分数进行质心剪枝，即在检索的早期阶段，通过跳过与所有查询向量距离较远的质心ID，对质心表示集合进行稀疏化处理。

In the PLAID engine, we implement centroid interaction and centroid pruning and implement optimized yet modular kernels for the data movement, decompression, and scoring components of late interaction with the residual representations of ColBERTv2 (§4.5). We extensively evaluate the quality and efficiency of PLAID within and outside the training domain (on MS MARCO v1 [36] and v2 [6], Wikipedia, and LoTTE [42]) and across a wide range of corpus sizes(2M - 140Mpassages),search depths $\left( {k = {10},{100},{1000}}\right)$ ,and hardware settings with single- and multi-threaded CPU and with a GPU (§5.2). We also conduct a detailed ablation study to understand the empirical sources of gains among centroid interaction, centroid pruning, and our faster kernels (§5.3).

在PLAID引擎中，我们实现了质心交互（centroid interaction）和质心剪枝（centroid pruning），并为后期交互（late interaction）的数据移动、解压缩和评分组件以及ColBERTv2的残差表示实现了优化且模块化的内核（§4.5）。我们在训练领域内外（在MS MARCO v1 [36]和v2 [6]、维基百科和LoTTE [42]上），并在广泛的语料库规模（200万 - 1.4亿个段落）、搜索深度$\left( {k = {10},{100},{1000}}\right)$以及单线程和多线程CPU和GPU的硬件设置下，对PLAID的质量和效率进行了广泛评估（§5.2）。我们还进行了详细的消融研究，以了解质心交互、质心剪枝和我们更快的内核带来增益的实证来源（§5.3）。

---

<!-- Footnote -->

${}^{1}$ Code maintained at https://github.com/stanford-futuredata/ColBERT.As of May'22, PLAID lies under the branch fast_search but will soon be merged upstream.

${}^{1}$ 代码维护在https://github.com/stanford-futuredata/ColBERT。截至2022年5月，PLAID位于fast_search分支下，但很快将合并到上游。

"Equal contribution.

同等贡献。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: MaxSim score MaxSim ... Passage Encoder Offline Indexing ... Passage Question Encoder Question -->

<img src="https://cdn.noedgeai.com/0195afc2-9af9-789e-9f2d-8c0aed3d29e7_1.jpg?x=163&y=236&w=692&h=420&r=0"/>

Figure 1: The late interaction architecture, given a query and a passage. Diagram from Khattab et al. [21] with permission.

图1：给定一个查询和一个段落的后期交互架构。经许可引用自Khattab等人[21]的图表。

<!-- Media -->

In summary, we make the following contributions:

综上所述，我们做出了以下贡献：

(1) We analyze centroid-only retrieval with ColBERTv2, showing that a pruned bag-of-centroids representation can support high-recall candidate generation (§3).

(1) 我们分析了使用ColBERTv2进行仅质心检索的情况，表明经过剪枝的质心袋表示可以支持高召回率的候选生成（§3）。

(2) We propose PLAID, a retrieval engine that introduces centroid interaction and centroid pruning as well as optimized implementations of these techniques for dramatically improving the latency of late-interaction search (§4).

(2) 我们提出了PLAID，这是一个检索引擎，它引入了质心交互和质心剪枝，以及这些技术的优化实现，以显著提高后期交互搜索的延迟（§4）。

(3) We extensively evaluate PLAID and conduct a large-scale evaluation up to ${140}\mathrm{M}$ passages,the largest to our knowledge with late-interaction retrievers (§5).

(3) 我们对PLAID进行了广泛评估，并对多达${140}\mathrm{M}$个段落进行了大规模评估，据我们所知，这是使用后期交互检索器进行的最大规模评估（§5）。

## 2 RELATED WORK

## 2 相关工作

### 2.1 Neural IR

### 2.1 神经信息检索

The IR community has introduced many neural IR models based on pre-trained Transformers. Whereas early models were primarily cross-encoders $\left\lbrack  {{27},{37}}\right\rbrack$ that attend jointly to queries and passages, many subsequent models target higher efficiency by producing independent representations for queries and passages. Some of those produce sparse term weights $\left\lbrack  {7,{32}}\right\rbrack$ ,whereas others encode each passage or query into a single vector $\left\lbrack  {{18},{39},{49}}\right\rbrack$ or multi-vector representation (the class we study; $\left\lbrack  {{12},{15},{21},{22},{42}}\right\rbrack$ ). These choices make different tradeoffs about efficiency and quality: whereas sparse term weights and single-vector models can be particularly lightweight in some settings, multi-vector late interaction [22] can often result in considerably stronger quality and robustness. Orthogonal to the choice of modeling query-document interactions, researchers have improved the supervision for neural models with harder negatives $\left\lbrack  {{21},{49},{52}}\right\rbrack$ as well as distillation and denoising $\left\lbrack  {{13},{39},{40}}\right\rbrack$ ,among other approaches. Our work extends ColBERTv2 [42], which combines late interaction modeling with hard negative and denoising supervision to achieve state-of-the-art quality among standalone retrievers.

信息检索（IR）社区基于预训练的Transformer引入了许多神经信息检索模型。早期的模型主要是交叉编码器$\left\lbrack  {{27},{37}}\right\rbrack$，它们联合处理查询和段落，而许多后续模型通过为查询和段落生成独立的表示来提高效率。其中一些模型生成稀疏的词项权重$\left\lbrack  {7,{32}}\right\rbrack$，而另一些模型则将每个段落或查询编码为单个向量$\left\lbrack  {{18},{39},{49}}\right\rbrack$或多向量表示（我们研究的类别；$\left\lbrack  {{12},{15},{21},{22},{42}}\right\rbrack$）。这些选择在效率和质量方面进行了不同的权衡：虽然稀疏词项权重和单向量模型在某些设置下可能特别轻量级，但多向量后期交互[22]通常可以带来显著更强的质量和鲁棒性。与建模查询 - 文档交互的选择正交，研究人员通过使用更难的负样本$\left\lbrack  {{21},{49},{52}}\right\rbrack$以及蒸馏和去噪$\left\lbrack  {{13},{39},{40}}\right\rbrack$等方法改进了神经模型的监督。我们的工作扩展了ColBERTv2 [42]，它将后期交互建模与难负样本和去噪监督相结合，在独立检索器中实现了最先进的质量。

<!-- Media -->

<!-- figureText: Query Encoding Decompression Scoring 287 200 300 Latency (ms) 200 300 Latency (ms) (b) PLAID ColBERTv2 ( $k = {1000}$ ) Candidate Generation Index Lookup 100 (a) Vanilla ColBERTv2 (nprobe $= 4$ ,ncandidates $= {2}^{16}$ ). 100 -->

<img src="https://cdn.noedgeai.com/0195afc2-9af9-789e-9f2d-8c0aed3d29e7_1.jpg?x=994&y=244&w=584&h=651&r=0"/>

Figure 2: Latency breakdown of MS MARCO v1 dev queries run with vanilla ColBERTv2 and PLAID ColBERTv2 on a TITAN V GPU. Vanilla ColBERTv2 is overwhelmingly bottle-necked with the cost of index lookup and decompression, a challenge that PLAID addresses.

图2：在TITAN V GPU上使用原始ColBERTv2和PLAID ColBERTv2运行MS MARCO v1开发集查询的延迟分解。原始ColBERTv2在索引查找和解压缩成本方面存在严重瓶颈，PLAID解决了这一挑战。

<!-- Media -->

### 2.2 Pruning for Sparse and Dense Retrieval

### 2.2 稀疏和密集检索的剪枝

For sparse retrieval models, traditional IR has a wealth of work on fast strategies for skipping documents for top- $k$ search. Strategies often keep metadata like term score upper bounds to skip lower-scoring candidates and most follow a Document-At-A-Time (DAAT) scoring approach $\left\lbrack  {5,8,9,{19},{33},{47}}\right\rbrack$ . Refer to Tonellotto et al. [46] for a detailed treatment of recent methods. A key difference to our settings is that these all strategies expect a set of precomputed scores (particularly, useful upper bounds on every term-document pair), whereas with late interaction the term-document interaction (i.e., the MaxSim score) is only known at query time after a matrix-vector multiplication. Our observations about the utility of centroids for accelerating late interaction successfully moves the problem closer to classical IR, but poses the challenge that the query-to-centroid scores are only known at query time.

对于稀疏检索模型，传统的信息检索在为前$k$搜索跳过文档的快速策略方面有大量工作。这些策略通常会保留词项得分上限等元数据，以跳过得分较低的候选，并且大多数采用逐文档（Document-At-A-Time，DAAT）评分方法$\left\lbrack  {5,8,9,{19},{33},{47}}\right\rbrack$。有关近期方法的详细处理，请参考Tonellotto等人[46]的研究。与我们的设置的一个关键区别是，所有这些策略都期望有一组预先计算的分数（特别是每个词项 - 文档对的有用上限），而在后期交互中，词项 - 文档交互（即MaxSim分数）只有在查询时经过矩阵 - 向量乘法后才知道。我们关于质心对加速后期交互的效用的观察成功地将问题向经典信息检索靠拢，但带来了查询到质心的分数只有在查询时才知道的挑战。

For dense retrieval models that use single-vector representations, approximate $k$ -nearest neighbor (ANN) search is a well-studied problem $\left\lbrack  {1,{16},{17},{31}}\right\rbrack$ . Our focus extends such work from a single vector to the late interaction of two matrices.

对于使用单向量表示的密集检索模型，近似$k$近邻（ANN）搜索是一个经过深入研究的问题$\left\lbrack  {1,{16},{17},{31}}\right\rbrack$。我们的研究将此类工作从单向量扩展到两个矩阵的后期交互。

## 3 ANALYSIS OF COLBERTV2 RETRIEVAL

## 3 ColBERTv2检索分析

We begin by a preliminary investigation of the latency (§3.2) and scoring patterns (§3.3) of ColBERTv2 retrieval that motivates our work on PLAID. To make this section self-contained, §3.1 reviews the modeling, storage, and supervision of ColBERTv2.

我们首先对ColBERTv2检索的延迟（§3.2）和评分模式（§3.3）进行初步研究，这激发了我们对PLAID的研究。为了使本节内容完整，§3.1回顾了ColBERTv2的建模、存储和监督。

<!-- Media -->

<!-- figureText: MS MARCO v1 LoTTE pooled 1.00 Recall of top 1000 0.95 0.90 0.80 0.70 0.60 600 800 1000 2000 4000 6000 8000 10000 #Passages #Passages (c) $k = {1000}$ 1.00 1.00 Recall of top 10 0.95 Recall of top 100 0.95 0.90 0.80 0.70 0.90 0.80 0.70 0.60 0.60 20 40 60 80 100 120 140 200 400 #Passages (a) $k = {10}$ (b) $k = {100}$ -->

<img src="https://cdn.noedgeai.com/0195afc2-9af9-789e-9f2d-8c0aed3d29e7_2.jpg?x=163&y=250&w=1473&h=436&r=0"/>

Figure 3: Recall of passages retrieved by a centroid-only version of ColBERTv2 with respect to the top $k$ passages retrieved by vanilla ColBERTv2. Centroids alone can identify virtually all of the top- $k$ passages retrieved with the full ColBERTv2 pipeline, within ${10} \cdot  k$ or fewer candidates,motivating our centroid interaction strategy.

图3：仅使用质心的ColBERTv2版本所检索到的段落相对于原始ColBERTv2检索到的前$k$个段落的召回率。仅质心就可以在${10} \cdot  k$个或更少的候选段落中识别出使用完整ColBERTv2管道检索到的几乎所有前$k$个段落，这激发了我们的质心交互策略。

<!-- Media -->

### 3.1 Modeling, Storage, and Retrieval

### 3.1 建模、存储和检索

PLAID optimizes retrieval for models using the late interaction architecture of ColBERT, which includes systems like ColBERTv2, Baleen [20], Hindsight [38], and DrDecr [24], among others. As depicted in Figure 1, a Transformer encodes queries and passages independently into vectors at the token level. For scalability, passage representations are pre-computed offline. At search time, the similarity between a query $q$ and a passage $d$ is computed as the summation of "MaxSim" operations, namely, the largest cosine similarity between each vector in the query matrix and all of the passage vectors:

PLAID针对使用ColBERT后期交互架构的模型优化检索，这些模型包括ColBERTv2、Baleen [20]、Hindsight [38]和DrDecr [24]等系统。如图1所示，Transformer在词元级别将查询和段落分别编码为向量。为了实现可扩展性，段落表示是离线预计算的。在搜索时，查询$q$和段落$d$之间的相似度计算为“MaxSim”操作的总和，即查询矩阵中的每个向量与所有段落向量之间的最大余弦相似度：

$$
{S}_{q,d} = \mathop{\sum }\limits_{{i = 1}}^{\left| Q\right| }\mathop{\max }\limits_{{j = 1}}^{\left| D\right| }{Q}_{i} \cdot  {D}_{j}^{T} \tag{1}
$$

where $Q$ and $D$ are the matrix representations of the query and passage, respectively. In doing so, this scoring function aligns each query token with the "most similar" passage token and estimates relevance as the sum of these term-level scores. Refer to Khattab and Zaharia [22] for a more complete discussion of late interaction.

其中$Q$和$D$分别是查询和段落的矩阵表示。通过这种方式，该评分函数将每个查询词元与“最相似”的段落词元对齐，并将相关性估计为这些词元级得分的总和。有关后期交互的更完整讨论，请参考Khattab和Zaharia [22]。

For storing the passage representations, we adopt the ColBERTv2 residual compression strategy, which reduces the index size by up to an order of magnitude over naive storage of late-interaction embeddings as vectors of 16-bit floating-point numbers. Instead, ColBERTv2's compression strategy efficiently clusters all token-level vectors and encodes each vector using the ID of its nearest cluster centroid as well as a quantized residual vector, wherein each dimension is 1 - or 2 -bit encoding of the delta between the centroid and the original uncompressed vector. Decompressing a vector requires locating its centroid ID, encoded using 4 bytes, and its residual, which consume 16 or 32 bytes for 1 - or 2-bit residuals, assuming the default 128-dimensional vectors.

为了存储段落表示，我们采用了ColBERTv2的残差压缩策略，与将后期交互嵌入作为16位浮点数向量进行简单存储相比，该策略可将索引大小最多缩小一个数量级。相反，ColBERTv2的压缩策略有效地对所有词元级向量进行聚类，并使用其最近聚类质心的ID以及量化残差向量对每个向量进行编码，其中每个维度是质心与原始未压缩向量之间差值的1位或2位编码。解压缩向量需要定位其质心ID（使用4字节编码）及其残差（假设默认的128维向量，1位或2位残差分别消耗16或32字节）。

While we adopt ColBERTv2's compression, we improve its retrieval strategy. We refer to the original retrieval strategy as "vanilla" ColBERTv2 retrieval. We refer to Santhanam et al. [42] for details of compression and retrieval in ColBERTv2.

虽然我们采用了ColBERTv2的压缩策略，但我们改进了其检索策略。我们将原始检索策略称为“原始”ColBERTv2检索。有关ColBERTv2中压缩和检索的详细信息，请参考Santhanam等人[42]。

<!-- Media -->

<!-- figureText: 100 0.2 0.4 0.6 0.8 1.0 Maximum Centroid Score 75 eCDF 50 25 0 -0.2 0.0 -->

<img src="https://cdn.noedgeai.com/0195afc2-9af9-789e-9f2d-8c0aed3d29e7_2.jpg?x=1032&y=849&w=505&h=339&r=0"/>

Figure 4: Centroid score distribution for each query among a random sample of 15 MS MARCO v1 dev queries evaluated with ColBERTv2.

图4：使用ColBERTv2评估的15个随机抽取的MS MARCO v1开发集查询中，每个查询的质心得分分布。

<!-- Media -->

### 3.2 ColBERTv2 Latency Breakdown

### 3.2 ColBERTv2延迟分解

Figure 2 presents a breakdown of query latency on MS MARCO Passage Ranking (v1) on a GPU, showing results for vanilla Col-BERTv2 (Figure 2a) against the new PLAID ColBERTv2 (Figure 2b). Latency is divided between query encoding, candidate generation, index lookups (i.e., to gather the compressed vector representations for candidate passages), residual decompression, and finally scoring (i.e., the final MaxSim computations).

图2展示了在GPU上对MS MARCO段落排名（v1）的查询延迟分解，显示了原始ColBERTv2（图2a）与新的PLAID ColBERTv2（图2b）的结果。延迟分为查询编码、候选生成、索引查找（即收集候选段落的压缩向量表示）、残差解压缩以及最终评分（即最终的MaxSim计算）。

For vanilla ColBERTv2, index lookup and residual decompression are overwhelming bottlenecks. Gathering vectors from the index is expensive because it consumes significant memory bandwidth: each vector in this setting is encoded with a 4-bit centroid ID and 32-byte residuals, each passage contains tens of vectors, and there can be up to ${2}^{16}$ candidate passages. Moreover,index lookup in vanilla ColBERTv2 also constructs padded tensors on the fly to deal with the variable length of passages. Decompression of residuals is comprised of several non-trivial operations such as unpacking bits and computing large sums, which can be expensive when ColBERTv2 produces a large initial candidate set $( \sim  {10} - {40}\mathrm{k}$ passages) as is the case for MS MARCO v1. While it is possible to use a smaller candidate set, doing so reduces recall (§5).

对于原始ColBERTv2，索引查找和残差解压缩是主要瓶颈。从索引中收集向量成本很高，因为它消耗大量的内存带宽：在这种情况下，每个向量使用4位质心ID和32字节残差进行编码，每个段落包含数十个向量，并且最多可能有${2}^{16}$个候选段落。此外，原始ColBERTv2中的索引查找还会动态构建填充张量以处理段落的可变长度。残差解压缩包括几个非平凡的操作，如解包位和计算大总和，当ColBERTv2生成大量初始候选集（$( \sim  {10} - {40}\mathrm{k}$个段落）时，这可能会很昂贵，就像MS MARCO v1的情况一样。虽然可以使用较小的候选集，但这样做会降低召回率（§5）。

### 3.3 Centroids Alone Identify Strong Candidates

### 3.3 仅质心即可识别强候选段落

This breakdown in Figure 2b demonstrates that exhaustively scoring a large number of candidates passages, particularly gathering and decompressing their residuals, can amount to a considerable cost. Whereas ColBERTv2 [42] exploits centroids to reduce the space footprint, our work demonstrates that the centroids can also accelerate search, while maintaining quality, by serving as proxies for the passage embeddings. Because of this, we can skip low-scoring passages without having to look up or decompress their residuals, adding some additional candidate generation overhead to achieve substantial savings in the subsequent stages (Figure 2b).

图2b中的分析表明，对大量候选段落进行详尽评分，尤其是收集和解压缩它们的残差，可能会产生相当大的成本。虽然ColBERTv2 [42]利用质心（centroids）来减少空间占用，但我们的研究表明，质心还可以在保持检索质量的同时，通过作为段落嵌入的代理来加速搜索。因此，我们可以跳过得分较低的段落，而无需查找或解压缩它们的残差，在后续阶段通过增加一些额外的候选生成开销来实现大量的成本节省（图2b）。

Effectively, we hypothesize that centroid-only retrieval can find the high-scoring passages otherwise retrieved by vanilla ColBERTv2. We test this hypothesis by comparing the top- $k$ passages retrieved by vanilla ColBERTv2 to a modified implementation that conducts retrieval using only the centroids and no residuals. We present the results in Figure 3. At $k \in  \{ {10},{100},{1000}\}$ ,the figure plots the average recall of the top- $k$ passages of vanilla ColBERTv2 within the passages retrieved by centroid-only ColBERTv2 at various depths. In other words,we report the fraction of the top- $k$ passages of vanilla ColBERTv2 that appear within the top- ${k}^{\prime }$ passages of centroid-only ColBERTv2,for ${k}^{\prime } \geq  k$ .

实际上，我们假设仅使用质心的检索方法能够找到原本由普通ColBERTv2检索到的高分段落。我们通过比较普通ColBERTv2检索到的前$k$个段落与仅使用质心而不使用残差的改进实现所检索到的段落来验证这一假设。我们将结果展示在图3中。在$k \in  \{ {10},{100},{1000}\}$处，该图绘制了在不同深度下，仅使用质心的ColBERTv2检索到的段落中，普通ColBERTv2前$k$个段落的平均召回率。换句话说，对于${k}^{\prime } \geq  k$，我们报告了普通ColBERTv2前$k$个段落在仅使用质心的ColBERTv2前${k}^{\prime }$个段落中出现的比例。

The results support our hypothesis, both in domain for MS MARCO v1 and out of domain using the LoTTE Pooled (dev) search queries [42]. For instance,if we retrieve ${10} \cdot  k$ passages using only centroids,those ${10} \cdot  k$ passages still contain ${99} + \%$ of the top $k$ passages retrieved by the vanilla ColBERTv2 full pipeline.

实验结果支持了我们的假设，无论是在MS MARCO v1的领域内，还是在使用LoTTE Pooled（开发集）搜索查询[42]的领域外。例如，如果我们仅使用质心检索${10} \cdot  k$个段落，那么这些${10} \cdot  k$个段落仍然包含普通ColBERTv2完整流程检索到的前$k$个段落中的${99} + \%$个。

### 3.4 Not All Centroids Are Important Per Query

### 3.4 并非所有质心对每个查询都重要

We further hypothesize that for a given query a small subset of the passage embedding clusters tend to be far more important than others in determining relevance. If this were in fact the case, then we could prioritize computation over these highly weighted centroids and discard the rest since we know they will not contribute significantly to the final ranking. We test this theory by randomly sampling 15 MS MARCO v1 queries and plotting an empirical CDF of each centroid's maximum relevance score observed across all query tokens, as shown in Figure 4. We do find that there is a small tail of highly weighted centroids whose relevance scores have far higher magnitude than all other centroids. While not shown in Figure 4, we also repeated this experiment with LoTTE pooled queries and found a very similar score distribution.

我们进一步假设，对于给定的查询，在确定相关性时，段落嵌入聚类的一个小子集往往比其他子集重要得多。如果实际情况确实如此，那么我们可以优先对这些权重较高的质心进行计算，并舍弃其余质心，因为我们知道它们对最终排名的贡献不大。我们通过随机抽样15个MS MARCO v1查询，并绘制每个质心在所有查询词元上观察到的最大相关性得分的经验累积分布函数（CDF）来验证这一理论，如图4所示。我们确实发现，存在一小部分权重较高的质心，其相关性得分的量级远高于其他所有质心。虽然图4中未展示，但我们也使用LoTTE汇总查询重复了该实验，发现得分分布非常相似。

## 4 PLAID

## 4 PLAID

Figure 5 illustrates the PLAID scoring pipeline, which consists of multiple consecutive stages for retrieval, filtering, and ranking. The first stage produces an initial candidate set by computing relevance scores for each centroid with respect to the query embeddings. In the intermediate stages, PLAID uses the novel techniques of centroid interaction and centroid pruning to aggressively yet effectively filter the candidate passages. Finally, PLAID ranks the final candidate set using fully reconstructed passage embeddings. We discuss each of these modules in more depth as follows.

图5展示了PLAID评分流程，它由用于检索、过滤和排序的多个连续阶段组成。第一阶段通过计算每个质心相对于查询嵌入的相关性得分来生成初始候选集。在中间阶段，PLAID使用质心交互和质心剪枝的新技术来积极而有效地过滤候选段落。最后，PLAID使用完全重建的段落嵌入对最终候选集进行排序。我们将在下面更深入地讨论这些模块。

### 4.1 Candidate Generation

### 4.1 候选生成

Given the query embedding matrix $Q$ and the list of centroid vectors $C$ in the index,PLAID computes the token-level query-centroid relevance scores ${S}_{c,q}$ as a matrix multiplication:

给定查询嵌入矩阵$Q$和索引中的质心向量列表$C$，PLAID通过矩阵乘法计算词元级别的查询 - 质心相关性得分${S}_{c,q}$：

$$
{S}_{c,q} = C \cdot  {Q}^{T} \tag{2}
$$

and then identifies the passages "close" to the top- $t$ centroids per query token as the initial candidate set. A passage is close to a centroid iff one or more of its tokens are assigned to that centroid by $k$ -means clustering during indexing. This value $t$ is referred to as nprobe in vanilla ColBERTv2 and we retain that terminology in PLAID ColBERTv2.

然后将每个查询词元“接近”前$t$个质心的段落识别为初始候选集。如果一个段落的一个或多个词元在索引过程中通过$k$ - 均值聚类被分配到某个质心，则该段落接近该质心。这个值$t$在普通ColBERTv2中被称为nprobe，我们在PLAID ColBERTv2中保留了这一术语。

The initial candidate generation in PLAID ColBERTv2 differs from the corresponding vanilla ColBERTv2 stage in two key aspects. First, while vanilla ColBERTv2 saves an inverted list mapping centroids to their corresponding embedding IDs, PLAID ColBERTv2 instead structures the inverted list as a map from centroids to the corresponding unique passage IDs. Storing passage IDs is advantageous over storing embedding IDs since there are far fewer passages than embeddings, meaning the inverted list has to store less information overall. This also enables PLAID ColBERTv2 to use 32-bit integers in the inverted list rather than potentially64-bit longs. ${}^{2}$ In practice,this translates to a space savings of ${2.7} \times$ in the MS MARCO v2 [6] inverted list (71 GB to 27 GB, with 140M passages).

PLAID ColBERTv2中的初始候选生成与普通ColBERTv2的相应阶段在两个关键方面有所不同。首先，普通ColBERTv2保存了一个将质心映射到其相应嵌入ID的倒排表，而PLAID ColBERTv2则将倒排表结构化为从质心到相应唯一段落ID的映射。存储段落ID比存储嵌入ID更有优势，因为段落数量远少于嵌入数量，这意味着倒排表总体上需要存储的信息更少。这也使得PLAID ColBERTv2能够在倒排表中使用32位整数，而不是可能的64位长整数。${}^{2}$实际上，这在MS MARCO v2 [6]倒排表中实现了${2.7} \times$的空间节省（从71GB减少到27GB，有1.4亿个段落）。

Second, and relatedly, if the initial candidate set was too large (as specified by the ncandidates hyperparameter) vanilla ColBERTv2 would prune it by scoring and ranking a subset of the candidate embedding vectors-in particular, the embeddings listed within the vanilla mapping from centroid IDs to embedding IDs-with full residual decompression,which is quite costly as we discuss in $\$ {3.2}$ . In contrast, PLAID ColBERTv2 does not impose any limit on the initial candidate size because the subsequent stages can cheaply filter the candidate passages with centroid interaction and pruning.

其次，与此相关的是，如果初始候选集过大（由超参数 ncandidates 指定），原始的 ColBERTv2 会通过对候选嵌入向量的一个子集进行打分和排序来修剪该集合，具体来说，就是对从质心 ID 到嵌入 ID 的原始映射中列出的嵌入进行打分和排序，并进行全残差解压缩，正如我们在 $\$ {3.2}$ 中所讨论的，这一过程成本相当高。相比之下，PLAID ColBERTv2 对初始候选集的大小没有任何限制，因为后续阶段可以通过质心交互和修剪以较低成本过滤候选段落。

### 4.2 Centroid Interaction

### 4.2 质心交互

Centroid interaction cheaply approximates per-passage relevance by substituting each token's embedding vector with its nearest centroid in the standard MaxSim formulation. By applying centroid interaction as an additional filtering stage, the scoring pipeline can skip the expensive embedding reconstruction process for a large fraction of the candidate passages. This results in significantly faster end-to-end retrieval. Intuitively, centroid interaction enables PLAID to emulate traditional bag-of-words retrieval wherein the centroid relevance scores take the role of the term relevance scores used in systems like BM25. However, because of its vector representations (of the query in particular), PLAID computes the centroid relevance scores at query time in contrast to the more traditional pre-computed term relevance scores.

质心交互通过在标准的 MaxSim 公式中用每个词元的嵌入向量的最近质心替代该向量，以较低成本近似计算每个段落的相关性。通过将质心交互作为一个额外的过滤阶段，打分流程可以为大部分候选段落跳过昂贵的嵌入重建过程。这使得端到端检索速度显著加快。直观地说，质心交互使 PLAID 能够模拟传统的词袋检索，其中质心相关性得分起到了像 BM25 等系统中使用的词项相关性得分的作用。然而，由于其向量表示（特别是查询的向量表示），与更传统的预计算词项相关性得分不同，PLAID 在查询时计算质心相关性得分。

The procedure works as follows. Recall that ${S}_{c,q}$ from Equation 2 stores the relevance scores for each centroid with respect to the query tokens. Suppose $I$ is the list of the centroid indices mapped to each of the tokens in the candidate set. Furthermore,let ${S}_{c,q}\left\lbrack  i\right\rbrack$ denote the $i$ -th row of ${S}_{c,q}$ . Then PLAID constructs the centroid-based approximate scores $\widetilde{D}$ as

该过程如下。回顾方程 2 中的 ${S}_{c,q}$ 存储了每个质心相对于查询词元的相关性得分。假设 $I$ 是映射到候选集中每个词元的质心索引列表。此外，设 ${S}_{c,q}\left\lbrack  i\right\rbrack$ 表示 ${S}_{c,q}$ 的第 $i$ 行。然后，PLAID 构建基于质心的近似得分 $\widetilde{D}$ 如下

$$
\widetilde{D} = \left\lbrack  \begin{matrix} {S}_{c,q}\left\lbrack  {I}_{1}\right\rbrack  \\  {S}_{c,q}\left\lbrack  {I}_{2}\right\rbrack  \\  \cdots \\  {S}_{c,q}\left\lbrack  {I}_{\left| \widetilde{D}\right| }\right\rbrack   \end{matrix}\right\rbrack   \tag{3}
$$

---

<!-- Footnote -->

${}^{2}$ This assumes no more than $\leq  {2}^{32}$ (4 billion) passages in the corpus,but this limit is ${30} \times$ larger than even MS MARCO v2 [6].

${}^{2}$ 这假设语料库中的段落不超过 $\leq  {2}^{32}$（40 亿）个，但这个限制比 MS MARCO v2 [6] 还要大得多。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: Centroid Centroids Residuals Scores Decompress Query Reconstructed Approx Passage Relevance Embedding Scores True Relevance MaxSim Scores MaxSim TopK ( $k$ ) PIDs Stage 3: Centroid Stage 4: Final Interaction w/out Ranking with Pruning Decompression Scores Centroids Query Embeddings Prune $\leq  {t}_{\mathrm{c}}$ Approx Scores Relevance Scores MaxSim TopK (nprobe) TopK (ndocs) PIDs PIDS Stage 1: Initial Stage 2: Centroid Candidate Interaction with Generation Pruning -->

<img src="https://cdn.noedgeai.com/0195afc2-9af9-789e-9f2d-8c0aed3d29e7_4.jpg?x=169&y=252&w=1459&h=598&r=0"/>

Figure 5: The PLAID scoring pipeline. The first stage generates an initial set of candidate passages using the centroids. Next the second and third stages leverage centroid pruning and centroid interaction respectively to refine the candidate set. Then the last stage performs full residual decompression to obtain the final passage ranking. We use the hyperparameter ndocs to specify the number of candidates returned by Stage 2, and in our experiments we have Stage 3 output $\frac{\text{ ndocs }}{4}$ passages.

图 5：PLAID 打分流程。第一阶段使用质心生成初始候选段落集。接下来，第二和第三阶段分别利用质心修剪和质心交互来细化候选集。然后，最后一个阶段进行全残差解压缩以获得最终的段落排名。我们使用超参数 ndocs 来指定第二阶段返回的候选数量，在我们的实验中，第三阶段输出 $\frac{\text{ ndocs }}{4}$ 个段落。

<!-- Media -->

Then to rank the candidate passages using $\widetilde{D}$ ,PLAID computes the MaxSim scores ${S}_{\widetilde{D}}$ as

然后，为了使用 $\widetilde{D}$ 对候选段落进行排名，PLAID 计算 MaxSim 得分 ${S}_{\widetilde{D}}$ 如下

$$
{S}_{\widetilde{D}} = \mathop{\sum }\limits_{i}^{\left| Q\right| }\mathop{\max }\limits_{{j = 1}}^{\left| \widetilde{D}\right| }{\widetilde{D}}_{i,j} \tag{4}
$$

The top $k$ most relevant passages drawn from ${S}_{\widetilde{D}}$ serve as the filtered candidate passage set.

从 ${S}_{\widetilde{D}}$ 中选出的前 $k$ 个最相关的段落作为过滤后的候选段落集。

PLAID includes optimized kernels to efficiently deploy centroid interaction (and more generally MaxSim operations); we discuss these in $§{4.5}$ .

PLAID 包含优化的内核，以有效地部署质心交互（更一般地说，是 MaxSim 操作）；我们在 $§{4.5}$ 中讨论这些内容。

### 4.3 Centroid Pruning

### 4.3 质心修剪

As an additional optimization, PLAID leverages the observation from §3.3 to first prune low-magnitude centroid scores before constructing $\widetilde{D}$ . In this filtering phase PLAID will only score tokens whose maximum corresponding centroid score meets the given threshold ${t}_{cs}$ . Concretely, $\widetilde{D}$ will only be comprised of tokens whose corresponding centroid (suppose centroid $i$ ) meets the following condition:

作为一项额外的优化，PLAID 利用第 3.3 节的观察结果，在构建 $\widetilde{D}$ 之前先修剪低幅度的质心得分。在这个过滤阶段，PLAID 只会对其对应质心的最大得分满足给定阈值 ${t}_{cs}$ 的词元进行打分。具体来说，$\widetilde{D}$ 将仅由其对应质心（假设为质心 $i$）满足以下条件的词元组成：

$$
\mathop{\max }\limits_{{j = 1}}^{\left| Q\right| }{S}_{c,{q}_{i,j}} \geq  {t}_{cs} \tag{5}
$$

We introduce the hyperparameter ndocs to refer to the number of candidate documents selected by Stage 2. We then found empirically that choosing $\frac{\text{ ndocs }}{4}$ candidates from Stage 3 produced good results; we use this heuristic for all the results presented in §5.

我们引入超参数 ndocs 来表示第二阶段选择的候选文档数量。然后我们通过实验发现，从第三阶段选择 $\frac{\text{ ndocs }}{4}$ 个候选文档能得到较好的结果；我们在第 5 节呈现的所有结果中都使用了这一启发式方法。

### 4.4 Scoring

### 4.4 打分

As in vanilla ColBERTv2, PLAID will reconstruct the original embed-dings of the final candidate passage set via residual decompression and rank these using MaxSim. Let $D$ be the reconstructed embedding vectors for the final candidate set after decompression. Then the final scores ${S}_{q,d}$ are computed using Equation 1.

与原始的 ColBERTv2 一样，PLAID 将通过残差解压缩重建最终候选段落集的原始嵌入，并使用 MaxSim 对这些段落进行排名。设 $D$ 是解压缩后最终候选集的重建嵌入向量。然后，使用方程 1 计算最终得分 ${S}_{q,d}$。

Section §4.5 discusses fast kernels for accelerating the MaxSim and decompression operations.

第 4.5 节讨论用于加速 MaxSim 和分解操作的快速内核。

### 4.5 Fast Kernels: Padding-Free MaxSim & Optimized Decompression

### 4.5 快速内核：无填充 MaxSim 和优化的解压缩

Figure 2a shows that index lookup operations are a large source of overhead for vanilla ColBERTv2. One reason these lookups are expensive is that they require reshaping and padding the $2\mathrm{D}$ index tensors with an extra dimension representing the maximum passage length. The resulting $3\mathrm{D}$ tensors facilitate batched MaxSim operations over ragged lists of token vectors. To avoid this padding, we instead implement custom C++ code that directly computes the MaxSim scores over the packed 2D index tensors (i.e., one where many 2D sub-tensors of various lengths are concatenated along the same dimension). Our kernel loops over each passage's corresponding token vectors to compute the per-passage maximum scores with respect to each query token and then sums the per-passage maximum scores across all query tokens. This design is PLAID, May 2022, Preprint trivial to parallelize across passages,and also enables $O\left( \left| Q\right| \right)$ per-thread memory usage by allocating a single output vector to store the maximum scores per query token and repeatedly updating this vector in-place. In contrast, the padding-based approach requires $O\left( {\left| D\right|  \cdot  \left| Q\right| }\right)$ space. We have incorporated this design into optimized implementations of centroid interaction as well as the final MaxSim operation (stage 4 in Figure 5). PLAID only implements these kernels for CPU execution. Adding corresponding GPU kernels remains future work.

图2a显示，索引查找操作是普通ColBERTv2开销的一大来源。这些查找操作成本高昂的一个原因是，它们需要对$2\mathrm{D}$索引张量进行重塑和填充，增加一个表示最大段落长度的额外维度。由此得到的$3\mathrm{D}$张量便于对不规则的词向量列表进行批量最大相似度（MaxSim）操作。为避免这种填充，我们改为实现自定义C++代码，直接在打包的二维索引张量（即多个不同长度的二维子张量沿同一维度拼接而成的张量）上计算最大相似度分数。我们的内核会遍历每个段落对应的词向量，计算每个段落相对于每个查询词的最大分数，然后将所有查询词对应的段落最大分数相加。这种设计于2022年5月预印本发表在PLAID上，在不同段落间易于并行化，并且通过分配单个输出向量来存储每个查询词的最大分数并就地重复更新该向量，实现了$O\left( \left| Q\right| \right)$每线程内存使用。相比之下，基于填充的方法需要$O\left( {\left| D\right|  \cdot  \left| Q\right| }\right)$空间。我们已将此设计融入质心交互以及最终最大相似度操作（图5中的阶段4）的优化实现中。PLAID仅实现了这些用于CPU执行的内核。添加相应的GPU内核仍是未来的工作。

<!-- Media -->

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">#Passages</td><td rowspan="2">#Tokens</td><td rowspan="2">#Queries</td><td colspan="2">ColBERTv2 Index Size (GiB)</td></tr><tr><td>Vanilla</td><td>PLAID</td></tr><tr><td>MS MARCO v1 [36]</td><td>8.8M</td><td>597.9M</td><td>6980</td><td>24.6</td><td>21.6</td></tr><tr><td>Wikipedia [18]</td><td>21.0M</td><td>2.6B</td><td>8757</td><td>105.2</td><td>92.0</td></tr><tr><td>LoTTE pooled [42]</td><td>2.4M</td><td>339.4M</td><td>2931</td><td>14.0</td><td>12.3</td></tr><tr><td>MS MARCO v2 [6]</td><td>138.4M</td><td>9.4B</td><td>3903</td><td>246.0</td><td>202.2</td></tr></table>

<table><tbody><tr><td rowspan="2">数据集</td><td rowspan="2">#段落数</td><td rowspan="2">#Token数</td><td rowspan="2">#查询数</td><td colspan="2">ColBERTv2索引大小（吉字节）</td></tr><tr><td>普通版</td><td>PLAID</td></tr><tr><td>MS MARCO v1 [36]</td><td>8.8M</td><td>597.9M</td><td>6980</td><td>24.6</td><td>21.6</td></tr><tr><td>维基百科 [18]</td><td>21.0M</td><td>2.6B</td><td>8757</td><td>105.2</td><td>92.0</td></tr><tr><td>LoTTE合并数据集 [42]</td><td>2.4M</td><td>339.4M</td><td>2931</td><td>14.0</td><td>12.3</td></tr><tr><td>MS MARCO v2 [6]</td><td>138.4M</td><td>9.4B</td><td>3903</td><td>246.0</td><td>202.2</td></tr></tbody></table>

Table 1: List of benchmarks used for evaluation with relevant statistics.

表1：用于评估的基准测试列表及相关统计信息。

<!-- Media -->

ColBERTv2's residual decompression scheme computes a list of centroid vectors,determines a fixed set of ${2}^{b}$ possible deltas from these centroids, and then stores the index into the set of deltas corresponding to each embedding vector. In particular, each compressed 8-bit value stores $\frac{8}{b}$ indices in the range $\left\lbrack  {0,{2}^{b}}\right)$ . Col-BERTv2 incurs significant overhead due to residual decompression, as shown in Figure 2a. This is partially due to the naïve decompression implementation, which required explicitly unpacking bits from the compressed representation and performing expensive bit shift and sum operations to recover the original values. Instead, PLAID pre-computes all ${2}^{8}$ possible lists of indices encoded by an 8-bit packed value. These outputs are stored in a lookup table so that the decompression function can simply retrieve the indices from the table rather than manually unpacking the bits. We include optimized implementations of this lookup-based decompression for both CPU and GPU execution. The GPU implementation uses a custom CUDA kernel that allocates a separate thread to decompress each individual byte in the compressed residual tensor (the thread block size is computed as $\frac{b \cdot  d}{8}$ for $d$ -dimensional embedding vectors). The CPU implementation instead parallelizes decompression at the granularity of individual passages.

ColBERTv2的残差解压缩方案会计算一个质心向量列表，从这些质心确定一组固定的 ${2}^{b}$ 个可能的差值，然后存储每个嵌入向量对应的差值集合的索引。具体而言，每个压缩的8位值存储范围在 $\left\lbrack  {0,{2}^{b}}\right)$ 内的 $\frac{8}{b}$ 个索引。如图2a所示，由于残差解压缩，Col - BERTv2会产生显著的开销。这部分是由于简单的解压缩实现方式，它需要从压缩表示中显式地解包位，并执行昂贵的位移和求和操作来恢复原始值。相反，PLAID预先计算了由一个8位打包值编码的所有 ${2}^{8}$ 个可能的索引列表。这些输出存储在一个查找表中，这样解压缩函数就可以直接从表中检索索引，而不是手动解包位。我们为CPU和GPU执行都提供了这种基于查找的解压缩的优化实现。GPU实现使用了一个自定义的CUDA内核，该内核为压缩残差张量中的每个单独字节分配一个单独的线程进行解压缩（对于 $d$ 维嵌入向量，线程块大小计算为 $\frac{b \cdot  d}{8}$ ）。而CPU实现则以单个段落为粒度并行化解压缩。

## 5 EVALUATION

## 5 评估

Our evaluation seeks to answer the following research questions:

我们的评估旨在回答以下研究问题：

(1) How does PLAID affect end-to-end latency and retrieval quality across IR benchmarks? (§5.2)

(1) 在信息检索（IR）基准测试中，PLAID对端到端延迟和检索质量有何影响？（§5.2）

(2) How much do each of PLAID's optimizations contribute to the performance speedups? (§5.3)

(2) PLAID的各项优化对性能加速的贡献有多大？（§5.3）

(3) How well does PLAID scale with respect to the corpus size and the parallelism degree? (§5.4)

(3) PLAID在语料库大小和并行度方面的扩展性如何？（§5.4）

### 5.1 Setup

### 5.1 实验设置

PLAID Implementation. The PLAID engine subsumes centroid interaction as well as optimizations for residual decompression. We implement PLAID modularly as an extension to ColBERTv2's

PLAID实现。PLAID引擎包含质心交互以及残差解压缩的优化。我们将PLAID作为ColBERTv2的扩展进行模块化实现

Keshav Santhanam*, Omar Khattab*, Christopher Potts, and Matei Zaharia PyTorch-based implementation, particularly its search components. For CPU execution, we implement the centroid interaction and decompression operations entirely in multithreaded C++ code. For GPUs, we implement centroid interaction in PyTorch and provide a CUDA kernel for fast decompression. Overall, PLAID constitutes roughly 300 lines of additional Python code and 700 lines of C++.

凯沙夫·桑塔南姆（Keshav Santhanam）*、奥马尔·哈塔卜（Omar Khattab）*、克里斯托弗·波茨（Christopher Potts）和马泰·扎哈里亚（Matei Zaharia）基于PyTorch的实现，特别是其搜索组件。对于CPU执行，我们完全用多线程C++代码实现质心交互和解压缩操作。对于GPU，我们在PyTorch中实现质心交互，并提供一个用于快速解压缩的CUDA内核。总体而言，PLAID大约包含300行额外的Python代码和700行C++代码。

<!-- Media -->

<table><tr><td>$k$</td><td>nprobe</td><td>${t}_{cs}$</td><td>ndocs</td></tr><tr><td>10</td><td>1</td><td>0.5</td><td>256</td></tr><tr><td>100</td><td>2</td><td>0.45</td><td>1024</td></tr><tr><td>1000</td><td>4</td><td>0.4</td><td>4096</td></tr></table>

<table><tbody><tr><td>$k$</td><td>网络探针（nprobe）</td><td>${t}_{cs}$</td><td>文档数量（ndocs）</td></tr><tr><td>10</td><td>1</td><td>0.5</td><td>256</td></tr><tr><td>100</td><td>2</td><td>0.45</td><td>1024</td></tr><tr><td>1000</td><td>4</td><td>0.4</td><td>4096</td></tr></tbody></table>

Table 2: PLAID hyperparameter configuration.

表2：PLAID超参数配置。

<!-- Media -->

Datasets. Our evaluation includes results from four different IR benchmarks, as listed in Table 1. We perform in-domain evaluation on the MS MARCO v1 and Wikipedia Open QA benchmarks, with retrievers trained specifically for these tasks, and out-of-domain evaluation on the StackExchange-based LoTTE Santhanam et al. [42] and the TREC 2021 Deep Learning Track [6] MS MARCO v2 benchmarks, with the ColBERTv2 retriever [42] trained on MS MARCO v1. For evaluation on Wikipedia we use the December 2018 dump [18] with queries from the NaturalQuestions (NQ) dataset [23]. Our LoTTE [42] evaluation uses the "pooled" dev dataset with "search"- style queries. For MS MARCO v2, we use the augmented passage version of the data [2] and include passage titles while ignoring headings. As we evaluate several configurations of the models, all of our evaluation is performed using development set queries.

数据集。我们的评估涵盖了四个不同信息检索（IR）基准的结果，如表1所示。我们在MS MARCO v1和维基百科开放问答（Wikipedia Open QA）基准上进行领域内评估，使用专门为这些任务训练的检索器；在基于StackExchange的LoTTE（Santhanam等人 [42]）和TREC 2021深度学习赛道（TREC 2021 Deep Learning Track [6]）的MS MARCO v2基准上进行领域外评估，使用在MS MARCO v1上训练的ColBERTv2检索器 [42]。在维基百科评估中，我们使用2018年12月的转储数据 [18]，查询来自自然问题（NaturalQuestions，NQ）数据集 [23]。我们对LoTTE [42] 的评估使用“合并”开发数据集和“搜索”风格的查询。对于MS MARCO v2，我们使用数据的增强段落版本 [2]，并包含段落标题，同时忽略标题。由于我们评估了模型的几种配置，所有评估均使用开发集查询进行。

Systems and hyperparameters. We report results for several systems for end-to-end results: vanilla ColBERTv2 and PLAID Col-BERTv2 as well as ColBERT (v1) [22], BM25 [41], SPLADEv2 [10], and DPR [18]. For vanilla ColBERTv2, we use the specific hyper-parameters reported in the ColBERTv2 paper for each benchmark dataset. We indicate these in the result tables with $\mathrm{p}$ (nprobe) and $\mathrm{c}$ (ncandidates). For PLAID ColBERTv2, we evaluate three different settings: $k = {10},k = {100}$ ,and $k = {1000}$ . The $k$ parameter controls the final number of scored documents as well as the retrieval hy-perparameters described in §4. Table 2 lists these hyperparameter configurations for each $k$ setting. We find empirically that ranking $\frac{\text{ ndocs }}{4}$ documents for the final scoring stage produces strong results. For both vanilla ColBERTv2 and PLAID ColBERTv2, we compress all datasets to 2 bits per dimension, with the exception of MS MARCO v2 where we compress to 1 bit.

系统和超参数。我们报告了几个系统的端到端结果：原生ColBERTv2、PLAID Col - BERTv2以及ColBERT（v1） [22]、BM25 [41]、SPLADEv2 [10] 和DPR [18]。对于原生ColBERTv2，我们使用ColBERTv2论文中针对每个基准数据集报告的特定超参数。我们在结果表中用 $\mathrm{p}$（探测数，nprobe）和 $\mathrm{c}$（候选数，ncandidates）表示这些参数。对于PLAID ColBERTv2，我们评估三种不同的设置：$k = {10},k = {100}$ 和 $k = {1000}$。$k$ 参数控制最终评分文档的数量以及第4节中描述的检索超参数。表2列出了每个 $k$ 设置的这些超参数配置。我们通过实验发现，在最终评分阶段对 $\frac{\text{ ndocs }}{4}$ 个文档进行排序可产生良好的结果。对于原生ColBERTv2和PLAID ColBERTv2，我们将所有数据集压缩为每维2位，但MS MARCO v2压缩为每维1位。

Hardware. We conduct all experiments on servers with 28 Intel Xeon Gold 6132 2.6 GHz CPU cores (2 threads per core for a total of 56 threads) and 4 NVIDIA TITAN V GPUs each. Every server has two NUMA sockets with roughly ${92}\mathrm{\;{ns}}$ intra-socket memory latency, 142 ns inter-socket memory latency, 72 GBps intra-socket memory bandwidth, and 33 GBps inter-socket memory bandwidth. Each TITAN V GPU has 12 GB of high-bandwidth memory.

硬件。我们在配备28个英特尔至强金牌6132 2.6 GHz CPU核心（每个核心2个线程，共56个线程）和4个英伟达TITAN V GPU的服务器上进行所有实验。每台服务器有两个非统一内存访问（NUMA）插槽，插槽内内存延迟约为 ${92}\mathrm{\;{ns}}$，插槽间内存延迟为142纳秒，插槽内内存带宽为72 GBps，插槽间内存带宽为33 GBps。每个TITAN V GPU有12 GB的高带宽内存。

Latency measurements. When measuring latency for end-to-end results, we compute the average latency of all queries (see Table 1 for query totals), and then report the minimum average latency across 3 trials. For other results we describe the specific measurement procedure in the relevant section. We discard the query encoding latency for neural models (ColBERTv1 [22], vanilla Col-BERTv2 [42], PLAID ColBERTv2, and SPLADEv2 [10]) following Mackenzie et al. [30]; prior work has shown that the cost of running the BERT model can be made negligible with standard techniques such as quantization, distillation, etc. [4]. We measure latency on an otherwise idle machine. We prepend commands with numactl --membind 0 to ensure intra-socket I/O operations. We do not do this for MS MARCO v2, since its large index may require both NUMA nodes. For GPU results we allow full usage of all 56 threads, but for CPU-only results we restrict usage to either 1 or 8 threads using torch. set_num_threads. For non-ColBERT systems we use the single-threaded latency numbers reported by Mackenzie et al. [30]. Note that these numbers were measured on a different hardware setup and using a different implementation and are therefore simply meant to establish PLAID ColBERTv2's competitive performance rather than serving as absolute comparisons.

延迟测量。在测量端到端结果的延迟时，我们计算所有查询的平均延迟（查询总数见表1），然后报告3次试验中的最小平均延迟。对于其他结果，我们在相关部分描述具体的测量过程。按照Mackenzie等人 [30] 的方法，我们忽略神经模型（ColBERTv1 [22]、原生Col - BERTv2 [42]、PLAID ColBERTv2和SPLADEv2 [10]）的查询编码延迟；先前的工作表明，使用量化、蒸馏等标准技术可以使运行BERT模型的成本忽略不计 [4]。我们在闲置的机器上测量延迟。我们在命令前加上numactl --membind 0以确保插槽内I/O操作。对于MS MARCO v2，我们不这样做，因为其大型索引可能需要两个NUMA节点。对于GPU结果，我们允许使用全部56个线程，但对于仅使用CPU的结果，我们使用torch.set_num_threads将线程使用限制为1个或8个。对于非ColBERT系统，我们使用Mackenzie等人 [30] 报告的单线程延迟数据。请注意，这些数据是在不同的硬件设置和不同的实现下测量的，因此仅用于证明PLAID ColBERTv2的竞争性能，而不是作为绝对比较。

<!-- Media -->

### 5.2 End-to-end Results

### 5.2 端到端结果

<table><tr><td rowspan="2">System</td><td>MRR@10</td><td rowspan="2">10 R@100 R@1k</td><td rowspan="2"/><td colspan="3">Latency (ms)</td></tr><tr><td/><td>1-CPU 8-CPU GPU</td><td/><td/></tr><tr><td>BM25 (PISA [34]; $k = {1000}$ )</td><td>${18.7}^{ * }$</td><td>-</td><td>-</td><td>${8.3}^{ * }$</td><td>-</td><td>-</td></tr><tr><td>SPLADEv2 (PISA; $k = {1000}$ )</td><td>${36.8}^{ * }$</td><td>-</td><td>${97.9}^{ * }$</td><td>${220.3}^{ * }$</td><td>-</td><td>-</td></tr><tr><td>ColBERTv1</td><td>36.1</td><td>87.3</td><td>95.2</td><td>-</td><td>-</td><td>54.3</td></tr><tr><td>Vanilla ColBERTv2 (p=2,c=2 ${}^{13}$ )</td><td>39.7</td><td>90.4</td><td>96.6</td><td>3485.1</td><td>921.8</td><td>53.4</td></tr><tr><td>Vanilla ColBERTv2 (p=4,c=2 ${}^{16}$ )</td><td>39.7</td><td>91.4</td><td>98.3</td><td>-</td><td>4568.5</td><td>259.6</td></tr><tr><td>PLAID ColBERTv2 $\left( {k = {10}}\right)$</td><td>39.4</td><td>-</td><td>-</td><td>185.5</td><td>31.5</td><td>11.5</td></tr><tr><td>PLAID ColBERTv2 $\left( {k = {100}}\right)$</td><td>39.8</td><td>90.6</td><td>-</td><td>222.3</td><td>52.9</td><td>20.2</td></tr><tr><td>PLAID ColBERTv2 $\left( {k = {1000}}\right)$</td><td>39.8</td><td>91.3</td><td>97.5</td><td>352.3</td><td>101.3</td><td>38.4</td></tr></table>

<table><tbody><tr><td rowspan="2">系统</td><td>前10的平均倒数排名（MRR@10）</td><td rowspan="2">前10、前100、前1000的召回率（10 R@100 R@1k）</td><td rowspan="2"></td><td colspan="3">延迟（毫秒）</td></tr><tr><td></td><td>1个CPU、8个CPU、GPU</td><td></td><td></td></tr><tr><td>BM25算法（PISA [34]; $k = {1000}$ ）</td><td>${18.7}^{ * }$</td><td>-</td><td>-</td><td>${8.3}^{ * }$</td><td>-</td><td>-</td></tr><tr><td>SPLADEv2模型（PISA; $k = {1000}$ ）</td><td>${36.8}^{ * }$</td><td>-</td><td>${97.9}^{ * }$</td><td>${220.3}^{ * }$</td><td>-</td><td>-</td></tr><tr><td>ColBERTv1模型</td><td>36.1</td><td>87.3</td><td>95.2</td><td>-</td><td>-</td><td>54.3</td></tr><tr><td>普通ColBERTv2模型（p=2,c=2 ${}^{13}$ ）</td><td>39.7</td><td>90.4</td><td>96.6</td><td>3485.1</td><td>921.8</td><td>53.4</td></tr><tr><td>普通ColBERTv2模型（p=4,c=2 ${}^{16}$ ）</td><td>39.7</td><td>91.4</td><td>98.3</td><td>-</td><td>4568.5</td><td>259.6</td></tr><tr><td>PLAID ColBERTv2模型 $\left( {k = {10}}\right)$</td><td>39.4</td><td>-</td><td>-</td><td>185.5</td><td>31.5</td><td>11.5</td></tr><tr><td>PLAID ColBERTv2模型 $\left( {k = {100}}\right)$</td><td>39.8</td><td>90.6</td><td>-</td><td>222.3</td><td>52.9</td><td>20.2</td></tr><tr><td>PLAID ColBERTv2模型 $\left( {k = {1000}}\right)$</td><td>39.8</td><td>91.3</td><td>97.5</td><td>352.3</td><td>101.3</td><td>38.4</td></tr></tbody></table>

Table 3: End-to-end in-domain evaluation on the MS MARCO v1 benchmark. Numbers marked with an asterisk are copied from Formal et al. [11] for SPLADEv2 quality and Mackenzie et al. [30] for latencies.

表3：在MS MARCO v1基准测试上的端到端领域内评估。标有星号的数字，质量数据取自福尔马尔等人 [11] 关于SPLADEv2的研究，延迟数据取自麦肯齐等人 [30] 的研究。

<table><tr><td rowspan="2">System</td><td rowspan="2">Success@5</td><td rowspan="2">Success@100</td><td colspan="2">Latency (ms)</td></tr><tr><td>CPU (8) GPU</td><td/></tr><tr><td>DPR</td><td>66.8</td><td>85.0</td><td>-</td><td>-</td></tr><tr><td>ColBERT-QA Retrieval (uncompressed)</td><td>75.3</td><td>89.2</td><td>-</td><td>-</td></tr><tr><td colspan="5">ColBERT-QA [21] Retriever with ColBERTv2 [42] residual compression</td></tr><tr><td>Vanilla ColBERT-QA Retrieval $\left( {\mathrm{p} = 4,\mathrm{c} = {2}^{15}}\right)$</td><td>74.3</td><td>89.0</td><td>5077.9</td><td>204.1</td></tr><tr><td>PLAID ColBERT-QA Retrieval $\left( {k = {10}}\right)$</td><td>73.3</td><td>-</td><td>67.1</td><td>13.6</td></tr><tr><td>PLAID ColBERT-QA Retrieval $\left( {k = {100}}\right)$</td><td>74.1</td><td>88.0</td><td>120.1</td><td>26.9</td></tr><tr><td>PLAID ColBERT-QA Retrieval $\left( {k = {1000}}\right)$</td><td>74.4</td><td>88.9</td><td>228.4</td><td>55.3</td></tr></table>

<table><tbody><tr><td rowspan="2">系统</td><td rowspan="2">成功率@5</td><td rowspan="2">成功率@100</td><td colspan="2">延迟（毫秒）</td></tr><tr><td>中央处理器（8） 图形处理器</td><td></td></tr><tr><td>动态像素比（DPR）</td><td>66.8</td><td>85.0</td><td>-</td><td>-</td></tr><tr><td>ColBERT问答检索（未压缩）</td><td>75.3</td><td>89.2</td><td>-</td><td>-</td></tr><tr><td colspan="5">采用ColBERTv2 [42]残差压缩的ColBERT问答[21]检索器</td></tr><tr><td>普通ColBERT问答检索 $\left( {\mathrm{p} = 4,\mathrm{c} = {2}^{15}}\right)$</td><td>74.3</td><td>89.0</td><td>5077.9</td><td>204.1</td></tr><tr><td>PLAID ColBERT问答检索 $\left( {k = {10}}\right)$</td><td>73.3</td><td>-</td><td>67.1</td><td>13.6</td></tr><tr><td>PLAID ColBERT问答检索 $\left( {k = {100}}\right)$</td><td>74.1</td><td>88.0</td><td>120.1</td><td>26.9</td></tr><tr><td>PLAID ColBERT问答检索 $\left( {k = {1000}}\right)$</td><td>74.4</td><td>88.9</td><td>228.4</td><td>55.3</td></tr></tbody></table>

Table 4: End-to-end in-domain retrieval evaluation on the Wikipedia open-domain question answering benchmark. We use the NQ checkpoint of ColBERT-QA [21], and apply ColBERTv2 compression. We compare vanilla ColBERTv2 retrieval against PLAID ColBERTv2 retrieval. DPR results from Karpukhin et al. [18]. We refer to Khattab et al. [21] for details on OpenQA retrieval evaluation.

表4：维基百科开放域问答基准上的端到端领域内检索评估。我们使用ColBERT - QA [21]的自然问答（NQ）检查点，并应用ColBERTv2压缩。我们将普通ColBERTv2检索与PLAID ColBERTv2检索进行比较。密集段落检索（DPR）结果来自Karpukhin等人[18]。有关开放域问答（OpenQA）检索评估的详细信息，请参考Khattab等人[21]。

<table><tr><td rowspan="2">System</td><td rowspan="2">Success@5</td><td rowspan="2">Success@100</td><td colspan="2">Latency (ms)</td></tr><tr><td>CPU (8)</td><td>GPU</td></tr><tr><td>BM25</td><td>${47.8}^{ * }$</td><td>77.6*</td><td>-</td><td>-</td></tr><tr><td>SPLADEv2</td><td>${67.0}^{ * }$</td><td>${89.0}^{ * }$</td><td>-</td><td>-</td></tr><tr><td>Vanilla ColBERTv2 (p=2, c=2 ${}^{13}$ )</td><td>69.3</td><td>90.3</td><td>1508.4</td><td>66.9</td></tr><tr><td>PLAID ColBERTv2 $\left( {k = {10}}\right)$</td><td>69.1</td><td>-</td><td>35.5</td><td>9.2</td></tr><tr><td>PLAID ColBERTv2 $\left( {k = {100}}\right)$</td><td>69.4</td><td>89.9</td><td>64.8</td><td>17.4</td></tr><tr><td>PLAID ColBERTv2 $\left( {k = {1000}}\right)$</td><td>69.6</td><td>90.5</td><td>163.1</td><td>27.3</td></tr></table>

<table><tbody><tr><td rowspan="2">系统</td><td rowspan="2">成功率@5</td><td rowspan="2">成功率@100</td><td colspan="2">延迟（毫秒）</td></tr><tr><td>中央处理器（8）</td><td>图形处理器</td></tr><tr><td>二元独立模型（BM25）</td><td>${47.8}^{ * }$</td><td>77.6*</td><td>-</td><td>-</td></tr><tr><td>SPLADEv2模型</td><td>${67.0}^{ * }$</td><td>${89.0}^{ * }$</td><td>-</td><td>-</td></tr><tr><td>原始ColBERTv2模型（p=2，c=2 ${}^{13}$ ）</td><td>69.3</td><td>90.3</td><td>1508.4</td><td>66.9</td></tr><tr><td>PLAID ColBERTv2模型 $\left( {k = {10}}\right)$</td><td>69.1</td><td>-</td><td>35.5</td><td>9.2</td></tr><tr><td>PLAID ColBERTv2模型 $\left( {k = {100}}\right)$</td><td>69.4</td><td>89.9</td><td>64.8</td><td>17.4</td></tr><tr><td>PLAID ColBERTv2模型 $\left( {k = {1000}}\right)$</td><td>69.6</td><td>90.5</td><td>163.1</td><td>27.3</td></tr></tbody></table>

Table 5: End-to-end out-of-domain evaluation on the (dev) pooled dataset of the LoTTE benchmark. Numbers marked with an asterisk were taken from Santhanam et al. [42].

表5：在LoTTE基准测试的（开发）合并数据集上进行的端到端域外评估。标有星号的数字取自Santhanam等人的文献[42]。

<table><tr><td rowspan="2">System</td><td rowspan="2">MRR@100</td><td rowspan="2">R@100</td><td rowspan="2">R@1k</td><td colspan="2">Latency (ms)</td></tr><tr><td>8-CPU</td><td>GPU</td></tr><tr><td>BM25 (Anserini [50]; Augmented)</td><td>8.7</td><td>40.3</td><td>69.3</td><td>-</td><td>-</td></tr><tr><td>Vanilla ColBERTv2 ( $\mathrm{p} = 4,\mathrm{c} = {2}^{16}$ )</td><td>18.0</td><td>68.2</td><td>88.1</td><td>5228.5</td><td>OOM</td></tr><tr><td>PLAID ColBERTv2 $\left( {k = {10}}\right)$</td><td>-</td><td>-</td><td>-</td><td>136.4</td><td>47.1</td></tr><tr><td>PLAID ColBERTv2 $\left( {k = {100}}\right)$</td><td>17.9</td><td>67.0</td><td>-</td><td>181.9</td><td>96.1</td></tr><tr><td>PLAID ColBERTv2 $\left( {k = {1000}}\right)$</td><td>18.0</td><td>68.4</td><td>85.7</td><td>251.3</td><td>OOM</td></tr></table>

<table><tbody><tr><td rowspan="2">系统</td><td rowspan="2">前100的平均倒数排名（MRR@100）</td><td rowspan="2">R@100</td><td rowspan="2">前1000的召回率（R@1k）</td><td colspan="2">延迟（毫秒）</td></tr><tr><td>8核CPU</td><td>图形处理器（GPU）</td></tr><tr><td>BM25算法（Anserini [50]；增强版）</td><td>8.7</td><td>40.3</td><td>69.3</td><td>-</td><td>-</td></tr><tr><td>普通ColBERTv2模型 ( $\mathrm{p} = 4,\mathrm{c} = {2}^{16}$ )</td><td>18.0</td><td>68.2</td><td>88.1</td><td>5228.5</td><td>内存不足（OOM）</td></tr><tr><td>PLAID ColBERTv2模型 $\left( {k = {10}}\right)$</td><td>-</td><td>-</td><td>-</td><td>136.4</td><td>47.1</td></tr><tr><td>PLAID ColBERTv2模型 $\left( {k = {100}}\right)$</td><td>17.9</td><td>67.0</td><td>-</td><td>181.9</td><td>96.1</td></tr><tr><td>PLAID ColBERTv2模型 $\left( {k = {1000}}\right)$</td><td>18.0</td><td>68.4</td><td>85.7</td><td>251.3</td><td>内存不足（OOM）</td></tr></tbody></table>

Table 6: End-to-end out-of-domain evaluation on the MS MARCO v2 benchmark. BM25 results from [3].

表6：在MS MARCO v2基准测试上的端到端域外评估。BM25的结果来自[3]。

<!-- Media -->

Table 3 presents in-domain results for the MS MARCO v1 benchmark. We observe that in the most conservative setting $\left( {k = {1000}}\right)$ , PLAID ColBERTv2 is able to match the MRR@10 and Recall@100 achieved by vanilla ColBERTv2 while delivering speedups of ${6.8} \times$ on GPU and ${45} \times$ on CPU. For some minimal reduction in quality, PLAID ColBERTv2 can further increase the speedups over vanilla ColBERTv2 to 12.9-22.6 $\times$ on GPU and 86.4-145 $\times$ on CPU. PLAID ColBERTv2 also achieves competitive latency compared to other systems (within ${1.6} \times$ of SPLADEv2) while outperforming them on retrieval quality.

表3展示了MS MARCO v1基准测试的域内结果。我们观察到，在最保守的设置$\left( {k = {1000}}\right)$下，PLAID ColBERTv2能够达到与原始ColBERTv2相同的MRR@10和Recall@100，同时在GPU上实现${6.8} \times$的加速，在CPU上实现${45} \times$的加速。为了使质量有一定程度的降低，PLAID ColBERTv2可以进一步将相对于原始ColBERTv2的加速比提高到在GPU上为12.9 - 22.6$\times$，在CPU上为86.4 - 145$\times$。与其他系统相比，PLAID ColBERTv2还实现了有竞争力的延迟（在SPLADEv2的${1.6} \times$范围内），同时在检索质量上优于它们。

We observe a similar trend with in-domain evaluation on the Wikipedia OpenQA benchmark as shown in Table 4. PLAID Col-BERTv2 achieves speedups of ${3.7} \times$ on GPU and ${22} \times$ on CPU with no quality loss compared to vanilla ColBERTv2, and speedups of ${7.6} - {15} \times$ on GPU and 42.3 $- {75.7} \times$ on CPU with minimal quality loss.

如表4所示，我们在维基百科开放问答（Wikipedia OpenQA）基准测试的域内评估中观察到了类似的趋势。与原始ColBERTv2相比，PLAID Col - BERTv2在GPU上实现了${3.7} \times$的加速，在CPU上实现了${22} \times$的加速，且没有质量损失；在质量损失极小的情况下，在GPU上实现了${7.6} - {15} \times$的加速，在CPU上实现了42.3$- {75.7} \times$的加速。

We confirm PLAID works well in out-of-domain settings, as well, as demonstrated by our results on the LoTTE "pooled" dataset. We see in Table 5 that PLAID ColBERTv2 outperforms vanilla Col-BERTv2 by ${2.5} \times$ on GPU and ${9.2} \times$ on CPU with $k = {1000}$ ; furthermore, this setting actually improves quality compared to vanilla ColBERTv2. With some quality loss PLAID ColBERTv2 can achieve speedups of ${3.8} - {7.3} \times$ on GPU and ${23.2} - {42.5} \times$ on CPU. Note that the CPU latencies achieved on LoTTE with PLAID ColBERTv2 are larger than those achieved on MS MARCO v1 because the average LoTTE passage length is roughly $2 \times$ that of MS MARCO v1.

我们的LoTTE“合并”数据集上的结果表明，PLAID在域外设置中也能很好地工作。从表5中我们可以看到，在$k = {1000}$的情况下，PLAID ColBERTv2在GPU上比原始Col - BERTv2性能提升${2.5} \times$，在CPU上提升${9.2} \times$；此外，与原始ColBERTv2相比，这种设置实际上提高了质量。在有一定质量损失的情况下，PLAID ColBERTv2在GPU上可以实现${3.8} - {7.3} \times$的加速，在CPU上可以实现${23.2} - {42.5} \times$的加速。请注意，使用PLAID ColBERTv2在LoTTE上实现的CPU延迟比在MS MARCO v1上实现的要大，因为LoTTE段落的平均长度大约是MS MARCO v1的$2 \times$。

Finally, Table 6 shows that PLAID ColBERTv2 scales effectively to MS MARCO v2, which is a large-scale dataset with 138M passages and 9.4B tokens (approximately ${16} \times$ bigger than MS MARCO v1). Continuing the trend we observe with other datasets, we find that PLAID ColBERTv2 is ${20.8} \times$ faster than vanilla ColBERTv2 on CPU with no quality loss up to 100 passages. We do find that when $k = {1000}$ both vanilla ColBERTv2 and PLAID ColBERTv2 run out of memory on GPU; we believe we can address this in PLAID by implementing custom padding-free MaxSim kernels for GPU execution as discussed in §4.5.

最后，表6显示PLAID ColBERTv2能够有效地扩展到MS MARCO v2，这是一个包含1.38亿个段落和940亿个标记的大规模数据集（大约比MS MARCO v1大${16} \times$）。延续我们在其他数据集上观察到的趋势，我们发现PLAID ColBERTv2在CPU上比原始ColBERTv2快${20.8} \times$，在最多100个段落的情况下没有质量损失。我们确实发现，当$k = {1000}$时，原始ColBERTv2和PLAID ColBERTv2在GPU上都会出现内存不足的情况；我们相信可以通过在PLAID中实现用于GPU执行的自定义无填充MaxSim内核来解决这个问题，如§4.5中所述。

<!-- Media -->

<!-- figureText: Vanilla ColBERTv2 1.0x 3.7x 5.2x 6.6x 2.5 5.0 7.5 Speedup (a) GPU. 1.0x 4.2x 8.6x 42.4x 15 30 45 Speedup (b) CPU (8 threads). + Centroid Interaction + Centroid Pruning + Fast Decompression 0.0 Vanilla ColBERTv2 + Centroid Interaction + Centroid Pruning + Fast Kernels 0 -->

<img src="https://cdn.noedgeai.com/0195afc2-9af9-789e-9f2d-8c0aed3d29e7_7.jpg?x=199&y=243&w=618&h=564&r=0"/>

Figure 6: Ablation of performance optimizations included in PLAID.

图6：PLAID中包含的性能优化的消融分析。

<!-- Media -->

### 5.3 Ablation

### 5.3 消融分析

Figure 6 presents an ablation analysis to break down PLAID's performance improvements for both GPU and CPU execution. Our measurements are taken from evaluation on a random sample of 500 MS MARCO v1 queries (note that this results in minor differences in the absolute numbers reported in Table 3). We consider vanilla ColBERTv2 as a baseline, and then add one stage of centroid interaction without pruning (stage 3 in Figure 5), followed by another stage of centroid interaction with centroid pruning (stage 2 in Figure 5), and then finally the optimized kernels described in §4.5. When applicable we use hyperparameters corresponding to the $k = {1000}$ setting described in Table 2 (i.e.,the most conservative setting).

图6展示了一项消融分析，以剖析PLAID在GPU和CPU执行方面的性能提升。我们的测量是基于对500个MS MARCO v1查询的随机样本进行评估得出的（请注意，这会导致与表3中报告的绝对值有细微差异）。我们将原始ColBERTv2作为基线，然后添加一个无剪枝的质心交互阶段（图5中的阶段3），接着添加一个有质心剪枝的质心交互阶段（图5中的阶段2），最后添加§4.5中描述的优化内核。适用时，我们使用与表2中描述的$k = {1000}$设置（即最保守的设置）相对应的超参数。

We find that both the algorithmic improvements to the scoring pipeline as well as the implementation optimizations are key to PLAID's performance. In particular, the centroid interaction stages alone deliver speedups of ${5.2} \times$ on GPU and ${8.6} \times$ on CPU,but adding the implementation optimizations result in additional speedups of ${1.3} \times$ on GPU and ${4.9} \times$ on CPU. Only enabling optimized C++ kernels on CPU without centroid interaction (not shown in Figure 6) results in an end-to-end speedup of just $3 \times$ compared to ${42.4} \times$ with the complete PLAID.

我们发现，评分流程的算法改进以及实现优化都是PLAID性能的关键。特别是，仅质心交互阶段就能在GPU上实现${5.2} \times$的加速，在CPU上实现${8.6} \times$的加速，但添加实现优化后，在GPU上还能额外实现${1.3} \times$的加速，在CPU上额外实现${4.9} \times$的加速。仅在CPU上启用优化的C++内核而不进行质心交互（图6未展示），与完整的PLAID相比，端到端加速仅为$3 \times$，而完整PLAID的加速为${42.4} \times$。

### 5.4 Scalability

### 5.4 可扩展性

We evaluate PLAID's scalability with respect to both the dataset size as well as the parallelism degree (on CPU).

我们从数据集大小和并行度（在CPU上）两方面评估了PLAID的可扩展性。

First, Figure 7 plots the end-to-end PLAID ColBERTv2 latencies we measured for each benchmark dataset versus the size of each dataset (measured in number of embeddings). While latencies across different datasets are not necessarily directly comparable (e.g due to different passage lengths), we nevertheless aim to analyze high-level trends from this figure. We find that in general, PLAID ColBERTv2 latencies appear to scale with respect to the square root of dataset size. This intuitively follows from the fact that ColBERTv2 sets the number of centroids proportionally to the square root of the number of embeddings, and the overhead of candidate generation is inversely correlated with the number of partitions.

首先，图7绘制了我们为每个基准数据集测量的端到端PLAID ColBERTv2延迟与每个数据集大小（以嵌入数量衡量）的关系。虽然不同数据集的延迟不一定直接可比（例如，由于段落长度不同），但我们仍旨在从该图中分析高层次的趋势。我们发现，总体而言，PLAID ColBERTv2延迟似乎与数据集大小的平方根成比例。这直观上源于以下事实：ColBERTv2将质心数量设置为与嵌入数量的平方根成比例，并且候选生成的开销与分区数量成反比。

<!-- Media -->

<!-- figureText: Latency (ms) k=100 $\mathrm{k} = {1000}$ ${2}^{8}$ Latency (ms) 0 Dataset size (# of embeddings) (b) CPU (8 threads). 2 ${2}^{ \circ  }$ ${2}^{32}$ Dataset size (# of embeddings) (a) GPU. -->

<img src="https://cdn.noedgeai.com/0195afc2-9af9-789e-9f2d-8c0aed3d29e7_7.jpg?x=928&y=245&w=715&h=355&r=0"/>

Figure 7: End-to-end latency versus dataset size (as measured in number of embeddings) for each setting of $k$ (note the log-log scale). Dataset sizes are taken from Table 1, and latency numbers are taken from Tables 3, 4, 5, and 6.

图7：每种$k$设置下的端到端延迟与数据集大小（以嵌入数量衡量）的关系（注意使用双对数刻度）。数据集大小取自表1，延迟数值取自表3、表4、表5和表6。

<!-- figureText: $\mathrm{k} = {10}$ $\mathrm{k} = {100}$ $\mathrm{k} = {1000}$ 2 2 #Threads 400 Latency (ms) 300 200 100 ${2}^{0}$ 2 -->

<img src="https://cdn.noedgeai.com/0195afc2-9af9-789e-9f2d-8c0aed3d29e7_7.jpg?x=1031&y=798&w=499&h=382&r=0"/>

Figure 8: PLAID scaling behavior with respect to the number of available CPU threads.

图8：PLAID相对于可用CPU线程数量的扩展行为。

<!-- Media -->

Next, Figure 8 plots the latency achieved by PLAID ColBERTv2 versus the number of available CPU threads,repeated for $k \in$ $\{ {10},{100},{1000}\}$ . We evaluate a random sample of ${500}\mathrm{{MS}}$ MARCO v1 queries to obtain the latency measurements. We observe that PLAID is able to take advantage of additional threads; in particular, executing with 16 threads results in a speedup of ${4.9} \times$ compared to single-threaded execution when $k = {1000}$ . While PLAID does not achieve perfect linear scaling, we speculate that possible explanations could include remaining inefficiencies in the existing vanilla ColBERTv2 candidate generation step (which we do not optimize at a low level for this work) or suboptimal load balancing between threads due to the non-uniform passage lengths. We defer more extensive profiling and potential solutions to future work.

接下来，图8绘制了PLAID ColBERTv2实现的延迟与可用CPU线程数量的关系，针对$k \in$ $\{ {10},{100},{1000}\}$重复进行。我们对${500}\mathrm{{MS}}$个MARCO v1查询的随机样本进行评估以获得延迟测量值。我们观察到，PLAID能够利用额外的线程；特别是，当$k = {1000}$时，与单线程执行相比，使用16个线程执行可实现${4.9} \times$的加速。虽然PLAID没有实现完美的线性扩展，但我们推测可能的解释包括现有普通ColBERTv2候选生成步骤中仍存在效率低下的问题（在这项工作中我们没有在底层对其进行优化），或者由于段落长度不均匀导致线程之间的负载平衡不理想。我们将更广泛的性能分析和潜在解决方案留待未来工作。

## 6 CONCLUSION

## 6 结论

In this work, we presented PLAID, an efficient engine for late interaction that accelerates retrieval by aggressively and cheaply filtering candidate passages. We showed that retrieval with only ColBERTv2 centroids retains high recall compared to vanilla Col-BERTv2, and the distribution of centroid relevance scores skews toward lower magnitude scores. Using these insights, we introduced the technique of centroid interaction and incorporated centroid interaction into multiple stages of the PLAID ColBERTv2 scoring pipeline. We also described our highly optimized implementation of PLAID that includes custom kernels for padding-free MaxSim and residual decompression operations. We found in our evaluation across several IR benchmarks that PLAID ColBERTv2 provides speedups of ${2.5} - {6.8} \times$ on GPU and ${9.2} - {45} \times$ on CPU with virtually no quality loss compared to vanilla ColBERTv2 while scaling effectively to a dataset of 140 million passages.

在这项工作中，我们提出了PLAID，这是一种用于后期交互的高效引擎，通过积极且低成本地过滤候选段落来加速检索。我们表明，仅使用ColBERTv2质心进行检索与普通ColBERTv2相比仍能保持较高的召回率，并且质心相关性得分的分布偏向于较低的分数。基于这些见解，我们引入了质心交互技术，并将质心交互融入到PLAID ColBERTv2评分流程的多个阶段。我们还描述了PLAID的高度优化实现，其中包括用于无填充MaxSim和残差解压缩操作的自定义内核。我们在多个信息检索（IR）基准测试中的评估发现，与普通ColBERTv2相比，PLAID ColBERTv2在GPU上实现了${2.5} - {6.8} \times$的加速，在CPU上实现了${9.2} - {45} \times$的加速，且几乎没有质量损失，同时能有效地扩展到包含1.4亿个段落的数据集。

## REFERENCES

## 参考文献

[1] Firas Abuzaid, Geet Sethi, Peter Bailis, and Matei Zaharia. 2019. To index or not to index: Optimizing exact maximum inner product search. In 2019 IEEE 35th International Conference on Data Engineering (ICDE). IEEE, 1250-1261.

[1] Firas Abuzaid、Geet Sethi、Peter Bailis和Matei Zaharia。2019年。是否进行索引：优化精确最大内积搜索。见2019年IEEE第35届国际数据工程会议（ICDE）。IEEE，1250 - 1261。

[2] Anserini GitHub Repo Authors. 2021. Passage Collection (Augmented). https://github.com/castorini/anserini/blob/master/docs/experiments-msmarco-v2.md#passage-collection-augmented

[2] Anserini GitHub仓库作者。2021年。段落集合（增强版）。https://github.com/castorini/anserini/blob/master/docs/experiments-msmarco-v2.md#passage-collection-augmented

[3] Anserini GitHub Repo Authors. 2022. Anserini Regressions: MS MARCO (V2) Passage Ranking. https://github.com/castorini/anserini/blob/master/docs/ regressions-msmarco-v2-passage-augmented.md

[3] Anserini GitHub仓库作者。2022年。Anserini回归测试：MS MARCO（V2）段落排序。https://github.com/castorini/anserini/blob/master/docs/ regressions-msmarco-v2-passage-augmented.md

[4] Jo Kristian Bergum. 2021. Pretrained Transformer Language Models for Search - part 3. https://blog.vespa.ai/pretrained-transformer-language-models-for-search-part-3/

[4] 乔·克里斯蒂安·伯古姆（Jo Kristian Bergum）。2021年。用于搜索的预训练Transformer语言模型 - 第3部分。https://blog.vespa.ai/pretrained-transformer-language-models-for-search-part-3/

[5] Andrei Z Broder, David Carmel, Michael Herscovici, Aya Soffer, and Jason Zien. 2003. Efficient query evaluation using a two-level retrieval process. In CIKM.

[5] 安德烈·Z·布罗德（Andrei Z Broder）、大卫·卡梅尔（David Carmel）、迈克尔·赫斯科维奇（Michael Herscovici）、阿亚·索弗（Aya Soffer）和杰森·齐恩（Jason Zien）。2003年。使用两级检索过程进行高效查询评估。发表于信息与知识管理大会（CIKM）。

[6] Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, and Jimmy Lin. 2022. Overview of the TREC 2021 deep learning track. In Text REtrieval Conference (TREC). TREC. https://www.microsoft.com/en-us/research/publication/ overview-of-the-trec-2021-deep-learning-track/

[6] 尼克·克拉斯韦尔（Nick Craswell）、巴斯卡尔·米特拉（Bhaskar Mitra）、埃米内·伊尔马兹（Emine Yilmaz）、丹尼尔·坎波斯（Daniel Campos）和吉米·林（Jimmy Lin）。2022年。2021年文本检索会议（TREC）深度学习赛道概述。发表于文本检索会议（Text REtrieval Conference，TREC）。TREC。https://www.microsoft.com/en-us/research/publication/ overview-of-the-trec-2021-deep-learning-track/

[7] Zhuyun Dai and Jamie Callan. 2020. Context-Aware Term Weighting For First Stage Passage Retrieval. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, SIGIR 2020, Virtual Event, China, July 25-30, 2020, Jimmy Huang, Yi Chang, Xueqi Cheng, Jaap Kamps, Vanessa Murdock, Ji-Rong Wen, and Yiqun Liu (Eds.). ACM, 1533-1536. https://doi.org/10.1145/3397271.3401204

[7] 戴竹云（Zhuyun Dai）和杰米·卡兰（Jamie Callan）。2020年。用于第一阶段段落检索的上下文感知词项加权。发表于第43届ACM信息检索研究与发展国际会议（Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval，SIGIR 2020），2020年7月25 - 30日，中国（线上会议），黄吉明（Jimmy Huang）、张毅（Yi Chang）、程学旗（Xueqi Cheng）、亚普·坎普斯（Jaap Kamps）、凡妮莎·默多克（Vanessa Murdock）、温 Ji - Rong（Ji - Rong Wen）和刘奕群（Yiqun Liu）（编）。美国计算机协会（ACM），1533 - 1536页。https://doi.org/10.1145/3397271.3401204

[8] Constantinos Dimopoulos, Sergey Nepomnyachiy, and Torsten Suel. 2013. Optimizing top-k document retrieval strategies for block-max indexes. In WSDM.

[8] 康斯坦丁诺斯·季莫普洛斯（Constantinos Dimopoulos）、谢尔盖·涅波姆尼亚奇（Sergey Nepomnyachiy）和托尔斯滕·苏埃尔（Torsten Suel）。2013年。针对块最大索引优化前k个文档检索策略。发表于网络搜索与数据挖掘会议（WSDM）。

[9] Shuai Ding and Torsten Suel. 2011. Faster top-k document retrieval using block-max indexes. In SIGIR.

[9] 丁帅（Shuai Ding）和托尔斯滕·苏埃尔（Torsten Suel）。2011年。使用块最大索引实现更快的前k个文档检索。发表于信息检索研究与发展会议（SIGIR）。

[10] Thibault Formal, Carlos Lassance, Benjamin Piwowarski, and Stéphane Clinchant. 2021. SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval. arXiv preprint arXiv:2109.10086 (2021). https://arxiv.org/abs/2109.10086

[10] 蒂博·福尔马尔（Thibault Formal）、卡洛斯·拉萨斯（Carlos Lassance）、本杰明·皮沃瓦尔斯基（Benjamin Piwowarski）和斯特凡·克兰尚（Stéphane Clinchant）。2021年。SPLADE v2：用于信息检索的稀疏词法与扩展模型。预印本arXiv:2109.10086（2021年）。https://arxiv.org/abs/2109.10086

[11] Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant. 2021. SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2288-2292.

[11] 蒂博·福尔马尔（Thibault Formal）、本杰明·皮沃瓦尔斯基（Benjamin Piwowarski）和斯特凡·克兰尚（Stéphane Clinchant）。2021年。SPLADE：用于第一阶段排序的稀疏词法与扩展模型。发表于第44届ACM信息检索研究与发展国际会议（Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval）。2288 - 2292页。

[12] Luyu Gao, Zhuyun Dai, and Jamie Callan. 2021. COIL: Revisit Exact Lexical Match in Information Retrieval with Contextualized Inverted List. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. Association for Computational Linguistics, Online, 3030-3042. https://doi.org/10.18653/v1/2021.naacl-main.241

[12] 高璐宇（Luyu Gao）、戴竹云（Zhuyun Dai）和杰米·卡兰（Jamie Callan）。2021年。COIL：通过上下文倒排列表重新审视信息检索中的精确词法匹配。发表于计算语言学协会北美分会2021年会议：人类语言技术（Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies）。计算语言学协会，线上会议，3030 - 3042页。https://doi.org/10.18653/v1/2021.naacl-main.241

[13] Sebastian Hofstätter, Sophia Althammer, Michael Schröder, Mete Sertkan, and Allan Hanbury. 2020. Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation. arXiv preprint arXiv:2010.02666 (2020). https://arxiv.org/abs/2010.02666

[13] 塞巴斯蒂安·霍夫施泰特（Sebastian Hofstätter）、索菲娅·阿尔塔默（Sophia Althammer）、迈克尔·施罗德（Michael Schröder）、梅特·塞尔特坎（Mete Sertkan）和艾伦·汉伯里（Allan Hanbury）。2020年。通过跨架构知识蒸馏改进高效神经排序模型。预印本arXiv:2010.02666（2020年）。https://arxiv.org/abs/2010.02666

[14] Sebastian Hofstätter, Omar Khattab, Sophia Althammer, Mete Sertkan, and Allan Hanbury. 2022. Introducing Neural Bag of Whole-Words with ColBERTer: Contextualized Late Interactions using Enhanced Reduction. arXiv preprint arXiv:2203.13088 (2022).

[14] 塞巴斯蒂安·霍夫施泰特（Sebastian Hofstätter）、奥马尔·哈塔卜（Omar Khattab）、索菲娅·阿尔塔默（Sophia Althammer）、梅特·塞尔特坎（Mete Sertkan）和艾伦·汉伯里（Allan Hanbury）。2022年。使用ColBERTer引入全词神经词袋：通过增强约简实现上下文延迟交互。预印本arXiv:2203.13088（2022年）。

[15] Samuel Humeau, Kurt Shuster, Marie-Anne Lachaux, and Jason Weston. 2020. Poly-encoders: Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net. https://openreview.net/forum?id=SkxgnnNFvH

[15] 塞缪尔·休莫（Samuel Humeau）、库尔特·舒斯特（Kurt Shuster）、玛丽 - 安妮·拉肖（Marie - Anne Lachaux）和杰森·韦斯顿（Jason Weston）。2020年。多编码器：用于快速准确多句子评分的架构和预训练策略。发表于第8届学习表征国际会议（8th International Conference on Learning Representations，ICLR 2020），2020年4月26 - 30日，埃塞俄比亚亚的斯亚贝巴。OpenReview.net。https://openreview.net/forum?id=SkxgnnNFvH

[16] Herve Jegou, Matthijs Douze, and Cordelia Schmid. 2010. Product quantization for nearest neighbor search. IEEE transactions on pattern analysis and machine intelligence 33, 1 (2010), 117-128.

[16] 埃尔韦·热古（Herve Jegou）、马蒂亚斯·杜泽（Matthijs Douze）和科迪莉亚·施密德（Cordelia Schmid）。2010年。用于最近邻搜索的乘积量化。《IEEE模式分析与机器智能汇刊》（IEEE transactions on pattern analysis and machine intelligence）33卷，第1期（2010年），117 - 128页。

[17] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-scale similarity search with gpus. IEEE Transactions on Big Data (2019).

[17] 杰夫·约翰逊（Jeff Johnson）、马蒂亚斯·杜泽（Matthijs Douze）和埃尔韦·热古（Hervé Jégou）。2019年。使用GPU进行十亿级相似度搜索。《IEEE大数据汇刊》（IEEE Transactions on Big Data）（2019年）。

[18] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for Open-Domain Question Answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). Association for Computational Linguistics, Online, 6769-6781. https://doi.org/10.18653/v1/2020.emnlp-main.550

[18] 弗拉基米尔·卡尔普欣（Vladimir Karpukhin）、巴拉斯·奥古兹（Barlas Oguz）、闵世元（Sewon Min）、帕特里克·刘易斯（Patrick Lewis）、莱德尔·吴（Ledell Wu）、谢尔盖·叶杜诺夫（Sergey Edunov）、陈丹琦（Danqi Chen）和易文涛（Wen-tau Yih）。2020年。用于开放域问答的密集段落检索。见《2020年自然语言处理经验方法会议论文集》（EMNLP）。计算语言学协会，线上会议，6769 - 6781页。https://doi.org/10.18653/v1/2020.emnlp-main.550

[19] Omar Khattab, Mohammad Hammoud, and Tamer Elsayed. 2020. Finding the best of both worlds: Faster and more robust top-k document retrieval. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval. 1031-1040.

[19] 奥马尔·哈塔卜（Omar Khattab）、穆罕默德·哈穆德（Mohammad Hammoud）和塔默·埃尔赛义德（Tamer Elsayed）。2020年。两全其美：更快、更稳健的前k个文档检索。见《第43届ACM国际信息检索研究与发展会议论文集》。1031 - 1040页。

[20] Omar Khattab, Christopher Potts, and Matei Zaharia. 2021. Baleen: Robust MultiHop Reasoning at Scale via Condensed Retrieval. In Thirty-Fifth Conference on Neural Information Processing Systems.

[20] 奥马尔·哈塔卜（Omar Khattab）、克里斯托弗·波茨（Christopher Potts）和马特·扎哈里亚（Matei Zaharia）。2021年。须鲸（Baleen）：通过浓缩检索实现大规模稳健多跳推理。见《第三十五届神经信息处理系统会议》。

[21] Omar Khattab, Christopher Potts, and Matei Zaharia. 2021. Relevance-guided Supervision for OpenQA with ColBERT. Transactions of the Association for Computational Linguistics 9 (2021), 929-944.

[21] 奥马尔·哈塔卜（Omar Khattab）、克里斯托弗·波茨（Christopher Potts）和马特·扎哈里亚（Matei Zaharia）。2021年。使用ColBERT为开放域问答提供相关性引导监督。《计算语言学协会汇刊》9（2021年），929 - 944页。

[22] Omar Khattab and Matei Zaharia. 2020. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, SIGIR 2020, Virtual Event, China, July 25-30, 2020, Jimmy Huang, Yi Chang, Xueqi Cheng, Jaap Kamps, Vanessa Murdock, Ji-Rong Wen, and Yiqun Liu (Eds.). ACM, 39-48. https://doi.org/10.1145/3397271.3401075

[22] 奥马尔·哈塔卜（Omar Khattab）和马特·扎哈里亚（Matei Zaharia）。2020年。ColBERT：通过基于BERT的上下文后期交互实现高效的段落搜索。见《第43届ACM国际信息检索研究与发展会议论文集》，SIGIR 2020，线上会议，中国，2020年7月25 - 30日，黄吉民（Jimmy Huang）、张毅（Yi Chang）、程学旗（Xueqi Cheng）、亚普·坎普斯（Jaap Kamps）、凡妮莎·默多克（Vanessa Murdock）、温 Ji - Rong（Ji - Rong Wen）和刘奕群（Yiqun Liu）（编）。美国计算机协会，39 - 48页。https://doi.org/10.1145/3397271.3401075

[23] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural Questions: A Benchmark for Question Answering Research. Transactions of the Association for Computational Linguistics 7 (2019), 452-466. https://doi.org/10.1162/tacl_a_00276

[23] 汤姆·夸特科夫斯基（Tom Kwiatkowski）、珍妮玛丽亚·帕洛马基（Jennimaria Palomaki）、奥利维亚·雷德菲尔德（Olivia Redfield）、迈克尔·柯林斯（Michael Collins）、安库尔·帕里克（Ankur Parikh）、克里斯·阿尔伯蒂（Chris Alberti）、丹妮尔·爱泼斯坦（Danielle Epstein）、伊利亚·波洛苏金（Illia Polosukhin）、雅各布·德夫林（Jacob Devlin）、肯顿·李（Kenton Lee）、克里斯蒂娜·图托纳娃（Kristina Toutanova）、利昂·琼斯（Llion Jones）、马修·凯尔西（Matthew Kelcey）、张明伟（Ming - Wei Chang）、安德鲁·M·戴（Andrew M. Dai）、雅各布·乌兹科雷特（Jakob Uszkoreit）、乐存（Quoc Le）和斯拉夫·彼得罗夫（Slav Petrov）。2019年。自然问题：问答研究的基准。《计算语言学协会汇刊》7（2019年），452 - 466页。https://doi.org/10.1162/tacl_a_00276

[24] Yulong Li, Martin Franz, Md Arafat Sultan, Bhavani Iyer, Young-Suk Lee, and Avirup Sil. 2021. Learning Cross-Lingual IR from an English Retriever. arXiv preprint arXiv:2112.08185 (2021).

[24] 李玉龙（Yulong Li）、马丁·弗朗茨（Martin Franz）、穆罕默德·阿拉法特·苏丹（Md Arafat Sultan）、巴瓦尼·伊耶尔（Bhavani Iyer）、李英淑（Young - Suk Lee）和阿维鲁普·西尔（Avirup Sil）。2021年。从英语检索器学习跨语言信息检索。预印本arXiv:2112.08185（2021年）。

[25] Jimmy Lin. 2022. A proposed conceptual framework for a representational approach to information retrieval. In ACM SIGIR Forum, Vol. 55. ACM New York, NY, USA, 1-29.

[25] 吉米·林（Jimmy Lin）。2022年。一种用于信息检索表征方法的概念框架提案。见《ACM SIGIR论坛》，第55卷。美国计算机协会，纽约，美国，1 - 29页。

[26] Simon Lupart and Stéphane Clinchant. 2022. Toward A Fine-Grained Analysis of Distribution Shifts in MSMARCO. arXiv preprint arXiv:2205.02870 (2022).

[26] 西蒙·卢帕特（Simon Lupart）和斯特凡·克兰尚（Stéphane Clinchant）。2022年。迈向对MSMARCO中分布偏移的细粒度分析。预印本arXiv:2205.02870（2022年）。

[27] Sean MacAvaney, Andrew Yates, Arman Cohan, and Nazli Goharian. 2019. CEDR: Contextualized Embeddings for Document Ranking. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval. 1101-1104. https://doi.org/10.1145/3331184.3331317

[27] 肖恩·麦卡瓦尼（Sean MacAvaney）、安德鲁·耶茨（Andrew Yates）、阿尔曼·科汉（Arman Cohan）和纳兹利·戈哈里安（Nazli Goharian）。2019年。CEDR：用于文档排名的上下文嵌入。见《第42届ACM国际信息检索研究与发展会议论文集》。1101 - 1104页。https://doi.org/10.1145/3331184.3331317

[28] Craig Macdonald and Nicola Tonellotto. 2021. On approximate nearest neighbour selection for multi-stage dense retrieval. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management. 3318-3322.

[28] 克雷格·麦克唐纳（Craig Macdonald）和尼古拉·托内洛托（Nicola Tonellotto）。2021年。关于多阶段密集检索的近似最近邻选择。见《第30届ACM国际信息与知识管理会议论文集》。3318 - 3322页。

[29] Craig Macdonald, Nicola Tonellotto, and Iadh Ounis. 2021. On Single and Multiple Representations in Dense Passage Retrieval. arXiv preprint arXiv:2108.06279 (2021).

[29] 克雷格·麦克唐纳（Craig Macdonald）、尼古拉·托内洛托（Nicola Tonellotto）和伊阿德·乌尼斯（Iadh Ounis）。2021年。密集段落检索中的单表示和多表示研究。预印本arXiv:2108.06279（2021年）。

[30] Joel Mackenzie, Andrew Trotman, and Jimmy Lin. 2021. Wacky Weights in Learned Sparse Representations and the Revenge of Score-at-a-Time Query Evaluation. arXiv preprint arXiv:2110.11540 (2021).

[30] 乔尔·麦肯齐（Joel Mackenzie）、安德鲁·特罗特曼（Andrew Trotman）和吉米·林（Jimmy Lin）。2021年。学习稀疏表示中的奇异权重与逐时评分查询评估的逆袭。预印本arXiv:2110.11540（2021年）。

[31] Yu A Malkov and Dmitry A Yashunin. 2018. Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. IEEE transactions on pattern analysis and machine intelligence 42, 4 (2018), 824-836.

[31] 尤·A·马尔科夫（Yu A Malkov）和德米特里·A·亚舒宁（Dmitry A Yashunin）。2018年。使用层次可导航小世界图进行高效且稳健的近似最近邻搜索。《电气与电子工程师协会模式分析与机器智能汇刊》（IEEE transactions on pattern analysis and machine intelligence）42卷，第4期（2018年），824 - 836页。

[32] Antonio Mallia, Omar Khattab, Torsten Suel, and Nicola Tonellotto. 2021. Learning passage impacts for inverted indexes. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. 1723-1727.

[32] 安东尼奥·马利亚（Antonio Mallia）、奥马尔·哈塔布（Omar Khattab）、托尔斯滕·苏埃尔（Torsten Suel）和尼古拉·托内洛托（Nicola Tonellotto）。2021年。学习倒排索引的段落影响。收录于《第44届国际计算机协会信息检索研究与发展会议论文集》（Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval）。1723 - 1727页。

[33] Antonio Mallia, Giuseppe Ottaviano, Elia Porciani, Nicola Tonellotto, and Rossano Venturini. 2017. Faster blockmax wand with variable-sized blocks. In SIGIR.

[33] 安东尼奥·马利亚（Antonio Mallia）、朱塞佩·奥塔维亚诺（Giuseppe Ottaviano）、埃利亚·波尔恰尼（Elia Porciani）、尼古拉·托内洛托（Nicola Tonellotto）和罗萨诺·文图里尼（Rossano Venturini）。2017年。具有可变大小块的更快块最大魔杖算法。收录于《信息检索研究与发展会议》（SIGIR）。

[34] Antonio Mallia, Michal Siedlaczek, Joel Mackenzie, and Torsten Suel. 2019. PISA: Performant indexes and search for academia. Proceedings of the Open-Source IR Replicability Challenge (2019).

[34] 安东尼奥·马利亚（Antonio Mallia）、米哈尔·谢德拉切克（Michal Siedlaczek）、乔尔·麦肯齐（Joel Mackenzie）和托尔斯滕·苏埃尔（Torsten Suel）。2019年。PISA：面向学术界的高性能索引与搜索。《开源信息检索可重复性挑战会议论文集》（Proceedings of the Open - Source IR Replicability Challenge）（2019年）。

[35] Antonios Minas Krasakis, Andrew Yates, and Evangelos Kanoulas. 2022. Zero-shot Query Contextualization for Conversational Search. arXiv e-prints (2022), arXiv-2204.

[35] 安东尼斯·米纳斯·克拉萨基斯（Antonios Minas Krasakis）、安德鲁·耶茨（Andrew Yates）和埃万杰洛斯·卡努拉斯（Evangelos Kanoulas）。2022年。用于对话式搜索的零样本查询上下文建模。《arXiv电子预印本》（arXiv e - prints）（2022年），arXiv - 2204。

[36] Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, and Li Deng. 2016. MS MARCO: A Human-Generated MAchine Reading COmprehension Dataset. arXiv preprint arXiv:1611.09268 (2016). https: //arxiv.org/abs/1611.09268

[36] 特里·阮（Tri Nguyen）、米尔·罗森伯格（Mir Rosenberg）、宋霞（Xia Song）、高剑锋（Jianfeng Gao）、索拉布·蒂瓦里（Saurabh Tiwary）、兰甘·马朱姆德（Rangan Majumder）和李登（Li Deng）。2016年。MS MARCO：一个人工生成的机器阅读理解数据集。《arXiv预印本》arXiv:1611.09268（2016年）。https: //arxiv.org/abs/1611.09268

[37] Rodrigo Nogueira and Kyunghyun Cho. 2019. Passage Re-ranking with BERT. arXiv preprint arXiv:1901.04085 (2019). https://arxiv.org/abs/1901.04085

[37] 罗德里戈·诺盖拉（Rodrigo Nogueira）和赵京焕（Kyunghyun Cho）。2019年。使用BERT进行段落重排序。《arXiv预印本》arXiv:1901.04085（2019年）。https://arxiv.org/abs/1901.04085

[38] Ashwin Paranjape, Omar Khattab, Christopher Potts, Matei Zaharia, and Christopher D Manning. 2022. Hindsight: Posterior-guided Training of Retrievers for Improved Open-ended Generation. In International Conference on Learning Representations. https://openreview.net/forum?id=Vr_BTpw3wz

[38] 阿什温·帕兰贾佩（Ashwin Paranjape）、奥马尔·哈塔布（Omar Khattab）、克里斯托弗·波茨（Christopher Potts）、马特·扎哈里亚（Matei Zaharia）和克里斯托弗·D·曼宁（Christopher D Manning）。2022年。后见之明：通过后验引导训练检索器以改进开放式生成。收录于《国际学习表征会议》（International Conference on Learning Representations）。https://openreview.net/forum?id=Vr_BTpw3wz

[39] Yingqi Qu, Yuchen Ding, Jing Liu, Kai Liu, Ruiyang Ren, Wayne Xin Zhao, Daxiang Dong, Hua Wu, and Haifeng Wang. 2021. RocketQA: An Optimized

[39] 曲英琦、丁雨晨、刘静、刘凯、任瑞阳、赵鑫、董大祥、吴华和王海峰。2021年。RocketQA：一种优化的

Training Approach to Dense Passage Retrieval for Open-Domain Question Answering. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. Association for Computational Linguistics, Online, 5835-5847. https: //doi.org/10.18653/v1/2021.naacl-main.466

用于开放域问答的密集段落检索训练方法。收录于《2021年计算语言学协会北美分会人类语言技术会议论文集》（Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies）。计算语言学协会，线上会议，5835 - 5847页。https: //doi.org/10.18653/v1/2021.naacl - main.466

[40] Ruiyang Ren, Yingqi Qu, Jing Liu, Wayne Xin Zhao, Qiaoqiao She, Hua Wu, Haifeng Wang, and Ji-Rong Wen. 2021. RocketQAv2: A Joint Training Method for Dense Passage Retrieval and Passage Re-ranking. arXiv preprint arXiv:2110.07367 (2021). https://arxiv.org/abs/2110.07367

[40] 任瑞阳、曲英琦、刘静、赵鑫、佘巧巧、吴华、王海峰和文继荣。2021年。RocketQAv2：一种用于密集段落检索和段落重排序的联合训练方法。《arXiv预印本》arXiv:2110.07367（2021年）。https://arxiv.org/abs/2110.07367

[41] Stephen E Robertson, Steve Walker, Susan Jones, Micheline M Hancock-Beaulieu, Mike Gatford, et al. 1995. Okapi at TREC-3. NIST Special Publication (1995).

[41] 斯蒂芬·E·罗伯逊（Stephen E Robertson）、史蒂夫·沃克（Steve Walker）、苏珊·琼斯（Susan Jones）、米歇琳·M·汉考克 - 博勒伊（Micheline M Hancock - Beaulieu）、迈克·加特福德（Mike Gatford）等。1995年。TREC - 3会议上的Okapi系统。《美国国家标准与技术研究院特别出版物》（NIST Special Publication）（1995年）。

[42] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. 2021. ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction. arXiv preprint arXiv:2112.01488 (2021).

[42] 凯沙夫·桑塔南姆（Keshav Santhanam）、奥马尔·哈塔布（Omar Khattab）、乔恩·萨德 - 法尔孔（Jon Saad - Falcon）、克里斯托弗·波茨（Christopher Potts）和马特·扎哈里亚（Matei Zaharia）。2021年。ColBERTv2：通过轻量级后期交互实现高效检索。《arXiv预印本》arXiv:2112.01488（2021年）。

[43] Katherine Thai, Yapei Chang, Kalpesh Krishna, and Mohit Iyyer. 2022. RELIC: Retrieving Evidence for Literary Claims. arXiv preprint arXiv:2203.10053 (2022).

[43] 凯瑟琳·泰（Katherine Thai）、张雅佩（Yapei Chang）、卡尔佩什·克里希纳（Kalpesh Krishna）和莫希特·伊耶尔（Mohit Iyyer）。2022年。RELIC：为文学论断检索证据。《arXiv预印本》arXiv:2203.10053（2022年）。

[44] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021. BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2). https://openreview.net/forum?id=wCu6T5xFjeJ

[44] 南丹·塔库尔（Nandan Thakur）、尼尔斯·赖默斯（Nils Reimers）、安德里亚斯·吕克莱（Andreas Rücklé）、阿比舍克·斯里瓦斯塔瓦（Abhishek Srivastava）和伊琳娜·古列维奇（Iryna Gurevych）。2021年。BEIR：一个用于信息检索模型零样本评估的异构基准。收录于《第三十五届神经信息处理系统数据集与基准会议（第二轮）》（Thirty - fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)）。https://openreview.net/forum?id=wCu6T5xFjeJ

[45] Nicola Tonellotto and Craig Macdonald. 2021. Query embedding pruning for dense retrieval. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management. 3453-3457.

[45] 尼古拉·托内洛托（Nicola Tonellotto）和克雷格·麦克唐纳德（Craig Macdonald）。2021年。用于密集检索的查询嵌入剪枝。收录于《第30届美国计算机协会信息与知识管理国际会议论文集》（Proceedings of the 30th ACM International Conference on Information & Knowledge Management）。3453 - 3457页。

[46] Nicola Tonellotto, Craig Macdonald, Iadh Ounis, et al. 2018. Efficient Query Processing for Scalable Web Search. Foundations and Trends ${}^{\circledR }$ in Information

[46] 尼古拉·托内洛托（Nicola Tonellotto）、克雷格·麦克唐纳（Craig Macdonald）、伊阿德·乌尼斯（Iadh Ounis）等。2018年。可扩展网络搜索的高效查询处理。《信息检索的基础与趋势》（Foundations and Trends ${}^{\circledR }$ in Information）

Retrieval (2018).

检索（2018年）。

[47] Howard Turtle and James Flood. 1995. Query evaluation: strategies and optimizations. IP & M (1995).

[47] 霍华德·特特尔（Howard Turtle）和詹姆斯·弗洛德（James Flood）。1995年。查询评估：策略与优化。《信息处理与管理》（IP & M）（1995年）。

[48] Xiao Wang, Craig Macdonald, Nicola Tonellotto, and Iadh Ounis. 2021. Pseudo-relevance feedback for multiple representation dense retrieval. In Proceedings of the 2021 ACM SIGIR International Conference on Theory of Information Retrieval. 297-306.

[48] 王潇（Xiao Wang）、克雷格·麦克唐纳（Craig Macdonald）、尼古拉·托内洛托（Nicola Tonellotto）和伊阿德·乌尼斯（Iadh Ounis）。2021年。多表示密集检索的伪相关反馈。见《2021年ACM SIGIR信息检索理论国际会议论文集》。第297 - 306页。

[49] Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N Bennett, Junaid Ahmed, and Arnold Overwijk. 2020. Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval. In International Conference on Learning Representations.

[49] 熊磊（Lee Xiong）、熊晨彦（Chenyan Xiong）、李烨（Ye Li）、邓国峰（Kwok - Fung Tang）、刘佳琳（Jialin Liu）、保罗·N·贝内特（Paul N Bennett）、朱奈德·艾哈迈德（Junaid Ahmed）和阿诺德·奥弗维克（Arnold Overwijk）。2020年。用于密集文本检索的近似最近邻负对比学习。见《国际学习表征会议》。

[50] Peilin Yang, Hui Fang, and Jimmy Lin. 2018. Anserini: Reproducible ranking baselines using Lucene. Journal of Data and Information Quality (JDIQ) 10,4 (2018), 1-20.

[50] 杨培林（Peilin Yang）、方慧（Hui Fang）和林吉米（Jimmy Lin）。2018年。安瑟里尼（Anserini）：使用Lucene的可重现排名基线。《数据与信息质量杂志》（Journal of Data and Information Quality，JDIQ）第10卷第4期（2018年），第1 - 20页。

[51] Hansi Zeng, Hamed Zamani, and Vishwa Vinay. 2022. Curriculum Learning for Dense Retrieval Distillation. arXiv preprint arXiv:2204.13679 (2022).

[51] 曾汉斯（Hansi Zeng）、哈米德·扎马尼（Hamed Zamani）和维什瓦·维奈（Vishwa Vinay）。2022年。密集检索蒸馏的课程学习。预印本arXiv:2204.13679（2022年）。

[52] Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Min Zhang, and Shaoping Ma. 2020. Learning To Retrieve: How to Train a Dense Retrieval Model Effectively and Efficiently. arXiv preprint arXiv:2010.10469 (2020). https://arxiv.org/abs/2010.10469

[52] 詹景涛（Jingtao Zhan）、毛佳欣（Jiaxin Mao）、刘奕群（Yiqun Liu）、张敏（Min Zhang）和马少平（Shaoping Ma）。2020年。学习检索：如何有效且高效地训练密集检索模型。预印本arXiv:2010.10469（2020年）。https://arxiv.org/abs/2010.10469

[53] Jingtao Zhan, Xiaohui Xie, Jiaxin Mao, Yiqun Liu, Min Zhang, and Shaoping Ma. 2022. Evaluating Extrapolation Performance of Dense Retrieval. arXiv preprint arXiv:2204.11447 (2022).

[53] 詹景涛（Jingtao Zhan）、谢晓辉（Xiaohui Xie）、毛佳欣（Jiaxin Mao）、刘奕群（Yiqun Liu）、张敏（Min Zhang）和马少平（Shaoping Ma）。2022年。评估密集检索的外推性能。预印本arXiv:2204.11447（2022年）。

[54] Wei Zhong, Jheng-Hong Yang, and Jimmy Lin. 2022. Evaluating Token-Level and Passage-Level Dense Retrieval Models for Math Information Retrieval. arXiv preprint arXiv:2203.11163 (2022).

[54] 钟伟（Wei Zhong）、杨政宏（Jheng - Hong Yang）和林吉米（Jimmy Lin）。2022年。评估用于数学信息检索的Token级和段落级密集检索模型。预印本arXiv:2203.11163（2022年）。