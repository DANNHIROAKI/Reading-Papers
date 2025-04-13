# Efficient Multi-Vector Dense Retrieval with Bit Vectors

# 基于位向量的高效多向量密集检索

Franco Maria Nardini ${}^{1}$ , Cosimo Rulli ${}^{1}$ , and Rossano Venturini ${}^{2}$

佛朗哥·玛丽亚·纳尔迪尼 ${}^{1}$，科西莫·鲁利 ${}^{1}$，罗萨诺·文图里尼 ${}^{2}$

${}^{1}$ ISTI-CNR,Pisa,Italy \{name.surname\}@isti.cnr.it ${}^{2}$ University of Pisa,Italy rossano.venturini@unipi.it

${}^{1}$ 意大利国家研究委员会信息科学与技术研究所（ISTI - CNR），比萨，意大利 \{name.surname\}@isti.cnr.it ${}^{2}$ 比萨大学，意大利 rossano.venturini@unipi.it

Abstract. Dense retrieval techniques employ pre-trained large language models to build a high-dimensional representation of queries and passages. These representations compute the relevance of a passage w.r.t. to a query using efficient similarity measures. In this line, multi-vector representations show improved effectiveness at the expense of a one-order-of-magnitude increase in memory footprint and query latency by encoding queries and documents on a per-token level. Recently, PLAID has tackled these problems by introducing a centroid-based term representation to reduce the memory impact of multi-vector systems. By exploiting a centroid interaction mechanism, PLAID filters out non-relevant documents, thus reducing the cost of the successive ranking stages. This paper proposes "Efficient Multi-Vector dense retrieval with Bit vectors" (EMVB), a novel framework for efficient query processing in multi-vector dense retrieval. First, EMVB employs a highly efficient pre-filtering step of passages using optimized bit vectors. Second, the computation of the centroid interaction happens column-wise, exploiting SIMD instructions, thus reducing its latency. Third, EMVB leverages Product Quantization (PQ) to reduce the memory footprint of storing vector representations while jointly allowing for fast late interaction. Fourth, we introduce a per-document term filtering method that further improves the efficiency of the last step. Experiments on MS MARCO and LoTTE show that EMVB is up to ${2.8} \times$ faster while reducing the memory footprint by ${1.8} \times$ with no loss in retrieval accuracy compared to PLAID.

摘要：密集检索技术利用预训练的大语言模型来构建查询和段落的高维表示。这些表示使用高效的相似度度量来计算段落相对于查询的相关性。在这方面，多向量表示通过对查询和文档进行逐词编码，在提高检索效果的同时，也使内存占用和查询延迟增加了一个数量级。最近，PLAID 通过引入基于质心的词项表示来解决这些问题，以减少多向量系统的内存影响。通过利用质心交互机制，PLAID 过滤掉不相关的文档，从而降低后续排序阶段的成本。本文提出了“基于位向量的高效多向量密集检索”（EMVB），这是一种用于多向量密集检索中高效查询处理的新颖框架。首先，EMVB 使用优化的位向量对段落进行高效的预过滤步骤。其次，质心交互的计算按列进行，利用单指令多数据（SIMD）指令，从而降低其延迟。第三，EMVB 利用乘积量化（PQ）来减少存储向量表示的内存占用，同时允许快速的后期交互。第四，我们引入了一种按文档进行词项过滤的方法，进一步提高了最后一步的效率。在 MS MARCO 和 LoTTE 上的实验表明，与 PLAID 相比，EMVB 速度提高了 ${2.8} \times$，同时内存占用减少了 ${1.8} \times$，且检索准确率没有损失。

## 1 Introduction

## 1 引言

The introduction of pre-trained large language models (LLM) has remarkably improved the effectiveness of information retrieval systems [138]262], thanks to the well-known ability of LLMs to model semantic and context [121]. In dense retrieval, LLMs have been successfully exploited to learn high-dimensional dense representations of passages and queries. These learned representations allow answering the user query through fast similarity operations, i.e., inner product or L2 distance. In this line, multi-vector techniques [1420] employ an LLM to build a dense representation for each token of a passage. These approaches offer superior effectiveness compared to single-vector techniques [24 27 or sparse retrieval techniques [5]. In this context, the similarity between the query and the passage is measured by using the late interaction mechanism [1420], which works by computing the sum of the maximum similarities between each term of the query and each term of a candidate passage. The improved effectiveness of multi-vector retrieval system comes at the price of its increased computational burden. First, producing a vector for each token causes the number of embeddings to be orders of magnitude larger than in a single-vector representation. Moreover, due to the large number of embeddings,identifying the candidate documents ${}^{3}$ is time-consuming. In addition, the late interaction step requires computing the maximum similarity operator between all the candidate embeddings and the query, which is also time-consuming.

预训练大语言模型（LLM）的引入显著提高了信息检索系统的效果 [138][262]，这得益于大语言模型对语义和上下文进行建模的能力 [121]。在密集检索中，大语言模型已成功用于学习段落和查询的高维密集表示。这些学习到的表示允许通过快速的相似度操作（即内积或 L2 距离）来回答用户查询。在这方面，多向量技术 [14][20] 使用大语言模型为段落的每个词项构建密集表示。与单向量技术 [24][27] 或稀疏检索技术 [5] 相比，这些方法具有更好的效果。在这种情况下，查询和段落之间的相似度通过后期交互机制 [14][20] 来衡量，该机制通过计算查询的每个词项与候选段落的每个词项之间的最大相似度之和来工作。多向量检索系统效果的提升是以增加计算负担为代价的。首先，为每个词项生成一个向量会导致嵌入的数量比单向量表示大几个数量级。此外，由于嵌入数量众多，识别候选文档 ${}^{3}$ 非常耗时。另外，后期交互步骤需要计算所有候选嵌入与查询之间的最大相似度算子，这也很耗时。

Early multi-vector retrieval systems, i.e., ColBERT [14], exploit an inverted index to store the embeddings and retrieve the candidate passages. Then, the representations of passages are retrieved and employed to compute the max-similarity score with the query. Despite being quite efficient, this approach requires maintaining the full-precision representation of each document term in memory. On MS MARCO [17], a widely adopted benchmark dataset for passage retrieval, the entire collection of embeddings used by ColBERT requires more than 140 GB [14] to be stored. ColBERTv2 [20] introduces a centroid-based compression technique to store the passage embeddings efficiently. Each embedding is stored by saving the ${id}$ of the closest centroid and then compressing the residual (i.e., the element-wise difference) by using 1 or 2 bits per component. ColBERTv2 saves up to ${10} \times$ space compared to ColBERT while being significantly more inefficient on modern CPUs, requiring up to 3 seconds to perform query processing on CPU [19]. The reduction of query processing time is achieved by Santhanam et al. with PLAID [19]. PLAID takes advantage of the embedding compressor of ColBERTv2 and also uses the centroid-based representation to discard non-relevant passages (centroid interaction [19]), thus performing the late interaction exclusively on a carefully selected batch of passages. PLAID allows for massive speedup compared to ColBERTv2, but its average query latency can be up to 400 msec. on CPU with single-thread execution [19].

早期的多向量检索系统，如 ColBERT [14]，利用倒排索引来存储嵌入并检索候选段落。然后，检索段落的表示并用于计算与查询的最大相似度得分。尽管这种方法相当高效，但它需要在内存中维护每个文档词项的全精度表示。在 MS MARCO [17]（一种广泛用于段落检索的基准数据集）上，ColBERT 使用的整个嵌入集合需要超过 140 GB [14] 的存储空间。ColBERTv2 [20] 引入了一种基于质心的压缩技术来高效存储段落嵌入。每个嵌入通过保存最接近质心的 ${id}$ 并使用每个分量 1 或 2 位来压缩残差（即逐元素差异）来存储。与 ColBERT 相比，ColBERTv2 最多可节省 ${10} \times$ 的空间，但在现代 CPU 上效率明显较低，在 CPU 上执行查询处理最多需要 3 秒 [19]。桑塔纳姆等人通过 PLAID [19] 实现了查询处理时间的减少。PLAID 利用了 ColBERTv2 的嵌入压缩器，还使用基于质心的表示来丢弃不相关的段落（质心交互 [19]），从而仅在精心选择的一批段落上进行后期交互。与 ColBERTv2 相比，PLAID 可以实现大幅加速，但其在 CPU 上单线程执行时的平均查询延迟可达 400 毫秒 [19]。

This paper presents EMVB, a novel framework for efficient query processing with multi-vector dense retrieval. First, we identify the most time-consuming steps of PLAID. These steps are i) extracting the top-nprobe closest centroids for each query term during the candidate passage selection, ii) computing the centroid interaction mechanism, and iii) decompression of the quantized residuals. Our method tackles the first and the second steps by introducing a highly efficient passage filtering approach based on optimized bit vectors. Our filter identifies a small set of crucial centroid scores, thus tearing down the cost of top-nprobe extraction. At the same time, it reduces the amount of passages for which we have to compute the centroid interaction. Moreover, we introduce a highly efficient column-wise reduction exploiting SIMD instructions to speed up this step. Finally, we improve the efficiency of the late interaction by introducing Product Quantization (PQ) [9]. PQ allows to obtain in pair or superior performance compared to the bitwise compressor of PLAID while being up to $3 \times$ faster. Finally, to further improve the efficiency of the last step of our pipeline, we introduce a dynamic passage-term-selection criterion for late interaction, thus reducing the cost of this step up to ${30}\%$ .

本文提出了EMVB，这是一个用于多向量密集检索的高效查询处理的新颖框架。首先，我们确定了PLAID中最耗时的步骤。这些步骤包括：i) 在候选段落选择过程中，为每个查询词提取最接近的前nprobe个质心；ii) 计算质心交互机制；iii) 对量化残差进行解压缩。我们的方法通过引入一种基于优化位向量的高效段落过滤方法来处理第一步和第二步。我们的过滤器识别出一小部分关键的质心分数，从而降低了提取前nprobe个质心的成本。同时，它减少了我们必须计算质心交互的段落数量。此外，我们引入了一种利用单指令多数据（SIMD）指令的高效列向归约方法来加速这一步骤。最后，我们通过引入乘积量化（PQ）[9]提高了后期交互的效率。与PLAID的按位压缩器相比，PQ能够获得相当或更优的性能，同时速度提高了$3 \times$。最后，为了进一步提高我们流程最后一步的效率，我们引入了一种用于后期交互的动态段落 - 词项选择标准，从而将这一步骤的成本降低了${30}\%$。

---

<!-- Footnote -->

${}^{3}$ the terms "document" and "passage" are used interchangeably in this paper.

${}^{3}$ 在本文中，“文档”和“段落”这两个术语可以互换使用。

<!-- Footnote -->

---

We experimentally evaluate EMVB against PLAID on two datasets: MS MARCO passage [17] (for in-domain evaluation) and LoTTE [20] (for out-of-domain evaluation). Results on MS MARCO show that EMVB is up to ${2.8} \times$ faster while reducing the memory footprint by ${1.8} \times$ with no loss in retrieval accuracy compared to PLAID. On the out-of-domain evaluation, EMVB delivers up to ${2.9} \times$ speedup compared to PLAID,with a minimal loss in retrieval quality.

我们在两个数据集上对EMVB和PLAID进行了实验评估：MS MARCO段落数据集[17]（用于领域内评估）和LoTTE数据集[20]（用于领域外评估）。MS MARCO数据集上的结果表明，与PLAID相比，EMVB的速度提高了${2.8} \times$，同时内存占用减少了${1.8} \times$，且检索准确率没有损失。在领域外评估中，与PLAID相比，EMVB的速度提高了${2.9} \times$，而检索质量仅有极小的损失。

The rest of this paper is organized as follows. In Section 2, we discuss the related work. In Section 3 we describe PLAID [19], the current state-of-the-art in multi-vector dense retrieval. We introduce EMVB in Section 4 and we experimentally evaluate it against PLAID in Section 5. Finally, Section 6 concludes our work.

本文的其余部分组织如下。在第2节中，我们讨论相关工作。在第3节中，我们描述PLAID [19]，这是当前多向量密集检索的最先进技术。我们在第4节中介绍EMVB，并在第5节中对其与PLAID进行实验评估。最后，第6节对我们的工作进行总结。

## 2 Related Work

## 2 相关工作

Dense retrieval encoders can be broadly classified into single-vector and multi-vector techniques. Single-vector encoders allow the encoding of an entire passage in a single dense vector [11]. In this line, ANCE [25] and STAR/ADORE [26] employ hard negatives to improve the training of dense retrievers by teaching them to distinguish between lexically-similar positive and negative passages. Multi-vector encoders have been introduced with ColBERT. The limitations of ColBERT and the efforts done to overcome them (ColBERTv2, PLAID) have been discussed in Section 1. COIL [6] rediscover the lessons of classical retrieval systems (e.g., BM25) by limiting the token interactions to lexical matching between queries and documents. CITADEL [16] is a recently proposed approach that introduces conditional token interaction by using dynamic lexical routing. Conditional token interaction means that the relevance of the query of a specific passage is estimated by only looking at some of their tokens. These tokens are selected by the so-called lexical routing, where a module of the ranking architecture is trained to determine which of the keys, i.e., words in the vocabulary, are activated by a query/passage. CITADEL significantly reduces the execution time on GPU,but turns out to be $2 \times$ slower than PLAID con CPU,at the same retrieval quality. Multi-vector dense retrieval is also exploited in conjunction with pseudo-relevance feedback both in ColBERT-PRF [23] and in CWPRF [22], showing that their combination boosts the effectiveness of the model.

密集检索编码器可以大致分为单向量和多向量技术。单向量编码器允许将整个段落编码为单个密集向量[11]。在这方面，ANCE [25]和STAR/ADORE [26]采用硬负样本，通过教导密集检索器区分词汇上相似的正样本和负样本段落来改进其训练。多向量编码器随着ColBERT的出现而引入。ColBERT的局限性以及为克服这些局限性所做的努力（ColBERTv2、PLAID）已在第1节中讨论过。COIL [6]通过将词元交互限制为查询和文档之间的词汇匹配，重新发掘了经典检索系统（如BM25）的经验。CITADEL [16]是最近提出的一种方法，它通过使用动态词汇路由引入了条件词元交互。条件词元交互意味着仅通过查看特定段落的部分词元来估计该段落与查询的相关性。这些词元由所谓的词汇路由选择，其中排名架构的一个模块经过训练，以确定词汇表中的哪些键（即单词）会被查询/段落激活。CITADEL显著减少了在GPU上的执行时间，但在相同的检索质量下，在CPU上的速度比PLAID慢$2 \times$。在ColBERT - PRF [23]和CWPRF [22]中，多向量密集检索还与伪相关反馈结合使用，表明它们的结合提高了模型的有效性。

Our Contribution: This work advances the state of the art of multi-vector dense retrieval by introducing EMVB, a novel framework that allows to speed up the retrieval performance of the PLAID pipeline significantly. To the best of our knowledge, this work is the first in the literature that proposes a highly efficient document filtering approach based on optimized bit vectors, a column-wise SIMD reduction to retrieve candidate passages and a late interaction mechanism that combines product quantization with a per-document term filtering.

我们的贡献：这项工作通过引入EMVB推动了多向量密集检索的技术发展，EMVB是一个新颖的框架，能够显著提高PLAID流程的检索性能。据我们所知，这项工作是文献中首次提出基于优化位向量的高效文档过滤方法、用于检索候选段落的列向SIMD归约方法，以及一种将乘积量化与按文档词项过滤相结合的后期交互机制。

## 3 Multi-vector Dense Retrieval

## 3 多向量密集检索

Consider a passage corpus $\mathcal{P}$ with ${n}_{P}$ passages. In a multi-vector dense retrieval scenario,an LLM encodes each token in $\mathcal{P}$ into a dense $d$ -dimensional vector ${T}_{j}$ . For each passage $P$ ,a dense representation $P = \left\{  {T}_{j}\right\}$ ,with $j = 0,\ldots ,{n}_{t}$ ,is produced,where ${n}_{t}$ is the number of tokens in the passage $P$ . Employing a token-level dense representation allows for boosting the effectiveness of the retrieval systems [14 20 19]. On the other hand, it produces significantly large collections of $d$ -dimensional vectors posing challenges to the applicability of such systems in real-world search scenarios both in terms of space (memory requirements) and time (latency of the query processor). To tackle the problem of memory requirements, ColBERTv2 [20] and successively PLAID [19] exploit a centroid-based vector compression technique. First, the K-means algorithm is employed to devise a clustering of the $d$ -dimensional space by identifying the set of $k$ centroids $\mathcal{C} = {\left\{  {C}_{i}\right\}  }_{i = 1}^{{n}_{c}}$ . Then,for each vector $x$ ,the residual $r$ between $x$ and its closest centroid $\bar{C}$ is computed so that $r = x - \bar{C}$ . The residual $r$ is compressed into $\widetilde{r}$ using a $b$ -bit encoder that represents each dimension of $r$ using $b$ bits,with $b \in  \{ 1,2\}$ . The memory impact of storing a $d$ -dimensional vector is given by $\left\lceil  {{\log }_{2}\left| C\right| }\right\rceil$ bits for the centroid index and $d \times  b$ bits for the compressed residual. This approach requires a time-expensive decompression phase to restore the approximate full-precision vector representation given the centroid id and the residual coding. For this reason, PLAID aims at decompressing as few candidate documents as possible. This is achieved by introducing a high-quality filtering step based on the centroid-approximated embedding representation, named centroid interaction [19]. In detail, the PLAID retrieval engine is composed of four different phases [19]. The first one regards the retrieval of the candidate passages. A list of candidate passages is built for each centroid. A passage belongs to a centroid ${C}_{i}$ candidate list if one or more tokens have ${C}_{i}$ as its closest centroid. For each query term ${q}_{i}$ ,with $i = 1,\ldots ,{n}_{q}$ ,the top- ${nprobe}$ closest centroids are computed, according to the dot product similarity measure. The set of unique documents associated with the top-nprobe centroids then moves to a second phase that acts as a filtering phase. In this phase,a token embedding ${T}_{j}$ with $j = 1,\ldots ,{n}_{p}$ is approximated using its closest centroid ${\bar{C}}^{{T}_{j}}$ . Hence,its distance with the $i$ -th query term ${q}_{i}$ is approximated with

考虑一个包含 ${n}_{P}$ 个段落的段落语料库 $\mathcal{P}$。在多向量密集检索场景中，大语言模型（LLM）将 $\mathcal{P}$ 中的每个标记编码为一个密集的 $d$ 维向量 ${T}_{j}$。对于每个段落 $P$，会生成一个密集表示 $P = \left\{  {T}_{j}\right\}$，其中 $j = 0,\ldots ,{n}_{t}$，这里 ${n}_{t}$ 是段落 $P$ 中的标记数量。采用标记级别的密集表示有助于提高检索系统的有效性 [14 20 19]。另一方面，它会生成大量的 $d$ 维向量集合，这在空间（内存需求）和时间（查询处理器的延迟）方面给此类系统在实际搜索场景中的应用带来了挑战。为了解决内存需求问题，ColBERTv2 [20] 以及后续的 PLAID [19] 采用了基于质心的向量压缩技术。首先，使用 K - 均值算法通过识别 $k$ 个质心的集合 $\mathcal{C} = {\left\{  {C}_{i}\right\}  }_{i = 1}^{{n}_{c}}$ 来对 $d$ 维空间进行聚类。然后，对于每个向量 $x$，计算 $x$ 与其最近质心 $\bar{C}$ 之间的残差 $r$，使得 $r = x - \bar{C}$。使用一个 $b$ 位编码器将残差 $r$ 压缩为 $\widetilde{r}$，该编码器使用 $b$ 位表示 $r$ 的每个维度，其中 $b \in  \{ 1,2\}$。存储一个 $d$ 维向量的内存影响由质心索引的 $\left\lceil  {{\log }_{2}\left| C\right| }\right\rceil$ 位和压缩残差的 $d \times  b$ 位给出。这种方法需要一个耗时的解压缩阶段，以便在给定质心 ID 和残差编码的情况下恢复近似的全精度向量表示。因此，PLAID 旨在尽可能少地解压缩候选文档。这是通过引入一个基于质心近似嵌入表示的高质量过滤步骤来实现的，该步骤称为质心交互 [19]。详细来说，PLAID 检索引擎由四个不同的阶段组成 [19]。第一个阶段是候选段落的检索。为每个质心构建一个候选段落列表。如果一个段落中的一个或多个标记的最近质心是 ${C}_{i}$，则该段落属于质心 ${C}_{i}$ 的候选列表。对于每个查询词 ${q}_{i}$，其中 $i = 1,\ldots ,{n}_{q}$，根据点积相似度度量计算最接近的前 ${nprobe}$ 个质心。与前 nprobe 个质心相关联的唯一文档集合随后进入作为过滤阶段的第二阶段。在这个阶段，使用其最近质心 ${\bar{C}}^{{T}_{j}}$ 来近似具有 $j = 1,\ldots ,{n}_{p}$ 的标记嵌入 ${T}_{j}$。因此，它与第 $i$ 个查询词 ${q}_{i}$ 的距离近似为

$$
{q}_{i} \cdot  {T}_{j} \simeq  {q}_{i} \cdot  {\bar{C}}^{{T}_{j}} = {\widetilde{T}}_{i,j}. \tag{1}
$$

Consider a candidate passage $P$ composed of ${n}_{p}$ tokens. The approximated score of $P$ consists in computing the dot product ${q}_{i} \cdot  {\bar{C}}^{{T}_{j}}$ for all the query terms ${q}_{i}$ and all the closest centroids of each token belonging to the passage, i.e.,

考虑一个由 ${n}_{p}$ 个标记组成的候选段落 $P$。$P$ 的近似得分包括为所有查询词 ${q}_{i}$ 以及属于该段落的每个标记的所有最近质心计算点积 ${q}_{i} \cdot  {\bar{C}}^{{T}_{j}}$，即

$$
{\bar{S}}_{q,P} = \mathop{\sum }\limits_{{i = 1}}^{{n}_{q}}\mathop{\max }\limits_{{j = 1\ldots {n}_{t}}}{q}_{i} \cdot  {\bar{C}}^{{T}_{j}} \tag{2}
$$

The third phase, named decompression, aims at reconstructing the full-precision representation of $P$ by combining the centroids and the residuals. This is done on the top-ndocs passages selected according to the filtering phase [19]. In the fourth phase, PLAID recomputes the final score of each passage with respect to the query $q$ using the decompressed-full-precision-representation according to late interaction mechanism (Equation 3). Passages are then ranked according to their similarity score and the top- $k$ passages are selected.

第三个阶段称为解压缩阶段，旨在通过组合质心和残差来重建 $P$ 的全精度表示。这是在根据过滤阶段选择的前 ndocs 个段落上进行的 [19]。在第四个阶段，PLAID 根据后期交互机制（公式 3）使用解压缩后的全精度表示重新计算每个段落相对于查询 $q$ 的最终得分。然后根据段落的相似度得分对段落进行排序，并选择前 $k$ 个段落。

$$
{S}_{q,P} = \mathop{\sum }\limits_{{i = 1}}^{{n}_{q}}\mathop{\max }\limits_{{j = 1\ldots {n}_{t}}}{q}_{i} \cdot  {T}_{j} \tag{3}
$$

PLAID execution time. We provide a breakdown of PLAID execution time across its different phases, namely retrieval, flltering, decompression, and late interaction. This experiment is conducted using the experimental settings detailed in Section 5. We report the execution time for different values of $k$ ,i.e., the number of retrieved passages.

PLAID执行时间。我们详细分析了PLAID在不同阶段的执行时间，即检索、过滤、解压缩和后期交互。本实验采用第5节中详细描述的实验设置进行。我们报告了不同$k$值（即检索到的段落数量）下的执行时间。

<!-- Media -->

<!-- figureText: k = 10 Retrieval Filtering Decompression Late Interaction 150 200 250 300 Execution time (msec.) k = 100 k = 1000 50 100 -->

<img src="https://cdn.noedgeai.com/0195ce97-67db-7928-9893-d85f4ae01a94_4.jpg?x=392&y=934&w=1015&h=334&r=0"/>

Fig. 1: Breakdown of the PLAID average query latency (in milliseconds) on CPU across its four phases.

图1：PLAID在CPU上四个阶段的平均查询延迟（以毫秒为单位）细分。

<!-- Media -->

## 4 EMVB

## 4 EMVB

We now present EMVB, our novel framework for efficient multi-vector dense retrieval. First, EMVB introduces a highly efficient pre-filtering phase that exploits optimized bit vectors. Second, we improve the efficiency of the centroid interaction step (Equation 1) by introducing column-wise max reduction with SIMD instructions. Third, EMVB leverages Product Quantization (PQ) to reduce the memory footprint of storing the vector representations while jointly allowing for a fast late interaction phase. Fourth, PQ is applied in conjunction with a novel per-passage term filtering approach that allows for further improving the efficiency of the late interaction. In the following subsections, we detail these four contributions behind EMVB.

我们现在介绍EMVB，这是我们提出的用于高效多向量密集检索的新颖框架。首先，EMVB引入了一个高效的预过滤阶段，该阶段利用了优化的位向量。其次，我们通过引入使用单指令多数据（SIMD）指令的逐列最大缩减方法，提高了质心交互步骤（公式1）的效率。第三，EMVB利用乘积量化（Product Quantization，PQ）来减少存储向量表示的内存占用，同时实现快速的后期交互阶段。第四，PQ与一种新颖的逐段落词过滤方法结合使用，进一步提高了后期交互的效率。在以下小节中，我们将详细介绍EMVB背后的这四项贡献。

### 4.1 Retrieval of Candidate Passages

### 4.1 候选段落的检索

Figure 1 shows that a consistent part of the computation required by PLAID is spent on the retrieval phase. We further break down these steps to evidence its most time-consuming part. The retrieval consists of i) computing the distances between the incoming query and the set of centroids, ii) extracting the top-nprobe closest centroids for each query term. The former step is efficiently carried out by leveraging high-performance matrix multiplication tools (e.g., Intel MKL [1821]). In the latter step, PLAID extracts the top-nprobe centroids using the numpy topk function, which implements the quickselect algorithm. Selecting the top-nprobe within the $\left| C\right|  = {2}^{18}$ centroids for each of the ${n}_{q}$ query terms costs up to $3 \times$ the matrix multiplication done in the first step. In Section 4.2, we show that our pre-filtering inherently speeds up the top-nprobe selection by tearing down the number of evaluated elements. In practice, we show how to efficiently filter out those centroids whose score is below a certain threshold and then execute quickselect exclusively on the surviving ones. As a consequence, in EMVB the cost of the top-nprobe extraction becomes negligible, being two orders of magnitude faster than the top-nprobe extraction on the full set of centroids.

图1显示，PLAID所需计算量中有相当一部分花在了检索阶段。我们进一步细分这些步骤，以找出最耗时的部分。检索包括：i）计算传入查询与质心集合之间的距离；ii）为每个查询词提取最接近的前nprobe个质心。前一步骤通过利用高性能矩阵乘法工具（例如英特尔数学核心库（Intel MKL [1821]））高效完成。在后一步骤中，PLAID使用numpy的topk函数提取前nprobe个质心，该函数实现了快速选择算法。为${n}_{q}$个查询词中的每个词在$\left| C\right|  = {2}^{18}$个质心中选择前nprobe个质心的成本高达第一步中矩阵乘法的$3 \times$倍。在4.2节中，我们表明我们的预过滤方法通过减少评估元素的数量，从本质上加快了前nprobe个质心的选择过程。实际上，我们展示了如何高效地过滤掉那些得分低于某个阈值的质心，然后仅对剩余的质心执行快速选择算法。因此，在EMVB中，提取前nprobe个质心的成本变得可以忽略不计，比在完整质心集合上提取前nprobe个质心快两个数量级。

### 4.2 Efficient Pre-Filtering of Candidate Passages

### 4.2 候选段落的高效预过滤

Figure 1 shows that the candidate filtering phase can be significantly expensive, especially for large values of $k$ . In this section,we propose a pre-filtering approach based on a novel bit vector representation of the centroids that efficiently allows the discarding of non-relevant passages.

图1显示，候选过滤阶段可能会非常耗时，尤其是在$k$值较大的情况下。在本节中，我们提出了一种基于质心的新颖位向量表示的预过滤方法，该方法可以高效地丢弃不相关的段落。

Given a passage $P$ ,our pre-filtering consists in determining whether ${\widetilde{T}}_{i,j}$ , for $i = 1,\ldots ,{n}_{q},j = 1,\ldots ,{n}_{t}$ is large or not. Recall that ${\widetilde{T}}_{i,j}$ represents the approximate score of the $j$ -th token of passage $P$ with respect to the $i$ -th term of the query ${q}_{i}$ ,as defined in Equation 1 . This can be obtained by checking whether ${\bar{C}}_{j}^{T}$ -the centroid associated with ${T}_{j}$ -belongs to the set of the closest centroids of ${q}_{i}$ . We introduce close ${}_{i}^{th}$ ,the set of centroids whose scores are greater than a certain threshold ${th}$ with respect to a query term ${q}_{i}$ . Given a passage $P$ ,we define the list of centroids ids ${I}_{P}$ ,where ${I}_{P}^{j}$ is the centroid id of ${\bar{C}}^{{T}_{j}}$ . The similarity of a passage with respect to a query can be estimated with our novel filtering function $F\left( {P,q}\right)  \in  \left\lbrack  {0,{n}_{q}}\right\rbrack$ with the following equation:

给定一个段落$P$，我们的预过滤过程是确定对于$i = 1,\ldots ,{n}_{q},j = 1,\ldots ,{n}_{t}$，${\widetilde{T}}_{i,j}$是否较大。回想一下，${\widetilde{T}}_{i,j}$表示段落$P$的第$j$个标记相对于查询${q}_{i}$的第$i$个词的近似得分，如公式1所定义。这可以通过检查与${T}_{j}$关联的质心${\bar{C}}_{j}^{T}$是否属于${q}_{i}$的最接近质心集合来实现。我们引入close ${}_{i}^{th}$，即相对于查询词${q}_{i}$得分大于某个阈值${th}$的质心集合。给定一个段落$P$，我们定义质心ID列表${I}_{P}$，其中${I}_{P}^{j}$是${\bar{C}}^{{T}_{j}}$的质心ID。段落相对于查询的相似度可以通过我们新颖的过滤函数$F\left( {P,q}\right)  \in  \left\lbrack  {0,{n}_{q}}\right\rbrack$用以下公式进行估计：

$$
F\left( {P,q}\right)  = \mathop{\sum }\limits_{{i = 1}}^{{n}_{q}}\mathbf{1}\left( {\exists j\text{ s.t. }{I}_{P}^{j} \in  {\operatorname{close}}_{i}^{th}}\right) . \tag{4}
$$

For a passage $P$ ,this counts how many query terms have at least one similar passage term in $P$ ,where "similar" describes the belonging of ${T}_{j}$ to ${\operatorname{close}}_{i}^{th}$ .

对于一个段落$P$，这计算了有多少个查询词在$P$中至少有一个相似的段落词，其中“相似”描述了${T}_{j}$属于${\operatorname{close}}_{i}^{th}$。

In Figure 2 (left), we compare the performance of our novel pre-filter working on top of the centroid interaction mechanism (orange, blue, green lines) against the performance of the centroid interaction mechanism on the entire set of candidate documents (red dashed line) on the MS MARCO dataset. The plot shows that our pre-filtering allows to efficiently discard non-relevant passages without harming the recall of the successive centroid interaction phase. For example, we can narrow the candidate passage set to just 1000 elements using ${th} = {0.4}$ without any loss in R@100.In the remainder of this section, we show how to implement this pre-filter efficiently.

在图2（左）中，我们在MS MARCO数据集上，将基于质心交互机制之上运行的新型预过滤器（橙色、蓝色、绿色线条）的性能与在整个候选文档集上运行的质心交互机制（红色虚线）的性能进行了比较。该图显示，我们的预过滤能够在不影响后续质心交互阶段召回率的情况下，有效剔除不相关的段落。例如，我们可以使用${th} = {0.4}$将候选段落集缩小到仅1000个元素，而不会损失R@100指标。在本节的其余部分，我们将展示如何高效实现此预过滤器。

<!-- Media -->

<!-- figureText: 0.8650 10.0 Naive IF Time (ms) 7.5 Branchless VecBranchless 5.0 Vectorized IF 2.5 0.2 0.4 0.6 Threshold value R@100 0.8625 0.8600 0.3 0.5 0.4 w/o pre-filtering 0.8575 1000 2000 3000 4000 5000 #Scored Passages -->

<img src="https://cdn.noedgeai.com/0195ce97-67db-7928-9893-d85f4ae01a94_6.jpg?x=391&y=338&w=1020&h=300&r=0"/>

Fig. 2: R@100 with various values of the threshold (left). Comparison of different algorithms to construct close ${}_{i}^{th}$ ,for different values of ${th}$ (right).

图2：不同阈值下的R@100（左）。针对不同的${th}$值，构建近邻${}_{i}^{th}$的不同算法的比较（右）。

<!-- Media -->

Building the bit vectors. Given ${th}$ ,the problem of computing close ${}_{i}^{th}$ is conceptually simple. Yet, an efficient implementation carefully considering modern CPUs' features is crucial for fast computation of Equation 4.

构建位向量。给定${th}$，计算近邻${}_{i}^{th}$的问题在概念上很简单。然而，为了快速计算方程4，仔细考虑现代CPU特性的高效实现至关重要。

Let ${CS} = q \cdot  {C}^{T}$ ,with ${CS} \in  {\left\lbrack  -1,1\right\rbrack  }^{{n}_{q} \times  \left| C\right| }$ be the score matrix between the query $q$ and the set of centroids $C$ (both matrices are ${L}_{2}$ normalized),where ${n}_{q}$ is the number of query tokens,and $\left| C\right|$ is the number of centroids. In the naïve if-based solution,we scan the $i$ -th row of ${CS}$ and select those $j$ s.t. $C{S}_{i,j} > {th}$ . It is possible to speed up this approach by taking advantage of SIMD instructions. In particular, the _mm512_cmp_epi32_mask allows one to compare 16 fp32 values at a time and store the comparison result in a mask variable. If mask $=  =$ 0 , we can skip to the successive 16 values because the comparison has failed for all the current $j$ s. Otherwise,we extract those indexes $J = \left\{  {j \in  \left\lbrack  {0,{15}}\right\rbrack   \mid  {\operatorname{mask}}_{j} = 1}\right\}$ .

设${CS} = q \cdot  {C}^{T}$，其中${CS} \in  {\left\lbrack  -1,1\right\rbrack  }^{{n}_{q} \times  \left| C\right| }$是查询$q$与质心集$C$之间的得分矩阵（两个矩阵均经过${L}_{2}$归一化），其中${n}_{q}$是查询词元的数量，$\left| C\right|$是质心的数量。在基于简单if语句的解决方案中，我们扫描${CS}$的第$i$行，并选择那些满足$C{S}_{i,j} > {th}$的$j$。可以通过利用单指令多数据（SIMD）指令来加速此方法。具体而言，_mm512_cmp_epi32_mask指令允许一次比较16个fp32值，并将比较结果存储在一个掩码变量中。如果掩码$=  =$为0，我们可以跳过接下来的16个值，因为当前所有$j$的比较都失败了。否则，我们提取那些索引$J = \left\{  {j \in  \left\lbrack  {0,{15}}\right\rbrack   \mid  {\operatorname{mask}}_{j} = 1}\right\}$。

The efficiency of such if-based algorithms mainly depends on the branch mis-prediction ratio. Modern CPUs speculate on the outcome of the if before the condition itself is computed by recognizing patterns in the execution flow of the algorithm. When the wrong branch is predicted, a control hazard happens, and the pipeline is flushed with a delay of 15-20 clock cycles,i.e.,about ${10}\mathrm{{ns}}$ . We tackle the inefficiency of branch misprediction by proposing a branchless algorithm. The branchless algorithm employs a pointer $p$ addressing a pre-allocated buffer. While scanning $C{S}_{i,j}$ ,it writes $j$ in the position indicated by $p$ . Then,it sums to $p$ the result of the comparison: 1 if $C{S}_{i,j} > {th},0$ otherwise. At the successive iteration, if the result of the comparison was $0,j + 1$ will override $j$ . Otherwise,it will be written in the successive memory location,and $j$ will be saved in the buffer. The branchless selection does not present any if instruction and consequently does not contain any branch in its execution flow. The branchless algorithm can be implemented more efficiently by leveraging SIMD instructions. In particular, the above-mentioned _mm512_cmp_epi32_mask instruction allows to compare 16 fp32 values at the time, and the _mm512_mask_compressstore allows to extract $J$ in a single instruction.

这种基于if语句的算法的效率主要取决于分支预测错误率。现代CPU通过识别算法执行流程中的模式，在条件本身被计算之前对if语句的结果进行推测。当预测了错误的分支时，会发生控制冒险，并且流水线会被刷新，延迟15 - 20个时钟周期，即约${10}\mathrm{{ns}}$。我们通过提出一种无分支算法来解决分支预测错误导致的效率问题。无分支算法使用一个指针$p$指向一个预分配的缓冲区。在扫描$C{S}_{i,j}$时，它将$j$写入$p$所指示的位置。然后，它将比较结果累加到$p$上：如果$C{S}_{i,j} > {th},0$则加1，否则不加。在后续迭代中，如果比较结果为$0,j + 1$，则会覆盖$j$。否则，它将被写入后续的内存位置，并且$j$将被保存在缓冲区中。无分支选择不包含任何if指令，因此在其执行流程中不包含任何分支。通过利用SIMD指令，可以更高效地实现无分支算法。具体而言，上述的_mm512_cmp_epi32_mask指令允许一次比较16个fp32值，而_mm512_mask_compressstore指令允许在一条指令中提取$J$。

Figure 2 (right) presents a comparison of our different approaches, namely "Naïve IF", the "Vectorized IF", the "Branchless", and the "VecBranchless" described above. Branchless algorithms present a constant execution time, regardless of the value of the threshold, while if-based approaches offer better performances as the value of ${th}$ increases. With ${th} \geq  {0.3}$ ,"Vectorized IF" is the most efficient approach,with a speedup up to $3 \times$ compared to its naïve counterpart.

图2（右）展示了我们不同方法的比较，即上述的“简单IF”、“向量化IF”、“无分支”和“向量化无分支”方法。无分支算法的执行时间是恒定的，与阈值的值无关，而基于if语句的方法随着${th}$值的增加表现更好。当${th} \geq  {0.3}$时，“向量化IF”是最有效的方法，与简单版本相比，速度提升高达$3 \times$。

Fast set membership. Once ${\operatorname{close}}_{i}^{th}$ is computed,we have to efficiently compute Equation 4. Here,given ${I}_{P}$ as a list of integers,we have to test if at least one of its members ${I}_{P}^{j}$ belongs to ${\operatorname{close}}_{i}^{th}$ ,with $i = 1,\ldots ,{n}_{q}$ . This can be efficiently done using bit vectors for representing ${\operatorname{close}}_{i}^{th}$ . A bit vector maps a set of integers up to $N$ into an array of $N$ bits,where the $e$ -th bit is set to one if and only if the integer $e$ belongs to the set. Adding and searching any integer $e$ can be performed in constant time with bit manipulation operators. Moreover, bit vectors require $N$ bits to be stored. In our case,since we have $\left| C\right|  = {2}^{18}$ ,a bit vector only requires ${32K}$ bytes to be stored.

快速集合成员查询。一旦计算出${\operatorname{close}}_{i}^{th}$，我们就需要高效地计算方程4。这里，给定作为整数列表的${I}_{P}$，我们需要测试其成员${I}_{P}^{j}$中是否至少有一个属于${\operatorname{close}}_{i}^{th}$，其中$i = 1,\ldots ,{n}_{q}$。这可以通过使用位向量来表示${\operatorname{close}}_{i}^{th}$高效地完成。位向量将一个最大为$N$的整数集合映射到一个包含$N$位的数组中，当且仅当整数$e$属于该集合时，第$e$位被置为1。使用位操作运算符，可以在常数时间内完成对任意整数$e$的添加和查找操作。此外，位向量需要存储$N$位。在我们的例子中，由于我们有$\left| C\right|  = {2}^{18}$，一个位向量只需要存储${32K}$字节。

Since we search through all the ${n}_{q}$ bit vectors at a time,we can further exploit the bit vector representation by stacking the bit vectors vertically (Figure 3). This allows to search a centroid index through all the close ${}_{i}^{th}$ at a time. The bits corresponding to the same centroid for different query terms are consecutive and fit a 32-bit word. This way, we can simultaneously test the membership for all the queries in constant time with a single bitwise operation. In detail, our algorithm works by initializing a mask $m$ of ${n}_{q} = {32}$ bits at zeros (Step 1,Figure 3). Then, for each term in the candidate documents, it performs a bitwise xor between the mask and the 32-bit word representing the membership to all the query terms (Step 2, Figure 3). Hence, Equation 4 can be obtained by counting the number of 1s in $m$ at the end of the execution with the popcnt operation featured by modern CPUs (Step 3, Figure 3).

由于我们一次要遍历所有的${n}_{q}$个位向量，我们可以通过将位向量垂直堆叠（图3）来进一步利用位向量表示法。这使得我们可以一次性在所有临近的${}_{i}^{th}$中搜索质心索引。不同查询词对应同一质心的位是连续的，并且可以放入一个32位字中。这样，我们可以通过一次位操作在常数时间内同时测试所有查询的成员关系。具体来说，我们的算法通过将一个${n}_{q} = {32}$位的掩码$m$初始化为零（图3步骤1）来工作。然后，对于候选文档中的每个词，它在掩码和表示所有查询词成员关系的32位字之间执行按位异或操作（图3步骤2）。因此，方程4可以通过在执行结束时使用现代CPU具备的popcnt操作对$m$中1的数量进行计数来得到（图3步骤3）。

Figure 4 (up) shows that our "Vectorized" set membership implementation delivers a speedup ranging from ${10} \times$ to ${16} \times$ a "Baseline" relying on a naïve usage of bit vectors. In particular, our bit vector-based pre-filtering can be up to ${30} \times$ faster than the centroid-interaction proposed in PLAID [19],cf. Figure 4 (down).

图4（上）显示，我们的“向量化”集合成员查询实现相对于依赖简单位向量使用的“基线”方法，加速比在${10} \times$到${16} \times$之间。特别是，我们基于位向量的预过滤方法比PLAID [19]中提出的质心交互方法快达${30} \times$倍，参见图4（下）。

### 4.3 Fast Filtering of Candidate Passages

### 4.3 候选段落的快速过滤

Our pre-filtering approach allows us to efficiently filter out non-relevant passages and is employed upstream of PLAID's centroid interaction (Equation 2). We now show how to improve the efficiency of the centroid interaction itself.

我们的预过滤方法使我们能够高效地过滤掉不相关的段落，并应用于PLAID质心交互（方程2）的上游。现在我们展示如何提高质心交互本身的效率。

Consider a passage $P$ and its associated centroid scores matrix $\widetilde{P} = {q}_{i} \cdot  {\bar{C}}^{{T}_{j}}$ . Explicitly building this matrix allows to reuse it in the scoring phase, in place of the costly decompression step (Section 4.4). To build $\widetilde{P}$ ,we transpose ${CS}$ into $C{S}^{T}$ of size $\left| C\right|  \times  {n}_{q}$ . The $i$ -th row of $C{S}^{T}$ allows access to all the ${n}_{q}$ query terms scores for the $i$ -th centroids. Given the ids of the closest centroids for each passage term (defined as ${I}_{P}$ in Section 4.2) we retrieve the scores for each centroid id. We build ${\widetilde{P}}^{T} - \widetilde{P}$ transposed -to allow the CPU to read and write contiguous memory locations. This grants more than $2 \times$ speedup compared to processing $\widetilde{P}$ . We now have ${\widetilde{P}}^{T}$ of shape ${n}_{t} \times  {n}_{q}$ . We need to max-reduce along the columns and then sum the obtained values to implement Equation 2. This is done by iterating on the ${\widetilde{P}}^{T}$ rows and packing them into AVX512 registers. Given that ${n}_{q} = {32}$ ,each AVX512 register can contain ${512}/{32} = {16}$ floating point values, so we need 2 registers for each row. We pack the first row into max_ $l$ and max $h$ . All the successive rows are packed into current_ $l$ and current_h. At each iteration,we compare max_l with current_l and max_h with current_ $h$ using the_mm512_cmp_ps_mask AVX512 instruction described before. The output mask $m$ is used to update the max $l$ and max $h$ by employing the _mm512_mask_blend_ps instruction. The _mm512_cmp_ps_mask has throughput 2 on IceLake Xeon CPUs,so each row of $\widetilde{P}$ is compared with max_ $l$ and max $h$ in the same clock cycle,on two different ports. The same holds for the _mm512_mask_blend_ps instruction, entailing that the max-reduce operation happens in 2 clock cycles without considering the memory loading. Finally, ${max}\_ l$ and ${max}\_ h$ are summed together,and the function _mm512_reduce_add_ps is used to ultimate the computation.

考虑一个段落 $P$ 及其关联的质心得分矩阵 $\widetilde{P} = {q}_{i} \cdot  {\bar{C}}^{{T}_{j}}$。显式构建此矩阵允许在评分阶段重用它，以替代代价高昂的解压缩步骤（第 4.4 节）。为了构建 $\widetilde{P}$，我们将 ${CS}$ 转置为大小为 $\left| C\right|  \times  {n}_{q}$ 的 $C{S}^{T}$。$C{S}^{T}$ 的第 $i$ 行允许访问第 $i$ 个质心的所有 ${n}_{q}$ 查询词得分。给定每个段落词的最近质心的 ID（在第 4.2 节中定义为 ${I}_{P}$），我们检索每个质心 ID 的得分。我们构建转置后的 ${\widetilde{P}}^{T} - \widetilde{P}$，以便 CPU 可以读写连续的内存位置。与处理 $\widetilde{P}$ 相比，这可实现超过 $2 \times$ 的加速。现在我们有形状为 ${n}_{t} \times  {n}_{q}$ 的 ${\widetilde{P}}^{T}$。我们需要沿列进行最大归约，然后对得到的值求和以实现方程 2。这是通过迭代 ${\widetilde{P}}^{T}$ 的行并将它们打包到 AVX512 寄存器中来完成的。鉴于 ${n}_{q} = {32}$，每个 AVX512 寄存器可以包含 ${512}/{32} = {16}$ 个浮点值，因此每行需要 2 个寄存器。我们将第一行打包到 max_ $l$ 和 max $h$ 中。所有后续行都打包到 current_ $l$ 和 current_h 中。在每次迭代中，我们使用之前描述的 _mm512_cmp_ps_mask AVX512 指令将 max_l 与 current_l 以及 max_h 与 current_ $h$ 进行比较。输出掩码 $m$ 用于通过使用 _mm512_mask_blend_ps 指令更新 max $l$ 和 max $h$。_mm512_cmp_ps_mask 在 IceLake Xeon CPU 上的吞吐量为 2，因此 $\widetilde{P}$ 的每一行在同一时钟周期内在两个不同端口上与 max_ $l$ 和 max $h$ 进行比较。_mm512_mask_blend_ps 指令也是如此，这意味着在不考虑内存加载的情况下，最大归约操作在 2 个时钟周期内完成。最后，将 ${max}\_ l$ 和 ${max}\_ h$ 相加，并使用 _mm512_reduce_add_ps 函数完成计算。

<!-- Media -->

<!-- figureText: Stacked ① m = 0 xor xor $\begin{array}{llll} 0 & 0 & 1 & 0 \end{array}$ $\begin{array}{llll} 0 & 1 & 1 & 0 \end{array}$ $F = \operatorname{popcnt}\left( m\right)$ Bit Vectors 32 bit word -->

<img src="https://cdn.noedgeai.com/0195ce97-67db-7928-9893-d85f4ae01a94_8.jpg?x=392&y=355&w=418&h=368&r=0"/>

Fig. 3: Vectorized Fast Set Membership algorithm based on bit vectors.

图 3：基于位向量的矢量化快速集合成员算法。

<!-- figureText: Time per 1.5 20 Baseline Vectorized 0.4 0.6 th 20 10 2000 4000 6000 8000 10000 #Candidate Documents 1.0 0.5 0.2 PLAID Ours -->

<img src="https://cdn.noedgeai.com/0195ce97-67db-7928-9893-d85f4ae01a94_8.jpg?x=898&y=333&w=518&h=385&r=0"/>

Fig. 4: Vectorized vs naïve Fast Set Membership (up). Ours vs PLAID filtering (down).

图 4：矢量化与朴素快速集合成员算法（上）。我们的方法与 PLAID 过滤算法（下）。

<!-- Media -->

We implement PLAID’s centroid interaction in $\mathrm{C} +  +$ and we compare its filtering time against our SIMD-based solution. The results of the comparison are reported for different values of candidate documents in Figure 4 (down). Thanks to the proficient read-write pattern and the highly efficient column-wise max-reduction,our method can be up to ${1.8} \times$ faster than the filtering proposed in PLAID.

我们在 $\mathrm{C} +  +$ 中实现了 PLAID 的质心交互，并将其过滤时间与我们基于 SIMD 的解决方案进行比较。图 4（下）报告了不同候选文档值的比较结果。由于高效的读写模式和高效的按列最大归约，我们的方法比 PLAID 中提出的过滤方法快达 ${1.8} \times$。

### 4.4 Late Interaction

### 4.4 后期交互

The $b$ -bit residual compressor proposed in previous approaches [20 19] requires a costly decompression step before the late interaction phase. Figure 1 shows that in PLAID decompressing the vectors costs up to $5 \times$ the late interaction phase.

先前方法 [20 19] 中提出的 $b$ 位残差压缩器在后期交互阶段之前需要一个代价高昂的解压缩步骤。图 1 显示，在 PLAID 中解压缩向量的成本高达后期交互阶段的 $5 \times$。

We propose compressing the residual $r$ by employing Product Quantization (PQ) [9]. PQ allows the computation of the dot product between an input query vector $q$ and the compressed residual ${r}_{pq}$ without decompression. Consider a query $q$ and a candidate passage $P$ . We decompose the computation of the max similarity operator (Equation 3) into

我们建议通过采用乘积量化（PQ）[9] 来压缩残差 $r$。PQ 允许在不进行解压缩的情况下计算输入查询向量 $q$ 与压缩残差 ${r}_{pq}$ 之间的点积。考虑一个查询 $q$ 和一个候选段落 $P$。我们将最大相似度运算符的计算（方程 3）分解为

$$
{S}_{q,P} = \mathop{\sum }\limits_{{i = 1}}^{{n}_{q}}\mathop{\max }\limits_{{j = 1\ldots {n}_{t}}}\left( {{q}_{i} \cdot  {\bar{C}}^{{T}_{j}} + {q}_{i} \cdot  {r}^{{T}_{j}}}\right)  \simeq  \mathop{\sum }\limits_{{i = 1}}^{{n}_{q}}\mathop{\max }\limits_{{j = 1\ldots {n}_{t}}}\left( {{q}_{i} \cdot  {\bar{C}}^{{T}_{j}} + {q}_{i} \cdot  {r}_{pq}^{{T}_{j}}}\right) , \tag{5}
$$

where and ${r}^{{T}_{j}} = {T}_{j} - {\bar{C}}^{{T}_{j}}$ . On the one hand,this decomposition allows to exploit the pre-computed $\widetilde{P}$ matrix. On the other hand,thanks to PQ,it computes the dot product between the query and the residuals without decompression.

其中  和 ${r}^{{T}_{j}} = {T}_{j} - {\bar{C}}^{{T}_{j}}$。一方面，这种分解允许利用预先计算的 $\widetilde{P}$ 矩阵。另一方面，由于使用了 PQ，它可以在不进行解压缩的情况下计算查询与残差之间的点积。

We replace PLAID's residual compression with PQ, particularly with JMPQ [4] which optimizes the codes of product quantization during the fine-tuning of the language model for the retrieval task. We tested $m = \{ {16},{32}\}$ ,where $m$ is the number of sub-spaces used to partition the vectors [9]. We experimentally verify that PQ reduces the latency of the late interaction phase up to ${3.6} \times$ compared to PLAID $b$ -bit compressor. Moreover,it delivers the same $\left( {m = {16}}\right)$ or superior performance $\left( {m = {32}}\right)$ in terms of MRR@10 when leveraging the JMPQ version.

我们用乘积量化（PQ）取代了PLAID（渐进式低秩近似索引解码）的残差压缩，特别是采用了JMPQ [4]，它在为检索任务微调语言模型期间优化了乘积量化的编码。我们测试了$m = \{ {16},{32}\}$，其中$m$是用于划分向量的子空间数量 [9]。我们通过实验验证，与PLAID $b$位压缩器相比，PQ将后期交互阶段的延迟降低了多达${3.6} \times$。此外，在使用JMPQ版本时，它在MRR@10方面实现了相同的$\left( {m = {16}}\right)$或更优的性能$\left( {m = {32}}\right)$。

We propose to further improve the efficiency of the scoring phase by hinging on the properties of Equation 5. We experimentally observe that, in many cases, ${q}_{i} \cdot  {\bar{C}}_{j}^{T} > {q}_{i} \cdot  {r}_{pq}^{{T}_{j}}$ ,meaning that the max operator on $j$ ,in many cases,is lead by the score between the query term and the centroid, rather than the score between the query term and the residual. We argue that it is possible to compute the scores on the residuals only for a reduced set of document terms ${\bar{J}}_{i}$ ,where $i$ identifies the index of the query term. In particular, ${\bar{J}}_{i} = \left\{  {j \mid  {q}_{i} \cdot  {\bar{C}}_{j}^{T} > t{h}_{r}}\right\}$ , where $t{h}_{r}$ is a second threshold that determines whether the score with the centroid is sufficiently large. With the introduction of this new per-term filter, Equation 5 now becomes computing the max operator on the set of passages in ${\bar{J}}_{i}$ ,i.e.,

我们提议通过利用公式5的特性进一步提高评分阶段的效率。我们通过实验观察到，在许多情况下，${q}_{i} \cdot  {\bar{C}}_{j}^{T} > {q}_{i} \cdot  {r}_{pq}^{{T}_{j}}$，这意味着在许多情况下，$j$上的最大值运算符由查询词与质心之间的得分主导，而不是查询词与残差之间的得分。我们认为，仅针对文档词项的一个缩减集合${\bar{J}}_{i}$计算残差得分是可行的，其中$i$标识查询词的索引。具体而言，${\bar{J}}_{i} = \left\{  {j \mid  {q}_{i} \cdot  {\bar{C}}_{j}^{T} > t{h}_{r}}\right\}$，其中$t{h}_{r}$是一个第二阈值，用于确定与质心的得分是否足够大。随着这个新的逐词过滤器的引入，公式5现在变为计算${\bar{J}}_{i}$中段落集合上的最大值运算符，即

$$
{S}_{q,P} = \mathop{\sum }\limits_{{i = 1}}^{{n}_{q}}\mathop{\max }\limits_{{j \in  {\bar{J}}_{i}}}\left( {{q}_{i} \cdot  {\bar{C}}^{{T}_{j}} + {q}_{i} \cdot  {r}_{pq}^{{T}_{j}}}\right) . \tag{6}
$$

In practice, we compute the residual scores only for those document terms whose centroid score is large enough. If ${\bar{J}}_{i} = \varnothing$ ,we compute ${S}_{q,P}$ as in Equation 5 . Figure 5 (left) reports the effectiveness of our approach. On the $y$ -axis,we report the percentage of the original effectiveness, computed as the ratio between the MRR@10 computed with Equation 6 and Equation 5. Filtering document terms according to Equation 6 does not harm the retrieval quality, as it delivers substantially the same MRR@10 of Equation 5. On the right side of Figure 5, we report the percentage of scored terms compared to the number of document terms computed using Equation 5 . With $t{h}_{r} = {0.5}$ ,we are able to reduce the number of scored terms of at least ${30}\%$ (right) without any performance degradation in terms of MRR@10.

在实践中，我们仅为那些质心得分足够高的文档词项计算残差得分。如果${\bar{J}}_{i} = \varnothing$，我们按照公式5计算${S}_{q,P}$。图5（左）展示了我们方法的有效性。在$y$轴上，我们报告了原始有效性的百分比，计算方法是公式6计算的MRR@10与公式5计算的MRR@10之比。根据公式6过滤文档词项不会损害检索质量，因为它实现了与公式5基本相同的MRR@10。在图5的右侧，我们报告了与使用公式5计算的文档词项数量相比，有得分的词项的百分比。使用$t{h}_{r} = {0.5}$，我们能够将有得分的词项数量至少减少${30}\%$（右），而在MRR@10方面没有任何性能下降。

<!-- Media -->

<!-- figureText: % original effectiveness 1.000 % scored terms 0.6 0.5 0.4 0.3 0.6 0.4 0.5 0.6 thr 0.999 #Docs 64 512 0.998 128 1024 256 2048 0.4 0.5 thr -->

<img src="https://cdn.noedgeai.com/0195ce97-67db-7928-9893-d85f4ae01a94_10.jpg?x=393&y=343&w=1018&h=333&r=0"/>

Fig. 5: Performance of our dynamic term-selection filtering for different values of $t{h}_{r}$ ,in terms of percentage of original effectiveness (left) and in terms of percentage of original number of scored terms (right). The percentage of original effectiveness is computed as the ratio between the MRR@10 computed with Equation 6 and Equation 5.

图5：我们的动态词项选择过滤在不同$t{h}_{r}$值下的性能，以原始有效性百分比（左）和原始有得分词项数量百分比（右）表示。原始有效性百分比计算为公式6计算的MRR@10与公式5计算的MRR@10之比。

<!-- Media -->

## 5 Experimental Evaluation

## 5 实验评估

Experimental Settings. This section compares our methodology against the state-of-the-art engine for multi-vector dense retrieval, namely PLAID [19]. We conduct experiments on the MS MARCO passages dataset [17] for the in-domain evaluation and on LoTTE [20] for the out-of-domain evaluation. We generate the embeddings for MS MARCO using the ColBERTv2 model. The generated dataset is composed of about ${600}\mathrm{M}d$ -dimensional vectors,with $d = {128}$ . Product Quantization is implemented using the FAISS [10] library, and optimized using the JMPQ technique [4] on MS MARCO. The implementation of EMBV is available on Github ${}^{4}$ . We compare EMVB against the original PLAID implementation [19], which also implements its core components in C++. Experiments are conducted on an Intel Xeon Gold 5318Y CPU clocked at 2.10 GHz, equipped with the AVX512 instruction set, with single-thread execution. Code is compiled using GCC 11.3.0 (with -03 compilation options) on a Linux 5.15.0-72 machine. When running experiments with AVX512 instruction on 512-bit registers, we ensure not to incur in the frequency scaling down event reported for Intel CPUs [15].

实验设置。本节将我们的方法与最先进的多向量密集检索引擎，即PLAID [19]进行比较。我们在MS MARCO段落数据集[17]上进行领域内评估实验，在LoTTE [20]上进行领域外评估实验。我们使用ColBERTv2模型为MS MARCO生成嵌入向量。生成的数据集由约${600}\mathrm{M}d$维向量组成，其中$d = {128}$。乘积量化使用FAISS [10]库实现，并在MS MARCO上使用JMPQ技术[4]进行优化。EMBV的实现可在Github ${}^{4}$上获取。我们将EMBV与原始的PLAID实现[19]进行比较，后者也用C++实现了其核心组件。实验在时钟频率为2.10 GHz的英特尔至强金牌5318Y CPU上进行，该CPU配备AVX512指令集，采用单线程执行。代码在Linux 5.15.0 - 72机器上使用GCC 11.3.0（使用 - 03编译选项）进行编译。当在512位寄存器上使用AVX512指令运行实验时，我们确保不会遇到英特尔CPU报告的频率降频事件[15]。

Evaluation. Table 1 compares EMVB against PLAID on the MS MARCO dataset, in terms of memory requirements (num. of bytes per embedding), average query latency (in milliseconds), MRR@10, and Recall@100, and 1000.

评估。表1在内存需求（每个嵌入的字节数）、平均查询延迟（以毫秒为单位）、MRR@10、Recall@100和Recall@1000方面，将EMVB与PLAID在MS MARCO数据集上进行了比较。

---

<!-- Footnote -->

${}^{4}$ https://github.com/CosimoRulli/emvb

${}^{4}$ https://github.com/CosimoRulli/emvb

<!-- Footnote -->

---

Franco Maria Nardini, Cosimo Rulli, and Rossano Venturini

佛朗哥·玛丽亚·纳尔迪尼（Franco Maria Nardini）、科西莫·鲁利（Cosimo Rulli）和罗萨诺·文图里尼（Rossano Venturini）

<!-- Media -->

<table><tr><td>$k$</td><td>Method</td><td>Latency (msec.) Bytes MRR@10 R@100 R@1000</td><td/><td/><td/><td/></tr><tr><td rowspan="3">10</td><td>PLAID</td><td>131</td><td>36</td><td>39.4</td><td>-</td><td>-</td></tr><tr><td>EMVB (m=16)</td><td>62 (2.1x)</td><td>20</td><td>39.4</td><td>-</td><td>-</td></tr><tr><td>EMVB (m=32)</td><td>61 $\left( {{2.1} \times  }\right)$</td><td>36</td><td>39.7</td><td>-</td><td>-</td></tr><tr><td rowspan="3">100</td><td>PLAID</td><td>180</td><td>36</td><td>39.8</td><td>90.6</td><td>-</td></tr><tr><td>EMVB (m=16)</td><td>${68}\left( {{2.6} \times  }\right)$</td><td>20</td><td>39.5</td><td>90.7</td><td>-</td></tr><tr><td>EMVB (m=32)</td><td>80 $\left( {{2.3} \times  }\right)$</td><td>36</td><td>39.9</td><td>90.7</td><td>-</td></tr><tr><td rowspan="3"/><td>PLAID</td><td>260</td><td>36</td><td>39.8</td><td>91.3</td><td>97.5</td></tr><tr><td>1000 EMVB (m=1)</td><td>93 (2.8x)</td><td>20</td><td>39.5</td><td>91.4</td><td>97.5</td></tr><tr><td>EMVB (m=32)</td><td>104 $\left( {{2.5} \times  }\right)$</td><td>36</td><td>39.9</td><td>91.4</td><td>97.5</td></tr></table>

<table><tbody><tr><td>$k$</td><td>方法</td><td>延迟（毫秒） 字节数 前10召回率均值（MRR@10） 前100召回率（R@100） 前1000召回率（R@1000）</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">10</td><td>PLAID</td><td>131</td><td>36</td><td>39.4</td><td>-</td><td>-</td></tr><tr><td>增强多向量块（EMVB）（m = 16）</td><td>62 (2.1x)</td><td>20</td><td>39.4</td><td>-</td><td>-</td></tr><tr><td>增强多向量块（EMVB）（m = 32）</td><td>61 $\left( {{2.1} \times  }\right)$</td><td>36</td><td>39.7</td><td>-</td><td>-</td></tr><tr><td rowspan="3">100</td><td>PLAID</td><td>180</td><td>36</td><td>39.8</td><td>90.6</td><td>-</td></tr><tr><td>增强多向量块（EMVB）（m = 16）</td><td>${68}\left( {{2.6} \times  }\right)$</td><td>20</td><td>39.5</td><td>90.7</td><td>-</td></tr><tr><td>增强多向量块（EMVB）（m = 32）</td><td>80 $\left( {{2.3} \times  }\right)$</td><td>36</td><td>39.9</td><td>90.7</td><td>-</td></tr><tr><td rowspan="3"></td><td>PLAID</td><td>260</td><td>36</td><td>39.8</td><td>91.3</td><td>97.5</td></tr><tr><td>1000 增强多向量块（EMVB）（m = 1）</td><td>93 (2.8x)</td><td>20</td><td>39.5</td><td>91.4</td><td>97.5</td></tr><tr><td>增强多向量块（EMVB）（m = 32）</td><td>104 $\left( {{2.5} \times  }\right)$</td><td>36</td><td>39.9</td><td>91.4</td><td>97.5</td></tr></tbody></table>

Table 1: Comparison between EMVB and PLAID in terms of average query latency, number of bytes per vector embeddings, MRR, and Recall on MS MARCO.

表1：EMVB和PLAID在平均查询延迟、每个向量嵌入的字节数、平均倒数排名（MRR）和微软机器阅读理解数据集（MS MARCO）上的召回率方面的比较。

<!-- Media -->

Results show that EMVB delivers superior performance along both the evaluated trade-offs. With $m = {16}$ ,EMVB almost halves the per-vector memory burden compared to PLAID,while being up to ${2.8} \times$ faster with almost no performance degradation regarding retrieval effectiveness. By doubling the number of sub-partitions per vector,i.e., $m = {32}$ ,EMVB outperforms the performance of PLAID in terms of MRR and Recall with the same memory footprint with up to ${2.5} \times$ speed up.

结果表明，在评估的两个权衡方面，EMVB都表现出卓越的性能。当$m = {16}$时，与PLAID相比，EMVB几乎将每个向量的内存负担减半，同时速度提高了${2.8} \times$，并且在检索效果方面几乎没有性能下降。通过将每个向量的子分区数量加倍，即$m = {32}$，EMVB在平均倒数排名（MRR）和召回率方面优于PLAID，在相同的内存占用情况下，速度提高了${2.5} \times$。

Table 2 compares EMVB and PLAID in the out-of-domain evaluation on the LoTTE dataset. As in PLAID [19], we employ Success@5 and Success@100 as retrieval quality metrics. On this dataset, EMVB offers slightly inferior performance in terms of retrieval quality. Recall that JMPQ [4] cannot be applied in the out-of-domain evaluation due to the lack of training queries. Instead, we employ Optimized Product Quantization (OPQ) [7], which searches for an optimal rotation of the dataset vectors to reduce the quality degradation that comes with PQ. To mitigate the retrieval quality loss, we only experiment PQ with $m = {32}$ ,given that an increased number of partitions offers a better representation of the original vector. On the other hand, EMVB can offer up to ${2.9} \times$ speedup compared to PLAID. This larger speedup compared to MS MARCO is due to the larger average document lengths in LoTTE. In this context, filtering nonrelevant documents using our bit vector-based approach has a remarkable impact on efficiency. Observe that for the out-of-domain evaluation, our pre-filtering method could be ingested into PLAID. This would allow to maintain the PLAID accuracy together with EMVB efficiency. Combinations of PLAID and EMVB are left for future work.

表2比较了EMVB和PLAID在LoTTE数据集上的域外评估情况。与PLAID [19]一样，我们采用前5名成功率（Success@5）和前100名成功率（Success@100）作为检索质量指标。在这个数据集上，EMVB在检索质量方面表现略逊一筹。请记住，由于缺乏训练查询，联合多投影量化（JMPQ）[4]无法应用于域外评估。相反，我们采用优化乘积量化（OPQ）[7]，它搜索数据集向量的最佳旋转，以减少乘积量化（PQ）带来的质量下降。为了减轻检索质量损失，考虑到增加分区数量可以更好地表示原始向量，我们仅对$m = {32}$的乘积量化（PQ）进行实验。另一方面，与PLAID相比，EMVB的速度可提高${2.9} \times$。与微软机器阅读理解数据集（MS MARCO）相比，这种更大的速度提升是由于LoTTE中平均文档长度更长。在这种情况下，使用我们基于位向量的方法过滤不相关文档对效率有显著影响。请注意，对于域外评估，我们的预过滤方法可以融入PLAID。这将允许在保持PLAID准确性的同时提高EMVB的效率。PLAID和EMVB的组合留待未来研究。

## 6 Conclusion

## 6 结论

We presented EMVB, a novel framework for efficient multi-vector dense retrieval. EMVB advances PLAID, the current state-of-the-art approach, by introducing four novel contributions. First, EMVB employs a highly efficient pre-filtering step of passages using optimized bit vectors for speeding up the candidate passage filtering phase. Second, the computation of the centroid interaction is carried out with reduced precision. Third, EMVB leverages Product Quantization to reduce the memory footprint of storing vector representations while jointly allowing for fast late interaction. Fourth, we introduce a per-passage term filter for late interaction, thus reducing the cost of this step of up to 30%. We experimentally evaluate EMVB against PLAID on two publicly available datasets, i.e., MS MARCO and LoTTE. Results show that, in the in-domain evaluation, EMVB is up to ${2.8} \times$ faster,and it reduces by ${1.8} \times$ the memory footprint with no loss in retrieval quality compared to PLAID. In the out-of-domain evaluation, EMVB is up to ${2.9} \times$ faster with little or no retrieval quality degradation.

我们提出了高效多向量密集检索的新型框架EMVB。EMVB通过四项创新改进了当前最先进的方法PLAID。首先，EMVB采用优化位向量对段落进行高效预过滤，加速候选段落过滤阶段。其次，以降低的精度进行质心交互计算。第三，EMVB利用乘积量化减少存储向量表示的内存占用，同时实现快速后期交互。第四，我们引入了用于后期交互的逐段落术语过滤器，将此步骤的成本降低了30%。我们在两个公开数据集（微软机器阅读理解数据集MS MARCO和LoTTE）上对EMVB和PLAID进行了实验评估。结果表明，在域内评估中，与PLAID相比，EMVB速度提高了${2.8} \times$，并将内存占用减少了${1.8} \times$，且检索质量没有损失。在域外评估中，EMVB速度提高了${2.9} \times$，检索质量几乎没有下降。

<!-- Media -->

<table><tr><td>$k$</td><td>Method</td><td/><td/><td/><td>Success@100</td></tr><tr><td rowspan="2">10</td><td>PLAID</td><td>131</td><td>36</td><td>69.1</td><td>-</td></tr><tr><td>EMVB (m=32)</td><td>82 (1.6x)</td><td>36</td><td>69.0</td><td>-</td></tr><tr><td rowspan="2">100</td><td>PLAID</td><td>202</td><td>36</td><td>69.4</td><td>89.9</td></tr><tr><td>EMVB (m=32)</td><td>129(1.6x)</td><td>36</td><td>69.0</td><td>89.9</td></tr><tr><td rowspan="2">1000</td><td>PLAID</td><td>411</td><td>36</td><td>69.6</td><td>90.5</td></tr><tr><td>EMVB (m=32)</td><td>${42}\left( {{2.9} \times  }\right)$</td><td>36</td><td>69.0</td><td>90.1</td></tr></table>

<table><tbody><tr><td>$k$</td><td>方法</td><td></td><td></td><td></td><td>前100准确率（Success@100）</td></tr><tr><td rowspan="2">10</td><td>PLAID（原文未明确含义，保留英文）</td><td>131</td><td>36</td><td>69.1</td><td>-</td></tr><tr><td>增强型多视图块（Enhanced Multi-View Block，EMVB） (m=32)</td><td>82 (1.6x)</td><td>36</td><td>69.0</td><td>-</td></tr><tr><td rowspan="2">100</td><td>PLAID（原文未明确含义，保留英文）</td><td>202</td><td>36</td><td>69.4</td><td>89.9</td></tr><tr><td>增强型多视图块（Enhanced Multi-View Block，EMVB） (m=32)</td><td>129(1.6x)</td><td>36</td><td>69.0</td><td>89.9</td></tr><tr><td rowspan="2">1000</td><td>PLAID（原文未明确含义，保留英文）</td><td>411</td><td>36</td><td>69.6</td><td>90.5</td></tr><tr><td>增强型多视图块（Enhanced Multi-View Block，EMVB） (m=32)</td><td>${42}\left( {{2.9} \times  }\right)$</td><td>36</td><td>69.0</td><td>90.1</td></tr></tbody></table>

Table 2: Comparison between EMVB and PLAID in terms of average query latency, number of bytes per vector embeddings, Success@5, and Success@100 on LoTTE.

表2：在LoTTE数据集上，EMVB和PLAID在平均查询延迟、每个向量嵌入的字节数、前5准确率（Success@5）和前100准确率（Success@100）方面的比较。

<!-- Media -->

Acknowledgements. This work was partially supported by the EU - NGEU, by the PNRR - M4C2 - Investimento 1.3, Partenariato Esteso PE00000013 - "FAIR - Future Artificial Intelligence Research" - Spoke 1 "Human-centered AI" funded by the European Commission under the NextGeneration EU program, by the PNRR ECS00000017 Tuscany Health Ecosystem Spoke 6 "Precision medicine & personalized healthcare", by the European Commission under the NextGenera-tion EU programme, by the Horizon Europe RIA "Extreme Food Risk Analytics" (EFRA), grant agreement n. 101093026, by the "Algorithms, Data Structures and Combinatorics for Machine Learning" (MIUR-PRIN 2017), and by the "Algorithmic Problems and Machine Learning" (MIUR-PRIN 2022).

致谢。这项工作部分得到了以下资助：欧盟下一代欧盟基金（EU - NGEU）；意大利国家复苏与韧性计划（PNRR）中的M4C2 - 投资1.3项目、扩展合作项目PE00000013 - “公平的未来人工智能研究”（FAIR - Future Artificial Intelligence Research） - 分支1 “以人为中心的人工智能”（Human - centered AI），该项目由欧盟委员会在下一代欧盟计划下资助；PNRR ECS00000017托斯卡纳健康生态系统分支6 “精准医学与个性化医疗”（Precision medicine & personalized healthcare），由欧盟委员会在下一代欧盟计划下资助；欧洲地平线研究与创新行动“极端食品风险分析”（Extreme Food Risk Analytics，EFRA），资助协议编号101093026；“机器学习的算法、数据结构和组合学”（Algorithms, Data Structures and Combinatorics for Machine Learning，MIUR - PRIN 2017）；以及“算法问题与机器学习”（Algorithmic Problems and Machine Learning，MIUR - PRIN 2022）。

## References

## 参考文献

1. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.D., Dhariwal, P., Nee-lakantan, A., Shyam, P., Sastry, G., Askell, A., et al.: Language models are few-shot learners. Advances in Neural Information Processing Systems (NIPS) (2020)

2. Bruch, S., Lucchese, C., Nardini, F.M.: Efficient and effective tree-based and neural learning to rank. Found. Trends Inf. Retr. (2023)

3. Dai, D., Sun, Y., Dong, L., Hao, Y., Sui, Z., Wei, F.: Why can gpt learn in-context? language models secretly perform gradient descent as meta optimizers. arXiv preprint arXiv:2212.10559 (2022)

4. Fang, Y., Zhan, J., Liu, Y., Mao, J., Zhang, M., Ma, S.: Joint optimization of multi-vector representation with product quantization. In: Natural Language Processing and Chinese Computing (2022)

5. Formal, T., Piwowarski, B., Clinchant, S.: Splade: Sparse lexical and expansion model for first stage ranking. In: Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (2021)

6. Gao, L., Dai, Z., Callan, J.: Coil: Revisit exact lexical match in information retrieval with contextualized inverted list. In: Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (2021)

7. Ge, T., He, K., Ke, Q., Sun, J.: Optimized product quantization. IEEE Transactions on Pattern Analysis and Machine Intelligence (2013)

8. Guu, K., Lee, K., Tung, Z., Pasupat, P., Chang, M.: Retrieval augmented language model pre-training. In: Proceedings of the International Conference on Machine Learning (ICML) (2020)

9. Jegou, H., Douze, M., Schmid, C.: Product quantization for nearest neighbor search. IEEE Transactions on Pattern Analysis and Machine Intelligence (2010)

10. Johnson, J., Douze, M., Jegou, H.: Billion-scale similarity search with gpus. IEEE Transactions on Big Data (2021)

11. Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., Yih, W.t.: Dense passage retrieval for open-domain question answering. In: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). Association for Computational Linguistics (2020)

12. Kenton, J.D.M.W.C., Toutanova, L.K.: Bert: Pre-training of deep bidirectional transformers for language understanding. In: Proceedings of NAACL-HLT (2019)

13. Khattab, O., Potts, C., Zaharia, M.: Baleen: Robust multi-hop reasoning at scale via condensed retrieval. Advances in Neural Information Processing Systems (NIPS) (2021)

14. Khattab, O., Zaharia, M.: Colbert: Efficient and effective passage search via con-textualized late interaction over bert. In: Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval. pp. ${39} - {48}\left( {2020}\right)$

15. Lemire, D., Downs, T.: Avx-512: when and how to use these new instructions (2023), https://lemire.me/blog/2018/09/07/ avx-512-when-and-how-to-use-these-new-instructions/

16. Li, M., Lin, S.C., Oguz, B., Ghoshal, A., Lin, J., Mehdad, Y., Yih, W.t., Chen, X.: Citadel: Conditional token interaction via dynamic lexical routing for efficient and effective multi-vector retrieval. arXiv e-prints (2022)

17. Nguyen, T., Rosenberg, M., Song, X., Gao, J., Tiwary, S., Majumder, R., Deng, L.: Ms marco: A human-generated machine reading comprehension dataset

18. Qian, G., Sural, S., Gu, Y., Pramanik, S.: Similarity between euclidean and cosine angle distance for nearest neighbor queries. In: Proceedings of the 2004 ACM symposium on Applied computing (2004)

19. Santhanam, K., Khattab, O., Potts, C., Zaharia, M.: Plaid: an efficient engine for late interaction retrieval. In: Proceedings of the 31st ACM International Conference on Information & Knowledge Management (2022)

20. Santhanam, K., Khattab, O., Saad-Falcon, J., Potts, C., Zaharia, M.: Colbertv2: Effective and efficient retrieval via lightweight late interaction. In: Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (2022)

21. Wang, E., Zhang, Q., Shen, B., Zhang, G., Lu, X., Wu, Q., Wang, Y.: Intel math kernel library. In: High-Performance Computing on the Intel(R) Xeon Phi ${}^{\mathrm{{TM}}}$ (2014)

22. Wang, X., MacAvaney, S., Macdonald, C., Ounis, I.: Effective contrastive weighting for dense query expansion. In: Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (2023)

23. Wang, X., Macdonald, C., Tonellotto, N., Ounis, I.: Colbert-prf: Semantic pseudo-relevance feedback for dense passage and document retrieval. ACM Transactions on the Web $\mathbf{{17}}\left( 1\right) ,1 - {39}\left( {2023}\right)$

24. Xiong, L., Xiong, C., Li, Y., Tang, K.F., Liu, J., Bennett, P.N., Ahmed, J., Over-wijk, A.: Approximate nearest neighbor negative contrastive learning for dense text retrieval. In: International Conference on Learning Representations

25. Xiong, L., Xiong, C., Li, Y., Tang, K.F., Liu, J., Bennett, P.N., Ahmed, J., Over-wijk, A.: Approximate nearest neighbor negative contrastive learning for dense text retrieval. In: International Conference on Learning Representations (2020)

26. Zhan, J., Mao, J., Liu, Y., Guo, J., Zhang, M., Ma, S.: Optimizing dense retrieval model training with hard negatives. In: Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (2021)

27. Zhan, J., Mao, J., Liu, Y., Guo, J., Zhang, M., Ma, S.: Learning discrete representations via constrained clustering for effective and efficient dense retrieval. In: Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining. pp. 1328-1336 (2022)