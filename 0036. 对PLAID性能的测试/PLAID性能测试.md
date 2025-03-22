# A Reproducibility Study of PLAID

# PLAID算法的可复现性研究

Sean MacAvaney

肖恩·麦卡瓦尼（Sean MacAvaney）

University of Glasgow

格拉斯哥大学（University of Glasgow）

Glasgow, United Kingdom

英国，格拉斯哥

sean.macavaney@glasgow.ac.uk

Nicola Tonellotto

尼古拉·托内洛托（Nicola Tonellotto）

University of Pisa

比萨大学（University of Pisa）

Pisa, Italy

意大利，比萨

nicola.tonellotto@unipi.it

## Abstract

## 摘要

The PLAID (Performance-optimized Late Interaction Driver) algorithm for ColBERTv2 uses clustered term representations to retrieve and progressively prune documents for final (exact) document scoring. In this paper, we reproduce and fill in missing gaps from the original work. By studying the parameters PLAID introduces, we find that its Pareto frontier is formed of a careful balance among its three parameters; deviations beyond the suggested settings can substantially increase latency without necessarily improving its effectiveness. We then compare PLAID with an important baseline missing from the paper: re-ranking a lexical system. We find that applying ColBERTv2 as a re-ranker atop an initial pool of BM25 results provides better efficiency-effectiveness trade-offs in low-latency settings. However, re-ranking cannot reach peak effectiveness at higher latency settings due to limitations in recall of lexical matching and provides a poor approximation of an exhaustive ColBERTv2 search. We find that recently proposed modifications to re-ranking that pull in the neighbors of top-scoring documents overcome this limitation, providing a Pareto frontier across all operational points for ColBERTv2 when evaluated using a well-annotated dataset. Curious about why re-ranking methods are highly competitive with PLAID, we analyze the token representation clusters PLAID uses for retrieval and find that most clusters are predominantly aligned with a single token and vice versa. Given the competitive trade-offs that re-ranking baselines exhibit, this work highlights the importance of carefully selecting pertinent baselines when evaluating the efficiency of retrieval engines.

用于ColBERTv2的PLAID（性能优化的后期交互驱动，Performance-optimized Late Interaction Driver）算法使用聚类词项表示来检索文档，并逐步筛选文档以进行最终（精确）的文档评分。在本文中，我们复现了该算法并填补了原研究中的空白。通过研究PLAID引入的参数，我们发现其帕累托前沿（Pareto frontier）是由三个参数之间的谨慎平衡形成的；超出建议设置的偏差可能会显著增加延迟，而不一定能提高其有效性。然后，我们将PLAID与论文中缺失的一个重要基线进行了比较：对词法系统进行重排序。我们发现，在BM25初始结果集的基础上应用ColBERTv2作为重排序器，在低延迟设置下能提供更好的效率 - 有效性权衡。然而，由于词法匹配召回率的限制，重排序在高延迟设置下无法达到最高有效性，并且不能很好地近似穷举式的ColBERTv2搜索。我们发现，最近提出的对重排序的改进方法，即引入得分最高文档的相邻文档，克服了这一限制，在使用标注良好的数据集进行评估时，为ColBERTv2在所有操作点上提供了帕累托前沿。由于好奇重排序方法为何能与PLAID具有高度竞争力，我们分析了PLAID用于检索的词元表示聚类，发现大多数聚类主要与单个词元对齐，反之亦然。鉴于重排序基线所展现出的有竞争力的权衡，这项工作强调了在评估检索引擎效率时仔细选择相关基线的重要性。

C https://github.com/seanmacavaney/plaidrepro

C https://github.com/seanmacavaney/plaidrepro

## CCS CONCEPTS

## 计算机协会概念分类体系（CCS CONCEPTS）

- Information systems $\rightarrow$ Retrieval models and ranking.

- 信息系统 $\rightarrow$ 检索模型与排序。

## KEYWORDS

## 关键词

Late Interaction, Efficiency, Reproducibility

后期交互、效率、可复现性

## ACM Reference Format:

## ACM引用格式：

Sean MacAvaney and Nicola Tonellotto. 2024. A Reproducibility Study of PLAID. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '24), July 14-18, 2024, Washington, DC, USA. ACM, New York, NY, USA, 9 pages. https: //doi.org/10.1145/3626772.3657856

肖恩·麦卡瓦尼（Sean MacAvaney）和尼古拉·托内洛托（Nicola Tonellotto）。2024年。PLAID算法的可复现性研究。收录于第47届ACM信息检索研究与发展国际会议（SIGIR '24）论文集，2024年7月14 - 18日，美国华盛顿特区。美国纽约州纽约市ACM协会，9页。https: //doi.org/10.1145/3626772.3657856

<!-- Media -->

<!-- figureText: 0.75 $\left( {2,{0.45},{4096}}\right) ,\ldots$ (2,0.3,8192) Latency (ms) (1,0.5,4096)。 $\left( {2,{0.45},{1024}}\right) \;\left( {4,{0.5},{1024}}\right)$ 0.70 0.65 0.60 -->

<img src="https://cdn.noedgeai.com/0195a4e0-ba45-7b07-899d-74f5bd2a9bdf_0.jpg?x=952&y=501&w=668&h=651&r=0"/>

Figure 1: The Pareto frontier of PLAID for ColBERTv2 on TREC DL 2019 over the three parameters it introduces (nprobe, ${t}_{CS}$ ,and ndocs). Several operational points are labeled to highlight the interdependence of PLAID's parameters.

图1：PLAID（性能优化的后期交互驱动程序）在TREC DL 2019上针对ColBERTv2模型在其引入的三个参数（nprobe、${t}_{CS}$和ndocs）下的帕累托前沿。标记了几个操作点以突出PLAID参数之间的相互依赖性。

<!-- Media -->

## 1 INTRODUCTION

## 1 引言

Relevance ranking is a central task in information retrieval. Numerous classes of models exist for the task, including lexical [21], dense [10], learned sparse [18], and late interaction [11]. While efficient exact top- $k$ retrieval algorithms exist for lexical and learned sparse retrieval systems (e.g., BlockMaxWAND [6]), dense and late interaction methods either need to perform expensive exhaustive scoring over the entire collection or resort to an approximation of top- $k$ retrieval. A myriad of approximate $k$ -nearest-neighbor approaches are available for (single-representation) dense models (e.g., HNSW [16]). However, these approaches generally do not apply directly to late interaction scoring mechanisms, so bespoke retrieval algorithms for late interaction models have been proposed.

相关性排序是信息检索中的核心任务。针对该任务存在众多类型的模型，包括词法模型[21]、稠密模型[10]、学习型稀疏模型[18]和后期交互模型[11]。虽然词法和学习型稀疏检索系统存在高效的精确前$k$检索算法（例如，BlockMaxWAND[6]），但稠密和后期交互方法要么需要对整个文档集合进行昂贵的穷举评分，要么只能采用前$k$检索的近似算法。对于（单表示）稠密模型，有大量近似的$k$最近邻方法可供使用（例如，HNSW[16]）。然而，这些方法通常不能直接应用于后期交互评分机制，因此已经有人提出了专门用于后期交互模型的检索算法。

PLAID (Performance-optimized Late Interaction Driver) [22] is one such retrieval algorithm. It is designed to efficiently retrieve and score documents for ColBERTv2 [23], a prominent late interaction model. PLAID first performs coarse-grained retrieval by matching the closest ColBERTv2 centroids (used for compressing term embeddings) to the query term embeddings. It then progressively filters the candidate documents by performing finer-grained estimations of a document's final relevance score. These filtering steps are controlled by three new parameters, which are discussed in more detail in Section 2.

PLAID（性能优化的后期交互驱动程序）[22]就是这样一种检索算法。它旨在为ColBERTv2[23]（一种著名的后期交互模型）高效地检索和评分文档。PLAID首先通过将最接近的ColBERTv2质心（用于压缩词项嵌入）与查询词项嵌入进行匹配来执行粗粒度检索。然后，它通过对文档的最终相关性得分进行更细粒度的估计，逐步过滤候选文档。这些过滤步骤由三个新参数控制，第2节将更详细地讨论这些参数。

The original PLAID paper answered several important research questions related to the overall effectiveness and efficiency compared to ColBERTv2's default retriever, the effect of the individual filtering stages, and its transferability. However, it also left several essential questions unanswered. This reproducibility paper aims to reproduce the core results of the paper and answer several additional questions. First, we explore the effect of PLAID's new parameters to better understand the practical decisions one must make when deploying a PLAID engine. Then we explore and report an important missing baseline (re-ranking a lexical retrieval system), which has been shown to be a highly-competitive approach for dense systems $\left\lbrack  {{12},{14}}\right\rbrack$ . Throughout our exploration,we also answer questions about how well PLAID applies to a dataset with many known relevant documents and how well it approximates an exhaustive ColBERTv2 search.

原始的PLAID论文回答了几个与整体有效性和效率（与ColBERTv2的默认检索器相比）、各个过滤阶段的效果以及其可迁移性相关的重要研究问题。然而，它也留下了几个关键问题未解答。这篇可复现性论文旨在复现该论文的核心结果并回答几个额外的问题。首先，我们探索PLAID新参数的影响，以便更好地理解在部署PLAID引擎时必须做出的实际决策。然后，我们探索并报告一个重要的缺失基线（对词法检索系统进行重排序），该方法已被证明是稠密系统$\left\lbrack  {{12},{14}}\right\rbrack$的一种极具竞争力的方法。在我们的探索过程中，我们还回答了关于PLAID在具有许多已知相关文档的数据集上的适用性如何，以及它对穷举式ColBERTv2搜索的近似程度如何等问题。

We find that PLAID's parameters need to be carefully set in conjunction with one another to avoid sub-optimal tradeoffs between effectiveness and efficiency. As shown in Figure 1, PLAID's Pareto frontier is a patchwork of parameter settings; changing one parameter without corresponding changes to the others can result in slower retrieval with no change to effectiveness. Further, we find that re-ranking lexical search results provides better efficiency-effectiveness trade-offs than PLAID in low-latency settings. For instance, competitive results can be achieved in a single-threaded setup in as low as $7\mathrm{{ms}}/$ query with re-ranking,compared to ${73}\mathrm{\;{ms}}$ /query for PLAID. Through an analysis of the token clusters, we confirm that a large proportion of tokens predominantly perform lexical matching, explaining why a lexical first stage is so competitive. We feel that our study provides important operational recommendations for those looking to use ColBERTv2 or similar models, both with and without the PLAID algorithm.

我们发现，PLAID的参数需要相互配合仔细设置，以避免在有效性和效率之间出现次优的权衡。如图1所示，PLAID的帕累托前沿是由一系列参数设置拼凑而成的；改变一个参数而不相应地改变其他参数可能会导致检索速度变慢，而有效性却没有变化。此外，我们发现，在低延迟设置下，对词法搜索结果进行重排序比PLAID能提供更好的效率 - 有效性权衡。例如，在单线程设置中，使用重排序方法在低至$7\mathrm{{ms}}/$次查询时就能取得有竞争力的结果，而PLAID则需要${73}\mathrm{\;{ms}}$次查询/每个查询。通过对词元簇的分析，我们证实了很大一部分词元主要进行词法匹配，这解释了为什么词法第一阶段如此具有竞争力。我们认为，我们的研究为那些希望使用ColBERTv2或类似模型的人（无论是否使用PLAID算法）提供了重要的操作建议。

## 2 BACKGROUND AND PRELIMINARIES

## 2 背景与预备知识

The late interaction class of ranking models applies a lightweight query-token to document-token "interaction" operator atop a con-textualized text encoder to estimate relevance between a query and a document. Perhaps most well-known is the ColBERT model [11], which applies a maximum-similarity operator over a pretrained transformer-based language model-though other late interaction operators (e.g., $\left\lbrack  {{15},{27}}\right\rbrack  )$ and contextualization strategies (e.g., $\left\lbrack  {5,9}\right\rbrack  )$ have also been proposed. Due to the nature of their scoring mechanism, late interaction models cannot efficiently identify the exact top- $k$ search results without an exhaustive scan over all documents, ${}^{1}$ nor can they directly use established approximate nearest neighbor algorithms. ${}^{2}$ Early work in late interaction approaches (e.g., $\left\lbrack  {5,9,{15}}\right\rbrack$ ) overcame this limitation through a re-ranking strategy, wherein a candidate set of documents are identified using an efficient first-stage lexical retriever such as BM25 [21]. Khattab and Zaharia [11] identified that this re-ranking strategy may be suboptimal, since the (lexically-matched) first stage results may not be aligned with those that the model will rank highly. Therefore, they proposed using approximate $k$ -nearest neighbour search over the token representations to identify documents to score instead.

排序模型的后期交互类在上下文文本编码器之上应用轻量级的查询词元到文档词元的“交互”算子，以估计查询与文档之间的相关性。也许最著名的是ColBERT模型 [11]，它在预训练的基于Transformer的语言模型上应用了最大相似度算子 —— 不过也有人提出了其他后期交互算子（例如 $\left\lbrack  {{15},{27}}\right\rbrack  )$）和上下文策略（例如 $\left\lbrack  {5,9}\right\rbrack  )$）。由于其评分机制的性质，后期交互模型如果不遍历所有文档，就无法高效地确定精确的前 $k$ 个搜索结果 ${}^{1}$，也无法直接使用已有的近似最近邻算法 ${}^{2}$。后期交互方法的早期工作（例如 $\left\lbrack  {5,9,{15}}\right\rbrack$）通过重排序策略克服了这一限制，即使用高效的第一阶段词法检索器（如BM25 [21]）来确定候选文档集。Khattab和Zaharia [11] 指出，这种重排序策略可能并非最优，因为（词法匹配的）第一阶段结果可能与模型高度排序的结果不一致。因此，他们建议对词元表示进行近似 $k$ 近邻搜索，以确定要评分的文档。

<!-- Media -->

<!-- figureText: Centroid Closest nprobe Pruned centroids Approx. scoring ndocs with threshold ${t}_{cs}$ Full scoring ndocs/4 pre-computation centroids (offline) identification -->

<img src="https://cdn.noedgeai.com/0195a4e0-ba45-7b07-899d-74f5bd2a9bdf_1.jpg?x=931&y=238&w=735&h=112&r=0"/>

Figure 2: The logical phases composing the candidate documents identification procedure in PLAID.

图2：PLAID中组成候选文档识别过程的逻辑阶段。

<!-- Media -->

To deal with the considerable space requirements to store the precomputed representations of document tokens, ColBERTv2 [23] implements a clustering solution to identify document token centroids that can be used to decompose a document token representation as a sum of a centroid and a quantized residual vector, reducing the storage requirements by one order of magnitude w.r.t. the original ColBERT. These cluster centroids can serve as proxies of document tokens $\left\lbrack  {{22},{24},{25}}\right\rbrack$ .

为了应对存储文档词元预计算表示所需的大量空间，ColBERTv2 [23] 实现了一种聚类解决方案，以确定文档词元质心，这些质心可用于将文档词元表示分解为质心和量化残差向量之和，相对于原始的ColBERT，将存储需求降低了一个数量级。这些聚类质心可以作为文档词元 $\left\lbrack  {{22},{24},{25}}\right\rbrack$ 的代理。

PLAID [22] further builds upon the centroids of ColBERTv2 to improve retrieval efficiency. PLAID selects and then progressively filters out candidate documents through three distinct phases, as illustrated in Figure 2. Firstly, given an encoded query token representation, its closest document token centroids are computed. The corresponding document identifiers are retrieved and merged together into a candidate set. The number of closest centroids to match per query token is a hyperparameter called nprobe. Naturally, the initial pool of documents increases in size as nprobe increases. Secondly, the set of candidate centroids is pruned by removing all centroids whose maximum similarity w.r.t. all query tokens is smaller than a threshold parameter ${t}_{cs}$ . Next,the pruned set of centroids is further pruned by selecting the top ndocs documents based on relevance scores computed with the late interactions mechanism on the unpruned centroids. Then, the top ndocs/ 4 approximately-scored documents are fully scored by decompressing the token representations and computing the exact ColBERTv2 relevance scores. Note that PLAID introduces a total of three hyperparameters, namely nprobe, ${t}_{cs}$ ,and ndocs. Although three suggested configurations of these settings were provided by the original PLAID paper, it does not explore the effects and inter-dependencies between them.

PLAID [22] 进一步基于ColBERTv2的质心来提高检索效率。如图2所示，PLAID通过三个不同的阶段选择并逐步过滤候选文档。首先，给定一个编码后的查询词元表示，计算其最接近的文档词元质心。检索相应的文档标识符并将它们合并到一个候选集中。每个查询词元要匹配的最接近质心的数量是一个名为nprobe的超参数。自然地，随着nprobe的增加，初始文档池的大小也会增加。其次，通过移除所有相对于所有查询词元的最大相似度小于阈值参数 ${t}_{cs}$ 的质心，对候选质心集进行修剪。接下来，根据在未修剪质心上使用后期交互机制计算的相关性得分，选择前ndocs个文档，对修剪后的质心集进一步修剪。然后，通过解压缩词元表示并计算精确的ColBERTv2相关性得分，对前ndocs/ 4个近似评分的文档进行全面评分。请注意，PLAID总共引入了三个超参数，即nprobe、${t}_{cs}$ 和ndocs。尽管原始的PLAID论文提供了这些设置的三种建议配置，但并未探讨它们之间的影响和相互依赖关系。

## 3 CORE RESULT REPRODUCTION

## 3 核心结果复现

We begin by reproducing the core results of PLAID. Specifically, we test that retrieving using PLAID's recommended operational points provides the absolute effectiveness and relative efficiency presented in the original paper. Given the limitation that the original paper experimented with sparsely-labeled evaluation sets, we test one sparsely-labeled dataset from the original paper and one dataset with more complete relevance assessments. We also add a new measurement that wasn't explored in the original work-the Rank-Biased Overlap (RBO) [26] with an exhaustive ColBERTv2 search-to test how good of an approximation PLAID is with respect to a complete search.

我们首先复现PLAID的核心结果。具体来说，我们测试使用PLAID推荐的操作点进行检索是否能提供原文中所呈现的绝对有效性和相对效率。鉴于原文仅在稀疏标注的评估集上进行了实验，我们测试了原文中的一个稀疏标注数据集和一个相关性评估更完整的数据集。我们还增加了一项原文未涉及的新指标 —— 与穷举式ColBERTv2搜索的排序偏差重叠度（Rank - Biased Overlap，RBO）[26]，以测试PLAID相对于完整搜索的近似程度。

Our experimental setup, detailed in the following section, includes both elements of both reproducibility and replicability per ACM’s definitions, ${}^{3}$ since we are a different team using some of the same artifacts (code, model, datasets, etc.), while also introducing other changes to the experimental setup (added an evaluation dataset, new measures, etc.).

我们的实验设置（将在下一节详细介绍）同时包含了ACM定义的可复现性和可重复性元素 ${}^{3}$，因为我们是不同的团队，使用了一些相同的工件（代码、模型、数据集等），同时也对实验设置进行了其他更改（增加了一个评估数据集、新的指标等）。

---

<!-- Footnote -->

${}^{1}$ In contrast,learned sparse models can be stored in inverted indices and retrieved using algorithms such as BlockMax-WAND [6].

${}^{1}$ 相比之下，学习型稀疏模型可以存储在倒排索引中，并使用诸如BlockMax - WAND [6] 之类的算法进行检索。

${}^{2}$ In contrast,single-representation dense retrieval models can be indexed and retrieved using algorithms such as HNSW [16].

${}^{2}$ 相比之下，单表示密集检索模型可以使用诸如HNSW [16] 之类的算法进行索引和检索。

${}^{3}$ https://www.acm.org/publications/policies/artifact-review-and-badging-current

${}^{3}$ https://www.acm.org/publications/policies/artifact-review-and-badging-current

<!-- Footnote -->

---

<!-- Media -->

Table 1: Suggested operational points from the original PLAID paper. We refer to them as (a), (b), and (c).

表1：原始PLAID论文中建议的操作要点。我们将它们称为（a）、（b）和（c）。

<table><tr><td>System</td><td>nprobe</td><td>${t}_{cs}$</td><td>ndocs</td></tr><tr><td>PLAID (a)</td><td>1</td><td>0.50</td><td>256</td></tr><tr><td>PLAID (b)</td><td>2</td><td>0.45</td><td>1024</td></tr><tr><td>PLAID (c)</td><td>4</td><td>0.40</td><td>4096</td></tr></table>

<table><tbody><tr><td>系统</td><td>网络探针（nprobe）</td><td>${t}_{cs}$</td><td>文档数量（ndocs）</td></tr><tr><td>普林斯顿负载分析接口数据库（PLAID）(a)</td><td>1</td><td>0.50</td><td>256</td></tr><tr><td>普林斯顿负载分析接口数据库（PLAID）(b)</td><td>2</td><td>0.45</td><td>1024</td></tr><tr><td>普林斯顿负载分析接口数据库（PLAID）(c)</td><td>4</td><td>0.40</td><td>4096</td></tr></tbody></table>

<!-- Media -->

### 3.1 Experimental Setup

### 3.1 实验设置

Model and Code. We reproduce PLAID starting form the released ColBERTv2 checkpoint ${}^{4}$ and the PLAID authors’ released code-base. ${}^{5}$ We release our modified version of the code and scripts to run our new experiments.

模型与代码。我们从发布的ColBERTv2检查点${}^{4}$和PLAID作者发布的代码库开始复现PLAID。${}^{5}$我们发布了修改后的代码版本和用于运行新实验的脚本。

Parameters. We use PLAID's recommended settings for the nprobe, ${t}_{cs}$ ,and ndocs parameters,as shown in Table 1. We refer to these operational settings as (a), (b), and (c) for simplicity, where each setting progressively filters fewer documents. PLAID performs a final top $k$ selection at the end of the process (i.e.,after fully scoring and sorting the filtered documents). We recognize that this step is unnecessary and only limits the apparent result set size. Therefore, in line with typical IR experimental procedures, we wet $k = {1000}$ across all settings. We also use the suggested settings of nbits=2 and nclusters=2 ${}^{18}$ .

参数。我们使用PLAID推荐的nprobe、${t}_{cs}$和ndocs参数设置，如表1所示。为简便起见，我们将这些操作设置分别称为(a)、(b)和(c)，其中每个设置逐步过滤的文档数量逐渐减少。PLAID在流程结束时（即对过滤后的文档进行全面评分和排序后）进行最终的前$k$选择。我们认识到这一步是不必要的，只会限制表面结果集的大小。因此，按照典型的信息检索（IR）实验流程，我们在所有设置下设置$k = {1000}$。我们还使用建议的nbits = 2和nclusters = 2设置${}^{18}$。

Baselines. We compare directly against the results reported by the original PLAID paper for our experimental settings (Table 3 in their paper). We further conducted an exhaustive search over ColBERTv ${2}^{6}$ to better contextualize the results and support the measurement of rank-biased overlap (described below).

基线。我们直接将实验设置的结果与原始PLAID论文报告的结果进行比较（他们论文中的表3）。我们还对ColBERTv${2}^{6}$进行了详尽搜索，以便更好地将结果置于上下文环境中，并支持对排名偏差重叠（下文描述）的测量。

Datasets. We evaluate on the MS MARCO v1 passage development dataset $\left\lbrack  {3,{19}}\right\rbrack$ ,which consists of 6,980 queries with sparse relevance assessments (1.1 per query). To make up for the limitations of these assessments, we also evaluate using the more comprehensive TREC DL 2019 dataset [4], which consists of 43 queries with 215 assessments per query. In line with the official task guidelines and the original PLAID paper, we do not augment the MS MARCO passage collection with titles [13].

数据集。我们在MS MARCO v1段落开发数据集$\left\lbrack  {3,{19}}\right\rbrack$上进行评估，该数据集包含6980个查询，带有稀疏的相关性评估（每个查询1.1个）。为弥补这些评估的局限性，我们还使用更全面的TREC DL 2019数据集[4]进行评估，该数据集包含43个查询，每个查询有215个评估。按照官方任务指南和原始PLAID论文，我们没有用标题扩充MS MARCO段落集合[13]。

Measures. For MS MARCO Dev, we evaluate using the official evaluation measure of mean Reciprocal Rank at depth 10 (RR@10), using MS MARCO's provided evaluation script. To understand the overall system's ability to retire the relevant passage, we measure the recall at depth 1000 (R@1k), which is also frequently used for the evaluation of Dev. To test how well PLAID approximates an exhaustive search, we measure Rank Biased Overlap (RBO) [26], with a persistence of 0.99 . We measure efficiency via the mean response time using a single CPU thread over the Dev set in milliseconds per query (ms/q). In line with the original paper, we only measure the time for retrieval, ignoring the time it takes to encode the query (which is identical across all approaches). For TREC DL 2019, we evaluate the official measure of nDCG@10, alongside nDCG@1k to test the quality of deeper rankings and $\mathrm{R}@1\mathrm{k}$ to test the ability of the algorithm to identify all known relevant passages to a given topic. Following standard conventions on TREC DL 2019, we use a minimum relevance score of 2 when computing recall. We use pytrec_eval [8] to compute these measurements.

评估指标。对于MS MARCO开发集，我们使用MS MARCO提供的评估脚本，采用官方评估指标——深度为10的平均倒数排名（RR@10）进行评估。为了解整个系统检索相关段落的能力，我们测量深度为1000的召回率（R@1k），这也是开发集评估中常用的指标。为测试PLAID对详尽搜索的近似程度，我们测量排名偏差重叠（RBO）[26]，持久性设置为0.99。我们通过在开发集上使用单个CPU线程的平均响应时间（以毫秒/查询（ms/q）为单位）来衡量效率。按照原始论文，我们只测量检索时间，忽略查询编码所需的时间（所有方法的查询编码时间相同）。对于TREC DL 2019，我们评估官方指标nDCG@10，同时评估nDCG@1k以测试更深层次排名的质量，并使用$\mathrm{R}@1\mathrm{k}$测试算法识别给定主题所有已知相关段落的能力。按照TREC DL 2019的标准惯例，我们在计算召回率时使用最小相关性得分2。我们使用pytrec_eval[8]来计算这些指标。

<!-- Media -->

Table 2: Core reproduction results. NR: Value was Not Reported in the original paper. $\mathrm{N}/\mathrm{A}$ : The latency for an exhaustive search over ColBERTv2 is excessive and not applicable to this study.

表2：核心复现结果。NR：原始论文中未报告该值。$\mathrm{N}/\mathrm{A}$：对ColBERTv2进行详尽搜索的延迟过高，不适用于本研究。

<table><tr><td rowspan="2"/><td colspan="4">MS MARCO Dev</td><td colspan="3">TREC DL 2019</td></tr><tr><td>RR@10</td><td>R@1k</td><td>RBO</td><td>ms/q</td><td>nDCG@10</td><td>nDCG@1k</td><td>R@1k</td></tr><tr><td colspan="8">PLAID Reproduction</td></tr><tr><td>(a)</td><td>0.394</td><td>0.833</td><td>0.612</td><td>80.5</td><td>0.739</td><td>0.553</td><td>0.555</td></tr><tr><td>(b)</td><td>0.397</td><td>0.933</td><td>0.890</td><td>103.4</td><td>0.745</td><td>0.707</td><td>0.786</td></tr><tr><td>(c)</td><td>0.397</td><td>0.975</td><td>0.983</td><td>163.9</td><td>0.745</td><td>0.760</td><td>0.871</td></tr><tr><td colspan="8">Original PLAID Results</td></tr><tr><td>(a)</td><td>0.394</td><td>NR</td><td>NR</td><td>185.5</td><td>NR</td><td>NR</td><td>NR</td></tr><tr><td>(b)</td><td>0.398</td><td>NR</td><td>NR</td><td>222.3</td><td>NR</td><td>NR</td><td>NR</td></tr><tr><td>(c)</td><td>0.398</td><td>0.975</td><td>NR</td><td>352.3</td><td>NR</td><td>NR</td><td>NR</td></tr><tr><td colspan="8">Exhaustive ColBERTv2 Search</td></tr><tr><td>-</td><td>0.397</td><td>0.984</td><td>1.000</td><td>N/A</td><td>0.745</td><td>0.769</td><td>0.894</td></tr></table>

<table><tbody><tr><td rowspan="2"></td><td colspan="4">MS MARCO开发集</td><td colspan="3">TREC DL 2019（文本检索会议深度学习2019）</td></tr><tr><td>前10召回率（RR@10）</td><td>前1000召回率（R@1k）</td><td>等级相关系数（RBO）</td><td>每查询毫秒数（ms/q）</td><td>前10归一化折损累计增益（nDCG@10）</td><td>前1000归一化折损累计增益（nDCG@1k）</td><td>前1000召回率（R@1k）</td></tr><tr><td colspan="8">PLAID复现结果</td></tr><tr><td>(a)</td><td>0.394</td><td>0.833</td><td>0.612</td><td>80.5</td><td>0.739</td><td>0.553</td><td>0.555</td></tr><tr><td>(b)</td><td>0.397</td><td>0.933</td><td>0.890</td><td>103.4</td><td>0.745</td><td>0.707</td><td>0.786</td></tr><tr><td>(c)</td><td>0.397</td><td>0.975</td><td>0.983</td><td>163.9</td><td>0.745</td><td>0.760</td><td>0.871</td></tr><tr><td colspan="8">原始PLAID结果</td></tr><tr><td>(a)</td><td>0.394</td><td>未报告（NR）</td><td>未报告（NR）</td><td>185.5</td><td>未报告（NR）</td><td>未报告（NR）</td><td>未报告（NR）</td></tr><tr><td>(b)</td><td>0.398</td><td>未报告（NR）</td><td>未报告（NR）</td><td>222.3</td><td>未报告（NR）</td><td>未报告（NR）</td><td>未报告（NR）</td></tr><tr><td>(c)</td><td>0.398</td><td>0.975</td><td>未报告（NR）</td><td>352.3</td><td>未报告（NR）</td><td>未报告（NR）</td><td>未报告（NR）</td></tr><tr><td colspan="8">详尽的ColBERTv2搜索</td></tr><tr><td>-</td><td>0.397</td><td>0.984</td><td>1.000</td><td>不适用（N/A）</td><td>0.745</td><td>0.769</td><td>0.894</td></tr></tbody></table>

<!-- Media -->

Hardware. The original PLAID paper evaluated multiple hardware configurations, including single-CPU, multi-CPU, and GPU settings. Given the algorithm's focus on efficiency, we exclusively use a single-threaded setting, recognizing that most parts of the algorithm can be trivially parallelized on either CPU or GPU. Also as was done in the original work, we load all embeddings into memory, eliminating the overheads of reading from disk. We conducted our experiments using a machine equipped with a ${3.4}\mathrm{{GHz}}\mathrm{{AMD}}$ Ryzen 9 5950X processor. (The original paper used a ${2.6}\mathrm{{GHz}}$ Intel Xeon Gold 6132 processor.)

硬件。原始的PLAID论文评估了多种硬件配置，包括单CPU、多CPU和GPU设置。鉴于该算法注重效率，我们仅使用单线程设置，因为我们认识到该算法的大部分部分可以在CPU或GPU上轻松实现并行化。同样，正如原始工作中所做的那样，我们将所有嵌入向量加载到内存中，消除了从磁盘读取的开销。我们使用配备了${3.4}\mathrm{{GHz}}\mathrm{{AMD}}$锐龙9 5950X处理器的机器进行了实验。（原始论文使用的是${2.6}\mathrm{{GHz}}$英特尔至强金牌6132处理器。）

### 3.2 Results

### 3.2 结果

Table 2 presents the results of our core reproduction study. We start by considering the effectiveness reported on MS MARCO Dev. We see virtually no difference across all three operational points in terms of the precision-oriented RR@10 measure. ${}^{7}$ In terms of efficiency, our absolute latency measurements are lower, though this is not surprising given that we are using a faster CPU. The approximate relative differences between each of the operational points are similar, however, e.g., operational point (b) provides a 37% speedup over (c) in both the original paper and our reproduction. Regarding R@1k and RBO, we see similar trends to that of RR@10: as the operational settings collectively consider more documents for final scoring, the measures improve. These results demonstrate that PLAID is working as expected: when more documents are considered, PLAID identifies a larger number of relevant documents (R@1k increases) and also produces a better approximation of an exhaustive ColBERTv2 search (RBO increases).

表2展示了我们核心复现研究的结果。我们首先考虑在MS MARCO开发集上报告的有效性。就以精确率为导向的RR@10指标而言，我们发现在所有三个操作点上几乎没有差异。${}^{7}$ 在效率方面，我们的绝对延迟测量值更低，不过考虑到我们使用的是更快的CPU，这并不令人惊讶。然而，每个操作点之间的近似相对差异是相似的，例如，在原始论文和我们的复现实验中，操作点（b）相对于（c）的速度都提高了37%。关于R@1k和RBO，我们观察到与RR@10类似的趋势：随着操作设置共同考虑更多文档进行最终评分，这些指标得到改善。这些结果表明PLAID的运行符合预期：当考虑更多文档时，PLAID能够识别出更多相关文档（R@1k增加），并且也能更好地近似穷举式的ColBERTv2搜索（RBO增加）。

When considering the results on TREC DL 2019, we observe similar trends to the Dev results. The precision-focused nDCG@10 measure improves slightly from (a) to (b), while nDCG@1k and $\mathrm{R}@1\mathrm{k}$ exhibit larger improvements across the settings due to the improved recall of the system. These results help further demonstrate PLAID's robustness in different evaluation settings.

在考虑TREC DL 2019上的结果时，我们观察到与开发集结果类似的趋势。以精确率为重点的nDCG@10指标从（a）到（b）略有改善，而nDCG@1k和$\mathrm{R}@1\mathrm{k}$在不同设置下表现出更大的改善，这是由于系统的召回率得到了提高。这些结果进一步证明了PLAID在不同评估设置下的鲁棒性。

---

<!-- Footnote -->

${}^{7}$ We note that tools such as repro_eval [1] are available to provide more fine-grained comparisons of individual rankings. In our study, we are primarily interested in the overall effectiveness, rather than the precise ordering of the results used to achieve these scores.

${}^{7}$ 我们注意到，像repro_eval [1]这样的工具可用于对单个排名进行更细粒度的比较。在我们的研究中，我们主要关注整体有效性，而不是用于获得这些分数的结果的精确排序。

${}^{4}$ https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz

${}^{4}$ https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz

${}^{5}$ https://github.com/stanford-futuredata/ColBERT

${}^{5}$ https://github.com/stanford-futuredata/ColBERT

${}^{6}$ I.e.,fully scoring all documents. We modified the codebase to support this option.

${}^{6}$ 即对所有文档进行全面评分。我们修改了代码库以支持此选项。

<!-- Footnote -->

---

In summary, we are able to reproduce PLAID's core results (in terms of precision and efficiency) successfully on a single CPU setting. We further validate that the trends hold when measuring PLAID with recall-oriented measures and when evaluating PLAID on a dataset with more complete relevance assessments. However, there are still several limitations with the original evaluation. Although we know three settings in which PLAID's parameters can work together to deliver efficient retrieval, we do not understand the effect of each individually. Further, although PLAID retrieval is quite fast in an absolute sense (down to around ${80}\mathrm{\;{ms}}/$ query on a single CPU core), we do not know how well this compares to highly-competitive re-ranking systems. These limitations are addressed in the following sections.

总之，我们能够在单CPU设置下成功复现PLAID的核心结果（在精确率和效率方面）。我们进一步验证了，当使用以召回率为导向的指标衡量PLAID时，以及在具有更完整相关性评估的数据集上评估PLAID时，这些趋势仍然成立。然而，原始评估仍然存在一些局限性。虽然我们知道PLAID的参数可以共同作用以实现高效检索的三种设置，但我们并不了解每个参数单独的影响。此外，尽管PLAID检索在绝对意义上相当快（在单个CPU核心上大约为${80}\mathrm{\;{ms}}/$查询），但我们不知道它与极具竞争力的重排序系统相比效果如何。这些局限性将在以下章节中得到解决。

## 4 PARAMETER STUDY

## 4 参数研究

Recall that PLAID introduces three new parameters: nprobe (the number of clusters retrieved for each token), ${t}_{cs}$ (the centroid pruning threshold), and ndocs (the maximum number of documents returned after centroid interaction pruning). Although the original paper suggested three settings for these parameters (see Table 1), it did not explain how these parameters were selected or how each parameter ultimately affects retrieval effectiveness or efficiency. In this section, we fill this gap.

回顾一下，PLAID引入了三个新参数：nprobe（为每个令牌检索的簇的数量）、${t}_{cs}$（质心剪枝阈值）和ndocs（质心交互剪枝后返回的最大文档数）。尽管原始论文为这些参数建议了三种设置（见表1），但它没有解释这些参数是如何选择的，以及每个参数最终如何影响检索的有效性或效率。在本节中，我们将填补这一空白。

### 4.1 Experimental Setup

### 4.1 实验设置

We extend the experimental setup from our core reproduction study presented in Section 3.1. We then performed a grid search over the following parameter settings: nprobe $\in  \{ 1,2,4,8\} ,{t}_{cs} \in$ $\{ {0.3},{0.4},{0.45},{0.5},{0.6}\}$ ,and ndocs $\in  \{ {256},{1024},{4096},{8192}\}$ .

我们扩展了第3.1节中介绍的核心复现研究的实验设置。然后，我们对以下参数设置进行了网格搜索：nprobe $\in  \{ 1,2,4,8\} ,{t}_{cs} \in$ $\{ {0.3},{0.4},{0.45},{0.5},{0.6}\}$ 以及ndocs $\in  \{ {256},{1024},{4096},{8192}\}$ 。

This set of parameters was initially seeded by performing a grid search over the suggested parameter settings. Given that nprobe already includes the minimum value of 1 , we extended it to 8 to check if introducing even more candidate documents from the first stage helps. For ${t}_{cs}$ ,we extended the parameter search in both directions: down to 0.3 (filtering out fewer documents based on the centroid scores) and up to 0.6 (filtering out more documents). Finally, we extended ndocs up to 8192 , based on our observations that low values of this parameter (e.g., 256) substantially harm effectiveness.

这组参数最初是通过对建议的参数设置进行网格搜索来确定的。鉴于nprobe已经包含最小值1，我们将其扩展到8，以检查从第一阶段引入更多候选文档是否有帮助。对于${t}_{cs}$，我们在两个方向上扩展了参数搜索范围：向下至0.3（基于质心分数过滤掉较少的文档），向上至0.6（过滤掉更多的文档）。最后，根据我们的观察，该参数的低值（例如256）会显著损害检索效果，因此我们将ndocs扩展到8192。

We also asked the PLAID authors about anything else to tweak with PLAID to maximize effectiveness or efficiency. They told us that these three parameters have the most substantial effect. Meanwhile, the indexing-time parameters of nbits and nclusters can also affect the final performance. However, see these two indexing parameters as settings of the ColBERTv2 model rather than the PLAID retriever, so in the interest of keeping the number of combinations manageable, we focus on the retriever's parameters.

我们还向PLAID的作者询问了是否还有其他可以调整PLAID以最大化效果或效率的方法。他们告诉我们，这三个参数的影响最为显著。同时，nbits和nclusters这两个索引时间参数也会影响最终性能。然而，我们将这两个索引参数视为ColBERTv2模型的设置，而非PLAID检索器的设置，因此为了使参数组合的数量易于管理，我们专注于检索器的参数。

### 4.2 Results

### 4.2 结果

Figure 3 presents the results of our parameter study. The figure breaks down the effect of each parameter when balancing retrieval latency (ms/q) and either MS MARCO Dev RR@10, Dev RBO, or DL19 nDCG@1k.Each evaluation covers a different possible goal of PLAID: finding a single relevant passage, mimicking an exhaustive search, and ranking all known relevant documents. To help visually isolate the effect of each parameter, lines connect the points that keep the other two parameters constant.

图3展示了我们的参数研究结果。该图分解了每个参数在平衡检索延迟（毫秒/查询）与MS MARCO Dev RR@10、Dev RBO或DL19 nDCG@1k时的影响。每次评估涵盖了PLAID的不同可能目标：找到单个相关段落、模拟穷举搜索以及对所有已知相关文档进行排序。为了从视觉上区分每个参数的影响，图中用线连接了保持其他两个参数不变的点。

From examining the figure, it is clear that ndocs consistently has the most substantial effect on both effectiveness and efficiency. Selecting too few documents to score (ndocs=256) consistently reduces effectiveness while only saving minimal latency (around ${10}\mathrm{{ms}}/\mathrm{q}$ compared to ndocs $= {1024})$ . Meanwhile,increasing ndocs further to 4096 does not benefit the quality of the top 10 results (RR@10). However, the change plays a consistent and important role in improving the quality of the results further down in the ranking (RBO and nDCG@1k). Finally, increasing ndocs=8192 provides no additional benefits regarding search result quality or the faithfulness of the approximation to an exhaustive search, while increasing latency substantially. Based on these observations, we recommend setting ndocs $\in  \left\lbrack  {{1024},{4098}}\right\rbrack$ ,since the benefits of the values outside this range are minimal.

通过查看该图，可以明显看出ndocs始终对效果和效率的影响最为显著。选择过少的文档进行评分（ndocs = 256）会持续降低检索效果，而仅能节省极少的延迟（与ndocs $= {1024})$ 相比约为${10}\mathrm{{ms}}/\mathrm{q}$）。同时，将ndocs进一步增加到4096对前10个结果的质量（RR@10）并无益处。然而，这一变化在提高排名靠后结果的质量（RBO和nDCG@1k）方面起到了持续且重要的作用。最后，将ndocs增加到8192在搜索结果质量或近似穷举搜索的准确性方面没有额外的好处，同时会显著增加延迟。基于这些观察，我们建议将ndocs设置为$\in  \left\lbrack  {{1024},{4098}}\right\rbrack$，因为超出此范围的值带来的好处微乎其微。

The next most influential parameter is nprobe. As expected, increasing the number of clusters matched for each token consistently increases the latency since more candidate documents are produced and processed throughout the pipeline. Setting the value too low (nprobe=1 and sometimes nprobe=2) can often substantially reduce effectiveness, however, since documents filtered out at this stage will have no chance to be retrieved. This is especially apparent in Dev RR@10.Meanwhile, setting this value too high can reduce efficiency without yielding any gains in effectiveness.

下一个最具影响力的参数是nprobe。正如预期的那样，增加每个令牌匹配的簇数量会持续增加延迟，因为在整个流程中会生成和处理更多的候选文档。然而，将该值设置得过低（nprobe = 1，有时nprobe = 2）通常会显著降低检索效果，因为在这个阶段被过滤掉的文档将没有机会被检索到。这在Dev RR@10中尤为明显。同时，将该值设置得过高会降低效率，而不会在效果上有任何提升。

Finally, ${t}_{cs}$ has the smallest effect on retrieval effectiveness,with changes to this parameter typically only adjusting the retrieval latency. This can be see by the roughly horizontal lines in Figure 3. However, as this threshold gets too high, it can have variable effects on both effectiveness and efficiency. For instance, with Dev RR@10, setting ${t}_{cs} = {0.6}$ sometimes reduces effectiveness and increases latency. Therefore,we recommend using ${t}_{cs} \in  \left\lbrack  {{0.4},{0.5}}\right\rbrack   -$ and preferably towards the higher end of the range to limit the effect on latency.

最后，${t}_{cs}$对检索效果的影响最小，该参数的变化通常仅会调整检索延迟。这可以从图3中大致水平的线条看出。然而，当这个阈值设置得过高时，它会对效果和效率产生不同的影响。例如，对于Dev RR@10，将${t}_{cs} = {0.6}$设置得过高有时会降低效果并增加延迟。因此，我们建议使用${t}_{cs} \in  \left\lbrack  {{0.4},{0.5}}\right\rbrack   -$，并且最好接近该范围的上限，以限制对延迟的影响。

We now consider the effect of all three parameters together. Achieving the Pareto frontier for PLAID involves tuning all three parameters in concert. For instance, the lowest retrieval latency requires a very low value of ndocs. However, lowering ndocs to 256 from 1024 without corresponding changes to the other parameters could simply yield worse effectiveness without making much of a dent in latency. Meanwhile, boosting ndocs without also adjusting nprobe will increase latency without improving effectiveness. Figure 1 (on Page 1) perhaps shows the effect of this patchwork of parameters most clearly, with the Pareto frontier formed of various combinations of nprobe $\in  \{ 1,2,4,8\} ,{t}_{cs} \in  \{ {0.3},{0.45},{0.5},{0.6}\}$ ,and ndocs $\in  \{ {256},{1024},{4096},{8192}\}$ .

现在我们考虑这三个参数的综合影响。要使PLAID达到帕累托最优，需要同时调整这三个参数。例如，要实现最低的检索延迟，需要将ndocs设置为非常低的值。然而，如果在不相应改变其他参数的情况下将ndocs从1024降低到256，可能只会降低检索效果，而对延迟的影响不大。同时，在不调整nprobe的情况下增加ndocs会增加延迟，而不会提高检索效果。图1（第1页）可能最清晰地展示了这些参数组合的影响，帕累托最优前沿由nprobe $\in  \{ 1,2,4,8\} ,{t}_{cs} \in  \{ {0.3},{0.45},{0.5},{0.6}\}$ 和ndocs $\in  \{ {256},{1024},{4096},{8192}\}$ 的各种组合构成。

In summary, each of PLAID's parameters plays a role in the final efficiency-effectiveness trade-offs of the algorithm. While ndocs plays the most important role, properly setting nprobe (and to a lesser extent, ${t}_{cs}$ ) is also necessary to achieve a good balance. In some ways, the importance of ndocs is unsurprising since the more documents you score precisely, the higher effectiveness one can expect (up to a point). But this begs some important questions. What is the impact of the source of the pool of documents for exact scoring? Is PLAID's progressive filtering process worth the computational cost compared to simpler and faster candidate generation processes? We answer these questions by exploring re-ranking baselines in the following section.

总之，PLAID的每个参数都在算法的最终效率 - 有效性权衡中发挥着作用。虽然文档数量（ndocs）起着最重要的作用，但正确设置探测数量（nprobe）（在较小程度上，还有${t}_{cs}$）对于实现良好的平衡也是必要的。从某种程度上来说，文档数量（ndocs）的重要性并不令人意外，因为精确评分的文档越多，人们所能期望的有效性就越高（达到一定程度）。但这也引出了一些重要的问题。用于精确评分的文档池来源会产生什么影响？与更简单、更快的候选生成过程相比，PLAID的渐进式过滤过程是否值得付出计算成本？我们将在接下来的部分通过探索重排序基线来回答这些问题。

<!-- Media -->

<!-- figureText: nprobe 150 200 100 150 200 Latency (ms/q) Latency (ms/q) 0.8 0.398 0.396 0.394 0.392 0.9 0.6 0.5 150 200 100 Latency (ms/q) -->

<img src="https://cdn.noedgeai.com/0195a4e0-ba45-7b07-899d-74f5bd2a9bdf_4.jpg?x=168&y=256&w=1474&h=1135&r=0"/>

Figure 3: Results of our study of PLAID’s parameters nprobe, ${t}_{cs}$ , and ndocs. Each row plots the same data points, with the colors representing each parameter value and the lines between them showing the effect with the other two parameters held constant. The dotted line shows the results of an exhaustive search, and the circled points highlight the three recommended settings from the original paper.

图3：我们对PLAID的参数探测数量（nprobe）、${t}_{cs}$和文档数量（ndocs）的研究结果。每一行绘制的是相同的数据点，颜色代表每个参数值，它们之间的线显示了在另外两个参数保持不变时的影响。虚线显示了穷举搜索的结果，圆圈标记的点突出显示了原论文推荐的三种设置。

<!-- Media -->

## 5 BASELINE STUDY

## 5 基线研究

The original paper compared PLAID's efficiency to three baselines: (1) Vanilla ColBERT(v2), which uses IVF indexes for each token for retrieval, in line with the method used by original ColBERT(v1) [11]; (2) SPLADEv2 [7], which is a learned sparse retriever [18]; and (3) BM25 [21], which is a traditional lexical retrieval model. Among these baselines, only Vanilla ColBERT(v2) represents an alternative retrieval engine; SPLADEv2 use other scoring mechanisms and act as points of reference. Curiously, the evaluation omitted the common approach of just re-ranking the results from an efficient-but-imprecise model like BM25. In this section, we compare PLAID with this baseline. Further, we compare both approaches with Lexically Accelerated Dense Retrieval (LADR) [12], which modifies the re-ranking algorithm to also consider the nearest neighbors of the top-scoring results encountered when re-ranking.

原论文将PLAID的效率与三个基线进行了比较：（1）普通ColBERT（v2），它为每个词元使用倒排文件（IVF）索引进行检索，与原始ColBERT（v1）使用的方法一致[11]；（2）SPLADEv2 [7]，它是一种学习型稀疏检索器[18]；（3）BM25 [21]，它是一种传统的词法检索模型。在这些基线中，只有普通ColBERT（v2）代表了一种替代的检索引擎；SPLADEv2使用其他评分机制，作为参考点。奇怪的是，评估中遗漏了一种常见的方法，即对像BM25这样高效但不精确的模型的结果进行重排序。在本节中，我们将PLAID与这个基线进行比较。此外，我们还将这两种方法与词法加速密集检索（LADR）[12]进行比较，LADR修改了重排序算法，在重排序时还会考虑得分最高的结果的最近邻。

### 5.1 Experimental Setup

### 5.1 实验设置

We use PLAID's experimental results from Section 4 as a starting point for our baseline study. We further modify the PLAID source code to support two more approaches: re-ranking and LADR.

我们将第4节中PLAID的实验结果作为基线研究的起点。我们进一步修改PLAID的源代码，以支持另外两种方法：重排序和词法加速密集检索（LADR）。

Re-Ranking. We use the efficient PISA engine [17] for BM25 retrieval, using default parameters and a BlockMaxWAND [6] index structure. We then re-rank those results using ColBERTv2's decompression and scoring function. Given that we found the number of candidate documents for scoring to be the most important parameter for PLAID, we vary the number of retrieved results from BM25 as each of the following values: $n \in  \{ {200},{500},{1000},{2000},{5000},{10000}\}$ . Note that due to the dynamic index pruning applied, performing initial retrieval is considerably faster for low values of $n$ than for higher ones-in addition to the cost of ColBERTv2 decompression and scoring.

重排序。我们使用高效的PISA引擎[17]进行BM25检索，使用默认参数和块最大魔杖（BlockMaxWAND）[6]索引结构。然后，我们使用ColBERTv2的解压缩和评分函数对这些结果进行重排序。鉴于我们发现用于评分的候选文档数量是PLAID最重要的参数，我们将BM25检索结果的数量设置为以下每个值：$n \in  \{ {200},{500},{1000},{2000},{5000},{10000}\}$。请注意，由于应用了动态索引剪枝，对于较低的$n$值，初始检索的速度比高值要快得多，此外还有ColBERTv2解压缩和评分的成本。

<!-- Media -->

<!-- figureText: 0.74 0.75 0.90 0.80 1.00 0.95 0.90 0.85 0.80 0.75 0.70 200 300 0 100 200 300 Latency (ms/q) Latency (ms/q) 0.70 0.73 0.65 0.72 0.71 0.55 0.40 1.00 0.95 0.39 0.90 0.85 0.38 0.80 0.37 0.75 200 300 100 Latency (ms/q) -->

<img src="https://cdn.noedgeai.com/0195a4e0-ba45-7b07-899d-74f5bd2a9bdf_5.jpg?x=177&y=254&w=1430&h=776&r=0"/>

Figure 4: Results of our baseline study. The lines connecting points for each approach represent its Pareto frontier.

图4：我们的基线研究结果。每种方法连接各点的线代表其帕累托前沿。

<!-- Media -->

LADR. We further build upon the re-ranker pipeline using LADR. This approach incorporates a nearest neighbor lookup for top-scoring ColBERTv2 results to overcome possible lexical mismatch from the first stage retrieval. In line with the procedure for PLAID, we perform a grid search over the number of initial BM25 candidates $n \in  \{ {100},{500},{1000}\}$ and the number of nearest neighbors to lookup $k \in  \{ {64},{128}\}$ . We use the precomputed nearest neighbor graph based on BM25 from the original LADR paper. By using the adaptive variant of LADR, we iteratively score the neighbors of the top $c \in  \{ {10},{20},{50}\}$ results until they converge.

词法加速密集检索（LADR）。我们进一步基于LADR构建重排序器管道。这种方法对得分最高的ColBERTv2结果进行最近邻查找，以克服第一阶段检索中可能出现的词法不匹配问题。与PLAID的过程一致，我们对初始BM25候选数量$n \in  \{ {100},{500},{1000}\}$和要查找的最近邻数量$k \in  \{ {64},{128}\}$进行网格搜索。我们使用原LADR论文中基于BM25预先计算的最近邻图。通过使用LADR的自适应变体，我们迭代地对得分最高的$c \in  \{ {10},{20},{50}\}$个结果的邻居进行评分，直到它们收敛。

Evaluation. We use the same datasets and evaluation measures as in Section 3.1. In line with this setting, we include the single-threaded first-stage retrieval latency from PISA for both additional baselines. In a multi-threaded or GPU environment, we note that this first-stage retrieval could be done in parallel with the Col-BERTv2 query encoding process, further reducing the cost of these baselines. However, given the single-threaded nature of our evaluation, we treat this as additional latency.

评估。我们使用与第3.1节相同的数据集和评估指标。根据这个设置，我们将PISA的单线程第一阶段检索延迟纳入两个额外基线的评估中。在多线程或GPU环境中，我们注意到这个第一阶段检索可以与Col - BERTv2查询编码过程并行进行，从而进一步降低这些基线的成本。然而，鉴于我们评估的单线程性质，我们将其视为额外的延迟。

### 5.2 Results

### 5.2 结果

Figure 4 presents the results from our baseline study. We begin by focusing on the BM25 re-ranking pipeline. We observe that this pipeline can retrieve substantially faster than the fastest PLAID pipeline (as low as $9\mathrm{\;{ms}}/\mathrm{q}$ at $n = {200}$ ,compared to ${73}\mathrm{\;{ms}}/\mathrm{q}$ for the fastest PLAID pipeline). Although this setting typically reduces the quality of results compared to the fastest PLAID pipelines (Dev RR@10, RBO, R@1k, and DL19 nDCG@10), it is still remarkably strong in terms of absolute effectiveness. For instance, its Dev RR@10 is 0.373 which is stronger than early BERT-based cross-encoders [20] and more recent learned sparse retrievers [7].

图4展示了我们基线研究的结果。我们首先关注BM25重排序管道。我们观察到，该管道的检索速度比最快的PLAID管道快得多（在$n = {200}$时低至$9\mathrm{\;{ms}}/\mathrm{q}$，而最快的PLAID管道为${73}\mathrm{\;{ms}}/\mathrm{q}$）。尽管与最快的PLAID管道相比，这种设置通常会降低结果质量（开发集前10召回率倒数（Dev RR@10）、排名相似性指标（RBO）、前1000召回率（R@1k）和2019年深度学习评测集前10归一化折损累积增益（DL19 nDCG@10）），但就绝对有效性而言，它仍然非常出色。例如，其开发集RR@10为0.373，比早期基于BERT的交叉编码器[20]和最近的学习型稀疏检索器[7]更强。

As the BM25 re-ranking pipeline considers more documents, the effectiveness gradually improves. In most cases, however, it continues to under-perform PLAID. For instance, when considering the top-10 documents via DL19 nDCG@10 and Dev RR@10, the Pareto frontier of the re-ranking pipeline always under-performs that of PLAID. Nevertheless, the low up-front dcost of performing lexical retrieval methods makes re-ranking an appealing choice when latency or computational cost are critical.

随着BM25重排序管道考虑的文档数量增加，其有效性逐渐提高。然而，在大多数情况下，它的表现仍不如PLAID。例如，通过DL19 nDCG@10和Dev RR@10考虑前10个文档时，重排序管道的帕累托前沿始终不如PLAID。尽管如此，执行词法检索方法的前期成本较低，使得在延迟或计算成本至关重要时，重排序成为一个有吸引力的选择。

Re-ranking is inherently limited by the recall of the first stage, however, and when the first stage only enables lexical matches, this can substantially limit the potential downstream effectiveness. We observe that LADR, as an efficient pseudo-relevance feedback to a re-ranking pipeline, can largely overcome this limitation. On DL19, LADR's Pareto frontier completely eclipses PLAID's, both in terms of nDCG and recall. (LADR's non-optimal operational points are also consistently competitive.) Meanwhile, on Dev, LADR provides competitive-albeit not always optimal-effectiveness. Given that Dev has sparse assessments and DL19 has dense ones, we know that LADR is selecting suitable relevant documents as candidates, even though they are not necessarily the ones ColBERTv2 would have identified through an exhaustive search. The RBO results on Dev further reinforce this: while PLAID can achieve a nearly perfect RBO compared with an exhaustive search, LADR maxes out at around 0.96 .

然而，重排序本质上受到第一阶段召回率的限制，当第一阶段仅支持词法匹配时，这可能会极大地限制下游的潜在有效性。我们观察到，LADR作为重排序管道的一种高效伪相关反馈，可以在很大程度上克服这一限制。在DL19上，LADR的帕累托前沿在归一化折损累积增益（nDCG）和召回率方面完全超越了PLAID。（LADR的非最优操作点也始终具有竞争力。）同时，在开发集上，LADR虽然并不总是最优的，但也具有竞争力。鉴于开发集的评估是稀疏的，而DL19的评估是密集的，我们知道LADR正在选择合适的相关文档作为候选，即使这些文档不一定是ColBERTv2通过穷举搜索会识别出的文档。开发集上的RBO结果进一步证实了这一点：虽然与穷举搜索相比，PLAID可以实现近乎完美的RBO，但LADR的最大值约为0.96。

In summary, re-ranking and its variant LADR are highly competitive baselines compared to PLAID, especially at the low-latency settings that PLAID targets. Although they do not necessarily identify the same documents that an exhaustive ColBERTv2 search would provide, the baselines typically provide alternative documents of high relevance.

综上所述，与PLAID相比，重排序及其变体LADR是极具竞争力的基线，尤其是在PLAID所针对的低延迟设置下。尽管它们不一定能识别出与ColBERTv2穷举搜索相同的文档，但这些基线通常能提供具有高度相关性的替代文档。

<!-- Media -->

<!-- figureText: 50 $\begin{array}{lllllllllll} {0.0} & {0.1} & {0.2} & {0.3} & {0.4} & {0.5} & {0.6} & {0.7} & {0.8} & {0.9} & {1.0} \end{array}$ Majority Token Proportion -->

<img src="https://cdn.noedgeai.com/0195a4e0-ba45-7b07-899d-74f5bd2a9bdf_6.jpg?x=165&y=247&w=684&h=321&r=0"/>

Figure 5: The distribution of Majority Token Proportions among clusters for ColBERTv2.

图5：ColBERTv2聚类中多数Token比例的分布。

<!-- Media -->

We note that re-ranking comes with downsides, however. It requires building and maintaining a lexical index alongside ColBERT's index, which adds storage costs, indexing time, and overall complexity to the retrieval system. Nonetheless, these costs are comparatively low compared to those of deploying a ColBERTv2 system itself. For instance, a ColBERTv2 index of MS MARCO v1 consumes around ${22}\mathrm{{GB}}$ of storage,while a lexical PISA index uses less than 1GB. Meanwhile, hybrid retrieval systems (i.e., those that combine signals from both a lexical and a neural model) will need to incur these costs anyway. LADR adds additional costs in building and maintaining a document proximity graph (around $2\mathrm{{GB}}$ for a graph with 64 neighbors per document on MS MARCO).

然而，我们注意到重排序也有缺点。它需要在ColBERT索引的基础上构建和维护一个词法索引，这会增加检索系统的存储成本、索引时间和整体复杂性。尽管如此，与部署ColBERTv2系统本身的成本相比，这些成本相对较低。例如，MS MARCO v1的ColBERTv2索引消耗约${22}\mathrm{{GB}}$的存储空间，而词法PISA索引使用不到1GB。同时，混合检索系统（即结合词法模型和神经模型信号的系统）无论如何都需要承担这些成本。LADR在构建和维护文档邻近图方面增加了额外成本（在MS MARCO上，每个文档有64个邻居的图大约需要$2\mathrm{{GB}}$）。

### 5.3 Cluster Analysis

### 5.3 聚类分析

Curious as to why re-ranking a lexical system is competitive compared to PLAID, we conduct an analysis of the token representation clusters PLAID uses for retrieval vis-à-vis the lexical form of the token. We use the ColBERTv2 MS MARCO v1 passage index from the previous experiments, and modify the source to log the original token ID alongside the cluster ID and residuals of each token. We then conduct our analysis using this mapping between the token IDs and cluster IDs.

我们好奇为什么词法系统的重排序与PLAID相比具有竞争力，因此我们分析了PLAID用于检索的Token表示聚类与Token的词法形式之间的关系。我们使用上一个实验中的ColBERTv2 MS MARCO v1段落索引，并修改源代码以记录每个Token的原始Token ID、聚类ID和残差。然后，我们使用Token ID和聚类ID之间的这种映射进行分析。

We start by investigating how homogeneous token clusters are. In other words, we ask the question: Do most of a cluster's representations come from the same source token? We first observe that most clusters map to multiple tokens (the median number of tokens a cluster maps to is 15 ,while only ${2.2}\%$ of tokens only map to a single token). However, this does not tell the complete story since the distribution of tokens within each cluster is highly skewed. To overcome this, we measure the proportion of each cluster that belongs to the majority (or plurality) token. Figure 5 presents the distribution of the majority token proportions across all clusters. We observe that ${39}\%$ of clusters have a majority proportion above 0.95 (i.e.,over ${95}\%$ of representations in these clusters come from the same token). Meanwhile, the median proportion among all clusters is 0.86 . Only 2.7% of clusters have a majority proportion less than 10%. Collectively, these results suggest that although clusters are frequently formed of multiple tokens, they are usually heavily dominated by a single token. In other words, they largely perform lexical matching.

我们首先研究标记（token）簇的同质性如何。换句话说，我们提出这样一个问题：一个簇中的大多数表示是否来自同一个源标记？我们首先观察到，大多数簇会映射到多个标记（一个簇映射到的标记数量中位数为15，而只有${2.2}\%$的标记仅映射到单个簇）。然而，这并不能说明全部情况，因为每个簇内标记的分布高度偏斜。为了克服这一问题，我们测量每个簇中属于多数（或 plurality）标记的比例。图5展示了所有簇中多数标记比例的分布情况。我们观察到，${39}\%$的簇的多数比例高于0.95（即，这些簇中超过${95}\%$的表示来自同一个标记）。同时，所有簇的比例中位数为0.86。只有2.7%的簇的多数比例低于10%。总体而言，这些结果表明，虽然簇通常由多个标记组成，但它们通常由单个标记主导。换句话说，它们主要进行词汇匹配。

Within a cluster, what exactly are the other matching tokens? Figure 6 provides example clusters for the MS MARCO query "do goldfish grow". Some of the matching clusters (48169 and 225987) perform rather opaque semantic matching over [CLS] and [SEP] tokens. These clusters match either other such control tokens or (much less frequently) function words like and, but, and the. We suspect these function words are coopted to help emphasize the central points of a passage, given that they typically do not provide much in terms of semantics on their own. Next, three clusters (48169, 30151, and 227745) each have majority token proportions below or near the median. However, many of the minority tokens within a cluster are just other morphological forms of the same word: grow, grows, growing, etc. In other words, they share a common stem. When merging stems, these three clusters all have majority token proportions above ${95}\%$ . The final two clusters (21395 and 130592) are dominated ( $> {90}\%$ majority token proportion) by a single token. Like the control tokens, these pick up on punctuation tokens, which we suspect are coopted to help emphasize particularly salient tokens within a passage. This qualitative analysis suggests that although some clusters likely perform semantic matching, Figure 5 may actually be underestimate the overall prevalence of lexical matching among PLAID clusters.

在一个簇中，其他匹配的标记究竟是什么呢？图6展示了MS MARCO查询“do goldfish grow”的示例簇。一些匹配簇（48169和225987）对[CLS]和[SEP]标记进行了相当不透明的语义匹配。这些簇要么匹配其他此类控制标记，要么（非常少见地）匹配诸如“and”“but”和“the”等功能词。我们推测，鉴于这些功能词本身通常在语义方面提供的信息不多，它们被用来帮助强调段落的中心点。接下来，三个簇（48169、30151和227745）的多数标记比例均低于或接近中位数。然而，簇内的许多少数标记只是同一个单词的其他形态形式，如“grow”“grows”“growing”等。换句话说，它们有一个共同的词干。合并词干后，这三个簇的多数标记比例均高于${95}\%$。最后两个簇（21395和130592）由单个标记主导（多数标记比例为$> {90}\%$）。与控制标记类似，这些簇捕捉到了标点标记，我们推测这些标点标记被用来帮助强调段落中特别突出的标记。这种定性分析表明，虽然一些簇可能进行语义匹配，但图5实际上可能低估了PLAID簇中词汇匹配的总体普遍性。

<!-- Media -->

<!-- figureText: Cluster 87188 Cluster 21395 Cluster 48169 Cluster 130592 Cluster 225987 99.1% ##fish 64.0% [CLS] 0.3 % 33.2% [SEP] 0.3% 0.8% the 0.18 - 0.3% but 0.8% [10 more] ${0.2}\% \left\lbrack  {2\text{ more }}\right\rbrack$ 2.18 [14 more] Cluster 227745 (count=3727) 89.7% grow $\left\lbrack  \mathrm{{PAD}}\right\rbrack   \times  {25}$ 3.1% growing 1.9% [9 more] 83.2% [CLS] 93.7% gold 84.4% grow 16.4% [SEP] 5.8 [D] 10.9% grows 0.2% and 0.25 * 2.7% growing 0.1% the 0.1% - 0.68 grown 0.1% [2 more] Cluster 30151 (count=1881) 86.3% grow 4.3% growing 1.08 [8 more] -->

<img src="https://cdn.noedgeai.com/0195a4e0-ba45-7b07-899d-74f5bd2a9bdf_6.jpg?x=931&y=236&w=712&h=380&r=0"/>

Figure 6: Example query, its corresponding clusters retrieved by PLAID, and the original tokens that contributed to each cluster. Bold tokens are (stemmed) lexical matches from the query.

图6：示例查询、PLAID检索到的相应簇以及对每个簇有贡献的原始标记。加粗标记是查询中（词干化后的）词汇匹配项。

<!-- Media -->

The observation that most clusters map to a single source token only tells half the story, however. Perhaps PLAID is effectively performing a form of dynamic pruning [2], wherein query terms only match to a certain subset of lexical matches (i.e., the most semantically related ones) rather than all of them. After all, Figure 6 showed three separate clusters with the same majority token (grow). Therefore, we ask the inverse of our first question: Do most of a token's representations map to the same cluster? Akin to the cluster analysis, we measure the majority cluster proportion for each token,and plot the distribution in Figure 7. Here, ${33}\%$ of tokens have a majority cluster proportion greater than 0.95 . Unlike our observations in Figure 5, the tail is flatter and more uniform, giving a median majority cluster proportion of 0.62 . These results suggest that although a sizable number of tokens map to a single prominent cluster, many tokens are spread among many different clusters. However, as can be seen in Figure 6, just because a token appears in many different clusters doesn't mean that it will necessarily be pruned off completely: two clusters that feature grow (30151 and 227745) are captured directly by the [PAD] "expansion" tokens of the query.

然而，大多数簇仅映射到单个源标记这一观察结果只说明了一半的情况。也许PLAID实际上在执行一种动态剪枝[2]，其中查询词只与词汇匹配的某个子集（即语义最相关的子集）匹配，而不是与所有匹配项匹配。毕竟，图6展示了三个具有相同多数标记（“grow”）的不同簇。因此，我们提出与第一个问题相反的问题：一个标记的大多数表示是否映射到同一个簇？与簇分析类似，我们测量每个标记的多数簇比例，并将分布情况绘制在图7中。在这里，${33}\%$的标记的多数簇比例大于0.95。与我们在图5中的观察结果不同，尾部更平坦、更均匀，多数簇比例的中位数为0.62。这些结果表明，虽然有相当数量的标记映射到一个突出的簇，但许多标记分布在许多不同的簇中。然而，如图6所示，仅仅因为一个标记出现在许多不同的簇中，并不意味着它一定会被完全剪枝掉：两个以“grow”为特征的簇（30151和227745）直接由查询的[PAD]“扩展”标记捕获。

<!-- Media -->

<img src="https://cdn.noedgeai.com/0195a4e0-ba45-7b07-899d-74f5bd2a9bdf_7.jpg?x=261&y=244&w=488&h=230&r=0"/>

Figure 7: The distribution of Majority Cluster Proportions among tokens for ColBERTv2.

图7：ColBERTv2中标记的多数簇比例分布。

<!-- Media -->

This analysis demonstrates that PLAID performs a considerable amount of lexical matching (though not exclusively so) when identifying documents to score. It also provides some insights into why re-ranking is competitive against PLAID.

这一分析表明，PLAID在识别要评分的文档时进行了大量的词汇匹配（尽管并非仅此而已）。它还为重新排序为何能与PLAID竞争提供了一些见解。

## 6 CONCLUSION

## 6 结论

In this paper, we conducted a reproducibility study of PLAID, an efficient retrieval engine for ColBERTv2. We were able to reproduce the study's main results, and showed that they successfully generalize to a dataset with more complete relevance assessments. We also showed that PLAID provides an excellent approximation of an exhaustive ColBERTv2 search. Using an in-depth investigation of PLAID's parameters, we found that they are highly interdependent, and the suggested settings are not necessarily optimal. Specifically, it is almost always worth increasing ndocs beyond the recommended 256 , given the low contribution to latency and high boost in effectiveness that the change provides. Meanwhile, the missing baseline of simply re-ranking a lexical system using ColBERTv2 (and its recent variant, LADR) provides better trade-offs in terms of efficiency and effectiveness at low-latency operational points. However, these baselines do not provide as strong of a true approximation of an exhaustive ColBERTv2 search. Finally, an analysis showed that PLAID relies heavily on lexical matches for the initial retrieval of documents.

在本文中，我们对PLAID（一种用于ColBERTv2的高效检索引擎）进行了可重复性研究。我们能够重现该研究的主要结果，并表明这些结果成功推广到了一个具有更完整相关性评估的数据集上。我们还表明，PLAID能很好地近似于全面的ColBERTv2搜索。通过对PLAID参数的深入研究，我们发现这些参数之间高度相互依赖，并且建议的设置不一定是最优的。具体而言，鉴于将ndocs增加到推荐的256以上对延迟的影响较小，而对有效性的提升较大，因此几乎总是值得这么做。同时，仅使用ColBERTv2（及其最新变体LADR）对词法系统进行重新排序的缺失基线，在低延迟操作点的效率和有效性方面提供了更好的权衡。然而，这些基线并不能像PLAID那样很好地近似于全面的ColBERTv2搜索。最后，分析表明PLAID在文档的初始检索中严重依赖词法匹配。

Our study provides important operational recommendations for those looking to deploy a ColBERTv2 system, both with and without the PLAID engine. It also further highlights the importance of comparing against versatile re-ranking systems when evaluating the efficiency of retrieval algorithms. Given the indirect way that PLAID performs first-stage lexical matching, future work could investigate methods for hybrid PLAID-lexical retrieval. By relying on PLAID for semantic matches and a traditional inverted index for lexical matches, we may be able to achieve the "best of both worlds": the high-quality ColBERTv2 approximation of PLAID and the high efficiency of re-ranking.

我们的研究为那些希望部署ColBERTv2系统的人（无论是否使用PLAID引擎）提供了重要的操作建议。它还进一步强调了在评估检索算法的效率时，与通用的重新排序系统进行比较的重要性。鉴于PLAID进行第一阶段词法匹配的间接方式，未来的工作可以研究混合PLAID - 词法检索的方法。通过依靠PLAID进行语义匹配，依靠传统的倒排索引进行词法匹配，我们或许能够实现“两全其美”：既获得PLAID对ColBERTv2的高质量近似，又获得重新排序的高效率。

## ACKNOWLEDGMENTS

## 致谢

This work is supported, in part, by the Spoke "FutureHPC & Big-Data" of the ICSC - Centro Nazionale di Ricerca in High-Performance Computing, Big Data and Quantum Computing, the Spoke "Human-centered AI" of the M4C2 - Investimento 1.3, Partenariato Esteso PE00000013 - "FAIR - Future Artificial Intelligence Research", funded by European Union - NextGenerationEU, the FoReLab project (Departments of Excellence), and the NEREO PRIN project funded by the Italian Ministry of Education and Research Grant no. 2022AEF-HAZ.

这项工作部分得到了以下机构的支持：ICSC - 国家高性能计算、大数据和量子计算研究中心的“未来高性能计算与大数据”分支；由欧盟 - 下一代欧盟资助的M4C2 - 投资1.3、扩展伙伴关系PE00000013 - “公平 - 未来人工智能研究”的“以人为中心的人工智能”分支；卓越部门的FoReLab项目；以及由意大利教育和研究部资助的NEREO PRIN项目（资助编号：2022AEF - HAZ）。

## REFERENCES

## 参考文献

[1] Timo Breuer, Nicola Ferro, Norbert Fuhr, Maria Maistro, Tetsuya Sakai, Philipp Schaer, and Ian Soboroff. 2020. How to Measure the Reproducibility of System-oriented IR Experiments. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, SIGIR 2020, Virtual Event, China, July 25-30, 2020, Jimmy X. Huang, Yi Chang, Xueqi Cheng, Jaap Kamps, Vanessa Murdock, Ji-Rong Wen, and Yiqun Liu (Eds.). ACM, 349-358. https://doi.org/10.1145/3397271.3401036

[1] Timo Breuer、Nicola Ferro、Norbert Fuhr、Maria Maistro、Tetsuya Sakai、Philipp Schaer和Ian Soboroff。2020年。如何衡量面向系统的信息检索实验的可重复性。收录于《第43届ACM SIGIR信息检索研究与发展国际会议论文集》，SIGIR 2020，线上会议，中国，2020年7月25 - 30日，Jimmy X. Huang、Yi Chang、Xueqi Cheng、Jaap Kamps、Vanessa Murdock、Ji - Rong Wen和Yiqun Liu（编）。ACM，第349 - 358页。https://doi.org/10.1145/3397271.3401036

[2] Andrei Z. Broder, David Carmel, Michael Herscovici, Aya Soffer, and Jason Y. Zien. 2003. Efficient query evaluation using a two-level retrieval process. In Proceedings of the 2003 ACM CIKM International Conference on Information and Knowledge Management, New Orleans, Louisiana, USA, November 2-8, 2003. ACM, 426-434. https://doi.org/10.1145/956863.956944

[2] Andrei Z. Broder、David Carmel、Michael Herscovici、Aya Soffer和Jason Y. Zien。2003年。使用两级检索过程进行高效查询评估。收录于《2003年ACM CIKM信息与知识管理国际会议论文集》，美国路易斯安那州新奥尔良，2003年11月2 - 8日。ACM，第426 - 434页。https://doi.org/10.1145/956863.956944

[3] Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, and Jimmy Lin. 2021. MS MARCO: Benchmarking Ranking Models in the Large-Data Regime. In SIGIR '21: The 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, Virtual Event, Canada, July 11-15, 2021, Fernando Diaz, Chirag Shah, Torsten Suel, Pablo Castells, Rosie Jones, and Tetsuya Sakai (Eds.). ACM, 1566-1576. https://doi.org/10.1145/3404835.3462804

[3] Nick Craswell、Bhaskar Mitra、Emine Yilmaz、Daniel Campos和Jimmy Lin。2021年。MS MARCO：在大数据环境下对排序模型进行基准测试。收录于《SIGIR '21：第44届ACM SIGIR信息检索研究与发展国际会议》，线上会议，加拿大，2021年7月11 - 15日，Fernando Diaz、Chirag Shah、Torsten Suel、Pablo Castells、Rosie Jones和Tetsuya Sakai（编）。ACM，第1566 - 1576页。https://doi.org/10.1145/3404835.3462804

[4] Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, and Ellen M. Voorhees. 2020. Overview of the TREC 2019 deep learning track. CoRR abs/2003.07820 (2020). arXiv:2003.07820 https://arxiv.org/abs/2003.07820

[4] Nick Craswell、Bhaskar Mitra、Emine Yilmaz、Daniel Campos和Ellen M. Voorhees。2020年。TREC 2019深度学习赛道概述。CoRR abs/2003.07820（2020年）。arXiv:2003.07820 https://arxiv.org/abs/2003.07820

[5] Zhuyun Dai, Chenyan Xiong, Jamie Callan, and Zhiyuan Liu. 2018. Convolutional Neural Networks for Soft-Matching N-Grams in Ad-hoc Search. In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining, WSDM 2018, Marina Del Rey, CA, USA, February 5-9, 2018, Yi Chang, Chengxiang Zhai, Yan Liu, and Yoelle Maarek (Eds.). ACM, 126-134. https://doi.org/10.1145/ 3159652.3159659

[5] 戴祝云（Zhuyun Dai）、熊晨燕（Chenyan Xiong）、杰米·卡兰（Jamie Callan）和刘志远（Zhiyuan Liu）。2018年。用于即席搜索中N元语法软匹配的卷积神经网络。收录于第十一届ACM网络搜索与数据挖掘国际会议论文集，WSDM 2018，美国加利福尼亚州玛丽娜德尔雷，2018年2月5 - 9日，易昌（Yi Chang）、翟成祥（Chengxiang Zhai）、刘燕（Yan Liu）和约埃尔·马雷克（Yoelle Maarek）（编）。ACM，126 - 134页。https://doi.org/10.1145/ 3159652.3159659

[6] Shuai Ding and Torsten Suel. 2011. Faster top-k document retrieval using block-max indexes. In Proceeding of the 34th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR 2011, Beijing, China, July 25-29, 2011, Wei-Ying Ma, Jian-Yun Nie, Ricardo Baeza-Yates, Tat-Seng Chua, and W. Bruce Croft (Eds.). ACM, 993-1002. https://doi.org/10.1145/2009916.2010048

[6] 丁帅（Shuai Ding）和托尔斯滕·苏埃尔（Torsten Suel）。2011年。使用块最大索引实现更快的前k个文档检索。收录于第34届ACM信息检索研究与发展国际会议论文集，SIGIR 2011，中国北京，2011年7月25 - 29日，马维英（Wei - Ying Ma）、聂建云（Jian - Yun Nie）、里卡多·贝萨 - 耶茨（Ricardo Baeza - Yates）、蔡达生（Tat - Seng Chua）和W.布鲁斯·克罗夫特（W. Bruce Croft）（编）。ACM，993 - 1002页。https://doi.org/10.1145/2009916.2010048

[7] Thibault Formal, Carlos Lassance, Benjamin Piwowarski, and Stéphane Clinchant. 2021. SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval. CoRR abs/2109.10086 (2021). arXiv:2109.10086 https://arxiv.org/abs/2109.10086

[7] 蒂博·福尔马尔（Thibault Formal）、卡洛斯·拉桑斯（Carlos Lassance）、本杰明·皮沃瓦尔斯基（Benjamin Piwowarski）和斯特凡·克兰尚（Stéphane Clinchant）。2021年。SPLADE v2：用于信息检索的稀疏词汇与扩展模型。计算机研究存储库预印本abs/2109.10086（2021年）。arXiv:2109.10086 https://arxiv.org/abs/2109.10086

[8] Christophe Van Gysel and Maarten de Rijke. 2018. Pytrec_eval: An Extremely Fast Python Interface to trec_eval. In The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval, SIGIR 2018, Ann Arbor, MI, USA, July 08-12, 2018, Kevyn Collins-Thompson, Qiaozhu Mei, Brian D. Davison, Yiqun Liu, and Emine Yilmaz (Eds.). ACM, 873-876. https://doi.org/10.1145/ 3209978.3210065

[8] 克里斯托夫·范·吉塞尔（Christophe Van Gysel）和马腾·德·里克（Maarten de Rijke）。2018年。Pytrec_eval：一个极其快速的trec_eval Python接口。收录于第41届ACM信息检索研究与发展国际会议论文集，SIGIR 2018，美国密歇根州安娜堡，2018年7月8 - 12日，凯文·柯林斯 - 汤普森（Kevyn Collins - Thompson）、梅巧珠（Qiaozhu Mei）、布莱恩·D·戴维森（Brian D. Davison）、刘奕群（Yiqun Liu）和埃米内·伊尔马兹（Emine Yilmaz）（编）。ACM，873 - 876页。https://doi.org/10.1145/ 3209978.3210065

[9] Sebastian Hofstätter, Markus Zlabinger, and Allan Hanbury. 2020. Interpretable & Time-Budget-Constrained Contextualization for Re-Ranking. In ECAI 2020 - 24th European Conference on Artificial Intelligence, 29 August-8 September 2020, Santiago de Compostela, Spain, August 29 - September 8, 2020 - Including 10th Conference on Prestigious Applications of Artificial Intelligence (PAIS 2020) (Frontiers in Artificial Intelligence and Applications, Vol. 325), Giuseppe De Giacomo, Alejandro Catalá, Bistra Dilkina, Michela Milano, Senén Barro, Alberto Bugarín, and Jérôme Lang (Eds.). IOS Press, 513-520. https://doi.org/10.3233/FAIA200133

[9] 塞巴斯蒂安·霍夫施泰特（Sebastian Hofstätter）、马库斯·兹拉宾格（Markus Zlabinger）和艾伦·汉伯里（Allan Hanbury）。2020年。用于重排序的可解释且受时间预算约束的上下文建模。收录于2020年第24届欧洲人工智能会议（ECAI 2020），2020年8月29日 - 9月8日，西班牙圣地亚哥 - 德孔波斯特拉，2020年8月29日 - 9月8日 - 包括第10届人工智能著名应用会议（PAIS 2020）（《人工智能与应用前沿》，第325卷），朱塞佩·德·贾科莫（Giuseppe De Giacomo）、亚历杭德罗·卡拉塔（Alejandro Catalá）、比斯特拉·迪尔基娜（Bistra Dilkina）、米凯拉·米兰诺（Michela Milano）、塞嫩·巴罗（Senén Barro）、阿尔韦托·布加林（Alberto Bugarín）和杰罗姆·朗（Jérôme Lang）（编）。IOS出版社，513 - 520页。https://doi.org/10.3233/FAIA200133

[10] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick S. H. Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for Open-Domain Question Answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020, Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu (Eds.). Association for Computational Linguistics, 6769-6781. https://doi.org/10.18653/ V1/2020.EMNLP-MAIN.550

[10] 弗拉基米尔·卡尔普欣（Vladimir Karpukhin）、巴拉斯·奥古兹（Barlas Oguz）、闵世文（Sewon Min）、帕特里克·S·H·刘易斯（Patrick S. H. Lewis）、莱德尔·吴（Ledell Wu）、谢尔盖·叶杜诺夫（Sergey Edunov）、陈丹琦（Danqi Chen）和易文涛（Wen - tau Yih）。2020年。用于开放域问答的密集段落检索。收录于2020年自然语言处理经验方法会议论文集，EMNLP 2020，线上会议，2020年11月16 - 20日，邦妮·韦伯（Bonnie Webber）、特雷弗·科恩（Trevor Cohn）、何玉兰（Yulan He）和刘洋（Yang Liu）（编）。计算语言学协会，6769 - 6781页。https://doi.org/10.18653/ V1/2020.EMNLP - MAIN.550

[11] Omar Khattab and Matei Zaharia. 2020. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, SIGIR 2020, Virtual Event, China, July 25-30, 2020, Jimmy X. Huang, Yi Chang, Xueqi Cheng, Jaap Kamps, Vanessa Murdock, Ji-Rong Wen, and Yiqun Liu (Eds.). ACM, 39-48. https://doi.org/10.1145/3397271.3401075

[11] 奥马尔·哈塔卜（Omar Khattab）和马特·扎哈里亚（Matei Zaharia）。2020年。ColBERT：通过基于BERT的上下文后期交互实现高效有效的段落搜索。收录于第43届ACM信息检索研究与发展国际会议论文集，SIGIR 2020，线上会议，中国，2020年7月25 - 30日，黄吉米·X（Jimmy X. Huang）、易昌（Yi Chang）、程学旗（Xueqi Cheng）、亚普·坎普斯（Jaap Kamps）、凡妮莎·默多克（Vanessa Murdock）、文继荣（Ji - Rong Wen）和刘奕群（Yiqun Liu）（编）。ACM，39 - 48页。https://doi.org/10.1145/3397271.3401075

[12] Hrishikesh Kulkarni, Sean MacAvaney, Nazli Goharian, and Ophir Frieder. 2023. Lexically-Accelerated Dense Retrieval. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR 2023, Taipei, Taiwan, July 23-27, 2023, Hsin-Hsi Chen, Wei-Jou (Edward) Duh, Hen-Hsen Huang, Makoto P. Kato, Josiane Mothe, and Barbara Poblete (Eds.). ACM, 152-162. https://doi.org/10.1145/3539618.3591715

[12] 赫里希克什·库尔卡尼（Hrishikesh Kulkarni）、肖恩·麦卡瓦尼（Sean MacAvaney）、纳兹利·戈哈里安（Nazli Goharian）和奥菲尔·弗里德（Ophir Frieder）。2023年。词汇加速的密集检索（Lexically-Accelerated Dense Retrieval）。见《第46届ACM国际信息检索研究与发展会议论文集》（Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval），SIGIR 2023，中国台湾台北，2023年7月23 - 27日，陈信希（Hsin-Hsi Chen）、杜伟舟（Edward Duh）、黄恒森（Hen-Hsen Huang）、加藤诚（Makoto P. Kato）、乔西安·莫特（Josiane Mothe）和芭芭拉·波布莱特（Barbara Poblete）（编）。美国计算机协会（ACM），152 - 162页。https://doi.org/10.1145/3539618.3591715

[13] Carlos Lassance and Stéphane Clinchant. 2023. The Tale of Two MSMARCO - and Their Unfair Comparisons. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR 2023, Taipei, Taiwan, July 23-27, 2023, Hsin-Hsi Chen, Wei-Jou (Edward) Duh, Hen-Hsen Huang, Makoto P. Kato, Josiane Mothe, and Barbara Poblete (Eds.). ACM, 2431-2435. https://doi.org/10.1145/3539618.3592071

[13] 卡洛斯·拉桑斯（Carlos Lassance）和斯特凡·克兰尚（Stéphane Clinchant）。2023年。两个MS MARCO数据集的故事及其不公平比较（The Tale of Two MSMARCO - and Their Unfair Comparisons）。见《第46届ACM国际信息检索研究与发展会议论文集》（Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval），SIGIR 2023，中国台湾台北，2023年7月23 - 27日，陈信希（Hsin-Hsi Chen）、杜伟舟（Edward Duh）、黄恒森（Hen-Hsen Huang）、加藤诚（Makoto P. Kato）、乔西安·莫特（Josiane Mothe）和芭芭拉·波布莱特（Barbara Poblete）（编）。美国计算机协会（ACM），2431 - 2435页。https://doi.org/10.1145/3539618.3592071

[14] Jurek Leonhardt, Koustav Rudra, Megha Khosla, Abhijit Anand, and Avishek Anand. 2021. Fast Forward Indexes for Efficient Document Ranking. CoRR abs/2110.06051 (2021). arXiv:2110.06051 https://arxiv.org/abs/2110.06051

[14] 尤雷克·莱昂哈特（Jurek Leonhardt）、库斯塔夫·鲁德拉（Koustav Rudra）、梅加·科斯拉（Megha Khosla）、阿比吉特·阿南德（Abhijit Anand）和阿维谢克·阿南德（Avishek Anand）。2021年。用于高效文档排名的快速前向索引（Fast Forward Indexes for Efficient Document Ranking）。计算机研究存储库（CoRR）论文编号abs/2110.06051（2021年）。预印本服务器arXiv编号：2110.06051 https://arxiv.org/abs/2110.06051

[15] Sean MacAvaney, Andrew Yates, Arman Cohan, and Nazli Goharian. 2019. CEDR: Contextualized Embeddings for Document Ranking. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR 2019, Paris, France, July 21-25, 2019, Benjamin Piwowarski, Max Chevalier, Éric Gaussier, Yoelle Maarek, Jian-Yun Nie, and Falk Scholer (Eds.). ACM, 1101-1104. https://doi.org/10.1145/3331184.3331317

[15] 肖恩·麦卡瓦尼（Sean MacAvaney）、安德鲁·耶茨（Andrew Yates）、阿曼·科汉（Arman Cohan）和纳兹利·戈哈里安（Nazli Goharian）。2019年。CEDR：用于文档排名的上下文嵌入（CEDR: Contextualized Embeddings for Document Ranking）。见《第42届ACM国际信息检索研究与发展会议论文集》（Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval），SIGIR 2019，法国巴黎，2019年7月21 - 25日，本杰明·皮沃瓦尔斯基（Benjamin Piwowarski）、马克斯·舍瓦利耶（Max Chevalier）、埃里克·高斯捷（Éric Gaussier）、约埃尔·马雷克（Yoelle Maarek）、聂建云（Jian-Yun Nie）和福尔克·肖勒（Falk Scholer）（编）。美国计算机协会（ACM），1101 - 1104页。https://doi.org/10.1145/3331184.3331317

[16] Yury A. Malkov and Dmitry A. Yashunin. 2020. Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs. IEEE Trans. Pattern Anal. Mach. Intell. 42, 4 (2020), 824-836. https://doi.org/10.1109/ TPAMI.2018.2889473

[16] 尤里·A·马尔科夫（Yury A. Malkov）和德米特里·A·亚舒宁（Dmitry A. Yashunin）。2020年。使用分层可导航小世界图的高效鲁棒近似最近邻搜索（Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs）。《电气与电子工程师协会模式分析与机器智能汇刊》（IEEE Trans. Pattern Anal. Mach. Intell.）42卷，第4期（2020年），824 - 836页。https://doi.org/10.1109/ TPAMI.2018.2889473

[17] Antonio Mallia, Michal Siedlaczek, Joel M. Mackenzie, and Torsten Suel. 2019. PISA: Performant Indexes and Search for Academia. In Proceedings of the Open-Source IR Replicability Challenge co-located with 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval, OSIRRC@SIGIR 2019, Paris, France, July 25, 2019 (CEUR Workshop Proceedings, Vol. 2409), Ryan Clancy, Nicola Ferro, Claudia Hauff, Jimmy Lin, Tetsuya Sakai, and Ze Zhong Wu (Eds.). CEUR-WS.org, 50-56. https://ceur-ws.org/Vol-2409/docker08.pdf

[17] 安东尼奥·马利亚（Antonio Mallia）、米哈尔·谢德拉切克（Michal Siedlaczek）、乔尔·M·麦肯齐（Joel M. Mackenzie）和托尔斯滕·苏埃尔（Torsten Suel）。2019年。PISA：面向学术界的高性能索引与搜索（PISA: Performant Indexes and Search for Academia）。见《与第42届ACM国际信息检索研究与发展会议同期举办的开源信息检索可复现性挑战会议论文集》（Proceedings of the Open-Source IR Replicability Challenge co-located with 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval），OSIRRC@SIGIR 2019，法国巴黎，2019年7月25日（CEUR研讨会论文集，第2409卷），瑞安·克兰西（Ryan Clancy）、尼古拉·费罗（Nicola Ferro）、克劳迪娅·豪夫（Claudia Hauff）、林吉米（Jimmy Lin）、酒井哲也（Tetsuya Sakai）和吴泽中（Ze Zhong Wu）（编）。CEUR - WS.org，50 - 56页。https://ceur-ws.org/Vol-2409/docker08.pdf

[18] Thong Nguyen, Sean MacAvaney, and Andrew Yates. 2023. A Unified Framework for Learned Sparse Retrieval. In Advances in Information Retrieval - 45th European Conference on Information Retrieval, ECIR 2023, Dublin, Ireland, April 2-6, 2023, Proceedings, Part III (Lecture Notes in Computer Science, Vol. 13982), Jaap Kamps, Lorraine Goeuriot, Fabio Crestani, Maria Maistro, Hideo Joho, Brian Davis, Cathal Gurrin, Udo Kruschwitz, and Annalina Caputo (Eds.). Springer, 101-116. https: //doi.org/10.1007/978-3-031-28241-6_7

[18] 通·阮（Thong Nguyen）、肖恩·麦卡瓦尼（Sean MacAvaney）和安德鲁·耶茨（Andrew Yates）。2023年。学习型稀疏检索的统一框架（A Unified Framework for Learned Sparse Retrieval）。见《信息检索进展 - 第45届欧洲信息检索会议》（Advances in Information Retrieval - 45th European Conference on Information Retrieval），ECIR 2023，爱尔兰都柏林，2023年4月2 - 6日，会议论文集，第三部分（《计算机科学讲义》，第13982卷），亚普·坎普斯（Jaap Kamps）、洛林·戈里奥（Lorraine Goeuriot）、法比奥·克雷斯塔尼（Fabio Crestani）、玛丽亚·梅斯特罗（Maria Maistro）、乔本秀夫（Hideo Joho）、布莱恩·戴维斯（Brian Davis）、卡哈尔·古林（Cathal Gurrin）、乌多·克鲁施维茨（Udo Kruschwitz）和安娜莉娜·卡普托（Annalina Caputo）（编）。施普林格出版社（Springer），101 - 116页。https: //doi.org/10.1007/978-3-031-28241-6_7

[19] Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, and Li Deng. 2016. MS MARCO: A Human Generated MAchine Reading COmprehension Dataset. In Proceedings of the Workshop on Cognitive Computation: Integrating neural and symbolic approaches 2016 co-located with the 30th Annual Conference on Neural Information Processing Systems (NIPS 2016), Barcelona, Spain, December 9, 2016 (CEUR Workshop Proceedings, Vol. 1773), Tarek Richard Besold, Antoine Bordes, Artur S. d'Avila Garcez, and Greg Wayne (Eds.). CEUR-WS.org. https://ceur-ws.org/Vol-1773/CoCoNIPS_2016_paper9.pdf

[19] 特里·阮（Tri Nguyen）、米尔·罗森伯格（Mir Rosenberg）、宋霞（Xia Song）、高剑锋（Jianfeng Gao）、索拉布·蒂瓦里（Saurabh Tiwary）、兰根·马朱姆德（Rangan Majumder）和李登（Li Deng）。2016年。MS MARCO：一个人工生成的机器阅读理解数据集。收录于《认知计算研讨会论文集：整合神经与符号方法（2016年）》，该研讨会与第30届神经信息处理系统年度会议（NIPS 2016）同期举办，于2016年12月9日在西班牙巴塞罗那举行（CEUR研讨会论文集，第1773卷），由塔里克·理查德·贝索尔德（Tarek Richard Besold）、安托万·博尔德斯（Antoine Bordes）、阿图尔·S·达维拉·加尔塞斯（Artur S. d'Avila Garcez）和格雷格·韦恩（Greg Wayne）编辑。CEUR - WS.org。https://ceur - ws.org/Vol - 1773/CoCoNIPS_2016_paper9.pdf

[20] Rodrigo Frassetto Nogueira and Kyunghyun Cho. 2019. Passage Re-ranking with BERT. CoRR abs/1901.04085 (2019). arXiv:1901.04085 http://arxiv.org/abs/1901.

[20] 罗德里戈·弗拉塞托·诺盖拉（Rodrigo Frassetto Nogueira）和赵京焕（Kyunghyun Cho）。2019年。使用BERT进行段落重排序。计算机研究报告库（CoRR），编号abs/1901.04085（2019年）。预印本服务器arXiv：1901.04085 http://arxiv.org/abs/1901.

04085

[21] Stephen E. Robertson, Steve Walker, Susan Jones, Micheline Hancock-Beaulieu, and Mike Gatford. 1994. Okapi at TREC-3. In Proceedings of The Third Text REtrieval Conference, TREC 1994, Gaithersburg, Maryland, USA, November 2-4, 1994 (NIST Special Publication, Vol. 500-225), Donna K. Harman (Ed.). National Institute of Standards and Technology (NIST), 109-126. http://trec.nist.gov/pubs/ trec3/papers/city.ps.gz

[21] 斯蒂芬·E·罗伯逊（Stephen E. Robertson）、史蒂夫·沃克（Steve Walker）、苏珊·琼斯（Susan Jones）、米歇琳·汉考克 - 博勒伊（Micheline Hancock - Beaulieu）和迈克·加特福德（Mike Gatford）。1994年。TREC - 3会议上的Okapi系统。收录于《第三届文本检索会议论文集（TREC 1994）》，会议于1994年11月2 - 4日在美国马里兰州盖瑟斯堡举行（美国国家标准与技术研究院特别出版物，第500 - 225卷），由唐娜·K·哈曼（Donna K. Harman）编辑。美国国家标准与技术研究院（NIST），第109 - 126页。http://trec.nist.gov/pubs/ trec3/papers/city.ps.gz

[22] Keshav Santhanam, Omar Khattab, Christopher Potts, and Matei Zaharia. 2022. PLAID: An Efficient Engine for Late Interaction Retrieval. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management, Atlanta, GA, USA, October 17-21, 2022, Mohammad Al Hasan and Li Xiong (Eds.). ACM, 1747-1756. https://doi.org/10.1145/3511808.3557325

[22] 凯沙夫·桑塔南（Keshav Santhanam）、奥马尔·卡塔布（Omar Khattab）、克里斯托弗·波茨（Christopher Potts）和马特·扎哈里亚（Matei Zaharia）。2022年。PLAID：一种用于后期交互检索的高效引擎。收录于《第31届ACM国际信息与知识管理会议论文集》，会议于2022年10月17 - 21日在美国佐治亚州亚特兰大举行，由穆罕默德·阿尔·哈桑（Mohammad Al Hasan）和李雄（Li Xiong）编辑。美国计算机协会（ACM），第1747 - 1756页。https://doi.org/10.1145/3511808.3557325

[23] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. 2022. ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL 2022, Seattle, WA, United States, July 10-15, 2022, Marine Carpuat, Marie-Catherine de Marneffe, and Iván Vladimir Meza Ruíz (Eds.). Association for Computational Linguistics, 3715-3734. https://doi.org/10.18653/V1/2022.NAACL-MAIN.272

[23] 凯沙夫·桑塔南（Keshav Santhanam）、奥马尔·卡塔布（Omar Khattab）、乔恩·萨德 - 法尔孔（Jon Saad - Falcon）、克里斯托弗·波茨（Christopher Potts）和马特·扎哈里亚（Matei Zaharia）。2022年。ColBERTv2：通过轻量级后期交互实现高效检索。收录于《2022年北美计算语言学协会人类语言技术会议论文集（NAACL 2022）》，会议于2022年7月10 - 15日在美国华盛顿州西雅图举行，由玛丽娜·卡尔普阿（Marine Carpuat）、玛丽 - 凯瑟琳·德·马尔内夫（Marie - Catherine de Marneffe）和伊万·弗拉基米尔·梅萨·鲁伊斯（Iván Vladimir Meza Ruíz）编辑。计算语言学协会，第3715 - 3734页。https://doi.org/10.18653/V1/2022.NAACL - MAIN.272

[24] Xiao Wang, Craig Macdonald, Nicola Tonellotto, and Iadh Ounis. 2021. Pseudo-Relevance Feedback for Multiple Representation Dense Retrieval. In ICTIR '21: The 2021 ACM SIGIR International Conference on the Theory of Information Retrieval, Virtual Event, Canada, July 11, 2021, Faegheh Hasibi, Yi Fang, and Akiko Aizawa (Eds.). ACM, 297-306. https://doi.org/10.1145/3471158.3472250

[24] 王晓（Xiao Wang）、克雷格·麦克唐纳（Craig Macdonald）、尼古拉·托内洛托（Nicola Tonellotto）和伊阿德·乌尼斯（Iadh Ounis）。2021年。多表示密集检索的伪相关反馈。收录于《2021年ACM SIGIR信息检索理论国际会议（ICTIR '21）论文集》，会议以线上形式在加拿大举行，于2021年7月11日召开，由法赫赫·哈西比（Faegheh Hasibi）、方毅（Yi Fang）和秋子·相泽（Akiko Aizawa）编辑。美国计算机协会（ACM），第297 - 306页。https://doi.org/10.1145/3471158.3472250

[25] Xiao Wang, Craig MacDonald, Nicola Tonellotto, and Iadh Ounis. 2023. ColBERT-PRF: Semantic Pseudo-Relevance Feedback for Dense Passage and Document Retrieval. ACM Trans. Web 17, 1 (2023), 3:1-3:39. https://doi.org/10.1145/3572405

[25] 王晓（Xiao Wang）、克雷格·麦克唐纳（Craig Macdonald）、尼古拉·托内洛托（Nicola Tonellotto）和伊阿德·乌尼斯（Iadh Ounis）。2023年。ColBERT - PRF：用于密集段落和文档检索的语义伪相关反馈。《ACM网络事务》第17卷，第1期（2023年），第3:1 - 3:39页。https://doi.org/10.1145/3572405

[26] William Webber, Alistair Moffat, and Justin Zobel. 2010. A similarity measure for indefinite rankings. ACM Trans. Inf. Syst. 28, 4 (2010), 20:1-20:38. https: //doi.org/10.1145/1852102.1852106

[26] 威廉·韦伯（William Webber）、阿利斯泰尔·莫法特（Alistair Moffat）和贾斯汀·佐贝尔（Justin Zobel）。2010年。一种用于不确定排名的相似度度量方法。《ACM信息系统事务》第28卷，第4期（2010年），第20:1 - 20:38页。https: //doi.org/10.1145/1852102.1852106

[27] Giulio Zhou and Jacob Devlin. 2021. Multi-Vector Attention Models for Deep Re-ranking. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021, Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih (Eds.). Association for Computational Linguistics, 5452-5456. https://doi.org/10.18653/V1/2021.EMNLP-MAIN.443

[27] 朱利奥·周（Giulio Zhou）和雅各布·德夫林（Jacob Devlin）。2021年。用于深度重排序的多向量注意力模型。见《2021年自然语言处理经验方法会议论文集》（Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing），EMNLP 2021，线上活动/多米尼加共和国蓬塔卡纳，2021年11月7 - 11日，玛丽 - 弗朗辛·莫恩斯（Marie-Francine Moens）、黄萱菁（Xuanjing Huang）、露西亚·斯佩西亚（Lucia Specia）和斯科特·文涛·易（Scott Wen-tau Yih）（编）。计算语言学协会，5452 - 5456。https://doi.org/10.18653/V1/2021.EMNLP - MAIN.443