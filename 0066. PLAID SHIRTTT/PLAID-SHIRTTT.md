# PLAID SHIRTTT for Large-Scale Streaming Dense Retrieval

# 用于大规模流式密集检索的PLAID分层索引技术（PLAID SHIRTTT）

Dawn Lawrie

道恩·劳里

HLTCOE, Johns Hopkins University Baltimore, Maryland, USA lawrie@jhu.edu

美国马里兰州巴尔的摩市约翰霍普金斯大学人机语言技术卓越中心 邮箱：lawrie@jhu.edu

Efsun Kayi

埃夫孙·卡伊

HLTCOE, Johns Hopkins University Baltimore, Maryland, USA ekayi1@jhu.edu

美国马里兰州巴尔的摩市约翰霍普金斯大学人机语言技术卓越中心 邮箱：ekayi1@jhu.edu

Eugene Yang

尤金·杨

HLTCOE, Johns Hopkins University

约翰霍普金斯大学人机语言技术卓越中心

Baltimore, Maryland, USA

美国马里兰州巴尔的摩市

eugene.yang@jhu.edu

James Mayfield

詹姆斯·梅菲尔德

HLTCOE, Johns Hopkins University Baltimore, Maryland, USA mayfield@jhu.edu

美国马里兰州巴尔的摩市约翰霍普金斯大学人机语言技术卓越中心 邮箱：mayfield@jhu.edu

Douglas W. Oard

道格拉斯·W·奥尔德

University of Maryland

马里兰大学

College Park, Maryland, USA

美国马里兰州大学公园市

oard@umd.edu

## Abstract

## 摘要

PLAID, an efficient implementation of the ColBERT late interaction bi-encoder using pretrained language models for ranking, consistently achieves state-of-the-art performance in monolingual, cross-language, and multilingual retrieval. PLAID differs from ColBERT by assigning terms to clusters and representing those terms as cluster centroids plus compressed residual vectors. While PLAID is effective in batch experiments, its performance degrades in streaming settings where documents arrive over time because representations of new tokens may be poorly modeled by the earlier tokens used to select cluster centroids. PLAID Streaming Hierarchical Indexing that Runs on Terabytes of Temporal Text (PLAID SHIRTTT) addresses this concern using multi-phase incremental indexing based on hierarchical sharding. Experiments on ClueWeb09 and the multilingual NeuCLIR collection demonstrate the effectiveness of this approach both for the largest collection indexed to date by the ColBERT architecture and in the multilingual setting, respectively.

PLAID是基于预训练语言模型对ColBERT后期交互双编码器进行的高效实现，用于排序，在单语言、跨语言和多语言检索中始终能达到最先进的性能。PLAID与ColBERT的不同之处在于，它将词项分配到聚类中，并将这些词项表示为聚类质心加上压缩残差向量。虽然PLAID在批量实验中效果显著，但在文档随时间陆续到达的流式环境中，其性能会下降，因为新标记的表示可能无法被用于选择聚类质心的早期标记很好地建模。基于分层分片的多阶段增量索引技术——适用于处理PB级时态文本的PLAID流式分层索引技术（PLAID SHIRTTT）解决了这一问题。在ClueWeb09和多语言NeuCLIR数据集上的实验分别证明了该方法在ColBERT架构迄今为止索引的最大数据集以及多语言环境中的有效性。

## CCS CONCEPTS

## 计算机协会分类系统概念

- Information systems $\rightarrow$ Search engine indexing; Language models; Web and social media search; Multilingual and cross-lingual retrieval.

- 信息系统 $\rightarrow$ 搜索引擎索引；语言模型；网络与社交媒体搜索；多语言和跨语言检索。

## KEYWORDS

## 关键词

Multi-vector dense retrieval, Streaming Content, Hierarchically Sharded Indexing, Large-scale document collections

多向量密集检索、流式内容、分层分片索引、大规模文档集合

## ACM Reference Format:

## ACM引用格式：

Dawn Lawrie, Efsun Kayi, Eugene Yang, James Mayfield, and Douglas W. Oard. 2024. PLAID SHIRTTT for Large-Scale Streaming Dense Retrieval. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '24), July 14-18, 2024, Washington, DC, USA. ACM, New York, NY, USA, 5 pages. https://doi.org/10.1145/ 3626772.3657964

道恩·劳里（Dawn Lawrie）、埃夫孙·卡伊（Efsun Kayi）、尤金·杨（Eugene Yang）、詹姆斯·梅菲尔德（James Mayfield）和道格拉斯·W·奥尔德（Douglas W. Oard）。2024年。用于大规模流式密集检索的PLAID SHIRTTT。见《第47届ACM信息检索研究与发展国际会议论文集（SIGIR '24）》，2024年7月14 - 18日，美国华盛顿特区。美国纽约州纽约市ACM，5页。https://doi.org/10.1145/ 3626772.3657964

## 1 INTRODUCTION

## 1 引言

Ranked retrieval using pretrained language models (PLMs) has shown great promise on research test collections [9]. Two main architectures have emerged, cross-encoders and bi-encoders [20]. Cross-encoders are generally used as rerankers because they must process the query and the document passages together. Often a lexical matcher like BM25 [21] is used to retrieve the documents to be reranked, meaning that while a cross-encoder can match queries and documents semantically, it will never be presented with documents that lack an exact query string match. ${}^{1}$

使用预训练语言模型（PLMs）进行排序检索在研究测试集上已显示出巨大潜力 [9]。目前出现了两种主要架构：交叉编码器和双编码器 [20]。交叉编码器通常用作重排器，因为它们必须将查询和文档段落一起处理。通常会使用像BM25 [21] 这样的词法匹配器来检索待重排的文档，这意味着虽然交叉编码器可以在语义上匹配查询和文档，但它永远不会处理那些与查询字符串没有精确匹配的文档。${}^{1}$

A bi-encoder, by contrast, encodes document passages separately from the query, enabling encoding in an offline indexing phase using GPUs. At query time, only the query needs to be encoded, which, given its short length, can be done quickly using a CPU. A bi-encoder's dense representations can rank documents that are semantically similar to the query, even without exact string matches. Moreover, if the bi-encoder's token representations are built from a multilingual Pretrained Language Model, documents in languages other than that of the query can also be ranked. This architecture could enable multilingual retrieval on the web. Bi-encoders offer the promise of flexible and effective first-stage ranking.

相比之下，双编码器将文档段落与查询分开编码，从而可以在离线索引阶段使用GPU进行编码。在查询时，只需要对查询进行编码，由于查询长度较短，可以使用CPU快速完成。双编码器的密集表示可以对与查询语义相似的文档进行排序，即使没有精确的字符串匹配。此外，如果双编码器的词元表示是基于多语言预训练语言模型构建的，那么也可以对与查询语言不同的文档进行排序。这种架构可以实现网络上的多语言检索。双编码器有望实现灵活有效的第一阶段排序。

Because of encoder limitations, bi-encoders normally break documents into passages; a useful heuristic is to use the highest passage score as the document score [10]. A bi-encoder can encode a passage using one or many vectors. The single vector approach, for which the present state of the art is Contriever [14], is efficient and effective in monolingual tasks. A query is also represented as a single vector. Passages are ranked by comparing the query vector to the passage vector. But in multilingual tasks such as Cross-Language Information Retrieval (CLIR), it is outperformed by ColBERT [15] (our focus in this paper), where each token is represented by a dense vector. At search time each query token is represented as a vector and passages are ranked based on the passage tokens that are closest to each query token. ColBERT is currently the state of the art for full-collection (i.e., end-to-end) CLIR [16] and Multilingual Information Retrieval (MLIR) [19]. PLAID [24], a space-efficient implementation of ColBERT, is thus an obvious architecture to consider for a high-volume multilingual document stream.

由于编码器的限制，双编码器通常会将文档拆分为段落；一种有用的启发式方法是使用最高段落得分作为文档得分 [10]。双编码器可以使用一个或多个向量对段落进行编码。单向量方法（目前最先进的是Contriever [14]）在单语言任务中高效且有效。查询也表示为单个向量。通过比较查询向量和段落向量对段落进行排序。但在跨语言信息检索（CLIR）等多语言任务中，它的性能不如ColBERT [15]（本文的重点），在ColBERT中，每个词元由一个密集向量表示。在搜索时，每个查询词元表示为一个向量，并根据最接近每个查询词元的段落词元对段落进行排序。ColBERT目前是全集合（即端到端）CLIR [16] 和多语言信息检索（MLIR）[19] 的最先进技术。因此，PLAID [24]（ColBERT的一种节省空间的实现）显然是处理大量多语言文档流时值得考虑的架构。

PLAID was designed for batch settings, because it needs access to all (or nearly all) the documents at the start of indexing. That is impractical in streaming settings, where documents are introduced over time. This paper proposes PLAID SHIRTTT (PLAID Streaming Hierarchical Indexing that Runs on Terabytes of Temporal Text). The key blocker to incremental indexing for streaming in the PLAID architecture is its reliance on cluster centroids for term representation. As the vocabulary in the new documents moves away from that in the documents from which the cluster centroids were built, performance degrades precipitously. This paper proposes hierarchical sharding to adapt PLAID to a streaming setting. Its main contributions include the architecture (PLAID SHIRTTT), ${}^{2}$ evaluation that demonstrates its effectiveness in both monolingual and multilingual settings, and the first known application of a ColBERT variant to a terabyte collection (ClueWeb09 [3]).

PLAID是为批量处理场景设计的，因为它在索引开始时需要访问所有（或几乎所有）文档。这在流式处理场景中是不切实际的，因为文档是随时间引入的。本文提出了PLAID SHIRTTT（运行在数TB时间文本上的PLAID流式分层索引）。PLAID架构中流式增量索引的关键障碍在于它依赖聚类质心来表示词项。随着新文档中的词汇与构建聚类质心的文档中的词汇差异增大，性能会急剧下降。本文提出了分层分片方法，以使PLAID适应流式处理场景。其主要贡献包括该架构（PLAID SHIRTTT）、${}^{2}$ 证明其在单语言和多语言场景中有效性的评估，以及首次将ColBERT变体应用于数TB的文档集合（ClueWeb09 [3]）。

---

<!-- Footnote -->

classroom use is granted without fee provided that copies are not made or distributed

允许在课堂上免费使用，但前提是不得为盈利或商业利益制作或分发副本

for profit or commercial advantage and that copies bear this notice and the full citation SIGIR '24, July 14-18, 2024, Washington, DC, USA

并且副本需带有此声明和完整引用信息：SIGIR '24，2024年7月14 - 18日，美国华盛顿特区

${}^{1}$ Of course,the query for a lexical matcher can result from automatic query ,rewriting and thus need not be the same as the query presented to the cross-encoder.

${}^{1}$ 当然，词法匹配器的查询可能是自动查询改写的结果，因此不必与提供给交叉编码器的查询相同。

<!-- Footnote -->

---

<!-- Media -->

Table 1: Effectiveness of centroids built at different points along the Chinese NeuCLIR stream.

表1：在中国NeuCLIR流的不同时间点构建的质心的有效性。

<table><tr><td>Stream Percent</td><td>5%</td><td>75%</td><td>90%</td><td>100%</td></tr><tr><td>nDCG@20</td><td>0.047</td><td>0.336</td><td>0.384</td><td>0.440</td></tr><tr><td>R@1000</td><td>0.301</td><td>0.540</td><td>0.655</td><td>0.795</td></tr></table>

<table><tbody><tr><td>流百分比</td><td>5%</td><td>75%</td><td>90%</td><td>100%</td></tr><tr><td>20位置归一化折损累积增益（nDCG@20）</td><td>0.047</td><td>0.336</td><td>0.384</td><td>0.440</td></tr><tr><td>R@1000</td><td>0.301</td><td>0.540</td><td>0.655</td><td>0.795</td></tr></tbody></table>

<!-- Media -->

## 2 BACKGROUND

## 2 背景

PLAID [24] and ColBERTv2 [25] addressed one of the main disadvantages of ColBERT [15]-the large index size (3 GB of text requires a multi-vector retrieval index over ${170}\mathrm{{GB}}$ ). This space is mainly consumed by dense term vectors. PLAID reduced index storage by approximating each term vector as the combination of one of a few canonical vectors with a residual. Canonical vectors are created by K-means clustering of term vectors drawn from a sampled set of documents, then selecting the centroid of each cluster. During indexing, the centroid nearest each document term is identified and a residual is computed to represent the distance between the centroid and the term vector. Only the cluster id and the residual is stored. The fewer bits used to represent the residual, the less storage required for the index but also the more lossy the compression. At retrieval time, document scores are the sum of the distance between each query term its nearest document term; closer document terms get higher scores.

PLAID [24]和ColBERTv2 [25]解决了ColBERT [15]的一个主要缺点——索引规模大（3GB文本需要一个超过${170}\mathrm{{GB}}$的多向量检索索引）。这个空间主要被密集词向量占用。PLAID通过将每个词向量近似表示为几个规范向量之一与一个残差的组合来减少索引存储。规范向量是通过对从一组抽样文档中提取的词向量进行K均值聚类，然后选择每个聚类的质心而创建的。在索引过程中，识别出每个文档词最近的质心，并计算一个残差来表示质心与词向量之间的距离。只存储聚类ID和残差。用于表示残差的比特数越少，索引所需的存储空间就越少，但压缩的损失也越大。在检索时，文档得分是每个查询词与其最近的文档词之间距离的总和；距离更近的文档词得分更高。

When using PLAID in a streaming environment, we would like fixed cluster centroids. Then all newly arriving documents could be indexed by finding each term's nearest centroid from this fixed set and computing their residuals. Table 1 shows that clustering once early in the stream does not work well. For instance, if centroids are generated from the first $5\%$ of the chronologically ordered Chinese documents in the NeuCLIR test collection (see Section 4), nDCG@20 drops precipitously from 0.440 to 0.047 . Even waiting until 75% of the stream has arrived still yields rather poor results (nDCG@20 0.336 ). The reason for this degradation is that language use changes over time. The NeuCLIR Collection spans 2016 to 2021, so the centroids created early in the stream do not, for example, have a cluster for COVID, and the nearest centroid cannot represent that concept well (because compressed residuals are lossy). One solution is to periodically re-compute cluster centroids. Doing so results in new centroids, thus requiring all prior documents to be re-indexed if retrospective search is to be supported.

在流式环境中使用PLAID时，我们希望有固定的聚类质心。这样，所有新到达的文档都可以通过从这个固定集合中找到每个词最近的质心并计算它们的残差来进行索引。表1显示，在流的早期进行一次聚类效果不佳。例如，如果质心是从NeuCLIR测试集中按时间顺序排列的中文文档的前$5\%$部分生成的（见第4节），nDCG@20会从0.440急剧下降到0.047。即使等到流中75%的文档到达，结果仍然相当糟糕（nDCG@20为0.336）。这种性能下降的原因是语言使用会随时间变化。NeuCLIR数据集涵盖了2016年到2021年，因此在流的早期创建的质心，例如，没有为“COVID”创建一个聚类，并且最近的质心不能很好地表示这个概念（因为压缩后的残差存在损失）。一种解决方案是定期重新计算聚类质心。这样做会产生新的质心，因此如果要支持回溯搜索，就需要对所有先前的文档重新进行索引。

<!-- Media -->

<!-- figureText: $\left| \text{Shard}\right|  = A$ Stream of Documents Sn ${S}_{i,j - 1}$ ${S}_{n,0}$ ${S}_{n,k}$ ${S}_{1,j - 1,0}$ ${S}_{1,j,0}$ ${S}_{n,0,0}$ ${S}_{n,k,0}$ $\left| \text{Shard}\right|  = B$ $\left| {\mathrm{S}}_{0,0}\right|$ ... ${\mathrm{S}}_{0,k}$ ${S}_{1,0}$ $\left| \text{Shard}\right|  < B$ ${\mathrm{S}}_{0.0.0}$ ${\mathrm{S}}_{0,k,0}$ ${\mathrm{S}}_{{1.0},0}$ ${t}_{0}$ -->

<img src="https://cdn.noedgeai.com/0195ae21-f96a-78ce-bf04-b2e4830ee9af_1.jpg?x=926&y=243&w=716&h=179&r=0"/>

Figure 1: The sharded stream. Large shards of size $A$ are further partitioned into shards of $B$ documents. At any point in the stream, the last shard is incomplete. The shaded boxes represent the shards that make up the index at ${t}_{\text{now }}$ .

图1：分片流。大小为$A$的大分片进一步划分为包含$B$个文档的小分片。在流的任何时刻，最后一个分片都是不完整的。阴影框表示在${t}_{\text{now }}$时刻构成索引的分片。

<!-- Media -->

Sharding is a common approach when collection size exceeds the capacity of a single index server [1]. Streaming adds the additional complexity of non-stationary statistics; from the first TREC routing task it was understood that in streaming tasks, collection statistics must be modeled using prior data [2]. Many streaming evaluations, such as the TREC Filtering [22] and Knowledge Base Acceleration (KBA) [13] tracks, focus on making yes/no decisions as documents arrive; in contrast, our focus is on optimally supporting ranked retrieval over the full collection through the current time. We therefore introduce PLAID SHIRTTT, which uses hierarchical sharding to balance indexing efficiency with the efficiency and effectiveness of full-collection ranked retrieval in streaming settings.

当数据集规模超过单个索引服务器的容量时，分片是一种常见的方法[1]。流式处理增加了非平稳统计的额外复杂性；从第一次TREC路由任务中可以了解到，在流式任务中，必须使用先验数据对数据集统计信息进行建模[2]。许多流式评估，如TREC过滤[22]和知识库加速（KBA）[13]赛道，专注于在文档到达时做出是/否的决策；相比之下，我们的重点是在当前时间内通过整个数据集最优地支持排序检索。因此，我们引入了PLAID SHIRTTT，它使用分层分片来平衡流式环境中索引效率与全数据集排序检索的效率和有效性。

## 3 PLAID SHIRTTT

## 3 PLAID SHIRTTT

PLAID's cluster centroids are themselves derived from an underlying PLM such as BERT [12]; ultimately the PLM determines document scores. Our approach to indexing streaming documents accumulates documents into shards. Each shard model is the set of dense vectors representing the cluster centroids of the document terms in that shard. There is tension between creating as large a shard as possible (to minimize the number of shards that must be searched) and creating shard models for newly arrived documents as soon as possible (since a shard model based on earlier documents can be suboptimal). PLAID SHIRTTT addresses this tension by indexing each document a fixed, small number of times to balance document ranking effectiveness against the efficiency costs of re-indexing and of having too many shards. Figure 1 illustrates re-indexing three times. The first time a document $d$ is indexed,a prior shard model is used. Once a sufficient number $B$ of documents has arrived a shard model containing document $d$ is created,and document $d$ is re-indexed. Once $k \times  B = A$ documents have arrived,the final shard model containing document $d$ is created,and document $d$ is once again re-indexed. The edge case of the start of the stream is handled by running ColBERT V1 (without centroids or residuals) until enough documents have arrived for an initial shard model.

PLAID的聚类质心本身是从诸如BERT [12]这样的基础预训练语言模型（PLM）中派生出来的；最终，预训练语言模型决定文档得分。我们对流式文档进行索引的方法是将文档累积成分片。每个分片模型是表示该分片中文档词聚类质心的一组密集向量。在创建尽可能大的分片（以最小化必须搜索的分片数量）和尽快为新到达的文档创建分片模型（因为基于早期文档的分片模型可能不是最优的）之间存在矛盾。PLAID SHIRTTT通过对每个文档进行固定的少量次数的索引来解决这个矛盾，以平衡文档排序效果与重新索引和分片过多带来的效率成本。图1展示了进行三次重新索引的情况。文档$d$第一次被索引时，使用先前的分片模型。一旦有足够数量（$B$）的文档到达，就会创建一个包含文档$d$的分片模型，并对文档$d$进行重新索引。一旦有$k \times  B = A$个文档到达，就会创建包含文档$d$的最终分片模型，并再次对文档$d$进行重新索引。流开始的特殊情况通过运行ColBERT V1（不使用质心或残差）来处理，直到有足够的文档到达以创建初始分片模型。

Hierarchical sharding limits the number of shards that need to be searched at any time. Each shard requires a CPU with sufficient memory to load the inverted cluster centroid index to support interactive search. For instance, searching one hundred shards requires one hundred CPUs, each with access to hundreds of gigabytes of memory. Thus,the optimal settings of $A$ and $B$ depend on the rate of the stream and the resources available to support interactive search. It may be advantageous to introduce more layers of the hierarchy to maintain a reasonable number of shards. Since each layer of the hierarchy involves re-indexing the documents seen so far using GPUs, this cost must also be considered when setting shard sizes. Finally,PLAID requires at least 2000 passages, ${}^{3}$ (each document may comprise more than one passage). We choose $A$ to create the largest shards our hardware will accommodate,and $B$ to balance the number of shards with the arrival rate (with larger $B$ , documents will be represented by older shard models longer). We assume that when implementing PLAID SHIRTTT that both the data-rates and how long documents must remain in the index are known.

分层分片限制了任何时候需要搜索的分片数量。每个分片都需要一个配备足够内存的CPU来加载倒排聚类质心索引，以支持交互式搜索。例如，搜索一百个分片需要一百个CPU，每个CPU都能访问数百GB的内存。因此，$A$和$B$的最优设置取决于数据流的速率以及可用于支持交互式搜索的资源。引入更多层次的分层结构以维持合理数量的分片可能是有利的。由于分层结构的每一层都涉及使用GPU对到目前为止所看到的文档进行重新索引，因此在设置分片大小时也必须考虑这一成本。最后，PLAID至少需要2000个段落，${}^{3}$（每个文档可能包含多个段落）。我们选择$A$来创建我们的硬件所能容纳的最大分片，并选择$B$来平衡分片数量和文档到达率（$B$越大，文档由较旧的分片模型表示的时间就越长）。我们假设在实现PLAID SHIRTTT时，数据速率以及文档必须在索引中保留的时长都是已知的。

---

<!-- Footnote -->

${}^{2}$ Code available at https://github.com/hltcoe/colbert-x.

${}^{2}$ 代码可在https://github.com/hltcoe/colbert-x获取。

<!-- Footnote -->

---

<!-- Media -->

Table 2: Collection Statistics

表2：数据集统计信息

<table><tr><td rowspan="2">Collection TREC Tracks</td><td>ClueWeb09</td><td colspan="2">NeuCLIR</td></tr><tr><td>Web 09-122012</td><td>2022</td><td>2023</td></tr><tr><td>#of Docs</td><td>504M</td><td colspan="2">10.00M</td></tr><tr><td>#of Passages</td><td>31B</td><td colspan="2">58.9M</td></tr><tr><td>#of Topics</td><td>20050</td><td>41</td><td>65</td></tr><tr><td>#Shards</td><td>108</td><td colspan="2">21</td></tr><tr><td>Sizes $\left( {A/B}\right)$</td><td>5M/500K</td><td colspan="2">500K/100K</td></tr></table>

<table><tbody><tr><td rowspan="2">TREC赛道集合</td><td>ClueWeb09（网络数据集名称，可不译）</td><td colspan="2">NeuCLIR（可能是特定的研究项目或技术名称，可不译）</td></tr><tr><td>网络09 - 122012</td><td>2022</td><td>2023</td></tr><tr><td>文档数量</td><td>504M</td><td colspan="2">10.00M</td></tr><tr><td>段落数量</td><td>31B</td><td colspan="2">58.9M</td></tr><tr><td>主题数量</td><td>20050</td><td>41</td><td>65</td></tr><tr><td>分片数量</td><td>108</td><td colspan="2">21</td></tr><tr><td>大小 $\left( {A/B}\right)$</td><td>500万/50万</td><td colspan="2">50万/10万</td></tr></tbody></table>

<!-- Media -->

At search time, each active shard is searched using PLAID's standard approach. Because all document scores are based on the same PLM (differing only in the effect of the lossy residuals), they are comparable and can be merged. Efficient implementation will search shards in parallel, merging results only for the highest ranked documents from each shard.

在搜索时，使用PLAID（PLAID是一种检索方法，原文未明确其全称）的标准方法对每个活动分片进行搜索。由于所有文档得分都基于同一个预训练语言模型（PLM，Pre-trained Language Model）（仅在有损残差的影响方面有所不同），因此它们具有可比性，可以进行合并。高效的实现方式将并行搜索各个分片，仅对每个分片中排名最高的文档的结果进行合并。

## 4 EXPERIMENT DESIGN

## 4 实验设计

Our experiments seek to answer three questions: (1) Is PLAID SHIRTTT effective for large collections that arrive as a stream? (2) How efficient is PLAID SHIRTTT? and (3) How effective is PLAID SHIRTTT for MLIR? Table 2 provides statistics for our two test collections. We chose ClueWeb09 [3] because its size matches our aspirations to work at large scale; retrieval evaluation is supported by extant relevance judgments for a large number of topics, and the documents can be date-ordered. This collection was used in four years of the TREC Web Track, where web-style topics of the era were developed. We report results over all 200 topics [4-7]. The query field of a topic was often vague; however, because dense retrieval benefits from queries with more context, we combined the query and description fields to form our queries. The reusability of this collection is questionable [23] when using evaluation measures that treat unjudged documents as not relevant. We follow Sakai [23] in using the compressed-list metrics ${\mathrm{{MAP}}}^{\prime }$ and ${\mathrm{{nDCG}}}^{\prime }@{20}$ that treat unjudged documents as not retrieved.

我们的实验旨在回答三个问题：（1）PLAID SHIRTTT（原文未明确其全称）对于以流形式到达的大型文档集合是否有效？（2）PLAID SHIRTTT的效率如何？（3）PLAID SHIRTTT在多语言信息检索（MLIR，Multilingual Information Retrieval）方面的效果如何？表2提供了我们两个测试文档集合的统计信息。我们选择ClueWeb09 [3]是因为其规模符合我们进行大规模研究的目标；大量主题的现有相关性判断支持对其进行检索评估，并且文档可以按日期排序。该文档集合在四年的文本检索会议网络赛道（TREC Web Track）中被使用，在该赛道中开发了当时的网络风格主题。我们报告了所有200个主题的结果 [4 - 7]。主题的查询字段通常比较模糊；然而，由于密集检索受益于具有更多上下文的查询，我们将查询字段和描述字段组合起来形成我们的查询。当使用将未判断文档视为不相关的评估指标时，该文档集合的可复用性存在疑问 [23]。我们遵循Sakai [23]的方法，使用压缩列表指标 ${\mathrm{{MAP}}}^{\prime }$和${\mathrm{{nDCG}}}^{\prime }@{20}$，这些指标将未判断文档视为未被检索到。

ClueWeb emphasizes scale, whereas NeuCLIR emphasizes mul-tilinguality. With ten million documents from the news subset of CommonCrawl, NeuCLIR is presently the largest available test collection for MLIR; It is tiny though relative to ClueWeb, and thus has relatively small shards. The NeuCLIR MLIR task is to retrieve relevant documents from any of the collection's three languages (Chinese, Persian, Russian). For queries, we follow the majority of TREC 2022 and 2023 NeuCLIR track participants [17, 18] in using the title and description fields of each of the 41 and 65 topics as queries, respectively. We report nDCG@20, the primary measure reported by the TREC 2022 and 2023 NeuCLIR tracks, as well as other measures.

ClueWeb强调规模，而NeuCLIR强调多语言性。NeuCLIR包含来自CommonCrawl新闻子集的一千万个文档，目前是可用于多语言信息检索（MLIR）的最大测试文档集合；不过，相对于ClueWeb而言，它规模较小，因此分片也相对较小。NeuCLIR的多语言信息检索任务是从该文档集合的三种语言（中文、波斯语、俄语）中检索相关文档。对于查询，我们遵循2022年和2023年文本检索会议NeuCLIR赛道大多数参与者 [17, 18]的做法，分别使用41个和65个主题的标题和描述字段作为查询。我们报告归一化折损累积增益（nDCG@20，Normalized Discounted Cumulative Gain at 20），这是2022年和2023年文本检索会议NeuCLIR赛道报告的主要指标，以及其他指标。

For each collection, we simulate a stream by ordering the documents by date, arbitrarily ordering documents with the same timestamp. ${}^{4}$ We extract text from web pages using the newspaper ${}^{5}$ Python package,which also associates a date with each article. ${}^{6}$ If newspaper cannot identify a date, we examine the warc header for dates, preferring the last-modified field over the date field. If this too fails, we use the article crawl date. While NeuCLIR was released with text extracted by newspaper, ClueWeb09 was released as raw web pages. About three million documents that fail processing or lack both a title and content once processed are not indexed.

对于每个文档集合，我们通过按日期对文档进行排序来模拟流，对具有相同时间戳的文档进行任意排序。${}^{4}$ 我们使用newspaper ${}^{5}$ Python包从网页中提取文本，该包还会为每篇文章关联一个日期。${}^{6}$ 如果newspaper无法识别日期，我们会检查网络存档（WARC，Web ARChive）头文件中的日期，优先选择最后修改字段而非日期字段。如果这也失败了，我们将使用文章的抓取日期。虽然NeuCLIR发布时附带了使用newspaper提取的文本，但ClueWeb09是以原始网页的形式发布的。大约三百万个在处理过程中失败或处理后既没有标题也没有内容的文档未被索引。

Our ClueWeb09 runs use the ColBERTv2 checkpoint fine-tuned from a BERT [12] Base model, as released by the ColBERT authors [25]. For NeuCLIR, we use the ColBERT-X model, fine-tuned with multilingual Translate-Train [19] from XLM-RoBERTa Large model [8] on English to Chinese, Persian, and Russian. We store one bit for each residual vector dimension. At search time, we retrieve the top fifty (Web topics) or one thousand (NeuCLIR topics) passages from each shard and aggregate passage scores with MaxP [10] to form document scores. Table 2 shows how we set the values of $A$ and $B$ for large and small shards respectively. The sizes for ClueWeb reflect our recommendation to balance hardware limits against number of shards; NeuCLIR sizes were selected to force creation of a non-trivial number of shards.

我们在ClueWeb09上的运行使用了基于BERT [12]基础模型微调的ColBERTv2检查点，该检查点由ColBERT的作者发布 [25]。对于NeuCLIR，我们使用ColBERT - X模型，该模型在英语到中文、波斯语和俄语的多语言翻译训练（Translate - Train） [19]中基于XLM - RoBERTa大型模型 [8]进行了微调。我们为每个残差向量维度存储一位信息。在搜索时，我们从每个分片中检索前五十个（网络主题）或一千个（NeuCLIR主题）段落，并使用最大段落得分聚合（MaxP，Maximum Passage Aggregation） [10]方法聚合段落得分以形成文档得分。表2显示了我们分别为大型和小型分片设置 $A$和$B$值的方式。ClueWeb的分片大小反映了我们在硬件限制和分片数量之间进行平衡的建议；选择NeuCLIR的分片大小是为了强制创建一定数量的非平凡分片。

## 5 RESULTS

## 5 结果

Our first research question asks whether PLAID SHIRTTT is effective for a large streaming collection. We use ClueWeb09 for these experiments. Results in Table 3 report performance on topics from 2012, alongside the best run from that year's TREC Web track for reference, uogTrA44xu [7]. These runs are not comparable because the latter used only the topic's short and often vague Query (Q) field, while our runs use the Query and Description (Q+D) fields as the query. We include the reference run uogTrA44xu to verify that our prime measures are similar to those of the track participants. Note that non-prime metrics are much lower, due to the low number of judged documents in the top twenty. Our main result is over all 200 topics for years 2009 to 2012. PLAID SHIRTTT significantly outperforms our BM25 baseline on both nDCG ${}^{\prime }$ @20 and ${\mathrm{{MAP}}}^{\prime }$ by a two-tailed paired $t$ -test with Bonferroni correction for two tests at $p > {0.05}$ . Compared to the Oracle Shard Model,which creates a single shard model for all of ClueWeb09, PLAID SHIRTTT achieves ${96}\%$ of its nDCG ${}^{\prime }@{20}$ performance. This indicates that document scores from different shards are remarkably comparable, leading to good hierarchical sharding effectiveness. We conclude that PLAID SHIRTTT is effective at scale, outperforming a BM25 lexical baseline at ranking judged relevant documents and judged non-relevant documents.

我们的第一个研究问题是，PLAID SHIRTTT（格子衬衫算法，此处为假设名称，需根据实际情况确定）对于大规模流式集合是否有效。我们在这些实验中使用了ClueWeb09数据集。表3中的结果报告了2012年主题的性能，同时列出了当年TREC网络赛道的最佳运行结果以供参考，即uogTrA44xu [7]。这些运行结果不可直接比较，因为后者仅使用了主题简短且往往模糊的查询（Q）字段，而我们的运行使用了查询和描述（Q + D）字段作为查询。我们纳入参考运行uogTrA44xu是为了验证我们的主要指标与赛道参与者的指标相似。请注意，非主要指标要低得多，这是因为前二十名中被评判的文档数量较少。我们的主要结果涵盖了2009年至2012年的所有200个主题。通过双尾配对$t$检验并在$p > {0.05}$水平上进行Bonferroni校正，PLAID SHIRTTT在nDCG ${}^{\prime }$ @20和${\mathrm{{MAP}}}^{\prime }$指标上显著优于我们的BM25基线。与为整个ClueWeb09创建单一分片模型的Oracle分片模型相比，PLAID SHIRTTT达到了其nDCG ${}^{\prime }@{20}$性能的${96}\%$。这表明不同分片的文档得分具有显著的可比性，从而实现了良好的分层分片效果。我们得出结论，PLAID SHIRTTT在大规模场景下是有效的，在对评判为相关的文档和评判为不相关的文档进行排名时，其性能优于BM25词法基线。

Our second question is what data rate can be accommodated with a specific hardware configuration. Here we compare PLAID SHIRTTT to BM25; the Oracle model is not considered because it does not represent the streaming setting that PLAID SHIRTTT addresses. Accounting for re-indexing as small shards are merged into large shards, indexing the terabyte-scale ClueWeb09 collection requires about 71 days on an NVidia V100 GPU. The number of documents per shard was fixed as shown in Table 2; the number of passages varied from ${24}\mathrm{M}$ to ${51}\mathrm{M}$ for large shards with $A$ documents, and from $2\mathrm{M}$ to $7\mathrm{M}$ for smaller shards with $B$ documents. Large shards took on average seven hours to build, while smaller shards required around thirty minutes. Thus we could index well over half a million documents per hour on this hardware. Since shards can be searched in parallel on separate CPUs, total search time is the search time of a large shard plus the time to merge results; this averaged 1.3 seconds per query on ClueWeb09. To field this configuration requires 108 CPUs each with ${200}\mathrm{{GB}}$ of memory to hold the indexes loaded into memory and service the queries. The index requires 8.4 TB of disk. For contrast, the sparse BM25 index used 0.4 TB of disk, took 36 days of CPU time to build, and BM25 query execution averaged 0.07 seconds on 1 CPU with 2GB memory. Thus PLAID SHIRTTTrequires significantly more compute resources to build the index and to search. Operationally, the added compute resources will need to be justified by the increase in effectiveness.

我们的第二个问题是，特定的硬件配置能够适应怎样的数据速率。在这里，我们将PLAID SHIRTTT与BM25进行比较；不考虑Oracle模型，因为它不代表PLAID SHIRTTT所针对的流式设置。考虑到将小分片合并为大分片时的重新索引操作，在NVIDIA V100 GPU上对TB级的ClueWeb09数据集进行索引大约需要71天。每个分片的文档数量固定，如表2所示；对于包含$A$个文档的大分片，段落数量从${24}\mathrm{M}$到${51}\mathrm{M}$不等，对于包含$B$个文档的小分片，段落数量从$2\mathrm{M}$到$7\mathrm{M}$不等。大分片平均需要七个小时来构建，而小分片大约需要三十分钟。因此，在这种硬件上，我们每小时可以索引超过五十万份文档。由于可以在不同的CPU上并行搜索分片，总搜索时间是大分片的搜索时间加上合并结果的时间；在ClueWeb09上，每个查询的平均搜索时间为1.3秒。要部署这种配置，需要108个CPU，每个CPU配备${200}\mathrm{{GB}}$的内存来存储加载到内存中的索引并处理查询。索引需要8.4 TB的磁盘空间。相比之下，稀疏的BM25索引使用了0.4 TB的磁盘空间，构建需要36天的CPU时间，并且在配备2GB内存的1个CPU上，BM25查询执行的平均时间为0.07秒。因此，PLAID SHIRTTT在构建索引和搜索时需要显著更多的计算资源。在实际操作中，增加的计算资源需要通过性能的提升来证明其合理性。

---

<!-- Footnote -->

${}^{4}$ https://huggingface.co/datasets/hltcoe/plaid-shirttt-doc-date

${}^{4}$ https://huggingface.co/datasets/hltcoe/plaid-shirttt-doc-date

${}^{5}$ https://pypi.org/project/newspaper3k/

${}^{5}$ https://pypi.org/project/newspaper3k/

${}^{6}$ Persian dates before 1900 are converted to the Gregorian calendar.

${}^{6}$ 1900年以前的波斯日期已转换为公历日期。

${}^{3}$ https://github.com/stanford-futuredata/ColBERT/issues/181#issuecomment- 1613956350

${}^{3}$ https://github.com/stanford-futuredata/ColBERT/issues/181#issuecomment- 1613956350

<!-- Footnote -->

---

<!-- Media -->

Table 3: ClueWeb09 Results. ${}^{ \dagger  }$ indicates statistically significant improvement over BM25. PLAID SHIRTTT percentages are percentage of Oracle.

表3：ClueWeb09结果。${}^{ \dagger  }$表示相对于BM25有统计学上的显著改进。PLAID SHIRTTT的百分比是相对于Oracle的百分比。

<table><tr><td>Topics</td><td>Queries</td><td>Approach</td><td>nDCG’@20</td><td>${\mathrm{{MAP}}}^{\prime }$</td><td>nDCG@20</td><td>MAP</td><td>Jg@20</td></tr><tr><td>Web12</td><td>Q</td><td>uogTrA44xu</td><td>0.348</td><td>0.331</td><td>0.339</td><td>0.217</td><td>1.000</td></tr><tr><td>Web12</td><td>Q+D</td><td>BM25</td><td>0.355</td><td>0.245</td><td>0.070</td><td>0.049</td><td>0.194</td></tr><tr><td>Web12</td><td>Q+D</td><td>Oracle Model</td><td>0.457</td><td>0.296</td><td>0.113</td><td>0.060</td><td>0.160</td></tr><tr><td>Web12</td><td>Q+D</td><td>PLAID SHIRTTT</td><td>0.437</td><td>0.288</td><td>0.100</td><td>0.058</td><td>0.152</td></tr><tr><td>Web09-12</td><td>Q+D</td><td>BM25</td><td>0.323</td><td>0.217</td><td>0.094</td><td>0.058</td><td>0.300</td></tr><tr><td>Web09-12</td><td>Q+D</td><td>Oracle Model</td><td>${\mathbf{{0.448}}}^{ \dagger  }$</td><td>${\mathbf{{0.278}}}^{ \dagger  }$</td><td>0.132</td><td>0.079</td><td>0.216</td></tr><tr><td>Web09-12</td><td>Q+D</td><td>PLAID SHIRTTT</td><td>${0.431}^{ \dagger  }\left( {{96}\% }\right)$</td><td>${0.272}^{ \dagger  }\left( {{98}\% }\right)$</td><td>0.129 (98%)</td><td>0.077 (97%)</td><td>0.215</td></tr></table>

<table><tbody><tr><td>主题</td><td>查询</td><td>方法</td><td>归一化折损累积增益（nDCG’@20）</td><td>${\mathrm{{MAP}}}^{\prime }$</td><td>归一化折损累积增益（nDCG@20）</td><td>平均准确率均值（MAP）</td><td>Jg@20</td></tr><tr><td>Web12</td><td>Q</td><td>uogTrA44xu</td><td>0.348</td><td>0.331</td><td>0.339</td><td>0.217</td><td>1.000</td></tr><tr><td>Web12</td><td>查询+文档（Q+D）</td><td>二元独立模型（BM25）</td><td>0.355</td><td>0.245</td><td>0.070</td><td>0.049</td><td>0.194</td></tr><tr><td>Web12</td><td>查询+文档（Q+D）</td><td>神谕模型（Oracle Model）</td><td>0.457</td><td>0.296</td><td>0.113</td><td>0.060</td><td>0.160</td></tr><tr><td>Web12</td><td>查询+文档（Q+D）</td><td>格子衬衫</td><td>0.437</td><td>0.288</td><td>0.100</td><td>0.058</td><td>0.152</td></tr><tr><td>Web09 - 12</td><td>查询+文档（Q+D）</td><td>二元独立模型（BM25）</td><td>0.323</td><td>0.217</td><td>0.094</td><td>0.058</td><td>0.300</td></tr><tr><td>Web09 - 12</td><td>查询+文档（Q+D）</td><td>神谕模型（Oracle Model）</td><td>${\mathbf{{0.448}}}^{ \dagger  }$</td><td>${\mathbf{{0.278}}}^{ \dagger  }$</td><td>0.132</td><td>0.079</td><td>0.216</td></tr><tr><td>Web09 - 12</td><td>查询+文档（Q+D）</td><td>格子衬衫</td><td>${0.431}^{ \dagger  }\left( {{96}\% }\right)$</td><td>${0.272}^{ \dagger  }\left( {{98}\% }\right)$</td><td>0.129 (98%)</td><td>0.077 (97%)</td><td>0.215</td></tr></tbody></table>

Table 4: NeuCLIR MLIR Results. ${}^{ \dagger  }$ indicates statistically significant improvement over PSQ.

表4：NeuCLIR多语言信息检索（MLIR）结果。${}^{ \dagger  }$表示相较于伪查询扩展（PSQ）有统计学上的显著提升。

<table><tr><td>Topics</td><td>Approach</td><td>nDCG@20</td><td>MAP</td><td>R@100</td><td>R@1000</td><td>Jg@20</td></tr><tr><td>NeuCLIR22</td><td>PSQ+HMM w/ Score Fusion</td><td>0.315</td><td>0.195</td><td>0.269</td><td>0.594</td><td>0.901</td></tr><tr><td>NeuCLIR22</td><td>Oracle Model</td><td>0.375</td><td>0.236</td><td>0.330</td><td>0.612</td><td>0.898</td></tr><tr><td>NeuCLIR22</td><td>PLAID SHIRTTT</td><td>${\mathbf{{0.381}}}^{ \dagger  }$</td><td>0.228 (97%)</td><td>0.306 (93%)</td><td>0.602 (98%)</td><td>0.901</td></tr><tr><td>NeuCLIR23</td><td>PSQ+HMM w/ Score Fusion</td><td>0.289</td><td>0.225</td><td>0.402</td><td>0.0.693</td><td>0.933</td></tr><tr><td>NeuCLIR23</td><td>Oracle Model</td><td>0.330</td><td>0.281</td><td>0.468</td><td>0.760</td><td>0.922</td></tr><tr><td>NeuCLIR23</td><td>PLAID SHIRTTT</td><td>${\mathbf{{0.337}}}^{ \dagger  }$</td><td>0.268 (95%)</td><td>0.431 (92%)</td><td>0.757 (100%)</td><td>0.927</td></tr></table>

<table><tbody><tr><td>主题</td><td>方法</td><td>归一化折损累积增益@20（nDCG@20）</td><td>平均准确率均值（MAP）</td><td>R@100</td><td>R@1000</td><td>Jg@20</td></tr><tr><td>NeuCLIR22</td><td>带分数融合的伪查询生成（PSQ）+隐马尔可夫模型（HMM）</td><td>0.315</td><td>0.195</td><td>0.269</td><td>0.594</td><td>0.901</td></tr><tr><td>NeuCLIR22</td><td>神谕模型</td><td>0.375</td><td>0.236</td><td>0.330</td><td>0.612</td><td>0.898</td></tr><tr><td>NeuCLIR22</td><td>格子衬衫</td><td>${\mathbf{{0.381}}}^{ \dagger  }$</td><td>0.228 (97%)</td><td>0.306 (93%)</td><td>0.602 (98%)</td><td>0.901</td></tr><tr><td>NeuCLIR23</td><td>带分数融合的伪查询生成（PSQ）+隐马尔可夫模型（HMM）</td><td>0.289</td><td>0.225</td><td>0.402</td><td>0.0.693</td><td>0.933</td></tr><tr><td>NeuCLIR23</td><td>神谕模型</td><td>0.330</td><td>0.281</td><td>0.468</td><td>0.760</td><td>0.922</td></tr><tr><td>NeuCLIR23</td><td>格子衬衫</td><td>${\mathbf{{0.337}}}^{ \dagger  }$</td><td>0.268 (95%)</td><td>0.431 (92%)</td><td>0.757 (100%)</td><td>0.927</td></tr></tbody></table>

<!-- Media -->

Our third question is about MLIR. Table 4 compares PLAID SHIRTTT to two other approaches on the NeuCLIR 2022 and 2023 datasets: PSQ+HMM [11, 26-28] with score fusion, and an Oracle Shard Model. PSQ+HMM is a fast non-neural baseline that relies on translation tables from statistical machine translation. This approach was chosen for its similarity to monolingual BM25 in being entirely non-neural. Since PSQ+HMM is a CLIR algorithm, MLIR is facilitated by score fusion across the languages without normalization. When PLAID SHIRTTT ranks the top 1000 documents in each shard, it achieves essentially the same score as the oracle model for nDCG@20 on both query sets, demonstrating the effectiveness of hierarchical sharding. We hypothesize that the strong performance on nDCG@20 may be the result of two factors that make it different from the ClueWeb experiments: (1) there are fewer shards; and (2) NeuCLIR data is more recent and was crawled closer to its creation date, so the dates associated with articles are likely more reliable than for ClueWeb. On the NeuCLIR datasets, Recall at 100 is most negatively affected by sharding relative to the oracle performance. This is not seen in Recall at 1000 , so there may be differences arising from variations in the lossy vector compression Finally, PLAID SHIRTTT statistically outperforms the PSQ baseline as measured by nDCG@20 (two-tailed paired t-test with correction).

我们的第三个问题是关于多语言信息检索（MLIR）的。表4比较了PLAID SHIRTTT与其他两种方法在NeuCLIR 2022和2023数据集上的表现：采用分数融合的PSQ+HMM [11, 26 - 28]，以及一个理想分片模型（Oracle Shard Model）。PSQ+HMM是一种快速的非神经基线方法，它依赖于统计机器翻译中的翻译表。选择这种方法是因为它与单语言的BM25类似，完全不涉及神经网络。由于PSQ+HMM是一种跨语言信息检索（CLIR）算法，多语言信息检索（MLIR）通过跨语言的分数融合来实现，无需归一化。当PLAID SHIRTTT对每个分片中的前1000个文档进行排名时，在两个查询集上，它在nDCG@20指标上的得分与理想模型基本相同，这证明了分层分片的有效性。我们假设，nDCG@20指标上的出色表现可能是由两个与ClueWeb实验不同的因素导致的：（1）分片数量较少；（2）NeuCLIR数据更新，且是在数据创建日期附近抓取的，因此与文章关联的日期可能比ClueWeb数据更可靠。在NeuCLIR数据集上，相对于理想模型的性能，分片对100召回率（Recall at 100）的负面影响最大。而在1000召回率（Recall at 1000）中并未出现这种情况，因此可能是有损向量压缩的差异导致了这种不同。最后，通过nDCG@20指标（经过校正的双尾配对t检验）衡量，PLAID SHIRTTT在统计上优于PSQ基线方法。

## 6 CONCLUSION

## 6 结论

PLAID SHIRTTT provides a way to support multi-vector dense retrieval over a streaming collection of documents at large scale. Its search speed supports interactive search. While the hierarchical sharding approach requires a document to be indexed more than once, it allows for good performance on retrieval regardless of how long ago the document arrived. While there is no doubt that utilizing a bi-encoder is more computationally expensive, the multilingual pretrained language model has additional benefits over sparse retrieval approaches. In a multilingual stream, a document language might be unknown, requiring the system to run Language Identification prior to PSQ+HMM; this adds an additional source of error and performance degradation, which is eliminated for PLAID. While PLAID SHIRTTT may not be worth the computational expense for monolingual retrieval, for multilingual retrieval PLAID, and thus PLAID SHIRTTT, has clear advantage.

PLAID SHIRTTT提供了一种在大规模流式文档集合上支持多向量密集检索的方法。其搜索速度能够支持交互式搜索。虽然分层分片方法要求文档被多次索引，但无论文档何时到达，它都能在检索方面实现良好的性能。虽然毫无疑问，使用双编码器的计算成本更高，但多语言预训练语言模型相对于稀疏检索方法有额外的优势。在多语言流中，文档的语言可能未知，这就要求系统在使用PSQ+HMM之前先进行语言识别；这会增加额外的错误来源并导致性能下降，而PLAID消除了这一问题。虽然对于单语言检索来说，PLAID SHIRTTT的计算成本可能不值得，但对于多语言检索，PLAID以及PLAID SHIRTTT具有明显的优势。

## REFERENCES

## 参考文献

[1] Eric W. Brown, James P. Callan, and W. Bruce Croft. 1994. Fast Incremental Indexing for Full-Text Information Retrieval. In Proceedings of 20th International Conference on Very Large Data Bases. 192-202.

[1] 埃里克·W·布朗（Eric W. Brown）、詹姆斯·P·卡兰（James P. Callan）和W·布鲁斯·克罗夫特（W. Bruce Croft）。1994年。全文信息检索的快速增量索引。见《第20届大型数据库国际会议论文集》。192 - 202页。

[2] Chris Buckley, James Allan, and Gerard Salton. 1995. Automatic Routing and Retrieval using SMART: TREC-2. Information Processing & Management 31, 3 (1995), 315-326.

[2] 克里斯·巴克利（Chris Buckley）、詹姆斯·艾伦（James Allan）和杰拉德·萨尔顿（Gerard Salton）。1995年。使用SMART进行自动路由和检索：TREC - 2。《信息处理与管理》31卷，第3期（1995年），315 - 326页。

[3] Jamie Callan, Mark Hoy, Changkuk Yoo, and Le Zhao. 2009. ClueWeb09 data set. https://lemurproject.org/clueweb09.php/.

[3] 杰米·卡兰（Jamie Callan）、马克·霍伊（Mark Hoy）、张库克·柳（Changkuk Yoo）和赵乐（Le Zhao）。2009年。ClueWeb09数据集。https://lemurproject.org/clueweb09.php/。

[4] Charlie Clarke, Nick Craswell, and Ian Soboroff. 2009. Overview of the TREC 2009 Web Track. In Proceedings of the Eighteenth Text REtrieval Conference (TREC 2009).

[4] 查理·克拉克（Charlie Clarke）、尼克·克拉斯韦尔（Nick Craswell）和伊恩·索博罗夫（Ian Soboroff）。2009年。TREC 2009网络赛道概述。见《第18届文本检索会议论文集（TREC 2009）》。

[5] Charlie Clarke, Nick Craswell, Ian Soboroff, and Gordan Cormack. 2010. Overview of the TREC 2010 Web Track. In Proceedings of the Nineteenth Text REtrieval Conference (TREC 2010).

[5] 查理·克拉克（Charlie Clarke）、尼克·克拉斯韦尔（Nick Craswell）、伊恩·索博罗夫（Ian Soboroff）和戈尔丹·科马克（Gordan Cormack）。2010年。TREC 2010网络赛道概述。见《第19届文本检索会议论文集（TREC 2010）》。

[6] Charlie Clarke, Nick Craswell, Ian Soboroff, and Ellen Voorhees. 2011. Overview of the TREC 2011 Web Track. In Proceedings of the Twentieth Text REtrieval Conference (TREC 2011).

[6] 查理·克拉克（Charlie Clarke）、尼克·克拉斯韦尔（Nick Craswell）、伊恩·索博罗夫（Ian Soboroff）和艾伦·沃里斯（Ellen Voorhees）。2011年。TREC 2011网络赛道概述。见《第20届文本检索会议论文集（TREC 2011）》。

[7] Charlie Clarke, Nick Craswell, and Ellen Voorhees. 2012. Overview of the TREC 2012 Web Track. In Proceedings of the Twenty-first Text REtrieval Conference (TREC 2012).

[7] 查理·克拉克（Charlie Clarke）、尼克·克拉斯韦尔（Nick Craswell）和艾伦·沃里斯（Ellen Voorhees）。2012年。TREC 2012网络赛道概述。见《第21届文本检索会议论文集（TREC 2012）》。

[8] Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. Unsupervised Cross-lingual Representation Learning at Scale. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. Online, 8440-8451. https://aclanthology.org/2020.acl-main.747

[8] 亚历克西斯·科诺（Alexis Conneau）、卡蒂凯·坎德尔瓦尔（Kartikay Khandelwal）、纳曼·戈亚尔（Naman Goyal）、维什拉夫·乔杜里（Vishrav Chaudhary）、纪尧姆·温泽克（Guillaume Wenzek）、弗朗西斯科·古兹曼（Francisco Guzmán）、爱德华·格雷夫（Edouard Grave）、迈尔·奥特（Myle Ott）、卢克·泽特尔莫耶（Luke Zettlemoyer）和韦塞林·斯托亚诺夫（Veselin Stoyanov）。2020年。大规模无监督跨语言表征学习。见《第58届计算语言学协会年会论文集》。线上会议，8440 - 8451页。https://aclanthology.org/2020.acl - main.747

[9] Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, Jimmy Lin, Ellen Voorhees, and Ian Soboroff. 2022. The TREC 2022 Deep Learning Track. In Proceedings of the Thirty-First Text Retrieval Conference. NIST.

[9] 尼克·克拉斯韦尔（Nick Craswell）、巴斯卡尔·米特拉（Bhaskar Mitra）、埃米内·伊尔马兹（Emine Yilmaz）、丹尼尔·坎波斯（Daniel Campos）、吉米·林（Jimmy Lin）、艾伦·沃里斯（Ellen Voorhees）和伊恩·索博罗夫（Ian Soboroff）。2022年。2022年文本检索会议（TREC）深度学习赛道。见第三十一届文本检索会议论文集。美国国家标准与技术研究院（NIST）。

[10] Zhuyun Dai and Jamie Callan. 2019. Deeper Text Understanding for IR with Contextual Neural Language Modeling. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval. ${985} - {988}$

[10] 戴竹云（Zhuyun Dai）和杰米·卡伦（Jamie Callan）。2019年。通过上下文神经语言建模实现信息检索（IR）中更深入的文本理解。见第42届国际计算机协会信息检索研究与发展会议论文集。${985} - {988}$

[11] Kareem Darwish and Douglas W Oard. 2003. Probabilistic Structured Query Methods. In Proceedings of the 26th Annual International ACM SIGIR Conference

[11] 卡里姆·达维什（Kareem Darwish）和道格拉斯·W·奥尔德（Douglas W Oard）。2003年。概率结构化查询方法。见第26届国际计算机协会信息检索研究与发展年度会议论文集

on Research and Development in Information Retrieval. 338-344.

信息检索研究与发展。338 - 344页。

[12] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). Minneapolis, Minnesota, 4171-4186. https://aclanthology.org/N19- 1423

[12] 雅各布·德夫林（Jacob Devlin）、张明伟（Ming - Wei Chang）、肯顿·李（Kenton Lee）和克里斯蒂娜·图托纳娃（Kristina Toutanova）。2019年。BERT：用于语言理解的深度双向变换器预训练。见2019年北美计算语言学协会人类语言技术会议论文集，第1卷（长论文和短论文）。明尼苏达州明尼阿波利斯，4171 - 4186页。https://aclanthology.org/N19 - 1423

[13] John R Frank, Max Kleiman-Weiner, Daniel A Roberts, Ellen Voorhees, and Ian Soboroff. 2014. Evaluating Stream Filtering for Entity Profile Updates in TREC 2012, 2013, and 2014. In Proceedings of the Twenty-Third Text Retrieval Conference (TREC).

[13] 约翰·R·弗兰克（John R Frank）、马克斯·克莱曼 - 韦纳（Max Kleiman - Weiner）、丹尼尔·A·罗伯茨（Daniel A Roberts）、艾伦·沃里斯（Ellen Voorhees）和伊恩·索博罗夫（Ian Soboroff）。2014年。评估2012年、2013年和2014年文本检索会议（TREC）中实体概要更新的流过滤。见第二十三届文本检索会议（TREC）论文集。

[14] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bo-janowski, Armand Joulin, and Edouard Grave. 2022. Unsupervised Dense Information Retrieval with Contrastive Learning. Transactions on Machine Learning Research (2022). https://openreview.net/forum?id=jKN1pXi7b0

[14] 高蒂尔·伊扎卡尔（Gautier Izacard）、玛蒂尔德·卡龙（Mathilde Caron）、卢卡斯·侯赛尼（Lucas Hosseini）、塞巴斯蒂安·里德尔（Sebastian Riedel）、彼得·博亚诺夫斯基（Piotr Bojanowski）、阿尔芒·朱林（Armand Joulin）和爱德华·格雷夫（Edouard Grave）。2022年。基于对比学习的无监督密集信息检索。机器学习研究汇刊（2022年）。https://openreview.net/forum?id=jKN1pXi7b0

[15] Omar Khattab and Matei Zaharia. 2020. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval. 39-48.

[15] 奥马尔·哈塔卜（Omar Khattab）和马特·扎哈里亚（Matei Zaharia）。2020年。ColBERT：通过基于BERT的上下文后期交互实现高效的段落搜索。见第43届国际计算机协会信息检索研究与发展会议论文集。39 - 48页。

[16] Dawn Lawrie, Sean MacAvaney, James Mayfield, Paul McNamee, Douglas W Oard, Luca Soldaini, and Eugene Yang. 2022. Overview of the TREC 2022 NeuCLIR Track. In Proceedings of the Thirty-First Text Retrieval Conference. NIST.

[16] 道恩·劳里（Dawn Lawrie）、肖恩·麦卡瓦尼（Sean MacAvaney）、詹姆斯·梅菲尔德（James Mayfield）、保罗·麦克纳米（Paul McNamee）、道格拉斯·W·奥尔德（Douglas W Oard）、卢卡·索尔达尼（Luca Soldaini）和尤金·杨（Eugene Yang）。2022年。2022年文本检索会议（TREC）神经跨语言信息检索（NeuCLIR）赛道概述。见第三十一届文本检索会议论文集。美国国家标准与技术研究院（NIST）。

[17] Dawn Lawrie, Sean MacAvaney, James Mayfield, Paul McNamee, Douglas W. Oard, Luca Soldanini, and Eugene Yang. 2022. Overview of the TREC 2022 NeuCLIR Track. In Proceedings of the Thirty-first Text REtrieval Conference (TREC 2022).

[17] 道恩·劳里（Dawn Lawrie）、肖恩·麦卡瓦尼（Sean MacAvaney）、詹姆斯·梅菲尔德（James Mayfield）、保罗·麦克纳米（Paul McNamee）、道格拉斯·W·奥尔德（Douglas W. Oard）、卢卡·索尔达尼（Luca Soldanini）和尤金·杨（Eugene Yang）。2022年。2022年文本检索会议（TREC）神经跨语言信息检索（NeuCLIR）赛道概述。见第三十一届文本检索会议（TREC 2022）论文集。

[18] Dawn Lawrie, Sean MacAvaney, James Mayfield, Paul McNamee, Douglas W. Oard, Luca Soldanini, and Eugene Yang. 2023. Overview of the TREC 2023 NeuCLIR Track. In Proceedings of the Thirty-second Text REtrieval Conference (TREC 2023).

[18] 道恩·劳里（Dawn Lawrie）、肖恩·麦卡瓦尼（Sean MacAvaney）、詹姆斯·梅菲尔德（James Mayfield）、保罗·麦克纳米（Paul McNamee）、道格拉斯·W·奥尔德（Douglas W. Oard）、卢卡·索尔达尼（Luca Soldanini）和尤金·杨（Eugene Yang）。2023年。2023年文本检索会议（TREC）神经跨语言信息检索（NeuCLIR）赛道概述。见第三十二届文本检索会议（TREC 2023）论文集。

[19] Dawn Lawrie, Eugene Yang, Douglas W Oard, and James Mayfield. 2023. Neural Approaches to Multilingual Information Retrieval. In European Conference on Information Retrieval. Springer, 521-536.

[19] 道恩·劳里（Dawn Lawrie）、尤金·杨（Eugene Yang）、道格拉斯·W·奥尔德（Douglas W Oard）和詹姆斯·梅菲尔德（James Mayfield）。2023年。多语言信息检索的神经方法。见欧洲信息检索会议论文集。施普林格出版社，521 - 536页。

[20] Jimmy Lin, Rodrigo Nogueira, and Andrew Yates. 2022. Pretrained Transformers for Text Ranking: BERT and Beyond. Springer Nature.

[20] 吉米·林（Jimmy Lin）、罗德里戈·诺盖拉（Rodrigo Nogueira）和安德鲁·耶茨（Andrew Yates）。2022年。用于文本排序的预训练变换器：BERT及其他。施普林格自然出版社。

[21] C. Manning, P. Raghavan, and H. Schutze. 2008. Introduction to Information Retrieval. Cambridge University Press.

[21] C. 曼宁（C. Manning）、P. 拉加万（P. Raghavan）和H. 舒尔茨（H. Schutze）。2008年。信息检索导论。剑桥大学出版社。

[22] Stephen E Robertson and Ian Soboroff. 2002. The TREC 2002 Filtering Track Report.. In Proceedings of the Eleventh Text Retrieval Conference.

[22] 斯蒂芬·E·罗伯逊（Stephen E Robertson）和伊恩·索博罗夫（Ian Soboroff）。2002年。2002年文本检索会议（TREC）过滤赛道报告。见第十一届文本检索会议论文集。

[23] Tetsuya Sakai. 2013. The Unreusability of Diversified Search Test Collections.. In The Fifth International Workshop on Evaluating Information Access.

[23] 酒井哲也（Tetsuya Sakai）。2013年。多样化搜索测试集的不可复用性。见第五届信息访问评估国际研讨会论文集。

[24] Keshav Santhanam, Omar Khattab, Christopher Potts, and Matei Zaharia. 2022. PLAID: An Efficient Engine for Late Interaction Retrieval. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management. 1747-1756.

[24] 凯沙夫·桑塔纳姆（Keshav Santhanam）、奥马尔·哈塔布（Omar Khattab）、克里斯托弗·波茨（Christopher Potts）和马泰·扎哈里亚（Matei Zaharia）。2022年。PLAID：一种用于后期交互检索的高效引擎。见第31届ACM国际信息与知识管理会议论文集。第1747 - 1756页。

[25] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. 2022. ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. 3715-3734. https://aclanthology.org/2022.naacl-main.272

[25] 凯沙夫·桑塔纳姆（Keshav Santhanam）、奥马尔·哈塔布（Omar Khattab）、乔恩·萨德 - 法尔孔（Jon Saad - Falcon）、克里斯托弗·波茨（Christopher Potts）和马泰·扎哈里亚（Matei Zaharia）。2022年。ColBERTv2：通过轻量级后期交互实现高效有效的检索。见2022年北美计算语言学协会人类语言技术会议论文集。第3715 - 3734页。https://aclanthology.org/2022.naacl - main.272

[26] Jianqiang Wang and Douglas W. Oard. 2012. Matching Meaning for Cross-Language Information Retrieval. Information Processing & Management 48,4 (2012), 631-653. https://doi.org/10.1016/j.ipm.2011.09.003

[26] 王健强（Jianqiang Wang）和道格拉斯·W·奥尔德（Douglas W. Oard）。2012年。跨语言信息检索的语义匹配。《信息处理与管理》（Information Processing & Management）第48卷，第4期（2012年），第631 - 653页。https://doi.org/10.1016/j.ipm.2011.09.003

[27] Jinxi Xu and Ralph Weischedel. 2000. Cross-Lingual Information Retrieval using Hidden Markov Models. In 2000 Joint SIGDAT Conference on Empirical Methods in Natural Language Processing and Very Large Corpora. 95-103.

[27] 徐锦熙（Jinxi Xu）和拉尔夫·韦舍德尔（Ralph Weischedel）。2000年。使用隐马尔可夫模型的跨语言信息检索。见2000年SIGDAT自然语言处理经验方法与超大语料库联合会议论文集。第95 - 103页。

[28] Eugene Yang, Suraj Nair, Dawn Lawrie, James Mayfield, Douglas W Oard, and Kevin Duh. 2024. Efficiency-Effectiveness Tradeoff of Probabilistic Structured Queries for Cross-Language Information Retrieval. arXiv preprint arXiv:2404.18797 (2024). https://arxiv.org/abs/2404.18797

[28] 尤金·杨（Eugene Yang）、苏拉杰·奈尔（Suraj Nair）、道恩·劳里（Dawn Lawrie）、詹姆斯·梅菲尔德（James Mayfield）、道格拉斯·W·奥尔德（Douglas W Oard）和凯文·杜（Kevin Duh）。2024年。跨语言信息检索中概率结构化查询的效率 - 效果权衡。预印本arXiv:2404.18797（2024年）。https://arxiv.org/abs/2404.18797