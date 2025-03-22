# SLIM: Sparsified Late Interaction for Multi-Vector Retrieval with Inverted Indexes

# SLIM：用于基于倒排索引的多向量检索的稀疏化后期交互

Minghan Li

李明翰

University of Waterloo

滑铁卢大学

Waterloo, Canada

加拿大滑铁卢

m692li@uwaterloo.ca

Sheng-Chieh Lin

林圣杰

University of Waterloo

滑铁卢大学

Waterloo, Canada

加拿大滑铁卢

s269lin@uwaterloo.ca

Xueguang Ma

马学光

University of Waterloo

滑铁卢大学

Waterloo, Canada

加拿大滑铁卢

x93ma@uwaterloo.ca

Jimmy Lin

吉米·林

University of Waterloo

滑铁卢大学

Waterloo, Canada

加拿大滑铁卢

jimmylin@uwaterloo.ca

## ABSTRACT

## 摘要

This paper introduces Sparsified Late Interaction for Multi-vector (SLIM) retrieval with inverted indexes. Multi-vector retrieval methods have demonstrated their effectiveness on various retrieval datasets, and among them, ColBERT is the most established method based on the late interaction of contextualized token embeddings of pre-trained language models. However, efficient ColBERT implementations require complex engineering and cannot take advantage of off-the-shelf search libraries, impeding their practical use. To address this issue, SLIM first maps each contextualized token vector to a sparse, high-dimensional lexical space before performing late interaction between these sparse token embeddings. We then introduce an efficient two-stage retrieval architecture that includes inverted index retrieval followed by a score refinement module to approximate the sparsified late interaction, which is fully compatible with off-the-shelf lexical search libraries such as Lucene. SLIM achieves competitive accuracy on MS MARCO Passages and BEIR compared to ColBERT while being much smaller and faster on CPUs. To our knowledge, we are the first to explore using sparse token representations for multi-vector retrieval. Source code and data are integrated into the Pyserini IR toolkit.

本文介绍了一种用于基于倒排索引的多向量检索的稀疏化后期交互（Sparsified Late Interaction for Multi - vector，SLIM）方法。多向量检索方法已在各种检索数据集上证明了其有效性，其中，ColBERT是基于预训练语言模型的上下文词元嵌入后期交互的最成熟方法。然而，高效的ColBERT实现需要复杂的工程，并且无法利用现成的搜索库，这阻碍了其实际应用。为了解决这个问题，SLIM首先将每个上下文词元向量映射到一个稀疏的高维词法空间，然后再对这些稀疏词元嵌入进行后期交互。然后，我们引入了一种高效的两阶段检索架构，该架构包括倒排索引检索，随后是一个得分细化模块，以近似稀疏化的后期交互，该架构与Lucene等现成的词法搜索库完全兼容。与ColBERT相比，SLIM在MS MARCO段落数据集和BEIR数据集上实现了具有竞争力的准确率，同时在CPU上的规模更小、速度更快。据我们所知，我们是首个探索将稀疏词元表示用于多向量检索的团队。源代码和数据已集成到Pyserini信息检索工具包中。

## CCS CONCEPTS

## 计算机协会概念分类体系

- Information systems $\rightarrow$ Retrieval models and ranking.

- 信息系统 $\rightarrow$ 检索模型与排序。

## KEYWORDS

## 关键词

Neural IR, Late Interaction, Inverted Indexes, Sparse Retrieval

神经信息检索（Neural IR）、后期交互（Late Interaction）、倒排索引（Inverted Indexes）、稀疏检索（Sparse Retrieval）

## ACM Reference Format:

## ACM引用格式：

Minghan Li, Sheng-Chieh Lin, Xueguang Ma, and Jimmy Lin. 2023. SLIM: Sparsified Late Interaction for Multi-Vector Retrieval with Inverted Indexes. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '23), July 23-27, 2023, Taipei, Taiwan. ACM, New York, NY, USA, 6 pages. https://doi.org/10.1145/3539618.3591977

李明翰、林圣杰、马学光和林吉米。2023年。SLIM：基于倒排索引的多向量检索的稀疏化后期交互。见《第46届ACM信息检索研究与发展国际会议论文集（SIGIR '23）》，2023年7月23 - 27日，中国台湾台北。美国纽约州纽约市ACM，6页。https://doi.org/10.1145/3539618.3591977

## 1 INTRODUCTION

## 1 引言

Pairwise token interaction $\left\lbrack  {{12},{19},{45}}\right\rbrack$ has been widely used in information retrieval tasks [31]. Interaction-based methods enable deep coupling between queries and documents, which often outperform representation-based methods $\left\lbrack  {2,{34}}\right\rbrack$ when strong text encoders are absent. However, with the rise of pre-trained transformer models [4, 27], representation-based methods such as DPR [17] and SPLADE [6] gain more popularity as the pre-trained representations capture rich semantics of the input texts. Moreover, these methods can leverage established search libraries such as FAISS [16] and Py-serini [22] for efficient retrieval. To combine the best of both worlds, models such as ColBERT [18] and COIL [9] that leverage late interaction are proposed, where the token interaction between queries and documents only happens at the last layer of contextualized embeddings. Their effectiveness and robustness are demonstrated on various retrieval and question-answering benchmarks. However, these models require token-level retrieval and aggregation, which results in large indexes and high retrieval latency. Therefore, different optimization schemes are proposed to improve efficiency in both time and space $\left\lbrack  {{13},{37}}\right\rbrack$ ,making multi-vector retrieval difficult to fit into off-the-shelf search libraries.

成对词元交互 $\left\lbrack  {{12},{19},{45}}\right\rbrack$ 已广泛应用于信息检索任务 [31]。基于交互的方法能够实现查询与文档之间的深度耦合，在缺乏强大文本编码器的情况下，其性能通常优于基于表示的方法 $\left\lbrack  {2,{34}}\right\rbrack$。然而，随着预训练Transformer模型的兴起 [4, 27]，像DPR [17] 和SPLADE [6] 这样基于表示的方法变得更受欢迎，因为预训练表示能够捕捉输入文本的丰富语义。此外，这些方法可以利用现有的搜索库，如FAISS [16] 和Py - serini [22] 进行高效检索。为了融合两者的优势，提出了像ColBERT [18] 和COIL [9] 这样利用后期交互的模型，其中查询和文档之间的词元交互仅发生在上下文嵌入的最后一层。它们的有效性和鲁棒性在各种检索和问答基准测试中得到了证明。然而，这些模型需要词元级别的检索和聚合，这导致索引庞大且检索延迟高。因此，提出了不同的优化方案来提高时间和空间效率 $\left\lbrack  {{13},{37}}\right\rbrack$，使得多向量检索难以适配现有的搜索库。

In this paper, we propose an efficient and compact approach called Sparsified Late Interaction for Multi-vector retrieval with inverted indexes, or SLIM. SLIM is as effective as the state-of-the-art multi-vector model ColBERT on MS MARCO Passages [32] and BEIR [40] without any distillation or hard negative mining while being more efficient. More importantly, SLIM is fully compatible with inverted indexes in existing search toolkits such as Pyserini, with only a few extra lines of code in the pre-processing and postprocessing steps. In contrast, methods such as ColBERT and COIL require custom implementations and optimizations [13, 37] in order to be practical to use. This presents a disadvantage because some researchers and practitioners may prefer neural retrievers that are compatible with existing infrastructure based on inverted indexes so that the models can be easily deployed in production.

在本文中，我们提出了一种高效且紧凑的方法，称为基于倒排索引的多向量检索的稀疏化后期交互（Sparsified Late Interaction for Multi - vector retrieval with inverted indexes），简称SLIM。在MS MARCO段落数据集 [32] 和BEIR数据集 [40] 上，SLIM在不进行任何蒸馏或难负样本挖掘的情况下，与最先进的多向量模型ColBERT效果相当，同时效率更高。更重要的是，SLIM与现有搜索工具包（如Pyserini）中的倒排索引完全兼容，仅需在预处理和后处理步骤中添加几行额外的代码。相比之下，像ColBERT和COIL这样的方法需要自定义实现和优化 [13, 37] 才能实际使用。这是一个劣势，因为一些研究人员和从业者可能更喜欢与基于倒排索引的现有基础设施兼容的神经检索器，以便模型能够轻松部署到生产环境中。

In order to leverage standard inverted index search, SLIM projects each contextualized token embedding to a high-dimensional, sparse lexical space [6] before performing the late interaction operation. Ideally, the trained sparse representations of documents can be compressed and indexed into an inverted file system for search. Our method is the first to explore using sparse representations for multi-vector retrieval with inverted indexes to our knowledge. However, unlike previous supervised sparse retrieval methods such as uniCOIL [21] or SPLADE [6], which compute sequence-level representations for each query/document, deploying token-level sparse vectors into inverted indexes is problematic. As shown in Figure 1a, the high-level idea is to convert each sparse token embedding into a sub-query/document and perform token-level inverted list retrieval before aggregation (e.g., scatter sum and max). However, this is incredibly slow in practice as latency increases rapidly when the posting lists are too long.

为了利用标准的倒排索引搜索，SLIM在执行后期交互操作之前，将每个上下文词元嵌入投影到一个高维、稀疏的词汇空间 [6]。理想情况下，训练好的文档稀疏表示可以被压缩并索引到倒排文件系统中进行搜索。据我们所知，我们的方法是首个探索使用稀疏表示进行基于倒排索引的多向量检索的方法。然而，与之前的有监督稀疏检索方法（如uniCOIL [21] 或SPLADE [6]，它们为每个查询/文档计算序列级表示）不同，将词元级稀疏向量部署到倒排索引中存在问题。如图1a所示，其核心思想是将每个稀疏词元嵌入转换为一个子查询/文档，并在聚合（例如，分散求和和取最大值）之前进行词元级倒排列表检索。然而，在实践中这非常慢，因为当倒排列表太长时，延迟会迅速增加。

---

<!-- Footnote -->

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.SIGIR '23, July 23-27, 2023, Taipei, Taiwan © 2023 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 978-1-4503-9408-6/23/07...\$15.00 https://doi.org/10.1145/3539618.3591977

允许个人或课堂使用本作品的全部或部分内容制作数字或硬拷贝，无需付费，但前提是这些拷贝不得用于盈利或商业目的，并且拷贝必须带有此声明和首页的完整引用。必须尊重本作品中除作者之外其他人拥有的版权组件。允许进行带引用的摘要。否则，如需复制、重新发布、发布到服务器或重新分发给列表，需要事先获得特定许可和/或支付费用。请从permissions@acm.org请求许可。SIGIR '23，2023年7月23 - 27日，中国台湾台北 © 2023版权归所有者/作者所有。出版权授权给ACM。ACM ISBN 978 - 1 - 4503 - 9408 - 6/23/07... 15.00美元 https://doi.org/10.1145/3539618.3591977

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: Q: Volume of Earth? Q: Volume of Earth? Lower Bound BERT $\{$ volume:2.6,size:1.8,of:1.8,earth:2.9,world: ${0.5}\}$ Upper Bound Query Fusion Max Pooling BERT Index D: Earth's mass is 6.6 sextillion ton (5.9722x10^24 kilograms). Its volume is Score Refinement 1.08321 x 10^12 km^3. The total surface area.. (b) Approximate SLIM implementation. volume \{volume: 2.6, size: 1.7, earth: 0.4\} BERT \{of: 1.5, size: 0.1\} earth \{earth:2.5, of: 0.3, world: 0.5\} BERT Index D: Earth's mass is 6.6 sextillion ton (5.9722x10^24 Scatter Max kilograms). Its volume is 1.08321 x 10^12 km^3. The Scatter Sum total surface area... (a) Naive SLIM implementation. -->

<img src="https://cdn.noedgeai.com/0195a55c-5239-7d2b-930d-9ff91122ace4_1.jpg?x=208&y=238&w=1385&h=482&r=0"/>

Figure 1: (a) The naive SLIM implementation takes the expansion of each token as sub-queries/documents. The token-level rankings are merged using scatter operations. (b) The approximate SLIM implementation first fuses the lower- and upper-bounds of scores for retrieval using Equations (8) and (9); the candidate list is then refined according to Equation (4).

图1：(a) 朴素的SLIM实现将每个词元的扩展作为子查询/文档。词元级别的排名使用分散操作进行合并。(b) 近似的SLIM实现首先使用公式(8)和(9)融合用于检索的分数上下界；然后根据公式(4)对候选列表进行细化。

<!-- Media -->

Therefore, instead of using the naive approach, a more economical way is to approximate the sparsified late interaction with a two-stage system. We first calculate the upper and lower bounds of the sparsified late-interaction scores by swapping the order of the operators in late interaction (see Section 2). In this way, the token-level sparse vectors are pooled before indexing to obtain sequence-level representations for the inverted index, yielding fast retrieval speed. After retrieval, a lightweight score refinement step is applied, where we use the sparse token vectors stored using Scipy [41] to rerank the candidate lists. As shown in Figure 1b, this two-stage design allows us to optimize the latency of the retrieval stage without worrying about accuracy loss as the score refinement will compensate for errors from the first stage. Experiments on MS MARCO Passages show that SLIM is able to achieve a similar ranking accuracy compared to ColBERT-v2, while using 40% less storage and achieving an ${83}\%$ decrease in latency. To sum up,our contributions in this paper are three-fold:

因此，与使用朴素方法不同，一种更经济的方式是用一个两阶段系统来近似稀疏化的后期交互。我们首先通过交换后期交互中运算符的顺序来计算稀疏化后期交互分数的上下界（见第2节）。通过这种方式，在索引之前对词元级别的稀疏向量进行合并，以获得用于倒排索引的序列级表示，从而实现快速检索速度。检索之后，应用一个轻量级的分数细化步骤，我们使用Scipy [41]存储的稀疏词元向量对候选列表进行重新排序。如图1b所示，这种两阶段设计使我们能够优化检索阶段的延迟，而不必担心精度损失，因为分数细化将弥补第一阶段的误差。在MS MARCO段落上的实验表明，与ColBERT - v2相比，SLIM能够实现相似的排名精度，同时减少40%的存储空间，并使延迟降低${83}\%$。综上所述，本文的贡献主要有三点：

- We are the first to use sparse representations for multi-vector retrieval with standard inverted indexes.

- 我们首次将稀疏表示用于使用标准倒排索引的多向量检索。

- We provide a two-stage implementation of SLIM by approximating late interaction followed by a score refinement step.

- 我们通过近似后期交互并随后进行分数细化步骤，提供了SLIM的两阶段实现。

- SLIM is fully compatible with off-the-shelf search toolkits such as Pyserini.

- SLIM与Pyserini等现成的搜索工具包完全兼容。

## 2 METHODOLOGY

## 2 方法

ColBERT [18] proposes late interaction between the tokens in a query $q = \left\{  {{q}_{1},{q}_{2},\cdots ,{q}_{N}}\right\}$ and a document $d = \left\{  {{d}_{1},{d}_{2},\cdots ,{d}_{M}}\right\}$ :

ColBERT [18]提出了查询$q = \left\{  {{q}_{1},{q}_{2},\cdots ,{q}_{N}}\right\}$和文档$d = \left\{  {{d}_{1},{d}_{2},\cdots ,{d}_{M}}\right\}$中词元之间的后期交互：

$$
s\left( {q,d}\right)  = \mathop{\sum }\limits_{{i = 1}}^{N}\mathop{\max }\limits_{{j = 1}}^{M}{v}_{{q}_{i}}^{T}{v}_{{d}_{j}} \tag{1}
$$

where ${v}_{{q}_{i}}$ and ${v}_{{d}_{j}}$ denote the last-layer contextualized token embed-dings of BERT. This operation exhaustively compares each query token to all document tokens. The latency and storage of ColBERT are bloated as many tokens do not contribute to either query or document semantics, and thus complex engineering optimizations are needed to make the model practical $\left\lbrack  {{37},{38}}\right\rbrack$ .

其中${v}_{{q}_{i}}$和${v}_{{d}_{j}}$表示BERT最后一层的上下文词元嵌入。此操作会将每个查询词元与所有文档词元进行详尽比较。由于许多词元对查询或文档语义没有贡献，ColBERT的延迟和存储开销很大，因此需要复杂的工程优化才能使该模型具有实用性$\left\lbrack  {{37},{38}}\right\rbrack$。

Unlike ColBERT, which only uses contextualized token embed-dings for computing similarity, SPLADE [5, 6] further utilizes the pre-trained Mask Language Modeling (MLM) layer to project each $c$ - dimensional token embedding to a high-dimensional, lexical space $V$ . Each dimension corresponds to a token and has a non-negative value due to the following activation:

与仅使用上下文词元嵌入来计算相似度的ColBERT不同，SPLADE [5, 6]进一步利用预训练的掩码语言模型（MLM）层将每个$c$维的词元嵌入投影到一个高维的词法空间$V$。由于以下激活函数，每个维度对应一个词元且具有非负值：

$$
{\phi }_{{d}_{j}} = \log \left( {1 + \operatorname{ReLU}\left( {{W}^{T}{v}_{{d}_{j}} + b}\right) }\right) , \tag{2}
$$

where ${\phi }_{{d}_{j}} \in  {\mathbb{R}}^{\left| V\right| },{v}_{{d}_{j}}$ is the token embedding of the $j$ th token of document $d;W$ and $b$ are the weights and biases of the MLM layer. To compute the final similarity score, SPLADE pools all the token embeddings into a sequence-level representation and uses the dot product between a query and a document as the similarity:

其中${\phi }_{{d}_{j}} \in  {\mathbb{R}}^{\left| V\right| },{v}_{{d}_{j}}$是文档$d;W$的第$j$个词元的词元嵌入，$b$是MLM层的权重和偏置。为了计算最终的相似度分数，SPLADE将所有词元嵌入合并为一个序列级表示，并使用查询和文档之间的点积作为相似度：

$$
s\left( {q,d}\right)  = {\left( \mathop{\max }\limits_{{i = 1}}^{N}{\phi }_{{q}_{i}}\right) }^{T}\left( {\mathop{\max }\limits_{{j = 1}}^{M}{\phi }_{{d}_{j}}}\right) . \tag{3}
$$

Here, max pooling is an element-wise operation and is feasible for sparse representations as the dimensions of all the token vectors are aligned (lexical space) and non-negative. These two properties play a vital role in the success of SPLADE and are also important for making SLIM efficient, as we shall see later.

这里，最大池化是一种逐元素操作，对于稀疏表示是可行的，因为所有词元向量的维度是对齐的（词法空间）且为非负值。这两个属性在SPLADE的成功中起着至关重要的作用，并且正如我们稍后将看到的，对于提高SLIM的效率也很重要。

Similar to ColBERT's late interaction, sparsified late interaction also takes advantages of the contextualized embeddings of BERT's last layer. But different from ColBERT, we first apply the sparse activation in Equation (2) before calculating the similarity in Equation (1):

与ColBERT的后期交互类似，稀疏化后期交互也利用了BERT最后一层的上下文嵌入。但与ColBERT不同的是，我们在计算公式(1)中的相似度之前，首先应用公式(2)中的稀疏激活：

$$
s\left( {q,d}\right)  = \mathop{\sum }\limits_{{i = 1}}^{N}\mathop{\max }\limits_{{j = 1}}^{M}{\phi }_{{q}_{i}}^{T}{\phi }_{{d}_{j}} \tag{4}
$$

where ${\phi }_{{d}_{j}}$ and ${\phi }_{{q}_{i}}$ are the representations in Equation (2). However, as shown in Figure 1a, to keep the token-level interaction, we must convert the sparse representation of each token in a query to a sub-query and retrieve the sub-documents from the index. Moreover, the token-level rankings need to be merged to yield the final ranked list using scatter operations, which further increases latency.

其中${\phi }_{{d}_{j}}$和${\phi }_{{q}_{i}}$是公式(2)中的表示。然而，如图1a所示，为了保持词元级别的交互，我们必须将查询中每个词元的稀疏表示转换为子查询，并从索引中检索子文档。此外，需要使用分散操作合并词元级别的排名以得到最终的排序列表，这进一步增加了延迟。

Due to the impracticality of the naive implementation, a natural solution would be to approximate Equation (4) during retrieval. We SLIM: Sparsified Late Interaction for Multi-Vector Retrieval with Inverted Indexes first unfold the dot product in Equation (4):

由于朴素实现不切实际，一个自然的解决方案是在检索期间近似公式(4)。我们SLIM：使用倒排索引进行多向量检索的稀疏化后期交互首先展开公式(4)中的点积：

$$
s\left( {q,d}\right)  = \mathop{\sum }\limits_{{i = 1}}^{N}\mathop{\max }\limits_{{j = 1}}^{M}\mathop{\sum }\limits_{{k = 1}}^{\left| V\right| }{\phi }_{{q}_{i}}^{\left( k\right) }{\phi }_{{d}_{j}}^{\left( k\right) }, \tag{5}
$$

<!-- Media -->

<table><tr><td rowspan="2">Models</td><td colspan="2">MARCO Dev</td><td colspan="2">TREC DL19</td><td colspan="2">TREC DL20</td><td rowspan="2">BEIR (13 tasks) nDCG@10</td><td colspan="2">Space&Time Efficiency</td></tr><tr><td>MRR@10</td><td>R@1k</td><td>nDCG@10</td><td>R@1k</td><td>nDCG@10</td><td>R@1k</td><td>Disk (GB)</td><td>Latency (ms/query)</td></tr><tr><td colspan="10">Models trained with only BM25 hard negatives from MS MARCO Passages</td></tr><tr><td>BM25</td><td>0.188</td><td>0.858</td><td>0.506</td><td>0.739</td><td>0.488</td><td>0.733</td><td>0.440</td><td>0.7</td><td>40</td></tr><tr><td>DPR</td><td>0.319</td><td>0.941</td><td>0.611</td><td>0.742</td><td>0.591</td><td>0.796</td><td>0.375</td><td>26.0</td><td>2015</td></tr><tr><td>SPLADE</td><td>0.340</td><td>0.965</td><td>0.683</td><td>0.813</td><td>0.671</td><td>0.823</td><td>0.453</td><td>2.6</td><td>475</td></tr><tr><td>COIL</td><td>0.353</td><td>0.967</td><td>0.704</td><td>0.835</td><td>0.688</td><td>0.841</td><td>0.483</td><td>78.5</td><td>3258</td></tr><tr><td>ColBERT</td><td>0.360</td><td>0.968</td><td>0.694</td><td>0.830</td><td>0.676</td><td>0.837</td><td>0.453</td><td>154.3</td><td>-</td></tr><tr><td>SLIM</td><td>0.358</td><td>0.962</td><td>0.701</td><td>0.824</td><td>0.640</td><td>0.854</td><td>0.451</td><td>18.2</td><td>580</td></tr><tr><td colspan="10">Models trained with further pre-training/hard-negative mining/distillation</td></tr><tr><td>coCondenser</td><td>0.382</td><td>0.984</td><td>0.674</td><td>0.820</td><td>0.684</td><td>0.839</td><td>0.420</td><td>26.0</td><td>2015</td></tr><tr><td>SPLADE-v2</td><td>0.368</td><td>0.979</td><td>0.729</td><td>0.865</td><td>0.718</td><td>0.890</td><td>0.499</td><td>4.1</td><td>2710</td></tr><tr><td>ColBERT-v2</td><td>0.397</td><td>0.985</td><td>0.744</td><td>0.882</td><td>0.750</td><td>0.894</td><td>0.500</td><td>29.0</td><td>3275</td></tr><tr><td>${\mathrm{{SLIM}}}^{+ + }$</td><td>0.404</td><td>0.968</td><td>0.714</td><td>0.842</td><td>0.702</td><td>0.855</td><td>0.490</td><td>17.3</td><td>550</td></tr></table>

<table><tbody><tr><td rowspan="2">模型</td><td colspan="2">MARCO开发集</td><td colspan="2">TREC DL19（文本检索会议2019年深度学习任务）</td><td colspan="2">TREC DL20（文本检索会议2020年深度学习任务）</td><td rowspan="2">BEIR（13项任务）归一化折损累积增益@10（nDCG@10）</td><td colspan="2">时空效率</td></tr><tr><td>前10名平均倒数排名（MRR@10）</td><td>前1000名召回率（R@1k）</td><td>归一化折损累积增益@10（nDCG@10）</td><td>前1000名召回率（R@1k）</td><td>归一化折损累积增益@10（nDCG@10）</td><td>前1000名召回率（R@1k）</td><td>磁盘（GB）</td><td>延迟（毫秒/查询）</td></tr><tr><td colspan="10">仅使用来自MS MARCO段落的BM25硬负样本训练的模型</td></tr><tr><td>二元独立模型（BM25）</td><td>0.188</td><td>0.858</td><td>0.506</td><td>0.739</td><td>0.488</td><td>0.733</td><td>0.440</td><td>0.7</td><td>40</td></tr><tr><td>密集段落检索器（DPR）</td><td>0.319</td><td>0.941</td><td>0.611</td><td>0.742</td><td>0.591</td><td>0.796</td><td>0.375</td><td>26.0</td><td>2015</td></tr><tr><td>稀疏潜在注意力文档编码器（SPLADE）</td><td>0.340</td><td>0.965</td><td>0.683</td><td>0.813</td><td>0.671</td><td>0.823</td><td>0.453</td><td>2.6</td><td>475</td></tr><tr><td>上下文感知倒排索引学习（COIL）</td><td>0.353</td><td>0.967</td><td>0.704</td><td>0.835</td><td>0.688</td><td>0.841</td><td>0.483</td><td>78.5</td><td>3258</td></tr><tr><td>基于上下文的BERT（ColBERT）</td><td>0.360</td><td>0.968</td><td>0.694</td><td>0.830</td><td>0.676</td><td>0.837</td><td>0.453</td><td>154.3</td><td>-</td></tr><tr><td>轻量级信息检索模型（SLIM）</td><td>0.358</td><td>0.962</td><td>0.701</td><td>0.824</td><td>0.640</td><td>0.854</td><td>0.451</td><td>18.2</td><td>580</td></tr><tr><td colspan="10">经过进一步预训练/硬负样本挖掘/蒸馏训练的模型</td></tr><tr><td>协同冷凝器（coCondenser）</td><td>0.382</td><td>0.984</td><td>0.674</td><td>0.820</td><td>0.684</td><td>0.839</td><td>0.420</td><td>26.0</td><td>2015</td></tr><tr><td>SPLADE-v2</td><td>0.368</td><td>0.979</td><td>0.729</td><td>0.865</td><td>0.718</td><td>0.890</td><td>0.499</td><td>4.1</td><td>2710</td></tr><tr><td>ColBERT-v2</td><td>0.397</td><td>0.985</td><td>0.744</td><td>0.882</td><td>0.750</td><td>0.894</td><td>0.500</td><td>29.0</td><td>3275</td></tr><tr><td>${\mathrm{{SLIM}}}^{+ + }$</td><td>0.404</td><td>0.968</td><td>0.714</td><td>0.842</td><td>0.702</td><td>0.855</td><td>0.490</td><td>17.3</td><td>550</td></tr></tbody></table>

Table 1: In-domain and out-of-domain evaluation on MS MARCO Passages, TREC DL 2019/2020, and BEIR. "- means not practical to evaluate on a single CPU. Latency is benchmarked on a single CPU and query encoding time is excluded.

表1：在MS MARCO段落、TREC DL 2019/2020和BEIR上的领域内和领域外评估。“-”表示在单个CPU上进行评估不切实际。延迟是在单个CPU上进行基准测试的，不包括查询编码时间。

<!-- Media -->

where ${\phi }_{{q}_{i}}^{\left( k\right) }$ is the $k$ th elements of ${\phi }_{{q}_{i}}$ . As each ${\phi }_{{q}_{i}}$ across different ${q}_{i}$ all shares the same lexical space and the values are non-negative, we can easily derive its upper-bound and lower-bound:

其中 ${\phi }_{{q}_{i}}^{\left( k\right) }$ 是 ${\phi }_{{q}_{i}}$ 的第 $k$ 个元素。由于不同 ${q}_{i}$ 下的每个 ${\phi }_{{q}_{i}}$ 都共享相同的词法空间，且值为非负，我们可以轻松得出其上限和下限：

$$
\mathop{\sum }\limits_{{i = 1}}^{N}\mathop{\max }\limits_{{j = 1}}^{M}\mathop{\sum }\limits_{{k = 1}}^{\left| V\right| }{e}_{{q}_{i}}^{\left( k\right) }{\phi }_{{d}_{j}}^{\left( k\right) } \leq  s\left( {q,d}\right)  \leq  \mathop{\sum }\limits_{{i = 1}}^{N}\mathop{\sum }\limits_{{k = 1}}^{\left| V\right| }\mathop{\max }\limits_{{j = 1}}^{M}{\phi }_{{q}_{i}}^{\left( k\right) }{\phi }_{{d}_{j}}^{\left( k\right) }, \tag{6}
$$

$$
{e}_{{q}_{i}}^{\left( k\right) } = \left\{  \begin{matrix} {\phi }_{{q}_{i}}^{\left( k\right) }, & \text{ if }k = \mathop{\operatorname{argmax}}\limits_{k}{\phi }_{{q}_{i}}^{\left( k\right) } \\  0, & \text{ otherwise } \end{matrix}\right.  \tag{7}
$$

Then,the lower-bound score ${s}_{l}\left( {q,d}\right)$ and upper-bound score ${s}_{h}\left( {q,d}\right)$ can be further factorized as:

然后，下限得分 ${s}_{l}\left( {q,d}\right)$ 和上限得分 ${s}_{h}\left( {q,d}\right)$ 可以进一步分解为：

$$
{s}_{l}\left( {q,d}\right)  = \mathop{\sum }\limits_{{i = 1}}^{N}\mathop{\max }\limits_{{j = 1}}^{M}\mathop{\sum }\limits_{{k = 1}}^{\left| V\right| }{e}_{{q}_{i}}^{\left( k\right) }{\phi }_{{d}_{j}}^{\left( k\right) } = {\left( \mathop{\sum }\limits_{{i = 1}}^{N}{e}_{{q}_{i}}\right) }^{T}\left( {\mathop{\max }\limits_{{j = 1}}^{M}{\phi }_{{d}_{j}}}\right) ;
$$

$$
{s}_{h}\left( {q,d}\right)  = \mathop{\sum }\limits_{{i = 1}}^{N}\mathop{\sum }\limits_{{k = 1}}^{\left| V\right| }\mathop{\max }\limits_{{j = 1}}^{M}{\phi }_{{q}_{i}}^{\left( k\right) }{\phi }_{{d}_{j}}^{\left( k\right) } = {\left( \mathop{\sum }\limits_{{i = 1}}^{N}{\phi }_{{q}_{i}}\right) }^{T}\left( {\mathop{\max }\limits_{{j = 1}}^{M}{\phi }_{{d}_{j}}}\right)  \tag{8}
$$

where both the sum and max operations will be element-wise if the targets are vectors. We see that these two equations resemble the form of SPLADE in Equation (3), where queries and documents are encoded into sequence-level representations independently. In this way, $\mathop{\max }\limits_{{j = 1}}^{M}{\phi }_{{d}_{j}}$ in Equation (8) can be pre-computed and indexed offline. To approximate $s\left( {q,d}\right)$ in Equation (5),we use the linear interpolation between the lower- and the upper-bound:

如果目标是向量，那么求和和取最大值操作将按元素进行。我们发现这两个方程与方程(3)中SPLADE的形式相似，在方程(3)中，查询和文档被独立编码为序列级表示。通过这种方式，方程(8)中的 $\mathop{\max }\limits_{{j = 1}}^{M}{\phi }_{{d}_{j}}$ 可以离线预计算并建立索引。为了近似方程(5)中的 $s\left( {q,d}\right)$，我们使用下限和上限之间的线性插值：

$$
{s}_{a}\left( {q,d}\right)  = \beta  \cdot  {s}_{l}\left( {q,d}\right)  + \left( {1 - \beta }\right)  \cdot  {s}_{h}\left( {q,d}\right) 
$$

$$
 = {\left( \mathop{\sum }\limits_{{i = 1}}^{N}\left( \beta  \cdot  {e}_{{q}_{i}} + \left( 1 - \beta \right)  \cdot  {\phi }_{{q}_{i}}\right) \right) }^{T}\left( {\mathop{\max }\limits_{{j = 1}}^{M}{\phi }_{{d}_{j}}}\right)  \tag{9}
$$

where ${s}_{a}\left( {q,d}\right)$ is the approximate score of SLIM and $\beta  \in  \left\lbrack  {0,1}\right\rbrack$ is the interpolation coefficient. As shown in Equation (9), we can first fuse the query representation before retrieval and fine-tune the coefficient $\beta$ on a validation set. It is worth mentioning that the approximation is applied after the SLIM model is trained using Equation (4). During indexing, we use Pyserini [22] to index the sequence level-representation $\mathop{\max }\limits_{{j = 1}}^{M}{\phi }_{{d}_{j}}$ and use Scipy [41] to store the sparse token vectors. During retrieval, we use Equation (9) to retrieve the top- $k$ candidates with the LuceneImpactSearcher in Pyserini. For score refinement, we extract the stored sparse token vectors for the top- $k$ documents and use Equation (4) to refine the candidate list. This two-stage breakdown yields a good effectiveness-efficiency trade-off, as we can aggressively reduce the first-stage retrieval latency without too much accuracy loss due to the score refinement.

其中 ${s}_{a}\left( {q,d}\right)$ 是SLIM的近似得分，$\beta  \in  \left\lbrack  {0,1}\right\rbrack$ 是插值系数。如方程(9)所示，我们可以在检索前先融合查询表示，并在验证集上微调系数 $\beta$。值得一提的是，该近似是在使用方程(4)训练SLIM模型之后应用的。在索引阶段，我们使用Pyserini [22] 对序列级表示 $\mathop{\max }\limits_{{j = 1}}^{M}{\phi }_{{d}_{j}}$ 进行索引，并使用Scipy [41] 存储稀疏词元向量。在检索阶段，我们使用方程(9)通过Pyserini中的LuceneImpactSearcher检索前 $k$ 个候选文档。为了细化得分，我们提取前 $k$ 个文档中存储的稀疏词元向量，并使用方程(4)细化候选列表。这种两阶段分解实现了良好的有效性 - 效率权衡，因为由于得分细化，我们可以在不过多损失准确性的情况下大幅降低第一阶段的检索延迟。

To further reduce the memory footprint and improve latency for the first-stage retrieval, we apply two post-hoc pruning strategies on the inverted index to remove: (1) tokens that have term importance below a certain threshold; (2) postings that exceed a certain length (i.e., the inverse document frequency is below a threshold). Since all the term importance values are non-negative, tokens with smaller weights contribute less to the final similarity. Moreover, tokens with long postings mean that they frequently occur in different documents, which yield low inverse document frequencies (IDFs) and therefore can be safely discarded.

为了进一步减少内存占用并提高第一阶段检索的延迟，我们对倒排索引应用了两种事后剪枝策略来移除：(1) 词项重要性低于某个阈值的词元；(2) 长度超过某个阈值的倒排列表（即，逆文档频率低于某个阈值）。由于所有词项重要性值均为非负，权重较小的词元对最终相似度的贡献较小。此外，倒排列表较长的词元意味着它们在不同文档中频繁出现，这会产生较低的逆文档频率（IDF），因此可以安全地丢弃。

## 3 EXPERIMENTS

## 3 实验

We evaluate our model and baselines on MS MARCO Passages [32] and its shared tasks, TREC DL 2019/2020 passage ranking tasks [3]. We train the baseline models on MS MARCO Passages and report results on its dev-small set and TREC DL 2019/2020 test queries following the same setup in CITADEL [20]. We further evaluate the models on the BEIR benchmark [40], which consists of a diverse set of 18 retrieval tasks across 9 domains. Following previous work $\left\lbrack  {5,{38}}\right\rbrack$ ,we only evaluate 13 datasets due to license restrictions. ${}^{1}$ The evaluation metrics are MRR@10,nDCG@10,and Recall@1000 (i.e., R@1K). Latency is benchmarked on a single Intel(R) Xeon(R) Platinum 8275CL CPU @ 3.00GHz and both the batch size and the number of threads are set to 1 .

我们在MS MARCO段落 [32] 及其共享任务、TREC DL 2019/2020段落排序任务 [3] 上评估我们的模型和基线模型。我们在MS MARCO段落上训练基线模型，并按照CITADEL [20] 中的相同设置，在其小开发集和TREC DL 2019/2020测试查询上报告结果。我们进一步在BEIR基准测试 [40] 上评估这些模型，该基准测试包含9个领域的18个不同检索任务。遵循先前的工作 $\left\lbrack  {5,{38}}\right\rbrack$，由于许可证限制，我们仅评估13个数据集。${}^{1}$ 评估指标为MRR@10、nDCG@10和Recall@1000（即，R@1K）。延迟是在单个英特尔(R)至强(R)铂金8275CL CPU @ 3.00GHz上进行基准测试的，批量大小和线程数均设置为1。

---

<!-- Footnote -->

${}^{1}$ CQADupstack,Signal-1M(RT),BioASQ,Robust04,TREC-NEWs are excluded.

${}^{1}$ CQADupstack、Signal - 1M(RT)、BioASQ、Robust04、TREC - NEWS被排除在外。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: 0.965 - w/ score refinement -->

<img src="https://cdn.noedgeai.com/0195a55c-5239-7d2b-930d-9ff91122ace4_3.jpg?x=222&y=269&w=1319&h=337&r=0"/>

Figure 2: Effectiveness-Efficiency trade-offs of SLIM (w/ and w/o score refinement) on MS MARCO Passages.

图2：SLIM（带和不带得分细化）在MS MARCO段落上的有效性 - 效率权衡。

<!-- Media -->

We follow the training procedure of CITADEL [20] to train SLIM and apply ${\ell }_{1}$ regularization on the sparse token representations for sparsity control. SLIM is trained only with BM25 hard negatives in MS MARCO Passages,while SLIM ${}^{+ + }$ is trained with cross-encoder distillation and hard negative mining. For indexing and pruning, the default token weight pruning threshold is 0.5 and the IDF threshold is 3 . We use Pyserini to index and retrieve the documents and use Scipy to store the sparse token vectors (CSR matrix). For retrieval, the fusion weight $\beta$ in Equation (9) is set to 0.01 . We first retrieve the top-4000 candidates using Pyserini's LuceneImpactSearcher and then use the original sparse token vectors stored by Scipy to refine the ranked list and output the top-1000 documents.

我们遵循CITADEL [20]的训练流程来训练SLIM，并对稀疏词元表示应用${\ell }_{1}$正则化以进行稀疏性控制。在MS MARCO段落数据集上，SLIM仅使用BM25硬负样本进行训练，而SLIM ${}^{+ + }$则通过交叉编码器蒸馏和硬负样本挖掘进行训练。对于索引和剪枝，默认的词元权重剪枝阈值为0.5，逆文档频率（IDF）阈值为3。我们使用Pyserini对文档进行索引和检索，并使用Scipy存储稀疏词元向量（压缩稀疏行矩阵，CSR matrix）。在检索时，公式（9）中的融合权重$\beta$设置为0.01。我们首先使用Pyserini的LuceneImpactSearcher检索前4000个候选文档，然后使用Scipy存储的原始稀疏词元向量对排序列表进行细化，并输出前1000个文档。

Table 1 shows the in-domain evaluation results on MS MARCO Passages and TREC DL 2019/2020. SLIM and SLIM ${}^{+ + }$ manage to achieve accuracy comparable to ColBERT and ColBERT-v2 on MS MARCO Passages. However, the results seem unstable on TREC DL, which is expected as TREC DL has far fewer queries. For out-of-domain evaluation, the results are a little bit worse than the current state of the art but still close to ColBERT. For latency and storage, SLIM has a much smaller disk requirement and lower latency compared to ColBERT. For example, ${\mathrm{{SLIM}}}^{+ + }$ achieves effectiveness comparable to ColBERT-v2 with an 83% decrease in latency on a single CPU and using ${40}\%$ less storage.

表1展示了在MS MARCO段落数据集和TREC DL 2019/2020数据集上的领域内评估结果。在MS MARCO段落数据集上，SLIM和SLIM ${}^{+ + }$的准确率与ColBERT和ColBERT - v2相当。然而，在TREC DL数据集上的结果似乎不稳定，这是意料之中的，因为TREC DL的查询数量要少得多。在领域外评估中，结果比当前的最优水平略差，但仍接近ColBERT。在延迟和存储方面，与ColBERT相比，SLIM对磁盘的需求小得多，延迟也更低。例如，${\mathrm{{SLIM}}}^{+ + }$在单CPU上实现了与ColBERT - v2相当的效果，延迟降低了83%，并且使用的存储减少了${40}\%$。

Next, we show that a two-stage implementation of SLIM provides a good trade-off between effectiveness and efficiency. Figure 2 plots CPU latency vs. MRR@10/Recall@1000 on MS MARCO Passages using IDF thresholding for SLIM (w/o hard-negative mining and distillation). We set the minimum IDF threshold to $\{ 0,{0.5},1,{1.5}$ , $2,{2.5},3\}$ and first-stage retrieval candidates top- $k$ to $\{ {1000},{1500}$ , ${2000},{2500},{3000},{3500},{4000}\}$ before score refinement (if any). Without the score refinement, the MRR@10 and Recall@1000 scores are high when the IDF threshold is small, but effectiveness drops rapidly when we increase the IDF threshold to trade accuracy for latency. In contrast, with the score refinement step, we see that with about 0.003 loss in MRR@10 and 0.01 loss in Recall@1000, latency improves drastically from about ${1800}\mathrm{\;{ms}}/$ query to ${500}\mathrm{\;{ms}}/$ query. The reason is that the score refinement step, which takes a small amount of time compared to retrieval, compensates for the errors from the aggressive pruning at the first stage, which shows that our novel lower- and upper-bound fusion provides a good approximation of SLIM when the IDF threshold is low, and that the score refinement step is important as it allows more aggressive pruning for the first-stage retrieval.

接下来，我们表明SLIM的两阶段实现方案在有效性和效率之间提供了良好的平衡。图2绘制了在MS MARCO段落数据集上，使用逆文档频率（IDF）阈值对SLIM（不进行硬负样本挖掘和蒸馏）进行处理时，CPU延迟与MRR@10/召回率@1000的关系。在分数细化（如果有）之前，我们将最小逆文档频率阈值设置为$\{ 0,{0.5},1,{1.5}$、$2,{2.5},3\}$，并将第一阶段检索候选文档的数量从top - $k$设置为$\{ {1000},{1500}$、${2000},{2500},{3000},{3500},{4000}\}$。如果不进行分数细化，当逆文档频率阈值较小时，MRR@10和召回率@1000得分较高，但当我们提高逆文档频率阈值以牺牲准确率来换取低延迟时，有效性会迅速下降。相比之下，进行分数细化步骤后，我们发现MRR@10损失约0.003，召回率@1000损失约0.01，但延迟从每个查询约${1800}\mathrm{\;{ms}}/$大幅提高到每个查询约${500}\mathrm{\;{ms}}/$。原因是，与检索相比，分数细化步骤耗时较少，它弥补了第一阶段激进剪枝带来的误差，这表明当逆文档频率阈值较低时，我们提出的新颖的上下界融合方法能够很好地近似SLIM，并且分数细化步骤很重要，因为它允许在第一阶段检索时进行更激进的剪枝。

## 4 RELATED WORK

## 4 相关工作

Dense retrieval [17] gains much popularity as it is supported by multiple approximate nearest neighbor search libraries $\left\lbrack  {{11},{16}}\right\rbrack$ . To improve effectiveness, hard negative mining [42, 43] and knowledge distillation $\left\lbrack  {{14},{26}}\right\rbrack$ are often deployed. Recently,further pretraining for retrieval $\left\lbrack  {7,8,{10},{15},{28}}\right\rbrack$ is proposed to improve the fine-tuned effectiveness of downstream tasks.

密集检索 [17] 由于得到了多个近似最近邻搜索库 $\left\lbrack  {{11},{16}}\right\rbrack$ 的支持而广受欢迎。为了提高有效性，经常采用硬负样本挖掘 [42, 43] 和知识蒸馏 $\left\lbrack  {{14},{26}}\right\rbrack$ 方法。最近，有人提出了用于检索的进一步预训练 $\left\lbrack  {7,8,{10},{15},{28}}\right\rbrack$ 方法，以提高下游任务的微调效果。

Sparse retrieval systems such as BM25 [35] and tf-idf [36] encode documents into bags of words. Recently, pre-trained language models are used to learn contextualized term importance $\left\lbrack  {1,6,{21},{23},{30}}\right\rbrack$ . These models leverage current search toolkits such as Pyserini [22] to perform sparse retrieval, or contribute to hybrid approaches with dense retrieval $\left\lbrack  {{13},{24},{25},{39}}\right\rbrack$ .

像BM25 [35] 和词频 - 逆文档频率（tf - idf） [36] 这样的稀疏检索系统将文档编码为词袋。最近，预训练语言模型被用于学习上下文相关的词项重要性 $\left\lbrack  {1,6,{21},{23},{30}}\right\rbrack$。这些模型利用Pyserini [22] 等当前的搜索工具包进行稀疏检索，或者与密集检索相结合形成混合检索方法 $\left\lbrack  {{13},{24},{25},{39}}\right\rbrack$。

Besides ColBERT [18], COIL [9] accelerates retrieval by combining exact match and inverted index search. CITADEL [20] further introduces lexical routing to avoid the lexical mismatch issue in COIL. ME-BERT [29] and MVR [44] propose to use a portion of token embeddings for late interaction. ALIGNER [33] frames multi-vector retrieval as an alignment problem and uses entropy-regularized linear programming to solve it.

除了ColBERT [18]之外，COIL [9]通过结合精确匹配和倒排索引搜索来加速检索。CITADEL [20]进一步引入了词法路由，以避免COIL中出现的词法不匹配问题。ME - BERT [29]和MVR [44]提议使用一部分词元嵌入进行后期交互。ALIGNER [33]将多向量检索构建为一个对齐问题，并使用熵正则化线性规划来解决它。

## 5 CONCLUSION

## 5 结论

In this paper, we propose an efficient yet effective implementation for multi-vector retrieval using sparsified late interaction, which yields fast retrieval speed, small index size, and full compatibility with off-the-shelf search toolkits such as Pyserini. The key ingredients of SLIM include sparse lexical projection using the MLM layer, computing the lower- and upper-bounds of the late interaction, as well as token weight and postings pruning. Experiments on both in-domain and out-of-domain information retrieval datasets show that SLIM achieves comparable accuracy to ColBERT-v2 while being more efficient in terms of both space and time.

在本文中，我们提出了一种使用稀疏化后期交互的高效且有效的多向量检索实现方法，该方法具有快速的检索速度、较小的索引大小，并且与Pyserini等现成的搜索工具包完全兼容。SLIM的关键要素包括使用掩码语言模型（MLM）层进行稀疏词法投影、计算后期交互的上下界，以及词元权重和倒排列表剪枝。在领域内和领域外信息检索数据集上的实验表明，SLIM在准确性上与ColBERT - v2相当，同时在空间和时间方面都更高效。

## ACKNOWLEDGEMENTS

## 致谢

This research was supported in part by the Natural Sciences and Engineering Research Council (NSERC) of Canada; computational resources were provided by Compute Canada.

本研究部分得到了加拿大自然科学与工程研究委员会（NSERC）的支持；计算资源由加拿大计算公司（Compute Canada）提供。

## REFERENCES

## 参考文献

[1] Yang Bai, Xiaoguang Li, Gang Wang, Chaoliang Zhang, Lifeng Shang, Jun Xu, Zhaowei Wang, Fangshan Wang, and Qun Liu. 2020. SparTerm: Learning Term-based Sparse Representation for Fast Text Retrieval. https://doi.org/10.48550/ ARXIV.2010.00768

[1] 白杨、李晓光、王刚、张朝亮、尚立峰、徐军、王兆伟、王房山和刘群。2020年。SparTerm：学习基于词项的稀疏表示以实现快速文本检索。https://doi.org/10.48550/ ARXIV.2010.00768

[2] Daniel Cer, Yinfei Yang, Sheng-yi Kong, Nan Hua, Nicole Limtiaco, Rhomni St. John, Noah Constant, Mario Guajardo-Cespedes, Steve Yuan, Chris Tar, Yun-Hsuan Sung, Brian Strope, and Ray Kurzweil. 2018. Universal Sentence Encoder. https://doi.org/10.48550/ARXIV.1803.11175

[2] 丹尼尔·塞尔、杨荫飞、孔圣毅、华楠、妮可·林蒂亚科、罗姆尼·圣约翰、诺亚·康斯坦特、马里奥·瓜哈尔多 - 塞斯佩德斯、史蒂夫·袁、克里斯·塔尔、宋韵璇、布莱恩·斯特罗普和雷·库兹韦尔。2018年。通用句子编码器。https://doi.org/10.48550/ARXIV.1803.11175

[3] Nick Craswell, Bhaskar Mitra, Emine Yilmaz, and Daniel Campos. 2021. Overview of the TREC 2020 deep learning track. https://doi.org/10.48550/ARXIV.2102.07662

[3] 尼克·克拉斯韦尔、巴斯卡尔·米特拉、埃米内·伊尔马兹和丹尼尔·坎波斯。2021年。2020年TREC深度学习赛道概述。https://doi.org/10.48550/ARXIV.2102.07662

[4] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). Association for Computational Linguistics, Minneapolis, Minnesota, 4171-4186. https://doi.org/10.18653/v1/N19-1423

[4] 雅各布·德夫林、张明伟、肯顿·李和克里斯蒂娜·图托纳娃。2019年。BERT：用于语言理解的深度双向变换器预训练。见《2019年北美计算语言学协会会议：人类语言技术》论文集，第1卷（长论文和短论文）。计算语言学协会，明尼阿波利斯，明尼苏达州，4171 - 4186。https://doi.org/10.18653/v1/N19 - 1423

[5] Thibault Formal, Carlos Lassance, Benjamin Piwowarski, and Stéphane Clinchant. 2021. SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval. https://doi.org/10.48550/ARXIV.2109.10086

[5] 蒂博·福尔马尔、卡洛斯·拉萨斯、本杰明·皮沃瓦尔斯基和斯特凡·克林尚。2021年。SPLADE v2：用于信息检索的稀疏词法和扩展模型。https://doi.org/10.48550/ARXIV.2109.10086

[6] Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant. 2021. SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. Association for Computing Machinery. https://doi.org/10.1145/3404835.3463098

[6] 蒂博·福尔马尔、本杰明·皮沃瓦尔斯基和斯特凡·克林尚。2021年。SPLADE：用于第一阶段排序的稀疏词法和扩展模型。见《第44届国际ACM SIGIR信息检索研究与发展会议论文集》。美国计算机协会。https://doi.org/10.1145/3404835.3463098

[7] Luyu Gao and Jamie Callan. 2021. Condenser: a Pre-training Architecture for Dense Retrieval. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, Online and Punta Cana, Dominican Republic, 981-993. https://doi.org/10.18653/v1/2021.emnlp-main.75

[7] 高宇和杰米·卡兰。2021年。Condenser：一种用于密集检索的预训练架构。见《2021年自然语言处理经验方法会议论文集》。计算语言学协会，线上和多米尼加共和国蓬塔卡纳，981 - 993。https://doi.org/10.18653/v1/2021.emnlp - main.75

[8] Luyu Gao and Jamie Callan. 2022. Unsupervised Corpus Aware Language Model Pre-training for Dense Passage Retrieval. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, Dublin, Ireland, 2843-2853. https://doi.org/10.18653/v1/2022.acl-long.203

[8] 高宇和杰米·卡兰。2022年。用于密集段落检索的无监督语料感知语言模型预训练。见《计算语言学协会第60届年会论文集（第1卷：长论文）》。计算语言学协会，爱尔兰都柏林，2843 - 2853。https://doi.org/10.18653/v1/2022.acl - long.203

[9] Luyu Gao, Zhuyun Dai, and Jamie Callan. 2021. COIL: Revisit Exact Lexical Match in Information Retrieval with Contextualized Inverted List. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. Association for Computational Linguistics, Online, 3030-3042. https://doi.org/10.18653/v1/2021.naacl-main.241

[9] 高璐宇（Luyu Gao）、戴竹云（Zhuyun Dai）和杰米·卡兰（Jamie Callan）。2021 年。COIL：利用上下文倒排列表重新审视信息检索中的精确词汇匹配。见《2021 年北美计算语言学协会人类语言技术会议论文集》。计算语言学协会，线上会议，3030 - 3042 页。https://doi.org/10.18653/v1/2021.naacl-main.241

[10] Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. SimCSE: Simple Contrastive Learning of Sentence Embeddings. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, Online and Punta Cana, Dominican Republic, 6894-6910. https://doi.org/10.18653/v1/2021.emnlp-main.552

[10] 高天宇（Tianyu Gao）、姚兴成（Xingcheng Yao）和陈丹琦（Danqi Chen）。2021 年。SimCSE：句子嵌入的简单对比学习。见《2021 年自然语言处理经验方法会议论文集》。计算语言学协会，线上及多米尼加共和国蓬塔卡纳，6894 - 6910 页。https://doi.org/10.18653/v1/2021.emnlp-main.552

[11] Ruiqi Guo, Philip Sun, Erik Lindgren, Quan Geng, David Simcha, Felix Chern, and Sanjiv Kumar. 2020. Accelerating Large-Scale Inference with Anisotropic Vector Quantization. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event (Proceedings of Machine Learning Research, Vol. 119). PMLR, 3887-3896. http://proceedings.mlr.press/ v119/guo20h.html

[11] 郭瑞琪（Ruiqi Guo）、菲利普·孙（Philip Sun）、埃里克·林德格伦（Erik Lindgren）、耿全（Quan Geng）、大卫·辛查（David Simcha）、陈菲利克斯（Felix Chern）和桑吉夫·库马尔（Sanjiv Kumar）。2020 年。利用各向异性向量量化加速大规模推理。见《第 37 届国际机器学习会议论文集，ICML 2020，2020 年 7 月 13 - 18 日，虚拟会议（机器学习研究会议录，第 119 卷）》。PMLR，3887 - 3896 页。http://proceedings.mlr.press/ v119/guo20h.html

[12] Hua He and Jimmy Lin. 2016. Pairwise Word Interaction Modeling with Deep Neural Networks for Semantic Similarity Measurement. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. Association for Computational Linguistics, San Diego, California, 937-948. https://doi.org/10.18653/v1/N16-1108

[12] 何华（Hua He）和林吉米（Jimmy Lin）。2016 年。用于语义相似度测量的深度神经网络成对词交互建模。见《2016 年北美计算语言学协会人类语言技术会议论文集》。计算语言学协会，加利福尼亚州圣地亚哥，937 - 948 页。https://doi.org/10.18653/v1/N16-1108

[13] Sebastian Hofstätter, Omar Khattab, Sophia Althammer, Mete Sertkan, and Allan Hanbury. 2022. Introducing Neural Bag of Whole-Words with ColBERTer: Contextualized Late Interactions Using Enhanced Reduction. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management. Association for Computing Machinery, 737-747. https://doi.org/10.1145/3511808.3557367

[14] 塞巴斯蒂安·霍夫施泰特（Sebastian Hofstätter）、林盛杰（Sheng-Chieh Lin）、杨政宏（Jheng-Hong Yang）、林吉米（Jimmy Lin）和艾伦·汉伯里（Allan Hanbury）。2021 年。通过平衡主题感知采样高效训练有效的密集检索器。见《第 44 届 ACM 国际信息检索研究与发展会议论文集》。美国计算机协会，113 - 122 页。https://doi.org/10.1145/3404835.3462891

[14] Sebastian Hofstätter, Sheng-Chieh Lin, Jheng-Hong Yang, Jimmy Lin, and Allan Hanbury. 2021. Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. Association for Computing Machinery, 113-122. https://doi.org/10.1145/3404835.3462891

[14] 塞巴斯蒂安·霍夫施泰特（Sebastian Hofstätter）、林盛杰（Sheng-Chieh Lin）、杨政宏（Jheng-Hong Yang）、林吉米（Jimmy Lin）和艾伦·汉伯里（Allan Hanbury）。2021 年。通过平衡主题感知采样高效训练有效的密集检索器。见《第 44 届 ACM 国际信息检索研究与发展会议论文集》。美国计算机协会，113 - 122 页。https://doi.org/10.1145/3404835.3462891

[15] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bo-janowski, Armand Joulin, and Edouard Grave. 2021. Unsupervised Dense Information Retrieval with Contrastive Learning. https://doi.org/10.48550/ARXIV.2112.09118

[15] 高蒂埃·伊扎卡尔（Gautier Izacard）、玛蒂尔德·卡龙（Mathilde Caron）、卢卡斯·侯赛尼（Lucas Hosseini）、塞巴斯蒂安·里德尔（Sebastian Riedel）、彼得·博亚诺夫斯基（Piotr Bojanowski）、阿尔芒·朱林（Armand Joulin）和爱德华·格雷夫（Edouard Grave）。2021 年。基于对比学习的无监督密集信息检索。https://doi.org/10.48550/ARXIV.2112.09118

[16] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2021. Billion-Scale Similarity Search with GPUs. IEEE Transactions on Big Data 7, 03 (jul 2021), 535-547. https://doi.org/10.1109/TBDATA.2019.2921572

[16] 杰夫·约翰逊（Jeff Johnson）、马蒂亚斯·杜泽（Matthijs Douze）和埃尔韦·热古（Hervé Jégou）。2021 年。基于 GPU 的十亿级相似度搜索。《IEEE 大数据汇刊》7 卷 03 期（2021 年 7 月），535 - 547 页。https://doi.org/10.1109/TBDATA.2019.2921572

[17] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for Open-Domain Question Answering. In Proceedings of the 2020 Conference on Empirical

[17] 弗拉基米尔·卡尔普欣（Vladimir Karpukhin）、巴拉斯·奥古兹（Barlas Oguz）、闵世元（Sewon Min）、帕特里克·刘易斯（Patrick Lewis）、莱德尔·吴（Ledell Wu）、谢尔盖·叶杜诺夫（Sergey Edunov）、陈丹琦（Danqi Chen）和易文涛（Wen-tau Yih）。2020 年。用于开放域问答的密集段落检索。见《2020 年自然语言处理经验方法会议论文集》

Methods in Natural Language Processing (EMNLP). Association for Computational Linguistics, Online, 6769-6781. https://doi.org/10.18653/v1/2020.emnlp-main.550

（EMNLP）。计算语言学协会，线上会议，6769 - 6781 页。https://doi.org/10.18653/v1/2020.emnlp-main.550

[18] Omar Khattab and Matei Zaharia. 2020. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. Association for Computing Machinery, 39-48. https://doi.org/10.1145/ 3397271.3401075

[18] 奥马尔·哈塔布（Omar Khattab）和马特·扎哈里亚（Matei Zaharia）。2020 年。ColBERT：通过基于 BERT 的上下文后期交互实现高效有效的段落搜索。见《第 44 届 ACM 国际信息检索研究与发展会议论文集》。美国计算机协会，39 - 48 页。https://doi.org/10.1145/ 3397271.3401075

[19] Wuwei Lan and Wei Xu. 2018. Neural Network Models for Paraphrase Identification, Semantic Textual Similarity, Natural Language Inference, and Question Answering. In Proceedings of the 27th International Conference on Computational Linguistics. Association for Computational Linguistics, Santa Fe, New Mexico, USA, 3890-3902. https://aclanthology.org/C18-1328

[19] 蓝武威（Wuwei Lan）和徐伟（Wei Xu）。2018年。用于释义识别、语义文本相似度、自然语言推理和问答的神经网络模型。见《第27届国际计算语言学会议论文集》。美国新墨西哥州圣达菲市计算语言学协会，3890 - 3902。https://aclanthology.org/C18-1328

[20] Minghan Li, Sheng-Chieh Lin, Barlas Oguz, Asish Ghoshal, Jimmy Lin, Yashar Mehdad, Wen-tau Yih, and Xilun Chen. 2022. CITADEL: Conditional Token Interaction via Dynamic Lexical Routing for Efficient and Effective Multi-Vector Retrieval. https://doi.org/10.48550/ARXIV.2211.10411

[20] 李明翰（Minghan Li）、林圣杰（Sheng - Chieh Lin）、巴拉斯·奥古兹（Barlas Oguz）、阿西什·戈沙尔（Asish Ghoshal）、吉米·林（Jimmy Lin）、亚沙尔·梅赫达德（Yashar Mehdad）、易文涛（Wen - tau Yih）和陈希伦（Xilun Chen）。2022年。CITADEL：通过动态词法路由实现高效有效的多向量检索的条件Token交互。https://doi.org/10.48550/ARXIV.2211.10411

[21] Jimmy Lin and Xueguang Ma. 2021. A Few Brief Notes on DeepImpact, COIL, and a Conceptual Framework for Information Retrieval Techniques. https: //doi.org/10.48550/ARXIV.2106.14807

[21] 吉米·林（Jimmy Lin）和马学光（Xueguang Ma）。2021年。关于DeepImpact、COIL和信息检索技术概念框架的几点简要说明。https://doi.org/10.48550/ARXIV.2106.14807

[22] Jimmy Lin, Xueguang Ma, Sheng-Chieh Lin, Jheng-Hong Yang, Ronak Pradeep, and Rodrigo Nogueira. 2021. Pyserini: A Python Toolkit for Reproducible Information Retrieval Research with Sparse and Dense Representations. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. Association for Computing Machinery, 2356-2362. https://doi.org/10.1145/3404835.3463238

[22] 吉米·林（Jimmy Lin）、马学光（Xueguang Ma）、林圣杰（Sheng - Chieh Lin）、杨政宏（Jheng - Hong Yang）、罗纳克·普拉迪普（Ronak Pradeep）和罗德里戈·诺盖拉（Rodrigo Nogueira）。2021年。Pyserini：一个用于使用稀疏和密集表示进行可重现信息检索研究的Python工具包。见《第44届国际ACM SIGIR信息检索研究与发展会议论文集》。美国计算机协会，2356 - 2362。https://doi.org/10.1145/3404835.3463238

[23] Sheng-Chieh Lin, Akari Asai, Minghan Li, Barlas Oguz, Jimmy Lin, Yashar Mehdad, Wen-tau Yih, and Xilun Chen. 2023. How to Train Your DRAGON: Diverse Augmentation Towards Generalizable Dense Retrieval. https://doi.org/ 10.48550/ARXIV.2302.07452

[23] 林圣杰（Sheng - Chieh Lin）、浅井朱里（Akari Asai）、李明翰（Minghan Li）、巴拉斯·奥古兹（Barlas Oguz）、吉米·林（Jimmy Lin）、亚沙尔·梅赫达德（Yashar Mehdad）、易文涛（Wen - tau Yih）和陈希伦（Xilun Chen）。2023年。如何训练你的DRAGON：实现可泛化密集检索的多样化增强。https://doi.org/ 10.48550/ARXIV.2302.07452

[24] Sheng-Chieh Lin, Minghan Li, and Jimmy Lin. 2022. Aggretriever: A Simple Approach to Aggregate Textual Representation for Robust Dense Passage Retrieval. https://doi.org/10.48550/ARXIV.2208.00511

[24] 林圣杰（Sheng - Chieh Lin）、李明翰（Minghan Li）和吉米·林（Jimmy Lin）。2022年。Aggretriever：一种用于鲁棒密集段落检索的聚合文本表示的简单方法。https://doi.org/10.48550/ARXIV.2208.00511

[25] Sheng-Chieh Lin and Jimmy Lin. 2022. A Dense Representation Framework for Lexical and Semantic Matching. https://doi.org/10.48550/ARXIV.2206.09912

[25] 林圣杰（Sheng - Chieh Lin）和吉米·林（Jimmy Lin）。2022年。一种用于词法和语义匹配的密集表示框架。https://doi.org/10.48550/ARXIV.2206.09912

[26] Sheng-Chieh Lin, Jheng-Hong Yang, and Jimmy Lin. 2021. In-Batch Negatives for Knowledge Distillation with Tightly-Coupled Teachers for Dense Retrieval. In Proceedings of the 6th Workshop on Representation Learning for NLP (RepL4NLP- 2021). Association for Computational Linguistics, Online, 163-173. https://doi.org/10.18653/v1/2021.repl4nlp-1.17

[26] 林圣杰（Sheng - Chieh Lin）、杨政宏（Jheng - Hong Yang）和吉米·林（Jimmy Lin）。2021年。用于密集检索的紧密耦合教师知识蒸馏的批内负样本。见《第6届自然语言处理表示学习研讨会（RepL4NLP - 2021）论文集》。计算语言学协会，线上会议，163 - 173。https://doi.org/10.18653/v1/2021.repl4nlp - 1.17

[27] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. RoBERTa: A Robustly Optimized BERT Pretraining Approach. https://doi.org/10.48550/ ARXIV.1907.11692

[27] 刘音涵（Yinhan Liu）、迈尔·奥特（Myle Ott）、纳曼·戈亚尔（Naman Goyal）、杜静菲（Jingfei Du）、曼达尔·乔希（Mandar Joshi）、陈丹琦（Danqi Chen）、奥默·利维（Omer Levy）、迈克·刘易斯（Mike Lewis）、卢克·泽特勒莫耶（Luke Zettlemoyer）和韦塞林·斯托扬诺夫（Veselin Stoyanov）。2019年。RoBERTa：一种稳健优化的BERT预训练方法。https://doi.org/10.48550/ ARXIV.1907.11692

[28] Shuqi Lu, Di He, Chenyan Xiong, Guolin Ke, Waleed Malik, Zhicheng Dou, Paul Bennett, Tie-Yan Liu, and Arnold Overwijk. 2021. Less is More: Pretrain a Strong Siamese Encoder for Dense Text Retrieval Using a Weak Decoder. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, Online and Punta Cana, Dominican Republic, 2780-2791. https://doi.org/10.18653/v1/2021.emnlp-main.220

[29] 陆舒琪（Shuqi Lu）、何迪（Di He）、熊晨彦（Chenyan Xiong）、柯国霖（Guolin Ke）、瓦利德·马利克（Waleed Malik）、窦志成（Zhicheng Dou）、保罗·贝内特（Paul Bennett）、刘铁岩（Tie - Yan Liu）和阿诺德·奥弗维克（Arnold Overwijk）。2021年。少即是多：使用弱解码器预训练用于密集文本检索的强孪生编码器。见《2021年自然语言处理经验方法会议论文集》。计算语言学协会，线上会议和多米尼加共和国蓬塔卡纳，2780 - 2791。https://doi.org/10.18653/v1/2021.emnlp - main.220

[29] Yi Luan, Jacob Eisenstein, Kristina Toutanova, and Michael Collins. 2021. Sparse, Dense, and Attentional Representations for Text Retrieval. Transactions of the Association for Computational Linguistics 9 (2021), 329-345. https://doi.org/10.1162/tacl_a_00369

[29] 栾义（Yi Luan）、雅各布·艾森斯坦（Jacob Eisenstein）、克里斯蒂娜·图托纳娃（Kristina Toutanova）和迈克尔·柯林斯（Michael Collins）。2021年。用于文本检索的稀疏、密集和注意力表示。《计算语言学协会汇刊》9（2021），329 - 345。https://doi.org/10.1162/tacl_a_00369

[30] Antonio Mallia, Omar Khattab, Torsten Suel, and Nicola Tonellotto. 2021. Learning Passage Impacts for Inverted Indexes. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. Association for Computing Machinery, 1723-1727. https://doi.org/10.1145/3404835.3463030

[30] 安东尼奥·马利亚（Antonio Mallia）、奥马尔·哈塔布（Omar Khattab）、托尔斯滕·苏埃尔（Torsten Suel）和尼古拉·托内洛托（Nicola Tonellotto）。2021年。学习倒排索引的段落影响。见《第44届国际计算机协会信息检索研究与发展会议论文集》。美国计算机协会，1723 - 1727。https://doi.org/10.1145/3404835.3463030

[31] Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze. 2008. Introduction to information retrieval. Cambridge University Press. https://doi.org/10.1017/CBO9780511809071

[31] 克里斯托弗·D·曼宁（Christopher D. Manning）、普拉巴卡尔·拉加万（Prabhakar Raghavan）和欣里希·舒尔策（Hinrich Schütze）。2008年。《信息检索导论》。剑桥大学出版社。https://doi.org/10.1017/CBO9780511809071

[32] Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Ti-wary, Rangan Majumder, and Li Deng. 2016. MS MARCO: A human generated machine reading comprehension dataset. In CoCo@ NIPS. https://www.microsoft.com/en-us/research/publication/ms-marco-human-generated-machine-reading-comprehension-dataset/

[32] 特里·阮（Tri Nguyen）、米尔·罗森伯格（Mir Rosenberg）、宋霞（Xia Song）、高剑锋（Jianfeng Gao）、索拉布·蒂瓦里（Saurabh Ti - wary）、兰甘·马朱姆德（Rangan Majumder）和李登（Li Deng）。2016年。MS MARCO：一个人工生成的机器阅读理解数据集。见CoCo@NIPS。https://www.microsoft.com/en - us/research/publication/ms - marco - human - generated - machine - reading - comprehension - dataset/

[33] Yujie Qian, Jinhyuk Lee, Sai Meher Karthik Duddu, Zhuyun Dai, Siddhartha Brahma, Iftekhar Naim, Tao Lei, and Vincent Y. Zhao. 2022. Multi-Vector Retrieval as Sparse Alignment. https://doi.org/10.48550/ARXIV.2211.01267

[33] 钱玉洁（Yujie Qian）、李晋赫（Jinhyuk Lee）、赛·梅赫尔·卡尔蒂克·杜杜（Sai Meher Karthik Duddu）、戴竹云（Zhuyun Dai）、悉达多·布拉马（Siddhartha Brahma）、伊夫特哈尔·奈姆（Iftekhar Naim）、雷涛（Tao Lei）和赵文森（Vincent Y. Zhao）。2022年。多向量检索作为稀疏对齐。https://doi.org/10.48550/ARXIV.2211.01267

[34] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). Association for Computational Linguistics, Hong Kong, China, 3982-3992. https://doi.org/10.18653/v1/D19-1410

[34] 尼尔斯·赖默斯（Nils Reimers）和伊琳娜·古列维奇（Iryna Gurevych）。2019年。句子BERT：使用孪生BERT网络的句子嵌入。见《2019年自然语言处理经验方法会议和第9届自然语言处理国际联合会议（EMNLP - IJCNLP）论文集》。计算语言学协会，中国香港，3982 - 3992。https://doi.org/10.18653/v1/D19 - 1410

[35] Stephen E. Robertson and Hugo Zaragoza. 2009. The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval 3, 4 (2009), 333-389. https://doi.org/10.1561/1500000019

[35] 斯蒂芬·E·罗伯逊（Stephen E. Robertson）和雨果·萨拉戈萨（Hugo Zaragoza）。2009年。概率相关性框架：BM25及其他。《信息检索基础与趋势》3，4（2009年），333 - 389。https://doi.org/10.1561/1500000019

[36] Gerard Salton and Christopher Buckley. 1988. Term-weighting approaches in automatic text retrieval. Information Processing & Management 24, 5 (1988), 513-523. https://doi.org/10.1016/0306-4573(88)90021-0

[36] 杰拉德·萨尔顿（Gerard Salton）和克里斯托弗·巴克利（Christopher Buckley）。1988年。自动文本检索中的词项加权方法。《信息处理与管理》24，5（1988年），513 - 523。https://doi.org/10.1016/0306 - 4573(88)90021 - 0

[37] Keshav Santhanam, Omar Khattab, Christopher Potts, and Matei Zaharia. 2022. PLAID: An Efficient Engine for Late Interaction Retrieval. https://doi.org/10.48550/ARXIV.2205.09707

[37] 凯沙夫·桑塔南（Keshav Santhanam）、奥马尔·哈塔布（Omar Khattab）、克里斯托弗·波茨（Christopher Potts）和马特·扎哈里亚（Matei Zaharia）。2022年。PLAID：一种用于后期交互检索的高效引擎。https://doi.org/10.48550/ARXIV.2205.09707

[38] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. 2022. ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. Association for Computational Linguistics, Seattle, United States, 3715-3734. https://doi.org/10.18653/v1/2022.naacl-main.272

[38] 凯沙夫·桑塔南（Keshav Santhanam）、奥马尔·哈塔布（Omar Khattab）、乔恩·萨德 - 法尔孔（Jon Saad - Falcon）、克里斯托弗·波茨（Christopher Potts）和马特·扎哈里亚（Matei Zaharia）。2022年。ColBERTv2：通过轻量级后期交互实现高效检索。见《2022年北美计算语言学协会人类语言技术会议论文集》。计算语言学协会，美国西雅图，3715 - 3734。https://doi.org/10.18653/v1/2022.naacl - main.272

[39] Tao Shen, Xiubo Geng, Chongyang Tao, Can Xu, Kai Zhang, and Daxin Jiang. 2022. UnifieR: A Unified Retriever for Large-Scale Retrieval. https://doi.org/10.48550/ARXIV.2205.11194

[39] 沈涛（Tao Shen）、耿秀波（Xiubo Geng）、陶重阳（Chongyang Tao）、徐灿（Can Xu）、张凯（Kai Zhang）和蒋大新（Daxin Jiang）。2022年。UnifieR：一种用于大规模检索的统一检索器。https://doi.org/10.48550/ARXIV.2205.11194

[40] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021. BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2). https://openreview.net/forum?id=wCu6T5xFjeJ

[40] 南丹·塔库尔（Nandan Thakur）、尼尔斯·赖默斯（Nils Reimers）、安德里亚斯·吕克莱（Andreas Rücklé）、阿比谢克·斯里瓦斯塔瓦（Abhishek Srivastava）和伊琳娜·古列维奇（Iryna Gurevych）。2021年。BEIR：一个用于信息检索模型零样本评估的异构基准。见《第三十五届神经信息处理系统数据集和基准赛道会议（第二轮）》。https://openreview.net/forum?id=wCu6T5xFjeJ

[41] Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern,

[41] 保利·维尔塔宁（Pauli Virtanen）、拉尔夫·戈默斯（Ralf Gommers）、特拉维斯·E·奥利芬特（Travis E. Oliphant）、马特·哈伯兰德（Matt Haberland）、泰勒·雷迪（Tyler Reddy）、大卫·库尔纳波（David Cournapeau）、叶夫根尼·布罗夫斯基（Evgeni Burovski）、佩鲁·彼得森（Pearu Peterson）、沃伦·韦克塞斯（Warren Weckesser）、乔纳森·布莱特（Jonathan Bright）、斯特凡·J·范德沃尔特（Stéfan J. van der Walt）、马修·布雷特（Matthew Brett）、约书亚·威尔逊（Joshua Wilson）、K·贾罗德·米尔曼（K. Jarrod Millman）、尼古拉·马约罗夫（Nikolay Mayorov）、安德鲁·R·J·尼尔森（Andrew R. J. Nelson）、埃里克·琼斯（Eric Jones）、罗伯特·克恩（Robert Kern）

Eric Larson, C J Carey, Ilhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. 2020. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods 17 (2020), 261-272. https://doi.org/10.1038/s41592-019-0686-2

埃里克·拉森（Eric Larson）、C·J·凯里（C J Carey）、伊尔汉·波拉特（Ilhan Polat）、冯宇（Yu Feng）、埃里克·W·摩尔（Eric W. Moore）、杰克·范德普拉斯（Jake VanderPlas）、丹尼斯·拉萨尔代（Denis Laxalde）、约瑟夫·佩克托尔德（Josef Perktold）、罗伯特·西姆尔曼（Robert Cimrman）、伊恩·亨里克森（Ian Henriksen）、E·A·金特罗（E. A. Quintero）、查尔斯·R·哈里斯（Charles R. Harris）、安妮·M·阿奇博尔德（Anne M. Archibald）、安东尼奥·H·里贝罗（Antônio H. Ribeiro）、法比安·佩德雷戈萨（Fabian Pedregosa）、保罗·范·米尔布雷赫特（Paul van Mulbregt）以及SciPy 1.0贡献者团队。2020年。《SciPy 1.0：Python科学计算基础算法》。《自然方法》（Nature Methods），第17卷（2020年），第261 - 272页。https://doi.org/10.1038/s41592-019-0686-2

[42] Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N. Bennett, Junaid Ahmed, and Arnold Overwijk. 2021. Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net. https://openreview.net/forum?id=zeFrfgyZln

[42] 熊磊（Lee Xiong）、熊晨燕（Chenyan Xiong）、李烨（Ye Li）、邓国峰（Kwok-Fung Tang）、刘佳琳（Jialin Liu）、保罗·N·贝内特（Paul N. Bennett）、朱奈德·艾哈迈德（Junaid Ahmed）和阿诺德·奥弗维克（Arnold Overwijk）。2021年。《用于密集文本检索的近似最近邻负对比学习》。收录于第九届学习表征国际会议（9th International Conference on Learning Representations，ICLR 2021），线上会议，奥地利，2021年5月3 - 7日。OpenReview.net。https://openreview.net/forum?id=zeFrfgyZln

[43] Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Jiafeng Guo, Min Zhang, and Shaoping Ma. 2021. Optimizing Dense Retrieval Model Training with Hard Negatives. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. Association for Computing Machinery, 1503-1512. https://doi.org/10.1145/3404835.3462880

[43] 詹景涛（Jingtao Zhan）、毛佳欣（Jiaxin Mao）、刘奕群（Yiqun Liu）、郭佳峰（Jiafeng Guo）、张敏（Min Zhang）和马少平（Shaoping Ma）。2021年。《利用难负样本优化密集检索模型训练》。收录于第44届ACM信息检索研究与发展国际会议论文集（Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval）。美国计算机协会，第1503 - 1512页。https://doi.org/10.1145/3404835.3462880

[44] Shunyu Zhang, Yaobo Liang, Ming Gong, Daxin Jiang, and Nan Duan. 2022. MultiView Document Representation Learning for Open-Domain Dense Retrieval. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, Dublin, Ireland, 5990-6000. https://doi.org/10.18653/v1/2022.acl-long.414

[44] 张顺宇（Shunyu Zhang）、梁耀波（Yaobo Liang）、龚鸣（Ming Gong）、蒋大新（Daxin Jiang）和段楠（Nan Duan）。2022年。《用于开放领域密集检索的多视图文档表示学习》。收录于计算语言学协会第60届年会论文集（Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics，第1卷：长论文）。计算语言学协会，爱尔兰都柏林，第5990 - 6000页。https://doi.org/10.18653/v1/2022.acl-long.414

[45] Yinan Zhang, Raphael Tang, and Jimmy Lin. 2019. Explicit Pairwise Word Interaction Modeling Improves Pretrained Transformers for English Semantic Similarity Tasks. https://doi.org/10.48550/ARXIV.1911.02847

[45] 张一楠（Yinan Zhang）、拉斐尔·唐（Raphael Tang）和吉米·林（Jimmy Lin）。2019年。《显式成对词交互建模改进预训练Transformer用于英语语义相似度任务》。https://doi.org/10.48550/ARXIV.1911.02847