# CITADEL: Conditional Token Interaction via Dynamic Lexical Routing for Efficient and Effective Multi-Vector Retrieval

# CITADEL：通过动态词法路由实现条件化Token交互以进行高效有效的多向量检索

Minghan Li ${}^{1 * }$ ,Sheng-Chieh Lin ${}^{1}$ ,Barlas Oguz ${}^{2}$ ,Asish Ghoshal ${}^{2}$ , Jimmy Lin ${}^{1}$ ,Yashar Mehdad ${}^{2}$ ,Wen-tau Yih ${}^{2}$ ,and Xilun Chen ${}^{2 \dagger  }$

李明翰 ${}^{1 * }$、林圣杰 ${}^{1}$、巴拉斯·奥古兹 ${}^{2}$、阿西什·戈沙尔 ${}^{2}$、吉米·林 ${}^{1}$、亚沙尔·梅赫达德 ${}^{2}$、易文涛 ${}^{2}$ 和陈希伦 ${}^{2 \dagger  }$

University of Waterloo ${}^{1}$ , Meta ${\mathrm{{AI}}}^{2}$

滑铁卢大学 ${}^{1}$、Meta公司 ${\mathrm{{AI}}}^{2}$

\{m6921i,s269lin,jimmylin\}@uwaterloo.ca

\{m6921i,s269lin,jimmylin\}@uwaterloo.ca

\{barlaso, aghoshal, mehdad, scotty ih, xilun\} @meta.com

\{barlaso, aghoshal, mehdad, scotty ih, xilun\} @meta.com

## Abstract

## 摘要

Multi-vector retrieval methods combine the merits of sparse (e.g. BM25) and dense (e.g. DPR) retrievers and achieve state-of-the-art performance on various retrieval tasks. These methods, however, are orders of magnitude slower and need more space to store their indexes compared to their single-vector counterparts. In this paper, we unify different multi-vector retrieval models from a token routing viewpoint and propose conditional token interaction via dynamic lexical routing, namely CITADEL, for efficient and effective multi-vector retrieval. CITADEL learns to route each token vector to the predicted lexical "keys" such that a query token vector only interacts with document token vectors routed to the same key. This design significantly reduces the computation cost while maintaining high accuracy. Notably, CITADEL achieves the same or slightly better performance than the previous state of the art, ColBERT-v2, on both in-domain (MS MARCO) and out-of-domain (BEIR) evaluations, while being nearly 40 times faster. Source code and data are available at https: //github.com/facebookresearch/ dpr-scale/tree/citadel.

多向量检索方法结合了稀疏（如BM25）和密集（如DPR）检索器的优点，并在各种检索任务中取得了最先进的性能。然而，与单向量检索方法相比，这些方法的速度要慢几个数量级，并且需要更多的空间来存储其索引。在本文中，我们从Token路由的角度统一了不同的多向量检索模型，并提出了通过动态词法路由实现条件化Token交互的方法，即CITADEL，用于高效有效的多向量检索。CITADEL学习将每个Token向量路由到预测的词法“键”，使得查询Token向量仅与路由到相同键的文档Token向量进行交互。这种设计在保持高精度的同时显著降低了计算成本。值得注意的是，CITADEL在领域内（MS MARCO）和领域外（BEIR）评估中都取得了与之前最先进的ColBERT - v2相同或略好的性能，同时速度快了近40倍。源代码和数据可在https://github.com/facebookresearch/dpr - scale/tree/citadel获取。

## 1 Introduction

## 1 引言

The goal of information retrieval (Manning et al., 2008) is to find a set of related documents from a large data collection given a query. Traditional bag-of-words systems (Robertson and Zaragoza, 2009; Lin et al., 2021a) calculate the ranking scores based on the query terms appearing in each document, which have been widely adopted in many applications such as web search (Nguyen et al., 2016; Noy et al., 2019) and open-domain question answering (Chen et al., 2017; Lee et al., 2019). Recently, dense retrieval (Karpukhin et al., 2020) based on pre-trained language models (Devlin et al., 2019; Liu et al., 2019) has been shown to be very effective. It circumvents the term mismatch problem in bag-of-words systems by encoding the queries and documents into low-dimensional embeddings and using their dot product as the similarity score (Figure 2a). However, dense retrieval is less robust on entity-heavy questions (Sciavolino et al., 2021) and out-of-domain datasets (Thakur et al., 2021), therefore calling for better solutions (Formal et al., 2021b; Gao and Callan, 2022).

信息检索的目标（Manning等人，2008）是根据给定的查询从大型数据集中找到一组相关文档。传统的词袋系统（Robertson和Zaragoza，2009；Lin等人，2021a）根据每个文档中出现的查询词来计算排名分数，这些系统已广泛应用于许多应用中，如网页搜索（Nguyen等人，2016；Noy等人，2019）和开放域问答（Chen等人，2017；Lee等人，2019）。最近，基于预训练语言模型的密集检索（Karpukhin等人，2020）（Devlin等人，2019；Liu等人，2019）已被证明非常有效。它通过将查询和文档编码为低维嵌入并使用它们的点积作为相似度分数来规避词袋系统中的词不匹配问题（图2a）。然而，密集检索在实体密集的问题（Sciavolino等人，2021）和领域外数据集（Thakur等人，2021）上的鲁棒性较差，因此需要更好的解决方案（Formal等人，2021b；Gao和Callan，2022）。

<!-- Media -->

<!-- figureText: 0.36 COIL-tok ${10}^{2}$ CITADEL + PQ DPR-768 0.30 DPR-128 ${10}^{0}$ -->

<img src="https://cdn.noedgeai.com/0195a46c-451c-7169-bf50-83435cd534a9_0.jpg?x=858&y=591&w=560&h=416&r=0"/>

Figure 1: GPU latency vs ranking quality (MRR@10) on MS MARCO passages with an A100 GPU. The circle size represents the relative index storage on disk. All models are trained without hard-negative mining, distillation, or further pre-training.

图1：在配备A100 GPU的MS MARCO段落上的GPU延迟与排名质量（MRR@10）的关系。圆圈大小表示磁盘上的相对索引存储量。所有模型均未进行难负样本挖掘、蒸馏或进一步预训练。

<!-- Media -->

In contrast, multi-vector retrieval has shown strong performance on both in-domain and out-of-domain evaluations by taking into account token-level interaction. Among them, ColBERT (Khattab and Zaharia, 2020) is arguably the most celebrated method that has been the state of the art on multiple datasets so far. However, its wider application is hindered by its large index size and high retrieval latency. This problem results from the redundancy in the token interaction of ColBERT where many tokens might not contribute to the sentence semantics at all. To improve this, COIL (Gao et al., 2021a) imposes an exact match constraint on ColBERT 11891 Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics Volume 1: Long Papers, pages 11891-11907 July 9-14, 2023 ©2023 Association for Computational Linguistics for conditional token interaction, where only token embeddings with the same token id could interact with each other. Although reducing the latency, the word mismatch problem reoccurs and the model may fail to match queries and passages that use different words to express the same meaning.

相比之下，多向量检索通过考虑Token级别的交互，在领域内和领域外评估中都表现出了强大的性能。其中，ColBERT（Khattab和Zaharia，2020）可以说是目前在多个数据集上最著名的最先进方法。然而，其较大的索引大小和较高的检索延迟阻碍了它的更广泛应用。这个问题源于ColBERT的Token交互中的冗余，其中许多Token可能根本对句子语义没有贡献。为了改进这一点，COIL（Gao等人，2021a）对ColBERT施加了精确匹配约束 11891 《计算语言学协会第61届年会论文集第1卷：长论文》，第11891 - 11907页，2023年7月9 - 14日 ©2023计算语言学协会 以实现条件化Token交互，其中只有具有相同Token ID的Token嵌入才能相互交互。虽然降低了延迟，但词不匹配问题再次出现，并且模型可能无法匹配使用不同词语表达相同含义的查询和段落。

---

<!-- Footnote -->

*This work is done during Minghan's internship at Meta.

*这项工作是李明翰在Meta实习期间完成的。

${}^{ \dagger  }$ Xilun and Minghan contributed equally to this work.

${}^{ \dagger  }$ 席伦和明翰对这项工作贡献相同。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: Passage Vector Question Encoder Passage Encoder (b) ColBERT: all-to-all routing MaxSim (CLS) Relativity Question Encoder Passage Encoder (d) CITADEL: dynamic lexical routing Relativity Question Encoder Passage Encoder [CLS] Theory (a) Single-vector retriever: no routing Question Encoder Passage Encoder (c) COIL: static lexical routing using exact match -->

<img src="https://cdn.noedgeai.com/0195a46c-451c-7169-bf50-83435cd534a9_1.jpg?x=207&y=165&w=1201&h=803&r=0"/>

Figure 2: A unified token routing view over different multi-vector and single-vector retrieval models. The cylinder in (d) represents the lexical key for token routing. The grey tokens in (c) and (d) represent the tokens that do not contribute to the final similarity. For (d), CITADEL routes "light", "theory", and "relativity" to the "Einstein" key to avoid term mismatch using a learned routing function.

图2：不同多向量和单向量检索模型的统一Token路由视图。(d)中的圆柱体表示Token路由的词法键。(c)和(d)中的灰色Token表示对最终相似度无贡献的Token。对于(d)，CITADEL使用学习到的路由函数将“light（光）”、“theory（理论）”和“relativity（相对论）”路由到“Einstein（爱因斯坦）”键，以避免术语不匹配。

<!-- Media -->

In this paper, we first give a unified view of existing multi-vector retrieval methods based on token routing (Section 2), providing a new lens through which we expose the limitations of current models. Under the token routing view, ColBERT could be seen as all-to-all routing, where each query token exhaustively interacts with all passage tokens (Figure 2b). COIL, on the other hand, could be seen as static lexical routing using an exact match constraint, as each query token only interacts with the passage tokens that have the same token id as the query token (Figure 2c).

在本文中，我们首先基于Token路由给出现有多向量检索方法的统一视图（第2节），为揭示当前模型的局限性提供了新视角。在Token路由视图下，ColBERT可被视为全连接路由，即每个查询Token与所有段落Token进行详尽交互（图2b）。另一方面，COIL可被视为使用精确匹配约束的静态词法路由，因为每个查询Token仅与具有与查询Token相同Token ID的段落Token进行交互（图2c）。

In contrast, we propose a novel conditional token interaction method using dynamic lexical routing called CITADEL as shown in Figure 2d. Instead of relying on static heuristics such as exact match, we train our model to dynamically moderate token interaction so that each query token only interacts with the most relevant tokens in the passage. This is achieved by using a lexical router, trained end-to-end with the rest of the model, to route each contextualized token embedding to a set of activated lexical "keys" in the vocabulary. In this way, each query token embedding only interacts with the passage token embeddings that have the same activated key, which is dynamically determined during computation. As we shall see in Section 5.1, this learning-based routing does not lose any accuracy compared to all-to-all routing while using fewer token interactions than COIL (Section 3.4), leading to a highly effective and efficient retriever.

相比之下，我们提出了一种名为CITADEL的新颖的条件Token交互方法，它使用动态词法路由，如图2d所示。我们的模型不依赖于精确匹配等静态启发式方法，而是经过训练以动态调节Token交互，使每个查询Token仅与段落中最相关的Token进行交互。这是通过使用一个与模型其余部分进行端到端训练的词法路由器来实现的，该路由器将每个上下文化的Token嵌入路由到词汇表中一组激活的词法“键”。通过这种方式，每个查询Token嵌入仅与具有相同激活键的段落Token嵌入进行交互，该激活键在计算过程中动态确定。正如我们将在第5.1节中看到的，这种基于学习的路由与全连接路由相比不会损失任何准确性，同时比COIL使用更少的Token交互（第3.4节），从而得到一个高效且有效的检索器。

Experiments on MS MARCO passages (Nguyen et al., 2016) and TREC DL show that CITADEL achieves the same level of accuracy as ColBERT-v2. We further test CITADEL on BEIR (Thakur et al., 2021) and CITADEL still manages to keep up with ColBERT-v2 (Santhanam et al., 2022b) which is the current state of the art. As for the latency, CITADEL can yield an average latency of ${3.21}\mathrm{\;{ms}}/$ query on MS MARCO passages using an A100 GPU,which is nearly ${40} \times$ faster than ColBERT-v2. By further combining with product quantization, CITADEL's index only takes 13.3 GB on MS MARCO passages and reduces the latency to ${0.9}\mathrm{\;{ms}}$ /query as shown in Figure 1.

在MS MARCO段落（Nguyen等人，2016）和TREC DL上的实验表明，CITADEL达到了与ColBERT - v2相同的准确率水平。我们进一步在BEIR（Thakur等人，2021）上测试了CITADEL，它仍然能够与当前最先进的ColBERT - v2（Santhanam等人，2022b）相媲美。在延迟方面，使用A100 GPU时，CITADEL在MS MARCO段落上的查询平均延迟为${3.21}\mathrm{\;{ms}}/$，比ColBERT - v2快近${40} \times$。通过进一步结合乘积量化，CITADEL在MS MARCO段落上的索引仅占用13.3 GB，并将延迟降低到${0.9}\mathrm{\;{ms}}$/查询，如图1所示。

## 2 A Unified Token Routing View of Multi-Vector Retrievers

## 2 多向量检索器的统一Token路由视图

We outline a unified view for understanding various neural retrievers using the concept of token routing that dictates token interaction.

我们概述了一个统一视图，用于使用规定Token交互的Token路由概念来理解各种神经检索器。

### 2.1 Single-Vector Retrieval

### 2.1 单向量检索

Given a collection of documents and a set of queries, single-vector models (Karpukhin et al., 2020; Izacard et al., 2022) use a bi-encoder structure where its query encoder ${\eta }_{Q}\left( \cdot \right)$ and document encoder ${\eta }_{D}\left( \cdot \right)$ are independent functions that map the input to a low-dimensional vector. Specifically, the similarity score $s$ between the query $q$ and document $d$ is defined by the dot product between their encoded vectors ${v}_{q} = {\eta }_{Q}\left( q\right)$ and ${v}_{d} = {\eta }_{D}\left( d\right)$ :

给定一组文档和一组查询，单向量模型（Karpukhin等人，2020；Izacard等人，2022）使用双编码器结构，其中其查询编码器${\eta }_{Q}\left( \cdot \right)$和文档编码器${\eta }_{D}\left( \cdot \right)$是将输入映射到低维向量的独立函数。具体而言，查询$q$和文档$d$之间的相似度得分$s$由它们编码后的向量${v}_{q} = {\eta }_{Q}\left( q\right)$和${v}_{d} = {\eta }_{D}\left( d\right)$的点积定义：

$$
s\left( {q,d}\right)  = {v}_{q}^{T}{v}_{d} \tag{1}
$$

As all the token embeddings are pooled before calculating the similarity score, no routing is committed as shown in Figure 2a.

由于在计算相似度得分之前对所有Token嵌入进行了池化，因此如图2a所示，未进行路由操作。

### 2.2 Multi-Vector Retrieval

### 2.2 多向量检索

ColBERT (Khattab and Zaharia, 2020) proposes late interaction between the tokens in a query $q = \left\{  {{q}_{1},{q}_{2},\cdots ,{q}_{N}}\right\}$ and a document $d =$ $\left\{  {{d}_{1},{d}_{2},\cdots ,{d}_{M}}\right\}   :$

ColBERT（Khattab和Zaharia，2020）提出了查询$q = \left\{  {{q}_{1},{q}_{2},\cdots ,{q}_{N}}\right\}$和文档$d =$中Token之间的后期交互$\left\{  {{d}_{1},{d}_{2},\cdots ,{d}_{M}}\right\}   :$

$$
s\left( {q,d}\right)  = \mathop{\sum }\limits_{{i = 1}}^{N}\mathop{\max }\limits_{j}{v}_{{q}_{i}}^{T}{v}_{{d}_{j}} \tag{2}
$$

where ${v}_{{q}_{i}}$ and ${v}_{{d}_{j}}$ denotes the last-layer contextu-alized token embeddings of BERT. This is known as the MaxSim operation which exhaustively compares each query token to all document tokens. We refer to this as all-to-all routing as shown in Figure 2b. The latency of ColBERT is inflated by the redundancy in the all-to-all routing, as many tokens do not contribute to the sentence semantics. This also drastically increases the storage, requiring complex engineering schemes to make it more practical (Santhanam et al., 2022b,a).

其中${v}_{{q}_{i}}$和${v}_{{d}_{j}}$表示BERT的最后一层上下文相关的词元嵌入（token embeddings）。这就是所谓的MaxSim操作，它会将每个查询词元与所有文档词元进行全面比较。我们将其称为全对全路由（all-to-all routing），如图2b所示。ColBERT的延迟因全对全路由中的冗余而增加，因为许多词元对句子语义并无贡献。这也极大地增加了存储需求，需要复杂的工程方案来使其更具实用性（Santhanam等人，2022b，a）。

Another representative multi-vector approach known as COIL (Gao et al., 2021a) proposes an exact match constraint on the MaxSim operation where only the embeddings with the same token id could interact with each other. Let ${\mathcal{J}}_{i} = \left\{  {j \mid  {d}_{j} = }\right.$ $\left. {{q}_{i},1 \leq  j \leq  M}\right\}$ be the subset of document tokens ${\left\{  {d}_{j}\right\}  }_{j = 1}^{M}$ that have the same token ID as query token ${q}_{i}$ ,then we have:

另一种具有代表性的多向量方法，即COIL（高等人，2021a），对MaxSim操作提出了精确匹配约束，即只有具有相同词元ID的嵌入才能相互作用。设${\mathcal{J}}_{i} = \left\{  {j \mid  {d}_{j} = }\right.$ $\left. {{q}_{i},1 \leq  j \leq  M}\right\}$是与查询词元${q}_{i}$具有相同词元ID的文档词元${\left\{  {d}_{j}\right\}  }_{j = 1}^{M}$的子集，那么我们有：

$$
s\left( {q,d}\right)  = \mathop{\sum }\limits_{{i = 1}}^{N}\mathop{\max }\limits_{{j \in  {\mathcal{J}}_{i}}}{v}_{{q}_{i}}^{T}{v}_{{d}_{j}}, \tag{3}
$$

It could be further combined with Equation (1) to improve the effectiveness if there's no word overlap between the query and documents.

如果查询和文档之间没有单词重叠，它可以进一步与公式（1）结合以提高有效性。

$$
s\left( {q,d}\right)  = {v}_{q}^{T}{v}_{d} + \mathop{\sum }\limits_{{i = 1}}^{N}\mathop{\max }\limits_{{j \in  {\mathcal{J}}_{i}}}{v}_{{q}_{i}}^{T}{v}_{{d}_{j}}. \tag{4}
$$

We refer to this token interaction as static lexical routing as shown in Figure 2c. As mentioned in Section 1, the word mismatch problem could happen if ${\mathcal{J}}_{i} = \varnothing$ for all ${q}_{i}$ ,which affects the retrieval accuracy. Moreover, common tokens such as "the" will be frequently routed, which will create much larger token indexes compared to those rare words. This bottlenecks the search latency as COIL needs to frequently iterate over large token indexes.

我们将这种词元交互称为静态词法路由（static lexical routing），如图2c所示。如第1节所述，如果对于所有${q}_{i}$都有${\mathcal{J}}_{i} = \varnothing$，则可能会出现单词不匹配问题，这会影响检索准确性。此外，像“the”这样的常用词元会被频繁路由，与那些稀有单词相比，这将创建大得多的词元索引。由于COIL需要频繁遍历大的词元索引，这会限制搜索延迟。

## 3 The CITADEL Method

## 3 CITADEL方法

### 3.1 Dynamic Lexical Routing

### 3.1 动态词法路由

Instead of using the wasteful all-to-all routing or the inflexible heuristics-based static routing, we would like our model to dynamically select which query and passage tokens should interact with each other based on their contextualized representation, which we refer to as dynamic lexical routing. Formally, the routing function (or router) routes each token to a set of lexical keys in the vocabulary and is defined as $\phi  : {\mathbb{R}}^{c} \rightarrow  {\mathbb{R}}^{\left| \mathcal{V}\right| }$ where $c$ is the embedding dimension and $\mathcal{V}$ is the lexicon of keys. For each contextualized token embedding, the router predicts a scalar score for each key in the lexicon indicating how relevant each token is to that key. Given a query token embedding ${v}_{{q}_{i}}$ and a document token vector ${v}_{{d}_{j}}$ ,the token level router representations are ${w}_{{q}_{i}} = \phi \left( {v}_{{q}_{i}}\right)$ and ${w}_{{d}_{j}} = \phi \left( {v}_{{d}_{j}}\right)$ , respectively. The elements in the router representations are then sorted in descending order and truncated by selecting the top- $K$ query keys and top- $L$ document keys,which are $\left\{  {{e}_{{q}_{i}}^{\left( 1\right) },{e}_{{q}_{i}}^{\left( 2\right) },\cdots ,{e}_{{q}_{i}}^{\left( K\right) }}\right\}$ and $\left\{  {{e}_{{d}_{j}}^{\left( 1\right) },{e}_{{d}_{j}}^{\left( 2\right) },\cdots ,{e}_{{d}_{j}}^{\left( L\right) }}\right\}$ for ${q}_{i}$ and ${d}_{j}$ ,respectively. In practice,we use $K = 1$ and $L = 5$ as the default option which will be discussed in Section 3.5 and Section 7. The corresponding routing weights for ${q}_{i}$ and ${d}_{j}$ are $\left\{  {{w}_{{q}_{i}}^{\left( 1\right) },{w}_{{q}_{i}}^{\left( 2\right) },\cdots ,{w}_{{q}_{i}}^{\left( K\right) }}\right\}$ and $\left\{  {{w}_{{d}_{j}}^{\left( 1\right) },{w}_{{d}_{j}}^{\left( 2\right) },\cdots ,{w}_{{d}_{j}}^{\left( L\right) }}\right\}$ ,respectively. The final similarity score is similar to Equation (3), but we substitute the static lexical routing subset ${\mathcal{J}}_{i}$ with a dynamic key set predicted by the router: ${\mathcal{E}}_{i}^{\left( k\right) } = \left\{  {j,l \mid  {e}_{{d}_{j}}^{\left( l\right) } = {e}_{{q}_{i}}^{\left( k\right) },1 \leq  j \leq  M,1 \leq  l \leq  L}\right\}$ for each key ${e}_{{q}_{i}}^{\left( k\right) }$ of the query token ${q}_{i}$ :

我们希望我们的模型基于查询和段落标记的上下文表示动态选择哪些标记应该相互交互，而不是使用浪费资源的全对全路由或缺乏灵活性的基于启发式的静态路由，我们将此过程称为动态词法路由。形式上，路由函数（或路由器）将每个标记路由到词汇表中的一组词法键，定义为$\phi  : {\mathbb{R}}^{c} \rightarrow  {\mathbb{R}}^{\left| \mathcal{V}\right| }$，其中$c$是嵌入维度，$\mathcal{V}$是键的词典。对于每个上下文标记嵌入，路由器为词典中的每个键预测一个标量分数，该分数表示每个标记与该键的相关性。给定查询标记嵌入${v}_{{q}_{i}}$和文档标记向量${v}_{{d}_{j}}$，标记级别的路由器表示分别为${w}_{{q}_{i}} = \phi \left( {v}_{{q}_{i}}\right)$和${w}_{{d}_{j}} = \phi \left( {v}_{{d}_{j}}\right)$。然后，路由器表示中的元素按降序排序，并通过选择前$K$个查询键和前$L$个文档键进行截断，对于${q}_{i}$和${d}_{j}$，它们分别是$\left\{  {{e}_{{q}_{i}}^{\left( 1\right) },{e}_{{q}_{i}}^{\left( 2\right) },\cdots ,{e}_{{q}_{i}}^{\left( K\right) }}\right\}$和$\left\{  {{e}_{{d}_{j}}^{\left( 1\right) },{e}_{{d}_{j}}^{\left( 2\right) },\cdots ,{e}_{{d}_{j}}^{\left( L\right) }}\right\}$。在实践中，我们默认使用$K = 1$和$L = 5$，这将在第3.5节和第7节中讨论。${q}_{i}$和${d}_{j}$的相应路由权重分别为$\left\{  {{w}_{{q}_{i}}^{\left( 1\right) },{w}_{{q}_{i}}^{\left( 2\right) },\cdots ,{w}_{{q}_{i}}^{\left( K\right) }}\right\}$和$\left\{  {{w}_{{d}_{j}}^{\left( 1\right) },{w}_{{d}_{j}}^{\left( 2\right) },\cdots ,{w}_{{d}_{j}}^{\left( L\right) }}\right\}$。最终相似度得分与公式（3）类似，但我们用路由器预测的动态键集替换了静态词法路由子集${\mathcal{J}}_{i}$：对于查询标记${q}_{i}$的每个键${e}_{{q}_{i}}^{\left( k\right) }$：

$$
s\left( {q,d}\right)  = \mathop{\sum }\limits_{{i = 1}}^{N}\mathop{\sum }\limits_{{k = 1}}^{K}\mathop{\max }\limits_{{j,l \in  {\mathcal{E}}_{i}^{\left( k\right) }}}{\left( {w}_{{q}_{i}}^{\left( k\right) } \cdot  {v}_{{q}_{i}}\right) }^{T}\left( {{w}_{{d}_{j}}^{\left( l\right) } \cdot  {v}_{{d}_{j}}}\right) , \tag{5}
$$

Optionally, all [CLS] tokens can be routed to an additional semantic key to complement our learned lexical routing. We then follow DPR (Karpukhin et al., 2020) to train the model contrastively. Given a query $q$ ,a positive document ${d}^{ + }$ ,and a set of negative documents ${D}^{ - }$ ,the constrastive loss is:

可选地，可以将所有[CLS]标记路由到一个额外的语义键，以补充我们学习到的词法路由。然后，我们遵循DPR（卡尔普欣等人，2020年）的方法对模型进行对比训练。给定一个查询$q$、一个正文档${d}^{ + }$和一组负文档${D}^{ - }$，对比损失为：

$$
{\mathcal{L}}_{\mathrm{e}} =  - \log \frac{\exp \left( {s\left( {q,{d}^{ + }}\right) }\right) }{\exp \left( {s\left( {q,{d}^{ + }}\right) }\right)  + \mathop{\sum }\limits_{{{d}^{ - } \in  {D}^{ - }}}\exp \left( {s\left( {q,{d}^{ - }}\right) }\right) }, \tag{6}
$$

such that the distance from the query to the positive document ${d}^{ + }$ is smaller than the query to the negative document ${d}^{ - }$ .

使得查询到正文档${d}^{ + }$的距离小于查询到负文档${d}^{ - }$的距离。

### 3.2 Router Optimization

### 3.2 路由器优化

To train the router representation $\phi \left( q\right)$ and $\phi \left( d\right)$ , we adopt a contrastive loss such that the number of overlapped keys between a query and documents are large for positive $\left( {q,{d}^{ + }}\right)$ pairs and small for negative pairs $\left( {q,{d}^{ - }}\right)$ . We first pool the router representation for each query and document over the tokens. Given a sequence of token-level router representations $\left\{  {\phi \left( {v}_{1}\right) ,\phi \left( {v}_{2}\right) ,\cdots ,\phi \left( {v}_{M}\right) }\right\}$ , the sequence-level representation is defined as:

为了训练路由表示 $\phi \left( q\right)$ 和 $\phi \left( d\right)$，我们采用对比损失，使得查询与文档之间的重叠键数量在正 $\left( {q,{d}^{ + }}\right)$ 对中较多，在负对 $\left( {q,{d}^{ - }}\right)$ 中较少。我们首先对每个查询和文档的路由表示按词元进行池化操作。给定词元级路由表示序列 $\left\{  {\phi \left( {v}_{1}\right) ,\phi \left( {v}_{2}\right) ,\cdots ,\phi \left( {v}_{M}\right) }\right\}$，序列级表示定义为：

$$
\Phi  = \mathop{\max }\limits_{{j = 1}}^{M}\phi \left( {v}_{j}\right)  \tag{7}
$$

where the max operator is applied element-wise. Similar to (Formal et al., 2021a), We find max pooling works the best in practice compared to other pooling methods. Subsequently, the contrastive loss for training the router is:

其中，最大运算符按元素应用。与（Formal 等人，2021a）类似，我们发现与其他池化方法相比，最大池化在实践中效果最佳。随后，用于训练路由的对比损失为：

$$
{\mathcal{L}}_{\mathrm{r}} =  - \log \frac{\exp \left( {{\Phi }_{q}^{T}{\Phi }_{{d}^{ + }}}\right) }{\exp \left( {{\Phi }_{q}^{T}{\Phi }_{{d}^{ + }}}\right)  + \mathop{\sum }\limits_{{{d}^{ - } \in  {D}^{ - }}}\exp \left( {{\Phi }_{q}^{T}{\Phi }_{{d}^{ - }}}\right) }. \tag{8}
$$

In addition, we follow SPLADE (Formal et al., ${2021}\mathrm{\;b}$ ,a) to initialize the router with the pre-trained Masked Language Modelling Layer (MLM). Without proper initialization, it is difficult to optimize the router due the large lexical space and sparse activation. With the pre-trained MLM initialization, the router expands words with similar semantic meaning to sets of keys with large overlap at the beginning of training, making the contrastive loss easier to optimize.

此外，我们遵循 SPLADE（Formal 等人，${2021}\mathrm{\;b}$，a）的方法，使用预训练的掩码语言建模层（Masked Language Modelling Layer，MLM）来初始化路由。如果没有适当的初始化，由于词汇空间大且激活稀疏，很难对路由进行优化。通过预训练的 MLM 初始化，路由在训练开始时会将具有相似语义的单词扩展为大量重叠的键集，使对比损失更容易优化。

### 3.3 Sparsely Activated Router Design

### 3.3 稀疏激活路由设计

Softmax activation is commonly used for computing the routing weights in conditional computation models (Fedus et al., 2022; Mustafa et al., 2022). However, softmax often yields a small probability over a large number of dimensions (in our case, about ${30},{000})$ and the product of two probability values are even smaller, which makes it not suitable for yielding the routing weights ${w}_{{q}_{i}}^{\left( k\right) }$ and ${w}_{{d}_{j}}^{\left( l\right) }$ in Equation (5) as the corresponding gradients are too small. Instead, we use the activation from SPLADE to compute the router representation for a token embedding ${v}_{j}$ :

在条件计算模型中，Softmax 激活函数通常用于计算路由权重（Fedus 等人，2022；Mustafa 等人，2022）。然而，Softmax 通常在大量维度上产生较小的概率（在我们的例子中，约为 ${30},{000})$，并且两个概率值的乘积更小，这使得它不适合用于计算公式（5）中的路由权重 ${w}_{{q}_{i}}^{\left( k\right) }$ 和 ${w}_{{d}_{j}}^{\left( l\right) }$，因为相应的梯度太小。相反，我们使用 SPLADE 的激活函数来计算词元嵌入 ${v}_{j}$ 的路由表示：

$$
\phi \left( {v}_{j}\right)  = \log \left( {1 + \operatorname{ReLU}\left( {{W}^{T}{v}_{j} + b}\right) }\right) , \tag{9}
$$

where $W$ and $b$ are the weights and biases of the Masked Language Modeling (MLM) layer of BERT. The SPLADE activation brings extra advantages as the ReLU activation filters irrelevant keys while the log-saturation suppresses overly large "wacky" weights (Mackenzie et al., 2021).

其中 $W$ 和 $b$ 是 BERT 的掩码语言建模（Masked Language Modeling，MLM）层的权重和偏置。SPLADE 激活带来了额外的优势，因为 ReLU 激活会过滤掉不相关的键，而对数饱和会抑制过大的“怪异”权重（Mackenzie 等人，2021）。

### 3.4 Regularization for Routing

### 3.4 路由正则化

${\ell }_{1}$ Regularization. Routing each token to more than one key increases the overall size of the index. Therefore,we propose to use ${\ell }_{1}$ regularization on the router representation to encourage the router to only keep the most meaningful token interaction by pushing more routing weights to 0 :

${\ell }_{1}$ 正则化。将每个词元路由到多个键会增加索引的总体大小。因此，我们建议对路由表示使用 ${\ell }_{1}$ 正则化，通过将更多的路由权重推向 0 来鼓励路由仅保留最有意义的词元交互：

$$
{\mathcal{L}}_{\mathrm{s}} = \frac{1}{B}\mathop{\sum }\limits_{{i = 1}}^{B}\mathop{\sum }\limits_{{j = 1}}^{T}\mathop{\sum }\limits_{{k = 1}}^{\left| \mathcal{V}\right| }\phi {\left( {v}_{ij}\right) }^{\left( k\right) }, \tag{10}
$$

where $\left| \mathcal{V}\right|$ is the number of keys, $B$ is the batch size, and $T$ is the sequence length. As shown in Figure 6, CITADEL has a sparsely activated set of keys, by routing important words to multiple lexical keys while ignoring many less salient words, leading to effective yet efficient retrieval.

其中 $\left| \mathcal{V}\right|$ 是键的数量，$B$ 是批量大小，$T$ 是序列长度。如图 6 所示，CITADEL 具有一组稀疏激活的键，它将重要的单词路由到多个词汇键，同时忽略许多不太突出的单词，从而实现了高效且有效的检索。

Load Balancing. As mentioned in Section 2.2, the retrieval latency of COIL is bottlenecked by frequently searching overly large token indexes. This results from the static lexical routing where common "keys" have a larger chance to be activated, which results in large token indexes during indexing. Therefore, a vital point for reducing the latency of multi-vector models is to evenly distribute each token embedding to different keys. Inspired by Switch Transformers (Fedus et al., 2022), we propose to minimize the load balancing loss that approximates the expected "evenness" of the number of tokens being routed to each key:

负载均衡。如第 2.2 节所述，COIL 的检索延迟受到频繁搜索过大词元索引的瓶颈限制。这是由于静态词汇路由导致的，在这种路由方式下，常见的“键”更有可能被激活，从而在索引过程中产生大的词元索引。因此，降低多向量模型延迟的一个关键点是将每个词元嵌入均匀地分配到不同的键上。受 Switch Transformers（Fedus 等人，2022）的启发，我们建议最小化负载均衡损失，该损失近似于路由到每个键的词元数量的预期“均匀性”：

$$
{\mathcal{L}}_{\mathrm{b}} = \mathop{\sum }\limits_{{k = 1}}^{\left| \mathcal{V}\right| }{f}_{k} \cdot  {p}_{k} \tag{11}
$$

${p}_{k}$ is the batch approximation of the marginal probability of a token vector routed to the $k$ -th key:

${p}_{k}$ 是词元向量路由到第 $k$ 个键的边际概率的批量近似值：

$$
{p}_{k} = \frac{1}{B}\mathop{\sum }\limits_{{i = 1}}^{B}\mathop{\sum }\limits_{{j = 1}}^{T}\frac{\exp \left( {{W}_{k}^{T}{v}_{ij} + {b}_{k}}\right) }{\mathop{\sum }\limits_{{k}^{\prime }}\exp \left( {{W}_{{k}^{\prime }}^{T}{v}_{ij} + {b}_{{k}^{\prime }}}\right) }, \tag{12}
$$

where $W$ and $b$ are the weights and bias of the routing function in Equation (9) and ${v}_{ij}$ is the $j$ -th token embedding in sample $i$ of the batch. ${f}_{k}$ is the batch approximation of the total number of tokens being dispatched to the $k$ -th key:

其中 $W$ 和 $b$ 是公式（9）中路由函数的权重和偏置，${v}_{ij}$ 是批次中样本 $i$ 的第 $j$ 个词元嵌入。${f}_{k}$ 是分配到第 $k$ 个键的词元总数的批量近似值：

$$
{f}_{k} = \frac{1}{B}\mathop{\sum }\limits_{{i = 1}}^{B}\mathop{\sum }\limits_{{j = 1}}^{T}\mathbb{I}\left\{  {\operatorname{argmax}\left( {p}_{ij}\right)  = k}\right\}  , \tag{13}
$$

where ${p}_{ij} = \operatorname{softmax}\left( {{W}^{T}{v}_{ij} + b}\right)$ . Finally,we obtain the loss for training CITADEL:

其中 ${p}_{ij} = \operatorname{softmax}\left( {{W}^{T}{v}_{ij} + b}\right)$ 。最后，我们得到训练CITADEL的损失函数：

$$
\mathcal{L} = {\mathcal{L}}_{\mathrm{e}} + {\mathcal{L}}_{\mathrm{r}} + \alpha  \cdot  {\mathcal{L}}_{\mathrm{b}} + \beta  \cdot  {\mathcal{L}}_{\mathrm{s}}, \tag{14}
$$

where $\alpha  \geq  0$ and $\beta  \geq  0$ . The ${\ell }_{1}$ and load balancing loss are applied on both queries and documents.

其中 $\alpha  \geq  0$ 和 $\beta  \geq  0$ 。${\ell }_{1}$ 和负载均衡损失同时应用于查询和文档。

### 3.5 Inverted Index Retrieval

### 3.5 倒排索引检索

CITADEL builds an inverted index like BM25 but we use a vector instead of a scalar for each token and the doc product as the term relevance.

CITADEL像BM25一样构建倒排索引，但我们为每个Token（词元）使用向量而非标量，并使用文档积作为词项相关性。

Indexing and Post-hoc Pruning. To reduce index storage, we prune the vectors with routing weights less than a threshold $\tau$ after training. For a key $e$ in the lexicon $\mathcal{V}$ ,the token index ${\mathcal{I}}_{e}$ consists of token embeddings ${v}_{{d}_{j}}$ and the routing weight ${w}_{{d}_{j}}^{e}$ for all documents $d$ in the corpus $\mathcal{C}$ is:

索引构建与事后剪枝。为减少索引存储，我们在训练后对路由权重小于阈值 $\tau$ 的向量进行剪枝。对于词典 $\mathcal{V}$ 中的键 $e$ ，Token（词元）索引 ${\mathcal{I}}_{e}$ 由Token（词元）嵌入 ${v}_{{d}_{j}}$ 组成，并且语料库 $\mathcal{C}$ 中所有文档 $d$ 的路由权重 ${w}_{{d}_{j}}^{e}$ 为：

$$
{\mathcal{I}}_{e} = \left\{  {{w}_{{d}_{j}}^{e} \cdot  {v}_{{d}_{j}} \mid  {w}_{{d}_{j}}^{e} > \tau ,1 \leq  j \leq  M,\forall d \in  \mathcal{C}}\right\}  . \tag{15}
$$

We will discuss the impact of posthoc pruning in Section 5.2, where we find that posthoc pruning can reduce the index size by $3 \times$ without significant accuracy loss. The final search index is defined as $\mathcal{I} = \left\{  {{\mathcal{I}}_{e} \mid  e \in  \mathcal{V}}\right\}$ ,where the load-balancing loss in Equation (11) will encourage the size distribution over ${\mathcal{I}}_{e}$ to be as even as possible. In practice,we set the number of maximal routing keys of each token to 5 for the document and 1 for the query. The intuition is that the documents usually contain more information and need more key capacity, which is discussed in Section 7 in detail.

我们将在5.2节讨论事后剪枝的影响，在该节中我们发现事后剪枝可以将索引大小减少 $3 \times$ ，而不会显著降低准确率。最终的搜索索引定义为 $\mathcal{I} = \left\{  {{\mathcal{I}}_{e} \mid  e \in  \mathcal{V}}\right\}$ ，其中公式(11)中的负载均衡损失将促使 ${\mathcal{I}}_{e}$ 上的大小分布尽可能均匀。在实践中，我们将每个Token（词元）的最大路由键数量设置为文档5个、查询1个。直觉上，文档通常包含更多信息，需要更多的键容量，这将在第7节详细讨论。

Token Retrieval. Given a query $q$ ,CITADEL first encodes it into a sequence of token vectors ${\left\{  {v}_{{q}_{i}}\right\}  }_{i = 1}^{N}$ ,and then route each vector to its top-1 key $e$ with a routing weight ${w}_{{q}_{i}}^{e}$ . The final representation ${w}_{{q}_{i}}^{e} \cdot  {v}_{{q}_{i}}$ will be sent to the corresponding token index ${\mathcal{I}}_{E}$ for vector search. The final ranking list will be merged from each query token's document ranking according to Equation (5).

Token（词元）检索。给定查询 $q$ ，CITADEL首先将其编码为Token（词元）向量序列 ${\left\{  {v}_{{q}_{i}}\right\}  }_{i = 1}^{N}$ ，然后使用路由权重 ${w}_{{q}_{i}}^{e}$ 将每个向量路由到其排名第一的键 $e$ 。最终表示 ${w}_{{q}_{i}}^{e} \cdot  {v}_{{q}_{i}}$ 将被发送到相应的Token（词元）索引 ${\mathcal{I}}_{E}$ 进行向量搜索。最终的排名列表将根据公式(5)从每个查询Token（词元）的文档排名中合并得到。

## 4 Experiments

## 4 实验

### 4.1 MS MARCO Passages Retrieval

### 4.1 MS MARCO段落检索

We evaluate MS MARCO passages (Nguyen et al., 2016) and its shared tasks, TREC DL 2019/2020 passage ranking tasks (Craswell et al., 2020). Dataset details are provided in Appendix A.1. Following standard practice, we train CITADEL and other baseline models on MS MARCO passages and report the results on its dev-small set and TREC DL 2019/2020 test queries. The evaluation metrics are MRR@10, nDCG@10, and Recall@1000 (i.e., $\mathrm{R}@1\mathrm{\;K}$ ). We provide a detailed implementation of CITADEL and other baselines in Appendix A.

我们评估了MS MARCO段落（Nguyen等人，2016）及其共享任务，即TREC DL 2019/2020段落排名任务（Craswell等人，2020）。数据集详细信息见附录A.1。按照标准做法，我们在MS MARCO段落上训练CITADEL和其他基线模型，并在其小开发集和TREC DL 2019/2020测试查询上报告结果。评估指标为MRR@10、nDCG@10和Recall@1000（即 $\mathrm{R}@1\mathrm{\;K}$ ）。我们在附录A中提供了CITADEL和其他基线模型的详细实现。

Table 1 shows the in-domain evaluation results on MS MARCO passage and TREC DL 2019/2020. We divide the models into two classes: ones trained with only labels and BM25 hard negatives and the others trained with further pretraining (Gao and Callan, 2022), hard negative mining (Xiong et al., 2021), or distillation from a cross-encoder ${}^{1}$ . CITADEL is trained only with BM25 hard negatives,while ${\mathrm{{CITADEL}}}^{ + }$ is trained with cross-encoder distillation and one-round hard negative mining. The default pruning threshold is $\tau  = {0.9}$ . As shown in Section ${5.2},\tau$ can be adjusted to strike different balances between latency, index size and accuracy. In both categories,CITADEL/CITADEL ${}^{ + }$ outperforms the baseline models on the MS MARCO passages dev set and greatly reduces the search latency on both GPU and CPU. For example, ${\mathrm{{CITADEL}}}^{ + }$ achieves an average latency of ${3.21}\mathrm{\;{ms}}$ /query which is close to DPR-768 (1.28 ms/query) on GPU, while having a 25% higher MRR@10 score. CITADEL also maintains acceptable index sizes on disk, which can be further reduced using product quantization (Section 5.3). Although not able to outperform several baselines on TREC DL 2019/2020, we perform t-test $\left( {\mathrm{p} < {0.05}}\right)$ on CITADEL and ${\mathrm{{CITADEL}}}^{ + }$ against other baselines in their sub-categories and

表1展示了在MS MARCO段落数据集和TREC DL 2019/2020数据集上的领域内评估结果。我们将模型分为两类：仅使用标签和BM25困难负样本进行训练的模型，以及使用进一步预训练（Gao和Callan，2022）、困难负样本挖掘（Xiong等人，2021）或从交叉编码器蒸馏${}^{1}$进行训练的模型。CITADEL仅使用BM25困难负样本进行训练，而${\mathrm{{CITADEL}}}^{ + }$则通过交叉编码器蒸馏和一轮困难负样本挖掘进行训练。默认的剪枝阈值为$\tau  = {0.9}$。如${5.2},\tau$节所示，可以对其进行调整，以在延迟、索引大小和准确率之间取得不同的平衡。在这两类模型中，CITADEL/CITADEL ${}^{ + }$在MS MARCO段落开发集上的表现均优于基线模型，并显著降低了在GPU和CPU上的搜索延迟。例如，${\mathrm{{CITADEL}}}^{ + }$在GPU上的平均查询延迟达到${3.21}\mathrm{\;{ms}}$，接近DPR - 768（1.28毫秒/查询），同时MRR@10得分高出25%。CITADEL在磁盘上的索引大小也处于可接受范围，使用积量化（第5.3节）还可以进一步减小。尽管CITADEL在TREC DL 2019/2020数据集上无法超越几个基线模型，但我们对CITADEL和${\mathrm{{CITADEL}}}^{ + }$与其子类别中的其他基线模型进行了t检验$\left( {\mathrm{p} < {0.05}}\right)$，并且

---

<!-- Footnote -->

https://huggingface.co/cross-encoder/ ms-marco-MiniLM-L-6-v2

https://huggingface.co/cross-encoder/ ms-marco-MiniLM-L-6-v2

<!-- Footnote -->

---

<!-- Media -->

<table><tr><td rowspan="2">Models</td><td colspan="2">MARCO Dev</td><td colspan="2">TREC DL19</td><td colspan="2">TREC DL20</td><td colspan="2">Index Storage</td><td colspan="3">Latency (ms/query)</td></tr><tr><td>MRR@10</td><td>R@1k</td><td>nDCG@10</td><td>R@1k</td><td>nDCG@10</td><td>R@1k</td><td>Disk (GB)</td><td>Factor ${}^{1}$</td><td>Encode (GPU)</td><td>Search (GPU)</td><td>Search (CPU)</td></tr><tr><td colspan="12">Models trained with only BM25 hard negatives</td></tr><tr><td>BM25</td><td>0.188</td><td>0.858</td><td>0.506</td><td>0.739</td><td>0.488</td><td>0.733</td><td>0.67</td><td>$\times  \mathbf{{0.22}}$</td><td>✘</td><td>✘</td><td>40.1</td></tr><tr><td>DPR-128</td><td>0.285</td><td>0.937</td><td>0.576</td><td>0.702</td><td>0.603</td><td>0.757</td><td>4.33</td><td>$\times  {1.42}$</td><td>7.09</td><td>0.63</td><td>430</td></tr><tr><td>DPR-768</td><td>0.319</td><td>0.941</td><td>0.611</td><td>0.742</td><td>0.591</td><td>0.796</td><td>26.0</td><td>$\times  {8.52}$</td><td>7.01</td><td>1.28</td><td>2015</td></tr><tr><td>SPLADE</td><td>0.340</td><td>0.965</td><td>0.683</td><td>0.813</td><td>0.671</td><td>0.823</td><td>2.60</td><td>$\times  {0.85}$</td><td>7.13</td><td>✘</td><td>475</td></tr><tr><td>COIL-tok</td><td>0.350</td><td>0.964</td><td>0.660</td><td>0.809</td><td>0.679</td><td>0.825</td><td>52.5</td><td>$\times  {17.2}$</td><td>10.7</td><td>46.8</td><td>1295</td></tr><tr><td>COIL-full</td><td>0.353</td><td>0.967</td><td>0.704</td><td>0.835</td><td>0.688</td><td>0.841</td><td>78.5</td><td>$\times  {25.7}$</td><td>10.8</td><td>47.9</td><td>3258</td></tr><tr><td>ColBERT</td><td>0.360</td><td>0.968</td><td>0.694</td><td>0.830</td><td>0.676</td><td>0.837</td><td>154</td><td>$\times  {50.5}$</td><td>10.9</td><td>178</td><td>-</td></tr><tr><td>CITADEL</td><td>0.362</td><td>0.975</td><td>0.687</td><td>0.829</td><td>0.661</td><td>0.830</td><td>78.3</td><td>$\times  {25.7}$</td><td>10.8</td><td>3.95</td><td>520</td></tr><tr><td colspan="12">Models trained with further pre-training/hard-negative mining/distillation</td></tr><tr><td>coCondenser</td><td>0.382</td><td>0.984</td><td>0.674</td><td>0.820</td><td>0.684</td><td>0.839</td><td>26.0</td><td>$\times  {8.52}$</td><td>7.01</td><td>1.28</td><td>2015</td></tr><tr><td>SPLADE-v2</td><td>0.368</td><td>0.979</td><td>0.729</td><td>0.865</td><td>0.718</td><td>0.890</td><td>4.12</td><td>$\times  {1.35}$</td><td>7.13</td><td>✘</td><td>2710</td></tr><tr><td>ColBERT-v2</td><td>0.397</td><td>0.985</td><td>0.744</td><td>0.882</td><td>0.750</td><td>0.894</td><td>29.0</td><td>$\times  {9.51}$</td><td>10.9</td><td>122</td><td>3275</td></tr><tr><td>ColBERT-PLAID2</td><td>0.397</td><td>0.984</td><td>0.744</td><td>0.882</td><td>0.749</td><td>0.894</td><td>22.1</td><td>$\times  {7.25}$</td><td>10.9</td><td>55.0</td><td>370</td></tr><tr><td>CITADEL ${}^{ + }$</td><td>0.399</td><td>0.981</td><td>0.703</td><td>0.830</td><td>0.702</td><td>0.859</td><td>81.3</td><td>$\times  {26.7}$</td><td>10.8</td><td>3.21</td><td>635</td></tr></table>

<table><tbody><tr><td rowspan="2">模型</td><td colspan="2">MARCO开发集</td><td colspan="2">TREC DL19（文本检索会议2019年深度学习任务）</td><td colspan="2">TREC DL20（文本检索会议2020年深度学习任务）</td><td colspan="2">索引存储</td><td colspan="3">延迟（毫秒/查询）</td></tr><tr><td>前10名平均倒数排名（MRR@10）</td><td>前1000名召回率（R@1k）</td><td>前10名归一化折损累积增益（nDCG@10）</td><td>前1000名召回率（R@1k）</td><td>前10名归一化折损累积增益（nDCG@10）</td><td>前1000名召回率（R@1k）</td><td>磁盘（GB）</td><td>因子 ${}^{1}$</td><td>编码（GPU）</td><td>搜索（GPU）</td><td>搜索（CPU）</td></tr><tr><td colspan="12">仅使用BM25困难负样本训练的模型</td></tr><tr><td>二元独立模型（BM25）</td><td>0.188</td><td>0.858</td><td>0.506</td><td>0.739</td><td>0.488</td><td>0.733</td><td>0.67</td><td>$\times  \mathbf{{0.22}}$</td><td>✘</td><td>✘</td><td>40.1</td></tr><tr><td>密集段落检索器-128（DPR-128）</td><td>0.285</td><td>0.937</td><td>0.576</td><td>0.702</td><td>0.603</td><td>0.757</td><td>4.33</td><td>$\times  {1.42}$</td><td>7.09</td><td>0.63</td><td>430</td></tr><tr><td>密集段落检索器-768（DPR-768）</td><td>0.319</td><td>0.941</td><td>0.611</td><td>0.742</td><td>0.591</td><td>0.796</td><td>26.0</td><td>$\times  {8.52}$</td><td>7.01</td><td>1.28</td><td>2015</td></tr><tr><td>稀疏学习注意力驱动的高效检索器（SPLADE）</td><td>0.340</td><td>0.965</td><td>0.683</td><td>0.813</td><td>0.671</td><td>0.823</td><td>2.60</td><td>$\times  {0.85}$</td><td>7.13</td><td>✘</td><td>475</td></tr><tr><td>基于词元的上下文感知倒排学习（COIL-tok）</td><td>0.350</td><td>0.964</td><td>0.660</td><td>0.809</td><td>0.679</td><td>0.825</td><td>52.5</td><td>$\times  {17.2}$</td><td>10.7</td><td>46.8</td><td>1295</td></tr><tr><td>全量的上下文感知倒排学习（COIL-full）</td><td>0.353</td><td>0.967</td><td>0.704</td><td>0.835</td><td>0.688</td><td>0.841</td><td>78.5</td><td>$\times  {25.7}$</td><td>10.8</td><td>47.9</td><td>3258</td></tr><tr><td>基于上下文的双向表示检索器（ColBERT）</td><td>0.360</td><td>0.968</td><td>0.694</td><td>0.830</td><td>0.676</td><td>0.837</td><td>154</td><td>$\times  {50.5}$</td><td>10.9</td><td>178</td><td>-</td></tr><tr><td>上下文感知的密集与稀疏嵌入学习检索器（CITADEL）</td><td>0.362</td><td>0.975</td><td>0.687</td><td>0.829</td><td>0.661</td><td>0.830</td><td>78.3</td><td>$\times  {25.7}$</td><td>10.8</td><td>3.95</td><td>520</td></tr><tr><td colspan="12">经过进一步预训练/困难负样本挖掘/蒸馏训练的模型</td></tr><tr><td>协同冷凝器（coCondenser）</td><td>0.382</td><td>0.984</td><td>0.674</td><td>0.820</td><td>0.684</td><td>0.839</td><td>26.0</td><td>$\times  {8.52}$</td><td>7.01</td><td>1.28</td><td>2015</td></tr><tr><td>稀疏学习注意力驱动的高效检索器v2（SPLADE-v2）</td><td>0.368</td><td>0.979</td><td>0.729</td><td>0.865</td><td>0.718</td><td>0.890</td><td>4.12</td><td>$\times  {1.35}$</td><td>7.13</td><td>✘</td><td>2710</td></tr><tr><td>基于上下文的双向表示检索器v2（ColBERT-v2）</td><td>0.397</td><td>0.985</td><td>0.744</td><td>0.882</td><td>0.750</td><td>0.894</td><td>29.0</td><td>$\times  {9.51}$</td><td>10.9</td><td>122</td><td>3275</td></tr><tr><td>基于上下文的双向表示检索器-并行局部注意力索引（ColBERT-PLAID2）</td><td>0.397</td><td>0.984</td><td>0.744</td><td>0.882</td><td>0.749</td><td>0.894</td><td>22.1</td><td>$\times  {7.25}$</td><td>10.9</td><td>55.0</td><td>370</td></tr><tr><td>上下文感知的密集与稀疏嵌入学习检索器 ${}^{ + }$（CITADEL ${}^{ + }$）</td><td>0.399</td><td>0.981</td><td>0.703</td><td>0.830</td><td>0.702</td><td>0.859</td><td>81.3</td><td>$\times  {26.7}$</td><td>10.8</td><td>3.21</td><td>635</td></tr></tbody></table>

${}^{1}$ Factor: Ratio of index size to plain text size.

${}^{1}$ 因子：索引大小与纯文本大小的比率。

${}^{2}$ The PLAID implementation of ColBERT contains complex engineering schemes and low-level optimization such as centroid interaction and fast kernels.

${}^{2}$ ColBERT的PLAID实现包含复杂的工程方案和底层优化，如质心交互和快速内核。

<!-- Media -->

Table 1: In-domain evaluation on MS MARCO passages and TREC DL 2019/2020. CITADEL ${}^{ + }$ is trained with cross-encoder distillation and hard-negative mining. The red region means CITADEL/CITADEL ${}^{ + }$ is better than the method while the green region means that there’s no statistical significance $\left( {p > {0.05}}\right)$ . " $\times$ " means not implemented and "-" means not practical to evaluate on a single CPU. show there is no statistical significance. The inconsistency is probably due to that we use the training data from Tevatron (Gao et al., 2022) where each passage is paired with a title. Lassance and Clinchant (2023) points out that neural retrievers trained on such data will result in slightly higher scores on MS MARCO dev small while lower scores on TREC DL 2019 and 2020.

表1：在MS MARCO段落和TREC DL 2019/2020上的领域内评估。CITADEL ${}^{ + }$ 通过交叉编码器蒸馏和难负样本挖掘进行训练。红色区域表示CITADEL/CITADEL ${}^{ + }$ 优于该方法，而绿色区域表示没有统计学意义 $\left( {p > {0.05}}\right)$ 。“ $\times$ ”表示未实现，“-”表示在单个CPU上进行评估不切实际。显示没有统计学意义。这种不一致可能是因为我们使用了来自Tevatron（Gao等人，2022）的训练数据，其中每个段落都与一个标题配对。Lassance和Clinchant（2023）指出，在这类数据上训练的神经检索器在MS MARCO开发小数据集上的得分会略高，而在TREC DL 2019和2020上的得分会较低。

<!-- Media -->

<table><tr><td>Methods</td><td>AA</td><td>CF</td><td>DB</td><td>$\mathrm{{Fe}}$</td><td>FQ</td><td>HQ</td><td>NF</td><td>NQ</td><td>Qu</td><td>SF</td><td>SD</td><td>TC</td><td>T2</td><td>Avg.</td></tr><tr><td colspan="15">Models trained with only BM25 hard negatives</td></tr><tr><td>BM25</td><td>0.315</td><td>0.213</td><td>0.313</td><td>0.753</td><td>0.236</td><td>0.603</td><td>0.325</td><td>0.329</td><td>0.789</td><td>0.665</td><td>0.158</td><td>0.656</td><td>0.367</td><td>0.440</td></tr><tr><td>DPR-768</td><td>0.323</td><td>0.167</td><td>0.295</td><td>0.651</td><td>0.224</td><td>0.441</td><td>0.244</td><td>0.410</td><td>0.750</td><td>0.479</td><td>0.103</td><td>0.604</td><td>0.185</td><td>0.375</td></tr><tr><td>SPLADE</td><td>0.445</td><td>0.201</td><td>0.370</td><td>0.740</td><td>0.289</td><td>0.640</td><td>0.322</td><td>0.469</td><td>0.834</td><td>0.633</td><td>0.149</td><td>0.661</td><td>0.201</td><td>0.453</td></tr><tr><td>COIL-full</td><td>0.295</td><td>0.216</td><td>0.398</td><td>0.840</td><td>0.313</td><td>0.713</td><td>0.331</td><td>0.519</td><td>0.838</td><td>0.707</td><td>0.155</td><td>0.668</td><td>0.281</td><td>0.483</td></tr><tr><td>ColBERT</td><td>0.233</td><td>0.184</td><td>0.392</td><td>0.771</td><td>0.317</td><td>0.593</td><td>0.305</td><td>0.524</td><td>0.854</td><td>0.671</td><td>0.165</td><td>0.677</td><td>0.202</td><td>0.453</td></tr><tr><td>CITADEL</td><td>0.503</td><td>0.191</td><td>0.406</td><td>0.784</td><td>0.298</td><td>0.653</td><td>0.324</td><td>0.510</td><td>0.844</td><td>0.674</td><td>0.152</td><td>0.687</td><td>0.294</td><td>0.486</td></tr><tr><td colspan="15">Models with further pre-training/hard-negative mining/distillation</td></tr><tr><td>coCondenser</td><td>0.440</td><td>0.133</td><td>0.347</td><td>0.511</td><td>0.281</td><td>0.533</td><td>0.319</td><td>0.467</td><td>0.863</td><td>0.591</td><td>0.130</td><td>0.708</td><td>0.143</td><td>0.420</td></tr><tr><td>SPLADE-v2</td><td>0.479</td><td>0.235</td><td>0.435</td><td>0.786</td><td>0.336</td><td>0.684</td><td>0.334</td><td>0.521</td><td>0.838</td><td>0.693</td><td>0.158</td><td>0.710</td><td>0.272</td><td>0.499</td></tr><tr><td>ColBERT-v2</td><td>0.463</td><td>0.176</td><td>0.446</td><td>0.785</td><td>0.356</td><td>0.667</td><td>0.338</td><td>0.562</td><td>0.854</td><td>0.693</td><td>0.165</td><td>0.738</td><td>0.263</td><td>0.500</td></tr><tr><td>CITADEL ${}^{ + }$</td><td>0.490</td><td>0.181</td><td>0.420</td><td>0.747</td><td>0.332</td><td>0.652</td><td>0.337</td><td>0.539</td><td>0.852</td><td>0.695</td><td>0.147</td><td>0.680</td><td>0.340</td><td>0.493</td></tr><tr><td>CITADEL ${}^{ + }$ (w/o reg.)</td><td>0.511</td><td>0.182</td><td>0.422</td><td>0.765</td><td>0.330</td><td>0.664</td><td>0.337</td><td>0.540</td><td>0.853</td><td>0.690</td><td>0.159</td><td>0.715</td><td>0.342</td><td>0.501</td></tr></table>

<table><tbody><tr><td>方法</td><td>AA</td><td>CF</td><td>DB</td><td>$\mathrm{{Fe}}$</td><td>FQ</td><td>HQ</td><td>NF</td><td>NQ</td><td>Qu</td><td>SF</td><td>SD</td><td>TC</td><td>T2</td><td>平均值</td></tr><tr><td colspan="15">仅使用BM25硬负样本训练的模型</td></tr><tr><td>BM25</td><td>0.315</td><td>0.213</td><td>0.313</td><td>0.753</td><td>0.236</td><td>0.603</td><td>0.325</td><td>0.329</td><td>0.789</td><td>0.665</td><td>0.158</td><td>0.656</td><td>0.367</td><td>0.440</td></tr><tr><td>DPR - 768</td><td>0.323</td><td>0.167</td><td>0.295</td><td>0.651</td><td>0.224</td><td>0.441</td><td>0.244</td><td>0.410</td><td>0.750</td><td>0.479</td><td>0.103</td><td>0.604</td><td>0.185</td><td>0.375</td></tr><tr><td>SPLADE</td><td>0.445</td><td>0.201</td><td>0.370</td><td>0.740</td><td>0.289</td><td>0.640</td><td>0.322</td><td>0.469</td><td>0.834</td><td>0.633</td><td>0.149</td><td>0.661</td><td>0.201</td><td>0.453</td></tr><tr><td>COIL - 全量</td><td>0.295</td><td>0.216</td><td>0.398</td><td>0.840</td><td>0.313</td><td>0.713</td><td>0.331</td><td>0.519</td><td>0.838</td><td>0.707</td><td>0.155</td><td>0.668</td><td>0.281</td><td>0.483</td></tr><tr><td>ColBERT</td><td>0.233</td><td>0.184</td><td>0.392</td><td>0.771</td><td>0.317</td><td>0.593</td><td>0.305</td><td>0.524</td><td>0.854</td><td>0.671</td><td>0.165</td><td>0.677</td><td>0.202</td><td>0.453</td></tr><tr><td>CITADEL</td><td>0.503</td><td>0.191</td><td>0.406</td><td>0.784</td><td>0.298</td><td>0.653</td><td>0.324</td><td>0.510</td><td>0.844</td><td>0.674</td><td>0.152</td><td>0.687</td><td>0.294</td><td>0.486</td></tr><tr><td colspan="15">经过进一步预训练/硬负样本挖掘/蒸馏的模型</td></tr><tr><td>coCondenser</td><td>0.440</td><td>0.133</td><td>0.347</td><td>0.511</td><td>0.281</td><td>0.533</td><td>0.319</td><td>0.467</td><td>0.863</td><td>0.591</td><td>0.130</td><td>0.708</td><td>0.143</td><td>0.420</td></tr><tr><td>SPLADE - v2</td><td>0.479</td><td>0.235</td><td>0.435</td><td>0.786</td><td>0.336</td><td>0.684</td><td>0.334</td><td>0.521</td><td>0.838</td><td>0.693</td><td>0.158</td><td>0.710</td><td>0.272</td><td>0.499</td></tr><tr><td>ColBERT - v2</td><td>0.463</td><td>0.176</td><td>0.446</td><td>0.785</td><td>0.356</td><td>0.667</td><td>0.338</td><td>0.562</td><td>0.854</td><td>0.693</td><td>0.165</td><td>0.738</td><td>0.263</td><td>0.500</td></tr><tr><td>CITADEL ${}^{ + }$</td><td>0.490</td><td>0.181</td><td>0.420</td><td>0.747</td><td>0.332</td><td>0.652</td><td>0.337</td><td>0.539</td><td>0.852</td><td>0.695</td><td>0.147</td><td>0.680</td><td>0.340</td><td>0.493</td></tr><tr><td>CITADEL ${}^{ + }$（无正则化）</td><td>0.511</td><td>0.182</td><td>0.422</td><td>0.765</td><td>0.330</td><td>0.664</td><td>0.337</td><td>0.540</td><td>0.853</td><td>0.690</td><td>0.159</td><td>0.715</td><td>0.342</td><td>0.501</td></tr></tbody></table>

Table 2: Out-of-domain evaluation on BEIR benchmark. nDCG@10 score is reported. Dataset Legend (Chen et al., 2022): TC=TREC-COVID, NF=NFCorpus, NQ=NaturalQuestions, HQ=HotpotQA, FQ=FiQA, AA=ArguAna, T2=Touché-2020 (v2), Qu=Quora, DB=DBPedia, SD=SCIDOCS, Fe=FEVER, CF=Climate-FEVER, SF=SciFact.

表2：BEIR基准测试的域外评估。报告了nDCG@10分数。数据集说明（Chen等人，2022）：TC=TREC-COVID（新冠疫情信息检索评测），NF=NFCorpus（国家功能语料库），NQ=NaturalQuestions（自然问题），HQ=HotpotQA（火锅问答），FQ=FiQA（金融问答），AA=ArguAna（论证分析），T2=Touché-2020（v2）（Touché 2020版本2），Qu=Quora（Quora问答平台），DB=DBPedia（DBPedia知识图谱），SD=SCIDOCS（科学文档），Fe=FEVER（事实验证），CF=Climate-FEVER（气候事实验证），SF=SciFact（科学事实）。

<!-- Media -->

### 4.2 BEIR: Out-of-Domain Evaluation

### 4.2 BEIR：域外评估

We evaluate on BEIR benchmark (Thakur et al., 2021) which consists of a diverse set of 18 retrieval tasks across 9 domains. We evaluate on 13 datasets following previous works (Santhanam et al., 2022b; Formal et al., 2021a). Table 2 shows the zero-shot evaluation results on BEIR. Without any pre-training or distillation, CITADEL manages to outperform all baselines in their sub-categories in terms of the average score. Compared with the distilled/pre-trained models, ${\mathrm{{CITADEL}}}^{ + }$ still manages to achieve comparable performance. Interestingly, we find that if no regularization like load balancing and L1 is applied during training, ${\text{CITADEL}}^{ + }$ can reach a much higher average score that even outperforms ColBERT-v2. Our conjecture is that the regularization reduces the number of token interactions and the importance of such token interaction is learned from training data. It is hence not surprising that the more aggressively we prune token interaction, the more likely that it would hurt out-of-domain accuracy that's not covered by the training data.

我们在BEIR基准测试（Thakur等人，2021）上进行评估，该基准测试包含9个领域的18个不同的检索任务。我们按照先前的研究（Santhanam等人，2022b；Formal等人，2021a）在13个数据集上进行评估。表2展示了BEIR上的零样本评估结果。在没有任何预训练或蒸馏的情况下，CITADEL在子类别中的平均得分方面成功超越了所有基线模型。与经过蒸馏/预训练的模型相比，${\mathrm{{CITADEL}}}^{ + }$仍然能够取得相当的性能。有趣的是，我们发现如果在训练期间不应用诸如负载均衡和L1正则化等方法，${\text{CITADEL}}^{ + }$可以达到更高的平均得分，甚至超过了ColBERT - v2。我们推测，正则化减少了词元交互的数量，并且这种词元交互的重要性是从训练数据中学习到的。因此，我们对词元交互进行越激进的剪枝，就越有可能损害训练数据未覆盖的域外准确率，这也就不足为奇了。

<!-- Media -->

<table><tr><td>Models</td><td>MRR@10</td><td>$\# \mathrm{{DP}} \times  {10}^{6}$</td></tr><tr><td>ColBERT</td><td>0.360</td><td>4213</td></tr><tr><td>COIL-full</td><td>0.353</td><td>45.6</td></tr><tr><td>CITADEL</td><td>0.362</td><td>10.5</td></tr><tr><td>DPR-128</td><td>0.285</td><td>8.84</td></tr></table>

<table><tbody><tr><td>模型</td><td>前10名平均倒数排名（MRR@10）</td><td>$\# \mathrm{{DP}} \times  {10}^{6}$</td></tr><tr><td>ColBERT（科尔伯特）</td><td>0.360</td><td>4213</td></tr><tr><td>COIL-full（全线圈）</td><td>0.353</td><td>45.6</td></tr><tr><td>CITADEL（堡垒）</td><td>0.362</td><td>10.5</td></tr><tr><td>DPR - 128（密集段落检索器 - 128）</td><td>0.285</td><td>8.84</td></tr></tbody></table>

Table 3: Maximal number of dot products per query on MS MARCO passages. # DP: number of dot products.

表3：在MS MARCO段落上每个查询的最大点积数量。# DP：点积数量。

<!-- figureText: 0.05 Indices (sorted in descending order of storage fraction) 0.04 0.01 0.00 -->

<img src="https://cdn.noedgeai.com/0195a46c-451c-7169-bf50-83435cd534a9_6.jpg?x=205&y=571&w=562&h=413&r=0"/>

Figure 3: Normalized disk storage over token indices in an inverted list of CITADEL and COIL.

图3：CITADEL和COIL倒排列表中按Token（词元）索引归一化后的磁盘存储情况。

<!-- Media -->

## 5 Performance Analysis

## 5 性能分析

### 5.1 Number of Token Interactions

### 5.1 Token（词元）交互数量

The actual latency is often impacted by engineering details and therefore FLOPS is often considered for comparing efficiency agnostic to the actual implementation. In our case, however, FLOPS is impacted by the vector dimension in the nearest neighbour search which is different across models. Therefore, we only compare the maximal number of dot products needed as a proxy for token interaction per query during retrieval as shown in Table 3. The number of dot products per query in CITADEL with pruning threshold $\tau  = {0.9}$ is comparable to DPR-128 and much lower than ColBERT and COIL, which is consistent with the latency numbers in Table 1. The reason is that CITADEL has a balanced inverted index credited to the ${\ell }_{1}$ regularization and the load balancing loss as shown in Figure 3. By applying the load balancing loss on the router prediction, CITADEL yields a more balanced and even index distribution where its largest index fraction is $8 \times$ smaller than COIL’s as shown in Figure 3. We also provide a detailed latency breakdown in Appendix A.4.

实际延迟通常会受到工程细节的影响，因此在比较效率时，通常会考虑每秒浮点运算次数（FLOPS），而不考虑实际实现方式。然而，在我们的情况下，FLOPS会受到最近邻搜索中向量维度的影响，而不同模型的向量维度是不同的。因此，如表3所示，我们仅比较检索期间每个查询所需的最大点积数量，以此作为Token（词元）交互的代理。CITADEL在剪枝阈值为$\tau  = {0.9}$时每个查询的点积数量与DPR - 128相当，且远低于ColBERT和COIL，这与表1中的延迟数据一致。原因在于，如图3所示，由于${\ell }_{1}$正则化和负载均衡损失，CITADEL具有平衡的倒排索引。通过对路由预测应用负载均衡损失，CITADEL产生了更平衡、更均匀的索引分布，如图3所示，其最大索引比例比COIL小$8 \times$。我们还在附录A.4中提供了详细的延迟分解。

<!-- Media -->

<!-- figureText: 78.3GB 109.9GB 147GB $\tau  = {0.7}$ t=0.5 ColBERT (178ms, 154GB) 10.0 12.5 15.0 17.5 20.0 GPU Latency (ms/query) $\tau  = {1.1}$ 0.36 30.3GB $\tau  = {1.3}$ 0.32 17.1GB t=1.5 0.30 0.0 5.0 7.5 -->

<img src="https://cdn.noedgeai.com/0195a46c-451c-7169-bf50-83435cd534a9_6.jpg?x=849&y=197&w=584&h=439&r=0"/>

Figure 4: Latency-memory-accuracy tradeoff on MS MARCO passages using post-hoc pruning. $\tau$ is the pruning threshold.

图4：使用事后剪枝在MS MARCO段落上的延迟 - 内存 - 准确率权衡。$\tau$是剪枝阈值。

<table><tr><td>Condition</td><td>MRR@10</td><td>Storage (GB)</td><td>Latency (ms)</td></tr><tr><td>original</td><td>0.362</td><td>78.3</td><td>3.95</td></tr><tr><td>nbits=2</td><td>0.361</td><td>13.3</td><td>0.90</td></tr><tr><td>nbits=1</td><td>0.356</td><td>11.0</td><td>0.92</td></tr></table>

<table><tbody><tr><td>条件</td><td>前10召回率（MRR@10）</td><td>存储空间（GB）</td><td>延迟（ms）</td></tr><tr><td>原始的</td><td>0.362</td><td>78.3</td><td>3.95</td></tr><tr><td>位数=2</td><td>0.361</td><td>13.3</td><td>0.90</td></tr><tr><td>位数=1</td><td>0.356</td><td>11.0</td><td>0.92</td></tr></tbody></table>

Table 4: Product Quantization. Pruning threshold is set to 0.9 . nbits: ratio of total bytes to the vector dimension.

表4：乘积量化。剪枝阈值设置为0.9。nbits：总字节数与向量维度的比率。

<!-- Media -->

### 5.2 Latency-Memory-Accuracy Tradeoff

### 5.2 延迟 - 内存 - 准确率权衡

Figure 4 shows the tradeoff among latency, memory, and MRR@10 on MS MARCO passages with post-hoc pruning. We try the pruning thresholds $\left\lbrack  {{0.5},{0.7},{0.9},{1.1},{1.3},{1.5}}\right\rbrack$ . We could see that the MRR@10 score barely decreases when we increase the threshold to from 0.5 to 1.1 , but the latency decreases by a large margin, from about 18 $\mathrm{{ms}}/$ query to ${0.61}\mathrm{\;{ms}}/$ query. The sweet spots are around (0.359 MRR@10, 49.3GB, 0.61 ms/query) and (0.362 MRR@10,78.5GB,3.95 ms/query). This simple pruning strategy is extremely effective and readers can see in Section 6 that it also yields interpretable document representations.

图4展示了在经过事后剪枝的MS MARCO段落上，延迟、内存和MRR@10之间的权衡关系。我们尝试了剪枝阈值$\left\lbrack  {{0.5},{0.7},{0.9},{1.1},{1.3},{1.5}}\right\rbrack$。可以看到，当我们将阈值从0.5提高到1.1时，MRR@10得分几乎没有下降，但延迟大幅降低，从约18$\mathrm{{ms}}/$/查询降至${0.61}\mathrm{\;{ms}}/$/查询。最佳平衡点大约在（MRR@10为0.359，49.3GB，0.61毫秒/查询）和（MRR@10为0.362，78.5GB，3.95毫秒/查询）处。这种简单的剪枝策略非常有效，读者可以在第6节中看到，它还能产生可解释的文档表示。

### 5.3 Combination with Product Quantization

### 5.3 与乘积量化相结合

We could further reduce the latency and storage with product quantization (Jégou et al., 2011) (PQ) as shown in Table 4. For nbits=2, we divide the vectors into sets of 4-dimensional sub-vectors and use 256 centroids for clustering the sub-vectors, while for nbits $= 1$ we set the sub-vector dim to 8 and the same for the rest. With only 2 bits per dimension, the MRR@10 score on MS MARCO Dev only drops 4% but the storage is reduced by ${83}\%$ and latency is reduced by ${76}\%$ .

如表4所示，我们可以通过乘积量化（Jégou等人，2011）（PQ）进一步降低延迟和存储需求。对于nbits = 2，我们将向量划分为4维子向量集，并使用256个质心对子向量进行聚类，而对于nbits$= 1$，我们将子向量维度设置为8，其余情况相同。每个维度仅使用2位时，MS MARCO Dev上的MRR@10得分仅下降4%，但存储需求减少了${83}\%$，延迟降低了${76}\%$。

<!-- Media -->

<table><tr><td>Threshold $\tau$</td><td>Sample documents from MS MARCO Passages</td></tr><tr><td>0.0</td><td>All medications have side effects, including drugs to treat arrhythmias. Most of the side effects aren't serious and disappear when the dose is changed or the medication is stopped. But some side effects are very serious. That's why some children are admitted to the hospital to begin the medication. Medications for Arrhythmia</td></tr><tr><td>0.9</td><td>All medications have side effects, including drugs to treat arrhythmias. Most of the side effects aren't serious and disappear when the dose is changed or the medication is stopped. But some side effects are very serious. That's why some children are admitted to the hospital to begin the medication. Medications for Arrhythmia</td></tr><tr><td>1.3</td><td>All medications have side effects, including drugs to treat arrhythmias. Most of the side effects aren't serious and disappear when the dose is changed or the medication is stopped. But some side effects are very serious. That's why some children are admitted to the hospital to begin the medication. Medications for $\textbf{Arrhythmia}$</td></tr></table>

<table><tbody><tr><td>阈值 $\tau$</td><td>来自MS MARCO段落的示例文档</td></tr><tr><td>0.0</td><td>所有药物都有副作用，包括治疗心律失常（arrhythmias）的药物。大多数副作用并不严重，在改变剂量或停药后会消失。但有些副作用非常严重。这就是为什么有些儿童需要住院开始用药。心律失常治疗药物</td></tr><tr><td>0.9</td><td>所有药物都有副作用，包括治疗心律失常（arrhythmias）的药物。大多数副作用并不严重，在改变剂量或停药后会消失。但有些副作用非常严重。这就是为什么有些儿童需要住院开始用药。心律失常治疗药物</td></tr><tr><td>1.3</td><td>所有药物都有副作用，包括治疗心律失常（arrhythmias）的药物。大多数副作用并不严重，在改变剂量或停药后会消失。但有些副作用非常严重。这就是为什么有些儿童需要住院开始用药。$\textbf{Arrhythmia}$治疗药物</td></tr></tbody></table>

Figure 5: Tokens in grey (pruned) have zero activated keys while bold tokens have at least one activated key. We leave out the expanded terms and routing weights due to space limit.

图5：灰色（已修剪）的Token（标记）激活键为零，而加粗的Token（标记）至少有一个激活键。由于篇幅限制，我们省略了扩展项和路由权重。

<table><tr><td>Models</td><td>Dev</td><td>DL19</td><td>Latency (ms)</td></tr><tr><td>COIL-full</td><td>0.353</td><td>0.704</td><td>47.9</td></tr><tr><td>COIL-tok</td><td>0.350</td><td>0.660</td><td>46.8</td></tr><tr><td>CITADEL</td><td>0.362</td><td>0.687</td><td>3.95</td></tr><tr><td>CITADEL-tok</td><td>0.360</td><td>0.665</td><td>1.64</td></tr></table>

<table><tbody><tr><td>模型</td><td>开发</td><td>DL19</td><td>延迟（毫秒）</td></tr><tr><td>COIL全量版</td><td>0.353</td><td>0.704</td><td>47.9</td></tr><tr><td>COIL词元版</td><td>0.350</td><td>0.660</td><td>46.8</td></tr><tr><td>堡垒（CITADEL）</td><td>0.362</td><td>0.687</td><td>3.95</td></tr><tr><td>堡垒词元版（CITADEL - tok）</td><td>0.360</td><td>0.665</td><td>1.64</td></tr></tbody></table>

Table 5: [CLS] ablation on MS MARCO passage. Dev: MRR@10.DL19: nDCG@10.

表5：在MS MARCO段落上对[CLS]进行消融实验。开发集：MRR@10。DL19：nDCG@10。

<!-- Media -->

## 6 Token Routing Analysis of CITADEL

## 6 CITADEL的Token路由分析

Qualitative Analysis. We visualize CITADEL representations and the effect of posthoc pruning in Figure 5. By increasing the pruning threshold, more keywords are pruned and finally leaving the most central word "arrhythmia" activated. We provide another example in Figure 6. We can see that a lot of redundant words that do not contribute to the final semantics are deactivated, meaning all their routing weights are 0 . For the activated tokens, we could see the routed keys are contextualized as many of them are related to emoji which is the theme of the document.

定性分析。我们在图5中可视化了CITADEL的表示以及事后剪枝的效果。通过提高剪枝阈值，更多的关键词被剪枝，最终只有最核心的词“心律失常（arrhythmia）”被激活。我们在图6中提供了另一个示例。我们可以看到，许多对最终语义没有贡献的冗余词被停用，这意味着它们的所有路由权重都为0。对于被激活的Token，我们可以看到路由的键是上下文相关的，因为其中许多与文档主题表情符号有关。

Quantitative Analysis. We analyze CITADEL's token distribution over the number of activated routing keys for the whole corpus as shown in Figure 7. With ${\ell }_{1}$ loss,around 50 tokens per passage are deactivated (i.e., all the routing weights of these 50 tokens are 0 ). As the pruning threshold increases, more tokens are deactivated, yielding a sparse representation for interpreting CITADEL's behaviours.

定量分析。如图7所示，我们分析了整个语料库中CITADEL的Token在激活的路由键数量上的分布。在${\ell }_{1}$损失下，每个段落大约有50个Token被停用（即，这50个Token的所有路由权重都为0）。随着剪枝阈值的增加，更多的Token被停用，从而产生了用于解释CITADEL行为的稀疏表示。

## 7 Ablation Studies

## 7 消融实验研究

Impact of [CLS] Table 5 shows the influence of removing the [CLS] vector for CITADEL on MS MARCO passage. Although removing [CLS] improves the latency by a large margin, the in-domain effectiveness is also adversely affected, especially on TREC DL 2019. Nevertheless, CITADEL-tok (w/o [CLS]) still outperforms its counterpart COIL-tok in both precision and latency.

[CLS]的影响 表5显示了在MS MARCO段落上移除CITADEL的[CLS]向量的影响。虽然移除[CLS]在很大程度上提高了延迟，但领域内有效性也受到了不利影响，尤其是在TREC DL 2019上。尽管如此，CITADEL - tok（无[CLS]）在精度和延迟方面仍然优于其对应模型COIL - tok。

<!-- Media -->

<table><tr><td>#Keys</td><td>MRR@10</td><td>Storage (GB)</td><td>Latency (ms)</td></tr><tr><td>1</td><td>0.347</td><td>53.6</td><td>1.28</td></tr><tr><td>3</td><td>0.360</td><td>136</td><td>14.7</td></tr><tr><td>5</td><td>0.364</td><td>185</td><td>18.6</td></tr><tr><td>7</td><td>0.370</td><td>196</td><td>20.4</td></tr><tr><td>9</td><td>0.367</td><td>221</td><td>19.6</td></tr></table>

<table><tbody><tr><td>#键</td><td>前10项平均倒数排名（MRR@10）</td><td>存储容量（GB）</td><td>延迟时间（ms）</td></tr><tr><td>1</td><td>0.347</td><td>53.6</td><td>1.28</td></tr><tr><td>3</td><td>0.360</td><td>136</td><td>14.7</td></tr><tr><td>5</td><td>0.364</td><td>185</td><td>18.6</td></tr><tr><td>7</td><td>0.370</td><td>196</td><td>20.4</td></tr><tr><td>9</td><td>0.367</td><td>221</td><td>19.6</td></tr></tbody></table>

Table 6: Number of routing keys for documents during training. No post-hoc pruning is applied.

表6：训练期间文档的路由键数量。未应用事后剪枝。

<!-- Media -->

Number of Routed Experts. Table 6 shows the influence of changing the maximum number of keys that each document token can be routed to during training and inference on MS MARCO passage. As the number of routing keys increases, the index storage also increases rapidly but so does the MRR@10 score which plateaus after reaching 7 keys. The latency does not increase as much after 3 routing keys due to the load balancing loss.

路由专家数量。表6展示了在MS MARCO段落的训练和推理过程中，改变每个文档Token（Token）可路由到的最大键数量所产生的影响。随着路由键数量的增加，索引存储量也会迅速增加，但MRR@10分数同样会提高，在达到7个键后趋于平稳。由于负载均衡损失，在有3个路由键之后，延迟的增加幅度不会太大。

## 8 Related Works

## 8 相关工作

Dense Retrieval. Supported by multiple approximate nearest neighbour search libraries (Johnson et al., 2021; Guo et al., 2020), dense retrieval (Karpukhin et al., 2020) gained much popularity due to its efficiency and flexibility. To improve effectiveness, techniques such as hard negative mining (Xiong et al., 2021; Zhan et al., 2021) and knowledge distillation (Lin et al., 2021b; Hof-stätter et al., 2021) are often deployed. Recently, retrieval-oriented pre-training(Gao et al., 2021b; Lu et al., 2021; Gao and Callan, 2021; Izacard et al., 2022; Gao and Callan, 2022) also draws much attention as they could substantially improve the fine-tuning performance of downstream tasks.

密集检索。在多个近似最近邻搜索库（Johnson等人，2021年；Guo等人，2020年）的支持下，密集检索（Karpukhin等人，2020年）因其效率和灵活性而广受欢迎。为了提高检索效果，人们经常采用诸如难负样本挖掘（Xiong等人，2021年；Zhan等人，2021年）和知识蒸馏（Lin等人，2021b；Hof - stätter等人，2021年）等技术。最近，面向检索的预训练（Gao等人，2021b；Lu等人，2021年；Gao和Callan，2021年；Izacard等人，2022年；Gao和Callan，2022年）也备受关注，因为它们可以显著提高下游任务的微调性能。

<!-- Media -->

Sample Document from MS MARCO Passages imag be a little hard to $\mathbf{{remember}}$ (remember, forget) all of 25(barney, many, 25, 1625)codes(code, codes), which created (invented, stumbled, created, swirled) dictionary (dictionary, spiked). You can find the facebook (facebook) $\textbf{emoticons}$ (emoticon, emoji, combinations, asher, grins)list for chats(chat, chatting) and status (winked, logo, status, glided) in the table above. list (list, kingsley, include)ing includes all the new ones which have been recently added, such as the $\textbf{penguin}$ (penguin, mccann, uefa) or shark(shark, chao) $\mathbf{{emoticon}}$ (emoticon, emoji, assassins). Facebook(facebook) Emoticons(emoticon, emoji, spp, programs)

来自MS MARCO段落的示例文档 要$\mathbf{{remember}}$（记住，忘记）25（巴尼，许多，25，1625）个代码（代码，代码集）可能有点困难，这些代码创建（发明，偶然发现，创建，旋转）了字典（字典，尖刺状的）。你可以在上面的表格中找到脸书（Facebook）$\textbf{emoticons}$（表情符号，表情，组合，阿舍，咧嘴笑）聊天（聊天，聊天行为）和状态（眨眼，标志，状态，滑行）列表。列表（列表，金斯利，包含）包含了所有最近添加的新表情，比如$\textbf{penguin}$（企鹅，麦卡恩，欧洲足联）或鲨鱼（鲨鱼，赵）$\mathbf{{emoticon}}$（表情符号，表情，刺客）。脸书（Facebook）表情符号（表情符号，表情，spp，程序）

Figure 6: Token routing analysis of CITADEL. Grey tokens are deactivated, while bold tokens are routed to at least one activated key (in blue).

图6：CITADEL的Token（Token）路由分析。灰色Token（Token）为停用状态，而粗体Token（Token）至少路由到一个激活键（蓝色）。

<!-- Media -->

Sparse Retrieval. Traditional sparse retrieval systems such as BM25 (Robertson and Zaragoza, 2009) and tf-idf (Salton and Buckley, 1988) represent the documents as a bag of words with static term weights. Recently, many works leverage pre-trained language models to learn contextualized term importance (Bai et al., 2020; Mallia et al., 2021; Formal et al., 2021b; Lin and Ma, 2021). These models could utilize existing inverted index libraries such as Pyserini (Lin et al., 2021a) to perform efficient sparse retrieval or even hybrid with dense retrieval (Hofstätter et al., 2022; Shen et al., 2022; Lin and Lin, 2022; Zhang et al., 2023).

稀疏检索。传统的稀疏检索系统，如BM25（Robertson和Zaragoza，2009年）和tf - idf（Salton和Buckley，1988年），将文档表示为具有静态词项权重的词袋。最近，许多研究利用预训练语言模型来学习上下文相关的词项重要性（Bai等人，2020年；Mallia等人，2021年；Formal等人，2021b；Lin和Ma，2021年）。这些模型可以利用现有的倒排索引库，如Pyserini（Lin等人，2021a）来进行高效的稀疏检索，甚至可以与密集检索相结合（Hof - stätter等人，2022年；Shen等人，2022年；Lin和Lin，2022年；Zhang等人，2023年）。

Multi-Vector Retrieval. ColBERT (Khattab and Zaharia, 2020; Santhanam et al., 2022b,a; Hof-stätter et al., 2022) probably has the most optimized library in multi-vector retrieval. COIL (Gao et al., 2021a) accelerates retrieval by combining with exact match and inverted vector search. ME-BERT (Luan et al., 2021) and MVR (Zhang et al., 2022) propose to use a fixed number of token embeddings for late interaction (e.g., top-k positions or special tokens). Concurrently to this work, ALIGNER (Qian et al., 2022) proposes to frame multi-vector retrieval as a sparse alignment problem between query tokens and document tokens using entropy-regularized linear programming. Our 110M model achieves higher in-domain and out-of-domain accuracy than their large variants.

多向量检索。ColBERT（Khattab和Zaharia，2020年；Santhanam等人，2022b,a；Hof - stätter等人，2022年）可能是多向量检索中优化程度最高的库。COIL（Gao等人，2021a）通过结合精确匹配和倒排向量搜索来加速检索。ME - BERT（Luan等人，2021年）和MVR（Zhang等人，2022年）提议使用固定数量的Token（Token）嵌入进行后期交互（例如，前k个位置或特殊Token（Token））。与本研究同期，ALIGNER（Qian等人，2022年）提议将多向量检索构建为一个使用熵正则化线性规划的查询Token（Token）和文档Token（Token）之间的稀疏对齐问题。我们的1.1亿参数模型在域内和域外的准确率都高于他们的大型变体模型。

## 9 Conclusion

## 9 结论

This paper proposes a novel multi-vector retrieval method that achieves state-of-the-art performance on several benchmark datasets while being ${40} \times$ faster than ColBERT-v2 and ${17} \times$ faster than the most efficient multi-vector retrieval library to date, PLAID, on GPUs. By jointly optimizing for the token index size and load balancing, our new dynamic lexical routing scheme greatly reduces the redundancy in the all-to-all token interaction of ColBERT while bridging the word-mismatch problem in COIL. Experiments on both in-domain and out-of-domain datasets demonstrate the effectiveness and efficiency of our model.

本文提出了一种新颖的多向量检索方法，该方法在多个基准数据集上达到了最先进的性能，同时在GPU上比ColBERT - v2快${40} \times$，比迄今为止最高效的多向量检索库PLAID快${17} \times$。通过联合优化Token索引大小和负载平衡，我们新的动态词法路由方案大大减少了ColBERT全对全Token交互中的冗余，同时解决了COIL中的词不匹配问题。在领域内和领域外数据集上的实验证明了我们模型的有效性和高效性。

<!-- Media -->

<!-- figureText: Number of activated keys per token -->

<img src="https://cdn.noedgeai.com/0195a46c-451c-7169-bf50-83435cd534a9_8.jpg?x=848&y=196&w=585&h=441&r=0"/>

Figure 7: Token distribution over number of activated experts per passage. $\tau$ is the pruning threshold.

图7：每个段落中激活专家数量的Token分布。$\tau$是剪枝阈值。

<!-- Media -->

## 10 Limitations

## 10 局限性

The limitation of CITADEL mainly shows in two aspects. First, at the beginning of training, the model needs to route each token vector to multiple activated keys for token interaction, which increases the computation cost compared to COIL and ColBERT. This results in slower training speed but it gets better when training approaches the end as more tokens are pruned by the ${\ell }_{1}$ regularization. Another drawback lies in the implementation of CITADEL, or more generally speaking, most multi-vector retrieval methods. The token-level retrieval and aggregation make them not compatible with established search libraries such as FAISS or Py-serini. Moreover, for time and space efficiency, multi-vector retrieval also requires more engineering efforts and low-level optimization. Recently, XTR (Lee et al., 2023) provides a solution that constrains the document-level retrieval to be consistent with the token-level retrieval during training, which can be used for streamlining CITADEL.

CITADEL的局限性主要体现在两个方面。首先，在训练开始时，模型需要将每个Token向量路由到多个激活的键以进行Token交互，与COIL和ColBERT相比，这增加了计算成本。这导致训练速度较慢，但随着训练接近尾声，由于更多的Token被${\ell }_{1}$正则化剪枝，情况会有所改善。另一个缺点在于CITADEL的实现，或者更普遍地说，大多数多向量检索方法的实现。Token级别的检索和聚合使得它们与现有的搜索库（如FAISS或Py - serini）不兼容。此外，为了提高时间和空间效率，多向量检索还需要更多的工程工作和底层优化。最近，XTR（Lee等人，2023）提供了一种解决方案，在训练期间将文档级检索约束为与Token级检索一致，可用于简化CITADEL。

## 11 Acknowledgement

## 11 致谢

We would like to thank Jun Yan and Zheng Lian for the helpful discussions on CITADEL.

我们要感谢闫军和连政就CITADEL进行的有益讨论。

## References

## 参考文献

Yang Bai, Xiaoguang Li, Gang Wang, Chaoliang Zhang, Lifeng Shang, Jun Xu, Zhaowei Wang, Fangshan Wang, and Qun Liu. 2020. Sparterm: Learning term-based sparse representation for fast text retrieval. ArXiv, abs/2010.00768.

白杨、李晓光、王刚、张朝亮、尚立峰、徐军、王兆伟、王房山和刘群。2020年。Sparterm：学习基于词项的稀疏表示以实现快速文本检索。arXiv，abs/2010.00768。

Alexander Bondarenko, Maik Fröbe, Meriem Be-loucif, Lukas Gienapp, Yamen Ajjour, Alexander Panchenko, Chris Biemann, Benno Stein, Henning Wachsmuth, Martin Potthast, and Matthias Hagen. 2020. Overview of touché 2020: Argument retrieval: Extended abstract. In Experimental IR Meets Mul-tilinguality, Multimodality, and Interaction: 11th International Conference of the CLEF Association, CLEF 2020, Thessaloniki, Greece, September 22-25, 2020, Proceedings, page 384-395, Berlin, Heidelberg. Springer-Verlag.

亚历山大·邦达连科、迈克·弗罗贝、梅里姆·贝 - 卢西夫、卢卡斯·吉纳普、亚门·阿乔尔、亚历山大·潘琴科、克里斯·比曼、本诺·斯坦、亨宁·瓦赫斯穆特、马丁·波塔斯塔和马蒂亚斯·哈根。2020年。2020年Touché概述：论点检索：扩展摘要。见《实验信息检索与多语言性、多模态和交互：CLEF协会第11届国际会议，CLEF 2020，希腊塞萨洛尼基，2020年9月22 - 25日，会议录》，第384 - 395页，柏林，海德堡。施普林格出版社。

Vera Boteva, Demian Gholipour, Artem Sokolov, and Stefan Riezler. 2016. A full-text learning to rank dataset for medical information retrieval. In ${Ad}$ - vances in Information Retrieval, pages 716-722, Cham. Springer International Publishing.

维拉·博特瓦、德米安·戈利普尔、阿尔乔姆·索科洛夫和斯特凡·里兹勒。2016年。用于医学信息检索的全文学习排序数据集。见《信息检索进展》，第716 - 722页，尚贝里。施普林格国际出版公司。

Danqi Chen, Adam Fisch, Jason Weston, and Antoine Bordes. 2017. Reading Wikipedia to answer open-domain questions. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1870-1879, Vancouver, Canada. Association for Computational Linguistics.

陈丹琦、亚当·菲施、杰森·韦斯顿和安托万·博尔德斯。2017年。阅读维基百科以回答开放领域问题。见《计算语言学协会第55届年会会议录（第1卷：长论文）》，第1870 - 1879页，加拿大温哥华。计算语言学协会。

Xilun Chen, Kushal Lakhotia, Barlas Oğuz, Anchit Gupta, Patrick Lewis, Stan Peshterliev, Yashar Mehdad, Sonal Gupta, and Wen-tau Yih. 2022. Salient phrase aware dense retrieval: Can a dense retriever imitate a sparse one? In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing.

陈希伦、库沙尔·拉科蒂亚、巴拉斯·奥古兹、安奇特·古普塔、帕特里克·刘易斯、斯坦·佩什特列夫、亚沙尔·梅赫达德、索纳尔·古普塔和文涛·易。2022年。显著短语感知的密集检索：密集检索器能否模仿稀疏检索器？见《2022年自然语言处理经验方法会议会议录》。

Arman Cohan, Sergey Feldman, Iz Beltagy, Doug Downey, and Daniel S. Weld. 2020. Specter: Document-level representation learning using citation-informed transformers.

阿尔曼·科汉、谢尔盖·费尔德曼、伊兹·贝尔塔吉、道格·唐尼和丹尼尔·S·韦尔德。2020年。Specter：使用引用感知的Transformer进行文档级表示学习。

Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Fernando Campos, and Ellen M. Voorhees. 2020. Overview of the trec 2020 deep learning track. ArXiv, abs/2003.07820.

尼克·克拉斯韦尔、巴斯卡尔·米特拉、埃米内·伊尔马兹、丹尼尔·费尔南多·坎波斯和埃伦·M·沃里斯。2020年。2020年TREC深度学习赛道概述。arXiv，abs/2003.07820。

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171-4186, Minneapolis, Minnesota. Association for Computational Linguistics.

雅各布·德夫林、张明伟、肯顿·李和克里斯蒂娜·图托纳娃。2019年。BERT：用于语言理解的深度双向Transformer预训练。见《计算语言学协会北美分会2019年会议：人类语言技术会议录，第1卷（长论文和短论文）》，第4171 - 4186页，明尼苏达州明尼阿波利斯。计算语言学协会。

Thomas Diggelmann, Jordan Boyd-Graber, Jannis Bu-lian, Massimiliano Ciaramita, and Markus Leippold. 2021. Climate-fever: A dataset for verification of real-world climate claims.

托马斯·迪格尔曼（Thomas Diggelmann）、乔丹·博伊德 - 格雷伯（Jordan Boyd - Graber）、扬尼斯·布利安（Jannis Bu - lian）、马西米利亚诺·恰拉米塔（Massimiliano Ciaramita）和马库斯·莱波尔德（Markus Leippold）。2021年。Climate - fever：用于验证现实世界气候主张的数据集。

William Fedus, Barret Zoph, and Noam Shazeer. 2022. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research, 23(120):1-39.

威廉·费杜斯（William Fedus）、巴雷特·佐夫（Barret Zoph）和诺姆·沙泽尔（Noam Shazeer）。2022年。开关变压器（Switch transformers）：通过简单高效的稀疏性扩展到万亿参数模型。《机器学习研究杂志》，23(120):1 - 39。

Thibault Formal, C. Lassance, Benjamin Piwowarski, and Stéphane Clinchant. 2021a. Splade v2: Sparse lexical and expansion model for information retrieval. ArXiv, abs/2109.10086.

蒂博·福尔马尔（Thibault Formal）、C. 拉桑斯（C. Lassance）、本杰明·皮沃瓦尔斯基（Benjamin Piwowarski）和斯特凡·克兰尚（Stéphane Clinchant）。2021a。Splade v2：用于信息检索的稀疏词汇和扩展模型。预印本平台ArXiv，abs/2109.10086。

Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant. 2021b. Splade: Sparse lexical and expansion model for first stage ranking. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. Association for Computing Machinery.

蒂博·福尔马尔（Thibault Formal）、本杰明·皮沃瓦尔斯基（Benjamin Piwowarski）和斯特凡·克兰尚（Stéphane Clinchant）。2021b。Splade：用于第一阶段排序的稀疏词汇和扩展模型。收录于第44届ACM国际信息检索研究与发展会议论文集。美国计算机协会。

Luyu Gao and Jamie Callan. 2021. Condenser: a pretraining architecture for dense retrieval. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 981-993, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

高璐宇（Luyu Gao）和杰米·卡兰（Jamie Callan）。2021年。凝聚器（Condenser）：一种用于密集检索的预训练架构。收录于2021年自然语言处理经验方法会议论文集，第981 - 993页，线上会议和多米尼加共和国蓬塔卡纳。计算语言学协会。

Luyu Gao and Jamie Callan. 2022. Unsupervised corpus aware language model pre-training for dense passage retrieval. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2843-2853, Dublin, Ireland. Association for Computational Linguistics.

高璐宇（Luyu Gao）和杰米·卡兰（Jamie Callan）。2022年。用于密集段落检索的无监督语料感知语言模型预训练。收录于计算语言学协会第60届年会论文集（第1卷：长论文），第2843 - 2853页，爱尔兰都柏林。计算语言学协会。

Luyu Gao, Zhuyun Dai, and Jamie Callan. 2021a. COIL: Revisit exact lexical match in information retrieval with contextualized inverted list. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3030-3042, Online. Association for Computational Linguistics.

高璐宇（Luyu Gao）、戴竹云（Zhuyun Dai）和杰米·卡兰（Jamie Callan）。2021a。COIL：通过上下文倒排列表重新审视信息检索中的精确词汇匹配。收录于计算语言学协会北美分会2021年人类语言技术会议论文集，第3030 - 3042页，线上会议。计算语言学协会。

Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. 2022. Tevatron: An efficient and flexible toolkit for dense retrieval.

高璐宇（Luyu Gao）、马学光（Xueguang Ma）、吉米·林（Jimmy Lin）和杰米·卡兰（Jamie Callan）。2022年。Tevatron：一种高效灵活的密集检索工具包。

Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021b. SimCSE: Simple contrastive learning of sentence em-beddings. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 6894-6910, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

高天宇（Tianyu Gao）、姚兴成（Xingcheng Yao）和陈丹琦（Danqi Chen）。2021b。SimCSE：简单的句子嵌入对比学习。收录于2021年自然语言处理经验方法会议论文集，第6894 - 6910页，线上会议和多米尼加共和国蓬塔卡纳。计算语言学协会。

Ruiqi Guo, Philip Sun, Erik Lindgren, Quan Geng, David Simcha, Felix Chern, and Sanjiv Kumar. 2020. Accelerating large-scale inference with anisotropic vector quantization. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of Proceedings of Machine Learning Research, pages 3887-3896. PMLR.

郭瑞琪（Ruiqi Guo）、菲利普·孙（Philip Sun）、埃里克·林德格伦（Erik Lindgren）、耿全（Quan Geng）、大卫·辛查（David Simcha）、费利克斯·陈（Felix Chern）和桑吉夫·库马尔（Sanjiv Kumar）。2020年。使用各向异性向量量化加速大规模推理。收录于第37届国际机器学习会议论文集，ICML 2020，2020年7月13 - 18日，线上会议，《机器学习研究会议录》第119卷，第3887 - 3896页。模式识别与机器学习会议录。

Faegheh Hasibi, Fedor Nikolaev, Chenyan Xiong, Krisz-tian Balog, Svein Erik Bratsberg, Alexander Kotov, and Jamie Callan. 2017. Dbpedia-entity v2: A test collection for entity search. In Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR '17, page 1265-1268, New York, NY, USA. Association for Computing Machinery.

法赫赫·哈西比（Faegheh Hasibi）、费多尔·尼古拉耶夫（Fedor Nikolaev）、熊晨燕（Chenyan Xiong）、克里斯蒂安·巴洛格（Krisztian Balog）、斯文·埃里克·布拉茨贝格（Svein Erik Bratsberg）、亚历山大·科托夫（Alexander Kotov）和杰米·卡兰（Jamie Callan）。2017年。Dbpedia - 实体v2：一个用于实体搜索的测试集。收录于第40届ACM国际信息检索研究与发展会议论文集，SIGIR '17，第1265 - 1268页，美国纽约州纽约市。美国计算机协会。

Geoffrey E. Hinton, Oriol Vinyals, and Jeffrey Dean. 2015. Distilling the knowledge in a neural network. ArXiv, abs/1503.02531.

杰弗里·E·辛顿（Geoffrey E. Hinton）、奥里奥尔·温亚尔斯（Oriol Vinyals）和杰弗里·迪恩（Jeffrey Dean）。2015年。提炼神经网络中的知识。预印本平台ArXiv，abs/1503.02531。

Sebastian Hofstätter, Omar Khattab, Sophia Althammer, Mete Sertkan, and Allan Hanbury. 2022. Introducing neural bag of whole-words with colberter: Contex-tualized late interactions using enhanced reduction. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management, page 737-747. Association for Computing Machinery.

塞巴斯蒂安·霍夫施泰特（Sebastian Hofstätter）、奥马尔·哈塔卜（Omar Khattab）、索菲娅·阿尔塔默（Sophia Althammer）、梅特·塞尔特坎（Mete Sertkan）和艾伦·汉伯里（Allan Hanbury）。2022年。通过ColBERTer引入全词神经词袋：使用增强约简的上下文后期交互。收录于第31届ACM国际信息与知识管理会议论文集，第737 - 747页。美国计算机协会。

Sebastian Hofstätter, Sheng-Chieh Lin, Jheng-Hong Yang, Jimmy Lin, and Allan Hanbury. 2021. Efficiently teaching an effective dense retriever with balanced topic aware sampling. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, page 113-122. Association for Computing Machinery.

塞巴斯蒂安·霍夫施泰特（Sebastian Hofstätter）、林圣杰（Sheng - Chieh Lin）、杨政宏（Jheng - Hong Yang）、吉米·林（Jimmy Lin）和艾伦·汉伯里（Allan Hanbury）。2021年。通过平衡主题感知采样高效训练有效的密集检索器。收录于第44届ACM国际信息检索研究与发展会议论文集，第113 - 122页。美国计算机协会。

Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. 2022. Unsupervised dense information retrieval with contrastive learning. Transactions on Machine Learning Research.

高蒂尔·伊扎卡尔（Gautier Izacard）、玛蒂尔德·卡龙（Mathilde Caron）、卢卡斯·侯赛尼（Lucas Hosseini）、塞巴斯蒂安·里德尔（Sebastian Riedel）、皮奥特·博亚诺夫斯基（Piotr Bojanowski）、阿尔芒·朱林（Armand Joulin）和爱德华·格雷夫（Edouard Grave）。2022年。通过对比学习进行无监督密集信息检索。《机器学习研究汇刊》。

Hervé Jégou, Matthijs Douze, and Cordelia Schmid. 2011. Product quantization for nearest neighbor search. IEEE Trans. Pattern Anal. Mach. Intell., 33(1):117-128.

埃尔韦·热古（Hervé Jégou）、马蒂亚斯·杜泽（Matthijs Douze）和科迪莉亚·施密德（Cordelia Schmid）。2011年。用于最近邻搜索的乘积量化。《电气与电子工程师协会模式分析与机器智能汇刊》（IEEE Trans. Pattern Anal. Mach. Intell.），33(1):117 - 128。

J. Johnson, M. Douze, and H. Jegou. 2021. Billion-scale similarity search with gpus. IEEE Transactions on Big Data, 7(03):535-547.

J. 约翰逊（J. Johnson）、M. 杜泽（M. Douze）和H. 热古（H. Jegou）。2021年。使用GPU进行十亿级相似度搜索。《电气与电子工程师协会大数据汇刊》（IEEE Transactions on Big Data），7(03):535 - 547。

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6769-6781, Online. Association for Computational Linguistics.

弗拉基米尔·卡尔普欣（Vladimir Karpukhin）、巴拉斯·奥古兹（Barlas Oguz）、闵世文（Sewon Min）、帕特里克·刘易斯（Patrick Lewis）、莱德尔·吴（Ledell Wu）、谢尔盖·叶杜诺夫（Sergey Edunov）、陈丹琦（Danqi Chen）和易文涛（Wen - tau Yih）。2020年。用于开放域问答的密集段落检索。《2020年自然语言处理经验方法会议论文集》（Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)），第6769 - 6781页，线上会议。计算语言学协会。

Omar Khattab and Matei Zaharia. 2020. Colbert: Efficient and effective passage search via contextual-ized late interaction over bert. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, page 39-48. Association for Computing Machinery.

奥马尔·哈塔卜（Omar Khattab）和马特·扎哈里亚（Matei Zaharia）。2020年。科尔伯特（Colbert）：通过基于BERT的上下文后期交互实现高效有效的段落搜索。《第44届美国计算机协会信息检索研究与发展国际会议论文集》（Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval），第39 - 48页。美国计算机协会。

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural questions: A benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:452-466.

汤姆·夸特科夫斯基（Tom Kwiatkowski）、珍妮玛丽亚·帕洛马基（Jennimaria Palomaki）、奥利维亚·雷德菲尔德（Olivia Redfield）、迈克尔·柯林斯（Michael Collins）、安库尔·帕里克（Ankur Parikh）、克里斯·阿尔贝蒂（Chris Alberti）、丹妮尔·爱泼斯坦（Danielle Epstein）、伊利亚·波洛苏欣（Illia Polosukhin）、雅各布·德夫林（Jacob Devlin）、肯顿·李（Kenton Lee）、克里斯蒂娜·图托纳娃（Kristina Toutanova）、利翁·琼斯（Llion Jones）、马修·凯尔西（Matthew Kelcey）、张明伟（Ming - Wei Chang）、安德鲁·M·戴（Andrew M. Dai）、雅各布·乌兹科雷特（Jakob Uszkoreit）、勒·奎克（Quoc Le）和斯拉夫·彼得罗夫（Slav Petrov）。2019年。自然问题：问答研究的基准。《计算语言学协会汇刊》（Transactions of the Association for Computational Linguistics），7:452 - 466。

Carlos Lassance and Stéphane Clinchant. 2023. The tale of two msmarco - and their unfair comparisons.

卡洛斯·拉桑斯（Carlos Lassance）和斯特凡·克兰尚（Stéphane Clinchant）。2023年。两个MSMarco数据集的故事——以及它们不公平的比较。

Jinhyuk Lee, Zhuyun Dai, Sai Meher Karthik Duddu, Tao Lei, Iftekhar Naim, Ming-Wei Chang, and Vincent Y. Zhao. 2023. Rethinking the role of token retrieval in multi-vector retrieval.

李晋赫（Jinhyuk Lee）、戴竹云（Zhuyun Dai）、赛·梅赫尔·卡尔蒂克·杜杜（Sai Meher Karthik Duddu）、雷涛（Tao Lei）、伊夫特哈尔·奈姆（Iftekhar Naim）、张明伟（Ming - Wei Chang）和赵文森（Vincent Y. Zhao）。2023年。重新思考多向量检索中词元检索的作用。

Kenton Lee, Ming-Wei Chang, and Kristina Toutanova. 2019. Latent retrieval for weakly supervised open domain question answering. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 6086-6096, Florence, Italy. Association for Computational Linguistics.

肯顿·李（Kenton Lee）、张明伟（Ming - Wei Chang）和克里斯蒂娜·图托纳娃（Kristina Toutanova）。2019年。用于弱监督开放域问答的潜在检索。《第57届计算语言学协会年会论文集》（Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics），第6086 - 6096页，意大利佛罗伦萨。计算语言学协会。

Jimmy Lin, Xueguang Ma, Sheng-Chieh Lin, Jheng-Hong Yang, Ronak Pradeep, and Rodrigo Nogueira. 2021a. Pyserini: A python toolkit for reproducible information retrieval research with sparse and dense representations. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, page 2356-2362. Association for Computing Machinery.

林吉米（Jimmy Lin）、马学光（Xueguang Ma）、林圣杰（Sheng - Chieh Lin）、杨政宏（Jheng - Hong Yang）、罗纳克·普拉迪普（Ronak Pradeep）和罗德里戈·诺盖拉（Rodrigo Nogueira）。2021a。Pyserini：一个用于使用稀疏和密集表示进行可重现信息检索研究的Python工具包。《第44届美国计算机协会信息检索研究与发展国际会议论文集》（Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval），第2356 - 2362页。美国计算机协会。

Jimmy J. Lin and Xueguang Ma. 2021. A few brief notes on deepimpact, coil, and a conceptual framework for information retrieval techniques. ArXiv, abs/2106.14807.

林吉米·J（Jimmy J. Lin）和马学光（Xueguang Ma）。2021年。关于DeepImpact、CoIL以及信息检索技术概念框架的几点简要说明。预印本平台（ArXiv），论文编号：abs/2106.14807。

Sheng-Chieh Lin and Jimmy Lin. 2022. A dense representation framework for lexical and semantic matching. ArXiv, abs/2206.09912.

林圣杰（Sheng - Chieh Lin）和林吉米（Jimmy Lin）。2022年。一个用于词汇和语义匹配的密集表示框架。预印本平台（ArXiv），论文编号：abs/2206.09912。

Sheng-Chieh Lin, Jheng-Hong Yang, and Jimmy Lin. 2021b. In-batch negatives for knowledge distillation with tightly-coupled teachers for dense retrieval. In Proceedings of the 6th Workshop on Representation Learning for NLP (RepL4NLP-2021), pages 163-173, Online. Association for Computational Linguistics.

林圣杰（Sheng - Chieh Lin）、杨政宏（Jheng - Hong Yang）和林吉米（Jimmy Lin）。2021b。使用紧密耦合教师进行知识蒸馏以用于密集检索的批内负样本。《第6届自然语言处理表示学习研讨会论文集》（Proceedings of the 6th Workshop on Representation Learning for NLP (RepL4NLP - 2021)），第163 - 173页，线上会议。计算语言学协会。

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. ArXiv, abs/1907.11692.

刘音涵（Yinhan Liu）、迈尔·奥特（Myle Ott）、纳曼·戈亚尔（Naman Goyal）、杜静飞（Jingfei Du）、曼达尔·乔希（Mandar Joshi）、陈丹琦（Danqi Chen）、奥默·利维（Omer Levy）、迈克·刘易斯（Mike Lewis）、卢克·泽特尔莫耶（Luke Zettlemoyer）和韦塞林·斯托扬诺夫（Veselin Stoyanov）。2019年。罗伯塔（Roberta）：一种稳健优化的BERT预训练方法。预印本平台（ArXiv），论文编号：abs/1907.11692。

Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In 7th International Conference on Learning Representations, ICLR 2019. OpenReview.net.

伊利亚·洛希洛夫（Ilya Loshchilov）和弗兰克·胡特（Frank Hutter）。2019年。解耦权重衰减正则化。《第7届国际学习表征会议》（7th International Conference on Learning Representations, ICLR 2019）。OpenReview.net。

Shuqi Lu, Di He, Chenyan Xiong, Guolin Ke, Waleed Malik, Zhicheng Dou, Paul Bennett, Tie-Yan Liu, and Arnold Overwijk. 2021. Less is more: Pretrain a strong Siamese encoder for dense text retrieval using a weak decoder. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 2780-2791, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

陆舒琪（Shuqi Lu）、何迪（Di He）、熊晨彦（Chenyan Xiong）、柯国霖（Guolin Ke）、瓦利德·马利克（Waleed Malik）、窦志成（Zhicheng Dou）、保罗·贝内特（Paul Bennett）、刘铁岩（Tie - Yan Liu）和阿诺德·奥弗维克（Arnold Overwijk）。2021年。少即是多：使用弱解码器预训练一个强大的孪生编码器用于密集文本检索。《2021年自然语言处理经验方法会议论文集》（Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing），第2780 - 2791页，线上会议和多米尼加共和国蓬塔卡纳。计算语言学协会。

Yi Luan, Jacob Eisenstein, Kristina Toutanova, and Michael Collins. 2021. Sparse, dense, and attentional

栾毅（Yi Luan）、雅各布·艾森斯坦（Jacob Eisenstein）、克里斯蒂娜·图托纳娃（Kristina Toutanova）和迈克尔·柯林斯（Michael Collins）。2021年。用于文本检索的稀疏、密集和注意力表示

representations for text retrieval. Transactions of the Association for Computational Linguistics, 9:329- 345.

《计算语言学协会汇刊》（Transactions of the Association for Computational Linguistics），9:329 - 345。

Joel Mackenzie, Andrew Trotman, and Jimmy Lin. 2021. Wacky weights in learned sparse representations and the revenge of score-at-a-time query evaluation. ArXiv, abs/2110.11540.

乔尔·麦肯齐（Joel Mackenzie）、安德鲁·特罗特曼（Andrew Trotman）和吉米·林（Jimmy Lin）。2021年。学习型稀疏表示中的奇异权重与逐次评分查询评估的逆袭。预印本（ArXiv），abs/2110.11540。

Macedo Maia, Siegfried Handschuh, André Freitas, Brian Davis, Ross McDermott, Manel Zarrouk, and Alexandra Balahur. 2018. Www'18 open challenge: Financial opinion mining and question answering. In Companion Proceedings of the The Web Conference 2018, WWW '18, page 1941-1942, Republic and Canton of Geneva, CHE. International World Wide Web Conferences Steering Committee.

马塞多·马亚（Macedo Maia）、齐格弗里德·汉舒（Siegfried Handschuh）、安德烈·弗雷塔斯（André Freitas）、布莱恩·戴维斯（Brian Davis）、罗斯·麦克德莫特（Ross McDermott）、马内尔·扎鲁克（Manel Zarrouk）和亚历山德拉·巴拉胡尔（Alexandra Balahur）。2018年。万维网会议2018（WWW '18）开放挑战：金融观点挖掘与问答。收录于《2018年万维网会议伴侣论文集》（Companion Proceedings of the The Web Conference 2018），WWW '18，第1941 - 1942页，瑞士日内瓦共和国和州。国际万维网会议指导委员会。

Antonio Mallia, Omar Khattab, Torsten Suel, and Nicola Tonellotto. 2021. Learning passage impacts for inverted indexes. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, page 1723-1727. Association for Computing Machinery.

安东尼奥·马利亚（Antonio Mallia）、奥马尔·哈塔布（Omar Khattab）、托尔斯滕·苏尔（Torsten Suel）和尼古拉·托内洛托（Nicola Tonellotto）。2021年。为倒排索引学习段落影响。收录于《第44届ACM SIGIR国际信息检索研究与发展会议论文集》（Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval），第1723 - 1727页。美国计算机协会。

Christopher D. Manning, Prabhakar Raghavan, and Hin-rich Schütze. 2008. Introduction to information retrieval. Cambridge University Press.

克里斯托弗·D·曼宁（Christopher D. Manning）、普拉巴卡尔·拉加万（Prabhakar Raghavan）和欣里希·舒策（Hin - rich Schütze）。2008年。《信息检索导论》（Introduction to information retrieval）。剑桥大学出版社。

Basil Mustafa, Carlos Riquelme, Joan Puigcerver, Rodolphe Jenatton, and Neil Houlsby. 2022. Multimodal contrastive learning with limoe: the language-image mixture of experts. ArXiv, abs/2206.02770.

巴兹尔·穆斯塔法（Basil Mustafa）、卡洛斯·里克尔梅（Carlos Riquelme）、琼·普伊格塞尔弗（Joan Puigcerver）、罗多尔夫·热纳通（Rodolphe Jenatton）和尼尔·霍尔斯比（Neil Houlsby）。2022年。使用Limoe进行多模态对比学习：专家的语言 - 图像混合。预印本（ArXiv），abs/2206.02770。

Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, and Li Deng. 2016. MS MARCO: A human generated machine reading comprehension dataset. In $\mathrm{{CoCo}}@\mathrm{{NIPS}}$ .

特里·阮（Tri Nguyen）、米尔·罗森伯格（Mir Rosenberg）、宋霞（Xia Song）、高剑锋（Jianfeng Gao）、索拉布·蒂瓦里（Saurabh Tiwary）、兰甘·马朱姆德（Rangan Majumder）和李邓（Li Deng）。2016年。MS MARCO：一个人工生成的机器阅读理解数据集。收录于$\mathrm{{CoCo}}@\mathrm{{NIPS}}$ 。

Natasha Noy, Matthew Burgess, and Dan Brickley. 2019. Google dataset search: Building a search engine for datasets in an open web ecosystem. In 28th Web Conference (WebConf 2019).

娜塔莎·诺伊（Natasha Noy）、马修·伯吉斯（Matthew Burgess）和丹·布里克利（Dan Brickley）。2019年。谷歌数据集搜索：在开放网络生态系统中构建数据集搜索引擎。收录于第28届万维网会议（WebConf 2019）。

Yujie Qian, Jinhyuk Lee, Sai Meher Karthik Duddu, Zhuyun Dai, Siddhartha Brahma, Iftekhar Naim, Tao Lei, and Vincent Zhao. 2022. Multi-vector retrieval as sparse alignment. ArXiv, abs/2211.01267.

钱玉洁（Yujie Qian）、李晋赫（Jinhyuk Lee）、赛·梅赫尔·卡尔蒂克·杜杜（Sai Meher Karthik Duddu）、戴竹云（Zhuyun Dai）、悉达多·布拉马（Siddhartha Brahma）、伊夫特哈尔·奈姆（Iftekhar Naim）、陶磊（Tao Lei）和文森特·赵（Vincent Zhao）。2022年。多向量检索作为稀疏对齐。预印本（ArXiv），abs/2211.01267。

Stephen E. Robertson and Hugo Zaragoza. 2009. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval, 3(4):333-389.

斯蒂芬·E·罗伯逊（Stephen E. Robertson）和雨果·萨拉萨尔（Hugo Zaragoza）。2009年。概率相关性框架：BM25及超越。《信息检索基础与趋势》（Foundations and Trends in Information Retrieval），3(4):333 - 389。

Gerard Salton and Christopher Buckley. 1988. Term-weighting approaches in automatic text retrieval. Information Processing & Management, 24(5):513- 523.

杰拉德·萨尔顿（Gerard Salton）和克里斯托弗·巴克利（Christopher Buckley）。1988年。自动文本检索中的词项加权方法。《信息处理与管理》（Information Processing & Management），24(5):513 - 523。

Keshav Santhanam, Omar Khattab, Christopher Potts, and Matei Zaharia. 2022a. Plaid: An efficient engine for late interaction retrieval. ArXiv, abs/2205.09707.

凯沙夫·桑塔南姆（Keshav Santhanam）、奥马尔·哈塔布（Omar Khattab）、克里斯托弗·波茨（Christopher Potts）和马特·扎哈里亚（Matei Zaharia）。2022a。Plaid：一种用于后期交互检索的高效引擎。预印本（ArXiv），abs/2205.09707。

Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. 2022b. Col-BERTv2: Effective and efficient retrieval via lightweight late interaction. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3715-3734, Seattle, United States. Association for Computational Linguistics.

凯沙夫·桑塔南姆（Keshav Santhanam）、奥马尔·哈塔布（Omar Khattab）、乔恩·萨德 - 法尔孔（Jon Saad - Falcon）、克里斯托弗·波茨（Christopher Potts）和马特·扎哈里亚（Matei Zaharia）。2022b。Col - BERTv2：通过轻量级后期交互实现高效检索。收录于《2022年计算语言学协会北美分会人类语言技术会议论文集》（Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies），第3715 - 3734页，美国西雅图。计算语言学协会。

Christopher Sciavolino, Zexuan Zhong, Jinhyuk Lee, and Danqi Chen. 2021. Simple entity-centric questions challenge dense retrievers. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 6138-6148, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

克里斯托弗·夏沃利诺（Christopher Sciavolino）、钟泽轩（Zexuan Zhong）、李晋赫（Jinhyuk Lee）和陈丹琦（Danqi Chen）。2021年。简单的以实体为中心的问题挑战密集检索器。收录于《2021年自然语言处理经验方法会议论文集》（Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing），第6138 - 6148页，线上及多米尼加共和国蓬塔卡纳。计算语言学协会。

Tao Shen, Xiubo Geng, Chongyang Tao, Can Xu, Kai Zhang, and Daxin Jiang. 2022. Unifier: A unified retriever for large-scale retrieval. ArXiv, abs/2205.11194.

沈涛（Tao Shen）、耿秀波（Xiubo Geng）、陶重阳（Chongyang Tao）、徐灿（Can Xu）、张凯（Kai Zhang）和蒋大新（Daxin Jiang）。2022年。Unifier：一种用于大规模检索的统一检索器。预印本（ArXiv），abs/2205.11194。

Nandan Thakur, Nils Reimers, Andreas Rücklé, Ab-hishek Srivastava, and Iryna Gurevych. 2021. BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2).

南丹·塔库尔（Nandan Thakur）、尼尔斯·赖默斯（Nils Reimers）、安德里亚斯·吕克莱（Andreas Rücklé）、阿比舍克·斯里瓦斯塔瓦（Ab-hishek Srivastava）和伊琳娜·古列维奇（Iryna Gurevych）。2021年。BEIR：信息检索模型零样本评估的异构基准。收录于第三十五届神经信息处理系统数据集与基准赛道会议（第二轮）。

James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. 2018. FEVER: a large-scale dataset for fact extraction and VERification. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 809-819, New Orleans, Louisiana. Association for Computational Linguistics.

詹姆斯·索恩（James Thorne）、安德里亚斯·弗拉乔斯（Andreas Vlachos）、克里斯托斯·克里斯托杜洛普洛斯（Christos Christodoulopoulos）和阿尔皮特·米塔尔（Arpit Mittal）。2018年。FEVER：用于事实提取与验证的大规模数据集。收录于2018年北美计算语言学协会人类语言技术会议论文集，第1卷（长论文），第809 - 819页，路易斯安那州新奥尔良。计算语言学协会。

Ellen Voorhees, Tasmeer Alam, Steven Bedrick, Dina Demner-Fushman, William R Hersh, Kyle Lo, Kirk Roberts, Ian Soboroff, and Lucy Lu Wang. 2020. Trec-covid: Constructing a pandemic information retrieval test collection.

艾伦·沃里斯（Ellen Voorhees）、塔斯米尔·阿拉姆（Tasmeer Alam）、史蒂文·贝德里克（Steven Bedrick）、迪娜·德姆纳 - 富什曼（Dina Demner-Fushman）、威廉·R·赫什（William R Hersh）、凯尔·洛（Kyle Lo）、柯克·罗伯茨（Kirk Roberts）、伊恩·索博罗夫（Ian Soboroff）和露西·陆·王（Lucy Lu Wang）。2020年。Trec - covid：构建大流行信息检索测试集。

Henning Wachsmuth, Shahbaz Syed, and Benno Stein. 2018. Retrieval of the best counterargument without prior topic knowledge. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 241-251, Melbourne, Australia. Association for Computational Linguistics.

亨宁·瓦克斯穆特（Henning Wachsmuth）、沙赫巴兹·赛义德（Shahbaz Syed）和本诺·施泰因（Benno Stein）。2018年。在无先验主题知识的情况下检索最佳反驳论点。收录于计算语言学协会第56届年会论文集（第1卷：长论文），第241 - 251页，澳大利亚墨尔本。计算语言学协会。

David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan, and Hannaneh Hajishirzi. 2020. Fact or fiction: Verifying scientific claims. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 7534-7550, Online. Association for Computational Linguistics.

大卫·瓦登（David Wadden）、林山川（Shanchuan Lin）、凯尔·洛（Kyle Lo）、露西·陆·王（Lucy Lu Wang）、玛德琳·范·祖伊伦（Madeleine van Zuylen）、阿尔曼·科汉（Arman Cohan）和汉纳内·哈吉希尔齐（Hannaneh Hajishirzi）。2020年。事实还是虚构：验证科学主张。收录于2020年自然语言处理经验方法会议论文集（EMNLP），第7534 - 7550页，线上会议。计算语言学协会。

Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N. Bennett, Junaid Ahmed, and Arnold Overwijk. 2021. Approximate nearest neighbor negative contrastive learning for dense text retrieval. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net.

熊磊（Lee Xiong）、熊晨彦（Chenyan Xiong）、李野（Ye Li）、邓国峰（Kwok-Fung Tang）、刘佳琳（Jialin Liu）、保罗·N·贝内特（Paul N. Bennett）、朱奈德·艾哈迈德（Junaid Ahmed）和阿诺德·奥弗维克（Arnold Overwijk）。2021年。用于密集文本检索的近似最近邻负对比学习。收录于第9届国际学习表征会议，ICLR 2021，虚拟会议，奥地利，2021年5月3 - 7日。OpenReview.net。

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2369-2380, Brussels, Belgium. Association for Computational Linguistics.

杨植麟（Zhilin Yang）、齐鹏（Peng Qi）、张赛增（Saizheng Zhang）、约书亚·本吉奥（Yoshua Bengio）、威廉·科恩（William Cohen）、鲁斯兰·萨拉赫丁诺夫（Ruslan Salakhutdinov）和克里斯托弗·D·曼宁（Christopher D. Manning）。2018年。HotpotQA：用于多样化、可解释多跳问答的数据集。收录于2018年自然语言处理经验方法会议论文集，第2369 - 2380页，比利时布鲁塞尔。计算语言学协会。

Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Jiafeng Guo, Min Zhang, and Shaoping Ma. 2021. Optimizing dense retrieval model training with hard negatives. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, page 1503-1512. Association for Computing Machinery.

詹景涛（Jingtao Zhan）、毛佳欣（Jiaxin Mao）、刘奕群（Yiqun Liu）、郭佳峰（Jiafeng Guo）、张敏（Min Zhang）和马少平（Shaoping Ma）。2021年。利用难负样本优化密集检索模型训练。收录于第44届国际ACM SIGIR信息检索研究与发展会议论文集，第1503 - 1512页。美国计算机协会。

Kai Zhang, Chongyang Tao, Tao Shen, Can Xu, Xiubo Geng, Binxing Jiao, and Daxin Jiang. 2023. Led: Lexicon-enlightened dense retriever for large-scale retrieval. In Proceedings of the ACM Web Conference 2023, WWW '23, page 3203-3213, New York, NY, USA. Association for Computing Machinery.

张凯（Kai Zhang）、陶重阳（Chongyang Tao）、沈涛（Tao Shen）、徐灿（Can Xu）、耿秀波（Xiubo Geng）、焦彬星（Binxing Jiao）和蒋大新（Daxin Jiang）。2023年。Led：用于大规模检索的词典启发式密集检索器。收录于2023年ACM网络会议论文集，WWW '23，第3203 - 3213页，美国纽约。美国计算机协会。

Shunyu Zhang, Yaobo Liang, Ming Gong, Daxin Jiang, and Nan Duan. 2022. Multi-view document representation learning for open-domain dense retrieval. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5990-6000, Dublin, Ireland. Association for Computational Linguistics.

张顺宇（Shunyu Zhang）、梁耀博（Yaobo Liang）、龚明（Ming Gong）、蒋大新（Daxin Jiang）和段楠（Nan Duan）。2022年。用于开放域密集检索的多视图文档表示学习。收录于计算语言学协会第60届年会论文集（第1卷：长论文），第5990 - 6000页，爱尔兰都柏林。计算语言学协会。

## A Implementations

## A 实现

### A.1 Datasets

### A.1 数据集

The MS MARCO passages corpus has around 8.8 million passages with an average length of 60 words. TREC DL 2019 and 2020 contain 43 and 54 test queries whose relevance sets are densely labelled with scores from 0 to 4 .

MS MARCO段落语料库约有880万个段落，平均长度为60个单词。TREC DL 2019和2020分别包含43个和54个测试查询，其相关集用0到4的分数进行了密集标注。

For out-of-domain evaluation, we use 13 datasets from BEIR, which includes TREC-COVID (Voorhees et al., 2020), NFCorpus (Boteva et al., 2016), Natural Questions (Kwiatkowski et al., 2019), HotpotQA (Yang et al., 2018), FiQA-2018 (Maia et al., 2018), ArguAna Coun-terargs Corpus (Wachsmuth et al., 2018), Touché- 2020 (Bondarenko et al.,2020), Quora ${}^{2}$ , DBPedia-Entity-v2 (Hasibi et al., 2017), SCIDOCS (Cohan et al., 2020), FEVER (Thorne et al., 2018), Climate-FEVER (Diggelmann et al., 2021), SciFact (Wad-den et al., 2020).

对于域外评估，我们使用了来自BEIR的13个数据集，其中包括TREC - COVID（沃里斯等人，2020年）、NFCorpus（博特瓦等人，2016年）、自然问题数据集（克维亚特科夫斯基等人，2019年）、HotpotQA（杨等人，2018年）、FiQA - 2018（马亚等人，2018年）、ArguAna反论点语料库（瓦克斯穆特等人，2018年）、Touché - 2020（邦达连科等人，2020年）、Quora ${}^{2}$ 、DBPedia实体数据集v2（哈西比等人，2017年）、SCIDOCS（科恩等人，2020年）、FEVER（索恩等人，2018年）、Climate - FEVER（迪格尔曼等人，2021年）、SciFact（瓦登等人，2020年）。

### A.2 Baselines for Section 4

### A.2 第4节的基线模型

All the baseline models below are trained and evaluated under the same setting of CITADEL (e.g., datasets, hyperparameters, and hardwares).

以下所有基线模型均在与CITADEL相同的设置（例如，数据集、超参数和硬件）下进行训练和评估。

Sparse Retrievers. BM25 (Robertson and Zaragoza, 2009) uses the term frequency and inverted document frequency as features to compute the similarity between documents. SPLADE (Formal et al., 2021b,a) leverages the pre-trained language model's MLM layer and ReLU activation to yield sparse term importance.

稀疏检索器。BM25（罗伯逊和萨拉戈萨，2009年）使用词频和逆文档频率作为特征来计算文档之间的相似度。SPLADE（福尔马尔等人，2021b,a）利用预训练语言模型的掩码语言模型（MLM）层和修正线性单元（ReLU）激活函数来得出稀疏的词重要性。

Dense Retrievers. DPR (Karpukhin et al., 2020) encodes the input text into a single vector. coCon-denser (Gao and Callan, 2022) pre-trains DPR in an unsupervised fashion before fine-tuning.

密集检索器。DPR（卡尔普欣等人，2020年）将输入文本编码为单个向量。coCondenser（高和卡兰，2022年）在微调之前以无监督的方式对DPR进行预训练。

Multi-Vector Retrievers. ColBERT (Khattab and Zaharia, 2020; Santhanam et al., 2022b) encodes each token into dense vectors and performs late interaction between query token vectors and document token vectors. COIL (Gao et al., 2021a) applies an exact match constraint on late interaction to improve efficiency and robustness.

多向量检索器。ColBERT（卡塔布和扎哈里亚，2020年；桑塔纳姆等人，2022b）将每个标记编码为密集向量，并在查询标记向量和文档标记向量之间进行后期交互。COIL（高等人，2021a）对后期交互应用精确匹配约束，以提高效率和鲁棒性。

### A.3 Training

### A.3 训练

For CITADEL, we use bert-base-uncased as the initial checkpoint for fine-tuning. Following COIL,

对于CITADEL，我们使用bert - base - uncased作为微调的初始检查点。遵循COIL的做法，

2 https://quoradata.quora.com/

2 https://quoradata.quora.com/

First-Quora-Dataset-Release-Question-Pairs ms-marco-MiniLM-L-6-v2 we set the [CLS] vector dimension to 128, token vector dimension to 32, maximal routing keys to 5 for document and 1 for query, $\alpha$ and $\beta$ in Equation (14) are set to be $1\mathrm{e} - 2$ and $1\mathrm{e} - 5$ ,respectively. We add the dot product of [CLS] vectors in Equation (1) to the final similarity score in Equation (5). All models are trained for 10 epochs with AdamW (Loshchilov and Hutter, 2019) optimizer, a learning rate of 2e-5 with 3000 warm-up steps and linear decay. Hard negatives are sampled from top-100 BM25 retrieval results. Each query is paired with 1 positive and 7 hard negatives for faster convergence. We use a batch size of 128 on MS MARCO passages with 32 A100 GPUs.

第一个Quora数据集发布 - 问题对 ms - marco - MiniLM - L - 6 - v2 我们将[CLS]向量维度设置为128，标记向量维度设置为32，文档的最大路由键设置为5，查询的最大路由键设置为1，公式（14）中的$\alpha$和$\beta$分别设置为$1\mathrm{e} - 2$和$1\mathrm{e} - 5$。我们将公式（1）中[CLS]向量的点积添加到公式（5）的最终相似度得分中。所有模型使用AdamW优化器（洛希奇洛夫和胡特，2019年）训练10个周期，学习率为2e - 5，有3000个热身步骤和线性衰减。硬负样本从BM25前100的检索结果中采样。每个查询与1个正样本和7个硬负样本配对，以实现更快的收敛。我们在MS MARCO段落上使用128的批量大小，使用32个A100 GPU。

<!-- Media -->

<!-- figureText: Token-Level Search Scatter Operations Sorting CPU latency (ms/query) -->

<img src="https://cdn.noedgeai.com/0195a46c-451c-7169-bf50-83435cd534a9_13.jpg?x=827&y=182&w=619&h=465&r=0"/>

Figure 8: Latency breakdown of inverted vector retrieval for CITADEL and COIL.

图8：CITADEL和COIL的倒排向量检索延迟分解。

<!-- Media -->

For a fair comparison with recent state-of-the-art models, we further train CITADEL using cross-encoder distillation and hard negative mining. First, we use the trained CITADEL model under the setting in the last paragraph to retrieve top-100 candidates from the corpus for the training queries. We then use the cross-encoder ${}^{3}$ to rerank the top-100 candidates and score each query-document pair. Finally, we re-initialize CITADEL with bert-base-uncased using the positives and negatives sample from the top-100 candidates scored by the cross-encoder, with a 1:1 ratio for the soft-label and hard-label loss mixing (Hinton et al., 2015). We also repeat another round of hard negative mining and distillation but it does not seem to improve the performance any further.

为了与近期的最先进模型进行公平比较，我们进一步使用交叉编码器蒸馏和硬负样本挖掘来训练CITADEL。首先，我们使用上一段设置下训练好的CITADEL模型从语料库中为训练查询检索前100个候选样本。然后，我们使用交叉编码器${}^{3}$对前100个候选样本进行重新排序，并对每个查询 - 文档对进行评分。最后，我们使用bert - base - uncased重新初始化CITADEL，使用交叉编码器评分的前100个候选样本中的正样本和负样本，软标签和硬标签损失混合比例为1:1（辛顿等人，2015年）。我们还重复了另一轮硬负样本挖掘和蒸馏，但似乎并没有进一步提高性能。

---

<!-- Footnote -->

3https://huggingface.co/cross-encoder/

3https://huggingface.co/cross - encoder/

<!-- Footnote -->

---

### A.4 Inference and Latency Breakdown

### A.4 推理与延迟分析

Pipeline. We implemented the retrieval pipeline with PyTorch (GPU) and Numpy (CPU), with a small Cython extension module for scatter operations similar to COIL’s ${}^{4}$ . As shown in Fig 8,our pipeline could be roughly decomposed into four independent parts: query encoding, token-level retrieval, scatter operations, and sorting. We use the same pipeline for COIL's retrieval process. For ColBERT's latency breakdown please refer to San-thanam et al. (2022a). The cost of query encoding comes from the forward pass of the query encoder, which could be independently optimized using quantization or weight pruning for neural networks. Besides that, the most expensive operation is the token-level retrieval, which is directly influenced by the token index size. We could see that a more balanced index size distribution as shown in Figure 3 has a much smaller token vector retrieval latency. The scatter operations are used to gather the token vectors from the same passage ids from different token indices, which is also related to the token index size distribution. Finally, we sort the aggregated ranking results and return the candidates.

流水线。我们使用PyTorch（GPU）和Numpy（CPU）实现了检索流水线，并使用一个小型的Cython扩展模块进行类似于COIL的${}^{4}$的分散操作。如图8所示，我们的流水线大致可以分解为四个独立的部分：查询编码、词元级检索、分散操作和排序。我们在COIL的检索过程中使用相同的流水线。关于ColBERT的延迟分析，请参考Santhanam等人（2022a）的研究。查询编码的成本来自查询编码器的前向传播，这可以通过神经网络的量化或权重剪枝进行独立优化。除此之外，最耗时的操作是词元级检索，它直接受词元索引大小的影响。我们可以看到，如图3所示的更均衡的索引大小分布具有更小的词元向量检索延迟。分散操作用于从不同的词元索引中收集来自相同段落ID的词元向量，这也与词元索引大小分布有关。最后，我们对聚合的排序结果进行排序并返回候选结果。

Hardwares and Latency Measurement. We measure all the retrieval models in Table 1 on a single A100 GPU for GPU search and a single Intel(R) Xeon(R) Platinum 8275CL CPU @ 3.00GHz for CPU search. All indices are stored in fp32 (token vectors) and int64 (corpus ids if necessary) on disk. We use a query batch size of 1 and return the top-1000 candidates by default to simulate streaming queries. We compute the average latency of all queries on MS MARCO passages' Dev set and then report the minimum average latency across 3 trials following PLAID (Santhanam et al., 2022a). $\mathrm{I}/\mathrm{O}$ time is excluded from the latency but the time of moving tensors from CPU to GPU during GPU retrieval is included.

硬件与延迟测量。我们在单个A100 GPU上对表1中的所有检索模型进行GPU搜索，并在单个英特尔（R）至强（R）铂金8275CL CPU @ 3.00GHz上进行CPU搜索。所有索引以fp32（词元向量）和int64（必要时为语料库ID）格式存储在磁盘上。我们默认使用查询批量大小为1，并返回前1000个候选结果以模拟流式查询。我们计算MS MARCO段落开发集上所有查询的平均延迟，然后按照PLAID（Santhanam等人，2022a）的方法报告3次试验中的最小平均延迟。$\mathrm{I}/\mathrm{O}$时间不计入延迟，但GPU检索期间将张量从CPU移动到GPU的时间包括在内。

---

<!-- Footnote -->

${}^{4}$ https://github.com/luyug/COIL/tree/main/ retriever

${}^{4}$ https://github.com/luyug/COIL/tree/main/ retriever

<!-- Footnote -->

---

## ACL 2023 Responsible NLP Checklist

## ACL 2023 负责任的自然语言处理清单

A For every submission:

A 对于每一篇投稿：

- A1. Did you describe the limitations of your work?

- A1. 你是否描述了你的工作的局限性？

10

A2. Did you discuss any potential risks of your work?

A2. 你是否讨论了你的工作的任何潜在风险？

This work provides an information retrieval method for public datasets.

这项工作为公共数据集提供了一种信息检索方法。

A3. Do the abstract and introduction summarize the paper's main claims?

A3. 摘要和引言是否总结了论文的主要观点？

1

A A4. Have you used AI writing assistants when working on this paper?

A A4. 你在撰写这篇论文时是否使用了人工智能写作助手？

Left blank.

留空。

B Did you use or create scientific artifacts?

B 你是否使用或创建了科学制品？

4, 5, Appendix A

4, 5, 附录A

B1. Did you cite the creators of artifacts you used?

B1. 你是否引用了你所使用的制品的创建者？

4, 5, Appendix A

4, 5, 附录A

B2. Did you discuss the license or terms for use and / or distribution of any artifacts?

B2. 你们是否讨论过任何制品的许可或使用及/或分发条款？

4, 5, Appendix A

4, 5, 附录A

B3. Did you discuss if your use of existing artifact(s) was consistent with their intended use, provided that it was specified? For the artifacts you create, do you specify intended use and whether that is compatible with the original access conditions (in particular, derivatives of data accessed for research purposes should not be used outside of research contexts)?

B3. 如果已有制品指定了预期用途，你们是否讨论过对这些制品的使用是否符合其预期用途？对于你们创建的制品，你们是否指定了预期用途，以及该用途是否与原始访问条件兼容（特别是，为研究目的获取的数据的衍生数据不应在研究环境之外使用）？

4, 5, Appendix A

4, 5, 附录A

B B4. Did you discuss the steps taken to check whether the data that was collected / used contains any information that names or uniquely identifies individual people or offensive content, and the steps taken to protect / anonymize it?

B4. 你们是否讨论过为检查所收集/使用的数据是否包含任何指名或唯一识别个人的信息或冒犯性内容而采取的步骤，以及为保护/匿名化这些信息而采取的步骤？

The public datasets in the paper are widely used for a long time.

论文中的公共数据集已被广泛使用很长时间。

B5. Did you provide documentation of the artifacts, e.g., coverage of domains, languages, and linguistic phenomena, demographic groups represented, etc.? Not applicable. Left blank.

B5. 你们是否提供了制品的文档，例如，涵盖的领域、语言、语言现象、所代表的人口群体等？不适用。留空。

4 B6. Did you report relevant statistics like the number of examples, details of train / test / dev splits, etc. for the data that you used / created? Even for commonly-used benchmark datasets, include the number of examples in train / validation / test splits, as these provide necessary context for a reader to understand experimental results. For example, small differences in accuracy on large test sets may be significant, while on small test sets they may not be. Left blank.

4 B6. 你们是否报告了所使用/创建的数据的相关统计信息，如示例数量、训练/测试/开发集划分的详细信息等？即使是常用的基准数据集，也应包括训练/验证/测试集划分中的示例数量，因为这些信息为读者理解实验结果提供了必要的背景。例如，大型测试集上的准确率微小差异可能具有显著性，而在小型测试集上则可能不具有显著性。留空。

## C Did you run computational experiments?

## C 你们是否进行了计算实验？

## 4,5,6,7

C1. Did you report the number of parameters in the models used, the total computational budget (e.g., GPU hours), and computing infrastructure used? Appendix A

C1. 你们是否报告了所使用模型的参数数量、总计算预算（如GPU小时数）以及所使用的计算基础设施？附录A

---

<!-- Footnote -->

The Responsible NLP Checklist used at ACL 2023 is adopted from NAACL 2022, with the addition of a question on AI writing assistance.

ACL 2023使用的负责任自然语言处理清单是从NAACL 2022沿用而来，并增加了一个关于人工智能写作辅助的问题。

<!-- Footnote -->

---

C2. Did you discuss the experimental setup, including hyperparameter search and best-found hyperparameter values?

C2. 你们是否讨论了实验设置，包括超参数搜索和找到的最佳超参数值？

Appendix A

附录A

C3. Did you report descriptive statistics about your results (e.g., error bars around results, summary statistics from sets of experiments), and is it transparent whether you are reporting the max, mean, etc. or just a single run?

C3. 你们是否报告了关于结果的描述性统计信息（如结果的误差范围、一组实验的汇总统计信息），并且是否明确说明了报告的是最大值、平均值等，还是仅一次运行的结果？

4

C4. If you used existing packages (e.g., for preprocessing, for normalization, or for evaluation), did you report the implementation, model, and parameter settings used (e.g., NLTK, Spacy, ROUGE, etc.)?

C4. 如果你们使用了现有软件包（如用于预处理、归一化或评估），你们是否报告了所使用的实现、模型和参数设置（如NLTK、Spacy、ROUGE等）？

Appendix A

附录A

## D [H] Did you use human annotators (e.g., crowdworkers) or research with human participants? Left blank.

## D [H] 你是否使用了人工标注员（例如众包工作者）或开展了涉及人类参与者的研究？留空。

D1. Did you report the full text of instructions given to participants, including e.g., screenshots, disclaimers of any risks to participants or annotators, etc.?

D1. 你是否报告了提供给参与者的完整说明文本，包括例如截图、对参与者或标注员的任何风险声明等内容？

Not applicable. Left blank.

不适用。留空。

] D2. Did you report information about how you recruited (e.g., crowdsourcing platform, students) and paid participants, and discuss if such payment is adequate given the participants' demographic (e.g., country of residence)?

] D2. 你是否报告了关于如何招募（例如众包平台、学生）和支付参与者报酬的信息，并讨论了考虑到参与者的人口统计学特征（例如居住国家），这种报酬是否足够？

Not applicable. Left blank.

不适用。留空。

D3. Did you discuss whether and how consent was obtained from people whose data you're using/curating? For example, if you collected data via crowdsourcing, did your instructions to crowdworkers explain how the data would be used? Not applicable. Left blank.

D3. 你是否讨论了是否以及如何获得了你所使用/整理数据的相关人员的同意？例如，如果你通过众包方式收集数据，你给众包工作者的说明是否解释了数据将如何使用？不适用。留空。

1 D4. Was the data collection protocol approved (or determined exempt) by an ethics review board? Not applicable. Left blank.

1 D4. 数据收集协议是否获得了伦理审查委员会的批准（或被认定豁免）？不适用。留空。

D5. Did you report the basic demographic and geographic characteristics of the annotator population that is the source of the data?

D5. 你是否报告了作为数据来源的标注人员群体的基本人口统计学和地理特征？

Not applicable. Left blank.

不适用。留空。