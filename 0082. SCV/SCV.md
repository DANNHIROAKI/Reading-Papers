# SCV: Light and Effective Multi-Vector Retrieval with Sequence Compressive Vectors

# SCV：基于序列压缩向量的轻量级高效多向量检索

Cheoneum Park ${}^{1 * }$ ,Seohyeong Jeong ${}^{2}$ ,Minsang Kim ${}^{2}$ ,Kyeongtae Lim ${}^{3}$ Yonghoon Lee ${}^{2}$

朴千任 ${}^{1 * }$，郑瑞英 ${}^{2}$，金民相 ${}^{2}$，林庆泰 ${}^{3}$，李勇勋 ${}^{2}$

${}^{1}$ Hanbat National University

${}^{1}$ 韩巴国立大学

${}^{2}$ SK Telecom

${}^{2}$ SK电信

${}^{3}$ Seoul National University of Science and Technology

${}^{3}$ 首尔科学技术大学

parkce@hanbat.ac.kr, jseoh95@gmail.com

parkce@hanbat.ac.kr, jseoh95@gmail.com

\{minsang.0804, yhlee95\}@sktelecom.com, ktlim@seoultech.ac.kr

\{minsang.0804, yhlee95\}@sktelecom.com, ktlim@seoultech.ac.kr

## Abstract

## 摘要

Recent advances in language models (LMs) has driven progress in information retrieval (IR), effectively extracting semantically relevant information. However, they face challenges in balancing computational costs with deeper query-document interactions. To tackle this, we present two mechanisms: 1) a light and effective multi-vector retrieval with sequence compression vectors, dubbed SCV and 2) coarse-to-fine vector search. The strengths of SCV stems from its application of span compressive vectors for scoring. By employing a non-linear operation to examine every token in the document, we abstract these into a span-level representation. These vectors effectively reduce the document's dimensional representation, enabling the model to engage comprehensively with tokens across the entire collection of documents, rather than the subset retrieved by Approximate Nearest Neighbor. Therefore, our framework performs a coarse single vector search during the inference stage and conducts a fine-grained multi-vector search end-to-end. This approach effectively reduces the cost required for search. We empirically show that SCV achieves the fastest latency compared to other state-of-the-art models and can obtain competitive performance on both in-domain and out-of-domain benchmark datasets.

语言模型（LM）的最新进展推动了信息检索（IR）的发展，能够有效地提取语义相关信息。然而，它们在平衡计算成本和更深入的查询 - 文档交互方面面临挑战。为了解决这个问题，我们提出了两种机制：1）一种基于序列压缩向量的轻量级高效多向量检索方法，称为SCV；2）从粗到细的向量搜索方法。SCV的优势在于其应用跨度压缩向量进行评分。通过采用非线性操作检查文档中的每个标记，我们将这些标记抽象为跨度级别的表示。这些向量有效地降低了文档的维度表示，使模型能够全面处理整个文档集合中的标记，而不仅仅是通过近似最近邻检索到的子集。因此，我们的框架在推理阶段进行粗粒度的单向量搜索，并进行端到端的细粒度多向量搜索。这种方法有效地降低了搜索所需的成本。我们通过实验证明，与其他最先进的模型相比，SCV实现了最快的延迟，并且在域内和域外基准数据集上都能取得有竞争力的性能。

## 1 Introduction

## 1 引言

Information retrieval (IR) is the task of finding a set of relevant documents from an indexed collection for a given query (Manning et al., 2008). Recently, in modern Retrieval-Augmented Generation (RAG) models (Shi et al., 2024; Anantha and Vodianik, 2024; Baek et al., 2023; Jeong et al., 2024), an effective neural IR is crucial for sourc-ing accurate and relevant clues in real-time, significantly improving the quality and contextual appropriateness of generated content. Neural IR can be largely divided into two categories; single-vector retrieval and multi-vector retrieval. The former approach (Karpukhin et al., 2020; Formal et al., 2021) relies on a single vector representation extracted from a document and calculates the relevance score with representations pooled from both queries and documents. In contrast, multi-vector retrieval methods such as ColBERT, GTR, COIL, and CITADEL (Khattab and Zaharia, 2020; Ni et al., 2022; Gao et al., 2021; Li et al., 2023) show promising performance by representing document text as token collections rather than single vectors.

信息检索（IR）是指从索引文档集合中为给定查询找到一组相关文档的任务（Manning等人，2008）。最近，在现代检索增强生成（RAG）模型中（Shi等人，2024；Anantha和Vodianik，2024；Baek等人，2023；Jeong等人，2024），有效的神经信息检索对于实时获取准确和相关的线索至关重要，这显著提高了生成内容的质量和上下文适用性。神经信息检索大致可分为两类：单向量检索和多向量检索。前者（Karpukhin等人，2020；Formal等人，2021）依赖于从文档中提取的单个向量表示，并通过查询和文档的池化表示来计算相关性得分。相比之下，像ColBERT、GTR、COIL和CITADEL这样的多向量检索方法（Khattab和Zaharia，2020；Ni等人，2022；Gao等人，2021；Li等人，2023）通过将文档文本表示为标记集合而非单个向量，展现出了良好的性能。

However, Khattab and Zaharia (2020) requires indexing all tokens in a collection of documents, leading to significant memory and computational burdens. To reduce this burden, a multi-stage retrieval approach is adopted. In the first stage, indexing and searching for relevant documents given the query are performed using approximate nearest neighbor (ANN) (Macdonald and Tonellotto, 2021). In the second stage,the top-k results are output by re-ranking, which is trained based on the extracted documents. Gao et al. (2021); Li et al. (2023) have further improved multi-vector retrieval methods by computing the score between the query and the document using semantically relevant tokens in the document rather than all the tokens, thus eliminating the stage of performing ANN.

然而，Khattab和Zaharia（2020）需要对文档集合中的所有标记进行索引，这导致了显著的内存和计算负担。为了减轻这种负担，采用了多阶段检索方法。在第一阶段，使用近似最近邻（ANN）对给定查询进行相关文档的索引和搜索（Macdonald和Tonellotto，2021）。在第二阶段，通过对提取的文档进行重新排序输出前k个结果。Gao等人（2021）；Li等人（2023）通过使用文档中语义相关的标记而非所有标记来计算查询和文档之间的得分，进一步改进了多向量检索方法，从而省去了进行ANN的阶段。

As another research effort in the stream of multi-vector retrieval approaches, we begin by asking the following questions: 1) Can we make single-stage retrieval possible in a multi-vector retrieval approach? Multi-stage retrieval requires additional ANN training for clustering based on the trained model for queries and documents at the token retrieval stage, the ANN training necessitates optimizing the number of clusters and requires high computing power proportional to the number of tokens in the collection. 2) Can we achieve lightweight indexing while minimizing the loss of contextual information? Prior studies (Gao et al., 2021; Li et al., 2023) have managed to implement lightweight indexing by removing document tokens that do not directly match those in the query and by employing an inverted index. Nevertheless, pruning tokens based solely on exact matches or indexed words limits the ability to leverage the full semantic richness of all document tokens. Although Li et al. (2023) compensates for the loss of semantic context through the use of a routing algorithm, it still demands considerable engineering effort and detailed optimization.

作为多向量检索方法研究方向的另一项工作，我们首先提出以下问题：1）我们能否在多向量检索方法中实现单阶段检索？多阶段检索在标记检索阶段需要基于为查询和文档训练的模型进行额外的ANN聚类训练，ANN训练需要优化聚类数量，并且需要与文档集合中标记数量成正比的高计算能力。2）我们能否在最小化上下文信息损失的同时实现轻量级索引？先前的研究（Gao等人，2021；Li等人，2023）通过去除与查询中标记不直接匹配的文档标记并采用倒排索引，成功实现了轻量级索引。然而，仅基于精确匹配或索引词来修剪标记限制了利用所有文档标记的完整语义丰富性的能力。尽管Li等人（2023）通过使用路由算法弥补了语义上下文的损失，但这仍然需要大量的工程工作和详细的优化。

---

<!-- Footnote -->

*Corresponding Author

*通讯作者

<!-- Footnote -->

---

We introduce a retrieval framework that utilizes a sequence compressive vector (SCV), processed through a coarse-to-fine vector search in end-to-end strategy. Our key idea involves transforming encoded representations of document tokens into span-level embeddings of arbitrary width, thereby compressing the sequence length. As our model performs indexing based on span representations of documents rather than at the token-level retrievers, the index size and the associated computational latency are significantly reduced. Since the lightweight index can perform million-scale retrieval with GPUs, this framework can load single-and multi-vector indexes simultaneously. Accordingly, our framework performs a coarse-to-fine vector search by initially finding a sufficient number of candidate documents with single-vector retrieval and then directly outputting the top-k relevant documents through multi-vector retrieval, using only a trained model without an external retrieval module at inference time.

我们引入了一个检索框架，该框架利用序列压缩向量（SCV），通过端到端策略进行从粗到细的向量搜索。我们的核心思想是将文档标记的编码表示转换为任意宽度的跨度级嵌入，从而压缩序列长度。由于我们的模型基于文档的跨度表示进行索引，而不是在标记级检索器上进行，因此索引大小和相关的计算延迟显著降低。由于轻量级索引可以使用GPU进行百万级别的检索，该框架可以同时加载单向量和多向量索引。因此，我们的框架通过以下方式进行从粗到细的向量搜索：首先使用单向量检索找到足够数量的候选文档，然后通过多向量检索直接输出前k个相关文档，在推理时仅使用训练好的模型，无需外部检索模块。

Additionally, we enhance our model by employing reranking using a cross-encoder (Urbanek et al., 2019). Our experimental results show that the proposed method outpaces the inverted list approach by a factor of 1.1. The SCV model delivers performance comparable to ColBERT and sets a new standard for the base-sized models with reranking. Our contributions can be summarized in threefold:

此外，我们通过使用交叉编码器进行重排序来增强我们的模型（Urbanek等人，2019）。我们的实验结果表明，所提出的方法比倒排表方法快1.1倍。SCV模型的性能与ColBERT相当，并为使用重排序的基础大小模型设定了新的标准。我们的贡献可以总结为以下三点：

- We introduce an efficient multi-vector retriever that utilizes tokens compression to span representations.

- 我们引入了一种高效的多向量检索器，该检索器利用标记压缩来实现跨度表示。

- The coarse-to-fine vector search framework can process through an end-to-end strategy in a single stage.

- 从粗到细的向量搜索框架可以在单个阶段通过端到端策略进行处理。

- Our approach is 207 times faster than ColBERT and 4.6 times faster than CITADEL.

- 我们的方法比ColBERT快207倍，比CITADEL快4.6倍。

<!-- Media -->

<!-- figureText: MaxSim SCV Encoder Query Encoder ${q}_{n}$ -->

<img src="https://cdn.noedgeai.com/0195afa2-270e-7c66-ae49-4113ee3e4312_1.jpg?x=844&y=192&w=616&h=356&r=0"/>

Figure 1: Sequence Compressive Vectors architecture overview.

图1：序列压缩向量架构概述。

<!-- Media -->

## 2 Method

## 2 方法

### 2.1 Preliminaries

### 2.1 预备知识

The input query is denoted as $Q = \left\{  {{q}_{1},{q}_{2},\ldots ,{q}_{n}}\right\}$ , and the document as $D = \left\{  {{d}_{1},{d}_{2},\ldots ,{d}_{m}}\right\}$ ,with the span sequence generated from document tokens represented by $S = \left\{  {{s}_{1},{s}_{2},\ldots ,{s}_{l}}\right\}$ . The $n$ , $m$ ,and $l$ are the length of the query,document, and span, respectively. Span sequence is produced using a sliding window algorithm, which maintains context information by allowing overlap of adjacent tokens when extracting tokens within the window. The width of the window is denoted by $W \in  \{ 2,4,8,{16}\}$ ,and the interval at which the window moves across tokens, skipping them at a fixed rate,is referred to as $0 \leq$ rate $\leq  1$ ,rate $\in  \mathbb{R}$ . The overall size of the span sequence is determined by the following equation:

输入查询表示为 $Q = \left\{  {{q}_{1},{q}_{2},\ldots ,{q}_{n}}\right\}$，文档表示为 $D = \left\{  {{d}_{1},{d}_{2},\ldots ,{d}_{m}}\right\}$，由文档标记生成的跨度序列表示为 $S = \left\{  {{s}_{1},{s}_{2},\ldots ,{s}_{l}}\right\}$。$n$、$m$ 和 $l$ 分别是查询、文档和跨度的长度。跨度序列使用滑动窗口算法生成，该算法在提取窗口内的标记时允许相邻标记重叠，以保留上下文信息。窗口的宽度表示为 $W \in  \{ 2,4,8,{16}\}$，窗口在标记上移动并以固定速率跳过标记的间隔称为 $0 \leq$ 速率 $\leq  1$、速率 $\in  \mathbb{R}$。跨度序列的总体大小由以下方程确定：

$$
l = \left\lceil  {\frac{m - W}{\left( {1 - \text{ rate }}\right) W} + 1}\right\rceil   \tag{1}
$$

### 2.2 Model Structure

### 2.2 模型结构

SCV retriever is a multi-vector retrieval model as illustrated in Figure 1. It compresses token information of the document by extracting fixed length spans and allowing the model to train span embed-dings. Pre-trained language model (PLM) (Devlin et al., 2019; Sanh et al., 2020), is used to encode the input sequence of the query, ${\mathbf{h}}_{{q}_{i}} = \operatorname{PLM}\left( {q}_{i}\right)$ , and the document, ${\mathbf{h}}_{{d}_{j}} = \operatorname{PLM}\left( {d}_{j}\right)$ ,where the language encoders are shared. Special tokens of [Q] and [D] are prefixed to the query and the document, respectively, to differentiate between query and document inputs. Given a document token vector, ${\mathbf{h}}_{{d}_{j}}$ ,the span level representation is computed as ${\mathbf{h}}_{s} = \phi \left( {\mathbf{h}}_{d}\right)$ ,where $\phi$ is a span compressive vector operation. We discuss this operation further in detail in Chapter 2.3.

如图1所示，SCV检索器是一个多向量检索模型。它通过提取固定长度的跨度并让模型训练跨度嵌入来压缩文档的标记信息。预训练语言模型（PLM）（Devlin等人，2019；Sanh等人，2020）用于对查询 ${\mathbf{h}}_{{q}_{i}} = \operatorname{PLM}\left( {q}_{i}\right)$ 和文档 ${\mathbf{h}}_{{d}_{j}} = \operatorname{PLM}\left( {d}_{j}\right)$ 的输入序列进行编码，其中语言编码器是共享的。分别在查询和文档前添加特殊标记 [Q] 和 [D]，以区分查询和文档输入。给定文档标记向量 ${\mathbf{h}}_{{d}_{j}}$，跨度级表示计算为 ${\mathbf{h}}_{s} = \phi \left( {\mathbf{h}}_{d}\right)$，其中 $\phi$ 是一个跨度压缩向量操作。我们将在第2.3章进一步详细讨论此操作。

<!-- Media -->

<!-- figureText: Sejong the Great Great was the inventor of Hangul ↑ ↑ Passage Encoder ... inventor of Hangul ↑ Sejong the -->

<img src="https://cdn.noedgeai.com/0195afa2-270e-7c66-ae49-4113ee3e4312_2.jpg?x=208&y=197&w=593&h=404&r=0"/>

Figure 2: SCV Encoder for Span Representation.

图2：用于跨度表示的SCV编码器。

<!-- Media -->

Our model leverages the full contextualized representations of query tokens and document spans. Within the SCV encoder, the compressed document span representations engage with the query token vector via a MaxSim (Khattab and Zaharia, 2020), which is used to calculate the document score. This process is articulated in the equation below:

我们的模型利用了查询标记和文档跨度的完整上下文表示。在SCV编码器中，压缩后的文档跨度表示通过MaxSim（Khattab和Zaharia，2020）与查询标记向量进行交互，该方法用于计算文档得分。此过程由以下方程表示：

$$
f\left( {Q,S}\right)  = \mathop{\sum }\limits_{{i = 1}}^{n}\mathop{\max }\limits_{{k = 1,\ldots ,l}}{\mathbf{h}}_{{q}_{i}}^{\top }{\mathbf{h}}_{{s}_{k}} \tag{2}
$$

where ${\mathbf{h}}_{{q}_{i}}$ and ${\mathbf{h}}_{{s}_{k}}$ denote the last-layer contex-tualized token embeddings of a query and span embeddings of a document.

其中 ${\mathbf{h}}_{{q}_{i}}$ 和 ${\mathbf{h}}_{{s}_{k}}$ 分别表示查询的最后一层上下文标记嵌入和文档的跨度嵌入。

Popular retrieval models (Gao et al., 2021; Li et al., 2023) use vectors of a CLS special token in query and document, respectively, to provide high level semantic matching between the query and document. We further leverage the [CLS] vector similarity, representing the aggregate sequence of both the query and document as follows.

流行的检索模型（Gao等人，2021年；Li等人，2023年）分别使用查询和文档中CLS特殊标记的向量，以实现查询和文档之间的高级语义匹配。我们进一步利用[CLS]向量相似度，将查询和文档的聚合序列表示如下。

$$
{\mathbf{v}}_{{q}_{cls}} = {\mathbf{W}}_{cls}{\mathbf{h}}_{q} + {\mathbf{b}}_{cls} \tag{3}
$$

$$
{\mathbf{v}}_{{d}_{cls}} = {\mathbf{W}}_{cls}{\mathbf{h}}_{d} + {\mathbf{b}}_{cls}
$$

### 2.3 Sequence Compressive Vectors

### 2.3 序列压缩向量

We introduce an end-to-end retrieval framework designed for multi-vector retrieval, which compresses token sequences from documents as depicted in Figure 2. For example, the process begins with the input sequence being encoded with contextu-alized token representations through an encoder. With $W = 3$ ,the model utilizes the sliding window method to extract token representations, subsequently compressing these into span-level information through diverse pooling techniques.

我们引入了一个专为多向量检索设计的端到端检索框架，该框架如图2所示，将文档中的标记序列进行压缩。例如，该过程首先通过编码器使用上下文相关的标记表示对输入序列进行编码。利用$W = 3$，模型采用滑动窗口方法提取标记表示，随后通过各种池化技术将这些表示压缩为跨度级别的信息。

The core idea of SCV lies in the span representation, ${\mathbf{h}}_{s}$ ,with the compression ratio influenced by $W$ and rate,as outlined in Equation 1. A feed-forward neural network with an activation function is used to encode lexical information. This encoded information is subsequently concatenated with pooled vectors from document tokens, resulting in the span representation, ${\mathbf{h}}_{s}$ ,for span $k$ :

序列压缩向量（SCV）的核心思想在于跨度表示${\mathbf{h}}_{s}$，其压缩比受$W$和速率的影响，如公式1所示。使用带有激活函数的前馈神经网络对词汇信息进行编码。随后，将此编码信息与文档标记的池化向量进行拼接，得到跨度$k$的跨度表示${\mathbf{h}}_{s}$：

$$
\phi \left( {\mathbf{h}}_{d}\right)  = \operatorname{GELU}\left( {\operatorname{FFNN}\left( {v}_{\text{comp }}\right) }\right) 
$$

$$
{v}_{\text{comp }} = \left\lbrack  {{\mathbf{g}}^{s};{\mathbf{g}}^{e};{\mathbf{g}}^{m};{\mathbf{g}}^{c};{\mathbf{h}}_{{d}_{\left\lbrack  j : j + W\right\rbrack  }}^{\text{sum }};{\mathbf{h}}_{{d}_{\left\lbrack  j : j + W\right\rbrack  }}^{\max };\alpha }\right\rbrack  
$$

$$
{\mathbf{g}}^{s} = \operatorname{FFNN}\left( {\mathbf{h}}_{{d}_{j}}\right) 
$$

$$
{\mathbf{g}}^{e} = \operatorname{FFNN}\left( {\mathbf{h}}_{{d}_{\left\lbrack  j + W\right\rbrack  }}\right) 
$$

$$
{\mathbf{g}}^{m} = {\mathbf{g}}^{s} \circ  {\mathbf{g}}^{e}
$$

$$
{\mathbf{g}}^{c} = \operatorname{GELU}\left( {\operatorname{FFNN}\left( \left\lbrack  {{\mathbf{g}}^{s};{\mathbf{g}}^{e}}\right\rbrack  \right) }\right) 
$$

$$
\alpha  = \max \left( {\operatorname{attn}\left( {{\mathbf{h}}_{{d}_{\left\lbrack  j : j + W\right\rbrack  }},{\mathbf{h}}_{{d}_{\left\lbrack  j : j + W\right\rbrack  }}}\right) {\mathbf{h}}_{{d}_{\left\lbrack  j : j + W\right\rbrack  }}}\right) 
$$

(4)

where $\circ$ denotes element-wise multiplication, ${\mathbf{h}}^{\text{sum }}$ and ${\mathbf{h}}^{\max }$ are pooled vectors for sum and max pooling,respectively. $\alpha$ represents a salient word using an attention mechanism (Bahdanau et al., 2015), which is highlighted for the most relevant parts of the sequence, and max pooling over words in each span. Max operation involves taking the most important feature (Kim, 2014) and sum operation captures the global intensity of features across the span is relevant (Tian et al., 2017). The above formula generalizes the span representation that includes the start and end boundary representations of the span, as well as the representation of salient words within the span.

其中$\circ$表示逐元素相乘，${\mathbf{h}}^{\text{sum }}$和${\mathbf{h}}^{\max }$分别是求和池化和最大池化的池化向量。$\alpha$使用注意力机制（Bahdanau等人，2015年）表示一个显著词，该机制会突出序列中最相关的部分，并对每个跨度中的词进行最大池化。最大操作涉及选取最重要的特征（Kim，2014年），求和操作则捕获整个跨度中特征的全局强度（Tian等人，2017年）。上述公式概括了跨度表示，其中包括跨度的起始和结束边界表示，以及跨度内显著词的表示。

### 2.4 Training

### 2.4 训练

We train SCV using loss of negative log likelihood based on similarity score of $f\left( {Q,S}\right)$ of Equation 2 for a query $q$ ,a positive sample ${d}^{ + }$ ,and a set of negative samples $N = \left\{  {{d}_{1}^{ - },{d}_{2}^{ - },\ldots ,{d}_{B}^{ - }}\right\}$ , where $B$ is the batch size. Our strategy involves contrastive learning with a focus on negative sample utilization. We utilize in-batch negatives (ib) (Karpukhin et al., 2020), pre-batch negatives (pb) (Kim et al., 2022), and hard negatives (hb) generated by BM25 (Robertson and Zaragoza, 2009) that are widely used in the retrieval tasks.

我们基于公式2中$f\left( {Q,S}\right)$的相似度得分，使用负对数似然损失来训练序列压缩向量（SCV），针对查询$q$、正样本${d}^{ + }$和一组负样本$N = \left\{  {{d}_{1}^{ - },{d}_{2}^{ - },\ldots ,{d}_{B}^{ - }}\right\}$进行训练，其中$B$是批量大小。我们的策略涉及对比学习，重点在于负样本的利用。我们使用批内负样本（ib）（Karpukhin等人，2020年）、批前负样本（pb）（Kim等人，2022年）以及由BM25（Robertson和Zaragoza，2009年）生成的难负样本（hb），这些在检索任务中被广泛使用。

$$
\mathcal{L} =  - \log \frac{\exp \left( {f\left( {q,{d}^{ + }}\right) }\right) }{\exp \left( {f\left( {q,{d}^{ + }}\right) }\right)  + \mathop{\sum }\limits_{{b \in  {N}_{\mathrm{{ib}}} \cup  {N}_{\mathrm{{pb}}} \cup  {N}_{\mathrm{{hb}}}}}\exp \left( {f\left( {q,{d}_{b}^{ - }}\right) }\right) }
$$

(5)

where the numbers of negatives are $\left| {N}_{\mathrm{{ib}}}\right|  = B - 1$ , $\left| {N}_{\mathrm{{pb}}}\right|  = B$ ,and $\left| {N}_{\mathrm{{hb}}}\right|  = H,H$ is a hyper-parameter for the number of hard negatives.

其中负样本的数量分别为$\left| {N}_{\mathrm{{ib}}}\right|  = B - 1$、$\left| {N}_{\mathrm{{pb}}}\right|  = B$，$\left| {N}_{\mathrm{{hb}}}\right|  = H,H$是难负样本数量的超参数。

We enhance the training of span representation-based retrieval scores between queries and documents by employing multi-task learning with the single vector retriever. Multi-vector retrieval model calculates SCV loss ${\mathcal{L}}_{SCV}$ and token-level all-to-all retriever loss ${\mathcal{L}}_{\text{tok }}$ ,respectively,according to Equation 5. Meanwhile, the single vector retrieval computes its loss ${\mathcal{L}}_{cls}$ by performing a dot-product with the score from Equation 3, and the total loss is obtained by summing all contributions.

我们通过使用单向量检索器进行多任务学习，来增强基于跨度表示的查询和文档之间检索得分的训练。多向量检索模型分别根据公式5计算序列压缩向量（SCV）损失${\mathcal{L}}_{SCV}$和标记级全对全检索器损失${\mathcal{L}}_{\text{tok }}$。同时，单向量检索通过与公式3的得分进行点积计算其损失${\mathcal{L}}_{cls}$，总损失通过将所有贡献相加得到。

The final loss equation is as follows:

最终的损失方程如下：

$$
\mathcal{L} = {\mathcal{L}}_{SCV} + {\mathcal{L}}_{\text{tok }} + \alpha {\mathcal{L}}_{\text{cls }} \tag{6}
$$

where $\alpha$ is used to scale loss of the single vector retriever.

其中$\alpha$用于缩放单向量检索器的损失。

In addition, we augment question synthetic data by prompting MS MARCO passages to GPT-4 ${}^{1}$ to enhance representations of span embeddings. Question generation is sequentially conducted to the passages,producing approximately ${180}\mathrm{k}$ questions, while ensuring that the development set of MS MARCO remains unseen. We perform lexical filtering and cleaning for the generated questions.

此外，我们通过向GPT - 4 ${}^{1}$输入MS MARCO段落来扩充问题合成数据，以增强跨度嵌入的表示。对段落依次进行问题生成，大约生成${180}\mathrm{k}$个问题，同时确保MS MARCO的开发集不被使用。我们对生成的问题进行词汇过滤和清理。

### 2.5 Coarse-to-fine Vector Search

### 2.5 粗到细的向量搜索

Even though sequence compression reduces the storage requirements, searching documents still results in increased computation proportional to the index size, leading to latency. To facilitate faster search times, we execute an coarse-to-fine vector search using a single model, as follows: The SCV model calculates dot product using the CLS token vectors for queries and documents and conducts multi-task learning. During inference time, based on the CLS token vectors trained in this manner, we first perform single-vector retrieval to extract the top- $N$ documents,with $N \in  \{ {10000},{20000},{50000},{100000}\}$ ,followed by multi-vector retrieval using the extracted document vectors to produce the top- $k$ final search results. Our framework is end-to-end process and light and fast as it performs model scoring without the need for the external retrieving such as ANN. Following the aforementioned process, we optionally apply re-ranking to enhance the search quality.

尽管序列压缩降低了存储需求，但搜索文档仍会导致与索引大小成比例的计算量增加，从而产生延迟。为了加快搜索速度，我们使用单一模型执行从粗到细的向量搜索，具体如下：SCV模型（序列压缩向量模型，Sequence Compression Vector model）使用查询和文档的CLS标记向量（分类标记向量，Classification token vectors）计算点积，并进行多任务学习。在推理阶段，基于以这种方式训练的CLS标记向量，我们首先进行单向量检索，提取前 $N$ 个文档（其中 $N \in  \{ {10000},{20000},{50000},{100000}\}$ ），然后使用提取的文档向量进行多向量检索，以生成前 $k$ 个最终搜索结果。我们的框架是一个端到端的过程，轻量且快速，因为它在进行模型评分时无需像近似最近邻搜索（ANN，Approximate Nearest Neighbor）这样的外部检索。按照上述过程，我们可以选择应用重排序来提高搜索质量。

<!-- Media -->

<table><tr><td rowspan="2">Models</td><td colspan="2">TREC DL 19</td><td rowspan="2">Index (GB)</td><td rowspan="2">Latency (ms/query)</td></tr><tr><td>nDCG@10</td><td>R@1k</td></tr><tr><td colspan="5">Models trained with only BM25 hard negatives</td></tr><tr><td>BM25</td><td>0.506</td><td>0.739</td><td>0.67</td><td>✘</td></tr><tr><td>DPR-768</td><td>0.611</td><td>0.742</td><td>26</td><td>1.28</td></tr><tr><td>COIL-tok</td><td>0.660</td><td>0.809</td><td>52.5</td><td>46.8</td></tr><tr><td>ColBERT</td><td>0.694</td><td>0.830</td><td>154</td><td>178</td></tr><tr><td>CITADEL</td><td>0.687</td><td>0.829</td><td>78.3</td><td>3.95</td></tr><tr><td>SCV</td><td>0.645</td><td>0.712</td><td>30</td><td>0.86</td></tr><tr><td colspan="5">Models trained with further methods</td></tr><tr><td>coCondenser</td><td>0.674</td><td>0.820</td><td>26</td><td>1.28</td></tr><tr><td>ColBERT-v2</td><td>0.744</td><td>0.882</td><td>29</td><td>122</td></tr><tr><td>ColBERT-PLAID</td><td>0.744</td><td>0.882</td><td>22.1</td><td>55</td></tr><tr><td>CITADEL+</td><td>0.703</td><td>0.830</td><td>26.7</td><td>3.21</td></tr></table>

<table><tbody><tr><td rowspan="2">模型</td><td colspan="2">TREC DL 19（文本检索会议深度学习任务2019）</td><td rowspan="2">索引（GB）</td><td rowspan="2">延迟（毫秒/查询）</td></tr><tr><td>归一化折损累积增益@10（nDCG@10）</td><td>召回率@1000（R@1k）</td></tr><tr><td colspan="5">仅使用BM25硬负样本训练的模型</td></tr><tr><td>二元独立模型25（BM25）</td><td>0.506</td><td>0.739</td><td>0.67</td><td>✘</td></tr><tr><td>密集段落检索器-768（DPR-768）</td><td>0.611</td><td>0.742</td><td>26</td><td>1.28</td></tr><tr><td>基于上下文的有序交互学习-词元（COIL-tok）</td><td>0.660</td><td>0.809</td><td>52.5</td><td>46.8</td></tr><tr><td>列伯特（ColBERT）</td><td>0.694</td><td>0.830</td><td>154</td><td>178</td></tr><tr><td>西塔德尔（CITADEL）</td><td>0.687</td><td>0.829</td><td>78.3</td><td>3.95</td></tr><tr><td>单向量上下文（SCV）</td><td>0.645</td><td>0.712</td><td>30</td><td>0.86</td></tr><tr><td colspan="5">使用其他方法训练的模型</td></tr><tr><td>协同凝聚器（coCondenser）</td><td>0.674</td><td>0.820</td><td>26</td><td>1.28</td></tr><tr><td>列伯特v2（ColBERT-v2）</td><td>0.744</td><td>0.882</td><td>29</td><td>122</td></tr><tr><td>列伯特-普莱德（ColBERT-PLAID）</td><td>0.744</td><td>0.882</td><td>22.1</td><td>55</td></tr><tr><td>西塔德尔+（CITADEL+）</td><td>0.703</td><td>0.830</td><td>26.7</td><td>3.21</td></tr></tbody></table>

Table 1: In-domain evaluation on TREC DL 2019. Performance reference is made to CITADEL, and latency includes the total time for query encoding and search.

表1：在TREC DL 2019上的领域内评估。性能参考以CITADEL为准，延迟包括查询编码和搜索的总时间。

<!-- Media -->

## 3 Experimental Results

## 3 实验结果

We train our model using the passage ranking dataset from MS MARCO ${}^{2}$ . For in-domain evaluation, we use the MS MARCO development set and TREC DL 2019, and for out-of-domain evaluation, we assess performance on the BEIR benchmark (Thakur et al., 2021). The MS MARCO development set contains 6,980 queries, while the TREC DL 2019 evaluation set provides annotations for 43 queries. The BEIR benchmark comprises 18 retrieval tasks across 9 domains, and we evaluate using 13 datasets following previous studies (San-thanam et al., 2022a; Li et al., 2023).

我们使用来自MS MARCO ${}^{2}$ 的段落排序数据集来训练我们的模型。对于领域内评估，我们使用MS MARCO开发集和TREC DL 2019；对于领域外评估，我们在BEIR基准测试（Thakur等人，2021）上评估性能。MS MARCO开发集包含6980个查询，而TREC DL 2019评估集为43个查询提供了注释。BEIR基准测试涵盖9个领域的18个检索任务，我们按照先前的研究（San-thanam等人，2022a；Li等人，2023）使用13个数据集进行评估。

As our evaluation metric, we employ nDCG@10, and Recall@1000 for MS MARCO, along with nDCG@10 for BEIR. We use a script of BEIR ${}^{3}$ to evaluate datasets.

作为评估指标，我们对MS MARCO采用nDCG@10和Recall@1000，对BEIR采用nDCG@10。我们使用BEIR ${}^{3}$ 的脚本评估数据集。

Experimental settings We initialize using DistilBERT-base (Sanh et al., 2019) as our backbone model. The experimental environment for training, indexing, and retrieval utilizes a Tesla A100 GPU, with an optimized batch size set to 630 . Evaluation during training is conducted with in-batch predictions of size $1\mathrm{k}$ ,and checkpoints are saved at the step showing the best performance. The SCV model is trained using the AdamW (Loshchilov and Hutter, 2017) optimizer, with a learning rate of ${5e} - 5$ and linear scheduling. Hard negatives are sampled from the top 1000 BM25 results (Gao et al., 2023), and each query uses 1 positive and 1 negative sample. The dimension size for both the CLS token layer and the SCV output layer is set to 128 . During training, the width of span embeddings(W)is set to 8,while for indexing, it is adjusted to 16 for MS MARCO and remains at 8 for BEIR. The sliding overlap rate (rate) is 0.2 , the dimension size for span embed-dings is 384, and the dropout rate is set to 0.1 . In Chapter 2.5, it is mentioned that inference is performed with $N$ set to ${10}\mathrm{\;k}$ . All hyper-parameters are optimized.

实验设置 我们使用DistilBERT-base（Sanh等人，2019）作为骨干模型进行初始化。训练、索引和检索的实验环境使用Tesla A100 GPU，优化后的批量大小设置为630。训练期间的评估通过大小为 $1\mathrm{k}$ 的批次内预测进行，并在表现最佳的步骤保存检查点。SCV模型使用AdamW（Loshchilov和Hutter，2017）优化器进行训练，学习率为 ${5e} - 5$ 并采用线性调度。硬负样本从BM25的前1000个结果中采样（Gao等人，2023），每个查询使用1个正样本和1个负样本。CLS标记层和SCV输出层的维度大小均设置为128。训练期间，跨度嵌入的宽度（W）设置为8，而在索引时，MS MARCO调整为16，BEIR保持为8。滑动重叠率（rate）为0.2，跨度嵌入的维度大小为384，丢弃率设置为0.1。在第2.5章中提到，推理时 $N$ 设置为 ${10}\mathrm{\;k}$ 。所有超参数均已优化。

---

<!-- Footnote -->

${}^{2}$ https://github.com/microsoft/MS MARCO-Passage-Ranking

${}^{2}$ https://github.com/microsoft/MS MARCO-Passage-Ranking

${}^{3}$ https://github.com/beir-cellar/beir

${}^{3}$ https://github.com/beir-cellar/beir

${}^{1}$ https://platform.openai.com/docs/models/gpt-4-and-gpt- 4-turbo

${}^{1}$ https://platform.openai.com/docs/models/gpt-4-and-gpt- 4-turbo

<!-- Footnote -->

---

<!-- Media -->

<table><tr><td>Methods</td><td>AA</td><td>CF</td><td>DB</td><td>Fe</td><td>FQ</td><td>HQ</td><td>NF</td><td>NQ</td><td>Qu</td><td>SF</td><td>SD</td><td>TC</td><td>T2</td><td>Avg.</td></tr><tr><td>BM25</td><td>0.315</td><td>0.213</td><td>0.313</td><td>0.753</td><td>0.236</td><td>0.603</td><td>0.325</td><td>0.329</td><td>0.789</td><td>0.665</td><td>0.158</td><td>0.656</td><td>0.367</td><td>0.440</td></tr><tr><td>DPR-768</td><td>0.323</td><td>0.167</td><td>0.295</td><td>0.651</td><td>0.224</td><td>0.441</td><td>0.244</td><td>0.410</td><td>0.750</td><td>0.479</td><td>0.103</td><td>0.604</td><td>0.185</td><td>0.375</td></tr><tr><td>ColBERT</td><td>0.233</td><td>0.184</td><td>0.392</td><td>0.771</td><td>0.317</td><td>0.593</td><td>0.305</td><td>0.524</td><td>0.854</td><td>0.671</td><td>0.165</td><td>0.677</td><td>0.202</td><td>0.453</td></tr><tr><td>GTR</td><td>0.511</td><td>0.215</td><td>0.392</td><td>0.660</td><td>0.349</td><td>0.535</td><td>0.308</td><td>0.495</td><td>0.881</td><td>0.600</td><td>0.149</td><td>0.539</td><td>0.215</td><td>0.452</td></tr><tr><td>CITADEL</td><td>0.503</td><td>0.191</td><td>0.406</td><td>0.784</td><td>0.298</td><td>0.653</td><td>0.324</td><td>0.510</td><td>0.844</td><td>0.674</td><td>0.152</td><td>0.687</td><td>0.294</td><td>0.486</td></tr><tr><td>SCV</td><td>0.464</td><td>0.139</td><td>0.351</td><td>0.675</td><td>0.272</td><td>0.535</td><td>0.315</td><td>0.425</td><td>0.774</td><td>0.656</td><td>0.135</td><td>0.668</td><td>0.262</td><td>0.436</td></tr></table>

<table><tbody><tr><td>方法</td><td>AA</td><td>CF</td><td>DB</td><td>铁（Fe）</td><td>FQ</td><td>HQ</td><td>NF</td><td>NQ</td><td>Qu</td><td>SF</td><td>SD</td><td>TC</td><td>T2</td><td>平均值（Avg.）</td></tr><tr><td>BM25</td><td>0.315</td><td>0.213</td><td>0.313</td><td>0.753</td><td>0.236</td><td>0.603</td><td>0.325</td><td>0.329</td><td>0.789</td><td>0.665</td><td>0.158</td><td>0.656</td><td>0.367</td><td>0.440</td></tr><tr><td>DPR - 768</td><td>0.323</td><td>0.167</td><td>0.295</td><td>0.651</td><td>0.224</td><td>0.441</td><td>0.244</td><td>0.410</td><td>0.750</td><td>0.479</td><td>0.103</td><td>0.604</td><td>0.185</td><td>0.375</td></tr><tr><td>ColBERT</td><td>0.233</td><td>0.184</td><td>0.392</td><td>0.771</td><td>0.317</td><td>0.593</td><td>0.305</td><td>0.524</td><td>0.854</td><td>0.671</td><td>0.165</td><td>0.677</td><td>0.202</td><td>0.453</td></tr><tr><td>GTR</td><td>0.511</td><td>0.215</td><td>0.392</td><td>0.660</td><td>0.349</td><td>0.535</td><td>0.308</td><td>0.495</td><td>0.881</td><td>0.600</td><td>0.149</td><td>0.539</td><td>0.215</td><td>0.452</td></tr><tr><td>CITADEL</td><td>0.503</td><td>0.191</td><td>0.406</td><td>0.784</td><td>0.298</td><td>0.653</td><td>0.324</td><td>0.510</td><td>0.844</td><td>0.674</td><td>0.152</td><td>0.687</td><td>0.294</td><td>0.486</td></tr><tr><td>SCV</td><td>0.464</td><td>0.139</td><td>0.351</td><td>0.675</td><td>0.272</td><td>0.535</td><td>0.315</td><td>0.425</td><td>0.774</td><td>0.656</td><td>0.135</td><td>0.668</td><td>0.262</td><td>0.436</td></tr></tbody></table>

Table 2: nDCG@10 on BEIR. Dataset Legend (Li et al., 2023): AA=ArguAna, CF=Climate-FEVER, DB=DBPedia, Fe=FEVER, FQ=FiQA, HQ=HotpotQA, NF=NFCorpus, NQ=NaturalQuestions, Qu=Quora, SF=SciFact, SD=SCIDOCS, TC=TREC-COVID, T2=Touché.

表2：BEIR数据集上的nDCG@10。数据集图例（Li等人，2023）：AA=论点分析数据集（ArguAna），CF=气候事实核查数据集（Climate-FEVER），DB=维基百科数据集（DBPedia），Fe=事实核查数据集（FEVER），FQ=金融问答数据集（FiQA），HQ=火锅问答数据集（HotpotQA），NF=自然语言处理语料库（NFCorpus），NQ=自然问题数据集（NaturalQuestions），Qu=Quora问答数据集，SF=科学事实核查数据集（SciFact），SD=科学文档数据集（SCIDOCS），TC=TREC新冠数据集（TREC-COVID），T2=Touché数据集。

<!-- Media -->

### 3.1 Results

### 3.1 结果

Results on MS MARCO Table 1 presents the performance on in-domain datasets along with index storage size and search latency. The comparison models utilize BM25 hard negatives or include further pre-training, hard-negative mining, and distillation for training, such as coCondenser (Gao and Callan, 2022), ColBERT-v2 (Santhanam et al., 2022c), and ColBERT-PLAID (Santhanam et al., 2022b). The experimental results show that while our SCV method achieves comparable performance to other models on TREC DL 19 using only BM25 hard negatives. In contrast, SCV's index size is a more compact 30GB, close to DPR-768, and reduces the size by approximately 5.13 times compared to ColBERT.

MS MARCO数据集上的结果 表1展示了在领域内数据集上的性能，以及索引存储大小和搜索延迟。对比模型使用了BM25硬负样本，或者在训练中进行了进一步的预训练、硬负样本挖掘和蒸馏，例如协同冷凝器模型（coCondenser，Gao和Callan，2022）、第二代ColBERT模型（ColBERT-v2，Santhanam等人，2022c）和ColBERT-PLAID模型（Santhanam等人，2022b）。实验结果表明，我们的SCV方法仅使用BM25硬负样本，在TREC DL 19数据集上取得了与其他模型相当的性能。相比之下，SCV的索引大小更紧凑，仅为30GB，接近DPR - 768模型，并且与ColBERT模型相比，大小缩小了约5.13倍。

SCV achieves a latency of ${0.86}\mathrm{\;{ms}}$ /query,making it the fastest among the multi-vector retrieval models, and approximately 3.7 times, 64 times, 141.8 times, and 207 times faster than CITADEL+, ColBERT-PLAID, ColBERT-v2, and ColBERT, respectively. Furthermore, our framework is approximately 1.5 times faster than the single vector retriever DPR-768. Most RAG or question answering pipeline services use single vector retriever due to processing speed issues. We expect that our approach can provide a faster and more accurate retrieval model for these systems.

SCV的查询延迟为${0.86}\mathrm{\;{ms}}$，使其成为多向量检索模型中最快的，分别比CITADEL+、ColBERT - PLAID、ColBERT - v2和ColBERT快约3.7倍、64倍、141.8倍和207倍。此外，我们的框架比单向量检索器DPR - 768快约1.5倍。由于处理速度问题，大多数检索增强生成（RAG）或问答流水线服务使用单向量检索器。我们期望我们的方法能够为这些系统提供一个更快、更准确的检索模型。

<!-- Media -->

<table><tr><td>Models</td><td>Size</td><td>TREC DL 19 nDCG@10</td></tr><tr><td colspan="3">Reranking models</td></tr><tr><td>monoBERT (Nogueira et al., 2019)</td><td>110M</td><td>0.723</td></tr><tr><td>SimLM (Wang et al., 2023)</td><td>110M</td><td>0.741</td></tr><tr><td>ListT5 (Yoon et al., 2024)</td><td>220M</td><td>0.718</td></tr><tr><td>SCV+CE</td><td>220M</td><td>0.744</td></tr><tr><td colspan="3">Ranking models with LLM</td></tr><tr><td>RankLLaMA (Ma et al., 2024)</td><td>7B</td><td>0.756</td></tr><tr><td>RankLLaMA</td><td>13B</td><td>0.760</td></tr><tr><td>RankVicuna (Pradeep et al., 2023)</td><td>7B</td><td>0.668</td></tr><tr><td>PRP (Qin et al., 2024)</td><td>20B</td><td>0.727</td></tr></table>

<table><tbody><tr><td>模型</td><td>大小</td><td>TREC DL 19 归一化折损累积增益@10（TREC DL 19 nDCG@10）</td></tr><tr><td colspan="3">重排序模型</td></tr><tr><td>单BERT模型（monoBERT，诺盖拉等人，2019年）</td><td>110M</td><td>0.723</td></tr><tr><td>SimLM模型（王等人，2023年）</td><td>110M</td><td>0.741</td></tr><tr><td>ListT5模型（尹等人，2024年）</td><td>220M</td><td>0.718</td></tr><tr><td>SCV+CE</td><td>220M</td><td>0.744</td></tr><tr><td colspan="3">基于大语言模型的排序模型</td></tr><tr><td>RankLLaMA模型（马等人，2024年）</td><td>7B</td><td>0.756</td></tr><tr><td>RankLLaMA模型</td><td>13B</td><td>0.760</td></tr><tr><td>RankVicuna模型（普拉迪普等人，2023年）</td><td>7B</td><td>0.668</td></tr><tr><td>PRP模型（秦等人，2024年）</td><td>20B</td><td>0.727</td></tr></tbody></table>

Table 3: In-domain Reranking evaluation on TREC DL 2019. Performance reference is made to RankLLaMA.

表3：TREC DL 2019的领域内重排序评估。性能参考为RankLLaMA。

<!-- Media -->

Results on BEIR We conduct an out-of-domain evaluation using the BEIR benchmark. Table 2 presents the zero-shot evaluation results on BEIR for retrieval models, including those extended with re-ranking. The experimental outcomes demonstrate that the SCV significantly outperforms a single-vector retriever and is competitive with multi-vector retrievers. SCV utilizes a compressed representation of span to generate multi-vector from token sequences, we expect its performance to fall between that of DPR and ColBERT. According to the experimental results, SCV shows scores close to the ColBERT, as we expected and specifically achieves higher scores on the AA, NF, and T2 datasets.

BEIR上的结果 我们使用BEIR基准进行了领域外评估。表2展示了BEIR上检索模型的零样本评估结果，包括那些扩展了重排序功能的模型。实验结果表明，SCV（Span Compressed Vector，跨度压缩向量）显著优于单向量检索器，并且与多向量检索器具有竞争力。SCV利用跨度的压缩表示从标记序列生成多向量，我们预计其性能介于DPR（Dense Passage Retrieval，密集段落检索）和ColBERT之间。根据实验结果，SCV的得分接近ColBERT，正如我们所预期的那样，并且在AA、NF和T2数据集上的得分特别高。

Results with Reranker To further enhance performance, we conducted reranking using the cross-encoder (CE) version ms-marco-MiniLM-L-6-v2 based on the SCV retrieval results. In contrast, all comparison models in Table 3 performed rerank-ing based on BM25 retrieval results. The SCV+CE pipeline achieved an nDCG of 0.744 on TREC DL 19, showing an improvement of 0.099 in nDCG compared to the SCV retriever in Table 3. This result is 0.21 higher than monoBERT, indicating that retrieving relevant candidates during the retrieval stage positively impacts reranking. Moreover, it is evident that reranking using the proposed method outperforms relatively recent studies such as SimLM and ListT5.

使用重排序器的结果 为了进一步提高性能，我们基于SCV的检索结果，使用交叉编码器（CE）版本的ms - marco - MiniLM - L - 6 - v2进行了重排序。相比之下，表3中的所有对比模型都是基于BM25的检索结果进行重排序的。SCV + CE管道在TREC DL 19上的nDCG（Normalized Discounted Cumulative Gain，归一化折损累计增益）达到了0.744，与表3中的SCV检索器相比，nDCG提高了0.099。这一结果比monoBERT高0.21，表明在检索阶段检索相关候选文档对重排序有积极影响。此外，很明显，使用所提出的方法进行重排序优于相对较新的研究，如SimLM和ListT5。

Unlike the previous experimental setup, the results in the following row are based on rerank-ing using LLMs. The LLM approach involves decoder-only variations, with model sizes including 7B, 13B, and 20B. In reranking, RankLLaMA- ${13}\mathrm{\;B}$ demonstrated the best performance,followed by RankLLaMA-7B and the PRP model. Overall, LLM-based models exhibited higher performance compared to methods using small language models (SLM) as the backbone, but the differences in model size were quite significant. Despite PRP having the largest scale with a model size of ${20}\mathrm{\;B}$ among the LLM-based methods, it showed relatively lower performance and lacked competitiveness against SLM backbone models. Therefore, in in-domain retrieval, a well-tuned combination of small retrieval and ranking models remains competitive compared to LLM-based ranking models.

与之前的实验设置不同，以下行中的结果是基于使用大语言模型（LLM）进行重排序的。大语言模型方法涉及仅解码器变体，模型大小包括7B、13B和20B。在重排序中，RankLLaMA - ${13}\mathrm{\;B}$表现最佳，其次是RankLLaMA - 7B和PRP模型。总体而言，基于大语言模型的模型比以小语言模型（SLM）为骨干的方法表现更好，但模型大小的差异相当显著。尽管在基于大语言模型的方法中，PRP的规模最大，模型大小为${20}\mathrm{\;B}$，但其表现相对较低，与以小语言模型为骨干的模型相比缺乏竞争力。因此，在领域内检索中，经过良好调优的小检索和排序模型的组合与基于大语言模型的排序模型相比仍具有竞争力。

## 4 Related Works

## 4 相关工作

Modern RAG with Retriever Recently, with the advent of LLMs, there has been significant development and study related to RAG pipelines. Study on the RAG framework includes not only methods to enhance LLM performance but also attempts to refine performance based on retrieval results. This includes methods for summarizing retrieved results (Kim et al., 2024) and creating new retrieval results (Asai et al., 2024). Shao et al. (2023) generates responses by re-retrieving chunks based on the retrieved chunks and generated results. Shi et al. (2024) enhances the retriever to improve the performance of the LM based on the RAG structure.

带有检索器的现代检索增强生成（RAG） 最近，随着大语言模型的出现，与RAG管道相关的研究和开发取得了显著进展。对RAG框架的研究不仅包括提高大语言模型性能的方法，还包括基于检索结果优化性能的尝试。这包括对检索结果进行总结的方法（Kim等人，2024）和创建新检索结果的方法（Asai等人，2024）。Shao等人（2023）通过基于检索到的文本块和生成的结果重新检索文本块来生成响应。Shi等人（2024）基于RAG结构增强检索器以提高语言模型的性能。

Neural Information Retrieval Deep language models have significantly influenced neural information retrieval. A prevalent method involves processing the query-document pair with BERT, using the output of BERT's [CLS] token to determine a relevance score (Karpukhin et al., 2020). (Khat-tab and Zaharia, 2020) represents document text as a collection of token rather than a single vector and apply late interaction between the document and the query, implementing a late interaction mechanism between the document and the query. This method enables comprehensive semantic and lexical matching between queries and documents, reaching state-of-the-art performance across numerous benchmarks. Yet, the scalability of their non-linear scoring function faces challenges when extended to millions of documents. Alternative strategies (Gao et al., 2021; Li et al., 2023; Lee et al., 2023) simplify the multi-vector retrieval by focusing on retrieving only the most relevant tokens for ranking candidates, effectively pruning the document tokens.

神经信息检索 深度语言模型对神经信息检索产生了重大影响。一种流行的方法是使用BERT处理查询 - 文档对，利用BERT的[CLS]标记的输出来确定相关性得分（Karpukhin等人，2020）。（Khattab和Zaharia，2020）将文档文本表示为标记的集合，而不是单个向量，并在文档和查询之间应用后期交互，实现了文档和查询之间的后期交互机制。这种方法能够在查询和文档之间进行全面的语义和词汇匹配，在众多基准测试中达到了最先进的性能。然而，当扩展到数百万个文档时，其非线性评分函数的可扩展性面临挑战。其他策略（Gao等人，2021；Li等人，2023；Lee等人，2023）通过仅关注检索最相关的标记来简化多向量检索，以对候选文档进行排序，有效地修剪了文档标记。

Span Representation Span representation has primarily been utilized in information extraction tasks for processing documents. (Lee et al., 2017) enables end-to-end coreference resolution by extracting span representations and ranking span pairs. Performance improves significantly when BERT is adapted to whole word masking, leading to the development of SpanBERT (Joshi et al., 2020), which trains the model by setting the mask token unit to spans. SpanBERT helps to span-based approaches. In nested named entity recognition tasks (Zhu et al., 2023; Zhu and Li, 2022; Wan et al., 2022; Zhang et al., 2023), span representation is employed to address the problem by handling the range of chunks that are entities through span-based modeling and attaching entity tags.

跨度表示 跨度表示主要用于信息提取任务中的文档处理。（Lee等人，2017）通过提取跨度表示并对跨度对进行排序，实现了端到端的共指消解。当BERT适应全词掩码时，性能显著提高，从而催生了SpanBERT（Joshi等人，2020），它通过将掩码标记单元设置为跨度来训练模型。SpanBERT有助于基于跨度的方法。在嵌套命名实体识别任务中（Zhu等人，2023；Zhu和Li，2022；Wan等人，2022；Zhang等人，2023），跨度表示用于通过基于跨度的建模处理作为实体的文本块范围并附加实体标签来解决问题。

## 5 Conclusion

## 5 结论

In this paper, we propose an end-to-end multi-vector retrieval framework utilizing sequence compression, named SCV. Our method achieves a latency of ${0.8}\mathrm{\;{ms}}$ /query when querying a million-scale index, which is 207 times faster than ColBERT and 4.6 times faster than the fastest multi-vector retriever, CITADEL, on GPUs. While SCV records performance comparable to other multi-vector retrieval models, its major strength lies in its very small latency. Leveraging this advantage for re-ranking, SCV achieves state-of-the-art results among other SLM-based ranking models and shows promise among re-ranking methods. Our model minimizes information loss in the document sequence by fully utilizing token information to create span representations. Compressing token vectors has a strong potential of more efficiently and effectively model retrieval tasks.

在本文中，我们提出了一个利用序列压缩的端到端多向量检索框架，名为SCV（Sequence Compression-based Vector retrieval，基于序列压缩的向量检索）。我们的方法在查询百万级索引时，每次查询的延迟为${0.8}\mathrm{\;{ms}}$，在GPU上比ColBERT快207倍，比最快的多向量检索器CITADEL快4.6倍。虽然SCV的性能与其他多向量检索模型相当，但其主要优势在于极低的延迟。利用这一优势进行重排序，SCV在其他基于大语言模型（SLM）的排序模型中取得了最先进的成果，并且在重排序方法中显示出了潜力。我们的模型通过充分利用词元信息创建跨度表示，最大限度地减少了文档序列中的信息损失。压缩词元向量在更高效地建模检索任务方面具有巨大潜力。

Finally, in the modern RAG, additional modules are configured, including not only retrieval and generation but also the use of retrieval, retrieval summarization, and iterative retrieval. We believe that as more of these components are added, the speed of retrieval becomes increasingly important in real-world services.

最后，在现代检索增强生成（RAG）系统中，配置了额外的模块，不仅包括检索和生成，还包括检索的使用、检索摘要和迭代检索。我们认为，随着添加的这些组件越来越多，在实际服务中检索速度变得越来越重要。

## 6 Limitations

## 6 局限性

The proposed RAG system is designed to be more suitable for practical service use, focusing on the speed of the RAG system. As a result, there may be a slight performance decline compared to existing SOTA models. However, implementing this algorithm into an operational system is not technically difficult, so there is potential to maximize its usability based on the code that will be released in the future.

所提出的RAG系统旨在更适合实际服务应用，重点关注RAG系统的速度。因此，与现有的最优（SOTA）模型相比，可能会有轻微的性能下降。然而，将该算法集成到实际运行系统在技术上并不困难，因此基于未来将发布的代码，有潜力最大限度地提高其可用性。

## References

## 参考文献

Raviteja Anantha and Danil Vodianik. 2024. Context tuning for retrieval augmented generation. In Proceedings of the 1st Workshop on Uncertainty-Aware NLP (UncertaiNLP 2024), pages 15-22, St Julians, Malta. Association for Computational Linguistics.

拉维特贾·阿南塔（Raviteja Anantha）和达尼尔·沃迪亚尼克（Danil Vodianik）。2024年。检索增强生成的上下文调优。见第一届不确定性感知自然语言处理研讨会（UncertaiNLP 2024）会议论文集，第15 - 22页，马耳他圣朱利安斯。计算语言学协会。

Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024. Self-RAG: Learning to retrieve, generate, and critique through self-reflection. In The Twelfth International Conference on Learning Representations.

浅井朱里（Akari Asai）、吴泽秋（Zeqiu Wu）、王一中（Yizhong Wang）、阿维鲁普·西尔（Avirup Sil）和哈南内·哈吉希尔齐（Hannaneh Hajishirzi）。2024年。自我RAG：通过自我反思学习检索、生成和评估。见第十二届国际学习表征会议论文集。

Jinheon Baek, Alham Fikri Aji, Jens Lehmann, and Sung Ju Hwang. 2023. Direct fact retrieval from knowledge graphs without entity linking. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10038-10055, Toronto, Canada. Association for Computational Linguistics.

白镇轩（Jinheon Baek）、阿尔哈姆·菲克里·阿吉（Alham Fikri Aji）、延斯·莱曼（Jens Lehmann）和黄成柱（Sung Ju Hwang）。2023年。无需实体链接的知识图谱直接事实检索。见计算语言学协会第61届年会会议论文集（第1卷：长论文），第10038 - 10055页，加拿大多伦多。计算语言学协会。

Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Ben-gio. 2015. Neural machine translation by jointly learning to align and translate. In 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings.

德米特里·巴达诺乌（Dzmitry Bahdanau）、曹京勋（Kyunghyun Cho）和约书亚·本吉奥（Yoshua Ben-gio）。2015年。通过联合学习对齐和翻译进行神经机器翻译。见第三届国际学习表征会议（ICLR 2015）会议论文集，美国加利福尼亚州圣地亚哥，2015年5月7 - 9日，会议轨道论文。

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171-4186, Minneapolis, Minnesota. Association for Computational Linguistics.

雅各布·德夫林（Jacob Devlin）、张明伟（Ming-Wei Chang）、肯顿·李（Kenton Lee）和克里斯蒂娜·图托纳娃（Kristina Toutanova）。2019年。BERT：用于语言理解的深度双向变换器预训练。见计算语言学协会北美分会2019年会议：人类语言技术会议论文集，第1卷（长论文和短论文），第4171 - 4186页，美国明尼苏达州明尼阿波利斯。计算语言学协会。

Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant. 2021. Splade: Sparse lexical and expansion model for first stage ranking. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR '21, page 2288-2292, New York, NY, USA. Association for Computing Machinery.

蒂博·福尔马尔（Thibault Formal）、本杰明·皮沃瓦尔斯基（Benjamin Piwowarski）和斯特凡·克兰尚（Stéphane Clinchant）。2021年。SPLADE：用于第一阶段排序的稀疏词法和扩展模型。见第44届国际计算机协会信息检索研究与发展会议（SIGIR '21）会议论文集，第2288 - 2292页，美国纽约。美国计算机协会。

Luyu Gao and Jamie Callan. 2022. Unsupervised corpus aware language model pre-training for dense passage retrieval. In Proceedings of the 60th Annual

高宇（Luyu Gao）和杰米·卡兰（Jamie Callan）。2022年。用于密集段落检索的无监督语料感知语言模型预训练。见计算语言学协会第60届年会

Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2843- 2853, Dublin, Ireland. Association for Computational Linguistics.

会议论文集（第1卷：长论文），第2843 - 2853页，爱尔兰都柏林。计算语言学协会。

Luyu Gao, Zhuyun Dai, and Jamie Callan. 2021. COIL: Revisit exact lexical match in information retrieval with contextualized inverted list. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3030-3042, Online. Association for Computational Linguistics.

高宇（Luyu Gao）、戴竹云（Zhuyun Dai）和杰米·卡兰（Jamie Callan）。2021年。COIL：通过上下文倒排列表重新审视信息检索中的精确词法匹配。见计算语言学协会北美分会2021年会议：人类语言技术会议论文集，第3030 - 3042页，线上会议。计算语言学协会。

Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. 2023. Tevatron: An efficient and flexible toolkit for neural retrieval. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR '23, page 3120-3124, New York, NY, USA. Association for Computing Machinery.

高宇（Luyu Gao）、马雪光（Xueguang Ma）、吉米·林（Jimmy Lin）和杰米·卡兰（Jamie Callan）。2023年。Tevatron：一个高效灵活的神经检索工具包。见第46届国际计算机协会信息检索研究与发展会议（SIGIR '23）会议论文集，第3120 - 3124页，美国纽约。美国计算机协会。

Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong Park. 2024. Adaptive-RAG: Learning to adapt retrieval-augmented large language models through question complexity. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 7036-7050, Mexico City, Mexico. Association for Computational Linguistics.

郑素英（Soyeong Jeong）、白镇轩（Jinheon Baek）、赵锡民（Sukmin Cho）、黄成柱（Sung Ju Hwang）和朴钟（Jong Park）。2024年。自适应检索增强生成（Adaptive-RAG）：通过问题复杂度学习适配检索增强大语言模型。见《2024年北美计算语言学协会人类语言技术会议论文集》（第1卷：长论文），第7036 - 7050页，墨西哥城，墨西哥。计算语言学协会。

Mandar Joshi, Danqi Chen, Yinhan Liu, Daniel S. Weld, Luke Zettlemoyer, and Omer Levy. 2020. Span-BERT: Improving pre-training by representing and predicting spans. Transactions of the Association for Computational Linguistics, 8:64-77.

曼达尔·乔希（Mandar Joshi）、陈丹琦（Danqi Chen）、刘音涵（Yinhan Liu）、丹尼尔·S·韦尔德（Daniel S. Weld）、卢克·泽特勒莫耶（Luke Zettlemoyer）和奥默·利维（Omer Levy）。2020年。跨度BERT（Span-BERT）：通过表示和预测跨度改进预训练。《计算语言学协会汇刊》，8：64 - 77。

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6769-6781, Online. Association for Computational Linguistics.

弗拉基米尔·卡尔普欣（Vladimir Karpukhin）、巴拉斯·奥古兹（Barlas Oguz）、闵世元（Sewon Min）、帕特里克·刘易斯（Patrick Lewis）、莱德尔·吴（Ledell Wu）、谢尔盖·叶杜诺夫（Sergey Edunov）、陈丹琦（Danqi Chen）和易文涛（Wen-tau Yih）。2020年。开放域问答的密集段落检索。见《2020年自然语言处理经验方法会议论文集》（EMNLP），第6769 - 6781页，线上会议。计算语言学协会。

Omar Khattab and Matei Zaharia. 2020. Colbert: Efficient and effective passage search via contextualized late interaction over bert. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR '20, page 39-48, New York, NY, USA. Association for Computing Machinery.

奥马尔·哈塔卜（Omar Khattab）和马特·扎哈里亚（Matei Zaharia）。2020年。科尔伯特（Colbert）：通过基于BERT的上下文后期交互实现高效有效的段落搜索。见《第43届ACM信息检索研究与发展国际会议论文集》，SIGIR '20，第39 - 48页，美国纽约。美国计算机协会。

Gyuwan Kim, Jinhyuk Lee, Barlas Oguz, Wen-han Xiong, Yizhe Zhang, Yashar Mehdad, and William Yang Wang. 2022. Bridging the training-inference gap for dense phrase retrieval. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 3713-3724, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

金圭焕（Gyuwan Kim）、李镇赫（Jinhyuk Lee）、巴拉斯·奥古兹（Barlas Oguz）、熊文瀚（Wen-han Xiong）、张一哲（Yizhe Zhang）、亚沙尔·梅赫达德（Yashar Mehdad）和王威廉（William Yang Wang）。2022年。弥合密集短语检索中训练与推理的差距。见《计算语言学协会研究成果：2022年自然语言处理经验方法会议》，第3713 - 3724页，阿拉伯联合酋长国阿布扎比。计算语言学协会。

Jaehyung Kim, Jaehyun Nam, Sangwoo Mo, Jongjin Park, Sang-Woo Lee, Minjoon Seo, Jung-Woo Ha, and Jinwoo Shin. 2024. Sure: Summarizing retrievals

金在亨（Jaehyung Kim）、南在贤（Jaehyun Nam）、莫相宇（Sangwoo Mo）、朴钟镇（Jongjin Park）、李相佑（Sang-Woo Lee）、徐敏俊（Minjoon Seo）、河正宇（Jung-Woo Ha）和申镇宇（Jinwoo Shin）。2024年。确定（Sure）：总结检索结果

using answer candidates for open-domain QA of LLMs. In The Twelfth International Conference on Learning Representations.

使用答案候选进行大语言模型的开放域问答。见《第十二届学习表征国际会议》。

Yoon Kim. 2014. Convolutional neural networks for sentence classification. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1746-1751, Doha, Qatar. Association for Computational Linguistics.

金允（Yoon Kim）。2014年。用于句子分类的卷积神经网络。见《2014年自然语言处理经验方法会议论文集》（EMNLP），第1746 - 1751页，卡塔尔多哈。计算语言学协会。

Jinhyuk Lee, Zhuyun Dai, Sai Meher Karthik Duddu, Tao Lei, Iftekhar Naim, Ming-Wei Chang, and Vincent Y. Zhao. 2023. Rethinking the role of token retrieval in multi-vector retrieval. CoRR, abs/2304.01982.

李镇赫（Jinhyuk Lee）、戴珠云（Zhuyun Dai）、赛·梅赫尔·卡尔蒂克·杜杜（Sai Meher Karthik Duddu）、雷涛（Tao Lei）、伊夫特哈尔·奈姆（Iftekhar Naim）、张明伟（Ming-Wei Chang）和赵文森（Vincent Y. Zhao）。2023年。重新思考多向量检索中词元检索的作用。计算机研究报告库（CoRR），abs/2304.01982。

Kenton Lee, Luheng He, Mike Lewis, and Luke Zettle-moyer. 2017. End-to-end neural coreference resolution. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 188-197, Copenhagen, Denmark. Association for Computational Linguistics.

肯顿·李（Kenton Lee）、何鲁衡（Luheng He）、迈克·刘易斯（Mike Lewis）和卢克·泽特勒莫耶（Luke Zettle-moyer）。2017年。端到端神经共指消解。见《2017年自然语言处理经验方法会议论文集》，第188 - 197页，丹麦哥本哈根。计算语言学协会。

Minghan Li, Sheng-Chieh Lin, Barlas Oguz, Asish Ghoshal, Jimmy Lin, Yashar Mehdad, Wen-tau Yih, and Xilun Chen. 2023. CITADEL: Conditional token interaction via dynamic lexical routing for efficient and effective multi-vector retrieval. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11891-11907, Toronto, Canada. Association for Computational Linguistics.

李明翰（Minghan Li）、林圣杰（Sheng-Chieh Lin）、巴拉斯·奥古兹（Barlas Oguz）、阿西什·戈沙尔（Asish Ghoshal）、林吉米（Jimmy Lin）、亚沙尔·梅赫达德（Yashar Mehdad）、易文涛（Wen-tau Yih）和陈希伦（Xilun Chen）。2023年。堡垒（CITADEL）：通过动态词法路由进行条件词元交互以实现高效有效的多向量检索。见《第61届计算语言学协会年会论文集》（第1卷：长论文），第11891 - 11907页，加拿大多伦多。计算语言学协会。

Ilya Loshchilov and Frank Hutter. 2017. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101.

伊利亚·洛希洛夫（Ilya Loshchilov）和弗兰克·胡特（Frank Hutter）。2017年。解耦权重衰减正则化。预印本arXiv：1711.05101。

Xueguang Ma, Liang Wang, Nan Yang, Furu Wei, and Jimmy Lin. 2024. Fine-tuning llama for multistage text retrieval. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR '24, page 2421-2425, New York, NY, USA. Association for Computing Machinery.

马学光、王亮、杨楠、魏富如和林吉米（Jimmy Lin）。2024年。微调羊驼（llama）进行多阶段文本检索。见《第47届ACM信息检索研究与发展国际会议论文集》，SIGIR '24，第2421 - 2425页，美国纽约。美国计算机协会。

Craig Macdonald and Nicola Tonellotto. 2021. On approximate nearest neighbour selection for multistage dense retrieval. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management, CIKM '21, page 3318-3322, New York, NY, USA. Association for Computing Machinery.

克雷格·麦克唐纳（Craig Macdonald）和尼古拉·托内洛托（Nicola Tonellotto）。2021年。关于多阶段密集检索的近似最近邻选择。见《第30届ACM信息与知识管理国际会议论文集》，CIKM '21，第3318 - 3322页，美国纽约。美国计算机协会。

C.D. Manning, P. Raghavan, and H. Schütze. 2008. Introduction to Information Retrieval. Cambridge University Press.

C.D. 曼宁（C.D. Manning）、P. 拉加万（P. Raghavan）和H. 舒尔策（H. Schütze）。2008年。《信息检索导论》。剑桥大学出版社。

Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Hernandez Abrego, Ji Ma, Vincent Zhao, Yi Luan, Keith Hall, Ming-Wei Chang, and Yinfei Yang. 2022. Large dual encoders are generalizable retrievers. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 9844-9855, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

倪建墨（Jianmo Ni）、曲晨（Chen Qu）、卢静（Jing Lu）、戴竹云（Zhuyun Dai）、古斯塔沃·埃尔南德斯·阿布雷戈（Gustavo Hernandez Abrego）、马骥（Ji Ma）、赵文森（Vincent Zhao）、栾毅（Yi Luan）、基思·霍尔（Keith Hall）、张明伟（Ming-Wei Chang）和杨荫飞（Yinfei Yang）。2022年。大型双编码器是可泛化的检索器。《2022年自然语言处理经验方法会议论文集》，第9844 - 9855页，阿拉伯联合酋长国阿布扎比。计算语言学协会。

Rodrigo Frassetto Nogueira, Wei Yang, Kyunghyun Cho, and Jimmy Lin. 2019. Multi-stage document

罗德里戈·弗拉塞托·诺盖拉（Rodrigo Frassetto Nogueira）、杨威（Wei Yang）、赵京焕（Kyunghyun Cho）和林吉米（Jimmy Lin）。2019年。多阶段文档

ranking with BERT. CoRR, abs/1910.14424.

基于BERT的排序。计算机研究报告，abs/1910.14424。

Ronak Pradeep, Sahel Sharifymoghaddam, and Jimmy Lin. 2023. Rankvicuna: Zero-shot listwise document reranking with open-source large language models. Preprint, arXiv:2309.15088.

罗纳克·普拉迪普（Ronak Pradeep）、萨赫尔·沙里菲莫加达姆（Sahel Sharifymoghaddam）和林吉米（Jimmy Lin）。2023年。Rankvicuna：使用开源大语言模型进行零样本列表式文档重排序。预印本，arXiv:2309.15088。

Zhen Qin, Rolf Jagerman, Kai Hui, Honglei Zhuang, Junru Wu, Le Yan, Jiaming Shen, Tianqi Liu, Jialu Liu, Donald Metzler, Xuanhui Wang, and Michael Bendersky. 2024. Large language models are effective text rankers with pairwise ranking prompting. In Findings of the Association for Computational Linguistics: NAACL 2024, pages 1504-1518, Mexico City, Mexico. Association for Computational Linguistics.

秦臻（Zhen Qin）、罗尔夫·贾格曼（Rolf Jagerman）、惠凯（Kai Hui）、庄宏磊（Honglei Zhuang）、吴俊儒（Junru Wu）、闫乐（Le Yan）、沈佳明（Jiaming Shen）、刘天奇（Tianqi Liu）、刘佳璐（Jialu Liu）、唐纳德·梅茨勒（Donald Metzler）、王宣辉（Xuanhui Wang）和迈克尔·本德尔斯基（Michael Bendersky）。2024年。大语言模型通过成对排序提示是有效的文本排序器。《计算语言学协会研究成果：2024年北美计算语言学协会人类语言技术会议》，第1504 - 1518页，墨西哥墨西哥城。计算语言学协会。

Stephen Robertson and Hugo Zaragoza. 2009. The probabilistic relevance framework: $\mathrm{{Bm}}{25}$ and beyond. Found. Trends Inf. Retr., 3(4):333-389.

斯蒂芬·罗伯逊（Stephen Robertson）和雨果·萨拉戈萨（Hugo Zaragoza）。2009年。概率相关性框架：$\mathrm{{Bm}}{25}$及超越。《信息检索趋势与基础》，3(4):333 - 389。

Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. 2019. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.

维克多·桑（Victor Sanh）、利桑德尔·德比特（Lysandre Debut）、朱利安·肖蒙（Julien Chaumond）和托马斯·沃尔夫（Thomas Wolf）。2019年。Distilbert，BERT的蒸馏版本：更小、更快、更便宜、更轻量。arXiv预印本arXiv:1910.01108。

Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. 2020. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. Preprint, arXiv:1910.01108.

维克多·桑（Victor Sanh）、利桑德尔·德比特（Lysandre Debut）、朱利安·肖蒙（Julien Chaumond）和托马斯·沃尔夫（Thomas Wolf）。2020年。Distilbert，BERT的蒸馏版本：更小、更快、更便宜、更轻量。预印本，arXiv:1910.01108。

Keshav Santhanam, Omar Khattab, Christopher Potts, and Matei Zaharia. 2022a. Plaid: An efficient engine for late interaction retrieval. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management, CIKM '22, page 1747-1756, New York, NY, USA. Association for Computing Machinery.

凯沙夫·桑塔南（Keshav Santhanam）、奥马尔·哈塔布（Omar Khattab）、克里斯托弗·波茨（Christopher Potts）和马特·扎哈里亚（Matei Zaharia）。2022a。Plaid：一种用于后期交互检索的高效引擎。《第31届ACM国际信息与知识管理会议论文集》，CIKM '22，第1747 - 1756页，美国纽约州纽约市。美国计算机协会。

Keshav Santhanam, Omar Khattab, Christopher Potts, and Matei Zaharia. 2022b. Plaid: An efficient engine for late interaction retrieval. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management, CIKM '22, page 1747-1756, New York, NY, USA. Association for Computing Machinery.

凯沙夫·桑塔南（Keshav Santhanam）、奥马尔·哈塔布（Omar Khattab）、克里斯托弗·波茨（Christopher Potts）和马特·扎哈里亚（Matei Zaharia）。2022b。Plaid：一种用于后期交互检索的高效引擎。《第31届ACM国际信息与知识管理会议论文集》，CIKM '22，第1747 - 1756页，美国纽约州纽约市。美国计算机协会。

Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. 2022c. Col-BERTv2: Effective and efficient retrieval via lightweight late interaction. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3715-3734, Seattle, United States. Association for Computational Linguistics.

凯沙夫·桑塔南（Keshav Santhanam）、奥马尔·哈塔布（Omar Khattab）、乔恩·萨德 - 法尔孔（Jon Saad-Falcon）、克里斯托弗·波茨（Christopher Potts）和马特·扎哈里亚（Matei Zaharia）。2022c。Col - BERTv2：通过轻量级后期交互实现高效检索。《2022年北美计算语言学协会人类语言技术会议论文集》，第3715 - 3734页，美国西雅图。计算语言学协会。

Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, and Weizhu Chen. 2023. Enhancing retrieval-augmented large language models with iterative retrieval-generation synergy. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 9248-9274, Singapore. Association for Computational Linguistics.

邵志宏（Zhihong Shao）、龚叶云（Yeyun Gong）、沈业龙（Yelong Shen）、黄民烈（Minlie Huang）、段楠（Nan Duan）和陈伟柱（Weizhu Chen）。2023年。通过迭代检索 - 生成协同增强检索增强大语言模型。《计算语言学协会研究成果：2023年自然语言处理经验方法会议》，第9248 - 9274页，新加坡。计算语言学协会。

Weijia Shi, Sewon Min, Michihiro Yasunaga, Min-joon Seo, Richard James, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. 2024. RE-PLUG: Retrieval-augmented black-box language models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 8371- 8384, Mexico City, Mexico. Association for Computational Linguistics.

施伟佳（Weijia Shi）、闵世元（Sewon Min）、安永宏（Michihiro Yasunaga）、徐敏俊（Min - joon Seo）、理查德·詹姆斯（Richard James）、迈克·刘易斯（Mike Lewis）、卢克·泽特尔莫耶（Luke Zettlemoyer）和易文涛（Wen - tau Yih）。2024年。RE - PLUG：检索增强黑盒语言模型。《2024年北美计算语言学协会人类语言技术会议论文集（第1卷：长论文）》，第8371 - 8384页，墨西哥墨西哥城。计算语言学协会。

Nandan Thakur, Nils Reimers, Andreas Rücklé, Ab-hishek Srivastava, and Iryna Gurevych. 2021. BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual.

南丹·塔库尔（Nandan Thakur）、尼尔斯·赖默斯（Nils Reimers）、安德烈亚斯·吕克莱（Andreas Rücklé）、阿比舍克·斯里瓦斯塔瓦（Ab - hishek Srivastava）和伊琳娜·古列维奇（Iryna Gurevych）。2021年。BEIR：信息检索模型零样本评估的异构基准。《神经信息处理系统数据集与基准赛道会议论文集1》，2021年神经信息处理系统数据集与基准会议，2021年12月，线上会议。

Zhiliang Tian, Rui Yan, Lili Mou, Yiping Song, Yan-song Feng, and Dongyan Zhao. 2017. How to make context more useful? an empirical study on context-aware neural conversational models. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 231-236, Vancouver, Canada. Association for Computational Linguistics.

田志良、闫锐、牟丽丽、宋一平、冯岩松和赵东岩。2017年。如何让上下文更有用？对上下文感知神经对话模型的实证研究。见《计算语言学协会第55届年会论文集（第2卷：短篇论文）》，第231 - 236页，加拿大温哥华。计算语言学协会。

Jack Urbanek, Angela Fan, Siddharth Karamcheti, Saachi Jain, Samuel Humeau, Emily Dinan, Tim Rocktäschel, Douwe Kiela, Arthur Szlam, and Jason Weston. 2019. Learning to speak and act in a fantasy text adventure game. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 673-683, Hong Kong, China. Association for Computational Linguistics.

杰克·乌尔班内克、安吉拉·范、悉达思·卡拉姆切蒂、萨奇·贾因、塞缪尔·休莫、艾米丽·迪南、蒂姆·罗克塔舍尔、杜威·基拉、亚瑟·斯拉姆和杰森·韦斯顿。2019年。在幻想文本冒险游戏中学习说话和行动。见《2019年自然语言处理经验方法会议和第9届自然语言处理国际联合会议（EMNLP - IJCNLP）论文集》，第673 - 683页，中国香港。计算语言学协会。

Juncheng Wan, Dongyu Ru, Weinan Zhang, and Yong Yu. 2022. Nested named entity recognition with span-level graphs. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 892-903, Dublin, Ireland. Association for Computational Linguistics.

万俊成、茹冬宇、张韦男和俞勇。2022年。基于跨度级图的嵌套命名实体识别。见《计算语言学协会第60届年会论文集（第1卷：长篇论文）》，第892 - 903页，爱尔兰都柏林。计算语言学协会。

Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. 2023. SimLM: Pre-training with representation bottleneck for dense passage retrieval. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2244-2258, Toronto, Canada. Association for Computational Linguistics. Yireun Kim, and Seung won Hwang. 2024. Listt5:

王亮、杨楠、黄晓龙、焦彬星、杨林军、蒋大新、兰甘·马宗达和魏富如。2023年。SimLM：通过表示瓶颈进行预训练以实现密集段落检索。见《计算语言学协会第61届年会论文集（第1卷：长篇论文）》，第2244 - 2258页，加拿大多伦多。计算语言学协会。金怡润和黄承元。2024年。Listt5:

Enwei Zhu and Jinpeng Li. 2022. Boundary smoothing for named entity recognition. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7096-7108, Dublin, Ireland. Association for Computational Linguistics.

朱恩伟和李金鹏。2022年。命名实体识别的边界平滑方法。见《计算语言学协会第60届年会论文集（第1卷：长篇论文）》，第7096 - 7108页，爱尔兰都柏林。计算语言学协会。

Enwei Zhu, Yiyang Liu, and Jinpeng Li. 2023. Deep span representations for named entity recognition. In Findings of the Association for Computational Linguistics: ACL 2023, pages 10565- 10582, Toronto, Canada. Association for Computational Linguistics.

朱恩伟、刘一阳和李金鹏。2023年。用于命名实体识别的深度跨度表示。见《计算语言学协会研究成果：ACL 2023》，第10565 - 10582页，加拿大多伦多。计算语言学协会。

## A Appendix

## 附录A

### A.1 Applied Hyperparameters

### A.1 应用的超参数

<!-- Media -->

<table><tr><td/><td>value</td></tr><tr><td>Backborn</td><td>DistilBERT-base</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>learning_rate</td><td>${5.0}\mathrm{e} - 5$</td></tr><tr><td>Dropout</td><td>0.05</td></tr><tr><td>lr_scheduler</td><td>cosine</td></tr><tr><td>Epoch</td><td>10</td></tr><tr><td>sequence_len</td><td>512</td></tr><tr><td>Batch size</td><td>630</td></tr><tr><td>Random Seed</td><td>1004</td></tr><tr><td>BM25 TOP $n$</td><td>1000</td></tr></table>

<table><tbody><tr><td></td><td>值</td></tr><tr><td>Backborn（原文可能有误，推测可能是Backbone，意为骨干）</td><td>DistilBERT基础版</td></tr><tr><td>优化器</td><td>AdamW优化算法</td></tr><tr><td>学习率</td><td>${5.0}\mathrm{e} - 5$</td></tr><tr><td>丢弃法</td><td>0.05</td></tr><tr><td>学习率调度器</td><td>余弦</td></tr><tr><td>轮次</td><td>10</td></tr><tr><td>序列长度</td><td>512</td></tr><tr><td>批量大小</td><td>630</td></tr><tr><td>随机种子</td><td>1004</td></tr><tr><td>BM25前$n$名</td><td>1000</td></tr></tbody></table>

Table 4: Applied hyperparameter settings.

表4：应用的超参数设置。

<!-- Media -->

### A.2 Details of Experimental Environments

### A.2 实验环境详情

The hyperparameter settings used in this study can be found in 4 . The model essentially adopts the DistilBERT-base model, and experiments were conducted based on the top 1000 search results retrieved by BM25. The batch size was set to 630 , utilizing the maximum size available on an A100 GPU. Specific learning rates and token sizes are provided in Table 4.

本研究中使用的超参数设置可在表4中找到。该模型本质上采用了DistilBERT-base模型（蒸馏BERT基础模型），并基于BM25检索到的前1000个搜索结果进行了实验。批量大小设置为630，采用了A100 GPU上可用的最大大小。具体的学习率和标记大小在表4中给出。

### A.3 Coarse-to-fine Search Overview

### A.3 粗到细搜索概述

SCV employs multi-task learning to jointly train single-vector and multi-vector retrieval. During the indexing phase, the [CLS] token vector is stored with span-level vectors for each document in the collection. At inference time, the SCV model retrieves the stored single vector and span vectors for each document, loading them into memory. An overview of this process is presented in Fig. 4, specifically in the On memory section. In the Process section, the [CLS] token vector of each document, loaded into memory, is used to compute similarity with the [CLS] vector of the encoded query. The top $N$ relevant document IDs are then selected. Without additional gathering operations, the system directly computes the maximum similarity between query token vectors and document span vectors,ultimately producing the top $K$ relevant document IDs. This approach eliminates the need for intermediate gathering operations, enabling a coarse-to-fine retrieval process. It efficiently identifies candidate relevant documents at a coarse level and performs fine-grained token- and span-level retrieval based on these candidates in an end-to-end manner. Compared to traditional two-stage methods, SCV offers a simpler and faster way to retrieve relevant documents.

SCV（单向量和多向量联合检索模型）采用多任务学习来联合训练单向量和多向量检索。在索引阶段，[CLS]标记向量与集合中每个文档的跨度级向量一起存储。在推理时，SCV模型检索每个文档存储的单向量和跨度向量，并将它们加载到内存中。这一过程的概述如图4所示，具体在“内存中”部分。在“处理”部分，加载到内存中的每个文档的[CLS]标记向量用于与编码查询的[CLS]向量计算相似度。然后选择前$N$个相关文档ID。无需额外的收集操作，系统直接计算查询标记向量与文档跨度向量之间的最大相似度，最终生成前$K$个相关文档ID。这种方法无需中间收集操作，实现了从粗到细的检索过程。它能在粗粒度级别高效识别候选相关文档，并基于这些候选文档以端到端的方式进行细粒度的标记和跨度级检索。与传统的两阶段方法相比，SCV提供了一种更简单、更快的检索相关文档的方式。

### A.4 Prompt template

### A.4 提示模板

We use GPT-4 for question augmentation. The prompt used for augmentation is shown in Figure 3, and passages from MSMARCO are randomly sampled and input along with the prompt.

我们使用GPT - 4进行问题扩充。用于扩充的提示如图3所示，并且从MSMARCO中随机采样段落并与提示一起输入。

<!-- Media -->

<table><tr><td>#Num. of Q</td><td>nDCG@100</td><td>Recall@100</td></tr><tr><td>w/o aug.</td><td>0.305</td><td>0.267</td></tr><tr><td>50k</td><td>0.301</td><td>0.265</td></tr><tr><td>100k</td><td>0.275</td><td>0.253</td></tr><tr><td>150k</td><td>0.315</td><td>0.278</td></tr><tr><td>200k</td><td>0.300</td><td>0.266</td></tr></table>

<table><tbody><tr><td>#查询数量</td><td>归一化折损累积增益@100（nDCG@100）</td><td>召回率@100（Recall@100）</td></tr><tr><td>无增强处理</td><td>0.305</td><td>0.267</td></tr><tr><td>50k</td><td>0.301</td><td>0.265</td></tr><tr><td>100k</td><td>0.275</td><td>0.253</td></tr><tr><td>150k</td><td>0.315</td><td>0.278</td></tr><tr><td>200k</td><td>0.300</td><td>0.266</td></tr></tbody></table>

Table 5: Ablation for question augmentation

表5：问题增强消融实验

<!-- Media -->

### A.5 Ablation for Query Augmentation

### A.5 查询增强消融实验

To make the model more robust by learning diverse expressions for the retriever's positive samples, we perform question augmentation using GPT-4. Table 5 shows the performance changes with the use of augmented questions. We create augmentation amounts of ${50}\mathrm{k},{100}\mathrm{k},{150}\mathrm{k}$ ,and ${200}\mathrm{k}$ ,and among these,using ${150}\mathrm{k}$ results in the best performance.

为了通过学习检索器正样本的多样化表达使模型更具鲁棒性，我们使用GPT - 4进行问题增强。表5展示了使用增强问题后的性能变化。我们创建了${50}\mathrm{k},{100}\mathrm{k},{150}\mathrm{k}$、${200}\mathrm{k}$的增强数量，其中使用${150}\mathrm{k}$取得了最佳性能。

### A.6 Reranking Result for Out-of-domain

### A.6 域外数据重排序结果

In Table 6, we measure the reranking performance on out-of-domain data using the BIER benchmark.

在表6中，我们使用BIER基准测试衡量了域外数据的重排序性能。

Leveraging the advantage of SCV's rapid latency, we perform a re-ranking on the top-1000 retrieval results. Compared to BM25+CE using the same re-ranking model, our approach exhibits superior performance, indicating its efficacy in identifying candidate documents for zero-shot scenarios.

利用SCV低延迟的优势，我们对前1000个检索结果进行了重排序。与使用相同重排序模型的BM25 + CE相比，我们的方法表现更优，这表明它在零样本场景下识别候选文档方面的有效性。

The experimental results show that the performance of the SCV retrieval stage is 0.436 according to Table 2, and reranking improves the score by 0.073 . Although it shows a lower average score compared to HYRR or RankT5-large, it is improved compared to BM25+CE, which uses the same ranking model CE.

实验结果表明，根据表2，SCV检索阶段的性能得分为0.436，重排序使分数提高了0.073。虽然与HYRR或RankT5 - large相比，其平均得分较低，但与使用相同排序模型CE的BM25 + CE相比有所提高。

<!-- Media -->

Please provide a high-quality answer to the part I requested. Take a deep breath and think slowly. Create as many questions as possible, over 20, using only the content included in the input document. Base the questions on 'when, where, what, why, who (or what), and how'. Gradually think and create questions of various types such as 'comparison, fact verification, quantity, keyword, conversational, domain-specific', etc. For question generation, use [G] as a delimiter to insert one question at a time, and indicate whether the answer to the generated question can be found in the input paragraph with [sufficientlaveragelinsufficientlnone]. To summarize the request, everything is in Korean, and the task is to create questions dependent on the given document. You are a child with a lot of knowledge. You can think of a wide variety of questions for a single entity. So, create various questions that can be made from the above document for me. Focus on questions that people would ask via web search or phone calls. Avoid vague questions that ask about articles or pronouns like 'this' or 'that'. And only create questions whose answers can be found in the given document. I will enter the document as [D]. [D]: \{Input passage\}

请针对我要求的部分提供高质量的答案。深呼吸，慢慢思考。仅使用输入文档中的内容尽可能多地创建问题，超过20个。基于“何时（when）、何地（where）、什么（what）、为什么（why）、谁（或什么，who/what）以及如何（how）”来提问。逐步思考并创建各种类型的问题，如“比较、事实验证、数量、关键词、对话、特定领域”等。对于问题生成，使用[G]作为分隔符逐个插入问题，并通过[sufficient/laverage/insufficient/none]表明生成问题的答案是否能在输入段落中找到。总结一下要求，所有内容使用韩语，任务是根据给定文档创建问题。你是一个知识渊博的孩子。你可以为单个实体想出各种各样的问题。所以，请为我创建可以从上述文档中提出的各种问题。重点关注人们通过网络搜索或电话可能会问的问题。避免询问像“这个”或“那个”这样的文章或代词的模糊问题。并且只创建答案可以在给定文档中找到的问题。我将输入文档标记为[D]。[D]：{输入段落}

Figure 3: Prompt template design for question generation.

图3：问题生成的提示模板设计

<table><tr><td>Methods</td><td>AA</td><td>CF</td><td>DB</td><td>Fe</td><td>FQ</td><td>HQ</td><td>NF</td><td>NQ</td><td>Qu</td><td>SF</td><td>SD</td><td>TC</td><td>T2</td><td>Avg.</td></tr><tr><td>BM25+CE</td><td>0.311</td><td>0.253</td><td>0.409</td><td>0.819</td><td>0.347</td><td>0.707</td><td>0.350</td><td>0.533</td><td>0.825</td><td>0.688</td><td>0.166</td><td>0.757</td><td>0.271</td><td>0.495</td></tr><tr><td>HYRR</td><td>0.344</td><td>0.272</td><td>0.385</td><td>0.868</td><td>0.408</td><td>0.706</td><td>0.379</td><td>0.532</td><td>0.861</td><td>0.734</td><td>0.183</td><td>0.796</td><td>0.368</td><td>0.526</td></tr><tr><td>RankT5-large</td><td>0.330</td><td>0.215</td><td>0.442</td><td>0.832</td><td>0.445</td><td>0.710</td><td>0.381</td><td>0.614</td><td>0.831</td><td>0.750</td><td>0.181</td><td>0.807</td><td>0.440</td><td>0.524</td></tr><tr><td>SCV+CE</td><td>0.508</td><td>0.240</td><td>0.452</td><td>0.804</td><td>0.365</td><td>0.691</td><td>0.339</td><td>0.570</td><td>0.826</td><td>0.673</td><td>0.164</td><td>0.720</td><td>0.267</td><td>0.509</td></tr></table>

<table><tbody><tr><td>方法</td><td>AA</td><td>CF</td><td>DB</td><td>铁（Fe）</td><td>FQ</td><td>HQ</td><td>NF</td><td>NQ</td><td>Qu</td><td>SF</td><td>SD</td><td>TC</td><td>T2</td><td>平均值（Avg.）</td></tr><tr><td>BM25+交叉熵（BM25+CE）</td><td>0.311</td><td>0.253</td><td>0.409</td><td>0.819</td><td>0.347</td><td>0.707</td><td>0.350</td><td>0.533</td><td>0.825</td><td>0.688</td><td>0.166</td><td>0.757</td><td>0.271</td><td>0.495</td></tr><tr><td>HYRR</td><td>0.344</td><td>0.272</td><td>0.385</td><td>0.868</td><td>0.408</td><td>0.706</td><td>0.379</td><td>0.532</td><td>0.861</td><td>0.734</td><td>0.183</td><td>0.796</td><td>0.368</td><td>0.526</td></tr><tr><td>RankT5-large</td><td>0.330</td><td>0.215</td><td>0.442</td><td>0.832</td><td>0.445</td><td>0.710</td><td>0.381</td><td>0.614</td><td>0.831</td><td>0.750</td><td>0.181</td><td>0.807</td><td>0.440</td><td>0.524</td></tr><tr><td>SCV+交叉熵（SCV+CE）</td><td>0.508</td><td>0.240</td><td>0.452</td><td>0.804</td><td>0.365</td><td>0.691</td><td>0.339</td><td>0.570</td><td>0.826</td><td>0.673</td><td>0.164</td><td>0.720</td><td>0.267</td><td>0.509</td></tr></tbody></table>

Table 6: nDCG@10 on BEIR. Dataset Legend is same to Table 2.

表6：BEIR数据集上的nDCG@10指标。数据集图例与表2相同。

<!-- figureText: Coarse Retrieval Fine Retrieval (K-docs) Doc Vector DB Docm (All-docs) Filter: top- $N$ doc ids Process Q Encoder D Encoder Spans On memory Tokens SCV structure -->

<img src="https://cdn.noedgeai.com/0195afa2-270e-7c66-ae49-4113ee3e4312_10.jpg?x=188&y=1380&w=1282&h=582&r=0"/>

Figure 4: Coarse-to-fine search overview. In the figure, yellow boxes represent the vectors of a single-vector retriever, while red boxes denote the vectors of individual spans. The empty boxes outlined in blue indicate token-level vectors for SCV but are not used during model runtime. The green box illustrates the abstract structure of the Q Encoder for questions, and the blue box represents the D Encoder for documents.

图4：粗到细搜索概述。在图中，黄色框代表单向量检索器的向量，而红色框表示各个片段的向量。蓝色轮廓的空框表示单分量向量（SCV）的词元级向量，但在模型运行时不使用。绿色框展示了问题的问题编码器（Q Encoder）的抽象结构，蓝色框代表文档的文档编码器（D Encoder）。

<!-- Media -->