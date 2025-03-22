# Compact Token Representations with Contextual Quantization for Efficient Document Re-ranking

# 用于高效文档重排序的上下文量化紧凑Token表示

Yingrui Yang, Yifan Qiao, Tao Yang

杨英锐，乔一帆，杨涛

Department of Computer Science, University of California at Santa Barbara, USA

美国加州大学圣巴巴拉分校计算机科学系

\{yingruiyang, yifanqiao, tyang\}@cs.ucsb.edu

\{yingruiyang, yifanqiao, tyang\}@cs.ucsb.edu

## Abstract

## 摘要

Transformer based re-ranking models can achieve high search relevance through context-aware soft matching of query tokens with document tokens. To alleviate runtime complexity of such inference, previous work has adopted a late interaction architecture with pre-computed contextual token representations at the cost of a large online storage. This paper proposes contextual quantization of token embed-dings by decoupling document-specific and document-independent ranking contributions during codebook-based compression. This allows effective online decompression and embedding composition for better search relevance. This paper presents an evaluation of the above compact token representation model in terms of relevance and space efficiency.

基于Transformer的重排序模型可以通过查询Token与文档Token的上下文感知软匹配来实现较高的搜索相关性。为了减轻此类推理的运行时复杂性，先前的工作采用了一种后期交互架构，预先计算上下文Token表示，但代价是需要大量的在线存储空间。本文提出了对Token嵌入进行上下文量化的方法，即在基于码本的压缩过程中，将特定于文档和与文档无关的排序贡献解耦。这使得能够进行有效的在线解压缩和嵌入组合，以提高搜索相关性。本文从相关性和空间效率方面对上述紧凑Token表示模型进行了评估。

## 1 Introduction

## 1 引言

Modern search engines for text documents typically employ multi-stage ranking. The first retrieval stage extracts top candidate documents matching a query from a large search index with a simple ranking method. The second stage or a later stage uses a more complex machine learning algorithm to re-rank top results thoroughly. Recently neural re-ranking techniques from transformer-based architectures have achieved impressive relevance scores for top $k$ document re-ranking,such as MacAvaney et al. (2019). However, using a transformer-based model to rank or re-rank is extremely expensive during the online inference (Lin et al., 2020). Various efforts have been made to reduce its computational complexity (e.g. Gao et al. (2020)).

现代文本文档搜索引擎通常采用多阶段排序。第一检索阶段使用简单的排序方法从大型搜索索引中提取与查询匹配的顶级候选文档。第二阶段或后续阶段使用更复杂的机器学习算法对顶级结果进行全面重排序。最近，基于Transformer架构的神经重排序技术在顶级$k$文档重排序方面取得了令人印象深刻的相关性得分，例如MacAvaney等人（2019年）的研究。然而，在在线推理过程中，使用基于Transformer的模型进行排序或重排序的成本极高（Lin等人，2020年）。人们已经做出了各种努力来降低其计算复杂性（例如Gao等人（2020年）的研究）。

A noticeable success in time efficiency improvement is accomplished in ColBERT (Khattab and Zaharia, 2020) which conducts late interaction of query terms and document terms during runtime inference so that token embeddings for documents can be pre-computed. Using ColBERT re-ranking after a sparse retrieval model called DeepImpact (Mallia et al., 2021) can further enhance relevance. Similarly BECR (Yang et al., 2022), CEDR-KNRM (MacAvaney et al., 2019), and PreTTR (MacAvaney et al., 2020) have also adopted the late interaction architecture in their efficient transformer based re-ranking schemes.

ColBERT（Khattab和Zaharia，2020年）在提高时间效率方面取得了显著成功，它在运行时推理期间对查询词和文档词进行后期交互，从而可以预先计算文档的Token嵌入。在名为DeepImpact（Mallia等人，2021年）的稀疏检索模型之后使用ColBERT重排序可以进一步提高相关性。类似地，BECR（Yang等人，2022年）、CEDR - KNRM（MacAvaney等人，2019年）和PreTTR（MacAvaney等人，2020年）也在其高效的基于Transformer的重排序方案中采用了后期交互架构。

While the above work delivers good search relevance with late interaction, their improvement in time efficiency has come at the cost of a large storage space in hosting token-based precomputed document embeddings. For example, for the MS MARCO document corpus, the footprint of embedding vectors in ColBERT takes up to 1.6TB and hosting them in a disk incurs substantial time cost when many embeddings are fetched for re-ranking. It is highly desirable to reduce embedding footprints and host them in memory as much as possible for fast and high-throughput access and for I/O latency and contention avoidance, especially when an online re-ranking server is required to efficiently process many queries simultaneously.

虽然上述工作通过后期交互实现了良好的搜索相关性，但它们在时间效率上的提升是以存储基于Token的预计算文档嵌入所需的大量存储空间为代价的。例如，对于MS MARCO文档语料库，ColBERT中的嵌入向量占用高达1.6TB的空间，并且当为了重排序而提取大量嵌入时，将它们存储在磁盘上会产生大量的时间成本。非常有必要减少嵌入占用的空间，并尽可能将它们存储在内存中，以便进行快速、高吞吐量的访问，避免I/O延迟和争用，特别是当需要在线重排序服务器同时高效处理多个查询时。

The contribution of this paper is to propose a compact representation for contextual token em-beddings of documents called Contextual Quantization (CQ). Specifically, we adopt codebook-based quantization to compress embeddings while explicitly decoupling the ranking contributions of document specific and document-independent information in contextual embeddings. These ranking contributions are recovered with weighted composition after quantization decoding during online inference. Our CQ scheme includes a neural network model that jointly learns context-aware decomposition and quantization with an objective to preserve correct ranking scores and order margins. Our evaluation shows that CQ can effectively reduce the storage space of contextual representation by about 14 times for the tested datasets with insignificant online embedding recovery overhead and a small relevance degradation for re-ranking passages or documents. 695 Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics Volume 1: Long Papers, pages 695 - 707 May 22-27, 2022 (c)2022 Association for Computational Linguistics

本文的贡献是为文档的上下文Token嵌入提出一种名为上下文量化（CQ）的紧凑表示方法。具体来说，我们采用基于码本的量化方法来压缩嵌入，同时明确地将上下文嵌入中特定于文档和与文档无关的信息的排序贡献解耦。在在线推理期间，这些排序贡献在量化解码后通过加权组合进行恢复。我们的CQ方案包括一个神经网络模型，该模型联合学习上下文感知分解和量化，目标是保留正确的排序得分和顺序边界。我们的评估表明，对于测试数据集，CQ可以有效地将上下文表示的存储空间减少约14倍，在线嵌入恢复开销不显著，并且在重排序段落或文档时相关性下降较小。《计算语言学协会第60届年会论文集第1卷：长论文》，第695 - 707页，2022年5月22 - 27日 (c)2022计算语言学协会

## 2 Problem Definition and Related Work

## 2 问题定义与相关工作

The problem of neural text document re-ranking is defined as follows. Given a query with multiple terms and a set of candidate documents, rank these documents mainly based on their embeddings and query-document similarity. With a BERT-based re-ranking algorithm, typically a term is represented by a token, and thus in this paper, word "term" is used interchangeably with "token". This paper is focused on minimizing the space cost of token embeddings for fast online re-ranking inference.

神经文本文档重排序问题定义如下。给定一个包含多个词项的查询和一组候选文档，主要根据它们的嵌入和查询 - 文档相似度对这些文档进行排序。使用基于BERT的重排序算法时，通常一个词项由一个Token表示，因此在本文中，“词项”和“Token”这两个词可以互换使用。本文的重点是最小化Token嵌入的空间成本，以实现快速的在线重排序推理。

Deep contextual re-ranking models. Neural re-ranking has pursued representation-based or interaction-based algorithms (Guo et al., 2016; Dai et al., 2018; Xiong et al., 2017). Embedding interaction based on query and document terms shows an advantage in these studies. The transformer architecture based on BERT (Devlin et al., 2019) has been adopted to re-ranking tasks by using BERT's [CLS] token representation to summarize query and document interactions (Nogueira and Cho, 2019; Yang et al., 2019; Dai and Callan, 2019; Nogueira et al., 2019a; Li et al., 2020). Recently BERT is integrated in late term interaction (MacA-vaney et al., 2019; Hofstätter et al., 2020c,b; Mitra et al., 2021) which delivers strong relevance scores for re-ranking.

深度上下文重排序模型。神经重排序一直采用基于表示或基于交互的算法（郭等人，2016年；戴等人，2018年；熊等人，2017年）。基于查询和文档术语的嵌入交互在这些研究中显示出优势。基于BERT（德夫林等人，2019年）的Transformer架构已被应用于重排序任务，通过使用BERT的[CLS]标记表示来总结查询和文档的交互（诺盖拉和赵，2019年；杨等人，2019年；戴和卡兰，2019年；诺盖拉等人，2019a；李等人，2020年）。最近，BERT被集成到后期术语交互中（麦卡瓦尼等人，2019年；霍夫施泰特等人，2020c、b；米特拉等人，2021年），这为重排序提供了强大的相关性得分。

Efficiency optimization for transformer-based re-ranking. Several approaches have been proposed to reduce the time complexity of transformer-based ranking. For example, architecture simplification (Hofstätter et al., 2020c; Mitra et al., 2021), late interaction with precomputed token embeddings (MacAvaney et al., 2020), early exiting (Xin et al., 2020), and model distillation (Gao et al., 2020; Hofstätter et al., 2020a; Chen et al., 2020b). We will focus on the compression of token representation following the late-interaction work of ColBERT (Khattab and Zaharia, 2020) and BECR (Yang et al., 2022) as they deliver fairly competitive relevance scores for several well-known ad-hoc TREC datasets. These late-interaction approaches follow a dual-encoder design that separately encodes the two sets of texts, studied in various NLP tasks (Zhan et al., 2020; Chen et al., 2020a; Reimers and Gurevych, 2019; Karpukhin et al., 2020; Zhang et al., 2020).

基于Transformer的重排序效率优化。已经提出了几种方法来降低基于Transformer的排序的时间复杂度。例如，架构简化（霍夫施泰特等人，2020c；米特拉等人，2021年）、使用预计算标记嵌入的后期交互（麦卡瓦尼等人，2020年）、提前退出（辛等人，2020年）和模型蒸馏（高等人，2020年；霍夫施泰特等人，2020a；陈等人，2020b）。我们将专注于遵循ColBERT（卡塔布和扎哈里亚，2020年）和BECR（杨等人，2022年）的后期交互工作对标记表示进行压缩，因为它们为几个著名的即席TREC数据集提供了相当有竞争力的相关性得分。这些后期交互方法采用双编码器设计，分别对两组文本进行编码，这在各种自然语言处理任务中都有研究（詹等人，2020年；陈等人，2020a；赖默斯和古雷维奇，2019年；卡尔普欣等人，2020年；张等人，2020年）。

Several previous re-ranking model attempted to reduce the space need for contextual token em-beddings. ColBERT has considered an option of using a smaller dimension per vector and limiting 2 bytes per number as a scalar quantization. BECR (Yang et al., 2022) uses LSH for hashing-based contextual embedding compression (Ji et al., 2019). PreTTR (MacAvaney et al., 2020) uses a single layer encoder model to reduce the dimensionality of each token embedding. Following PreTTR, a contemporaneous work called SDR in Cohen et al. (2021) considers an autoencoder to reduce the dimension of representations, followed by an off-the-shelf scalar quantizer. For the autoencoder, it combines static BERT embeddings with contextual embeddings. Inspired by this study, our work decomposes contextual embeddings to decouple ranking contributions during vector quantization. Unlike SDR, CQ jointly learns the codebooks and decomposition for the document-independent and dependent components guided by a ranking loss.

之前有几个重排序模型试图减少上下文标记嵌入的空间需求。ColBERT考虑了每个向量使用较小维度并将每个数字限制为2字节作为标量量化的选项。BECR（杨等人，2022年）使用局部敏感哈希（LSH）进行基于哈希的上下文嵌入压缩（季等人，2019年）。PreTTR（麦卡瓦尼等人，2020年）使用单层编码器模型来降低每个标记嵌入的维度。在PreTTR之后，科恩等人（2021年）的一项同期工作SDR考虑使用自编码器来降低表示的维度，然后使用现成的标量量化器。对于自编码器，它将静态BERT嵌入与上下文嵌入相结合。受这项研究的启发，我们的工作在向量量化过程中分解上下文嵌入以分离排序贡献。与SDR不同，上下文量化（CQ）在排序损失的指导下联合学习与文档无关和相关组件的码本和分解。

Vector quantization. Vector quantization with codebooks was developed for data compression to assist approximate nearest neighbor search, for example, product quantizer (PQ) from Jégou et al. (2011), optimized product quantizer (OPQ) from Ge et al. (2013); residual additive quantizer(RQ) from Ai et al. (2015) and local search additive quantizer (LSQ) from Martinez et al. (2018). Recently such a technique has been used for compressing static word embeddings (Shu and Nakayama, 2018) and document representation vectors in a dense retrieval scheme called JPQ (Zhan et al., 2021a). None of the previous work has worked on quantization of contextual token vectors for the re-ranking task, and that is the focus of this paper.

向量量化。带有码本的向量量化是为数据压缩而开发的，以辅助近似最近邻搜索，例如，热古等人（2011年）提出的乘积量化器（PQ）、葛等人（2013年）提出的优化乘积量化器（OPQ）、艾等人（2015年）提出的残差加法量化器（RQ）以及马丁内斯等人（2018年）提出的局部搜索加法量化器（LSQ）。最近，这种技术已被用于压缩静态词嵌入（舒和中山，2018年）以及一种名为JPQ（詹等人，2021a）的密集检索方案中的文档表示向量。之前的工作都没有针对重排序任务对上下文标记向量进行量化，而这正是本文的重点。

## 3 Contextual Quantization

## 3 上下文量化

Applying vector quantization naively to token embedding compression does not ensure the ranking effectiveness because a quantizer-based compression is not lossless, and critical ranking signals could be lost during data transformation. To achieve a high compression ratio while maintaining the competitiveness in relevance, we consider the ranking contribution of a contextual token embedding for soft matching containing two components: 1) document specific component derived from the self attention among context in a document, 2) document-independent and corpus-specific component generated by the transformer model. Since for a reasonable sized document set, the second component is invariant to documents, its storage space is negligible compared to the first component. Thus the second part does not need compression. We focus on compressing the first component using codebooks. This decomposition strategy can reduce the relevance loss due to compression approximation, which allows a more aggressive compression ratio. Our integrated vector quantizer with contextual decomposition contains a ranking-oriented scheme with an encoder and decoder network for jointly learning codebooks and composition weights. Thus, the online composition of decompressed document-dependent information with document-independent information can retain a good relevance.

直接将向量量化应用于词元嵌入压缩并不能保证排序效果，因为基于量化器的压缩并非无损压缩，在数据转换过程中关键的排序信号可能会丢失。为了在保持相关性竞争力的同时实现高压缩率，我们考虑上下文词元嵌入对软匹配的排序贡献，它包含两个部分：1) 从文档上下文的自注意力中得出的文档特定部分；2) 由Transformer模型生成的与文档无关且特定于语料库的部分。对于规模合理的文档集，第二部分与文档无关，与第一部分相比，其存储空间可以忽略不计。因此，第二部分不需要压缩。我们专注于使用码本压缩第一部分。这种分解策略可以减少因压缩近似导致的相关性损失，从而允许采用更激进的压缩率。我们集成了上下文分解的向量量化器包含一个面向排序的方案，该方案带有编码器和解码器网络，用于联合学习码本和组合权重。因此，将解压缩后的与文档相关的信息和与文档无关的信息进行在线组合可以保持良好的相关性。

<!-- Media -->

<!-- figureText: OFFLINE ONLINE ${f}_{\mathbf{q},\mathbf{d}}$ Quantization Token Interaction Module Decoder $\widehat{\mathbf{E}}\left( {t}_{1}^{\Delta }\right) ,\cdots ,\widehat{\mathbf{E}}\left( {t}_{n}^{\Delta }\right)$ $\mathbf{E}\left( {q}_{1}\right) ,\cdots ,\mathbf{E}\left( {q}_{l}\right)$ $\widehat{\mathbf{E}}\left( {t}_{1}\right) ,\cdots ,\widehat{\mathbf{E}}\left( {t}_{n}\right)$ Composition Query Encoder ${q}_{1},\cdots ,{q}_{l}$ Query q Quantization ${\mathbf{s}}^{1},\cdots ,{\mathbf{s}}^{n}$ Code and Encoder Codebooks $\mathbf{E}\left( {\bar{t}}_{1}\right) ,\ldots ,\mathbf{E}\left( {\bar{t}}_{n}\right)$ $\mathbf{E}\left( {t}_{1}\right) ,\ldots ,\mathbf{E}\left( {t}_{n}\right)$ Doc-independent embeddings Doc Encoder Doc Encoder ${t}_{1},\cdots ,{t}_{n}$ Document d Vocabulary $V$ -->

<img src="https://cdn.noedgeai.com/0195ae43-4463-7b03-a8df-c4b334942da3_2.jpg?x=286&y=187&w=1092&h=401&r=0"/>

Figure 1: Offline processing and online ranking with contextual quantization

图1：使用上下文量化的离线处理和在线排序

<!-- Media -->

### 3.1 Vector Quantization and Contextual Decomposition

### 3.1 向量量化与上下文分解

A vector quantizer consists of two steps as discussed in Shu and Nakayama (2018). In the compression step, it encodes a real-valued vector (such as a token embedding vector in our case) into a short code using a neural encoder. The short code is a list of reference indices to the codewords in codebooks. During the decompression step, a neural decoder is employed to reconstruct the original vector from the code and codebooks.

如Shu和Nakayama（2018年）所讨论的，向量量化器包括两个步骤。在压缩步骤中，它使用神经编码器将实值向量（如我们这里的词元嵌入向量）编码为短代码。短代码是码本中码字的参考索引列表。在解压缩步骤中，使用神经解码器根据代码和码本重建原始向量。

The quantizer learns a set of $M$ codebooks $\left\{  {{\mathcal{C}}^{1},{\mathcal{C}}^{2},\cdots ,{\mathcal{C}}^{M}}\right\}$ and each codebook contains $K$ codewords $\left( {{\mathcal{C}}^{m} = \left\{  {{\mathbf{c}}_{1}^{m},{\mathbf{c}}_{2}^{m},\cdots ,{\mathbf{c}}_{K}^{m}}\right\}  }\right)$ of dimension $h$ . Then for any D-dimensional real valued vector $\mathbf{x} \in  {\mathbb{R}}^{D}$ ,the encoder compresses $\mathbf{x}$ into an $M$ dimensional code vector $\mathbf{s}$ . Each entry of code $\mathbf{s}$ is an integer $j$ ,denoting the $j$ -th codeword in codebook ${\mathcal{C}}^{m}$ . After locating all $M$ codewords as $\left\lbrack  {{\mathbf{c}}^{1},\cdots ,{\mathbf{c}}^{M}}\right\rbrack$ ,the original vector can be recovered with two options. For a product quantizer, the dimension of codeword is $h = D/M$ ,and the decompressed vector is $\widehat{\mathbf{x}} = {\mathbf{c}}^{1} \circ  {\mathbf{c}}^{2}\cdots  \circ  {\mathbf{c}}^{M}$ where symbol $\circ$ denotes vector concatenation. For an additive quantizerthe decompressed vector is $\widehat{\mathbf{x}} = \mathop{\sum }\limits_{{j = 1}}^{M}{\mathbf{c}}^{j}.$

量化器学习一组 $M$ 个码本 $\left\{  {{\mathcal{C}}^{1},{\mathcal{C}}^{2},\cdots ,{\mathcal{C}}^{M}}\right\}$ ，每个码本包含 $K$ 个维度为 $h$ 的码字 $\left( {{\mathcal{C}}^{m} = \left\{  {{\mathbf{c}}_{1}^{m},{\mathbf{c}}_{2}^{m},\cdots ,{\mathbf{c}}_{K}^{m}}\right\}  }\right)$ 。然后，对于任何D维实值向量 $\mathbf{x} \in  {\mathbb{R}}^{D}$ ，编码器将 $\mathbf{x}$ 压缩为一个 $M$ 维的代码向量 $\mathbf{s}$ 。代码 $\mathbf{s}$ 的每个条目是一个整数 $j$ ，表示码本 ${\mathcal{C}}^{m}$ 中的第 $j$ 个码字。在定位所有 $M$ 个码字为 $\left\lbrack  {{\mathbf{c}}^{1},\cdots ,{\mathbf{c}}^{M}}\right\rbrack$ 后，可以通过两种方式恢复原始向量。对于乘积量化器，码字的维度为 $h = D/M$ ，解压缩后的向量为 $\widehat{\mathbf{x}} = {\mathbf{c}}^{1} \circ  {\mathbf{c}}^{2}\cdots  \circ  {\mathbf{c}}^{M}$ ，其中符号 $\circ$ 表示向量拼接。对于加法量化器，解压缩后的向量为 $\widehat{\mathbf{x}} = \mathop{\sum }\limits_{{j = 1}}^{M}{\mathbf{c}}^{j}.$

Codebook-based contextual quantization. Now we describe how codebook-based compression is used in our contextual quantization. Given a token $t$ ,we consider its contextual embedding vector $\mathbf{E}\left( t\right)$ as a weighted combination of two components: $\mathbf{E}\left( {t}^{\Delta }\right)$ and $\mathbf{E}\left( \bar{t}\right) .\;\mathbf{E}\left( {t}^{\Delta }\right)$ captures the document-dependent component,and $\mathbf{E}\left( \bar{t}\right)$ captures the document-independent component discussed earlier. For a transformer model such as BERT, $\mathbf{E}\left( t\right)$ is the token output from the last encoder layer,and we obtain $\mathbf{E}\left( \bar{t}\right)$ by feeding [CLS] $\circ  t \circ$ [SEP] into BERT model and taking last layer’s output for $t$ .

基于码本的上下文量化。现在我们描述基于码本的压缩如何应用于我们的上下文量化。给定一个标记 $t$，我们将其上下文嵌入向量 $\mathbf{E}\left( t\right)$ 视为两个分量的加权组合：$\mathbf{E}\left( {t}^{\Delta }\right)$ 和 $\mathbf{E}\left( \bar{t}\right) .\;\mathbf{E}\left( {t}^{\Delta }\right)$ 捕获与文档相关的分量，而 $\mathbf{E}\left( \bar{t}\right)$ 捕获前面讨论过的与文档无关的分量。对于像 BERT 这样的Transformer模型，$\mathbf{E}\left( t\right)$ 是最后一个编码器层的标记输出，我们通过将 [CLS] $\circ  t \circ$ [SEP] 输入到 BERT 模型中，并取 $t$ 的最后一层输出来得到 $\mathbf{E}\left( \bar{t}\right)$。

During offline data compression, we do not explicitly derive $\mathbf{E}\left( {t}^{\Delta }\right)$ as we only need to store the compressed format of such a value, represented as a code. Let $\widehat{\mathbf{E}}\left( {t}^{\Delta }\right)$ be the recovered vector with codebook-based decompression, as a close approximation of $\mathbf{E}\left( {t}^{\Delta }\right)$ . Let $\widehat{\mathbf{E}}\left( t\right)$ be the final composed embedding used for online ranking with late-interaction. Then $\widehat{\mathbf{E}}\left( t\right)  = g\left( {\widehat{\mathbf{E}}\left( {t}^{\Delta }\right) ,\mathbf{E}\left( \bar{t}\right) }\right)$ where $g\left( \text{.}\right) {isafemplefeed} - {forwardnetworktocombine}$ two ranking contribution components.

在离线数据压缩期间，我们不会显式地推导 $\mathbf{E}\left( {t}^{\Delta }\right)$，因为我们只需要存储这样一个值的压缩格式，以代码形式表示。设 $\widehat{\mathbf{E}}\left( {t}^{\Delta }\right)$ 是通过基于码本的解压缩恢复的向量，作为 $\mathbf{E}\left( {t}^{\Delta }\right)$ 的近似值。设 $\widehat{\mathbf{E}}\left( t\right)$ 是用于在线排序的最终组合嵌入，采用后期交互方式。那么 $\widehat{\mathbf{E}}\left( t\right)  = g\left( {\widehat{\mathbf{E}}\left( {t}^{\Delta }\right) ,\mathbf{E}\left( \bar{t}\right) }\right)$，其中 $g\left( \text{.}\right) {isafemplefeed} - {forwardnetworktocombine}$ 是两个排序贡献分量。

The encoder/decoder neural architecture for contextual quantization. We denote a token in a document $\mathbf{d}$ as $t$ . The input to the quantization encoder is $\mathbf{E}\left( t\right)  \circ  \mathbf{E}\left( \bar{t}\right)$ . The output of the quantization encoder is the code vector $\mathbf{s}$ of dimension $M$ . Let code $\mathbf{s}$ be $\left( {{s}_{1},\cdots ,{s}_{m},\cdots ,{s}_{M}}\right)$ and each entry ${s}_{m}$ will be computed below in Eq. 4. This computation uses the hidden layer $\mathbf{h}$ defined as:

上下文量化的编码器/解码器神经网络架构。我们将文档 $\mathbf{d}$ 中的一个标记表示为 $t$。量化编码器的输入是 $\mathbf{E}\left( t\right)  \circ  \mathbf{E}\left( \bar{t}\right)$。量化编码器的输出是维度为 $M$ 的代码向量 $\mathbf{s}$。设代码 $\mathbf{s}$ 为 $\left( {{s}_{1},\cdots ,{s}_{m},\cdots ,{s}_{M}}\right)$，每个条目 ${s}_{m}$ 将在下面的公式 4 中计算。此计算使用定义为如下的隐藏层 $\mathbf{h}$：

$$
\mathbf{h} = \tanh \left( {{\mathbf{w}}_{0}\left( {\mathbf{E}\left( t\right)  \circ  \mathbf{E}\left( \bar{t}\right) }\right)  + {\mathbf{b}}_{0}}\right) . \tag{1}
$$

The dimension of $\mathbf{h}$ is fixed as $1 \times  {MK}/2$ . The hidden layer $\mathbf{a}$ is computed by a feed forward layer with a softplus activation (Eq. 2) with an output dimension of $M \times  K$ after reshaping,Let ${\mathbf{a}}^{m}$ be the $m$ -th row of this output.

$\mathbf{h}$ 的维度固定为 $1 \times  {MK}/2$。隐藏层 $\mathbf{a}$ 由一个前馈层计算得出，该前馈层使用 softplus 激活函数（公式 2），重塑后输出维度为 $M \times  K$。设 ${\mathbf{a}}^{m}$ 为该输出的第 $m$ 行。

$$
{\mathbf{a}}^{m} = \operatorname{softplus}\left( {{\mathbf{w}}_{1}^{m}\mathbf{h} + {\mathbf{b}}_{1}^{m}}\right) . \tag{2}
$$

To derive a discrete code entry for ${s}_{m}$ ,following the previous work (Shu and Nakayama, 2018), we apply the Gumbel-softmax trick (Maddison et al., 2017; Jang et al., 2017) as shown in Eq. 3, where the temperature $\tau$ is fixed at 1 and ${\epsilon }_{k}$ is a noise term sampled from the Gumbel distribution $- \log \left( {-\log \left( {\operatorname{Uniform}\left\lbrack  {0,1}\right\rbrack  }\right) }\right)$ . Here ${\mathbf{p}}^{m}$ is a vector with dimension $K.{\left( {\mathbf{p}}^{m}\right) }_{j}$ is the $j$ -th entry of the vector. Similarly, ${\left( {\mathbf{a}}^{m}\right) }_{j}$ is the $j$ -th entry of ${\mathbf{a}}^{m}$ .

为了为 ${s}_{m}$ 推导一个离散代码条目，遵循先前的工作（Shu 和 Nakayama，2018），我们应用 Gumbel - softmax 技巧（Maddison 等人，2017；Jang 等人，2017），如公式 3 所示，其中温度 $\tau$ 固定为 1，${\epsilon }_{k}$ 是从 Gumbel 分布 $- \log \left( {-\log \left( {\operatorname{Uniform}\left\lbrack  {0,1}\right\rbrack  }\right) }\right)$ 中采样的噪声项。这里 ${\mathbf{p}}^{m}$ 是一个维度为 $K.{\left( {\mathbf{p}}^{m}\right) }_{j}$ 的向量，是该向量的第 $j$ 个条目。类似地，${\left( {\mathbf{a}}^{m}\right) }_{j}$ 是 ${\mathbf{a}}^{m}$ 的第 $j$ 个条目。

$$
{\left( {\mathbf{p}}^{m}\right) }_{j} = \frac{\exp \left( {\log \left( {{\left( {\mathbf{a}}^{m}\right) }_{j} + {\epsilon }_{j}}\right) /\tau }\right) }{\mathop{\sum }\limits_{{{j}^{\prime } = 1}}^{K}\exp \left( {\log \left( {{\left( {\mathbf{a}}^{m}\right) }_{{j}^{\prime }} + {\epsilon }_{{j}^{\prime }}}\right) /\tau }\right) }. \tag{3}
$$

$$
{s}_{m} = \arg \mathop{\max }\limits_{{1 \leq  j \leq  K}}{\left( {\mathbf{p}}^{m}\right) }_{j}. \tag{4}
$$

In the decompression stage, the input to the quantization decoder is the code $\mathbf{s}$ ,and this decoder accesses $M$ codebooks $\left\{  {{\mathcal{C}}^{1},{\mathcal{C}}^{2},\cdots ,{\mathcal{C}}^{M}}\right\}$ as $M$ parameter matrices of size $K \times  h$ which will be learned. For each $m$ -entry of code $\mathbf{s},{s}_{m}$ value is the index of row vector in ${\mathcal{C}}^{m}$ to be used as its corresponding codeword. Once all codewords ${\mathbf{c}}^{1}$ to ${\mathbf{c}}^{M}$ are fetched,we recover the approximate vector $\widehat{\mathbf{E}}\left( {t}^{\Delta }\right)$ as $\mathop{\sum }\limits_{{j = 1}}^{M}{\mathbf{c}}^{j}$ for additive quantization or ${\mathbf{c}}^{1} \circ  {\mathbf{c}}^{2}\cdots  \circ  {\mathbf{c}}^{M}$ for product quantization.

在解压缩阶段，量化解码器的输入是代码 $\mathbf{s}$，并且该解码器将 $M$ 个码本 $\left\{  {{\mathcal{C}}^{1},{\mathcal{C}}^{2},\cdots ,{\mathcal{C}}^{M}}\right\}$ 作为大小为 $K \times  h$ 的 $M$ 个参数矩阵进行访问，这些参数矩阵将被学习。对于代码 $\mathbf{s},{s}_{m}$ 的每个 $m$ 项，其值是 ${\mathcal{C}}^{m}$ 中要用作其相应码字的行向量的索引。一旦获取了所有码字 ${\mathbf{c}}^{1}$ 到 ${\mathbf{c}}^{M}$，我们就可以恢复近似向量 $\widehat{\mathbf{E}}\left( {t}^{\Delta }\right)$，对于加法量化为 $\mathop{\sum }\limits_{{j = 1}}^{M}{\mathbf{c}}^{j}$，对于乘积量化为 ${\mathbf{c}}^{1} \circ  {\mathbf{c}}^{2}\cdots  \circ  {\mathbf{c}}^{M}$。

Next, we perform a composition with a one-layer or two-layer feed-forward network to derive the contextual embedding as $\widehat{\mathbf{E}}\left( t\right)  = g\left( {\widehat{\mathbf{E}}\left( {{t}^{\Delta },\mathbf{E}\left( \bar{t}\right) }\right) }\right.$ . With one feed-forward layer,

接下来，我们使用一层或两层前馈网络进行组合，以推导出上下文嵌入 $\widehat{\mathbf{E}}\left( t\right)  = g\left( {\widehat{\mathbf{E}}\left( {{t}^{\Delta },\mathbf{E}\left( \bar{t}\right) }\right) }\right.$。使用一层前馈层时，

$$
\widehat{\mathbf{E}}\left( t\right)  = \tanh \left( {{\mathbf{w}}_{2}\left( {\widehat{\mathbf{E}}\left( {t}^{\Delta }\right)  \circ  \mathbf{E}\left( \bar{t}\right) }\right)  + {\mathbf{b}}_{2}}\right) . \tag{5}
$$

The above encoder and decoder for quantization have parameter ${\mathbf{w}}_{0},{\mathbf{b}}_{0},{\mathbf{w}}_{1},{\mathbf{b}}_{1},{\mathbf{w}}_{2},{\mathbf{b}}_{2}$ ,and $\left\{  {{\mathcal{C}}^{1},{\mathcal{C}}^{2},\cdots ,{\mathcal{C}}^{M}}\right\}$ . These parameters are learned through training. Once these parameters are learned, the quantization model is fixed and the code for any new token embedding can be computed using Eq. 4 in offline processing.

上述用于量化的编码器和解码器具有参数 ${\mathbf{w}}_{0},{\mathbf{b}}_{0},{\mathbf{w}}_{1},{\mathbf{b}}_{1},{\mathbf{w}}_{2},{\mathbf{b}}_{2}$ 和 $\left\{  {{\mathcal{C}}^{1},{\mathcal{C}}^{2},\cdots ,{\mathcal{C}}^{M}}\right\}$。这些参数通过训练学习得到。一旦学习到这些参数，量化模型就固定下来，并且可以在离线处理中使用公式 4 计算任何新标记嵌入的代码。

Figure 1 depicts the flow of offline learning and the online inference with context quantization. Given a query with $l$ tokens $\left\{  {{q}_{1},{q}_{2},..{q}_{l}}\right\}$ , and a documents with $n$ tokens $\left\{  {{t}_{1},{t}_{2},..{t}_{n}}\right\}$ ,The query token embeddings encoded with a transformer based model (e.g. BERT) are denoted as $\mathbf{E}\left( {q}_{1}\right) ,\cdots ,\mathbf{E}\left( {q}_{l}\right)$ . The embeddings for document tokens through codebook base decompression are $\widehat{\mathbf{E}}\left( {t}_{1}\right) ,\cdots \widehat{\mathbf{E}}\left( {t}_{n}\right)$ . The online inference then uses the interaction of query tokens and document tokens defined in a re-ranking algorithm such as ColBERT to derive a ranking score (denoted as ${f}_{\mathbf{q},\mathbf{d}}$ ).

图 1 描述了离线学习和使用上下文量化的在线推理流程。给定一个包含 $l$ 个标记 $\left\{  {{q}_{1},{q}_{2},..{q}_{l}}\right\}$ 的查询，以及一个包含 $n$ 个标记 $\left\{  {{t}_{1},{t}_{2},..{t}_{n}}\right\}$ 的文档，使用基于变压器的模型（例如 BERT）编码的查询标记嵌入表示为 $\mathbf{E}\left( {q}_{1}\right) ,\cdots ,\mathbf{E}\left( {q}_{l}\right)$。通过基于码本的解压缩得到的文档标记嵌入为 $\widehat{\mathbf{E}}\left( {t}_{1}\right) ,\cdots \widehat{\mathbf{E}}\left( {t}_{n}\right)$。然后，在线推理使用重排序算法（如 ColBERT）中定义的查询标记和文档标记的交互来得出排名分数（表示为 ${f}_{\mathbf{q},\mathbf{d}}$）。

The purpose of injecting $\mathbf{E}\left( \bar{t}\right)$ in Eq. 1 is to decouple the document-independent ranking contribution from contextual embedding $\widehat{\mathbf{E}}\left( {t}^{\Delta }\right)$ so that this quantization encoder model will be learned to implicitly extract and compress the document-dependent ranking contribution.

在公式 1 中引入 $\mathbf{E}\left( \bar{t}\right)$ 的目的是将与文档无关的排名贡献从上下文嵌入 $\widehat{\mathbf{E}}\left( {t}^{\Delta }\right)$ 中分离出来，以便学习该量化编码器模型来隐式提取和压缩与文档相关的排名贡献。

Table 1 gives an example with several token codes produced by CQ for different sentences representing different contexts, and illustrates context awareness of CQ's encoding with a small codebook dimension $\left( {\mathrm{M} = \mathrm{K} = 4}\right)$ . For example,1 in code [4, $4,3,1\rbrack$ means the 4 -th dimension uses the first codeword of the corresponding codebook. Training of CQ uses the MS MARCO passage dataset discussed in Section 4 and these sentences are not from this dataset. Our observation from this example is described as follows. First, in general token codes in the same sentences are closer to each other, and token codes in different sentences, even with the same word "bank", are far away with a visible Hamming distance. Thus CQ coding allows a context-based separation among tokens residing in different contexts. Second, by looking at boldfaced tokens at each sentence, their distance in terms of contextual semantics and proximity is reflected to some degree in their CQ codes. For instance, a small Hamming code distance of three words "actor", "poet" and "writer" resembles their semantic and positional closeness. A larger code distance of two "bank"s in the ${3}^{rd}$ and ${4}^{th}$ sentences relates with their word sense and positional difference.

表1给出了一个示例，展示了CQ为代表不同上下文的不同句子生成的几个标记代码，并说明了在小码本维度$\left( {\mathrm{M} = \mathrm{K} = 4}\right)$下CQ编码的上下文感知能力。例如，代码[4, $4,3,1\rbrack$中的1表示第4维使用了相应码本的第一个码字。CQ的训练使用了第4节中讨论的MS MARCO段落数据集，这些句子并非来自该数据集。我们从这个示例中得到的观察结果如下。首先，一般来说，同一句子中的标记代码彼此更接近，而不同句子中的标记代码，即使包含相同的单词“bank”，也会有明显的汉明距离。因此，CQ编码允许对处于不同上下文中的标记进行基于上下文的区分。其次，通过观察每个句子中的粗体标记，它们在上下文语义和邻近性方面的距离在一定程度上反映在它们的CQ代码中。例如，“actor”（演员）、“poet”（诗人）和“writer”（作家）这三个词的汉明码距离较小，这与它们在语义和位置上的接近程度相似。${3}^{rd}$和${4}^{th}$句子中两个“bank”的代码距离较大，这与它们的词义和位置差异有关。

Training loss for parameter learning. We have explored three training loss functions. The first option is to follow a general quantizer (Shu and Nakayama, 2018) using the mean squared error (MSE) between the reconstructed and original embedding vectors of all token ${t}_{i}$ . Namely ${\mathcal{L}}_{MSE} = \sum {\begin{Vmatrix}\mathbf{E}\left( {t}_{i}\right)  - \widehat{\mathbf{E}}\left( {t}_{i}\right) \end{Vmatrix}}_{2}^{2}.$

用于参数学习的训练损失。我们探索了三种训练损失函数。第一种选择是遵循通用量化器（Shu和Nakayama，2018），使用所有标记的重构嵌入向量和原始嵌入向量之间的均方误差（MSE）${t}_{i}$。即${\mathcal{L}}_{MSE} = \sum {\begin{Vmatrix}\mathbf{E}\left( {t}_{i}\right)  - \widehat{\mathbf{E}}\left( {t}_{i}\right) \end{Vmatrix}}_{2}^{2}.$

The second option is the pairwise cross-entropy loss based on rank orders. After warming up with the MSE loss, we further train the quantizer using ${\mathcal{L}}_{\text{PairwiseCE }} = \sum \left( {-\mathop{\sum }\limits_{{j = {\mathbf{d}}^{ + },{\mathbf{d}}^{ - }}}{P}_{j}\log {P}_{j}}\right)$ where ${\mathbf{d}}^{ + }$ and ${\mathbf{d}}^{ - }$ are positive and negative documents for query $q$ .

第二种选择是基于排名顺序的成对交叉熵损失。在用MSE损失进行预热后，我们进一步使用${\mathcal{L}}_{\text{PairwiseCE }} = \sum \left( {-\mathop{\sum }\limits_{{j = {\mathbf{d}}^{ + },{\mathbf{d}}^{ - }}}{P}_{j}\log {P}_{j}}\right)$来训练量化器，其中${\mathbf{d}}^{ + }$和${\mathbf{d}}^{ - }$分别是查询$q$的正文档和负文档。

We adopt the third option which borrows the idea of MarginMSE loss from Hofstätter et al. (2020a) proposed for BERT-based ranking model distillation. In MarginMSE, a student model is trained to mimic the teacher model in terms of both ranking scores as well as the document relative order margins. In our case, the teacher model is the ranking model without quantization and the student model is the ranking model with quantization. It is defined as ${\mathcal{L}}_{\text{MarginMSE }} = \sum {\left( \left( {f}_{\mathbf{q},{\mathbf{d}}^{ + }} - {f}_{\mathbf{q},{\mathbf{d}}^{ - }}\right)  - \left( {\widehat{f}}_{\mathbf{q},{\mathbf{d}}^{ + }} - {\widehat{f}}_{\mathbf{q},{\mathbf{d}}^{ - }}\right) \right) }^{2}$ , where ${f}_{\mathbf{q},\mathbf{d}}$ and ${\widehat{f}}_{\mathbf{q},\mathbf{d}}$ denote the ranking score with and without quantization, respectively. The above loss function distills the ColBERT ranking characteristics into the CQ model for better preservation of ranking effectiveness.

我们采用第三种选择，借鉴了Hofstätter等人（2020a）为基于BERT的排序模型蒸馏提出的MarginMSE损失的思想。在MarginMSE中，训练学生模型使其在排序分数和文档相对顺序差距方面模仿教师模型。在我们的例子中，教师模型是未进行量化的排序模型，学生模型是进行了量化的排序模型。它被定义为${\mathcal{L}}_{\text{MarginMSE }} = \sum {\left( \left( {f}_{\mathbf{q},{\mathbf{d}}^{ + }} - {f}_{\mathbf{q},{\mathbf{d}}^{ - }}\right)  - \left( {\widehat{f}}_{\mathbf{q},{\mathbf{d}}^{ + }} - {\widehat{f}}_{\mathbf{q},{\mathbf{d}}^{ - }}\right) \right) }^{2}$，其中${f}_{\mathbf{q},\mathbf{d}}$和${\widehat{f}}_{\mathbf{q},\mathbf{d}}$分别表示量化和未量化时的排序分数。上述损失函数将ColBERT排序特征提炼到CQ模型中，以更好地保留排序效果。

<!-- Media -->

<table><tr><td>Context</td><td>Token codes</td></tr><tr><td>William Shakespeare was widely regarded as the world's greatest actor, poet, writer and dramatist.</td><td>writer actor poet $\left\lbrack  {4,4,3,1}\right\rbrack$$\left\lbrack  {4,4,3,1}\right\rbrack$$\left\lbrack  {1,4,3,1}\right\rbrack$</td></tr><tr><td>I would like to have either a cup of coffee or a good fiction to kill time.</td><td>coffee fiction [3,3,3,4]$\left\lbrack  {3,1,3,4}\right\rbrack$</td></tr><tr><td>She sat on the river bank across from a series of wide, large steps leading up a hill to the bank of America building.</td><td>1 ${}^{st}$ bank ${2}^{nd}$ bank $\left\lbrack  {3,1,4,2}\right\rbrack$$\left\lbrack  {4,1,3,1}\right\rbrack$</td></tr><tr><td>Some language techniques can recognize word senses in phrases such as a river bank and a bank building.</td><td>${1}^{st}$ bank ${2}^{nd}$ bank [4,3,2,2]$\left\lbrack  {3,1,1,4}\right\rbrack$</td></tr><tr><td>If you get a cold, you should drink a lot of water and get some rest.</td><td>${1}^{st}$ get ${2}^{nd}$ get $\left\lbrack  {2,2,4,2}\right\rbrack$$\left\lbrack  {2,1,2,4}\right\rbrack$</td></tr></table>

<table><tbody><tr><td>上下文</td><td>Token代码</td></tr><tr><td>威廉·莎士比亚被广泛认为是世界上最伟大的演员、诗人、作家和剧作家。</td><td>作家 演员 诗人 $\left\lbrack  {4,4,3,1}\right\rbrack$$\left\lbrack  {4,4,3,1}\right\rbrack$$\left\lbrack  {1,4,3,1}\right\rbrack$</td></tr><tr><td>我想要么喝杯咖啡，要么看一本好的小说来消磨时间。</td><td>咖啡 小说 [3,3,3,4]$\left\lbrack  {3,1,3,4}\right\rbrack$</td></tr><tr><td>她坐在河岸边，对面是一系列宽阔的大台阶，台阶通向一座小山，山上是美国银行大楼。</td><td>1 ${}^{st}$ 河岸 ${2}^{nd}$ 银行 $\left\lbrack  {3,1,4,2}\right\rbrack$$\left\lbrack  {4,1,3,1}\right\rbrack$</td></tr><tr><td>一些语言技术可以识别短语中的词义，如“河岸”（river bank）和“银行大楼”（bank building）。</td><td>${1}^{st}$ 河岸 ${2}^{nd}$ 银行 [4,3,2,2]$\left\lbrack  {3,1,1,4}\right\rbrack$</td></tr><tr><td>如果你感冒了，你应该多喝水，多休息。</td><td>${1}^{st}$ 患 ${2}^{nd}$ 进行 $\left\lbrack  {2,2,4,2}\right\rbrack$$\left\lbrack  {2,1,2,4}\right\rbrack$</td></tr></tbody></table>

Table 1: Example context-aware token codes produced by CQ using M=K=4 for the illustration purpose.

表1：为便于说明，CQ使用M = K = 4生成的示例上下文感知令牌代码。

<!-- Media -->

### 3.2 Related Online Space and Time Cost

### 3.2 相关在线空间和时间成本

Online space for document embeddings. The storage cost of the precomputed document embed-dings in a late-interaction re-ranking algorithm is dominating its online space need. To recover token-based document embeddings, an online server with contextual quantization stores three parts: codebooks, the short codes of tokens in each document, and the document-independent embeddings.

文档嵌入的在线空间。在后期交互重排序算法中，预计算文档嵌入的存储成本主导了其在线空间需求。为了恢复基于令牌的文档嵌入，具有上下文量化的在线服务器存储三部分内容：码本、每个文档中令牌的短代码以及与文档无关的嵌入。

Given a document collection of $Z$ documents of length $n$ tokens on average,let $V$ be the number of the distinct tokens. For $M$ codebooks with $M * K$ codewords of dimension $h$ ,we store each entry of a codeword with a 4-byte floating point number. Thus the space cost of codebooks is $M * K * h * 4$ bytes, and the space for document-independent em-beddings of dimension $D$ is $V * D * 4$ bytes. When $M = {16},K = {256},D = {128}$ as in our experiments, if we use the product quantization with the hidden dimension $h = 8$ ,the codebook size is 131 MB. In the WordPiece English token set for BERT, $V \approx  {32K}$ and the space for document-independent embeddings cost about ${16.4}\mathrm{{MB}}$ . Thus the space cost of the above two parts is insignificant.

给定一个包含$Z$个文档的文档集合，平均每个文档长度为$n$个令牌，设$V$为不同令牌的数量。对于$M$个码本，每个码本有$M * K$个维度为$h$的码字，我们用4字节浮点数存储每个码字的每个条目。因此，码本的空间成本为$M * K * h * 4$字节，维度为$D$的与文档无关的嵌入的空间为$V * D * 4$字节。在我们的实验中，当$M = {16},K = {256},D = {128}$时，如果我们使用隐藏维度为$h = 8$的乘积量化，码本大小为131 MB。在用于BERT的WordPiece英文令牌集中，$V \approx  {32K}$，与文档无关的嵌入的空间成本约为${16.4}\mathrm{{MB}}$。因此，上述两部分的空间成本微不足道。

The online space cost of token-based document embeddings is $Z * n * \left( {\frac{M{\log }_{2}K}{8} + 2}\right)$ bytes. Here each contextual token embedding of length $D$ is encoded into a code of length $M$ and the space of each code costs ${\log }_{2}K$ bits. For each document, we also need to store the IDs of its tokens in order to access document-independent embeddings. We use 2 bytes per token ID in our evaluation because the BERT dictionary based on WordPiece (Wu et al., 2016) tokenizer has about 32,000 tokens.

基于令牌的文档嵌入的在线空间成本为$Z * n * \left( {\frac{M{\log }_{2}K}{8} + 2}\right)$字节。这里，每个长度为$D$的上下文令牌嵌入被编码为长度为$M$的代码，每个代码的空间成本为${\log }_{2}K$位。对于每个文档，我们还需要存储其令牌的ID，以便访问与文档无关的嵌入。在我们的评估中，每个令牌ID使用2字节，因为基于WordPiece（Wu等人，2016）分词器的BERT字典大约有32,000个令牌。

In comparison, the space for document embed-dings in ColBERT with 2 bytes per number costs $Z * D * n * 2$ bytes. Then the space ratio of ColBERT without $\mathrm{{CQ}}$ and with $\mathrm{{CQ}}$ is about $\frac{{2D} \times  8}{M{\log }_{2}K + 2 \times  8}$ , which is about 14:1 when $D = {128},M = {16}$ and $K = {256}$ . BECR uses 5 layers of the refinement outcome with the BERT encoder for each token and stores each layer of the embedding with a 256 bit LSH signature. Thus the space cost ratio of BECR over ColBERT-CQ is approximately $\frac{5 \times  {256}}{M{\log }_{2}K + 2 \times  8}$ , which is about 9:1 when $M = {16}$ and $K = {256}$ . We can adjust the parameters of each of ColBERT, BECR, and ColBERT-CQ for a smaller space with a degraded relevance, and their space ratio to CQ remains large, which will be discussed in Section 4.

相比之下，ColBERT中每个数字使用2字节的文档嵌入空间成本为$Z * D * n * 2$字节。那么，没有$\mathrm{{CQ}}$和有$\mathrm{{CQ}}$的ColBERT的空间比约为$\frac{{2D} \times  8}{M{\log }_{2}K + 2 \times  8}$，当$D = {128},M = {16}$和$K = {256}$时，该比例约为14:1。BECR对每个令牌使用BERT编码器的5层细化结果，并使用256位LSH签名存储每层嵌入。因此，BECR与ColBERT - CQ的空间成本比约为$\frac{5 \times  {256}}{M{\log }_{2}K + 2 \times  8}$，当$M = {16}$和$K = {256}$时，该比例约为9:1。我们可以调整ColBERT、BECR和ColBERT - CQ的参数以在相关性降低的情况下获得更小的空间，并且它们与CQ的空间比仍然很大，这将在第4节中讨论。

Time cost for online decompression and composition. Let $k$ be the number of documents to re-rank. The cost of decompression with the short code of a token using the cookbooks is $O\left( {M * h}\right)$ for a product quantizer and $O\left( {M * D}\right)$ for an additive quantizer. Notice $M * h = D$ . For a one-layer feed-forward network as a composition to recover the final embedding, the total time cost for decompression and composition is $O\left( {\mathrm{k} * n * {D}^{2}}\right)$ with a product quantizer,and $O\left( {\mathrm{\;k} * n\left( {M * D + {D}^{2}}\right) }\right)$ with an additive quantizer. When using two hidden layers with $D$ dimensions in the first layer output, there is some extra time cost but the order of time complexity remains unchanged.

在线解压缩和组合的时间成本。设$k$为要重排序的文档数量。使用码本对令牌的短代码进行解压缩的成本，对于乘积量化器为$O\left( {M * h}\right)$，对于加法量化器为$O\left( {M * D}\right)$。注意$M * h = D$。对于作为组合以恢复最终嵌入的单层前馈网络，使用乘积量化器进行解压缩和组合的总时间成本为$O\left( {\mathrm{k} * n * {D}^{2}}\right)$，使用加法量化器为$O\left( {\mathrm{\;k} * n\left( {M * D + {D}^{2}}\right) }\right)$。当在第一层输出中使用具有$D$个维度的两个隐藏层时，会有一些额外的时间成本，但时间复杂度的阶数保持不变。

Noted that because of using feed-forward layers in final recovery, our contextual quantizer cannot take advantage of an efficiency optimization called asymmetric distance computation in Jégou et al. (2011). Since embedding recovery is only applied to top $k$ documents after the first-stage retrieval, the time efficiency for re-ranking is still reasonable without such an optimization.

注意到由于在最终恢复阶段使用了前馈层，我们的上下文量化器无法利用Jégou等人（2011年）提出的一种名为非对称距离计算的效率优化方法。由于嵌入恢复仅应用于第一阶段检索后的前$k$个文档，因此即使没有这种优化，重排序的时间效率仍然是合理的。

## 4 Experiments and Evaluation Results

## 4 实验与评估结果

### 4.1 Settings

### 4.1 设置

<!-- Media -->

<table><tr><td>Dataset</td><td>#Query</td><td>#Doc</td><td>Mean Doc Length</td><td>#Judgments per query</td></tr><tr><td>MS MARCO passage Dev</td><td>6980</td><td>8.8M</td><td>67.5</td><td>1</td></tr><tr><td>TREC DL 19 passage</td><td>200</td><td>-</td><td>-</td><td>21</td></tr><tr><td>TREC DL 20 passage</td><td>200</td><td>-</td><td>-</td><td>18</td></tr><tr><td>MS MARCO doc Dev</td><td>5193</td><td>3.2M</td><td>1460</td><td>1</td></tr><tr><td>TREC DL 19 doc</td><td>200</td><td>-</td><td>-</td><td>33</td></tr></table>

<table><tbody><tr><td>数据集</td><td>#查询</td><td>#文档</td><td>平均文档长度</td><td>每个查询的判断数量</td></tr><tr><td>MS MARCO段落开发集</td><td>6980</td><td>8.8M</td><td>67.5</td><td>1</td></tr><tr><td>TREC DL 19段落数据集</td><td>200</td><td>-</td><td>-</td><td>21</td></tr><tr><td>TREC DL 20段落数据集</td><td>200</td><td>-</td><td>-</td><td>18</td></tr><tr><td>MS MARCO文档开发集</td><td>5193</td><td>3.2M</td><td>1460</td><td>1</td></tr><tr><td>TREC DL 19文档数据集</td><td>200</td><td>-</td><td>-</td><td>33</td></tr></tbody></table>

Table 2: Dataset statistics. Mean doc length is the average number of WordPiece (Wu et al., 2016) tokens.

表2：数据集统计信息。平均文档长度是WordPiece（Wu等人，2016）标记的平均数量。

<!-- Media -->

Datasets and metrics. The well-known MS MARCO passage and document ranking datasets are used. As summarized the in Table 2, our evaluation uses the MS MARCO document and passage collections for document and passage ranking (Craswell et al., 2020; Campos et al., 2016). The original document and passage ranking tasks provide 367,013 and 502,940 training queries respectively, with about one judgment label per query. The development query sets are used for relevance evaluation. The TREC Deep Learning (DL) 2019 and 2020 tracks provide 200 test queries with many judgment labels per query for each task.

数据集和指标。使用了著名的MS MARCO段落和文档排序数据集。如表2总结所示，我们的评估使用MS MARCO文档和段落集合进行文档和段落排序（Craswell等人，2020；Campos等人，2016）。原始的文档和段落排序任务分别提供了367,013和502,940个训练查询，每个查询约有一个判断标签。开发查询集用于相关性评估。TREC深度学习（DL）2019和2020赛道为每个任务提供了200个测试查询，每个查询有多个判断标签。

Following the official leader-board standard, for the development sets, we report mean reciprocal rank (MRR@10, MRR@100) for relevance instead of using normalized discounted cumulative gain (NDCG) (Järvelin and Kekäläinen, 2002) because such a set has about one judgment label per query, which is too sparse to use NDCG. For TREC DL test sets which have many judgement lables per query, we report the commonly used NDCG@10 score. We also measure the dominating space need of the embeddings in bytes and re-ranking time latency in milliseconds. To evaluate latency, we uses an Amazon AWS g4dn instance with Intel Cascade Lake CPUs and an NVIDIA T4 GPU.

遵循官方排行榜标准，对于开发集，我们报告相关性的平均倒数排名（MRR@10，MRR@100），而不是使用归一化折损累积增益（NDCG）（Järvelin和Kekäläinen，2002），因为这样的集合每个查询约有一个判断标签，过于稀疏，无法使用NDCG。对于TREC DL测试集，每个查询有多个判断标签，我们报告常用的NDCG@10分数。我们还测量嵌入的主要空间需求（以字节为单位）和重排序时间延迟（以毫秒为单位）。为了评估延迟，我们使用了一个配备英特尔Cascade Lake CPU和NVIDIA T4 GPU的亚马逊AWS g4dn实例。

In all tables below that compare relevance, we perform paired t-test on ${95}\%$ confidence levels. In Tables 3,4,and 5,we mark the results with ${}^{4 \dagger  }$ ,if the compression method result in statistically significant degradation from the ColBERT baseline. In Table 6, ${}^{* \dagger  }$ ,is marked for numbers with statistically significant degradation from default setting in the first row.

在下面所有比较相关性的表格中，我们在${95}\%$置信水平上进行配对t检验。在表3、4和5中，如果压缩方法导致与ColBERT基线相比有统计学上显著的下降，我们用${}^{4 \dagger  }$标记结果。在表6中，与第一行默认设置相比有统计学上显著下降的数字用${}^{* \dagger  }$标记。

Choices of first-stage retrieval models. To retrieve top 1,000 results before re-ranking, we consider the standard fast BM25 method (Robertson and Zaragoza, 2009). We have also considered sparse and dense retrievers that outperform BM25. We have used uniCOIL (Lin and Ma, 2021; Gao et al., 2021) as an alternative sparse retriever in Table 3 because it achieves a similar level of relevance as end-to-end ColBERT with a dense retriever, and that of other learned sparse representations (Mallia et al., 2021; Formal et al., 2021b,a). ColBERT+uniCOIL has 0.369 MRR while end-to-end ColBERT has 0.360 MRR on MSMARCO Dev set. Moreover, retrieval with a sparse representation such as uniCOIL and BM25 normally uses much less computing resources than a dense retriever. Relevance numbers reported in some of the previous work on dense retrieval are derived from the exact search as an upper bound of accuracy. When non-exact retrieval techniques such as approximate nearest neighbor or maximum inner product search are used on a more affordable platform for large datasets, there is a visible loss of relevance (Lewis et al., 2021). It should be emphasized that the first stage model can be done by either a sparse or a dense retrieval, and this does not affect the applicability of $\mathrm{{CQ}}$ for the second stage as the focus of this paper.

第一阶段检索模型的选择。为了在重排序之前检索前1000个结果，我们考虑使用标准的快速BM25方法（Robertson和Zaragoza，2009）。我们还考虑了性能优于BM25的稀疏和密集检索器。在表3中，我们使用uniCOIL（Lin和Ma，2021；Gao等人，2021）作为替代的稀疏检索器，因为它与使用密集检索器的端到端ColBERT以及其他学习到的稀疏表示（Mallia等人，2021；Formal等人，2021b,a）达到了相似的相关性水平。在MSMARCO开发集上，ColBERT + uniCOIL的MRR为0.369，而端到端ColBERT的MRR为0.360。此外，使用uniCOIL和BM25等稀疏表示进行检索通常比密集检索器使用的计算资源少得多。之前一些关于密集检索的工作中报告的相关性数字是从精确搜索中得出的，作为准确性的上限。当在更经济实惠的平台上对大型数据集使用非精确检索技术（如近似最近邻或最大内积搜索）时，会出现明显的相关性损失（Lewis等人，2021）。应该强调的是，第一阶段模型可以通过稀疏或密集检索来完成，这并不影响本文重点关注的$\mathrm{{CQ}}$在第二阶段的适用性。

Re-ranking models and quantizers compared. We demonstrate the use of $\mathrm{{CQ}}$ for token compression in ColBERT in this paper. We compare its relevance with ColBERT, BECR and PreTTR. We chose to apply CQ to ColBERT because assuming embeddings are in memory, ColBERT is one of the fastest recent online re-ranking algorithms with strong relevance scores and CQ addresses its embedding storage weakness. Other re-ranking models compared include: BERT-base (Devlin et al., 2019), a cross encoder re-ranker, which takes a query and a document at run time and uses the last layers output from the BERT [CLS] token to generate a ranking score; TILDEv2 (Zhuang and Zuccon, 2021), which expands each document and additively aggregates precomputed neural scores.

比较的重排序模型和量化器。在本文中，我们展示了$\mathrm{{CQ}}$在ColBERT中用于标记压缩的应用。我们将其相关性与ColBERT、BECR和PreTTR进行比较。我们选择将CQ应用于ColBERT，因为假设嵌入存储在内存中，ColBERT是最近最快的在线重排序算法之一，具有较强的相关性分数，而CQ解决了其嵌入存储的弱点。其他比较的重排序模型包括：BERT - base（Devlin等人，2019），一种交叉编码器重排序器，它在运行时接受一个查询和一个文档，并使用BERT [CLS]标记的最后一层输出生成一个排序分数；TILDEv2（Zhuang和Zuccon，2021），它扩展每个文档并累加预计算的神经分数。

We also evaluate the use of unsupervised quantization methods discussed in Section 2 for ColBERT, including two product quantizers (PQ and OPQ), and two additive quantizers (RQ and LSQ).

我们还评估了第2节中讨论的无监督量化方法在ColBERT中的应用，包括两种乘积量化器（PQ和OPQ）和两种加法量化器（RQ和LSQ）。

Appendix A has additional details on the retrievers considered, re-ranker implementation, training, and relevance numbers cited.

附录A提供了有关所考虑的检索器、重排序器实现、训练和引用的相关性数字的更多详细信息。

<!-- Media -->

<table><tr><td>Model Specs.</td><td>Dev MRR@10</td><td>TREC DL19 NDCG@10</td><td>TREC DL20 NDCG@10</td></tr><tr><td colspan="4">Retrieval choices</td></tr><tr><td>BM25</td><td>0.172</td><td>0.425</td><td>0.453</td></tr><tr><td>docT5query</td><td>0.259</td><td>0.590</td><td>0.597</td></tr><tr><td>DeepCT*</td><td>0.243</td><td>0.572</td><td>-</td></tr><tr><td>TCT-ColBERT(v2)</td><td>0.358</td><td>-</td><td>-</td></tr><tr><td>JPQ*</td><td>0.341</td><td>0.677</td><td>-</td></tr><tr><td>DeepImpact</td><td>0.328</td><td>0.695</td><td>0.628</td></tr><tr><td>uniCOIL</td><td>0.347</td><td>0.703</td><td>0.675</td></tr><tr><td/><td colspan="3">Re-ranking baselines ( +BM25 retrieval)</td></tr><tr><td>BERT-base</td><td>0.349</td><td>0.682</td><td>0.655</td></tr><tr><td>BECR</td><td>0.323</td><td>0.682</td><td>0.655</td></tr><tr><td>TILDEv2*</td><td>0.333</td><td>0.676</td><td>0.686</td></tr><tr><td>ColBERT</td><td>0.355</td><td>0.701</td><td>0.723</td></tr><tr><td/><td colspan="3">Quantization ( +BM25 retrieval)</td></tr><tr><td>ColBERT-PO</td><td>${0.290}^{ \dagger  }\left( {-{18.3}\% }\right)$</td><td>0.684 (-2.3%)</td><td>0.714 (-1.2%)</td></tr><tr><td>ColBERT-OPQ</td><td>${0.324}^{ \dagger  }\left( {-{8.7}\% }\right)$</td><td>0.691 (-1.4%)</td><td>${0.688}^{ \dagger  }$ (-4.8%)</td></tr><tr><td>ColBERT-RO</td><td>-</td><td>${0.675}^{ \dagger  }\left( {-{3.7}\% }\right)$</td><td>0.696 (-3.7%)</td></tr><tr><td>ColBERT-LSQ</td><td>-</td><td>${0.664}^{ \dagger  }\left( {-{5.3}\% }\right)$</td><td>${0.656}^{ \dagger  }\left( {-{9.3}\% }\right)$</td></tr><tr><td>ColBERT-CO</td><td>0.352 (-0.8%)</td><td>0.704 (+0.4%)</td><td>0.716 (-1.0%)</td></tr><tr><td/><td colspan="3">( +uniCOIL retrieval)</td></tr><tr><td>ColBERT</td><td>0.369</td><td>0.692</td><td>0.701</td></tr><tr><td>ColBERT-CQ</td><td>${0.360}^{ \dagger  }\left( {-{2.4}\% }\right)$</td><td>0.696 (+0.6%)</td><td>0.720 (+2.7%)</td></tr></table>

<table><tbody><tr><td>模型规格</td><td>开发集前10名平均倒数排名（Dev MRR@10）</td><td>TREC DL19前10名归一化折损累积增益（TREC DL19 NDCG@10）</td><td>TREC DL20前10名归一化折损累积增益（TREC DL20 NDCG@10）</td></tr><tr><td colspan="4">检索选项</td></tr><tr><td>二元独立模型（BM25）</td><td>0.172</td><td>0.425</td><td>0.453</td></tr><tr><td>文档T5查询模型（docT5query）</td><td>0.259</td><td>0.590</td><td>0.597</td></tr><tr><td>深度上下文词项（DeepCT*）</td><td>0.243</td><td>0.572</td><td>-</td></tr><tr><td>TCT - 基于上下文的双向编码器表示（TCT - ColBERT(v2)）</td><td>0.358</td><td>-</td><td>-</td></tr><tr><td>联合量化查询（JPQ*）</td><td>0.341</td><td>0.677</td><td>-</td></tr><tr><td>深度影响模型（DeepImpact）</td><td>0.328</td><td>0.695</td><td>0.628</td></tr><tr><td>统一线圈模型（uniCOIL）</td><td>0.347</td><td>0.703</td><td>0.675</td></tr><tr><td></td><td colspan="3">重排序基线（+二元独立模型检索）</td></tr><tr><td>基础BERT模型（BERT - base）</td><td>0.349</td><td>0.682</td><td>0.655</td></tr><tr><td>双向编码器跨注意力重排序（BECR）</td><td>0.323</td><td>0.682</td><td>0.655</td></tr><tr><td>增强的双向编码器表示（TILDEv2*）</td><td>0.333</td><td>0.676</td><td>0.686</td></tr><tr><td>基于上下文的双向编码器表示（ColBERT）</td><td>0.355</td><td>0.701</td><td>0.723</td></tr><tr><td></td><td colspan="3">量化（+二元独立模型检索）</td></tr><tr><td>基于上下文的双向编码器表示 - 乘积量化（ColBERT - PO）</td><td>${0.290}^{ \dagger  }\left( {-{18.3}\% }\right)$</td><td>0.684 (-2.3%)</td><td>0.714 (-1.2%)</td></tr><tr><td>基于上下文的双向编码器表示 - 优化乘积量化（ColBERT - OPQ）</td><td>${0.324}^{ \dagger  }\left( {-{8.7}\% }\right)$</td><td>0.691 (-1.4%)</td><td>${0.688}^{ \dagger  }$ (-4.8%)</td></tr><tr><td>基于上下文的双向编码器表示 - 旋转量化（ColBERT - RO）</td><td>-</td><td>${0.675}^{ \dagger  }\left( {-{3.7}\% }\right)$</td><td>0.696 (-3.7%)</td></tr><tr><td>基于上下文的双向编码器表示 - 最小二乘量化（ColBERT - LSQ）</td><td>-</td><td>${0.664}^{ \dagger  }\left( {-{5.3}\% }\right)$</td><td>${0.656}^{ \dagger  }\left( {-{9.3}\% }\right)$</td></tr><tr><td>基于上下文的双向编码器表示 - 聚类量化（ColBERT - CO）</td><td>0.352 (-0.8%)</td><td>0.704 (+0.4%)</td><td>0.716 (-1.0%)</td></tr><tr><td></td><td colspan="3">（+统一线圈模型检索）</td></tr><tr><td>基于上下文的双向编码器表示（ColBERT）</td><td>0.369</td><td>0.692</td><td>0.701</td></tr><tr><td>基于上下文的双向编码器表示 - 码本量化（ColBERT - CQ）</td><td>${0.360}^{ \dagger  }\left( {-{2.4}\% }\right)$</td><td>0.696 (+0.6%)</td><td>0.720 (+2.7%)</td></tr></tbody></table>

Table 3: Relevance scores for MS MARCO passage ranking. The % degradation from ColBERT is listed and ’ $\dagger$ ’ is marked for statistically significant drop.

表3：MS MARCO段落排序的相关性得分。列出了相对于ColBERT的性能下降百分比，统计上显著下降的情况标记为’ $\dagger$ ’。

<!-- Media -->

### 4.2 A Comparison of Relevance

### 4.2 相关性比较

Table 3 and Table 4 show the ranking relevance in NDCG and MRR of the different methods and compare against the use of CQ with ColBERT (marked as ColBERT-CQ). We either report our experiment results or cite the relevance numbers from other papers with a * mark for such a model. For quantization approaches,we adopt $\mathrm{M} = {16},\mathrm{\;K} = {256}$ ,i.e. compression ratio 14:1 compared to ColBERT.

表3和表4展示了不同方法在归一化折损累积增益（NDCG）和平均倒数排名（MRR）方面的排序相关性，并与使用ColBERT结合上下文查询（CQ）（标记为ColBERT - CQ）的情况进行了比较。我们要么报告自己的实验结果，要么引用其他论文中的相关性数值，并对这类模型标记*号。对于量化方法，我们采用$\mathrm{M} = {16},\mathrm{\;K} = {256}$，即与ColBERT相比压缩比为14:1。

For the passage task, ColBERT outperforms other re-rankers in relevance for the tested cases. ColBERT-CQ after BM25 or uniCOIL retrieval only has a small relevance degradation with around $1\%$ or less,while only requiring $3\%$ of the storage of ColBERT. The relevance of the ColBERT-CQ+uniCOIL combination is also competitive to the one reported in Mallia et al. (2021) for the ColBERT+DeepImpact combination which has MRR 0.362 for the Dev query set, NDCG@100.722 for TREC DL 2019 and 0.691 for TREC DL 2020.

对于段落任务，在测试用例中，ColBERT在相关性方面优于其他重排器。在BM25或uniCOIL检索后使用ColBERT - CQ，相关性仅有小幅下降，降幅约为$1\%$或更小，而仅需ColBERT $3\%$的存储空间。ColBERT - CQ + uniCOIL组合的相关性也与Mallia等人（2021年）报告的ColBERT + DeepImpact组合的相关性相当，后者在开发查询集上的MRR为0.362，在2019年TREC DL上的NDCG@10为0.722，在2020年TREC DL上为0.691。

For the document re-ranking task, Table 4 similarly confirms the effectiveness of ColBERT-CQ. ColBERT-CQ and ColBERT after BM25 retrieval also perform well in general compared to the relevance results of the other baselines.

对于文档重排任务，表4同样证实了ColBERT - CQ的有效性。与其他基线的相关性结果相比，在BM25检索后使用ColBERT - CQ和ColBERT总体上也表现良好。

From both Table 3 and Table 4, we observe that in general, CQ significantly outperforms the other quantization approaches (PQ, OPQ, RQ, and LSQ). As an example, we further explain this by plotting the ranking score of ColBERT with and without a quantizer in Figure 2(a). Compared to OPQ, CQ trained with two loss functions generates ranking scores much closer to the original ColBERT ranking score, and this is also reflected in Kendall's $\tau$ correlation coefficients of top 1,000 re-ranked results between a quantized ColBERT and the original ColBERT (Figure 2(b)). There are two reasons that CQ outperforms the other quantizers: 1) The previous quantizers do not perform contextual decomposition to isolate intrinsic context-independent information in embeddings, and thus their approximation yields more relevance loss; 2) Their training loss function is not tailored to the re-ranking task.

从表3和表4中我们可以观察到，总体而言，上下文量化（CQ）明显优于其他量化方法（乘积量化（PQ）、优化乘积量化（OPQ）、残差量化（RQ）和最小平方量化（LSQ））。例如，我们通过在图2(a)中绘制有无量化器时ColBERT的排序得分来进一步解释这一点。与OPQ相比，使用两种损失函数训练的CQ生成的排序得分与原始ColBERT的排序得分更接近，这也反映在量化后的ColBERT与原始ColBERT前1000个重排结果的肯德尔$\tau$相关系数上（图2(b)）。CQ优于其他量化器有两个原因：1) 之前的量化器没有进行上下文分解以分离嵌入中固有的与上下文无关的信息，因此它们的近似会导致更多的相关性损失；2) 它们的训练损失函数并非针对重排任务量身定制。

<!-- Media -->

<table><tr><td>Model Specs.</td><td>Dev MRR@100</td><td>TREC DL19 NDCG@10</td></tr><tr><td colspan="3">Retrieval choices</td></tr><tr><td>BM25</td><td>0.203</td><td>0.446</td></tr><tr><td>docT5query</td><td>0.289</td><td>0.569</td></tr><tr><td>DeepCT*</td><td>0.320</td><td>0.544</td></tr><tr><td>TCT-ColBERT(v2)</td><td>0.351</td><td>-</td></tr><tr><td>JPQ*</td><td>0.401</td><td>0.623</td></tr><tr><td>uniCOIL</td><td>0.343</td><td>0.641</td></tr><tr><td colspan="3">Re-ranking baselines ( +BM25 retrieval)</td></tr><tr><td>BERT-base*</td><td>0.393</td><td>0.670</td></tr><tr><td>ColBERT</td><td>0.410</td><td>0.714</td></tr><tr><td colspan="3">Quantization ( +BM25 retrieval)</td></tr><tr><td>ColBERT-PQ</td><td>${0.400}^{ \dagger  }\left( {-{2.4}\% }\right)$</td><td>0.702 (-1.7%)</td></tr><tr><td>ColBERT-OPQ</td><td>${0.404}^{ \dagger  }\left( {-{1.5}\% }\right)$</td><td>0.704 (-1.4%)</td></tr><tr><td>ColBERT-RQ</td><td>-</td><td>0.704 (-1.4%)</td></tr><tr><td>ColBERT-LSQ</td><td>-</td><td>0.707 (-1.0%)</td></tr><tr><td>ColBERT-CO</td><td>${0.405}^{ \dagger  }\left( {-{1.2}\% }\right)$</td><td>0.712 (-0.3%)</td></tr></table>

<table><tbody><tr><td>模型规格</td><td>开发集前100的平均倒数排名（Dev MRR@100）</td><td>TREC DL19前10的归一化折损累积增益（TREC DL19 NDCG@10）</td></tr><tr><td colspan="3">检索选项</td></tr><tr><td>二元独立模型（BM25）</td><td>0.203</td><td>0.446</td></tr><tr><td>文档T5查询模型（docT5query）</td><td>0.289</td><td>0.569</td></tr><tr><td>深度上下文词项加权模型（DeepCT*）</td><td>0.320</td><td>0.544</td></tr><tr><td>TCT - 基于上下文的双向编码器表示（TCT - ColBERT(v2)）</td><td>0.351</td><td>-</td></tr><tr><td>联合量化模型（JPQ*）</td><td>0.401</td><td>0.623</td></tr><tr><td>统一线圈模型（uniCOIL）</td><td>0.343</td><td>0.641</td></tr><tr><td colspan="3">重排序基线（+BM25检索）</td></tr><tr><td>基础BERT模型（BERT - base*）</td><td>0.393</td><td>0.670</td></tr><tr><td>基于上下文的双向编码器表示检索模型（ColBERT）</td><td>0.410</td><td>0.714</td></tr><tr><td colspan="3">量化（+BM25检索）</td></tr><tr><td>基于上下文的双向编码器表示乘积量化模型（ColBERT - PQ）</td><td>${0.400}^{ \dagger  }\left( {-{2.4}\% }\right)$</td><td>0.702 (-1.7%)</td></tr><tr><td>基于上下文的双向编码器表示优化乘积量化模型（ColBERT - OPQ）</td><td>${0.404}^{ \dagger  }\left( {-{1.5}\% }\right)$</td><td>0.704 (-1.4%)</td></tr><tr><td>基于上下文的双向编码器表示残差量化模型（ColBERT - RQ）</td><td>-</td><td>0.704 (-1.4%)</td></tr><tr><td>基于上下文的双向编码器表示最小二乘量化模型（ColBERT - LSQ）</td><td>-</td><td>0.707 (-1.0%)</td></tr><tr><td>基于上下文的双向编码器表示码本优化模型（ColBERT - CO）</td><td>${0.405}^{ \dagger  }\left( {-{1.2}\% }\right)$</td><td>0.712 (-0.3%)</td></tr></tbody></table>

Table 4: Relevance scores for MS MARCO document ranking. The % degradation from ColBERT is listed and ’ $\dagger$ ’ is marked for statistically significant drop.

表4：MS MARCO文档排名的相关性得分。列出了相对于ColBERT的性能下降百分比，统计上显著下降的情况标记为’ $\dagger$ ’。

<!-- Media -->

### 4.3 Effectiveness on Space Reduction

### 4.3 空间缩减效果

<!-- Media -->

<!-- figureText: OPQ OPO CO-MSI CQ-MarginMSE Kedall's $\tau$ coefficient (b) Compressed Ranking Scores CQ-MSE CQ-MarginMSE ${t}^{ * }$ Original Ranking Scores (a) -->

<img src="https://cdn.noedgeai.com/0195ae43-4463-7b03-a8df-c4b334942da3_6.jpg?x=871&y=1512&w=590&h=337&r=0"/>

Figure 2: (a) Ranking score by quantized ColBERT with OPQ and CQ using two loss functions, vs. original ColBERT score. (b) Distribution of Kendall’s $\tau$ correlation coefficient between the 1,000 ranked results of quantized and original ColBERT.

图2：(a) 使用两种损失函数的OPQ和CQ量化后的ColBERT排名得分与原始ColBERT得分对比。(b) 量化后的ColBERT和原始ColBERT的1000个排名结果之间的肯德尔 $\tau$ 相关系数分布。

<!-- Media -->

Table 5 shows the estimated space size in bytes for embeddings in the MS MARCO document and passage corpora, and compares CQ with other approaches. Each MS MARCO document is divided into overlapped passage segments of size up to 400 tokens, and there are 60 tokens overlapped between two consecutive passage segments, following the ColBERT setup. As a result, the number of Word-Piece tokens per document changes from 1460 to about 2031 due to the addition of overlapping contextual tokens.

表5展示了MS MARCO文档和段落语料库中嵌入向量的估计字节空间大小，并将CQ与其他方法进行了比较。按照ColBERT的设置，每个MS MARCO文档被划分为最大400个词元的重叠段落片段，相邻两个段落片段之间有60个词元重叠。因此，由于添加了重叠的上下文词元，每个文档的词块（Word - Piece）词元数量从1460个变为约2031个。

<!-- Media -->

<table><tr><td rowspan="2">Model</td><td rowspan="2">Doc task Space</td><td colspan="4">Passage task</td></tr><tr><td>Space</td><td>Disk I/O</td><td>Latency</td><td>MRR@10</td></tr><tr><td>BECR</td><td>791G</td><td>89.9G</td><td>-</td><td>8ms</td><td>0.323</td></tr><tr><td>PreTTR*</td><td>-</td><td>2.6T</td><td>>182ms</td><td>>1000ms</td><td>0.358</td></tr><tr><td>TILDEv2*</td><td>-</td><td>5.2G</td><td>-</td><td>-</td><td>0.326</td></tr><tr><td>ColBERT</td><td>1.6T</td><td>143G</td><td>>182ms</td><td>16ms</td><td>0.355</td></tr><tr><td>ColBERT-small*</td><td>300G</td><td>26.8G</td><td>-</td><td>-</td><td>0.339</td></tr><tr><td>ColBERT-OPQ</td><td>112G</td><td>10.2G</td><td>-</td><td>56ms</td><td>${0.324}^{ \dagger  }$</td></tr><tr><td colspan="6">ColBERT-CQ</td></tr><tr><td>undecomposed</td><td>112G</td><td>10.2G</td><td>-</td><td>17ms</td><td>${0.339}^{ \dagger  }$</td></tr><tr><td>K=256</td><td>112G</td><td>10.2G</td><td>-</td><td>17ms</td><td>0.352</td></tr><tr><td>K=16</td><td>62G</td><td>5.6G</td><td>-</td><td>17ms</td><td>${0.339}^{ \dagger  }$</td></tr><tr><td>K=4</td><td>37G</td><td>3.4G</td><td>-</td><td>17ms</td><td>${0.326}^{ \dagger  }$</td></tr></table>

<table><tbody><tr><td rowspan="2">模型</td><td rowspan="2">文档任务空间</td><td colspan="4">段落任务</td></tr><tr><td>空间</td><td>磁盘输入/输出（Disk I/O）</td><td>延迟</td><td>前10名平均倒数排名（MRR@10）</td></tr><tr><td>二元错误纠正率（BECR）</td><td>791G</td><td>89.9G</td><td>-</td><td>8毫秒</td><td>0.323</td></tr><tr><td>预训练文本转换率（PreTTR*）</td><td>-</td><td>2.6T</td><td>大于182毫秒</td><td>大于1000毫秒</td><td>0.358</td></tr><tr><td>波浪号v2（TILDEv2*）</td><td>-</td><td>5.2G</td><td>-</td><td>-</td><td>0.326</td></tr><tr><td>科尔伯特（ColBERT）</td><td>1.6T</td><td>143G</td><td>大于182毫秒</td><td>16毫秒</td><td>0.355</td></tr><tr><td>小型科尔伯特（ColBERT-small*）</td><td>300G</td><td>26.8G</td><td>-</td><td>-</td><td>0.339</td></tr><tr><td>科尔伯特-最优乘积量化（ColBERT-OPQ）</td><td>112G</td><td>10.2G</td><td>-</td><td>56毫秒</td><td>${0.324}^{ \dagger  }$</td></tr><tr><td colspan="6">科尔伯特-上下文查询（ColBERT-CQ）</td></tr><tr><td>未分解的</td><td>112G</td><td>10.2G</td><td>-</td><td>17毫秒</td><td>${0.339}^{ \dagger  }$</td></tr><tr><td>K=256</td><td>112G</td><td>10.2G</td><td>-</td><td>17毫秒</td><td>0.352</td></tr><tr><td>K=16</td><td>62G</td><td>5.6G</td><td>-</td><td>17毫秒</td><td>${0.339}^{ \dagger  }$</td></tr><tr><td>K=4</td><td>37G</td><td>3.4G</td><td>-</td><td>17毫秒</td><td>${0.326}^{ \dagger  }$</td></tr></tbody></table>

Table 5: Embedding space size in bytes for the document ranking task and for the passage ranking task. Re-ranking time per query and relevance for top 1,000 passages in milliseconds on a GPU using the Dev query set. $\mathrm{M} = {16}$ . For ColBERT-OPQ and ColBERT-CQ-undecomposed, $\mathrm{K} = {256}$ . Assume passage embeddings in PreTTR and ColBERT do not fit in memory. ‘ ${}^{ \dagger  }$ ’ is marked for MRR numbers with statistically significant degradation from the ColBERT baseline.

表5：文档排序任务和段落排序任务的嵌入空间大小（以字节为单位）。使用开发查询集在GPU上对前1000个段落进行重新排序时，每个查询的重新排序时间和相关性（以毫秒为单位）。$\mathrm{M} = {16}$ 。对于ColBERT - OPQ和ColBERT - CQ - 未分解模型，$\mathrm{K} = {256}$ 。假设PreTTR和ColBERT中的段落嵌入无法全部存入内存。与ColBERT基线相比，平均倒数排名（MRR）数值出现统计学显著下降的情况标记为‘ ${}^{ \dagger  }$ ’。

<!-- Media -->

To demonstrate the tradeoff, we also list their estimated time latency and relevance in passage re-ranking as a reference and notice that more relevance comparison results are in Tables 3 and 4. The latency is the total time for embedding decompression/recovery and re-ranking.

为了展示这种权衡关系，我们还列出了它们在段落重新排序中的估计时间延迟和相关性作为参考，并注意到更多的相关性比较结果见表3和表4。延迟时间是嵌入解压缩/恢复和重新排序的总时间。

For PreTTR and ColBERT, we assume that their passage embedding data cannot fit in memory given their large data sizes. The disk I/O latency number is based on their passage embedding size and our test on a Samsung 870 QVO solid-state disk drive to fetch 1,000 passage embeddings randomly. Their I/O latency takes ${110}\mathrm{\;{ms}}$ or ${182}\mathrm{\;{ms}}$ with single-thread $\mathrm{I}/\mathrm{O}$ and with no $\mathrm{I}/\mathrm{O}$ contention,and their disk access can incur much more time when multiple queries are processed in parallel in a server dealing with many clients. For example, fetching 1,000 passage embeddings for each of ColBERT and PreTTR takes about $1,{001}\mathrm{\;{ms}}$ and $3,{870}\mathrm{\;{ms}}$ respectively when the server is handling 16 and 64 queries simultaneously with multiple threads.

对于PreTTR和ColBERT，考虑到它们的段落嵌入数据量很大，我们假设这些数据无法全部存入内存。磁盘I/O延迟时间是根据它们的段落嵌入大小以及我们在三星870 QVO固态硬盘上随机提取1000个段落嵌入的测试得出的。在单线程 $\mathrm{I}/\mathrm{O}$ 且无 $\mathrm{I}/\mathrm{O}$ 争用的情况下，它们的I/O延迟为 ${110}\mathrm{\;{ms}}$ 或 ${182}\mathrm{\;{ms}}$ ，并且当服务器同时处理多个客户端的多个查询时，磁盘访问可能会花费更多时间。例如，当服务器使用多线程同时处理16个和64个查询时，分别为ColBERT和PreTTR提取1000个段落嵌入大约需要 $1,{001}\mathrm{\;{ms}}$ 和 $3,{870}\mathrm{\;{ms}}$ 。

For other methods, their passage embedding data is relatively small and we assume that it can be preloaded in memory. The query latency reported in the 4-th column of Table 5 excludes the first-stage retrieval time. The default ColBERT uses embedding dimension 128 and 2 byte floating numbers. ColBERT-small denotes an optional configuration suggested from the ColBERT paper using 24 embedding dimensions and 2-byte floating numbers with a degraded relevance performance.

对于其他方法，它们的段落嵌入数据相对较小，我们假设可以将其预加载到内存中。表5第4列报告的查询延迟不包括第一阶段的检索时间。默认的ColBERT使用128维嵌入和2字节浮点数。ColBERT - small表示ColBERT论文中建议的一种可选配置，使用24维嵌入和2字节浮点数，但相关性性能有所下降。

As shown in Table 5, the embedding footprint of ColBERT CQ uses about 112GB and 10.2GB, respectively for document and passage re-ranking tasks. By looking at the latency difference of ColBERT with and without CQ, the time overhead of CQ for decompression and embedding recovery takes $1\mathrm{\;{ms}}$ per query,which is insignificant.

如表5所示，ColBERT CQ在文档和段落重新排序任务中的嵌入占用空间分别约为112GB和10.2GB。通过比较有和没有CQ的ColBERT的延迟差异，CQ在解压缩和嵌入恢复方面的时间开销为每个查询 $1\mathrm{\;{ms}}$ ，这一开销并不显著。

Compared with another quantizer ColBERT-OPQ, ColBERT-CQ can achieve the same level of space saving with $K = {256}$ while having a substantial relevance improvement. ColBERT-CQ with $K = 4$ achieves the same level of relevance as ColBERT-OPQ while yielding a storage reduction of ${67}\%$ and a latency reduction of about ${70}\%$ . Comparing ColBERT-CQ with no contextual decomposition, under the same space cost, ColBERT-CQ’s relevance is $4\%$ higher. CQ with $K = {16}$ achieves the same level relevance as ColBERT-CQ-undecomposed with $K = {256}$ ,while the storage of CQ reduces by 44%. Comparing with ColBERT-small which adopts more aggressive space reduction,ColBERT-CQ with $K = {16}$ would be competitive in relevance while its space is about $4\mathrm{x}$ smaller.

与另一种量化器ColBERT - OPQ相比，ColBERT - CQ在 $K = {256}$ 的情况下可以实现相同程度的空间节省，同时相关性有显著提高。使用 $K = 4$ 的ColBERT - CQ与ColBERT - OPQ达到相同的相关性水平，同时存储量减少了 ${67}\%$ ，延迟减少了约 ${70}\%$ 。与无上下文分解的ColBERT - CQ相比，在相同的空间成本下，ColBERT - CQ的相关性高出 $4\%$ 。使用 $K = {16}$ 的CQ与使用 $K = {256}$ 的未分解ColBERT - CQ达到相同的相关性水平，而CQ的存储量减少了44%。与采用更激进空间缩减策略的ColBERT - small相比，使用 $K = {16}$ 的ColBERT - CQ在相关性方面具有竞争力，同时其空间占用约小 $4\mathrm{x}$ 。

Comparing with other non-ColBERT baselines (BECR, PreTTR, and TILDEv2), ColBERT-CQ strikes a good balance across relevance, space and latency. For the fast CPU based model (BECR, TILDEv2), our model achieves better relevance with either lower or comparable space usage. For BECR, its embedding footprint with 89.9GB may fit in memory for MS MARCO passages, it becomes very expensive to configure a machine with much more memory for BECR's MS MARCO document embeddings with about 791GB.

与其他非ColBERT基线模型（BECR、PreTTR和TILDEv2）相比，ColBERT - CQ在相关性、空间和延迟方面取得了很好的平衡。对于基于快速CPU的模型（BECR、TILDEv2），我们的模型在空间使用更低或相当的情况下实现了更好的相关性。对于BECR，其89.9GB的嵌入占用空间对于MS MARCO段落数据可能可以存入内存，但为BECR的约791GB的MS MARCO文档嵌入配置内存更大的机器成本非常高。

### 4.4 Design Options for CQ

### 4.4 CQ的设计选项

Table 6 shows the relevance scores for the TREC deep learning passage ranking task with different design options for CQ. As an alternative setting, the codebooks in this table use $\mathrm{M} = {16}$ and $\mathrm{K} = {32}$ with compression ratio 21:1 compared to ColBERT. Row 1 is the default design configuration for CQ with product operators and 1 composition layer, and the MarginMSE loss function.

表6显示了在TREC深度学习段落排序任务中，CQ采用不同设计选项时的相关性得分。作为一种替代设置，此表中的码本使用 $\mathrm{M} = {16}$ 和 $\mathrm{K} = {32}$ ，与ColBERT相比压缩比为21:1。第1行是CQ的默认设计配置，采用乘积运算符和1个组合层，以及MarginMSE损失函数。

<!-- Media -->

<table><tr><td/><td>TREC19</td><td>TREC20</td></tr><tr><td>CQ, Product, 1 layer, MarginMSE</td><td>0.687</td><td>0.713</td></tr><tr><td colspan="3">Different model configurations</td></tr><tr><td>No decomposition. Product</td><td>${0.663}^{ \dagger  }$</td><td>0.686</td></tr><tr><td>No decomposition. Additive</td><td>${0.656}^{ \dagger  }$</td><td>0.693</td></tr><tr><td colspan="3">CQ, Product, 1 layer,</td></tr><tr><td>raw static embedding</td><td>${0.655}^{ \dagger  }$</td><td>${0.683}^{ \dagger  }$</td></tr><tr><td>CQ, Additive, 1 layer</td><td>0.693</td><td>0.703</td></tr><tr><td>CQ, Product, 2 layers</td><td>0.683</td><td>0.707</td></tr><tr><td>CQ, Additive, 2 layers</td><td>0.688</td><td>0.703</td></tr><tr><td colspan="3">Different training loss functions</td></tr><tr><td>CQ, Product, 1 layer, MSE</td><td>0.679</td><td>0.704</td></tr><tr><td>CQ, Product, 1 layer, PairwiseCE</td><td>0.683</td><td>0.705</td></tr></table>

<table><tbody><tr><td></td><td>TREC19（文本检索会议19）</td><td>TREC20（文本检索会议20）</td></tr><tr><td>复合查询（CQ），积性模型，1层，边缘均方误差损失（MarginMSE）</td><td>0.687</td><td>0.713</td></tr><tr><td colspan="3">不同的模型配置</td></tr><tr><td>无分解。积性模型</td><td>${0.663}^{ \dagger  }$</td><td>0.686</td></tr><tr><td>无分解。加性模型</td><td>${0.656}^{ \dagger  }$</td><td>0.693</td></tr><tr><td colspan="3">复合查询（CQ），积性模型，1层</td></tr><tr><td>原始静态嵌入</td><td>${0.655}^{ \dagger  }$</td><td>${0.683}^{ \dagger  }$</td></tr><tr><td>复合查询（CQ），加性模型，1层</td><td>0.693</td><td>0.703</td></tr><tr><td>复合查询（CQ），积性模型，2层</td><td>0.683</td><td>0.707</td></tr><tr><td>复合查询（CQ），加性模型，2层</td><td>0.688</td><td>0.703</td></tr><tr><td colspan="3">不同的训练损失函数</td></tr><tr><td>复合查询（CQ），积性模型，1层，均方误差损失（MSE）</td><td>0.679</td><td>0.704</td></tr><tr><td>复合查询（CQ），积性模型，1层，成对交叉熵损失（PairwiseCE）</td><td>0.683</td><td>0.705</td></tr></tbody></table>

Table 6: NDCG@10 of different design options for CQ in TREC DL passage ranking. If the compression method result in statistically significant degradation from the default setting, ${}^{4 \dagger  }$ ,is marked.

表6：TREC DL段落排序中CQ不同设计选项的NDCG@10。如果压缩方法导致与默认设置相比出现统计学上的显著性能下降，则标记为${}^{4 \dagger  }$。

<!-- Media -->

Different architecture or quantization options. Rows 2 and 3 of Table 6 denote CQ using product or additive operators without decomposing each embedding into two components, and there is about $4\%$ degradation without such decomposition.

不同的架构或量化选项。表6的第2行和第3行表示CQ使用乘积或加法运算符，而不将每个嵌入分解为两个组件，在不进行这种分解的情况下，性能下降约$4\%$。

Row 4 changes CQ using the raw static embed-dings of tokens from BERT instead of the upper layer outcome of BERT encoder and there is an up to 4.7% degradation. Notice such a strategy is used in SDR. From Row 5 to Row 7, we change CQ to use additive operators or use a two-layer composition. The performance of product or additive operators is in a similar level while the benefit of using two layers is relatively small.

第4行将CQ使用的标记的原始静态嵌入从BERT编码器的上层输出改为BERT的原始静态嵌入，性能下降高达4.7%。注意，这种策略在SDR中使用。从第5行到第7行，我们将CQ改为使用加法运算符或使用两层组合。乘积或加法运算符的性能处于相似水平，而使用两层的好处相对较小。

Different training loss functions for CQ. Last two rows of Table 6 use the MSE and PairwiseCE loss functions, respectively. There is an about 1.2% improvement using MarginMSE. Figure 2 gives an explanation why MarginMSE is more effective. While CQ trained with MSE and MarginMSE generates ranking scores close to the original ranking scores in Figure 2(a), the distribution of Kendall's $\tau$ correlation coefficients of 1,000 passages in Figure 2(b) shows that the passage rank order derived by CQ with the MarginMSE loss has a better correlation with that by ColBERT.

CQ的不同训练损失函数。表6的最后两行分别使用了MSE和PairwiseCE损失函数。使用MarginMSE有大约1.2%的性能提升。图2解释了为什么MarginMSE更有效。虽然用MSE和MarginMSE训练的CQ生成的排序分数与图2(a)中的原始排序分数接近，但图2(b)中1000个段落的肯德尔$\tau$相关系数分布表明，使用MarginMSE损失的CQ得出的段落排序顺序与ColBERT得出的排序顺序有更好的相关性。

## 5 Concluding Remarks

## 5 结论

Our evaluation shows the effectiveness of CQ used for ColBERT in compressing the space of token embeddings with about ${14} : 1$ ratio while incurring a small relevance degradation in MS MARCO passage and document re-ranking tasks. The quantized token-based document embeddings for the tested cases can be hosted in memory for fast and high-throughput access. This is accomplished by a neural network that decomposes ranking contributions of contextual embeddings, and jointly trains context-aware decomposition and quantization with a loss function preserving ranking accuracy. The online time cost to decompress and recover embeddings is insignificant with $1\mathrm{\;{ms}}$ for the tested cases. The CQ implementation is available at https://github.com/yingrui-yang/ContextualQuantizer.

我们的评估表明，CQ用于ColBERT在以约${14} : 1$的比例压缩标记嵌入空间方面是有效的，同时在MS MARCO段落和文档重排序任务中仅导致较小的相关性下降。对于测试用例，基于量化标记的文档嵌入可以存储在内存中，以便进行快速和高吞吐量的访问。这是通过一个神经网络实现的，该网络分解上下文嵌入的排序贡献，并使用一个保留排序准确性的损失函数联合训练上下文感知的分解和量化。对于测试用例，解压缩和恢复嵌入的在线时间成本在$1\mathrm{\;{ms}}$的情况下可以忽略不计。CQ的实现可在https://github.com/yingrui-yang/ContextualQuantizer获取。

Our CQ framework is also applicable to the contemporaneous work ColBERTv2 (Santhanam et al., 2021). Using uniCOIL scores for the first-stage sparse retrieval and ColBERTv2+CQ (M=16, K=256) for top 1,000 passage reranking, we achieve 0.387 MRR@10 on the MSMARCO passage Dev set, 0.746 NDCG@10 on TREC DL19, and 0.726 NDCG@10 on DL20 with about 10.2GB embedding space footprint. Notice that ColBERTv2 achieves a higher MRR@10 number 0.397 for the passage Dev set when used as a standalone retriever (Santhanam et al., 2021) and dense retrieval with such a multi-vector representation is likely to be much more expensive than retrieval with a sparse representation on a large dataset. The previous work in dense retrieval has often employed faster but approximate search, but that comes with a visible loss of relevance (Lewis et al., 2021). Thus the above relevance number using ColBERTv2+CQ for re-ranking with uni-COIL sparse retrieval is fairly strong, achievable with a reasonable latency and limited computing resource. Its embedding space size is ${2.8}\mathrm{x}$ smaller than the 29GB space cost in the standalone Col-BERTv2 (Santhanam et al., 2021) for MS MARCO passages. Our future work is to investigate the above issue further and study the use of $\mathrm{{CQ}}$ in the other late-interaction re-ranking methods.

我们的CQ框架也适用于同期工作ColBERTv2（Santhanam等人，2021）。使用uniCOIL分数进行第一阶段的稀疏检索，并使用ColBERTv2+CQ（M = 16，K = 256）对前1000个段落进行重排序，我们在MSMARCO段落开发集上实现了0.387的MRR@10，在TREC DL19上实现了0.746的NDCG@10，在DL20上实现了0.726的NDCG@10，嵌入空间占用约10.2GB。注意，当ColBERTv2作为独立的检索器使用时，在段落开发集上实现了更高的MRR@10值0.397（Santhanam等人，2021），并且在大型数据集上使用这种多向量表示进行密集检索可能比使用稀疏表示进行检索昂贵得多。之前在密集检索方面的工作通常采用更快但近似的搜索，但这会导致明显的相关性损失（Lewis等人，2021）。因此，上述使用ColBERTv2+CQ结合uni - COIL稀疏检索进行重排序的相关性指标相当不错，可以在合理的延迟和有限的计算资源下实现。其嵌入空间大小比独立的Col - BERTv2（Santhanam等人，2021）在MS MARCO段落上的29GB空间成本小${2.8}\mathrm{x}$。我们未来的工作是进一步研究上述问题，并研究在其他后期交互重排序方法中使用$\mathrm{{CQ}}$的情况。

Acknowledgments. We thank Cindy Zhao, Ji-ahua Wang, and anonymous referees for their valuable comments and/or help. This work is supported in part by NSF IIS-2040146 and by a Google faculty research award. It has used the Extreme Science and Engineering Discovery Environment supported by NSF ACI-1548562. Any opinions, findings, conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the NSF.

致谢。我们感谢Cindy Zhao、王佳华和匿名评审人员提供的宝贵意见和/或帮助。这项工作部分得到了美国国家科学基金会（NSF）IIS - 2040146项目以及谷歌教师研究奖的支持。它使用了由NSF ACI - 1548562支持的极端科学与工程发现环境。本材料中表达的任何观点、发现、结论或建议均为作者本人的观点，不一定反映美国国家科学基金会的观点。

## References

## 参考文献

Liefu Ai, Junqing Yu, Zenbin Wu, Yunfeng He, and Tao Guan. 2015. Optimized residual vector quantization for efficient approximate nearest neighbor search. Multimedia Systems, 23:169-181.

艾立富、余俊清、吴振斌、何云峰和管涛。2015。用于高效近似最近邻搜索的优化残差向量量化。《多媒体系统》，23:169 - 181。

Daniel Fernando Campos, Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, Li Deng, and Bhaskar Mitra. 2016. Ms marco: A human generated machine reading comprehension dataset. ArXiv, abs/1611.09268.

丹尼尔·费尔南多·坎波斯（Daniel Fernando Campos）、特里·阮（Tri Nguyen）、米尔·罗森伯格（Mir Rosenberg）、宋霞（Xia Song）、高剑锋（Jianfeng Gao）、索拉布·蒂瓦里（Saurabh Tiwary）、兰甘·马朱姆德（Rangan Majumder）、邓力（Li Deng）和巴斯卡尔·米特拉（Bhaskar Mitra）。2016年。MS MARCO：一个人工生成的机器阅读理解数据集。预印本库（ArXiv），编号：abs/1611.09268。

Jiecao Chen, Liu Yang, Karthik Raman, Michael Ben-dersky, Jung-Jung Yeh, Yun Zhou, Marc Najork, D. Cai, and Ehsan Emadzadeh. 2020a. Dipair: Fast and accurate distillation for trillion-scale text matching and pair modeling. In EMNLP.

陈杰草（Jiecao Chen）、刘洋（Liu Yang）、卡尔蒂克·拉曼（Karthik Raman）、迈克尔·本德尔斯基（Michael Ben-dersky）、叶俊杰（Jung-Jung Yeh）、周云（Yun Zhou）、马克·纳约克（Marc Najork）、蔡D（D. Cai）和埃桑·埃马扎德（Ehsan Emadzadeh）。2020a。Dipair：用于万亿级文本匹配和配对建模的快速准确蒸馏方法。发表于自然语言处理经验方法会议（EMNLP）。

Xuanang Chen, B. He, Kai Hui, L. Sun, and Yingfei Sun. 2020b. Simplified tinybert: Knowledge distillation for document retrieval. ArXiv, abs/2009.07531.

陈轩昂（Xuanang Chen）、何B（B. He）、惠凯（Kai Hui）、孙L（L. Sun）和孙英飞（Yingfei Sun）。2020b。简化版TinyBERT：用于文档检索的知识蒸馏。预印本库（ArXiv），编号：abs/2009.07531。

Nachshon Cohen, Amit Portnoy, Besnik Fetahu, and Amir Ingber. 2021. Sdr: Efficient neural re-ranking using succinct document representation. ArXiv, 2110.02065.

纳赫雄·科恩（Nachshon Cohen）、阿米特·波特诺伊（Amit Portnoy）、贝斯尼克·费塔胡（Besnik Fetahu）和阿米尔·英格伯（Amir Ingber）。2021年。SDR：使用简洁文档表示的高效神经重排序方法。预印本库（ArXiv），编号：2110.02065。

Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Fernando Campos, and Ellen M. Voorhees. 2020. Overview of the trec 2020 deep learning track. ArXiv, abs/2102.07662.

尼克·克拉斯韦尔（Nick Craswell）、巴斯卡尔·米特拉（Bhaskar Mitra）、埃米内·伊尔马兹（Emine Yilmaz）、丹尼尔·费尔南多·坎波斯（Daniel Fernando Campos）和埃伦·M·沃里斯（Ellen M. Voorhees）。2020年。2020年文本检索会议（TREC）深度学习赛道综述。预印本库（ArXiv），编号：abs/2102.07662。

Zhuyun Dai and J. Callan. 2019. Deeper text understanding for ir with contextual neural language modeling. Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval.

戴珠云（Zhuyun Dai）和J.卡兰（J. Callan）。2019年。通过上下文神经语言建模实现信息检索（IR）中更深入的文本理解。第42届美国计算机协会信息检索研究与发展国际会议（ACM SIGIR）论文集。

Zhuyun Dai and Jamie Callan. 2020. Context-aware term weighting for first stage passage retrieval. SI-GIR.

戴珠云（Zhuyun Dai）和杰米·卡兰（Jamie Callan）。2020年。用于第一阶段段落检索的上下文感知词项加权方法。信息检索研究与发展会议（SIGIR）。

Zhuyun Dai, Chenyan Xiong, Jamie Callan, and Zhiyuan Liu. 2018. Convolutional neural networks for soft-matching n-grams in ad-hoc search. In ${WSDM}$ ,pages ${126} - {134}$ .

戴珠云（Zhuyun Dai）、熊晨彦（Chenyan Xiong）、杰米·卡兰（Jamie Callan）和刘志远（Zhiyuan Liu）。2018年。用于即席搜索中n元组软匹配的卷积神经网络。发表于${WSDM}$，第${126} - {134}$页。

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In ${NAACL}$ .

雅各布·德夫林（Jacob Devlin）、张明伟（Ming-Wei Chang）、肯顿·李（Kenton Lee）和克里斯蒂娜·图托纳娃（Kristina Toutanova）。2019年。BERT：用于语言理解的深度双向变换器预训练。发表于${NAACL}$。

Thibault Formal, C. Lassance, Benjamin Piwowarski, and Stéphane Clinchant. 2021a. Splade v2: Sparse lexical and expansion model for information retrieval. ArXiv, abs/2109.10086.

蒂博·福尔马尔（Thibault Formal）、C.拉桑斯（C. Lassance）、本杰明·皮沃瓦尔斯基（Benjamin Piwowarski）和斯特凡·克林尚（Stéphane Clinchant）。2021a。SPLADE v2：用于信息检索的稀疏词法和扩展模型。预印本库（ArXiv），编号：abs/2109.10086。

Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant. 2021b. SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking, pages 2288- 2292. ACM.

蒂博·福尔马尔（Thibault Formal）、本杰明·皮沃瓦尔斯基（Benjamin Piwowarski）和斯特凡·克林尚（Stéphane Clinchant）。2021b。SPLADE：用于第一阶段排序的稀疏词法和扩展模型，第2288 - 2292页。美国计算机协会（ACM）。

Luyu Gao and Jamie Callan. 2021. Condenser: a pretraining architecture for dense retrieval. In Proceedings of the 2021 Conference on Empirical Methods

高璐宇（Luyu Gao）和杰米·卡兰（Jamie Callan）。2021年。冷凝器（Condenser）：一种用于密集检索的预训练架构。发表于2021年自然语言处理经验方法会议论文集

in Natural Language Processing, pages 981-993, Online and Punta Cana, Dominican Republic. Asso-

，第981 - 993页，线上会议及多米尼加共和国蓬塔卡纳。

ciation for Computational Linguistics.

计算语言学协会。

Luyu Gao, Zhuyun Dai, and J. Callan. 2020. Understanding bert rankers under distillation. Proceedings of SIGIR.

高璐宇（Luyu Gao）、戴珠云（Zhuyun Dai）和J.卡兰（J. Callan）。2020年。理解蒸馏下的BERT排序器。信息检索研究与发展会议（SIGIR）论文集。

Luyu Gao, Zhuyun Dai, and Jamie Callan. 2021. COIL: revisit exact lexical match in information retrieval with contextualized inverted list. NAACL.

高璐宇（Luyu Gao）、戴珠云（Zhuyun Dai）和杰米·卡兰（Jamie Callan）。2021年。COIL：通过上下文倒排列表重新审视信息检索中的精确词法匹配。北美计算语言学协会会议（NAACL）。

Tiezheng Ge, Kaiming He, Qifa Ke, and Jian Sun. 2013. Optimized product quantization for approximate nearest neighbor search. CVPR, pages 2946-2953.

葛铁铮、何恺明、柯琦发和孙剑。2013年。用于近似最近邻搜索的优化乘积量化。《计算机视觉与模式识别会议论文集》（CVPR），第2946 - 2953页。

J. Guo, Y. Fan, Qingyao Ai, and W. Croft. 2016. A deep relevance matching model for ad-hoc retrieval. CIKM.

郭J.、范Y.、艾清瑶和克罗夫特W.。2016年。用于即席检索的深度相关性匹配模型。《信息与知识管理大会》（CIKM）。

Sebastian Hofstätter, Sophia Althammer, Michael Schröder, Mete Sertkan, and Allan Hanbury. 2020a. Improving efficient neural ranking models with cross-architecture knowledge distillation. ArXiv, abs/2010.02666.

塞巴斯蒂安·霍夫施泰特、索菲娅·阿尔塔默、迈克尔·施罗德、梅特·塞尔特坎和艾伦·汉伯里。2020a。通过跨架构知识蒸馏改进高效神经排序模型。《预印本》（ArXiv），编号：abs/2010.02666。

Sebastian Hofstätter, Hamed Zamani, Bhaskar Mitra, Nick Craswell, and A. Hanbury. 2020b. Local self-attention over long text for efficient document retrieval. SIGIR.

塞巴斯蒂安·霍夫施泰特、哈米德·扎马尼、巴斯卡尔·米特拉、尼克·克拉斯韦尔和汉伯里A.。2020b。用于高效文档检索的长文本局部自注意力机制。《信息检索研究与发展会议》（SIGIR）。

Sebastian Hofstätter, Markus Zlabinger, and A. Hanbury. 2020c. Interpretable & time-budget-constrained con-textualization for re-ranking. In ${ECAI}$ .

塞巴斯蒂安·霍夫施泰特、马库斯·兹拉宾格和汉伯里A.。2020c。用于重排序的可解释且受时间预算约束的上下文建模。见${ECAI}$。

Eric Jang, Shixiang Shane Gu, and Ben Poole. 2017. Categorical reparameterization with gumbel-softmax. ${ICLR}$ .

埃里克·张、顾世翔和本·普尔。2017年。使用Gumbel - Softmax进行类别重参数化。${ICLR}$。

Kalervo Järvelin and Jaana Kekäläinen. 2002. Cumulated gain-based evaluation of ir techniques. ACM Transactions on Information Systems (TOIS), 20(4):422-446.

卡勒沃·亚尔维林和亚娜·凯卡拉伊宁。2002年。基于累积增益的信息检索技术评估。《美国计算机协会信息系统汇刊》（ACM Transactions on Information Systems，TOIS），20(4):422 - 446。

Hervé Jégou, Matthijs Douze, and Cordelia Schmid. 2011. Product quantization for nearest neighbor search. IEEE Transactions on Pattern Analysis and Machine Intelligence, 33:117-128.

埃尔韦·热古、马蒂亚斯·杜泽和科迪莉亚·施密德。2011年。用于最近邻搜索的乘积量化。《电气与电子工程师协会模式分析与机器智能汇刊》（IEEE Transactions on Pattern Analysis and Machine Intelligence），33:117 - 128。

Shiyu Ji, Jinjin Shao, and Tao Yang. 2019. Efficient interaction-based neural ranking with locality sensitive hashing. In ${WWW}$ .

季世玉、邵金金和杨涛。2019年。基于局部敏感哈希的高效基于交互的神经排序。见${WWW}$。

Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2017. Billion-scale similarity search with gpus. IEEE Transactions on Big Data.

杰夫·约翰逊、马蒂亚斯·杜泽和埃尔韦·热古。2017年。使用GPU进行十亿级相似度搜索。《电气与电子工程师协会大数据汇刊》（IEEE Transactions on Big Data）。

V. Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Yu Wu, Sergey Edunov, Danqi Chen, and Wen tau Yih. 2020. Dense passage retrieval for open-domain question answering. ArXiv, abs/2010.08191.

卡尔普欣V.、奥古兹B.、闵世元、帕特里克·刘易斯、吴乐德尔·余、谢尔盖·叶杜诺夫、陈丹琦和易文涛。2020年。用于开放域问答的密集段落检索。《预印本》（ArXiv），编号：abs/2010.08191。

O. Khattab and M. Zaharia. 2020. Colbert: Efficient and effective passage search via contextualized late interaction over bert. SIGIR.

哈塔卜O.和扎哈里亚M.。2020年。ColBERT：通过基于BERT的上下文后期交互实现高效有效的段落搜索。《信息检索研究与发展会议》（SIGIR）。

Diederik P. Kingma and Jimmy Ba. 2015. Adam: A method for stochastic optimization. CoRR, abs/1412.6980.

迪德里克·P·金马和吉米·巴。2015年。Adam：一种随机优化方法。《计算机研究报告》（CoRR），编号：abs/1412.6980。

Patrick Lewis, Barlas Oğuz, Wenhan Xiong, Fabio Petroni, Wen tau Yih, and Sebastian Riedel. 2021. Boosted dense retriever.

帕特里克·刘易斯、奥古兹B.、熊文瀚、法比奥·佩特罗尼、易文涛和塞巴斯蒂安·里德尔。2021年。增强型密集检索器。

Canjia Li, A. Yates, Sean MacAvaney, B. He, and Yingfei Sun. 2020. Parade: Passage representation aggregation for document reranking. ArXiv, abs/2008.09093.

李灿佳、耶茨A.、肖恩·麦卡瓦尼、何B.和孙英飞。2020年。PARADE：用于文档重排序的段落表示聚合。《预印本》（ArXiv），编号：abs/2008.09093。

Jimmy Lin, Rodrigo Nogueira, and A. Yates. 2020. Pre-trained transformers for text ranking: Bert and beyond. ArXiv, abs/2010.06467.

林吉米、罗德里戈·诺盖拉和耶茨A.。2020年。用于文本排序的预训练Transformer：BERT及其他。《预印本》（ArXiv），编号：abs/2010.06467。

Jimmy J. Lin and Xueguang Ma. 2021. A few brief notes on deepimpact, coil, and a conceptual framework for information retrieval techniques. ArXiv, abs/2106.14807.

吉米·J·林（Jimmy J. Lin）和马学光（Xueguang Ma）。2021年。关于深度影响（deepimpact）、线圈（coil）以及信息检索技术概念框架的几点简要说明。预印本平台（ArXiv），论文编号：abs/2106.14807。

Sheng-Chieh Lin, Jheng-Hong Yang, and Jimmy J. Lin. 2021. In-batch negatives for knowledge distillation with tightly-coupled teachers for dense retrieval. In REPL4NLP.

林圣杰（Sheng-Chieh Lin）、杨政宏（Jheng-Hong Yang）和吉米·J·林（Jimmy J. Lin）。2021年。使用紧密耦合教师进行知识蒸馏以实现密集检索的批内负样本方法。发表于自然语言处理表示学习研讨会（REPL4NLP）。

Sean MacAvaney, F. Nardini, R. Perego, N. Tonellotto, Nazli Goharian, and O. Frieder. 2020. Efficient document re-ranking for transformers by precomputing term representations. SIGIR.

肖恩·麦卡瓦尼（Sean MacAvaney）、F·纳尔迪尼（F. Nardini）、R·佩雷戈（R. Perego）、N·托内洛托（N. Tonellotto）、纳兹利·戈哈里安（Nazli Goharian）和O·弗里德（O. Frieder）。2020年。通过预计算词项表示实现变压器模型的高效文档重排序。发表于国际信息检索研究与发展会议（SIGIR）。

Sean MacAvaney, Andrew Yates, Arman Cohan, and Nazli Goharian. 2019. Cedr: Contextualized embed-dings for document ranking. SIGIR.

肖恩·麦卡瓦尼（Sean MacAvaney）、安德鲁·耶茨（Andrew Yates）、阿曼·科汉（Arman Cohan）和纳兹利·戈哈里安（Nazli Goharian）。2019年。Cedr：用于文档排名的上下文嵌入。发表于国际信息检索研究与发展会议（SIGIR）。

Chris J. Maddison, Andriy Mnih, and Yee Whye Teh. 2017. The concrete distribution: A continuous relaxation of discrete random variables. ICLR.

克里斯·J·麦迪逊（Chris J. Maddison）、安德里·米尼（Andriy Mnih）和叶怀· Teh（Yee Whye Teh）。2017年。具体分布：离散随机变量的连续松弛。发表于国际学习表征会议（ICLR）。

Antonio Mallia, O. Khattab, Nicola Tonellotto, and Torsten Suel. 2021. Learning passage impacts for inverted indexes. SIGIR.

安东尼奥·马利亚（Antonio Mallia）、O·哈塔布（O. Khattab）、尼古拉·托内洛托（Nicola Tonellotto）和托尔斯滕·苏尔（Torsten Suel）。2021年。为倒排索引学习段落影响。发表于国际信息检索研究与发展会议（SIGIR）。

Julieta Martinez, Shobhit Zakhmi, Holger H. Hoos, and J. Little. 2018. Lsq++: Lower running time and higher recall in multi-codebook quantization. In ${ECCV}$ .

朱丽叶塔·马丁内斯（Julieta Martinez）、肖比特·扎赫米（Shobhit Zakhmi）、霍尔格·H·胡斯（Holger H. Hoos）和J·利特尔（J. Little）。2018年。Lsq++：多码本量化中更低的运行时间和更高的召回率。发表于${ECCV}$。

Bhaskar Mitra, Sebastian Hofstätter, Hamed Zamani, and Nick Craswell. 2021. Conformer-kernel with query term independence for document retrieval. ${SI}$ - ${GIR}$ .

巴斯卡尔·米特拉（Bhaskar Mitra）、塞巴斯蒂安·霍夫施塔特（Sebastian Hofstätter）、哈米德·扎马尼（Hamed Zamani）和尼克·克拉斯韦尔（Nick Craswell）。2021年。具有查询词独立性的文档检索的卷积核方法。发表于${SI}$ - ${GIR}$。

Rodrigo Nogueira and Kyunghyun Cho. 2019. Passage re-ranking with bert. ArXiv, abs/1901.04085.

罗德里戈·诺盖拉（Rodrigo Nogueira）和赵京焕（Kyunghyun Cho）。2019年。使用BERT进行段落重排序。预印本平台（ArXiv），论文编号：abs/1901.04085。

Rodrigo Nogueira, W. Yang, Kyunghyun Cho, and Jimmy Lin. 2019a. Multi-stage document ranking with bert. ArXiv, abs/1910.14424.

罗德里戈·诺盖拉（Rodrigo Nogueira）、W·杨（W. Yang）、赵京焕（Kyunghyun Cho）和吉米·林（Jimmy Lin）。2019a。使用BERT进行多阶段文档排名。预印本平台（ArXiv），论文编号：abs/1910.14424。

Rodrigo Nogueira, Wei Yang, Jimmy J. Lin, and Kyunghyun Cho. 2019b. Document expansion by query prediction. ArXiv, abs/1904.08375.

罗德里戈·诺盖拉（Rodrigo Nogueira）、杨威（Wei Yang）、吉米·J·林（Jimmy J. Lin）和赵京焕（Kyunghyun Cho）。2019b。通过查询预测进行文档扩展。预印本平台（ArXiv），论文编号：abs/1904.08375。

Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks.

尼尔斯·赖默斯（Nils Reimers）和伊琳娜·古列维奇（Iryna Gurevych）。2019年。句子BERT：使用孪生BERT网络的句子嵌入。

In ${EMNLP}/{IJCNLP}$ .

发表于${EMNLP}/{IJCNLP}$。

Ruiyang Ren, Yingqi Qu, Jing Liu, Wayne Xin Zhao, QiaoQiao She, Hua Wu, Haifeng Wang, and Ji-Rong Wen. 2021. RocketQAv2: A joint training method for dense passage retrieval and passage re-ranking. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 2825-2835, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

任瑞阳（Ruiyang Ren）、曲英琦（Yingqi Qu）、刘静（Jing Liu）、赵鑫（Wayne Xin Zhao）、佘巧巧（QiaoQiao She）、吴华（Hua Wu）、王海峰（Haifeng Wang）和文继荣（Ji-Rong Wen）。2021年。RocketQAv2：一种用于密集段落检索和段落重排序的联合训练方法。发表于2021年自然语言处理经验方法会议论文集，第2825 - 2835页，线上会议和多米尼加共和国蓬塔卡纳。计算语言学协会。

Stephen E. Robertson and Hugo Zaragoza. 2009. The probabilistic relevance framework: $\mathrm{{Bm}}{25}$ and beyond. Found. Trends Inf. Retr., 3:333-389.

斯蒂芬·E·罗伯逊（Stephen E. Robertson）和雨果·萨拉戈萨（Hugo Zaragoza）。2009年。概率相关性框架：$\mathrm{{Bm}}{25}$及超越。信息检索趋势与基础，第3卷：333 - 389页。

Keshav Santhanam, O. Khattab, Jon Saad-Falcon, Christopher Potts, and Matei A. Zaharia. 2021. Colbertv2: Effective and efficient retrieval via lightweight late interaction. ArXiv, abs/2112.01488.

凯沙夫·桑塔南（Keshav Santhanam）、O·哈塔布（O. Khattab）、乔恩·萨德 - 法尔孔（Jon Saad-Falcon）、克里斯托弗·波茨（Christopher Potts）和马泰·A·扎哈里亚（Matei A. Zaharia）。2021年。Colbertv2：通过轻量级后期交互实现高效有效的检索。预印本平台（ArXiv），论文编号：abs/2112.01488。

Raphael Shu and Hideki Nakayama. 2018. Compressing word embeddings via deep compositional code learning. ICLR.

拉斐尔·舒（Raphael Shu）和中山英树（Hideki Nakayama）。2018年。通过深度组合代码学习压缩词嵌入。国际学习表征会议（ICLR）。

Y. Wu, M. Schuster, Z. Chen, Q. Le, M. Norouzi, W. Macherey, M. Krikun, Y. Cao, Q. Gao, K. Macherey, J. Klingner, A. Shah, M. Johnson, X. Liu, L. Kaiser, S. Gouws, Y. Kato, T. Kudo, H. Kazawa, K. Stevens, G. Kurian, N. Patil, W. Wang, C. Young, J. Smith, J. Riesa, A. Rudnick, O. Vinyals, G. Corrado, M. Hughes, and J. Dean. 2016. Google's neural machine translation system: Bridging the gap between human and machine translation. ArXiv, abs/1609.08144.

吴宇（Y. Wu）、米夏埃尔·舒斯特（M. Schuster）、陈泽（Z. Chen）、乐启昆（Q. Le）、莫哈迈德·诺鲁兹（M. Norouzi）、威廉·马赫雷（W. Macherey）、米哈伊尔·克里昆（M. Krikun）、曹宇（Y. Cao）、高奇（Q. Gao）、凯瑟琳·马赫雷（K. Macherey）、约书亚·克林纳（J. Klingner）、阿南德·沙阿（A. Shah）、迈克尔·约翰逊（M. Johnson）、刘晓（X. Liu）、卢卡斯·凯泽（L. Kaiser）、斯蒂芬·古兹（S. Gouws）、加藤洋（Y. Kato）、工藤拓（T. Kudo）、风间浩（H. Kazawa）、凯文·史蒂文斯（K. Stevens）、乔治·库里安（G. Kurian）、尼廷·帕蒂尔（N. Patil）、王巍（W. Wang）、杨晨（C. Young）、乔纳森·史密斯（J. Smith）、约书亚·里萨（J. Riesa）、亚历克斯·鲁德尼克（A. Rudnick）、奥里奥尔·温亚尔斯（O. Vinyals）、格雷格·科拉多（G. Corrado）、马修·休斯（M. Hughes）和约书亚·迪恩（J. Dean）。2016年。谷歌的神经机器翻译系统：缩小人类和机器翻译之间的差距。预印本平台（ArXiv），论文编号：abs/1609.08144。

J. Xin, Rodrigo Nogueira, Y. Yu, and Jimmy Lin. 2020. Early exiting bert for efficient document ranking. In SUSTAINLP.

辛杰（J. Xin）、罗德里戈·诺盖拉（Rodrigo Nogueira）、余洋（Y. Yu）和林吉米（Jimmy Lin）。2020年。用于高效文档排名的早期退出BERT模型。可持续自然语言处理研讨会（SUSTAINLP）。

Chenyan Xiong, Zhuyun Dai, J. Callan, Zhiyuan Liu, and R. Power. 2017. End-to-end neural ad-hoc ranking with kernel pooling. SIGIR.

熊晨彦（Chenyan Xiong）、戴珠云（Zhuyun Dai）、约翰·卡兰（J. Callan）、刘志远（Zhiyuan Liu）和罗伯特·鲍尔（R. Power）。2017年。基于核池化的端到端神经临时排序。信息检索研究与发展会议（SIGIR）。

Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N. Bennett, Junaid Ahmed, and Arnold Overwijk. 2021. Approximate nearest neighbor negative contrastive learning for dense text retrieval. In International Conference on Learning Representations.

熊磊（Lee Xiong）、熊晨彦（Chenyan Xiong）、李烨（Ye Li）、邓国峰（Kwok-Fung Tang）、刘佳琳（Jialin Liu）、保罗·N·贝内特（Paul N. Bennett）、朱奈德·艾哈迈德（Junaid Ahmed）和阿诺德·奥弗维克（Arnold Overwijk）。2021年。用于密集文本检索的近似最近邻负对比学习。国际学习表征会议。

Wei Yang, Haotian Zhang, and Jimmy Lin. 2019. Simple applications of bert for ad hoc document retrieval. ArXiv, abs/1903.10972.

杨威（Wei Yang）、张昊天（Haotian Zhang）和林吉米（Jimmy Lin）。2019年。BERT在临时文档检索中的简单应用。预印本平台（ArXiv），论文编号：abs/1903.10972。

Yingrui Yang, Yifan Qiao, Jinjin Shao, Xifeng Yan, and Tao Yang. 2022. Lightweight composite re-ranking for efficient keyword search with BERT. WSDM.

杨英锐（Yingrui Yang）、乔一帆（Yifan Qiao）、邵金金（Jinjin Shao）、闫夕峰（Xifeng Yan）和杨涛（Tao Yang）。2022年。基于BERT的高效关键词搜索轻量级复合重排序。网络搜索与数据挖掘会议（WSDM）。

Jingtao Zhan, J. Mao, Yiqun Liu, Min Zhang, and Shaoping Ma. 2020. Learning to retrieve: How to train a dense retrieval model effectively and efficiently. ArXiv, abs/2010.10469.

詹景涛（Jingtao Zhan）、毛杰（J. Mao）、刘一群（Yiqun Liu）、张敏（Min Zhang）和马少平（Shaoping Ma）。2020年。学习检索：如何有效且高效地训练密集检索模型。预印本平台（ArXiv），论文编号：abs/2010.10469。

Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Jiafeng Guo, Min Zhang, and Shaoping Ma. 2021a. Jointly optimizing query encoder and product quantization to improve retrieval performance. CIKM.

詹景涛（Jingtao Zhan）、毛佳欣（Jiaxin Mao）、刘一群（Yiqun Liu）、郭佳峰（Jiafeng Guo）、张敏（Min Zhang）和马少平（Shaoping Ma）。2021a。联合优化查询编码器和乘积量化以提高检索性能。信息与知识管理会议（CIKM）。

Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Jiafeng Guo, Min Zhang, and Shaoping Ma. 2021b. Optimizing dense retrieval model training with hard negatives. CoRR, abs/2104.08051.

詹景涛（Jingtao Zhan）、毛佳欣（Jiaxin Mao）、刘一群（Yiqun Liu）、郭佳峰（Jiafeng Guo）、张敏（Min Zhang）和马少平（Shaoping Ma）。2021b。利用难负样本优化密集检索模型训练。计算机研究与评论（CoRR），论文编号：abs/2104.08051。

Y. Zhang, Ping Nie, Xiubo Geng, A. Ramamurthy, L. Song, and Daxin Jiang. 2020. Dc-bert: Decoupling question and document for efficient contextual encoding. In ${SIGIR}$ .

张宇（Y. Zhang）、聂平（Ping Nie）、耿秀波（Xiubo Geng）、阿南德·拉马穆尔蒂（A. Ramamurthy）、宋乐（L. Song）和蒋大新（Daxin Jiang）。2020年。DC - BERT：解耦问题和文档以实现高效上下文编码。发表于${SIGIR}$。

Shengyao Zhuang and G. Zuccon. 2021. Fast passage re-ranking with contextualized exact term matching and efficient passage expansion. ArXiv, abs/2108.08513.

庄圣耀（Shengyao Zhuang）和贾科莫·祖科恩（G. Zuccon）。2021年。基于上下文精确术语匹配和高效段落扩展的快速段落重排序。预印本平台（ArXiv），论文编号：abs/2108.08513。

## A Details on Retrieval Choices, Numbers Cited, and Model Implementations

## A 检索选择、引用数据和模型实现的详细信息

First-stage retrieval models considered. To retrieve top results before re-ranking, we have considered the recent work in sparse and dense retrieval that outperforms BM25. For sparse retrieval with inverted indices, DeepCT (Dai and Callan, 2020) uses deep learning to assign more sophisticated term weights for soft matching. The docT5query work (Nogueira et al., 2019b) uses a neural model to pre-process and expand documents. The recent work on sparse representations includes DeepIm-pact (Mallia et al., 2021), uniCOIL (Lin and Ma, 2021; Gao et al., 2021), and SPLADE (Formal et al., 2021b,a), for learning neural contextualized term weights with document expansion. Instead of using a sparse inverted index, an alternative retrieval method is to use a dense representation of each document, e.g. (Lin et al., 2021; Zhan et al., 2021a; Xiong et al., 2021; Gao and Callan, 2021; Zhan et al., 2021b; Ren et al., 2021). We use BM25 because it is a standard reference point. We have also used uniCOIL for passage re-ranking because a uniCOIL-based sparse retriever is fairly efficient and its tested relevance result is comparable to that of the end-to-end ColBERT as a dense retriever and other learned sparse representations mentioned above. Certainly $\mathrm{{CQ}}$ is applicable for re-ranking with any of dense or sparse retrievers or their hybrid combination.

考虑的第一阶段检索模型。为了在重排序之前检索到顶级结果，我们考虑了在稀疏和密集检索方面的最新研究，这些研究的性能优于BM25（二元独立模型的改进版）。对于使用倒排索引的稀疏检索，DeepCT（戴和卡兰，2020年）使用深度学习为软匹配分配更复杂的词项权重。docT5query研究（诺盖拉等人，2019b）使用神经模型对文档进行预处理和扩展。最近关于稀疏表示的研究包括DeepIm-pact（马利亚等人，2021年）、uniCOIL（林和马，2021年；高等人，2021年）和SPLADE（福尔马尔等人，2021b、a），用于通过文档扩展学习神经上下文词项权重。除了使用稀疏倒排索引，另一种检索方法是使用每个文档的密集表示，例如（林等人，2021年；詹等人，2021a；熊等人，2021年；高和卡兰，2021年；詹等人，2021b；任等人，2021年）。我们使用BM25是因为它是一个标准参考点。我们还使用uniCOIL进行段落重排序，因为基于uniCOIL的稀疏检索器相当高效，并且其测试的相关性结果与作为密集检索器的端到端ColBERT以及上述其他学习到的稀疏表示的结果相当。当然，$\mathrm{{CQ}}$适用于使用任何密集或稀疏检索器或它们的混合组合进行重排序。

Model numbers cited from other papers. As marked in Tables 3 and 4, for DeepCT, JPQ and TILDEv2, we copy the relevance numbers reported in their papers. For TCT-ColBERT(v2), DeepIm-pact and uniCOIL, we obtain their performance using the released checkpoints of Pyserini ${}^{1}$ . For PreTTR (MacAvaney et al., 2020) on the passage task and BERT-base on the document task, we cite the relevance performance reported in Hofstätter et al. (2020a). There are two reasons to list the relevance numbers from other papers. One reason is that for some chosen algorithms, the running of our implementation version or their code delivers a performance lower than what has been reported in the authors' original papers, perhaps due to the difference in training setup. Thus, we think it is fairer to report the results from the authors' papers. Another reason is that for some algorithms, the authors did not release code and we do not have implementations.

从其他论文中引用的模型指标。如表3和表4所示，对于DeepCT、JPQ和TILDEv2，我们复制了它们论文中报告的相关性指标。对于TCT-ColBERT(v2)、DeepIm-pact和uniCOIL，我们使用Pyserini ${}^{1}$ 发布的检查点来获取它们的性能。对于段落任务中的PreTTR（麦卡瓦尼等人，2020年）和文档任务中的BERT-base，我们引用了霍夫施泰特等人（2020a）报告的相关性性能。列出其他论文中的相关性指标有两个原因。一个原因是，对于一些选定的算法，我们实现版本的运行或它们的代码所产生的性能低于作者原始论文中报告的性能，这可能是由于训练设置的差异。因此，我们认为报告作者论文中的结果更公平。另一个原因是，对于一些算法，作者没有发布代码，我们也没有实现。

In storage space estimation of Table 5, for BECR, we use the default 128 bit LSH footprint with 5 layers. For PreTTR we uses 3 layers with dimension 768 and two bytes per number following Hofstätter et al. (2020a). For TILDEv2, we directly cite the space cost from its paper.

在表5的存储空间估计中，对于BECR，我们使用默认的128位局部敏感哈希（LSH）占用空间，有5层。对于PreTTR，我们按照霍夫施泰特等人（2020a）的方法，使用3层，维度为768，每个数字用两个字节表示。对于TILDEv2，我们直接引用了其论文中的空间成本。

Model implementation and training. For baseline model parameters, we use the recommended set of parameters from the authors' original papers. For ColBERT, we use the default version that the authors selected for fair comparison. The ColBERT code follows the original version released ${}^{2}$ and BERT implementation is from Huggingface ${}^{3}$ . For BERT-base and ColBERT, training uses pairwise softmax cross-entropy loss over the released or derived triples in a form of $\left( {\mathbf{q},{\mathbf{d}}^{ + },{\mathbf{d}}^{ - }}\right)$ for the MS MARCO passage task. For the MS MARCO document re-ranking task, we split each positive long document into segments with 400 tokens each and transfer the positive label of such a document to each divided segment. The negative samples are obtained using the BM25 top 100 negative documents. The above way we select training triples for document re-ranking may be less ideal and can deserve an improvement in the future.

模型实现和训练。对于基线模型参数，我们使用作者原始论文中推荐的参数集。对于ColBERT，我们使用作者为公平比较而选择的默认版本。ColBERT代码遵循发布的原始版本 ${}^{2}$ ，BERT实现来自Huggingface ${}^{3}$ 。对于BERT-base和ColBERT，在MS MARCO段落任务中，训练使用基于发布或派生的三元组的成对softmax交叉熵损失，形式为 $\left( {\mathbf{q},{\mathbf{d}}^{ + },{\mathbf{d}}^{ - }}\right)$ 。对于MS MARCO文档重排序任务，我们将每个正的长文档分割成每个包含400个词元的段落，并将该文档的正标签转移到每个分割的段落上。负样本通过BM25排名前100的负文档获得。我们为文档重排序选择训练三元组的上述方式可能不太理想，未来值得改进。

When training ColBERT, we use gradient accumulation and perform batch propagation every 32 training triplets. All models are trained using Adam optimizer (Kingma and Ba, 2015). The learning rate is $3\mathrm{e} - 6$ for ColBERT and $2\mathrm{e} - 5$ for BERT-base following the setup in its original paper. For ColBERT on the document dataset, we obtained the model checkpoint from the authors.

训练ColBERT时，我们使用梯度累积，每32个训练三元组进行一次批量传播。所有模型都使用Adam优化器（金马和巴，2015年）进行训练。按照原始论文的设置，ColBERT的学习率为 $3\mathrm{e} - 6$ ，BERT-base的学习率为 $2\mathrm{e} - 5$ 。对于文档数据集上的ColBERT，我们从作者那里获得了模型检查点。

Our CQ implementation leverages the open source code ${}^{4}$ for Shu and Nakayama (2018). For PQ, OPQ, RQ, and LSQ, we uses off-the-shelf implementation from Facebook’s faiss ${}^{5}$ library (Johnson et al., 2017). To get training instances for each quantizer, we generate the contextual embed-dings of randomly-selected 500,000 tokens from passages or documents using ColBERT.

我们的复合量化（CQ，Compositional Quantization）实现借鉴了Shu和Nakayama（2018）的开源代码${}^{4}$。对于乘积量化（PQ，Product Quantization）、优化乘积量化（OPQ，Optimized Product Quantization）、残差量化（RQ，Residual Quantization）和最小平方量化（LSQ，Least Squares Quantization），我们使用了Facebook的Faiss${}^{5}$库（Johnson等人，2017）中的现成实现。为了为每个量化器获取训练实例，我们使用ColBERT从段落或文档中随机选择500,000个标记生成上下文嵌入。

When using the MSE loss, learning rate is 0.0001 , batch size is 128 , and the number of training epochs is200,000. When fine-tuning with PairwiseCE or MarginMSE, we freeze the encoder based on the MSE loss, set the learning rate to be $3\mathrm{e} - 6$ ,and then train for additional 800 batch iterations with 32 training pairs per batch.

使用均方误差（MSE，Mean Squared Error）损失时，学习率为0.0001，批量大小为128，训练轮数为200,000。当使用成对交叉熵（PairwiseCE，Pairwise Cross-Entropy）或边际均方误差（MarginMSE，Margin Mean Squared Error）进行微调时，我们基于均方误差损失冻结编码器，将学习率设置为$3\mathrm{e} - 6$，然后以每批32个训练对进行额外的800个批量迭代训练。

---

<!-- Footnote -->

${}^{2}$ https://github.com/stanford-futuredata/ColBERT

${}^{2}$ https://github.com/stanford-futuredata/ColBERT

${}^{3}$ https://huggingface.co/transformers/model_doc/bert.html

${}^{3}$ https://huggingface.co/transformers/model_doc/bert.html

${}^{4}$ github.com/mingu600/compositional_code_learning.git

${}^{4}$ github.com/mingu600/compositional_code_learning.git

${}^{5}$ https://github.com/facebookresearch/faiss

${}^{5}$ https://github.com/facebookresearch/faiss

${}^{1}$ https://github.com/castorini/pyserini/

${}^{1}$ https://github.com/castorini/pyserini/

<!-- Footnote -->

---