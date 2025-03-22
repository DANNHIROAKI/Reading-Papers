# Multi-View Document Representation Learning for Open-Domain Dense Retrieval

# 用于开放域密集检索的多视图文档表示学习

Shunyu Zhang ${}^{1, * }$ Yaobo Liang ${}^{2}$ , Ming Gong ${}^{3}$ , Daxin Jiang ${}^{3}$ , Nan Duan ${}^{2}$

张顺宇 ${}^{1, * }$ 梁耀波 ${}^{2}$ ，龚明 ${}^{3}$ ，蒋大新 ${}^{3}$ ，段楠 ${}^{2}$

${}^{1}$ Intelligent Computing and Machine Learning Lab,School of ASEE,Beihang University

${}^{1}$ 北京航空航天大学航空科学与工程学院智能计算与机器学习实验室

${}^{2}$ Microsoft Research Asia ${}^{3}$ Microsoft STC Asia

${}^{2}$ 微软亚洲研究院 ${}^{3}$ 微软亚洲互联网工程院

1 zhangshunyu@buaa.edu.cn

1 zhangshunyu@buaa.edu.cn

${}^{2,3}$ \{yalia, migon, djiang, nanduan\}@microsoft.com

${}^{2,3}$ \{yalia, migon, djiang, nanduan\}@microsoft.com

## Abstract

## 摘要

Dense retrieval has achieved impressive advances in first-stage retrieval from a large-scale document collection, which is built on bi-encoder architecture to produce single vector representation of query and document. However, a document can usually answer multiple potential queries from different views. So the single vector representation of a document is hard to match with multi-view queries, and faces a semantic mismatch problem. This paper proposes a multi-view document representation learning framework, aiming to produce multiview embeddings to represent documents and enforce them to align with different queries. First, we propose a simple yet effective method of generating multiple embeddings through viewers. Second, to prevent multi-view embed-dings from collapsing to the same one, we further propose a global-local loss with annealed temperature to encourage the multiple viewers to better align with different potential queries. Experiments show our method outperforms recent works and achieves state-of-the-art results.

密集检索在从大规模文档集合中进行第一阶段检索方面取得了显著进展，它基于双编码器架构为查询和文档生成单向量表示。然而，一个文档通常可以从不同视角回答多个潜在查询。因此，文档的单向量表示难以与多视图查询匹配，面临语义不匹配问题。本文提出了一个多视图文档表示学习框架，旨在生成多视图嵌入来表示文档，并促使它们与不同查询对齐。首先，我们提出了一种通过查看器生成多个嵌入的简单而有效的方法。其次，为防止多视图嵌入收敛为相同的嵌入，我们进一步提出了一种带有退火温度的全局 - 局部损失，以鼓励多个查看器更好地与不同潜在查询对齐。实验表明，我们的方法优于近期工作，并取得了最先进的结果。

## 1 Introduction

## 1 引言

Over the past few years, with the advancements in pre-trained language models (Devlin et al., 2019; Liu et al., 2019), dense retrieval has become an important and effective approach in open-domain text retrieval (Karpukhin et al., 2020; Lee et al., 2019; Qu et al., 2021; Xiong et al., 2020). A typical dense retriever usually adopts a bi-encoder (Huang et al., 2013; Reimers and Gurevych, 2019) architecture to encode input query and document into a single low-dimensional vector (usually CLS token), and computes the relevance scores between their representations. In real-world applications, the embedding vectors of all the documents are pre -computed in advance, and the retrieval process can be efficiently boosted by the approximate nearest neighbor (ANN) technique (Johnson et al., 2019). To enhance bi-encoder's capacity, recent studies carefully design sophisticated methods to train it effectively, including constructing more challenging hard negatives (Zhan et al., 2021; Xiong et al., 2020; Qu et al., 2021) and continually pre-train the 5990 language models (Gao and Callan, 2021a; Oğuz et al., 2021) for a better transfer.

在过去几年中，随着预训练语言模型的发展（Devlin 等人，2019；Liu 等人，2019），密集检索已成为开放域文本检索中的一种重要且有效的方法（Karpukhin 等人，2020；Lee 等人，2019；Qu 等人，2021；Xiong 等人，2020）。典型的密集检索器通常采用双编码器（Huang 等人，2013；Reimers 和 Gurevych，2019）架构，将输入的查询和文档编码为单个低维向量（通常是 CLS 标记），并计算它们表示之间的相关性得分。在实际应用中，所有文档的嵌入向量会预先计算，并且检索过程可以通过近似最近邻（ANN）技术（Johnson 等人，2019）得到有效加速。为了增强双编码器的能力，近期研究精心设计了复杂的方法来有效训练它，包括构建更具挑战性的难负样本（Zhan 等人，2021；Xiong 等人，2020；Qu 等人，2021）以及持续预训练语言模型（Gao 和 Callan，2021a；Oğuz 等人，2021）以实现更好的迁移。

<!-- Media -->

<!-- figureText: Title: IPod video and music libraries on individual seat-back displays. Originally KLM A3: MP3, AAC/M4A, Protected AAC, AIFF, WAV, Audible audiobook, and Document: Beginning in mid-2007, four major airlines, United, Continental, Delta, and Emirates, reached agreements to install iPod seat connections. The free service will allow passengers to power and charge an iPod, and view and Air France were reported to be part of the deal with Apple, but they later released statements explaining that they were only contemplating the possibility of incorporating such systems. The iPod line can play several audio file formats including MP3, AAC/M4A, Protected AAC, AIFF, WAV, ability to display JPEG, BMP, GIF, TIFF, and PNG image file formats. Q1: Where can people using iPods on planes view the device's interface? A1: Individual seat-back displays. Q2: What are two airlines that considered implementing iPod connections but did not join the 2007 agreement? A2: KLM and Air France. Apple Lossless. Q4: What is the name of an audio format developed by Apple? (a) An example from SQuAD Open Dataset. Query (b) Our proposed MVR method. -->

<img src="https://cdn.noedgeai.com/0195aeeb-a9d7-7d11-ad76-1517c72ca387_0.jpg?x=862&y=676&w=580&h=836&r=0"/>

Figure 1: The illustration of our multi-view document representation learning framework. The triangles and circles mean document and query vectors separately. Usually, one document can be asked to different potential queries from multiple views. Our method comes from this observation and generates multi-view representations for documents to better align with different potential queries.

图 1：我们的多视图文档表示学习框架示意图。三角形和圆形分别表示文档和查询向量。通常，一个文档可以从多个视角被提出不同的潜在查询。我们的方法基于这一观察，为文档生成多视图表示，以更好地与不同潜在查询对齐。

<!-- Media -->

---

<!-- Footnote -->

* Work done during internship at Microsoft Research Asia.

* 本工作是在微软亚洲研究院实习期间完成的。

<!-- Footnote -->

---

However, being limited to the single vector representation, bi-encoder faces the upper boundary of representation capacity according to theoretical analysis in Luan et al. (2021). In the real example from SQuAD dev dataset, we also find that a single vector representation can't match well to multi-view queries, as shown in Figure.1. The document corresponds to four different questions from different views, and each of them matches to different sentences and answers. In the traditional bi-encoder, the document is represented to a single vector while it should be recalled by multiple diverse queries, which limits the capacity of the bi-encode.

然而，根据 Luan 等人（2021）的理论分析，由于受限于单向量表示，双编码器面临表示能力的上限。在来自 SQuAD 开发数据集的实际示例中，我们也发现单向量表示无法很好地匹配多视图查询，如图 1 所示。该文档对应来自不同视角的四个不同问题，每个问题都与不同的句子和答案匹配。在传统的双编码器中，文档被表示为单个向量，而它应该被多个不同的查询召回，这限制了双编码器的能力。

As for the multi-vector models, cross-encoder architectures perform better by computing deeply-contextualized representations of query-document pairs, but are computationally expensive and impractical for first-stage large-scale retrieval (Reimers and Gurevych, 2019; Humeau et al., 2020). Some recent studies try to borrow from cross-encoder and extend bi-encoder by employing more delicate structures, which allow the multiple vector representations and dense interaction between query and document embeddings. including late interaction (Khattab and Zaharia, 2020) and attention-based aggregator (Humeau et al., 2020; Tang et al., 2021). However, most of them contain softmax or sum operators that can't be decomposed into max over inner products, and so fast ANN retrieval can't be directly applied.

对于多向量模型，交叉编码器架构通过计算查询 - 文档对的深度上下文表示表现更好，但计算成本高，对于第一阶段的大规模检索不切实际（Reimers 和 Gurevych，2019；Humeau 等人，2020）。一些近期研究尝试借鉴交叉编码器并通过采用更精细的结构扩展双编码器，这些结构允许查询和文档嵌入之间进行多向量表示和密集交互，包括后期交互（Khattab 和 Zaharia，2020）和基于注意力的聚合器（Humeau 等人，2020；Tang 等人，2021）。然而，它们中的大多数包含无法分解为内积最大值的 softmax 或求和运算符，因此无法直接应用快速 ANN 检索。

Based on these observations, we propose MultiView document Representations learning framework, $\mathbf{{MVR}}$ in short. MVR originates from our observation that a document commonly has several semantic units, and can answer multiple potential queries which contain individual semantic content. It is just like given a specified document, different askers raise different questions from diverse views. Therefore, we propose a simple yet effective method to generate multi-view representations through viewers, optimized by a global-local loss with annealed temperature to improve the representation space.

基于这些观察，我们提出了多视图文档表示学习框架（MultiView document Representations learning framework，简称MVR）。MVR源于我们的观察，即一个文档通常有几个语义单元，并且可以回答多个包含单个语义内容的潜在查询。这就好比给定一个特定的文档，不同的提问者从不同的视角提出不同的问题。因此，我们提出了一种简单而有效的方法，通过视图器（viewer）生成多视图表示，并通过带有退火温度的全局 - 局部损失进行优化，以改善表示空间。

Prior work has found [CLS] token tends to aggregate the overall meaning of the whole input segment (Kovaleva et al., 2019; Clark et al., 2019), which is inconsistent with our goal of generating multi-view embeddings. So we first modify the bi-encoder architecture, abandon [CLS] token and add multiple [Viewer] tokens to the document input. The representation of the viewers in the last layer is then used as the multi-view representations.

先前的工作发现[CLS]标记倾向于聚合整个输入片段的整体含义（Kovaleva等人，2019；Clark等人，2019），这与我们生成多视图嵌入的目标不一致。因此，我们首先修改双编码器架构，放弃[CLS]标记，并在文档输入中添加多个[视图器]（Viewer）标记。然后，将最后一层中视图器的表示用作多视图表示。

To encourage the multiple viewers to better align with different potential queries, we propose a global-local loss equipped with an annealed temperature. In the previous work, the contrastive loss between positive and negative samples is widely applied (Karpukhin et al., 2020). Apart from global contrastive loss, we formulate a local uniformity loss between multi-view document embeddings, to better keep the uniformity among multiple viewers and prevent them from collapsing into the same one. In addition, we adopt an annealed temperature which gradually sharpens the distribution of viewers, to help multiple viewers better match to different potential queries, which is also validated in our experiment.

为了鼓励多个视图器更好地与不同的潜在查询对齐，我们提出了一种配备退火温度的全局 - 局部损失。在先前的工作中，正样本和负样本之间的对比损失被广泛应用（Karpukhin等人，2020）。除了全局对比损失，我们还制定了多视图文档嵌入之间的局部均匀性损失，以更好地保持多个视图器之间的均匀性，并防止它们坍缩为同一个。此外，我们采用了一种退火温度，它会逐渐锐化视图器的分布，以帮助多个视图器更好地匹配不同的潜在查询，这在我们的实验中也得到了验证。

The contributions of this paper are as follows:

本文的贡献如下：

- We propose a simple yet effective method to generate multi-view document representations through multiple viewers.

- 我们提出了一种简单而有效的方法，通过多个视图器生成多视图文档表示。

- To optimize the training of multiple viewers, we propose a global-local loss with annealed temperature to make multiple viewers to better align to different semantic views.

- 为了优化多个视图器的训练，我们提出了一种带有退火温度的全局 - 局部损失，以使多个视图器更好地与不同的语义视图对齐。

- Experimental results on open-domain retrieval datasets show that our approach achieves state-of-the-art retrieval performance. Further analysis proves the effectiveness of our method.

- 在开放域检索数据集上的实验结果表明，我们的方法实现了最先进的检索性能。进一步的分析证明了我们方法的有效性。

## 2 Background and Related Work

## 2 背景与相关工作

### 2.1 Retriever and Ranker Architecture

### 2.1 检索器和排序器架构

With the development of deep language model (Devlin et al., 2019), fine-tuned deep pre-trained BERT achieve advanced re-ranking performance (Dai and Callan, 2019; Nogueira and Cho, 2019). The initial approach is the cross-encoder based re-ranker, as shown in Figure.2(a). It feeds the concatenation of query and document text to BERT and outputs the [CLS] token's embeddings to produce a relevance score. Benefiting from deeply-contextualized representations of query-document pairs, the deep LM helps bridge both vocabulary mismatch and semantic mismatch. However, cross-encoder based rankers need computationally expensive cross-attention operations (Khattab and Zaharia, 2020; Gao and Callan, 2021a), so it is impractical for large-scale first-stage retrieval and is usually deployed in second-stage re-ranking.

随着深度语言模型的发展（Devlin等人，2019），经过微调的深度预训练BERT实现了先进的重排序性能（Dai和Callan，2019；Nogueira和Cho，2019）。最初的方法是基于交叉编码器的重排序器，如图2（a）所示。它将查询和文档文本的拼接输入到BERT中，并输出[CLS]标记的嵌入以生成相关性得分。得益于查询 - 文档对的深度上下文表示，深度语言模型有助于弥合词汇不匹配和语义不匹配的问题。然而，基于交叉编码器的排序器需要计算成本高昂的交叉注意力操作（Khattab和Zaharia，2020；Gao和Callan，2021a），因此对于大规模的第一阶段检索来说是不切实际的，通常部署在第二阶段的重排序中。

<!-- Media -->

<!-- figureText: Score Score Attention Pooler Query Encoder Doc Encoder Query Encoder Doc Encoder (c) Late Interaction (e.g., ColBERT) (d) Attention-based Aggregator (e.g., PolyEncoder) Score Cross-encoder Query Encoder Doc Encoder [CLS] (a) Cross-encoder(e.g., BERT) (b) Bi-encoder (e.g., DPR) -->

<img src="https://cdn.noedgeai.com/0195aeeb-a9d7-7d11-ad76-1517c72ca387_2.jpg?x=186&y=186&w=1275&h=289&r=0"/>

Figure 2: The comparison of different model architectures designed for retrieval/re-ranking.

图2：为检索/重排序设计的不同模型架构的比较。

<!-- Media -->

As for first-stage retrieval, bi-encoder is the most adopted architecture (Karpukhin et al., 2020) for it can be easily and efficiently employed with support from approximate nearest neighbor (ANN) (Johnson et al., 2019). As illustrated in Figure.2(b), it feeds the query and document to the individual encoders to generate single vector representations, and the relevance score is measured by the similarity of their embeddings. Equipped with deep LM, bi-encoder based retriever has achieved promising performance (Karpukhin et al., 2020). And later studies have further improved its performance through different carefully designed methods, which will be introduced in Sec.2.2

对于第一阶段检索，双编码器是最常用的架构（Karpukhin等人，2020），因为它可以在近似最近邻（ANN）的支持下轻松高效地使用（Johnson等人，2019）。如图2（b）所示，它将查询和文档输入到单独的编码器中以生成单向量表示，相关性得分通过它们嵌入的相似度来衡量。配备了深度语言模型，基于双编码器的检索器取得了有前景的性能（Karpukhin等人，2020）。后来的研究通过不同的精心设计的方法进一步提高了其性能，这将在2.2节中介绍。

Beside the typical bi-encoder, there are some variants(Gao et al., 2020; Chen et al., 2020; Mehri and Eric, 2021) proposing to employ dense interactions based on Bi-encoder. As shown in Fig.2(c), ColBERT (Khattab and Zaharia, 2020) adopts the late interaction paradigm, which computes token-wise dot scores between all the terms' vectors, sequentially followed by max-pooler and sum-pooler to produce a relevance score. Later on, Gao et al. (2021) improve it by scoring only on overlapping token vectors with inverted lists. Another variant is the attention-based aggregator, as shown in Fig.2(d). It utilizes the attention mechanism to compress the document embeddings to interact with the query vector for a final relevance score. There are several works (Humeau et al., 2020; Luan et al., 2021; Tang et al., 2021) built on this paradigm. Specifically, Poly-Encoder(learnt-k) (Humeau et al.,2020) sets $k$ learnable codes to attend them over the document embeddings. DRPQ (Tang et al., 2021) achieve better results by iterative K-means clustering on the document vectors to generate multiple embeddings, followed by attention-based interaction with query. However, the dense interaction methods can't be directly deployed with ANN, because both the sum-pooler and attention operator can't be decomposed into max over inner products, and the fast ANN search cannot be applied. So they usually first approximately recall a set of candidates then refine them by exhaustively re-ranking, While MVR can be directly applied in first-stage retrieval.

除了典型的双编码器（bi - encoder）之外，还有一些变体（高等人，2020年；陈等人，2020年；梅赫里和埃里克，2021年）提议在双编码器的基础上采用密集交互。如图2（c）所示，ColBERT（卡塔布和扎哈里亚，2020年）采用后期交互范式，该范式计算所有词项向量之间的逐词点积得分，随后依次通过最大池化器和求和池化器来生成相关性得分。后来，高等人（2021年）通过仅对倒排列表中的重叠词项向量进行评分来改进它。另一种变体是基于注意力的聚合器，如图2（d）所示。它利用注意力机制压缩文档嵌入，使其与查询向量交互以得到最终的相关性得分。有几项工作（休莫等人，2020年；栾等人，2021年；唐等人，2021年）基于此范式开展。具体而言，Poly - Encoder（learnt - k）（休莫等人，2020年）设置了$k$个可学习的代码，用于在文档嵌入上对它们进行关注。DRPQ（唐等人，2021年）通过对文档向量进行迭代K均值聚类以生成多个嵌入，然后与查询进行基于注意力的交互，取得了更好的结果。然而，密集交互方法不能直接与近似最近邻搜索（ANN）一起部署，因为求和池化器和注意力算子都不能分解为内积上的最大值，因此无法应用快速的ANN搜索。所以它们通常先大致召回一组候选对象，然后通过穷举重排序对其进行优化，而多视图表示（MVR）可以直接应用于第一阶段的检索。

Another related method is ME-BERT(Luan et al., 2021),which adopts the first $k$ document token embeddings as the document representation. However,only adopting the first-k embeddings may lose beneficial information in the latter part of the document, while our viewer tokens can extract from the whole document. In Sec.5.2, we also find the multiple embeddings in MEBERT will collapse to the same [CLS], while our global-local loss can address this problem.

另一种相关方法是ME - BERT（栾等人，2021年），它采用前$k$个文档词项嵌入作为文档表示。然而，仅采用前k个嵌入可能会丢失文档后半部分的有用信息，而我们的查看器词元可以从整个文档中提取信息。在5.2节中，我们还发现ME - BERT中的多个嵌入会坍缩为相同的[CLS]，而我们的全局 - 局部损失可以解决这个问题。

### 2.2 Effective Dense Retrieval

### 2.2 有效的密集检索

In addition to the aforementioned work focusing on the architecture design, there exist loads of work that proposes to improve the effectiveness of dense retrieval. Existing approaches of learning dense passage retriever can be divided into two categories: (1) pre-training for retrieval (Chang et al., 2020; Lee et al., 2019; Guu et al., 2020) and (2) fine-tuning pre-trained language models (PLMs) on labeled data (Karpukhin et al., 2020; Xiong et al., 2020; Qu et al., 2021).

除了上述专注于架构设计的工作之外，还有大量工作致力于提高密集检索的有效性。现有的学习密集段落检索器的方法可以分为两类：（1）为检索进行预训练（张等人，2020年；李等人，2019年；古等人，2020年）和（2）在标注数据上微调预训练语言模型（PLMs）（卡尔普欣等人，2020年；熊等人，2020年；曲等人，2021年）。

In the first category, Lee et al. (2019) and Chang et al. (2020) propose different pre-training task and demonstrate the effectiveness of pre-training in dense retrievers. Recently, DPR-PAQ (Oğuz et al., 2021) proposes domain matched pre-training, while Condenser (Gao and Callan, 2021a,b) enforces the model to produce an information-rich CLS representation with continual pre-training.

在第一类中，李等人（2019年）和张等人（2020年）提出了不同的预训练任务，并证明了预训练在密集检索器中的有效性。最近，DPR - PAQ（奥古兹等人，2021年）提出了领域匹配预训练，而Condenser（高和卡兰，2021a，b）通过持续预训练强制模型生成信息丰富的CLS表示。

As for the second class, recent work (Karpukhin et al., 2020; Xiong et al., 2020; Qu et al., 2021; Zhan et al., 2021) shows the key of fine-tuning an effective dense retriever revolves around hard negatives. DPR (Karpukhin et al., 2020) adopts in-batch negatives and BM25 hard negatives. ANCE (Xiong et al., 2020) proposes to construct hard negatives dynamically during training. RocketQA (Qu et al., 2021; Ren et al., 2021b) shows the cross-encoder can filter and mine higher-quality hard negatives. Li et al. (2021) and Ren et al. (2021a) demonstrate that passage-centric and query-centric negatives can make the training more robust. It is worth mentioning that distilling the knowledge from cross-encoder-based re-ranker into bi-encoder-based retriever (Sachan et al., 2021; Izacard and Grave, 2021; Ren et al., 2021a,b; Zhang et al., 2021) can improve the bi-encoder's performance. Most of these works are built upon bi-encoder and naturally inherit its limit of a single vector representation, while our work modified the bi-encoder to produce multi-view embeddings, and is also orthogonal to these strategies.

至于第二类，近期的工作（卡尔普欣等人，2020年；熊等人，2020年；曲等人，2021年；詹等人，2021年）表明，微调一个有效的密集检索器的关键在于困难负样本。DPR（卡尔普欣等人，2020年）采用批内负样本和BM25困难负样本。ANCE（熊等人，2020年）提议在训练期间动态构建困难负样本。RocketQA（曲等人，2021年；任等人，2021b）表明交叉编码器可以过滤和挖掘更高质量的困难负样本。李等人（2021年）和任等人（2021a）证明了以段落为中心和以查询为中心的负样本可以使训练更加稳健。值得一提的是，将基于交叉编码器的重排序器的知识蒸馏到基于双编码器的检索器中（萨坎等人，2021年；伊扎卡德和格雷夫，2021年；任等人，2021a，b；张等人，2021年）可以提高双编码器的性能。这些工作大多基于双编码器，自然地继承了其单一向量表示的局限性，而我们的工作对双编码器进行了修改以生成多视图嵌入，并且与这些策略是正交的。

## 3 Methodology

## 3 方法

### 3.1 Preliminary

### 3.1 预备知识

We start with a bi-encoder using BERT as its backbone neural network, as shown in Figure 2(b). A typical Bi-encoder adopts dual encoder architecture which maps the query and document to single dimensional real-valued vectors separately.

我们首先使用以BERT为骨干神经网络的双编码器，如图2(b)所示。典型的双编码器采用双编码器架构，该架构分别将查询和文档映射到一维实值向量。

Given a query $q$ and a document collection $D = \left\{  {{d}_{1},\ldots ,{d}_{j},\ldots ,{d}_{n}}\right\}$ ,dense retrievers leverage the same BERT encoder to get the representations of queries and documents. Then the similarity score $f\left( {q,d}\right)$ of query $q$ and document $d$ can be calculated with their dense representations:

给定一个查询$q$和一个文档集合$D = \left\{  {{d}_{1},\ldots ,{d}_{j},\ldots ,{d}_{n}}\right\}$，密集检索器利用相同的BERT编码器来获取查询和文档的表示。然后，可以使用它们的密集表示来计算查询$q$和文档$d$的相似度得分$f\left( {q,d}\right)$：

$$
f\left( {q,d}\right)  = \operatorname{sim}\left( {{E}_{Q}\left( q\right) ,{E}_{D}\left( d\right) }\right)  \tag{1}
$$

Where $\operatorname{sim}\left( \cdot \right)$ is the similarity function to estimate the relevance between two embeddings, e.g., cosine distance, euclidean distance, etc. And the inner-product on the [CLS] representations is a widely adopted setting (Karpukhin et al., 2020; Xiong et al., 2020).

其中$\operatorname{sim}\left( \cdot \right)$是用于估计两个嵌入之间相关性的相似度函数，例如余弦距离、欧几里得距离等。并且在[CLS]表示上进行内积是一种广泛采用的设置（卡尔普欣等人，2020年；熊等人，2020年）。

A conventional contrastive-learning loss is widely applied for training query and passage encoders supervised by the target task's training set. For a given query $q$ ,it computed negative log likelihood of a positive document ${d}^{ + }$ against a set of negatives $\left\{  {{d}_{1}^{ - },{d}_{2}^{ - },..{d}_{l}^{ - }}\right\}$ .

传统的对比学习损失被广泛应用于在目标任务的训练集监督下训练查询和段落编码器。对于给定的查询$q$，它计算正文档${d}^{ + }$相对于一组负文档$\left\{  {{d}_{1}^{ - },{d}_{2}^{ - },..{d}_{l}^{ - }}\right\}$的负对数似然。

$$
\mathcal{L} =  - \log \frac{{e}^{f\left( {q,{d}^{ + }}\right) /\tau }}{{e}^{f\left( {q,{d}^{ + }}\right) /\tau } + \mathop{\sum }\limits_{l}{e}^{f\left( {q,{d}_{l}^{ - }}\right) /\tau }} \tag{2}
$$

Where $\tau$ is hyper-parameter of temperature-scaled factor, and an appropriate temperature can help in better optimization (Sachan et al., 2021; Li et al., 2021).

其中$\tau$是温度缩放因子的超参数，适当的温度有助于更好地进行优化（萨坎等人，2021年；李等人，2021年）。

### 3.2 Multi-Viewer Based Architecture

### 3.2 基于多视角器的架构

Limited to single vector representation, the typical bi-encoder faces a challenge that a document contains multiple semantics and can be asked by different potential queries from multi-view. Though some previous studies incorporate dense interaction to allow multiple representations and somehow improve the effectiveness, they usually lead to more additional expensive computation and complicated structure. Therefore, we propose a simple yet effective method to produce multi-view representations by multiple viewers and we will describe it in detail.

由于受限于单向量表示，典型的双编码器面临一个挑战，即文档包含多种语义，并且可以从多个视角被不同的潜在查询询问。尽管一些先前的研究引入了密集交互以允许有多种表示，并在一定程度上提高了有效性，但它们通常会导致更多额外的昂贵计算和复杂的结构。因此，我们提出了一种简单而有效的方法，通过多个视角器来生成多视角表示，我们将详细描述该方法。

As pre-trained BERT has benefited a wide scale of the downstream tasks including sentence-level ones, some work has found [CLS] tend to aggregate the overall meaning of the whole sentence (Koval-eva et al., 2019; Clark et al., 2019). However, our model tends to capture more fine-grained semantic units in a document, so we introduce multiple viewers. Rather than use the latent representation of the [CLS] token, we adopt newly added multiple viewer tokens [VIE] to replace [CLS], which are randomly initialized. For documents input, we add different $\left\lbrack  {{VI}{E}_{i}}\right\rbrack  \left( {\mathrm{i} = 1,2,\ldots ,\mathrm{n}}\right)$ at the beginning of sentence tokens. To avoid effect on the positional encoding of the original input sentences, we set all the position ids of $\left\lbrack  {{VI}{E}_{i}}\right\rbrack$ to 0,and the document sentence tokens start from 1 as the original. Then We leverage the dual encoder to get the representations of queries and documents:

由于预训练的BERT使包括句子级任务在内的广泛下游任务受益，一些研究发现[CLS]倾向于聚合整个句子的整体含义（科瓦列娃等人，2019年；克拉克等人，2019年）。然而，我们的模型倾向于捕捉文档中更细粒度的语义单元，因此我们引入了多个视角器。我们不使用[CLS]标记的潜在表示，而是采用新添加的多个视角器标记[VIE]来代替[CLS]，这些标记是随机初始化的。对于文档输入，我们在句子标记的开头添加不同的$\left\lbrack  {{VI}{E}_{i}}\right\rbrack  \left( {\mathrm{i} = 1,2,\ldots ,\mathrm{n}}\right)$。为了避免对原始输入句子的位置编码产生影响，我们将$\left\lbrack  {{VI}{E}_{i}}\right\rbrack$的所有位置ID都设置为0，并且文档句子标记像原来一样从1开始。然后，我们利用双编码器来获取查询和文档的表示：

$$
E\left( q\right)  = {\operatorname{Enc}}_{q}\left( {\left\lbrack  {VIE}\right\rbrack   \circ  q \circ  \left\lbrack  {SEP}\right\rbrack  }\right)  \tag{3}
$$

$$
E\left( d\right)  = {\operatorname{Enc}}_{d}\left( {\left\lbrack  {\mathrm{{VIE}}}_{1}\right\rbrack  \cdots \left\lbrack  {\mathrm{{VIE}}}_{n}\right\rbrack   \circ  d \circ  \left\lbrack  \mathrm{{SEP}}\right\rbrack  }\right) 
$$

Where $\circ$ is the concatenation operation. [VIE] and [SEP] are special tokens in BERT. ${\operatorname{Enc}}_{q}$ and ${\operatorname{Enc}}_{d}$ mean query and document encoder. We use the last layer hidden states as the query and document embeddings.

其中$\circ$是拼接操作。[VIE]和[SEP]是BERT中的特殊标记。${\operatorname{Enc}}_{q}$和${\operatorname{Enc}}_{d}$分别表示查询编码器和文档编码器。我们使用最后一层的隐藏状态作为查询和文档的嵌入。

The representations of the [VIE] tokens are used as representations of query $q$ and document $d$ ,which are denoted as ${E}_{0}\left( q\right)$ and ${E}_{i}\left( d\right) (i =$ $0,1,\ldots ,k - 1)$ ,respectively. As the query is much shorter than the document and usually represents one concrete meaning, we retain the typical setting to produce only one embedding for the query.

[VIE]标记的表示被用作查询$q$和文档$d$的表示，分别表示为${E}_{0}\left( q\right)$和${E}_{i}\left( d\right) (i =$$0,1,\ldots ,k - 1)$。由于查询比文档短得多，并且通常表示一个具体的含义，我们保留了典型的设置，即为查询仅生成一个嵌入。

<!-- Media -->

<!-- figureText: MaxScore MaxScore ... MaxScore Negatives' Scores Global Contrastive Loss Aggregate Score Local Uniformity Loss Individual Scores $\left\lbrack  {\mathrm{{VIE}}}_{2}\right\rbrack$ $\left\lbrack  {\mathrm{{VIE}}}_{\mathrm{n}}\right\rbrack$ B... Document Encoder $\left\lbrack  {\mathrm{{VIE}}}_{2}\right\rbrack$ ... $\left\lbrack  {\mathrm{{VIE}}}_{\mathrm{n}}\right\rbrack$ ${\mathrm{B}}_{1}$ MaxScore ${\mathrm{{Score}}}_{1}$ Score, Score, [VIE] ${\mathrm{A}}_{\mathrm{m}}$ $\left\lbrack  {\mathrm{{VIE}}}_{1}\right\rbrack$ Query Encoder [VIE] ${\mathrm{A}}_{2}$ ... ${\mathrm{A}}_{\mathrm{m}}$ $\left\lbrack  {\mathrm{{VIE}}}_{1}\right\rbrack$ -->

<img src="https://cdn.noedgeai.com/0195aeeb-a9d7-7d11-ad76-1517c72ca387_4.jpg?x=345&y=186&w=956&h=336&r=0"/>

Figure 3: The general framework of multi-view representation learning with global-local loss. The gray blocks indicates the categories of scores in different layers.

图3：具有全局 - 局部损失的多视角表示学习的总体框架。灰色块表示不同层中的得分类别。

<!-- Media -->

Then the similarity score $f\left( {q,d}\right)$ of query $q$ and document $d$ can be calculated with their dense representations. As shown in Figure.3, we first compute the Individual Scores between the single query embedding and document's multi-view em-beddings, in which we adopt the inner-product. The resulted score corresponding to $\left\lbrack  {{VI}{E}_{i}}\right\rbrack$ is denoted as ${f}_{i}\left( {q,d}\right) \left( {i = 0,1,\ldots ,k - 1}\right)$ . The we adopt a max-pooler to aggregate individual score to the ${Ag}$ - gregate Score $f\left( {q,d}\right)$ as the similarity score for the given query and document pairs:

然后，可以利用查询 $q$ 和文档 $d$ 的密集表示来计算它们的相似度得分 $f\left( {q,d}\right)$。如图3所示，我们首先计算单个查询嵌入与文档的多视图嵌入之间的个体得分，其中采用内积运算。对应于 $\left\lbrack  {{VI}{E}_{i}}\right\rbrack$ 的结果得分记为 ${f}_{i}\left( {q,d}\right) \left( {i = 0,1,\ldots ,k - 1}\right)$。然后，我们采用最大池化器将个体得分聚合为 ${Ag}$ - 聚合得分 $f\left( {q,d}\right)$，作为给定查询 - 文档对的相似度得分：

$$
f\left( {q,d}\right)  = \mathop{\operatorname{Max}}\limits_{i}\left\{  {{f}_{i}\left( {q,d}\right) }\right\}   \tag{4}
$$

$$
 = \mathop{\operatorname{Max}}\limits_{i}\left\{  {\operatorname{sim}\left( {{E}_{0}\left( q\right) ,{E}_{i}\left( d\right) }\right) }\right\}  
$$

### 3.3 Global-Local Loss

### 3.3 全局 - 局部损失

In order to encourage the multiple viewers to better align to different potential queries, we introduce a Global-Local Loss to optimize the training of multi-view architecture. It combines the global contrastive loss and the local uniformity loss.

为了促使多个视图更好地与不同的潜在查询对齐，我们引入了全局 - 局部损失来优化多视图架构的训练。它结合了全局对比损失和局部均匀性损失。

$$
\mathcal{L} = {\mathcal{L}}_{\text{global }} + \lambda {\mathcal{L}}_{\text{local }} \tag{5}
$$

The global contrastive loss is inherited from the traditional bi-encoder. Given the query and a positive document ${d}^{ + }$ against a set of negatives $\left\{  {{d}_{1}^{ - },{d}_{2}^{ - },..{d}_{l}^{ - }}\right\}$ ,It is computed as follows:

全局对比损失继承自传统的双编码器。给定一个查询和一个正文档 ${d}^{ + }$ 以及一组负文档 $\left\{  {{d}_{1}^{ - },{d}_{2}^{ - },..{d}_{l}^{ - }}\right\}$，其计算方式如下：

$$
{\mathcal{L}}_{\text{global }} =  - \log \frac{{e}^{f\left( {q,{d}^{ + }}\right) /\tau }}{{e}^{f\left( {q,{d}^{ + }}\right) /\tau } + \mathop{\sum }\limits_{l}{e}^{f\left( {q,{d}_{l}^{ - }}\right) /\tau }} \tag{6}
$$

To improve the uniformity of multi-view embedding space, we propose applying Local Uniformity Loss among the different viewers in Eq.7. For a specific query, one of the multi-view document representations will be matched by max-score in Eq.4. The local uniformity loss enforces the selected viewer to more closely align with the query and distinguish from other viewers.

为了提高多视图嵌入空间的均匀性，我们建议在公式7中的不同视图之间应用局部均匀性损失。对于一个特定的查询，多视图文档表示中的一个将通过公式4中的最大得分进行匹配。局部均匀性损失促使所选视图与查询更紧密地对齐，并与其他视图区分开来。

$$
{\mathcal{L}}_{\text{local }} =  - \log \frac{{e}^{f\left( {q,{d}^{ + }}\right) /\tau }}{\mathop{\sum }\limits_{k}{e}^{{fi}\left( {q,{d}^{ + }}\right) /\tau }} \tag{7}
$$

To further encourage more different viewers to be activated, we adopt an annealed temperature in Eq.8, to gradually tune the sharpness of viewers' softmax distribution. In the start stage of training with a high temperature, the softmax values tend to have a uniform distribution over the viewers, to make every viewer fair to be selected and get back gradient from train data. As the training process goes, the temperature decreases to make optimization more stable.

为了进一步促使更多不同的视图被激活，我们在公式8中采用退火温度，以逐渐调整视图的softmax分布的锐度。在训练的初始阶段，温度较高，softmax值在视图上倾向于均匀分布，从而使每个视图都有公平的机会被选中并从训练数据中获取梯度。随着训练过程的进行，温度降低，以使优化更加稳定。

$$
\tau  = \max \{ {0.3},\exp \left( {-{\alpha t}}\right) \}  \tag{8}
$$

Where $\alpha$ is a hyper-parameter to control the annealing speed, $t$ denotes the training epochs,and the temperature updates every epoch. To simplify the settings, we use the same annealed temperature in ${\mathcal{L}}_{\text{local }}$ and ${\mathcal{L}}_{\text{global }}$ and our experiments validate the annealed temperature works mainly in conjunction with ${\mathcal{L}}_{\text{local }}$ through multiple viewers.

其中 $\alpha$ 是一个控制退火速度的超参数，$t$ 表示训练轮数，并且温度每轮更新一次。为了简化设置，我们在 ${\mathcal{L}}_{\text{local }}$ 和 ${\mathcal{L}}_{\text{global }}$ 中使用相同的退火温度，并且我们的实验验证了退火温度主要通过多个视图与 ${\mathcal{L}}_{\text{local }}$ 协同工作。

During inference, we build the index of all the reviewer embeddings of documents, and then our model directly retrieves from the built index leveraging approximate nearest neighbor (ANN) technique. However, both Poly-Encoder (Humeau et al., 2020) and DRPQ (Tang et al., 2021) adopt attention-based aggregator containing softmax or sum operator so that the fast ANN can't be directly applied. Though DRPQ proposes to approximate softmax to max operation, it still needs to first recall a set of candidates then rerank them using the complex aggregator, leading to expensive computation and complicated procedure. In contrast, MVR can be directly applied in first-stage retrieval without post-computing process as them. Though the size of the index will grow by viewer number $k$ ,the time complexity can be sublinear in index size (An-doni et al., 2018) due to the efficiency of ANN technique(Johnson et al., 2019).

在推理过程中，我们构建文档所有视图嵌入的索引，然后我们的模型利用近似最近邻（ANN）技术直接从构建的索引中进行检索。然而，Poly - Encoder（Humeau等人，2020）和DRPQ（Tang等人，2021）都采用了包含softmax或求和运算符的基于注意力的聚合器，因此无法直接应用快速ANN。尽管DRPQ提出将softmax近似为最大操作，但它仍然需要先召回一组候选文档，然后使用复杂的聚合器对它们进行重新排序，这导致计算成本高且过程复杂。相比之下，MVR可以直接应用于第一阶段的检索，而无需像它们那样进行后计算过程。尽管索引的大小会随着视图数量 $k$ 而增加，但由于ANN技术的效率（Johnson等人，2019），时间复杂度可以是索引大小的亚线性（Andoni等人，2018）。

<!-- Media -->

<table><tr><td rowspan="2">Method</td><td colspan="3">SQuAD</td><td colspan="3">Natural Question</td><td colspan="3">Trivia QA</td></tr><tr><td>R@5</td><td>R@20</td><td>R@100</td><td>R@5</td><td>R@20</td><td>R@100</td><td>R@5</td><td>R@20</td><td>R@100</td></tr><tr><td>BM25 (Yang et al., 2017)</td><td>-</td><td>-</td><td>-</td><td>-</td><td>59.1</td><td>73.7</td><td>-</td><td>66.9</td><td>76.7</td></tr><tr><td>DPR (Karpukhin et al., 2020)</td><td>-</td><td>76.4</td><td>84.8</td><td>-</td><td>74.4</td><td>85.3</td><td>-</td><td>79.3</td><td>84.9</td></tr><tr><td>ANCE (Xiong et al., 2020)</td><td>-</td><td>-</td><td>-</td><td>-</td><td>81.9</td><td>87.5</td><td>-</td><td>80.3</td><td>85.3</td></tr><tr><td>RocketQA (Qu et al., 2021)</td><td>-</td><td>-</td><td>-</td><td>74.0</td><td>82.7</td><td>88.5</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Condenser (Gao and Callan, 2021a)</td><td>-</td><td>-</td><td>-</td><td>-</td><td>83.2</td><td>88.4</td><td>-</td><td>81.9</td><td>86.2</td></tr><tr><td>DPR-PAQ (Oguz et al., 2021)</td><td>-</td><td>-</td><td>-</td><td>74.5</td><td>83.7</td><td>88.6</td><td>-</td><td>-</td><td>-</td></tr><tr><td>DRPQ (Tang et al., 2021)</td><td>-</td><td>80.5</td><td>88.6</td><td>-</td><td>82.3</td><td>88.2</td><td>-</td><td>80.5</td><td>85.8</td></tr><tr><td>coCondenser (Gao and Callan, 2021b)</td><td>-</td><td>-</td><td>-</td><td>75.8</td><td>84.3</td><td>89.0</td><td>76.8</td><td>83.2</td><td>87.3</td></tr><tr><td>coCondenser(reproduced)</td><td>73.2</td><td>81.8</td><td>88.7</td><td>75.4</td><td>84.1</td><td>88.8</td><td>76.4</td><td>82.7</td><td>86.8</td></tr><tr><td>MVR</td><td>76.4</td><td>84.2</td><td>89.8</td><td>76.2</td><td>84.8</td><td>89.3</td><td>77.1</td><td>83.4</td><td>87.4</td></tr></table>

<table><tbody><tr><td rowspan="2">方法</td><td colspan="3">斯坦福问答数据集（SQuAD）</td><td colspan="3">自然问题数据集（Natural Question）</td><td colspan="3">常识问答数据集（Trivia QA）</td></tr><tr><td>R@5</td><td>R@20</td><td>R@100</td><td>R@5</td><td>R@20</td><td>R@100</td><td>R@5</td><td>R@20</td><td>R@100</td></tr><tr><td>BM25算法（杨等人，2017年）</td><td>-</td><td>-</td><td>-</td><td>-</td><td>59.1</td><td>73.7</td><td>-</td><td>66.9</td><td>76.7</td></tr><tr><td>密集段落检索器（DPR，卡尔普欣等人，2020年）</td><td>-</td><td>76.4</td><td>84.8</td><td>-</td><td>74.4</td><td>85.3</td><td>-</td><td>79.3</td><td>84.9</td></tr><tr><td>ANCE算法（熊等人，2020年）</td><td>-</td><td>-</td><td>-</td><td>-</td><td>81.9</td><td>87.5</td><td>-</td><td>80.3</td><td>85.3</td></tr><tr><td>火箭问答模型（RocketQA，曲等人，2021年）</td><td>-</td><td>-</td><td>-</td><td>74.0</td><td>82.7</td><td>88.5</td><td>-</td><td>-</td><td>-</td></tr><tr><td>凝聚器模型（Condenser，高和卡兰，2021a）</td><td>-</td><td>-</td><td>-</td><td>-</td><td>83.2</td><td>88.4</td><td>-</td><td>81.9</td><td>86.2</td></tr><tr><td>DPR - PAQ模型（奥古兹等人，2021年）</td><td>-</td><td>-</td><td>-</td><td>74.5</td><td>83.7</td><td>88.6</td><td>-</td><td>-</td><td>-</td></tr><tr><td>DRPQ模型（唐等人，2021年）</td><td>-</td><td>80.5</td><td>88.6</td><td>-</td><td>82.3</td><td>88.2</td><td>-</td><td>80.5</td><td>85.8</td></tr><tr><td>协同凝聚器模型（coCondenser，高和卡兰，2021b）</td><td>-</td><td>-</td><td>-</td><td>75.8</td><td>84.3</td><td>89.0</td><td>76.8</td><td>83.2</td><td>87.3</td></tr><tr><td>协同凝聚器模型（复现版）</td><td>73.2</td><td>81.8</td><td>88.7</td><td>75.4</td><td>84.1</td><td>88.8</td><td>76.4</td><td>82.7</td><td>86.8</td></tr><tr><td>多向量表示（MVR）</td><td>76.4</td><td>84.2</td><td>89.8</td><td>76.2</td><td>84.8</td><td>89.3</td><td>77.1</td><td>83.4</td><td>87.4</td></tr></tbody></table>

Table 1: Retrieval performance on SQuAD dev, Natural Question test and Trivia QA test. The best performing models are marked bold and the results unavailable are left blank.

表1：在SQuAD开发集、自然问题测试集和Trivia QA测试集上的检索性能。表现最佳的模型用粗体标记，无法获取的结果留空。

<!-- Media -->

## 4 Experiments

## 4 实验

### 4.1 Datasets

### 4.1 数据集

Natural Questions (Kwiatkowski et al., 2019) is a popular open-domain retrieval dataset, in which the questions are real Google search queries and answers were manually annotated from Wikipedia. TriviaQA (Joshi et al., 2017) contains a set of trivia questions with answers that were originally scraped from the Web.

自然问题数据集（Natural Questions，Kwiatkowski等人，2019年）是一个流行的开放域检索数据集，其中的问题是真实的谷歌搜索查询，答案是从维基百科手动标注的。Trivia QA数据集（Joshi等人，2017年）包含一组琐事问题及其答案，这些答案最初是从网络上抓取的。

SQuAD Open(Rajpurkar et al., 2016) contains the questions and answers originating from a reading comprehension dataset, and it has been used widely used for open-domain retrieval research.

SQuAD开放数据集（SQuAD Open，Rajpurkar等人，2016年）包含源自阅读理解数据集的问题和答案，它已被广泛用于开放域检索研究。

We follow the same procedure in (Karpukhin et al., 2020) to preprocess and extract the passage candidate set from the English Wikipedia dump, resulting to about two million passages that are nonoverlapping chunks of 100 words. Both NQ and TQA have about ${60}\mathrm{\;K}$ training data after processing and SQuAd has ${70}\mathrm{k}$ . Currently,the authors release all the datasets of NQ and TQ. For SQuAD, only the development set is available. So we conduct experiments on these three datasets, and evaluate the top $5/{20}/{100}$ accuracy on the SQuAD dev set and test set of NQ and TQ. We have counted how many queries correspond to one same document and the average number of queries of SQuAD, Natural Questions and Trivia QA are 2.7, 1.5 and 1.2, which indicates the multi-view problem is common in open-domain retrieval.

我们遵循（Karpukhin等人，2020年）中的相同步骤，对英文维基百科转储进行预处理并提取候选段落集，得到约两百万个段落，这些段落是不重叠的100个单词的片段。处理后，自然问题数据集（NQ）和Trivia QA数据集（TQA）都有约${60}\mathrm{\;K}$的训练数据，而SQuAD数据集有${70}\mathrm{k}$。目前，作者发布了自然问题数据集（NQ）和Trivia QA数据集（TQ）的所有数据。对于SQuAD数据集，只有开发集可用。因此，我们在这三个数据集上进行实验，并评估SQuAD开发集以及自然问题数据集（NQ）和Trivia QA数据集（TQ）测试集上的前$5/{20}/{100}$准确率。我们统计了有多少查询对应于同一文档，SQuAD、自然问题数据集和Trivia QA的平均查询数分别为2.7、1.5和1.2，这表明多视角问题在开放域检索中很常见。

### 4.2 Implementation Details

### 4.2 实现细节

We train MVR model following the hyper-parameter setting of DPR (Karpukhin et al., 2020). All models are trained for 40 epochs on 8 Tesla V100 32GB GPUs. We tune different viewer numbers on the SQuAD dataset and find the best is 8 , then we adopt it on all the datasets. To make a fair comparison, we follow coCondenser (Gao and Callan, 2021b) to adopt mined hard negatives and warm-up pre-training strategies, which are also adopted in recent works (Oğuz et al., 2021; Gao and Callan, 2021a) and show promotion. Note that we only adopt these strategies when compared to them, while in the ablation studies our models are built only with the raw DPR model. During inference, we apply the passage encoder to encode all the passages and index them using the Faiss IndexFlatIP index (Johnson et al., 2019).

我们按照密集段落检索器（DPR，Karpukhin等人，2020年）的超参数设置来训练多视角检索（MVR）模型。所有模型在8块32GB的特斯拉V100 GPU上训练40个轮次。我们在SQuAD数据集上调整不同的视角数量，发现最佳数量为8，然后在所有数据集上采用该数量。为了进行公平比较，我们遵循协同冷凝器（coCondenser，Gao和Callan，2021b）采用挖掘难负样本和预热预训练策略，这些策略也被近期的研究（Oğuz等人，2021年；Gao和Callan，2021a）采用并显示出提升效果。请注意，我们仅在与这些方法进行比较时采用这些策略，而在消融实验中，我们的模型仅基于原始的密集段落检索器（DPR）模型构建。在推理过程中，我们使用段落编码器对所有段落进行编码，并使用Faiss IndexFlatIP索引（Johnson等人，2019年）对它们进行索引。

### 4.3 Retrieval Performance

### 4.3 检索性能

We compare our MVR model with previous state-of-the-art methods. Among these methods, DRPQ (Tang et al., 2021) achieved the best results in multiple embeddings methods, which is the main compared baseline for our model. In addition, we also compare to the recent strong dense retriever, including ANCE (Xiong et al., 2020), Ro-cekteQA (Qu et al., 2021), Condenser (Gao and Callan, 2021a), DPR-PAQ (Oguz et al., 2021) and coCondenser (Gao and Callan, 2021b). For co-Condenser, we reproduced its results and find it a little lower than his reported one, maybe due to its private repo and tricks. Overall, these methods mainly focus on mining hard negative samples, knowledge distillation or pre-training strategies to improve dense retrieval. And our MVR framework is orthogonal to them and can be combined with them for better promotion.

我们将我们的多视角检索（MVR）模型与之前的最先进方法进行比较。在这些方法中，基于查询感知蒸馏的密集检索（DRPQ，Tang等人，2021年）在多嵌入方法中取得了最佳结果，这是我们模型的主要比较基线。此外，我们还与近期强大的密集检索器进行比较，包括基于对抗负样本的上下文编码器（ANCE，Xiong等人，2020年）、Ro-cekteQA（Qu等人，2021年）、冷凝器（Condenser，Gao和Callan，2021a）、基于预定义答案查询的密集段落检索器（DPR - PAQ，Oguz等人，2021年）和协同冷凝器（coCondenser，Gao和Callan，2021b）。对于协同冷凝器（co - Condenser），我们复现了其结果，发现比其报告的结果略低，可能是由于其私有代码库和技巧的原因。总体而言，这些方法主要侧重于挖掘难负样本、知识蒸馏或预训练策略来改进密集检索。而我们的多视角检索（MVR）框架与它们正交，可以与它们结合以获得更好的提升效果。

<!-- Media -->

<table><tr><td>Models</td><td>R@5</td><td>R@20</td><td>R@100</td></tr><tr><td>DPR $\left( {\mathrm{k} = 1}\right)$</td><td>66.2</td><td>76.8</td><td>85.2</td></tr><tr><td>ME-BERT(k=4)</td><td>66.8</td><td>77.6</td><td>85.5</td></tr><tr><td>ME-BERT(k=8)</td><td>67.3</td><td>77.9</td><td>86.1</td></tr><tr><td>MVR(k=4)</td><td>68.5</td><td>78.5</td><td>85.8</td></tr><tr><td>MVR(k=6)</td><td>72.3</td><td>80.3</td><td>86.4</td></tr><tr><td>MVR(k=8)</td><td>75.5</td><td>83.2</td><td>87.9</td></tr><tr><td>MVR(k=12)</td><td>74.8</td><td>82.9</td><td>87.4</td></tr></table>

<table><tbody><tr><td>模型</td><td>R@5</td><td>R@20</td><td>R@100</td></tr><tr><td>密集段落检索器（DPR $\left( {\mathrm{k} = 1}\right)$）</td><td>66.2</td><td>76.8</td><td>85.2</td></tr><tr><td>多证据BERT（ME-BERT(k=4)）</td><td>66.8</td><td>77.6</td><td>85.5</td></tr><tr><td>多证据BERT（ME-BERT(k=8)）</td><td>67.3</td><td>77.9</td><td>86.1</td></tr><tr><td>多视图推理（MVR(k=4)）</td><td>68.5</td><td>78.5</td><td>85.8</td></tr><tr><td>多视图推理（MVR(k=6)）</td><td>72.3</td><td>80.3</td><td>86.4</td></tr><tr><td>多视图推理（MVR(k=8)）</td><td>75.5</td><td>83.2</td><td>87.9</td></tr><tr><td>多视图推理（MVR(k=12)）</td><td>74.8</td><td>82.9</td><td>87.4</td></tr></tbody></table>

Table 2: Performance of different viewers' number in MVR and compared models.

表2：MVR及对比模型在不同观众数量下的性能表现。

<table><tr><td>Models</td><td>R@5</td><td>R@20</td><td>R@100</td></tr><tr><td>(0)MVR $\left( {\alpha  = {0.1}}\right)$</td><td>75.5</td><td>83.2</td><td>87.9</td></tr><tr><td>(1)w/o LC loss</td><td>73.7</td><td>82.1</td><td>86.5</td></tr><tr><td>(2)w/o Annealed $\tau$ (Fixed=1)</td><td>74.3</td><td>81.9</td><td>86.8</td></tr><tr><td>(3)w/o LC loss + Annealed $\tau$</td><td>72.8</td><td>81.0</td><td>85.8</td></tr><tr><td>(4)w/o Multiple Viewers</td><td>66.7</td><td>77.1</td><td>85.7</td></tr><tr><td>(5)Fixed $\tau  = {10}$</td><td>70.2</td><td>79.3</td><td>85.4</td></tr><tr><td>(6)Fixed $\tau  = {0.3}$</td><td>74.6</td><td>82.4</td><td>87.3</td></tr><tr><td>(7)Fixed $\tau  = {0.1}$</td><td>72.3</td><td>81.2</td><td>85.9</td></tr><tr><td>(8)Annealed $\tau \left( {\alpha  = {0.3}}\right)$</td><td>74.7</td><td>82.0</td><td>87.4</td></tr><tr><td>(9)Annealed $\tau \left( {\alpha  = {0.03}}\right)$</td><td>73.9</td><td>81.8</td><td>86.5</td></tr></table>

<table><tbody><tr><td>模型</td><td>R@5</td><td>R@20</td><td>R@100</td></tr><tr><td>(0)多视图重建（MVR） $\left( {\alpha  = {0.1}}\right)$</td><td>75.5</td><td>83.2</td><td>87.9</td></tr><tr><td>(1)无局部一致性损失</td><td>73.7</td><td>82.1</td><td>86.5</td></tr><tr><td>(2)无退火 $\tau$ （固定值 = 1）</td><td>74.3</td><td>81.9</td><td>86.8</td></tr><tr><td>(3)无局部一致性损失 + 退火 $\tau$</td><td>72.8</td><td>81.0</td><td>85.8</td></tr><tr><td>(4)无多视角</td><td>66.7</td><td>77.1</td><td>85.7</td></tr><tr><td>(5)固定 $\tau  = {10}$</td><td>70.2</td><td>79.3</td><td>85.4</td></tr><tr><td>(6)固定 $\tau  = {0.3}$</td><td>74.6</td><td>82.4</td><td>87.3</td></tr><tr><td>(7)固定 $\tau  = {0.1}$</td><td>72.3</td><td>81.2</td><td>85.9</td></tr><tr><td>(8)退火 $\tau \left( {\alpha  = {0.3}}\right)$</td><td>74.7</td><td>82.0</td><td>87.4</td></tr><tr><td>(9)退火 $\tau \left( {\alpha  = {0.03}}\right)$</td><td>73.9</td><td>81.8</td><td>86.5</td></tr></tbody></table>

Table 3: Ablation study on Global-local Loss on SQuAD dev set.

表3：SQuAD开发集上全局 - 局部损失的消融研究。

<!-- Media -->

As shown in Table 1, we can see that our proposed MVR performs better than other models. Compared to DRPQ which performs best in the previous multi-vector models, MVR can outperform it by a large margin, further confirming the superiority of our multi-view representation. It's worth noting that our model improves more on the SQuAD dataset, maybe due to the dataset containing more documents that correspond to multiple queries as we state in Sec.4.1. It indicates that MVR can address the multi-view problem better than other models.

如表1所示，我们可以看到我们提出的MVR（多视图表示，Multi - View Representation）比其他模型表现更好。与之前多向量模型中表现最佳的DRPQ相比，MVR能大幅超越它，进一步证实了我们的多视图表示的优越性。值得注意的是，我们的模型在SQuAD数据集上的提升更大，可能是因为如我们在4.1节所述，该数据集包含更多对应多个查询的文档。这表明MVR比其他模型能更好地解决多视图问题。

### 4.4 Ablation Study

### 4.4 消融研究

Impact of Viewers' Number: We conduct ablation studies on the development set of SQuAD open. For fair comparison, we build all the models mentioned in the following based on DPR toolkit, including MEBERT and MVR. The results are shown in Table 2, and the first block shows the results of DPR and MEBERT which adopt the first $k$ token embeddings. Compared to DPR and MEBERT, our model shows strong performance, which indicates the multi-view representation is effective and useful. Then, we analyze how the different numbers of viewers $\left( {k = 4,6,8,{12}}\right)$ affect performance in MVR. We find that the model achieves the best performance when $k = 8$ . When $\mathrm{k}$ increase to $k = {12}$ or larger, it leads little decrease in the performance, maybe due to there being not so many individual views in a document.

视角数量的影响：我们在SQuAD开放开发集上进行了消融研究。为了进行公平比较，我们基于DPR（密集段落检索器，Dense Passage Retriever）工具包构建了以下提到的所有模型，包括MEBERT和MVR。结果如表2所示，第一部分展示了采用前$k$个词元嵌入的DPR和MEBERT的结果。与DPR和MEBERT相比，我们的模型表现出色，这表明多视图表示是有效且有用的。然后，我们分析了不同数量的视角$\left( {k = 4,6,8,{12}}\right)$如何影响MVR的性能。我们发现当$k = 8$时，模型达到最佳性能。当$\mathrm{k}$增加到$k = {12}$或更大时，性能略有下降，可能是因为文档中没有那么多独立的视角。

<!-- Media -->

<table><tr><td>Method</td><td>Doc Encoding</td><td>Retrieval</td></tr><tr><td>DPR</td><td>2.5ms</td><td>10ms</td></tr><tr><td>ColBERT</td><td>2.5ms</td><td>320ms</td></tr><tr><td>MEBERT</td><td>2.5ms</td><td>25ms</td></tr><tr><td>DRPQ</td><td>5.3ms</td><td>45ms</td></tr><tr><td>MVR</td><td>2.5ms</td><td>25ms</td></tr></table>

<table><tbody><tr><td>方法</td><td>文档编码</td><td>检索</td></tr><tr><td>密集段落检索器（DPR）</td><td>2.5毫秒</td><td>10毫秒</td></tr><tr><td>ColBERT模型</td><td>2.5毫秒</td><td>320毫秒</td></tr><tr><td>MEBERT模型</td><td>2.5毫秒</td><td>25毫秒</td></tr><tr><td>DRPQ模型</td><td>5.3毫秒</td><td>45毫秒</td></tr><tr><td>多视图检索（MVR）</td><td>2.5毫秒</td><td>25毫秒</td></tr></tbody></table>

Table 4: Time cost of online and offline computing in SQuAD retrieval task.

表4：SQuAD检索任务中在线和离线计算的时间成本。

<!-- Media -->

Analysis on Global-local Loss: In this part, we conduct more detailed ablation study and analysis of our proposed Global-local Loss. As shown in Table 3 , we gradually reduce the strategies adopted in our model. We find not having either local uniformity loss (LC loss in table) or annealed temperature damages performance, and it decreases more without both of them. We also provide more experimental results to show the effectiveness of the annealed temperature. We first tune the fixed temperature, find it between 0.3 to 1 is beneficial. We adopt the annealed temperature annealed from 1 to 0.3 gradually as in Eq.8,finding a suitable speed $\left( {\alpha  = {0.1}}\right)$ can better help with optimization during training. Note that the model w/o Multiple Viewers can be seen as DPR with annealed $\tau$ ,just little higher than raw DPR in Table 2,while annealed $\tau$ improves more when using multi-viewer. It indicates our annealed strategy plays a more important role in multi-view learning.

全局 - 局部损失分析：在这部分，我们对所提出的全局 - 局部损失进行更详细的消融研究和分析。如表3所示，我们逐步减少模型中采用的策略。我们发现，不采用局部一致性损失（表中的LC损失）或退火温度都会损害性能，若两者都不采用，性能下降更明显。我们还提供了更多实验结果来展示退火温度的有效性。我们首先调整固定温度，发现0.3到1之间的温度是有益的。我们采用如公式8所示从1逐渐退火到0.3的退火温度，发现合适的退火速度$\left( {\alpha  = {0.1}}\right)$能在训练过程中更好地辅助优化。注意，没有多视角模块的模型可以看作是采用了退火$\tau$的DPR（密集段落检索器），在表2中仅略高于原始的DPR，而在使用多视角模块时，退火$\tau$能带来更大的提升。这表明我们的退火策略在多视角学习中起着更重要的作用。

Efficiency Analysis: We test the efficiency of our model on 4 Nvidia Tesla V100 GPU for the SQuAD dev set, as shown in Table 4. We record the encoding time per document and retrieval time per query, and don't include the query encoding time for it is equal for all the models. To compare our approach with other different models, we also record the retrieval time of other related models. We can see that our model spends the same encoding time as DPR, while DRPQ needs additional time to run K-means clustering. With the support of Faiss, the retrieval time MVR cost is near to DPR and less than ColBERT (Khattab and Zaharia, 2020) and DRPQ (Tang et al., 2021) which need additional post-computing as we state in Sec.2.1.

效率分析：如表4所示，我们在4块英伟达特斯拉V100 GPU上对SQuAD开发集测试了我们模型的效率。我们记录了每个文档的编码时间和每个查询的检索时间，并且不包括查询编码时间，因为所有模型的查询编码时间是相同的。为了将我们的方法与其他不同模型进行比较，我们还记录了其他相关模型的检索时间。我们可以看到，我们的模型与DPR的编码时间相同，而DRPQ（动态检索预训练查询）需要额外的时间来运行K - 均值聚类。在Faiss的支持下，MVR（多视角检索器）的检索时间接近DPR，并且少于ColBERT（卡塔布和扎哈里亚，2020年）和DRPQ（唐等人，2021年），正如我们在2.1节中所述，后两者需要额外的后计算。

<!-- Media -->

<table><tr><td>Models</td><td>R@5</td><td>R@20</td><td>R@100</td></tr><tr><td>DPR</td><td>66.2</td><td>76.8</td><td>85.2</td></tr><tr><td>MVR</td><td>75.5</td><td>83.2</td><td>87.9</td></tr><tr><td>Sentence-level</td><td>62.1</td><td>73.2</td><td>81.9</td></tr><tr><td>4-equal-splits</td><td>57.2</td><td>69.3</td><td>78.5</td></tr><tr><td>8-equal-splits</td><td>44.2</td><td>57.9</td><td>69.4</td></tr></table>

<table><tbody><tr><td>模型</td><td>R@5</td><td>R@20</td><td>R@100</td></tr><tr><td>双编码器检索器（DPR）</td><td>66.2</td><td>76.8</td><td>85.2</td></tr><tr><td>多向量检索器（MVR）</td><td>75.5</td><td>83.2</td><td>87.9</td></tr><tr><td>句子级别</td><td>62.1</td><td>73.2</td><td>81.9</td></tr><tr><td>四等分</td><td>57.2</td><td>69.3</td><td>78.5</td></tr><tr><td>八等分</td><td>44.2</td><td>57.9</td><td>69.4</td></tr></tbody></table>

Table 5: Performance of different sentence-level retrieval models on SQuAD dev.

表5：不同句子级检索模型在SQuAD开发集上的性能。

<!-- Media -->

## 5 Further Analysis

## 5 进一步分析

### 5.1 Comparison to Sentence-level Retrieval

### 5.1 与句子级检索的比较

To analyze the difference between MVR and sentence-level retrieval which is another way to produce multiple embeddings, we design several models as shown in Table 5. Sentence-level means that we split all the passages into individual sentences with NLTK toolkit ${}^{1}$ ,resulting to an average of 5.5 sentences per passage. The new positives are the sentences containing answers in the original positives, and new negatives are all the split sentences of original negatives. K-equal-splits means the DPR's original 100-words-long passages are split into $k$ equal long sequences and training positives and negatives as Sentence-level's methods. Compared to MVR, Sentence-level drops a lot even lower than DPR maybe for they lose contextual information in passages. It also indicates that the multi-view embeddings of MVR do not just correspond to sentence embeddings, but capture semantic meanings from different views. For even a single sentence may contain diverse information that can answer different potential queries (as in Fig.1). The k-equal-split methods perform worse much for it further lose the sentence structure and may contain more noise.

为了分析多视图表示（MVR）与句子级检索（另一种生成多个嵌入的方法）之间的差异，我们设计了几个模型，如表5所示。句子级是指我们使用NLTK工具包${}^{1}$将所有段落拆分为单个句子，平均每个段落有5.5个句子。新的正样本是原始正样本中包含答案的句子，新的负样本是原始负样本的所有拆分句子。K等分是指将DPR原来100词长的段落拆分为$k$个等长序列，并按照句子级的方法训练正样本和负样本。与MVR相比，句子级检索的性能大幅下降，甚至低于DPR，可能是因为它们丢失了段落中的上下文信息。这也表明MVR的多视图嵌入不仅仅对应于句子嵌入，而是从不同视图捕捉语义信息。因为即使是单个句子也可能包含可以回答不同潜在查询的多样化信息（如图1所示）。K等分方法的性能更差，因为它进一步丢失了句子结构，并且可能包含更多噪声。

### 5.2 Analysis of Multi-View Embeddings

### 5.2 多视图嵌入分析

To further show the effectiveness of our proposed MVR framework, we evaluate the distribution of multi-view document representations. We conduct evaluations on the randomly sampled subset of SQuAD development set,which contains ${1.5}\mathrm{k}$ query-doc pairs and each document has an average of 4.8 corresponding questions. We adopt two metrics Local Variation and Perplexity (Brown et al., 1992)(denoted as ${LV}$ and ${PPL}$ ) to illustrate the effect of different methods. We first compute the normalized scores between the document's multiview embeddings and query embedding as in Eq.4, and record the scores ${f}_{i}\left( {q,d}\right)$ of all the viewers. Then Local Variation of a query-doc pair can be computed as follows, and then we average it on all the pairs.

为了进一步展示我们提出的MVR框架的有效性，我们评估了多视图文档表示的分布。我们对SQuAD开发集的随机抽样子集进行评估，该子集包含${1.5}\mathrm{k}$个查询 - 文档对，每个文档平均有4.8个相应的问题。我们采用两个指标局部变异（Local Variation）和困惑度（Perplexity）（Brown等人，1992）（分别表示为${LV}$和${PPL}$）来说明不同方法的效果。我们首先按照公式4计算文档的多视图嵌入与查询嵌入之间的归一化分数，并记录所有视图的分数${f}_{i}\left( {q,d}\right)$。然后可以按如下方式计算查询 - 文档对的局部变异，然后对所有对求平均值。

$$
{LV} = \operatorname{Max}\left( {{f}_{i}\left( {q,d}\right) }\right)  - \frac{\mathop{\sum }\limits_{k}{f}_{i}\left( {q,d}\right)  - \operatorname{Max}\left( {{f}_{i}\left( {q,d}\right) }\right) }{k - 1} \tag{9}
$$

<!-- Media -->

<table><tr><td>Models</td><td>PPL</td><td>LV</td></tr><tr><td>ME-BERT</td><td>1.02</td><td>0.159</td></tr><tr><td>MVR</td><td>3.19</td><td>0.126</td></tr><tr><td>MVR w/o LC loss</td><td>3.23</td><td>0.052</td></tr><tr><td>MVR w/o Annealed $\tau$</td><td>2.95</td><td>0.118</td></tr></table>

<table><tbody><tr><td>模型</td><td>困惑度（PPL）</td><td>潜在变量（LV）</td></tr><tr><td>多嵌入BERT（ME - BERT）</td><td>1.02</td><td>0.159</td></tr><tr><td>多视图重建（MVR）</td><td>3.19</td><td>0.126</td></tr><tr><td>无局部一致性损失的多视图重建（MVR w/o LC loss）</td><td>3.23</td><td>0.052</td></tr><tr><td>无退火$\tau$的多视图重建（MVR w/o Annealed $\tau$）</td><td>2.95</td><td>0.118</td></tr></tbody></table>

Table 6: Analysis of multi-view embeddings produced by different methods.

表6：不同方法生成的多视图嵌入分析。

<!-- Media -->

The Local Variation measures the distance of the max scores to the average of the others, which can curve the uniformity of different viewers. The higher it is, the more diversely the multi-view em-beddings are distributed.

局部变异（Local Variation）衡量的是最大得分与其他得分平均值之间的距离，它可以反映不同视角的一致性。该值越高，多视图嵌入的分布就越分散。

Then we collect the index of the viewer having the max score, and group the indexes of different queries corresponding to the same documents. Next, we can get the distributions of different indexes denoted as ${p}_{i}$ . The Perplexity can be computed as follows:

然后，我们收集得分最高的视角的索引，并将对应于同一文档的不同查询的索引进行分组。接下来，我们可以得到用${p}_{i}$表示的不同索引的分布。困惑度（Perplexity）的计算方法如下：

$$
{PPL} = \exp \left( {-\mathop{\sum }\limits_{m}{p}_{i}\log {p}_{i}}\right)  \tag{10}
$$

If different viewers are matched to totally different queries,the ${p}_{i}$ tends to be a uniform distribution and PPL goes up. The comparison results are shown in Table 6. When evaluating MEBERT, we find its multiple embeddings collapse into the same [CLS] embeddings rather than using the different token embeddings. So its PPL is near to one and Local Variation is too high. For MVR model, we find that without local uniformity loss (LC loss in short), the Local Variation drops rapidly, indicating our proposed LC loss can improve the uniformity of different viewers. In addition, MVR w/o annealed $\tau$ will damage the PPL,which also confirms it does help activate more viewers and align them better with different queries.

如果不同的视角与完全不同的查询相匹配，${p}_{i}$趋于均匀分布，困惑度（PPL）会上升。比较结果如表6所示。在评估MEBERT时，我们发现它的多个嵌入会坍缩为相同的[CLS]嵌入，而不是使用不同的词元嵌入。因此，它的困惑度接近1，局部变异过高。对于MVR模型，我们发现如果没有局部一致性损失（简称LC损失），局部变异会迅速下降，这表明我们提出的LC损失可以提高不同视角的一致性。此外，没有退火$\tau$的MVR会损害困惑度，这也证实了它确实有助于激活更多视角，并使它们与不同的查询更好地对齐。

### 5.3 Qualitative Analysis

### 5.3 定性分析

As shown in Table 7, there are two examples retrieved by DPR and MVR for qualitative analysis. The top scoring passages retrieved by DPR can't give a clear answer for the queries, though they seem to have a similar topic to the queries. In contrast, our MVR is able to return the correct answers when the passages contain rich information and diverse semantics. Take the second sample as an example, the passage retrieved by DPR is around Ordovician in the question but there are no more details answering the question. In comparison, MVR mines more fine-grained information in the passage and return the correct answer ${485.4} \pm  {1.9}\mathrm{{Ma}}$ (Ma means million years ago). It indicates that DPR can only capture the rough meaning of a passage from a general view, while our MVR is able to dive into the passage and capture more fine-grained semantic information from multiple views.

如表7所示，有两个由DPR和MVR检索到的示例用于定性分析。DPR检索到的得分最高的段落虽然似乎与查询有相似的主题，但无法为查询提供明确的答案。相比之下，当段落包含丰富的信息和多样的语义时，我们的MVR能够返回正确的答案。以第二个样本为例，DPR检索到的段落围绕问题中的奥陶纪（Ordovician）展开，但没有更多回答问题的细节。相比之下，MVR在段落中挖掘出更细粒度的信息，并返回正确答案${485.4} \pm  {1.9}\mathrm{{Ma}}$（Ma表示百万年前）。这表明DPR只能从总体上捕捉段落的大致含义，而我们的MVR能够深入段落，从多个视角捕捉更细粒度的语义信息。

---

<!-- Footnote -->

1www.nltk.org

1www.nltk.org

<!-- Footnote -->

---

<!-- Media -->

<table><tr><td>Question</td><td>Passage received by DPR</td><td>Passage retrieved by MVR</td></tr><tr><td>What continent ranged over the majority of the southern hemisphere of earth in the Cambrian?</td><td>Title: Mesozoic ... and the Khingan Mountains in Manchuria. This orogeny was related to the opening of the Arctic Ocean and subduction of the North China and Siberian cratons under the Pacific Ocean. In contrast, the era featured the dramatic rifting of the supercontinent Pangaea, which gradually split into a northern continent, Laurasia, and a southern continent, Gondwana. This created the passive continental margin that characterizes most of the Atlantic coastline (such as along the U.S. East Coast) today. By the end of the era, the continents ...</td><td>Title: Geological history of Earth ... Laurentia, Baltica and Siberia remained independent con- tinents following the break-up of the supercontinent of Pan- notia. Gondwana started to drift toward the South Pole. Pan- thalassa covered most of the southern hemisphere, and minor oceans included the Proto-Tethys Ocean, Iapetus Ocean and Khanty Ocean. The Ordovician period started at a major extinction event called the Cambrian -Ordovician extinction event some time about ${485.4} \pm  {1.9}\mathrm{{Ma}}$ . During the Ordovician the southern continents were collected into a single continent called Gondwana. Gondwana started the period in ...</td></tr><tr><td>How long ago did the Ordovician period begin?</td><td>Title: Ordovician ... is a geologic period and system, the second of six periods of the Paleozoic Era. The Ordovician spans 41.2 million years from the end of the Cambrian Period million years ago (Mya) to the start of the Silurian Period Mya. The Ordovician, named after the Celtic tribe of the Ordovices, was defined by Charles Lapworth in 1879 to resolve a dispute between followers of Adam Sedgwick and Roderick Murchison, who were placing the same rock beds in northern Wales into the Cambrian and Silurian systems, respectively. . . .</td><td>Title: Geological history of Earth ... Laurentia, Baltica and Siberia remained independent conti- nents following the break-up of the supercontinent of Pannotia. Gondwana started to drift toward the South Pole. Panthalassa covered most of the southern hemisphere, and minor oceans included the Proto-Tethys Ocean, Iapetus Ocean and Khanty Ocean. The Ordovician period started at a major extinction event called the Cambrian-Ordovician extinction event some time about $\mathbf{{485.4} \pm  {1.9}}\mathrm{{Ma}}$ . During the Ordovician the south- ern continents were collected into a single continent called Gondwana. Gondwana started the period in ...</td></tr></table>

<table><tbody><tr><td>问题</td><td>文档处理模块（DPR）接收的段落</td><td>多向量检索器（MVR）检索到的段落</td></tr><tr><td>在寒武纪时期，哪个大陆覆盖了地球南半球的大部分地区？</td><td>标题：中生代……以及满洲的兴安岭。这次造山运动与北冰洋的开启以及华北和西伯利亚克拉通在太平洋下的俯冲有关。相比之下，这一时期的显著特征是超大陆盘古大陆（Pangaea）的剧烈裂解，它逐渐分裂成北方大陆劳亚大陆（Laurasia）和南方大陆冈瓦纳大陆（Gondwana）。这造就了如今大西洋大部分海岸线（如美国东海岸沿线）所具有的被动大陆边缘。到这个时期结束时，各大陆……</td><td>标题：地球地质历史……在超大陆潘诺西亚大陆（Pannotia）解体后，劳伦西亚大陆（Laurentia）、波罗的大陆（Baltica）和西伯利亚大陆（Siberia）仍然是独立的大陆。冈瓦纳大陆（Gondwana）开始向南极漂移。泛大洋（Panthalassa）覆盖了南半球的大部分地区，较小的海洋包括原特提斯洋（Proto - Tethys Ocean）、伊阿珀托斯洋（Iapetus Ocean）和汉特洋（Khanty Ocean）。奥陶纪始于约${485.4} \pm  {1.9}\mathrm{{Ma}}$发生的一次重大灭绝事件，即寒武纪 - 奥陶纪灭绝事件。在奥陶纪时期，南方的各个大陆合并成了一个名为冈瓦纳大陆（Gondwana）的单一大陆。冈瓦纳大陆在这个时期开始时……</td></tr><tr><td>奥陶纪始于多久之前？</td><td>标题：奥陶纪……是一个地质时期和地层系统，是古生代六个时期中的第二个。奥陶纪从寒武纪结束（百万年前，Mya）开始，到志留纪开始（百万年前，Mya）结束，跨度为4120万年。奥陶纪以凯尔特部落奥多维塞人（Ordovices）命名，由查尔斯·拉普沃思（Charles Lapworth）于1879年定义，旨在解决亚当·塞奇威克（Adam Sedgwick）和罗德里克·默奇森（Roderick Murchison）的追随者之间的争议，他们分别将威尔士北部的同一岩层归入寒武纪和志留纪地层系统。……</td><td>标题：地球地质历史……在超大陆潘诺西亚大陆（Pannotia）解体后，劳伦西亚大陆（Laurentia）、波罗的大陆（Baltica）和西伯利亚大陆（Siberia）仍然是独立的大陆。冈瓦纳大陆（Gondwana）开始向南极漂移。泛大洋（Panthalassa）覆盖了南半球的大部分地区，较小的海洋包括原特提斯洋（Proto - Tethys Ocean）、伊阿珀托斯洋（Iapetus Ocean）和汉特洋（Khanty Ocean）。奥陶纪始于约$\mathbf{{485.4} \pm  {1.9}}\mathrm{{Ma}}$发生的一次重大灭绝事件，即寒武纪 - 奥陶纪灭绝事件。在奥陶纪时期，南方的各个大陆合并成了一个名为冈瓦纳大陆（Gondwana）的单一大陆。冈瓦纳大陆在这个时期开始时……</td></tr></tbody></table>

Table 7: Examples of passages returned from DPR and MVR. Correct answers are written in bold.

表7：从DPR（密集段落检索器）和MVR（多视图检索器）返回的段落示例。正确答案用粗体显示。

<!-- Media -->

## 6 Conclusions

## 6 结论

In this paper, we propose a novel Multi-View Representation Learning framework. Specifically, we present a simple yet effective method to generate multi-view document representations through multiple viewers. To optimize the training of multiple viewers, we propose a global-local loss with annealed temperature to enable multiple viewers to better align with different semantic views. We conduct experiments on three open-domain retrieval datasets, and achieve state-of-the-art retrieval performance. Our further analysis proves the effectiveness of different components in our method.

在本文中，我们提出了一种新颖的多视图表示学习框架。具体而言，我们提出了一种简单而有效的方法，通过多个视图生成器来生成多视图文档表示。为了优化多个视图生成器的训练，我们提出了一种带有退火温度的全局 - 局部损失函数，使多个视图生成器能够更好地与不同的语义视图对齐。我们在三个开放域检索数据集上进行了实验，并取得了最先进的检索性能。我们的进一步分析证明了我们方法中不同组件的有效性。

## 7 Acknowledgements

## 7 致谢

We thank Yuan Chai, Junhe Zhao, Yimin Fan, Jun-jie Huang and Hang Zhang for their discussions and suggestions on writing this paper.

我们感谢柴源、赵俊和、范逸民、黄俊杰和张航在撰写本文过程中的讨论和建议。

## References

## 参考文献

Alexandr Andoni, Piotr Indyk, and Ilya Razenshteyn. 2018. Approximate nearest neighbor search in high dimensions. In Proceedings of the International Congress of Mathematicians: Rio de Janeiro 2018, pages 3287-3318. World Scientific.

亚历山大·安多尼（Alexandr Andoni）、彼得·因迪克（Piotr Indyk）和伊利亚·拉赞施泰因（Ilya Razenshteyn）。2018年。高维近似最近邻搜索。见《国际数学家大会会议录：2018年里约热内卢》，第3287 - 3318页。世界科学出版社。

Peter F. Brown, Stephen A. Della Pietra, Vincent J. Della Pietra, Jennifer C. Lai, and Robert L. Mercer. 1992. An estimate of an upper bound for the entropy of English. Computational Linguistics, 18(1):31-40.

彼得·F·布朗（Peter F. Brown）、斯蒂芬·A·德拉·皮埃特拉（Stephen A. Della Pietra）、文森特·J·德拉·皮埃特拉（Vincent J. Della Pietra）、詹妮弗·C·赖（Jennifer C. Lai）和罗伯特·L·默瑟（Robert L. Mercer）。1992年。英语熵上限的估计。《计算语言学》，18(1):31 - 40。

Wei-Cheng Chang, Felix X. Yu, Yin-Wen Chang, Yim-ing Yang, and Sanjiv Kumar. 2020. Pre-training tasks for embedding-based large-scale retrieval. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26- 30, 2020. OpenReview.net.

张维正（Wei - Cheng Chang）、余飞立克斯·X（Felix X. Yu）、张殷文（Yin - Wen Chang）、杨一鸣（Yim - ing Yang）和桑吉夫·库马尔（Sanjiv Kumar）。2020年。基于嵌入的大规模检索的预训练任务。见第8届国际学习表征会议（ICLR 2020），2020年4月26 - 30日，埃塞俄比亚亚的斯亚贝巴。OpenReview.net。

Jiecao Chen, Liu Yang, Karthik Raman, Michael Ben-dersky, Jung-Jung Yeh, Yun Zhou, Marc Najork, Danyang Cai, and Ehsan Emadzadeh. 2020. DiPair: Fast and accurate distillation for trillion-scale text matching and pair modeling. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 2925-2937, Online. Association for Computational Linguistics.

陈杰草（Jiecao Chen）、杨柳（Liu Yang）、卡尔蒂克·拉曼（Karthik Raman）、迈克尔·本德尔斯基（Michael Ben - dersky）、叶荣荣（Jung - Jung Yeh）、周云（Yun Zhou）、马克·纳约克（Marc Najork）、蔡丹阳（Danyang Cai）和埃桑·埃马扎德（Ehsan Emadzadeh）。2020年。DiPair：用于万亿级文本匹配和配对建模的快速准确蒸馏方法。见《计算语言学协会研究成果：EMNLP 2020》，第2925 - 2937页，线上会议。计算语言学协会。

Kevin Clark, Urvashi Khandelwal, Omer Levy, and Christopher D. Manning. 2019. What does BERT look at? an analysis of BERT's attention. In Proceedings of the 2019 ACL Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, pages 276-286, Florence, Italy. Association for Computational Linguistics.

凯文·克拉克（Kevin Clark）、乌尔瓦希·坎德尔瓦尔（Urvashi Khandelwal）、奥默·利维（Omer Levy）和克里斯托弗·D·曼宁（Christopher D. Manning）。2019年。BERT在关注什么？对BERT注意力机制的分析。见《2019年ACL黑盒NLP研讨会会议录：分析和解释用于NLP的神经网络》，第276 - 286页，意大利佛罗伦萨。计算语言学协会。

Zhuyun Dai and Jamie Callan. 2019. Deeper text understanding for ir with contextual neural language modeling. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 985-988.

戴竹云（Zhuyun Dai）和杰米·卡兰（Jamie Callan）。2019年。通过上下文神经语言建模实现信息检索中的更深入文本理解。见第42届国际ACM SIGIR信息检索研究与发展会议会议录，第985 - 988页。

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171-4186, Minneapolis, Minnesota. Association for Computational Linguistics.

雅各布·德夫林（Jacob Devlin）、张明伟（Ming - Wei Chang）、肯顿·李（Kenton Lee）和克里斯蒂娜·图托纳娃（Kristina Toutanova）。2019年。BERT：用于语言理解的深度双向Transformer预训练。见《2019年北美计算语言学协会人类语言技术会议会议录》，第1卷（长论文和短论文），第4171 - 4186页，美国明尼苏达州明尼阿波利斯。计算语言学协会。

Luyu Gao and Jamie Callan. 2021a. Condenser: a pretraining architecture for dense retrieval. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 981-993.

高璐宇（Luyu Gao）和杰米·卡兰（Jamie Callan）。2021a。冷凝器（Condenser）：一种用于密集检索的预训练架构。见《2021年自然语言处理经验方法会议会议录》，第981 - 993页。

Luyu Gao and Jamie Callan. 2021b. Unsupervised corpus aware language model pre-training for dense passage retrieval. arXiv preprint arXiv:2108.05540.

高璐宇（Luyu Gao）和杰米·卡兰（Jamie Callan）。2021b。用于密集段落检索的无监督语料感知语言模型预训练。arXiv预印本arXiv:2108.05540。

Luyu Gao, Zhuyun Dai, and Jamie Callan. 2020. Modularized transfomer-based ranking framework. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020, pages 4180- 4190. Association for Computational Linguistics.

高璐宇（Luyu Gao）、戴竹云（Zhuyun Dai）和杰米·卡兰（Jamie Callan）。2020年。基于模块化Transformer的排序框架。见《2020年自然语言处理经验方法会议（EMNLP 2020）会议录》，线上会议，2020年11月16 - 20日，第4180 - 4190页。计算语言学协会。

Luyu Gao, Zhuyun Dai, and Jamie Callan. 2021. COIL: Revisit exact lexical match in information retrieval with contextualized inverted list. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3030-3042, Online. Association for Computational Linguistics.

高璐宇、戴珠云（音译）和杰米·卡兰（Jamie Callan）。2021年。COIL：利用上下文倒排列表重新审视信息检索中的精确词汇匹配。见《2021年计算语言学协会北美分会人类语言技术会议论文集》，第3030 - 3042页，线上会议。计算语言学协会。

Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasu-pat, and Mingwei Chang. 2020. Retrieval augmented language model pre-training. In International Conference on Machine Learning, pages 3929-3938. PMLR.

凯文·古（Kelvin Guu）、肯顿·李（Kenton Lee）、佐拉·董（Zora Tung）、帕努蓬·帕苏帕特（Panupong Pasu - pat）和张明伟（Mingwei Chang）。2020年。检索增强语言模型预训练。见《国际机器学习会议》，第3929 - 3938页。机器学习研究会议论文集（PMLR）。

Po-Sen Huang, Xiaodong He, Jianfeng Gao, Li Deng, Alex Acero, and Larry Heck. 2013. Learning deep structured semantic models for web search using clickthrough data. In Proceedings of the 22nd ACM international conference on Information & Knowledge Management, pages 2333-2338.

黄伯森（Po - Sen Huang）、何晓东、高剑锋、邓力、亚历克斯·阿塞罗（Alex Acero）和拉里·赫克（Larry Heck）。2013年。利用点击数据学习用于网络搜索的深度结构化语义模型。见《第22届ACM国际信息与知识管理会议论文集》，第2333 - 2338页。

Samuel Humeau, Kurt Shuster, Marie-Anne Lachaux, and Jason Weston. 2020. Poly-encoders: Architectures and pre-training strategies for fast and accurate multi-sentence scoring. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net.

塞缪尔·休莫（Samuel Humeau）、库尔特·舒斯特（Kurt Shuster）、玛丽 - 安妮·拉肖（Marie - Anne Lachaux）和杰森·韦斯顿（Jason Weston）。2020年。多编码器：用于快速准确多句子评分的架构和预训练策略。见《第8届国际学习表征会议（ICLR 2020）》，2020年4月26 - 30日，埃塞俄比亚亚的斯亚贝巴。OpenReview.net。

Gautier Izacard and Edouard Grave. 2021. Distilling knowledge from reader to retriever for question answering. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net.

高蒂尔·伊扎卡尔（Gautier Izacard）和爱德华·格雷夫（Edouard Grave）。2021年。在问答任务中从阅读器向检索器蒸馏知识。见《第9届国际学习表征会议（ICLR 2021）》，2021年5月3 - 7日，奥地利线上会议。OpenReview.net。

Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-scale similarity search with gpus. IEEE Transactions on Big Data.

杰夫·约翰逊（Jeff Johnson）、马蒂亚斯·杜泽（Matthijs Douze）和埃尔韦·热古（Hervé Jégou）。2019年。基于GPU的十亿级相似度搜索。《IEEE大数据汇刊》。

Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. 2017. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1601-1611, Vancouver, Canada. Association for Computational Linguistics.

曼达尔·乔希（Mandar Joshi）、崔恩索尔（Eunsol Choi）、丹尼尔·韦尔德（Daniel Weld）和卢克·泽特尔莫尔（Luke Zettlemoyer）。2017年。TriviaQA：一个用于阅读理解的大规模远程监督挑战数据集。见《计算语言学协会第55届年会论文集（第1卷：长论文）》，第1601 - 1611页，加拿大温哥华。计算语言学协会。

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick S. H. Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020, pages 6769-6781. Association for Computational Linguistics.

弗拉基米尔·卡尔普欣（Vladimir Karpukhin）、巴拉斯·奥古兹（Barlas Oguz）、闵世文（Sewon Min）、帕特里克·S. H. 刘易斯（Patrick S. H. Lewis）、莱德尔·吴（Ledell Wu）、谢尔盖·叶杜诺夫（Sergey Edunov）、陈丹琦和易文涛。2020年。用于开放域问答的密集段落检索。见《2020年自然语言处理经验方法会议（EMNLP 2020）论文集》，2020年11月16 - 20日，线上会议，第6769 - 6781页。计算语言学协会。

Omar Khattab and Matei Zaharia. 2020. Colbert: Efficient and effective passage search via contextual-ized late interaction over BERT. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, SIGIR 2020, Virtual Event, China, July 25-30, 2020, pages 39-48. ACM.

奥马尔·哈塔卜（Omar Khattab）和马特·扎哈里亚（Matei Zaharia）。2020年。Colbert：通过基于BERT的上下文后期交互实现高效有效的段落搜索。见《第43届ACM SIGIR国际信息检索研究与发展会议（SIGIR 2020）论文集》，2020年7月25 - 30日，中国线上会议，第39 - 48页。美国计算机协会（ACM）。

Olga Kovaleva, Alexey Romanov, Anna Rogers, and Anna Rumshisky. 2019. Revealing the dark secrets of bert. arXiv preprint arXiv:1908.08593.

奥尔加·科瓦列娃（Olga Kovaleva）、阿列克谢·罗曼诺夫（Alexey Romanov）、安娜·罗杰斯（Anna Rogers）和安娜·鲁姆希斯基（Anna Rumshisky）。2019年。揭示BERT的隐秘奥秘。arXiv预印本arXiv:1908.08593。

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural questions: A benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:452-466.

汤姆·夸特科夫斯基（Tom Kwiatkowski）、珍妮玛丽亚·帕洛马基（Jennimaria Palomaki）、奥利维亚·雷德菲尔德（Olivia Redfield）、迈克尔·柯林斯（Michael Collins）、安库尔·帕里克（Ankur Parikh）、克里斯·阿尔贝蒂（Chris Alberti）、丹妮尔·爱泼斯坦（Danielle Epstein）、伊利亚·波洛苏金（Illia Polosukhin）、雅各布·德夫林（Jacob Devlin）、肯顿·李（Kenton Lee）、克里斯蒂娜·图托纳娃（Kristina Toutanova）、利翁·琼斯（Llion Jones）、马修·凯尔西（Matthew Kelcey）、张明伟（Ming - Wei Chang）、安德鲁·M. 戴（Andrew M. Dai）、雅各布·乌兹科雷特（Jakob Uszkoreit）、勒·奎克（Quoc Le）和斯拉夫·彼得罗夫（Slav Petrov）。2019年。自然问题：问答研究的基准。《计算语言学协会汇刊》，7：452 - 466。

Kenton Lee, Ming-Wei Chang, and Kristina Toutanova. 2019. Latent retrieval for weakly supervised open domain question answering. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 6086-6096, Florence, Italy. Association for Computational Linguistics.

肯顿·李（Kenton Lee）、张明伟（Ming - Wei Chang）和克里斯蒂娜·图托纳娃（Kristina Toutanova）。2019年。用于弱监督开放域问答的潜在检索。见《计算语言学协会第57届年会论文集》，第6086 - 6096页，意大利佛罗伦萨。计算语言学协会。

Yizhi Li, Zhenghao Liu, Chenyan Xiong, and Zhiyuan Liu. 2021. More robust dense retrieval with contrastive dual learning. In Proceedings of the 2021 ACM SIGIR International Conference on Theory of Information Retrieval, pages 287-296.

李一之、刘正浩、熊晨彦和刘志远。2021年。通过对比对偶学习实现更鲁棒的密集检索。见《2021年ACM SIGIR国际信息检索理论会议论文集》，第287 - 296页。

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized BERT pretraining approach. CoRR, abs/1907.11692.

刘音汉、迈尔·奥特、纳曼·戈亚尔、杜静飞、曼达尔·乔希、陈丹琦、奥默·利维、迈克·刘易斯、卢克·泽特尔莫耶尔和韦塞林·斯托亚诺夫。2019年。RoBERTa：一种对BERT预训练方法的强力优化。计算机研究报告库（CoRR），编号：abs/1907.11692。

Yi Luan, Jacob Eisenstein, Kristina Toutanova, and Michael Collins. 2021. Sparse, dense, and attentional representations for text retrieval. Transactions of the Association for Computational Linguistics, 9:329- 345.

栾义、雅各布·艾森斯坦、克里斯蒂娜·图托纳娃和迈克尔·柯林斯。2021年。用于文本检索的稀疏、密集和注意力表示。《计算语言学协会汇刊》，第9卷：329 - 345页。

Shikib Mehri and Mihail Eric. 2021. Example-driven intent prediction with observers. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2979-2992, Online. Association for Computational Linguistics.

希基布·梅赫里和米哈伊尔·埃里克。2021年。基于观察者的示例驱动意图预测。《2021年北美计算语言学协会人类语言技术会议论文集》，第2979 - 2992页，线上会议。计算语言学协会。

Rodrigo Nogueira and Kyunghyun Cho. 2019. Passage re-ranking with bert. arXiv preprint arXiv:1901.04085.

罗德里戈·诺盖拉和赵京焕。2019年。使用BERT进行段落重排序。预印本论文库（arXiv）预印本，编号：arXiv:1901.04085。

Barlas Oğuz, Kushal Lakhotia, Anchit Gupta, Patrick Lewis, Vladimir Karpukhin, Aleksandra Piktus, Xilun Chen, Sebastian Riedel, Wen-tau Yih, Sonal Gupta, et al. 2021. Domain-matched pretraining tasks for dense retrieval. arXiv preprint arXiv:2107.13602.

巴拉斯·奥古兹、库沙尔·拉科蒂亚、安奇特·古普塔、帕特里克·刘易斯、弗拉基米尔·卡尔普欣、亚历山德拉·皮克图斯、陈希伦、塞巴斯蒂安·里德尔、易文涛、索纳尔·古普塔等。2021年。用于密集检索的领域匹配预训练任务。预印本论文库（arXiv）预印本，编号：arXiv:2107.13602。

Yingqi Qu, Yuchen Ding, Jing Liu, Kai Liu, Ruiyang Ren, Wayne Xin Zhao, Daxiang Dong, Hua Wu, and Haifeng Wang. 2021. RocketQA: An optimized training approach to dense passage retrieval for open-domain question answering. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 5835-5847, Online. Association for Computational Linguistics.

曲英琦、丁雨晨、刘静、刘凯、任瑞阳、赵鑫、董大祥、吴华和王海峰。2021年。RocketQA：一种用于开放域问答的密集段落检索的优化训练方法。《2021年北美计算语言学协会人类语言技术会议论文集》，第5835 - 5847页，线上会议。计算语言学协会。

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. SQuAD: 100,000+ questions for machine comprehension of text. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 2383-2392, Austin, Texas. Association for Computational Linguistics.

普拉纳夫·拉朱帕尔卡、张健、康斯坦丁·洛皮列夫和珀西·梁。2016年。SQuAD：用于文本机器理解的10万多个问题。《2016年自然语言处理经验方法会议论文集》，第2383 - 2392页，美国得克萨斯州奥斯汀市。计算语言学协会。

Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982-3992, Hong Kong, China. Association for Computational Linguistics.

尼尔斯·赖默斯和伊琳娜·古列维奇。2019年。Sentence - BERT：使用孪生BERT网络的句子嵌入。《2019年自然语言处理经验方法会议和第9届自然语言处理国际联合会议（EMNLP - IJCNLP）论文集》，第3982 - 3992页，中国香港。计算语言学协会。

Ruiyang Ren, Shangwen Lv, Yingqi Qu, Jing Liu, Wayne Xin Zhao, QiaoQiao She, Hua Wu, Haifeng Wang, and Ji-Rong Wen. 2021a. PAIR: Leveraging passage-centric similarity relation for improving dense passage retrieval. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 2173-2183, Online. Association for Computational Linguistics.

任瑞阳、吕尚文、曲英琦、刘静、赵鑫、佘巧巧、吴华、王海峰和文继荣。2021a。PAIR：利用以段落为中心的相似关系改进密集段落检索。《计算语言学协会研究成果：ACL - IJCNLP 2021》，第2173 - 2183页，线上会议。计算语言学协会。

Ruiyang Ren, Yingqi Qu, Jing Liu, Wayne Xin Zhao, Qiaoqiao She, Hua Wu, Haifeng Wang, and Ji-Rong Wen. 2021b. Rocketqav2: A joint training method for dense passage retrieval and passage re-ranking. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 2825-2835.

任瑞阳、曲英琦、刘静、赵鑫、佘巧巧、吴华、王海峰和文继荣。2021b。RocketQAv2：一种用于密集段落检索和段落重排序的联合训练方法。《2021年自然语言处理经验方法会议论文集》，第2825 - 2835页。

Devendra Sachan, Mostofa Patwary, Mohammad Shoeybi, Neel Kant, Wei Ping, William L. Hamilton, and Bryan Catanzaro. 2021. End-to-end training of neural retrievers for open-domain question answering. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 6648-6662, Online. Association for Computational Linguistics.

德文德拉·萨坎、莫斯托法·帕特瓦里、穆罕默德·肖伊比、尼尔·康德、魏平、威廉·L·汉密尔顿和布莱恩·卡坦扎罗。2021年。用于开放域问答的神经检索器的端到端训练。《第59届计算语言学协会年会和第11届自然语言处理国际联合会议论文集（第1卷：长论文）》，第6648 - 6662页，线上会议。计算语言学协会。

Hongyin Tang, Xingwu Sun, Beihong Jin, Jingang Wang, Fuzheng Zhang, and Wei Wu. 2021. Improving document representations by generating pseudo query embeddings for dense retrieval. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 5054-5064, Online. Association for Computational Linguistics.

唐红印、孙兴武、金北宏、王金刚、张福正和吴伟。2021年。通过生成伪查询嵌入改进密集检索的文档表示。《第59届计算语言学协会年会和第11届自然语言处理国际联合会议论文集（第1卷：长论文）》，第5054 - 5064页，线上会议。计算语言学协会。

Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed, and Arnold Overwijk. 2020. Approximate nearest neighbor negative contrastive learning for dense text retrieval. CoRR, abs/2007.00808.

李雄（Lee Xiong）、熊晨燕（Chenyan Xiong）、李烨（Ye Li）、邓国峰（Kwok-Fung Tang）、刘佳琳（Jialin Liu）、保罗·贝内特（Paul Bennett）、朱奈德·艾哈迈德（Junaid Ahmed）和阿诺德·奥弗维克（Arnold Overwijk）。2020年。用于密集文本检索的近似最近邻负对比学习。计算机研究报告（CoRR），编号：abs/2007.00808。

Peilin Yang, Hui Fang, and Jimmy Lin. 2017. Anserini: Enabling the use of lucene for information retrieval research. In Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 1253-1256.

杨佩琳（Peilin Yang）、方慧（Hui Fang）和吉米·林（Jimmy Lin）。2017年。安瑟里尼（Anserini）：支持在信息检索研究中使用Lucene。见《第40届ACM国际信息检索研究与发展会议论文集》，第1253 - 1256页。

Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Jiafeng Guo, Min Zhang, and Shaoping Ma. 2021. Optimizing dense retrieval model training with hard negatives. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 1503-1512.

詹景涛（Jingtao Zhan）、毛佳欣（Jiaxin Mao）、刘奕群（Yiqun Liu）、郭佳峰（Jiafeng Guo）、张敏（Min Zhang）和马少平（Shaoping Ma）。2021年。使用难负样本优化密集检索模型训练。见《第44届ACM国际信息检索研究与发展会议论文集》，第1503 - 1512页。

Hang Zhang, Yeyun Gong, Yelong Shen, Jiancheng Lv, Nan Duan, and Weizhu Chen. 2021. Adversarial retriever-ranker for dense text retrieval. arXiv preprint arXiv:2110.03611.

张航（Hang Zhang）、龚叶云（Yeyun Gong）、沈业龙（Yelong Shen）、吕建成（Jiancheng Lv）、段楠（Nan Duan）和陈伟柱（Weizhu Chen）。2021年。用于密集文本检索的对抗式检索器 - 排序器。预印本arXiv：2110.03611。

## A Scale Factor of Global-Local Loss

## 全局 - 局部损失的缩放因子

We have tuned the scale factor $\lambda$ of the Global-local loss in Eq.5. The performances on SQuAD dev set are shown in Table 8. We find that a suitable scaling factor $\left( {\lambda  = {0.01}}\right)$ can improve more than others. Analysing other results, we infer that a large factor of local uniformity loss may lead to much impact on optimization of global loss, while a smaller one will degenerate into the form without local uniformity loss.

我们对公式5中全局 - 局部损失的缩放因子$\lambda$进行了调整。表8展示了在斯坦福问答数据集（SQuAD）开发集上的性能表现。我们发现，合适的缩放因子$\left( {\lambda  = {0.01}}\right)$比其他因子能带来更多的性能提升。通过分析其他结果，我们推断局部一致性损失的因子较大时可能会对全局损失的优化产生较大影响，而较小时则会退化为没有局部一致性损失的形式。

<!-- Media -->

<table><tr><td>$\lambda$</td><td>R@5</td><td>R@20</td><td>R@100</td></tr><tr><td>0.5</td><td>72.4</td><td>80.4</td><td>85.9</td></tr><tr><td>0.05</td><td>74.7</td><td>82.5</td><td>87.3</td></tr><tr><td>0.01</td><td>75.5</td><td>83.2</td><td>87.9</td></tr><tr><td>0.001</td><td>72.9</td><td>82.2</td><td>85.7</td></tr></table>

<table><tr><td>$\lambda$</td><td>R@5</td><td>R@20</td><td>R@100</td></tr><tr><td>0.5</td><td>72.4</td><td>80.4</td><td>85.9</td></tr><tr><td>0.05</td><td>74.7</td><td>82.5</td><td>87.3</td></tr><tr><td>0.01</td><td>75.5</td><td>83.2</td><td>87.9</td></tr><tr><td>0.001</td><td>72.9</td><td>82.2</td><td>85.7</td></tr></table>

Table 8: Performance on SQuAD dev set under different setting of scale factor.

表8：在不同缩放因子设置下，模型在斯坦福问答数据集（SQuAD）开发集上的性能表现。

<!-- Media -->