# XTR meets ColBERTv2: Adding ColBERTv2 Optimizations to XTR

# XTR与ColBERTv2相遇：将ColBERTv2优化方法应用于XTR

Riyaz Ahmed Bhat and Jaydeep Sen

里亚兹·艾哈迈德·巴特（Riyaz Ahmed Bhat）和杰迪普·森（Jaydeep Sen）

IBM Research, India

印度IBM研究院

riyaz.bhat@ibm.com, jaydesen@in.ibm.com

riyaz.bhat@ibm.com，jaydesen@in.ibm.com

## Abstract

## 摘要

XTR (Lee et al., 2023) introduced an efficient multi-vector retrieval method that addresses the limitations of the ColBERT (Khat-tab and Zaharia, 2020) model by simplifying retrieval into a single stage through a modified learning objective. While XTR eliminates the need for multistage retrieval, it doesn't incorporate the efficiency optimizations from Col-BERTv2 (Santhanam et al., 2022), which improve indexing and retrieval speed. In this work, we enhance XTR by integrating Col-BERTv2's optimizations, showing that the combined approach preserves the strengths of both models. This results in a more efficient and scalable solution for multi-vector retrieval, while maintaining XTR's streamlined retrieval process. We have released the code as an addition to the PrimeQA (PrimeQA, 2023) toolkit.

XTR（李等人，2023年）引入了一种高效的多向量检索方法，该方法通过修改学习目标将检索简化为单阶段，从而解决了ColBERT（卡塔布和扎哈里亚，2020年）模型的局限性。虽然XTR消除了多阶段检索的需求，但它并未纳入Col - BERTv2（桑塔纳姆等人，2022年）的效率优化方法，这些优化方法可提高索引和检索速度。在这项工作中，我们通过集成Col - BERTv2的优化方法来增强XTR，结果表明这种组合方法保留了两个模型的优势。这为多向量检索带来了更高效、可扩展的解决方案，同时保持了XTR简化的检索过程。我们已将代码作为PrimeQA（PrimeQA，2023年）工具包的补充发布。

## 1 Introduction

## 1 引言

Retrieval refers to the task of retrieving relevant documents from a larger corpus of documents, given a search query. Retrieval is one of the most active research fields in NLP owing to its many applications such as semantic search (Fazz-inga and Lukasiewicz, 2010), Open-domain Question Answering (Voorhees and Tice, 2000; Chen and Yih, 2020), Retrieval Augmented Generation (RAG) (Cai et al., 2019; Lewis et al., 2020; Guu et al., 2020). Research in Retrieval technologies has been evolving through multiple paradigms which can broadly be divided into (1) Sparse Retrievers (Robertson and Zaragoza, 2009) (2) Dense Retrievers (Karpukhin et al., 2020; Chang et al., 2019; Guu et al., 2020; Xu et al., 2022; Khattab and Zaharia, 2020; Luan et al., 2021; Santhanam et al., 2022) and very recently (3) Differential Search Index based retrievers (Tay et al., 2022). Each paradigm of retrievers has its advantages and disadvantages stemming from the methodology adopted and the limitations in those. Therefore, in practical applications, we have seen hybrid approaches that employ different kinds of retrievers to build a robust pipeline for accurate retrieval.

检索是指在给定搜索查询的情况下，从更大的文档语料库中检索相关文档的任务。由于检索在语义搜索（法津加和卢卡西维茨，2010年）、开放域问答（沃里斯和泰斯，2000年；陈和易，2020年）、检索增强生成（RAG）（蔡等人，2019年；刘易斯等人，2020年；古等人，2020年）等诸多应用，它是自然语言处理（NLP）中最活跃的研究领域之一。检索技术的研究经历了多种范式的演变，大致可分为（1）稀疏检索器（罗伯逊和萨拉戈萨，2009年）（2）密集检索器（卡尔普欣等人，2020年；张等人，2019年；古等人，2020年；徐等人，2022年；卡塔布和扎哈里亚，2020年；栾等人，2021年；桑塔纳姆等人，2022年），以及最近出现的（3）基于差分搜索索引的检索器（泰等人，2022年）。每种检索器范式都因其采用的方法和存在的局限性而各有优缺点。因此，在实际应用中，我们看到了采用不同类型检索器构建强大准确检索流程的混合方法。

Sparse retrievers rely on lexical overlap to retrieve relevant documents. They largely follow bag-of-words based similarity notions to score the documents using TF-IDF score. The most popular sparse retriever called BM25 (Robertson and Zaragoza, 2009) introduces robustness in using tf-idf scores for scoring documents. Sparse retrievers often employ an inverted index for word search which is very fast and easy to maintain. Although sparse retrievers are easy to use and interpretable, their accuracy is mainly limited by the need for relevant keyword overlaps for accurate document retrieval. Dense retrievers (Karpukhin et al., 2020; Chang et al., 2019; Guu et al., 2020; Xu et al., 2022; Khattab and Zaharia, 2020; Luan et al., 2021; San-thanam et al., 2022) try to address this problem by using neural models to encode words into an embedding space. Dense retrievers compute semantic similarity in the embedding space, where two different words if semantically similar, should have their embedding vectors close to each other and hence would produce a good similarity match.

稀疏检索器依靠词汇重叠来检索相关文档。它们主要遵循基于词袋的相似性概念，使用词频 - 逆文档频率（TF - IDF）分数对文档进行评分。最流行的稀疏检索器BM25（罗伯逊和萨拉戈萨，2009年）在使用TF - IDF分数对文档进行评分时引入了鲁棒性。稀疏检索器通常采用倒排索引进行单词搜索，这种方法速度快且易于维护。虽然稀疏检索器易于使用且具有可解释性，但它们的准确性主要受限于准确文档检索所需的相关关键词重叠。密集检索器（卡尔普欣等人，2020年；张等人，2019年；古等人，2020年；徐等人，2022年；卡塔布和扎哈里亚，2020年；栾等人，2021年；桑塔纳姆等人，2022年）试图通过使用神经模型将单词编码到嵌入空间来解决这个问题。密集检索器在嵌入空间中计算语义相似度，在这个空间中，如果两个不同的单词语义相似，它们的嵌入向量应该彼此接近，从而产生良好的相似度匹配。

Multi-vector retrievers like ColBERT (Khattab and Zaharia, 2020) are more effective than single-vector models like DPR (Karpukhin et al., 2020) because they can capture finer semantic details between queries and documents. This makes them better suited for handling complex queries and retrieving more relevant results, while single-vector models tend to miss subtle nuances due to their limited representational capacity. However, index management for multi-vector retrievers is resource-intensive and demands specialized techniques to reduce memory footprint. ColBERTv2 (Santhanam et al., 2022) tackles this challenge by implementing strategies such as token representation compression, an aggressive residual compression mechanism, and a denoised supervision approach, which 358 help reduce the memory footprint without compromising retrieval performance. Despite these optimizations, ColBERT still suffers from slow inference due to its multi-stage retrieval process, which involves token similarity computation, gathering, and reranking. To simplify this process, XTR (Lee et al., 2023) has recently proposed limiting ColBERT's retrieval to just the token similarity stage by modifying the training objective.

像ColBERT（卡塔布和扎哈里亚，2020年）这样的多向量检索器比像DPR（卡尔普欣等人，2020年）这样的单向量模型更有效，因为它们可以捕捉查询和文档之间更精细的语义细节。这使得它们更适合处理复杂查询并检索更相关的结果，而单向量模型由于其有限的表示能力往往会错过细微的差别。然而，多向量检索器的索引管理资源密集，需要专门的技术来减少内存占用。ColBERTv2（桑塔纳姆等人，2022年）通过实施诸如令牌表示压缩、激进的残差压缩机制和去噪监督方法等策略来应对这一挑战，这些方法有助于在不影响检索性能的情况下减少内存占用。尽管有这些优化，ColBERT由于其多阶段检索过程（包括令牌相似度计算、收集和重新排序）仍然存在推理速度慢的问题。为了简化这个过程，XTR（李等人，2023年）最近提出通过修改训练目标将ColBERT的检索限制在令牌相似度阶段。

The optimizations proposed by ColBERTv2 and XTR for multi-vector retrieval are complementary, and integrating them into a unified system can further improve retrieval performance. In this work, we propose exactly that, and propose ColXTR. We adopt the XTR training objective to train ColXTR, with some modifications. Unlike XTR, we introduce a projection layer to reduce the dimensionality of the encoded token vectors during training, thereby minimizing both query and index space costs. After training, we apply optimizations from ColBERTv2 and adapt XTR's inference which relies on missing token imputation to further enhance both indexing and retrieval efficiency.

ColBERTv2和XTR为多向量检索提出的优化方法是互补的，将它们集成到一个统一的系统中可以进一步提高检索性能。在这项工作中，我们正是提出了这样的方案，并提出了ColXTR。我们采用XTR训练目标来训练ColXTR，并进行了一些修改。与XTR不同的是，我们引入了一个投影层，在训练过程中降低编码后的词元向量的维度，从而将查询和索引空间成本降至最低。训练完成后，我们应用ColBERTv2的优化方法，并调整XTR依赖于缺失词元插补的推理过程，以进一步提高索引和检索效率。

Our contributions are as follows:

我们的贡献如下：

- We develop, ColXTR, a multi-vector retrieval model that integrates the strengths of both ColBERTv2 and XTR.

- 我们开发了ColXTR，这是一种集成了ColBERTv2和XTR优势的多向量检索模型。

- We empirically show that our novel compression techniques proposed on top of XTR reduce the index size by ${97}\%$ ,thus making it a lightweight system for practical usage.

- 我们通过实验证明，我们在XTR基础上提出的新型压缩技术将索引大小减少了${97}\%$，从而使其成为一个适用于实际应用的轻量级系统。

## 2 Related Work

## 2 相关工作

Classical IR models like BM25 (Robertson and Zaragoza, 2009) etc retrieve a ranked set of documents based on their lexical overlap with the query tokens. Due to its simplicity and strong performance on many domain-specific datasets, it is still considered as a strong baseline. With the popularity of neural language models like BERT (Devlin et al., 2018) etc, the use of such neural language models to obtain continuous representations for words (tokens) or documents has become quite popular. These neural language model based IR systems can be broadly classified into two categories a) single vector and b) multi-vector approaches.

像BM25（罗伯逊和萨拉戈萨，2009年）等经典信息检索（IR）模型会根据文档与查询词元的词法重叠情况，检索出一组排序后的文档。由于其简单性以及在许多特定领域数据集上的强大性能，它仍然被视为一个强大的基线模型。随着像BERT（德夫林等人，2018年）等神经语言模型的普及，使用此类神经语言模型来获取单词（词元）或文档的连续表示已变得相当流行。这些基于神经语言模型的信息检索系统大致可分为两类：a) 单向量方法和b) 多向量方法。

Single vector approaches obtain a single vector representation $\left( {v \in  {\mathbf{R}}^{d}}\right)$ for the query and the documents. Usually, cosine similarity is used to compare the representation of the query and the document and obtain a similarity score. The documents are ordered based on their similarity score and the top-k similar documents are retrieved. (Karpukhin et al., 2020) used separate encoders to encode both the query and the documents. They used BM25 to obtain negative passages for the contrastive loss. In addition, the authors also used in-batch negatives during training. (Chang et al., 2019; Guu et al., 2020; Xu et al., 2022) rely on data augmentation techniques like Inverse Cloze Tasks to create data for training.

单向量方法为查询和文档获取单个向量表示$\left( {v \in  {\mathbf{R}}^{d}}\right)$。通常，使用余弦相似度来比较查询和文档的表示，并获得相似度得分。文档根据其相似度得分进行排序，并检索出前k个最相似的文档。（卡尔普欣等人，2020年）使用单独的编码器对查询和文档进行编码。他们使用BM25来获取用于对比损失的负样本段落。此外，作者在训练过程中还使用了批次内负样本。（张等人，2019年；古等人，2020年；徐等人，2022年）依赖于诸如反向完形填空任务等数据增强技术来创建训练数据。

Multi-vector approaches on the other hand obtain multiple vectors per query or documents. The reasoning behind this is that a single vector representation will be unable to capture the fine interactions between the query and the document needed to retrieve the relevant document. ColBERT (Khat-tab and Zaharia, 2020) obtains contextualized representation from BERT for every sub-word token in the query and the document separately. They calculate the similarity of each query token representation with all the document token representations and the maximum similarity score is noted. The final similarity score is the summation of all the maximum similarity scores per token obtained in the previous step. (Luan et al., 2021) use a single vector per query but multiple vectors to represent the documents. ColBERTv2 (Santhanam et al., 2022) adopt the late interaction of ColBERT (Khattab and Zaharia, 2020). While ColBERT obtains negative documents from a model like BM25, ColBERTv2 uses ColBERT to retrieve top-k documents for a given query. The retrieved query-documents pairs are passed through a cross-encoder to obtain a similarity score. The model is trained using KL-Divergence loss to distill the cross-encoder scores. Recently, XTR (Lee et al., 2023) proposed an efficient multi-vector retrieval method that addresses the limitation of the ColBERT model. More details about the XTR model are discussed in Section 3.1.

另一方面，多向量方法为每个查询或文档获取多个向量。其背后的原因是，单个向量表示无法捕捉到检索相关文档所需的查询和文档之间的精细交互。ColBERT（卡塔布和扎哈里亚，2020年）分别从BERT中为查询和文档中的每个子词词元获取上下文表示。他们计算每个查询词元表示与所有文档词元表示的相似度，并记录最大相似度得分。最终相似度得分是上一步中每个词元获得的所有最大相似度得分的总和。（栾等人，2021年）每个查询使用单个向量，但使用多个向量来表示文档。ColBERTv2（桑塔南姆等人，2022年）采用了ColBERT（卡塔布和扎哈里亚，2020年）的后期交互方式。虽然ColBERT从像BM25这样的模型中获取负样本文档，但ColBERTv2使用ColBERT为给定查询检索前k个文档。将检索到的查询 - 文档对通过一个交叉编码器以获得相似度得分。该模型使用KL散度损失进行训练，以提炼交叉编码器的得分。最近，XTR（李等人，2023年）提出了一种高效的多向量检索方法，解决了ColBERT模型的局限性。关于XTR模型的更多细节将在3.1节中讨论。

## 3 System Overview

## 3 系统概述

Our system is built on XTR, and we use the same notations and expressions from the original paper to provide an overview.

我们的系统基于XTR构建，我们使用原论文中的相同符号和表达式进行概述。

### 3.1 XTR: Training and Inference

### 3.1 XTR：训练和推理

XTR was recently proposed to improve the training and inference efficiency of multi-vector retrieval models based on Colbert architecture. Unlike single-vector retrieval models that use one dense embedding for input text and determine similarity with a dot product, multi-vector retrieval models employ multiple dense embeddings for each query and document. These models usually utilize all contextualized word representations of the input text, enhancing overall model expressiveness. For a query $Q = {q}_{i = 1}^{n}$ and a document $D = {d}_{j = 1}^{m}$ ,where ${q}_{i}$ and ${d}_{j}$ represent d-dimensional vectors for query tokens and document tokens, multi-vector retrieval models determine the query-document similarity as follows:

XTR最近被提出，旨在提高基于Colbert架构的多向量检索模型的训练和推理效率。与使用一个密集嵌入来表示输入文本并通过点积确定相似度的单向量检索模型不同，多向量检索模型为每个查询和文档使用多个密集嵌入。这些模型通常利用输入文本的所有上下文词表示，增强了整体模型的表达能力。对于一个查询$Q = {q}_{i = 1}^{n}$和一个文档$D = {d}_{j = 1}^{m}$，其中${q}_{i}$和${d}_{j}$分别表示查询词元和文档词元的d维向量，多向量检索模型按如下方式确定查询 - 文档相似度：

$$
f\left( {Q,D}\right)  = \mathop{\sum }\limits_{i}^{n}\mathop{\max }\limits_{{j \in  \left| D\right| }}{q}_{i}^{T}{d}_{j} = \mathop{\sum }\limits_{i}^{n}\mathop{\sum }\limits_{i}^{m}{A}_{ij}{q}_{i}^{T}{d}_{j}
$$

(1)

<!-- Media -->

<!-- figureText: ColBERT inference Doc ${}_{1}$ (b) Gathering (c) Scoring (training) new ${f}_{\text{ColBERT }}$ Query ${\text{Score}}_{1}$ new ${\text{Score}}_{2}$ FLOPs/query (Scoring) (b) Scoring ${f}_{\text{ColBERT }}$ Score, ${f}_{\text{XTR }}$ 4,000 - Cheaper Recall@100 on BEIR ColBERT 64.5 Score, XTR 68.0 (a) Token Retrieval Query ${\text{Doc}}_{2}$ XTR training & inference ${\text{Doc}}_{1}$ (a) Token Retrieval ${f}_{{}_{\mathrm{{XTR}}}}$ Query ${f}_{\text{XTR }}$ -->

<img src="https://cdn.noedgeai.com/0195b38e-7712-7696-9b0e-05c275ecc794_2.jpg?x=245&y=203&w=1180&h=761&r=0"/>

Figure 1: XTR retrieval (figure reused from original paper)

图1：XTR检索（图引自原论文）

<!-- Media -->

${P}_{ij} = {q}_{i}^{T}{d}_{j}$ and $A \in  \{ 0,1{\} }^{nxm}$ denotes the alignment matrix with ${A}_{ij}$ being the token-level alignment between the query token vector ${q}_{i}$ and the document token vector ${d}_{j}$ . In ColBERT,sum-of-max operator sets ${A}_{ij} = {\mathbb{1}}_{\left\lbrack  j = {\operatorname{argmax}}_{{j}^{\prime }}\left( {P}_{i{j}^{\prime }}\right) \right\rbrack  }$ , where the argmax is over tokens from a single document D,and $\mathbb{1}\left\lbrack  *\right\rbrack$ is an indicator function.

${P}_{ij} = {q}_{i}^{T}{d}_{j}$ 和 $A \in  \{ 0,1{\} }^{nxm}$ 表示对齐矩阵，其中 ${A}_{ij}$ 是查询Token向量 ${q}_{i}$ 和文档Token向量 ${d}_{j}$ 之间的Token级对齐。在ColBERT中，最大和运算符设置 ${A}_{ij} = {\mathbb{1}}_{\left\lbrack  j = {\operatorname{argmax}}_{{j}^{\prime }}\left( {P}_{i{j}^{\prime }}\right) \right\rbrack  }$ ，其中argmax是针对单个文档D中的Token，并且 $\mathbb{1}\left\lbrack  *\right\rbrack$ 是一个指示函数。

Figure 1 demonstrates how XTR streamlines the retrieval process for multi-vector models such as ColBERT by using tokens retrieved in the initial phase to score documents directly, maintaining retrieval performance. This is accomplished by adjusting the training objective to simulate the token retrieval stage through a different alignment strategy,denoted as $\widehat{A}$ . Specifically,the alignment is defined as ${\widehat{A}}_{ij} = \mathbb{1}\left\lbrack  {j \in  {\operatorname{topk}}_{{j}^{\prime }}\left( {P}_{i{j}^{\prime }}\right) }\right\rbrack$ ,where the top-k operator is applied over tokens from mini-batch documents,returning the indices of the $k$ largest values. The modified equation is as follows:

图1展示了XTR如何通过使用初始阶段检索到的Token直接对文档进行评分，从而简化了ColBERT等多向量模型的检索过程，同时保持了检索性能。这是通过调整训练目标来实现的，即通过一种不同的对齐策略（表示为 $\widehat{A}$ ）来模拟Token检索阶段。具体来说，对齐定义为 ${\widehat{A}}_{ij} = \mathbb{1}\left\lbrack  {j \in  {\operatorname{topk}}_{{j}^{\prime }}\left( {P}_{i{j}^{\prime }}\right) }\right\rbrack$ ，其中top - k运算符应用于小批量文档中的Token，返回 $k$ 个最大值的索引。修改后的方程如下：

$$
f\left( {Q,D}\right)  = \frac{1}{Z}\mathop{\sum }\limits_{i}^{n}\mathop{\max }\limits_{{j \in  \left| D\right| }}{\widehat{A}}_{ij}{q}_{i}^{T}{d}_{j} \tag{2}
$$

Here,the normalizer $Z$ denotes the count of query tokens that retrieve at least one document token from $D$ . If all ${\widehat{A}}_{ij} = 0,Z$ is clipped to a small values causing $f\left( {Q,D}\right)$ to become 0 . During training, the cross-entropy loss over in-batch negatives is used, expressed as:

这里，归一化器 $Z$ 表示从 $D$ 中检索到至少一个文档Token的查询Token的数量。如果所有 ${\widehat{A}}_{ij} = 0,Z$ 都被裁剪为一个小值，会导致 $f\left( {Q,D}\right)$ 变为0。在训练期间，使用批次内负样本的交叉熵损失，表达式如下：

$$
{\mathcal{L}}_{CE} =  - \log \frac{\exp f\left( {Q,{D}^{ + }}\right) )}{\mathop{\sum }\limits_{{b = 1}}^{B}\exp f\left( {Q,{D}_{b}}\right) )} \tag{3}
$$

Scoring Documents using Retrieved Tokens: During inference, multi-vector retrieval models first have a set of candidate documents ${\widehat{D}}_{1 : C}$ from the token retrieval stage:

使用检索到的Token对文档进行评分：在推理过程中，多向量检索模型首先从Token检索阶段获得一组候选文档 ${\widehat{D}}_{1 : C}$ ：

$$
{\widehat{D}}_{1 : C} = \widehat{D} \mid  {d}_{j} \in  \widehat{D} \land  {d}_{j} \in  \text{ top } - {k}^{\prime }\left( {q * }\right)  \tag{4}
$$

Here, $\operatorname{top} - k\left( {q * }\right)$ represents a union of the top- ${k}^{\prime }$ document tokens (from the entire corpus) based on the inner product scores with each query vector. With $n$ query token vectors,there are $\mathrm{C}\left( { \leq  n{k}^{\prime }}\right)$ candidate documents. Traditional methods load entire token vectors for each document and compute equation 1 for every query and candidate document pair. In contrast, XTR scores the documents solely based on retrieved token similarity. This significantly reduces computational costs during the scoring stage by eliminating redundant inner product computations and unnecessary (non-max) inner products. Moreover, the resource-intensive gathering stage, which involves loading all document token vectors for computing equation 1 , is eliminated entirely.

这里， $\operatorname{top} - k\left( {q * }\right)$ 表示基于与每个查询向量的内积得分的前 ${k}^{\prime }$ 个文档Token（来自整个语料库）的并集。对于 $n$ 个查询Token向量，有 $\mathrm{C}\left( { \leq  n{k}^{\prime }}\right)$ 个候选文档。传统方法会为每个文档加载整个Token向量，并为每个查询和候选文档对计算方程1。相比之下，XTR仅根据检索到的Token相似度对文档进行评分。这通过消除冗余的内积计算和不必要的（非最大值）内积，显著降低了评分阶段的计算成本。此外，完全消除了资源密集型的收集阶段，该阶段涉及加载所有文档Token向量以计算方程1。

Missing Similarity Imputation: During inference, ${k}^{\prime }$ document tokens are retrieved for each of the $n$ query tokens. Assuming each document token belongs to a unique document, this results in $C = n{k}^{\prime }$ candidate documents. In the absence of the gathering stage, there is a single token similarity to score each document. However, during training with either equation 1 or equation 2 , each positive document has up to $n$ (max) token similarities to average,which tends to converge to $n$ as training progresses. Therefore, during inference, the missing similarity for each query token is imputed by treating each candidate document as if it were positive,with $n$ token similarities. For every candidate document $\widehat{D}$ ,the following scoring function is defined:

缺失相似度插补：在推理过程中，为 $n$ 个查询Token中的每个查询Token检索 ${k}^{\prime }$ 个文档Token。假设每个文档Token属于一个唯一的文档，这会产生 $C = n{k}^{\prime }$ 个候选文档。在没有收集阶段的情况下，每个文档只有一个Token相似度用于评分。然而，在使用方程1或方程2进行训练时，每个正文档最多有 $n$ 个（最大）Token相似度用于求平均值，随着训练的进行，这个值趋于 $n$ 。因此，在推理过程中，通过将每个候选文档视为正文档（具有 $n$ 个Token相似度）来插补每个查询Token的缺失相似度。对于每个候选文档 $\widehat{D}$ ，定义以下评分函数：

$$
f\left( {Q,\widehat{D}}\right)  = \mathop{\sum }\limits_{i}^{n}\left\lbrack  {\mathop{\max }\limits_{{j \in  \left| D\right| }}{\widehat{A}}_{ij}{q}_{i}^{T}{d}_{j}\left( {1 - {\widehat{A}}_{ij}}\right) {m}_{i}}\right\rbrack   \tag{5}
$$

This is similar to equation 2 , but it introduces ${m}_{i} \in  \mathbb{R}$ ,estimating the missing similarity for each ${q}_{i}$ . The definition of $\widehat{A}$ is similar to the one in equation 2,except that it uses ${k}^{\prime }$ for the top- $k$ operator. For each ${q}_{i}$ ,if ${\widehat{A}}_{i * } = 0$ and ${m}_{i} \geq  0,{q}_{i}$ considers the missing similarity ${m}_{i}$ as the maximum value. Crucially, XTR eliminates the need to recompute any ${q}_{i}^{T}{d}_{j}$ . When ${\widehat{A}}_{ij} = 1$ ,the retrieval score from the token retrieval stage is already known,and when ${\widehat{A}}_{ij} = 0$ ,there is no need to compute it as ${\widehat{A}}_{ij}{q}_{i}^{T}{d}_{j} = 0$ . Note that when every ${\widehat{A}}_{ij} = 1$ ,the equation becomes the sum-of-max operator. Conversely, when no document tokens of $\widehat{D}$ were retrieved for ${q}_{i}$ (i.e., ${\widehat{A}}_{i * } = 0$ ),the model falls back to the imputed score ${m}_{i}$ . This provides an approximate sum-of-max result, as the missing similarity would have a score less than or equal to the score of the last retrieved token.

这与方程2类似，但它引入了${m}_{i} \in  \mathbb{R}$，用于估计每个${q}_{i}$的缺失相似度。$\widehat{A}$的定义与方程2中的定义类似，只是它对前$k$个运算符使用了${k}^{\prime }$。对于每个${q}_{i}$，如果${\widehat{A}}_{i * } = 0$且${m}_{i} \geq  0,{q}_{i}$，则将缺失相似度${m}_{i}$视为最大值。至关重要的是，XTR无需重新计算任何${q}_{i}^{T}{d}_{j}$。当${\widehat{A}}_{ij} = 1$时，词元检索阶段的检索分数已经已知；当${\widehat{A}}_{ij} = 0$时，由于${\widehat{A}}_{ij}{q}_{i}^{T}{d}_{j} = 0$，无需计算该分数。请注意，当每个${\widehat{A}}_{ij} = 1$时，该方程变为最大和运算符。相反，当没有为${q}_{i}$检索到$\widehat{D}$的任何文档词元时（即${\widehat{A}}_{i * } = 0$），模型会退回到估算分数${m}_{i}$。这提供了一个近似的最大和结果，因为缺失相似度的分数会小于或等于最后一个检索到的词元的分数。

## 4 ColBERTv2: Indexing and Retrieval

## 4 ColBERTv2：索引与检索

XTR uses a basic MIPS library for indexing, resulting in a significant increase in space requirements. More precisely, it employs the ScaNN library (Guo et al., 2020; Sun, 2020) to store all contextualized vectors without compressing dimensionality. In contrast, our approach draws inspiration from Col-BERTv2's (Santhanam et al., 2022) indexing strategy for multi-vector models, aiming to improve both space utilization and inference efficiency. Col-BERTv2 achieves these enhancements by combining an aggressive residual compression mechanism with a denoised supervision strategy.

XTR使用基本的最大内积搜索（MIPS）库进行索引，这显著增加了空间需求。更确切地说，它采用了ScaNN库（Guo等人，2020；Sun，2020）来存储所有上下文向量，而不进行降维压缩。相比之下，我们的方法借鉴了Col - BERTv2（Santhanam等人，2022）针对多向量模型的索引策略，旨在提高空间利用率和推理效率。ColBERTv2通过将激进的残差压缩机制与去噪监督策略相结合来实现这些改进。

Figure 2 shows an overview of how ColBERTv2 attempts to do efficient index management with reduced storage. Contextualized vectors exhibit clustering in regions that capture highly specific token semantics, with evidence suggesting that vectors corresponding to each sense of a word cluster closely, demonstrating only minor variation due to context (Santhanam et al., 2022). Leveraging this regularity, a residual representation is introduced in ColBERTv2, significantly reducing the space requirements of late interaction models without any need for architectural or training changes. In this approach,given a set of centroids $C$ ,each vector $v$ is encoded as the index of its closest centroid ${C}_{t}$ and a quantized vector $\widetilde{r}$ that approximates the residual $r = v - {C}_{t}$ . During search,the centroid index $t$ and residual $\widetilde{r}$ are used to recover an approximate $\widetilde{v} = {C}_{t} + \widetilde{r}$ . Each dimension of $r$ is quantized into one or two bits to encode $\widetilde{r}$ .

图2展示了ColBERTv2如何尝试以减少存储的方式进行高效索引管理的概述。上下文向量在捕捉高度特定词元语义的区域呈现聚类现象，有证据表明，对应于一个词的每个义项的向量紧密聚类，仅因上下文而有微小变化（Santhanam等人，2022）。利用这一规律，ColBERTv2引入了残差表示，在无需进行架构或训练更改的情况下，显著降低了后期交互模型的空间需求。在这种方法中，给定一组质心$C$，每个向量$v$被编码为其最近质心${C}_{t}$的索引和一个近似残差$r = v - {C}_{t}$的量化向量$\widetilde{r}$。在搜索过程中，质心索引$t$和残差$\widetilde{r}$用于恢复近似的$\widetilde{v} = {C}_{t} + \widetilde{r}$。$r$的每个维度被量化为一到两位以编码$\widetilde{r}$。

### 4.1 Indexing

### 4.1 索引

In the indexing stage, following ColBERTv2 we undertake a three-stage process for a given document collection, efficiently precomputing and organizing the embeddings for rapid nearest neighbor search.

在索引阶段，我们遵循ColBERTv2的方法，对给定的文档集合进行三阶段处理，高效地预计算和组织嵌入向量，以便进行快速最近邻搜索。

- Centroid Selection: In the initial stage, we choose a set of cluster centroids $C$ . These centroids serve a dual purpose in supporting both residual encoding and nearest neighbor search. To reduce memory usage, k-means clustering is applied to the embeddings produced by the T5 encoder, considering only a sample of passages.

- 质心选择：在初始阶段，我们选择一组聚类质心$C$。这些质心具有双重用途，既支持残差编码，又支持最近邻搜索。为了减少内存使用，对T5编码器生成的嵌入向量应用k - 均值聚类，且仅考虑部分段落样本。

<!-- Media -->

<!-- figureText: Original 4,096 bytes 2 7 128 x 32-bit ${128} \times  {32} - b/1$ 4,096 bytes 8 bytes ${1.024} \times  {32} - b/t$ vector 0 1 Sliced into 128 x 32-bit 128 x 32-bit subvectors Replaced with id of B-bit 8-bit 8-bit 8-bit 8-bit 8-bit 8-bit 8-bit 8-bit nearest centroid -->

<img src="https://cdn.noedgeai.com/0195b38e-7712-7696-9b0e-05c275ecc794_4.jpg?x=479&y=192&w=690&h=356&r=0"/>

Figure 2: Overview of ColBERTv2 index optimization

图2：ColBERTv2索引优化概述

<!-- Media -->

- Passage Encoding: With the centroids selected, every passage in the corpus undergoes encoding. This involves invoking the T5 encoder, compressing the output embeddings, and assigning each embedding to its nearest centroid while computing a quantized residual. The compressed representations are then saved to disk once a chunk of passages is encoded.

- 段落编码：选择质心后，语料库中的每个段落都进行编码。这包括调用T5编码器，压缩输出的嵌入向量，并将每个嵌入向量分配给其最近的质心，同时计算量化残差。对一批段落进行编码后，将压缩表示保存到磁盘。

- Index Inversion: To facilitate rapid nearest neighbor search, the embedding IDs corresponding to each centroid are grouped together, and this inverted list is stored on disk. During search, this enables quick identification of token-level embed-dings similar to those in a query.

- 索引反转：为了便于快速最近邻搜索，将对应于每个质心的嵌入向量ID分组在一起，并将这个倒排列表存储在磁盘上。在搜索过程中，这可以快速识别与查询中词元嵌入向量相似的词元级嵌入向量。

### 4.2 Retrieval

### 4.2 检索

During the retrieval phase, we use the trained ColXTR to encode a query, generating contextual-ized representations denoted as $Q$ . Following this, we compute the inner product between these query representations and centroids $\left( {{Q}^{T}C}\right)$ ,identifying the nearest centroids for each query token embedding. Leveraging the inverted list, we then identify the document token embeddings that are closer to these centroids.

在检索阶段，我们使用训练好的ColXTR对查询进行编码，生成表示为$Q$的上下文表示。随后，我们计算这些查询表示与质心$\left( {{Q}^{T}C}\right)$之间的内积，为每个查询词元嵌入确定最近的质心。利用倒排表，我们接着识别出更接近这些质心的文档词元嵌入。

Subsequently, we decompress these document token embeddings and calculate their inner product with the corresponding query vectors. The resulting similarity scores are organized based on document ids. In instances where a score for a particular query token and document token is missing, we impute it as per the equation 5 . Finally, the documents are directly reranked using these similarity scores.

随后，我们对这些文档词元嵌入进行解压缩，并计算它们与相应查询向量的内积。得到的相似度得分根据文档ID进行组织。在特定查询词元和文档词元的得分缺失的情况下，我们根据公式5进行插补。最后，使用这些相似度得分直接对文档进行重排序。

## 5 Experiments

## 5 实验

We finetune the encoder of t5-base (Raffel et al., 2020) with XTR learning objective on MSMarco training set with a learning rate of $1 - \mathrm{e}3$ . XTR uses ${k}_{\text{train }}$ parameter which we set to 320 . We also employ a projection layer that compresses the encoded representations from 768 dimensions to 128 dimensions. The model is trained on a single A100 80GB GPU, with a batch size of 48. Moreover, we trained the model with hard negatives mined from BM25, one per positive query/document pair in a batch. The model is trained for ${50}\mathrm{\;K}$ steps,and the best model based on the development set is used for the evaluation.

我们在MSMarco训练集上以$1 - \mathrm{e}3$的学习率，使用XTR学习目标对t5-base（拉菲尔等人，2020年）的编码器进行微调。XTR使用${k}_{\text{train }}$参数，我们将其设置为320。我们还采用了一个投影层，将编码表示从768维压缩到128维。该模型在单个A100 80GB GPU上进行训练，批量大小为48。此外，我们使用从BM25挖掘的难负样本对模型进行训练，每个批次中的每个正查询/文档对对应一个难负样本。模型训练${50}\mathrm{\;K}$步，并使用基于开发集的最佳模型进行评估。

During retrieval,we use variable $k$ depending on the size of the index. For smaller indexes $( > 1\mathrm{M}$ documents),we set $k$ to 500,while for larger ones, we increased it to 100,000 . For each query token, we probed top 10 centroids.

在检索过程中，我们根据索引的大小使用变量$k$。对于较小的索引（$( > 1\mathrm{M}$个文档），我们将$k$设置为500，而对于较大的索引，我们将其增加到100,000。对于每个查询词元，我们探查前10个质心。

### 5.1 Benchmark

### 5.1 基准测试

We use datasets from BIER (Thakur et al., 2021) benchmark as our evaluation benchmark. BIER is a popular benchmark in IR community which is a collection of 18 datasets of varying domains as well as tasks. Because we build on top of XTR, we chose the same subset from BIER which was used for XTR benchmarking to be comparable. The datasets are as follows: (1) AR: ArguAna, (2) TO: Touché-2020, (3) FE: Fever,(4) CF: Climate-Fever, (5) SF: Scifact, (6) CV: TREC-COVID, (7) NF: NF-Corpus, (8) NQ: Natural Questions, (9) HQ: Hot-potQA, (10) FQ: FiQA-2018, (11) SD: SCIDOCS, (12) DB: DBPedia, (13) QU: Quora.

我们使用BIER（塔库尔等人，2021年）基准测试中的数据集作为我们的评估基准。BIER是信息检索（IR）领域中一个流行的基准测试，它是一个包含18个不同领域和任务的数据集集合。由于我们是在XTR的基础上进行构建的，为了具有可比性，我们从BIER中选择了与XTR基准测试相同的子集。这些数据集如下：（1）AR：ArguAna，（2）TO：Touché - 2020，（3）FE：Fever，（4）CF：Climate - Fever，（5）SF：Scifact，（6）CV：TREC - COVID，（7）NF：NF - Corpus，（8）NQ：Natural Questions，（9）HQ：Hot - potQA，（10）FQ：FiQA - 2018，（11）SD：SCIDOCS，（12）DB：DBPedia，（13）QU：Quora。

### 5.2 Evaluation Metric

### 5.2 评估指标

We use Normalised Discounted Cumulative Gain (NDCG) as our evaluation metric, psrticularly, we report NDCG@10.As suggested in (Thakur et al., 2021) NDCG is a robust metric to measure retrieval and reranker performance because it also considers the rank of the retrieved documents while computing the score and thus is a more informative metric than just recall.

我们使用归一化折损累计增益（NDCG）作为我们的评估指标，具体而言，我们报告NDCG@10。正如（塔库尔等人，2021年）所建议的，NDCG是衡量检索和重排序器性能的一个可靠指标，因为它在计算得分时还考虑了检索到的文档的排名，因此比单纯的召回率更具信息量。

<!-- Media -->

<table><tr><td>Datasets</td><td>$\mathbf{{AR}}$</td><td>TO</td><td>$\mathbf{{FE}}$</td><td>$\mathbf{{CF}}$</td><td>SF</td><td>CV</td><td>$\mathbf{{NF}}$</td><td>$\mathbf{{NQ}}$</td><td>HQ</td><td>$\mathbf{{FQ}}$</td><td>SD</td><td>$\mathbf{{DB}}$</td><td>$\mathbf{{QU}}$</td><td>$\mathbf{{Avg}.}$</td></tr><tr><td>BM25</td><td>39.7</td><td>44.0</td><td>65.1</td><td>17.0</td><td>67.9</td><td>59.5</td><td>32.2</td><td>31.0</td><td>63.0</td><td>23.6</td><td>14.9</td><td>31.8</td><td>78.9</td><td>43.7</td></tr><tr><td>Colbert</td><td>23.3</td><td>20.2</td><td>77.1</td><td>18.4</td><td>67.1</td><td>67.7</td><td>30.5</td><td>52.4</td><td>59.3</td><td>31.7</td><td>14.5</td><td>39.2</td><td>85.4</td><td>44.8</td></tr><tr><td>Colbert v2</td><td>46.3</td><td>26.3</td><td>78.5</td><td>17.6</td><td>69.3</td><td>73.8</td><td>33.8</td><td>56.2</td><td>66.7</td><td>35.6</td><td>15.4</td><td>44.6</td><td>85.2</td><td>49.9</td></tr><tr><td>XTR</td><td>40.7</td><td>31.3</td><td>73.7</td><td>20.7</td><td>71.0</td><td>73.6</td><td>34.0</td><td>53.0</td><td>64.7</td><td>34.7</td><td>14.5</td><td>40.9</td><td>86.1</td><td>49.1</td></tr><tr><td>ColXTR</td><td>49.3</td><td>29.1</td><td>73.1</td><td>12.6</td><td>71.9</td><td>69.6</td><td>34.3</td><td>41.1</td><td>61.1</td><td>33.4</td><td>15.7</td><td>27.2</td><td>81.9</td><td>46.2</td></tr></table>

<table><tbody><tr><td>数据集</td><td>$\mathbf{{AR}}$</td><td>到</td><td>$\mathbf{{FE}}$</td><td>$\mathbf{{CF}}$</td><td>旧金山（San Francisco）</td><td>计算机视觉（Computer Vision）</td><td>$\mathbf{{NF}}$</td><td>$\mathbf{{NQ}}$</td><td>高画质（High Quality）</td><td>$\mathbf{{FQ}}$</td><td>标准清晰度（Standard Definition）</td><td>$\mathbf{{DB}}$</td><td>$\mathbf{{QU}}$</td><td>$\mathbf{{Avg}.}$</td></tr><tr><td>BM25算法</td><td>39.7</td><td>44.0</td><td>65.1</td><td>17.0</td><td>67.9</td><td>59.5</td><td>32.2</td><td>31.0</td><td>63.0</td><td>23.6</td><td>14.9</td><td>31.8</td><td>78.9</td><td>43.7</td></tr><tr><td>科尔伯特（Colbert）</td><td>23.3</td><td>20.2</td><td>77.1</td><td>18.4</td><td>67.1</td><td>67.7</td><td>30.5</td><td>52.4</td><td>59.3</td><td>31.7</td><td>14.5</td><td>39.2</td><td>85.4</td><td>44.8</td></tr><tr><td>科尔伯特v2</td><td>46.3</td><td>26.3</td><td>78.5</td><td>17.6</td><td>69.3</td><td>73.8</td><td>33.8</td><td>56.2</td><td>66.7</td><td>35.6</td><td>15.4</td><td>44.6</td><td>85.2</td><td>49.9</td></tr><tr><td>XTR</td><td>40.7</td><td>31.3</td><td>73.7</td><td>20.7</td><td>71.0</td><td>73.6</td><td>34.0</td><td>53.0</td><td>64.7</td><td>34.7</td><td>14.5</td><td>40.9</td><td>86.1</td><td>49.1</td></tr><tr><td>科尔XTR</td><td>49.3</td><td>29.1</td><td>73.1</td><td>12.6</td><td>71.9</td><td>69.6</td><td>34.3</td><td>41.1</td><td>61.1</td><td>33.4</td><td>15.7</td><td>27.2</td><td>81.9</td><td>46.2</td></tr></tbody></table>

Table 1: ColXTR as Retriever

表1：将ColXTR用作检索器

<!-- Media -->

### 5.3 Baselines

### 5.3 基线模型

To compare the performance of ColXTR, we use several baselines, including BM25, the most popular sparse retriever. Additionally, we compare ColXTR with most relevant baselines such as ColBERT, ColBERTv2, the original XTR work.

为了比较ColXTR的性能，我们使用了几个基线模型，其中包括最流行的稀疏检索器BM25。此外，我们还将ColXTR与最相关的基线模型进行了比较，如ColBERT、ColBERTv2以及原始的XTR研究成果。

### 5.4 Results

### 5.4 结果

In this section, we review the experimental results and optimization benefits of ColXTR in detail.

在本节中，我们将详细回顾ColXTR的实验结果和优化优势。

As shown in Table 1, we see BM25, as expected, scores lower than other retrievers due to its reliance on lexical overlap. ColBERT improves upon BM25, while XTR further boosts performance over ColBERT. ColBERTv2, benefiting from distillation training, achieves the highest scores overall. Our system, ColXTR, performs better than ColBERT but falls short of XTR and ColBERTv2. This drop in accuracy can be attributed to the lack of hard negative mining, lack of distillation training, and the use of compressed embeddings. While Col-BERTv2 is computationally expensive at inference, and XTR poses challenges in index management, ColXTR adopts XTR-style training with ColBERT-style compressed representations to make it more lightweight while maintaining comparable performance.

如表1所示，正如预期的那样，由于BM25依赖于词法重叠，其得分低于其他检索器。ColBERT在BM25的基础上有所改进，而XTR相较于ColBERT进一步提升了性能。得益于蒸馏训练，ColBERTv2总体得分最高。我们的系统ColXTR的表现优于ColBERT，但不如XTR和ColBERTv2。这种准确率的下降可归因于缺乏难负样本挖掘、未进行蒸馏训练以及使用了压缩嵌入。虽然ColBERTv2在推理时计算成本较高，且XTR在索引管理方面存在挑战，但ColXTR采用了XTR风格的训练方式和ColBERT风格的压缩表示，使其在保持相当性能的同时更加轻量级。

### 5.5 ColXTR Optimization Impact

### 5.5 ColXTR优化的影响

Here we discuss the implications of the design choices and optimizations we have incorporated in ColXTR in making it a lightweight system, easy to deploy and manage.

在这里，我们讨论在ColXTR中所采用的设计选择和优化措施，这些措施使其成为一个易于部署和管理的轻量级系统。

In $\mathbf{{ColXTR}}$ we try to combine the different set of optimizations proposed in XTR and ColBERT together. We follow the XTR style retrieval with missing token imputation during inference, without the expensive gathering stage of ColBERT. As reported in XTR (Lee et al., 2023), this makes the inference ${400}\mathrm{x}$ times faster than ColBERT.

在$\mathbf{{ColXTR}}$中，我们尝试将XTR和ColBERT中提出的不同优化集结合起来。我们采用XTR风格的检索方式，在推理过程中进行缺失词元插补，避免了ColBERT中昂贵的收集阶段。正如XTR（Lee等人，2023）所报道的，这使得推理速度比ColBERT快${400}\mathrm{x}$倍。

On the other hand, XTR uses full token representations for indexing, and retaining the original size of 768, as in XTR, would result in a significantly larger memory footprint and overhead. Instead of that, we applied ColBERT like approach where we learn to compress the representation to lower dimensions and make further optimizations with residual compression. This reduces the index size by a huge margin,almost a shrink of ${97}\%$ ,making the index management much cheaper and easier.

另一方面，XTR使用完整的词元表示进行索引，如果像XTR那样保留768的原始大小，将导致显著更大的内存占用和开销。相反，我们采用了类似ColBERT的方法，即学习将表示压缩到更低维度，并通过残差压缩进行进一步优化。这极大地减小了索引大小，几乎缩小了${97}\%$，使得索引管理成本更低、更轻松。

<!-- Media -->

<table><tr><td>Datasets</td><td>Faiss HNSW Flat Index(in GB)</td><td>Colbert index(in GB)</td></tr><tr><td>$\mathbf{{NQ}}$</td><td>860</td><td>25</td></tr><tr><td>NFCorpus</td><td>2.4</td><td>0.091</td></tr><tr><td>TREC COVID</td><td>67</td><td>3</td></tr><tr><td>Touché 2020</td><td>481</td><td>7</td></tr></table>

<table><tbody><tr><td>数据集</td><td>Faiss HNSW扁平索引（以GB为单位）</td><td>Colbert索引（以GB为单位）</td></tr><tr><td>$\mathbf{{NQ}}$</td><td>860</td><td>25</td></tr><tr><td>NFCorpus（原文）</td><td>2.4</td><td>0.091</td></tr><tr><td>TREC COVID（原文）</td><td>67</td><td>3</td></tr><tr><td>Touché 2020（原文）</td><td>481</td><td>7</td></tr></tbody></table>

Table 2: Comparison of Faiss HNSW Flat indices and ColBERT indices in terms of size, with both using embedding dimensions of 128 .

表2：Faiss HNSW Flat索引和ColBERT索引在大小方面的比较，两者的嵌入维度均为128。

<!-- Media -->

In Table 2, we give some empirical numbers to establish how the ColBERTv2 optimizations we discussed in Section 4 help in reducing the index size. Considering the first dataset, NQ, as an example, we can see it offers almost upto 97% shrinkage over the original index. On an average, we see the index size reduced by ${98}\%$ across 4 datasets, which validates the need for our optimizations in designing ColXTR making the index management and deployment much cheaper and easier.

在表2中，我们给出了一些实证数据，以说明我们在第4节中讨论的ColBERTv2优化如何有助于减小索引大小。以第一个数据集NQ为例，我们可以看到，与原始索引相比，它的索引大小几乎缩小了97%。平均而言，在4个数据集上，我们发现索引大小减少了${98}\%$，这验证了我们在设计ColXTR时进行优化的必要性，使得索引管理和部署更加经济和便捷。

## 6 Conclusion

## 6 结论

We have proposed ColXTR, an optimized multi-vector retrieval model built on top of t5-base that combines the best of both worlds: ColBERTv2 like index optimization and runtime optimizations from XTR for speedy inference. We posit this is a need of the hour for meeting industry needs for scalability with practical resource constraints. We empirically show that the lightweight training and inference pipeline for ColXTR provides competitive and in some cases even better performance than state-of-the-art retrieval models, while reducing the index footprint almost by ${97}\%$ . We believe ColXTR can potentially become a default choice for using neural retrievers in industry.

我们提出了ColXTR，这是一个基于t5-base构建的优化多向量检索模型，它结合了两者的优势：类似ColBERTv2的索引优化以及来自XTR的运行时优化，以实现快速推理。我们认为，在实际资源受限的情况下，满足行业对可扩展性的需求是当务之急。我们通过实证表明，ColXTR的轻量级训练和推理管道具有竞争力，在某些情况下甚至比最先进的检索模型表现更好，同时将索引占用空间几乎减少了${97}\%$。我们相信，ColXTR有可能成为行业中使用神经检索器的默认选择。

## References

## 参考文献

Deng Cai, Yan Wang, Wei Bi, Zhaopeng Tu, Xi-aojiang Liu, and Shuming Shi. 2019. Retrieval-guided dialogue response generation via a matching-to-generation framework. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 1866-1875, Hong Kong, China. Association for Computational Linguistics.

蔡登、王岩、毕伟、涂兆鹏、刘晓江和史树明。2019年。通过匹配到生成框架进行检索引导的对话回复生成。《2019年自然语言处理经验方法会议和第9届自然语言处理国际联合会议论文集》（EMNLP - IJCNLP），第1866 - 1875页，中国香港。计算语言学协会。

Wei-Cheng Chang, X Yu Felix, Yin-Wen Chang, Yim-ing Yang, and Sanjiv Kumar. 2019. Pre-training tasks for embedding-based large-scale retrieval. In International Conference on Learning Representations.

张维政、X Yu Felix、张尹文、杨一鸣和桑吉夫·库马尔。2019年。基于嵌入的大规模检索的预训练任务。《国际学习表征会议》。

Danqi Chen and Wen-tau Yih. 2020. Open-domain question answering. In Proceedings of the 58th annual meeting of the association for computational linguistics: tutorial abstracts, pages 34-37.

陈丹琦和易文涛。2020年。开放域问答。《计算语言学协会第58届年会教程摘要集》，第34 - 37页。

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. BERT: pre-training of deep bidirectional transformers for language understanding. CoRR, abs/1810.04805.

雅各布·德夫林、张明伟、肯顿·李和克里斯蒂娜·图托纳娃。2018年。BERT：用于语言理解的深度双向变换器的预训练。CoRR，abs/1810.04805。

Bettina Fazzinga and Thomas Lukasiewicz. 2010. Semantic search on the web. Semantic Web, 1(1-2):89- 96.

贝蒂娜·法津加和托马斯·卢卡谢维奇。2010年。网络语义搜索。《语义网》，1(1 - 2)：89 - 96。

Ruiqi Guo, Philip Sun, Erik Lindgren, Quan Geng, David Simcha, Felix Chern, and Sanjiv Kumar. 2020. Accelerating large-scale inference with anisotropic vector quantization. In International Conference on Machine Learning.

郭瑞琦、菲利普·孙、埃里克·林德格伦、耿全、大卫·辛查、费利克斯·陈和桑吉夫·库马尔。2020年。通过各向异性向量量化加速大规模推理。《国际机器学习会议》。

Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasu-pat, and Mingwei Chang. 2020. Retrieval augmented language model pre-training. In International conference on machine learning, pages 3929-3938. PMLR.

凯尔文·古、肯顿·李、佐拉·董、帕努蓬·帕苏帕特和张明伟。2020年。检索增强语言模型预训练。《国际机器学习会议》，第3929 - 3938页。PMLR。

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and

弗拉基米尔·卡尔普欣、巴拉斯·奥古兹、闵世元、帕特里克·刘易斯、莱德尔·吴、谢尔盖·叶杜诺夫、陈丹琦和

Wen-tau Yih. 2020. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6769-6781.

易文涛。2020年。开放域问答的密集段落检索。《2020年自然语言处理经验方法会议论文集》（EMNLP），第6769 - 6781页。

Omar Khattab and Matei Zaharia. 2020. Colbert: Efficient and effective passage search via contextualized late interaction over bert. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, pages 39- 48.

奥马尔·哈塔卜和马特·扎哈里亚。2020年。Colbert：通过基于BERT的上下文后期交互实现高效有效的段落搜索。《第43届国际ACM SIGIR信息检索研究与发展会议论文集》，第39 - 48页。

Jinhyuk Lee, Zhuyun Dai, Sai Meher Karthik Duddu, Tao Lei, Iftekhar Naim, Ming-Wei Chang, and Vincent Y Zhao. 2023. Rethinking the role of token retrieval in multi-vector retrieval. In Thirty-seventh Conference on Neural Information Processing Systems.

李晋赫、戴竹云、赛·梅赫尔·卡尔蒂克·杜杜、雷涛、伊夫特哈尔·奈姆、张明伟和赵文森·Y。2023年。重新思考多向量检索中词元检索的作用。《第三十七届神经信息处理系统会议》。

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-täschel, et al. 2020. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems, 33:9459-9474.

帕特里克·刘易斯（Patrick Lewis）、伊桑·佩雷斯（Ethan Perez）、亚历山德拉·皮克图斯（Aleksandra Piktus）、法比奥·彼得罗尼（Fabio Petroni）、弗拉基米尔·卡尔普欣（Vladimir Karpukhin）、纳曼·戈亚尔（Naman Goyal）、海因里希·屈特勒（Heinrich Küttler）、迈克·刘易斯（Mike Lewis）、文涛·伊（Wen-tau Yih）、蒂姆·罗克 - 塔舍尔（Tim Rock-täschel）等。2020年。用于知识密集型自然语言处理任务的检索增强生成。《神经信息处理系统进展》，33:9459 - 9474。

Yi Luan, Jacob Eisenstein, Kristina Toutanova, and Michael Collins. 2021. Sparse, dense, and attentional representations for text retrieval. Transactions of the Association for Computational Linguistics, 9:329- 345.

栾毅（Yi Luan）、雅各布·艾森斯坦（Jacob Eisenstein）、克里斯蒂娜·图托纳娃（Kristina Toutanova）和迈克尔·柯林斯（Michael Collins）。2021年。用于文本检索的稀疏、密集和注意力表示。《计算语言学协会汇刊》，9:329 - 345。

PrimeQA. 2023. Primeqa: Toolkit for open domain qa. https://github.com/primeqa/primeqa.

PrimeQA。2023年。Primeqa：开放领域问答工具包。https://github.com/primeqa/primeqa。

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1-67.

科林·拉菲尔（Colin Raffel）、诺姆·沙泽尔（Noam Shazeer）、亚当·罗伯茨（Adam Roberts）、凯瑟琳·李（Katherine Lee）、沙兰·纳朗（Sharan Narang）、迈克尔·马泰纳（Michael Matena）、周燕琪（Yanqi Zhou）、李伟（Wei Li）和彼得·J·刘（Peter J. Liu）。2020年。用统一的文本到文本转换器探索迁移学习的极限。《机器学习研究杂志》，21(140):1 - 67。

Stephen E. Robertson and Hugo Zaragoza. 2009. The probabilistic relevance framework: BM25 and beyond. Found. Trends Inf. Retr., 3(4):333-389.

斯蒂芬·E·罗伯逊（Stephen E. Robertson）和雨果·萨拉戈萨（Hugo Zaragoza）。2009年。概率相关性框架：BM25及其他。《信息检索基础与趋势》，3(4):333 - 389。

Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. 2022. Col-BERTv2: Effective and efficient retrieval via lightweight late interaction. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3715-3734, Seattle, United States. Association for Computational Linguistics.

凯沙夫·桑塔南姆（Keshav Santhanam）、奥马尔·哈塔卜（Omar Khattab）、乔恩·萨德 - 法尔孔（Jon Saad-Falcon）、克里斯托弗·波茨（Christopher Potts）和马特·扎哈里亚（Matei Zaharia）。2022年。Col - BERTv2：通过轻量级后期交互实现高效检索。《2022年计算语言学协会北美分会人类语言技术会议论文集》，第3715 - 3734页，美国西雅图。计算语言学协会。

Philip Sun. 2020. Announcing scann: Efficient vector similarity search. Google AI Blog.

菲利普·孙（Philip Sun）。2020年。宣布推出SCANN：高效向量相似度搜索。谷歌人工智能博客。

Yi Tay, Vinh Tran, Mostafa Dehghani, Jianmo Ni, Dara Bahri, Harsh Mehta, Zhen Qin, Kai Hui, Zhe Zhao, Jai Gupta, et al. 2022. Transformer memory as a differentiable search index. Advances in Neural Information Processing Systems, 35:21831-21843.

易·泰（Yi Tay）、阮文（Vinh Tran）、莫斯塔法·德赫加尼（Mostafa Dehghani）、倪建谟（Jianmo Ni）、达拉·巴里（Dara Bahri）、哈什·梅塔（Harsh Mehta）、秦震（Zhen Qin）、凯·许（Kai Hui）、赵哲（Zhe Zhao）、贾伊·古普塔（Jai Gupta）等。2022年。将Transformer内存作为可微搜索索引。《神经信息处理系统进展》，35:21831 - 21843。

Nandan Thakur, Nils Reimers, Andreas Rücklé, Ab-hishek Srivastava, and Iryna Gurevych. 2021. Beir: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2).

南丹·塔库尔（Nandan Thakur）、尼尔斯·赖默斯（Nils Reimers）、安德里亚斯·吕克莱（Andreas Rücklé）、阿比舍克·斯里瓦斯塔瓦（Ab - hishek Srivastava）和伊琳娜·古列维奇（Iryna Gurevych）。2021年。BEIR：信息检索模型零样本评估的异构基准。《第三十五届神经信息处理系统数据集和基准赛道会议（第二轮）》。

Ellen M. Voorhees and Dawn M. Tice. 2000. The TREC- 8 question answering track. In Proceedings of the Second International Conference on Language Resources and Evaluation (LREC'00), Athens, Greece. European Language Resources Association (ELRA).

艾伦·M·沃里斯（Ellen M. Voorhees）和道恩·M·泰斯（Dawn M. Tice）。2000年。TREC - 8问答赛道。《第二届语言资源与评估国际会议（LREC'00）论文集》，希腊雅典。欧洲语言资源协会（ELRA）。

Canwen Xu, Daya Guo, Nan Duan, and Julian McAuley. 2022. Laprador: Unsupervised pretrained dense retriever for zero-shot text retrieval. In Findings of the Association for Computational Linguistics: ACL 2022, pages 3557-3569.

徐灿文（Canwen Xu）、郭达亚（Daya Guo）、段楠（Nan Duan）和朱利安·麦考利（Julian McAuley）。2022年。Laprador：用于零样本文本检索的无监督预训练密集检索器。《计算语言学协会研究成果：ACL 2022》，第3557 - 3569页。