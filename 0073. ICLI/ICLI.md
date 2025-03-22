# BERT-based Dense Intra-ranking and Contextualized Late Interaction via Multi-task Learning for Long Document Retrieval

# 基于BERT的密集内部排序和通过多任务学习实现的上下文延迟交互用于长文档检索

Minghan Li

李明翰

Univ. Grenoble Alpes, CNRS, LIG

格勒诺布尔阿尔卑斯大学，法国国家科学研究中心，格勒诺布尔信息学实验室

Grenoble, France

法国，格勒诺布尔

minghan.li@univ-grenoble-alpes.fr

Eric Gaussier

埃里克·高斯耶

Univ. Grenoble Alpes, CNRS, LIG

格勒诺布尔阿尔卑斯大学，法国国家科学研究中心，格勒诺布尔信息学实验室

Grenoble, France

法国，格勒诺布尔

eric.gaussier@imag.fr

## Abstract

## 摘要

Combining query tokens and document tokens and inputting them to pre-trained transformer models like BERT, an approach known as interaction-based, has shown state-of-the-art effectiveness for information retrieval. However, the computational complexity of this approach is high due to the online self-attention computation. In contrast, dense retrieval methods in representation-based approaches are known to be efficient, however less effective. A tradeoff between the two is reached with late interaction methods like ColBERT, which attempt to benefit from both approaches: con-textualized token embeddings can be pre-calculated over BERT for fine-grained effective interaction while preserving efficiency. However, despite its success in passage retrieval, it's not straightforward to use this approach for long document retrieval. In this paper, we propose a cascaded late interaction approach using a single model for long document retrieval. Fast intra-ranking by dot product is used to select relevant passages, then fine-grained interaction of pre-stored token embeddings is used to generate passage scores which are aggregated to the final document score. Multi-task learning is used to train a BERT model to optimize both a dot product and a fine-grained interaction loss functions. Our experiments reveal that the proposed approach obtains near state-of-the-art level effectiveness while being efficient on such collections as TREC 2019.

将查询词元和文档词元组合并输入到像BERT这样的预训练Transformer模型中，这种被称为基于交互的方法，在信息检索方面已显示出最先进的有效性。然而，由于在线自注意力计算，这种方法的计算复杂度很高。相比之下，基于表示的方法中的密集检索方法以高效著称，但效果较差。像ColBERT这样的延迟交互方法在两者之间取得了平衡，它试图从这两种方法中受益：可以在BERT上预先计算上下文词元嵌入，以实现细粒度的有效交互，同时保持效率。然而，尽管它在段落检索中取得了成功，但将这种方法用于长文档检索并不直接。在本文中，我们提出了一种使用单一模型的级联延迟交互方法用于长文档检索。通过点积进行快速内部排序来选择相关段落，然后使用预先存储的词元嵌入进行细粒度交互以生成段落得分，这些得分汇总为最终的文档得分。使用多任务学习来训练一个BERT模型，以优化点积和细粒度交互损失函数。我们的实验表明，所提出的方法在TREC 2019等数据集上既高效又能达到接近最先进水平的有效性。

## CCS CONCEPTS

## 计算机协会概念分类

- Information systems $\rightarrow$ Retrieval models and ranking.

- 信息系统 $\rightarrow$ 检索模型和排序。

## KEYWORDS

## 关键词

Neural IR, BERT-based models, late interaction, multi-task learning

神经信息检索，基于BERT的模型，延迟交互，多任务学习

## ACM Reference Format:

## ACM引用格式：

Minghan Li and Eric Gaussier. 2022. BERT-based Dense Intra-ranking and Contextualized Late Interaction via Multi-task Learning for Long Document Retrieval. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '22), July 11-15, 2022, Madrid, Spain. ACM, New York, NY, USA, 6 pages. https://doi.org/10.1145/3477495.3531856

李明翰和埃里克·高斯耶。2022年。基于BERT的密集内部排序和通过多任务学习实现的上下文延迟交互用于长文档检索。收录于第45届ACM信息检索研究与发展国际会议论文集（SIGIR '22），2022年7月11 - 15日，西班牙马德里。美国纽约州纽约市ACM，6页。https://doi.org/10.1145/3477495.3531856

## 1 INTRODUCTION

## 1 引言

Information retrieval (IR) plays an important role in our daily life in the era of big data. Retrieving relevant documents given a query is the central part of many applications in our daily life, e.g., web search. Deep neural networks have shown great success on a variety of tasks including information retrieval $\left\lbrack  {3,5,{12},{22},{27}}\right\rbrack$ with pre-trained Transformer [26] architectures like BERT [4] leading to state-of-the-art performances [2, 15-17, 20, 22]. BERT-based neural IR approaches can be classified into three categories [15]: interaction-based methods, representation-based methods and late-interaction methods. This first method , like a vanilla BERT model [22], where the query tokens and document tokens are concatenated as BERT inputs and applied full self-attention, is viewed to be extremely effective [7] but suffers from high computational complexity. On the other hand, representation-based methods generate two representations [24] for a query and a document respectively. When the document representations can be pre-stored, this method enables efficient fast retrieval at the expense effectiveness. To take advantage of both approaches, late-interaction methods have been proposed, ColBERT [15] being certainly the most well-known representative of this category. In ColBERT, token level passage em-beddings are pre-stored, which are then late interacted with query embeddings to produce a relevance score. This method is slightly more efficient than representation-based methods, but definitely more effective.

在大数据时代，信息检索（Information retrieval，IR）在我们的日常生活中扮演着重要角色。根据查询请求检索相关文档是我们日常生活中许多应用的核心部分，例如网络搜索。深度神经网络在包括信息检索 $\left\lbrack  {3,5,{12},{22},{27}}\right\rbrack$ 在内的各种任务中取得了巨大成功，基于预训练的Transformer [26] 架构（如BERT [4]）的模型实现了最先进的性能 [2, 15 - 17, 20, 22]。基于BERT的神经信息检索方法可分为三类 [15]：基于交互的方法、基于表示的方法和后期交互方法。第一种方法，如普通的BERT模型 [22]，将查询词元（token）和文档词元连接起来作为BERT的输入，并应用全自注意力机制，被认为极其有效 [7]，但计算复杂度较高。另一方面，基于表示的方法分别为查询和文档生成两种表示 [24]。当文档表示可以预先存储时，这种方法能够以牺牲一定有效性为代价实现高效的快速检索。为了兼顾两种方法的优势，人们提出了后期交互方法，ColBERT [15] 无疑是这一类方法中最著名的代表。在ColBERT中，词元级别的段落嵌入被预先存储，然后与查询嵌入进行后期交互以生成相关性得分。这种方法比基于表示的方法效率略高，且效果明显更好。

ColBERT was primarily used for passage ranking and, as most BERT methods, suffers from the major drawback that it cannot directly handle long documents. Although researchers $\left\lbrack  {9,{16} - {18}}\right\rbrack$ has proposed some methods for long document information retrieval, they are designed for interaction-based methods that are computational expensive. So far, there have been no attempts to adapt late interaction methods to long documents.

ColBERT主要用于段落排序，并且和大多数BERT方法一样，存在一个主要缺点，即它无法直接处理长文档。尽管研究人员 $\left\lbrack  {9,{16} - {18}}\right\rbrack$ 已经提出了一些用于长文档信息检索的方法，但这些方法是为计算成本较高的基于交互的方法设计的。到目前为止，还没有人尝试将后期交互方法应用于长文档。

We address this problem here through a BERT-based dense intra-ranking and contextualized late interaction (ICLI) with multi-task learning. Efficiency is guaranteed by the pre-calculation of self attention, and effectiveness by the fact that pre-stored token embeddings are interacted in a fine-grained way. To the best of our knowledge, this is the first attempt to adapt a late interaction method for long document retrieval. Experimental results show that the proposed approach obtains near SOTA level effectiveness while being efficient on such collections as TREC 2019.

我们在这里通过基于BERT的密集内部排序和上下文后期交互（ICLI）以及多任务学习来解决这个问题。通过预先计算自注意力机制来保证效率，通过以细粒度方式对预先存储的词元嵌入进行交互来保证效果。据我们所知，这是首次尝试将后期交互方法应用于长文档检索。实验结果表明，所提出的方法在TREC 2019等数据集上实现了接近最先进水平的效果，同时具有较高的效率。

## 2 RELATED WORK

## 2 相关工作

As already mentioned, neural IR can be classified into three types: interaction-based, representation-based [6] and late interaction-based methods. The first effective interaction-based neural IR approaches, as DRMM [5], KNRM [27] or Conv-KNRM [3], were proposed before the introduction of BERT. After transformer-based BERT models were proposed, the field of neural IR has seen rapid improvements. Nogueira and Cho [22] proposed a simple usage of BERT for passage re-ranking where the query and passage tokens are concatenated and processed by the BERT and the [CLS] output of BERT is used to assess the relevance, and got results that outperformed other neural IR models largely. Hofstätter et al. [11] introduced the TK model which relies on transformers to learn contextualized embeddings and kernel matching for ranking. Early representation-based models as DSSM [12] were appealing because of their extremely low latency. These models have also benefited from BERT-based architectures, leading to so-called dense retrieval models [24, 28]. Despite their efficiency, representation-based models are however less effective than interaction-based models. A trade-off between the two approaches is realized with late interaction methods like ColBERT [15], which relies first on separate representations for queries and documents and which approximates the effectiveness of interaction-based methods in a late interaction step. It places the high latency passage processing by BERT to offline stage and the contextualized token embeddings are stored, which enable fine-grained late interaction. ColBERT is slightly less efficient than dense retrieval methods, but more effective. However, as other BERT-based models, this model cannot directly handle long documents, due to the complexity of the self-attention step.

如前所述，神经信息检索可分为三种类型：基于交互的方法、基于表示的方法 [6] 和基于后期交互的方法。在引入BERT之前，就已经提出了一些有效的基于交互的神经信息检索方法，如DRMM [5]、KNRM [27] 或Conv - KNRM [3]。在基于Transformer的BERT模型提出后，神经信息检索领域取得了快速进展。Nogueira和Cho [22] 提出了一种简单的BERT用于段落重排序的方法，即将查询和段落词元连接起来，由BERT进行处理，并使用BERT的 [CLS] 输出评估相关性，其结果大大优于其他神经信息检索模型。Hofstätter等人 [11] 引入了TK模型，该模型依靠Transformer学习上下文嵌入和核匹配进行排序。早期基于表示的模型（如DSSM [12]）因其极低的延迟而具有吸引力。这些模型也受益于基于BERT的架构，从而产生了所谓的密集检索模型 [24, 28]。尽管基于表示的模型效率较高，但它们的效果不如基于交互的模型。通过像ColBERT [15] 这样的后期交互方法实现了两种方法之间的权衡，该方法首先依赖于查询和文档的单独表示，并在后期交互步骤中近似基于交互方法的效果。它将BERT的高延迟段落处理放在离线阶段，并存储上下文词元嵌入，从而实现细粒度的后期交互。ColBERT的效率略低于密集检索方法，但效果更好。然而，与其他基于BERT的模型一样，由于自注意力步骤的复杂性，该模型无法直接处理长文档。

---

<!-- Footnote -->

classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.SIGIR '22, July 11-15, 2022, Madrid, Spain. © 2022 Association for Computing Machinery.

允许在课堂上免费使用，前提是不得为了盈利或商业利益而制作或分发副本，并且副本必须带有此声明和完整的引用信息。允许在注明出处的情况下进行摘要。如需以其他方式复制、重新发布、发布到服务器或分发给列表，则需要事先获得特定许可和/或支付费用。请向permissions@acm.org请求许可。SIGIR '22，2022年7月11 - 15日，西班牙马德里。© 2022美国计算机协会。

<!-- Footnote -->

---

To deal with long document retrieval, a variety of methods have been proposed, e.g., modified attention methods like Long-former and QDS-Transformer $\left\lbrack  {1,{13}}\right\rbrack$ ,sliding window and local relevance aggregation like TKL [10], passage representation aggregation methods like PARADE [16], and the recent selecting key block/passage for evaluation methods like KeyBLD and IDCM $\left\lbrack  {9,{17},{18}}\right\rbrack$ . Recently,the KeyBLD model family $\left\lbrack  {{17},{18}}\right\rbrack$ ,that first filters a long document by selecting key blocks on which to ground the document relevance, has shown SOTA level performance while also being memory efficient. In parallel, the IDCM model [9] was proposed, the core idea of which is also to first select key passages on which to ground the document relevance. In this paper, we introduce a late interaction method for long document retrieval based on the same idea but in a different way that enables both effectiveness and efficiency. The details are described in Section 3.

为了处理长文档检索问题，人们已经提出了多种方法，例如，像Long - former和QDS - Transformer $\left\lbrack  {1,{13}}\right\rbrack$ 这样的改进注意力方法，像TKL [10] 这样的滑动窗口和局部相关性聚合方法，像PARADE [16] 这样的段落表示聚合方法，以及最近用于评估的选择关键块/段落的方法，如KeyBLD和IDCM $\left\lbrack  {9,{17},{18}}\right\rbrack$ 。最近，KeyBLD模型族 $\left\lbrack  {{17},{18}}\right\rbrack$ 展示了最优（SOTA）水平的性能，同时还具有内存效率，该模型族首先通过选择关键块来过滤长文档，以此确定文档相关性。与此同时，IDCM模型 [9] 被提出，其核心思想同样是首先选择关键段落来确定文档相关性。在本文中，我们基于相同的思想但以不同的方式引入了一种用于长文档检索的后期交互方法，该方法兼具有效性和效率。具体细节将在第3节中描述。

## 3 METHOD

## 3 方法

As mentioned above, the selecting key block related methods [9, 17, 18] have been shown to achieve SOTA level effectiveness for long document information retrieval. Nevertheless, these models have been designed for interaction-based models which have high computational complexity due to their online self-attention computation and not good solutions for real world online search scenarios where low latency is crucial. We propose to extend them to late interaction retrieval methods for long documents by using a cascaded ranking approach based on dense intra-ranking and late interaction. However, designing this is non-trival for being both effective and efficient. In the following, we firstly introduce the overall architecture and then describe the details and the reason of some choices.

如上所述，与选择关键块相关的方法 [9, 17, 18] 已被证明在长文档信息检索中能够达到最优（SOTA）水平的有效性。然而，这些模型是为基于交互的模型设计的，由于其在线自注意力计算，这些模型具有较高的计算复杂度，并且对于低延迟至关重要的现实世界在线搜索场景而言，没有很好的解决方案。我们提议通过使用基于密集内部排序和后期交互的级联排序方法，将它们扩展到长文档的后期交互检索方法中。然而，要设计出既有效又高效的方法并非易事。接下来，我们首先介绍整体架构，然后描述一些选择的细节和原因。

The overall architecture of the proposed method is depicted in Fig. 1 and described later.

所提出方法的整体架构如图1所示，并将在后面进行描述。

<!-- Media -->

<!-- figureText: fine-grained select first passage and other top ranking passages for late interaction FFN2 BERT BERT query -->

<img src="https://cdn.noedgeai.com/0195a598-8158-7879-a4b4-aaf9b80eb638_1.jpg?x=929&y=232&w=733&h=466&r=0"/>

Figure 1: The architecture of proposed approach. Contextu-alized embeddings are calculated by the BERT model and a feedforward neural network for late interaction. The [CLS] embedding is also inputted to another feedforward neural network for passage ranking. The selected passages' late interaction scores are aggregated to obtain the document relevance score. The document tokens can be pre-stored.

图1：所提出方法的架构。上下文嵌入由BERT模型和用于后期交互的前馈神经网络计算得出。[CLS]嵌入也被输入到另一个用于段落排序的前馈神经网络中。所选段落的后期交互得分被聚合以获得文档相关性得分。文档标记可以预先存储。

<!-- Media -->

### 3.1 Contextualized Document Embedding

### 3.1 上下文文档嵌入

The ColBERT model's efficiency comes from its offline pre-computed contextualized passage token embeddings. Here, we want to also rely on the pre-stored contextualized document embeddings to enable fast retrieval. Due to the quadratic complexity of self-attention mechanism, transformer based models including BERT can only handle limited number of tokens, so they are not able to process long documents directly. To tackle this, following previous work $\left\lbrack  {9,{16},{17}}\right\rbrack$ ,we firstly segment a long document into passages,which can be inputted to a BERT model separately to obtain contextual-ized embeddings. During training, the BERT model can be learned end-to-end and enables storing contextualized embeddings.

ColBERT模型的效率源于其离线预计算的上下文段落标记嵌入。在这里，我们也希望依靠预先存储的上下文文档嵌入来实现快速检索。由于自注意力机制的二次复杂度，包括BERT在内的基于Transformer的模型只能处理有限数量的标记，因此它们无法直接处理长文档。为了解决这个问题，遵循先前的工作 $\left\lbrack  {9,{16},{17}}\right\rbrack$ ，我们首先将长文档分割成段落，这些段落可以分别输入到BERT模型中以获得上下文嵌入。在训练过程中，BERT模型可以进行端到端学习，并能够存储上下文嵌入。

Given a document $D$ ,we segment it into passages ${P}_{1}{P}_{2}\ldots {P}_{k}$ , which can be done in a sliding window way. For each passage, with a BERT tokenizer,we can obatin its tokens ${p}_{1}{p}_{2}\ldots {p}_{m}$ ,then BERT's [CLS] token is prepended to the tokens of each passage. The BERT model will compute the contextualized token embeddings ${E}_{p}^{cls}{E}_{p}^{1}\ldots {E}_{p}^{m}$ ,where each ${E}_{p}$ is of dimension ${768}\left( {\dim \left( {{E}_{p} = {768}}\right) }\right)$ for a bert-base-uncased model (as shown in the right part of Fig. 1).

给定一个文档 $D$ ，我们将其分割成段落 ${P}_{1}{P}_{2}\ldots {P}_{k}$ ，这可以通过滑动窗口的方式完成。对于每个段落，使用BERT分词器，我们可以获得其标记 ${p}_{1}{p}_{2}\ldots {p}_{m}$ ，然后将BERT的[CLS]标记添加到每个段落的标记之前。BERT模型将计算上下文标记嵌入 ${E}_{p}^{cls}{E}_{p}^{1}\ldots {E}_{p}^{m}$ ，其中对于bert - base - uncased模型，每个 ${E}_{p}$ 的维度为 ${768}\left( {\dim \left( {{E}_{p} = {768}}\right) }\right)$ （如图1的右半部分所示）。

Each embedding ${E}_{p}$ is then passed into a one-layer feedforward neural network ${FF}{N}_{1}$ ,referred to as compressor1 (the blue module in Fig. 1), to obtain a low dimensional vector for late interaction: ${V}_{p} = {FF}{N}_{1}\left( {E}_{p}\right)$ ,where $\dim \left( {V}_{p}\right)  = {128}$ . It is worth mentioning that the [CLS] embedding is also passed to a one-layer feedforward neural network ${FF}{N}_{2}$ ,referred to as compressor2 (the green module in Fig. 1), for dense intra-ranking of the passages in a document. This is to say, ${V}_{p}^{{cl}{s}_{1}} = {FF}{N}_{1}\left( {E}_{p}^{cls}\right) ,{V}_{p}^{{cl}{s}_{2}} = {FF}{N}_{2}\left( {E}_{p}^{cls}\right)$ with,for each ${V}_{p}^{cls},\dim \left( {V}_{p}^{cls}\right)  = {128}$ . The choice of using two compressors is based on the fact that a vector for intra-ranking should contain query/passage representation information while a vector participating into late interaction should be trained together with other tokens in a different way. Using two compressed vectors allows one to capture these differences and gives more flexibility.

然后，将每个嵌入向量${E}_{p}$输入到一个单层前馈神经网络${FF}{N}_{1}$（称为压缩器1，即图1中的蓝色模块）中，以获得用于后期交互的低维向量：${V}_{p} = {FF}{N}_{1}\left( {E}_{p}\right)$，其中$\dim \left( {V}_{p}\right)  = {128}$。值得一提的是，[CLS]嵌入向量也会被输入到一个单层前馈神经网络${FF}{N}_{2}$（称为压缩器2，即图1中的绿色模块）中，用于对文档中的段落进行密集的内部排序。也就是说，对于每个${V}_{p}^{cls},\dim \left( {V}_{p}^{cls}\right)  = {128}$，有${V}_{p}^{{cl}{s}_{1}} = {FF}{N}_{1}\left( {E}_{p}^{cls}\right) ,{V}_{p}^{{cl}{s}_{2}} = {FF}{N}_{2}\left( {E}_{p}^{cls}\right)$。使用两个压缩器的选择基于这样一个事实：用于内部排序的向量应包含查询/段落表示信息，而参与后期交互的向量应与其他标记以不同的方式一起训练。使用两个压缩向量可以捕捉这些差异，并提供更大的灵活性。

Online computation is used during training. During deployment, the contextualized tokens of each document are pre-computed and stored for efficient late interaction.

训练期间使用在线计算。在部署过程中，会预先计算并存储每个文档的上下文标记，以实现高效的后期交互。

### 3.2 Contextualized Query Embedding

### 3.2 上下文查询嵌入

Similar to a passage in a document, BERT's [CLS] token is prepended to the tokenized query tokens, which are passed to the BERT model to obtain contextualized query token embeddings ${E}_{q}^{cls}{E}_{q}^{1}\ldots {E}_{q}^{n}$ ,with, for each ${E}_{q},\dim \left( {E}_{q}\right)  = {768}$ . Then using ${FF}{N}_{1}$ ,one obtains a low dimensional vector ${V}_{q}$ for each ${E}_{q}$ . For the [CLS] token, ${V}_{q}^{{cl}{s}_{1}}$ and ${V}_{q}^{{cl}{s}_{2}}$ are also calculated.

与文档中的段落类似，BERT的[CLS]标记会被添加到分词后的查询标记之前，然后将这些标记输入到BERT模型中，以获得上下文查询标记嵌入向量${E}_{q}^{cls}{E}_{q}^{1}\ldots {E}_{q}^{n}$，对于每个${E}_{q},\dim \left( {E}_{q}\right)  = {768}$。然后，使用${FF}{N}_{1}$，为每个${E}_{q}$获得一个低维向量${V}_{q}$。对于[CLS]标记，还会计算${V}_{q}^{{cl}{s}_{1}}$和${V}_{q}^{{cl}{s}_{2}}$。

Different from documents, the query is always computed online, which is also the same case for ColBERT and dense retrieval models. Nevertheless, the computational cost is relatively small as the queries are shorter and only required to be computed once to retrieve different documents.

与文档不同，查询总是在线计算的，这与ColBERT和密集检索模型的情况相同。不过，由于查询较短，并且只需要计算一次就能检索不同的文档，因此计算成本相对较小。

### 3.3 Intra-ranking for Key Passage Filtering

### 3.3 关键段落过滤的内部排序

The token embeddings of a long document can be pre-stored, however, they are normally too long which may result in high latency and may contain noisy information. Previous works $\left\lbrack  {9,{18}}\right\rbrack$ first select key passages according to the query to make it more efficient and even more accurate. BM25 and learning based methods can be used where the later normally performs better as it enables semantic matching. In the case of late interaction, we also want to use this step and to use learning based method for informative passage selection. To rely on pre-stored embeddings, inspired by dense retrieval where a query and a passage are represented by a low dimensional vector respectively, normally using the [CLS] embedding, we want to rely on this which allows us to do semantic matching and is naturally part of the BERT model. To do so, the [CLS] embedding of a passage or a query from BERT is inputted to a compresser layer 2 (the ${FF}{N}_{2}$ module in Fig. 1),to obtain the representation of a passage. Dot product is used during inference, to select the most informative chunks, with the pre-stored passage representations and the online query representation.

长文档的标记嵌入向量可以预先存储，然而，它们通常太长，这可能会导致高延迟，并且可能包含噪声信息。先前的工作$\left\lbrack  {9,{18}}\right\rbrack$首先根据查询选择关键段落，以提高效率，甚至提高准确性。可以使用BM25和基于学习的方法，其中基于学习的方法通常表现更好，因为它可以实现语义匹配。在后期交互的情况下，我们也希望使用这一步骤，并使用基于学习的方法进行信息段落选择。为了依赖预先存储的嵌入向量，受密集检索的启发，在密集检索中，查询和段落分别由一个低维向量表示，通常使用[CLS]嵌入向量，我们希望依赖于此，这使我们能够进行语义匹配，并且它自然是BERT模型的一部分。为此，将来自BERT的段落或查询的[CLS]嵌入向量输入到压缩器层2（图1中的${FF}{N}_{2}$模块）中，以获得段落的表示。在推理过程中，使用点积，结合预先存储的段落表示和在线查询表示，来选择最具信息性的块。

As the first passage of a document usually carries important information, as illustrated in [18], we always select this passage in our approach. This strategy is also consistent with truncation-based methods which truncate long documents and rely only on their beginning in order to apply BERT-based models. Given the representation ${V}_{q}^{{cl}{s}_{2}}$ of a given query and pre-stored representations ${V}_{p}^{{cl}{s}_{2}}$ of passages in a given document,we use the standard dot product to score passages:

正如文献[18]所示，文档的第一段通常包含重要信息，因此在我们的方法中，我们总是选择这一段。这种策略也与基于截断的方法一致，这些方法会截断长文档，并仅依赖文档的开头来应用基于BERT的模型。给定一个给定查询的表示${V}_{q}^{{cl}{s}_{2}}$和一个给定文档中段落的预先存储的表示${V}_{p}^{{cl}{s}_{2}}$，我们使用标准点积对段落进行评分：

$$
{S}_{q,p}^{1} = {V}_{q}^{{cl}{s}_{2}} \cdot  {\left( {V}_{p}^{{cl}{s}_{2}}\right) }^{T}. \tag{1}
$$

This selection process provides "top"- $k$ passages for each query-document pair. Having the first passage in the "top"- $k$ list furthermore allows one to train compressor2 through a standard ranking loss, as described below. During training, this step is eval model for PyTorch, which means no backpropagation.

该选择过程为每个查询 - 文档对提供“顶级” - $k$ 段落。“顶级” - $k$ 列表中的第一段还允许我们通过标准排序损失来训练压缩器2，如下所述。在训练期间，此步骤是PyTorch的评估模型，这意味着不进行反向传播。

The above approach finally amounts to learning a representation useful for selecting passages and is in line with previous work [18] that has shown that learning how to select key blocks in a document can outperform methods that select key blocks using standard ranking functions as BM25 [25].

上述方法最终相当于学习一种对选择段落有用的表示，并且与先前的工作 [18] 一致，该工作表明，学习如何选择文档中的关键块可以优于使用标准排序函数（如BM25 [25]）选择关键块的方法。

### 3.4 Fine-grained Late Interaction

### 3.4 细粒度后期交互

After having selected the $k$ passages,we adopt,for each passage,the late interaction approach of ColBERT to obtain the query-passage relevance score:

在选择了 $k$ 段落之后，我们针对每个段落采用ColBERT的后期交互方法来获得查询 - 段落相关性得分：

$$
{S}_{q,p}^{2} = \mathop{\sum }\limits_{{i \in  \left\lbrack  {{cl}{s}_{1},1\ldots n}\right\rbrack  }}\mathop{\max }\limits_{{j \in  \left\lbrack  {{cl}{s}_{1},1\ldots m}\right\rbrack  }}{V}_{q}^{i} \cdot  {\left( {V}_{p}^{j}\right) }^{T}. \tag{2}
$$

These scores are then simply aggregated through a weighted sum:

然后通过加权求和简单地聚合这些得分：

$$
{S}_{q,d} = {w}_{1}{S}_{q,{p}_{1}}^{2} + \ldots  + {w}_{k}{S}_{q,{p}_{k}}^{2}, \tag{3}
$$

where the weights $\left\{  {{w}_{1},\ldots ,{w}_{k}}\right\}$ are real numbers.

其中权重 $\left\{  {{w}_{1},\ldots ,{w}_{k}}\right\}$ 是实数。

### 3.5 Multi-task Learning

### 3.5 多任务学习

Since the [CLS] contextualized vectors are used for two tasks: intra passage ranking (Section 3.3) and late interaction (Section 3.4), we adopt a multi-task learning approach to ensure that both tasks are taken into account to fine-tune the BERT model. As mentioned before, (a) the first passage of a document usually contains relevant information and is always selected, (b) this first passage is furthermore always used by relatively strong baselines based on truncation, and (c) its use ensures the training of the first task. Indeed, we train the [CLS] vector to be used for intra-passage ranking using the first passage of documents as explained below. The second task directly relies on the scores of the selected passages (Eq. 2) and the document labels. The loss functions associated with these tow tasks are given in the following section.

由于 [CLS] 上下文向量用于两个任务：段落内排序（第3.3节）和后期交互（第3.4节），我们采用多任务学习方法来确保在微调BERT模型时同时考虑这两个任务。如前所述，(a) 文档的第一段通常包含相关信息，并且总是被选中；(b) 基于截断的相对较强的基线方法总是使用这第一段；(c) 使用它可以确保第一个任务的训练。实际上，如下所述，我们使用文档的第一段来训练用于段落内排序的 [CLS] 向量。第二个任务直接依赖于所选段落的得分（公式2）和文档标签。与这两个任务相关的损失函数将在以下部分给出。

### 3.6 Loss Functions

### 3.6 损失函数

Following [9], we use the pairwise RankNet loss for each task, defined by:

遵循文献 [9]，我们为每个任务使用成对的RankNet损失，定义如下：

$$
\mathcal{L}\left( {q,{d}^{ + },{d}^{ - };\Theta }\right)  =  - \log \left( {\operatorname{Sigmoid}\left( {{S}_{q,{d}^{ + }} - {S}_{q,{d}^{ - }}}\right) }\right) ,
$$

where $q$ is a query, $\left( {{d}_{q}^{ + },{d}_{q}^{ - }}\right)$ is a positive and negative training document pair for $q,\Theta$ represents the parameters of the model and ${S}_{q,d}$ is the score provided by the model for document $d$ . For task 1, the loss function is given by:

其中 $q$ 是一个查询，$\left( {{d}_{q}^{ + },{d}_{q}^{ - }}\right)$ 是 $q,\Theta$ 的一个正训练文档对和负训练文档对，${S}_{q,d}$ 表示模型的参数，$d$ 是模型为文档 $d$ 提供的得分。对于任务1，损失函数如下：

$$
{\mathcal{L}}_{1}\left( {q,{d}^{ + },{d}^{ - };\Theta }\right)  = \operatorname{RankNet}\left( {q,{S}_{q,{p}_{1 + }}^{1},{S}_{q,{p}_{1 - }}^{1}}\right) ,
$$

where ${S}_{q,{p}_{1 + }}^{1}$ (resp. ${S}_{q,{p}_{1 - }}^{1}$ ) is the relevance score of the first passage of a positive (resp. negative) document for query $q$ according to Eq. 1. For task 2, the loss is given by:

其中 ${S}_{q,{p}_{1 + }}^{1}$（分别地，${S}_{q,{p}_{1 - }}^{1}$）是根据公式1，正（分别地，负）文档的第一段与查询 $q$ 的相关性得分。对于任务2，损失如下：

$$
{\mathcal{L}}_{2}\left( {q,{d}^{ + },{d}^{ - };\Theta }\right)  = \operatorname{RankNet}\left( {q,{S}_{q,{d}_{ + }},{S}_{q,{d}_{ - }}}\right) ,
$$

where ${S}_{q,{d}_{ + }}$ (resp. ${S}_{q,{d}_{ - }}$ ) is the relevance score of a positive (resp. negative) document for query $q$ according to Eq. 3.

其中 ${S}_{q,{d}_{ + }}$（分别地，${S}_{q,{d}_{ - }}$）是根据公式3，正（分别地，负）文档与查询 $q$ 的相关性得分。

As the two losses may have different scales, we combine them to obtain the final loss $\mathcal{L}\left( {q,{d}^{ + },{d}^{ - };\Theta }\right)$ following [19],which adjusts the proposal of [14] to enforce positive regularization:

由于这两个损失可能具有不同的尺度，我们根据文献 [19] 将它们组合起来以获得最终损失 $\mathcal{L}\left( {q,{d}^{ + },{d}^{ - };\Theta }\right)$，该文献调整了文献 [14] 的提议以实施正正则化：

$$
\mathcal{L} = \frac{1}{2{\sigma }_{1}^{2}}{\mathcal{L}}_{1} + \frac{1}{2{\sigma }_{2}^{2}}{\mathcal{L}}_{2} + \log \left( {1 + {\sigma }_{1}^{2}}\right)  + \log \left( {1 + {\sigma }_{2}^{2}}\right) ,
$$

where ${\sigma }_{1}$ and ${\sigma }_{2}$ are parameters of the model.

其中 ${\sigma }_{1}$ 和 ${\sigma }_{2}$ 是模型的参数。

## 4 EXPERIMENTS

## 4 实验

In this section, we evaluate the proposed approach for both effectiveness and efficiency. Besides, an ablation study is performed to analyze the model.

在本节中，我们从有效性和效率两方面评估所提出的方法。此外，还进行了消融研究以分析该模型。

### 4.1 Datasets

### 4.1 数据集

We make use here of the widely used test collection TREC 2019 Deep Learning Track Document Collection. We use two test sets: TREC 2019 and 2020 test queries, where the task is to rerank top 100 documents for each test query. The first test set is widely studied $\left\lbrack  {9,{13},{18}}\right\rbrack$ on MS MARCO v1 corpus. We here also evaluate it on MS MARCO v2 corpus ${}^{1}$ ,which is said to be larger,cleaner and more realistic. To be specific, for TREC 2019 and 2020 test set, we use the same official data with MS MARCO v1 corpus, where the training set is 'msmarco-doctrain-top100', validation set is ' msmarco-docdev-top100'. For TREC 2019 test set, we also additionally evaluate it on the MS MARCO v2 corpus, where the training set is 'docv2_train_top100', validation set is 'docv2_dev_top100'.

我们在此使用广泛应用的测试集TREC 2019深度学习赛道文档集（TREC 2019 Deep Learning Track Document Collection）。我们使用两个测试集：TREC 2019和2020测试查询集，其任务是对每个测试查询的前100篇文档进行重排序。第一个测试集在MS MARCO v1语料库上得到了广泛研究$\left\lbrack  {9,{13},{18}}\right\rbrack$。我们在此还在MS MARCO v2语料库${}^{1}$上对其进行评估，据说该语料库规模更大、更干净且更贴近实际。具体而言，对于TREC 2019和2020测试集，我们使用与MS MARCO v1语料库相同的官方数据，其中训练集为“msmarco - doctrain - top100”，验证集为“msmarco - docdev - top100”。对于TREC 2019测试集，我们还额外在MS MARCO v2语料库上对其进行评估，其中训练集为“docv2_train_top100”，验证集为“docv2_dev_top100”。

### 4.2 Baseline Models

### 4.3 基线模型

The proposed approach is compared with 5 baselines:

将所提出的方法与5个基线模型进行比较：

- BM25: we use the BM25 [25] implementation of Anserini [29], with default hyperparameters;

- BM25：我们使用Anserini [29]实现的BM25 [25]，采用默认超参数；

- BERT-CAT: this is a BERT interaction-based [22] baseline;

- BERT - CAT：这是一个基于BERT交互的[22]基线模型；

- TK: this model [11] generates contextualized embeddings using Transformer and does kernel matching.

- TK：该模型[11]使用Transformer生成上下文嵌入并进行核匹配。

- TKL: this model [10] extends TK model for long documents;

- TKL：该模型[10]对TK模型进行扩展以处理长文档；

- ColBERT: this is the SOTA late interaction model [15] for passage ranking.

- ColBERT：这是用于段落排序的最先进的后期交互模型[15]。

In addition, we also propose to use BM25 to do intra-ranking to select passages. The other steps are unchanged but we do not need multi-task learning in this case as there is no learning of the selection module. Only late interaction loss, ${\mathcal{L}}_{2}$ is required. In this setting, we use the BM25 model proposed in [17] which has provided very good results for block selection on several collections.

此外，我们还提议使用BM25进行内部排序以选择段落。其他步骤保持不变，但在这种情况下我们不需要多任务学习，因为不需要对选择模块进行学习。只需要后期交互损失${\mathcal{L}}_{2}$。在这种设置下，我们使用[17]中提出的BM25模型，该模型在多个数据集的块选择任务中取得了很好的效果。

### 4.3 Experimental Settings

### 4.3 实验设置

Our implementation is based on the matchmaker open-source framework ${}^{2}$ ,using automatic mixed precision [21]. For the MS MARCO v1 corpus, the goal is to rerank the official top 100 documents. For the MS MARCO v2 corpus, as TREC 2019 DL has no official top retrieved documents, we rely on Anserini [29] with default hyperparameters to obtain top 100 documents which are then used for reranking. All neural-IR models, except TKL, are trained for 40000 steps where each step contains 8 positive-negative pairs. For TKL, each step is set to 32 pairs and is trained for 10000 steps. All models are trainied using the pairwise RankNet loss. The negative documents in the positive-negative pairs are sampled randomly, from each query's official top 100 documents that have no label. We conduct 10 validations during training to store the best performing model.

我们的实现基于matchmaker开源框架${}^{2}$，采用自动混合精度[21]。对于MS MARCO v1语料库，目标是对官方前100篇文档进行重排序。对于MS MARCO v2语料库，由于TREC 2019 DL没有官方的前检索文档，我们依靠Anserini [29]并使用默认超参数来获取前100篇文档，然后对这些文档进行重排序。除TKL外，所有神经信息检索（neural - IR）模型都训练40000步，每步包含8个正负样本对。对于TKL，每步设置为32个样本对，训练10000步。所有模型都使用成对的RankNet损失进行训练。正负样本对中的负文档是从每个查询的官方前100篇无标签文档中随机采样得到的。我们在训练过程中进行10次验证以保存性能最佳的模型。

For BERT-based models, the learning rate for BERT is set to 1.0e-05, the other learning rates are set to 1.0e-3. TK and TKL use 1.0e-4, the learning rate for kernels is 1.0e-3. Adam optimizer is used to train the models. For BERT-CAT, TK and ColBERT, we only consider the first 400 tokens; indeed, as these tokens are concatenated with the query, this ensures that one does not exceed the maximum allowed size of 512 tokens in BERT (note that previous studies, as [10], only considered 200 tokens for these kind models). The BERT model used for queries and documents is shared. The maximum document length is set to 3000 tokens, while the passage length is set to 200 tokens (without overlap as we did not see significant difference during preliminary experiments). This means that a document can split into a maximum of 30 passages (when the document length is shorter than 3000 , the obtained passages can be less). For TKL, we set the passage size to 40 , with an overlap of 10 , as done in the original paper. Both TK and TKL use lowercased texts as GloVe [23] embeddings are lowercased version, while BERT based models use original texts (bert-base-uncased model is used and in this case every token will be made lowercase automatically). For the intra-ranking step, we select 4 passages (an extreme case is the original passage number is less than 4 , if so we pad 0 as the scores).

对于基于BERT的模型，BERT的学习率设置为1.0e - 05，其他学习率设置为1.0e - 3。TK和TKL使用1.0e - 4的学习率，核的学习率为1.0e - 3。使用Adam优化器来训练模型。对于BERT - CAT、TK和ColBERT，我们只考虑前400个标记；实际上，由于这些标记会与查询进行拼接，这样可以确保不超过BERT允许的最大标记数512（注意，如[10]等先前的研究对于这类模型只考虑200个标记）。用于查询和文档的BERT模型是共享的。最大文档长度设置为3000个标记，而段落长度设置为200个标记（不进行重叠，因为我们在初步实验中未发现显著差异）。这意味着一篇文档最多可以拆分为30个段落（当文档长度小于3000时，得到的段落数可能更少）。对于TKL，我们将段落大小设置为40，重叠部分为10，与原论文中的设置相同。TK和TKL都使用小写文本，因为GloVe [23]嵌入是小写版本，而基于BERT的模型使用原始文本（使用bert - base - uncased模型，在这种情况下每个标记会自动转换为小写）。在内部排序步骤中，我们选择4个段落（极端情况是原始段落数少于4个，如果是这样，我们将分数填充为0）。

<!-- Media -->

Table 1: Results on TREC 2019 DL collection of MS MARCO v1 and v2 corpus. Best results are in bold.

表1：在MS MARCO v1和v2语料库的TREC 2019 DL数据集上的结果。最佳结果以粗体显示。

<table><tr><td rowspan="2">Model</td><td colspan="2">MS MARCO v1</td><td colspan="2">MS MARCO v2</td></tr><tr><td>NDCG@10</td><td>MAP</td><td>NDCG@10</td><td>MAP</td></tr><tr><td>BM25</td><td>0.5176</td><td>0.2434</td><td>0.2368</td><td>0.0865</td></tr><tr><td>BERT-CAT</td><td>0.6519</td><td>0.2627</td><td>0.3754</td><td>0.1144</td></tr><tr><td>TK</td><td>0.5850</td><td>0.2491</td><td>0.3290</td><td>0.1086</td></tr><tr><td>TKL</td><td>0.6213</td><td>0.2656</td><td>0.3351</td><td>0.1108</td></tr><tr><td>ColBERT</td><td>0.6504</td><td>0.2688</td><td>0.3788</td><td>0.1133</td></tr><tr><td>ICLI-BM25</td><td>0.6806</td><td>0.2703</td><td>0.3926</td><td>0.1160</td></tr><tr><td>ICLI-dot</td><td>0.7048</td><td>0.2768</td><td>0.4049</td><td>0.1146</td></tr></table>

<table><tbody><tr><td rowspan="2">模型</td><td colspan="2">微软机器阅读理解数据集第一版（MS MARCO v1）</td><td colspan="2">微软机器阅读理解数据集第二版（MS MARCO v2）</td></tr><tr><td>前10名归一化折损累积增益（NDCG@10）</td><td>平均准确率均值（MAP）</td><td>前10名归一化折损累积增益（NDCG@10）</td><td>平均准确率均值（MAP）</td></tr><tr><td>二元独立模型（BM25）</td><td>0.5176</td><td>0.2434</td><td>0.2368</td><td>0.0865</td></tr><tr><td>基于BERT的分类模型（BERT - CAT）</td><td>0.6519</td><td>0.2627</td><td>0.3754</td><td>0.1144</td></tr><tr><td>TK</td><td>0.5850</td><td>0.2491</td><td>0.3290</td><td>0.1086</td></tr><tr><td>TKL</td><td>0.6213</td><td>0.2656</td><td>0.3351</td><td>0.1108</td></tr><tr><td>基于上下文的双向编码器表示（ColBERT）</td><td>0.6504</td><td>0.2688</td><td>0.3788</td><td>0.1133</td></tr><tr><td>交互式跨语言索引与BM25结合（ICLI - BM25）</td><td>0.6806</td><td>0.2703</td><td>0.3926</td><td>0.1160</td></tr><tr><td>交互式跨语言索引点积模型（ICLI - dot）</td><td>0.7048</td><td>0.2768</td><td>0.4049</td><td>0.1146</td></tr></tbody></table>

Table 2: Results on TREC 2020 DL dataset, corpus MS MARCO v1. Best results are in bold.

表2：TREC 2020 DL数据集（语料库为MS MARCO v1）的结果。最佳结果以粗体显示。

<table><tr><td>Model</td><td>NDCG@10</td><td>MAP</td></tr><tr><td>BM25</td><td>0.5286</td><td>0.3793</td></tr><tr><td>BERT-CAT</td><td>0.6211</td><td>0.4112</td></tr><tr><td>TK</td><td>0.5732</td><td>0.3660</td></tr><tr><td>TKL</td><td>0.5677</td><td>0.3633</td></tr><tr><td>ColBERT</td><td>0.5951</td><td>0.3907</td></tr><tr><td>ICLI-BM25</td><td>0.5940</td><td>0.3783</td></tr><tr><td>ICLI-dot</td><td>0.6042</td><td>0.3938</td></tr></table>

<table><tbody><tr><td>模型</td><td>前10项归一化折损累积增益（NDCG@10）</td><td>平均准确率均值（MAP）</td></tr><tr><td>二元独立模型25（BM25）</td><td>0.5286</td><td>0.3793</td></tr><tr><td>基于BERT的分类模型（BERT - CAT）</td><td>0.6211</td><td>0.4112</td></tr><tr><td>TK</td><td>0.5732</td><td>0.3660</td></tr><tr><td>TKL</td><td>0.5677</td><td>0.3633</td></tr><tr><td>列BERT（ColBERT）</td><td>0.5951</td><td>0.3907</td></tr><tr><td>交互式跨语言索引 - 二元独立模型25（ICLI - BM25）</td><td>0.5940</td><td>0.3783</td></tr><tr><td>交互式跨语言索引 - 点积（ICLI - dot）</td><td>0.6042</td><td>0.3938</td></tr></tbody></table>

<!-- Media -->

To learn both the complete BERT model and the aggregation scores (Eq. 3),we first fix the aggregation scores to $\left\lbrack  {{0.4},{0.3},{0.2},{0.1}}\right\rbrack$ and learn BERT, prior to fix BERT parameters and adjust the aggregation scores. The choice for the initial values for the aggregation scores is arbitrary and simply reflects the fact that passages with higher relevance scores are more important for deciding the relevance of the entire document.

为了同时学习完整的BERT模型和聚合分数（公式3），我们首先将聚合分数固定为$\left\lbrack  {{0.4},{0.3},{0.2},{0.1}}\right\rbrack$并学习BERT，然后再固定BERT参数并调整聚合分数。聚合分数初始值的选择是任意的，这仅仅反映了一个事实，即相关性分数较高的段落对于确定整个文档的相关性更为重要。

Finally, results are reported according to NDCG@10 and MAP, two standard metrics on the collections considered.

最后，根据NDCG@10和MAP这两个针对所考虑的数据集的标准指标来报告结果。

### 4.4 Results

### 4.4 结果

The results obtained on the TREC 2019 DL test collection with MS MARCO v1 and v2 are shown in Table 1. With MS MARCO v1, the proposed approach ICLI with dot product for passage filtering achieves SOTA results: for NDCG@10, it reaches 0.7048 , which is ${8.36}\%$ higer than ColBERT. With MS MARCO v2,it also obtains the best result on NDCG@10.Using BM25 to filter passages for late interaction, ICLI-BM25 is the second best method on MS MARCO v1, the best method on MAP and second best on NDCG@10 on MS MARCO v2. This proves that the proposed approach is effective compared to other late interaction methods.

在使用MS MARCO v1和v2的TREC 2019 DL测试集上获得的结果如表1所示。使用MS MARCO v1时，提出的采用点积进行段落过滤的ICLI方法取得了最优结果（SOTA）：对于NDCG@10，其达到了0.7048，比ColBERT高${8.36}\%$。使用MS MARCO v2时，它在NDCG@10上也取得了最佳结果。使用BM25对段落进行过滤以进行后期交互时，ICLI - BM25在MS MARCO v1上是第二好的方法，在MS MARCO v2上在MAP指标上是最佳方法，在NDCG@10上是第二好的方法。这证明了与其他后期交互方法相比，所提出的方法是有效的。

---

<!-- Footnote -->

${}^{1}$ https://microsoft.github.io/msmarco/TREC-Deep-Learning-2021

${}^{1}$ https://microsoft.github.io/msmarco/TREC-Deep-Learning-2021

${}^{2}$ https://github.com/sebastian-hofstaetter/matchmaker

${}^{2}$ https://github.com/sebastian-hofstaetter/matchmaker

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: Different positions' ratios of NDCG@10 with position1 TREC 2019 TREC 2020 -->

<img src="https://cdn.noedgeai.com/0195a598-8158-7879-a4b4-aaf9b80eb638_4.jpg?x=249&y=235&w=511&h=392&r=0"/>

Figure 2: The NDCG@10 result of different positions compared with the first position.

图2：不同位置与第一个位置相比的NDCG@10结果。

<!-- Media -->

The results obtained on the TREC 2020 DL test collection are displayed in Table 2. As once can see, BERT-CAT using only the first 400 tokens is the best performing method, while the proposed approach is only slightly better than ColBERT. A similar trend is observed on the TK and TKL models as TK, with only the first 400 tokens, obtains better results than TKL with 3000 tokens. These results are consistent with the findings reported in [8]: the authors also observed that for TKL, the 2K model outperforms the 4K model on TREC 2020.

在TREC 2020 DL测试集上获得的结果如表2所示。可以看出，仅使用前400个标记的BERT - CAT是性能最佳的方法，而所提出的方法仅比ColBERT略好。在TK和TKL模型上也观察到了类似的趋势，因为仅使用前400个标记的TK比使用3000个标记的TKL取得了更好的结果。这些结果与文献[8]中报告的发现一致：作者还观察到，对于TKL，在TREC 2020上2K模型的性能优于4K模型。

To better understand what's happening on the TREC 2020 test collection, we display in Fig. 2, for both TREC 2019 and TREC 2020, the ratio of the NDCG@10 score of the ${\mathrm{i}}^{th}$ (from 1 to 8) passage of a document. To be specific, we use each passage's relevance score as the document relevance score to calculate the NDCG@10 with the label. Each passage's relevance score with the query is obtained using Sentence-BERT [24] pre-trained on MS MARCO passage collection ${}^{3}$ . We consider here the first 8 passages,each passage containing 400 tokens. As one can note, the relevance information decreases with the position, and, compared to passages in TREC 2019, passages in TREC 2020 tend to be less relevant when their position in the document increases. We believe that this, at least partly, explains the unexpected results observed here and in previous studies on TREC 2020. This said, the ICLI-dot model is still comparable with ColBERT and better than the TK family on this collection.

为了更好地理解在TREC 2020测试集上发生的情况，我们在图2中展示了TREC 2019和TREC 2020中文档的第${\mathrm{i}}^{th}$（从1到8）个段落的NDCG@10分数的比率。具体来说，我们使用每个段落的相关性分数作为文档相关性分数，结合标签来计算NDCG@10。每个段落与查询的相关性分数是使用在MS MARCO段落数据集${}^{3}$上预训练的Sentence - BERT [24]获得的。这里我们考虑前8个段落，每个段落包含400个标记。可以注意到，相关性信息随位置的增加而减少，并且与TREC 2019中的段落相比，TREC 2020中的段落随着其在文档中位置的增加，相关性往往更低。我们认为，这至少部分解释了这里以及之前关于TREC 2020的研究中观察到的意外结果。也就是说，ICLI - dot模型在这个数据集上仍然可与ColBERT相媲美，并且优于TK系列模型。

### 4.5 Reranking Latency

### 4.5 重排序延迟

Following [15], latency is used to measure the average time (in seconds) for loading the pre-stored document vectors from disk to the GPU, for computing the query representation and for applying the fine-grained interaction module to rerank 100 documents for a query on the TREC 2019 MS MARCO v1 test collection (total 43

根据文献[15]，延迟用于衡量将预存储的文档向量从磁盘加载到GPU、计算查询表示以及应用细粒度交互模块对TREC 2019 MS MARCO v1测试集上的一个查询对应的100个文档进行重排序的平均时间（以秒为单位）（总共43

<!-- Media -->

Table 3: Average reranking latencies (seconds) on TREC 2019 ${DL}$ test set,corpus MS MARCO v1 for 100 documents with a query. Best result is in bold.

表3：在TREC 2019 ${DL}$测试集上，针对一个查询对100个文档进行重排序的平均延迟（秒），语料库为MS MARCO v1。最佳结果用粗体显示。

<table><tr><td>Model</td><td>Latency</td></tr><tr><td>BERT-CAT</td><td>1.0234</td></tr><tr><td>TKL</td><td>0.5002</td></tr><tr><td>ICLI-dot</td><td>0.3420</td></tr></table>

<table><tbody><tr><td>模型</td><td>延迟</td></tr><tr><td>伯特 - 猫（BERT - CAT）</td><td>1.0234</td></tr><tr><td>特克尔（TKL）</td><td>0.5002</td></tr><tr><td>国际气候与土地研究所点积法（ICLI - dot）</td><td>0.3420</td></tr></tbody></table>

Table 4: Ablation study on TREC 2019 DL dataset, corpus MS MARCO v1.

表4：在TREC 2019 DL数据集（语料库为MS MARCO v1）上的消融研究。

<table><tr><td>Model</td><td>NDCG@10</td><td>MAP</td></tr><tr><td>Complete</td><td>0.7048</td><td>0.2768</td></tr><tr><td>W/o compressor1</td><td>0.6798</td><td>0.2687</td></tr><tr><td>Compressor2 = compressor1</td><td>0.6975</td><td>0.2719</td></tr><tr><td>Score aggregation: equal weights</td><td>0.6634</td><td>0.2614</td></tr><tr><td>Score aggregation: learned weights</td><td>0.6646</td><td>0.2725</td></tr></table>

<table><tbody><tr><td>模型</td><td>前10项归一化折损累积增益（NDCG@10）</td><td>平均准确率均值（MAP）</td></tr><tr><td>完整的</td><td>0.7048</td><td>0.2768</td></tr><tr><td>无压缩器1</td><td>0.6798</td><td>0.2687</td></tr><tr><td>压缩器2 = 压缩器1</td><td>0.6975</td><td>0.2719</td></tr><tr><td>得分聚合：等权重</td><td>0.6634</td><td>0.2614</td></tr><tr><td>得分聚合：学习到的权重</td><td>0.6646</td><td>0.2725</td></tr></tbody></table>

<!-- Media -->

queries). Average latency results for reranking, on a RTX 8000 GPU, each query's top 100 documents for TREC 2019 test set are reported in Table 3. The average latency for BERT-CAT amounts to 1.0234 , to 0.5002 for TKL, and to 0.3420 for ICLI-dot. This demonstrates that the proposed approach is more efficient than BERT-CAT, by a factor of 3 , and TKL, by a factor of 1.46 .

查询）。表3报告了在RTX 8000 GPU上对TREC 2019测试集的每个查询的前100个文档进行重排序的平均延迟结果。BERT - CAT的平均延迟为1.0234，TKL为0.5002，ICLI - dot为0.3420。这表明所提出的方法比BERT - CAT效率高3倍，比TKL效率高1.46倍。

### 4.6 Ablation Study

### 4.6 消融研究

Lastly, Table 4 reports the results of the ablation study we performed on the ICLI-dot model on MS MARCO v1. We provide in the first line the results obtained with the complete model. The second line corresponds to removing the compressor for intra-ranking module, in which case the model uses the original 768 dimensional [CLS] embeddings (this embedding is also directly learned with task 1). The third line means without compressor2 , in which case the model uses the same compressor for intra-ranking and late interaction. The fourth line corresponds to initializing the aggregation weights (Eq. 3) to $\left\lbrack  {{0.25},{0.25},{0.25},{0.25}}\right\rbrack$ . Lastly,the fifth line shows the results when directly learning the score aggregation weights. As one can note, comparing with the proposed design, the above modifications show lower results, suggesting the importance of using different compressors and validating our choice of learning weights for score aggregation by partly decoupling their learning from the one of the BERT model.

最后，表4报告了我们在MS MARCO v1上对ICLI - dot模型进行的消融研究结果。第一行给出了使用完整模型获得的结果。第二行对应于移除内部排序模块的压缩器，在这种情况下，模型使用原始的768维[CLS]嵌入（此嵌入也通过任务1直接学习）。第三行表示不使用压缩器2，在这种情况下，模型对内部排序和后期交互使用相同的压缩器。第四行对应于将聚合权重（公式3）初始化为$\left\lbrack  {{0.25},{0.25},{0.25},{0.25}}\right\rbrack$。最后，第五行显示了直接学习得分聚合权重时的结果。可以注意到，与所提出的设计相比，上述修改的结果较差，这表明使用不同压缩器的重要性，并验证了我们通过将得分聚合权重的学习与BERT模型的学习部分解耦来学习这些权重的选择。

## 5 CONCLUSION

## 5 结论

In this paper, we have attempted to adapt late interaction methods for long document retrieval by first learning the [CLS] vectors for fast intra-passage ranking, and then by applying late interaction on contextualized token vectors to obtain the fine-grained relevance scores for each selected passage. Score aggregation and multi-task learning methods are furthermore used to combine the various ingredients of our approach. Experimental results demonstrate the efficiency and efficacy of the proposed approach on such collections as TREC 2019.

在本文中，我们尝试通过首先学习[CLS]向量进行快速的段落内排序，然后对上下文标记向量应用后期交互以获得每个选定段落的细粒度相关性得分，来调整后期交互方法以用于长文档检索。此外，还使用得分聚合和多任务学习方法来组合我们方法的各个组成部分。实验结果证明了所提出的方法在TREC 2019等数据集上的效率和有效性。

## ACKNOWLEDGMENTS

## 致谢

This work has been partially supported by MIAI@Grenoble Alpes (ANR-19-P3IA-0003) and the Chinese Scholarship Council (CSC) grant No. 201906960018.

这项工作得到了格勒诺布尔阿尔卑斯人工智能跨学科研究所（MIAI@Grenoble Alpes，项目编号：ANR - 19 - P3IA - 0003）和中国国家留学基金管理委员会（CSC）（资助编号：201906960018）的部分支持。

---

<!-- Footnote -->

${}^{3}$ The msmarco-distilbert-base-v4 version from https://www.sbert.net/docs/pretrained-models/msmarco-v3.html

${}^{3}$ 来自https://www.sbert.net/docs/pretrained - models/msmarco - v3.html的msmarco - distilbert - base - v4版本

<!-- Footnote -->

---

## REFERENCES

## 参考文献

[1] Iz Beltagy, Matthew E Peters, and Arman Cohan. 2020. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150 (2020).

[1] Iz Beltagy、Matthew E Peters和Arman Cohan。2020年。Longformer：长文档Transformer。预印本arXiv：2004.05150（2020年）。

[2] Zhuyun Dai and Jamie Callan. 2019. Deeper text understanding for IR with contextual neural language modeling. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval. 985-988.

[2] 戴珠云（Zhuyun Dai）和Jamie Callan。2019年。通过上下文神经语言建模实现信息检索中更深入的文本理解。收录于第42届ACM SIGIR国际信息检索研究与发展会议论文集。第985 - 988页。

[3] Zhuyun Dai, Chenyan Xiong, Jamie Callan, and Zhiyuan Liu. 2018. Convolutional neural networks for soft-matching n-grams in ad-hoc search. In Proceedings of the eleventh ACM international conference on web search and data mining. 126-134.

[3] 戴珠云（Zhuyun Dai）、熊晨彦（Chenyan Xiong）、Jamie Callan和刘志远（Zhiyuan Liu）。2018年。用于即席搜索中n元组软匹配的卷积神经网络。收录于第十一届ACM国际网络搜索与数据挖掘会议论文集。第126 - 134页。

[4] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. CoRR abs/1810.04805 (2018).

[4] Jacob Devlin、张明伟（Ming - Wei Chang）、Kenton Lee和Kristina Toutanova。2018年。BERT：用于语言理解的深度双向Transformer预训练。预印本CoRR abs/1810.04805（2018年）。

[5] Jiafeng Guo, Yixing Fan, Qingyao Ai, and W Bruce Croft. 2016. A deep relevance matching model for ad-hoc retrieval. In Proceedings of the 25th ACM international on conference on information and knowledge management. 55-64.

[5] 郭佳峰（Jiafeng Guo）、范益兴（Yixing Fan）、艾清瑶（Qingyao Ai）和W Bruce Croft。2016年。用于即席检索的深度相关性匹配模型。收录于第25届ACM国际信息与知识管理会议论文集。第55 - 64页。

[6] Jiafeng Guo, Yixing Fan, Liang Pang, Liu Yang, Qingyao Ai, Hamed Zamani, Chen Wu, W Bruce Croft, and Xueqi Cheng. 2020. A deep look into neural ranking models for information retrieval. Information Processing & Management 57, 6 (2020), 102067.

[6] 郭佳峰（Jiafeng Guo）、范益兴（Yixing Fan）、庞亮（Liang Pang）、刘洋（Liu Yang）、艾清瑶（Qingyao Ai）、Hamed Zamani、吴晨（Chen Wu）、W Bruce Croft和程学旗（Xueqi Cheng）。2020年。信息检索中神经排序模型的深入研究。《信息处理与管理》，第57卷，第6期（2020年），第102067页。

[7] Sebastian Hofstätter, Sophia Althammer, Michael Schröder, Mete Sertkan, and Allan Hanbury. 2020. Improving efficient neural ranking models with cross-architecture knowledge distillation. arXiv preprint arXiv:2010.02666 (2020).

[7] Sebastian Hofstätter、Sophia Althammer、Michael Schröder、Mete Sertkan和Allan Hanbury。2020年。通过跨架构知识蒸馏改进高效神经排序模型。预印本arXiv：2010.02666（2020年）。

[8] Sebastian Hofstätter and Allan Hanbury. 2020. Evaluating Transformer-Kernel Models at TREC Deep Learning 2020. In TREC.

[8] 塞巴斯蒂安·霍夫施塔特（Sebastian Hofstätter）和艾伦·汉伯里（Allan Hanbury）。2020年。在2020年TREC深度学习会议上评估Transformer内核模型。收录于TREC会议论文集。

[9] Sebastian Hofstätter, Bhaskar Mitra, Hamed Zamani, Nick Craswell, and Allan Hanbury. 2021. Intra-document cascading: Learning to select passages for neural document ranking. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. 1349-1358.

[9] 塞巴斯蒂安·霍夫施塔特（Sebastian Hofstätter）、巴斯卡尔·米特拉（Bhaskar Mitra）、哈米德·扎马尼（Hamed Zamani）、尼克·克拉斯韦尔（Nick Craswell）和艾伦·汉伯里（Allan Hanbury）。2021年。文档内级联：学习为神经文档排序选择段落。收录于第44届ACM国际信息检索研究与发展会议论文集。第1349 - 1358页。

[10] Sebastian Hofstätter, Hamed Zamani, Bhaskar Mitra, Nick Craswell, and Allan Hanbury. 2020. Local Self-Attention over Long Text for Efficient Document Retrieval. In Proc. of SIGIR.

[10] 塞巴斯蒂安·霍夫施塔特（Sebastian Hofstätter）、哈米德·扎马尼（Hamed Zamani）、巴斯卡尔·米特拉（Bhaskar Mitra）、尼克·克拉斯韦尔（Nick Craswell）和艾伦·汉伯里（Allan Hanbury）。2020年。用于高效文档检索的长文本局部自注意力机制。收录于SIGIR会议论文集。

[11] Sebastian Hofstätter, Markus Zlabinger, and Allan Hanbury. 2020. Interpretable & Time-Budget-Constrained Contextualization for Re-Ranking. In ECAI 2020. IOS Press, 513-520.

[11] 塞巴斯蒂安·霍夫施塔特（Sebastian Hofstätter）、马库斯·兹拉宾格（Markus Zlabinger）和艾伦·汉伯里（Allan Hanbury）。2020年。用于重排序的可解释且受时间预算约束的上下文建模。收录于2020年欧洲人工智能会议（ECAI 2020）论文集。IOS出版社，第513 - 520页。

[12] Po-Sen Huang, Xiaodong He, Jianfeng Gao, Li Deng, Alex Acero, and Larry Heck. 2013. Learning deep structured semantic models for web search using clickthrough data. In Proceedings of the 22nd ACM international conference on Information & Knowledge Management. 2333-2338.

[12] 黄伯森（Po - Sen Huang）、何晓东（Xiaodong He）、高剑锋（Jianfeng Gao）、邓力（Li Deng）、亚历克斯·阿塞罗（Alex Acero）和拉里·赫克（Larry Heck）。2013年。利用点击数据学习用于网络搜索的深度结构化语义模型。收录于第22届ACM国际信息与知识管理会议论文集。第2333 - 2338页。

[13] Jyun-Yu Jiang, Chenyan Xiong, Chia-Jung Lee, and Wei Wang. 2020. Long Document Ranking with Query-Directed Sparse Transformer. In Findings of the Association for Computational Linguistics: EMNLP 2020. 4594-4605.

[13] 江俊宇（Jyun - Yu Jiang）、熊晨艳（Chenyan Xiong）、李佳蓉（Chia - Jung Lee）和王伟（Wei Wang）。2020年。使用查询导向的稀疏Transformer进行长文档排序。收录于计算语言学协会成果：2020年自然语言处理经验方法会议（EMNLP 2020）。第4594 - 4605页。

[14] Alex Kendall, Yarin Gal, and Roberto Cipolla. 2018. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings of the IEEE conference on computer vision and pattern recognition. 7482-7491.

[14] 亚历克斯·肯德尔（Alex Kendall）、亚林·加尔（Yarin Gal）和罗伯托·奇波拉（Roberto Cipolla）。2018年。利用不确定性权衡场景几何和语义损失的多任务学习。收录于IEEE计算机视觉与模式识别会议论文集。第7482 - 7491页。

[15] Omar Khattab and Matei Zaharia. 2020. Colbert: Efficient and effective passage search via contextualized late interaction over bert. In Proceedings of the 43rd

[15] 奥马尔·哈塔卜（Omar Khattab）和马特·扎哈里亚（Matei Zaharia）。2020年。ColBERT：通过基于BERT的上下文后期交互实现高效有效的段落搜索。收录于第43届

International ACM SIGIR conference on research and development in Information

ACM国际信息检索研究与发展会议

Retrieval. 39-48.

论文集。第39 - 48页。

[16] Canjia Li, Andrew Yates, Sean MacAvaney, Ben He, and Yingfei Sun. 2020. PARADE: Passage representation aggregation for document reranking. arXiv preprint arXiv:2008.09093 (2020).

[16] 李灿佳（Canjia Li）、安德鲁·耶茨（Andrew Yates）、肖恩·麦卡瓦尼（Sean MacAvaney）、何本（Ben He）和孙英飞（Yingfei Sun）。2020年。PARADE：用于文档重排序的段落表示聚合。预印本arXiv:2008.09093 (2020)。

[17] Minghan Li and Eric Gaussier. 2021. KeyBLD: Selecting Key Blocks with Local Pre-ranking for Long Document Information Retrieval. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2207-2211.

[17] 李明翰（Minghan Li）和埃里克·高斯捷（Eric Gaussier）。2021年。KeyBLD：通过局部预排序选择关键块进行长文档信息检索。收录于第44届ACM国际信息检索研究与发展会议论文集。第2207 - 2211页。

[18] Minghan Li, Diana Nicoleta Popa, Johan Chagnon, Yagmur Gizem Cinar, and Eric Gaussier. 2021. The Power of Selecting Key Blocks with Local Pre-ranking for Long Document Information Retrieval. arXiv preprint arXiv:2111.09852 (2021).

[18] 李明翰（Minghan Li）、戴安娜·尼科莱塔·波帕（Diana Nicoleta Popa）、约翰·查格农（Johan Chagnon）、亚格穆尔·吉泽姆·奇纳尔（Yagmur Gizem Cinar）和埃里克·高斯捷（Eric Gaussier）。2021年。通过局部预排序选择关键块进行长文档信息检索的强大能力。预印本arXiv:2111.09852 (2021)。

[19] Lukas Liebel and Marco Körner. 2018. Auxiliary tasks in multi-task learning. arXiv preprint arXiv:1805.06334 (2018).

[19] 卢卡斯·利贝尔（Lukas Liebel）和马尔科·克尔纳（Marco Körner）。2018年。多任务学习中的辅助任务。预印本arXiv:1805.06334 (2018)。

[20] Sean MacAvaney, Andrew Yates, Arman Cohan, and Nazli Goharian. 2019. CEDR: Contextualized embeddings for document ranking. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval. 1101-1104.

[20] 肖恩·麦卡瓦尼（Sean MacAvaney）、安德鲁·耶茨（Andrew Yates）、阿曼·科汉（Arman Cohan）和纳兹利·戈哈里安（Nazli Goharian）。2019年。CEDR：用于文档排序的上下文嵌入。收录于第42届ACM国际信息检索研究与发展会议论文集。第1101 - 1104页。

[21] Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, et al. 2018. Mixed Precision Training. In International Conference on Learning Representations.

[21] 保柳斯·米基凯维丘斯（Paulius Micikevicius）、沙兰·纳朗（Sharan Narang）、乔纳·阿尔本（Jonah Alben）、格雷戈里·迪亚莫斯（Gregory Diamos）、埃里希·埃尔森（Erich Elsen）、大卫·加西亚（David Garcia）、鲍里斯·金斯伯格（Boris Ginsburg）、迈克尔·休斯顿（Michael Houston）、奥列克西·库恰耶夫（Oleksii Kuchaiev）、甘尼什·文卡特什（Ganesh Venkatesh）等。2018年。混合精度训练。收录于国际学习表征会议论文集。

[22] Rodrigo Nogueira and Kyunghyun Cho. 2019. Passage Re-ranking with BERT. arXiv preprint arXiv:1901.04085 (2019).

[22] 罗德里戈·诺盖拉（Rodrigo Nogueira）和赵京焕（Kyunghyun Cho）。2019年。使用BERT进行段落重排序。预印本arXiv:1901.04085 (2019)。

[23] Jeffrey Pennington, Richard Socher, and Christopher D Manning. 2014. Glove: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). 1532-1543.

[23] 杰弗里·彭宁顿（Jeffrey Pennington）、理查德·索舍尔（Richard Socher）和克里斯托弗·D·曼宁（Christopher D Manning）。2014年。GloVe：用于词表示的全局向量。收录于《2014年自然语言处理经验方法会议论文集》（EMNLP）。第1532 - 1543页。

[24] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Foint Conference on Natural Language Processing (EMNLP-IJCNLP). 3982-3992.

[24] 尼尔斯·赖默斯（Nils Reimers）和伊琳娜·古列维奇（Iryna Gurevych）。2019年。Sentence - BERT：使用孪生BERT网络的句子嵌入。收录于《2019年自然语言处理经验方法会议和第9届自然语言处理国际联合会议论文集》（EMNLP - IJCNLP）。第3982 - 3992页。

[25] Stephen E. Robertson and Hugo Zaragoza. 2009. The Probabilistic Relevance Framework: BM25 and Beyond. Found. Trends Inf. Retr. 3, 4 (2009), 333-389.

[25] 斯蒂芬·E·罗伯逊（Stephen E. Robertson）和雨果·萨拉戈萨（Hugo Zaragoza）。2009年。概率相关性框架：BM25及其扩展。《信息检索趋势与基础》3, 4 (2009)，第333 - 389页。

[26] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Proceedings of the 31st International Conference on Neural Information Processing Systems. 6000-6010.

[26] 阿什什·瓦斯瓦尼（Ashish Vaswani）、诺姆·沙泽尔（Noam Shazeer）、尼基·帕尔马（Niki Parmar）、雅各布·乌斯库雷特（Jakob Uszkoreit）、利昂·琼斯（Llion Jones）、艾丹·N·戈麦斯（Aidan N Gomez）、卢卡斯·凯泽（Łukasz Kaiser）和伊利亚·波洛苏金（Illia Polosukhin）。2017年。注意力就是你所需要的一切。收录于《第31届神经信息处理系统国际会议论文集》。第6000 - 6010页。

[27] Chenyan Xiong, Zhuyun Dai, Jamie Callan, Zhiyuan Liu, and Russell Power. 2017. End-to-end neural ad-hoc ranking with kernel pooling. In Proceedings of the 40th International ACM SIGIR conference on research and development in information retrieval. 55-64.

[27] 熊晨彦（Chenyan Xiong）、戴珠云（Zhuyun Dai）、杰米·卡兰（Jamie Callan）、刘志远（Zhiyuan Liu）和拉塞尔·鲍尔（Russell Power）。2017年。使用核池化的端到端神经临时排序。收录于《第40届ACM SIGIR信息检索研究与发展国际会议论文集》。第55 - 64页。

[28] Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N Bennett, Junaid Ahmed, and Arnold Overwijk. 2020. Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval. In International Conference on Learning Representations.

[28] 熊磊（Lee Xiong）、熊晨彦（Chenyan Xiong）、李烨（Ye Li）、邓国峰（Kwok - Fung Tang）、刘佳琳（Jialin Liu）、保罗·N·贝内特（Paul N Bennett）、朱奈德·艾哈迈德（Junaid Ahmed）和阿诺德·奥弗维克（Arnold Overwijk）。2020年。用于密集文本检索的近似最近邻负对比学习。收录于《国际学习表征会议》。

[29] Peilin Yang, Hui Fang, and Jimmy Lin. 2018. Anserini: Reproducible ranking baselines using Lucene. Journal of Data and Information Quality (JDIQ) 10,4 (2018), 1-20.

[29] 杨培林（Peilin Yang）、方慧（Hui Fang）和林吉米（Jimmy Lin）。2018年。Anserini：使用Lucene的可重现排序基线。《数据与信息质量杂志》（JDIQ）10, 4 (2018)，第1 - 20页。