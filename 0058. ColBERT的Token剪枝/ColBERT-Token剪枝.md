# Learned Token Pruning in Contextualized Late Interaction over BERT (ColBERT)

# BERT上下文晚期交互（ColBERT）中的学习型Token剪枝

Carlos Lassance

卡洛斯·拉桑斯（Carlos Lassance）

Naver Labs Europe

欧洲NAVER实验室（Naver Labs Europe）

Meylan, France

法国梅ylan（Meylan）

first.lastatnaverlabs.com

first.lastatnaverlabs.com

Maroua Maachou

马鲁阿·马乔（Maroua Maachou）

Naver Labs Europe

欧洲NAVER实验室（Naver Labs Europe）

Meylan, France

法国梅ylan（Meylan, France）

first.last-internatnaverlabs.com

first.last - internat@naverlabs.com

Joohee Park

朴珠熙（Joohee Park）

Naver

NAVER

Seoul, Korea

韩国首尔（Seoul, Korea）

james.first.lastatnavercorp.com

詹姆斯.名.姓@navercorp.com

Stéphane Clinchant

斯特凡·克兰尚（Stéphane Clinchant）

Naver Labs Europe

欧洲NAVER实验室（Naver Labs Europe）

Meylan, France

法国梅ylan（Meylan）

first.lastatnaverlabs.com

名.姓@naverlabs.com

## Abstract

## 摘要

BERT-based rankers have been shown very effective as rerankers in information retrieval tasks. In order to extend these models to full-ranking scenarios, the ColBERT model has been recently proposed, which adopts a late interaction mechanism. This mechanism allows for the representation of documents to be precomputed in advance. However, the late-interaction mechanism leads to large index size, as one needs to save a representation for each token of every document. In this work, we focus on token pruning techniques in order to mitigate this problem. We test four methods, ranging from simpler ones to the use of a single layer of attention mechanism to select the tokens to keep at indexing time. Our experiments show that for the MS MARCO-passages collection, indexes can be pruned up to ${70}\%$ of their original size,without a significant drop in performance. We also evaluate on the MS MARCO-documents collection and the BEIR benchmark, which reveals some challenges for the proposed mechanism.

基于BERT（Bidirectional Encoder Representations from Transformers，双向编码器表征变换器）的排序器在信息检索任务中作为重排序器已被证明非常有效。为了将这些模型扩展到全排序场景，最近提出了ColBERT模型，该模型采用了后期交互机制。这种机制允许预先计算文档的表示。然而，后期交互机制会导致索引规模较大，因为需要为每个文档的每个词元保存一个表示。在这项工作中，我们专注于词元剪枝技术以缓解这个问题。我们测试了四种方法，从较简单的方法到使用单层注意力机制在索引时选择要保留的词元。我们的实验表明，对于MS MARCO（Microsoft Machine Reading Comprehension，微软机器阅读理解）段落数据集，索引可以被剪枝至其原始大小的${70}\%$，而性能不会显著下降。我们还在MS MARCO文档数据集和BEIR（Benchmarking IR，信息检索基准测试）基准上进行了评估，这揭示了所提出机制面临的一些挑战。

## CCS CONCEPTS

## 计算机与通信安全概念

- Information systems $\rightarrow$ Information retrieval.

- 信息系统 $\rightarrow$ 信息检索。

## KEYWORDS

## 关键词

information retrieval, token pruning, BERT, ColBERT

信息检索，词元剪枝，BERT，ColBERT

## ACM Reference Format:

## ACM引用格式：

Carlos Lassance, Maroua Maachou, Joohee Park, and Stéphane Clinchant. 2022. Learned Token Pruning in Contextualized Late Interaction over BERT (ColBERT). In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '22), July 11-15, 2022, Madrid, Spain. ACM, New York, NY, USA, 5 pages. https://doi.org/10.1145/3477495.3531835

卡洛斯·拉桑斯（Carlos Lassance）、马鲁阿·马乔（Maroua Maachou）、朴珠熙（Joohee Park）和斯特凡·克兰尚（Stéphane Clinchant）。2022年。基于BERT的上下文后期交互中的学习型Token剪枝（ColBERT）。收录于第45届国际计算机协会信息检索研究与发展会议（SIGIR '22）论文集，2022年7月11 - 15日，西班牙马德里。美国纽约州纽约市美国计算机协会，5页。https://doi.org/10.1145/3477495.3531835

## 1 INTRODUCTION

## 1 引言

In recent Information Retrieval (IR) systems, Pretrained Language Models (PLM) [11] have taken the state of the art by storm. Two main families of PLM retrieval methods have been developed:

在最近的信息检索（IR）系统中，预训练语言模型（PLM）[11]彻底改变了现有技术水平。目前已经开发出了两大类PLM检索方法：

(1) Representation-based [13], where both query and documents are encoded separately into a single representation and scoring is performed via distance between representations;

（1）基于表示的方法[13]，该方法将查询和文档分别编码为单一表示，并通过表示之间的距离进行评分；

(2) Interaction-based [12], where a query-document pair is treated jointly by a neural network to generate a score.

（2）基于交互的方法[12]，该方法通过神经网络联合处理查询 - 文档对以生成评分。

On one hand, the former is very efficient as representations of documents can be indexed and only the query has to be computed during inference time. On the other hand, the latter has better performance as it is able to perform a more thorough scoring between queries and documents. In order to bridge the gap between these two families, the ColBERT [9] method indexes a representation per token, which allows to precompute document representations and part of the capability of an interaction model (each contextualized token of the query interacts with each precomputed contextualized token of the document). However, the ColBERT advantage comes with an important cost on the index size, since every token (rather than a pooled version of a document) needs to be indexed.

一方面，前者非常高效，因为文档的表示可以被索引，并且在推理时只需计算查询。另一方面，后者性能更好，因为它能够在查询和文档之间进行更全面的评分。为了弥合这两类方法之间的差距，ColBERT [9]方法为每个词元（token）建立表示索引，这使得可以预先计算文档表示以及交互模型的部分能力（查询的每个上下文词元与文档的每个预先计算的上下文词元进行交互）。然而，ColBERT的优势伴随着索引大小方面的巨大代价，因为每个词元（而不是文档的聚合版本）都需要被索引。

In this work we investigate ColBERT models, by looking into two characteristics of the representations: normalization and query expansion. We then focus on the index size by limiting the amount of tokens to be saved in each document using 4 methods based on i) position of the token, ii) Inverse Document Frequency (IDF) of the token, iii) special tokens, and iv) attention mechanism of the last layer. To empirically evaluate our method, we perform our investigation under the most common benchmark for neural IR (MS MARCO on both passage and document tasks), showing that we are able to greatly improve efficiency (in terms of index size and complexity) while still maintaining acceptable effectiveness (in terms of MRR and Recall).

在这项工作中，我们通过研究表征的两个特征：归一化和查询扩展，来研究ColBERT模型。然后，我们通过使用4种方法限制每个文档中要保存的词元数量来关注索引大小，这4种方法分别基于：i) 词元的位置，ii) 词元的逆文档频率（Inverse Document Frequency，IDF），iii) 特殊词元，以及iv) 最后一层的注意力机制。为了对我们的方法进行实证评估，我们在最常见的神经信息检索（IR）基准（MS MARCO的段落和文档任务）下进行研究，结果表明我们能够在大幅提高效率（从索引大小和复杂度方面衡量）的同时，仍保持可接受的有效性（从平均倒数排名（MRR）和召回率方面衡量）。

## 2 RELATED WORK

## 2 相关工作

Efficient Pretrained LMs: In NLP, there has been a lot of work seeking to improve the efficiency of pretrained LMs. For instance, quantization and distillation have been extensively studied in this context [14]. Closest to our work, Power-BERT [6] and length adaptive transformers [10] have been proposed to reduce the number of FLOPS operations, by eliminating tokens in the PLM layers.

高效预训练语言模型：在自然语言处理（NLP）领域，有很多工作致力于提高预训练语言模型的效率。例如，在这方面，量化和蒸馏已经得到了广泛研究 [14]。与我们的工作最接近的是，Power - BERT [6] 和长度自适应变压器 [10] 被提出通过消除预训练语言模型（PLM）层中的词元来减少浮点运算次数（FLOPS）。

---

<!-- Footnote -->

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. © 2022 Copyright held by the owner/author(s). Publication rights licensed to ACM. https://doi.org/10.1145/3477495.3531835

允许个人或课堂使用本作品的全部或部分内容制作数字或硬拷贝，无需付费，但前提是这些拷贝不得用于盈利或商业目的，且必须在首页注明此声明和完整引用信息。对于本作品中作者之外其他人拥有版权的部分，必须予以尊重。允许进行带引用的摘要。若要以其他方式复制、重新发布、上传到服务器或分发给列表，则需要事先获得特定许可和/或支付费用。请向permissions@acm.org申请许可。© 2022 版权归所有者/作者所有。出版权已授予美国计算机协会（ACM）。https://doi.org/10.1145/3477495.3531835

<!-- Footnote -->

---

Index pruning in IR:. Pruning indexes is a traditional method in Information Retrieval to reduce latency and memory requirements and has been studied thoroughly in many contributions (e.g [21] and [2]), as far as we are aware this is the first application to PLM token-level indexing.

信息检索（IR）中的索引剪枝：剪枝索引是信息检索（Information Retrieval）中减少延迟和内存需求的传统方法，许多研究（例如[21]和[2]）已对其进行了深入研究。据我们所知，这是首次将其应用于预训练语言模型（PLM）的词元级索引。

Improving ColBERT Efficiency. One way to mitigate the problem of large index size of ColBERT is to reduce the dimensionality and apply quantization to the document tokens. Note that this is already done by default as the output of the PLM (most of the time of 768 dimensions) is projected into a smaller space (128 dimensions). In the original paper, the authors show very good performance by both reducing the dimensionality and quantizing the tokens [9]. In this sense, a binarization technique [20] has been proposed for information retrieval, and concurrently with this work, experiments have shown that it is indeed possible to binarize ColBERT tokens [15].

提高ColBERT效率。缓解ColBERT索引规模过大问题的一种方法是降低维度并对文档标记（token）进行量化。请注意，默认情况下已经进行了此操作，因为预训练语言模型（PLM）的输出（大多数情况下为768维）会被投影到一个更小的空间（128维）中。在原论文中，作者通过降低维度和量化标记展示了非常好的性能[9]。从这个意义上说，已经提出了一种二值化技术[20]用于信息检索，并且与本工作同时进行的实验表明，确实有可能对ColBERT标记进行二值化[15]。

The present work is orthogonal to the previously presented research direction, in that we aim to reduce the index size by removing tokens from the index, instead of reducing their size. We consider the combination of those research directions as necessary to improve ColBERT models, but leave it as future work. Lastly, ColBERT has been extended to perform pseudo relevance feedback in [19] and query pruning has been studied to improve latency in [17].

本工作与之前提出的研究方向相互独立，因为我们的目标是通过从索引中移除标记来减小索引大小，而不是减小标记的尺寸。我们认为结合这些研究方向对于改进ColBERT模型是必要的，但将其留作未来的工作。最后，文献[19]将ColBERT扩展以执行伪相关反馈，文献[17]研究了查询剪枝以改善延迟问题。

## 3 METHODOLOGY AND COLBERT

## 3 方法与ColBERT

The ColBERT model is based on a transformer [3] encoder for documents and queries. Each item $Q$ or $D$ is encoded into a representation $\mathbf{R}$ which is a matrix of dimensions(t,d)where $t$ is the amount of tokens in the item and $d$ is the encoding dimension of each token. The score of a document(D)for a given query(Q)is given by the following formula:

ColBERT模型基于一个用于文档和查询的Transformer [3]编码器。每个项目$Q$或$D$都被编码为一个表示$\mathbf{R}$，它是一个维度为(t,d)的矩阵，其中$t$是项目中的标记数量，$d$是每个标记的编码维度。给定查询(Q)下文档(D)的得分由以下公式给出：

$$
s\left( {D,Q}\right)  = \mathop{\sum }\limits_{{j = 0}}^{\left| {t}_{Q}\right| }\mathop{\max }\limits_{{i = 0}}^{\left| {t}_{D}\right| }{\left( {\mathbf{R}}_{D} \cdot  {\mathbf{R}}_{Q}\right) }_{i,j}. \tag{1}
$$

This late-interaction model improves computation efficiency compared to a full-interaction model,because representations ${\mathbf{R}}_{D}$ for all documents $\left( {D \in  \mathcal{D}}\right)$ can be precomputed. In this way,during inference only the query representation needs to be computed. However, this incurs a very large representation size by document (dt) which can quickly become intractable when the amount of documents increases. As those are the only factors impacting index size, there are two solutions: i) reduce the amount of dimensions per token, and ii) reduce the amount of tokens per document. While many solutions exist for the first case, we are not aware of any previous work for the second one.

与全交互模型相比，这种后期交互模型提高了计算效率，因为所有文档 $\left( {D \in  \mathcal{D}}\right)$ 的表示 ${\mathbf{R}}_{D}$ 可以预先计算。这样，在推理过程中，只需要计算查询表示。然而，这会导致每个文档 (dt) 的表示规模非常大，当文档数量增加时，很快就会变得难以处理。由于这些是影响索引规模的唯一因素，因此有两种解决方案：i) 减少每个词元的维度数量；ii) 减少每个文档的词元数量。虽然针对第一种情况有很多解决方案，但我们不知道之前有针对第二种情况的任何工作。

Normalization and query expansion: In an effort to ensure that the scores of each query and document are not dependent on their length, each token is normalized in the 12-hyper-sphere. Also queries are expanded so that they have the same length (32 tokens). We show via experiments that this is not actually needed for all PLM backbones (c.f. Section 4), which reduces the inference cost significantly.

归一化和查询扩展：为了确保每个查询和文档的得分不依赖于它们的长度，每个词元在 12 维超球中进行归一化。此外，查询会进行扩展，使其具有相同的长度（32 个词元）。我们通过实验表明，并非所有预训练语言模型 (PLM) 骨干网络都需要这样做（参见第 4 节），这显著降低了推理成本。

### 3.1 Limiting the amount of tokens

### 3.1 限制词元数量

We investigate four diverse methods for limiting the amount of tokens that are stored for each document. These methods are integrated during the training of our ColBERT models ${}^{1}$ . In other words, we add a pooling layer on top of ColBERT to select a maximum of $k$ tokens per document and then use the ColBERT scoring equation restricted to the $\mathrm{k}$ selected tokens.

我们研究了四种不同的方法来限制为每个文档存储的词元（token）数量。这些方法在我们的ColBERT模型${}^{1}$的训练过程中被整合。换句话说，我们在ColBERT之上添加了一个池化层，以每个文档最多选择$k$个词元，然后使用仅限于$\mathrm{k}$个所选词元的ColBERT评分公式。

First $k$ tokens: One very simple,but also very strong,baseline is to keep only the $k$ first tokens of a document. Indeed such a baseline takes advantage of the inherent bias of the MS MARCO dataset where the first tokens are considered to be the most important [8]. This is the only method where we do not remove punctuation.

前$k$个词元：一种非常简单但也非常有效的基线方法是只保留文档的前$k$个词元。实际上，这样的基线方法利用了MS MARCO数据集的固有偏差，在该数据集中，前几个词元被认为是最重要的[8]。这是唯一一种我们不删除标点符号的方法。

Top $k$ IDF tokens: Another possibility is to choose the rarest tokens of a document, in other words, the ones with the highest Inverse Document Frequency (IDF). This should allow us to keep only the most defining words of a document.

前$k$个逆文档频率（IDF）词元：另一种可能性是选择文档中最罕见的词元，换句话说，就是具有最高逆文档频率（Inverse Document Frequency，IDF）的词元。这应该能让我们只保留文档中最具代表性的词汇。

$k$ -Unused Tokens (UNU). We also test the possibility of adding $k$ special tokens (’unused’ terms from the BERT vocabulary) to the start of the document and keep only those tokens. The kept tokens are always the same for all documents and always in the same positions, which increases the consistency for the model, which we posit leads to easier optimization. However this approach has some drawbacks: i) increased computation time as it forces documents to have at least $k$ tokens,and ii) possibly missing context from long documents as truncation is done with less "real-word" tokens.

$k$ - 未使用的Token（UNU）。我们还测试了在文档开头添加$k$个特殊Token（来自BERT词汇表的“未使用”术语）并仅保留这些Token的可能性。对于所有文档，保留的Token始终相同，并且位置也始终相同，这提高了模型的一致性，我们认为这有助于更轻松地进行优化。然而，这种方法存在一些缺点：i) 由于它要求文档至少包含$k$个Token，因此会增加计算时间；ii) 由于截断操作使用的“真实词汇”Token较少，可能会丢失长文档中的上下文信息。

Attention score (ATT):. We propose to use the last layer of attention of the PLM as a way to detect the most important $k < t$ tokens of the document. Recall that the attention of a document $\left( {A}_{D}\right)$ is a three-dimensional tensor $\left( {h,t,{t}^{\prime }}\right)$ ,where the first dimension is the number of attention heads (number of different "attentions") and the last two dimensions represent the amount of attention a token $t$ "pays" to token ${t}^{\prime }$ ,which is normalized so that for each token $t$ the sum of attentions ${t}^{\prime }$ is 1 . We propose an importance score per token which is the sum of attention payed to the token over all heads:

注意力分数（ATT）：我们提议使用预训练语言模型（PLM）最后一层的注意力机制来检测文档中最重要的 $k < t$ 标记。回顾一下，文档 $\left( {A}_{D}\right)$ 的注意力是一个三维张量 $\left( {h,t,{t}^{\prime }}\right)$，其中第一个维度是注意力头的数量（不同“注意力”的数量），最后两个维度表示标记 $t$ 对标记 ${t}^{\prime }$ 的关注程度，该值经过归一化处理，使得对于每个标记 $t$，其对所有标记 ${t}^{\prime }$ 的注意力之和为 1。我们为每个标记提出了一个重要性分数，即所有注意力头对该标记的注意力之和：

$$
i{\left( t\right) }_{D} = \mathop{\sum }\limits_{{i = 0}}^{h}\mathop{\sum }\limits_{{j = 0}}^{\left| {t}_{D}\right| }{\left( {A}_{D}\right) }_{i,t,j} \tag{2}
$$

So that at the end we only keep $k < t$ tokens,based on the top- $k$ scores. Note that this score is computed over the last layer of the document encoding, and not in the interaction between documents and queries. In other words it is independent from queries.

这样，最后我们根据前 $k$ 个分数只保留 $k < t$ 个标记。请注意，此分数是在文档编码的最后一层计算的，而不是在文档与查询的交互中计算的。换句话说，它与查询无关。

## 4 EXPERIMENTS

## 4 实验

Experimental setup: In this work, we train our ColBERT [9] models using the MINILM-12-384 [18] ${}^{2}$ . Models are trained during 150k steps, with a linear learning rate from 0 to the final learning rate (2e- 5) warmup of ${10}\mathrm{k}$ steps and linear learning rate annealing (2e-5 to 0 ) for the remaining ${140}\mathrm{k}$ steps. We consider both the passage and the document task and use the original triplet files provided by the MS MARCO organizers [1]. We evaluate our models in the full ranking scenario, using a brute force implementation, which differs from the original ColBERT paper, which uses approximate nearest neighbors for that scenario. By not using ANN search, we avoid the need of ANN-hyperparameter tuning and/or the risk of unfair comparisons. Nevertheless, we have experimented with ANN and results are very close to the ones presented in the initial paragraphs. Finally, we train and evaluate "passage" models with a max length of 180 tokens (truncating longer sequences), while "document" models have a max length of 512 .

实验设置：在这项工作中，我们使用MINILM - 12 - 384 [18] ${}^{2}$来训练我们的ColBERT [9]模型。模型训练150k步，学习率采用线性变化，从0开始，在${10}\mathrm{k}$步内预热到最终学习率（2e - 5），并在剩余的${140}\mathrm{k}$步内线性退火（从2e - 5到0）。我们同时考虑了段落和文档任务，并使用MS MARCO组织者提供的原始三元组文件[1]。我们在全排序场景下评估我们的模型，采用暴力实现方法，这与原始的ColBERT论文不同，该论文在该场景中使用近似最近邻方法。不使用ANN搜索，我们避免了对ANN超参数进行调优的需要和/或不公平比较的风险。尽管如此，我们也对ANN进行了实验，结果与前面段落中呈现的结果非常接近。最后，我们训练和评估“段落”模型，其最大长度为180个词元（截断较长的序列），而“文档”模型的最大长度为512。

---

<!-- Footnote -->

${}^{1}$ We tested both during and just after training and noticed that integrating this during training improved the results.

${}^{1}$ 我们在训练期间和训练刚结束时都进行了测试，发现训练期间融入这一操作能改善结果。

${}^{2}$ available at https://huggingface.co/microsoft/MiniLM-L12-H384-uncased

${}^{2}$ 可在https://huggingface.co/microsoft/MiniLM-L12-H384-uncased获取

<!-- Footnote -->

---

Investigating ColBERT modelling: First, we investigate the default ColBERT model and verify if the normalization and query expansion operations are helpful for all PLM. Results for this ablation are presented in Table 1. We observe that when using MINILM instead of BERT there is no need for normalization and expansion. Therefore we can skip these steps which reduces latency (search complexity depends linearly to the amount of tokens in the query). For the rest of this work, we only use models without these operations.

研究ColBERT模型：首先，我们研究了默认的ColBERT模型，并验证归一化和查询扩展操作是否对所有预训练语言模型（PLM）都有帮助。这一消融实验的结果见表1。我们观察到，当使用MiniLM（小型语言模型）而非BERT（双向编码器表示变换器）时，无需进行归一化和扩展操作。因此，我们可以跳过这些步骤，从而减少延迟（搜索复杂度与查询中的词元数量呈线性关系）。在本文的其余部分，我们仅使用不包含这些操作的模型。

<!-- Media -->

Table 1: Ablation of token normalization and query expansion on the MINILM ColBERT model.

表1：MiniLM ColBERT模型上的词元归一化和查询扩展消融实验。

<table><tr><td>Normalization</td><td>Expansion</td><td>MRR@10</td><td>Recall@1000</td></tr><tr><td>X</td><td>X</td><td>0.362</td><td>96.3%</td></tr><tr><td>X</td><td/><td>0.055</td><td>41.2%</td></tr><tr><td/><td>X</td><td>0.363</td><td>97.0%</td></tr><tr><td/><td/><td>0.365</td><td>97.1%</td></tr></table>

<table><tbody><tr><td>归一化（Normalization）</td><td>扩展（Expansion）</td><td>前10项平均倒数排名（MRR@10）</td><td>前1000项召回率（Recall@1000）</td></tr><tr><td>X</td><td>X</td><td>0.362</td><td>96.3%</td></tr><tr><td>X</td><td></td><td>0.055</td><td>41.2%</td></tr><tr><td></td><td>X</td><td>0.363</td><td>97.0%</td></tr><tr><td></td><td></td><td>0.365</td><td>97.1%</td></tr></tbody></table>

<!-- Media -->

Reducing index size on passage dataset: We now evaluate the ability of the 4 proposed token pruning methods (c.f. Section 3.1). Results are displayed in Table 2. We notice that when keeping the top $k = {50}$ ,almost all the tested methods allow us to reduce the amount of tokens by ${30}\%$ while keeping similar acceptable performance (less than 0.01 MRR@10 drop, less than 1% Recall@1000 drop). On the other hand,when we drop to $k = {10}$ there is a noticeable drop in MRR@10 performance for all methods, with Unused Tokens getting the best results, which shows that this technique is very useful for small $k$ but not for larger ones. Note that further tests in other datasets could be useful to verify not only this difference between Unused Tokens and the other methods, but also of the First method that uses the inherent bias of MS MARCO to the start of the document [8]. Note that $\%$ of tokens may vary due to: the size of passages $\left( {k = {50}}\right)$ ; the use or not of punctuation (First); repeated scores (attention and IDF) or because a method increases the original passage length (Unused Tokens).

减少段落数据集的索引大小：我们现在评估4种提出的Token剪枝方法的能力（参见第3.1节）。结果显示在表2中。我们注意到，当保留前$k = {50}$个时，几乎所有测试的方法都能让我们将Token数量减少${30}\%$，同时保持相近的可接受性能（MRR@10下降小于0.01，Recall@1000下降小于1%）。另一方面，当我们减少到$k = {10}$时，所有方法的MRR@10性能都有明显下降，其中未使用Token（Unused Tokens）方法取得了最佳结果，这表明该技术对小的$k$很有用，但对较大的$k$则不然。请注意，在其他数据集上进行进一步测试不仅有助于验证未使用Token方法与其他方法之间的这种差异，还有助于验证利用MS MARCO文档开头固有偏差的第一种方法[8]。请注意，Token的$\%$可能会因以下因素而有所不同：段落大小$\left( {k = {50}}\right)$；是否使用标点符号（第一种方法）；重复得分（注意力和逆文档频率），或者因为某种方法增加了原始段落长度（未使用Token方法）。

Reducing index size on document dataset: In the case of the document task, the mean amount of tokens per document is higher than in the case of passages. Due to this disparity we test only the case of $k = {50}$ ,which allows us to cut ${85}\%$ of the document tokens. While $k = {50}$ provides a noticeable reduction of index size,it still means that the amount of data to keep for every document is large (50d). For example,while the document $\left( {k = {50}}\right)$ relative reduction of index size is the same as passage with $k = {10}$ ,the final index size is still twice as large. Lastly, as it was the case with the smaller token size on the passage task, both Unused Tokens and First present the best results.

减少文档数据集的索引大小：在文档任务中，每个文档的平均Token数量比段落任务中的要多。由于这种差异，我们仅测试了$k = {50}$的情况，这使我们能够削减${85}\%$的文档Token。虽然$k = {50}$能显著减小索引大小，但这仍意味着每个文档需要保留的数据量很大（50维）。例如，虽然文档$\left( {k = {50}}\right)$的索引大小相对缩减幅度与使用$k = {10}$的段落相同，但最终的索引大小仍是其两倍。最后，就像段落任务中Token大小较小时的情况一样，“未使用Token”和“首个Token”这两种方法都取得了最佳效果。

<!-- Media -->

Table 2: Results for different methods and $k$ tokens to keep on MS MARCO. We use MRR@10 for the passage dataset and MRR@100 for the document one. Index size consider $2\mathrm{\;{kb}}$ per vector (128 dimensions in fp16).

表2：在MS MARCO数据集上不同方法和保留$k$个Token的结果。我们对段落数据集使用MRR@10，对文档数据集使用MRR@100。索引大小考虑每个向量的$2\mathrm{\;{kb}}$（fp16格式下为128维）。

<table><tr><td>Method</td><td>% tokens</td><td>Size (Gb)</td><td>MRR</td><td>Recall@1000</td></tr><tr><td>Baseline</td><td>100%</td><td>142</td><td>0.365</td><td>97.1%</td></tr><tr><td colspan="5">Passage $\left( {\mathrm{k} = {50}}\right)$</td></tr><tr><td>First</td><td>72.6%</td><td>103</td><td>0.357</td><td>96.7%</td></tr><tr><td>IDF</td><td>71.9%</td><td>102</td><td>0.355</td><td>96.7%</td></tr><tr><td>Unused Tokens</td><td>74.1%</td><td>101</td><td>0.342</td><td>96.2%</td></tr><tr><td>Attention Score</td><td>71.1%</td><td>105</td><td>0.358</td><td>96.7%</td></tr><tr><td colspan="5">Passage $\left( {\mathrm{k} = {10}}\right)$</td></tr><tr><td>First</td><td>14.8%</td><td>21</td><td>0.302</td><td>93.3%</td></tr><tr><td>IDF</td><td>15.3%</td><td>22</td><td>0.290</td><td>91.0%</td></tr><tr><td>Unused Tokens</td><td>14.8%</td><td>21</td><td>0.314</td><td>94.0%</td></tr><tr><td>Attention Score</td><td>14.8%</td><td>21</td><td>0.281</td><td>91.9%</td></tr><tr><td colspan="5">Document (k=50)</td></tr><tr><td>Baseline</td><td>100%</td><td>290</td><td>0.380</td><td>95.6%</td></tr><tr><td>First</td><td>13.1%</td><td>38</td><td>0.347</td><td>91.6%</td></tr><tr><td>IDF</td><td>13.5%</td><td>39</td><td>0.225</td><td>81.2%</td></tr><tr><td>Unused Tokens</td><td>13.1%</td><td>38</td><td>0.354</td><td>93.1%</td></tr><tr><td>Attention Score</td><td>13.1%</td><td>38</td><td>0.306</td><td>90.9%</td></tr></table>

<table><tbody><tr><td>方法</td><td>% 标记</td><td>大小（吉字节）</td><td>平均倒数排名（Mean Reciprocal Rank）</td><td>召回率@1000</td></tr><tr><td>基线</td><td>100%</td><td>142</td><td>0.365</td><td>97.1%</td></tr><tr><td colspan="5">段落 $\left( {\mathrm{k} = {50}}\right)$</td></tr><tr><td>首先</td><td>72.6%</td><td>103</td><td>0.357</td><td>96.7%</td></tr><tr><td>逆文档频率（IDF）</td><td>71.9%</td><td>102</td><td>0.355</td><td>96.7%</td></tr><tr><td>未使用的Token</td><td>74.1%</td><td>101</td><td>0.342</td><td>96.2%</td></tr><tr><td>注意力得分</td><td>71.1%</td><td>105</td><td>0.358</td><td>96.7%</td></tr><tr><td colspan="5">段落 $\left( {\mathrm{k} = {10}}\right)$</td></tr><tr><td>首先</td><td>14.8%</td><td>21</td><td>0.302</td><td>93.3%</td></tr><tr><td>逆文档频率（IDF）</td><td>15.3%</td><td>22</td><td>0.290</td><td>91.0%</td></tr><tr><td>未使用的Token</td><td>14.8%</td><td>21</td><td>0.314</td><td>94.0%</td></tr><tr><td>注意力得分</td><td>14.8%</td><td>21</td><td>0.281</td><td>91.9%</td></tr><tr><td colspan="5">文档（k = 50）</td></tr><tr><td>基线</td><td>100%</td><td>290</td><td>0.380</td><td>95.6%</td></tr><tr><td>首先</td><td>13.1%</td><td>38</td><td>0.347</td><td>91.6%</td></tr><tr><td>逆文档频率（IDF）</td><td>13.5%</td><td>39</td><td>0.225</td><td>81.2%</td></tr><tr><td>未使用的Token</td><td>13.1%</td><td>38</td><td>0.354</td><td>93.1%</td></tr><tr><td>注意力得分</td><td>13.1%</td><td>38</td><td>0.306</td><td>90.9%</td></tr></tbody></table>

<!-- Media -->

Results on the BEIR benchmark. We now verify how do the networks learned with token pruning generalize outside the MS MARCO domain. To do so, we use the BEIR benchmark [16], using the subset of the 13 datasets that are ready to use ${}^{3}$ . The models we compare to are the standard ColBERT [9], using the numbers reported in [16], our baseline ColBERT, that uses the MINILM PLM and does not perform neither expansion or normalization and the 4 models trained for passage retrieval with $k = {50}$ (i.e. max sequence length of 180 tokens). The results are displayed in Table 3.

BEIR基准测试的结果。我们现在验证通过词元剪枝学习的网络在MS MARCO领域之外的泛化能力。为此，我们使用BEIR基准测试 [16]，采用13个可直接使用的数据集子集 ${}^{3}$。我们对比的模型包括：标准的ColBERT模型 [9]（使用 [16] 中报告的数据）、我们的基线ColBERT模型（该模型使用MINILM预训练语言模型（Pre - trained Language Model，PLM），且不进行扩展或归一化操作），以及使用 $k = {50}$ 为段落检索训练的4个模型（即最大序列长度为180个词元）。结果显示在表3中。

First, we note that all the models trained for this work outperform the original ColBERT [9] on Arguana, which creates some noise on the results. We believe that this comes from the fact that Arguana queries are actually documents and so [9] underperforms due to its query expansion, but we did not delve deeper to validate this claim. Second, we note a slightly larger loss of performance to the one we had previously seen in MS MARCO (around 6% mean nDCG@10 loss on BEIR, compared to 2% MRR@10 on MS MARCO), this comes mostly because the size of the documents is increased for the BEIR datasets (i.e. more compression). However, this is not the only (and maybe not even the major) component of the performance loss, as datasets with similar size to MS MARCO (for example HotpotQA) also present steep losses of performance. Finally, there is a larger gain of performance for the attention method compared to the others than we had seen on MS MARCO, which could possibly show that outside of the inherent bias of MS MARCO for the start of passages [8], such methods could indeed thrive.

首先，我们注意到，为本研究训练的所有模型在Arguana数据集上的表现都优于原始的ColBERT模型[9]，这给结果带来了一些干扰。我们认为，这是因为Arguana数据集中的查询实际上是文档，因此文献[9]中的模型由于其查询扩展机制而表现不佳，但我们并未深入验证这一说法。其次，我们注意到，与之前在MS MARCO数据集上的表现相比，模型在BEIR数据集上的性能损失略大（BEIR数据集上的平均nDCG@10损失约为6%，而MS MARCO数据集上的MRR@10损失为2%），这主要是因为BEIR数据集中文档的规模更大（即需要更多的压缩处理）。然而，这并非性能损失的唯一（甚至可能不是主要）因素，因为与MS MARCO规模相近的数据集（例如HotpotQA）也出现了显著的性能下降。最后，与其他方法相比，注意力机制在性能提升方面比在MS MARCO数据集上更为显著，这可能表明，除了MS MARCO数据集对段落开头存在固有偏差[8]之外，这类方法确实具有良好的应用前景。

---

<!-- Footnote -->

${}^{3}$ The other 5 datasets either have a license that makes it harder to obtain or are not evaluated in the same way as the other 13

${}^{3}$ 另外5个数据集要么有难以获取的许可协议，要么没有像其他13个数据集那样以相同的方式进行评估

<!-- Footnote -->

---

<!-- Media -->

Table 3: NDCG@10 results on BEIR. Bold values are the best in the row for its category (baseline or pruned).

表3：BEIR数据集上的NDCG@10指标结果。加粗的值是该行中对应类别（基线或剪枝）的最佳结果。

<table><tr><td rowspan="2">Dataset</td><td colspan="2">Colbert Baselines</td><td colspan="4">Top-50 pruned, trained on passage</td></tr><tr><td>[9, 16]</td><td>Ours</td><td>First</td><td>IDF</td><td>UNU</td><td>ATT</td></tr><tr><td>ArguAna</td><td>0.233</td><td>0.446</td><td>0.421</td><td>0.424</td><td>0.410</td><td>0.443</td></tr><tr><td>Climate-FEVER</td><td>0.184</td><td>0.168</td><td>0.140</td><td>0.140</td><td>0.123</td><td>0.157</td></tr><tr><td>DBPedia</td><td>0.392</td><td>0.404</td><td>0.375</td><td>0.383</td><td>0.372</td><td>0.382</td></tr><tr><td>FEVER</td><td>0.771</td><td>0.734</td><td>0.660</td><td>0.613</td><td>0.638</td><td>0.654</td></tr><tr><td>FiQA-2018</td><td>0.317</td><td>0.326</td><td>0.265</td><td>0.286</td><td>0.274</td><td>0.306</td></tr><tr><td>HotpotQA</td><td>0.593</td><td>0.631</td><td>0.533</td><td>0.556</td><td>0.515</td><td>0.573</td></tr><tr><td>NFCorpus</td><td>0.305</td><td>0.319</td><td>0.286</td><td>0.259</td><td>0.285</td><td>0.288</td></tr><tr><td>NQ</td><td>0.524</td><td>0.507</td><td>0.475</td><td>0.442</td><td>0.462</td><td>0.490</td></tr><tr><td>Quora</td><td>0.854</td><td>0.850</td><td>0.850</td><td>0.844</td><td>0.807</td><td>0.852</td></tr><tr><td>SCIDOCS</td><td>0.145</td><td>0.150</td><td>0.135</td><td>0.130</td><td>0.138</td><td>0.136</td></tr><tr><td>SciFact</td><td>0.671</td><td>0.633</td><td>0.545</td><td>0.475</td><td>0.502</td><td>0.564</td></tr><tr><td>TREC-COVID</td><td>0.677</td><td>0.715</td><td>0.691</td><td>0.600</td><td>0.665</td><td>0.678</td></tr><tr><td>Touché-2020</td><td>0.202</td><td>0.230</td><td>0.232</td><td>0.209</td><td>0.228</td><td>0.203</td></tr><tr><td>Avg. zero-shot</td><td>0.451</td><td>0.470</td><td>0.431</td><td>0.412</td><td>0.417</td><td>0.440</td></tr></table>

<table><tbody><tr><td rowspan="2">数据集</td><td colspan="2">科尔伯特基线（Colbert Baselines）</td><td colspan="4">前50名筛选，在段落上训练</td></tr><tr><td>[9, 16]</td><td>我们的方法</td><td>第一个</td><td>逆文档频率（IDF）</td><td>联合国大学（UNU）</td><td>美国电话电报公司（ATT）</td></tr><tr><td>论证分析（ArguAna）</td><td>0.233</td><td>0.446</td><td>0.421</td><td>0.424</td><td>0.410</td><td>0.443</td></tr><tr><td>气候事实核查（Climate-FEVER）</td><td>0.184</td><td>0.168</td><td>0.140</td><td>0.140</td><td>0.123</td><td>0.157</td></tr><tr><td>DBpedia知识图谱（DBPedia）</td><td>0.392</td><td>0.404</td><td>0.375</td><td>0.383</td><td>0.372</td><td>0.382</td></tr><tr><td>事实核查（FEVER）</td><td>0.771</td><td>0.734</td><td>0.660</td><td>0.613</td><td>0.638</td><td>0.654</td></tr><tr><td>金融问答数据集2018（FiQA-2018）</td><td>0.317</td><td>0.326</td><td>0.265</td><td>0.286</td><td>0.274</td><td>0.306</td></tr><tr><td>火锅问答数据集（HotpotQA）</td><td>0.593</td><td>0.631</td><td>0.533</td><td>0.556</td><td>0.515</td><td>0.573</td></tr><tr><td>新闻事实语料库（NFCorpus）</td><td>0.305</td><td>0.319</td><td>0.286</td><td>0.259</td><td>0.285</td><td>0.288</td></tr><tr><td>自然问答数据集（NQ）</td><td>0.524</td><td>0.507</td><td>0.475</td><td>0.442</td><td>0.462</td><td>0.490</td></tr><tr><td>奎若问答平台（Quora）</td><td>0.854</td><td>0.850</td><td>0.850</td><td>0.844</td><td>0.807</td><td>0.852</td></tr><tr><td>科学文献数据集（SCIDOCS）</td><td>0.145</td><td>0.150</td><td>0.135</td><td>0.130</td><td>0.138</td><td>0.136</td></tr><tr><td>科学事实（SciFact）</td><td>0.671</td><td>0.633</td><td>0.545</td><td>0.475</td><td>0.502</td><td>0.564</td></tr><tr><td>TREC新冠数据集（TREC-COVID）</td><td>0.677</td><td>0.715</td><td>0.691</td><td>0.600</td><td>0.665</td><td>0.678</td></tr><tr><td>Touché 2020评测任务（Touché-2020）</td><td>0.202</td><td>0.230</td><td>0.232</td><td>0.209</td><td>0.228</td><td>0.203</td></tr><tr><td>平均零样本</td><td>0.451</td><td>0.470</td><td>0.431</td><td>0.412</td><td>0.417</td><td>0.440</td></tr></tbody></table>

Table 4: NDCG@10 results on BEIR. Bold values are the best in the row for the pruned models

表4：BEIR数据集上的NDCG@10结果。加粗值是剪枝模型所在行中的最佳值

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">ColBERT Document Baseline</td><td colspan="4">Top-50 pruned, trained on document</td></tr><tr><td>First</td><td>IDF</td><td>UNU</td><td>ATT</td></tr><tr><td>ArguAna</td><td>0,436</td><td>0,407</td><td>0,434</td><td>0,407</td><td>0,443</td></tr><tr><td>Climate-FEVER</td><td>0,206</td><td>0,152</td><td>0,110</td><td>0,158</td><td>0,133</td></tr><tr><td>DBPedia</td><td>0,405</td><td>0,377</td><td>0,387</td><td>0,363</td><td>0,373</td></tr><tr><td>FEVER</td><td>0,810</td><td>0,655</td><td>0,475</td><td>0,694</td><td>0,631</td></tr><tr><td>FiQA-2018</td><td>0,348</td><td>0,279</td><td>0,271</td><td>0,295</td><td>0,305</td></tr><tr><td>HotpotQA</td><td>0,635</td><td>0,533</td><td>0,554</td><td>0,520</td><td>0,555</td></tr><tr><td>NFCorpus</td><td>0,330</td><td>0,286</td><td>0,230</td><td>0,290</td><td>0,281</td></tr><tr><td>NQ</td><td>0,509</td><td>0,472</td><td>0,438</td><td>0,466</td><td>0,472</td></tr><tr><td>Quora</td><td>0,851</td><td>0,846</td><td>0,847</td><td>0,802</td><td>0,848</td></tr><tr><td>SCIDOCS</td><td>0,152</td><td>0,129</td><td>0,122</td><td>0,131</td><td>0,136</td></tr><tr><td>SciFact</td><td>0,680</td><td>0,533</td><td>0,436</td><td>0,569</td><td>0,542</td></tr><tr><td>TREC-COVID</td><td>0,709</td><td>0,700</td><td>0,561</td><td>0,671</td><td>0,607</td></tr><tr><td>Touché-2020</td><td>0,230</td><td>0,222</td><td>0,171</td><td>0,234</td><td>0,209</td></tr><tr><td>Avg. zero-shot</td><td>0,485</td><td>0,430</td><td>0,387</td><td>0,431</td><td>0,426</td></tr></table>

<table><tbody><tr><td rowspan="2">数据集</td><td rowspan="2">ColBERT文档基线</td><td colspan="4">前50个筛选，在文档上训练</td></tr><tr><td>第一个</td><td>逆文档频率（IDF）</td><td>联合国大学（UNU）</td><td>美国电话电报公司（ATT）</td></tr><tr><td>ArguAna</td><td>0,436</td><td>0,407</td><td>0,434</td><td>0,407</td><td>0,443</td></tr><tr><td>气候事实核查数据集（Climate-FEVER）</td><td>0,206</td><td>0,152</td><td>0,110</td><td>0,158</td><td>0,133</td></tr><tr><td>DBpedia知识图谱（DBPedia）</td><td>0,405</td><td>0,377</td><td>0,387</td><td>0,363</td><td>0,373</td></tr><tr><td>事实核查数据集（FEVER）</td><td>0,810</td><td>0,655</td><td>0,475</td><td>0,694</td><td>0,631</td></tr><tr><td>金融问答数据集2018版（FiQA - 2018）</td><td>0,348</td><td>0,279</td><td>0,271</td><td>0,295</td><td>0,305</td></tr><tr><td>火锅问答数据集（HotpotQA）</td><td>0,635</td><td>0,533</td><td>0,554</td><td>0,520</td><td>0,555</td></tr><tr><td>新闻事实核查语料库（NFCorpus）</td><td>0,330</td><td>0,286</td><td>0,230</td><td>0,290</td><td>0,281</td></tr><tr><td>自然问题数据集（NQ）</td><td>0,509</td><td>0,472</td><td>0,438</td><td>0,466</td><td>0,472</td></tr><tr><td>Quora问答平台</td><td>0,851</td><td>0,846</td><td>0,847</td><td>0,802</td><td>0,848</td></tr><tr><td>科学文献数据集（SCIDOCS）</td><td>0,152</td><td>0,129</td><td>0,122</td><td>0,131</td><td>0,136</td></tr><tr><td>科学事实数据集（SciFact）</td><td>0,680</td><td>0,533</td><td>0,436</td><td>0,569</td><td>0,542</td></tr><tr><td>新冠疫情信息检索评测（TREC-COVID）</td><td>0,709</td><td>0,700</td><td>0,561</td><td>0,671</td><td>0,607</td></tr><tr><td>2020年论证检索评测（Touché-2020）</td><td>0,230</td><td>0,222</td><td>0,171</td><td>0,234</td><td>0,209</td></tr><tr><td>平均零样本</td><td>0,485</td><td>0,430</td><td>0,387</td><td>0,431</td><td>0,426</td></tr></tbody></table>

<!-- Media -->

We also tested the models trained for document retrieval and display the results in Table 4. Compared to passage retrieval, the baseline achieves a better mean performance, while some of the pruned models present reduced average performance (Attention Score and IDF), others gain performance (Unused Token) or keep the same performance (First). The loss of performance of Attention and IDF seems to come from the fact that with more tokens these models tend to repeat words (c.f. Section 5), while on the other hand the gain of the Unused Tokens seems to come from the fact that as the networks can treat more tokens, there is less loss of information from increasing the sequence size (so less of the real sequence is thresholded out of the input). Lastly, it is not surprising that First keeps the same performance as it keeps the same tokens independently of its training and/or max sequence length.

我们还测试了为文档检索训练的模型，并将结果展示在表4中。与段落检索相比，基线模型实现了更好的平均性能，而一些剪枝模型的平均性能有所下降（注意力得分和逆文档频率），其他模型则获得了性能提升（未使用Token）或保持相同性能（首个）。注意力和逆文档频率模型性能下降似乎是因为，随着Token数量增多，这些模型往往会重复用词（参见第5节），而另一方面，未使用Token模型性能提升似乎是因为，由于网络可以处理更多Token，增加序列长度时信息损失更少（因此输入中被阈值过滤掉的真实序列更少）。最后，首个模型保持相同性能并不奇怪，因为无论其训练情况和/或最大序列长度如何，它保留的Token都相同。

Results using ANN. For completeness, we also run retrieval of passages following the approximate nearest neighbor scheme described in the original ColBERT paper [9] and available at: https: //github.com/stanford-futuredata/ColBERT/. For the first stage token retrieval we use a probing of the 128 closests clusters and perform full retrieval on all documents that had tokens in the top $8\mathrm{k}$ of each query token. Results are available in Table 5.

使用人工神经网络（ANN）的结果。为了完整性，我们还按照原始ColBERT论文[9]中描述的近似最近邻方案进行了段落检索，该方案可在以下网址获取：https://github.com/stanford - futuredata/ColBERT/。对于第一阶段的词元检索，我们对最接近的128个聚类进行探测，并对每个查询词元的前$8\mathrm{k}$个词元所在的所有文档进行全量检索。结果见表5。

<!-- Media -->

Table 5: Results for different methods and $k$ tokens to keep on MS MARCO-passage using ANN retrieval

表5：在MS MARCO段落数据集上使用人工神经网络（ANN）检索时不同方法和保留$k$个词元的结果

<table><tr><td>Method</td><td>% tokens</td><td>MRR@10</td><td>Recall@1000</td></tr><tr><td>Baseline</td><td>100%</td><td>0.365</td><td>96.5%</td></tr><tr><td colspan="4">$\mathrm{k} = {50}$</td></tr><tr><td>First</td><td>72.6%</td><td>0.357</td><td>96.0%</td></tr><tr><td>IDF</td><td>71.9%</td><td>0.355</td><td>96.0%</td></tr><tr><td>Attention Score</td><td>71.1%</td><td>0.358</td><td>96.2%</td></tr><tr><td colspan="4">$\mathrm{k} = {10}$</td></tr><tr><td>First</td><td>14.8%</td><td>0.302</td><td>93.2%</td></tr><tr><td>IDF</td><td>15.3%</td><td>0.290</td><td>90.9%</td></tr><tr><td>Attention Score</td><td>14.8%</td><td>0.281</td><td>92.1%</td></tr></table>

<table><tbody><tr><td>方法</td><td>% 标记</td><td>前10名平均倒数排名（MRR@10）</td><td>前1000名召回率（Recall@1000）</td></tr><tr><td>基线</td><td>100%</td><td>0.365</td><td>96.5%</td></tr><tr><td colspan="4">$\mathrm{k} = {50}$</td></tr><tr><td>第一个</td><td>72.6%</td><td>0.357</td><td>96.0%</td></tr><tr><td>逆文档频率（IDF）</td><td>71.9%</td><td>0.355</td><td>96.0%</td></tr><tr><td>注意力得分</td><td>71.1%</td><td>0.358</td><td>96.2%</td></tr><tr><td colspan="4">$\mathrm{k} = {10}$</td></tr><tr><td>第一个</td><td>14.8%</td><td>0.302</td><td>93.2%</td></tr><tr><td>逆文档频率（IDF）</td><td>15.3%</td><td>0.290</td><td>90.9%</td></tr><tr><td>注意力得分</td><td>14.8%</td><td>0.281</td><td>92.1%</td></tr></tbody></table>

<!-- Media -->

Notably, the difference in MRR@10 is negligeable as none of the 7 networks we evaluate shows a difference of more than 0.001 MRR@10.On the other hand, when we look into Recall@1000, differences start to appear as the baseline loses ${0.6}\%$ ,the $k = {50}$ models lose in average ${0.7}\%$ . However for the $k = {10}$ models there is no such loss of performance. This confirms that while results using ANN are very close to those of brute-force, they could have impacted the results, especially if we had applied the same ANN to the datasets from BEIR as the number of tokens (and so the size of clusters) varies from dataset to dataset.

值得注意的是，MRR@10的差异可以忽略不计，因为我们评估的7个网络中，没有一个网络的MRR@10差异超过0.001。另一方面，当我们查看Recall@1000时，差异开始显现，因为基线模型损失了${0.6}\%$，$k = {50}$模型平均损失了${0.7}\%$。然而，对于$k = {10}$模型，并没有出现这样的性能损失。这证实了虽然使用近似最近邻搜索（ANN）的结果与暴力搜索的结果非常接近，但它们可能会影响结果，特别是如果我们将相同的ANN应用于BEIR数据集，因为不同数据集的词元数量（以及聚类大小）各不相同。

## 5 RESULT ANALYSIS

## 5 结果分析

Analysis of indexed documents: We analyzed the tokens selected by the attention mechanism on the document set. One problem we observed is about repetitions: it selects of the same token at different position. For instance, we show two examples below of the selected tokens of two different documents to demonstrate the repetition and the stemming effect (dog vs dogs, cake vs ##cake):

索引文档分析：我们分析了文档集上注意力机制选择的词元。我们观察到的一个问题是重复问题：它会在不同位置选择相同的词元。例如，我们在下面展示了两个不同文档所选词元的两个示例，以说明重复现象和词干提取效果（“dog”与“dogs”、“cake”与“##cake”）：

['[CLS]', 'dog', 'nasal', 'poly', 'removal', 'dog', 'nasal', 'poly', 'removal', 'poly', 'grow', 'body', 'parts', 'dogs', 'nasal', 'poly', 'dog', 'being', 'dog', 'nasal', 'poly', 'dog', 'dog', 'nasal', 'poly', 'removal', 'nasal', 'poly', 'dog', 'poly', 'dog', 'dogs', 'nasal', 'poly', 'dogs', 'hide', '", 'dogs', 'dog', 'nasal', 'growth', 'nasal', 'poly', 'dogs', '", 'dogs', 'nose', 'dogs', 'nose', '[SEP]']

['[CLS]', '狗', '鼻息肉', '摘除', '狗', '鼻息肉', '摘除', '息肉', '生长', '身体部位', '狗', '鼻息肉', '狗', '是', '狗', '鼻息肉', '狗', '狗', '鼻息肉', '摘除', '鼻息肉', '狗', '息肉', '狗', '狗', '鼻息肉', '狗', '隐藏', '狗', '狗', '鼻部生长物', '鼻息肉', '狗', '狗', '鼻子', '狗', '鼻子', '[SEP]']

['[CLS]', 'ba', 'mini', 'cup', '##cake', 'cake', 'mix', 'ba', 'mini', 'cup', '##cake', 'cake', 'mix', 'cup', '##cake', 'sweet', 'ba', 'cake', 'mini', 'cup', '##cake', 'cup', 'cake', 'mini', 'cup', '##cake', 'oven', 'mini', 'cup', '##cake', 'pan', 'mini', 'cake', 'mini', 'cup', '##cake', 'mini', 'cup', 'mini', 'cup', '##cake', 'cup', '##cake', 'cup', '##cake', 'cup', '##cake', 'mini', 'cup', '[SEP]']

['[分类符]', '制作', '迷你', '纸杯蛋糕', '蛋糕预拌粉', '制作', '迷你', '纸杯蛋糕', '蛋糕预拌粉', '纸杯蛋糕', '甜品', '制作', '蛋糕', '迷你', '纸杯蛋糕', '纸杯蛋糕', '迷你', '纸杯蛋糕', '烤箱', '迷你', '纸杯蛋糕', '烤盘', '迷你', '蛋糕', '迷你', '纸杯蛋糕', '迷你', '纸杯', '迷你', '纸杯蛋糕', '纸杯蛋糕', '纸杯蛋糕', '纸杯蛋糕', '迷你', '纸杯', '[分隔符]']

These results indicates that token selection methods should take into account either the repetition problem or model the diversity of the selected tokens, which we leave for future work.

这些结果表明，标记选择方法应考虑重复问题或对所选标记的多样性进行建模，我们将此留待未来研究。

ColBERT relevance: Note that while ColBERT was the state of the art for MS MARCO full ranking during its release it has been overtook by dense $\left\lbrack  {5,7}\right\rbrack$ and sparse $\left\lbrack  4\right\rbrack$ siamese models that have better training procedure, namely: i) hard-negative mining [4, 5, 7], ii) distillation [4, 7], and iii) pretraining [5]. When this work was conducted, these techniques were not yet tested for the ColBERT model, so we did not apply them, but concurrently to this work they have been shown beneficial in a pre-print [15]. Finally, ColBERT has also demonstrated interesting properties such as better zero-shot performance [16] and being able to combine with traditional IR techniques such as PRF [19].

ColBERT相关性：请注意，虽然ColBERT（科尔伯特）在发布时是MS MARCO全排序任务的最优模型，但它已被具有更好训练流程的密集$\left\lbrack  {5,7}\right\rbrack$和稀疏$\left\lbrack  4\right\rbrack$孪生模型所超越，这些模型的训练流程主要包括：i) 难负样本挖掘[4, 5, 7]；ii) 知识蒸馏[4, 7]；iii) 预训练[5]。在开展本研究时，这些技术尚未在ColBERT模型上进行测试，因此我们未应用这些技术，但在本研究进行的同时，一篇预印本论文[15]表明这些技术是有益的。最后，ColBERT还展现出了一些有趣的特性，例如更好的零样本性能[16]，并且能够与传统信息检索技术（如伪相关反馈，PRF）相结合[19]。

## 6 CONCLUSION

## 6 结论

In this work we investigate the ColBERT model and test different methods to reduce its late-interaction complexity and its problematic index size. We first verify that for some PLM (namely MINILM) we do not need to perform normalization of tokens and query augmentation, which thus improve latency. Furthermore, we have also demonstrated that some very simple methods allow to remove 30% of the tokens from the passage index without incurring in any consequent performance loss. On the other hand the MS MARCO-document collection reveals challenges for such mechanism, where even pruning ${90}\%$ of tokens may not be enough reduction. The combination of such token pruning methods with already studied embedding compression methods could lead to further improvement of ColBERT-based IR systems.

在这项工作中，我们研究了ColBERT模型，并测试了不同的方法来降低其后期交互复杂度和解决其索引规模过大的问题。我们首先验证了对于某些预训练语言模型（PLM，即MiniLM），我们无需对词元进行归一化和查询增强，从而提高了响应时间。此外，我们还证明了一些非常简单的方法可以从段落索引中移除30%的词元，而不会导致任何性能损失。另一方面，MS MARCO文档集显示出这种机制面临的挑战，在该文档集中，即使修剪${90}\%$的词元可能也不足以实现缩减。将这些词元修剪方法与已经研究过的嵌入压缩方法相结合，可能会进一步改进基于ColBERT的信息检索（IR）系统。

## REFERENCES

## 参考文献

[1] Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, Mir Rosenberg, Xia Song, Alina Stoica, Saurabh Tiwary, and Tong Wang. 2018. MS MARCO: A Human Generated MAchine Reading COmprehension Dataset. arXiv:1611.09268 [cs.CL]

[1] 帕亚尔·巴贾杰（Payal Bajaj）、丹尼尔·坎波斯（Daniel Campos）、尼克·克拉斯韦尔（Nick Craswell）、李邓（Li Deng）、高剑锋（Jianfeng Gao）、刘旭东（Xiaodong Liu）、兰根·马朱姆德（Rangan Majumder）、安德鲁·麦克纳马拉（Andrew McNamara）、巴斯卡尔·米特拉（Bhaskar Mitra）、特里·阮（Tri Nguyen）、米尔·罗森伯格（Mir Rosenberg）、宋霞（Xia Song）、阿利娜·斯托伊卡（Alina Stoica）、索拉布·蒂瓦里（Saurabh Tiwary）和王彤（Tong Wang）。2018年。MS MARCO：一个人工生成的机器阅读理解数据集。arXiv:1611.09268 [计算机科学 - 计算语言学（cs.CL）]

[2] David Carmel, Doron Cohen, Ronald Fagin, Eitan Farchi, Michael Herscovici, Yoelle S Maarek, and Aya Soffer. 2001. Static index pruning for information retrieval systems. In Proceedings of the 24th annual international ACM SIGIR conference on Research and development in information retrieval. 43-50.

[2] 大卫·卡梅尔（David Carmel）、多伦·科恩（Doron Cohen）、罗纳德·法金（Ronald Fagin）、伊坦·法尔基（Eitan Farchi）、迈克尔·赫斯科维奇（Michael Herscovici）、约埃尔·S·马雷克（Yoelle S Maarek）和阿亚·索弗（Aya Soffer）。2001年。信息检索系统的静态索引剪枝。见第24届ACM SIGIR国际信息检索研究与发展年度会议论文集。第43 - 50页。

[3] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. CoRR abs/1810.04805 (2018). arXiv:1810.04805 http://arxiv.org/abs/1810.04805

[3] 雅各布·德夫林（Jacob Devlin）、张明伟（Ming-Wei Chang）、肯顿·李（Kenton Lee）和克里斯蒂娜·图托纳娃（Kristina Toutanova）。2018年。BERT：用于语言理解的深度双向变换器的预训练。计算机研究报告库（CoRR）abs/1810.04805 (2018)。预印本服务器（arXiv）：1810.04805 http://arxiv.org/abs/1810.04805

[4] Thibault Formal, Carlos Lassance, Benjamin Piwowarski, and Stéphane Clinchant. 2021. SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval. arXiv:2109.10086 [cs.IR]

[4] 蒂博·福尔马尔（Thibault Formal）、卡洛斯·拉桑斯（Carlos Lassance）、本杰明·皮沃瓦尔斯基（Benjamin Piwowarski）和斯特凡·克兰尚（Stéphane Clinchant）。2021年。SPLADE v2：用于信息检索的稀疏词汇与扩展模型。预印本服务器（arXiv）：2109.10086 [计算机科学 - 信息检索（cs.IR）]

[5] Luyu Gao and Jamie Callan. 2021. Unsupervised Corpus Aware Language Model Pre-training for Dense Passage Retrieval. arXiv:2108.05540 [cs.IR]

[5] 高璐宇（Luyu Gao）和杰米·卡兰（Jamie Callan）。2021年。用于密集段落检索的无监督语料感知语言模型预训练。预印本服务器（arXiv）：2108.05540 [计算机科学 - 信息检索（cs.IR）]

[6] Saurabh Goyal, Anamitra Roy Choudhury, Saurabh Raje, Venkatesan Chakar-avarthy, Yogish Sabharwal, and Ashish Verma. 2020. PoWER-BERT: Accelerating

[6] 索拉布·戈亚尔（Saurabh Goyal）、阿纳米特拉·罗伊·乔杜里（Anamitra Roy Choudhury）、索拉布·拉杰（Saurabh Raje）、文卡特桑·查卡拉瓦尔蒂（Venkatesan Chakar-avarthy）、约吉什·萨巴尔瓦尔（Yogish Sabharwal）和阿希什·维尔马（Ashish Verma）。2020年。PoWER - BERT：加速

BERT Inference via Progressive Word-vector Elimination. In Proceedings of the 37th International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 119), Hal Daumé III and Aarti Singh (Eds.). PMLR, 3690-3699. https://proceedings.mlr.press/v119/goyal20a.html

通过渐进式词向量消除实现BERT推理。见第37届国际机器学习会议论文集（机器学习研究会议录，第119卷），哈尔·多梅三世（Hal Daumé III）和阿尔蒂·辛格（Aarti Singh）编。机器学习研究会议录（PMLR），3690 - 3699页。https://proceedings.mlr.press/v119/goyal20a.html

[7] Sebastian Hofstätter, Sheng-Chieh Lin, Jheng-Hong Yang, Jimmy Lin, and Allan Hanbury. 2021. Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (Virtual Event, Canada) (SIGIR '21). Association for Computing Machinery, New York, NY, USA, 113-122. https://doi.org/10.1145/3404835.3462891

[7] 塞巴斯蒂安·霍夫施泰特（Sebastian Hofstätter）、林圣杰（Sheng - Chieh Lin）、杨政宏（Jheng - Hong Yang）、林吉米（Jimmy Lin）和艾伦·汉伯里（Allan Hanbury）。2021年。通过平衡主题感知采样高效训练有效的密集检索器。见第44届国际计算机协会信息检索研究与发展会议论文集（加拿大线上会议）（SIGIR '21）。美国计算机协会，美国纽约州纽约市，113 - 122页。https://doi.org/10.1145/3404835.3462891

[8] Sebastian Hofstätter, Aldo Lipani, Sophia Althammer, Markus Zlabinger, and Allan Hanbury. 2021. Mitigating the Position Bias of Transformer Models in Passage Re-Ranking. arXiv preprint arXiv:2101.06980 (2021).

[8] 塞巴斯蒂安·霍夫施泰特（Sebastian Hofstätter）、阿尔多·利帕尼（Aldo Lipani）、索菲亚·阿尔塔默（Sophia Althammer）、马库斯·兹拉宾格（Markus Zlabinger）和艾伦·汉伯里（Allan Hanbury）。2021年。缓解Transformer模型在段落重排序中的位置偏差。预印本arXiv:2101.06980（2021年）。

[9] Omar Khattab and Matei Zaharia. 2020. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (Virtual Event, China) (SIGIR '20). Association for Computing Machinery, New York, NY, USA, 39-48. https://doi.org/10.1145/3397271.3401075

[9] 奥马尔·哈塔卜（Omar Khattab）和马泰·扎哈里亚（Matei Zaharia）。2020年。ColBERT：通过基于BERT的上下文延迟交互实现高效有效的段落搜索。见第43届国际计算机协会信息检索研究与发展会议论文集（线上会议，中国）（SIGIR '20）。美国计算机协会，美国纽约州纽约市，第39 - 48页。https://doi.org/10.1145/3397271.3401075

[10] Gyuwan Kim and Kyunghyun Cho. 2021. Length-Adaptive Transformer: Train Once with Length Drop, Use Anytime with Search. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). Association for Computational Linguistics, Online, 6501-6511. https: //doi.org/10.18653/v1/2021.acl-long.508

[10] 金圭焕（Gyuwan Kim）和赵京贤（Kyunghyun Cho）。2021年。长度自适应Transformer：通过长度丢弃训练一次，随时通过搜索使用。见计算语言学协会第59届年会暨第11届自然语言处理国际联合会议论文集（第1卷：长论文）。计算语言学协会，线上会议，第6501 - 6511页。https://doi.org/10.18653/v1/2021.acl-long.508

[11] Jimmy Lin, Rodrigo Nogueira, and Andrew Yates. 2020. Pretrained Transformers for Text Ranking: BERT and Beyond. arXiv:2010.06467 [cs] (Oct. 2020). http: //arxiv.org/abs/2010.06467 ZSCC: NoCitationData[s0] arXiv: 2010.06467.

[11] 林俊杰（Jimmy Lin）、罗德里戈·诺盖拉（Rodrigo Nogueira）和安德鲁·耶茨（Andrew Yates）。2020年。用于文本排序的预训练Transformer模型：BERT及其他。arXiv:2010.06467 [计算机科学]（2020年10月）。http: //arxiv.org/abs/2010.06467 ZSCC: 无引用数据[s0] arXiv: 2010.06467。

[12] Rodrigo Nogueira and Kyunghyun Cho. 2019. Passage Re-ranking with BERT. arXiv:1901.04085 [cs.IR]

[12] 罗德里戈·诺盖拉（Rodrigo Nogueira）和赵京贤（Kyunghyun Cho）。2019年。使用BERT进行段落重排序。arXiv:1901.04085 [计算机科学.信息检索]

[13] Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084 (2019).

[13] 尼尔斯·赖默斯（Nils Reimers）和伊琳娜·古列维奇（Iryna Gurevych）。2019年。句子BERT：使用孪生BERT网络的句子嵌入。arXiv预印本arXiv:1908.10084（2019年）。

[14] Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. 2019. Dis-tilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108 (2019).

[14] 维克多·桑（Victor Sanh）、利桑德尔·德比特（Lysandre Debut）、朱利安·肖蒙（Julien Chaumond）和托马斯·沃尔夫（Thomas Wolf）。2019年。DistilBERT，BERT的蒸馏版本：更小、更快、更便宜、更轻量。arXiv预印本arXiv:1910.01108（2019年）。

[15] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. 2021. ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction. arXiv:2112.01488 [cs.IR]

[15] 凯沙夫·桑塔纳姆（Keshav Santhanam）、奥马尔·哈塔卜（Omar Khattab）、乔恩·萨德 - 法尔孔（Jon Saad - Falcon）、克里斯托弗·波茨（Christopher Potts）和马泰·扎哈里亚（Matei Zaharia）。2021年。ColBERTv2：通过轻量级后期交互实现高效检索。预印本arXiv:2112.01488 [计算机科学：信息检索（cs.IR）]

[16] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021. BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2). https://openreview.net/forum?id=wCu6T5xFjeJ

[16] 南丹·塔库尔（Nandan Thakur）、尼尔斯·赖默斯（Nils Reimers）、安德里亚斯·吕克莱（Andreas Rücklé）、阿比舍克·斯里瓦斯塔瓦（Abhishek Srivastava）和伊琳娜·古列维奇（Iryna Gurevych）。2021年。BEIR：信息检索模型零样本评估的异构基准。第三十五届神经信息处理系统会议数据集与基准赛道（第二轮）。https://openreview.net/forum?id=wCu6T5xFjeJ

[17] Nicola Tonellotto and Craig Macdonald. 2021. Query Embedding Pruning for Dense Retrieval. CoRR abs/2108.10341 (2021). arXiv:2108.10341 https://arxiv.org/abs/2108.10341

[17] 尼古拉·托内洛托（Nicola Tonellotto）和克雷格·麦克唐纳（Craig Macdonald）。2021年。密集检索的查询嵌入剪枝。计算机研究存储库（CoRR）abs/2108.10341 (2021)。预印本arXiv:2108.10341 https://arxiv.org/abs/2108.10341

[18] Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. 2020. MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. arXiv:2002.10957 [cs.CL]

[18] 王文慧（Wenhui Wang）、魏富如（Furu Wei）、李东（Li Dong）、鲍航波（Hangbo Bao）、杨楠（Nan Yang）和周明（Ming Zhou）。2020年。《MiniLM：预训练Transformer与任务无关的压缩的深度自注意力蒸馏》。arXiv:2002.10957 [计算机科学 - 计算语言学（cs.CL）]

[19] Xiao Wang, Craig Macdonald, Nicola Tonellotto, and Iadh Ounis. 2021. Pseudo-Relevance Feedback for Multiple Representation Dense Retrieval. In ICTIR '21, Faegheh Hasibi, Yi Fang, and Akiko Aizawa (Eds.). ACM, 297-306. https://doi.org/10.1145/3471158.3472250

[19] 王晓（Xiao Wang）、克雷格·麦克唐纳（Craig Macdonald）、尼古拉·托内洛托（Nicola Tonellotto）和伊阿德·乌尼斯（Iadh Ounis）。2021年。《多表示密集检索的伪相关反馈》。收录于ICTIR '21会议论文集，法赫赫·哈西比（Faegheh Hasibi）、方毅（Yi Fang）和相泽秋子（Akiko Aizawa）（编）。美国计算机协会（ACM），第297 - 306页。https://doi.org/10.1145/3471158.3472250

[20] Ikuya Yamada, Akari Asai, and Hannaneh Hajishirzi. 2021. Efficient Passage Retrieval with Hashing for Open-domain Question Answering. arXiv:2106.00882 [cs.CL]

[20] 山田育也（Ikuya Yamada）、浅井朱里（Akari Asai）和汉娜·哈吉希尔齐（Hannaneh Hajishirzi）。2021年。用于开放域问答的高效哈希段落检索。arXiv预印本：2106.00882 [计算机科学 - 计算语言学（cs.CL）]

[21] Justin Zobel and Alistair Moffat. 2006. Inverted files for text search engines. ACM computing surveys (CSUR) 38, 2 (2006), 6-es.

[21] 贾斯汀·佐贝尔（Justin Zobel）和阿利斯泰尔·莫法特（Alistair Moffat）。2006年。文本搜索引擎的倒排文件。《美国计算机协会计算调查》（ACM Computing Surveys，CSUR）38卷，第2期（2006年），第6页起。