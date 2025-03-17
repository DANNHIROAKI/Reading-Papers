# MULTI-VECTOR RETRIEVAL AS SPARSE ALIGNMENT

# 多向量检索作为稀疏对齐

Yujie Qian*, Jinhyuk Lee, Sai Meher Karthik Duddu, Zhuyun Dai, Siddhartha Brahma, Iftekhar Naim,Tao Lei,Vincent Y. Zhao*‡

钱宇杰（Yujie Qian）*，李晋赫（Jinhyuk Lee），赛·梅赫尔·卡尔蒂克·杜杜（Sai Meher Karthik Duddu），戴竹云（Zhuyun Dai），悉达多·布拉马（Siddhartha Brahma），伊夫特哈尔·奈姆（Iftekhar Naim），雷涛（Tao Lei），赵文森（Vincent Y. Zhao）*‡

Google Research

谷歌研究院

## Abstract

## 摘要

Multi-vector retrieval models improve over single-vector dual encoders on many information retrieval tasks. In this paper, we cast the multi-vector retrieval problem as sparse alignment between query and document tokens. We propose ALIGNER, a novel multi-vector retrieval model that learns sparsified pairwise alignments between query and document tokens (e.g. 'dog' vs. 'puppy') and per-token unary saliences reflecting their relative importance for retrieval. We show that controlling the sparsity of pairwise token alignments often brings significant performance gains. While most factoid questions focusing on a specific part of a document require a smaller number of alignments, others requiring a broader understanding of a document favor a larger number of alignments. Unary saliences, on the other hand, decide whether a token ever needs to be aligned with others for retrieval (e.g. 'kind' from 'what kind of currency is used in new zealand'). With sparsified unary saliences, we are able to prune a large number of query and document token vectors and improve the efficiency of multi-vector retrieval. We learn the sparse unary saliences with entropy-regularized linear programming, which outperforms other methods to achieve sparsity. In a zero-shot setting, ALIGNER scores 51.1 points nDCG@10, achieving a new retriever-only state-of-the-art on 13 tasks in the BEIR benchmark. In addition, adapting pairwise alignments with a few examples $\left( { \leq  8}\right)$ further improves the performance up to 15.7 points nDCG@10 for argument retrieval tasks. The unary saliences of ALIGNER helps us to keep only ${20}\%$ of the document token representations with minimal performance loss. We further show that our model often produces interpretable alignments and significantly improves its performance when initialized from larger language models.

多向量检索模型在许多信息检索任务上比单向量双编码器表现更优。在本文中，我们将多向量检索问题视为查询词和文档词元之间的稀疏对齐问题。我们提出了ALIGNER（对齐器），这是一种新颖的多向量检索模型，它学习查询词和文档词元之间的稀疏成对对齐（例如“dog（狗）”与“puppy（小狗）”），以及反映每个词元在检索中相对重要性的一元显著性。我们表明，控制成对词元对齐的稀疏性通常会带来显著的性能提升。虽然大多数聚焦于文档特定部分的事实类问题只需要较少数量的对齐，但其他需要更广泛理解文档的问题则更适合较多数量的对齐。另一方面，一元显著性决定了一个词元在检索时是否需要与其他词元对齐（例如“what kind of currency is used in new zealand（新西兰使用哪种货币）”中的“kind（种类）”）。通过稀疏化的一元显著性，我们能够修剪大量的查询词和文档词元向量，从而提高多向量检索的效率。我们使用熵正则化线性规划来学习稀疏的一元显著性，这种方法在实现稀疏性方面优于其他方法。在零样本设置下，ALIGNER在nDCG@10指标上得分为51.1分，在BEIR基准测试的13项任务中仅作为检索器就达到了新的最优水平。此外，用少量示例$\left( { \leq  8}\right)$调整成对对齐方式，在论点检索任务中，nDCG@10指标的性能最多可提高15.7分。ALIGNER的一元显著性帮助我们仅保留${20}\%$的文档词元表示，同时将性能损失降至最低。我们进一步表明，我们的模型通常能产生可解释的对齐方式，并且在从更大的语言模型初始化时，其性能会显著提升。

## 1 INTRODUCTION

## 1 引言

Neural information retrieval (IR) has become a promising research direction for improving traditional IR systems. The most-commonly adopted approach called the dual encoder operates by representing every query and document as a single dense vector. Given sufficient annotations, dual encoders directly learn task-driven similarity between vectors, and often surpass traditional IR systems on complex tasks such as question answering (Lee et al., 2019; Karpukhin et al., 2020; Ni et al., 2021). However, these models can struggle to generalize over out-of-domain datasets (Thakur et al., 2021) and/or entity-centric questions (Sciavolino et al. 2021) due to the limited representational capacity of single vectors. As a remedy, multi-vector retrieval models (Khattab & Zaharia, 2020; Luan et al. 2021; Gao et al. 2021) instead use multiple vectors, typically the contextualized token vectors, to represent the text. These models largely improve the model expressiveness, and exhibit much stronger performance and robustness compared to their single-vector counterparts.

神经信息检索（Neural information retrieval，IR）已成为改进传统信息检索系统的一个有前景的研究方向。最常用的方法是双编码器（dual encoder），它将每个查询和文档表示为一个单一的密集向量。在有足够标注的情况下，双编码器可以直接学习向量之间的任务驱动相似度，并且在问答等复杂任务上通常优于传统的信息检索系统（Lee等人，2019年；Karpukhin等人，2020年；Ni等人，2021年）。然而，由于单个向量的表示能力有限，这些模型在处理域外数据集（Thakur等人，2021年）和/或以实体为中心的问题（Sciavolino等人，2021年）时可能难以泛化。作为一种解决方案，多向量检索模型（Khattab和Zaharia，2020年；Luan等人，2021年；Gao等人，2021年）使用多个向量（通常是上下文相关的词元向量）来表示文本。与单向量模型相比，这些模型大大提高了模型的表达能力，并表现出更强的性能和鲁棒性。

Existing multi-vector retrieval models such as ColBERT (Khattab & Zaharia, 2020) computes query-document similarity by selecting the highest scoring document token for each query token and aggregating the scores. This sum-of-max method has two major limitations. First, restricting the selection to a single document token can be highly sub-optimal for some retrieval tasks. As we will show in our experiments, the retrieval performance can be improved by more than 16 points

现有的多向量检索模型，如ColBERT（卡塔布和扎哈里亚，2020年），通过为每个查询词元选择得分最高的文档词元并汇总得分来计算查询 - 文档相似度。这种取最大值求和的方法有两个主要局限性。首先，将选择限制在单个文档词元上，对于某些检索任务而言可能远非最优。正如我们将在实验中展示的那样，通过放宽这一限制，在nDCG@10指标上的检索性能可以提高超过16个百分点

---

<!-- Footnote -->

*Equal contribution.

*同等贡献。

${}^{ \dagger  }$ Currently at MIT CSAIL. Work done during Google internship.

${}^{ \dagger  }$ 目前就职于麻省理工学院计算机科学与人工智能实验室（MIT CSAIL）。此工作是在谷歌实习期间完成的。

*Correspondence: vzhao@google.com

*联系方式：vzhao@google.com

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: (a) Our formulation (b) Alignment in earlier work -->

<img src="https://cdn.noedgeai.com/01957c04-82c5-7814-a5e1-15045c5de529_1.jpg?x=322&y=231&w=1150&h=206&r=0"/>

Figure 1: (a) We formulate multi-vector retrieval as token-level sparse alignment; (b) Earlier models can be covered by our formulation as using different alignments.

图1：(a) 我们将多向量检索表述为词元级别的稀疏对齐；(b) 早期模型可以被我们的表述所涵盖，即使用不同的对齐方式。

<!-- Media -->

nDCG@10 by relaxing this constraint. Second, the method also leads to a large search index and expensive computation. Specifically, the retrieval and storage cost scales linearly with the query and document length, making multi-vector retrieval models an inferior choice for efficiency-demanding applications. We directly tackle these challenges to build faster and more accurate models.

其次，该方法还会导致搜索索引庞大且计算成本高昂。具体而言，检索和存储成本与查询和文档长度呈线性关系，这使得多向量检索模型对于有高效需求的应用来说是一个较差的选择。我们直接应对这些挑战，以构建更快、更准确的模型。

The representation learning problem of multi-vector retrieval can be formulated as optimizing token-level alignment. Specifically, we use a sparse alignment matrix to aggregate token-level similarities, where each element indicates the alignment of a pair of tokens. From this point of view, we are able to formulate different retrieval models in a unified manner (Figure 1) and discern the drawbacks of existing models.

多向量检索的表示学习问题可以表述为优化词元级别的对齐。具体而言，我们使用稀疏对齐矩阵来聚合词元级别的相似度，其中每个元素表示一对词元的对齐情况。从这个角度来看，我们能够以统一的方式构建不同的检索模型（图1），并识别现有模型的缺点。

Based on our formulation, we propose ALIGNER, a novel multi-vector retrieval model that consists of pairwise alignment and unary salience. Pairwise alignments form the basis of ALIGNER, where pairs of query and document tokens are sparsely aligned based on their contextual representations. It is discovered that changing the sparsity of alignment can significantly impact the performance on retrieval tasks. For instance, factoid questions often favor a small number of alignments since they often focus on a small part of a document. However, other queries for different tasks (e.g., argument retrieval and fact checking) require a larger number of alignments for a broader understanding of a document. Our findings also support the claim of Dai et al. (2022b) that retrieval tasks with different intents should be modeled differently.

基于我们的表述，我们提出了ALIGNER，这是一种新颖的多向量检索模型，由成对对齐和一元显著性组成。成对对齐是ALIGNER的基础，其中查询词元和文档词元对基于它们的上下文表示进行稀疏对齐。研究发现，改变对齐的稀疏性会显著影响检索任务的性能。例如，事实类问题通常倾向于少量的对齐，因为它们通常关注文档的一小部分。然而，其他不同任务的查询（例如，论点检索和事实核查）需要更多的对齐，以便更全面地理解文档。我们的研究结果也支持戴等人（Dai et al., 2022b）的观点，即具有不同意图的检索任务应该采用不同的建模方式。

ALIGNER also learns unary saliences, which decides whether each token ever needs to be aligned with any other token for retrieval. This corresponds to masking an entire row or column of the alignment matrix, rather than individual token alignments. To sparsify entire rows or columns, we introduce an algorithm that produces sparse token salience and is end-to-end differentiable based on a novel formulation of entropy-regularized linear programming. Sparsified unary saliences allow us to prune a large number of document and query token representations, making multi-vector retrieval a more efficient and affordable solution.

对齐器（ALIGNER）还会学习一元显著性，它决定了每个标记在检索时是否需要与其他任何标记对齐。这对应于屏蔽对齐矩阵的整行或整列，而不是单个标记对齐。为了使整行或整列变得稀疏，我们引入了一种算法，该算法能产生稀疏的标记显著性，并且基于一种新颖的熵正则化线性规划公式实现端到端可微。稀疏化的一元显著性使我们能够修剪大量的文档和查询标记表示，从而使多向量检索成为一种更高效、更经济的解决方案。

We evaluate ALIGNER on the BEIR benchmark (Thakur et al. 2021), which covers a diverse set of retrieval tasks in multiple domains ${}^{1}$ In a zero-shot setting,we show that simply scaling our model achieves the state-of-the-art performance, outperforming prior neural retrievers without contrastive pre-training, model-based hard negative mining, or distillation. By adapting the pairwise alignments with a few examples from the target task — similar to the setup of Dai et al. (2022b) — ALIGNER can be further improved by up to 15.7 points nDCG@10 on argument retrieval tasks. Meanwhile, pruning with our unary saliences can reduce ${50}\%$ of query tokens for better run-time efficiency and 80% of document tokens for better storage footprint, with less than 1 point decrease of nDCG@10.The pairwise alignments and unary saliences are also highly interpretable so that they often serve as concise rationales for retrieval.

我们在BEIR基准测试（Thakur等人，2021年）上对ALIGNER进行了评估，该基准测试涵盖了多个领域的一系列多样化检索任务${}^{1}$。在零样本设置下，我们表明，简单地扩展我们的模型就能达到最先进的性能，优于那些没有进行对比预训练、基于模型的难负例挖掘或蒸馏的先前神经检索器。通过使用目标任务中的少量示例来调整成对对齐方式（类似于Dai等人（2022b）的设置），在论点检索任务上，ALIGNER的nDCG@10指标最多可提高15.7分。同时，使用我们的一元显著性进行剪枝可以减少${50}\%$的查询词元，以提高运行时效率，并减少80%的文档词元，以优化存储占用，而nDCG@10的下降幅度不到1分。成对对齐方式和一元显著性也具有很高的可解释性，因此它们通常可作为检索的简洁理由。

## 2 MULTI-VECTOR RETRIEVAL AS SPARSE ALIGNMENT

## 2 多向量检索作为稀疏对齐

Given a query $Q$ and a collection of $N$ documents $\mathcal{C} = \left\{  {{D}^{\left( 1\right) },\ldots ,{D}^{\left( N\right) }}\right\}$ ,a key problem in retrieval is how to represent these textual inputs in order to facilitate efficient search. To this end, one approach is lexical retrieval using sparse bag-of-words representation of the text; the other approach is dense retrieval, which this work focuses on. Dense retrieval models learn a parameterized function that encodes the query and documents into query representation $\mathbf{q}$ and document representations $\left\{  {{\mathbf{d}}^{\left( 1\right) },\ldots ,{\mathbf{d}}^{\left( N\right) }}\right\}$ respectively. Typically,each representation is a single $d$ -dimensional vector. For retrieval,the similarity function is often defined as $\operatorname{sim}\left( {Q,{D}^{\left( i\right) }}\right)  = {\mathbf{q}}^{\top }{\mathbf{d}}^{\left( i\right) }$ ,and documents having high similarity scores to the query are retrieved.

给定一个查询 $Q$ 和一个包含 $N$ 篇文档的集合 $\mathcal{C} = \left\{  {{D}^{\left( 1\right) },\ldots ,{D}^{\left( N\right) }}\right\}$，检索中的一个关键问题是如何表示这些文本输入，以便进行高效搜索。为此，一种方法是使用文本的稀疏词袋表示进行词法检索；另一种方法是密集检索，本工作重点关注的就是这种方法。密集检索模型学习一个参数化函数，该函数分别将查询和文档编码为查询表示 $\mathbf{q}$ 和文档表示 $\left\{  {{\mathbf{d}}^{\left( 1\right) },\ldots ,{\mathbf{d}}^{\left( N\right) }}\right\}$。通常，每个表示都是一个单一的 $d$ 维向量。对于检索，相似度函数通常定义为 $\operatorname{sim}\left( {Q,{D}^{\left( i\right) }}\right)  = {\mathbf{q}}^{\top }{\mathbf{d}}^{\left( i\right) }$，并检索与查询具有高相似度得分的文档。

---

<!-- Footnote -->

${}^{1}$ We will release our model checkpoints to encourage future research.

${}^{1}$ 我们将发布我们的模型检查点，以鼓励未来的研究。

<!-- Footnote -->

---

### 2.1 MULTI-VECTOR RETRIEVAL

### 2.1 多向量检索

Instead of representing each query and document as a single fixed-length vector, multi-vector retrieval represents them with multiple token vectors, mainly to improve the limited capacity of fixed-length representations. Specifically,a query $Q = \left\{  {{q}_{1},\ldots ,{q}_{n}}\right\}$ and a document $D = \left\{  {{d}_{1},\ldots ,{d}_{m}}\right\}$ are encoded into a set of vectors $\left\{  {{\mathbf{q}}_{1},\ldots ,{\mathbf{q}}_{n}}\right\}$ and $\left\{  {{\mathbf{d}}_{1},\ldots ,{\mathbf{d}}_{m}}\right\}$ . The similarity function between a query and a document is re-defined for multi-vector retrieval. For instance, ColBERT (Khattab & Zaharia, 2020) designs the similarity function as follows:

多向量检索不是将每个查询和文档表示为单个固定长度的向量，而是用多个词元向量来表示它们，主要是为了改善固定长度表示能力有限的问题。具体来说，一个查询 $Q = \left\{  {{q}_{1},\ldots ,{q}_{n}}\right\}$ 和一个文档 $D = \left\{  {{d}_{1},\ldots ,{d}_{m}}\right\}$ 被编码为一组向量 $\left\{  {{\mathbf{q}}_{1},\ldots ,{\mathbf{q}}_{n}}\right\}$ 和 $\left\{  {{\mathbf{d}}_{1},\ldots ,{\mathbf{d}}_{m}}\right\}$ 。针对多向量检索，重新定义了查询和文档之间的相似度函数。例如，ColBERT（卡塔布和扎哈里亚，2020年）将相似度函数设计如下：

$$
\operatorname{sim}\left( {Q,D}\right)  = \mathop{\sum }\limits_{{i = 1}}^{n}\mathop{\max }\limits_{{j = 1\ldots m}}{\mathbf{q}}_{i}^{\top }{\mathbf{d}}_{j}
$$

For retrieval,instead of indexing $N$ document vectors,multi-vector retrieval pre-computes $N \times  \bar{m}$ document token vectors where $\bar{m}$ is the average length of documents. Then,it retrieves $K$ document token vectors for each query token vector with Maximum Inner-Product Search (MIPS), resulting in $n \times  K$ candidate document tokens. The retrieved tokens are used to trace back the original documents (Lee et al. 2021a), often followed by a final refinement stage that scores the similarity $\operatorname{sim}\left( {Q,D}\right)$ with all token representations of each document and the query (Khattab & Zaharia,2020). We adopt the same practice of ColBERT in our experiments.

在检索方面，多向量检索不是对$N$文档向量进行索引，而是预先计算$N \times  \bar{m}$文档词元向量，其中$\bar{m}$是文档的平均长度。然后，它使用最大内积搜索（MIPS）为每个查询词元向量检索$K$文档词元向量，从而得到$n \times  K$候选文档词元。检索到的词元用于回溯原始文档（Lee等人，2021a），之后通常会有一个最终的细化阶段，该阶段会计算每个文档的所有词元表示与查询之间的相似度$\operatorname{sim}\left( {Q,D}\right)$（Khattab和Zaharia，2020）。我们在实验中采用了与ColBERT相同的做法。

### 2.2 SPARSE ALIGNMENT FORMULATION

### 2.2 稀疏对齐公式化

A key design question for retrieval models is defining the similarity function in a manner that balances model expressiveness and inference cost. To facilitate our discussion, we formalize the similarities used in previous methods into a class of sparse alignment functions. The formulation also leads to a principled extension over existing work, which we will describe in § 3

检索模型的一个关键设计问题是以一种平衡模型表达能力和推理成本的方式定义相似度函数。为了便于讨论，我们将先前方法中使用的相似度形式化为一类稀疏对齐函数。这种公式化还对现有工作进行了有原则的扩展，我们将在§ 3中进行描述。

We begin by defining a similarity matrix $\mathbf{S} \in  {\mathbb{R}}^{n \times  m}$ computed from all pairs of query and document tokens,where ${\mathbf{S}}_{i,j} = {\mathbf{q}}_{i}^{\top }{\mathbf{d}}_{j}$ . Then,we use an alignment matrix $\mathbf{A} \in  {\left\lbrack  0,1\right\rbrack  }^{n \times  m}$ to compute the similarity between $Q$ and $D$ as follows:

我们首先定义一个从所有查询词和文档词对计算得出的相似度矩阵$\mathbf{S} \in  {\mathbb{R}}^{n \times  m}$，其中${\mathbf{S}}_{i,j} = {\mathbf{q}}_{i}^{\top }{\mathbf{d}}_{j}$。然后，我们使用一个对齐矩阵$\mathbf{A} \in  {\left\lbrack  0,1\right\rbrack  }^{n \times  m}$来计算$Q$和$D$之间的相似度，如下所示：

$$
\operatorname{sim}\left( {Q,D}\right)  = \frac{1}{Z}\mathop{\sum }\limits_{{i = 1}}^{n}\mathop{\sum }\limits_{{j = 1}}^{m}{\mathbf{S}}_{i,j}{\mathbf{A}}_{i,j} \tag{1}
$$

where $Z$ is a normalization term defined as $Z = \mathop{\sum }\limits_{{i,j}}{\mathbf{A}}_{i,j}$ . The alignment matrix $\mathbf{A}$ can be directly derived from $S$ or computed as a function of $Q$ and $D$ .

其中$Z$是一个归一化项，定义为$Z = \mathop{\sum }\limits_{{i,j}}{\mathbf{A}}_{i,j}$。对齐矩阵$\mathbf{A}$可以直接从$S$推导得出，或者作为$Q$和$D$的函数进行计算。

On the top of our formulation,the alignment matrix $\mathbf{A}$ is constrained to be sparsely activated: $\parallel \mathbf{A}{\parallel }_{0} \leq  \sigma$ where $\parallel  \cdot  {\parallel }_{0}$ is the number of non-zero elements in a matrix. Sparse activation assumes that only a few query-document token matches are critical for retrieval, inspired by traditional retrieval methods. Indeed, most existing dense retrieval models already enforce the sparse alignment with their own heuristics. Figure 1 illustrates how different models can be described under our formulation:

在我们的公式体系中，对齐矩阵 $\mathbf{A}$ 被约束为稀疏激活：$\parallel \mathbf{A}{\parallel }_{0} \leq  \sigma$ 其中 $\parallel  \cdot  {\parallel }_{0}$ 是矩阵中非零元素的数量。稀疏激活假设只有少数查询 - 文档词元匹配对检索至关重要，这一假设受到传统检索方法的启发。实际上，大多数现有的密集检索模型已经通过自身的启发式方法来强制实现稀疏对齐。图 1 展示了在我们的公式体系下如何描述不同的模型：

- Dense passage retriever (DPR; Karpukhin et al., 2020) uses a single [CLS] vector to represent each query and document. This is equivalent to setting ${A}_{1,1} = 1$ and 0 otherwise, resulting in $\parallel \mathbf{A}{\parallel }_{0} = 1$ .

- 密集段落检索器（DPR；卡尔普欣等人，2020 年）使用单个 [CLS] 向量来表示每个查询和文档。这相当于设置 ${A}_{1,1} = 1$，否则为 0，从而得到 $\parallel \mathbf{A}{\parallel }_{0} = 1$。

- ME-BERT (Luan et al. 2021) uses the first $k$ document token vectors for multi-vector representations of documents but a single vector for query. The similarity function is $\mathop{\max }\limits_{{j = 1\ldots k}}{\mathbf{q}}_{1}^{\top }{\mathbf{d}}_{j}$ ,which is equivalent to setting ${A}_{1,j} = 1$ when ${\mathbf{S}}_{1,j}$ is the maximum within ${\mathbf{S}}_{1,1}$ to ${\mathbf{S}}_{1,k}$ ,and 0 otherwise. The alignment sparsity is $\parallel \mathbf{A}{\parallel }_{0} = 1$ .

- ME - BERT（Luan等人，2021年）使用前$k$个文档标记向量来进行文档的多向量表示，但查询使用单个向量。相似度函数为$\mathop{\max }\limits_{{j = 1\ldots k}}{\mathbf{q}}_{1}^{\top }{\mathbf{d}}_{j}$，当${\mathbf{S}}_{1,j}$是${\mathbf{S}}_{1,1}$到${\mathbf{S}}_{1,k}$范围内的最大值时，该函数等同于设置${A}_{1,j} = 1$，否则为0。对齐稀疏度为$\parallel \mathbf{A}{\parallel }_{0} = 1$。

- ColBERT uses the sum-of-max similarity function $\mathop{\sum }\limits_{{i = 1}}^{n}\mathop{\max }\limits_{{j = 1\ldots m}}{\mathbf{S}}_{i,j}$ that is equivalent to setting an alignment matrix to select the maximum element from each row of $\mathbf{S}$ ,i.e., ${\mathbf{A}}_{i,j} = 1$ when ${\mathbf{S}}_{i,j}$ is the maximum within ${\mathbf{S}}_{i, : } \cdot  \parallel \mathbf{A}{\parallel }_{0} = n$ in this case.

- ColBERT（科尔伯特）使用最大和相似度函数$\mathop{\sum }\limits_{{i = 1}}^{n}\mathop{\max }\limits_{{j = 1\ldots m}}{\mathbf{S}}_{i,j}$，该函数等同于设置一个对齐矩阵，从$\mathbf{S}$的每一行中选择最大元素，即当${\mathbf{S}}_{i,j}$是${\mathbf{S}}_{i, : } \cdot  \parallel \mathbf{A}{\parallel }_{0} = n$中的最大值时，在这种情况下为${\mathbf{A}}_{i,j} = 1$。

- COIL (Gao et al., 2021), similar to ColBERT, also selects the maximum element from each row of $\mathbf{S}$ ,but requires a lexical exact match for a selected pair,i.e., ${\mathbf{A}}_{i,j} = 1$ when ${\mathbf{S}}_{i,j}$ is the maximum within $\left\{  {{\mathbf{S}}_{i,{j}^{\prime }} \mid  {q}_{i} = {d}_{{j}^{\prime }}}\right\}  .\parallel \mathbf{A}{\parallel }_{0} \leq  n$ in this case.

- COIL（高等人，2021年）与ColBERT类似，同样从$\mathbf{S}$的每一行中选择最大元素，但要求所选的一对元素在词汇上完全匹配，即当${\mathbf{S}}_{i,j}$是$\left\{  {{\mathbf{S}}_{i,{j}^{\prime }} \mid  {q}_{i} = {d}_{{j}^{\prime }}}\right\}  .\parallel \mathbf{A}{\parallel }_{0} \leq  n$中的最大值时，此时为${\mathbf{A}}_{i,j} = 1$。

<!-- Media -->

<!-- figureText: similarity alignment pairwise alignment unary salience ${u}^{q} \otimes  {u}^{d}$ ${\mathbf{S}}_{i,j} = {\mathbf{q}}_{i}^{\top }{\mathbf{d}}_{j}$ salience ${u}^{q}$ $\left( {000}\right) \left( {000}\right) \ldots \left( {000}\right)$ query $Q$ document $D$ -->

<img src="https://cdn.noedgeai.com/01957c04-82c5-7814-a5e1-15045c5de529_3.jpg?x=323&y=235&w=1151&h=245&r=0"/>

Figure 2: ALIGNER factorizes the alignment matrix into pairwise alignments and unary saliences. Pairwise alignment focuses on the alignment of individual token pairs. Unary saliences are determined by per-token salience features.

图2：ALIGNER将对齐矩阵分解为成对对齐和一元显著性。成对对齐侧重于单个标记对的对齐。一元显著性由每个标记的显著性特征决定。

<!-- Media -->

The choice of similarity and sparsity can have a large impact on model capacity and efficiency. For instance, ColBERT is more expressive and robust than DPR (Thakur et al. 2021), but its retrieval and storage costs are much higher. Our work seeks to further advance expressiveness while retaining a strong efficiency. We describe our method in the next section.

相似度和稀疏性的选择会对模型的能力和效率产生重大影响。例如，ColBERT（科尔伯特）比DPR（密集段落检索器，Thakur等人，2021年）更具表现力和鲁棒性，但它的检索和存储成本要高得多。我们的工作旨在在保持高效率的同时进一步提高表现力。我们将在下一节描述我们的方法。

## 3 ALIGNER

## 3 ALIGNER（对齐器）

In this section, we present ALIGNER built upon the sparse alignment formulation. ALIGNER fac-torizes the alignment matrix into pairwise alignment and unary salience:

在本节中，我们介绍基于稀疏对齐公式构建的ALIGNER（对齐器）。ALIGNER（对齐器）将对齐矩阵分解为成对对齐和一元显著性：

$$
\mathbf{A} = \widetilde{\mathbf{A}} \odot  \left( {{\mathbf{u}}^{q} \otimes  {\mathbf{u}}^{d}}\right)  \tag{2}
$$

where $\odot$ is the Hadamard product and $\otimes$ is the outer product of two vectors. Pairwise alignment $\widetilde{\mathbf{A}} \in  {\mathbb{R}}^{n \times  m}$ determines which pairs of query and document tokens should be aligned,with the sparsity constraints tailored for downstream tasks (§3.1). Unary salience ${\mathbf{u}}^{q} \in  {\mathbb{R}}^{n}$ and ${\mathbf{u}}^{d} \in  {\mathbb{R}}^{m}$ are sparse token weights deciding whether a token ever needs to be aligned (§3.2).

其中 $\odot$ 是哈达玛积（Hadamard product），$\otimes$ 是两个向量的外积。成对对齐 $\widetilde{\mathbf{A}} \in  {\mathbb{R}}^{n \times  m}$ 确定查询和文档标记的哪些对应该对齐，其稀疏性约束是为下游任务量身定制的（§3.1）。一元显著性 ${\mathbf{u}}^{q} \in  {\mathbb{R}}^{n}$ 和 ${\mathbf{u}}^{d} \in  {\mathbb{R}}^{m}$ 是稀疏标记权重，用于决定一个标记是否需要进行对齐（§3.2）。

The factorization is introduced based on two critical hypotheses. First, the optimal sparsity of alignment can be task-dependent. Instead of imposing top-1 constraint as in ColBERT, activating more than one alignments for a query token can enhance retrieval performance for certain tasks. In our analyses for instance, we observe factoid questions that only concern a specific part of a document require a small number of alignments, while some other queries (such as fact checking) require more alignments for a broader understanding of the document. We explore different search spaces of the pairwise alignment matrix $\widetilde{\mathbf{A}}$ in order to achieve better retrieval performance for each downstream task. Second, alignment is only needed for very few tokens. For example, we analyzed 2000 most retrieved documents in our preliminary study, and found only 12.8% document tokens are retrieved by at least one query ${}^{2}$ . Intuitively,tokens that are uninformative do not need to be aligned and stored, corresponding to sparse activation over an entire row or column of $\mathbf{A}$ . ALIGNER directly learns the row and column sparsity as unary salience, and utilizes them to enhance retrieval efficiency.

因式分解是基于两个关键假设引入的。首先，对齐的最优稀疏性可能取决于任务。与ColBERT（一种模型）中施加top - 1约束不同，为一个查询词元激活多个对齐方式可以提高某些任务的检索性能。例如，在我们的分析中，我们观察到只涉及文档特定部分的事实类问题只需要少量对齐，而其他一些查询（如事实核查）则需要更多对齐以更全面地理解文档。为了为每个下游任务实现更好的检索性能，我们探索了成对对齐矩阵$\widetilde{\mathbf{A}}$的不同搜索空间。其次，只需要对极少数词元进行对齐。例如，在我们的初步研究中，我们分析了2000篇被检索最多的文档，发现只有12.8%的文档词元被至少一个查询${}^{2}$检索到。直观地说，无信息价值的词元不需要进行对齐和存储，这对应于$\mathbf{A}$的整行或整列的稀疏激活。ALIGNER（一种工具或模型）直接将行和列的稀疏性作为一元显著性进行学习，并利用它们来提高检索效率。

### 3.1 ADAPTING PAIRWISE ALIGNMENT

### 3.1 调整成对对齐

Queries and documents can have varied distributions. For example, a query can be a single entity, a natural question, or a few sentences, and a document can range from a short paragraph to a long article. The search intent also changes from task to task (Dai et al. 2022b). These changes can lead to different optimal alignment strategies. We explore the following sparse alignment variants that go beyond the top-1 strategy commonly adopted in existing work:

查询和文档可能具有不同的分布。例如，查询可以是单个实体、一个自然问题或几句话，而文档可以从短段落到长篇文章不等。搜索意图也会因任务而异（戴等人，2022b）。这些变化可能导致不同的最优对齐策略。我们探索了以下稀疏对齐变体，它们超越了现有工作中常用的前1策略：

- Top- $k$ . Each query token is aligned with $k$ document tokens with highest similarity scores. Precisely, ${\widetilde{\mathbf{A}}}_{i,j} = 1$ when the $j$ -th token is within top- $k$ of the row ${\mathbf{S}}_{i}$ . When $k = 1$ ,it is equivalent to ColBERT.

- 前 $k$ 。每个查询标记与相似度得分最高的 $k$ 个文档标记对齐。具体来说，当第 $j$ 个标记位于行 ${\mathbf{S}}_{i}$ 的前 $k$ 范围内时，${\widetilde{\mathbf{A}}}_{i,j} = 1$ 。当 $k = 1$ 时，它等同于ColBERT。

- Top- $p$ . This strategy is similar to top- $k$ ,but instead of aligning each query token with exactly $k$ tokens,it makes the number of alignments proportional to the document length, i.e.,each query token aligns with $\max \left( {\lfloor p \cdot  m\rfloor ,1}\right)$ tokens where $m$ is the document length and $p \in  \left\lbrack  {0,1}\right\rbrack$ is the alignment ratio. Despite their simplicity, these variants can indeed enhance retrieval accuracy significantly on tasks such as argument retrieval. More importantly, while it is possible to train separate models for different alignment variants, we are interested in fast test-time adaptation using a single shared model as many important retrieval tasks lack sufficient training data (Thakur et al. 2021). Specifically, we first train ALIGNER using a fixed alignment strategy such as top-1 in a source domain, and adapt the alignment strategy to each target task without changing the model parameters ${}^{3}$ We use the following few-shot alignment adaptation method. Given a corpus $\left\{  {{D}^{\left( 1\right) },\ldots ,{D}^{\left( N\right) }}\right\}$ ,and a few relevance-annotated query-document pairs from the target task $\left\{  {\left( {{Q}^{1},{D}_{ + }^{1}}\right) ,\ldots \left( {{Q}^{K},{D}_{ + }^{K}}\right) }\right\}$ ,we first retrieve candidate documents with the learned token representations, and decide the pairwise alignment strategy based on the ranking performance on the annotated data. This adaptation can be performed efficiently because the alignment only concerns the computation of similarity score (Eq. 1) in the refinement stage. In practice, for some tasks, we are able to find a well-suited alignment strategy and improve the retrieval performance with as few as 8 annotated examples.

- 前$p$。这种策略与前$k$策略类似，但不是将每个查询词元（query token）精确地与$k$个词元对齐，而是使对齐数量与文档长度成比例，即每个查询词元与$\max \left( {\lfloor p \cdot  m\rfloor ,1}\right)$个词元对齐，其中$m$是文档长度，$p \in  \left\lbrack  {0,1}\right\rbrack$是对齐比例。尽管这些变体很简单，但它们确实可以在诸如论点检索等任务上显著提高检索准确率。更重要的是，虽然可以为不同的对齐变体训练单独的模型，但我们对使用单个共享模型进行快速测试时自适应感兴趣，因为许多重要的检索任务缺乏足够的训练数据（塔库尔等人，2021年）。具体来说，我们首先在源领域中使用固定的对齐策略（如前1策略）训练对齐器（ALIGNER），并在不改变模型参数的情况下将对齐策略适应于每个目标任务${}^{3}$。我们使用以下少样本对齐自适应方法。给定一个语料库$\left\{  {{D}^{\left( 1\right) },\ldots ,{D}^{\left( N\right) }}\right\}$，以及来自目标任务的少量带有相关性标注的查询 - 文档对$\left\{  {\left( {{Q}^{1},{D}_{ + }^{1}}\right) ,\ldots \left( {{Q}^{K},{D}_{ + }^{K}}\right) }\right\}$，我们首先使用学习到的词元表示检索候选文档，并根据标注数据上的排序性能确定成对对齐策略。这种自适应可以高效执行，因为对齐只涉及细化阶段中相似度得分的计算（公式1）。实际上，对于某些任务，我们能够找到非常合适的对齐策略，并仅使用8个标注示例就提高检索性能。

---

<!-- Footnote -->

${}^{2}$ The analysis was performed on MS MARCO (Nguyen et al. 2016) using our implementation of ColBERT.

${}^{2}$ 我们使用自己实现的ColBERT模型，在MS MARCO数据集（阮等人，2016年）上进行了分析。

<!-- Footnote -->

---

### 3.2 LEARNING UNARY SALIENCE

### 3.2 学习一元显著性

ALIGNER predicts token saliences from their token representations. For brevity, we only present the formulation for document salience, and query salience is defined similarly. Specifically, the salience of the $i$ -th document token ${u}_{i}^{d}$ is defined as:

ALIGNER模型根据词元表示来预测词元的显著性。为简洁起见，我们仅给出文档显著性的公式，查询显著性的定义与之类似。具体而言，第 $i$ 个文档词元 ${u}_{i}^{d}$ 的显著性定义如下：

$$
{u}_{i}^{d} = {\lambda }_{i}^{d} \cdot  f\left( {{\mathbf{W}}^{d}{\mathbf{d}}_{i} + {b}^{d}}\right)  \tag{3}
$$

where ${\mathbf{W}}^{d}$ and ${b}^{d}$ are learnable parameters. $f$ is a non-linear activation function and we use ReLU such that salience is always non-negative. ${\mathbf{\lambda }}^{d} = \left\{  {\lambda }_{i}^{d}\right\}$ are gating variables to control the overall sparsity of ${\mathbf{u}}^{d}$ ,which we will elaborate next.

其中 ${\mathbf{W}}^{d}$ 和 ${b}^{d}$ 是可学习的参数。 $f$ 是非线性激活函数，我们使用ReLU函数，以确保显著性始终为非负。 ${\mathbf{\lambda }}^{d} = \left\{  {\lambda }_{i}^{d}\right\}$ 是门控变量，用于控制 ${\mathbf{u}}^{d}$ 的整体稀疏性，我们将在下文中详细阐述。

For the document salience to be meaningful, we enforce salience sparsity as an inductive bias. ALIGNER jointly optimizes sparse salience with other parts of the model. Since tokens with zero salience do not contribute to computing similarity, our model will be encouraged to identify more important tokens in order to retain good retrieval performance. Note that during training we do not have any explicit annotation on which tokens are important. Instead, ${\mathbf{u}}^{d}$ (and similarly ${\mathbf{u}}^{q}$ ) are directly optimized to minimize the training loss,under the sparsity constraint that ${\begin{Vmatrix}{\mathbf{\lambda }}^{d}\end{Vmatrix}}_{0} = \left\lceil  {{\mathbf{\alpha }}^{d} \cdot  m}\right\rceil$ , where ${\alpha }^{d}$ is a constant sparsity ratio and $m$ is the document length.

为使文档显著性具有意义，我们将显著性稀疏性作为归纳偏置。ALIGNER（对齐器）联合优化稀疏显著性与模型的其他部分。由于显著性为零的标记对计算相似度没有贡献，我们的模型将被激励识别更重要的标记，以保持良好的检索性能。请注意，在训练过程中，我们没有关于哪些标记重要的任何明确注释。相反，${\mathbf{u}}^{d}$（以及类似的${\mathbf{u}}^{q}$）在${\begin{Vmatrix}{\mathbf{\lambda }}^{d}\end{Vmatrix}}_{0} = \left\lceil  {{\mathbf{\alpha }}^{d} \cdot  m}\right\rceil$的稀疏性约束下直接进行优化，以最小化训练损失，其中${\alpha }^{d}$是一个恒定的稀疏率，$m$是文档长度。

Of course, a key question is how we can optimize the unary salience component given the controlled sparsity. We leverage a novel technique called entropy-regularized linear programming to enable end-to-end optimization. Specifically,let $k = \left\lceil  {{\mathbf{\alpha }}^{d} \cdot  m}\right\rceil$ denotes the desired sparsity, ${s}_{i} = f\left( {{\mathbf{W}}^{d}{\mathbf{d}}_{i} + }\right.$ ${b}^{d}$ ) denotes the token score before the sparse gate ${\lambda }_{i}^{d}$ is applied,and $s,{\mathbf{\lambda }}^{d} \in  {\mathbb{R}}^{m}$ be the vectors $\left\{  {s}_{i}\right\}$ and $\left\{  {\lambda }_{i}^{d}\right\}$ respectively. ${\mathbf{\lambda }}^{d}$ is computed by solving the following optimization problem:

当然，一个关键问题是，在给定受控稀疏性的情况下，我们如何优化一元显著性分量。我们利用一种名为熵正则化线性规划（entropy-regularized linear programming）的新技术来实现端到端优化。具体而言，设$k = \left\lceil  {{\mathbf{\alpha }}^{d} \cdot  m}\right\rceil$表示所需的稀疏性，${s}_{i} = f\left( {{\mathbf{W}}^{d}{\mathbf{d}}_{i} + }\right.$（${b}^{d}$）表示应用稀疏门${\lambda }_{i}^{d}$之前的词元得分，$s,{\mathbf{\lambda }}^{d} \in  {\mathbb{R}}^{m}$分别为向量$\left\{  {s}_{i}\right\}$和$\left\{  {\lambda }_{i}^{d}\right\}$。${\mathbf{\lambda }}^{d}$通过求解以下优化问题来计算：

$$
\mathop{\max }\limits_{\mathbf{\lambda }}{\mathbf{s}}^{\top }\mathbf{\lambda } + {\varepsilon H}\left( \mathbf{\lambda }\right) \;\text{ s.t. }\;{\mathbf{1}}^{\top }\mathbf{\lambda } = k,\;{\lambda }_{i} \in  \left\lbrack  {0,1}\right\rbrack  ,\forall i = 1,\ldots ,m. \tag{4}
$$

where $H\left( \cdot \right)$ is the elementwise entropy function 4 and $\varepsilon  > 0$ is a small constant. The optimization can be seen as a relaxed top- $k$ operation. Without the entropy term ${\varepsilon H}\left( \cdot \right)$ ,it becomes an instance of linear programming where the solution ${\mathbf{\lambda }}^{d}$ is a binary mask indicating the top- $k$ values of $s$ ,i.e., ${\lambda }_{i}^{d} = 1$ if and only if ${s}_{i}$ is one of top- $k$ values in $s$ . This top- $k$ optimization is smoothed by adding the small entropy term ${\varepsilon H}\left( \cdot \right)$ and by relaxing ${\lambda }_{i}$ from exact binary to $\left\lbrack  {0,1}\right\rbrack$ . Given small $\varepsilon$ ,this still produce a sparse solution ${\mathbf{\lambda }}^{d}$ and can be solved using simple vector operations. Specifically,let $a \in  \mathbb{R}$ and ${b}_{i} \in  \mathbb{R}$ for $i = 1,\cdots ,m$ be auxiliary variables that are initialized to zero. We iteratively update these variables using the following equations:

其中 $H\left( \cdot \right)$ 是逐元素熵函数4，$\varepsilon  > 0$ 是一个小常数。该优化可以看作是一种松弛的前 $k$ 操作。如果没有熵项 ${\varepsilon H}\left( \cdot \right)$，它就变成了线性规划的一个实例，其中解 ${\mathbf{\lambda }}^{d}$ 是一个二进制掩码，指示 $s$ 的前 $k$ 个值，即当且仅当 ${s}_{i}$ 是 $s$ 中的前 $k$ 个值之一时，${\lambda }_{i}^{d} = 1$ 成立。通过添加小的熵项 ${\varepsilon H}\left( \cdot \right)$ 并将 ${\lambda }_{i}$ 从精确的二进制松弛为 $\left\lbrack  {0,1}\right\rbrack$，对这个前 $k$ 优化进行了平滑处理。给定小的 $\varepsilon$，这仍然会产生一个稀疏解 ${\mathbf{\lambda }}^{d}$，并且可以使用简单的向量运算来求解。具体来说，令 $a \in  \mathbb{R}$ 和对于 $i = 1,\cdots ,m$ 的 ${b}_{i} \in  \mathbb{R}$ 为初始化为零的辅助变量。我们使用以下方程迭代更新这些变量：

$$
{a}^{\prime } = \varepsilon \ln \left( k\right)  - \varepsilon \ln \left\{  {\mathop{\sum }\limits_{i}\exp \left( \frac{{s}_{i} + {b}_{i}}{\varepsilon }\right) }\right\}  ,\;{b}_{i}^{\prime } = \min \left( {-{s}_{i} - {a}^{\prime },0}\right) . \tag{5}
$$

In practice,it is sufficient to run only a few iterations and the final solution is given by ${\lambda }_{i} =$ $\exp \left( \frac{{s}_{i} + {b}_{i} + a}{\varepsilon }\right)$ . These vector operations are differentiable so $\mathbf{\lambda }$ can be end-to-end trained with other parts of our model. The full derivation of this iterative algorithm is given in Appendix A.1.

在实践中，仅运行几次迭代就足够了，最终解由 ${\lambda }_{i} =$ $\exp \left( \frac{{s}_{i} + {b}_{i} + a}{\varepsilon }\right)$ 给出。这些向量运算具有可微性，因此 $\mathbf{\lambda }$ 可以与我们模型的其他部分进行端到端训练。该迭代算法的完整推导过程见附录 A.1。

---

<!-- Footnote -->

${}^{3}$ We have also explored a differentiable alignment with sparsity constraints (Appendix A.2),but alignment adaptation is still necessary to achieve good performance on target tasks.

${}^{3}$ 我们还探索了一种带有稀疏性约束的可微对齐方法（附录 A.2），但为了在目标任务上取得良好性能，仍然需要进行对齐调整。

${}^{4}H\left( \mathbf{\lambda }\right)  = \mathop{\sum }\limits_{{i = 1}}^{m} - {\lambda }_{i}\log {\lambda }_{i}$

<!-- Footnote -->

---

<!-- Media -->

<table><tr><td/><td>Supervision</td><td>Hard Negatives</td><td>Distillation</td><td>Retriever</td><td>Per-domain</td><td>#Param.</td></tr><tr><td>Splade, ${}_{\mathrm{v}2}$</td><td>MS MARCO</td><td>model-based</td><td>✓</td><td>lexical</td><td/><td>110M</td></tr><tr><td>ColBERTv2</td><td>MS MARCO</td><td>model-based</td><td>✓</td><td>multi-vector</td><td/><td>110M</td></tr><tr><td>${\mathrm{{GTR}}}_{\mathrm{{xxl}}}$</td><td>Pre-train + MS MARCO</td><td>fixed</td><td/><td>single-vector</td><td/><td>6B</td></tr><tr><td>Promptagator</td><td>Few $\left( { \leq  8}\right)$</td><td/><td/><td>single-vector</td><td>✓</td><td>110M</td></tr><tr><td>${\mathrm{{ALIGNER}}}_{\mathrm{{xxl}}}$</td><td>MS MARCO + ${\mathrm{{Few}}}^{ * }\left( { \leq  8}\right)$</td><td>fixed</td><td/><td>multi-vector</td><td/><td>6B</td></tr></table>

<table><tbody><tr><td></td><td>监督</td><td>难负样本（Hard Negatives）</td><td>蒸馏</td><td>检索器</td><td>按领域</td><td>#参数数量（#Param.）</td></tr><tr><td>斯普拉德（Splade），${}_{\mathrm{v}2}$</td><td>微软机器阅读理解数据集（MS MARCO）</td><td>基于模型的</td><td>✓</td><td>词汇的</td><td></td><td>110M</td></tr><tr><td>科尔伯特v2（ColBERTv2）</td><td>微软机器阅读理解数据集（MS MARCO）</td><td>基于模型的</td><td>✓</td><td>多向量</td><td></td><td>110M</td></tr><tr><td>${\mathrm{{GTR}}}_{\mathrm{{xxl}}}$</td><td>预训练 + MS MARCO（微软机器阅读理解数据集）</td><td>固定的</td><td></td><td>单向量</td><td></td><td>6B</td></tr><tr><td>提示生成器（Promptagator）</td><td>少量 $\left( { \leq  8}\right)$</td><td></td><td></td><td>单向量</td><td>✓</td><td>110M</td></tr><tr><td>${\mathrm{{ALIGNER}}}_{\mathrm{{xxl}}}$</td><td>微软机器阅读理解数据集（MS MARCO） + ${\mathrm{{Few}}}^{ * }\left( { \leq  8}\right)$</td><td>固定的</td><td></td><td>多向量</td><td></td><td>6B</td></tr></tbody></table>

Table 1: Comparison of different retrieval models. *: optionally used for alignment adaptation.

表1：不同检索模型的比较。*：可选择用于对齐调整。

<!-- Media -->

Pruning Multi-vector Retrieval With the learned unary salience, we can naturally prune tokens for multi-vector retrieval. Pruning document tokens reduces the number of vectors in search index, and pruning query tokens reduces the number of searches. In our experiments, we control them using two pruning ratios ${\beta }^{q}$ and ${\beta }^{d}$ respectively. For each document,we obtain the token salience using Eq. (3) and only store the top ${\beta }^{d}$ percent of tokens in the index. Similarly we select the top ${\beta }^{q}$ percent query tokens to perform max inner-product search. Note that we vary these two ratios to control retrieval efficiency,and these ratios can be smaller than the sparsity ratio ${\alpha }^{q}$ and ${\alpha }^{d}$ which we use as constraints at training time. In the refinement stage, we still use the full model with all token vectors for scoring.

剪枝多向量检索：利用学习到的一元显著性，我们可以自然地对多向量检索中的词元进行剪枝。对文档词元进行剪枝可以减少搜索索引中的向量数量，对查询词元进行剪枝可以减少搜索次数。在我们的实验中，我们分别使用两个剪枝率${\beta }^{q}$和${\beta }^{d}$来控制它们。对于每个文档，我们使用公式(3)获取词元显著性，并仅在索引中存储前${\beta }^{d}$%的词元。同样，我们选择前${\beta }^{q}$%的查询词元来执行最大内积搜索。请注意，我们通过改变这两个比率来控制检索效率，并且这些比率可以小于我们在训练时用作约束的稀疏率${\alpha }^{q}$和${\alpha }^{d}$。在细化阶段，我们仍然使用包含所有词元向量的完整模型进行评分。

## 4 EXPERIMENTS

## 4 实验

### 4.1 EXPERIMENTAL SETUP

### 4.1 实验设置

ALIGNER uses shared transformer encoder initialized from T5 version 1.1 (Raffel et al. 2020). We project token embeddings to 128 dimension and apply L2 normalization. Following GTR (Ni et al., 2021), we finetune ALIGNER on MS MARCO with hard negatives released by RocketQA (Qu et al. 2021). The models are trained with a batch size of 256 for ${25}\mathrm{k}$ steps,using query sequence length of 64 and document sequence length of 256. We train ALIGNER with top-1 pairwise alignment ${}^{5}$

ALIGNER使用从T5 1.1版本（拉菲尔等人，2020年）初始化的共享Transformer编码器。我们将词元嵌入投影到128维并应用L2归一化。遵循GTR（倪等人，2021年）的方法，我们在MS MARCO数据集上对ALIGNER进行微调，使用的是RocketQA发布的难负样本（曲等人，2021年）。这些模型以256的批量大小训练${25}\mathrm{k}$步，查询序列长度为64，文档序列长度为256。我们使用top - 1成对对齐方式训练ALIGNER ${}^{5}$

For retrieval, we pre-compute the token encodings of all the documents in the corpus, and use ScaNN (Guo et al. 2020) to index and perform max inner-product search (MIPS). We retrieve 4,000 nearest neighbors for each query token ${}^{6}$ and return the top-1,000 after the refinement stage. We evaluate ALIGNER on the BEIR benchmark (Thakur et al. 2021) and compare with state-of-the-art retrieval models shown in Table 1. Note that ALIGNER does not rely on contrastive model pretraining (Izacard et al. 2022; Ni et al. 2021), model-based hard negative mining (Santhanam et al. 2021), or distillation (Santhanam et al. 2021). We intentionally decide this simple recipe and focus on studying the impact of pairwise alignment and unary salience.

在检索方面，我们预先计算语料库中所有文档的词元编码，并使用ScaNN（郭等人，2020年）进行索引和执行最大内积搜索（MIPS）。对于每个查询词元${}^{6}$，我们检索出4000个最近邻，并在精炼阶段后返回前1000个。我们在BEIR基准测试（塔库尔等人，2021年）上评估ALIGNER，并与表1中所示的最先进的检索模型进行比较。请注意，ALIGNER不依赖于对比模型预训练（伊扎卡尔等人，2022年；倪等人，2021年）、基于模型的难负样本挖掘（桑塔南姆等人，2021年）或蒸馏（桑塔南姆等人，2021年）。我们有意采用这种简单的方法，并专注于研究成对对齐和一元显著性的影响。

For few-shot alignment adaptation of ALIGNER (§3.1), we split the test data into multiple folds such that each fold contains 8 examples. Then we find the best alignment strategy that maximizes nDCG@10 on each fold with $k \in  \{ 1,2,4,6,8\}$ for top- $k$ and $p \in  \{ {0.5}\% ,1\% ,{1.5}\% ,2\% \}$ for top- $p$ . Based on the best alignment strategy from each fold, we measure the retrieval performance on the remaining test examples with the best strategy. We report the average (± std.) of these test scores where the number of test scores equals the number of folds. The average of few-shot adaptation indicates the expected performance of using few examples to choose the best alignment strategy.

对于ALIGNER（§3.1）的小样本对齐适配，我们将测试数据划分为多个子集，使每个子集包含8个示例。然后，我们找到在每个子集上使nDCG@10最大化的最佳对齐策略，其中前$k$个使用$k \in  \{ 1,2,4,6,8\}$，前$p$个使用$p \in  \{ {0.5}\% ,1\% ,{1.5}\% ,2\% \}$。基于每个子集的最佳对齐策略，我们用该最佳策略对剩余的测试示例进行检索性能评估。我们报告这些测试得分的平均值（±标准差），其中测试得分的数量等于子集的数量。小样本适配的平均值表明了使用少量示例来选择最佳对齐策略的预期性能。

### 4.2 RETRIEVAL ACCURACY

### 4.2 检索准确率

Table 2 shows the document retrieval performance of ALIGNER on both MS MARCO and the BEIR benchmark. For this experiment, we do not prune any query or document tokens with unary saliences,but show their effects in §4.3 instead. ALIGNER ${}_{\mathrm{{xxl}}}$ outperforms all baselines on MS-MARCO, showing how multi-vector retrieval models can benefit from large pretrained language models. ALIGNER ${}_{\mathrm{{xxl}}}$ also outperforms ${\mathrm{{GTR}}}_{\mathrm{{xxl}}}$ on 9 out of 13 BEIR datasets and advances the retriever-only state-of-the-art $\left( {\mathrm{{ColBERT}}}_{\mathrm{v}2}\right)$ by 1.2 points nDCG@10 on average. Figure 3 shows that our multi-vector retriever model scales better than single-vector dual encoder GTR.

表2展示了ALIGNER在MS MARCO和BEIR基准测试上的文档检索性能。在本实验中，我们没有对任何具有一元显著性的查询或文档标记进行修剪，而是在§4.3中展示其影响。ALIGNER ${}_{\mathrm{{xxl}}}$在MS - MARCO上的表现优于所有基线模型，这表明多向量检索模型如何从大型预训练语言模型中受益。ALIGNER ${}_{\mathrm{{xxl}}}$在13个BEIR数据集中的9个上也优于${\mathrm{{GTR}}}_{\mathrm{{xxl}}}$，并且在平均nDCG@10指标上比仅使用检索器的当前最优模型$\left( {\mathrm{{ColBERT}}}_{\mathrm{v}2}\right)$提高了1.2分。图3显示，我们的多向量检索器模型的扩展性优于单向量双编码器GTR。

---

<!-- Footnote -->

${}^{5}$ We have trained models with other top- $k$ and top- $p$ pairwise alignments,but the MS MARCO training data favors top-1 alignment (see Appendix A.4 for details).

${}^{5}$ 我们已经使用其他前$k$和前$p$成对对齐方式训练了模型，但MS MARCO训练数据更倾向于前1对齐（详情见附录A.4）。

${}^{6}$ Unlike ColBERT,ALIGNER does not use pad token embeddings for retrieval. Hence,retrieving 4,000 neighbors per query token results in a similar number of retrieved candidates to ColBERT.

${}^{6}$ 与ColBERT（科尔伯特）不同，ALIGNER（对齐器）在检索时不使用填充标记嵌入。因此，每个查询标记检索4000个邻居所得到的候选检索结果数量与ColBERT（科尔伯特）相近。

<!-- Footnote -->

---

<!-- Media -->

<table><tr><td/><td>BM25</td><td>5 Splade ${}_{\mathrm{v}2}^{ * }$</td><td>ColBERT ${}_{\mathrm{v}2}^{ * }$</td><td>${\mathrm{{GTR}}}_{\mathrm{{xxl}}}$</td><td/><td>${\mathrm{{PTR}}}^{ \dagger  }$</td><td>${\mathrm{{ALIGNER}}}_{\mathrm{{xxl}}}^{ \dagger  }$</td></tr><tr><td>MS MARCO</td><td>18.7</td><td>36.8</td><td>39.7</td><td>38.8</td><td>40.3</td><td>-</td><td>-</td></tr><tr><td>ArguAna</td><td>31.5</td><td>47.9</td><td>46.3</td><td>54.0</td><td>33.8</td><td>59.4</td><td>${47.9} \pm  {3.0}$</td></tr><tr><td>Touché-2020</td><td>36.7</td><td>27.2</td><td>26.3</td><td>25.6</td><td>34.5</td><td>34.5</td><td>${50.2}^{\pm {1.1}}$</td></tr><tr><td>FEVER</td><td>75.3</td><td>78.6</td><td>78.5</td><td>74.0</td><td>74.2</td><td>77.0</td><td>${73.9}^{\pm {4.8}}$</td></tr><tr><td>Climate-FEVER</td><td>21.3</td><td>23.5</td><td>17.6</td><td>26.7</td><td>19.7</td><td>24.0</td><td>${22.8}^{\pm {2.9}}$</td></tr><tr><td>SciFact</td><td>66.5</td><td>69.3</td><td>69.3</td><td>66.2</td><td>73.1</td><td>65.0</td><td>${71.4}^{\pm {2.2}}$</td></tr><tr><td>TREC-COVID</td><td>65.6</td><td>71.0</td><td>73.8</td><td>50.1</td><td>75.8</td><td>75.6</td><td>${79.3}^{\pm {3.0}}$</td></tr><tr><td>NFCorpus</td><td>32.5</td><td>33.4</td><td>33.8</td><td>34.2</td><td>35.2</td><td>33.4</td><td>${33.4}^{\pm {2.0}}$</td></tr><tr><td>NO</td><td>32.9</td><td>52.1</td><td>56.2</td><td>56.8</td><td>60.5</td><td>-</td><td>${56.6}^{\pm {5.1}}$</td></tr><tr><td>HotpotQA</td><td>60.3</td><td>68.4</td><td>66.7</td><td>59.9</td><td>65.2</td><td>61.4</td><td>${63.2}^{\pm {3.3}}$</td></tr><tr><td>FiQA</td><td>23.6</td><td>33.6</td><td>35.6</td><td>46.7</td><td>43.5</td><td>46.2</td><td>${39.9}^{\pm {4.5}}$</td></tr><tr><td>SCIDOCS</td><td>15.8</td><td>15.8</td><td>15.4</td><td>16.1</td><td>17.1</td><td>18.4</td><td>${16.3}^{\pm {1.2}}$</td></tr><tr><td>DBPedia</td><td>31.3</td><td>43.5</td><td>44.6</td><td>40.8</td><td>45.0</td><td>38.0</td><td>${43.5}^{\pm {2.4}}$</td></tr><tr><td>Quora</td><td>78.9</td><td>83.5</td><td>85.2</td><td>89.2</td><td>86.0</td><td>-</td><td>${85.3}^{\pm {2.1}}$</td></tr><tr><td>Average</td><td>44.0</td><td>49.8</td><td>49.9</td><td>49.3</td><td>51.1</td><td>-</td><td>${52.6}^{\pm {3.1}}$</td></tr><tr><td>– NQ / Quora</td><td>41.9</td><td>46.6</td><td>46.2</td><td>44.9</td><td>47.0</td><td>47.8</td><td>${49.3}^{\pm {3.0}}$</td></tr></table>

<table><tbody><tr><td></td><td>BM25（原文未变，为专业术语）</td><td>5 Splade ${}_{\mathrm{v}2}^{ * }$（原文未变，可能为特定专业概念）</td><td>ColBERT ${}_{\mathrm{v}2}^{ * }$（原文未变，可能为特定专业概念）</td><td>${\mathrm{{GTR}}}_{\mathrm{{xxl}}}$</td><td></td><td>${\mathrm{{PTR}}}^{ \dagger  }$</td><td>${\mathrm{{ALIGNER}}}_{\mathrm{{xxl}}}^{ \dagger  }$</td></tr><tr><td>微软机器阅读理解数据集（MS MARCO）</td><td>18.7</td><td>36.8</td><td>39.7</td><td>38.8</td><td>40.3</td><td>-</td><td>-</td></tr><tr><td>ArguAna（原文未变，可能为特定名称）</td><td>31.5</td><td>47.9</td><td>46.3</td><td>54.0</td><td>33.8</td><td>59.4</td><td>${47.9} \pm  {3.0}$</td></tr><tr><td>Touché-2020（原文未变，可能为特定活动或项目名称）</td><td>36.7</td><td>27.2</td><td>26.3</td><td>25.6</td><td>34.5</td><td>34.5</td><td>${50.2}^{\pm {1.1}}$</td></tr><tr><td>发热</td><td>75.3</td><td>78.6</td><td>78.5</td><td>74.0</td><td>74.2</td><td>77.0</td><td>${73.9}^{\pm {4.8}}$</td></tr><tr><td>气候发热数据集（Climate-FEVER）</td><td>21.3</td><td>23.5</td><td>17.6</td><td>26.7</td><td>19.7</td><td>24.0</td><td>${22.8}^{\pm {2.9}}$</td></tr><tr><td>科学事实数据集（SciFact）</td><td>66.5</td><td>69.3</td><td>69.3</td><td>66.2</td><td>73.1</td><td>65.0</td><td>${71.4}^{\pm {2.2}}$</td></tr><tr><td>信息检索会议新冠专题数据集（TREC-COVID）</td><td>65.6</td><td>71.0</td><td>73.8</td><td>50.1</td><td>75.8</td><td>75.6</td><td>${79.3}^{\pm {3.0}}$</td></tr><tr><td>自然语言处理医学语料库（NFCorpus）</td><td>32.5</td><td>33.4</td><td>33.8</td><td>34.2</td><td>35.2</td><td>33.4</td><td>${33.4}^{\pm {2.0}}$</td></tr><tr><td>否</td><td>32.9</td><td>52.1</td><td>56.2</td><td>56.8</td><td>60.5</td><td>-</td><td>${56.6}^{\pm {5.1}}$</td></tr><tr><td>火锅问答数据集（HotpotQA）</td><td>60.3</td><td>68.4</td><td>66.7</td><td>59.9</td><td>65.2</td><td>61.4</td><td>${63.2}^{\pm {3.3}}$</td></tr><tr><td>金融问答数据集（FiQA）</td><td>23.6</td><td>33.6</td><td>35.6</td><td>46.7</td><td>43.5</td><td>46.2</td><td>${39.9}^{\pm {4.5}}$</td></tr><tr><td>科学文献数据集（SCIDOCS）</td><td>15.8</td><td>15.8</td><td>15.4</td><td>16.1</td><td>17.1</td><td>18.4</td><td>${16.3}^{\pm {1.2}}$</td></tr><tr><td>DBpedia数据集（DBPedia）</td><td>31.3</td><td>43.5</td><td>44.6</td><td>40.8</td><td>45.0</td><td>38.0</td><td>${43.5}^{\pm {2.4}}$</td></tr><tr><td>Quora问答平台（Quora）</td><td>78.9</td><td>83.5</td><td>85.2</td><td>89.2</td><td>86.0</td><td>-</td><td>${85.3}^{\pm {2.1}}$</td></tr><tr><td>平均值</td><td>44.0</td><td>49.8</td><td>49.9</td><td>49.3</td><td>51.1</td><td>-</td><td>${52.6}^{\pm {3.1}}$</td></tr><tr><td>– NQ / 奎若网（Quora）</td><td>41.9</td><td>46.6</td><td>46.2</td><td>44.9</td><td>47.0</td><td>47.8</td><td>${49.3}^{\pm {3.0}}$</td></tr></tbody></table>

Table 2: Results on MS MARCO (top; MRR@10) and the BEIR benchmark (bottom; nDCG@10). Best scores before adaptation are denoted in boldface. ${}^{ * }$ : trained with distillation. PTR: Promptagator (Dai et al.,2022b). ${}^{ \dagger  }$ : uses few examples $\left( { \leq  8}\right)$ for task-specific adaptation.

表2：MS MARCO（上；MRR@10）和BEIR基准测试（下；nDCG@10）的结果。适配前的最佳分数用粗体表示。${}^{ * }$：通过蒸馏训练。PTR：Promptagator（戴等人，2022b）。${}^{ \dagger  }$：使用少量示例$\left( { \leq  8}\right)$进行特定任务适配。

<!-- figureText: 40 ${10}^{9}$ #Param. ${10}^{8}$ -->

<img src="https://cdn.noedgeai.com/01957c04-82c5-7814-a5e1-15045c5de529_6.jpg?x=1104&y=333&w=375&h=337&r=0"/>

Figure 3: Averaged nDCG@10 on BEIR of retrieval models with different sizes. AA: alignment adaptation.

图3：不同规模的检索模型在BEIR上的平均nDCG@10。AA：对齐适配。

<table><tr><td/><td colspan="6">Top- $k$</td><td colspan="5">Top- $p$</td></tr><tr><td/><td>1</td><td>2</td><td>4</td><td>6</td><td>8</td><td>16</td><td>0.5%</td><td>1%</td><td>1.5%</td><td>2%</td><td>5%</td></tr><tr><td>ArguAna</td><td>28.8</td><td>24.4</td><td>18.3</td><td>14.5</td><td>11.4</td><td>5.1</td><td>33.3</td><td>45.5</td><td>48.1</td><td>46.9</td><td>32.6</td></tr><tr><td>Touché-2020</td><td>34.8</td><td>50.0</td><td>51.1</td><td>49.3</td><td>46.0</td><td>33.7</td><td>31.3</td><td>24.2</td><td>20.3</td><td>15.8</td><td>5.2</td></tr><tr><td>SciFact</td><td>70.4</td><td>68.8</td><td>65.0</td><td>60.9</td><td>55.6</td><td>38.6</td><td>71.1</td><td>69.4</td><td>67.2</td><td>62.7</td><td>39.0</td></tr><tr><td>TREC-COVID</td><td>68.3</td><td>74.0</td><td>73.2</td><td>67.2</td><td>61.7</td><td>41.8</td><td>66.8</td><td>64.4</td><td>56.7</td><td>46.7</td><td>30.9</td></tr><tr><td>FiOA</td><td>33.4</td><td>30.8</td><td>23.8</td><td>19.5</td><td>15.5</td><td>8.43</td><td>33.4</td><td>28.5</td><td>24.6</td><td>19.0</td><td>7.7</td></tr><tr><td>SCIDOCS</td><td>14.1</td><td>14.3</td><td>13.1</td><td>11.4</td><td>9.7</td><td>4.82</td><td>14.4</td><td>14.9</td><td>14.9</td><td>14.8</td><td>8.1</td></tr><tr><td>DBPedia</td><td>41.6</td><td>39.4</td><td>29.6</td><td>20.2</td><td>14.2</td><td>3.94</td><td>41.6</td><td>41.7</td><td>39.9</td><td>36.6</td><td>17.0</td></tr><tr><td>Average</td><td>41.6</td><td>43.1</td><td>39.2</td><td>36.5</td><td>30.6</td><td>19.5</td><td>41.7</td><td>41.2</td><td>38.8</td><td>34.6</td><td>20.1</td></tr></table>

<table><tbody><tr><td></td><td colspan="6">前 $k$</td><td colspan="5">前 $p$</td></tr><tr><td></td><td>1</td><td>2</td><td>4</td><td>6</td><td>8</td><td>16</td><td>0.5%</td><td>1%</td><td>1.5%</td><td>2%</td><td>5%</td></tr><tr><td>论点分析（ArguAna）</td><td>28.8</td><td>24.4</td><td>18.3</td><td>14.5</td><td>11.4</td><td>5.1</td><td>33.3</td><td>45.5</td><td>48.1</td><td>46.9</td><td>32.6</td></tr><tr><td>妙语交锋2020（Touché-2020）</td><td>34.8</td><td>50.0</td><td>51.1</td><td>49.3</td><td>46.0</td><td>33.7</td><td>31.3</td><td>24.2</td><td>20.3</td><td>15.8</td><td>5.2</td></tr><tr><td>科学事实（SciFact）</td><td>70.4</td><td>68.8</td><td>65.0</td><td>60.9</td><td>55.6</td><td>38.6</td><td>71.1</td><td>69.4</td><td>67.2</td><td>62.7</td><td>39.0</td></tr><tr><td>文本检索会议新冠专题（TREC-COVID）</td><td>68.3</td><td>74.0</td><td>73.2</td><td>67.2</td><td>61.7</td><td>41.8</td><td>66.8</td><td>64.4</td><td>56.7</td><td>46.7</td><td>30.9</td></tr><tr><td>FiOA（原文未变，可能是特定缩写暂不明确通用译法）</td><td>33.4</td><td>30.8</td><td>23.8</td><td>19.5</td><td>15.5</td><td>8.43</td><td>33.4</td><td>28.5</td><td>24.6</td><td>19.0</td><td>7.7</td></tr><tr><td>SCIDOCS（原文未变，可能是特定缩写暂不明确通用译法）</td><td>14.1</td><td>14.3</td><td>13.1</td><td>11.4</td><td>9.7</td><td>4.82</td><td>14.4</td><td>14.9</td><td>14.9</td><td>14.8</td><td>8.1</td></tr><tr><td>DBPedia（DBPedia，一种结构化的多语言百科知识图谱）</td><td>41.6</td><td>39.4</td><td>29.6</td><td>20.2</td><td>14.2</td><td>3.94</td><td>41.6</td><td>41.7</td><td>39.9</td><td>36.6</td><td>17.0</td></tr><tr><td>平均值</td><td>41.6</td><td>43.1</td><td>39.2</td><td>36.5</td><td>30.6</td><td>19.5</td><td>41.7</td><td>41.2</td><td>38.8</td><td>34.6</td><td>20.1</td></tr></tbody></table>

Table 3: nDCG@10 on the BEIR benchmark with different $k$ and $p$ in ${\mathrm{{ALIGNER}}}_{\text{base }}$ . ALIGNER ${\mathrm{{ER}}}_{\text{base }}$ is trained with top- $k = 1$ .

表3：在BEIR基准测试中，${\mathrm{{ALIGNER}}}_{\text{base }}$ 里不同的 $k$ 和 $p$ 对应的nDCG@10值。ALIGNER ${\mathrm{{ER}}}_{\text{base }}$ 采用前 $k = 1$ 进行训练。

<!-- figureText: 50 ALIGNER DA 30 -->

<img src="https://cdn.noedgeai.com/01957c04-82c5-7814-a5e1-15045c5de529_6.jpg?x=1126&y=936&w=325&h=269&r=0"/>

Figure 4: Averaged nDCG@10 on BEIR of ALIGNER trained with different alignments. DA: Appendix A. 2

图4：采用不同对齐方式训练的ALIGNER在BEIR上的平均nDCG@10值。DA：附录A. 2

<!-- Media -->

Alignment Adaptation In the rightmost column of Table 2, we show the effect of adapting pairwise alignment with ALIGNER on the BEIR benchmark. With only 8 examples for finding the proper alignment sparsity,its expected performance reaches ${52.6}\mathrm{{nDCG}}@{10}$ on average. Alignment-adapted ALIGNER also benefits from scaling up, and consistently outperforms its non-adapted counterparts, as shown in Figure 3. The gains are further explained in Table 3 where we show individual task's performance under various alignment strategies. Although ALIGNER is trained with top-1 alignment, top-1 is not always the best strategy at inference time. Specifically, for ArguAna, we observe 16 points improvement by adjusting the number of alignments proportional to the document length with $p = {1.5}\%$ . Other tasks such as Touché-2020 also prefer other alignment strategies, which shows that different tasks might require different sparsity. In general, keeping the sparsity low enough is preferable and supports our hypothesis that pairwise alignments should be sparse.

对齐适配 在表2的最右列，我们展示了使用ALIGNER对成对对齐进行适配在BEIR基准测试上的效果。仅使用8个示例来寻找合适的对齐稀疏度时，其预期性能平均达到${52.6}\mathrm{{nDCG}}@{10}$。经对齐适配的ALIGNER也从规模扩大中受益，并且始终优于未适配的同类方法，如图3所示。表3进一步解释了这些提升，我们在表3中展示了不同对齐策略下各个任务的性能。尽管ALIGNER是使用top - 1对齐进行训练的，但在推理时top - 1并不总是最佳策略。具体而言，对于ArguAna（论点分析）任务，通过使用$p = {1.5}\%$根据文档长度按比例调整对齐数量，我们观察到性能提升了16个百分点。其他任务，如Touché - 2020（触摸2020）也倾向于其他对齐策略，这表明不同的任务可能需要不同的稀疏度。一般来说，保持足够低的稀疏度是更可取的，这支持了我们的假设，即成对对齐应该是稀疏的。

We further check whether this observation holds when ALIGNER is trained with other pairwise alignment strategies. Figure 4 shows ALIGNER variants trained on four alternative strategies. We evaluate their performance with training-time alignment strategy (default) and the optimal alignment strategy selected per dataset (oracle). While these models perform differently with their default alignments, they perform similarly after oracle alignment adaptation.

我们进一步检查当使用其他成对对齐策略训练ALIGNER（对齐器）时，这一观察结果是否仍然成立。图4展示了采用四种替代策略训练的ALIGNER变体。我们使用训练时对齐策略（默认）和为每个数据集选择的最优对齐策略（神谕）来评估它们的性能。虽然这些模型在默认对齐方式下表现不同，但在经过神谕对齐调整后，它们的表现相似。

Figure 5 shows the effectiveness of few-shot alignment adaptation — dynamically selecting task-specific alignment strategy based on a few examples. When the default alignment $\left( {\text{top-}k = 1}\right)$ is not optimal, we can identify a good alignment strategy using only 8 examples, which significantly improves model performance on argument retrieval tasks. Using 16 examples further improves the average score and reduces the variance. However, when the default alignment is already optimal

图5展示了少样本对齐调整的有效性——基于少量示例动态选择特定任务的对齐策略。当默认对齐方式$\left( {\text{top-}k = 1}\right)$并非最优时，我们仅使用8个示例就能确定一个良好的对齐策略，这显著提高了模型在论点检索任务中的性能。使用16个示例可进一步提高平均得分并降低方差。然而，当默认对齐方式已经是最优时

<!-- Media -->

<!-- figureText: FEVER -->

<img src="https://cdn.noedgeai.com/01957c04-82c5-7814-a5e1-15045c5de529_7.jpg?x=336&y=236&w=1118&h=297&r=0"/>

Figure 5: ALIGNE ${\mathrm{R}}_{\mathrm{{xxl}}}$ with few-shot alignment adaptation. We report nDCG@10 on BEIR.

图5：采用少样本对齐调整的ALIGNER（对齐器）。我们报告了BEIR数据集上的nDCG@10指标。

<!-- figureText: — ALIGNER (full) Document token ratio ${\beta }^{d}$ $\vdash$ ALIGNER, ${\beta }^{q} = {50}\%  \rightarrow$ - ReLU, ${\beta }^{q} = {50}\%$ ALIGNER, ${\beta }^{q} = {10}\%$ -->

<img src="https://cdn.noedgeai.com/01957c04-82c5-7814-a5e1-15045c5de529_7.jpg?x=315&y=619&w=381&h=374&r=0"/>

Figure 6: ALIGNER with unary salience on MS MARCO. ${\beta }^{q}$ and ${\beta }^{d}$ are ratios to prune query and document tokens, respectively.

图6：在MS MARCO数据集上采用一元显著性的ALIGNER（对齐器）。${\beta }^{q}$和${\beta }^{d}$分别是修剪查询和文档标记的比例。

<!-- figureText: Touché-2020 TREC-COVID Climate-FEVER DBPedia Document token ratio ${\beta }^{d}$ HotpotQA -->

<img src="https://cdn.noedgeai.com/01957c04-82c5-7814-a5e1-15045c5de529_7.jpg?x=728&y=611&w=756&h=389&r=0"/>

Figure 7: ALIGNER with unary salience on BEIR. We set query pruning ratio ${\beta }^{q} = {50}\%$ and vary document pruning ratio ${\beta }^{d}$ . We omit datasets with small corpora. We also report the performance of ALIGNER with alignment adaptation (AA).

图7：在BEIR（基准检索数据集）上具有一元显著性的ALIGNER（对齐器）。我们设置查询剪枝率 ${\beta }^{q} = {50}\%$ 并改变文档剪枝率 ${\beta }^{d}$。我们省略了语料库较小的数据集。我们还报告了具有对齐自适应（AA）功能的ALIGNER的性能。

<!-- Media -->

(top- $k = 1$ is optimal for QA tasks),few-shot alignment adaptation hurts performance due to the variance of our few-shot method. Nevertheless, ALIGNER outperforms Promptagator (Dai et al., 2022b), another few-shot retrieval baseline, in 6 out of 11 datasets.

（对于问答任务，前 $k = 1$ 是最优的），由于我们的少样本方法存在差异，少样本对齐自适应会降低性能。尽管如此，在11个数据集中，有6个数据集上ALIGNER的表现优于另一个少样本检索基线Promptagator（戴等人，2022b）。

### 4.3 RETRIEVAL EFFICIENCY

### 4.3 检索效率

The next experiment shows how ALIGNER's unary salience impacts retrieval efficiency. We train ${\text{ALIGNER}}_{\text{base }}$ with salience sparsity ratios ${\alpha }^{q} = {50}\%$ and ${\alpha }^{d} = {40}\%$ based on empirical performance. The gating variables are optimized with $\varepsilon  = {0.002}$ . At retrieval time,we prune query and document tokens with ratios ${\beta }^{q}$ and ${\beta }^{d}$ (§3.2).

下一个实验展示了ALIGNER（对齐器）的一元显著性如何影响检索效率。我们根据实证性能，以显著性稀疏率${\alpha }^{q} = {50}\%$和${\alpha }^{d} = {40}\%$训练${\text{ALIGNER}}_{\text{base }}$。使用$\varepsilon  = {0.002}$对门控变量进行优化。在检索时，我们以比率${\beta }^{q}$和${\beta }^{d}$对查询和文档标记进行剪枝（§3.2）。

Figure 6 shows the ALIGNER performance on MS MARCO with various pruning ratios. When pruned at the same ratio as training $\left( {{\beta }^{q} = {50}\% }\right.$ and $\left. {{\beta }_{d} = {40}\% }\right)$ ,the model is close to a full ALIGNER model without pruning (MRR@1038.1 vs. 38.8), but greatly saves the computation cost. We can further prune tokens by adjusting ${\beta }^{d}$ and ${\beta }_{q}$ . The model achieves ${37.3}\mathrm{{MRR}}@{10}$ with is ${\beta }^{d} = {10}\%$ , i.e. it remains accurate with only ${10}\%$ of the original index size. Decreasing the query pruning ratio ${\beta }^{q}$ to 30% does not sacrifice performance too much,although deceasing ${\beta }^{q}$ to 10% leads to worse performance. Figure 6 also compares ALIGNER's entropy-regularized linear program (Eq. 4) with alternative methods. With just a ReLU gate and no sparsity constraints ('ReLU' in Figure 6), the model performance retains good when ${\beta }^{d} = {40}\%$ ,but drops significantly for smaller ${\beta }^{d}$ . Removing the entropy regularization in Eq. 4 leads to simply selecting the hard top- $k$ tokens with the highest predicted salience (’Hard’ in Figure 6). The hard top- $k$ solution has worse performance for all ${\beta }^{d}$ .

图6展示了ALIGNER在MS MARCO数据集上不同剪枝率下的性能表现。当以与训练时相同的比例（$\left( {{\beta }^{q} = {50}\% }\right.$和$\left. {{\beta }_{d} = {40}\% }\right)$）进行剪枝时，该模型的性能接近未进行剪枝的完整ALIGNER模型（MRR@10为38.1，而未剪枝模型为38.8），但大幅节省了计算成本。我们可以通过调整${\beta }^{d}$和${\beta }_{q}$进一步对词元进行剪枝。该模型在${\beta }^{d} = {10}\%$的情况下达到了${37.3}\mathrm{{MRR}}@{10}$，即仅使用原始索引大小的${10}\%$就能保持较高的准确性。将查询剪枝率${\beta }^{q}$降低到30%不会过多牺牲性能，不过将${\beta }^{q}$降至10%会导致性能变差。图6还将ALIGNER的熵正则化线性规划（公式4）与其他方法进行了比较。仅使用ReLU门且无稀疏性约束（图6中的‘ReLU’）时，当${\beta }^{d} = {40}\%$时模型性能良好，但对于较小的${\beta }^{d}$，性能会显著下降。去掉公式4中的熵正则化会导致简单地选择预测显著性最高的硬前$k$个词元（图6中的‘Hard’）。对于所有的${\beta }^{d}$，硬前$k$个词元的解决方案性能都较差。

ALIGNER's salience estimation also generalizes to other retrieval datasets. As shown in Figure 7, pruning with ${\beta }_{d} = {10}\%$ with ${\beta }^{q} = {50}\%$ causes minimal performance decrease for a majority of BEIR datasets. We even observe performance increase for Touché-2020, as the model can only retrieve salient tokens after pruning. Besides, we show that alignment adaptation can be combined with pruning, resulting in an effective yet efficient retrieval model.

ALIGNER（对齐器）的显著性估计也适用于其他检索数据集。如图7所示，使用${\beta }_{d} = {10}\%$和${\beta }^{q} = {50}\%$进行剪枝对大多数BEIR（基准检索）数据集的性能影响极小。我们甚至观察到在Touché - 2020数据集上性能有所提升，因为模型在剪枝后只能检索到显著的标记。此外，我们表明对齐自适应可以与剪枝相结合，从而得到一个有效且高效的检索模型。

<!-- Media -->

<table><tr><td>Query</td><td>what happens when stop drinking alcohol</td></tr><tr><td>Doc.</td><td>alcohol . symptoms of alcohol withdrawal may begin from 4 to 12 hours after you cut down or stop drinking , or as long as several days after the last drink , and can last a few days . they can range from mild to life - threatening. 1 mild withdrawal symptoms may include : 2 intense worry. 3 nausea or vomiting - 4 s hak iness . 5 sweat ing .</td></tr><tr><td>Query</td><td>where is the heart in the human body</td></tr><tr><td>Doc.</td><td>heart the heart is a muscular organ in most animals, which pumps blood through the blood vessels of the circul atory system. [ 1 ] blood provides the body with oxygen and nutrients, as well as assists in the removal of metabolic waste s . [ 2 ] in humans, the heart is located between the lungs, in the middle compartment of the chest. [ 3 ]</td></tr></table>

<table><tbody><tr><td>查询</td><td>停止饮酒会发生什么</td></tr><tr><td>医生</td><td>酒精。戒酒症状可能在你减少饮酒量或停止饮酒后的4到12小时开始出现，也可能在最后一次饮酒几天后才出现，并且可能持续数天。这些症状的严重程度从轻微到危及生命不等。1. 轻微的戒酒症状可能包括：2. 极度焦虑。3. 恶心或呕吐。4. 颤抖。5. 出汗。</td></tr><tr><td>查询</td><td>人体的心脏在哪里</td></tr><tr><td>医生</td><td>心脏 心脏是大多数动物体内的一个肌肉器官，它通过循环系统的血管输送血液。[1] 血液为身体提供氧气和营养物质，并帮助清除代谢废物。[2] 在人类体内，心脏位于两肺之间，胸腔的中间腔室。[3]</td></tr></tbody></table>

Table 4: Examples of the pairwise alignment and unary salience learned by ALIGNER. Three most salient query tokens and their top-1 pairwise-aligned document tokens are indicated with the same color. We highlight top 50% query tokens and 20% document tokens according to their salience.

表4：ALIGNER学习到的成对对齐（pairwise alignment）和一元显著性（unary salience）示例。三个最显著的查询词元（query tokens）及其排名第一的成对对齐文档词元（document tokens）用相同颜色表示。我们根据显著性突出显示了前50%的查询词元和前20%的文档词元。

<!-- Media -->

### 4.4 INTERPRETABILITY

### 4.4 可解释性

Table 4 shows examples of the pairwise alignment and unary salience learned by ALIGNER. The model aligns query tokens to contexually similar tokens, but not necessarily identical tokens. The salience features are also highlighted in Table 4. Important noun phrases and verbs are usually assigned higher salience, which is consistent with human intuition. We show more examples of alignments for different tasks in the Appendix A.3. In general, we observe question answering tasks usually require fewer alignments for each query token, while other tasks that require a broad understanding of the document favor larger number of alignments.

表4展示了ALIGNER学习到的成对对齐和一元显著性示例。该模型将查询词元与上下文相似的词元对齐，但不一定是相同的词元。表4中也突出显示了显著性特征。重要的名词短语和动词通常被赋予更高的显著性，这与人类的直觉一致。我们在附录A.3中展示了不同任务的更多对齐示例。一般来说，我们观察到问答任务通常每个查询词元所需的对齐较少，而其他需要广泛理解文档的任务则倾向于更多的对齐。

## 5 RELATED WORK

## 5 相关工作

Recent research on information retrieval often improves the retrieval accuracy with contrastive pretraining (Ni et al., 2021; Izacard et al., 2022; Oguz et al., 2022), model-based hard negative mining (Xiong et al., 2020; Lu et al., 2021; Qu et al., 2021) and knowledge distillation (Santhanam et al., 2021, Zhang et al., 2022; Reddi et al., 2021). Retrieval efficiency is improved via quantization (San-thanam et al., 2021) or lower-dimensional vectors (Hofstätter et al., 2022). These improvements are orthogonal to this work.

近期关于信息检索的研究通常通过对比预训练（倪等人，2021年；伊扎卡德等人，2022年；奥古兹等人，2022年）、基于模型的难负样本挖掘（熊等人，2020年；陆等人，2021年；曲等人，2021年）和知识蒸馏（桑塔南姆等人，2021年；张等人，2022年；雷迪等人，2021年）来提高检索准确率。通过量化（桑塔南姆等人，2021年）或低维向量（霍夫施泰特等人，2022年）来提高检索效率。这些改进与本研究无关。

Term importance and salience have a long history in information retrieval: from term frequency (tf) and inverse document frequency (idf), to recent BERT-based importance measures such as DeepCT (Dai & Callan, 2020), SPARTA (Zhao et al., 2021) and Splade (Formal et al., 2021b a). These works mostly focus on sparse lexical retrieval and learn term weights for sparse bag-of-words representations. Term importance in multi-vector dense retrieval is less explored. Our work is probably most related to a recent work from Hofstätter et al. (2022), which prunes ColBERT by predicting salience scores from a word's embedding with a ReLU gate and L1-norm regularization.

术语重要性和显著性在信息检索领域有着悠久的历史：从词频（tf）和逆文档频率（idf），到最近基于BERT的重要性度量方法，如DeepCT（戴和卡兰，2020年）、SPARTA（赵等人，2021年）和Splade（福尔马尔等人，2021b a）。这些研究大多聚焦于稀疏词汇检索，并为稀疏词袋表示学习术语权重。多向量密集检索中的术语重要性则较少被探索。我们的研究可能与霍夫施泰特等人（2022年）最近的一项研究最为相关，该研究通过使用ReLU门和L1范数正则化从单词嵌入中预测显著性得分来对ColBERT进行剪枝。

Recently, Promptagator (Dai et al., 2022b) points out the importance of using a few annotated examples to adapt to a new retrieval task. Promptagator achieves few-shot task adaptation via query generation (Ma et al., 2021; Lee et al., 2021b; Dai et al., 2022a) using large language models (Sanh et al., 2022; Brown et al., 2020; Wei et al., 2022), which has high inference cost. ALIGNER is more versatile and can be fast adapted to a new task via few-shot alignment adaptation.

最近，Promptagator（戴等人，2022b）指出了使用少量带注释的示例来适应新的检索任务的重要性。Promptagator通过使用大语言模型（桑等人，2022；布朗等人，2020；魏等人，2022）进行查询生成（马等人，2021；李等人，2021b；戴等人，2022a）实现少样本任务适应，这具有较高的推理成本。ALIGNER更具通用性，并且可以通过少样本对齐适应快速适应新任务。

## 6 CONCLUSION

## 6 结论

In this paper, we introduce ALIGNER, a novel sparse alignment method for multi-vector document retrieval. We first formulate different retrieval models with token-level sparse alignments and propose ALIGNER to tackle the limitations of existing models. Specifically, ALIGNER uses pairwise alignments and unary saliences that allow us to adapt to different tasks and prune unimportant tokens, respectively. As a result, we achieve strong performance on both zero-shot and few-shot document retrieval tasks while drastically improving the run-time and storage complexity of multi-vector retrieval. With its interpretable alignments and better performance with large language models, we envision that our multi-vector retrieval model can serve as a strong standalone retriever in the future.

在本文中，我们介绍了ALIGNER（对齐器），这是一种用于多向量文档检索的新型稀疏对齐方法。我们首先用词元级别的稀疏对齐构建了不同的检索模型，并提出了ALIGNER来解决现有模型的局限性。具体而言，ALIGNER分别使用成对对齐和一元显著性，使我们能够适应不同的任务并修剪不重要的词元。因此，我们在零样本和少样本文档检索任务中都取得了出色的性能，同时大幅降低了多向量检索的运行时间和存储复杂度。凭借其可解释的对齐方式以及在大语言模型上更好的性能，我们预计我们的多向量检索模型未来可以作为强大的独立检索器。

## 7 AUTHOR CONTRIBUTIONS

## 7 作者贡献

Yujie Qian and Vincent Y. Zhao are the leading authors of this work who designed the model and experiments. All the authors have contributed to the code, experiments, and paper writing. Sai Meher Karthik Duddu conducted the interpretability analysis. Jinhyuk Lee, Zhuyun Dai, Iftekhar Naim, and Tao Lei advised on the research direction. Tao Lei and Siddhartha Brahma proposed the algorithm for entropy-regularized linear programming. Vincent initiated the project and proposed the modeling framework. Vincent and Tao were the co-hosts of Yujie's internship.

钱玉洁（Yujie Qian）和赵文森（Vincent Y. Zhao）是这项工作的主要作者，他们设计了模型和实验。所有作者都参与了代码编写、实验和论文撰写。杜赛·梅赫尔·卡尔蒂克（Sai Meher Karthik Duddu）进行了解释性分析。李镇赫（Jinhyuk Lee）、戴竹云（Zhuyun Dai）、伊夫特哈尔·奈姆（Iftekhar Naim）和陶磊为研究方向提供了建议。陶磊和悉达多·布拉马（Siddhartha Brahma）提出了熵正则化线性规划算法。文森特发起了该项目并提出了建模框架。文森特和陶磊是钱玉洁实习期间的共同导师。

## REFERENCES

## 参考文献

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 1877-1901. Curran Associates, Inc., 2020.

汤姆·布朗（Tom Brown）、本杰明·曼（Benjamin Mann）、尼克·赖德（Nick Ryder）、梅兰妮·苏比亚（Melanie Subbiah）、贾里德·D·卡普兰（Jared D Kaplan）、普拉富拉·达里瓦尔（Prafulla Dhariwal）、阿温德·尼尔坎坦（Arvind Neelakantan）、普拉纳夫·沙姆（Pranav Shyam）、吉里什·萨斯特里（Girish Sastry）、阿曼达·阿斯凯尔（Amanda Askell）、桑迪尼·阿加瓦尔（Sandhini Agarwal）、阿里尔·赫伯特 - 沃斯（Ariel Herbert - Voss）、格雷琴·克鲁格（Gretchen Krueger）、汤姆·海宁汉（Tom Henighan）、雷翁·蔡尔德（Rewon Child）、阿迪亚·拉梅什（Aditya Ramesh）、丹尼尔·齐格勒（Daniel Ziegler）、杰弗里·吴（Jeffrey Wu）、克莱门斯·温特（Clemens Winter）、克里斯·黑塞（Chris Hesse）、马克·陈（Mark Chen）、埃里克·西格勒（Eric Sigler）、马泰乌什·利特温（Mateusz Litwin）、斯科特·格雷（Scott Gray）、本杰明·切斯（Benjamin Chess）、杰克·克拉克（Jack Clark）、克里斯托弗·伯纳尔（Christopher Berner）、山姆·麦坎德利什（Sam McCandlish）、亚历克·拉德福德（Alec Radford）、伊利亚·苏茨克维（Ilya Sutskever）和达里奥·阿莫迪（Dario Amodei）。语言模型是少样本学习者。见H. 拉罗谢尔（H. Larochelle）、M. 兰扎托（M. Ranzato）、R. 哈德塞尔（R. Hadsell）、M.F. 巴尔坎（M.F. Balcan）和H. 林（H. Lin）（编），《神经信息处理系统进展》，第33卷，第1877 - 1901页。柯伦联合公司（Curran Associates, Inc.），2020年。

Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. Advances in neural information processing systems, 26, 2013.

马尔科·库图里（Marco Cuturi）。辛克霍恩距离：最优传输的光速计算。《神经信息处理系统进展》，26，2013年。

Zhuyun Dai and Jamie Callan. Context-aware term weighting for first-stage passage retrieval. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval, 2020.

戴竹云（Zhuyun Dai）和杰米·卡兰（Jamie Callan）。用于第一阶段段落检索的上下文感知词项加权。见《第42届ACM信息检索研究与发展国际会议论文集》，2020年。

Zhuyun Dai, Arun Tejasvi Chaganty, Vincent Y. Zhao, Aida Amini, Qazi Mamunur Rashid, Mike Green, and Kelvin Guu. Dialog inpainting: Turning documents into dialogs. In Kamalika Chaud-huri, Stefanie Jegelka, Le Song, Csaba Szepesvári, Gang Niu, and Sivan Sabato (eds.), International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, volume 162 of Proceedings of Machine Learning Research, pp. 4558-4586. PMLR, 2022a.

戴竹云（Zhuyun Dai）、阿伦·特贾斯维·查甘蒂（Arun Tejasvi Chaganty）、赵文宇（Vincent Y. Zhao）、艾达·阿米尼（Aida Amini）、卡齐·马穆努尔·拉希德（Qazi Mamunur Rashid）、迈克·格林（Mike Green）和凯尔文·顾（Kelvin Guu）。对话补全：将文档转化为对话。见卡玛利卡·乔杜里（Kamalika Chaud-huri）、斯特凡妮·耶格尔卡（Stefanie Jegelka）、宋乐（Le Song）、乔鲍·谢佩斯瓦里（Csaba Szepesvári）、牛刚（Gang Niu）和西万·萨巴托（Sivan Sabato）（编），《国际机器学习会议（ICML 2022）》，2022年7月17 - 23日，美国马里兰州巴尔的摩，《机器学习研究会议录》第162卷，第4558 - 4586页。机器学习研究会议录（PMLR），2022a。

Zhuyun Dai, Vincent Y. Zhao, Ji Ma, Yi Luan, Jianmo Ni, Jing Lu, Anton Bakalov, Kelvin Guu, Keith B. Hall, and Ming-Wei Chang. Promptagator: Few-shot dense retrieval from 8 examples. arXiv preprint arXiv:2209.11755, 2022b.

戴竹云（Zhuyun Dai）、赵文森（Vincent Y. Zhao）、马骥（Ji Ma）、栾毅（Yi Luan）、倪建谟（Jianmo Ni）、卢静（Jing Lu）、安东·巴卡洛夫（Anton Bakalov）、凯尔文·顾（Kelvin Guu）、基思·B·霍尔（Keith B. Hall）和张明伟（Ming-Wei Chang）。Promptagator：基于8个示例的少样本密集检索。预印本arXiv:2209.11755，2022b。

Thibault Formal, Carlos Lassance, Benjamin Piwowarski, and Stéphane Clinchant. SPLADE v2: Sparse lexical and expansion model for information retrieval. CoRR, abs/2109.10086, 2021a.

蒂博·福尔马尔（Thibault Formal）、卡洛斯·拉桑斯（Carlos Lassance）、本杰明·皮沃瓦尔斯基（Benjamin Piwowarski）和斯特凡·克兰尚（Stéphane Clinchant）。SPLADE v2：用于信息检索的稀疏词法与扩展模型。计算机研究报告，编号abs/2109.10086，2021a。

Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant. SPLADE: sparse lexical and expansion model for first stage ranking. In Fernando Diaz, Chirag Shah, Torsten Suel, Pablo Castells, Rosie Jones, and Tetsuya Sakai (eds.), SIGIR '21: The 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, Virtual Event, Canada, July 11-15, 2021, pp. 2288-2292. ACM, 2021b. doi: 10.1145/3404835.3463098.

蒂博·福尔马尔（Thibault Formal）、本杰明·皮沃瓦尔斯基（Benjamin Piwowarski）和斯特凡·克兰尚（Stéphane Clinchant）。SPLADE：用于第一阶段排序的稀疏词汇与扩展模型。收录于费尔南多·迪亚兹（Fernando Diaz）、奇拉格·沙阿（Chirag Shah）、托尔斯滕·苏埃尔（Torsten Suel）、巴勃罗·卡斯特尔斯（Pablo Castells）、罗西·琼斯（Rosie Jones）和酒井哲也（Tetsuya Sakai） 编著的《SIGIR '21：第44届国际计算机协会信息检索研究与发展会议论文集》，2021年7月11 - 15日，加拿大线上会议，第2288 - 2292页。美国计算机协会，2021b。doi: 10.1145/3404835.3463098。

Luyu Gao, Zhuyun Dai, and Jamie Callan. Coil: Revisit exact lexical match in information retrieval with contextualized inverted list. In Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 2021.

高璐宇（Luyu Gao）、戴竹云（Zhuyun Dai）和杰米·卡伦（Jamie Callan）。Coil：借助上下文倒排列表重新审视信息检索中的精确词汇匹配。收录于北美计算语言学协会人类语言技术会议，2021年。

Ruiqi Guo, Philip Sun, Erik Lindgren, Quan Geng, David Simcha, Felix Chern, and Sanjiv Kumar. Accelerating large-scale inference with anisotropic vector quantization. In International Conference on Machine Learning, pp. 3887-3896. PMLR, 2020.

郭瑞琪（Ruiqi Guo）、菲利普·孙（Philip Sun）、埃里克·林德格伦（Erik Lindgren）、耿全（Quan Geng）、大卫·辛查（David Simcha）、费利克斯·陈（Felix Chern）和桑吉夫·库马尔（Sanjiv Kumar）。利用各向异性向量量化加速大规模推理。收录于国际机器学习会议，第3887 - 3896页。机器学习研究会议录，2020年。

Sebastian Hofstätter, Omar Khattab, Sophia Althammer, Mete Sertkan, and Allan Hanbury. Introducing neural bag of whole-words with colberter: Contextualized late interactions using enhanced reduction. arXiv preprint arXiv:2203.13088, 2022.

塞巴斯蒂安·霍夫施泰特（Sebastian Hofstätter）、奥马尔·哈塔卜（Omar Khattab）、索菲娅·阿尔塔默（Sophia Althammer）、梅特·塞尔特坎（Mete Sertkan）和艾伦·汉伯里（Allan Hanbury）。利用ColBERTer引入全词神经词袋：使用增强约简的上下文后期交互。预印本arXiv:2203.13088，2022年。

Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. Unsupervised dense information retrieval with contrastive learning. Transactions on Machine Learning Research, 2022.

高蒂埃·伊扎卡尔（Gautier Izacard）、玛蒂尔德·卡龙（Mathilde Caron）、卢卡斯·侯赛尼（Lucas Hosseini）、塞巴斯蒂安·里德尔（Sebastian Riedel）、彼得·博亚诺夫斯基（Piotr Bojanowski）、阿尔芒·朱兰（Armand Joulin）和爱德华·格雷夫（Edouard Grave）。基于对比学习的无监督密集信息检索。《机器学习研究汇刊》，2022年。

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 6769-6781, 2020.

弗拉基米尔·卡尔普欣（Vladimir Karpukhin）、巴拉斯·奥古兹（Barlas Oguz）、闵世元（Sewon Min）、帕特里克·刘易斯（Patrick Lewis）、莱德尔·吴（Ledell Wu）、谢尔盖·叶杜诺夫（Sergey Edunov）、陈丹琦（Danqi Chen）和易文涛（Wen-tau Yih）。开放域问答的密集段落检索。《2020年自然语言处理经验方法会议（EMNLP）论文集》，第6769 - 6781页，2020年。

Omar Khattab and Matei Zaharia. Colbert: Efficient and effective passage search via contextualized late interaction over bert. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, pp. 39-48, 2020.

奥马尔·哈塔卜（Omar Khattab）和马泰·扎哈里亚（Matei Zaharia）。《科尔伯特：通过基于BERT的上下文延迟交互实现高效有效的段落搜索》。发表于第43届国际计算机协会信息检索研究与发展会议论文集，第39 - 48页，2020年。

Harold W Kuhn and Albert W Tucker. Nonlinear programming. In Traces and emergence of nonlinear programming, pp. 247-258. Springer, 2014.

哈罗德·W·库恩（Harold W Kuhn）和阿尔伯特·W·塔克（Albert W Tucker）。《非线性规划》。收录于《非线性规划的轨迹与出现》，第247 - 258页。施普林格出版社，2014年。

Jinhyuk Lee, Alexander Wettig, and Danqi Chen. Phrase retrieval learns passage retrieval, too. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 3661-3672, 2021a.

李镇赫（Jinhyuk Lee）、亚历山大·韦蒂格（Alexander Wettig）和陈丹琦（Danqi Chen）。《短语检索也能学习段落检索》。发表于2021年自然语言处理经验方法会议论文集，第3661 - 3672页，2021a。

Kenton Lee, Ming-Wei Chang, and Kristina Toutanova. Latent retrieval for weakly supervised open domain question answering. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 6086-6096, 2019.

肯顿·李（Kenton Lee）、张明伟（Ming - Wei Chang）和克里斯蒂娜·图托纳娃（Kristina Toutanova）。《用于弱监督开放域问答的潜在检索》。发表于第57届计算语言学协会年会论文集，第6086 - 6096页，2019年。

Kenton Lee, Kelvin Guu, Luheng He, Tim Dozat, and Hyung Won Chung. Neural data augmentation via example extrapolation. CoRR, abs/2102.01335, 2021b.

肯顿·李（Kenton Lee）、凯尔文·顾（Kelvin Guu）、何路恒（Luheng He）、蒂姆·多扎特（Tim Dozat）和郑亨元（Hyung Won Chung）。《通过示例外推进行神经数据增强》。计算机研究报告，编号abs/2102.01335，2021b。

Jing Lu, Gustavo Hernandez Abrego, Ji Ma, Jianmo Ni, and Yinfei Yang. Multi-stage training with improved negative contrast for neural passage retrieval. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 6091-6103, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/ v1/2021.emnlp-main.492.

陆静（Jing Lu）、古斯塔沃·埃尔南德斯·阿布雷戈（Gustavo Hernandez Abrego）、马骥（Ji Ma）、倪建谟（Jianmo Ni）和杨荫飞（Yinfei Yang）。用于神经段落检索的改进负对比多阶段训练。《2021年自然语言处理经验方法会议论文集》，第6091 - 6103页，线上及多米尼加共和国蓬塔卡纳，2021年11月。计算语言学协会。doi: 10.18653/ v1/2021.emnlp-main.492。

Yi Luan, Jacob Eisenstein, Kristina Toutanova, and Michael Collins. Sparse, dense, and attentional representations for text retrieval. Transactions of the Association for Computational Linguistics, 9:329-345, 2021.

易鸾（Yi Luan）、雅各布·艾森斯坦（Jacob Eisenstein）、克里斯蒂娜·图托纳娃（Kristina Toutanova）和迈克尔·柯林斯（Michael Collins）。用于文本检索的稀疏、密集和注意力表示。《计算语言学协会汇刊》（Transactions of the Association for Computational Linguistics），9:329 - 345，2021年。

Ji Ma, Ivan Korotkov, Yinfei Yang, Keith B. Hall, and Ryan T. McDonald. Zero-shot neural passage retrieval via domain-targeted synthetic question generation. In Paola Merlo, Jörg Tiedemann, and Reut Tsarfaty (eds.), Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, EACL 2021, Online, April 19 - 23, 2021, pp. 1075-1088. Association for Computational Linguistics, 2021. doi: 10.18653/v1/2021. eacl-main.92.

马骥（Ji Ma）、伊万·科罗特科夫（Ivan Korotkov）、殷飞·杨（Yinfei Yang）、基思·B·霍尔（Keith B. Hall）和瑞安·T·麦克唐纳（Ryan T. McDonald）。通过领域目标合成问题生成实现零样本神经段落检索。见保拉·梅洛（Paola Merlo）、约尔格·蒂德曼（Jörg Tiedemann）和鲁特·萨尔法蒂（Reut Tsarfaty）（编），《计算语言学协会欧洲分会第16届会议论文集：主卷》（Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume），EACL 2021，线上会议，2021年4月19 - 23日，第1075 - 1088页。计算语言学协会（Association for Computational Linguistics），2021年。doi: 10.18653/v1/2021. eacl - main.92。

Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, and Li Deng. MS MARCO: A human generated machine reading comprehension dataset. In Tarek Richard Besold, Antoine Bordes, Artur S. d'Avila Garcez, and Greg Wayne (eds.), Proceedings of the Workshop on Cognitive Computation: Integrating neural and symbolic approaches 2016 co-located with the 30th Annual Conference on Neural Information Processing Systems (NIPS 2016), Barcelona, Spain, December 9, 2016, volume 1773 of CEUR Workshop Proceedings. CEUR-WS.org, 2016. URL http://ceur-ws.org/Vol-1773/CoCoNIPS_2016_ paper9.pdf

特里·阮（Tri Nguyen）、米尔·罗森伯格（Mir Rosenberg）、宋霞（Xia Song）、高剑锋（Jianfeng Gao）、索拉布·蒂瓦里（Saurabh Tiwary）、兰根·马朱姆德（Rangan Majumder）和李邓（Li Deng）。《MS MARCO：一个人工生成的机器阅读理解数据集》。收录于塔里克·理查德·贝索尔德（Tarek Richard Besold）、安托万·博尔德斯（Antoine Bordes）、阿图尔·S·达维拉·加尔塞斯（Artur S. d'Avila Garcez）和格雷格·韦恩（Greg Wayne）主编的《认知计算研讨会论文集：整合神经与符号方法（2016）》，该研讨会与第30届神经信息处理系统年度会议（NIPS 2016）同期举办，于2016年12月9日在西班牙巴塞罗那举行，CEUR研讨会论文集第1773卷。CEUR - WS.org，2016年。网址：http://ceur - ws.org/Vol - 1773/CoCoNIPS_2016_ paper9.pdf

Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Hernández Ábrego, Ji Ma, Vincent Y Zhao, Yi Luan, Keith B Hall, Ming-Wei Chang, et al. Large dual encoders are generalizable retrievers. arXiv preprint arXiv:2112.07899, 2021.

倪建墨（Jianmo Ni）、曲晨（Chen Qu）、卢静（Jing Lu）、戴竹云（Zhuyun Dai）、古斯塔沃·埃尔南德斯·阿夫雷戈（Gustavo Hernández Ábrego）、马骥（Ji Ma）、赵文森（Vincent Y Zhao）、栾义（Yi Luan）、基思·B·霍尔（Keith B Hall）、张明伟（Ming - Wei Chang）等。《大型双编码器是可泛化的检索器》。预印本arXiv:2112.07899，2021年。

Barlas Oguz, Kushal Lakhotia, Anchit Gupta, Patrick Lewis, Vladimir Karpukhin, Aleksandra Pik-tus, Xilun Chen, Sebastian Riedel, Scott Yih, Sonal Gupta, and Yashar Mehdad. Domain-matched pre-training tasks for dense retrieval. In Findings of the Association for Computational Linguistics: NAACL 2022, pp. 1524-1534, Seattle, United States, July 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.findings-naacl.114.

巴拉斯·奥古兹（Barlas Oguz）、库沙尔·拉科蒂亚（Kushal Lakhotia）、安奇特·古普塔（Anchit Gupta）、帕特里克·刘易斯（Patrick Lewis）、弗拉基米尔·卡尔普欣（Vladimir Karpukhin）、亚历山德拉·皮克图斯（Aleksandra Pik-tus）、陈希伦（Xilun Chen）、塞巴斯蒂安·里德尔（Sebastian Riedel）、斯科特·伊（Scott Yih）、索纳尔·古普塔（Sonal Gupta）和亚沙尔·梅赫达德（Yashar Mehdad）。用于密集检索的领域匹配预训练任务。见《计算语言学协会研究成果：2022年北美计算语言学协会年会》，第1524 - 1534页，美国西雅图，2022年7月。计算语言学协会。doi: 10.18653/v1/2022.findings - naacl.114。

Yingqi Qu, Yuchen Ding, Jing Liu, Kai Liu, Ruiyang Ren, Wayne Xin Zhao, Daxiang Dong, Hua Wu, and Haifeng Wang. Rocketqa: An optimized training approach to dense passage retrieval for open-domain question answering. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 5835-5847, 2021.

曲英琦（Yingqi Qu）、丁雨晨（Yuchen Ding）、刘静（Jing Liu）、刘凯（Kai Liu）、任瑞阳（Ruiyang Ren）、赵鑫（Wayne Xin Zhao）、董大祥（Daxiang Dong）、吴华（Hua Wu）和王海峰（Haifeng Wang）。Rocketqa：一种用于开放域问答的密集段落检索优化训练方法。见《2021年计算语言学协会北美分会人类语言技术会议论文集》，第5835 - 5847页，2021年。

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21:1-67, 2020.

科林·拉菲尔（Colin Raffel）、诺姆·沙泽尔（Noam Shazeer）、亚当·罗伯茨（Adam Roberts）、凯瑟琳·李（Katherine Lee）、沙兰·纳朗（Sharan Narang）、迈克尔·马特纳（Michael Matena）、周燕琪（Yanqi Zhou）、李伟（Wei Li）和彼得·J·刘（Peter J Liu）。使用统一的文本到文本转换器探索迁移学习的极限。《机器学习研究杂志》，21:1 - 67，2020年。

Sashank J. Reddi, Rama Kumar Pasumarthi, Aditya Krishna Menon, Ankit Singh Rawat, Felix X. Yu, Seungyeon Kim, Andreas Veit, and Sanjiv Kumar. Rankdistil: Knowledge distillation for ranking. In AISTATS, pp. 2368-2376, 2021. URL http://proceedings.mlr.press/ v130/reddi21a.html

萨尚克·J·雷迪（Sashank J. Reddi）、拉玛·库马尔·帕苏马尔蒂（Rama Kumar Pasumarthi）、阿迪亚·克里希纳·梅农（Aditya Krishna Menon）、安基特·辛格·拉瓦特（Ankit Singh Rawat）、费利克斯·X·于（Felix X. Yu）、李承妍（Seungyeon Kim）、安德烈亚斯·维特（Andreas Veit）和桑吉夫·库马尔（Sanjiv Kumar）。Rankdistil：用于排序的知识蒸馏。收录于《人工智能与统计学年会》（AISTATS），第2368 - 2376页，2021年。网址：http://proceedings.mlr.press/ v130/reddi21a.html

Victor Sanh, Albert Webson, Colin Raffel, Stephen Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Arun Raja, Manan Dey, M Saiful Bari, Canwen Xu, Urmish Thakker, Shanya Sharma Sharma, Eliza Szczechla, Taewoon Kim, Gunjan Chhablani, Nihal V. Nayak, Debajyoti Datta, Jonathan Chang, Mike Tian-Jian Jiang, Han Wang, Matteo Manica, Sheng Shen, Zheng Xin Yong, Harshit Pandey, Rachel Bawden, Thomas Wang, Trishala Neeraj, Jos Rozen, Abheesht Sharma, Andrea Santilli, Thibault Févry, Jason Alan Fries, Ryan Teehan, Teven Le Scao, Stella Biderman, Leo Gao, Thomas Wolf, and Alexander M. Rush. Multitask prompted training enables zero-shot task generalization. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. URL https://openreview.net/forum?id=9Vrb9DOWI4

维克多·桑（Victor Sanh）、阿尔伯特·韦伯森（Albert Webson）、科林·拉菲尔（Colin Raffel）、斯蒂芬·巴赫（Stephen Bach）、林唐·苏塔维卡（Lintang Sutawika）、扎伊德·阿利亚菲（Zaid Alyafeai）、安托万·查芬（Antoine Chaffin）、阿诺·斯蒂格勒（Arnaud Stiegler）、阿伦·拉贾（Arun Raja）、马南·戴伊（Manan Dey）、M·赛富尔·巴里（M Saiful Bari）、徐灿文（Canwen Xu）、乌尔米什·萨克（Urmish Thakker）、珊亚·夏尔马·夏尔马（Shanya Sharma Sharma）、伊莱扎·什切赫拉（Eliza Szczechla）、金泰云（Taewoon Kim）、贡扬·查布拉尼（Gunjan Chhablani）、尼哈尔·V·纳亚克（Nihal V. Nayak）、德巴乔蒂·达塔（Debajyoti Datta）、乔纳森·张（Jonathan Chang）、迈克·田健·江（Mike Tian-Jian Jiang）、王翰（Han Wang）、马泰奥·马尼卡（Matteo Manica）、沈盛（Sheng Shen）、郑新勇（Zheng Xin Yong）、哈什特·潘迪（Harshit Pandey）、瑞秋·鲍登（Rachel Bawden）、托马斯·王（Thomas Wang）、特里沙拉·尼尔拉吉（Trishala Neeraj）、乔斯·罗森（Jos Rozen）、阿比什特·夏尔马（Abheesht Sharma）、安德里亚·桑蒂利（Andrea Santilli）、蒂博·费夫里（Thibault Févry）、杰森·艾伦·弗里斯（Jason Alan Fries）、瑞安·蒂汉（Ryan Teehan）、特文·勒·斯考（Teven Le Scao）、斯特拉·比德曼（Stella Biderman）、高磊（Leo Gao）、托马斯·沃尔夫（Thomas Wolf）和亚历山大·M·拉什（Alexander M. Rush）。多任务提示训练实现零样本任务泛化。见第十届国际学习表征会议（The Tenth International Conference on Learning Representations，ICLR 2022），虚拟会议，2022年4月25 - 29日。OpenReview.net，2022年。网址：https://openreview.net/forum?id=9Vrb9DOWI4

Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. Colbertv2: Effective and efficient retrieval via lightweight late interaction. arXiv preprint arXiv:2112.01488, 2021.

凯沙夫·桑塔纳姆（Keshav Santhanam）、奥马尔·哈塔卜（Omar Khattab）、乔恩·萨德 - 法尔孔（Jon Saad-Falcon）、克里斯托弗·波茨（Christopher Potts）和马泰·扎哈里亚（Matei Zaharia）。《Colbertv2：通过轻量级后期交互实现有效且高效的检索》。预印本 arXiv:2112.01488，2021年。

Christopher Sciavolino, Zexuan Zhong, Jinhyuk Lee, and Danqi Chen. Simple entity-centric questions challenge dense retrievers. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 6138-6148, 2021.

克里斯托弗·夏沃利诺（Christopher Sciavolino）、钟泽轩（Zexuan Zhong）、李晋赫（Jinhyuk Lee）和陈丹琦（Danqi Chen）。《以简单实体为中心的问题对密集检索器构成挑战》。收录于《2021年自然语言处理经验方法会议论文集》，第6138 - 6148页，2021年。

Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. Beir: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021.

南丹·塔库尔（Nandan Thakur）、尼尔斯·赖默斯（Nils Reimers）、安德里亚斯·吕克莱（Andreas Rücklé）、阿比舍克·斯里瓦斯塔瓦（Abhishek Srivastava）和伊琳娜·古列维奇（Iryna Gurevych）。《BEIR：信息检索模型零样本评估的异构基准》。收录于《第三十五届神经信息处理系统大会数据集与基准赛道（第二轮）》，2021年。

Jason Wei, Maarten Paul Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew Mingbo Dai, and Quoc V. Le. Finetuned language models are zero-shot learners. In International Conference on Learning Representations, 2022.

杰森·魏（Jason Wei）、马腾·保罗·博斯马（Maarten Paul Bosma）、文森特·赵（Vincent Zhao）、凯尔文·顾（Kelvin Guu）、亚当斯·魏·余（Adams Wei Yu）、布莱恩·莱斯特（Brian Lester）、杜楠（Nan Du）、戴明波（Andrew Mingbo Dai）和奎克·V·乐（Quoc V. Le）。《微调语言模型是零样本学习者》。收录于《国际学习表征会议》，2022年。

Stephen J Wright. Coordinate descent algorithms. Mathematical Programming, 151(1):3-34, 2015.

斯蒂芬·J·赖特（Stephen J Wright）。坐标下降算法。《数学规划》（Mathematical Programming），151(1):3 - 34, 2015年。

Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N Bennett, Junaid Ahmed, and Arnold Overwijk. Approximate nearest neighbor negative contrastive learning for dense text retrieval. In International Conference on Learning Representations, 2020.

熊李（Lee Xiong）、熊晨燕（Chenyan Xiong）、李烨（Ye Li）、邓国峰（Kwok - Fung Tang）、刘佳琳（Jialin Liu）、保罗·N·贝内特（Paul N Bennett）、朱奈德·艾哈迈德（Junaid Ahmed）和阿诺德·奥弗维克（Arnold Overwijk）。用于密集文本检索的近似最近邻负对比学习。收录于《国际学习表征会议》（International Conference on Learning Representations），2020年。

Hang Zhang, Yeyun Gong, Yelong Shen, Jiancheng Lv, Nan Duan, and Weizhu Chen. Adversarial retriever-ranker for dense text retrieval. In International Conference on Learning Representations, 2022.

张航（Hang Zhang）、龚叶云（Yeyun Gong）、沈业龙（Yelong Shen）、吕建成（Jiancheng Lv）、段楠（Nan Duan）和陈伟柱（Weizhu Chen）。用于密集文本检索的对抗式检索 - 排序器。收录于《国际学习表征会议》（International Conference on Learning Representations），2022年。

Tiancheng Zhao, Xiaopeng Lu, and Kyusong Lee. SPARTA: efficient open-domain question answering via sparse transformer matching retrieval. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT, pp. 565-575, 2021.

赵天成（Tiancheng Zhao）、卢小鹏（Xiaopeng Lu）和李奎松（Kyusong Lee）。SPARTA：通过稀疏Transformer匹配检索实现高效的开放域问答。收录于《计算语言学协会北美分会2021年会议：人类语言技术》（Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies），NAACL - HLT，第565 - 575页，2021年。

## A APPENDIX

## A 附录

### A.1 DERIVATION OF THE ITERATIVE UPDATES

### A.1 迭代更新的推导

We present the derivation of Eq 5 for solving optimization problem (4) in Section 3.2. The maximization problem (4) can be written as an equivalent minimization problem:

我们给出第3.2节中求解优化问题(4)的方程5的推导过程。最大化问题(4)可以写成一个等价的最小化问题：

$$
\mathop{\max }\limits_{\mathbf{\lambda }}{\mathbf{s}}^{\top }\mathbf{\lambda } + {\varepsilon H}\left( \mathbf{\lambda }\right) 
$$

$$
 \Leftrightarrow  \;\mathop{\min }\limits_{\mathbf{\lambda }} - {\mathbf{s}}^{\top }\mathbf{\lambda } - {\varepsilon H}\left( \mathbf{\lambda }\right) 
$$

$$
 \Leftrightarrow  \;\mathop{\min }\limits_{\mathbf{\lambda }} - {\mathbf{s}}^{\top }\mathbf{\lambda } - {\varepsilon H}\left( \mathbf{\lambda }\right)  - \varepsilon {\mathbf{1}}^{\top }\mathbf{\lambda } \tag{6}
$$

$$
\text{s.t.}{\mathbf{1}}^{\top }\mathbf{\lambda } = k,{\lambda }_{i} \in  \left\lbrack  {0,1}\right\rbrack  ,i = 1,\ldots ,m\text{.}
$$

Note the term $\varepsilon {\mathbf{1}}^{\top }\mathbf{\lambda }$ will be a constant $\varepsilon  \times  k$ ,but we include it in the minimization object to make our derivation simpler later.

注意，项$\varepsilon {\mathbf{1}}^{\top }\mathbf{\lambda }$将是一个常数$\varepsilon  \times  k$，但我们将其包含在最小化目标中，以便后续推导更简单。

Now,let $a \in  \mathbb{R}$ and $\mathbf{b} \in  {\mathbb{R}}^{m}$ be the Lagrangian variables corresponding to the linear constraints ${\mathbf{1}}^{\top }\mathbf{\lambda } = k$ and ${\lambda }_{i} \leq  1\forall i$ [7] The minimization problem is equivalent to its Lagrangian expression:

现在，设$a \in  \mathbb{R}$和$\mathbf{b} \in  {\mathbb{R}}^{m}$为对应于线性约束${\mathbf{1}}^{\top }\mathbf{\lambda } = k$和${\lambda }_{i} \leq  1\forall i$的拉格朗日变量[7]。该最小化问题等价于其拉格朗日表达式：

$$
\mathop{\min }\limits_{{\mathbf{\lambda } \in  {\mathbb{R}}^{m}}}\mathop{\max }\limits_{{a \in  \mathbb{R},\mathbf{b} \leq  \mathbf{0}}} - {\mathbf{s}}^{\top }\mathbf{\lambda } - {\varepsilon H}\left( \mathbf{\lambda }\right)  - \varepsilon {\mathbf{1}}^{\top }\mathbf{\lambda } + a\left( {k - {\mathbf{1}}^{\top }\mathbf{\lambda }}\right)  + {\mathbf{b}}^{\top }\left( {\mathbf{1} - \mathbf{\lambda }}\right)  \tag{7}
$$

The objective function (6) is strongly convex and the solution space of $\lambda$ is a convex set. As a result, strong duality holds and we can instead solve the dual problem that exchanges the min and max operators in [7]

目标函数(6)是强凸的，且$\lambda$的解空间是一个凸集。因此，强对偶性成立，我们可以转而求解在[7]中交换了最小和最大运算符的对偶问题。

$$
\mathop{\max }\limits_{{a \in  \mathbb{R},\mathbf{b} \leq  \mathbf{0}}}\mathop{\min }\limits_{{\lambda  \in  {\mathbb{R}}^{m}}} - {\mathbf{s}}^{\top }\mathbf{\lambda } - {\varepsilon H}\left( \mathbf{\lambda }\right)  - \varepsilon {\mathbf{1}}^{\top }\mathbf{\lambda } + a\left( {k - {\mathbf{1}}^{\top }\mathbf{\lambda }}\right)  + {\mathbf{b}}^{\top }\left( {\mathbf{1} - \mathbf{\lambda }}\right)  \tag{8}
$$

The optimal solution $\left( {a,\mathbf{b},\mathbf{\lambda }}\right)$ must have the Karush-Kuhn-Tucker (KKT) conditions hold (Kuhn & Tucker, 2014), namely

最优解 $\left( {a,\mathbf{b},\mathbf{\lambda }}\right)$ 必须满足卡罗需-库恩-塔克（Karush-Kuhn-Tucker，KKT）条件（库恩（Kuhn）和塔克（Tucker），2014 年），即

$$
\frac{\partial \left( {-{\mathbf{s}}^{\top }\mathbf{\lambda } - {\varepsilon H}\left( \mathbf{\lambda }\right)  + \varepsilon {\mathbf{1}}^{\top }\mathbf{\lambda } + a\left( {k - {\mathbf{1}}^{\top }\mathbf{\lambda }}\right)  + {\mathbf{b}}^{\top }\left( {\mathbf{1} - \mathbf{\lambda }}\right) }\right) }{\partial \mathbf{\lambda }} = 0
$$

$$
 \Leftrightarrow  \;\mathbf{\lambda } = \exp \left( \frac{\mathbf{s} + a + \mathbf{b}}{\varepsilon }\right)  \Leftrightarrow  \;{\lambda }_{i} = \exp \left( \frac{{s}_{i} + a + {\mathbf{b}}_{i}}{\varepsilon }\right) \forall i = 1,\ldots ,m
$$

Substituting $\lambda$ using the above equation in (8),the dual problem now has a simple form:

将上述方程中的 $\lambda$ 代入式（8），对偶问题现在具有一个简单的形式：

$$
\mathop{\max }\limits_{{a \in  \mathbb{R},b \leq  0}}k \cdot  a + {\mathbf{1}}^{\top }\mathbf{b} - {\mathbf{1}}^{\top }\exp \left( \frac{\mathbf{s} + a + \mathbf{b}}{\varepsilon }\right) 
$$

We can solve this problem using coordinate descent (Wright, 2015) by successively maximizing the function with either $a$ or $b$ fixed. This leads to the iterative updates (Eq. 5) described in Section 3.2

我们可以使用坐标下降法（赖特（Wright），2015 年）来解决这个问题，即通过依次固定 $a$ 或 $b$ 来最大化该函数。这将得到第 3.2 节中描述的迭代更新公式（式 5）

$$
{a}^{\prime } = \varepsilon \ln \left( k\right)  - \varepsilon \ln \left\{  {\mathop{\sum }\limits_{i}\exp \left( \frac{{s}_{i} + {b}_{i}}{\varepsilon }\right) }\right\}  
$$

$$
{b}_{i}^{\prime } = \min \left( {-{s}_{i} - {a}^{\prime },0}\right) 
$$

Discussion In short, we solve the dual problem of optimization (4) by performing coordinate decent of the dual variables $a$ and $\mathbf{b}$ . That is,we find the optimal $a$ that maximizes the dual objective given a fixed $\mathbf{b}$ ,and vice versa.

讨论 简而言之，我们通过对对偶变量 $a$ 和 $\mathbf{b}$ 进行坐标下降来解决优化问题（4）的对偶问题。也就是说，我们在固定 $\mathbf{b}$ 的情况下找到使对偶目标最大化的最优 $a$，反之亦然。

This iterative algorithm is also closely related to the Sinkhorn algorithm of Optimal Transport (OT). In fact, Sinkhorn algorithm solves the entropy-regularized version of Optimal Transport (Cuturi, 2013). However, our work concerns an different optimization instance. While OT solves a transportation problem where the solution space is defined with the marginal constraints over the rows and columns of a transportation matrix, our optimization problem is constrained with a total budget $\left( {\mathop{\sum }\limits_{i}{\lambda }_{i} = k}\right)$ and upper bounds $\left( {{\lambda }_{i} \leq  1\forall i}\right)$ . This leads to different iterative updates.

这种迭代算法还与最优传输（Optimal Transport，OT）的辛克霍恩算法（Sinkhorn algorithm）密切相关。事实上，辛克霍恩算法解决的是最优传输的熵正则化版本（库蒂里（Cuturi），2013 年）。然而，我们的工作涉及一个不同的优化实例。最优传输解决的是一个运输问题，其解空间是由运输矩阵的行和列的边缘约束定义的，而我们的优化问题则受到总预算 $\left( {\mathop{\sum }\limits_{i}{\lambda }_{i} = k}\right)$ 和上限 $\left( {{\lambda }_{i} \leq  1\forall i}\right)$ 的约束。这导致了不同的迭代更新。

---

<!-- Footnote -->

${}^{7}{\lambda }_{i} \geq  0\forall i$ is already implied by the entropy term $H\left( \mathbf{\lambda }\right)$ in the objective.

目标函数中的熵项 $H\left( \mathbf{\lambda }\right)$ 已经隐含了 ${}^{7}{\lambda }_{i} \geq  0\forall i$。

<!-- Footnote -->

---

### A.2 DIFFERENTIABLE ALIGNMENT WITH SPARSITY CONSTRAINTS

### A.2 具有稀疏性约束的可微对齐

Besides the Top- $k$ and Top- $p$ alignments in §3.1,we also explore a differentiable pairwise alignment with sparsity contraints (DA). Both Top- $k$ adn Top- $p$ are doing hard selection of alignments,i.e., ${\widetilde{\mathbf{A}}}_{i,j}$ is either 1 or 0 . We relax it by introducing soft sparsity constraints. Similar to our formulation for unary salience (§3.2),we determine the alignment $\widetilde{\mathbf{A}}$ by the following optimization problem:

除了§3.1中的Top- $k$和Top- $p$对齐方式外，我们还探索了一种带有稀疏性约束的可微成对对齐方式（DA）。Top- $k$和Top- $p$都在进行对齐的硬选择，即${\widetilde{\mathbf{A}}}_{i,j}$要么为1，要么为0。我们通过引入软稀疏性约束来放宽这一条件。与我们对一元显著性的公式化（§3.2）类似，我们通过以下优化问题来确定对齐$\widetilde{\mathbf{A}}$：

$$
\mathop{\max }\limits_{\mathbf{A}}\langle \mathbf{S},\mathbf{A}\rangle  + {\varepsilon H}\left( \mathbf{A}\right) 
$$

$$
\text{s.t.}\mathop{\sum }\limits_{j}{\mathbf{A}}_{i,j} = k,i = 1,\ldots ,n \tag{9}
$$

$$
{\mathbf{A}}_{i,j} \in  \left\lbrack  {0,1}\right\rbrack  ,i = 1,\ldots ,n,j = 1,\ldots ,m
$$

where $H\left( \cdot \right)$ is the elementwise entropy function and $\varepsilon  > 0$ is a small constant. We constrain the sum of each row of $\widetilde{\mathbf{A}}$ to equal $k$ . When $\varepsilon  = 0$ ,the solution of Eq. 9 is the same as Top- $k$ . When $\varepsilon  > 0$ ,the entropy term makes the optimization problem strongly concave,which can be solved by the same algorithm in Appendix A.1. The solution is differentiable, thus can be trained end-to-end in our model.

其中 $H\left( \cdot \right)$ 是逐元素熵函数，$\varepsilon  > 0$ 是一个小常数。我们将 $\widetilde{\mathbf{A}}$ 的每一行元素之和约束为等于 $k$。当 $\varepsilon  = 0$ 时，方程9的解与Top - $k$ 相同。当 $\varepsilon  > 0$ 时，熵项使优化问题具有强凹性，可通过附录A.1中的相同算法求解。该解是可微的，因此可以在我们的模型中进行端到端训练。

<!-- Media -->

### A.3 QUALITATIVE ANALYSIS

### A.3 定性分析

<table><tr><td>Dataset</td><td>Query</td><td>Gold Document</td></tr><tr><td>Quora</td><td>what is the best birth- day gift for a friend?</td><td>what ${}^{\left( 3\right) }$ is ${}^{\left( 2\right) }$ a ${}^{\left( 4\right) }$ good ${}^{\left( 1\right) }$ birthday gift for a friend?</td></tr><tr><td>MS MARCO (dev)</td><td>when would you use a fathom measurement</td><td>a fathom ${}^{\left( 1\right) }$ is a unit of length in the imperial and the u.s. customary systems equal to $6{\text{feet}}^{\left( 3\right) }\left( {{1.8288}{\mathrm{\;m}}^{\left( 4\right) }}\right)$ ,used especially for measur- ing the depth of water. there are two yards ( 6 feet) in an imperial fathom(2) ${}^{\left( 2\right) 3}$ .</td></tr><tr><td>Touché- 2020</td><td>should animals be used for scientific or commercial testing?</td><td>animal testing should not be allowed...[truncated]... albeit the non- precocious mistakes of scientists ${}^{\left( 2\right) }$ . also...[truncated]... skeptic of the scientist ${}^{\left( 3\right) }$ in question’s abilities ... [truncated]... continuous use if animals for clinical ${}^{\left( 4\right) }$ and basic research." ... [truncated]... majority of the scientific ${}^{\left( 1\right) }$ community thinks on this issue, ...</td></tr></table>

<table><tbody><tr><td>数据集</td><td>查询</td><td>标准文档</td></tr><tr><td>Quora（夸夸网）</td><td>给朋友的最佳生日礼物是什么？</td><td>给朋友的${}^{\left( 3\right) }$是${}^{\left( 2\right) }$一${}^{\left( 4\right) }$份好${}^{\left( 1\right) }$的生日礼物是什么？</td></tr><tr><td>微软机器阅读理解数据集（开发集）（MS MARCO (dev)）</td><td>你什么时候会使用英寻这个度量单位</td><td>英寻（fathom ${}^{\left( 1\right) }$）是英制和美国惯用制中的长度单位，等于$6{\text{feet}}^{\left( 3\right) }\left( {{1.8288}{\mathrm{\;m}}^{\left( 4\right) }}\right)$，尤其用于测量水深。一英寻等于两码（6英尺）(2) ${}^{\left( 2\right) 3}$。</td></tr><tr><td>妙哉——2020</td><td>动物应该用于科学或商业测试吗？</td><td>不应允许进行动物实验……[截断]……尽管科学家们会犯一些并非早有预谋的错误${}^{\left( 2\right) }$。此外……[截断]……对相关科学家${}^{\left( 3\right) }$的能力表示怀疑……[截断]……持续将动物用于临床${}^{\left( 4\right) }$和基础研究。”……[截断]……科学界${}^{\left( 1\right) }$的大多数人对这个问题的看法是……</td></tr></tbody></table>

<!-- Media -->

Table 5: Examples of pairwise alignment with the top- $k$ value up to 4 for the Quora,MS MARCO, and Touché-2020 datasets. Query tokens being aligned are shown in blue, and corresponding aligned document tokens are shown in red. The superscript on the document token(k)indicates top- $k$ alignment. We notice that the top-1 alignment quality is generally good across all three tasks. However, larger value of $k$ results in spurious irrelevant alignments for Quora and MS MARCO,while remains fairly useful for Touché-2020.

表5：针对Quora、MS MARCO和Touché - 2020数据集，前$k$值高达4的成对对齐示例。正在对齐的查询词元以蓝色显示，对应的对齐文档词元以红色显示。文档词元(k)上的上标表示前$k$对齐。我们注意到，在所有三项任务中，前1对齐质量总体上较好。然而，对于Quora和MS MARCO，$k$值越大，会导致出现虚假的不相关对齐，而对于Touché - 2020，该值较大时仍然相当有用。

Table 5 shows examples of top- $k$ pairwise alignments of a query token (highlighted in blue) to the corresponding document tokens for several different tasks. For question-answering (e.g., MS MARCO) and duplicate question retrieval (Quora), fewer alignments seem to be preferable, and as $k$ increases,we start to see spurious alignments to unrelated documents tokens. For argument retrieval tasks such as Touché-2020,on the other hand,larger value of $k$ tends to provide useful semantically relevant alignments (e.g., scientific vs clinical). These qualitative examples provide intuitive insights regarding why different alignment strategies are helpful for different tasks, and why alignment adaptation is necessary.

表5展示了在几个不同任务中，查询词元（以蓝色突出显示）与相应文档词元的前$k$个成对对齐示例。对于问答任务（例如MS MARCO）和重复问题检索（Quora），较少的对齐似乎更可取，并且随着$k$的增加，我们开始看到与不相关文档词元的虚假对齐。另一方面，对于诸如Touché - 2020之类的论点检索任务，$k$值越大往往能提供有用的语义相关对齐（例如，科学与临床方面的对齐）。这些定性示例直观地说明了为什么不同的对齐策略对不同任务有帮助，以及为什么对齐调整是必要的。

### A.4 Results on MS MARCO

### A.4 在MS MARCO上的结果

Table 6 shows the retrieval performance of ALIGNER and previous models on the MS MARCO dev set. We deliberately kept the training configuration of ALIGNER relatively simpler (e.g., no distillation or model-based hard negatives). However, ALIGNER still achieves the best MRR@10 simply because of scaling to larger pretrained language models. We have also trained ALIGNER with other alignment strategies, such as top-4, top-1%, and DA (Appendix A.2). However, the results suggest top-1 is favorable in MS MARCO.

表6展示了ALIGNER（对齐器）和先前模型在MS MARCO开发集上的检索性能。我们特意将ALIGNER的训练配置保持得相对简单（例如，不进行蒸馏或使用基于模型的难负样本）。然而，仅仅因为扩展到更大的预训练语言模型，ALIGNER仍然实现了最佳的MRR@10（前10个结果的平均倒数排名）。我们还使用其他对齐策略（如前4、前1%和DA（附录A.2））训练了ALIGNER。不过，结果表明在MS MARCO中，前1策略更有利。

<!-- Media -->

<table><tr><td>Model</td><td>MRR@10</td><td>Recall@1000</td></tr><tr><td>BM25</td><td>18.7</td><td>85.7</td></tr><tr><td>${\mathrm{{SPLADE}}}_{\mathrm{v}2}$</td><td>36.8</td><td>97.9</td></tr><tr><td>DPR</td><td>31.1</td><td>95.2</td></tr><tr><td>${\mathrm{{GTR}}}_{\text{base }}$</td><td>36.6</td><td>98.3</td></tr><tr><td>${\mathrm{{GTR}}}_{\text{large }}$</td><td>37.9</td><td>99.1</td></tr><tr><td>${\mathrm{{GTR}}}_{\mathrm{{xl}}}$</td><td>38.5</td><td>98.9</td></tr><tr><td>${\mathrm{{GTR}}}_{\mathrm{{xxl}}}$</td><td>38.8</td><td>99.0</td></tr><tr><td>ColBERT</td><td>36.0</td><td>96.8</td></tr><tr><td>ColBERTv2</td><td>39.7</td><td>98.4</td></tr><tr><td>COIL</td><td>35.5</td><td>96.3</td></tr><tr><td>ME-BERT</td><td>33.4</td><td>-</td></tr><tr><td>${\mathrm{{ALIGNER}}}_{\text{base}}$</td><td>38.8</td><td>97.8</td></tr><tr><td>${\mathrm{{ALIGNER}}}_{\text{large }}$</td><td>39.4</td><td>98.3</td></tr><tr><td>${\mathrm{{ALIGNER}}}_{\mathrm{x\rbrack }}$</td><td>39.9</td><td>98.4</td></tr><tr><td>${\mathrm{{ALIGNER}}}_{\mathrm{{xxl}}}$</td><td>40.3</td><td>98.7</td></tr><tr><td>ALIGNERbase (top-4)</td><td>37.1</td><td>97.5</td></tr><tr><td>${\mathrm{{ALIGNER}}}_{\text{base }}$ (top-1%)</td><td>38.8</td><td>97.6</td></tr><tr><td>ALIGNER ${}_{\text{base }}$ (DA)</td><td>37.8</td><td>97.4</td></tr></table>

<table><tbody><tr><td>模型</td><td>前10名平均倒数排名（MRR@10）</td><td>前1000名召回率（Recall@1000）</td></tr><tr><td>二元独立模型（BM25）</td><td>18.7</td><td>85.7</td></tr><tr><td>${\mathrm{{SPLADE}}}_{\mathrm{v}2}$</td><td>36.8</td><td>97.9</td></tr><tr><td>密集段落检索器（DPR）</td><td>31.1</td><td>95.2</td></tr><tr><td>${\mathrm{{GTR}}}_{\text{base }}$</td><td>36.6</td><td>98.3</td></tr><tr><td>${\mathrm{{GTR}}}_{\text{large }}$</td><td>37.9</td><td>99.1</td></tr><tr><td>${\mathrm{{GTR}}}_{\mathrm{{xl}}}$</td><td>38.5</td><td>98.9</td></tr><tr><td>${\mathrm{{GTR}}}_{\mathrm{{xxl}}}$</td><td>38.8</td><td>99.0</td></tr><tr><td>列伯特（ColBERT）</td><td>36.0</td><td>96.8</td></tr><tr><td>科尔伯特v2（ColBERTv2）</td><td>39.7</td><td>98.4</td></tr><tr><td>线圈（COIL）</td><td>35.5</td><td>96.3</td></tr><tr><td>多编码器BERT（ME-BERT）</td><td>33.4</td><td>-</td></tr><tr><td>${\mathrm{{ALIGNER}}}_{\text{base}}$</td><td>38.8</td><td>97.8</td></tr><tr><td>${\mathrm{{ALIGNER}}}_{\text{large }}$</td><td>39.4</td><td>98.3</td></tr><tr><td>${\mathrm{{ALIGNER}}}_{\mathrm{x\rbrack }}$</td><td>39.9</td><td>98.4</td></tr><tr><td>${\mathrm{{ALIGNER}}}_{\mathrm{{xxl}}}$</td><td>40.3</td><td>98.7</td></tr><tr><td>对齐器基础版（前4名）（ALIGNERbase (top-4)）</td><td>37.1</td><td>97.5</td></tr><tr><td>[乳胶0]（前1%）（${\mathrm{{ALIGNER}}}_{\text{base }}$ (top-1%)）</td><td>38.8</td><td>97.6</td></tr><tr><td>对齐器 [乳胶0]（数据增强）（ALIGNER ${}_{\text{base }}$ (DA)）</td><td>37.8</td><td>97.4</td></tr></tbody></table>

Table 6: Retrieval performance on MS MARCO. The top half shows baselines from previous work. The botton half shows different ALIGNER models. DA: differential alignment. See Appendix A. 2

表6：在MS MARCO上的检索性能。上半部分展示了以往工作的基线。下半部分展示了不同的ALIGNER模型。DA：差异对齐。参见附录A.2

### A.5 FULL RESULT TABLES ON BEIR

### A.5 BEIR上的完整结果表

Tables 7 to 10 presents complete results of ALIGNER's performance on the BEIR datasets initialized from T5 base,large,large,XL,and XXL checkpoints. We set $k = 1$ during training,and show results across different inference-time alignment strategies (both top- $k$ and top- $p$ ). As expected,model accuracy improves as we scale to larger models. Moreover, we observe similar benefits of alignment adaptation across all the different model sizes.

表7至表10展示了ALIGNER在BEIR数据集上的完整性能结果，这些数据集分别从T5的基础版、大型版、大型版、XL版和XXL版检查点进行初始化。我们在训练期间设置了$k = 1$，并展示了不同推理时对齐策略（前$k$和前$p$）下的结果。正如预期的那样，随着模型规模的增大，模型准确率有所提高。此外，我们在所有不同规模的模型中都观察到了对齐自适应带来的类似益处。

<table><tr><td colspan="2" rowspan="2">${\mathrm{{ALIGNER}}}_{\text{base }}$</td><td colspan="5">Top- $k$</td><td colspan="4">Top- $p$</td></tr><tr><td>1*</td><td>2</td><td>4</td><td>6</td><td>8</td><td>0.5%</td><td>1%</td><td>1.5%</td><td>2%</td></tr><tr><td rowspan="2">A.R.</td><td>ArguAna</td><td>28.8</td><td>24.4</td><td>18.3</td><td>14.5</td><td>11.4</td><td>33.3</td><td>45.5</td><td>48.1</td><td>46.9</td></tr><tr><td>Touché-2020</td><td>34.8</td><td>50.0</td><td>51.1</td><td>49.3</td><td>46.0</td><td>31.3</td><td>24.2</td><td>20.3</td><td>15.8</td></tr><tr><td rowspan="3">F.C.</td><td>FEVER</td><td>72.4</td><td>75.0</td><td>68.3</td><td>57.6</td><td>49.0</td><td>72.5</td><td>55.0</td><td>44.9</td><td>29.9</td></tr><tr><td>Climate-FEVER</td><td>18.1</td><td>20.8</td><td>23.0</td><td>23.0</td><td>22.7</td><td>18.2</td><td>13.7</td><td>13.8</td><td>10.8</td></tr><tr><td>SciFact</td><td>70.4</td><td>68.8</td><td>65.0</td><td>60.9</td><td>55.6</td><td>71.1</td><td>69.4</td><td>67.2</td><td>62.7</td></tr><tr><td rowspan="5">Q.A.</td><td>NQ</td><td>52.2</td><td>48.3</td><td>36.3</td><td>26.6</td><td>19.9</td><td>52.2</td><td>49.2</td><td>43.5</td><td>36.3</td></tr><tr><td>HotpotQA</td><td>61.7</td><td>58.6</td><td>36.0</td><td>21.9</td><td>13.9</td><td>61.7</td><td>60.1</td><td>54.3</td><td>47.3</td></tr><tr><td>FiQA</td><td>33.4</td><td>30.8</td><td>23.8</td><td>19.5</td><td>15.5</td><td>33.4</td><td>28.5</td><td>24.6</td><td>19.0</td></tr><tr><td>BioASQ</td><td>49.6</td><td>45.8</td><td>37.4</td><td>30.8</td><td>24.5</td><td>47.4</td><td>37.7</td><td>31.9</td><td>25.0</td></tr><tr><td>NFCorpus</td><td>34.0</td><td>33.2</td><td>32.0</td><td>29.9</td><td>27.8</td><td>33.6</td><td>31.7</td><td>30.5</td><td>28.6</td></tr><tr><td rowspan="4">MISC.</td><td>TREC-COVID</td><td>68.3</td><td>74.0</td><td>73.2</td><td>67.2</td><td>61.7</td><td>66.8</td><td>64.4</td><td>56.7</td><td>46.7</td></tr><tr><td>SCIDOCS</td><td>14.1</td><td>14.3</td><td>13.1</td><td>11.4</td><td>9.7</td><td>14.4</td><td>14.9</td><td>14.9</td><td>14.8</td></tr><tr><td>DBPedia</td><td>41.6</td><td>39.4</td><td>29.6</td><td>20.2</td><td>14.2</td><td>41.6</td><td>41.7</td><td>39.9</td><td>36.6</td></tr><tr><td>Quora</td><td>82.3</td><td>64.9</td><td>30.6</td><td>13.3</td><td>6.3</td><td>82.3</td><td>82.3</td><td>82.3</td><td>82.3</td></tr><tr><td/><td>Average</td><td>47.3</td><td>46.3</td><td>38.4</td><td>31.9</td><td>27.0</td><td>47.1</td><td>44.2</td><td>40.9</td><td>35.9</td></tr></table>

<table><tbody><tr><td colspan="2" rowspan="2">${\mathrm{{ALIGNER}}}_{\text{base }}$</td><td colspan="5">顶部 - $k$</td><td colspan="4">顶部 - $p$</td></tr><tr><td>1*</td><td>2</td><td>4</td><td>6</td><td>8</td><td>0.5%</td><td>1%</td><td>1.5%</td><td>2%</td></tr><tr><td rowspan="2">评估报告（Assessment Report）</td><td>论证分析（Argument Analysis）</td><td>28.8</td><td>24.4</td><td>18.3</td><td>14.5</td><td>11.4</td><td>33.3</td><td>45.5</td><td>48.1</td><td>46.9</td></tr><tr><td>妙语 - 2020</td><td>34.8</td><td>50.0</td><td>51.1</td><td>49.3</td><td>46.0</td><td>31.3</td><td>24.2</td><td>20.3</td><td>15.8</td></tr><tr><td rowspan="3">足球俱乐部（Football Club）</td><td>发热（FEVER）</td><td>72.4</td><td>75.0</td><td>68.3</td><td>57.6</td><td>49.0</td><td>72.5</td><td>55.0</td><td>44.9</td><td>29.9</td></tr><tr><td>气候发热数据集（Climate-FEVER）</td><td>18.1</td><td>20.8</td><td>23.0</td><td>23.0</td><td>22.7</td><td>18.2</td><td>13.7</td><td>13.8</td><td>10.8</td></tr><tr><td>科学事实数据集（SciFact）</td><td>70.4</td><td>68.8</td><td>65.0</td><td>60.9</td><td>55.6</td><td>71.1</td><td>69.4</td><td>67.2</td><td>62.7</td></tr><tr><td rowspan="5">问答（Q.A.）</td><td>自然问题数据集（NQ）</td><td>52.2</td><td>48.3</td><td>36.3</td><td>26.6</td><td>19.9</td><td>52.2</td><td>49.2</td><td>43.5</td><td>36.3</td></tr><tr><td>火锅问答数据集（HotpotQA）</td><td>61.7</td><td>58.6</td><td>36.0</td><td>21.9</td><td>13.9</td><td>61.7</td><td>60.1</td><td>54.3</td><td>47.3</td></tr><tr><td>金融问答数据集（FiQA）</td><td>33.4</td><td>30.8</td><td>23.8</td><td>19.5</td><td>15.5</td><td>33.4</td><td>28.5</td><td>24.6</td><td>19.0</td></tr><tr><td>生物医学问答数据集（BioASQ）</td><td>49.6</td><td>45.8</td><td>37.4</td><td>30.8</td><td>24.5</td><td>47.4</td><td>37.7</td><td>31.9</td><td>25.0</td></tr><tr><td>新闻事实语料库（NFCorpus）</td><td>34.0</td><td>33.2</td><td>32.0</td><td>29.9</td><td>27.8</td><td>33.6</td><td>31.7</td><td>30.5</td><td>28.6</td></tr><tr><td rowspan="4">其他</td><td>TREC新冠疫情问答数据集（TREC - COVID）</td><td>68.3</td><td>74.0</td><td>73.2</td><td>67.2</td><td>61.7</td><td>66.8</td><td>64.4</td><td>56.7</td><td>46.7</td></tr><tr><td>科学文献数据集（SCIDOCS）</td><td>14.1</td><td>14.3</td><td>13.1</td><td>11.4</td><td>9.7</td><td>14.4</td><td>14.9</td><td>14.9</td><td>14.8</td></tr><tr><td>DBpedia（DBPedia）</td><td>41.6</td><td>39.4</td><td>29.6</td><td>20.2</td><td>14.2</td><td>41.6</td><td>41.7</td><td>39.9</td><td>36.6</td></tr><tr><td>Quora（Quora）</td><td>82.3</td><td>64.9</td><td>30.6</td><td>13.3</td><td>6.3</td><td>82.3</td><td>82.3</td><td>82.3</td><td>82.3</td></tr><tr><td></td><td>平均值</td><td>47.3</td><td>46.3</td><td>38.4</td><td>31.9</td><td>27.0</td><td>47.1</td><td>44.2</td><td>40.9</td><td>35.9</td></tr></tbody></table>

Table 7: nDCG@10 on the BEIR benchmark with different $k$ and $p$ in ALIGNER ${}_{\text{base }}$ . *: alignment strategy during training $\left( {k = 1}\right)$ .

表7：在BEIR基准测试中，ALIGNER ${}_{\text{base }}$ 使用不同的 $k$ 和 $p$ 时的nDCG@10值。*：训练期间的对齐策略 $\left( {k = 1}\right)$。

<table><tr><td colspan="2" rowspan="2">${\mathrm{{ALIGNER}}}_{\text{large }}$</td><td colspan="5">Top- $k$</td><td colspan="4">Top- $p$</td></tr><tr><td>1*</td><td>2</td><td>4</td><td>6</td><td>8</td><td>0.5%</td><td>1%</td><td>1.5%</td><td>2%</td></tr><tr><td rowspan="2">A.R.</td><td>ArguAna</td><td>29.5</td><td>25.2</td><td>19.5</td><td>16.0</td><td>13.6</td><td>33.2</td><td>44.5</td><td>47.9</td><td>46.8</td></tr><tr><td>Touché-2020</td><td>36.7</td><td>47.5</td><td>53.0</td><td>53.4</td><td>52.0</td><td>32.5</td><td>25.4</td><td>20.7</td><td>16.3</td></tr><tr><td rowspan="3">F.C.</td><td>FEVER</td><td>72.9</td><td>75.2</td><td>69.6</td><td>61.8</td><td>53.6</td><td>72.8</td><td>52.8</td><td>43.0</td><td>29.3</td></tr><tr><td>Climate-FEVER</td><td>18.6</td><td>20.7</td><td>23.3</td><td>23.4</td><td>23.5</td><td>18.6</td><td>13.1</td><td>14.2</td><td>11.4</td></tr><tr><td>SciFact</td><td>71.5</td><td>70.7</td><td>69.5</td><td>67.7</td><td>64.1</td><td>72.2</td><td>70.6</td><td>69.6</td><td>66.8</td></tr><tr><td rowspan="5">Q.A.</td><td>NQ</td><td>57.2</td><td>52.8</td><td>43.4</td><td>36.3</td><td>31.2</td><td>57.2</td><td>53.7</td><td>47.5</td><td>41.0</td></tr><tr><td>HotpotQA</td><td>63.6</td><td>62.6</td><td>44.8</td><td>32.1</td><td>24.4</td><td>63.6</td><td>61.8</td><td>55.6</td><td>48.5</td></tr><tr><td>FiQA</td><td>39.4</td><td>35.6</td><td>30.4</td><td>26.2</td><td>23.4</td><td>39.4</td><td>33.3</td><td>28.4</td><td>22.8</td></tr><tr><td>BioASQ</td><td>53.3</td><td>50.7</td><td>43.1</td><td>36.8</td><td>31.6</td><td>49.2</td><td>38.2</td><td>33.2</td><td>27.4</td></tr><tr><td>NFCorpus</td><td>35.5</td><td>33.9</td><td>32.5</td><td>31.2</td><td>29.6</td><td>34.6</td><td>32.1</td><td>30.8</td><td>29.6</td></tr><tr><td rowspan="4">MISC.</td><td>TREC-COVID</td><td>71.9</td><td>79.4</td><td>77.3</td><td>74.4</td><td>69.3</td><td>70.0</td><td>66.1</td><td>59.7</td><td>52.4</td></tr><tr><td>SCIDOCS</td><td>15.3</td><td>15.5</td><td>14.9</td><td>13.6</td><td>12.5</td><td>15.4</td><td>15.8</td><td>16.0</td><td>15.8</td></tr><tr><td>DBPedia</td><td>43.5</td><td>41.9</td><td>34.7</td><td>29.0</td><td>24.3</td><td>43.5</td><td>43.5</td><td>41.5</td><td>37.4</td></tr><tr><td>Quora</td><td>84.5</td><td>75.5</td><td>46.4</td><td>20.9</td><td>8.4</td><td>84.5</td><td>84.5</td><td>84.5</td><td>84.5</td></tr><tr><td/><td>Average</td><td>49.5</td><td>49.1</td><td>43.0</td><td>37.3</td><td>33.0</td><td>49.0</td><td>45.4</td><td>42.3</td><td>37.9</td></tr></table>

<table><tbody><tr><td colspan="2" rowspan="2">${\mathrm{{ALIGNER}}}_{\text{large }}$</td><td colspan="5">顶部 - $k$</td><td colspan="4">顶部 - $p$</td></tr><tr><td>1*</td><td>2</td><td>4</td><td>6</td><td>8</td><td>0.5%</td><td>1%</td><td>1.5%</td><td>2%</td></tr><tr><td rowspan="2">评估报告（A.R.）</td><td>论证分析（ArguAna）</td><td>29.5</td><td>25.2</td><td>19.5</td><td>16.0</td><td>13.6</td><td>33.2</td><td>44.5</td><td>47.9</td><td>46.8</td></tr><tr><td>妙语连珠2020（Touché - 2020）</td><td>36.7</td><td>47.5</td><td>53.0</td><td>53.4</td><td>52.0</td><td>32.5</td><td>25.4</td><td>20.7</td><td>16.3</td></tr><tr><td rowspan="3">足球俱乐部（F.C.）</td><td>发热（FEVER）</td><td>72.9</td><td>75.2</td><td>69.6</td><td>61.8</td><td>53.6</td><td>72.8</td><td>52.8</td><td>43.0</td><td>29.3</td></tr><tr><td>气候发热数据集（Climate-FEVER）</td><td>18.6</td><td>20.7</td><td>23.3</td><td>23.4</td><td>23.5</td><td>18.6</td><td>13.1</td><td>14.2</td><td>11.4</td></tr><tr><td>科学事实数据集（SciFact）</td><td>71.5</td><td>70.7</td><td>69.5</td><td>67.7</td><td>64.1</td><td>72.2</td><td>70.6</td><td>69.6</td><td>66.8</td></tr><tr><td rowspan="5">问答（Q.A.）</td><td>自然问题数据集（NQ）</td><td>57.2</td><td>52.8</td><td>43.4</td><td>36.3</td><td>31.2</td><td>57.2</td><td>53.7</td><td>47.5</td><td>41.0</td></tr><tr><td>火锅问答数据集（HotpotQA）</td><td>63.6</td><td>62.6</td><td>44.8</td><td>32.1</td><td>24.4</td><td>63.6</td><td>61.8</td><td>55.6</td><td>48.5</td></tr><tr><td>金融问答数据集（FiQA）</td><td>39.4</td><td>35.6</td><td>30.4</td><td>26.2</td><td>23.4</td><td>39.4</td><td>33.3</td><td>28.4</td><td>22.8</td></tr><tr><td>生物医学问答数据集（BioASQ）</td><td>53.3</td><td>50.7</td><td>43.1</td><td>36.8</td><td>31.6</td><td>49.2</td><td>38.2</td><td>33.2</td><td>27.4</td></tr><tr><td>新闻事实语料库（NFCorpus）</td><td>35.5</td><td>33.9</td><td>32.5</td><td>31.2</td><td>29.6</td><td>34.6</td><td>32.1</td><td>30.8</td><td>29.6</td></tr><tr><td rowspan="4">其他</td><td>TREC新冠疫情问答数据集（TREC - COVID）</td><td>71.9</td><td>79.4</td><td>77.3</td><td>74.4</td><td>69.3</td><td>70.0</td><td>66.1</td><td>59.7</td><td>52.4</td></tr><tr><td>科学文献数据集（SCIDOCS）</td><td>15.3</td><td>15.5</td><td>14.9</td><td>13.6</td><td>12.5</td><td>15.4</td><td>15.8</td><td>16.0</td><td>15.8</td></tr><tr><td>DBPedia（DBPedia）</td><td>43.5</td><td>41.9</td><td>34.7</td><td>29.0</td><td>24.3</td><td>43.5</td><td>43.5</td><td>41.5</td><td>37.4</td></tr><tr><td>Quora（Quora）</td><td>84.5</td><td>75.5</td><td>46.4</td><td>20.9</td><td>8.4</td><td>84.5</td><td>84.5</td><td>84.5</td><td>84.5</td></tr><tr><td></td><td>平均值</td><td>49.5</td><td>49.1</td><td>43.0</td><td>37.3</td><td>33.0</td><td>49.0</td><td>45.4</td><td>42.3</td><td>37.9</td></tr></tbody></table>

Table 8: nDCG@10 on the BEIR benchmark with different $k$ and $p$ in ALIGNERlarge. *: alignment strategy during training $\left( {k = 1}\right)$ .

表8：在BEIR基准测试中，ALIGNERlarge使用不同的$k$和$p$时的nDCG@10值。*：训练期间的对齐策略$\left( {k = 1}\right)$。

<table><tr><td colspan="2" rowspan="2">${\mathrm{{ALIGNER}}}_{\mathrm{{xl}}}$</td><td colspan="5">Top- $k$</td><td colspan="4">Top- $p$</td></tr><tr><td>1*</td><td>2</td><td>4</td><td>6</td><td>8</td><td>0.5%</td><td>1%</td><td>1.5%</td><td>2%</td></tr><tr><td rowspan="2">A.R.</td><td>ArguAna</td><td>32.4</td><td>28.3</td><td>22.5</td><td>18.9</td><td>16.3</td><td>35.3</td><td>44.9</td><td>47.2</td><td>47.0</td></tr><tr><td>Touché-2020</td><td>36.2</td><td>46.4</td><td>53.2</td><td>51.0</td><td>50.5</td><td>33.0</td><td>26.1</td><td>21.9</td><td>18.2</td></tr><tr><td rowspan="3">F.C.</td><td>FEVER</td><td>72.9</td><td>75.6</td><td>70.9</td><td>63.2</td><td>56.0</td><td>72.9</td><td>56.9</td><td>48.2</td><td>34.1</td></tr><tr><td>Climate-FEVER</td><td>18.7</td><td>21.2</td><td>23.2</td><td>23.9</td><td>24.2</td><td>18.8</td><td>15.2</td><td>16.6</td><td>14.3</td></tr><tr><td>SciFact</td><td>71.5</td><td>70.8</td><td>69.7</td><td>67.6</td><td>65.6</td><td>73.0</td><td>71.6</td><td>69.9</td><td>69.5</td></tr><tr><td rowspan="5">Q.A.</td><td>NQ</td><td>58.8</td><td>54.9</td><td>46.0</td><td>39.5</td><td>34.1</td><td>58.8</td><td>55.5</td><td>50.1</td><td>43.8</td></tr><tr><td>HotpotQA</td><td>63.9</td><td>62.6</td><td>45.5</td><td>32.9</td><td>25.0</td><td>63.9</td><td>62.5</td><td>57.5</td><td>51.4</td></tr><tr><td>FiQA</td><td>40.8</td><td>37.4</td><td>32.2</td><td>28.9</td><td>25.2</td><td>40.8</td><td>35.0</td><td>31.2</td><td>25.7</td></tr><tr><td>BioASQ</td><td>53.6</td><td>50.4</td><td>43.0</td><td>36.6</td><td>32.2</td><td>50.2</td><td>39.7</td><td>34.4</td><td>28.9</td></tr><tr><td>NFCorpus</td><td>35.4</td><td>34.3</td><td>33.0</td><td>31.5</td><td>29.7</td><td>35.0</td><td>33.1</td><td>31.6</td><td>30.0</td></tr><tr><td rowspan="4">MISC.</td><td>TREC-COVID</td><td>75.1</td><td>80.6</td><td>80.5</td><td>78.1</td><td>73.5</td><td>72.5</td><td>69.4</td><td>62.0</td><td>54.0</td></tr><tr><td>SCIDOCS</td><td>15.4</td><td>16.0</td><td>15.5</td><td>14.3</td><td>13.3</td><td>15.6</td><td>16.2</td><td>16.5</td><td>16.4</td></tr><tr><td>DBPedia</td><td>43.6</td><td>42.2</td><td>36.2</td><td>30.6</td><td>26.5</td><td>43.6</td><td>43.6</td><td>42.4</td><td>39.4</td></tr><tr><td>Quora</td><td>85.3</td><td>76.5</td><td>50.1</td><td>26.9</td><td>13.1</td><td>85.3</td><td>85.3</td><td>85.3</td><td>85.3</td></tr><tr><td/><td>Average</td><td>50.3</td><td>49.8</td><td>44.4</td><td>38.9</td><td>34.7</td><td>49.9</td><td>46.8</td><td>43.9</td><td>39.8</td></tr></table>

<table><tbody><tr><td colspan="2" rowspan="2">${\mathrm{{ALIGNER}}}_{\mathrm{{xl}}}$</td><td colspan="5">顶部 - $k$</td><td colspan="4">顶部 - $p$</td></tr><tr><td>1*</td><td>2</td><td>4</td><td>6</td><td>8</td><td>0.5%</td><td>1%</td><td>1.5%</td><td>2%</td></tr><tr><td rowspan="2">评估报告（A.R.）</td><td>论证分析（ArguAna）</td><td>32.4</td><td>28.3</td><td>22.5</td><td>18.9</td><td>16.3</td><td>35.3</td><td>44.9</td><td>47.2</td><td>47.0</td></tr><tr><td>妙语连珠2020（Touché - 2020）</td><td>36.2</td><td>46.4</td><td>53.2</td><td>51.0</td><td>50.5</td><td>33.0</td><td>26.1</td><td>21.9</td><td>18.2</td></tr><tr><td rowspan="3">足球俱乐部（F.C.）</td><td>发热（FEVER）</td><td>72.9</td><td>75.6</td><td>70.9</td><td>63.2</td><td>56.0</td><td>72.9</td><td>56.9</td><td>48.2</td><td>34.1</td></tr><tr><td>气候发热数据集（Climate-FEVER）</td><td>18.7</td><td>21.2</td><td>23.2</td><td>23.9</td><td>24.2</td><td>18.8</td><td>15.2</td><td>16.6</td><td>14.3</td></tr><tr><td>科学事实数据集（SciFact）</td><td>71.5</td><td>70.8</td><td>69.7</td><td>67.6</td><td>65.6</td><td>73.0</td><td>71.6</td><td>69.9</td><td>69.5</td></tr><tr><td rowspan="5">问答（Q.A.）</td><td>自然问题数据集（NQ）</td><td>58.8</td><td>54.9</td><td>46.0</td><td>39.5</td><td>34.1</td><td>58.8</td><td>55.5</td><td>50.1</td><td>43.8</td></tr><tr><td>火锅问答数据集（HotpotQA）</td><td>63.9</td><td>62.6</td><td>45.5</td><td>32.9</td><td>25.0</td><td>63.9</td><td>62.5</td><td>57.5</td><td>51.4</td></tr><tr><td>金融问答数据集（FiQA）</td><td>40.8</td><td>37.4</td><td>32.2</td><td>28.9</td><td>25.2</td><td>40.8</td><td>35.0</td><td>31.2</td><td>25.7</td></tr><tr><td>生物医学问答数据集（BioASQ）</td><td>53.6</td><td>50.4</td><td>43.0</td><td>36.6</td><td>32.2</td><td>50.2</td><td>39.7</td><td>34.4</td><td>28.9</td></tr><tr><td>新闻事实语料库（NFCorpus）</td><td>35.4</td><td>34.3</td><td>33.0</td><td>31.5</td><td>29.7</td><td>35.0</td><td>33.1</td><td>31.6</td><td>30.0</td></tr><tr><td rowspan="4">其他</td><td>TREC新冠数据集（TREC - COVID）</td><td>75.1</td><td>80.6</td><td>80.5</td><td>78.1</td><td>73.5</td><td>72.5</td><td>69.4</td><td>62.0</td><td>54.0</td></tr><tr><td>科学文献数据集（SCIDOCS）</td><td>15.4</td><td>16.0</td><td>15.5</td><td>14.3</td><td>13.3</td><td>15.6</td><td>16.2</td><td>16.5</td><td>16.4</td></tr><tr><td>DBPedia（DBPedia）</td><td>43.6</td><td>42.2</td><td>36.2</td><td>30.6</td><td>26.5</td><td>43.6</td><td>43.6</td><td>42.4</td><td>39.4</td></tr><tr><td>Quora（Quora）</td><td>85.3</td><td>76.5</td><td>50.1</td><td>26.9</td><td>13.1</td><td>85.3</td><td>85.3</td><td>85.3</td><td>85.3</td></tr><tr><td></td><td>平均值</td><td>50.3</td><td>49.8</td><td>44.4</td><td>38.9</td><td>34.7</td><td>49.9</td><td>46.8</td><td>43.9</td><td>39.8</td></tr></tbody></table>

Table 9: nDCG@10 on the BEIR benchmark with different $k$ and $p$ in ${\mathrm{{ALIGNER}}}_{\mathrm{{xl}}}$ . *: alignment strategy during training $\left( {k = 1}\right)$ .

表9：在BEIR基准测试中，${\mathrm{{ALIGNER}}}_{\mathrm{{xl}}}$ 里不同 $k$ 和 $p$ 对应的nDCG@10值。*：训练 $\left( {k = 1}\right)$ 期间的对齐策略。

<table><tr><td colspan="2" rowspan="2">${\mathrm{{ALIGNER}}}_{\mathrm{{xxl}}}$</td><td colspan="5">Top- $k$</td><td colspan="4">Top- $p$</td></tr><tr><td>1*</td><td>2</td><td>4</td><td>6</td><td>8</td><td>0.5%</td><td>1%</td><td>1.5%</td><td>2%</td></tr><tr><td rowspan="2">A.R.</td><td>ArguAna</td><td>33.8</td><td>29.6</td><td>24.1</td><td>20.4</td><td>18.2</td><td>36.6</td><td>46.9</td><td>49.8</td><td>49.7</td></tr><tr><td>Touché-2020</td><td>34.5</td><td>47.4</td><td>49.8</td><td>51.1</td><td>50.6</td><td>30.1</td><td>21.8</td><td>16.3</td><td>11.7</td></tr><tr><td rowspan="3">F.C.</td><td>FEVER</td><td>74.2</td><td>77.0</td><td>73.9</td><td>67.9</td><td>62.2</td><td>75.2</td><td>47.1</td><td>36.2</td><td>24.1</td></tr><tr><td>Climate-FEVER</td><td>19.7</td><td>21.6</td><td>23.7</td><td>23.2</td><td>24.3</td><td>19.7</td><td>12.0</td><td>12.3</td><td>9.3</td></tr><tr><td>SciFact</td><td>73.1</td><td>71.2</td><td>71.3</td><td>69.1</td><td>67.0</td><td>74.4</td><td>71.5</td><td>69.9</td><td>69.5</td></tr><tr><td rowspan="5">Q.A.</td><td>NQ</td><td>60.5</td><td>56.0</td><td>49.1</td><td>44.0</td><td>40.1</td><td>60.4</td><td>54.5</td><td>45.8</td><td>37.9</td></tr><tr><td>HotpotQA</td><td>65.2</td><td>63.4</td><td>48.8</td><td>37.7</td><td>30.1</td><td>65.2</td><td>63.0</td><td>56.0</td><td>48.1</td></tr><tr><td>FiQA</td><td>43.5</td><td>40.3</td><td>36.8</td><td>33.8</td><td>31.4</td><td>43.5</td><td>35.9</td><td>30.4</td><td>24.0</td></tr><tr><td>BioASQ</td><td>54.8</td><td>51.1</td><td>43.3</td><td>38.2</td><td>34.4</td><td>49.6</td><td>35.8</td><td>29.4</td><td>24.9</td></tr><tr><td>NFCorpus</td><td>35.2</td><td>34.0</td><td>32.6</td><td>31.5</td><td>30.3</td><td>34.1</td><td>29.7</td><td>28.0</td><td>27.1</td></tr><tr><td rowspan="4">MISC.</td><td>TREC-COVID</td><td>75.8</td><td>81.4</td><td>80.1</td><td>75.6</td><td>71.9</td><td>75.1</td><td>65.4</td><td>54.0</td><td>45.9</td></tr><tr><td>SCIDOCS</td><td>17.1</td><td>16.8</td><td>16.3</td><td>15.6</td><td>14.2</td><td>17.1</td><td>17.0</td><td>16.6</td><td>15.9</td></tr><tr><td>DBPedia</td><td>45.0</td><td>43.2</td><td>38.3</td><td>33.9</td><td>30.6</td><td>45.0</td><td>44.9</td><td>42.6</td><td>35.7</td></tr><tr><td>Quora</td><td>86.0</td><td>79.0</td><td>58.6</td><td>38.2</td><td>22.8</td><td>86.0</td><td>86.0</td><td>86.0</td><td>86.0</td></tr><tr><td/><td>Average</td><td>51.3</td><td>50.9</td><td>46.2</td><td>41.4</td><td>37.7</td><td>50.8</td><td>45.1</td><td>40.9</td><td>36.4</td></tr></table>

<table><tbody><tr><td colspan="2" rowspan="2">${\mathrm{{ALIGNER}}}_{\mathrm{{xxl}}}$</td><td colspan="5">顶部 - $k$</td><td colspan="4">顶部 - $p$</td></tr><tr><td>1*</td><td>2</td><td>4</td><td>6</td><td>8</td><td>0.5%</td><td>1%</td><td>1.5%</td><td>2%</td></tr><tr><td rowspan="2">艺术总监（Artistic Director，这里推测A.R.可能是类似缩写，需结合具体场景，暂按常见缩写处理）</td><td>论证分析（ArguAna，推测是Argument Analysis的缩写，需结合具体场景）</td><td>33.8</td><td>29.6</td><td>24.1</td><td>20.4</td><td>18.2</td><td>36.6</td><td>46.9</td><td>49.8</td><td>49.7</td></tr><tr><td>妙啊 - 2020</td><td>34.5</td><td>47.4</td><td>49.8</td><td>51.1</td><td>50.6</td><td>30.1</td><td>21.8</td><td>16.3</td><td>11.7</td></tr><tr><td rowspan="3">足球俱乐部（Football Club）</td><td>发热（FEVER）</td><td>74.2</td><td>77.0</td><td>73.9</td><td>67.9</td><td>62.2</td><td>75.2</td><td>47.1</td><td>36.2</td><td>24.1</td></tr><tr><td>气候发热数据集（Climate-FEVER）</td><td>19.7</td><td>21.6</td><td>23.7</td><td>23.2</td><td>24.3</td><td>19.7</td><td>12.0</td><td>12.3</td><td>9.3</td></tr><tr><td>科学事实数据集（SciFact）</td><td>73.1</td><td>71.2</td><td>71.3</td><td>69.1</td><td>67.0</td><td>74.4</td><td>71.5</td><td>69.9</td><td>69.5</td></tr><tr><td rowspan="5">问答（Q.A.）</td><td>自然问题数据集（NQ）</td><td>60.5</td><td>56.0</td><td>49.1</td><td>44.0</td><td>40.1</td><td>60.4</td><td>54.5</td><td>45.8</td><td>37.9</td></tr><tr><td>火锅问答数据集（HotpotQA）</td><td>65.2</td><td>63.4</td><td>48.8</td><td>37.7</td><td>30.1</td><td>65.2</td><td>63.0</td><td>56.0</td><td>48.1</td></tr><tr><td>金融问答数据集（FiQA）</td><td>43.5</td><td>40.3</td><td>36.8</td><td>33.8</td><td>31.4</td><td>43.5</td><td>35.9</td><td>30.4</td><td>24.0</td></tr><tr><td>生物医学问答数据集（BioASQ）</td><td>54.8</td><td>51.1</td><td>43.3</td><td>38.2</td><td>34.4</td><td>49.6</td><td>35.8</td><td>29.4</td><td>24.9</td></tr><tr><td>新闻事实语料库（NFCorpus）</td><td>35.2</td><td>34.0</td><td>32.6</td><td>31.5</td><td>30.3</td><td>34.1</td><td>29.7</td><td>28.0</td><td>27.1</td></tr><tr><td rowspan="4">其他</td><td>TREC新冠疫情问答数据集（TREC - COVID）</td><td>75.8</td><td>81.4</td><td>80.1</td><td>75.6</td><td>71.9</td><td>75.1</td><td>65.4</td><td>54.0</td><td>45.9</td></tr><tr><td>科学文献数据集（SCIDOCS）</td><td>17.1</td><td>16.8</td><td>16.3</td><td>15.6</td><td>14.2</td><td>17.1</td><td>17.0</td><td>16.6</td><td>15.9</td></tr><tr><td>DBPedia（DBPedia）</td><td>45.0</td><td>43.2</td><td>38.3</td><td>33.9</td><td>30.6</td><td>45.0</td><td>44.9</td><td>42.6</td><td>35.7</td></tr><tr><td>Quora（Quora）</td><td>86.0</td><td>79.0</td><td>58.6</td><td>38.2</td><td>22.8</td><td>86.0</td><td>86.0</td><td>86.0</td><td>86.0</td></tr><tr><td></td><td>平均值</td><td>51.3</td><td>50.9</td><td>46.2</td><td>41.4</td><td>37.7</td><td>50.8</td><td>45.1</td><td>40.9</td><td>36.4</td></tr></tbody></table>

Table 10: nDCG@10 on the BEIR benchmark with different $k$ and $p$ in ALIGNER ${}_{\mathrm{{xxl}}}$ . *: alignment strategy during training $\left( {k = 1}\right)$ .

表10：在BEIR基准测试中，ALIGNER ${}_{\mathrm{{xxl}}}$ 使用不同的 $k$ 和 $p$ 时的nDCG@10值。*：训练期间的对齐策略 $\left( {k = 1}\right)$。

<table><tr><td colspan="2" rowspan="2">${\mathrm{{ALIGNER}}}_{\text{base }}$</td><td colspan="5">Top- $k$</td><td colspan="4">Top- $p$</td></tr><tr><td>1</td><td>2</td><td>${4}^{ * }$</td><td>6</td><td>8</td><td>0.5%</td><td>1%</td><td>1.5%</td><td>2%</td></tr><tr><td rowspan="2">A.R.</td><td>ArguAna</td><td>26.5</td><td>23.6</td><td>18.8</td><td>15.9</td><td>13.6</td><td>30.7</td><td>42.8</td><td>45.7</td><td>46.2</td></tr><tr><td>Touché-2020</td><td>25.2</td><td>32.4</td><td>43.9</td><td>48.2</td><td>49.5</td><td>19.7</td><td>14.6</td><td>11.9</td><td>9.5</td></tr><tr><td rowspan="3">F.C.</td><td>FEVER</td><td>59.0</td><td>67.0</td><td>72.8</td><td>73.6</td><td>73.0</td><td>59.0</td><td>34.3</td><td>25.4</td><td>18.4</td></tr><tr><td>Climate-FEVER</td><td>14.7</td><td>17.5</td><td>20.9</td><td>22.8</td><td>23.7</td><td>14.7</td><td>7.8</td><td>7.1</td><td>5.1</td></tr><tr><td>SciFact</td><td>70.1</td><td>70.6</td><td>70.4</td><td>69.2</td><td>68.0</td><td>69.6</td><td>66.7</td><td>65.6</td><td>64.9</td></tr><tr><td rowspan="5">Q.A.</td><td>NQ</td><td>42.3</td><td>49.4</td><td>52.4</td><td>51.2</td><td>49.1</td><td>42.3</td><td>37.3</td><td>30.9</td><td>25.3</td></tr><tr><td>HotpotQA</td><td>57.6</td><td>61.0</td><td>60.4</td><td>57.4</td><td>53.0</td><td>57.6</td><td>55.3</td><td>49.4</td><td>43.4</td></tr><tr><td>FiQA</td><td>29.8</td><td>32.2</td><td>32.0</td><td>30.1</td><td>27.8</td><td>29.9</td><td>22.2</td><td>17.7</td><td>14.0</td></tr><tr><td>BioASQ</td><td>48.3</td><td>50.9</td><td>49.0</td><td>48.0</td><td>45.9</td><td>41.1</td><td>26.4</td><td>21.1</td><td>17.4</td></tr><tr><td>NFCorpus</td><td>31.8</td><td>33.5</td><td>33.9</td><td>33.6</td><td>33.0</td><td>29.9</td><td>26.3</td><td>25.7</td><td>25.2</td></tr><tr><td rowspan="4">MISC.</td><td>TREC-COVID</td><td>54.8</td><td>63.5</td><td>72.9</td><td>75.4</td><td>74.5</td><td>53.1</td><td>48.7</td><td>42.5</td><td>38.1</td></tr><tr><td>SCIDOCS</td><td>13.2</td><td>14.1</td><td>14.7</td><td>14.6</td><td>14.0</td><td>13.2</td><td>12.2</td><td>11.7</td><td>11.5</td></tr><tr><td>DBPedia</td><td>31.5</td><td>39.1</td><td>42.1</td><td>39.9</td><td>36.9</td><td>31.4</td><td>31.4</td><td>29.7</td><td>25.1</td></tr><tr><td>Quora</td><td>82.7</td><td>80.5</td><td>72.1</td><td>57.4</td><td>41.9</td><td>82.7</td><td>82.7</td><td>82.7</td><td>82.7</td></tr><tr><td/><td>Average</td><td>42.0</td><td>45.4</td><td>46.9</td><td>45.5</td><td>43.1</td><td>41.1</td><td>36.3</td><td>33.4</td><td>30.5</td></tr></table>

<table><tbody><tr><td colspan="2" rowspan="2">${\mathrm{{ALIGNER}}}_{\text{base }}$</td><td colspan="5">顶部 - $k$</td><td colspan="4">顶部 - $p$</td></tr><tr><td>1</td><td>2</td><td>${4}^{ * }$</td><td>6</td><td>8</td><td>0.5%</td><td>1%</td><td>1.5%</td><td>2%</td></tr><tr><td rowspan="2">艺术总监（Artistic Director，这里推测A.R.可能是类似缩写，需结合具体语境，暂按常见缩写处理）</td><td>论证分析（ArguAna，推测是Argument Analysis的缩写，需结合具体语境）</td><td>26.5</td><td>23.6</td><td>18.8</td><td>15.9</td><td>13.6</td><td>30.7</td><td>42.8</td><td>45.7</td><td>46.2</td></tr><tr><td>妙啊 - 2020</td><td>25.2</td><td>32.4</td><td>43.9</td><td>48.2</td><td>49.5</td><td>19.7</td><td>14.6</td><td>11.9</td><td>9.5</td></tr><tr><td rowspan="3">足球俱乐部（Football Club）</td><td>发热（FEVER）</td><td>59.0</td><td>67.0</td><td>72.8</td><td>73.6</td><td>73.0</td><td>59.0</td><td>34.3</td><td>25.4</td><td>18.4</td></tr><tr><td>气候发热数据集（Climate-FEVER）</td><td>14.7</td><td>17.5</td><td>20.9</td><td>22.8</td><td>23.7</td><td>14.7</td><td>7.8</td><td>7.1</td><td>5.1</td></tr><tr><td>科学事实数据集（SciFact）</td><td>70.1</td><td>70.6</td><td>70.4</td><td>69.2</td><td>68.0</td><td>69.6</td><td>66.7</td><td>65.6</td><td>64.9</td></tr><tr><td rowspan="5">问答（Q.A.）</td><td>自然问题数据集（NQ）</td><td>42.3</td><td>49.4</td><td>52.4</td><td>51.2</td><td>49.1</td><td>42.3</td><td>37.3</td><td>30.9</td><td>25.3</td></tr><tr><td>火锅问答数据集（HotpotQA）</td><td>57.6</td><td>61.0</td><td>60.4</td><td>57.4</td><td>53.0</td><td>57.6</td><td>55.3</td><td>49.4</td><td>43.4</td></tr><tr><td>金融问答数据集（FiQA）</td><td>29.8</td><td>32.2</td><td>32.0</td><td>30.1</td><td>27.8</td><td>29.9</td><td>22.2</td><td>17.7</td><td>14.0</td></tr><tr><td>生物医学问答数据集（BioASQ）</td><td>48.3</td><td>50.9</td><td>49.0</td><td>48.0</td><td>45.9</td><td>41.1</td><td>26.4</td><td>21.1</td><td>17.4</td></tr><tr><td>新闻事实语料库（NFCorpus）</td><td>31.8</td><td>33.5</td><td>33.9</td><td>33.6</td><td>33.0</td><td>29.9</td><td>26.3</td><td>25.7</td><td>25.2</td></tr><tr><td rowspan="4">其他</td><td>TREC新冠数据集（TREC - COVID）</td><td>54.8</td><td>63.5</td><td>72.9</td><td>75.4</td><td>74.5</td><td>53.1</td><td>48.7</td><td>42.5</td><td>38.1</td></tr><tr><td>科学文献数据集（SCIDOCS）</td><td>13.2</td><td>14.1</td><td>14.7</td><td>14.6</td><td>14.0</td><td>13.2</td><td>12.2</td><td>11.7</td><td>11.5</td></tr><tr><td>DBPedia（DBPedia）</td><td>31.5</td><td>39.1</td><td>42.1</td><td>39.9</td><td>36.9</td><td>31.4</td><td>31.4</td><td>29.7</td><td>25.1</td></tr><tr><td>Quora（Quora）</td><td>82.7</td><td>80.5</td><td>72.1</td><td>57.4</td><td>41.9</td><td>82.7</td><td>82.7</td><td>82.7</td><td>82.7</td></tr><tr><td></td><td>平均值</td><td>42.0</td><td>45.4</td><td>46.9</td><td>45.5</td><td>43.1</td><td>41.1</td><td>36.3</td><td>33.4</td><td>30.5</td></tr></tbody></table>

Table 11: nDCG@10 on the BEIR benchmark with different $k$ and $p$ in ALIGNERbase. *: alignment strategy during training $\left( {k = 4}\right)$ .

表11：在BEIR基准测试中，ALIGNERbase使用不同的$k$和$p$时的nDCG@10值。*：训练期间的对齐策略$\left( {k = 4}\right)$。

<table><tr><td colspan="2" rowspan="2">${\mathrm{{ALIGNER}}}_{\text{base }}$</td><td colspan="5">Top- $k$</td><td colspan="4">Top- $p$</td></tr><tr><td>1</td><td>2</td><td>4</td><td>6</td><td>8</td><td>0.5%</td><td>1%*</td><td>1.5%</td><td>2%</td></tr><tr><td rowspan="2">A.R.</td><td>ArguAna</td><td>28.4</td><td>23.7</td><td>17.9</td><td>14.2</td><td>11.1</td><td>32.9</td><td>45.3</td><td>48.0</td><td>47.7</td></tr><tr><td>Touché-2020</td><td>38.1</td><td>51.3</td><td>53.0</td><td>51.3</td><td>50.0</td><td>35.0</td><td>26.6</td><td>22.6</td><td>17.1</td></tr><tr><td rowspan="3">F.C.</td><td>FEVER</td><td>72.1</td><td>74.6</td><td>67.7</td><td>57.5</td><td>49.2</td><td>72.2</td><td>56.1</td><td>46.2</td><td>31.2</td></tr><tr><td>Climate-FEVER</td><td>17.8</td><td>20.3</td><td>22.0</td><td>22.1</td><td>21.8</td><td>17.8</td><td>13.5</td><td>14.4</td><td>11.3</td></tr><tr><td>SciFact</td><td>69.0</td><td>67.9</td><td>64.8</td><td>61.5</td><td>57.2</td><td>70.5</td><td>70.3</td><td>68.1</td><td>63.9</td></tr><tr><td rowspan="5">Q.A.</td><td>NO</td><td>52.6</td><td>48.2</td><td>36.4</td><td>26.4</td><td>20.2</td><td>52.5</td><td>50.4</td><td>44.7</td><td>38.2</td></tr><tr><td>HotpotQA</td><td>60.6</td><td>57.7</td><td>35.1</td><td>21.1</td><td>13.1</td><td>60.7</td><td>59.6</td><td>53.9</td><td>47.6</td></tr><tr><td>FiQA</td><td>33.7</td><td>29.4</td><td>23.5</td><td>18.8</td><td>15.8</td><td>33.6</td><td>29.8</td><td>26.2</td><td>20.6</td></tr><tr><td>BioASQ</td><td>40.3</td><td>48.4</td><td>39.9</td><td>33.1</td><td>26.4</td><td>50.0</td><td>40.3</td><td>35.3</td><td>28.8</td></tr><tr><td>NFCorpus</td><td>34.7</td><td>33.7</td><td>32.1</td><td>30.2</td><td>27.8</td><td>34.3</td><td>32.7</td><td>31.6</td><td>29.6</td></tr><tr><td rowspan="4">MISC.</td><td>TREC-COVID</td><td>67.6</td><td>73.4</td><td>73.1</td><td>67.7</td><td>62.5</td><td>67.2</td><td>65.7</td><td>60.6</td><td>52.0</td></tr><tr><td>SCIDOCS</td><td>13.9</td><td>14.2</td><td>13.1</td><td>11.5</td><td>9.9</td><td>14.1</td><td>14.6</td><td>14.8</td><td>14.9</td></tr><tr><td>DBPedia</td><td>41.3</td><td>40.3</td><td>29.5</td><td>21.2</td><td>14.0</td><td>41.2</td><td>41.3</td><td>40.3</td><td>37.1</td></tr><tr><td>Quora</td><td>83.6</td><td>60.8</td><td>22.8</td><td>8.9</td><td>4.3</td><td>83.6</td><td>83.6</td><td>83.6</td><td>83.6</td></tr><tr><td/><td>Average</td><td>43.6</td><td>42.9</td><td>35.4</td><td>29.7</td><td>25.5</td><td>44.4</td><td>42.0</td><td>39.3</td><td>34.9</td></tr></table>

<table><tbody><tr><td colspan="2" rowspan="2">${\mathrm{{ALIGNER}}}_{\text{base }}$</td><td colspan="5">顶部 - $k$</td><td colspan="4">顶部 - $p$</td></tr><tr><td>1</td><td>2</td><td>4</td><td>6</td><td>8</td><td>0.5%</td><td>1%*</td><td>1.5%</td><td>2%</td></tr><tr><td rowspan="2">评估报告（A.R.）</td><td>论证分析（ArguAna）</td><td>28.4</td><td>23.7</td><td>17.9</td><td>14.2</td><td>11.1</td><td>32.9</td><td>45.3</td><td>48.0</td><td>47.7</td></tr><tr><td>妙语连珠2020（Touché - 2020）</td><td>38.1</td><td>51.3</td><td>53.0</td><td>51.3</td><td>50.0</td><td>35.0</td><td>26.6</td><td>22.6</td><td>17.1</td></tr><tr><td rowspan="3">足球俱乐部（F.C.）</td><td>发烧</td><td>72.1</td><td>74.6</td><td>67.7</td><td>57.5</td><td>49.2</td><td>72.2</td><td>56.1</td><td>46.2</td><td>31.2</td></tr><tr><td>气候事实核查（Climate-FEVER）</td><td>17.8</td><td>20.3</td><td>22.0</td><td>22.1</td><td>21.8</td><td>17.8</td><td>13.5</td><td>14.4</td><td>11.3</td></tr><tr><td>科学事实核查（SciFact）</td><td>69.0</td><td>67.9</td><td>64.8</td><td>61.5</td><td>57.2</td><td>70.5</td><td>70.3</td><td>68.1</td><td>63.9</td></tr><tr><td rowspan="5">问答</td><td>否</td><td>52.6</td><td>48.2</td><td>36.4</td><td>26.4</td><td>20.2</td><td>52.5</td><td>50.4</td><td>44.7</td><td>38.2</td></tr><tr><td>火锅问答（HotpotQA）</td><td>60.6</td><td>57.7</td><td>35.1</td><td>21.1</td><td>13.1</td><td>60.7</td><td>59.6</td><td>53.9</td><td>47.6</td></tr><tr><td>金融问答数据集（FiQA）</td><td>33.7</td><td>29.4</td><td>23.5</td><td>18.8</td><td>15.8</td><td>33.6</td><td>29.8</td><td>26.2</td><td>20.6</td></tr><tr><td>生物医学问答数据集（BioASQ）</td><td>40.3</td><td>48.4</td><td>39.9</td><td>33.1</td><td>26.4</td><td>50.0</td><td>40.3</td><td>35.3</td><td>28.8</td></tr><tr><td>新闻金融语料库（NFCorpus）</td><td>34.7</td><td>33.7</td><td>32.1</td><td>30.2</td><td>27.8</td><td>34.3</td><td>32.7</td><td>31.6</td><td>29.6</td></tr><tr><td rowspan="4">其他</td><td>TREC新冠数据集（TREC - COVID）</td><td>67.6</td><td>73.4</td><td>73.1</td><td>67.7</td><td>62.5</td><td>67.2</td><td>65.7</td><td>60.6</td><td>52.0</td></tr><tr><td>科学文献数据集（SCIDOCS）</td><td>13.9</td><td>14.2</td><td>13.1</td><td>11.5</td><td>9.9</td><td>14.1</td><td>14.6</td><td>14.8</td><td>14.9</td></tr><tr><td>DBPedia（DBPedia）</td><td>41.3</td><td>40.3</td><td>29.5</td><td>21.2</td><td>14.0</td><td>41.2</td><td>41.3</td><td>40.3</td><td>37.1</td></tr><tr><td>Quora（Quora）</td><td>83.6</td><td>60.8</td><td>22.8</td><td>8.9</td><td>4.3</td><td>83.6</td><td>83.6</td><td>83.6</td><td>83.6</td></tr><tr><td></td><td>平均值</td><td>43.6</td><td>42.9</td><td>35.4</td><td>29.7</td><td>25.5</td><td>44.4</td><td>42.0</td><td>39.3</td><td>34.9</td></tr></tbody></table>

Table 12: nDCG@10 on the BEIR benchmark with different $k$ and $p$ in ALIGNERbase. *: alignment strategy during training $\left( {p = 1\% }\right)$ .

表12：在BEIR基准测试中，ALIGNERbase使用不同的$k$和$p$时的nDCG@10值。*：训练期间的对齐策略$\left( {p = 1\% }\right)$。

<table><tr><td rowspan="2" colspan="2">${\mathrm{{ALIGNER}}}_{\text{base }}$</td><td colspan="5">$k$</td><td colspan="5">$\varepsilon$</td></tr><tr><td>1*</td><td>2</td><td>4</td><td>6</td><td>8</td><td>0.01</td><td>0.02</td><td>0.04</td><td>0.06</td><td>0.1</td></tr><tr><td rowspan="2">A.R.</td><td>ArguAna</td><td>30.8</td><td>26.3</td><td>20.1</td><td>15.8</td><td>12.4</td><td>30.8</td><td>33.2</td><td>38.8</td><td>43.5</td><td>50.1</td></tr><tr><td>Touché-2020</td><td>37.4</td><td>50.1</td><td>52.0</td><td>51.1</td><td>49.0</td><td>37.4</td><td>35.0</td><td>29.8</td><td>26.9</td><td>19.4</td></tr><tr><td rowspan="3">F.C.</td><td>FEVER</td><td>68.8</td><td>71.4</td><td>63.7</td><td>46.1</td><td>44.3</td><td>68.8</td><td>68.0</td><td>66.4</td><td>64.5</td><td>56.2</td></tr><tr><td>Climate-FEVER</td><td>16.7</td><td>19.4</td><td>21.2</td><td>21.6</td><td>21.3</td><td>16.7</td><td>16.2</td><td>15.6</td><td>15.6</td><td>15.4</td></tr><tr><td>SciFact</td><td>69.8</td><td>68.2</td><td>64.7</td><td>61.8</td><td>57.0</td><td>69.8</td><td>69.5</td><td>70.3</td><td>71.0</td><td>70.1</td></tr><tr><td rowspan="5">Q.A.</td><td>NQ</td><td>52.0</td><td>47.4</td><td>35.0</td><td>25.8</td><td>18.5</td><td>52.0</td><td>51.6</td><td>50.8</td><td>50.0</td><td>46.0</td></tr><tr><td>HotpotQA</td><td>59.6</td><td>56.1</td><td>33.4</td><td>20.6</td><td>12.6</td><td>59.6</td><td>59.5</td><td>60.0</td><td>61.0</td><td>59.8</td></tr><tr><td>FiQA</td><td>32.9</td><td>29.6</td><td>23.0</td><td>18.6</td><td>14.5</td><td>32.9</td><td>32.7</td><td>32.9</td><td>32.4</td><td>30.9</td></tr><tr><td>BioASQ</td><td>50.4</td><td>47.2</td><td>38.0</td><td>31.4</td><td>24.5</td><td>50.4</td><td>50.4</td><td>50.8</td><td>50.4</td><td>43.7</td></tr><tr><td>NFCorpus</td><td>34.1</td><td>33.6</td><td>31.6</td><td>29.6</td><td>27.4</td><td>34.1</td><td>34.4</td><td>34.2</td><td>34.2</td><td>33.5</td></tr><tr><td rowspan="4">MISC.</td><td>TREC-COVID</td><td>63.9</td><td>73.2</td><td>72.5</td><td>67.7</td><td>62.8</td><td>63.9</td><td>65.5</td><td>63.6</td><td>63.8</td><td>54.2</td></tr><tr><td>SCIDOCS</td><td>14.0</td><td>14.4</td><td>13.4</td><td>11.6</td><td>9.6</td><td>14.0</td><td>14.1</td><td>14.2</td><td>14.5</td><td>15.4</td></tr><tr><td>DBPedia</td><td>40.9</td><td>38.7</td><td>28.7</td><td>19.9</td><td>13.2</td><td>40.9</td><td>40.5</td><td>39.9</td><td>39.7</td><td>38.6</td></tr><tr><td>Quora</td><td>82.8</td><td>61.2</td><td>21.5</td><td>7.4</td><td>3.5</td><td>82.8</td><td>83.1</td><td>83.5</td><td>84.0</td><td>85.0</td></tr><tr><td/><td>Average</td><td>43.6</td><td>42.4</td><td>34.6</td><td>28.6</td><td>24.7</td><td>43.6</td><td>43.6</td><td>43.4</td><td>43.4</td><td>41.2</td></tr></table>

<table><tbody><tr><td rowspan="2" colspan="2">${\mathrm{{ALIGNER}}}_{\text{base }}$</td><td colspan="5">$k$</td><td colspan="5">$\varepsilon$</td></tr><tr><td>1*</td><td>2</td><td>4</td><td>6</td><td>8</td><td>0.01</td><td>0.02</td><td>0.04</td><td>0.06</td><td>0.1</td></tr><tr><td rowspan="2">自动推理（Automated Reasoning）</td><td>阿古阿纳</td><td>30.8</td><td>26.3</td><td>20.1</td><td>15.8</td><td>12.4</td><td>30.8</td><td>33.2</td><td>38.8</td><td>43.5</td><td>50.1</td></tr><tr><td>妙触 - 2020</td><td>37.4</td><td>50.1</td><td>52.0</td><td>51.1</td><td>49.0</td><td>37.4</td><td>35.0</td><td>29.8</td><td>26.9</td><td>19.4</td></tr><tr><td rowspan="3">足球俱乐部（Football Club）</td><td>事实核查证据验证（Fact Extraction and VERification）</td><td>68.8</td><td>71.4</td><td>63.7</td><td>46.1</td><td>44.3</td><td>68.8</td><td>68.0</td><td>66.4</td><td>64.5</td><td>56.2</td></tr><tr><td>气候事实核查证据验证（Climate - Fact Extraction and VERification）</td><td>16.7</td><td>19.4</td><td>21.2</td><td>21.6</td><td>21.3</td><td>16.7</td><td>16.2</td><td>15.6</td><td>15.6</td><td>15.4</td></tr><tr><td>科学事实（SciFact）</td><td>69.8</td><td>68.2</td><td>64.7</td><td>61.8</td><td>57.0</td><td>69.8</td><td>69.5</td><td>70.3</td><td>71.0</td><td>70.1</td></tr><tr><td rowspan="5">问答（Q.A.）</td><td>自然问题（NQ）</td><td>52.0</td><td>47.4</td><td>35.0</td><td>25.8</td><td>18.5</td><td>52.0</td><td>51.6</td><td>50.8</td><td>50.0</td><td>46.0</td></tr><tr><td>火锅问答（HotpotQA）</td><td>59.6</td><td>56.1</td><td>33.4</td><td>20.6</td><td>12.6</td><td>59.6</td><td>59.5</td><td>60.0</td><td>61.0</td><td>59.8</td></tr><tr><td>金融问答（FiQA）</td><td>32.9</td><td>29.6</td><td>23.0</td><td>18.6</td><td>14.5</td><td>32.9</td><td>32.7</td><td>32.9</td><td>32.4</td><td>30.9</td></tr><tr><td>生物问答（BioASQ）</td><td>50.4</td><td>47.2</td><td>38.0</td><td>31.4</td><td>24.5</td><td>50.4</td><td>50.4</td><td>50.8</td><td>50.4</td><td>43.7</td></tr><tr><td>近场通信语料库（NFCorpus）</td><td>34.1</td><td>33.6</td><td>31.6</td><td>29.6</td><td>27.4</td><td>34.1</td><td>34.4</td><td>34.2</td><td>34.2</td><td>33.5</td></tr><tr><td rowspan="4">杂项（MISC.）</td><td>文本检索会议新冠专题（TREC-COVID）</td><td>63.9</td><td>73.2</td><td>72.5</td><td>67.7</td><td>62.8</td><td>63.9</td><td>65.5</td><td>63.6</td><td>63.8</td><td>54.2</td></tr><tr><td>科学文献数据集（SCIDOCS）</td><td>14.0</td><td>14.4</td><td>13.4</td><td>11.6</td><td>9.6</td><td>14.0</td><td>14.1</td><td>14.2</td><td>14.5</td><td>15.4</td></tr><tr><td>DBPedia（可音译为“迪比佩迪亚”，它是一个从维基百科中提取结构化信息的项目）</td><td>40.9</td><td>38.7</td><td>28.7</td><td>19.9</td><td>13.2</td><td>40.9</td><td>40.5</td><td>39.9</td><td>39.7</td><td>38.6</td></tr><tr><td>Quora问答社区（Quora）</td><td>82.8</td><td>61.2</td><td>21.5</td><td>7.4</td><td>3.5</td><td>82.8</td><td>83.1</td><td>83.5</td><td>84.0</td><td>85.0</td></tr><tr><td></td><td>平均</td><td>43.6</td><td>42.4</td><td>34.6</td><td>28.6</td><td>24.7</td><td>43.6</td><td>43.6</td><td>43.4</td><td>43.4</td><td>41.2</td></tr></tbody></table>

Table 13: nDCG@10 on the BEIR benchmark with ALIGNE ${\mathrm{R}}_{\text{base }}$ and differentiable alignment (Appendix A.2). ${}^{ * }$ : alignment strategy during training $\left( {k = 1}\right)$ .

表13：在BEIR基准测试中使用ALIGNE ${\mathrm{R}}_{\text{base }}$ 和可微对齐（附录A.2）的nDCG@10。${}^{ * }$：训练期间的对齐策略 $\left( {k = 1}\right)$。

<!-- Media -->