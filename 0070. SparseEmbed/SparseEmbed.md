# SparseEmbed: Learning Sparse Lexical Representations with Contextual Embeddings for Retrieval

# SparseEmbed：利用上下文嵌入学习用于检索的稀疏词汇表征

Weize Kong

孔伟泽

weize@google.com

Google Research

谷歌研究院

Jeffrey M. Dudek

杰弗里·M·杜德克

jdudek@google.com

Google Research

谷歌研究院

Cheng Li

李程

chgli@google.com

Google Research

谷歌研究院

Mingyang Zhang

张明阳

mingyang@google.com

Google Research

谷歌研究院

Michael Bendersky

迈克尔·本德尔斯基

bemike@google.com

Google Research

谷歌研究院

## ABSTRACT

## 摘要

In dense retrieval, prior work has largely improved retrieval effectiveness using multi-vector dense representations, exemplified by ColBERT. In sparse retrieval, more recent work, such as SPLADE, demonstrated that one can also learn sparse lexical representations to achieve comparable effectiveness while enjoying better interpretability. In this work, we combine the strengths of both the sparse and dense representations for first-stage retrieval. Specifically, we propose SparseEmbed - a novel retrieval model that learns sparse lexical representations with contextual embeddings. Compared with SPLADE, our model leverages the contextual embed-dings to improve model expressiveness. Compared with ColBERT, our sparse representations are trained end-to-end to optimize both efficiency and effectiveness.

在密集检索中，以往的工作通过多向量密集表征（以ColBERT为代表）在很大程度上提高了检索效果。在稀疏检索中，最近的工作（如SPLADE）表明，人们也可以学习稀疏词汇表征，在获得更好可解释性的同时达到相当的检索效果。在这项工作中，我们结合了稀疏表征和密集表征的优势用于第一阶段检索。具体来说，我们提出了SparseEmbed——一种利用上下文嵌入学习稀疏词汇表征的新型检索模型。与SPLADE相比，我们的模型利用上下文嵌入来提高模型的表达能力。与ColBERT相比，我们的稀疏表征是端到端训练的，以同时优化效率和效果。

## CCS CONCEPTS

## 计算机协会分类系统概念

- Information systems $\rightarrow$ Document representation; Query representation; Retrieval models and ranking.

- 信息系统 $\rightarrow$ 文档表征；查询表征；检索模型与排序。

## KEYWORDS

## 关键词

Sparse Retrieval; Dense Retrieval; Contextual Embeddings

稀疏检索；密集检索；上下文嵌入

## ACM Reference Format:

## ACM引用格式：

Weize Kong, Jeffrey M. Dudek, Cheng Li, Mingyang Zhang, and Michael Bendersky. 2023. SparseEmbed: Learning Sparse Lexical Representations with Contextual Embeddings for Retrieval. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '23), July 23-27, 2023, Taipei, Taiwan. ACM, New York, NY, USA, 5 pages. https://doi.org/10.1145/3539618.3592065

孔维泽、杰弗里·M·杜德克、李成、张明阳和迈克尔·本德尔斯基。2023年。SparseEmbed：利用上下文嵌入学习用于检索的稀疏词汇表示。收录于第46届ACM信息检索研究与发展国际会议（SIGIR '23）论文集，2023年7月23 - 27日，中国台湾台北。美国纽约州纽约市ACM协会，5页。https://doi.org/10.1145/3539618.3592065

## 1 INTRODUCTION

## 1 引言

Retrieval, in contrast to ranking, concerns retrieving a relatively small set of documents from a large corpus. Dense retrieval [25] represents queries and documents using dense vectors (also called embeddings) and retrieval is usually accomplished via approximate nearest neighbor search (ANNS) [1]. Sparse retrieval [2, 6, 26] instead uses sparse representations, mostly based on lexical features (e.g., BM25), and can be served via inverted index.

与排序不同，检索关注的是从大型语料库中检索出相对少量的文档。密集检索[25]使用密集向量（也称为嵌入）来表示查询和文档，检索通常通过近似最近邻搜索（ANNS）[1]来完成。而稀疏检索[2、6、26]则使用稀疏表示，主要基于词汇特征（例如BM25），并且可以通过倒排索引来实现。

<!-- Media -->

<!-- figureText: Contextual Scoring embeddings -->

<img src="https://cdn.noedgeai.com/0195a56c-b637-7da9-ac00-223adc4b6381_0.jpg?x=942&y=646&w=683&h=400&r=0"/>

Figure 1: SparseEmbed architecture overview.

图1：SparseEmbed架构概述。

<!-- Media -->

There are exciting advancements from neural information retrieval for both dense and sparse retrieval. For dense retrieval, prior work $\left\lbrack  {{14},{17}}\right\rbrack$ found that single-vector representations could be inadequate to capture all the key information and proposed to use multi-vector representations to improve model expressiveness, as exemplified by ColBERT [13]. However, ColBERT is also expensive to be deployed for large scale retrieval systems due to its large ANNS index size and quadratic-time scoring method (Section 4). For sparse retrieval, SPLADE [5-7] and other recent work [2] demonstrated that one can also learn sparse lexical representations to achieve comparable performance while offer enhanced interpretability and easier deployment.

神经信息检索在密集检索和稀疏检索方面都取得了令人振奋的进展。对于密集检索，先前的工作$\left\lbrack  {{14},{17}}\right\rbrack$发现单向量表示可能不足以捕捉所有关键信息，并提出使用多向量表示来提高模型的表达能力，例如ColBERT[13]。然而，由于其ANNS索引规模大且采用二次时间评分方法（第4节），ColBERT在大规模检索系统中的部署成本也很高。对于稀疏检索，SPLADE[5 - 7]和其他近期工作[2]表明，也可以学习稀疏词汇表示来实现相当的性能，同时提供更强的可解释性和更易于部署。

In this work, we aim to combine the strengths of sparse and dense representations for retrieval, and propose SparseEmbed - a novel retrieval model that learns sparse lexical representations with contextual embeddings for retrieval. As illustrated in Figure 1, SparseEmbed first encodes a given query (or document) into a sparse vector over the lexical vocabulary feature space, following SPLADE. For example, query "big apple stands for" could be encoded as \{"big", "apple", "nyc", ...\} with weights for each word. Note that the encoder could generate expansion terms (e.g., "nyc") which do not appear in the original input.

在这项工作中，我们旨在结合稀疏表示和密集表示在检索方面的优势，并提出SparseEmbed——一种新颖的检索模型，它利用上下文嵌入学习用于检索的稀疏词汇表示。如图1所示，SparseEmbed首先按照SPLADE的方法，将给定的查询（或文档）编码为词汇特征空间上的稀疏向量。例如，查询“big apple stands for”可以编码为{"big", "apple", "nyc", ...}，每个单词都有相应的权重。请注意，编码器可能会生成原始输入中未出现的扩展词（例如“nyc”）。

To improve model expressiveness, we borrow the idea from ColBERT to use contextual embeddings. Specifically, SparseEmbed further generates a contextual embedding for each term activated in the sparse vector. The contextual embeddings are pooled from underlying transformer sequence encodings to capture contextual information for each term. For example, embeddings for "apple" can then capture semantic difference when the word appears in the context of "big apple" versus "apple stock".

为了提高模型的表达能力，我们借鉴了ColBERT的思想，使用上下文嵌入。具体来说，SparseEmbed会为稀疏向量中激活的每个词进一步生成一个上下文嵌入。上下文嵌入是从底层的Transformer序列编码中汇集而来的，用于捕捉每个词的上下文信息。例如，当“apple”这个词出现在“big apple”和“apple stock”的上下文中时，其嵌入可以捕捉到语义差异。

SparseEmbed offers several strengths, inherited from SPLADE, ColBERT. First, SparseEmbed improves model expressiveness with contextual embeddings, compared against SPLADE which is limited to lexical matching. Second, SparseEmbed provides efficiency and cost advantages compared with ColBERT. We apply sparsity loss (Section 2.4) on the sparse vector to control the number of contextual embeddings. This enables optimizing index and querying cost during training. In addition, SparseEmbed is more efficient than ColBERT in scoring contextual embeddings (Section 2.3), since it only needs to compare contextual embeddings for matching query and document terms (linear complexity) instead of all query-document term pairs in ColBERT's late interaction (quadratic complexity). Third, SparseEmbed can be served using inverted index via attaching the contextual embeddings in the posting lists (Section 2.5), as with COIL [9]. Different from COIL, SparseEmbed can encode input text with expansion terms addressing the classic lexical mismatch problem COIL faces.

SparseEmbed继承了SPLADE和ColBERT的一些优点。首先，与仅限于词汇匹配的SPLADE相比，SparseEmbed通过上下文嵌入提高了模型的表达能力。其次，与ColBERT相比，SparseEmbed具有效率和成本优势。我们对稀疏向量应用稀疏性损失（第2.4节）来控制上下文嵌入的数量。这使得在训练过程中可以优化索引和查询成本。此外，SparseEmbed在对上下文嵌入进行评分时比ColBERT更高效（第2.3节），因为它只需要比较匹配的查询和文档词的上下文嵌入（线性复杂度），而不是像ColBERT的后期交互那样比较所有查询 - 文档词对（二次复杂度）。第三，与COIL[9]一样，SparseEmbed可以通过在倒排列表中附加上下文嵌入，使用倒排索引来实现。与COIL不同的是，SparseEmbed可以对包含扩展词的输入文本进行编码，从而解决了COIL面临的经典词汇不匹配问题。

Our contributions can be summarized as follows:

我们的贡献总结如下：

- We propose SparseEmbed, a sparse-dense hybrid model that learns sparse lexical representations with dense contextual embedding for first stage retrieval.

- 我们提出了SparseEmbed，这是一种稀疏 - 密集混合模型，它利用密集上下文嵌入学习用于第一阶段检索的稀疏词汇表示。

- We design a lightweight contextual embedding layer (Section 2.2) and use a top-k layer (Section 2.1) for SparseEmbed. These enable combining sparse retrieval as in SPLADE and dense retrieval as in ColBERT effectively and practically.

- 我们为SparseEmbed设计了一个轻量级的上下文嵌入层（第2.2节），并使用了一个top - k层（第2.1节）。这些使得能够有效地、实际地结合SPLADE中的稀疏检索和ColBERT中的密集检索。

- We conduct experiments on public datasets which test the effectiveness of SparseEmbed and demonstrate its capability of effectiveness-efficiency trade-off.

- 我们在公共数据集上进行了实验，验证了SparseEmbed的有效性，并展示了其在有效性 - 效率权衡方面的能力。

## 2 MODEL

## 2 模型

As illustrated in Figure 1,SparseEmbed first encodes a query $Q =$ $\left( {{q}_{1},{q}_{2},\ldots ,{q}_{\left| Q\right| }}\right)$ into a sparse vector $w \in  {\mathbb{R}}^{\left| V\right| }$ over a vocabulary $V$ . $w$ can be viewed as a term weight vector with only a few terms activated, i.e., having non-zero weights. Second, the model computes a contextual embedding for each activated term. Lastly, we apply scoring on top of the sparse-dense hybrid representations from the query and document side. Note that we encode the document $D = \left( {{d}_{1},{d}_{2},\ldots ,{d}_{\left| D\right| }}\right)$ in the same way as the query. We describe each parts in detail below, using the query side as an example.

如图1所示，SparseEmbed首先将查询 $Q =$ $\left( {{q}_{1},{q}_{2},\ldots ,{q}_{\left| Q\right| }}\right)$ 编码为词汇表 $V$ 上的稀疏向量 $w \in  {\mathbb{R}}^{\left| V\right| }$。$w$ 可以看作是一个只有少数项被激活（即具有非零权重）的词项权重向量。其次，模型为每个被激活的词项计算上下文嵌入。最后，我们对查询端和文档端的稀疏 - 密集混合表示进行评分。请注意，我们以与查询相同的方式对文档 $D = \left( {{d}_{1},{d}_{2},\ldots ,{d}_{\left| D\right| }}\right)$ 进行编码。下面我们以查询端为例详细描述各个部分。

### 2.1 Sparse Vector

### 2.1 稀疏向量

We follow SPLADE $\left\lbrack  {5,6}\right\rbrack$ to compute the sparse vector $w \in  {\mathbb{R}}^{\left| V\right| }$ , as illustrated in Figure 2. We first feed the query $Q$ into a BERT encoder,producing the sequence encodings $S \in  {\mathbb{R}}^{\left| Q\right|  \times  H}$ ,where $H$ is the hidden size. Then we use BERT’s MLM head to compute the MLM logits, $M \in  {\mathbb{R}}^{\left| Q\right|  \times  \left| V\right| }$ . Lastly,we transform the logits and apply max-pooling to compute the sparse vector $w$ as follows,

如图2所示，我们遵循SPLADE $\left\lbrack  {5,6}\right\rbrack$ 来计算稀疏向量 $w \in  {\mathbb{R}}^{\left| V\right| }$。我们首先将查询 $Q$ 输入到BERT编码器中，生成序列编码 $S \in  {\mathbb{R}}^{\left| Q\right|  \times  H}$，其中 $H$ 是隐藏层大小。然后我们使用BERT的掩码语言模型（MLM）头来计算MLM对数几率 $M \in  {\mathbb{R}}^{\left| Q\right|  \times  \left| V\right| }$。最后，我们对对数几率进行转换并应用最大池化来计算稀疏向量 $w$，如下所示：

$$
{w}_{i} = \mathop{\max }\limits_{{j = 1..\left| Q\right| }}\log \left( {1 + \operatorname{ReLU}\left( {m}_{j,i}\right) }\right) , \tag{1}
$$

where ${w}_{i}$ is the i-th value in $w$ and ${m}_{j,i}$ is the logits in $M$ . See SPLADE [5] for more details.

其中 ${w}_{i}$ 是 $w$ 中的第i个值，${m}_{j,i}$ 是 $M$ 中的对数几率。更多细节请参考SPLADE [5]。

Different from SPLADE, to facilitate the computation of contextual embeddings in the next step,we apply a top-k layer which select $\mathrm{k}$ dimensions with the highest weights in $w$ ,and zero-mask the other dimensions. This process helps bound the number of contextual embeddings we need to process.

与SPLADE不同，为了便于下一步计算上下文嵌入，我们应用了一个top - k层，该层选择 $w$ 中权重最高的 $\mathrm{k}$ 个维度，并将其他维度置零。这个过程有助于限制我们需要处理的上下文嵌入的数量。

<!-- Media -->

<!-- figureText: MLM logits, $M$ MLM layer Transformer layers Q = ("big", "apple", "stands", "for") -->

<img src="https://cdn.noedgeai.com/0195a56c-b637-7da9-ac00-223adc4b6381_1.jpg?x=972&y=234&w=620&h=517&r=0"/>

Figure 2: Sparse vector computation.

图2：稀疏向量计算。

<!-- Media -->

### 2.2 Contextual Embedding

### 2.2 上下文嵌入

We compute a context embedding for each term activated in the sparse vector,i.e.,terms with ${w}_{i} > 0$ . Different from ColBERT [3], we can't directly use the sequence encodings from the BERT encoder as contextual embeddings. This is because some activated terms may not appear in the input, as a result there are no corresponding sequence encodings for them.

我们为稀疏向量中被激活的每个词项（即 ${w}_{i} > 0$ 的词项）计算上下文嵌入。与ColBERT [3] 不同，我们不能直接使用BERT编码器的序列编码作为上下文嵌入。这是因为一些被激活的词项可能不会出现在输入中，因此它们没有对应的序列编码。

To address this, we use an attention layer to pool from the sequence encodings for each activated term, e.g., term #2 and #6 as illustrated in Figure 3. Our attention layer is lightweight. Since the MLM logits already measure the association between tokens in the input sequence and terms in the vocabulary, we can directly use the logits to compute the attention scores over the input sequence. Specifically,we compute the contextual embedding for the $i$ -th term in the vocabulary, ${e}_{i} \in  {\mathbb{R}}^{H}$ ,as,

为了解决这个问题，我们使用一个注意力层从每个被激活的词项的序列编码中进行池化，例如，如图3所示的词项#2和#6。我们的注意力层是轻量级的。由于MLM对数几率已经衡量了输入序列中的标记与词汇表中的词项之间的关联，我们可以直接使用对数几率来计算输入序列上的注意力分数。具体来说，我们计算词汇表中第 $i$ 个词项的上下文嵌入 ${e}_{i} \in  {\mathbb{R}}^{H}$ 如下：

$$
{e}_{i} = \operatorname{softmax}\left( {m}_{i}^{T}\right) S, \tag{2}
$$

<!-- Media -->

<!-- figureText: Select MLM logits as Attention layer Sequence encodings, $S$ Transformer layers attention scores Sparse vector, $w$ -->

<img src="https://cdn.noedgeai.com/0195a56c-b637-7da9-ac00-223adc4b6381_1.jpg?x=924&y=1177&w=724&h=349&r=0"/>

Figure 3: Contextual embedding computation.

图3：上下文嵌入计算。

<!-- Media -->

where ${m}_{i} \in  {\mathbb{R}}^{\left| Q\right| }$ is the $i$ -th column in MLM logits $M$ and $S \in$ ${\mathbb{R}}^{\left| Q\right|  \times  H}$ is the sequence encodings. With this attention layer we can represent all the terms including expansion terms.

其中 ${m}_{i} \in  {\mathbb{R}}^{\left| Q\right| }$ 是MLM对数几率 $M$ 中的第 $i$ 列，$S \in$ ${\mathbb{R}}^{\left| Q\right|  \times  H}$ 是序列编码。通过这个注意力层，我们可以表示包括扩展词项在内的所有词项。

Lastly,we project the resulting embedding from hidden size $H$ to a smaller size using a linear layer and apply ReLU activation to ensure values are all non-negatives,i.e.,ReLU(linear_layer $\left( {e}_{i}\right)$ ). The dimension reduction helps reduce querying and index space cost (see Section 2.5). The non-negative values in embeddings ensure dot-products computed upon are also non-negative. This enables querying time optimization when aggregating scores, e.g., early stop on candidate documents with low accumulated scores. In the end,we have a set of contextual embeddings $\left\{  {{e}_{i} \mid  i \in  I}\right\}$ for the activated terms in the vocabulary,where $I$ are the indices (to the vocabulary) for the activated terms, $I = \left\{  {i \mid  {w}_{i} > 0}\right\}$ .

最后，我们使用线性层将得到的嵌入从隐藏大小$H$投影到更小的大小，并应用ReLU激活函数以确保值均为非负，即ReLU(线性层$\left( {e}_{i}\right)$)。降维有助于降低查询和索引空间成本（见2.5节）。嵌入中的非负值确保计算得到的点积也是非负的。这在聚合分数时能够实现查询时间优化，例如，对累积分数较低的候选文档提前终止处理。最后，我们得到了词汇表中激活词项的一组上下文嵌入$\left\{  {{e}_{i} \mid  i \in  I}\right\}$，其中$I$是激活词项（在词汇表中）的索引，$I = \left\{  {i \mid  {w}_{i} > 0}\right\}$。

### 2.3 Scoring

### 2.3 评分

After encoding a query and a document, we compute their relevance score using dot-products between query and document contextual embeddings of matching terms, or more formally,

对查询和文档进行编码后，我们使用匹配词项的查询和文档上下文嵌入之间的点积来计算它们的相关性得分，更正式地说，

$$
s\left( {Q,D}\right)  = \mathop{\sum }\limits_{{\left( {i,j}\right)  \in  {I}^{Q} \times  {I}^{D},i = j}}{\left( {e}_{i}^{Q}\right) }^{T}{e}_{j}^{D}, \tag{3}
$$

where superscript $Q$ and $D$ indicate query and document variables, e.g., ${e}_{i}^{Q}$ is a contextual embedding from the query side. ${I}^{Q}$ and ${I}^{D}$ are the indices to the vocabulary for the activated terms. $i = j$ means ${e}_{i}^{Q}$ and ${e}_{j}^{D}$ are for the same term. This scoring uses linear time complexity, $O\left( {\min \left( {{\begin{Vmatrix}{w}^{Q}\end{Vmatrix}}_{0},{\begin{Vmatrix}{w}^{D}\end{Vmatrix}}_{0}}\right) }\right)$ ; while late interaction scoring in ColBERT [3] uses quadratic time, $O\left( {\left| Q\right| \left| D\right| }\right)$ .

其中上标$Q$和$D$分别表示查询和文档变量，例如，${e}_{i}^{Q}$是查询端的上下文嵌入。${I}^{Q}$和${I}^{D}$是激活词项在词汇表中的索引。$i = j$表示${e}_{i}^{Q}$和${e}_{j}^{D}$对应同一个词项。这种评分方法的时间复杂度为线性，即$O\left( {\min \left( {{\begin{Vmatrix}{w}^{Q}\end{Vmatrix}}_{0},{\begin{Vmatrix}{w}^{D}\end{Vmatrix}}_{0}}\right) }\right)$；而ColBERT [3]中的后期交互评分的时间复杂度为二次，即$O\left( {\left| Q\right| \left| D\right| }\right)$。

### 2.4 Losses

### 2.4 损失函数

We train SparseEmbed with two types of losses - sparsity loss and ranking loss. Sparsity loss is important as it not only encourages sparsity in the sparse vector $w$ ,but also in turn controls the number of contextual embeddings. Using sparsity loss, we can reduce both scoring cost (or the pairs of contextual embeddings we need to compare) and index space cost (or the number of document contextual embeddings to store).

我们使用两种类型的损失函数来训练SparseEmbed——稀疏性损失和排序损失。稀疏性损失很重要，因为它不仅能促进稀疏向量$w$的稀疏性，还能反过来控制上下文嵌入的数量。通过使用稀疏性损失，我们可以降低评分成本（即需要比较的上下文嵌入对的数量）和索引空间成本（即需要存储的文档上下文嵌入的数量）。

We follow SPLADE [6] to use FLOPS loss [19]. It is a smooth relaxation of the average number of floating-point operations necessary to score a document based on its sparse vector, or equivalently the average number of contextual embedding pairs we need to compute dot-product for SparseEmbed scoring (See Section 2.3). L1 regularization loss is also applicable, but was found to produce less balanced sparse vectors than FLOPS loss [19].

我们遵循SPLADE [6]的方法使用FLOPS损失[19]。它是基于文档的稀疏向量对文档进行评分所需的平均浮点运算次数的平滑松弛，或者等效地说，是为SparseEmbed评分计算点积所需的上下文嵌入对的平均数量（见2.3节）。L1正则化损失也适用，但发现它产生的稀疏向量不如FLOPS损失[19]产生的稀疏向量平衡。

Ranking loss aims to improve ranking. In our experiment, we use a distillation dataset for training. Each example is a triplet, containing a query $Q$ ,a positive document ${D}^{ + }$ and a negative document ${D}^{ - }$ , with distillation scores from a teacher model. We use MarginMSE loss [10] on two heads - the score based on the contextual em-beddings ${e}_{i}$ in Equation 3 and the score based on sparse vectors defined as $s\left( {Q,D}\right)  = {\left( {w}^{Q}\right) }^{T}{w}^{D}$ . We denote them as ${\mathcal{L}}_{\text{MarginMSE }}^{e}$ and ${\mathcal{L}}_{\text{MarginMSE }}^{w}$ respectively. ${\mathcal{L}}_{\text{MarginMSE }}^{w}$ helps the model to learn to select terms for generating contextual embeddings.

排序损失旨在提高排序性能。在我们的实验中，我们使用蒸馏数据集进行训练。每个示例是一个三元组，包含一个查询$Q$、一个正文档${D}^{ + }$和一个负文档${D}^{ - }$，以及来自教师模型的蒸馏分数。我们在两个头部使用MarginMSE损失[10]——基于公式3中的上下文嵌入${e}_{i}$的分数和基于定义为$s\left( {Q,D}\right)  = {\left( {w}^{Q}\right) }^{T}{w}^{D}$的稀疏向量的分数。我们分别将它们表示为${\mathcal{L}}_{\text{MarginMSE }}^{e}$和${\mathcal{L}}_{\text{MarginMSE }}^{w}$。${\mathcal{L}}_{\text{MarginMSE }}^{w}$有助于模型学习选择用于生成上下文嵌入的词项。

Finally, we combine sparsity and ranking losses together,

最后，我们将稀疏性损失和排序损失结合起来，

$$
\mathcal{L} = {\mathcal{L}}_{\text{MarginMSE }}^{e} + {\lambda }^{w}{\mathcal{L}}_{\text{MarginMSE }}^{w} + {\lambda }^{Q}{\mathcal{L}}_{\text{FLOPS }}^{Q} + {\lambda }^{D}{\mathcal{L}}_{\text{FLOPS }}^{D}, \tag{4}
$$

where weight ${\lambda }^{Q},{\lambda }^{D}$ can be used to control model sparsity for efficiency-effectiveness trade-off,and weight ${\lambda }^{w}$ is fixed to 0.1 in our experiment.

其中权重${\lambda }^{Q},{\lambda }^{D}$可用于控制模型的稀疏性，以实现效率 - 效果的权衡，在我们的实验中，权重${\lambda }^{w}$固定为0.1。

### 2.5 Inverted Index

### 2.5 倒排索引

Similar to COIL [9], SparseEmbed can be served with inverted index. We index documents based on their activated terms in the same way as other lexical IR systems, except that their associated contextual embeddings are also attached to the posting lists. At querying time, for each activated term of the query, we retrieve documents from its posting list together with the document contextual embeddings. Instead of scoring based on term frequency and inverse document frequency, the relevance score is computed based the contextual embeddings as defined in Equation 3.

与COIL [9]类似，SparseEmbed可以与倒排索引一起使用。我们基于文档的激活词项对其进行索引，方式与其他词法信息检索（IR）系统相同，不同之处在于，其关联的上下文嵌入也会附加到倒排列表中。在查询时，对于查询的每个激活词项，我们从其倒排列表中检索文档以及文档的上下文嵌入。相关性得分不是基于词频和逆文档频率来计算，而是基于上下文嵌入来计算，如公式3所定义。

## 3 EXPERIMENTS

## 3 实验

### 3.1 Experimental Setup

### 3.1 实验设置

Data. We follow SPLADE [7] to use the MS MARCO passage dataset ${}^{1}$ for in-domain retrieval experiments and the BEIR [22] benchmark datasets for zero-shot experiments. The MS MARCO dataset contains ${8.8}\mathrm{M}$ passages, ${500}\mathrm{k}$ training queries and 6980 dev queries. We train models on the public msmarco-hard-negatives distillation dataset ${}^{2}$ . It contains 50 hard negatives mined from BM25 and 12 dense retrievers for each training queries with distillation scores from a cross-attention teacher model. From this dataset, we sample ${25.6}\mathrm{M}$ triplets $\left( {Q,{D}^{ + },{D}^{ - }}\right)$ for training.

数据。我们遵循SPLADE [7]的做法，使用MS MARCO段落数据集 ${}^{1}$进行领域内检索实验，使用BEIR [22]基准数据集进行零样本实验。MS MARCO数据集包含 ${8.8}\mathrm{M}$个段落、 ${500}\mathrm{k}$个训练查询和6980个开发查询。我们在公开的msmarco-hard-negatives蒸馏数据集 ${}^{2}$上训练模型。该数据集为每个训练查询包含从BM25和12个密集检索器中挖掘出的50个难负样本，以及来自一个交叉注意力教师模型的蒸馏分数。我们从该数据集中采样 ${25.6}\mathrm{M}$个三元组 $\left( {Q,{D}^{ + },{D}^{ - }}\right)$进行训练。

Implementation. Following SPLADE++ [7], we implement SparseEm-bed using bert-base-uncased BERT encoder initialized from the CoCondenser [8] pretrained checkpoint. Queries and documents share the same BERT encoder, but use distinct MLM heads and contextual embedding projection layers (Section 2.2). $k$ in the top- $k$ layer (Section 2.1) is set to 64,256 for queries and documents respectively. We train models with the MSEMargin loss and the FLOPS loss (Section 2.4). We quadratically increase the FLOPS loss weights (Equation 4) at each training step until ${50}\mathrm{k}$ steps,from which it remains constant,as in SPLADE[6]. We train for ${150}\mathrm{k}$ steps with batch size 128. We re-implement SPLADE++ [7], an improved hard-negative distillation version of the original model $\left\lbrack  {5,6}\right\rbrack$ ,using the same settings as SparseEmbed. For other baselines, including Col-BERTv2 [21], we cite metrics from prior work for comparison.

实现。遵循SPLADE++ [7]，我们使用从CoCondenser [8]预训练检查点初始化的bert-base-uncased BERT编码器来实现SparseEmbed。查询和文档共享同一个BERT编码器，但使用不同的掩码语言模型（MLM）头和上下文嵌入投影层（第2.2节）。顶层（第2.1节）中的 $k$分别为查询和文档设置为64,256。我们使用均方误差边界损失（MSEMargin loss）和浮点运算次数损失（FLOPS loss）来训练模型（第2.4节）。与SPLADE [6]一样，我们在每个训练步骤二次增加FLOPS损失权重（公式4），直到 ${50}\mathrm{k}$步，此后保持不变。我们以128的批量大小训练 ${150}\mathrm{k}$步。我们使用与SparseEmbed相同的设置重新实现了SPLADE++ [7]，它是原始模型 $\left\lbrack  {5,6}\right\rbrack$的改进版难负样本蒸馏模型。对于其他基线模型，包括Col-BERTv2 [21]，我们引用先前工作中的指标进行比较。

Metrics & Evaluation. We evaluate models on MS MARCO dev queries for in-domain evaluation and on BEIR datasets for zero-shot evaluation. In addition to ranking metrics, we report an efficiency measure called TERMS, which estimates the average number of matched terms in a random query and a random document based on their sparse vectors:

指标与评估。我们在MS MARCO开发查询上对模型进行领域内评估，在BEIR数据集上进行零样本评估。除了排序指标外，我们还报告了一个名为TERMS的效率指标，该指标基于随机查询和随机文档的稀疏向量估计它们之间匹配词项的平均数量：

$$
\text{ TERMS } = \mathop{\sum }\limits_{{Q \in  \mathcal{Q}}}\frac{1}{\left| \mathcal{Q}\right| }{\begin{Vmatrix}{w}^{Q}\end{Vmatrix}}_{0} \cdot  \mathop{\sum }\limits_{{D \in  \mathcal{D}}}\frac{1}{\left| \mathcal{D}\right| }{\begin{Vmatrix}{w}^{D}\end{Vmatrix}}_{0}, \tag{5}
$$

where $Q$ is the test query set and $\mathcal{D}$ is the document corpus, $w$ is the sparse vectors. TERMS is closely related to the FLOPS measure [6] that estimates the average number of floating-point operations for scoring one document. For SPLADE model, the prior work [6] estimated its FLOPS exactly the same as TERMS. For SparseEmbed, we can estimate FLOPS $=$ TERMS $\times  {H}^{\prime }$ ,where ${H}^{\prime }$ is the contextual embedding size after projection and accounts for the floating-point operations involved in an embedding dot-product (Section 2.3).

其中 $Q$是测试查询集， $\mathcal{D}$是文档语料库， $w$是稀疏向量。TERMS与FLOPS指标 [6]密切相关，FLOPS指标用于估计对一个文档进行评分所需的平均浮点运算次数。对于SPLADE模型，先前的工作 [6]估计其FLOPS与TERMS完全相同。对于SparseEmbed，我们可以估计FLOPS $=$ TERMS $\times  {H}^{\prime }$，其中 ${H}^{\prime }$是投影后上下文嵌入的大小，它考虑了嵌入点积中涉及的浮点运算（第2.3节）。

---

<!-- Footnote -->

${}^{1}$ https://github.com/microsoft/MSMARCO-Passage-Ranking

${}^{1}$ https://github.com/microsoft/MSMARCO-Passage-Ranking

${}^{2}$ https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives

${}^{2}$ https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives

<!-- Footnote -->

---

### 3.2 Results

### 3.2 结果

In-domain Evaluation. We report the in-domain evaluation results on MS MARCO in Table 1. TERMS and FLOPS are described in Section 3.1. ${\lambda }^{q},{\lambda }^{d}$ are FLOPS loss weights (Equation 4). SPLADE ${}^{o} +  +$ is our SPLADE++ re-implementation. Superscript $S$ and $L$ for SparseEm-bed denote different FLOPS loss weights used. Subscript 16, 32, 64 denote the projection dimension for contextual embeddings (Section 2.2). Upper section numbers are copied from the cited papers.

领域内评估。我们在表1中报告了在MS MARCO上的领域内评估结果。TERMS和FLOPS在第3.1节中进行了描述。 ${\lambda }^{q},{\lambda }^{d}$是FLOPS损失权重（公式4）。SPLADE ${}^{o} +  +$是我们重新实现的SPLADE++。SparseEmbed的上标 $S$和 $L$表示使用的不同FLOPS损失权重。下标16、32、64表示上下文嵌入的投影维度（第2.2节）。上方的章节编号摘自引用的论文。

<!-- Media -->

<table><tr><td>Model</td><td>MRR@10</td><td>R@1k</td><td>TERMS</td><td>FLOPS</td><td>${\lambda Q},{\lambda }^{D}$</td></tr><tr><td>BM25 [7]</td><td>18.4</td><td>85.4</td><td>-</td><td>-</td><td>-</td></tr><tr><td>COIL-full [9]</td><td>35.5</td><td>96.4</td><td>-</td><td>-</td><td>-</td></tr><tr><td>ColBERT [13]</td><td>36.8</td><td>96.9</td><td>-</td><td>-</td><td>-</td></tr><tr><td>ColBERTv2 [21]</td><td>39.7</td><td>98.3</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SPLADE [6]</td><td>32.2</td><td>95.5</td><td>-</td><td>-</td><td/></tr><tr><td>SPLADE++ [7]</td><td>38.0</td><td>98.2</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SPLADE ${}^{o} +  +$</td><td>37.8</td><td>98.2</td><td>1.22</td><td>1.22</td><td>$4{e}^{-1},5{e}^{-1}$</td></tr><tr><td>SparseEmbed ${}_{32}^{S}$</td><td>38.4</td><td>98.2</td><td>0.57</td><td>${0.57} \times  {32}$</td><td>$4{e}^{-2},5{e}^{-2}$</td></tr><tr><td>SparseEmbed ${}_{16}^{L}$</td><td>38.8</td><td>98.1</td><td>0.74</td><td>${0.74} \times  {16}$</td><td>$4{e}^{-3},5{e}^{-3}$</td></tr><tr><td>SparseEmbed ${}_{32}^{L}$</td><td>39.0</td><td>98.2</td><td>4.46</td><td>${4.46} \times  {32}$</td><td>$4{e}^{-3},5{e}^{-3}$</td></tr><tr><td>SparseEmbed ${}_{64}^{L}$</td><td>39.2</td><td>98.1</td><td>1.63</td><td>${1.63} \times  {64}$</td><td>$4{e}^{-3},5{e}^{-3}$</td></tr></table>

<table><tbody><tr><td>模型</td><td>前10名平均倒数排名（MRR@10）</td><td>前1000名召回率（R@1k）</td><td>术语</td><td>浮点运算次数（FLOPS）</td><td>${\lambda Q},{\lambda }^{D}$</td></tr><tr><td>BM25算法 [7]</td><td>18.4</td><td>85.4</td><td>-</td><td>-</td><td>-</td></tr><tr><td>COIL-full模型 [9]</td><td>35.5</td><td>96.4</td><td>-</td><td>-</td><td>-</td></tr><tr><td>ColBERT模型 [13]</td><td>36.8</td><td>96.9</td><td>-</td><td>-</td><td>-</td></tr><tr><td>ColBERTv2模型 [21]</td><td>39.7</td><td>98.3</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SPLADE模型 [6]</td><td>32.2</td><td>95.5</td><td>-</td><td>-</td><td></td></tr><tr><td>SPLADE++模型 [7]</td><td>38.0</td><td>98.2</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SPLADE模型 ${}^{o} +  +$</td><td>37.8</td><td>98.2</td><td>1.22</td><td>1.22</td><td>$4{e}^{-1},5{e}^{-1}$</td></tr><tr><td>稀疏嵌入模型（SparseEmbed） ${}_{32}^{S}$</td><td>38.4</td><td>98.2</td><td>0.57</td><td>${0.57} \times  {32}$</td><td>$4{e}^{-2},5{e}^{-2}$</td></tr><tr><td>稀疏嵌入模型（SparseEmbed） ${}_{16}^{L}$</td><td>38.8</td><td>98.1</td><td>0.74</td><td>${0.74} \times  {16}$</td><td>$4{e}^{-3},5{e}^{-3}$</td></tr><tr><td>稀疏嵌入模型（SparseEmbed） ${}_{32}^{L}$</td><td>39.0</td><td>98.2</td><td>4.46</td><td>${4.46} \times  {32}$</td><td>$4{e}^{-3},5{e}^{-3}$</td></tr><tr><td>稀疏嵌入模型（SparseEmbed） ${}_{64}^{L}$</td><td>39.2</td><td>98.1</td><td>1.63</td><td>${1.63} \times  {64}$</td><td>$4{e}^{-3},5{e}^{-3}$</td></tr></tbody></table>

Table 1: Evaluation results for MS MARCO passage retrieval.

表1：MS MARCO段落检索的评估结果。

<!-- Media -->

First, all runs of SparseEmbed outperform SPLADE models on MRR@10, demonstrating the effectiveness of SparseEmbed. For example,SparseEmbed ${}_{16}^{S}$ improves SPLADE ${}^{o} +  +$ by $+ {2.6}\%$ on MRR@10, while using less TERMS (0.74 v.s. 1.22). This indicates the performance gain is not due to SparseEmbed using more activated terms but due to its contextual embeddings. In fact, we find SparseEm-bed is always more effective at a similar TERMS measure, when comparing metrics reported in Figure 1 of the SPLADE++ paper [7]. That said, SPLADE++ is still more efficient according to FLOPS.

首先，SparseEmbed的所有运行结果在MRR@10指标上均优于SPLADE模型，证明了SparseEmbed的有效性。例如，SparseEmbed ${}_{16}^{S}$ 在MRR@10上比SPLADE ${}^{o} +  +$ 提高了 $+ {2.6}\%$，同时使用的词元（TERMS）更少（0.74对比1.22）。这表明性能提升并非因为SparseEmbed使用了更多激活的词元，而是由于其上下文嵌入。事实上，当比较SPLADE++论文 [7] 图1中报告的指标时，我们发现SparseEmbed在相似的词元度量下总是更有效。也就是说，根据浮点运算次数（FLOPS），SPLADE++仍然更高效。

Second,comparing SparseEmbed ${}_{32}^{S}$ and SparseEmbed ${}_{32}^{L}$ ,we show that one can adjust the FLOPS loss weights (Section 2.4) to make trade-off between efficiency and effectiveness. Comparing different SparseEmbed ${}_{ * }^{L}$ runs,we find larger contextual embedding projection size is more effective. Third, ColBERTv2 is still more effective than SparseEmbed. However, ColBERT is also more expensive to serve due to its quadratic time-complexity scoring (Section 2.3).

其次，比较SparseEmbed ${}_{32}^{S}$ 和SparseEmbed ${}_{32}^{L}$，我们表明可以调整浮点运算次数损失权重（第2.4节）来在效率和有效性之间进行权衡。比较不同的SparseEmbed ${}_{ * }^{L}$ 运行结果，我们发现更大的上下文嵌入投影尺寸更有效。第三，ColBERTv2仍然比SparseEmbed更有效。然而，由于ColBERT的评分具有二次时间复杂度（第2.3节），其服务成本也更高。

Zero-shot Evaluation. We report zero-shot evaluation results on the same 13 BEIR datasets as the prior work $\left\lbrack  {5,7}\right\rbrack$ in Table 2.

零样本评估。我们在表2中报告了与先前工作 $\left\lbrack  {5,7}\right\rbrack$ 相同的13个BEIR数据集上的零样本评估结果。

First, SparseEmbed demonstrates strong out-of-domain generalizability. It achieves the best average NDCG@10, slightly better than SPLADE++. Second, while ColBERTv2 perform strongly in the in-domain setting (Table 1), both SparseEmbed and SPLADE++ perform better in the zero-shot setting than ColBERT on average NDCG@10.This indicates sparse retrieval models may have some inductive bias for out-of-domain generalizability.

首先，SparseEmbed表现出强大的域外泛化能力。它实现了最佳的平均NDCG@10，略优于SPLADE++。其次，虽然ColBERTv2在域内设置中表现出色（表1），但在零样本设置中，SparseEmbed和SPLADE++在平均NDCG@10上均比ColBERT表现更好。这表明稀疏检索模型可能对域外泛化具有一定的归纳偏置。

## 4 RELATED WORK

## 4 相关工作

Our work is related to both dense retrieval and sparse retrieval. For dense retrieval,prior work $\left\lbrack  {{14},{17}}\right\rbrack$ found single-vector dense representations could be inadequate. ColBERT [13] proposed to use multi-vector representations by directly using contextual embeddings from a BERT encoder. This significantly improves model expressiveness,but is also expensive to scale. Follow-up work $\left\lbrack  {{12},{15},{21},{23}}\right\rbrack$ has explored pruning or compression methods to improve efficiency. We also use contextual embeddings, but we rely on the underlying sparse lexical representation learning to optimize contextual embedding selection for both efficiency and effectiveness. This is achieved by sparsity loss and ranking loss (Section 2.4). Our model is also more efficient in scoring, as it avoids the quadratic contextual embedding interaction in ColBERT (Section 2.3). Another orthogonal research direction, started from dense retrieval, is hard negative mining [20, 24] and distillation [10, 11, 16, 20].

我们的工作与密集检索和稀疏检索都相关。对于密集检索，先前的工作 $\left\lbrack  {{14},{17}}\right\rbrack$ 发现单向量密集表示可能不够。ColBERT [13] 提出直接使用BERT编码器的上下文嵌入来使用多向量表示。这显著提高了模型的表达能力，但扩展成本也很高。后续工作 $\left\lbrack  {{12},{15},{21},{23}}\right\rbrack$ 探索了剪枝或压缩方法以提高效率。我们也使用上下文嵌入，但我们依靠底层的稀疏词汇表示学习来优化上下文嵌入选择，以兼顾效率和有效性。这通过稀疏性损失和排序损失来实现（第2.4节）。我们的模型在评分方面也更高效，因为它避免了ColBERT中的二次上下文嵌入交互（第2.3节）。另一个从密集检索开始的正交研究方向是难负样本挖掘 [20, 24] 和蒸馏 [10, 11, 16, 20]。

<!-- Media -->

Weize Kong, Jeffrey M. Dudek, Cheng Li, Mingyang Zhang, & Michael Bendersky

孔伟泽，杰弗里·M·杜德克，李成，张明阳，迈克尔·本德尔斯基

<table><tr><td>Dataset</td><td>BM25</td><td>ColBERTv2[21]</td><td>SPLADE++[7]</td><td>SparseEmbed ${}_{64}^{L}$</td></tr><tr><td>ArguAna</td><td>31.5</td><td>46.3</td><td>52.5</td><td>51.2</td></tr><tr><td>Climate-FEVER</td><td>21.3</td><td>17.6</td><td>23.0</td><td>21.8</td></tr><tr><td>DBPedia</td><td>31.3</td><td>44.6</td><td>43.6</td><td>45.7</td></tr><tr><td>FEVER</td><td>75.3</td><td>78.5</td><td>79.3</td><td>79.6</td></tr><tr><td>FiQA-2018</td><td>23.6</td><td>35.6</td><td>34.8</td><td>33.5</td></tr><tr><td>HotpotQA</td><td>60.3</td><td>66.7</td><td>68.7</td><td>69.7</td></tr><tr><td>NFCorpus</td><td>32.5</td><td>33.8</td><td>34.8</td><td>34.1</td></tr><tr><td>NQ</td><td>32.9</td><td>56.2</td><td>53.7</td><td>54.4</td></tr><tr><td>Quora</td><td>78.9</td><td>85.2</td><td>83.4</td><td>84.9</td></tr><tr><td>SCIDOCS</td><td>15.8</td><td>15.4</td><td>15.9</td><td>16.0</td></tr><tr><td>SciFact</td><td>66.5</td><td>69.3</td><td>70.2</td><td>70.6</td></tr><tr><td>TREC-COVID</td><td>65.6</td><td>73.8</td><td>72.7</td><td>72.4</td></tr><tr><td>Touché-2020</td><td>36.7</td><td>26.3</td><td>24.5</td><td>27.3</td></tr><tr><td>Average</td><td>44.0</td><td>49.9</td><td>50.5</td><td>50.9</td></tr></table>

<table><tbody><tr><td>数据集</td><td>BM25算法</td><td>ColBERTv2模型[21]</td><td>SPLADE++模型[7]</td><td>稀疏嵌入 ${}_{64}^{L}$</td></tr><tr><td>ArguAna数据集</td><td>31.5</td><td>46.3</td><td>52.5</td><td>51.2</td></tr><tr><td>Climate-FEVER数据集</td><td>21.3</td><td>17.6</td><td>23.0</td><td>21.8</td></tr><tr><td>DBPedia数据集</td><td>31.3</td><td>44.6</td><td>43.6</td><td>45.7</td></tr><tr><td>FEVER数据集</td><td>75.3</td><td>78.5</td><td>79.3</td><td>79.6</td></tr><tr><td>FiQA-2018数据集</td><td>23.6</td><td>35.6</td><td>34.8</td><td>33.5</td></tr><tr><td>HotpotQA数据集</td><td>60.3</td><td>66.7</td><td>68.7</td><td>69.7</td></tr><tr><td>NFCorpus数据集</td><td>32.5</td><td>33.8</td><td>34.8</td><td>34.1</td></tr><tr><td>NQ数据集</td><td>32.9</td><td>56.2</td><td>53.7</td><td>54.4</td></tr><tr><td>Quora数据集</td><td>78.9</td><td>85.2</td><td>83.4</td><td>84.9</td></tr><tr><td>SCIDOCS数据集</td><td>15.8</td><td>15.4</td><td>15.9</td><td>16.0</td></tr><tr><td>SciFact数据集</td><td>66.5</td><td>69.3</td><td>70.2</td><td>70.6</td></tr><tr><td>TREC-COVID数据集</td><td>65.6</td><td>73.8</td><td>72.7</td><td>72.4</td></tr><tr><td>Touché-2020数据集</td><td>36.7</td><td>26.3</td><td>24.5</td><td>27.3</td></tr><tr><td>平均值</td><td>44.0</td><td>49.9</td><td>50.5</td><td>50.9</td></tr></tbody></table>

Table 2: NDCG@10 on BEIR datasets in zero-shot setting. Baseline metric values are copied from the cited papers.

表2：零样本设置下BEIR数据集的NDCG@10。基线指标值摘自引用的论文。

<!-- Media -->

Sparse retrieval such as BM25 has been studied for decades. More recent work $\left\lbrack  {2,4,6,{18},{26}}\right\rbrack$ started learning sparse lexical representation using neural networks. In this category, SPLADE [5- 7] have shown that one could learn a sparse lexical encoder to achieve comparable performance as other dense retrieval models, and offers enhanced interpretability. We build SparseEmbed on top of SPLADE and add contextual embeddings to improve model expressiveness.

像BM25这样的稀疏检索已经研究了数十年。最近的工作$\left\lbrack  {2,4,6,{18},{26}}\right\rbrack$开始使用神经网络学习稀疏词汇表示。在这一类别中，SPLADE [5 - 7]表明可以学习一个稀疏词汇编码器，以达到与其他密集检索模型相当的性能，并提供更强的可解释性。我们在SPLADE的基础上构建了SparseEmbed，并添加上下文嵌入以提高模型的表达能力。

To the best of our knowledge, there is limited work in sparse-dense hybrid representations for retrieval. COIL [9] uses the term-level contextual embeddings to perform lexical match. However, COIL does not learn the sparse representation as SparseEmbed (Section 2.1), it simply encodes the text input based on term occurrence, which can lead to lexical mismatch issues.

据我们所知，用于检索的稀疏 - 密集混合表示的相关工作有限。COIL [9]使用词级上下文嵌入来进行词汇匹配。然而，COIL不像SparseEmbed那样学习稀疏表示（第2.1节），它只是基于词的出现对文本输入进行编码，这可能会导致词汇不匹配的问题。

## 5 CONCLUSIONS

## 5 结论

We present SparseEmbed, a retrieval model that learns sparse lexical representations with contextual embeddings. The model combines the strengths of sparse retrieval like SPLADE and multi-vector dense retrieval like ColBERT. With contextual embeddings, SparseEmbed improves model expressiveness over SPLADE. With the sparse representations and sparsity loss, SparseEmbed provides efficiency advantages over ColBERT in both querying and index space cost. This sparse-dense hybrid representations can also be served via inverted index. Our experiments demonstrate both in-domain and out-of-domain effectiveness, as well as the effectiveness-efficiency trade-off SparseEmbed offers.

我们提出了SparseEmbed，这是一个通过上下文嵌入学习稀疏词汇表示的检索模型。该模型结合了像SPLADE这样的稀疏检索和像ColBERT这样的多向量密集检索的优势。通过上下文嵌入，SparseEmbed比SPLADE提高了模型的表达能力。借助稀疏表示和稀疏性损失，SparseEmbed在查询和索引空间成本方面比ColBERT具有效率优势。这种稀疏 - 密集混合表示也可以通过倒排索引来实现。我们的实验证明了SparseEmbed在领域内和领域外的有效性，以及它所提供的有效性 - 效率权衡。

## REFERENCES

## 参考文献

[1] Alexandr Andoni, Piotr Indyk, and Ilya Razenshteyn. Approximate nearest neighbor search in high dimensions. In Proceedings of the International Congress of Mathematicians: Rio de Janeiro 2018, pages 3287-3318. World Scientific, 2018.

[1] Alexandr Andoni、Piotr Indyk和Ilya Razenshteyn。高维近似最近邻搜索。见《国际数学家大会会议录：2018年里约热内卢》，第3287 - 3318页。世界科学出版社，2018年。

[2] Yang Bai, Xiaoguang Li, Gang Wang, Chaoliang Zhang, Lifeng Shang, Jun Xu, Zhaowei Wang, Fangshan Wang, and Qun Liu. Sparterm: Learning term-based sparse representation for fast text retrieval. arXiv preprint arXiv:2010.00768, 2020.

[2] 白杨、李晓光、王刚、张朝亮、尚立峰、徐军、王兆伟、王房山和刘群。Sparterm：为快速文本检索学习基于词的稀疏表示。预印本arXiv:2010.00768，2020年。

[3] Ronan Collobert and Jason Weston. A unified architecture for natural language processing: Deep neural networks with multitask learning. In Proc. of ICML, pages 160-167, 2008.

[3] Ronan Collobert和Jason Weston。自然语言处理的统一架构：具有多任务学习的深度神经网络。见《机器学习国际会议论文集》，第160 - 167页，2008年。

[4] Zhuyun Dai and Jamie Callan. Context-aware term weighting for first stage passage retrieval. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, pages 1533-1536, 2020.

[4] 戴竹云（Zhuyun Dai）和Jamie Callan。用于第一阶段段落检索的上下文感知词加权。见《第43届ACM信息检索研究与发展国际会议论文集》，第1533 - 1536页，2020年。

[5] Thibault Formal, Carlos Lassance, Benjamin Piwowarski, and Stéphane Clinchant. Splade v2: Sparse lexical and expansion model for information retrieval. arXiv preprint arXiv:2109.10086, 2021.

[5] Thibault Formal、Carlos Lassance、Benjamin Piwowarski和Stéphane Clinchant。Splade v2：用于信息检索的稀疏词汇和扩展模型。预印本arXiv:2109.10086，2021年。

[6] Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant. Splade: Sparse lexical and expansion model for first stage ranking. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 2288-2292, 2021.

[6] Thibault Formal、Benjamin Piwowarski和Stéphane Clinchant。Splade：用于第一阶段排序的稀疏词汇和扩展模型。见《第44届ACM信息检索研究与发展国际会议论文集》，第2288 - 2292页，2021年。

[7] Thibault Formal, Carlos Lassance, Benjamin Piwowarski, and Stéphane Clinchant. From distillation to hard negative sampling: Making sparse neural ir models more effective. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 2353-2359, 2022.

[7] Thibault Formal、Carlos Lassance、Benjamin Piwowarski和Stéphane Clinchant。从蒸馏到难负样本采样：使稀疏神经信息检索模型更有效。见《第45届ACM信息检索研究与发展国际会议论文集》，第2353 - 2359页，2022年。

[8] Luyu Gao and Jamie Callan. Unsupervised corpus aware language model pretraining for dense passage retrieval. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2843-2853, 2022.

[8] 高宇（Luyu Gao）和Jamie Callan。用于密集段落检索的无监督语料库感知语言模型预训练。见《计算语言学协会第60届年会论文集（第1卷：长论文）》，第2843 - 2853页，2022年。

[9] Luyu Gao, Zhuyun Dai, and Jamie Callan. Coil: Revisit exact lexical match in information retrieval with contextualized inverted list. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3030-3042, 2021.

[9] 高宇（Luyu Gao）、戴竹云（Zhuyun Dai）和Jamie Callan。COIL：通过上下文倒排列表重新审视信息检索中的精确词汇匹配。见《计算语言学协会北美分会2021年会议：人类语言技术》，第3030 - 3042页，2021年。

[10] Sebastian Hofstätter, Sophia Althammer, Michael Schröder, Mete Sertkan, and Allan Hanbury. Improving efficient neural ranking models with cross-architecture knowledge distillation. arXiv preprint arXiv:2010.02666, 2020.

[10] Sebastian Hofstätter、Sophia Althammer、Michael Schröder、Mete Sertkan和Allan Hanbury。通过跨架构知识蒸馏改进高效神经排序模型。预印本arXiv:2010.02666，2020年。

[11] Sebastian Hofstätter, Sheng-Chieh Lin, Jheng-Hong Yang, Jimmy Lin, and Allan Hanbury. Efficiently teaching an effective dense retriever with balanced topic aware sampling. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 113-122, 2021.

[11] 塞巴斯蒂安·霍夫施泰特（Sebastian Hofstätter）、林圣杰（Sheng-Chieh Lin）、杨政宏（Jheng-Hong Yang）、林吉米（Jimmy Lin）和艾伦·汉伯里（Allan Hanbury）。通过平衡主题感知采样高效训练有效的密集检索器。见《第44届国际ACM SIGIR信息检索研究与发展会议论文集》，第113 - 122页，2021年。

[12] Sebastian Hofstätter, Omar Khattab, Sophia Althammer, Mete Sertkan, and Allan Hanbury. Introducing neural bag of whole-words with colberter: Contextualized late interactions using enhanced reduction. In Proceedings of the 31st ACM International Conference on Information and Knowledge Management, page 737-747, 2022.

[12] 塞巴斯蒂安·霍夫施泰特（Sebastian Hofstätter）、奥马尔·卡塔布（Omar Khattab）、索菲亚·阿尔塔默（Sophia Althammer）、梅特·塞尔特坎（Mete Sertkan）和艾伦·汉伯里（Allan Hanbury）。使用ColBERTer引入全词神经词袋：利用增强约简的上下文后期交互。见《第31届ACM国际信息与知识管理会议论文集》，第737 - 747页，2022年。

[13] Omar Khattab and Matei Zaharia. Colbert: Efficient and effective passage search via contextualized late interaction over bert. In Proc. of SIGIR, pages 39-48, 2020.

[13] 奥马尔·卡塔布（Omar Khattab）和马特·扎哈里亚（Matei Zaharia）。ColBERT：通过基于BERT的上下文后期交互实现高效有效的段落搜索。见《SIGIR会议论文集》，第39 - 48页，2020年。

[14] Weize Kong, Swaraj Khadanga, Cheng Li, Shaleen Kumar Gupta, Mingyang Zhang, Wensong Xu, and Michael Bendersky. Multi-aspect dense retrieval. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 3178-3186, 2022.

[14] 孔伟泽（Weize Kong）、斯瓦拉吉·卡丹加（Swaraj Khadanga）、李成（Cheng Li）、沙林·库马尔·古普塔（Shaleen Kumar Gupta）、张明阳（Mingyang Zhang）、徐文松（Wensong Xu）和迈克尔·本德尔斯基（Michael Bendersky）。多方面密集检索。见《第28届ACM SIGKDD知识发现与数据挖掘会议论文集》，第3178 - 3186页，2022年。

[15] Carlos Lassance, Maroua Maachou, Joohee Park, and Stéphane Clinchant. A study on token pruning for colbert. arXiv preprint arXiv:2112.06540, 2021.

[15] 卡洛斯·拉萨斯（Carlos Lassance）、马鲁阿·马乔（Maroua Maachou）、朴珠熙（Joohee Park）和斯特凡·克林尚（Stéphane Clinchant）。关于ColBERT的词元剪枝研究。预印本arXiv:2112.06540，2021年。

[16] Sheng-Chieh Lin, Jheng-Hong Yang, and Jimmy Lin. In-batch negatives for knowledge distillation with tightly-coupled teachers for dense retrieval. In Proceedings of the 6th Workshop on Representation Learning for NLP (RepL4NLP- 2021), pages 163-173, 2021.

[16] 林圣杰（Sheng-Chieh Lin）、杨政宏（Jheng-Hong Yang）和林吉米（Jimmy Lin）。使用紧密耦合教师进行知识蒸馏的批次内负样本用于密集检索。见《第6届自然语言处理表示学习研讨会（RepL4NLP - 2021）论文集》，第163 - 173页，2021年。

[17] Yi Luan, Jacob Eisenstein, Kristina Toutanova, and Michael Collins. Sparse, dense, and attentional representations for text retrieval. arXiv preprint arXiv:2005.00181, 2020.

[17] 栾义（Yi Luan）、雅各布·艾森斯坦（Jacob Eisenstein）、克里斯蒂娜·图托纳娃（Kristina Toutanova）和迈克尔·柯林斯（Michael Collins）。用于文本检索的稀疏、密集和注意力表示。预印本arXiv:2005.00181，2020年。

[18] Antonio Mallia, Omar Khattab, Torsten Suel, and Nicola Tonellotto. Learning passage impacts for inverted indexes. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 1723-1727, 2021.

[18] 安东尼奥·马利亚（Antonio Mallia）、奥马尔·卡塔布（Omar Khattab）、托尔斯滕·苏尔（Torsten Suel）和尼古拉·托内洛托（Nicola Tonellotto）。学习倒排索引的段落影响。见《第44届国际ACM SIGIR信息检索研究与发展会议论文集》，第1723 - 1727页，2021年。

[19] Biswajit Paria, Chih-Kuan Yeh, Ian EH Yen, Ning Xu, Pradeep Ravikumar, and Barnabás Póczos. Minimizing flops to learn efficient sparse representations. In International Conference on Learning Representations, 2019.

[19] 比斯瓦吉特·帕里亚（Biswajit Paria）、叶志宽（Chih-Kuan Yeh）、伊恩·EH·严（Ian EH Yen）、徐宁（Ning Xu）、普拉迪普·拉维库马尔（Pradeep Ravikumar）和巴尔纳巴斯·波乔斯（Barnabás Póczos）。最小化浮点运算以学习高效的稀疏表示。见《国际学习表征会议》，2019年。

[20] Yingqi Qu, Yuchen Ding, Jing Liu, Kai Liu, Ruiyang Ren, Wayne Xin Zhao, Dax-iang Dong, Hua Wu, and Haifeng Wang. Rocketqa: An optimized training approach to dense passage retrieval for open-domain question answering. arXiv preprint arXiv:2010.08191, 2020.

[20] 曲英琦（Yingqi Qu）、丁宇辰（Yuchen Ding）、刘静（Jing Liu）、刘凯（Kai Liu）、任瑞阳（Ruiyang Ren）、赵鑫（Wayne Xin Zhao）、董大祥（Daxiang Dong）、吴华（Hua Wu）和王海峰（Haifeng Wang）。RocketQA：用于开放域问答的密集段落检索的优化训练方法。预印本arXiv:2010.08191，2020年。

[21] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. Colbertv2: Effective and efficient retrieval via lightweight late interaction. arXiv preprint arXiv:2112.01488, 2021.

[21] 凯沙夫·桑塔南姆（Keshav Santhanam）、奥马尔·卡塔布（Omar Khattab）、乔恩·萨德 - 法尔孔（Jon Saad-Falcon）、克里斯托弗·波茨（Christopher Potts）和马特·扎哈里亚（Matei Zaharia）。ColBERTv2：通过轻量级后期交互实现高效有效的检索。预印本arXiv:2112.01488，2021年。

[22] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021.

[22] 南丹·塔库尔（Nandan Thakur）、尼尔斯·赖默斯（Nils Reimers）、安德烈亚斯·吕克莱（Andreas Rücklé）、阿比舍克·斯里瓦斯塔瓦（Abhishek Srivastava）和伊琳娜·古列维奇（Iryna Gurevych）。BEIR：信息检索模型零样本评估的异构基准。见《第三十五届神经信息处理系统数据集与基准赛道会议（第二轮）》，2021年。

[23] Nicola Tonellotto and Craig Macdonald. Query embedding pruning for dense retrieval. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management, pages 3453-3457, 2021.

[23] 尼古拉·托内洛托（Nicola Tonellotto）和克雷格·麦克唐纳（Craig Macdonald）。密集检索的查询嵌入剪枝。见《第30届ACM国际信息与知识管理会议论文集》，第3453 - 3457页，2021年。

[24] Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed, and Arnold Overwijk. Approximate nearest neighbor negative contrastive learning for dense text retrieval. arXiv preprint arXiv:2007.00808, 2020.

[24] 熊磊（Lee Xiong）、熊晨艳（Chenyan Xiong）、李烨（Ye Li）、邓国峰（Kwok-Fung Tang）、刘佳琳（Jialin Liu）、保罗·贝内特（Paul Bennett）、朱奈德·艾哈迈德（Junaid Ahmed）和阿诺德·奥弗维克（Arnold Overwijk）。用于密集文本检索的近似最近邻负对比学习。预印本arXiv:2007.00808，2020年。

[25] Andrew Yates, Rodrigo Nogueira, and Jimmy Lin. Pretrained transformers for text ranking: Bert and beyond. In Proceedings of the 14th ACM International Conference on web search and data mining, pages 1154-1156, 2021.

[25] 安德鲁·耶茨（Andrew Yates）、罗德里戈·诺盖拉（Rodrigo Nogueira）和林吉米（Jimmy Lin）。用于文本排序的预训练Transformer：BERT及其他。见《第14届ACM国际网络搜索与数据挖掘会议论文集》，第1154 - 1156页，2021年。

[26] Hamed Zamani, Mostafa Dehghani, W Bruce Croft, Erik Learned-Miller, and Jaap Kamps. From neural re-ranking to neural ranking: Learning a sparse representation for inverted indexing. In Proceedings of the 27th ACM international conference on information and knowledge management, pages 497-506, 2018.

[26] 哈米德·扎马尼（Hamed Zamani）、穆斯塔法·德赫加尼（Mostafa Dehghani）、W·布鲁斯·克罗夫特（W Bruce Croft）、埃里克·利尔内德 - 米勒（Erik Learned - Miller）和亚普·坎普斯（Jaap Kamps）。从神经重排序到神经排序：学习用于倒排索引的稀疏表示。见《第27届ACM国际信息与知识管理会议论文集》，第497 - 506页，2018年。