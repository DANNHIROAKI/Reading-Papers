# Sparse, Dense, and Attentional Representations for Text Retrieval

# 用于文本检索的稀疏、密集和注意力表示

Yi Luan,Jacob Eisenstein*, Kristina Toutanova*, Michael Collins

栾毅，雅各布·艾森斯坦*，克里斯蒂娜·图托纳娃*，迈克尔·柯林斯

Google Research

谷歌研究院

\{luanyi, jeisenstein, kristout, mjcollins\}@google.com

\{luanyi, jeisenstein, kristout, mjcollins\}@google.com

## Abstract

## 摘要

Dual encoders perform retrieval by encoding documents and queries into dense low-dimensional vectors, scoring each document by its inner product with the query. We investigate the capacity of this architecture relative to sparse bag-of-words models and attentional neural networks. Using both theoretical and empirical analysis, we establish connections between the encoding dimension, the margin between gold and lower-ranked documents, and the document length, suggesting limitations in the capacity of fixed-length encodings to support precise retrieval of long documents. Building on these insights, we propose a simple neural model that combines the efficiency of dual encoders with some of the expressiveness of more costly attentional architectures, and explore sparse-dense hybrids to capitalize on the precision of sparse retrieval. These models outperform strong alternatives in large-scale retrieval.

双编码器通过将文档和查询编码为密集的低维向量，并通过文档与查询的内积对每个文档进行评分来执行检索。我们研究了这种架构相对于稀疏词袋模型和注意力神经网络的能力。通过理论和实证分析，我们建立了编码维度、黄金文档与低排名文档之间的差距以及文档长度之间的联系，这表明固定长度编码在支持精确检索长文档方面的能力存在局限性。基于这些见解，我们提出了一种简单的神经模型，该模型将双编码器的效率与更昂贵的注意力架构的一些表达能力相结合，并探索稀疏 - 密集混合模型以利用稀疏检索的精度。这些模型在大规模检索中优于强大的替代方案。

## 1 Introduction

## 1 引言

Retrieving relevant documents is a core task for language technology, and is a component of applications such as information extraction and question answering (e.g., Narasimhan et al., 2016; Kwok et al., 2001; Voorhees, 2001). While classical information retrieval has focused on heuristic weights for sparse bag-of-words representations (Spärck Jones, 1972), more recent work has adopted a two-stage retrieval and ranking pipeline, where a large number of documents are retrieved using sparse high dimensional query/document representations, and are further reranked with learned neural models (Mitra and Craswell, 2018). This two-stage approach has achieved state-of-the-art results on IR benchmarks (Nogueira and Cho, 2019; Yang et al., 2019; Nogueira et al., 2019a), especially since sizable annotated data has become available for training deep neural models (Dietz et al., 2018; Craswell et al., 2020). However, this pipeline suffers from a strict upper bound imposed by any recall errors in the first-stage retrieval model: for example, the recall @1000 for BM25 reported by Yan et al. (2020) is 69.4.

检索相关文档是语言技术的核心任务，也是信息提取和问答等应用的组成部分（例如，纳拉西姆汉等人，2016 年；郭等人，2001 年；沃里斯，2001 年）。虽然经典信息检索侧重于为稀疏词袋表示设置启发式权重（斯帕克·琼斯，1972 年），但最近的工作采用了两阶段检索和排序流程，即使用稀疏高维查询/文档表示检索大量文档，然后使用学习到的神经模型对其进行进一步排序（米特拉和克拉斯韦尔，2018 年）。这种两阶段方法在信息检索基准测试中取得了最先进的成果（诺盖拉和赵，2019 年；杨等人，2019 年；诺盖拉等人，2019a），特别是自从有了大量带注释的数据用于训练深度神经网络模型以来（迪茨等人，2018 年；克拉斯韦尔等人，2020 年）。然而，这种流程受到第一阶段检索模型召回误差的严格上限限制：例如，严等人（2020 年）报告的 BM25 在 1000 个结果下的召回率为 69.4%。

<!-- Media -->

<!-- figureText: 100 BM25-uni BM25-bi DE-BERT-768 BERT-init 400 recall@1 80 60 40 20 50 100 200 passage length -->

<img src="https://cdn.noedgeai.com/0195aec6-1905-71b4-8b6e-19da2d5cc901_0.jpg?x=851&y=559&w=653&h=268&r=0"/>

Figure 1: Recall @1 for retrieving passage containing a query from three million candidates. The figure compares a fine-tuned BERT-based dual encoder (DE-BERT- 768), an off-the-shelf BERT-based encoder with average pooling (BERT-init), and sparse term-based retrieval (BM25), while binning passages by length.

图 1：从三百万个候选段落中检索包含查询的段落的 1 召回率。该图比较了微调后的基于 BERT 的双编码器（DE - BERT - 768）、使用平均池化的现成基于 BERT 的编码器（BERT - init）和基于稀疏词项的检索（BM25），同时按段落长度进行分组。

<!-- Media -->

A promising alternative is to perform first-stage retrieval using learned dense low-dimensional encodings of documents and queries (Huang et al., 2013; Reimers and Gurevych, 2019; Gillick et al., 2019; Karpukhin et al., 2020). The dual encoder model scores each document by the inner product between its encoding and that of the query. Unlike full attentional architectures, which require extensive computation on each candidate document, the dual encoder can be easily applied to very large document collections thanks to efficient algorithms for inner product search; unlike untrained sparse retrieval models, it can exploit machine learning to generalize across related terms.

一种有前景的替代方法是使用学习到的文档和查询的密集低维编码进行第一阶段检索（黄等人，2013 年；赖默斯和古雷维奇，2019 年；吉利克等人，2019 年；卡尔普欣等人，2020 年）。双编码器模型通过文档编码与查询编码的内积对每个文档进行评分。与需要对每个候选文档进行大量计算的全注意力架构不同，由于内积搜索的高效算法，双编码器可以轻松应用于非常大的文档集合；与未训练的稀疏检索模型不同，它可以利用机器学习在相关词项之间进行泛化。

To assess the relevance of a document to an information-seeking query, models must both (i) check for precise term overlap (for example, presence of key entities in the query) and (ii) compute semantic similarity generalizing across related concepts. Sparse retrieval models excel at the first sub-problem, while learned dual encoders can be better at the second. Recent history in NLP might suggest that learned dense representations should always outperform sparse features overall, but this is not necessarily true: as shown in Figure 1, the BM25 model (Robertson et al., 2009) can outperform a dual encoder based on BERT, particularly on longer documents and on a task that requires precise detection of word overlap. ${}^{1}$ This raises questions about the limitations of dual encoders, and the circumstances in which these powerful models do not yet reach the state of the art. Here we explore these questions using both theoretical and empirical tools, and propose a new architecture that leverages the strengths of dual encoders while avoiding some of their weaknesses.

为了评估文档与信息查询的相关性，模型必须同时（i）检查精确的词项重叠（例如，查询中关键实体的存在）和（ii）计算跨相关概念的语义相似度。稀疏检索模型在第一个子问题上表现出色，而学习到的双编码器在第二个问题上可能表现更好。自然语言处理的近期历史可能表明，学习到的密集表示在总体上应该总是优于稀疏特征，但情况并非一定如此：如图 1 所示，BM25 模型（罗伯逊等人，2009 年）可以优于基于 BERT 的双编码器，特别是在较长的文档和需要精确检测词项重叠的任务上。${}^{1}$ 这引发了关于双编码器局限性的问题，以及这些强大模型尚未达到最先进水平的情况。在这里，我们使用理论和实证工具来探索这些问题，并提出一种新的架构，该架构利用双编码器的优势，同时避免其一些弱点。

---

<!-- Footnote -->

*equal contribution

*同等贡献

<!-- Footnote -->

---

We begin with a theoretical investigation of compressive dual encoders - dense encodings whose dimension is below the vocabulary size — and analyze their ability to preserve distinctions made by sparse bag-of-words retrieval models, which we term their fidelity. Fidelity is important for the sub-problem of detecting precise term overlap, and is a tractable proxy for capacity. Using the theory of dimensionality reduction, we relate fidelity to the normalized margin between the gold retrieval result and its competitors, and show that this margin is in turn related to the length of documents in the collection. We validate the theory with an empirical investigation of the effects of random projection compression on sparse BM25 retrieval using queries and documents from TREC-CAR, a recent IR benchmark (Dietz et al., 2018).

我们首先对压缩双编码器（即维度低于词汇量的密集编码）进行理论研究，并分析它们保留稀疏词袋检索模型所做区分的能力，我们将这种能力称为保真度。保真度对于检测精确术语重叠这一子问题很重要，并且是容量的一个易处理的代理指标。利用降维理论，我们将保真度与黄金检索结果与其竞争对手之间的归一化差距联系起来，并表明这个差距反过来又与集合中文档的长度有关。我们通过对使用TREC - CAR（近期的信息检索基准，Dietz等人，2018年）中的查询和文档进行随机投影压缩对稀疏BM25检索的影响进行实证研究，验证了这一理论。

Next, we offer a multi-vector encoding model, which is computationally feasible for retrieval like the dual-encoder architecture and achieves significantly better quality. A simple hybrid that interpolates models based on dense and sparse representations leads to further improvements.

接下来，我们提出一种多向量编码模型，该模型在计算上与双编码器架构一样适用于检索，并且能显著提高质量。一种基于密集和稀疏表示对模型进行插值的简单混合方法可带来进一步的改进。

We compare the performance of dual encoders, multi-vector encoders, and their sparse-dense hybrids with classical sparse retrieval models and attentional neural networks, as well as state-of-the-art published results where available. Our evaluations include open retrieval benchmarks (MS MARCO passage and document), and passage retrieval for question answering (Natural Questions). We confirm prior findings that full attentional architectures excel at reranking tasks, but are not efficient enough for large-scale retrieval. Of the more efficient alternatives, the hybridized multi-vector encoder is at or near the top in every evaluation, outperforming state-of-the-art retrieval results in MS MARCO. Our code is publicly available at https://github.com/google-research/language/tree/ master/language/multivec.

我们将双编码器、多向量编码器及其稀疏 - 密集混合模型的性能与经典的稀疏检索模型和注意力神经网络进行比较，并在有可用数据的情况下与最先进的已发表结果进行比较。我们的评估包括开放检索基准（MS MARCO段落和文档）以及问答的段落检索（自然问题）。我们证实了先前的研究结果，即全注意力架构在重排序任务中表现出色，但对于大规模检索来说效率不够。在更高效的替代方案中，混合多向量编码器在每次评估中都处于或接近领先地位，在MS MARCO中优于最先进的检索结果。我们的代码可在https://github.com/google - research/language/tree/master/language/multivec上公开获取。

## 2 Analyzing dual encoder fidelity

## 2 分析双编码器的保真度

A query or a document is a sequence of words drawn from some vocabulary $\mathcal{V}$ . Throughout this section we assume a representation of queries and documents typically used in sparse bag-of-words models: each query $q$ and document $d$ is a vector in ${\mathbb{R}}^{v}$ where $v$ is the vocabulary size. We take the inner product $\langle q,d\rangle$ to be the relevance score of document $d$ for query $q$ . This framework accounts for a several well-known ranking models, including boolean inner product, TF-IDF, and BM25.

查询或文档是从某个词汇表$\mathcal{V}$中抽取的单词序列。在本节中，我们假设采用稀疏词袋模型中通常使用的查询和文档表示方法：每个查询$q$和文档$d$都是${\mathbb{R}}^{v}$中的向量，其中$v$是词汇表的大小。我们将内积$\langle q,d\rangle$作为文档$d$相对于查询$q$的相关性得分。这个框架涵盖了几种著名的排序模型，包括布尔内积、TF - IDF和BM25。

We will compare sparse retrieval models with compressive dual encoders, for which we write $f\left( d\right)$ and $f\left( q\right)$ to indicate compression of $d$ and $q$ to ${\mathbb{R}}^{k}$ ,with $k \ll  v$ ,and where $k$ does not vary with the document length. For these models, the relevance score is the inner product $\langle f\left( q\right) ,f\left( d\right) \rangle$ . (In § 3, we consider encoders that apply to sequences of tokens rather than vectors of counts.)

我们将把稀疏检索模型与压缩双编码器进行比较，对于压缩双编码器，我们用$f\left( d\right)$和$f\left( q\right)$表示将$d$和$q$压缩到${\mathbb{R}}^{k}$，其中$k \ll  v$，并且$k$不随文档长度而变化。对于这些模型，相关性得分是内积$\langle f\left( q\right) ,f\left( d\right) \rangle$。（在§3中，我们考虑应用于标记序列而非计数向量的编码器。）

A fundamental question is how the capacity of dual encoders varies with the embedding size $k$ . In this section we focus on the related, more tractable notion of fidelity: how much can we compress the input while maintaining the ability to mimic the performance of bag-of-words retrieval? We explore this question mainly through the encoding model of random projections, but also discuss more general dimensionality reduction in $§{2.2}$ .

一个基本问题是双编码器的容量如何随嵌入大小$k$变化。在本节中，我们关注与之相关且更易处理的保真度概念：在保持模拟词袋检索性能的能力的同时，我们可以对输入进行多大程度的压缩？我们主要通过随机投影的编码模型来探讨这个问题，但也会在$§{2.2}$中讨论更一般的降维方法。

### 2.1 Random projections

### 2.1 随机投影

To establish baselines on the fidelity of compressive dual encoder retrieval, we now consider encoders based on random projections (Vem-pala,2004). The encoder is defined as $f\left( x\right)  =$ ${Ax}$ ,where $A \in  {\mathbb{R}}^{k \times  v}$ is a random matrix. In Rademacher embeddings,each element ${a}_{i,j}$ of the matrix $A$ is sampled with equal probability from two possible values: $\left\{  {-\frac{1}{\sqrt{k}},\frac{1}{\sqrt{k}}}\right\}$ . In Gaussian embeddings,each ${a}_{i,j} \sim  N\left( {0,{k}^{-1/2}}\right)$ . A pairwise ranking error occurs when $\left\langle  {q,{d}_{1}}\right\rangle   >$ $\left\langle  {q,{d}_{2}}\right\rangle$ but $\left\langle  {{Aq},A{d}_{1}}\right\rangle   < \left\langle  {{Aq},A{d}_{2}}\right\rangle$ . Using such random projections, it is possible to bound the probability of any such pairwise error in terms of the embedding size.

为了建立压缩双编码器检索保真度的基线，我们现在考虑基于随机投影的编码器（韦姆帕拉，2004年）。编码器定义为 $f\left( x\right)  =$ ${Ax}$ ，其中 $A \in  {\mathbb{R}}^{k \times  v}$ 是一个随机矩阵。在拉德马赫嵌入（Rademacher embeddings）中，矩阵 $A$ 的每个元素 ${a}_{i,j}$ 以相等的概率从两个可能的值中采样： $\left\{  {-\frac{1}{\sqrt{k}},\frac{1}{\sqrt{k}}}\right\}$ 。在高斯嵌入（Gaussian embeddings）中，每个 ${a}_{i,j} \sim  N\left( {0,{k}^{-1/2}}\right)$ 。当 $\left\langle  {q,{d}_{1}}\right\rangle   >$ $\left\langle  {q,{d}_{2}}\right\rangle$ 但 $\left\langle  {{Aq},A{d}_{1}}\right\rangle   < \left\langle  {{Aq},A{d}_{2}}\right\rangle$ 时，会出现成对排序错误。使用这种随机投影，可以根据嵌入大小来界定任何此类成对错误的概率。

---

<!-- Footnote -->

${}^{1}$ See $§4$ for experimental details.

${}^{1}$ 实验细节见 $§4$ 。

<!-- Footnote -->

---

Definition 2.1. For a query $q$ and pair of documents $\left( {{d}_{1},{d}_{2}}\right)$ such that $\left\langle  {q,{d}_{1}}\right\rangle   \geq  \left\langle  {q,{d}_{2}}\right\rangle$ ,the normalized margin is defined as, $\mu \left( {q,{d}_{1},{d}_{2}}\right)  =$ $\frac{\left\langle  q,{d}_{1} - {d}_{2}\right\rangle  }{\parallel q\parallel  \times  \parallel {d}_{1} - {d}_{2}\parallel }$ .

定义2.1。对于查询 $q$ 和文档对 $\left( {{d}_{1},{d}_{2}}\right)$ ，使得 $\left\langle  {q,{d}_{1}}\right\rangle   \geq  \left\langle  {q,{d}_{2}}\right\rangle$ ，归一化边际定义为 $\mu \left( {q,{d}_{1},{d}_{2}}\right)  =$ $\frac{\left\langle  q,{d}_{1} - {d}_{2}\right\rangle  }{\parallel q\parallel  \times  \parallel {d}_{1} - {d}_{2}\parallel }$ 。

Lemma 1. Define a matrix $A \in  {\mathbb{R}}^{k \times  d}$ of Gaussian or Rademacher embeddings. Define vectors $q,{d}_{1},{d}_{2}$ such that $\mu \left( {q,{d}_{1},{d}_{2}}\right)  = \epsilon  > 0$ . A ranking error occurs when $\left\langle  {{Aq},A{d}_{2}}\right\rangle   \geq  \left\langle  {{Aq},A{d}_{1}}\right\rangle$ . If $\beta$ is the probability of such an error then,

引理1。定义一个高斯或拉德马赫嵌入的矩阵 $A \in  {\mathbb{R}}^{k \times  d}$ 。定义向量 $q,{d}_{1},{d}_{2}$ ，使得 $\mu \left( {q,{d}_{1},{d}_{2}}\right)  = \epsilon  > 0$ 。当 $\left\langle  {{Aq},A{d}_{2}}\right\rangle   \geq  \left\langle  {{Aq},A{d}_{1}}\right\rangle$ 时，会出现排序错误。如果 $\beta$ 是此类错误的概率，那么

$$
\beta  \leq  4\exp \left( {-\frac{k}{2}\left( {{\epsilon }^{2}/2 - {\epsilon }^{3}/3}\right) }\right) . \tag{1}
$$

The proof, which builds on well-known results about random projections,is found in $§$ A.1. By solving (1) for $k$ ,we can derive an embedding size that guarantees a desired upper bound on the pairwise error probability,

该证明基于关于随机投影的著名结果，见 $§$ A.1。通过对（1）式求解 $k$ ，我们可以推导出一个嵌入大小，该大小能保证成对错误概率的期望上限。

$$
k \geq  2{\left( {\epsilon }^{2}/2 - {\epsilon }^{3}/3\right) }^{-1}\ln \frac{4}{\beta }. \tag{2}
$$

It is convenient to derive a simpler but looser quadratic bound (proved in $§$ A.2):

推导一个更简单但更宽松的二次界（证明见 $§$ A.2）会更方便：

Corollary 1. Define vectors $q,{d}_{1},{d}_{2}$ such that $\epsilon  = \mu \left( {q,{d}_{1},{d}_{2}}\right)  > 0$ . If $A \in  {\mathbb{R}}^{k \times  v}$ is a matrix of random Gaussian or Rademacher embeddings such that $k > {12}{\epsilon }^{-2}\ln \frac{4}{\beta }$ ,then $\Pr \left( {\left\langle  {{Aq},A{d}_{1}}\right\rangle   \leq  }\right.$ $\left. \left\langle  {{Aq},A{d}_{2}}\right\rangle  \right)  \leq  \beta$ .

推论1。定义向量 $q,{d}_{1},{d}_{2}$ ，使得 $\epsilon  = \mu \left( {q,{d}_{1},{d}_{2}}\right)  > 0$ 。如果 $A \in  {\mathbb{R}}^{k \times  v}$ 是一个随机高斯或拉德马赫嵌入的矩阵，使得 $k > {12}{\epsilon }^{-2}\ln \frac{4}{\beta }$ ，那么 $\Pr \left( {\left\langle  {{Aq},A{d}_{1}}\right\rangle   \leq  }\right.$ $\left. \left\langle  {{Aq},A{d}_{2}}\right\rangle  \right)  \leq  \beta$ 。

On the tightness of the bound. Let ${k}^{ * }\left( {q,{d}_{1},{d}_{2}}\right)$ denote the lowest dimension Gaussian or Rademacher random projection following the definition in Lemma 1, for which $\Pr \left( {\left\langle  {{Aq},A{d}_{1}}\right\rangle   < \left\langle  {{Aq},A{d}_{2}}\right\rangle  }\right)  \leq  \beta$ , for a given document pair $\left( {{d}_{1},{d}_{2}}\right)$ and query $q$ with normalized margin $\epsilon$ . Our lemma places an upper bound on ${k}^{ * }$ ,saying that ${k}^{ * }\left( {q,{d}_{1},{d}_{2}}\right)  \leq  2{\left( {\epsilon }^{2}/2 - {\epsilon }^{3}/3\right) }^{-1}\ln \frac{4}{\beta }$ . Any $k \geq  {k}^{ * }\left( {q,{d}_{1},{d}_{2}}\right)$ has sufficiently low probability of error,but lower values of $k$ could potentially also have the desired property. Later in this section we perform empirical evaluation to study the tightness of the bound; although theoretical tightness (up to a constant factor) is suggested by results on the optimality of the distributional Johnson-Lindenstrauss lemma (Johnson and Lindenstrauss, 1984; Jayram and Woodruff, 2013; Kane et al., 2011), here we study the question only empirically.

关于界的紧性。设 ${k}^{ * }\left( {q,{d}_{1},{d}_{2}}\right)$ 表示遵循引理 1 定义的最低维高斯或拉德马赫随机投影，对于给定的文档对 $\left( {{d}_{1},{d}_{2}}\right)$ 和查询 $q$ 以及归一化间隔 $\epsilon$，有 $\Pr \left( {\left\langle  {{Aq},A{d}_{1}}\right\rangle   < \left\langle  {{Aq},A{d}_{2}}\right\rangle  }\right)  \leq  \beta$ 。我们的引理对 ${k}^{ * }$ 给出了一个上界，即 ${k}^{ * }\left( {q,{d}_{1},{d}_{2}}\right)  \leq  2{\left( {\epsilon }^{2}/2 - {\epsilon }^{3}/3\right) }^{-1}\ln \frac{4}{\beta }$ 。任何 $k \geq  {k}^{ * }\left( {q,{d}_{1},{d}_{2}}\right)$ 出现错误的概率都足够低，但 $k$ 的较低值也可能具有所需的性质。在本节后面，我们进行实证评估以研究该界的紧性；尽管分布型约翰逊 - 林登施特劳斯引理（Johnson 和 Lindenstrauss，1984；Jayram 和 Woodruff，2013；Kane 等人，2011）的最优性结果表明了理论上的紧性（至多相差一个常数因子），但在这里我们仅通过实证研究这个问题。

#### 2.1.1 Recall-at- $r$

#### 2.1.1 $r$ 召回率

In retrieval applications, it is important to return the desired result within the top $r$ search results. For query $q$ ,define ${d}_{1}$ as the document that maximizes some inner product ranking metric. The probability of returning ${d}_{1}$ in the top $r$ results after random projection can be bounded by a function of the embedding size and normalized margin:

在检索应用中，在前 $r$ 个搜索结果中返回所需结果非常重要。对于查询 $q$，将 ${d}_{1}$ 定义为使某个内积排名指标最大化的文档。随机投影后在前 $r$ 个结果中返回 ${d}_{1}$ 的概率可以由嵌入大小和归一化间隔的函数来界定：

Lemma 2. Consider a query $q$ ,with target document ${d}_{1}$ ,and document collection $\mathcal{D}$ that excludes ${d}_{1}$ ,and such that $\forall {d}_{2} \in  \mathcal{D},\mu \left( {q,{d}_{1},{d}_{2}}\right)  > 0$ . Define ${r}_{0}$ to be any integer such that $1 \leq  {r}_{0} \leq  \left| \mathcal{D}\right|$ . Define $\epsilon$ to be the ${r}_{0}$ ’th smallest normalized margin $\mu \left( {q,{d}_{1},{d}_{2}}\right)$ for any ${d}_{2} \in  \mathcal{D}$ ,and for simplicity assume that only a single document ${d}_{2} \in  \mathcal{D}$ has $\mu \left( {q,{d}_{1},{d}_{2}}\right)  = \epsilon {.}^{2}$

引理 2。考虑一个查询 $q$，目标文档为 ${d}_{1}$，文档集合为 $\mathcal{D}$（不包括 ${d}_{1}$），且满足 $\forall {d}_{2} \in  \mathcal{D},\mu \left( {q,{d}_{1},{d}_{2}}\right)  > 0$ 。定义 ${r}_{0}$ 为任意整数，使得 $1 \leq  {r}_{0} \leq  \left| \mathcal{D}\right|$ 。定义 $\epsilon$ 为任意 ${d}_{2} \in  \mathcal{D}$ 的第 ${r}_{0}$ 小归一化间隔 $\mu \left( {q,{d}_{1},{d}_{2}}\right)$，为简单起见，假设只有一个文档 ${d}_{2} \in  \mathcal{D}$ 具有 $\mu \left( {q,{d}_{1},{d}_{2}}\right)  = \epsilon {.}^{2}$

Define a matrix $A \in  {\mathbb{R}}^{k \times  d}$ of Gaussian or Rademacher embeddings. Define $R$ to be a random variable such that $R =  \mid  \left\{  {{d}_{2} \in  \mathcal{D}}\right.$ : $\left. \left\langle  {{Aq},A{d}_{1}\rangle  \leq  \left\langle  {{Aq},A{d}_{2}}\right\rangle  }\right\rangle  \right\}$ ,and let $C = 4(\left| \mathcal{D}\right|  -$ $\left. {{r}_{0} + 1}\right)$ . Then

定义一个高斯或拉德马赫嵌入矩阵 $A \in  {\mathbb{R}}^{k \times  d}$ 。定义 $R$ 为一个随机变量，使得 $R =  \mid  \left\{  {{d}_{2} \in  \mathcal{D}}\right.$ ： $\left. \left\langle  {{Aq},A{d}_{1}\rangle  \leq  \left\langle  {{Aq},A{d}_{2}}\right\rangle  }\right\rangle  \right\}$ ，并设 $C = 4(\left| \mathcal{D}\right|  -$ $\left. {{r}_{0} + 1}\right)$ 。那么

$$
\Pr \left( {R \geq  {r}_{0}}\right)  \leq  C\exp \left( {-\frac{k}{2}\left( {{\epsilon }^{2}/2 - {\epsilon }^{3}/3}\right) }\right) .
$$

The proof is in $§$ A.3. A direct consequence of the lemma is that to achieve recall-at- ${r}_{0} = 1$ for a given $\left( {q,{d}_{1},\mathcal{D}}\right)$ triple with probability $\geq  1 - \beta$ ,it is sufficient to set

证明见 $§$ A.3。该引理的一个直接推论是，对于给定的 $\left( {q,{d}_{1},\mathcal{D}}\right)$ 三元组，要以概率 $\geq  1 - \beta$ 实现 ${r}_{0} = 1$ 召回率，只需设置

$$
k \geq  \frac{2}{{\epsilon }^{2}/2 - {\epsilon }^{3}/3}\ln \frac{4\left( {\left| \mathcal{D}\right|  - {r}_{0} + 1}\right) }{\beta }, \tag{3}
$$

where $\epsilon$ is the ${r}_{0}$ ’th smallest normalized margin.

其中 $\epsilon$ 是第 ${r}_{0}$ 小归一化间隔。

As with the bound on pairwise relevance errors in Lemma 1, Lemma 2 implies an upper bound on the minimum random projection dimension ${k}^{ * }\left( {q,{d}_{1},\mathcal{D}}\right)$ that recalls ${d}_{1}$ in the top ${r}_{0}$ results with probability $\geq  1 - \beta$ . Due to the application of the union bound and worst-case assumptions about the normalized margins of documents in ${\mathcal{D}}_{\epsilon }$ ,this bound is potentially loose. Later in this section we examine the empirical relationship between maximum document length, the distribution of normalized margins,and ${k}^{ * }$ .

与引理1中成对相关性误差的界类似，引理2给出了最小随机投影维度 ${k}^{ * }\left( {q,{d}_{1},\mathcal{D}}\right)$ 的上界，该维度能以概率 $\geq  1 - \beta$ 在排名前 ${r}_{0}$ 的结果中召回 ${d}_{1}$。由于使用了联合界以及对 ${\mathcal{D}}_{\epsilon }$ 中文档归一化间距的最坏情况假设，这个界可能是宽松的。在本节后面，我们将研究最大文档长度、归一化间距的分布和 ${k}^{ * }$ 之间的实证关系。

#### 2.1.2 Application to Boolean inner product

#### 2.1.2 对布尔内积的应用

Boolean inner product is a retrieval function in which $d,q \in  \{ 0,1{\} }^{v}$ over a vocabulary of size $v$ ,with ${d}_{i}$ indicating the presence of term $i$ in the document (and analogously for ${q}_{i}$ ). The relevance score $\langle q,d\rangle$ is then the number of terms that appear in both $q$ and $d$ . For this simple retrieval function, it is possible to compute an embedding size that guarantees a desired pairwise error probability over an entire dataset of documents.

布尔内积是一种检索函数，其中在大小为 $v$ 的词汇表上 $d,q \in  \{ 0,1{\} }^{v}$，${d}_{i}$ 表示术语 $i$ 在文档中出现（${q}_{i}$ 同理）。然后，相关性得分 $\langle q,d\rangle$ 就是同时出现在 $q$ 和 $d$ 中的术语数量。对于这个简单的检索函数，可以计算出一个嵌入大小，该大小能保证在整个文档数据集中达到所需的成对误差概率。

---

<!-- Footnote -->

${}^{2}$ The case where multiple documents are tied with normalized margin $\epsilon$ is straightforward but slightly complicates the analysis.

${}^{2}$ 多个文档的归一化间距 $\epsilon$ 相同的情况很直接，但会使分析稍微复杂一些。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: 10000 4000 6000 8000 ${\left( {\varepsilon }^{2}/2 - {\varepsilon }^{3}/3\right) }^{-1}$ for bin 7500 5000 2500 0 2000 -->

<img src="https://cdn.noedgeai.com/0195aec6-1905-71b4-8b6e-19da2d5cc901_3.jpg?x=232&y=181&w=538&h=274&r=0"/>

Figure 2: Minimum $k$ sufficient for Rademacher em-beddings to approximate BM25 pairwise rankings on TREC-CAR with error rate $\beta  < {.05}$ .

图2：在TREC - CAR数据集上，拉德马赫（Rademacher）嵌入以误差率 $\beta  < {.05}$ 近似BM25成对排名所需的最小 $k$。

<!-- Media -->

Corollary 2. For a set of documents $\mathcal{D} = \{ d \in$ $\left. {\{ 0,1{\} }^{v}}\right\}$ and a query $q \in  \{ 0,1{\} }^{v}$ ,let ${L}_{D} =$ $\mathop{\max }\limits_{{d \in  \mathcal{D}}}\parallel d{\parallel }^{2}$ and ${L}_{Q} = \parallel q{\parallel }^{2}$ . Let $A \in  {\mathbb{R}}^{k \times  v}$ be a matrix of random Rademacher or Gaussian embeddings such that $k \geq  {24}{L}_{Q}{L}_{D}\ln \frac{4}{\beta }$ . Then for any ${d}_{1},{d}_{2} \in  \mathcal{D}$ such that $\left\langle  {q,{d}_{1}}\right\rangle   > \left\langle  {q,{d}_{2}}\right\rangle$ ,the probability that $\left\langle  {{Aq},A{d}_{1}}\right\rangle   \leq  \left\langle  {{Aq},A{d}_{2}}\right\rangle$ is $\leq  \beta$ .

推论2。对于一组文档 $\mathcal{D} = \{ d \in$ $\left. {\{ 0,1{\} }^{v}}\right\}$ 和一个查询 $q \in  \{ 0,1{\} }^{v}$，设 ${L}_{D} =$ $\mathop{\max }\limits_{{d \in  \mathcal{D}}}\parallel d{\parallel }^{2}$ 且 ${L}_{Q} = \parallel q{\parallel }^{2}$。设 $A \in  {\mathbb{R}}^{k \times  v}$ 是一个随机拉德马赫或高斯嵌入矩阵，使得 $k \geq  {24}{L}_{Q}{L}_{D}\ln \frac{4}{\beta }$。那么对于任何满足 $\left\langle  {q,{d}_{1}}\right\rangle   > \left\langle  {q,{d}_{2}}\right\rangle$ 的 ${d}_{1},{d}_{2} \in  \mathcal{D}$，$\left\langle  {{Aq},A{d}_{1}}\right\rangle   \leq  \left\langle  {{Aq},A{d}_{2}}\right\rangle$ 的概率为 $\leq  \beta$。

The proof is in $§$ A.4. The corollary shows that for boolean inner product ranking, we can guarantee any desired error bound $\beta$ by choosing an embedding size $k$ that grows linearly in ${L}_{D}$ ,the number of unique terms in the longest document.

证明见 $§$ A.4。该推论表明，对于布尔内积排名，我们可以通过选择一个嵌入大小 $k$ 来保证任何所需的误差界 $\beta$，该嵌入大小与最长文档中唯一术语的数量 ${L}_{D}$ 呈线性增长关系。

#### 2.1.3 Application to TF-IDF and BM25

#### 2.1.3 对TF - IDF和BM25的应用

Both TF-IDF (Spärck Jones, 1972) and BM25 (Robertson et al., 2009) can be written as inner products between bag-of-words representations of the document and query as described earlier in this section. Set the query representation ${\widetilde{q}}_{i} = {q}_{i} \times  {\mathrm{{IDF}}}_{i}$ ,where ${q}_{i}$ indicates the presence of the term in the query and ${\mathrm{{IDF}}}_{i}$ indicates the inverse document frequency of term $i$ . The TF-IDF score is then $\langle \widetilde{q},d\rangle$ . For BM25,we define $\widetilde{d} \in  {\mathbb{R}}^{v}$ ,with each ${\widetilde{d}}_{i}$ a function of the count ${d}_{i}$ and the document length (and hyperparameters); $\operatorname{BM25}\left( {q,d}\right)$ is then $\langle \widetilde{q},\widetilde{d}\rangle$ . Due to its practical utility in retrieval, we now focus on BM25.

TF-IDF（斯帕克·琼斯，1972年）和BM25（罗伯逊等人，2009年）都可以表示为文档和查询的词袋表示之间的内积，如本节前面所述。设查询表示为${\widetilde{q}}_{i} = {q}_{i} \times  {\mathrm{{IDF}}}_{i}$，其中${q}_{i}$表示查询中是否存在该词项，${\mathrm{{IDF}}}_{i}$表示词项$i$的逆文档频率。那么TF-IDF得分就是$\langle \widetilde{q},d\rangle$。对于BM25，我们定义$\widetilde{d} \in  {\mathbb{R}}^{v}$，其中每个${\widetilde{d}}_{i}$是计数${d}_{i}$和文档长度（以及超参数）的函数；然后$\operatorname{BM25}\left( {q,d}\right)$就是$\langle \widetilde{q},\widetilde{d}\rangle$。由于BM25在检索中的实际效用，我们现在将重点放在BM25上。

Pairwise accuracy. We use empirical data to test the applicability of Lemma 1 to the BM25 relevance model. We select query-document triples $\left( {q,{d}_{1},{d}_{2}}\right)$ from the TREC-CAR dataset (Dietz et al.,2018) by considering all possible $\left( {q,{d}_{2}}\right)$ , and selecting ${d}_{1} = \arg \mathop{\max }\limits_{d}\operatorname{BM25}\left( {q,d}\right)$ . We bin the triples by the normalized margin $\epsilon$ ,and compute the quantity ${\left( {\epsilon }^{2}/2 - {\epsilon }^{3}/3\right) }^{-1}$ . According to Lemma 1, the minimum embedding size of a random projection ${k}^{ * }$ which has $\leq  \beta$ probability of making an error on a triple with normalized margin $\epsilon$ is upper bounded by a linear function of this quantity. In particular,for $\beta  = {.05}$ ,the Lemma entails that ${k}^{ * } \leq  {8.76}{\left( {\epsilon }^{2}/2 - {\epsilon }^{3}/3\right) }^{-1}$ . In this experiment we measure the empirical value of ${k}^{ * }$ to evaluate the tightness of the bound.

成对准确率。我们使用实证数据来测试引理1对BM25相关性模型的适用性。我们从TREC - CAR数据集（迪茨等人，2018年）中选择查询 - 文档三元组$\left( {q,{d}_{1},{d}_{2}}\right)$，考虑所有可能的$\left( {q,{d}_{2}}\right)$，并选择${d}_{1} = \arg \mathop{\max }\limits_{d}\operatorname{BM25}\left( {q,d}\right)$。我们根据归一化差距$\epsilon$对三元组进行分组，并计算量${\left( {\epsilon }^{2}/2 - {\epsilon }^{3}/3\right) }^{-1}$。根据引理1，随机投影${k}^{ * }$在归一化差距为$\epsilon$的三元组上出错概率为$\leq  \beta$时的最小嵌入大小，上限由该量的线性函数确定。特别是，对于$\beta  = {.05}$，引理意味着${k}^{ * } \leq  {8.76}{\left( {\epsilon }^{2}/2 - {\epsilon }^{3}/3\right) }^{-1}$。在这个实验中，我们测量${k}^{ * }$的实证值来评估该界限的紧密程度。

The results are shown on the $x$ -axis of Figure 2. For each bin we compute the minimum embedding size required to obtain 95% pairwise accuracy in ranking ${d}_{1}$ vs ${d}_{2}$ ,using a grid of 40 possible values for $k$ between 32 and 9472,shown on the $y$ - axis. (We exclude examples that had higher values of ${\left( {\epsilon }^{2}/2 - {\epsilon }^{3}/3\right) }^{-1}$ than the range shown because they did not reach 95% accuracy for the explored range of $k$ .) The figure shows that the theoretical bound is tight up to a constant factor, and that the minimum embedding size that yields desired fidelity grows linearly with ${\left( {\epsilon }^{2}/2 - {\epsilon }^{3}/3\right) }^{-1}$ .

结果显示在图2的$x$轴上。对于每个分组，我们计算在对${d}_{1}$和${d}_{2}$进行排序时获得95%成对准确率所需的最小嵌入大小，使用$k$在32到9472之间的40个可能值的网格，显示在$y$轴上。（我们排除了${\left( {\epsilon }^{2}/2 - {\epsilon }^{3}/3\right) }^{-1}$值高于所示范围的示例，因为在探索的$k$范围内它们未达到95%的准确率。）该图表明，理论界限在一个常数因子范围内是紧密的，并且产生所需保真度的最小嵌入大小随${\left( {\epsilon }^{2}/2 - {\epsilon }^{3}/3\right) }^{-1}$线性增长。

Margins and document length. For boolean inner product, it was possible to express the minimum possible normalized margin (and therefore a sufficient embedding size) in terms of ${L}_{Q}$ and ${L}_{D}$ ,the maximum number of unique terms across all queries and documents, respectively. Unfortunately, it is difficult to analytically derive a minimum normalized margin $\epsilon$ for either TF-IDF or BM25: because each term may have a unique inverse document frequency, the minimum non-zero margin $\left\langle  {q,{d}_{1} - {d}_{2}}\right\rangle$ decreases with the number of terms in the query as each additional term creates more ways in which two documents can receive nearly the same score. We therefore study empirically how normalized margins vary with maximum document length. Using the TREC-CAR retrieval dataset, we bin documents by length. For each query, we compute the normalized margins between the document with best BM25 in the bin and all other documents in the bin, and look at the 10th, 100th, and 1000th smallest normalized margins. The distribution over these normalized margins is shown in Figure 3a, revealing that normalized margins decrease with document length. In practice, the observed minimum normalized margin for a collection of documents and queries is found to be much lower for BM25 compared to boolean inner product. For example, for the col-

边距与文档长度。对于布尔内积，可以根据 ${L}_{Q}$ 和 ${L}_{D}$（分别表示所有查询和文档中唯一词项的最大数量）来表示最小可能的归一化边距（进而确定足够的嵌入大小）。不幸的是，很难通过解析方法为 TF - IDF 或 BM25 推导出最小归一化边距 $\epsilon$：因为每个词项可能有唯一的逆文档频率，最小非零边距 $\left\langle  {q,{d}_{1} - {d}_{2}}\right\rangle$ 会随着查询中词项数量的增加而减小，因为每个额外的词项都会使两个文档获得几乎相同分数的可能性增加。因此，我们通过实证研究归一化边距如何随最大文档长度变化。使用 TREC - CAR 检索数据集，我们按长度对文档进行分组。对于每个查询，我们计算该组中 BM25 得分最高的文档与该组中所有其他文档之间的归一化边距，并查看第 10 小、第 100 小和第 1000 小的归一化边距。这些归一化边距的分布如图 3a 所示，表明归一化边距随文档长度的增加而减小。实际上，与布尔内积相比，BM25 在一组文档和查询中观察到的最小归一化边距要低得多。例如，对于图 2 中使用的集

<!-- Media -->

<!-- figureText: normalized margin 0.20 rank 10 4000 Recall@10 3000 0.75 min k 0.9 2000 0.95 1000 50 100 150 document length (b) Each line shows the minimum random projec- tion dimension $k$ that achieves a desired value of recall-at-10 for each bin of documents. rank 100 0.15 rank 1000 0.10 0.05 50 100 150 document length (a) Each datapoint is the median normalized margin per bin, and the shaded areas show the 25th and 75th quantiles. -->

<img src="https://cdn.noedgeai.com/0195aec6-1905-71b4-8b6e-19da2d5cc901_4.jpg?x=224&y=180&w=1202&h=425&r=0"/>

<!-- Media -->

Figure 3: Random projection on BM25 retrieval in TREC-CAR dataset, with documents binned by length. lection used in Figure 2, the minimum normalized margin for BM25 is ${6.8}\mathrm{e} - {06}$ ,while for boolean inner product it is 0.0169 .

图 3：TREC - CAR 数据集中对 BM25 检索进行随机投影，文档按长度分组。图 2 中使用的集合，BM25 的最小归一化边距为 ${6.8}\mathrm{e} - {06}$，而布尔内积的最小归一化边距为 0.0169。

## Document length and encoding dimension.

## 文档长度与编码维度

Figure $3\mathrm{\;b}$ shows the growth in minimum random projection dimension required to reach desired recall-at-10, using the same document bins as in Figure 3a. As predicted, the required dimension increases with the document length, while the normalized margin shrinks.

图 $3\mathrm{\;b}$ 展示了要达到期望的前 10 召回率所需的最小随机投影维度的增长情况，使用的文档分组与图 3a 相同。正如预期的那样，所需维度随文档长度的增加而增加，而归一化边距则缩小。

### 2.2 Bounds on general encoding functions

### 2.2 通用编码函数的边界

We derived upper bounds on minimum required encoding for random linear projections above, and found the bounds on $\left( {q,{d}_{1},{d}_{2}}\right)$ triples to be empirically tight up to a constant factor. More general non-linear and learned encoders could be more efficient. However, there are general theoretical results showing that it is impossible for any encoder to guarantee an inner product distortion $\left| {\langle f\left( x\right) ,f\left( y\right) \rangle -\langle x,y\rangle }\right|  \leq  \epsilon$ with an encoding that does not grow as $\Omega \left( {\epsilon }^{-2}\right)$ (Larsen and Nelson, 2017; Alon and Klartag,2017),for vectors $x,y$ with norm $\leq  1$ . These results suggest more general capacity limitations for fixed-length dual encoders when document length grows.

我们在上面推导了随机线性投影所需最小编码的上界，并发现 $\left( {q,{d}_{1},{d}_{2}}\right)$ 三元组的边界在经验上在一个常数因子范围内是紧密的。更通用的非线性和学习型编码器可能更高效。然而，有通用的理论结果表明，对于范数为 $\leq  1$ 的向量 $x,y$，任何编码器都不可能用不随 $\Omega \left( {\epsilon }^{-2}\right)$ 增长的编码来保证内积失真 $\left| {\langle f\left( x\right) ,f\left( y\right) \rangle -\langle x,y\rangle }\right|  \leq  \epsilon$（Larsen 和 Nelson，2017；Alon 和 Klartag，2017）。这些结果表明，当文档长度增加时，固定长度的双编码器存在更普遍的容量限制。

In our setting, BM25, TF-IDF, and boolean inner product can all be reformulated equivalently as inner products in a space with vectors of norm at most 1 by ${L}_{2}$ -normalizing each query vector and rescaling all document vectors by $\sqrt{{L}_{D}} =$ $\mathop{\max }\limits_{d}\parallel d\parallel$ ,a constant factor that grows with the length of the longest document. Now suppose we desire to limit the distortion on the unnormalized inner products to some value $\leq  \widetilde{\epsilon }$ ,which might guarantee a desired performance characteristic. This corresponds to decreasing the maximum normalized inner product distortion $\epsilon$ by a factor of $\sqrt{{L}_{D}}$ . According to the general bounds on dimensionality reduction mentioned in the previous paragraph, this could necessitate an increase in the encoding size by a factor of ${L}_{D}$ .

在我们的设定中，通过对每个查询向量进行 ${L}_{2}$ 归一化，并将所有文档向量按 $\sqrt{{L}_{D}} =$ $\mathop{\max }\limits_{d}\parallel d\parallel$（一个随最长文档长度增长的常数因子）进行重新缩放，BM25、TF - IDF 和布尔内积都可以等效地重新表述为范数至多为 1 的向量空间中的内积。现在假设我们希望将未归一化内积的失真限制在某个值 $\leq  \widetilde{\epsilon }$，这可能保证了期望的性能特征。这相当于将最大归一化内积失真 $\epsilon$ 降低 $\sqrt{{L}_{D}}$ 倍。根据上一段提到的降维通用边界，这可能需要将编码大小增加 ${L}_{D}$ 倍。

However, there a number of caveats to this theoretical argument. First, the theory states only that there exist vector sets that cannot be encoded into representations that grow more slowly than $\Omega \left( {\epsilon }^{-2}\right)$ ; actual documents and queries might be easier to encode if, for example, they are generated from some simple underlying stochastic process. Second, our construction achieves $\parallel d\parallel  \leq  1$ by rescaling all document vectors by a constant factor, but there may be other ways to constrain the norms while using the embedding space more efficiently. Third, in the non-linear case it might be possible to eliminate ranking errors without achieving low inner product distortion. Finally, from a practical perspective, the generalization offered by learned dual encoders might overwhelm any sacrifices in fidelity, when evaluated on real tasks of interest. Lacking theoretical tools to settle these questions, we present a set of empirical investigations in later sections of this paper. But first we explore a lightweight modification to the dual encoder, which offers gains in expressivity at limited additional computational cost.

然而，这一理论论证存在一些需要注意的地方。首先，该理论仅表明存在一些向量集无法被编码为增长速度慢于 $\Omega \left( {\epsilon }^{-2}\right)$ 的表示形式；例如，如果实际文档和查询是由某种简单的底层随机过程生成的，那么它们可能更容易被编码。其次，我们的构造通过将所有文档向量按一个常数因子进行重新缩放来实现 $\parallel d\parallel  \leq  1$，但可能存在其他方法在更有效地利用嵌入空间的同时约束范数。第三，在非线性情况下，有可能在不实现低内积失真的情况下消除排序误差。最后，从实际角度来看，当在感兴趣的实际任务上进行评估时，学习到的双编码器所提供的泛化能力可能会超过在保真度上的任何牺牲。由于缺乏解决这些问题的理论工具，我们将在本文的后续章节中进行一组实证研究。但首先，我们将探索对双编码器进行的一种轻量级修改，这种修改能在有限的额外计算成本下提高表达能力。

## 3 Multi-Vector Encodings

## 3 多向量编码

The theoretical analysis suggests that fixed-length vector representations of documents may in general need to be large for long documents, if fidelity with respect to sparse high-dimensional representations is important. Cross-attentional architectures can achieve higher fidelity, but are impractical for large-scale retrieval (Nogueira et al., 2019b; Reimers and Gurevych, 2019; Humeau et al., 2020). We therefore propose a new architecture that represents each document as a fixed-size set of $m$ vectors. Relevance scores are computed as the maximum inner product over this set.

理论分析表明，如果相对于稀疏高维表示的保真度很重要，那么一般来说，长文档的固定长度向量表示可能需要很大。交叉注意力架构可以实现更高的保真度，但对于大规模检索来说并不实用（诺盖拉等人，2019b；赖默斯和古雷维奇，2019；于莫等人，2020）。因此，我们提出了一种新的架构，该架构将每个文档表示为一组固定大小的 $m$ 向量。相关性得分计算为该向量集上的最大内积。

Formally,let $\mathbf{x} = \left( {{x}_{1},\ldots ,{x}_{T}}\right)$ represent a sequence of tokens,with ${x}_{1}$ equal to the special token [CLS], and define y analogously. Then $\left\lbrack  {{h}_{1}\left( \mathrm{x}\right) ,\ldots ,{h}_{T}\left( \mathrm{x}\right) }\right\rbrack$ represents the sequence of contextualized embeddings at the top level of a deep transformer. We define a single-vector representation of the query $\mathrm{x}$ as ${f}^{\left( 1\right) }\left( \mathrm{x}\right)  = {h}_{1}\left( \mathrm{x}\right)$ , and a multi-vector representation of document y as ${f}^{\left( m\right) }\left( \mathrm{y}\right)  = \left\lbrack  {{h}_{1}\left( \mathrm{y}\right) ,\ldots ,{h}_{m}\left( \mathrm{y}\right) }\right\rbrack$ ,the first $m$ representation vectors for the sequence of tokens in $y$ ,with $m < T$ . The relevance score is defined as $\mathop{\max }\limits_{{j = 1\ldots m}}\left\langle  {{f}^{\left( 1\right) }\left( \mathbf{x}\right) ,{f}_{j}^{\left( m\right) }\left( \mathbf{y}\right) }\right\rangle  .$

形式上，设 $\mathbf{x} = \left( {{x}_{1},\ldots ,{x}_{T}}\right)$ 表示一个标记序列，其中 ${x}_{1}$ 等于特殊标记 [CLS]，并类似地定义 y。那么 $\left\lbrack  {{h}_{1}\left( \mathrm{x}\right) ,\ldots ,{h}_{T}\left( \mathrm{x}\right) }\right\rbrack$ 表示深度变换器顶层的上下文嵌入序列。我们将查询 $\mathrm{x}$ 的单向量表示定义为 ${f}^{\left( 1\right) }\left( \mathrm{x}\right)  = {h}_{1}\left( \mathrm{x}\right)$，将文档 y 的多向量表示定义为 ${f}^{\left( m\right) }\left( \mathrm{y}\right)  = \left\lbrack  {{h}_{1}\left( \mathrm{y}\right) ,\ldots ,{h}_{m}\left( \mathrm{y}\right) }\right\rbrack$，即 $y$ 中标记序列的前 $m$ 个表示向量，其中 $m < T$。相关性得分定义为 $\mathop{\max }\limits_{{j = 1\ldots m}}\left\langle  {{f}^{\left( 1\right) }\left( \mathbf{x}\right) ,{f}_{j}^{\left( m\right) }\left( \mathbf{y}\right) }\right\rangle  .$

Although this scoring function is not a dual encoder, the search for the highest-scoring document can be implemented efficiently with standard approximate nearest-neighbor search by adding multiple(m)entries for each document to the search index data structure. If some vector ${f}_{j}^{\left( m\right) }\left( \mathrm{y}\right)$ yields the largest inner product with the query vector ${f}^{\left( 1\right) }\left( \mathrm{x}\right)$ ,it is easy to show the corresponding document must be the one that maximizes the relevance score ${\psi }^{\left( m\right) }\left( {\mathrm{x},\mathrm{y}}\right)$ . The size of the index must grow by a factor of $m$ ,but due to the efficiency of contemporary approximate nearest neighbor and maximum inner product search, the time complexity can be sublinear in the size of the index (Andoni et al., 2019; Guo et al., 2016b). Thus, a model using $m$ vectors of size $k$ to represent documents is more efficient at run-time than a dual encoder that uses a single vector of size ${mk}$ .

尽管这种评分函数不是双编码器，但通过向搜索索引数据结构中为每个文档添加多个（m）条目，可以使用标准的近似最近邻搜索有效地实现对得分最高文档的搜索。如果某个向量 ${f}_{j}^{\left( m\right) }\left( \mathrm{y}\right)$ 与查询向量 ${f}^{\left( 1\right) }\left( \mathrm{x}\right)$ 的内积最大，那么很容易证明对应的文档一定是使相关性得分 ${\psi }^{\left( m\right) }\left( {\mathrm{x},\mathrm{y}}\right)$ 最大化的文档。索引的大小必须增加 $m$ 倍，但由于当代近似最近邻和最大内积搜索的高效性，时间复杂度可以是索引大小的亚线性（安多尼等人，2019；郭等人，2016b）。因此，使用大小为 $k$ 的 $m$ 个向量来表示文档的模型在运行时比使用大小为 ${mk}$ 的单个向量的双编码器更高效。

This efficiency is a key difference from the POLY-ENCODER (Humeau et al., 2020), which computes a fixed number of vectors per query, and aggregates them by softmax attention against document vectors. Yang et al. (2018b) propose a similar architecture for language modeling. Because of the use of softmax in these approaches, it is not possible to decompose the relevance score into a max over inner products, and so fast nearest-neighbor search cannot be applied. In addition, these works did not address retrieval from a large document collection.

这种效率是与POLY - ENCODER（休莫等人，2020年）的关键区别，后者为每个查询计算固定数量的向量，并通过针对文档向量的softmax注意力机制对它们进行聚合。杨等人（2018b）提出了一种类似的语言建模架构。由于这些方法中使用了softmax，无法将相关性得分分解为内积的最大值，因此无法应用快速最近邻搜索。此外，这些研究没有解决从大型文档集合中进行检索的问题。

Analysis. To see why multi-vector encodings can enable smaller encodings per vector, consider an idealized setting in which each document vector is the sum of $m$ orthogonal segments such that $d = \mathop{\sum }\limits_{{i = 1}}^{m}{d}^{\left( i\right) }$ and each query refers to exactly one segment in the gold document. ${}^{3}$ An orthogonal segmentation can be obtained by choosing the segments as a partition of the vocabulary.

分析。为了理解多向量编码为何能使每个向量的编码更小，考虑一个理想化的场景，其中每个文档向量是$m$个正交段的和，使得$d = \mathop{\sum }\limits_{{i = 1}}^{m}{d}^{\left( i\right) }$，并且每个查询恰好指向黄金文档中的一个段。${}^{3}$通过将这些段选择为词汇表的一个划分，可以得到正交分割。

Theorem 1. Define vectors $q,{d}_{1},{d}_{2} \in  {\mathbb{R}}^{v}$ such that $\left\langle  {q,{d}_{1}}\right\rangle   > \left\langle  {q,{d}_{2}}\right\rangle$ ,and assume that both ${d}_{1}$ and ${d}_{2}$ can be decomposed into $m$ segments such that: ${d}_{1} = \mathop{\sum }\limits_{{i = 1}}^{m}{d}_{1}^{\left( i\right) }$ ,and analogously for ${d}_{2}$ ; all segments across both documents are orthogonal. If there exists an $i$ such that $\left\langle  {q,{d}_{1}}\right\rangle   = \left\langle  {q,{d}_{1}^{\left( i\right) }}\right\rangle$ and $\left\langle  {q,{d}_{2}}\right\rangle   \geq  \left\langle  {q,{d}_{2}^{\left( i\right) }}\right\rangle$ ,then $\mu \left( {q,{d}_{1}^{\left( i\right) },{d}_{2}^{\left( i\right) }}\right)  \geq$ $\mu \left( {q,{d}_{1},{d}_{2}}\right)$ . (The proof is in $§{A.5}$ .)

定理1。定义向量$q,{d}_{1},{d}_{2} \in  {\mathbb{R}}^{v}$，使得$\left\langle  {q,{d}_{1}}\right\rangle   > \left\langle  {q,{d}_{2}}\right\rangle$，并假设${d}_{1}$和${d}_{2}$都可以分解为$m$个段，使得：${d}_{1} = \mathop{\sum }\limits_{{i = 1}}^{m}{d}_{1}^{\left( i\right) }$，并且对于${d}_{2}$类似；两个文档的所有段都是正交的。如果存在一个$i$，使得$\left\langle  {q,{d}_{1}}\right\rangle   = \left\langle  {q,{d}_{1}^{\left( i\right) }}\right\rangle$和$\left\langle  {q,{d}_{2}}\right\rangle   \geq  \left\langle  {q,{d}_{2}^{\left( i\right) }}\right\rangle$，那么$\mu \left( {q,{d}_{1}^{\left( i\right) },{d}_{2}^{\left( i\right) }}\right)  \geq$ $\mu \left( {q,{d}_{1},{d}_{2}}\right)$。（证明见$§{A.5}$。）

Remark. The BM25 score can be computed from non-negative representations of the document and query; if the segmentation corresponds to a partition of the vocabulary, then the segments will also be non-negative,and thus the condition $\left\langle  {q,{d}_{2}}\right\rangle   \geq$ $\left\langle  {q,{d}_{2}^{\left( i\right) }}\right\rangle$ holds for all $i$ .

备注。BM25得分可以从文档和查询的非负表示中计算得出；如果分割对应于词汇表的一个划分，那么这些段也将是非负的，因此条件$\left\langle  {q,{d}_{2}}\right\rangle   \geq$ $\left\langle  {q,{d}_{2}^{\left( i\right) }}\right\rangle$对所有$i$都成立。

The relevant case is when the same segment is maximal for both documents, $\left\langle  {q,{d}_{2}^{\left( i\right) }}\right\rangle   =$ $\mathop{\max }\limits_{j}\left\langle  {q,{d}_{2}^{\left( j\right) }}\right\rangle$ ,as will hold for "simple" queries that are well-aligned with the segmentation. Then the normalized margin in the multi-vector model will be at least as large as in the equivalent single vector representation. The relationship to encoding size follows from the theory in the previous section: Theorem 1 implies that if we set ${f}_{i}^{\left( m\right) }\left( \mathrm{y}\right)  = A{d}^{\left( i\right) }$ (for appropriate $A$ ),then an increase in the normalized margin enables the use of a smaller encoding dimension $k$ while still supporting the same pairwise error rate. There are now $m$ times more "documents" to evaluate,but Lemma 2 shows that this exerts only a logarithmic increase on the encoding size for a desired recall@r.But while we hope this argument is illuminating, the assumptions of orthogonal segments and perfect segment match against the query are quite strong. We must therefore rely on empirical analysis to validate the efficacy of multi-vector encoding in realistic applications.

相关的情况是，当同一段对于两个文档都是最大的，即$\left\langle  {q,{d}_{2}^{\left( i\right) }}\right\rangle   =$ $\mathop{\max }\limits_{j}\left\langle  {q,{d}_{2}^{\left( j\right) }}\right\rangle$，对于与分割良好对齐的“简单”查询会出现这种情况。那么多向量模型中的归一化边际至少会和等效的单向量表示中的一样大。与编码大小的关系遵循上一节的理论：定理1表明，如果我们设置${f}_{i}^{\left( m\right) }\left( \mathrm{y}\right)  = A{d}^{\left( i\right) }$（对于合适的$A$），那么归一化边际的增加使得在仍然支持相同的成对错误率的情况下，可以使用更小的编码维度$k$。现在需要评估的“文档”数量增加了$m$倍，但引理2表明，对于期望的召回率@r，这只会使编码大小呈对数增长。虽然我们希望这个论证具有启发性，但正交段和查询与段的完美匹配的假设非常强。因此，我们必须依靠实证分析来验证多向量编码在实际应用中的有效性。

Cross-attention. Cross-attentional architectures can be viewed as a generalization of the multi-vector model: (1) set $m = {T}_{\max }$ (one vector per token); (2) compute one vector per token in the query; (3) allow more expressive aggregation over vectors than the simple max employed above. Any sparse scoring function (e.g., BM25) can be mimicked by a cross-attention model, which need only compute identity between individual words; this can be achieved by random projection word embeddings whose dimension is proportional to the log of the vocabulary size. By definition, the required representation also grows linearly with the number of tokens in the passage and query. As with the POLY-ENCODER, retrieval in the cross-attention model cannot be performed efficiently at scale using fast nearest-neighbor search. In contemporaneous work, Khattab and Zaharia (2020) propose an approach with ${T}_{Y}$ vectors per query and ${T}_{X}$ vectors per document,using a simple sum-of-max for aggregation of the inner products. They apply this approach to retrieval via re-ranking results of ${T}_{Y}$ nearest-neighbor searches. Our multi-vector model uses fixed length representations instead, and a single nearest neighbor search per query.

交叉注意力。交叉注意力架构可以被视为多向量模型的一种推广：（1）设置$m = {T}_{\max }$（每个词元对应一个向量）；（2）为查询中的每个词元计算一个向量；（3）允许对向量进行比上述简单的最大值聚合更具表现力的聚合。任何稀疏评分函数（例如，BM25）都可以由交叉注意力模型模拟，该模型只需计算单个单词之间的一致性；这可以通过随机投影词嵌入来实现，其维度与词汇表大小的对数成正比。根据定义，所需的表示也会随着段落和查询中词元的数量线性增长。与POLY - 编码器一样，使用快速最近邻搜索无法大规模高效地在交叉注意力模型中进行检索。在同期的工作中，Khattab和Zaharia（2020）提出了一种方法，每个查询使用${T}_{Y}$个向量，每个文档使用${T}_{X}$个向量，使用简单的最大和来聚合内积。他们通过对${T}_{Y}$次最近邻搜索的结果进行重排序，将这种方法应用于检索。我们的多向量模型则使用固定长度的表示，并且每个查询只进行一次最近邻搜索。

---

<!-- Footnote -->

${}^{3}$ Here we use(d,q)rather than(x,y)because we describe vector encodings rather than token sequences.

${}^{3}$ 这里我们使用(d,q)而不是(x,y)，因为我们描述的是向量编码而不是词元序列。

<!-- Footnote -->

---

## 4 Experimental Setup

## 4 实验设置

The full IR task requires detection of both precise word overlap and semantic generalization. Our theoretical results focus on the first aspect, and derive theoretical and empirical bounds on the sufficient dimensionality to achieve high fidelity with respect to sparse bag-of-words models as document length grows, for two types of linear random projections. The theoretical setup differs from modeling for realistic information-seeking scenarios in at least two ways.

完整的信息检索（IR）任务需要检测精确的单词重叠和语义泛化。我们的理论结果聚焦于第一个方面，并针对两种类型的线性随机投影，推导出了随着文档长度增加，在稀疏词袋模型方面实现高保真度所需的足够维度的理论和实证界限。该理论设置与现实信息检索场景的建模至少在两个方面有所不同。

First, trained non-linear dual encoders might be able to detect precise word overlap with much lower-dimensional encodings, especially for queries and documents with a natural distribution, which may exhibit a low-dimensional subspace structure. Second, the semantic generalization aspect of the IR task may be more important than the first aspect for practical applications, and our theory does not make predictions about how encoder dimensionality relates to such ability to compute general semantic similarity.

首先，经过训练的非线性双编码器可能能够使用低得多的维度编码来检测精确的单词重叠，特别是对于具有自然分布的查询和文档，这些查询和文档可能呈现出低维子空间结构。其次，对于实际应用而言，信息检索任务的语义泛化方面可能比第一个方面更重要，并且我们的理论并未对编码器维度与计算一般语义相似度的能力之间的关系做出预测。

We relate the theoretical analysis to text retrieval in practice through experimental studies on three tasks. The first task,described in $\$ 5$ ,tests the ability of models to retrieve natural language documents that exactly contain a query and evaluates both BM25 and deep neural dual encoders on a task of detecting precise word overlap, defined over texts with a natural distribution. The second task,described in $§6$ ,is the passage retrieval sub-problem of the open-domain QA version of the Natural Questions (Kwiatkowski et al., 2019; Lee et al., 2019); this benchmark reflects the need to capture graded notions of similarly and has a natural query text distribution. For both of these tasks, we perform controlled experiments varying the maximum length of the documents in the collection, which enables assessing the relationship between encoder dimension and document length.

我们通过对三个任务的实验研究，将理论分析与实际的文本检索联系起来。第一个任务在$\$ 5$中描述，测试模型检索精确包含查询内容的自然语言文档的能力，并在检测精确单词重叠的任务上评估BM25和深度神经双编码器，该任务是在具有自然分布的文本上定义的。第二个任务在$§6$中描述，是自然问题（Natural Questions）开放域问答版本（Kwiatkowski等人，2019；Lee等人，2019）中的段落检索子问题；这个基准反映了捕捉相似性分级概念的需求，并且具有自然的查询文本分布。对于这两个任务，我们进行了控制实验，改变集合中文档的最大长度，这使得我们能够评估编码器维度与文档长度之间的关系。

To evaluate the performance of our best models in comparison to state-of-the-art works on large-scale retrieval and ranking,in $§7$ we report results on a third group of tasks focusing on passage/document ranking: the passage and document-level MS MARCO retrieval datasets (Nguyen et al., 2016; Craswell et al., 2020). Here we follow the standard two-stage retrieval and ranking system: a first-stage retrieval from a large document collection, followed by reranking with a cross-attention model. We focus on the impact of the first-stage retrieval model.

为了评估我们的最佳模型与大规模检索和排序方面的最先进工作相比的性能，在$§7$中，我们报告了第三组聚焦于段落/文档排序的任务的结果：段落和文档级别的MS MARCO检索数据集（Nguyen等人，2016；Craswell等人，2020）。在这里，我们遵循标准的两阶段检索和排序系统：从大型文档集合中进行第一阶段检索，然后使用交叉注意力模型进行重排序。我们关注第一阶段检索模型的影响。

### 4.1 Models

### 4.1 模型

Our experiments compare compressive and sparse dual encoders, cross attention, and hybrid models.

我们的实验比较了压缩和稀疏双编码器、交叉注意力和混合模型。

BM25. We use case-insensitive wordpiece tok-enizations of texts and default BM25 parameters from the gensim library. We apply either unigram (BM25-uni) or combined unigram+bigram representations (BM25-bi).

BM25。我们使用文本的不区分大小写的词块分词，并使用gensim库中的默认BM25参数。我们应用一元（BM25 - uni）或组合的一元 + 二元表示（BM25 - bi）。

Dual encoders from BERT (DE-BERT). We encode queries and documents using BERT-base, which is a pre-trained transformer network (12 layers, 768 dimensions) (Devlin et al., 2019). We implement dual encoders from BERT as a special case of the multi-vector model formalized in § 3, with number of vectors for the document $m = 1$ : the representations for queries and documents are the top layer representations at the [CLS] token. This approach is widely used for retrieval (Lee et al., 2019; Reimers and Gurevych, 2019; Humeau et al.,2020; Xiong et al.,2020). ${}^{4}$ For lower-dimensional encodings, we learn down-projections from $d = {768}$ to $k \in  {32},{64},{128},{512},{}^{5}$ implemented as a single feed-forward layer, followed by layer normalization. All parameters are fine-tuned for the retrieval tasks. We refer to these models as DE-BERT- $k$ .

基于BERT的双编码器（DE - BERT）。我们使用BERT - base对查询和文档进行编码，BERT - base是一个预训练的Transformer网络（12层，768维）（Devlin等人，2019年）。我们将基于BERT的双编码器实现为第3节中形式化的多向量模型的一种特殊情况，文档的向量数量为$m = 1$：查询和文档的表示是[CLS]标记处的顶层表示。这种方法广泛用于信息检索（Lee等人，2019年；Reimers和Gurevych，2019年；Humeau等人，2020年；Xiong等人，2020年）。${}^{4}$对于低维编码，我们学习从$d = {768}$到$k \in  {32},{64},{128},{512},{}^{5}$的降维投影，该投影实现为一个单层前馈网络，随后进行层归一化。所有参数都针对检索任务进行微调。我们将这些模型称为DE - BERT - $k$。

---

<!-- Footnote -->

${}^{4}$ Based on preliminary experiments with pooling strategies we use the [CLS] vectors (without the feed-forward projection learned on the next sentence prediction task).

${}^{4}$基于对池化策略的初步实验，我们使用[CLS]向量（不使用在句子预测任务中学习到的前馈投影）。

${}^{5}$ We experimented with adding a similar layer for $d =$ 768, but this did not offer empirical gains.

${}^{5}$我们尝试为$d =$ 768添加一个类似的层，但这在实验中并未带来收益。

<!-- Footnote -->

---

Cross-Attentional BERT. The most expressive model we consider is cross-attentional BERT, which we implement by applying the BERT encoder to the concatenation of the query and document,with a special [SEP] separator between $\mathrm{x}$ and $y$ . The relevance score is a learned linear function of the encoding of the [CLS] token. Due to the computational cost, cross-attentional BERT is applied only in reranking as in prior work (Nogueira and Cho, 2019; Yang et al., 2019). These models are referred to as Cross-Attention.

交叉注意力BERT。我们考虑的最具表现力的模型是交叉注意力BERT，我们通过将BERT编码器应用于查询和文档的拼接来实现该模型，在$\mathrm{x}$和$y$之间使用特殊的[SEP]分隔符。相关性得分是[CLS]标记编码的一个学习到的线性函数。由于计算成本的原因，与先前的工作一样，交叉注意力BERT仅用于重排序（Nogueira和Cho，2019年；Yang等人，2019年）。这些模型被称为交叉注意力模型。

Multi-Vector Encoding from BERT (ME-BERT). In $§3$ we introduced a model in which every document is represented by exactly $m$ vectors. We use $m = 8$ as a good compromise between cost and accuracy in $§5$ and $\$ 6$ ,and find values of 3 to 4 for $m$ more accurate on the datasets in $§7$ . In addition to using BERT output representations directly, we consider down-projected representations, implemented using a feed-forward layer with dimension ${768} \times  k$ . A model with $k$ -dimensional em-beddings is referred to as ME-BERT- $k$ .

基于BERT的多向量编码（ME - BERT）。在$§3$中，我们引入了一个模型，其中每个文档恰好由$m$个向量表示。在$§5$和$\$ 6$中，我们使用$m = 8$作为成本和准确性之间的一个很好的折衷方案，并发现对于$§7$中的数据集，$m$取值为3到4时更准确。除了直接使用BERT的输出表示外，我们还考虑降维后的表示，通过一个维度为${768} \times  k$的前馈层来实现。一个具有$k$维嵌入的模型被称为ME - BERT - $k$。

Sparse-Dense Hybrids (HYBRID). A natural approach to balancing between the fidelity of sparse representations and the generalization of learned dense ones is to build a hybrid. To do this, we linearly combine a sparse and dense system's scores using a single trainable weight $\lambda$ ,tuned on a development set. For example, a hybrid model of ME-BERT and BM25-uni is referred to as HYBRID-ME-BERT-uni. We implement approximate search to retrieve using a linear combination of two systems by re-ranking $n$ -best top scoring candidates from each system. Prior and concurrent work has also used hybrid sparse-dense models (Guo et al., 2016a; Seo et al., 2019; Karpukhin et al., 2020; Ma et al., 2020; Gao et al., 2020). Our contribution is to assess the impact of sparse-dense hybrids as the document length grows.

稀疏 - 密集混合模型（HYBRID）。在稀疏表示的保真度和学习到的密集表示的泛化能力之间进行平衡的一种自然方法是构建混合模型。为此，我们使用一个可训练的权重$\lambda$对稀疏系统和密集系统的得分进行线性组合，该权重在开发集上进行调整。例如，ME - BERT和BM25 - uni的混合模型被称为HYBRID - ME - BERT - uni。我们通过对每个系统的$n$个得分最高的候选结果进行重排序，实现了使用两个系统的线性组合进行近似搜索和检索。先前和同期的工作也使用了稀疏 - 密集混合模型（Guo等人，2016a；Seo等人，2019年；Karpukhin等人，2020年；Ma等人，2020年；Gao等人，2020年）。我们的贡献是评估随着文档长度的增加，稀疏 - 密集混合模型的影响。

### 4.2 Learning and Inference

### 4.2 学习与推理

For the experiments in $\$ 5$ and $\$ 6$ ,all trained models are initialized from BERT-base, and all parameters are fine-tuned using a cross-entropy loss with 7 sampled negatives from a pre-computed 200-document list and additional in-batch negatives (with a total number of 1024 candidates in a batch); the pre-computed candidates include 100 top neighbors from BM25 and 100 random samples. This is similar to the method by Lee et al. (2019), but with additional fixed candidates, also used in concurrent work (Karpukhin et al., 2020). Given a model trained in this way, for the scalable methods, we also applied hard-negative mining as in Gillick et al. (2019) and used one iteration when beneficial. More sophisticated negative selection is proposed in concurrent work (Xiong et al., 2020). For retrieval from large document collections with the scalable models, we used ScaNN: an efficient approximate nearest neighbor search library (Guo et al., 2020); in most experiments, we use exact search settings but also evaluate approximate search in Section § 7. In $§7$ ,the same general approach with slightly different hyperpa-rameters (detailed in that section) was used, to enable more direct comparisons to prior work.

对于$\$ 5$和$\$ 6$中的实验，所有训练好的模型均从BERT-base初始化，并且使用交叉熵损失对所有参数进行微调，从预先计算的200篇文档列表中采样7个负样本，并使用额外的批次内负样本（一批中总共有1024个候选样本）；预先计算的候选样本包括来自BM25的100个最邻近样本和100个随机样本。这与Lee等人（2019）的方法类似，但增加了额外的固定候选样本，并发工作中也采用了这种方法（Karpukhin等人，2020）。对于以这种方式训练的模型，对于可扩展方法，我们还采用了Gillick等人（2019）中的难负样本挖掘方法，并在有益时使用一次迭代。并发工作中提出了更复杂的负样本选择方法（Xiong等人，2020）。对于使用可扩展模型从大型文档集合中进行检索，我们使用了ScaNN：一种高效的近似最近邻搜索库（Guo等人，2020）；在大多数实验中，我们使用精确搜索设置，但也在第7节评估了近似搜索。在$§7$中，使用了相同的通用方法，但超参数略有不同（该节中有详细说明），以便与先前的工作进行更直接的比较。

## 5 Containing Passage ICT Task

## 5 包含段落的逆完形填空任务

We begin with experiments on the task of retrieving a Wikipedia passage $y$ containing a sequence of words $x$ . We create a dataset using Wikipedia, following the Inverse Cloze Task definition by Lee et al. (2019), but adapted to suit the goals of our study. The task is defined by first breaking Wikipedia texts into segments of length at most $l$ . These form the document collection $\mathcal{D}$ . Queries ${x}_{i}$ are generated by sampling sub-sequences from the documents ${y}_{i}$ . We use queries of lengths between 5 and 25,and do not remove the queries ${x}_{i}$ from their corresponding documents ${y}_{i}$ .

我们首先进行关于检索包含单词序列$x$的维基百科段落$y$的任务实验。我们按照Lee等人（2019）定义的逆完形填空任务，使用维基百科创建了一个数据集，但进行了调整以适应我们的研究目标。该任务的定义是先将维基百科文本分割成长度最多为$l$的片段。这些片段构成文档集合$\mathcal{D}$。查询${x}_{i}$是通过从文档${y}_{i}$中采样子序列生成的。我们使用长度在5到25之间的查询，并且不将查询${x}_{i}$从其对应的文档${y}_{i}$中移除。

We create a dataset with one million queries and evaluate retrieval against four document collections ${\mathcal{D}}_{l}$ ,for $l \in  {50},{100},{200},{400}$ . Each ${\mathcal{D}}_{l}$ contains three million documents of maximum length $l$ tokens. In addition to original Wikipedia passages,each ${\mathcal{D}}_{l}$ contains synthetic distractor documents, which contain the large majority of words in $x$ but differ by one or two tokens. $5\mathrm{\;K}$ queries are used for evaluation, leaving the rest for training and validation. Although checking containment is a straightforward machine learning task, it is a good testbed for assessing the fidelity of compressive neural models. BM25-bi achieves over 95 MRR@10 across collections for this task.

我们创建了一个包含一百万个查询的数据集，并针对四个文档集合${\mathcal{D}}_{l}$进行检索评估，其中$l \in  {50},{100},{200},{400}$。每个${\mathcal{D}}_{l}$包含三百万个最大长度为$l$个词元的文档。除了原始的维基百科段落外，每个${\mathcal{D}}_{l}$还包含合成干扰文档，这些文档包含$x$中的大部分单词，但有一两个词元不同。$5\mathrm{\;K}$个查询用于评估，其余的用于训练和验证。尽管检查包含关系是一个直接的机器学习任务，但它是评估压缩神经模型保真度的良好测试平台。BM25-bi在该任务的各个集合上的MRR@10超过95。

Figure 4 (left) shows test set results on rerank-ing, where models need to select one of 200 passages (top 100 BM25-bi and 100 random candidates). It is interesting to see how strong the sparse models are relative to even a 768-dimensional DE-BERT. As the document length increases, the performance of both the sparse and dense dual encoders worsens; the accuracy of the DE-BERT models falls most rapidly, widening the gap to BM25.

图4（左）显示了重排序的测试集结果，其中模型需要从200个段落（前100个BM25-bi候选和100个随机候选）中选择一个。有趣的是，即使与768维的DE - BERT相比，稀疏模型的表现也很强。随着文档长度的增加，稀疏和密集双编码器的性能都会变差；DE - BERT模型的准确率下降最快，与BM25的差距扩大。

<!-- Media -->

<!-- figureText: Passage Ranking for ICT Cross-Attention Passage Retrieval for ICT DE-BERT-32 DE-BERT-64 DE-BERT-128 DE-BERT-512 DE-BERT-768 ME-BERT-64 ME-BERT-768 HYBRID-ME-BERT-uni HYBRID-ME-BERT-bi 100 200 400 BM25-uni BM25-bi 100 100 MRR@10 80 MRR@10 80 60 40 60 40 50 100 200 400 50 -->

<img src="https://cdn.noedgeai.com/0195aec6-1905-71b4-8b6e-19da2d5cc901_8.jpg?x=196&y=173&w=1347&h=317&r=0"/>

Figure 4: Results on the containing passage ICT task as maximum passage length varies (50 to 400 tokens). Left: Reranking 200 candidates; Right: Retrieval from three million candidates. Exact numbers refer to Table 3.

图4：包含段落的逆完形填空任务结果随最大段落长度变化（50到400个词元）。左：对200个候选进行重排序；右：从三百万个候选中进行检索。具体数字参考表3。

<!-- Media -->

Full cross attention is nearly perfect and does not degrade with document length. ME-BERT-768 which uses 8 vectors of dimension 768 to represent documents strongly outperforms the best DE-BERT model. Even ME-BERT-64, which uses 8 vectors of size only 64 instead (thus requiring the same document collection size as DE-BERT-512 and being faster at inference time), outperforms the DE-BERT models by a large margin.

全交叉注意力几乎是完美的，并且不会随文档长度而下降。使用8个768维向量表示文档的ME - BERT - 768明显优于最佳的DE - BERT模型。即使是ME - BERT - 64，它仅使用8个64维向量（因此所需的文档集合大小与DE - BERT - 512相同，并且推理速度更快），也大幅优于DE - BERT模型。

Figure 4 (right) shows results for the much more challenging task of retrieval from three million candidates. For the latter setting, we only evaluate models that can efficiently retrieve nearest neighbors from such a large set. We see similar behavior to the reranking setting, with the multi-vector methods exceeding BM25-uni performance for all lengths and DE-BERT models under-performing BM25-uni. The hybrid model outperforms both components in the combination with largest improvements over ME-BERT for the longest-document collection.

图4（右）显示了从三百万个候选中进行检索这一更具挑战性任务的结果。对于后一种设置，我们只评估能够从如此大的集合中有效检索最近邻的模型。我们看到与重排序设置类似的情况，多向量方法在所有长度上都超过了BM25 - uni的性能，而DE - BERT模型的表现不如BM25 - uni。混合模型在组合中优于两个组件，在最长文档集合上相对于ME - BERT有最大的改进。

## 6 Retrieval for Open-domain QA

## 6 开放域问答的检索

For this task we similarly use English Wikipedia ${}^{6}$ as four different document collections, of maximum passage length $l \in  \{ {50},{100},{200},{400}\}$ , and corresponding approximate sizes of 39 million, 27.3 million, 16.1 million, and 10.2 million documents, respectively. Here we use real user queries contained in the Natural Questions dataset (Kwiatkowski et al., 2019). We follow the setup in Lee et al. (2019). There are ${87},{925}\mathrm{{QA}}$ pairs in training and 3,610 QA pairs in the test set. We hold out a subset of training for development.

对于此任务，我们同样使用英文维基百科 ${}^{6}$ 作为四个不同的文档集合，最大段落长度为 $l \in  \{ {50},{100},{200},{400}\}$，对应的文档大致数量分别为 3900 万、2730 万、1610 万和 1020 万。这里我们使用自然问题数据集（Kwiatkowski 等人，2019 年）中包含的真实用户查询。我们遵循 Lee 等人（2019 年）的设置。训练集中有 ${87},{925}\mathrm{{QA}}$ 对问答对，测试集中有 3610 对问答对。我们留出一部分训练数据用于开发。

For document retrieval, a passage is correct for a query $x$ if it contains a string that matches exactly an annotator-provided short answer for the question. We form a reranking task by considering the top 100 results from BM25-uni and 100 random samples, and also consider the full retrieval setting. BM25-uni is used here instead of BM25-bi, because it is the stronger model for this task.

对于文档检索，如果一个段落包含与注释者提供的问题简短答案完全匹配的字符串，那么该段落对于查询 $x$ 就是正确的。我们通过考虑 BM25-uni 的前 100 个结果和 100 个随机样本形成一个重排序任务，同时也考虑全量检索设置。这里使用 BM25-uni 而非 BM25-bi，因为它在这个任务中是更强的模型。

Our theoretical results do not make direct predictions for performance of compressive dual encoder models relative to BM25 on this task. They do tell us that as the document length grows, low-dimensional compressive dual encoders may not be able to measure weighted term overlap precisely, potentially leading to lower performance on the task. Therefore, we would expect that higher dimensional dual encoders, multi-vector encoders, and hybrid models become more useful for collections with longer documents.

我们的理论结果并未直接预测压缩双编码器模型相对于 BM25 在该任务上的性能。但它们确实告诉我们，随着文档长度的增加，低维压缩双编码器可能无法精确测量加权词项重叠，这可能导致该任务的性能下降。因此，我们预计高维双编码器、多向量编码器和混合模型对于较长文档的集合会更有用。

Figure 5 (left) shows heldout set results on the reranking task. To fairly compare systems that operate over collections of different-sized passages, we allow each model to select approximately the same number of tokens (400) and evaluate on whether an answer is contained in them. For example,models retrieving from ${\mathcal{D}}_{50}$ return their top 8 passages,and ones retrieving from ${\mathcal{D}}_{100}$ retrieve top 4. The figure shows this recall@400 tokens across models. The relative performance of BM25-uni and DE-BERT is different from that seen in the ICT task, due to the semantic generalizations needed. Nevertheless, higher-dimensional DE-BERT models generally perform better, and multi-vector models provide further benefits, especially for longer-document collections; ME-BERT-768 outperforms DE-BERT-768 and ME-BERT- 64 outperforms DE-BERT-512; Cross-Attention is still substantially stronger.

图 5（左）展示了重排序任务的预留集结果。为了公平比较在不同大小段落集合上运行的系统，我们允许每个模型选择大致相同数量的词元（400 个），并评估其中是否包含答案。例如，从 ${\mathcal{D}}_{50}$ 中检索的模型返回其前 8 个段落，从 ${\mathcal{D}}_{100}$ 中检索的模型检索前 4 个段落。该图展示了各模型在 400 个词元下的召回率。由于需要进行语义泛化，BM25-uni 和 DE-BERT 的相对性能与在 ICT 任务中看到的不同。尽管如此，高维 DE-BERT 模型通常表现更好，多向量模型提供了进一步的优势，特别是对于较长文档的集合；ME-BERT-768 优于 DE-BERT-768，ME-BERT-64 优于 DE-BERT-512；交叉注意力模型仍然明显更强。

---

<!-- Footnote -->

6https://archive.org/download/ enwiki-20181220

6https://archive.org/download/ enwiki-20181220

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: Natural Questions Reranking Natural Questions Retrieval Cross-Attention DE-BERT-32 DE-BERT-64 DE-BERT-128 DE-BERT-512 DE-BERT-768 ME-BERT-64 ME-BERT-768 HYBRID-ME-BERT-uni 100 200 400 BM25-uni recall@400 tokens 50 recall@400 tokens 50 40 30 40 30 50 100 200 400 50 -->

<img src="https://cdn.noedgeai.com/0195aec6-1905-71b4-8b6e-19da2d5cc901_9.jpg?x=198&y=170&w=1255&h=282&r=0"/>

Figure 5: Results on NQ passage recall as maximum passage length varies (50 to 400 tokens). Left: Reranking of 200 passages; Right: Open domain retrieval result on all of (English) Wikipedia. Exact numbers refer to Table 3.

图 5：随着最大段落长度变化（50 到 400 个词元），自然问题（NQ）段落召回率的结果。左：200 个段落的重排序；右：在整个（英文）维基百科上的开放域检索结果。具体数字参考表 3。

<!-- Media -->

Figure 5 (right) shows heldout set results for the task of retrieving from Wikipedia for each of the four document collections ${\mathcal{D}}_{l}$ . Unlike the reranking setting, only higher-dimensional DE-BERT models outperform BM25 for passages longer than 50 . The hybrid models offer large improvements over their components, capturing both precise word overlap and semantic similarity. The gain from adding BM25 to ME-BERT and DE-BERT increases as the length of the documents in the collection grows, which is consistent with our expectations based on the theory.

图 5（右）展示了从四个文档集合 ${\mathcal{D}}_{l}$ 中的每一个在维基百科上进行检索任务的预留集结果。与重排序设置不同，对于长度超过 50 个词元的段落，只有高维 DE-BERT 模型的性能优于 BM25。混合模型相对于其组件有很大的改进，既能捕捉精确的词重叠，又能捕捉语义相似性。随着集合中文档长度的增加，在 ME-BERT 和 DE-BERT 中加入 BM25 带来的增益也会增加，这与我们基于理论的预期一致。

## 7 Large-Scale Supervised IR

## 7 大规模监督信息检索

The previous experimental sections focused on understanding the relationship between compressive encoder representation dimensionality and document length. Here we evaluate whether our newly proposed multi-vector retrieval model ME-BERT, its corresponding dual encoder baseline DE-BERT, and sparse-dense hybrids compare favorably to state-of-the-art models for large-scale supervised retrieval and ranking on IR benchmarks.

之前的实验部分着重于理解压缩编码器表示维度与文档长度之间的关系。在此，我们评估我们新提出的多向量检索模型ME - BERT（多向量双向编码器表示变换器）、其对应的双编码器基线模型DE - BERT（双编码器双向编码器表示变换器）以及稀疏 - 密集混合模型，在信息检索（IR）基准测试的大规模有监督检索和排序任务中，是否优于现有最先进的模型。

Datasets. The MS MARCO passage ranking task focuses on ranking passages from a collection of about ${8.8}\mathrm{\;{mln}}$ . About ${532}\mathrm{k}$ queries paired with relevant passages are provided for training. The MS MARCO document ranking task is on ranking full documents instead. The full collection contains about 3 million documents and the training set has about 367 thousand queries. We report results on the passage and document development sets, comprising 6,980 and 5,193 queries, respectively in Table 1. We report MS MARCO and TREC DL 2019 (Craswell et al., 2020) test results in Table 2.

数据集。MS MARCO段落排序任务的重点是对约${8.8}\mathrm{\;{mln}}$个段落集合中的段落进行排序。训练时提供了约${532}\mathrm{k}$个与相关段落配对的查询。而MS MARCO文档排序任务则是对完整文档进行排序。完整的文档集合包含约300万篇文档，训练集有大约36.7万个查询。我们在表1中报告了段落和文档开发集的结果，分别包含6980个和5193个查询。我们在表2中报告了MS MARCO和TREC DL 2019（克拉斯韦尔等人，2020年）的测试结果。

Model Settings. For MS MARCO passage we apply models on the provided passage collections.

模型设置。对于MS MARCO段落，我们将模型应用于所提供的段落集合。

<!-- Media -->

<table><tr><td rowspan="2"/><td rowspan="2">Model</td><td>MS-Passage</td><td>MS-Doc</td></tr><tr><td>MRR</td><td>MRR</td></tr><tr><td rowspan="12">Retrieval</td><td>BM25</td><td>0.167</td><td>0.249</td></tr><tr><td>BM25-E</td><td>0.184</td><td>0.209</td></tr><tr><td>Doc2Query</td><td>0.215</td><td>-</td></tr><tr><td>DOCT5QUERY</td><td>0.278</td><td>-</td></tr><tr><td>DEEPCT</td><td>0.243</td><td>-</td></tr><tr><td>HDCT</td><td>-</td><td>0.300</td></tr><tr><td>DE-BERT</td><td>0.302</td><td>0.288</td></tr><tr><td>ME-BERT</td><td>0.334</td><td>0.333</td></tr><tr><td>DE-HYBRID</td><td>0.304</td><td>0.313</td></tr><tr><td>DE-HYBRID-E</td><td>0.309</td><td>0.315</td></tr><tr><td>ME-HYBRID</td><td>0.338</td><td>0.346</td></tr><tr><td>ME-HYBRID-E</td><td>0.343</td><td>0.339</td></tr><tr><td rowspan="6">Reranking</td><td>MULTI-STAGE</td><td>0.390</td><td>-</td></tr><tr><td>IDST</td><td>0.408</td><td>-</td></tr><tr><td>Leaderboard</td><td>0.439</td><td>-</td></tr><tr><td>DE-BERT</td><td>0.391</td><td>0.339</td></tr><tr><td>ME-BERT</td><td>0.395</td><td>0.353</td></tr><tr><td>ME-HYBRID</td><td>0.394</td><td>0.353</td></tr></table>

<table><tbody><tr><td rowspan="2"></td><td rowspan="2">模型</td><td>微软段落（MS-Passage）</td><td>微软文档（MS-Doc）</td></tr><tr><td>平均倒数排名（MRR）</td><td>平均倒数排名（MRR）</td></tr><tr><td rowspan="12">检索</td><td>二元独立模型改进算法（BM25）</td><td>0.167</td><td>0.249</td></tr><tr><td>改进的二元独立模型算法（BM25-E）</td><td>0.184</td><td>0.209</td></tr><tr><td>文档转查询（Doc2Query）</td><td>0.215</td><td>-</td></tr><tr><td>基于T5的文档转查询（DOCT5QUERY）</td><td>0.278</td><td>-</td></tr><tr><td>深度上下文词项加权（DEEPCT）</td><td>0.243</td><td>-</td></tr><tr><td>分层深度上下文词项加权（HDCT）</td><td>-</td><td>0.300</td></tr><tr><td>深度编码器BERT（DE-BERT）</td><td>0.302</td><td>0.288</td></tr><tr><td>多编码器BERT（ME-BERT）</td><td>0.334</td><td>0.333</td></tr><tr><td>深度混合模型（DE-HYBRID）</td><td>0.304</td><td>0.313</td></tr><tr><td>改进的深度混合模型（DE-HYBRID-E）</td><td>0.309</td><td>0.315</td></tr><tr><td>多编码器混合模型（ME-HYBRID）</td><td>0.338</td><td>0.346</td></tr><tr><td>改进的多编码器混合模型（ME-HYBRID-E）</td><td>0.343</td><td>0.339</td></tr><tr><td rowspan="6">重排序</td><td>多阶段（MULTI-STAGE）</td><td>0.390</td><td>-</td></tr><tr><td>IDST（未明确通用译法，保留原词）</td><td>0.408</td><td>-</td></tr><tr><td>排行榜</td><td>0.439</td><td>-</td></tr><tr><td>深度编码器BERT（DE-BERT）</td><td>0.391</td><td>0.339</td></tr><tr><td>多编码器BERT（ME-BERT）</td><td>0.395</td><td>0.353</td></tr><tr><td>多编码器混合模型（ME-HYBRID）</td><td>0.394</td><td>0.353</td></tr></tbody></table>

Table 1: Development set results on MS MARCO-Passage (MS-Passage), MS MARCO-Document (MS-Doc) showing MRR@10.

表1：MS MARCO段落（MS-Passage）、MS MARCO文档（MS-Doc）开发集结果，展示了MRR@10指标。

<!-- Media -->

For MS MARCO document, we follow Yan et al. (2020) and break documents into a set of overlapping passages with length up to 482 tokens, each including the document URL and title. For each task, we train the models on that task's training data only. We initialize the retriever and reranker models with BERT-large. We train dense retrieval models on positive and negative candidates from the 1000-best list of BM25, additionally using one iteration of hard negative mining when beneficial. For ME-BERT,we used $m = 3$ for the passage and $m = 4$ for the document task.

对于MS MARCO文档，我们遵循Yan等人（2020年）的方法，将文档拆分为一组长度最多为482个词元的重叠段落，每个段落都包含文档的URL和标题。对于每个任务，我们仅使用该任务的训练数据来训练模型。我们使用BERT-large初始化检索器和重排器模型。我们在BM25的前1000个最佳候选列表中的正、负候选样本上训练密集检索模型，在有益的情况下额外使用一轮难负样本挖掘。对于ME-BERT，我们在段落任务中使用$m = 3$，在文档任务中使用$m = 4$。

Results. Table 1 comparatively evaluates our models on the dev sets of two tasks. The state of the art prior work follows the two-stage retrieval and reranking approach, where an efficient first-stage system retrieves a (usually large) list of candidates from the document collection, and a second stage more expensive model such as cross-attention BERT reranks the candidates.

结果。表1对我们的模型在两个任务的开发集上进行了对比评估。现有最先进的先前工作采用两阶段检索和重排方法，即高效的第一阶段系统从文档集合中检索出一个（通常数量较大的）候选列表，然后第二阶段使用更复杂的模型（如交叉注意力BERT）对候选样本进行重排。

Our focus is on improving the first stage, and we compare to prior works in two settings: Retrieval, top part of Table 1, where only first-stage efficient retrieval systems are used and Reranking, bottom part of the table, where more expensive second-stage models are employed to re-rank candidates. Figure 6 delves into the impact of the first-stage retrieval systems as the number of candidates the second stage reranker has access to is substantially reduced, improving efficiency.

我们的重点是改进第一阶段，我们在两种设置下与先前的工作进行比较：检索（表1的上半部分），仅使用第一阶段的高效检索系统；重排（表的下半部分），使用更复杂的第二阶段模型对候选样本进行重排。图6深入探讨了第一阶段检索系统的影响，随着第二阶段重排器可访问的候选样本数量大幅减少，效率得到了提高。

<!-- Media -->

<!-- figureText: MS MARCO passage MS MARCO document BM25-uni Deep-CT DE-BERT ME-BERT Hybrid-ME-BERT-uni 20 50 100 200 1000 Retrieval depth 40 36 MRR@10 35 MRR@10 34 32 30 30 25 10 20 50 100 200 1000 10 Retrieval depth -->

<img src="https://cdn.noedgeai.com/0195aec6-1905-71b4-8b6e-19da2d5cc901_10.jpg?x=196&y=173&w=1263&h=319&r=0"/>

Figure 6: MRR@10 when reranking at different retrieval depth (10 to 1000 candidates) for MS MARCO.

图6：MS MARCO在不同检索深度（10到1000个候选样本）下重排时的MRR@10指标。

<table><tr><td>Model</td><td>MRR(MS)</td><td>RR</td><td>NDCG@10</td><td>Holes@10</td></tr><tr><td colspan="5">Passage Retrieval</td></tr><tr><td>BM25-Anserini</td><td>0.186</td><td>0.825</td><td>0.506</td><td>0.000</td></tr><tr><td>DE-BERT</td><td>0.295</td><td>0.936</td><td>0.639</td><td>0.165</td></tr><tr><td>ME-BERT</td><td>0.323</td><td>0.968</td><td>0.687</td><td>0.109</td></tr><tr><td>DE-HYBRID-E</td><td>0.306</td><td>0.951</td><td>0.659</td><td>0.105</td></tr><tr><td>ME-HYBRID-E</td><td>0.336</td><td>0.977</td><td>0.706</td><td>0.051</td></tr></table>

<table><tbody><tr><td>模型</td><td>平均倒数排名（Mean Reciprocal Rank，MRR(MS)）</td><td>召回率（Recall Rate，RR）</td><td>归一化折损累积增益@10（Normalized Discounted Cumulative Gain@10，NDCG@10）</td><td>漏洞率@10（Holes@10）</td></tr><tr><td colspan="5">段落检索</td></tr><tr><td>BM25 - 安瑟里尼（BM25 - Anserini）</td><td>0.186</td><td>0.825</td><td>0.506</td><td>0.000</td></tr><tr><td>DE - 伯特模型（DE - BERT）</td><td>0.295</td><td>0.936</td><td>0.639</td><td>0.165</td></tr><tr><td>ME - 伯特模型（ME - BERT）</td><td>0.323</td><td>0.968</td><td>0.687</td><td>0.109</td></tr><tr><td>DE - 混合 - E（DE - HYBRID - E）</td><td>0.306</td><td>0.951</td><td>0.659</td><td>0.105</td></tr><tr><td>ME - 混合 - E（ME - HYBRID - E）</td><td>0.336</td><td>0.977</td><td>0.706</td><td>0.051</td></tr></tbody></table>

<table><tr><td colspan="5">Document Retrieval</td></tr><tr><td>Base-Indri</td><td>0.192</td><td>0.785</td><td>0.517</td><td>0.002</td></tr><tr><td>DE-BERT</td><td>-</td><td>0.841</td><td>0.510</td><td>0.188</td></tr><tr><td>ME-BERT</td><td>-</td><td>0.877</td><td>0.588</td><td>0.109</td></tr><tr><td>DE-HYBRID-E</td><td>0.287</td><td>0.890</td><td>0.595</td><td>0.084</td></tr><tr><td>ME-HYBRID-E</td><td>0.310</td><td>0.914</td><td>0.610</td><td>0.063</td></tr></table>

<table><tbody><tr><td colspan="5">文档检索</td></tr><tr><td>基础Indri模型</td><td>0.192</td><td>0.785</td><td>0.517</td><td>0.002</td></tr><tr><td>DE - BERT模型（DE - BERT）</td><td>-</td><td>0.841</td><td>0.510</td><td>0.188</td></tr><tr><td>ME - BERT模型（ME - BERT）</td><td>-</td><td>0.877</td><td>0.588</td><td>0.109</td></tr><tr><td>DE - 混合 - E模型（DE - HYBRID - E）</td><td>0.287</td><td>0.890</td><td>0.595</td><td>0.084</td></tr><tr><td>ME - 混合 - E模型（ME - HYBRID - E）</td><td>0.310</td><td>0.914</td><td>0.610</td><td>0.063</td></tr></tbody></table>

Table 2: Test set first-pass retrieval results on the passage and document TREC 2019 DL evaluation as well as MS MARCO eval MRR@10 (passage) and MRR@100 (document) under MRR(MS).

表2：在段落和文档TREC 2019 DL评估以及MS MARCO评估的MRR@10（段落）和MRR@100（文档）下，基于平均倒数排名（MS）的测试集首轮检索结果。

<!-- Media -->

We report results in comparison to the following systems: 1) MULTI-STAGE (Nogueira and Lin, 2019), which reranks BM25 candidates with a cascade of BERT models. 2) DOC2QUERY (Nogueira et al., 2019b) and DOCT5QUERY (Nogueira and Lin, 2019), which use neural models to expand documents before indexing and scoring with sparse retrieval models. 3) DEEPCT (Dai and Callan, 2020b), which learns to map BERT's con-textualized text representations to context-aware term weights. 4) HDCT (Dai and Callan, 2020a) uses a hierachical approach that combines passage-level term weights into document level term weights. 5) IDST, a two-stage cascade ranking pipeline by Yan et al. (2020), and 6) Leader-board, which is the best score on the MS MARCO-passage leaderboard as of Sept 18, ${2020}{.}^{7}$

我们报告与以下系统对比的结果：1) 多阶段系统（MULTI - STAGE，诺盖拉（Nogueira）和林（Lin），2019年），该系统使用一系列BERT模型对BM25候选结果进行重排序。2) DOC2QUERY（诺盖拉等人，2019b）和DOCT5QUERY（诺盖拉和林，2019年），这两个系统在使用稀疏检索模型进行索引和评分之前，使用神经模型对文档进行扩展。3) 深度上下文词项加权模型（DEEPCT，戴（Dai）和卡兰（Callan），2020b），该模型学习将BERT的上下文文本表示映射到上下文感知的词项权重。4) 分层深度上下文词项加权模型（HDCT，戴和卡兰，2020a）采用分层方法，将段落级别的词项权重合并为文档级别的词项权重。5) 两阶段级联排序管道（IDST），由严（Yan）等人（2020年）提出，以及6) 排行榜（Leader - board），截至${2020}{.}^{7}$ 9月18日，它是MS MARCO段落排行榜上的最佳得分。

We also compare our models to both our own BM25 implementation described in $§{4.1}$ ,and external publicly available sparse model implementations, denoted with BM25-E. For the passage task, BM25-E is the Anserini (Yang et al., 2018a) system with default parameters. For the document task, BM25-E is the official IndriQueryLikelihood baseline. We report on dense-sparse hybrids using both our own BM25, and the external sparse systems; the latter hybrids are indicated by a suffix -E.

我们还将我们的模型与$§{4.1}$中描述的我们自己实现的BM25模型，以及外部公开可用的稀疏模型实现（表示为BM25 - E）进行比较。对于段落任务，BM25 - E是使用默认参数的安瑟里尼（Anserini，杨（Yang）等人，2018a）系统。对于文档任务，BM25 - E是官方的Indri查询似然基线。我们报告了使用我们自己的BM25模型和外部稀疏系统的密集 - 稀疏混合模型的结果；后者的混合模型用后缀 - E表示。

Looking at the top part of Table 1, we can see that our DE-BERT model already outperforms or is competitive with prior systems. The multi-vector model brings larger improvement on the dataset containing longer documents (MS MARCO document), and the sparse-dense hybrid models bring improvements over dense-only models on both datasets. According to a Wilcoxon signed rank test for statistical significance, all differences between DE-BERT, ME-BERT, DE-HYBRID-E, and ME-HYBRID-E are statistically significant on both development sets with $p$ -value $< {.0001}$ .

查看表1的上半部分，我们可以看到我们的DE - BERT模型已经优于或与先前的系统具有竞争力。多向量模型在包含较长文档的数据集（MS MARCO文档）上带来了更大的改进，而稀疏 - 密集混合模型在两个数据集上都比仅使用密集模型有所改进。根据用于统计显著性的威尔科克森符号秩检验，DE - BERT、ME - BERT、DE - HYBRID - E和ME - HYBRID - E之间的所有差异在两个开发集上的$p$值为$< {.0001}$时都具有统计学意义。

When a large number of candidates can be reranked, the impact of the first-stage system decreases. In the bottom part of the table we see that our models are comparable to systems reranking BM25 candidates.The accuracy of the first-stage system is particularly important when the cost of reranking a large set of candidates is prohibitive. Figure 6 shows the performance of systems that rerank a smaller number of candidates. We see that, when a very small number of candidates can be scored with expensive cross-attention models, the multi-vector ME-BERT and hybrid models achieve large improvements compared to prior systems on both MS MARCO tasks.

当可以对大量候选结果进行重排序时，第一阶段系统的影响会减小。在表的下半部分，我们看到我们的模型与对BM25候选结果进行重排序的系统相当。当对大量候选结果进行重排序的成本过高时，第一阶段系统的准确性尤为重要。图6显示了对较少数量候选结果进行重排序的系统的性能。我们看到，当只能使用昂贵的交叉注意力模型对非常少的候选结果进行评分时，多向量ME - BERT模型和混合模型在两个MS MARCO任务上与先前的系统相比都取得了显著改进。

---

<!-- Footnote -->

${}^{7}$ https://microsoft.github.io/ msmarco/

${}^{7}$ https://microsoft.github.io/ msmarco/

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: 0.34 ME-BERT DE-BERT 1.5 2 2.5 Running time (milliseconds/query) 0.32 0.3 0.28 0 0.5 1 -->

<img src="https://cdn.noedgeai.com/0195aec6-1905-71b4-8b6e-19da2d5cc901_11.jpg?x=202&y=172&w=516&h=246&r=0"/>

Figure 7: Quality/running time tradeoff for DE-BERT and ME-BERT on the MS MARCO passage dev set. Dashed lines show quality with exact search.

图7：DE - BERT和ME - BERT在MS MARCO段落开发集上的质量/运行时间权衡。虚线表示精确搜索的质量。

<!-- Media -->

Table 2 shows test results for dense models, external sparse model baselines, and hybrids of the two (without reranking). In addition to test set (eval) results on the MS MARCO passage task, we report metrics on the manually annotated passage and document retrieval test set at TREC DL 2019. We report the fraction of unrated items as Holes @10 following Xiong et al. (2020).

表2显示了密集模型、外部稀疏模型基线以及两者的混合模型（不进行重排序）的测试结果。除了MS MARCO段落任务的测试集（评估）结果外，我们还报告了2019年TREC DL手动标注的段落和文档检索测试集的指标。我们按照熊（Xiong）等人（2020年）的方法，报告未评级项目的比例，记为Holes @10。

Time and space analysis Figure 7 compares the running time/quality trade-off curves for DE-BERT and ME-BERT on the MS MARCO passage task using the ScaNN (Guo et al., 2020) library on a 160 Intel(R) Xeon(R) CPU @ 2.20GHz cores machine with 1.88TB memory. Both models use one vector of size $k = {1024}$ per query; DE-BERT uses one and ME-BERT uses 3 vectors of size $k = {1024}$ per document. The size of the document index for DE-BERT is ${34.2}\mathrm{{GB}}$ and the size of the index for ME-BERT is about 3 times larger. The indexing time was ${1.52}\mathrm{\;h}$ and ${3.02}\mathrm{\;h}$ for DE-BERT and ME-BERT, respectively. The ScaNN configuration we use is num_leaves=5000, and num_leaves_to_search ranges from 25 to 2000 (from less to more exact search) and time per query is measured when using parallel inference on all 160 cores. In the higher quality range of the curves, ME-BERT achieves substantially higher MRR than DE-BERT for the same inference time per query.

时空分析 图7展示了在MS MARCO段落任务中，DE - BERT和ME - BERT在一台配备1.88TB内存、拥有160个主频为2.20GHz的英特尔（Intel®）至强（Xeon®）CPU核心的机器上，使用ScaNN库（Guo等人，2020）时的运行时间/质量权衡曲线。两个模型每个查询都使用一个大小为$k = {1024}$的向量；DE - BERT每个文档使用一个大小为$k = {1024}$的向量，而ME - BERT每个文档使用3个大小为$k = {1024}$的向量。DE - BERT的文档索引大小为${34.2}\mathrm{{GB}}$，ME - BERT的索引大小约为其3倍。DE - BERT和ME - BERT的索引时间分别为${1.52}\mathrm{\;h}$和${3.02}\mathrm{\;h}$。我们使用的ScaNN配置为num_leaves = 5000，num_leaves_to_search的范围从25到2000（从较不精确到更精确的搜索），并且在所有160个核心上使用并行推理时测量每个查询的时间。在曲线的高质量范围内，对于相同的每个查询推理时间，ME - BERT的平均倒数排名（MRR）明显高于DE - BERT。

## 8 Related work

## 8 相关工作

We have mentioned research on improving the accuracy of retrieval models throughout the paper. Here we focus on work related to our central focus on the capacity of dense dual encoder representations relative to sparse bags-of-words.

我们在整篇论文中都提到了提高检索模型准确性的研究。在这里，我们专注于与我们的核心关注点相关的工作，即密集双编码器表示相对于稀疏词袋的容量。

In compressive sensing it is possible to recover a bag of words vector $x$ from the projection ${Ax}$ for suitable $A$ . Bounds for the sufficient dimensionality of isotropic Gaussian projections (Can-des and Tao, 2005; Arora et al., 2018) are more pessimistic than the bound described in $§2$ ,but this is unsurprising because the task of recovering bags-of-words from a compressed measurement is strictly harder than recovering inner products.

在压缩感知中，对于合适的$A$，可以从投影${Ax}$中恢复词袋向量$x$。各向同性高斯投影的足够维度界限（Can - des和Tao，2005；Arora等人，2018）比$§2$中描述的界限更悲观，但这并不奇怪，因为从压缩测量中恢复词袋的任务比恢复内积严格更难。

Subramani et al. (2019) ask whether it is possible to exactly recover sentences (token sequences) from pretrained decoders, using vector embed-dings that are added as a bias to the decoder hidden state. Because their decoding model is more expressive (and thus more computationally intensive) than inner product retrieval, the theoretical issues examined here do not apply. Nonetheless, Subramani et al. empirically observe a similar dependence between sentence length and embedding size. Wieting and Kiela (2019) represent sentences as bags of random projections, finding that high-dimensional projections $\left( {k = {4096}}\right)$ perform nearly as well as trained encoding models. These empirical results provide further empirical support for the hypothesis that bag-of-words vectors from real text are "hard to embed" in the sense of Larsen and Nelson (2017). Our contribution is to systematically explore the relationship between document length and encoding dimension, focusing on the case of exact inner product-based retrieval. We leave the combination of representation learning and approximate retrieval for future work.

Subramani等人（2019）探讨了是否可以使用作为偏置添加到解码器隐藏状态的向量嵌入，从预训练的解码器中精确恢复句子（标记序列）。由于他们的解码模型比内积检索更具表达性（因此计算量更大），这里研究的理论问题并不适用。尽管如此，Subramani等人通过实验观察到句子长度和嵌入大小之间存在类似的依赖关系。Wieting和Kiela（2019）将句子表示为随机投影的词袋，发现高维投影$\left( {k = {4096}}\right)$的性能与训练过的编码模型几乎一样好。这些实验结果为Larsen和Nelson（2017）提出的真实文本的词袋向量“难以嵌入”这一假设提供了进一步的实验支持。我们的贡献是系统地探索文档长度和编码维度之间的关系，重点关注基于精确内积的检索情况。我们将表示学习和近似检索的结合留作未来的工作。

## 9 Conclusion

## 9 结论

Transformers perform well on an unreasonable range of problems in natural language processing. Yet the computational demands of large-scale retrieval push us to seek other architectures: cross-attention over contextualized embeddings is too slow, but dual encoding into fixed-length vectors may be insufficiently expressive, sometimes failing even to match the performance of sparse bag-of-words competitors. We have used both theoretical and empirical techniques to characterize the fidelity of fixed-length dual encoders, focusing on the role of document length. Based on these observations, we propose hybrid models that yield strong performance while maintaining scalability.

Transformer在自然语言处理的众多问题上表现出色。然而，大规模检索的计算需求促使我们寻找其他架构：基于上下文嵌入的交叉注意力机制速度太慢，而将其双编码为固定长度的向量可能表达能力不足，有时甚至无法达到稀疏词袋竞争对手的性能。我们使用理论和实验技术来描述固定长度双编码器的保真度，重点关注文档长度的作用。基于这些观察，我们提出了在保持可扩展性的同时具有强大性能的混合模型。

Acknowledgments We thank Ming-Wei Chang, Jon Clark, William Cohen, Kelvin Guu, Sanjiv Kumar, Kenton Lee, Jimmy Lin, Ankur Parikh, Ice Pasupat, Iulia Turc, William A. Woods, Vincent Zhao, and the anonymous reviewers for helpful discussions of this work.

致谢 我们感谢张明伟（Ming - Wei Chang）、乔恩·克拉克（Jon Clark）、威廉·科恩（William Cohen）、凯尔文·顾（Kelvin Guu）、桑吉夫·库马尔（Sanjiv Kumar）、肯顿·李（Kenton Lee）、吉米·林（Jimmy Lin）、安库尔·帕里克（Ankur Parikh）、艾斯·帕苏帕特（Ice Pasupat）、尤利娅·图尔克（Iulia Turc）、威廉·A·伍兹（William A. Woods）、文森特·赵（Vincent Zhao）以及匿名审稿人对这项工作的有益讨论。

References

参考文献

Dimitris Achlioptas. 2003. Database-friendly random projections: Johnson-Lindenstrauss with binary coins. Journal of Computer and System Sciences, 66(4):671-687.

迪米特里斯·阿赫利奥普塔斯（Dimitris Achlioptas）。2003年。对数据库友好的随机投影：使用二元硬币的约翰逊 - 林登施特劳斯引理。《计算机与系统科学杂志》，66(4)：671 - 687。

Noga Alon and Bo'az Klartag. 2017. Optimal compression of approximate inner products and dimension reduction. In58th Annual Symposium on Foundations of Computer Science (FOCS).

诺加·阿隆（Noga Alon）和博阿兹·克拉塔格（Bo'az Klartag）。2017年。近似内积的最优压缩和降维。在第58届计算机科学基础年度研讨会（FOCS）上。

Alexandr Andoni, Piotr Indyk, and Ilya Razen-shteyn. 2019. Approximate nearest neighbor search in high dimensions. Proceedings of the International Congress of Mathematicians (ICM 2018).

亚历山大·安多尼（Alexandr Andoni）、彼得·因迪克（Piotr Indyk）和伊利亚·拉曾施泰因（Ilya Razenshteyn）。2019年。高维近似最近邻搜索。《国际数学家大会会议录（ICM 2018）》。

Sanjeev Arora, Mikhail Khodak, Nikunj Saun-shi, and Kiran Vodrahalli. 2018. A compressed sensing view of unsupervised text embeddings, bag-of-n-grams, and LSTMs. In Proceedings of the International Conference on Learning Representations (ICLR).

桑吉夫·阿罗拉（Sanjeev Arora）、米哈伊尔·霍达克（Mikhail Khodak）、尼孔吉·绍恩希（Nikunj Saunshi）和基兰·沃德拉哈利（Kiran Vodrahalli）。2018年。无监督文本嵌入、n元词袋和长短期记忆网络（LSTM）的压缩感知视角。收录于《国际学习表征会议论文集》（ICLR）。

Shai Ben-David, Nadav Eiron, and Hans Ulrich Simon. 2002. Limitations of learning via em-beddings in Euclidean half spaces. Journal of Machine Learning Research, 3(Nov):441-461.

沙伊·本 - 戴维德（Shai Ben - David）、纳达夫·艾龙（Nadav Eiron）和汉斯·乌尔里希·西蒙（Hans Ulrich Simon）。2002年。通过欧几里得半空间嵌入进行学习的局限性。《机器学习研究杂志》，3（11月）：441 - 461。

Emmanuel J Candes and Terence Tao. 2005. Decoding by linear programming. IEEE transactions on information theory, 51(12):4203-4215.

伊曼纽尔·J·坎德斯（Emmanuel J Candes）和陶哲轩（Terence Tao）。2005年。通过线性规划进行解码。《电气与电子工程师协会信息论汇刊》，51（12）：4203 - 4215。

Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, and Ellen M. Voorhees. 2020. Overview of the TREC 2019 deep learning track. In Text REtrieval Conference (TREC). TREC.

尼克·克拉斯韦尔（Nick Craswell）、巴斯卡尔·米特拉（Bhaskar Mitra）、埃米内·伊尔马兹（Emine Yilmaz）、丹尼尔·坎波斯（Daniel Campos）和埃伦·M·沃里斯（Ellen M. Voorhees）。2020年。2019年文本检索会议（TREC）深度学习赛道综述。收录于《文本检索会议》（TREC）。TREC。

Zhuyun Dai and Jamie Callan. 2020a. Context-aware document term weighting for ad-hoc search. In Proceedings of The Web Conference 2020.

戴珠云（Zhuyun Dai）和杰米·卡兰（Jamie Callan）。2020a。用于临时搜索的上下文感知文档词项加权。收录于《2020年万维网会议论文集》。

Zhuyun Dai and Jamie Callan. 2020b. Context-aware sentence/passage term importance estimation for first stage retrieval. Proceedings of the ACM SIGIR International Conference on Theory of Information Retrieval.

戴珠云（Zhuyun Dai）和杰米·卡兰（Jamie Callan）。2020b。用于第一阶段检索的上下文感知句子/段落词项重要性估计。《美国计算机协会信息检索理论国际会议论文集》。

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language

雅各布·德夫林（Jacob Devlin）、张明伟（Ming - Wei Chang）、肯顿·李（Kenton Lee）和克里斯蒂娜·图托纳娃（Kristina Toutanova）。2019年。BERT：用于语言理解的深度双向变换器预训练

understanding. In Proceedings of the 2019 Conference of the North American Chapter of

收录于《2019年北美计算语言学协会人类语言技术会议论文集》

the Association for Computational Linguistics: Human Language Technologies.

计算语言学协会：人类语言技术分会。

Laura Dietz, Ben Gamari, Jeff Dalton, and Nick Craswell. 2018. TREC complex answer retrieval overview. In Text REtrieval Conference (TREC).

劳拉·迪茨（Laura Dietz）、本·加马里（Ben Gamari）、杰夫·道尔顿（Jeff Dalton）和尼克·克拉斯韦尔（Nick Craswell）。2018年。文本检索会议（TREC）复杂答案检索综述。收录于《文本检索会议》（TREC）。

Luyu Gao, Zhuyun Dai, Zhen Fan, and Jamie Callan. 2020. Complementing lexical retrieval with semantic residual embedding. CoRR, abs/2004.13969. Version 1.

高璐宇（Luyu Gao）、戴珠云（Zhuyun Dai）、樊震（Zhen Fan）和杰米·卡兰（Jamie Callan）。2020年。用语义残差嵌入补充词汇检索。计算机研究报告库（CoRR），编号：abs/2004.13969。版本1。

Daniel Gillick, Sayali Kulkarni, Larry Lansing, Alessandro Presta, Jason Baldridge, Eugene Ie, and Diego Garcia-Olano. 2019. Learning dense representations for entity retrieval. In Proceedings of the 23rd Conference on Computational Natural Language Learning (CoNLL).

丹尼尔·吉利克（Daniel Gillick）、萨亚利·库尔卡尼（Sayali Kulkarni）、拉里·兰辛（Larry Lansing）、亚历山德罗·普雷斯塔（Alessandro Presta）、杰森·鲍德里奇（Jason Baldridge）、伊恩·尤金（Eugene Ie）和迭戈·加西亚 - 奥拉诺（Diego Garcia - Olano）。2019年。学习用于实体检索的密集表示。收录于《第23届计算自然语言学习会议论文集》（CoNLL）。

Jiafeng Guo, Yixing Fan, Qingyao Ai, and W. Bruce Croft. 2016a. A deep relevance matching model for ad-hoc retrieval. In Proceedings of the 25th ACM International on Conference on Information and Knowledge Management.

郭佳峰（Jiafeng Guo）、范益兴（Yixing Fan）、艾清瑶（Qingyao Ai）和W.布鲁斯·克罗夫特（W. Bruce Croft）。2016a。用于临时检索的深度相关性匹配模型。收录于《第25届美国计算机协会信息与知识管理国际会议论文集》。

Ruiqi Guo, Sanjiv Kumar, Krzysztof Choroman-ski, and David Simcha. 2016b. Quantization based fast inner product search. In Proceedings of the International Conference on Artificial Intelligence and Statistics (AISTATS).

郭瑞琪（Ruiqi Guo）、桑吉夫·库马尔（Sanjiv Kumar）、克日什托夫·乔罗曼斯基（Krzysztof Choromanski）和大卫·辛查（David Simcha）。2016b。基于量化的快速内积搜索。收录于《国际人工智能与统计会议论文集》（AISTATS）。

Ruiqi Guo, Philip Sun, Erik Lindgren, Quan Geng, David Simcha, Felix Chern, and Sanjiv Kumar. 2020. Accelerating large-scale inference with anisotropic vector quantization. In Proceedings of the 37th International Conference on Machine Learning.

郭瑞琪（Ruiqi Guo）、菲利普·孙（Philip Sun）、埃里克·林德格伦（Erik Lindgren）、耿全（Quan Geng）、大卫·辛查（David Simcha）、费利克斯·陈（Felix Chern）和桑吉夫·库马尔（Sanjiv Kumar）。2020年。用各向异性向量量化加速大规模推理。收录于《第37届国际机器学习会议论文集》。

Po-Sen Huang, Xiaodong He, Jianfeng Gao, Li Deng, Alex Acero, and Larry Heck. 2013. Learning deep structured semantic models for web search using clickthrough data. In Proceedings of the International Conference on Information and Knowledge Management (CIKM).

黄伯森（Po - Sen Huang）、何晓东（Xiaodong He）、高剑锋（Jianfeng Gao）、邓力（Li Deng）、亚历克斯·阿塞罗（Alex Acero）和拉里·赫克（Larry Heck）。2013年。利用点击数据学习用于网络搜索的深度结构化语义模型。收录于《信息与知识管理国际会议论文集》（CIKM）。

Samuel Humeau, Kurt Shuster, Marie-Anne Lachaux, and Jason Weston. 2020. Poly-encoders: Transformer architectures and pretraining strategies for fast and accurate multi-

塞缪尔·于莫（Samuel Humeau）、库尔特·舒斯特（Kurt Shuster）、玛丽 - 安妮·拉肖（Marie - Anne Lachaux）和杰森·韦斯顿（Jason Weston）。2020年。多编码器：用于快速准确多

sentence scoring. In Proceedings of the International Conference on Learning Representations (ICLR).

句子评分的Transformer架构和预训练策略。发表于国际学习表征会议（ICLR）论文集。

Thathachar S Jayram and David P Woodruff. 2013. Optimal bounds for Johnson-Lindenstrauss transforms and streaming problems with subconstant error. ${ACM}$ Transactions on Algorithms (TALG), 9(3):1-17.

塔塔查尔·S·杰拉姆（Thathachar S Jayram）和大卫·P·伍德拉夫（David P Woodruff）。2013年。约翰逊 - 林登施特劳斯变换（Johnson - Lindenstrauss transforms）的最优边界以及具有次常数误差的流式问题。《算法汇刊》（Transactions on Algorithms，TALG），9(3):1 - 17。

William B Johnson and Joram Lindenstrauss. 1984. Extensions of Lipschitz mappings into a Hilbert space. Contemporary mathematics, 26(189-206):1.

威廉·B·约翰逊（William B Johnson）和约拉姆·林登施特劳斯（Joram Lindenstrauss）。1984年。利普希茨映射到希尔伯特空间的扩展。《当代数学》，26(189 - 206):1。

Daniel Kane, Raghu Meka, and Jelani Nelson. 2011. Almost optimal explicit Johnson-Lindenstrauss families. In Approximation, Randomization, and Combinatorial Optimization. Algorithms and Techniques.

丹尼尔·凯恩（Daniel Kane）、拉古·梅卡（Raghu Meka）和杰拉尼·纳尔逊（Jelani Nelson）。2011年。几乎最优的显式约翰逊 - 林登施特劳斯族。发表于《近似、随机化与组合优化：算法与技术》。

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP).

弗拉基米尔·卡尔普欣（Vladimir Karpukhin）、巴拉斯·奥古兹（Barlas Oguz）、闵世元（Sewon Min）、帕特里克·刘易斯（Patrick Lewis）、莱德尔·吴（Ledell Wu）、谢尔盖·叶杜诺夫（Sergey Edunov）、陈丹琦（Danqi Chen）和易文涛（Wen - tau Yih）。2020年。用于开放域问答的密集段落检索。发表于2020年自然语言处理经验方法会议（EMNLP）论文集。

Omar Khattab and Matei Zaharia. 2020. ColBERT. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval.

奥马尔·哈塔卜（Omar Khattab）和马特·扎哈里亚（Matei Zaharia）。2020年。ColBERT。发表于第43届国际计算机协会信息检索研究与发展会议论文集。

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural questions: A benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:453-466.

汤姆·夸特科夫斯基（Tom Kwiatkowski）、珍妮玛丽亚·帕洛马基（Jennimaria Palomaki）、奥利维亚·雷德菲尔德（Olivia Redfield）、迈克尔·柯林斯（Michael Collins）、安库尔·帕里克（Ankur Parikh）、克里斯·阿尔伯蒂（Chris Alberti）、丹妮尔·爱泼斯坦（Danielle Epstein）、伊利亚·波洛苏金（Illia Polosukhin）、雅各布·德夫林（Jacob Devlin）、肯顿·李（Kenton Lee）、克里斯蒂娜·图托纳娃（Kristina Toutanova）、利翁·琼斯（Llion Jones）、马修·凯尔西（Matthew Kelcey）、张明伟（Ming - Wei Chang）、安德鲁·M·戴（Andrew M. Dai）、雅各布·乌斯佐赖特（Jakob Uszkoreit）、勒·奎克（Quoc Le）和斯拉夫·彼得罗夫（Slav Petrov）。2019年。自然问题：问答研究的基准。《计算语言学协会汇刊》，7:453 - 466。

Cody Kwok, Oren Etzioni, and Daniel S Weld. 2001. Scaling question answering to the web. ACM Transactions on Information Systems (TOIS), 19(3):242-262.

科迪·郭（Cody Kwok）、奥伦·埃齐奥尼（Oren Etzioni）和丹尼尔·S·韦尔德（Daniel S Weld）。2001年。将问答扩展到网络。《ACM信息系统汇刊》（ACM Transactions on Information Systems，TOIS），19(3):242 - 262。

Kasper Green Larsen and Jelani Nelson. 2017. Optimality of the Johnson-Lindenstrauss lemma. In 2017 IEEE 58th Annual Symposium on Foundations of Computer Science (FOCS).

卡斯珀·格林·拉森（Kasper Green Larsen）和杰拉尼·纳尔逊（Jelani Nelson）。2017年。约翰逊 - 林登施特劳斯引理的最优性。发表于2017年IEEE第58届计算机科学基础年度研讨会（FOCS）论文集。

Kenton Lee, Ming-Wei Chang, and Kristina

肯顿·李（Kenton Lee）、张明伟（Ming - Wei Chang）和克里斯蒂娜

Toutanova. 2019. Latent retrieval for weakly supervised open domain question answering. In Proceedings of the Association for Computational Linguistics (ACL).

·图托纳娃（Kristina Toutanova）。2019年。用于弱监督开放域问答的潜在检索。发表于计算语言学协会会议（ACL）论文集。

Ji Ma, Ivan Korotkov, Yinfei Yang, Keith Hall, and Ryan T. McDonald. 2020. Zero-shot neural retrieval via domain-targeted synthetic query generation. CoRR, abs/2004.14503.

马骥（Ji Ma）、伊万·科罗特科夫（Ivan Korotkov）、杨荫飞（Yinfei Yang）、基思·霍尔（Keith Hall）和瑞安·T·麦克唐纳（Ryan T. McDonald）。2020年。通过领域目标合成查询生成实现零样本神经检索。计算机研究存储库（CoRR），abs/2004.14503。

Bhaskar Mitra and Nick Craswell. 2018. An introduction to neural information retrieval. Foundations and Trends® in Information Retrieval, ${13}\left( 1\right)  : 1 - {126}$ .

巴斯卡尔·米特拉（Bhaskar Mitra）和尼克·克拉斯韦尔（Nick Craswell）。2018年。神经信息检索导论。《信息检索基础与趋势》。

Karthik Narasimhan, Adam Yala, and Regina Barzilay. 2016. Improving information extraction by acquiring external evidence with reinforcement learning. In Proceedings of Empirical Methods in Natural Language Processing (EMNLP).

卡尔蒂克·纳拉西姆汉（Karthik Narasimhan）、亚当·亚拉（Adam Yala）和雷吉娜·巴尔齐莱（Regina Barzilay）。2016年。通过强化学习获取外部证据改进信息提取。发表于自然语言处理经验方法会议（EMNLP）论文集。

Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, and Li Deng. 2016. MS MARCO: A human generated machine reading comprehension dataset.

特里·阮（Tri Nguyen）、米尔·罗森伯格（Mir Rosenberg）、宋霞（Xia Song）、高剑锋（Jianfeng Gao）、索拉布·蒂瓦里（Saurabh Tiwary）、兰甘·马宗达（Rangan Majumder）和李邓（Li Deng）。2016年。MS MARCO：一个人工生成的机器阅读理解数据集。

Rodrigo Nogueira and Kyunghyun Cho. 2019. Passage re-ranking with BERT. CoRR, abs/1901.04085.

罗德里戈·诺盖拉（Rodrigo Nogueira）和赵京焕（Kyunghyun Cho）。2019年。使用BERT进行段落重排序。计算机研究报告（CoRR），编号：abs/1901.04085。

Rodrigo Nogueira and Jimmy Lin. 2019. From doc2query to doctttttquery. https:// github.com/castorini/docTTTTTquery.

罗德里戈·诺盖拉（Rodrigo Nogueira）和吉米·林（Jimmy Lin）。2019年。从doc2query到doctttttquery。https:// github.com/castorini/docTTTTTquery。

Rodrigo Nogueira, Wei Yang, Kyunghyun Cho, and Jimmy Lin. 2019a. Multi-stage document ranking with BERT. CoRR, abs/1910.14424.

罗德里戈·诺盖拉（Rodrigo Nogueira）、杨威（Wei Yang）、赵京焕（Kyunghyun Cho）和吉米·林（Jimmy Lin）。2019a。使用BERT进行多阶段文档排序。计算机研究报告（CoRR），编号：abs/1910.14424。

Rodrigo Nogueira, Wei Yang, Jimmy Lin, and Kyunghyun Cho. 2019b. Document expansion by query prediction. CoRR, abs/1904.08375.

罗德里戈·诺盖拉（Rodrigo Nogueira）、杨威（Wei Yang）、吉米·林（Jimmy Lin）和赵京焕（Kyunghyun Cho）。2019b。通过查询预测进行文档扩展。计算机研究报告（CoRR），编号：abs/1904.08375。

Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In Proceedings of Empirical Methods in Natural Language Processing (EMNLP).

尼尔斯·赖默斯（Nils Reimers）和伊琳娜·古列维奇（Iryna Gurevych）。2019年。句子BERT（Sentence-BERT）：使用孪生BERT网络的句子嵌入。收录于自然语言处理经验方法会议（EMNLP）论文集。

Stephen Robertson, Hugo Zaragoza, et al. 2009. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends $\mathbb{R}$ in Information Retrieval, 3(4):333-389.

斯蒂芬·罗伯逊（Stephen Robertson）、雨果·萨拉戈萨（Hugo Zaragoza）等。2009年。概率相关性框架：BM25及其拓展。信息检索基础与趋势，3(4):333 - 389。

Minjoon Seo, Jinhyuk Lee, Tom Kwiatkowski, Ankur Parikh, Ali Farhadi, and Hannaneh Ha-jishirzi. 2019. Real-time open-domain question answering with dense-sparse phrase index. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics.

徐民俊（Minjoon Seo）、李镇赫（Jinhyuk Lee）、汤姆·夸特科夫斯基（Tom Kwiatkowski）、安库尔·帕里克（Ankur Parikh）、阿里·法尔哈迪（Ali Farhadi）和汉娜·哈吉希尔齐（Hannaneh Ha - jishirzi）。2019年。使用密集 - 稀疏短语索引的实时开放域问答。收录于计算语言学协会第57届年会论文集。

Karen Spärck Jones. 1972. A statistical interpretation of term specificity and its application in retrieval. Journal of documentation, 28(1):11- 21.

凯伦·斯帕克·琼斯（Karen Spärck Jones）。1972年。术语特异性的统计解释及其在检索中的应用。文献学期刊，28(1):11 - 21。

Nishant Subramani, Samuel Bowman, and Kyunghyun Cho. 2019. Can unconditional language models recover arbitrary sentences? In Advances in Neural Information Processing Systems.

尼尚特·苏布拉马尼（Nishant Subramani）、塞缪尔·鲍曼（Samuel Bowman）和赵京焕（Kyunghyun Cho）。2019年。无条件语言模型能否恢复任意句子？收录于神经信息处理系统进展。

Santosh S. Vempala. 2004. The random projection method, volume 65. American Mathematical Society.

桑托什·S·文帕拉（Santosh S. Vempala）。2004年。随机投影方法，第65卷。美国数学学会。

Ellen M. Voorhees. 2001. The TREC question answering track. Natural Language Engineering, 7(4):361-378.

埃伦·M·沃里斯（Ellen M. Voorhees）。2001年。文本检索会议（TREC）问答赛道。自然语言工程，7(4):361 - 378。

John Wieting and Douwe Kiela. 2019. No training required: Exploring random encoders for sentence classification. In Proceedings of the International Conference on Learning Representations (ICLR).

约翰·维廷（John Wieting）和杜韦·基拉（Douwe Kiela）。2019年。无需训练：探索用于句子分类的随机编码器。收录于国际学习表征会议（ICLR）论文集。

Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed, and Arnold Overwijk. 2020. Approximate nearest neighbor negative contrastive learning for dense text retrieval. CoRR, abs/2007.00808. Version 1.

熊磊（Lee Xiong）、熊晨彦（Chenyan Xiong）、李烨（Ye Li）、邓国峰（Kwok - Fung Tang）、刘佳琳（Jialin Liu）、保罗·贝内特（Paul Bennett）、朱奈德·艾哈迈德（Junaid Ahmed）和阿诺德·奥弗维克（Arnold Overwijk）。2020年。用于密集文本检索的近似最近邻负对比学习。计算机研究报告（CoRR），编号：abs/2007.00808。版本1。

Ming Yan, Chenliang Li, Chen Wu, Bin Bi, Wei Wang, Jiangnan Xia, and Luo Si. 2020. IDST at TREC 2019 deep learning track: Deep cascade ranking with generation-based document expansion and pre-trained language modeling. In Text REtrieval Conference (TREC).

严明（Ming Yan）、李晨亮（Chenliang Li）、吴晨（Chen Wu）、毕斌（Bin Bi）、王巍（Wei Wang）、夏江南（Jiangnan Xia）和司罗（Luo Si）。2020年。IDST团队参加2019年文本检索会议（TREC）深度学习赛道：基于生成的文档扩展和预训练语言模型的深度级联排序。收录于文本检索会议（TREC）论文集。

Peilin Yang, Hui Fang, and Jimmy Lin. 2018a. Anserini: Reproducible ranking baselines using lucene. J. Data and Information Quality, 10(4).

杨培林（Peilin Yang）、方慧（Hui Fang）和吉米·林（Jimmy Lin）。2018a。安瑟里尼（Anserini）：使用Lucene的可复现排序基线。数据与信息质量期刊，10(4)。

Wei Yang, Haotian Zhang, and Jimmy Lin. 2019. Simple applications of BERT for ad hoc document retrieval. CoRR, abs/1903.10972.

杨威（Wei Yang）、张昊天（Haotian Zhang）和吉米·林（Jimmy Lin）。2019年。BERT在临时文档检索中的简单应用。计算机研究报告（CoRR），编号：abs/1903.10972。

Zhilin Yang, Zihang Dai, Ruslan Salakhutdinov, and William W. Cohen. 2018b. Breaking the softmax bottleneck: A high-rank RNN language model. In Proceedings of the International Conference on Learning Representations (ICLR).

杨志林（Zhilin Yang）、戴子航（Zihang Dai）、鲁斯兰·萨拉赫丁诺夫（Ruslan Salakhutdinov）和威廉·W·科恩（William W. Cohen）。2018b。打破softmax瓶颈：一种高秩RNN语言模型。发表于国际学习表征会议（ICLR）论文集。

## A Proofs

## 证明

### A.1 Lemma 1

### A.1 引理1

Proof. For both distributions of embeddings, the error on the squared norm can be bounded with high probability (Achlioptas, 2003, Lemma 5.1):

证明。对于嵌入的两种分布，平方范数的误差在高概率下可以被界定（阿赫利奥普塔斯（Achlioptas），2003，引理5.1）：

$$
\Pr \left( {\left| {\parallel {Ax}{\parallel }^{2} - \parallel x{\parallel }^{2}}\right|  > \epsilon \parallel x{\parallel }^{2}}\right) 
$$

$$
 < 2\exp \left( {-\frac{k}{2}\left( {{\epsilon }^{2}/2 - {\epsilon }^{3}/3}\right) }\right) . \tag{4}
$$

This bound implies an analogous bound on the absolute error of the inner product (Ben-David et al., 2002, corollary 19),

这个界意味着内积绝对误差也有类似的界（本 - 戴维（Ben - David）等人，2002，推论19），

$$
\Pr \left( {\left| {\langle {Ax},{Ay}\rangle -\langle x,y\rangle }\right|  \geq  \frac{\epsilon }{2}\left( {\parallel x{\parallel }^{2} + \parallel y{\parallel }^{2}}\right) }\right) 
$$

$$
 \leq  4\exp \left( {-\frac{k}{2}\left( {{\epsilon }^{2}/2 - {\epsilon }^{3}/3}\right) }\right) . \tag{5}
$$

Let $\bar{q} = q/\parallel q\parallel$ and $\bar{d} = \left( {{d}_{1} - {d}_{2}}\right) /\begin{Vmatrix}{{d}_{1} - {d}_{2}}\end{Vmatrix}$ . Then $\mu \left( {q,{d}_{1},{d}_{2}}\right)  = \langle \bar{q},\bar{d}\rangle$ . A ranking error occurs if and only if $\langle A\bar{q},A\bar{d}\rangle  \leq  0$ ,which implies $\left| {\langle A\bar{q},A\bar{d}\rangle -\langle \bar{q},\bar{d}\rangle }\right|  \geq  \epsilon$ . By construction $\parallel \bar{q}\parallel  = \parallel \bar{d}\parallel  = 1$ ,so the probability of an inner product distortion $\geq  \epsilon$ is bounded by the righthand side of (5).

设$\bar{q} = q/\parallel q\parallel$和$\bar{d} = \left( {{d}_{1} - {d}_{2}}\right) /\begin{Vmatrix}{{d}_{1} - {d}_{2}}\end{Vmatrix}$ 。那么$\mu \left( {q,{d}_{1},{d}_{2}}\right)  = \langle \bar{q},\bar{d}\rangle$ 。当且仅当$\langle A\bar{q},A\bar{d}\rangle  \leq  0$时会发生排序误差，这意味着$\left| {\langle A\bar{q},A\bar{d}\rangle -\langle \bar{q},\bar{d}\rangle }\right|  \geq  \epsilon$ 。根据构造$\parallel \bar{q}\parallel  = \parallel \bar{d}\parallel  = 1$，所以内积失真$\geq  \epsilon$的概率由(5)式的右侧界定。

### A.2 Corollary 1

### A.2 推论1

Proof. We have $\epsilon  = \mu \left( {q,{d}_{1},{d}_{2}}\right)  = \langle \bar{q},\bar{d}\rangle  \leq  1$ by the Cauchy-Schwarz inequality. For $\epsilon  \leq  1$ ,we have ${\epsilon }^{2}/6 \leq  {\epsilon }^{2}/2 - {\epsilon }^{3}/3$ . We can then loosen the bound in (1) to $\beta  \leq  4\exp \left( {-\frac{k}{2}\frac{{\epsilon }^{2}}{6}}\right)$ . Taking the natural $\log$ yields $\ln \beta  \leq  \ln 4 - {\epsilon }^{2}k/{12}$ ,which can be rearranged into $k \geq  {12}{\epsilon }^{-2}\ln \frac{4}{\beta }$ .

证明。根据柯西 - 施瓦茨不等式，我们有$\epsilon  = \mu \left( {q,{d}_{1},{d}_{2}}\right)  = \langle \bar{q},\bar{d}\rangle  \leq  1$ 。对于$\epsilon  \leq  1$，我们有${\epsilon }^{2}/6 \leq  {\epsilon }^{2}/2 - {\epsilon }^{3}/3$ 。然后我们可以将(1)式中的界放宽到$\beta  \leq  4\exp \left( {-\frac{k}{2}\frac{{\epsilon }^{2}}{6}}\right)$ 。取自然$\log$得到$\ln \beta  \leq  \ln 4 - {\epsilon }^{2}k/{12}$，可以将其重排为$k \geq  {12}{\epsilon }^{-2}\ln \frac{4}{\beta }$ 。

### A.3 Lemma 2

### A.3 引理2

Proof. For convenience define $\mu \left( {d}_{2}\right)  =$ $\mu \left( {q,{d}_{1},{d}_{2}}\right)$ . Define $\epsilon$ as in the theorem statement, and ${\mathcal{D}}_{\epsilon } = \left\{  {{d}_{2} \in  \mathcal{D} : \mu \left( {q,{d}_{1},{d}_{2}}\right)  \geq  \epsilon }\right\}$ . We have

证明。为方便起见，定义$\mu \left( {d}_{2}\right)  =$ $\mu \left( {q,{d}_{1},{d}_{2}}\right)$ 。如定理陈述中那样定义$\epsilon$，以及${\mathcal{D}}_{\epsilon } = \left\{  {{d}_{2} \in  \mathcal{D} : \mu \left( {q,{d}_{1},{d}_{2}}\right)  \geq  \epsilon }\right\}$ 。我们有

$$
\Pr \left( {R \geq  {r}_{0}}\right)  \leq  \Pr \left( {\exists {d}_{2} \in  {\mathcal{D}}_{\epsilon } : A{q}_{1} \leq  A{q}_{2}}\right) 
$$

$$
 \leq  \mathop{\sum }\limits_{{{d}_{2} \in  {\mathcal{D}}_{\epsilon }}}4\exp \left( {-\frac{k}{2}\left( {\mu {\left( {d}_{2}\right) }^{2}/2 - \mu {\left( {d}_{2}\right) }^{3}/3}\right) }\right) 
$$

$$
 \leq  4\left| {\mathcal{D}}_{\epsilon }\right| \exp \left( {-\frac{k}{2}\left( {{\epsilon }^{2}/2 - {\epsilon }^{3}/3}\right) }\right) .
$$

The first inequality follows because the event $R \geq$ ${r}_{0}$ implies the event $\exists {d}_{2} \in  {\mathcal{D}}_{\epsilon } : A{q}_{1} \leq  A{q}_{2}$ . The second inequality follows by a combination of Lemma 1 and the union bound. The final inequality follows because for any ${d}_{2} \in  {\mathcal{D}}_{\epsilon }$ , $\mu \left( {q,{d}_{1},{d}_{2}}\right)  \geq  \epsilon$ . The theorem follows because $\left| {\mathcal{D}}_{\epsilon }\right|  = \left| \mathcal{D}\right|  - {r}_{0} + 1$

第一个不等式成立是因为事件$R \geq$ ${r}_{0}$意味着事件$\exists {d}_{2} \in  {\mathcal{D}}_{\epsilon } : A{q}_{1} \leq  A{q}_{2}$ 。第二个不等式由引理1和联合界组合得出。最后一个不等式成立是因为对于任何${d}_{2} \in  {\mathcal{D}}_{\epsilon }$，$\mu \left( {q,{d}_{1},{d}_{2}}\right)  \geq  \epsilon$ 。定理得证是因为$\left| {\mathcal{D}}_{\epsilon }\right|  = \left| \mathcal{D}\right|  - {r}_{0} + 1$

### A.4 Corollary 2

### A.4 推论2

Proof. For the retrieval function $\mathop{\max }\limits_{d}\langle q,d\rangle$ , the minimum non-zero unnormalized margin $\left\langle  {q,{d}_{1}}\right\rangle   - \left\langle  {q,{d}_{2}}\right\rangle$ is 1 when $q$ and $d$ are Boolean vectors. Therefore the normalized margin has lower bound $\mu \left( {q,{d}_{1},{d}_{2}}\right)  \geq  1/\left( {\parallel q\parallel  \times  \begin{Vmatrix}{{d}_{1} - {d}_{2}}\end{Vmatrix}}\right)$ . For non-negative ${d}_{1}$ and ${d}_{2}$ we have $\begin{Vmatrix}{{d}_{1} - {d}_{2}}\end{Vmatrix} \leq$ $\sqrt{{\begin{Vmatrix}{d}_{1}\end{Vmatrix}}^{2} + {\begin{Vmatrix}{d}_{2}\end{Vmatrix}}^{2}} \leq  \sqrt{2{L}_{D}}$ . Preserving a normalized margin of $\epsilon  = {\left( 2{L}_{Q}{L}_{D}\right) }^{-\frac{1}{2}}$ is therefore sufficient to avoid any pairwise errors. By plugging this value into Corollary 1, we see that setting $k \geq  {24}{L}_{Q}{L}_{D}\ln \frac{4}{\beta }$ ensures that the probability of any pairwise error is $\leq  \beta$ .

证明。对于检索函数 $\mathop{\max }\limits_{d}\langle q,d\rangle$，当 $q$ 和 $d$ 为布尔向量时，最小非零未归一化间隔 $\left\langle  {q,{d}_{1}}\right\rangle   - \left\langle  {q,{d}_{2}}\right\rangle$ 为 1。因此，归一化间隔的下界为 $\mu \left( {q,{d}_{1},{d}_{2}}\right)  \geq  1/\left( {\parallel q\parallel  \times  \begin{Vmatrix}{{d}_{1} - {d}_{2}}\end{Vmatrix}}\right)$。对于非负的 ${d}_{1}$ 和 ${d}_{2}$，我们有 $\begin{Vmatrix}{{d}_{1} - {d}_{2}}\end{Vmatrix} \leq$ $\sqrt{{\begin{Vmatrix}{d}_{1}\end{Vmatrix}}^{2} + {\begin{Vmatrix}{d}_{2}\end{Vmatrix}}^{2}} \leq  \sqrt{2{L}_{D}}$。因此，保持 $\epsilon  = {\left( 2{L}_{Q}{L}_{D}\right) }^{-\frac{1}{2}}$ 的归一化间隔足以避免任何成对错误。将此值代入推论 1，我们发现设置 $k \geq  {24}{L}_{Q}{L}_{D}\ln \frac{4}{\beta }$ 可确保任何成对错误的概率为 $\leq  \beta$。

### A.5 Theorem 1

### A.5 定理 1

Proof. Recall that $\mu \left( {q,{d}_{1},{d}_{2}}\right)  = \frac{\left\langle  q,{d}_{1} - {d}_{2}\right\rangle  }{\parallel q\parallel  \times  \begin{Vmatrix}{{d}_{1} - {d}_{2}}\end{Vmatrix}}$ . By assumption we have $\left\langle  {q,{d}_{1}^{\left( i\right) }}\right\rangle   = \left\langle  {q,{d}_{1}}\right\rangle$ and $\mathop{\max }\limits_{j}\left\langle  {q,{d}_{2}^{\left( j\right) }}\right\rangle   \leq  \left\langle  {q,{d}_{2}}\right\rangle$ ,implying that

证明。回顾 $\mu \left( {q,{d}_{1},{d}_{2}}\right)  = \frac{\left\langle  q,{d}_{1} - {d}_{2}\right\rangle  }{\parallel q\parallel  \times  \begin{Vmatrix}{{d}_{1} - {d}_{2}}\end{Vmatrix}}$。根据假设，我们有 $\left\langle  {q,{d}_{1}^{\left( i\right) }}\right\rangle   = \left\langle  {q,{d}_{1}}\right\rangle$ 和 $\mathop{\max }\limits_{j}\left\langle  {q,{d}_{2}^{\left( j\right) }}\right\rangle   \leq  \left\langle  {q,{d}_{2}}\right\rangle$，这意味着

$$
\left\langle  {q,{d}_{1}^{\left( i\right) } - {d}_{2}^{\left( i\right) }}\right\rangle   \geq  \left\langle  {q,{d}_{1} - {d}_{2}}\right\rangle   \tag{6}
$$

In the denominator,we expand $\begin{Vmatrix}{{d}_{1} - {d}_{2}}\end{Vmatrix} =$ $\begin{Vmatrix}{\left( {{d}_{1}^{\left( i\right) } - {d}_{2}^{\left( i\right) }}\right)  + \left( {{d}_{1}^{\left( \neg i\right) } - {d}_{2}^{\left( \neg i\right) }}\right) }\end{Vmatrix}$ ,where ${d}^{\left( \neg i\right) } =$ $\mathop{\sum }\limits_{{j \neq  i}}{d}^{\left( j\right) }$ . Plugging this into the squared norm,

在分母中，我们展开 $\begin{Vmatrix}{{d}_{1} - {d}_{2}}\end{Vmatrix} =$ $\begin{Vmatrix}{\left( {{d}_{1}^{\left( i\right) } - {d}_{2}^{\left( i\right) }}\right)  + \left( {{d}_{1}^{\left( \neg i\right) } - {d}_{2}^{\left( \neg i\right) }}\right) }\end{Vmatrix}$，其中 ${d}^{\left( \neg i\right) } =$ $\mathop{\sum }\limits_{{j \neq  i}}{d}^{\left( j\right) }$。将其代入平方范数中

$$
{\begin{Vmatrix}{d}_{1} - {d}_{2}\end{Vmatrix}}^{2}
$$

$$
 = {\begin{Vmatrix}\left( {d}_{1}^{\left( i\right) } - {d}_{2}^{\left( i\right) }\right)  + \left( {d}_{1}^{\left( \neg i\right) } - {d}_{2}^{\left( \neg i\right) }\right) \end{Vmatrix}}^{2} \tag{7}
$$

$$
 = {\begin{Vmatrix}{d}_{1}^{\left( i\right) } - {d}_{2}^{\left( i\right) }\end{Vmatrix}}^{2} + {\begin{Vmatrix}{d}_{1}^{\left( \neg i\right) } - {d}_{2}^{\left( \neg i\right) }\end{Vmatrix}}^{2} \tag{8}
$$

$$
 + 2\left\langle  {{d}_{1}^{\left( i\right) } - {d}_{2}^{\left( i\right) },{d}_{1}^{\left( \neg i\right) } - {d}_{2}^{\left( \neg i\right) }}\right\rangle  
$$

$$
 = {\begin{Vmatrix}{d}_{1}^{\left( i\right) } - {d}_{2}^{\left( i\right) }\end{Vmatrix}}^{2} + {\begin{Vmatrix}{d}_{1}^{\left( \neg i\right) } - {d}_{2}^{\left( \neg i\right) }\end{Vmatrix}}^{2} \tag{9}
$$

$$
 \geq  {\begin{Vmatrix}{d}_{1}^{\left( i\right) } - {d}_{2}^{\left( i\right) }\end{Vmatrix}}^{2}. \tag{10}
$$

The inner product $\left\langle  {{d}_{1}^{\left( i\right) } - {d}_{2}^{\left( i\right) },{d}_{1}^{\left( \neg i\right) } - {d}_{2}^{\left( \neg i\right) }}\right\rangle   = 0$ because the segments are orthogonal. The combination of (6) and (10) completes the theorem.

由于各段是正交的，所以内积 $\left\langle  {{d}_{1}^{\left( i\right) } - {d}_{2}^{\left( i\right) },{d}_{1}^{\left( \neg i\right) } - {d}_{2}^{\left( \neg i\right) }}\right\rangle   = 0$。结合式 (6) 和式 (10)，定理得证。

<!-- Media -->

<table><tr><td>Model</td><td colspan="4">Reranking</td><td colspan="4">Retrieval</td></tr><tr><td>Passage length</td><td>50</td><td>100</td><td>200</td><td>400</td><td>50</td><td>100</td><td>200</td><td>400</td></tr><tr><td colspan="9">ICT task (MRR@10)</td></tr><tr><td>Cross-Attention</td><td>99.9</td><td>99.9</td><td>99.8</td><td>99.6</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>HYBRID-ME-BERT-uni</td><td>-</td><td>-</td><td>-</td><td>-</td><td>98.2</td><td>97.0</td><td>94.4</td><td>91.9</td></tr><tr><td>HYBRID-ME-BERT-bi</td><td>-</td><td>-</td><td>-</td><td>-</td><td>99.3</td><td>99.0</td><td>97.3</td><td>96.1</td></tr><tr><td>ME-BERT-768</td><td>98.0</td><td>96.7</td><td>92.4</td><td>89.8</td><td>96.8</td><td>96.1</td><td>91.1</td><td>85.2</td></tr><tr><td>ME-BERT-64</td><td>96.3</td><td>94.2</td><td>89.0</td><td>83.7</td><td>92.9</td><td>91.7</td><td>84.6</td><td>72.8</td></tr><tr><td>DE-BERT-768</td><td>91.7</td><td>87.8</td><td>79.7</td><td>74.1</td><td>90.2</td><td>85.6</td><td>72.9</td><td>63.0</td></tr><tr><td>DE-BERT-512</td><td>91.4</td><td>87.2</td><td>78.9</td><td>73.1</td><td>89.4</td><td>81.5</td><td>66.8</td><td>55.8</td></tr><tr><td>DE-BERT-128</td><td>90.5</td><td>85.0</td><td>75.0</td><td>68.1</td><td>85.7</td><td>75.4</td><td>58.0</td><td>47.3</td></tr><tr><td>DE-BERT-64</td><td>88.8</td><td>82.0</td><td>70.7</td><td>63.8</td><td>82.8</td><td>68.9</td><td>48.5</td><td>38.3</td></tr><tr><td>DE-BERT-32</td><td>83.6</td><td>74.9</td><td>62.6</td><td>55.9</td><td>70.1</td><td>53.2</td><td>34.0</td><td>27.6</td></tr><tr><td>BM25-uni</td><td>92.1</td><td>88.6</td><td>84.6</td><td>81.8</td><td>92.1</td><td>88.6</td><td>84.6</td><td>81.8</td></tr><tr><td>BM25-bi</td><td>98.0</td><td>97.1</td><td>95.9</td><td>94.5</td><td>98.0</td><td>97.1</td><td>95.9</td><td>94.5</td></tr></table>

<table><tbody><tr><td>模型</td><td colspan="4">重排序</td><td colspan="4">检索</td></tr><tr><td>段落长度</td><td>50</td><td>100</td><td>200</td><td>400</td><td>50</td><td>100</td><td>200</td><td>400</td></tr><tr><td colspan="9">信息对比任务（MRR@10）</td></tr><tr><td>交叉注意力</td><td>99.9</td><td>99.9</td><td>99.8</td><td>99.6</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>混合多嵌入BERT单向模型（HYBRID-ME-BERT-uni）</td><td>-</td><td>-</td><td>-</td><td>-</td><td>98.2</td><td>97.0</td><td>94.4</td><td>91.9</td></tr><tr><td>混合多嵌入BERT双向模型（HYBRID-ME-BERT-bi）</td><td>-</td><td>-</td><td>-</td><td>-</td><td>99.3</td><td>99.0</td><td>97.3</td><td>96.1</td></tr><tr><td>多嵌入BERT-768（ME-BERT-768）</td><td>98.0</td><td>96.7</td><td>92.4</td><td>89.8</td><td>96.8</td><td>96.1</td><td>91.1</td><td>85.2</td></tr><tr><td>多嵌入BERT-64（ME-BERT-64）</td><td>96.3</td><td>94.2</td><td>89.0</td><td>83.7</td><td>92.9</td><td>91.7</td><td>84.6</td><td>72.8</td></tr><tr><td>动态嵌入BERT-768（DE-BERT-768）</td><td>91.7</td><td>87.8</td><td>79.7</td><td>74.1</td><td>90.2</td><td>85.6</td><td>72.9</td><td>63.0</td></tr><tr><td>动态嵌入BERT-512（DE-BERT-512）</td><td>91.4</td><td>87.2</td><td>78.9</td><td>73.1</td><td>89.4</td><td>81.5</td><td>66.8</td><td>55.8</td></tr><tr><td>动态嵌入BERT-128（DE-BERT-128）</td><td>90.5</td><td>85.0</td><td>75.0</td><td>68.1</td><td>85.7</td><td>75.4</td><td>58.0</td><td>47.3</td></tr><tr><td>动态嵌入BERT-64（DE-BERT-64）</td><td>88.8</td><td>82.0</td><td>70.7</td><td>63.8</td><td>82.8</td><td>68.9</td><td>48.5</td><td>38.3</td></tr><tr><td>动态嵌入BERT-32（DE-BERT-32）</td><td>83.6</td><td>74.9</td><td>62.6</td><td>55.9</td><td>70.1</td><td>53.2</td><td>34.0</td><td>27.6</td></tr><tr><td>BM25单向模型</td><td>92.1</td><td>88.6</td><td>84.6</td><td>81.8</td><td>92.1</td><td>88.6</td><td>84.6</td><td>81.8</td></tr><tr><td>BM25双向模型</td><td>98.0</td><td>97.1</td><td>95.9</td><td>94.5</td><td>98.0</td><td>97.1</td><td>95.9</td><td>94.5</td></tr></tbody></table>

NQ (Recall@400 tokens)

NQ（400个词元召回率（Recall@400 tokens））

<table><tr><td>Cross-Attention</td><td>48.9</td><td>55.5</td><td>54.2</td><td>47.6</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>HYBRID-ME-BERT-uni</td><td>-</td><td>-</td><td>-</td><td>-</td><td>45.7</td><td>49.5</td><td>48.5</td><td>42.9</td></tr><tr><td>ME-BERT-768</td><td>43.6</td><td>49.6</td><td>46.5</td><td>38.7</td><td>42.0</td><td>43.3</td><td>40.4</td><td>34.4</td></tr><tr><td>ME-BERT-64</td><td>44.4</td><td>48.7</td><td>44.5</td><td>38.2</td><td>42.2</td><td>43.4</td><td>38.9</td><td>33.0</td></tr><tr><td>DE-BERT-768</td><td>42.9</td><td>47.7</td><td>44.4</td><td>36.6</td><td>44.2</td><td>44.0</td><td>40.1</td><td>32.2</td></tr><tr><td>DE-BERT-512</td><td>43.8</td><td>48.5</td><td>44.1</td><td>36.5</td><td>43.3</td><td>43.2</td><td>38.8</td><td>32.7</td></tr><tr><td>DE-BERT-128</td><td>42.8</td><td>45.7</td><td>41.2</td><td>35.7</td><td>38.0</td><td>36.7</td><td>32.8</td><td>27.0</td></tr><tr><td>DE-BERT-64</td><td>42.6</td><td>45.7</td><td>42.5</td><td>35.4</td><td>37.4</td><td>35.1</td><td>32.6</td><td>26.6</td></tr><tr><td>DE-BERT-32</td><td>42.4</td><td>45.8</td><td>42.1</td><td>34.0</td><td>36.3</td><td>34.7</td><td>31.0</td><td>24.9</td></tr><tr><td>BM25-uni</td><td>30.1</td><td>35.7</td><td>34.1</td><td>30.1</td><td>30.1</td><td>35.7</td><td>34.1</td><td>30.1</td></tr></table>

<table><tbody><tr><td>交叉注意力（Cross-Attention）</td><td>48.9</td><td>55.5</td><td>54.2</td><td>47.6</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>混合多编码器BERT统一模型（HYBRID-ME-BERT-uni）</td><td>-</td><td>-</td><td>-</td><td>-</td><td>45.7</td><td>49.5</td><td>48.5</td><td>42.9</td></tr><tr><td>多编码器BERT 768模型（ME-BERT-768）</td><td>43.6</td><td>49.6</td><td>46.5</td><td>38.7</td><td>42.0</td><td>43.3</td><td>40.4</td><td>34.4</td></tr><tr><td>多编码器BERT 64模型（ME-BERT-64）</td><td>44.4</td><td>48.7</td><td>44.5</td><td>38.2</td><td>42.2</td><td>43.4</td><td>38.9</td><td>33.0</td></tr><tr><td>双编码器BERT 768模型（DE-BERT-768）</td><td>42.9</td><td>47.7</td><td>44.4</td><td>36.6</td><td>44.2</td><td>44.0</td><td>40.1</td><td>32.2</td></tr><tr><td>双编码器BERT 512模型（DE-BERT-512）</td><td>43.8</td><td>48.5</td><td>44.1</td><td>36.5</td><td>43.3</td><td>43.2</td><td>38.8</td><td>32.7</td></tr><tr><td>双编码器BERT 128模型（DE-BERT-128）</td><td>42.8</td><td>45.7</td><td>41.2</td><td>35.7</td><td>38.0</td><td>36.7</td><td>32.8</td><td>27.0</td></tr><tr><td>双编码器BERT 64模型（DE-BERT-64）</td><td>42.6</td><td>45.7</td><td>42.5</td><td>35.4</td><td>37.4</td><td>35.1</td><td>32.6</td><td>26.6</td></tr><tr><td>双编码器BERT 32模型（DE-BERT-32）</td><td>42.4</td><td>45.8</td><td>42.1</td><td>34.0</td><td>36.3</td><td>34.7</td><td>31.0</td><td>24.9</td></tr><tr><td>BM25统一模型（BM25-uni）</td><td>30.1</td><td>35.7</td><td>34.1</td><td>30.1</td><td>30.1</td><td>35.7</td><td>34.1</td><td>30.1</td></tr></tbody></table>

<!-- Media -->

Table 3: Results on ICT task and NQ task (correspond to Fig. 4 and Fig. 5).

表3：信息与通信技术（ICT）任务和自然问答（NQ）任务的结果（对应图4和图5）。