# Rethinking the Role of Token Retrieval in Multi-Vector Retrieval

# 重新思考Token检索在多向量检索中的作用

Jinhyuk Lee* Zhuyun Dai Sai Meher Karthik Duddu

李镇赫（Jinhyuk Lee）* 戴竹云（Zhuyun Dai） 赛·梅赫尔·卡蒂克·杜杜（Sai Meher Karthik Duddu）

Tao Lei Iftekhar Naim Ming-Wei Chang Vincent Y. Zhao

陶磊（Tao Lei） 伊夫特哈尔·奈姆（Iftekhar Naim） 张明伟（Ming-Wei Chang） 赵文森（Vincent Y. Zhao）

Google DeepMind

谷歌（Google） 深度思维（DeepMind）

## Abstract

## 摘要

Multi-vector retrieval models such as ColBERT [Khattab and Zaharia, 2020] allow token-level interactions between queries and documents, and hence achieve state of the art on many information retrieval benchmarks. However, their nonlinear scoring function cannot be scaled to millions of documents, necessitating a three-stage process for inference: retrieving initial candidates via token retrieval, accessing all token vectors, and scoring the initial candidate documents. The non-linear scoring function is applied over all token vectors of each candidate document, making the inference process complicated and slow. In this paper, we aim to simplify the multi-vector retrieval by rethinking the role of token retrieval. We present XTR, ConteXtualized Token Retriever, which introduces a simple, yet novel, objective function that encourages the model to retrieve the most important document tokens first. The improvement to token retrieval allows XTR to rank candidates only using the retrieved tokens rather than all tokens in the document, and enables a newly designed scoring stage that is two-to-three orders of magnitude cheaper than that of ColBERT. On the popular BEIR benchmark, XTR advances the state-of-the-art by ${2.8}\mathrm{{nDCG}}@{10}$ without any distillation. Detailed analysis confirms our decision to revisit the token retrieval stage, as XTR demonstrates much better recall of the token retrieval stage compared to ColBERT.

像ColBERT [Khattab和Zaharia，2020]这样的多向量检索模型允许查询和文档之间进行词元（Token）级别的交互，因此在许多信息检索基准测试中达到了最先进水平。然而，它们的非线性评分函数无法扩展到数百万个文档，因此推理需要一个三阶段过程：通过词元（Token）检索获取初始候选文档、访问所有词元（Token）向量，以及对初始候选文档进行评分。非线性评分函数应用于每个候选文档的所有词元（Token）向量，这使得推理过程复杂且缓慢。在本文中，我们旨在通过重新思考词元（Token）检索的作用来简化多向量检索。我们提出了XTR，即上下文词元（Token）检索器，它引入了一个简单但新颖的目标函数，鼓励模型首先检索最重要的文档词元（Token）。对词元（Token）检索的改进使XTR仅使用检索到的词元（Token）而不是文档中的所有词元（Token）对候选文档进行排序，并实现了一个新设计的评分阶段，其成本比ColBERT低两到三个数量级。在流行的BEIR基准测试中，XTR在不进行任何蒸馏的情况下将最先进水平提高了${2.8}\mathrm{{nDCG}}@{10}$。详细分析证实了我们重新审视词元（Token）检索阶段的决定，因为与ColBERT相比，XTR在词元（Token）检索阶段表现出更好的召回率。

## 1 Introduction

## 1 引言

The performance of a dense retrieval model is largely affected by how it defines expressive representations over queries and documents, and whether it can efficiently retrieve and score a document using these vector representations. For example, dual encoder models [Yih et al., 2011, Lee et al., 2019, Karpukhin et al., 2020, Ni et al., 2021 encode queries and documents into single vectors and compute query-document similarities using dot products. While these models are very efficient for retrieval, their expressivity is limited due to the absence of token-level modeling for scoring. In contrast, multi-vector models such as ColBERT [Khattab and Zaharia, 2020, Santhanam et al., 2022b] are directly designed to capture token-level interactions. By utilizing a (non-linear) scoring function over all query and document token representations, multi-vector models enjoy much better model expressivity and often achieve superior results across various benchmarks [Thakur et al., 2021].

密集检索模型的性能在很大程度上受其如何定义查询和文档的有效表示，以及能否使用这些向量表示高效地检索和对文档进行评分的影响。例如，双编码器模型[Yih等人，2011年；Lee等人，2019年；Karpukhin等人，2020年；Ni等人，2021年]将查询和文档编码为单个向量，并使用点积计算查询 - 文档相似度。虽然这些模型在检索方面非常高效，但由于缺乏用于评分的词元级建模，其表达能力有限。相比之下，像ColBERT[Khattab和Zaharia，2020年；Santhanam等人，2022b]这样的多向量模型是直接为捕捉词元级交互而设计的。通过对所有查询和文档词元表示使用（非线性）评分函数，多向量模型具有更好的模型表达能力，并且在各种基准测试中通常能取得更优的结果[Thakur等人，2021年]。

The enhanced model expressivity, however, comes at a great cost of inference complexity. Unlike the case in dual encoders, the non-linear scoring function in multi-vector retrieval models prohibits the use of efficient Maximum Inner Product Search (MIPS) [Ram and Gray, 2012, Shrivastava and Li, 2014, 2015, Shen et al. 2015, for finding the maximum scoring documents. As a result, models such as ColBERT adopt an intricate and resource-intensive inference pipeline, which typically consists of three stages: 1) token retrieval: using each query token to retrieve document tokens, with their source documents becoming candidates; 2) gathering: collecting all the token embeddings from each candidate document, including those that are not retrieved in the first stage (most document tokens are not retrieved); and 3) scoring: ranking candidates using a non-linear function based on all the token embeddings per document.

然而，增强的模型表达能力是以推理复杂度大幅增加为代价的。与双编码器的情况不同，多向量检索模型中的非线性评分函数阻碍了使用高效的最大内积搜索（Maximum Inner Product Search，MIPS）[Ram和Gray，2012；Shrivastava和Li，2014，2015；Shen等人，2015]来查找得分最高的文档。因此，像ColBERT这样的模型采用了一个复杂且资源密集型的推理流程，该流程通常包括三个阶段：1) 词元检索：使用每个查询词元来检索文档词元，其源文档成为候选文档；2) 收集：从每个候选文档中收集所有词元嵌入，包括那些在第一阶段未被检索到的词元（大多数文档词元未被检索到）；3) 评分：使用基于每个文档所有词元嵌入的非线性函数对候选文档进行排序。

---

<!-- Footnote -->

*Correspondence: jinhyuklee@google.com

*通信地址：jinhyuklee@google.com

<!-- Footnote -->

---

This procedure leads to two major issues. First, compared to the token retrieval stage, gathering all document token embeddings and re-scoring the documents can introduce orders of magnitude additional data loading and floating operation cost, making multi-vector models extremely expensive to deploy. Secondly, while the candidate documents are decided in the token retrieval stage, previous training objectives are designed for the scoring stage. This creates a significant training-inference gap causing multi-vector models achieve sub-optimal (and often poor) recall performance. Clearly, the three-stage pipeline has largely limited the potential of multi-vector models, raising an interesting research question - can the token retrieval stage alone be sufficient for great performance?

这一过程导致了两个主要问题。首先，与词元检索阶段相比，收集所有文档词元嵌入并对文档进行重新评分会带来数量级的额外数据加载和浮点运算成本，使得多向量模型的部署成本极高。其次，虽然候选文档是在词元检索阶段确定的，但之前的训练目标是为评分阶段设计的。这造成了显著的训练 - 推理差距，导致多向量模型的召回性能欠佳（且往往很差）。显然，三阶段流程在很大程度上限制了多向量模型的潜力，从而引出了一个有趣的研究问题——仅词元检索阶段是否足以实现出色的性能？

We present XTR, ContXextualized Token Retriever: a simplified and efficient method for multi-vector retrieval, through re-thinking the role of token retrieval. The key insight of XTR is that the token retrieval in multi-vector models should be trained to retrieve the most salient and informative document tokens, so that the score between a query and document can be computed using only the retrieved information, just like how single-vector retrieval models work. By doing so, the gathering step can be completely eliminated, and the cost of scoring is significantly reduced as only a fraction of the tokens need to be considered and the dot products from the token retrieval can be reused. To improve the quality of the token retrieval, XTR proposes a novel, yet simple, training objective, which dramatically improves retrieval accuracy, doubling the chances of a gold token being retrieved in the top- $k$ results. Furthermore,despite the improved token retrieval,some relevant tokens may still be missed (i.e., not retrieved). To address this issue, we propose a simple method, called missing similarity imputation, which accounts for the contribution of the missing tokens to the overall score.

我们提出了XTR（上下文感知Token检索器，ContXextualized Token Retriever）：这是一种通过重新思考Token检索作用而实现的简化且高效的多向量检索方法。XTR的关键见解在于，多向量模型中的Token检索应经过训练，以检索出最突出且信息丰富的文档Token，这样就可以仅使用检索到的信息来计算查询与文档之间的得分，就像单向量检索模型的工作方式一样。通过这样做，可以完全省去收集步骤，并且由于只需要考虑一小部分Token，并且Token检索中的点积可以重复使用，因此评分成本显著降低。为了提高Token检索的质量，XTR提出了一种新颖而简单的训练目标，该目标显著提高了检索准确性，使黄金Token在排名前 $k$ 的结果中被检索到的概率翻倍。此外，尽管Token检索有所改进，但仍可能会遗漏一些相关Token（即未被检索到）。为了解决这个问题，我们提出了一种简单的方法，称为缺失相似度插补，该方法考虑了缺失Token对总体得分的贡献。

XTR streamlines the inference process, bringing it closer to the straightforward procedure of dual encoders, while maintaining and enhancing the expressive scoring function of multi-vector retrieval models. On the BEIR [Thakur et al. 2021] and LoTTE [Santhanam et al., 2022b] benchmarks, XTR attains state-of-the-art performance, requiring neither distillation nor hard negatiave mining. Notably, our model surpasses state-of-the-art dual-encoder GTR [Ni et al., 2021] by 3.6 nDCG@10 on BEIR without any additional training data. On the EntityQuestions benchmark [Sciavolino et al. 2021], XTR outperforms the previous state-of-the-art by 4.1 points on top-20 retrieval accuracy. XTR also does not require any secondary pre-training for retrieval and greatly outperforms mContriever [Izacard et al. 2022] on MIRACL, which contains multilingual retrieval tasks in 18 languages [Zhang et al., 2022b]. Our analysis supports that XTR indeed benefits from retrieving more contextualized tokens in relevant contexts, while making the scoring stage two-to-three orders of magnitude cheaper.

XTR简化了推理过程，使其更接近双编码器的直接流程，同时保持并增强了多向量检索模型的表达性评分函数。在BEIR [Thakur等人，2021]和LoTTE [Santhanam等人，2022b]基准测试中，XTR达到了最先进的性能，既不需要蒸馏也不需要难负样本挖掘。值得注意的是，在BEIR上，我们的模型在没有任何额外训练数据的情况下，比最先进的双编码器GTR [Ni等人，2021]的nDCG@10指标高出3.6。在EntityQuestions基准测试[Sciavolino等人，2021]中，XTR在前20检索准确率上比之前的最先进模型高出4.1分。XTR在检索时也不需要任何二次预训练，并且在包含18种语言的多语言检索任务的MIRACL上大大优于mContriever [Izacard等人，2022] [Zhang等人，2022b]。我们的分析表明，XTR确实受益于在相关上下文中检索更多上下文相关的标记，同时使评分阶段的成本降低两到三个数量级。

## 2 Background

## 2 背景

### 2.1 Multi-vector Retrieval

### 2.1 多向量检索

Single-vector retrieval models, also known as dual encoders, encode an input text sequence as a single dense embedding and define the similarity of a query and a document based on the dot product [Lee et al., 2019, Karpukhin et al., 2020]. Multi-vector retrieval models, on the other hand, make use of multiple dense embeddings for each query and document, typically leveraging all contextualized word representations of the input to gain improved model expressivity.

单向量检索模型，也称为双编码器，将输入的文本序列编码为单个密集嵌入，并基于点积来定义查询与文档的相似度 [Lee 等人，2019；Karpukhin 等人，2020]。另一方面，多向量检索模型为每个查询和文档使用多个密集嵌入，通常利用输入的所有上下文词表示来提高模型的表达能力。

Consider a query $Q = {\left\{  {\mathbf{q}}_{i}\right\}  }_{i = 1}^{n}$ and a document $D = {\left\{  {\mathbf{d}}_{j}\right\}  }_{j = 1}^{m}$ where ${\mathbf{q}}_{i}$ and ${\mathbf{d}}_{j}$ denote the $d$ - dimensional query token vector and the document token vector, respectively. Multi-vector retrieval models compute the query-document similarity as follows: $f\left( {Q,D}\right)  = \mathop{\sum }\limits_{{i = 1}}^{n}\mathop{\sum }\limits_{{j = 1}}^{m}{\mathbf{A}}_{ij}{\mathbf{P}}_{ij}$ where ${\mathbf{P}}_{ij} = {\mathbf{q}}_{i}^{\top }{\mathbf{d}}_{j}$ and $\mathbf{A} \in  \{ 0,1{\} }^{n \times  m}$ denotes the alignment matrix with ${\mathbf{A}}_{ij}$ being the token-level alignment between the query token vector ${\mathbf{q}}_{i}$ and the document token vector ${\mathbf{d}}_{j}$ . The sum-of-max operator of ColBERT [Khattab and Zaharia,2020] sets ${\mathbf{A}}_{ij} = {\mathbb{1}}_{\left\lbrack  j = {\operatorname{argmax}}_{{j}^{\prime }}\left( {\mathbf{P}}_{i{j}^{\prime }}\right) \right\rbrack  }$ where the argmax operator is over $1 \leq  {j}^{\prime } \leq  m$ (i.e.,tokens from a single document $D$ ) and ${\mathbb{1}}_{\left\lbrack  *\right\rbrack  }$ is an indicator function. Then, ${f}_{\text{ColBERT }}\left( {Q,D}\right)$ is defined as follows:

考虑一个查询 $Q = {\left\{  {\mathbf{q}}_{i}\right\}  }_{i = 1}^{n}$ 和一个文档 $D = {\left\{  {\mathbf{d}}_{j}\right\}  }_{j = 1}^{m}$，其中 ${\mathbf{q}}_{i}$ 和 ${\mathbf{d}}_{j}$ 分别表示 $d$ 维的查询词元向量和文档词元向量。多向量检索模型按如下方式计算查询 - 文档相似度：$f\left( {Q,D}\right)  = \mathop{\sum }\limits_{{i = 1}}^{n}\mathop{\sum }\limits_{{j = 1}}^{m}{\mathbf{A}}_{ij}{\mathbf{P}}_{ij}$ 其中 ${\mathbf{P}}_{ij} = {\mathbf{q}}_{i}^{\top }{\mathbf{d}}_{j}$ 和 $\mathbf{A} \in  \{ 0,1{\} }^{n \times  m}$ 表示对齐矩阵，${\mathbf{A}}_{ij}$ 是查询词元向量 ${\mathbf{q}}_{i}$ 和文档词元向量 ${\mathbf{d}}_{j}$ 之间的词元级对齐。ColBERT [Khattab 和 Zaharia，2020] 的最大求和运算符设定 ${\mathbf{A}}_{ij} = {\mathbb{1}}_{\left\lbrack  j = {\operatorname{argmax}}_{{j}^{\prime }}\left( {\mathbf{P}}_{i{j}^{\prime }}\right) \right\rbrack  }$，其中 argmax 运算符作用于 $1 \leq  {j}^{\prime } \leq  m$（即来自单个文档 $D$ 的词元），${\mathbb{1}}_{\left\lbrack  *\right\rbrack  }$ 是一个指示函数。然后，${f}_{\text{ColBERT }}\left( {Q,D}\right)$ 定义如下：

$$
{f}_{\text{ColBERT }}\left( {Q,D}\right)  = \frac{1}{n}\mathop{\sum }\limits_{{i = 1}}^{n}\mathop{\sum }\limits_{{j = 1}}^{m}{\mathbf{A}}_{ij}{\mathbf{P}}_{ij} = \frac{1}{n}\mathop{\sum }\limits_{{i = 1}}^{n}\mathop{\max }\limits_{{1 \leq  j \leq  m}}{\mathbf{q}}_{i}^{\top }{\mathbf{d}}_{j}. \tag{1}
$$

<!-- Media -->

<!-- figureText: ColBERT inference (b) Gathering (c) Scoring (training) OOD new Query 2000 2000 OOD new FLOPs/query (Scoring) (b) Scoring Score, $4,{000} \times$ Cheaper Recall@100 on BEIR XTR (a) Token Retrieval Query OOOO 2000 XTR training & inference (a) Token Retrieval ${f}_{\mathrm{{XTR}}}$ -->

<img src="https://cdn.noedgeai.com/01958a7f-8417-7506-8c2f-6473566d79b1_2.jpg?x=396&y=203&w=1004&h=640&r=0"/>

Figure 1: Overview of XTR. ColBERT has the three-stage inference combining (a) the token retrieval, (b) the gathering and (c) the scoring stages (§2.2). XTR leverages the token retrieval for both training and inference. XTR efficiently obtains the score of each candidate document by applying ${f}_{\mathrm{{XTR}}}$ (or ${f}_{{\mathrm{{XTR}}}^{\prime }}$ ) on the retrieved tokens,completely removing the gathering stage (§3.2).

图1：XTR概述。ColBERT（科尔伯特）采用三阶段推理，包括（a）词元检索、（b）收集和（c）评分阶段（§2.2）。XTR在训练和推理中均利用词元检索。XTR通过对检索到的词元应用${f}_{\mathrm{{XTR}}}$（或${f}_{{\mathrm{{XTR}}}^{\prime }}$）来高效获取每个候选文档的分数，完全省去了收集阶段（§3.2）。

<!-- Media -->

Here,we include the normalizer $n$ ,which was not included in the original sum-of-max,as it stabilizes training while not affecting the ranking during inference. After computing the query-document similarity, multi-vector retrieval models are typically trained with a cross-entropy loss over in-batch negatives [Santhanam et al.,2022b,Qian et al.,2022]. Specifically,given a positive document ${D}^{ + }$ for $Q$ and a set of mini-batch documents ${D}_{1 : B} = \left\lbrack  {{D}_{1},\ldots ,{D}_{B}}\right\rbrack$ where ${D}^{ + } \in  {D}_{1 : B}$ ,they minimize the cross-entropy loss defined as: ${\mathcal{L}}_{\mathrm{{CE}}} =  - \log \frac{\exp f\left( {Q,{D}^{ + }}\right) }{\mathop{\sum }\limits_{{b = 1}}^{B}\exp f\left( {Q,{D}_{b}}\right) }$ .

在这里，我们纳入了归一化器 $n$，而原始的最大和（sum-of-max）中并未包含该归一化器，因为它能在不影响推理阶段排序的情况下稳定训练。在计算查询 - 文档相似度后，多向量检索模型通常使用针对批次内负样本的交叉熵损失进行训练 [Santhanam 等人，2022b；Qian 等人，2022]。具体而言，给定 $Q$ 的正文档 ${D}^{ + }$ 以及一组小批量文档 ${D}_{1 : B} = \left\lbrack  {{D}_{1},\ldots ,{D}_{B}}\right\rbrack$（其中 ${D}^{ + } \in  {D}_{1 : B}$），它们会最小化定义为 ${\mathcal{L}}_{\mathrm{{CE}}} =  - \log \frac{\exp f\left( {Q,{D}^{ + }}\right) }{\mathop{\sum }\limits_{{b = 1}}^{B}\exp f\left( {Q,{D}_{b}}\right) }$ 的交叉熵损失。

### 2.2 Three-stage inference of Multi-vector Retrieval

### 2.2 多向量检索的三阶段推理

Unlike dual encoder models, finding the maximum scoring document-the document that maximizes eq. (1) - cannot be directly handled by MIPS as the scoring function uses a non-linear, sum-of-max operation. Instead, a multi-vector retrieval model typically takes the following steps for the inference. 1) Token Retrieval: for each of the $n$ query token vectors,it first retrieves ${k}^{\prime }$ document token vectors, which is simply used to form initial candidate document set by taking the union of source documents of retrieved tokens. The total number of candidate documents is up to $n{k}^{\prime }$ if each token is coming from a unique document ${}^{2}{\left| 2\rangle \text{ Gathering: since the scoring function eq. (1) requires the computation}\right| }^{2}$ over all document tokens, multi-vector models need to load all of the token vectors of the candidate documents. To optimize the loading process, a RAM-based index is often employed. 3) Scoring: to provide final ranks of candidate documents, multi-vector models score all the candidate documents with eq. (1). This stage is also called refinement. Note that the training of typical multi-vector models only takes care of the scoring stage with mini-batch documents. Finally,top- $k$ documents are returned based on the computed scores. The three-stage inference is illustrated in the top of Figure 1

与双编码器模型不同，由于评分函数使用了非线性的最大和运算，因此无法直接使用最大内积搜索（MIPS）来找出得分最高的文档（即最大化公式 (1) 的文档）。相反，多向量检索模型通常按以下步骤进行推理。1) Token 检索：对于 $n$ 个查询 Token 向量中的每一个，它首先检索 ${k}^{\prime }$ 个文档 Token 向量，这些向量仅用于通过取检索到的 Token 所在源文档的并集来形成初始候选文档集。如果每个 Token 都来自一个唯一的文档，那么候选文档的总数最多为 $n{k}^{\prime }$。${}^{2}{\left| 2\rangle \text{ Gathering: since the scoring function eq. (1) requires the computation}\right| }^{2}$ 在所有文档 Token 中，多向量模型需要加载候选文档的所有 Token 向量。为了优化加载过程，通常会使用基于随机存取存储器（RAM）的索引。3) 评分：为了提供候选文档的最终排名，多向量模型使用公式 (1) 对所有候选文档进行评分。这个阶段也称为细化阶段。请注意，典型的多向量模型的训练仅处理小批量文档的评分阶段。最后，根据计算出的分数返回前 $k$ 个文档。图 1 的顶部展示了这个三阶段推理过程

## 3 XTR: Contextualized Token Retriever

## 3 XTR：上下文感知的Token检索器

Unlike existing multi-vector models that follow the retrieve-gather-score stages, XTR directly scores documents utilizing the tokens retrieved from the token retrieval stage. In this section, we start by showing why the existing cross entropy loss with the sum-of-max scoring function would fail on the first-stage token retrieval. Then, we introduce simple but important modifications for XTR.

与现有的遵循检索 - 收集 - 评分阶段的多向量模型不同，XTR直接利用从Token检索阶段检索到的Token对文档进行评分。在本节中，我们首先说明为什么现有的带有最大和评分函数的交叉熵损失在第一阶段的Token检索中会失效。然后，我们介绍对XTR进行的简单但重要的修改。

---

<!-- Footnote -->

${}^{2}$ In fact,each candidate document of a T5-based ColBERT is retrieved by 1.48 tokens per on average. meaning that the most of the candidate documents are unique.

${}^{2}$ 事实上，基于T5的ColBERT模型平均每个候选文档由1.48个Token检索得到，这意味着大多数候选文档是唯一的。

<!-- Footnote -->

---

Given a positive document ${D}^{ + }$ and a set of negative documents ${D}_{1 : r}^{ - } = \left\lbrack  {{D}_{1}^{ - },\ldots ,{D}_{r}^{ - }}\right\rbrack$ for a query $Q$ ,the first-stage token retrieval needs to retrieve the tokens of ${D}^{ + }$ ,but not the tokens of negative documents. However, the following example shows that the sum-of-max operator used by ColBERT is not specifically designed to retrieve tokens of relevant documents.

对于一个查询$Q$，给定一个正文档${D}^{ + }$和一组负文档${D}_{1 : r}^{ - } = \left\lbrack  {{D}_{1}^{ - },\ldots ,{D}_{r}^{ - }}\right\rbrack$，第一阶段的Token检索需要检索${D}^{ + }$的Token，而不是负文档的Token。然而，下面的例子表明，ColBERT使用的最大和运算符并非专门为检索相关文档的Token而设计。

Failure case Assume that ${f}_{\text{ColBERT }}\left( {Q,{D}^{ + }}\right)  = {0.8}$ where all the individual max token similarity (i.e., ${\mathbf{q}}_{i}^{\top }{\mathbf{d}}_{j}^{ + }$ where ${\mathbf{A}}_{ij} = 1$ ) is 0.8 . On the other hand,assume ${f}_{\text{ColBERT }}\left( {Q,{D}^{ - }}\right)  = {0.2}$ for all ${D}^{ - } \in  {D}_{1 : r}^{ - }$ where each ${D}^{ - }$ has a highly peaked token similarity greater than 0.8 but others close to zero (i.e.,there exists ${\mathbf{q}}_{i}^{\top }{\mathbf{d}}_{j}^{ - } > {0.8}$ where ${\mathbf{A}}_{ij} = 1$ while other ${\mathbf{q}}_{i}^{\top }{\mathbf{d}}_{j}^{ - } \rightarrow  0$ ). Since the sum-of-max operator only cares about the document-level scores, the cross entropy loss would be close to zero during training ${}^{3}$ However,for each of $n$ query tokens,if there exists at least one negative document token that has a high token similarity greater than 0.8,the token retrieval with top- ${k}^{\prime } = 1$ would fail to retrieve any tokens of ${D}^{ + }$ . As a result,multi-vector retrieval model with the sum-of-max operator will not be able to lower the high scores of some negative tokens. Figure 2 shows that the sum-of-max training causes many document tokens to have unreasonably high scores regardless of their actual relevance to the query tokens.

失败案例 假设${f}_{\text{ColBERT }}\left( {Q,{D}^{ + }}\right)  = {0.8}$，其中所有单个最大词元相似度（即${\mathbf{q}}_{i}^{\top }{\mathbf{d}}_{j}^{ + }$，其中${\mathbf{A}}_{ij} = 1$）为 0.8。另一方面，假设对于所有${D}^{ - } \in  {D}_{1 : r}^{ - }$有${f}_{\text{ColBERT }}\left( {Q,{D}^{ - }}\right)  = {0.2}$，其中每个${D}^{ - }$的词元相似度具有很高的峰值且大于 0.8，但其他的接近零（即，存在${\mathbf{q}}_{i}^{\top }{\mathbf{d}}_{j}^{ - } > {0.8}$，其中${\mathbf{A}}_{ij} = 1$，而其他${\mathbf{q}}_{i}^{\top }{\mathbf{d}}_{j}^{ - } \rightarrow  0$）。由于最大求和算子只关注文档级别的分数，因此在训练期间交叉熵损失将接近零${}^{3}$。然而，对于$n$中的每个查询词元，如果至少存在一个负文档词元的词元相似度大于 0.8，那么使用前${k}^{\prime } = 1$的词元检索将无法检索到${D}^{ + }$的任何词元。结果，使用最大求和算子的多向量检索模型将无法降低一些负词元的高分。图 2 显示，最大求和训练导致许多文档词元的分数高得不合理，而不管它们与查询词元的实际相关性如何。

<!-- Media -->

<!-- figureText: 0.00 -->

<img src="https://cdn.noedgeai.com/01958a7f-8417-7506-8c2f-6473566d79b1_3.jpg?x=1018&y=376&w=426&h=308&r=0"/>

Figure 2: Density histogram of 4,000 token retrieval scores (cosine similarity). Training with ${f}_{\text{ColBERT }}$ (T5-ColBERT; [4] causes many document tokens to have extremely high scores regardless of their actual relevance with respect to the input query tokens. XTR mitigates this problem with a better training objective.

图2：4000个Token检索分数（余弦相似度）的密度直方图。使用${f}_{\text{ColBERT }}$进行训练（T5-ColBERT；[4]）会导致许多文档Token的得分极高，而不论它们与输入查询Token的实际相关性如何。XTR通过更好的训练目标缓解了这个问题。

<!-- Media -->

### 3.1 In-Batch Token Retrieval

### 3.1 批次内Token检索

To train multi-vector retrieval models to directly retrieve tokens of relevant documents, we simulate the token retrieval stage during training. This can be simply achieved by employing a different alignment strategy $\widehat{\mathbf{A}}$ . Specifically,we set the alignment ${\widehat{\mathbf{A}}}_{ij} = {\mathbb{1}}_{\left\lbrack  j \in  \operatorname{top} - {k}_{{j}^{\prime }}\left( {\mathbf{P}}_{i{j}^{\prime }}\right) \right\rbrack  }$ where the top- $k$ operator is applied over $1 \leq  {j}^{\prime } \leq  {mB}$ (i.e.,tokens from $B$ mini-batch documents) returning the indices of $k$ largest values. During training,we use a hyperparameter ${k}_{\text{train }}$ for the top- $k$ operator. Then, we simply modify eq. (1) as follows:

为了训练多向量检索模型以直接检索相关文档的标记，我们在训练期间模拟标记检索阶段。这可以通过采用不同的对齐策略 $\widehat{\mathbf{A}}$ 简单实现。具体而言，我们设置对齐方式 ${\widehat{\mathbf{A}}}_{ij} = {\mathbb{1}}_{\left\lbrack  j \in  \operatorname{top} - {k}_{{j}^{\prime }}\left( {\mathbf{P}}_{i{j}^{\prime }}\right) \right\rbrack  }$，其中 top- $k$ 运算符应用于 $1 \leq  {j}^{\prime } \leq  {mB}$（即来自 $B$ 小批量文档的标记），返回 $k$ 个最大值的索引。在训练期间，我们为 top- $k$ 运算符使用一个超参数 ${k}_{\text{train }}$。然后，我们简单地将等式 (1) 修改如下：

$$
{f}_{\mathrm{{XTR}}}\left( {Q,D}\right)  = \frac{1}{Z}\mathop{\sum }\limits_{{i = 1}}^{n}\mathop{\max }\limits_{{1 \leq  j \leq  m}}{\widehat{\mathbf{A}}}_{ij}{\mathbf{q}}_{i}^{\top }{\mathbf{d}}_{j}. \tag{2}
$$

The intuition is that we consider the token similarities within $D$ only when they are high enough to be retrieved within top- ${k}_{\text{train }}$ from a mini-batch. Here,we use a normalizer $Z = \left| \left\{  {i \mid  \exists j,\text{ s.t. }{\widehat{\mathbf{A}}}_{ij} > \underline{0}}\right\}  \right|$ , which is essentially the number of query tokens that retrieved at least one document token of ${D}^{4}$ . If all ${\widehat{\mathbf{A}}}_{ij} = 0$ ,we clip $Z$ to a small number and ${f}_{\mathrm{{XTR}}}\left( {Q,D}\right)$ becomes 0 . As a result,our model cannot assign a high token similarity to negative documents as it blocks tokens of positive documents to be retrieved. With the previous failure case where ${f}_{\text{ColBERT }}$ assigned a high score on ${D}^{ + }$ even though it cannot be retrieved,our similarity function incurs a high loss as ${f}_{\mathrm{{XTR}}}\left( {Q,{D}^{ + }}\right)  = 0$ during training (since tokens of ${D}^{ + }$ were not retrieved). For training,we use the same cross entropy loss defined in §2.1 with our new scoring function. Note that the training data only contains document-level annotations, but XTR encourages important tokens from positive documents to be retrieved.

其直觉是，我们仅当$D$内的标记相似度足够高，能够从一个小批量中在前${k}_{\text{train }}$个中被检索到时，才会考虑这些相似度。在这里，我们使用一个归一化因子$Z = \left| \left\{  {i \mid  \exists j,\text{ s.t. }{\widehat{\mathbf{A}}}_{ij} > \underline{0}}\right\}  \right|$，它本质上是能检索到${D}^{4}$中至少一个文档标记的查询标记的数量。如果所有的${\widehat{\mathbf{A}}}_{ij} = 0$，我们将$Z$裁剪为一个小的数值，并且${f}_{\mathrm{{XTR}}}\left( {Q,D}\right)$变为0。因此，我们的模型不会给负样本文档赋予高的标记相似度，因为它会阻止正样本文档的标记被检索到。在之前的失败案例中，即使${D}^{ + }$无法被检索到，${f}_{\text{ColBERT }}$仍给它赋予了高分，我们的相似度函数在训练期间会产生高损失，因为${f}_{\mathrm{{XTR}}}\left( {Q,{D}^{ + }}\right)  = 0$（因为${D}^{ + }$的标记未被检索到）。在训练时，我们使用第2.1节中定义的相同的交叉熵损失，并结合我们新的评分函数。请注意，训练数据仅包含文档级别的标注，但XTR鼓励从正样本文档中检索重要的标记。

### 3.2 Scoring Documents using Retrieved Tokens

### 3.2 使用检索到的Token对文档进行评分

During inference,multi-vector retrieval models first have a set of candidate documents ${\widehat{D}}_{1 : C}$ from the token retrieval stage:

在推理过程中，多向量检索模型首先会从Token检索阶段得到一组候选文档 ${\widehat{D}}_{1 : C}$：

$$
{\widehat{D}}_{1 : C} = \left\{  {\widehat{D} \mid  {d}_{j} \in  \widehat{D} \land  {d}_{j} \in  \operatorname{top} - {k}^{\prime }\left( {\mathbf{q}}_{ * }\right) }\right\}  . \tag{3}
$$

Here,top- ${k}^{\prime }\left( {\mathbf{q}}_{ * }\right)$ is a union of top- ${k}^{\prime }$ document tokens (from the entire corpus) based on the inner product scores with each query vector (i.e., ${\mathbf{q}}^{\top }\mathbf{d}$ ). Given the $n$ query token vectors,there are $C$ $\left( { \leq  n{k}^{\prime }}\right)$ candidate documents. Previous methods load the entire token vectors of each document and compute eq. (1) for every query and candidate document pair,which takes $\mathcal{O}\left( {{n}^{2}{k}^{\prime }\bar{m}d}\right)$ computation per query $\left( {\bar{m} = \text{average document length}}\right)$ . Instead,we propose to score the documents solely using the retrieved token similarity. This significantly reduces the computational cost for the scoring stage since re-using the token retrieval scores removes computing redundant inner products and unnecessary (non-max) inner products. Furthermore, the expensive gathering stage (which requires loading all the document token vectors for computing eq. (1) can be removed completely. Unlike previous work [Macdonald and Tonellotto, 2021] that leverages token retrieval to sort first-stage candidate documents before the scoring stage, we aim to directly provide the final scores of documents.

在此，top - ${k}^{\prime }\left( {\mathbf{q}}_{ * }\right)$ 是基于与每个查询向量的内积得分（即 ${\mathbf{q}}^{\top }\mathbf{d}$ ），由 top - ${k}^{\prime }$ 文档词元（来自整个语料库）构成的并集。给定 $n$ 个查询词元向量，有 $C$ $\left( { \leq  n{k}^{\prime }}\right)$ 个候选文档。先前的方法会加载每个文档的所有词元向量，并为每对查询和候选文档计算公式 (1)，每个查询 $\left( {\bar{m} = \text{average document length}}\right)$ 需要 $\mathcal{O}\left( {{n}^{2}{k}^{\prime }\bar{m}d}\right)$ 次计算。相反，我们提议仅使用检索到的词元相似度对文档进行评分。由于复用词元检索得分避免了计算冗余的内积和不必要的（非最大值）内积，这显著降低了评分阶段的计算成本。此外，昂贵的数据收集阶段（需要加载所有文档词元向量来计算公式 (1)）可以完全省去。与先前的工作 [Macdonald 和 Tonellotto，2021] 不同，该工作在评分阶段之前利用词元检索对第一阶段的候选文档进行排序，我们的目标是直接给出文档的最终得分。

---

<!-- Footnote -->

${}^{3}$ Indeed,our derivative analysis in Appendix A shows that the token-level similarity would not change if the document-level scores are already well discriminated.

${}^{3}$ 实际上，我们在附录A中的导数分析表明，如果文档级得分已经能够很好地区分，那么标记级相似度将不会改变。

${}^{4}$ We tried different normalizers such as $n$ and found that $Z$ works the best while stabilizing the training.

${}^{4}$ 我们尝试了不同的归一化方法，如 $n$，发现 $Z$ 在稳定训练的同时效果最佳。

<!-- Footnote -->

---

<!-- Media -->

<img src="https://cdn.noedgeai.com/01958a7f-8417-7506-8c2f-6473566d79b1_4.jpg?x=351&y=207&w=1084&h=227&r=0"/>

Figure 3: Comparison of ${f}_{\text{ColBERT }}$ in eq. (1) and ${f}_{{\mathrm{{XTR}}}^{\prime }}$ in eq. (4). Assume that ${D}_{a}$ and ${D}_{b}$ were selected as initial candidate documents from the token retrieval stage. ${f}_{\text{ColBERT }}$ loads all token vectors of ${D}_{a}$ and ${D}_{b}$ and exhaustively recomputes pairwise token similarity to obtain the max values (red boxes). On the other hand, ${f}_{{\mathrm{{XTR}}}^{\prime }}$ does not load any token vectors and reuses retrieval scores from the first-stage token retrieval. Assume that, with the top-2 token retrieval results, the first query token retrieved each max score of ${D}_{a}$ and ${D}_{b}$ ,but the second query token retrieved two tokens only from ${D}_{a}$ but not ${D}_{b}$ . We impute the missing similarity $m$ for ${D}_{b}$ (denoted as yellow dashed box) by finding its upper bound using the top-2 score (denoted as ${s}_{2}$ ) of the second query token (i.e., $m \leq  {s}_{2} \leq  {s}_{1}$ ).

图3：公式(1)中的${f}_{\text{ColBERT }}$与公式(4)中的${f}_{{\mathrm{{XTR}}}^{\prime }}$的比较。假设从词元检索阶段选择了${D}_{a}$和${D}_{b}$作为初始候选文档。${f}_{\text{ColBERT }}$加载${D}_{a}$和${D}_{b}$的所有词元向量，并详尽地重新计算词元对之间的相似度以获得最大值（红色框）。另一方面，${f}_{{\mathrm{{XTR}}}^{\prime }}$不加载任何词元向量，而是复用第一阶段词元检索的检索分数。假设在top - 2词元检索结果中，第一个查询词元分别检索到了${D}_{a}$和${D}_{b}$的最大分数，但第二个查询词元仅从${D}_{a}$中检索到两个词元，而未从${D}_{b}$中检索到。我们通过使用第二个查询词元的top - 2分数（表示为${s}_{2}$）找到其上限（即$m \leq  {s}_{2} \leq  {s}_{1}$）来估算${D}_{b}$缺失的相似度$m$（用黄色虚线框表示）。

<table><tr><td/><td>Scoring</td><td>Estimated FLOPs/query</td><td>Setting</td></tr><tr><td>${f}_{\text{ColBERT }}$ ${f}_{{\mathrm{{XTR}}}^{\prime }}$</td><td>${n}^{2}{k}^{\prime }\left( {2\bar{m}d + \bar{m} + 1}\right)$ ${n}^{2}{k}^{\prime }\left( {\bar{r} + 1}\right)$</td><td>${0.36} \times  {10}^{9}$ ${0.09} \times  {10}^{6}$</td><td>$M = 3 \times  {10}^{9},n = {16},d = {128}$ , ${k}^{\prime } = {100},\bar{m} = {55},\bar{r} = {2.5}$</td></tr></table>

<table><tbody><tr><td></td><td>评分</td><td>预估每秒浮点运算次数/查询（FLOPs/query）</td><td>设置</td></tr><tr><td>${f}_{\text{ColBERT }}$ ${f}_{{\mathrm{{XTR}}}^{\prime }}$</td><td>${n}^{2}{k}^{\prime }\left( {2\bar{m}d + \bar{m} + 1}\right)$ ${n}^{2}{k}^{\prime }\left( {\bar{r} + 1}\right)$</td><td>${0.36} \times  {10}^{9}$ ${0.09} \times  {10}^{6}$</td><td>$M = 3 \times  {10}^{9},n = {16},d = {128}$ , ${k}^{\prime } = {100},\bar{m} = {55},\bar{r} = {2.5}$</td></tr></tbody></table>

Table 1: FLOPs comparison of ColBERT and XTR for the scoring stage. XTR only adds minimal complexity for scoring each candidate document. The setting is derived from MS MARCO.

表1：ColBERT和XTR在评分阶段的浮点运算次数（FLOPs）比较。XTR在对每个候选文档进行评分时仅增加了极小的复杂度。该设置源自MS MARCO数据集。

<!-- Media -->

Missing similarity imputation During inference,we retrieve ${k}^{\prime }$ document tokens for each of $n$ query tokens. Assume that each document token belongs to a unique document,providing $C = n{k}^{\prime }$ candidate documents in total. This leaves us with a single token similarity to score each document in the absence of the gathering stage. However, during training-either with eq. (1) or eq. (2) - each positive document has up to $n$ (max) token similarities to average,which mostly converges to $n$ as training proceeds. Hence, during inference, we impute the missing similarity for each query token treating each of candidate documents as if it were positive with $n$ token similarities.

缺失相似度插补 在推理过程中，我们为$n$个查询词元中的每个词元检索${k}^{\prime }$个文档词元。假设每个文档词元都属于一个唯一的文档，那么总共会有$C = n{k}^{\prime }$个候选文档。在没有收集阶段的情况下，这使得我们只能用单个词元相似度来对每个文档进行评分。然而，在训练过程中（无论是使用公式(1)还是公式(2)），每个正样本文档最多有$n$个（最大）词元相似度需要求平均值，随着训练的进行，这个值大多会收敛到$n$。因此，在推理过程中，我们将每个候选文档视为具有$n$个词元相似度的正样本，对每个查询词元的缺失相似度进行插补。

For every candidate document $\widehat{D}$ ,we first define the following scoring function for the inference:

对于每个候选文档$\widehat{D}$，我们首先为推理定义以下评分函数：

$$
{f}_{{\mathrm{{XTR}}}^{\prime }}\left( {Q,\widehat{D}}\right)  = \frac{1}{n}\mathop{\sum }\limits_{{i = 1}}^{n}\mathop{\max }\limits_{{1 \leq  j \leq  m}}\left\lbrack  {{\widehat{\mathbf{A}}}_{ij}{\mathbf{q}}_{i}^{\top }{\mathbf{d}}_{j} + \left( {1 - {\widehat{\mathbf{A}}}_{ij}}\right) {m}_{i}}\right\rbrack  . \tag{4}
$$

This is similar to eq. (2),but introduces ${m}_{i} \in  \mathbb{R}$ ,which estimates the missing similarity for each ${q}_{i}$ . $\widehat{\mathbf{A}}$ is defined similar to the one described in eq. (2) except that it uses ${k}^{\prime }$ for the top- $k$ operator. Each ${q}_{i}$ would take the missing similarity ${m}_{i}$ as the maximum value if ${\widehat{\mathbf{A}}}_{i * } = 0$ and ${m}_{i} \geq  0$ . Importantly, ${f}_{{\mathrm{{XTR}}}^{\prime }}$ removes the need of recomputing any ${\mathbf{q}}_{i}^{\top }{\mathbf{d}}_{j}$ since when ${\widehat{\mathbf{A}}}_{ij} = 1$ we already know the retrieval score from the token retrieval stage,and when ${\widehat{\mathbf{A}}}_{ij} = 0$ we simply don’t need to compute it as ${\widehat{\mathbf{A}}}_{ij}{\mathbf{q}}_{i}^{\top }{\mathbf{d}}_{j} = 0$ . Note that when every ${\widehat{\mathbf{A}}}_{ij} = 1$ ,the equation becomes the sum-of-max operator. On the other hand,when no document tokens of $\widehat{D}$ were retrieved for ${q}_{i}$ (i.e., ${\widehat{\mathbf{A}}}_{i * } = 0$ ),we fall back to the imputed score ${m}_{i}$ ,which provides an approximated sum-of-max result. In fact,we can find the upper bound of the missing similarity. For every token retrieval with ${\mathbf{q}}_{i}$ , the missing similarity of the query token for $\widehat{D}$ will be upper bounded by its last top- ${k}^{\prime }$ score. Specifically,for each query token ${q}_{i}$ ,we have the following top- ${k}^{\prime }$ token similarity during inference: $\left\lbrack  {{\mathbf{q}}_{i}^{\top }{\mathbf{d}}_{\left( 1\right) },\ldots {\mathbf{q}}_{i}^{\top }{\mathbf{d}}_{\left( {k}^{\prime }\right) }}\right\rbrack$ . Here,each ${\mathbf{d}}_{\left( *\right) }$ could come from a different document. Since the missing similarity would have a score less than equal to the score of the last retrieved token, we know that ${m}_{i} \leq  {\mathbf{q}}_{i}^{\top }{\mathbf{d}}_{\left( {k}^{\prime }\right) }$ . With a larger ${k}^{\prime }$ ,the upper bound becomes tighter. In our experiments,we show that simply choosing ${m}_{i} = {\mathbf{q}}_{i}^{\top }{\mathbf{d}}_{\left( {k}^{\prime }\right) }$ works well especially when a model is trained with ${f}_{\mathrm{{XTR}}}{}^{5}$ While we also tried more complicated imputation methods based on regression, our method was competitive enough despite its simplicity. The imputation process is illustrated in Figure 3

这与公式 (2) 类似，但引入了 ${m}_{i} \in  \mathbb{R}$，它用于估计每个 ${q}_{i}$ 的缺失相似度。$\widehat{\mathbf{A}}$ 的定义与公式 (2) 中描述的类似，只是它在 top- $k$ 运算符中使用了 ${k}^{\prime }$。如果 ${\widehat{\mathbf{A}}}_{i * } = 0$ 且 ${m}_{i} \geq  0$，每个 ${q}_{i}$ 会将缺失相似度 ${m}_{i}$ 作为最大值。重要的是，${f}_{{\mathrm{{XTR}}}^{\prime }}$ 消除了重新计算任何 ${\mathbf{q}}_{i}^{\top }{\mathbf{d}}_{j}$ 的需求，因为当 ${\widehat{\mathbf{A}}}_{ij} = 1$ 时，我们已经从词元检索阶段得知了检索分数，而当 ${\widehat{\mathbf{A}}}_{ij} = 0$ 时，由于 ${\widehat{\mathbf{A}}}_{ij}{\mathbf{q}}_{i}^{\top }{\mathbf{d}}_{j} = 0$，我们根本不需要计算它。请注意，当每个 ${\widehat{\mathbf{A}}}_{ij} = 1$ 时，该方程就变成了最大求和运算符。另一方面，当没有为 ${q}_{i}$ 检索到 $\widehat{D}$ 的文档词元时（即 ${\widehat{\mathbf{A}}}_{i * } = 0$），我们会采用估算分数 ${m}_{i}$，它提供了一个近似的最大求和结果。实际上，我们可以找到缺失相似度的上界。对于每次使用 ${\mathbf{q}}_{i}$ 进行的词元检索，$\widehat{D}$ 对于查询词元的缺失相似度将以其最后一个 top- ${k}^{\prime }$ 分数为上界。具体来说，对于每个查询词元 ${q}_{i}$，我们在推理过程中有以下 top- ${k}^{\prime }$ 词元相似度：$\left\lbrack  {{\mathbf{q}}_{i}^{\top }{\mathbf{d}}_{\left( 1\right) },\ldots {\mathbf{q}}_{i}^{\top }{\mathbf{d}}_{\left( {k}^{\prime }\right) }}\right\rbrack$。这里，每个 ${\mathbf{d}}_{\left( *\right) }$ 可能来自不同的文档。由于缺失相似度的分数小于等于最后检索到的词元的分数，我们知道 ${m}_{i} \leq  {\mathbf{q}}_{i}^{\top }{\mathbf{d}}_{\left( {k}^{\prime }\right) }$。${k}^{\prime }$ 越大，上界就越紧密。在我们的实验中，我们表明，简单地选择 ${m}_{i} = {\mathbf{q}}_{i}^{\top }{\mathbf{d}}_{\left( {k}^{\prime }\right) }$ 效果很好，特别是当模型使用 ${f}_{\mathrm{{XTR}}}{}^{5}$ 进行训练时。虽然我们也尝试了基于回归的更复杂的估算方法，但我们的方法尽管简单，却具有足够的竞争力。估算过程如图 3 所示

<!-- Media -->

<table><tr><td/><td>MS</td><td>AR</td><td>TO</td><td>FE</td><td>CF</td><td>SF</td><td>CV</td><td>NF</td><td>NQ</td><td>HQ</td><td>FQ</td><td>SD</td><td>DB</td><td>QU</td><td>Avg.</td></tr><tr><td colspan="16">One Retriever per Domain</td></tr><tr><td>GenQ</td><td>40.8</td><td>49.3</td><td>18.2</td><td>66.9</td><td>17.5</td><td>64.4</td><td>61.9</td><td>31.9</td><td>35.8</td><td>53.4</td><td>30.8</td><td>14.3</td><td>32.8</td><td>83.0</td><td>43.1</td></tr><tr><td>${\mathrm{{PTR}}}_{\text{retriever }}$</td><td>-</td><td>58.8</td><td>25.6</td><td>76.2</td><td>23.5</td><td>63.8</td><td>70.2</td><td>33.7</td><td>45.6</td><td>61.7</td><td>43.0</td><td>18.3</td><td>34.4</td><td>87.5</td><td>49.4</td></tr><tr><td colspan="16">One Retriever for All</td></tr><tr><td>BM25</td><td>22.8</td><td>31.5</td><td>36.7</td><td>75.3</td><td>21.3</td><td>66.5</td><td>65.6</td><td>32.5</td><td>32.9</td><td>60.3</td><td>23.6</td><td>15.8</td><td>31.3</td><td>78.9</td><td>44.0</td></tr><tr><td>ColBERT</td><td>40.1</td><td>23.3</td><td>20.2</td><td>77.1</td><td>18.4</td><td>67.1</td><td>67.7</td><td>30.5</td><td>52.4</td><td>59.3</td><td>31.7</td><td>14.5</td><td>39.2</td><td>85.4</td><td>45.1</td></tr><tr><td>${\mathrm{{GTR}}}_{\text{base }}$</td><td>42.0</td><td>51.1</td><td>21.5</td><td>66.0</td><td>24.1</td><td>60.0</td><td>53.9</td><td>30.8</td><td>49.5</td><td>53.5</td><td>34.9</td><td>14.9</td><td>39.2</td><td>88.1</td><td>45.2</td></tr><tr><td>T5-ColBERT ${}_{\text{base}}$</td><td>45.6</td><td>28.8</td><td>31.1</td><td>72.4</td><td>18.1</td><td>70.4</td><td>68.3</td><td>34.0</td><td>52.2</td><td>61.7</td><td>33.4</td><td>14.1</td><td>41.6</td><td>82.3</td><td>46.8</td></tr><tr><td>$\mathbf{{XT}{R}_{base}}$</td><td>45.0</td><td>40.7</td><td>31.3</td><td>73.7</td><td>20.7</td><td>71.0</td><td>73.6</td><td>34.0</td><td>53.0</td><td>64.7</td><td>34.7</td><td>14.5</td><td>40.9</td><td>86.1</td><td>49.1</td></tr><tr><td>${\text{Splade}}_{v2} \uparrow   \uparrow$</td><td>43.3</td><td>47.9</td><td>27.2</td><td>78.6</td><td>23.5</td><td>69.3</td><td>71.0</td><td>33.4</td><td>52.1</td><td>68.4</td><td>33.6</td><td>15.8</td><td>43.5</td><td>83.8</td><td>49.9</td></tr><tr><td>ColBERT ${}_{\mathrm{v}2}{}^{\bullet  \bullet  }$</td><td>-</td><td>46.3</td><td>26.3</td><td>78.5</td><td>17.6</td><td>69.3</td><td>73.8</td><td>33.8</td><td>56.2</td><td>66.7</td><td>35.6</td><td>15.4</td><td>44.6</td><td>85.2</td><td>49.9</td></tr><tr><td>${\mathrm{{GTR}}}_{\mathrm{{xxl}}}$</td><td>44.2</td><td>54.0</td><td>23.3</td><td>74.0</td><td>26.7</td><td>66.2</td><td>50.1</td><td>34.2</td><td>56.8</td><td>59.9</td><td>46.7</td><td>16.1</td><td>40.8</td><td>89.2</td><td>49.1</td></tr><tr><td>T5-ColBERT ${}_{\mathrm{{xx}}1}$</td><td>47.3</td><td>33.8</td><td>31.0</td><td>74.2</td><td>19.7</td><td>73.1</td><td>75.8</td><td>35.2</td><td>60.5</td><td>65.2</td><td>43.5</td><td>17.1</td><td>45.0</td><td>86.0</td><td>50.8</td></tr><tr><td>${\mathbf{{XTR}}}_{\mathbf{{xxl}}}$</td><td>46.6</td><td>44.2</td><td>30.9</td><td>77.0</td><td>24.5</td><td>74.3</td><td>78.9</td><td>35.3</td><td>60.9</td><td>66.2</td><td>43.8</td><td>17.1</td><td>44.3</td><td>88.1</td><td>52.7</td></tr></table>

<table><tbody><tr><td></td><td>女士（Ms）</td><td>增强现实（Augmented Reality，AR）</td><td>到；向（To）</td><td>铁（Fe，化学元素符号）</td><td>比较（Compare，CF可能是缩写）</td><td>科幻小说（Science Fiction，SF）</td><td>简历（CV）</td><td>未找到（NF）</td><td>正常查询（NQ）</td><td>高画质（HQ）</td><td>模糊查询（FQ）</td><td>标准清晰度（SD）</td><td>分贝（DB）</td><td>查询（QU）</td><td>平均值（Avg.）</td></tr><tr><td colspan="16">每个领域一个检索器</td></tr><tr><td>通用问题（GenQ）</td><td>40.8</td><td>49.3</td><td>18.2</td><td>66.9</td><td>17.5</td><td>64.4</td><td>61.9</td><td>31.9</td><td>35.8</td><td>53.4</td><td>30.8</td><td>14.3</td><td>32.8</td><td>83.0</td><td>43.1</td></tr><tr><td>${\mathrm{{PTR}}}_{\text{retriever }}$</td><td>-</td><td>58.8</td><td>25.6</td><td>76.2</td><td>23.5</td><td>63.8</td><td>70.2</td><td>33.7</td><td>45.6</td><td>61.7</td><td>43.0</td><td>18.3</td><td>34.4</td><td>87.5</td><td>49.4</td></tr><tr><td colspan="16">一个检索器用于所有领域</td></tr><tr><td>BM25（原文未变，为专业术语）</td><td>22.8</td><td>31.5</td><td>36.7</td><td>75.3</td><td>21.3</td><td>66.5</td><td>65.6</td><td>32.5</td><td>32.9</td><td>60.3</td><td>23.6</td><td>15.8</td><td>31.3</td><td>78.9</td><td>44.0</td></tr><tr><td>科尔伯特（ColBERT）</td><td>40.1</td><td>23.3</td><td>20.2</td><td>77.1</td><td>18.4</td><td>67.1</td><td>67.7</td><td>30.5</td><td>52.4</td><td>59.3</td><td>31.7</td><td>14.5</td><td>39.2</td><td>85.4</td><td>45.1</td></tr><tr><td>${\mathrm{{GTR}}}_{\text{base }}$</td><td>42.0</td><td>51.1</td><td>21.5</td><td>66.0</td><td>24.1</td><td>60.0</td><td>53.9</td><td>30.8</td><td>49.5</td><td>53.5</td><td>34.9</td><td>14.9</td><td>39.2</td><td>88.1</td><td>45.2</td></tr><tr><td>T5 - 科尔伯特（T5 - ColBERT） ${}_{\text{base}}$</td><td>45.6</td><td>28.8</td><td>31.1</td><td>72.4</td><td>18.1</td><td>70.4</td><td>68.3</td><td>34.0</td><td>52.2</td><td>61.7</td><td>33.4</td><td>14.1</td><td>41.6</td><td>82.3</td><td>46.8</td></tr><tr><td>$\mathbf{{XT}{R}_{base}}$</td><td>45.0</td><td>40.7</td><td>31.3</td><td>73.7</td><td>20.7</td><td>71.0</td><td>73.6</td><td>34.0</td><td>53.0</td><td>64.7</td><td>34.7</td><td>14.5</td><td>40.9</td><td>86.1</td><td>49.1</td></tr><tr><td>${\text{Splade}}_{v2} \uparrow   \uparrow$</td><td>43.3</td><td>47.9</td><td>27.2</td><td>78.6</td><td>23.5</td><td>69.3</td><td>71.0</td><td>33.4</td><td>52.1</td><td>68.4</td><td>33.6</td><td>15.8</td><td>43.5</td><td>83.8</td><td>49.9</td></tr><tr><td>科尔伯特（ColBERT） ${}_{\mathrm{v}2}{}^{\bullet  \bullet  }$</td><td>-</td><td>46.3</td><td>26.3</td><td>78.5</td><td>17.6</td><td>69.3</td><td>73.8</td><td>33.8</td><td>56.2</td><td>66.7</td><td>35.6</td><td>15.4</td><td>44.6</td><td>85.2</td><td>49.9</td></tr><tr><td>${\mathrm{{GTR}}}_{\mathrm{{xxl}}}$</td><td>44.2</td><td>54.0</td><td>23.3</td><td>74.0</td><td>26.7</td><td>66.2</td><td>50.1</td><td>34.2</td><td>56.8</td><td>59.9</td><td>46.7</td><td>16.1</td><td>40.8</td><td>89.2</td><td>49.1</td></tr><tr><td>T5 - 科尔伯特（T5 - ColBERT） ${}_{\mathrm{{xx}}1}$</td><td>47.3</td><td>33.8</td><td>31.0</td><td>74.2</td><td>19.7</td><td>73.1</td><td>75.8</td><td>35.2</td><td>60.5</td><td>65.2</td><td>43.5</td><td>17.1</td><td>45.0</td><td>86.0</td><td>50.8</td></tr><tr><td>${\mathbf{{XTR}}}_{\mathbf{{xxl}}}$</td><td>46.6</td><td>44.2</td><td>30.9</td><td>77.0</td><td>24.5</td><td>74.3</td><td>78.9</td><td>35.3</td><td>60.9</td><td>66.2</td><td>43.8</td><td>17.1</td><td>44.3</td><td>88.1</td><td>52.7</td></tr></tbody></table>

<table><tr><td rowspan="2"/><td colspan="6">LoTTE Search</td><td colspan="6">LoTTE Forum</td></tr><tr><td>Writing</td><td>Rec.</td><td>Sci.</td><td>Tech.</td><td>Life.</td><td>Pooled</td><td>Writing</td><td>Rec.</td><td>Sci.</td><td>Tech.</td><td>Life.</td><td>Pooled</td></tr><tr><td>BM25</td><td>60.3</td><td>56.5</td><td>32.7</td><td>41.8</td><td>63.8</td><td>48.3</td><td>64.0</td><td>55.4</td><td>37.1</td><td>39.4</td><td>60.6</td><td>47.2</td></tr><tr><td>ColBERT</td><td>74.7</td><td>68.5</td><td>53.6</td><td>61.9</td><td>80.2</td><td>67.3</td><td>71.0</td><td>65.6</td><td>41.8</td><td>48.5</td><td>73.0</td><td>58.2</td></tr><tr><td>${\mathrm{{GTR}}}_{\text{base }}$</td><td>74.1</td><td>65.7</td><td>49.8</td><td>58.1</td><td>82.0</td><td>65.0</td><td>69.2</td><td>62.0</td><td>33.7</td><td>47.6</td><td>72.2</td><td>54.9</td></tr><tr><td>${\mathbf{{XTR}}}_{\mathbf{{base}}}$</td><td>77.0</td><td>69.4</td><td>54.9</td><td>63.2</td><td>82.1</td><td>69.0</td><td>73.9</td><td>68.7</td><td>42.2</td><td>51.9</td><td>74.4</td><td>60.1</td></tr><tr><td>${\text{Splade}}_{v2}{}^{\bullet  \bullet  }$</td><td>77.1</td><td>69.0</td><td>55.4</td><td>62.4</td><td>82.3</td><td>68.9</td><td>73.0</td><td>67.1</td><td>43.7</td><td>50.8</td><td>74.0</td><td>60.1</td></tr><tr><td>ColBERTv2 ${}^{\bullet  \bullet  }$</td><td>80.1</td><td>72.3</td><td>56.7</td><td>66.1</td><td>84.7</td><td>71.6</td><td>76.3</td><td>70.8</td><td>46.1</td><td>53.6</td><td>76.9</td><td>63.4</td></tr><tr><td>${\mathrm{{GTR}}}_{\mathrm{{xx}}1}$</td><td>83.9</td><td>78.0</td><td>60.0</td><td>69.5</td><td>87.4</td><td>76.0</td><td>79.5</td><td>73.5</td><td>43.1</td><td>62.6</td><td>81.9</td><td>66.9</td></tr><tr><td>${\mathbf{{XTR}}}_{\mathbf{{xxl}}}$</td><td>83.3</td><td>79.3</td><td>60.8</td><td>73.7</td><td>89.1</td><td>77.3</td><td>83.4</td><td>78.4</td><td>51.8</td><td>64.5</td><td>83.9</td><td>71.2</td></tr></table>

<table><tbody><tr><td rowspan="2"></td><td colspan="6">乐天搜索（LoTTE Search）</td><td colspan="6">乐天论坛（LoTTE Forum）</td></tr><tr><td>写作</td><td>推荐（Rec.）</td><td>科学（Sci.）</td><td>技术（Tech.）</td><td>生活。</td><td>合并的</td><td>写作</td><td>推荐（Rec.）</td><td>科学（Sci.）</td><td>技术（Tech.）</td><td>生活。</td><td>合并的</td></tr><tr><td>BM25（原文未变，可能是特定专业术语，无通用中文对应）</td><td>60.3</td><td>56.5</td><td>32.7</td><td>41.8</td><td>63.8</td><td>48.3</td><td>64.0</td><td>55.4</td><td>37.1</td><td>39.4</td><td>60.6</td><td>47.2</td></tr><tr><td>科尔伯特（ColBERT）</td><td>74.7</td><td>68.5</td><td>53.6</td><td>61.9</td><td>80.2</td><td>67.3</td><td>71.0</td><td>65.6</td><td>41.8</td><td>48.5</td><td>73.0</td><td>58.2</td></tr><tr><td>${\mathrm{{GTR}}}_{\text{base }}$</td><td>74.1</td><td>65.7</td><td>49.8</td><td>58.1</td><td>82.0</td><td>65.0</td><td>69.2</td><td>62.0</td><td>33.7</td><td>47.6</td><td>72.2</td><td>54.9</td></tr><tr><td>${\mathbf{{XTR}}}_{\mathbf{{base}}}$</td><td>77.0</td><td>69.4</td><td>54.9</td><td>63.2</td><td>82.1</td><td>69.0</td><td>73.9</td><td>68.7</td><td>42.2</td><td>51.9</td><td>74.4</td><td>60.1</td></tr><tr><td>${\text{Splade}}_{v2}{}^{\bullet  \bullet  }$</td><td>77.1</td><td>69.0</td><td>55.4</td><td>62.4</td><td>82.3</td><td>68.9</td><td>73.0</td><td>67.1</td><td>43.7</td><td>50.8</td><td>74.0</td><td>60.1</td></tr><tr><td>科尔伯特v2 ${}^{\bullet  \bullet  }$（ColBERTv2 ${}^{\bullet  \bullet  }$）</td><td>80.1</td><td>72.3</td><td>56.7</td><td>66.1</td><td>84.7</td><td>71.6</td><td>76.3</td><td>70.8</td><td>46.1</td><td>53.6</td><td>76.9</td><td>63.4</td></tr><tr><td>${\mathrm{{GTR}}}_{\mathrm{{xx}}1}$</td><td>83.9</td><td>78.0</td><td>60.0</td><td>69.5</td><td>87.4</td><td>76.0</td><td>79.5</td><td>73.5</td><td>43.1</td><td>62.6</td><td>81.9</td><td>66.9</td></tr><tr><td>${\mathbf{{XTR}}}_{\mathbf{{xxl}}}$</td><td>83.3</td><td>79.3</td><td>60.8</td><td>73.7</td><td>89.1</td><td>77.3</td><td>83.4</td><td>78.4</td><td>51.8</td><td>64.5</td><td>83.9</td><td>71.2</td></tr></tbody></table>

-: cross-encoder distillation $\;\blacklozenge$ : model-based hard negatives

-：交叉编码器蒸馏 $\;\blacklozenge$ ：基于模型的硬负样本

Table 2: (top) nDCG@10 on MS MARCO (in-domain) and BEIR (zero-shot). The last column shows the average over 13 BEIR datasets. (bottom) Top-5 retrieval accuracy on LoTTE datasets (zero-shot).

表2：（上）MS MARCO（领域内）和BEIR（零样本）上的nDCG@10。最后一列显示了13个BEIR数据集的平均值。（下）LoTTE数据集（零样本）上的前5名检索准确率。

<!-- Media -->

Table 1 shows the estimated FLOPs of ColBERT and XTR (see Appendix B) for more details). Due to the differences in hardware and infrastructure, we mainly compared the theoretical FLOPs. XTR reduces the FLOPs at the scoring stage by ${4000} \times$ making multi-vector retrieval more efficient.

表1展示了ColBERT和XTR的估计浮点运算次数（更多细节见附录B）。由于硬件和基础设施的差异，我们主要比较了理论浮点运算次数。XTR通过 ${4000} \times$ 提高多向量检索效率，降低了评分阶段的浮点运算次数。

## 4 Experiments

## 4 实验

Experimental Setting Following Ni et al. [2021], we fine-tune XTR on MS MARCO with a fixed set of hard negatives from RocketQA [Qu et al. 2021]. Then, we test XTR on MS MARCO (MS; in-domain) and zero-shot IR datasets. For the zero-shot evaluation, we use 13 datasets from BEIR [Thakur et al. 2021] (see Appendix C for acronyms), 12 datasets from LoTTE [Santhanam et al. 2022b], and 4 datasets on open-domain QA passage retrieval (EQ: EntityQuestions [Sciavolino et al. 2021], NQ, TQA: TriviaQA, SQD: SQuAD). We also train multilingual XTR (mXTR) and evaluate it on MIRACL [Zhang et al., 2022b], which contains retrieval tasks in 18 languages. The performance gap between T5-ColBERT [Qian et al. 2022] and XTR shows the improvement with our methods on a multi-vector retrieval model. For implementation details and baselines, see Appendix C For the relationship between hyperparameters (e.g., ${k}_{\text{train }}$ and ${k}^{\prime }$ ),see §5.3

实验设置 遵循Ni等人 [2021] 的方法，我们在MS MARCO数据集上对XTR进行微调，并使用来自RocketQA [Qu等人，2021] 的一组固定难负样本。然后，我们在MS MARCO（MS；领域内）和零样本信息检索（IR）数据集上测试XTR。对于零样本评估，我们使用了来自BEIR [Thakur等人，2021] 的13个数据集（缩写详情见附录C）、来自LoTTE [Santhanam等人，2022b] 的12个数据集，以及4个开放域问答段落检索数据集（EQ：实体问题数据集（EntityQuestions）[Sciavolino等人，2021]、NQ、TQA：常识问答数据集（TriviaQA）、SQD：斯坦福问答数据集（SQuAD））。我们还训练了多语言XTR（mXTR），并在MIRACL [Zhang等人，2022b] 上对其进行评估，该数据集包含18种语言的检索任务。T5 - ColBERT [Qian等人，2022] 和XTR之间的性能差距表明了我们的方法在多向量检索模型上的改进。实现细节和基线模型见附录C。关于超参数（例如，${k}_{\text{train }}$ 和 ${k}^{\prime }$）之间的关系，请参阅§5.3

---

<!-- Footnote -->

${}^{5}$ We found that directly training with ${f}_{{\mathrm{{XTR}}}^{\prime }}$ instead of ${f}_{\mathrm{{XTR}}}$ fails to converge,which we leave as future work.

${}^{5}$ 我们发现，直接使用 ${f}_{{\mathrm{{XTR}}}^{\prime }}$ 而非 ${f}_{\mathrm{{XTR}}}$ 进行训练无法收敛，我们将此留作未来的工作。

<!-- Footnote -->

---

<!-- Media -->

<table><tr><td rowspan="2"/><td colspan="2">EQ</td><td colspan="2">$\mathbf{{NQ}}$</td><td colspan="2">TQA</td><td colspan="2">SQD</td></tr><tr><td>Top-20</td><td>Top-100</td><td>Top-20</td><td>Top-100</td><td>Top-20</td><td>Top-100</td><td>Top-20</td><td>Top-100</td></tr><tr><td>BM25*</td><td>71.4</td><td>80.0</td><td>62.9</td><td>78.3</td><td>76.4</td><td>83.2</td><td>71.1</td><td>81.8</td></tr><tr><td>${\mathrm{{DPR}}}_{\text{multi }} + {\mathrm{{BM25}}}^{ \bullet  }$</td><td>73.3</td><td>82.6</td><td>82.6</td><td>88.6</td><td>82.6</td><td>86.5</td><td>75.1</td><td>84.4</td></tr><tr><td>${\mathrm{{ART}}}_{\mathrm{{MS}}\text{ MARCO }}$</td><td>75.3</td><td>81.9</td><td>-</td><td>-</td><td>78.0</td><td>84.1</td><td>68.4</td><td>80.4</td></tr><tr><td>${\mathrm{{GTR}}}_{\text{base }}$</td><td>73.3</td><td>80.6</td><td>78.5</td><td>86.5</td><td>76.2</td><td>83.4</td><td>65.9</td><td>77.6</td></tr><tr><td>${\mathrm{{GTR}}}_{\mathrm{{xxl}}}$</td><td>75.3</td><td>82.5</td><td>83.5</td><td>89.8</td><td>81.7</td><td>86.6</td><td>70.4</td><td>80.6</td></tr><tr><td>${\mathrm{{DPR}}}_{\text{multi }}$</td><td>56.7</td><td>70.0</td><td>79.5</td><td>86.1</td><td>78.9</td><td>84.8</td><td>52.0</td><td>67.7</td></tr><tr><td>ColBERT</td><td>-</td><td>-</td><td>79.1</td><td>-</td><td>80.3</td><td>-</td><td>76.5</td><td>-</td></tr><tr><td>${\mathbf{{XTR}}}_{\mathbf{{base}}}$</td><td>79.0</td><td>85.2</td><td>79.3</td><td>88.1</td><td>80.3</td><td>85.5</td><td>78.2</td><td>85.9</td></tr><tr><td>${\mathbf{{XTR}}}_{\mathbf{{xxl}}}$</td><td>79.4</td><td>85.9</td><td>84.9</td><td>90.5</td><td>83.3</td><td>87.1</td><td>81.1</td><td>87.6</td></tr></table>

<table><tbody><tr><td rowspan="2"></td><td colspan="2">情商（EQ）</td><td colspan="2">$\mathbf{{NQ}}$</td><td colspan="2">总质量保证（TQA）</td><td colspan="2">供应商质量开发（SQD）</td></tr><tr><td>前20名（Top-20）</td><td>前100名（Top-100）</td><td>前20名（Top-20）</td><td>前100名（Top-100）</td><td>前20名（Top-20）</td><td>前100名（Top-100）</td><td>前20名（Top-20）</td><td>前100名（Top-100）</td></tr><tr><td>改进版BM25算法（BM25*）</td><td>71.4</td><td>80.0</td><td>62.9</td><td>78.3</td><td>76.4</td><td>83.2</td><td>71.1</td><td>81.8</td></tr><tr><td>${\mathrm{{DPR}}}_{\text{multi }} + {\mathrm{{BM25}}}^{ \bullet  }$</td><td>73.3</td><td>82.6</td><td>82.6</td><td>88.6</td><td>82.6</td><td>86.5</td><td>75.1</td><td>84.4</td></tr><tr><td>${\mathrm{{ART}}}_{\mathrm{{MS}}\text{ MARCO }}$</td><td>75.3</td><td>81.9</td><td>-</td><td>-</td><td>78.0</td><td>84.1</td><td>68.4</td><td>80.4</td></tr><tr><td>${\mathrm{{GTR}}}_{\text{base }}$</td><td>73.3</td><td>80.6</td><td>78.5</td><td>86.5</td><td>76.2</td><td>83.4</td><td>65.9</td><td>77.6</td></tr><tr><td>${\mathrm{{GTR}}}_{\mathrm{{xxl}}}$</td><td>75.3</td><td>82.5</td><td>83.5</td><td>89.8</td><td>81.7</td><td>86.6</td><td>70.4</td><td>80.6</td></tr><tr><td>${\mathrm{{DPR}}}_{\text{multi }}$</td><td>56.7</td><td>70.0</td><td>79.5</td><td>86.1</td><td>78.9</td><td>84.8</td><td>52.0</td><td>67.7</td></tr><tr><td>科尔伯特（ColBERT）</td><td>-</td><td>-</td><td>79.1</td><td>-</td><td>80.3</td><td>-</td><td>76.5</td><td>-</td></tr><tr><td>${\mathbf{{XTR}}}_{\mathbf{{base}}}$</td><td>79.0</td><td>85.2</td><td>79.3</td><td>88.1</td><td>80.3</td><td>85.5</td><td>78.2</td><td>85.9</td></tr><tr><td>${\mathbf{{XTR}}}_{\mathbf{{xxl}}}$</td><td>79.4</td><td>85.9</td><td>84.9</td><td>90.5</td><td>83.3</td><td>87.1</td><td>81.1</td><td>87.6</td></tr></tbody></table>

-: sparse component $\;\blacklozenge$ : retrieval pre-training

-: 稀疏分量 $\;\blacklozenge$ : 检索预训练

Table 3: Zero-shot passage retrieval accuracy on open-domain question answering datasets. In-domain performances are underlined and all the other performances are based on the zero-shot evaluation. For EntityQuestions, we report macro-averaged performances over different relations.

表3：开放域问答数据集上的零样本段落检索准确率。领域内性能加了下划线，其他所有性能均基于零样本评估。对于实体问题（EntityQuestions），我们报告了不同关系的宏平均性能。

<table><tr><td/><td>ar</td><td>bn</td><td>en</td><td>es</td><td>fa</td><td>fi</td><td>fr</td><td>hi</td><td>id</td><td>já</td><td>ko</td><td>ru</td><td>SW</td><td>te</td><td>th</td><td>zh</td><td>de</td><td>yo</td><td>Avg.</td></tr><tr><td>BM25</td><td>48.1</td><td>50.8</td><td>35.1</td><td>31.9</td><td>33.3</td><td>55.1</td><td>18.3</td><td>45.8</td><td>44.9</td><td>36.9</td><td>41.9</td><td>33.4</td><td>38.3</td><td>49.4</td><td>48.4</td><td>18.0</td><td>-</td><td>-</td><td>-</td></tr><tr><td>mDPR</td><td>49.9</td><td>44.3</td><td>39.4</td><td>47.8</td><td>48.0</td><td>47.2</td><td>43.5</td><td>38.3</td><td>27.2</td><td>43.9</td><td>41.9</td><td>40.7</td><td>29.9</td><td>35.6</td><td>35.8</td><td>51.2</td><td>-</td><td>-</td><td>-</td></tr><tr><td>BM25 + mDPR</td><td>67.3</td><td>65.4</td><td>54.9</td><td>64.1</td><td>59.4</td><td>67.2</td><td>52.3</td><td>61.6</td><td>44.3</td><td>57.6</td><td>60.9</td><td>53.2</td><td>44.6</td><td>60.2</td><td>59.9</td><td>52.6</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="20">Trained on English MS MARCO</td></tr><tr><td>mContriever (en)</td><td>55.3</td><td>54.2</td><td>37.9</td><td>34.1</td><td>42.6</td><td>51.2</td><td>31.5</td><td>40.6</td><td>36.8</td><td>38.3</td><td>46.2</td><td>39.9</td><td>44.4</td><td>48.7</td><td>52.4</td><td>27.4</td><td>32.9</td><td>32.9</td><td>41.5</td></tr><tr><td>${\mathbf{{mXTR}}}_{\text{base }}\left( \mathrm{{en}}\right)$</td><td>66.1</td><td>64.7</td><td>49.4</td><td>40.5</td><td>47.9</td><td>62.2</td><td>37.5</td><td>51.4</td><td>46.9</td><td>56.8</td><td>64.0</td><td>49.8</td><td>43.0</td><td>67.7</td><td>69.2</td><td>47.2</td><td>34.5</td><td>40.6</td><td>52.2</td></tr><tr><td>${\mathbf{{mXTR}}}_{\mathbf{{xxI}}}\left( \mathrm{{en}}\right)$</td><td>74.1</td><td>75.5</td><td>56.0</td><td>52.4</td><td>56.1</td><td>75.1</td><td>51.4</td><td>61.8</td><td>52.0</td><td>68.7</td><td>67.4</td><td>61.3</td><td>69.7</td><td>76.0</td><td>76.9</td><td>56.9</td><td>51.7</td><td>60.3</td><td>63.5</td></tr><tr><td colspan="20">Trained on English MS MARCO + MIRACL (16 languages)</td></tr><tr><td>mContriever</td><td>64.6</td><td>66.4</td><td>41.2</td><td>40.3</td><td>46.3</td><td>61.9</td><td>42.9</td><td>41.9</td><td>44.6</td><td>55.6</td><td>55.4</td><td>48.1</td><td>65.3</td><td>77.6</td><td>69.3</td><td>45.9</td><td>39.6</td><td>41.9</td><td>52.7</td></tr><tr><td>${\mathbf{{mXTR}}}_{\mathbf{{base}}}$</td><td>73.0</td><td>73.9</td><td>46.1</td><td>42.6</td><td>51.0</td><td>70.5</td><td>39.3</td><td>51.3</td><td>54.2</td><td>62.3</td><td>67.7</td><td>54.5</td><td>69.7</td><td>80.7</td><td>76.1</td><td>51.4</td><td>36.1</td><td>46.8</td><td>58.2</td></tr><tr><td>${\mathbf{{mXTR}}}_{\mathbf{{xxI}}}$</td><td>77.8</td><td>78.4</td><td>52.5</td><td>48.9</td><td>56.0</td><td>76.0</td><td>52.9</td><td>61.5</td><td>54.9</td><td>73.4</td><td>68.5</td><td>66.2</td><td>79.4</td><td>84.3</td><td>80.7</td><td>58.9</td><td>52.8</td><td>62.4</td><td>65.9</td></tr></table>

<table><tbody><tr><td></td><td>阿拉伯语（Arabic）</td><td>孟加拉语（Bengali）</td><td>英语（English）</td><td>西班牙语（Spanish）</td><td>波斯语（Farsi）</td><td>芬兰语（Finnish）</td><td>法语（fr）</td><td>你好</td><td>印尼语（id）</td><td>是的（捷克语“já”）</td><td>韩语（ko）</td><td>俄语（ru）</td><td>西南（Southwest）</td><td>这（在一些语言中可能是无意义的词，这里按可能的语境推测）</td><td>这；那</td><td>这；之（在一些特定语境中）</td><td>的</td><td>哟；唷</td><td>平均值</td></tr><tr><td>BM25算法</td><td>48.1</td><td>50.8</td><td>35.1</td><td>31.9</td><td>33.3</td><td>55.1</td><td>18.3</td><td>45.8</td><td>44.9</td><td>36.9</td><td>41.9</td><td>33.4</td><td>38.3</td><td>49.4</td><td>48.4</td><td>18.0</td><td>-</td><td>-</td><td>-</td></tr><tr><td>多语言密集段落检索器（mDPR）</td><td>49.9</td><td>44.3</td><td>39.4</td><td>47.8</td><td>48.0</td><td>47.2</td><td>43.5</td><td>38.3</td><td>27.2</td><td>43.9</td><td>41.9</td><td>40.7</td><td>29.9</td><td>35.6</td><td>35.8</td><td>51.2</td><td>-</td><td>-</td><td>-</td></tr><tr><td>BM25算法 + 多语言密集段落检索器（mDPR）</td><td>67.3</td><td>65.4</td><td>54.9</td><td>64.1</td><td>59.4</td><td>67.2</td><td>52.3</td><td>61.6</td><td>44.3</td><td>57.6</td><td>60.9</td><td>53.2</td><td>44.6</td><td>60.2</td><td>59.9</td><td>52.6</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="20">基于英文MS MARCO数据集训练</td></tr><tr><td>多语言对比检索器（mContriever，英文）</td><td>55.3</td><td>54.2</td><td>37.9</td><td>34.1</td><td>42.6</td><td>51.2</td><td>31.5</td><td>40.6</td><td>36.8</td><td>38.3</td><td>46.2</td><td>39.9</td><td>44.4</td><td>48.7</td><td>52.4</td><td>27.4</td><td>32.9</td><td>32.9</td><td>41.5</td></tr><tr><td>${\mathbf{{mXTR}}}_{\text{base }}\left( \mathrm{{en}}\right)$</td><td>66.1</td><td>64.7</td><td>49.4</td><td>40.5</td><td>47.9</td><td>62.2</td><td>37.5</td><td>51.4</td><td>46.9</td><td>56.8</td><td>64.0</td><td>49.8</td><td>43.0</td><td>67.7</td><td>69.2</td><td>47.2</td><td>34.5</td><td>40.6</td><td>52.2</td></tr><tr><td>${\mathbf{{mXTR}}}_{\mathbf{{xxI}}}\left( \mathrm{{en}}\right)$</td><td>74.1</td><td>75.5</td><td>56.0</td><td>52.4</td><td>56.1</td><td>75.1</td><td>51.4</td><td>61.8</td><td>52.0</td><td>68.7</td><td>67.4</td><td>61.3</td><td>69.7</td><td>76.0</td><td>76.9</td><td>56.9</td><td>51.7</td><td>60.3</td><td>63.5</td></tr><tr><td colspan="20">基于英文MS MARCO + MIRACL（16种语言）进行训练</td></tr><tr><td>mContriever</td><td>64.6</td><td>66.4</td><td>41.2</td><td>40.3</td><td>46.3</td><td>61.9</td><td>42.9</td><td>41.9</td><td>44.6</td><td>55.6</td><td>55.4</td><td>48.1</td><td>65.3</td><td>77.6</td><td>69.3</td><td>45.9</td><td>39.6</td><td>41.9</td><td>52.7</td></tr><tr><td>${\mathbf{{mXTR}}}_{\mathbf{{base}}}$</td><td>73.0</td><td>73.9</td><td>46.1</td><td>42.6</td><td>51.0</td><td>70.5</td><td>39.3</td><td>51.3</td><td>54.2</td><td>62.3</td><td>67.7</td><td>54.5</td><td>69.7</td><td>80.7</td><td>76.1</td><td>51.4</td><td>36.1</td><td>46.8</td><td>58.2</td></tr><tr><td>${\mathbf{{mXTR}}}_{\mathbf{{xxI}}}$</td><td>77.8</td><td>78.4</td><td>52.5</td><td>48.9</td><td>56.0</td><td>76.0</td><td>52.9</td><td>61.5</td><td>54.9</td><td>73.4</td><td>68.5</td><td>66.2</td><td>79.4</td><td>84.3</td><td>80.7</td><td>58.9</td><td>52.8</td><td>62.4</td><td>65.9</td></tr></tbody></table>

Table 4: nDCG@10 on 18 multilingual retrieval tasks from MIRACL. Each row shows the performance of a single multilingual retrieval model. The last two surprise languages (de and yo) are not included in the training dataset of MIRACL. The last column shows the average over 18 languages.

表4：MIRACL的18项多语言检索任务的nDCG@10。每行展示了单个多语言检索模型的性能。最后两种意外语言（德语“de”和约鲁巴语“yo”）未包含在MIRACL的训练数据集中。最后一列展示了18种语言的平均值。

<!-- Media -->

### 4.1 In-domain Document Retrieval

### 4.1 领域内文档检索

MS MARCO The first column of Table 2 (top) shows nDCG@10 on MS MARCO (see Table D. 1 for recall@100). XTR outperforms most models and remains competitive with T5-ColBERT. This is encouraging since XTR significantly reduces the cost of the gathering-scoring stage. Note that MS MARCO may fail to reflect the actual improvement of state-of-the-art [Arabzadeh et al., 2022].

MS MARCO 表2（上）的第一列展示了MS MARCO的nDCG@10（召回率@100详见表D.1）。XTR的表现优于大多数模型，并且与T5 - ColBERT相比仍具竞争力。这令人鼓舞，因为XTR显著降低了收集 - 评分阶段的成本。请注意，MS MARCO可能无法反映最先进技术的实际改进[Arabzadeh等人，2022]。

### 4.2 Zero-shot Document Retrieval

### 4.2 零样本文档检索

BEIR & LoTTE Table 2 (top; except the first columns) shows nDCG@10 on BEIR (see Table D.I for recall@100). ${\mathrm{{XTR}}}_{\mathrm{{xxl}}}$ achieves the new state-of-the-art performances significantly outperforming both per-domain models and single model state-of-the-art. Simply scaling XTR removes the needs of designing distillation or hard negative mining pipelines [Santhanam et al., 2022b Formal et al., 2021]. Results on LoTTE (Table 2 bottom) also show that ${\mathrm{{XTR}}}_{\text{base }}$ is better than ColBERT and competitive with distillation-based models while ${\mathrm{{XTR}}}_{\mathrm{{xxl}}}$ advances the state-of-the-art.

BEIR和LoTTE表2（顶部；除第一列外）展示了BEIR上的nDCG@10指标（召回率@100指标见附录表D.I）。${\mathrm{{XTR}}}_{\mathrm{{xxl}}}$取得了新的最优性能，显著优于各领域模型和单模型最优水平。简单扩展XTR模型就无需设计蒸馏或难负样本挖掘流程[Santhanam等人，2022b；Formal等人，2021]。LoTTE数据集上的结果（表2底部）同样表明，${\mathrm{{XTR}}}_{\text{base }}$优于ColBERT模型，并且与基于蒸馏的模型具有竞争力，而${\mathrm{{XTR}}}_{\mathrm{{xxl}}}$则进一步提升了最优水平。

Passage retrieval for open-domain QA Table 3 shows results on four open-domain QA datasets. While previous work often includes sparse retrievers (e.g., BM25) [Chen et al. 2021] or contrastive pre-training [Ram et al., 2022, Sachan et al., 2022a b] to achieve better performances on EntityQues-tions, XTR simply fine-tuned on MS MARCO achieves the state-of-the-art performance.

开放域问答的段落检索 表3展示了在四个开放域问答数据集上的结果。虽然以往的工作通常会引入稀疏检索器（例如BM25）[Chen等人，2021]或对比预训练[Ram等人，2022；Sachan等人，2022a，b]以在实体问题上取得更好的性能，但XTR仅在MS MARCO上进行微调就实现了最先进的性能。

### 4.3 Multilingual Document Retrieval

### 4.3 多语言文档检索

MIRACL Since XTR does not need any secondary pre-training, we expect it to be better at multilingual retrieval by better utilizing the multilingual language models. We train a multilingual version of XTR with mT5 [Xue et al. 2021] and test it on multilingual retrieval tasks in 18 languages. Table 4 shows that mXTR greatly outperforms mContriever that uses expensive contrastive pretraining, as well as the hybrid model, BM25 + mDPR.

MIRACL 由于XTR不需要任何二次预训练，我们期望它能通过更好地利用多语言模型在多语言检索方面表现更优。我们使用mT5 [Xue等人，2021]训练了一个多语言版本的XTR，并在18种语言的多语言检索任务上对其进行测试。表4显示，mXTR大大优于使用昂贵对比预训练的mContriever，以及混合模型BM25 + mDPR。

<!-- Media -->

<!-- figureText: 0.10 T5-ColBERT-xxl T5-Colbert-xxl 600 800 1000 200 400 600 800 1000 Token Rank k Token Rank k TREC-COVID ArquAna T5-ColBERT-xxl XTR-xxI XTR-xxI 800 1000 0 200 600 1000 T5-ColBERT-xxl 200 400 600 800 1000 200 400 Token Rank k MS MARCO T5-ColBERT-xxl XTR-xxI 200 600 800 400 -->

<img src="https://cdn.noedgeai.com/01958a7f-8417-7506-8c2f-6473566d79b1_7.jpg?x=310&y=204&w=1171&h=392&r=0"/>

Figure 4: (top) Gold token retrieval performances of T5-ColBERT and XTR. We plot the probability of each retrieved document token at rank $k$ coming from the gold document. (bottom) Lexical token retrieval performances of T5-ColBERT and XTR. We plot the probability of each retrieved document token at rank $k$ being lexically identical to its query token.

图4：（上）T5-ColBERT和XTR的黄金标记（Gold token）检索性能。我们绘制了排名为$k$的每个检索文档标记来自黄金文档（Gold document）的概率。（下）T5-ColBERT和XTR的词汇标记（Lexical token）检索性能。我们绘制了排名为$k$的每个检索文档标记在词汇上与其查询标记相同的概率。

<table><tr><td>Model</td><td>Imputation</td><td>MRR@10</td><td>R@1000</td></tr><tr><td rowspan="2">T5-ColBERT ${}_{\text{base}}$</td><td>None</td><td>0.0</td><td>0.0</td></tr><tr><td>top- ${k}^{\prime }$ score</td><td>27.7</td><td>91.8</td></tr><tr><td rowspan="4">${\mathrm{{XTR}}}_{\text{base }}$</td><td>None</td><td>22.6</td><td>88.7</td></tr><tr><td>${m}_{i} = 0$</td><td>36.2</td><td>97.3</td></tr><tr><td>${m}_{i} = {0.2}$</td><td>36.4</td><td>97.3</td></tr><tr><td>top- ${k}^{\prime }$ score</td><td>37.4</td><td>98.0</td></tr></table>

<table><tbody><tr><td>模型</td><td>插补（Imputation）</td><td>前10项平均倒数排名（MRR@10）</td><td>R@1000</td></tr><tr><td rowspan="2">T5 - 科尔伯特（T5 - ColBERT） ${}_{\text{base}}$</td><td>无</td><td>0.0</td><td>0.0</td></tr><tr><td>前 ${k}^{\prime }$ 得分</td><td>27.7</td><td>91.8</td></tr><tr><td rowspan="4">${\mathrm{{XTR}}}_{\text{base }}$</td><td>无</td><td>22.6</td><td>88.7</td></tr><tr><td>${m}_{i} = 0$</td><td>36.2</td><td>97.3</td></tr><tr><td>${m}_{i} = {0.2}$</td><td>36.4</td><td>97.3</td></tr><tr><td>前 ${k}^{\prime }$ 得分</td><td>37.4</td><td>98.0</td></tr></tbody></table>

Table 5: Impact of training objectives and imputation methods comparing T5-ColBERT and XTR. For both models, we apply ${f}_{{\mathrm{{XTR}}}^{\prime }}$ during inference. We report MRR@10 and Recall@1000 on the MS MARCO development set.

表5：比较T5-ColBERT和XTR时训练目标和插补方法的影响。对于这两个模型，我们在推理过程中应用${f}_{{\mathrm{{XTR}}}^{\prime }}$。我们报告了MS MARCO开发集上的MRR@10和Recall@1000。

<!-- figureText: ${10}^{2}$ ${10}^{3}$ MS MARCC T5-ColBERT-base (f ColBERT) T5-ColBERT-base (f XTR') XTR-base ${10}^{5}$ -->

<img src="https://cdn.noedgeai.com/01958a7f-8417-7506-8c2f-6473566d79b1_7.jpg?x=1080&y=774&w=390&h=282&r=0"/>

Figure 5: Recall @100 of XTR and T5-ColBERT with different ${k}^{\prime }$ . For T5-ColBERT,we use either ${f}_{{\mathrm{{XTR}}}^{\prime }}$ or ${f}_{\text{ColBERT }}$ .

图5：不同${k}^{\prime }$下XTR和T5-ColBERT的Recall @100。对于T5-ColBERT，我们使用${f}_{{\mathrm{{XTR}}}^{\prime }}$或${f}_{\text{ColBERT }}$。

<!-- Media -->

## 5 Analysis

## 5 分析

### 5.1 Towards Better Token Retrieval

### 5.1 迈向更好的Token（词元）检索

Gold token retrieval If the tokens of gold documents are not retrieved at all, multi-vector retrieval models would fail to retrieve the gold documents. Hence, a better token retrieval would contain these gold tokens more often in their top results. In Figure 4 (top), we show the probability of a token at the rank $k$ coming from the gold documents of a query. To compute the probability for the rank $k$ ,we simply count the number of an event where a token at rank $k$ belongs to the gold document and divide it by the number of tokens at rank $k$ . While this is measuring the precision of the token retrieval,we observed a similar trend for the recall of gold tokens. Compared to T5-ColBERT, XTR retrieves gold tokens with higher probability, even on MS MARCO. This shows that the training objective of XTR encourages it to retrieve tokens from more relevant context.

黄金标记检索 如果根本没有检索到黄金文档（gold documents）的标记，多向量检索模型将无法检索到黄金文档。因此，更好的标记检索会在其顶级结果中更频繁地包含这些黄金标记。在图4（顶部）中，我们展示了排名为$k$的标记来自查询的黄金文档的概率。为了计算排名为$k$的概率，我们只需统计排名为$k$的标记属于黄金文档的事件数量，并将其除以排名为$k$的标记数量。虽然这是在衡量标记检索的精确率，但我们观察到黄金标记的召回率也有类似的趋势。与T5 - ColBERT相比，即使在MS MARCO数据集上，XTR也能以更高的概率检索到黄金标记。这表明XTR的训练目标促使它从更相关的上下文中检索标记。

Lexical token retrieval In Figure 4 (bottom),we show the probability of a token at the rank $k$ being the same as its query token (e.g., 'insulin' retrieving 'insulin's). T5-ColBERT has very high probability of retrieving the same token across different ranks and datasets. However, it is unclear to what extent the token retrieval stage should behave as sparse retrieval, as it might suffer from the vocabulary mismatch problem. XTR effectively lowers the reliance on the lexical matching while preserving a good amount of lexical precision so that it would achieve a high retrieval accuracy on the entity-centric dataset (§4.2). In fact, Table 6 in Appendix shows that having lower lexical matching doesn't necessarily mean a lower retrieval quality, but often means better contextualization.

词法标记检索 在图4（底部）中，我们展示了排名为$k$的标记与查询标记相同的概率（例如，用“胰岛素（insulin）”检索“胰岛素的（insulin's）”）。T5 - ColBERT在不同排名和数据集上检索到相同标记的概率非常高。然而，目前尚不清楚标记检索阶段应在多大程度上表现为稀疏检索，因为它可能会受到词汇不匹配问题的影响。XTR在有效降低对词法匹配的依赖的同时，保留了相当高的词法精度，从而在以实体为中心的数据集上实现了较高的检索准确率（§4.2）。事实上，附录中的表6表明，较低的词法匹配并不一定意味着较低的检索质量，而往往意味着更好的上下文关联。

### 5.2 Efficient Scoring

### 5.2 高效评分

In Table 5,we show how we can employ the efficient scoring function ${f}_{{\mathrm{{XTR}}}^{\prime }}$ in XTR with minimal performance losses. We apply ${f}_{{\mathrm{{XTR}}}^{\prime }}$ on both T5-ColBERT and XTR,and show their performances on MS MARCO. With T5-ColBERT,even if we use the top- ${k}^{\prime }$ score for the imputation,the performance is much worse than the original sum-of-max scoring. With XTR, the performance greatly improves as it has better token retrieval. Figure 5 shows how Recall @100 improves with larger ${k}^{\prime }$ ’s as it provides more exact upper bound for the missing similarity imputation. Table D. 2 shows that even if we use smaller ${k}^{\prime }$ ,XTR still maintains high performances on BEIR.

在表5中，我们展示了如何在XTR中采用高效评分函数${f}_{{\mathrm{{XTR}}}^{\prime }}$，同时将性能损失降至最低。我们在T5 - ColBERT和XTR上都应用了${f}_{{\mathrm{{XTR}}}^{\prime }}$，并展示了它们在MS MARCO上的性能表现。对于T5 - ColBERT，即使我们使用前${k}^{\prime }$个分数进行插补，其性能也远不如原始的最大和评分。而对于XTR，由于它具有更好的词元检索能力，其性能有了显著提升。图5展示了随着${k}^{\prime }$值的增大，Recall @100是如何提高的，因为它为缺失相似度插补提供了更精确的上限。表D. 2显示，即使我们使用较小的${k}^{\prime }$，XTR在BEIR上仍能保持较高的性能。

<!-- Media -->

<!-- figureText: 0.40 XTR-base (k train = 32) - XTR-base (k train = 64) $\rightarrow   -$ XTR-base (k train $= {320}$ ) - T5-ColBERT-base (f XTR') -->

<img src="https://cdn.noedgeai.com/01958a7f-8417-7506-8c2f-6473566d79b1_8.jpg?x=327&y=207&w=363&h=265&r=0"/>

Figure 6: MRR@10 of XTR with different ${k}_{\text{train }}$ and ${k}^{\prime }$ . For T5-ColBERT,we also use ${f}_{{\mathrm{{XTR}}}^{\prime }}$ with the top- ${k}^{\prime }$ score imputation method for the inference.

图6：不同${k}_{\text{train }}$和${k}^{\prime }$下XTR的前10平均倒数排名（MRR@10）。对于T5 - ColBERT，我们还在推理时使用了前${k}^{\prime }$得分插补法的${f}_{{\mathrm{{XTR}}}^{\prime }}$。

<!-- figureText: MS MARCO ArguAna 0.44 XTR-base (batch_size $= {256}$ ) XTR-base (batch size $= {320}$ ) 300 0.46 0.44 XTR-base (batch size $= {256}$ ) XTR-base (batch size $= {320}$ ) 250 k train -->

<img src="https://cdn.noedgeai.com/01958a7f-8417-7506-8c2f-6473566d79b1_8.jpg?x=717&y=226&w=755&h=248&r=0"/>

Figure 7: Effect of training XTR with different batch sizes and ${k}_{\text{train }}$ . For each point of the graph,we train ${\mathrm{{XTR}}}_{\text{base }}$ with the specified training batch size(128,256,320)and ${k}_{\text{train }}({32}$ , ${64},{128},{256})$ and evaluate on each dataset (MS MARCO and ArguAna). nDCG@10 of each model is reported.

图7：不同批量大小和${k}_{\text{train }}$对训练XTR的影响。对于图中的每个点，我们使用指定的训练批量大小（128、256、320）和${k}_{\text{train }}({32}$、${64},{128},{256})$训练${\mathrm{{XTR}}}_{\text{base }}$，并在每个数据集（MS MARCO和ArguAna）上进行评估。报告了每个模型的前10归一化折损累积增益（nDCG@10）。

<!-- Media -->

### 5.3 Relationship between Hyperparameters

### 5.3 超参数之间的关系

${\mathbf{k}}_{\text{train }}$ vs. ${\mathbf{k}}^{\prime }$ In Figure 6,we show MRR@10 of XTR trained with different ${k}_{\text{train }}$ and evaluated with different ${k}^{\prime }$ on the MS MARCO development set. While all variants of XTR prefer larger ${k}^{\prime }$ ,ones trained with smaller ${k}_{\text{train }}$ show higher performances than others under small ${k}^{\prime }$ settings. XTR with larger ${k}_{\text{train }}$ exhibits better performances than ones with smaller ${k}_{\text{train }}$ as ${k}^{\prime }$ becomes larger.

${\mathbf{k}}_{\text{train }}$与${\mathbf{k}}^{\prime }$的对比 在图6中，我们展示了在MS MARCO开发集上，使用不同的${k}_{\text{train }}$训练并使用不同的${k}^{\prime }$评估的XTR模型的前10召回率均值（MRR@10）。虽然XTR的所有变体都倾向于更大的${k}^{\prime }$，但在较小的${k}^{\prime }$设置下，使用较小的${k}_{\text{train }}$训练的模型比其他模型表现更好。随着${k}^{\prime }$增大，使用较大的${k}_{\text{train }}$训练的XTR模型比使用较小的${k}_{\text{train }}$训练的模型表现更优。

Training batch size vs. ${k}_{\text{train }}$ In Figure 7,we show the relationship between the training batch size and ${k}_{\text{train }}$ during training XTR. In this experiment,we use ${k}^{\prime } = {40},{000}$ . While it is evident that XTR mostly favors large training batch sizes,the optimal top- ${k}_{\text{train }}$ can be different for different datasets. While most datasets including MS MARCO favored a large enough ${k}_{\text{train }}$ ,ArguAna prefers smaller ${k}_{\text{train }}$ . We hypothesize that this is due to the longer query length in ArguAna,which makes multi-vector models fall short compared to dual-encoders (see GTR vs. T5-ColBERT in Table 2).

训练批次大小与${k}_{\text{train }}$ 在图7中，我们展示了训练XTR期间训练批次大小与${k}_{\text{train }}$ 之间的关系。在这个实验中，我们使用${k}^{\prime } = {40},{000}$ 。显然，XTR大多倾向于较大的训练批次大小，但对于不同的数据集，最优的前${k}_{\text{train }}$ 可能不同。虽然包括MS MARCO在内的大多数数据集都倾向于足够大的${k}_{\text{train }}$ ，但ArguAna更喜欢较小的${k}_{\text{train }}$ 。我们假设这是由于ArguAna中的查询长度较长，这使得多向量模型与双编码器相比表现不佳（见表2中的GTR与T5 - ColBERT）。

### 5.4 Qualitative Analysis

### 5.4 定性分析

Table 6 shows a prediction sample from MS MARCO. For T5-ColBERT, all of the top retrieved tokens are exact lexical matches. Surprisingly, none of the retrieved passages are about the query, demonstrating T5-ColBERT's failure to retrieve tokens from the correct context. In contrast, XTR retrieves fewer exact lexically matching tokens, but the contexts of the retrieved tokens are much more related to the query. This example explains the lower lexical token retrieval probability of XTR compared to T5-ColBERT in Figure 4 (bottom), but higher gold token retrieval performance in Figure 4 (top). For more qualitative examples, please see Appendix E,

表6展示了来自MS MARCO的一个预测样本。对于T5 - ColBERT模型而言，所有排名靠前的检索词元均为精确的词汇匹配。令人惊讶的是，检索到的段落中没有一个与查询内容相关，这表明T5 - ColBERT无法从正确的上下文中检索词元。相比之下，XTR模型检索到的精确词汇匹配词元较少，但检索到的词元上下文与查询内容的相关性要高得多。此示例解释了图4（下方）中XTR与T5 - ColBERT相比词元检索概率较低的原因，但也说明了图4（上方）中XTR在关键词元检索方面表现更优的原因。更多定性示例请见附录E。

## 6 Related Work

## 6 相关工作

One of the main limitations of dense retrieval models is that encoding the query and document into a single vector constrains the representational power of the models. Polyencoder [Humeau et al. 2020], MEBERT [Luan et al. 2021], and MVR [Zhang et al. 2022a] propose to use multiple embeddings, instead of one, to represent the query or the document. A more recent approach is token-level multi-vector retrieval, which stores and retrieves with every token embedding. ColBERT [Khattab and Zaharia, 2020] is probably the most renowned model in this family. ALIGNER (i.e. T5- ColBERT) [Qian et al. 2022] extends ColBERT by scaling up the backbone langauge model and studying various strategies for aggregating the token-level alignment scores. These token-level retrieval models show strong effectiveness and out-of-domain generalization ability.

密集检索模型的主要局限性之一是，将查询和文档编码为单个向量会限制模型的表示能力。多编码器（Polyencoder）[Humeau等人，2020年]、MEBERT [Luan等人，2021年]和MVR [Zhang等人，2022a]提出使用多个嵌入向量而非单个向量来表示查询或文档。最近的一种方法是词元级多向量检索，它使用每个词元嵌入进行存储和检索。ColBERT [Khattab和Zaharia，2020年]可能是这类模型中最著名的模型。ALIGNER（即T5 - ColBERT）[Qian等人，2022年]通过扩大基础语言模型规模并研究聚合词元级对齐分数的各种策略，对ColBERT进行了扩展。这些词元级检索模型表现出了很强的有效性和跨领域泛化能力。

Efforts for reducing serving costs of multi-vector models have been mostly focused on the token-level retrieval stage. COIL [Gao et al. 2021] accelerates token-level retrieval by confining retrieval within exact match tokens, sharing the spirit of classic inverted indexing. CITADEL [Li et al. 2022] relaxes COIL with a lexical routing mechanism where a query token vector only retrieves from a subset of document token vectors routed to the same key. PLAID [Santhanam et al., 2022a] optimizes the speed of ColBERT by pruning weaker candidates in the earlier stages of retrieval and using better vector quantization. ColBERT-v2 [Santhanam et al. 2022b] further adopts residual representations with cluster centroids to improve the efficiency of ColBERT. On the other hand, how to accelerate the scoring stage remains under-explored. To the best of our knowledge, XTR is the first work to simplify the scoring stage and remove the gathering stage in multi-vector retrieval.

降低多向量模型服务成本的工作主要集中在词元级检索阶段。COIL（[高等人，2021年]）通过将检索限制在精确匹配的词元内来加速词元级检索，这与经典的倒排索引原理一致。CITADEL（[李等人，2022年]）通过一种词法路由机制对COIL进行了改进，在该机制中，查询词元向量仅从路由到同一键的文档词元向量子集中进行检索。PLAID（[桑塔南姆等人，2022a]）通过在检索的早期阶段修剪较弱的候选对象并使用更好的向量量化来优化ColBERT的速度。ColBERT - v2（[桑塔南姆等人，2022b]）进一步采用带有聚类质心的残差表示来提高ColBERT的效率。另一方面，如何加速评分阶段仍有待深入研究。据我们所知，XTR是首个简化评分阶段并去除多向量检索中收集阶段的工作。

<!-- Media -->

T5-ColBERT token retrieval for "what is the usual pay for stock associates at michael?"

针对“在迈克尔公司，库存员工的通常工资是多少？”的T5 - ColBERT词元检索

<table><tr><td>Rank</td><td>Token</td><td>Context of Token</td><td>Relevance</td></tr><tr><td>1</td><td>usual</td><td>routine passport services: the usual waiting time in logan to get your passport is four (4) to eight (8) weeks for routine applications.</td><td>No</td></tr><tr><td>2</td><td>usual</td><td>the usual pay days are the 1st and 16th of each month. for annual educational paraprofessionals there is no payroll lag.</td><td>No</td></tr><tr><td>5</td><td>usual</td><td>the usual part xiii tax rate is ${25}\%$ (unless a tax treaty between canada and your home country reduces the rate).</td><td>No</td></tr><tr><td>50</td><td>usual</td><td>this is where one can challenge the judgment debtor's claim. one option creditors have is to try and make a deal with the debtor to take less than 25% (the usual amount of a wage levy).</td><td>No</td></tr><tr><td>100</td><td>usual</td><td>the usual maximum inventory is 1 talisman, 26 elemental runes, and 26 pure essence. the ingredients must be brought to an opposing altar ... from the runes being crafted.</td><td>No</td></tr></table>

<table><tbody><tr><td>排名</td><td>Token</td><td>Token的上下文</td><td>相关性</td></tr><tr><td>1</td><td>通常的</td><td>常规护照服务：在洛根，常规护照申请的通常等待时间为四（4）至八（8）周。</td><td>不</td></tr><tr><td>2</td><td>通常的</td><td>通常的发薪日是每月的1号和16号。对于年度教育辅助人员来说，工资发放没有延迟。</td><td>不</td></tr><tr><td>5</td><td>通常的</td><td>通常的第十三部分税率是${25}\%$（除非加拿大与本国之间的税收协定降低了该税率）。</td><td>不</td></tr><tr><td>50</td><td>通常的</td><td>这里是可以对判决债务人的主张提出质疑的地方。债权人的一个选择是尝试与债务人达成协议，接受低于25%（通常的征税金额）的款项。</td><td>不</td></tr><tr><td>100</td><td>通常的</td><td>通常的最大库存是1个护符、26个元素符文和26份纯净精华。这些材料必须被带到与正在制作的符文相对的祭坛……</td><td>不</td></tr></tbody></table>

XTR token retrieval for "what is the usual pay for stock associates at michael?"

针对“迈克尔公司的库存员工通常薪资是多少？”的XTR令牌检索

<table><tr><td>Rank</td><td>Token</td><td>Context of Token</td><td>Relevance</td></tr><tr><td>1</td><td>usual</td><td>store manager. 1 salary: the usual salary a store manager receives can be anywhere around \$52,000 to \$115,000 annually.</td><td>No</td></tr><tr><td>2</td><td>usual</td><td>1 salary: the usual salary a store manager receives can be anywhere around $\$ {52},{000}$ to $\$ {115},{000}$ annually. 2 bonuses: publix provide bonuses that could reach up to \$40,000.</td><td>No</td></tr><tr><td>5</td><td>average</td><td>average salaries for michaels stores stock associate: \$9. michaels stores hourly pay trends based on salaries posted anonymously by michaels stores employees.</td><td>$\mathbf{{Yes}}$</td></tr><tr><td>50</td><td>V</td><td>i think the avg starting pay is closer to ${30}\mathrm{\;k}$ for asst mgr trainees. it is an hourly position until you are fully trained (40 hours per week).</td><td>No</td></tr><tr><td>100</td><td>average</td><td>average macys salaries. the average salary for macys jobs is $\$ {32},{000}$ . average macys salaries can vary greatly due to company, location, indus- try, experience and benefits.</td><td>No</td></tr></table>

<table><tbody><tr><td>排名</td><td>Token</td><td>Token的上下文</td><td>相关性</td></tr><tr><td>1</td><td>通常的</td><td>商店经理。1. 薪资：商店经理通常的年薪在52000美元到115000美元之间。</td><td>不</td></tr><tr><td>2</td><td>通常的</td><td>1. 薪资：商店经理的通常年薪在$\$ {52},{000}$到$\$ {115},{000}$之间。2. 奖金：大众超市（Publix）提供的奖金最高可达4万美元。</td><td>不</td></tr><tr><td>5</td><td>平均</td><td>迈克尔斯商店（Michaels Stores）库存助理的平均薪资：每小时9美元。迈克尔斯商店的时薪趋势基于该店员工匿名发布的薪资信息。</td><td>$\mathbf{{Yes}}$</td></tr><tr><td>50</td><td>V</td><td>我认为助理经理实习生的平均起薪更接近${30}\mathrm{\;k}$。在你完成全面培训（每周40小时）之前，这是一个按小时计酬的岗位。</td><td>不</td></tr><tr><td>100</td><td>平均</td><td>梅西百货（Macy's）的平均薪资。梅西百货岗位的平均薪资是$\$ {32},{000}$。梅西百货的平均薪资会因公司、地点、行业、经验和福利的不同而有很大差异。</td><td>不</td></tr></tbody></table>

Table 6: Token retrieval example from MS MARCO. Among the top 100 retrieved tokens, 100% of T5-ColBERT tokens are lexically identical as the query token usual while only 8% of XTR tokens are lexically identical. XTR retrieves the relevant passage by retrieving average for usual.

表6：来自MS MARCO的Token检索示例。在检索到的前100个Token中，T5 - ColBERT的Token有100%在词法上与查询Token“usual”相同，而XTR的Token只有8%在词法上相同。XTR通过为“usual”检索“average”来检索相关段落。

<!-- Media -->

## 7 Conclusion

## 7 结论

Multi-vector retrieval leverages query and document token representations for effective information retrieval. In this paper, we propose XTR that simplifies the existing three-stage inference of multi-vector models by improving the initial token retrieval stage. Specifically, XTR scores documents solely based on the retrieved tokens, which is also optimized during training with in-batch document tokens. As a result, XTR achieves state-of-the-art performances on zero-shot information retrieval benchmarks while greatly reducing the FLOPs of the scoring stage. We further show that our objective function indeed encourages better token retrieval, retrieving more tokens from gold documents, whose contexts are better aligned with the query.

多向量检索利用查询和文档的Token表示进行有效的信息检索。在本文中，我们提出了XTR，它通过改进初始Token检索阶段，简化了现有的多向量模型的三阶段推理。具体来说，XTR仅根据检索到的Token对文档进行评分，这在训练过程中也会使用批次内的文档Token进行优化。因此，XTR在零样本信息检索基准测试中达到了最先进的性能，同时大大减少了评分阶段的浮点运算次数。我们进一步表明，我们的目标函数确实鼓励更好的Token检索，从黄金文档中检索更多的Token，这些Token的上下文与查询的对齐效果更好。

## Limitations

## 局限性

In most of our experiments, XTR was trained on MS MARCO, a large-scale retrieval dataset in English. While our experiments were conducted in a fair setting where most baseline models also utilize MS MARCO, future use cases might need to remove its dependency on MS MARCO due to the license or language-specific issue. We believe that LLM-based retrieval dataset generation [Dai et al. 2022] would be able to mitigate the problem in the future.

在我们的大多数实验中，XTR是在MS MARCO（一个大规模的英文检索数据集）上进行训练的。虽然我们的实验是在一个公平的环境中进行的，大多数基线模型也使用了MS MARCO，但由于许可或特定语言问题，未来的用例可能需要消除对MS MARCO的依赖。我们相信，基于大语言模型（LLM）的检索数据集生成方法[戴等人，2022年]未来能够缓解这一问题。

## Acknowledgements

## 致谢

We would like to thank the anonymous reviewers for their helpful feedback. We also thank Nicholas Monath, Raphael Hoffmann, Kelvin Guu, Slav Petrov, and others at Google DeepMind for their helpful comments and discussion.

我们要感谢匿名审稿人提供的有益反馈。我们还要感谢谷歌DeepMind的尼古拉斯·莫纳思（Nicholas Monath）、拉斐尔·霍夫曼（Raphael Hoffmann）、凯文·古（Kelvin Guu）、斯拉夫·彼得罗夫（Slav Petrov）以及其他人员提供的有益评论和讨论。

## References

## 参考文献

Negar Arabzadeh, Alexandra Vtyurina, Xinyi Yan, and Charles LA Clarke. Shallow pooling for sparse labels. Information Retrieval Journal, 25(4):365-385, 2022.

内加尔·阿拉布扎德赫（Negar Arabzadeh）、亚历山德拉·夫秋里娜（Alexandra Vtyurina）、严心怡（Xinyi Yan）和查尔斯·L·A·克拉克（Charles LA Clarke）。稀疏标签的浅层池化。《信息检索期刊》，25(4):365 - 385，2022年。

Xilun Chen, Kushal Lakhotia, Barlas Oğuz, Anchit Gupta, Patrick Lewis, Stan Peshterliev, Yashar Mehdad, Sonal Gupta, and Wen-tau Yih. Salient phrase aware dense retrieval: Can a dense retriever imitate a sparse one? arXiv preprint arXiv:2110.06918, 2021.

陈希伦（Xilun Chen）、库沙尔·拉科蒂亚（Kushal Lakhotia）、巴拉斯·奥古兹（Barlas Oğuz）、安奇特·古普塔（Anchit Gupta）、帕特里克·刘易斯（Patrick Lewis）、斯坦·佩什特列夫（Stan Peshterliev）、亚沙尔·梅赫达德（Yashar Mehdad）、索纳尔·古普塔（Sonal Gupta）和易文涛（Wen-tau Yih）。显著短语感知的密集检索：密集检索器能否模仿稀疏检索器？预印本 arXiv:2110.06918，2021年。

Zhuyun Dai, Vincent Y Zhao, Ji Ma, Yi Luan, Jianmo Ni, Jing Lu, Anton Bakalov, Kelvin Guu, Keith B Hall, and Ming-Wei Chang. Promptagator: Few-shot dense retrieval from 8 examples. arXiv preprint arXiv:2209.11755, 2022.

戴竹云（Zhuyun Dai）、文森特·Y·赵（Vincent Y Zhao）、马骥（Ji Ma）、栾义（Yi Luan）、倪建谟（Jianmo Ni）、卢静（Jing Lu）、安东·巴卡洛夫（Anton Bakalov）、凯尔文·顾（Kelvin Guu）、基思·B·霍尔（Keith B Hall）和张明伟（Ming-Wei Chang）。Promptagator：基于8个示例的少样本密集检索。预印本 arXiv:2209.11755，2022年。

Thibault Formal, Carlos Lassance, Benjamin Piwowarski, and Stéphane Clinchant. Splade v2: Sparse lexical and expansion model for information retrieval. arXiv preprint arXiv:2109.10086, 2021.

蒂博·福尔马尔（Thibault Formal）、卡洛斯·拉桑斯（Carlos Lassance）、本杰明·皮沃瓦尔斯基（Benjamin Piwowarski）和斯特凡·克兰尚（Stéphane Clinchant）。Splade v2：用于信息检索的稀疏词法和扩展模型。预印本 arXiv:2109.10086，2021年。

Luyu Gao, Zhuyun Dai, and Jamie Callan. COIL: revisit exact lexical match in information retrieval with contextualized inverted list. In Kristina Toutanova, Anna Rumshisky, Luke Zettlemoyer, Dilek Hakkani-Tür, Iz Beltagy, Steven Bethard, Ryan Cotterell, Tanmoy Chakraborty, and Yichao Zhou, editors, Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2021, Online, June 6-11, 2021, pages 3030-3042. Association for Computational Linguistics, 2021.

高璐宇（Luyu Gao）、戴竹云（Zhuyun Dai）和杰米·卡兰（Jamie Callan）。COIL：利用上下文倒排列表重新审视信息检索中的精确词汇匹配。收录于克里斯蒂娜·图托纳娃（Kristina Toutanova）、安娜·鲁姆希斯基（Anna Rumshisky）、卢克·泽特尔莫耶（Luke Zettlemoyer）、迪莱克·哈卡尼 - 图尔（Dilek Hakkani-Tür）、伊兹·贝尔塔吉（Iz Beltagy）、史蒂文·贝瑟德（Steven Bethard）、瑞安·科特雷尔（Ryan Cotterell）、坦莫伊·查克拉博蒂（Tanmoy Chakraborty）和周轶超（Yichao Zhou）主编的《2021年北美计算语言学协会人类语言技术会议论文集》（Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies），NAACL - HLT 2021，线上会议，2021年6月6 - 11日，第3030 - 3042页。计算语言学协会，2021年。

Ruiqi Guo, Philip Sun, Erik Lindgren, Quan Geng, David Simcha, Felix Chern, and Sanjiv Kumar. Accelerating large-scale inference with anisotropic vector quantization. In International Conference on Machine Learning, pages 3887-3896. PMLR, 2020.

郭瑞琪（Ruiqi Guo）、菲利普·孙（Philip Sun）、埃里克·林德格伦（Erik Lindgren）、耿权（Quan Geng）、大卫·辛查（David Simcha）、费利克斯·陈（Felix Chern）和桑吉夫·库马尔（Sanjiv Kumar）。利用各向异性向量量化加速大规模推理。收录于《国际机器学习会议》（International Conference on Machine Learning），第3887 - 3896页。机器学习研究会议录（PMLR），2020年。

Samuel Humeau, Kurt Shuster, Marie-Anne Lachaux, and Jason Weston. Poly-encoders: Architectures and pre-training strategies for fast and accurate multi-sentence scoring. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020.

塞缪尔·于莫（Samuel Humeau）、库尔特·舒斯特（Kurt Shuster）、玛丽 - 安妮·拉肖（Marie - Anne Lachaux）和杰森·韦斯顿（Jason Weston）。多编码器：用于快速准确多句子评分的架构和预训练策略。见第8届国际学习表征会议（8th International Conference on Learning Representations，ICLR 2020），2020年4月26 - 30日，埃塞俄比亚亚的斯亚贝巴。OpenReview.net，2020年。

Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. Unsupervised dense information retrieval with contrastive learning. Transactions on Machine Learning Research, 2022.

高蒂埃·伊扎卡尔（Gautier Izacard）、玛蒂尔德·卡龙（Mathilde Caron）、卢卡斯·侯赛尼（Lucas Hosseini）、塞巴斯蒂安·里德尔（Sebastian Riedel）、彼得·博亚诺夫斯基（Piotr Bojanowski）、阿尔芒·朱兰（Armand Joulin）和爱德华·格雷夫（Edouard Grave）。基于对比学习的无监督密集信息检索。《机器学习研究汇刊》（Transactions on Machine Learning Research），2022年。

Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Yu Wu, Sergey Edunov, Danqi Chen, and Wen tau Yih. Dense passage retrieval for open-domain question answering. ArXiv, abs/2004.04906, 2020.

弗拉基米尔·卡尔普欣（Vladimir Karpukhin）、巴拉斯·奥古兹（Barlas Oğuz）、闵世元（Sewon Min）、帕特里克·刘易斯（Patrick Lewis）、余文涛（Ledell Yu Wu）、谢尔盖·叶杜诺夫（Sergey Edunov）、陈丹琦（Danqi Chen）和易文涛（Wen tau Yih）。开放域问答的密集段落检索。预印本平台ArXiv，编号abs/2004.04906，2020年。

Omar Khattab and Matei Zaharia. Colbert: Efficient and effective passage search via contextualized late interaction over bert. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, pages 39-48, 2020.

奥马尔·哈塔卜（Omar Khattab）和马泰·扎哈里亚（Matei Zaharia）。《科尔伯特：通过基于BERT的上下文延迟交互实现高效有效的段落搜索》。发表于第43届ACM信息检索研究与发展国际会议论文集，第39 - 48页，2020年。

Kenton Lee, Ming-Wei Chang, and Kristina Toutanova. Latent retrieval for weakly supervised open domain question answering. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 6086-6096, Florence, Italy, July 2019. Association for Computational Linguistics.

肯顿·李（Kenton Lee）、张明伟（Ming - Wei Chang）和克里斯蒂娜·图托纳娃（Kristina Toutanova）。《弱监督开放域问答的潜在检索》。发表于第57届计算语言学协会年会论文集，第6086 - 6096页，意大利佛罗伦萨，2019年7月。计算语言学协会。

Minghan Li, Sheng-Chieh Lin, Barlas Oguz, Asish Ghoshal, Jimmy Lin, Yashar Mehdad, Wen-tau Yih, and Xilun Chen. Citadel: Conditional token interaction via dynamic lexical routing for efficient and effective multi-vector retrieval. arXiv preprint arXiv:2211.10411, 2022.

李明翰（Minghan Li）、林圣杰（Sheng - Chieh Lin）、巴拉斯·奥古兹（Barlas Oguz）、阿西什·戈沙尔（Asish Ghoshal）、吉米·林（Jimmy Lin）、亚沙尔·梅赫达德（Yashar Mehdad）、易文涛（Wen - tau Yih）和陈希伦（Xilun Chen）。《城堡：通过动态词法路由实现条件化词元交互以进行高效有效的多向量检索》。预印本arXiv:2211.10411，2022年。

Yi Luan, Jacob Eisenstein, Kristina Toutanova, and Michael Collins. Sparse, Dense, and Attentional Representations for Text Retrieval. Transactions of the Association for Computational Linguistics, 9:329-345, 04 2021.

栾义（Yi Luan）、雅各布·艾森斯坦（Jacob Eisenstein）、克里斯蒂娜·图托纳娃（Kristina Toutanova）和迈克尔·柯林斯（Michael Collins）。用于文本检索的稀疏、密集和注意力表示。《计算语言学协会汇刊》（Transactions of the Association for Computational Linguistics），9:329 - 345，2021年4月。

Craig Macdonald and Nicola Tonellotto. On approximate nearest neighbour selection for multi-stage dense retrieval. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management, pages 3318-3322, 2021.

克雷格·麦克唐纳（Craig Macdonald）和尼古拉·托内洛托（Nicola Tonellotto）。关于多级密集检索的近似最近邻选择。《第30届ACM国际信息与知识管理大会论文集》（Proceedings of the 30th ACM International Conference on Information & Knowledge Management），第3318 - 3322页，2021年。

Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Hern'andez 'Abrego, Ji Ma, Vincent Zhao, Yi Luan, Keith B. Hall, Ming-Wei Chang, and Yinfei Yang. Large dual encoders are generalizable retrievers. In Conference on Empirical Methods in Natural Language Processing, 2021.

倪建谟（Jianmo Ni）、曲晨（Chen Qu）、卢静（Jing Lu）、戴竹云（Zhuyun Dai）、古斯塔沃·埃尔南德斯·阿夫雷戈（Gustavo Hern'andez 'Abrego）、马骥（Ji Ma）、赵文森（Vincent Zhao）、栾义（Yi Luan）、基思·B·霍尔（Keith B. Hall）、张明伟（Ming - Wei Chang）和杨荫飞（Yinfei Yang）。大型双编码器是可泛化的检索器。《自然语言处理经验方法会议》（Conference on Empirical Methods in Natural Language Processing），2021年。

Yujie Qian, Jinhyuk Lee, Sai Meher Karthik Duddu, Zhuyun Dai, Siddhartha Brahma, Iftekhar Naim, Tao Lei, and Vincent Y Zhao. Multi-vector retrieval as sparse alignment. arXiv preprint arXiv:2211.01267, 2022.

钱玉洁（Yujie Qian）、李晋赫（Jinhyuk Lee）、赛·梅赫尔·卡尔蒂克·杜杜（Sai Meher Karthik Duddu）、戴竹云（Zhuyun Dai）、悉达多·布拉马（Siddhartha Brahma）、伊夫特哈尔·奈姆（Iftekhar Naim）、雷涛（Tao Lei）和赵文森（Vincent Y Zhao）。多向量检索作为稀疏对齐。预印本arXiv:2211.01267，2022年。

Yingqi Qu, Yuchen Ding, Jing Liu, Kai Liu, Ruiyang Ren, Wayne Xin Zhao, Daxiang Dong, Hua Wu, and Haifeng Wang. Rocketqa: An optimized training approach to dense passage retrieval for open-domain question answering. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 5835-5847, 2021.

曲英琦、丁宇辰、刘静、刘凯、任瑞阳、赵新宇（Wayne Xin Zhao）、董大祥、吴华和王海峰。Rocketqa：一种用于开放域问答的密集段落检索优化训练方法。见《2021年北美计算语言学协会人类语言技术会议论文集》，第5835 - 5847页，2021年。

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, 21(1):5485-5551, 2020.

科林·拉菲尔（Colin Raffel）、诺姆·沙泽尔（Noam Shazeer）、亚当·罗伯茨（Adam Roberts）、凯瑟琳·李（Katherine Lee）、沙兰·纳朗（Sharan Narang）、迈克尔·马特纳（Michael Matena）、周燕琪、李威和刘彼得（Peter J Liu）。用统一的文本到文本转换器探索迁移学习的极限。《机器学习研究杂志》，21(1):5485 - 5551，2020年。

Ori Ram, Gal Shachaf, Omer Levy, Jonathan Berant, and Amir Globerson. Learning to retrieve passages without supervision. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2687-2700, 2022.

奥里·拉姆（Ori Ram）、加尔·沙查夫（Gal Shachaf）、奥默·利维（Omer Levy）、乔纳森·贝兰特（Jonathan Berant）和阿米尔·格洛伯森（Amir Globerson）。无监督学习段落检索。见《2022年北美计算语言学协会人类语言技术会议论文集》，第2687 - 2700页，2022年。

Parikshit Ram and Alexander G Gray. Maximum inner-product search using cone trees. In Proceedings of the 18th ACM SIGKDD international conference on Knowledge discovery and data mining, pages 931-939, 2012.

帕里克希特·拉姆（Parikshit Ram）和亚历山大·G·格雷（Alexander G Gray）。使用锥树进行最大内积搜索。见《第18届ACM SIGKDD国际知识发现与数据挖掘会议论文集》，第931 - 939页，2012年。

Devendra Singh Sachan, Mike Lewis, Mandar Joshi, Armen Aghajanyan, Wen-tau Yih, Joelle Pineau, and Luke Zettlemoyer. Improving passage retrieval with zero-shot question generation. arXiv preprint arXiv:2204.07496, 2022a.

德文德拉·辛格·萨坎（Devendra Singh Sachan）、迈克·刘易斯（Mike Lewis）、曼达尔·乔希（Mandar Joshi）、阿尔缅·阿加贾尼扬（Armen Aghajanyan）、文涛·易（Wen - tau Yih）、乔埃尔·皮诺（Joelle Pineau）和卢克·泽特尔莫耶（Luke Zettlemoyer）。通过零样本问题生成改进段落检索。预印本arXiv:2204.07496，2022a。

Devendra Singh Sachan, Mike Lewis, Dani Yogatama, Luke Zettlemoyer, Joelle Pineau, and Manzil Zaheer. Questions are all you need to train a dense passage retriever. arXiv preprint arXiv:2206.10658, 2022b.

德文德拉·辛格·萨坎（Devendra Singh Sachan）、迈克·刘易斯（Mike Lewis）、达尼·约加塔马（Dani Yogatama）、卢克·泽特尔莫耶（Luke Zettlemoyer）、乔埃尔·皮诺（Joelle Pineau）和曼齐尔·扎希尔（Manzil Zaheer）。训练密集段落检索器只需问题。预印本arXiv:2206.10658，2022b。

Keshav Santhanam, Omar Khattab, Christopher Potts, and Matei Zaharia. Plaid: an efficient engine for late interaction retrieval. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management, pages 1747-1756, 2022a.

凯沙夫·桑塔纳姆（Keshav Santhanam）、奥马尔·哈塔卜（Omar Khattab）、克里斯托弗·波茨（Christopher Potts）和马泰·扎哈里亚（Matei Zaharia）。Plaid：用于延迟交互检索的高效引擎。见《第31届ACM信息与知识管理国际会议论文集》，第1747 - 1756页，2022a。

Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. Colbertv2: Effective and efficient retrieval via lightweight late interaction. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3715-3734, 2022b.

凯沙夫·桑塔纳姆（Keshav Santhanam）、奥马尔·哈塔卜（Omar Khattab）、乔恩·萨德 - 法尔孔（Jon Saad - Falcon）、克里斯托弗·波茨（Christopher Potts）和马泰·扎哈里亚（Matei Zaharia）。Colbertv2：通过轻量级延迟交互实现高效检索。见《2022年计算语言学协会北美分会人类语言技术会议论文集》，第3715 - 3734页，2022b。

Christopher Sciavolino, Zexuan Zhong, Jinhyuk Lee, and Danqi Chen. Simple entity-centric questions challenge dense retrievers. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 6138-6148, 2021.

克里斯托弗·夏沃利诺（Christopher Sciavolino）、钟泽轩（Zexuan Zhong）、李晋赫（Jinhyuk Lee）和陈丹琦（Danqi Chen）。简单的以实体为中心的问题对密集检索器构成挑战。见《2021年自然语言处理经验方法会议论文集》，第6138 - 6148页，2021。

Fumin Shen, Wei Liu, Shaoting Zhang, Yang Yang, and Heng Tao Shen. Learning binary codes for maximum inner product search. In Proceedings of the IEEE International Conference on Computer Vision, pages 4148-4156, 2015.

沈富敏（Fumin Shen）、刘伟（Wei Liu）、张绍廷（Shaoting Zhang）、杨洋（Yang Yang）和沈恒涛（Heng Tao Shen）。学习用于最大内积搜索的二进制代码。见《IEEE国际计算机视觉会议论文集》，第4148 - 4156页，2015。

Anshumali Shrivastava and Ping Li. Asymmetric lsh (alsh) for sublinear time maximum inner product search (mips). In Advances in Neural Information Processing Systems, pages 2321-2329, 2014.

安舒马利·什里瓦斯塔瓦（Anshumali Shrivastava）和平·李（Ping Li）。用于次线性时间最大内积搜索（MIPS）的非对称局部敏感哈希（ALSH）。《神经信息处理系统进展》，第2321 - 2329页，2014年。

Anshumali Shrivastava and Ping Li. Improved asymmetric locality sensitive hashing (alsh) for maximum inner product search (mips). In Conference on Uncertainty in Artificial Intelligence, 2015.

安舒马利·什里瓦斯塔瓦（Anshumali Shrivastava）和平·李（Ping Li）。用于最大内积搜索（MIPS）的改进型非对称局部敏感哈希（ALSH）。《人工智能中的不确定性会议》，2015年。

Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. Beir: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021.

南丹·塔库尔（Nandan Thakur）、尼尔斯·赖默斯（Nils Reimers）、安德里亚斯·吕克莱（Andreas Rücklé）、阿比舍克·斯里瓦斯塔瓦（Abhishek Srivastava）和伊琳娜·古列维奇（Iryna Gurevych）。BEIR：用于信息检索模型零样本评估的异构基准。《第三十五届神经信息处理系统数据集与基准赛道会议（第二轮）》，2021年。

Kexin Wang, Nandan Thakur, Nils Reimers, and Iryna Gurevych. Gpl: Generative pseudo labeling for unsupervised domain adaptation of dense retrieval. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2345-2360, 2022.

王可欣（Kexin Wang）、南丹·塔库尔（Nandan Thakur）、尼尔斯·赖默斯（Nils Reimers）和伊琳娜·古列维奇（Iryna Gurevych）。GPL：用于密集检索无监督领域适应的生成式伪标签。《计算语言学协会北美分会2022年人类语言技术会议论文集》，第2345 - 2360页，2022年。

Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. mt5: A massively multilingual pre-trained text-to-text transformer. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 483-498, 2021.

薛林亭（Linting Xue）、诺亚·康斯坦特（Noah Constant）、亚当·罗伯茨（Adam Roberts）、米希尔·卡尔（Mihir Kale）、拉米·阿尔-鲁福（Rami Al-Rfou）、阿迪亚·悉檀特（Aditya Siddhant）、阿迪亚·巴鲁阿（Aditya Barua）和科林·拉菲尔（Colin Raffel）。mt5：一种大规模多语言预训练的文本到文本转换器。见《2021年北美计算语言学协会人类语言技术会议论文集》，第483 - 498页，2021年。

Wen-tau Yih, Kristina Toutanova, John C. Platt, and Christopher Meek. Learning discriminative projections for text similarity measures. In Conference on Computational Natural Language Learning, 2011.

易文涛（Wen-tau Yih）、克里斯蒂娜·图托纳娃（Kristina Toutanova）、约翰·C·普拉特（John C. Platt）和克里斯托弗·米克（Christopher Meek）。学习文本相似度度量的判别投影。见《计算自然语言学习会议》，2011年。

Shunyu Zhang, Yaobo Liang, Ming Gong, Daxin Jiang, and Nan Duan. Multi-view document representation learning for open-domain dense retrieval. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022, pages 5990-6000. Association for Computational Linguistics, 2022a.

张顺宇（Shunyu Zhang）、梁耀波（Yaobo Liang）、龚鸣（Ming Gong）、蒋大新（Daxin Jiang）和段楠（Nan Duan）。用于开放域密集检索的多视图文档表示学习。见斯马兰达·穆雷桑（Smaranda Muresan）、普雷斯拉夫·纳科夫（Preslav Nakov）和阿琳·比利亚维森西奥（Aline Villavicencio）编，《第60届计算语言学协会年会论文集（第1卷：长论文）》，ACL 2022，爱尔兰都柏林，2022年5月22 - 27日，第5990 - 6000页。计算语言学协会，2022a。

Xinyu Zhang, Nandan Thakur, Odunayo Ogundepo, Ehsan Kamalloo, David Alfonso-Hermelo, Xiaoguang Li, Qun Liu, Mehdi Rezagholizadeh, and Jimmy Lin. Making a miracl: Multilingual information retrieval across a continuum of languages. arXiv preprint arXiv:2210.09984, 2022b.

张新宇（Xinyu Zhang）、南丹·塔库尔（Nandan Thakur）、奥敦纳约·奥贡德波（Odunayo Ogundepo）、埃桑·卡马卢（Ehsan Kamalloo）、大卫·阿方索 - 埃尔梅洛（David Alfonso-Hermelo）、李晓光（Xiaoguang Li）、刘群（Qun Liu）、迈赫迪·雷扎戈利扎德（Mehdi Rezagholizadeh）和吉米·林（Jimmy Lin）。创造奇迹：跨连续语言的多语言信息检索。预印本 arXiv:2210.09984，2022b。

## A Derivatives w.r.t. Similarity Scores

## 关于相似度得分的导数

Sum-of-max Here,we use a cross-entropy loss ${\mathcal{L}}_{\mathrm{{CE}}}$ with the sum-of-max operator ${f}_{\text{ColBERT }}$ and analyze the derivatives with respect to the token similarity scores.

最大和法 在这里，我们使用带有最大和算子 ${f}_{\text{ColBERT }}$ 的交叉熵损失 ${\mathcal{L}}_{\mathrm{{CE}}}$，并分析关于词元相似度得分的导数。

$$
{\mathcal{L}}_{\mathrm{{CE}}} =  - \log \frac{\exp f\left( {Q,{D}^{ + }}\right) }{\mathop{\sum }\limits_{{b = 1}}^{B}\exp f\left( {Q,{D}_{b}}\right) } =  - {f}_{\text{ColBERT }}\left( {Q,{D}^{ + }}\right)  + \log \mathop{\sum }\limits_{{b = 1}}^{B}\exp {f}_{\text{ColBERT }}\left( {Q,{D}_{b}}\right)  \tag{5}
$$

$$
{f}_{\text{ColBERT }}\left( {Q,D}\right)  = \frac{1}{n}\mathop{\sum }\limits_{{i = 1}}^{n}\mathop{\sum }\limits_{{j = 1}}^{m}{\mathbf{A}}_{ij}{\mathbf{P}}_{ij} = \frac{1}{n}\mathop{\sum }\limits_{{i = 1}}^{n}{\mathbf{P}}_{i\widehat{j}} \tag{6}
$$

Here,we denote $\widehat{j}$ as the index of the row-wise maximum value,dependent on each $i$ (i.e., ${\mathbf{A}}_{ij} = 1$ ). Given the cross-entropy loss with the sum-of-max operator, we compute the gradient with respect to one of the maximum token similarities ${\mathbf{P}}_{i\widehat{j}}^{ + }$ for a positive document ${D}^{ + } \in  {D}_{1 : B}$ :

在这里，我们将 $\widehat{j}$ 表示为逐行最大值的索引，该索引取决于每个 $i$（即 ${\mathbf{A}}_{ij} = 1$）。给定带有最大和算子的交叉熵损失，我们计算正文档 ${D}^{ + } \in  {D}_{1 : B}$ 相对于最大标记相似度之一 ${\mathbf{P}}_{i\widehat{j}}^{ + }$ 的梯度：

$$
\frac{\partial {\mathcal{L}}_{\mathrm{{CE}}}}{\partial {\mathbf{P}}_{i\widehat{j}}^{ + }} =  - \frac{f\left( {Q,{D}^{ + }}\right) }{\partial {\mathbf{P}}_{i\widehat{j}}^{ + }} + \frac{1}{\mathop{\sum }\limits_{{b = 1}}^{B}\exp f\left( {Q,{D}_{b}}\right) }\frac{\partial }{\partial {\mathbf{P}}_{i\widehat{j}}^{ + }}\mathop{\sum }\limits_{{b = 1}}^{B}\exp f\left( {Q,{D}_{b}}\right) 
$$

$$
 =  - \frac{\partial }{\partial {\mathbf{P}}_{i\widehat{j}}^{ + }}\frac{1}{n}\mathop{\sum }\limits_{{i = 1}}^{n}\mathop{\max }\limits_{{1 \leq  j \leq  m}}{\mathbf{P}}_{ij}^{ + } + \frac{1}{\mathop{\sum }\limits_{{b = 1}}^{B}\exp f\left( {Q,{D}_{b}}\right) }\mathop{\sum }\limits_{{b = 1}}^{B}\frac{\partial }{\partial {\mathbf{P}}_{i\widehat{j}}^{ + }}\exp f\left( {Q,{D}_{b}}\right) 
$$

$$
 =  - \frac{1}{n} + \frac{1}{\mathop{\sum }\limits_{{b = 1}}^{B}\exp f\left( {Q,{D}_{b}}\right) }\mathop{\sum }\limits_{{b = 1}}^{B}\exp f\left( {Q,{D}_{b}}\right) \frac{\partial f\left( {Q,{D}_{b}}\right) }{\partial {\mathbf{P}}_{i\widehat{j}}^{ + }}
$$

$$
 =  - \frac{1}{n} + \frac{1}{n}\frac{\exp f\left( {Q,{D}^{ + }}\right) }{\mathop{\sum }\limits_{{b = 1}}^{B}\exp f\left( {Q,{D}_{b}}\right) } =  - \frac{1}{n}\left\lbrack  {1 - P\left( {{D}^{ + } \mid  Q,{D}_{1 : B}}\right) }\right\rbrack  .
$$

Similarly,the gradient w.r.t. a maximum token similarity ${\mathbf{P}}_{i\widehat{j}}^{ - }$ for a negative document ${D}^{ - } \in  {D}_{1 : B}$ is computed as follows:

类似地，负文档 ${D}^{ - } \in  {D}_{1 : B}$ 相对于最大标记相似度 ${\mathbf{P}}_{i\widehat{j}}^{ - }$ 的梯度计算如下：

$$
\frac{\partial {\mathcal{L}}_{\mathrm{{CE}}}}{\partial {\mathbf{P}}_{i\widehat{j}}^{ - }} =  - \frac{f\left( {Q,{D}^{ + }}\right) }{\partial {\mathbf{P}}_{i\widehat{j}}^{ - }} + \frac{1}{\mathop{\sum }\limits_{{b = 1}}^{B}\exp f\left( {Q,{D}_{b}}\right) }\frac{\partial }{\partial {\mathbf{P}}_{i\widehat{j}}^{ - }}\mathop{\sum }\limits_{{b = 1}}^{B}\exp f\left( {Q,{D}_{b}}\right) 
$$

$$
 = \frac{1}{n}\frac{\exp f\left( {Q,{D}^{ - }}\right) }{\mathop{\sum }\limits_{{b = 1}}^{B}\exp f\left( {Q,{D}_{b}}\right) } = \frac{1}{n}P\left( {{D}^{ - } \mid  Q,{D}_{1 : B}}\right) .
$$

Hence,the positive token-level score ${\mathbf{P}}_{i\widehat{j}}^{ + }$ will gradually increase until $P\left( {{D}^{ + } \mid  Q,{D}_{1 : B}}\right)  \rightarrow  1$ and the negative token-level score ${\mathbf{P}}_{i\widehat{j}}^{ - }$ will decrease until $P\left( {{D}^{ - } \mid  Q,{D}_{1 : B}}\right)  \rightarrow  0$ . This shows that the token-level scores are trained based on the document-level scores, which might stagnate the token-level scores. For instance,even if ${\mathbf{P}}_{i\widehat{j}}^{ - }$ is very high-later causing ${\mathbf{d}}_{\widehat{j}}^{ - }$ to be retrieved instead of ones from positive documents—it will not be penalized as long as $P\left( {{D}^{ - } \mid  Q,{D}_{1 : B}}\right)$ is low enough.

因此，正的词元级得分 ${\mathbf{P}}_{i\widehat{j}}^{ + }$ 将逐渐增加，直至达到 $P\left( {{D}^{ + } \mid  Q,{D}_{1 : B}}\right)  \rightarrow  1$，而负的词元级得分 ${\mathbf{P}}_{i\widehat{j}}^{ - }$ 将逐渐降低，直至达到 $P\left( {{D}^{ - } \mid  Q,{D}_{1 : B}}\right)  \rightarrow  0$。这表明词元级得分是基于文档级得分进行训练的，这可能会使词元级得分陷入停滞。例如，即使 ${\mathbf{P}}_{i\widehat{j}}^{ - }$ 非常高（这随后会导致检索到 ${\mathbf{d}}_{\widehat{j}}^{ - }$ 而非正文档中的内容），只要 $P\left( {{D}^{ - } \mid  Q,{D}_{1 : B}}\right)$ 足够低，就不会受到惩罚。

In-batch token retrieval Compared to the sum-of-max operator,our in-batch sum-of-max ${f}_{\mathrm{{XTR}}}$ considers the max values only when they are retrieved over other negative tokens in the mini-batch.

批量内词元检索 与最大和算子相比，我们的批量内最大和 ${f}_{\mathrm{{XTR}}}$ 仅在小批量中的其他负词元被检索时才考虑最大值。

$$
{f}_{\mathrm{{XTR}}}\left( {Q,{D}_{1 : B}}\right)  = \frac{1}{Z}\mathop{\sum }\limits_{{i = 1}}^{n}\mathop{\sum }\limits_{{j = 1}}^{m}{\mathbf{A}}_{ij}{\widehat{\mathbf{A}}}_{ij}{\mathbf{P}}_{ij} = \frac{1}{Z}\mathop{\sum }\limits_{{i = 1}}^{n}{\mathbf{P}}_{i\bar{j}}
$$

Here,we denote $\bar{j}$ as the index of the row-wise maximum value that is also within the mini-batch top- ${k}_{\text{train }}$ given ${q}_{i}$ (i.e.,satisfies both ${\mathbf{A}}_{ij} = 1$ and ${\widehat{\mathbf{A}}}_{ij} = 1$ ). If there is no such $\bar{j}$ ,we simply use ${\mathbf{P}}_{i\bar{j}} = 0$ . We also use a normalizer $Z$ ,which is the number of non-zero ${\mathbf{P}}_{i\bar{j}}$ . In this analysis,we assume $Z > 0$ since if every ${\mathbf{P}}_{i\bar{j}}$ is zero,the gradient is undefined.

在此，我们将 $\bar{j}$ 表示为行方向最大值的索引，该最大值也在给定 ${q}_{i}$ 的小批量前 ${k}_{\text{train }}$ 范围内（即同时满足 ${\mathbf{A}}_{ij} = 1$ 和 ${\widehat{\mathbf{A}}}_{ij} = 1$ ）。如果不存在这样的 $\bar{j}$ ，我们直接使用 ${\mathbf{P}}_{i\bar{j}} = 0$ 。我们还使用一个归一化因子 $Z$ ，它是非零 ${\mathbf{P}}_{i\bar{j}}$ 的数量。在这个分析中，我们假设 $Z > 0$ ，因为如果每个 ${\mathbf{P}}_{i\bar{j}}$ 都为零，梯度是未定义的。

The gradient w.r.t. the maximum token similarity ${\mathbf{P}}_{i\bar{j}}^{ + }$ (non-zero) for a positive document ${D}^{ + } \in  {D}_{1 : B}$ is computed as follows:

关于正文档 ${D}^{ + } \in  {D}_{1 : B}$ 的最大词元相似度 ${\mathbf{P}}_{i\bar{j}}^{ + }$（非零）的梯度计算如下：

$$
\frac{\partial {\mathcal{L}}_{\mathrm{{CE}}}}{\partial {\mathbf{P}}_{i\bar{j}}^{ + }} =  - \frac{f\left( {Q,{D}^{ + }}\right) }{\partial {\mathbf{P}}_{i\bar{j}}^{ + }} + \frac{1}{\mathop{\sum }\limits_{{b = 1}}^{B}\exp f\left( {Q,{D}_{b}}\right) }\frac{\partial }{\partial {\mathbf{P}}_{i\bar{j}}^{ + }}\mathop{\sum }\limits_{{b = 1}}^{B}\exp f\left( {Q,{D}_{b}}\right) 
$$

$$
 =  - \frac{1}{{Z}^{ + }}\left\lbrack  {1 - \frac{\exp f\left( {Q,{D}^{ + }}\right) }{\mathop{\sum }\limits_{{b = 1}}^{B}\exp f\left( {Q,{D}_{b}}\right) }}\right\rbrack  
$$

$$
 =  - \frac{1}{{Z}^{ + }}\left\lbrack  {1 - P\left( {{D}^{ + } \mid  Q,{D}_{1 : B}}\right) }\right\rbrack  .
$$

This is a very similar result compared to the sum-of-max operator except that 1) the gradient is defined only when ${\mathbf{P}}_{i\bar{j}}^{ + }$ is non-zero (i.e. retrieved) and 2) it is dependent on ${Z}^{ + }$ ,which means that the gradient will be large whenever there is a small number of retrieved tokens from the positive document. If only a handful of tokens are retrieved for ${D}^{ + }$ ,our objective function increases ${\mathbf{P}}_{i\bar{j}}^{ + }$ .

与最大和算子相比，这是一个非常相似的结果，不同之处在于：1) 仅当 ${\mathbf{P}}_{i\bar{j}}^{ + }$ 非零（即被检索到）时才定义梯度；2) 它依赖于 ${Z}^{ + }$，这意味着每当从正文档中检索到的词元数量较少时，梯度就会很大。如果为 ${D}^{ + }$ 仅检索到少数词元，我们的目标函数会增大 ${\mathbf{P}}_{i\bar{j}}^{ + }$。

For negative similarity score ${\mathbf{P}}_{i\bar{j}}^{ - }$ ,we have the following:

对于负相似度得分 ${\mathbf{P}}_{i\bar{j}}^{ - }$，我们有以下情况：

$$
\frac{\partial {\mathcal{L}}_{\mathrm{{CE}}}}{\partial {\mathbf{P}}_{i\bar{j}}^{ - }} =  - \frac{f\left( {Q,{D}^{ + }}\right) }{\partial {\mathbf{P}}_{i\bar{j}}^{ - }} + \frac{1}{\mathop{\sum }\limits_{{b = 1}}^{B}\exp f\left( {Q,{D}_{b}}\right) }\frac{\partial }{\partial {\mathbf{P}}_{i\bar{j}}^{ - }}\mathop{\sum }\limits_{{b = 1}}^{B}\exp f\left( {Q,{D}_{b}}\right) 
$$

$$
 =  - \frac{1}{{Z}^{ - }}\left\lbrack  {-\frac{\exp f\left( {Q,{D}^{ - }}\right) }{\mathop{\sum }\limits_{{b = 1}}^{B}\exp f\left( {Q,{D}_{b}}\right) }}\right\rbrack  
$$

$$
 = \frac{1}{{Z}^{ - }}P\left( {{D}^{ - } \mid  Q,{D}_{1 : B}}\right) .
$$

Again,it is similar to the sum-of-max result,but it depends on ${Z}^{ - }$ . In this case,even when $P\left( {{D}^{ - } \mid  Q,{D}_{1 : B}}\right)$ is low,if there is a small number of retrieved tokens from ${D}^{ - }$ (i.e.,small ${Z}^{ - }$ ), ${\mathbf{P}}_{i\widehat{j}}^{ - }$ will be decreased significantly. Note that when ${Z}^{ - }$ is large, ${Z}^{ + }$ naturally becomes smaller as they compete for in-batch token retrieval, which causes positive tokens to have higher scores.

同样，它与最大和结果类似，但它取决于${Z}^{ - }$。在这种情况下，即使$P\left( {{D}^{ - } \mid  Q,{D}_{1 : B}}\right)$较低，如果从${D}^{ - }$中检索到的标记数量较少（即${Z}^{ - }$较小），${\mathbf{P}}_{i\widehat{j}}^{ - }$也会显著降低。请注意，当${Z}^{ - }$较大时，由于它们在批量内标记检索中相互竞争，${Z}^{ + }$自然会变小，这使得正标记具有更高的分数。

## B Inference Complexity

## B 推理复杂度

We compare the complexity of ColBERT and XTR during the scoring stage in terms of FLOPs. We do not measure the complexity for the online query encoding and maximum inner product search (MIPS), which have been extensively studied for both dual encoders and multi-vector retrieval [Santhanam et al., 2022a b, Guo et al., 2020].

我们从浮点运算次数（FLOPs）的角度比较了ColBERT和XTR在评分阶段的复杂度。我们不衡量在线查询编码和最大内积搜索（MIPS）的复杂度，因为双编码器和多向量检索在这方面都已经有了广泛的研究[Santhanam等人，2022a b；Guo等人，2020]。

For the scoring stage,both ColBERT and XTR have $\mathcal{O}\left( {n{k}^{\prime }}\right)$ candidate documents. Here,we assume the worst case $n{k}^{\prime }$ where each document token comes from a unique document. For each candidate document,ColBERT loads a set of document vectors of $\bar{m}d$ floating points $(\bar{m} =$ average document length) and computes eq. (1) with the query vectors of ${nd}$ floating points. Computing eq. (1) per candidate document requires ${2n}\bar{m}d$ FLOPs for token-level inner products, $n\bar{m}$ for finding the row-wise max,and $n$ for the final average. In total,ColBERT requires ${n}^{2}{k}^{\prime }\left( {2\bar{m}d + \bar{m} + 1}\right)$ FLOPs for the scoring stage. Note that this does not include the latency of loading the $\mathcal{O}\left( {n{k}^{\prime }\bar{m}d}\right)$ floating points onto the memory,which amounts up to ${450}\mathrm{{MB}}$ per query when $n = {16},{k}^{\prime } = {1000},\bar{m} = {55},d = {128}$ .

在评分阶段，ColBERT和XTR都有$\mathcal{O}\left( {n{k}^{\prime }}\right)$个候选文档。在这里，我们假设最坏的情况$n{k}^{\prime }$，即每个文档标记都来自一个唯一的文档。对于每个候选文档，ColBERT加载一组$\bar{m}d$个浮点（$(\bar{m} =$平均文档长度）的文档向量，并使用${nd}$个浮点的查询向量计算公式(1)。为每个候选文档计算公式(1)需要${2n}\bar{m}d$次浮点运算（FLOPs）用于标记级别的内积计算，$n\bar{m}$次用于查找逐行最大值，以及$n$次用于最终求平均值。总体而言，ColBERT在评分阶段需要${n}^{2}{k}^{\prime }\left( {2\bar{m}d + \bar{m} + 1}\right)$次浮点运算。请注意，这并不包括将$\mathcal{O}\left( {n{k}^{\prime }\bar{m}d}\right)$个浮点加载到内存中的延迟，当$n = {16},{k}^{\prime } = {1000},\bar{m} = {55},d = {128}$时，每次查询的延迟总计达${450}\mathrm{{MB}}$。

On the other hand, XTR first imputes the missing similarity, which is simply done by caching the ${k}^{\prime }$ -th token retrieval score for each query token. Then,each of $n{k}^{\prime }$ candidate documents requires $n\bar{r}$ FLOPs for finding row-wise max and $n$ for the average where $\bar{r}$ is the average number of retrieved tokens per each candidate document. In total,we have ${n}^{2}{k}^{\prime }\left( {\bar{r} + 1}\right)$ FLOPs. Table 1 shows the estimated FLOPs of the two models. XTR reduces the FLOPs at the scoring stage by ${4000} \times$ making multi-vector retrieval more efficient and practical.

另一方面，XTR（跨模态检索器，Cross - Modality Token Retriever）首先对缺失的相似度进行插补，这可以通过为每个查询词缓存第 ${k}^{\prime }$ 个词元的检索分数来简单实现。然后，$n{k}^{\prime }$ 个候选文档中的每一个都需要 $n\bar{r}$ 次浮点运算（FLOPs）来查找每行的最大值，以及 $n$ 次浮点运算来计算平均值，其中 $\bar{r}$ 是每个候选文档中检索到的词元的平均数量。总体而言，我们需要 ${n}^{2}{k}^{\prime }\left( {\bar{r} + 1}\right)$ 次浮点运算。表 1 展示了这两个模型估计的浮点运算次数。XTR 在评分阶段将浮点运算次数减少了 ${4000} \times$，从而使多向量检索更加高效和实用。

## C Implementation Details

## C 语言实现细节

XTR uses ${k}_{\text{train }}$ for retrieving in-batch document tokens. Since we retrieve over mini-batches,the size of mini-batch affects the performance for different ${k}_{\text{train }}$ ,which is shown in §5.3. In our experiments, we tried ${k}_{\text{train }} = \{ {32},{64},{128},{256},{320}\}$ for each batch size and choose the best model based on their performance on the MS MARCO development set. For inference,XTR uses ${k}^{\prime }$ for the token retrieval. We use ${k}^{\prime } = {40},{000}$ ,which is possible due to the efficient scoring stage of ${\mathrm{{XTR}}}^{6}$ We analyze the effect of using different ${k}^{\prime }$ ’s as well as its relationship to ${k}_{\text{train }}$ in §5.3. We initialize XTR from the base and xxl versions of the T5 encoder [Raffel et al. 2020] and provide XTR_base and ${\mathrm{{XTR}}}_{\mathrm{{xxl}}}$ . For multilingual XTR,we initialize XTR from mT5 [Xue et al. 2021]. We fine-tune XTR for 50,000 iterations with the learning rate to 1e-3. Up to 256 chips of TPU v3 accelerator were used depending on the size of the model. We use ScaNN [Guo et al. 2020] for the MIPS during the token retrieval stage. For BEIR, we use 13 datasets (AR: ArguAna. TO: Touché-2020. FE: Fever. CF: Climate-Fever. SF: Scifact. CV: TREC-COVID. NF: NFCorpus. NQ: Natural Questions. HQ: HotpotQA. FQ: FiQA-2018. SD: SCIDOCS. DB: DBPedia. QU: Quora).

XTR使用${k}_{\text{train }}$来检索批量文档中的词元。由于我们是在小批量数据上进行检索，因此小批量的大小会影响不同${k}_{\text{train }}$的性能，这在§5.3中有所展示。在我们的实验中，我们针对每个批量大小尝试了${k}_{\text{train }} = \{ {32},{64},{128},{256},{320}\}$，并根据它们在MS MARCO开发集上的性能选择最佳模型。在推理阶段，XTR使用${k}^{\prime }$进行词元检索。我们使用${k}^{\prime } = {40},{000}$，这得益于${\mathrm{{XTR}}}^{6}$高效的评分阶段。我们在§5.3中分析了使用不同${k}^{\prime }$的效果以及它与${k}_{\text{train }}$的关系。我们从T5编码器的基础版本和xxl版本[Raffel等人，2020]初始化XTR，并提供XTR_base和${\mathrm{{XTR}}}_{\mathrm{{xxl}}}$。对于多语言XTR，我们从mT5[Xue等人，2021]初始化XTR。我们以1e - 3的学习率对XTR进行50000次迭代的微调。根据模型的大小，最多使用256个TPU v3加速器芯片。在词元检索阶段，我们使用ScaNN[Guo等人，2020]进行最大内积搜索（MIPS）。对于BEIR，我们使用13个数据集（AR：ArguAna（论点分析）。TO：Touché - 2020。FE：Fever（事实验证）。CF：Climate - Fever（气候事实验证）。SF：Scifact（科学事实验证）。CV：TREC - COVID（新冠相关TREC任务）。NF：NFCorpus。NQ：Natural Questions（自然问题）。HQ：HotpotQA（火锅问答）。FQ：FiQA - 2018。SD：SCIDOCS（科学文档）。DB：DBPedia（维基数据）。QU：Quora（问答平台））。

Baselines There are two main paradigms on training retriever models for the out-of-domain evaluation. The first group trains a single retriever for each dataset (or domain) by generating queries for each out-of-domain corpus. Typically,this approach generates $N$ datasets to train $N$ independent models for $N$ different domains. For this one-retriever-per-domain approaches,we include GenQ [Thakur et al. 2021], GPL [Wang et al. 2022], and Promptagator [Dai et al., 2022]. The second group builds a single retriever-typically trained on a large-scale IR dataset such as MS MARCO-and directly applies it on the out-of-domain corpora and queries. For this one-retriever-for-all approaches,we present results of state-of-the-art retrievers including ${\mathrm{{Splade}}}_{\mathrm{v}2}$ [Formal et al., 2021], ColBERT_p2 [Santhanam et al., 2022b], and GTR_x1 [Ni et al., 2021]. We also show the results of T5-ColBERT ${}_{\mathrm{{xxl}}}$ [Qian et al. 2022],which is a T5-initialized ColBERT model and shares the same backbone LM and training dataset with XTR. Note that T5-ColBERT uses the heavy scoring stage based on the original sum-of-max. All of our one-retriever-for-all baselines, as well as XTR, are trained on English MS MARCO, unless otherwise stated.

基线 在为跨领域评估训练检索器模型方面，有两种主要范式。第一类方法是通过为每个跨领域语料库生成查询，为每个数据集（或领域）训练一个单独的检索器。通常，这种方法会生成$N$个数据集，以针对$N$个不同领域训练$N$个独立的模型。对于这种每个领域一个检索器的方法，我们纳入了GenQ（塔库尔等人，2021年）、GPL（王等人，2022年）和Promptagator（戴等人，2022年）。第二类方法是构建一个单一的检索器——通常在大规模信息检索（IR）数据集（如MS MARCO）上进行训练——并直接将其应用于跨领域语料库和查询。对于这种一检索器通用的方法，我们展示了包括${\mathrm{{Splade}}}_{\mathrm{v}2}$（福尔马尔等人，2021年）、ColBERT_p2（桑塔纳姆等人，2022b）和GTR_x1（倪等人，2021年）等最先进检索器的结果。我们还展示了T5 - ColBERT ${}_{\mathrm{{xxl}}}$（钱等人，2022年）的结果，它是一个由T5初始化的ColBERT模型，与XTR共享相同的主干语言模型（LM）和训练数据集。请注意，T5 - ColBERT使用基于原始最大和的繁重评分阶段。除非另有说明，我们所有的一检索器通用基线以及XTR均在英文MS MARCO数据集上进行训练。

---

<!-- Footnote -->

${}^{6}$ In fact,XTR with ${k}^{\prime } = {40},{000}$ has still two-to-three orders of magnitude cheaper scoring stage than ColBERT with ${k}^{\prime } = 1,{000}$ and T5-ColBERT with ${k}^{\prime } = 4,{000}$ .

${}^{6}$ 事实上，使用 ${k}^{\prime } = {40},{000}$ 的XTR（可扩展文本检索器，eXtensible Text Retriever）在评分阶段的成本仍比使用 ${k}^{\prime } = 1,{000}$ 的ColBERT（基于上下文的双向编码器表示，Contextualized Bidirectional Encoder Representations from Transformers）和使用 ${k}^{\prime } = 4,{000}$ 的T5 - ColBERT低两到三个数量级。

<!-- Footnote -->

---

## D Additional Results

## D 附加结果

In Table D.1, we show Recall @100 on BEIR.

在表D.1中，我们展示了BEIR（基准信息检索，Benchmarking Information Retrieval）上的前100召回率。

<!-- Media -->

<table><tr><td/><td>MS</td><td>AR</td><td>TO</td><td>FE</td><td>CF</td><td>SF</td><td>CV</td><td>NF</td><td>NQ</td><td>HQ</td><td>FQ</td><td>SD</td><td>DB</td><td>QU</td><td>Avg.</td></tr><tr><td colspan="16">One Retriever per Domain</td></tr><tr><td>GenQ</td><td>88.4</td><td>97.8</td><td>45.1</td><td>92.8</td><td>45.0</td><td>89.3</td><td>45.6</td><td>28.0</td><td>86.2</td><td>67.3</td><td>61.8</td><td>33.2</td><td>43.1</td><td>98.9</td><td>64.2</td></tr><tr><td>${\mathrm{{PTR}}}_{\text{retriever }}$</td><td>-</td><td>98.9</td><td>47.5</td><td>94.1</td><td>53.1</td><td>91.8</td><td>55.9</td><td>30.6</td><td>89.8</td><td>74.6</td><td>76.5</td><td>41.6</td><td>46.3</td><td>99.6</td><td>69.2</td></tr><tr><td colspan="16">One Retriever for All</td></tr><tr><td>BM25</td><td>65.8</td><td>94.2</td><td>53.8</td><td>93.1</td><td>43.6</td><td>90.8</td><td>49.8</td><td>25.0</td><td>76.0</td><td>74.0</td><td>53.9</td><td>35.6</td><td>39.8</td><td>97.3</td><td>63.6</td></tr><tr><td>ColBERT</td><td>86.5</td><td>91.4</td><td>43.9</td><td>93.4</td><td>44.4</td><td>87.8</td><td>46.4</td><td>25.4</td><td>91.2</td><td>74.8</td><td>60.3</td><td>34.4</td><td>46.1</td><td>98.9</td><td>64.5</td></tr><tr><td>${\mathrm{{GTR}}}_{\text{base }}$</td><td>89.8</td><td>97.4</td><td>44.3</td><td>92.3</td><td>52.2</td><td>87.2</td><td>41.1</td><td>27.5</td><td>89.3</td><td>67.6</td><td>67.0</td><td>34.0</td><td>41.8</td><td>99.6</td><td>64.7</td></tr><tr><td>T5-ColBERT ${}_{\text{base}}$</td><td>91.8</td><td>76.0</td><td>49.9</td><td>90.4</td><td>46.2</td><td>91.3</td><td>55.4</td><td>27.6</td><td>90.5</td><td>78.3</td><td>63.0</td><td>34.2</td><td>50.5</td><td>97.9</td><td>65.5</td></tr><tr><td>$\mathbf{{XT}{R}_{base}}$</td><td>91.0</td><td>92.1</td><td>50.8</td><td>92.5</td><td>51.6</td><td>90.5</td><td>57.3</td><td>28.0</td><td>91.6</td><td>80.7</td><td>63.5</td><td>34.8</td><td>52.0</td><td>98.9</td><td>68.0</td></tr><tr><td>${\mathrm{{GTR}}}_{\mathrm{{xxl}}}$</td><td>91.6</td><td>98.3</td><td>46.6</td><td>94.7</td><td>55.6</td><td>90.0</td><td>40.7</td><td>30.0</td><td>94.6</td><td>75.2</td><td>78.0</td><td>36.6</td><td>49.4</td><td>99.7</td><td>68.4</td></tr><tr><td>T5-ColBERTxxl</td><td>93.3</td><td>81.4</td><td>50.1</td><td>91.7</td><td>49.8</td><td>94.6</td><td>60.3</td><td>29.0</td><td>95.5</td><td>81.6</td><td>72.5</td><td>38.5</td><td>54.6</td><td>99.1</td><td>69.1</td></tr><tr><td>${\mathbf{{XTR}}}_{\mathbf{{xxl}}}$</td><td>93.0</td><td>95.6</td><td>52.7</td><td>93.7</td><td>56.2</td><td>95.0</td><td>62.1</td><td>30.7</td><td>95.8</td><td>82.2</td><td>73.0</td><td>39.4</td><td>54.5</td><td>99.3</td><td>71.6</td></tr></table>

<table><tbody><tr><td></td><td>女士（Ms）</td><td>增强现实（Augmented Reality，AR）</td><td>到；向（To）</td><td>铁（Fe，化学元素符号）</td><td>比较（Compare，CF）</td><td>科幻小说（Science Fiction，SF）</td><td>简历（CV）</td><td>网络功能（NF）</td><td>自然查询（NQ）</td><td>高画质（HQ）</td><td>模糊查询（FQ）</td><td>标准偏差（SD）</td><td>分贝（DB）</td><td>查询（QU）</td><td>平均值（Avg.）</td></tr><tr><td colspan="16">每个领域一个检索器</td></tr><tr><td>通用问题（GenQ）</td><td>88.4</td><td>97.8</td><td>45.1</td><td>92.8</td><td>45.0</td><td>89.3</td><td>45.6</td><td>28.0</td><td>86.2</td><td>67.3</td><td>61.8</td><td>33.2</td><td>43.1</td><td>98.9</td><td>64.2</td></tr><tr><td>${\mathrm{{PTR}}}_{\text{retriever }}$</td><td>-</td><td>98.9</td><td>47.5</td><td>94.1</td><td>53.1</td><td>91.8</td><td>55.9</td><td>30.6</td><td>89.8</td><td>74.6</td><td>76.5</td><td>41.6</td><td>46.3</td><td>99.6</td><td>69.2</td></tr><tr><td colspan="16">一个检索器用于所有领域</td></tr><tr><td>BM25（原文未变，为专业术语）</td><td>65.8</td><td>94.2</td><td>53.8</td><td>93.1</td><td>43.6</td><td>90.8</td><td>49.8</td><td>25.0</td><td>76.0</td><td>74.0</td><td>53.9</td><td>35.6</td><td>39.8</td><td>97.3</td><td>63.6</td></tr><tr><td>科尔伯特（ColBERT）</td><td>86.5</td><td>91.4</td><td>43.9</td><td>93.4</td><td>44.4</td><td>87.8</td><td>46.4</td><td>25.4</td><td>91.2</td><td>74.8</td><td>60.3</td><td>34.4</td><td>46.1</td><td>98.9</td><td>64.5</td></tr><tr><td>${\mathrm{{GTR}}}_{\text{base }}$</td><td>89.8</td><td>97.4</td><td>44.3</td><td>92.3</td><td>52.2</td><td>87.2</td><td>41.1</td><td>27.5</td><td>89.3</td><td>67.6</td><td>67.0</td><td>34.0</td><td>41.8</td><td>99.6</td><td>64.7</td></tr><tr><td>T5 - 科尔伯特 ${}_{\text{base}}$（T5 - ColBERT ${}_{\text{base}}$）</td><td>91.8</td><td>76.0</td><td>49.9</td><td>90.4</td><td>46.2</td><td>91.3</td><td>55.4</td><td>27.6</td><td>90.5</td><td>78.3</td><td>63.0</td><td>34.2</td><td>50.5</td><td>97.9</td><td>65.5</td></tr><tr><td>$\mathbf{{XT}{R}_{base}}$</td><td>91.0</td><td>92.1</td><td>50.8</td><td>92.5</td><td>51.6</td><td>90.5</td><td>57.3</td><td>28.0</td><td>91.6</td><td>80.7</td><td>63.5</td><td>34.8</td><td>52.0</td><td>98.9</td><td>68.0</td></tr><tr><td>${\mathrm{{GTR}}}_{\mathrm{{xxl}}}$</td><td>91.6</td><td>98.3</td><td>46.6</td><td>94.7</td><td>55.6</td><td>90.0</td><td>40.7</td><td>30.0</td><td>94.6</td><td>75.2</td><td>78.0</td><td>36.6</td><td>49.4</td><td>99.7</td><td>68.4</td></tr><tr><td>T5 - 科尔伯特超大版本（T5 - ColBERTxxl）</td><td>93.3</td><td>81.4</td><td>50.1</td><td>91.7</td><td>49.8</td><td>94.6</td><td>60.3</td><td>29.0</td><td>95.5</td><td>81.6</td><td>72.5</td><td>38.5</td><td>54.6</td><td>99.1</td><td>69.1</td></tr><tr><td>${\mathbf{{XTR}}}_{\mathbf{{xxl}}}$</td><td>93.0</td><td>95.6</td><td>52.7</td><td>93.7</td><td>56.2</td><td>95.0</td><td>62.1</td><td>30.7</td><td>95.8</td><td>82.2</td><td>73.0</td><td>39.4</td><td>54.5</td><td>99.3</td><td>71.6</td></tr></tbody></table>

Table D.1: Recall @100 on MS-MARCO and BEIR. The last column shows the average over 13 BEIR benchmarks. Compared to GTR, T5-ColBERT only marginally improves the recall. On the other hand, XTR greatly improves the recall showing the importance of having a better token retrieval.

表D.1：在MS-MARCO和BEIR数据集上的前100召回率（Recall @100）。最后一列显示了13个BEIR基准测试的平均值。与GTR相比，T5-ColBERT仅略微提高了召回率。另一方面，XTR显著提高了召回率，这表明了拥有更好的词元检索的重要性。

In Table D.2,we show nDCG@10 and Recall@100 on BEIR with different ${k}^{\prime }$ .

在表D.2中，我们展示了在BEIR数据集上使用不同的${k}^{\prime }$时的前10归一化折损累积增益（nDCG@10）和前100召回率（Recall@100）。

<table><tr><td>${k}^{\prime }$</td><td>MS</td><td>AR</td><td>TO</td><td>FE</td><td>CF</td><td>SF</td><td>CV</td><td>NF</td><td>NQ</td><td>HQ</td><td>FQ</td><td>SD</td><td>DB</td><td>QU</td><td>Avg.</td></tr><tr><td colspan="16">nDCG@10</td></tr><tr><td>40,000</td><td>45.0</td><td>40.7</td><td>31.3</td><td>73.7</td><td>20.7</td><td>71.0</td><td>73.6</td><td>34.0</td><td>53.0</td><td>64.7</td><td>34.7</td><td>14.5</td><td>40.9</td><td>86.1</td><td>49.1</td></tr><tr><td>1,000</td><td>43.2</td><td>44.6</td><td>29.0</td><td>72.1</td><td>20.4</td><td>71.7</td><td>67.5</td><td>34.2</td><td>49.8</td><td>61.3</td><td>33.0</td><td>15.9</td><td>37.0</td><td>86.3</td><td>47.9</td></tr><tr><td colspan="16">Recall@100</td></tr><tr><td>40,000</td><td>91.0</td><td>92.1</td><td>50.8</td><td>92.5</td><td>51.6</td><td>90.5</td><td>57.3</td><td>28.0</td><td>91.6</td><td>80.7</td><td>63.5</td><td>34.8</td><td>52.0</td><td>98.9</td><td>68.0</td></tr><tr><td>1,000</td><td>88.8</td><td>96.4</td><td>48.0</td><td>92.5</td><td>53.3</td><td>93.1</td><td>48.1</td><td>28.6</td><td>88.8</td><td>78.3</td><td>62.5</td><td>37.0</td><td>47.0</td><td>99.1</td><td>67.1</td></tr></table>

<table><tbody><tr><td>${k}^{\prime }$</td><td>女士（Ms）</td><td>增强现实（Augmented Reality，AR）</td><td>到；向（To）</td><td>铁（Fe，化学元素符号）</td><td>比较（Compare，CF可能是其缩写）</td><td>科幻小说（Science Fiction，SF）</td><td>简历（CV）</td><td>正常频率（NF）</td><td>噪声系数（NQ）</td><td>高画质（HQ）</td><td>模糊查询（FQ）</td><td>标准偏差（SD）</td><td>数据库（Database，缩写DB）</td><td>查询（Query，缩写QU）</td><td>平均值（Average，缩写Avg.）</td></tr><tr><td colspan="16">归一化折损累积增益@10（Normalized Discounted Cumulative Gain@10，缩写nDCG@10）</td></tr><tr><td>40,000</td><td>45.0</td><td>40.7</td><td>31.3</td><td>73.7</td><td>20.7</td><td>71.0</td><td>73.6</td><td>34.0</td><td>53.0</td><td>64.7</td><td>34.7</td><td>14.5</td><td>40.9</td><td>86.1</td><td>49.1</td></tr><tr><td>1,000</td><td>43.2</td><td>44.6</td><td>29.0</td><td>72.1</td><td>20.4</td><td>71.7</td><td>67.5</td><td>34.2</td><td>49.8</td><td>61.3</td><td>33.0</td><td>15.9</td><td>37.0</td><td>86.3</td><td>47.9</td></tr><tr><td colspan="16">召回率@100（Recall@100）</td></tr><tr><td>40,000</td><td>91.0</td><td>92.1</td><td>50.8</td><td>92.5</td><td>51.6</td><td>90.5</td><td>57.3</td><td>28.0</td><td>91.6</td><td>80.7</td><td>63.5</td><td>34.8</td><td>52.0</td><td>98.9</td><td>68.0</td></tr><tr><td>1,000</td><td>88.8</td><td>96.4</td><td>48.0</td><td>92.5</td><td>53.3</td><td>93.1</td><td>48.1</td><td>28.6</td><td>88.8</td><td>78.3</td><td>62.5</td><td>37.0</td><td>47.0</td><td>99.1</td><td>67.1</td></tr></tbody></table>

Table D.2: nDCG@10 and Recall@100 of ${\mathrm{{XTR}}}_{\text{base }}$ on MS-MARCO and BEIR with different ${k}^{\prime }$ . The last column shows the average over 13 BEIR benchmarks.

表D.2：${\mathrm{{XTR}}}_{\text{base }}$在MS - MARCO和BEIR数据集上，使用不同的${k}^{\prime }$时的nDCG@10和Recall@100指标。最后一列显示了在13个BEIR基准测试上的平均值。

<!-- Media -->

## E Qualitative Analysis

## E 定性分析

In Table 6-E.5, we show token retrieval results from T5-ColBERT and XTR.

在表6 - E.5中，我们展示了T5 - ColBERT和XTR的词元检索结果。

<!-- Media -->

T5-ColBERT token retrieval for "lauren london age?"

T5 - ColBERT对“劳伦·伦敦（Lauren London）的年龄是多少？”的词元检索

<table><tr><td colspan="4">tauren iondon age :</td></tr><tr><td>Rank</td><td>Token</td><td>Context of Token</td><td>Relevance</td></tr><tr><td>1</td><td>la</td><td>laura bush laura lane welch bush (born november 4, 1946) is the wife of the ${43}\mathrm{{rd}}$ president of the united states,george w. bush.</td><td>No</td></tr><tr><td>2</td><td>la</td><td>is laura branigan dead? laura branigan died on august 26, 2004 at the age of 47.</td><td>No</td></tr><tr><td>5</td><td>la</td><td>laika death in space. laika died within hours from overheating. her body temperature got way too hot for her to survive. the heat in her spacecraft had risen to 40 degrees celsius (104 degrees fahrenheit).</td><td>No</td></tr><tr><td>50</td><td>la</td><td>singer laura branigan dies at 47 singer laura branigan dies at 47. laura branigan, a grammy-nominated pop singer best known for her 1982 platinum hit gloria, has died.</td><td>No</td></tr><tr><td>100</td><td>la</td><td>lauren bacall lauren bacall ( born betty joan perske; september 16, 1924 august)</td><td>No</td></tr></table>

<table><tbody><tr><td colspan="4">牛头人伦敦时代 :</td></tr><tr><td>等级</td><td>Token</td><td>Token的上下文</td><td>相关性</td></tr><tr><td>1</td><td>啦</td><td>劳拉·布什 劳拉·莱恩·韦尔奇·布什（生于1946年11月4日）是美国${43}\mathrm{{rd}}$总统乔治·W·布什的妻子。</td><td>不</td></tr><tr><td>2</td><td>啦</td><td>劳拉·布兰妮根去世了吗？劳拉·布兰妮根于2004年8月26日去世，享年47岁。</td><td>不</td></tr><tr><td>5</td><td>啦</td><td>莱卡死于太空。莱卡因过热在数小时内死亡。她的体温过高，无法存活。她所在航天器内的温度升至40摄氏度（104华氏度）。</td><td>不</td></tr><tr><td>50</td><td>啦</td><td>歌手劳拉·布兰妮根47岁去世 歌手劳拉·布兰妮根47岁去世。劳拉·布兰妮根是一位获得格莱美提名的流行歌手，以1982年的白金单曲《格洛丽亚》而闻名，她已经离世。</td><td>不</td></tr><tr><td>100</td><td>啦</td><td>劳伦·白考尔 劳伦·白考尔（原名贝蒂·琼·珀斯克；1924年9月16日 - 8月）</td><td>不</td></tr></tbody></table>

XTR token retrieval for "lauren london age?"

针对“劳伦·伦敦（Lauren London）的年龄是多少？”的XTR令牌检索

<table><tr><td>Rank</td><td>Token</td><td>Context of Token</td><td>Relevance</td></tr><tr><td>1</td><td>la</td><td>lauren london birthday, age, family & biography 33 years, 1 month, 23 days old age lauren london will turn 34 on 05 december, 2018.</td><td>Yes</td></tr><tr><td>2</td><td>la</td><td>lauren london current age 33 years old. lauren london height 5 feet 7 inches $\left( {{1.5}\mathrm{\;m}/{157}\mathrm{\;{cm}}}\right)$ and her weight 119 lbs(54kg).</td><td>Yes</td></tr><tr><td>5</td><td>la</td><td>until now, lauren taylor's age is 28 year old and have gemini constellation. count down 363 days will come next birthday of lauren taylor!</td><td>No</td></tr><tr><td>50</td><td>la</td><td>if dwayne johnson, 43, and his longtime girlfriend, lauren hashian, 31, have a baby, would they have a pebble? the furious 7 star and his bae are reportedly expecting their first child together.</td><td>No</td></tr><tr><td>100</td><td>la</td><td>laura bush biography after his defeat, bush returned to is oil business and laura became a housewife, but soon returned to politics to help her father-in-law, george h.w. bush's presidential campaign in 1980.</td><td>No</td></tr></table>

<table><tbody><tr><td>排名</td><td>Token</td><td>Token的上下文</td><td>相关性</td></tr><tr><td>1</td><td>啦</td><td>劳伦·伦敦（Lauren London）的生日、年龄、家庭及个人简介 劳伦·伦敦现年33岁1个月零23天，她将于2018年12月5日年满34岁。</td><td>是的</td></tr><tr><td>2</td><td>啦</td><td>劳伦·伦敦（Lauren London）目前33岁。她身高5英尺7英寸（约$\left( {{1.5}\mathrm{\;m}/{157}\mathrm{\;{cm}}}\right)$），体重119磅（54公斤）。</td><td>是的</td></tr><tr><td>5</td><td>啦</td><td>截至目前，劳伦·泰勒（Lauren Taylor）28岁，星座是双子座。距离劳伦·泰勒下一个生日还有363天！</td><td>不</td></tr><tr><td>50</td><td>啦</td><td>如果43岁的道恩·强森（Dwayne Johnson）和他相恋多年、31岁的女友劳伦·哈希安（Lauren Hashian）有了孩子，他们会迎来一个小宝贝吗？据报道，这位《速度与激情7》的主演和他的女友即将迎来他们的第一个孩子。</td><td>不</td></tr><tr><td>100</td><td>啦</td><td>劳拉·布什（Laura Bush）的生平：在丈夫竞选失败后，布什回到了石油行业，劳拉则成为了一名家庭主妇，但不久后她又投身政治，在1980年帮助她的公公乔治·H·W·布什（George H.W. Bush）进行总统竞选活动。</td><td>不</td></tr></tbody></table>

<!-- Media -->

Table E.1: Token retrieval example from MS MARCO for the token "la" in the query "lauren london age". Among the top 100 retrieved tokens, 100% of T5-ColBERT tokens are lexically identical as the query token 1a and 100% of XTR tokens are also lexically identical. However, top retrieved results from XTR contain the correct entity (Lauren London) while those from T5-ColBERT are about wrong entities (Laura Bush, Laura Branigan, etc.).

表E.1：在查询“劳伦·伦敦（Lauren London）年龄”中针对Token“la”从MS MARCO进行Token检索的示例。在检索到的前100个Token中，T5 - ColBERT的Token有100%在词法上与查询Token“la”相同，XTR的Token也有100%在词法上相同。然而，XTR检索到的顶级结果包含正确的实体（劳伦·伦敦），而T5 - ColBERT检索到的结果则是关于错误的实体（劳拉·布什、劳拉·布兰尼根等）。

<!-- Media -->

T5-ColBERT token retrieval for "temple university student population?"

针对“天普大学（Temple University）学生人数是多少？”进行T5 - ColBERT的Token检索

<table><tr><td>Rank</td><td>Token</td><td>Context of Token</td><td>Relevance</td></tr><tr><td>1</td><td>temple</td><td>about temple university tuition, cost, financial aid, scholarships, and admission rates</td><td>No</td></tr><tr><td>2</td><td>temple</td><td>overview the application fee at temple university is \$55 . it is selective, with an acceptance rate of 61.7 percent and an early acceptance rate of 78 percent.</td><td>No</td></tr><tr><td>5</td><td>temple</td><td>the application fee at temple university is $\$ {55}$ . it is selective,with an acceptance rate of 61.7 percent and an early acceptance rate of 78 percent.</td><td>No</td></tr><tr><td>50</td><td>temple</td><td>temple university staff accountants earn \$52,000 annually, or \$25 per hour, which is 14% higher than the national average for all staff accoun- tants at $\$ {45},{000}$ annually and ${16}\%$ lower than the national salary average for all working americans</td><td>No</td></tr><tr><td>100</td><td>temple</td><td>browse expedia's selection and check out the best hotels close to temple university for the world-class spas and restaurants, or snatch up one of the cheap hotel deals near temple university</td><td>No</td></tr></table>

<table><tbody><tr><td>排名</td><td>Token</td><td>Token的上下文</td><td>相关性</td></tr><tr><td>1</td><td>寺庙</td><td>关于天普大学（Temple University）的学费、费用、经济援助、奖学金和录取率</td><td>不</td></tr><tr><td>2</td><td>寺庙</td><td>概述：天普大学（Temple University）的申请费是55美元。该校具有选择性，录取率为61.7%，提前录取率为78%。</td><td>不</td></tr><tr><td>5</td><td>寺庙</td><td>天普大学（Temple University）的申请费是$\$ {55}$。该校具有选择性，录取率为61.7%，提前录取率为78%。</td><td>不</td></tr><tr><td>50</td><td>寺庙</td><td>天普大学（Temple University）的会计人员年薪为52000美元，即每小时25美元，比全国所有会计人员的平均年薪$\$ {45},{000}$高出14%，比所有美国在职人员的全国平均工资低${16}\%$</td><td>不</td></tr><tr><td>100</td><td>寺庙</td><td>浏览亿客行（Expedia）的精选酒店，看看天普大学（Temple University）附近提供世界级水疗服务和餐厅的最佳酒店，或者抢购天普大学附近的廉价酒店套餐</td><td>不</td></tr></tbody></table>

XTR token retrieval for "temple university student population?"

针对“天普大学（Temple University）学生人数”的XTR令牌检索？

<table><tr><td>Rank</td><td>Token</td><td>Context of Token</td><td>Relevance</td></tr><tr><td>1</td><td>temple</td><td>by gender, the school has 18,009 male and 19,476 female students. by race/ethnicity, 20,664 white, 4,466 black, and 3,819 asian students are attending at temple university.</td><td>Yes</td></tr><tr><td>2</td><td>temple</td><td>below tables and charts represent the enrollment statistics including school degree, gender, race/ethnicity, and tranfer-in students at the school. at temple university, 37,485 students are enrolled ....</td><td>Yes</td></tr><tr><td>5</td><td>temple</td><td>temple university the big picture: how many students were on campus in fall 2015? of the 28,886 new freshman applicants, 56% were admitted and 31% of the admitted students enrolled at temple university in fall 2015.</td><td>Yes</td></tr><tr><td>50</td><td>temple</td><td>temple university was founded in 1884 by russell conwell, a yale- educated boston lawyer, orator, and ordained baptist minister</td><td>No</td></tr><tr><td>100</td><td>temple</td><td>kaiser said temple's endowment fund is low because the university is late to the idea of fundraising.</td><td>No</td></tr></table>

<table><tbody><tr><td>排名</td><td>Token</td><td>Token的上下文</td><td>相关性</td></tr><tr><td>1</td><td>寺庙</td><td>按性别划分，该校有18009名男学生和19476名女学生。按种族/族裔划分，天普大学（Temple University）有20664名白人学生、4466名黑人学生和3819名亚裔学生。</td><td>是的</td></tr><tr><td>2</td><td>寺庙</td><td>以下表格和图表展示了该校的招生统计数据，包括学位类型、性别、种族/族裔以及转校生情况。天普大学（Temple University）共有37485名学生就读……</td><td>是的</td></tr><tr><td>5</td><td>寺庙</td><td>天普大学（Temple University）概况：2015年秋季有多少学生在校？在28886名新生申请者中，56%被录取，而被录取的学生中有31%于2015年秋季入读天普大学。</td><td>是的</td></tr><tr><td>50</td><td>寺庙</td><td>天普大学（Temple University）由拉塞尔·康威尔（Russell Conwell）于1884年创办，他是一位毕业于耶鲁大学的波士顿律师、演说家，也是一位受戒的浸信会牧师。</td><td>不是</td></tr><tr><td>100</td><td>寺庙</td><td>凯泽表示，天普大学（Temple University）的捐赠基金较少，因为该校开展筹款活动起步较晚。</td><td>不是</td></tr></tbody></table>

<!-- Media -->

Table E.2: Token retrieval example from MS MARCO for the token "temple" in the query "temple university student population?". Among the top 100 retrieved tokens, 100% of T5-ColBERT tokens are lexically identical as the query token temple and 100% of XTR tokens are also lexically identical. However, top retrieved results from XTR are of the correct context (student population) while those from T5-ColBERT are off-topic (e.g., tuition, salary, etc.).

表E.2：在查询“天普大学（Temple University）学生人数？”中针对Token“temple”从MS MARCO进行的Token检索示例。在检索到的前100个Token中，T5 - ColBERT的Token有100%在词法上与查询Token“temple”相同，XTR的Token也有100%在词法上相同。然而，XTR检索出的前几个结果符合正确的上下文（学生人数），而T5 - ColBERT检索出的结果则偏离主题（例如，学费、薪资等）。

<!-- Media -->

T5-ColBERT token retrieval for "aire is expressed in some skin tumors"

针对“aire在一些皮肤肿瘤中表达”的T5 - ColBERT Token检索

<table><tr><td>Rank</td><td>Token</td><td>Context of Token</td><td>Relevance</td></tr><tr><td>1</td><td>aire</td><td>acids: structures, properties, and functions (university science books, sausalito, ca, 2000). humans expressing a defective form of the transcrip- tion factor aire (autoimmune regulator) develop multiorgan autoimmune disease.</td><td>No</td></tr><tr><td>2</td><td>aire</td><td>the primary biochemical defect in apeced is unknown. we have isolated a novel gene, aire, encoding for a putative nuclear protein featuring two phd-type zinc-finger motifs, suggesting its involvement in transcriptional regulation.</td><td>No</td></tr><tr><td>5</td><td>aire</td><td>control of central and peripheral tolerance by aire. the negative selection of self-reactive thymocytes depends on the expression of tissue-specific antigens by medullary thymic epithelial cells.</td><td>No</td></tr><tr><td>50</td><td>aire</td><td>we found that a human patient and mice with defects in aire develop similar lung pathology, demonstrating that the aire-deficient model of autoimmunity is a suitable translational system in which to unravel fundamental mechanisms of ild pathogenesis.</td><td>No</td></tr><tr><td>100</td><td>air</td><td>cool air initiates just downstream of the major sense transcript poly(a) site and terminates either early or extends into the flc promoter region.</td><td>No</td></tr></table>

<table><tbody><tr><td>排名</td><td>Token</td><td>Token的上下文</td><td>相关性</td></tr><tr><td>1</td><td>自身免疫调节因子（AIRE）</td><td>酸：结构、性质和功能（大学科学书籍，加利福尼亚州索萨利托，2000 年）。表达转录因子自身免疫调节因子（AIRE，autoimmune regulator）缺陷形式的人类会患上多器官自身免疫性疾病。</td><td>否</td></tr><tr><td>2</td><td>自身免疫调节因子（AIRE）</td><td>自身免疫性多内分泌腺病-念珠菌病-外胚层营养不良综合征（APECED）的主要生化缺陷尚不清楚。我们分离出了一个新基因，即自身免疫调节因子基因（AIRE），它编码一种假定的核蛋白，具有两个植物同源结构域（PHD）型锌指基序，表明其参与转录调控。</td><td>否</td></tr><tr><td>5</td><td>自身免疫调节因子（AIRE）</td><td>自身免疫调节因子（AIRE）对中枢和外周免疫耐受的调控。自身反应性胸腺细胞的阴性选择依赖于胸腺髓质上皮细胞对组织特异性抗原的表达。</td><td>否</td></tr><tr><td>50</td><td>自身免疫调节因子（AIRE）</td><td>我们发现，一名AIRE缺陷的人类患者和AIRE缺陷小鼠出现了相似的肺部病变，这表明AIRE缺陷的自身免疫模型是一个合适的转化研究体系，可用于揭示特发性间质性肺炎（ILD）发病的基本机制。</td><td>否</td></tr><tr><td>100</td><td>空气</td><td>冷诱导的非编码反义RNA（COOLAIR）起始于主要有义转录本多聚腺苷酸[poly(A)]位点的下游，要么提前终止，要么延伸至开花位点C（FLC）启动子区域。</td><td>否</td></tr></tbody></table>

XTR token retrieval for "aire is expressed in some skin tumors"

针对“aire基因在某些皮肤肿瘤中表达”的XTR令牌检索

<table><tr><td>Rank</td><td>Token</td><td>Context of Token</td><td>Relevance</td></tr><tr><td>1</td><td>aire</td><td>keratin-dependent regulation of aire and gene expression in skin tumor keratinocytes expression of the intermediate filament protein keratin 17 (k17) is robustly upregulated in inflammatory skin diseases and in many tumors....</td><td>Yes</td></tr><tr><td>2</td><td>aire</td><td>the thymic transcription factor autoimmune regulator (aire) prevents autoimmunity in part by promoting expression of tissue-specific self- antigens, which include many cancer antigens. for example, aire- deficient patients are predisposed to vitiligo, an autoimmune disease of melanocytes that is often triggered by efficacious immunotherapies against melanoma.</td><td>Yes</td></tr><tr><td>5</td><td>aire</td><td>aire regulates negative selection of organ-specific $t$ cells autoimmune polyendocrinopathy syndrome type 1 is a recessive mendelian disorder resulting from mutations in a novel gene, aire, and is characterized by a spectrum of organ-specific autoimmune diseases.</td><td>No</td></tr><tr><td>50</td><td>aire</td><td>here we demonstrate a novel role for a cd4+3- inducer cell population, previously linked to development of organized secondary lymphoid structures and maintenance of $\mathrm{t}$ cell memory in the functional regulation of aire-mediated promiscuous gene expression in the thymus.</td><td>No</td></tr><tr><td>100</td><td>air</td><td>this localization is dependent on the presence of sperm in the spermath- eca. after fertilization, air-2 remains associated with chromosomes during each meiotic division.</td><td>No</td></tr></table>

<table><tbody><tr><td>排名</td><td>Token</td><td>Token的上下文</td><td>相关性</td></tr><tr><td>1</td><td>自身免疫调节因子（AIRE）</td><td>皮肤肿瘤角质形成细胞中自身免疫调节因子（AIRE）的角蛋白依赖性调节及基因表达 中间丝蛋白角蛋白17（K17）的表达在炎症性皮肤病和许多肿瘤中显著上调...</td><td>是的</td></tr><tr><td>2</td><td>自身免疫调节因子（AIRE）</td><td>胸腺转录因子自身免疫调节因子（AIRE）部分通过促进组织特异性自身抗原的表达来预防自身免疫，这些自身抗原包括许多癌症抗原。例如，AIRE缺陷患者易患白癜风，这是一种黑素细胞的自身免疫性疾病，通常由针对黑色素瘤的有效免疫疗法引发。</td><td>是的</td></tr><tr><td>5</td><td>自身免疫调节因子（AIRE）</td><td>AIRE调节器官特异性$t$细胞的阴性选择。自身免疫性多内分泌腺病综合征1型是一种由新基因AIRE突变导致的隐性孟德尔疾病，其特征是一系列器官特异性自身免疫性疾病。</td><td>不</td></tr><tr><td>50</td><td>自身免疫调节因子（AIRE）</td><td>在这里，我们证明了CD4 + 3 - 诱导细胞群的新作用，该细胞群先前与有组织的二级淋巴结构的发育以及$\mathrm{t}$细胞记忆的维持有关，在胸腺中对AIRE介导的杂乱基因表达进行功能调节。</td><td>不</td></tr><tr><td>100</td><td>空气</td><td>这种定位依赖于受精囊（spermath-eca）中精子的存在。受精后，空气蛋白2（air-2）在每次减数分裂期间都与染色体保持关联。</td><td>不</td></tr></tbody></table>

<!-- Media -->

Table E.3: Token retrieval example from MS MARCO for the token "aire" in the query "aire is expressed in some skin tumors". Among the top 100 retrieved tokens, 77% of T5-ColBERT tokens are lexically identical as the query token aire and 77% of XTR tokens are also lexically identical. Top retrieved results from XTR are relevant to the query (about cancer, tumor, skin, and melanocyte), while those from T5-ColBERT are off-topic.

表E.3：从MS MARCO中检索查询“aire在某些皮肤肿瘤中表达”中Token（“aire”）的示例。在检索到的前100个Token中，77%的T5 - ColBERT Token在词法上与查询Token“aire”相同，77%的XTR Token也在词法上相同。XTR检索到的顶级结果与查询相关（关于癌症、肿瘤、皮肤和黑素细胞），而T5 - ColBERT检索到的结果则偏离主题。

<!-- Media -->

T5-ColBERT for "women with a higher birth weight are more likely to develop breast cancer later in life"

T5 - ColBERT用于查询“出生体重较高的女性在晚年更易患乳腺癌”

<table><tr><td>Rank</td><td>Token</td><td>Context of Token</td><td>Relevance</td></tr><tr><td>1</td><td>later</td><td>context exposure to cardiovascular risk factors during childhood and adolescence may be associated with the development of atheroscle- rosis later in life.</td><td>No</td></tr><tr><td>2</td><td>later</td><td>$\mathrm{n}$ despite the high incidence of febrile seizures,their contribution to the development of epilepsy later in life has remained controversial.</td><td>No</td></tr><tr><td>5</td><td>later</td><td>prospectively collected data from two intervention studies in adults with severe malaria were analysed focusing on laboratory features</td><td>No</td></tr><tr><td>50</td><td>later</td><td>on presentation and their association with a later requirement for rrt. they did have a limited amount of proteolytic activity and were able to kill s. aureus. with time, the nuclear envelope ruptured, and dna</td><td>No</td></tr><tr><td>100</td><td>late</td><td>filled the cytoplasm presumably for later lytic net production finally, we address the need for a careful consideration of potential benefits of bisphosphonate therapy and the risk for osteonecrosis of the jaw, a recently recognized late-toxicity of their use.</td><td>No</td></tr></table>

<table><tbody><tr><td>排名</td><td>Token</td><td>Token的上下文</td><td>相关性</td></tr><tr><td>1</td><td>以后</td><td>儿童和青少年时期暴露于心血管危险因素的背景情况可能与日后生活中动脉粥样硬化的发展有关。</td><td>否</td></tr><tr><td>2</td><td>以后</td><td>$\mathrm{n}$ 尽管高热惊厥的发生率很高，但它们对日后癫痫发展的影响仍存在争议。</td><td>否</td></tr><tr><td>5</td><td>以后</td><td>对两项成人重症疟疾干预研究中前瞻性收集的数据进行了分析，重点关注实验室特征</td><td>否</td></tr><tr><td>50</td><td>以后</td><td>在就诊时的表现以及它们与后期需要肾脏替代治疗（RRT）的关联。它们确实具有有限的蛋白水解活性，并且能够杀死金黄色葡萄球菌（S. aureus）。随着时间的推移，核膜破裂，DNA</td><td>否</td></tr><tr><td>100</td><td>晚期</td><td>充满了细胞质，推测是为了后期形成溶解性网状结构。最后，我们强调需要仔细权衡双膦酸盐治疗的潜在益处以及颌骨坏死的风险，颌骨坏死是使用双膦酸盐最近被认识到的一种晚期毒性反应。</td><td>否</td></tr></tbody></table>

XTR for "women with a higher birth weight are more likely to develop breast cancer later in life."

XTR代表“出生体重较高的女性在晚年更易患乳腺癌”。

<table><tr><td>Rank</td><td>Token</td><td>Context of Token</td><td>Relevance</td></tr><tr><td>1</td><td>later</td><td>life course breast cancer risk factors and adult breast density (united kingdom) objective to determine whether risk factors in childhood and early adulthood affect later mammographic breast density.</td><td>Yes</td></tr><tr><td>2</td><td>later</td><td>exposure to cardiovascular risk factors during childhood and ado- lescence may be associated with the development of atherosclerosis later in life.</td><td>No</td></tr><tr><td>5</td><td>subsequent</td><td>emerging evidence suggests an association between female prenatal experience and her subsequent risk of developing breast cancer.</td><td>Yes</td></tr><tr><td>50</td><td>later</td><td>our nested case-control study of eh progression included 138 cases, who were diagnosed with eh and then with carcinoma (1970-2003) at least 1 year (median, 6.5 years) later, and 241 controls....</td><td>No</td></tr><tr><td>100</td><td>during</td><td>obesity and being overweight during adulthood have been consis- tently linked to increased risk for development of dementia later in life, especially alzheimer's disease.</td><td>No</td></tr></table>

<table><tbody><tr><td>排名</td><td>Token</td><td>Token的上下文</td><td>相关性</td></tr><tr><td>1</td><td>后期</td><td>生命历程中的乳腺癌风险因素与成年乳腺密度（英国） 目的：确定儿童期和成年早期的风险因素是否会影响后期的乳腺钼靶密度。</td><td>是的</td></tr><tr><td>2</td><td>后期</td><td>儿童和青少年时期暴露于心血管危险因素可能与日后动脉粥样硬化（atherosclerosis）的发生有关。</td><td>否</td></tr><tr><td>5</td><td>随后的</td><td>新出现的证据表明，女性孕期经历与她随后患乳腺癌的风险之间存在关联。</td><td>是的</td></tr><tr><td>50</td><td>后期</td><td>我们对子宫内膜增生（eh）进展进行的巢式病例对照研究纳入了138例病例，这些患者被诊断为子宫内膜增生，随后至少1年后（中位时间为6.5年）被诊断为癌症（1970 - 2003年），以及241例对照……</td><td>否</td></tr><tr><td>100</td><td>在……期间</td><td>成年期的肥胖和超重一直与晚年患痴呆症（尤其是阿尔茨海默病）的风险增加有关。</td><td>否</td></tr></tbody></table>

<!-- Media -->

Table E.4: Token retrieval example from Scifact for the token "later" in the query "women with a higher birth weight are more likely to develop breast cancer later in life". Among the top 100 retrieved tokens, 72% of T5-ColBERT tokens are lexically identical as the query token later while only 33% of XTR tokens are lexically identical. Top retrieved results from XTR can retrieves synonyms (sebsequent) from relevant context, while those from T5-ColBERT are off-topic.

表E.4：从Scifact中检索查询“出生体重较高的女性在晚年更易患乳腺癌”中“later（以后）”这一Token（标记）的示例。在检索到的前100个Token中，T5 - ColBERT检索到的Token中有72%在词法上与查询Token“later”相同，而XTR检索到的Token中只有33%在词法上相同。XTR的顶级检索结果可以从相关上下文中检索到同义词（后续的），而T5 - ColBERT的检索结果则偏离主题。

<!-- Media -->

T5-ColBERT for "venules have a thinner or absent smooth layer compared to arterioles."

T5 - ColBERT用于处理“与小动脉相比，小静脉的平滑肌层较薄或缺失”。

<table><tr><td>Rank</td><td>Token</td><td>Context of Token</td><td>Relevance</td></tr><tr><td>1</td><td>thinner</td><td>platelet cd40l is associated with smaller plaques and thinner caps, while p-selectin is associated with smaller core size. conclusions: blood cell activation is significantly associated with atherosclerotic changes of the carotid wall.</td><td>No</td></tr><tr><td>2</td><td>thin</td><td>the periosteum is a thin, cellular and fibrous tissue that tightly ad- heres to the outer surface of all but the articulated surface of bone and appears to play a pivotal role in driving fracture pain.</td><td>No</td></tr><tr><td>5</td><td>thin</td><td>immunohistological scoring showed significantly (p<0.0001) higher median 5hmc levels in ben and dcn than in thin ssm, thick ssm, and cmd.</td><td>No</td></tr><tr><td>50</td><td>weak</td><td>subarachnoid haemorrhage (1-43 [1-25-1-63]), and stable angina (1-41 [1-36-1-46]), and weakest for abdominal aortic aneurysm (1-08 [1·00-1·17]).</td><td>No</td></tr><tr><td>100</td><td>slight</td><td>the ucp-2 gene expression was widely detected in the whole body with substantial levels in the wat and with slight levels in the skeletal muscle and bat.</td><td>No</td></tr></table>

<table><tbody><tr><td>排名</td><td>Token</td><td>Token的上下文</td><td>相关性</td></tr><tr><td>1</td><td>更薄的</td><td>血小板CD40L与较小的斑块和较薄的纤维帽相关，而P-选择素与较小的脂质核心大小相关。结论：血细胞活化与颈动脉壁的动脉粥样硬化改变显著相关。</td><td>否</td></tr><tr><td>2</td><td>薄的</td><td>骨膜是一种薄的、细胞性和纤维性组织，它紧密附着在除关节面以外的所有骨的外表面，并且似乎在引发骨折疼痛方面起着关键作用。</td><td>否</td></tr><tr><td>5</td><td>薄的</td><td>免疫组织化学评分显示，与薄型浅表扩散型黑色素瘤（thin ssm）、厚型浅表扩散型黑色素瘤（thick ssm）和结节性黑色素瘤（cmd）相比，雀斑样痣（ben）和皮肤纤维瘤（dcn）中的5-羟甲基胞嘧啶（5hmc）中位水平显著更高（p<0.0001）。</td><td>否</td></tr><tr><td>50</td><td>弱的</td><td>蛛网膜下腔出血（1 - 43 [1 - 25 - 1 - 63]）和稳定型心绞痛（1 - 41 [1 - 36 - 1 - 46]），而腹主动脉瘤的相关性最弱（1 - 08 [1·00 - 1·17]）。</td><td>否</td></tr><tr><td>100</td><td>轻微的</td><td>解偶联蛋白 - 2（UCP - 2）基因表达在全身广泛检测到，在白色脂肪组织（WAT）中表达水平较高，而在骨骼肌和棕色脂肪组织（BAT）中表达水平轻微。</td><td>否</td></tr></tbody></table>

XTR for "venules have a thinner or absent smooth layer compared to arterioles."

XTR表示“与小动脉相比，小静脉的平滑肌层较薄或缺失”。

<table><tr><td>Rank</td><td>Token</td><td>Context of Token</td><td>Relevance</td></tr><tr><td>1</td><td>thinner</td><td>platelet cd40l is associated with smaller plaques and thinner caps, while p-selectin is associated with smaller core size. conclusions: blood cell activation is significantly associated with atherosclerotic changes of the carotid wall.</td><td>No</td></tr><tr><td>2</td><td>thin</td><td>the periosteum is a thin, cellular and fibrous tissue that tightly ad- heres to the outer surface of all but the articulated surface of bone and appears to play a pivotal role in driving fracture pain.</td><td>No</td></tr><tr><td>5</td><td>thick</td><td>in dense fibrotic zones, thickening of the arterial and venous wall with severe luminal narrowing was present in each patient.</td><td>No</td></tr><tr><td>50</td><td>small</td><td>we assessed vasomotor function of the adipose microvasculature using videomicroscopy of small arterioles isolated from different fat compartments.</td><td>No</td></tr><tr><td>100</td><td>particle</td><td>context circulating concentration of lipoprotein(a) (lp[a]), a large glycoprotein attached to a low-density lipoprotein-like particle, may be associated with risk of coronary heart disease (chd) and stroke.</td><td>No</td></tr></table>

<table><tbody><tr><td>排名</td><td>Token</td><td>Token的上下文</td><td>相关性</td></tr><tr><td>1</td><td>更薄的</td><td>血小板CD40L与较小的斑块和较薄的纤维帽相关，而P-选择素与较小的脂质核心相关。结论：血细胞活化与颈动脉壁的动脉粥样硬化改变显著相关。</td><td>否</td></tr><tr><td>2</td><td>薄的</td><td>骨膜是一种薄的、细胞性和纤维性组织，它紧密附着在除关节面以外的所有骨的外表面，并且似乎在引发骨折疼痛方面起着关键作用。</td><td>否</td></tr><tr><td>5</td><td>厚的</td><td>在致密纤维化区域，每位患者均出现动脉和静脉壁增厚，并伴有严重的管腔狭窄。</td><td>否</td></tr><tr><td>50</td><td>小的</td><td>我们使用从不同脂肪隔室分离出的小动脉的视频显微镜技术评估了脂肪微血管的血管运动功能。</td><td>否</td></tr><tr><td>100</td><td>颗粒</td><td>背景：脂蛋白(a)（Lp[a]）是一种附着在低密度脂蛋白样颗粒上的大型糖蛋白，其循环浓度可能与冠心病（CHD）和中风风险相关。</td><td>否</td></tr></tbody></table>

<!-- Media -->

Table E.5: Token retrieval example from Scifact for the token "thinner" in the query "vanules have a thinner or absent smooth later compared to arterioles". Among the top 100 retrieved tokens, only $1\%$ of T5-ColBERT tokens are lexically identical as the query token thinner and only $1\%$ of XTR tokens are also lexically identical.

表E.5：从Scifact中检索查询“与小动脉相比，微静脉的平滑肌层更薄或缺失”中Token（标记）“thinner（更薄的）”的示例。在检索到的前100个Token中，T5 - ColBERT的Token中只有$1\%$个在词法上与查询Token“thinner”相同，XTR的Token中也只有$1\%$个在词法上相同。