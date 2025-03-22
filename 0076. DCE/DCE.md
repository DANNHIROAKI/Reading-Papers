# Learning Diverse Document Representations with Deep Query Interactions for Dense Retrieval

# 通过深度查询交互学习多样化文档表示以进行密集检索

Zehan Li*

李泽涵*

Beihang University

北京航空航天大学

lizehan@buaa.edu.cn

Nan Yang

南阳

Microsoft Research Asia

微软亚洲研究院

nanya@microsoft.com

Liang Wang

王亮

Microsoft Research Asia

微软亚洲研究院

wangliang@microsoft.com

Furu Wei

魏富如

Microsoft Research Asia

微软亚洲研究院

fuwei@microsoft.com

## Abstract

## 摘要

In this paper, we propose a new dense retrieval model which learns diverse document representations with deep query interactions. Our model encodes each document with a set of generated pseudo-queries to get query-informed, multi-view document representations. It not only enjoys high inference efficiency like the vanilla dual-encoder models, but also enables deep query-document interactions in document encoding and provides multi-faceted representations to better match different queries. Experiments on several benchmarks demonstrate the effectiveness of the proposed method, out-performing strong dual encoder baselines. ${}^{1}$

在本文中，我们提出了一种新的密集检索模型，该模型通过深度查询交互学习多样化的文档表示。我们的模型使用一组生成的伪查询对每个文档进行编码，以获得查询感知的多视图文档表示。它不仅像普通的双编码器模型一样具有较高的推理效率，还能在文档编码过程中实现深度的查询 - 文档交互，并提供多方面的表示以更好地匹配不同的查询。在多个基准测试上的实验证明了所提出方法的有效性，优于强大的双编码器基线模型。${}^{1}$

## 1 Introduction

## 1 引言

Document retrieval plays an important role in information retrieval (IR) tasks such as web search and open domain question answering (Chen et al., 2017). Early works such as BM25-based retriever (Robertson and Zaragoza, 2009) rely on lexical term matching to calculate the relevance of a pair of texts. Recently, neural network based dense retrieval (Karpukhin et al., 2020) has gained traction in research community. Dense retrieval learns a neural encoder to map queries and documents into a dense, low-dimensional vector space, and is less vulnerable to term mismatch problem compared to lexical match-based approaches.

文档检索在诸如网络搜索和开放域问答等信息检索（IR）任务中起着重要作用（Chen 等人，2017）。早期的工作，如基于 BM25 的检索器（Robertson 和 Zaragoza，2009），依赖于词法术语匹配来计算一对文本的相关性。最近，基于神经网络的密集检索（Karpukhin 等人，2020）在研究界受到了关注。密集检索学习一个神经编码器，将查询和文档映射到一个密集的低维向量空间中，与基于词法匹配的方法相比，它更不容易受到术语不匹配问题的影响。

There are two architectures to model the relevance between queries and documents. Dual encoder architecture encodes query and document separately into fixed-dimensional vectors (Karpukhin et al., 2020), where the similarity between query and document is usually instantiated as a dot product or cosine similarity between their vectors. As there are no interactions between query and document, dual encoder approach permits efficient inference with vector space search on precomputed document vectors. Cross encoder architecture feeds the concatenation of a query and document pair into one encoder to calculate its relevance score (Nogueira and Cho, 2019). Compared to dual encoder, cross encoder is more accurate due to the deep interaction between query and document, but comes with computation costs infeasible for first-stage retrieval. It is highly desirable to design a retrieval model which can match the performance of the cross encoder approach while maintaining the inference efficiency of the dual encoder approach.

有两种架构可用于建模查询和文档之间的相关性。双编码器架构分别将查询和文档编码为固定维度的向量（Karpukhin 等人，2020），其中查询和文档之间的相似度通常表示为它们向量之间的点积或余弦相似度。由于查询和文档之间没有交互，双编码器方法允许通过对预计算的文档向量进行向量空间搜索来进行高效推理。交叉编码器架构将查询和文档对的拼接输入到一个编码器中，以计算其相关性得分（Nogueira 和 Cho，2019）。与双编码器相比，交叉编码器由于查询和文档之间的深度交互而更准确，但计算成本过高，不适用于第一阶段的检索。因此，非常有必要设计一种检索模型，它既能达到交叉编码器方法的性能，又能保持双编码器方法的推理效率。

To this end, previous works mainly focus on two directions: late-interaction and distillation. The first solution is to design a hybrid architecture, where the early layers act as a dual encoder while the late layers work like a cross encoder (MacA-vaney et al., 2020; Khattab and Zaharia, 2020; Humeau et al., 2020). Its effectiveness comes with the cost of retrieval latency due to the computation involved with late layers. Another solution is knowledge distillation (Hinton et al., 2015), using the cross encoder to augment the training data (Qu et al., 2021; Ren et al., 2021), or distilling the ranking scores or attention matrices of a more powerful cross encoder reranker to a dual encoder retriever (Hofstätter et al., 2021; Ren et al., 2021; Lu et al., 2022).

为此，先前的工作主要集中在两个方向：后期交互和知识蒸馏。第一种解决方案是设计一种混合架构，其中早期层充当双编码器，而后期层则像交叉编码器一样工作（MacA - vaney 等人，2020；Khattab 和 Zaharia，2020；Humeau 等人，2020）。其有效性是以检索延迟为代价的，因为后期层涉及计算。另一种解决方案是知识蒸馏（Hinton 等人，2015），使用交叉编码器来扩充训练数据（Qu 等人，2021；Ren 等人，2021），或者将更强大的交叉编码器重排器的排名分数或注意力矩阵蒸馏到双编码器检索器中（Hofstätter 等人，2021；Ren 等人，2021；Lu 等人，2022）。

In this paper, we propose to achieve this goal by pre-computing the interaction-based representations. As depicted in Figure 1c, the document representations are obtained by feeding the concatenation of query and document through a cross encoder while the query representation is obtained in the same way as in the vanilla dual encoder. For every document, we use a query generation model to generate several queries which will each concatenate with the document to obtain a separate document representation.

在本文中，我们提议通过预计算基于交互的表示来实现这一目标。如图 1c 所示，文档表示是通过将查询和文档的拼接输入到交叉编码器中获得的，而查询表示的获取方式与普通双编码器相同。对于每个文档，我们使用一个查询生成模型生成几个查询，每个查询将与文档拼接以获得单独的文档表示。

---

<!-- Footnote -->

*Work done during internship at Microsoft Research Asia.

*在微软亚洲研究院实习期间完成的工作。

${}^{1}$ The code is available at https://github.com/ jordane95/dual-cross-encoder.

${}^{1}$ 代码可在https://github.com/ jordane95/dual-cross-encoder获取。

<!-- Footnote -->

---

Our model has the following advantages. Firstly, we can obtain document representation with deep query interactions without much additional inference cost. Additionally, we can naturally get multiview document representations (Luan et al., 2021; Tang et al., 2021; Zhang et al., 2022) by treating the query as explicit view extractor.

我们的模型具有以下优势。首先，我们可以在无需太多额外推理成本的情况下，通过深度查询交互获得文档表示。此外，通过将查询视为显式视图提取器，我们可以自然地获得多视图文档表示（Luan等人，2021年；Tang等人，2021年；Zhang等人，2022年）。

We follow the popular contrastive learning paradigm for learning such representations. Experiments on several retrieval benchmarks demonstrate the effectiveness of the proposed approach.

我们遵循流行的对比学习范式来学习此类表示。在多个检索基准上的实验证明了所提出方法的有效性。

To summarize, our main contributions are as follows:

综上所述，我们的主要贡献如下：

- We propose a new model architecture for dense retrieval, which can benefit from deep query-document interaction with low inference latency and learn multi-view document representations to better match different queries.

- 我们提出了一种用于密集检索的新模型架构，该架构可以从深度查询 - 文档交互中受益，同时具有较低的推理延迟，并学习多视图文档表示以更好地匹配不同的查询。

- We show the effectiveness of this model over various baselines by experiments on several large-scale retrieval benchmarks.

- 我们通过在几个大规模检索基准上的实验，证明了该模型相对于各种基线模型的有效性。

## 2 Related Work

## 2 相关工作

### 2.1 Dense Retrieval

### 2.1 密集检索

Dense passage retrieval (DPR) (Karpukhin et al., 2020) learns a two-tower BERT encoder to represent question and passage as vectors and takes their dot product as relevance score. The training of such dense retrievers can be optimized with more sophisticated negative sampling strategy (Xiong et al., 2021; Qu et al., 2021; Hofstätter et al., 2021; Zhan et al., 2021; Yang et al., 2021), or knowledge distillation from a more powerful cross-encoder teacher (Qu et al., 2021; Ren et al., 2021; Hofstät-ter et al., 2021; Lu et al., 2022).

密集段落检索（DPR）（Karpukhin等人，2020年）学习一个双塔BERT编码器，将问题和段落表示为向量，并将它们的点积作为相关性得分。此类密集检索器的训练可以通过更复杂的负采样策略（Xiong等人，2021年；Qu等人，2021年；Hofstätter等人，2021年；Zhan等人，2021年；Yang等人，2021年）进行优化，或者通过从更强大的交叉编码器教师进行知识蒸馏（Qu等人，2021年；Ren等人，2021年；Hofstät - ter等人，2021年；Lu等人，2022年）。

Recently, some work have been devoted to trading off the efficiency and effectiveness with a late-interaction architecture. Humeau et al. (2020) compress the query context into multiple dense vectors with a Poly-Encoder architecture. The relevance score is modeled by a attention-weighted sum of individual matching scores. Tang et al. (2021) further improve the multi-encoding scheme through $k$ -means clustering over all document tokens’ em-beddings. ColBERT (Khattab and Zaharia, 2020) learns word level representations for both query and document and calculates the relevance score with a MaxSim operation followed by a sum pooling aggregator. Although powerful, they cannot fully utilize the maximum inner product search (MIPS). In contrast, we employ a pre-interaction mechanism combined with a max pooler which is compatible with MIPS.

最近，一些工作致力于通过后期交互架构来平衡效率和有效性。Humeau等人（2020年）使用Poly - 编码器架构将查询上下文压缩为多个密集向量。相关性得分通过各个匹配得分的注意力加权和进行建模。Tang等人（2021年）通过对所有文档标记的嵌入进行$k$ - 均值聚类，进一步改进了多编码方案。ColBERT（Khattab和Zaharia，2020年）学习查询和文档的词级表示，并通过MaxSim操作和求和池化聚合器计算相关性得分。虽然这些方法很强大，但它们不能充分利用最大内积搜索（MIPS）。相比之下，我们采用了一种预交互机制，并结合了一个与MIPS兼容的最大池化器。

Multi-vector encoding is essential in these late-interaction models, but is also gradually borrowed to learn effective dense retrieval models. Luan et al. (2021) represent each document with its first $k$ token embeddings. To learn multi-view document representations, Zhang et al. (2022) substitute the [CLS] token with $k$ special [VIE] tokens as view extractors and propose a local contrastive loss with annealing temperature between different views. In comparison, our model learns diverse document representations through interactions with different queries.

多向量编码在这些后期交互模型中至关重要，但也逐渐被用于学习有效的密集检索模型。Luan等人（2021年）用文档的前$k$个标记嵌入来表示每个文档。为了学习多视图文档表示，Zhang等人（2022年）用$k$个特殊的[VIE]标记代替[CLS]标记作为视图提取器，并提出了一种在不同视图之间具有退火温度的局部对比损失。相比之下，我们的模型通过与不同查询的交互来学习多样化的文档表示。

### 2.2 Query Generation

### 2.2 查询生成

Query generation (QG) is originally introduced to the IR community as a document expansion technique (Nogueira et al., 2019). Nogueira and Lin (2019) show that appending the T5-generated queries to the document before building the inverted index can bring substantial improvements over BM25. More recently, Mallia et al. (2021) use generated queries as term expansion to learn better sparse representations for documents.

查询生成（QG）最初作为一种文档扩展技术被引入信息检索领域（Nogueira等人，2019年）。Nogueira和Lin（2019年）表明，在构建倒排索引之前，将T5生成的查询附加到文档中可以比BM25带来显著的改进。最近，Mallia等人（2021年）使用生成的查询作为术语扩展，以学习更好的文档稀疏表示。

In the context of dense retrieval, query generation is usually used for domain adaptation in data scarcity scenarios. For example, Ma et al. (2020) use QG model trained on general domain to generate synthetic queries on target domain for model training. To reduce noise in generated data, Wang et al. (2022) further introduce a cross encoder for pseudo labeling. Different from the previous work, we mainly use the generated queries to learn query-informed document representations.

在密集检索的背景下，查询生成通常用于数据稀缺场景中的领域适应。例如，Ma等人（2020年）使用在通用领域训练的QG模型在目标领域生成合成查询，用于模型训练。为了减少生成数据中的噪声，Wang等人（2022年）进一步引入了一个交叉编码器进行伪标记。与以往的工作不同，我们主要使用生成的查询来学习查询感知的文档表示。

## 3 Method

## 3 方法

### 3.1 Task Definition

### 3.1 任务定义

Given a query $q$ and a collection of $N$ documents $\mathcal{D} = \left\{  {{d}_{1},{d}_{2},\ldots ,{d}_{i},\ldots ,{d}_{N}}\right\}$ ,a retriever aims to find a set of $K$ relevant documents ${\mathcal{D}}_{ + } =$ $\left\{  {{d}_{{i}_{1}},{d}_{{i}_{2}},\ldots ,{d}_{{i}_{j}},\ldots ,{d}_{{i}_{K}}}\right\}  ,{}^{2}$ by ranking the document in the corpus $\mathcal{D}$ according to its relevance score with respect to the query $q$ ,for next stage re-ranking or downstream applications.

给定一个查询 $q$ 和一个包含 $N$ 篇文档的集合 $\mathcal{D} = \left\{  {{d}_{1},{d}_{2},\ldots ,{d}_{i},\ldots ,{d}_{N}}\right\}$，检索器的目标是通过根据文档与查询 $q$ 的相关性得分对语料库 $\mathcal{D}$ 中的文档进行排序，找出一组 $K$ 相关文档 ${\mathcal{D}}_{ + } =$ $\left\{  {{d}_{{i}_{1}},{d}_{{i}_{2}},\ldots ,{d}_{{i}_{j}},\ldots ,{d}_{{i}_{K}}}\right\}  ,{}^{2}$，以供下一阶段的重排序或下游应用使用。

---

<!-- Footnote -->

${}^{2}$ Usually $K \ll  N$ .

${}^{2}$ 通常 $K \ll  N$。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: score score score CLS CLS Query Document Encoder Encoder Query Query Document (c) Dual Cross Encoder CLS CLS MLP Query Document Cross Encoder Encoder Encoder Query Document Query Document (a) Dual Encoder (b) Cross Encoder -->

<img src="https://cdn.noedgeai.com/0195aecd-bcf8-79f9-9c72-7b83cb96be92_2.jpg?x=193&y=189&w=1250&h=420&r=0"/>

Figure 1: Illustration of different matching paradigms with different architectures.

图 1：不同架构的不同匹配范式示意图。

<!-- Media -->

### 3.2 Dual Encoder

### 3.2 双编码器

We first introduce the dual encoder (DE) architecture for dense retrieval. In this framework, a query encoder $D{E}_{q}$ and a document encoder $D{E}_{d}$ are used to encode the query and document into low-dimensional vectors, respectively. To measure their relevance, a lightweight dot product between the two vectors is usually adopted to enable fast search,

我们首先介绍用于密集检索的双编码器（DE）架构。在这个框架中，查询编码器 $D{E}_{q}$ 和文档编码器 $D{E}_{d}$ 分别用于将查询和文档编码为低维向量。为了衡量它们的相关性，通常采用两个向量之间的轻量级点积来实现快速搜索。

$$
s\left( {q,d}\right)  = D{E}_{q}\left( q\right)  \cdot  D{E}_{d}\left( d\right) . \tag{1}
$$

The common design choice for the encoders is using multi-layer Transformers (Vaswani et al., 2017) initialized from pre-trained language models (PLMs), such as BERT (Devlin et al., 2019). How to get the representation from BERT is also an interesting question but beyond the scope of this paper. For simplicity, we directly take the [CLS] vector at the final layer as the text representation. The two encoders can share or use separate parameters. We tie the encoder parameters in main experiments but also provide results of untied parameters in ablation study.

编码器的常见设计选择是使用从预训练语言模型（PLM）初始化的多层 Transformer（Vaswani 等人，2017），例如 BERT（Devlin 等人，2019）。如何从 BERT 中获取表示也是一个有趣的问题，但超出了本文的范围。为了简单起见，我们直接将最后一层的 [CLS] 向量作为文本表示。两个编码器可以共享参数或使用单独的参数。我们在主要实验中绑定编码器参数，但在消融研究中也提供未绑定参数的结果。

### 3.3 Cross Encoder

### 3.3 交叉编码器

The cross encoder (CE) takes the concatenation of query and document as input and uses deep neural network to model their deep interactions. Given a pair of query and document consisting of multiple tokens, we feed their concatenation through a cross encoder to get the interaction-aware representation,

交叉编码器（CE）将查询和文档的拼接作为输入，并使用深度神经网络对它们的深度交互进行建模。给定由多个标记组成的查询 - 文档对，我们将它们的拼接输入交叉编码器以获得感知交互的表示。

$$
\mathbf{r} = {CE}\left( {q + d}\right) . \tag{2}
$$

Then a multi-layer perceptron (MLP) is applied on top of the interaction-aware representation to predict the relevance score,

然后在感知交互的表示之上应用多层感知机（MLP）来预测相关性得分。

$$
s\left( {q,d}\right)  = {MLP}\left( \mathbf{r}\right) . \tag{3}
$$

The cross encoder is also usually instanced as a multi-layer Transformer network initialized from BERT. It can model term-level interactions between query and document, providing more fine-grained relevance estimation.

交叉编码器通常也实例化为一个从 BERT 初始化的多层 Transformer 网络。它可以对查询和文档之间的词级交互进行建模，提供更细粒度的相关性估计。

### 3.4 Dual Cross Encoder

### 3.4 双交叉编码器

We present our dual cross encoder where the document encoder acts as a cross encoder whereas the query encoder works like a dual encoder. Specifically, the query representation and document representation with query interaction are calculated as

我们提出了双交叉编码器，其中文档编码器充当交叉编码器，而查询编码器的工作方式类似于双编码器。具体来说，查询表示和带有查询交互的文档表示计算如下

$$
\mathbf{q} = D{E}_{q}\left( q\right) , \tag{4}
$$

$$
\mathbf{d} = C{E}_{d}\left( {q + d}\right) . \tag{5}
$$

Their similarity is measured by a dot product like in the vanilla dual encoder,

它们的相似度通过与普通双编码器中类似的点积来衡量。

$$
s\left( {q,d}\right)  = \mathbf{q} \cdot  \mathbf{d}. \tag{6}
$$

Query Generation. Note that the query from the query encoder side and document encoder side do not necessarily have to be the same since we only have access to the gold query for documents appearing in the training set. It is impractical to manually write potential queries for each document in the whole corpus. Hence, we use a T5 model (Raffel et al., 2020) fine-tuned on the doc-to-query task to generate queries for each document. We empirically adopt 10 queries decoded with top- $k$ sampling strategy (Fan et al., 2018) to encourage the query generation diversity.

查询生成。请注意，查询编码器端和文档编码器端的查询不一定必须相同，因为我们只能访问训练集中出现的文档的真实查询。为整个语料库中的每个文档手动编写潜在查询是不切实际的。因此，我们使用在文档到查询任务上微调的 T5 模型（Raffel 等人，2020）为每个文档生成查询。我们根据经验采用使用前 $k$ 采样策略（Fan 等人，2018）解码的 10 个查询，以促进查询生成的多样性。

The advantages of this architecture are twofold. On the one hand, we can model the query-document interaction in the pre-computed document representations. On the other hand, we can enjoy the retrieval efficiency of the vanilla dual encoder by pre-computing the interaction-aware document representations.

这种架构有两个优点。一方面，我们可以在预计算的文档表示中对查询 - 文档交互进行建模。另一方面，我们可以通过预计算感知交互的文档表示来享受普通双编码器的检索效率。

### 3.5 Training

### 3.5 训练

The conventional way to train a dense retriever requires a set of $\left( {q,{d}_{ + },{d}_{ - }}\right)$ pairs. The model is trained by optimizing the contrastive loss,

训练密集检索器的传统方法需要一组$\left( {q,{d}_{ + },{d}_{ - }}\right)$对。通过优化对比损失来训练模型，

$$
L\left( {q,{d}_{ + },{\mathcal{D}}_{ - }}\right)  =  - \log \frac{{e}^{s\left( {q,{d}_{ + }}\right) }}{\mathop{\sum }\limits_{{d \in  \left\{  {d}_{ + }\right\}   \cup  {\mathcal{D}}_{ - }}}{e}^{s\left( {q,d}\right) }}, \tag{7}
$$

where ${\mathcal{D}}_{ - }$ contains a set of negative documents ${d}_{ - }$ for query $q$ . Following Karpukhin et al. (2020), we include both BM25 hard negatives and in-batch negatives in ${\mathcal{D}}_{ - }$ .

其中${\mathcal{D}}_{ - }$包含查询$q$的一组负文档${d}_{ - }$。遵循卡尔普欣等人（2020年）的方法，我们在${\mathcal{D}}_{ - }$中同时纳入了BM25硬负样本和批次内负样本。

Constructing Positives and Negatives. Fusing query information into document representation requires redefining the positive and negative pairs. For a given query $q$ ,our framework potentially permits four types of positive and negatives, namely, $\left( {{q}_{ + } + {d}_{ + }}\right) ,\left( {{q}_{ + } + {d}_{ - }}\right) ,\left( {{q}_{ - } + {d}_{ + }}\right)$ and $\left( {{q}_{ - } + {d}_{ - }}\right)$ . To train our model, we convert the traditional positive and negative pair from the training set into that in our framework with the mapping function

构建正样本和负样本。将查询信息融合到文档表示中需要重新定义正样本对和负样本对。对于给定的查询$q$，我们的框架可能允许四种类型的正样本和负样本，即$\left( {{q}_{ + } + {d}_{ + }}\right) ,\left( {{q}_{ + } + {d}_{ - }}\right) ,\left( {{q}_{ - } + {d}_{ + }}\right)$和$\left( {{q}_{ - } + {d}_{ - }}\right)$。为了训练我们的模型，我们使用映射函数将训练集中的传统正样本对和负样本对转换为我们框架中的正样本对和负样本对

$$
f : \left( {q,{d}_{ + },{d}_{ - }}\right)  \mapsto  \left( {q,q + {d}_{ + },q + {d}_{ - }}\right) , \tag{8}
$$

where the + is the concatenation operation used in cross encoder. This mapping leads to the positive of type $\left( {{q}_{ + } + {d}_{ + }}\right)$ and the following two types of negatives. We leave the exploration of other types of negatives to future work.

其中“+”是交叉编码器中使用的拼接操作。这种映射产生了类型为$\left( {{q}_{ + } + {d}_{ + }}\right)$的正样本和以下两种类型的负样本。我们将探索其他类型负样本的工作留待未来进行。

Hard Negatives. The negative documents ${d}_{ - }$ are usually randomly sampled from BM25 top-ranked documents. After the mapping function defined above, these negatives fall in the type of negative $\left( {{q}_{ + } + {d}_{ - }}\right)$ ,which serve as hard negatives in our framework. This type of negatives can teach the model to learn more fine-grained information, as ${d}_{ - }$ is usually topically related to the gold query but cannot exactly answer the query. It also prevents our model from learning the shortcut, i.e., only learning matching signals from the query side and ignoring the document side information.

硬负样本。负文档${d}_{ - }$通常是从BM25排名靠前的文档中随机采样得到的。在经过上述定义的映射函数处理后，这些负样本属于负样本类型$\left( {{q}_{ + } + {d}_{ - }}\right)$，在我们的框架中作为硬负样本。这种类型的负样本可以促使模型学习更细粒度的信息，因为${d}_{ - }$通常在主题上与真实查询相关，但不能准确回答该查询。它还能防止我们的模型学习捷径，即只从查询端学习匹配信号而忽略文档端的信息。

In-Batch Negatives. To improve the training efficiency, we also adopt in-batch negatives to train our model. In our framework, the in-batch negatives belong to the negative type $\left( {{q}_{ - } + {d}_{ - }}\right)$ . This type of negatives is simple and can enable the model to learn topic-level discrimination ability.

批次内负样本。为了提高训练效率，我们还采用批次内负样本对模型进行训练。在我们的框架中，批次内负样本属于负样本类型$\left( {{q}_{ - } + {d}_{ - }}\right)$。这种类型的负样本比较简单，可以使模型学习主题级别的区分能力。

Data Augmentation. Regarding the generated queries as weakly annotated data, we can first pretrain our model on these noisy data as a warm-up stage and then fine-tune it on the human-annotated high-quality training set.

数据增强。将生成的查询视为弱标注数据，我们可以先在这些有噪声的数据上对模型进行预训练作为热身阶段，然后在人工标注的高质量训练集上进行微调。

### 3.6 Inference

### 3.6 推理

Index We encode the corpus following the same format as Equation 5, to get multi-view document representations with deep query interactions.

索引 我们按照与公式5相同的格式对语料库进行编码，以获得具有深度查询交互的多视图文档表示。

Denoting ${\mathbf{d}}_{j}^{i}$ as the $i$ -th view of the $j$ -th document ${d}_{j} \in  \mathcal{D}$ ,we have

将${\mathbf{d}}_{j}^{i}$表示为第$j$个文档${d}_{j} \in  \mathcal{D}$的第$i$个视图，我们有

$$
{q}_{j}^{i} \sim  {P}_{QG}\left( {q \mid  {d}_{j}}\right) , \tag{9}
$$

$$
{\mathbf{d}}_{j}^{i} = C{E}_{d}\left( {{q}_{j}^{i} + {d}_{j}}\right) , \tag{10}
$$

where ${P}_{QG}\left( {q \mid  d}\right)$ denotes the query generation model, $i \in  \{ 1,\ldots ,k\}$ and $j \in  \{ 1,\ldots ,N\}$ .

其中${P}_{QG}\left( {q \mid  d}\right)$表示查询生成模型，$i \in  \{ 1,\ldots ,k\}$和$j \in  \{ 1,\ldots ,N\}$。

Retrieval When a query comes, we encode it with the query encoder to get its contextualized representation $\mathbf{q}$ as in Equation 4. We adopted multi-vector encodings for a document ${d}_{j}$ ,the relevance score between the query $q$ and the document ${d}_{j}$ is taken as the max pooling of its different views’ scores,

检索 当一个查询到来时，我们使用查询编码器对其进行编码，以获得如公式4所示的上下文表示$\mathbf{q}$。我们对文档${d}_{j}$采用多向量编码，查询$q$与文档${d}_{j}$之间的相关性得分取其不同视图得分的最大池化值，

$$
s\left( {q,{d}_{j}}\right)  = \mathop{\max }\limits_{i}{\mathbf{q}}^{T}{\mathbf{d}}_{j}^{i}. \tag{11}
$$

This operation is compatible with MIPS for efficiency optimization, ${}^{3}$

此操作与最大内积搜索（MIPS）兼容，以进行效率优化，${}^{3}$

$$
p = \arg \mathop{\max }\limits_{d}{\mathbf{q}}^{T}\mathbf{d} \tag{12}
$$

## 4 Experiment

## 4 实验

In this section, we evaluate our model on different retrieval benchmarks and compare it with various baselines.

在本节中，我们在不同的检索基准上评估我们的模型，并将其与各种基线模型进行比较。

### 4.1 Datasets

### 4.1 数据集

We conduct experiments on the following retrieval benchmarks.

我们在以下检索基准上进行实验。

MS MARCO is a retrieval benchmark that originates from a machine reading comprehension dataset containing real user queries collected from Bing search and passages from web collection (Bajaj et al., 2016). We evaluate our model on the passage retrieval task. The corpus contains about ${8.8}\mathrm{M}$ passages. The training set consists of about ${500}\mathrm{k}$ annotated query-document pairs. The dev set has 6980 annotated queries. Since the test set is not publicly available, we evaluate on the dev set following previous work.

MS MARCO是一个检索基准，它源自一个机器阅读理解数据集，该数据集包含从必应搜索收集的真实用户查询以及来自网络集合的段落（巴贾杰等人，2016年）。我们在段落检索任务上评估我们的模型。语料库包含约${8.8}\mathrm{M}$个段落。训练集由约${500}\mathrm{k}$个带注释的查询 - 文档对组成。开发集有6980个带注释的查询。由于测试集不公开，我们按照以往的工作在开发集上进行评估。

---

<!-- Footnote -->

${}^{3}$ Note that to get top- $K$ documents,we first retrieve ${10K}$ documents to ensure that we have at least $K$ documents after pooling.

${}^{3}$ 请注意，为了获取前$K$个文档，我们首先检索${10K}$个文档，以确保在合并后至少有$K$个文档。

<!-- Footnote -->

---

TREC Deep Learning (DL) tracks provide test sets with more elaborate annotations to evaluate the real capacity of ranking models. We evaluate on the 2019 and 2020 test set (Craswell et al., 2020b,a). The 2019 test set contains 43 annotated queries and the 2020 test set contains 54 annotated queries. Both of them share the same corpus with the MS MARCO passage retrieval benchmark.

TREC深度学习（DL）赛道提供了注释更详尽的测试集，用于评估排序模型的实际能力。我们在2019年和2020年的测试集上进行评估（克拉斯韦尔等人，2020b，a）。2019年测试集包含43个带注释的查询，2020年测试集包含54个带注释的查询。它们都与MS MARCO段落检索基准共享相同的语料库。

### 4.2 Evaluation Metrics

### 4.2 评估指标

Following previous work, we mainly evaluate the retrieval performance on MS MARCO passage retrieval benchmark with MRR@10 but also report the score of Recall @1000. For datasets from TREC DL tracks, we evaluate with nDCG@10.

按照以往的工作，我们主要使用MRR@10评估在MS MARCO段落检索基准上的检索性能，但也报告Recall @1000的分数。对于来自TREC DL赛道的数据集，我们使用nDCG@10进行评估。

### 4.3 Baselines

### 4.3 基线模型

We mainly compare our model against the DPR (Karpukhin et al., 2020) baseline with a dual encoder architecture, but also report results of the following models most related to ours.

我们主要将我们的模型与具有双编码器架构的DPR（卡尔普欣等人，2020年）基线模型进行比较，但也报告以下与我们的模型最相关的模型的结果。

- BM25 (Robertson and Zaragoza, 2009) is the traditional lexical retriever.

- BM25（罗伯逊和萨拉戈萨，2009年）是传统的词法检索器。

- DocT5Query (Nogueira and Lin, 2019) appends generated queries to the document before building the inverted index of BM25.

- DocT5Query（诺盖拉和林，2019年）在构建BM25的倒排索引之前，将生成的查询附加到文档中。

- DeepImpact (Mallia et al., 2021) learns sparse representation for documents using generated queries as expanded terms.

- DeepImpact（马利亚等人，2021年）使用生成的查询作为扩展项来学习文档的稀疏表示。

- ANCE (Xiong et al., 2021) trains the DPR model with iterative hard negative mining strategy. We include this baseline since this technique is used in ME-BERT and DRPQ.

- ANCE（熊等人，2021年）使用迭代难负例挖掘策略训练DPR模型。我们纳入这个基线模型是因为ME - BERT和DRPQ中使用了这项技术。

- ME-BERT (Luan et al., 2021) utilizes the first $k$ token embeddings as multi-vector encodings for documents and adopts max pooling for score aggregation.

- ME - BERT（栾等人，2021年）将前$k$个标记嵌入用作文档的多向量编码，并采用最大池化进行分数聚合。

- DRPQ (Tang et al., 2021) improves over ME-BERT by performing a $k$ -means over all tokens' embeddings and utilizing a attention-based score aggregator.

- DRPQ（唐等人，2021年）通过对所有标记的嵌入执行$k$ - 均值聚类，并使用基于注意力的分数聚合器，对ME - BERT进行了改进。

- ColBERT (Khattab and Zaharia, 2020) represents query and document at token-level and uses a MaxSim pooler followed by a sum aggregator to calculate the relevance score.

- ColBERT（哈塔卜和扎哈里亚，2020年）在标记级别表示查询和文档，并使用MaxSim池化器，然后使用求和聚合器来计算相关性分数。

### 4.4 Implementation

### 4.4 实现

We implement our model based on the tevatron toolkit (Gao et al., 2022). For a fair comparison with our model, we re-implement the DPR baseline using the same set of hyperparameters.

我们基于Tevatron工具包（高等人，2022年）实现了我们的模型。为了与我们的模型进行公平比较，我们使用相同的超参数集重新实现了DPR（密集段落检索器，Dense Passage Retriever）基线模型。

We train all the models on 8 NVIDIA Telsa V100 GPUs with 32GB memory. We initialize all the encoders with bert-base-uncased. The max sequence length is 16 for query and 128 for passage. The number of positive and negative passages follows a ratio of 1:7 for each sample. We set the batch size to 32 . We use both officially provided BM25 negatives and in-batch negatives to train the models. We use Adam optimizer with the learning rate of $5 \times  {10}^{-6}$ ,linear decay with ${10}\%$ warmup steps.

我们在8块显存为32GB的英伟达特斯拉V100 GPU上训练所有模型。我们使用bert-base-uncased初始化所有编码器。查询的最大序列长度为16，段落的最大序列长度为128。每个样本的正段落和负段落数量之比为1:7。我们将批量大小设置为32。我们使用官方提供的BM25负样本和批次内负样本来训练模型。我们使用Adam优化器，学习率为$5 \times  {10}^{-6}$，采用线性衰减，并设置${10}\%$个热身步骤。

In the preliminary study without data augmentation, we train both models for 10 epochs. To make full use of generated queries, we first pre-train the models for 10 epochs on the corpus with a batch size of 256 and only in-batch negatives, and then fine-tune the models for 20 epochs till convergence on the training set. We haven't tuned other hyper-parameters. The pre-training stage takes about 15 hours and the fine-tuning stage takes about 8 hours.

在没有数据增强的初步研究中，我们对两个模型都进行了10个轮次的训练。为了充分利用生成的查询，我们首先在语料库上以256的批量大小和仅使用批次内负样本对模型进行10个轮次的预训练，然后在训练集上对模型进行20个轮次的微调直至收敛。我们没有调整其他超参数。预训练阶段大约需要15小时，微调阶段大约需要8小时。

During inference, we use IndexFlat IP of the faiss library (Johnson et al., 2021) to index the corpus and perform an exact search.

在推理过程中，我们使用faiss库的IndexFlat IP（约翰逊等人，2021年）对语料库进行索引并进行精确搜索。

### 4.5 Results

### 4.5 结果

Table 1 illustrates the evaluation results of our model and the baselines.

表1展示了我们的模型和基线模型的评估结果。

We first compare our model against the DPR dual encoder baseline. We can observe substantial improvements in terms of MRR@10 and nDCG@10 across all these datasets, which demonstrate the effectiveness of our approach. The Recall@1k also exhibits a slight improvement.

我们首先将我们的模型与DPR双编码器基线模型进行比较。我们可以观察到，在所有这些数据集上，MRR@10（前10个结果的平均倒数排名，Mean Reciprocal Rank at 10）和nDCG@10（前10个结果的归一化折损累积增益，Normalized Discounted Cumulative Gain at 10）都有显著提升，这证明了我们方法的有效性。Recall@1k（前1000个结果的召回率）也有轻微提升。

Our approach is also competitive with other baselines. On MS MARCO, it surpasses other baselines and is comparable to ColBERT, while being more efficient. On TREC DL 19, the results are comparable to ME-BERT, which used a more powerful large-size model as backbone and the hard negative mining technique of ANCE. On TREC DL 20, our model even outperforms the ColBERT model.

我们的方法与其他基线模型相比也具有竞争力。在MS MARCO数据集上，它超越了其他基线模型，与ColBERT相当，同时效率更高。在TREC DL 19数据集上，结果与ME-BERT相当，ME-BERT使用了更强大的大尺寸模型作为骨干，并采用了ANCE的难负样本挖掘技术。在TREC DL 20数据集上，我们的模型甚至优于ColBERT模型。

### 4.6 Ablation Study

### 4.6 消融研究

We conduct ablation studies on our model design choice.

我们对模型的设计选择进行了消融研究。

<!-- Media -->

<table><tr><td rowspan="2">Model</td><td rowspan="2">PLM</td><td colspan="2">MS MARCO</td><td>TREC DL 19</td><td>TREC DL 20</td></tr><tr><td>MRR@10</td><td>Recall@1k</td><td>nDCG@10</td><td>nDCG@10</td></tr><tr><td colspan="6">Sparse</td></tr><tr><td>BM25</td><td>-</td><td>18.4</td><td>85.3</td><td>50.6</td><td>48.0</td></tr><tr><td>DocT5Query</td><td>-</td><td>27.7</td><td>94.7</td><td>64.8</td><td>61.9</td></tr><tr><td>DeepImpact</td><td>${\text{BERT}}_{\text{base}}$</td><td>32.6</td><td>94.8</td><td>69.5</td><td>65.1</td></tr><tr><td colspan="6">Dense</td></tr><tr><td>DPR</td><td>${\text{BERT}}_{\text{base}}$</td><td>31.4</td><td>95.3</td><td>59.0</td><td>62.1</td></tr><tr><td>ANCE</td><td>${\mathrm{{RoBERTa}}}_{\text{base }}$</td><td>33.0</td><td>95.9</td><td>64.8</td><td>-</td></tr><tr><td>ME-BERT</td><td>${\text{BERT}}_{\text{large}}$</td><td>33.4</td><td>-</td><td>68.7</td><td>-</td></tr><tr><td>DRPQ</td><td>${\text{BERT}}_{\text{base}}$</td><td>34.5</td><td>96.4</td><td>-</td><td>-</td></tr><tr><td>ColBERT</td><td>${\text{BERT}}_{\text{base}}$</td><td>36.0</td><td>96.8</td><td>69.4</td><td>67.6</td></tr><tr><td>Ours</td><td>${\text{BERT}}_{base}$</td><td>36.0</td><td>96.4</td><td>68.3</td><td>68.9</td></tr></table>

<table><tbody><tr><td rowspan="2">模型</td><td rowspan="2">预训练语言模型（PLM）</td><td colspan="2">微软机器阅读理解数据集（MS MARCO）</td><td>文本检索会议深度学习任务19（TREC DL 19）</td><td>文本检索会议深度学习任务20（TREC DL 20）</td></tr><tr><td>前10名平均倒数排名（MRR@10）</td><td>前1000名召回率（Recall@1k）</td><td>前10名归一化折损累积增益（nDCG@10）</td><td>前10名归一化折损累积增益（nDCG@10）</td></tr><tr><td colspan="6">稀疏</td></tr><tr><td>二元独立模型的改进算法（BM25）</td><td>-</td><td>18.4</td><td>85.3</td><td>50.6</td><td>48.0</td></tr><tr><td>文档T5查询模型（DocT5Query）</td><td>-</td><td>27.7</td><td>94.7</td><td>64.8</td><td>61.9</td></tr><tr><td>深度影响模型（DeepImpact）</td><td>${\text{BERT}}_{\text{base}}$</td><td>32.6</td><td>94.8</td><td>69.5</td><td>65.1</td></tr><tr><td colspan="6">密集</td></tr><tr><td>密集段落检索器（DPR）</td><td>${\text{BERT}}_{\text{base}}$</td><td>31.4</td><td>95.3</td><td>59.0</td><td>62.1</td></tr><tr><td>自适应神经上下文编码器（ANCE）</td><td>${\mathrm{{RoBERTa}}}_{\text{base }}$</td><td>33.0</td><td>95.9</td><td>64.8</td><td>-</td></tr><tr><td>多编码器BERT模型（ME - BERT）</td><td>${\text{BERT}}_{\text{large}}$</td><td>33.4</td><td>-</td><td>68.7</td><td>-</td></tr><tr><td>深度表示查询生成模型（DRPQ）</td><td>${\text{BERT}}_{\text{base}}$</td><td>34.5</td><td>96.4</td><td>-</td><td>-</td></tr><tr><td>列BERT模型（ColBERT）</td><td>${\text{BERT}}_{\text{base}}$</td><td>36.0</td><td>96.8</td><td>69.4</td><td>67.6</td></tr><tr><td>我们的（模型）</td><td>${\text{BERT}}_{base}$</td><td>36.0</td><td>96.4</td><td>68.3</td><td>68.9</td></tr></tbody></table>

Table 1: Evaluation results on MS MARCO passage retrieval benchmark and TREC DL track. DocT5Query and DeepImpact can be seen as the sparse counterparts of our model. Both ME-BERT and DRPQ learn multi-vector encodings for documents, and have used the hard negative mining technique proposed in ANCE. ColBERT learns term-level representations of both query and document for late interaction. Results not available are marked as '-'.

表1：在MS MARCO段落检索基准和TREC DL赛道上的评估结果。DocT5Query和DeepImpact可视为我们模型的稀疏对应模型。ME - BERT和DRPQ都为文档学习多向量编码，并采用了ANCE中提出的难负样本挖掘技术。ColBERT学习查询和文档的词项级表示以进行后期交互。无法获取的结果标记为“ - ”。

<!-- Media -->

#### 4.6.1 Effect of Data Augmentation

#### 4.6.1 数据增强的效果

We used the generated queries as data augmentation for pre-training. We ablate on the effect of pretraining in this section. The results of different training stages on MS MARCO dev set are shown in Table 2.

我们使用生成的查询作为预训练的数据增强手段。本节我们将分析预训练的效果。在MS MARCO开发集上不同训练阶段的结果如表2所示。

<!-- Media -->

<table><tr><td>MRR@10</td><td>Pretrain</td><td>Finetune</td><td>Full</td></tr><tr><td>DPR</td><td>25.6</td><td>31.4</td><td>34.2</td></tr><tr><td>Ours</td><td>26.1</td><td>33.2</td><td>36.0</td></tr></table>

<table><tbody><tr><td>前10项平均倒数排名（MRR@10）</td><td>预训练（Pretrain）</td><td>微调（Finetune）</td><td>全量（Full）</td></tr><tr><td>密集段落检索器（DPR）</td><td>25.6</td><td>31.4</td><td>34.2</td></tr><tr><td>我们的方法</td><td>26.1</td><td>33.2</td><td>36.0</td></tr></tbody></table>

Table 2: Ablation of different training stages on MS MARCO dev set. Pretrain: only use generated data for training; Finetune: only use data from training set for training; Full: Pretrain + Finetune.

表2：不同训练阶段在MS MARCO开发集上的消融实验。预训练：仅使用生成的数据进行训练；微调：仅使用训练集中的数据进行训练；全流程：预训练 + 微调。

<!-- Media -->

We can see that using generated data for pretraining gives a MRR@10 score comparable to DocT5Query but lower than directly fine-tuning using data from the training set. The top- $k$ sampling decoding strategy in query generation may introduce some noise, which explains why the pretraining underperforms directly fine-tuning with high-quality data. However, the pre-training stage is still beneficial for the fine-tuning stage.

我们可以看到，使用生成的数据进行预训练得到的MRR@10分数与DocT5Query相当，但低于直接使用训练集中的数据进行微调的分数。查询生成中的前$k$采样解码策略可能会引入一些噪声，这解释了为什么预训练的表现不如使用高质量数据进行直接微调。然而，预训练阶段对微调阶段仍然是有益的。

The results on TREC DL track are shown in Table 3. Our model still consistently outperforms the dual encoder baseline under different settings. The improvements are more significant on this benchmark since the annotation is more complete. Notably, our model without data augmentation is comparable to the DPR baseline with data augmentation on this benchmark.

TREC DL赛道上的结果如表3所示。在不同设置下，我们的模型仍然始终优于双编码器基线模型。由于标注更加完整，在这个基准测试上的改进更为显著。值得注意的是，在这个基准测试上，我们未进行数据增强的模型与进行了数据增强的DPR基线模型表现相当。

<!-- Media -->

<table><tr><td>nDCG@10</td><td>DL 19</td><td>DL 20</td></tr><tr><td colspan="3">w/o Data Augmentation</td></tr><tr><td>DPR</td><td>59.0</td><td>62.1</td></tr><tr><td>Ours</td><td>63.0</td><td>67.6</td></tr><tr><td colspan="3">w/ Data Augmentation</td></tr><tr><td>DPR</td><td>63.1</td><td>66.5</td></tr><tr><td>Ours</td><td>68.3</td><td>68.9</td></tr></table>

<table><tbody><tr><td>10位归一化折损累计增益(nDCG@10)</td><td>深度学习19(DL 19)</td><td>深度学习20(DL 20)</td></tr><tr><td colspan="3">无数据增强(w/o Data Augmentation)</td></tr><tr><td>密集段落检索器(DPR)</td><td>59.0</td><td>62.1</td></tr><tr><td>我们的方法</td><td>63.0</td><td>67.6</td></tr><tr><td colspan="3">有数据增强(w/ Data Augmentation)</td></tr><tr><td>密集段落检索器(DPR)</td><td>63.1</td><td>66.5</td></tr><tr><td>我们的方法</td><td>68.3</td><td>68.9</td></tr></tbody></table>

Table 3: Results on TREC DL track under different settings.

表3：不同设置下TREC DL赛道的结果。

<!-- Media -->

#### 4.6.2 Effect of Sharing Parameters

#### 4.6.2 参数共享的影响

Sharing the encoder parameters can reduce the number of model parameters to half. We tie our encoder parameters in main experiments but also provide ablation of untied encoder parameters in Table 4 to study its effect.

共享编码器参数可以将模型参数数量减少一半。我们在主要实验中绑定了编码器参数，但也在表4中提供了未绑定编码器参数的消融实验，以研究其影响。

<!-- Media -->

<table><tr><td>MRR@10</td><td>tie</td><td>untie</td></tr><tr><td>DPR</td><td>31.4</td><td>31.7</td></tr><tr><td>Ours</td><td>33.2</td><td>33.8</td></tr></table>

<table><tbody><tr><td>前10项平均倒数排名（MRR@10）</td><td>平局；系；打结</td><td>解开；松开</td></tr><tr><td>密集段落检索（DPR）</td><td>31.4</td><td>31.7</td></tr><tr><td>我们的方法</td><td>33.2</td><td>33.8</td></tr></tbody></table>

Table 4: Results of tie / untie encoder parameters on MS MARCO dev set.

表4：MS MARCO开发集上绑定/解绑编码器参数的结果。

<!-- Media -->

We can observe that using two sets of encoder parameters gives slightly better performance but not so significantly. Using separate encoders brings more improvements to our model, which is normal since the nature of two encoders in our model is more asymmetric than that in the vanilla dual encoder.

我们可以观察到，使用两组编码器参数会带来稍好的性能，但提升并不显著。使用单独的编码器为我们的模型带来了更多改进，这是正常的，因为我们模型中两个编码器的性质比普通双编码器中的编码器更不对称。

## 5 Analysis

## 5 分析

Our experimental results in the previous section demonstrate that it is indeed beneficial to incorporate query interactions into the document representations. The generated queries are crucial to the success of our model. In this section, we analyse the influence of query quality and query diversity to the retrieval performance.

上一节的实验结果表明，将查询交互融入文档表示确实是有益的。生成的查询对我们模型的成功至关重要。在本节中，我们分析查询质量和查询多样性对检索性能的影响。

### 5.1 On the Query Quality

### 5.1 查询质量

The number of queries is an important factor in our framework. Too few queries have low diversity while too many queries will sacrifice efficiency. Thus we provide an analysis here to study its effect. We evaluate the query generation performance on the dev set of MS MARCO and reveal its relation with the retrieval performance.

查询数量是我们框架中的一个重要因素。查询太少则多样性不足，而查询太多会牺牲效率。因此，我们在此进行分析以研究其影响。我们在MS MARCO开发集上评估查询生成性能，并揭示其与检索性能的关系。

To measure the generation quality, we calculate the maximum ROUGE-L score between generated queries and the gold query on the dev set. For retrieval performance, we report the MRR@10.Figure 2 illustrates the evolution of the two metrics with different number of queries. ${}^{4}$

为了衡量生成质量，我们计算开发集上生成的查询与黄金查询之间的最大ROUGE - L分数。对于检索性能，我们报告MRR@10。图2展示了这两个指标随查询数量变化的情况。${}^{4}$

<!-- Media -->

<!-- figureText: 35 80 70 60 50 Generation 40 6 Number of queries $k$ Retrieval MRR@10 30 25 20 2 4 -->

<img src="https://cdn.noedgeai.com/0195aecd-bcf8-79f9-9c72-7b83cb96be92_6.jpg?x=209&y=1172&w=573&h=424&r=0"/>

Figure 2: The evolution of ROUGE-L and MRR@10 on MS MARCO dev set when varying number of queries from 1 to 10 .

图2：在MS MARCO开发集上，当查询数量从1变化到10时，ROUGE - L和MRR@10的变化情况。

<!-- Media -->

We can see that as the number of queries grows, the retrieval performance becomes better because of the improved generation quality. The correlation between the two metrics is shown in Figure 3. The Pearson coefficient is 0.9958 , indicating a strong positive correlation. Keep increasing the number of queries will consistently improve the retrieval performance but more marginally.

我们可以看到，随着查询数量的增加，由于生成质量的提高，检索性能变得更好。这两个指标之间的相关性如图3所示。皮尔逊系数为0.9958，表明存在强正相关。持续增加查询数量将持续改善检索性能，但提升幅度会越来越小。

<!-- Media -->

<!-- figureText: 35 55 60 65 70 Rouge-Lmax $\times  \mathrm{R} = {0.9958}$ MRR@10 30 25 40 45 50 -->

<img src="https://cdn.noedgeai.com/0195aecd-bcf8-79f9-9c72-7b83cb96be92_6.jpg?x=889&y=193&w=522&h=421&r=0"/>

Figure 3: Correlation between generation and retrieval performance on MS MARCO dev set.

图3：MS MARCO开发集上生成性能与检索性能之间的相关性。

<!-- Media -->

### 5.2 On the Query Diversity

### 5.2 查询多样性

Intuitively, more diverse queries can potentially hit more types of queries. We used top- $k$ sampling strategy to encourage the query generation diversity. However, whether and to what extent the generated queries are diverse remains unclear. To this end, we adopt the self-BLEU (Zhu et al., 2018) to measure the query generation diversity for a document.

直观地说，更多样化的查询可能会命中更多类型的查询。我们使用top - $k$采样策略来促进查询生成的多样性。然而，生成的查询是否多样以及多样的程度仍不清楚。为此，我们采用自BLEU（Zhu等人，2018）来衡量文档的查询生成多样性。

We partition the documents of MS MARCO dev set to subsets of different query diversity according to their self-BLEU-4 score and measure the retrieval performance on these subsets. The statistics are shown in Figure 4. First, we observe that most of the documents have high query generation diversity thanks to the top- $k$ sampling strategy (see Figure 4a). Second, the retrieval performance drops with higher diversity (see Figure 4b). One possible reason is that the QG model will generate more diverse queries when it doesn't know the right one. As such, higher diversity indicates lower quality (see Figure 4c) and the retrieval performance drops with lower generation quality (see Figure 3). It would be desirable to design a diversity metric that takes into account the generation quality.

我们根据自BLEU - 4分数将MS MARCO开发集的文档划分为不同查询多样性的子集，并测量这些子集上的检索性能。统计结果如图4所示。首先，我们观察到，由于top - $k$采样策略，大多数文档具有较高的查询生成多样性（见图4a）。其次，检索性能随着多样性的增加而下降（见图4b）。一个可能的原因是，当QG模型不知道正确的查询时，它会生成更多样化的查询。因此，更高的多样性意味着更低的质量（见图4c），并且检索性能会随着生成质量的降低而下降（见图3）。设计一个考虑生成质量的多样性指标是很有必要的。

### 5.3 Case Study

### 5.3 案例研究

We conduct a case study on the dev set to intuitively compare our model and the dual encoder baseline, as well as to illustrate the QG performance.

我们在开发集上进行了一个案例研究，以直观地比较我们的模型和双编码器基线，并说明QG性能。

Table 5 shows an example drawn from the MS MARCO dev set. Our model can retrieve the correct passage by generating the right query. DPR retrieves a hard negative passage where the content is corresponding to the query keywords but can not correctly answer the query. By generating queries, our model can better distinguish the difference among document meanings.

表5展示了一个从MS MARCO开发集中选取的示例。我们的模型可以通过生成正确的查询来检索到正确的段落。DPR检索到一个难负样本段落，其中的内容与查询关键词对应，但不能正确回答查询。通过生成查询，我们的模型可以更好地区分文档含义之间的差异。

---

<!-- Footnote -->

${}^{4}$ Please refer to Table 6 in Appendix A for exact numbers.

${}^{4}$ 确切数字请参考附录A中的表6。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: 2000 Diversity Diversity (b) Retrieval v.s. Diversity. (c) Quality v.s. Diversity. 0.5 0.1 Diversity (a) Diversity distribution. -->

<img src="https://cdn.noedgeai.com/0195aecd-bcf8-79f9-9c72-7b83cb96be92_7.jpg?x=196&y=211&w=1215&h=293&r=0"/>

Figure 4: Statistics of query diversity on MS MARCO dev set. We divided the diversity into 5 levels based on an average division of the self-BLEU-4 score.

图4：MS MARCO开发集上查询多样性的统计数据。我们根据自BLEU - 4得分的平均划分将多样性分为5个等级。

<table><tr><td>Query</td><td>how old is canada</td></tr><tr><td>Ours Rank 1</td><td>Canada was finally established as a country in 1867. It is 148 years old as of July 1 2015. Canada has been a country for 147 years. The first attempt at colonization occurred in 1000 A.D. by the Norsemen. There was no further European exploration until 1497 A.D. when the Italian sailor John Cabot came along. It then started being inhabited by more Europeans.</td></tr><tr><td>Generated Queries</td><td>when was canada established when was canada discovered what year was canada founded how long has canada been a country how old is canada</td></tr><tr><td>DPR Rank 1</td><td>it depends where you live but in Canada you have to be at least 16 years old.</td></tr><tr><td>Generated Queries</td><td>what is the legal age to be in canada how old do you have to be to live in canada how old do you have to be to enter canada as a citizen at what age can i go to canada to study in canada what is the minimum age to join the military</td></tr></table>

<table><tbody><tr><td>查询</td><td>加拿大有多少年历史了</td></tr><tr><td>我们的排名第1</td><td>加拿大最终于1867年建国。截至2015年7月1日，它已有148年历史。加拿大作为一个国家已经存在了147年。公元1000年，北欧人首次尝试在该地殖民。直到公元1497年，意大利水手约翰·卡博特（John Cabot）到来，欧洲人才再次进行探索。此后，更多欧洲人开始在此定居。</td></tr><tr><td>生成的查询</td><td>加拿大何时建立 加拿大何时被发现 加拿大哪一年成立 加拿大建国多久了 加拿大有多少年历史了</td></tr><tr><td>深度检索排名第1</td><td>这取决于你居住在哪里，但在加拿大，你至少要年满16岁。</td></tr><tr><td>生成的查询</td><td>在加拿大合法的年龄是多少 居住在加拿大需要多大年龄 作为公民进入加拿大需要多大年龄 我多少岁可以去加拿大 在加拿大学习的最低年龄是多少 参军的最低年龄是多少</td></tr></tbody></table>

Table 5: Case Study on MS MARCO dev set.

表5：MS MARCO开发集的案例研究。

<!-- Media -->

## 6 Discussion

## 6 讨论

The ranking task is usually approached with a two-stage pipeline: retrieve-then-rerank. The two stages usually use different architectures due to the effectiveness and efficiency trade-off. Dual encoder is more efficient for retrieval, while cross encoder is more powerful for reranking. How to take advantage of each other's strengths for mutual improvements is a hot topic of research. We propose a new dual cross encoder architecture to benefit from both with a pre-interaction mechanism.

排序任务通常采用两阶段流水线方法：先检索后重排。由于需要在有效性和效率之间进行权衡，这两个阶段通常使用不同的架构。双塔编码器（Dual encoder）在检索方面效率更高，而交叉编码器（Cross encoder）在重排方面更强大。如何利用彼此的优势实现相互改进是一个热门的研究话题。我们提出了一种新的双塔交叉编码器架构，通过预交互机制同时受益于两者的优势。

One limitation of our framework is that there exists a discrepancy between training and inference. We used the gold query to train the model but do not have access to the gold query during inference. Generating more queries would bridge this gap, but at the cost of efficiency. We wish to close this gap with improved training strategy or improved query generation quality in the future.

我们的框架存在一个局限性，即训练和推理之间存在差异。我们使用真实查询（gold query）来训练模型，但在推理过程中无法获取真实查询。生成更多查询可以缩小这一差距，但会牺牲效率。我们希望未来通过改进训练策略或提高查询生成质量来缩小这一差距。

## 7 Conclusion

## 7 结论

We proposed a novel dense retrieval model to bridge the gap between dual encoder and cross encoder. In our framework, the document representations are obtained by pre-interacting with a set of generated pseudo-queries through a cross encoder. Our approach enables multi-view document representation with deep query interaction while maintaining the inference efficiency of the dual encoder approach. We demonstrated its effectiveness compared to dual encoder baseline via experiments on various retrieval benchmarks. In the future work, we would like to explore how to better incorporate generated queries for model training and how to improve the query generation quality for better retrieval performance.

我们提出了一种新颖的密集检索模型，以缩小双塔编码器和交叉编码器之间的差距。在我们的框架中，文档表示是通过交叉编码器与一组生成的伪查询进行预交互获得的。我们的方法能够在保持双塔编码器方法推理效率的同时，通过深度查询交互实现多视角文档表示。通过在各种检索基准上的实验，我们证明了该方法相对于双塔编码器基线的有效性。在未来的工作中，我们希望探索如何更好地将生成的查询纳入模型训练，以及如何提高查询生成质量以获得更好的检索性能。

## References

## 参考文献

Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, Mir Rosenberg, Xia Song, Alina Stoica, Saurabh Ti-wary, and Tong Wang. 2016. Ms marco: A human generated machine reading comprehension dataset.

帕亚尔·巴贾杰（Payal Bajaj）、丹尼尔·坎波斯（Daniel Campos）、尼克·克拉斯韦尔（Nick Craswell）、李·邓（Li Deng）、高剑锋（Jianfeng Gao）、刘晓东（Xiaodong Liu）、兰甘·马朱姆德（Rangan Majumder）、安德鲁·麦克纳马拉（Andrew McNamara）、巴斯卡尔·米特拉（Bhaskar Mitra）、特里·阮（Tri Nguyen）、米尔·罗森伯格（Mir Rosenberg）、宋霞（Xia Song）、阿琳娜·斯托伊卡（Alina Stoica）、索拉布·蒂瓦里（Saurabh Ti-wary）和王彤（Tong Wang）。2016年。MS MARCO：一个人工生成的机器阅读理解数据集。

Danqi Chen, Adam Fisch, Jason Weston, and Antoine Bordes. 2017. Reading Wikipedia to answer open-domain questions. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1870- 1879, Vancouver, Canada. Association for Computational Linguistics.

陈丹琦（Danqi Chen）、亚当·菲施（Adam Fisch）、杰森·韦斯顿（Jason Weston）和安托万·博尔德斯（Antoine Bordes）。2017年。阅读维基百科以回答开放领域问题。见《第55届计算语言学协会年会论文集（第1卷：长论文）》，第1870 - 1879页，加拿大温哥华。计算语言学协会。

Nick Craswell, Bhaskar Mitra, Emine Yilmaz, and Daniel Campos. 2020a. Overview of the TREC 2020 deep learning track. In Proceedings of the Twenty-Ninth Text REtrieval Conference, TREC 2020, Virtual Event [Gaithersburg, Maryland, USA], November 16-20, 2020, volume 1266 of NIST Special Publication. National Institute of Standards and Technology (NIST).

尼克·克拉斯韦尔（Nick Craswell）、巴斯卡尔·米特拉（Bhaskar Mitra）、埃米内·伊尔马兹（Emine Yilmaz）和丹尼尔·坎波斯（Daniel Campos）。2020a。2020年TREC深度学习赛道概述。见《第29届文本检索会议论文集》，TREC 2020，虚拟会议[美国马里兰州盖瑟斯堡]，2020年11月16 - 20日，美国国家标准与技术研究院（NIST）特别出版物第1266卷。美国国家标准与技术研究院（NIST）。

Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, and Ellen M. Voorhees. 2020b. Overview of the TREC 2019 deep learning track. CoRR, abs/2003.07820.

尼克·克拉斯韦尔（Nick Craswell）、巴斯卡尔·米特拉（Bhaskar Mitra）、埃米内·伊尔马兹（Emine Yilmaz）、丹尼尔·坎波斯（Daniel Campos）和埃伦·M·沃里斯（Ellen M. Voorhees）。2020b。2019年TREC深度学习赛道概述。预印本，arXiv:2003.07820。

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171-4186, Minneapolis, Minnesota. Association for Computational Linguistics.

雅各布·德夫林（Jacob Devlin）、张明伟（Ming-Wei Chang）、肯顿·李（Kenton Lee）和克里斯蒂娜·图托纳娃（Kristina Toutanova）。2019年。BERT：用于语言理解的深度双向变换器预训练。见《2019年北美计算语言学协会会议：人类语言技术论文集，第1卷（长论文和短论文）》，第4171 - 4186页，美国明尼苏达州明尼阿波利斯。计算语言学协会。

Angela Fan, Mike Lewis, and Yann Dauphin. 2018. Hierarchical neural story generation. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 889-898, Melbourne, Australia. Association for Computational Linguistics.

安吉拉·范（Angela Fan）、迈克·刘易斯（Mike Lewis）和扬·多芬（Yann Dauphin）。2018年。分层神经故事生成。见《第56届计算语言学协会年会论文集（第1卷：长论文）》，第889 - 898页，澳大利亚墨尔本。计算语言学协会。

Luyu Gao, Xueguang Ma, Jimmy J. Lin, and Jamie Callan. 2022. Tevatron: An efficient and flexible toolkit for dense retrieval. ArXiv, abs/2203.05765.

高璐宇（Luyu Gao）、马雪光（Xueguang Ma）、林吉米·J（Jimmy J. Lin）和杰米·卡兰（Jamie Callan）。2022年。Tevatron：一个高效灵活的密集检索工具包。预印本，arXiv:2203.05765。

Geoffrey E. Hinton, Oriol Vinyals, and Jeffrey Dean. 2015. Distilling the knowledge in a neural network. CoRR, abs/1503.02531.

杰弗里·E·辛顿（Geoffrey E. Hinton）、奥里奥尔·温亚尔斯（Oriol Vinyals）和杰弗里·迪恩（Jeffrey Dean）。2015年。提炼神经网络中的知识。预印本，arXiv:1503.02531。

Sebastian Hofstätter, Sheng-Chieh Lin, Jheng-Hong Yang, Jimmy Lin, and Allan Hanbury. 2021. Efficiently teaching an effective dense retriever with balanced topic aware sampling. In SIGIR '21: The 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, Virtual Event, Canada, July 11-15, 2021, pages 113-122. ACM.

塞巴斯蒂安·霍夫施泰特（Sebastian Hofstätter）、林盛杰（Sheng-Chieh Lin）、杨政宏（Jheng-Hong Yang）、林吉米（Jimmy Lin）和艾伦·汉伯里（Allan Hanbury）。2021年。通过平衡主题感知采样高效训练有效的密集检索器。见《SIGIR '21：第44届国际ACM SIGIR信息检索研究与发展会议》，虚拟会议，加拿大，2021年7月11 - 15日，第113 - 122页。美国计算机协会（ACM）。

Samuel Humeau, Kurt Shuster, Marie-Anne Lachaux, and Jason Weston. 2020. Poly-encoders: Architec-

塞缪尔·于莫（Samuel Humeau）、库尔特·舒斯特（Kurt Shuster）、玛丽 - 安妮·拉肖（Marie - Anne Lachaux）和杰森·韦斯顿（Jason Weston）。2020年。多编码器：用于快速准确多句子评分的架构和预训练策略。

tures and pre-training strategies for fast and accurate multi-sentence scoring. In International Conference on Learning Representations.

在国际学习表征会议上。

Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2021. Billion-scale similarity search with gpus. IEEE Transactions on Big Data, 7(3):535-547.

杰夫·约翰逊（Jeff Johnson）、马蒂伊斯·杜泽（Matthijs Douze）和埃尔韦·热古（Hervé Jégou）。2021年。基于GPU的十亿级相似度搜索。《电气与电子工程师协会大数据汇刊》（IEEE Transactions on Big Data），7(3):535 - 547。

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6769- 6781, Online. Association for Computational Linguistics.

弗拉基米尔·卡尔普欣（Vladimir Karpukhin）、巴拉斯·奥古兹（Barlas Oguz）、闵世文（Sewon Min）、帕特里克·刘易斯（Patrick Lewis）、莱德尔·吴（Ledell Wu）、谢尔盖·叶杜诺夫（Sergey Edunov）、陈丹琦（Danqi Chen）和易文涛（Wen - tau Yih）。2020年。用于开放域问答的密集段落检索。《2020年自然语言处理经验方法会议论文集》（Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)），第6769 - 6781页，线上会议。计算语言学协会。

Omar Khattab and Matei Zaharia. 2020. Colbert: Efficient and effective passage search via contextual-ized late interaction over BERT. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, SI-GIR 2020, Virtual Event, China, July 25-30, 2020, pages 39-48. ACM.

奥马尔·哈塔卜（Omar Khattab）和马特·扎哈里亚（Matei Zaharia）。2020年。科尔伯特（Colbert）：通过基于BERT的上下文延迟交互实现高效有效的段落搜索。《第43届美国计算机协会信息检索研究与发展国际会议论文集》（Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval），SIGIR 2020，虚拟会议，中国，2020年7月25 - 30日，第39 - 48页。美国计算机协会。

Yuxiang Lu, Yiding Liu, Jiaxiang Liu, Yunsheng Shi, Zhengjie Huang, Shikun Feng, Yu Sun, Hao Tian, Hua Wu, Shuaiqiang Wang, Dawei Yin, and Haifeng Wang. 2022. Ernie-search: Bridging cross-encoder with dual-encoder via self on-the-fly distillation for dense passage retrieval. CoRR, abs/2205.09153.

卢宇翔、刘一丁、刘佳祥、史云生、黄正杰、冯世坤、孙宇、田浩、吴华、王帅强、尹大为和王海峰。2022年。厄尼搜索（Ernie - search）：通过实时自蒸馏将交叉编码器与双编码器相结合用于密集段落检索。计算机研究存储库（CoRR），abs/2205.09153。

Yi Luan, Jacob Eisenstein, Kristina Toutanova, and Michael Collins. 2021. Sparse, dense, and attentional representations for text retrieval. Transactions of the Association for Computational Linguistics, 9:329-345.

栾义、雅各布·艾森斯坦（Jacob Eisenstein）、克里斯蒂娜·图托纳娃（Kristina Toutanova）和迈克尔·柯林斯（Michael Collins）。2021年。用于文本检索的稀疏、密集和注意力表示。《计算语言学协会汇刊》（Transactions of the Association for Computational Linguistics），9:329 - 345。

Ji Ma, Ivan Korotkov, Yinfei Yang, Keith Hall, and Ryan McDonald. 2020. Zero-shot neural passage retrieval via domain-targeted synthetic question generation. arXiv preprint arXiv:2004.14503.

马骥、伊万·科罗特科夫（Ivan Korotkov）、杨荫飞（Yinfei Yang）、基思·霍尔（Keith Hall）和瑞安·麦克唐纳（Ryan McDonald）。2020年。通过领域目标合成问题生成实现零样本神经段落检索。预印本arXiv:2004.14503。

Sean MacAvaney, Franco Maria Nardini, Raffaele Perego, Nicola Tonellotto, Nazli Goharian, and Ophir Frieder. 2020. Efficient document re-ranking for transformers by precomputing term representations. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, SIGIR 2020, Virtual Event, China, July 25-30, 2020, pages 49-58. ACM.

肖恩·麦卡瓦尼（Sean MacAvaney）、佛朗哥·玛丽亚·纳尔迪尼（Franco Maria Nardini）、拉斐尔·佩雷戈（Raffaele Perego）、尼古拉·托内洛托（Nicola Tonellotto）、纳兹利·戈哈瑞安（Nazli Goharian）和奥菲尔·弗里德（Ophir Frieder）。2020年。通过预计算词项表示实现变压器模型的高效文档重排序。《第43届美国计算机协会信息检索研究与发展国际会议论文集》（Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval），SIGIR 2020，虚拟会议，中国，2020年7月25 - 30日，第49 - 58页。美国计算机协会。

Antonio Mallia, Omar Khattab, Torsten Suel, and Nicola Tonellotto. 2021. Learning passage impacts for inverted indexes. In SIGIR '21: The 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, Virtual Event, Canada, July 11-15, 2021, pages 1723-1727. ACM.

安东尼奥·马利亚（Antonio Mallia）、奥马尔·哈塔卜（Omar Khattab）、托尔斯滕·苏尔（Torsten Suel）和尼古拉·托内洛托（Nicola Tonellotto）。2021年。学习倒排索引的段落影响。在SIGIR '21：第44届美国计算机协会信息检索研究与发展国际会议上，虚拟会议，加拿大，2021年7月11 - 15日，第1723 - 1727页。美国计算机协会。

Rodrigo Nogueira and Jimmy Lin. 2019. From doc2query to doctttttquery.

罗德里戈·诺盖拉（Rodrigo Nogueira）和吉米·林（Jimmy Lin）。2019年。从文档到查询（doc2query）到文档多多多查询（doctttttquery）。

Rodrigo Frassetto Nogueira and Kyunghyun Cho. 2019. Passage re-ranking with BERT. CoRR, abs/1901.04085.

罗德里戈·弗拉塞托·诺盖拉（Rodrigo Frassetto Nogueira）和赵京焕（Kyunghyun Cho）。2019年。使用BERT进行段落重排序。计算机研究存储库（CoRR），abs/1901.04085。

Rodrigo Frassetto Nogueira, Wei Yang, Jimmy Lin, and Kyunghyun Cho. 2019. Document expansion by query prediction. CoRR, abs/1904.08375.

罗德里戈·弗拉塞托·诺盖拉（Rodrigo Frassetto Nogueira）、杨威、吉米·林（Jimmy Lin）和赵京焕（Kyunghyun Cho）。2019年。通过查询预测进行文档扩展。计算机研究存储库（CoRR），abs/1904.08375。

Yingqi Qu, Yuchen Ding, Jing Liu, Kai Liu, Ruiyang Ren, Wayne Xin Zhao, Daxiang Dong, Hua Wu, and Haifeng Wang. 2021. RocketQA: An optimized training approach to dense passage retrieval for open-domain question answering. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 5835-5847, Online. Association for Computational Linguistics.

曲英琦、丁宇辰、刘静、刘凯、任瑞阳、赵鑫、董大祥、吴华和王海峰。2021年。火箭问答（RocketQA）：一种用于开放域问答的密集段落检索优化训练方法。《2021年计算语言学协会北美分会人类语言技术会议论文集》（Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies），第5835 - 5847页，线上会议。计算语言学协会。

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1-67.

科林·拉菲尔（Colin Raffel）、诺姆·沙泽尔（Noam Shazeer）、亚当·罗伯茨（Adam Roberts）、凯瑟琳·李（Katherine Lee）、沙兰·纳朗（Sharan Narang）、迈克尔·马泰纳（Michael Matena）、周燕琦、李伟和彼得·J·刘（Peter J. Liu）。2020年。用统一的文本到文本变压器探索迁移学习的极限。《机器学习研究杂志》（Journal of Machine Learning Research），21(140):1 - 67。

Ruiyang Ren, Yingqi Qu, Jing Liu, Wayne Xin Zhao, QiaoQiao She, Hua Wu, Haifeng Wang, and Ji-Rong Wen. 2021. RocketQAv2: A joint training method for dense passage retrieval and passage re-ranking. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 2825-2835, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

任瑞阳、曲英琦、刘静、赵鑫、佘巧巧、吴华、王海峰和文继荣。2021年。RocketQAv2：一种用于密集段落检索和段落重排序的联合训练方法。见《2021年自然语言处理经验方法会议论文集》，第2825 - 2835页，线上会议及多米尼加共和国蓬塔卡纳。计算语言学协会。

Stephen E. Robertson and Hugo Zaragoza. 2009. The probabilistic relevance framework: BM25 and beyond. Found. Trends Inf. Retr., 3(4):333-389.

斯蒂芬·E·罗伯逊和雨果·萨拉戈萨。2009年。概率相关性框架：BM25及超越。《信息检索趋势与基础》，3(4):333 - 389。

Hongyin Tang, Xingwu Sun, Beihong Jin, Jingang Wang, Fuzheng Zhang, and Wei Wu. 2021. Improving document representations by generating pseudo query embeddings for dense retrieval. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 5054-5064, Online. Association for Computational Linguistics.

唐红印、孙兴武、金蓓宏、王金刚、张福正和吴伟。2021年。通过生成伪查询嵌入改进密集检索的文档表示。见《计算语言学协会第59届年会和第11届自然语言处理国际联合会议论文集（第1卷：长论文）》，第5054 - 5064页，线上会议。计算语言学协会。

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.

阿什什·瓦斯瓦尼、诺姆·沙泽尔、尼基·帕尔马尔、雅各布·乌斯库雷特、利昂·琼斯、艾丹·N·戈麦斯、卢卡斯·凯泽和伊利亚·波洛苏金。2017年。注意力就是你所需要的一切。见《神经信息处理系统进展》，第30卷。柯伦联合公司。

Kexin Wang, Nandan Thakur, Nils Reimers, and Iryna Gurevych. 2022. GPL: Generative pseudo labeling for unsupervised domain adaptation of dense retrieval. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2345-2360, Seattle, United States. Association for Computational Linguistics.

王可欣、南丹·塔库尔、尼尔斯·赖默斯和伊琳娜·古列维奇。2022年。GPL：用于密集检索无监督领域适应的生成式伪标签。见《计算语言学协会北美分会2022年会议：人类语言技术》，第2345 - 2360页，美国西雅图。计算语言学协会。

Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N. Bennett, Junaid Ahmed, and Arnold Overwijk. 2021. Approximate nearest neighbor negative contrastive learning for dense text retrieval. In International Conference on Learning Representations.

熊磊、熊晨彦、李晔、邓国峰、刘佳琳、保罗·N·贝内特、朱奈德·艾哈迈德和阿诺德·奥弗维克。2021年。用于密集文本检索的近似最近邻负对比学习。见《学习表征国际会议》。

Nan Yang, Furu Wei, Binxing Jiao, Daxing Jiang, and Linjun Yang. 2021. xMoCo: Cross momentum contrastive learning for open-domain question answering. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 6120-6129, Online. Association for Computational Linguistics.

杨楠、魏富如、焦彬星、蒋大兴和杨林军。2021年。xMoCo：用于开放域问答的跨动量对比学习。见《计算语言学协会第59届年会和第11届自然语言处理国际联合会议论文集（第1卷：长论文）》，第6120 - 6129页，线上会议。计算语言学协会。

Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Jiafeng Guo, Min Zhang, and Shaoping Ma. 2021. Optimizing dense retrieval model training with hard negatives. In ${SI}$ - GIR '21: The 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, Virtual Event, Canada, July 11-15, 2021, pages 1503-1512. ACM.

詹景涛、毛佳鑫、刘奕群、郭佳峰、张敏和马少平。2021年。使用难负样本优化密集检索模型训练。见${SI}$ - GIR '21：第44届ACM SIGIR信息检索研究与发展国际会议，虚拟会议，加拿大，2021年7月11 - 15日，第1503 - 1512页。美国计算机协会。

Shunyu Zhang, Yaobo Liang, Ming Gong, Daxin Jiang, and Nan Duan. 2022. Multi-view document representation learning for open-domain dense retrieval. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5990-6000, Dublin, Ireland. Association for Computational Linguistics.

张顺宇、梁耀波、龚明、蒋大兴和段楠。2022年。用于开放域密集检索的多视图文档表示学习。见《计算语言学协会第60届年会论文集（第1卷：长论文）》，第5990 - 6000页，爱尔兰都柏林。计算语言学协会。

Yaoming Zhu, Sidi Lu, Lei Zheng, Jiaxian Guo, Weinan Zhang, Jun Wang, and Yong Yu. 2018. Texy-gen: A benchmarking platform for text generation models. In The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval, SIGIR 2018, Ann Arbor, MI, USA, July 08- 12, 2018, pages 1097-1100. ACM.

朱耀明、卢思迪、郑磊、郭佳贤、张韦南、王军和俞勇。2018年。Texy - gen：文本生成模型的基准测试平台。见第41届ACM SIGIR信息检索研究与发展国际会议，SIGIR 2018，美国密歇根州安娜堡，2018年7月8 - 12日，第1097 - 1100页。美国计算机协会。

## A Appendix

## A 附录

<!-- Media -->

<table><tr><td>$k$</td><td>ROUGE-L</td><td>MRR@10</td></tr><tr><td>1</td><td>42.49</td><td>27.74</td></tr><tr><td>2</td><td>50.93</td><td>30.09</td></tr><tr><td>3</td><td>55.67</td><td>31.15</td></tr><tr><td>4</td><td>58.45</td><td>31.66</td></tr><tr><td>5</td><td>60.63</td><td>31.92</td></tr><tr><td>6</td><td>62.28</td><td>32.38</td></tr><tr><td>7</td><td>63.57</td><td>32.67</td></tr><tr><td>8</td><td>64.62</td><td>32.88</td></tr><tr><td>9</td><td>65.46</td><td>32.96</td></tr><tr><td>10</td><td>66.22</td><td>33.23</td></tr></table>

<table><tbody><tr><td>$k$</td><td>长文本召回率（ROUGE-L）</td><td>前10平均倒数排名（MRR@10）</td></tr><tr><td>1</td><td>42.49</td><td>27.74</td></tr><tr><td>2</td><td>50.93</td><td>30.09</td></tr><tr><td>3</td><td>55.67</td><td>31.15</td></tr><tr><td>4</td><td>58.45</td><td>31.66</td></tr><tr><td>5</td><td>60.63</td><td>31.92</td></tr><tr><td>6</td><td>62.28</td><td>32.38</td></tr><tr><td>7</td><td>63.57</td><td>32.67</td></tr><tr><td>8</td><td>64.62</td><td>32.88</td></tr><tr><td>9</td><td>65.46</td><td>32.96</td></tr><tr><td>10</td><td>66.22</td><td>33.23</td></tr></tbody></table>

Table 6: Results of generation and retrieval performance on MS MARCO dev set when varying number of queries (correspond to Figure 2 and Figure 3).

表6：在MS MARCO开发集上，当查询数量变化时（对应图2和图3）的生成和检索性能结果。

<!-- Media -->