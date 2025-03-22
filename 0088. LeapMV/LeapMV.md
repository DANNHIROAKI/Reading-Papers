# Token Pruning Optimization for Efficient Multi-Vector Dense Retrieval

# 高效多向量密集检索的Token剪枝优化

Shanxiu ${\mathrm{{He}}}^{1\left\lbrack  {{0009} - {0008} - {8581} - {6733}}\right\rbrack  }$ ,Mutasem Al-Darabsah ${}^{2\left\lbrack  {{0009} - {0003} - {3986} - {6311}}\right\rbrack  }$ , Suraj Nair ${}^{2\left\lbrack  {{0000} - {0003} - {2283} - {7672}}\right\rbrack  }$ ,Jonathan May ${}^{2,3\left\lbrack  {{0000} - {0002} - {5284} - {477X}}\right\rbrack  }$ ,Tarun Agarwal ${}^{2\left\lbrack  {{0009} - {0004} - {5682} - {8234}}\right\rbrack  }$ ,Tao Yang ${}^{1\left\lbrack  {{0000} - {0003} - {1902} - {3387}}\right\rbrack  }$ ,and Choon Hui ${\operatorname{Teo}}^{2\left\lbrack  {{0000} - {0002} - {5724} - {9478}}\right\rbrack  }$ ${}^{1}$ University of California,Santa Barbara,USA \{shanxiuhe,tyang\}@cs.ucsb.edu ${}^{2}$ Amazon,Palo Alto,USA \{mutasema, srjnair, jnatmay, tagar, choonhui\}@amazon.com ${}^{3}$ University of Southern California,Los Angeles,USA

单秀 ${\mathrm{{He}}}^{1\left\lbrack  {{0009} - {0008} - {8581} - {6733}}\right\rbrack  }$ ，穆塔塞姆·阿尔 - 达拉布萨赫 ${}^{2\left\lbrack  {{0009} - {0003} - {3986} - {6311}}\right\rbrack  }$ ，苏拉杰·奈尔 ${}^{2\left\lbrack  {{0000} - {0003} - {2283} - {7672}}\right\rbrack  }$ ，乔纳森·梅 ${}^{2,3\left\lbrack  {{0000} - {0002} - {5284} - {477X}}\right\rbrack  }$ ，塔伦·阿加瓦尔 ${}^{2\left\lbrack  {{0009} - {0004} - {5682} - {8234}}\right\rbrack  }$ ，陶阳 ${}^{1\left\lbrack  {{0000} - {0003} - {1902} - {3387}}\right\rbrack  }$ ，以及春辉 ${\operatorname{Teo}}^{2\left\lbrack  {{0000} - {0002} - {5724} - {9478}}\right\rbrack  }$ ${}^{1}$ 美国加利福尼亚大学圣巴巴拉分校 \{shanxiuhe,tyang\}@cs.ucsb.edu ${}^{2}$ 美国帕洛阿尔托亚马逊公司 \{mutasema, srjnair, jnatmay, tagar, choonhui\}@amazon.com ${}^{3}$ 美国南加州大学洛杉矶分校

jonmay@isi.edu

Abstract. Multi-vector dense retrieval with ColBERT has been shown to be effective in striking a good relevance and efficiency tradeoff for both in-domain and out-of-domain datasets through late interaction between queries and documents. However, the efficiency of ColBERT for a large-scale retrieval dataset is still constrained by its large memory footprint, as one embedding is stored per token; thus, previous work has studied static pruning of less significant tokens to enhance efficiency. To improve the adaptivity of prior work in zero-shot retrieval settings, this paper proposes a neural classification method that learns pruning decisions with Gumbel-Softmax, and provides an extension to adjust pruning decisions and meet memory space reduction requirements. We evaluate the effectiveness of our proposed method against several baseline approaches on out-of-domain datasets LoTTE and BEIR, and the in-domain MS MARCO passage dataset.

摘要：通过查询与文档之间的后期交互，基于ColBERT的多向量密集检索已被证明能在领域内和领域外数据集上有效平衡相关性和效率。然而，对于大规模检索数据集，ColBERT的效率仍受其巨大内存占用的限制，因为每个Token都存储一个嵌入；因此，先前的工作研究了对不太重要的Token进行静态剪枝以提高效率。为了提高先前工作在零样本检索场景中的适应性，本文提出了一种使用Gumbel - Softmax学习剪枝决策的神经分类方法，并提供了一种扩展方法来调整剪枝决策以满足内存空间缩减要求。我们在领域外数据集LoTTE和BEIR以及领域内MS MARCO段落数据集上，将我们提出的方法与几种基线方法进行了有效性评估。

Keywords: Multi-vector neural representations - Late-interaction dense retrieval - Pruning - Space and time efficiency

关键词：多向量神经表示 - 后期交互密集检索 - 剪枝 - 时空效率

## 1 Introduction

## 1 引言

A transformer-based cross-encoder for document ranking delivers impressive performance but comes with extremely high computational costs during inference. To reduce complexity, many dense retrieval methods adopt a single-vector paradigm to represent queries and documents. However, deploying such single-vector models for large search datasets with diverse content domains and applications, such as product search involving a large number of entities, is challenging. As pointed out in the previous work $\left\lbrack  {{16},{25},{26}}\right\rbrack$ ,the single-vector representation paradigms have limited expressive power when handling out-of-domain datasets with scarce training data (including zero-shot retrieval) and in answering entity-centric questions. ColBERT [12] with a multi-vector representation can be viewed as a middle-ground between the single-vector bi-encoder and the cross-encoder design. Previous studies demonstrate that, compared to sparse representations like SPLADE $\left\lbrack  {7,8}\right\rbrack$ ,ColBERTv2 can achieve comparable performance on the MS MARCO and BEIR datasets $\left\lbrack  {{23},{28}}\right\rbrack$ ,while also offering a relevance advantage when searching an out-of-domain data collection called LoTTE [23]. As a result, multi-vector dense representations can be desirable in certain search applications, especially on GPU-rich platforms where sparse retrieval cannot fully leverage GPU resources.

基于Transformer的用于文档排序的交叉编码器性能出色，但在推理过程中计算成本极高。为降低复杂度，许多密集检索方法采用单向量范式来表示查询和文档。然而，将此类单向量模型应用于具有不同内容领域和应用的大型搜索数据集（如涉及大量实体的产品搜索）具有挑战性。正如先前工作 $\left\lbrack  {{16},{25},{26}}\right\rbrack$ 所指出的，单向量表示范式在处理训练数据稀缺的领域外数据集（包括零样本检索）以及回答以实体为中心的问题时表达能力有限。具有多向量表示的ColBERT [12] 可被视为单向量双编码器和交叉编码器设计之间的折中方案。先前的研究表明，与SPLADE $\left\lbrack  {7,8}\right\rbrack$ 等稀疏表示相比，ColBERTv2在MS MARCO和BEIR数据集 $\left\lbrack  {{23},{28}}\right\rbrack$ 上可取得相当的性能，同时在搜索名为LoTTE的领域外数据集合时还具有相关性优势 [23]。因此，多向量密集表示在某些搜索应用中可能是理想的，特别是在拥有丰富GPU资源的平台上，因为稀疏检索无法充分利用GPU资源。

The expressiveness advantage of ColBERT over single-vector representations comes at a significant cost, namely, an order of magnitude increase in space and computational complexity. Reducing space complexity is important as the memory size of GPUs is much more limited than CPUs and maintaining high memory bandwidth becomes challenging as GPU memory capacity expands. Several efficiency optimization approaches have been developed recently for ColBERT, including offline techniques for space reduction such as quantization to compress the vector size [23], and token pruning [1, 14]. Approaches of online inference acceleration include DESSERT [6], EMVB [20], CITADEL [16], and XTR [15].

与单向量表示相比，ColBERT（一种基于深度学习的信息检索模型）的表达优势是以显著的代价为前提的，即空间和计算复杂度增加了一个数量级。降低空间复杂度非常重要，因为GPU（图形处理器）的内存大小比CPU（中央处理器）要有限得多，并且随着GPU内存容量的扩展，维持高内存带宽变得具有挑战性。最近为ColBERT开发了几种效率优化方法，包括用于减少空间的离线技术，如量化以压缩向量大小[23]和词元剪枝[1, 14]。在线推理加速方法包括DESSERT [6]、EMVB [20]、CITADEL [16]和XTR [15]。

Our paper focuses on token pruning optimization to minimize the number of tokens representing each document, thereby reducing space usage and accelerating inference time. Existing techniques like Top- $k$ token pruning [14] face limitations due to the static nature of parameter $k$ ,which is not learned,making them less effective for out-of-domain search. Similarly, IDF-based pruning [1], while effective in some cases, can be unsafe for certain domains as removing frequently occurring tokens might inadvertently discard critical information. Increasing the IDF method parameter often leads to a significant relevance drop under tight space budgets. These challenges highlight the need for learnable context-aware pruning to improve adaptivity.

我们的论文专注于词元剪枝优化，以最小化表示每个文档的词元数量，从而减少空间使用并加快推理时间。现有的技术，如Top - $k$词元剪枝[14]，由于参数$k$的静态性质（该参数不是通过学习得到的）而存在局限性，这使得它们在域外搜索中效果不佳。同样，基于逆文档频率（IDF）的剪枝[1]虽然在某些情况下有效，但对于某些领域可能不安全，因为移除频繁出现的词元可能会无意中丢弃关键信息。在空间预算紧张的情况下，增加IDF方法的参数通常会导致相关性显著下降。这些挑战凸显了需要可学习的上下文感知剪枝来提高适应性。

This paper develops a context-aware neural module that decides the pruning of tokens for each document in ColBERT-like dense retrieval. Our method, LeapMV (Learnable Pruning for Multi-Vector representations), is trained to minimize the multi-vector dense retrieval loss for one domain and is applicable to zero-shot retrieval in searching out-of-domain datasets. LeapMV can be extended to adjust its pruning aggressiveness to meet different space reduction requirements. Our evaluation shows that LeapMV improves the adaptivity of the previously proposed static pruning techniques in meeting different space reduction targets, which results in latency saving proportionally. Under the same relevance accuracy budget for LoTTE,LeapMV consumes up to ${3.3} \times$ less space and is up-to ${2.8} \times$ faster than the top- $k$ token pruning and IDF methods.

本文开发了一个上下文感知的神经模块，用于在类似ColBERT的密集检索中决定每个文档的词元剪枝。我们的方法LeapMV（多向量表示的可学习剪枝）经过训练，以最小化一个领域的多向量密集检索损失，并且适用于域外数据集搜索中的零样本检索。LeapMV可以进行扩展，以调整其剪枝的激进程度，以满足不同的空间缩减要求。我们的评估表明，LeapMV提高了先前提出的静态剪枝技术在满足不同空间缩减目标方面的适应性，从而按比例节省了延迟。在LoTTE数据集相同的相关性准确率预算下，与Top - $k$词元剪枝和IDF方法相比，LeapMV最多可减少${3.3} \times$的空间消耗，并且速度最多快${2.8} \times$。

## 2 Background and Design Considerations

## 2 背景和设计考虑

The multi-vector representation in ColBERT encodes the tokens in each document (and a query) using embeddings and its index is built based on a set of token embedding vectors for each document. During inference time, ColBERT calculates the similarity between a document $d$ and the given query $q$ via the following "MaxSim" operations,

ColBERT中的多向量表示使用嵌入对每个文档（和查询）中的词元进行编码，并且其索引是基于每个文档的一组词元嵌入向量构建的。在推理时，ColBERT通过以下“最大相似度（MaxSim）”操作计算文档$d$与给定查询$q$之间的相似度：

$$
{S}_{q,d} = \mathop{\sum }\limits_{{i = 1}}^{N}\mathop{\max }\limits_{{j = 1}}^{M}{Q}_{i} \cdot  {D}_{j}^{T} \tag{1}
$$

where $Q$ is the encoded token embedding set for a given query and $D$ the token embedding sequence for each document $d.N$ and $M$ are the token sequence length for a query and a document respectively.

其中$Q$是给定查询的编码词元嵌入集，$D$是每个文档的词元嵌入序列，$d.N$和$M$分别是查询和文档的词元序列长度。

The required ColBERT online memory size for hosting the compressed multi-vector document representation is approximately proportional to the product of the average document length, the token embedding dimension, and the number of documents. This can easily flood out memory space for hosting a large search data collection with a long language-model embedding length. Furthermore, the online inference time complexity is proportional to the average number of tokens per document. Thus reducing the number of tokens per document can lead to near-linear decreases in the memory usage and search inference time.

用于存储压缩多向量文档表示的ColBERT在线内存大小大约与平均文档长度、词元嵌入维度和文档数量的乘积成正比。对于具有较长语言模型嵌入长度的大型搜索数据集，这很容易耗尽内存空间。此外，在线推理时间复杂度与每个文档的平均词元数量成正比。因此，减少每个文档的词元数量可以使内存使用和搜索推理时间近乎线性地减少。

Top- $k$ token pruning. Static token pruning in [14] keeps top- $k$ tokens per document,and several heuristics have been proposed to select such $k$ items. Although the ColBERT model is optimized through training based on the given $k$ hyperparameter,the trained ColBERT model based on one domain (e.g. MS MARCO) with a fixed $k$ value can lose its advantage when this model is directly applied in a zero-shot manner to other search domains which have different document characteristics including lengths and IDF distributions.

Top - $k$词元剪枝。文献[14]中的静态词元剪枝保留每个文档的前$k$个词元，并且已经提出了几种启发式方法来选择这些$k$个词元。尽管ColBERT模型是基于给定的$k$超参数通过训练进行优化的，但基于一个领域（如MS MARCO）且具有固定$k$值的训练好的ColBERT模型，当以零样本方式直接应用于具有不同文档特征（包括长度和逆文档频率分布）的其他搜索领域时，可能会失去其优势。

IDF-based token pruning. The study in [1] compares several IDF-based static pruning methods for ColBERT with MS MARCO and finds the best method is called IDF uniform which removes top- $\tau$ BERT tokens (including stopwords) globally in all documents. Uniformly pruning popular words in all documents can inadvertently remove meaningful tokens in some contexts. For example, "the who" is the name of a well-known rock band, while "who" can stand for the World Health Organization-both of which could be pruned by the above IDF uniform method. When a larger space reduction is required, increasing the IDF filtering parameter can incur a significant relevance loss as demonstrated in our evaluation. Thus we seek a learned neural classifier for context-aware token pruning, applicable for zero-shot retrieval.

基于逆文档频率（IDF）的词元剪枝。文献[1]的研究比较了几种基于IDF的ColBERT静态剪枝方法与MS MARCO数据集，发现最佳方法称为IDF统一剪枝，该方法在所有文档中全局移除前$\tau$个BERT词元（包括停用词）。在所有文档中统一剪枝常见词可能会无意中移除某些上下文中有意义的词元。例如，“the who”是一个著名摇滚乐队的名称，而“who”可以代表世界卫生组织——这两个词都可能被上述IDF统一剪枝方法移除。当需要更大的空间缩减时，增加IDF过滤参数会导致显著的相关性损失，正如我们的评估所示。因此，我们寻求一种用于上下文感知词元剪枝的可学习神经分类器，适用于零样本检索。

Text chunking. We have also considered using text chunking, which originated in NLP research [22]. Chunking divides text into syntactically related, non-overlapping groups of words and reduces the number of representations per document by considering each chunk as a merged token. We have applied text chunking such as sentence chunking and word chunking for ColBERT and have found significant relevance degradations of such methods compared to other baseline methods.

文本分块。我们还考虑过使用文本分块技术，该技术起源于自然语言处理（NLP）研究 [22]。分块将文本划分为句法相关、互不重叠的词组，并通过将每个块视为一个合并的标记来减少每个文档的表示数量。我们已为ColBERT应用了诸如句子分块和单词分块等文本分块方法，发现与其他基线方法相比，这些方法的相关性显著下降。

Quantization and online inference optimization. ColBERTv2 [23] and its highly optimized implementation PLAID [24] quantize every dimension of the encoded vector and represent the document as a summation from the cluster centroid and the residuals. Residual compression can further reduce the index size while preserving retrieval quality. ColBERTv2 and PLAID also use an inverted index with clustering to improve online search speed. Our evaluation has built LeapMV on top of PLAID, taking advantage of its highly optimized implementation. DESSERT [6] proposes to speed up online query processing using a randomized algorithm for LSH-based vector set queries. DESSERT does not reduce storage space, and it uses much more space than ColBERTv2 or PLAID, though its online inference can be fairly fast. CITADEL [16] proposes to route each token vector to the predicted lexical "keys" such that a query token vector only interacts with document token vectors routed to the same key. While CITADEL is faster than PLAID and intends to use quantization, it has been reported to use more space than PLAID when desiring competitive relevance. EMVB [20] proposes an efficient query processing framework for multi-vector dense retrieval with centroid-based document pre-filtering and space compression with product quantization. XTR [15] approximates document rank scoring of multi-vector dense retrieval by retrieving the most important query-specific document tokens first and accounting for the contribution of the missing tokens to the overall score with missing similarity imputation.

量化和在线推理优化。ColBERTv2 [23] 及其高度优化的实现PLAID [24] 对编码向量的每个维度进行量化，并将文档表示为聚类质心和残差的总和。残差压缩可以在保持检索质量的同时进一步减小索引大小。ColBERTv2和PLAID还使用带聚类的倒排索引来提高在线搜索速度。我们的评估在PLAID的基础上构建了LeapMV，利用了其高度优化的实现。DESSERT [6] 提出使用基于局部敏感哈希（LSH）的向量集查询的随机算法来加速在线查询处理。DESSERT不会减少存储空间，并且与ColBERTv2或PLAID相比，它占用的空间要大得多，尽管其在线推理速度相当快。CITADEL [16] 提出将每个标记向量路由到预测的词汇“键”，以便查询标记向量仅与路由到相同键的文档标记向量进行交互。虽然CITADEL比PLAID更快，并且打算使用量化，但据报道，在追求有竞争力的相关性时，它比PLAID占用更多的空间。EMVB [20] 提出了一个用于多向量密集检索的高效查询处理框架，该框架基于质心的文档预过滤和乘积量化的空间压缩。XTR [15] 通过首先检索最重要的特定查询文档标记，并通过缺失相似度插补来考虑缺失标记对总体得分的贡献，来近似多向量密集检索的文档排名得分。

LeapMV is orthogonal and complementary to the above online techniques because online optimization can be applied after offline processing with LeapMV's static token pruning of each document.

LeapMV与上述在线技术是正交且互补的，因为在使用LeapMV对每个文档进行静态标记剪枝的离线处理之后，可以应用在线优化。

## 3 Learnable Pruning for Multi-Vector Representations

## 3 多向量表示的可学习剪枝

We now describe the LeapMV module,as illustrated in Figure 1. Let $D$ be the sequence of $M$ tokens representing each document $d : D = \left( {{D}_{1},{D}_{2},\cdots ,{D}_{M}}\right)$ , where ${D}_{j} \in  {\mathbb{R}}^{h}$ . To decide if the $j$ -th token can be pruned or not,we let ${D}_{j}^{\prime } \in$ ${\mathbb{R}}^{\left( {{2c} + 1}\right)  \cdot  h}$ represent the concatenation of a contextualized window with a window size of $c$ before and after the $j$ -th token. Namely, ${D}_{j}^{\prime } = \left( {{D}_{j - c} \circ  \cdots  \circ  {D}_{j} \circ  \cdots  \circ  {D}_{j + c}}\right)$ , which forms a representation of total size $\left( {{2c} + 1}\right)  \cdot  h$ . As shown in Figure 1,the above representation is forwarded to one fully-connected neural network layer with dropout as a token-level classifier to obtain:

我们现在描述LeapMV模块，如图1所示。设 $D$ 为表示每个文档 $d : D = \left( {{D}_{1},{D}_{2},\cdots ,{D}_{M}}\right)$ 的 $M$ 个标记的序列，其中 ${D}_{j} \in  {\mathbb{R}}^{h}$ 。为了确定第 $j$ 个标记是否可以被剪枝，我们让 ${D}_{j}^{\prime } \in$ ${\mathbb{R}}^{\left( {{2c} + 1}\right)  \cdot  h}$ 表示第 $j$ 个标记前后窗口大小为 $c$ 的上下文窗口的拼接。即 ${D}_{j}^{\prime } = \left( {{D}_{j - c} \circ  \cdots  \circ  {D}_{j} \circ  \cdots  \circ  {D}_{j + c}}\right)$ ，它形成了总大小为 $\left( {{2c} + 1}\right)  \cdot  h$ 的表示。如图1所示，上述表示被输入到一个带有丢弃层（dropout）的全连接神经网络层作为标记级分类器，以获得：

$$
{Y}_{j} = W\left( {{D}_{j}^{\prime } \circ  \xi }\right)  + b\text{,with}j \in  \{ 1,2,\cdots ,M\}  \tag{2}
$$

where $W$ is a weight matrix transforming the inputs into classifier logits. $\xi  \in$ ${\mathbb{R}}^{h}$ ,is a vector of independent noise,drawn from a Bernoulli distribution with probability $1 - p$ with $p$ as dropout rate,and $b$ is a bias term. Logits ${Y}_{j} \in  {\mathbb{R}}^{2}$ denote classification scores for $j$ -th token,i.e. dimension 0 is the unnormalized probability for keeping the token (decision [KEEP]) and dimension 1 the logit for dropping (decision [DROP]).

其中 $W$ 是一个将输入转换为分类器对数几率（logits）的权重矩阵。 $\xi  \in$ ${\mathbb{R}}^{h}$ 是一个独立噪声向量，从概率为 $1 - p$ 的伯努利分布中抽取， $p$ 为丢弃率， $b$ 是一个偏置项。对数几率 ${Y}_{j} \in  {\mathbb{R}}^{2}$ 表示第 $j$ 个标记的分类得分，即维度0是保留该标记的未归一化概率（决策 [保留]），维度1是丢弃该标记的对数几率（决策 [丢弃]）。

To obtain discrete decisions for pruning or keeping a token, we apply Straight-through Gumbel-Softmax $\left\lbrack  {9,{10},{18}}\right\rbrack$ on ${Y}_{j}$ ,i.e.,

为了获得剪枝或保留标记的离散决策，我们对 ${Y}_{j}$ 应用直通Gumbel-Softmax $\left\lbrack  {9,{10},{18}}\right\rbrack$，即：

$$
{\left( {P}_{j}\right) }_{i} = \frac{\exp \left( {\log \left( {{\left( {Y}_{j}\right) }_{i} + {\epsilon }_{i}}\right) /\tau }\right) }{\mathop{\sum }\limits_{{{i}^{\prime } = 0}}^{1}\exp \left( {\log \left( {{\left( {Y}_{j}\right) }_{{i}^{\prime }} + {\epsilon }_{{i}^{\prime }}}\right) /\tau }\right) }. \tag{3}
$$

$$
{u}_{j} = \underset{i \in  \{ 0,1\} }{\arg \max }{\left( {P}_{j}\right) }_{i} \tag{4}
$$

<!-- Media -->

<!-- figureText: Pruning [KEEP] [KEEP] [DROP] [KEEP] DM Final Representations ... ${\mathrm{t}}_{\mathrm{M}}$ Decision Learnable Token Pruning Gumbel-Softmax Linear Layer Dropout Layer Original ${\mathrm{D}}_{1}$ ${\mathrm{D}}_{2}$ ${\mathrm{D}}_{3}$ Embeddings ColBERTv2 Linear Layer BERT Encoder 4 ${\mathrm{t}}_{1}$ ${\mathrm{t}}_{3}$ -->

<img src="https://cdn.noedgeai.com/0195b3d8-eaa8-76c2-97e1-8b1485bd5fdf_4.jpg?x=498&y=344&w=806&h=724&r=0"/>

Fig. 1. LeapMV model architecture for pruning document tokens

图1. 用于剪枝文档标记的LeapMV模型架构

<!-- Media -->

Here ${P}_{j}$ is a vector with normalized probability,where ${\left( {P}_{j}\right) }_{i}$ is the $i$ -th entry of vector ${P}_{j}$ ,and ${\left( {Y}_{j}\right) }_{i}$ is the $i$ -th entry of vector ${Y}_{j}$ . The hyperparameter $\tau$ is the softmax temperature,and ${\epsilon }_{0},{\epsilon }_{1}$ are i.i.d samples from the Gumbel distribution $- \log \left( {-\log \left( {\operatorname{Uniform}\left\lbrack  {0,1}\right\rbrack  }\right) }\right)$ . We obtain ${u}_{j} \in  \{ 0,1\}$ as the prediction for the $j$ th-token [KEEP] or [DROP] decision for document $d$ .

这里 ${P}_{j}$ 是一个具有归一化概率的向量，其中 ${\left( {P}_{j}\right) }_{i}$ 是向量 ${P}_{j}$ 的第 $i$ 个元素，并且 ${\left( {Y}_{j}\right) }_{i}$ 是向量 ${Y}_{j}$ 的第 $i$ 个元素。超参数 $\tau$ 是softmax温度，并且 ${\epsilon }_{0},{\epsilon }_{1}$ 是来自Gumbel分布 $- \log \left( {-\log \left( {\operatorname{Uniform}\left\lbrack  {0,1}\right\rbrack  }\right) }\right)$ 的独立同分布样本。我们得到 ${u}_{j} \in  \{ 0,1\}$ 作为文档 $d$ 的第 $j$ 个标记的 [保留] 或 [丢弃] 决策的预测结果。

Online inference. For scoring shown in Formula (1), only the token predicted as [KEEP] would be included for MaxSim computation. With token pruning under the predicted value ${u}_{j}$ as 0 or 1 for $j$ -token of a document,we compute the rank score of each document with a MaxSim operation as:

在线推理。对于公式 (1) 中所示的评分，只有被预测为 [保留] 的标记才会被纳入MaxSim计算。在文档的第 $j$ 个标记的预测值 ${u}_{j}$ 为0或1的情况下进行标记剪枝，我们通过MaxSim操作计算每个文档的排名分数如下：

$$
{S}_{q,d} = \mathop{\sum }\limits_{{i = 1}}^{N}\mathop{\max }\limits_{{j = 1}}^{M}{Q}_{i} \cdot  \left( {\left( {1 - {u}_{j}}\right)  \cdot  {D}_{j}^{T}}\right) . \tag{5}
$$

In the above expression, tokens predicted as [DROP] are not used for calculating matching scores between queries and documents. The online inference complexity is proportional to the average number of kept tokens instead of the original number of tokens.

在上述表达式中，被预测为 [丢弃] 的标记不用于计算查询和文档之间的匹配分数。在线推理的复杂度与保留标记的平均数量成正比，而不是与原始标记的数量成正比。

Training. During model training, our scheme performs token-level classification, uses the classification results to perform modified late interaction, and receives feedback similar to the original training of ColBERTv2, which includes a knowledge distillation loss from the cross-encoder teacher as well as in-batch negative loss. During training with backpropagation, the Gumbel-SoftMax function provides a differentiable approximation to the argmax function that prunes tokens if they are not selected. It is known that the Gumbel-Softmax distribution is smooth for $\tau  > 0$ ,yielding a well-defined gradient $\frac{\partial {\left( {P}_{j}\right) }_{i}}{\partial {\left( {I}_{j}\right) }_{i}}$ for the predicted soft-max probability. Thus this is used to compute gradients, essentially by replacing discretely classified pruning choices with Gumbel-Softmax samples.

训练。在模型训练期间，我们的方案进行标记级别的分类，使用分类结果进行改进的后期交互，并接收与ColBERTv2的原始训练类似的反馈，其中包括来自交叉编码器教师的知识蒸馏损失以及批次内负样本损失。在使用反向传播进行训练时，Gumbel-SoftMax函数为argmax函数提供了一个可微的近似，该函数在标记未被选中时将其剪枝。已知Gumbel-Softmax分布在 $\tau  > 0$ 时是平滑的，从而为预测的soft-max概率产生一个定义良好的梯度 $\frac{\partial {\left( {P}_{j}\right) }_{i}}{\partial {\left( {I}_{j}\right) }_{i}}$。因此，这用于计算梯度，本质上是用Gumbel-Softmax样本替换离散分类的剪枝选择。

For low temperatures, the softmax computation of Formula (3) smoothly approaches the discrete argmax computation while preserving the relative order of the Gumbels $\left( {\log \left( {{Y}_{j} + \epsilon }\right) /\tau }\right)$ . Samples from Gumbel Softmax distributions become near identical to samples from a [KEEP] and [DROP] class distribution. Thus the MaxSim scoring in training loss computation is approximated as

对于低温情况，公式 (3) 的softmax计算在保留Gumbel值 $\left( {\log \left( {{Y}_{j} + \epsilon }\right) /\tau }\right)$ 的相对顺序的同时，平滑地接近离散的argmax计算。来自Gumbel Softmax分布的样本变得几乎与来自 [保留] 和 [丢弃] 类别分布的样本相同。因此，训练损失计算中的MaxSim评分近似为

$$
{\widetilde{S}}_{q,d} = \mathop{\sum }\limits_{{i = 1}}^{N}\mathop{\max }\limits_{{j = 1}}^{M}{Q}_{i} \cdot  \left( {{\left( {P}_{j}\right) }_{0} \cdot  {D}_{j}^{T}}\right) . \tag{6}
$$

Given a training dataset,let ${\mathcal{V}}^{ + }$ be the subset of all positive documents for training query $q$ ,and ${\mathcal{V}}^{ - }$ be a subset containing all negative documents for query $q$ . The top one probability distribution of positive or negative document ${d}_{i}$ is computed as:

给定一个训练数据集，设 ${\mathcal{V}}^{ + }$ 是训练查询 $q$ 的所有正文档的子集，并且 ${\mathcal{V}}^{ - }$ 是包含查询 $q$ 的所有负文档的子集。正文档或负文档 ${d}_{i}$ 的最高概率分布计算如下：

$$
\frac{\exp \left( {\widetilde{S}}_{q,{d}_{i}}\right) }{\mathop{\sum }\limits_{{{d}_{j} \in  {\mathcal{V}}^{ + } \cup  {\mathcal{V}}^{ - }}}\exp \left( {\widetilde{S}}_{q,{d}_{j}}\right) }.
$$

To train the ColBERT model with the above token pruning classifier, we use the cross-entropy rank loss with KL divergence for knowledge distillation [23].

为了使用上述标记剪枝分类器训练ColBERT模型，我们使用带有KL散度的交叉熵排名损失进行知识蒸馏 [23]。

Notice that there is a difference from the original scoring Formula (1) to the actual online scoring Formula (5) due to token pruning, and there is a training approximation from Formula (1) to Formula (6) for pruning optimization through SGD learning. Our evaluation shows such an approximation leads to a limited relevance drop while gaining a large advantage for space and time-saving.

请注意，由于标记剪枝，原始评分公式 (1) 与实际在线评分公式 (5) 存在差异，并且通过随机梯度下降 (SGD) 学习进行剪枝优化时，从公式 (1) 到公式 (6) 存在训练近似。我们的评估表明，这种近似在相关性上的下降有限，同时在节省空间和时间方面获得了很大的优势。

Adjustment with a space reduction target. Our evaluation finds that the above LeapMV module in Figure 1 yields approximately 50% token pruning. We can modify the Gumbel-Softmax probability outcome for each document towards a more or less aggressive token pruning target based on a desired space reduction goal. Let $\alpha$ be the target percentage of tokens to prune in this document.

根据空间缩减目标进行调整。我们的评估发现，图1中的上述LeapMV模块大约实现了50%的标记剪枝。我们可以根据期望的空间缩减目标，针对每个文档修改Gumbel-Softmax概率结果，以实现更激进或更保守的标记剪枝目标。设 $\alpha$ 是该文档中要剪枝的标记的目标百分比。

Given the outcome of Formula (3) for each token in a document, we sort ${\left( {P}_{j}\right) }_{0}$ in a non-descending order where $1 \leq  j \leq  M$ for this document with $M$ tokens. Without loss of generality,assume after such sorting, ${\left( {P}_{j}\right) }_{0} \leq  {\left( {P}_{j + 1}\right) }_{0}$ where $1 \leq  j \leq  M - 1$ . Pruning tokens from the sorted position 1 to $x$ while keeping all other tokens from position $x + 1$ meets the target pruning ratio $\alpha$ . Then we set

给定文档中每个词元的公式(3)的结果，我们将${\left( {P}_{j}\right) }_{0}$按非降序排序，其中对于这个包含$M$个词元的文档有$1 \leq  j \leq  M$。不失一般性，假设经过这样的排序后，有${\left( {P}_{j}\right) }_{0} \leq  {\left( {P}_{j + 1}\right) }_{0}$，其中$1 \leq  j \leq  M - 1$。从排序后的位置1到$x$修剪词元，同时保留从位置$x + 1$开始的所有其他词元，可满足目标修剪率$\alpha$。然后我们设置

$$
\delta  = {0.5} - {0.5}\left( {{\left( {P}_{x}\right) }_{0} + {\left( {P}_{x + 1}\right) }_{0}}\right) .
$$

We can show that adding offset $\delta$ to the first dimension of the vectors $\left( {P}_{j}\right)$ of this document will allow LeapMV to prune $\alpha$ percentage of tokens approximately. This is because the above offset $\delta$ satisfies

我们可以证明，给该文档向量$\left( {P}_{j}\right)$的第一个维度加上偏移量$\delta$，将使LeapMV大约能修剪$\alpha$百分比的词元。这是因为上述偏移量$\delta$满足

$$
{\left( {P}_{j}\right) }_{0} + \delta  \leq  {\left( {P}_{j}\right) }_{1} - \delta \text{ where }1 \leq  j \leq  x
$$

and

并且

$$
{\left( {P}_{j}\right) }_{0} + \delta  \geq  {\left( {P}_{j}\right) }_{1} - \delta \text{ where }x + 1 \leq  j \leq  M.
$$

With the randomization introduced in Formula (3), the chance of being equal in the above inequalities is small. Then tokens from the sorted positions 1 to $x$ are pruned approximately when applying Formula (4).

由于公式(3)中引入了随机化，上述不等式中出现相等情况的概率很小。那么在应用公式(4)时，大约会修剪掉排序位置从1到$x$的词元。

Thus to adjust the chance of being pruned towards target ratio $\alpha$ ,we modify the vectors $\left( {P}_{j}\right)$ for each document as follows after computation of Formula (3) and before computing Formula (4):

因此，为了将被修剪的概率调整到目标比率$\alpha$，我们在计算公式(3)之后、计算公式(4)之前，对每个文档的向量$\left( {P}_{j}\right)$进行如下修改：

$$
{\left( {P}_{j}\right) }_{0} = {\left( {P}_{j}\right) }_{0} + \delta \text{and}{\left( {P}_{j}\right) }_{1} = {\left( {P}_{j}\right) }_{1} - \delta \text{where}1 \leq  j \leq  M\text{.}
$$

The evaluation in Section 4 finds that while LeapMV cuts about ${50}\%$ of tokens in its default setting without $\alpha$ adjustment,choosing $\alpha$ betwen 25% and ${75}\%$ gives a range towards the desired space reduction target while incurring a small or modest relevance loss.

第4节的评估发现，在默认设置下，LeapMV在不进行$\alpha$调整的情况下会修剪掉约${50}\%$的词元，选择介于25%和${75}\%$之间的$\alpha$，可以在实现所需的空间缩减目标的同时，仅产生较小或适度的相关性损失。

## 4 Evaluation

## 4 评估

Datasets. To evaluate LeapMV for its out-of-domain retrieval quality, we perform experiments on two out-of-domain search data collections:

数据集。为了评估LeapMV的域外检索质量，我们在两个域外搜索数据集合上进行了实验：

LoTTE [23], a collection of 12 test sets, focuses on information-seeking user queries and answers on diverse long-tail topics from StackExchange forums. Topics for each dataset include writing, recreation, science, technology, and lifestyle. Each topic has 400-2100 queries and ${100}\mathrm{k} - 2\mathrm{M}$ passages. The pooled set with ${2.8}\mathrm{M}$ passages contains the passages and queries aggregated across all topics to form a more diverse corpus. LoTTE also provides search and forum queries, gathered from filtered Google search queries [11] and StackExchange communities respectively. The performance is reported separately for each group.

LoTTE [23]是一个包含12个测试集的集合，专注于StackExchange论坛上关于各种长尾主题的信息查询和答案。每个数据集的主题包括写作、娱乐、科学、技术和生活方式。每个主题有400 - 2100个查询和${100}\mathrm{k} - 2\mathrm{M}$个段落。包含${2.8}\mathrm{M}$个段落的合并集包含了所有主题的段落和查询，形成了一个更多样化的语料库。LoTTE还分别提供了从过滤后的谷歌搜索查询[11]和StackExchange社区收集的搜索查询和论坛查询。针对每个组分别报告性能。

BEIR collection [26] consists of 13 publicly available datasets on fact-checking, citation prediction, duplicate question retrieval, argument retrieval, news retrieval, question answering, tweet retrieval, bio-medical IR, and entity retrieval tasks. The size of these data sets varies from a few thousand to ${5.4}\mathrm{M}$ .

BEIR集合[26]由13个公开可用的数据集组成，涉及事实核查、引文预测、重复问题检索、论点检索、新闻检索、问答、推文检索、生物医学信息检索和实体检索任务。这些数据集的大小从几千到${5.4}\mathrm{M}$不等。

Model training uses MS MARCO data [3] with 8.8M passages. We also report in-domain retrieval evaluations with the development set with 6980 queries.

模型训练使用包含880万个段落的MS MARCO数据[3]。我们还报告了使用包含6980个查询的开发集进行的域内检索评估结果。

Implementation. LeapMV is trained under the same setting as the Col-BERTv2, such as the starting checkpoint, learning rates, the number of training steps, and training data, with the exception of using a step accumulation of 2 to reach a batch size of 32. To be more specific, we initialize LeapMV with Col-BERTv1.9 [4], the starting checkpoint released by ColBERT official repository, and train using four 40GB A100 GPUs for ${400}\mathrm{k}$ steps. We train the model using 64-way mined hard negatives provided by ColBERT, consisting of a query, a highly-ranked passage (or gold labels), and one or more negative passages with the same cross-encoder teacher [19]. For indexing, we follow the default PLAID setting of hyper-parameters when the search depth is 1000 unless otherwise mentioned. We adopt a dropout rate of $p = {0.1}$ for Formula (2) and temperature $\tau  = 2/3$ for Formula (3),similar as $\left\lbrack  {5,{18}}\right\rbrack$ .

实现。LeapMV在与Col - BERTv2相同的设置下进行训练，如起始检查点、学习率、训练步数和训练数据等，不同之处在于使用步长累积为2以达到32的批量大小。更具体地说，我们使用ColBERT官方仓库发布的起始检查点Col - BERTv1.9 [4]初始化LeapMV，并使用四个40GB的A100 GPU训练${400}\mathrm{k}$步。我们使用ColBERT提供的64路挖掘的硬负样本训练模型，这些负样本由一个查询、一个高排名的段落（或真实标签）以及一个或多个具有相同交叉编码器教师的负段落组成[19]。对于索引，除非另有说明，当搜索深度为1000时，我们遵循默认的PLAID超参数设置。我们对公式(2)采用$p = {0.1}$的丢弃率，对公式(3)采用$\tau  = 2/3$的温度参数，与$\left\lbrack  {5,{18}}\right\rbrack$类似。

Metrics. We follow the standard practice in the literature, reporting, success@5 (recall@5) for LoTTE, NDCG@10 for BEIR, and mean reciprocal rank at 10 (MRR@10) and recall at 1000 for MS MARCO Dev set.

指标。我们遵循文献中的标准做法，报告LoTTE的success@5（recall@5）、BEIR的NDCG@10，以及MS MARCO开发集的10处平均倒数排名（MRR@10）和1000处召回率。

For efficiency, we report the total amount of in-memory space in gigabytes needed to host the search data and the mean latency to execute a test query in milliseconds with a retrieval depth of 1000. For GPU latency, we measure on one NVIDIA A40 GPU and allow full access to 16 threads available. For CPU latency, we measure on an AMD EPYC 7763 (AMD Milan) running with 8 threads or 1 thread. On average,the 8-thread CPU time is about ${3.7}\mathrm{x}$ faster than the 1-thread CPU. We report the 8-thread CPU time.

为了评估效率，我们报告了存储搜索数据所需的内存总空间（以GB为单位），以及在检索深度为1000的情况下执行测试查询的平均延迟（以毫秒为单位）。对于GPU延迟，我们在一块NVIDIA A40 GPU上进行测量，并允许完全访问可用的16个线程。对于CPU延迟，我们在运行8个线程或1个线程的AMD EPYC 7763（AMD米兰）上进行测量。平均而言，8线程CPU时间比1线程CPU时间快约${3.7}\mathrm{x}$。我们报告的是8线程CPU时间。

Baselines. We compare several baselines discussed in Section 2, which directly reduces the number of tokens in the ColBERT multi-vector representation.

基线方法。我们比较了第2节中讨论的几种基线方法，这些方法直接减少了ColBERT多向量表示中的标记数量。

- Text chunking. We will compare two text chunking methods: phrase-based and sentence-based chunking. Both methods aggregate the embeddings for each phrase or each sentence by mean pooling. We use NLTK-tagger-chunker as labels of boundaries for each phrase and use NLTK sentence tokenization package to obtain sentence partitions [2].

- 文本分块。我们将比较两种文本分块方法：基于短语的分块和基于句子的分块。这两种方法都通过均值池化聚合每个短语或每个句子的嵌入。我们使用NLTK标记器 - 分块器作为每个短语边界的标签，并使用NLTK句子分词包来获得句子划分[2]。

- Static pruning. We have implemented the first- $k$ and attention- $k$ methods of top $k$ pruning [14],where we keep the first $k$ tokens of a document or the top $k$ tokens that receive the most amount of attention in the last layer of the document encoder. We also implemented IDF-uniform with parameter IDF threshold $\tau \left\lbrack  1\right\rbrack$ ,which is the recommended technique from the paper. IDF parameter $\tau$ means to prune the top- $\tau$ tokens with the highest IDF value in MS MARCO corpus uniformly in all documents.

- 静态剪枝。我们实现了前$k$和注意力$k$的前$k$剪枝方法[14]，即保留文档的前$k$个标记，或者保留文档编码器最后一层中获得最多注意力的前$k$个标记。我们还实现了参数为IDF阈值$\tau \left\lbrack  1\right\rbrack$的IDF统一剪枝方法，这是论文中推荐的技术。IDF参数$\tau$表示在所有文档中统一剪去MS MARCO语料库中IDF值最高的前$\tau$个标记。

### 4.1 Zero-shot retrieval performance in out-of-domain datasets

### 4.1 跨领域数据集的零样本检索性能

We evaluate LeapMV outside the training domain to analyze its zero-shot effectiveness against pruning baselines and some state-of-the-art dense retrieval models.

我们在训练领域之外评估LeapMV，以分析其相对于剪枝基线方法和一些最先进的密集检索模型的零样本有效性。

Table 1 presents, under the same relevance accuracy constraints, the index space and latency requirements for different pruning methods in LoTTE pooled dataset. Accuracy 98% indicates that the method retains at least 98% of Success@5 performance compared to PLAID ColBERTv2. Each table entry lists the minimum space size (in GB) and GPU latency (in ms) for each method after pruning, based the PLAID framework, to meet or exceed the specified accuracy.

表1展示了在相同的相关性准确率约束下，LoTTE合并数据集中不同剪枝方法的索引空间和延迟要求。准确率98%表示与PLAID ColBERTv2相比，该方法保留了至少98%的Success@5性能。每个表格条目列出了基于PLAID框架，每种剪枝方法在满足或超过指定准确率后所需的最小空间大小（以GB为单位）和GPU延迟（以毫秒为单位）。

<!-- Media -->

Table 1. Space and latency for LoTTE pooled dataset under different accuracy constraints. PLAID ColBERTv2 takes ${19.6}\mathrm{\;{ms}}$ for the search queries and ${24.8}\mathrm{\;{ms}}$ for the forum queries with an index size of ${13}\mathrm{{GB}}$ .

表1. 不同准确率约束下LoTTE合并数据集的空间和延迟。PLAID ColBERTv2在搜索查询上耗时${19.6}\mathrm{\;{ms}}$，在论坛查询上耗时${24.8}\mathrm{\;{ms}}$，索引大小为${13}\mathrm{{GB}}$。

<table><tr><td rowspan="2">Accuracy (%)</td><td colspan="2">98%</td><td colspan="2">97%</td><td colspan="2">95%</td><td colspan="2">93%</td></tr><tr><td>Space</td><td>Latency</td><td>Space</td><td>Latency</td><td>Space</td><td>Latency</td><td>Space</td><td>Latency</td></tr><tr><td colspan="9">Search test queries</td></tr><tr><td>First</td><td>-</td><td>-</td><td>12.0${2.0}\left( {{2.3} \times  }\right)$</td><td>11.9 (1.5 x)</td><td>9.2$\left( {{2.6} \times  }\right)$</td><td>10.3$\left( {{1.5} \times  }\right)$</td><td>6.9$\left( {{2.5} \times  }\right)$</td><td>9.0$\left( {{1.5} \times  }\right)$</td></tr><tr><td>Attn</td><td>10.0 (1.5x)</td><td>10.6(1.2x)</td><td>8.4(1.6x)</td><td>${9.8}\left( {{1.2} \times  }\right)$</td><td>6.8$\left( {{1.9} \times  }\right)$</td><td>${8.8}\left( {{1.3} \times  }\right)$</td><td>5.9$\left( {{2.1} \times  }\right)$</td><td>8.0(1.3x)</td></tr><tr><td>IDF</td><td>-</td><td>-</td><td>8.8(1.7x)</td><td>20.7 (2.6x)</td><td>7.5$5\left( {{2.1} \times  }\right)$</td><td>17.8 (2.7x)</td><td>6.2$\left( {{2.2} \times  }\right)$</td><td>16.8$\left( {{2.7} \times  }\right)$</td></tr><tr><td>LeapMV</td><td>6.5</td><td>8.5</td><td>5.2</td><td>8.0</td><td>3.5</td><td>6.7</td><td>2.8</td><td>6.2</td></tr><tr><td colspan="9">Forum test queries</td></tr><tr><td>First</td><td>-</td><td>-</td><td>-</td><td>-</td><td>1.0 (3.1x)</td><td>13.3(1.8x)</td><td>9.2(3.3x)</td><td>12.2(1.8x)</td></tr><tr><td>Attn</td><td>11.0(1.7x)</td><td>13.8$\left( {{1.4} \times  }\right)$</td><td>10.0(1.7x)</td><td>13.0(1.4x)</td><td>7.6$\left( {{2.2} \times  }\right)$</td><td>10.7$\left( {{1.4} \times  }\right)$</td><td>5.9$\left( {{2.1} \times  }\right)$</td><td>8.8(1.3x)</td></tr><tr><td>IDF</td><td>8.8(1.4x)</td><td>24.0$\left( {{2.4} \times  }\right)$</td><td>8.0(1.4x)</td><td>21.7$\left( {{2.3} \times  }\right)$</td><td>7.2$\left( {{2.1} \times  }\right)$</td><td>20.8(2.8x)</td><td>6.2$\left( {{2.2} \times  }\right)$</td><td>18.3$\left( {{2.7} \times  }\right)$</td></tr><tr><td>LeapMV</td><td>6.5</td><td>10.1</td><td>5.9</td><td>9.4</td><td>3.5</td><td>7.4</td><td>2.8</td><td>6.7</td></tr></table>

<table><tbody><tr><td rowspan="2">准确率（%）</td><td colspan="2">98%</td><td colspan="2">97%</td><td colspan="2">95%</td><td colspan="2">93%</td></tr><tr><td>空间</td><td>延迟</td><td>空间</td><td>延迟</td><td>空间</td><td>延迟</td><td>空间</td><td>延迟</td></tr><tr><td colspan="9">搜索测试查询</td></tr><tr><td>第一</td><td>-</td><td>-</td><td>12.0${2.0}\left( {{2.3} \times  }\right)$</td><td>11.9 (1.5 x)</td><td>9.2$\left( {{2.6} \times  }\right)$</td><td>10.3$\left( {{1.5} \times  }\right)$</td><td>6.9$\left( {{2.5} \times  }\right)$</td><td>9.0$\left( {{1.5} \times  }\right)$</td></tr><tr><td>注意力（Attn）</td><td>10.0 (1.5x)</td><td>10.6(1.2x)</td><td>8.4(1.6x)</td><td>${9.8}\left( {{1.2} \times  }\right)$</td><td>6.8$\left( {{1.9} \times  }\right)$</td><td>${8.8}\left( {{1.3} \times  }\right)$</td><td>5.9$\left( {{2.1} \times  }\right)$</td><td>8.0(1.3x)</td></tr><tr><td>逆文档频率（IDF）</td><td>-</td><td>-</td><td>8.8(1.7x)</td><td>20.7 (2.6x)</td><td>7.5$5\left( {{2.1} \times  }\right)$</td><td>17.8 (2.7x)</td><td>6.2$\left( {{2.2} \times  }\right)$</td><td>16.8$\left( {{2.7} \times  }\right)$</td></tr><tr><td>跳跃多向量（LeapMV）</td><td>6.5</td><td>8.5</td><td>5.2</td><td>8.0</td><td>3.5</td><td>6.7</td><td>2.8</td><td>6.2</td></tr><tr><td colspan="9">论坛测试查询</td></tr><tr><td>第一</td><td>-</td><td>-</td><td>-</td><td>-</td><td>1.0 (3.1x)</td><td>13.3(1.8x)</td><td>9.2(3.3x)</td><td>12.2(1.8x)</td></tr><tr><td>注意力（Attn）</td><td>11.0(1.7x)</td><td>13.8$\left( {{1.4} \times  }\right)$</td><td>10.0(1.7x)</td><td>13.0(1.4x)</td><td>7.6$\left( {{2.2} \times  }\right)$</td><td>10.7$\left( {{1.4} \times  }\right)$</td><td>5.9$\left( {{2.1} \times  }\right)$</td><td>8.8(1.3x)</td></tr><tr><td>逆文档频率（IDF）</td><td>8.8(1.4x)</td><td>24.0$\left( {{2.4} \times  }\right)$</td><td>8.0(1.4x)</td><td>21.7$\left( {{2.3} \times  }\right)$</td><td>7.2$\left( {{2.1} \times  }\right)$</td><td>20.8(2.8x)</td><td>6.2$\left( {{2.2} \times  }\right)$</td><td>18.3$\left( {{2.7} \times  }\right)$</td></tr><tr><td>跳跃多向量（LeapMV）</td><td>6.5</td><td>10.1</td><td>5.9</td><td>9.4</td><td>3.5</td><td>7.4</td><td>2.8</td><td>6.7</td></tr></tbody></table>

Table 2. Relevance for LoTTE pooled dataset under space budgets (percentage of the PLAID space size). Entries are best Success@5 one method achieves under a space constraint.

表2. LoTTE合并数据集在空间预算（PLAID空间大小的百分比）下的相关性。条目为一种方法在空间约束下实现的最佳前5准确率（Success@5）。

<table><tr><td>Space budget</td><td>75%</td><td>50%</td><td>40%</td><td>32.5%</td><td>25%</td><td>17.5%</td></tr><tr><td>LoTTE search queries</td><td colspan="6">Maximum Success@5</td></tr><tr><td>First</td><td>68.3</td><td>65.3</td><td>63.1</td><td>61.1</td><td>57.8</td><td>53.8</td></tr><tr><td>Attn</td><td>70.6</td><td>67.9</td><td>66.6</td><td>65.1</td><td>60.5</td><td>49.8</td></tr><tr><td>IDF-uniform</td><td>70.3</td><td>67.5</td><td>62.6</td><td>60.7</td><td>59.5</td><td>-</td></tr><tr><td>LeapMV</td><td>70.7</td><td>70.6</td><td>70.0</td><td>69.4</td><td>67.8</td><td>66.1</td></tr><tr><td>LoTTE forum queries</td><td colspan="6">Maximum Success@5</td></tr><tr><td>First</td><td>58.9</td><td>56.3</td><td>55.2</td><td>53.1</td><td>50.3</td><td>45.6</td></tr><tr><td>Attn</td><td>61.8</td><td>59.1</td><td>58.1</td><td>56.0</td><td>51.8</td><td>43.0</td></tr><tr><td>IDF-uniform</td><td>62.5</td><td>59.9</td><td>56.3</td><td>54.0</td><td>51.8</td><td>48.8</td></tr><tr><td>LeapMV</td><td>62.7</td><td>62.2</td><td>61.4</td><td>60.9</td><td>59.8</td><td>57.5</td></tr></table>

<table><tbody><tr><td>空间预算</td><td>75%</td><td>50%</td><td>40%</td><td>32.5%</td><td>25%</td><td>17.5%</td></tr><tr><td>乐天搜索查询（LoTTE search queries）</td><td colspan="6">最大成功率@5</td></tr><tr><td>第一个</td><td>68.3</td><td>65.3</td><td>63.1</td><td>61.1</td><td>57.8</td><td>53.8</td></tr><tr><td>注意</td><td>70.6</td><td>67.9</td><td>66.6</td><td>65.1</td><td>60.5</td><td>49.8</td></tr><tr><td>逆文档频率均匀分布（IDF-uniform）</td><td>70.3</td><td>67.5</td><td>62.6</td><td>60.7</td><td>59.5</td><td>-</td></tr><tr><td>LeapMV（保留原词，可能为特定名称）</td><td>70.7</td><td>70.6</td><td>70.0</td><td>69.4</td><td>67.8</td><td>66.1</td></tr><tr><td>乐天论坛查询（LoTTE forum queries）</td><td colspan="6">最大成功率@5</td></tr><tr><td>第一个</td><td>58.9</td><td>56.3</td><td>55.2</td><td>53.1</td><td>50.3</td><td>45.6</td></tr><tr><td>注意</td><td>61.8</td><td>59.1</td><td>58.1</td><td>56.0</td><td>51.8</td><td>43.0</td></tr><tr><td>逆文档频率均匀分布（IDF-uniform）</td><td>62.5</td><td>59.9</td><td>56.3</td><td>54.0</td><td>51.8</td><td>48.8</td></tr><tr><td>LeapMV（保留原词，可能为特定名称）</td><td>62.7</td><td>62.2</td><td>61.4</td><td>60.9</td><td>59.8</td><td>57.5</td></tr></tbody></table>

<!-- Media -->

To meet the desired accuracy,we adjust $k$ for the first- $k$ and attention- $k$ methods,IDF threshold $\tau$ in IDF-uniform,and $\alpha$ pruning ratio in LeapMV. The entry '-' means that such a method did not deliver the corresponding accuracy under the tested settings. Table 1 demonstrates that, under the same relevance constraint, LeapMV requires significantly less space and operates much faster. For example,with less than $2\%$ loss (98% accuracy),LeapMV achieves $2 \times$ reduction in space and ${2.5} \times$ speedup on GPU compared to PLAID. To achieve ${97}\%$ of accuracy, LeapMV requires 5.2GB for search test queries, while IDF-uniform uses ${1.7} \times$ more space and takes ${2.6} \times$ longer.

为达到期望的准确率，我们针对首次$k$和注意力$k$方法调整$k$，调整IDF统一法（IDF-uniform）中的IDF阈值$\tau$，以及跳跃式多向量法（LeapMV）中的$\alpha$剪枝率。条目“ - ”表示该方法在测试设置下未能达到相应的准确率。表1显示，在相同的相关性约束下，跳跃式多向量法（LeapMV）所需的空间显著减少，运行速度也快得多。例如，在损失小于$2\%$（准确率为98%）的情况下，与格子索引法（PLAID）相比，跳跃式多向量法（LeapMV）的空间减少了$2 \times$，在GPU上的速度提升了${2.5} \times$。为达到${97}\%$的准确率，跳跃式多向量法（LeapMV）的搜索测试查询需要5.2GB的空间，而IDF统一法（IDF-uniform）使用的空间多${1.7} \times$，耗时多${2.6} \times$。

Table 2 lists the retrieval quality, measured with Success@5 in LoTTE, under the same space budget constraint. Space size 75% means to have the index of at most about ${75}\%$ of original ColBERTv2 PLAID,which is approximately 13GB for LoTTE pooled dataset. For example, using about 50% of the space Col-BERTv2 PLAID needs (about ${6.5}\mathrm{{GB}}$ ),LeapMV can deliver Success@5 of 70.6 while other methods can deliver at most 67.9. When adhering to a tight space budget,such as ${17.5}\%$ of the original index,the baseline methods experience a decrease in performance from 25-31% compared to PLAID in search queries. In contrast, LeapMV only shows an 8.3% decline. Overall, LeapMV consistently delivers better relevance under the same space budget and is robust under more aggressive compression ratio.

表2列出了在相同的空间预算约束下，用LoTTE中的前5检索成功率（Success@5）衡量的检索质量。空间大小为75%意味着索引最多约为原始ColBERTv2格子索引法（PLAID）的${75}\%$，对于LoTTE合并数据集而言，这大约是13GB。例如，使用Col - BERTv2格子索引法（PLAID）所需空间的约50%（约${6.5}\mathrm{{GB}}$），跳跃式多向量法（LeapMV）的前5检索成功率（Success@5）可达70.6，而其他方法最多只能达到67.9。当遵循严格的空间预算时，如原始索引的${17.5}\%$，与格子索引法（PLAID）相比，基准方法在搜索查询中的性能下降了25 - 31%。相比之下，跳跃式多向量法（LeapMV）仅下降了8.3%。总体而言，在相同的空间预算下，跳跃式多向量法（LeapMV）始终能提供更好的相关性，并且在更激进的压缩比下表现稳健。

<!-- Media -->

Table 3. Zero-shot retrieval performance on LoTTE test set benchmark.

表3. LoTTE测试集基准上的零样本检索性能。

<table><tr><td/><td>Rocket Retro -QAv2</td><td>-MAE</td><td>PLAID</td><td>First $\mathrm{k} = {50}$</td><td>Attn k=50</td><td>IDF $\tau  = {100}$</td><td>IDF $\tau  = {300}$</td><td>LeapMV</td></tr><tr><td>Space</td><td>-</td><td>-</td><td>13GB</td><td>5.1GB</td><td>5.1GB</td><td>8GB</td><td>6.2GB</td><td>6.5GB</td></tr><tr><td/><td colspan="8">LoTTE search testqueries(Success@5)</td></tr><tr><td>Writing</td><td>78.0</td><td>-</td><td>80.6</td><td>74.2</td><td>76.3</td><td>76.5</td><td>74.3</td><td>76.2</td></tr><tr><td>Recreation</td><td>72.1</td><td>-</td><td>72.3</td><td>64.1</td><td>67.2</td><td>72.4</td><td>71.4</td><td>72.2</td></tr><tr><td>Science</td><td>55.3</td><td>-</td><td>56.7</td><td>49.8</td><td>53.2</td><td>55.4</td><td>54.3</td><td>55.3</td></tr><tr><td>Technology</td><td>63.4</td><td>-</td><td>66.1</td><td>56.5</td><td>59.4</td><td>63.6</td><td>62.6</td><td>67.1</td></tr><tr><td>Lifestyle</td><td>82.1</td><td>-</td><td>84.3</td><td>73.1</td><td>80.9</td><td>83.4</td><td>82.5</td><td>84.7</td></tr><tr><td>Pooled</td><td>69.8</td><td>66.8</td><td>72.1</td><td>63.1</td><td>66.6</td><td>69.3</td><td>67.5</td><td>70.2</td></tr><tr><td>% Loss</td><td>-</td><td>-</td><td>0%</td><td>12.5%</td><td>7.6%</td><td>3.9%</td><td>6.4%</td><td>2.6%</td></tr><tr><td/><td colspan="8">LoTTE forum test quequeries(Success@5)</td></tr><tr><td>Writing</td><td>71.5</td><td>-</td><td>76.2</td><td>70.9</td><td>73.8</td><td>74.0</td><td>72.0</td><td>75.8</td></tr><tr><td>Recreation</td><td>65.7</td><td>-</td><td>71.4</td><td>60.4</td><td>66.5</td><td>70.9</td><td>69.3</td><td>69.6</td></tr><tr><td>Science</td><td>38.0</td><td>-</td><td>47.2</td><td>39.0</td><td>39.9</td><td>45.9</td><td>44.1</td><td>44.4</td></tr><tr><td>Technology</td><td>47.3</td><td>-</td><td>53.7</td><td>48.3</td><td>46.8</td><td>52.1</td><td>49.9</td><td>53.3</td></tr><tr><td>Lifestyle</td><td>73.7</td><td>-</td><td>76.9</td><td>68.0</td><td>72.7</td><td>76.5</td><td>75.2</td><td>77.1</td></tr><tr><td>Pooled</td><td>57.7</td><td>58.5</td><td>63.5</td><td>55.2</td><td>58.1</td><td>62.1</td><td>59.9</td><td>62.1</td></tr><tr><td>% Loss</td><td>-</td><td>-</td><td>0%</td><td>13.1%</td><td>8.5%</td><td>2.2%</td><td>5.7%</td><td>2.2%</td></tr></table>

<table><tbody><tr><td></td><td>复古火箭 - QAv2</td><td>- 平均绝对误差（MAE）</td><td>格子图案（PLAID）</td><td>第一个 $\mathrm{k} = {50}$</td><td>注意力机制 k = 50</td><td>逆文档频率 $\tau  = {100}$（IDF $\tau  = {100}$）</td><td>逆文档频率 $\tau  = {300}$（IDF $\tau  = {300}$）</td><td>跳跃式多视图（LeapMV）</td></tr><tr><td>空间</td><td>-</td><td>-</td><td>13GB</td><td>5.1GB</td><td>5.1GB</td><td>8GB</td><td>6.2GB</td><td>6.5GB</td></tr><tr><td></td><td colspan="8">乐天搜索测试查询（前5个结果成功）（LoTTE search testqueries(Success@5)）</td></tr><tr><td>写作</td><td>78.0</td><td>-</td><td>80.6</td><td>74.2</td><td>76.3</td><td>76.5</td><td>74.3</td><td>76.2</td></tr><tr><td>娱乐</td><td>72.1</td><td>-</td><td>72.3</td><td>64.1</td><td>67.2</td><td>72.4</td><td>71.4</td><td>72.2</td></tr><tr><td>科学</td><td>55.3</td><td>-</td><td>56.7</td><td>49.8</td><td>53.2</td><td>55.4</td><td>54.3</td><td>55.3</td></tr><tr><td>技术</td><td>63.4</td><td>-</td><td>66.1</td><td>56.5</td><td>59.4</td><td>63.6</td><td>62.6</td><td>67.1</td></tr><tr><td>生活方式</td><td>82.1</td><td>-</td><td>84.3</td><td>73.1</td><td>80.9</td><td>83.4</td><td>82.5</td><td>84.7</td></tr><tr><td>合并的</td><td>69.8</td><td>66.8</td><td>72.1</td><td>63.1</td><td>66.6</td><td>69.3</td><td>67.5</td><td>70.2</td></tr><tr><td>损失百分比</td><td>-</td><td>-</td><td>0%</td><td>12.5%</td><td>7.6%</td><td>3.9%</td><td>6.4%</td><td>2.6%</td></tr><tr><td></td><td colspan="8">乐天论坛测试查询（前5个结果成功）（LoTTE forum test quequeries(Success@5)）</td></tr><tr><td>写作</td><td>71.5</td><td>-</td><td>76.2</td><td>70.9</td><td>73.8</td><td>74.0</td><td>72.0</td><td>75.8</td></tr><tr><td>娱乐</td><td>65.7</td><td>-</td><td>71.4</td><td>60.4</td><td>66.5</td><td>70.9</td><td>69.3</td><td>69.6</td></tr><tr><td>科学</td><td>38.0</td><td>-</td><td>47.2</td><td>39.0</td><td>39.9</td><td>45.9</td><td>44.1</td><td>44.4</td></tr><tr><td>技术</td><td>47.3</td><td>-</td><td>53.7</td><td>48.3</td><td>46.8</td><td>52.1</td><td>49.9</td><td>53.3</td></tr><tr><td>生活方式</td><td>73.7</td><td>-</td><td>76.9</td><td>68.0</td><td>72.7</td><td>76.5</td><td>75.2</td><td>77.1</td></tr><tr><td>合并的</td><td>57.7</td><td>58.5</td><td>63.5</td><td>55.2</td><td>58.1</td><td>62.1</td><td>59.9</td><td>62.1</td></tr><tr><td>损失百分比</td><td>-</td><td>-</td><td>0%</td><td>13.1%</td><td>8.5%</td><td>2.2%</td><td>5.7%</td><td>2.2%</td></tr></tbody></table>

<!-- Media -->

To assess the performance for each subtopics of LoTTE, Table 3 presents the Success@5 (Recall@5) of different methods with several parameter settings with index sizes comparable to our default LeapMV. As a point of reference, we also include the average NDCG@10 of two recent single-vector dense retrievers Retro-MAE [17] and RocketQAv2 [21]. The result from Table 3 shows that LeapMV can deliver about $2 \times$ reduction in space usage while incurring only ${2.6}\%$ and ${2.2}\%$ relevance loss. In comparison,IDF-uniform with $\tau  = {100}$ can deliver visibly smaller ${1.6} \times$ space reduction while having a larger average loss of relevance $({3.9}\%$ and 2.2%). These findings are consistent with the results discussed in Table 1 and Table 2.

为评估LoTTE各子主题的性能，表3展示了不同方法在几种参数设置下的Success@5（Recall@5），这些参数设置下的索引大小与我们默认的LeapMV相当。作为参考，我们还列出了最近的两种单向量密集检索器Retro - MAE [17]和RocketQAv2 [21]的平均NDCG@10。表3的结果显示，LeapMV可以将空间使用量减少约$2 \times$，同时仅产生${2.6}\%$和${2.2}\%$的相关性损失。相比之下，具有$\tau  = {100}$的IDF - uniform可以实现明显更小的${1.6} \times$空间缩减，但平均相关性损失更大（$({3.9}\%$和2.2%）。这些发现与表1和表2中讨论的结果一致。

Table 4 shows the zero-shot retrieval performance, measured in NDCG@10, across 13 BEIR datasets for out-of-domain search. For relevance comparison, we also list NDCG@10 of single-vector dense retrievers RetroMAE and Rock-etQAv2. The targeted space reduction is about ${50}\%$ of the PLAID ColBERTv2 size,achieved by selecting $k = {50}$ for top- $k$ pruning and $\tau  = {1000}$ for IDF. By default,LeapMV reduces space usage by about $2 \times$ ,with only a ${3.28}\%$ relevance loss compared to ColBERTv2 PLAID. In comparison,the first- $k$ ,attention- $k$ , and IDF uniform have an NDCG@10 loss varying from 8.98% to 16%.

表4展示了在13个BEIR数据集上进行域外搜索时的零样本检索性能，以NDCG@10衡量。为进行相关性比较，我们还列出了单向量密集检索器RetroMAE和RocketQAv2的NDCG@10。目标空间缩减约为PLAID ColBERTv2大小的${50}\%$，这是通过选择$k = {50}$进行前$k$剪枝和选择$\tau  = {1000}$用于逆文档频率（IDF）实现的。默认情况下，与ColBERTv2 PLAID相比，LeapMV将空间使用量减少了约$2 \times$，而相关性损失仅为${3.28}\%$。相比之下，前$k$、注意力$k$和IDF均匀方法的NDCG@10损失在8.98%至16%之间。

<!-- Media -->

Table 4. Zero-shot retrieval performance on 13 BEIR datasets.

表4. 13个BEIR数据集上的零样本检索性能。

<table><tr><td>BEIR Datasets</td><td>Rocket- Retro- QAv2</td><td>MAE</td><td>PLAID</td><td colspan="3">First- $k$Attn- $k$IDF- $\tau$</td><td>LeapMV</td></tr><tr><td>% of PLAID space</td><td>-</td><td>-</td><td>100%</td><td>54%</td><td>54%</td><td>49%</td><td>49%</td></tr><tr><td>SciFact</td><td>56.8</td><td>65.3</td><td>69.3</td><td>54.9</td><td>47.5</td><td>62.9</td><td>68.9</td></tr><tr><td>NFCorpus</td><td>29.3</td><td>30.8</td><td>33.8</td><td>30.7</td><td>26.5</td><td>31.9</td><td>34.1</td></tr><tr><td>ArguAna</td><td>45.1</td><td>43.3</td><td>46.3</td><td>44.5</td><td>45.1</td><td>41.4</td><td>44.9</td></tr><tr><td>SCIDOCS</td><td>13.1</td><td>15.0</td><td>15.4</td><td>14.8</td><td>13.9</td><td>14.6</td><td>15.2</td></tr><tr><td>Touche-2020</td><td>24.7</td><td>23.7</td><td>26.3</td><td>22.8</td><td>22.1</td><td>26.2</td><td>24.0</td></tr><tr><td>FiQA</td><td>30.2</td><td>31.6</td><td>35.6</td><td>26.2</td><td>30.0</td><td>30.3</td><td>34.6</td></tr><tr><td>T-COVID</td><td>67.5</td><td>77.2</td><td>73.8</td><td>68.5</td><td>55.2</td><td>61.7</td><td>75.2</td></tr><tr><td>NQ</td><td>50.5</td><td>51.8</td><td>56.1</td><td>48.5</td><td>48.4</td><td>48.8</td><td>54.5</td></tr><tr><td>DBPedia</td><td>35.6</td><td>39.0</td><td>44.6</td><td>42.5</td><td>42.6</td><td>41.6</td><td>42.5</td></tr><tr><td>HotpotQA</td><td>53.3</td><td>63.5</td><td>66.7</td><td>57.8</td><td>58.5</td><td>62.6</td><td>59.6</td></tr><tr><td>FEVER</td><td>67.6</td><td>77.4</td><td>78.5</td><td>63.3</td><td>56.5</td><td>74.8</td><td>74.1</td></tr><tr><td>C-FEVER</td><td>18.0</td><td>23.2</td><td>17.6</td><td>13.7</td><td>11.5</td><td>17.6</td><td>18.1</td></tr><tr><td>Quora</td><td>74.9</td><td>84.7</td><td>85.2</td><td>85.4</td><td>85.4</td><td>76.9</td><td>82.5</td></tr><tr><td>Average NDCG@10</td><td>43.6</td><td>48.2</td><td>50.0</td><td>44.1</td><td>41.8</td><td>45.5</td><td>48.3</td></tr><tr><td>% Loss</td><td>-</td><td>-</td><td>0%</td><td>11.69%</td><td>16.35%</td><td>8.98%</td><td>3.28%</td></tr></table>

<table><tbody><tr><td>BEIR数据集</td><td>火箭-复古-QA v2</td><td>平均绝对误差（MAE）</td><td>PLAID（暂未找到通用中文译名）</td><td colspan="3">首个 - $k$注意力 - $k$逆文档频率 - $\tau$</td><td>LeapMV（暂未找到通用中文译名）</td></tr><tr><td>PLAID空间占比</td><td>-</td><td>-</td><td>100%</td><td>54%</td><td>54%</td><td>49%</td><td>49%</td></tr><tr><td>科学事实数据集（SciFact）</td><td>56.8</td><td>65.3</td><td>69.3</td><td>54.9</td><td>47.5</td><td>62.9</td><td>68.9</td></tr><tr><td>自然语言处理领域语料库（NFCorpus）</td><td>29.3</td><td>30.8</td><td>33.8</td><td>30.7</td><td>26.5</td><td>31.9</td><td>34.1</td></tr><tr><td>论证分析数据集（ArguAna）</td><td>45.1</td><td>43.3</td><td>46.3</td><td>44.5</td><td>45.1</td><td>41.4</td><td>44.9</td></tr><tr><td>科学文档数据集（SCIDOCS）</td><td>13.1</td><td>15.0</td><td>15.4</td><td>14.8</td><td>13.9</td><td>14.6</td><td>15.2</td></tr><tr><td>图什2020数据集（Touche - 2020）</td><td>24.7</td><td>23.7</td><td>26.3</td><td>22.8</td><td>22.1</td><td>26.2</td><td>24.0</td></tr><tr><td>金融问答数据集（FiQA）</td><td>30.2</td><td>31.6</td><td>35.6</td><td>26.2</td><td>30.0</td><td>30.3</td><td>34.6</td></tr><tr><td>T - 新冠数据集（T - COVID）</td><td>67.5</td><td>77.2</td><td>73.8</td><td>68.5</td><td>55.2</td><td>61.7</td><td>75.2</td></tr><tr><td>自然问题数据集（NQ）</td><td>50.5</td><td>51.8</td><td>56.1</td><td>48.5</td><td>48.4</td><td>48.8</td><td>54.5</td></tr><tr><td>DBPedia（多语言百科知识图谱，暂未找到通用中文译名）</td><td>35.6</td><td>39.0</td><td>44.6</td><td>42.5</td><td>42.6</td><td>41.6</td><td>42.5</td></tr><tr><td>火锅问答数据集（HotpotQA）</td><td>53.3</td><td>63.5</td><td>66.7</td><td>57.8</td><td>58.5</td><td>62.6</td><td>59.6</td></tr><tr><td>事实验证数据集（FEVER）</td><td>67.6</td><td>77.4</td><td>78.5</td><td>63.3</td><td>56.5</td><td>74.8</td><td>74.1</td></tr><tr><td>中文事实验证数据集（C - FEVER）</td><td>18.0</td><td>23.2</td><td>17.6</td><td>13.7</td><td>11.5</td><td>17.6</td><td>18.1</td></tr><tr><td>Quora问答平台</td><td>74.9</td><td>84.7</td><td>85.2</td><td>85.4</td><td>85.4</td><td>76.9</td><td>82.5</td></tr><tr><td>平均归一化折损累积增益@10（Average NDCG@10）</td><td>43.6</td><td>48.2</td><td>50.0</td><td>44.1</td><td>41.8</td><td>45.5</td><td>48.3</td></tr><tr><td>损失百分比</td><td>-</td><td>-</td><td>0%</td><td>11.69%</td><td>16.35%</td><td>8.98%</td><td>3.28%</td></tr></tbody></table>

<!-- Media -->

### 4.2 In-domain search with MS MARCO

### 4.2 使用MS MARCO进行领域内搜索

Table 5 lists the performance of LeapMV for MS MARCO passage ranking in comparison with the original PLAID ColBERTv2, phrase chunking, sentence chunking,top- $k$ static pruning,and IDF-uniform. The results from Table 5 show the relevance of LeapMV is fairly close to static top- $k$ pruning which uses a fixed top $k$ parameter. LeapMV incurs a relevance loss of 2.7% for MRR@10 and 1.6% for Recall@1K compared to the original PLAID implementation of ColBERTv2, while achieving approximately $2 \times$ space reduction and ${1.8} \times$ faster GPU latency. In contrast,IDF uniform with $\tau  = {100}$ uses ${33}\%$ more space than LeapMV default while their Recall@1K difference is within 1%. When increasing $\tau$ to 3000,IDF’s space cost 5.4GB approaches LeapMV with $\alpha  = {80}\%$ ,but results in a rapid decline in relevance, making it substantially less effective than LeapMV. Similar conclusions can be drawn with the first $k$ and the attention $k$ method, demonstrating the robustness of LeapMV. The two text chunking methods are not competitive due to their significant losses in retrieval performance.

表5列出了LeapMV在MS MARCO段落排序任务中的性能，并与原始的PLAID ColBERTv2、短语分块、句子分块、前$k$静态剪枝和IDF均匀法进行了比较。表5的结果显示，LeapMV的相关性与使用固定前$k$参数的静态前$k$剪枝方法相当接近。与ColBERTv2的原始PLAID实现相比，LeapMV在MRR@10上的相关性损失为2.7%，在Recall@1K上的相关性损失为1.6%，同时实现了约$2 \times$的空间缩减和${1.8} \times$的GPU延迟加速。相比之下，使用$\tau  = {100}$的IDF均匀法比LeapMV默认设置占用${33}\%$更多的空间，而它们在Recall@1K上的差异在1%以内。当将$\tau$增加到3000时，IDF的空间成本达到5.4GB，接近使用$\alpha  = {80}\%$的LeapMV，但相关性急剧下降，使其效果远不如LeapMV。对于前$k$和注意力$k$方法也可以得出类似的结论，这证明了LeapMV的鲁棒性。由于两种文本分块方法在检索性能上有显著损失，因此它们不具竞争力。

As a reference, we also list the relevance performance of single-vector dense retrieval RetroMAE and RocketQAv2 [21]. The released RetroMAE checkpoint gives 0.359 MRR@10 using the standard MS MARCO, and this is below 0.416 reported in [17]. The reason was explained in [13] that, following RocketQAv2 [21], the paper evaluates the modified MS MARCO dataset with title annotation, which is not fair. Since the original dataset released does not utilize title information, all experiments in ColBERT and our evaluation follows the standard approach to use the original MS MARCO without title annotation.

作为参考，我们还列出了单向量密集检索方法RetroMAE和RocketQAv2 [21]的相关性性能。发布的RetroMAE检查点在标准MS MARCO数据集上的MRR@10为0.359，低于文献[17]中报告的0.416。文献[13]解释了原因，即该论文遵循RocketQAv2 [21]的做法，评估了带有标题注释的修改版MS MARCO数据集，这是不公平的。由于最初发布的数据集未使用标题信息，因此ColBERT中的所有实验和我们的评估都遵循标准方法，使用没有标题注释的原始MS MARCO数据集。

<!-- Media -->

Table 5. End-to-end in-domain retrieval performance for MS MARCO passages.

表5. MS MARCO段落的端到端领域内检索性能。

<table><tr><td rowspan="2">Methods</td><td colspan="2">Relevance</td><td colspan="3">Latency ms (Speedup)</td><td colspan="2">Space</td></tr><tr><td colspan="2">MRR@10 R@1K</td><td>GPU</td><td>8-thr.</td><td>CPU</td><td>GB</td><td>(Ratio)</td></tr><tr><td>RocketQA</td><td>38.8</td><td>98.1</td><td>--</td><td>-</td><td>-</td><td>27</td><td>-</td></tr><tr><td>RetroMAE (no title anno.)</td><td>36.0</td><td>97.7</td><td>--</td><td>-</td><td>-</td><td>27</td><td>-</td></tr><tr><td>PLAID ColBERTv2</td><td>39.4</td><td>97.6</td><td>18.9$\left( {1 \times  }\right)$</td><td>69.1</td><td>$\left( {1 \times  }\right)$</td><td>22</td><td>$\left( {1 \times  }\right)$</td></tr><tr><td>Phase Chunking</td><td>30.0</td><td>89.6</td><td>11.5(1.6x)</td><td>56.5</td><td>$\left( {{1.2} \times  }\right)$</td><td>12</td><td>(1.8x)</td></tr><tr><td>Sentence Chunking</td><td>22.0</td><td>69.4</td><td>5.6(3.4x)</td><td>16.3</td><td>$\left( {{4.2} \times  }\right)$</td><td>2.1</td><td>(10.5x)</td></tr><tr><td>First $\mathrm{k} = {15}$</td><td>31.7</td><td>84.7</td><td>5.9$\left( {{3.2} \times  }\right)$</td><td>30.6</td><td>$\left( {{2.3} \times  }\right)$</td><td>4.9</td><td>$\left( {{4.5} \times  }\right)$</td></tr><tr><td>First $\mathrm{k} = {50}$</td><td>38.0</td><td>95.8</td><td>8.7$\left( {{2.2} \times  }\right)$</td><td>59.8</td><td>$\left( {{1.2} \times  }\right)$</td><td>16</td><td>$\left( {{1.4} \times  }\right)$</td></tr><tr><td>Attn k=15</td><td>28.2</td><td>86.4</td><td>5.7(3.3x)</td><td>29.1</td><td>$\left( {{2.4} \times  }\right)$</td><td>4.8</td><td>(4.6x)</td></tr><tr><td>Attn k=50</td><td>38.6</td><td>97.1</td><td>8.6$\left( {{2.2} \times  }\right)$</td><td>62.2</td><td>$\left( {{1.1} \times  }\right)$</td><td>16</td><td>(1.4x)</td></tr><tr><td>IDF $\tau  = {3000}$</td><td>27.3</td><td>86.1</td><td>8.9(2.1x)</td><td>27.3</td><td>$\left( {{2.5} \times  }\right)$</td><td>5.4</td><td>(4.1x)</td></tr><tr><td>IDF $\tau  = {100}$</td><td>38.5</td><td>97.0</td><td>13.0$\left( {{1.5} \times  }\right)$</td><td>54.6</td><td>(1.3x)</td><td>15</td><td>$\left( {{1.5} \times  }\right)$</td></tr><tr><td>LeapMV, $\alpha  = {0.8}$</td><td>35.8</td><td>93.6</td><td>6.0$\left( {{3.2} \times  }\right)$</td><td>26.1</td><td>(2.6x)</td><td>4.9</td><td>(4.5x)</td></tr><tr><td>LeapMV, default</td><td>38.4</td><td>96.0</td><td>8.0$\left( {{2.4} \times  }\right)$</td><td>49.9</td><td>$\left( {{1.4} \times  }\right)$</td><td>12</td><td>(1.8x)</td></tr><tr><td>LeapMV, $\alpha  = {0.25}$</td><td>38.8</td><td>96.2</td><td>9.5$\left( {{2.0} \times  }\right)$</td><td>59.5</td><td>$\left( {{1.2} \times  }\right)$</td><td>16</td><td>$\left( {{1.4} \times  }\right)$</td></tr></table>

<table><tbody><tr><td rowspan="2">方法</td><td colspan="2">相关性</td><td colspan="3">延迟（毫秒）（加速比）</td><td colspan="2">空间</td></tr><tr><td colspan="2">前10平均倒数排名（MRR@10） 前1000召回率（R@1K）</td><td>图形处理器（GPU）</td><td>8线程</td><td>中央处理器（CPU）</td><td>吉字节（GB）</td><td>（比率）</td></tr><tr><td>火箭问答（RocketQA）</td><td>38.8</td><td>98.1</td><td>--</td><td>-</td><td>-</td><td>27</td><td>-</td></tr><tr><td>回溯掩码自编码器（RetroMAE）（无标题注释）</td><td>36.0</td><td>97.7</td><td>--</td><td>-</td><td>-</td><td>27</td><td>-</td></tr><tr><td>PLAID ColBERTv2</td><td>39.4</td><td>97.6</td><td>18.9$\left( {1 \times  }\right)$</td><td>69.1</td><td>$\left( {1 \times  }\right)$</td><td>22</td><td>$\left( {1 \times  }\right)$</td></tr><tr><td>阶段分块</td><td>30.0</td><td>89.6</td><td>11.5(1.6x)</td><td>56.5</td><td>$\left( {{1.2} \times  }\right)$</td><td>12</td><td>(1.8x)</td></tr><tr><td>句子分块</td><td>22.0</td><td>69.4</td><td>5.6(3.4x)</td><td>16.3</td><td>$\left( {{4.2} \times  }\right)$</td><td>2.1</td><td>(10.5x)</td></tr><tr><td>第一个 $\mathrm{k} = {15}$</td><td>31.7</td><td>84.7</td><td>5.9$\left( {{3.2} \times  }\right)$</td><td>30.6</td><td>$\left( {{2.3} \times  }\right)$</td><td>4.9</td><td>$\left( {{4.5} \times  }\right)$</td></tr><tr><td>第一个 $\mathrm{k} = {50}$</td><td>38.0</td><td>95.8</td><td>8.7$\left( {{2.2} \times  }\right)$</td><td>59.8</td><td>$\left( {{1.2} \times  }\right)$</td><td>16</td><td>$\left( {{1.4} \times  }\right)$</td></tr><tr><td>注意力机制 k=15</td><td>28.2</td><td>86.4</td><td>5.7(3.3x)</td><td>29.1</td><td>$\left( {{2.4} \times  }\right)$</td><td>4.8</td><td>(4.6x)</td></tr><tr><td>注意力机制 k=50</td><td>38.6</td><td>97.1</td><td>8.6$\left( {{2.2} \times  }\right)$</td><td>62.2</td><td>$\left( {{1.1} \times  }\right)$</td><td>16</td><td>(1.4x)</td></tr><tr><td>逆文档频率（IDF） $\tau  = {3000}$</td><td>27.3</td><td>86.1</td><td>8.9(2.1x)</td><td>27.3</td><td>$\left( {{2.5} \times  }\right)$</td><td>5.4</td><td>(4.1x)</td></tr><tr><td>逆文档频率（IDF） $\tau  = {100}$</td><td>38.5</td><td>97.0</td><td>13.0$\left( {{1.5} \times  }\right)$</td><td>54.6</td><td>(1.3x)</td><td>15</td><td>$\left( {{1.5} \times  }\right)$</td></tr><tr><td>跳跃多向量（LeapMV）, $\alpha  = {0.8}$</td><td>35.8</td><td>93.6</td><td>6.0$\left( {{3.2} \times  }\right)$</td><td>26.1</td><td>(2.6x)</td><td>4.9</td><td>(4.5x)</td></tr><tr><td>跳跃多向量（LeapMV）, 默认</td><td>38.4</td><td>96.0</td><td>8.0$\left( {{2.4} \times  }\right)$</td><td>49.9</td><td>$\left( {{1.4} \times  }\right)$</td><td>12</td><td>(1.8x)</td></tr><tr><td>跳跃多向量（LeapMV）, $\alpha  = {0.25}$</td><td>38.8</td><td>96.2</td><td>9.5$\left( {{2.0} \times  }\right)$</td><td>59.5</td><td>$\left( {{1.2} \times  }\right)$</td><td>16</td><td>$\left( {{1.4} \times  }\right)$</td></tr></tbody></table>

<!-- Media -->

### 4.3 Ablation studies

### 4.3 消融研究

## We present two ablation studies regarding model architecture design.

## 我们针对模型架构设计进行了两项消融研究。

Impact of window size choices. Table 6 shows the impact of the window size $c$ in Formula (2) on the MS MARCO passage,Dev set. Due to the cost of model training,we train for ${300}\mathrm{k}$ steps instead of ${400}\mathrm{k}$ steps for this study. The result shows that imposing a wider context window does not yield much performance gain. We suspect that document token encodings already contain sufficient contextualized information in MS MARCO. For simplicity, all evaluations including zero-shot retrieval use $c = 0$ .

窗口大小选择的影响。表6展示了公式(2)中的窗口大小 $c$ 对MS MARCO段落开发集的影响。由于模型训练成本的原因，在本研究中我们训练了 ${300}\mathrm{k}$ 步而非 ${400}\mathrm{k}$ 步。结果表明，采用更宽的上下文窗口并不会带来太多性能提升。我们推测，在MS MARCO中，文档标记编码已经包含了足够的上下文信息。为简单起见，包括零样本检索在内的所有评估均使用 $c = 0$。

<!-- Media -->

Table 6. Impact of window size choice $c$ on MS MARCO dev set.

表6. 窗口大小选择 $c$ 对MS MARCO开发集的影响。

<table><tr><td>c = 0</td><td>c = 2</td><td>c = 5</td></tr><tr><td>MRR@10 Recall@1K</td><td>MRR@10 Recall@1K</td><td>MRR@10 Recall@1K</td></tr><tr><td>38.396.0</td><td>38.296.2</td><td>38.295.8</td></tr></table>

<table><tbody><tr><td>c = 0</td><td>c = 2</td><td>c = 5</td></tr><tr><td>前10项平均倒数排名（MRR@10） 前1000项召回率（Recall@1K）</td><td>前10项平均倒数排名（MRR@10） 前1000项召回率（Recall@1K）</td><td>前10项平均倒数排名（MRR@10） 前1000项召回率（Recall@1K）</td></tr><tr><td>38.396.0</td><td>38.296.2</td><td>38.295.8</td></tr></tbody></table>

<!-- Media -->

Linear layer versus transformer in LeapMV. Figure 1 uses a linear layer to compute the pruning decision. Table 7 shows the impact of replacing a single linear layer with a more complex transformer encoder [27]. For the transformer encoder,hidden dimension $d$ is equal to the default ColBERT encoding of 128, 8 attention heads,and feed-forward layer dimension of ${4d}$ . Despite integrating this more complex module, both modules yield a similar relevance. For better indexing efficiency, we choose the linear layer with fewer parameters for LeapMV.

LeapMV中线性层与Transformer的对比。图1使用线性层来计算剪枝决策。表7展示了用更复杂的Transformer编码器 [27] 替换单个线性层的影响。对于Transformer编码器，隐藏维度 $d$ 等于默认的ColBERT编码维度128，有8个注意力头，前馈层维度为 ${4d}$ 。尽管集成了这个更复杂的模块，但两个模块产生的相关性相近。为了提高索引效率，我们为LeapMV选择参数更少的线性层。

<!-- Media -->

Table 7. Choice of linear layer versus transformer network on MS MARCO dev set for computing pruning decisions in LeapMV architecture.

表7. 在LeapMV架构中计算剪枝决策时，MS MARCO开发集上线性层与Transformer网络的选择。

<table><tr><td>Linear Layer</td><td>Transformer</td></tr><tr><td>MRR@10 Recall@1K</td><td>MRR@10 Recall@1K</td></tr><tr><td>38.496.0</td><td>38.495.9</td></tr></table>

<table><tbody><tr><td>线性层（Linear Layer）</td><td>变换器（Transformer）</td></tr><tr><td>前10平均倒数排名（MRR@10） 前1000召回率（Recall@1K）</td><td>前10平均倒数排名（MRR@10） 前1000召回率（Recall@1K）</td></tr><tr><td>38.496.0</td><td>38.495.9</td></tr></tbody></table>

<!-- Media -->

## 5 Concluding Remarks

## 5 结论

We introduce LeapMV, a learned token pruning scheme designed for efficient multi-vector dense retrieval with a moderate relevance tradeoff to enhance efficiency. LeapMV makes a context-aware pruning decision for each document and is more adaptive than existing baselines. The degree of pruning in LeapMV can be adjusted with parameter $\alpha$ to meet the targeted space reduction goals, resulting in a near-linear reduction in both GPU and CPU latency.

我们介绍了LeapMV，这是一种经过学习的词元剪枝方案，旨在以适度的相关性权衡实现高效的多向量密集检索，从而提高效率。LeapMV会为每个文档做出上下文感知的剪枝决策，比现有的基线方法更具适应性。LeapMV的剪枝程度可以通过参数$\alpha$进行调整，以实现目标空间缩减目标，从而使GPU和CPU延迟近乎线性降低。

Our evaluation shows the advantage of LeapMV over the previous baselines in adapting to various space reduction goals. For instance, Table 1 shows that under the same relevance accuracy budget,LeapMV consumes up to ${3.3} \times$ less space and is up to ${2.8} \times$ faster for LoTTE than top- $k$ token pruning and IDF methods. Compared to the optimized PLAID ColBERTv2, LeapMV achieves $2 \times$ space saving and ${2.5} \times$ GPU latency speedup while incurring less than $2\%$ relevance loss on LoTTE forum queries. LeapMV yields more space and time saving when accuracy can be relaxed further while maintaining performance significantly better than its competitors under tight space budgets. Our current evaluation follows the default parameter setting in PLAID with a retrieval depth of 1000 , and a future study is to investigate the impact of varying these settings.

我们的评估显示，LeapMV在适应各种空间缩减目标方面优于先前的基线方法。例如，表1显示，在相同的相关性准确率预算下，与前$k$词元剪枝和逆文档频率（IDF）方法相比，LeapMV在LoTTE数据集上最多可节省${3.3} \times$的空间，速度最多可提高${2.8} \times$。与优化后的PLAID ColBERTv2相比，LeapMV在LoTTE论坛查询上实现了$2 \times$的空间节省和${2.5} \times$的GPU延迟加速，同时相关性损失小于$2\%$。当可以进一步放宽准确率要求时，LeapMV能节省更多的空间和时间，并且在严格的空间预算下，其性能仍显著优于竞争对手。我们目前的评估遵循PLAID中的默认参数设置，检索深度为1000，未来的研究将探讨改变这些设置的影响。

Acknowledgments. We thank anonymous referees for their valuable comments. The major portion of this work was supported by Amazon. It was partially supported by U.S. NSF IIS-2225942 and its ACCESS program in using its computing resource. Any opinions, findings, conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of Amazon or the NSF.

致谢。我们感谢匿名评审人员提出的宝贵意见。这项工作的主要部分由亚马逊公司资助。它还得到了美国国家科学基金会（NSF）IIS - 2225942项目及其ACCESS计划在计算资源使用方面的部分支持。本材料中表达的任何观点、研究结果、结论或建议均为作者本人的观点，不一定反映亚马逊公司或美国国家科学基金会的观点。

## References

## 参考文献

1. Acquavia, A., Macdonald, C., Tonellotto, N.: Static pruning for multi-representation dense retrieval. In: Proceedings of the ACM Symposium on Document Engineering 2023. DocEng '23, Association for Computing Machinery, New York, NY, USA (2023). https://doi.org/10.1145/3573128.3604896, https: //doi.org/10.1145/3573128.3604896

1. Acquavia, A., Macdonald, C., Tonellotto, N.：多表示密集检索的静态剪枝。见：《2023年ACM文档工程研讨会论文集》。DocEng '23，美国计算机协会，纽约，纽约州，美国（2023）。https://doi.org/10.1145/3573128.3604896，https://doi.org/10.1145/3573128.3604896

2. Bird, S., Loper, E.: NLTK: The natural language toolkit. In: Proceedings of the ACL Interactive Poster and Demonstration Sessions. pp. 214-217. Association for Computational Linguistics, Barcelona, Spain (Jul 2004), https://aclanthology.org/P04-3031

2. Bird, S., Loper, E.：NLTK：自然语言工具包。见：《ACL交互式海报与演示会议论文集》。第214 - 217页。计算语言学协会，西班牙巴塞罗那（2004年7月），https://aclanthology.org/P04 - 3031

3. Campos, D.F., Nguyen, T., Rosenberg, M., Song, X., Gao, J., Tiwary, S., Ma-jumder, R., Deng, L., Mitra, B.: Ms marco: A human generated machine reading comprehension dataset. ArXiv abs/1611.09268 (2016)

3. Campos, D.F., Nguyen, T., Rosenberg, M., Song, X., Gao, J., Tiwary, S., Ma - jumder, R., Deng, L., Mitra, B.：Ms marco：一个人工生成的机器阅读理解数据集。预印本arXiv:1611.09268（2016）

4. colbertv1.9: https://huggingface.co/colbert-ir/colbertv1.9 (2022)

4. colbertv1.9：https://huggingface.co/colbert - ir/colbertv1.9（2022）

5. Devlin, J., Chang, M.W., Lee, K., Toutanova, K.: Bert: Pre-training of deep bidirectional transformers for language understanding. In: NAACL (2019)

5. Devlin, J., Chang, M.W., Lee, K., Toutanova, K.：BERT：用于语言理解的深度双向Transformer预训练。见：《北美计算语言学协会会议（NAACL）》（2019）

6. Engels, J., Coleman, B., Lakshman, V., Shrivastava, A.: DESSERT: An efficient algorithm for vector set search with vector set queries. In: Thirty-seventh Conference on Neural Information Processing Systems (2023), https://openreview.net/forum?id=kXfrlWXLwH

6. Engels, J., Coleman, B., Lakshman, V., Shrivastava, A.：DESSERT：一种用于向量集查询的向量集搜索高效算法。见：《第三十七届神经信息处理系统会议》（2023），https://openreview.net/forum?id = kXfrlWXLwH

7. Formal, T., Lassance, C., Piwowarski, B., Clinchant, S.: From distillation to hard negative sampling: Making sparse neural IR models more effective. SIGIR (2022)

7. Formal, T., Lassance, C., Piwowarski, B., Clinchant, S.：从蒸馏到难负样本采样：使稀疏神经信息检索模型更有效。《信息检索研究与发展会议（SIGIR）》（2022）

8. Formal, T., Piwowarski, B., Clinchant, S.: SPLADE: Sparse lexical and expansion model for first stage ranking. SIGIR (2021)

8. Formal, T., Piwowarski, B., Clinchant, S.：SPLADE：用于第一阶段排序的稀疏词法和扩展模型。《信息检索研究与发展会议（SIGIR）》（2021）

9. Gumbel, E.: Statistical Theory of Extreme Values and Some Practical Applications: A Series of Lectures. Applied mathematics series, U.S. Government Printing Office (1954)

9. Gumbel, E.：《极值统计理论及其一些实际应用：系列讲座》。应用数学系列，美国政府印刷局（1954）

10. Jang, E., Gu, S.S., Poole, B.: Categorical reparameterization with gumbel-softmax. ICLR (2017)

10. Jang, E., Gu, S.S., Poole, B.：使用Gumbel - Softmax进行类别重参数化。《国际学习表征会议（ICLR）》（2017）

11. Khashabi, D., Ng, A., Khot, T., Sabharwal, A., Hajishirzi, H., Callison-Burch, C.: Gooaq: Open question answering with diverse answer types (2021), https: //arxiv.org/abs/2104.08727

11. Khashabi, D., Ng, A., Khot, T., Sabharwal, A., Hajishirzi, H., Callison - Burch, C.：Gooaq：具有多种答案类型的开放问答（2021），https://arxiv.org/abs/2104.08727

12. Khattab, O., Zaharia, M.A.: Colbert: Efficient and effective passage search via contextualized late interaction over bert. SIGIR (2020)

12. 哈塔布（Khattab），O.，扎哈里亚（Zaharia），M.A.：《科尔伯特（Colbert）：通过基于BERT的上下文后期交互实现高效有效的段落搜索》。SIGIR（2020）

13. Lassance, C., Clinchant, S.: The tale of two msmarco - and their unfair comparisons. In: Proceed. of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval. p. 2431-2435. SIGIR '23, ACM, New York, NY, USA (2023)

13. 拉桑斯（Lassance），C.，克林尚（Clinchant），S.：《两个MS MARCO数据集的故事——以及它们之间的不公平比较》。收录于《第46届ACM SIGIR信息检索研究与发展国际会议论文集》。第2431 - 2435页。SIGIR '23，美国计算机协会，美国纽约州纽约市（2023）

14. Lassance, C., Maachou, M., Park, J., Clinchant, S.: Learned token pruning in contextualized late interaction over bert (colbert). In: Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval. p. 2232-2236. SIGIR '22, Association for Computing Machinery, New York, NY, USA (2022). https://doi.org/10.1145/3477495.3531835, https: //doi.org/10.1145/3477495.3531835

14. 拉桑斯（Lassance），C.，马乔（Maachou），M.，朴（Park），J.，克林尚（Clinchant），S.：《基于BERT（科尔伯特）的上下文后期交互中的学习型词元剪枝》。收录于《第45届ACM SIGIR信息检索研究与发展国际会议论文集》。第2232 - 2236页。SIGIR '22，美国计算机协会，美国纽约州纽约市（2022）。https://doi.org/10.1145/3477495.3531835，https://doi.org/10.1145/3477495.3531835

15. Lee, J., Dai, Z., Duddu, S.M.K., Lei, T., Naim, I., Chang, M.W., Zhao, V.Y.: Rethinking the role of token retrieval in multi-vector retrieval. In: Thirty-seventh Conference on Neural Information Processing Systems (2023), https://openreview.net/forum?id=ZQzmOZ47jz

15. 李（Lee），J.，戴（Dai），Z.，杜杜（Duddu），S.M.K.，雷（Lei），T.，奈姆（Naim），I.，张（Chang），M.W.，赵（Zhao），V.Y.：《重新思考词元检索在多向量检索中的作用》。收录于《第三十七届神经信息处理系统大会》（2023），https://openreview.net/forum?id=ZQzmOZ47jz

16. Li, M., Lin, S.C., Oguz, B., Ghoshal, A., Lin, J., Mehdad, Y., tau Yih, W., Chen, X.: CITADEL: Conditional token interaction via dynamic lexical routing for efficient and effective multi-vector retrieval. pp. 11891-11907 (2023)

16. 李（Li），M.，林（Lin），S.C.，奥古兹（Oguz），B.，戈沙尔（Ghoshal），A.，林（Lin），J.，梅赫达德（Mehdad），Y.，易（Yih），W.，陈（Chen），X.：《CITADEL：通过动态词法路由实现条件词元交互以进行高效有效的多向量检索》。第11891 - 11907页（2023）

17. Liu, Z., Xiao, S., Shao, Y., Cao, Z.: RetroMAE-2: Duplex masked auto-encoder for pre-training retrieval-oriented language models. In: Rogers, A., Boyd-Graber, J., Okazaki, N. (eds.) Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 2635-2648. Association for Computational Linguistics, Toronto, Canada (Jul 2023)

17. 刘（Liu），Z.，肖（Xiao），S.，邵（Shao），Y.，曹（Cao），Z.：《RetroMAE - 2：用于预训练面向检索的语言模型的双工掩码自编码器》。收录于罗杰斯（Rogers），A.，博伊德 - 格雷伯（Boyd - Graber），J.，冈崎（Okazaki），N.（编）《第61届计算语言学协会年会论文集（第1卷：长论文）》。第2635 - 2648页。计算语言学协会，加拿大多伦多（2023年7月）

18. Maddison, C.J., Mnih, A., Teh, Y.W.: The concrete distribution: A continuous relaxation of discrete random variables. ICLR (2017)

18. 麦迪逊（Maddison），C.J.，米尼（Mnih），A.， Teh，Y.W.：《具体分布：离散随机变量的连续松弛》。ICLR（2017）

19. MiniLM-L-6-v2: https://huggingface.co/cross-encoder/ms-marco-minilm-l-6-v2 (2022)

19. MiniLM - L - 6 - v2：https://huggingface.co/cross - encoder/ms - marco - minilm - l - 6 - v2（2022）

20. Nardini, F.M., Rulli, C., Venturini, R.: Efficient multi-vector dense retrieval with bit vectors. In: Goharian, N., Tonellotto, N., He, Y., Lipani, A., McDonald, G., Macdonald, C., Ounis, I. (eds.) Advances in Information Retrieval (ECIR 2024). pp. 3-17. Springer Nature Switzerland, Cham (2024)

20. 纳尔迪尼（Nardini），F.M.，鲁利（Rulli），C.，文图里尼（Venturini），R.：《使用位向量实现高效的多向量密集检索》。收录于戈哈里安（Goharian），N.，托内洛托（Tonellotto），N.，何（He），Y.，利帕尼（Lipani），A.，麦克唐纳（McDonald），G.，麦克唐纳德（Macdonald），C.，奥尼斯（Ounis），I.（编）《信息检索进展（ECIR 2024）》。第3 - 17页。施普林格自然瑞士出版社，尚姆（2024）

21. Ren, R., Qu, Y., Liu, J., Zhao, W.X., She, Q., Wu, H., Wang, H., Wen, J.R.: RocketQAv2: A joint training method for dense passage retrieval and passage re-ranking. In: Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. pp. 2825-2835. ACM, Online and Punta Cana, Dominican Republic (Nov 2021)

21. 任（Ren），R.，曲（Qu），Y.，刘（Liu），J.，赵（Zhao），W.X.，佘（She），Q.，吴（Wu），H.，王（Wang），H.，文（Wen），J.R.：《RocketQAv2：一种用于密集段落检索和段落重排序的联合训练方法》。收录于《2021年自然语言处理经验方法会议论文集》。第2825 - 2835页。美国计算机协会，线上及多米尼加共和国蓬塔卡纳（2021年11月）

22. Sang, E.F.T.K., Buchholz, S.: Introduction to the conll-2000 shared task: Chunking (2000), https://arxiv.org/abs/cs/0009008

22. 桑（Sang），E.F.T.K.，布赫霍尔茨（Buchholz），S.：《CoNLL - 2000共享任务介绍：组块分析》（2000），https://arxiv.org/abs/cs/0009008

23. Santhanam, K., Khattab, O., Saad-Falcon, J., Potts, C., Zaharia, M.A.: Col-bertv2: Effective and efficient retrieval via lightweight late interaction. ArXiv abs/2112.01488 (Dec 2021)

23. 桑塔南（Santhanam），K.，哈塔布（Khattab），O.，萨德 - 法尔孔（Saad - Falcon），J.，波茨（Potts），C.，扎哈里亚（Zaharia），M.A.：《科尔伯特v2（Col - bertv2）：通过轻量级后期交互实现高效有效的检索》。arXiv预印本abs/2112.01488（2021年12月）

24. Santhanam, K., Khattab, O., Potts, C., Zaharia, M.: Plaid: An efficient engine for late interaction retrieval. In: Proceedings of the 31st ACM International Conference on Information & Knowledge Management. p. 1747-1756. CIKM '22 (2022)

24. 桑塔南（Santhanam），K.，哈塔布（Khattab），O.，波茨（Potts），C.，扎哈里亚（Zaharia），M.：《Plaid：一种用于后期交互检索的高效引擎》。收录于《第31届ACM信息与知识管理国际会议论文集》。第1747 - 1756页。CIKM '22（2022）

25. Sciavolino, C., Zhong, Z., Lee, J., Chen, D.: Simple entity-centric questions challenge dense retrievers. In: Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. pp. 6138-6148. ACL (Nov 2021)

25. 夏沃利诺（Sciavolino），C.，钟（Zhong），Z.，李（Lee），J.，陈（Chen），D.：《简单的以实体为中心的问题对密集检索器的挑战》。收录于《2021年自然语言处理经验方法会议论文集》。第6138 - 6148页。计算语言学协会（2021年11月）

26. Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., Gurevych, I.: BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. In: NeurIPS (2021)

26. 塔库尔（Thakur），N.；赖默斯（Reimers），N.；吕克莱（Rücklé），A.；斯里瓦斯塔瓦（Srivastava），A.；古雷维奇（Gurevych），I.：BEIR：信息检索模型零样本评估的异构基准。见：神经信息处理系统大会（NeurIPS）（2021）

27. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.: Attention is all you need (2023), https://arxiv.org/abs/1706.03762

27. 瓦斯瓦尼（Vaswani），A.；沙泽尔（Shazeer），N.；帕尔马尔（Parmar），N.；乌兹科雷特（Uszkoreit），J.；琼斯（Jones），L.；戈麦斯（Gomez），A.N.；凯泽（Kaiser），L.；波洛苏金（Polosukhin），I.：注意力就是你所需要的一切（2023），https://arxiv.org/abs/1706.03762

28. Yang, Y., Qiao, Y., He, S., Yang, T.: Weighted kl-divergence for document ranking model refinement. arxiv and SIGIR 2024 (2024)

28. 杨（Yang），Y.；乔（Qiao），Y.；何（He），S.；杨（Yang），T.：用于文档排序模型优化的加权KL散度。预印本平台（arxiv）和国际信息检索研究与发展会议（SIGIR）2024（2024）