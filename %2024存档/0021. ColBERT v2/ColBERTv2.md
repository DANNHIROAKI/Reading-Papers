# ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction

# ColBERTv2：通过轻量级后期交互实现高效有效的检索

Keshav Santhanam*

凯沙夫·桑塔纳姆（Keshav Santhanam）*

Stanford University

斯坦福大学

Omar Khattab*

奥马尔·哈塔卜（Omar Khattab）*

Stanford University

斯坦福大学

Jon Saad-Falcon

乔恩·萨德 - 法尔孔（Jon Saad - Falcon）

Georgia Institute of Technology

佐治亚理工学院

Christopher Potts

克里斯托弗·波茨（Christopher Potts）

Stanford University

斯坦福大学

Matei Zaharia

马泰·扎哈里亚（Matei Zaharia）

Stanford University

斯坦福大学

## Abstract

## 摘要

Neural information retrieval (IR) has greatly advanced search and other knowledge-intensive language tasks. While many neural IR methods encode queries and documents into single-vector representations, late interaction models produce multi-vector representations at the granularity of each token and decompose relevance modeling into scalable token-level computations. This decomposition has been shown to make late interaction more effective, but it inflates the space footprint of these models by an order of magnitude. In this work, we introduce ColBERTv2, a retriever that couples an aggressive residual compression mechanism with a denoised supervision strategy to simultaneously improve the quality and space footprint of late interaction. We evaluate ColBERTv2 across a wide range of benchmarks, establishing state-of-the-art quality within and outside the training domain while reducing the space footprint of late interaction models by $6 - {10} \times$ .

神经信息检索（Neural information retrieval，IR）极大地推动了搜索及其他知识密集型语言任务的发展。虽然许多神经信息检索方法将查询和文档编码为单向量表示，但后期交互模型以每个标记为粒度生成多向量表示，并将相关性建模分解为可扩展的标记级计算。这种分解已被证明能使后期交互更有效，但会使这些模型的空间占用增加一个数量级。在这项工作中，我们引入了ColBERTv2，这是一种检索器，它将积极的残差压缩机制与去噪监督策略相结合，以同时提高后期交互的质量和减少空间占用。我们在广泛的基准测试中评估了ColBERTv2，在训练领域内外均达到了最先进的质量水平，同时将后期交互模型的空间占用减少了$6 - {10} \times$。

## 1 Introduction

## 1 引言

Neural information retrieval (IR) has quickly dominated the search landscape over the past 2-3 years, dramatically advancing not only passage and document search (Nogueira and Cho, 2019) but also many knowledge-intensive NLP tasks like open-domain question answering (Guu et al., 2020), multi-hop claim verification (Khattab et al., 2021a), and open-ended generation (Paranjape et al., 2022).

在过去两到三年里，神经信息检索（Neural information retrieval，IR）迅速主导了搜索领域，不仅极大地推动了段落和文档搜索（诺盖拉和赵，2019），还推动了许多知识密集型自然语言处理任务，如开放域问答（古等人，2020）、多跳声明验证（哈塔卜等人，2021a）和开放式生成（帕兰贾佩等人，2022）。

Many neural IR methods follow a single-vector similarity paradigm: a pretrained language model is used to encode each query and each document into a single high-dimensional vector, and relevance is modeled as a simple dot product between both vectors. An alternative is late interaction, introduced in ColBERT (Khattab and Zaharia, 2020), where queries and documents are encoded at a finer-granularity into multi-vector representations, and relevance is estimated using rich yet scalable interactions between these two sets of vectors. ColBERT produces an embedding for every token in the query (and document) and models relevance as the sum of maximum similarities between each query vector and all vectors in the document.

许多神经信息检索方法遵循单向量相似性范式：使用预训练语言模型将每个查询和每个文档编码为单个高维向量，并将相关性建模为两个向量之间的简单点积。另一种方法是后期交互，由ColBERT（哈塔卜和扎哈里亚，2020）提出，在这种方法中，查询和文档以更细的粒度编码为多向量表示，并使用这两组向量之间丰富且可扩展的交互来估计相关性。ColBERT为查询（和文档）中的每个标记生成一个嵌入，并将相关性建模为每个查询向量与文档中所有向量之间的最大相似度之和。

By decomposing relevance modeling into token-level computations, late interaction aims to reduce the burden on the encoder: whereas single-vector models must capture complex query-document relationships within one dot product, late interaction encodes meaning at the level of tokens and delegates query-document matching to the interaction mechanism. This added expressivity comes at a cost: existing late interaction systems impose an order-of-magnitude larger space footprint than single-vector models, as they must store billions of small vectors for Web-scale collections. Considering this challenge, it might seem more fruitful to focus instead on addressing the fragility of single-vector models (Menon et al., 2022) by introducing new supervision paradigms for negative mining (Xiong et al., 2020), pretraining (Gao and Callan, 2021), and distillation (Qu et al., 2021). Indeed, recent single-vector models with highly-tuned supervision strategies (Ren et al., 2021b; Formal et al., 2021a) sometimes perform on-par or even better than "vanilla" late interaction models, and it is not necessarily clear whether late interaction architectures-with their fixed token-level inductive biases-admit similarly large gains from improved supervision.

通过将相关性建模分解为Token级别的计算，后期交互旨在减轻编码器的负担：单向量模型必须在一次点积运算中捕捉复杂的查询 - 文档关系，而后期交互则在Token级别对语义进行编码，并将查询 - 文档匹配任务委托给交互机制。这种增强的表达能力是有代价的：现有的后期交互系统比单向量模型占用的空间要大一个数量级，因为对于网络规模的文档集合，它们必须存储数十亿个小向量。考虑到这一挑战，通过引入用于负样本挖掘（Xiong等人，2020）、预训练（Gao和Callan，2021）和知识蒸馏（Qu等人，2021）的新监督范式来解决单向量模型的脆弱性问题（Menon等人，2022），似乎更有成效。事实上，最近采用高度调优监督策略的单向量模型（Ren等人，2021b；Formal等人，2021a）有时表现与“普通”后期交互模型相当，甚至更好，而且尚不清楚具有固定Token级别归纳偏置的后期交互架构是否能从改进的监督中获得同样大的收益。

In this work, we show that late interaction retrievers naturally produce lightweight token representations that are amenable to efficient storage off-the-shelf and that they can benefit drastically from denoised supervision. We couple those in ColBERTv2, ${}^{1}$ a new late-interaction retriever that employs a simple combination of distillation from a cross-encoder and hard-negative mining (§3.2) to boost quality beyond any existing method, and then uses a residual compression mechanism (§3.3) to reduce the space footprint of late interaction by $6 - {10} \times$ while preserving quality. As a result,Col-BERTv2 establishes state-of-the-art retrieval quality both within and outside its training domain with a competitive space footprint with typical single-vector models.

在这项工作中，我们表明后期交互检索器自然会产生轻量级的Token表示，这些表示适合使用现成的方法进行高效存储，并且它们可以从去噪监督中大幅受益。我们在ColBERTv2中结合了这些优势，${}^{1}$ 这是一种新的后期交互检索器，它采用了从交叉编码器进行知识蒸馏和难负样本挖掘的简单组合方法（§3.2）来提升性能，使其超越任何现有方法，然后使用残差压缩机制（§3.3）在保持性能的同时将后期交互的空间占用减少 $6 - {10} \times$。因此，Col - BERTv2在训练领域内外都实现了最先进的检索性能，并且在空间占用方面与典型的单向量模型具有竞争力。

---

<!-- Footnote -->

${}^{1}$ Code,models,and LoTTE data are maintained at https: //github.com/stanford-futuredata/ColBERT

${}^{1}$ 代码、模型和LoTTE数据维护在https://github.com/stanford - futuredata/ColBERT

*Equal contribution.

*同等贡献。

<!-- Footnote -->

---

When trained on MS MARCO Passage Ranking, ColBERTv2 achieves the highest MRR@10 of any standalone retriever. In addition to in-domain quality, we seek a retriever that generalizes "zero-shot" to domain-specific corpora and long-tail topics, ones that are often under-represented in large public training sets. To this end, we evaluate Col-BERTv2 on a wide array of out-of-domain benchmarks. These include three Wikipedia Open-QA retrieval tests and 13 diverse retrieval and semantic-similarity tasks from BEIR (Thakur et al., 2021). In addition, we introduce a new benchmark, dubbed LoTTE, for Long-Tail Topic-stratified Evaluation for IR that features 12 domain-specific search tests, spanning StackExchange communities and using queries from GooAQ (Khashabi et al., 2021). LoTTE focuses on relatively long-tail topics in its passages, unlike the Open-QA tests and many of the BEIR tasks, and evaluates models on their capacity to answer natural search queries with a practical intent, unlike many of BEIR's semantic-similarity tasks. On 22 of 28 out-of-domain tests, ColBERTv2 achieves the highest quality, outperforming the next best retriever by up to $8\%$ relative gain, while using its compressed representations.

在MS MARCO段落排序数据集上进行训练时，ColBERTv2实现了所有独立检索器中最高的MRR@10。除了域内性能外，我们还希望找到一种能够“零样本”泛化到特定领域语料库和长尾主题的检索器，这些主题在大型公共训练集中往往代表性不足。为此，我们在广泛的域外基准测试中评估了Col - BERTv2。这些测试包括三个维基百科开放问答检索测试以及来自BEIR的13个不同的检索和语义相似度任务（Thakur等人，2021）。此外，我们还引入了一个新的基准测试，称为LoTTE，用于信息检索的长尾主题分层评估，它包含12个特定领域的搜索测试，涵盖StackExchange社区并使用来自GooAQ的查询（Khashabi等人，2021）。与开放问答测试和许多BEIR任务不同，LoTTE关注其段落中的相对长尾主题，并且与许多BEIR的语义相似度任务不同，它评估模型回答具有实际意图的自然搜索查询的能力。在28个域外测试中的22个测试中，ColBERTv2使用其压缩表示实现了最高性能，相对于次优检索器的相对增益高达 $8\%$。

This work makes the following contributions:

这项工作做出了以下贡献：

1. We propose ColBERTv2, a retriever that combines denoised supervision and residual compression, leveraging the token-level decomposition of late interaction to achieve high robustness with a reduced space footprint.

1. 我们提出了ColBERTv2，这是一种结合了去噪监督和残差压缩的检索器，利用后期交互的Token级别分解，在减少空间占用的同时实现了高鲁棒性。

2. We introduce LoTTE, a new resource for out-of-domain evaluation of retrievers. LoTTE focuses on natural information-seeking queries over long-tail topics, an important yet understudied application space.

2. 我们引入了LoTTE，这是一种用于检索器域外评估的新资源。LoTTE关注针对长尾主题的自然信息查询，这是一个重要但研究不足的应用领域。

3. We evaluate ColBERTv2 across a wide range of settings, establishing state-of-the-art quality within and outside the training domain.

3. 我们在广泛的设置中评估了ColBERTv2，在训练领域内外都确立了最先进的性能。

## 2 Background & Related Work

## 2 背景与相关工作

### 2.1 Token-Decomposed Scoring in Neural IR

### 2.1 神经信息检索中的Token分解评分

Many neural IR approaches encode passages as a single high-dimensional vector, trading off the higher quality of cross-encoders for improved efficiency and scalability (Karpukhin et al., 2020; Xiong et al., 2020; Qu et al., 2021). ColBERT's (Khattab and Zaharia, 2020) late interaction paradigm addresses this tradeoff by computing multi-vector embeddings and using a scalable "MaxSim" operator for retrieval. Several other systems leverage multi-vector representations, including Poly-encoders (Humeau et al., 2020), PreTTR (MacAvaney et al., 2020), and MORES (Gao et al., 2020), but these target attention-based re-ranking as opposed to ColBERT's scalable MaxSim end-to-end retrieval.

许多神经信息检索（IR）方法将段落编码为单个高维向量，以牺牲交叉编码器的高质量为代价，换取更高的效率和可扩展性（Karpukhin等人，2020年；Xiong等人，2020年；Qu等人，2021年）。ColBERT（Khattab和Zaharia，2020年）的后期交互范式通过计算多向量嵌入并使用可扩展的“MaxSim”算子进行检索来解决这一权衡问题。其他几个系统也利用了多向量表示，包括Poly编码器（Humeau等人，2020年）、PreTTR（MacAvaney等人，2020年）和MORES（Gao等人，2020年），但这些系统针对的是基于注意力的重排序，而不是ColBERT可扩展的MaxSim端到端检索。

ME-BERT (Luan et al., 2021) generates token-level document embeddings similar to ColBERT, but retains a single embedding vector for queries. COIL (Gao et al., 2021) also generates token-level document embeddings, but the token interactions are restricted to lexical matching between query and document terms. uniCOIL (Lin and Ma, 2021) limits the token embedding vectors of COIL to a single dimension, reducing them to scalar weights that extend models like DeepCT (Dai and Callan, 2020) and DeepImpact (Mallia et al., 2021). To produce scalar weights, SPLADE (Formal et al., 2021b) and SPLADEv2 (Formal et al., 2021a) produce a sparse vocabulary-level vector that retains the term-level decomposition of late interaction while simplifying the storage into one dimension per token. The SPLADE family also piggybacks on the language modeling capacity acquired by BERT during pretraining. SPLADEv2 has been shown to be highly effective, within and across domains, and it is a central point of comparison in the experiments we report on in this paper.

ME - BERT（Luan等人，2021年）生成类似于ColBERT的词元级文档嵌入，但为查询保留单个嵌入向量。COIL（Gao等人，2021年）也生成词元级文档嵌入，但词元交互仅限于查询和文档术语之间的词法匹配。uniCOIL（Lin和Ma，2021年）将COIL的词元嵌入向量限制为一维，将其简化为标量权重，扩展了像DeepCT（Dai和Callan，2020年）和DeepImpact（Mallia等人，2021年）这样的模型。为了生成标量权重，SPLADE（Formal等人，2021b）和SPLADEv2（Formal等人，2021a）生成一个稀疏的词汇级向量，在简化每个词元存储为一维的同时，保留了后期交互的词级分解。SPLADE系列还借助了BERT在预训练期间获得的语言建模能力。SPLADEv2已被证明在域内和跨域方面都非常有效，并且是本文实验中的一个核心比较点。

### 2.2 Vector Compression for Neural IR

### 2.2 神经信息检索的向量压缩

There has been a surge of recent interest in compressing representations for IR. Izacard et al. (2020) explore dimension reduction, product quantization (PQ), and passage filtering for single-vector retrievers. BPR (Yamada et al., 2021a) learns to directly hash embeddings to binary codes using a differentiable tanh function. JPQ (Zhan et al., 2021a) and its extension, RepCONC (Zhan et al., 2022), use PQ to compress embeddings, and jointly train the query encoder along with the centroids produced by $\mathrm{{PQ}}$ via a ranking-oriented loss.

最近，人们对信息检索表示的压缩产生了浓厚的兴趣。Izacard等人（2020年）探索了单向量检索器的降维、乘积量化（PQ）和段落过滤。BPR（Yamada等人，2021a）学习使用可微的双曲正切函数将嵌入直接哈希为二进制代码。JPQ（Zhan等人，2021a）及其扩展RepCONC（Zhan等人，2022年）使用PQ压缩嵌入，并通过面向排序的损失函数联合训练查询编码器和$\mathrm{{PQ}}$生成的质心。

SDR (Cohen et al., 2021) uses an autoencoder to reduce the dimensionality of the contextual embed-dings used for attention-based re-ranking and then applies a quantization scheme for further compression. DensePhrases (Lee et al., 2021a) is a system for Open-QA that relies on a multi-vector encoding of passages, though its search is conducted at the level of individual vectors and not aggregated with late interaction. Very recently, Lee et al. (2021b) propose a quantization-aware finetuning method based on PQ to reduce the space footprint of DensePhrases. While DensePhrases is effective at Open-QA, its retrieval quality-as measured by top-20 retrieval accuracy on NaturalQuestions and TriviaQA—is competitive with DPR (Karpukhin et al., 2020) and considerably less effective than ColBERT (Khattab et al., 2021b).

SDR（Cohen等人，2021年）使用自动编码器降低用于基于注意力的重排序的上下文嵌入的维度，然后应用量化方案进行进一步压缩。DensePhrases（Lee等人，2021a）是一个开放问答（Open - QA）系统，它依赖于段落的多向量编码，不过其搜索是在单个向量级别进行的，而不是通过后期交互进行聚合。最近，Lee等人（2021b）提出了一种基于PQ的量化感知微调方法，以减少DensePhrases的空间占用。虽然DensePhrases在开放问答方面很有效，但其检索质量（以NaturalQuestions和TriviaQA上的前20名检索准确率衡量）与DPR（Karpukhin等人，2020年）相当，并且远不如ColBERT（Khattab等人，2021b）有效。

In this work, we focus on late-interaction retrieval and investigate compression using a residual compression approach that can be applied off-the-shelf to late interaction models, without special training. We show in Appendix A that ColBERT's representations naturally lend themselves to residual compression. Techniques in the family of residual compression are well-studied (Barnes et al., 1996) and have previously been applied across several domains, including approximate nearest neighbor search (Wei et al., 2014; Ai et al., 2017), neural network parameter and activation quantization (Li et al., 2021b,a), and distributed deep learning (Chen et al., 2018; Liu et al., 2020). To the best of our knowledge, ColBERTv2 is the first approach to use residual compression for scalable neural IR.

在这项工作中，我们专注于后期交互检索，并研究使用残差压缩方法进行压缩，该方法可以直接应用于后期交互模型，无需特殊训练。我们在附录A中表明，ColBERT的表示自然适用于残差压缩。残差压缩家族的技术已经得到了深入研究（Barnes等人，1996年），并且此前已应用于多个领域，包括近似最近邻搜索（Wei等人，2014年；Ai等人，2017年）、神经网络参数和激活量化（Li等人，2021b，a）以及分布式深度学习（Chen等人，2018年；Liu等人，2020年）。据我们所知，ColBERTv2是第一个将残差压缩用于可扩展神经信息检索的方法。

### 2.3 Improving the Quality of Single-Vector Representations

### 2.3 提高单向量表示的质量

Instead of compressing multi-vector representations as we do, much recent work has focused on improving the quality of single-vector models, which are often very sensitive to the specifics of supervision. This line of work can be decomposed into three directions: (1) distillation of more expressive architectures (Hofstätter et al., 2020; Lin et al., 2020) including explicit denoising (Qu et al., 2021; Ren et al., 2021b), (2) hard negative sampling (Xiong et al., 2020; Zhan et al., 2020a, 2021b), and (3) improved pretraining (Gao and Callan, 2021; Oğuz et al., 2021). We adopt similar techniques to (1) and (2) for ColBERTv2's multi-vector representations (see §3.2).

与我们压缩多向量表示的做法不同，近期的许多工作都聚焦于提升单向量模型的质量，这类模型通常对监督的具体细节非常敏感。这方面的工作可分为三个方向：（1）蒸馏更具表现力的架构（霍夫施泰特等人，2020年；林等人，2020年），包括显式去噪（曲等人，2021年；任等人，2021b）；（2）难负样本采样（熊等人，2020年；詹等人，2020a、2021b）；（3）改进预训练（高和卡兰，2021年；奥古兹等人，2021年）。我们针对ColBERTv2的多向量表示采用了与（1）和（2）类似的技术（见§3.2）。

<!-- Media -->

<!-- figureText: Question Encoder score Passage Encoder ... Passage Question -->

<img src="https://cdn.noedgeai.com/0195afc1-371d-7589-8122-942ba256f893_2.jpg?x=880&y=186&w=535&h=324&r=0"/>

Figure 1: The late interaction architecture, given a query and a passage. Diagram from Khattab et al. (2021b) with permission.

图1：给定一个查询和一个段落的后期交互架构。经许可引用卡塔布等人（2021b）的图表。

<!-- Media -->

### 2.4 Out-of-Domain Evaluation in IR

### 2.4 信息检索中的跨领域评估

Recent progress in retrieval has mostly focused on large-data evaluation, where many tens of thousands of annotated training queries are associated with the test domain, as in MS MARCO or Natural Questions (Kwiatkowski et al., 2019). In these benchmarks, queries tend to reflect high-popularity topics like movies and athletes in Wikipedia. In practice, user-facing IR and QA applications often pertain to domain-specific corpora, for which little to no training data is available and whose topics are under-represented in large public collections.

检索领域的近期进展大多集中在大数据评估上，在这种评估中，数以万计的带注释训练查询与测试领域相关联，就像微软机器阅读理解数据集（MS MARCO）或自然问题数据集（夸特科夫斯基等人，2019年）那样。在这些基准测试中，查询往往反映了维基百科中电影和运动员等热门话题。实际上，面向用户的信息检索和问答应用通常涉及特定领域的语料库，而这些语料库可用的训练数据很少甚至没有，并且其主题在大型公共数据集中的代表性不足。

This out-of-domain regime has received recent attention with the BEIR (Thakur et al., 2021) benchmark. BEIR combines several existing datasets into a heterogeneous suite for "zero-shot IR" tasks, spanning bio-medical, financial, and scientific domains. While the BEIR datasets provide a useful testbed, many capture broad semantic relatedness tasks-like citations, counter arguments, or duplicate questions-instead of natural search tasks, or else they focus on high-popularity entities like those in Wikipedia. In $\$ 4$ ,we introduce LoTTE,a new dataset for out-of-domain retrieval, exhibiting natural search queries over long-tail topics.

这种跨领域情况最近因BEIR基准测试（塔库尔等人，2021年）而受到关注。BEIR将几个现有数据集组合成一个用于“零样本信息检索”任务的异构套件，涵盖生物医学、金融和科学领域。虽然BEIR数据集提供了一个有用的测试平台，但许多数据集捕捉的是广泛的语义相关性任务，如引用、反驳论点或重复问题，而不是自然搜索任务，或者它们关注的是维基百科中那样的热门实体。在$\$ 4$中，我们引入了LoTTE，一个用于跨领域检索的新数据集，展示了关于长尾主题的自然搜索查询。

## 3 ColBERTv2

## 3 ColBERTv2

We now introduce ColBERTv2, which improves the quality of multi-vector retrieval models (§3.2) while reducing their space footprint (§3.3).

我们现在介绍ColBERTv2，它在减少多向量检索模型空间占用（§3.3）的同时提高了其质量（§3.2）。

### 3.1 Modeling

### 3.1 建模

ColBERTv2 adopts the late interaction architecture of ColBERT, depicted in Figure 1. Queries and passages are independently encoded with BERT (Devlin et al., 2019), and the output embeddings encoding each token are projected to a lower dimension. During offline indexing,every passage $d$ in the corpus is encoded into a set of vectors, and these vectors are stored. At search time,the query $q$ is encoded into a multi-vector representation, and its similarity to a passage $d$ is computed as the summation of query-side "MaxSim" operations, namely, the largest cosine similarity between each query token embedding and all passage token embeddings:

ColBERTv2采用了图1所示的ColBERT后期交互架构。查询和段落分别用BERT（德夫林等人，2019年）进行编码，编码每个标记的输出嵌入被投影到较低维度。在离线索引期间，语料库中的每个段落$d$被编码为一组向量，并存储这些向量。在搜索时，查询$q$被编码为多向量表示，其与段落$d$的相似度计算为查询端“最大相似度”操作的总和，即每个查询标记嵌入与所有段落标记嵌入之间的最大余弦相似度：

$$
{S}_{q,d} = \mathop{\sum }\limits_{{i = 1}}^{N}\mathop{\max }\limits_{{j = 1}}^{M}{Q}_{i} \cdot  {D}_{j}^{T} \tag{1}
$$

where $Q$ is an matrix encoding the query with $N$ vectors and $D$ encodes the passage with $M$ vectors. The intuition of this architecture is to align each query token with the most contextually relevant passage token, quantify these matches, and combine the partial scores across the query. We refer to Khattab and Zaharia (2020) for a more detailed treatment of late interaction.

其中$Q$是一个用$N$个向量编码查询的矩阵，$D$用$M$个向量编码段落。这种架构的直觉是将每个查询标记与上下文最相关的段落标记对齐，量化这些匹配，并将查询中的部分得分组合起来。关于后期交互的更详细处理，请参考卡塔布和扎哈里亚（2020年）的研究。

### 3.2 Supervision

### 3.2 监督

Training a neural retriever typically requires positive and negative passages for each query in the training set. Khattab and Zaharia (2020) train ColBERT using the official $\left\langle  {q,{d}^{ + },{d}^{ - }}\right\rangle$ triples of MS MARCO. For each query,a positive ${d}^{ + }$ is human-annotated,and each negative ${d}^{ - }$ is sampled from unannotated BM25-retrieved passages.

训练一个神经检索器通常需要训练集中每个查询对应的正例和负例段落。卡塔布和扎哈里亚（2020年）使用微软机器阅读理解数据集（MS MARCO）的官方$\left\langle  {q,{d}^{ + },{d}^{ - }}\right\rangle$三元组来训练ColBERT。对于每个查询，一个正例${d}^{ + }$是人工标注的，每个负例${d}^{ - }$是从未标注的基于BM25检索的段落中采样得到的。

Subsequent work has identified several weaknesses in this standard supervision approach (see $§{2.3}$ ). Our goal is to adopt a simple,uniform supervision scheme that selects challenging negatives and avoids rewarding false positives or penalizing false negatives. To this end, we start with a ColBERT model trained with triples as in Khat-tab et al. (2021b), using this to index the training passages with ColBERTv2 compression.

后续工作发现了这种标准监督方法的几个弱点（见$§{2.3}$）。我们的目标是采用一种简单、统一的监督方案，该方案选择具有挑战性的负例，避免奖励误报或惩罚漏报。为此，我们从一个用三元组训练的ColBERT模型开始，就像卡塔布等人（2021b）所做的那样，用ColBERTv2压缩对训练段落进行索引。

For each training query,we retrieve the top- $k$ passages. We feed each of those query-passage pairs into a cross-encoder reranker. We use a 22M-parameter MiniLM (Wang et al., 2020) cross-encoder trained with distillation by Thakur et al. (2021). ${}^{2}$ This small model has been shown to exhibit very strong performance while being relatively efficient for inference, making it suitable for distillation.

对于每个训练查询，我们检索前 $k$ 个段落。我们将这些查询 - 段落对分别输入到一个交叉编码器重排器中。我们使用一个由Thakur等人（2021年）通过蒸馏训练的、具有2200万个参数的MiniLM（Wang等人，2020年）交叉编码器。${}^{2}$ 这个小型模型已被证明在推理时相对高效的同时，还能展现出非常强的性能，使其适合用于蒸馏。

We then collect $w$ -way tuples consisting of a query, a highly-ranked passage (or labeled positive), and one or more lower-ranked passages. In this work,we use $w = {64}$ passages per example. Like RocketQAv2 (Ren et al., 2021b), we use a KL-Divergence loss to distill the cross-encoder's scores into the ColBERT architecture. We use KL-Divergence as ColBERT produces scores (i.e., the sum of cosine similarities) with a restricted scale, which may not align directly with the output scores of the cross-encoder. We also employ in-batch negatives per GPU, where a cross-entropy loss is applied to the positive score of each query against all passages corresponding to other queries in the same batch. We repeat this procedure once to refresh the index and thus the sampled negatives.

然后，我们收集由一个查询、一个高排名段落（或标记为正例的段落）以及一个或多个低排名段落组成的 $w$ 元组。在这项工作中，我们每个示例使用 $w = {64}$ 个段落。与RocketQAv2（Ren等人，2021b）类似，我们使用KL散度损失将交叉编码器的分数蒸馏到ColBERT架构中。我们使用KL散度是因为ColBERT产生的分数（即余弦相似度之和）范围有限，可能无法直接与交叉编码器的输出分数对齐。我们还在每个GPU上使用批次内负例，对每个查询的正例分数与同一批次中对应其他查询的所有段落应用交叉熵损失。我们重复这个过程一次以刷新索引，从而更新采样的负例。

Denoised training with hard negatives has been positioned in recent work as ways to bridge the gap between single-vector and interaction-based models, including late interaction architectures like ColBERT. Our results in $§5$ reveal that such supervision can improve multi-vector models dramatically, resulting in state-of-the-art retrieval quality.

在最近的工作中，使用难负例的去噪训练被定位为缩小单向量模型和基于交互的模型（包括像ColBERT这样的后期交互架构）之间差距的方法。我们在 $§5$ 中的结果表明，这种监督可以显著改进多向量模型，从而实现最先进的检索质量。

### 3.3 Representation

### 3.3 表示

We hypothesize that the ColBERT vectors cluster into regions that capture highly-specific token semantics. We test this hypothesis in Appendix A, where evidence suggests that vectors corresponding to each sense of a word cluster closely, with only minor variation due to context. We exploit this regularity with a residual representation that dramatically reduces the space footprint of late interaction models, completely off-the-shelf without architectural or training changes. Given a set of centroids $C$ ,ColBERTv2 encodes each vector $v$ as the index of its closest centroid ${C}_{t}$ and a quantized vector $\widetilde{r}$ that approximates the residual $r = v - {C}_{t}$ . At search time,we use the centroid index $t$ and residual $\widetilde{r}$ recover an approximate $\widetilde{v} = {C}_{t} + \widetilde{r}$ .

我们假设ColBERT向量会聚类成能够捕捉高度特定的词元语义的区域。我们在附录A中检验了这一假设，证据表明对应于一个词的每种语义的向量会紧密聚类，仅因上下文存在微小变化。我们利用这种规律性采用了一种残差表示，它能显著减少后期交互模型的空间占用，完全可以直接使用，无需进行架构或训练方面的更改。给定一组质心 $C$ ，ColBERTv2将每个向量 $v$ 编码为其最近质心的索引 ${C}_{t}$ 以及一个近似残差 $r = v - {C}_{t}$ 的量化向量 $\widetilde{r}$ 。在搜索时，我们使用质心索引 $t$ 和残差 $\widetilde{r}$ 来恢复一个近似向量 $\widetilde{v} = {C}_{t} + \widetilde{r}$ 。

To encode $\widetilde{r}$ ,we quantize every dimension of $r$ into one or two bits. In principle,our $b$ -bit encoding of $n$ -dimensional vectors needs $\lceil \log \left| C\right| \rceil  + {bn}$ bits per vector. In practice,with $n = {128}$ ,we use four bytes to capture up to ${2}^{32}$ centroids and 16 or 32 bytes (for $b = 1$ or $b = 2$ ) to encode the residual. This total of 20 or 36 bytes per vector contrasts with ColBERT's use of 256-byte vector encodings at 16-bit precision. While many alternatives can be explored for compression, we find that this simple encoding largely preserves model quality, while considerably lowering storage costs against typical 32- or 16-bit precision used by existing late interaction systems.

为了对 $\widetilde{r}$ 进行编码，我们将 $r$ 的每个维度量化为一到两位。原则上，我们对 $n$ 维向量的 $b$ 位编码每个向量需要 $\lceil \log \left| C\right| \rceil  + {bn}$ 位。实际上，当 $n = {128}$ 时，我们使用四个字节来表示最多 ${2}^{32}$ 个质心，并使用16或32个字节（对应 $b = 1$ 或 $b = 2$ ）来编码残差。每个向量总共20或36个字节，这与ColBERT在16位精度下使用的256字节向量编码形成对比。虽然可以探索许多其他压缩方法，但我们发现这种简单的编码在很大程度上保留了模型质量，同时与现有后期交互系统使用的典型32位或16位精度相比，显著降低了存储成本。

This centroid-based encoding can be considered a natural extension of product quantization to multi-vector representations. Product quantization (Gray, 1984; Jegou et al., 2010) compresses a single vector by splitting it into small sub-vectors and encoding each of them using an ID within a codebook. In our approach, each representation is already a matrix that is naturally divided into a number of small vectors (one per token). We encode each vector using its nearest centroid plus a residual. Refer to Appendix B for tests of the impact of compression on retrieval quality and a comparison with a baseline compression method for ColBERT akin to BPR (Yamada et al., 2021b).

这种基于质心的编码可以被视为乘积量化向多向量表示的自然扩展。乘积量化（Gray，1984年；Jegou等人，2010年）通过将单个向量拆分为小子向量，并使用码本中的ID对每个子向量进行编码来压缩该向量。在我们的方法中，每个表示已经是一个矩阵，自然地被划分为多个小向量（每个词元一个）。我们使用每个向量最近的质心加上一个残差来对其进行编码。有关压缩对检索质量影响的测试以及与类似于BPR（Yamada等人，2021b）的ColBERT基线压缩方法的比较，请参考附录B。

---

<!-- Footnote -->

${}^{2}$ https://huggingface.co/cross-encoder/ ms-marco-MiniLM-L-6-v2

${}^{2}$ https://huggingface.co/cross-encoder/ ms-marco-MiniLM-L-6-v2

<!-- Footnote -->

---

### 3.4 Indexing

### 3.4 索引

Given a corpus of passages, the indexing stage precomputes all passage embeddings and organizes their representations to support fast nearest-neighbor search. ColBERTv2 divides indexing into three stages, described below.

给定一个段落语料库，索引阶段会预先计算所有段落嵌入，并组织它们的表示以支持快速最近邻搜索。ColBERTv2将索引分为三个阶段，如下所述。

Centroid Selection. In the first stage, Col-BERTv2 selects a set of cluster centroids $C$ . These are embeddings that ColBERTv2 uses to support residual encoding $\left( {§{3.3}}\right)$ and also for nearest-neighbor search (§3.5). Standardly, we find that setting $\left| C\right|$ proportionally to the square root of ${n}_{\text{embeddings }}$ in the corpus works well empirically. ${}^{3}$ Khattab and Zaharia (2020) only clustered the vectors after computing the representations of all passages, but doing so requires storing them uncompressed. To reduce memory consumption, we apply $k$ -means clustering to the embeddings produced by invoking our BERT encoder over only a sample of all passages, proportional to the square root of the collection size, an approach we found to perform well in practice.

质心选择。在第一阶段，Col - BERTv2选择一组聚类质心$C$。这些是Col - BERTv2用于支持残差编码$\left( {§{3.3}}\right)$以及最近邻搜索（§3.5）的嵌入。通常，我们发现将$\left| C\right|$设置为语料库中${n}_{\text{embeddings }}$的平方根的比例在经验上效果很好。Khattab和Zaharia（2020）仅在计算所有段落的表示后对向量进行聚类，但这样做需要以未压缩的形式存储它们。为了减少内存消耗，我们对仅通过对所有段落的一个样本调用我们的BERT编码器所产生的嵌入应用$k$ - 均值聚类，该样本与集合大小的平方根成比例，我们发现这种方法在实践中表现良好。

Passage Encoding. Having selected the centroids, we encode every passage in the corpus. This entails invoking the BERT encoder and compressing the output embeddings as described in $\$ {3.3}$ , assigning each embedding to the nearest centroid and computing a quantized residual. Once a chunk of passages is encoded, the compressed representations are saved to disk.

段落编码。选择质心后，我们对语料库中的每个段落进行编码。这需要调用BERT编码器，并按照$\$ {3.3}$中所述压缩输出嵌入，将每个嵌入分配给最近的质心并计算量化残差。一旦对一批段落进行了编码，压缩后的表示就会保存到磁盘上。

Index Inversion. To support fast nearest-neighbor search, we group the embedding IDs that correspond to each centroid together, and save this inverted list to disk. At search time, this allows us to quickly find token-level embeddings similar to those in a query.

索引反转。为了支持快速的最近邻搜索，我们将对应于每个质心的嵌入ID分组在一起，并将这个倒排列表保存到磁盘上。在搜索时，这使我们能够快速找到与查询中的嵌入相似的词元级嵌入。

### 3.5 Retrieval

### 3.5 检索

Given a query representation $Q$ ,retrieval starts with candidate generation. For every vector ${Q}_{i}$ in the query,the nearest ${n}_{\text{probe }} \geq  1$ centroids are found. Using the inverted list, ColBERTv2 identifies the passage embeddings close to these centroids, decompresses them, and computes their cosine similarity with every query vector. The scores are then grouped by passage ID for each query vector, and scores corresponding to the same passage are max-reduced. This allows ColBERTv2 to conduct an approximate "MaxSim" operation per query vector. This computes a lower-bound on the true MaxSim (§3.1) using the embeddings identified via the inverted list, which resembles the approximation explored for scoring by Macdonald and Tonellotto (2021) but is applied for candidate generation.

给定一个查询表示$Q$，检索从候选生成开始。对于查询中的每个向量${Q}_{i}$，找到最近的${n}_{\text{probe }} \geq  1$个质心。使用倒排列表，Col - BERTv2识别出接近这些质心的段落嵌入，对其进行解压缩，并计算它们与每个查询向量的余弦相似度。然后，分数按每个查询向量的段落ID进行分组，对应于同一段落的分数进行最大归约。这使Col - BERTv2能够对每个查询向量进行近似的“最大相似度（MaxSim）”操作。这使用通过倒排列表识别的嵌入计算真实MaxSim（§3.1）的下界，这类似于Macdonald和Tonellotto（2021）为评分所探索的近似方法，但应用于候选生成。

These lower bounds are summed across the query tokens,and the top-scoring ${n}_{\text{candidate }}$ candidate passages based on these approximate scores are selected for ranking, which loads the complete set of embeddings of each passage, and conducts the same scoring function using all embeddings per document following Equation 1. The result passages are then sorted by score and returned.

这些下界在查询词元上求和，根据这些近似分数选择得分最高的${n}_{\text{candidate }}$个候选段落进行排序，这会加载每个段落的完整嵌入集，并根据公式1使用每个文档的所有嵌入进行相同的评分函数计算。然后，结果段落按分数排序并返回。

## 4 LoTTE: Long-Tail, Cross-Domain Retrieval Evaluation

## 4 LoTTE：长尾、跨领域检索评估

We introduce LoTTE (pronounced latte), a new dataset for Long-Tail Topic-stratified Evaluation for IR. To complement the out-of-domain tests of BEIR (Thakur et al., 2021), as motivated in §2.4, LoTTE focuses on natural user queries that pertain to long-tail topics, ones that might not be covered by an entity-centric knowledge base like Wikipedia. LoTTE consists of 12 test sets, each with 500-2000 queries and ${100}\mathrm{k} - 2\mathrm{M}$ passages.

我们引入了LoTTE（发音为“拿铁”），这是一个用于信息检索（IR）的长尾主题分层评估的新数据集。为了补充BEIR（Thakur等人，2021）的域外测试，如§2.4中所述，LoTTE专注于与长尾主题相关的自然用户查询，这些主题可能未被像维基百科这样以实体为中心的知识库所涵盖。LoTTE由12个测试集组成，每个测试集包含500 - 2000个查询和${100}\mathrm{k} - 2\mathrm{M}$个段落。

The test sets are explicitly divided by topic, and each test set is accompanied by a validation set of related but disjoint queries and passages. We elect to make the passage texts disjoint to encourage more realistic out-of-domain transfer tests, allowing for minimal development on related but distinct topics. The test (and dev) sets include a "pooled" setting. In the pooled setting, the passages and queries are aggregated across all test (or dev) topics to evaluate out-of-domain retrieval across a larger and more diverse corpus.

测试集按主题明确划分，每个测试集都配有一个相关但不相交的查询和段落验证集。我们选择使段落文本不相交，以鼓励更现实的域外迁移测试，允许在相关但不同的主题上进行最少的开发。测试（和开发）集包括一个“合并”设置。在合并设置中，段落和查询在所有测试（或开发）主题上进行聚合，以评估在更大、更多样化的语料库上的域外检索。

Table 1 outlines the composition of LoTTE. We derive the topics and passage corpora from the answer posts across various StackExchange forums. StackExchange is a set of question-and-answer communities that target individual topics (e.g., "physics" or "bicycling"). We gather forums from five overarching domains: writing, recreation, science, technology, and lifestyle. To evaluate retrievers, we collect Search and Forum queries, each of which is associated with one or more target answer posts in its corpus. Example queries, and short snippets from posts that answer them in the corpora, are shown in Table 2.

表1概述了LoTTE的组成。我们从各个StackExchange论坛的答案帖子中获取主题和段落语料库。StackExchange是一组针对单个主题（例如“物理学”或“骑自行车”）的问答社区。我们从五个总体领域收集论坛：写作、娱乐、科学、技术和生活方式。为了评估检索器，我们收集搜索和论坛查询，每个查询都与语料库中的一个或多个目标答案帖子相关联。表2显示了示例查询以及语料库中回答这些查询的帖子的简短片段。

---

<!-- Footnote -->

${}^{3}$ We round down to the nearest power of two larger than ${16} \times  \sqrt{{n}_{\text{embeddings }}}$ ,inspired by FAISS (Johnson et al.,2019).

${}^{3}$ 受FAISS（Johnson等人，2019）的启发，我们向下取整到比${16} \times  \sqrt{{n}_{\text{embeddings }}}$大的最接近的2的幂。

<!-- Footnote -->

---

<!-- Media -->

<table><tr><td rowspan="2">Topic</td><td rowspan="2">Question Set</td><td colspan="3">Dev</td><td colspan="3">Test</td></tr><tr><td>#Questions</td><td>#Passages</td><td>Subtopics</td><td>#Questions</td><td>#Passages</td><td>Subtopics</td></tr><tr><td rowspan="2">Writing</td><td>Search</td><td>497</td><td rowspan="2">277k</td><td>ESL, Linguistics,</td><td>1071</td><td rowspan="2">200k</td><td rowspan="2">English</td></tr><tr><td>Forum</td><td>2003</td><td>Worldbuilding</td><td>2000</td></tr><tr><td rowspan="2">Recreation</td><td>Search</td><td>563</td><td rowspan="2">263k</td><td>Sci-Fi, RPGs,</td><td>924</td><td rowspan="2">167k</td><td rowspan="2">Gaming, Anime, Movies</td></tr><tr><td>Forum</td><td>2002</td><td>Photography</td><td>2002</td></tr><tr><td rowspan="2">Science</td><td>Search</td><td>538</td><td rowspan="2">344k</td><td>Chemistry,</td><td>617</td><td rowspan="2">1.694M</td><td rowspan="2">Math, Physics, Biology</td></tr><tr><td>Forum</td><td>2013</td><td>Statistics, Academia</td><td>2017</td></tr><tr><td rowspan="2">Technology</td><td>Search</td><td>916</td><td rowspan="2">1.276M</td><td>Web Apps,</td><td>596</td><td rowspan="2">639k</td><td rowspan="2">Apple, Android, UNIX, Security</td></tr><tr><td>Forum</td><td>2003</td><td>Ubuntu, SysAdmin</td><td>2004</td></tr><tr><td rowspan="2">Lifestyle</td><td>Search</td><td>417</td><td rowspan="2">269k</td><td>DIY, Music, Bicycles,</td><td>661</td><td rowspan="2">119k</td><td rowspan="2">Cooking, Sports, Travel</td></tr><tr><td>Forum</td><td>2076</td><td>Car Maintenance</td><td>2002</td></tr><tr><td rowspan="2">Pooled</td><td>Search</td><td>2931</td><td rowspan="2">2.4M</td><td rowspan="2">All of the above</td><td>3869</td><td rowspan="2">2.8M</td><td rowspan="2">All of the above</td></tr><tr><td>Forum</td><td>10097</td><td>10025</td></tr></table>

<table><tbody><tr><td rowspan="2">主题</td><td rowspan="2">问题集</td><td colspan="3">开发</td><td colspan="3">测试</td></tr><tr><td>#问题</td><td>#文章</td><td>子主题</td><td>#问题</td><td>#文章</td><td>子主题</td></tr><tr><td rowspan="2">写作</td><td>搜索</td><td>497</td><td rowspan="2">277k</td><td>英语作为第二语言教学、语言学</td><td>1071</td><td rowspan="2">200k</td><td rowspan="2">英语</td></tr><tr><td>论坛</td><td>2003</td><td>世界构建</td><td>2000</td></tr><tr><td rowspan="2">娱乐</td><td>搜索</td><td>563</td><td rowspan="2">263k</td><td>科幻、角色扮演游戏</td><td>924</td><td rowspan="2">167k</td><td rowspan="2">游戏、动漫、电影</td></tr><tr><td>论坛</td><td>2002</td><td>摄影</td><td>2002</td></tr><tr><td rowspan="2">科学</td><td>搜索</td><td>538</td><td rowspan="2">344k</td><td>化学</td><td>617</td><td rowspan="2">1.694M</td><td rowspan="2">数学、物理、生物</td></tr><tr><td>论坛</td><td>2013</td><td>统计学、学术界</td><td>2017</td></tr><tr><td rowspan="2">技术</td><td>搜索</td><td>916</td><td rowspan="2">1.276M</td><td>网络应用程序</td><td>596</td><td rowspan="2">639k</td><td rowspan="2">苹果、安卓、UNIX系统、安全</td></tr><tr><td>论坛</td><td>2003</td><td>乌班图系统、系统管理</td><td>2004</td></tr><tr><td rowspan="2">生活方式</td><td>搜索</td><td>417</td><td rowspan="2">269k</td><td>自己动手做、音乐、自行车</td><td>661</td><td rowspan="2">119k</td><td rowspan="2">烹饪、体育、旅行</td></tr><tr><td>论坛</td><td>2076</td><td>汽车保养</td><td>2002</td></tr><tr><td rowspan="2">合并的</td><td>搜索</td><td>2931</td><td rowspan="2">2.4M</td><td rowspan="2">以上所有</td><td>3869</td><td rowspan="2">2.8M</td><td rowspan="2">以上所有</td></tr><tr><td>论坛</td><td>10097</td><td>10025</td></tr></tbody></table>

Table 1: Composition of LoTTE showing topics, question sets, and a sample of corresponding subtopics. Search Queries are taken from GooAQ, while Forum Queries are taken directly from the StackExchange archive. The pooled datasets combine the questions and passages from each of the subtopics.

表1：LoTTE的构成，展示了主题、问题集以及相应子主题的示例。搜索查询取自GooAQ，而论坛查询则直接取自StackExchange存档。合并后的数据集整合了每个子主题的问题和段落。

<!-- Media -->

Search Queries. We collect search queries from GooAQ (Khashabi et al., 2021), a recent dataset of Google search-autocomplete queries and their answer boxes, which we filter for queries whose answers link to a specific StackExchange post. As Khashabi et al. (2021) hypothesize, Google Search likely maps these natural queries to their answers by relying on a wide variety of signals for relevance, including expert annotations, user clicks, and hyperlinks as well as specialized QA components for various question types with access to the post title and question body. Using those annotations as ground truth, we evaluate the models on their capacity for retrieval using only free text of the answer posts (i.e., no hyperlinks or user clicks, question title or body, etc.), posing a significant challenge for IR and NLP systems trained only on public datasets.

搜索查询。我们从GooAQ（Khashabi等人，2021年）收集搜索查询，这是一个近期的谷歌搜索自动补全查询及其答案框的数据集，我们筛选出答案链接到特定StackExchange帖子的查询。正如Khashabi等人（2021年）所假设的，谷歌搜索可能通过依赖各种相关性信号将这些自然查询映射到其答案，这些信号包括专家注释、用户点击、超链接以及可访问帖子标题和问题正文的各种问题类型的专业问答组件。我们以这些注释为真实标准，评估模型仅使用答案帖子的自由文本（即无超链接、用户点击、问题标题或正文等）进行检索的能力，这对仅在公共数据集上训练的信息检索（IR）和自然语言处理（NLP）系统构成了重大挑战。

Forum Queries. We collect the forum queries by extracting post titles from the StackExchange communities to use as queries and collect their corresponding answer posts as targets. We select questions in order of their popularity and sample questions according to the proportional contribution of individual communities within each topic.

论坛查询。我们通过从StackExchange社区提取帖子标题作为查询，并收集相应的答案帖子作为目标来收集论坛查询。我们按照问题的受欢迎程度选择问题，并根据每个主题内各个社区的比例贡献对问题进行抽样。

<!-- Media -->

Q: what is the difference between root and stem in linguistics? A: A root is the form to which derivational affixes are added to form a stem. A stem is the form to which inflectional affixes are added to form a word.

问：在语言学中，词根（root）和词干（stem）有什么区别？答：词根是添加派生词缀以形成词干的形式。词干是添加屈折词缀以形成单词的形式。

Q: are there any airbenders left? A: the Fire Nation had wiped out all Airbenders while Aang was frozen. Tenzin and his 3 children are the only Airbenders left in Korra’s time.

问：还有气宗（Airbenders）吗？答：在安昂（Aang）被冰封期间，火之国（Fire Nation）消灭了所有气宗。在科拉（Korra）的时代，只有坦津（Tenzin）和他的三个孩子是仅存的气宗。

Q: Why are there two Hydrogen atoms on some periodic tables? A: some periodic tables show hydrogen in both places to emphasize that hydrogen isn't really a member of the first group or the seventh group.

问：为什么有些元素周期表上有两个氢原子的位置？答：有些元素周期表在两个位置都显示氢，以强调氢实际上既不属于第一族也不属于第七族。

Q: How can cache be that fast? A: the cache memory sits right next to the CPU on the same die (chip), it is made using SRAM which is much, much faster than the DRAM.

问：缓存怎么能这么快？答：缓存内存与CPU位于同一芯片上，它使用静态随机存取存储器（SRAM）制造，比动态随机存取存储器（DRAM）快得多。

Table 2: Examples of queries and shortened snippets of answer passages from LoTTE. The first two examples show "search" queries, whereas the last two are "forum" queries. Snippets are shortened for presentation.

表2：LoTTE中的查询示例和答案段落的简短摘录。前两个示例是“搜索”查询，而后两个是“论坛”查询。摘录为展示而进行了缩短。

<!-- Media -->

These queries tend to have a wider variety than the "search" queries, while the search queries may exhibit more natural patterns. Table 3 compares a random samples of search and forum queries. It can be seen that search queries tend to be brief, knowledge-based questions with direct answers, whereas forum queries tend to reflect more open-ended questions. Both query sets target topics that exceed the scope of a general-purpose knowledge repository such as Wikipedia.

这些查询往往比“搜索”查询具有更多样性，而搜索查询可能表现出更自然的模式。表3比较了随机抽取的搜索查询和论坛查询样本。可以看出，搜索查询往往是简短的、基于知识的问题，有直接的答案，而论坛查询往往反映更开放性的问题。两个查询集的目标主题都超出了维基百科等通用知识库的范围。

For search as well as forum queries, the resulting evaluation set consists of a query and a target set of StackExchange answer posts (in particular, the answer posts from the target StackExchange page). Similar to evaluation in the Open-QA literature (Karpukhin et al., 2020; Khattab et al.,

对于搜索查询和论坛查询，最终的评估集由一个查询和一组StackExchange答案帖子（特别是目标StackExchange页面上的答案帖子）组成。与开放问答文献中的评估类似（Karpukhin等人，2020年；Khattab等人，

<!-- Media -->

Q: what is xerror in rpart? Q: is sub question one word? Q: how to open a garage door without making noise? Q: is docx and dotx the same? Q: are upvotes and downvotes anonymous? Q: what is the difference between descriptive essay and narrative essay? Q: how to change default user profile in chrome? Q: does autohotkey need to be installed? Q: how do you tag someone on facebook with a youtube video? Q: has mjolnir ever been broken?

问：rpart中的xerror是什么？问：“sub question”是一个词吗？问：如何在不发出噪音的情况下打开车库门？问：docx和dotx一样吗？问：点赞和点踩是匿名的吗？问：描述性论文和叙事性论文有什么区别？问：如何更改Chrome中的默认用户配置文件？问：AutoHotkey需要安装吗？问：如何在Facebook上用YouTube视频标记某人？问：雷神之锤（Mjolnir）曾经被破坏过吗？

Q: Snoopy can balance on an edge atop his doghouse. Is any reason given for this? Q: How many Ents were at the Entmoot? Q: What does a hexagonal sun tell us about the camera lens/sensor? Q: Should I simply ignore it if authors assume that $\operatorname{Im}$ male in their response to my review of their article? Q: Why is the 2s orbital lower in energy than the ${2p}$ orbital when the electrons in ${2s}$ are usually farther from the nucleus? Q: Are there reasons to use colour filters with digital cameras? Q: How does the current know how much to flow,before having seen the resistor? Q: What is the difference between Fact and Truth? Q: hAs a DM, how can I handle my Druid spying on everything with Wild shape as a spider? Q: What does 1x1 convolution mean

问：史努比（Snoopy）能在狗屋顶部的边缘保持平衡。有给出原因吗？问：树须会议（Entmoot）上有多少树人（Ents）？问：六边形的太阳能告诉我们关于相机镜头/传感器的什么信息？问：如果作者在回复我对他们文章的评论时假设$\operatorname{Im}$为男性，我应该直接忽略吗？问：为什么2s轨道的能量比${2p}$轨道低，而${2s}$中的电子通常离原子核更远？问：使用数码相机时使用彩色滤镜有什么原因吗？问：电流在遇到电阻器之前如何知道要流动多少？问：事实（Fact）和真相（Truth）有什么区别？问：作为地下城主（DM），我如何处理我的德鲁伊（Druid）以蜘蛛的野性变形形态窥探一切的情况？问：在神经网络中，1x1卷积是什么意思

in a neural network?

在神经网络中？

Table 3: Comparison of a random sample of search queries (top) vs. forum queries (bottom).

表3：随机抽取的搜索查询（上）与论坛查询（下）的比较。

<!-- Media -->

2021b), we evaluate retrieval quality by computing the success@5 (S@5) metric. Specifically, we award a point to the system for each query where it finds an accepted or upvoted (score $\geq  1$ ) answer from the target page in the top-5 hits.

在2021b的研究中，我们通过计算前5命中成功率（Success@5，简称S@5）指标来评估检索质量。具体而言，对于每个查询，如果系统在前5个命中结果中找到了目标页面上被接受或获得点赞（得分 $\geq  1$ ）的答案，我们就给该系统记一分。

Appendix D reports on the breakdown of constituent communities per topic, the construction procedure of LoTTE as well as licensing considerations, and relevant statistics. Figures 5 and 6 quantitatively compare the search and forum queries.

附录D报告了每个主题下子社区的细分情况、LoTTE（主题文本集合，原词为LoTTE）的构建过程、许可相关考虑因素以及相关统计数据。图5和图6对搜索查询和论坛查询进行了定量比较。

## 5 Evaluation

## 5 评估

We now evaluate ColBERTv2 on passage retrieval tasks, testing its quality within the training domain (§5.1) as well as outside the training domain in zero-shot settings (§5.2). Unless otherwise stated, we compress ColBERTv2 embeddings to $b = 2$ bits per dimension in our evaluation.

我们现在对ColBERTv2在段落检索任务上进行评估，测试其在训练领域内（§5.1）以及零样本设置下训练领域外（§5.2）的性能。除非另有说明，在评估中我们将ColBERTv2的嵌入向量压缩至每个维度 $b = 2$ 比特。

### 5.1 In-Domain Retrieval Quality

### 5.1 领域内检索质量

Similar to related work, we train for IR tasks on MS MARCO Passage Ranking (Nguyen et al., 2016). Within the training domain, our development-set results are shown in Table 4, comparing ColBERTv2 with vanilla ColBERT as well as state-of-the-art single-vector systems.

与相关工作类似，我们在MS MARCO段落排序任务（Nguyen等人，2016）上进行信息检索（IR）任务的训练。在训练领域内，我们开发集的结果如表4所示，将ColBERTv2与原始的ColBERT以及最先进的单向量系统进行了比较。

While ColBERT outperforms single-vector systems like RepBERT, ANCE, and even TAS-B, improvements in supervision such as distillation from cross-encoders enable systems like SPLADEv2,

虽然ColBERT的性能优于像RepBERT、ANCE甚至TAS - B这样的单向量系统，但通过如从交叉编码器进行知识蒸馏等监督方式的改进，使得像SPLADEv2这样的系统……

<!-- Media -->

Method Official Dev (7k) Local Eval (5k) MRR@10 R@50 R@1k MRR@10

方法 官方开发集（7k） 本地评估（5k） 前10平均倒数排名（MRR@10） 前50召回率（R@50） 前1000召回率（R@1k） 前10平均倒数排名（MRR@10）

Models without Distillation or Special Pretraining

未进行蒸馏或特殊预训练的模型

RepBERT 30.4 - 94.3 - - -

RepBERT 30.4 - 94.3 - - -

DPR 31.1 - 95.2 - - -

DPR 31.1 - 95.2 - - -

ANCE 33.0 95.9 - -

ANCE 33.0 95.9 - -

LTRe 34.1 96.2

LTRe 34.1 96.2

ColBERT 82.9 36.7

ColBERT 82.9 36.7

<table><tr><td colspan="7">Models with Distillation or Special Pretraining</td></tr><tr><td>TAS-B</td><td>34.7</td><td>-</td><td>97.8</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SPLADEv2</td><td>36.8</td><td>-</td><td>97.9</td><td>37.9</td><td>84.9</td><td>98.0</td></tr><tr><td>PAIR</td><td>37.9</td><td>86.4</td><td>98.2</td><td>-</td><td>-</td><td>-</td></tr><tr><td>coCondenser</td><td>38.2</td><td>-</td><td>98.4</td><td>-</td><td>-</td><td>-</td></tr><tr><td>RocketQAv2</td><td>38.8</td><td>86.2</td><td>98.1</td><td>39.8</td><td>85.8</td><td>97.9</td></tr><tr><td>ColBERTv2</td><td>39.7</td><td>86.8</td><td>98.4</td><td>40.8</td><td>86.3</td><td>98.3</td></tr></table>

<table><tbody><tr><td colspan="7">采用蒸馏或特殊预训练的模型</td></tr><tr><td>TAS - B（原文未变）</td><td>34.7</td><td>-</td><td>97.8</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SPLADEv2（原文未变）</td><td>36.8</td><td>-</td><td>97.9</td><td>37.9</td><td>84.9</td><td>98.0</td></tr><tr><td>PAIR（原文未变）</td><td>37.9</td><td>86.4</td><td>98.2</td><td>-</td><td>-</td><td>-</td></tr><tr><td>coCondenser（原文未变）</td><td>38.2</td><td>-</td><td>98.4</td><td>-</td><td>-</td><td>-</td></tr><tr><td>RocketQAv2（原文未变）</td><td>38.8</td><td>86.2</td><td>98.1</td><td>39.8</td><td>85.8</td><td>97.9</td></tr><tr><td>ColBERTv2（原文未变）</td><td>39.7</td><td>86.8</td><td>98.4</td><td>40.8</td><td>86.3</td><td>98.3</td></tr></tbody></table>

Table 4: In-domain performance on the development set of MS MARCO Passage Ranking as well the "Local Eval" test set described by Khattab and Zaharia (2020). Dev-set results for baseline systems are from their respective papers: Zhan et al. (2020b), Xiong et al. (2020) for DPR and ANCE, Zhan et al. (2020a), Khattab and Zaharia (2020), Hofstätter et al. (2021), Gao and Callan (2021), Ren et al. (2021a), Formal et al. (2021a), and Ren et al. (2021b).

表4：在MS MARCO段落排名开发集以及Khattab和Zaharia（2020年）描述的“本地评估”测试集上的领域内性能。基线系统的开发集结果来自各自的论文：DPR和ANCE的结果分别来自Zhan等人（2020b）、Xiong等人（2020年）；其他结果来自Zhan等人（2020a）、Khattab和Zaharia（2020年）、Hofstätter等人（2021年）、Gao和Callan（2021年）、Ren等人（2021a）、Formal等人（2021a）以及Ren等人（2021b）。

<!-- Media -->

PAIR, and RocketQAv2 to achieve higher quality than vanilla ColBERT. These supervision gains challenge the value of fine-grained late interaction, and it is not inherently clear whether the stronger inductive biases of ColBERT-like models permit it to accept similar gains under distillation, especially when using compressed representations. Despite this, we find that with denoised supervision and residual compression, ColBERTv2 achieves the highest quality across all systems. As we discuss in $\$ {5.3}$ ,it exhibits space footprint competitive with these single-vector models and much lower than vanilla ColBERT.

PAIR和RocketQAv2比普通的ColBERT质量更高。这些监督增益对细粒度后期交互的价值提出了挑战，目前尚不清楚类似ColBERT的模型更强的归纳偏置是否能使其在蒸馏过程中获得类似的增益，尤其是在使用压缩表示时。尽管如此，我们发现通过去噪监督和残差压缩，ColBERTv2在所有系统中质量最高。正如我们在$\$ {5.3}$中所讨论的，它的空间占用与这些单向量模型相当，并且远低于普通的ColBERT。

Besides the official dev set, we evaluated Col-BERTv2, SPLADEv2, and RocketQAv2 on the "Local Eval" test set described by Khattab and Za-haria (2020) for MS MARCO, which consists of 5000 queries disjoint with the training and the official dev sets. These queries are obtained from labeled ${50}\mathrm{k}$ queries that are provided in the official MS MARCO Passage Ranking task as additional validation data. ${}^{4}$ On this test set, ColBERTv2 obtains ${40.8}\%$ MRR@10,considerably outperforming the baselines, including RocketQAv2 which makes use of document titles in addition to the passage text unlike the other systems. Table 5: Zero-shot evaluation results. Sub-table (a) reports results on BEIR and sub-table (b) reports results on the Wikipedia Open QA and the test sets of the LoTTE benchmark. On BEIR, we test ColBERTv2 and Rock-etQAv2 and copy the results for ANCE, TAS-B, and ColBERT from Thakur et al. (2021), for MoDIR and DPR-MSMARCO (DPR-M) from Xin et al. (2021), and for SPLADEv2 from Formal et al. (2021a).

除了官方开发集之外，我们还在Khattab和Zaharia（2020年）为MS MARCO描述的“本地评估”测试集上评估了Col - BERTv2、SPLADEv2和RocketQAv2，该测试集包含5000个查询，与训练集和官方开发集不相交。这些查询来自官方MS MARCO段落排名任务中作为额外验证数据提供的带标签的${50}\mathrm{k}$查询。${}^{4}$在这个测试集上，ColBERTv2的MRR@10达到${40.8}\%$，显著优于基线系统，包括RocketQAv2，与其他系统不同，RocketQAv2除了段落文本之外还利用了文档标题。表5：零样本评估结果。子表（a）报告了在BEIR上的结果，子表（b）报告了在维基百科开放问答和LoTTE基准测试集上的结果。在BEIR上，我们测试了ColBERTv2和RocketQAv2，并从Thakur等人（2021年）处复制了ANCE、TAS - B和ColBERT的结果，从Xin等人（2021年）处复制了MoDIR和DPR - MSMARCO（DPR - M）的结果，从Formal等人（2021a）处复制了SPLADEv2的结果。

---

<!-- Footnote -->

${}^{4}$ These are sampled from delta between qrels.dev. tsv and qrels.dev.small.tsv on https://microsoft.github.io/msmarco/Datasets.We refer to Khattab and Zaharia (2020) for details. All our query IDs will be made public to aid reproducibility.

${}^{4}$这些是从https://microsoft.github.io/msmarco/Datasets上的qrels.dev.tsv和qrels.dev.small.tsv之间的差异中采样得到的。详情请参考Khattab和Zaharia（2020年）。我们所有的查询ID将公开以方便复现。

<!-- Footnote -->

---

<!-- Media -->

<table><tr><td rowspan="2">Corpus</td><td colspan="4">Models without Distillation</td><td colspan="4">Models with Distillation</td></tr><tr><td>LUTHION</td><td>IN-UICI</td><td>HONV</td><td>AIGON</td><td>g-swi-</td><td>tavoraway</td><td>TAHAVIdS</td><td>talilition</td></tr><tr><td colspan="9">BEIR Search Tasks (nDCG@10)</td></tr><tr><td>DBPedia</td><td>39.2</td><td>23.6</td><td>28.1</td><td>28.4</td><td>38.4</td><td>35.6</td><td>43.5</td><td>44.6</td></tr><tr><td>FiQA</td><td>31.7</td><td>27.5</td><td>29.5</td><td>29.6</td><td>30.0</td><td>30.2</td><td>33.6</td><td>35.6</td></tr><tr><td>$\mathbf{{NQ}}$</td><td>52.4</td><td>39.8</td><td>44.6</td><td>44.2</td><td>46.3</td><td>50.5</td><td>52.1</td><td>56.2</td></tr><tr><td>HotpotQA</td><td>59.3</td><td>37.1</td><td>45.6</td><td>46.2</td><td>58.4</td><td>53.3</td><td>68.4</td><td>66.7</td></tr><tr><td>NFCorpus</td><td>30.5</td><td>20.8</td><td>23.7</td><td>24.4</td><td>31.9</td><td>29.3</td><td>33.4</td><td>33.8</td></tr><tr><td>T-COVID</td><td>67.7</td><td>56.1</td><td>65.4</td><td>67.6</td><td>48.1</td><td>67.5</td><td>71.0</td><td>73.8</td></tr><tr><td>Touché (v2)</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>24.7</td><td>27.2</td><td>26.3</td></tr><tr><td colspan="9">BEIR Semantic Relatedness Tasks (nDCG@10)</td></tr><tr><td>ArguAna</td><td>23.3</td><td>41.4</td><td>41.5</td><td>41.8</td><td>42.7</td><td>45.1</td><td>47.9</td><td>46.3</td></tr><tr><td>C-FEVER</td><td>18.4</td><td>17.6</td><td>19.8</td><td>20.6</td><td>22.8</td><td>18.0</td><td>23.5</td><td>17.6</td></tr><tr><td>FEVER</td><td>77.1</td><td>58.9</td><td>66.9</td><td>68.0</td><td>70.0</td><td>67.6</td><td>78.6</td><td>78.5</td></tr><tr><td>Quora</td><td>85.4</td><td>84.2</td><td>85.2</td><td>85.6</td><td>83.5</td><td>74.9</td><td>83.8</td><td>85.2</td></tr><tr><td>SCIDOCS</td><td>14.5</td><td>10.8</td><td>12.2</td><td>12.4</td><td>14.9</td><td>13.1</td><td>15.8</td><td>15.4</td></tr><tr><td>SciFact</td><td>67.1</td><td>47.8</td><td>50.7</td><td>50.2</td><td>64.3</td><td>56.8</td><td>69.3</td><td>69.3</td></tr></table>

<table><tbody><tr><td rowspan="2">语料库</td><td colspan="4">无蒸馏的模型</td><td colspan="4">有蒸馏的模型</td></tr><tr><td>LUTHION</td><td>IN - UICI</td><td>HONV</td><td>AIGON</td><td>g - swi -</td><td>tavoraway</td><td>TAHAVIdS</td><td>talilition</td></tr><tr><td colspan="9">BEIR搜索任务（nDCG@10）</td></tr><tr><td>DBPedia（DBpedia是一个从维基百科中提取结构化信息并将其转换为机器可读的知识图谱的项目）</td><td>39.2</td><td>23.6</td><td>28.1</td><td>28.4</td><td>38.4</td><td>35.6</td><td>43.5</td><td>44.6</td></tr><tr><td>FiQA（金融问答数据集）</td><td>31.7</td><td>27.5</td><td>29.5</td><td>29.6</td><td>30.0</td><td>30.2</td><td>33.6</td><td>35.6</td></tr><tr><td>$\mathbf{{NQ}}$</td><td>52.4</td><td>39.8</td><td>44.6</td><td>44.2</td><td>46.3</td><td>50.5</td><td>52.1</td><td>56.2</td></tr><tr><td>HotpotQA（火锅问答数据集）</td><td>59.3</td><td>37.1</td><td>45.6</td><td>46.2</td><td>58.4</td><td>53.3</td><td>68.4</td><td>66.7</td></tr><tr><td>NFCorpus（可能是特定的语料库，暂无通用中文名称）</td><td>30.5</td><td>20.8</td><td>23.7</td><td>24.4</td><td>31.9</td><td>29.3</td><td>33.4</td><td>33.8</td></tr><tr><td>T - COVID</td><td>67.7</td><td>56.1</td><td>65.4</td><td>67.6</td><td>48.1</td><td>67.5</td><td>71.0</td><td>73.8</td></tr><tr><td>Touché（v2）</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>24.7</td><td>27.2</td><td>26.3</td></tr><tr><td colspan="9">BEIR语义相关性任务（nDCG@10）</td></tr><tr><td>ArguAna（可能是特定的数据集或项目，暂无通用中文名称）</td><td>23.3</td><td>41.4</td><td>41.5</td><td>41.8</td><td>42.7</td><td>45.1</td><td>47.9</td><td>46.3</td></tr><tr><td>C - FEVER</td><td>18.4</td><td>17.6</td><td>19.8</td><td>20.6</td><td>22.8</td><td>18.0</td><td>23.5</td><td>17.6</td></tr><tr><td>FEVER（事实提取与验证数据集）</td><td>77.1</td><td>58.9</td><td>66.9</td><td>68.0</td><td>70.0</td><td>67.6</td><td>78.6</td><td>78.5</td></tr><tr><td>Quora（问答平台Quora）</td><td>85.4</td><td>84.2</td><td>85.2</td><td>85.6</td><td>83.5</td><td>74.9</td><td>83.8</td><td>85.2</td></tr><tr><td>SCIDOCS（科学文档相关数据集）</td><td>14.5</td><td>10.8</td><td>12.2</td><td>12.4</td><td>14.9</td><td>13.1</td><td>15.8</td><td>15.4</td></tr><tr><td>SciFact（科学事实验证数据集）</td><td>67.1</td><td>47.8</td><td>50.7</td><td>50.2</td><td>64.3</td><td>56.8</td><td>69.3</td><td>69.3</td></tr></tbody></table>

(a)

<table><tr><td>Corpus</td><td>LUTHOO</td><td>STINS</td><td>HONV</td><td>tavoids</td><td>TAHAVIdS</td><td>talilition</td></tr><tr><td colspan="7">OOD Wikipedia Open QA (Success@5)</td></tr><tr><td>NQ-dev</td><td>65.7</td><td>44.6</td><td>-</td><td>-</td><td>65.6</td><td>68.9</td></tr><tr><td>TQ-dev</td><td>72.6</td><td>67.6</td><td>-</td><td>-</td><td>74.7</td><td>76.7</td></tr><tr><td>SQuAD-dev</td><td>60.0</td><td>50.6</td><td>-</td><td>-</td><td>60.4</td><td>65.0</td></tr><tr><td colspan="7">LoTTE Search Test Queries (Success@5)</td></tr><tr><td>Writing</td><td>74.7</td><td>60.3</td><td>74.4</td><td>78.0</td><td>77.1</td><td>80.1</td></tr><tr><td>Recreation</td><td>68.5</td><td>56.5</td><td>64.7</td><td>72.1</td><td>69.0</td><td>72.3</td></tr><tr><td>Science</td><td>53.6</td><td>32.7</td><td>53.6</td><td>55.3</td><td>55.4</td><td>56.7</td></tr><tr><td>Technology</td><td>61.9</td><td>41.8</td><td>59.6</td><td>63.4</td><td>62.4</td><td>66.1</td></tr><tr><td>Lifestyle</td><td>80.2</td><td>63.8</td><td>82.3</td><td>82.1</td><td>82.3</td><td>84.7</td></tr><tr><td>Pooled</td><td>67.3</td><td>48.3</td><td>66.4</td><td>69.8</td><td>68.9</td><td>71.6</td></tr><tr><td colspan="7">LoTTE Forum Test Queries (Success@5)</td></tr><tr><td>Writing</td><td>71.0</td><td>64.0</td><td>68.8</td><td>71.5</td><td>73.0</td><td>76.3</td></tr><tr><td>Recreation</td><td>65.6</td><td>55.4</td><td>63.8</td><td>65.7</td><td>67.1</td><td>70.8</td></tr><tr><td>Science</td><td>41.8</td><td>37.1</td><td>36.5</td><td>38.0</td><td>43.7</td><td>46.1</td></tr><tr><td>Technology</td><td>48.5</td><td>39.4</td><td>46.8</td><td>47.3</td><td>50.8</td><td>53.6</td></tr><tr><td>Lifestyle</td><td>73.0</td><td>60.6</td><td>73.1</td><td>73.7</td><td>74.0</td><td>76.9</td></tr><tr><td>Pooled</td><td>58.2</td><td>47.2</td><td>55.7</td><td>57.7</td><td>60.1</td><td>63.4</td></tr><tr><td colspan="7">$f(x) = \frac{1}{2\;\pi }{\int }_{0}^{\infty }\frac{f(t)}{t\;{dt}}{dt} = \frac{1}{2\;\pi }{\int }_{0}^{\infty }\frac{f(t)}{t\;{dt}}{dt}$</td></tr></table>

<table><tbody><tr><td>语料库</td><td>卢图（LUTHOO）</td><td>斯廷斯（STINS）</td><td>洪夫（HONV）</td><td>避免（tavoids）</td><td>塔哈维德（TAHAVIdS）</td><td>塔利利申（talilition）</td></tr><tr><td colspan="7">开放在线百科开放问答（5次尝试内成功率）（OOD Wikipedia Open QA (Success@5)）</td></tr><tr><td>自然问答开发集（NQ-dev）</td><td>65.7</td><td>44.6</td><td>-</td><td>-</td><td>65.6</td><td>68.9</td></tr><tr><td>任务问答开发集（TQ-dev）</td><td>72.6</td><td>67.6</td><td>-</td><td>-</td><td>74.7</td><td>76.7</td></tr><tr><td>斯坦福问答开发集（SQuAD-dev）</td><td>60.0</td><td>50.6</td><td>-</td><td>-</td><td>60.4</td><td>65.0</td></tr><tr><td colspan="7">乐天搜索测试查询（5次尝试内成功率）（LoTTE Search Test Queries (Success@5)）</td></tr><tr><td>写作</td><td>74.7</td><td>60.3</td><td>74.4</td><td>78.0</td><td>77.1</td><td>80.1</td></tr><tr><td>娱乐</td><td>68.5</td><td>56.5</td><td>64.7</td><td>72.1</td><td>69.0</td><td>72.3</td></tr><tr><td>科学</td><td>53.6</td><td>32.7</td><td>53.6</td><td>55.3</td><td>55.4</td><td>56.7</td></tr><tr><td>技术</td><td>61.9</td><td>41.8</td><td>59.6</td><td>63.4</td><td>62.4</td><td>66.1</td></tr><tr><td>生活方式</td><td>80.2</td><td>63.8</td><td>82.3</td><td>82.1</td><td>82.3</td><td>84.7</td></tr><tr><td>合并的</td><td>67.3</td><td>48.3</td><td>66.4</td><td>69.8</td><td>68.9</td><td>71.6</td></tr><tr><td colspan="7">乐天论坛测试查询（5次尝试内成功率）（LoTTE Forum Test Queries (Success@5)）</td></tr><tr><td>写作</td><td>71.0</td><td>64.0</td><td>68.8</td><td>71.5</td><td>73.0</td><td>76.3</td></tr><tr><td>娱乐</td><td>65.6</td><td>55.4</td><td>63.8</td><td>65.7</td><td>67.1</td><td>70.8</td></tr><tr><td>科学</td><td>41.8</td><td>37.1</td><td>36.5</td><td>38.0</td><td>43.7</td><td>46.1</td></tr><tr><td>技术</td><td>48.5</td><td>39.4</td><td>46.8</td><td>47.3</td><td>50.8</td><td>53.6</td></tr><tr><td>生活方式</td><td>73.0</td><td>60.6</td><td>73.1</td><td>73.7</td><td>74.0</td><td>76.9</td></tr><tr><td>合并的</td><td>58.2</td><td>47.2</td><td>55.7</td><td>57.7</td><td>60.1</td><td>63.4</td></tr><tr><td colspan="7">$f(x) = \frac{1}{2\;\pi }{\int }_{0}^{\infty }\frac{f(t)}{t\;{dt}}{dt} = \frac{1}{2\;\pi }{\int }_{0}^{\infty }\frac{f(t)}{t\;{dt}}{dt}$</td></tr></tbody></table>

(b)

<!-- Media -->

### 5.2 Out-of-Domain Retrieval Quality

### 5.2 域外检索质量

Next, we evaluate ColBERTv2 outside the training domain using BEIR (Thakur et al., 2021), Wikipedia Open QA retrieval as in Khattab et al. (2021b), and LoTTE. We compare against a wide range of recent and state-of-the-art retrieval systems from the literature.

接下来，我们使用BEIR（Thakur等人，2021年）、Khattab等人（2021b）中的维基百科开放问答检索以及LoTTE，在训练领域之外评估ColBERTv2。我们将其与文献中一系列近期和最先进的检索系统进行比较。

BEIR. We start with BEIR, reporting the quality of models that do not incorporate distillation from cross-encoders, namely, ColBERT (Khattab and Zaharia, 2020), DPR-MARCO (Xin et al., 2021), ANCE (Xiong et al., 2020), and MoDIR (Xin et al., 2021), as well as models that do utilize distillation, namely, TAS-B (Hofstätter et al., 2021), SPLADEv2 (Formal et al., 2021a), and also Rock-etQAv2, which we test ourselves using the official checkpoint trained on MS MARCO. We divide the table into "search" (i.e., natural queries and questions) and "semantic relatednes" (e.g., citation-relatedness and claim verification) tasks to reflect the nature of queries in each dataset. ${}^{5}$

BEIR。我们从BEIR开始，报告未纳入交叉编码器蒸馏的模型的质量，即ColBERT（Khattab和Zaharia，2020年）、DPR - MARCO（Xin等人，2021年）、ANCE（Xiong等人，2020年）和MoDIR（Xin等人，2021年），以及利用了蒸馏的模型，即TAS - B（Hofstätter等人，2021年）、SPLADEv2（Formal等人，2021a），还有我们使用在MS MARCO上训练的官方检查点自行测试的Rock - etQAv2。我们将表格分为“搜索”（即自然查询和问题）和“语义相关性”（例如，引用相关性和声明验证）任务，以反映每个数据集中查询的性质。${}^{5}$

Table 5a reports results with the official nDCG@10 metric. Among the models without distillation, we see that the vanilla ColBERT model outperforms the single-vector systems DPR, ANCE, and MoDIR across all but three tasks. ColBERT often outpaces all three systems by large margins and, in fact, outperforms the TAS-B model, which utilizes distillation, on most datasets. Shifting our attention to models with distillation, we see a similar pattern: while distillation-based models are generally stronger than their vanilla counterparts, the models that decompose scoring into term-level interactions, ColBERTv2 and SPLADEv2, are almost always the strongest.

表5a报告了使用官方nDCG@10指标的结果。在未进行蒸馏的模型中，我们发现原始的ColBERT模型在除三个任务之外的所有任务中都优于单向量系统DPR、ANCE和MoDIR。ColBERT通常大幅领先于这三个系统，实际上，在大多数数据集上，它的表现优于利用了蒸馏的TAS - B模型。将注意力转向使用蒸馏的模型，我们看到了类似的模式：虽然基于蒸馏的模型通常比原始模型更强，但将评分分解为词项级交互的模型ColBERTv2和SPLADEv2几乎总是最强的。

Looking more closely into the comparison between SPLADEv2 and ColBERTv2, we see that ColBERTv2 has an advantage on six benchmarks and ties SPLADEv2 on two, with the largest improvements attained on NQ, TREC-COVID, and FiQA-2018, all of which feature natural search queries. On the other hand, SPLADEv2 has the lead on five benchmarks, displaying the largest gains on Climate-FEVER (C-FEVER) and Hot-PotQA. In C-FEVER, the input queries are sentences making climate-related claims and, as a result, do not reflect the typical characteristics of search queries. In HotPotQA, queries are written by crowdworkers who have access to the target passages. This is known to lead to artificial lexical bias (Lee et al., 2019), where crowdworkers copy terms from the passages into their questions as in the Open-SQuAD benchmark.

更仔细地观察SPLADEv2和ColBERTv2之间的比较，我们发现ColBERTv2在六个基准测试中具有优势，在两个基准测试中与SPLADEv2打平，在NQ、TREC - COVID和FiQA - 2018上取得了最大的改进，所有这些都以自然搜索查询为特点。另一方面，SPLADEv2在五个基准测试中领先，在Climate - FEVER（C - FEVER）和Hot - PotQA上显示出最大的增益。在C - FEVER中，输入查询是提出与气候相关声明的句子，因此不能反映搜索查询的典型特征。在HotPotQA中，查询是由能够访问目标段落的众包工作者编写的。众所周知，这会导致人为的词汇偏差（Lee等人，2019年），就像在Open - SQuAD基准测试中一样，众包工作者会将段落中的词项复制到他们的问题中。

---

<!-- Footnote -->

${}^{5}$ Following Formal et al. (2021a),we conduct our evalu-ationg using the publicly-available datasets in BEIR. Refer to $§\mathrm{E}$ for details.

${}^{5}$ 遵循Formal等人（2021a）的方法，我们使用BEIR中公开可用的数据集进行评估。详情请参考$§\mathrm{E}$。

<!-- Footnote -->

---

Wikipedia Open QA. As a further test of out-of-domain generalization, we evaluate the MS MARCO-trained ColBERTv2, SPLADEv2, and vanilla ColBERT on retrieval for open-domain question answering, similar to the out-of-domain setting of Khattab et al. (2021b). We report Success@5 (sometimes referred to as Recall@5), which is the percentage of questions whose short answer string overlaps with one or more of the top-5 passages. For the queries, we use the development set questions of the open-domain versions (Lee et al., 2019; Karpukhin et al., 2020) of Natural Questions (NQ; Kwiatkowski et al. 2019), TriviaQA (TQ; Joshi et al. 2017), and SQuAD (Ra-jpurkar et al., 2016) datasets in Table 5b. As a baseline, we include the BM25 (Robertson et al., 1995) results using the Anserini (Yang et al., 2018a) toolkit. We observe that ColBERTv2 outperforms BM25, vanilla ColBERT, and SPLADEv2 across the three query sets, with improvements of up to 4.6 points over SPLADEv2.

维基百科开放问答。作为对域外泛化能力的进一步测试，我们评估了在MS MARCO上训练的ColBERTv2、SPLADEv2和原始ColBERT在开放域问答检索方面的性能，类似于Khattab等人（2021b）的域外设置。我们报告了Success@5（有时也称为Recall@5），即短答案字符串与前5个段落中的一个或多个重叠的问题的百分比。对于查询，我们在表5b中使用自然问题（NQ；Kwiatkowski等人，2019年）、TriviaQA（TQ；Joshi等人，2017年）和SQuAD（Rajpurkar等人，2016年）数据集的开放域版本（Lee等人，2019年；Karpukhin等人，2020年）的开发集问题。作为基线，我们纳入了使用Anserini（Yang等人，2018a）工具包的BM25（Robertson等人，1995年）的结果。我们观察到，在三个查询集上，ColBERTv2的表现优于BM25、原始ColBERT和SPLADEv2，与SPLADEv2相比，提升幅度高达4.6个百分点。

LoTTE. Next, we analyze performance on the LoTTE test benchmark, which focuses on natural queries over long-tail topics and exhibits a different annotation pattern to the datasets in the previous OOD evaluations. In particular, LoTTE uses automatic Google rankings (for the "search" queries) and organic StackExchange question-answer pairs (for "forum" queries), complimenting the pooling-based annotation of datasets like TREC-COVID (in BEIR) and the answer overlap metrics of Open-QA retrieval. We report Success@5 for each corpus on both search queries and forum queries.

乐天（LoTTE）。接下来，我们分析在乐天测试基准上的性能，该基准侧重于长尾主题的自然查询，并且与之前分布外（OOD）评估中的数据集呈现出不同的标注模式。具体而言，乐天使用谷歌自动排名（针对“搜索”查询）和StackExchange上自然产生的问答对（针对“论坛”查询），对像TREC - COVID（BEIR中的数据集）基于池化的标注以及开放问答（Open - QA）检索的答案重叠指标进行补充。我们报告了每个语料库在搜索查询和论坛查询上的前5名成功率（Success@5）。

Overall, we see that ANCE and vanilla ColBERT outperform BM25 on all topics, and that the three methods using distillation are generally the strongest. Similar to the Wikipedia-OpenQA results, we find that ColBERTv2 outperforms the baselines across all topics for both query types, improving upon SPLADEv2 and RocketQAv2 by up to 3.7 and 8.1 points, respectively. Considering the baselines, we observe that while RocketQAv2 tends to have a slight advantage over SPLADEv2 on the "search" queries, SPLADEv2 is considerably more effective on the "forum" tests. We hypothesize that the search queries, obtained from Google (through GooAQ) are more similar to MS MARCO than the forum queries and, as a result, the latter stresses generalization more heavily, rewarding term-decomposed models like SPLADEv2 and ColBERTv2.

总体而言，我们发现ANCE和普通的ColBERT在所有主题上的表现都优于BM25，并且使用蒸馏的三种方法总体上表现最强。与维基百科开放问答（Wikipedia - OpenQA）的结果类似，我们发现ColBERTv2在两种查询类型的所有主题上都优于基线模型，分别比SPLADEv2和RocketQAv2提高了多达3.7和8.1个百分点。考虑到基线模型，我们观察到，虽然RocketQAv2在“搜索”查询上往往比SPLADEv2略有优势，但SPLADEv2在“论坛”测试中明显更有效。我们假设，从谷歌（通过GooAQ）获得的搜索查询比论坛查询更类似于MS MARCO，因此，后者更强调泛化能力，使像SPLADEv2和ColBERTv2这样的词项分解模型更具优势。

### 5.3 Efficiency

### 5.3 效率

ColBERTv2's residual compression approach significantly reduces index sizes compared to vanilla ColBERT. Whereas ColBERT requires 154 GiB to store the index for MS MARCO, ColBERTv2 only requires ${16}\mathrm{{GiB}}$ or ${25}\mathrm{{GiB}}$ when compressing embeddings to 1 or 2 bit(s) per dimension, respectively,resulting in compression ratios of $6 - {10} \times$ . This storage figure includes ${4.5}\mathrm{{GiB}}$ for storing the inverted list.

与普通的ColBERT相比，ColBERTv2的残差压缩方法显著减小了索引大小。普通的ColBERT存储MS MARCO的索引需要154 GiB，而ColBERTv2在将嵌入压缩到每维1位或2位时，分别只需要${16}\mathrm{{GiB}}$或${25}\mathrm{{GiB}}$，压缩比达到$6 - {10} \times$。这个存储量包括用于存储倒排表的${4.5}\mathrm{{GiB}}$。

This matches the storage for a typical single-vector model on MS MARCO, with 4-byte lossless floating-point storage for one 768-dimensional vector for each of the $9\mathrm{M}$ passages amounting to a little over 25 GiBs. In practice, the storage for a single-vector model could be even larger when using a nearest-neighbor index like HNSW for fast search. Conversely, single-vector representations could be themselves compressed very aggressively (Zhan et al., 2021a, 2022), though often exacerbating the loss in quality relative to late interaction methods like ColBERTv2.

这与MS MARCO上典型的单向量模型的存储量相当，对于$9\mathrm{M}$个段落，每个段落用一个768维的向量进行4字节无损浮点存储，总量略超过25 GiB。实际上，当使用像HNSW这样的近邻索引进行快速搜索时，单向量模型的存储量可能会更大。相反，单向量表示本身可以被非常激进地压缩（Zhan等人，2021a，2022），不过相对于ColBERTv2这样的后期交互方法，往往会加剧质量损失。

We discuss the impact of our compression method on search quality in Appendix B and present query latency results on the order of ${50} -$ 250 milliseconds per query in Appendix C.

我们在附录B中讨论了我们的压缩方法对搜索质量的影响，并在附录C中给出了每个查询约${50} -$ 250毫秒的查询延迟结果。

## 6 Conclusion

## 6 结论

We introduced ColBERTv2, a retriever that advances the quality and space efficiency of multi-vector representations. We hypothesized that cluster centroids capture context-aware semantics of the token-level representations and proposed a residual representation that leverages these patterns to dramatically reduce the footprint of multi-vector systems off-the-shelf. We then explored improved supervision for multi-vector retrieval and found that their quality improves considerably upon distillation from a cross-encoder system. The proposed ColBERTv2 considerably outperforms existing retrievers in within-domain and out-of-domain evaluations, which we conducted extensively across 28 datasets, establishing state-of-the-art quality while exhibiting competitive space footprint.

我们介绍了ColBERTv2，这是一种提高了多向量表示质量和空间效率的检索器。我们假设聚类中心捕获了词元级表示的上下文感知语义，并提出了一种利用这些模式的残差表示，以显著减少现成多向量系统的占用空间。然后，我们探索了对多向量检索的改进监督，发现从交叉编码器系统进行蒸馏后，其质量有了显著提高。我们在28个数据集上进行了广泛的评估，结果表明，所提出的ColBERTv2在域内和域外评估中明显优于现有的检索器，在展现出有竞争力的空间占用的同时，达到了当前最优的质量。

## Acknowledgements

## 致谢

This research was supported in part by affiliate members and other supporters of the Stanford DAWN project-Ant Financial, Facebook, Google, and VMware-as well as Cisco, SAP, Virtusa, and the NSF under CAREER grant CNS-1651570. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation.

这项研究部分得到了斯坦福黎明（DAWN）项目的附属成员和其他支持者的资助，包括蚂蚁金服、Facebook、谷歌和VMware，以及思科、SAP、Virtusa和美国国家科学基金会（NSF）的职业发展资助（CAREER grant CNS - 1651570）。本材料中表达的任何观点、研究结果、结论或建议均为作者本人的观点，不一定反映美国国家科学基金会的观点。

## Broader Impact & Ethical Considerations

## 更广泛的影响与伦理考量

This work is primarily an effort toward retrieval models that generalize better while performing reasonably efficiently in terms of space consumption. Strong out-of-the-box generalization to small domain-specific applications can serve many users in practice, particularly where training data is not available. Moreover, retrieval holds significant promise for many downstream NLP tasks, as it can help make language models smaller and thus more efficient (i.e., by decoupling knowledge from computation), more transparent (i.e., by allowing users to check the sources the model relied on when making a claim or prediction), and easier to update (i.e., by allowing developers to replace or add documents to the corpus without retraining the model) (Guu et al., 2020; Borgeaud et al., 2021; Khattab et al., 2021a). Nonetheless, such work poses risks in terms of misuse, particularly toward misinformation, as retrieval can surface results that are relevant yet inaccurate, depending on the contents of a corpus. Moreover, generalization from training on a large-scale dataset can propagate the biases of that dataset well beyond its typical reach to new domains and applications.

这项工作主要致力于开发检索模型，使其在空间消耗方面表现合理高效的同时，具有更好的泛化能力。对于特定领域的小型应用，强大的开箱即用泛化能力在实践中可以服务于众多用户，特别是在没有可用训练数据的情况下。此外，检索在许多下游自然语言处理（NLP）任务中具有巨大潜力，因为它可以帮助缩小语言模型的规模，从而提高效率（即通过将知识与计算分离），增强透明度（即允许用户检查模型在进行声明或预测时所依赖的来源），并便于更新（即允许开发人员在不重新训练模型的情况下替换或添加语料库中的文档）（Guu等人，2020年；Borgeaud等人，2021年；Khattab等人，2021a）。尽管如此，这项工作在滥用方面存在风险，特别是在传播错误信息方面，因为根据语料库的内容，检索可能会呈现出相关但不准确的结果。此外，在大规模数据集上进行训练所实现的泛化，可能会将该数据集的偏差传播到远超其典型适用范围的新领域和新应用中。

While our contributions have made ColBERT's late interaction more efficient at storage costs, large-scale distillation with hard negatives increases system complexity and accordingly increases training cost, when compared with the straightforward training paradigm of the original ColBERT model. While ColBERTv2 is efficient in terms of latency and storage at inference time, we suspect that under extreme resource constraints, simpler model designs like SPLADEv2 or RocketQAv2 could lend themselves to easier-to-optimize environments. We leave low-level systems optimizations of all systems to future work. Another worthwhile dimension for future exploration of tradeoffs is re-ranking architectures over various systems with cross-encoders, which are known to be expensive yet precise due to their highly expressive capacity.

虽然我们的研究成果使ColBERT的后期交互在存储成本方面更加高效，但与原始ColBERT模型直接的训练范式相比，使用难负样本进行大规模蒸馏会增加系统的复杂性，相应地也会提高训练成本。虽然ColBERTv2在推理时的延迟和存储方面表现高效，但我们怀疑在极端资源限制下，像SPLADEv2或RocketQAv2这样更简单的模型设计可能更适合易于优化的环境。我们将所有系统的底层系统优化工作留待未来研究。未来值得探索的另一个权衡维度是，在各种系统上使用交叉编码器进行重排序架构的研究，由于交叉编码器具有高度的表达能力，虽然成本高昂，但精度较高。

## Research Limitations

## 研究局限性

While we evaluate ColBERTv2 on a wide range of tests, all of our benchmarks are in English and, in line with related work, our out-of-domain tests evaluate models that are trained on MS MARCO. We expect our approach to work effectively for other languages and when all models are trained using other, smaller training set (e.g., NaturalQuestions), but we leave such tests to future work.

虽然我们在广泛的测试中对ColBERTv2进行了评估，但我们所有的基准测试均使用英语，并且与相关研究一致，我们的跨领域测试评估的是在MS MARCO数据集上训练的模型。我们预计我们的方法在其他语言以及所有模型使用其他较小的训练集（如NaturalQuestions）进行训练时也能有效工作，但我们将此类测试留待未来研究。

We have observed consistent gains for Col-BERTv2 against existing state-of-the-art systems across many diverse settings. Despite this, almost all IR datasets contain false negatives (i.e., relevant but unlabeled passages) and thus some caution is needed in interpreting any individual result. Nonetheless, we intentionally sought out benchmarks with dissimilar annotation biases: for instance, TREC-COVID (in BEIR) annotates the pool of documents retrieved by the systems submitted at the time of the competition, LoTTE uses automatic Google rankings (for "search" queries) and StackExchange question-answer pairs (for "forum" queries), and the Open-QA tests rely on passage-answer overlap for factoid questions. ColBERTv2 performed well in all of these settings. We discuss other issues pertinent to LoTTE in Appendix $\$ \mathrm{D}$ .

我们观察到，在许多不同的设置下，ColBERTv2相对于现有的最先进系统都有持续的性能提升。尽管如此，几乎所有的信息检索（IR）数据集都包含假阴性样本（即相关但未标记的段落），因此在解释任何单个结果时需要谨慎。尽管如此，我们有意选择了具有不同标注偏差的基准测试：例如，TREC - COVID（BEIR中的数据集）对竞赛当时提交的系统所检索到的文档池进行标注，LoTTE使用谷歌的自动排名（针对“搜索”查询）和StackExchange问答对（针对“论坛”查询），而开放式问答测试则依赖于事实类问题的段落 - 答案重叠度。ColBERTv2在所有这些设置中都表现良好。我们在附录$\$ \mathrm{D}$中讨论了与LoTTE相关的其他问题。

We have compared with a wide range of strong baselines—including sparse retrieval and single-vector models-and found reliable patterns across tests. However, we caution that empirical trends can change as innovations are introduced to each of these families of models and that it can be difficult to ensure exact apple-to-apple comparisons across families of models, since each of them calls for different sophisticated tuning strategies. We thus primarily used results and models from the rich recent literature on these problems, with models like RocketQAv2 and SPLADEv2.

我们与广泛的强大基线模型进行了比较，包括稀疏检索模型和单向量模型，并在测试中发现了可靠的模式。然而，我们提醒，随着这些模型家族中不断引入创新，实证趋势可能会发生变化，而且由于每个模型家族都需要不同的复杂调优策略，因此很难确保跨模型家族进行完全对等的比较。因此，我们主要使用了近期关于这些问题的丰富文献中的结果和模型，如RocketQAv2和SPLADEv2。

On the representational side, we focus on reducing the storage cost using residual compression, achieving strong gains in reducing footprint while largely preserving quality. Nonetheless, we have not exhausted the space of more sophisticated optimizations possible, and we would expect more sophisticated forms of residual compression and composing our approach with dropping tokens (Zhou and Devlin, 2021) to open up possibilities for further reductions in space footprint.

在表示方面，我们专注于使用残差压缩来降低存储成本，在大幅保留质量的同时，显著减少了模型占用空间。尽管如此，我们尚未穷尽更复杂优化的可能性，我们预计更复杂形式的残差压缩以及将我们的方法与词元丢弃策略（Zhou和Devlin，2021年）相结合，将为进一步减少空间占用开辟新的可能性。

## References

## 参考文献

## Stack Exchange Data Dump.

## Stack Exchange数据转储。

Liefu Ai, Junqing Yu, Zebin Wu, Yunfeng He, and Tao Guan. 2017. Optimized Residual Vector Quantization for Efficient Approximate Nearest Neighbor Search. Multimedia Systems, 23(2):169-181.

艾立富、余俊清、吴泽斌、何云峰和管涛。2017年。用于高效近似最近邻搜索的优化残差向量量化。《多媒体系统》，23(2):169 - 181。

Sören Auer, Christian Bizer, Georgi Kobilarov, Jens Lehmann, Richard Cyganiak, and Zachary Ives. 2007. DBpedia: A Nucleus for a Web of Open Data. In The semantic web, pages 722-735. Springer.

索伦·奥尔、克里斯蒂安·比泽、格奥尔基·科比拉罗夫、延斯·莱曼、理查德·西加尼亚克和扎卡里·艾夫斯。2007年。DBpedia：开放数据网络的核心。见《语义网》，第722 - 735页。施普林格出版社。

Christopher F Barnes, Syed A Rizvi, and Nasser M Nasrabadi. 1996. Advances in Residual Vector Quantization: A Review. IEEE transactions on image processing, 5(2):226-262.

克里斯托弗·F·巴恩斯、赛义德·A·里兹维和纳赛尔·M·纳斯拉巴迪。1996年。残差向量量化的进展：综述。《IEEE图像处理汇刊》，5(2):226 - 262。

Alexander Bondarenko, Maik Fröbe, Meriem Be-loucif, Lukas Gienapp, Yamen Ajjour, Alexander Panchenko, Chris Biemann, Benno Stein, Henning Wachsmuth, Martin Potthast, et al. 2020. Overview of touché 2020: Argument Retrieval. In International Conference of the Cross-Language Evaluation Forum for European Languages, pages 384- 395. Springer.

亚历山大·邦达连科（Alexander Bondarenko）、迈克·弗勒贝（Maik Fröbe）、梅里姆·贝卢西夫（Meriem Be-loucif）、卢卡斯·吉纳普（Lukas Gienapp）、亚门·阿乔尔（Yamen Ajjour）、亚历山大·潘琴科（Alexander Panchenko）、克里斯·比曼（Chris Biemann）、本诺·施泰因（Benno Stein）、亨宁·瓦赫斯穆特（Henning Wachsmuth）、马丁·波塔斯塔（Martin Potthast）等。2020年。2020年Touché概述：论点检索。见《欧洲语言跨语言评估论坛国际会议》，第384 - 395页。施普林格出版社。

Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George van den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, et al. 2021. Improving language models by retrieving from trillions of tokens. arXiv preprint arXiv:2112.04426.

塞巴斯蒂安·博尔若（Sebastian Borgeaud）、亚瑟·门施（Arthur Mensch）、乔丹·霍夫曼（Jordan Hoffmann）、特雷弗·蔡（Trevor Cai）、伊丽莎·拉瑟福德（Eliza Rutherford）、凯蒂·米利坎（Katie Millican）、乔治·范登德里舍（George van den Driessche）、让 - 巴蒂斯特·莱斯皮奥（Jean - Baptiste Lespiau）、博格丹·达莫克（Bogdan Damoc）、艾丹·克拉克（Aidan Clark）等。2021年。通过从数万亿个标记中检索来改进语言模型。预印本arXiv:2112.04426。

Vera Boteva, Demian Gholipour, Artem Sokolov, and Stefan Riezler. 2016. A Full-text Learning to Rank Dataset for Medical Information Retrieval. In ${Eu}$ - ropean Conference on Information Retrieval, pages 716-722. Springer.

维拉·博特娃（Vera Boteva）、德米安·戈利普尔（Demian Gholipour）、阿尔乔姆·索科洛夫（Artem Sokolov）和斯特凡·里兹勒（Stefan Riezler）。2016年。用于医学信息检索的全文学习排序数据集。见《欧洲信息检索会议》，第716 - 722页。施普林格出版社。

Chia-Yu Chen, Jungwook Choi, Daniel Brand, Ankur Agrawal, Wei Zhang, and Kailash Gopalakrishnan. 2018. Adacomp : Adaptive residual gradient compression for data-parallel distributed training. In Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence, (AAAI-18), the 30th innovative Applications of Artificial Intelligence (IAAI- 18), and the 8th AAAI Symposium on Educational Advances in Artificial Intelligence (EAAI-18), New Orleans, Louisiana, USA, February 2-7, 2018, pages 2827-2835. AAAI Press.

陈家宇（Chia - Yu Chen）、崔正旭（Jungwook Choi）、丹尼尔·布兰德（Daniel Brand）、安库尔·阿格拉瓦尔（Ankur Agrawal）、张伟（Wei Zhang）和凯拉什·戈帕拉克里什南（Kailash Gopalakrishnan）。2018年。Adacomp：用于数据并行分布式训练的自适应残差梯度压缩。见《第三十二届AAAI人工智能会议（AAAI - 18）、第三十届人工智能创新应用会议（IAAI - 18）和第八届AAAI人工智能教育进展研讨会（EAAI - 18）论文集》，美国路易斯安那州新奥尔良，2018年2月2 - 7日，第2827 - 2835页。AAAI出版社。

Arman Cohan, Sergey Feldman, Iz Beltagy, Doug Downey, and Daniel Weld. 2020. SPECTER: Document-level representation learning using citation-informed transformers. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 2270-2282, Online. Association for Computational Linguistics.

阿尔曼·科汉（Arman Cohan）、谢尔盖·费尔德曼（Sergey Feldman）、伊兹·贝尔塔吉（Iz Beltagy）、道格·唐尼（Doug Downey）和丹尼尔·韦尔德（Daniel Weld）。2020年。SPECTER：使用引用感知的Transformer进行文档级表示学习。见《第五十八届计算语言学协会年会论文集》，第2270 - 2282页，线上会议。计算语言学协会。

Nachshon Cohen, Amit Portnoy, Besnik Fetahu, and Amir Ingber. 2021. SDR: Efficient Neural Re-ranking using Succinct Document Representation. arXiv preprint arXiv:2110.02065.

纳赫雄·科恩（Nachshon Cohen）、阿米特·波特诺伊（Amit Portnoy）、贝斯尼克·费塔胡（Besnik Fetahu）和阿米尔·英格伯（Amir Ingber）。2021年。SDR：使用简洁文档表示的高效神经重排序。预印本arXiv:2110.02065。

Zhuyun Dai and Jamie Callan. 2020. Context-aware term weighting for first stage passage retrieval. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, SIGIR 2020, Virtual Event, China, July 25-30, 2020, pages 1533-1536. ACM.

戴竹云（Zhuyun Dai）和杰米·卡兰（Jamie Callan）。2020年。用于第一阶段段落检索的上下文感知词项加权。见《第四十三届ACM SIGIR信息检索研究与发展国际会议（SIGIR 2020）论文集》，虚拟会议，中国，2020年7月25 - 30日，第1533 - 1536页。美国计算机协会。

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171-4186, Minneapolis, Minnesota. Association for Computational Linguistics.

雅各布·德夫林（Jacob Devlin）、张明伟（Ming - Wei Chang）、肯顿·李（Kenton Lee）和克里斯蒂娜·图托纳娃（Kristina Toutanova）。2019年。BERT：用于语言理解的深度双向Transformer预训练。见《2019年北美计算语言学协会人类语言技术会议论文集》，第1卷（长论文和短论文），第4171 - 4186页，美国明尼苏达州明尼阿波利斯。计算语言学协会。

Thomas Diggelmann, Jordan Boyd-Graber, Jannis Bu-lian, Massimiliano Ciaramita, and Markus Leippold. 2020. CLIMATE-FEVER: A Dataset for Verification of Real-World Climate Claims. arXiv preprint arXiv:2012.00614.

托马斯·迪格尔曼（Thomas Diggelmann）、乔丹·博伊德 - 格拉伯（Jordan Boyd - Graber）、扬尼斯·比利安（Jannis Bulián）、马西米利亚诺·恰拉米塔（Massimiliano Ciaramita）和马库斯·莱波尔德（Markus Leippold）。2020年。CLIMATE - FEVER：用于验证现实世界气候主张的数据集。预印本arXiv:2012.00614。

Thibault Formal, Carlos Lassance, Benjamin Pi-wowarski, and Stéphane Clinchant. 2021a. SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval. arXiv preprint arXiv:2109.10086.

蒂博·福尔马尔（Thibault Formal）、卡洛斯·拉萨斯（Carlos Lassance）、本杰明·皮沃瓦尔斯基（Benjamin Piwowarski）和斯特凡·克林尚（Stéphane Clinchant）。2021a。SPLADE v2：用于信息检索的稀疏词法和扩展模型。预印本arXiv:2109.10086。

Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant. 2021b. SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 2288-2292.

蒂博·福尔马尔（Thibault Formal）、本杰明·皮沃瓦尔斯基（Benjamin Piwowarski）和斯特凡·克林尚（Stéphane Clinchant）。2021b。SPLADE：用于第一阶段排序的稀疏词法和扩展模型。见《第四十四届ACM SIGIR信息检索研究与发展国际会议论文集》，第2288 - 2292页。

Luyu Gao and Jamie Callan. 2021. Unsupervised corpus aware language model pre-training for dense passage retrieval. arXiv preprint arXiv:2108.05540.

高宇（Luyu Gao）和杰米·卡兰（Jamie Callan）。2021年。用于密集段落检索的无监督语料库感知语言模型预训练。预印本arXiv:2108.05540。

Luyu Gao, Zhuyun Dai, and Jamie Callan. 2020. Modularized transfomer-based ranking framework. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4180-4190, Online. Association for Computational Linguistics.

高宇（Luyu Gao）、戴竹云（Zhuyun Dai）和杰米·卡兰（Jamie Callan）。2020年。基于模块化Transformer的排序框架。见《2020年自然语言处理经验方法会议（EMNLP）论文集》，第4180 - 4190页，线上会议。计算语言学协会。

Luyu Gao, Zhuyun Dai, and Jamie Callan. 2021. COIL: Revisit exact lexical match in information retrieval with contextualized inverted list. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3030-3042, Online. Association for Computational Linguistics.

高璐宇（Luyu Gao）、戴竹云（Zhuyun Dai）和杰米·卡兰（Jamie Callan）。2021年。COIL：通过上下文倒排列表重新审视信息检索中的精确词汇匹配。见《2021年计算语言学协会北美分会人类语言技术会议论文集》，第3030 - 3042页，线上会议。计算语言学协会。

Robert Gray. 1984. Vector quantization. IEEE Assp Magazine, 1(2):4-29.

罗伯特·格雷（Robert Gray）。1984年。矢量量化。《IEEE声学、语音和信号处理杂志》，1(2):4 - 29。

Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasu-pat, and Ming-Wei Chang. 2020. Realm: Retrieval-augmented language model pre-training. arXiv preprint arXiv:2002.08909.

凯文·古（Kelvin Guu）、肯顿·李（Kenton Lee）、佐拉·通（Zora Tung）、帕努蓬·帕苏帕特（Panupong Pasu - pat）和张明伟（Ming - Wei Chang）。2020年。Realm：检索增强的语言模型预训练。arXiv预印本arXiv:2002.08909。

Sebastian Hofstätter, Sophia Althammer, Michael Schröder, Mete Sertkan, and Allan Hanbury. 2020. Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation. arXiv preprint arXiv:2010.02666.

塞巴斯蒂安·霍夫施泰特（Sebastian Hofstätter）、索菲亚·阿尔塔默（Sophia Althammer）、迈克尔·施罗德（Michael Schröder）、梅特·塞尔特坎（Mete Sertkan）和艾伦·汉伯里（Allan Hanbury）。2020年。通过跨架构知识蒸馏改进高效神经排序模型。arXiv预印本arXiv:2010.02666。

Sebastian Hofstätter, Sheng-Chieh Lin, Jheng-Hong Yang, Jimmy Lin, and Allan Hanbury. 2021. Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling. arXiv preprint arXiv:2104.06967.

塞巴斯蒂安·霍夫施泰特（Sebastian Hofstätter）、林圣杰（Sheng - Chieh Lin）、杨政宏（Jheng - Hong Yang）、吉米·林（Jimmy Lin）和艾伦·汉伯里（Allan Hanbury）。2021年。通过平衡主题感知采样高效训练有效的密集检索器。arXiv预印本arXiv:2104.06967。

Samuel Humeau, Kurt Shuster, Marie-Anne Lachaux, and Jason Weston. 2020. Poly-encoders: Architectures and pre-training strategies for fast and accurate multi-sentence scoring. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenRe-view.net.

塞缪尔·休莫（Samuel Humeau）、库尔特·舒斯特（Kurt Shuster）、玛丽 - 安妮·拉肖（Marie - Anne Lachaux）和杰森·韦斯顿（Jason Weston）。2020年。多编码器：用于快速准确多句子评分的架构和预训练策略。见《第8届学习表征国际会议（ICLR 2020）》，2020年4月26 - 30日，埃塞俄比亚亚的斯亚贝巴。OpenReview.net。

Gautier Izacard, Fabio Petroni, Lucas Hosseini, Nicola De Cao, Sebastian Riedel, and Edouard Grave. 2020. A memory efficient baseline for open domain question answering. arXiv preprint arXiv:2012.15156.

高蒂尔·伊扎卡尔（Gautier Izacard）、法比奥·彼得罗尼（Fabio Petroni）、卢卡斯·侯赛尼（Lucas Hosseini）、尼古拉·德·曹（Nicola De Cao）、塞巴斯蒂安·里德尔（Sebastian Riedel）和爱德华·格雷夫（Edouard Grave）。2020年。开放域问答的内存高效基线。arXiv预印本arXiv:2012.15156。

Herve Jegou, Matthijs Douze, and Cordelia Schmid. 2010. Product quantization for nearest neighbor search. IEEE transactions on pattern analysis and machine intelligence, 33(1):117-128.

埃尔韦·热古（Herve Jegou）、马蒂亚斯·杜泽（Matthijs Douze）和科迪莉亚·施密德（Cordelia Schmid）。2010年。用于最近邻搜索的乘积量化。《IEEE模式分析与机器智能汇刊》，33(1):117 - 128。

Yichen Jiang, Shikha Bordia, Zheng Zhong, Charles Dognin, Maneesh Singh, and Mohit Bansal. 2020. HoVer: A dataset for many-hop fact extraction and claim verification. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 3441-3460, Online. Association for Computational Linguistics.

蒋逸尘（Yichen Jiang）、希卡·博尔迪亚（Shikha Bordia）、钟政（Zheng Zhong）、查尔斯·多尼恩（Charles Dognin）、马内什·辛格（Maneesh Singh）和莫希特·班萨尔（Mohit Bansal）。2020年。HoVer：用于多跳事实提取和声明验证的数据集。见《计算语言学协会研究成果：2020年自然语言处理经验方法会议（EMNLP 2020）》，第3441 - 3460页，线上会议。计算语言学协会。

Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-scale similarity search with gpus. IEEE Transactions on Big Data.

杰夫·约翰逊（Jeff Johnson）、马蒂亚斯·杜泽（Matthijs Douze）和埃尔韦·热古（Hervé Jégou）。2019年。基于GPU的十亿级相似度搜索。《IEEE大数据汇刊》。

Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. 2017. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1601-1611, Vancouver, Canada. Association for Computational Linguistics.

曼达尔·乔希（Mandar Joshi）、崔恩索尔（Eunsol Choi）、丹尼尔·韦尔德（Daniel Weld）和卢克·泽特尔莫耶（Luke Zettlemoyer）。2017年。TriviaQA：用于阅读理解的大规模远程监督挑战数据集。见《计算语言学协会第55届年会论文集（第1卷：长论文）》，第1601 - 1611页，加拿大温哥华。计算语言学协会。

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6769- 6781, Online. Association for Computational Linguistics.

弗拉基米尔·卡尔普欣（Vladimir Karpukhin）、巴拉斯·奥古兹（Barlas Oguz）、闵世元（Sewon Min）、帕特里克·刘易斯（Patrick Lewis）、莱德尔·吴（Ledell Wu）、谢尔盖·叶杜诺夫（Sergey Edunov）、陈丹琦（Danqi Chen）和易文涛（Wen - tau Yih）。2020年。开放域问答的密集段落检索。见《2020年自然语言处理经验方法会议（EMNLP）论文集》，第6769 - 6781页，线上会议。计算语言学协会。

Daniel Khashabi, Amos Ng, Tushar Khot, Ashish Sab-harwal, Hannaneh Hajishirzi, and Chris Callison-Burch. 2021. GooAQ: Open Question Answering with Diverse Answer Types. arXiv preprint arXiv:2104.08727.

丹尼尔·哈沙比（Daniel Khashabi）、阿莫斯·吴（Amos Ng）、图沙尔·霍特（Tushar Khot）、阿什什·萨巴瓦尔（Ashish Sab - harwal）、汉娜内·哈吉希尔齐（Hannaneh Hajishirzi）和克里斯·卡利森 - 伯奇（Chris Callison - Burch）。2021年。GooAQ：具有多样化答案类型的开放问答。arXiv预印本arXiv:2104.08727。

Omar Khattab, Christopher Potts, and Matei Zaharia. 2021a. Baleen: Robust Multi-Hop Reasoning at Scale via Condensed Retrieval. In Thirty-Fifth Conference on Neural Information Processing Systems.

奥马尔·哈塔卜（Omar Khattab）、克里斯托弗·波茨（Christopher Potts）和马特·扎哈里亚（Matei Zaharia）。2021a。Baleen：通过压缩检索实现大规模鲁棒多跳推理。见《第三十五届神经信息处理系统会议》。

Omar Khattab, Christopher Potts, and Matei Zaharia. 2021b. Relevance-guided supervision for openqa with ColBERT. Transactions of the Association for Computational Linguistics, 9:929-944.

奥马尔·哈塔卜（Omar Khattab）、克里斯托弗·波茨（Christopher Potts）和马特·扎哈里亚（Matei Zaharia）。2021b。使用ColBERT为开放问答提供相关性引导监督。《计算语言学协会汇刊》，9:929 - 944。

Omar Khattab and Matei Zaharia. 2020. Colbert: Efficient and effective passage search via contextual-ized late interaction over BERT. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, SI-GIR 2020, Virtual Event, China, July 25-30, 2020, pages 39-48. ACM.

奥马尔·哈塔卜（Omar Khattab）和马特·扎哈里亚（Matei Zaharia）。2020年。ColBERT：通过基于BERT的上下文后期交互实现高效有效的段落搜索。见《第43届ACM SIGIR信息检索研究与发展国际会议（SIGIR 2020）论文集》，2020年7月25 - 30日，中国线上会议，第39 - 48页。美国计算机协会。

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural questions: A benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:452-466.

汤姆·夸特科夫斯基（Tom Kwiatkowski）、珍妮玛丽亚·帕洛马基（Jennimaria Palomaki）、奥利维亚·雷德菲尔德（Olivia Redfield）、迈克尔·柯林斯（Michael Collins）、安库尔·帕里克（Ankur Parikh）、克里斯·阿尔伯蒂（Chris Alberti）、丹妮尔·爱泼斯坦（Danielle Epstein）、伊利亚·波洛苏金（Illia Polosukhin）、雅各布·德夫林（Jacob Devlin）、肯顿·李（Kenton Lee）、克里斯蒂娜·图托纳娃（Kristina Toutanova）、利昂·琼斯（Llion Jones）、马修·凯尔西（Matthew Kelcey）、张明伟（Ming-Wei Chang）、安德鲁·M·戴（Andrew M. Dai）、雅各布·乌兹科雷特（Jakob Uszkoreit）、乐存（Quoc Le）和斯拉夫·彼得罗夫（Slav Petrov）。2019年。自然问题：问答研究的基准。《计算语言学协会汇刊》，7：452 - 466。

Jinhyuk Lee, Mujeen Sung, Jaewoo Kang, and Danqi Chen. 2021a. Learning dense representations of phrases at scale. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 6634-6647, Online. Association for Computational Linguistics.

李镇赫（Jinhyuk Lee）、成武珍（Mujeen Sung）、姜在宇（Jaewoo Kang）和陈丹琦（Danqi Chen）。2021a。大规模学习短语的密集表示。《计算语言学协会第59届年会和第11届自然语言处理国际联合会议论文集（第1卷：长论文）》，第6634 - 6647页，线上会议。计算语言学协会。

Jinhyuk Lee, Alexander Wettig, and Danqi Chen. 2021b. Phrase retrieval learns passage retrieval, too. arXiv preprint arXiv:2109.08133.

李镇赫（Jinhyuk Lee）、亚历山大·韦蒂格（Alexander Wettig）和陈丹琦（Danqi Chen）。2021b。短语检索也能学习段落检索。预印本arXiv：2109.08133。

Kenton Lee, Ming-Wei Chang, and Kristina Toutanova. 2019. Latent retrieval for weakly supervised open domain question answering. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 6086-6096, Florence, Italy. Association for Computational Linguistics.

肯顿·李（Kenton Lee）、张明伟（Ming-Wei Chang）和克里斯蒂娜·图托纳娃（Kristina Toutanova）。2019年。弱监督开放域问答的潜在检索。《计算语言学协会第57届年会论文集》，第6086 - 6096页，意大利佛罗伦萨。计算语言学协会。

Yue Li, Wenrui Ding, Chunlei Liu, Baochang Zhang, and Guodong Guo. 2021a. TRQ: Ternary Neural Networks With Residual Quantization. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 8538-8546.

李悦、丁文瑞、刘春雷、张宝昌和郭国栋。2021a。TRQ：具有残差量化的三值神经网络。《AAAI人工智能会议论文集》，第35卷，第8538 - 8546页。

Zefan Li, Bingbing Ni, Teng Li, Xiaokang Yang, Wen-jun Zhang, and Wen Gao. 2021b. Residual Quantization for Low Bit-width Neural Networks. IEEE Transactions on Multimedia.

李泽凡、倪冰冰、李腾、杨小康、张文军和高文。2021b。低比特宽度神经网络的残差量化。《IEEE多媒体汇刊》。

Jimmy Lin and Xueguang Ma. 2021. A Few Brief Notes on DeepImpact, COIL, and a Conceptual Framework for Information Retrieval Techniques. arXiv preprint arXiv:2106.14807.

林吉米（Jimmy Lin）和马雪光（Xueguang Ma）。2021年。关于DeepImpact、COIL以及信息检索技术概念框架的几点简要说明。预印本arXiv：2106.14807。

Sheng-Chieh Lin, Jheng-Hong Yang, and Jimmy Lin.

林圣杰（Sheng-Chieh Lin）、杨政宏（Jheng-Hong Yang）和林吉米（Jimmy Lin）。

2020. Distilling Dense Representations for Ranking using Tightly-Coupled Teachers. arXiv preprint arXiv:2010.11386.

2020年。使用紧密耦合教师蒸馏用于排序的密集表示。预印本arXiv：2010.11386。

Xiaorui Liu, Yao Li, Jiliang Tang, and Ming Yan. 2020. A double residual compression algorithm for efficient distributed learning. In The 23rd International Conference on Artificial Intelligence and Statistics, AISTATS 2020, 26-28 August 2020, Online [Palermo, Sicily, Italy], volume 108 of Proceedings of Machine Learning Research, pages 133-143. PMLR.

刘晓瑞、李瑶、唐继良和严明。2020年。一种用于高效分布式学习的双残差压缩算法。《第23届人工智能与统计国际会议（AISTATS 2020）论文集》，2020年8月26 - 28日，线上会议[意大利西西里岛巴勒莫]，《机器学习研究会议录》第108卷，第133 - 143页。机器学习研究会议录（PMLR）。

Yi Luan, Jacob Eisenstein, Kristina Toutanova, and Michael Collins. 2021. Sparse, Dense, and Attentional Representations for Text Retrieval. Transactions of the Association for Computational Linguistics, 9:329-345.

栾义、雅各布·艾森斯坦（Jacob Eisenstein）、克里斯蒂娜·图托纳娃（Kristina Toutanova）和迈克尔·柯林斯（Michael Collins）。2021年。用于文本检索的稀疏、密集和注意力表示。《计算语言学协会汇刊》，9：329 - 345。

Sean MacAvaney, Franco Maria Nardini, Raffaele Perego, Nicola Tonellotto, Nazli Goharian, and Ophir Frieder. 2020. Efficient document re-ranking for transformers by precomputing term representations. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, SIGIR 2020, Virtual Event, China, July 25-30, 2020, pages 49-58. ACM.

肖恩·麦卡瓦尼（Sean MacAvaney）、佛朗哥·玛丽亚·纳尔迪尼（Franco Maria Nardini）、拉斐尔·佩雷戈（Raffaele Perego）、尼古拉·托内洛托（Nicola Tonellotto）、纳兹利·戈哈瑞安（Nazli Goharian）和奥菲尔·弗里德（Ophir Frieder）。2020年。通过预计算词项表示实现变压器的高效文档重排序。《第43届ACM信息检索研究与发展国际会议（SIGIR 2020）论文集》，虚拟会议，中国，2020年7月25 - 30日，第49 - 58页。美国计算机协会（ACM）。

Craig Macdonald and Nicola Tonellotto. 2021. On approximate nearest neighbour selection for multistage dense retrieval. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management, pages 3318-3322.

克雷格·麦克唐纳（Craig Macdonald）和尼古拉·托内洛托（Nicola Tonellotto）。2021年。关于多级密集检索的近似最近邻选择。《第30届ACM信息与知识管理国际会议论文集》，第3318 - 3322页。

Macedo Maia, Siegfried Handschuh, André Freitas, Brian Davis, Ross McDermott, Manel Zarrouk, and Alexandra Balahur. 2018. WWW'18 Open Challenge: Financial Opinion Mining and Question Answering. In Companion Proceedings of the The Web Conference 2018, pages 1941-1942.

马塞多·马亚（Macedo Maia）、西格弗里德·汉舒（Siegfried Handschuh）、安德烈·弗雷塔斯（André Freitas）、布莱恩·戴维斯（Brian Davis）、罗斯·麦克德莫特（Ross McDermott）、马内尔·扎鲁克（Manel Zarrouk）和亚历山德拉·巴拉胡尔（Alexandra Balahur）。2018年。WWW'18开放挑战：金融观点挖掘与问答。《2018年万维网会议伴侣论文集》，第1941 - 1942页。

Antonio Mallia, Omar Khattab, Torsten Suel, and Nicola Tonellotto. 2021. Learning passage impacts for inverted indexes. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 1723-1727.

安东尼奥·马利亚（Antonio Mallia）、奥马尔·哈塔布（Omar Khattab）、托尔斯滕·苏尔（Torsten Suel）和尼古拉·托内洛托（Nicola Tonellotto）。2021年。为倒排索引学习段落影响。《第44届ACM信息检索研究与发展国际会议论文集》，第1723 - 1727页。

Aditya Krishna Menon, Sadeep Jayasumana, Se-ungyeon Kim, Ankit Singh Rawat, Sashank J. Reddi, and Sanjiv Kumar. 2022. In defense of dual-encoders for neural ranking.

阿迪蒂亚·克里希纳·梅农（Aditya Krishna Menon）、萨迪普·贾亚苏马纳（Sadeep Jayasumana）、金成妍（Se-ungyeon Kim）、安基特·辛格·拉瓦特（Ankit Singh Rawat）、萨尚克·J·雷迪（Sashank J. Reddi）和桑吉夫·库马尔（Sanjiv Kumar）。2022年。为神经排序的双编码器辩护。

Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, and Li Deng. 2016. MS MARCO: A human-generated MAchine reading COmprehension dataset. arXiv preprint arXiv:1611.09268.

特里·阮（Tri Nguyen）、米尔·罗森伯格（Mir Rosenberg）、宋霞（Xia Song）、高剑锋（Jianfeng Gao）、索拉布·蒂瓦里（Saurabh Tiwary）、兰甘·马朱姆德（Rangan Majumder）和李登（Li Deng）。2016年。MS MARCO：一个人工生成的机器阅读理解数据集。预印本arXiv:1611.09268。

Rodrigo Nogueira and Kyunghyun Cho. 2019. Passage Re-ranking with BERT. arXiv preprint arXiv:1901.04085.

罗德里戈·诺盖拉（Rodrigo Nogueira）和赵京焕（Kyunghyun Cho）。2019年。使用BERT进行段落重排序。预印本arXiv:1901.04085。

Barlas Oğuz, Kushal Lakhotia, Anchit Gupta, Patrick

巴拉斯·奥古兹（Barlas Oğuz）、库沙尔·拉科蒂亚（Kushal Lakhotia）、安奇特·古普塔（Anchit Gupta）、帕特里克

Lewis, Vladimir Karpukhin, Aleksandra Piktus, Xilun Chen, Sebastian Riedel, Wen-tau Yih, Sonal Gupta, et al. 2021. Domain-matched Pre-training Tasks for Dense Retrieval. arXiv preprint arXiv:2107.13602.

刘易斯（Lewis）、弗拉基米尔·卡尔普欣（Vladimir Karpukhin）、亚历山德拉·皮克图斯（Aleksandra Piktus）、陈希伦（Xilun Chen）、塞巴斯蒂安·里德尔（Sebastian Riedel）、文涛·易（Wen-tau Yih）、索纳尔·古普塔（Sonal Gupta）等。2021年。用于密集检索的领域匹配预训练任务。预印本arXiv:2107.13602。

Ashwin Paranjape, Omar Khattab, Christopher Potts, Matei Zaharia, and Christopher D Manning. 2022. Hindsight: Posterior-guided training of retrievers for improved open-ended generation. In International Conference on Learning Representations.

阿什温·帕兰贾佩（Ashwin Paranjape）、奥马尔·哈塔卜（Omar Khattab）、克里斯托弗·波茨（Christopher Potts）、马特·扎哈里亚（Matei Zaharia）和克里斯托弗·D·曼宁（Christopher D Manning）。2022年。后见之明：通过后验引导训练检索器以改进开放式生成。发表于国际学习表征会议。

Yingqi Qu, Yuchen Ding, Jing Liu, Kai Liu, Ruiyang Ren, Wayne Xin Zhao, Daxiang Dong, Hua Wu, and Haifeng Wang. 2021. RocketQA: An optimized training approach to dense passage retrieval for open-domain question answering. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 5835-5847, Online. Association for Computational Linguistics.

曲英琦（Yingqi Qu）、丁宇辰（Yuchen Ding）、刘静（Jing Liu）、刘凯（Kai Liu）、任瑞阳（Ruiyang Ren）、赵鑫（Wayne Xin Zhao）、董大祥（Daxiang Dong）、吴华（Hua Wu）和王海峰（Haifeng Wang）。2021年。RocketQA：一种用于开放域问答的密集段落检索优化训练方法。发表于2021年北美计算语言学协会人类语言技术会议论文集，第5835 - 5847页，线上会议。计算语言学协会。

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. SQuAD: 100,000+ questions for machine comprehension of text. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 2383-2392, Austin, Texas. Association for Computational Linguistics.

普拉纳夫·拉朱帕尔卡（Pranav Rajpurkar）、张健（Jian Zhang）、康斯坦丁·洛皮列夫（Konstantin Lopyrev）和珀西·梁（Percy Liang）。2016年。SQuAD：用于文本机器理解的10万多个问题。发表于2016年自然语言处理经验方法会议论文集，第2383 - 2392页，美国得克萨斯州奥斯汀。计算语言学协会。

Ruiyang Ren, Shangwen Lv, Yingqi Qu, Jing Liu, Wayne Xin Zhao, QiaoQiao She, Hua Wu, Haifeng Wang, and Ji-Rong Wen. 2021a. PAIR: Leveraging passage-centric similarity relation for improving dense passage retrieval. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pages 2173-2183, Online. Association for Computational Linguistics.

任瑞阳（Ruiyang Ren）、吕尚文（Shangwen Lv）、曲英琦（Yingqi Qu）、刘静（Jing Liu）、赵鑫（Wayne Xin Zhao）、佘巧巧（QiaoQiao She）、吴华（Hua Wu）、王海峰（Haifeng Wang）和文继荣（Ji-Rong Wen）。2021a。PAIR：利用以段落为中心的相似性关系改进密集段落检索。发表于计算语言学协会研究成果：ACL - IJCNLP 2021，第2173 - 2183页，线上会议。计算语言学协会。

Ruiyang Ren, Yingqi Qu, Jing Liu, Wayne Xin Zhao, Qiaoqiao She, Hua Wu, Haifeng Wang, and Ji-Rong Wen. 2021b. RocketQAv2: A Joint Training Method for Dense Passage Retrieval and Passage Re-ranking. arXiv preprint arXiv:2110.07367.

任瑞阳（Ruiyang Ren）、曲英琦（Yingqi Qu）、刘静（Jing Liu）、赵鑫（Wayne Xin Zhao）、佘巧巧（Qiaoqiao She）、吴华（Hua Wu）、王海峰（Haifeng Wang）和文继荣（Ji-Rong Wen）。2021b。RocketQAv2：一种用于密集段落检索和段落重排序的联合训练方法。预印本arXiv:2110.07367。

Stephen E Robertson, Steve Walker, Susan Jones, Micheline M Hancock-Beaulieu, Mike Gatford, et al. 1995. Okapi at TREC-3. NIST Special Publication.

斯蒂芬·E·罗伯逊（Stephen E Robertson）、史蒂夫·沃克（Steve Walker）、苏珊·琼斯（Susan Jones）、米歇琳·M·汉考克 - 博略（Micheline M Hancock-Beaulieu）、迈克·加特福德（Mike Gatford）等。1995年。TREC - 3中的Okapi系统。美国国家标准与技术研究院特别出版物。

Nandan Thakur, Nils Reimers, Andreas Rücklé, Ab-hishek Srivastava, and Iryna Gurevych. 2021. BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models. arXiv preprint arXiv:2104.08663.

南丹·塔库尔（Nandan Thakur）、尼尔斯·赖默斯（Nils Reimers）、安德里亚斯·吕克莱（Andreas Rücklé）、阿比舍克·斯里瓦斯塔瓦（Ab - hishek Srivastava）和伊琳娜·古列维奇（Iryna Gurevych）。2021年。BEIR：一个用于信息检索模型零样本评估的异构基准。预印本arXiv:2104.08663。

James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. 2018. FEVER: a large-scale dataset for fact extraction and VERification. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 809-819, New Orleans, Louisiana. Association for Computational Linguistics.

詹姆斯·索恩（James Thorne）、安德里亚斯·弗拉乔斯（Andreas Vlachos）、克里斯托斯·克里斯托杜洛普洛斯（Christos Christodoulopoulos）和阿尔皮特·米塔尔（Arpit Mittal）。2018年。FEVER：一个用于事实提取和验证的大规模数据集。发表于2018年北美计算语言学协会人类语言技术会议论文集，第1卷（长论文），第809 - 819页，美国路易斯安那州新奥尔良。计算语言学协会。

Ellen Voorhees, Tasmeer Alam, Steven Bedrick, Dina Demner-Fushman, William R Hersh, Kyle Lo, Kirk Roberts, Ian Soboroff, and Lucy Lu Wang. 2021. TREC-COVID: Constructing a Pandemic Information Retrieval Test Collection. In ACM SIGIR Forum, volume 54, pages 1-12. ACM New York, NY, USA.

埃伦·沃里斯（Ellen Voorhees）、塔斯米尔·阿拉姆（Tasmeer Alam）、史蒂文·贝德里克（Steven Bedrick）、迪娜·德姆纳 - 富什曼（Dina Demner - Fushman）、威廉·R·赫什（William R Hersh）、凯尔·洛（Kyle Lo）、柯克·罗伯茨（Kirk Roberts）、伊恩·索博罗夫（Ian Soboroff）和王璐（Lucy Lu Wang）。2021年。TREC - COVID：构建一个大流行信息检索测试集。发表于ACM SIGIR论坛，第54卷，第1 - 12页。美国纽约州纽约市ACM协会。

Henning Wachsmuth, Shahbaz Syed, and Benno Stein. 2018. Retrieval of the best counterargument without prior topic knowledge. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 241-251, Melbourne, Australia. Association for Computational Linguistics.

亨宁·瓦克斯穆特（Henning Wachsmuth）、沙赫巴兹·赛义德（Shahbaz Syed）和本诺·斯坦（Benno Stein）。2018年。在没有先验主题知识的情况下检索最佳反驳论点。发表于计算语言学协会第56届年会论文集（第1卷：长论文），第241 - 251页，澳大利亚墨尔本。计算语言学协会。

David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan, and Hannaneh Hajishirzi. 2020. Fact or fiction: Verifying scientific claims. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 7534-7550, Online. Association for Computational Linguistics.

大卫·瓦登（David Wadden）、林山川（Shanchuan Lin）、凯尔·洛（Kyle Lo）、王露西·陆（Lucy Lu Wang）、玛德琳·范·聚伦（Madeleine van Zuylen）、阿尔曼·科汉（Arman Cohan）和汉娜内·哈吉希尔齐（Hannaneh Hajishirzi）。2020年。事实还是虚构：验证科学主张。见《2020年自然语言处理经验方法会议论文集》（EMNLP），第7534 - 7550页，线上会议。计算语言学协会。

Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. 2020. MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. arXiv preprint arXiv:2002.10957.

王文慧（Wenhui Wang）、魏富如（Furu Wei）、李东（Li Dong）、鲍航波（Hangbo Bao）、杨楠（Nan Yang）和周明（Ming Zhou）。2020年。MiniLM：用于预训练Transformer模型与任务无关压缩的深度自注意力蒸馏。预印本arXiv:2002.10957。

Benchang Wei, Tao Guan, and Junqing Yu. 2014. Projected Residual Vector Quantization for ANN Search. IEEE multimedia, 21(3):41-51.

魏本昌（Benchang Wei）、管涛（Tao Guan）和余俊清（Junqing Yu）。2014年。用于近似最近邻搜索的投影残差向量量化。《IEEE多媒体》，21(3):41 - 51。

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pier-ric Cistac, Tim Rault, Remi Louf, Morgan Funtow-icz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. 2020. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38-45, Online. Association for Computational Linguistics.

托马斯·沃尔夫（Thomas Wolf）、利桑德尔·德比特（Lysandre Debut）、维克多·桑（Victor Sanh）、朱利安·肖蒙（Julien Chaumond）、克莱门特·德朗格（Clement Delangue）、安东尼·莫伊（Anthony Moi）、皮埃尔 - 里克·西斯塔克（Pier - ric Cistac）、蒂姆·劳尔特（Tim Rault）、雷米·卢夫（Remi Louf）、摩根·丰托维茨（Morgan Funtowicz）、乔·戴维森（Joe Davison）、山姆·施莱弗（Sam Shleifer）、帕特里克·冯·普拉滕（Patrick von Platen）、克拉拉·马（Clara Ma）、亚辛·杰尔尼（Yacine Jernite）、朱利安·普鲁（Julien Plu）、徐灿文（Canwen Xu）、特文·勒·斯考（Teven Le Scao）、西尔万·古格（Sylvain Gugger）、玛丽亚玛·德拉梅（Mariama Drame）、昆汀·勒霍斯特（Quentin Lhoest）和亚历山大·拉什（Alexander Rush）。2020年。Transformer：最先进的自然语言处理技术。见《2020年自然语言处理经验方法会议：系统演示论文集》，第38 - 45页，线上会议。计算语言学协会。

Ji Xin, Chenyan Xiong, Ashwin Srinivasan, Ankita Sharma, Damien Jose, and Paul N Bennett. 2021. Zero-Shot Dense Retrieval with Momentum Adversarial Domain Invariant Representations. arXiv preprint arXiv:2110.07581.

辛吉（Ji Xin）、熊晨彦（Chenyan Xiong）、阿什温·斯里尼瓦桑（Ashwin Srinivasan）、安基塔·夏尔马（Ankita Sharma）、达米安·何塞（Damien Jose）和保罗·N·贝内特（Paul N Bennett）。2021年。基于动量对抗域不变表示的零样本密集检索。预印本arXiv:2110.07581。

Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N Bennett, Junaid Ahmed, and Arnold Overwijk. 2020. Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval. In International Conference on Learning Representations.

熊磊（Lee Xiong）、熊晨彦（Chenyan Xiong）、李烨（Ye Li）、邓国峰（Kwok - Fung Tang）、刘佳琳（Jialin Liu）、保罗·N·贝内特（Paul N Bennett）、朱奈德·艾哈迈德（Junaid Ahmed）和阿诺德·奥弗维克（Arnold Overwijk）。2020年。用于密集文本检索的近似最近邻负对比学习。见《国际学习表征会议》。

Ikuya Yamada, Akari Asai, and Hannaneh Hajishirzi. 2021a. Efficient passage retrieval with hashing for open-domain question answering. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pages 979-986, Online.

山田育也（Ikuya Yamada）、浅井朱里（Akari Asai）和汉娜内·哈吉希尔齐（Hannaneh Hajishirzi）。2021a。用于开放域问答的高效哈希段落检索。见《计算语言学协会第59届年会和第11届自然语言处理国际联合会议论文集》（第2卷：短篇论文），第979 - 986页，线上会议。

Association for Computational Linguistics.

计算语言学协会。

Ikuya Yamada, Akari Asai, and Hannaneh Hajishirzi. 2021b. Efficient passage retrieval with hashing for open-domain question answering. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pages 979-986, Online. Association for Computational Linguistics.

山田育也（Ikuya Yamada）、浅井朱里（Akari Asai）和汉娜·哈吉希尔齐（Hannaneh Hajishirzi）。2021b。用于开放域问答的基于哈希的高效段落检索。见《计算语言学协会第59届年会和第11届自然语言处理国际联合会议论文集（第2卷：短篇论文）》，第979 - 986页，线上会议。计算语言学协会。

Peilin Yang, Hui Fang, and Jimmy Lin. 2018a. Anserini: Reproducible ranking baselines using lucene. Journal of Data and Information Quality $\left( {JDIQ}\right) ,{10}\left( 4\right)  : 1 - {20}$ .

杨沛霖（Peilin Yang）、方慧（Hui Fang）和林吉米（Jimmy Lin）。2018a。Anserini：使用Lucene的可复现排名基线。《数据与信息质量杂志》$\left( {JDIQ}\right) ,{10}\left( 4\right)  : 1 - {20}$。

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018b. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2369-2380, Brussels, Belgium. Association for Computational Linguistics.

杨志林（Zhilin Yang）、齐鹏（Peng Qi）、张赛征（Saizheng Zhang）、约书亚·本吉奥（Yoshua Bengio）、威廉·科恩（William Cohen）、鲁斯兰·萨拉胡季诺夫（Ruslan Salakhutdinov）和克里斯托弗·D·曼宁（Christopher D. Manning）。2018b。HotpotQA：一个用于多样化、可解释多跳问答的数据集。见《2018年自然语言处理经验方法会议论文集》，第2369 - 2380页，比利时布鲁塞尔。计算语言学协会。

Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Jiafeng Guo, Min Zhang, and Shaoping Ma. 2021a. Jointly Optimizing Query Encoder and Product Quantization to Improve Retrieval Performance. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management, pages 2487-2496.

詹景涛、毛佳鑫、刘奕群、郭佳峰、张敏和马少平。2021a。联合优化查询编码器和乘积量化以提高检索性能。见《第30届ACM信息与知识管理国际会议论文集》，第2487 - 2496页。

Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Jiafeng Guo, Min Zhang, and Shaoping Ma. 2021b. Optimizing Dense Retrieval Model Training with Hard Negatives. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 1503-1512.

詹景涛、毛佳鑫、刘奕群、郭佳峰、张敏和马少平。2021b。使用难负样本优化密集检索模型训练。见《第44届ACM SIGIR信息检索研究与发展国际会议论文集》，第1503 - 1512页。

Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Jiafeng Guo, Min Zhang, and Shaoping Ma. 2022. Learning discrete representations via constrained clustering for effective and efficient dense retrieval. In Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining, WSDM '22, page 1328-1336. Association for Computing Machinery.

詹景涛、毛佳鑫、刘奕群、郭佳峰、张敏和马少平。2022。通过约束聚类学习离散表示以实现有效且高效的密集检索。见《第15届ACM网络搜索与数据挖掘国际会议论文集，WSDM '22》，第1328 - 1336页。美国计算机协会。

Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Min Zhang, and Shaoping Ma. 2020a. Learning to retrieve: How to train a dense retrieval model effectively and efficiently. arXiv preprint arXiv:2010.10469.

詹景涛、毛佳鑫、刘奕群、张敏和马少平。2020a。学习检索：如何有效且高效地训练密集检索模型。预印本arXiv:2010.10469。

Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Min Zhang, and Shaoping Ma. 2020b. Repbert: Contextualized text embeddings for first-stage retrieval. arXiv preprint arXiv:2006.15498.

詹景涛、毛佳鑫、刘奕群、张敏和马少平。2020b。Repbert：用于第一阶段检索的上下文文本嵌入。预印本arXiv:2006.15498。

Giulio Zhou and Jacob Devlin. 2021. Multi-vector attention models for deep re-ranking. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 5452-5456, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

朱利奥·周（Giulio Zhou）和雅各布·德夫林（Jacob Devlin）。2021。用于深度重排序的多向量注意力模型。见《2021年自然语言处理经验方法会议论文集》，第5452 - 5456页，线上会议和多米尼加共和国蓬塔卡纳。计算语言学协会。

<!-- Media -->

<!-- figureText: ColBERT Random 75 Proportion 50 25 4 64 1024 #Distinct Clusters per Token (b) Number of distinct clus- ters each token appears in. 75 Proportion 50 25 16 256 4096 #Distinct Tokens per Cluster (a) Number of distinct tokens appearing in each cluster. -->

<img src="https://cdn.noedgeai.com/0195afc1-371d-7589-8122-942ba256f893_14.jpg?x=193&y=199&w=610&h=408&r=0"/>

Figure 2: Empirical CDFs analyzing semantic properties of MS MARCO token-level embeddings both encoded by ColBERT and randomly generated. The em-beddings are partitioned into ${2}^{18}$ clusters and correspond to roughly 27,000 distinct tokens.

图2：分析由ColBERT编码和随机生成的MS MARCO词元级嵌入语义属性的经验累积分布函数。这些嵌入被划分为${2}^{18}$个簇，对应大约27,000个不同的词元。

<!-- Media -->

## A Analysis of ColBERT's Semantic Space

## A ColBERT语义空间分析

ColBERT (Khattab and Zaharia, 2020) decomposes representations and similarity computation at the token level. Because of this compositional architecture, we hypothesize that ColBERT exhibits a "lightweight" semantic space: without any special re-training, vectors corresponding to each sense of a word would cluster very closely, with only minor variation due to context.

ColBERT（卡塔布（Khattab）和扎哈里亚（Zaharia），2020）在词元级别分解表示和相似度计算。由于这种组合式架构，我们假设ColBERT呈现出一个“轻量级”语义空间：在无需任何特殊再训练的情况下，对应于一个词每个义项的向量会紧密聚类，仅因上下文产生微小变化。

If this hypothesis is true, we would expect the embeddings corresponding to each token in the vocabulary to localize in only a small number of regions in the embedding space, corresponding to the contextual "senses" of the token. To validate this hypothesis, we analyze the ColBERT embeddings corresponding to the tokens in the MS MARCO Passage Ranking (Nguyen et al., 2016) collection: we perform $k$ -means clustering on the nearly ${600}\mathrm{M}$ embeddings-corresponding to 27,000 unique tokens—into $k = {2}^{18}$ clusters. As a baseline, we repeat this clustering with random embeddings but keep the true distribution of tokens. Figure 2 presents empirical cumulative distribution function (eCDF) plots representing the number of distinct non-stopword tokens appearing in each cluster (2a) and the number of distinct clusters in which each token appears (2b). ${}^{6}$ Most tokens appear in a very small fraction of the number of centroids: in particular, we see that roughly ${90}\%$ of clusters have $\leq  {16}$ distinct tokens with the ColBERT embeddings, whereas less than 50% of clusters have $\leq  {16}$ distinct tokens with the random embeddings. This suggests that the centroids effectively map the ColBERT semantic space.

如果这一假设成立，我们预计词汇表中每个标记对应的嵌入会仅定位在嵌入空间的少数区域，这些区域对应于该标记的上下文“语义”。为了验证这一假设，我们分析了与MS MARCO段落排名（Nguyen等人，2016）数据集中的标记相对应的ColBERT嵌入：我们对近${600}\mathrm{M}$个嵌入（对应27,000个唯一标记）进行了$k$均值聚类，将其分为$k = {2}^{18}$个簇。作为基线，我们使用随机嵌入重复此聚类过程，但保持标记的真实分布。图2展示了经验累积分布函数（eCDF）图，分别表示每个簇中出现的不同非停用词标记的数量（图2a）以及每个标记出现的不同簇的数量（图2b）。${}^{6}$大多数标记仅出现在极少数的质心中：特别是，我们发现使用ColBERT嵌入时，大约${90}\%$的簇有$\leq  {16}$个不同的标记，而使用随机嵌入时，不到50%的簇有$\leq  {16}$个不同的标记。这表明质心有效地映射了ColBERT语义空间。

Table 6 presents examples to highlight the semantic space captured by the centroids. The most frequently appearing tokens in cluster #917 relate to photography; these include, for example, 'photos' and 'photographs'. If we then examine the additional clusters in which these tokens appear, we find that there is substantial semantic overlap between these new clusters (e.g., Photos-Photo, Photo-Image-Picture) and cluster #917. We observe a similar effect with tokens appearing in cluster #216932, comprising tornado-related terms.

表6给出了一些示例，以突出质心所捕捉的语义空间。簇#917中最常出现的标记与摄影相关；例如，包括“photos（照片）”和“photographs（照片）”。如果我们接着检查这些标记出现的其他簇，会发现这些新簇（例如，Photos - Photo、Photo - Image - Picture）与簇#917之间存在大量的语义重叠。对于出现在簇#216932中的标记（包含与龙卷风相关的术语），我们也观察到了类似的效果。

This analysis indicates that cluster centroids can summarize the ColBERT representations with high precision. In $§{3.3}$ ,we propose a residual compression mechanism that uses these centroids along with minor refinements at the dimension level to efficiently encode late-interaction vectors.

这一分析表明，簇质心可以高精度地总结ColBERT表示。在$§{3.3}$中，我们提出了一种残差压缩机制，该机制利用这些质心以及在维度级别上的细微调整来有效地编码后期交互向量。

## B Impact of Compression

## B 压缩的影响

Our residual compression approach (§3.3) preserves approximately the same quality as the uncompressed embeddings. In particular, when applied to a vanilla ColBERT model on MS MARCO whose MRR@10 is ${36.2}\%$ and Recall@50 is ${82.1}\%$ ,the quality of the model with 2 -bit compression is ${36.2}\%$ MRR@10 and ${82.3}\%$ Recall@50.With 1-bit compression, the model achieves 35.5% MRR@10 and 81.6% Recall@50. ${}^{7}$

我们的残差压缩方法（§3.3）保留了与未压缩嵌入大致相同的质量。特别是，当将其应用于MS MARCO上的原始ColBERT模型（其MRR@10为${36.2}\%$，Recall@50为${82.1}\%$）时，采用2位压缩的模型质量为MRR@10 ${36.2}\%$和Recall@50 ${82.3}\%$。采用1位压缩时，模型的MRR@10达到35.5%，Recall@50达到81.6%。${}^{7}$

We also tested the residual compression approach on late-interaction retrievers that conduct downstream tasks, namely, ColBERT-QA (Khat-tab et al., 2021b) for the NaturalQuestions open-domain QA task, and Baleen (Khattab et al., 2021a) for multi-hop reasoning on HoVer for claim verification. On the NQ dev set, ColBERT-QA's success@5 (success@20) dropped only marginally from 75.3% (84.3%) to 74.3% (84.2%) and its downstream Open-QA answer exact match dropped from 47.9% to 47.7%, when using 2-bit compression for retrieval and using the same checkpoints of ColBERT-QA otherwise.

我们还在执行下游任务的后期交互检索器上测试了残差压缩方法，即用于自然问题开放域问答任务的ColBERT - QA（Khat - tab等人，2021b），以及用于HoVer上多跳推理以进行声明验证的Baleen（Khattab等人，2021a）。在NQ开发集上，当使用2位压缩进行检索且其他方面使用ColBERT - QA的相同检查点时，ColBERT - QA的success@5（success@20）仅从75.3%（84.3%）略微下降到74.3%（84.2%），其下游开放问答答案的精确匹配率从47.9%下降到47.7%。

---

<!-- Footnote -->

${}^{7}$ We contrast this with an early implementation of compression for ColBERT, which used binary representations as in BPR (Yamada et al., 2021a) without residual centroids, and achieves 34.8% (35.7%) MRR@10 and 80.5% (81.8%) Recall@50 with 1-bit (2-bit) binarization. Like the original ColBERT, this form of compression relied on a separate FAISS index for candidate generation.

${}^{7}$ 我们将此与ColBERT的早期压缩实现进行对比，该早期实现如BPR（Yamada等人，2021a）中那样使用二进制表示，且没有残差质心，在1位（2位）二值化时，其MRR@10达到34.8%（35.7%），Recall@50达到80.5%（81.8%）。与原始的ColBERT一样，这种压缩形式依赖于单独的FAISS索引来生成候选。

${}^{6}$ We rank tokens by number of clusters they appear in and designate the top-1% (under 300) as stopwords.

${}^{6}$ 我们根据标记出现的簇的数量对其进行排名，并将前1%（不到300个）指定为停用词。

<!-- Footnote -->

---

<!-- Media -->

<table><tr><td rowspan="2">Cluster ID</td><td rowspan="2">Most Common Tokens</td><td colspan="2">Most Common Clusters Per Token</td></tr><tr><td>Token</td><td>Clusters</td></tr><tr><td rowspan="3">917</td><td rowspan="3">'photos', 'photo', 'pictures', 'photographs', 'images', 'photography', 'photograph'</td><td>‘photos’</td><td>Photos-Photo, Photos-Pictures-Photo</td></tr><tr><td>‘photo’</td><td>Photo-Image-Picture, Photo-Picture-Photograph, Photo-Picture-Photography</td></tr><tr><td>'pictures'</td><td>Pictures-Picture-Images, Picture-Pictures-Artists, Pictures-Photo-Picture</td></tr><tr><td rowspan="3">216932</td><td rowspan="3">'tornado', 'tornadoes', 'storm' 'hurricane', 'storms'</td><td>‘tornado’</td><td>Tornado-Hurricane-Storm, Tornadoes-Tornado-Blizzard</td></tr><tr><td>‘tornadoes’</td><td>Tornadoes-Tornado-Storms, Tornadoes-Tornado-Blizzard, Tornado-Hurricane-Storm</td></tr><tr><td>‘storm’</td><td>Storm-Storms, Storm-Storms-Weather, Storm-Storms-Tempest</td></tr></table>

<table><tbody><tr><td rowspan="2">集群ID</td><td rowspan="2">最常见的Token</td><td colspan="2">每个Token最常见的集群</td></tr><tr><td>Token</td><td>集群</td></tr><tr><td rowspan="3">917</td><td rowspan="3">“照片”、“相片”、“图片”、“摄影作品”、“图像”、“摄影”、“照片（photograph）”</td><td>“照片（photos）”</td><td>照片-相片，照片-图片-相片</td></tr><tr><td>“相片（photo）”</td><td>相片-图像-图片，相片-图片-摄影作品，相片-图片-摄影</td></tr><tr><td>“图片”</td><td>图片-图片-图像，图片-图片-艺术家，图片-照片-图片</td></tr><tr><td rowspan="3">216932</td><td rowspan="3">“龙卷风”、“龙卷风（复数）”、“风暴”、“飓风”、“风暴（复数）”</td><td>“龙卷风（tornado）”</td><td>龙卷风-飓风-风暴，龙卷风（复数）-龙卷风-暴风雪</td></tr><tr><td>“龙卷风（复数）（tornadoes）”</td><td>龙卷风（复数）-龙卷风-风暴，龙卷风（复数）-龙卷风-暴风雪，龙卷风-飓风-风暴</td></tr><tr><td>“风暴（storm）”</td><td>风暴-风暴（复数），风暴-风暴（复数）-天气，风暴-风暴（复数）-暴风雨</td></tr></tbody></table>

Table 6: Examples of clusters taken from all MS MARCO passages. We present the tokens that appear most frequently in the selected clusters as well as additional clusters the top tokens appear in.

表6：从所有MS MARCO段落中选取的聚类示例。我们展示了所选聚类中出现频率最高的标记，以及这些顶级标记所在的其他聚类。

<!-- figureText: 250 probe bits candidates probe $\times  2 \uparrow  {14}$ probe $\times  2 \uparrow  {12}$ 69.5 74.0 74.5 75.0 75.5 76.0 Success@5 Query Latency (ms) 200 150 100 ✘ 38.50 38.75 39.00 39.25 39.50 39.75 68.0 68.5 69.0 MRR@10 Success@5 -->

<img src="https://cdn.noedgeai.com/0195afc1-371d-7589-8122-942ba256f893_15.jpg?x=188&y=1321&w=1276&h=391&r=0"/>

Figure 3: Latency vs. retrieval quality with varying parameter configurations for three datasets of different collection sizes. We sweep a range of values for the number of centroids per vector (probe), the number of bits used for residual compression, and the number of candidates. Note that retrieval quality is measured in MRR@10 for MS MARCO and Success@5 for LoTTE datasets. Results toward the bottom right corner (higher quality, lower latency) are best.

图3：针对三种不同集合大小的数据集，不同参数配置下的延迟与检索质量对比。我们对每个向量的质心数量（探测值）、用于残差压缩的比特数以及候选数量进行了一系列取值测试。请注意，对于MS MARCO数据集，检索质量以MRR@10衡量；对于LoTTE数据集，以Success@5衡量。朝右下角（质量更高、延迟更低）的结果最佳。

<!-- Media -->

Similarly, on the HoVer (Jiang et al., 2020) dev set, Baleen's retrieval R@100 dropped from 92.2% to only ${90.6}\%$ but its sentence-level exact match remained roughly the same,going from 39.2% to 39.4%. We hypothesize that the supervision methods applied in ColBERTv2 (§3.2) can also be applied to lift quality in downstream tasks by improving the recall of retrieval for these tasks. We leave such exploration for future work.

同样，在HoVer（Jiang等人，2020）开发集上，Baleen的检索R@100从92.2%降至仅${90.6}\%$，但其句子级精确匹配率大致保持不变，从39.2%变为39.4%。我们假设，ColBERTv2（§3.2）中应用的监督方法也可用于通过提高这些下游任务的检索召回率来提升任务质量。我们将此类探索留待未来工作。

## C Retrieval Latency

## C 检索延迟

Figure 3 evaluates the latency of ColBERTv2 across three collections of varying sizes, namely, MS MARCO, LoTTE Pooled (dev), and LoTTE Lifestyle (dev),which contain approximately $9\mathrm{M}$ passages, ${2.4}\mathrm{M}$ answer posts,and ${270}\mathrm{k}$ answer posts, respectively. We average latency across three runs of the MS MARCO dev set and the LoTTE "search" queries. Search is executed using a Titan V GPU on a server with two Intel Xeon Gold 6132 CPUs, each with 28 hardware execution contexts.

图3评估了ColBERTv2在三种不同大小集合上的延迟，即MS MARCO、LoTTE Pooled（开发集）和LoTTE Lifestyle（开发集），它们分别包含约$9\mathrm{M}$个段落、${2.4}\mathrm{M}$个答案帖子和${270}\mathrm{k}$个答案帖子。我们对MS MARCO开发集和LoTTE“搜索”查询的三次运行的延迟进行了平均。搜索在配备两块英特尔至强金牌6132 CPU（每块CPU有28个硬件执行上下文）的服务器上使用Titan V GPU执行。

The figure varies three settings of ColBERTv2. In particular, we evaluate indexing with 1-bit and 2-bit encoding $\left( {§{3.4}}\right)$ and searching by probing the nearest1,2,or 4 centroids to each query vector (§3.5). When probing probe centroids per vector, we score either probe $\times  {2}^{12}$ or probe $\times  {2}^{14}$ candidates per query. ${}^{8}$

该图展示了ColBERTv2的三种设置变化。具体而言，我们评估了1比特和2比特编码的索引$\left( {§{3.4}}\right)$，并通过探测每个查询向量最近的1、2或4个质心进行搜索（§3.5）。当每个向量探测probe个质心时，我们为每个查询对probe $\times  {2}^{12}$或probe $\times  {2}^{14}$个候选进行打分。${}^{8}$

To begin with, we notice that the quality reported on the $x$ -axis varies only within a relatively narrow range. For instance, the axis ranges from 38.50 through 39.75 for MS MARCO, and all but two of the cheapest settings score above 39.00 . Similarly, the $y$ -axis varies between approximately 50 milliseconds per query up to 250 milliseconds (mostly under 150 milliseconds) using our relatively simple Python-based implementation.

首先，我们注意到$x$轴上报告的质量仅在相对较窄的范围内变化。例如，对于MS MARCO，该轴范围从38.50到39.75，除了两种最便宜的设置外，所有设置的得分都高于39.00。同样，使用我们相对简单的基于Python的实现，$y$轴在每个查询约50毫秒到250毫秒之间变化（大部分低于150毫秒）。

Digging deeper, we see that the best quality in these metrics can be achieved or approached closely with around 100 milliseconds of latency across all three datasets, despite their various sizes and characteristics, and that 2-bit indexing reliably outperforms 1-bit indexing but the loss from more aggressive compression is small.

进一步分析，我们发现，尽管这三个数据集大小和特征各异，但在所有三个数据集上，约100毫秒的延迟就可以实现或接近这些指标的最佳质量，并且2比特索引始终优于1比特索引，但更激进的压缩带来的损失较小。

## D LoTTE

## D LoTTE

Domain coverage Table 9 presents the full distribution of communities in the LoTTE dev dataset.

领域覆盖 表9展示了LoTTE开发数据集中社区的完整分布。

<!-- Media -->

<!-- figureText: Writing 400 600 Words per passage Recreation Science Technology Lifestyle Pooled 200 -->

<img src="https://cdn.noedgeai.com/0195afc1-371d-7589-8122-942ba256f893_16.jpg?x=853&y=193&w=601&h=242&r=0"/>

Figure 4: LoTTE words per passage

图4：LoTTE每个段落的单词数

<!-- figureText: [Search] Writing 10 15 20 Words per query [Search] Recreation [Search] Science [Search] Technology [Search] Lifestyle [Search] Pooled [Forum] Writing [Forum] Recreation [Forum] Science [Forum] Technology [Forum] Lifestyle [Forum] Pooled -->

<img src="https://cdn.noedgeai.com/0195afc1-371d-7589-8122-942ba256f893_16.jpg?x=839&y=522&w=614&h=363&r=0"/>

Figure 5: LoTTE words per query

图5：LoTTE每个查询的单词数

<!-- Media -->

The topics covered by LoTTE cover a wide range of linguistic phenomena given the diversity in topics and communities represented. However, since all posts are submitted by anonymous users we do not have demographic information regarding the identify of the contributors. All posts are written in English.

鉴于所代表的主题和社区的多样性，LoTTE涵盖的主题涉及广泛的语言现象。然而，由于所有帖子均由匿名用户提交，我们没有关于贡献者身份的人口统计信息。所有帖子均用英语撰写。

Passages As mentioned in $\$ 4$ ,we construct LoTTE collections by selecting passages from the StackExchange archive with positive scores. We remove HTML tags from passages and filter out empty passages. For each passage we record its corresponding query and save the query-to-passage mapping to keep track of the posted answers corresponding to each query.

段落 如$\$ 4$中所述，我们通过从StackExchange存档中选择得分较高的段落来构建LoTTE集合。我们从段落中去除HTML标签并过滤掉空段落。对于每个段落，我们记录其对应的查询，并保存查询到段落的映射，以跟踪每个查询对应的已发布答案。

Search queries We construct the list of LoTTE search queries by drawing from GooAQ queries that appear in the StackExchange post archive. We first shuffle the list of GooAQ queries so that in cases where multiple queries exist for the same answer passage we randomly select the query to include in LoTTE rather than always selecting the first appearing query. We verify that every query has at least one corresponding answer passage.

搜索查询 我们通过从StackExchange帖子存档中出现的GooAQ查询中抽取来构建LoTTE搜索查询列表。我们首先对GooAQ查询列表进行洗牌，这样在同一答案段落有多个查询的情况下，我们会随机选择一个查询纳入LoTTE，而不是总是选择最先出现的查询。我们验证了每个查询至少有一个对应的答案段落。

Forum queries For each LoTTE topic and its constituent communities we first compute the fraction of the total queries attributed to each individual community. We then use this distribution to construct a truncated query set by selecting the highest ranked queries from each community as determined by 1) the query scores and 2) the query view counts. We only use queries which have an accepted answer. We ensure that each community contributes at least 50 queries to the truncated set whenever possible. We set the overall size of the truncated set to be 2000 queries, though note that the total can exceed this due to rounding and/or the minimum per-community query count. We remove all quotation marks and HTML tags.

论坛查询 对于每个LoTTE主题及其组成社区，我们首先计算归因于每个单独社区的查询占总查询的比例。然后，我们使用此分布来构建一个截断查询集，方法是从每个社区中选择排名最高的查询，这些查询由以下两个因素确定：1) 查询得分；2) 查询浏览量。我们仅使用有被采纳答案的查询。只要可能，我们确保每个社区至少为截断集贡献50个查询。我们将截断集的总体大小设置为2000个查询，但请注意，由于四舍五入和/或每个社区的最小查询数量要求，总数可能会超过这个数字。我们会去除所有引号和HTML标签。

---

<!-- Footnote -->

${}^{8}$ These settings are selected based on preliminary exploration of these parameters, which indicated that performance for larger probe values tends to require scoring a larger number of candidates.

${}^{8}$ 这些设置是基于对这些参数的初步探索而选择的，该探索表明，对于较大的探测值，性能往往需要对更多的候选对象进行评分。

<!-- Footnote -->

---

<!-- Media -->

[Search] Writing

[搜索] 写作

<!-- figureText: 15 20 Answers per query 5 10 -->

<img src="https://cdn.noedgeai.com/0195afc1-371d-7589-8122-942ba256f893_17.jpg?x=383&y=193&w=419&h=362&r=0"/>

[Search] Recreation [Search] Science [Search] Technology [Search] Lifestyle [Search] Pooled [Forum] Writing [Forum] Recreation [Forum] Science [Forum] Technology [Forum] Lifestyle [Forum] Pooled

[搜索] 娱乐 [搜索] 科学 [搜索] 技术 [搜索] 生活方式 [搜索] 汇总 [论坛] 写作 [论坛] 娱乐 [论坛] 科学 [论坛] 技术 [论坛] 生活方式 [论坛] 汇总

Figure 6: LoTTE answers per query

图6：每个查询对应的LoTTE答案

Corpus LUTHOO SZINS HONV tavoids TAHAVIdS talilition

语料库 LUTHOO SZINS HONV tavoids TAHAVIdS talilition

LoTTE Search Dev Queries (Success@5)

LoTTE搜索开发查询（前5名成功率）

Writing 76.3 47.3 75.7 79.5 78.9 81.7

写作 76.3 47.3 75.7 79.5 78.9 81.7

Recreation 71.8 56.3 66.1 73.0 70.7 76.0

娱乐 71.8 56.3 66.1 73.0 70.7 76.0

Science 71.7 52.2 66.9 67.7 73.4 74.2

科学 71.7 52.2 66.9 67.7 73.4 74.2

Technology 52.8 35.8 55.7 54.3 56.3 59.3

技术 52.8 35.8 55.7 54.3 56.3 59.3

Lifestyle 73.1 54.4 69.8 72.4 71.2 75.8

生活方式 73.1 54.4 69.8 72.4 71.2 75.8

Pooled 65.4 45.6 63.7 66.4 67.0 69.3 LoTTE Forum Dev Queries (Success@5)

汇总 65.4 45.6 63.7 66.4 67.0 69.3 LoTTE论坛开发查询（前5名成功率）

<table><tr><td>Writing</td><td>75.5</td><td>66.2</td><td>74.4</td><td>75.5</td><td>78.1</td><td>80.8</td></tr><tr><td>Recreation</td><td>69.1</td><td>56.6</td><td>65.9</td><td>69.0</td><td>68.9</td><td>71.8</td></tr><tr><td>Science</td><td>58.2</td><td>51.3</td><td>56.3</td><td>56.7</td><td>59.9</td><td>62.6</td></tr><tr><td>Technology</td><td>39.6</td><td>30.7</td><td>38.8</td><td>39.9</td><td>42.1</td><td>45.0</td></tr><tr><td>Lifestyle</td><td>61.1</td><td>48.2</td><td>61.8</td><td>62.0</td><td>61.8</td><td>65.8</td></tr><tr><td>Pooled</td><td>59.1</td><td>47.8</td><td>57.4</td><td>58.9</td><td>60.6</td><td>63.7</td></tr></table>

<table><tbody><tr><td>写作</td><td>75.5</td><td>66.2</td><td>74.4</td><td>75.5</td><td>78.1</td><td>80.8</td></tr><tr><td>娱乐</td><td>69.1</td><td>56.6</td><td>65.9</td><td>69.0</td><td>68.9</td><td>71.8</td></tr><tr><td>科学</td><td>58.2</td><td>51.3</td><td>56.3</td><td>56.7</td><td>59.9</td><td>62.6</td></tr><tr><td>技术</td><td>39.6</td><td>30.7</td><td>38.8</td><td>39.9</td><td>42.1</td><td>45.0</td></tr><tr><td>生活方式</td><td>61.1</td><td>48.2</td><td>61.8</td><td>62.0</td><td>61.8</td><td>65.8</td></tr><tr><td>汇集的；集中的</td><td>59.1</td><td>47.8</td><td>57.4</td><td>58.9</td><td>60.6</td><td>63.7</td></tr></tbody></table>

Table 7: Zero-shot evaluation results on the dev sets of the LoTTE benchmark.

表7：LoTTE基准测试开发集上的零样本评估结果。

<!-- Media -->

Statistics Figure 4 plots the number of words per passage in each LoTTE dev corpus. Figures 5 and 6 plot the number of words and number of corresponding answer passages respectively per query, split across search and forum queries.

统计数据 图4展示了每个LoTTE开发语料库中每篇文章的单词数量。图5和图6分别展示了每个查询的单词数量和相应答案文章的数量，并按搜索查询和论坛查询进行了划分。

Dev Results Table 7 presents out-of-domain evaluation results on the LoTTE dev queries. Continuing the trend we observed in 5 , ColBERTv2 consistently outperforms all other models we tested.

开发集结果 表7展示了在LoTTE开发集查询上的域外评估结果。延续我们在5中观察到的趋势，ColBERTv2始终优于我们测试的所有其他模型。

Licensing and Anonymity The original Stack-Exchange post archive is licensed under a Creative Commons BY-SA 4.0 license (sta). Personal data is removed from the archive before being uploaded, though all posts are public; when we release LoTTE publicly we will include URLs to the original posts for proper attribution as required by the license. The GooAQ dataset is licensed under an Apache license, version 2.0 (Khashabi et al., 2021). We will also release LoTTE with a CC BY-SA 4.0 license. The search queries can be used for non-commercial research purposes only as per the GooAQ license.

许可与匿名性 原始的Stack-Exchange帖子存档根据知识共享署名 - 相同方式共享4.0许可协议（sta）进行许可。在上传之前，存档中的个人数据已被移除，尽管所有帖子都是公开的；当我们公开发布LoTTE时，我们将按照许可要求包含原始帖子的URL以进行适当的归因。GooAQ数据集根据Apache许可证2.0版本进行许可（Khashabi等人，2021年）。我们也将以知识共享署名 - 相同方式共享4.0许可协议发布LoTTE。根据GooAQ许可协议，搜索查询仅可用于非商业研究目的。

## E Datasets in BEIR

## E BEIR中的数据集

Table 8 lists the BEIR datasets we used in our evaluation, including their respective license information as well as the numbers of documents as well as the number of test set queries. We refer to Thakur et al. (2021) for a more detailed description of each of the datasets.

表8列出了我们在评估中使用的BEIR数据集，包括它们各自的许可信息、文档数量以及测试集查询的数量。关于每个数据集的更详细描述，我们参考Thakur等人（2021年）的研究。

Our Touché evaluation uses an updated version of the data in BEIR, which we use for evaluating the models we run (i.e., ColBERTv2 and RocketQAv2) as well as SPLADEv2.

我们的Touché评估使用了BEIR中数据的更新版本，我们用它来评估我们运行的模型（即ColBERTv2和RocketQAv2）以及SPLADEv2。

<!-- Media -->

<table><tr><td>Dataset</td><td>License</td><td>#Passages</td><td>#Test Queries</td></tr><tr><td>ArguAna (Wachsmuth et al., 2018)</td><td>CC BY 4.0</td><td>8674</td><td>1406</td></tr><tr><td>Climate-Fever (Diggelmann et al., 2020)</td><td>Not reported</td><td>5416593</td><td>1535</td></tr><tr><td>DBPedia (Auer et al., 2007)</td><td>CC BY-SA 3.0</td><td>4635922</td><td>400</td></tr><tr><td>FEVER (Thorne et al., 2018)</td><td>CC BY-SA 3.0</td><td/><td/></tr><tr><td>FiQA-2018 (Maia et al., 2018)</td><td>Not reported</td><td>57638</td><td>648</td></tr><tr><td>HotpotQA (Yang et al., 2018b)</td><td>CC BY-SA 4.0</td><td>5233329</td><td>7405</td></tr><tr><td>NFCorpus (Boteva et al., 2016)</td><td>Not reported</td><td>3633</td><td>323</td></tr><tr><td>NQ (Kwiatkowski et al., 2019)</td><td>CC BY-SA 3.0</td><td>2681468</td><td>3452</td></tr><tr><td>SCIDOCS (Cohan et al., 2020)</td><td>GNU General Public License v3.0</td><td>25657</td><td>1000</td></tr><tr><td>SciFact (Wadden et al., 2020)</td><td>CC BY-NC 2.0</td><td>5183</td><td>300</td></tr><tr><td>Quora</td><td>Not reported</td><td>522931</td><td>10000</td></tr><tr><td>Touché-2020 (Bondarenko et al., 2020)</td><td>CC BY 4.0</td><td>382545</td><td>49</td></tr><tr><td>TREC-COVID (Voorhees et al., 2021)</td><td>Dataset License Agreement</td><td>171332</td><td>50</td></tr></table>

<table><tbody><tr><td>数据集</td><td>许可协议</td><td>#段落数量</td><td>#测试查询数量</td></tr><tr><td>ArguAna数据集（瓦克斯穆特等人，2018年）</td><td>知识共享署名4.0国际许可协议（CC BY 4.0）</td><td>8674</td><td>1406</td></tr><tr><td>Climate - Fever数据集（迪格尔曼等人，2020年）</td><td>未报告</td><td>5416593</td><td>1535</td></tr><tr><td>DBPedia数据集（奥尔等人，2007年）</td><td>知识共享署名 - 相同方式共享3.0许可协议（CC BY - SA 3.0）</td><td>4635922</td><td>400</td></tr><tr><td>FEVER数据集（索恩等人，2018年）</td><td>知识共享署名 - 相同方式共享3.0许可协议（CC BY - SA 3.0）</td><td></td><td></td></tr><tr><td>FiQA - 2018数据集（马亚等人，2018年）</td><td>未报告</td><td>57638</td><td>648</td></tr><tr><td>HotpotQA数据集（杨等人，2018b）</td><td>知识共享署名 - 相同方式共享4.0许可协议（CC BY - SA 4.0）</td><td>5233329</td><td>7405</td></tr><tr><td>NFCorpus数据集（博特瓦等人，2016年）</td><td>未报告</td><td>3633</td><td>323</td></tr><tr><td>NQ数据集（克维亚特科夫斯基等人，2019年）</td><td>知识共享署名 - 相同方式共享3.0许可协议（CC BY - SA 3.0）</td><td>2681468</td><td>3452</td></tr><tr><td>SCIDOCS数据集（科汉等人，2020年）</td><td> GNU通用公共许可证第3.0版</td><td>25657</td><td>1000</td></tr><tr><td>SciFact数据集（瓦登等人，2020年）</td><td>知识共享署名 - 非商业性使用2.0许可协议（CC BY - NC 2.0）</td><td>5183</td><td>300</td></tr><tr><td>Quora问答平台</td><td>未报告</td><td>522931</td><td>10000</td></tr><tr><td>Touché - 2020数据集（邦达连科等人，2020年）</td><td>知识共享署名4.0国际许可协议（CC BY 4.0）</td><td>382545</td><td>49</td></tr><tr><td>TREC - COVID数据集（沃里斯等人，2021年）</td><td>数据集许可协议</td><td>171332</td><td>50</td></tr></tbody></table>

Table 8: BEIR dataset information.

表8：BEIR数据集信息。

<!-- Media -->

We also tested on the Open-QA benchmarks NQ, TQ, and SQuAD, each of which has approximately $9\mathrm{\;k}$ dev-set questions and muli-hop HoVer,whose development set has $4\mathrm{k}$ claims. In the compression evaluation $§\mathrm{B}$ ,we used models trained in-domain on NQ and HoVer,whose training sets contain ${79}\mathrm{k}$ and ${18}\mathrm{k}$ queries,respectively.

我们还在开放问答基准数据集NQ、TQ和SQuAD上进行了测试，每个数据集的开发集大约有$9\mathrm{\;k}$个问题，以及多跳的HoVer数据集，其开发集有$4\mathrm{k}$条声明。在压缩评估$§\mathrm{B}$中，我们使用了在NQ和HoVer数据集上进行领域内训练的模型，它们的训练集分别包含${79}\mathrm{k}$和${18}\mathrm{k}$个查询。

## F Implementation & Hyperparameters

## F 实现与超参数

We implement ColBERTv2 using Python 3.7, PyTorch 1.9, and HuggingFace Transformers 4.10 (Wolf et al., 2020), extending the original implementation of ColBERT by Khattab and Zaharia (2020). We use FAISS 1.7 (Johnson et al., 2019) for $k$ -means clustering, ${}^{9}$ though unlike ColBERT we do not use it for nearest-neighbor search. Instead, we implement our candidate generation mechanism (§3.5) using PyTorch primitives in Python.

我们使用Python 3.7、PyTorch 1.9和HuggingFace Transformers 4.10（Wolf等人，2020）实现了ColBERTv2，扩展了Khattab和Zaharia（2020）对ColBERT的原始实现。我们使用FAISS 1.7（Johnson等人，2019）进行$k$ -均值聚类${}^{9}$，不过与ColBERT不同的是，我们不使用它进行最近邻搜索。相反，我们使用Python中的PyTorch原语实现了候选生成机制（§3.5）。

We conducted our experiments on an internal cluster,typically using up to four 12GB Titan V GPUs for each of the inference tasks (e.g., indexing, computing distillation scores, and retrieval) and four 80GB A100 GPUs for training, though GPUs with smaller RAM can be used via gradient accumulation. Using this infrastructure, computing the distillation scores takes under a day, training a 64-way model on MS MARCO for 400,000 steps takes around five days, and indexing takes approximately two hours. We very roughly estimate an upper bound total of 20 GPU-months for all experimentation, development, and evaluation performed for this work over a period of several months.

我们在内部集群上进行实验，通常每个推理任务（例如索引、计算蒸馏分数和检索）最多使用四块12GB的Titan V GPU，训练则使用四块80GB的A100 GPU，不过也可以通过梯度累积使用内存较小的GPU。利用这个基础设施，计算蒸馏分数不到一天，在MS MARCO数据集上训练一个64路模型400,000步大约需要五天，索引大约需要两个小时。我们非常粗略地估计，在几个月的时间里，为这项工作进行的所有实验、开发和评估总共最多需要20个GPU月。

Like ColBERT, our encoder is a bert-base-uncased model that is shared between the query and passage encoders and which has ${110}\mathrm{M}$ parameters. We retain the default vector dimension suggested by Khattab and Zaharia (2020) and used in subsequent work, namely, $d = {128}$ . For the experiments reported in this paper, we train on MS MARCO training set. We use simple defaults with limited manual exploration on the official development set for the learning rate $\left( {10}^{-5}\right)$ ,batch size (32 examples),and warm up (for 20,000 steps) with linear decay.

与ColBERT一样，我们的编码器是一个bert-base-uncased模型，该模型在查询编码器和段落编码器之间共享，并且有${110}\mathrm{M}$个参数。我们保留了Khattab和Zaharia（2020）建议并在后续工作中使用的默认向量维度，即$d = {128}$。对于本文报告的实验，我们在MS MARCO训练集上进行训练。我们在官方开发集上使用简单的默认设置，对学习率$\left( {10}^{-5}\right)$、批量大小（32个样本）和线性衰减的热身步骤（20,000步）进行了有限的手动探索。

Hyperparameters corresponding to retrieval are explored in $§\mathrm{C}$ . We default to probe $= 2$ ,but use probe $= 4$ on the largest datasets,namely, MS MARCO and Wikipedia. By default we set candidates $=$ probe $* {2}^{12}$ ,but for Wikipedia we set candidates $=$ probe $* {2}^{13}$ and for MS MARCO we set candidates $=$ probe $* {2}^{14}$ . We leave extensive tuning of hyperparameters to future work.

与检索对应的超参数在$§\mathrm{C}$中进行了探索。我们默认使用探测$= 2$，但在最大的数据集（即MS MARCO和维基百科）上使用探测$= 4$。默认情况下，我们设置候选$=$探测$* {2}^{12}$，但对于维基百科，我们设置候选$=$探测$* {2}^{13}$，对于MS MARCO，我们设置候选$=$探测$* {2}^{14}$。我们将超参数的广泛调优留待未来工作。

We train on MS MARCO using 64-way tuples for distillation, sampling them from the top-500 retrieved passages per query. The training set of MS MARCO contains approximately ${800}\mathrm{k}$ queries, though only about ${500}\mathrm{k}$ have associated labels. We apply distillation using all ${800}\mathrm{k}$ queries,where each training example contains exactly one "positive", defined as a passage labeled as positive or the top-ranked passage by the cross-encoder teacher, irrespective of its label.

我们在MS MARCO数据集上使用64路元组进行蒸馏训练，从每个查询检索到的前500个段落中采样。MS MARCO的训练集大约包含${800}\mathrm{k}$个查询，但只有约${500}\mathrm{k}$个有相关标签。我们使用所有${800}\mathrm{k}$个查询进行蒸馏，每个训练样本恰好包含一个“正样本”，正样本定义为被标记为正例的段落或交叉编码器教师模型排名最高的段落，无论其标签如何。

We train for ${400}\mathrm{\;k}$ steps,initializing from a pre-finetuned checkpoint using 32-way training examples and ${150}\mathrm{k}$ steps. To generate the top- $k$ passages per training query, we apply two rounds, following Khattab et al. (2021b). We start from a model trained with hard triples (akin to Khattab et al. (2021b)), train with distillation, and then use the distilled model to retrieve for the second round of training. Preliminary experiments indicate that quality has low sensitivity to this initialization and two-round training, suggesting that both of them could be avoided to reduce the cost of training.

我们训练${400}\mathrm{\;k}$步，从一个使用32路训练样本预微调的检查点开始初始化，并训练${150}\mathrm{k}$步。为了为每个训练查询生成前$k$个段落，我们按照Khattab等人（2021b）的方法进行两轮操作。我们从一个使用硬三元组训练的模型开始（类似于Khattab等人（2021b）），进行蒸馏训练，然后使用蒸馏后的模型为第二轮训练进行检索。初步实验表明，质量对这种初始化和两轮训练的敏感性较低，这表明可以避免这两者以降低训练成本。

Unless otherwise stated, the results shown represent a single run. The latency results in $§3$ are averages of three runs. To evaluate for Open-QA retrieval, we use evaluation scripts from Khattab et al. (2021b), which checks if the short answer string appears in the (titled) Wikipedia passage. This adapts the DPR (Karpukhin et al., 2020) evaluation code. ${}^{10}$ We use the preprocessed Wikipedia Dec 2018 dump released by Karpukhin et al. (2020).

除非另有说明，所示结果均为单次运行的结果。$§3$中的延迟结果是三次运行的平均值。为了评估开放问答（Open-QA）检索，我们使用了Khattab等人（2021b）的评估脚本，该脚本会检查简短答案字符串是否出现在（带标题的）维基百科文章中。这对DPR（密集段落检索器，Dense Passage Retrieval，Karpukhin等人，2020）的评估代码进行了调整。${}^{10}$我们使用了Karpukhin等人（2020）发布的2018年12月预处理后的维基百科转储数据。

For out-of-domain evaluation, we elected to follow Thakur et al. (2021) and set the maximum document length of ColBERT, RocketQAv2, and ColBERTv2 to 300 tokens on BEIR and LoTTE. Formal et al. (2021a) selected maximum sequence length 256 for SPLADEv2 both on MS MARCO and on BEIR for both queries and documents, and we retained this default when testing their system on LoTTE. Unless otherwise stated, we keep the default query maximum sequence length for Col-BERTv2 and RocketQAv2, which is 32 tokens. For the ArguAna test in BEIR, as the queries are themselves long documents, we set the maximum query length used by ColBERTv2 and RocketQAv2 to 300. For Climate-FEVER, as the queries are relatively long sentence claims, we set the maximum query length used by ColBERTv2 to 64.

对于跨领域评估，我们选择遵循Thakur等人（2021）的方法，在BEIR和LoTTE数据集上将ColBERT、RocketQAv2和ColBERTv2的最大文档长度设置为300个词元。Formal等人（2021a）在MS MARCO和BEIR数据集上，针对查询和文档都为SPLADEv2选择了最大序列长度为256，我们在LoTTE数据集上测试他们的系统时保留了这一默认设置。除非另有说明，我们为Col-BERTv2和RocketQAv2保留默认的最大查询序列长度，即32个词元。对于BEIR数据集中的ArguAna测试，由于查询本身就是长文档，我们将ColBERTv2和RocketQAv2使用的最大查询长度设置为300。对于Climate-FEVER数据集，由于查询是相对较长的句子声明，我们将ColBERTv2使用的最大查询长度设置为64。

We use the open source BEIR implementation ${}^{11}$ and SPLADEv2 evaluation ${}^{12}$ code as the basis for our evaluations of SPLADEv2 and ANCE as well as for BM25 on LoTTE. We use the Anserini (Yang et al., 2018a) toolkit for BM25 on the Wikipedia Open-QA retrieval tests as in Khattab et al. (2021b). We use the implementation developed by the Rock-etQAv2 authors for evaluating RocketQAv2. ${}^{13}$

我们使用开源的BEIR实现代码${}^{11}$和SPLADEv2评估代码${}^{12}$作为基础，对SPLADEv2和ANCE以及LoTTE数据集上的BM25进行评估。我们使用Anserini（Yang等人，2018a）工具包，按照Khattab等人（2021b）的方法在维基百科开放问答检索测试中进行BM25评估。我们使用RocketQAv2作者开发的实现代码来评估RocketQAv2。${}^{13}$

---

<!-- Footnote -->

${}^{10}$ https://github.com/facebookresearch/DPR/blob/ main/dpr/data/qa_validation.py

${}^{10}$ https://github.com/facebookresearch/DPR/blob/ main/dpr/data/qa_validation.py

11https://github.com/UKPLab/beir

11https://github.com/UKPLab/beir

${}^{12}$ https://github.com/naver/splade

${}^{12}$ https://github.com/naver/splade

${}^{13}$ https://github.com/PaddlePaddle/RocketQA

${}^{13}$ https://github.com/PaddlePaddle/RocketQA

${}^{9}$ https://github.com/facebookresearch/faiss

${}^{9}$ https://github.com/facebookresearch/faiss

<!-- Footnote -->

---

<!-- Media -->

<table><tr><td>Topic</td><td>Communities</td><td>#Passages</td><td>#Search queries</td><td>#Forum queries</td></tr><tr><td rowspan="5">Writing</td><td>ell.stackexchange.com</td><td>108143</td><td>433</td><td>1196</td></tr><tr><td>literature.stackexchange.com</td><td>4778</td><td>7</td><td>58</td></tr><tr><td>writing.stackexchange.com</td><td>29330</td><td>23</td><td>163</td></tr><tr><td>linguistics.stackexchange.com</td><td>12302</td><td>22</td><td>116</td></tr><tr><td>worldbuilding.stackexchange.com</td><td>122519</td><td>12</td><td>470</td></tr><tr><td rowspan="4">Recreation</td><td>rpg.stackexchange.com</td><td>89066</td><td>91</td><td>621</td></tr><tr><td>boardgames.stackexchange.com</td><td>20340</td><td>67</td><td>179</td></tr><tr><td>scifi.stackexchange.com</td><td>102561</td><td>343</td><td>852</td></tr><tr><td>photo.stackexchange.com</td><td>51058</td><td>62</td><td>350</td></tr><tr><td rowspan="8">Science</td><td>chemistry.stackexchange.com</td><td>39435</td><td>245</td><td>267</td></tr><tr><td>stats.stackexchange.com</td><td>144084</td><td>137</td><td>949</td></tr><tr><td>academia.stackexchange.com</td><td>76450</td><td>66</td><td>302</td></tr><tr><td>astronomy.stackexchange.com</td><td>14580</td><td>15</td><td>88</td></tr><tr><td>earthscience.stackexchange.com</td><td>6734</td><td>10</td><td>50</td></tr><tr><td>engineering.stackexchange.com</td><td>12064</td><td>16</td><td>77</td></tr><tr><td>datascience.stackexchange.com</td><td>23234</td><td>15</td><td>156</td></tr><tr><td>philosophy.stackexchange.com</td><td>27061</td><td>34</td><td>124</td></tr><tr><td rowspan="5">Technology</td><td>superuser.com</td><td>418266</td><td>441</td><td>648</td></tr><tr><td>electronics.stackexchange.com</td><td>205891</td><td>118</td><td>314</td></tr><tr><td>askubuntu.com</td><td>296291</td><td>132</td><td>480</td></tr><tr><td>serverfault.com</td><td>323943</td><td>148</td><td>506</td></tr><tr><td>webapps.stackexchange.com</td><td>31831</td><td>77</td><td>55</td></tr><tr><td rowspan="11">Lifestyle</td><td>pets.stackexchange.com</td><td>10070</td><td>20</td><td>87</td></tr><tr><td>lifehacks.stackexchange.com</td><td>7893</td><td>2</td><td>50</td></tr><tr><td>gardening.stackexchange.com</td><td>20601</td><td>16</td><td>182</td></tr><tr><td>parenting.stackexchange.com</td><td>18357</td><td>10</td><td>87</td></tr><tr><td>crafts.stackexchange.com</td><td>3094</td><td>4</td><td>50</td></tr><tr><td>outdoors.stackexchange.com</td><td>13324</td><td>16</td><td>76</td></tr><tr><td>coffee.stackexchange.com</td><td>2249</td><td>11</td><td>50</td></tr><tr><td>music.stackexchange.com</td><td>47399</td><td>65</td><td>287</td></tr><tr><td>diy.stackexchange.com</td><td>82659</td><td>135</td><td>732</td></tr><tr><td>bicycles.stackexchange.com</td><td>35567</td><td>40</td><td>229</td></tr><tr><td>mechanics.stackexchange.com</td><td>27680</td><td>98</td><td>246</td></tr></table>

<table><tbody><tr><td>主题</td><td>社区</td><td>#段落</td><td>#搜索查询</td><td>#论坛查询</td></tr><tr><td rowspan="5">写作</td><td>ell.stackexchange.com</td><td>108143</td><td>433</td><td>1196</td></tr><tr><td>literature.stackexchange.com</td><td>4778</td><td>7</td><td>58</td></tr><tr><td>writing.stackexchange.com</td><td>29330</td><td>23</td><td>163</td></tr><tr><td>linguistics.stackexchange.com</td><td>12302</td><td>22</td><td>116</td></tr><tr><td>worldbuilding.stackexchange.com</td><td>122519</td><td>12</td><td>470</td></tr><tr><td rowspan="4">娱乐</td><td>rpg.stackexchange.com</td><td>89066</td><td>91</td><td>621</td></tr><tr><td>boardgames.stackexchange.com</td><td>20340</td><td>67</td><td>179</td></tr><tr><td>scifi.stackexchange.com</td><td>102561</td><td>343</td><td>852</td></tr><tr><td>photo.stackexchange.com</td><td>51058</td><td>62</td><td>350</td></tr><tr><td rowspan="8">科学</td><td>chemistry.stackexchange.com</td><td>39435</td><td>245</td><td>267</td></tr><tr><td>stats.stackexchange.com</td><td>144084</td><td>137</td><td>949</td></tr><tr><td>academia.stackexchange.com</td><td>76450</td><td>66</td><td>302</td></tr><tr><td>astronomy.stackexchange.com</td><td>14580</td><td>15</td><td>88</td></tr><tr><td>earthscience.stackexchange.com</td><td>6734</td><td>10</td><td>50</td></tr><tr><td>engineering.stackexchange.com</td><td>12064</td><td>16</td><td>77</td></tr><tr><td>datascience.stackexchange.com</td><td>23234</td><td>15</td><td>156</td></tr><tr><td>philosophy.stackexchange.com</td><td>27061</td><td>34</td><td>124</td></tr><tr><td rowspan="5">技术</td><td>superuser.com</td><td>418266</td><td>441</td><td>648</td></tr><tr><td>electronics.stackexchange.com</td><td>205891</td><td>118</td><td>314</td></tr><tr><td>askubuntu.com</td><td>296291</td><td>132</td><td>480</td></tr><tr><td>serverfault.com</td><td>323943</td><td>148</td><td>506</td></tr><tr><td>webapps.stackexchange.com</td><td>31831</td><td>77</td><td>55</td></tr><tr><td rowspan="11">生活方式</td><td>pets.stackexchange.com</td><td>10070</td><td>20</td><td>87</td></tr><tr><td>lifehacks.stackexchange.com</td><td>7893</td><td>2</td><td>50</td></tr><tr><td>gardening.stackexchange.com</td><td>20601</td><td>16</td><td>182</td></tr><tr><td>parenting.stackexchange.com</td><td>18357</td><td>10</td><td>87</td></tr><tr><td>crafts.stackexchange.com</td><td>3094</td><td>4</td><td>50</td></tr><tr><td>outdoors.stackexchange.com</td><td>13324</td><td>16</td><td>76</td></tr><tr><td>coffee.stackexchange.com</td><td>2249</td><td>11</td><td>50</td></tr><tr><td>music.stackexchange.com</td><td>47399</td><td>65</td><td>287</td></tr><tr><td>diy.stackexchange.com</td><td>82659</td><td>135</td><td>732</td></tr><tr><td>bicycles.stackexchange.com</td><td>35567</td><td>40</td><td>229</td></tr><tr><td>mechanics.stackexchange.com</td><td>27680</td><td>98</td><td>246</td></tr></tbody></table>

Table 9: Per-community distribution of LoTTE dev dataset passages and questions.

表9：LoTTE开发数据集段落和问题的社区分布情况。

<!-- Media -->