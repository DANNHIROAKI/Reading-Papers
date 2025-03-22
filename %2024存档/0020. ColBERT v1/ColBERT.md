# ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT

# ColBERT：通过基于BERT的上下文延迟交互实现高效有效的段落搜索

Omar Khattab

奥马尔·哈塔卜

Stanford University

斯坦福大学

okhattab@stanford.edu

Matei Zaharia

马特·扎哈里亚

Stanford University

斯坦福大学

matei@cs.stanford.edu

## ABSTRACT

## 摘要

Recent progress in Natural Language Understanding (NLU) is driving fast-paced advances in Information Retrieval (IR), largely owed to fine-tuning deep language models (LMs) for document ranking. While remarkably effective, the ranking models based on these LMs increase computational cost by orders of magnitude over prior approaches, particularly as they must feed each query-document pair through a massive neural network to compute a single relevance score. To tackle this, we present ColBERT, a novel ranking model that adapts deep LMs (in particular, BERT) for efficient retrieval. ColBERT introduces a late interaction architecture that independently encodes the query and the document using BERT and then employs a cheap yet powerful interaction step that models their fine-grained similarity. By delaying and yet retaining this fine-granular interaction, ColBERT can leverage the expressiveness of deep LMs while simultaneously gaining the ability to pre-compute document representations offline, considerably speeding up query processing. Beyond reducing the cost of re-ranking the documents retrieved by a traditional model, ColBERT's pruning-friendly interaction mechanism enables leveraging vector-similarity indexes for end-to-end retrieval directly from a large document collection. We extensively evaluate ColBERT using two recent passage search datasets. Results show that ColBERT's effectiveness is competitive with existing BERT-based models (and outperforms every non-BERT baseline), while executing two orders-of-magnitude faster and requiring four orders-of-magnitude fewer FLOPs per query.

自然语言理解（NLU）领域的最新进展推动了信息检索（IR）的快速发展，这在很大程度上归功于为文档排序微调深度语言模型（LM）。虽然基于这些语言模型的排序模型非常有效，但与先前的方法相比，它们的计算成本增加了几个数量级，特别是因为它们必须将每个查询 - 文档对输入到一个庞大的神经网络中，才能计算出一个相关性得分。为了解决这个问题，我们提出了ColBERT，这是一种新颖的排序模型，它对深度语言模型（特别是BERT）进行调整以实现高效检索。ColBERT引入了一种延迟交互架构，该架构使用BERT分别对查询和文档进行编码，然后采用一个低成本但强大的交互步骤来建模它们的细粒度相似度。通过延迟但保留这种细粒度交互，ColBERT可以利用深度语言模型的表达能力，同时获得离线预计算文档表示的能力，从而显著加快查询处理速度。除了降低对传统模型检索到的文档进行重新排序的成本外，ColBERT的利于剪枝的交互机制还能够利用向量相似度索引直接从大型文档集合中进行端到端检索。我们使用两个最新的段落搜索数据集对ColBERT进行了广泛评估。结果表明，ColBERT的有效性与现有的基于BERT的模型相当（并且优于所有非BERT基线模型），同时执行速度快两个数量级，每个查询所需的浮点运算次数少四个数量级。

## ACM Reference format:

## ACM引用格式：

Omar Khattab and Matei Zaharia. 2020. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. In Proceedings of Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, Virtual Event, China, July 25-30, 2020 (SIGIR '20), 10 pages. DOI: 10.1145/3397271.3401075

奥马尔·哈塔卜和马特·扎哈里亚。2020年。ColBERT：通过基于BERT的上下文延迟交互实现高效有效的段落搜索。收录于《第43届ACM SIGIR国际信息检索研究与发展会议论文集》，虚拟会议，中国，2020年7月25 - 30日（SIGIR '20），共10页。DOI：10.1145/3397271.3401075

## 1 INTRODUCTION

## 1 引言

Over the past few years, the Information Retrieval (IR) community has witnessed the introduction of a host of neural ranking models, including DRMM [7], KNRM [4, 36], and Duet [20, 22]. In contrast

在过去几年中，信息检索（IR）领域见证了许多神经排序模型的引入，包括深度相关性匹配模型（DRMM）[7]、核化相关性匹配模型（KNRM）[4, 36]和二重奏模型（Duet）[20, 22]。相比之下

<!-- Media -->

<!-- figureText: ${10}^{5}$ Bag-of-Words (BoW) Model BERT-large BERT-base ColBERT (full retrieval) fT+ConvKNRM ColBERT (re-rank) 0.30 0.35 0.40 MRR@10 BoW Model with NLU Augmentation Query Latency (ms) Neural Matching Mode ${10}^{4}$ Deep Language Model ColBERT (ours) ${10}^{3}$ ${10}^{2}$ BM25 0.20 0.25 -->

<img src="https://cdn.noedgeai.com/0195afb9-1285-7cff-a7ae-bc74a3288ac4_0.jpg?x=938&y=501&w=698&h=347&r=0"/>

Figure 1: Effectiveness (MRR@10) versus Mean Query Latency (log-scale) for a number of representative ranking models on MS MARCO Ranking [24]. The figure also shows ColBERT. Neural re-rankers run on top of the official BM25 top-1000 results and use a Tesla V100 GPU. Methodology and detailed results are in §4.

图1：在MS MARCO排序数据集[24]上，多个代表性排序模型的有效性（MRR@10）与平均查询延迟（对数刻度）的关系。图中还展示了ColBERT。神经重排器基于官方BM25的前1000个结果运行，并使用特斯拉V100 GPU。方法和详细结果见§4。

<!-- Media -->

to prior learning-to-rank methods that rely on hand-crafted features, these models employ embedding-based representations of queries and documents and directly model local interactions (i.e., fine-granular relationships) between their contents. Among them, a recent approach has emerged that fine-tunes deep pre-trained language models (LMs) like ELMo [29] and BERT [5] for estimating relevance. By computing deeply-contextualized semantic representations of query-document pairs, these LMs help bridge the pervasive vocabulary mismatch $\left\lbrack  {{21},{42}}\right\rbrack$ between documents and queries [30]. Indeed, in the span of just a few months, a number of ranking models based on BERT have achieved state-of-the-art results on various retrieval benchmarks $\left\lbrack  {3,{18},{25},{39}}\right\rbrack$ and have been proprietarily adapted for deployment by ${\mathrm{{Google}}}^{1}$ and ${\mathrm{{Bing}}}^{2}$ .

与之前依赖手工特征的排序学习方法不同，这些模型采用基于嵌入的查询和文档表示，并直接对其内容之间的局部交互（即细粒度关系）进行建模。其中，最近出现了一种方法，即对像ELMo [29]和BERT [5]这样的深度预训练语言模型（LM）进行微调以估计相关性。通过计算查询 - 文档对的深度上下文语义表示，这些语言模型有助于弥合文档和查询之间普遍存在的词汇不匹配问题$\left\lbrack  {{21},{42}}\right\rbrack$[30]。事实上，在短短几个月的时间里，一些基于BERT的排序模型在各种检索基准测试中取得了最先进的结果$\left\lbrack  {3,{18},{25},{39}}\right\rbrack$，并已被${\mathrm{{Google}}}^{1}$和${\mathrm{{Bing}}}^{2}$进行专有化调整以进行部署。

However, the remarkable gains delivered by these LMs come at a steep increase in computational cost. Hofstätter et al. [9] and MacAvaney et al. [18] observe that BERT-based models in the literature are ${100} - {1000} \times$ more computationally expensive than prior models-some of which are arguably not inexpensive to begin with [13]. This quality-cost tradeoff is summarized by Figure 1, which compares two BERT-based rankers [25, 27] against a representative set of ranking models. The figure uses MS MARCO Ranking [24], a recent collection of $9\mathrm{M}$ passages and $1\mathrm{M}$ queries from Bing’s logs. It reports retrieval effectiveness (MRR@10) on the official validation set as well as average query latency (log-scale) using a high-end server that dedicates one Tesla V100 GPU per query for neural re-rankers. Following the re-ranking setup of MS MARCO, ColBERT (re-rank), the Neural Matching Models, and the Deep LMs re-rank the MS MARCO's official top-1000 documents per query. Other methods, including ColBERT (full retrieval), directly retrieve the top-1000 results from the entire collection.

然而，这些语言模型（LM）带来的显著提升是以计算成本的大幅增加为代价的。霍夫施泰特（Hofstätter）等人[9]和麦卡瓦尼（MacAvaney）等人[18]观察到，文献中基于BERT的模型比之前的模型计算成本高出${100} - {1000} \times$——其中一些模型原本的计算成本就不低[13]。这种质量 - 成本的权衡关系总结在图1中，该图将两个基于BERT的排序器[25, 27]与一组具有代表性的排序模型进行了比较。该图使用了MS MARCO排序数据集[24]，这是最近从必应（Bing）日志中收集的$9\mathrm{M}$段落和$1\mathrm{M}$查询的集合。它报告了官方验证集上的检索效果（MRR@10），以及使用高端服务器（为每个查询分配一个特斯拉V100 GPU用于神经重排序器）的平均查询延迟（对数刻度）。按照MS MARCO的重排序设置，ColBERT（重排序）、神经匹配模型和深度语言模型会对每个查询对应的MS MARCO官方前1000个文档进行重排序。其他方法，包括ColBERT（全检索），则直接从整个集合中检索前1000个结果。

---

<!-- Footnote -->

Permission to make digital or hard copies of all or part of this work for personal or

允许个人或

classroom use is granted without fee provided that copies are not made or distributed

课堂使用本作品的全部或部分内容的数字或硬拷贝，前提是不进行复制或分发

on the first page. Copyrights for components of this work owned by others than the

在首页。必须尊重本作品中除作者之外其他人拥有的组件的版权。允许进行带引用的摘要。否则，进行复制、

author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or

重新发布、发布到服务器或重新分发给列表，需要事先获得特定许可和/或支付费用。请向permissions@acm.org请求许可。978 - 1 - 4503 - 8016 - 4/20/07...\$15.00 DOI: 10.1145/3397271.3401075

republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. 978-1-4503-8016-4/20/07...\$15.00 DOI: 10.1145/3397271.3401075

重新发布、发布到服务器或重新分发给列表，需要事先获得特定许可和/或支付费用。请向permissions@acm.org请求许可。978 - 1 - 4503 - 8016 - 4/20/07...\$15.00 DOI: 10.1145/3397271.3401075

${}^{1}$ https://blog.google/products/search/search-language-understanding-bert/

${}^{1}$ https://blog.google/products/search/search-language-understanding-bert/

${}^{2}$ https://azure.microsoft.com/en-us/blog/bing-delivers-its-largest-improvement-in-search-experience-using-azure-gpus/

${}^{2}$ https://azure.microsoft.com/en-us/blog/bing-delivers-its-largest-improvement-in-search-experience-using-azure-gpus/

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: Document Query Document Query Document (c) All-to-all Interaction (d) Late Interaction (e.g., BERT) (i.e., the proposed ColBERT) Query Document Query (a) Representation-based Similarity (b) Query-Document Interaction (e.g., DSSM, SNRM) (e.g., DRMM, KNRM, Conv-KNRM) -->

<img src="https://cdn.noedgeai.com/0195afb9-1285-7cff-a7ae-bc74a3288ac4_1.jpg?x=147&y=234&w=1506&h=412&r=0"/>

Figure 2: Schematic diagrams illustrating query-document matching paradigms in neural IR. The figure contrasts existing approaches (sub-figures (a), (b), and (c)) with the proposed late interaction paradigm (sub-figure (d)).

图2：说明神经信息检索（IR）中查询 - 文档匹配范式的示意图。该图将现有方法（子图（a）、（b）和（c））与所提出的后期交互范式（子图（d））进行了对比。

<!-- Media -->

As the figure shows, BERT considerably improves search precision, raising MRR@10 by almost 7% against the best previous methods; simultaneously, it increases latency by up to tens of thousands of milliseconds even with a high-end GPU. This poses a challenging tradeoff since raising query response times by as little as ${100}\mathrm{\;{ms}}$ is known to impact user experience and even measurably diminish revenue [17]. To tackle this problem, recent work has started exploring using Natural Language Understanding (NLU) techniques to augment traditional retrieval models like BM25 [32]. For example, Nogueira et al. [26, 28] expand documents with NLU-generated queries before indexing with BM25 scores and Dai & Callan [2] replace BM25's term frequency with NLU-estimated term importance. Despite successfully reducing latency, these approaches generally reduce precision substantially relative to BERT.

如图所示，BERT显著提高了搜索精度，与之前的最佳方法相比，MRR@10提高了近7%；同时，即使使用高端GPU，其延迟也会增加多达数万毫秒。这带来了一个具有挑战性的权衡问题，因为已知查询响应时间仅增加${100}\mathrm{\;{ms}}$就会影响用户体验，甚至会显著减少收入[17]。为了解决这个问题，近期的工作开始探索使用自然语言理解（NLU）技术来增强像BM25这样的传统检索模型[32]。例如，诺盖拉（Nogueira）等人[26, 28]在使用BM25分数进行索引之前，用NLU生成的查询扩展文档，戴（Dai）和卡兰（Callan）[2]用NLU估计的词项重要性取代了BM25的词频。尽管这些方法成功降低了延迟，但与BERT相比，它们通常会大幅降低精度。

To reconcile efficiency and contextualization in IR, we propose ColBERT, a ranking model based on contextualized late interaction over BERT. As the name suggests, ColBERT proposes a novel late interaction paradigm for estimating relevance between a query $q$ and a document $d$ . Under late interaction, $q$ and $d$ are separately encoded into two sets of contextual embeddings, and relevance is evaluated using cheap and pruning-friendly computations between both sets-that is, fast computations that enable ranking without exhaustively evaluating every possible candidate.

为了在信息检索中兼顾效率和上下文感知，我们提出了ColBERT，这是一种基于BERT的上下文后期交互的排序模型。顾名思义，ColBERT提出了一种新颖的后期交互范式，用于估计查询$q$和文档$d$之间的相关性。在后期交互中，$q$和$d$分别被编码为两组上下文嵌入，并且使用两组之间廉价且便于剪枝的计算来评估相关性——即能够在不详尽评估每个可能候选的情况下进行排序的快速计算。

Figure 2 contrasts our proposed late interaction approach with existing neural matching paradigms. On the left, Figure 2 (a) illustrates representation-focused rankers, which independently compute an embedding for $q$ and another for $d$ and estimate relevance as a single similarity score between two vectors $\left\lbrack  {{12},{41}}\right\rbrack$ . Moving to the right, Figure 2 (b) visualizes typical interaction-focused rankers. Instead of summarizing $q$ and $d$ into individual embeddings,these rankers model word- and phrase-level relationships across $q$ and $d$ and match them using a deep neural network (e.g., with CNNs/MLPs [22] or kernels [36]). In the simplest case, they feed the neural network an interaction matrix that reflects the similiarity between every pair of words across $q$ and $d$ . Further right,Figure 2 (c) illustrates a more powerful interaction-based paradigm, which models the interactions between words within as well as across $q$ and $d$ at the same time, as in BERT's transformer architecture [25].

图2将我们提出的后期交互方法与现有的神经匹配范式进行了对比。在左侧，图2(a)展示了以表示为中心的排序器，该排序器独立计算$q$的嵌入表示和$d$的嵌入表示，并将相关性估计为两个向量$\left\lbrack  {{12},{41}}\right\rbrack$之间的单一相似度得分。往右看，图2(b)展示了典型的以交互为中心的排序器。这些排序器并不将$q$和$d$总结为单独的嵌入表示，而是对$q$和$d$之间的词级和短语级关系进行建模，并使用深度神经网络（例如，使用卷积神经网络/多层感知器[22]或核函数[36]）对它们进行匹配。在最简单的情况下，它们将一个反映$q$和$d$中每对词之间相似度的交互矩阵输入到神经网络中。再往右，图2(c)展示了一种更强大的基于交互的范式，该范式同时对$q$和$d$内部以及它们之间的词的交互进行建模，就像BERT的Transformer架构[25]那样。

These increasingly expressive architectures are in tension. While interaction-based models (i.e., Figure 2 (b) and (c)) tend to be superior for IR tasks [8, 21], a representation-focused model-by isolating the computations among $q$ and $d$ -makes it possible to precompute document representations offline [41], greatly reducing the computational load per query. In this work, we observe that the fine-grained matching of interaction-based models and the pre-computation of document representations of representation-based models can be combined by retaining yet judiciously delaying the query-document interaction. Figure 2 (d) illustrates an architecture that precisely does so. As illustrated, every query embedding interacts with all document embeddings via a MaxSim operator, which computes maximum similarity (e.g., cosine similarity), and the scalar outputs of these operators are summed across query terms. This paradigm allows ColBERT to exploit deep LM-based representations while shifting the cost of encoding documents offline and amortizing the cost of encoding the query once across all ranked documents. Additionally, it enables ColBERT to leverage vector-similarity search indexes (e.g., $\left\lbrack  {1,{15}}\right\rbrack$ ) to retrieve the top- $k$ results directly from a large document collection,substantially improving recall over models that only re-rank the output of term-based retrieval.

这些表达能力越来越强的架构之间存在矛盾。虽然基于交互的模型（即图2(b)和(c)）在信息检索（IR）任务中往往表现更优[8, 21]，但以表示为中心的模型通过将$q$和$d$之间的计算隔离开来，使得离线预计算文档表示成为可能[41]，从而大大降低了每个查询的计算负载。在这项工作中，我们发现，通过保留但明智地延迟查询 - 文档交互，可以将基于交互的模型的细粒度匹配和基于表示的模型的文档表示预计算结合起来。图2(d)展示了一种正是这样做的架构。如图所示，每个查询嵌入通过一个MaxSim算子与所有文档嵌入进行交互，该算子计算最大相似度（例如，余弦相似度），并且这些算子的标量输出会在查询词项上求和。这种范式使ColBERT能够利用基于深度语言模型（LM）的表示，同时将文档编码的成本转移到离线阶段，并将查询编码的成本分摊到所有排序的文档上。此外，它使ColBERT能够利用向量相似度搜索索引（例如，$\left\lbrack  {1,{15}}\right\rbrack$）直接从大型文档集合中检索前$k$个结果，与仅对基于词项的检索输出进行重排序的模型相比，显著提高了召回率。

As Figure 1 illustrates, ColBERT can serve queries in tens or few hundreds of milliseconds. For instance, when used for re-ranking as in "ColBERT (re-rank)",it delivers over ${170} \times$ speedup (and requires ${14},{000} \times$ fewer FLOPs) relative to existing BERT-based models, while being more effective than every non-BERT baseline (§4.2 & 4.3). ColBERT's indexing-the only time it needs to feed documents through BERT-is also practical: it can index the MS MARCO collection of $9\mathrm{M}$ passages in about 3 hours using a single server with four GPUs (§4.5), retaining its effectiveness with a space footprint of as little as few tens of GiBs. Our extensive ablation study (§4.4) shows that late interaction, its implementation via MaxSim operations, and crucial design choices within our BERT-based encoders are all essential to ColBERT's effectiveness.

如图1所示，ColBERT可以在几十或几百毫秒内处理查询。例如，当像“ColBERT（重排序）”那样用于重排序时，与现有的基于BERT的模型相比，它的速度提升超过${170} \times$（并且所需的浮点运算次数（FLOPs）减少${14},{000} \times$），同时比所有非BERT基线模型更有效（§4.2和4.3）。ColBERT的索引编制（这是它唯一需要将文档输入BERT的阶段）也是可行的：使用一台配备四个GPU的服务器，它可以在大约3小时内对包含$9\mathrm{M}$个段落的MS MARCO集合进行索引编制（§4.5），并且只需几十吉字节（GiB）的存储空间就能保持其有效性。我们广泛的消融研究（§4.4）表明，后期交互、通过MaxSim操作实现的后期交互以及我们基于BERT的编码器中的关键设计选择，对于ColBERT的有效性都至关重要。

Our main contributions are as follows.

我们的主要贡献如下。

(1) We propose late interaction (§3.1) as a paradigm for efficient and effective neural ranking.

(1) 我们提出后期交互（§3.1）作为一种高效且有效的神经排序范式。

(2) We present ColBERT (§3.2 & 3.3), a highly-effective model that employs novel BERT-based query and document encoders within the late interaction paradigm.

(2) 我们介绍了ColBERT（§3.2和3.3），这是一个在后期交互范式下采用新颖的基于BERT的查询和文档编码器的高效模型。

(3) We show how to leverage ColBERT both for re-ranking on top of a term-based retrieval model (§3.5) and for searching a full collection using vector similarity indexes (§3.6).

(3) 我们展示了如何利用ColBERT在基于词项的检索模型之上进行重排序（§3.5）以及使用向量相似度索引搜索完整集合（§3.6）。

(4) We evaluate ColBERT on MS MARCO and TREC CAR, two recent passage search collections.

(4) 我们在MS MARCO和TREC CAR这两个近期的段落搜索集合上对ColBERT进行了评估。

## 2 RELATED WORK

## 2 相关工作

Neural Matching Models. Over the past few years, IR researchers have introduced numerous neural architectures for ranking. In this work, we compare against KNRM [4, 36], Duet [20, 22], Con-vKNRM [4], and fastText+ConvKNRM [10]. KNRM proposes a differentiable kernel-pooling technique for extracting matching signals from an interaction matrix, while Duet combines signals from exact-match-based as well as embedding-based similarities for ranking. Introduced in 2018,ConvKNRM learns to match $n$ - grams in the query and the document. Lastly, fastText+ConvKNRM (abbreviated fT+ConvKNRM) tackles the absence of rare words from typical word embeddings lists by adopting sub-word token embeddings.

神经匹配模型。在过去几年里，信息检索（IR）研究人员引入了众多用于排序的神经架构。在这项工作中，我们将与核化神经排名模型（KNRM）[4, 36]、二重奏模型（Duet）[20, 22]、卷积核化神经排名模型（ConvKNRM）[4]以及快速文本+卷积核化神经排名模型（fastText+ConvKNRM）[10]进行比较。KNRM提出了一种可微的核池化技术，用于从交互矩阵中提取匹配信号，而Duet则结合了基于精确匹配和基于嵌入的相似度信号进行排序。ConvKNRM于2018年提出，用于学习查询和文档中$n$ - 元组的匹配。最后，快速文本+卷积核化神经排名模型（缩写为fT+ConvKNRM）通过采用子词标记嵌入来解决典型词嵌入列表中缺少稀有词的问题。

In 2018, Zamani et al. [41] introduced SNRM, a representation-focused IR model that encodes each query and each document as a single, sparse high-dimensional vector of "latent terms". By producing a sparse-vector representation for each document, SNRM is able to use a traditional IR inverted index for representing documents, allowing fast end-to-end retrieval. Despite highly promising results and insights, SNRM's effectiveness is substantially outperformed by the state of the art on the datasets with which it was evaluated (e.g., see [18, 38]). While SNRM employs sparsity to allow using inverted indexes, we relax this assumption and compare a (dense) BERT-based representation-focused model against our late-interaction ColBERT in our ablation experiments in §4.4. For a detailed overview of existing neural ranking models, we refer the readers to two recent surveys of the literature $\left\lbrack  {8,{21}}\right\rbrack$ .

2018年，扎马尼（Zamani）等人[41]引入了稀疏神经排名模型（SNRM），这是一个注重表示的信息检索模型，它将每个查询和每个文档编码为一个单一的、稀疏的高维“潜在词项”向量。通过为每个文档生成稀疏向量表示，SNRM能够使用传统的信息检索倒排索引来表示文档，从而实现快速的端到端检索。尽管取得了非常有前景的结果和见解，但在对其进行评估的数据集上，SNRM的有效性远不如当前的先进技术（例如，参见[18, 38]）。虽然SNRM利用稀疏性来允许使用倒排索引，但我们放宽了这一假设，并在§4.4的消融实验中，将一个（密集的）基于BERT的注重表示的模型与我们的后期交互模型ColBERT进行了比较。有关现有神经排名模型的详细概述，我们建议读者参考最近的两篇文献综述$\left\lbrack  {8,{21}}\right\rbrack$。

Language Model Pretraining for IR. Recent work in NLU emphasizes the importance pre-training language representation models in an unsupervised fashion before subsequently fine-tuning them on downstream tasks. A notable example is BERT [5], a bidirectional transformer-based language model whose fine-tuning advanced the state of the art on various NLU benchmarks. Nogueira et al. [25], MacAvaney et al. [18], and Dai & Callan [3] investigate incorporating such LMs (mainly BERT, but also ELMo [29]) on different ranking datasets. As illustrated in Figure 2 (c), the common approach (and the one adopted by Nogueira et al. on MS MARCO and TREC CAR) is to feed the query-document pair through BERT and use an MLP on top of BERT's [CLS] output token to produce a relevance score. Subsequent work by Nogueira et al. [27] introduced duoBERT, which fine-tunes BERT to compare the relevance of a pair of documents given a query. Relative to their single-document BERT, this gives duoBERT a 1% MRR@10 advantage on MS MARCO while increasing the cost by at least ${1.4} \times$ .

用于信息检索的语言模型预训练。自然语言理解（NLU）的最新工作强调了在无监督的情况下预训练语言表示模型，然后在下游任务上对其进行微调的重要性。一个显著的例子是双向编码器表示来自变换器（BERT）[5]，这是一种基于双向变换器的语言模型，其微调推动了各种自然语言理解基准测试的技术发展。诺盖拉（Nogueira）等人[25]、麦卡瓦尼（MacAvaney）等人[18]以及戴（Dai）和卡兰（Callan）[3]研究了在不同的排名数据集上整合此类语言模型（主要是BERT，但也包括深度语境化词表示（ELMo）[29]）。如图2（c）所示，常见的方法（也是诺盖拉等人在微软机器阅读理解数据集（MS MARCO）和文本检索会议文集数据集（TREC CAR）上采用的方法）是将查询 - 文档对输入BERT，并在BERT的[CLS]输出标记之上使用多层感知机（MLP）来生成相关性得分。诺盖拉等人随后的工作[27]引入了对偶BERT（duoBERT），它对BERT进行微调，以比较给定查询下一对文档的相关性。相对于他们的单文档BERT，这使得duoBERT在MS MARCO上的前10名平均倒数排名（MRR@10）提高了1%，同时成本至少增加了${1.4} \times$。

BERT Optimizations. As discussed in $§1$ ,these LM-based rankers can be highly expensive in practice. While ongoing efforts in the NLU literature for distilling [14, 33], compressing [40], and pruning [19] BERT can be instrumental in narrowing this gap, they generally achieve significantly smaller speedups than our redesigned architecture for IR, due to their generic nature, and more aggressive optimizations often come at the cost of lower quality.

BERT优化。正如$§1$中所讨论的，这些基于语言模型的排名器在实践中可能成本很高。虽然自然语言理解文献中正在进行的关于提炼[14, 33]、压缩[40]和剪枝[19]BERT的工作有助于缩小这一差距，但由于其通用性，它们通常比我们为信息检索重新设计的架构实现的加速要小得多，而且更激进的优化往往会以降低质量为代价。

<!-- Media -->

<!-- figureText: Query Encoder, ${f}_{O}$ score MaxSim ... Document Encoder, ${f}_{D}$ Offline Indexing ... Document Query -->

<img src="https://cdn.noedgeai.com/0195afb9-1285-7cff-a7ae-bc74a3288ac4_2.jpg?x=938&y=234&w=690&h=420&r=0"/>

Figure 3: The general architecture of ColBERT given a query $q$ and a document $d$ .

图3：给定查询$q$和文档$d$时ColBERT的总体架构。

<!-- Media -->

Efficient NLU-based Models. Recently, a direction emerged that employs expensive NLU computation offline. This includes doc2query [28] and DeepCT [2]. The doc2query model expands each document with a pre-defined number of synthetic queries queries generated by a seq2seq transformer model that is trained to generate queries given a document. It then relies on a BM25 index for retrieval from the (expanded) documents. DeepCT uses BERT to produce the term frequency component of BM25 in a context-aware manner, essentially representing a feasible realization of the term-independence assumption with neural networks [23]. Lastly, docTTTTTquery [26] is identical to doc2query except that it fine-tunes a pre-trained model (namely, T5 [31]) for generating the predicted queries.

高效的基于自然语言理解的模型。最近，出现了一种离线使用昂贵的自然语言理解计算的方向。这包括文档到查询模型（doc2query）[28]和深度上下文词项权重模型（DeepCT）[2]。doc2query模型用预定义数量的合成查询扩展每个文档，这些合成查询由一个序列到序列的变换器模型生成，该模型经过训练可以根据文档生成查询。然后，它依靠BM25索引从（扩展后的）文档中进行检索。DeepCT使用BERT以上下文感知的方式生成BM25的词项频率分量，本质上代表了用神经网络对词项独立性假设的一种可行实现[23]。最后，文档到多个查询模型（docTTTTTquery）[26]与doc2query相同，只是它微调了一个预训练模型（即文本到文本转移变压器（T5）[31]）来生成预测查询。

Concurrently with our drafting of this paper, Hofstätter et al. [11] published their Transformer-Kernel (TK) model. At a high level, TK improves the KNRM architecture described earlier: while KNRM employs kernel pooling on top of word-embedding-based interaction, TK uses a Transformer [34] component for contextually encoding queries and documents before kernel pooling. TK establishes a new state-of-the-art for non-BERT models on MS MARCO (Dev); however, the best non-ensemble MRR@10 it achieves is 31% while ColBERT reaches up to 36%. Moreover, due to indexing document representations offline and employing a MaxSim-based late interaction mechanism, ColBERT is much more scalable, enabling end-to-end retrieval which is not supported by TK.

在我们撰写本文的同时，霍夫施泰特（Hofstätter）等人 [11] 发表了他们的Transformer内核（Transformer-Kernel，TK）模型。总体而言，TK改进了前文所述的核化神经网络匹配模型（KNRM）架构：KNRM在基于词嵌入的交互之上采用核池化，而TK在核池化之前使用Transformer [34] 组件对查询和文档进行上下文编码。TK在微软机器阅读理解数据集（MS MARCO）（开发集）上为非BERT模型树立了新的最优水平；然而，它所达到的最佳非集成式前10名平均倒数排名（MRR@10）为31%，而ColBERT可达36%。此外，由于离线索引文档表示并采用基于最大相似度（MaxSim）的后期交互机制，ColBERT的可扩展性更强，能够实现端到端检索，而TK不支持这一点。

## 3 COLBERT

## 3 ColBERT模型

ColBERT prescribes a simple framework for balancing the quality and cost of neural IR, particularly deep language models like BERT. As introduced earlier, delaying the query-document interaction can facilitate cheap neural re-ranking (i.e., through pre-computation) and even support practical end-to-end neural retrieval (i.e., through pruning via vector-similarity search). ColBERT addresses how to do so while still preserving the effectiveness of state-of-the-art models, which condition the bulk of their computations on the joint query-document pair.

ColBERT规定了一个简单的框架，用于平衡神经信息检索（IR）的质量和成本，特别是像BERT这样的深度语言模型。如前文所述，延迟查询 - 文档交互可以促进低成本的神经重排序（即通过预计算），甚至支持实用的端到端神经检索（即通过向量相似度搜索进行剪枝）。ColBERT解决了如何在保持最优模型有效性的同时实现这一点的问题，这些最优模型的大部分计算都基于联合查询 - 文档对。

Even though ColBERT's late-interaction framework can be applied to a wide variety of architectures (e.g., CNNs, RNNs, transformers, etc.), we choose to focus this work on bi-directional transformer-based encoders (i.e., BERT) owing to their state-of-the-art effectiveness yet very high computational cost.

尽管ColBERT的后期交互框架可以应用于多种架构（例如卷积神经网络（CNNs）、循环神经网络（RNNs）、Transformer等），但由于基于双向Transformer的编码器（即BERT）具有最优的有效性，但计算成本非常高，我们选择将这项工作的重点放在它们上。

### 3.1 Architecture

### 3.1 架构

Figure 3 depicts the general architecture of ColBERT, which comprises: (a) a query encoder ${f}_{Q}$ ,(b) a document encoder ${f}_{D}$ ,and (c) the late interaction mechanism. Given a query $q$ and document $d$ , ${f}_{Q}$ encodes $q$ into a bag of fixed-size embeddings ${E}_{q}$ while ${f}_{D}$ encodes $d$ into another bag ${E}_{d}$ . Crucially,each embeddings in ${E}_{q}$ and ${E}_{d}$ is contextualized based on the other terms in $q$ or $d$ ,respectively. We describe our BERT-based encoders in §3.2.

图3描绘了ColBERT的总体架构，它包括：（a）查询编码器 ${f}_{Q}$ ，（b）文档编码器 ${f}_{D}$ ，以及（c）后期交互机制。给定一个查询 $q$ 和文档 $d$ ， ${f}_{Q}$ 将 $q$ 编码为一组固定大小的嵌入 ${E}_{q}$ ，而 ${f}_{D}$ 将 $d$ 编码为另一组 ${E}_{d}$ 。关键的是， ${E}_{q}$ 和 ${E}_{d}$ 中的每个嵌入分别基于 $q$ 或 $d$ 中的其他术语进行上下文处理。我们将在§3.2中描述基于BERT的编码器。

Using ${E}_{q}$ and ${E}_{d}$ ,ColBERT computes the relevance score between $q$ and $d$ via late interaction,which we define as a summation of maximum similarity (MaxSim) operators. In particular, we find the maximum cosine similarity of each $v \in  {E}_{q}$ with vectors in ${E}_{d}$ , and combine the outputs via summation. Besides cosine, we also evaluate squared L2 distance as a measure of vector similarity. Intuitively, this interaction mechanism softly searches for each query term ${t}_{q} -$ in a manner that reflects its context in the query-against the document's embeddings, quantifying the strength of the "match" via the largest similarity score between ${t}_{q}$ and a document term ${t}_{d}$ . Given these term scores, it then estimates the document relevance by summing the matching evidence across all query terms.

使用 ${E}_{q}$ 和 ${E}_{d}$ ，ColBERT通过后期交互计算 $q$ 和 $d$ 之间的相关性得分，我们将其定义为最大相似度（MaxSim）算子的总和。具体来说，我们找到 $v \in  {E}_{q}$ 中的每个向量与 ${t}_{q} -$ 中的向量的最大余弦相似度，并通过求和组合输出。除了余弦相似度，我们还评估平方L2距离作为向量相似度的度量。直观地说，这种交互机制以一种反映查询中查询项上下文的方式，在文档嵌入中软搜索每个查询项 ${t}_{q} -$ ，通过 ${t}_{q}$ 和文档项 ${t}_{d}$ 之间的最大相似度得分来量化“匹配”的强度。给定这些项得分，然后通过对所有查询项的匹配证据求和来估计文档相关性。

While more sophisticated matching is possible with other choices such as deep convolution and attention layers (i.e., as in typical interaction-focused models), a summation of maximum similarity computations has two distinctive characteristics. First, it stands out as a particularly cheap interaction mechanism, as we examine its FLOPs in §4.2. Second, and more importantly, it is amenable to highly-efficient pruning for top- $k$ retrieval,as we evaluate in §4.3. This enables using vector-similarity algorithms for skipping documents without materializing the full interaction matrix or even considering each document in isolation. Other cheap choices (e.g., a summation of average similarity scores, instead of maximum) are possible; however, many are less amenable to pruning. In $\$ {4.4}$ , we conduct an extensive ablation study that empirically verifies the advantage of our MaxSim-based late interaction against alternatives.

虽然使用其他选择（如深度卷积和注意力层，即典型的以交互为重点的模型）可以进行更复杂的匹配，但最大相似度计算的求和有两个显著特点。首先，正如我们在§4.2中研究其浮点运算次数（FLOPs）时所发现的，它是一种特别低成本的交互机制。其次，更重要的是，正如我们在§4.3中评估的那样，它适合用于前 $k$ 检索的高效剪枝。这使得可以使用向量相似度算法跳过文档，而无需具体化完整的交互矩阵，甚至无需单独考虑每个文档。其他低成本的选择（例如，平均相似度得分的求和，而不是最大值）也是可能的；然而，许多选择不太适合剪枝。在 $\$ {4.4}$ 中，我们进行了广泛的消融研究，通过实验验证了我们基于MaxSim的后期交互相对于其他方法的优势。

### 3.2 Query & Document Encoders

### 3.2 查询和文档编码器

Prior to late interaction, ColBERT encodes each query or document into a bag of embeddings, employing BERT-based encoders. We share a single BERT model among our query and document encoders but distinguish input sequences that correspond to queries and documents by prepending a special token $\left\lbrack  Q\right\rbrack$ to queries and another token [D] to documents.

在后期交互之前，ColBERT使用基于BERT的编码器将每个查询或文档编码为一组嵌入。我们在查询和文档编码器之间共享一个BERT模型，但通过在查询前添加一个特殊标记 $\left\lbrack  Q\right\rbrack$ ，在文档前添加另一个标记 [D] 来区分对应于查询和文档的输入序列。

Query Encoder. Given a textual query $q$ ,we tokenize it into its BERT-based WordPiece [35] tokens ${q}_{1}{q}_{2}\ldots {q}_{l}$ . We prepend the token [Q] to the query. We place this token right after BERT's sequence-start token [CLS]. If the query has fewer than a pre-defined number of tokens ${N}_{q}$ ,we pad it with BERT’s special [mask] tokens up to length ${N}_{q}$ (otherwise,we truncate it to the first ${N}_{q}$ tokens). This padded sequence of input tokens is then passed into BERT's deep transformer architecture, which computes a contextualized representation of each token.

查询编码器。给定一个文本查询 $q$，我们将其分词为基于BERT的词块（WordPiece）[35] 标记 ${q}_{1}{q}_{2}\ldots {q}_{l}$。我们在查询前添加标记 [Q]。我们将此标记放在BERT的序列起始标记 [CLS] 之后。如果查询的标记数量少于预定义的数量 ${N}_{q}$，我们用BERT的特殊 [mask] 标记对其进行填充，直至长度达到 ${N}_{q}$（否则，我们将其截断为前 ${N}_{q}$ 个标记）。然后，将这个填充后的输入标记序列输入到BERT的深度变换器（transformer）架构中，该架构会计算每个标记的上下文表示。

We denote the padding with masked tokens as query augmentation, a step that allows BERT to produce query-based embeddings at the positions corresponding to these masks. Query augmentation is intended to serve as a soft, differentiable mechanism for learning to expand queries with new terms or to re-weigh existing terms based on their importance for matching the query. As we show in §4.4, this operation is essential for ColBERT's effectiveness.

我们将使用掩码标记进行填充的操作称为查询增强，这一步骤允许BERT在与这些掩码对应的位置生成基于查询的嵌入表示。查询增强旨在作为一种软的、可微的机制，用于学习用新术语扩展查询，或根据现有术语对匹配查询的重要性重新加权。正如我们在§4.4中所示，此操作对于ColBERT的有效性至关重要。

Given BERT's representation of each token, our encoder passes the contextualized output representations through a linear layer with no activations. This layer serves to control the dimension of ColBERT’s embeddings,producing $m$ -dimensional embeddings for the layer’s output size $m$ . As we discuss later in more detail, we typically fix $m$ to be much smaller than BERT’s fixed hidden dimension.

给定BERT对每个标记的表示，我们的编码器将上下文输出表示通过一个无激活函数的线性层。该层用于控制ColBERT嵌入表示的维度，为该层的输出大小 $m$ 生成 $m$ 维的嵌入表示。正如我们稍后将详细讨论的，我们通常将 $m$ 固定为远小于BERT的固定隐藏维度。

While ColBERT's embedding dimension has limited impact on the efficiency of query encoding, this step is crucial for controlling the space footprint of documents, as we show in §4.5. In addition, it can have a significant impact on query execution time, particularly the time taken for transferring the document representations onto the GPU from system memory (where they reside before processing a query). In fact, as we show in §4.2, gathering, stacking, and transferring the embeddings from CPU to GPU can be the most expensive step in re-ranking with ColBERT. Finally, the output embeddings are normalized so each has L2 norm equal to one. The result is that the dot-product of any two embeddings becomes equivalent to their cosine similarity,falling in the $\left\lbrack  {-1,1}\right\rbrack$ range.

虽然ColBERT的嵌入维度对查询编码效率的影响有限，但正如我们在§4.5中所示，这一步骤对于控制文档的空间占用至关重要。此外，它可能对查询执行时间产生重大影响，特别是将文档表示从系统内存（在处理查询之前它们存储在此处）传输到GPU所需的时间。事实上，正如我们在§4.2中所示，收集、堆叠并将嵌入表示从CPU传输到GPU可能是使用ColBERT进行重排序时最耗时的步骤。最后，对输出嵌入表示进行归一化处理，使其L2范数等于1。这样，任意两个嵌入表示的点积就等同于它们的余弦相似度，其值落在 $\left\lbrack  {-1,1}\right\rbrack$ 范围内。

Document Encoder. Our document encoder has a very similar architecture. We first segment a document $d$ into its constituent tokens ${d}_{1}{d}_{2}\ldots {d}_{m}$ ,to which we prepend BERT’s start token [CLS] followed by our special token [D] that indicates a document sequence. Unlike queries, we do not append [mask] tokens to documents. After passing this input sequence through BERT and the subsequent linear layer, the document encoder filters out the embeddings corresponding to punctuation symbols, determined via a pre-defined list. This filtering is meant to reduce the number of embeddings per document, as we hypothesize that (even contextualized) embeddings of punctuation are unnecessary for effectiveness.

文档编码器。我们的文档编码器具有非常相似的架构。我们首先将文档 $d$ 分割为其组成标记 ${d}_{1}{d}_{2}\ldots {d}_{m}$，并在其前添加BERT的起始标记 [CLS]，然后添加我们的特殊标记 [D] 以表示文档序列。与查询不同，我们不会在文档后添加 [mask] 标记。将此输入序列通过BERT和后续的线性层后，文档编码器会过滤掉与标点符号对应的嵌入表示，这些标点符号是通过预定义列表确定的。此过滤操作旨在减少每个文档的嵌入表示数量，因为我们假设（即使是上下文相关的）标点符号的嵌入表示对于有效性来说是不必要的。

In summary,given $q = {q}_{0}{q}_{1}\ldots {q}_{l}$ and $d = {d}_{0}{d}_{1}\ldots {d}_{n}$ ,we compute the bags of embeddings ${E}_{q}$ and ${E}_{d}$ in the following manner,where #refers to the [mask] tokens:

综上所述，给定 $q = {q}_{0}{q}_{1}\ldots {q}_{l}$ 和 $d = {d}_{0}{d}_{1}\ldots {d}_{n}$，我们按以下方式计算嵌入表示集合 ${E}_{q}$ 和 ${E}_{d}$，其中 # 表示 [mask] 标记：

$$
{E}_{q} \mathrel{\text{:=}} \text{Normalize( CNN( BERT("[Q]}{q}_{0}{q}_{1}\ldots {q}_{l}\text{##...#")))} \tag{1}
$$

$$
{E}_{d} \mathrel{\text{:=}} \text{Filter( Normalize( CNN( BERT("[D]}{d}_{0}{d}_{1}\ldots {d}_{n}\text{"))) ) )} \tag{2}
$$

### 3.3 Late Interaction

### 3.3 后期交互

Given the representation of a query $q$ and a document $d$ ,the relevance score of $d$ to $q$ ,denoted as ${S}_{q,d}$ ,is estimated via late interaction between their bags of contextualized embeddings. As mentioned before, this is conducted as a sum of maximum similarity computations, namely cosine similarity (implemented as dot-products due to the embedding normalization) or squared L2 distance.

给定查询 $q$ 和文档 $d$ 的表示，文档 $d$ 相对于查询 $q$ 的相关性得分（表示为 ${S}_{q,d}$）通过它们的上下文嵌入表示集合之间的后期交互来估计。如前所述，这是通过最大相似度计算的总和来进行的，即余弦相似度（由于嵌入表示已归一化，实现为点积）或平方L2距离。

$$
{S}_{q,d} \mathrel{\text{:=}} \mathop{\sum }\limits_{{i \in  \left\lbrack  \left| {E}_{q}\right| \right\rbrack  }}\mathop{\max }\limits_{{j \in  \left\lbrack  \left| {E}_{d}\right| \right\rbrack  }}{E}_{{q}_{i}} \cdot  {E}_{{d}_{j}}^{T} \tag{3}
$$

ColBERT is differentiable end-to-end. We fine-tune the BERT encoders and train from scratch the additional parameters (i.e., the linear layer and the [Q] and [D] markers' embeddings) using the Adam [16] optimizer. Notice that our interaction mechanism has no trainable parameters. Given a triple $\left\langle  {q,{d}^{ + },{d}^{ - }}\right\rangle$ with query $q$ , positive document ${d}^{ + }$ and negative document ${d}^{ - }$ ,ColBERT is used to produce a score for each document individually and is optimized via pairwise softmax cross-entropy loss over the computed scores of ${d}^{ + }$ and ${d}^{ - }$ .

ColBERT是端到端可微的。我们使用Adam [16] 优化器对BERT编码器进行微调，并从头开始训练额外的参数（即线性层以及 [Q] 和 [D] 标记的嵌入表示）。请注意，我们的交互机制没有可训练的参数。给定一个三元组 $\left\langle  {q,{d}^{ + },{d}^{ - }}\right\rangle$，其中包含查询 $q$、正文档 ${d}^{ + }$ 和负文档 ${d}^{ - }$，ColBERT用于分别为每个文档生成一个得分，并通过对 ${d}^{ + }$ 和 ${d}^{ - }$ 计算出的得分进行成对的softmax交叉熵损失来进行优化。

### 3.4 Offline Indexing: Computing & Storing Document Embeddings

### 3.4 离线索引：计算与存储文档嵌入向量

By design, ColBERT isolates almost all of the computations between queries and documents, largely to enable pre-computing document representations offline. At a high level, our indexing procedure is straight-forward: we proceed over the documents in the collection in batches,running our document encoder ${f}_{D}$ on each batch and storing the output embeddings per document. Although indexing a set of documents is an offline process, we incorporate a few simple optimizations for enhancing the throughput of indexing. As we show in $\$ {4.5}$ ,these optimizations can considerably reduce the offline cost of indexing.

从设计上看，ColBERT几乎将查询与文档之间的所有计算隔离开来，主要是为了能够离线预计算文档表示。从高层次来看，我们的索引过程很直接：我们分批处理集合中的文档，对每一批文档运行我们的文档编码器${f}_{D}$，并存储每个文档的输出嵌入向量。虽然对一组文档进行索引是一个离线过程，但我们采用了一些简单的优化方法来提高索引的吞吐量。正如我们在$\$ {4.5}$中所示，这些优化可以显著降低离线索引的成本。

To begin with, we exploit multiple GPUs, if available, for faster encoding of batches of documents in parallel. When batching, we pad all documents to the maximum length of a document within the batch. ${}^{3}$ To make capping the sequence length on a per-batch basis more effective, our indexer proceeds through documents in groups of $B$ (e.g., $B = {100},{000}$ ) documents. It sorts these documents by length and then feeds batches of $b$ (e.g., $b = {128}$ ) documents of comparable length through our encoder. This length-based bucketing is sometimes refered to as a BucketIterator in some libraries (e.g., allenNLP). Lastly, while most computations occur on the GPU, we found that a non-trivial portion of the indexing time is spent on pre-processing the text sequences, primarily BERT's WordPiece to-kenization. Exploiting that these operations are independent across documents in a batch, we parallelize the pre-processing across the available CPU cores.

首先，如果有多个GPU可用，我们会利用它们并行地更快地对一批批文档进行编码。在分批时，我们将所有文档填充到该批次中文档的最大长度。${}^{3}$为了使按批次限制序列长度更有效，我们的索引器以$B$（例如，$B = {100},{000}$）个文档为一组来处理文档。它按长度对这些文档进行排序，然后将长度相近的$b$（例如，$b = {128}$）个文档一批批地输入到我们的编码器中。这种基于长度的分桶在一些库（例如，allenNLP）中有时被称为BucketIterator。最后，虽然大多数计算在GPU上进行，但我们发现索引时间中有相当一部分花在了文本序列的预处理上，主要是BERT的WordPiece分词。由于这些操作在一批文档中是相互独立的，我们在可用的CPU核心上并行进行预处理。

Once the document representations are produced, they are saved to disk using 32-bit or 16-bit values to represent each dimension. As we describe in $§{3.5}$ and 3.6,these representations are either simply loaded from disk for ranking or are subsequently indexed for vector-similarity search, respectively.

一旦生成了文档表示，就会使用32位或16位值来表示每个维度，并将其保存到磁盘上。正如我们在$§{3.5}$和3.6节中所描述的，这些表示要么直接从磁盘加载用于排序，要么随后进行索引以进行向量相似度搜索。

### 3.5 Top- $k$ Re-ranking with ColBERT

### 3.5 使用ColBERT进行前$k$名重排序

Recall that ColBERT can be used for re-ranking the output of another retrieval model, typically a term-based model, or directly for end-to-end retrieval from a document collection. In this section, we discuss how we use ColBERT for ranking a small set of $k$ (e.g., $k = {1000}$ ) documents given a query $q$ . Since $k$ is small,we rely on batch computations to exhaustively score each document (unlike our approach in §3.6). To begin with, our query serving subsystem loads the indexed documents representations into memory, representing each document as a matrix of embeddings.

请记住，ColBERT可用于对另一个检索模型（通常是基于词项的模型）的输出进行重排序，或者直接用于从文档集合中进行端到端检索。在本节中，我们将讨论如何使用ColBERT对给定查询$q$的一小部分$k$（例如，$k = {1000}$）个文档进行排序。由于$k$数量较少，我们依靠批量计算来对每个文档进行全面评分（与我们在§3.6中的方法不同）。首先，我们的查询服务子系统将索引的文档表示加载到内存中，将每个文档表示为一个嵌入矩阵。

Given a query $q$ ,we compute its bag of contextualized embed-dings ${E}_{q}$ (Equation 1) and,concurrently,gather the document representations into a 3-dimensional tensor $D$ consisting of $k$ document matrices. We pad the $k$ documents to their maximum length to facilitate batched operations,and move the tensor $D$ to the GPU’s memory. On the GPU,we compute a batch dot-product of ${E}_{q}$ and $D$ ,possibly over multiple mini-batches. The output materializes a 3-dimensional tensor that is a collection of cross-match matrices between $q$ and each document. To compute the score of each document, we reduce its matrix across document terms via a max-pool (i.e., representing an exhaustive implementation of our MaxSim computation) and reduce across query terms via a summation. Finally,we sort the $k$ documents by their total scores.

给定一个查询$q$，我们计算其上下文嵌入向量包${E}_{q}$（公式1），同时将文档表示收集到一个由$k$个文档矩阵组成的三维张量$D$中。我们将$k$个文档填充到它们的最大长度以方便批量操作，并将张量$D$移动到GPU内存中。在GPU上，我们计算${E}_{q}$和$D$的批量点积，可能会分多个小批量进行。输出结果是一个三维张量，它是查询$q$与每个文档之间的交叉匹配矩阵的集合。为了计算每个文档的得分，我们通过最大池化（即，代表我们的MaxSim计算的全面实现）对文档词项的矩阵进行降维，并通过求和对查询词项进行降维。最后，我们根据总得分对$k$个文档进行排序。

Relative to existing neural rankers (especially, but not exclusively, BERT-based ones), this computation is very cheap that, in fact, its cost is dominated by the cost of gathering and transferring the pre-computed embeddings. To illustrate,ranking $k$ documents via typical BERT rankers requires feeding BERT $k$ different inputs each of length $l = \left| q\right|  + \left| {d}_{i}\right|$ for query $q$ and documents ${d}_{i}$ ,where attention has quadratic cost in the length of the sequence. In contrast, ColBERT feeds BERT only a single, much shorter sequence of length $l = \left| q\right|$ . Consequently,ColBERT is not only cheaper,it also scales much better with $k$ as we examine in $§{4.2}$ .

相对于现有的神经排序器（特别是但不限于基于BERT的排序器），这种计算成本非常低，实际上，其成本主要由收集和传输预计算的嵌入向量的成本决定。例如，通过典型的BERT排序器对$k$个文档进行排序，需要向BERT输入$k$个不同的输入，每个输入的长度为$l = \left| q\right|  + \left| {d}_{i}\right|$，用于查询$q$和文档${d}_{i}$，其中注意力机制在序列长度上的成本是二次的。相比之下，ColBERT只向BERT输入一个长度短得多的单一序列$l = \left| q\right|$。因此，ColBERT不仅成本更低，而且正如我们在$§{4.2}$中所研究的，它在处理$k$个文档时的扩展性也更好。

### 3.6 End-to-end Top- $k$ Retrieval with ColBERT

### 3.6 使用ColBERT进行端到端前$k$名检索

As mentioned before, ColBERT's late-interaction operator is specifically designed to enable end-to-end retrieval from a large collection, largely to improve recall relative to term-based retrieval approaches. This section is concerned with cases where the number of documents to be ranked is too large for exhaustive evaluation of each possible candidate document, particularly when we are only interested in the highest scoring ones. Concretely, we focus here on retrieving the top- $k$ results directly from a large document collection with $N$ (e.g., $N = {10},{000},{000}$ ) documents,where $k \ll  N$ .

如前所述，ColBERT（上下文感知双向编码器表示法）的后期交互算子是专门为实现从大型集合中进行端到端检索而设计的，主要是为了相对于基于词项的检索方法提高召回率。本节关注的是待排序文档数量过多，以至于无法对每个可能的候选文档进行详尽评估的情况，特别是当我们只对得分最高的文档感兴趣时。具体来说，我们这里关注的是直接从包含 $N$（例如 $N = {10},{000},{000}$）个文档的大型文档集合中检索前 $k$ 个结果，其中 $k \ll  N$。

To do so, we leverage the pruning-friendly nature of the MaxSim operations at the backbone of late interaction. Instead of applying MaxSim between one of the query embeddings and all of one document's embeddings, we can use fast vector-similarity data structures to efficiently conduct this search between the query embedding and all document embeddings across the full collection. For this, we employ an off-the-shelf library for large-scale vector-similarity search,namely faiss [15] from Facebook. ${}^{4}$ In particular, at the end of offline indexing (§3.4), we maintain a mapping from each embedding to its document of origin and then index all document embeddings into faiss.

为此，我们利用后期交互核心的最大相似度（MaxSim）操作对剪枝友好的特性。我们可以使用快速向量相似度数据结构，在查询嵌入和整个集合中的所有文档嵌入之间高效地进行搜索，而不是在一个查询嵌入和一个文档的所有嵌入之间应用最大相似度操作。为此，我们采用了一个现成的大规模向量相似度搜索库，即 Facebook 的 Faiss [15]。${}^{4}$ 特别是在离线索引（§3.4）结束时，我们维护每个嵌入到其原始文档的映射，然后将所有文档嵌入索引到 Faiss 中。

Subsequently, when serving queries, we use a two-stage procedure to retrieve the top- $k$ documents from the entire collection. Both stages rely on ColBERT's scoring: the first is an approximate stage aimed at filtering while the second is a refinement stage. For the first stage,we concurrently issue ${N}_{q}$ vector-similarity queries (corresponding to each of the embeddings in ${E}_{q}$ ) onto our faiss index. This retrieves the top- ${k}^{\prime }$ (e.g., ${k}^{\prime } = k/2$ ) matches for that vector over all document embeddings. We map each of those to its document of origin,producing ${N}_{q} \times  {k}^{\prime }$ document IDs,only $K \leq  {N}_{q} \times  {k}^{\prime }$ of which are unique. These $K$ documents likely contain one or more embeddings that are highly similar to the query embeddings. For the second stage, we refine this set by exhaustively re-ranking only those $K$ documents in the usual manner described in $§{3.5}$ .

随后，在处理查询时，我们使用两阶段过程从整个集合中检索前 $k$ 个文档。两个阶段都依赖于 ColBERT 的评分：第一阶段是旨在过滤的近似阶段，而第二阶段是细化阶段。在第一阶段，我们同时向 Faiss 索引发出 ${N}_{q}$ 个向量相似度查询（对应于 ${E}_{q}$ 中的每个嵌入）。这会在所有文档嵌入中为该向量检索前 ${k}^{\prime }$（例如 ${k}^{\prime } = k/2$）个匹配项。我们将每个匹配项映射到其原始文档，生成 ${N}_{q} \times  {k}^{\prime }$ 个文档 ID，其中只有 $K \leq  {N}_{q} \times  {k}^{\prime }$ 个是唯一的。这 $K$ 个文档可能包含一个或多个与查询嵌入高度相似的嵌入。在第二阶段，我们按照 $§{3.5}$ 中描述的常规方式对这 $K$ 个文档进行详尽的重新排序，以细化这个集合。

---

<!-- Footnote -->

${}^{3}$ The public BERT implementations we saw simply pad to a pre-defined length.

${}^{3}$ 我们看到的公开 BERT（双向编码器表示法）实现只是填充到预定义的长度。

${}^{4}$ https://github.com/facebookresearch/faiss

${}^{4}$ https://github.com/facebookresearch/faiss

<!-- Footnote -->

---

In our faiss-based implementation, we use an IVFPQ index ("inverted file with product quantization"). This index partitions the embedding space into $P$ (e.g., $P = {1000}$ ) cells based on $k$ -means clustering and then assigns each document embedding to its nearest cell based on the selected vector-similarity metric. For serving queries, when searching for the top- ${k}^{\prime }$ matches for a single query embedding,only the nearest $p$ (e.g., $p = {10}$ ) partitions are searched. To improve memory efficiency,every embedding is divided into $s$ (e.g., $s = {16}$ ) sub-vectors,each represented using one byte. Moreover, the index conducts the similarity computations in this compressed domain, leading to cheaper computations and thus faster search.

在我们基于 Faiss 的实现中，我们使用 IVFPQ 索引（“带乘积量化的倒排文件”）。该索引基于 $k$ -均值聚类将嵌入空间划分为 $P$（例如 $P = {1000}$）个单元，然后根据所选的向量相似度度量将每个文档嵌入分配到其最近的单元。在处理查询时，当为单个查询嵌入搜索前 ${k}^{\prime }$ 个匹配项时，只搜索最近的 $p$（例如 $p = {10}$）个分区。为了提高内存效率，每个嵌入被划分为 $s$（例如 $s = {16}$）个子向量，每个子向量用一个字节表示。此外，该索引在这个压缩域中进行相似度计算，从而降低计算成本，加快搜索速度。

## 4 EXPERIMENTAL EVALUATION

## 4 实验评估

We now turn our attention to empirically testing ColBERT, addressing the following research questions.

现在我们将注意力转向对 ColBERT 进行实证测试，解决以下研究问题。

${\mathbf{{RQ}}}_{1}$ : In a typical re-ranking setup,how well can ColBERT bridge the existing gap (highlighted in $\$ 1$ ) between highly-efficient and highly-effective neural models? (§4.2)

${\mathbf{{RQ}}}_{1}$：在典型的重排序设置中，ColBERT 能在多大程度上弥合 $\$ 1$ 中强调的高效神经模型和高效能神经模型之间的现有差距？（§4.2）

${\mathbf{{RQ}}}_{2}$ : Beyond re-ranking,can ColBERT effectively support end-to-end retrieval directly from a large collection? (§4.3)

${\mathbf{{RQ}}}_{2}$：除了重排序之外，ColBERT 能否有效地支持直接从大型集合中进行端到端检索？（§4.3）

${\mathbf{{RQ}}}_{3}$ : What does each component of ColBERT (e.g.,late interaction, query augmentation) contribute to its quality? (§4.4)

${\mathbf{{RQ}}}_{3}$：ColBERT 的每个组件（例如后期交互、查询增强）对其性能有什么贡献？（§4.4）

${\mathbf{{RQ}}}_{4}$ : What are ColBERT’s indexing-related costs in terms of offline computation and memory overhead? (§4.5)

${\mathbf{{RQ}}}_{4}$：就离线计算和内存开销而言，ColBERT 的索引相关成本是多少？（§4.5）

### 4.1 Methodology

### 4.1 方法

4.1.1 Datasets & Metrics. Similar to related work $\left\lbrack  {2,{27},{28}}\right\rbrack$ , we conduct our experiments on the MS MARCO Ranking [24] (henceforth, MS MARCO) and TREC Complex Answer Retrieval (TREC-CAR) [6] datasets. Both of these recent datasets provide large training data of the scale that facilitates training and evaluating deep neural networks. We describe both in detail below.

4.1.1 数据集与指标。与相关工作 $\left\lbrack  {2,{27},{28}}\right\rbrack$ 类似，我们在 MS MARCO 排序数据集 [24]（以下简称 MS MARCO）和 TREC 复杂答案检索数据集（TREC - CAR）[6] 上进行实验。这两个近期的数据集都提供了大规模的训练数据，便于训练和评估深度神经网络。我们将在下面详细介绍这两个数据集。

MS MARCO. MS MARCO is a dataset (and a corresponding competition) introduced by Microsoft in 2016 for reading comprehension and adapted in 2018 for retrieval. It is a collection of ${8.8}\mathrm{M}$ passages from Web pages, which were gathered from Bing's results to $1\mathrm{M}$ real-world queries. Each query is associated with sparse relevance judgements of one (or very few) documents marked as relevant and no documents explicitly indicated as irrelevant. Per the official evaluation, we use MRR@10 to measure effectiveness.

MS MARCO。MS MARCO 是微软在 2016 年为阅读理解引入的一个数据集（以及相应的竞赛），并在 2018 年进行了检索方面的调整。它是从网页中收集的 ${8.8}\mathrm{M}$ 个段落的集合，这些段落是从必应（Bing）对 $1\mathrm{M}$ 个现实世界查询的结果中收集而来的。每个查询都与稀疏的相关性判断相关联，其中一个（或极少数）文档被标记为相关，且没有文档被明确标记为不相关。根据官方评估，我们使用 MRR@10 来衡量有效性。

We use three sets of queries for evaluation. The official development and evaluation sets contain roughly $7\mathrm{k}$ queries. However, the relevance judgements of the evaluation set are held-out by Mi-crosoft and effectiveness results can only be obtained by submitting to the competition's organizers. We submitted our main re-ranking ColBERT model for the results in §4.2. In addition, the collection includes roughly ${55}\mathrm{k}$ queries (with labels) that are provided as additional validation data. We re-purpose a random sample of $5\mathrm{\;k}$ queries among those (i.e., ones not in our development or training sets) as a "local" evaluation set. Along with the official development set, we use this held-out set for testing our models as well as baselines in §4.3. We do so to avoid submitting multiple variants of the same model at once, as the organizers discourage too many submissions by the same team.

我们使用三组查询进行评估。官方的开发集和评估集大约包含 $7\mathrm{k}$ 个查询。然而，评估集的相关性判断由微软保密，有效性结果只能通过提交给竞赛组织者来获得。我们提交了主要的重排序 ColBERT 模型以获取§4.2 中的结果。此外，该集合还包含大约 ${55}\mathrm{k}$ 个（带标签的）查询，这些查询作为额外的验证数据提供。我们将其中 $5\mathrm{\;k}$ 个查询的随机样本（即不在我们的开发集或训练集中的查询）重新用作“本地”评估集。除了官方开发集之外，我们还使用这个保留集在§4.3 中测试我们的模型以及基线模型。我们这样做是为了避免一次性提交同一模型的多个变体，因为组织者不鼓励同一团队提交过多的结果。

TREC CAR. Introduced by Dietz [6] et al. in 2017, TREC CAR is a synthetic dataset based on Wikipedia that consists of about ${29}\mathrm{M}$ passages. Similar to related work [25],we use the first four of five pre-defined folds for training and the fifth for validation. This amounts to roughly $3\mathrm{M}$ queries generated by concatenating the title of a Wikipedia page with the heading of one of its sections. That section's passages are marked as relevant to the corresponding query. Our evaluation is conducted on the test set used in TREC 2017 CAR, which contains 2,254 queries.

TREC CAR。TREC CAR 由迪茨（Dietz）[6] 等人在 2017 年引入，是一个基于维基百科的合成数据集，包含大约 ${29}\mathrm{M}$ 个段落。与相关工作 [25] 类似，我们使用五个预定义折中的前四个进行训练，第五个用于验证。这大约相当于通过将维基百科页面的标题与其中一个部分的标题连接起来生成的 $3\mathrm{M}$ 个查询。该部分的段落被标记为与相应查询相关。我们的评估是在 2017 年 TREC CAR 测试集上进行的，该测试集包含 2254 个查询。

4.1.2 Implementation. Our ColBERT models are implemented using Python 3 and PyTorch 1. We use the popular transformers ${}^{5}$ library for the pre-trained BERT model. Similar to [25], we fine-tune all ColBERT models with learning rate $3 \times  {10}^{-6}$ with a batch size 32. We fix the number of embeddings per query at ${N}_{q} = {32}$ . We set our ColBERT embedding dimension $m$ to be 128; $§{4.5}$ demonstrates ColBERT's robustness to a wide range of embedding dimensions.

4.1.2 实现。我们的 ColBERT 模型使用 Python 3 和 PyTorch 1 实现。我们使用流行的 transformers ${}^{5}$ 库来使用预训练的 BERT 模型。与 [25] 类似，我们以学习率 $3 \times  {10}^{-6}$ 和批量大小 32 对所有 ColBERT 模型进行微调。我们将每个查询的嵌入数量固定为 ${N}_{q} = {32}$。我们将 ColBERT 嵌入维度 $m$ 设置为 128；$§{4.5}$ 证明了 ColBERT 对广泛的嵌入维度具有鲁棒性。

For MS MARCO, we initialize the BERT components of the ColBERT query and document encoders using Google's official pre-trained ${\mathrm{{BERT}}}_{\text{base }}$ model. Further,we train all models for ${200}\mathrm{k}$ iterations. For TREC CAR, we follow related work [2, 25] and use a different pre-trained model to the official ones. To explain, the official BERT models were pre-trained on Wikipedia, which is the source of TREC CAR's training and test sets. To avoid leaking test data into train, Nogueira and Cho's [25] pre-train a randomly-initialized BERT model on the Wiki pages corresponding to training subset of TREC CAR. They release their BERTlarge pre-trained model, which we fine-tune for ColBERT's experiments on TREC CAR. Since fine-tuning this model is significantly slower than ${\mathrm{{BERT}}}_{\text{base }}$ ,we train on TREC CAR for only ${125}\mathrm{k}$ iterations.

对于 MS MARCO，我们使用谷歌官方的预训练 ${\mathrm{{BERT}}}_{\text{base }}$ 模型初始化 ColBERT 查询和文档编码器的 BERT 组件。此外，我们对所有模型进行 ${200}\mathrm{k}$ 次迭代训练。对于 TREC CAR，我们遵循相关工作 [2, 25]，并使用与官方不同的预训练模型。具体来说，官方的 BERT 模型是在维基百科上进行预训练的，而维基百科是 TREC CAR 训练集和测试集的来源。为了避免将测试数据泄露到训练中，诺盖拉（Nogueira）和赵（Cho）[25] 在与 TREC CAR 训练子集对应的维基页面上对一个随机初始化的 BERT 模型进行预训练。他们发布了他们的 BERTlarge 预训练模型，我们在 TREC CAR 上对 ColBERT 实验进行微调。由于微调这个模型比 ${\mathrm{{BERT}}}_{\text{base }}$ 慢得多，我们在 TREC CAR 上只进行 ${125}\mathrm{k}$ 次迭代训练。

In our re-ranking results, unless stated otherwise, we use 4 bytes per dimension in our embeddings and employ cosine as our vector-similarity function. For end-to-end ranking, we use (squared) L2 distance, as we found our faiss index was faster at L2-based retrieval. For our faiss index, we set the number of partitions to $P = 2,{000}$ ,and search the nearest $p = {10}$ to each query embedding to retrieve ${k}^{\prime } = k = {1000}$ document vectors per query embedding. We divide each embedding into $s = {16}$ sub-vectors,each encoded using one byte. To represent the index used for the second stage of our end-to-end retrieval procedure, we use 16-bit values per dimension.

在我们的重排序结果中，除非另有说明，我们在嵌入向量中每个维度使用4字节，并采用余弦作为向量相似度函数。对于端到端排序，我们使用（平方）L2距离，因为我们发现基于L2的检索在我们的faiss索引中速度更快。对于我们的faiss索引，我们将分区数设置为$P = 2,{000}$，并为每个查询嵌入搜索最近的$p = {10}$个向量，以每个查询嵌入检索${k}^{\prime } = k = {1000}$个文档向量。我们将每个嵌入向量划分为$s = {16}$个子向量，每个子向量使用一个字节进行编码。为了表示端到端检索过程第二阶段使用的索引，我们每个维度使用16位值。

4.1.3 Hardware & Time Measurements. To evaluate the latency of neural re-ranking models in §4.2, we use a single Tesla V100 GPU that has 32 GiBs of memory on a server with two Intel Xeon Gold 6132 CPUs, each with 14 physical cores (24 hyperthreads), and 469 GiBs of RAM. For the mostly CPU-based retrieval experiments in $§{4.3}$ and the indexing experiments in $§{4.5}$ ,we use another server with the same CPU and system memory specifications but which has four Titan V GPUs attached, each with 12 GiBs of memory. Across all experiments, only one GPU is dedicated per query for retrieval (i.e., for methods with neural computations) but we use up to all four GPUs during indexing.

4.1.3 硬件与时间测量。为了评估§4.2中神经重排序模型的延迟，我们在一台服务器上使用了一块具有32GB内存的Tesla V100 GPU，该服务器配备了两颗Intel Xeon Gold 6132 CPU，每颗CPU有14个物理核心（24个超线程），以及469GB的RAM。对于$§{4.3}$中主要基于CPU的检索实验和$§{4.5}$中的索引实验，我们使用了另一台服务器，其CPU和系统内存规格相同，但连接了四块Titan V GPU，每块GPU有12GB内存。在所有实验中，每次查询仅使用一块GPU进行检索（即用于涉及神经计算的方法），但在索引过程中我们最多会使用全部四块GPU。

---

<!-- Footnote -->

${}^{5}$ https://github.com/huggingface/transformers

${}^{5}$ https://github.com/huggingface/transformers

<!-- Footnote -->

---

<!-- Media -->

<table><tr><td>Method</td><td>MRR@10 (Dev)</td><td>MRR@10 (Eval)</td><td>Re-ranking Latency (ms)</td><td>FLOPs/query</td><td/></tr><tr><td>BM25 (official)</td><td>16.7</td><td>16.5</td><td>-</td><td>-</td><td/></tr><tr><td>KNRM</td><td>19.8</td><td>19.8</td><td>3</td><td>592M (0.085x)</td><td/></tr><tr><td>Duet</td><td>24.3</td><td>24.5</td><td>22</td><td>159B (23x)</td><td/></tr><tr><td>fastText+ConvKNRM</td><td>29.0</td><td>27.7</td><td>28</td><td>${78}\mathrm{\;B}\left( {{11} \times  }\right)$</td><td/></tr><tr><td>${\mathrm{{BERT}}}_{\text{base }}\left\lbrack  {25}\right\rbrack$</td><td>34.7</td><td>-</td><td>10,700</td><td>97T (13,900X)</td><td/></tr><tr><td>${\mathrm{{BERT}}}_{\text{base }}$ (our training)</td><td>36.0</td><td>-</td><td>10,700</td><td>97T (13,900X)</td><td/></tr><tr><td>${\mathrm{{BERT}}}_{\text{large }}\left\lbrack  {25}\right\rbrack$</td><td>36.5</td><td>35.9</td><td>32,900</td><td>340T (48,600X)</td><td/></tr><tr><td>ColBERT (over BERT ${}_{\text{base}}$ )</td><td>34.9</td><td>34.9</td><td>61</td><td>7B (1x)</td><td/></tr></table>

<table><tbody><tr><td>方法</td><td>前10名平均倒数排名（开发集）</td><td>前10名平均倒数排名（评估集）</td><td>重排序延迟（毫秒）</td><td>每次查询的浮点运算次数</td><td></td></tr><tr><td>二元独立模型（官方）</td><td>16.7</td><td>16.5</td><td>-</td><td>-</td><td></td></tr><tr><td>核化非线性匹配模型（KNRM）</td><td>19.8</td><td>19.8</td><td>3</td><td>5.92亿（0.085倍）</td><td></td></tr><tr><td>二重奏模型（Duet）</td><td>24.3</td><td>24.5</td><td>22</td><td>1590亿（23倍）</td><td></td></tr><tr><td>快速文本+卷积核化非线性匹配模型（fastText+ConvKNRM）</td><td>29.0</td><td>27.7</td><td>28</td><td>${78}\mathrm{\;B}\left( {{11} \times  }\right)$</td><td></td></tr><tr><td>${\mathrm{{BERT}}}_{\text{base }}\left\lbrack  {25}\right\rbrack$</td><td>34.7</td><td>-</td><td>10,700</td><td>97万亿（13900倍）</td><td></td></tr><tr><td>[公式0]（我们的训练）</td><td>36.0</td><td>-</td><td>10,700</td><td>97万亿（13900倍）</td><td></td></tr><tr><td>${\mathrm{{BERT}}}_{\text{large }}\left\lbrack  {25}\right\rbrack$</td><td>36.5</td><td>35.9</td><td>32,900</td><td>340万亿（48600倍）</td><td></td></tr><tr><td>基于BERT [公式0]的ColBERT模型</td><td>34.9</td><td>34.9</td><td>61</td><td>70亿（1倍）</td><td></td></tr></tbody></table>

Table 1: "Re-ranking" results on MS MARCO. Each neural model re-ranks the official top-1000 results produced by BM25. Latency is reported for re-ranking only. To obtain the end-to-end latency in Figure 1, we add the BM25 latency from Table 2.

表1：MS MARCO上的“重排序”结果。每个神经模型对BM25生成的官方前1000个结果进行重排序。仅报告重排序的延迟。要获得图1中的端到端延迟，我们需加上表2中的BM25延迟。

<table><tr><td>Method</td><td>MRR@10 (Dev)</td><td>MRR@10 (Local Eval)</td><td>Latency (ms)</td><td>Recall@50</td><td>Recall@200</td><td>Recall@1000</td><td/></tr><tr><td>BM25 (official)</td><td>16.7</td><td>-</td><td>-</td><td>-</td><td>-</td><td>81.4</td><td/></tr><tr><td>BM25 (Anserini)</td><td>18.7</td><td>19.5</td><td>62</td><td>59.2</td><td>73.8</td><td>85.7</td><td/></tr><tr><td>doc2query</td><td>21.5</td><td>22.8</td><td>85</td><td>64.4</td><td>77.9</td><td>89.1</td><td/></tr><tr><td>DeepCT</td><td>24.3</td><td>-</td><td>62 (est.)</td><td>69 [2]</td><td>82 [2]</td><td>91 [2]</td><td/></tr><tr><td>docTTTTTquery</td><td>27.7</td><td>28.4</td><td>87</td><td>75.6</td><td>86.9</td><td>94.7</td><td/></tr><tr><td>ColBERTL2 (re-rank)</td><td>34.8</td><td>36.4</td><td>-</td><td>75.3</td><td>80.5</td><td>81.4</td><td/></tr><tr><td>ColBERTL2 (end-to-end)</td><td>36.0</td><td>36.7</td><td>458</td><td>82.9</td><td>92.3</td><td>96.8</td><td/></tr></table>

<table><tbody><tr><td>方法</td><td>前10名平均倒数排名（开发集）(MRR@10 (Dev))</td><td>前10名平均倒数排名（本地评估）(MRR@10 (Local Eval))</td><td>延迟（毫秒）(Latency (ms))</td><td>前50名召回率(Recall@50)</td><td>前200名召回率(Recall@200)</td><td>前1000名召回率(Recall@1000)</td><td></td></tr><tr><td>BM25（官方）</td><td>16.7</td><td>-</td><td>-</td><td>-</td><td>-</td><td>81.4</td><td></td></tr><tr><td>BM25（Anserini）</td><td>18.7</td><td>19.5</td><td>62</td><td>59.2</td><td>73.8</td><td>85.7</td><td></td></tr><tr><td>文档到查询(doc2query)</td><td>21.5</td><td>22.8</td><td>85</td><td>64.4</td><td>77.9</td><td>89.1</td><td></td></tr><tr><td>深度上下文词项加权(DeepCT)</td><td>24.3</td><td>-</td><td>62（估计值）</td><td>69 [2]</td><td>82 [2]</td><td>91 [2]</td><td></td></tr><tr><td>文档到多个查询(docTTTTTquery)</td><td>27.7</td><td>28.4</td><td>87</td><td>75.6</td><td>86.9</td><td>94.7</td><td></td></tr><tr><td>ColBERTL2（重排序）</td><td>34.8</td><td>36.4</td><td>-</td><td>75.3</td><td>80.5</td><td>81.4</td><td></td></tr><tr><td>ColBERTL2（端到端）</td><td>36.0</td><td>36.7</td><td>458</td><td>82.9</td><td>92.3</td><td>96.8</td><td></td></tr></tbody></table>

Table 2: End-to-end retrieval results on MS MARCO. Each model retrieves the top-1000 documents per query directly from the entire 8.8M document collection.

表2：MS MARCO上的端到端检索结果。每个模型直接从整个880万文档集合中为每个查询检索前1000个文档。

<!-- Media -->

### 4.2 Quality-Cost Tradeoff: Top- $k$ Re-ranking

### 4.2 质量 - 成本权衡：前 $k$ 重排序

In this section, we examine ColBERT's efficiency and effectiveness at re-ranking the top- $k$ results extracted by a bag-of-words retrieval model, which is the most typical setting for testing and deploying neural ranking models. We begin with the MS MARCO dataset. We compare against KNRM, Duet, and fastText+ConvKNRM, a representative set of neural matching models that have been previously tested on MS MARCO. In addition, we compare against the natural adaptation of BERT for ranking by Nogueira and Cho [25], in particular, ${\mathrm{{BERT}}}_{\text{base }}$ and its deeper counterpart ${\mathrm{{BERT}}}_{\text{large }}$ . We also report results for "BERT ${}_{\text{base }}$ (our training)",which is based on Nogueira and Cho's base model (including hyperparameters) but is trained with the same loss function as ColBERT (§3.3) for 200k iterations, allowing for a more direct comparison of the results.

在本节中，我们研究ColBERT在对词袋检索模型提取的前 $k$ 结果进行重排序时的效率和有效性，这是测试和部署神经排序模型最典型的设置。我们从MS MARCO数据集开始。我们将其与KNRM、Duet和fastText + ConvKNRM进行比较，这是一组有代表性的神经匹配模型，之前已在MS MARCO上进行过测试。此外，我们还与Nogueira和Cho [25] 提出的用于排序的BERT自然适配模型进行比较，特别是 ${\mathrm{{BERT}}}_{\text{base }}$ 及其更深层的对应模型 ${\mathrm{{BERT}}}_{\text{large }}$。我们还报告了“BERT ${}_{\text{base }}$（我们的训练）”的结果，该模型基于Nogueira和Cho的基础模型（包括超参数），但使用与ColBERT相同的损失函数（§3.3）进行了20万次迭代训练，以便更直接地比较结果。

We report the competition's official metric, namely MRR@10, on the validation set (Dev) and the evaluation set (Eval). We also report the re-ranking latency, which we measure using a single Tesla V100 GPU, and the FLOPs per query for each neural ranking model. For ColBERT, our reported latency subsumes the entire computation from gathering the document representations, moving them to the GPU, tokenizing then encoding the query, and applying late interaction to compute document scores. For the baselines, we measure the scoring computations on the GPU and exclude the CPU-based text preprocessing (similar to [9]). In principle, the baselines can pre-compute the majority of this preprocessing (e.g., document tokenization) offline and parallelize the rest across documents online, leaving only a negligible cost. We estimate the FLOPs per query of each model using the torchprofile ${}^{6}$ library.

我们报告了竞赛的官方指标，即验证集（Dev）和评估集（Eval）上的MRR@10。我们还报告了重排序延迟，我们使用单个Tesla V100 GPU进行测量，以及每个神经排序模型每个查询的浮点运算次数（FLOPs）。对于ColBERT，我们报告的延迟包括从收集文档表示、将它们移动到GPU、对查询进行分词和编码，以及应用后期交互来计算文档得分的整个计算过程。对于基线模型，我们在GPU上测量评分计算，并排除基于CPU的文本预处理（与 [9] 类似）。原则上，基线模型可以离线预计算大部分预处理（例如，文档分词），并在线跨文档并行处理其余部分，只留下可忽略不计的成本。我们使用torchprofile ${}^{6}$ 库估计每个模型每个查询的FLOPs。

We now proceed to study the results, which are reported in Table 1. To begin with, we notice the fast progress from KNRM in 2017 to the BERT-based models in 2019, manifesting itself in over 16% increase in MRR@10.As described in §1, the simultaneous increase in computational cost is difficult to miss. Judging by their rather monotonic pattern of increasingly larger cost and higher effectiveness, these results appear to paint a picture where expensive models are necessary for high-quality ranking.

现在我们来研究表1中报告的结果。首先，我们注意到从2017年的KNRM到2019年基于BERT的模型取得了快速进展，表现为MRR@10提高了超过16%。如§1所述，计算成本的同步增加也很明显。从它们成本越来越高且效果越来越好的相当单调的模式来看，这些结果似乎描绘了一幅高质量排序需要昂贵模型的画面。

In contrast with this trend, ColBERT (which employs late interaction over ${\mathrm{{BERT}}}_{\text{base }}$ ) performs no worse than the original adaptation of ${\mathrm{{BERT}}}_{\text{base }}$ for ranking by Nogueira and Cho [25,27] and is only marginally less effective than ${\mathrm{{BERT}}}_{\text{large }}$ and our training of ${\mathrm{{BERT}}}_{\text{base }}$ (described above). While highly competitive in effectiveness, ColBERT is orders of magnitude cheaper than BERT base, in particular,by over ${170} \times$ in latency and ${13},{900} \times$ in FLOPs. This highlights the expressiveness of our proposed late interaction mechanism, particularly when coupled with a powerful pre-trained LM like BERT. While ColBERT's re-ranking latency is slightly higher than the non-BERT re-ranking models shown (i.e., by 10s of milliseconds), this difference is explained by the time it takes to gather, stack, and transfer the document embeddings to the GPU. In particular, the query encoding and interaction in ColBERT consume only 13 milliseconds of its total execution time. We note that ColBERT's latency and FLOPs can be considerably reduced by padding queries to a shorter length, using smaller vector dimensions (the MRR@10 of which is tested in $\$ {4.5}$ ),employing quantization of the document vectors, and storing the embeddings on GPU if sufficient memory exists. We leave these directions for future work.

与这一趋势相反，ColBERT（在 ${\mathrm{{BERT}}}_{\text{base }}$ 上采用后期交互）的性能不比Nogueira和Cho [25,27] 最初用于排序的 ${\mathrm{{BERT}}}_{\text{base }}$ 适配模型差，并且仅比 ${\mathrm{{BERT}}}_{\text{large }}$ 和我们训练的 ${\mathrm{{BERT}}}_{\text{base }}$（如上所述）略逊一筹。虽然在有效性方面极具竞争力，但ColBERT的成本比BERT基础模型低几个数量级，特别是在延迟方面降低了超过 ${170} \times$，在FLOPs方面降低了 ${13},{900} \times$。这凸显了我们提出的后期交互机制的表达能力，特别是当它与像BERT这样强大的预训练语言模型结合使用时。虽然ColBERT的重排序延迟比所示的非BERT重排序模型略高（即高几十毫秒），但这种差异可以用收集、堆叠和将文档嵌入转移到GPU所需的时间来解释。特别是，ColBERT中的查询编码和交互仅消耗其总执行时间的13毫秒。我们注意到，通过将查询填充到更短的长度、使用更小的向量维度（其MRR@10在 $\$ {4.5}$ 中进行了测试）、对文档向量进行量化，以及在有足够内存的情况下将嵌入存储在GPU上，可以显著降低ColBERT的延迟和FLOPs。我们将这些方向留作未来的工作。

---

<!-- Footnote -->

${}^{6}$ https://github.com/mit-han-lab/torchprofile

${}^{6}$ https://github.com/mit-han-lab/torchprofile

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: Million FLOPs (log-scale) ${10}^{9}$ BERT。 (our training) 2000 1000 200 0.31 0.33 0.35 0.37 MRR@10 ${10}^{8}$ Colbert ${10}^{7}$ ${10}^{6}$ ${10}^{5}$ ${10}^{4}$ ${10}^{3}$ 0.27 0.29 -->

<img src="https://cdn.noedgeai.com/0195afb9-1285-7cff-a7ae-bc74a3288ac4_7.jpg?x=164&y=352&w=692&h=379&r=0"/>

Figure 4: FLOPs (in millions) and MRR@10 as functions of the re-ranking depth $k$ . Since the official BM25 ranking is not ordered,the initial top- $k$ retrieval is conducted with Anserini's BM25.

图4：浮点运算次数（以百万计）和MRR@10作为重排序深度 $k$ 的函数。由于官方的BM25排序未排序，最初的前 $k$ 检索使用Anserini的BM25进行。

<!-- Media -->

Diving deeper into the quality-cost tradeoff between BERT and ColBERT, Figure 4 demonstrates the relationships between FLOPs and effectiveness (MRR@10) as a function of the re-ranking depth $k$ when re-ranking the top- $k$ results by BM25,comparing ColBERT and ${\mathrm{{BERT}}}_{\text{base }}$ (our training). We conduct this experiment on MS MARCO (Dev). We note here that as the official top-1000 ranking does not provide the BM25 order (and also lacks documents beyond the top-1000 per query), the models in this experiment re-rank the Anserini [37] toolkit's BM25 output. Consequently, both MRR@10 values at $k = {1000}$ are slightly higher from those reported in Table 1.

深入探究BERT和ColBERT之间的质量 - 成本权衡问题，图4展示了在使用BM25对前$k$个结果进行重排序时，浮点运算次数（FLOPs）与有效性（MRR@10）之间的关系，该关系是重排序深度$k$的函数，并对ColBERT和${\mathrm{{BERT}}}_{\text{base }}$（我们的训练模型）进行了比较。我们在MS MARCO（开发集）上进行了此实验。这里需要注意的是，由于官方的前1000名排名未提供BM25排序顺序（并且每个查询的前1000名之外的文档也缺失），因此本实验中的模型对Anserini [37]工具包的BM25输出进行重排序。因此，在$k = {1000}$处的两个MRR@10值均略高于表1中报告的值。

Studying the results in Figure 4, we notice that not only is ColBERT much cheaper than BERT for the same model size (i.e., 12- layer "base" transformer encoder), it also scales better with the number of ranked documents. In part, this is because ColBERT only needs to process the query once, irrespective of the number of documents evaluated. For instance,at $k = {10}$ ,BERT requires nearly ${180} \times$ more FLOPs than ColBERT; at $k = {1000}$ ,BERT’s overhead jumps to ${13},{900} \times$ . It then reaches ${23},{000} \times$ at $k = {2000}$ . In fact,our informal experimentation shows that this orders-of-magnitude gap in FLOPs makes it practical to run ColBERT entirely on the CPU, although CPU-based re-ranking lies outside our scope.

研究图4中的结果，我们注意到，对于相同的模型大小（即12层“基础”Transformer编码器），ColBERT不仅比BERT的计算成本低得多，而且随着排序文档数量的增加，其扩展性也更好。部分原因在于，无论评估的文档数量如何，ColBERT只需对查询进行一次处理。例如，在$k = {10}$时，BERT所需的浮点运算次数比ColBERT多近${180} \times$；在$k = {1000}$时，BERT的额外运算量跃升至${13},{900} \times$。在$k = {2000}$时，该值达到${23},{000} \times$。实际上，我们的非正式实验表明，这种浮点运算次数上的数量级差距使得在CPU上完全运行ColBERT成为可行，尽管基于CPU的重排序不在我们的研究范围内。

<!-- Media -->

<table><tr><td>Method</td><td>MAP</td><td>MRR@10</td></tr><tr><td>BM25 (Anserini)</td><td>15.3</td><td>-</td></tr><tr><td>doc2query</td><td>18.1</td><td>-</td></tr><tr><td>DeepCT</td><td>24.6</td><td>33.2</td></tr><tr><td>${\mathrm{{BM25} + {BERT}}}_{\text{base}}$</td><td>31.0</td><td>-</td></tr><tr><td>$\mathrm{{BM25}} + {\mathrm{{BERT}}}_{\text{large }}$</td><td>33.5</td><td>-</td></tr><tr><td>BM25 + ColBERT</td><td>31.3</td><td>44.3</td></tr></table>

<table><tbody><tr><td>方法</td><td>平均准确率均值（Mean Average Precision，MAP）</td><td>前10的平均倒数排名（Mean Reciprocal Rank at 10，MRR@10）</td></tr><tr><td>BM25算法（Anserini）</td><td>15.3</td><td>-</td></tr><tr><td>文档到查询（doc2query）</td><td>18.1</td><td>-</td></tr><tr><td>深度上下文词项（DeepCT）</td><td>24.6</td><td>33.2</td></tr><tr><td>${\mathrm{{BM25} + {BERT}}}_{\text{base}}$</td><td>31.0</td><td>-</td></tr><tr><td>$\mathrm{{BM25}} + {\mathrm{{BERT}}}_{\text{large }}$</td><td>33.5</td><td>-</td></tr><tr><td>BM25算法 + ColBERT模型</td><td>31.3</td><td>44.3</td></tr></tbody></table>

Table 3: Results on TREC CAR.

表3：TREC CAR数据集上的结果。

<!-- Media -->

Having studied our results on MS MARCO, we now consider TREC CAR, whose official metric is MAP. Results are summarized in Table 3, which includes a number of important baselines (BM25, doc2query, and DeepCT) in addition to re-ranking baselines that have been tested on this dataset. These results directly mirror those with MS MARCO.

在研究了我们在MS MARCO数据集上的结果后，我们现在考虑TREC CAR数据集，其官方评估指标是平均准确率均值（Mean Average Precision，MAP）。结果总结在表3中，除了在该数据集上经过测试的重排序基线模型外，还包括一些重要的基线模型（BM25、doc2query和DeepCT）。这些结果与MS MARCO数据集上的结果直接对应。

### 4.3 End-to-end Top- $k$ Retrieval

### 4.3 端到端前$k$检索

Beyond cheap re-ranking,ColBERT is amenable to top- $k$ retrieval directly from a full collection. Table 2 considers full retrieval, wherein each model retrieves the top-1000 documents directly from MS MARCO's 8.8M documents per query. In addition to MRR@10 and latency in milliseconds, the table reports Recall@50, Recall@200, and Recall@1000, important metrics for a full-retrieval model that essentially filters down a large collection on a per-query basis.

除了低成本的重排序之外，ColBERT还适用于直接从完整文档集合中进行前$k$检索。表2考虑了全量检索，其中每个模型针对每个查询直接从MS MARCO的880万个文档中检索前1000个文档。除了前10的平均倒数排名（Mean Reciprocal Rank at 10，MRR@10）和以毫秒为单位的延迟外，该表还报告了召回率@50、召回率@200和召回率@1000，这些是全量检索模型的重要指标，该模型本质上是按查询对大型文档集合进行筛选。

We compare against BM25, in particular MS MARCO's official BM25 ranking as well as a well-tuned baseline based on the Anserini toolkit. ${}^{7}$ While many other traditional models exist,we are not aware of any that substantially outperform Anserini’s BM25 implementation (e.g., see RM3 in [28], LMDir in [2], or Microsoft's proprietary feature-based RankSVM on the leaderboard).

我们将其与BM25进行比较，特别是MS MARCO的官方BM25排名以及基于Anserini工具包的调优良好的基线模型。${}^{7}$虽然存在许多其他传统模型，但我们不知道有任何模型能显著优于Anserini的BM25实现（例如，参见文献[28]中的RM3、文献[2]中的LMDir，或排行榜上微软专有的基于特征的RankSVM）。

We also compare against doc2query, DeepCT, and docTTTT-Tquery. All three rely on a traditional bag-of-words model (primarily BM25) for retrieval. Crucially, however, they re-weigh the frequency of terms per document and/or expand the set of terms in each document before building the BM25 index. In particular, doc2query expands each document with a pre-defined number of synthetic queries generated by a seq2seq transformer model (which docTTTTquery replaced with a pre-trained language model, T5 [31]). In contrast, DeepCT uses BERT to produce the term frequency component of BM25 in a context-aware manner.

我们还将其与doc2query、DeepCT和docTTTT - Tquery进行比较。这三种方法都依赖于传统的词袋模型（主要是BM25）进行检索。然而，关键的是，它们在构建BM25索引之前会重新权衡每个文档中词项的频率和/或扩展每个文档中的词项集合。具体来说，doc2query使用由序列到序列（seq2seq）Transformer模型生成的预定义数量的合成查询来扩展每个文档（docTTTTquery用预训练语言模型T5 [31]取代了该模型）。相比之下，DeepCT使用BERT以上下文感知的方式生成BM25的词频分量。

For the latency of Anserini's BM25, doc2query, and docTTTT-query, we use the authors’ [26, 28] Anserini-based implementation. While this implementation supports multi-threading, it only utilizes parallelism across different queries. We thus report single-threaded latency for these models, noting that simply parallelizing their computation over shards of the index can substantially decrease their already-low latency. For DeepCT, we only estimate its latency using that of BM25 (as denoted by (est.) in the table), since DeepCT re-weighs BM25's term frequency without modifying the index otherwise. ${}^{8}$ As discussed in $§{4.1}$ ,we use ${\mathrm{{ColBERT}}}_{\mathrm{L}2}$ for end-to-end retrieval, which employs negative squared L2 distance as its vector-similarity function. For its latency, we measure the time for faiss-based candidate filtering and the subsequent re-ranking. In this experiment, faiss uses all available CPU cores.

对于Anserini的BM25、doc2query和docTTTT - query的延迟，我们使用作者在文献[26, 28]中基于Anserini的实现。虽然该实现支持多线程，但它仅在不同查询之间利用并行性。因此，我们报告这些模型的单线程延迟，并指出，简单地在索引分片上并行计算可以显著降低它们原本就较低的延迟。对于DeepCT，我们仅使用BM25的延迟来估计其延迟（如表中用(est.)表示），因为DeepCT在不修改索引的情况下重新权衡BM25的词频。${}^{8}$如文献$§{4.1}$中所讨论的，我们使用${\mathrm{{ColBERT}}}_{\mathrm{L}2}$进行端到端检索，它采用负平方L2距离作为其向量相似度函数。对于其延迟，我们测量基于faiss的候选过滤和后续重排序的时间。在这个实验中，faiss使用所有可用的CPU核心。

Looking at Table 2, we first see Anserini's BM25 baseline at 18.7 MRR@10, noticing its very low latency as implemented in Anserini (which extends the well-known Lucene system), owing to both very cheap operations and decades of bag-of-words top- $k$ retrieval optimizations. The three subsequent baselines, namely doc2query, DeepCT, and docTTTTquery, each brings a decisive enhancement to effectiveness. These improvements come at negligible overheads in latency, since these baselines ultimately rely on BM25-based retrieval. The most effective among these three, docTTTTquery, demonstrates a massive $9\%$ gain over vanilla BM25 by fine-tuning the recent language model T5.

查看表2，我们首先看到Anserini的BM25基线模型的MRR@10为18.7，注意到它在Anserini（扩展了著名的Lucene系统）中的实现具有非常低的延迟，这得益于非常低成本的操作和数十年的词袋前$k$检索优化。接下来的三个基线模型，即doc2query、DeepCT和docTTTTquery，都在有效性方面带来了决定性的提升。由于这些基线模型最终依赖于基于BM25的检索，因此这些改进在延迟方面的开销可以忽略不计。在这三个模型中最有效的docTTTTquery，通过微调最近的语言模型T5，相对于原始的BM25显示出巨大的$9\%$提升。

---

<!-- Footnote -->

${}^{7}$ http://anserini.io/

${}^{7}$ http://anserini.io/

${}^{8}$ In practice,a myriad of reasons could still cause DeepCT’s latency to differ slightly from BM25’s. For instance,the top- $k$ pruning strategy employed,if any,could interact differently with a changed distribution of scores.

${}^{8}$实际上，有许多原因仍可能导致DeepCT的延迟与BM25的延迟略有不同。例如，如果采用了前$k$剪枝策略，它可能会与变化后的分数分布产生不同的交互。

<!-- Footnote -->

---

Shifting our attention to ColBERT's end-to-end retrieval effectiveness, we see its major gains in MRR@10 over all of these end-to-end models. In fact, using ColBERT in the end-to-end setup is superior in terms of MRR@10 to re-ranking with the same model due to the improved recall. Moving beyond MRR@10, we also see large gains in Recall@k for $k$ equals to 50,200,and 1000. For instance, its Recall@50 actually exceeds the official BM25's Recall@1000 and even all but docTTTTTquery's Recall@200, emphasizing the value of end-to-end retrieval (instead of just re-ranking) with ColBERT.

将我们的注意力转向ColBERT（上下文嵌入表示的对比学习）的端到端检索效果，我们可以看到它在MRR@10（前10个结果的平均倒数排名）指标上相对于所有这些端到端模型都有显著提升。事实上，由于召回率的提高，在端到端设置中使用ColBERT在MRR@10方面优于使用相同模型进行重排序。除了MRR@10，我们还发现在$k$等于50、200和1000时，Recall@k（前k个结果的召回率）也有大幅提升。例如，其Recall@50实际上超过了官方BM25（基于词频和逆文档频率的检索算法）的Recall@1000，甚至除了docTTTTTquery之外的所有模型的Recall@200，这凸显了使用ColBERT进行端到端检索（而不仅仅是重排序）的价值。

### 4.4 Ablation Studies

### 4.4 消融研究

<!-- Media -->

<!-- figureText: ColBERT via average similarity (5-layer) [B] MRR@10 ColBERT without query augmentation (5-layer) [C] ColBERT (5-layer) [D] ColBERT (12-layer) [E] CoIBERT + e2e retrieval (12-layer) [F] -->

<img src="https://cdn.noedgeai.com/0195afb9-1285-7cff-a7ae-bc74a3288ac4_8.jpg?x=158&y=690&w=703&h=229&r=0"/>

Figure 5: Ablation results on MS MARCO (Dev). Between brackets is the number of BERT layers used in each model.

图5：MS MARCO（开发集）上的消融实验结果。括号内是每个模型中使用的BERT（双向编码器表示变换器）层数。

<!-- Media -->

The results from §4.2 indicate that ColBERT is highly effective despite the low cost and simplicity of its late interaction mechanism. To better understand the source of this effectiveness, we examine a number of important details in ColBERT's interaction and encoder architecture. For this ablation, we report MRR@10 on the validation set of MS MARCO in Figure 5, which shows our main re-ranking ColBERT model [E], with MRR@10 of 34.9%.

§4.2的结果表明，尽管ColBERT的后期交互机制成本低且简单，但它非常有效。为了更好地理解这种有效性的来源，我们研究了ColBERT交互和编码器架构中的一些重要细节。在这次消融实验中，我们在图5中报告了MS MARCO验证集上的MRR@10，图中展示了我们的主要重排序ColBERT模型[E]，其MRR@10为34.9%。

Due to the cost of training all models, we train a copy of our main model that retains only the first 5 layers of BERT out of 12 (i.e., model [D]) and similarly train all our ablation models for 200k iterations with five BERT layers. To begin with, we ask if the fine-granular interaction in late interaction is necessary. Model [A] tackles this question: it uses BERT to produce a single embedding vector for the query and another for the document, extracted from BERT's [CLS] contextualized embedding and expanded through a linear layer to dimension 4096 (which equals ${N}_{q} \times  {128} = {32} \times  {128}$ ). Relevance is estimated as the inner product of the query's and the document's embeddings, which we found to perform better than cosine similarity for single-vector re-ranking. As the results show, this model is considerably less effective than ColBERT, reinforcing the importance of late interaction.

由于训练所有模型的成本较高，我们训练了一个主模型的副本，该副本仅保留了BERT 12层中的前5层（即模型[D]），并同样对所有消融模型进行了200k次迭代训练，使用5层BERT。首先，我们探讨后期交互中的细粒度交互是否必要。模型[A]解决了这个问题：它使用BERT为查询生成一个单一的嵌入向量，为文档生成另一个嵌入向量，这些向量从BERT的[CLS]上下文嵌入中提取，并通过一个线性层扩展到维度4096（等于${N}_{q} \times  {128} = {32} \times  {128}$）。相关性估计为查询和文档嵌入的内积，我们发现这种方法在单向量重排序方面比余弦相似度更有效。结果显示，该模型的效果远不如ColBERT，这强化了后期交互的重要性。

Subsequently, we ask if our MaxSim-based late interaction is better than other simple alternatives. We test a model [B] that replaces ColBERT's maximum similarity with average similarity. The results suggest the importance of individual terms in the query paying special attention to particular terms in the document. Similarly, the figure emphasizes the importance of our query augmentation mechanism: without query augmentation [C], ColBERT has a noticeably lower MRR@10.Lastly, we see the impact of end-to-end retrieval not only on recall but also on MRR@10.By retrieving directly from the full collection, ColBERT is able to retrieve to the top-10 documents missed entirely from BM25's top-1000.

随后，我们询问基于MaxSim（最大相似度）的后期交互是否优于其他简单的替代方案。我们测试了一个模型[B]，它用平均相似度取代了ColBERT的最大相似度。结果表明，查询中的单个术语特别关注文档中的特定术语非常重要。同样，该图强调了我们的查询增强机制的重要性：如果没有查询增强[C]，ColBERT的MRR@10会明显降低。最后，我们看到端到端检索不仅对召回率有影响，对MRR@10也有影响。通过直接从完整集合中检索，ColBERT能够将BM25前1000个结果中完全遗漏的文档检索到前10个。

<!-- Media -->

<!-- figureText: Basic ColBERT Indexing $\begin{array}{lllll} {10000} & {20000} & {30000} & {40000} & {50000} \end{array}$ Throughput (documents/minute) +multi-GPU document processing +per-batch maximum sequence length +length-based bucketing +multi-core pre-processing -->

<img src="https://cdn.noedgeai.com/0195afb9-1285-7cff-a7ae-bc74a3288ac4_8.jpg?x=941&y=245&w=695&h=202&r=0"/>

Figure 6: Effect of ColBERT's indexing optimizations on the offline indexing throughput.

图6：ColBERT索引优化对离线索引吞吐量的影响。

<!-- Media -->

### 4.5 Indexing Throughput & Footprint

### 4.5 索引吞吐量和空间占用

Lastly, we examine the indexing throughput and space footprint of ColBERT. Figure 6 reports indexing throughput on MS MARCO documents with ColBERT and four other ablation settings, which individually enable optimizations described in $§{3.4}$ on top of basic batched indexing. Based on these throughputs, ColBERT can index MS MARCO in about three hours. Note that any BERT-based model must incur the computational cost of processing each document at least once. While ColBERT encodes each document with BERT exactly once, existing BERT-based rankers would repeat similar computations on possibly hundreds of documents for each query.

最后，我们研究了ColBERT的索引吞吐量和空间占用情况。图6报告了使用ColBERT和其他四种消融设置对MS MARCO文档进行索引的吞吐量，这些设置在基本批量索引的基础上分别启用了$§{3.4}$中描述的优化。基于这些吞吐量，ColBERT大约需要三个小时就能对MS MARCO进行索引。请注意，任何基于BERT的模型都必须至少对每个文档进行一次计算处理。虽然ColBERT使用BERT对每个文档只进行一次编码，但现有的基于BERT的排序器可能会对每个查询的数百个文档重复进行类似的计算。

<!-- Media -->

<table><tr><td>Setting</td><td>Dimension(m)</td><td>Bytes/Dim</td><td>Space(GiBs)</td><td>MRR@10</td></tr><tr><td>Re-rank Cosine</td><td>128</td><td>4</td><td>286</td><td>34.9</td></tr><tr><td>End-to-end L2</td><td>128</td><td>2</td><td>154</td><td>36.0</td></tr><tr><td>Re-rank L2</td><td>128</td><td>2</td><td>143</td><td>34.8</td></tr><tr><td>Re-rank Cosine</td><td>48</td><td>4</td><td>54</td><td>34.4</td></tr><tr><td>Re-rank Cosine</td><td>24</td><td>2</td><td>27</td><td>33.9</td></tr></table>

<table><tbody><tr><td>设置</td><td>维度（米）</td><td>字节/维度</td><td>空间（吉字节）</td><td>前10召回率均值（MRR@10）</td></tr><tr><td>重排序余弦相似度</td><td>128</td><td>4</td><td>286</td><td>34.9</td></tr><tr><td>端到端L2距离</td><td>128</td><td>2</td><td>154</td><td>36.0</td></tr><tr><td>重排序L2距离</td><td>128</td><td>2</td><td>143</td><td>34.8</td></tr><tr><td>重排序余弦相似度</td><td>48</td><td>4</td><td>54</td><td>34.4</td></tr><tr><td>重排序余弦相似度</td><td>24</td><td>2</td><td>27</td><td>33.9</td></tr></tbody></table>

Table 4: Space Footprint vs MRR@ 10 (Dev) on MS MARCO.

表4：MS MARCO数据集上的空间占用与MRR@10（验证集）

<!-- Media -->

Table 4 reports the space footprint of ColBERT under various settings as we reduce the embeddings dimension and/or the bytes per dimension. Interestingly, the most space-efficient setting, that is, re-ranking with cosine similarity with 24-dimensional vectors stored as 2-byte floats, is only 1% worse in MRR@10 than the most space-consuming one, while the former requires only 27 GiBs to represent the MS MARCO collection.

表4展示了在我们降低嵌入维度和/或每维度字节数的各种设置下，ColBERT模型的空间占用情况。有趣的是，最节省空间的设置，即使用24维向量（以2字节浮点数存储）通过余弦相似度进行重排序，其MRR@10仅比最耗费空间的设置低1%，而前者仅需27GB来表示MS MARCO数据集。

## 5 CONCLUSIONS

## 5 结论

In this paper, we introduced ColBERT, a novel ranking model that employs contextualized late interaction over deep LMs (in particular, BERT) for efficient retrieval. By independently encoding queries and documents into fine-grained representations that interact via cheap and pruning-friendly computations, ColBERT can leverage the expressiveness of deep LMs while greatly speeding up query processing. In addition, doing so allows using ColBERT for end-to-end neural retrieval directly from a large document collection. Our results show that ColBERT is more than ${170} \times$ faster and requires ${14},{000} \times$ fewer FLOPs/query than existing BERT-based models,all while only minimally impacting quality and while outperforming every non-BERT baseline.

在本文中，我们介绍了ColBERT，这是一种新颖的排序模型，它在深度语言模型（特别是BERT）上采用上下文延迟交互进行高效检索。通过将查询和文档独立编码为细粒度的表示，并通过低成本且便于剪枝的计算进行交互，ColBERT可以利用深度语言模型的表达能力，同时大大加快查询处理速度。此外，这样做还允许直接从大型文档集合中使用ColBERT进行端到端的神经检索。我们的实验结果表明，与现有的基于BERT的模型相比，ColBERT的速度提高了${170} \times$倍以上，每个查询所需的浮点运算次数（FLOPs）减少了${14},{000} \times$，同时对检索质量的影响极小，并且优于所有非BERT的基线模型。

Acknowledgments. OK was supported by the Eltoukhy Family Graduate Fellowship at the Stanford School of Engineering. This research was supported in part by affiliate members and other supporters of the Stanford DAWN project-Ant Financial, Facebook, Google, Infosys, NEC, and VMware-as well as Cisco, SAP, and the NSF under CAREER grant CNS-1651570. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation.

致谢。OK得到了斯坦福工程学院Eltoukhy家族研究生奖学金的支持。本研究部分得到了斯坦福DAWN项目的附属成员和其他支持者的资助，包括蚂蚁金服、Facebook、谷歌、Infosys、NEC和VMware，以及思科、SAP和美国国家科学基金会（NSF）的职业发展奖（CAREER grant CNS - 1651570）。本材料中表达的任何观点、研究结果、结论或建议均为作者本人的观点，不一定反映美国国家科学基金会的观点。

## REFERENCES

## 参考文献

[1] Firas Abuzaid, Geet Sethi, Peter Bailis, and Matei Zaharia. 2019. To Index or Not to Index: Optimizing Exact Maximum Inner Product Search. In 2019 IEEE 35th International Conference on Data Engineering (ICDE). IEEE, 1250-1261.

[1] Firas Abuzaid、Geet Sethi、Peter Bailis和Matei Zaharia。2019年。是否进行索引：优化精确最大内积搜索。收录于2019年IEEE第35届国际数据工程会议（ICDE）。IEEE，1250 - 1261页。

[2] Zhuyun Dai and Jamie Callan. 2019. Context-Aware Sentence/Passage Term Importance Estimation For First Stage Retrieval. arXiv preprint arXiv:1910.10687 (2019).

[2] Zhuyun Dai和Jamie Callan。2019年。用于第一阶段检索的上下文感知句子/段落术语重要性估计。预印本arXiv:1910.10687（2019年）。

[3] Zhuyun Dai and Jamie Callan. 2019. Deeper Text Understanding for IR with Contextual Neural Language Modeling. arXiv preprint arXiv:1905.09217 (2019).

[3] Zhuyun Dai和Jamie Callan。2019年。通过上下文神经语言建模实现信息检索中的更深入文本理解。预印本arXiv:1905.09217（2019年）。

[4] Zhuyun Dai, Chenyan Xiong, Jamie Callan, and Zhiyuan Liu. 2018. Convolutional neural networks for soft-matching n-grams in ad-hoc search. In Proceedings of the eleventh ACM international conference on web search and data mining. 126-134.

[4] Zhuyun Dai、Chenyan Xiong、Jamie Callan和Zhiyuan Liu。2018年。用于即席搜索中n - 元组软匹配的卷积神经网络。收录于第十一届ACM国际网络搜索与数据挖掘会议论文集。126 - 134页。

[5] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805 (2018).

[5] Jacob Devlin、Ming - Wei Chang、Kenton Lee和Kristina Toutanova。2018年。BERT：用于语言理解的深度双向变换器的预训练。预印本arXiv:1810.04805（2018年）。

[6] Laura Dietz, Manisha Verma, Filip Radlinski, and Nick Craswell. 2017. TREC Complex Answer Retrieval Overview.. In TREC.

[6] Laura Dietz、Manisha Verma、Filip Radlinski和Nick Craswell。2017年。TREC复杂答案检索概述。收录于TREC会议。

[7] Jiafeng Guo, Yixing Fan, Qingyao Ai, and W Bruce Croft. 2016. A deep relevance matching model for ad-hoc retrieval. In Proceedings of the 25th ACM International on Conference on Information and Knowledge Management. ACM, 55-64.

[7] Jiafeng Guo、Yixing Fan、Qingyao Ai和W Bruce Croft。2016年。用于即席检索的深度相关性匹配模型。收录于第25届ACM国际信息与知识管理会议论文集。ACM，55 - 64页。

[8] Jiafeng Guo, Yixing Fan, Liang Pang, Liu Yang, Qingyao Ai, Hamed Zamani, Chen Wu, W Bruce Croft, and Xueqi Cheng. 2019. A deep look into neural ranking models for information retrieval. arXiv preprint arXiv:1903.06902 (2019).

[8] Jiafeng Guo、Yixing Fan、Liang Pang、Liu Yang、Qingyao Ai、Hamed Zamani、Chen Wu、W Bruce Croft和Xueqi Cheng。2019年。深入探究用于信息检索的神经排序模型。预印本arXiv:1903.06902（2019年）。

[9] Sebastian Hofstätter and Allan Hanbury. 2019. Let's measure run time! Extending the IR replicability infrastructure to include performance aspects. arXiv preprint arXiv:1907.04614 (2019).

[9] Sebastian Hofstätter和Allan Hanbury。2019年。让我们来衡量运行时间！将信息检索可重复性基础设施扩展到包括性能方面。预印本arXiv:1907.04614（2019年）。

[10] Sebastian Hofstätter, Navid Rekabsaz, Carsten Eickhoff, and Allan Hanbury. 2019. On the effect of low-frequency terms on neural-IR models. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval. 1137-1140.

[10] Sebastian Hofstätter、Navid Rekabsaz、Carsten Eickhoff和Allan Hanbury。2019年。低频术语对神经信息检索模型的影响。收录于第42届ACM SIGIR国际信息检索研究与发展会议论文集。1137 - 1140页。

[11] Sebastian Hofstätter, Markus Zlabinger, and Allan Hanbury. 2019. TU Wien@ TREC Deep Learning'19-Simple Contextualization for Re-ranking. arXiv preprint arXiv:1912.01385 (2019).

[11] 塞巴斯蒂安·霍夫施泰特（Sebastian Hofstätter）、马库斯·兹拉宾格（Markus Zlabinger）和艾伦·汉伯里（Allan Hanbury）。2019年。维也纳工业大学（TU Wien）@TREC深度学习2019——用于重排序的简单上下文建模。预印本arXiv:1912.01385 (2019)。

[12] Po-Sen Huang, Xiaodong He, Jianfeng Gao, Li Deng, Alex Acero, and Larry Heck. 2013. Learning deep structured semantic models for web search using clickthrough data. In Proceedings of the 22nd ACM international conference on Information & Knowledge Management. 2333-2338.

[12] 黄伯森（Po-Sen Huang）、何晓东（Xiaodong He）、高剑锋（Jianfeng Gao）、邓力（Li Deng）、亚历克斯·阿塞罗（Alex Acero）和拉里·赫克（Larry Heck）。2013年。利用点击数据学习用于网络搜索的深度结构化语义模型。见第22届ACM信息与知识管理国际会议论文集。2333 - 2338。

[13] Shiyu Ji, Jinjin Shao, and Tao Yang. 2019. Efficient Interaction-based Neural Ranking with Locality Sensitive Hashing. In The World Wide Web Conference. ACM, 2858-2864.

[13] 季世玉（Shiyu Ji）、邵金金（Jinjin Shao）和杨涛（Tao Yang）。2019年。基于局部敏感哈希的高效基于交互的神经排序。见万维网会议论文集。ACM，2858 - 2864。

[14] Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, and Qun Liu. 2019. Tinybert: Distilling bert for natural language understanding. arXiv preprint arXiv:1909.10351 (2019).

[14] 焦小奇（Xiaoqi Jiao）、尹宜春（Yichun Yin）、尚立峰（Lifeng Shang）、蒋鑫（Xin Jiang）、陈晓（Xiao Chen）、李林林（Linlin Li）、王芳（Fang Wang）和刘群（Qun Liu）。2019年。TinyBERT：为自然语言理解蒸馏BERT模型。预印本arXiv:1909.10351 (2019)。

[15] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2017. Billion-scale similarity search with GPUs. arXiv preprint arXiv:1702.08734 (2017).

[15] 杰夫·约翰逊（Jeff Johnson）、马蒂亚斯·杜泽（Matthijs Douze）和埃尔韦·热古（Hervé Jégou）。2017年。使用GPU进行十亿级相似度搜索。预印本arXiv:1702.08734 (2017)。

[16] Diederik P Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014).

[16] 迪德里克·P·金马（Diederik P Kingma）和吉米·巴（Jimmy Ba）。2014年。Adam：一种随机优化方法。预印本arXiv:1412.6980 (2014)。

[17] Ron Kohavi, Alex Deng, Brian Frasca, Toby Walker, Ya Xu, and Nils Pohlmann. 2013. Online controlled experiments at large scale. In SIGKDD.

[17] 罗恩·科哈维（Ron Kohavi）、亚历克斯·邓（Alex Deng）、布莱恩·弗拉斯卡（Brian Frasca）、托比·沃克（Toby Walker）、徐雅（Ya Xu）和尼尔斯·波尔曼（Nils Pohlmann）。2013年。大规模在线控制实验。见知识发现与数据挖掘会议（SIGKDD）论文集。

[18] Sean MacAvaney, Andrew Yates, Arman Cohan, and Nazli Goharian. 2019. Cedr: Contextualized embeddings for document ranking. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval. ACM, 1101-1104.

[18] 肖恩·麦卡瓦尼（Sean MacAvaney）、安德鲁·耶茨（Andrew Yates）、阿尔曼·科汉（Arman Cohan）和纳兹利·戈哈瑞安（Nazli Goharian）。2019年。CEDR：用于文档排序的上下文嵌入。见第42届ACM信息检索研究与发展国际会议论文集。ACM，1101 - 1104。

[19] Paul Michel, Omer Levy, and Graham Neubig. 2019. Are Sixteen Heads Really Better than One?. In Advances in Neural Information Processing Systems. 14014- 14024.

[19] 保罗·米歇尔（Paul Michel）、奥默·利维（Omer Levy）和格雷厄姆·纽比格（Graham Neubig）。2019年。十六个头真的比一个头好吗？见神经信息处理系统进展。14014 - 14024。

[20] Bhaskar Mitra and Nick Craswell. 2019. An Updated Duet Model for Passage

[20] 巴斯卡尔·米特拉（Bhaskar Mitra）和尼克·克拉斯韦尔（Nick Craswell）。2019年。用于段落重排序的更新版二重奏模型

Re-ranking. arXiv preprint arXiv:1903.07666 (2019).

预印本arXiv:1903.07666 (2019)。

[21] Bhaskar Mitra, Nick Craswell, et al. 2018. An introduction to neural information retrieval. Foundations and Trends® in Information Retrieval 13, 1 (2018), 1-126.

[21] 巴斯卡尔·米特拉（Bhaskar Mitra）、尼克·克拉斯韦尔（Nick Craswell）等。2018年。神经信息检索导论。信息检索基础与趋势® 13, 1 (2018)，1 - 126。

[22] Bhaskar Mitra, Fernando Diaz, and Nick Craswell. 2017. Learning to match using local and distributed representations of text for web search. In Proceedings of the 26th International Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 1291-1299.

[22] 巴斯卡尔·米特拉（Bhaskar Mitra）、费尔南多·迪亚兹（Fernando Diaz）和尼克·克拉斯韦尔（Nick Craswell）。2017年。学习使用文本的局部和分布式表示进行网络搜索匹配。见第26届万维网国际会议论文集。万维网国际会议指导委员会，1291 - 1299。

[23] Bhaskar Mitra, Corby Rosset, David Hawking, Nick Craswell, Fernando Diaz, and Emine Yilmaz. 2019. Incorporating query term independence assumption for efficient retrieval and ranking using deep neural networks. arXiv preprint arXiv:1907.03693 (2019).

[23] 巴斯卡尔·米特拉（Bhaskar Mitra）、科比·罗塞特（Corby Rosset）、大卫·霍金（David Hawking）、尼克·克拉斯韦尔（Nick Craswell）、费尔南多·迪亚兹（Fernando Diaz）和埃米内·伊尔马兹（Emine Yilmaz）。2019年。利用深度神经网络结合查询词独立性假设进行高效检索和排序。预印本arXiv:1907.03693 (2019)。

[24] Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, and Li Deng. 2016. MS MARCO: A Human-Generated MAchine Reading COmprehension Dataset. (2016).

[24] 特里·阮（Tri Nguyen）、米尔·罗森伯格（Mir Rosenberg）、宋霞（Xia Song）、高剑锋（Jianfeng Gao）、索拉布·蒂瓦里（Saurabh Tiwary）、兰甘·马朱姆德（Rangan Majumder）和邓力（Li Deng）。2016年。MS MARCO：一个人工生成的机器阅读理解数据集。(2016)。

[25] Rodrigo Nogueira and Kyunghyun Cho. 2019. Passage Re-ranking with BERT. arXiv preprint arXiv:1901.04085 (2019).

[25] 罗德里戈·诺盖拉（Rodrigo Nogueira）和赵京焕（Kyunghyun Cho）。2019年。使用BERT进行段落重排序。预印本arXiv:1901.04085 (2019)。

[26] Rodrigo Nogueira, Jimmy Lin, and AI Epistemic. 2019. From doc2query to docTTTTTquery. (2019).

[26] 罗德里戈·诺盖拉（Rodrigo Nogueira）、吉米·林（Jimmy Lin）和人工智能认知（AI Epistemic）。2019年。从文档到查询（doc2query）到文档到超多查询（docTTTTTquery）。(2019年)。

[27] Rodrigo Nogueira, Wei Yang, Kyunghyun Cho, and Jimmy Lin. 2019. Multi-Stage Document Ranking with BERT. arXiv preprint arXiv:1910.14424 (2019).

[27] 罗德里戈·诺盖拉（Rodrigo Nogueira）、杨威（Wei Yang）、赵京焕（Kyunghyun Cho）和吉米·林（Jimmy Lin）。2019年。基于BERT的多阶段文档排序。预印本arXiv:1910.14424 (2019年)。

[28] Rodrigo Nogueira, Wei Yang, Jimmy Lin, and Kyunghyun Cho. 2019. Document Expansion by Query Prediction. arXiv preprint arXiv:1904.08375 (2019).

[28] 罗德里戈·诺盖拉（Rodrigo Nogueira）、杨威（Wei Yang）、吉米·林（Jimmy Lin）和赵京焕（Kyunghyun Cho）。2019年。通过查询预测进行文档扩展。预印本arXiv:1904.08375 (2019年)。

[29] Matthew E Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. 2018. Deep contextualized word representations. arXiv preprint arXiv:1802.05365 (2018).

[29] 马修·E·彼得斯（Matthew E Peters）、马克·诺伊曼（Mark Neumann）、莫希特·伊耶尔（Mohit Iyyer）、马特·加德纳（Matt Gardner）、克里斯托弗·克拉克（Christopher Clark）、肯顿·李（Kenton Lee）和卢克·泽特尔莫耶（Luke Zettlemoyer）。2018年。深度上下文词表示。预印本arXiv:1802.05365 (2018年)。

[30] Yifan Qiao, Chenyan Xiong, Zhenghao Liu, and Zhiyuan Liu. 2019. Understanding the Behaviors of BERT in Ranking. arXiv preprint arXiv:1904.07531 (2019).

[30] 乔一帆（Yifan Qiao）、熊晨彦（Chenyan Xiong）、刘正浩（Zhenghao Liu）和刘志远（Zhiyuan Liu）。2019年。理解BERT在排序中的行为。预印本arXiv:1904.07531 (2019年)。

[31] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2019. Exploring the limits of transfer learning with a unified text-to-text transformer. arXiv preprint arXiv:1910.10683 (2019).

[31] 科林·拉菲尔（Colin Raffel）、诺姆·沙泽尔（Noam Shazeer）、亚当·罗伯茨（Adam Roberts）、凯瑟琳·李（Katherine Lee）、沙兰·纳朗（Sharan Narang）、迈克尔·马特纳（Michael Matena）、周燕琪（Yanqi Zhou）、李威（Wei Li）和彼得·J·刘（Peter J Liu）。2019年。用统一的文本到文本转换器探索迁移学习的极限。预印本arXiv:1910.10683 (2019年)。

[32] Stephen E Robertson, Steve Walker, Susan Jones, Micheline M Hancock-Beaulieu, Mike Gatford, et al. 1995. Okapi at TREC-3. NIST Special Publication (1995).

[32] 斯蒂芬·E·罗伯逊（Stephen E Robertson）、史蒂夫·沃克（Steve Walker）、苏珊·琼斯（Susan Jones）、米歇琳·M·汉考克 - 博略（Micheline M Hancock - Beaulieu）、迈克·加特福德（Mike Gatford）等。1995年。TREC - 3会议上的Okapi系统。美国国家标准与技术研究院特别出版物 (1995年)。

[33] Raphael Tang, Yao Lu, Linqing Liu, Lili Mou, Olga Vechtomova, and Jimmy Lin. 2019. Distilling task-specific knowledge from BERT into simple neural networks. arXiv preprint arXiv:1903.12136 (2019).

[33] 拉斐尔·唐（Raphael Tang）、姚璐（Yao Lu）、刘林清（Linqing Liu）、牟丽丽（Lili Mou）、奥尔加·韦克托莫娃（Olga Vechtomova）和吉米·林（Jimmy Lin）。2019年。将特定任务知识从BERT蒸馏到简单神经网络中。预印本arXiv:1903.12136 (2019年)。

[34] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in neural information processing systems. 5998-6008.

[34] 阿什什·瓦斯瓦尼（Ashish Vaswani）、诺姆·沙泽尔（Noam Shazeer）、尼基·帕尔马尔（Niki Parmar）、雅各布·乌斯库雷特（Jakob Uszkoreit）、利昂·琼斯（Llion Jones）、艾丹·N·戈麦斯（Aidan N Gomez）、卢卡斯·凯泽（Lukasz Kaiser）和伊利亚·波洛苏金（Illia Polosukhin）。2017年。注意力就是你所需要的一切。《神经信息处理系统进展》。5998 - 6008页。

[35] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. 2016. Google's neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144 (2016).

[35] 吴永辉（Yonghui Wu）、迈克·舒斯特（Mike Schuster）、陈志峰（Zhifeng Chen）、勒·奎克·V（Quoc V Le）、穆罕默德·诺鲁兹（Mohammad Norouzi）、沃尔夫冈·马赫雷（Wolfgang Macherey）、马克西姆·克里昆（Maxim Krikun）、曹原（Yuan Cao）、高琴（Qin Gao）、克劳斯·马赫雷（Klaus Macherey）等。2016年。谷歌的神经机器翻译系统：缩小人类和机器翻译之间的差距。预印本arXiv:1609.08144 (2016年)。

[36] Chenyan Xiong, Zhuyun Dai, Jamie Callan, Zhiyuan Liu, and Russell Power. 2017. End-to-end neural ad-hoc ranking with kernel pooling. In Proceedings of the 40th International ACM SIGIR conference on research and development in information retrieval. 55-64.

[36] 熊晨彦（Chenyan Xiong）、戴竹云（Zhuyun Dai）、杰米·卡兰（Jamie Callan）、刘志远（Zhiyuan Liu）和拉塞尔·鲍尔（Russell Power）。2017年。基于核池化的端到端神经临时排序。《第40届国际ACM SIGIR信息检索研究与发展会议论文集》。55 - 64页。

[37] Peilin Yang, Hui Fang, and Jimmy Lin. 2018. Anserini: Reproducible ranking baselines using Lucene. Journal of Data and Information Quality (JDIQ) 10,4 (2018), 1-20.

[37] 杨培林（Peilin Yang）、方慧（Hui Fang）和吉米·林（Jimmy Lin）。2018年。Anserini：使用Lucene的可重现排序基线。《数据与信息质量杂志》（JDIQ）10,4 (2018年)，1 - 20页。

[38] Wei Yang, Kuang Lu, Peilin Yang, and Jimmy Lin. 2019. Critically Examining the" Neural Hype" Weak Baselines and the Additivity of Effectiveness Gains from Neural Ranking Models. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval. 1129-1132.

[38] 杨威（Wei Yang）、卢匡（Kuang Lu）、杨培林（Peilin Yang）和吉米·林（Jimmy Lin）。2019年。批判性审视“神经热潮”：薄弱基线以及神经排序模型有效性增益的可加性。《第42届国际ACM SIGIR信息检索研究与发展会议论文集》。1129 - 1132页。

[39] Zeynep Akkalyoncu Yilmaz, Wei Yang, Haotian Zhang, and Jimmy Lin. 2019. Cross-domain modeling of sentence-level evidence for document retrieval. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). 3481-3487.

[39] 泽内普·阿卡利永丘·伊尔马兹（Zeynep Akkalyoncu Yilmaz）、杨威（Wei Yang）、张浩天（Haotian Zhang）和吉米·林（Jimmy Lin）。2019年。用于文档检索的句子级证据的跨领域建模。《2019年自然语言处理经验方法会议和第9届自然语言处理国际联合会议论文集》（EMNLP - IJCNLP）。3481 - 3487页。

[40] Ofir Zafrir, Guy Boudoukh, Peter Izsak, and Moshe Wasserblat. 2019. Q8bert: Quantized 8bit bert. arXiv preprint arXiv:1910.06188 (2019).

[40] 奥菲尔·扎夫里尔（Ofir Zafrir）、盖伊·布杜赫（Guy Boudoukh）、彼得·伊萨克（Peter Izsak）和摩西·瓦瑟布拉特（Moshe Wasserblat）。2019年。Q8bert：量化8位BERT。预印本arXiv:1910.06188 (2019年)。

[41] Hamed Zamani, Mostafa Dehghani, W Bruce Croft, Erik Learned-Miller, and Jaap Kamps. 2018. From neural re-ranking to neural ranking: Learning a sparse representation for inverted indexing. In Proceedings of the 27th ACM International Conference on Information and Knowledge Management. ACM, 497-506.

[41] 哈米德·扎马尼（Hamed Zamani）、穆斯塔法·德赫加尼（Mostafa Dehghani）、W·布鲁斯·克罗夫特（W Bruce Croft）、埃里克·利尔内德 - 米勒（Erik Learned - Miller）和亚普·坎普斯（Jaap Kamps）。2018年。从神经重排序到神经排序：学习用于倒排索引的稀疏表示。见第27届ACM国际信息与知识管理会议论文集。美国计算机协会（ACM），497 - 506页。

[42] Le Zhao. 2012. Modeling and solving term mismatch for full-text retrieval. Ph.D. Dissertation. Carnegie Mellon University.

[42] 赵乐（Le Zhao）。2012年。全文检索中术语不匹配问题的建模与求解。博士学位论文。卡内基梅隆大学（Carnegie Mellon University）。