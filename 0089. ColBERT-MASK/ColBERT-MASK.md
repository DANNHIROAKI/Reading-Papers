# Beneath the [MASK]: An Analysis of Structural Query Tokens in ColBERT

# [掩码]之下：ColBERT中结构查询Token（Token）的分析

Ben Giacalone ${}^{1\left\lbrack  {{0009} - {0006} - {8525} - {959X}}\right\rbrack  }$ ,Greg Paiement ${}^{1\left\lbrack  {{0009} - {0007} - {0819} - {5315}}\right\rbrack  }$ ,

本·贾卡洛内 ${}^{1\left\lbrack  {{0009} - {0006} - {8525} - {959X}}\right\rbrack  }$ ，格雷格·佩门特 ${}^{1\left\lbrack  {{0009} - {0007} - {0819} - {5315}}\right\rbrack  }$ ，

Quinn Tucker ${}^{1\left\lbrack  {{0009} - {0007} - {8885} - {245X}}\right\rbrack  }$ ,and Richard Zanibbi ${}^{1\left\lbrack  {{0000} - {0001} - {5921} - {9750}}\right\rbrack  }$

奎因·塔克 ${}^{1\left\lbrack  {{0009} - {0007} - {8885} - {245X}}\right\rbrack  }$ ，以及理查德·扎尼比 ${}^{1\left\lbrack  {{0000} - {0001} - {5921} - {9750}}\right\rbrack  }$

Rochester Institute of Technology, Rochester NY 14623, USA

美国纽约州罗切斯特市罗切斯特理工学院，邮编14623

Abstract. ColBERT is a highly effective and interpretable retrieval model based on token embeddings. For scoring, the model adds cosine similarities between the most similar pairs of query and document token embeddings. Previous work on interpreting how tokens affect scoring pay little attention to non-text tokens used in ColBERT such as [MASK]. Using MS MARCO and the TREC 2019-2020 deep passage retrieval task, we show that $\left\lbrack  \mathrm{{MASK}}\right\rbrack$ embeddings may be replaced by other query and structural token embeddings to obtain similar effectiveness, and that [Q] and [MASK] are sensitive to token order, while [CLS] and [SEP] are not.

摘要。ColBERT是一种基于Token嵌入（Token embeddings）的高效且可解释的检索模型。在评分时，该模型会累加查询和文档Token嵌入中最相似对之间的余弦相似度。以往关于解释Token如何影响评分的研究很少关注ColBERT中使用的非文本Token，如 [掩码] 。通过使用MS MARCO和TREC 2019 - 2020深度段落检索任务，我们发现 $\left\lbrack  \mathrm{{MASK}}\right\rbrack$ 嵌入可以被其他查询和结构Token嵌入替代以获得相似的效果，并且 [Q] 和 [掩码] 对Token顺序敏感，而 [CLS] 和 [SEP] 则不敏感。

Keywords: ColBERT - interpretability - embeddings - relevance scoring

关键词：ColBERT - 可解释性 - 嵌入 - 相关性评分

## 1 Introduction

## 1 引言

The ColBERT [4] retrieval model uses BERT [2] to produce token embeddings for document and query passages. Typically, candidate documents are retrieved using dense retrieval on embedded tokens $\left\lbrack  {{15},{17}}\right\rbrack$ ,and then re-scored using the sum of maximum cosine similarities between each query token embedding and its most similar document token embedding via the MaxSim operator. Rescoring improves retrieval effectiveness, and is more interpretable than dense retrieval models that use single vectors (e.g. the BERT [CLS] token), because query tokens contribute individually to document rank scores [3], and token embeddings can be analyzed directly.

ColBERT [4] 检索模型使用BERT [2] 为文档和查询段落生成Token嵌入（Token embeddings）。通常，候选文档通过对嵌入Token $\left\lbrack  {{15},{17}}\right\rbrack$ 进行密集检索来获取，然后通过MaxSim算子，使用每个查询Token嵌入与其最相似的文档Token嵌入之间的最大余弦相似度之和进行重新评分。重新评分提高了检索效果，并且比使用单个向量的密集检索模型（例如BERT [CLS] Token）更具可解释性，因为查询Token会单独对文档排名得分产生影响 [3] ，并且可以直接分析Token嵌入。

Interestingly, not all tokens used in ColBERT's scoring are text tokens. Some are structural tokens that mark locations and segments in a token sequence. ColBERT employs a modified BERT model to create contextualized embeddings for every document and query token, including structural BERT tokens. Structural tokens include [CLS], which appears at the input start, followed by [Q] or [D] to signify whether a passage is from a query or a document. Text tokens from the query are next, followed by [SEP] after the final text token. Query token sequences shorter than the input size are padded with [MASK] tokens, ${}^{1}$ and document token sequences are padded with [PAD] tokens. Below are example query and document passage tokenizations with input sizes of 32 and 180 tokens, respectively. Subscripts are used to indicate token position in the input.

有趣的是，ColBERT评分中使用的并非所有Token都是文本Token。有些是用于标记Token序列中位置和片段的结构Token。ColBERT采用经过修改的BERT模型为每个文档和查询Token（包括BERT结构Token）创建上下文嵌入。结构Token包括 [CLS] ，它出现在输入的开头，后面跟着 [Q] 或 [D] 以表示段落是来自查询还是文档。接下来是查询中的文本Token，最后一个文本Token后面跟着 [SEP] 。比输入大小短的查询Token序列用 [掩码] Token填充 ${}^{1}$ ，文档Token序列用 [PAD] Token填充。以下分别是输入大小为32和180个Token的查询和文档段落分词示例。下标用于表示Token在输入中的位置。

---

<!-- Footnote -->

${}^{1}$ [MASK] was originally devised for BERT to represent a "hidden" input token in its masked token prediction training task.

${}^{1}$ [掩码] 最初是为BERT在其掩码Token预测训练任务中表示“隐藏”的输入Token而设计的。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: of [Q] 0.3 COST [CLS] 0.2 0.1 0.0 -0.1 -0.2 hulless 0.3 0.2 0.1 「SEI swim spa pools 0.20 0.25 0.30 0.35 -->

<img src="https://cdn.noedgeai.com/0195b3f5-2270-7ac8-ad46-761ec932728e_1.jpg?x=612&y=343&w=587&h=535&r=0"/>

Fig. 1. PCA embeddings for the MS MARCO query "cost of endless pools swim spa". [MASK] tokens (red points) cluster around query words and structural tokens (black).

图1. MS MARCO查询“无尽泳池游泳水疗的成本”的主成分分析（PCA）嵌入。[掩码] Token（红点）聚集在查询词和结构Token（黑点）周围。

<!-- Media -->

Q: ${\left\lbrack  \mathrm{{CLS}}\right\rbrack  }_{1}\;{\left\lbrack  \mathrm{Q}\right\rbrack  }_{2}\;{\operatorname{cost}}_{3}\;{\mathrm{{of}}}_{4}\;{\mathrm{{endless}}}_{5}\;{\mathrm{{pools}}}_{5}\;{\mathrm{{swim}}}_{6}\;{\mathrm{{spa}}}_{7}\;{\left\lbrack  \mathrm{{SEP}}\right\rbrack  }_{8}\;{\left\lbrack  \mathrm{{MASK}}\right\rbrack  }_{9}\;{\left\lbrack  \mathrm{{MASK}}\right\rbrack  }_{10}$ ${\left\lbrack  \text{MASK}\right\rbrack  }_{11}\ldots \;{\left\lbrack  \text{MASK}\right\rbrack  }_{30}\;{\left\lbrack  \text{MASK}\right\rbrack  }_{31}\;{\left\lbrack  \text{MASK}\right\rbrack  }_{32}$

D: ${\left\lbrack  \mathrm{{CLS}}\right\rbrack  }_{1}{\left\lbrack  \mathrm{D}\right\rbrack  }_{2}{\text{prices}}_{3}\ldots {\operatorname{swim}}_{12}{\operatorname{spa}}_{13} \cdot  {14}\;{\left\lbrack  \mathrm{{SEP}}\right\rbrack  }_{15}\;{\left\lbrack  \mathrm{{PAD}}\right\rbrack  }_{16}\;{\left\lbrack  \mathrm{{PAD}}\right\rbrack  }_{17}\;{\left\lbrack  \mathrm{{PAD}}\right\rbrack  }_{18}$ ... [PAD] 178 [PAD] 179 [PAD] 180

D: ${\left\lbrack  \mathrm{{CLS}}\right\rbrack  }_{1}{\left\lbrack  \mathrm{D}\right\rbrack  }_{2}{\text{prices}}_{3}\ldots {\operatorname{swim}}_{12}{\operatorname{spa}}_{13} \cdot  {14}\;{\left\lbrack  \mathrm{{SEP}}\right\rbrack  }_{15}\;{\left\lbrack  \mathrm{{PAD}}\right\rbrack  }_{16}\;{\left\lbrack  \mathrm{{PAD}}\right\rbrack  }_{17}\;{\left\lbrack  \mathrm{{PAD}}\right\rbrack  }_{18}$ ... [PAD] 178 [PAD] 179 [PAD] 180

Figure 1 shows the token embeddings for the query above. ${}^{2}$ As seen in Figure 1, [MASK] tokens tend to cluster around other query tokens, giving them additional weight $\left\lbrack  {4,{13}}\right\rbrack$ . The original ColBERT paper suggests [MASK] tokens provide a form of query augmentation through term re-weighting and query expansion. Wang et al. $\left\lbrack  {{12},{13}}\right\rbrack$ study a version of ColBERT that performs query expansion using pseudo-relevance feedback, and find that [MASK] tokens generally do not expand the query by matching terms outside the query, and instead need to add them explicitly. In this way, [MASK] tokens primarily weight query tokens by matching query text tokens in documents. Wang et al. [14] also find that for many ColBERT based models, using only query structural tokens for retrieval ([CLS], [SEP], [Q], [MASK]) is nearly as effective as using all token embeddings for retrieval, and outperforms using only low IDF query token embeddings.

图1展示了上述查询的词元嵌入（token embeddings）情况。${}^{2}$ 如图1所示，[MASK]词元倾向于聚集在其他查询词元周围，从而为它们赋予额外的权重$\left\lbrack  {4,{13}}\right\rbrack$。原始的ColBERT论文表明，[MASK]词元通过词项重新加权和查询扩展的方式提供了一种查询增强形式。Wang等人$\left\lbrack  {{12},{13}}\right\rbrack$研究了一个使用伪相关反馈进行查询扩展的ColBERT版本，发现[MASK]词元通常不会通过匹配查询之外的词项来扩展查询，而是需要显式地添加这些词项。通过这种方式，[MASK]词元主要通过匹配文档中的查询文本词元来对查询词元进行加权。Wang等人[14]还发现，对于许多基于ColBERT的模型而言，仅使用查询结构词元（[CLS]、[SEP]、[Q]、[MASK]）进行检索的效果几乎与使用所有词元嵌入进行检索的效果相当，并且优于仅使用低逆文档频率（IDF）查询词元嵌入进行检索的效果。

Previous studies of ColBERT's retrieval behavior have focused on text tokens. In considering why ColBERT's ranking mechanism outperforms standard lexical models such as BM25, Formal et al. [3] focus on query text tokens, and find that tokens with high Inverse Document Frequency (IDF) produce more exact matches in ColBERT query/document token alignments (e.g. (Q: pool, D: pool)) while low IDF terms produce more inexact matches (e.g. (Q:is,D:and)). Low IDF token embeddings also tend to shift position more, and removing high IDF tokens perturbs ranking more than removing low IDF tokens. MacAvaney et al. [5] also found a sensitivity for text tokens in ColBERT, with misspellings harming retrieval more than in lexical models. Curiously, they also find ColBERT increasing document scores when non-relevant tokens are appended to a document token sequence, while appending relevant terms decreases rank scores even after controlling for document length. Perhaps appending relevant terms produces an 'unnatural' token sequence for the embedding language model which interferes with token embedding/contextualization and MaxSim scoring.

以往对ColBERT检索行为的研究主要集中在文本词元上。在探讨ColBERT的排序机制为何优于诸如BM25等标准词法模型时，Formal等人[3]聚焦于查询文本词元，发现具有高逆文档频率（IDF）的词元在ColBERT查询/文档词元对齐中会产生更多精确匹配（例如（查询：pool，文档：pool）），而低IDF词项则会产生更多不精确匹配（例如（查询：is，文档：and））。低IDF词元嵌入也更倾向于改变位置，并且移除高IDF词元对排序的干扰比移除低IDF词元更大。MacAvaney等人[5]还发现ColBERT对文本词元较为敏感，拼写错误对检索的影响比在词法模型中更大。奇怪的是，他们还发现当将不相关的词元附加到文档词元序列中时，ColBERT会提高文档得分，而附加相关词项即使在控制文档长度之后也会降低排名得分。也许附加相关词项会为嵌入语言模型生成一个“不自然”的词元序列，从而干扰词元嵌入/上下文处理和最大相似度（MaxSim）评分。

---

<!-- Footnote -->

${}^{2}$ Interactive version: https://cs.rit.edu/~bsg8294/colbert/query_viz.html

${}^{2}$ 交互式版本：https://cs.rit.edu/~bsg8294/colbert/query_viz.html

<!-- Footnote -->

---

In this paper, we extend inquiries into how tokens impact retrieval in ColBERT by shifting focus to structural tokens, and [MASK] in particular. In the next Section we present experiments to address the following research questions:

在本文中，我们将研究重点转移到结构词元，尤其是[MASK]词元，从而拓展对词元如何影响ColBERT检索的研究。在下一节中，我们将通过实验来回答以下研究问题：

RQ1. Do [MASK] tokens perform more than just term weighting?

RQ1. [MASK]词元的作用是否仅仅是词项加权？

RQ2. How sensitive are [CLS], [SEP], [Q], and [MASK] to query token order?

RQ2. [CLS]、[SEP]、[Q]和[MASK]对查询词元顺序的敏感程度如何？

## 2 Methodology and Experimental Designs

## 2 方法与实验设计

For our experiments, we use the ColBERT v1 model integrated within PyTer-rier [6]. The state-of-the-art ColBERT v2 [9] model adds index compression and training with hard negatives and distillation to improve rank quality. Index compression and embedding modifications may alter retrieval candidates and token cosine similarities, and we plan to check this in future work. However, we wish to first study the simpler, original ColBERT model. We also believe insights into the workings of ColBERT v1 and models inspired by it (e.g. the text/image model FILIP [16]) will be beneficial for the research community.

在我们的实验中，我们使用集成在PyTer - rier中的ColBERT v1模型[6]。最先进的ColBERT v2模型[9]增加了索引压缩以及使用难负样本和蒸馏进行训练的方法，以提高排名质量。索引压缩和嵌入修改可能会改变检索候选集和词元余弦相似度，我们计划在未来的工作中对此进行验证。然而，我们希望首先研究更简单的原始ColBERT模型。我们还认为，深入了解ColBERT v1及其启发的模型（例如文本/图像模型FILIP [16]）的工作原理将对研究界有益。

Implementation, Datasets, and Metrics. We use a ColBERT v1 checkpoint from the University of Glasgow trained on passage ranking triples for ${44}\mathrm{k}$ batches, ${}^{3}$ and run experiments on a server with 4 Intel Xeon E5-2667v4 CPUs, 4 NVIDIA RTX2080-Ti GPUs, and 512 GB RAM. For our experiments, we use two datasets:

实现、数据集和指标。我们使用格拉斯哥大学在段落排名三元组上训练的ColBERT v1检查点，训练了${44}\mathrm{k}$批次，${}^{3}$ 并在一台配备4个英特尔至强E5 - 2667v4 CPU、4个英伟达RTX2080 - Ti GPU和512 GB内存的服务器上进行实验。在我们的实验中，我们使用了两个数据集：

1. MS MARCO [7]'s passage retrieval dev set (8.8 million documents, 1 million queries, binary relevance judgements). Each query has at most 1 matching document. We use this dataset for query statistics (e.g. cosine distances between query embeddings).

1. MS MARCO [7]的段落检索开发集（880万篇文档、100万个查询、二元相关性判断）。每个查询最多有1篇匹配文档。我们使用这个数据集来进行查询统计（例如查询嵌入之间的余弦距离）。

2. A dataset combining test queries from the TREC 2019 [11] and 2020 [1] deep passage retrieval task (99 queries, graded relevance judgements). Collection documents are the same as MS MARCO. We use this dataset for experiments focused upon retrieval quality. ${}^{4}$

2. 一个将TREC 2019 [11]和2020 [1]深度段落检索任务的测试查询组合而成的数据集（99个查询、分级相关性判断）。集合文档与MS MARCO相同。我们使用这个数据集来进行聚焦于检索质量的实验。${}^{4}$

For the TREC 2019-2020 collection, the relevance scale is between 0 and 3 with a score of 2 considered relevant for metrics using binary relevance (e.g. MAP). We examine relevance scores thresholded at 1, 2, and 3 to see the effect of binarizing at different relevance grades. We use MRR@10 to characterize effectiveness for top results, MAP to characterize effectiveness for complete rankings, and to complement MAP we use nDCG@k measures $\left( {k \in  \{ {10},{1000}\} }\right)$ to utilize graded relevance labels from the TREC data.

对于TREC 2019 - 2020数据集，相关性等级范围为0到3，在使用二元相关性的指标（如平均准确率均值，MAP）中，得分为2被视为相关。我们研究了阈值分别设为1、2和3的相关性得分，以了解在不同相关性等级进行二元化处理的效果。我们使用前10个结果的平均倒数排名（MRR@10）来衡量顶部结果的有效性，使用平均准确率均值（MAP）来衡量完整排名的有效性，并且为了补充MAP，我们使用归一化折损累积增益（nDCG@k）指标 $\left( {k \in  \{ {10},{1000}\} }\right)$ 来利用TREC数据中的分级相关性标签。

---

<!-- Footnote -->

${}^{3}$ http://www.dcs.gla.ac.uk/ craigm/ecir2021-tutorial/colbert_model_checkpoint.zip

${}^{3}$ http://www.dcs.gla.ac.uk/ craigm/ecir2021-tutorial/colbert_model_checkpoint.zip

${}^{4}$ Running the TREC test queries takes roughly 15 minutes to complete using a multithreaded Rust program: https://github.com/Boxxfish/IR2023-Project

${}^{4}$ 使用多线程的Rust程序运行TREC测试查询大约需要15分钟才能完成：https://github.com/Boxxfish/IR2023 - Project

<!-- Footnote -->

---

RQ1: Do [MASK] tokens perform more than just term weighting? Figure 1 illustrates how [MASK] token embeddings cluster around query terms, which was consistent for MS MARCO queries we examined. As mentioned previously, [MASK] tokens have been identified as having two roles in scoring: (1) query term weighting through matching document terms to [MASK] tokens with embeddings similar to non-[MASK] query tokens, and (2) query expansion through [MASK] embed-dings shifting toward potentially relevant tokens outside of the query.

研究问题1（RQ1）：[MASK]标记的作用是否不仅仅是词项加权？图1展示了[MASK]标记嵌入如何围绕查询词项聚类，这与我们所研究的MS MARCO查询情况一致。如前所述，[MASK]标记在评分中被认为有两个作用：（1）通过将文档词项与嵌入与非[MASK]查询标记相似的[MASK]标记进行匹配来实现查询词项加权；（2）通过[MASK]嵌入向查询之外的潜在相关标记偏移来实现查询扩展。

In this experiment, we test whether the clustering of [MASK] tokens around query tokens indicates that term weighting is the only role [MASK] tokens actually play in ColBERT scoring. To do this, we replace structural token em-beddings in a query with text token embeddings from the same query. This forces ColBERT to perform term weighting: replacing structural token embed-dings by their nearest text embeddings cannot introduce new terms or perform "soft weighting" by increasing the weight of multiple query tokens. We use the TREC 2019-2020 collection, and compare four token remapping conditions: (1) no remapping, (2) remapping [MASK] tokens to text tokens, (3) remapping all structural token embeddings ([CLS], [SEP], [Q], and [MASK]) to text tokens, and finally (4) remapping each [MASK] to its most similar embedded text token or non-[MASK] structural token (i.e. [CLS], [SEP], or [Q]). We hypothesize that replacing [MASK] embeddings by non-[MASK] embeddings in queries will reduce effectiveness by preventing matches with terms that do not appear in the query (i.e. by preventing query expansion).

在这个实验中，我们测试[MASK]标记围绕查询标记的聚类是否表明词项加权是[MASK]标记在ColBERT评分中实际发挥的唯一作用。为此，我们用同一查询中的文本标记嵌入替换查询中的结构标记嵌入。这迫使ColBERT进行词项加权：用最接近的文本嵌入替换结构标记嵌入不会引入新的词项，也不会通过增加多个查询标记的权重来进行“软加权”。我们使用TREC 2019 - 2020数据集，并比较四种标记重新映射条件：（1）不进行重新映射；（2）将[MASK]标记重新映射为文本标记；（3）将所有结构标记嵌入（[CLS]、[SEP]、[Q]和[MASK]）重新映射为文本标记；最后（4）将每个[MASK]重新映射为与其最相似的嵌入文本标记或非[MASK]结构标记（即[CLS]、[SEP]或[Q]）。我们假设在查询中用非[MASK]嵌入替换[MASK]嵌入会通过阻止与查询中未出现的词项匹配（即阻止查询扩展）来降低有效性。

RQ2: How sensitive are [CLS], [SEP], [Q], and [MASK] to query token order? As shown in the example above, ColBERT begins every query token sequence passed to BERT with the structural tokens [CLS] and [Q], followed by the text tokens and the structural token [SEP] marking the end of the text tokens, and finally zero or more [MASK] tokens to fill out the fixed-size input (e.g. 32 tokens). [CLS] is included in BERT's training objective function, and aggregates context across entire query and document passages resulting in a "summary" representation. We thus expect queries with similar wording and intent to produce similar [CLS] embeddings, even when the query word order changes. We expect the same pattern to hold for [SEP] which terminates every query and document passage. In contrast, we expect [MASK] embeddings to vary more than [CLS] and [SEP] tokens when words are re-ordered, because of their observed clustering around query terms and resulting weighting of individual terms in scoring. We expect [Q] embeddings to also vary more than [CLS] and [SEP], because [Q] is absent in the original BERT training objective.

研究问题2（RQ2）：[CLS]、[SEP]、[Q]和[MASK]对查询标记顺序的敏感程度如何？如上述示例所示，ColBERT将传递给BERT的每个查询标记序列都以结构标记[CLS]和[Q]开头，接着是文本标记，然后是标记文本标记结束的结构标记[SEP]，最后是零个或多个[MASK]标记以填充固定大小的输入（例如32个标记）。[CLS]包含在BERT的训练目标函数中，它会聚合整个查询和文档段落的上下文，从而得到一个“摘要”表示。因此，我们预计措辞和意图相似的查询即使查询词序发生变化，也会产生相似的[CLS]嵌入。我们预计对于终止每个查询和文档段落的[SEP]也会有相同的模式。相比之下，由于观察到[MASK]标记围绕查询词项聚类并在评分中对单个词项进行加权，我们预计当词序重新排列时，[MASK]嵌入的变化会比[CLS]和[SEP]标记更大。我们还预计[Q]嵌入的变化也会比[CLS]和[SEP]更大，因为[Q]在原始BERT训练目标中并不存在。

To study how query word order influences contextualization for query structural tokens, we reorder query text terms prior to contextualization similar to Rau et. al [8]. Randomly shuffling query tokens may alter the meaning of a query, so we limit the permutations considered. Specifically, we transform queries of the

为了研究查询词序如何影响查询结构标记的上下文化，我们在上下文化之前对查询文本词项进行重新排序，这与Rau等人 [8] 的方法类似。随机打乱查询标记可能会改变查询的含义，因此我们限制了所考虑的排列方式。具体来说，我们对……的查询进行转换

<!-- Media -->

Table 1. Replacing structural token embeddings with other query token embeddings (TREC 2019-2020, RQ1). Maximum values are in bold; significant differences from "None" are shown with a dagger $(p < {0.05}$ ,Bonferroni-corrected $t$ -tests).

表1. 用其他查询标记嵌入替换结构标记嵌入（TREC 2019 - 2020，RQ1）。最大值用粗体表示；与“无”有显著差异的用匕首符号 $(p < {0.05}$ 表示（经过Bonferroni校正的 $t$ 检验）。

<table><tr><td rowspan="2">METRIC</td><td colspan="4">Structural Token Remapping</td></tr><tr><td>None</td><td>All $\left\lbrack  \mathrm{X}\right\rbrack   \rightarrow$ Text</td><td>[MASK] $\rightarrow$ Text</td><td>[MASK] $\rightarrow$ Str. & Text</td></tr><tr><td colspan="5">Binary Relevance</td></tr><tr><td>MAP(rel $\geq  1$ )</td><td>0.447</td><td>0.454</td><td>†0.462</td><td>†0.462</td></tr><tr><td>MRR(rel $\geq  1$ )@10</td><td>0.930</td><td>0.924</td><td>0.929</td><td>0.923</td></tr><tr><td>MAP(rel $\geq  2$ )</td><td>0.450</td><td>0.444</td><td>0.454</td><td>0.457</td></tr><tr><td>MRR(rel $\geq  2$ )@10</td><td>0.851</td><td>0.820</td><td>0.835</td><td>0.837</td></tr><tr><td>MAP(rel $\geq  3$ )</td><td>0.366</td><td>0.362</td><td>0.373</td><td>0.372</td></tr><tr><td>MRR(rel $\geq  3$ )@10</td><td>0.557</td><td>0.560</td><td>0.563</td><td>0.563</td></tr><tr><td colspan="5">Graded Relevance</td></tr><tr><td>nDCG@10</td><td>0.689</td><td>0.685</td><td>0.691</td><td>0.694</td></tr><tr><td>nDCG@1000</td><td>0.680</td><td>0.673</td><td>0.683</td><td>0.684</td></tr></table>

<table><tbody><tr><td rowspan="2">指标</td><td colspan="4">结构Token重映射</td></tr><tr><td>无</td><td>所有 $\left\lbrack  \mathrm{X}\right\rbrack   \rightarrow$ 文本</td><td>[MASK] $\rightarrow$ 文本</td><td>[MASK] $\rightarrow$ 结构与文本</td></tr><tr><td colspan="5">二元相关性</td></tr><tr><td>平均准确率均值（rel $\geq  1$ ）</td><td>0.447</td><td>0.454</td><td>†0.462</td><td>†0.462</td></tr><tr><td>前10个结果的平均倒数排名（rel $\geq  1$ ）</td><td>0.930</td><td>0.924</td><td>0.929</td><td>0.923</td></tr><tr><td>平均准确率均值（rel $\geq  2$ ）</td><td>0.450</td><td>0.444</td><td>0.454</td><td>0.457</td></tr><tr><td>前10个结果的平均倒数排名（rel $\geq  2$ ）</td><td>0.851</td><td>0.820</td><td>0.835</td><td>0.837</td></tr><tr><td>平均准确率均值（rel $\geq  3$ ）</td><td>0.366</td><td>0.362</td><td>0.373</td><td>0.372</td></tr><tr><td>前10个结果的平均倒数排名（rel $\geq  3$ ）</td><td>0.557</td><td>0.560</td><td>0.563</td><td>0.563</td></tr><tr><td colspan="5">分级相关性</td></tr><tr><td>前10个结果的归一化折损累积增益</td><td>0.689</td><td>0.685</td><td>0.691</td><td>0.694</td></tr><tr><td>前1000个结果的归一化折损累积增益</td><td>0.680</td><td>0.673</td><td>0.683</td><td>0.684</td></tr></tbody></table>

<!-- Media -->

form "what is ..." into "... is what", moving the first two text tokens to the end of the query in the opposite order. To further avoid accidental semantic shifting, we only examine queries that are 3-8 tokens long. 12,513 queries in the MS MARCO dev set fit this criteria. As a baseline, we also apply the same reordering pattern to all queries 3-8 tokens long, without requiring the first two tokens to be "what is". 68,318 queries in the dev set fit this criteria. For the reasons given above, we hypothesize that [Q] and [MASK] embeddings will change more than [SEP] and [CLS] under this reordering. We use the cosine distance to quantify the shift in token embeddings after reordering the query text tokens.

将“what is ...”形式转换为“... is what”，把前两个文本标记按相反顺序移到查询的末尾。为进一步避免意外的语义偏移，我们仅考察长度为3 - 8个标记的查询。MS MARCO开发集中有12,513个查询符合这一标准。作为基线，我们还对所有长度为3 - 8个标记的查询应用相同的重新排序模式，而不要求前两个标记为“what is”。开发集中有68,318个查询符合这一标准。基于上述原因，我们假设在这种重新排序下，[Q]和[MASK]嵌入的变化将比[SEP]和[CLS]更大。我们使用余弦距离来量化查询文本标记重新排序后标记嵌入的偏移。

## 3 Results

## 3 结果

RQ1: Do [MASK] tokens perform more than just term weighting? In Table 1 we see replacing embeddings for all structural tokens with their closest text token embedding produces non-significant reductions in metrics other than MRR@10 (rel $\geq  3$ ). The two conditions mapping only [MASK] have very similar metrics, but surprisingly produce slight increases in MAP and nDCG@10/@1000 over both the "None" and "All" conditions. For MAP(rel $\geq  1$ ),the increase is significant (1.5%). MRR@10(rel $\geq  3$ ) is also slightly higher than standard ColBERT (but not significantly so). These small increases are likely from incorporating additional context through the [CLS], [Q], and [SEP] tokens (especially [CLS]). This contradicts our hypothesis that remapping [MASK] embeddings would harm performance, and is also interesting because [MASK] tokens comprise most of the input for short queries. In other words, its appears that strong retrieval performance with ColBERT is possible even when using only a few text token embeddings, provided that term weighting is taken into account.

RQ1：[MASK]标记的作用是否不仅仅是词项加权？在表1中，我们看到用最接近的文本标记嵌入替换所有结构标记的嵌入后，除MRR@10（相关性 $\geq  3$ ）外，其他指标的下降并不显著。仅映射[MASK]的两种情况的指标非常相似，但令人惊讶的是，与“无”和“全部”情况相比，MAP和nDCG@10/@1000略有增加。对于MAP（相关性 $\geq  1$ ），增加显著（1.5%）。MRR@10（相关性 $\geq  3$ ）也略高于标准ColBERT（但不显著）。这些微小的增加可能是通过[CLS]、[Q]和[SEP]标记（尤其是[CLS]）纳入了额外的上下文信息。这与我们的假设相矛盾，即重新映射[MASK]嵌入会损害性能，而且这也很有趣，因为[MASK]标记构成了短查询的大部分输入。换句话说，即使只使用少量文本标记嵌入，只要考虑词项加权，ColBERT也有可能实现强大的检索性能。

RQ2: How sensitive are [CLS], [SEP], [Q], and [MASK] to query token order? In Figure 2(a), [QUERY:3] is the third token and first text token (always "what") while [QUERY:5] is the fifth token containing the text token after "what is." We see distinct differences in how cosine distances are distributed for [CLS], [SEP], [QUERY:3], and [QUERY:5] versus [Q], [MASK:13], and [MASK:32]. The first group shows barely any shift, while the latter group shows large shifts, with higher variation. Figure 2(b) shows results for queries reordered similarly, but without requiring them to start with "what is". For example, "airplane flights to florida" produces the somewhat unnatural query "to florida flights airplane". All tokens show larger representational shifts in this condition; however, we again find that [CLS], [SEP], and the [QUERY:3/5] text token embeddings vary far less than the [Q] and [MASK] embeddings.

RQ2：[CLS]、[SEP]、[Q]和[MASK]对查询标记顺序的敏感程度如何？在图2(a)中，[QUERY:3]是第三个标记和第一个文本标记（始终为“what”），而[QUERY:5]是包含“what is”之后文本标记的第五个标记。我们看到[CLS]、[SEP]、[QUERY:3]和[QUERY:5]与[Q]、[MASK:13]和[MASK:32]的余弦距离分布存在明显差异。第一组几乎没有变化，而第二组变化较大，且变化幅度更大。图2(b)显示了类似重新排序的查询结果，但不要求查询以“what is”开头。例如，“airplane flights to florida”会生成一个有点不自然的查询“to florida flights airplane”。在这种情况下，所有标记的表示变化都更大；然而，我们再次发现[CLS]、[SEP]和[QUERY:3/5]文本标记嵌入的变化远小于[Q]和[MASK]嵌入。

<!-- Media -->

<!-- figureText: 1.0 1.2 1.0 0.8 0.2 0.0 CLS Q QUERY:3 QUERY:5 SEP MASK:13 MASK:32 (b) All queries: " ${12}\ldots$ " $\rightarrow$ " $\ldots {21}$ " 0.8 0.6 0.4 0.2 0.0 CLS QUERY 3 QUERY:5 SEP (a) "what is ..." $\rightarrow$ "... is what" -->

<img src="https://cdn.noedgeai.com/0195b3f5-2270-7ac8-ad46-761ec932728e_5.jpg?x=388&y=337&w=1033&h=441&r=0"/>

Fig. 2. Distribution of cosine distance $\left( {1 - \cos \left( {\mathbf{e},{\mathbf{e}}^{\prime }}\right) }\right)$ for token embeddings before and after query token reordering (MS MARCO, RQ2). For brevity not all tokens are shown, but the general trend of higher variance holds for all [MASK] tokens. Left: Cosine distances for queries starting with "what is". Right: Cosine distances without requiring queries to start with "what is". [QUERY:3] and [QUERY:5] are the first and third text tokens, respectively; [MASK:13] represents the [MASK] token at position 13, and [MASK:32] represents the final [MASK] input token at position 32.

图2. 查询标记重新排序前后标记嵌入的余弦距离 $\left( {1 - \cos \left( {\mathbf{e},{\mathbf{e}}^{\prime }}\right) }\right)$ 分布（MS MARCO，RQ2）。为简洁起见，并非所有标记都显示出来，但所有[MASK]标记的高方差总体趋势是一致的。左图：以“what is”开头的查询的余弦距离。右图：不要求查询以“what is”开头的余弦距离。[QUERY:3]和[QUERY:5]分别是第一个和第三个文本标记；[MASK:13]表示位置13处的[MASK]标记，[MASK:32]表示位置32处的最后一个[MASK]输入标记。

<!-- Media -->

Despite our efforts to avoid it, some "what is" queries have their meaning altered by our reordering. For example, "what is some examples homogeneous" becomes "some examples homogeneous is what", which may change the query from a request for examples to asking for a definition. When we filtered out queries containing the word "example", the variance of [QUERY:3] dropped from ${8.53} \cdot  {10}^{-4}$ to ${7.73} \cdot  {10}^{-4}$ ,while the variance of [Q] had less of a proportional drop $\left( {{2.07} \cdot  {10}^{-2}\text{to}{2.06} \cdot  {10}^{-2}}\right)$ ,indicating some of the variance of non- $\left\lbrack  \mathrm{Q}\right\rbrack$ or [MASK] embeddings may be due to these edge cases.

尽管我们努力避免，但一些“what is”查询的含义因我们的重新排序而改变。例如，“what is some examples homogeneous”变成了“some examples homogeneous is what”，这可能会使查询从请求示例变为请求定义。当我们过滤掉包含“example”一词的查询时，[QUERY:3]的方差从 ${8.53} \cdot  {10}^{-4}$ 降至 ${7.73} \cdot  {10}^{-4}$ ，而[Q]的方差下降比例较小 $\left( {{2.07} \cdot  {10}^{-2}\text{to}{2.06} \cdot  {10}^{-2}}\right)$ ，这表明非 $\left\lbrack  \mathrm{Q}\right\rbrack$ 或[MASK]嵌入的部分方差可能是由于这些边缘情况造成的。

## 4 Discussion and Conclusion

## 4 讨论与结论

To our surprise, replacing [MASK] token embeddings in queries with either their most similar text token embedding in the same query, or the most similar text or non- $\left\lbrack  \mathrm{{MASK}}\right\rbrack$ structural token embedding from the query yielded similar effectiveness to standard ColBERT for the TREC 2019-2020 dataset. There is even a small significant increase in MAP when weakly-relevant documents are considered relevant (i.e. MAP(rel $\geq  1$ )). The differences between mapping [MASK] to only text tokens or to both text and non-MASK structural tokens was statistically insigificant for all metrics observed. So if [MASK] tokens have effects other than term weighting in scoring, their role appears to be minor (RQ1).

令我们惊讶的是，对于TREC 2019 - 2020数据集，将查询中的[MASK]标记嵌入替换为同一查询中与其最相似的文本标记嵌入，或者替换为查询中最相似的文本或非$\left\lbrack  \mathrm{{MASK}}\right\rbrack$结构标记嵌入，其效果与标准ColBERT相似。当将弱相关文档视为相关文档时（即MAP(rel $\geq  1$ )），平均准确率均值（MAP）甚至有小幅显著提升。在所有观察到的指标中，将[MASK]仅映射到文本标记与映射到文本和非[MASK]结构标记之间的差异在统计上不显著。因此，如果[MASK]标记在评分中的作用不仅仅是词项加权，那么它们的作用似乎很小（研究问题1）。

This suggests a possible optimization. We can multiply each non-[MASK] query token embedding's score contribution by the number of [MASK] token em-beddings most similar to it. Regarding interpretability, using [MASK] only to weights tokens in this manner simplifies the ColBERT scoring model both conceptually and computationally. For short queries, most of the input to ColBERT is [MASK] tokens, and so the number of query tokens to match against document tokens with MaxSim may be a fraction of the full token input size. A related approach is described by Tonellotto et al. [10], where query embeddings are dropped after contextualization based on frequency statistics. However, rather than pruning a set number of tokens based on collection frequency, we would use all token embeddings to retrieve candidates, and then weight non-[MASK] query tokens using fewer nearest neighbor lookups during scoring.

这表明了一种可能的优化方法。我们可以将每个非[MASK]查询标记嵌入的得分贡献乘以与其最相似的[MASK]标记嵌入的数量。在可解释性方面，以这种方式仅使用[MASK]对标记进行加权，从概念和计算上简化了ColBERT评分模型。对于短查询，ColBERT的大部分输入是[MASK]标记，因此使用最大相似度（MaxSim）与文档标记进行匹配的查询标记数量可能只是完整标记输入大小的一部分。托内洛托（Tonellotto）等人[10]描述了一种相关方法，即根据频率统计在上下文处理后丢弃查询嵌入。然而，我们不是根据集合频率修剪一定数量的标记，而是使用所有标记嵌入来检索候选文档，然后在评分时使用较少的最近邻查找对非[MASK]查询标记进行加权。

However, the question of [MASK] tokens' role in retrieving candidates still remains, as this paper has focused on the final scoring step; all query tokens were used to retrieve candidates in our experiments. How might limiting or removing the use of $\left\lbrack  \mathrm{{MASK}}\right\rbrack$ tokens in the first-stage dense retrieval impact performance? We wonder about the small statistically insignificant improvements seen in MAP and MRR for highly relevant documents $\left( {\mathrm{{rel}} \geq  3}\right)$ ,as well as nDCG@10, nDCG@1000, and MAP. Are these stable and/or significant in other collections? To better understand [MASK], one possible experiment is appending different numbers of [MASK] tokens to each query. This may reveal whether having fewer [MASK] tokens causes them to move closer to non-[MASK] embeddings, and whether having more [MASK] tokens might improve term weighting.

然而，[MASK]标记在检索候选文档中的作用问题仍然存在，因为本文主要关注最终评分步骤；在我们的实验中，所有查询标记都用于检索候选文档。在第一阶段的密集检索中限制或取消使用$\left\lbrack  \mathrm{{MASK}}\right\rbrack$标记会对性能产生怎样的影响？我们想知道，对于高度相关文档$\left( {\mathrm{{rel}} \geq  3}\right)$，在平均准确率均值（MAP）和平均倒数排名（MRR）以及归一化折损累积增益（nDCG）@10、nDCG@1000和MAP方面观察到的微小且在统计上不显著的改进情况。这些改进在其他数据集中是否稳定和/或显著？为了更好地理解[MASK]，一个可能的实验是向每个查询追加不同数量的[MASK]标记。这可能揭示较少的[MASK]标记是否会使它们更接近非[MASK]嵌入，以及更多的[MASK]标记是否可能改善词项加权。

Regarding the effect of token ordering on contextualized embeddings (RQ2), our findings are consistent with our original hypothesis: [CLS] and [SEP] em-beddings do not vary greatly for similar queries with a different token ordering, while [Q] and [MASK] do. The shift in [Q] is the most interesting result here; the model may be treating [Q] similar to another [MASK] token. Some early analysis suggests that a query [CLS] tends to match a document [CLS], a query [SEP] tends to match ending punctuation, and [MASK] tends to match tokens other than [CLS] or [SEP] (see our interactive visualization for ColBERT scoring ${}^{2}$ ). We have not observed [Q] matching to any specific document tokens.

关于标记顺序对上下文嵌入的影响（研究问题2），我们的发现与我们最初的假设一致：对于标记顺序不同但相似的查询，[CLS]和[SEP]嵌入变化不大，而[Q]和[MASK]嵌入则会变化。这里最有趣的结果是[Q]的变化；模型可能将[Q]视为另一个[MASK]标记。一些早期分析表明，查询中的[CLS]倾向于与文档中的[CLS]匹配，查询中的[SEP]倾向于与结尾标点匹配，而[MASK]倾向于与[CLS]或[SEP]之外的标记匹配（见我们关于ColBERT评分的交互式可视化${}^{2}$）。我们尚未观察到[Q]与任何特定文档标记匹配。

In the future we would also like to validate our results using ColBERT v2. We believe that our results should hold for the newer model - if [MASK] s continue to cluster around query word embeddings, we expect [MASK] s will continue to act as term weights, and the training process in ColBERT v2 should not alter how [Q] is processed. We would also like to extend our evaluation to additional datasets, since we have only focused on MS MARCO and the MS MARCO-derived TREC 2019-2020 datasets in our experiments.

未来，我们还希望使用ColBERT v2验证我们的结果。我们相信我们的结果对于新模型也应该成立——如果[MASK]继续围绕查询词嵌入聚类，我们预计[MASK]将继续作为词项权重，并且ColBERT v2的训练过程不应改变[Q]的处理方式。我们还希望将评估扩展到其他数据集，因为在我们的实验中，我们仅关注了MS MARCO和源自MS MARCO的TREC 2019 - 2020数据集。

## References

## 参考文献

1. Craswell, N., Mitra, B., Yilmaz, E., Campos, D.: Overview of the TREC 2020 deep learning track. In: Voorhees, E.M., Ellis, A. (eds.) Proc. Text REtrieval Conference (TREC). NIST Special Publication, vol. 1266 (2020)

1. 克拉斯韦尔（Craswell）, N., 米特拉（Mitra）, B., 伊尔马兹（Yilmaz）, E., 坎波斯（Campos）, D.: TREC 2020深度学习赛道概述。见: 沃里斯（Voorhees）, E.M., 埃利斯（Ellis）, A. (编) 文本检索会议（TREC）论文集。美国国家标准与技术研究院（NIST）特别出版物, 第1266卷 (2020)

2. Devlin, J., Chang, M.W., Lee, K., Toutanova, K.: BERT: Pre-training of deep bidirectional transformers for language understanding. In: Proc. North American Chapter of the Association for Computational Linguistics (NAACL). pp. 4171- 4186 (2019). https://doi.org/10.18653/v1/N19-1423, https://aclanthology.org/ N19-1423

2. 德夫林（Devlin）, J., 张（Chang）, M.W., 李（Lee）, K., 图托纳娃（Toutanova）, K.: BERT：用于语言理解的深度双向变换器预训练。见: 北美计算语言学协会（NAACL）会议论文集。第4171 - 4186页 (2019)。https://doi.org/10.18653/v1/N19 - 1423, https://aclanthology.org/ N19 - 1423

3. Formal, T., Piwowarski, B., Clinchant, S.: A white box analysis of ColBERT. In: Hiemstra, D., Moens, M.F., Mothe, J., Perego, R., Potthast, M., Sebastiani, F. (eds.) Proc. European Conference on Information Retrieval (ECIR). LNCS, vol. 12657, pp. 257-263. Springer (2021)

3. 福尔马尔（Formal, T.）、皮沃瓦尔斯基（Piwowarski, B.）、克林尚（Clinchant, S.）：对ColBERT的白盒分析。见：希姆斯特拉（Hiemstra, D.）、莫恩斯（Moens, M.F.）、莫特（Mothe, J.）、佩雷戈（Perego, R.）、波塔斯塔（Potthast, M.）、塞巴斯蒂亚尼（Sebastiani, F.） 编，《欧洲信息检索会议（ECIR）论文集》。《计算机科学讲义》（LNCS），第12657卷，第257 - 263页。施普林格出版社（2021年）

4. Khattab, O., Zaharia, M.: ColBERT: Efficient and effective passage search via contextualized late interaction over BERT. In: Huang, J.X., Chang, Y., Cheng, X., Kamps, J., Murdock, V., Wen, J., Liu, Y. (eds.) Proc. SIGIR. pp. 39-48 (2020). https://doi.org/10.1145/3397271.3401075

4. 哈塔布（Khattab, O.）、扎哈里亚（Zaharia, M.）：ColBERT：通过基于BERT的上下文延迟交互实现高效有效的段落搜索。见：黄（Huang, J.X.）、张（Chang, Y.）、程（Cheng, X.）、坎普斯（Kamps, J.）、默多克（Murdock, V.）、文（Wen, J.）、刘（Liu, Y.） 编，《信息检索研究与发展会议（SIGIR）论文集》。第39 - 48页（2020年）。https://doi.org/10.1145/3397271.3401075

5. MacAvaney, S., Feldman, S., Goharian, N., Downey, D., Cohan, A.: AB-NIRML: Analyzing the Behavior of Neural IR Models. Transactions of the Association for Computational Linguistics $\mathbf{{10}},{224} - {239}\left( {032022}\right)$ . https://doi.org/10.1162/tacl_a_00457, https://doi.org/10.1162/tacl\\_a\\ _00457

5. 麦卡瓦尼（MacAvaney, S.）、费尔德曼（Feldman, S.）、戈哈里安（Goharian, N.）、唐尼（Downey, D.）、科汉（Cohan, A.）：AB - NIRML：分析神经信息检索模型的行为。《计算语言学协会汇刊》$\mathbf{{10}},{224} - {239}\left( {032022}\right)$ 。https://doi.org/10.1162/tacl_a_00457，https://doi.org/10.1162/tacl\\_a\\ _00457

6. Macdonald, C., Tonellotto, N., MacAvaney, S., Ounis, I.: PyTerrier: Declarative experimentation in Python from BM25 to dense retrieval. In: Proc. Intl. Conf. Information & Knowledge Management (CIKM). pp. 4526-4533 (2021). https://doi.org/10.1145/3459637.3482013

6. 麦克唐纳（Macdonald, C.）、托内洛托（Tonellotto, N.）、麦卡瓦尼（MacAvaney, S.）、乌尼斯（Ounis, I.）：PyTerrier：从BM25到密集检索的Python声明式实验。见：《国际信息与知识管理会议（CIKM）论文集》。第4526 - 4533页（2021年）。https://doi.org/10.1145/3459637.3482013

7. Nguyen, T., Rosenberg, M., Song, X., Gao, J., Tiwary, S., Majumder, R., Deng, L.: MS MARCO: A human generated machine reading comprehension dataset. In: Besold, T.R., Bordes, A., d'Avila Garcez, A.S., Wayne, G. (eds.) Proceedings of the Workshop on Cognitive Computation: Integrating neural and symbolic approaches 2016 co-located with the 30th Annual Conference on Neural Information Processing Systems (NIPS 2016), Barcelona, Spain, December 9, 2016. CEUR Workshop Proceedings, vol. 1773. CEUR-WS.org (2016), https: //ceur-ws.org/Vol-1773/CoCoNIPS\\_2016\\_paper9.pdf

7. 阮（Nguyen, T.）、罗森伯格（Rosenberg, M.）、宋（Song, X.）、高（Gao, J.）、蒂瓦里（Tiwary, S.）、马朱姆德（Majumder, R.）、邓（Deng, L.）：MS MARCO：一个人工生成的机器阅读理解数据集。见：贝索尔德（Besold, T.R.）、博尔德斯（Bordes, A.）、达维拉·加尔塞斯（d'Avila Garcez, A.S.）、韦恩（Wayne, G.） 编，《认知计算研讨会论文集：整合神经与符号方法（2016年）》，与第30届神经信息处理系统年度会议（NIPS 2016）同期举办，西班牙巴塞罗那，2016年12月9日。CEUR研讨会论文集，第1773卷。CEUR - WS.org（2016年），https: //ceur - ws.org/Vol - 1773/CoCoNIPS\\_2016\\_paper9.pdf

8. Rau, D., Kamps, J.: The role of complex NLP in transformers for text ranking. In: Proc. ICTIR. pp. 153-160 (2022). https://doi.org/10.1145/3539813.3545144, http://dx.doi.org/10.1145/3539813.3545144

8. 劳（Rau, D.）、坎普斯（Kamps, J.）：复杂自然语言处理在文本排序Transformer模型中的作用。见：《信息检索理论会议（ICTIR）论文集》。第153 - 160页（2022年）。https://doi.org/10.1145/3539813.3545144，http://dx.doi.org/10.1145/3539813.3545144

9. Santhanam, K., Khattab, O., Saad-Falcon, J., Potts, C., Zaharia, M.: ColBERTv2: Effective and efficient retrieval via lightweight late interaction. In: Carpuat, M., de Marneffe, M.C., Meza Ruiz, I.V. (eds.) Proc. North American Chapter of the Association for Computational Linguistics (NAACL). pp. 3715-3734 (2022). https://doi.org/10.18653/v1/2022.naacl-main.272

9. 桑塔南（Santhanam, K.）、哈塔布（Khattab, O.）、萨德 - 法尔孔（Saad - Falcon, J.）、波茨（Potts, C.）、扎哈里亚（Zaharia, M.）：ColBERTv2：通过轻量级延迟交互实现高效有效的检索。见：卡尔普瓦特（Carpuat, M.）、德马尔内夫（de Marneffe, M.C.）、梅萨·鲁伊斯（Meza Ruiz, I.V.） 编，《计算语言学协会北美分会会议（NAACL）论文集》。第3715 - 3734页（2022年）。https://doi.org/10.18653/v1/2022.naacl - main.272

10. Tonellotto, N., Macdonald, C.: Query embedding pruning for dense retrieval. In: Proc. Intl. Conf. Information & Knowledge Management (CIKM). pp. 3453-3457 (2021). https://doi.org/10.1145/3459637.3482162, https://doi.org/ 10.1145/3459637.3482162

10. 托内洛托（Tonellotto, N.）、麦克唐纳（Macdonald, C.）：密集检索的查询嵌入剪枝。见：《国际信息与知识管理会议（CIKM）论文集》。第3453 - 3457页（2021年）。https://doi.org/10.1145/3459637.3482162，https://doi.org/ 10.1145/3459637.3482162

11. Voorhees, E.M., Ellis, A. (eds.): Proceedings of the Twenty-Eighth Text REtrieval Conference, TREC 2019, Gaithersburg, Maryland, USA, November 13-15, 2019, NIST Special Publication, vol. 1250. National Institute of Standards and Technology (NIST) (2019), https://trec.nist.gov/pubs/trec28/trec2019.html

11. 沃里斯（Voorhees, E.M.）、埃利斯（Ellis, A.） 编：《第28届文本检索会议（TREC 2019）论文集》，美国马里兰州盖瑟斯堡，2019年11月13 - 15日，美国国家标准与技术研究院（NIST）特别出版物，第1250卷。美国国家标准与技术研究院（NIST）（2019年），https://trec.nist.gov/pubs/trec28/trec2019.html

12. Wang, X., Macdonald, C., Ounis, I.: Improving zero-shot retrieval using dense external expansion. Information Processing Management $\mathbf{{59}}\left( 5\right) ,{103026}$ (2022). https://doi.org/https://doi.org/10.1016/j.ipm.2022.103026, https://www.sciencedirect.com/science/article/pii/S0306457322001364

12. 王，X.，麦克唐纳，C.，乌尼斯，I.：利用密集外部扩展改进零样本检索。《信息处理与管理》$\mathbf{{59}}\left( 5\right) ,{103026}$ (2022)。https://doi.org/https://doi.org/10.1016/j.ipm.2022.103026，https://www.sciencedirect.com/science/article/pii/S0306457322001364

13. Wang, X., MacDonald, C., Tonellotto, N., Ounis, I.: ColBERT-PRF: Semantic pseudo-relevance feedback for dense passage and document retrieval. ACM Trans. Web 17(1) (2023). https://doi.org/10.1145/3572405, https://doi.org/10.1145/ 3572405

13. 王，X.，麦克唐纳，C.，托内洛托，N.，乌尼斯，I.：ColBERT - PRF：用于密集段落和文档检索的语义伪相关反馈。《ACM 网络汇刊》17(1) (2023)。https://doi.org/10.1145/3572405，https://doi.org/10.1145/ 3572405

14. Wang, X., Macdonald, C., Tonellotto, N., Ounis, I.: Reproducibility, repli-cability, and insights into dense multi-representation retrieval models: From ColBERT to Col*. In: Proc. SIGIR. pp. 2552-2561. ACM (2023). https://doi.org/10.1145/3539618.3591916

14. 王，X.，麦克唐纳，C.，托内洛托，N.，乌尼斯，I.：密集多表示检索模型的可重复性、可复现性及见解：从 ColBERT 到 Col*。见：《SIGIR 会议论文集》。第 2552 - 2561 页。ACM (2023)。https://doi.org/10.1145/3539618.3591916

15. Xiong, L., Xiong, C., Li, Y., Tang, K., Liu, J., Bennett, P.N., Ahmed, J., Over-wijk, A.: Approximate nearest neighbor negative contrastive learning for dense text retrieval. In: Proc. Int. Conf. Learning Representations (ICLR). OpenReview.net (2021), https://openreview.net/forum?id=zeFrfgyZln

15. 熊，L.，熊，C.，李，Y.，唐，K.，刘，J.，贝内特，P.N.，艾哈迈德，J.，奥弗维克，A.：用于密集文本检索的近似最近邻负对比学习。见：《国际学习表征会议（ICLR）论文集》。OpenReview.net (2021)，https://openreview.net/forum?id=zeFrfgyZln

16. Yao, L., Huang, R., Hou, L., Lu, G., Niu, M., Xu, H., Liang, X., Li, Z., Jiang, X., Xu, C.: FILIP: fine-grained interactive language-image pre-training. In: The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net (2022), https://openreview.net/forum?id= cpDhcsEDC2

16. 姚，L.，黄，R.，侯，L.，陆，G.，牛，M.，徐，H.，梁，X.，李，Z.，江，X.，徐，C.：FILIP：细粒度交互式语言 - 图像预训练。见：第十届国际学习表征会议，ICLR 2022，虚拟会议，2022 年 4 月 25 - 29 日。OpenReview.net (2022)，https://openreview.net/forum?id= cpDhcsEDC2

17. Zhan, J., Mao, J., Liu, Y., Zhang, M., Ma, S.: RepBERT: Contextualized text em-beddings for first-stage retrieval. CoRR abs/2006.15498 (2020), https://arxiv.org/abs/2006.15498

17. 詹，J.，毛，J.，刘，Y.，张，M.，马，S.：RepBERT：用于第一阶段检索的上下文文本嵌入。CoRR 预印本 abs/2006.15498 (2020)，https://arxiv.org/abs/2006.15498