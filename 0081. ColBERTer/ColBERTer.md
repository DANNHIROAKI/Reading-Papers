# Introducing Neural Bag of Whole-Words with ColBERTer: Contextualized Late Interactions using Enhanced Reduction

# 借助ColBERTer引入全词神经词袋：使用增强约简的上下文晚期交互

Sebastian

塞巴斯蒂安

Hofstätter

霍夫施泰特

TU Wien

维也纳工业大学

Omar Khattab

奥马尔·哈塔卜

Stanford University

斯坦福大学

Sophia

索菲娅

Althammer

阿尔塔默

TU Wien

维也纳工业大学

Mete Sertkan

梅特·塞尔特坎

TU Wien

维也纳工业大学

Allan Hanbury

艾伦·汉伯里

TU Wien

维也纳工业大学

## ABSTRACT

## 摘要

Recent progress in neural information retrieval has demonstrated large gains in quality, while often sacrificing efficiency and interpretability compared to classical approaches. We propose ColBERTer, a neural retrieval model using contextualized late interaction (ColBERT) with enhanced reduction. Along the effectiveness Pareto frontier, ColBERTer dramatically lowers ColBERT's storage requirements while simultaneously improving the interpretability of its token-matching scores. To this end, ColBERTer fuses single-vector retrieval, multi-vector refinement, and optional lexical matching components into one model. For its multi-vector component, ColBERTer reduces the number of stored vectors by learning unique whole-word representations and learning to identify and remove word representations that are not essential to effective scoring. We employ an explicit multi-task, multi-stage training to facilitate using very small vector dimensions. Results on the MS MARCO and TREC-DL collection show that ColBERTer reduces the storage footprint by up to ${2.5} \times$ ,while maintaining effectiveness. With just one dimension per token in its smallest setting, ColBERTer achieves index storage parity with the plaintext size, with very strong effectiveness results. Finally, we demonstrate ColBERTer's robustness on seven high-quality out-of-domain collections, yielding statistically significant gains over traditional retrieval baselines.

神经信息检索领域的近期进展表明，与传统方法相比，其在质量上有了显著提升，但往往牺牲了效率和可解释性。我们提出了ColBERTer，这是一种使用增强约简的上下文晚期交互（ColBERT）的神经检索模型。在有效性帕累托前沿上，ColBERTer显著降低了ColBERT的存储需求，同时提高了其词元匹配分数的可解释性。为此，ColBERTer将单向量检索、多向量细化和可选的词法匹配组件融合到一个模型中。对于其多向量组件，ColBERTer通过学习独特的全词表示并识别和去除对有效评分并非必需的词表示，减少了存储向量的数量。我们采用显式的多任务、多阶段训练，以促进使用非常小的向量维度。在MS MARCO和TREC - DL数据集上的实验结果表明，ColBERTer在保持有效性的同时，最多可将存储占用空间减少${2.5} \times$。在其最小设置下，每个词元仅使用一个维度，ColBERTer实现了索引存储与纯文本大小相当，且有效性结果非常出色。最后，我们在七个高质量的域外数据集上证明了ColBERTer的鲁棒性，相较于传统检索基线取得了具有统计学意义的提升。

## CCS CONCEPTS

## CCS概念

- Information systems $\rightarrow$ Learning to rank;

- 信息系统 $\rightarrow$ 学习排序;

## KEYWORDS

## 关键词

Neural Ranking; Dense-Sparse Hybrid Retrieval

神经排序；稠密 - 稀疏混合检索

## ACM Reference Format:

## ACM引用格式：

Sebastian Hofstätter, Omar Khattab, Sophia Althammer, Mete Sertkan, and Allan Hanbury. 2022. Introducing Neural Bag of Whole-Words with ColBERTer: Contextualized Late Interactions using Enhanced Reduction. In Proceedings of the 31st ACM Int'l Conference on Information and Knowledge Management (CIKM '22), Oct. 17-21, 2022, Atlanta, GA, USA. ACM, New York, NY, USA, 11 pages. https://doi.org/10.1145/3511808.3557367

塞巴斯蒂安·霍夫施泰特（Sebastian Hofstätter）、奥马尔·哈塔布（Omar Khattab）、索菲亚·阿尔塔默（Sophia Althammer）、梅特·塞尔特坎（Mete Sertkan）和艾伦·汉伯里（Allan Hanbury）。2022年。通过ColBERTer引入全词神经词袋：使用增强约简的上下文后期交互。收录于第31届ACM国际信息与知识管理会议论文集（CIKM '22），2022年10月17 - 21日，美国佐治亚州亚特兰大市。美国纽约州纽约市ACM，11页。https://doi.org/10.1145/3511808.3557367

## 1 INTRODUCTION

## 1 引言

Traditional retrieval systems have long relied on bag-of-words representations to search text collections. This has led to mature architectures,in which compact inverted indexes enable fast top- $k$

传统的检索系统长期以来一直依赖词袋表示法来搜索文本集合。这催生了成熟的架构，其中紧凑的倒排索引能够实现快速的前$k$

Q does doxycycline contain sulfa BERT tokenized (9 subword-tokens): 'does', 'do', '##xy', '##cy', '##cl', '##ine', 'contain', 'sul', '##fa'

问题：强力霉素是否含有磺胺 BERT分词（9个子词标记）：“does”、“do”、“##xy”、“##cy”、“##cl”、“##ine”、“contain”、“sul”、“##fa”

<!-- Media -->

ColBERTer BOW ${}^{2}$ (30 saved vectors from 84 subword-tokens): photosensitivity doxycycline 12.9 sulfa 14.2 sunburned rash clothing sunlight allergic compound drugs containing 6.6 take safely wear . is no 4.7 exposed ..

ColBERTer词袋 ${}^{2}$ （从84个子词标记中保存的30个向量）：光敏性 强力霉素 12.9 磺胺 14.2 晒伤 皮疹 衣物 阳光 过敏 化合物 含有的药物 6.6 安全服用 穿着 。 没有 4.7 暴露于 。。

Fulltext: No doxycycline is not a sulfa containing compound, so you may take it safely if you are allergic to sulfa drugs. You should be aware, however, that doxycycline may cause photosensitivity, so you should wear appropriate clothing, or you may get easily sunburned or develop a rash if you are exposed to sunlight.

全文：不，强力霉素不是含磺胺的化合物，所以如果你对磺胺类药物过敏，可以安全服用。不过，你应该知道，强力霉素可能会导致光敏性，所以你应该穿合适的衣服，否则如果暴露在阳光下，你很容易晒伤或出皮疹。

<!-- Media -->

Figure 1: Example of ColBERTer’s BOW ${}^{2}$ (Bag Of Whole-Words): ColBERTer stores and matches unique whole-word representations. The words in ${\mathrm{{BOW}}}^{2}$ are ordered by implicitly learned query-independent term importance. Matched words are highlighted in blue with whole-word scores displayed in a user-friendly way next to them.

图1：ColBERTer的全词词袋（Bag Of Whole - Words）示例 ${}^{2}$ ：ColBERTer存储并匹配唯一的全词表示。${\mathrm{{BOW}}}^{2}$ 中的单词按隐式学习到的与查询无关的词重要性排序。匹配的单词以蓝色突出显示，并在旁边以用户友好的方式显示全词得分。

retrieval strategies, while also exhibiting interpretable behavior, where retrieval scores can directly be attributed to contributions from individual terms. Despite these qualities, recent progress in Information Retrieval (IR) has firmly demonstrated that pre-trained language models can considerably boost effectiveness over classical approaches. This progress has raises questions about how to control the computational cost and how to ensure interpretability of these neural models. This has sparked an unprecedented tension in IR between achieving the best retrieval quality, maintaining low computational costs, and prioritizing interpretable modeling.

检索策略，同时还表现出可解释的行为，其中检索得分可以直接归因于单个词的贡献。尽管有这些优点，但信息检索（IR）领域的最新进展已经充分证明，预训练语言模型相比传统方法可以显著提高检索效果。这一进展引发了关于如何控制计算成本以及如何确保这些神经模型可解释性的问题。这在信息检索领域引发了一场前所未有的紧张局面，即在实现最佳检索质量、保持低计算成本和优先考虑可解释建模之间进行权衡。

For practical applications, IR architectures are confined to strict cost constraints around query latency and space footprint. While disk space might be affordable, keeping large pre-computed representations in memory-as often needed for low query latency-increases hardware costs considerably. For multi-vector models like ColBERT [22], space consumption is determined by a multiplication of three variables: 1) the number of vectors per document; 2) the number of dimensions per vector; 3) the number of bytes per dimension. This work is motivated by the observation that reducing any of these three variables directly reduces the storage requirement proportionally and yet different choices carry different impact on effectiveness. Well-studied low hanging fruits for good tradeoffs include reducing the number of dimensions and reducing the number of bytes with quantization $\left\lbrack  {{12},{19},{25},{34}}\right\rbrack$ . Reducing the number of vectors offers a rich design space around model architecture and retrieval strategy.

对于实际应用，信息检索架构受到查询延迟和空间占用方面严格成本约束的限制。虽然磁盘空间成本可能可以承受，但为了实现低查询延迟，通常需要将大量预计算的表示存储在内存中，这会显著增加硬件成本。对于像ColBERT [22] 这样的多向量模型，空间消耗由三个变量的乘积决定：1）每个文档的向量数量；2）每个向量的维度数量；3）每个维度的字节数。这项工作的动机在于观察到，减少这三个变量中的任何一个都能直接按比例降低存储需求，而且不同的选择对检索效果有不同的影响。经过充分研究的、能实现良好权衡的简单方法包括减少维度数量和通过量化 $\left\lbrack  {{12},{19},{25},{34}}\right\rbrack$ 减少字节数。减少向量数量为模型架构和检索策略提供了丰富的设计空间。

Besides efficiency, the accelerating adoption of machine learning coincides with indications that future regulatory environments will require deployed models to provide transparent and reliably interpretable output to their users. ${}^{1}$ This need for interpretability is especially pronounced in IR, where the ranking models are demanded to be fair and transparent [6]. Despite this, the two largest classes of neural models at the moment-namely, cross-encoders and single-vector bi-encoders-rely on opaque aggregations that conceal the contributions of query and document terms on retrieval scores.

除了效率问题，机器学习的加速应用也伴随着这样一种迹象，即未来的监管环境将要求部署的模型向用户提供透明且可靠可解释的输出。 ${}^{1}$ 这种可解释性需求在信息检索领域尤为明显，因为在该领域，排序模型被要求做到公平和透明 [6] 。尽管如此，目前最大的两类神经模型，即交叉编码器和单向量双编码器，都依赖于不透明的聚合方式，这种方式掩盖了查询词和文档词对检索得分的贡献。

This paper presents a novel end-to-end retrieval model called ColBERTer. ColBERTer extends the popular ColBERT model with effective enhanced reduction approaches. These reductions increase the level of interpretability and reduce the storage and latency cost greatly, while maintaining the quality of retrieval.

本文提出了一种名为ColBERTer的新型端到端检索模型。ColBERTer通过有效的增强约简方法扩展了流行的ColBERT模型。这些约简方法提高了可解释性水平，大幅降低了存储和延迟成本，同时保持了检索质量。

ColBERTer fuses a single-vector retrieval and multi-vector refinement model into one with explicit multi-task training. Next, ColBERTer introduces neural Bag of Whole-Words (BOW ${}^{2}$ ) representations for increasing interpretability and reducing the number of stored vectors in the ranking process. The ${\mathrm{{BOW}}}^{2}$ consist of the aggregation of all subword token representations contained in a unique whole word. To further reduce the number of vectors, ColBERTer learns to remove ${\mathrm{{BOW}}}^{2}$ representations with simplified contextualized stopwords (CS) [17]. And to reduce the dimensionality of the token vectors down to one, our methods employ an Exact Matching (EM) component that aligns representations across only lexical matches from the query and document, a model variant we call Uni-ColBERTer following the nomenclature of Lin and Ma [26].

ColBERTer通过显式的多任务训练，将单向量检索和多向量细化模型融合为一体。接下来，ColBERTer引入了神经全词词袋（BOW ${}^{2}$ ）表示法，以提高可解释性并减少排序过程中存储的向量数量。${\mathrm{{BOW}}}^{2}$ 由唯一全词中包含的所有子词标记表示的聚合组成。为了进一步减少向量数量，ColBERTer学习使用简化的上下文停用词（CS） [17] 去除 ${\mathrm{{BOW}}}^{2}$ 表示。为了将标记向量的维度降至一维，我们的方法采用了精确匹配（EM）组件，该组件仅根据查询和文档中的词汇匹配来对齐表示，我们按照Lin和Ma [26] 的命名法将这种模型变体称为Uni - ColBERTer。

Figure 1 illustrates ColBERTer’s BOW ${}^{2}$ representation and how we can display whole-word scores to the user in a keyword view. By aggregating all subwords to whole words, the whole-word scores of this complex medical-domain query illustrate ColBERTer's interpretability capabilities, without cherry picking examples that only contain words that are fully part of BERT's vocabulary.

图1展示了ColBERTer的BOW ${}^{2}$ 表示法，以及我们如何在关键词视图中向用户展示全词得分。通过将所有子词聚合为全词，这个复杂医学领域查询的全词得分展示了ColBERTer的可解释性能力，而无需特意挑选仅包含完全属于BERT词汇表的单词的示例。

The ColBERTer architecture enables various indexing and retrieval scenarios. Building on recent work $\left\lbrack  {{12},{26}}\right\rbrack$ ,we provide a holistic categorization and ablation study of five possible usage scenarios of ColBERTer encoded sequences: sparse token retrieval, dense single vector retrieval, as well as refining either one of the retrieval sources and a full hybrid mode. Specifically, we study:

ColBERTer架构支持各种索引和检索场景。基于近期的工作 $\left\lbrack  {{12},{26}}\right\rbrack$ ，我们对ColBERTer编码序列的五种可能使用场景进行了全面分类和消融研究：稀疏标记检索、密集单向量检索，以及对任一检索源进行细化和全混合模式。具体来说，我们研究：

RQ1 Which aggregation and training regime works best for combined retrieval and refinement capabilities of ColBERTer?

RQ1 哪种聚合和训练机制最适合ColBERTer的组合检索和细化能力？

We find that multi-task learning with two weighted loss functions for retrieval and refinement and a learned score aggregation of both consistently outperforms fixed score aggregation. We investigate jointly training aggregation, ${\mathrm{{BOW}}}^{2}$ ,and contextualized stopwords with a weighted multi-task loss. We find that tuning the weights improves the tradeoff between removed vectors and retrieval quality, but that the results are robust to small hyperpa-rameter changes.

我们发现，使用两个加权损失函数分别进行检索和细化的多任务学习，以及对两者进行学习得分聚合，始终优于固定得分聚合。我们研究了使用加权多任务损失联合训练聚合、${\mathrm{{BOW}}}^{2}$ 和上下文停用词。我们发现，调整权重可以改善去除向量和检索质量之间的权衡，但结果对小的超参数变化具有鲁棒性。

Following our definition of dense and sparse combinations, we study various deployment scenarios and answer:

根据我们对密集和稀疏组合的定义，我们研究了各种部署场景并回答：

RQ2 What is ColBERTer's best indexing and refinement strategy?

RQ2 ColBERTer的最佳索引和细化策略是什么？

Interestingly, we find that a full hybrid retrieval deployment is unnecessary, and only results in very modest and not significant gains compared to a sparse or dense index with passage refinement of the other component. While a dense index produces higher recall than a sparse one, the effect on the top 10 results becomes negligible after refinement, especially on TREC-DL. This novel result could lead to less complexity in deployment, as only one index is required. Practitioners could choose to keep a sparse index, if they already made significant investments or choose only a dense approximate nearest neighbor index for more predictable query latency. Both sparse and dense encodings of ColBERTer can be optimized with common indexing improvements.

有趣的是，我们发现全混合检索部署是不必要的，与使用另一个组件进行段落细化的稀疏或密集索引相比，仅能带来非常微小且不显著的收益。虽然密集索引比稀疏索引产生更高的召回率，但在细化后，对前10个结果的影响变得可以忽略不计，尤其是在TREC - DL上。这一新颖的结果可以降低部署的复杂性，因为只需要一个索引。从业者如果已经进行了大量投资，可以选择保留稀疏索引，或者仅选择密集近似最近邻索引以获得更可预测的查询延迟。ColBERTer的稀疏和密集编码都可以通过常见的索引改进方法进行优化。

With our hyperparameters fixed, we aim to understand the quality effect of reducing storage factors along 2 axes of ColBERTer:

在固定超参数的情况下，我们旨在了解沿ColBERTer的两个轴减少存储因素对质量的影响：

RQ3 How do different configurations of dimensionality and vector count affect the retrieval quality of ColBERTer?

RQ3 不同的维度和向量数量配置如何影响ColBERTer的检索质量？

We study the effect of ${\mathrm{{BOW}}}^{2}$ ,CS,and EM reductions on across dimensions(32,16,8,and 1)and find that,while retrieval quality is reduced with each dimension reduction, the delta is small. Furthermore,we observe that ${\mathrm{{BOW}}}^{2}$ and $\mathrm{{CS}}$ reductions result - on every dimension setting - in a Pareto improvement over simply reducing the number of dimensions.

我们研究了 ${\mathrm{{BOW}}}^{2}$ 、CS和EM减少在不同维度（32、16、8和1）上的影响，发现虽然每次维度减少都会降低检索质量，但降幅很小。此外，我们观察到，在每个维度设置下，${\mathrm{{BOW}}}^{2}$ 和 $\mathrm{{CS}}$ 减少都比单纯减少维度数量带来帕累托改进。

While we want to emphasize that it becomes increasingly hard to contrast neural retrieval architectures - due to the diversity surrounding training procedures - and make conclusive statements about "SOTA" - due to evaluation uncertainty - we still compare ColBERTer to related approaches:

虽然我们想强调，由于训练过程的多样性，越来越难以对比神经检索架构，并且由于评估的不确定性，很难对“最优技术（SOTA）”做出确定性的陈述，但我们仍然将ColBERTer与相关方法进行了比较：

## RQ4 How does the fully optimized ColBERTer system compare to other end-to-end retrieval approaches?

## RQ4 完全优化的ColBERTer系统与其他端到端检索方法相比如何？

We find that ColBERTer improves effectiveness compared to related approaches, especially for systems with low storage footprint. Uni-ColBERTer especially outperforms previous single-dimension token encoding approaches, while offering improved transparency with score mappings to whole words.

我们发现，与相关方法相比，ColBERTer提高了有效性，特别是对于存储占用较小的系统。Uni - ColBERTer尤其优于以前的单维标记编码方法，同时通过全词得分映射提供了更高的透明度。

To evaluate the robustness of ColBERTer we test it on seven high-quality and diverse collections from different domains. We use a meta-analysis [45] that reveals whether statistical significant gains are achieved over multiple collections. We investigate:

为了评估ColBERTer的鲁棒性，我们在来自不同领域的七个高质量、多样化的集合上对其进行了测试。我们使用元分析 [45] 来揭示在多个集合上是否实现了统计上显著的收益。我们研究：

RQ5 How robust is ColBERTer when applied out of domain?

RQ5 ColBERTer在跨领域应用时的鲁棒性如何？

We find that ColBERTer with token embeddings of 32 or Uni-ColBERTer with 1 dimension both show an overall significantly higher retrieval effectiveness compared to BM25, with not a single collection worse than BM25. Compared to a TAS-Balanced trained dense retriever [16] ColBERTer is not statistically significantly worse on any single collection. While we observe an overall positive effect it is not statistically significant within a ${95}\%$ confidence interval. This robust analysis tries to not overestimate the benefits of ColBERTer, while at the same time giving us more confidence in the results. We publish our code, trained models, and documentation at: github.com/sebastian-hofstaetter/colberter

我们发现，具有32维Token嵌入（Token embeddings）的ColBERTer或具有1维的Uni - ColBERTer与BM25相比，整体检索效果显著更高，且没有一个数据集的表现比BM25差。与经过TAS - Balanced训练的密集检索器[16]相比，ColBERTer在任何单个数据集上的表现都没有在统计上显著更差。虽然我们观察到了整体的积极效果，但在${95}\%$置信区间内，这种效果在统计上并不显著。这种稳健性分析试图不过高估计ColBERTer的优势，同时让我们对结果更有信心。我们在以下网址发布了代码、训练好的模型和文档：github.com/sebastian - hofstaetter/colberter

## 2 BACKGROUND

## 2 背景

This section empirically motivates storing unique whole-word representations, reviews the single-vector ${\mathrm{{BERT}}}_{\mathrm{{DOT}}}$ and multi-vector ColBERT architectures, and describes other related approaches.

本节从实证角度阐述了存储唯一全词表示的动机，回顾了单向量${\mathrm{{BERT}}}_{\mathrm{{DOT}}}$和多向量ColBERT架构，并描述了其他相关方法。

---

<!-- Footnote -->

${}^{1}$ Such as a recent 2021 proposal by the EU Commission on AI regulation,see: https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:52021PC0206 (Art. 13)

${}^{1}$ 例如欧盟委员会2021年最近提出的关于人工智能监管的提案，参见：https://eur - lex.europa.eu/legal - content/EN/TXT/?uri = CELEX:52021PC0206 (第13条)

<!-- Footnote -->

---

### 2.1 Tokenization

### 2.1 分词

Many modern neural IR models use a BERT [9] variant to contextualize sequences and are thus locked into a specific tokenization scheme. The BERT tokenizer first splits full text on whitespace and punctuation characters and then uses the WordPiece algorithm [44] to split words to sub-word tokens in a reduced vocabulary. TAg-gregating unique-stemmed whole-words only stores from 59% to ${36}\%$ of the original sub-word units of BERT in our used collections. Related multi-vector methods, such as ColBERT or (Uni)COIL, generally save all BERT tokens,while our ${\mathrm{{BOW}}}^{2}$ aggregation (§3.2) saves only stemmed unique whole-words.

许多现代神经信息检索（IR）模型使用BERT[9]的变体来对序列进行上下文处理，因此被限制在特定的分词方案中。BERT分词器首先根据空格和标点符号对全文进行分割，然后使用WordPiece算法[44]将单词分割成简化词汇表中的子词Token。TAg - 聚合唯一词干化全词在我们使用的数据集中仅存储BERT原始子词单元的59%到${36}\%$。相关的多向量方法，如ColBERT或（Uni）COIL，通常会保存所有BERT的Token，而我们的${\mathrm{{BOW}}}^{2}$聚合（§3.2）仅保存词干化的唯一全词。

### 2.2 BERT ${}_{\text{DOT }}$ and ColBERT Architectures

### 2.2 BERT ${}_{\text{DOT }}$和ColBERT架构

${\mathrm{{BERT}}}_{\mathrm{{DOT}}}$ matches a single vector of the query with a single vector of a passage, produced by independent BERT computations [30, 32, 55]. ColBERT [22] delays the interactions between query and document to after the BERT computation. For more information we refer the reader to Hofstätter et al. [15].

${\mathrm{{BERT}}}_{\mathrm{{DOT}}}$ 将查询的单个向量与段落的单个向量进行匹配，这些向量由独立的BERT计算产生[30, 32, 55]。ColBERT[22]将查询和文档之间的交互延迟到BERT计算之后。更多信息请读者参考Hofstätter等人的文献[15]。

### 2.3 Related Work

### 2.3 相关工作

Vector Reduction. Previous neural IR work on reducing the number of vectors produce fixed sizes across all passages. Lassance et al. [23] prune ColBERT representations to either 50 or 10 vectors by sorting tokens by Inverse Document Frequency (IDF) or attention scores from BERT. Zhou and Devlin [57] extend ColBERT with temporal pooling, sliding a window over the passage to create a vector per step with a fixed target count. Luan et al. [33] represent each passage with a fixed number of embeddings of the CLS token and the first $m$ token of the passage,and compute relevance as the maximum score of the embeddings. Humeau et al. [18] compute a fixed number of vectors per query, and aggregate them by softmax attention against document vectors. Lee et al. [24] learn phrase (multi-word) representations for QA collections. This reduces the vector count, but it depends on the availability of exact answer spans in passages and is therefore not universally applicable in IR. Tonellotto and Macdonald [47] prune the embeddings of the query terms but the document embeddings.

向量缩减。以往关于减少向量数量的神经信息检索工作会在所有段落上生成固定大小的向量。Lassance等人[23]通过根据逆文档频率（IDF）或BERT的注意力分数对Token进行排序，将ColBERT表示修剪为50个或10个向量。Zhou和Devlin[57]通过时间池化扩展了ColBERT，在段落上滑动窗口，以固定的目标数量为每一步创建一个向量。Luan等人[33]用固定数量的CLS Token和段落的前$m$个Token的嵌入来表示每个段落，并将相关性计算为嵌入的最大分数。Humeau等人[18]为每个查询计算固定数量的向量，并通过针对文档向量的softmax注意力对它们进行聚合。Lee等人[24]为问答数据集学习短语（多词）表示。这减少了向量数量，但它依赖于段落中是否存在确切的答案跨度，因此在信息检索中并非普遍适用。Tonellotto和Macdonald[47]修剪查询词的嵌入，但不修剪文档嵌入。

In summary, unlike related vector reduction techniques we: 1) reduce a dynamic number of vectors per passage; 2) keep a mapping between human-readable tokens and vectors, allowing scoring information to be used in the user interface; 3) learn the full pruning process end-to-end without term-based supervision.

总之，与相关的向量缩减技术不同，我们：1）减少每个段落中动态数量的向量；2）保留人类可读的Token和向量之间的映射，以便在用户界面中使用评分信息；3）在没有基于词项的监督下端到端地学习整个修剪过程。

Vector Compression. Ma et al. [34] study various methods to reduce the dimension of dense retrieval vectors. Unlike our study, they find that learned dimension reduction performs poorly. Also for single vector retrieval Zhan et al. [56] optimize product quantization as part of the training. Recently, Santhanam et al. [43] study residual compression of all saved token vectors as part of the ColBERT end-to-end retrieval setting. There are concurrent efforts revisiting lexical matching with learned sparse representations $\left\lbrack  {{10},{12},{26}}\right\rbrack$ or learned passage impacts [37],which employ the efficiency of exact lexical matches. Different to our work, they focus on reducing the number of dimensions of the learned em-beddings without reducing the number of stored tokens. Many of these approaches can be considered complementary to our proposed methods, and future work should evaluate how well these methods compose to achieve even larger compression rates.

向量压缩。Ma等人[34]研究了各种减少密集检索向量维度的方法。与我们的研究不同，他们发现学习到的降维方法效果不佳。同样对于单向量检索，Zhan等人[56]在训练过程中优化乘积量化。最近，Santhanam等人[43]在ColBERT端到端检索设置中研究了所有保存的Token向量的残差压缩。同时，也有一些工作重新审视了使用学习到的稀疏表示$\left\lbrack  {{10},{12},{26}}\right\rbrack$或学习到的段落影响[37]进行词法匹配，这些方法利用了精确词法匹配的效率。与我们的工作不同，它们专注于减少学习到的嵌入的维度，而不减少存储的Token数量。这些方法中的许多可以被认为是我们提出的方法的补充，未来的工作应该评估这些方法如何组合以实现更大的压缩率。

## 3 ColBERTer: ENHANCED REDUCTION

## 3 ColBERTer：增强型缩减

ColBERT with enhanced reduction, or ColBERTer, combines the encoding architectures of ${\mathrm{{BERT}}}_{\mathrm{{DOT}}}$ and $\mathrm{{ColBERT}}$ ,while extremely reducing the token storage and latency requirements along the effectiveness Pareto frontier. Our enhancements maintain model transparency, creating a concrete mapping of scoring sources and human-readable whole-words.

具有增强降维功能的ColBERT，即ColBERTer，结合了${\mathrm{{BERT}}}_{\mathrm{{DOT}}}$和$\mathrm{{ColBERT}}$的编码架构，同时在有效性帕累托前沿上极大地降低了词元存储和延迟要求。我们的改进保持了模型的透明度，创建了评分来源的具体映射和人类可读的完整单词。

ColBERTer independently encodes the query and the document using a transformer encoder like BERT, producing token-level representation similar to ColBERT:

ColBERTer使用类似BERT的Transformer编码器独立编码查询和文档，生成类似于ColBERT的词元级表示：

$$
{\widetilde{q}}_{1 : m + 2} = \operatorname{BERT}\left( \left\lbrack  {\mathrm{{CLS}};{q}_{1 : m};\mathrm{{SEP}}}\right\rbrack  \right) 
$$

$$
{\widetilde{p}}_{1 : n + 2} = \operatorname{BERT}\left( \left\lbrack  {\mathrm{{CLS}};{p}_{1 : n};\mathrm{{SEP}}}\right\rbrack  \right)  \tag{1}
$$

To maximize transparency, we do not apply the query augmentation mechanism of Khattab and Zaharia [22] (see §2.2), which appends MASK tokens to the query with the goal of implicit - and thus potentially opaque - query expansion.

为了最大限度地提高透明度，我们不应用Khattab和Zaharia [22]的查询增强机制（见§2.2），该机制将掩码（MASK）词元附加到查询中，目的是进行隐式的——因此可能不透明的——查询扩展。

### 3.1 2-Way Dimension Reduction

### 3.1 双向降维

Given the transformer encoder output, ColBERTer uses linear layers to reduce the dimensionality of the output vectors in two ways: 1) we use the linear layer ${W}_{CLS}$ to control the dimension of the first CLS-token representation (e.g. 128 dimensions):

给定Transformer编码器的输出，ColBERTer使用线性层以两种方式降低输出向量的维度：1) 我们使用线性层${W}_{CLS}$来控制第一个CLS词元表示的维度（例如128维）：

$$
{q}_{CLS} = {\widetilde{q}}_{1} * {W}_{CLS} \tag{2}
$$

$$
{p}_{CLS} = {\widetilde{p}}_{1} * {W}_{CLS}
$$

and 2) the layer ${W}_{t}$ projects the remaining tokens down to the token embedding dimension (usually smaller, e.g. 32):

2) 层${W}_{t}$将其余词元投影到词元嵌入维度（通常更小，例如32维）：

$$
{\dot{q}}_{1 : m} = {\widetilde{q}}_{2 : m + 1} * {W}_{t}
$$

$$
{\dot{p}}_{1 : n} = {\widetilde{p}}_{2 : n + 1} * {W}_{t} \tag{3}
$$

This 2-way reduction combined with our novel training work-flow (§4.1) serves to reduce our space footprint compared to ColBERT and at the same time provides more expressive encodings than a single vector ${\mathrm{{BERT}}}_{\mathrm{{DOT}}}$ model. Furthermore,it enables a multitude of potential dense and sparse retrieval workflows (§4.2).

这种双向降维与我们新颖的训练工作流程（§4.1）相结合，与ColBERT相比减少了我们的空间占用，同时比单向量${\mathrm{{BERT}}}_{\mathrm{{DOT}}}$模型提供了更具表现力的编码。此外，它还支持多种潜在的密集和稀疏检索工作流程（§4.2）。

### 3.2 BOW ${}^{2}$ : Bag of Unique Whole-Words

### 3.2 BOW ${}^{2}$：唯一完整单词袋

Given the token representations $\left( {\dot{q}}_{1 : m}\right.$ and $\left. {\dot{p}}_{1 : n}\right)$ ,ColBERTer applies its novel key transformation: ${\mathrm{{BOW}}}^{2}$ to the sequence of vectors. Whereas ColBERT and COIL maintain one vector for each BERT token, including tokens corresponding to sub-words in the BERT vocabulary, we create a single representation for each unique whole word. This serves to further reduce the storage overhead of our model by reducing the number of tokens, while preserving an explicit mapping of score parts to human understandable words.

给定词元表示$\left( {\dot{q}}_{1 : m}\right.$和$\left. {\dot{p}}_{1 : n}\right)$，ColBERTer对向量序列应用其新颖的关键变换：${\mathrm{{BOW}}}^{2}$。虽然ColBERT和COIL为每个BERT词元（包括对应于BERT词汇表中子词的词元）维护一个向量，但我们为每个唯一的完整单词创建一个单一表示。这通过减少词元数量进一步降低了我们模型的存储开销，同时保留了得分部分与人类可理解单词的显式映射。

During tokenization we build a mapping between each sub-word token and corresponding unique whole word (as defined by a simple split on punctuation and whitespace characters). The words can also be transformed through classical IR techniques such as stemming. Then, inside the model we aggregate whole word representations for each whole word $w$ in passage $p$ by computing the mean of the embeddings of $w$ ’s constituent sub-words ${\dot{p}}_{i}$ . We get the set of unique whole-word representation of the passage $p$ :

在分词过程中，我们构建每个子词词元与相应唯一完整单词之间的映射（通过对标点符号和空白字符进行简单分割来定义）。这些单词也可以通过词干提取等经典信息检索技术进行转换。然后，在模型内部，我们通过计算$w$的组成子词${\dot{p}}_{i}$的嵌入的平均值，为段落$p$中的每个完整单词$w$聚合完整单词表示。我们得到段落$p$的唯一完整单词表示集合：

$$
{\widehat{p}}_{1 : \widehat{n}} = \left\{  {\left. {\frac{1}{\left| {\dot{p}}_{i} \in  w\right| }\mathop{\sum }\limits_{{{\dot{p}}_{i} \in  w}}{\dot{p}}_{i}}\right| \;\forall w \in  {\mathrm{{BOW}}}^{2}\left( p\right) }\right\}   \tag{4}
$$

We apply the same procedure symmetrically to the query vectors ${\dot{q}}_{1 : m}$ from equation (7) as well to produce ${\widehat{q}}_{1 : \widehat{m}}$ . The resulting sets are still dynamic in length as their length now depends on the number of whole words $(\widehat{n}$ and $\widehat{m}$ for passage and query sequences respectively). We refer to the new sets as bag of words, as we only save one word once and the order of the vectors now does not matter anymore, because the language model contextualization already happened.

我们对称地对公式(7)中的查询向量${\dot{q}}_{1 : m}$应用相同的过程，以生成${\widehat{q}}_{1 : \widehat{m}}$。得到的集合长度仍然是动态的，因为它们的长度现在分别取决于段落和查询序列的完整单词数量$(\widehat{n}$和$\widehat{m}$）。我们将新集合称为单词袋，因为我们每个单词只保存一次，并且向量的顺序现在不再重要，因为语言模型的上下文处理已经完成。

### 3.3 Simplified Contextualized Stopwords

### 3.3 简化的上下文停用词

To further reduce the number of passage tokens to store, we adopt a simplified version of Hofstätter et al. [17]'s contextualized stop-words (CS), which was first introduced for the TK-Sparse model. CS learns a removal gate of tokens solely based on their context-dependent vector representations. We simplify the original implementation of CS and adapt the removal process to fit into the encoding phase of the ColBERTer model.

为了进一步减少要存储的段落词元数量，我们采用了Hofstätter等人[17]的上下文停用词（CS）的简化版本，该方法最初是为TK - 稀疏模型引入的。CS仅基于词元的上下文相关向量表示学习一个词元移除门。我们简化了CS的原始实现，并调整移除过程以适应ColBERTer模型的编码阶段。

Every whole-word passage vector ${\widehat{p}}_{j}$ is transformed by a linear layer (with weights ${W}_{s}$ and bias ${b}_{s}$ ),followed by a ReLU activation, to compute a single-dimensional stopword removal gate ${r}_{j}$ :

每个完整单词段落向量${\widehat{p}}_{j}$通过一个线性层（权重为${W}_{s}$，偏置为${b}_{s}$）进行变换，然后进行ReLU激活，以计算一个一维的停用词移除门${r}_{j}$：

$$
{r}_{j} = \operatorname{ReLU}\left( {{\widehat{p}}_{j}{W}_{s} + {b}_{s}}\right)  \tag{5}
$$

The original implementation [17] masks scores after TK's kernel-activation, meaning the non-zero gates have to be saved as well, which increases the systems' complexity. In contrast, we directly apply the gate to the representation vectors. In particular, we drop every representation where the gate ${r}_{j} = 0$ ,and otherwise scale the magnitude of the remaining representations using their gate scores:

原始实现[17]在TK的核激活后屏蔽得分，这意味着非零门也必须保存，从而增加了系统的复杂性。相比之下，我们直接将门应用于表示向量。具体来说，我们丢弃门值为${r}_{j} = 0$的每个表示，否则使用它们的门得分缩放其余表示的大小：

$$
{\widehat{p}}_{j} = {\widehat{p}}_{j} * {\widehat{r}}_{j} \tag{6}
$$

This fully differentiable approach allows us to learn the stopword gate during training and remove all nullified vectors at indexing time, as they do not contribute to document scores. Applying the stopword gate directly to the representation vector allows us to observe much more stable training than the authors of TK-Sparse observed - we do not need to adapt the training procedure with special mechanisms to keep the model from collapsing. Following Hofstätter et al. [17] we train the removal gate with a regularization loss, forcing the stopword removal gate to become active during training (§4.1).

这种完全可微的方法使我们能够在训练期间学习停用词门控，并在索引时移除所有无效向量，因为它们对文档得分没有贡献。将停用词门控直接应用于表示向量，我们观察到的训练比TK - 稀疏模型的作者所观察到的更加稳定——我们不需要采用特殊机制来调整训练过程以防止模型崩溃。遵循霍夫施泰特等人 [17] 的方法，我们使用正则化损失来训练移除门控，迫使停用词移除门控在训练期间生效（§4.1）。

### 3.4 Matching & Score Aggregation

### 3.4 匹配与得分聚合

After we complete the independent encoding of query and passage sequences, we need to match and score them. ColBERTer creates two scores, one for the CLS vector and one for the token vectors. The CLS score is a dot product of the two CLS vectors:

在完成查询和段落序列的独立编码后，我们需要对它们进行匹配和打分。ColBERTer生成两个得分，一个是CLS向量得分，另一个是词元向量得分。CLS得分是两个CLS向量的点积：

$$
{s}_{CLS} = {q}_{CLS} \cdot  {p}_{CLS} \tag{7}
$$

The token score follows the scoring regime of ColBERT, with a match matrix of word-by-word dot product and max-pooling the document word dimension followed by a sum over all query words:

词元得分遵循ColBERT的评分机制，通过逐词点积得到匹配矩阵，对文档词元维度进行最大池化，然后对所有查询词元求和：

$$
{s}_{\text{token }} = \mathop{\sum }\limits_{{j = 1}}^{\widehat{m}}\mathop{\max }\limits_{{i = 1\ldots \widehat{n}}}{\widehat{q}}_{j}^{T} \cdot  {\widehat{p}}_{i} \tag{8}
$$

The final score of a query-passage pair is computed with a learned aggregation of the two score components:

查询 - 段落对的最终得分是通过对两个得分组件进行学习聚合计算得出的：

$$
{s}_{\text{ColBERTer }} = \sigma \left( \gamma \right)  * {s}_{\text{CLS }} + \left( {1 - \sigma \left( \gamma \right) }\right)  * {s}_{\text{token }} \tag{9}
$$

where $\sigma$ is the sigmoid function,and $\gamma$ is a trainable scalar parameter. For ablations, $\sigma \left( \gamma \right)$ can be set to a fixed number,such as 0.5 . While the learned weighting factor may seems superfluous, as the upstream linear layers could already learn to change the magnitudes of the two components,we show in $\$ {6.1}$ that the explicit weighting is crucial to the effectiveness of both components.

其中 $\sigma$ 是Sigmoid函数，$\gamma$ 是可训练的标量参数。在消融实验中，$\sigma \left( \gamma \right)$ 可以设置为一个固定的数字，例如0.5。虽然学习到的加权因子可能看起来多余，因为上游的线性层已经可以学习改变两个组件的大小，但我们在 $\$ {6.1}$ 中表明，显式加权对两个组件的有效性至关重要。

### 3.5 Uni-ColBERTer: Extreme Reduction with Lexical Matching

### 3.5 Uni - ColBERTer：基于词法匹配的极致降维

While ColBERTer considerably reduces the dimension of the representations already, we found in pilot studies that for an embedding dimension of 8 or lower the full match matrix is detrimental to the effectiveness. Lin and Ma [26] showed that a token score model can be effectively reduced to one dimension in UniCOIL. This reduces the token representations to scalar weights, necessitating an alternative mechanism to match query tokens with "similar" document tokens.

虽然ColBERTer已经显著降低了表示的维度，但我们在初步研究中发现，对于嵌入维度为8或更低的情况，完整的匹配矩阵会降低有效性。林和马 [26] 表明，在UniCOIL中，词元得分模型可以有效地降为一维。这将词元表示简化为标量权重，因此需要一种替代机制来将查询词元与“相似”的文档词元进行匹配。

To fit the same reduction we need to apply more techniques to our ColBERTer architecture to create Uni-ColBERTer with single dimensional whole word vectors. While we now occupy the same bytes per vector, our vector reduction techniques make Uni-COLBERTer 2.5 times smaller than UniCOIL (on MSMARCO).

为了实现相同的降维，我们需要对ColBERTer架构应用更多技术，以创建具有单维全词向量的Uni - ColBERTer。虽然现在每个向量占用的字节数相同，但我们的向量降维技术使Uni - COLBERTer比UniCOIL小2.5倍（在MSMARCO数据集上）。

To reduce the token encoding to 1 dimension we apply a second linear layer after the contextualized stopword component:

为了将词元编码降为一维，我们在上下文停用词组件之后应用第二个线性层：

$$
{\widehat{q}}_{1 : m + 2} = {\widehat{q}}_{1 : \widehat{m}} * {W}_{u} \tag{10}
$$

$$
{\widehat{p}}_{1 : n + 2} = {\widehat{p}}_{1 : \widehat{n}} * {W}_{u}
$$

Furthermore, we need to apply a lexical match bias, following COIL's, to only match identical words with each other. This creates engineering challenge: we do not build a global vocabulary with ids of whole-words during training nor inference as doing so would make it difficult to saturate modern GPUs, requiring multiple synchronized CPU processes (4-10 depending on the system) that prepare the input with tokenization, data transformation, and subsequent tensor batching of sequences. To keep track of a global vocabulary, these CPU processes would need to synchronize with a read-write dictionary on every token. This is very challenging at best in python multiprocessing while keeping the necessary speed to fully use even a single GPU.

此外，我们需要像COIL那样应用词法匹配偏差，只对相同的单词进行匹配。这带来了工程上的挑战：我们在训练和推理过程中都不构建带有全词ID的全局词汇表，因为这样做会使现代GPU难以饱和，需要多个同步的CPU进程（根据系统不同为4 - 10个）来进行词元化、数据转换以及后续序列的张量批处理以准备输入。为了跟踪全局词汇表，这些CPU进程需要在每个词元上与读写字典进行同步。在Python多进程中，要在保持充分利用单个GPU所需速度的同时做到这一点，即使是最好的情况也非常具有挑战性。

To overcome this problem, we propose approximate lexical interactions by creating an $\mathrm{n}$ -bit hash $H$ from every whole-word without accounting for potential collisions and applying a mask of equal hashes to the match matrix. Depending on the selection of bits to keep this introduces different numbers of collisions. ${}^{2}$ Depending on the collection size one can adjust the number of bits to save from the hash. With the hashed global id of whole words we can adjust the match matrix of whole-words for low dimension token models as follows:

为了克服这个问题，我们提出通过为每个全词创建一个 $\mathrm{n}$ 位哈希 $H$ 来进行近似词法交互，不考虑潜在的冲突，并将相等哈希的掩码应用于匹配矩阵。根据保留的位数选择，这会引入不同数量的冲突。${}^{2}$ 根据数据集的大小，可以调整从哈希中保留的位数。有了全词的哈希全局ID，我们可以按如下方式调整低维词元模型的全词匹配矩阵：

$$
{s}_{\text{token }} = \mathop{\sum }\limits_{1}^{\widehat{m}}\mathop{\max }\limits_{{1..\widehat{n}}}\mathop{\max }\limits_{{\left| {H\left( {w}_{\widehat{n}}\right) }\right|  = H\left( {w}_{\widehat{m}}\right) }}{\widehat{q}}_{1 : \widehat{m} + 2}^{T} \cdot  {\widehat{p}}_{1 : \widehat{n} + 2} \tag{11}
$$

In practice, we implement this procedure by masking the full match matrix, so that the operation works on batched tensors. Besides allowing us reduce the token dimensionality to one, the lexical matching component of Uni-ColBERTer enables the sparse indexing of tokens in an inverted index, following UniCOIL.

在实践中，我们通过对完整匹配矩阵进行掩码操作来实现这个过程，以便该操作可以在批量张量上进行。除了使我们能够将词元维度降为一维之外，Uni - ColBERTer的词法匹配组件还能像UniCOIL一样，在倒排索引中对词元进行稀疏索引。

---

<!-- Footnote -->

${}^{2}$ On MSMARCO we found that the first 32 bits of sha256 produce very few collisions (303 collisions out of 1.6 million hashes).

${}^{2}$ 在MSMARCO数据集上，我们发现sha256的前32位产生的冲突非常少（160万个哈希中只有303个冲突）。

<!-- Footnote -->

---

## 4 MODEL LIFECYCLE

## 4 模型生命周期

In this section we describe how we train our ColBERTer architecture and how we can deploy the trained model into a retrieval system.

在本节中，我们将描述如何训练ColBERTer架构，以及如何将训练好的模型部署到检索系统中。

### 4.1 Training Workflow

### 4.1 训练工作流程

We train our ColBERTer model with triples of one query, and two passages where one is more relevant than the other. To incorporate the degree of relevance, as provided by a teacher model we use the Margin-MSE loss [15], formalized as follows:

我们使用一个查询和两个段落组成的三元组来训练我们的ColBERTer模型，其中一个段落比另一个更相关。为了纳入由教师模型提供的相关性程度，我们使用Margin - MSE损失函数[15]，其形式化表示如下：

$$
{\mathcal{L}}_{\text{MarginMSE }}\left( {M}_{s}\right)  = \operatorname{MSE}\left( {{M}_{s}^{ + } - {M}_{s}^{ - },{M}_{t}^{ + } - {M}_{t}^{ - }}\right)  \tag{12}
$$

Where a teacher model ${M}_{t}$ provides a teacher signal for our student model ${M}_{s}$ (in our case ColBERTer’s output parts). From the outside ColBERTer looks and acts like a single model, however it is in essence a multi-task model: aggregating sequences into a single vector, representing individual words, and actively removing uninformative words. Therefore, we need to train these three components in a balanced form, with a combined loss function:

其中，教师模型${M}_{t}$为我们的学生模型${M}_{s}$（在我们的例子中是ColBERTer的输出部分）提供教师信号。从外部看，ColBERTer看起来和表现得像一个单一的模型，但实际上它是一个多任务模型：将序列聚合为单个向量、表示单个单词并主动去除无信息的单词。因此，我们需要用一个组合损失函数以平衡的形式训练这三个组件：

$$
\mathcal{L} = {\alpha }_{b} * {\mathcal{L}}_{b} + {\alpha }_{CLS} * {\mathcal{L}}_{CLS} + {\alpha }_{CS} * {\mathcal{L}}_{CS} \tag{13}
$$

where $\alpha$ ’s are hyperparamters governing the weighting of the individual losses, which we explain in the following. The combined loss for both sub-scores ${\mathcal{L}}_{b}$ uses MarginMSE supervision on the final score:

其中$\alpha$是控制各个损失权重的超参数，我们将在下面进行解释。两个子分数${\mathcal{L}}_{b}$的组合损失在最终分数上使用MarginMSE监督：

$$
{\mathcal{L}}_{b} = {\mathcal{L}}_{\text{MarginMSE }}\left( {s}_{\text{ColBERTer }}\right)  \tag{14}
$$

In pilot studies and shown in $\$ {6.1}$ we observed that training ColBERTer only with a combined loss strongly reduces the effectiveness of the CLS vector alone. To overcome this issue and be able to use single vector retrieval we define ${\mathcal{L}}_{CLS}$ as:

在初步研究中，如$\$ {6.1}$所示，我们观察到仅使用组合损失训练ColBERTer会极大降低CLS向量单独使用时的有效性。为了克服这个问题并能够使用单向量检索，我们将${\mathcal{L}}_{CLS}$定义为：

$$
{\mathcal{L}}_{CLS} = {\mathcal{L}}_{\text{MarginMSE }}\left( {s}_{CLS}\right)  \tag{15}
$$

Finally, to actually force the model to learn sparsity in the removal gate vector $r$ of the contextualized stopword component,we follow Hofstätter et al. [17] and add an ${\mathcal{L}}_{CS}$ loss of the L1-norm of the positive $\&$ negative $r$ :

最后，为了真正迫使模型在上下文停用词组件的移除门向量 $r$ 中学习稀疏性，我们遵循霍夫施泰特（Hofstätter）等人 [17] 的方法，添加正 $\&$ 负 $r$ 的 L1 范数的 ${\mathcal{L}}_{CS}$ 损失：

$$
{\mathcal{L}}_{CS} = {\begin{Vmatrix}{r}^{ + }\end{Vmatrix}}_{1} + {\begin{Vmatrix}{r}^{ - }\end{Vmatrix}}_{1} \tag{16}
$$

This introduces some tension in training: the sparsity loss needs to move as many entries to close to zero, while the token loss as part of ${\mathcal{L}}_{b}$ needs non-zeros to determine relevance matches. To reduce volatility, we train the enhanced reduction components one after another. We start with a ColBERT checkpoint, followed by the 2-way dimensionality reduction, ${\mathrm{{BOW}}}^{2}$ and $\mathrm{{CS}}$ ,and finally for Uni-ColBERTer we apply another round of reduction.

这在训练中引入了一些矛盾：稀疏性损失需要使尽可能多的元素接近零，而作为 ${\mathcal{L}}_{b}$ 一部分的词元损失需要非零元素来确定相关性匹配。为了降低波动性，我们依次训练增强的降维组件。我们从 ColBERT 检查点开始，接着进行双向降维，即 ${\mathrm{{BOW}}}^{2}$ 和 $\mathrm{{CS}}$，最后对于 Uni - ColBERTer 我们再进行一轮降维。

### 4.2 Indexing and Query Workflow

### 4.2 索引和查询工作流程

Once we have trained our ColBERTer model we need to decide how to deploy it into a wider retrieval workflow. ColBERTer's passage encoding can be fully pre-computed in an offline setting, which allows for low latency query-time retrieval.

一旦我们训练好了 ColBERTer 模型，就需要决定如何将其部署到更广泛的检索工作流程中。ColBERTer 的段落编码可以在离线环境中完全预先计算，这使得查询时的检索延迟较低。

Previous works, such as COIL [12] or ColBERT [22] have already established many of the potential workflows. We aim to give a holistic overview of the possible usage scenarios, including ablation

之前的工作，如 COIL [12] 或 ColBERT [22] 已经建立了许多潜在的工作流程。我们旨在全面概述可能的使用场景，包括消融实验

<!-- Media -->

<!-- figureText: Full hybrid: retrieve both, refine both BOW CLS $\rightarrow  \mathbf{0}$ Fill in missing Retrieve CLS only $\rightarrow  \mathbf{O}$ Score Score Retrieve ${\mathrm{{BOW}}}^{2}$ only BOW 。 Score Score CLS Merge Candidates Retrieve CLS then refine ${\mathrm{{BOW}}}^{2}$ CLS BOW ${}^{2}$ Candidates Store then refine CLS BOW CLS Candidate: Store -->

<img src="https://cdn.noedgeai.com/0195aefe-9fbd-7011-8ec7-b40c2affab11_4.jpg?x=1011&y=253&w=522&h=415&r=0"/>

Figure 2: The potential retrieval and refine workflows of ColBERTer at query time. Broadly categorized by: full hybrid (O), single index, then refine with the other (2 + 8), or only one index for ablation purposes (4 + 6).

图 2：查询时 ColBERTer 的潜在检索和精炼工作流程。大致分类如下：全混合（O）、单一索引，然后用另一个索引进行精炼（2 + 8），或者出于消融实验目的仅使用一个索引（4 + 6）。

<!-- Media -->

studies to select the best method with the lowest complexity. We give a schematic overview over ColBERTer's retrieval workflows in Figure 2. We assume that all passages have been encoded and stored accessibly by their id. Each of the two storage categories can be transformed into an index structure for fast retrieval: the CLS index uses an (approximate) nearest neighbor index, while the ${\mathrm{{BOW}}}^{2}$ index could use either a dense nearest neighbor index,or a classic inverted index (with activated exact matching component).

研究以选择复杂度最低的最佳方法。我们在图 2 中给出了 ColBERTer 检索工作流程的示意图。我们假设所有段落都已编码并可通过其 ID 进行访问存储。两种存储类别中的每一种都可以转换为用于快速检索的索引结构：CLS 索引使用（近似）最近邻索引，而 ${\mathrm{{BOW}}}^{2}$ 索引可以使用密集最近邻索引，或者经典的倒排索引（带有激活的精确匹配组件）。

Figure 2 O shows how we can index both scoring components of ColBERTer and then use the id-based storages to fill in missing scores for passages retrieved only by one index. A similar workflow has been explored by Lin and Lin [28] and Gao et al. [12]. Figure 2 2 & 3 utilize only one retrieval index and fill up the missing scores from the complementary id-based storage. This approach works vice-versa for dense or sparse indices, and represents a clear complexity and additional index storage reduction, at the potential of lower recall. This is akin to a two stage retrieve and re-rank pipeline $\left\lbrack  {{15},{16},{29}}\right\rbrack$ ,but such pipeline have been mostly studied with a separate model per stage (which requires larger indexing resources than our single model). Figure 2 4 & 6 represent ablation studies that only rely on one or the other index while disregarding the other scoring part.

图 2 O 展示了我们如何对 ColBERTer 的两个评分组件进行索引，然后使用基于 ID 的存储来填充仅由一个索引检索到的段落的缺失分数。林（Lin）和林（Lin）[28] 以及高（Gao）等人 [12] 已经探索了类似的工作流程。图 2 2 和 3 仅使用一个检索索引，并从互补的基于 ID 的存储中填充缺失分数。这种方法对于密集或稀疏索引反之亦然，并且在可能降低召回率的情况下，实现了明显的复杂度和额外索引存储的减少。这类似于两阶段检索和重排序管道 $\left\lbrack  {{15},{16},{29}}\right\rbrack$，但这种管道大多是针对每个阶段使用单独的模型进行研究的（这比我们的单一模型需要更大的索引资源）。图 2 4 和 6 表示仅依赖一个或另一个索引而忽略另一个评分部分的消融实验。

Different workflows may considerably affect complexity, storage, and effectiveness. We thus always indicate the type of query workflow used (numbers given in Figure 2) in our results section and conduct an ablation study in §6.1.

不同的工作流程可能会显著影响复杂度、存储和有效性。因此，我们在结果部分始终指明所使用的查询工作流程类型（图 2 中给出的编号），并在§6.1 中进行消融实验。

## 5 EXPERIMENT DESIGN

## 5 实验设计

Our main training and inference dependencies are PyTorch [38], HuggingFace Transformers [54], and the nearest neighbor search library Faiss [20]. For training we utilize TAS-Balanced [16] retrieved negatives with BERT-based teacher ensemble scores [15].

我们主要的训练和推理依赖项是 PyTorch [38]、HuggingFace Transformers [54] 和最近邻搜索库 Faiss [20]。在训练中，我们使用 TAS - 平衡 [16] 检索到的负样本以及基于 BERT 的教师集成分数 [15]。

### 5.1 Passage Collection & Query Sets

### 5.1 段落集合和查询集

For training and in-domain evaluation we use the MSMARCO-Passage (V1) collection [3] with the sparsely-judged MSMARCO-DEV query set of 6,980 queries (used in the leaderboard) as well as the densely-judged 97 query set of combined TREC-DL '19 [7] and ’20 [8]. For TREC graded relevance $(0 =$ non relevant to 3 = perfect), we use the recommended binarization point of 2 for the recall metric. For out of domain experiments we refer to the ir_datasets catalogue [35] for collection specific information, as we utilized the standardized test sets for the collections.

在训练和领域内评估中，我们使用 MSMARCO - 段落（V1）集合 [3]，以及稀疏评判的 MSMARCO - 开发查询集（包含 6980 个查询，用于排行榜），还有密集评判的 97 个查询集（由 TREC - DL '19 [7] 和 '20 [8] 组合而成）。对于 TREC 分级相关性（$(0 =$ 从不相关到 3 = 完美），我们使用推荐的二值化点 2 来计算召回率指标。在领域外实验中，由于我们使用了集合的标准化测试集，因此我们参考 ir_datasets 目录 [35] 获取特定集合的信息。

<!-- Media -->

Table 1: Analysis of different score aggregation and training methods for ColBERTer (2-way dim reduction only; CLS dim: 128, token dim: 32; Workflow &) in terms of retrieval effectiveness. We compare refining full-retrieval results from ColBERTer's CLS vector (Own) and a TAS-Balanced retriever (TAS) with different multi-task loss weights ${\alpha }_{b}$ and ${\alpha }_{CLS}$ . Highest Own in bold, lowest underlined.

表1：针对ColBERTer（仅进行2路降维；CLS维度：128，词元维度：32；工作流程&）不同得分聚合和训练方法在检索效果方面的分析。我们比较了使用不同多任务损失权重${\alpha }_{b}$和${\alpha }_{CLS}$时，对来自ColBERTer的CLS向量（自身）和TAS平衡检索器（TAS）的全检索结果进行优化的情况。最高的自身结果用粗体表示，最低的用下划线表示。

<table><tr><td colspan="2" rowspan="2">Train Loss</td><td colspan="4">TREC-DL’19+20</td><td colspan="4">MSMARCO DEV</td></tr><tr><td colspan="2">nDCG@10</td><td colspan="2">R@1K</td><td colspan="2">MRR@10</td><td colspan="2">R@1K</td></tr><tr><td>${\alpha }_{b}$</td><td>${\alpha }_{CLS}$</td><td>Own TAS</td><td/><td>Own</td><td>TAS</td><td>Own</td><td>TAS</td><td>Own</td><td>TAS</td></tr><tr><td colspan="10">Fixed Score Aggregation</td></tr><tr><td>11</td><td>0</td><td>.684</td><td>.740</td><td>.565</td><td>.861</td><td>.336</td><td>.386</td><td>.773</td><td>.978</td></tr><tr><td colspan="10">LearnedScore</td></tr><tr><td>21</td><td>0.1</td><td>.726</td><td>.728</td><td>.783</td><td>.861</td><td>.384</td><td>.386</td><td>.952</td><td>.978</td></tr><tr><td>31</td><td>0.2</td><td>.728</td><td>.731</td><td>.794</td><td>.861</td><td>.384</td><td>.385</td><td>.957</td><td>.978</td></tr><tr><td>41</td><td>0.5</td><td>.734</td><td>.734</td><td>.807</td><td>.861</td><td>.386</td><td>.386</td><td>.961</td><td>.978</td></tr><tr><td>51</td><td>1.0</td><td>.730</td><td>.730</td><td>.806</td><td>.861</td><td>.381</td><td>.381</td><td>.962</td><td>.978</td></tr></table>

<table><tbody><tr><td colspan="2" rowspan="2">训练损失</td><td colspan="4">TREC-DL 2019 + 2020（TREC-DL’19+20）</td><td colspan="4">MSMARCO 开发集（MSMARCO DEV）</td></tr><tr><td colspan="2">前 10 名归一化折损累积增益（nDCG@10）</td><td colspan="2">前 1000 名召回率（R@1K）</td><td colspan="2">前 10 名平均倒数排名（MRR@10）</td><td colspan="2">前 1000 名召回率（R@1K）</td></tr><tr><td>${\alpha }_{b}$</td><td>${\alpha }_{CLS}$</td><td>自有 TAS</td><td></td><td>自有</td><td>TAS</td><td>自有</td><td>TAS</td><td>自有</td><td>TAS</td></tr><tr><td colspan="10">固定分数聚合</td></tr><tr><td>11</td><td>0</td><td>.684</td><td>.740</td><td>.565</td><td>.861</td><td>.336</td><td>.386</td><td>.773</td><td>.978</td></tr><tr><td colspan="10">学习分数（LearnedScore）</td></tr><tr><td>21</td><td>0.1</td><td>.726</td><td>.728</td><td>.783</td><td>.861</td><td>.384</td><td>.386</td><td>.952</td><td>.978</td></tr><tr><td>31</td><td>0.2</td><td>.728</td><td>.731</td><td>.794</td><td>.861</td><td>.384</td><td>.385</td><td>.957</td><td>.978</td></tr><tr><td>41</td><td>0.5</td><td>.734</td><td>.734</td><td>.807</td><td>.861</td><td>.386</td><td>.386</td><td>.961</td><td>.978</td></tr><tr><td>51</td><td>1.0</td><td>.730</td><td>.730</td><td>.806</td><td>.861</td><td>.381</td><td>.381</td><td>.962</td><td>.978</td></tr></tbody></table>

<!-- Media -->

### 5.2 Parameter Settings

### 5.2 参数设置

Our model instances use a 6-layer DistilBERT [42] encoder as their initialization starting point. For our CLS vector we followed guidance by Ma et al. [34] to utilize 128 dimensions, as it provides sufficient capacity for retrieval. For token vectors, we study and present multiple parameter configurations between 32 and 1 dimension. We initialize models with final token output smaller than 32 with the checkpoint of the 32 dimensional model. The ${\mathrm{{BOW}}}^{2}$ and CS components do not need any parameterization, other than using a Porter stemmer to aggregate unique words. These components only need to be parameterized in terms of the training loss influence $\alpha$ ’s. We thoroughly studied the robustness of the model to various configurations in $\$ {6.1}$ .

我们的模型实例使用一个 6 层的 DistilBERT [42] 编码器作为其初始化起点。对于我们的 CLS 向量，我们遵循 Ma 等人 [34] 的指导，采用 128 维，因为它为检索提供了足够的容量。对于词元向量，我们研究并展示了 32 维到 1 维之间的多种参数配置。我们使用 32 维模型的检查点来初始化最终词元输出小于 32 维的模型。${\mathrm{{BOW}}}^{2}$ 和 CS 组件除了使用波特词干提取器（Porter stemmer）来聚合唯一单词外，不需要任何参数化。这些组件只需要在训练损失影响 $\alpha$ 方面进行参数化。我们在 $\$ {6.1}$ 中深入研究了模型对各种配置的鲁棒性。

## 6 RESULTS

## 6 结果

We now address our research question: we study the source of ColBERTer's effectiveness, and under which conditions its components work; then we compare our results to related approaches; and additionally we investigate the robustness of ColBERTer out of domain in Appendix A.

现在我们来探讨我们的研究问题：我们研究 ColBERTer 有效性的来源，以及其组件在哪些条件下起作用；然后我们将我们的结果与相关方法进行比较；此外，我们在附录 A 中研究了 ColBERTer 在域外的鲁棒性。

### 6.1 Source of Effectiveness

### 6.1 有效性来源

Our first investigation seeks to understand the relation between the CLS retrieval and token refinement capabilities. The related COIL architecture [12] aggregates their two-way dimension reduction in a sum without explicit weighting and feeds the sum through a single loss function. COIL uses both representation types (namely, CLS and token representations) as index, therefore it is not necessary for any of the components to work standalone. In the ColBERTer architecture, we want to support full retrieval capabilities of the CLS vector as candidate generator. If it fails, the quality of the refinement process does not matter anymore. Therefore, we study:

我们的第一项研究旨在理解 CLS 检索能力和词元细化能力之间的关系。相关的 COIL 架构 [12] 在没有明确加权的情况下对其双向降维结果进行求和聚合，并通过单一损失函数处理该和。COIL 使用两种表示类型（即 CLS 和词元表示）作为索引，因此任何一个组件都不需要独立工作。在 ColBERTer 架构中，我们希望支持 CLS 向量作为候选生成器的完整检索能力。如果它失败了，那么细化过程的质量就不再重要了。因此，我们研究：

RQ1 Which aggregation and training regime works best for combined retrieval and refinement capabilities of ColBERTer?

RQ1 哪种聚合和训练机制最适合 ColBERTer 的组合检索和细化能力？

<!-- Media -->

Table 2: Analysis of the bag of whole-words $\left( {\mathrm{{BOW}}}^{2}\right)$ and con-textualized stopword training of ColBERTer (CLS dim: 128, token dim: 32; Workflow Q) using different multi-task loss parameters.

表 2：使用不同多任务损失参数对 ColBERTer（CLS 维度：128，词元维度：32；工作流 Q）的全词袋 $\left( {\mathrm{{BOW}}}^{2}\right)$ 和上下文停用词训练的分析。

<table><tr><td/><td>Train Loss</td><td/><td rowspan="2">BOW ${}^{2}$ Vectors #Saved % Stop.</td><td rowspan="2"/><td colspan="2">DL’19+20</td><td colspan="2">$\mathbf{{DEV}}$</td></tr><tr><td/><td>${\alpha }_{b}{\alpha }_{CLS}{\alpha }_{CS}$</td><td/><td/><td>R@1K</td><td>MRR@10 R@1K</td><td/></tr><tr><td colspan="9">BOW2only</td></tr><tr><td>1</td><td>0.5</td><td>0</td><td>43.2</td><td>0 %</td><td>.731</td><td>.815</td><td>.387</td><td>.963</td></tr><tr><td>21</td><td>0.1</td><td>0</td><td>43.2</td><td>0 %</td><td>.736</td><td>.806</td><td>.387</td><td>.960</td></tr><tr><td colspan="9">${\mathrm{{BOW}}}^{2} +$ Contextualized Stopwords</td></tr><tr><td>31</td><td>0.5</td><td>1</td><td>29.1</td><td>33 %</td><td>.731</td><td>.811</td><td>.382</td><td>.965</td></tr><tr><td>41</td><td>0.1</td><td>1</td><td>27.8</td><td>36 %</td><td>.729</td><td>.802</td><td>.385</td><td>.960</td></tr><tr><td>51</td><td>0.1</td><td>0.75</td><td>30.9</td><td>29 %</td><td>.730</td><td>.805</td><td>.387</td><td>.961</td></tr><tr><td>61</td><td>0.1</td><td>0.5</td><td>36.7</td><td>15 %</td><td>.725</td><td>.806</td><td>.387</td><td>.962</td></tr></table>

<table><tbody><tr><td></td><td>训练损失</td><td></td><td rowspan="2">词袋（BOW） ${}^{2}$ 向量 #已保存 % 停用。</td><td rowspan="2"></td><td colspan="2">DL19 + 20</td><td colspan="2">$\mathbf{{DEV}}$</td></tr><tr><td></td><td>${\alpha }_{b}{\alpha }_{CLS}{\alpha }_{CS}$</td><td></td><td></td><td>1000项召回率（R@1K）</td><td>前10项平均倒数排名（MRR@10） 1000项召回率（R@1K）</td><td></td></tr><tr><td colspan="9">仅使用词袋2（BOW2only）</td></tr><tr><td>1</td><td>0.5</td><td>0</td><td>43.2</td><td>0 %</td><td>.731</td><td>.815</td><td>.387</td><td>.963</td></tr><tr><td>21</td><td>0.1</td><td>0</td><td>43.2</td><td>0 %</td><td>.736</td><td>.806</td><td>.387</td><td>.960</td></tr><tr><td colspan="9">${\mathrm{{BOW}}}^{2} +$ 上下文停用词</td></tr><tr><td>31</td><td>0.5</td><td>1</td><td>29.1</td><td>33 %</td><td>.731</td><td>.811</td><td>.382</td><td>.965</td></tr><tr><td>41</td><td>0.1</td><td>1</td><td>27.8</td><td>36 %</td><td>.729</td><td>.802</td><td>.385</td><td>.960</td></tr><tr><td>51</td><td>0.1</td><td>0.75</td><td>30.9</td><td>29 %</td><td>.730</td><td>.805</td><td>.387</td><td>.961</td></tr><tr><td>61</td><td>0.1</td><td>0.5</td><td>36.7</td><td>15 %</td><td>.725</td><td>.806</td><td>.387</td><td>.962</td></tr></tbody></table>

<!-- Media -->

To isolate the CLS retrieval performance for workflow Q (dense CLS retrieval,followed by ${\mathrm{{BOW}}}^{2}$ storage refinement) we compare different training and aggregation strategies with ColBERTer's CLS retrieval vs. re-ranking the candidate set retrieved by a stan-dalone TAS-Balanced retriever in Table 1. Using COIL's aggregation and training approach (by fixing $\sigma \left( \gamma \right)  = {0.5}$ in Eq. 9 and setting ${\alpha }_{CLS} = 0$ ) we observe in line 1 that the CLS retrieval component fails substantially, compared to utilizing TAS-B. We postulate that this happens, as the token refinement component is more capable in determining relevance and therefore it dominates the changes in gradients, which minimizes the standalone capabilities of CLS retrieval. Now, with our proposed multi-task and learned score aggregation (lines 2-5) we observe much better CLS retrieval performance. While it still lacks a bit behind TAS-B in recall, these deficiencies do not manifest itself after refining the token scores for top-10 results in both TREC-DL and MSMARCO DEV. We selected the best performing setting in line 4 for our future experiments.

为了分离工作流Q（密集CLS检索，随后进行${\mathrm{{BOW}}}^{2}$存储细化）的CLS检索性能，我们在表1中比较了不同的训练和聚合策略，包括ColBERTer的CLS检索与对独立TAS - 平衡检索器检索到的候选集进行重新排序。使用COIL的聚合和训练方法（通过固定公式9中的$\sigma \left( \gamma \right)  = {0.5}$并设置${\alpha }_{CLS} = 0$），我们在第1行观察到，与使用TAS - B相比，CLS检索组件的性能大幅下降。我们推测出现这种情况是因为词元细化组件在确定相关性方面更有能力，因此它主导了梯度的变化，从而使CLS检索的独立能力最小化。现在，使用我们提出的多任务和学习得分聚合方法（第2 - 5行），我们观察到CLS检索性能有了很大提升。虽然在召回率方面仍略落后于TAS - B，但在TREC - DL和MSMARCO DEV中对前10个结果的词元得分进行细化后，这些不足并不明显。我们选择第4行中性能最佳的设置用于未来的实验。

The next addition in our multi-task framework is the learned removal of stopwords. This adds a third loss function ${\mathcal{L}}_{CS}$ that conflicts with the objective of the main ${\mathcal{L}}_{b}$ loss. Table 2 shows the tradeoff between retained ${\mathrm{{BOW}}}^{2}$ vectors and effectiveness. In lines 1 & 2 we see ColBERTer without the stopword components, here 43 vectors are saved with unique ${\mathrm{{BOW}}}^{2}$ for MSMARCO (compared to 77 for all subword tokens). In lines 3 to 6 we study different loss weighting combinations with CS. While the ratio of removed stop-words is rather sensitive to the selected parameters, the effectivness values largely remain constant for lines 4 to 6 . Based on the MRR value of the DEV set (with the smallest effectiveness change, but still ${29}\%$ removed vectors) we select configuration 5 going forward, although we stress that our approach would also work well with the other settings, and cherry picking parameters is not needed. This setting reduces the number of vectors and thus footprint by a factor of 2.5 compared to ColBERT, while keeping the same top-10 effectiveness (comparing Table 2 line 5 vs. Table 1 line 1 (TAS-B re-ranked).

我们多任务框架的下一个改进是学习去除停用词。这增加了第三个损失函数${\mathcal{L}}_{CS}$，它与主要的${\mathcal{L}}_{b}$损失目标相冲突。表2显示了保留的${\mathrm{{BOW}}}^{2}$向量与有效性之间的权衡。在第1行和第2行中，我们看到没有停用词组件的ColBERTer，这里MSMARCO保存了43个具有唯一${\mathrm{{BOW}}}^{2}$的向量（相比之下，所有子词词元有77个）。在第3行到第6行中，我们研究了使用CS的不同损失权重组合。虽然去除停用词的比例对所选参数相当敏感，但第4行到第6行的有效性值基本保持不变。基于DEV集的MRR值（有效性变化最小，但仍有${29}\%$个向量被去除），我们选择配置5继续进行实验，不过我们强调，我们的方法在其他设置下也能很好地工作，不需要精心挑选参数。与ColBERT相比，此设置将向量数量和占用空间减少了2.5倍，同时保持了相同的前10名有效性（比较表2第5行与表1第1行（TAS - B重新排序））。

Future work could use a conservative loss setting (such as line 6) that does not force a lot of the word removal gates to become zero (so as to not take away capacity from the loss surface for the ranking tasks), followed by the removal words with a non-zero (but still small) threshold during inference.

未来的工作可以使用保守的损失设置（如第6行），该设置不会迫使很多词去除门变为零（以免减少排名任务损失面的容量），然后在推理过程中去除阈值非零（但仍然较小）的词。

Following the ablation of training possibilities, we now turn towards the possible usage scenarios, as laid out in §4.2, and answer:

在对训练可能性进行消融实验之后，我们现在转向§4.2中列出的可能使用场景，并回答：

<!-- Media -->

Table 3: Analysis of the retrieval quality for different query-time retrieval and refinement workflows of ColBERTer with vector dimension of 8 or 1 (Uni-ColBERTer). nDCG and MRR at cutoff10.

表3：对向量维度为8或1的ColBERTer（Uni - ColBERTer）的不同查询时检索和细化工作流的检索质量分析。截断值为10时的nDCG和MRR。

<table><tr><td rowspan="2">Workflow</td><td rowspan="2">Model</td><td colspan="2">DL’19+20</td><td colspan="2">$\mathbf{{DEV}}$</td></tr><tr><td>nDCG R@1K</td><td/><td/><td>MRR R@1K</td></tr><tr><td colspan="6">Retrieval Only Ablation</td></tr><tr><td rowspan="2">1 5 BOW ${}^{2}$ only 2</td><td>ColBERTer (Dim8)</td><td>.323</td><td>.780</td><td>.131</td><td>.895</td></tr><tr><td>Uni-ColBERTer</td><td>.280</td><td>.758</td><td>.122</td><td>.880</td></tr><tr><td rowspan="2">3 CLS only 4</td><td>ColBERTer (Dim8)</td><td>.669</td><td>.795</td><td>.326</td><td>.958</td></tr><tr><td>Uni-ColBERTer</td><td>.674</td><td>.789</td><td>.328</td><td>.958</td></tr><tr><td colspan="6">Single Retrieval > Refinement</td></tr><tr><td rowspan="2">5 3 ${\mathrm{{BOW}}}^{2} >$ CLS 6</td><td>ColBERTer (Dim8)</td><td>.730</td><td>.780</td><td>.373</td><td>.895</td></tr><tr><td>Uni-ColBERTer</td><td>.724</td><td>.673</td><td>.369</td><td>.880</td></tr><tr><td rowspan="2">7 2 CLS $> {\mathrm{{BOW}}}^{2}$ 8</td><td>ColBERTer (Dim8)</td><td>.733</td><td>.795</td><td>.375</td><td>.958</td></tr><tr><td>Uni-ColBERTer</td><td>.727</td><td>.789</td><td>.373</td><td>.958</td></tr><tr><td colspan="6">Hybrid Retrieval & Refinement</td></tr><tr><td rowspan="2">9 1 Merge (2+3) 10</td><td>ColBERTer (Dim8)</td><td>.734</td><td>.873</td><td>.376</td><td>.981</td></tr><tr><td>Uni-ColBERTer</td><td>.728</td><td>.865</td><td>.374</td><td>.979</td></tr></table>

<table><tbody><tr><td rowspan="2">工作流</td><td rowspan="2">模型</td><td colspan="2">DL’19 + 20</td><td colspan="2">$\mathbf{{DEV}}$</td></tr><tr><td>归一化折损累积增益（nDCG），前1000召回率（R@1K）</td><td></td><td></td><td>平均倒数排名（MRR），前1000召回率（R@1K）</td></tr><tr><td colspan="6">仅检索消融实验</td></tr><tr><td rowspan="2">1 5 词袋模型（BOW） ${}^{2}$ 仅 2</td><td>ColBERTer（维度8）</td><td>.323</td><td>.780</td><td>.131</td><td>.895</td></tr><tr><td>统一ColBERTer</td><td>.280</td><td>.758</td><td>.122</td><td>.880</td></tr><tr><td rowspan="2">3 仅CLS 4</td><td>ColBERTer（维度8）</td><td>.669</td><td>.795</td><td>.326</td><td>.958</td></tr><tr><td>统一ColBERTer</td><td>.674</td><td>.789</td><td>.328</td><td>.958</td></tr><tr><td colspan="6">单次检索 > 精调</td></tr><tr><td rowspan="2">5 3 ${\mathrm{{BOW}}}^{2} >$ CLS 6</td><td>ColBERTer（维度8）</td><td>.730</td><td>.780</td><td>.373</td><td>.895</td></tr><tr><td>统一ColBERTer</td><td>.724</td><td>.673</td><td>.369</td><td>.880</td></tr><tr><td rowspan="2">7 2 CLS $> {\mathrm{{BOW}}}^{2}$ 8</td><td>ColBERTer（维度8）</td><td>.733</td><td>.795</td><td>.375</td><td>.958</td></tr><tr><td>统一ColBERTer</td><td>.727</td><td>.789</td><td>.373</td><td>.958</td></tr><tr><td colspan="6">混合检索与精调</td></tr><tr><td rowspan="2">9 1 合并（2 + 3） 10</td><td>ColBERTer（维度8）</td><td>.734</td><td>.873</td><td>.376</td><td>.981</td></tr><tr><td>统一ColBERTer</td><td>.728</td><td>.865</td><td>.374</td><td>.979</td></tr></tbody></table>

<!-- Media -->

RQ2 What is ColBERTer's best indexing and refinement strategy?

RQ2 ColBERTer的最佳索引和优化策略是什么？

This study uses ColBERTer with exact matching with 8 and 1 dimensions (Uni-ColBERTer) for ${\mathrm{{BOW}}}^{2}$ vectors,as these are more likely to be used in an inverted index. The inverted index lookup is performed by our hashed id, with potential but highly unlikely conflicts. Then we follow the approach of COIL and UniCOIL to compute dot products for all entries of a posting list for all exact matches between the query and the inverted index, followed by a summation per document, and subsequent sorting to receive a ranked list.

本研究使用ColBERTer对${\mathrm{{BOW}}}^{2}$向量进行精确匹配，维度分别为8维和1维（单向量ColBERTer，Uni - ColBERTer），因为这些维度更有可能用于倒排索引。倒排索引查找通过我们的哈希ID执行，可能会有冲突，但冲突的可能性极低。然后，我们采用COIL和UniCOIL的方法，对查询与倒排索引之间所有精确匹配的倒排列表的所有条目计算点积，接着对每个文档进行求和，随后排序以得到一个排序列表。

Table 3 presents the results of our study grouped by the type of indexing and retrieval. For all indexing schemes, we use the same trained models. We start with an ablation of only one of the two scoring parts in line 1-4. Unsurprisingly, using only one of the scoring parts of ColBERTer lowers effectiveness. What is surprising, though, is the magnitude of the effectiveness drop of the inverted index only workflow $\Theta$ compared to both using only CLS retrieval (workflow $\Phi$ ) or refining the results with CLS scores (workflow Q). Continuing the results, in the single retrieval then refinement section in line 5-8, we see that once we combine both scoring parts, the underlying indexing approach matters very little at the top-10 effectiveness (comparing lines 5 & 7, as well as lines 6 & 8), only the reduced recall of the ${\mathrm{{BOW}}}^{2}$ indexing is carried over. This a great result for the robustness of our system, showing that it can be deployed in a variety of approaches, and practitioners are not locked into a specific retrieval approach. For example if one has made large investments in an inverted index system, they could build on these investments with Uni-ColBERTer.

表3按索引和检索类型对我们的研究结果进行了分组展示。对于所有索引方案，我们使用相同的训练模型。我们从第1 - 4行开始，对两个评分部分中的仅一个进行消融实验。不出所料，仅使用ColBERTer的一个评分部分会降低有效性。然而，令人惊讶的是，仅使用倒排索引的工作流$\Theta$与仅使用CLS检索（工作流$\Phi$）或使用CLS分数优化结果（工作流Q）相比，有效性下降的幅度很大。继续看结果，在第5 - 8行的单次检索然后优化部分，我们发现，一旦我们将两个评分部分结合起来，底层的索引方法对前10名的有效性影响很小（比较第5行和第7行，以及第6行和第8行），只是${\mathrm{{BOW}}}^{2}$索引的召回率降低会延续下来。这对我们系统的鲁棒性来说是一个很好的结果，表明它可以以多种方法部署，并且从业者不必局限于特定的检索方法。例如，如果有人在倒排索引系统上进行了大量投资，他们可以利用单向量ColBERTer（Uni - ColBERTer）在这些投资的基础上继续发展。

Finally, we investigate a hybrid indexing workflow0, where both index types generate candidates and all candidates are refined with the complimentary scoring part. We observe that the recall does increase compared to only one index, however, these improvements do not manifest themselves in the top-10 effectiveness. Here, the results are very close to the simpler workflows $\mathbf{Q}\& \mathbf{\Theta }$ . Therefore,to keep it simple we continue to use workflow2and would suggest it as the primary way of using ColBERTer, if no previous investments make workflow 3 more attractive.

最后，我们研究了一种混合索引工作流0，其中两种索引类型都生成候选结果，并且所有候选结果都用互补的评分部分进行优化。我们观察到，与仅使用一种索引相比，召回率确实有所提高，然而，这些改进并没有体现在前10名的有效性上。在这里，结果与更简单的工作流$\mathbf{Q}\& \mathbf{\Theta }$非常接近。因此，为了简单起见，我们继续使用工作流2，并建议将其作为使用ColBERTer的主要方式，如果之前没有投资使得工作流3更具吸引力的话。

<!-- Media -->

<!-- figureText: (Hofstätter et al.) TAS-B TAS-B + ColBERT (768 dims) ColBERTer [DimRed + BOW ${}^{2}$ ] ColBERTer [DimRed + BOW ${}^{2} +$ CS] Uni-ColBERTer 0.34 0.36 0.38 0.40 MRR@10 Effectiveness (Khattab et al.) ColBERT (Hofstätter et al.) ColBERT-T2 Index Size Factor ( $\times$ Plaintext Size) ${10}^{3}$ ${10}^{2}$ 10 1 0.30 0.32 -->

<img src="https://cdn.noedgeai.com/0195aefe-9fbd-7011-8ec7-b40c2affab11_6.jpg?x=944&y=224&w=673&h=580&r=0"/>

Figure 3: Tradeoff between storage requirements and effectiveness on MSMARCO Dev. Note the log scale of the $y$ -axis.

图3：MSMARCO开发集上存储需求与有效性之间的权衡。注意$y$轴采用对数刻度。

<!-- Media -->

A general observation in the neural IR community is that more capacity in the number of vector dimensions usually leads to better results, albeit with diminishing returns. To see how our enhanced reduction fit into this assumption, we study:

神经信息检索（IR）领域的一个普遍观察结果是，向量维度数量的增加通常会带来更好的结果，尽管收益会递减。为了了解我们增强的降维方法如何符合这一假设，我们研究：

RQ3 How do different configurations of dimensionality and vector count affect the retrieval quality of ColBERTer?

RQ3 不同的维度和向量数量配置如何影响ColBERTer的检索质量？

We must test whether ColBERTer's reductions of the number of vectors improves effectiveness or reduces costs when compared with merely reducing the number of dimensions. In Figure 3 we show the tradeoff between storage requirements and effectiveness of our model configurations and closely related baselines.

我们必须测试与仅减少维度数量相比，ColBERTer减少向量数量是否能提高有效性或降低成本。在图3中，我们展示了我们的模型配置和密切相关的基线模型在存储需求和有效性之间的权衡。

First, we observe that the results of the single vector TAS-B [16] and multi-vector staged pipeline of TAS-B + ColBERT (ours) form a corridor in which our ColBERTer results are expected to reside. Conforming with the expectations, all ColBERTer results are between the two in terms of effectiveness.

首先，我们观察到单向量TAS - B [16]和TAS - B + ColBERT（我们的模型）的多向量分阶段管道的结果形成了一个通道，我们预期ColBERTer的结果会落在这个通道内。符合预期的是，所有ColBERTer的结果在有效性方面都介于两者之间。

Figure 3 displays 3 ColBERTer reduction configurations for 32, 16, 8, and 1 (Uni-ColBERTer) token vector dimensions. Within each configuration, we observe that increased capacity improves effectiveness at the cost of larger storage. Between configurations, we see that removing half the vectors is more efficient and at the same time equal or even slightly improved effectiveness. Thus, using our enhanced reductions improves the Pareto frontier, compared to just reducing the dimensionality. In the case of Uni-ColBERTer, there is no way of further reducing the dimensionality, so every removed vector enables previously unattainable efficiency gains. Our most efficient Uni-ColBERTer with all $\left( {\mathrm{{BO}}{\mathrm{W}}^{2}}\right.$ and $\left. \mathrm{{CS}}\right)$ reductions enabled reaches parity with the plaintext size it indexes. This includes the dense index which at 128 dimensions roughly takes up $2/3$ of the total space.

图3展示了32、16、8和1（单向量ColBERTer，Uni - ColBERTer）维词向量维度的3种ColBERTer降维配置。在每个配置中，我们观察到增加容量以更大的存储为代价提高了有效性。在不同配置之间，我们发现减少一半的向量更有效，同时有效性相同甚至略有提高。因此，与仅减少维度相比，使用我们增强的降维方法改善了帕累托最优边界。在单向量ColBERTer的情况下，无法进一步减少维度，因此每减少一个向量都能实现以前无法达到的效率提升。我们最有效的单向量ColBERTer，启用了所有$\left( {\mathrm{{BO}}{\mathrm{W}}^{2}}\right.$和$\left. \mathrm{{CS}}\right)$降维，达到了与它所索引的纯文本大小相当的效果。这包括128维的密集索引，其大约占用了总空间的$2/3$。

### 6.2 Comparing to Related Work

### 6.2 与相关工作的比较

Fast and complex developments in neural IR make it increasingly difficult to contrast retrieval models, as numerous factors influence effectiveness, including training data sampling, distillation, and generational training, and it is crucial to also compare systems by their the efficiency. We believe it is important to show that we do not observe substantial differences in effectiveness compared to other systems of similar efficiency and that small deviations of effectiveness should not strongly impact our overall assessment, even if those small differences come out in our favor. With that in mind, we study:

神经信息检索（neural IR）的快速且复杂的发展使得对比检索模型变得越来越困难，因为众多因素会影响检索效果，包括训练数据采样、蒸馏和生成式训练，而且通过效率来比较系统也至关重要。我们认为，重要的是要表明，与其他效率相近的系统相比，我们并未观察到检索效果上的显著差异，并且即使那些微小的效果差异对我们有利，这些小偏差也不应强烈影响我们的整体评估。考虑到这一点，我们开展以下研究：

<!-- Media -->

Table 4: Comparing ColBERTers retrieval effectiveness to related approaches grouped by storage requirements. The storage factor refers to ratio of index to plaintext size of ${3.05}\mathrm{{GB}}$ . * indicates an estimation by us.

表4：按存储要求对相关方法进行分组，比较ColBERTers的检索效果。存储因子指的是索引与${3.05}\mathrm{{GB}}$纯文本大小的比率。*表示我们的估算值。

<table><tr><td rowspan="2" colspan="2">Model</td><td colspan="2">Storage</td><td rowspan="2">Query Latency</td><td rowspan="2">Interpret. Ranking</td><td colspan="2">TREC-DL’19</td><td colspan="2">TREC-DL’20</td><td colspan="2">$\mathbf{{DEV}}$</td></tr><tr><td>Total</td><td>Factor</td><td>nDCG@10</td><td>R@1K</td><td>nDCG@10</td><td>R@1K</td><td>MRR@10</td><td>R@1K</td></tr><tr><td colspan="12">Low Storage Systems (max. 2x Factor)</td></tr><tr><td>1[36]</td><td>BM25 (PISA)</td><td>0.7 GB</td><td>$\times  {0.2}$</td><td>8 ms</td><td>✓</td><td>.501</td><td>.739</td><td>.475</td><td>.806</td><td>.194</td><td>.868</td></tr><tr><td>2[56]</td><td>JPQ</td><td>0.8 GB</td><td>$\times  {0.3}$</td><td>90 ms</td><td>✘</td><td>.677</td><td>-</td><td>-</td><td>-</td><td>.341</td><td>-</td></tr><tr><td>3[26]</td><td>UniCOIL-Tok</td><td>N/A</td><td>N/A</td><td>N/A</td><td>✓</td><td>-</td><td>-</td><td>-</td><td>-</td><td>.315</td><td>-</td></tr><tr><td>4[36]</td><td>UniCOIL-Tok (+docT5query)</td><td>1.4 GB</td><td>$\times  {0.5}$</td><td>37 ms</td><td>✓</td><td>-</td><td>-</td><td>-</td><td>-</td><td>.352</td><td>-</td></tr><tr><td>5$\left\lbrack  {{10},{36}}\right\rbrack$</td><td>SPLADEv2 (PISA)</td><td>4.3 GB</td><td>$\times  {1.4}$</td><td>220 ms</td><td>✘</td><td>.729</td><td>-</td><td>-</td><td>-</td><td>.369</td><td>.979</td></tr><tr><td>6[28]</td><td>DSR-SPLADE + Dense-CLS (Dim 128)</td><td>5 GB</td><td>$\times  {1.6}$</td><td>32 ms</td><td>✘</td><td>.709</td><td>-</td><td>.673</td><td>-</td><td>.344</td><td>-</td></tr><tr><td>7</td><td>Uni-ColBERTer (Dim 1)</td><td>3.3 GB</td><td>$\times  {1.1}$</td><td>55 ms</td><td>✓</td><td>.727</td><td>.761</td><td>.726</td><td>.812</td><td>.373</td><td>.958</td></tr><tr><td>8</td><td>ColBERTer w. EM (Dim 8)</td><td>5.8 GB</td><td>$\times  {1.9}$</td><td>55 ms</td><td>✓</td><td>.732</td><td>.764</td><td>.734</td><td>.819</td><td>.375</td><td>.958</td></tr><tr><td colspan="12">Higher Storage Systems</td></tr><tr><td>9 [12]</td><td>COIL (Dim 128, 8)</td><td>${12.5}^{ * }{\mathrm{{GB}}}^{ * }$</td><td>$\times  {4.1}^{ * }$</td><td>21 ms</td><td>✓</td><td>.694</td><td>-</td><td>-</td><td>-</td><td>.347</td><td>.956</td></tr><tr><td>10 [12]</td><td>COIL (Dim 768, 32)</td><td>54.7 GB*</td><td>$\times  {17.9}$</td><td>41 ms</td><td>✓</td><td>.704</td><td>-</td><td>-</td><td>-</td><td>.355</td><td>.963</td></tr><tr><td>11 [28]</td><td>DSR-SPLADE + Dense-CLS (Dim 256)</td><td>11 GB</td><td>$\times  {3.6}$</td><td>34 ms</td><td>✘</td><td>.711</td><td>-</td><td>.678</td><td>-</td><td>.348</td><td>-</td></tr><tr><td>12 [26,29]</td><td>TCT-ColBERTv2 + UniCOIL (+dT5q)</td><td>14.4 GB*</td><td>$\times  {4.7}^{ * }$</td><td>110 ms</td><td>✓</td><td>-</td><td>-</td><td>-</td><td>-</td><td>.378</td><td>-</td></tr><tr><td>13</td><td>ColBERTer (Dim 16)</td><td>9.9 GB</td><td>$\times  {3.2}$</td><td>51 ms</td><td>✓</td><td>.726</td><td>.782</td><td>.719</td><td>.829</td><td>.383</td><td>.961</td></tr><tr><td>14</td><td>ColBERTer (Dim 32)</td><td>18.8 GB</td><td>$\times  {6.2}$</td><td>51 ms</td><td>✓</td><td>.727</td><td>.781</td><td>.733</td><td>.825</td><td>.387</td><td>.961</td></tr></table>

<table><tbody><tr><td rowspan="2" colspan="2">模型</td><td colspan="2">存储</td><td rowspan="2">查询延迟</td><td rowspan="2">解释. 排名</td><td colspan="2">2019年文本检索会议深度学习评测（TREC-DL’19）</td><td colspan="2">2020年文本检索会议深度学习评测（TREC-DL’20）</td><td colspan="2">$\mathbf{{DEV}}$</td></tr><tr><td>总计</td><td>因子</td><td>前10名归一化折损累积增益（nDCG@10）</td><td>前1000名召回率（R@1K）</td><td>前10名归一化折损累积增益（nDCG@10）</td><td>前1000名召回率（R@1K）</td><td>前10名平均倒数排名（MRR@10）</td><td>前1000名召回率（R@1K）</td></tr><tr><td colspan="12">低存储系统（最大2倍因子）</td></tr><tr><td>1[36]</td><td>二元独立模型25（BM25，比萨实现（PISA））</td><td>0.7GB</td><td>$\times  {0.2}$</td><td>8毫秒</td><td>✓</td><td>.501</td><td>.739</td><td>.475</td><td>.806</td><td>.194</td><td>.868</td></tr><tr><td>2[56]</td><td>联合量化投影查询（JPQ）</td><td>0.8GB</td><td>$\times  {0.3}$</td><td>90毫秒</td><td>✘</td><td>.677</td><td>-</td><td>-</td><td>-</td><td>.341</td><td>-</td></tr><tr><td>3[26]</td><td>统一线圈词元模型（UniCOIL-Tok）</td><td>不适用</td><td>不适用</td><td>不适用</td><td>✓</td><td>-</td><td>-</td><td>-</td><td>-</td><td>.315</td><td>-</td></tr><tr><td>4[36]</td><td>统一线圈词元模型（+文档T5查询（docT5query））</td><td>1.4GB</td><td>$\times  {0.5}$</td><td>37毫秒</td><td>✓</td><td>-</td><td>-</td><td>-</td><td>-</td><td>.352</td><td>-</td></tr><tr><td>5$\left\lbrack  {{10},{36}}\right\rbrack$</td><td>稀疏学习注意力文档嵌入v2（SPLADEv2，比萨实现（PISA））</td><td>4.3GB</td><td>$\times  {1.4}$</td><td>220毫秒</td><td>✘</td><td>.729</td><td>-</td><td>-</td><td>-</td><td>.369</td><td>.979</td></tr><tr><td>6[28]</td><td>动态稀疏表示-稀疏学习注意力文档嵌入 + 密集分类器（维度128）</td><td>5GB</td><td>$\times  {1.6}$</td><td>32毫秒</td><td>✘</td><td>.709</td><td>-</td><td>.673</td><td>-</td><td>.344</td><td>-</td></tr><tr><td>7</td><td>统一ColBERTer（维度1）</td><td>3.3GB</td><td>$\times  {1.1}$</td><td>55毫秒</td><td>✓</td><td>.727</td><td>.761</td><td>.726</td><td>.812</td><td>.373</td><td>.958</td></tr><tr><td>8</td><td>带精确匹配的ColBERTer（维度8）</td><td>5.8GB</td><td>$\times  {1.9}$</td><td>55毫秒</td><td>✓</td><td>.732</td><td>.764</td><td>.734</td><td>.819</td><td>.375</td><td>.958</td></tr><tr><td colspan="12">高存储系统</td></tr><tr><td>9 [12]</td><td>线圈模型（维度128，8）</td><td>${12.5}^{ * }{\mathrm{{GB}}}^{ * }$</td><td>$\times  {4.1}^{ * }$</td><td>21毫秒</td><td>✓</td><td>.694</td><td>-</td><td>-</td><td>-</td><td>.347</td><td>.956</td></tr><tr><td>10 [12]</td><td>线圈模型（维度768，32）</td><td>54.7GB*</td><td>$\times  {17.9}$</td><td>41毫秒</td><td>✓</td><td>.704</td><td>-</td><td>-</td><td>-</td><td>.355</td><td>.963</td></tr><tr><td>11 [28]</td><td>动态稀疏表示-稀疏学习注意力文档嵌入 + 密集分类器（维度256）</td><td>11GB</td><td>$\times  {3.6}$</td><td>34毫秒</td><td>✘</td><td>.711</td><td>-</td><td>.678</td><td>-</td><td>.348</td><td>-</td></tr><tr><td>12 [26,29]</td><td>TCT-ColBERTv2 + 统一线圈模型（+文档T5查询）</td><td>14.4GB*</td><td>$\times  {4.7}^{ * }$</td><td>110毫秒</td><td>✓</td><td>-</td><td>-</td><td>-</td><td>-</td><td>.378</td><td>-</td></tr><tr><td>13</td><td>ColBERTer（维度16）</td><td>9.9GB</td><td>$\times  {3.2}$</td><td>51毫秒</td><td>✓</td><td>.726</td><td>.782</td><td>.719</td><td>.829</td><td>.383</td><td>.961</td></tr><tr><td>14</td><td>ColBERTer（维度32）</td><td>18.8GB</td><td>$\times  {6.2}$</td><td>51毫秒</td><td>✓</td><td>.727</td><td>.781</td><td>.733</td><td>.825</td><td>.387</td><td>.961</td></tr></tbody></table>

<!-- Media -->

RQ4 How does the fully optimized ColBERTer system compare to other end-to-end retrieval approaches?

RQ4 完全优化后的ColBERTer系统与其他端到端检索方法相比表现如何？

Table 4 groups models by our main efficiency focus: the storage requirements, measured as the factor of the plaintext size.

表4根据我们主要关注的效率指标对模型进行了分组：存储需求，以纯文本大小的倍数来衡量。

Low Storage Systems. We find that ColBERTer improves on the existing Pareto frontier compared to other approaches, especially for cases with low storage footprint. Uni-ColBERTer (line 7) especially outperforms previous single-dimension token encoding approaches, while at the same time offering improved transparency with whole-word score attributions. We can further improve the dense retrieval component with a technique similar to JPQ [56] (line 2) to reduce our storage footprint.

低存储系统。我们发现，与其他方法相比，ColBERTer在现有的帕累托前沿（Pareto frontier）上有所改进，特别是在存储占用较低的情况下。Uni - ColBERTer（第7行）尤其优于以往的单维词元编码方法，同时通过全词得分归因提供了更高的透明度。我们可以使用类似于JPQ [56]的技术（第2行）进一步改进密集检索组件，以减少存储占用。

Higher Storage Systems. While 32 dimensions per token sounds small, the resulting storage increase is staggering. ColBERTer outperforms similarly sized architectures as well, but a fair comparison becomes more difficult than in the low storage systems, as the absolute size differences become much larger. Another curious observation is that larger ColBERTer models (lines 13 & 14) seem to be slightly faster than our smaller instances (lines $7\& 8$ ). We believe this is due to our non-optimized python code to lookup the top- 1000 token storage memory locations per query,which takes ${10}\mathrm{\;{ms}}$ for ColBERTer without exact matching and ${15}\mathrm{\;{ms}}$ for ColBERTer with exact matching as there we need to access 2 locations per passage (one for the values and one for the ids). There is potential for lower-level optimizations in future work.

高存储系统。虽然每个词元32维听起来不多，但由此导致的存储增加却十分惊人。ColBERTer也优于类似规模的架构，但与低存储系统相比，进行公平比较变得更加困难，因为绝对规模差异变得更大。另一个有趣的观察结果是，较大的ColBERTer模型（第13和14行）似乎比我们较小的实例（$7\& 8$行）稍快。我们认为这是由于我们未优化的Python代码用于查找每个查询的前1000个词元存储内存位置，对于没有精确匹配的ColBERTer需要${10}\mathrm{\;{ms}}$，对于有精确匹配的ColBERTer需要${15}\mathrm{\;{ms}}$，因为在这种情况下我们需要访问每个段落的2个位置（一个用于值，一个用于ID）。未来的工作有进行底层优化的潜力。

### 6.3 Out-of-Domain Robustness

### 6.3 跨领域鲁棒性

In this section we evaluate the zero-shot performance of our ColBERTer architecture, when it is applied on retrieval collections from domains outside the training data to answer:

在本节中，我们评估了ColBERTer架构在应用于训练数据之外领域的检索集合时的零样本性能，以回答：

RQ5 How robust is ColBERTer when applied out of domain?

RQ5 ColBERTer在跨领域应用时的鲁棒性如何？

Our main aim is to present an analysis grounded in robust evaluation [50, 58] that does not fall for common problematic shortcuts in IR evaluation like influence of effect sizes $\left\lbrack  {{11},{52}}\right\rbrack$ ,relying on too shallow pooled collections $\left\lbrack  {2,{31},{53}}\right\rbrack$ ,not accounting for pool bias in old collections $\left\lbrack  {5,{40},{41}}\right\rbrack$ ,and aggregating metrics over different collections which are not comparable [45]. We first describe our evaluation methodology and then discuss our results presented in Figure 4.

我们的主要目标是基于鲁棒评估[50, 58]进行分析，避免陷入信息检索（IR）评估中常见的有问题的捷径，如效应量的影响$\left\lbrack  {{11},{52}}\right\rbrack$、依赖过于浅层的池化集合$\left\lbrack  {2,{31},{53}}\right\rbrack$、未考虑旧集合中的池化偏差$\left\lbrack  {5,{40},{41}}\right\rbrack$以及对不可比的不同集合的指标进行汇总[45]。我们首先描述评估方法，然后讨论图4中呈现的结果。

Methodology. We selected seven datasets from the ir_datasets catalogue [35]: Bio medical (TREC Covid [49, 51], TripClick [39], NFCorpus [4]), Entity centric (DBPedia Entity [14]), informal language (Antique [13], TREC Podcast [21]), news cables (TREC Robust 04 [48]). The datasets are not based on web collections, have at least 50 queries, and importantly contain judgements from both relevant and non-relevant categories. Three datasets are also part of the BEIR [46] catalogue. We choose not to use other datasets from BEIR, as they do not contain non-relevant judgements, which makes it impossible to conduct pooling bias corrections.

方法。我们从ir_datasets目录[35]中选择了七个数据集：生物医学（TREC Covid [49, 51]、TripClick [39]、NFCorpus [4]）、以实体为中心（DBPedia Entity [14]）、非正式语言（Antique [13]、TREC Podcast [21]）、新闻电报（TREC Robust 04 [48]）。这些数据集并非基于网络集合，至少有50个查询，并且重要的是包含相关和不相关类别的判断。其中三个数据集也是BEIR [46]目录的一部分。我们选择不使用BEIR中的其他数据集，因为它们不包含不相关判断，这使得无法进行池化偏差校正。

We follow Sakai [40] to correct our metric measurements for pool bias by observing only measuring effectiveness on judged passages, which means removing all retrieved passages that are not judged and then re-assigning the ranks of the remaining ones. This is in contrast with the default assumption that non-judged passages are not relevant, which naturally favors methods that have been part of the pooling process. Additionally, we follow Soboroff [45] to utilize an effect size analysis that is popular in medicine and social sciences. Soboroff [45] proposed to use this effect size as meta analysis tool to be able to compare statistical significance across different retrieval collections. In this work we combine the evaluation approaches of Sakai [40] and Soboroff [45] for the first time to increase our confidence in results and analysis.

我们遵循Sakai [40]的方法，通过仅在有判断的段落上测量有效性来校正我们的指标测量中的池化偏差，这意味着移除所有未被判断的检索段落，然后重新分配剩余段落的排名。这与默认假设（即未判断的段落不相关）形成对比，默认假设自然有利于参与池化过程的方法。此外，我们遵循Soboroff [45]的方法，利用在医学和社会科学中流行的效应量分析。Soboroff [45]建议将这种效应量用作元分析工具，以便能够比较不同检索集合的统计显著性。在这项工作中，我们首次将Sakai [40]和Soboroff [45]的评估方法结合起来，以提高我们对结果和分析的信心。

<!-- Media -->

<table><tr><td/><td colspan="2">(a) BM25 vs. Uni-ColBERTer (Dim1,BOW ${}^{2} +$ CS)</td><td colspan="2">(b) BM25 vs. ColBERTer (Dim32,BOW ${}^{2}$ )</td><td colspan="2">(c) TAS-B vs. ColBERTer (Dim32,BOW ${}^{2}$ )</td></tr><tr><td/><td>Effect Size</td><td>Weight Mean CI 95%</td><td>Effect Size</td><td>Weight Mean CI 95%</td><td>Effect Size</td><td>WeightMeanCI 95%</td></tr><tr><td>TREC Covid</td><td rowspan="9">Standardized Mean Difference</td><td/><td rowspan="9"> <img src="https://cdn.noedgeai.com/0195aefe-9fbd-7011-8ec7-b40c2affab11_8.jpg?x=749&y=317&w=452&h=279&r=0"/> </td><td/><td rowspan="9">$\begin{array}{lllll} {0.00} & {0.25} & {0.50} & {0.75} & {1.00} \end{array}$ Standardized Mean Difference</td><td>4.7% 0.27 [-0.13,0.67]</td></tr><tr><td>TripClick</td><td/><td/><td>25.1% -0.03 [-0.11, 0.05]</td></tr><tr><td>NFCorpus</td><td/><td/><td>16.3% -0.02 [-0.18, 0.14]</td></tr><tr><td>DBPedia Entity</td><td/><td/><td>${18.4}\% {0.05}\left\lbrack  {-{0.09},{0.18}}\right\rbrack$</td></tr><tr><td>Antique</td><td/><td/><td>12.8% 0.31 [0.11, 0.51]</td></tr><tr><td>TREC Podcast</td><td>${12.3}\% {0.33}\left\lbrack  {{0.05},{0.62}}\right\rbrack$</td><td/><td>8.1% 0.06 [-0.22, 0.34]</td></tr><tr><td>TREC Robust 04</td><td>15.2% 0.04 [-0.13, 0.22]</td><td/><td>14.7% 0.10 [-0.08, 0.28]</td></tr><tr><td>Summary Effect (RE)</td><td>${0.37}\left\lbrack  {{0.15},{0.59}}\right\rbrack$</td><td>0.40 [0.15, 0.65]</td><td>${0.07}\left\lbrack  {-{0.05},{0.18}}\right\rbrack$</td></tr><tr><td/><td/><td/><td/></tr></table>

<table><tbody><tr><td></td><td colspan="2">(a) BM25与Uni - ColBERTer（维度1，词袋模型 ${}^{2} +$ 余弦相似度）对比</td><td colspan="2">(b) BM25与ColBERTer（维度32，词袋模型 ${}^{2}$ ）对比</td><td colspan="2">(c) TAS - B与ColBERTer（维度32，词袋模型 ${}^{2}$ ）对比</td></tr><tr><td></td><td>效应量</td><td>加权均值95%置信区间</td><td>效应量</td><td>加权均值95%置信区间</td><td>效应量</td><td>加权均值95%置信区间</td></tr><tr><td>TREC新冠数据集</td><td rowspan="9">标准化均值差异</td><td></td><td rowspan="9"> <img src="https://cdn.noedgeai.com/0195aefe-9fbd-7011-8ec7-b40c2affab11_8.jpg?x=749&y=317&w=452&h=279&r=0"/> </td><td></td><td rowspan="9">$\begin{array}{lllll} {0.00} & {0.25} & {0.50} & {0.75} & {1.00} \end{array}$ 标准化均值差异</td><td>4.7% 0.27 [-0.13,0.67]</td></tr><tr><td>三元点击数据集</td><td></td><td></td><td>25.1% -0.03 [-0.11, 0.05]</td></tr><tr><td>自然语言处理领域语料库（NFCorpus）</td><td></td><td></td><td>16.3% -0.02 [-0.18, 0.14]</td></tr><tr><td>DBPedia实体数据集</td><td></td><td></td><td>${18.4}\% {0.05}\left\lbrack  {-{0.09},{0.18}}\right\rbrack$</td></tr><tr><td>古董数据集</td><td></td><td></td><td>12.8% 0.31 [0.11, 0.51]</td></tr><tr><td>TREC播客数据集</td><td>${12.3}\% {0.33}\left\lbrack  {{0.05},{0.62}}\right\rbrack$</td><td></td><td>8.1% 0.06 [-0.22, 0.34]</td></tr><tr><td>TREC鲁棒04数据集</td><td>15.2% 0.04 [-0.13, 0.22]</td><td></td><td>14.7% 0.10 [-0.08, 0.28]</td></tr><tr><td>汇总效应（随机效应模型）</td><td>${0.37}\left\lbrack  {{0.15},{0.59}}\right\rbrack$</td><td>0.40 [0.15, 0.65]</td><td>${0.07}\left\lbrack  {-{0.05},{0.18}}\right\rbrack$</td></tr><tr><td></td><td></td><td></td><td></td></tr></tbody></table>

Figure 4: Effect size evaluation of out of domain robustness. We compare three pairings between control vs. treatment. The comparison is dependent on the effect size of each collection. Mean NDCG@10 differences are standardized with the effect size. Confidence intervals are plotted around the standardized mean difference $\blacklozenge$ . The Summary Effect is computed with the Random-Effect (RE) model. We see an overall significant improvement for ColBERTer (Dim1 and Dim32) to BM25.

图4：域外鲁棒性的效应量评估。我们比较了对照组与实验组之间的三对配对。这种比较取决于每个集合的效应量。平均NDCG@10差异通过效应量进行标准化。围绕标准化平均差异绘制了置信区间 $\blacklozenge$。汇总效应使用随机效应（RE）模型计算。我们发现ColBERTer（维度1和维度32）相对于BM25有总体显著的改进。

<!-- Media -->

We take the standardized mean difference (SMD) in nDCG@10 score between a baseline model and our model as the effect. Besides the variability within a collection, we assume a between collection heterogeneity [45]. Following Soboroff [45], we use a random-effect model to estimate the summary effect of our model and each individual effect's contribution, i.e., weight. We use the DerSimonian and Laird estimate [1] to obtain the between collection variance. We illustrate the outcome of our meta-analysis as forest plots. Diamonds $\Phi$ show the effect in each collection and,in turn,in summary. Each effect is accompanied by its ${95}\%$ confidence interval - the grey line. The dotted vertical line marks null effect, i.e., zero SMD in nDCG@10 score between our model and the compared baseline. A confidence interval crossing the null effect line indicates that the corresponding effect is statistically not significant; in all other cases,it contains the actual effect of our model ${95}\%$ of the time.

我们将基线模型和我们的模型之间nDCG@10分数的标准化平均差异（SMD）作为效应。除了集合内的变异性之外，我们假设集合间存在异质性 [45]。遵循Soboroff [45] 的方法，我们使用随机效应模型来估计我们模型的汇总效应以及每个个体效应的贡献，即权重。我们使用DerSimonian和Laird估计 [1] 来获得集合间的方差。我们以森林图的形式展示元分析的结果。菱形 $\Phi$ 显示了每个集合中的效应，进而显示了汇总效应。每个效应都伴随着其 ${95}\%$ 置信区间——灰色线条。虚线垂直线标记为零效应，即我们的模型与比较的基线之间nDCG@10分数的SMD为零。一个跨越零效应线的置信区间表明相应的效应在统计上不显著；在所有其他情况下，它在 ${95}\%$ 的时间内包含了我们模型的实际效应。

As baseline, we utilize BM25 as implemented by Pyserini [27]. We apply our models, trained on MSMARCO, end-to-end in a zero-shot fashion with our default settings for retrieval. We compare a ColBERTer version with 32 token dimensions, as well as Uni-ColBERTer with a single token dimension and exact matching prior.

作为基线，我们使用Pyserini实现的BM25 [27]。我们应用在MSMARCO上训练的模型，以零样本的方式端到端地进行检索，并使用我们的默认设置。我们比较了具有32个词元维度的ColBERTer版本，以及具有单个词元维度和精确匹配先验的Uni - ColBERTer。

Discussion. Figure 4a illustrates the effect of using Uni-ColBERTer instead of BM25 across collections and the corresponding summary effect. Compared to the retrospective approach of hypothesis testing with p-values, confidence intervals are predictive [45]. Considering the TripClick collection, for example, we expect the effect to be between .09 and .25 95% of the time, indicating that we can detect the effect size of .17 SMD at the given confidence level and underlining the significant effectiveness gains using Uni-ColBERTer over BM25. Only on TREC Robust 04 is the small improved difference inside a 95% confidence interval. Overall, by judging the summary effect in Figure 4a, we expect that choosing Uni-ColBERTer over BM25 consistently and significantly improves effectiveness. Similarly, considering Figure 4b, we expect ColBERTer (Dim32) to consistently and significantly outperform BM25. However, comparing the summary effects in Figure 4a and Figure 4b, we expect Uni-ColBERTer and ColBERTer (Dim32) to behave similarly if run against BM25, suggesting to use the more efficient model. We also compare our model to an effective neural dense retriever TAS-B [16], shown to work well out of domain [46]. We report the effect of using ColBERTer (Dim32) vs. TAS-B in Figure 4c, which paints a less clear image than in the other two cases. Most collections overlap inside the ${95}\%$ CI,including the summary effect model,suggesting the models are equally effective. Only the Antique collection is significantly improved by ColBERTer. TREC Covid is a curious case: looking at absolute numbers, one would easily assume a substantial improvement but because it only evaluates 50 queries the confidence interval is very wide. Finally, what does this mean for a deployment decision of ColBERTer vs. TAS-B? We need to consider other aspects, such as transparency. We argue ColBERTer increases transparency over TAS-B as laid out in this paper and it does not show a single collection with significantly worse results, favoring the selection of ColBERTer.

讨论。图4a展示了在各个集合中使用Uni - ColBERTer而非BM25的效应以及相应的汇总效应。与使用p值进行假设检验的回顾性方法相比，置信区间具有预测性 [45]。例如，考虑TripClick集合，我们预计效应在95%的时间内介于0.09和0.25之间，这表明我们可以在给定的置信水平下检测到0.17 SMD的效应量，并强调了使用Uni - ColBERTer相对于BM25在有效性上的显著提升。只有在TREC Robust 04上，微小的改进差异处于95%的置信区间内。总体而言，通过判断图4a中的汇总效应，我们预计选择Uni - ColBERTer而非BM25会持续且显著地提高有效性。类似地，考虑图4b，我们预计ColBERTer（维度32）会持续且显著地优于BM25。然而，比较图4a和图4b中的汇总效应，我们预计如果与BM25对比，Uni - ColBERTer和ColBERTer（维度32）的表现会相似，这表明应使用更高效的模型。我们还将我们的模型与一种有效的神经密集检索器TAS - B [16] 进行了比较，该检索器在域外表现良好 [46]。我们在图4c中报告了使用ColBERTer（维度32）与TAS - B的效应，与前两种情况相比，其结果不太清晰。大多数集合在 ${95}\%$ 置信区间内重叠，包括汇总效应模型，这表明这些模型的有效性相当。只有Antique集合通过ColBERTer有显著改进。TREC Covid是一个有趣的案例：从绝对值来看，人们很容易认为有实质性的改进，但由于它只评估了50个查询，置信区间非常宽。最后，这对于ColBERTer与TAS - B的部署决策意味着什么呢？我们需要考虑其他方面，如透明度。我们认为，正如本文所述，ColBERTer比TAS - B提高了透明度，并且没有一个集合的结果明显更差，因此更倾向于选择ColBERTer。

## 7 CONCLUSION

## 7 结论

In this paper, we proposed ColBERTer, an efficient and effective retrieval model that improves the storage efficiency, the retrieval complexity, and the interpretability of the ColBERT architecture along the effectiveness Pareto frontier. To this end, ColBERTer learns whole-word representations that exclude contextualized stop-words,yielding ${2.5} \times$ fewer vectors than ColBERT while supporting user-friendly query-document scoring patterns at the level of whole words. ColBERTer also uses a multi-task, multi-stage training objective-as well as an optional lexical matching component-that together enable it to aggressively reduce the vector dimension to 1. Extensive empirical evaluation shows that ColBERTer is highly effective on MS MARCO and TREC-DL and highly robust out of domain, while demonstrating highly-competitive storage efficiency with prior dense and sparse models.

在本文中，我们提出了ColBERTer，这是一种高效且有效的检索模型，它沿着有效性帕累托前沿提高了ColBERT架构的存储效率、检索复杂度和可解释性。为此，ColBERTer学习排除上下文停用词的全词表示，与ColBERT相比产生 ${2.5} \times$ 更少的向量，同时支持全词级别的用户友好查询 - 文档评分模式。ColBERTer还使用多任务、多阶段训练目标以及可选的词法匹配组件，这些共同使其能够积极地将向量维度降低到1。广泛的实证评估表明，ColBERTer在MS MARCO和TREC - DL上非常有效，并且在域外具有高度鲁棒性，同时与先前的密集和稀疏模型相比，展示出极具竞争力的存储效率。

---

<!-- Footnote -->

Acknowledgements. This work has received funding from the European Union's Horizon 2020 research and innovation program under grant agreement No 822670 and from the EU Horizon 2020 ITN/ETN project on Domain Specific Systems for Information Extraction and Retrieval (H2020-EU.1.3.1., ID: 860721).

致谢。本研究工作获得了欧盟“地平线2020”研究与创新计划（资助协议编号：822670）以及欧盟“地平线2020”信息技术网络/欧洲培训网络（ITN/ETN）特定领域信息提取与检索系统项目（H2020 - EU.1.3.1.，编号：860721）的资助。

<!-- Footnote -->

---

## REFERENCES

## 参考文献

[1] 2015. Meta-analysis in clinical trials revisited. Contemporary Clinical Trials 45 (2015), 139-145. https://doi.org/10.1016/j.cct.2015.09.002 10th Anniversary Special Issue.

[1] 2015年。重新审视临床试验中的荟萃分析。《当代临床试验》45（2015），139 - 145。https://doi.org/10.1016/j.cct.2015.09.002 十周年特刊。

[2] Negar Arabzadeh, Alexandra Vtyurina, Xinyi Yan, and Charles LA Clarke. 2021. Shallow pooling for sparse labels. arXiv preprint arXiv:2109.00062 (2021).

[2] 内加尔·阿拉布扎德赫（Negar Arabzadeh）、亚历山德拉·维图里娜（Alexandra Vtyurina）、严心怡（Xinyi Yan）和查尔斯·L·A·克拉克（Charles L A Clarke）。2021年。用于稀疏标签的浅层池化。预印本arXiv:2109.00062（2021）。

[3] Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew Mcnamara, Bhaskar Mitra, and Tri Nguyen. 2016. MS MARCO : A Human Generated MAchine Reading COmprehension Dataset. In Proc. of NIPS.

[3] 帕亚尔·巴贾杰（Payal Bajaj）、丹尼尔·坎波斯（Daniel Campos）、尼克·克拉斯韦尔（Nick Craswell）、李邓（Li Deng）、高剑锋（Jianfeng Gao）、刘晓东（Xiaodong Liu）、兰甘·马朱姆德（Rangan Majumder）、安德鲁·麦克纳马拉（Andrew Mcnamara）、巴斯卡尔·米特拉（Bhaskar Mitra）和特里·阮（Tri Nguyen）。2016年。MS MARCO：一个人工生成的机器阅读理解数据集。收录于神经信息处理系统大会（NIPS）论文集。

[4] Vera Boteva, Demian Gholipour, Artem Sokolov, and Stefan Riezler. 2016. A Full-Text Learning to Rank Dataset for Medical Information Retrieval. In Proceedings of the European Conference on Information Retrieval (ECIR) (Padova, Italy). Springer.

[4] 维拉·博特瓦（Vera Boteva）、德米安·戈利波尔（Demian Gholipour）、阿尔乔姆·索科洛夫（Artem Sokolov）和斯特凡·里兹勒（Stefan Riezler）。2016年。一个用于医学信息检索的全文学习排序数据集。收录于欧洲信息检索会议（ECIR）论文集（意大利帕多瓦）。施普林格出版社。

[5] Chris Buckley and Ellen M Voorhees. 2004. Retrieval evaluation with incomplete information. In Proceedings of the 27th annual international ACM SIGIR conference on Research and development in information retrieval. 25-32.

[5] 克里斯·巴克利（Chris Buckley）和埃伦·M·沃里斯（Ellen M Voorhees）。2004年。信息不完整情况下的检索评估。收录于第27届ACM信息检索研究与发展国际年会论文集。25 - 32页。

[6] Carlos Castillo. 2019. Fairness and Transparency in Ranking. SIGIR Forum 52, 2 (jan 2019), 64-71. https://doi.org/10.1145/3308774.3308783

[6] 卡洛斯·卡斯蒂略（Carlos Castillo）。2019年。排序中的公平性与透明度。《SIGIR论坛》52，2（2019年1月），64 - 71。https://doi.org/10.1145/3308774.3308783

[7] Nick Craswell, Bhaskar Mitra, Emine Yilmaz, and Daniel Campos. 2019. Overview of the TREC 2019 deep learning track. In TREC.

[7] 尼克·克拉斯韦尔（Nick Craswell）、巴斯卡尔·米特拉（Bhaskar Mitra）、埃米内·伊尔马兹（Emine Yilmaz）和丹尼尔·坎波斯（Daniel Campos）。2019年。2019年文本检索会议（TREC）深度学习赛道综述。收录于文本检索会议（TREC）论文集。

[8] Nick Craswell, Bhaskar Mitra, Emine Yilmaz, and Daniel Campos. 2020. Overview of the TREC 2020 Deep Learning Track. In TREC.

[8] 尼克·克拉斯韦尔（Nick Craswell）、巴斯卡尔·米特拉（Bhaskar Mitra）、埃米内·伊尔马兹（Emine Yilmaz）和丹尼尔·坎波斯（Daniel Campos）。2020年。2020年文本检索会议（TREC）深度学习赛道综述。收录于文本检索会议（TREC）论文集。

[9] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805 (2018).

[9] 雅各布·德夫林（Jacob Devlin）、张明伟（Ming - Wei Chang）、肯顿·李（Kenton Lee）和克里斯蒂娜·图托纳娃（Kristina Toutanova）。2018年。BERT：用于语言理解的深度双向变换器预训练。预印本arXiv:1810.04805（2018）。

[10] Thibault Formal, Carlos Lassance, Benjamin Piwowarski, and Stéphane Clinchant. 2021. SPLADE v2: Sparse lexical and expansion model for information retrieval. arXiv preprint arXiv:2109.10086 (2021).

[10] 蒂博·福尔马尔（Thibault Formal）、卡洛斯·拉桑斯（Carlos Lassance）、本杰明·皮沃瓦尔斯基（Benjamin Piwowarski）和斯特凡·克兰尚（Stéphane Clinchant）。2021年。SPLADE v2：用于信息检索的稀疏词法与扩展模型。预印本arXiv:2109.10086（2021）。

[11] Norbert Fuhr. 2018. Some Common Mistakes In IR Evaluation, And How They Can Be Avoided. SIGIR Forum 51, 3 (feb 2018), 32-41. https://doi.org/10.1145/ 3190580.3190586

[11] 诺伯特·富尔（Norbert Fuhr）。2018年。信息检索评估中的一些常见错误及避免方法。《SIGIR论坛》51，3（2018年2月），32 - 41。https://doi.org/10.1145/ 3190580.3190586

[12] Luyu Gao, Zhuyun Dai, and Jamie Callan. 2021. COIL: Revisit Exact Lexical Match in Information Retrieval with Contextualized Inverted List. arXiv preprint arXiv:2104.07186 (2021).

[12] 高璐宇（Luyu Gao）、戴竹云（Zhuyun Dai）和杰米·卡兰（Jamie Callan）。2021年。COIL：利用上下文倒排列表重新审视信息检索中的精确词法匹配。预印本arXiv:2104.07186（2021）。

[13] Helia Hashemi, Mohammad Aliannejadi, Hamed Zamani, and W Bruce Croft. 2020. ANTIQUE: A non-factoid question answering benchmark. In Proc. of ECIR.

[13] 赫利亚·哈希米（Helia Hashemi）、穆罕默德·阿里安内贾迪（Mohammad Aliannejadi）、哈米德·扎马尼（Hamed Zamani）和W·布鲁斯·克罗夫特（W Bruce Croft）。2020年。ANTIQUE：一个非事实类问答基准。收录于欧洲信息检索会议（ECIR）论文集。

[14] Faegheh Hasibi, Fedor Nikolaev, Chenyan Xiong, K. Balog, S. E. Bratsberg, Alexander Kotov, and J. Callan. 2017. DBpedia-Entity v2: A Test Collection for Entity Search. Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval (2017).

[14] 法赫赫·哈西比（Faegheh Hasibi）、费多尔·尼古拉耶夫（Fedor Nikolaev）、熊晨彦（Chenyan Xiong）、K·巴洛格（K. Balog）、S·E·布拉茨贝格（S. E. Bratsberg）、亚历山大·科托夫（Alexander Kotov）和J·卡兰（J. Callan）。2017年。DBpedia - 实体v2：一个实体搜索测试集。收录于第40届ACM信息检索研究与发展国际会议论文集（2017）。

[15] Sebastian Hofstätter, Sophia Althammer, Michael Schröder, Mete Sertkan, and Allan Hanbury. 2020. Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation. arXiv:2010.02666 (2020).

[15] 塞巴斯蒂安·霍夫施泰特（Sebastian Hofstätter）、索菲娅·阿尔塔默（Sophia Althammer）、迈克尔·施罗德（Michael Schröder）、梅特·塞尔特坎（Mete Sertkan）和艾伦·汉伯里（Allan Hanbury）。2020年。通过跨架构知识蒸馏改进高效神经排序模型。arXiv:2010.02666 (2020)。

[16] Sebastian Hofstätter, Sheng-Chieh Lin, Jheng-Hong Yang, Jimmy Lin, and Allan Hanbury. 2021. Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling. In Proceedings of the 44rd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '21).

[16] 塞巴斯蒂安·霍夫施泰特（Sebastian Hofstätter）、林圣杰（Sheng-Chieh Lin）、杨政宏（Jheng-Hong Yang）、林吉米（Jimmy Lin）和艾伦·汉伯里（Allan Hanbury）。2021年。通过平衡主题感知采样有效训练高效密集检索器。收录于第44届国际计算机协会信息检索研究与发展会议论文集（SIGIR '21）。

[17] Sebastian Hofstätter, Aldo Lipani, Markus Zlabinger, and Allan Hanbury. 2020. Learning to Re-Rank with Contextualized Stopwords. In Proc. of CIKM.

[17] 塞巴斯蒂安·霍夫施泰特（Sebastian Hofstätter）、阿尔多·利帕尼（Aldo Lipani）、马库斯·兹拉宾格（Markus Zlabinger）和艾伦·汉伯里（Allan Hanbury）。2020年。学习使用上下文停用词进行重排序。收录于信息与知识管理大会论文集（CIKM）。

[18] Samuel Humeau, Kurt Shuster, Marie-Anne Lachaux, and Jason Weston. 2020. Poly-encoders: Transformer Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring. 1Proceedings of the International Conference on Learning Representations (ICLR) 2020.

[18] 塞缪尔·休莫（Samuel Humeau）、库尔特·舒斯特（Kurt Shuster）、玛丽 - 安妮·拉肖（Marie-Anne Lachaux）和杰森·韦斯顿（Jason Weston）。2020年。多编码器：用于快速准确多句子评分的Transformer架构和预训练策略。2020年国际学习表征会议论文集（ICLR）。

[19] Shiyu Ji, Jinjin Shao, and Tao Yang. 2019. Efficient Interaction-based Neural Ranking with Locality Sensitive Hashing. In Proc of. WWW.

[19] 季世玉（Shiyu Ji）、邵金金（Jinjin Shao）和杨涛（Tao Yang）。2019年。基于局部敏感哈希的高效基于交互的神经排序。收录于万维网会议论文集（WWW）。

[20] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2017. Billion-Scale Similarity Search with GPUs. arXiv:1702.08734 (2017).

[20] 杰夫·约翰逊（Jeff Johnson）、马蒂亚斯·杜泽（Matthijs Douze）和埃尔韦·热古（Hervé Jégou）。2017年。使用GPU进行十亿级相似度搜索。arXiv:1702.08734 (2017)。

[21] Rosie Jones, Ben Carterette, Ann Clifton, Maria Eskevich, Gareth JF Jones, Jussi Karlgren, Aasish Pappu, Sravana Reddy, and Yongze Yu. 2021. Trec 2020 podcasts track overview. arXiv preprint arXiv:2103.15953 (2021).

[21] 罗西·琼斯（Rosie Jones）、本·卡特雷特（Ben Carterette）、安·克利夫顿（Ann Clifton）、玛丽亚·埃斯凯维奇（Maria Eskevich）、加雷斯·JF·琼斯（Gareth JF Jones）、尤西·卡尔格伦（Jussi Karlgren）、阿西什·帕普（Aasish Pappu）、斯拉瓦娜·雷迪（Sravana Reddy）和于永泽（Yongze Yu）。2021年。2020年TREC播客赛道概述。arXiv预印本arXiv:2103.15953 (2021)。

[22] Omar Khattab and Matei Zaharia. 2020. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. In Proc. of SIGIR.

[22] 奥马尔·哈塔卜（Omar Khattab）和马泰·扎哈里亚（Matei Zaharia）。2020年。ColBERT：通过基于BERT的上下文后期交互实现高效有效的段落搜索。收录于信息检索研究与发展会议论文集（SIGIR）。

[23] Carlos Lassance, Maroua Maachou, Joohee Park, and Stéphane Clinchant. 2021. A Study on Token Pruning for ColBERT. arXiv:2112.06540 [cs.IR]

[23] 卡洛斯·拉桑斯（Carlos Lassance）、马鲁阿·马乔（Maroua Maachou）、朴珠熙（Joohee Park）和斯特凡·克兰尚（Stéphane Clinchant）。2021年。ColBERT的Token剪枝研究。arXiv:2112.06540 [计算机科学.信息检索（cs.IR）]

[24] Jinhyuk Lee, Mujeen Sung, Jaewoo Kang, and Danqi Chen. 2020. Learning dense representations of phrases at scale. arXiv preprint arXiv:2012.12624 (2020).

[24] 李晋赫（Jinhyuk Lee）、宋穆珍（Mujeen Sung）、姜在宇（Jaewoo Kang）和陈丹琦（Danqi Chen）。2020年。大规模学习短语的密集表示。arXiv预印本arXiv:2012.12624 (2020)。

[25] Patrick Lewis, Barlas Oğuz, Wenhan Xiong, Fabio Petroni, Wen tau Yih, and Sebastian Riedel. 2021. Boosted Dense Retriever. arXiv preprint arXiv:2112.07771 (2021).

[25] 帕特里克·刘易斯（Patrick Lewis）、巴拉斯·奥古兹（Barlas Oğuz）、熊文瀚（Wenhan Xiong）、法比奥·佩特罗尼（Fabio Petroni）、文涛·易（Wen tau Yih）和塞巴斯蒂安·里德尔（Sebastian Riedel）。2021年。增强型密集检索器。arXiv预印本arXiv:2112.07771 (2021)。

[26] Jimmy Lin and Xueguang Ma. 2021. A few brief notes on deepimpact, coil, and a conceptual framework for information retrieval techniques. arXiv preprint arXiv:2106.14807 (2021).

[26] 林吉米（Jimmy Lin）和马学光（Xueguang Ma）。2021年。关于deepimpact、coil以及信息检索技术概念框架的几点简要说明。arXiv预印本arXiv:2106.14807 (2021)。

[27] Jimmy Lin, Xueguang Ma, Sheng-Chieh Lin, Jheng-Hong Yang, Ronak Pradeep, and Rodrigo Nogueira. 2021. Pyserini: A Python Toolkit for Reproducible Information Retrieval Research with Sparse and Dense Representations. In Proc. of SIGIR.

[27] 林吉米（Jimmy Lin）、马学光（Xueguang Ma）、林圣杰（Sheng-Chieh Lin）、杨政宏（Jheng-Hong Yang）、罗纳克·普拉迪普（Ronak Pradeep）和罗德里戈·诺盖拉（Rodrigo Nogueira）。2021年。Pyserini：用于基于稀疏和密集表示的可重现信息检索研究的Python工具包。收录于信息检索研究与发展会议论文集（SIGIR）。

[28] Sheng-Chieh Lin and Jimmy Lin. 2021. Densifying Sparse Representations for Passage Retrieval by Representational Slicing. arXiv preprint arXiv:2112.04666 (2021).

[28] 林圣杰（Sheng-Chieh Lin）和林吉米（Jimmy Lin）。2021年。通过表示切片对段落检索的稀疏表示进行密集化。arXiv预印本arXiv:2112.04666 (2021)。

[29] Sheng-Chieh Lin, Jheng-Hong Yang, and Jimmy Lin. 2020. Distilling Dense Representations for Ranking using Tightly-Coupled Teachers. arXiv:2010.11386 (2020).

[29] 林圣杰（Sheng-Chieh Lin）、杨政宏（Jheng-Hong Yang）和林吉米（Jimmy Lin）。2020年。使用紧密耦合的教师模型蒸馏用于排序的密集表示。arXiv:2010.11386 (2020)。

[30] Wenhao Lu, Jian Jiao, and Ruofei Zhang. 2020. TwinBERT: Distilling Knowledge to Twin-Structured BERT Models for Efficient Retrieval. arXiv:2002.06275 (2020).

[30] 卢文豪（Wenhao Lu）、焦健（Jian Jiao）和张若飞（Ruofei Zhang）。2020年。TwinBERT：将知识蒸馏到双结构BERT模型以实现高效检索。arXiv:2002.06275 (2020)。

[31] Xiaolu Lu, Alistair Moffat, and J Shane Culpepper. 2016. The effect of pooling and evaluation depth on IR metrics. Information Retrieval Journal 19, 4 (2016), 416-445.

[31] 陆晓璐（Xiaolu Lu）、阿利斯泰尔·莫法特（Alistair Moffat）和J·谢恩·卡尔佩珀（J Shane Culpepper）。2016年。合并与评估深度对信息检索指标的影响。《信息检索期刊》（Information Retrieval Journal）第19卷，第4期（2016年），第416 - 445页。

[32] Yi Luan, Jacob Eisenstein, Kristina Toutanova, and Michael Collins. 2020. Sparse, Dense, and Attentional Representations for Text Retrieval. arXiv preprint arXiv:2005.00181 (2020).

[32] 栾毅（Yi Luan）、雅各布·艾森斯坦（Jacob Eisenstein）、克里斯蒂娜·图托纳娃（Kristina Toutanova）和迈克尔·柯林斯（Michael Collins）。2020年。用于文本检索的稀疏、密集和注意力表示。预印本arXiv:2005.00181（2020年）。

[33] Yi Luan, Jacob Eisenstein, Kristina Toutanova, and Michael Collins. 2021. Sparse, Dense, and Attentional Representations for Text Retrieval. Transactions of the Association for Computational Linguistics 9 (2021), 329-345. https://doi.org/10.1162/tacl_a_00369

[33] 栾毅（Yi Luan）、雅各布·艾森斯坦（Jacob Eisenstein）、克里斯蒂娜·图托纳娃（Kristina Toutanova）和迈克尔·柯林斯（Michael Collins）。2021年。用于文本检索的稀疏、密集和注意力表示。《计算语言学协会汇刊》（Transactions of the Association for Computational Linguistics）第9卷（2021年），第329 - 345页。https://doi.org/10.1162/tacl_a_00369

[34] Xueguang Ma, Minghan Li, Kai Sun, Ji Xin, and Jimmy Lin. 2021. Simple and Effective Unsupervised Redundancy Elimination to Compress Dense Vectors for Passage Retrieval. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing.

[34] 马学光（Xueguang Ma）、李明翰（Minghan Li）、孙凯（Kai Sun）、辛吉（Ji Xin）和林吉米（Jimmy Lin）。2021年。用于段落检索的简单有效的无监督冗余消除以压缩密集向量。收录于《2021年自然语言处理经验方法会议论文集》。

[35] Sean MacAvaney, Andrew Yates, Sergey Feldman, Doug Downey, Arman Cohan, and Nazli Goharian. 2021. Simplified Data Wrangling with ir_datasets. In SIGIR.

[35] 肖恩·麦卡瓦尼（Sean MacAvaney）、安德鲁·耶茨（Andrew Yates）、谢尔盖·费尔德曼（Sergey Feldman）、道格·唐尼（Doug Downey）、阿尔曼·科汉（Arman Cohan）和纳兹利·戈哈里安（Nazli Goharian）。2021年。使用ir_datasets简化数据处理。收录于《信息检索研究与发展会议》（SIGIR）。

[36] Joel Mackenzie, Andrew Trotman, and Jimmy Lin. 2021. Wacky weights in learned sparse representations and the revenge of score-at-a-time query evaluation. arXiv preprint arXiv:2110.11540 (2021).

[36] 乔尔·麦肯齐（Joel Mackenzie）、安德鲁·特罗特曼（Andrew Trotman）和林吉米（Jimmy Lin）。2021年。学习到的稀疏表示中的奇异权重与逐次评分查询评估的逆袭。预印本arXiv:2110.11540（2021年）。

[37] Antonio Mallia, Omar Khattab, Torsten Suel, and Nicola Tonellotto. 2021. Learning Passage Impacts for Inverted Indexes. Association for Computing Machinery, New York, NY, USA, 1723-1727. https://doi.org/10.1145/3404835.3463030

[37] 安东尼奥·马利亚（Antonio Mallia）、奥马尔·哈塔布（Omar Khattab）、托尔斯滕·苏尔（Torsten Suel）和尼古拉·托内洛托（Nicola Tonellotto）。2021年。为倒排索引学习段落影响。美国计算机协会，纽约，美国，第1723 - 1727页。https://doi.org/10.1145/3404835.3463030

[38] Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, et al. 2017. Automatic differentiation in PyTorch. In NIPS-W.

[38] 亚当·帕兹克（Adam Paszke）、山姆·格罗斯（Sam Gross）、苏米特·钦塔拉（Soumith Chintala）、格雷戈里·查南（Gregory Chanan）等。2017年。PyTorch中的自动微分。收录于《神经信息处理系统大会研讨会》（NIPS - W）。

[39] Navid Rekabsaz, Oleg Lesota, Markus Schedl, Jon Brassey, and Carsten Eickhoff. 2021. TripClick: The Log Files of a Large Health Web Search Engine. In SIGIR.

[39] 纳维德·雷卡布萨兹（Navid Rekabsaz）、奥列格·莱索塔（Oleg Lesota）、马库斯·舍德尔（Markus Schedl）、乔恩·布拉西（Jon Brassey）和卡斯滕·艾克霍夫（Carsten Eickhoff）。2021年。TripClick：一个大型健康网络搜索引擎的日志文件。收录于《信息检索研究与发展会议》（SIGIR）。

[40] Tetsuya Sakai. 2007. Alternatives to Bpref. In Proc. of SIGIR.

[40] 酒井哲也（Tetsuya Sakai）。2007年。Bpref的替代方案。收录于《信息检索研究与发展会议论文集》（Proc. of SIGIR）。

[41] Tetsuya Sakai. 2008. Comparing Metrics across TREC and NTCIR: The Robustness to System Bias. In Proc. of CIKM.

[41] 酒井哲也（Tetsuya Sakai）。2008年。跨TREC和NTCIR比较指标：对系统偏差的鲁棒性。收录于《信息与知识管理会议论文集》（Proc. of CIKM）。

[42] Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. 2019. Dis-tilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108 (2019).

[42] 维克多·桑（Victor Sanh）、利桑德尔·德比特（Lysandre Debut）、朱利安·肖蒙（Julien Chaumond）和托马斯·沃尔夫（Thomas Wolf）。2019年。DistilBERT：BERT的蒸馏版本，更小、更快、更便宜、更轻量。预印本arXiv:1910.01108（2019年）。

[43] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. 2021. ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction. arXiv preprint arXiv:2112.01488 (2021).

[43] 凯沙夫·桑塔南姆（Keshav Santhanam）、奥马尔·哈塔布（Omar Khattab）、乔恩·萨德 - 法尔孔（Jon Saad - Falcon）、克里斯托弗·波茨（Christopher Potts）和马特·扎哈里亚（Matei Zaharia）。2021年。ColBERTv2：通过轻量级后期交互实现高效检索。预印本arXiv:2112.01488（2021年）。

[44] Mike Schuster and Kaisuke Nakajima. 2012. Japanese and korean voice search. In 2012 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 5149-5152.

[44] 迈克·舒斯特（Mike Schuster）和中岛佳介（Kaisuke Nakajima）。2012年。日语和韩语语音搜索。收录于《2012年电气与电子工程师协会国际声学、语音和信号处理会议》（2012 IEEE International Conference on Acoustics, Speech and Signal Processing，ICASSP）。电气与电子工程师协会，第5149 - 5152页。

[45] Ian Soboroff. 2018. Meta-Analysis for Retrieval Experiments Involving Multiple Test Collections. In Proc. of CIKM.

[45] 伊恩·索博罗夫（Ian Soboroff）。2018年。涉及多个测试集的检索实验的元分析。收录于《信息与知识管理会议论文集》（Proc. of CIKM）。

[46] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021. BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models. arXiv:2104.08663 [cs.IR]

[46] 南丹·塔库尔（Nandan Thakur）、尼尔斯·赖默斯（Nils Reimers）、安德烈亚斯·吕克莱（Andreas Rücklé）、阿比舍克·斯里瓦斯塔瓦（Abhishek Srivastava）和伊琳娜·古列维奇（Iryna Gurevych）。2021年。BEIR：信息检索模型零样本评估的异构基准。arXiv:2104.08663 [计算机科学 - 信息检索（cs.IR）]

[47] Nicola Tonellotto and Craig Macdonald. 2021. Query Embedding Pruning for Dense Retrieval. Association for Computing Machinery, New York, NY, USA, 3453-3457. https://doi.org/10.1145/3459637.3482162

[47] 尼古拉·托内洛托（Nicola Tonellotto）和克雷格·麦克唐纳（Craig Macdonald）。2021年。用于密集检索的查询嵌入剪枝。美国计算机协会，美国纽约州纽约市，3453 - 3457。https://doi.org/10.1145/3459637.3482162

[48] Ellen Voorhees. 2004. Overview of the TREC 2004 Robust Retrieval Track. In TREC.

[48] 艾伦·沃里斯（Ellen Voorhees）。2004年。2004年TREC鲁棒检索赛道概述。见TREC会议论文集。

[49] E. Voorhees, Tasmeer Alam, Steven Bedrick, Dina Demner-Fushman, W. Hersh, Kyle Lo, Kirk Roberts, I. Soboroff, and Lucy Lu Wang. 2020. TREC-COVID: Constructing a Pandemic Information Retrieval Test Collection. ArXiv abs/2005.04474 (2020).

[49] E. 沃里斯（E. Voorhees）、塔斯米尔·阿拉姆（Tasmeer Alam）、史蒂文·贝德里克（Steven Bedrick）、迪娜·德姆纳 - 富什曼（Dina Demner - Fushman）、W. 赫什（W. Hersh）、凯尔·洛（Kyle Lo）、柯克·罗伯茨（Kirk Roberts）、I. 索博罗夫（I. Soboroff）和露西·陆·王（Lucy Lu Wang）。2020年。TREC - COVID：构建大流行信息检索测试集。预印本arXiv:2005.04474 (2020)。

[50] Ellen M Voorhees. 2001. The philosophy of information retrieval evaluation. In Workshop of the cross-language evaluation forum for european languages. Springer, 355-370.

[50] 艾伦·M·沃里斯（Ellen M Voorhees）。2001年。信息检索评估的哲学。见欧洲语言跨语言评估论坛研讨会论文集。施普林格出版社，355 - 370。

[51] Lucy Lu Wang, Kyle Lo, Yoganand Chandrasekhar, Russell Reas, Jiangjiang Yang, Darrin Eide, K. Funk, Rodney Michael Kinney, Ziyang Liu, W. Merrill, P. Mooney, D. Murdick, Devvret Rishi, Jerry Sheehan, Zhihong Shen, B. Stilson, A. Wade, K. Wang, Christopher Wilhelm, Boya Xie, D. Raymond, Daniel S. Weld, Oren Etzioni, and Sebastian Kohlmeier. 2020. CORD-19: The Covid-19 Open Research Dataset. ArXiv (2020).

[51] 露西·陆·王（Lucy Lu Wang）、凯尔·洛（Kyle Lo）、约根南德·钱德拉塞卡尔（Yoganand Chandrasekhar）、拉塞尔·里斯（Russell Reas）、蒋江·杨（Jiangjiang Yang）、达林·艾德（Darrin Eide）、K. 芬克（K. Funk）、罗德尼·迈克尔·金尼（Rodney Michael Kinney）、刘子阳（Ziyang Liu）、W. 梅里尔（W. Merrill）、P. 穆尼（P. Mooney）、D. 默迪克（D. Murdick）、德夫雷特·里希（Devvret Rishi）、杰里·希恩（Jerry Sheehan）、沈志宏（Zhihong Shen）、B. 斯蒂尔森（B. Stilson）、A. 韦德（A. Wade）、K. 王（K. Wang）、克里斯托弗·威廉（Christopher Wilhelm）、谢博雅（Boya Xie）、D. 雷蒙德（D. Raymond）、丹尼尔·S·韦尔德（Daniel S. Weld）、奥伦·埃齐奥尼（Oren Etzioni）和塞巴斯蒂安·科尔迈尔（Sebastian Kohlmeier）。2020年。CORD - 19：新冠肺炎开放研究数据集。预印本arXiv (2020)。

[52] William Webber, Alistair Moffat, and Justin Zobel. 2008. Statistical Power in Retrieval Experimentation. In Proc. of CIKM.

[52] 威廉·韦伯（William Webber）、阿利斯泰尔·莫法特（Alistair Moffat）和贾斯汀·佐贝尔（Justin Zobel）。2008年。检索实验中的统计功效。见CIKM会议论文集。

[53] William Webber and Laurence A. F. Park. 2009. Score Adjustment for Correction of Pooling Bias. In Proc. of SIGIR.

[53] 威廉·韦伯（William Webber）和劳伦斯·A. F. 帕克（Laurence A. F. Park）。2009年。用于校正池化偏差的分数调整。见SIGIR会议论文集。

[54] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. 2020. Transformers: State-of-the-Art Natural Language

[54] 托马斯·沃尔夫（Thomas Wolf）、利桑德尔·德比特（Lysandre Debut）、维克多·桑（Victor Sanh）、朱利安·肖蒙（Julien Chaumond）、克莱门特·德朗格（Clement Delangue）、安东尼·莫伊（Anthony Moi）、皮埃尔里克·西斯塔克（Pierric Cistac）、蒂姆·劳尔特（Tim Rault）、雷米·卢夫（Remi Louf）、摩根·丰托维奇（Morgan Funtowicz）、乔·戴维森（Joe Davison）、山姆·施莱弗（Sam Shleifer）、帕特里克·冯·普拉滕（Patrick von Platen）、克拉拉·马（Clara Ma）、亚辛·杰尔尼（Yacine Jernite）、朱利安·普鲁（Julien Plu）、徐灿文（Canwen Xu）、特文·勒·斯考（Teven Le Scao）、西尔万·古热（Sylvain Gugger）、玛丽亚玛·德拉梅（Mariama Drame）、昆汀·勒霍斯特（Quentin Lhoest）和亚历山大·拉什（Alexander Rush）。2020年。Transformers：最先进的自然语言

Processing. In Proc. EMNLP: System Demonstrations. 38-45.

处理。见EMNLP系统演示会议论文集。38 - 45。

[55] Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed, and Arnold Overwijk. 2020. Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval. arXiv preprint arXiv:2007.00808 (2020).

[55] 熊磊（Lee Xiong）、熊晨彦（Chenyan Xiong）、李烨（Ye Li）、邓国峰（Kwok - Fung Tang）、刘佳琳（Jialin Liu）、保罗·贝内特（Paul Bennett）、朱奈德·艾哈迈德（Junaid Ahmed）和阿诺德·奥弗维克（Arnold Overwijk）。2020年。用于密集文本检索的近似最近邻负对比学习。预印本arXiv:2007.00808 (2020)。

[56] Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Jiafeng Guo, Min Zhang, and Shaoping Ma. 2021. Jointly optimizing query encoder and product quantization to improve retrieval performance. In Proc. of CIKM.

[56] 詹景涛、毛佳欣、刘奕群、郭佳峰、张敏和马少平。2021年。联合优化查询编码器和乘积量化以提高检索性能。见CIKM会议论文集。

[57] Giulio Zhou and Jacob Devlin. 2021. Multi-Vector Attention Models for Deep Re-ranking. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. 5452-5456.

[57] 朱利奥·周（Giulio Zhou）和雅各布·德夫林（Jacob Devlin）。2021年。用于深度重排序的多向量注意力模型。见2021年自然语言处理经验方法会议论文集。5452 - 5456。

[58] Justin Zobel. 1998. How Reliable Are the Results of Large-Scale Information Retrieval Experiments?. In Proc. of SIGIR.

[58] 贾斯汀·佐贝尔（Justin Zobel）。1998年。大规模信息检索实验结果的可靠性如何？见SIGIR会议论文集。