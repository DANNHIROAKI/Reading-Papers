# Multivariate Representation Learning for Information Retrieval

# 信息检索的多元表示学习

Hamed Zamani

哈米德·扎马尼

University of Massachusetts Amherst

马萨诸塞大学阿默斯特分校

United States

美国

zamani@cs.umass.edu

Michael Bendersky

迈克尔·本德尔斯基

Google Research

谷歌研究院

United States

美国

bemike@google.com

## ABSTRACT

## 摘要

Dense retrieval models use bi-encoder network architectures for learning query and document representations. These representations are often in the form of a vector representation and their similarities are often computed using the dot product function. In this paper, we propose a new representation learning framework for dense retrieval. Instead of learning a vector for each query and document, our framework learns a multivariate distribution and uses negative multivariate KL divergence to compute the similarity between distributions. For simplicity and efficiency reasons, we assume that the distributions are multivariate normals and then train large language models to produce mean and variance vectors for these distributions. We provide a theoretical foundation for the proposed framework and show that it can be seamlessly integrated into the existing approximate nearest neighbor algorithms to perform retrieval efficiently. We conduct an extensive suite of experiments on a wide range of datasets, and demonstrate significant improvements compared to competitive dense retrieval models.

密集检索模型使用双编码器网络架构来学习查询和文档表示。这些表示通常采用向量表示的形式，并且它们的相似度通常使用点积函数来计算。在本文中，我们提出了一种用于密集检索的新表示学习框架。我们的框架不是为每个查询和文档学习一个向量，而是学习一个多元分布，并使用负多元KL散度来计算分布之间的相似度。出于简单性和效率的考虑，我们假设这些分布是多元正态分布，然后训练大语言模型为这些分布生成均值和方差向量。我们为所提出的框架提供了理论基础，并表明它可以无缝集成到现有的近似最近邻算法中，以高效地执行检索。我们在广泛的数据集上进行了一系列广泛的实验，并证明与有竞争力的密集检索模型相比有显著改进。

## CCS CONCEPTS

## 计算社区联盟（CCS）概念

- Information systems $\rightarrow$ Document representation; Query representation; Retrieval models and ranking.

- 信息系统 $\rightarrow$ 文档表示；查询表示；检索模型与排序。

## KEYWORDS

## 关键词

Neural information retrieval; dense retrieval; learning to rank; approximate nearest neighbor search

神经信息检索；密集检索；排序学习；近似最近邻搜索

## ACM Reference Format:

## ACM引用格式：

Hamed Zamani and Michael Bendersky. 2023. Multivariate Representation Learning for Information Retrieval. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '23), July 23-27, 2023, Taipei, Taiwan. ACM, New York, NY, USA, 11 pages. https://doi.org/10.1145/3539618.3591740

哈米德·扎马尼和迈克尔·本德尔斯基。2023年。信息检索的多元表示学习。见第46届ACM信息检索研究与发展国际会议（SIGIR '23）论文集，2023年7月23 - 27日，中国台湾台北。美国纽约州纽约市ACM，11页。https://doi.org/10.1145/3539618.3591740

## 1 INTRODUCTION

## 1 引言

Utilizing implicit or explicit relevance labels to learn retrieval models, also called learning-to-rank models, is at the core of information retrieval research. Due to efficiency and even sometimes effectiveness reservations, learning-to-rank models have been mostly used for reranking documents retrieved by an efficient retrieval model, such as BM25 [39]. Therefore, the performance of learning-to-rank models was bounded by the quality of candidate documents selected for reranking. In 2018, the SNRM model [55] has revolutionized

利用隐式或显式相关性标签来学习检索模型（也称为排序学习模型）是信息检索研究的核心。由于效率甚至有时是效果方面的考虑，排序学习模型大多用于对由高效检索模型（如BM25 [39]）检索到的文档进行重排序。因此，排序学习模型的性能受到为重排序所选候选文档质量的限制。2018年，SNRM模型 [55] 彻底改变了

the way we look at learning-to-rank models by arguing that bi-encoder neural networks can be used for representing queries and documents, and document representations can be then indexed for efficient retrieval at query time. The model applied learned latent sparse representations for queries and documents, and indexed the document representations using an inverted index. In 2020, the DPR model [23] demonstrated that even bi-encoder networks with dense representations can be used for efficient retrieval. They took advantage of approximate nearest neighbor algorithms for indexing dense document representations. This category of models, often called dense retrieval models, has attracted much attention and led to state-of-the-art performance on a wide range of retrieval tasks $\left\lbrack  {{18},{24},{37},{53},{57}}\right\rbrack$ .

我们看待排序学习模型的方式，它认为双编码器神经网络可用于表示查询和文档，然后可以对文档表示进行索引，以便在查询时进行高效检索。该模型对查询和文档应用学习到的潜在稀疏表示，并使用倒排索引对文档表示进行索引。2020年，DPR模型 [23] 表明，即使是具有密集表示的双编码器网络也可用于高效检索。它们利用近似最近邻算法对密集文档表示进行索引。这类模型通常被称为密集检索模型，已引起广泛关注，并在广泛的检索任务中取得了最先进的性能 $\left\lbrack  {{18},{24},{37},{53},{57}}\right\rbrack$ 。

Existing sparse and dense representation learning models can be seen as instantiations of Salton et al.'s vector space models [41], i.e., queries and documents are represented using vectors and relevance is defined using vector similarity functions, such as inner product or cosine similarity. Such approaches suffer from a major shortcoming: they do not represent the model's confidence on the learned representations. Inspired by prior work on modeling uncertainty in information retrieval (e.g., $\left\lbrack  {7,8,{52}}\right\rbrack$ ),this paper builds upon the following hypothesis:

现有的稀疏和密集表示学习模型可以看作是萨尔顿（Salton）等人的向量空间模型 [41] 的实例化，即使用向量表示查询和文档，并使用向量相似性函数（如内积或余弦相似度）定义相关性。这种方法存在一个主要缺点：它们没有表示模型对学习到的表示的置信度。受信息检索中不确定性建模的先前工作（例如，$\left\lbrack  {7,8,{52}}\right\rbrack$ ）的启发，本文基于以下假设构建：

Neural retrieval models would benefit from modeling uncertainty (or confidence) in the learned query and document representations.

神经检索模型将从对学习到的查询和文档表示中的不确定性（或置信度）进行建模中受益。

Therefore, we propose a generic framework that represents each query and document using a multivariate distribution, called the MRL framework. In other words, instead of representing queries and documents using $k$ -dimensional vectors,we can assign a probability to each point in this $k$ -dimensional space; the higher the probability, the higher the confidence that the model assigns to each point. For $k = 2$ ,Figure 1(a) depicts the representation of a query and a document in existing single-vector dense retrieval models. ${}^{1}$ On the other hand, Figure 1(b) demonstrates the representations that we envision for queries and documents.

因此，我们提出了一个通用框架，该框架使用多元分布来表示每个查询和文档，称为MRL框架。换句话说，我们不是使用 $k$ 维向量来表示查询和文档，而是可以为这个 $k$ 维空间中的每个点分配一个概率；概率越高，模型对每个点的置信度就越高。对于 $k = 2$ ，图1（a）描绘了现有单向量密集检索模型中查询和文档的表示。 ${}^{1}$ 另一方面，图1（b）展示了我们设想的查询和文档的表示。

To reduce the complexity of the model, we assume that the representations are multivariate normal distributions with a diagonal covariance matrix; meaning that the representation dimensions are orthogonal and independent. With this assumption, we learn two $k$ -dimensional vectors for each query or document: a mean vector and a variance vector. In addition to uncertainty, such probabilistic modeling can implicitly represent breadth of information in queries and documents. For instance, a document that covers multiple topics and potentially satisfies a diverse set of information needs may be represented by a multivariate distribution with large variance values.

为了降低模型的复杂度，我们假设这些表示是具有对角协方差矩阵的多元正态分布；这意味着表示维度是正交且独立的。基于这个假设，我们为每个查询或文档学习两个 $k$ 维向量：一个均值向量和一个方差向量。除了不确定性之外，这种概率建模还可以隐式表示查询和文档中的信息广度。例如，一个涵盖多个主题并可能满足各种信息需求的文档可以用具有大方差值的多元分布来表示。

---

<!-- Footnote -->

${}^{1}$ The third dimension is only used for consistent presentation. One can consider the probability of 1 for one point in the two-dimensional space and zero elsewhere.

${}^{1}$ 第三维仅用于一致呈现。可以考虑二维空间中一个点的概率为1，其他地方为0。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: (a) Representation learning in existing single-vector dense retrieval (b) Representation learning in MRL models -->

<img src="https://cdn.noedgeai.com/0195a507-1150-7ca4-ab73-3cb2f130644c_1.jpg?x=280&y=255&w=1191&h=487&r=0"/>

Figure 1: Existing dense retrieval methods use a vector to represent any input. Figure 1(a) demonstrates example representations they learn for two inputs (e.g., a query and a document). The proposed framework learns multivariate distributions to represent each input, which is depicted in Figure 1(b).

图1：现有的密集检索方法使用向量来表示任何输入。图1（a）展示了它们为两个输入（例如，一个查询和一个文档）学习的示例表示。所提出的框架学习多元分布来表示每个输入，如图1（b）所示。

<!-- Media -->

MRL uses negative multivariate Kullback-Leibler (KL) divergence between query and document representations to compute the relevance scores. We prove that the relevance scores can be computed efficiently by proposing solutions that can be implemented using existing approximate nearest neighbor search algorithms. We also demonstrate that one can simply implement the MRL framework using existing pre-trained large language models, such as BERT [13].

MRL使用查询和文档表示之间的负多元KL散度（Kullback - Leibler divergence）来计算相关性得分。我们通过提出可以使用现有近似最近邻搜索算法实现的解决方案，证明了可以高效地计算相关性得分。我们还证明了可以使用现有的预训练大语言模型（如BERT [13]）简单地实现MRL框架。

We show that an implementation of MRL that uses a single vector with 768 dimensions to represent multivariate representations for each query and document significantly outperforms existing single vector dense retrieval models on several standard text retrieval benchmarks. MRL also often outperforms ColBERTv2 [43], a state-of-the-art multi vector dense retrieval model, while using significantly less storage and having significantly lower query latency. We further demonstrate that MRL also performs effectively in zero-shot settings when applied to unseen domains. Besides, we also demonstrate that the norm of variance vectors learned by MRL are a strong indicator of the retrieval effectiveness and can be used as a pre-retrieval query performance predictor.

我们表明，MRL的一种实现方式，即使用一个768维的单向量来表示每个查询和文档的多元表示，在几个标准文本检索基准上显著优于现有的单向量密集检索模型。MRL通常还优于ColBERTv2 [43] （一种最先进的多向量密集检索模型），同时使用的存储空间显著减少，查询延迟也显著降低。我们进一步证明，MRL在应用于未见领域的零样本设置中也能有效执行。此外，我们还证明，MRL学习到的方差向量的范数是检索有效性的一个强有力指标，可以用作预检索查询性能预测器。

We believe that MRL smooths the path towards developing more advanced probabilistic dense retrieval models and its applications can be extended to recommender systems, conversational systems, and a wide range of retrieval-enhanced machine learning models.

我们相信，MRL为开发更先进的概率密集检索模型铺平了道路，其应用可以扩展到推荐系统、对话系统和广泛的检索增强机器学习模型。

## 2 RELATED WORK

## 2 相关工作

Variance of retrieval performance among different topics has been a long-standing research theme in the information retrieval community. For instance, TREC 2004 Robust Track organizers noted that solely optimizing the average metric aggregates (e.g., MAP) "further improves the effectiveness of the already-effective topics, sometimes at the expense of the poor performers" [49]. Moreover, identifying poorly performing topics is hard, and failure to do so leads to degraded user perception of the retrieval system as "an individual user does not see the average performance of the system, but only the effectiveness of the system on his or her requests" [49]. These insights led the information retrieval community to consider query performance prediction [5] - a notion that certain signals can predict the performance of a search query. Such predictions can be helpful in guiding the retrieval system in taking further actions as needed for more difficult queries, e.g., suggesting alternative query reformulations [1].

不同主题间检索性能的差异一直是信息检索领域的一个长期研究主题。例如，2004年文本检索会议（TREC）的鲁棒性赛道（Robust Track）组织者指出，仅优化平均指标聚合（如平均准确率均值，MAP）“会进一步提高原本检索效果就好的主题的有效性，有时却会牺牲检索效果差的主题的性能” [49]。此外，识别检索效果差的主题很困难，若无法做到这一点，会降低用户对检索系统的评价，因为“单个用户看不到系统的平均性能，只能看到系统对其查询请求的检索效果” [49]。这些观点促使信息检索领域开始考虑查询性能预测 [5] —— 即某些信号可以预测搜索查询的性能这一概念。这样的预测有助于指导检索系统针对更难的查询采取进一步的行动，例如，建议替代的查询改写方式 [1]。

A degree of query ambiguity with respect to the underlying corpus has been shown to be a valuable predictor of poor performance of search queries [11]. Therefore, dealing with retrieval uncertainty has been proposed as a remedy. For instance, Collins-Thompson and Callan [8] propose estimating query uncertainty by repeatedly fitting a Dirichlet distribution over bootstrap samples from the top- $k$ retrieved documents. They show that a Bayesian combination of multiple boostrap samples (which takes into account sample variance) leads to both significantly better retrieval metrics, and better retrieval robustness (less queries hurt by the query expansion methods). In a related vein, Zhu et al. [62] develop a risk-aware language model based on the Dirichlet distribution (as a conjugate prior to the multinomial distribution). They use the variance of the Dirichlet distribution for adjusting the risk in the final ranking score (i.e., revising the relevance estimates downwards in face of high variance).

相对于基础语料库而言，查询的歧义程度已被证明是搜索查询性能不佳的一个重要预测指标 [11]。因此，处理检索的不确定性被提议作为一种解决方案。例如，柯林斯 - 汤普森（Collins - Thompson）和卡兰（Callan）[8] 提出通过对前 $k$ 个检索到的文档进行自助采样（bootstrap samples），并反复拟合狄利克雷分布（Dirichlet distribution）来估计查询的不确定性。他们表明，对多个自助采样进行贝叶斯组合（考虑了样本方差）不仅能显著提高检索指标，还能增强检索的鲁棒性（减少因查询扩展方法而受影响的查询数量）。与此相关，朱（Zhu）等人 [62] 基于狄利克雷分布（作为多项分布的共轭先验）开发了一种风险感知语言模型。他们利用狄利克雷分布的方差来调整最终排名得分中的风险（即，在方差较大时向下修正相关性估计值）。

The idea of risk adjustment inspired by the financial investment literature was further developed by Wang and Zhu into the portfolio theory for information retrieval [52]. Portfolio theory generalizes the probability ranking principle (PRP) by considering both the uncertainty of relevance predictions and correlations between retrieved documents. It also demonstrates that one way to address uncertainty is via diversification [6]. The portfolio theory-based approach to retrieval has since been applied in several domains including recommendation [44], quantum-based information retrieval [63], and computational advertising [59], among others.

受金融投资文献启发的风险调整理念被王（Wang）和朱（Zhu）进一步发展为信息检索的投资组合理论 [52]。投资组合理论通过同时考虑相关性预测的不确定性以及检索到的文档之间的相关性，对概率排序原则（PRP）进行了推广。它还表明，解决不确定性的一种方法是通过多样化 [6]。此后，基于投资组合理论的检索方法已应用于多个领域，包括推荐系统 [44]、基于量子的信息检索 [63] 和计算广告 [59] 等。

While, as this prior research shows, there has been an extensive exploration of risk and mean-variance trade-offs in the statistical language models for information retrieval, there has been so far much less discussion of these topics in the context of neural (aka dense) models for retrieval. As a notable exception to this, Cohen et al. [7] recently proposed a Bayesian neural relevance model, where a posterior is approximated using Monte Carlo sampling based on drop-out [14]. A similar approach was proposed by Penha and Hauff [34] in the context of conversational search. These approaches, which employ variational inference at training time, can only be applied for reranking. In contrast, in this work we model uncertainty at the level of query and document representations, and demonstrate how such representations can be efficiently and effectively used for retrieval using any of the existing approximate nearest neighbor methods.

正如先前的研究所示，在用于信息检索的统计语言模型中，已经对风险和均值 - 方差权衡进行了广泛的探索，但到目前为止，在基于神经网络（即密集型）的检索模型背景下，对这些主题的讨论还少得多。值得注意的一个例外是，科恩（Cohen）等人 [7] 最近提出了一种贝叶斯神经相关性模型，其中使用基于随机失活（drop - out）的蒙特卡罗采样 [14] 来近似后验分布。彭哈（Penha）和豪夫（Hauff）[34] 在对话式搜索的背景下也提出了类似的方法。这些在训练时采用变分推理的方法只能用于重排序。相比之下，在这项工作中，我们在查询和文档表示层面上对不确定性进行建模，并展示了如何使用现有的任何近似最近邻方法，高效且有效地利用这些表示进行检索。

Outside the realm of information retrieval research, various forms of representations that go beyond Euclidean vectors have been explored, including order embeddings [46], hyperbolic embed-dings [31], and probabilistic box embeddings [47], Such representations have been shown to be effective for various NLP tasks that involve modeling complex relationship or structures. Similar to our work, Vilnis and McCallum [48] used Gaussian distributions for representation learning by proposing Gaussian embeddings for words. In this work, we focus on query and document representations in the retrieval setting.

在信息检索研究领域之外，已经探索了各种超越欧几里得向量的表示形式，包括顺序嵌入（order embeddings）[46]、双曲嵌入（hyperbolic embeddings）[31] 和概率盒嵌入（probabilistic box embeddings）[47]。这些表示形式已被证明对涉及复杂关系或结构建模的各种自然语言处理任务是有效的。与我们的工作类似，维尔尼斯（Vilnis）和麦卡勒姆（McCallum）[48] 通过提出词的高斯嵌入（Gaussian embeddings），使用高斯分布进行表示学习。在这项工作中，我们专注于检索场景下的查询和文档表示。

Some prior work, as a way to achieve semantically richer representations, model queries and documents using a combination of multiple vectors $\left\lbrack  {{26},{43},{61}}\right\rbrack$ . While such representations were shown to lead to better retrieval effectiveness, they do come at significant computational and storage costs. We demonstrate that our multivariate distribution representations are significantly more efficient than multi-vector ones, while attaining comparable or better performance on a wide range of collections.

一些先前的工作为了实现语义更丰富的表示，使用多个向量 $\left\lbrack  {{26},{43},{61}}\right\rbrack$ 的组合来对查询和文档进行建模。虽然这些表示形式被证明能提高检索效果，但它们确实会带来显著的计算和存储成本。我们证明了我们的多元分布表示比多向量表示效率高得多，同时在广泛的数据集上能达到相当或更好的性能。

## 3 THE MRL FRAMEWORK

## 3 MRL框架

Existing single vector dense retrieval models uses a $k$ -dimensional latent vector to represent each query or each query token $\lbrack {17},{23}$ , ${53},{57}\rbrack$ . We argue that these dense retrieval models can benefit from modeling uncertainty in representation learning. That means the model may produce a representation for a clear navigational query with high confidence, while it may have lower confidence in representing an ambiguous query. Same argument applies to the documents. However, the existing frameworks for dense retrieval do not model such confidence or uncertainty in representations. In this paper, we present MRL- a generic framework for modeling uncertainty in representation learning for information retrieval. MRL models each query (or document) using a $k$ -variate distribution - a group of $k$ continuous random variables using which we can compute the probability of any given vector in a $k$ -dimensional space being a representation of the input query (or document). Formally, MRL encodes each query $q$ and each document $d$ as follows:

现有的单向量密集检索模型使用一个$k$维的潜在向量来表示每个查询或每个查询词元 $\lbrack {17},{23}$ , ${53},{57}\rbrack$。我们认为，这些密集检索模型可以从表示学习中的不确定性建模中受益。这意味着模型可能会以高置信度为明确的导航式查询生成一个表示，而在表示模糊查询时可能置信度较低。同样的观点也适用于文档。然而，现有的密集检索框架并未对表示中的这种置信度或不确定性进行建模。在本文中，我们提出了MRL——一个用于信息检索表示学习中不确定性建模的通用框架。MRL使用一个$k$元分布（一组$k$个连续随机变量）来对每个查询（或文档）进行建模，利用该分布我们可以计算$k$维空间中任何给定向量作为输入查询（或文档）表示的概率。形式上，MRL按如下方式对每个查询$q$和每个文档$d$进行编码：

$$
\mathrm{Q} = {\left( {Q}_{1},{Q}_{2},\cdots ,{Q}_{k}\right) }^{\top } = {\operatorname{EncodER}}_{\mathrm{Q}}\left( q\right) 
$$

$$
\mathrm{D} = {\left( {D}_{1},{D}_{2},\cdots ,{D}_{k}\right) }^{\top } = {\operatorname{EncodER}}_{\mathrm{D}}\left( d\right)  \tag{1}
$$

where ${\mathrm{{ENCODER}}}_{\mathrm{Q}}$ and ${\mathrm{{ENCODER}}}_{\mathrm{D}}$ respectively denote query and document encoders. Each ${Q}_{i}$ and ${D}_{i}$ is a random variable; thus $\mathrm{Q}$ and D are $k$ -variate distributions representing the query and the document. The superscript ${}_{\mathrm{T}}$ denotes the transpose of the vector.

其中${\mathrm{{ENCODER}}}_{\mathrm{Q}}$和${\mathrm{{ENCODER}}}_{\mathrm{D}}$分别表示查询编码器和文档编码器。每个${Q}_{i}$和${D}_{i}$都是一个随机变量；因此$\mathrm{Q}$和D是表示查询和文档的$k$元分布。上标${}_{\mathrm{T}}$表示向量的转置。

In this paper,we assume that $\mathrm{Q}$ and $\mathrm{D}$ both are $k$ -variate normal distributions. The reasons for this assumption are: (1) we can define each $k$ -variate normal distribution using a mean vector and a covariance matrix, (2) lower order distributions (i.e., any combination of the $k$ dimensions) and conditional distributions are also normal, which makes it easily extensible, (3) linear functions of multivariate normal distributions are also multivariate normal, leading to simple aggregation approaches. A $k$ -variate normal distribution can be represented using a $k \times  1$ mean vector $\mathbf{M} = {\left( {\mu }_{1},{\mu }_{2},\cdots ,{\mu }_{k}\right) }^{\top }$ and a $k \times  k$ covariance matrix $\sum$ as follows: ${\mathcal{N}}_{k}\left( {\mathbf{M},\sum }\right)$ . We compute the representations as $k$ independent normal distributions,thus the covariance matrix is diagonal. Therefore, our representations are modeled as follows:

在本文中，我们假设$\mathrm{Q}$和$\mathrm{D}$均为$k$元正态分布。做出这一假设的原因如下：（1）我们可以使用一个均值向量和一个协方差矩阵来定义每个$k$元正态分布；（2）低阶分布（即$k$个维度的任何组合）和条件分布也是正态分布，这使其易于扩展；（3）多元正态分布的线性函数也是多元正态分布，从而产生简单的聚合方法。一个$k$元正态分布可以使用一个$k \times  1$维均值向量$\mathbf{M} = {\left( {\mu }_{1},{\mu }_{2},\cdots ,{\mu }_{k}\right) }^{\top }$和一个$k \times  k$维协方差矩阵$\sum$表示如下：${\mathcal{N}}_{k}\left( {\mathbf{M},\sum }\right)$。我们将表示计算为$k$个独立的正态分布，因此协方差矩阵是对角矩阵。因此，我们的表示建模如下：

$$
{\mathcal{N}}_{k}\left\lbrack  {\left( \begin{matrix} {\mu }_{1} \\  {\mu }_{2} \\  \vdots \\  {\mu }_{k} \end{matrix}\right) ,\left( \begin{matrix} {\sigma }_{1}^{2} & 0 & \ldots & 0 \\  0 & {\sigma }_{2}^{2} & \ldots & 0 \\  \vdots & &  \ddots  & \\  0 & 0 & \ldots & {\sigma }_{k}^{2} \end{matrix}\right) }\right\rbrack   \tag{2}
$$

With this formulation, we can re-write Equation (1) as follows:

通过这种公式化，我们可以将方程（1）重写如下：

$$
\mathrm{Q} \sim  {\mathcal{N}}_{k}\left( {{\mathbf{M}}_{Q},{\mathbf{\sum }}_{Q}}\right) ,\;{\mathbf{M}}_{Q},{\mathbf{\sum }}_{Q} = {\operatorname{EncoDER}}_{\mathrm{Q}}\left( q\right) 
$$

$$
\mathrm{D} \sim  {\mathcal{N}}_{k}\left( {{\mathbf{M}}_{D},{\mathbf{\sum }}_{D}}\right) ,\;{\mathbf{M}}_{D},{\mathbf{\sum }}_{D} = {\operatorname{EncoDER}}_{\mathrm{D}}\left( d\right)  \tag{3}
$$

where ${\mathbf{M}}_{Q} = {\left( {\mu }_{q1},{\mu }_{q2},\cdots ,{\mu }_{qk}\right) }^{\top },{\mathbf{\sum }}_{Q} = {\left( {\sigma }_{q1}^{2},{\sigma }_{q2}^{2},\cdots ,{\sigma }_{qk}^{2}\right) }^{\top } \times  {I}_{k}$ , ${\mathbf{M}}_{D} = {\left( {\mu }_{d1},{\mu }_{d2},\cdots ,{\mu }_{dk}\right) }^{\top }$ ,and ${\mathbf{\sum }}_{D} = {\left( {\sigma }_{d1}^{2},{\sigma }_{d2}^{2},\cdots ,{\sigma }_{dk}^{2}\right) }^{\top } \times  {I}_{k}.$ Therefore, it is safe to claim that MRL uses large language models to learn a $k$ -dimensional mean vector and a $k$ -dimensional variance vector for representing each input query and document. This representation for a query and a document is plotted in Figure 1(b) $\left( {k = 2\text{in the plot).}}\right)$

其中${\mathbf{M}}_{Q} = {\left( {\mu }_{q1},{\mu }_{q2},\cdots ,{\mu }_{qk}\right) }^{\top },{\mathbf{\sum }}_{Q} = {\left( {\sigma }_{q1}^{2},{\sigma }_{q2}^{2},\cdots ,{\sigma }_{qk}^{2}\right) }^{\top } \times  {I}_{k}$、${\mathbf{M}}_{D} = {\left( {\mu }_{d1},{\mu }_{d2},\cdots ,{\mu }_{dk}\right) }^{\top }$和${\mathbf{\sum }}_{D} = {\left( {\sigma }_{d1}^{2},{\sigma }_{d2}^{2},\cdots ,{\sigma }_{dk}^{2}\right) }^{\top } \times  {I}_{k}.$。因此，可以有把握地说，MRL使用大语言模型来学习一个$k$维的均值向量和一个$k$维的方差向量，以表示每个输入查询和文档。查询和文档的这种表示如图1（b）所示 $\left( {k = 2\text{in the plot).}}\right)$

Using the flexible modeling offered by the MRL framework, we can compute the probability of any $k$ dimensional vector representing each query or document. In more detail, the probability of vector $\mathbf{x} = {\left( {x}_{1},{x}_{2},\cdots ,{x}_{k}\right) }^{\top }$ being generated from the $k$ -variate normal distribution in Equation (2) is equal to:

利用MRL框架提供的灵活建模方法，我们可以计算表示每个查询或文档的任意$k$维向量的概率。更详细地说，向量$\mathbf{x} = {\left( {x}_{1},{x}_{2},\cdots ,{x}_{k}\right) }^{\top }$由方程(2)中的$k$元正态分布生成的概率等于：

$$
p\left( \mathbf{x}\right)  = \frac{1}{{\left( 2\pi \right) }^{\frac{k}{2}}\det {\left( \mathbf{\sum }\right) }^{\frac{1}{2}}}\exp \left( {-\frac{1}{2}{\left( \mathbf{x} - \mathbf{M}\right) }^{\top }{\mathbf{\sum }}^{-1}\left( {\mathbf{x} - \mathbf{M}}\right) }\right)  \tag{4}
$$

where $\det \left( \cdot \right)$ denotes the determinant of the given matrix. This formulation enables us to compute the probability of any $k$ -dimensional vector being a representation for each query and document.

其中$\det \left( \cdot \right)$表示给定矩阵的行列式。这种公式使我们能够计算任意$k$维向量作为每个查询和文档表示的概率。

Once the queries and documents are represented, MRL computes the relevance score for a pair of query and document using the negative Kullback-Leibler divergence (negative KL divergence) between two $k$ -variate distributions: $- {\mathrm{{KLD}}}_{k}\left( {\mathrm{Q}\parallel \mathrm{D}}\right)$ . The KL divergence can be computed as follows:

一旦表示出查询和文档，MRL会使用两个$k$元分布之间的负库尔贝克 - 莱布勒散度（负KL散度）$- {\mathrm{{KLD}}}_{k}\left( {\mathrm{Q}\parallel \mathrm{D}}\right)$来计算查询 - 文档对的相关性得分。KL散度的计算方法如下：

$$
{\mathrm{{KLD}}}_{k}\left( {\mathrm{Q}\parallel \mathrm{D}}\right)  = {\mathbb{E}}_{\mathrm{Q}}\left\lbrack  {\log \frac{\mathrm{Q}}{\mathrm{D}}}\right\rbrack   = {\mathbb{E}}_{\mathrm{Q}}\left\lbrack  {\log \mathrm{Q} - \log \mathrm{D}}\right\rbrack  
$$

$$
 = \frac{1}{2}{\mathbb{E}}_{\mathrm{Q}}\left\lbrack  {-\log \det \left( {\mathbf{\sum }}_{Q}\right)  - {\left( \mathbf{x} - {\mathbf{M}}_{Q}\right) }^{\top }{\mathbf{\sum }}_{Q}^{-1}\left( {\mathbf{x} - {\mathbf{M}}_{Q}}\right) }\right. 
$$

$$
\left. {+\log \det \left( {\mathbf{\sum }}_{D}\right)  + {\left( \mathbf{x} - {\mathbf{M}}_{D}\right) }^{\top }{\mathbf{\sum }}_{D}^{-1}\left( {\mathbf{x} - {\mathbf{M}}_{D}}\right) }\right\rbrack  
$$

$$
 = \frac{1}{2}\log \frac{\det \left( {\mathbf{\sum }}_{D}\right) }{\det \left( {\mathbf{\sum }}_{Q}\right) } - \frac{1}{2}{\mathbb{E}}_{Q}\left\lbrack  {{\left( \mathbf{x} - {\mathbf{M}}_{Q}\right) }^{\top }{\mathbf{\sum }}_{Q}^{-1}\left( {\mathbf{x} - {\mathbf{M}}_{Q}}\right) }\right\rbrack  
$$

$$
 + \frac{1}{2}{\mathbb{E}}_{\mathrm{Q}}\left\lbrack  {{\left( \mathbf{x} - {\mathbf{M}}_{D}\right) }^{\top }{\mathbf{\sum }}_{D}^{-1}\left( {\mathbf{x} - {\mathbf{M}}_{D}}\right) }\right\rbrack  
$$

(5)

Since ${\left( \mathbf{x} - {\mathbf{M}}_{Q}\right) }^{\top }{\mathbf{\sum }}_{Q}^{-1}\left( {\mathbf{x} - {\mathbf{M}}_{Q}}\right)$ is a real scalar (i.e., $\in  \mathbb{R}$ ,it is equivalent to $\operatorname{tr}\left\{  {{\left( \mathbf{x} - {\mathbf{M}}_{Q}\right) }^{\top }{\mathbf{\sum }}_{Q}^{-1}\left( {\mathbf{x} - {\mathbf{M}}_{Q}}\right) }\right\}$ ,where $\operatorname{tr}\{  \cdot  \}$ denotes the trace of the given matrix. Since $\operatorname{tr}\{ {XY}\}  = \operatorname{tr}\{ {YX}\}$ for any two matrices $X \in  {\mathbb{R}}^{a \times  b}$ and $Y \in  {\mathbb{R}}^{b \times  a}$ ,we have:

由于${\left( \mathbf{x} - {\mathbf{M}}_{Q}\right) }^{\top }{\mathbf{\sum }}_{Q}^{-1}\left( {\mathbf{x} - {\mathbf{M}}_{Q}}\right)$是一个实标量（即$\in  \mathbb{R}$，它等价于$\operatorname{tr}\left\{  {{\left( \mathbf{x} - {\mathbf{M}}_{Q}\right) }^{\top }{\mathbf{\sum }}_{Q}^{-1}\left( {\mathbf{x} - {\mathbf{M}}_{Q}}\right) }\right\}$，其中$\operatorname{tr}\{  \cdot  \}$表示给定矩阵的迹。由于对于任意两个矩阵$X \in  {\mathbb{R}}^{a \times  b}$和$Y \in  {\mathbb{R}}^{b \times  a}$有$\operatorname{tr}\{ {XY}\}  = \operatorname{tr}\{ {YX}\}$，我们有：

$$
\operatorname{tr}\left\{  {{\left( \mathbf{x} - {\mathbf{M}}_{Q}\right) }^{\top }{\mathbf{\sum }}_{Q}^{-1}\left( {\mathbf{x} - {\mathbf{M}}_{Q}}\right) }\right\}   = \operatorname{tr}\left\{  {\left( {\mathbf{x} - {\mathbf{M}}_{Q}}\right) {\left( \mathbf{x} - {\mathbf{M}}_{Q}\right) }^{\top }{\mathbf{\sum }}_{Q}^{-1}}\right\}  
$$

Therefore,since $\mathbb{E}\left\lbrack  {\operatorname{tr}\{ X\} }\right\rbrack   = \operatorname{tr}\{ \mathbb{E}\left\lbrack  X\right\rbrack  \}$ for any square matrix $X$ , we can rewrite Equation (5) as follows:

因此，由于对于任意方阵$X$有$\mathbb{E}\left\lbrack  {\operatorname{tr}\{ X\} }\right\rbrack   = \operatorname{tr}\{ \mathbb{E}\left\lbrack  X\right\rbrack  \}$，我们可以将方程(5)重写为如下形式：

$$
{\mathrm{{KLD}}}_{k}\left( {\mathrm{Q}\parallel \mathrm{D}}\right)  = \frac{1}{2}\log \frac{\det \left( {\mathbf{\sum }}_{D}\right) }{\det \left( {\mathbf{\sum }}_{Q}\right) }
$$

$$
 - \frac{1}{2}\operatorname{tr}\left\{  {{\mathbb{E}}_{\mathrm{Q}}\left\lbrack  {\left( {\mathbf{x} - {\mathbf{M}}_{Q}}\right) {\left( \mathbf{x} - {\mathbf{M}}_{Q}\right) }^{\top }{\mathbf{\sum }}_{Q}^{-1}}\right\rbrack  }\right\}  
$$

$$
 + \frac{1}{2}\operatorname{tr}\left\{  {{\mathbb{E}}_{\mathrm{Q}}\left\lbrack  {{\left( \mathbf{x} - {\mathbf{M}}_{D}\right) }^{\top }{\mathbf{\sum }}_{D}^{-1}\left( {\mathbf{x} - {\mathbf{M}}_{D}}\right) }\right\rbrack  }\right\}   \tag{6}
$$

Given the definition of the covariance matrix, we know that ${\mathbf{\sum }}_{Q} = {\mathbb{E}}_{\mathrm{Q}}\left\lbrack  {\left( {\mathbf{x} - {\mathbf{M}}_{Q}}\right) {\left( \mathbf{x} - {\mathbf{M}}_{Q}\right) }^{\top }}\right\rbrack$ . Therefore,we have:

根据协方差矩阵的定义，我们知道${\mathbf{\sum }}_{Q} = {\mathbb{E}}_{\mathrm{Q}}\left\lbrack  {\left( {\mathbf{x} - {\mathbf{M}}_{Q}}\right) {\left( \mathbf{x} - {\mathbf{M}}_{Q}\right) }^{\top }}\right\rbrack$。因此，我们有：

$$
\operatorname{tr}\left\{  {{\mathbb{E}}_{Q}\left\lbrack  {\left( {\mathbf{x} - {\mathbf{M}}_{Q}}\right) {\left( \mathbf{x} - {\mathbf{M}}_{Q}\right) }^{\top }{\mathbf{\sum }}_{Q}^{-1}}\right\rbrack  }\right\}  
$$

$$
 = \operatorname{tr}\left\{  {{\mathbb{E}}_{\mathbf{Q}}\left\lbrack  {{\mathbf{\sum }}_{Q}{\mathbf{\sum }}_{Q}^{-1}}\right\rbrack  }\right\}   = \operatorname{tr}\left\{  {I}_{k}\right\}   = k \tag{7}
$$

In addition,since $Q$ is a multivariate normal distribution,for any matrix $A$ we have ${\mathbb{E}}_{Q}\left\lbrack  {{\mathbf{x}}^{\top }A\mathbf{x}}\right\rbrack   = \operatorname{tr}\left\{  {A{\mathbf{\sum }}_{Q}}\right\}   + {\mathbf{M}}_{Q}^{\top }A{\mathbf{M}}_{Q}$ . This results in:

此外，由于$Q$是一个多元正态分布，对于任意矩阵$A$，我们有${\mathbb{E}}_{Q}\left\lbrack  {{\mathbf{x}}^{\top }A\mathbf{x}}\right\rbrack   = \operatorname{tr}\left\{  {A{\mathbf{\sum }}_{Q}}\right\}   + {\mathbf{M}}_{Q}^{\top }A{\mathbf{M}}_{Q}$。这导致：

$$
\operatorname{tr}\left\{  {{\mathbb{E}}_{\mathrm{Q}}\left\lbrack  {{\left( \mathbf{x} - {\mathbf{M}}_{D}\right) }^{\top }{\mathbf{\sum }}_{D}^{-1}\left( {\mathbf{x} - {\mathbf{M}}_{D}}\right) }\right\rbrack  }\right\}   = 
$$

$$
\operatorname{tr}\left\{  {{\mathbf{\sum }}_{D}^{-1}{\mathbf{\sum }}_{Q}}\right\}   + {\left( {\mathbf{M}}_{Q} - {\mathbf{M}}_{D}\right) }^{\top }{\mathbf{\sum }}_{D}^{-1}\left( {{\mathbf{M}}_{Q} - {\mathbf{M}}_{D}}\right)  \tag{8}
$$

Using Equations (7) and (8), we can rewrite Equation (6) as follows:

使用方程(7)和(8)，我们可以将方程(6)重写为如下形式：

$$
\frac{1}{2}\left\lbrack  {\log \frac{\det \left( {\mathbf{\sum }}_{D}\right) }{\det \left( {\mathbf{\sum }}_{Q}\right) } - k + \operatorname{tr}\left\{  {{\mathbf{\sum }}_{D}^{-1}{\mathbf{\sum }}_{Q}}\right\}   + {\left( {\mathbf{M}}_{Q} - {\mathbf{M}}_{D}\right) }^{\top }{\mathbf{\sum }}_{D}^{-1}\left( {{\mathbf{M}}_{Q} - {\mathbf{M}}_{D}}\right) }\right\rbrack   \tag{9}
$$

This equation can be further simplified. Based on our earlier assumption that the covariance matrices are diagonal,then $\det \left( {\mathbf{\sum }}_{D}\right)  =$ $\mathop{\prod }\limits_{{i = 1}}^{k}{\sigma }_{di}^{2}$ . In addition,since we are using KL divergence to rank documents,constant values (e.g., $k$ ) or document independent values (e.g., $\log \det \left( {\mathbf{\sum }}_{Q}\right)$ ) do not impact document ordering. Therefore, there can be omitted and we can use the following equation to rank the documents using negative multivariate KL-divergence:

这个方程可以进一步简化。基于我们之前的假设，即协方差矩阵是对角矩阵，那么$\det \left( {\mathbf{\sum }}_{D}\right)  =$ $\mathop{\prod }\limits_{{i = 1}}^{k}{\sigma }_{di}^{2}$。此外，由于我们使用KL散度对文档进行排序，常数值（例如$k$）或与文档无关的值（例如$\log \det \left( {\mathbf{\sum }}_{Q}\right)$）不会影响文档的排序。因此，可以省略这些值，我们可以使用以下方程通过负多元KL散度对文档进行排序：

$$
\operatorname{score}\left( {q,d}\right)  =  - {\mathrm{{KLD}}}_{k}\left( {\mathrm{Q}\parallel \mathrm{D}}\right) 
$$

$$
{ = }^{\text{rank }} - \frac{1}{2}\left\lbrack  {\mathop{\sum }\limits_{{i = 1}}^{k}\log {\sigma }_{di}^{2} + \frac{\mathop{\prod }\limits_{{i = 1}}^{k}{\sigma }_{qi}^{2}}{\mathop{\prod }\limits_{{i = 1}}^{k}{\sigma }_{di}^{2}} + \mathop{\sum }\limits_{{i = 1}}^{k}\frac{{\left( {\mu }_{qi} - {\mu }_{di}\right) }^{2}}{{\sigma }_{di}^{2}}}\right\rbrack  
$$

(10)

In Section 4.3, we explain how to efficiently compute this scoring function using approximate nearest neighbor methods.

在4.3节中，我们将解释如何使用近似最近邻方法高效地计算这个评分函数。

## 4 MRL IMPLEMENTATION

## 4 MRL实现

In this section, we first describe our network architecture for implementing the the query and document encoders ${\mathrm{{ENCODER}}}_{\mathrm{Q}}$ and ${\text{ENCODER}}_{\mathrm{D}}$ (see Equation (3)). Next,we explain our optimization approach for training the models.

在本节中，我们首先描述用于实现查询和文档编码器 ${\mathrm{{ENCODER}}}_{\mathrm{Q}}$ 和 ${\text{ENCODER}}_{\mathrm{D}}$ 的网络架构（见公式 (3)）。接下来，我们解释训练模型的优化方法。

### 4.1 Encoder Architecture

### 4.1 编码器架构

Pretrained large language models (LLMs) have demonstrated promising results in various information retrieval tasks [10, 17, 33, 57]. Therefore, we decide to adapt existing pretrained LLMs to learn a $k$ -variate normal distribution for each given input. As described above,each $k$ -variate normal distribution can be modeled using a $k$ -dimensional mean vector and a $k$ -dimensional variance vector. We use two special tokens as the input of pretrained LLMs to obtain these two vectors. For example, we convert an input query 'neural information retrieval' to '[CLS] [VAR] neural information retrieval [SEP]' and feed it to BERT-base [13]. Let ${\overrightarrow{q}}_{\left\lbrack  \mathrm{{CLS}}\right\rbrack  } \in  {\mathbb{R}}^{1 \times  {768}}$ and ${\overrightarrow{q}}_{\left\lbrack  \mathrm{{VAR}}\right\rbrack  } \in  {\mathbb{R}}^{1 \times  {768}}$ respectively denote the representations produced by BERT for the first two tokens [CLS] and [VAR]. We obtain the mean and variance vectors for query $q$ using two separate dense projection layers on ${\overrightarrow{q}}_{\left\lbrack  \mathrm{{CLS}}\right\rbrack  }$ and ${\overrightarrow{q}}_{\left\lbrack  \mathrm{{VAR}}\right\rbrack  }$ , as follows:

预训练大语言模型（LLMs）在各种信息检索任务中已展现出良好的效果 [10, 17, 33, 57]。因此，我们决定采用现有的预训练大语言模型，为每个给定输入学习一个 $k$ 元正态分布。如上所述，每个 $k$ 元正态分布可以用一个 $k$ 维的均值向量和一个 $k$ 维的方差向量来建模。我们使用两个特殊标记作为预训练大语言模型的输入，以获得这两个向量。例如，我们将输入查询“神经信息检索”转换为“[CLS] [VAR] 神经信息检索 [SEP]”，并将其输入到 BERT-base 模型 [13] 中。设 ${\overrightarrow{q}}_{\left\lbrack  \mathrm{{CLS}}\right\rbrack  } \in  {\mathbb{R}}^{1 \times  {768}}$ 和 ${\overrightarrow{q}}_{\left\lbrack  \mathrm{{VAR}}\right\rbrack  } \in  {\mathbb{R}}^{1 \times  {768}}$ 分别表示 BERT 为前两个标记 [CLS] 和 [VAR] 生成的表示。我们通过在 ${\overrightarrow{q}}_{\left\lbrack  \mathrm{{CLS}}\right\rbrack  }$ 和 ${\overrightarrow{q}}_{\left\lbrack  \mathrm{{VAR}}\right\rbrack  }$ 上使用两个独立的密集投影层，得到查询 $q$ 的均值和方差向量，如下所示：

$$
{M}_{Q} = {\overrightarrow{q}}_{\left\lbrack  \mathrm{{CLS}}\right\rbrack  }{W}_{M}
$$

$$
{\mathbf{\sum }}_{Q} = \frac{1}{\beta }\log \left( {1 + \exp \left( {\beta .{\overrightarrow{q}}_{\left\lbrack  \mathrm{{VAR}}\right\rbrack  }{W}_{\sum }}\right) }\right) .{I}_{k} \tag{11}
$$

where ${W}_{M} \in  {\mathbb{R}}^{{768} \times  k}$ and ${W}_{\sum } \in  {\mathbb{R}}^{{768} \times  k}$ are the projection layer parameters. To compute the diagonal covariance matrix, we use the softplus function (i.e., $\frac{1}{\beta }\log \left( {1 + \exp \left( {\beta .x}\right) }\right)$ ) for the following reasons: (1) it is continuous and differentiable, thus it can be used in gradient descent-based optimization, (2) softplus ensures that variance values are always positive, (3) zero is its lower bound $\left( {\mathop{\lim }\limits_{{x \rightarrow   - \infty }}\frac{1}{\beta }\log \left( {1 + \exp \left( {\beta .x}\right) }\right)  = 0}\right)$ ,yet it is never equal to zero, thus it does not cause numeric instability in KL-divergence calculation (see Equation (10)),and (4) for large $x$ values,it can be approximated using a linear function,i.e., $\mathop{\lim }\limits_{{x \rightarrow  \infty }}\frac{1}{\beta }\log \left( {1 + \exp \left( {\beta .x}\right) }\right)  = x$ , ensuring numerical stability for large input values. To better demonstrate its properties, Figure 2 in our experiments plots softplus for various values of $\beta$ - a hyper-parameter that specifies the softplus formation. The $k \times  k$ identity matrix ${I}_{k}$ in Equation (11) is used to convert the variance vector to a diagonal covariance matrix.

其中 ${W}_{M} \in  {\mathbb{R}}^{{768} \times  k}$ 和 ${W}_{\sum } \in  {\mathbb{R}}^{{768} \times  k}$ 是投影层参数。为了计算对角协方差矩阵，我们使用软加函数（即 $\frac{1}{\beta }\log \left( {1 + \exp \left( {\beta .x}\right) }\right)$ ），原因如下：(1) 它是连续且可微的，因此可用于基于梯度下降的优化；(2) 软加函数确保方差值始终为正；(3) 零是其下界 $\left( {\mathop{\lim }\limits_{{x \rightarrow   - \infty }}\frac{1}{\beta }\log \left( {1 + \exp \left( {\beta .x}\right) }\right)  = 0}\right)$，但它永远不等于零，因此不会在 KL 散度计算中导致数值不稳定（见公式 (10)）；(4) 对于较大的 $x$ 值，它可以用线性函数近似，即 $\mathop{\lim }\limits_{{x \rightarrow  \infty }}\frac{1}{\beta }\log \left( {1 + \exp \left( {\beta .x}\right) }\right)  = x$，确保大输入值的数值稳定性。为了更好地展示其特性，我们实验中的图 2 绘制了不同 $\beta$ 值下的软加函数，$\beta$ 是指定软加函数形式的超参数。公式 (11) 中的 $k \times  k$ 单位矩阵 ${I}_{k}$ 用于将方差向量转换为对角协方差矩阵。

Note that MRL does not explicitly compute variance, instead learns representations for the [VAR] token such that it minimizes the loss function based on negative multivariate KL divergence scoring. Therefore, the model implicitly learns how to represent latent variance vectors.

请注意，MRL 并不显式计算方差，而是学习 [VAR] 标记的表示，以便基于负多元 KL 散度评分最小化损失函数。因此，模型隐式地学习如何表示潜在方差向量。

The mean vector and covariance matrices for document representations are also computed similarly. In our experiments, all parameters (including parameters in BERT and the dense projection layers) are updated and shared between the query and document encoders (i.e., ${\mathrm{{ENCODER}}}_{\mathrm{Q}}$ and ${\mathrm{{ENCODER}}}_{\mathrm{D}}$ ).

文档表示的均值向量和协方差矩阵的计算方式类似。在我们的实验中，所有参数（包括 BERT 和密集投影层中的参数）都会更新，并在查询和文档编码器（即 ${\mathrm{{ENCODER}}}_{\mathrm{Q}}$ 和 ${\mathrm{{ENCODER}}}_{\mathrm{D}}$ ）之间共享。

### 4.2 Model Training

### 4.2 模型训练

Recent research has suggested that dense retrieval models can significantly benefit from knowledge distillation $\left\lbrack  {{18},{37},{43}}\right\rbrack$ . Following these models, we use a BERT-based cross-encoder re-ranking model as the teacher model. Let ${D}_{q}$ be a set of documents selected for query $q$ for knowledge distillation. We use the following listwise

近期研究表明，密集检索模型可以从知识蒸馏中显著受益 $\left\lbrack  {{18},{37},{43}}\right\rbrack$。遵循这些模型，我们使用基于BERT的交叉编码器重排序模型作为教师模型。设 ${D}_{q}$ 为为查询 $q$ 选择的用于知识蒸馏的文档集合。我们使用以下列表式

loss function for each query $q$ as follows:

每个查询 $q$ 的损失函数如下：

$$
\mathop{\sum }\limits_{{d,{d}^{\prime } \in  {D}_{q}}}\mathbb{1}\left\{  {{y}_{q}^{T}\left( d\right)  > {y}_{q}^{T}\left( {d}^{\prime }\right) }\right\}  \left| {\frac{1}{{\pi }_{q}\left( d\right) } - \frac{1}{{\pi }_{q}\left( {d}^{\prime }\right) }}\right| \log \left( {1 + {e}^{{y}_{q}^{S}\left( {d}^{\prime }\right)  - {y}_{q}^{S}\left( d\right) }}\right) 
$$

(12)

where ${\pi }_{q}\left( d\right)$ denotes the rank of document $d$ in the result list produced by the student dense retrieval model,and ${y}_{q}^{T}\left( d\right)$ and ${y}_{q}^{S}\left( d\right)$ respectively denote the scores produced by the teacher and the student models for the pair of query $q$ and document $d$ . This knowledge distillation listwise loss function is inspired by LambdaRank [3] and is also used by Zeng et al. [57] for dense retrieval distillation.

其中 ${\pi }_{q}\left( d\right)$ 表示文档 $d$ 在学生密集检索模型生成的结果列表中的排名，${y}_{q}^{T}\left( d\right)$ 和 ${y}_{q}^{S}\left( d\right)$ 分别表示教师模型和学生模型针对查询 $q$ 和文档 $d$ 这一对所生成的分数。这个知识蒸馏列表式损失函数的灵感来源于LambdaRank [3]，曾等人 [57] 也将其用于密集检索蒸馏。

For each query $q$ ,the document set ${D}_{q}$ is constructed based on the following steps:

对于每个查询 $q$，文档集 ${D}_{q}$ 基于以下步骤构建：

- ${D}_{q}$ includes all positive documents from the relevance judgments (i.e., qrel).

- ${D}_{q}$ 包含相关性判断（即qrel）中的所有正文档。

- ${D}_{q}$ includes ${m}_{\mathrm{{BM}}{25}} \in  \mathbb{R}$ documents from the top 100 documents retrieved by BM25.

- ${D}_{q}$ 包含从BM25检索出的前100个文档中的 ${m}_{\mathrm{{BM}}{25}} \in  \mathbb{R}$ 个文档。

- ${D}_{q}$ includes ${m}_{\text{hard }} \in  \mathbb{R}$ documents from the top 100 documents retrieved by student model (i.e., negative sampling using the model itself every 5000 steps).

- ${D}_{q}$ 包含从学生模型检索出的前100个文档中的 ${m}_{\text{hard }} \in  \mathbb{R}$ 个文档（即每5000步使用模型本身进行负采样）。

In addition, we take advantage of the other passages in the batch as in-batch negatives. Although in-batch negatives resemble randomly sampled negatives that can be distinguished easily from other documents, it is efficient since passage representations can be reused within the batch [23].

此外，我们利用批次中的其他段落作为批次内负样本。尽管批次内负样本类似于随机采样的负样本，很容易与其他文档区分开来，但它很高效，因为段落表示可以在批次内重复使用 [23]。

### 4.3 Efficient Retrieval

### 4.3 高效检索

Existing dense retrieval models use approximate nearest neighbor (ANN) approaches for efficient retrieval. However, using ANN algorithms in the proposed MRL framework is not trivial. The reason is that MRL uses the negative $k$ -variate KL divergence formulation presented in Equation (10) to compute relevance scores. This is while existing ANN algorithms only support simple similarity functions such as dot product, cosine similarity, or negative Euclidean distance. To address this issue, we convert Equation (10) to a dot product formulation. Let us expand the last term in Equation (10): ${}^{2}$

现有的密集检索模型使用近似最近邻（ANN）方法进行高效检索。然而，在提出的MRL框架中使用ANN算法并非易事。原因是MRL使用公式（10）中给出的负 $k$ 元KL散度公式来计算相关性分数。而现有的ANN算法仅支持简单的相似性函数，如点积、余弦相似度或负欧几里得距离。为了解决这个问题，我们将公式（10）转换为点积公式。让我们展开公式（10）中的最后一项：${}^{2}$

$$
 - \left\lbrack  {\underset{\text{doc prior }}{\underbrace{\mathop{\sum }\limits_{{i = 1}}^{k}\log {\sigma }_{di}^{2}}} + \underset{\text{ Me prior }}{\underbrace{\mathop{\prod }\limits_{{i = 1}}^{k}{\sigma }_{qi}^{2}}} + \mathop{\sum }\limits_{{i = 1}}^{k}\frac{{\mu }_{qi}^{2}}{{\sigma }_{di}^{2}} + \underset{\text{ doc prior }}{\underbrace{\mathop{\sum }\limits_{{i = 1}}^{k}\frac{{\mu }_{di}^{2}}{{\sigma }_{di}^{2}}}} - \mathop{\sum }\limits_{{i = 1}}^{k}\frac{2{\mu }_{di}{\mu }_{qi}}{{\sigma }_{di}^{2}}}\right\rbrack  
$$

(13)

The first and the fourth terms in Equation (13) are document priors, thus they are query independent and can be pre-computed. Therefore,let ${\gamma }_{d} =  - \mathop{\sum }\limits_{{i = 1}}^{k}\left( {\log {\sigma }_{di}^{2} + \frac{{\mu }_{di}^{2}}{{\sigma }_{di}^{2}}}\right)$ denote the document prior score. Therefore, the scoring function in Equation (10) can be formulated as the dot product of the following two vectors:

公式（13）中的第一项和第四项是文档先验，因此它们与查询无关，可以预先计算。因此，设 ${\gamma }_{d} =  - \mathop{\sum }\limits_{{i = 1}}^{k}\left( {\log {\sigma }_{di}^{2} + \frac{{\mu }_{di}^{2}}{{\sigma }_{di}^{2}}}\right)$ 表示文档先验分数。因此，公式（10）中的评分函数可以表示为以下两个向量的点积：

$$
\overrightarrow{q} = \left\lbrack  {1,{\Pi }_{q},{\mu }_{q1}^{2},{\mu }_{q2}^{2},\cdots ,{\mu }_{qk}^{2},{\mu }_{q1},{\mu }_{q2},\cdots ,{\mu }_{qk}}\right\rbrack  
$$

$$
\overrightarrow{d} = \left\lbrack  {{\gamma }_{d},\frac{-1}{{\Pi }_{d}},\frac{-1}{{\sigma }_{d1}^{2}},\frac{-1}{{\sigma }_{d2}^{2}},\cdots ,\frac{-1}{{\sigma }_{dk}^{2}},\frac{2{\mu }_{d1}}{{\sigma }_{d1}^{2}},\frac{2{\mu }_{d2}}{{\sigma }_{d2}^{2}},\cdots ,\frac{2{\mu }_{dk}}{{\sigma }_{dk}^{2}}}\right\rbrack   \tag{14}
$$

where ${\Pi }_{q} = \mathop{\prod }\limits_{{i = 1}}^{k}{\sigma }_{qi}^{2}$ and ${\Pi }_{d} = \mathop{\prod }\limits_{{i = 1}}^{k}{\sigma }_{di}^{2}$ are pre-computed scalars. The dot product of $\overrightarrow{q} \in  {\mathbb{R}}^{1 \times  \left( {{2k} + 2}\right) }$ and $\overrightarrow{d} \in  {\mathbb{R}}^{1 \times  \left( {{2k} + 2}\right) }$ is equal to the retrieval score formulated in Equation (10). More importantly, $\overrightarrow{q}$ is document independent and $\overrightarrow{d}$ is query independent. Therefore, we can use existing approximate nearest neighbor algorithms, such as HNSW [30], and existing tools, such as FAISS [22], to index all $\overrightarrow{d}$ vectors and conduct efficient retrieval for any query vector $\overrightarrow{q}$ .

其中 ${\Pi }_{q} = \mathop{\prod }\limits_{{i = 1}}^{k}{\sigma }_{qi}^{2}$ 和 ${\Pi }_{d} = \mathop{\prod }\limits_{{i = 1}}^{k}{\sigma }_{di}^{2}$ 是预先计算的标量。$\overrightarrow{q} \in  {\mathbb{R}}^{1 \times  \left( {{2k} + 2}\right) }$ 和 $\overrightarrow{d} \in  {\mathbb{R}}^{1 \times  \left( {{2k} + 2}\right) }$ 的点积等于公式（10）中给出的检索分数。更重要的是，$\overrightarrow{q}$ 与文档无关，$\overrightarrow{d}$ 与查询无关。因此，我们可以使用现有的近似最近邻算法，如HNSW [30]，以及现有的工具，如FAISS [22]，对所有 $\overrightarrow{d}$ 向量进行索引，并对任何查询向量 $\overrightarrow{q}$ 进行高效检索。

## 5 DISCUSSION

## 5 讨论

In this section, we attempt to shed some light on the behavior of retrieval using MRL, by providing theoretical answers to the following questions.

在本节中，我们试图通过为以下问题提供理论答案，来阐明使用MRL进行检索的行为。

### Q1.How does MRL rank two documents with identi- cal covariance matrices?

### Q1. MRL如何对具有相同协方差矩阵的两个文档进行排序？

Let $d$ and ${d}^{\prime }$ be two documents,represented by the mean vectors ${\mathbf{M}}_{D}$ and ${\mathbf{M}}_{{D}^{\prime }}$ and identical covariance matrix ${\mathbf{\sum }}_{D} = {\mathbf{\sum }}_{{D}^{\prime }}$ . Therefore, given Equation (10) we have:

设 $d$ 和 ${d}^{\prime }$ 为两个文档，分别由均值向量 ${\mathbf{M}}_{D}$ 和 ${\mathbf{M}}_{{D}^{\prime }}$ 以及相同的协方差矩阵 ${\mathbf{\sum }}_{D} = {\mathbf{\sum }}_{{D}^{\prime }}$ 表示。因此，根据方程 (10)，我们有：

$\operatorname{score}\left( {q,d}\right)  - \operatorname{score}\left( {q,{d}^{\prime }}\right) { = }^{\text{rank }}\mathop{\sum }\limits_{{i = 1}}^{k}\left\lbrack  {{\left( {\mu }_{qi} - {\mu }_{{d}^{\prime }i}\right) }^{2} - {\left( {\mu }_{qi} - {\mu }_{di}\right) }^{2}}\right\rbrack$

This shows that in case of identical covariance matrices, MRL assigns a higher relevance score to the document whose mean vector is closest to the query mean vector with respect to Euclidean distance.

这表明，在协方差矩阵相同的情况下，最大相关性学习（MRL，Maximum Relevance Learning）会给均值向量在欧几里得距离上最接近查询均值向量的文档分配更高的相关性得分。

A remark of this finding is that if the covariance matrix is constant for all documents (i.e., if we ignore uncertainty), MRL can be reduced to existing dense retrieval formulation, where negative Euclidean distance is used to measure vector similarity. Therefore, MRL is a generalized form of this dense retrieval formulation.

这一发现的一个注解是，如果所有文档的协方差矩阵是恒定的（即，如果我们忽略不确定性），最大相关性学习（MRL）可以简化为现有的密集检索公式，其中使用负欧几里得距离来衡量向量相似度。因此，最大相关性学习（MRL）是这种密集检索公式的广义形式。

Q2. Popular dense retrieval models use inner product to compute the similarity between query and document vectors. What happens if we use inner product in MRL?

问题 2：流行的密集检索模型使用内积来计算查询向量和文档向量之间的相似度。如果我们在最大相关性学习（MRL）中使用内积会怎样？

Inner product or dot product cannot be defined for multivariate distributions, however, one can take several samples from the query and document distributions and compute their dot product similarity. Since the query distribution $\mathrm{Q} \sim  {\mathcal{N}}_{k}\left( {{\mathbf{M}}_{Q},{\mathbf{\sum }}_{Q}}\right)$ and the document distribution $\mathrm{D} \sim  {\mathcal{N}}_{k}\left( {{\mathbf{M}}_{D},{\mathbf{\sum }}_{D}}\right)$ are independent,the expected value of their product is:

对于多元分布，无法定义内积或点积。然而，可以从查询分布和文档分布中抽取多个样本，并计算它们的点积相似度。由于查询分布 $\mathrm{Q} \sim  {\mathcal{N}}_{k}\left( {{\mathbf{M}}_{Q},{\mathbf{\sum }}_{Q}}\right)$ 和文档分布 $\mathrm{D} \sim  {\mathcal{N}}_{k}\left( {{\mathbf{M}}_{D},{\mathbf{\sum }}_{D}}\right)$ 是相互独立的，它们乘积的期望值为：

$$
\mathbb{E}\left\lbrack  {\mathrm{Q} \cdot  \mathrm{D}}\right\rbrack   = \mathbb{E}\left\lbrack  \mathrm{Q}\right\rbrack   \cdot  \mathbb{E}\left\lbrack  \mathrm{D}\right\rbrack   = {\mathbf{M}}_{Q} \cdot  {\mathbf{M}}_{D}
$$

That means, in expectation, the dot product of samples from multivariate distributions will be equivalent to the dot product of their mean vectors. Therefore, with this formulation (i.e., using expected dot product instead of negative KL divergence) the results produced by MRL will be equivalent to the existing dense retrieval models and representation uncertainties are not considered.

这意味着，从期望上来说，多元分布样本的点积将等同于它们均值向量的点积。因此，使用这种公式（即，使用期望点积而不是负 KL 散度），最大相关性学习（MRL）产生的结果将等同于现有的密集检索模型，并且不考虑表示的不确定性。

Q3. Negative KL divergence has been used in the language modeling framework of information retrieval [27]. How is it connected with the proposed MRL framework?

问题 3：负 KL 散度已被用于信息检索的语言建模框架中 [27]。它与所提出的最大相关性学习（MRL）框架有什么联系？

Lafferty and Zhai [27] extended the query likelihood retrieval model of Ponte and Croft [35] by computing negative KL divergence between unigram query and document language models. Similarly, MRL uses negative KL divergence to compute relevance scores, however, there are several fundamental differences. First, Lafferty and Zhai [27] compute the distributions based on term occurrences in queries and documents through maximum likelihood estimation, while MRL learns latent distributions based on the contextual representations learned from queries and documents. Second, Lafferty and Zhai [27] use univariate distributions for queries and documents, while MRL uses high-dimensional multivariate distributions.

拉弗蒂（Lafferty）和翟（Zhai） [27] 通过计算一元查询语言模型和文档语言模型之间的负 KL 散度，扩展了庞特（Ponte）和克罗夫特（Croft） [35] 的查询似然检索模型。类似地，最大相关性学习（MRL）使用负 KL 散度来计算相关性得分，然而，存在几个根本差异。首先，拉弗蒂（Lafferty）和翟（Zhai） [27] 通过最大似然估计，基于查询和文档中的词项出现情况来计算分布，而最大相关性学习（MRL）基于从查询和文档中学习到的上下文表示来学习潜在分布。其次，拉弗蒂（Lafferty）和翟（Zhai） [27] 对查询和文档使用单变量分布，而最大相关性学习（MRL）使用高维多元分布。

---

<!-- Footnote -->

${}^{2}$ We drop multiplication to $\frac{1}{2}$ as it does not impact document ordering.

${}^{2}$ 我们去掉与 $\frac{1}{2}$ 的乘法，因为它不影响文档排序。

<!-- Footnote -->

---

<!-- Media -->

Table 1: Characteristics and statistics of the datasets in our experiments.

表 1：我们实验中数据集的特征和统计信息。

<table><tr><td>$\mathbf{{Dataset}}$</td><td>Domain</td><td>#queries</td><td>#documents</td><td>avg doc length</td></tr><tr><td>MS MARCO DEV</td><td>Miscellaneous</td><td>6,980</td><td>8,841,823</td><td>56</td></tr><tr><td>TREC DL ’19</td><td>Miscellaneous</td><td>43</td><td>8,841,823</td><td>56</td></tr><tr><td>TREC DL ’20</td><td>Miscellaneous</td><td>54</td><td>8,841,823</td><td>56</td></tr><tr><td>SciFact</td><td>Scientific fact retrieval</td><td>300</td><td>5,183</td><td>214</td></tr><tr><td>FiQA</td><td>Financial answer retrieval</td><td>648</td><td>57,638</td><td>132</td></tr><tr><td>TREC COVID</td><td>Bio-medical retrieval for Covid-19</td><td>50</td><td>171,332</td><td>161</td></tr><tr><td>CQADupStack</td><td>Duplicate question retrieval</td><td>13,145</td><td>457,199</td><td>129</td></tr></table>

<table><tbody><tr><td>$\mathbf{{Dataset}}$</td><td>领域</td><td>#查询次数</td><td>#文档数量</td><td>平均文档长度</td></tr><tr><td>MS MARCO开发集</td><td>其他</td><td>6,980</td><td>8,841,823</td><td>56</td></tr><tr><td>TREC文档检索评测2019（TREC DL ’19）</td><td>其他</td><td>43</td><td>8,841,823</td><td>56</td></tr><tr><td>TREC文档检索评测2020（TREC DL ’20）</td><td>其他</td><td>54</td><td>8,841,823</td><td>56</td></tr><tr><td>科学事实数据集（SciFact）</td><td>科学事实检索</td><td>300</td><td>5,183</td><td>214</td></tr><tr><td>金融问答数据集（FiQA）</td><td>金融答案检索</td><td>648</td><td>57,638</td><td>132</td></tr><tr><td>TREC新冠疫情信息检索评测（TREC COVID）</td><td>新冠疫情生物医学信息检索</td><td>50</td><td>171,332</td><td>161</td></tr><tr><td>重复问题数据集（CQADupStack）</td><td>重复问题检索</td><td>13,145</td><td>457,199</td><td>129</td></tr></tbody></table>

<!-- Media -->

## 6 EXPERIMENTS

## 6 实验

To evaluate the impact of multivariate representation learning, we first run experiments on standard passage retrieval collections from MS MARCO and TREC Deep Learning Tracks. We also study the parameter sensitivity of the model in this task. We further demonstrate the ability of multivariate representations to better model distribution shift when applied to zero-shot retrieval settings, i.e., retrieval on a target collection that is significantly different from the training set. Our experiments also shows that the norm of learned variance vectors is correlated with the retrieval performance of the model.

为了评估多变量表示学习的影响，我们首先在来自MS MARCO和TREC深度学习赛道的标准段落检索集合上进行实验。我们还研究了模型在该任务中的参数敏感性。我们进一步证明了多变量表示在应用于零样本检索设置（即在与训练集显著不同的目标集合上进行检索）时，能够更好地对分布偏移进行建模。我们的实验还表明，学习到的方差向量的范数与模型的检索性能相关。

### 6.1 Datasets

### 6.1 数据集

In this section, we introduce our training set and evaluation sets whose characteristics and statistics are reported in Table 1.

在本节中，我们介绍我们的训练集和评估集，其特征和统计信息见表1。

Training Set. We train our ranking model on the MS MARCO passage retrieval training set. The MS MARCO collection [4] contains approximately ${8.8}\mathrm{M}$ passages and its training set includes ${503}\mathrm{\;K}$ unique queries. The MS MARCO training set was originally constructed for a machine reading comprehension tasks, thus it did not follow the standard IR annotation guidelines (e.g., pooling). The training set contains an average of 1.1 relevant passage per query, even though there exist several relevant documents that are left adjudged. This is one of the reasons that knowledge distillation help dense retrieval models learn more robust representations.

训练集。我们在MS MARCO段落检索训练集上训练我们的排序模型。MS MARCO集合 [4] 包含约 ${8.8}\mathrm{M}$ 个段落，其训练集包括 ${503}\mathrm{\;K}$ 个唯一查询。MS MARCO训练集最初是为机器阅读理解任务构建的，因此它没有遵循标准的信息检索（IR）标注指南（例如，池化）。训练集中每个查询平均有1.1个相关段落，尽管存在一些未被判定的相关文档。这是知识蒸馏有助于密集检索模型学习更鲁棒表示的原因之一。

Passage Retrieval Evaluation Sets. We evaluate our models on three query sets for the passage retrieval task. They all use the MS MARCO passage collection. These evaluation query sets are: (1) MS MARCO DEV: the standard development set of MS MARCO passage retrieval task that consists of 6980 queries with incomplete relevance annotations (similar to the training set), (2) TREC-DL'19: passage retrieval query set used in the first iteration of TREC Deep Learning Track in 2019 [9] which includes 43 queries, and (3) TREC-DL'20: the passage retrieval query set of TREC Deep Learning Track 2020 [10] with 54 queries. Relevance annotation for TREC DL tracks was curated using standard pooling techniques. Therefore, we can consider them as datasets with complete relevance annotations.

段落检索评估集。我们在三个查询集上对段落检索任务的模型进行评估。它们都使用MS MARCO段落集合。这些评估查询集包括：（1）MS MARCO DEV：MS MARCO段落检索任务的标准开发集，由6980个查询组成，相关性标注不完整（与训练集类似）；（2）TREC - DL'19：2019年TREC深度学习赛道第一轮使用的段落检索查询集 [9]，包含43个查询；（3）TREC - DL'20：2020年TREC深度学习赛道的段落检索查询集 [10]，包含54个查询。TREC DL赛道的相关性标注是使用标准池化技术整理的。因此，我们可以将它们视为具有完整相关性标注的数据集。

Zero-Shot Passage Retrieval Evaluation Sets. To demonstrate the generalization of retrieval models to different domains, we perform a zero-shot passage retrieval experiment (i.e., the models are trained on the MS MARCO training set). To do so, we use four domains which diverse properties. (1) SciFact [51]: a dataset for scientific fact retrieval with 300 queries, (2) FiQA [29]: a passage retrieval dataset for natural language questions in the financial domain with 648 queries, (3) TREC COVID [50]: a task of retrieving abstracts of bio-medical articles in response to 50 queries related to the Covid-19 pandemic, and (4) CQADupStack [19]: the task of duplicated question retrieval on 12 diverse StackExchange websites with 13,145 test queries. To be consistent with the literature, we used the BEIR [45] version of all these collections.

零样本段落检索评估集。为了证明检索模型在不同领域的泛化能力，我们进行了零样本段落检索实验（即模型在MS MARCO训练集上进行训练）。为此，我们使用了四个具有不同属性的领域。（1）SciFact [51]：一个用于科学事实检索的数据集，有300个查询；（2）FiQA [29]：一个用于金融领域自然语言问题的段落检索数据集，有648个查询；（3）TREC COVID [50]：一个针对与新冠疫情相关的50个查询检索生物医学文章摘要的任务；（4）CQADupStack [19]：在12个不同的StackExchange网站上进行重复问题检索的任务，有13145个测试查询。为了与文献保持一致，我们使用了所有这些集合的BEIR [45] 版本。

### 6.2 Experimental Setup

### 6.2 实验设置

We implemented and trained our models using TensorFlow. The network parameters were optimized with Adam [25] with linear scheduling with the warmup of 4000 steps. In our experiments, the learning rate was selected from $\left\lbrack  {1 \times  {10}^{-6},1 \times  {10}^{-5}}\right\rbrack$ with a step size of $1 \times  {10}^{-6}$ . The batch size was set to 512 . The parameter $\beta$ was selected from $\left\lbrack  {{0.5},1,{2.5},5,{7.5},{10}}\right\rbrack$ . To have a fair comparison with the baselines that often use 768 dimensions for representing queries and documents using BERT,we set the parameter $k$ (i.e.,the number of random variables in our multivariate normal distributions) to $\frac{768}{2} - 1 = {381}$ (see Section 4.3 for more information). In our experiments, we use the DistilBERT [42] with the pre-trained checkpoint made available from TAS-B [18] as the initialization. As the re-ranking teacher model, we use a BERT cross-encoder, similar to that of Nogueira and Cho [33]. Hyper-parameter selection and early stopping was conducted based on the performance in terms of MRR on the MS MARCO validation set.

我们使用TensorFlow实现并训练我们的模型。网络参数使用Adam [25] 优化器进行优化，采用线性调度，预热步数为4000步。在我们的实验中，学习率从 $\left\lbrack  {1 \times  {10}^{-6},1 \times  {10}^{-5}}\right\rbrack$ 中选择，步长为 $1 \times  {10}^{-6}$。批量大小设置为512。参数 $\beta$ 从 $\left\lbrack  {{0.5},1,{2.5},5,{7.5},{10}}\right\rbrack$ 中选择。为了与通常使用BERT将查询和文档表示为768维的基线模型进行公平比较，我们将参数 $k$（即我们的多元正态分布中的随机变量数量）设置为 $\frac{768}{2} - 1 = {381}$（更多信息见第4.3节）。在我们的实验中，我们使用DistilBERT [42] 并以TAS - B [18] 提供的预训练检查点进行初始化。作为重排序教师模型，我们使用一个BERT交叉编码器，类似于Nogueira和Cho [33] 所使用的。超参数选择和提前停止是基于模型在MS MARCO验证集上的平均倒数排名（MRR）性能进行的。

### 6.3 Evaluation Metrics

### 6.3 评估指标

We use appropriate metrics for each evaluation set based on their properties. For MS MARCO Dev, we use MRR@10 which is the standard metric for this dataset, and we followed TREC Deep Learning Track's recommendation on using NDCG@10 [21] as the evaluation metrics. To complement our result analysis, we also use mean average precision of the top 1000 retrieved documents (MAP), which is a common recall-oriented metric. For zero-shot evaluation, we follow BEIR's recommendation and use NDCG@10 to be consistent with the literature [45]. The two-tailed paired t-test with Bonferroni correction is used to identify statistically significant performance differences $\left( {p\_ \text{value} < {0.05}}\right)$ .

我们根据每个评估集的特性为其使用合适的指标。对于MS MARCO开发集，我们使用MRR@10（前10个结果的平均倒数排名），这是该数据集的标准指标，并且我们遵循TREC深度学习赛道的建议，使用NDCG@10 [21]作为评估指标。为了补充我们的结果分析，我们还使用前1000个检索文档的平均准确率均值（MAP），这是一种常见的面向召回率的指标。对于零样本评估，我们遵循BEIR的建议，使用NDCG@10以与文献[45]保持一致。采用经过Bonferroni校正的双尾配对t检验来确定具有统计学意义的性能差异$\left( {p\_ \text{value} < {0.05}}\right)$。

<!-- Media -->

Table 2: The passage retrieval results obtained by the proposed approach and the baselines. The highest value in each column is bold-faced. The superscript ${}^{ * }$ denotes statistically significant improvements compared to all the baselines based on two-tailed paired t-test with Bonferroni correction at the 95% confidence level. “-” denotes the results that are not applicable or available.

表2：所提出的方法和基线方法获得的段落检索结果。每列中的最高值用粗体表示。上标${}^{ * }$表示在95%置信水平下，与所有基线方法相比，基于经过Bonferroni校正的双尾配对t检验具有统计学意义的改进。“-”表示不适用或无法获取的结果。

<table><tr><td rowspan="2">Model</td><td rowspan="2">Encoder</td><td rowspan="2">#params</td><td colspan="2">MS MARCO DEV</td><td colspan="2">TREC-DL’19</td><td colspan="2">TREC-DL’20</td></tr><tr><td>MRR@10</td><td>$\mathbf{{MAP}}$</td><td>NDCG@10</td><td>$\mathbf{{MAP}}$</td><td>NDCG@10</td><td>MAP</td></tr><tr><td colspan="9">Single Vector Dense Retrieval Models</td></tr><tr><td>ANCE [53]</td><td>BERT-Base</td><td>110M</td><td>0.330</td><td>0.336</td><td>0.648</td><td>0.371</td><td>0.646</td><td>0.408</td></tr><tr><td>ADORE [58]</td><td>BERT-Base</td><td>110M</td><td>0.347</td><td>0.352</td><td>0.683</td><td>0.419</td><td>0.666</td><td>0.442</td></tr><tr><td>RocketQA [37]</td><td>ERNIE-Base</td><td>110M</td><td>0.370</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Contriever-FT [20]</td><td>BERT-Base</td><td>110M</td><td>-</td><td>-</td><td>0.621</td><td>-</td><td>0.632</td><td>-</td></tr><tr><td>TCT-ColBERT [28]</td><td>BERT-Base</td><td>110M</td><td>0.335</td><td>0.342</td><td>0.670</td><td>0.391</td><td>0.668</td><td>0.430</td></tr><tr><td>Margin-MSE [17]</td><td>DistilBERT</td><td>66M</td><td>0.325</td><td>0.331</td><td>0.699</td><td>0.405</td><td>0.645</td><td>0.416</td></tr><tr><td>TAS-B [18]</td><td>DistilBERT</td><td>66M</td><td>0.344</td><td>0.351</td><td>0.717</td><td>0.447</td><td>0.685</td><td>0.455</td></tr><tr><td>CLDRD [57]</td><td>DistilBERT</td><td>66M</td><td>0.382</td><td>0.386</td><td>0.725</td><td>0.453</td><td>0.687</td><td>0.465</td></tr><tr><td>MRL (ours)</td><td>DistilBERT</td><td>66M</td><td>${\mathbf{{0.393}}}^{ * }$</td><td>${\mathbf{{0.402}}}^{ * }$</td><td>0.738</td><td>${\mathbf{{0.472}}}^{ * }$</td><td>${0.701}^{ * }$</td><td>${\mathbf{{0.479}}}^{ * }$</td></tr><tr><td colspan="9">Some Sparse Retrieval Models (For Reference)</td></tr><tr><td>BM25 [39]</td><td>-</td><td>-</td><td>0.187</td><td>0.196</td><td>0.497</td><td>0.290</td><td>0.487</td><td>0.288</td></tr><tr><td>DeepCT [12]</td><td>BERT-Base</td><td>110M</td><td>0.243</td><td>0.250</td><td>0.550</td><td>0.341</td><td>0.556</td><td>0.343</td></tr><tr><td>docT5query [32]</td><td>T5-Base</td><td>220M</td><td>0.272</td><td>0.281</td><td>0.642</td><td>0.403</td><td>0.619</td><td>0.407</td></tr><tr><td colspan="9">Multi Vector Dense Retrieval Model (For Reference)</td></tr><tr><td>ColBERTv2 [43]</td><td>DistilBERT</td><td>66M</td><td>0.384</td><td>0.389</td><td>0.733</td><td>0.464</td><td>0.712</td><td>0.473</td></tr></table>

<table><tbody><tr><td rowspan="2">模型</td><td rowspan="2">编码器</td><td rowspan="2">#参数</td><td colspan="2">MS MARCO开发集</td><td colspan="2">TREC-DL 2019</td><td colspan="2">TREC-DL 2020</td></tr><tr><td>前10名平均倒数排名（MRR@10）</td><td>$\mathbf{{MAP}}$</td><td>前10名归一化折损累积增益（NDCG@10）</td><td>$\mathbf{{MAP}}$</td><td>前10名归一化折损累积增益（NDCG@10）</td><td>平均准确率均值（MAP）</td></tr><tr><td colspan="9">单向量密集检索模型</td></tr><tr><td>ANCE [53]</td><td>BERT基础版</td><td>110M</td><td>0.330</td><td>0.336</td><td>0.648</td><td>0.371</td><td>0.646</td><td>0.408</td></tr><tr><td>ADORE [58]</td><td>BERT基础版</td><td>110M</td><td>0.347</td><td>0.352</td><td>0.683</td><td>0.419</td><td>0.666</td><td>0.442</td></tr><tr><td>火箭问答（RocketQA） [37]</td><td>ERNIE基础版</td><td>110M</td><td>0.370</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Contriever微调版（Contriever-FT） [20]</td><td>BERT基础版</td><td>110M</td><td>-</td><td>-</td><td>0.621</td><td>-</td><td>0.632</td><td>-</td></tr><tr><td>TCT-ColBERT [28]</td><td>BERT基础版</td><td>110M</td><td>0.335</td><td>0.342</td><td>0.670</td><td>0.391</td><td>0.668</td><td>0.430</td></tr><tr><td>边缘均方误差（Margin-MSE） [17]</td><td>蒸馏BERT（DistilBERT）</td><td>66M</td><td>0.325</td><td>0.331</td><td>0.699</td><td>0.405</td><td>0.645</td><td>0.416</td></tr><tr><td>TAS-B [18]</td><td>蒸馏BERT（DistilBERT）</td><td>66M</td><td>0.344</td><td>0.351</td><td>0.717</td><td>0.447</td><td>0.685</td><td>0.455</td></tr><tr><td>CLDRD [57]</td><td>蒸馏BERT（DistilBERT）</td><td>66M</td><td>0.382</td><td>0.386</td><td>0.725</td><td>0.453</td><td>0.687</td><td>0.465</td></tr><tr><td>最大相关性学习（MRL，我们的方法）</td><td>蒸馏BERT（DistilBERT）</td><td>66M</td><td>${\mathbf{{0.393}}}^{ * }$</td><td>${\mathbf{{0.402}}}^{ * }$</td><td>0.738</td><td>${\mathbf{{0.472}}}^{ * }$</td><td>${0.701}^{ * }$</td><td>${\mathbf{{0.479}}}^{ * }$</td></tr><tr><td colspan="9">一些稀疏检索模型（供参考）</td></tr><tr><td>二元独立模型（BM25） [39]</td><td>-</td><td>-</td><td>0.187</td><td>0.196</td><td>0.497</td><td>0.290</td><td>0.487</td><td>0.288</td></tr><tr><td>深度上下文词项（DeepCT） [12]</td><td>BERT基础版</td><td>110M</td><td>0.243</td><td>0.250</td><td>0.550</td><td>0.341</td><td>0.556</td><td>0.343</td></tr><tr><td>文档T5查询（docT5query） [32]</td><td>T5基础版</td><td>220M</td><td>0.272</td><td>0.281</td><td>0.642</td><td>0.403</td><td>0.619</td><td>0.407</td></tr><tr><td colspan="9">多向量密集检索模型（供参考）</td></tr><tr><td>ColBERTv2 [43]</td><td>蒸馏BERT（DistilBERT）</td><td>66M</td><td>0.384</td><td>0.389</td><td>0.733</td><td>0.464</td><td>0.712</td><td>0.473</td></tr></tbody></table>

<!-- Media -->

### 6.4 Experimental Results

### 6.4 实验结果

Baselines. We also compare against the following state-of-the-art dense retrieval models with single vector representations:

基线模型。我们还将其与以下具有单向量表示的最先进的密集检索模型进行了比较：

- ANCE [53] and ADORE [58]: two effective dense retrieval models based on BERT-Base [13] that use the model itself to mine hard negative documents.

- ANCE [53] 和 ADORE [58]：两种基于 BERT-Base [13] 的有效密集检索模型，它们使用模型本身来挖掘难负样本文档。

- RocketQA [37], Margin-MSE [17], and TAS-B [18]: effective dense retrieval models that use knowledge distillation from a BERT reranking model (a cross-encoder) in addition to various techniques for negative sampling.

- RocketQA [37]、Margin-MSE [17] 和 TAS-B [18]：有效的密集检索模型，除了采用各种负采样技术外，还利用 BERT 重排序模型（交叉编码器）进行知识蒸馏。

- Contriever-FT [20]: a single vector dense retrieval model that is pre-trained for retrieval tasks and then fine-tuned on MS MARCO. This model has shown effective performance on out-of-distribution target domain datasets.

- Contriever-FT [20]：一种单向量密集检索模型，它针对检索任务进行预训练，然后在 MS MARCO 上进行微调。该模型在分布外目标领域数据集上表现出了有效的性能。

- TCT-ColBERT [28]: a single vector dense retrieval model that is trained through knowledge distillation where a multi vector dense retrieval model (i.e., ColBERT [24]) is used as the teacher model.

- TCT-ColBERT [28]：一种单向量密集检索模型，通过知识蒸馏进行训练，其中多向量密集检索模型（即 ColBERT [24]）被用作教师模型。

- CLDRD [57]: the state-of-the-art single vector dense retrieval model that uses knowledge distillation from a reranking teacher model through gradual increase of training data difficulty (curriculum learning).

- CLDRD [57]：最先进的单向量密集检索模型，它通过逐步增加训练数据难度（课程学习）从重排序教师模型进行知识蒸馏。

Even though MRL is a single vector dense retrieval model, as a point of reference, we use a state-of-the-art dense retrieval model with multiple vectors (i.e., ColBERTv2 [43]). For demonstrating a fair comparison, all baselines are trained and tuned in the same way as the proposed approach.

尽管 MRL 是一种单向量密集检索模型，但作为参考，我们使用了一种具有多向量的最先进的密集检索模型（即 ColBERTv2 [43]）。为了进行公平比较，所有基线模型都以与所提出的方法相同的方式进行训练和调优。

We also compare our model against the following baselines that use inverted index for computing relevance scores (sometimes called sparse retrieval models):

我们还将我们的模型与以下使用倒排索引计算相关性得分的基线模型（有时称为稀疏检索模型）进行了比较：

- BM25 [39]: a simple yet strong term matching model for document retrieval that computes relevance scores based on term frequency in each document, document length, and inverse document frequency in the collection. We use the Galago search engine [36] to compute BM25 scores and tuned BM25 parameters using the training set.

- BM25 [39]：一种简单但强大的文档检索词匹配模型，它根据每个文档中的词频、文档长度和集合中的逆文档频率来计算相关性得分。我们使用 Galago 搜索引擎 [36] 来计算 BM25 得分，并使用训练集对 BM25 参数进行调优。

- DeepCT [12]: an approach that uses BERT to compute a weight for each word in the vocabulary for each document and query. Then words with highest weights are then selected and added to the inverted index with their weights. This approach can be seen as a contextual bag-of-words query and document expansion approach.

- DeepCT [12]：一种使用 BERT 为每个文档和查询计算词汇表中每个单词权重的方法。然后选择权重最高的单词，并将其及其权重添加到倒排索引中。这种方法可以看作是一种上下文词袋查询和文档扩展方法。

- docT5query [32]: a sequence-to-sequence model based on T5 [38] that is trained on MS MARCO to generate queries from any relevance passage. The documents are then expanded using the generated queries.

- docT5query [32]：一种基于 T5 [38] 的序列到序列模型，在 MS MARCO 上进行训练，以从任何相关段落生成查询。然后使用生成的查询对文档进行扩展。

The Passage Retrieval Results. The passage retrieval results are presented in Table 2. According to the table, all dense retrieval models perform substantially better than BM25 and DeepCT, demonstrating the effectiveness of such approaches for in-domain passage retrieval tasks. We observe that the approaches that use knowledge distillation (i.e., every dense retrieval model, except for ANCE, ADORE, and Contriever-FT) generally perform better than others. The recent CLDRD model shows the strongest retrieval results among all single vector dense retrieval models. The multi vector dense retrieval approach (ColBERTv2) outperforms all single vector dense retrieval baselines. Note that ColBERTv2 stores a vector for each token in the documents and thus it requires significantly larger storage for storing the ANN index and also suffers from substantially higher query latency (see Table 3 for more information). We show that MRL outperforms all baselines in terms of all the evaluation metrics used in the study. The improvements compared to all baselines are statistically significant, except for NDCG@10 in TREC-DL'19; the p_value (corrected using Bonferroni correction) for MRL versus CLDRD in this case was 0.07381 . Note that this dataset only contains 43 queries and significance tests are impacted by sampled size. MRL performs significantly better than any other baseline in this case.

段落检索结果。段落检索结果见表 2。根据该表，所有密集检索模型的性能都明显优于 BM25 和 DeepCT，这证明了此类方法在领域内段落检索任务中的有效性。我们观察到，使用知识蒸馏的方法（即除了 ANCE、ADORE 和 Contriever-FT 之外的每个密集检索模型）通常比其他方法表现更好。最近的 CLDRD 模型在所有单向量密集检索模型中显示出最强的检索结果。多向量密集检索方法（ColBERTv2）优于所有单向量密集检索基线模型。请注意，ColBERTv2 为文档中的每个标记存储一个向量，因此它需要显著更大的存储空间来存储 ANN 索引，并且查询延迟也显著更高（更多信息见表 3）。我们表明，MRL 在本研究中使用的所有评估指标方面都优于所有基线模型。与所有基线模型相比的改进在统计上是显著的，但在 TREC-DL'19 中的 NDCG@10 除外；在这种情况下，MRL 与 CLDRD 的 p 值（使用 Bonferroni 校正）为 0.07381。请注意，该数据集仅包含 43 个查询，显著性测试会受到样本大小的影响。在这种情况下，MRL 的表现明显优于任何其他基线模型。

<!-- Media -->

Table 3: A comparison of storage requirement and query latency between single vector and multi vector dense retrieval models with DistilBERT encoders on MS MARCO collection with 8.8 million passages. We ran this experiment on a machine with 16 Core i7-4790 CPU @ 3.60GHz.

表 3：在包含 880 万个段落的 MS MARCO 集合上，使用 DistilBERT 编码器的单向量和多向量密集检索模型之间的存储需求和查询延迟比较。我们在一台配备 16 核 i7 - 4790 CPU @ 3.60GHz 的机器上运行了此实验。

<table><tr><td/><td>storage requirement</td><td>query latency</td></tr><tr><td>Single vector DR</td><td>26GB</td><td>${89}\mathrm{\;{ms}}/$ query</td></tr><tr><td>Multi vector DR</td><td>192GB</td><td>438 ms / query</td></tr></table>

<table><tbody><tr><td></td><td>存储需求</td><td>查询延迟</td></tr><tr><td>单向量数据缩减（Single vector DR）</td><td>26GB</td><td>${89}\mathrm{\;{ms}}/$ 查询</td></tr><tr><td>多向量数据缩减（Multi vector DR）</td><td>192GB</td><td>每次查询438毫秒</td></tr></tbody></table>

Table 4: Sensitivity of MRL's retrieval performance to different values of $\beta$ .

表4：最大相关学习（MRL）检索性能对$\beta$不同取值的敏感性。

<table><tr><td rowspan="2"/><td colspan="2">MS MARCO DEV</td><td colspan="2">TREC-DL’19</td><td colspan="2">TREC-DL’20</td></tr><tr><td>MRR@10</td><td>$\mathbf{{MAP}}$</td><td>NDCG@10</td><td>$\mathbf{{MAP}}$</td><td>NDCG@10</td><td>MAP</td></tr><tr><td>$\beta  = {0.1}$</td><td>0.385</td><td>0.384</td><td>0.723</td><td>0.448</td><td>0.693</td><td>0.466</td></tr><tr><td>$\beta  = {0.25}$</td><td>0.399</td><td>0.415</td><td>0.743</td><td>0.468</td><td>0.704</td><td>0.478</td></tr><tr><td>$\beta  = {0.5}$</td><td>0.403</td><td>0.408</td><td>0.742</td><td>0.481</td><td>0.703</td><td>0.486</td></tr><tr><td>$\beta  = 1$</td><td>0.403</td><td>0.412</td><td>0.748</td><td>0.480</td><td>0.711</td><td>0.489</td></tr><tr><td>$\beta  = 5$</td><td>0.405</td><td>0.421</td><td>0.749</td><td>0.484</td><td>0.716</td><td>0.489</td></tr><tr><td>$\beta  = {10}$</td><td>0.402</td><td>0.421</td><td>0.758</td><td>0.489</td><td>0.701</td><td>0.483</td></tr></table>

<table><tbody><tr><td rowspan="2"></td><td colspan="2">微软机器阅读理解数据集开发集（MS MARCO DEV）</td><td colspan="2">2019年文本检索会议深度学习评测任务（TREC-DL’19）</td><td colspan="2">2020年文本检索会议深度学习评测任务（TREC-DL’20）</td></tr><tr><td>前10名平均倒数排名（MRR@10）</td><td>$\mathbf{{MAP}}$</td><td>前10名归一化折损累计增益（NDCG@10）</td><td>$\mathbf{{MAP}}$</td><td>前10名归一化折损累计增益（NDCG@10）</td><td>平均准确率均值（MAP）</td></tr><tr><td>$\beta  = {0.1}$</td><td>0.385</td><td>0.384</td><td>0.723</td><td>0.448</td><td>0.693</td><td>0.466</td></tr><tr><td>$\beta  = {0.25}$</td><td>0.399</td><td>0.415</td><td>0.743</td><td>0.468</td><td>0.704</td><td>0.478</td></tr><tr><td>$\beta  = {0.5}$</td><td>0.403</td><td>0.408</td><td>0.742</td><td>0.481</td><td>0.703</td><td>0.486</td></tr><tr><td>$\beta  = 1$</td><td>0.403</td><td>0.412</td><td>0.748</td><td>0.480</td><td>0.711</td><td>0.489</td></tr><tr><td>$\beta  = 5$</td><td>0.405</td><td>0.421</td><td>0.749</td><td>0.484</td><td>0.716</td><td>0.489</td></tr><tr><td>$\beta  = {10}$</td><td>0.402</td><td>0.421</td><td>0.758</td><td>0.489</td><td>0.701</td><td>0.483</td></tr></tbody></table>

<!-- Media -->

Parameter Sensitivity Analysis. To measure the sensitivity of MRL’s performance to the value of $\beta$ ,we change $\beta$ from 0.1 to 10 and report the results in Table 4. To get a sense of the impact of these values, please see Figure 2. The results show that the model is not sensitive to the value of $\beta$ unless it is smaller than or equal to $\leq  {0.25}$ . Therefore,for a $\beta$ value of around 1 or larger,the model shows a robust and strong performance.

参数敏感性分析。为了衡量MRL性能对$\beta$值的敏感性，我们将$\beta$的值从0.1改变到10，并将结果列于表4中。若想了解这些值的影响，请参考图2。结果表明，除非$\beta$的值小于或等于$\leq  {0.25}$，否则模型对$\beta$的值不敏感。因此，当$\beta$的值约为1或更大时，模型表现出稳健且强大的性能。

The Zero-Shot Retrieval Results. All datasets used in Table 2 are based on the MS MARCO passage collection and their queries are similar to that of our training set. To evaluate the model's performance under distribution shift, we conduct a zero-shot retrieval experiment on four diverse datasets: SciFact, FiQA, TREC COVID, and CQADupStack (see Section 6.1). In this experiment, we do not re-train any model and the ones trained on MS MARCO training set and used in Table 2 are used for zero-shot evaluation on these datasets. The results are reported in Table 5. We observe that many neural retrieval models struggle with outperforming BM25 on SciFact and TREC COVID datasets. In general, the improvements observed compared to BM25 by the best performing models are not as large as the ones we observe in Table 2. This highlights the difficulty of handling domain shift by neural retrieval models. Generally speaking, the multi vector dense retrieval model (ColBERTv2) shows a more robust performance in zero-shot settings. It outperforms all single vector dense retrieval models on TREC COVID and CQADupStack. MRL performs better on the other two datasets: SciFact and FiQA. Again, we highlight that MRL has substantially lower storage requirements compared to ColBERTv2 and it also has significantly faster query processing time. Refer to Table 3 for more information.

零样本检索结果。表2中使用的所有数据集均基于MS MARCO段落集合，且其查询与我们训练集的查询相似。为了评估模型在分布偏移情况下的性能，我们在四个不同的数据集上进行了零样本检索实验：SciFact、FiQA、TREC COVID和CQADupStack（详见6.1节）。在这个实验中，我们不重新训练任何模型，而是使用在MS MARCO训练集上训练的模型（即表2中使用的模型）对这些数据集进行零样本评估。结果列于表5中。我们观察到，许多神经检索模型在SciFact和TREC COVID数据集上难以超越BM25。总体而言，与BM25相比，表现最佳的模型所取得的改进并不像我们在表2中观察到的那么大。这凸显了神经检索模型处理领域偏移的难度。一般来说，多向量密集检索模型（ColBERTv2）在零样本设置下表现出更稳健的性能。它在TREC COVID和CQADupStack数据集上优于所有单向量密集检索模型。MRL在另外两个数据集（SciFact和FiQA）上表现更好。再次强调，与ColBERTv2相比，MRL的存储需求大幅降低，查询处理时间也显著缩短。更多信息请参考表3。

<!-- Media -->

<!-- figureText: $\beta  = {0.1}$ $\beta  = {0.25}$ $- \beta  = 1$ $- \beta  = 5$ -->

<img src="https://cdn.noedgeai.com/0195a507-1150-7ca4-ab73-3cb2f130644c_7.jpg?x=990&y=236&w=583&h=476&r=0"/>

Figure 2: The softplus curve that is used to compute the variance vector for different values of $\beta$ . Softplus is a monotonic and increasing function with a lower bound of zero. It’s value for large $x$ values can be approximated using the linear function $y = x$ for numeric stability.

图2：用于计算不同$\beta$值下方差向量的软加（softplus）曲线。软加是一个单调递增函数，下界为零。为了数值稳定性，对于较大的$x$值，其值可以用线性函数$y = x$近似。

Table 5: The zero-shot retrieval results obtained by the proposed approach and the baselines, in terms of NDCG@10.The highest value in each column is bold-faced. The superscript ${}^{ * }$ denotes statistically significant improvements compared to all the baselines based on two-tailed paired t-test with Bonferroni correction at the 95% confidence level.

表5：所提出的方法和基线模型在零样本检索方面的结果，以NDCG@10衡量。每列中的最高值用粗体表示。上标${}^{ * }$表示在95%置信水平下，基于经过邦费罗尼校正（Bonferroni correction）的双尾配对t检验，与所有基线模型相比具有统计学显著的改进。

<table><tr><td>Model</td><td>SciFact</td><td>FiQA</td><td>TREC COVID</td><td>CQA DupStack</td></tr><tr><td colspan="5">Single Vector DR Models</td></tr><tr><td>ANCE [53]</td><td>0.507</td><td>0.295</td><td>0.654</td><td>0.296</td></tr><tr><td>ADORE [58]</td><td>0.514</td><td>0.255</td><td>0.590</td><td>0.273</td></tr><tr><td>RocketQA [37]</td><td>0.606</td><td>0.319</td><td>0.658</td><td>0.316</td></tr><tr><td>Contriever-FT [20]</td><td>0.677</td><td>0.329</td><td>0.596</td><td>0.321</td></tr><tr><td>TCT-ColBERT [28]</td><td>0.614</td><td>0.316</td><td>0.661</td><td>0.309</td></tr><tr><td>Margin-MSE [17]</td><td>0.608</td><td>0.298</td><td>0.673</td><td>0.297</td></tr><tr><td>TAS-B [18]</td><td>0.643</td><td>0.300</td><td>0.481</td><td>0.314</td></tr><tr><td>CLDRD [57]</td><td>0.637</td><td>0.348</td><td>0.571</td><td>0.327</td></tr><tr><td>MRL (ours)</td><td>${\mathbf{{0.683}}}^{ * }$</td><td>${\mathbf{{0.371}}}^{ * }$</td><td>0.668</td><td>${0.341}^{ * }$</td></tr><tr><td colspan="5">Some Sparse Retrieval Models (For Reference)</td></tr><tr><td>BM25 [39]</td><td>0.665</td><td>0.236</td><td>0.656</td><td>0.299</td></tr><tr><td>DeepCT [12]</td><td>0.630</td><td>0.191</td><td>0.406</td><td>0.268</td></tr><tr><td>docT5query [32]</td><td>0.675</td><td>0.291</td><td>0.713</td><td>0.325</td></tr><tr><td colspan="5">Models (Fo)Reference)</td></tr><tr><td>ColBERTv2 [43] (DistilBERT)</td><td>0.682</td><td>0.359</td><td>0.696</td><td>0.357</td></tr></table>

<table><tbody><tr><td>模型</td><td>科学事实数据集（SciFact）</td><td>金融问答数据集（FiQA）</td><td>TREC新冠数据集（TREC COVID）</td><td>问答重复堆叠数据集（CQA DupStack）</td></tr><tr><td colspan="5">单向量密集检索模型（Single Vector DR Models）</td></tr><tr><td>ANCE模型 [53]</td><td>0.507</td><td>0.295</td><td>0.654</td><td>0.296</td></tr><tr><td>ADORE模型 [58]</td><td>0.514</td><td>0.255</td><td>0.590</td><td>0.273</td></tr><tr><td>火箭问答模型（RocketQA） [37]</td><td>0.606</td><td>0.319</td><td>0.658</td><td>0.316</td></tr><tr><td>微调的Contriever模型（Contriever - FT） [20]</td><td>0.677</td><td>0.329</td><td>0.596</td><td>0.321</td></tr><tr><td>TCT - ColBERT模型 [28]</td><td>0.614</td><td>0.316</td><td>0.661</td><td>0.309</td></tr><tr><td>Margin - MSE模型 [17]</td><td>0.608</td><td>0.298</td><td>0.673</td><td>0.297</td></tr><tr><td>TAS - B模型 [18]</td><td>0.643</td><td>0.300</td><td>0.481</td><td>0.314</td></tr><tr><td>CLDRD模型 [57]</td><td>0.637</td><td>0.348</td><td>0.571</td><td>0.327</td></tr><tr><td>最大相关性学习模型（MRL，我们的模型）</td><td>${\mathbf{{0.683}}}^{ * }$</td><td>${\mathbf{{0.371}}}^{ * }$</td><td>0.668</td><td>${0.341}^{ * }$</td></tr><tr><td colspan="5">一些稀疏检索模型（供参考）</td></tr><tr><td>BM25模型 [39]</td><td>0.665</td><td>0.236</td><td>0.656</td><td>0.299</td></tr><tr><td>DeepCT模型 [12]</td><td>0.630</td><td>0.191</td><td>0.406</td><td>0.268</td></tr><tr><td>docT5query模型 [32]</td><td>0.675</td><td>0.291</td><td>0.713</td><td>0.325</td></tr><tr><td colspan="5">模型（供参考）</td></tr><tr><td>ColBERTv2模型 [43]（DistilBERT）</td><td>0.682</td><td>0.359</td><td>0.696</td><td>0.357</td></tr></tbody></table>

Table 6: Pre-retrieval query performance prediction results in terms of Pearson’s $\rho$ and Kendall’s $\tau$ correlations. The superscript ${}^{ \dagger  }$ denotes that the obtained correlations by MRL $\left| {\sum }_{Q}\right|$ are significant.

表6：基于皮尔逊（Pearson）$\rho$相关性和肯德尔（Kendall）$\tau$相关性的预检索查询性能预测结果。上标${}^{ \dagger  }$表示MRL $\left| {\sum }_{Q}\right|$获得的相关性具有显著性。

<table><tr><td rowspan="2">QPP Model</td><td colspan="2">TREC-DL’19</td><td colspan="2">TREC-DL’20</td></tr><tr><td>$\mathbf{P} - \rho$</td><td>$\mathrm{K} - \tau$</td><td>P- $\rho$</td><td>$\mathbf{K} - \tau$</td></tr><tr><td>Max VAR [5]</td><td>0.138</td><td>0.148</td><td>0.230</td><td>0.266</td></tr><tr><td>Max SCQ [60]</td><td>0.119</td><td>0.109</td><td>0.182</td><td>0.237</td></tr><tr><td>Avg IDF [5]</td><td>0.172</td><td>0.166</td><td>0.246</td><td>0.240</td></tr><tr><td>SCS [16]</td><td>0.160</td><td>0.174</td><td>0.231</td><td>0.275</td></tr><tr><td>Max PMI [15]</td><td>0.098</td><td>0.116</td><td>0.155</td><td>0.194</td></tr><tr><td>${P}_{\text{clarity }}\left\lbrack  {40}\right\rbrack$</td><td>0.167</td><td>0.174</td><td>0.191</td><td>0.217</td></tr><tr><td>Max DC [2]</td><td>0.341</td><td>0.294</td><td>0.234</td><td>0.244</td></tr><tr><td>MRL $\left| {\sum }_{Q}\right|$</td><td>${0.271}^{ \dagger  }$</td><td>${0.259}^{ \dagger  }$</td><td>${\mathbf{{0.272}}}^{ \dagger  }$</td><td>${\mathbf{{0.298}}}^{ \dagger  }$</td></tr></table>

<table><tbody><tr><td rowspan="2">查询性能预测模型（QPP Model）</td><td colspan="2">2019年文本检索会议深度学习评测任务（TREC-DL’19）</td><td colspan="2">2020年文本检索会议深度学习评测任务（TREC-DL’20）</td></tr><tr><td>$\mathbf{P} - \rho$</td><td>$\mathrm{K} - \tau$</td><td>P- $\rho$</td><td>$\mathbf{K} - \tau$</td></tr><tr><td>最大方差（Max VAR [5]）</td><td>0.138</td><td>0.148</td><td>0.230</td><td>0.266</td></tr><tr><td>最大子查询相关性（Max SCQ [60]）</td><td>0.119</td><td>0.109</td><td>0.182</td><td>0.237</td></tr><tr><td>平均逆文档频率（Avg IDF [5]）</td><td>0.172</td><td>0.166</td><td>0.246</td><td>0.240</td></tr><tr><td>语义内容得分（SCS [16]）</td><td>0.160</td><td>0.174</td><td>0.231</td><td>0.275</td></tr><tr><td>最大点互信息（Max PMI [15]）</td><td>0.098</td><td>0.116</td><td>0.155</td><td>0.194</td></tr><tr><td>${P}_{\text{clarity }}\left\lbrack  {40}\right\rbrack$</td><td>0.167</td><td>0.174</td><td>0.191</td><td>0.217</td></tr><tr><td>最大依赖度中心性（Max DC [2]）</td><td>0.341</td><td>0.294</td><td>0.234</td><td>0.244</td></tr><tr><td>最大风险最小化损失（MRL $\left| {\sum }_{Q}\right|$）</td><td>${0.271}^{ \dagger  }$</td><td>${0.259}^{ \dagger  }$</td><td>${\mathbf{{0.272}}}^{ \dagger  }$</td><td>${\mathbf{{0.298}}}^{ \dagger  }$</td></tr></tbody></table>

<!-- Media -->

Exploring the Learned Variance Vectors. In our exploration towards understanding the representations learned by MRL, we realize that the norm of our covariance matrix for each query is correlated with the ranking performance of our retrieval model for that query. This observation motivated us to use the learned $\left| {\sum }_{Q}\right|$ for each query as a pre-retrieval query performance predictor (QPP). Some other well known pre-retrieval (i.e., based solely on the query and collection content, not any retrieved results) performance predictors include distribution of the query term IDF weights, the similarity between a query and the underlying collection; and the variability with which query terms occur in documents [60].

探索学习到的方差向量。在我们试图理解多变量表示学习（MRL）所学习到的表示的过程中，我们发现每个查询的协方差矩阵的范数与我们的检索模型针对该查询的排序性能相关。这一观察结果促使我们将为每个查询学习到的$\left| {\sum }_{Q}\right|$用作预检索查询性能预测器（QPP）。其他一些知名的预检索（即仅基于查询和集合内容，而非任何检索结果）性能预测器包括查询词逆文档频率（IDF）权重的分布、查询与基础集合之间的相似度，以及查询词在文档中出现的变异性[60]。

We compare our prediction against some of these commonly used unsupervised pre-retrieval QPP methods in Table 6. They include:

我们在表6中将我们的预测结果与一些常用的无监督预检索QPP方法进行了比较。这些方法包括：

- Max VAR [5]: VAR uses the maximum variance of query term weight in the collection.

- 最大方差（Max VAR）[5]：VAR使用集合中查询词权重的最大方差。

- Max SCQ [60]: It computes a TF-IDF formulation for each query term and returns the maximum value.

- 最大查询词得分（Max SCQ）[60]：它为每个查询词计算一个词频 - 逆文档频率（TF - IDF）公式，并返回最大值。

- Avg IDF [5]: This baseline uses an inverse document frequency formulation for each query term and return the average score.

- 平均逆文档频率（Avg IDF）[5]：此基线方法为每个查询词使用逆文档频率公式，并返回平均得分。

- SCS [16]: It computes the KL divergence between the unigram query language model and the collection language model.

- 对称散度得分（SCS）[16]：它计算一元查询语言模型与集合语言模型之间的KL散度。

- Max PMI [15]: It uses the point-wise mutual information of query terms in the collection and returns the maximum value.

- 最大点互信息（Max PMI）[15]：它使用集合中查询词的点互信息，并返回最大值。

- ${P}_{\text{clarity }}\left\lbrack  {40}\right\rbrack$ : This baseline uses Gaussian mixture models in the embedding space as soft clustering and uses term similarity to compute the probability of each query term being generated by each cluster.

- ${P}_{\text{clarity }}\left\lbrack  {40}\right\rbrack$：此基线方法在嵌入空间中使用高斯混合模型进行软聚类，并使用词相似度来计算每个查询词由每个聚类生成的概率。

- Max DC [2]: This approach uses pre-trained embeddings to construct an ego network and computes Degree Centrality (DC) as the number of links incident upon the ego.

- 最大度中心性（Max DC）[2]：这种方法使用预训练的嵌入来构建自我中心网络，并将度中心性（DC）计算为与自我中心节点相连的边的数量。

Following the QPP literature [5, 11, 15, 54], we use the following two evaluation metrics: Pearson’s $\rho$ correlation (a linear correlation metric) and Kendall’s $\tau$ correlation (a rank-based correlation metric). We only report the results on the TREC DL datasets, since MS MARCO DEV only contains one relevant document per query and may not be suitable for performance prediction tasks. We observe that relative to existing pre-retrieval QPP approaches,MRL $\left| {\sum }_{Q}\right|$ has a high correlation with the actual retrieval performance. All of these correlations are significant $\left( {{p}_{ - }\text{value} < {0.05}}\right)$ . Note that MRL is not optimized for performance prediction and its goal is not QPP and these results just provide insights into what a model with multivariate representation may learn.

遵循QPP相关文献[5, 11, 15, 54]，我们使用以下两个评估指标：皮尔逊$\rho$相关性（一种线性相关指标）和肯德尔$\tau$相关性（一种基于排名的相关指标）。我们仅报告在TREC DL数据集上的结果，因为MS MARCO DEV每个查询仅包含一个相关文档，可能不适用于性能预测任务。我们观察到，相对于现有的预检索QPP方法，MRL $\left| {\sum }_{Q}\right|$与实际检索性能具有高度相关性。所有这些相关性都是显著的$\left( {{p}_{ - }\text{value} < {0.05}}\right)$。请注意，MRL并非针对性能预测进行优化，其目标也不是QPP，这些结果只是为具有多变量表示的模型可能学习到的内容提供了见解。

## 7 CONCLUSIONS AND FUTURE WORK

## 7 结论与未来工作

This paper introduced MRL- a novel representation learning paradigm for neural information retrieval. It uses multivariate normal distributions for representing queries and documents, where the mean and variance vectors for each input query or document are learned using large language models. We suggested a theoretically sound and empirically strong retrieval model based on multivariate Kullback-Leibler (KL) divergence between the learned representations. We showed that the proposed formulated can be approximated and used in existing approximate nearest neighbor search algorithms for efficient retrieval. Experiments on a wide range of datasets showed that MRL advances state-of-the-art in single vector dense retrieval and sometimes even outperforms the state-of-the-art multi vector dense retrieval model, while being more efficient and requiring orders of magnitude less storage. We showed that the norm of variance vectors learned for each query is correlated with the model's retrieval performance, and thus it can be used as a pre-retrieval query performance predictor.

本文介绍了多变量表示学习（MRL）——一种用于神经信息检索的新型表示学习范式。它使用多元正态分布来表示查询和文档，其中每个输入查询或文档的均值和方差向量是使用大语言模型学习得到的。我们基于学习到的表示之间的多元Kullback - Leibler（KL）散度提出了一个理论上合理且经验上有效的检索模型。我们表明，所提出的公式可以进行近似处理，并用于现有的近似最近邻搜索算法以实现高效检索。在广泛的数据集上进行的实验表明，MRL在单向量密集检索方面达到了当前的先进水平，有时甚至优于最先进的多向量密集检索模型，同时效率更高，所需的存储空间也大幅减少。我们还表明，为每个查询学习到的方差向量的范数与模型的检索性能相关，因此它可以用作预检索查询性能预测器。

Multivariate representation learning opens up many exciting directions for future exploration. Given the flexibility of multivariate normal distributions, the representations learned by the model can be easily extended. For instance, linear interpolation of multivariate normals is a multivariate normal distribution. Therefore, one can easily extend this formulation to many settings, such as (pseudo-) relevance feedback, context-aware retrieval, session search, personalized search, and conversational retrieval. Furthermore, such a representation learning approach can be extended to applications beyond standard IR problems. They can be used in representation learning for users and items in collaborative filtering models, graph embedding for link prediction and knowledge graph construction, and information extraction. Another promising research direction is enhancing retrieval-enhanced machine learning (REML) models [56] using multivariate representations. MRL enables REML models to be aware of the retrieval confidence and data distribution for making final predictions.

多变量表示学习为未来的探索开辟了许多令人兴奋的方向。鉴于多元正态分布的灵活性，模型学习到的表示可以很容易地进行扩展。例如，多元正态分布的线性插值仍然是一个多元正态分布。因此，可以很容易地将此公式扩展到许多场景，如（伪）相关性反馈、上下文感知检索、会话搜索、个性化搜索和对话式检索。此外，这种表示学习方法可以扩展到标准信息检索问题之外的应用。它们可用于协同过滤模型中用户和物品的表示学习、用于链接预测和知识图谱构建的图嵌入，以及信息提取。另一个有前途的研究方向是使用多变量表示来增强检索增强机器学习（REML）模型[56]。MRL使REML模型能够在进行最终预测时考虑检索置信度和数据分布。

## ACKNOWLEDGMENTS

## 致谢

This work was supported in part by the Google Visiting Scholar program and in part by the Center for Intelligent Information Retrieval. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect those of the sponsors.

这项工作部分得到了谷歌访问学者计划的支持，部分得到了智能信息检索中心的支持。本材料中表达的任何观点、研究结果、结论或建议均为作者本人的观点，不一定反映资助者的观点。

## REFERENCES

## 参考文献

[1] Negar Arabzadeh, Bhaskar Mitra, and Ebrahim Bagheri. 2021. MS MARCO Chameleons: Challenging the MS MARCO Leaderboard with Extremely Obstinate Queries. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management. 4426-4435.

[1] 内加尔·阿拉布扎德赫（Negar Arabzadeh）、巴斯卡尔·米特拉（Bhaskar Mitra）和易卜拉欣·巴盖里（Ebrahim Bagheri）。2021年。MS MARCO变色龙：用极难处理的查询挑战MS MARCO排行榜。见《第30届ACM信息与知识管理国际会议论文集》。4426 - 4435页。

[2] Negar Arabzadeh, Fattane Zarrinkalam, Jelena Jovanovic, Feras N. Al-Obeidat, and Ebrahim Bagheri. 2020. Neural embedding-based specificity metrics for pre-retrieval query performance prediction. Inf. Process. Manag. 57, 4 (2020), 102248.

[2] 内加尔·阿拉布扎德赫（Negar Arabzadeh）、法塔内·扎林卡拉姆（Fattane Zarrinkalam）、耶莱娜·约万诺维奇（Jelena Jovanovic）、费拉斯·N·奥贝达特（Feras N. Al - Obeidat）和易卜拉欣·巴盖里（Ebrahim Bagheri）。2020年。基于神经嵌入的特异性指标用于预检索查询性能预测。《信息处理与管理》，57卷，第4期（2020年），文章编号102248。

[3] Christopher J. C. Burges. 2010. From RankNet to LambdaRank to LambdaMART: An Overview. Technical Report. Microsoft Research.

[3] 克里斯托弗·J·C·伯吉斯（Christopher J. C. Burges）。2010年。从RankNet到LambdaRank再到LambdaMART：综述。技术报告。微软研究院。

[4] Daniel Fernando Campos, Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, Li Deng, and Bhaskar Mitra. 2016. MS MARCO: A Human Generated MAchine Reading COmprehension Dataset. 30th Conference on Neural Information Processing Systems, NIPS (2016).

[4] 丹尼尔·费尔南多·坎波斯（Daniel Fernando Campos）、特里·阮（Tri Nguyen）、米尔·罗森伯格（Mir Rosenberg）、宋霞（Xia Song）、高剑锋（Jianfeng Gao）、索拉布·蒂瓦里（Saurabh Tiwary）、兰甘·马朱姆德（Rangan Majumder）、李邓（Li Deng）和巴斯卡尔·米特拉（Bhaskar Mitra）。2016年。MS MARCO：一个人工生成的机器阅读理解数据集。第30届神经信息处理系统会议，NIPS（2016年）。

[5] David Carmel and Elad Yom-Tov. 2010. Estimating the Query Difficulty for Information Retrieval (1st ed.). Morgan and Claypool Publishers.

[5] 大卫·卡梅尔（David Carmel）和埃拉德·约姆 - 托夫（Elad Yom - Tov）。2010年。信息检索查询难度估计（第1版）。摩根与克莱普尔出版社。

[6] Charles L.A. Clarke, Maheedhar Kolla, Gordon V. Cormack, Olga Vechtomova, Azin Ashkan, Stefan Büttcher, and Ian MacKinnon. 2008. Novelty and Diversity in Information Retrieval Evaluation. In Proceedings of the 31st Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '08). Association for Computing Machinery, New York, NY, USA, 659-666. https://doi.org/10.1145/1390334.1390446

[6] 查尔斯·L·A·克拉克（Charles L.A. Clarke）、马赫德哈尔·科拉（Maheedhar Kolla）、戈登·V·科马克（Gordon V. Cormack）、奥尔加·韦赫托莫娃（Olga Vechtomova）、阿津·阿什坎（Azin Ashkan）、斯特凡·比特彻（Stefan Büttcher）和伊恩·麦金农（Ian MacKinnon）。2008年。信息检索评估中的新颖性和多样性。见《第31届ACM信息检索研究与发展年度国际会议论文集（SIGIR '08）》。美国计算机协会，纽约，美国，659 - 666页。https://doi.org/10.1145/1390334.1390446

[7] Daniel Cohen, Bhaskar Mitra, Oleg Lesota, Navid Rekabsaz, and Carsten Eickhoff. 2021. Not All Relevance Scores Are Equal: Efficient Uncertainty and Calibration Modeling for Deep Retrieval Models. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '21). Association for Computing Machinery, New York, NY, USA, 654-664. https://doi.org/10.1145/3404835.3462951

[7] 丹尼尔·科恩（Daniel Cohen）、巴斯卡尔·米特拉（Bhaskar Mitra）、奥列格·莱索塔（Oleg Lesota）、纳维德·雷卡布萨兹（Navid Rekabsaz）和卡斯滕·艾克霍夫（Carsten Eickhoff）。2021年。并非所有相关性得分都相同：深度检索模型的高效不确定性和校准建模。见《第44届ACM信息检索研究与发展国际会议论文集（SIGIR '21）》。美国计算机协会，纽约，美国，654 - 664页。https://doi.org/10.1145/3404835.3462951

[8] Kevyn Collins-Thompson and Jamie Callan. 2007. Estimation and use of uncertainty in pseudo-relevance feedback. In Proceedings of the 30th annual international ACM SIGIR conference on Research and development in information retrieval. 303-310.

[8] 凯文·柯林斯 - 汤普森（Kevyn Collins - Thompson）和杰米·卡伦（Jamie Callan）。2007年。伪相关反馈中不确定性的估计与应用。见《第30届ACM信息检索研究与发展年度国际会议论文集》。303 - 310页。

[9] Nick Craswell, Bhaskar Mitra, Emine Yilmaz, and Daniel Campos. 2019. Overview of the TREC 2019 Deep Learning Track. In TREC.

[9] 尼克·克拉斯韦尔（Nick Craswell）、巴斯卡尔·米特拉（Bhaskar Mitra）、埃米内·伊尔马兹（Emine Yilmaz）和丹尼尔·坎波斯（Daniel Campos）。2019年。2019年TREC深度学习赛道综述。见《TREC会议论文集》。

[10] Nick Craswell, Bhaskar Mitra, Emine Yilmaz, and Daniel Campos. 2020. Overview of the TREC 2020 Deep Learning Track. In TREC.

[10] 尼克·克拉斯韦尔（Nick Craswell）、巴斯卡尔·米特拉（Bhaskar Mitra）、埃米内·伊尔马兹（Emine Yilmaz）和丹尼尔·坎波斯（Daniel Campos）。2020年。2020年TREC深度学习赛道综述。见《TREC会议论文集》。

[11] Steve Cronen-Townsend, Yun Zhou, and W. Bruce Croft. 2002. Predicting Query Performance. In Proceedings of the 25th Annual International ACM SI-GIR Conference on Research and Development in Information Retrieval (SIGIR '02). Association for Computing Machinery, New York, NY, USA, 299-306. https://doi.org/10.1145/564376.564429

[11] 史蒂夫·克罗宁 - 汤森（Steve Cronen - Townsend）、周云（Yun Zhou）和W·布鲁斯·克罗夫特（W. Bruce Croft）。2002年。查询性能预测。见《第25届ACM信息检索研究与发展年度国际会议论文集（SIGIR '02）》。美国计算机协会，纽约，美国，299 - 306页。https://doi.org/10.1145/564376.564429

[12] Zhuyun Dai and Jamie Callan. 2020. Context-Aware Sentence/Passage Term Importance Estimation For First Stage Retrieval. Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval abs/1910.10687 (2020).

[12] 戴竹云（Zhuyun Dai）和杰米·卡伦（Jamie Callan）。2020年。用于第一阶段检索的上下文感知句子/段落术语重要性估计。《第43届ACM信息检索研究与发展国际会议论文集》，论文编号abs/1910.10687（2020年）。

[13] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). Association for Computational Linguistics, Minneapolis, Minnesota, 4171-4186. https://doi.org/10.18653/v1/N19-1423

[13] 雅各布·德夫林（Jacob Devlin）、张明伟（Ming - Wei Chang）、肯顿·李（Kenton Lee）和克里斯蒂娜·图托纳娃（Kristina Toutanova）。2019年。BERT：用于语言理解的深度双向变换器预训练。见《2019年北美计算语言学协会会议：人类语言技术卷1（长论文和短论文）》。计算语言学协会，明尼阿波利斯，明尼苏达州，4171 - 4186页。https://doi.org/10.18653/v1/N19 - 1423

[14] Yarin Gal and Zoubin Ghahramani. 2016. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In international conference on machine learning. PMLR, 1050-1059.

[14] 亚林·加尔（Yarin Gal）和邹宾·加哈拉尼（Zoubin Ghahramani）。2016年。将Dropout作为贝叶斯近似：在深度学习中表示模型不确定性。见《国际机器学习会议》。机器学习研究会议录，1050 - 1059页。

[15] Claudia Hauff. 2010. Predicting the Effectiveness of Queries and Retrieval Systems. SIGIR Forum 44, 1 (aug 2010), 88. https://doi.org/10.1145/1842890.1842906

[15] 克劳迪娅·豪夫（Claudia Hauff）. 2010年. 预测查询和检索系统的有效性. 《SIGIR论坛》44, 1（2010年8月）, 88. https://doi.org/10.1145/1842890.1842906

[16] Ben He and Iadh Ounis. 2004. Inferring Query Performance Using Pre-retrieval Predictors. In String Processing and Information Retrieval, Alberto Apostolico and Massimo Melucci (Eds.). Springer Berlin Heidelberg, Berlin, Heidelberg, 43-54.

[16] 何本（Ben He）和伊阿德·乌尼斯（Iadh Ounis）. 2004年. 使用预检索预测器推断查询性能. 收录于《字符串处理与信息检索》, 阿尔贝托·阿波斯托利科（Alberto Apostolico）和马西莫·梅卢奇（Massimo Melucci） 编. 德国施普林格出版社, 柏林, 海德堡, 43 - 54.

[17] Sebastian Hofstätter, Sophia Althammer, Michael Schröder, Mete Sertkan, and Allan Hanbury. 2020. Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation. ArXiv abs/2010.02666 (2020).

[17] 塞巴斯蒂安·霍夫施塔特（Sebastian Hofstätter）、索菲娅·阿尔塔默（Sophia Althammer）、迈克尔·施罗德（Michael Schröder）、梅特·塞尔特坎（Mete Sertkan）和艾伦·汉伯里（Allan Hanbury）. 2020年. 通过跨架构知识蒸馏改进高效神经排序模型. 《arXiv预印本》abs/2010.02666（2020年）.

[18] Sebastian Hofstätter, Sheng-Chieh Lin, Jheng-Hong Yang, Jimmy J. Lin, and Allan Hanbury. 2021. Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling. Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (2021).

[18] 塞巴斯蒂安·霍夫施塔特（Sebastian Hofstätter）、林圣杰（Sheng - Chieh Lin）、杨政宏（Jheng - Hong Yang）、林吉米·J（Jimmy J. Lin）和艾伦·汉伯里（Allan Hanbury）. 2021年. 通过平衡主题感知采样高效训练有效的密集检索器. 《第44届国际ACM SIGIR信息检索研究与发展会议论文集》（2021年）.

[19] Doris Hoogeveen, Karin M. Verspoor, and Timothy Baldwin. 2015. CQADup-Stack: A Benchmark Data Set for Community Question-Answering Research. In Proceedings of the 20th Australasian Document Computing Symposium (ADCS '15). Association for Computing Machinery, New York, NY, USA, Article 3, 8 pages. https://doi.org/10.1145/2838931.2838934

[19] 多丽丝·胡格文（Doris Hoogeveen）、卡琳·M·弗斯波尔（Karin M. Verspoor）和蒂莫西·鲍德温（Timothy Baldwin）. 2015年. CQADup - Stack：社区问答研究的基准数据集. 收录于《第20届澳大利亚文档计算研讨会论文集》（ADCS '15）. 美国计算机协会, 美国纽约州纽约市, 文章编号3, 8页. https://doi.org/10.1145/2838931.2838934

[20] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bo-janowski, Armand Joulin, and Edouard Grave. 2022. Unsupervised Dense Information Retrieval with Contrastive Learning. Transactions on Machine Learning Research (2022).

[20] 高蒂尔·伊扎卡尔（Gautier Izacard）、玛蒂尔德·卡龙（Mathilde Caron）、卢卡斯·侯赛尼（Lucas Hosseini）、塞巴斯蒂安·里德尔（Sebastian Riedel）、彼得·博亚诺夫斯基（Piotr Bojanowski）、阿尔芒·朱兰（Armand Joulin）和爱德华·格雷夫（Edouard Grave）. 2022年. 通过对比学习进行无监督密集信息检索. 《机器学习研究汇刊》（2022年）.

[21] Kalervo Järvelin and Jaana Kekäläinen. 2002. Cumulated Gain-Based Evaluation of IR Techniques. ACM Trans. Inf. Syst. 20, 4 (oct 2002), 422-446. https://doi.org/ 10.1145/582415.582418

[21] 卡莱沃·亚尔维林（Kalervo Järvelin）和亚娜·凯卡拉伊宁（Jaana Kekäläinen）. 2002年. 基于累积增益的信息检索技术评估. 《ACM信息系统汇刊》20, 4（2002年10月）, 422 - 446. https://doi.org/ 10.1145/582415.582418

[22] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-scale similarity search with GPUs. IEEE Transactions on Big Data 7, 3 (2019), 535-547.

[22] 杰夫·约翰逊（Jeff Johnson）、马蒂亚斯·杜泽（Matthijs Douze）和埃尔韦·热古（Hervé Jégou）. 2019年. 使用GPU进行十亿级相似性搜索. 《IEEE大数据汇刊》7, 3（2019年）, 535 - 547.

[23] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for Open-Domain Question Answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP '20). Association for Computational Linguistics, Online, 6769-6781. https://doi.org/10.18653/v1/2020.emnlp-main.550

[23] 弗拉基米尔·卡尔普欣（Vladimir Karpukhin）、巴拉斯·奥古兹（Barlas Oguz）、闵世元（Sewon Min）、帕特里克·刘易斯（Patrick Lewis）、莱德尔·吴（Ledell Wu）、谢尔盖·叶杜诺夫（Sergey Edunov）、陈丹琦（Danqi Chen）和易文涛（Wen - tau Yih）. 2020年. 用于开放域问答的密集段落检索. 收录于《2020年自然语言处理经验方法会议论文集》（EMNLP '20）. 计算语言学协会, 线上会议, 6769 - 6781. https://doi.org/10.18653/v1/2020.emnlp - main.550

[24] O. Khattab and Matei A. Zaharia. 2020. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '20).

[24] O. 哈塔布（O. Khattab）和马泰·A·扎哈里亚（Matei A. Zaharia）. 2020年. ColBERT：通过基于BERT的上下文后期交互实现高效有效的段落搜索. 收录于《第43届国际ACM SIGIR信息检索研究与发展会议论文集》（SIGIR '20）.

[25] Diederik P. Kingma and Jimmy Ba. 2015. Adam: A Method for Stochastic Optimization. In Proceedings of the 3rd International Conference for Learning Representations (ICLR '15).

[25] 迪德里克·P·金马（Diederik P. Kingma）和吉米·巴（Jimmy Ba）. 2015年. Adam：一种随机优化方法. 收录于《第3届国际学习表征会议论文集》（ICLR '15）.

[26] Weize Kong, Swaraj Khadanga, Cheng Li, Shaleen Kumar Gupta, Mingyang Zhang, Wensong Xu, and Michael Bendersky. 2022. Multi-Aspect Dense Retrieval. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '22). Association for Computing Machinery, New York, NY, USA, 3178-3186. https://doi.org/10.1145/3534678.3539137

[26] 孔维泽（Weize Kong）、斯瓦拉吉·卡丹加（Swaraj Khadanga）、李成（Cheng Li）、沙林·库马尔·古普塔（Shaleen Kumar Gupta）、张明阳（Mingyang Zhang）、徐文松（Wensong Xu）和迈克尔·本德尔斯基（Michael Bendersky）. 2022年. 多方面密集检索. 收录于《第28届ACM SIGKDD知识发现与数据挖掘会议论文集》（KDD '22）. 美国计算机协会, 美国纽约州纽约市, 3178 - 3186. https://doi.org/10.1145/3534678.3539137

[27] John Lafferty and Chengxiang Zhai. 2001. Document Language Models, Query Models, and Risk Minimization for Information Retrieval. In Proceedings of the 24th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '01). Association for Computing Machinery, New York, NY, USA, 111-119. https://doi.org/10.1145/383952.383970

[27] 约翰·拉弗蒂（John Lafferty）和翟成祥（Chengxiang Zhai）. 2001年. 文档语言模型、查询模型和信息检索的风险最小化. 收录于《第24届年度国际ACM SIGIR信息检索研究与发展会议论文集》（SIGIR '01）. 美国计算机协会, 美国纽约州纽约市, 111 - 119. https://doi.org/10.1145/383952.383970

[28] Sheng-Chieh Lin, Jheng-Hong Yang, and Jimmy J. Lin. 2021. Distilling Dense Representations for Ranking using Tightly-Coupled Teachers. Proceedings of the 6th Workshop on Representation Learning for NLP (RepL4NLP-2021) (2021).

[28] 林圣杰（Sheng-Chieh Lin）、杨政宏（Jheng-Hong Yang）和吉米·J·林（Jimmy J. Lin）。2021年。使用紧密耦合教师模型提炼用于排序的密集表示。第6届自然语言处理表示学习研讨会（RepL4NLP - 2021）会议论文集（2021年）。

[29] Macedo Maia, Siegfried Handschuh, Andre Freitas, Brian Davis, Ross McDermott, Manel Zarrouk, and Alexandra Balahur. 2018. WWW'18 Open Challenge: Financial Opinion Mining and Question Answering. WWW '18: Companion Proceedings of the The Web Conference 2018, 1941-1942.

[29] 马塞多·马亚（Macedo Maia）、西格弗里德·汉舒（Siegfried Handschuh）、安德烈·弗雷塔斯（Andre Freitas）、布莱恩·戴维斯（Brian Davis）、罗斯·麦克德莫特（Ross McDermott）、马内尔·扎鲁克（Manel Zarrouk）和亚历山德拉·巴拉胡尔（Alexandra Balahur）。2018年。WWW'18开放挑战赛：金融观点挖掘与问答。WWW '18：2018年万维网会议配套会议论文集，1941 - 1942页。

[30] Yu A. Malkov and D. A. Yashunin. 2020. Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs. IEEE Trans. Pattern Anal. Mach. Intell. 42, 4 (apr 2020), 824-836. https://doi.org/ 10.1109/TPAMI.2018.2889473

[30] 尤·A·马尔科夫（Yu A. Malkov）和D·A·亚舒宁（D. A. Yashunin）。2020年。使用层次可导航小世界图进行高效稳健的近似最近邻搜索。《电气与电子工程师协会模式分析与机器智能汇刊》42卷，第4期（2020年4月），824 - 836页。https://doi.org/ 10.1109/TPAMI.2018.2889473

[31] Maximillian Nickel and Douwe Kiela. 2017. Poincaré embeddings for learning hierarchical representations. Advances in neural information processing systems 30 (2017).

[31] 马克西米利安·尼克（Maximillian Nickel）和杜韦·基拉（Douwe Kiela）。2017年。用于学习层次表示的庞加莱嵌入。《神经信息处理系统进展》30卷（2017年）。

[32] Rodrigo Nogueira. 2019. From doc2query to docTTTTTquery.

[32] 罗德里戈·诺盖拉（Rodrigo Nogueira）。2019年。从doc2query到docTTTTTquery。

[33] Rodrigo Nogueira and Kyunghyun Cho. 2019. Passage Re-ranking with BERT. ArXiv abs/1901.04085 (2019).

[33] 罗德里戈·诺盖拉（Rodrigo Nogueira）和赵京焕（Kyunghyun Cho）。2019年。使用BERT进行段落重排序。预印本arXiv:1901.04085（2019年）。

[34] Gustavo Penha and Claudia Hauff. 2021. On the calibration and uncertainty of neural learning to rank models for conversational search. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume. 160-170.

[34] 古斯塔沃·佩尼亚（Gustavo Penha）和克劳迪娅·豪夫（Claudia Hauff）。2021年。对话搜索中神经学习排序模型的校准与不确定性研究。《计算语言学协会欧洲分会第16届会议：主卷》会议论文集，160 - 170页。

[35] Jay M. Ponte and W. Bruce Croft. 1998. A Language Modeling Approach to Information Retrieval. In Proceedings of the 21st Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '98). Association for Computing Machinery, New York, NY, USA, 275-281. https: //doi.org/10.1145/290941.291008

[35] 杰伊·M·庞特（Jay M. Ponte）和W·布鲁斯·克罗夫特（W. Bruce Croft）。1998年。一种基于语言建模的信息检索方法。《第21届ACM信息检索研究与发展国际会议（SIGIR '98）》会议论文集。美国计算机协会，美国纽约州纽约市，275 - 281页。https: //doi.org/10.1145/290941.291008

[36] The Lemur Project. [n.d.]. Galago. https://www.lemurproject.org/galago.php

[36] 狐猴项目（The Lemur Project）。[未注明日期]。加拉戈（Galago）。https://www.lemurproject.org/galago.php

[37] Yingqi Qu, Yuchen Ding, Jing Liu, Kai Liu, Ruiyang Ren, Xin Zhao, Daxiang Dong, Hua Wu, and Haifeng Wang. 2021. RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL.

[37] 曲英琦（Yingqi Qu）、丁雨晨（Yuchen Ding）、刘静（Jing Liu）、刘凯（Kai Liu）、任瑞阳（Ruiyang Ren）、赵鑫（Xin Zhao）、董大祥（Daxiang Dong）、吴华（Hua Wu）和王海峰（Haifeng Wang）。2021年。RocketQA：用于开放域问答的密集段落检索的优化训练方法。《计算语言学协会北美分会2021年会议：人类语言技术》会议论文集，NAACL。

[38] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. Journal of Machine Learning Research 21, 140 (2020), 1-67. http://jmlr.org/papers/v21/20-074.html

[38] 科林·拉菲尔（Colin Raffel）、诺姆·沙泽尔（Noam Shazeer）、亚当·罗伯茨（Adam Roberts）、凯瑟琳·李（Katherine Lee）、沙兰·纳朗（Sharan Narang）、迈克尔·马蒂纳（Michael Matena）、周燕琪（Yanqi Zhou）、李伟（Wei Li）和彼得·J·刘（Peter J. Liu）。2020年。使用统一的文本到文本转换器探索迁移学习的极限。《机器学习研究杂志》21卷，第140期（2020年），1 - 67页。http://jmlr.org/papers/v21/20 - 074.html

[39] Stephen Robertson, S. Walker, S. Jones, M. M. Hancock-Beaulieu, and M. Gatford. 1995. Okapi at TREC-3. In Proceedings of the Third Text REtrieval Conference (TREC-3). Gaithersburg, MD: NIST, 109-126.

[39] 斯蒂芬·罗伯逊（Stephen Robertson）、S·沃克（S. Walker）、S·琼斯（S. Jones）、M·M·汉考克 - 博利厄（M. M. Hancock - Beaulieu）和M·加特福德（M. Gatford）。1995年。TREC - 3会议上的Okapi系统。《第三届文本检索会议（TREC - 3）》会议论文集。美国马里兰州盖瑟斯堡：美国国家标准与技术研究院，109 - 126页。

[40] Dwaipayan Roy, Debasis Ganguly, Mandar Mitra, and Gareth J.F. Jones. 2019. Estimating Gaussian mixture models in the local neighbourhood of embedded word vectors for query performance prediction. Information Processing & Management 56, 3 (2019), 1026-1045. https://doi.org/10.1016/j.ipm.2018.10.009

[40] 德维帕扬·罗伊（Dwaipayan Roy）、德巴西斯·冈古利（Debasis Ganguly）、曼达尔·米特拉（Mandar Mitra）和加雷斯·J·F·琼斯（Gareth J.F. Jones）。2019年。在嵌入词向量的局部邻域中估计高斯混合模型用于查询性能预测。《信息处理与管理》56卷，第3期（2019年），1026 - 1045页。https://doi.org/10.1016/j.ipm.2018.10.009

[41] G. Salton, A. Wong, and C. S. Yang. 1975. A Vector Space Model for Automatic Indexing. Commun. ACM 18, 11 (nov 1975), 613-620. https://doi.org/10.1145/ 361219.361220

[41] G·萨尔顿（G. Salton）、A·王（A. Wong）和C·S·杨（C. S. Yang）。1975年。一种用于自动索引的向量空间模型。《美国计算机协会通讯》18卷，第11期（1975年11月），613 - 620页。https://doi.org/10.1145/ 361219.361220

[42] Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. 2019. Dis-tilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. ArXiv abs/1910.01108 (2019).

[42] 维克多·桑（Victor Sanh）、利桑德尔·德比特（Lysandre Debut）、朱利安·肖蒙（Julien Chaumond）和托马斯·沃尔夫（Thomas Wolf）。2019年。DistilBERT：BERT的提炼版本，更小、更快、更便宜、更轻量。预印本arXiv:1910.01108（2019年）。

[43] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. 2022. ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. Association for Computational Linguistics, Seattle, United States, 3715-3734.

[43] 凯沙夫·桑塔纳姆（Keshav Santhanam）、奥马尔·哈塔卜（Omar Khattab）、乔恩·萨德 - 法尔孔（Jon Saad - Falcon）、克里斯托弗·波茨（Christopher Potts）和马泰·扎哈里亚（Matei Zaharia）。2022 年。ColBERTv2：通过轻量级后期交互实现高效检索。收录于《2022 年北美计算语言学协会人类语言技术会议论文集》。美国计算语言学协会，美国西雅图，3715 - 3734。

[44] Yue Shi, Xiaoxue Zhao, Jun Wang, Martha Larson, and Alan Hanjalic. 2012. Adaptive Diversification of Recommendation Results via Latent Factor Portfolio. In Proceedings of the 35th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '12). Association for Computing Machinery, New York, NY, USA, 175-184. https://doi.org/10.1145/2348283.2348310

[44] 史悦（Yue Shi）、赵小雪（Xiaoxue Zhao）、王军（Jun Wang）、玛莎·拉尔森（Martha Larson）和艾伦·汉贾利奇（Alan Hanjalic）。2012 年。通过潜在因子组合实现推荐结果的自适应多样化。收录于《第 35 届 ACM 国际信息检索研究与发展会议论文集（SIGIR '12）》。美国计算机协会，美国纽约州纽约市，175 - 184。https://doi.org/10.1145/2348283.2348310

[45] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021. BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

[45] 南丹·塔库尔（Nandan Thakur）、尼尔斯·赖默斯（Nils Reimers）、安德里亚斯·吕克莱（Andreas Rücklé）、阿比舍克·斯里瓦斯塔瓦（Abhishek Srivastava）和伊琳娜·古列维奇（Iryna Gurevych）。2021 年。BEIR：信息检索模型零样本评估的异构基准。收录于《第三十五届神经信息处理系统数据集与基准会议》。

[46] Ivan Vendrov, Ryan Kiros, Sanja Fidler, and Raquel Urtasun. 2016. Order-Embeddings of Images and Language. In Proceedings of the 2016 International Conference on Learning Representations (ICLR '16).

[46] 伊万·文德罗夫（Ivan Vendrov）、瑞安·基罗斯（Ryan Kiros）、桑贾·菲德勒（Sanja Fidler）和拉克尔·乌尔塔松（Raquel Urtasun）。2016 年。图像与语言的顺序嵌入。收录于《2016 年国际学习表征会议论文集（ICLR '16）》。

[47] Luke Vilnis, Xiang Li, Shikhar Murty, and Andrew McCallum. 2018. Probabilistic Embedding of Knowledge Graphs with Box Lattice Measures. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, ACL 2018, Melbourne, Australia, July 15-20, 2018, Volume 1: Long Papers, Iryna Gurevych and Yusuke Miyao (Eds.). Association for Computational Linguistics, 263-272. https://doi.org/10.18653/v1/P18-1025

[47] 卢克·维尔尼斯（Luke Vilnis）、李翔（Xiang Li）、希卡尔·穆尔蒂（Shikhar Murty）和安德鲁·麦卡勒姆（Andrew McCallum）。2018 年。基于盒格度量的知识图谱概率嵌入。收录于《第 56 届计算语言学协会年会论文集，ACL 2018，澳大利亚墨尔本，2018 年 7 月 15 - 20 日，第 1 卷：长论文》，伊琳娜·古列维奇（Iryna Gurevych）和宫尾悠介（Yusuke Miyao）（编）。美国计算语言学协会，263 - 272。https://doi.org/10.18653/v1/P18 - 1025

[48] Luke Vilnis and Andrew McCallum. 2015. Word Representations via Gaussian Embedding. In Proceedings of the 3rd International Conference on Learning Representations (ICLR '15), Yoshua Bengio and Yann LeCun (Eds.).

[48] 卢克·维尔尼斯（Luke Vilnis）和安德鲁·麦卡勒姆（Andrew McCallum）。2015 年。通过高斯嵌入实现词表征。收录于《第 3 届国际学习表征会议论文集（ICLR '15）》，约书亚·本吉奥（Yoshua Bengio）和扬·勒昆（Yann LeCun）（编）。

[49] Ellen M. Voorhees. 2004. Overview of the TREC 2004 Robust Retrieval Track. In The Thirteenth Text REtrieval Conference, TREC 2004. 70-80.

[49] 艾伦·M·沃里斯（Ellen M. Voorhees）。2004 年。TREC 2004 鲁棒检索赛道概述。收录于《第十三届文本检索会议，TREC 2004》。70 - 80。

[50] Ellen M. Voorhees, Tasmeer Alam, Steven Bedrick, Dina Demner-Fushman, William R. Hersh, Kyle Lo, Kirk Roberts, Ian Soboroff, and Lucy Lu Wang. 2020. TREC-COVID: Constructing a Pandemic Information Retrieval Test Collection. CoRR (2020).

[50] 艾伦·M·沃里斯（Ellen M. Voorhees）、塔斯米尔·阿拉姆（Tasmeer Alam）、史蒂文·贝德里克（Steven Bedrick）、迪娜·德姆纳 - 富什曼（Dina Demner - Fushman）、威廉·R·赫什（William R. Hersh）、凯尔·洛（Kyle Lo）、柯克·罗伯茨（Kirk Roberts）、伊恩·索博罗夫（Ian Soboroff）和王璐（Lucy Lu Wang）。2020 年。TREC - COVID：构建大流行信息检索测试集。计算机研究报告（2020 年）。

[51] David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan, and Hannaneh Hajishirzi. 2020. Fact or Fiction: Verifying Scientific Claims. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). Association for Computational Linguistics, Online, 7534-7550. https://doi.org/10.18653/v1/2020.emnlp-main.609

[51] 大卫·瓦登（David Wadden）、林山川（Shanchuan Lin）、凯尔·洛（Kyle Lo）、王璐（Lucy Lu Wang）、玛德琳·范·祖伊伦（Madeleine van Zuylen）、阿尔曼·科汉（Arman Cohan）和哈南内·哈吉希尔齐（Hannaneh Hajishirzi）。2020 年。事实还是虚构：验证科学主张。收录于《2020 年自然语言处理经验方法会议论文集（EMNLP）》。美国计算语言学协会，线上会议，7534 - 7550。https://doi.org/10.18653/v1/2020.emnlp - main.609

[52] Jun Wang and Jianhan Zhu. 2009. Portfolio Theory of Information Retrieval. In Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '09). Association for Computing Machinery, New York, NY, USA, 115-122. https://doi.org/10.1145/1571941.1571963

[52] 王军（Jun Wang）和朱建汉（Jianhan Zhu）。2009 年。信息检索的投资组合理论。收录于《第 32 届 ACM 国际信息检索研究与发展会议论文集（SIGIR '09）》。美国计算机协会，美国纽约州纽约市，115 - 122。https://doi.org/10.1145/1571941.1571963

[53] Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed, and Arnold Overwijk. 2021. Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval. In Proceedings of the 9th International Conference on Learning Representations (ICLR '21).

[53] 熊磊（Lee Xiong）、熊晨艳（Chenyan Xiong）、李烨（Ye Li）、邓国峰（Kwok - Fung Tang）、刘佳琳（Jialin Liu）、保罗·贝内特（Paul Bennett）、朱奈德·艾哈迈德（Junaid Ahmed）和阿诺德·奥弗维克（Arnold Overwijk）。2021 年。用于密集文本检索的近似最近邻负对比学习。收录于《第 9 届国际学习表征会议论文集（ICLR '21）》。

[54] Hamed Zamani, W. Bruce Croft, and J. Shane Culpepper. 2018. Neural Query Performance Prediction Using Weak Supervision from Multiple Signals. In The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval (SIGIR '18). Association for Computing Machinery, New York, NY, USA, 105-114. https://doi.org/10.1145/3209978.3210041

[54] 哈米德·扎马尼（Hamed Zamani）、W. 布鲁斯·克罗夫特（W. Bruce Croft）和 J. 谢恩·卡尔佩珀（J. Shane Culpepper）。2018 年。利用多信号弱监督进行神经查询性能预测。收录于《第 41 届 ACM 国际信息检索研究与发展会议论文集（SIGIR '18）》。美国计算机协会，美国纽约州纽约市，105 - 114。https://doi.org/10.1145/3209978.3210041

[55] Hamed Zamani, Mostafa Dehghani, W. Bruce Croft, Erik Learned-Miller, and Jaap Kamps. 2018. From Neural Re-Ranking to Neural Ranking: Learning a Sparse Representation for Inverted Indexing. In Proceedings of the 27th ACM International Conference on Information and Knowledge Management (CIKM '18). Association for Computing Machinery, New York, NY, USA, 497-506. https: //doi.org/10.1145/3269206.3271800

[55] 哈米德·扎马尼（Hamed Zamani）、莫斯塔法·德赫加尼（Mostafa Dehghani）、W. 布鲁斯·克罗夫特（W. Bruce Croft）、埃里克·利尔内德 - 米勒（Erik Learned - Miller）和亚普·坎普斯（Jaap Kamps）。2018 年。从神经重排序到神经排序：学习用于倒排索引的稀疏表示。见第 27 届 ACM 国际信息与知识管理会议论文集（CIKM '18）。美国计算机协会，美国纽约州纽约市，497 - 506。https: //doi.org/10.1145/3269206.3271800

[56] Hamed Zamani, Fernando Diaz, Mostafa Dehghani, Donald Metzler, and Michael Bendersky. 2022. Retrieval-Enhanced Machine Learning. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '22). Association for Computing Machinery, New York, NY, USA, 2875-2886. https://doi.org/10.1145/3477495.3531722

[56] 哈米德·扎马尼（Hamed Zamani）、费尔南多·迪亚兹（Fernando Diaz）、莫斯塔法·德赫加尼（Mostafa Dehghani）、唐纳德·梅茨勒（Donald Metzler）和迈克尔·本德尔斯基（Michael Bendersky）。2022 年。检索增强机器学习。见第 45 届 ACM 国际信息检索研究与发展会议论文集（SIGIR '22）。美国计算机协会，美国纽约州纽约市，2875 - 2886。https://doi.org/10.1145/3477495.3531722

[57] Hansi Zeng, Hamed Zamani, and Vishwa Vinay. 2022. Curriculum Learning for Dense Retrieval Distillation. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '22). Association for Computing Machinery, New York, NY, USA, 1979-1983 https://doi.org/10.1145/3477495.3531791

[57] 曾汉斯（Hansi Zeng）、哈米德·扎马尼（Hamed Zamani）和维什瓦·维奈（Vishwa Vinay）。2022 年。密集检索蒸馏的课程学习。见第 45 届 ACM 国际信息检索研究与发展会议论文集（SIGIR '22）。美国计算机协会，美国纽约州纽约市，1979 - 1983 https://doi.org/10.1145/3477495.3531791

[58] Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Jiafeng Guo, Min Zhang, and Shaoping Ma. 2021. Optimizing Dense Retrieval Model Training with Hard Negatives. Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (2021).

[58] 詹景涛、毛佳欣、刘奕群、郭佳峰、张敏和马少平。2021 年。利用难负样本优化密集检索模型训练。第 44 届 ACM 国际信息检索研究与发展会议论文集（2021）。

[59] Dell Zhang and Jinsong Lu. 2009. Batch-Mode Computational Advertising Based on Modern Portfolio Theory. In Advances in Information Retrieval Theory, Leif Azzopardi, Gabriella Kazai, Stephen Robertson, Stefan Rüger, Milad Shokouhi, Dawei Song, and Emine Yilmaz (Eds.). Springer Berlin Heidelberg, Berlin, Heidelberg, 380-383.

[59] 戴尔·张（Dell Zhang）和卢劲松（Jinsong Lu）。2009 年。基于现代投资组合理论的批量模式计算广告。见《信息检索理论进展》，莱夫·阿佐帕迪（Leif Azzopardi）、加布里埃拉·卡扎伊（Gabriella Kazai）、斯蒂芬·罗伯逊（Stephen Robertson）、斯特凡·吕格（Stefan Rüger）、米拉德·肖库希（Milad Shokouhi）、宋大为（Dawei Song）和埃米内·伊尔马兹（Emine Yilmaz）（编）。施普林格·柏林·海德堡出版社，德国柏林、海德堡，380 - 383。

[60] Ying Zhao, Falk Scholer, and Yohannes Tsegay. 2008. Effective Pre-Retrieval Query Performance Prediction Using Similarity and Variability Evidence. In Proceedings of the IR Research, 30th European Conference on Advances in Information Retrieval

[60] 赵莹、福尔克·肖勒（Falk Scholer）和约哈内斯·特塞盖（Yohannes Tsegay）。2008 年。利用相似度和变异性证据进行有效的预检索查询性能预测。见信息检索研究会议论文集，第 30 届欧洲信息检索进展会议

(ECIR'08). Springer-Verlag, Berlin, Heidelberg, 52-64.

(ECIR'08)。施普林格出版社，德国柏林、海德堡，52 - 64。

[61] Giulio Zhou and Jacob Devlin. 2021. Multi-vector attention models for deep re-ranking. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. 5452-5456.

[61] 朱利奥·周（Giulio Zhou）和雅各布·德夫林（Jacob Devlin）。2021 年。用于深度重排序的多向量注意力模型。见 2021 年自然语言处理经验方法会议论文集。5452 - 5456。

[62] Jianhan Zhu, Jun Wang, Michael Taylor, and Ingemar J Cox. 2009. Risk-aware information retrieval. In Advances in Information Retrieval: 31th European Conference on IR Research, ECIR 2009, Toulouse, France, April 6-9, 2009. Proceedings 31. Springer, 17-28.

[62] 朱建汉、王军、迈克尔·泰勒（Michael Taylor）和英格玛·J·考克斯（Ingemar J Cox）。2009 年。风险感知信息检索。见《信息检索进展：第 31 届欧洲信息检索研究会议，ECIR 2009，法国图卢兹，2009 年 4 月 6 - 9 日》。会议论文集 31。施普林格出版社，17 - 28。

[63] Guido Zuccon, Leif Azzopardi, and C.J. "Keith" van Rijsbergen. 2010. Has Portfolio Theory Got Any Principles?. In Proceedings of the 33rd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '10). Association for Computing Machinery, New York, NY, USA, 755-756. https: //doi.org/10.1145/1835449.1835600

[63] 圭多·祖科恩（Guido Zuccon）、莱夫·阿佐帕迪（Leif Azzopardi）和 C.J. “基思”·范·里斯伯根（C.J. "Keith" van Rijsbergen）。2010 年。投资组合理论有原则吗？见第 33 届 ACM 国际信息检索研究与发展会议论文集（SIGIR '10）。美国计算机协会，美国纽约州纽约市，755 - 756。https: //doi.org/10.1145/1835449.1835600