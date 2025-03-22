# In-Batch Negatives for Knowledge Distillation with Tightly-Coupled Teachers for Dense Retrieval

# 用于密集检索的紧密耦合教师知识蒸馏中的批次内负样本

Sheng-Chieh Lin*, Jheng-Hong Yang* and Jimmy Lin

林圣杰*、杨政宏*和林吉米

David R. Cheriton School of Computer Science

大卫·R·切里顿计算机科学学院

University of Waterloo

滑铁卢大学

## Abstract

## 摘要

We present an efficient training approach to text retrieval with dense representations that applies knowledge distillation using the ColBERT late-interaction ranking model. Specifically, we propose to transfer the knowledge from a bi-encoder teacher to a student by distilling knowledge from ColBERT's expressive MaxSim operator into a simple dot product. The advantage of the bi-encoder teacher-student setup is that we can efficiently add in-batch negatives during knowledge distillation, enabling richer interactions between teacher and student models. In addition, using ColBERT as the teacher reduces training cost compared to a full cross-encoder. Experiments on the MS MARCO passage and document ranking tasks and data from the TREC 2019 Deep Learning Track demonstrate that our approach helps models learn robust representations for dense retrieval effectively and efficiently.

我们提出了一种使用密集表示进行文本检索的高效训练方法，该方法利用ColBERT后期交互排序模型进行知识蒸馏。具体而言，我们建议通过将ColBERT富有表现力的MaxSim算子的知识提炼到简单的点积中，将双编码器教师模型的知识转移到学生模型。双编码器师生设置的优势在于，我们可以在知识蒸馏过程中高效地添加批次内负样本，从而使教师模型和学生模型之间实现更丰富的交互。此外，与全交叉编码器相比，使用ColBERT作为教师模型可降低训练成本。在MS MARCO段落和文档排序任务以及TREC 2019深度学习赛道的数据上进行的实验表明，我们的方法有助于模型有效且高效地学习用于密集检索的鲁棒表示。

## 1 Introduction

## 1 引言

For well over half a century, solutions to the ad hoc retrieval problem-where the system's task is return a list of top $k$ texts from an arbitrarily large corpus $\mathcal{D}$ that maximizes some metric of quality such as average precision or NDCG-has been dominated by sparse vector representations, for example, bag-of-words BM25. Even in modern multi-stage ranking architectures, which take advantage of large pretrained transformers such as BERT (Devlin et al., 2019), the models are deployed as rerankers over initial candidates retrieved based on sparse vector representations; this is sometimes called "first-stage retrieval". One well-known example of this design is the BERT-based reranker of Nogueira and Cho (2019); see Lin et al. (2020) for a recent survey.

半个多世纪以来，即席检索问题（系统的任务是从任意大的语料库 $\mathcal{D}$ 中返回前 $k$ 篇文本的列表，以最大化平均精度或归一化折损累积增益等质量指标）的解决方案一直以稀疏向量表示为主，例如词袋模型BM25。即使在现代多阶段排序架构中，利用了如BERT（Devlin等人，2019）等大型预训练变压器模型，这些模型也被用作基于稀疏向量表示检索到的初始候选文档的重排器；这有时被称为“第一阶段检索”。这种设计的一个著名例子是Nogueira和Cho（2019）基于BERT的重排器；有关最新综述，请参阅Lin等人（2020）。

The standard reranker architecture, while effective, exhibits high query latency, on the order of seconds per query (Hofstätter and Hanbury, 2019; Khattab and Zaharia, 2020) because expensive neural inference must be applied at query time on query-passage pairs. This design is known as a cross-encoder (Humeau et al., 2020), which exploits query-passage attention interactions across all transformer layers. As an alternative, a bi-encoder design provides an approach to ranking with dense representations that is far more efficient than cross-encoders (Lee et al., 2019; Reimers and Gurevych, 2019; Khattab and Zaharia, 2020; Karpukhin et al., 2020; Luan et al., 2021; Xiong et al., 2021; Qu et al., 2020; Hofstätter et al., 2021). Prior to retrieval, the vector representations can be precomputed for each of the texts in a corpus. When retrieving texts in response to a given query, computationally expensive transformer inference is replaced by much faster approximate nearest neighbor (ANN) search (Liu et al., 2004; Malkov and Yashunin, 2020).

标准的重排器架构虽然有效，但查询延迟较高，每个查询约需数秒（Hofstätter和Hanbury，2019；Khattab和Zaharia，2020），因为在查询时必须对查询 - 段落对进行昂贵的神经推理。这种设计被称为交叉编码器（Humeau等人，2020），它利用了所有变压器层上的查询 - 段落注意力交互。作为一种替代方案，双编码器设计提供了一种使用密集表示进行排序的方法，其效率远高于交叉编码器（Lee等人，2019；Reimers和Gurevych，2019；Khattab和Zaharia，2020；Karpukhin等人，2020；Luan等人，2021；Xiong等人，2021；Qu等人，2020；Hofstätter等人，2021）。在检索之前，可以为语料库中的每篇文本预先计算向量表示。当根据给定查询检索文本时，计算成本高昂的变压器推理被速度快得多的近似最近邻（ANN）搜索所取代（Liu等人，2004；Malkov和Yashunin，2020）。

Recently, researchers have proposed bi-encoders that produce multiple vectors to represent a query (or a passage) (Humeau et al., 2020; Luan et al., 2021; Khattab and Zaharia, 2020), which have proven to be effective both theoretically and empirically. However, the main disadvantage of these designs is their high storage requirements. For example, ColBERT (Khattab and Zaharia, 2020) requires storing all the WordPiece token vectors of each text (passage) in the corpus. On the MS MARCO passage corpus comprising ${8.8}\mathrm{M}$ passages,for example, this requires 154 GiB.

最近，研究人员提出了能够生成多个向量来表示查询（或段落）的双编码器（Humeau等人，2020；Luan等人，2021；Khattab和Zaharia，2020），这些方法在理论和实践上都被证明是有效的。然而，这些设计的主要缺点是存储需求高。例如，ColBERT（Khattab和Zaharia，2020）需要存储语料库中每篇文本（段落）的所有WordPiece词元向量。例如，在包含 ${8.8}\mathrm{M}$ 个段落的MS MARCO段落语料库中，这需要154 GiB的存储空间。

Of course, a common alternative is to produce single vectors for queries and passages (Reimers and Gurevych, 2019). Although this design is less storage-demanding, it sacrifices ranking effectiveness since its structure breaks rich interactions between queries and passages compared to multi-vector bi-encoders or cross-encoders. Hence, improving the effectiveness of single-vector bi-encoders represents an important problem.

当然，一种常见的替代方法是为查询和段落生成单向量（Reimers和Gurevych，2019）。虽然这种设计对存储的要求较低，但它牺牲了排序效果，因为与多向量双编码器或交叉编码器相比，其结构破坏了查询和段落之间丰富的交互。因此，提高单向量双编码器的有效性是一个重要问题。

---

<!-- Footnote -->

*Contributed equally.

* 贡献相同。

<!-- Footnote -->

---

One approach to improving the effectiveness of single-vector bi-encoders is hard negative mining, by training with carefully selected negative examples that emphasize discrimination between relevant and non-relevant texts. There are several approaches to accomplish this. Karpukhin et al. (2020) and Qu et al. (2020) leverage large in-batch negatives to enrich training signals. Guu et al. (2020) and Xiong et al. (2021) propose to mine hard negatives using the trained bi-encoder itself. By searching for global negative samples from an asynchronously updated ANN index, the bi-encoder can learn information not present in the training data produced by sparse representations (Xiong et al., 2021). However, both large in-batch negative sampling and asynchronous ANN index updates are computationally demanding. The later is especially impractical for large corpora since it requires periodic inference over all texts in the corpus to ensure that the best negative examples are retrieved.

提高单向量双编码器有效性的一种方法是难负样本挖掘，即通过使用精心挑选的负样本来进行训练，这些负样本强调相关文本和不相关文本之间的区分度。有几种方法可以实现这一点。卡尔普欣等人（2020年）和曲等人（2020年）利用大量的批内负样本（in-batch negatives）来丰富训练信号。古等人（2020年）和熊等人（2021年）提出使用训练好的双编码器本身来挖掘难负样本。通过从异步更新的近似最近邻（ANN）索引中搜索全局负样本，双编码器可以学习到稀疏表示所生成的训练数据中不存在的信息（熊等人，2021年）。然而，大量的批内负采样和异步ANN索引更新在计算上都要求很高。后者对于大型语料库尤其不切实际，因为它需要对语料库中的所有文本进行定期推理，以确保检索到最佳的负样本。

There is also work that explores knowledge distillation (KD) (Hinton et al., 2015) to enhance retrieval effectiveness and efficiency. Most related to our study is Hofstätter et al. (2020), who demonstrate that KD using a cross-encoder teacher significantly improves the effectiveness of bi-encoders for dense retrieval. Similarly, Barkan et al. (2020) investigate the effectiveness of distilling a trained cross-encoder into a bi-encoder for sentence similarity tasks. Gao et al. (2020a) explore KD combinations of different objectives such as language modeling and ranking. However, the above papers use computationally expensive cross-encoder teacher models; thus, combining them for KD with more advanced negative sampling techniques can be impractical.

也有研究探索使用知识蒸馏（KD）（辛顿等人，2015年）来提高检索的有效性和效率。与我们的研究最相关的是霍夫施泰特等人（2020年）的工作，他们证明了使用交叉编码器作为教师模型进行知识蒸馏可以显著提高双编码器在密集检索中的有效性。类似地，巴尔坎等人（2020年）研究了将训练好的交叉编码器知识蒸馏到双编码器中用于句子相似度任务的有效性。高等人（2020a）探索了不同目标（如语言建模和排序）的知识蒸馏组合。然而，上述论文使用的交叉编码器教师模型在计算上成本高昂；因此，将它们与更先进的负采样技术结合用于知识蒸馏可能不切实际。

In light of existing work on hard negative mining and knowledge distillation, we propose to improve the effectiveness of single-vector bi-encoders with a more efficient KD approach: in-batch KD using a bi-encoder teacher. The advantage of our design is that, during distillation, it enables the efficient exploitation of all possible query-passage pairs within a minibatch, which we call tight coupling (illustrated in Figure 1). This is a key difference between our KD approach and previous methods for dense retrieval, where only the scores of given query-passage triplets (not all combinations) are computed due to the computational costs of cross-encoders (Hofstätter et al., 2020; Gao et al., 2020a; Barkan et al., 2020).

鉴于现有的难负样本挖掘和知识蒸馏工作，我们提出了一种更高效的知识蒸馏方法来提高单向量双编码器的有效性：使用双编码器作为教师模型的批内知识蒸馏（in-batch KD）。我们设计的优势在于，在蒸馏过程中，它能够高效地利用小批量内所有可能的查询 - 段落对，我们称之为紧密耦合（如图1所示）。这是我们的知识蒸馏方法与之前用于密集检索的方法的关键区别，在之前的方法中，由于交叉编码器的计算成本，只计算给定查询 - 段落三元组（而非所有组合）的得分（霍夫施泰特等人，2020年；高等人，2020a；巴尔坎等人，2020年）。

<!-- Media -->

<!-- figureText: Teacher Target: Pairwise KD ${\mathbf{q}}_{2}$ Target: In-batch KD Encoder Student Teacher -->

<img src="https://cdn.noedgeai.com/0195aec9-3fad-7d98-91c4-c0e2a3659794_1.jpg?x=849&y=170&w=610&h=288&r=0"/>

Figure 1: Illustration of the differences between pairwise knowledge distillation and our proposed in-batch knowledge distillation.

图1：成对知识蒸馏与我们提出的批内知识蒸馏之间差异的示意图。

<!-- Media -->

The contribution of this work is a simple technique for efficiently adding in-batch negative samples during knowledge distillation when training a single-vector bi-encoder. For the remainder of this paper, we refer to this technique as "in-batch KD" for convenience. We empirically show that our model, even trained with BM25 negatives, can be more effective than cross-encoder teachers. With hard negatives, our method approaches the state of the art in dense retrieval. Our in-batch KD technique is able to incorporate hard negatives in a computationally efficient manner, without requiring large amounts of GPU memory for large batch sizes or expensive periodic index refreshes.

这项工作的贡献在于提出了一种简单的技术，用于在训练单向量双编码器时在知识蒸馏过程中高效地添加批内负样本。为了方便起见，在本文的其余部分，我们将这种技术称为“批内知识蒸馏（in-batch KD）”。我们通过实验表明，即使使用BM25负样本进行训练，我们的模型也可能比交叉编码器教师模型更有效。使用难负样本时，我们的方法接近密集检索的当前最优水平。我们的批内知识蒸馏技术能够以计算高效的方式纳入难负样本，无需为大批次大小使用大量的GPU内存，也无需进行昂贵的定期索引刷新。

## 2 Background

## 2 背景

We focus on improving the training efficiency and retrieval effectiveness of dense retrieval and begin by formalizing it as a dense representation learning problem. To be more specific, we propose to use knowledge distillation to enrich training signals and stabilize the representation learning procedure of bi-encoder models in the context of the well-known Noise-Contrastive Estimation (NCE) framework.

我们专注于提高密集检索的训练效率和检索有效性，并首先将其形式化为一个密集表示学习问题。更具体地说，我们提议在著名的噪声对比估计（NCE）框架下，使用知识蒸馏来丰富训练信号并稳定双编码器模型的表示学习过程。

### 2.1 Dense Retrieval with Bi-encoders

### 2.1 使用双编码器进行密集检索

The bi-encoder design has been widely adopted for dense retrieval (Lee et al., 2019; Chang et al., 2020; Guu et al., 2020; Karpukhin et al., 2020; Luan et al., 2021; Qu et al., 2020; Xiong et al., 2021), where queries and passages are encoded in a low-dimensional space. It aims to learn low-dimensional representations that pull queries and relevant passages together and push queries and non-relevant passages apart.

双编码器设计已被广泛应用于密集检索（李等人，2019年；张等人，2020年；古等人，2020年；卡尔普欣等人，2020年；栾等人，2021年；曲等人，2020年；熊等人，2021年），在这种设计中，查询和段落被编码到一个低维空间中。其目标是学习低维表示，使查询和相关段落相互靠近，而使查询和不相关段落相互远离。

Following the work of Mnih and Kavukcuoglu (2013), we formulate a common objective for dense representation learning for passage retrieval. Given a query $q$ and a parameterized scoring function ${\phi }_{\theta }$ that computes the relevance between a query and a candidate passage $p$ ,we define a probability distribution over documents in a corpus $\mathcal{D}$ with respect to relevance, as follows:

遵循姆尼和卡武库奥卢（2013年）的工作，我们为段落检索的密集表示学习制定了一个通用目标。给定一个查询 $q$ 和一个参数化的评分函数 ${\phi }_{\theta }$ ，该函数计算查询与候选段落 $p$ 之间的相关性，我们定义了语料库 $\mathcal{D}$ 中文档相对于相关性的概率分布，如下所示：

$$
{P}_{\theta }^{q}\left( {p,\mathcal{D}}\right)  = \frac{\exp \left( {{\phi }_{\theta }\left( {q,p}\right) }\right) }{\mathop{\sum }\limits_{{{p}^{\prime } \in  \mathcal{D}}}\exp \left( {{\phi }_{\theta }\left( {q,{p}^{\prime }}\right) }\right) }
$$

$$
 = \frac{\exp \left( {{\mathbf{h}}_{q} \cdot  {\mathbf{h}}_{p}}\right) }{\mathop{\sum }\limits_{{{p}^{\prime } \in  \mathcal{D}}}\exp \left( {{\mathbf{h}}_{q} \cdot  {\mathbf{h}}_{{p}^{\prime }}}\right) }, \tag{1}
$$

where ${\mathbf{h}}_{q}\left( {\mathbf{h}}_{p}\right)  \in  {\mathbb{R}}^{d}$ denotes the query (passage) representation produced by the bi-encoder. A typical bi-encoder uses a simple scoring function for ${\phi }_{\theta }$ ,for example,the inner product of two vectors, as shown above.

其中 ${\mathbf{h}}_{q}\left( {\mathbf{h}}_{p}\right)  \in  {\mathbb{R}}^{d}$ 表示由双编码器生成的查询（段落）表示。典型的双编码器对 ${\phi }_{\theta }$ 使用简单的评分函数，例如，如上文所示的两个向量的内积。

The main challenge of evaluating and computing gradients of Eq. (1) is the prohibitively expensive computation cost given the number of passages in the corpus $\mathcal{D}$ ,typically millions (or even more). This is already setting aside the cost of using pre-trained transformers such as BERT as the encoder to compute ${\mathbf{h}}_{q}$ and ${\mathbf{h}}_{p}$ .

评估和计算公式 (1) 的梯度的主要挑战在于，考虑到语料库 $\mathcal{D}$ 中的段落数量（通常有数百万甚至更多），计算成本高得令人望而却步。这还未考虑使用如 BERT 等预训练的变换器作为编码器来计算 ${\mathbf{h}}_{q}$ 和 ${\mathbf{h}}_{p}$ 的成本。

Thus, previous work approximates Eq. (1) by NCE,which samples $p \in  {\mathcal{D}}^{ + }$ from training data and ${p}^{\prime } \in  {\mathcal{D}}^{\prime } = \left\{  {{\mathcal{D}}^{ + } \cup  {\mathcal{D}}^{ - }}\right\}$ ,where ${\mathcal{D}}^{ - }$ is from a noisy distribution such as candidates retrieved by BM25 (Nogueira and Cho, 2019), filtered by fine-tuned transformers (Qu et al., 2020), or retrieved by an asynchronously updated bi-encoder model itself (Xiong et al., 2021). Another simple yet effective approach is in-batch negative sampling, as used by Karpukhin et al. (2020), which takes $p$ and ${p}^{\prime }$ of other queries within a minibatch as negative examples in NCE.

因此，先前的工作通过噪声对比估计（NCE）来近似公式 (1)，该方法从训练数据中采样 $p \in  {\mathcal{D}}^{ + }$ 和 ${p}^{\prime } \in  {\mathcal{D}}^{\prime } = \left\{  {{\mathcal{D}}^{ + } \cup  {\mathcal{D}}^{ - }}\right\}$，其中 ${\mathcal{D}}^{ - }$ 来自噪声分布，例如通过 BM25 检索到的候选样本（Nogueira 和 Cho，2019），经过微调的变换器过滤后的样本（Qu 等人，2020），或者由异步更新的双编码器模型本身检索到的样本（Xiong 等人，2021）。另一种简单而有效的方法是批量内负采样，如 Karpukhin 等人（2020）所使用的，该方法将小批量内其他查询的 $p$ 和 ${p}^{\prime }$ 作为 NCE 中的负样本。

### 2.2 Knowledge Distillation

### 2.2 知识蒸馏

Other than designing sophisticated sampling methods for ${p}^{\prime }$ ,training bi-encoder models using knowledge distillation (KD) with effective teacher models is another promising approach (Hofstätter et al., 2020). In this case, we aim to make the bi-encoder model mimic the teacher model's probability distribution as follows:

除了为 ${p}^{\prime }$ 设计复杂的采样方法外，使用有效的教师模型进行知识蒸馏（KD）来训练双编码器模型是另一种有前景的方法（Hofstätter 等人，2020）。在这种情况下，我们的目标是让双编码器模型模仿教师模型的概率分布，如下所示：

$$
{P}_{\theta ;\text{ student }}^{q}\left( {p,{\mathcal{D}}^{\prime }}\right)  = \frac{\exp \left( {{\mathbf{h}}_{q} \cdot  {\mathbf{h}}_{p}}\right) }{\mathop{\sum }\limits_{{{p}^{\prime } \in  {\mathcal{D}}^{\prime }}}\exp \left( {{\mathbf{h}}_{q} \cdot  {\mathbf{h}}_{{p}^{\prime }}}\right) }
$$

$$
 \approx  \frac{\exp \left( {{\phi }_{\widehat{\theta }}\left( {q,p}\right) /\tau }\right) }{\mathop{\sum }\limits_{{{p}^{\prime } \in  {\mathcal{D}}^{\prime }}}\exp \left( {{\phi }_{\widehat{\theta }}\left( {q,{p}^{\prime }}\right) /\tau }\right) }
$$

$$
 = {P}_{\widehat{\theta };\text{ teacher }}^{q}\left( {p,{\mathcal{D}}^{\prime }}\right) , \tag{2}
$$

where ${\phi }_{\widehat{\theta }}$ denotes the relevance score estimated by a pretrained model parameterized by $\widehat{\theta }$ and $\tau$ ,the temperature hyperparameter used in the KD framework. To improve retrieval effectiveness, one can leverage pre-computed scores from pretrained models such as cross-encoders, e.g., BERT, bi-encoders, e.g., ColBERT, or ensembled scores from multiple models ${\phi }_{\widehat{\theta }} = \mathop{\sum }\limits_{j}{\phi }_{\widehat{\theta };j}$ .

其中 ${\phi }_{\widehat{\theta }}$ 表示由以 $\widehat{\theta }$ 和 $\tau$ 为参数的预训练模型估计的相关性得分，$\tau$ 是 KD 框架中使用的温度超参数。为了提高检索效果，可以利用来自预训练模型（如交叉编码器，例如 BERT；双编码器，例如 ColBERT）的预计算得分，或者来自多个模型的集成得分 ${\phi }_{\widehat{\theta }} = \mathop{\sum }\limits_{j}{\phi }_{\widehat{\theta };j}$。

## 3 Our Approach

## 3 我们的方法

### 3.1 In-batch Knowledge Distillation

### 3.1 批量内知识蒸馏

Using KD in Eq. (2) provides soft labels for bi-encoder training, and can be integrated with the previously mentioned NCE framework. In this work, we propose to enhance teacher-student interactions by adding in-batch negatives to our knowledge distillation. Specifically,we estimate ${\phi }_{\theta }$ on in-batch examples from a minibatch $\mathcal{B}$ guided by an auxiliary teacher model ${\phi }_{\widehat{\theta }}$ through the minimization of Kullback-Leibler (KL) divergence of the two distributions:

在公式 (2) 中使用知识蒸馏为双编码器训练提供软标签，并且可以与前面提到的 NCE 框架相结合。在这项工作中，我们提出通过在知识蒸馏中加入批量内负样本来增强师生交互。具体来说，我们通过最小化两个分布的 Kullback-Leibler（KL）散度，在辅助教师模型 ${\phi }_{\widehat{\theta }}$ 的指导下，对小批量 $\mathcal{B}$ 中的批量内示例估计 ${\phi }_{\theta }$：

$$
\underset{\theta }{\arg \min }\mathop{\sum }\limits_{{q \in  {\mathcal{Q}}_{\mathcal{B}}}}\mathop{\sum }\limits_{{p \in  {\mathcal{D}}_{\mathcal{B}}^{\prime }}}{\mathcal{L}}_{{\phi }_{\theta },{\phi }_{\widehat{\theta }}} \tag{3}
$$

where ${\mathcal{L}}_{{\phi }_{\theta },{\phi }_{\widehat{\theta }}}$ is:

其中 ${\mathcal{L}}_{{\phi }_{\theta },{\phi }_{\widehat{\theta }}}$ 为：

$$
{P}_{\widehat{\theta };\text{ teacher }}^{q}\left( {p,{\mathcal{D}}_{\mathcal{B}}^{\prime }}\right) \log \frac{{P}_{\widehat{\theta };\text{ teacher }}^{q}\left( {p,{\mathcal{D}}_{\mathcal{B}}^{\prime }}\right) }{{P}_{\theta ;\text{ student }}^{q}\left( {p,{\mathcal{D}}_{\mathcal{B}}^{\prime }}\right) }. \tag{4}
$$

Note that here we consider all pairwise relationship between queries and passages within a minibatch that contains a query set ${\mathcal{Q}}_{\mathcal{B}}$ and a passage set ${\mathcal{D}}_{\mathcal{B}}^{\prime }$ .

请注意，这里我们考虑了包含查询集 ${\mathcal{Q}}_{\mathcal{B}}$ 和段落集 ${\mathcal{D}}_{\mathcal{B}}^{\prime }$ 的小批量内查询和段落之间的所有成对关系。

### 3.2 Teacher Model Choice

### 3.2 教师模型选择

A cross-encoder has been shown to be an effective teacher (Hofstätter et al., 2020; Gao et al., 2020a) since it allows rich interactions between the intermediate transformer representations of a query $q$ and a passage $p$ . For example,a "vanilla" cross-encoder design using BERT can be denoted as:

交叉编码器已被证明是一种有效的教师模型（Hofstätter 等人，2020；Gao 等人，2020a），因为它允许查询 $q$ 和段落 $p$ 的中间变换器表示之间进行丰富的交互。例如，使用 BERT 的“普通”交叉编码器设计可以表示为：

$$
{\phi }_{\widehat{\theta };\text{ Cat }} \triangleq  {Wf}\left( {\mathbf{h}}_{q \oplus  p}\right) , \tag{5}
$$

where the ranking score is first computed by the hidden representation of the concatenation $q \oplus  p$ from BERT (along with the standard special tokens) and then mapped to a scalar by a pooling operation $f$ and a mapping matrix $W$ .

其中排名得分首先由 BERT 对拼接结果 $q \oplus  p$ 的隐藏表示（连同标准特殊标记）计算得出，然后通过池化操作 $f$ 和映射矩阵 $W$ 映射为一个标量。

Although effective, due to BERT's quadratic complexity with respect to input sequence length, this design makes exhaustive combinations between a query and possible candidates impractical, since this requires evaluating cross-encoders ${\left| \mathcal{B}\right| }^{2}$ times to compute Eq. (3) using Eq. (5). Thus, an alternative is to conduct pairwise KD by computing the KL divergence of only two probabilities of a positive pair(q,p)and a negative pair $\left( {q,{p}^{\prime }}\right)$ for each query $q$ . However,this might not yield a good approximation of Eq. (2).

虽然这种设计有效，但由于BERT相对于输入序列长度具有二次复杂度，这使得在查询和可能的候选对象之间进行穷举组合变得不切实际，因为这需要使用公式(5)对交叉编码器进行${\left| \mathcal{B}\right| }^{2}$次评估来计算公式(3)。因此，另一种方法是通过仅计算每个查询$q$的正样本对(q,p)和负样本对$\left( {q,{p}^{\prime }}\right)$的两个概率的KL散度来进行成对知识蒸馏(KD)。然而，这可能无法很好地近似公式(2)。

A bi-encoder can also be leveraged as a teacher model, which has the advantage that it is more feasible to perform exhaustive comparisons between queries and passages since they are passed through the encoder independently. Among bi-encoder designs, ColBERT is a representative model that uses late interactions of multiple vectors $\left( {\left\{  {{\mathbf{h}}_{q}^{1},\ldots ,{\mathbf{h}}_{q}^{i}}\right\}  ,\left\{  {{\mathbf{h}}_{p}^{1},\ldots ,{\mathbf{h}}_{p}^{j}}\right\}  }\right)$ to improve the robustness of dense retrieval, as compared to inner products of pairs of single vectors $\left( {{\mathbf{h}}_{q},{\mathbf{h}}_{p}}\right)$ . Specifically, Khattab and Zaharia (2020) propose the following fine-grained scoring function:

双编码器也可以用作教师模型，其优点是由于查询和段落是独立通过编码器的，因此在查询和段落之间进行穷举比较更为可行。在双编码器设计中，ColBERT是一个具有代表性的模型，与单向量对$\left( {{\mathbf{h}}_{q},{\mathbf{h}}_{p}}\right)$的内积相比，它使用多个向量$\left( {\left\{  {{\mathbf{h}}_{q}^{1},\ldots ,{\mathbf{h}}_{q}^{i}}\right\}  ,\left\{  {{\mathbf{h}}_{p}^{1},\ldots ,{\mathbf{h}}_{p}^{j}}\right\}  }\right)$的后期交互来提高密集检索的鲁棒性。具体来说，Khattab和Zaharia(2020)提出了以下细粒度评分函数：

$$
{\phi }_{\widehat{\theta };\operatorname{MaxSim}} \triangleq  \mathop{\sum }\limits_{{i \in  \left| {\mathbf{h}}_{q}\right| }}\mathop{\max }\limits_{{j \in  \left| {\mathbf{h}}_{p}\right| }}{\mathbf{h}}_{q}^{i} \cdot  {\mathbf{h}}_{p}^{j}, \tag{6}
$$

where $i$ and $j$ are the indices of token representations of a query $q$ and a passage $p$ of ColBERT (Khattab and Zaharia, 2020).

其中$i$和$j$是ColBERT(Khattab和Zaharia, 2020)的查询$q$和段落$p$的词元表示的索引。

The contribution of our work is in-batch knowledge distillation with a tightly-coupled teacher. The computation of ${\phi }_{\widehat{\theta };\text{ MaxSim }}$ enables exhaustive inference over all query-passage combinations in the minibatch $\mathcal{B}$ with only $2 \cdot  \left| \mathcal{B}\right|$ computation cost, enabling enriched interactions between teacher and student. We call this design Tightly-Coupled Teacher ColBERT (TCT-ColBERT). Table 1 provides a training cost comparison between different teachers. When training with pairwise KD, cross-encoders exhibit the highest training cost. On the other hand, ColBERT enables in-batch KD at a modest training cost compared to pairwise KD.

我们工作的贡献在于使用紧密耦合的教师模型进行批量内知识蒸馏。${\phi }_{\widehat{\theta };\text{ MaxSim }}$的计算使得能够以仅$2 \cdot  \left| \mathcal{B}\right|$的计算成本对小批量$\mathcal{B}$中的所有查询 - 段落组合进行穷举推理，从而增强了教师模型和学生模型之间的交互。我们将这种设计称为紧密耦合教师ColBERT(TCT - ColBERT)。表1提供了不同教师模型之间的训练成本比较。在使用成对KD进行训练时，交叉编码器的训练成本最高。另一方面，与成对KD相比，ColBERT能够以适度的训练成本实现批量内KD。

TCT-ColBERT provides a flexible design for bi-encoders, as long as the encoders produce query and passage representations independently. For simplicity, our student model adopts shared encoder weights for both the query and the passage, just like the teacher model ColBERT. Following Khattab and Zaharia (2020), for each query (passage), we prepend the [CLS] token and another special $\left\lbrack  Q\right\rbrack  \left( \left\lbrack  D\right\rbrack  \right)$ token in the input sequence for both our teacher and student models. The student encoder outputs single-vector dense representations $\left( {{\mathbf{h}}_{q},{\mathbf{h}}_{p}}\right)$ by performing average pooling over the token embeddings from the final layer.

只要编码器能够独立生成查询和段落的表示，TCT - ColBERT就为双编码器提供了一种灵活的设计。为简单起见，我们的学生模型和教师模型ColBERT一样，对查询和段落采用共享的编码器权重。遵循Khattab和Zaharia(2020)的方法，对于每个查询(段落)，我们在输入序列的开头添加[CLS]词元和另一个特殊的$\left\lbrack  Q\right\rbrack  \left( \left\lbrack  D\right\rbrack  \right)$词元，用于我们的教师模型和学生模型。学生编码器通过对最后一层的词元嵌入进行平均池化，输出单向量密集表示$\left( {{\mathbf{h}}_{q},{\mathbf{h}}_{p}}\right)$。

<!-- Media -->

Table 1: Training cost comparison. We report the training time per batch against the baseline (without a teacher model) on a single TPU-v2. Our backbone model is BERT-base, with batch size 96. The in-batch cross-encoder training time is not available because it exceeds the memory limit.

表1：训练成本比较。我们报告了在单个TPU - v2上每批次的训练时间与基线(无教师模型)的对比情况。我们的骨干模型是BERT - base，批量大小为96。批量内交叉编码器的训练时间不可用，因为它超出了内存限制。

<table><tr><td>Teacher / KD strategy</td><td>Pairwise</td><td>In-batch</td></tr><tr><td>Cross-encoder $\left( {\phi }_{\widehat{\theta } : \text{ Cat }}\right)$</td><td>+48.1%</td><td>OOM</td></tr><tr><td>ColBERT $\left( {\phi }_{\widehat{\theta };\text{MaxSim}}\right)$</td><td>+32.7%</td><td>+33.5%</td></tr></table>

<table><tbody><tr><td>教师/知识蒸馏（KD）策略</td><td>成对的</td><td>批次内</td></tr><tr><td>交叉编码器 $\left( {\phi }_{\widehat{\theta } : \text{ Cat }}\right)$</td><td>+48.1%</td><td>内存溢出（OOM）</td></tr><tr><td>ColBERT $\left( {\phi }_{\widehat{\theta };\text{MaxSim}}\right)$</td><td>+32.7%</td><td>+33.5%</td></tr></tbody></table>

<!-- Media -->

### 3.3 Hard Negative Sampling

### 3.3 难负样本采样

Given that in-batch negative sampling is an efficient way to add more information into knowledge distillation, we wonder whether our tightly-coupled teacher design works well when applied to more sophisticated sampling methods. Following the work of Xiong et al. (2021), we use our pretrained bi-encoder model, namely TCT-ColBERT, to encode the corpus and sample "hard" negatives for each query to create new training triplets by using the negatives ${\mathcal{D}}^{ - }$ of the bi-encoder instead of BM25. Specifically, we explore three different training strategies:

鉴于批次内负采样是一种向知识蒸馏中添加更多信息的有效方法，我们想知道当应用于更复杂的采样方法时，我们紧密耦合的教师模型设计是否能良好工作。遵循熊（Xiong）等人（2021年）的工作，我们使用预训练的双编码器模型，即TCT - ColBERT，对语料库进行编码，并为每个查询采样“难”负样本，通过使用双编码器的负样本${\mathcal{D}}^{ - }$而非BM25来创建新的训练三元组。具体而言，我们探索了三种不同的训练策略：

1. HN: we train the bi-encoder using in-batch hard negatives without the guide of ColBERT.

1. HN：我们在没有ColBERT指导的情况下，使用批次内难负样本训练双编码器。

2. TCT HN: we train the bi-encoder with TCT-ColBERT;

2. TCT HN：我们使用TCT - ColBERT训练双编码器；

3. TCT HN+: we first fine-tune our ColBERT teacher with augmented training data containing hard negatives and then distill its knowledge into the bi-encoder student through TCT-ColBERT.

3. TCT HN +：我们首先使用包含难负样本的增强训练数据对我们的ColBERT教师模型进行微调，然后通过TCT - ColBERT将其知识蒸馏到双编码器学生模型中。

We empirically explore the effectiveness of these strategies for both passage and document retrieval.

我们通过实验探索了这些策略在段落和文档检索中的有效性。

## 4 Experiments

## 4 实验

In this section, we conduct experiments on the MS MARCO passage and document corpora. For passage ranking, we first train models on BM25 negatives as warm-up and compare different KD methods. We then further train models on the hard negatives retrieved by the BM25 warmed-up checkpoint. For document ranking, following previous work (Xiong et al., 2021; Zhan et al., 2020; Lu et al., 2021), we start with our BM25 warmed-up checkpoint for passage ranking and conduct additional hard negative training.

在本节中，我们在MS MARCO段落和文档语料库上进行实验。对于段落排序，我们首先在BM25负样本上训练模型作为预热，并比较不同的知识蒸馏（KD）方法。然后，我们进一步在由BM25预热检查点检索到的难负样本上训练模型。对于文档排序，遵循先前的工作（熊（Xiong）等人，2021年；詹（Zhan）等人，2020年；陆（Lu）等人，2021年），我们从用于段落排序的BM25预热检查点开始，并进行额外的难负样本训练。

<!-- Media -->

Table 2: Passage retrieval results with BM25 negative training. For knowledge distillation (KD) methods, the effectiveness of teacher (T) models is also reported. All our implemented models are labeled with a number and superscripts represent significant improvements over the labeled model (paired $t$ -test, $p < {0.05}$ ).

表2：使用BM25负样本训练的段落检索结果。对于知识蒸馏（KD）方法，还报告了教师（T）模型的有效性。我们实现的所有模型都标有编号，上标表示相对于标记模型的显著改进（配对$t$检验，$p < {0.05}$）。

<table><tr><td rowspan="2">Strategy</td><td rowspan="2">Model</td><td rowspan="2">#params of Teacher</td><td colspan="2">MARCO Dev</td><td colspan="2">TREC-DL '19</td></tr><tr><td>MRR@10 (T/S)</td><td>R@1K</td><td>NDCG@10 (T/S)</td><td>R@1K</td></tr><tr><td>-</td><td>(1) Baseline</td><td>-</td><td>- 1.310</td><td>.945</td><td>- 1.626</td><td>.658</td></tr><tr><td rowspan="4">Pairwise KD</td><td>KD-T1 (Hofstätter et al., 2020)</td><td>110M</td><td>.376 / .304</td><td>.931</td><td>.730 / .631</td><td>.702</td></tr><tr><td>KD-T2 (Hofstätter et al., 2020)</td><td>467M</td><td>.399 / .315</td><td>.947</td><td>.743 / .668</td><td>.737</td></tr><tr><td>(2) KD-T2 (Ours)</td><td>467M</td><td>${.399}^{ \land  }{.341}^{1}$</td><td>${.964}^{1}$</td><td>.743 / .659 ${}^{1}$</td><td>${.708}^{1}$</td></tr><tr><td>(3) KD-ColBERT</td><td>110M</td><td>$\cdot  {350}^{ \circ  }{1.339}^{1}$</td><td>${.962}^{1}$</td><td>.730 / .670 ${}^{1}$</td><td>${.710}^{1}$</td></tr><tr><td>In-batch KD</td><td>(4) TCT-ColBERT</td><td>110M</td><td>.350 / .344 ${}^{1,3}$</td><td>${\mathbf{{967}}}^{1,3}$</td><td>$\cdot  {730}^{ \cdot  } \cdot  {685}^{1}$</td><td>${.745}^{1,2,3}$</td></tr></table>

<table><tbody><tr><td rowspan="2">策略</td><td rowspan="2">模型</td><td rowspan="2">教师模型的参数数量</td><td colspan="2">MARCO开发集</td><td colspan="2">TREC-DL '19（2019年文本检索会议 - 深度学习任务）</td></tr><tr><td>前10名平均倒数排名（教师模型/学生模型）（MRR@10 (T/S)）</td><td>前1000名召回率（R@1K）</td><td>前10名归一化折损累积增益（教师模型/学生模型）（NDCG@10 (T/S)）</td><td>前1000名召回率（R@1K）</td></tr><tr><td>-</td><td>(1) 基线模型</td><td>-</td><td>- 1.310</td><td>.945</td><td>- 1.626</td><td>.658</td></tr><tr><td rowspan="4">成对知识蒸馏（Pairwise KD）</td><td>KD - T1（霍夫施泰特等人，2020年）</td><td>110M</td><td>.376 / .304</td><td>.931</td><td>.730 / .631</td><td>.702</td></tr><tr><td>KD - T2（霍夫施泰特等人，2020年）</td><td>467M</td><td>.399 / .315</td><td>.947</td><td>.743 / .668</td><td>.737</td></tr><tr><td>(2) KD - T2（我们的方法）</td><td>467M</td><td>${.399}^{ \land  }{.341}^{1}$</td><td>${.964}^{1}$</td><td>.743 / .659 ${}^{1}$</td><td>${.708}^{1}$</td></tr><tr><td>(3) KD - ColBERT</td><td>110M</td><td>$\cdot  {350}^{ \circ  }{1.339}^{1}$</td><td>${.962}^{1}$</td><td>.730 / .670 ${}^{1}$</td><td>${.710}^{1}$</td></tr><tr><td>批次内知识蒸馏（In - batch KD）</td><td>(4) TCT - ColBERT</td><td>110M</td><td>.350 / .344 ${}^{1,3}$</td><td>${\mathbf{{967}}}^{1,3}$</td><td>$\cdot  {730}^{ \cdot  } \cdot  {685}^{1}$</td><td>${.745}^{1,2,3}$</td></tr></tbody></table>

<!-- Media -->

### 4.1 Passage Retrieval

### 4.1 段落检索

We perform ad hoc passage retrieval on the MS MARCO passage ranking dataset (Bajaj et al., 2016),which consists of a collection of ${8.8}\mathrm{M}$ passages from web pages and a set of $\sim  {0.5}\mathrm{M}$ relevant (query, passage) pairs as training data. We evaluate model effectiveness on two test sets of queries:

我们在MS MARCO段落排名数据集（Bajaj等人，2016年）上进行即席段落检索，该数据集包含从网页收集的${8.8}\mathrm{M}$个段落，以及一组$\sim  {0.5}\mathrm{M}$个相关的（查询，段落）对作为训练数据。我们在两个查询测试集上评估模型的有效性：

1. MARCO Dev: the development set of MS MARCO comprises 6980 queries, with an average of one relevant passage per query.

1. MARCO Dev：MS MARCO的开发集包含6980个查询，每个查询平均有一个相关段落。

2. TREC-DL '19 (Craswell et al., 2019): the organizers of the Deep Learning Track at the 2019 Text REtrieval Conference (TREC) released 43 queries with multi-graded(0 - 3)relevance labels on $9\mathrm{\;K}$ (query,passage) pairs.

2. TREC - DL '19（Craswell等人，2019年）：2019年文本检索会议（TREC）深度学习赛道的组织者发布了43个查询，这些查询在$9\mathrm{\;K}$个（查询，段落）对上带有多级（0 - 3）相关性标签。

To evaluate output quality, we report MRR@10 (NDCG@10) for MARCO Dev (TREC-DL ’19) and Recall@1K, denoted as R@1K.To compare with current state-of-the-art models, we evaluate our design, TCT-ColBERT, under two approaches for negative sampling: (1) BM25 and (2) hard negatives retrieved by the bi-encoder itself.

为了评估输出质量，我们报告了MARCO Dev（TREC - DL ’19）的MRR@10（NDCG@10）和Recall@1K，记为R@1K。为了与当前最先进的模型进行比较，我们在两种负采样方法下评估我们的设计TCT - ColBERT：（1）BM25和（2）由双编码器自身检索到的难负样本。

#### 4.1.1 Training with BM25 Negatives

#### 4.1.1 使用BM25负样本进行训练

In this setting, models are trained using the official public data triples. train.small, where negative samples are produced by BM25. We compare different bi-encoder models using BERT-base as the backbone, which uses single 768-dim vectors to represent each query and passage:

在这种设置下，模型使用官方公开数据三元组train.small进行训练，其中负样本由BM25生成。我们比较了以BERT - base为骨干的不同双编码器模型，这些模型使用单个768维向量来表示每个查询和段落：

1. Baseline: a single-vector bi-encoder trained with in-batch negatives, as discussed in Section 2.1, which is similar to Karpukhin et al. (2020) but with a smaller batch size.

1. 基线模型：如2.1节所述，使用批次内负样本训练的单向量双编码器，与Karpukhin等人（2020年）的模型类似，但批次大小更小。

2. Pairwise KD: the approach of Hofstätter et al. (2020), who improve ranking effectiveness using cross-encoders with pairwise KD. We also compare against two models, KD-T1 and KD-T2, which use BERT-base bi-encoders as student models. In the former, the student is distilled from a BERT-base cross-encoder, while the latter is distilled from ensembled cross-encoders comprising BERT-base, BERT-large, and ALBERT-large. These figures reported in Table 2 are copied from Hofstätter et al. (2020). For a fair comparison with our models based on KL-divergence KD, we also implement our KD-T2 using the precomputed pairwise softmax probabilities provided by Hof-stätter et al. (2020) (who use MSE margin loss for KD). In addition, we adopt pairwise softmax probabilities from fine-tuned ColBERT to train KD-ColBERT for comparison.

2. 成对知识蒸馏（Pairwise KD）：Hofstätter等人（2020年）的方法，他们使用带有成对知识蒸馏的交叉编码器提高排名效果。我们还与两个模型KD - T1和KD - T2进行了比较，这两个模型使用BERT - base双编码器作为学生模型。在前者中，学生模型从BERT - base交叉编码器中进行知识蒸馏，而后者则从包含BERT - base、BERT - large和ALBERT - large的集成交叉编码器中进行知识蒸馏。表2中报告的这些数据复制自Hofstätter等人（2020年）。为了与我们基于KL散度知识蒸馏的模型进行公平比较，我们还使用Hofstätter等人（2020年）提供的预计算成对softmax概率来实现我们的KD - T2（他们使用均方误差边际损失进行知识蒸馏）。此外，我们采用经过微调的ColBERT的成对softmax概率来训练KD - ColBERT进行比较。

All our models are fine-tuned with batch size 96 and learning rate $7 \times  {10}^{-6}$ for ${500}\mathrm{\;K}$ steps on a single TPU-V2. For TCT-ColBERT, there are two steps in our training procedure: (1) fine-tune ${\phi }_{\widehat{\theta };\text{ MaxSim }}$ as our teacher model,(2) freeze ${\phi }_{\widehat{\theta } : \text{ MaxSim }}$ and distill knowledge into our student model ${\phi }_{\theta }$ . We keep all the hyperparameter settings the same but adjust temperature $\tau  = {0.25}$ for $\mathrm{{KD}}$ at the second step. For all our models, including the baseline, we initialize the student model using the fine-tuned weights of the teacher model in the first step. We limit the input tokens to 32 (150) for queries (passages). To evaluate effectiveness, we encode all passages in the corpus and conduct brute force search over the vector representations.

我们所有的模型都在单个TPU - V2上以批次大小96和学习率$7 \times  {10}^{-6}$进行${500}\mathrm{\;K}$步的微调。对于TCT - ColBERT，我们的训练过程有两个步骤：（1）将${\phi }_{\widehat{\theta };\text{ MaxSim }}$微调为我们的教师模型，（2）冻结${\phi }_{\widehat{\theta } : \text{ MaxSim }}$并将知识蒸馏到我们的学生模型${\phi }_{\theta }$中。我们保持所有超参数设置不变，但在第二步中调整$\mathrm{{KD}}$的温度$\tau  = {0.25}$。对于我们所有的模型，包括基线模型，我们在第一步中使用教师模型的微调权重来初始化学生模型。我们将查询（段落）的输入令牌限制为32（150）。为了评估有效性，我们对语料库中的所有段落进行编码，并对向量表示进行暴力搜索。

Our main results,including paired $t$ -test for significance testing, are shown in Table 2. In addition to the effectiveness of the student models, we also show the effectiveness of the teacher models for the KD methods. ${}^{1}$

我们的主要结果，包括用于显著性检验的配对$t$检验，如表2所示。除了学生模型的有效性外，我们还展示了知识蒸馏方法中教师模型的有效性。${}^{1}$

First, we see that pairwise KD methods show significant improvements over the baseline, indicating that information from BM25 negatives cannot be fully exploited without teacher models. Second, although KD-T2 improves the bi-encoder's effectiveness over KD-T1, it is not consistently better than KD-ColBERT in terms of students' effectiveness. We suspect that they have comparable capabilities to discriminate most paired passages (BM25 negative vs. positive samples), i.e., ColBERT is good enough to guide bi-encoder student models to discriminate them. On the other hand, our TCT-ColBERT model, which uses only one teacher model and adds only ${33}\%$ more training time over the baseline, yields the best effectiveness, demonstrating the advantages of our proposed in-batch KD - exhaustive exploitation of all query-document combinations in a minibatch.

首先，我们发现成对知识蒸馏（pairwise KD）方法相较于基线有显著提升，这表明如果没有教师模型，来自BM25负样本的信息无法得到充分利用。其次，尽管KD - T2相较于KD - T1提高了双编码器的有效性，但在学生模型的有效性方面，它并不总是优于KD - ColBERT。我们推测，它们在区分大多数成对段落（BM25负样本与正样本）方面具有相当的能力，即ColBERT足以指导双编码器学生模型进行区分。另一方面，我们的TCT - ColBERT模型仅使用一个教师模型，且相较于基线仅增加了${33}\%$的训练时间，却取得了最佳效果，这证明了我们提出的批量内知识蒸馏（in - batch KD）的优势——充分利用小批量中所有查询 - 文档组合的信息。

---

<!-- Footnote -->

${}^{1}$ We report our trained ColBERT’s accuracy by reranking the top-1000 candidates provided officially.

${}^{1}$ 我们通过对官方提供的前1000个候选文档进行重排序，报告了我们训练的ColBERT的准确率。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: 0.360 0.750 KD-T2 0.730 KD-ColBERT TCT-ColBERT NDCG@10 0.710 0.690 0.670 0.650 Index Size $\left( {10}^{6}\right)$ (b) TREC-DL ’19 KD-T2 0.355 KD-ColBERT TCT-ColBERT MRR@10 0.350 0.345 0.340 0.335 5 9 Index Size $\left( {10}^{6}\right)$ (a) MARCO Dev -->

<img src="https://cdn.noedgeai.com/0195aec9-3fad-7d98-91c4-c0e2a3659794_5.jpg?x=290&y=174&w=1065&h=402&r=0"/>

Figure 2: Passage retrieval effectiveness on a synthetic corpus comprising relevant passages and BM25 results as additional "distractors" randomly sampled from the corpus are added.

图2：在一个合成语料库上的段落检索效果，该语料库包含相关段落，并且添加了从语料库中随机采样的BM25结果作为额外的“干扰项”。

<!-- Media -->

To understand why TCT-ColBERT yields better results, we study the models' retrieval effectiveness against carefully selected distractors. We start with a small synthetic corpus composed of the relevant passages and the top-1000 BM25 candidates of the 6980 (43) queries from MARCO Dev (TREC-DL '19). To increase the corpus size, we gradually add passages uniformly sampled from the corpus without replacement. From Figure 2, we see that the three KD models exhibit nearly the same effectiveness when the corpus only contains BM25 candidates. This shows that the bi-encoders learn to discriminate relevant passages from the BM25 negative samples well. However, as the index size increases, TCT-ColBERT demonstrates better ranking effectiveness than the other pairwise KD methods, indicating that the learned representations are more robust. We attribute this robustness against "distractors" to the enriched information from in-batch KD, where we are able to exploit all in-batch query-document combinations.

为了理解为什么TCT - ColBERT能取得更好的结果，我们研究了模型针对精心挑选的干扰项的检索效果。我们从一个小的合成语料库开始，该语料库由相关段落和来自MARCO Dev（TREC - DL '19）的6980（43）个查询的前1000个BM25候选文档组成。为了增加语料库的规模，我们逐渐无放回地添加从语料库中均匀采样的段落。从图2中可以看出，当语料库仅包含BM25候选文档时，三种知识蒸馏模型表现出几乎相同的效果。这表明双编码器能够很好地学会从BM25负样本中区分相关段落。然而，随着索引规模的增加，TCT - ColBERT的排序效果优于其他成对知识蒸馏方法，这表明所学的表示更具鲁棒性。我们将这种对“干扰项”的鲁棒性归因于批量内知识蒸馏所提供的丰富信息，在这种方法中，我们能够利用批量内所有查询 - 文档组合的信息。

#### 4.1.2 Training with Hard Negatives

#### 4.1.2 使用难负样本进行训练

In this subsection, we evaluate TCT-ColBERT when training with hard negatives (HNs). We compare our model to four competitive approaches:

在本小节中，我们评估了TCT - ColBERT在使用难负样本（HNs）进行训练时的表现。我们将我们的模型与四种有竞争力的方法进行了比较：

1. ANCE (Xiong et al., 2021) is the most representative work, which proposes asynchronous index refreshes to mine hard negatives. The model is trained for ${600}\mathrm{\;K}$ steps with index refreshes every ${10}\mathrm{\;K}$ steps. ANCE uses RoBERTa-base as its backbone.

1. ANCE（Xiong等人，2021）是最具代表性的工作，它提出了异步索引更新来挖掘难负样本。该模型训练${600}\mathrm{\;K}$步，每${10}\mathrm{\;K}$步进行一次索引更新。ANCE使用RoBERTa - base作为其骨干网络。

2. LTRe (Zhan et al., 2020) further improves from an ANCE checkpoint by adding more training steps with the same hard negative mining approach; thus, the computation cost of index refreshes from ANCE cannot be neglected. LTRe also use RoBERTa-base as its backbone.

2. LTRe（Zhan等人，2020）在ANCE的检查点基础上进一步改进，通过使用相同的难负样本挖掘方法增加了更多的训练步骤；因此，ANCE中索引更新的计算成本不可忽视。LTRe也使用RoBERTa - base作为其骨干网络。

3. SEED-Encoder (Lu et al., 2021) leverages a pretraining strategy to enhance the capability of the bi-encoder, which is further fine-tuned with HNs using asynchronous index refreshes.

3. SEED - Encoder（Lu等人，2021）利用预训练策略来增强双编码器的能力，然后使用异步索引更新对其进行难负样本的微调。

4. RocketQA (Qu et al., 2020) trains a bi-encoder model using hard negatives denoised by a cross-encoder, ERNIE-2.0-Large (Sun et al., 2019). It further demonstrates that training bi-encoders with many in-batch negatives (batch size up to 4096) significantly improves ranking effectiveness; however, this approach is computationally expensive (the authors report using $8 \times  \mathrm{V}{100}$ GPUs for training). To the best of our knowledge, RocketQA represents the state of the art in single-vector bi-encoders for dense retrieval. For a more fair comparison, we also report the ranking effectiveness of their model trained with a smaller batch size of 128 .

4. RocketQA（Qu等人，2020）使用由交叉编码器ERNIE - 2.0 - Large（Sun等人，2019）去噪的难负样本训练双编码器模型。它进一步表明，使用大量批量内负样本（批量大小高达4096）训练双编码器可以显著提高排序效果；然而，这种方法的计算成本很高（作者报告使用$8 \times  \mathrm{V}{100}$个GPU进行训练）。据我们所知，RocketQA代表了用于密集检索的单向量双编码器的当前最优水平。为了进行更公平的比较，我们还报告了其使用较小批量大小128训练的模型的排序效果。

For all the approaches above, we directly copy the reported effectiveness from the original papers.

对于上述所有方法，我们直接从原始论文中复制了报告的效果。

<!-- Media -->

Table 3: Passage retrieval results with hard negative training. All our implemented models are labeled with a number and superscripts represent significant improvements over the labeled model (paired $t$ -test, $p < {0.05}$ ).

表3：使用难负样本训练的段落检索结果。我们实现的所有模型都标有编号，上标表示相较于标注模型有显著改进（配对$t$检验，$p < {0.05}$）。

<table><tr><td rowspan="2">Model</td><td rowspan="2">#Index Refresh</td><td rowspan="2">Batch Size</td><td colspan="2">MARCO Dev</td><td colspan="2">TREC-DL ’19</td></tr><tr><td>MRR@10</td><td>R@1K</td><td>NDCG@10</td><td>R@1K</td></tr><tr><td>ANCE (Xiong et al., 2021)</td><td>60</td><td>32</td><td>.330</td><td>.959</td><td>.648</td><td>-</td></tr><tr><td>LTRe (Zhan et al., 2020)</td><td>60</td><td>32</td><td>.341</td><td>.962</td><td>.675</td><td>-</td></tr><tr><td>SEED-Encoder (Lu et al., 2021)</td><td>≥10 (est.)</td><td>-</td><td>.339</td><td>.961</td><td>-</td><td>-</td></tr><tr><td>RocketQA (Qu et al., 2020)</td><td>1</td><td>128</td><td>.310</td><td>-</td><td>-</td><td>-</td></tr><tr><td>RocketQA (Qu et al., 2020)</td><td>1</td><td>4096</td><td>.364</td><td>-</td><td>-</td><td>-</td></tr><tr><td>(1) TCT-ColBERT</td><td>0</td><td>96</td><td>.344</td><td>.967</td><td>.685</td><td>.745</td></tr><tr><td>(2) w/ HN</td><td>1</td><td>96</td><td>.237</td><td>.929</td><td>.543</td><td>.674</td></tr><tr><td>(3) w/ TCT HN</td><td>1</td><td>96</td><td>${.354}^{1,2}$</td><td>${\mathbf{{.971}}}^{1,2}$</td><td>${.705}^{2}$</td><td>.765 ${}^{1,2}$</td></tr><tr><td>(4) w/ TCT HN+</td><td>1</td><td>96</td><td>${.359}^{1,2}$</td><td>${.970}^{1}$</td><td>${\mathbf{{.719}}}^{1,2}$</td><td>${.760}^{1}$</td></tr></table>

<table><tbody><tr><td rowspan="2">模型</td><td rowspan="2">#索引刷新</td><td rowspan="2">批量大小</td><td colspan="2">MARCO开发集</td><td colspan="2">TREC-DL ’19（2019年文本检索会议深度学习评测任务）</td></tr><tr><td>前10名平均倒数排名（MRR@10）</td><td>前1000名召回率（R@1K）</td><td>前10名归一化折损累积增益（NDCG@10）</td><td>前1000名召回率（R@1K）</td></tr><tr><td>ANCE（熊等人，2021年）</td><td>60</td><td>32</td><td>.330</td><td>.959</td><td>.648</td><td>-</td></tr><tr><td>LTRe（詹等人，2020年）</td><td>60</td><td>32</td><td>.341</td><td>.962</td><td>.675</td><td>-</td></tr><tr><td>SEED编码器（陆等人，2021年）</td><td>≥10（估计值）</td><td>-</td><td>.339</td><td>.961</td><td>-</td><td>-</td></tr><tr><td>火箭问答（曲等人，2020年）</td><td>1</td><td>128</td><td>.310</td><td>-</td><td>-</td><td>-</td></tr><tr><td>火箭问答（曲等人，2020年）</td><td>1</td><td>4096</td><td>.364</td><td>-</td><td>-</td><td>-</td></tr><tr><td>(1) TCT - 科尔伯特（TCT - ColBERT）</td><td>0</td><td>96</td><td>.344</td><td>.967</td><td>.685</td><td>.745</td></tr><tr><td>(2) 带困难负样本（w/ HN）</td><td>1</td><td>96</td><td>.237</td><td>.929</td><td>.543</td><td>.674</td></tr><tr><td>(3) 带TCT困难负样本（w/ TCT HN）</td><td>1</td><td>96</td><td>${.354}^{1,2}$</td><td>${\mathbf{{.971}}}^{1,2}$</td><td>${.705}^{2}$</td><td>.765 ${}^{1,2}$</td></tr><tr><td>(4) 带TCT增强困难负样本（w/ TCT HN+）</td><td>1</td><td>96</td><td>${.359}^{1,2}$</td><td>${.970}^{1}$</td><td>${\mathbf{{.719}}}^{1,2}$</td><td>${.760}^{1}$</td></tr></tbody></table>

<!-- Media -->

For our TCT-ColBERT model, following the settings of the above approaches, we first use our TCT-ColBERT model trained on BM25 negatives as a warm-up starting point and index all ${8.8}\mathrm{M}$ MARCO passages. Using the warmed-up index, we retrieve top-200 passages for each training query and randomly sample (with replacement) hard negatives from the 200 candidates to form our training data. Note that due to resource limitations we do not conduct experiments with asynchronous index refreshes since multiple V100 GPUs are required for such a model training scheme. ${}^{2}$ In this experiment, all the hyperparameter settings are the same as the ones in the BM25 negative training, except for training steps,which is set to ${100}\mathrm{\;K}$ for both student and teacher training.

对于我们的TCT - ColBERT模型，遵循上述方法的设置，我们首先使用在BM25负样本上训练的TCT - ColBERT模型作为预热起点，并对所有${8.8}\mathrm{M}$ MARCO段落进行索引。利用预热后的索引，我们为每个训练查询检索前200个段落，并从这200个候选段落中有放回地随机采样难负样本（hard negatives）以形成我们的训练数据。请注意，由于资源限制，我们没有进行异步索引刷新的实验，因为这种模型训练方案需要多个V100 GPU。${}^{2}$ 在这个实验中，除了训练步数（学生模型和教师模型的训练步数均设置为${100}\mathrm{\;K}$）之外，所有超参数设置都与BM25负样本训练中的设置相同。

Table 3 reports the results of our experiments with hard negative training. First, we observe that our TCT-ColBERT model trained with BM25 negatives marginally outperforms the other models trained with HNs, except for RocketQA. Comparing the different training strategies discussed in Section 3.3 (second main block of the table), we see that the ranking effectiveness of TCT-ColBERT (HN) degrades when training on hard negatives without the guide of a teacher. This is consistent with the findings of Qu et al. (2020) that hard negatives contain noisy information (i.e., some hard negatives may actually be relevant). Also, Xiong et al. (2021) show that training bi-encoders with hard negatives can be unstable: hard negatives benefit ranking effectiveness only under certain hyper-parameter settings.

表3报告了我们进行难负样本训练实验的结果。首先，我们观察到，除了RocketQA之外，使用BM25负样本训练的TCT - ColBERT模型略优于使用难负样本（HNs）训练的其他模型。比较第3.3节讨论的不同训练策略（表中的第二个主要部分），我们发现，在没有教师模型指导的情况下对难负样本进行训练时，TCT - ColBERT（HN）的排序效果会下降。这与Qu等人（2020）的研究结果一致，即难负样本包含噪声信息（即，一些难负样本实际上可能是相关的）。此外，Xiong等人（2021）表明，使用难负样本训练双编码器可能不稳定：难负样本仅在某些超参数设置下才有利于排序效果。

In contrast, hard negative training using ColBERT's in-batch KD further boosts ranking effectiveness, especially when our teacher (ColBERT) is trained with the same hard negative samples beforehand. It is also worth noting that our TCT-ColBERT (w/ TCT HN+) with batch size 96 yields competitive ranking effectiveness compared to RocketQA (the current state of the art), which uses batch size 4096. These results demonstrate the advantages of our TCT design: our approach effectively exploits hard negatives in a computationally efficient manner (i.e., without the need for large batch sizes or periodic index refreshes).

相比之下，使用ColBERT的批内知识蒸馏（in - batch KD）进行难负样本训练进一步提高了排序效果，特别是当我们的教师模型（ColBERT）事先使用相同的难负样本进行训练时。还值得注意的是，批量大小为96的TCT - ColBERT（w/ TCT HN+）与批量大小为4096的RocketQA（当前的最优模型）相比，产生了具有竞争力的排序效果。这些结果证明了我们TCT设计的优势：我们的方法以计算高效的方式（即，无需大的批量大小或定期刷新索引）有效地利用了难负样本。

### 4.2 Document Retrieval

### 4.2 文档检索

To validate the effectiveness and generality of our training strategy, we conduct further experiments on document retrieval using the MS MARCO document ranking dataset. This dataset contains ${3.2}\mathrm{M}$ web pages gathered from passages in the MS MARCO passage ranking dataset. Similar to the passage condition, we evaluate model effectiveness on two test sets of queries:

为了验证我们训练策略的有效性和通用性，我们使用MS MARCO文档排序数据集对文档检索进行了进一步的实验。该数据集包含从MS MARCO段落排序数据集中的段落收集的${3.2}\mathrm{M}$个网页。与段落检索情况类似，我们在两个查询测试集上评估模型的有效性：

1. MARCO Dev: the development set contains 5193 queries, each with exactly one relevant document.

1. MARCO开发集（Dev）：开发集包含5193个查询，每个查询恰好有一个相关文档。

2. TREC-DL '19: graded relevance judgments are available from the TREC 2019 Deep Learning Track, but on only 43 queries.

2. TREC - DL '19：可以从TREC 2019深度学习赛道获得分级相关性判断，但仅针对43个查询。

Per official guidelines, we report different metrics for the two query sets: MRR@100 for MARCO Dev and NDCG@10 for TREC-DL '19.

根据官方指南，我们为这两个查询集报告不同的指标：MARCO开发集使用MRR@100，TREC - DL '19使用NDCG@10。

Following the FirstP setting for document retrieval described in Xiong et al. (2021), we feed the first 512 tokens of each document for encoding, and start with the warmed-up checkpoint for our encoder's parameters trained for passage retrieval (using BM25 negatives, as described in Section 4.1.1). The settings for fine-tuning our warmed-up encoder (e.g., learning rate, training steps, top-200 negative sampling) are the same as passage retrieval except for batch size, which is set to 64 .

遵循Xiong等人（2021）中描述的文档检索的FirstP设置，我们输入每个文档的前512个标记进行编码，并从为段落检索训练的编码器参数的预热检查点开始（使用BM25负样本，如第4.1.1节所述）。微调我们预热后的编码器的设置（例如，学习率、训练步数、前200个负样本采样）与段落检索相同，除了批量大小设置为64。

---

<!-- Footnote -->

${}^{2}$ Re-encoding the entire corpus takes $\sim  {10}$ hours on one GPU.

${}^{2}$ 在一个GPU上重新编码整个语料库需要$\sim  {10}$小时。

<!-- Footnote -->

---

<!-- Media -->

Table 4: Document retrieval results using the FirstP approach. All our implemented models are labeled with a number and superscripts represent significant improvements over the labeled model (paired $t$ -test, $p < {0.05}$ ).

表4：使用FirstP方法的文档检索结果。我们实现的所有模型都标有编号，上标表示相对于标记模型的显著改进（配对$t$检验，$p < {0.05}$）。

<table><tr><td rowspan="2">Model</td><td>MARCO Dev</td><td>TREC-DL ’19</td></tr><tr><td>MRR@100</td><td>NDCG@10</td></tr><tr><td>ANCE (Xiong et al., 2021)</td><td>.368</td><td>.614</td></tr><tr><td>LTRe (Zhan et al., 2020)</td><td>-</td><td>.634</td></tr><tr><td>SEED-Encoder (Lu et al., 2021)</td><td>.394</td><td>-</td></tr><tr><td>(1) TCT-ColBERT</td><td>.339</td><td>.573</td></tr><tr><td>(2) w/ TCT HN+</td><td>${.392}^{1}$</td><td>.613</td></tr><tr><td>(3) w/2 × TCT HN+</td><td>${\mathbf{{.418}}}^{1,2}$</td><td>${\mathbf{{.650}}}^{1,2}$</td></tr></table>

<table><tbody><tr><td rowspan="2">模型</td><td>MARCO开发集</td><td>TREC-DL ’19（2019年文本检索会议 - 深度学习任务）</td></tr><tr><td>前100名平均倒数排名（MRR@100）</td><td>前10名归一化折损累积增益（NDCG@10）</td></tr><tr><td>ANCE（熊等人，2021年）</td><td>.368</td><td>.614</td></tr><tr><td>LTRe（詹等人，2020年）</td><td>-</td><td>.634</td></tr><tr><td>SEED编码器（陆等人，2021年）</td><td>.394</td><td>-</td></tr><tr><td>(1) TCT - 科尔伯特</td><td>.339</td><td>.573</td></tr><tr><td>(2) 搭配TCT HN+</td><td>${.392}^{1}$</td><td>.613</td></tr><tr><td>(3) 搭配2 × TCT HN+</td><td>${\mathbf{{.418}}}^{1,2}$</td><td>${\mathbf{{.650}}}^{1,2}$</td></tr></tbody></table>

<!-- Media -->

Ranking effectiveness is reported in Table 4. First, we observe that TCT-ColBERT (our warmed-up checkpoint) performs far worse than other approaches to document retrieval using the FirstP method. This may be due to the fact that FirstP document retrieval is very different from passage retrieval, making zero-shot transfer ineffective. After applying HN training on both teacher and student models (condition 2), the ranking effectiveness increases significantly. In addition, we find that another iteration of training with an index refresh (condition 3) further improves ranking effectiveness. To sum up, in the document ranking task, TCT-ColBERT yields competitive effectiveness with a one-time index refresh and outperforms other computationally expensive methods with one additional index refresh.

表4报告了排序效果。首先，我们观察到TCT - ColBERT（我们预热的检查点）在使用FirstP方法进行文档检索时的表现远不如其他方法。这可能是因为FirstP文档检索与段落检索有很大不同，使得零样本迁移无效。在教师模型和学生模型上都应用困难负样本（HN）训练后（条件2），排序效果显著提高。此外，我们发现通过刷新索引进行另一轮训练（条件3）进一步提高了排序效果。综上所述，在文档排序任务中，TCT - ColBERT通过一次性索引刷新就能产生有竞争力的效果，并且在再进行一次索引刷新后，其性能优于其他计算成本高的方法。

### 4.3 Dense-Sparse Hybrids

### 4.3 稠密 - 稀疏混合方法

In our final set of experiments, we show that dense retrieval with single-vector representations can be integrated with results from sparse retrieval to further increase effectiveness. We illustrate the end-to-end tradeoffs in terms of quality, time, and space of different dense-sparse hybrid combinations on the passage retrieval tasks.

在最后一组实验中，我们表明单向量表示的稠密检索可以与稀疏检索的结果相结合，以进一步提高检索效果。我们在段落检索任务中说明了不同稠密 - 稀疏混合组合在质量、时间和空间方面的端到端权衡。

Many papers (Luan et al., 2021; Gao et al., 2020b; Ma et al., 2021; Lin et al., 2021) have demonstrated that sparse retrieval can complement dense retrieval via a simple linear combination of their scores. In our implementation, for each query $q$ ,we use sparse and dense techniques to retrieve the top-1000 passages, ${\mathcal{D}}_{sp}$ and ${\mathcal{D}}_{ds}$ , with their relevance scores, ${\phi }_{sp}\left( {q,p \in  {\mathcal{D}}_{sp}}\right)$ and ${\phi }_{ds}\left( {q,p \in  {\mathcal{D}}_{ds}}\right)$ ,respectively. Then,we compute the final relevance score for each retrieved passage $\phi \left( {q,p}\right)$ ,where $p \in  {\mathcal{D}}_{sp} \cup  {\mathcal{D}}_{ds}$ ,as follows:

许多论文（Luan等人，2021；Gao等人，2020b；Ma等人，2021；Lin等人，2021）已经证明，稀疏检索可以通过简单的线性组合其得分来补充稠密检索。在我们的实现中，对于每个查询$q$，我们使用稀疏和稠密技术分别检索前1000个段落${\mathcal{D}}_{sp}$和${\mathcal{D}}_{ds}$，并得到它们的相关性得分${\phi }_{sp}\left( {q,p \in  {\mathcal{D}}_{sp}}\right)$和${\phi }_{ds}\left( {q,p \in  {\mathcal{D}}_{ds}}\right)$。然后，我们计算每个检索到的段落$\phi \left( {q,p}\right)$的最终相关性得分，其中$p \in  {\mathcal{D}}_{sp} \cup  {\mathcal{D}}_{ds}$，如下所示：

$$
\left\{  \begin{array}{ll} \alpha  \cdot  {\phi }_{sp}\left( {q,p}\right)  + \mathop{\min }\limits_{{p \in  {\mathcal{D}}_{ds}}}{\phi }_{ds}\left( {q,p}\right) , & \text{ if }p \notin  {D}_{ds} \\  \alpha  \cdot  \mathop{\min }\limits_{{p \in  {\mathcal{D}}_{sp}}}{\phi }_{sp}\left( {q,p}\right)  + {\phi }_{ds}\left( {q,p}\right) , & \text{ if }p \notin  {D}_{sp} \\  \alpha  \cdot  {\phi }_{sp}\left( {q,p}\right)  + {\phi }_{ds}\left( {q,p}\right) , & \text{ otherwise. } \end{array}\right. 
$$

This technique is an approximation of a linear combination of sparse and dense retrieval scores. Specifically,if $p \notin  {\mathcal{D}}_{sp}\left( \right.$ or $\left. {\mathcal{D}}_{ds}\right)$ ,we instead use the minimum score of ${\phi }_{sp}\left( {q,p \in  {\mathcal{D}}_{sp}}\right)$ ,or ${\phi }_{ds}(q,p \in$ $\left. {\mathcal{D}}_{ds}\right)$ as a substitute.

这种技术是稀疏和稠密检索得分线性组合的一种近似方法。具体来说，如果$p \notin  {\mathcal{D}}_{sp}\left( \right.$或$\left. {\mathcal{D}}_{ds}\right)$，我们将使用${\phi }_{sp}\left( {q,p \in  {\mathcal{D}}_{sp}}\right)$或${\phi }_{ds}(q,p \in$ $\left. {\mathcal{D}}_{ds}\right)$的最低得分作为替代。

For the sparse and dense retrieval combinations,we tune the hyperparameter $\alpha$ on 6000 randomly sampled queries from the MS MARCO training set. We conduct dense-sparse hybrid experiments with sparse retrieval (BM25 ranking) on the original passages (denoted BM25) and on passages with docTTTTTquery document expansion (Nogueira and Lin, 2019) (denoted doc2query-T5). To characterize end-to-end effectiveness and efficiency, we perform sparse retrieval with the Py-serini toolkit (Lin et al., 2021) and dense retrieval with Faiss (Johnson et al., 2017), but implement the score combination in separate custom code.

对于稀疏和稠密检索组合，我们在从MS MARCO训练集中随机抽取的6000个查询上调整超参数$\alpha$。我们在原始段落（记为BM25）和经过docTTTTTquery文档扩展的段落（Nogueira和Lin，2019）（记为doc2query - T5）上进行了结合稀疏检索（BM25排序）的稠密 - 稀疏混合实验。为了描述端到端的效果和效率，我们使用Py - serini工具包（Lin等人，2021）进行稀疏检索，使用Faiss（Johnson等人，2017）进行稠密检索，但在单独的自定义代码中实现得分组合。

Table 5 shows passage retrieval results in terms of ranking effectiveness, query latency, and storage requirements (i.e., index size) for each model and Table 6 reports the component latencies of our TCT-ColBERT dense-sparse hybrid. ${}^{3}$ The cross-encoder reranker of Nogueira and Cho (2019) provides a point of reference for multi-stage reranking designs, which is effective but slow.

表5显示了每个模型在段落检索方面的排序效果、查询延迟和存储要求（即索引大小），表6报告了我们的TCT - ColBERT稠密 - 稀疏混合模型的组件延迟。${}^{3}$ Nogueira和Cho（2019）的交叉编码器重排器为多阶段重排设计提供了一个参考点，它有效但速度较慢。

Generally, dense retrieval methods (whether single-vector or multi-vector) are more effective but slower than sparse retrieval methods, which rely on bag-of-words querying using inverted indexes. Single-vector dense models also require more space than sparse retrieval methods. Moving from single-vector to multi-vector dense models, we see that ColBERT exhibits higher effectiveness but is slower and requires much more storage.

一般来说，稠密检索方法（无论是单向量还是多向量）比稀疏检索方法更有效，但速度更慢，稀疏检索方法依赖于使用倒排索引的词袋查询。单向量稠密模型也比稀疏检索方法需要更多的空间。从单向量稠密模型转向多向量稠密模型，我们发现ColBERT表现出更高的效果，但速度更慢，并且需要更多的存储空间。

---

<!-- Footnote -->

${}^{3}$ Here we assume running dense and sparse retrieval in parallel.

${}^{3}$ 这里我们假设同时并行运行稠密和稀疏检索。

<!-- Footnote -->

---

<!-- Media -->

Table 5: End-to-end comparisons of output quality, query latency, and storage requirements for passage retrieval.

表5：段落检索在输出质量、查询延迟和存储要求方面的端到端比较。

<table><tr><td rowspan="2"/><td colspan="2">Ranking effectiveness</td><td>Latency</td><td>Storage</td></tr><tr><td>MARCO Dev</td><td>TREC-DL ’19</td><td>ms/q</td><td>GiB</td></tr><tr><td colspan="5">Sparse retrieval</td></tr><tr><td>BM25 with Anserini (Yang et al., 2018)</td><td>.184</td><td>.506</td><td>55</td><td>4</td></tr><tr><td>DeepCT (Dai and Callan, 2020)</td><td>.243</td><td>.551</td><td>55</td><td>4</td></tr><tr><td>doc2query-T5 (Nogueira and Lin, 2019)</td><td>.277</td><td>.551</td><td>64</td><td>14</td></tr><tr><td colspan="5">Dense retrieval: single-vector</td></tr><tr><td>TAS-B (Hofstätter et al., 2021)</td><td>.343</td><td>.722</td><td>64</td><td>13</td></tr><tr><td>RocketQA (Qu et al., 2020)</td><td>.370</td><td>-</td><td>${107}^{\mathrm{b}}$</td><td>${13}^{\mathrm{a}}$</td></tr><tr><td>TCT-ColBERT</td><td>.344</td><td>.685</td><td>107</td><td>13</td></tr><tr><td>TCT-ColBERT (w/ TCT HN+)</td><td>.359</td><td>.719</td><td>107</td><td>13</td></tr><tr><td colspan="5">Dense retrieval: multi-vector</td></tr><tr><td>ME-BERT (Luan et al., 2021)</td><td>.334</td><td>.687</td><td>-</td><td>96</td></tr><tr><td>ColBERT (Khattab and Zaharia, 2020)</td><td>.360</td><td>-</td><td>458</td><td>154</td></tr><tr><td colspan="5">Hybrid dense + sparse</td></tr><tr><td>CLEAR (Gao et al., 2020b)</td><td>.338</td><td>.699</td><td>-</td><td>${17}^{\mathrm{a}}$</td></tr><tr><td>ME-HYBRID-E (Luan et al., 2021)</td><td>.343</td><td>.706</td><td>-</td><td>100</td></tr><tr><td>TAS-B + doc2query-T5 (Hofstätter et al., 2021)</td><td>.360</td><td>.753</td><td>67</td><td>${27}^{\mathrm{a}}$</td></tr><tr><td>TCT-ColBERT + BM25</td><td>.356</td><td>.720</td><td>110</td><td>17</td></tr><tr><td>TCT-ColBERT + doc2query-T5</td><td>.366</td><td>.734</td><td>110</td><td>27</td></tr><tr><td>TCT-ColBERT (w/ TCT HN+) + BM25</td><td>.369</td><td>.730</td><td>110</td><td>17</td></tr><tr><td>TCT-ColBERT (w/ TCT HN+) + doc2query-T5</td><td>.375</td><td>.741</td><td>110</td><td>27</td></tr><tr><td colspan="5">Multi-stage reranking</td></tr><tr><td>BM25 + BERT-large (Nogueira and Cho, 2019)</td><td>.365</td><td>.736</td><td>3500</td><td>4</td></tr><tr><td>TAS-B + doc2query-T5 + Mono-Duo-T5 (Hofstätter et al., 2021)</td><td>.421</td><td>.759</td><td>12800</td><td>${27}^{\mathrm{a}}$</td></tr><tr><td>RocketQA with reranking (Qu et al., 2020)</td><td>.439</td><td>-</td><td>-</td><td>${13}^{\mathrm{a}}$</td></tr></table>

<table><tbody><tr><td rowspan="2"></td><td colspan="2">排序效果</td><td>延迟</td><td>存储</td></tr><tr><td>MARCO开发集</td><td>TREC-DL ’19（2019年文本检索会议 - 文档检索任务）</td><td>毫秒/查询</td><td>吉字节（GiB）</td></tr><tr><td colspan="5">稀疏检索</td></tr><tr><td>使用Anserini的BM25算法（杨等人，2018年）</td><td>.184</td><td>.506</td><td>55</td><td>4</td></tr><tr><td>DeepCT算法（戴和卡兰，2020年）</td><td>.243</td><td>.551</td><td>55</td><td>4</td></tr><tr><td>doc2query - T5算法（诺盖拉和林，2019年）</td><td>.277</td><td>.551</td><td>64</td><td>14</td></tr><tr><td colspan="5">密集检索：单向量</td></tr><tr><td>TAS - B算法（霍夫施泰特等人，2021年）</td><td>.343</td><td>.722</td><td>64</td><td>13</td></tr><tr><td>RocketQA算法（曲等人，2020年）</td><td>.370</td><td>-</td><td>${107}^{\mathrm{b}}$</td><td>${13}^{\mathrm{a}}$</td></tr><tr><td>TCT - ColBERT算法</td><td>.344</td><td>.685</td><td>107</td><td>13</td></tr><tr><td>TCT - ColBERT算法（使用TCT HN+）</td><td>.359</td><td>.719</td><td>107</td><td>13</td></tr><tr><td colspan="5">密集检索：多向量</td></tr><tr><td>ME - BERT算法（栾等人，2021年）</td><td>.334</td><td>.687</td><td>-</td><td>96</td></tr><tr><td>ColBERT算法（哈塔卜和扎哈里亚，2020年）</td><td>.360</td><td>-</td><td>458</td><td>154</td></tr><tr><td colspan="5">混合密集 + 稀疏检索</td></tr><tr><td>CLEAR算法（高等人，2020b）</td><td>.338</td><td>.699</td><td>-</td><td>${17}^{\mathrm{a}}$</td></tr><tr><td>ME - HYBRID - E算法（栾等人，2021年）</td><td>.343</td><td>.706</td><td>-</td><td>100</td></tr><tr><td>TAS - B + doc2query - T5算法（霍夫施泰特等人，2021年）</td><td>.360</td><td>.753</td><td>67</td><td>${27}^{\mathrm{a}}$</td></tr><tr><td>TCT - ColBERT + BM25算法</td><td>.356</td><td>.720</td><td>110</td><td>17</td></tr><tr><td>TCT - ColBERT + doc2query - T5算法</td><td>.366</td><td>.734</td><td>110</td><td>27</td></tr><tr><td>TCT - ColBERT（使用TCT HN+）+ BM25算法</td><td>.369</td><td>.730</td><td>110</td><td>17</td></tr><tr><td>TCT - ColBERT（使用TCT HN+）+ doc2query - T5算法</td><td>.375</td><td>.741</td><td>110</td><td>27</td></tr><tr><td colspan="5">多阶段重排序</td></tr><tr><td>BM25 + BERT - large算法（诺盖拉和赵，2019年）</td><td>.365</td><td>.736</td><td>3500</td><td>4</td></tr><tr><td>TAS - B + doc2query - T5 + Mono - Duo - T5算法（霍夫施泰特等人，2021年）</td><td>.421</td><td>.759</td><td>12800</td><td>${27}^{\mathrm{a}}$</td></tr><tr><td>带重排序的RocketQA算法（曲等人，2020年）</td><td>.439</td><td>-</td><td>-</td><td>${13}^{\mathrm{a}}$</td></tr></tbody></table>

${}^{a}$ We estimate dense index size using 16-bit floats; for hybrid,we add the sizes of sparse and dense indexes.

${}^{a}$ 我们使用16位浮点数来估算密集索引（dense index）的大小；对于混合索引，我们将稀疏索引（sparse index）和密集索引的大小相加。

${}^{\mathrm{b}}$ We assume latency comparable to our settings.

${}^{\mathrm{b}}$ 我们假设延迟与我们的设置相当。

Table 6: Component latencies per query of our model.

表6：我们模型每个查询的组件延迟。

<table><tr><td>Stage</td><td>latency (ms)</td><td>device</td></tr><tr><td>BERT query encoder</td><td>7</td><td>GPU</td></tr><tr><td>Dot product search</td><td>100</td><td>GPU</td></tr><tr><td>Score combination</td><td>3</td><td>CPU</td></tr></table>

<table><tbody><tr><td>阶段</td><td>延迟（毫秒）</td><td>设备</td></tr><tr><td>BERT查询编码器</td><td>7</td><td>图形处理器（GPU）</td></tr><tr><td>点积搜索</td><td>100</td><td>图形处理器（GPU）</td></tr><tr><td>分数组合</td><td>3</td><td>中央处理器（CPU）</td></tr></tbody></table>

<!-- Media -->

Finally, when integrated with sparse retrieval methods, TCT-ColBERT is able to beat a basic multi-stage reranking design (BM25 + BERT-large), but with much lower query latency, although at the cost of increased storage. Hybrid TCT-ColBERT (w/ TCT HN+) + doc2query-T5 compares favorably with a recent advanced model, TAS-B + doc2query-T5 (Hofstätter et al., 2021), which introduces topic-aware sampling and dual teachers, incorporating part of our TCT-ColBERT work. Nevertheless, even the best hybrid variant of TCT-ColBERT alone, without further rerank-ing, remains quite some distance from RocketQA, the current state of the art (with reranking using cross-encoders). This suggests that there remain relevance signals that require full attention interactions to exploit.

最后，当与稀疏检索方法结合使用时，TCT - ColBERT（紧密耦合Transformer上下文表示 - 基于上下文的双向编码器表示法）能够击败基本的多阶段重排序设计（BM25 + BERT - large），并且查询延迟要低得多，不过代价是增加了存储需求。混合TCT - ColBERT（搭配TCT HN +）+ doc2query - T5与近期的先进模型TAS - B + doc2query - T5（霍夫施泰特等人，2021年）相比表现更优，后者引入了主题感知采样和双教师机制，并融入了我们TCT - ColBERT工作的一部分。然而，即使是TCT - ColBERT最佳的混合变体，在不进行进一步重排序的情况下，与当前最先进的RocketQA（使用交叉编码器进行重排序）仍有相当大的差距。这表明仍存在一些相关性信号，需要通过充分的注意力交互才能加以利用。

## 5 Conclusions

## 5 结论

Improving the effectiveness of single-vector bi-encoders is an important research direction in dense retrieval because of lower latency and storage requirements compared to multi-vector approaches. We propose a teacher-student knowledge distillation approach using tightly coupled bi-encoders that enables exhaustive use of query-passage combinations in each minibatch. More importantly, a bi-encoder teacher requires less computation than a cross-encoder teacher. Finally, our approach leads to robust learned representations.

提高单向量双编码器的有效性是密集检索中的一个重要研究方向，因为与多向量方法相比，它对延迟和存储的要求更低。我们提出了一种使用紧密耦合双编码器的师生知识蒸馏方法，该方法能够充分利用每个小批量中的查询 - 段落组合。更重要的是，双编码器教师所需的计算量比交叉编码器教师少。最后，我们的方法能够得到稳健的学习表示。

Overall, our hard negative sampling strategy leads to an effective and efficient dense retrieval technique, which can be further combined with sparse retrieval techniques in dense-sparse hybrids. Together, these designs provide a promising solution for end-to-end text retrieval that balances quality, query latency, and storage requirements.

总体而言，我们的难负样本采样策略产生了一种高效且有效的密集检索技术，该技术可以在密集 - 稀疏混合模型中进一步与稀疏检索技术相结合。这些设计共同为端到端文本检索提供了一个有前景的解决方案，能够在检索质量、查询延迟和存储需求之间取得平衡。

## Acknowledgements

## 致谢

This research was supported in part by the Canada First Research Excellence Fund and the Natural Sciences and Engineering Research Council (NSERC) of Canada.

本研究部分得到了加拿大优先研究卓越基金和加拿大自然科学与工程研究委员会（NSERC）的支持。

## References

## 参考文献

Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, et al. 2016. MS MARCO: A human generated machine reading comprehension dataset. arXiv:1611.09268.

帕亚尔·巴贾杰、丹尼尔·坎波斯、尼克·克拉斯韦尔、李·邓、高剑锋、刘晓东、兰甘·马朱姆德、安德鲁·麦克纳马拉、巴斯卡尔·米特拉、特里·阮等人。2016年。MS MARCO：一个人工生成的机器阅读理解数据集。arXiv:1611.09268。

Oren Barkan, Noam Razin, Itzik Malkiel, Ori Katz, Avi Caciularu, and Noam Koenigstein. 2020. Scalable attentive sentence-pair modeling via distilled sentence embedding. In Proc. AAAI.

奥伦·巴尔坎、诺姆·拉津、伊茨克·马尔基尔、奥里·卡茨、阿维·卡丘拉鲁和诺姆·科尼格斯泰因。2020年。通过蒸馏句子嵌入实现可扩展的注意力句子对建模。见《AAAI会议论文集》。

Wei-Cheng Chang, Felix X. Yu, Yin-Wen Chang, Yim-ing Yang, and Sanjiv Kumar. 2020. Pre-training tasks for embedding-based large-scale retrieval. In Proc. ICLR.

张维成、余菲利克斯·X、张尹文、杨一鸣和桑吉夫·库马尔。2020年。基于嵌入的大规模检索的预训练任务。见《ICLR会议论文集》。

Nick Craswell, Bhaskar Mitra, and Daniel Campos. 2019. Overview of the TREC 2019 deep learning track. In Proc. TREC.

尼克·克拉斯韦尔、巴斯卡尔·米特拉和丹尼尔·坎波斯。2019年。2019年TREC深度学习赛道概述。见《TREC会议论文集》。

Zhuyun Dai and Jamie Callan. 2020. Context-aware term weighting for first stage passage retrieval. In Proc. SIGIR, page 1533-1536.

戴竹云（音译）和杰米·卡兰。2020年。第一阶段段落检索的上下文感知词项加权。见《SIGIR会议论文集》，第1533 - 1536页。

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proc. NAACL, pages 4171-4186.

雅各布·德夫林、张明伟、肯顿·李和克里斯蒂娜·图托纳娃。2019年。BERT：用于语言理解的深度双向Transformer预训练。见《NAACL会议论文集》，第4171 - 4186页。

Luyu Gao, Zhuyun Dai, and Jamie Callan. 2020a. Understanding BERT rankers under distillation. In Proc. ICTIR, pages 149-152.

高璐宇（音译）、戴竹云（音译）和杰米·卡兰。2020a。理解蒸馏下的BERT排序器。见《ICTIR会议论文集》，第149 - 152页。

Luyu Gao, Zhuyun Dai, Zhen Fan, and Jamie Callan. 2020b. Complementing lexical retrieval with semantic residual embedding. arXiv:2004.13969.

高璐宇（音译）、戴竹云（音译）、范震（音译）和杰米·卡兰。2020b。用语义残差嵌入补充词法检索。arXiv:2004.13969。

Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pa-supat, and Ming-Wei Chang. 2020. REALM: Retrieval-augmented language model pre-training. arXiv:2002.08909.

凯尔文·古、肯顿·李、佐拉·董、帕努蓬·帕苏帕特和张明伟。2020年。REALM：检索增强的语言模型预训练。arXiv:2002.08909。

Geoffrey Hinton, Oriol Vinyals, and Jeffrey Dean. 2015. Distilling the knowledge in a neural network. In Proc. NeurIPS: Deep Learning and Representation Learning Workshop.

杰弗里·辛顿（Geoffrey Hinton）、奥里奥尔·温亚尔斯（Oriol Vinyals）和杰弗里·迪恩（Jeffrey Dean）。2015年。神经网络中的知识蒸馏。见《神经信息处理系统大会（NeurIPS）：深度学习与表征学习研讨会论文集》。

Sebastian Hofstätter, Sophia Althammer, Michael Schröder, Mete Sertkan, and Allan Hanbury. 2020. Improving efficient neural ranking models with cross-architecture knowledge distillation. arXiv:2010.02666v2.

塞巴斯蒂安·霍夫施泰特（Sebastian Hofstätter）、索菲娅·阿尔塔默（Sophia Althammer）、迈克尔·施罗德（Michael Schröder）、梅特·塞尔特坎（Mete Sertkan）和艾伦·汉伯里（Allan Hanbury）。2020年。通过跨架构知识蒸馏改进高效神经排序模型。预印本arXiv:2010.02666v2。

Sebastian Hofstätter and Allan Hanbury. 2019. Let's measure run time! Extending the IR replicability infrastructure to include performance aspects. In Proc. OSIRRC: CEUR Workshop, pages 12-16.

塞巴斯蒂安·霍夫施泰特（Sebastian Hofstätter）和艾伦·汉伯里（Allan Hanbury）。2019年。让我们来测量运行时间！将信息检索（IR）可重复性基础设施扩展到包含性能方面。见《开放源码信息检索评测竞赛（OSIRRC）：CEUR研讨会论文集》，第12 - 16页。

Sebastian Hofstätter, Sheng-Chieh Lin, Jheng-Hong Yang, Jimmy Lin, and Allan Hanbury. 2021. Efficiently teaching an effective dense retriever with balanced topic aware sampling. In Proc. SIGIR.

塞巴斯蒂安·霍夫施泰特（Sebastian Hofstätter）、林圣杰（Sheng - Chieh Lin）、杨政宏（Jheng - Hong Yang）、林吉米（Jimmy Lin）和艾伦·汉伯里（Allan Hanbury）。2021年。通过平衡主题感知采样高效训练有效的密集检索器。见《信息检索研究与发展会议（SIGIR）论文集》。

Samuel Humeau, Kurt Shuster, Marie-Anne Lachaux, and Jason Weston. 2020. Poly-encoders: Architec-

塞缪尔·休莫（Samuel Humeau）、库尔特·舒斯特（Kurt Shuster）、玛丽 - 安妮·拉肖（Marie - Anne Lachaux）和杰森·韦斯顿（Jason Weston）。2020年。多编码器：

tures and pre-training strategies for fast and accurate multi-sentence scoring. In Proc. ICLR.

用于快速准确多句子评分的架构和预训练策略。见《国际学习表征会议（ICLR）论文集》。

Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2017. Billion-scale similarity search with GPUs. arXiv:1702.08734.

杰夫·约翰逊（Jeff Johnson）、马蒂伊斯·杜泽（Matthijs Douze）和埃尔韦·热古（Hervé Jégou）。2017年。基于图形处理器（GPU）的十亿级相似度搜索。预印本arXiv:1702.08734。

Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-domain question answering. In Proc. EMNLP, pages 6769- 6781.

弗拉基米尔·卡尔普欣（Vladimir Karpukhin）、巴拉斯·奥古兹（Barlas Oğuz）、闵世文（Sewon Min）、莱德尔·吴（Ledell Wu）、谢尔盖·叶杜诺夫（Sergey Edunov）、陈丹琦（Danqi Chen）和易文涛（Wen - tau Yih）。2020年。用于开放域问答的密集段落检索。见《自然语言处理经验方法会议（EMNLP）论文集》，第6769 - 6781页。

Omar Khattab and Matei Zaharia. 2020. ColBERT: Efficient and effective passage search via contextual-ized late interaction over BERT. In Proc. SIGIR, page 39-48.

奥马尔·哈塔卜（Omar Khattab）和马泰·扎哈里亚（Matei Zaharia）。2020年。ColBERT：通过基于BERT的上下文延迟交互实现高效有效的段落搜索。见《信息检索研究与发展会议（SIGIR）论文集》，第39 - 48页。

Kenton Lee, Ming-Wei Chang, and Kristina Toutanova. 2019. Latent retrieval for weakly supervised open domain question answering. In Proc. ACL, pages 6086-6096.

肯顿·李（Kenton Lee）、张明伟（Ming - Wei Chang）和克里斯蒂娜·图托纳娃（Kristina Toutanova）。2019年。用于弱监督开放域问答的潜在检索。见《计算语言学协会年会（ACL）论文集》，第6086 - 6096页。

Jimmy Lin, Xueguang Ma, Sheng-Chieh Lin, Jheng-Hong Yang, Ronak Pradeep, and Rodrigo Nogueira. 2021. Pyserini: A Python toolkit for reproducible information retrieval research with sparse and dense representations. In Proc. SIGIR.

马学光（Xueguang Ma）、孙凯（Kai Sun）、罗纳克·普拉迪普（Ronak Pradeep）和林吉米（Jimmy Lin）。2021年。用于可重复信息检索研究的Python工具包Pyserini：稀疏和密集表示。见《信息检索研究与发展会议（SIGIR）论文集》。

Jimmy Lin, Rodrigo Nogueira, and Andrew Yates. 2020. Pretrained transformers for text ranking: BERT and beyond. arXiv:2010.06467.

林吉米（Jimmy Lin）、罗德里戈·诺盖拉（Rodrigo Nogueira）和安德鲁·耶茨（Andrew Yates）。2020年。用于文本排序的预训练Transformer：BERT及其他。预印本arXiv:2010.06467。

Ting Liu, Andrew W. Moore, Alexander Gray, and Ke Yang. 2004. An investigation of practical approximate nearest neighbor algorithms. In Proc. NeurIPS, page 825-832.

刘挺、安德鲁·W·摩尔（Andrew W. Moore）、亚历山大·格雷（Alexander Gray）和杨柯。2004年。实用近似最近邻算法研究。见《神经信息处理系统大会（NeurIPS）论文集》，第825 - 832页。

Shuqi Lu, Chenyan Xiong, Di He, Guolin Ke, Waleed Malik, Zhicheng Dou, Paul Bennett, Tieyan Liu, and Arnold Overwijk. 2021. Less is more: Pretraining a strong siamese encoder using a weak decoder. arXiv:2102.09206.

陆书琪、熊晨彦、何迪、柯国霖、瓦利德·马利克（Waleed Malik）、窦志成、保罗·贝内特（Paul Bennett）、刘铁岩和阿诺德·奥弗维克（Arnold Overwijk）。2021年。少即是多：使用弱解码器预训练强大的孪生编码器。预印本arXiv:2102.09206。

Yi Luan, Jacob Eisenstein, Kristina Toutanova, and Michael Collins. 2021. Sparse, dense, and attentional representations for text retrieval. Transactions of the Association for Computational Linguistics, 9:329-345.

栾义、雅各布·艾森斯坦（Jacob Eisenstein）、克里斯蒂娜·图托纳娃（Kristina Toutanova）和迈克尔·柯林斯（Michael Collins）。2021年。用于文本检索的稀疏、密集和注意力表示。《计算语言学协会汇刊》，9：329 - 345。

Xueguang Ma, Kai Sun, Ronak Pradeep, and Jimmy Lin. 2021. A replication study of dense passage retriever. arXiv:2104.05740.

马学光（Xueguang Ma）、孙凯（Kai Sun）、罗纳克·普拉迪普（Ronak Pradeep）和林吉米（Jimmy Lin）。2021年。密集段落检索器的复现研究。预印本arXiv:2104.05740。

Yu A. Malkov and D. A. Yashunin. 2020. Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. Transactions on Pattern Analysis and Machine Intelligence, 42(4):824-836.

尤·A·马尔科夫（Yu A. Malkov）和德米特里·A·亚舒宁（D. A. Yashunin）。2020年。使用层次可导航小世界图进行高效且稳健的近似最近邻搜索。《模式分析与机器智能汇刊》（Transactions on Pattern Analysis and Machine Intelligence），42(4):824 - 836。

Andriy Mnih and Koray Kavukcuoglu. 2013. Learning word embeddings efficiently with noise-contrastive estimation. In Proc. NIPS, pages 2265-2273.

安德里·米尼（Andriy Mnih）和科雷·卡武库奥卢（Koray Kavukcuoglu）。2013年。利用噪声对比估计高效学习词嵌入。见《神经信息处理系统大会论文集》（Proc. NIPS），第2265 - 2273页。

Rodrigo Nogueira and Kyunghyun Cho. 2019. Passage re-ranking with BERT. arXiv:1901.04085.

罗德里戈·诺盖拉（Rodrigo Nogueira）和赵京焕（Kyunghyun Cho）。2019年。使用BERT进行段落重排序。预印本：arXiv:1901.04085。

Rodrigo Nogueira and Jimmy Lin. 2019. From doc2query to docTTTTTquery.

罗德里戈·诺盖拉（Rodrigo Nogueira）和吉米·林（Jimmy Lin）。2019年。从doc2query到docTTTTTquery。

Yingqi Qu, Yuchen Ding, Jing Liu, Kai Liu, Ruiyang Ren, Wayne Xin Zhao, Daxiang Dong, Hua Wu, and Haifeng Wang. 2020. RocketQA: An optimized training approach to dense passage retrieval for open-domain question answering. arxiv:2010.08191v1.

曲英琦、丁宇辰、刘静、刘凯、任瑞阳、赵鑫、董大祥、吴华和王海峰。2020年。RocketQA：一种用于开放域问答的密集段落检索的优化训练方法。预印本：arxiv:2010.08191v1。

Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In Proc. EMNLP, pages 3982-3992.

尼尔斯·赖默斯（Nils Reimers）和伊琳娜·古列维奇（Iryna Gurevych）。2019年。Sentence - BERT：使用孪生BERT网络的句子嵌入。见《自然语言处理经验方法会议论文集》（Proc. EMNLP），第3982 - 3992页。

Yu Sun, Shuohuan Wang, Yukun Li, Shikun Feng, Hao Tian, Hua Wu, and Haifeng Wang. 2019. ERNIE 2.0: A continual pre-training framework for language understanding. arXiv:1907.12412.

孙宇、王硕欢、李雨坤、冯世坤、田浩、吴华和王海峰。2019年。ERNIE 2.0：一种用于语言理解的持续预训练框架。预印本：arXiv:1907.12412。

Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed, and Arnold Overwijk. 2021. Approximate nearest neighbor negative contrastive learning for dense text retrieval. In Proc. ICLR.

熊磊、熊晨彦、李野、邓国峰、刘佳琳、保罗·贝内特、朱奈德·艾哈迈德和阿诺德·奥弗维克。2021年。用于密集文本检索的近似最近邻负对比学习。见《国际学习表征会议论文集》（Proc. ICLR）。

Peilin Yang, Hui Fang, and Jimmy Lin. 2018. Anserini: Reproducible ranking baselines using Lucene. Journal of Data and Information Quality, 10(4):Article 16.

杨培林、方辉和吉米·林（Jimmy Lin）。2018年。Anserini：使用Lucene的可重现排序基线。《数据与信息质量杂志》（Journal of Data and Information Quality），10(4):第16篇。

Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Min Zhang, and Shaoping Ma. 2020. Learning to retrieve: How to train a dense retrieval model effectively and efficiently. arXiv:2010.10469.

詹景涛、毛佳欣、刘奕群、张敏和马少平。2020年。学习检索：如何有效且高效地训练密集检索模型。预印本：arXiv:2010.10469。