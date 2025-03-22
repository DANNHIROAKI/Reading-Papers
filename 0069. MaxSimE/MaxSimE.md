# MaxSimE: Explaining Transformer-based Semantic Similarity via Contextualized Best Matching Token Pairs

# MaxSimE：通过上下文最佳匹配Token对解释基于Transformer的语义相似度

Eduardo Brito

爱德华多·布里托

Henri Iser

亨利·伊瑟

eduardo.alfredo.brito.chacon@iais.fraunhofer.de

henri.iser@iais.fraunhofer.de

Fraunhofer Institute for Intelligent Analysis and Information Systems IAIS

弗劳恩霍夫智能分析与信息系统研究所（IAIS）

Sankt Augustin, Germany

德国圣奥古斯丁

Lamarr Institute for Machine Learning and Artificial Intelligence

拉玛尔机器学习与人工智能研究所

Sankt Augustin, Germany

德国圣奥古斯丁

## Abstract

## 摘要

Current semantic search approaches rely on black-box language models, such as BERT, which limit their interpretability and transparency. In this work, we propose MaxSimE, an explanation method for language models applied to measure semantic similarity. Our approach is inspired by the explainable-by-design ColBERT architecture and generates explanations by matching contextualized query tokens to the most similar tokens from the retrieved document according to the cosine similarity of their embeddings. Unlike existing post-hoc explanation methods, which may lack fidelity to the model and thus fail to provide trustworthy explanations in critical settings, we demonstrate that MaxSimE can generate faithful explanations under certain conditions and how it improves the interpretability of semantic search results on ranked documents from the LoTTe benchmark, showing its potential for trustworthy information retrieval.

当前的语义搜索方法依赖于黑盒语言模型，如BERT，这限制了它们的可解释性和透明度。在这项工作中，我们提出了MaxSimE，这是一种应用于测量语义相似度的语言模型解释方法。我们的方法受到了设计上可解释的ColBERT架构的启发，通过根据上下文查询Token与检索文档中最相似Token的嵌入余弦相似度进行匹配来生成解释。与现有的事后解释方法不同，这些方法可能缺乏对模型的保真度，因此在关键场景中无法提供可信的解释，我们证明了MaxSimE在某些条件下可以生成忠实的解释，以及它如何提高了LoTTe基准中排名文档的语义搜索结果的可解释性，显示了其在可信信息检索方面的潜力。

## CCS CONCEPTS

## 计算机与通信安全概念（CCS CONCEPTS）

- Information systems $\rightarrow$ Similarity measures; Query representation; Document representation; - Computing methodologies $\rightarrow$ Neural networks.

- 信息系统 $\rightarrow$ 相似度度量；查询表示；文档表示； - 计算方法 $\rightarrow$ 神经网络。

## KEYWORDS

## 关键词

explainable search, semantic similarity, ad-hoc explanations, neural models, trustworthy information retrieval

可解释搜索、语义相似度、临时解释、神经模型、可信信息检索

## ACM Reference Format:

## ACM引用格式：

Eduardo Brito and Henri Iser. 2023. MaxSimE: Explaining Transformer-based Semantic Similarity via Contextualized Best Matching Token Pairs. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '23), July 23-27, 2023, Taipei, Taiwan. ACM, New York, NY, USA, 5 pages. https://doi.org/10.1145/3539618.3592017

爱德华多·布里托和亨利·伊瑟。2023年。MaxSimE：通过上下文最佳匹配Token对解释基于Transformer的语义相似度。收录于第46届ACM信息检索研究与发展国际会议（SIGIR '23）论文集，2023年7月23 - 27日，中国台湾台北。美国纽约州纽约市ACM，共5页。https://doi.org/10.1145/3539618.3592017

## 1 INTRODUCTION

## 1 引言

Modern ranking systems often depend on pre-trained language models to compute representations for queries and documents [12]. Because of the black-box nature of the large deep neural networks they mostly rely on, these models are not suitable when the user requires some explanation to trust the system or to correct it when its output is erroneous [1]. Among the recent Transformer-based [20] approaches, ColBERT [8] introduces a late interaction mechanism to a pre-trained BERT model [5]. This additional layer is used to calculate a similarity score between a query and a document by matching each token vector representation from the query to the closest document token representations, summing them all into a global similarity score. This sum of similarity scores over query terms is similar to more standard ranking methods such as BM25 [16] and we can exploit it to generate explanations about the similarity score. Under the hood, this so-called MaxSim operation matches each query token to the most semantically similar document token within their respective contexts. Since we can compute the cosine similarity between any two token representations, we can show the matched tokens by decreasing order of similarity i.e., by decreasing contribution to the global similarity score, so that we can visualize why a retrieved document is (not) similar to the input query. Since the BERT tokens are mostly (sub)words, the matched token pairs can be interpretable terms that are found to be similar.

现代排序系统通常依赖预训练语言模型来计算查询和文档的表示 [12]。由于它们主要依赖的大型深度神经网络具有黑盒性质，当用户需要一些解释来信任系统或在系统输出错误时进行纠正时，这些模型并不适用 [1]。在最近基于Transformer的 [20] 方法中，ColBERT [8] 为预训练的BERT模型 [5] 引入了后期交互机制。这个额外的层用于通过将查询中的每个词元向量表示与最接近的文档词元表示进行匹配，计算查询和文档之间的相似度得分，并将它们全部汇总为一个全局相似度得分。查询词的相似度得分之和类似于更标准的排序方法，如BM25 [16]，我们可以利用它来生成关于相似度得分的解释。本质上，这种所谓的MaxSim操作在各自的上下文中将每个查询词元与语义最相似的文档词元进行匹配。由于我们可以计算任意两个词元表示之间的余弦相似度，我们可以按照相似度降序（即对全局相似度得分的贡献降序）显示匹配的词元，这样我们就可以直观地看到为什么检索到的文档与输入查询（不）相似。由于BERT词元大多是（子）词，匹配的词元对可以是被认为相似的可解释术语。

In this work, we provide examples of where these matches seem informative and discuss the limitations of their interpretability. Additionally, we extend this approach to more 'standard' BERT-based models and compare the resulting explanations to those obtained from ColBERTv2. Our contribution on MaxSim-based Explanations (MaxSimE) ${}^{1}$ is twofold:

在这项工作中，我们提供了这些匹配似乎具有信息价值的示例，并讨论了它们可解释性的局限性。此外，我们将这种方法扩展到更多“标准”的基于BERT的模型，并将得到的解释与从ColBERTv2获得的解释进行比较。我们基于MaxSim的解释（MaxSimE） ${}^{1}$ 的贡献有两个方面：

(1) We propose an explainability method for Transformer-based semantic similarity, whose fidelity is maximal when applied to models fine-tuned via late interaction such as ColBERTv2 [17]. Visualizing the contextualized best matching tokens can help to confirm a highly ranked document or to hint at some model failure e.g., paired tokens wrongly contributing to a high similarity score.

(1) 我们提出了一种基于Transformer的语义相似度可解释性方法，当应用于通过后期交互进行微调的模型（如ColBERTv2 [17]）时，该方法的保真度最高。可视化上下文最佳匹配词元有助于确认排名较高的文档，或暗示某些模型故障，例如，配对词元错误地对高相似度得分做出贡献。

(2) We intrinsically measure the correctness of MaxSimE taking ColBERTv2 as a proxy for the ground truth to discuss the settings where our explanations are most informative while considering their limitations as well.

(2) 我们以ColBERTv2为真实情况的代理，从本质上衡量MaxSimE的正确性，以讨论我们的解释最具信息价值的设置，同时也考虑它们的局限性。

---

<!-- Footnote -->

${}^{1}$ Source code available on https://github.com/fraunhofer-iais/MaxSimE

${}^{1}$ 源代码可在https://github.com/fraunhofer-iais/MaxSimE获取

<!-- Footnote -->

---

## 2 RELATED WORK

## 2 相关工作

From the wide spectrum of available explanation methods $\lbrack 1,3$ , 11], feature attribution aims to identify the important features or terms that contribute to a particular result. Among them, Local Interpretable Model-agnostic Explanations (LIME) [15] is a popular method that has been adapted for information retrieval tasks $\left\lbrack  {{18},{21}}\right\rbrack$ . More recent approaches focus on generating explanations that consider not only individual retrieved documents but also the context of the entire search result list to provide more coherent and diverse explanations [23]. While all these approaches provide post-hoc explanations, whose fidelity to the ranker cannot be guaranteed, we focus instead on an explainable by architecture approach. Formal et al. [6] report how BERT-based representations implicitly capture term importance and how the ColBERT fine-tuning approach amplifies this effect, improving the retrieval results. Our approach explicitly exploits this fact to generate explanations highlighting the matched terms and their contribution to the similarity score. Some frameworks focus on inspecting ranking models by evaluating on diagnostic datasets to detect global properties of the tested ranking models $\left\lbrack  {4,{10},{13}}\right\rbrack$ . They progress towards a better understanding of why contextualized word embeddings outperform traditional term-based IR methods. Our approach does not aim to analyze model behavior as a whole like them but rather explain a similarity score i.e., to provide local explanations. Calculating semantic similarity based on token embeddings is not a new idea and it has been explored to rank documents [22]. However, we do not aim to build a ranking model from the computed similarity scores but to explain existing models instead.

在广泛可用的解释方法 $\lbrack 1,3$ , 11] 中，特征归因旨在识别对特定结果有贡献的重要特征或术语。其中，局部可解释模型无关解释（LIME） [15] 是一种流行的方法，已被应用于信息检索任务 $\left\lbrack  {{18},{21}}\right\rbrack$。最近的方法侧重于生成不仅考虑单个检索文档，还考虑整个搜索结果列表上下文的解释，以提供更连贯和多样化的解释 [23]。虽然所有这些方法都提供事后解释，其对排序器的保真度无法保证，但我们转而关注一种基于架构的可解释方法。Formal等人 [6] 报告了基于BERT的表示如何隐式地捕捉术语重要性，以及ColBERT微调方法如何放大这种效果，从而改善检索结果。我们的方法明确利用这一事实来生成解释，突出显示匹配的术语及其对相似度得分的贡献。一些框架专注于通过在诊断数据集上进行评估来检查排序模型，以检测被测试排序模型的全局属性 $\left\lbrack  {4,{10},{13}}\right\rbrack$。它们有助于更好地理解为什么上下文词嵌入优于传统的基于术语的信息检索方法。我们的方法并不像它们那样旨在整体分析模型行为，而是解释相似度得分，即提供局部解释。基于词元嵌入计算语义相似度并不是一个新想法，并且已经被用于对文档进行排序 [22]。然而，我们的目标不是从计算出的相似度得分构建排序模型，而是解释现有模型。

## 3 MAXSIME

## 3 MAXSIME

MaxSimE is a method to generate local explanations for document retrieval systems using language models from which the semantic similarity between two tokens can be measured by the cosine similarity between their vector representations. Its purpose is to provide insights into why a document was retrieved given a query by highlighting the tokens in both the query and the document that contribute the most to their similarity score. We adopt the notation from Santhanam et al. [17] and define a similarity function ${S}_{q,d}$ between a query $q$ of $N$ tokens and a document $d$ of $M$ tokens as the summation of query-side MaxSim operations, namely, the maximum cosine similarity between each query token embedding and all document token embeddings (implemented as dot-products assuming normalized embeddings):

MaxSimE是一种为文档检索系统生成局部解释的方法，它使用语言模型，通过两个词元向量表示之间的余弦相似度来衡量它们的语义相似度。其目的是通过突出查询和文档中对相似度得分贡献最大的词元，深入解释为什么给定查询后会检索到某一文档。我们采用Santhanam等人[17]的符号，将包含$N$个词元的查询$q$与包含$M$个词元的文档$d$之间的相似度函数${S}_{q,d}$定义为查询端MaxSim操作的总和，即每个查询词元嵌入与所有文档词元嵌入之间的最大余弦相似度（假设嵌入已归一化，则通过点积实现）：

$$
{S}_{q,d} \mathrel{\text{:=}} \mathop{\sum }\limits_{{i = 1}}^{N}\mathop{\max }\limits_{{j = 1}}^{M}{Q}_{i} \cdot  {D}_{{d}_{j}}^{T} \tag{1}
$$

where $\mathrm{Q}$ is a matrix of $\mathrm{N}$ vectors encoding $q$ and $\mathrm{D}$ a matrix of $\mathrm{M}$ vectors encoding $d$ ,being each vector an embedding of a token.

其中$\mathrm{Q}$是一个包含$\mathrm{N}$个向量的矩阵，对$q$进行编码，$\mathrm{D}$是一个包含$\mathrm{M}$个向量的矩阵，对$d$进行编码，每个向量都是一个词元的嵌入。

We match each query token to the most similar document token (given a context) according to the MaxSim operation, as displayed in Figure 1. Formally,given a query embedding ${q}_{i}$ ,our matching function ${f}_{\text{match }}$ returns the document token embedding ${d}_{j}$ with

根据MaxSim操作，我们将每个查询词元与最相似的文档词元（给定上下文）进行匹配，如图1所示。形式上，给定一个查询嵌入${q}_{i}$，我们的匹配函数${f}_{\text{match }}$返回文档词元嵌入${d}_{j}$，其

<!-- Media -->

<!-- figureText: Query Embeddings Document Embeddings ${d}_{1}$ ${d}_{3}$ ${q}_{1}$ ${q}_{2}$ $\mathop{\max }\limits_{{j = 1}}^{M}{q}_{i} \cdot  {d}_{j}$ -->

<img src="https://cdn.noedgeai.com/0195a561-6f40-7ee8-8f9a-e8c25f80596a_1.jpg?x=941&y=239&w=681&h=456&r=0"/>

Figure 1: Visualization of the MaxSim operation. Each embedding represents a token created by the BERT tokenizer. Given a query $q$ and a document $d$ ,for a query embedding ${q}_{i}$ , MaxSim selects the closest document embedding ${d}_{j}$ . When the represented query token is an interpretable term, this is equivalent to finding the most semantically similar term appearing in $d$ ,represented by the document embedding ${d}_{j}$ .

图1：MaxSim操作的可视化。每个嵌入表示由BERT分词器创建的一个词元。给定一个查询$q$和一个文档$d$，对于一个查询嵌入${q}_{i}$，MaxSim选择最接近的文档嵌入${d}_{j}$。当所表示的查询词元是一个可解释的术语时，这相当于找到出现在$d$中语义最相似的术语，由文档嵌入${d}_{j}$表示。

<!-- Media -->

the highest dot product to ${q}_{i}$ :

与${q}_{i}$的点积最高：

$$
{f}_{\text{match }}\left( {q}_{i}\right)  \mathrel{\text{:=}} \arg \mathop{\max }\limits_{{d}_{j}}{q}_{i} \cdot  {d}_{j} \tag{2}
$$

$$
i \in  \llbracket 1..N\rrbracket ,j \in  \llbracket 1..M\rrbracket 
$$

Applying our matching function to all embeddings from a query results in a list of token pairs with the highest similarity according to the cosine similarity of their respective embeddings. These token pairs with their respective similarity scores (computed from their dot product) construct an explanation about "why" document ${d}_{j}$ was retrieved given ${q}_{i}$ as a query.

将我们的匹配函数应用于查询中的所有嵌入，会得到一个根据各自嵌入的余弦相似度具有最高相似度的词元对列表。这些词元对及其各自的相似度得分（根据它们的点积计算）构成了一个关于“为什么”给定查询${q}_{i}$时会检索到文档${d}_{j}$的解释。

## 4 EXPERIMENTS

## 4 实验

### 4.1 Data

### 4.1 数据

Our experiments are performed on the LoTTE benchmark, a collection of questions and answers sourced from StackExchange. The benchmark covers a wide range of topics, including writing, recreation, science, technology, and lifestyle [17]. To pair documents, we use ColBERTv2 to rank the documents, and we select the top-1- ranked document for each question.

我们的实验在LoTTE基准数据集上进行，该数据集是一个从StackExchange收集的问答集合。该基准涵盖了广泛的主题，包括写作、娱乐、科学、技术和生活方式[17]。为了对文档进行配对，我们使用ColBERTv2对文档进行排序，并为每个问题选择排名第一的文档。

### 4.2 Fully Faithful Explanations from ColBERT-based Models

### 4.2 基于ColBERT模型的完全忠实解释

We apply our approach first to a ColBERTv2 model to generate explanations. The first observed explanations seem to be informative from a qualitative point of view, as seen in the example from Table 1. The fidelity of the explanations is maximal because ColBERTv2 scoring is directly reliant on the sum of query side MaxSim scores, and the similarity function has been optimized through fine-tuning, thereby giving more significance to the best matching token pairs. In addition, these explanations come at no cost, since the MaxSim scores for each query token are already computed in the retrieval process. Considering that ColBERTv2 approaches state-of-the-art level according to most of the metrics from the BEIR benchmark for dense retrieval [19], we assume these explanations to be our "gold standard" for further experiments.

我们首先将我们的方法应用于ColBERTv2模型以生成解释。从定性的角度来看，最初观察到的解释似乎很有信息量，如表1中的示例所示。解释的保真度是最大的，因为ColBERTv2的评分直接依赖于查询端MaxSim得分的总和，并且相似度函数已经通过微调进行了优化，从而使最佳匹配的词元对更具重要性。此外，这些解释无需额外成本，因为每个查询词元的MaxSim得分在检索过程中已经计算过。考虑到根据BEIR基准中用于密集检索的大多数指标，ColBERTv2接近了当前的先进水平[19]，我们将这些解释作为进一步实验的“黄金标准”。

<!-- Media -->

Table 1: Matched tokens from the query "Why do kittens love packets?" and first ranked document by the pretrained ColBERTv2 model. MaxSim was performed on ColBERTv2 and S-BERT ${}_{\text{base }}$ ,sorted by descending similarity score.

表1：查询“为什么小猫喜欢包装袋？”与预训练的ColBERTv2模型排名第一的文档匹配的词元。MaxSim在ColBERTv2和S - BERT${}_{\text{base }}$上执行，按相似度得分降序排序。

<table><tr><td rowspan="2">Query Token</td><td colspan="2">ColBERTv2</td><td colspan="2">S-BERT ${}_{base}$</td></tr><tr><td>Token</td><td>Score</td><td>Token</td><td>Score</td></tr><tr><td>why</td><td>because</td><td>0.874</td><td>because</td><td>0.911</td></tr><tr><td>kitten</td><td>[D]</td><td>0.809</td><td>cats</td><td>0.891</td></tr><tr><td>##s</td><td>they</td><td>0.756</td><td>they</td><td>0.874</td></tr><tr><td>[CLS]</td><td>[CLS]</td><td>0.728</td><td>[CLS]</td><td>0.843</td></tr><tr><td>do</td><td>which</td><td>0.722</td><td>to</td><td>0.848</td></tr><tr><td>love</td><td>love</td><td>0.694</td><td>love</td><td>0.912</td></tr><tr><td>packets</td><td>boxes</td><td>0.485</td><td>dart</td><td>0.787</td></tr><tr><td>?</td><td>boxes</td><td>0.466</td><td>means</td><td>0.843</td></tr></table>

<table><tbody><tr><td rowspan="2">查询Token</td><td colspan="2">ColBERTv2</td><td colspan="2">S-BERT ${}_{base}$</td></tr><tr><td>Token</td><td>得分</td><td>Token</td><td>得分</td></tr><tr><td>为什么</td><td>因为</td><td>0.874</td><td>因为</td><td>0.911</td></tr><tr><td>小猫</td><td>[D]</td><td>0.809</td><td>猫</td><td>0.891</td></tr><tr><td>##s</td><td>它们</td><td>0.756</td><td>它们</td><td>0.874</td></tr><tr><td>[CLS]</td><td>[CLS]</td><td>0.728</td><td>[CLS]</td><td>0.843</td></tr><tr><td>做；助动词（无实义）</td><td>哪一个</td><td>0.722</td><td>到；向</td><td>0.848</td></tr><tr><td>爱</td><td>爱</td><td>0.694</td><td>爱</td><td>0.912</td></tr><tr><td>包裹</td><td>盒子</td><td>0.485</td><td>猛冲；飞镖</td><td>0.787</td></tr><tr><td>?</td><td>盒子</td><td>0.466</td><td>意味着</td><td>0.843</td></tr></tbody></table>

<!-- Media -->

### 4.3 Explanations from Other BERT-based Models

### 4.3 其他基于BERT模型的解释

We generate explanations with our approach from other BERT-based models that were not fine-tuned with a late interaction mechanism like ColBERT. We aim to confirm if these explanations are trustworthy and we thus compare the resulting explanations with those extracted from ColBERTv2 as in Section 4.2, assuming the latter as the reference. As shown in the example from Table 1, the matched tokens partially coincide with those obtained from the ColBERTv2 model although the contribution of the token pairs to the similarity score differs to a greater extent. Performance-wise, generating explanations for non-ColBERT architectures involves $N \cdot  M$ cosine distance computations (see Equation 1).

我们使用我们的方法从其他基于BERT的模型中生成解释，这些模型没有像ColBERT那样使用后期交互机制进行微调。我们旨在确认这些解释是否可信，因此我们将生成的解释与第4.2节中从ColBERTv2提取的解释进行比较，并将后者作为参考。如表1中的示例所示，匹配的Token（标记）与从ColBERTv2模型获得的Token部分重合，尽管Token对相似度得分的贡献差异较大。从性能方面来看，为非ColBERT架构生成解释涉及$N \cdot  M$余弦距离计算（见公式1）。

### 4.4 Evaluation

### 4.4 评估

Despite the absence of ground truth and user feedback, we aim to evaluate the correctness of our explanations extracted from several BERT-based models by comparing them with the ColBERTv2 explanations we generated in Section 4.2, which we take as a proxy for a "gold standard". Let $T$ be the number of correctly retrieved document tokens, $P$ the number of retrieved query/token pairs according to the gold standard,and $N$ the number of query tokens. For each query document, we evaluate the following metrics on the Top-1 document retrieved by ColBERTv2:

尽管缺乏真实标签和用户反馈，我们旨在通过将从几个基于BERT的模型中提取的解释与我们在第4.2节中生成的ColBERTv2解释进行比较，来评估这些解释的正确性，我们将ColBERTv2解释作为“黄金标准”的代理。设$T$为正确检索到的文档Token数量，$P$为根据黄金标准检索到的查询/Token对的数量，$N$为查询Token的数量。对于每个查询文档，我们在ColBERTv2检索到的Top-1文档上评估以下指标：

(1) Token precision: $\frac{T}{N}$

(1) Token精度：$\frac{T}{N}$

(2) Matching accuracy: $\frac{P}{N}$

(2) 匹配准确率：$\frac{P}{N}$

(3) Spearman's rank correlation of the matching token scores with the gold standard.

(3) 匹配Token得分与黄金标准的斯皮尔曼等级相关性。

Notice that the matching accuracy is a stricter variant of the token precision since the token precision just measures how many of the expected document tokens were retrieved (independent from the query tokens they were matched to), whereas the matching accuracy only counts the matches where the tokens are correct both from the query and the document side. The Spearman's rank correlation is intended to capture the similarity in terms of ranking query tokens.

请注意，匹配准确率是Token精度的更严格变体，因为Token精度仅衡量检索到了多少预期的文档Token（与它们匹配的查询Token无关），而匹配准确率仅计算查询和文档两侧Token都正确的匹配。斯皮尔曼等级相关性旨在捕捉查询Token排名方面的相似性。

We compare the explanations from two variants of model architectures: Cross-Encoders, which use a regression head to compute the similarity of two input texts directly; and Bi-Encoders, which produce one embedding per document either by Mean/Max Pooling token embeddings or by selecting the [CLS] token embedding so that the similarity of two texts is measured by the cosine similarity of the respective embeddings. Bi-Encoders therefore also use a late-interaction mechanism for similarity estimation whereas Cross-Encoders are fully attention-based. We analyze the effect this has on the generated explanations. For Cross-Encoders we choose the MSMARCO pretrained TinyBERT and MiniLM-L6 model, provided by the sentence-transformers library [14]. For Bi-Encoders we compare the S-BERT ${}_{\text{base }}$ model with its distilled variant DistilBERT and with the MiniLM-L6 model.

我们比较了两种模型架构变体的解释：交叉编码器（Cross-Encoders），它使用回归头直接计算两个输入文本的相似度；以及双编码器（Bi-Encoders），它通过对Token嵌入进行均值/最大池化或选择[CLS] Token嵌入为每个文档生成一个嵌入，以便通过相应嵌入的余弦相似度来衡量两个文本的相似度。因此，双编码器也使用后期交互机制进行相似度估计，而交叉编码器则完全基于注意力机制。我们分析了这对生成的解释的影响。对于交叉编码器，我们选择了由sentence-transformers库[14]提供的MSMARCO预训练的TinyBERT和MiniLM-L6模型。对于双编码器，我们将S-BERT ${}_{\text{base }}$模型与其蒸馏变体DistilBERT以及MiniLM-L6模型进行比较。

### 4.5 Results

### 4.5 结果

We first analyze the explanations generated by both ColBERTv2 and the Bi-Encoder S-BERT base. Table 1 shows token matching pairs of both models. Qualitatively, we can observe that both explanations match similar document tokens to the query. Partially these matches coincide between the two models. From Figure 2 we can observe a noticeable difference in absolute score values, especially in the ranking of the matching token pairs. In comparison, S-BERT base yields higher scores for query tokens ranked lower by ColBERTv2. Furthermore, the score values produced by ColBERTv2 exhibit a greater degree of variance, especially for these lower-ranked tokens. We assume that this is due to the fine-tuning of ColBERTv2 token representations with the MaxSim late-interaction mechanism, which forces the model to also perform a fine-grained ranking on the token level.

我们首先分析了ColBERTv2和双编码器S-BERT基础模型生成的解释。表1显示了两个模型的Token匹配对。从定性角度来看，我们可以观察到两种解释都将相似的文档Token与查询进行匹配。部分匹配在两个模型之间是一致的。从图2中我们可以观察到绝对得分值存在明显差异，特别是在匹配Token对的排名方面。相比之下，S-BERT基础模型为ColBERTv2排名较低的查询Token给出了更高的分数。此外，ColBERTv2产生的得分值具有更大的方差，特别是对于这些排名较低的Token。我们认为这是由于使用MaxSim后期交互机制对ColBERTv2的Token表示进行了微调，这迫使模型在Token级别上也进行细粒度的排名。

When evaluating the correctness of our explanations on non-ColBERT models, we observe that token precision is generally high across most models (as displayed in Table 2). All metrics have high variance, which suggests that the quality of the explanations is highly dependent on the query sentence. Especially the matching accuracy and ranking of the tokens are inconsistent throughout the dataset. For the smaller model MiniLM-L6, we see that the Bi-Encoder variant provides explanations closer to our gold standard. This could be explained by the fact that the late-interaction mechanism used in sentence transformers (especially with mean pooling) is more similar to the MaxSim operation than the regression head in Cross-Encoders.

在评估非ColBERT模型解释的正确性时，我们观察到大多数模型的Token精度普遍较高（如表2所示）。所有指标的方差都很大，这表明解释的质量高度依赖于查询语句。特别是Token的匹配准确率和排名在整个数据集中并不一致。对于较小的模型MiniLM-L6，我们发现双编码器变体提供的解释更接近我们的黄金标准。这可以解释为，句子转换器中使用的后期交互机制（特别是均值池化）比交叉编码器中的回归头更类似于MaxSim操作。

### 4.6 Discussion

### 4.6 讨论

We observe that, although the non-ColBERT models were not trained using the MaxSim operations, the generated explanations largely align with those of ColBERTv2, as demonstrated by the example in Table 1. The similarity between the explanations suggests that they similarly capture term importance, in line with previous white box analysis on ColBERT [6]. Considering that the ranking performance does differ, we guess that the different similarity value distributions assigned to the matches have a noticeable impact on the global similarity score and thus on the ranked documents. The distributions in Figure 2 illustrate how ColBERTv2 weights with significantly higher similarity scores for the most semantic relevant terms than the rest of the tokens, whereas the similarity score difference among embeddings coming from ${\mathrm{{BERT}}}_{\text{base }}$ is clearly less differentiated. Despite this, the high token precision (displayed in Table 2) implies that non-ColBERT models frequently match the same tokens as ColBERT.

我们观察到，尽管非ColBERT模型并非使用最大相似度（MaxSim）操作进行训练，但生成的解释在很大程度上与ColBERTv2的解释一致，如表1中的示例所示。这些解释之间的相似性表明，它们以相似的方式捕捉了词项的重要性，这与之前对ColBERT的白盒分析结果一致[6]。考虑到排序性能确实存在差异，我们推测，分配给匹配项的不同相似度值分布会对全局相似度得分产生显著影响，进而影响文档的排序。图2中的分布情况显示，与其他词元相比，ColBERTv2对语义最相关的词项赋予了显著更高的相似度得分，而来自${\mathrm{{BERT}}}_{\text{base }}$的嵌入向量之间的相似度得分差异明显较小。尽管如此，高词元精度（如表2所示）表明，非ColBERT模型经常与ColBERT匹配相同的词元。

<!-- Media -->

<!-- figureText: 1.0 6 Top 8 ranked query tokens ColBERTv2 0.0 ${\mathrm{S - {BERT}}}_{\text{base }}$ 3 -->

<img src="https://cdn.noedgeai.com/0195a561-6f40-7ee8-8f9a-e8c25f80596a_3.jpg?x=165&y=250&w=689&h=492&r=0"/>

Figure 2: Cosine similarity distribution of the top 8 ranked query tokens for each query from the LoTTE dataset.

图2：来自LoTTE数据集的每个查询中排名前8的查询词元的余弦相似度分布。

Table 2: Similarity of explanations from BERT-based models to our ColBERTv2 gold standard measured by token precision (TP), match accuracy (MA), and Spearman's rank correlation (SR).

表2：基于BERT的模型的解释与我们的ColBERTv2黄金标准之间的相似度，通过词元精度（TP）、匹配准确率（MA）和斯皮尔曼等级相关性（SR）来衡量。

<table><tr><td>Model</td><td>TP</td><td>$\mathbf{{MA}}$</td><td>SR</td></tr><tr><td colspan="4">Bi-Encoders</td></tr><tr><td>S-BERTbase</td><td>${0.730} \pm  {0.153}$</td><td>$\mathbf{{0.471}} \pm  {0.213}$</td><td>${0.427} \pm  {0.380}$</td></tr><tr><td>DistilBERT</td><td>0.740 ± 0.163</td><td>${0.444} \pm  {0.212}$</td><td>${0.349} \pm  {0.386}$</td></tr><tr><td>MiniLM-L6</td><td>${0.664} \pm  {0.149}$</td><td>${0.411} \pm  {0.200}$</td><td>$\mathbf{{0.473}} \pm  {0.376}$</td></tr><tr><td colspan="4">Cross-Encoders</td></tr><tr><td>TinyBERT</td><td>$\mathbf{{0.749}} \pm  {0.158}$</td><td>${0.446} \pm  {0.204}$</td><td>${0.391} \pm  {0.343}$</td></tr><tr><td>MiniLM-L6</td><td>${0.387} \pm  {0.233}$</td><td>${0.307} \pm  {0.192}$</td><td>${0.270} \pm  {0.284}$</td></tr></table>

<table><tbody><tr><td>模型</td><td>真阳性（True Positive）</td><td>$\mathbf{{MA}}$</td><td>召回率（Success Rate）</td></tr><tr><td colspan="4">双编码器（Bi-Encoders）</td></tr><tr><td>基础语义BERT模型（S-BERTbase）</td><td>${0.730} \pm  {0.153}$</td><td>$\mathbf{{0.471}} \pm  {0.213}$</td><td>${0.427} \pm  {0.380}$</td></tr><tr><td>蒸馏BERT模型（DistilBERT）</td><td>0.740 ± 0.163</td><td>${0.444} \pm  {0.212}$</td><td>${0.349} \pm  {0.386}$</td></tr><tr><td>小型语言模型L6层（MiniLM-L6）</td><td>${0.664} \pm  {0.149}$</td><td>${0.411} \pm  {0.200}$</td><td>$\mathbf{{0.473}} \pm  {0.376}$</td></tr><tr><td colspan="4">交叉编码器（Cross-Encoders）</td></tr><tr><td>微型BERT模型（TinyBERT）</td><td>$\mathbf{{0.749}} \pm  {0.158}$</td><td>${0.446} \pm  {0.204}$</td><td>${0.391} \pm  {0.343}$</td></tr><tr><td>小型语言模型L6层（MiniLM-L6）</td><td>${0.387} \pm  {0.233}$</td><td>${0.307} \pm  {0.192}$</td><td>${0.270} \pm  {0.284}$</td></tr></tbody></table>

<!-- Media -->

Although we demonstrated how we can generate meaningful explanations for both ColBERT and other BERT-based models using the MaxSim operation, we acknowledge two main limitations of our approach: the limited faithfulness to the model for non-ColBERT architectures and the limited interpretability of some explanations because of the contribution of the [MASK] tokens to the similarity score.

尽管我们展示了如何使用最大相似度（MaxSim）操作，为ColBERT和其他基于BERT的模型生成有意义的解释，但我们也承认我们的方法存在两个主要局限性：对于非ColBERT架构的模型，解释与模型的忠实度有限；由于[MASK]标记对相似度得分的影响，一些解释的可解释性有限。

First, our explanations from non-late-interaction-based models i.e., Bi- and Cross-encoders [14], cannot guarantee faithfulness to their respective ranking models because their computed similarity usually comes from either a regression head or from the cosine similarity of [CLS] or mean pooled embeddings. Although Cross-Encoder models may achieve better evaluation scores, their computational cost is much higher, becoming impractical for most setups. Hence, we favor late interaction models for ranking not only because of their efficiency on ranking tasks but also because we can extract fully faithful explanations from the underlying language model.

首先，我们从非后期交互模型（即双编码器和交叉编码器 [14]）中得到的解释，无法保证与各自的排序模型的忠实度，因为它们计算的相似度通常来自回归头，或者来自[CLS]标记或平均池化嵌入的余弦相似度。尽管交叉编码器模型可能获得更好的评估分数，但它们的计算成本要高得多，对于大多数设置来说不切实际。因此，我们更倾向于使用后期交互模型进行排序，不仅因为它们在排序任务上的效率，还因为我们可以从底层语言模型中提取完全忠实的解释。

Second, Khattab and Zaharia [8] use [MASK] tokens within the ColBERTv2 model for query expansion. These non-interpretable tokens are also included in the late-interaction scoring mechanism, leading to best-matching token pairs that cannot be explained in a meaningful way. Depending on the length of the query, these [MASK] tokens make up for up to ${62}\%$ of the final score of the retrieved document. Nonetheless, Lassance et al. [9] show that these special tokens can be safely removed without affecting model performance in a significant way.

其次，Khattab和Zaharia [8] 在ColBERTv2模型中使用[MASK]标记进行查询扩展。这些不可解释的标记也包含在后期交互评分机制中，导致最佳匹配的标记对无法以有意义的方式进行解释。根据查询的长度，这些[MASK]标记在检索到的文档的最终得分中占比高达${62}\%$。尽管如此，Lassance等人 [9] 表明，可以安全地移除这些特殊标记，而不会对模型性能产生重大影响。

Finally, we could only evaluate the correctness of the explanations extracted from the different models by comparing them to our ColBERTv2 gold standard, which we consider confirmed when they correlate but we cannot discard otherwise. Other explainability aspects such as plausibility [7] are yet to be assessed as well. Despite the limitations, we find our first exploratory results promising and we hope to motivate more work towards trustworthy information retrieval.

最后，我们只能通过将从不同模型中提取的解释与我们的ColBERTv2黄金标准进行比较，来评估这些解释的正确性。当它们相关时，我们认为解释是正确的，但反之我们也不能排除其错误的可能性。其他可解释性方面，如合理性 [7]，也有待评估。尽管存在这些局限性，我们认为我们的初步探索性结果很有前景，并希望能推动更多关于可信信息检索的研究。

## 5 CONCLUSION AND FUTURE WORK

## 5 结论与未来工作

We leveraged the MaxSim operation from the ColBERT approach to generate explanations for the documents retrieved by the ranking system, based on the most relevant document tokens that match those of the query. We also demonstrated that our method can be applied to other BERT-based models, although we cannot guarantee its fidelity for those models. The correlation between the explanations generated by the different models confirms that our proposed method can provide insights into the underlying model, and can be used as a proxy to evaluate explanation correctness. Our presented method enables "explanations for free" i.e., without needing to learn any explanation model, from similarity functions constructed upon BERT-based language models. Our proposed approach may have applications beyond information retrieval e.g., text classification use cases where unfaithful explanations from black-box models are not acceptable and where a similarity-based classifier can be used without a dramatic performance loss compared to the best-performing black-box deep learning model [2]; or even less related areas where Transformer-based models can deal with a concept of semantic similarity such as computer vision [24]. In future work, we aim to systematically compare the ranking performance of different BERT-based models with our evaluation results, including additional evaluation criteria and benchmark datasets where Col-BERTv2 was not fine-tuned. From a more applied perspective, we also plan to apply our approach to domain-specific settings e.g., information retrieval on legal texts to support lawyers finding previous similar legal cases when facing a new one, which is an opportunity to assess the plausibility of our explanations.

我们利用ColBERT方法中的最大相似度（MaxSim）操作，根据与查询最相关的文档标记，为排序系统检索到的文档生成解释。我们还证明了我们的方法可以应用于其他基于BERT的模型，尽管我们不能保证该方法对这些模型的忠实度。不同模型生成的解释之间的相关性证实，我们提出的方法可以深入了解底层模型，并可作为评估解释正确性的代理。我们提出的方法能够“免费获得解释”，即无需学习任何解释模型，就可以从基于BERT的语言模型构建的相似度函数中获得解释。我们提出的方法可能在信息检索之外有应用，例如在文本分类场景中，黑盒模型的不可信解释是不可接受的，并且与性能最佳的黑盒深度学习模型 [2] 相比，基于相似度的分类器可以在性能损失不大的情况下使用；甚至在相关性较低的领域，如计算机视觉 [24]，基于Transformer的模型可以处理语义相似度的概念。在未来的工作中，我们旨在将不同基于BERT的模型的排序性能与我们的评估结果进行系统比较，包括额外的评估标准和未对ColBERTv2进行微调的基准数据集。从更实际的应用角度来看，我们还计划将我们的方法应用于特定领域的设置，例如对法律文本进行信息检索，以支持律师在面对新的法律案件时查找以前类似的法律案例，这是评估我们解释合理性的一个机会。

## ACKNOWLEDGMENTS

## 致谢

This research has been funded by the Federal Ministry of Education and Research of Germany and the state of North-Rhine Westphalia as part of the Lamarr Institute for Machine Learning and Artificial Intelligence. We thank Christian Bauckhage, Katharina Beckh, and Stefan Rüping for their feedback prior to our paper submission.

这项研究由德国联邦教育与研究部以及北莱茵 - 威斯特法伦州资助，是拉马尔机器学习与人工智能研究所的一部分。我们感谢Christian Bauckhage、Katharina Beckh和Stefan Rüping在我们提交论文之前提供的反馈。

## REFERENCES

## 参考文献

[1] Katharina Beckh, Sebastian Müller, Matthias Jakobs, Vanessa Toborek, Hanx-iao Tan, Raphael Fischer, Pascal Welke, Sebastian Houben, and Laura von Rue-den. 2023. Harnessing Prior Knowledge for Explainable Machine Learning: An Overview. In First IEEE Conference on Secure and Trustworthy Machine Learning. https://openreview.net/forum?id=1KE7TlU4bOt

[1] Katharina Beckh、Sebastian Müller、Matthias Jakobs、Vanessa Toborek、Hanx - iao Tan、Raphael Fischer、Pascal Welke、Sebastian Houben和Laura von Rue - den。2023年。利用先验知识进行可解释机器学习：综述。见第一届IEEE安全可信机器学习会议。https://openreview.net/forum?id=1KE7TlU4bOt

[2] Eduardo Brito, Vishwani Gupta, Eric Hahn, and Sven Giesselbach. 2022. Assessing the Performance Gain on Retail Article Categorization at the Expense of Explainability and Resource Efficiency. In KI 2022: Advances in Artificial Intelligence,Ralph Bergmann,Lukas Malburg,Stephanie C. Rodermund,and Ingo J. Timm (Eds.). Springer International Publishing, Cham, 45-52.

[2] Eduardo Brito、Vishwani Gupta、Eric Hahn和Sven Giesselbach。2022年。以可解释性和资源效率为代价评估零售商品分类的性能提升。见《KI 2022：人工智能进展》，Ralph Bergmann、Lukas Malburg、Stephanie C. Rodermund和Ingo J. Timm（编）。施普林格国际出版公司，尚姆，第45 - 52页。

[3] Nadia Burkart and Marco F Huber. 2021. A survey on the explainability of supervised machine learning. Journal of Artificial Intelligence Research 70 (2021), 245-317.

[3] Nadia Burkart和Marco F Huber。2021年。监督机器学习可解释性综述。《人工智能研究杂志》70（2021），第245 - 317页。

[4] Arthur Câmara and Claudia Hauff. 2020. Diagnosing BERT with Retrieval Heuristics. In Advances in Information Retrieval, Joemon M. Jose, Emine Yilmaz, João Magalhães, Pablo Castells, Nicola Ferro, Mário J. Silva, and Flávio Martins (Eds.). Springer International Publishing, Cham, 605-618.

[4] 亚瑟·卡马拉（Arthur Câmara）和克劳迪娅·豪夫（Claudia Hauff）。2020年。用检索启发式方法诊断BERT。见《信息检索进展》，若埃蒙·M·若泽（Joemon M. Jose）、埃米内·伊尔马兹（Emine Yilmaz）、若昂·马加良斯（João Magalhães）、巴勃罗·卡斯特尔斯（Pablo Castells）、尼古拉·费罗（Nicola Ferro）、马里奥·J·席尔瓦（Mário J. Silva）和弗拉维奥·马丁斯（Flávio Martins）（编）。施普林格国际出版公司，尚姆，第605 - 618页。

[5] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). Association for Computational Linguistics, Minneapolis, Minnesota, 4171-4186. https://doi.org/10.18653/v1/N19-1423

[5] 雅各布·德夫林（Jacob Devlin）、张明伟（Ming - Wei Chang）、肯顿·李（Kenton Lee）和克里斯蒂娜·图托纳娃（Kristina Toutanova）。2019年。BERT：用于语言理解的深度双向Transformer预训练。见《2019年北美计算语言学协会会议：人类语言技术》论文集，第1卷（长论文和短论文）。计算语言学协会，明尼苏达州明尼阿波利斯，第4171 - 4186页。https://doi.org/10.18653/v1/N19 - 1423

[6] Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant. 2021. A White Box Analysis of ColBERT. In Advances in Information Retrieval, Djoerd Hiemstra, Marie-Francine Moens, Josiane Mothe, Raffaele Perego, Martin Potthast, and Fabrizio Sebastiani (Eds.). Springer International Publishing, Cham, 257-263.

[6] 蒂博·福尔马尔（Thibault Formal）、本杰明·皮沃瓦尔斯基（Benjamin Piwowarski）和斯特凡·克兰尚（Stéphane Clinchant）。2021年。ColBERT的白盒分析。见《信息检索进展》，乔尔德·希姆斯特拉（Djoerd Hiemstra）、玛丽 - 弗朗辛·莫恩斯（Marie - Francine Moens）、乔西安·莫特（Josiane Mothe）、拉斐尔·佩雷戈（Raffaele Perego）、马丁·波塔斯塔（Martin Potthast）和法布里齐奥·塞巴斯蒂亚尼（Fabrizio Sebastiani）（编）。施普林格国际出版公司，尚姆，第257 - 263页。

[7] Alon Jacovi and Yoav Goldberg. 2020. Towards Faithfully Interpretable NLP Systems: How should we define and evaluate faithfulness? https://doi.org/10.48550/ARXIV.2004.03685

[7] 阿隆·雅科维（Alon Jacovi）和约阿夫·戈德堡（Yoav Goldberg）。2020年。迈向可忠实解释的自然语言处理系统：我们应该如何定义和评估忠实性？https://doi.org/10.48550/ARXIV.2004.03685

[8] Omar Khattab and Matei Zaharia. 2020. ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (Virtual Event, China) (SIGIR '20). Association for Computing Machinery, New York, NY, USA, 39-48. https://doi.org/10.1145/3397271.3401075

[8] 奥马尔·哈塔卜（Omar Khattab）和马特·扎哈里亚（Matei Zaharia）。2020年。ColBERT：通过基于BERT的上下文后期交互实现高效有效的段落搜索。见《第43届ACM SIGIR国际信息检索研究与发展会议论文集》（线上会议，中国）（SIGIR '20）。美国计算机协会，美国纽约州纽约市，第39 - 48页。https://doi.org/10.1145/3397271.3401075

[9] Carlos Lassance, Maroua Maachou, Joohee Park, and Stéphane Clinchant. 2021. A Study on Token Pruning for ColBERT. https://doi.org/10.48550/arXiv.2112.06540 arXiv:2112.06540 [cs].

[9] 卡洛斯·拉桑斯（Carlos Lassance）、马鲁阿·马乔（Maroua Maachou）、朴珠熙（Joohee Park）和斯特凡·克兰尚（Stéphane Clinchant）。2021年。ColBERT的Token剪枝研究。https://doi.org/10.48550/arXiv.2112.06540 arXiv:2112.06540 [计算机科学]。

[10] Sean MacAvaney, Sergey Feldman, Nazli Goharian, Doug Downey, and Arman Cohan. 2022. ABNIRML: Analyzing the Behavior of Neural IR Models. Transactions of the Association for Computational Linguistics 10 (2022), 224-239. https://doi.org/10.1162/tacl_a_00457

[10] 肖恩·麦卡瓦尼（Sean MacAvaney）、谢尔盖·费尔德曼（Sergey Feldman）、纳兹利·戈哈里安（Nazli Goharian）、道格·唐尼（Doug Downey）和阿尔曼·科汉（Arman Cohan）。2022年。ABNIRML：分析神经信息检索模型的行为。《计算语言学协会汇刊》10（2022年），第224 - 239页。https://doi.org/10.1162/tacl_a_00457

[11] Christoph Molnar. 2020. Interpretable machine learning. https://christophm.github.io/interpretable-ml-book/

[11] 克里斯托夫·莫尔纳尔（Christoph Molnar）。2020年。可解释机器学习。https://christophm.github.io/interpretable - ml - book/

[12] Gerhard Paaß and Sven Giesselbach. 2023. Foundation Models for Natural Language Processing-Pre-trained Language Models Integrating Media. Springer Nature 2023. https://arxiv.org/abs/2302.08575

[12] 格哈德·帕斯（Gerhard Paaß）和斯文·吉塞尔巴赫（Sven Giesselbach）。2023年。用于自然语言处理的基础模型——集成媒体的预训练语言模型。施普林格自然出版集团2023年。https://arxiv.org/abs/2302.08575

[13] David Rau and Jaap Kamps. 2022. How Different Are Pre-Trained Transformers For Text Ranking?. In Advances in Information Retrieval: 44th European Conference

[13] 大卫·劳（David Rau）和亚普·坎普斯（Jaap Kamps）。2022年。用于文本排序的预训练Transformer有何不同？见《信息检索进展：第44届欧洲信息检索研究会议》

on IR Research, ECIR 2022, Stavanger, Norway, April 10-14, 2022, Proceedings, Part II (Stavanger, Norway). Springer-Verlag, Berlin, Heidelberg, 207-214. https: //doi.org/10.1007/978-3-030-99739-7_24

，ECIR 2022，挪威斯塔万格，2022年4月10 - 14日，会议录，第二部分（挪威斯塔万格）。施普林格出版社，柏林，海德堡，第207 - 214页。https://doi.org/10.1007/978 - 3 - 030 - 99739 - 7_24

[14] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). Association for Computational Linguistics, Hong Kong, China, 3982-3992. https://doi.org/10.18653/v1/D19-1410

[14] 尼尔斯·赖默斯（Nils Reimers）和伊琳娜·古列维奇（Iryna Gurevych）。2019年。Sentence - BERT：使用孪生BERT网络的句子嵌入。见《2019年自然语言处理经验方法会议和第9届自然语言处理国际联合会议论文集》（EMNLP - IJCNLP）。计算语言学协会，中国香港，第3982 - 3992页。https://doi.org/10.18653/v1/D19 - 1410

[15] Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. 2016. Model-Agnostic Interpretability of Machine Learning. https://doi.org/10.48550/ARXIV.1606.05386

[15] 马尔科·图利奥·里贝罗（Marco Tulio Ribeiro）、萨米尔·辛格（Sameer Singh）和卡洛斯·盖斯特林（Carlos Guestrin）。2016年。机器学习的模型无关可解释性。https://doi.org/10.48550/ARXIV.1606.05386

[16] Stephen Robertson, Hugo Zaragoza, et al. 2009. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends ${}^{\circledR }$ in Information Retrieval 3, 4 (2009), 333-389.

[16] 斯蒂芬·罗伯逊（Stephen Robertson）、雨果·萨拉戈萨（Hugo Zaragoza）等。2009年。概率相关性框架：BM25及超越。《信息检索基础与趋势》${}^{\circledR }$ 第3卷，第4期（2009年），第333 - 389页。

[17] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. 2022. ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. Association for Computational Linguistics, Seattle, United States, 3715-3734. https://doi.org/10.18653/v1/2022.naacl-main.272

[17] 凯沙夫·桑塔南姆（Keshav Santhanam）、奥马尔·卡塔布（Omar Khattab）、乔恩·萨德 - 法尔孔（Jon Saad - Falcon）、克里斯托弗·波茨（Christopher Potts）和马泰·扎哈里亚（Matei Zaharia）。2022 年。ColBERTv2：通过轻量级后期交互实现高效检索。收录于《2022 年北美计算语言学协会人类语言技术会议论文集》。美国计算语言学协会，西雅图，美国，3715 - 3734。https://doi.org/10.18653/v1/2022.naacl - main.272

[18] Jaspreet Singh and Avishek Anand. 2019. EXS: Explainable Search Using Local Model Agnostic Interpretability. In Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining. ACM, Melbourne VIC Australia, 770-773. https://doi.org/10.1145/3289600.3290620

[18] 贾斯普里特·辛格（Jaspreet Singh）和阿维谢克·阿南德（Avishek Anand）。2019 年。EXS：使用局部模型无关可解释性的可解释搜索。收录于《第十二届 ACM 网络搜索与数据挖掘国际会议论文集》。美国计算机协会，澳大利亚维多利亚州墨尔本，770 - 773。https://doi.org/10.1145/3289600.3290620

[19] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021. BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2). https://openreview.net/forum?id=wCu6T5xFjeJ

[19] 南丹·塔库尔（Nandan Thakur）、尼尔斯·赖默斯（Nils Reimers）、安德里亚斯·吕克莱（Andreas Rücklé）、阿比舍克·斯里瓦斯塔瓦（Abhishek Srivastava）和伊琳娜·古列维奇（Iryna Gurevych）。2021 年。BEIR：信息检索模型零样本评估的异构基准。收录于《第三十五届神经信息处理系统数据集与基准会议（第二轮）》。https://openreview.net/forum?id=wCu6T5xFjeJ

[20] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems 30 (2017).

[20] 阿什什·瓦斯瓦尼（Ashish Vaswani）、诺姆·沙泽尔（Noam Shazeer）、尼基·帕尔马尔（Niki Parmar）、雅各布·乌斯库赖特（Jakob Uszkoreit）、利昂·琼斯（Llion Jones）、艾丹·N·戈麦斯（Aidan N Gomez）、卢卡斯·凯泽（Łukasz Kaiser）和伊利亚·波洛苏金（Illia Polosukhin）。2017 年。注意力就是你所需要的一切。《神经信息处理系统进展》30（2017 年）。

[21] Manisha Verma and Debasis Ganguly. 2019. LIRME: Locally Interpretable Ranking Model Explanation. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval (Paris, France) (SIGIR'19). Association for Computing Machinery, New York, NY, USA, 1281-1284. https: //doi.org/10.1145/3331184.3331377

[21] 玛尼莎·维尔马（Manisha Verma）和德巴西斯·冈古利（Debasis Ganguly）。2019 年。LIRME：局部可解释排序模型解释。收录于《第 42 届 ACM SIGIR 信息检索研究与发展国际会议论文集（法国巴黎）（SIGIR'19）》。美国计算机协会，美国纽约州纽约市，1281 - 1284。https: //doi.org/10.1145/3331184.3331377

[22] Chenyan Xiong, Zhuyun Dai, Jamie Callan, Zhiyuan Liu, and Russell Power. 2017. End-to-End Neural Ad-Hoc Ranking with Kernel Pooling. In Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval (Shinjuku, Tokyo, Japan) (SIGIR '17). Association for Computing Machinery, New York, NY, USA, 55-64. https://doi.org/10.1145/ 3077136.3080809

[22] 熊晨彦（Chenyan Xiong）、戴珠云（Zhuyun Dai）、杰米·卡伦（Jamie Callan）、刘志远（Zhiyuan Liu）和拉塞尔·鲍尔（Russell Power）。2017 年。基于核池化的端到端神经即席排序。收录于《第 40 届 ACM SIGIR 信息检索研究与发展国际会议论文集（日本东京新宿）（SIGIR '17）》。美国计算机协会，美国纽约州纽约市，55 - 64。https://doi.org/10.1145/ 3077136.3080809

[23] Puxuan Yu, Razieh Rahimi, and James Allan. 2022. Towards Explainable Search Results: A Listwise Explanation Generator. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (Madrid, Spain) (SIGIR '22). Association for Computing Machinery, New York, NY, USA, 669-680. https://doi.org/10.1145/3477495.3532067

[23] 余普轩（Puxuan Yu）、拉齐耶·拉希米（Razieh Rahimi）和詹姆斯·艾伦（James Allan）。2022 年。迈向可解释搜索结果：一种列表式解释生成器。收录于《第 45 届 ACM SIGIR 信息检索研究与发展国际会议论文集（西班牙马德里）（SIGIR '22）》。美国计算机协会，美国纽约州纽约市，669 - 680。https://doi.org/10.1145/3477495.3532067

[24] Chao Zhang, Stephan Liwicki, and Roberto Cipolla. 2022. Beyond the CLS Token: Image Reranking using Pretrained Vision Transformers. In 33rd British Machine Vision Conference 2022, BMVC 2022, London, UK, November 21-24, 2022. BMVA Press. https://bmvc2022.mpi-inf.mpg.de/0080.pdf

[24] 张超（Chao Zhang）、斯蒂芬·利维茨基（Stephan Liwicki）和罗伯托·奇波拉（Roberto Cipolla）。2022 年。超越 CLS 标记（Token）：使用预训练视觉变换器进行图像重排序。收录于《2022 年第 33 届英国机器视觉会议（BMVC 2022）》，英国伦敦，2022 年 11 月 21 - 24 日。英国机器视觉协会出版社。https://bmvc2022.mpi - inf.mpg.de/0080.pdf