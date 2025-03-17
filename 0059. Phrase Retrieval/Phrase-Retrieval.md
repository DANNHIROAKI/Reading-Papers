# Phrase Retrieval Learns Passage Retrieval, Too

# 短语检索也能学习段落检索

Jinhyuk Lee ${}^{1,2 * }$ Alexander Wettig ${}^{1}$ Danqi Chen ${}^{1}$

李镇赫（Jinhyuk Lee） ${}^{1,2 * }$ 亚历山大·韦蒂格（Alexander Wettig） ${}^{1}$ 陈丹琦（Danqi Chen） ${}^{1}$

Department of Computer Science,Princeton University ${}^{1}$

普林斯顿大学（Princeton University）计算机科学系 ${}^{1}$

Department of Computer Science and Engineering, Korea University ${}^{2}$

韩国大学（Korea University）计算机科学与工程系 ${}^{2}$

\{jinhyuklee,awettig,danqic\}@cs.princeton.edu

\{jinhyuklee,awettig,danqic\}@cs.princeton.edu

## Abstract

## 摘要

Dense retrieval methods have shown great promise over sparse retrieval methods in a range of NLP problems. Among them, dense phrase retrieval-the most fine-grained retrieval unit-is appealing because phrases can be directly used as the output for question answering and slot filling tasks. ${}^{1}$ In this work, we follow the intuition that retrieving phrases naturally entails retrieving larger text blocks and study whether phrase retrieval can serve as the basis for coarse-level retrieval including passages and documents. We first observe that a dense phrase-retrieval system, without any retraining, already achieves better passage retrieval accuracy ( $+ 3 - 5\%$ in top-5 accuracy) compared to passage retrievers, which also helps achieve superior end-to-end QA performance with fewer passages. Then, we provide an interpretation for why phrase-level supervision helps learn better fine-grained entailment compared to passage-level supervision, and also show that phrase retrieval can be improved to achieve competitive performance in document-retrieval tasks such as entity linking and knowledge-grounded dialogue. Finally, we demonstrate how phrase filtering and vector quantization can reduce the size of our index by $4 - {10x}$ ,making dense phrase retrieval a practical and versatile solution in multi-granularity retrieval. ${}^{2}$

在一系列自然语言处理（NLP）问题中，密集检索方法相较于稀疏检索方法展现出了巨大的潜力。其中，密集短语检索（最细粒度的检索单元）颇具吸引力，因为短语可直接用作问答和槽填充任务的输出。${}^{1}$ 在这项工作中，我们基于这样一种直觉：检索短语自然会涉及检索更大的文本块，并研究短语检索是否可作为包括段落和文档的粗粒度检索的基础。我们首先观察到，一个密集短语检索系统在无需任何再训练的情况下，与段落检索器相比，已经实现了更高的段落检索准确率（前5准确率达到 $+ 3 - 5\%$），这也有助于在使用更少段落的情况下实现更优的端到端问答性能。然后，我们解释了为什么与段落级监督相比，短语级监督有助于学习更好的细粒度蕴含关系，并且还表明，在诸如实体链接和基于知识的对话等文档检索任务中，短语检索可以得到改进以实现具有竞争力的性能。最后，我们展示了短语过滤和向量量化如何将我们的索引大小缩小 $4 - {10x}$，使密集短语检索成为多粒度检索中一种实用且通用的解决方案。${}^{2}$

## 1 Introduction

## 1 引言

Dense retrieval aims to retrieve relevant contexts from a large corpus, by learning dense representations of queries and text segments. Recently, dense retrieval of passages (Lee et al., 2019; Karpukhin et al., 2020; Xiong et al., 2021) has been shown to outperform traditional sparse retrieval methods such as TF-IDF and BM25 in a range of knowledge-intensive NLP tasks (Petroni et al., 2021), including open-domain question answering (QA) (Chen et al., 2017), entity linking (Wu et al., 2020), and knowledge-grounded dialogue (Dinan et al., 2019).

密集检索旨在通过学习查询和文本片段的密集表示，从大型语料库中检索相关上下文。最近，在一系列知识密集型自然语言处理（NLP）任务（彼得罗尼等人，2021）中，包括开放领域问答（QA）（陈等人，2017）、实体链接（吴等人，2020）和基于知识的对话（迪南等人，2019），段落的密集检索（李等人，2019；卡尔普欣等人，2020；熊等人，2021）已被证明优于传统的稀疏检索方法，如词频 - 逆文档频率（TF - IDF）和BM25。

<!-- Media -->

<!-- figureText: Passage DPR Passage [CLS] Equations of motion [SEP] In physics, equations of motion are equations that describe the behavior of a physical system in terms of its motion as a function of time. More ... Passage [CLS] Equations of motion [SEP] In physics -equations of motion are equations that describe - the-behavior of a physical system in terms of its motion as a function of time. More .. representation ⑥ ① Passage DensePhrases OOD ① $\vdots$ -->

<img src="https://cdn.noedgeai.com/01957d26-d273-7c1a-b6dd-f745aa37d7cb_0.jpg?x=861&y=590&w=582&h=395&r=0"/>

Figure 1: Comparison of passage representations from DPR (Karpukhin et al., 2020) and DensePhrases (Lee et al., 2021). Unlike using a single vector for each passage, DensePhrases represents each passage with multiple phrase vectors and the score of a passage can be computed by the maximum score of phrases within it.

图1：对比DPR（卡尔普欣等人，2020）和DensePhrases（李等人，2021）的段落表示。与为每个段落使用单个向量不同，DensePhrases用多个短语向量表示每个段落，并且段落的得分可以通过其中短语的最高得分来计算。

<!-- Media -->

One natural design choice of these dense retrieval methods is the retrieval unit. For instance, the dense passage retriever (DPR) (Karpukhin et al., 2020) encodes a fixed-size text block of 100 words as the basic retrieval unit. On the other extreme, recent work (Seo et al., 2019; Lee et al., 2021) demonstrates that phrases can be used as a retrieval unit. In particular, Lee et al. (2021) show that learning dense representations of phrases alone can achieve competitive performance in a number of open-domain QA and slot filling tasks. This is particularly appealing since the phrases can directly serve as the output, without relying on an additional reader model to process text passages.

这些密集检索方法的一个自然设计选择是检索单元。例如，密集段落检索器（DPR）（卡尔普欣等人，2020年）将100个单词的固定大小文本块编码为基本检索单元。另一个极端情况是，近期的研究（徐等人，2019年；李等人，2021年）表明，可以将短语用作检索单元。特别是，李等人（2021年）表明，仅学习短语的密集表示就可以在许多开放领域问答和槽填充任务中取得有竞争力的表现。这特别有吸引力，因为短语可以直接作为输出，而无需依赖额外的阅读器模型来处理文本段落。

In this work, we draw on an intuitive motivation that every single phrase is embedded within a larger text context and ask the following question: If a retriever is able to locate phrases, can 3661 we directly make use of it for passage and even document retrieval as well? We formulate phrase-based passage retrieval, in which the score of a passage is determined by the maximum score of phrases within it (see Figure 1 for an illustration). By evaluating DensePhrases (Lee et al., 2021) on popular QA datasets, we observe that it achieves competitive or even better passage retrieval accuracy compared to DPR, without any re-training or modification to the original model (Table 1). The gains are especially pronounced for top- $k$ accuracy when $k$ is smaller (e.g.,5),which also helps achieve strong open-domain QA accuracy with a much smaller number of passages as input to a generative reader model (Izacard and Grave, 2021b).

在这项工作中，我们基于一个直观的动机，即每个短语都嵌入在更大的文本上下文中，并提出以下问题：如果一个检索器能够定位短语，我们能否直接利用它进行段落甚至文档检索呢？我们提出了基于短语的段落检索方法，其中段落的得分由其内部短语的最高得分决定（如图1所示）。通过在流行的问答数据集上评估DensePhrases（李等人，2021），我们发现与DPR相比，它在无需对原始模型进行任何重新训练或修改的情况下，实现了具有竞争力甚至更好的段落检索准确率（表1）。当$k$较小时（例如5），前$k$准确率的提升尤为明显，这也有助于在向生成式阅读器模型输入数量少得多的段落时实现较高的开放域问答准确率（伊扎卡德和格拉夫，2021b）。

---

<!-- Footnote -->

*This work was done when JL worked as a visiting research scholar at Princeton University.

*这项工作是JL在普林斯顿大学(Princeton University)担任访问研究学者时完成的。

${}^{1}$ Following previous work (Seo et al.,2018,2019),the term phrase denotes any contiguous text segment up to $L$ words, which is not necessarily a linguistic phrase (see Section 2).

${}^{1}$ 遵循先前的研究（徐（Seo）等人，2018年，2019年），术语“短语”表示长度不超过 $L$ 个单词的任何连续文本片段，它不一定是语言学意义上的短语（见第2节）。

${}^{2}$ Our code and models are available at https:// github.com/princeton-nlp/DensePhrases.

${}^{2}$ 我们的代码和模型可在https:// github.com/princeton-nlp/DensePhrases获取。

<!-- Footnote -->

---

To better understand the nature of dense retrieval methods, we carefully analyze the training objectives of phrase and passage retrieval methods. While the in-batch negative losses in both models encourage them to retrieve topically relevant passages, we find that phrase-level supervision in DensePhrases provides a stronger training signal than using hard negatives from BM25, and helps DensePhrases retrieve correct phrases, and hence passages. Following this positive finding, we further explore whether phrase retrieval can be extended to retrieval of coarser granularities, or other NLP tasks. Through fine-tuning of the query encoder with document-level supervision, we are able to obtain competitive performance on entity linking (Hoffart et al., 2011) and knowledge-grounded dialogue retrieval (Dinan et al., 2019) in the KILT benchmark (Petroni et al., 2021).

为了更好地理解密集检索方法的本质，我们仔细分析了短语和段落检索方法的训练目标。虽然两种模型中的批内负样本损失都促使它们检索主题相关的段落，但我们发现，“密集短语”（DensePhrases）中的短语级监督比使用来自BM25的难负样本提供了更强的训练信号，并有助于“密集短语”（DensePhrases）检索正确的短语，进而检索到正确的段落。基于这一积极的发现，我们进一步探索短语检索是否可以扩展到更粗粒度的检索或其他自然语言处理任务。通过使用文档级监督对查询编码器进行微调，我们能够在KILT基准测试（彼得罗尼（Petroni）等人，2021年）中的实体链接（霍法特（Hoffart）等人，2011年）和基于知识的对话检索（迪南（Dinan）等人，2019年）任务上取得有竞争力的性能。

Finally, we draw connections to multi-vector passage encoding models (Khattab and Zaharia, 2020; Luan et al., 2021), where phrase retrieval models can be viewed as learning a dynamic set of vectors for each passage. We show that a simple phrase filtering strategy learned from QA datasets gives us a control over the trade-off between the number of vectors per passage and the retrieval accuracy. Since phrase retrievers encode a larger number of vectors, we also propose a quantization-aware fine-tuning method based on Optimized Product Quantization (Ge et al., 2013), reducing the size of the phrase index from 307GB to ${69}\mathrm{{GB}}$ (or under ${30}\mathrm{{GB}}$ with more aggressive phrase filtering) for full English Wikipedia, without any performance degradation. This matches the index size of passage retrievers and makes dense phrase retrieval a practical and versatile solution for multi-granularity retrieval.

最后，我们将其与多向量段落编码模型（卡塔布和扎哈里亚，2020年；栾等人，2021年）建立联系，在这些模型中，短语检索模型可被视为为每个段落学习一组动态向量。我们表明，从问答数据集学习到的简单短语过滤策略使我们能够控制每个段落的向量数量与检索准确率之间的权衡。由于短语检索器会编码大量向量，我们还提出了一种基于优化乘积量化（葛等人，2013年）的量化感知微调方法，在不降低任何性能的情况下，将英文维基百科全文的短语索引大小从307GB缩减至${69}\mathrm{{GB}}$（若采用更激进的短语过滤策略，可缩减至${30}\mathrm{{GB}}$以下）。这与段落检索器的索引大小相匹配，使密集短语检索成为多粒度检索的一种实用且通用的解决方案。

## 2 Background

## 2 背景

Passage retrieval Given a set of documents $\mathcal{D}$ , passage retrieval aims to provide a set of relevant passages for a question $q$ . Typically,each document in $\mathcal{D}$ is segmented into a set of disjoint passages and we denote the entire set of passages in $\mathcal{D}$ as $\mathcal{P} = \left\{  {{p}_{1},\ldots ,{p}_{M}}\right\}$ ,where each passage can be a natural paragraph or a fixed-length text block. A passage retriever is designed to return top- $k$ passages ${\mathcal{P}}_{k} \subset  \mathcal{P}$ with the goal of retrieving passages that are relevant to the question. In open-domain QA, passages are considered relevant if they contain answers to the question. However, many other knowledge-intensive NLP tasks (e.g., knowledge-grounded dialogue) provide human-annotated evidence passages or documents.

段落检索 给定一组文档$\mathcal{D}$，段落检索旨在为问题$q$提供一组相关段落。通常，$\mathcal{D}$中的每个文档都会被分割成一组不相交的段落，我们将$\mathcal{D}$中的所有段落集合表示为$\mathcal{P} = \left\{  {{p}_{1},\ldots ,{p}_{M}}\right\}$，其中每个段落可以是一个自然段落或一个固定长度的文本块。段落检索器旨在返回前$k$个段落${\mathcal{P}}_{k} \subset  \mathcal{P}$，目标是检索与问题相关的段落。在开放领域问答中，如果段落包含问题的答案，则认为这些段落是相关的。然而，许多其他知识密集型自然语言处理任务（例如，基于知识的对话）会提供人工标注的证据段落或文档。

While traditional passage retrieval models rely on sparse representations such as BM25 (Robertson and Zaragoza, 2009), recent methods show promising results with dense representations of passages and questions, and enable retrieving passages that may have low lexical overlap with questions. Specifically, Karpukhin et al. (2020) introduce DPR that has a passage encoder ${E}_{p}\left( \cdot \right)$ and a question encoder ${E}_{q}\left( \cdot \right)$ trained on QA datasets and retrieves passages by using the inner product as a similarity function between a passage and a question:

虽然传统的段落检索模型依赖于诸如BM25（罗伯逊和萨拉戈萨，2009年）之类的稀疏表示，但最近的方法在段落和问题的密集表示方面显示出了有前景的结果，并且能够检索出与问题词汇重叠度可能较低的段落。具体而言，卡尔普欣等人（2020年）引入了DPR，它有一个段落编码器${E}_{p}\left( \cdot \right)$和一个问题编码器${E}_{q}\left( \cdot \right)$，这些编码器在问答数据集上进行训练，并通过使用内积作为段落和问题之间的相似度函数来检索段落：

$$
f\left( {p,q}\right)  = {E}_{p}{\left( p\right) }^{\top }{E}_{q}\left( q\right) . \tag{1}
$$

For open-domain QA where a system is required to provide an exact answer string $a$ ,the retrieved top $k$ passages ${\mathcal{P}}_{k}$ are subsequently fed into a reading comprehension model such as a BERT model (Devlin et al., 2019), and this is called the retriever-reader approach (Chen et al., 2017).

对于需要系统提供确切答案字符串$a$的开放域问答，检索到的前$k$个段落${\mathcal{P}}_{k}$随后会被输入到阅读理解模型（如BERT模型（德夫林等人，2019年））中，这被称为检索器 - 阅读器方法（陈等人，2017年）。

Phrase retrieval While passage retrievers require another reader model to find an answer, Seo et al. (2019) introduce the phrase retrieval approach that encodes phrases in each document and performs similarity search over all phrase vectors to directly locate the answer. Following previous work (Seo et al., 2018, 2019), we use the term 'phrase' to denote any contiguous text segment up to $L$ words (including single words), which is not necessarily a linguistic phrase and we take phrases up to length $L = {20}$ . Given a phrase ${s}^{\left( p\right) }$ from a passage $p$ , their similarity function $f$ is computed as:

短语检索 虽然段落检索器需要另一个阅读器模型来寻找答案，但徐（Seo）等人（2019年）引入了短语检索方法，该方法对每个文档中的短语进行编码，并对所有短语向量进行相似度搜索，以直接定位答案。遵循先前的研究（徐（Seo）等人，2018年，2019年），我们使用“短语”一词来表示长度不超过 $L$ 个词（包括单个词）的任何连续文本片段，它不一定是语言学意义上的短语，并且我们采用长度不超过 $L = {20}$ 的短语。给定来自段落 $p$ 的短语 ${s}^{\left( p\right) }$，其相似度函数 $f$ 计算如下：

$$
f\left( {{s}^{\left( p\right) },q}\right)  = {E}_{s}{\left( {s}^{\left( p\right) }\right) }^{\top }{E}_{q}\left( q\right) , \tag{2}
$$

where ${E}_{s}\left( \cdot \right)$ and ${E}_{q}\left( \cdot \right)$ denote the phrase encoder and the question encoder, respectively. Since this formulates open-domain QA purely as a maximum inner product search (MIPS), it can drastically improve end-to-end efficiency. While previous work (Seo et al., 2019; Lee et al., 2020) relied on a combination of dense and sparse vectors, Lee et al. (2021) demonstrate that dense representations of phrases alone are sufficient to close the performance gap with retriever-reader systems. For more details on how phrase representations are learned, we refer interested readers to Lee et al. (2021).

其中 ${E}_{s}\left( \cdot \right)$ 和 ${E}_{q}\left( \cdot \right)$ 分别表示短语编码器和问题编码器。由于这种方法将开放域问答纯粹表述为最大内积搜索（MIPS），因此可以显著提高端到端效率。此前的研究（徐（Seo）等人，2019年；李（Lee）等人，2020年）依赖于密集向量和稀疏向量的组合，而李（Lee）等人（2021年）证明，仅短语的密集表示就足以缩小与检索器 - 阅读器系统之间的性能差距。关于如何学习短语表示的更多细节，我们建议感兴趣的读者参考李（Lee）等人（2021年）的研究。

<!-- Media -->

<table><tr><td rowspan="2">Retriever</td><td colspan="5">Natural Questions</td><td colspan="5">TriviaQA</td></tr><tr><td>Top-1</td><td>Top-5</td><td>Top-20</td><td>MRR@20</td><td>P@20</td><td>Top-1</td><td>Top-5</td><td>Top-20</td><td>MRR@20</td><td>P@20</td></tr><tr><td>${\text{DPR}}^{\diamondsuit }$</td><td>46.0</td><td>68.1</td><td>79.8</td><td>55.7</td><td>16.5</td><td>${54.4}^{ \dagger  }$</td><td>-</td><td>${79.4}^{ \ddagger  }$</td><td>-</td><td>-</td></tr><tr><td>DPR*</td><td>44.2</td><td>66.8</td><td>79.2</td><td>54.2</td><td>17.7</td><td>54.6</td><td>70.8</td><td>79.5</td><td>61.7</td><td>30.3</td></tr><tr><td>DensePhrases ${}^{\diamondsuit }$</td><td>50.1</td><td>69.5</td><td>79.8</td><td>58.7</td><td>20.5</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>DensePhrases</td><td>51.1</td><td>69.9</td><td>78.7</td><td>59.3</td><td>22.7</td><td>62.7</td><td>75.0</td><td>80.9</td><td>68.2</td><td>38.4</td></tr></table>

<table><tbody><tr><td rowspan="2">检索器</td><td colspan="5">自然问题（Natural Questions）</td><td colspan="5">问答琐事集（TriviaQA）</td></tr><tr><td>排名第一</td><td>排名前五</td><td>排名前二十</td><td>20时的平均倒数排名（MRR@20）</td><td>P@20</td><td>排名第一</td><td>排名前五</td><td>排名前二十</td><td>20时的平均倒数排名（MRR@20）</td><td>P@20</td></tr><tr><td>${\text{DPR}}^{\diamondsuit }$</td><td>46.0</td><td>68.1</td><td>79.8</td><td>55.7</td><td>16.5</td><td>${54.4}^{ \dagger  }$</td><td>-</td><td>${79.4}^{ \ddagger  }$</td><td>-</td><td>-</td></tr><tr><td>密集段落检索器（DPR*）</td><td>44.2</td><td>66.8</td><td>79.2</td><td>54.2</td><td>17.7</td><td>54.6</td><td>70.8</td><td>79.5</td><td>61.7</td><td>30.3</td></tr><tr><td>密集短语模型（DensePhrases ${}^{\diamondsuit }$）</td><td>50.1</td><td>69.5</td><td>79.8</td><td>58.7</td><td>20.5</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>密集短语模型（DensePhrases）</td><td>51.1</td><td>69.9</td><td>78.7</td><td>59.3</td><td>22.7</td><td>62.7</td><td>75.0</td><td>80.9</td><td>68.2</td><td>38.4</td></tr></tbody></table>

Table 1: Open-domain QA passage retrieval results. We retrieve top $k$ passages from DensePhrases using Eq. (3). We report top- $k$ passage retrieval accuracy (Top- $k$ ),mean reciprocal rank at $k$ (MRR@ $k$ ),and precision at $k$ (P@ $k$ ). ${}^{\diamondsuit }$ : trained on each dataset independently. ${}^{\spadesuit }$ : trained on multiple open-domain QA datasets. See $§{3.1}$ for more details. ${}^{ \dagger  }$ : (Yang and Seo,2020). ${}^{ \ddagger  }$ : (Karpukhin et al.,2020).

表1：开放域问答段落检索结果。我们使用公式(3)从DensePhrases中检索前$k$个段落。我们报告前$k$个段落的检索准确率（Top - $k$）、在$k$处的平均倒数排名（MRR@ $k$）以及在$k$处的精确率（P@ $k$）。${}^{\diamondsuit }$：在每个数据集上独立训练。${}^{\spadesuit }$：在多个开放域问答数据集上训练。更多细节见$§{3.1}$。${}^{ \dagger  }$：（杨和徐，2020年）。${}^{ \ddagger  }$：（卡尔普欣等人，2020年）

<!-- Media -->

## 3 Phrase Retrieval for Passage Retrieval

## 3 用于段落检索的短语检索

Phrases naturally have their source texts from which they are extracted. Based on this fact, we define a simple phrase-based passage retrieval strategy, where we retrieve passages based on the phrase-retrieval score:

短语自然有其被提取出来的源文本。基于这一事实，我们定义了一种简单的基于短语的段落检索策略，即根据短语检索得分来检索段落：

$$
\widetilde{f}\left( {p,q}\right)  \mathrel{\text{:=}} \mathop{\max }\limits_{{{s}^{\left( p\right) } \in  \mathcal{S}\left( p\right) }}{E}_{s}{\left( {s}^{\left( p\right) }\right) }^{\top }{E}_{q}\left( q\right) , \tag{3}
$$

where $\mathcal{S}\left( p\right)$ denotes the set of phrases in the passage $p$ . In practice,we first retrieve a slightly larger number of phrases, compute the score for each passage,and return top $k$ unique passages. ${}^{3}$ Based on our definition, phrases can act as a basic retrieval unit of any other granularity such as sentences or documents by simply changing $\mathcal{S}\left( p\right)$ (e.g., ${s}^{\left( d\right) } \in  \mathcal{S}\left( d\right)$ for a document $d$ ). Note that,since the cost of score aggregation is negligible, the inference speed of phrase-based passage retrieval is the same as for phrase retrieval, which is shown to be efficient in Lee et al. (2021). In this section, we evaluate the passage retrieval performance (Eq. (3)) and also how phrase-based passage retrieval can contribute to end-to-end open-domain QA.

其中 $\mathcal{S}\left( p\right)$ 表示段落 $p$ 中的短语集合。在实践中，我们首先检索数量略多的短语，计算每个段落的得分，并返回前 $k$ 个唯一的段落。${}^{3}$ 根据我们的定义，通过简单地改变 $\mathcal{S}\left( p\right)$（例如，对于文档 $d$ 为 ${s}^{\left( d\right) } \in  \mathcal{S}\left( d\right)$），短语可以作为任何其他粒度（如句子或文档）的基本检索单元。请注意，由于得分聚合的成本可以忽略不计，基于短语的段落检索的推理速度与短语检索相同，Lee 等人（2021 年）的研究表明这种检索方式是高效的。在本节中，我们评估段落检索性能（公式 (3)），以及基于短语的段落检索如何有助于端到端的开放域问答。

### 3.1 Experiment: Passage Retrieval

### 3.1 实验：段落检索

Datasets We use two open-domain QA datasets: Natural Questions (Kwiatkowski et al., 2019) and TriviaQA (Joshi et al., 2017), following the standard train/dev/test splits for the open-domain QA evaluation. For both models, we use the 2018-12- 20 Wikipedia snapshot. To provide a fair comparison, we use Wikipedia articles pre-processed for DPR, which are split into 21-million text blocks and each text block has exactly 100 words. Note that while DPR is trained in this setting, DensePhrases is trained with natural paragraphs. ${}^{4}$

数据集 我们使用两个开放领域问答数据集：自然问答数据集（Natural Questions，Kwiatkowski等人，2019年）和琐事问答数据集（TriviaQA，Joshi等人，2017年），遵循开放领域问答评估的标准训练/开发/测试划分。对于这两个模型，我们使用2018年12月20日的维基百科快照。为了进行公平比较，我们使用为密集段落检索器（DPR）预先处理过的维基百科文章，这些文章被分割成2100万个文本块，每个文本块正好有100个单词。请注意，虽然密集段落检索器（DPR）是在这种设置下进行训练的，但密集短语模型（DensePhrases）是使用自然段落进行训练的。${}^{4}$

Models For DPR, we use publicly available checkpoints ${}^{5}$ trained on each dataset $\left( {\mathrm{{DPR}}}^{\diamondsuit }\right)$ or multiple QA datasets (DPR ${}^{\spadesuit }$ ),which we find to perform slightly better than the ones reported in Karpukhin et al. (2020). For DensePhrases, we train it on Natural Questions (DensePhrases ${}^{\diamondsuit }$ ) or multiple QA datasets (DensePhrases ${}^{\spadesuit }$ ) with the code provided by the authors. ${}^{6}$ Note that we do not make any modification to the architecture or training methods of DensePhrases and achieve similar open-domain QA accuracy as reported. For phrase-based passage retrieval, we compute Eq. (3) with DensePhrases and return top $k$ passages.

用于DPR（密集段落检索器，Dense Passage Retrieval）的模型，我们使用在每个数据集$\left( {\mathrm{{DPR}}}^{\diamondsuit }\right)$或多个问答数据集（DPR ${}^{\spadesuit }$）上训练的公开可用的检查点${}^{5}$，我们发现这些模型的表现比卡尔普欣（Karpukhin）等人（2020年）报告的模型略好。对于密集短语模型（DensePhrases），我们使用作者提供的代码在自然问答数据集（密集短语模型 ${}^{\diamondsuit }$）或多个问答数据集（密集短语模型 ${}^{\spadesuit }$）上对其进行训练。${}^{6}$请注意，我们没有对密集短语模型的架构或训练方法进行任何修改，并且实现了与报告中相似的开放领域问答准确率。对于基于短语的段落检索，我们使用密集短语模型计算公式（3），并返回前$k$个段落。

Metrics Following previous work on passage retrieval for open-domain QA,we measure the top- $k$ passage retrieval accuracy (Top- $k$ ),which denotes the proportion of questions whose top $k$ retrieved passages contain at least one of the gold answers. To further characterize the behavior of each system, we also include the following evaluation metrics: mean reciprocal rank at $k\left( {\operatorname{MRR}@k}\right)$ and precision at $k\left( {\mathrm{P}@k}\right)$ . MRR $@k$ is the average reciprocal rank of the first relevant passage (that contains an answer) in the top $k$ passages. Higher MRR@ $k$ means relevant passages appear at higher ranks. Meanwhile, $\mathrm{P}@k$ is the average proportion of relevant passages in the top $k$ passages. Higher $\mathrm{P}@k$ denotes that a larger proportion of top $k$ passages contains the answers.

评估指标 遵循之前在开放域问答的段落检索方面的工作，我们衡量前 $k$ 段落检索准确率（Top- $k$ ），它表示前 $k$ 检索到的段落中至少包含一个标准答案的问题所占的比例。为了进一步描述每个系统的性能，我们还纳入了以下评估指标：$k\left( {\operatorname{MRR}@k}\right)$ 处的平均倒数排名（Mean Reciprocal Rank，MRR）和 $k\left( {\mathrm{P}@k}\right)$ 处的精确率。MRR $@k$ 是前 $k$ 个段落中第一个相关段落（包含答案）的平均倒数排名。MRR@ $k$ 越高，意味着相关段落的排名越靠前。同时，$\mathrm{P}@k$ 是前 $k$ 个段落中相关段落的平均比例。$\mathrm{P}@k$ 越高，表示前 $k$ 个段落中包含答案的比例越大。

---

<!-- Footnote -->

${}^{4}$ We expect DensePhrases to achieve even higher performance if it is re-trained with 100-word text blocks. We leave it for future investigation.

${}^{4}$ 我们预计，如果使用 100 个单词的文本块对 DensePhrases 进行重新训练，它将取得更高的性能。我们将其留待未来研究。

${}^{5}$ https://github.com/facebookresearch/DPR.

${}^{5}$ https://github.com/facebookresearch/DPR.

${}^{6}{\mathrm{{DPR}}}^{ \bullet  }$ is trained on NaturalQuestions,TriviaQA,Curat-edTREC (Baudiš and Šedivý, 2015), and WebQuestions (Be-rant et al., 2013). DensePhrases ${}^{ \bullet  }$ additionally includes SQuAD (Rajpurkar et al., 2016), although it does not contribute to Natural Questions and TriviaQA much.

${}^{6}{\mathrm{{DPR}}}^{ \bullet  }$ 在自然问题数据集（NaturalQuestions）、琐事问答数据集（TriviaQA）、精选TREC数据集（Curat - edTREC，鲍迪什（Baudiš）和塞迪维（Šedivý），2015年）和网络问题数据集（WebQuestions，贝兰特（Be - rant）等人，2013年）上进行训练。密集短语模型（DensePhrases）${}^{ \bullet  }$ 还包含了斯坦福问答数据集（SQuAD，拉杰普尔卡（Rajpurkar）等人，2016年），尽管它对自然问题数据集和琐事问答数据集的贡献不大。

${}^{3}$ In most cases,retrieving ${2k}$ phrases is sufficient for obtaining $k$ unique passages. If not,we try ${4k}$ and so on.

${}^{3}$ 在大多数情况下，检索 ${2k}$ 短语足以获取 $k$ 唯一的段落。如果不够，我们尝试 ${4k}$ 等等。

<!-- Footnote -->

---

Results As shown in Table 1, DensePhrases achieves competitive passage retrieval accuracy with DPR, while having a clear advantage on top-1 or top-5 accuracy for both Natural Questions (+6.9% Top-1) and TriviaQA (+8.1% Top-1). Although the top-20 (and top-100, which is not shown) accuracy is similar across different models, MRR@20 and P@20 reveal interesting aspects of DensePhrases-it ranks relevant passages higher and provides a larger number of correct passages. Our results suggest that DensePhrases can also retrieve passages very accurately, even though it was not explicitly trained for that purpose. For the rest of the paper,we mainly compare the DPR $\uparrow$ and DensePhrases ${}^{\spadesuit }$ models,which were both trained on multiple QA datasets.

结果 如表1所示，DensePhrases在段落检索准确率方面与DPR相当，同时在自然问答（Natural Questions，Top-1准确率提高6.9%）和常识问答（TriviaQA，Top-1准确率提高8.1%）的Top-1或Top-5准确率上具有明显优势。尽管不同模型的Top-20（以及未展示的Top-100）准确率相近，但MRR@20和P@20揭示了DensePhrases的有趣之处——它能将相关段落排名更高，并提供更多正确的段落。我们的结果表明，即使DensePhrases并非专门为此目的进行训练，它也能非常准确地检索段落。在本文的其余部分，我们主要比较DPR $\uparrow$和DensePhrases ${}^{\spadesuit }$模型，这两个模型均在多个问答数据集上进行了训练。

### 3.2 Experiment: Open-domain QA

### 3.2 实验：开放域问答

Recently, Izacard and Grave (2021b) proposed the Fusion-in-Decoder (FiD) approach where they feed top 100 passages from DPR into a generative model T5 (Raffel et al., 2020) and achieve the state-of-the-art on open-domain QA benchmarks. Since their generative model computes the hidden states of all tokens in 100 passages, it requires large GPU memory and Izacard and Grave (2021b) used 64 Tesla V100 32GB for training.

最近，伊扎卡德（Izacard）和格拉夫（Grave）（2021b）提出了解码器融合（Fusion-in-Decoder，FiD）方法，他们将来自密集段落检索器（Dense Passage Retriever，DPR）的前100个段落输入到生成式模型T5（拉菲尔等人，2020）中，并在开放领域问答基准测试中取得了最先进的成果。由于他们的生成式模型会计算100个段落中所有标记的隐藏状态，因此需要大量的GPU内存，伊扎卡德和格拉夫（2021b）在训练时使用了64块32GB的特斯拉V100 GPU。

In this section, we use our phrase-based passage retrieval with DensePhrases to replace DPR in FiD and see if we can use a much smaller number of passages to achieve comparable performance, which can greatly reduce the computational requirements. We train our model with ${424}\mathrm{{GB}}$ RTX GPUs for training T5-base, which are more affordable with academic budgets. Note that training T5-base with 5 or 10 passages can also be done with 11GB GPUs. We keep all the hyperparameters the same as in Izacard and Grave (2021b). ${}^{7}$

在本节中，我们使用基于短语的段落检索方法DensePhrases来替代FiD中的DPR，看看是否可以使用数量少得多的段落来达到相当的性能，这可以大大降低计算需求。我们使用${424}\mathrm{{GB}}$ RTX GPU来训练T5-base模型，这些GPU在学术预算范围内更具性价比。请注意，使用5个或10个段落训练T5-base模型也可以用11GB的GPU完成。我们保持所有超参数与伊扎卡德和格拉夫（2021b）中的设置相同。${}^{7}$

<!-- Media -->

<table><tr><td rowspan="2">Model</td><td colspan="2">NaturalQ</td><td rowspan="2">TriviaQA Test</td></tr><tr><td>Dev</td><td>Test</td></tr><tr><td>ORQA (Lee et al., 2019)</td><td>-</td><td>33.3</td><td>45.0</td></tr><tr><td>REALM (Guu et al., 2020)</td><td>-</td><td>40.4</td><td>-</td></tr><tr><td>DPR (reader: BERT-base)</td><td>-</td><td>41.5</td><td>56.8</td></tr><tr><td>DensePhrases</td><td>-</td><td>41.3</td><td>53.5</td></tr></table>

<table><tbody><tr><td rowspan="2">模型</td><td colspan="2">自然问答（NaturalQ）</td><td rowspan="2">常识问答测试（TriviaQA Test）</td></tr><tr><td>开发集</td><td>测试集</td></tr><tr><td>开放检索问答模型（ORQA，李等人，2019年）</td><td>-</td><td>33.3</td><td>45.0</td></tr><tr><td>REALM（古等人，2020年）</td><td>-</td><td>40.4</td><td>-</td></tr><tr><td>DPR（阅读器：BERT-base）</td><td>-</td><td>41.5</td><td>56.8</td></tr><tr><td>密集短语</td><td>-</td><td>41.3</td><td>53.5</td></tr></tbody></table>

FiD with DPR (Izacard and Grave, 2021b) FiD with DensePhrases (ours)

使用DPR的FiD（伊扎卡德和格雷夫，2021b） 使用密集短语（我们的方法）的FiD

<table><tr><td>Reader: T5-base</td><td>$k = 5$</td><td>37.8</td><td>-</td><td>-</td></tr><tr><td/><td>$k = {10}$</td><td>42.3</td><td>-</td><td>-</td></tr><tr><td/><td>$k = {25}$</td><td>45.3</td><td>-</td><td>-</td></tr><tr><td/><td>$k = {50}$</td><td>45.7</td><td>-</td><td>-</td></tr><tr><td/><td>$k = {100}$</td><td>46.5</td><td>48.2</td><td>65.0</td></tr></table>

<table><tbody><tr><td>阅读器：T5-base</td><td>$k = 5$</td><td>37.8</td><td>-</td><td>-</td></tr><tr><td></td><td>$k = {10}$</td><td>42.3</td><td>-</td><td>-</td></tr><tr><td></td><td>$k = {25}$</td><td>45.3</td><td>-</td><td>-</td></tr><tr><td></td><td>$k = {50}$</td><td>45.7</td><td>-</td><td>-</td></tr><tr><td></td><td>$k = {100}$</td><td>46.5</td><td>48.2</td><td>65.0</td></tr></tbody></table>

<table><tr><td>Reader: T5-base</td><td>$k = 5$</td><td>44.2</td><td>45.9</td><td>59.5</td></tr><tr><td/><td>$k = {10}$</td><td>45.5</td><td>45.9</td><td>61.0</td></tr><tr><td/><td>$k = {25}$</td><td>46.4</td><td>47.2</td><td>63.4</td></tr><tr><td/><td>$k = {50}$</td><td>47.2</td><td>47.9</td><td>64.5</td></tr></table>

<table><tbody><tr><td>阅读器：T5基础版</td><td>$k = 5$</td><td>44.2</td><td>45.9</td><td>59.5</td></tr><tr><td></td><td>$k = {10}$</td><td>45.5</td><td>45.9</td><td>61.0</td></tr><tr><td></td><td>$k = {25}$</td><td>46.4</td><td>47.2</td><td>63.4</td></tr><tr><td></td><td>$k = {50}$</td><td>47.2</td><td>47.9</td><td>64.5</td></tr></tbody></table>

Table 2: Open-domain QA results. We report exact match (EM) of each model by feeding top $k$ passages into a T5-base model. DensePhrases can greatly reduce the computational cost of running generative reader models while having competitive performance.

表2：开放域问答结果。我们通过将前$k$个段落输入到T5-base模型中，报告了每个模型的精确匹配率（EM）。密集短语（DensePhrases）可以在具有有竞争力的性能的同时，大大降低运行生成式阅读器模型的计算成本。

<!-- Media -->

Results As shown in Table 2, using DensePhrases as a passage retriever achieves competitive performance to DPR-based FiD and significantly improves upon the performance of original DensePhrases (NQ = 41.3 EM without a reader). Its better retrieval quality at top- $k$ for smaller $k$ indeed translates to better open-domain QA accuracy, achieving $+ {6.4}\%$ gain compared to DPR-based FiD when $k = 5$ . To obtain similar performance with using 100 passages in FiD, DensePhrases needs fewer passages ( $k = {25}$ or 50 ),which can fit in GPUs with smaller RAM.

结果 如表2所示，使用密集短语（DensePhrases）作为段落检索器的性能与基于密集段落检索器（DPR）的FiD相当，并且显著优于原始密集短语（DensePhrases）的性能（没有阅读器时自然问答数据集（NQ）的精确匹配率为41.3）。对于较小的$k$值，它在前$k$个段落中的更好检索质量确实转化为更好的开放域问答准确率，当$k = 5$时，与基于密集段落检索器（DPR）的FiD相比，实现了$+ {6.4}\%$的提升。为了获得与在FiD中使用100个段落相似的性能，密集短语（DensePhrases）需要更少的段落（$k = {25}$或50个），这些段落可以适配内存较小的图形处理器（GPU）。

## 4 A Unified View of Dense Retrieval

## 4 密集检索的统一视角

As shown in the previous section, phrase-based passage retrieval is able to achieve competitive passage retrieval accuracy, despite that the models were not explicitly trained for that. In this section, we compare the training objectives of DPR and DensePhrases in detail and explain how DensePhrases learns passage retrieval.

如前一节所示，基于短语的段落检索能够实现具有竞争力的段落检索准确率，尽管这些模型并未针对该任务进行显式训练。在本节中，我们将详细比较DPR（密集段落检索器）和DensePhrases（密集短语）的训练目标，并解释DensePhrases是如何学习段落检索的。

### 4.1 Training Objectives

### 4.1 训练目标

Both DPR and DensePhrases set out to learn a similarity function $f$ between a passage or phrase and a question. Passages and phrases differ primarily in characteristic length, so we refer to either as a retrieval unit $x.{}^{8}$ DPR and DensePhrases both adopt a dual-encoder approach with inner product similarity as shown in Eq. (1) and (2), and they are initialized with BERT (Devlin et al., 2019) and SpanBERT (Joshi et al., 2020), respectively.

DPR和DensePhrases都旨在学习段落或短语与问题之间的相似度函数$f$。段落和短语的主要区别在于特征长度，因此我们将二者都称为检索单元$x.{}^{8}$。DPR和DensePhrases均采用双编码器方法，使用内积相似度，如公式(1)和(2)所示，并且它们分别使用BERT（伯特，德夫林等人，2019年）和SpanBERT（跨度伯特，乔希等人，2020年）进行初始化。

---

<!-- Footnote -->

${}^{7}$ We also accumulate gradients for 16 steps to match the effective batch size of the original work.

${}^{7}$ 我们还累积16步的梯度，以匹配原始工作的有效批量大小。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: DPR DensePhrases ① -[CLS] who sings don't stand so close to me [CLS] Don't Stand So Close to Me [SEP] "Don't Stand So Close to Me" is a hit song by the British rock-band the Police, released in September ... In-batch Negative [CLS] Barack Obama [SEP] Barack Hussein ① Obama is an American politician and attorne who served as the 44th president of the ... In-passage Negative released in September 1980 as the lead single ... from their third album Zenyatta Mondatta Question ① [CLS] who sings don't stand so close to me ① -[CLS] Don't Stand So Close to Me [SEP] "Don't Stand So Close to Me" is a hit song by the British rock band the Police, released in September ... In-batch Negative [CLS] Barack Obama [SEP] Barack Hussein Obama is an American politician and attorney who served as the 44th president of the .. Hard Negative ① a song written by Stephen Allen Davis, and recorded by American country music singer .. -->

<img src="https://cdn.noedgeai.com/01957d26-d273-7c1a-b6dd-f745aa37d7cb_4.jpg?x=206&y=191&w=1255&h=479&r=0"/>

Figure 2: Comparison of training objectives of DPR and DensePhrases. While both models use in-batch negatives, DensePhrases use in-passage negatives (phrases) compared to BM25 hard-negative passages in DPR. Note that each phrase in DensePhrases can directly serve as an answer to open-domain questions.

图2：DPR和DensePhrases训练目标的比较。虽然这两种模型都使用批次内负样本，但与DPR中使用的基于BM25的难负样本段落相比，DensePhrases使用段落内负样本（短语）。请注意，DensePhrases中的每个短语都可以直接作为开放领域问题的答案。

<!-- Media -->

These dual-encoder models are then trained with a negative log-likelihood loss for discriminating positive retrieval units from negative ones:

然后，使用负对数似然损失对这些双编码器模型进行训练，以区分正检索单元和负检索单元：

$$
\mathcal{L} =  - \log \frac{{e}^{f\left( {{x}^{ + },q}\right) }}{{e}^{f\left( {{x}^{ + },q}\right) } + \mathop{\sum }\limits_{{{x}^{ - } \in  {\mathcal{X}}^{ - }}}{e}^{f\left( {{x}^{ - },q}\right) }}, \tag{4}
$$

where ${x}^{ + }$ is the positive phrase or passage corresponding to question $q$ ,and ${\mathcal{X}}^{ - }$ is a set of negative examples. The choice of negatives is critical in this setting and both DPR and DensePhrases make important adjustments.

其中${x}^{ + }$是与问题$q$对应的正短语或段落，${\mathcal{X}}^{ - }$是一组负样本。在这种情况下，负样本的选择至关重要，DPR和DensePhrases都进行了重要调整。

In-batch negatives In-batch negatives are a common way to define ${\mathcal{X}}^{ - }$ ,since they are available at no extra cost when encoding a mini-batch of examples. Specifically,in a mini-batch of $B$ examples, we can add $B - 1$ in-batch negatives for each positive example. Since each mini-batch is randomly sampled from the set of all training passages, in-batch negative passages are usually topically negative,i.e.,models can discriminate between ${x}^{ + }$ and ${\mathcal{X}}^{ - }$ based on their topic only.

批次内负样本 批次内负样本是定义 ${\mathcal{X}}^{ - }$ 的常用方法，因为在对一小批示例进行编码时，无需额外成本即可获得它们。具体而言，在包含 $B$ 个示例的小批次中，我们可以为每个正样本添加 $B - 1$ 个批次内负样本。由于每个小批次都是从所有训练段落集合中随机采样得到的，因此批次内负段落通常在主题上是负相关的，即模型仅根据主题就可以区分 ${x}^{ + }$ 和 ${\mathcal{X}}^{ - }$。

Hard negatives Although topic-related features are useful in identifying broadly relevant passages, they often lack the precision to locate the exact passage containing the answer in a large corpus. Karpukhin et al. (2020) propose to use additional hard negatives which have a high BM25 lexical overlap with a given question but do not contain the answer. These hard negatives are likely to share a similar topic and encourage DPR to learn more fine-grained features to rank ${x}^{ + }$ over the hard negatives. Figure 2 (left) shows an illustrating example.

难负样本 虽然与主题相关的特征有助于识别大致相关的段落，但在大型语料库中，它们往往缺乏定位包含答案的确切段落的精度。卡尔普欣等人（Karpukhin et al., 2020）提议使用额外的难负样本，这些难负样本与给定问题在BM25词法上有很高的重叠度，但不包含答案。这些难负样本可能具有相似的主题，并促使密集段落检索器（DPR）学习更细粒度的特征，以便将 ${x}^{ + }$ 排在难负样本之上。图2（左）展示了一个示例。

In-passage negatives While DPR is limited to use positive passages ${x}^{ + }$ which contain the answer, DensePhrases is trained to predict that the positive phrase ${x}^{ + }$ is the answer. Thus,the fine-grained structure of phrases allows for another source of negatives, in-passage negatives. In particular, DensePhrases augments the set of negatives ${\mathcal{X}}^{ - }$ to encompass all phrases within the same passage that do not express the answer. ${}^{9}$ See Figure 2 (right) for an example. We hypothesize that these in-passage negatives achieve a similar effect as DPR's hard negatives: They require the model to go beyond simple topic modeling since they share not only the same topic but also the same context. Our phrase-based passage retriever might benefit from this phrase-level supervision, which has already been shown to be useful in the context of distilling knowledge from reader to retriever (Izac-ard and Grave, 2021a; Yang and Seo, 2020).

段落内负样本 虽然密集段落检索器（DPR）仅限于使用包含答案的正样本段落 ${x}^{ + }$，但密集短语模型（DensePhrases）经过训练，可以预测正样本短语 ${x}^{ + }$ 就是答案。因此，短语的细粒度结构提供了另一种负样本来源，即段落内负样本。具体而言，密集短语模型将负样本集合 ${\mathcal{X}}^{ - }$ 扩展到同一段落中所有不表达答案的短语。 ${}^{9}$ 示例见图2（右）。我们假设这些段落内负样本能达到与密集段落检索器的难负样本类似的效果：它们要求模型超越简单的主题建模，因为这些负样本不仅与正样本主题相同，而且上下文也相同。我们基于短语的段落检索器可能会从这种短语级别的监督中受益，在将知识从阅读器蒸馏到检索器的背景下，这种监督已被证明是有用的（伊扎克 - 阿德和格拉夫（Izac-ard and Grave），2021a；杨和徐（Yang and Seo），2020）。

### 4.2 Topical vs. Hard Negatives

### 4.2 主题负样本与难负样本

To address our hypothesis, we would like to study how these different types of negatives used by DPR and DensePhrases affect their reliance on topical and fine-grained entailment cues. We characterize their passage retrieval based on two metrics (losses): ${\mathcal{L}}_{\text{topic }}$ and ${\mathcal{L}}_{\text{hard }}$ . We use Eq. (4) to define both ${\mathcal{L}}_{\text{topic }}$ and ${\mathcal{L}}_{\text{hard }}$ ,but use different sets of negatives ${\mathcal{X}}^{ - }$ . For ${\mathcal{L}}_{\text{topic }},{\mathcal{X}}^{ - }$ contains passages that are topically different from the gold passage-In practice, we randomly sample passages from English Wikipedia. For ${\mathcal{L}}_{\text{hard }},{\mathcal{X}}^{ - }$ uses negatives containing topically similar passages,such that ${\mathcal{L}}_{\text{hard }}$ estimates how accurately models locate a passage that contains the exact answer among topically similar passages. From a positive passage paired with a question, we create a single hard negative by removing the sentence that contains the answer. ${}^{10}$ In our analysis, both metrics are estimated on the Natural Questions development set, which provides a set of questions and (gold) positive passages.

为了验证我们的假设，我们希望研究DPR（Dense Passage Retrieval，密集段落检索）和DensePhrases所使用的这些不同类型的负样本如何影响它们对主题和细粒度蕴含线索的依赖。我们基于两个指标（损失）来描述它们的段落检索情况：${\mathcal{L}}_{\text{topic }}$和${\mathcal{L}}_{\text{hard }}$。我们使用公式(4)来定义${\mathcal{L}}_{\text{topic }}$和${\mathcal{L}}_{\text{hard }}$，但使用不同的负样本集${\mathcal{X}}^{ - }$。对于${\mathcal{L}}_{\text{topic }},{\mathcal{X}}^{ - }$，其包含的段落与黄金段落主题不同——在实践中，我们从英文维基百科中随机抽取段落。对于${\mathcal{L}}_{\text{hard }},{\mathcal{X}}^{ - }$，使用包含主题相似段落的负样本，这样${\mathcal{L}}_{\text{hard }}$就能估计模型在主题相似的段落中定位包含确切答案的段落的准确程度。从与问题配对的正样本段落中，我们通过移除包含答案的句子来创建一个单一的难负样本。${}^{10}$在我们的分析中，这两个指标都是在自然问题开发集上进行估计的，该数据集提供了一组问题和（黄金）正样本段落。

---

<!-- Footnote -->

${}^{9}$ Technically,DensePhrases treats start and end representations of phrases independently and use start (or end) representations other than the positive one as negatives.

${}^{9}$ 从技术上讲，密集短语模型（DensePhrases）独立处理短语的起始和结束表示，并将除正样本之外的起始（或结束）表示用作负样本。

${}^{8}$ Note that phrases may overlap,whereas passages are usually disjoint segments with each other.

${}^{8}$ 请注意，短语可能会重叠，而段落通常是相互不相交的片段。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: 0.6 - DensePhrase in-batch in-passage only dataset ${10}^{-1}$ in-batch 0.5 + BM25 multi. dataset 0.2 ${10}^{-3}$ ${10}^{-2}$ -->

<img src="https://cdn.noedgeai.com/01957d26-d273-7c1a-b6dd-f745aa37d7cb_5.jpg?x=198&y=195&w=601&h=408&r=0"/>

Figure 3: Comparison of DPR and DensePhrases on NQ (dev) with ${\mathcal{L}}_{\text{topic }}$ and ${\mathcal{L}}_{\text{hard }}$ . Starting from each model trained with in-batch negatives (in-batch), we show the effect of using hard negatives (+BM25), in-passage negatives (+in-passage), as well as training on multiple QA datasets (+multi. dataset). The $x$ -axis is in log-scale for better visualization. For both metrics, lower numbers are better.

图3：在自然问答数据集（NQ，开发集）上，使用${\mathcal{L}}_{\text{topic }}$和${\mathcal{L}}_{\text{hard }}$对密集段落检索模型（DPR）和密集短语模型（DensePhrases）进行比较。从使用批次内负样本（in-batch）训练的每个模型开始，我们展示了使用难负样本（+BM25）、段落内负样本（+in-passage）以及在多个问答数据集上进行训练（+multi. dataset）的效果。$x$轴采用对数刻度以便更好地可视化。对于这两个指标，数值越低越好。

<!-- Media -->

Results Figure 3 shows the comparison of DPR and DensePhrases trained on NQ with the two losses. For DensePhrases, we compute the passage score using $\widetilde{f}\left( {p,q}\right)$ as described in Eq. (3). First, we observe that in-batch negatives are highly effective at reducing ${\mathcal{L}}_{\text{topic }}$ as DensePhrases trained with only in-passage negatives has a relatively high ${\mathcal{L}}_{\text{topic }}$ . Furthermore,we observe that using in-passage negatives in DensePhrases (+in-passage) significantly lowers ${\mathcal{L}}_{\text{hard }}$ ,even lower than DPR that uses BM25 hard negatives (+BM25). Using multiple datasets (+multi. dataset) further improves ${\mathcal{L}}_{\text{hard }}$ for both models. DPR has generally better (lower) ${\mathcal{L}}_{\text{topic }}$ than DensePhrases,which might be due to the smaller training batch size of DensePhrases (hence a smaller number of in-batch negatives) compared to DPR. The results suggest that DensePhrases relies less on topical features and is better at retrieving passages based on fine-grained entailment cues. This might contribute to the better ranking of the retrieved passages in Table 1, where DensePhrases shows better MRR@20 and P@20 while top-20 accuracy is similar.

结果 图3展示了在自然问答数据集（NQ）上使用两种损失函数训练的密集段落检索器（DPR）和密集短语检索器（DensePhrases）的对比情况。对于密集短语检索器（DensePhrases），我们按照公式（3）所述，使用$\widetilde{f}\left( {p,q}\right)$来计算段落得分。首先，我们发现批内负样本在降低${\mathcal{L}}_{\text{topic }}$方面非常有效，因为仅使用段落内负样本训练的密集短语检索器（DensePhrases）的${\mathcal{L}}_{\text{topic }}$相对较高。此外，我们还发现，在密集短语检索器（DensePhrases）中使用段落内负样本（+段落内）能显著降低${\mathcal{L}}_{\text{hard }}$，甚至比使用BM25硬负样本（+BM25）的密集段落检索器（DPR）还要低。对于这两种模型，使用多个数据集（+多数据集）能进一步改善${\mathcal{L}}_{\text{hard }}$。总体而言，密集段落检索器（DPR）的${\mathcal{L}}_{\text{topic }}$比密集短语检索器（DensePhrases）更好（更低），这可能是因为与密集段落检索器（DPR）相比，密集短语检索器（DensePhrases）的训练批次大小更小（因此批内负样本数量也更少）。这些结果表明，密集短语检索器（DensePhrases）对主题特征的依赖较小，更擅长基于细粒度的蕴含线索来检索段落。这可能是表1中检索段落排名更好的原因，在表1中，密集短语检索器（DensePhrases）的MRR@20和P@20表现更好，而前20准确率相近。

<!-- Media -->

<table><tr><td>Type</td><td>$\mathcal{D} = \{ p\}$</td><td>$\mathcal{D} = {\mathcal{D}}_{\text{small }}$</td></tr><tr><td>DensePhrases</td><td>71.8</td><td>61.3</td></tr><tr><td>+ BM25 neg.</td><td>71.8</td><td>60.6</td></tr><tr><td>+ Same-phrase neg.</td><td>72.1</td><td>60.9</td></tr></table>

<table><tbody><tr><td>类型</td><td>$\mathcal{D} = \{ p\}$</td><td>$\mathcal{D} = {\mathcal{D}}_{\text{small }}$</td></tr><tr><td>密集短语（DensePhrases）</td><td>71.8</td><td>61.3</td></tr><tr><td>+ BM25负样本</td><td>71.8</td><td>60.6</td></tr><tr><td>+ 同短语负样本</td><td>72.1</td><td>60.9</td></tr></tbody></table>

Table 3: Effect of using hard negatives in DensePhrases on the NQ development set. We report EM when a single gold passage is given $\left( {\mathcal{D} = \{ p\} }\right)$ or $6\mathrm{\;K}$ passages are given by gathering all the gold passages from NQ development set $\left( {\mathcal{D} = {\mathcal{D}}_{\text{small }}}\right)$ . The two hard negatives do not give any noticeable improvement in DensePhrases.

表3：在自然问答（NQ）开发集上，在密集短语（DensePhrases）中使用难负样本（hard negatives）的效果。我们报告了在给定单个正确段落$\left( {\mathcal{D} = \{ p\} }\right)$或通过收集自然问答开发集$\left( {\mathcal{D} = {\mathcal{D}}_{\text{small }}}\right)$中的所有正确段落给出$6\mathrm{\;K}$个段落时的精确匹配率（EM）。这两种难负样本在密集短语中并未带来明显的提升。

<!-- Media -->

Hard negatives for DensePhrases? We test two different kinds of hard negatives in DensePhrases to see whether its performance can further improve in the presence of in-passage negatives. For each training question, we mine for a hard negative passage, either by BM25 similarity or by finding another passage that contains the gold-answer phrase, but possibly with a wrong context. Then we use all phrases from the hard negative passage as additional hard negatives in ${\mathcal{X}}^{ - }$ along with the existing in-passage negatives. As shown in Table 3, DensePhrases obtains no substantial improvements from additional hard negatives, indicating that in-passage negatives are already highly effective at producing good phrase (or passage) representations.

密集短语（DensePhrases）的难负样本（Hard negatives）？我们在密集短语（DensePhrases）中测试了两种不同类型的难负样本，以查看在存在段落内负样本（in-passage negatives）的情况下，其性能是否能进一步提升。对于每个训练问题，我们会挖掘一个难负段落，要么通过BM25相似度挖掘，要么找到另一个包含正确答案短语但可能上下文错误的段落。然后，我们将难负段落中的所有短语作为额外的难负样本，与现有的段落内负样本一起用于${\mathcal{X}}^{ - }$。如表3所示，密集短语（DensePhrases）从额外的难负样本中并未获得实质性的改进，这表明段落内负样本在生成良好的短语（或段落）表示方面已经非常有效。

## 5 Improving Coarse-grained Retrieval

## 5 改进粗粒度检索

While we showed that DensePhrases implicitly learns passage retrieval, Figure 3 indicates that DensePhrases might not be very good for retrieval tasks where topic matters more than fine-grained entailment, for instance, the retrieval of a single evidence document for entity linking. In this section, we propose a simple method that can adapt DensePhrases to larger retrieval units, especially when the topical relevance is more important.

虽然我们已经表明密集短语（DensePhrases）能隐式学习段落检索，但图3表明，对于主题比细粒度蕴含关系更重要的检索任务，例如实体链接的单一证据文档检索，密集短语（DensePhrases）可能效果不太好。在本节中，我们提出一种简单的方法，该方法可以使密集短语（DensePhrases）适应更大的检索单元，特别是在主题相关性更为重要的情况下。

---

<!-- Footnote -->

${}^{10}$ While ${\mathcal{L}}_{\text{hard }}$ with this type of hard negatives might favor DensePhrases,using BM25 hard negatives for ${\mathcal{L}}_{\text{hard }}$ would favor DPR since DPR was directly trained on BM25 hard negatives. Nonetheless,we observed similar trends in ${\mathcal{L}}_{\text{hard }}$ regardless of the choice of hard negatives.

${}^{10}$ 虽然 ${\mathcal{L}}_{\text{hard }}$ 使用此类难负样本（hard negatives）可能有利于密集短语检索模型（DensePhrases），但对 ${\mathcal{L}}_{\text{hard }}$ 使用基于二元独立模型（BM25）的难负样本则可能有利于密集段落检索模型（DPR），因为 DPR 是直接在基于 BM25 的难负样本上进行训练的。尽管如此，无论选择何种难负样本，我们在 ${\mathcal{L}}_{\text{hard }}$ 中都观察到了相似的趋势。

<!-- Footnote -->

---

Method We modify the query-side fine-tuning proposed by Lee et al. (2021), which drastically improves the performance of DensePhrases by reducing the discrepancy between training and inference time. Since it is prohibitive to update the large number of phrase representations after indexing, only the query encoder is fine-tuned over the entire set of phrases in Wikipedia. Given a question $q$ and an annotated document set ${\mathcal{D}}^{ * }$ ,we minimize:

方法 我们对李等人（Lee et al., 2021）提出的查询端微调方法进行了改进，该方法通过减少训练和推理时间之间的差异，显著提高了密集短语检索模型（DensePhrases）的性能。由于在索引后更新大量短语表示的计算成本过高，因此仅在维基百科的整个短语集上对查询编码器进行微调。给定一个问题 $q$ 和一个带注释的文档集 ${\mathcal{D}}^{ * }$，我们将以下目标函数最小化：

$$
{\mathcal{L}}_{\text{doc }} =  - \log \frac{\mathop{\sum }\limits_{{s \in  \widetilde{\mathcal{S}}\left( q\right) ,d\left( s\right)  \in  {\mathcal{D}}^{ * }}}{e}^{f\left( {s,q}\right) }}{\mathop{\sum }\limits_{{s \in  \widetilde{\mathcal{S}}\left( q\right) }}{e}^{f\left( {s,q}\right) }}, \tag{5}
$$

where $\widetilde{\mathcal{S}}\left( q\right)$ denotes top $k$ phrases for the question $q$ ,out of the entire set of phrase vectors. To retrieve coarse-grained text better, we simply check the condition $d\left( s\right)  \in  {\mathcal{D}}^{ * }$ ,which means $d\left( s\right)$ ,the source document of $s$ ,is included in the set of annotated gold documents ${\mathcal{D}}^{ * }$ for the question. With ${\mathcal{L}}_{\text{doc }}$ , the model is trained to retrieve any phrases that are contained in a relevant document. Note that $d\left( s\right)$ can be changed to reflect any desired level of granularity such as passages.

其中 $\widetilde{\mathcal{S}}\left( q\right)$ 表示问题 $q$ 在整个短语向量集中排名前 $k$ 的短语。为了更好地检索粗粒度文本，我们只需检查条件 $d\left( s\right)  \in  {\mathcal{D}}^{ * }$，这意味着 $s$ 的源文档 $d\left( s\right)$ 包含在该问题的标注黄金文档集 ${\mathcal{D}}^{ * }$ 中。通过 ${\mathcal{L}}_{\text{doc }}$，模型被训练来检索相关文档中包含的任何短语。请注意，$d\left( s\right)$ 可以更改以反映任何所需的粒度级别，例如段落。

Datasets We test DensePhrases trained with ${\mathcal{L}}_{\text{doc }}$ on entity linking (Hoffart et al., 2011; Guo and Bar-bosa, 2018) and knowledge-grounded dialogue (Dinan et al., 2019) tasks in KILT (Petroni et al., 2021). Entity linking contains three datasets: AIDA CoNLL-YAGO (AY2) (Hoffart et al., 2011), WNED-WIKI (WnWi) (Guo and Barbosa, 2018), and WNED-CWEB (WnCw) (Guo and Barbosa, 2018). Each query in entity linking datasets contains a named entity marked with special tokens (i.e., [ START_ENT ], [END_ENT ] ), which need to be linked to one of the Wikipedia articles. For knowledge-grounded dialogue, we use Wizard of Wikipedia (WoW) (Dinan et al., 2019) where each query consists of conversation history, and the generated utterances should be grounded in one of the Wikipedia articles. We follow the KILT guidelines and evaluate the document (i.e., Wikipedia article) retrieval performance of our models given each query. We use R-precision, the proportion of successfully retrieved pages in the top $R$ results,where $\mathrm{R}$ is the number of distinct pages in the provenance set. However, in the tasks considered, R-precision is equivalent to precision $@1$ ,since each question is annotated with only one document.

数据集 我们在KILT（彼得罗尼等人，2021年）中的实体链接（霍法特等人，2011年；郭和巴尔博萨，2018年）和基于知识的对话（迪南等人，2019年）任务上测试了用${\mathcal{L}}_{\text{doc }}$训练的DensePhrases。实体链接包含三个数据集：AIDA CoNLL - YAGO（AY2）（霍法特等人，2011年）、WNED - WIKI（WnWi）（郭和巴尔博萨，2018年）和WNED - CWEB（WnCw）（郭和巴尔博萨，2018年）。实体链接数据集中的每个查询都包含一个用特殊标记（即[START_ENT]、[END_ENT]）标记的命名实体，该实体需要与一篇维基百科文章建立链接。对于基于知识的对话，我们使用《维基百科向导》（WoW）（迪南等人，2019年），其中每个查询由对话历史组成，生成的话语应基于某一篇维基百科文章。我们遵循KILT指南，评估给定每个查询时我们模型的文档（即维基百科文章）检索性能。我们使用R - 准确率，即前$R$个结果中成功检索到的页面的比例，其中$\mathrm{R}$是出处集中不同页面的数量。然而，在考虑的任务中，R - 准确率等同于准确率$@1$，因为每个问题仅用一篇文档进行标注。

Models DensePhrases is trained with the original query-side fine-tuning loss (denoted as ${\mathcal{L}}_{\text{phrase }}$ ) or with ${\mathcal{L}}_{\text{doc }}$ as described in Eq. (5). When

DensePhrases模型使用原始的查询端微调损失（表示为${\mathcal{L}}_{\text{phrase }}$）进行训练，或者按照公式（5）所述使用${\mathcal{L}}_{\text{doc }}$进行训练。当

<!-- Media -->

<table><tr><td rowspan="2">Model</td><td colspan="3">Entity Linking</td><td rowspan="2">Dialogue $\mathbf{{WoW}}$</td></tr><tr><td>AY2</td><td>WnWi</td><td>WnCw</td></tr><tr><td colspan="5">Retriever Only</td></tr><tr><td>TF-IDF</td><td>3.7</td><td>0.2</td><td>2.1</td><td>49.0</td></tr><tr><td>DPR</td><td>1.8</td><td>0.3</td><td>0.5</td><td>25.5</td></tr><tr><td>DensePhrases- ${\mathcal{L}}_{\text{phrase }}$</td><td>7.7</td><td>12.5</td><td>6.4</td><td>-</td></tr><tr><td>DensePhrases- ${\mathcal{L}}_{\text{doc }}$</td><td>61.6</td><td>32.1</td><td>37.4</td><td>47.0</td></tr><tr><td>DPR*</td><td>26.5</td><td>4.9</td><td>1.9</td><td>41.1</td></tr><tr><td>DensePhrases- ${\mathcal{L}}_{\text{doc }} \uparrow$</td><td>68.4</td><td>47.5</td><td>47.5</td><td>55.7</td></tr></table>

<table><tbody><tr><td rowspan="2">模型</td><td colspan="3">实体链接</td><td rowspan="2">对话 $\mathbf{{WoW}}$</td></tr><tr><td>AY2</td><td>WnWi</td><td>WnCw</td></tr><tr><td colspan="5">仅检索器</td></tr><tr><td>词频-逆文档频率（TF-IDF）</td><td>3.7</td><td>0.2</td><td>2.1</td><td>49.0</td></tr><tr><td>密集段落检索器（DPR）</td><td>1.8</td><td>0.3</td><td>0.5</td><td>25.5</td></tr><tr><td>密集短语 - ${\mathcal{L}}_{\text{phrase }}$</td><td>7.7</td><td>12.5</td><td>6.4</td><td>-</td></tr><tr><td>密集短语 - ${\mathcal{L}}_{\text{doc }}$</td><td>61.6</td><td>32.1</td><td>37.4</td><td>47.0</td></tr><tr><td>密集段落检索器*（DPR*）</td><td>26.5</td><td>4.9</td><td>1.9</td><td>41.1</td></tr><tr><td>密集短语（DensePhrases） - ${\mathcal{L}}_{\text{doc }} \uparrow$</td><td>68.4</td><td>47.5</td><td>47.5</td><td>55.7</td></tr></tbody></table>

<table><tr><td colspan="5">Retriever + Additional Components</td></tr><tr><td>RAG</td><td>72.6</td><td>48.1</td><td>47.6</td><td>57.8</td></tr><tr><td>BLINK + flair</td><td>81.5</td><td>80.2</td><td>68.8</td><td>-</td></tr></table>

<table><tbody><tr><td colspan="5">检索器 + 附加组件</td></tr><tr><td>检索增强生成（RAG）</td><td>72.6</td><td>48.1</td><td>47.6</td><td>57.8</td></tr><tr><td>眨眼模型（BLINK） + 天赋模型（flair）</td><td>81.5</td><td>80.2</td><td>68.8</td><td>-</td></tr></tbody></table>

Table 4: Results on the KILT test set. We report page-level R-precision on each task, which is equivalent to precision@1 on these datasets. ${}^{ \star  }$ : Multi-task models.

表4：KILT测试集上的结果。我们报告了每个任务的页面级R准确率，这相当于这些数据集上的精确率@1。${}^{ \star  }$：多任务模型。

<!-- Media -->

DensePhrases is trained with ${\mathcal{L}}_{\text{phrase }}$ ,it labels any phrase that matches the title of gold document as positive. After training, DensePhrases returns the document that contains the top passage. For baseline retrieval methods, we report the performance of TF-IDF and DPR from Petroni et al. (2021). We also include a multi-task version of DPR and DensePhrases, which uses the entire KILT training datasets. ${}^{11}$ While not our main focus of comparison, we also report the performance of other baselines from Petroni et al. (2021), which uses generative models (e.g., RAG (Lewis et al., 2020)) or task-specific models (e.g., BLINK (Wu et al., 2020), which has additional entity linking pre-training). Note that these methods use additional components such as a generative model or a cross-encoder model on top of retrieval models.

DensePhrases模型使用${\mathcal{L}}_{\text{phrase }}$进行训练，它将与黄金文档标题匹配的任何短语标记为正样本。训练完成后，DensePhrases会返回包含排名最高段落的文档。对于基线检索方法，我们报告了佩特罗尼等人（Petroni et al., 2021）提出的TF-IDF和DPR的性能。我们还纳入了DPR和DensePhrases的多任务版本，该版本使用了整个KILT训练数据集。${}^{11}$虽然这不是我们比较的主要重点，但我们也报告了佩特罗尼等人（Petroni et al., 2021）提出的其他基线方法的性能，这些方法使用了生成式模型（例如，RAG（刘易斯等人，Lewis et al., 2020））或特定任务模型（例如，BLINK（吴等人，Wu et al., 2020），该模型进行了额外的实体链接预训练）。请注意，这些方法在检索模型的基础上还使用了额外的组件，如生成式模型或交叉编码器模型。

Results Table 4 shows the results on three entity linking tasks and a knowledge-grounded dialogue task. On all tasks, we find that DensePhrases with ${\mathcal{L}}_{\text{doc }}$ performs much better than DensePhrases with ${\mathcal{L}}_{\text{phrase }}$ and also matches the performance of RAG that uses an additional large generative model to generate the document titles. Using ${\mathcal{L}}_{\text{phrase }}$ does very poorly since it focuses on phrase-level entailment, rather than document-level relevance. Compared to the multi-task version of DPR (i.e., DPR*), DensePhrases- ${\mathcal{L}}_{\text{doc }} *$ can be easily adapted to non-QA tasks like entity linking and generalizes better on tasks without training sets (WnWi, WnCw).

结果 表4展示了在三项实体链接任务和一项基于知识的对话任务上的结果。在所有任务中，我们发现使用${\mathcal{L}}_{\text{doc }}$的DensePhrases的表现远优于使用${\mathcal{L}}_{\text{phrase }}$的DensePhrases，并且其性能与使用额外大型生成模型来生成文档标题的RAG相当。使用${\mathcal{L}}_{\text{phrase }}$的表现非常差，因为它关注的是短语级别的蕴含关系，而非文档级别的相关性。与多任务版本的DPR（即DPR*）相比，DensePhrases - ${\mathcal{L}}_{\text{doc }} *$可以轻松适应实体链接等非问答任务，并且在没有训练集的任务（WnWi、WnCw）上具有更好的泛化能力。

---

<!-- Footnote -->

${}^{11}$ We follow the same steps described in Petroni et al. (2021) for training the multi-task version of DensePhrases.

${}^{11}$ 我们遵循佩特罗尼等人（Petroni et al., 2021）所描述的相同步骤来训练多任务版本的DensePhrases。

<!-- Footnote -->

---

## 6 DensePhrases as a Multi-Vector Passage Encoder

## 6 作为多向量段落编码器的DensePhrases

In this section, we demonstrate that DensePhrases can be interpreted as a multi-vector passage encoder, which has recently been shown to be very effective for passage retrieval (Luan et al., 2021; Khattab and Zaharia, 2020). Since this type of multi-vector encoding models requires a large disk footprint, we show that we can control the number of vectors per passage (and hence the index size) through filtering. We also introduce quantization techniques to build more efficient phrase retrieval models without a significant performance drop.

在本节中，我们证明了密集短语模型（DensePhrases）可以被解释为一种多向量段落编码器，最近的研究表明，这种编码器在段落检索方面非常有效（栾（Luan）等人，2021年；卡塔布（Khattab）和扎哈里亚（Zaharia），2020年）。由于这种类型的多向量编码模型需要较大的磁盘空间，我们展示了可以通过过滤来控制每个段落的向量数量（从而控制索引大小）。我们还引入了量化技术，以构建更高效的短语检索模型，同时不会显著降低性能。

### 6.1 Multi-Vector Encodings

### 6.1 多向量编码

Since we represent passages not by a single vector, but by a set of phrase vectors (decomposed as token-level start and end vectors, see Lee et al. (2021)), we notice similarities to previous work, which addresses the capacity limitations of dense, fixed-length passage encodings. While these approaches store a fixed number of vectors per passage (Luan et al., 2021; Humeau et al., 2020) or all token-level vectors (Khattab and Zaharia, 2020), phrase retrieval models store a dynamic number of phrase vectors per passage, where many phrases are filtered by a model trained on QA datasets.

由于我们不是用单个向量来表示段落，而是用一组短语向量（分解为词元级别的起始和结束向量，见李（Lee）等人（2021年）），我们注意到这与之前的工作有相似之处，之前的工作解决了密集、固定长度段落编码的容量限制问题。虽然这些方法为每个段落存储固定数量的向量（栾（Luan）等人，2021年；于莫（Humeau）等人，2020年）或所有词元级别的向量（卡塔布（Khattab）和扎哈里亚（Zaharia），2020年），但短语检索模型为每个段落存储动态数量的短语向量，其中许多短语会被在问答数据集上训练的模型过滤掉。

Specifically, Lee et al. (2021) trains a binary classifier (or a phrase filter) to filter phrases based on their phrase representations. This phrase filter is supervised by the answer annotations in QA datasets, hence denotes candidate answer phrases. In our experiment, we tune the filter threshold to control the number of vectors per passage for passage retrieval.

具体而言，李（Lee）等人（2021年）训练了一个二元分类器（或短语过滤器），以根据短语表示来过滤短语。该短语过滤器由问答（QA）数据集中的答案注释进行监督，因此表示候选答案短语。在我们的实验中，我们调整过滤器阈值以控制用于段落检索的每个段落的向量数量。

### 6.2 Efficient Phrase Retrieval

### 6.2 高效短语检索

The multi-vector encoding models as well as ours are prohibitively large since they contain multiple vector representations for every passage in the entire corpus. We introduce a vector quantization-based method that can safely reduce the size of our phrase index, without performance degradation.

多向量编码模型以及我们的模型规模都过大，因为它们为整个语料库中的每个段落都包含多个向量表示。我们引入了一种基于向量量化的方法，该方法可以在不降低性能的情况下安全地减小我们的短语索引的大小。

Optimized product quantization Since the multi-vector encoding models are prohibitively large due to their multiple representations, we further introduce a vector quantization-based method that can safely reduce the size of our phrase index, without performance degradation. We use Product Quantization (PQ) (Jegou et al., 2010) where the original vector space is decomposed into the Cartesian product of subspaces. Using PQ, the memory usage of using $N$ number of $d$ -dimensional centroid vectors reduces from ${Nd}$ to ${N}^{1/M}d$ with $M$ subspaces while each database vector requires ${\log }_{2}N$ bits. Among different variants of PQ, we use Optimized Product Quantization (OPQ) (Ge et al., 2013),which learns an orthogonal matrix $R$ to better decompose the original vector space. See Ge et al. (2013) for more details on OPQ.

优化乘积量化 由于多向量编码模型因具有多种表示形式而规模过大，我们进一步引入了一种基于向量量化的方法，该方法可以在不降低性能的情况下，安全地减小短语索引的大小。我们使用乘积量化（Product Quantization，PQ）（杰古等人，2010年），即将原始向量空间分解为子空间的笛卡尔积。使用PQ时，使用$N$个$d$维质心向量的内存使用量，在有$M$个子空间的情况下，从${Nd}$减少到${N}^{1/M}d$，而每个数据库向量需要${\log }_{2}N$位。在PQ的不同变体中，我们使用优化乘积量化（Optimized Product Quantization，OPQ）（葛等人，2013年），它学习一个正交矩阵$R$以更好地分解原始向量空间。有关OPQ的更多详细信息，请参阅葛等人（2013年）的研究。

<!-- Media -->

<!-- figureText: w/o OPQ #vec/ $p = {28}$ . w/o Query-side fine-tuning Size (GB) - DensePhrase 60 307 -->

<img src="https://cdn.noedgeai.com/01957d26-d273-7c1a-b6dd-f745aa37d7cb_7.jpg?x=857&y=194&w=588&h=407&r=0"/>

Figure 4: Top-5 passage retrieval accuracy on Natural Questions (dev) for different index sizes of DensePhrases. The index size (GB) and the average number of saved vectors per passage $\left( {\# \text{vec}/p}\right)$ are controlled by the filtering threshold $\tau$ . For instance, #vec $/p$ reduces from 28.0 to 5.1 with higher $\tau$ ,which also reduces the index size from 69GB to ${23}\mathrm{{GB}}$ . OPQ: Optimized Product Quantization (Ge et al., 2013).

图4：DensePhrases不同索引大小在自然问题数据集（验证集）上的前5段落检索准确率。索引大小（GB）和每段保存的平均向量数 $\left( {\# \text{vec}/p}\right)$ 由过滤阈值 $\tau$ 控制。例如，随着 $\tau$ 增大，#vec $/p$ 从28.0降至5.1，这也将索引大小从69GB降至 ${23}\mathrm{{GB}}$。OPQ：优化乘积量化（Ge等人，2013年）。

<!-- Media -->

Quantization-aware training While this type of aggressive vector quantization can significantly reduce memory usage, it often comes at the cost of performance degradation due to the quantization loss. To mitigate this problem, we use quantization-aware query-side fine-tuning motivated by the recent successes on quantization-aware training (Jacob et al., 2018). Specifically, during query-side fine-tuning, we reconstruct the phrase vectors using the trained (optimized) product quantizer, which are then used to minimize Eq. (5).

量化感知训练 虽然这种激进的向量量化可以显著减少内存使用，但由于量化损失，它往往会导致性能下降。为缓解这一问题，受量化感知训练近期取得的成功（Jacob等人，2018年）启发，我们采用量化感知的查询端微调方法。具体而言，在查询端微调期间，我们使用训练好（优化后）的乘积量化器重构短语向量，然后用这些向量来最小化公式（5）。

### 6.3 Experimental Results

### 6.3 实验结果

In Figure 4, we present the top-5 passage retrieval accuracy with respect to the size of the phrase index in DensePhrases. First, applying OPQ can reduce the index size of DensePhrases from 307GB to 69GB, while the top-5 retrieval accuracy is poor without quantization-aware query-side fine-tuning. Furthermore,by tuning the threshold $\tau$ for the phrase filter, the number of vectors per each passage (#vec $/p$ ) can be reduced without hurting the performance significantly. The performance improves with a larger number of vectors per passage, which aligns with the findings of multi-vector encoding models (Khattab and Zaharia, 2020; Luan et al., 2021). Our results show that having 8.8 vectors per passage in DensePhrases has similar retrieval accuracy with DPR.

在图4中，我们展示了DensePhrases中前5名段落检索准确率与短语索引大小的关系。首先，应用OPQ（乘积量化优化，Optimized Product Quantization）可以将DensePhrases的索引大小从307GB缩减至69GB，不过在没有进行量化感知的查询端微调时，前5名检索准确率较低。此外，通过调整短语过滤器的阈值$\tau$，可以在不显著影响性能的情况下减少每个段落的向量数量（#vec $/p$）。每个段落的向量数量越多，性能越好，这与多向量编码模型的研究结果一致（卡塔布（Khattab）和扎哈里亚（Zaharia），2020年；栾（Luan）等人，2021年）。我们的结果表明，DensePhrases中每个段落有8.8个向量时，其检索准确率与DPR（密集段落检索器，Dense Passage Retriever）相近。

## 7 Related Work

## 7 相关工作

Text retrieval has a long history in information retrieval, either for serving relevant information to users directly or for feeding them to computationally expensive downstream systems. While traditional research has focused on designing heuristics, such as sparse vector models like TF-IDF and BM25, it has recently become an active area of interest for machine learning researchers. This was precipitated by the emergence of open-domain QA as a standard problem setting (Chen et al., 2017) and the spread of the retriever-reader paradigm (Yang et al., 2019; Nie et al., 2019). The interest has spread to include a more diverse set of downstream tasks, such as fact checking (Thorne et al., 2018), entity-linking (Wu et al., 2020) or dialogue generation (Dinan et al., 2019), where the problems require access to large corpora or knowledge sources. Recently, REALM (Guu et al., 2020) and RAG (retrieval-augmented generation) (Lewis et al., 2020) have been proposed as general-purpose pre-trained models with explicit access to world knowledge through the retriever. There has also been a line of work to integrate text retrieval with structured knowledge graphs (Sun et al., 2018, 2019; Min et al., 2020). We refer to Lin et al. (2020) for a comprehensive overview of neural text retrieval methods.

文本检索在信息检索领域有着悠久的历史，它既可以直接为用户提供相关信息，也可以将信息提供给计算成本高昂的下游系统。传统研究主要集中于设计启发式方法，例如像TF-IDF和BM25这样的稀疏向量模型，但最近它已成为机器学习研究人员关注的一个活跃领域。这一转变是由开放域问答作为标准问题设定的出现（陈等人，2017年）以及检索器 - 阅读器范式的普及（杨等人，2019年；聂等人，2019年）所推动的。这种关注已经扩展到更多样化的下游任务，如事实核查（索恩等人，2018年）、实体链接（吴等人，2020年）或对话生成（迪南等人，2019年），这些任务需要访问大型语料库或知识源。最近，REALM（古等人，2020年）和RAG（检索增强生成）（刘易斯等人，2020年）被提出作为通用的预训练模型，它们通过检索器可以明确访问世界知识。还有一系列工作致力于将文本检索与结构化知识图谱相结合（孙等人，2018年、2019年；闵等人，2020年）。关于神经文本检索方法的全面概述，我们参考林等人（2020年）的研究。

## 8 Conclusion

## 8 结论

In this paper, we show that phrase retrieval models also learn passage retrieval without any modification. By drawing connections between the objectives of DPR and DensePhrases, we provide a better understanding of how phrase retrieval learns passage retrieval, which is also supported by several empirical evaluations on multiple benchmarks. Specifically, phrase-based passage retrieval has better retrieval quality on top $k$ passages when $k$ is small, and this translates to an efficient use of passages for open-domain QA. We also show that DensePhrases can be fine-tuned for more coarse-grained retrieval units, serving as a basis for any retrieval unit. We plan to further evaluate phrase-based passage retrieval on standard information retrieval tasks such as MS MARCO.

在本文中，我们表明短语检索模型无需任何修改即可学习段落检索。通过建立密集段落检索器（DPR）和密集短语检索器（DensePhrases）目标之间的联系，我们能更好地理解短语检索是如何学习段落检索的，这也得到了多个基准测试的实证评估的支持。具体而言，基于短语的段落检索在取前$k$个段落且$k$值较小时具有更好的检索质量，这意味着在开放域问答中能更有效地利用段落。我们还表明，密集短语检索器（DensePhrases）可以针对更粗粒度的检索单元进行微调，为任何检索单元提供基础。我们计划在标准信息检索任务（如微软机器阅读理解数据集（MS MARCO））上进一步评估基于短语的段落检索。

## Acknowledgements

## 致谢

We thank Chris Sciavolino, Xingcheng Yao, the members of the Princeton NLP group, and the anonymous reviewers for helpful discussion and valuable feedback. This research is supported by the James Mi *91 Research Innovation Fund for Data Science and gifts from Apple and Amazon. It was also supported in part by the ICT Creative Consilience program (IITP-2021-0-01819) supervised by the IITP (Institute for Information & communications Technology Planning & Evaluation) and National Research Foundation of Korea (NRF- 2020R1A2C3010638).

我们感谢克里斯·夏沃利诺（Chris Sciavolino）、姚兴成（Xingcheng Yao）、普林斯顿自然语言处理小组（Princeton NLP group）的成员以及匿名审稿人，感谢他们富有启发性的讨论和宝贵的反馈。本研究得到了詹姆斯·米*91数据科学研究创新基金（James Mi *91 Research Innovation Fund for Data Science）以及苹果（Apple）和亚马逊（Amazon）的捐赠支持。部分研究还得到了由韩国信息通信技术规划与评估院（Institute for Information & communications Technology Planning & Evaluation，IITP）监管的信息通信技术创意融合项目（ICT Creative Consilience program，IITP - 2021 - 0 - 01819）以及韩国国家研究基金会（National Research Foundation of Korea，NRF - 2020R1A2C3010638）的支持。

## Ethical Considerations

## 伦理考量

Models introduced in our work often use question answering datasets such as Natural Questions to build phrase or passage representations. Some of the datasets, like SQuAD, are created from a small number of popular Wikipedia articles, hence could make our model biased towards a small number of topics. We hope that inventing an alternative training method that properly regularizes our model could mitigate this problem. Although our efforts have been made to reduce the computational cost of retrieval models, using passage retrieval models as external knowledge bases will inevitably increase the resource requirements for future experiments. Further efforts should be made to make retrieval more affordable for independent researchers.

我们工作中引入的模型通常使用诸如自然问答（Natural Questions）等问答数据集来构建短语或段落表示。一些数据集，如斯坦福问答数据集（SQuAD），是从少数热门维基百科文章中创建的，因此可能会使我们的模型偏向于少数主题。我们希望发明一种适当规范模型的替代训练方法来缓解这个问题。尽管我们已经努力降低检索模型的计算成本，但将段落检索模型用作外部知识库不可避免地会增加未来实验的资源需求。应该进一步努力使独立研究人员能够更经济地进行检索。

## References

## 参考文献

Petr Baudiš and Jan Šedivý. 2015. Modeling of the question answering task in the YodaQA system. In International Conference of the Cross-Language Evaluation Forum for European Languages.

彼得·鲍迪什（Petr Baudiš）和扬·塞迪维（Jan Šedivý）。2015年。尤达问答系统（YodaQA）中问答任务的建模。见欧洲语言跨语言评估论坛国际会议论文集。

Jonathan Berant, Andrew Chou, Roy Frostig, and Percy Liang. 2013. Semantic parsing on Freebase from question-answer pairs. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pages 1533-1544, Seattle, Washington, USA. Association for Computational Linguistics.

乔纳森·贝兰特（Jonathan Berant）、安德鲁·周（Andrew Chou）、罗伊·弗罗斯蒂格（Roy Frostig）和梁珀西（Percy Liang）。2013年。基于问答对在Freebase上进行语义解析。见2013年自然语言处理经验方法会议论文集，第1533 - 1544页，美国华盛顿州西雅图市。计算语言学协会。

Danqi Chen, Adam Fisch, Jason Weston, and Antoine Bordes. 2017. Reading Wikipedia to answer open-domain questions. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1870- 1879, Vancouver, Canada. Association for Computational Linguistics.

陈丹琦（Danqi Chen）、亚当·菲施（Adam Fisch）、杰森·韦斯顿（Jason Weston）和安托万·博尔德斯（Antoine Bordes）。2017年。阅读维基百科以回答开放领域问题。见计算语言学协会第55届年会论文集（第1卷：长论文），第1870 - 1879页，加拿大温哥华市。计算语言学协会。

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171-4186, Minneapolis, Minnesota. Association for Computational Linguistics.

雅各布·德夫林（Jacob Devlin）、张明伟（Ming-Wei Chang）、肯顿·李（Kenton Lee）和克里斯蒂娜·图托纳娃（Kristina Toutanova）。2019年。BERT：用于语言理解的深度双向变换器的预训练。见《2019年北美计算语言学协会人类语言技术分会会议论文集》第1卷（长论文和短论文），第4171 - 4186页，明尼苏达州明尼阿波利斯市。计算语言学协会。

Emily Dinan, Stephen Roller, Kurt Shuster, Angela Fan, Michael Auli, and Jason Weston. 2019. Wizard of wikipedia: Knowledge-powered conversational agents. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. OpenReview.net.

艾米丽·迪南（Emily Dinan）、斯蒂芬·罗勒（Stephen Roller）、库尔特·舒斯特（Kurt Shuster）、安吉拉·范（Angela Fan）、迈克尔·奥利（Michael Auli）和杰森·韦斯顿（Jason Weston）。2019年。维基百科精灵：知识驱动的对话代理。见第七届国际学习表征会议（ICLR 2019），美国路易斯安那州新奥尔良市，2019年5月6 - 9日。OpenReview.net。

Tiezheng Ge, Kaiming He, Qifa Ke, and Jian Sun. 2013. Optimized product quantization. IEEE transactions on pattern analysis and machine intelligence, 36(4):744-755.

葛铁铮（Tiezheng Ge）、何恺明（Kaiming He）、柯其发（Qifa Ke）和孙剑（Jian Sun）。2013年。优化乘积量化。《电气与电子工程师协会模式分析与机器智能汇刊》（IEEE transactions on pattern analysis and machine intelligence），36(4):744 - 755。

Zhaochen Guo and Denilson Barbosa. 2018. Robust named entity disambiguation with random walks. Semantic Web, 9(4):459-479.

郭兆晨（Zhaochen Guo）和德尼尔森·巴尔博萨（Denilson Barbosa）。2018年。基于随机游走的鲁棒命名实体消歧。《语义网》（Semantic Web），9(4):459 - 479。

Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pa-supat, and Ming-Wei Chang. 2020. REALM: Retrieval-augmented language model pre-training. In International Conference on Machine Learning.

凯尔文·顾（Kelvin Guu）、肯顿·李（Kenton Lee）、佐拉·通（Zora Tung）、帕努蓬·帕苏帕特（Panupong Pa - supat）和张明伟（Ming - Wei Chang）。2020年。REALM：检索增强的语言模型预训练。见国际机器学习会议（International Conference on Machine Learning）。

Johannes Hoffart, Mohamed Amir Yosef, Ilaria Bor-dino, Hagen Fürstenau, Manfred Pinkal, Marc Spaniol, Bilyana Taneva, Stefan Thater, and Gerhard Weikum. 2011. Robust disambiguation of named entities in text. In Proceedings of the 2011 Conference on Empirical Methods in Natural Language Processing, pages 782-792, Edinburgh, Scotland, UK. Association for Computational Linguistics.

约翰内斯·霍法特（Johannes Hoffart）、穆罕默德·阿米尔·约瑟夫（Mohamed Amir Yosef）、伊拉里亚·博尔迪诺（Ilaria Bor - dino）、哈根·弗尔斯特瑙（Hagen Fürstenau）、曼弗雷德·平卡尔（Manfred Pinkal）、马克·斯帕尼奥尔（Marc Spaniol）、比利亚娜·塔纳娃（Bilyana Taneva）、斯特凡·塔特（Stefan Thater）和格哈德·魏库姆（Gerhard Weikum）。2011年。文本中命名实体的鲁棒消歧。见2011年自然语言处理经验方法会议论文集（Proceedings of the 2011 Conference on Empirical Methods in Natural Language Processing），第782 - 792页，英国苏格兰爱丁堡。计算语言学协会（Association for Computational Linguistics）。

Samuel Humeau, Kurt Shuster, Marie-Anne Lachaux, and Jason Weston. 2020. Poly-encoders: Architectures and pre-training strategies for fast and accurate multi-sentence scoring. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenRe-view.net.

塞缪尔·于莫（Samuel Humeau）、库尔特·舒斯特（Kurt Shuster）、玛丽 - 安妮·拉绍（Marie - Anne Lachaux）和杰森·韦斯顿（Jason Weston）。2020年。多编码器：用于快速准确多句子评分的架构和预训练策略。见第8届国际学习表征会议（8th International Conference on Learning Representations，ICLR 2020），2020年4月26 - 30日，埃塞俄比亚亚的斯亚贝巴。OpenReview.net。

Gautier Izacard and Edouard Grave. 2021a. Distilling knowledge from reader to retriever for question answering. In International Conference on Learning Representations.

高蒂埃·伊扎卡尔（Gautier Izacard）和爱德华·格雷夫（Edouard Grave）。2021a。在问答任务中从阅读器向检索器进行知识蒸馏。见国际学习表征会议。

Gautier Izacard and Edouard Grave. 2021b. Leveraging passage retrieval with generative models for open domain question answering. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 874-880, Online. Association for Computational Linguistics.

高蒂埃·伊扎卡尔（Gautier Izacard）和爱德华·格雷夫（Edouard Grave）。2021b。在开放领域问答中利用生成模型进行段落检索。见第16届欧洲计算语言学协会分会会议论文集：主卷，第874 - 880页，线上会议。计算语言学协会。

Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew G. Howard, Hartwig Adam, and Dmitry Kalenichenko. 2018. Quantization and training of neural networks for efficient integer-arithmetic-only inference. In 2018 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2018, Salt Lake City, UT, USA, June 18-22, 2018, pages 2704-2713. IEEE Computer Society.

贝努瓦·雅各布（Benoit Jacob）、斯克尔曼塔斯·克利吉斯（Skirmantas Kligys）、陈博（Bo Chen）、朱梦龙（Menglong Zhu）、马修·唐（Matthew Tang）、安德鲁·G·霍华德（Andrew G. Howard）、哈特维希·亚当（Hartwig Adam）和德米特里·卡列尼琴科（Dmitry Kalenichenko）。2018年。用于高效纯整数运算推理的神经网络量化与训练。见《2018年电气与电子工程师协会计算机视觉与模式识别会议（2018 IEEE Conference on Computer Vision and Pattern Recognition，CVPR 2018）论文集》，美国犹他州盐湖城，2018年6月18 - 22日，第2704 - 2713页。电气与电子工程师协会计算机学会。

Herve Jegou, Matthijs Douze, and Cordelia Schmid. 2010. Product quantization for nearest neighbor search. IEEE transactions on pattern analysis and machine intelligence, 33(1):117-128.

埃尔韦·热古（Herve Jegou）、马蒂伊斯·杜泽（Matthijs Douze）和科迪莉亚·施密德（Cordelia Schmid）。2010年。用于最近邻搜索的积量化。《电气与电子工程师协会模式分析与机器智能汇刊》，33(1):117 - 128。

Mandar Joshi, Danqi Chen, Yinhan Liu, Daniel S. Weld, Luke Zettlemoyer, and Omer Levy. 2020. SpanBERT: Improving pre-training by representing and predicting spans. Transactions of the Association for Computational Linguistics, 8:64-77.

曼达尔·乔希（Mandar Joshi）、陈丹琦（Danqi Chen）、刘音涵（Yinhan Liu）、丹尼尔·S·韦尔德（Daniel S. Weld）、卢克·泽特尔莫尔（Luke Zettlemoyer）和奥默·利维（Omer Levy）。2020年。SpanBERT：通过表示和预测片段改进预训练。《计算语言学协会汇刊》，8:64 - 77。

Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. 2017. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1601-1611, Vancouver, Canada. Association for Computational Linguistics.

曼达尔·乔希（Mandar Joshi）、崔恩索尔（Eunsol Choi）、丹尼尔·韦尔德（Daniel Weld）和卢克·泽特尔莫耶（Luke Zettlemoyer）。2017年。《TriviaQA：一个用于阅读理解的大规模远程监督挑战数据集》。发表于第55届计算语言学协会年会论文集（第1卷：长论文），第1601 - 1611页，加拿大温哥华。计算语言学协会。

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6769- 6781, Online. Association for Computational Linguistics.

弗拉基米尔·卡尔普欣（Vladimir Karpukhin）、巴拉斯·奥古兹（Barlas Oguz）、闵世元（Sewon Min）、帕特里克·刘易斯（Patrick Lewis）、莱德尔·吴（Ledell Wu）、谢尔盖·叶杜诺夫（Sergey Edunov）、陈丹琦（Danqi Chen）和易文涛（Wen - tau Yih）。2020年。《用于开放域问答的密集段落检索》。发表于2020年自然语言处理经验方法会议（EMNLP）论文集，第6769 - 6781页，线上会议。计算语言学协会。

Omar Khattab and Matei Zaharia. 2020. Colbert: Efficient and effective passage search via contextual-ized late interaction over BERT. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, SI-GIR 2020, Virtual Event, China, July 25-30, 2020, pages 39-48. ACM.

奥马尔·哈塔卜（Omar Khattab）和马泰·扎哈里亚（Matei Zaharia）。2020年。科尔伯特（Colbert）：通过基于BERT的上下文延迟交互实现高效有效的段落搜索。收录于第43届ACM信息检索研究与发展国际会议（43rd International ACM SIGIR conference on research and development in Information Retrieval）论文集，SIGIR 2020，线上会议，中国，2020年7月25 - 30日，第39 - 48页。美国计算机协会（ACM）。

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural questions: A benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:452-466.

汤姆·夸特科夫斯基（Tom Kwiatkowski）、珍妮玛丽亚·帕洛马基（Jennimaria Palomaki）、奥利维亚·雷德菲尔德（Olivia Redfield）、迈克尔·柯林斯（Michael Collins）、安库尔·帕里克（Ankur Parikh）、克里斯·阿尔伯蒂（Chris Alberti）、丹妮尔·爱泼斯坦（Danielle Epstein）、伊利亚·波洛苏欣（Illia Polosukhin）、雅各布·德夫林（Jacob Devlin）、肯顿·李（Kenton Lee）、克里斯蒂娜·图托纳娃（Kristina Toutanova）、利翁·琼斯（Llion Jones）、马修·凯尔西（Matthew Kelcey）、张明伟（Ming-Wei Chang）、安德鲁·M·戴（Andrew M. Dai）、雅各布·乌兹科雷特（Jakob Uszkoreit）、乐存（Quoc Le）和斯拉夫·彼得罗夫（Slav Petrov）。2019年。自然问题：问答研究的基准。计算语言学协会汇刊（Transactions of the Association for Computational Linguistics），7：452 - 466。

Jinhyuk Lee, Minjoon Seo, Hannaneh Hajishirzi, and Jaewoo Kang. 2020. Contextualized sparse representations for real-time open-domain question answering. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 912-919, Online. Association for Computational Linguistics.

李镇赫（Jinhyuk Lee）、徐民俊（Minjoon Seo）、汉娜·哈吉希尔齐（Hannaneh Hajishirzi）和姜在宇（Jaewoo Kang）。2020年。用于实时开放域问答的上下文稀疏表示。见《第58届计算语言学协会年会论文集》，第912 - 919页，线上会议。计算语言学协会。

Jinhyuk Lee, Mujeen Sung, Jaewoo Kang, and Danqi Chen. 2021. Learning dense representations of phrases at scale. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 6634-6647, Online. Association for Computational Linguistics.

李镇赫（Jinhyuk Lee）、成武珍（Mujeen Sung）、姜在宇（Jaewoo Kang）和陈丹琦（Danqi Chen）。2021年。大规模学习短语的密集表示。见《第59届计算语言学协会年会暨第11届自然语言处理国际联合会议论文集（第1卷：长论文）》，第6634 - 6647页，线上会议。计算语言学协会。

Kenton Lee, Ming-Wei Chang, and Kristina Toutanova. 2019. Latent retrieval for weakly supervised open domain question answering. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 6086-6096, Florence, Italy. Association for Computational Linguistics.

肯顿·李（Kenton Lee）、张明伟（Ming-Wei Chang）和克里斯蒂娜·图托纳娃（Kristina Toutanova）。2019年。弱监督开放域问答的潜在检索。见《第57届计算语言学协会年会论文集》，第6086 - 6096页，意大利佛罗伦萨。计算语言学协会。

Patrick S. H. Lewis, Ethan Perez, Aleksandra Pik-tus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledge-intensive NLP tasks. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual.

帕特里克·S·H·刘易斯（Patrick S. H. Lewis）、伊桑·佩雷斯（Ethan Perez）、亚历山德拉·皮克图斯（Aleksandra Pik-tus）、法比奥·彼得罗尼（Fabio Petroni）、弗拉基米尔·卡尔普欣（Vladimir Karpukhin）、纳曼·戈亚尔（Naman Goyal）、海因里希·屈特勒（Heinrich Küttler）、迈克·刘易斯（Mike Lewis）、文涛·伊（Wen-tau Yih）、蒂姆·罗克塔舍尔（Tim Rocktäschel）、塞巴斯蒂安·里德尔（Sebastian Riedel）和杜韦·基拉（Douwe Kiela）。2020年。用于知识密集型自然语言处理任务的检索增强生成。《神经信息处理系统进展33：2020年神经信息处理系统年度会议（NeurIPS 2020）》，2020年12月6 - 12日，线上会议。

Jimmy Lin, Rodrigo Nogueira, and Andrew Yates. 2020. Pretrained transformers for text ranking: BERT and beyond. arXiv preprint arXiv:2010.06467.

吉米·林（Jimmy Lin）、罗德里戈·诺盖拉（Rodrigo Nogueira）和安德鲁·耶茨（Andrew Yates）。2020年。用于文本排序的预训练Transformer：BERT及其他。arXiv预印本arXiv:2010.06467。

Yi Luan, Jacob Eisenstein, Kristina Toutanova, and Michael Collins. 2021. Sparse, dense, and attentional representations for text retrieval. Transactions of the Association for Computational Linguistics, 9:329-345.

栾义（Yi Luan）、雅各布·艾森斯坦（Jacob Eisenstein）、克里斯蒂娜·图托纳娃（Kristina Toutanova）和迈克尔·柯林斯（Michael Collins）。2021年。用于文本检索的稀疏、密集和注意力表示。《计算语言学协会汇刊》，9：329 - 345。

Sewon Min, Danqi Chen, Luke Zettlemoyer, and Han-naneh Hajishirzi. 2020. Knowledge guided text retrieval and reading for open domain question answering. ArXiv preprint, abs/1911.03868.

闵世元（Sewon Min）、陈丹琦（Danqi Chen）、卢克·泽特尔莫耶（Luke Zettlemoyer）和汉娜·哈吉希尔齐（Han-naneh Hajishirzi）。2020年。用于开放域问答的知识引导文本检索与阅读。预印本，arXiv:1911.03868。

Yixin Nie, Songhe Wang, and Mohit Bansal. 2019. Revealing the importance of semantic retrieval for machine reading at scale. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2553-2566, Hong Kong, China. Association for Computational Linguistics.

聂一新（Yixin Nie）、王松鹤（Songhe Wang）和莫希特·班萨尔（Mohit Bansal）。2019年。揭示语义检索在大规模机器阅读中的重要性。见《2019年自然语言处理经验方法会议和第9届自然语言处理国际联合会议论文集》（EMNLP - IJCNLP），第2553 - 2566页，中国香港。计算语言学协会。

Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yazdani, Nicola De Cao, James Thorne, Yacine Jernite, Vladimir Karpukhin, Jean Maillard, Vassilis Plachouras, Tim Rocktäschel, and Sebastian Riedel. 2021. KILT: a benchmark for knowledge intensive language tasks. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2523-2544, Online. Association for Computational Linguistics.

法比奥·彼得罗尼（Fabio Petroni）、亚历山德拉·皮克图斯（Aleksandra Piktus）、安吉拉·范（Angela Fan）、帕特里克·刘易斯（Patrick Lewis）、马吉德·亚兹达尼（Majid Yazdani）、尼古拉·德·曹（Nicola De Cao）、詹姆斯·索恩（James Thorne）、亚辛·杰尔尼（Yacine Jernite）、弗拉基米尔·卡尔普欣（Vladimir Karpukhin）、让·梅拉德（Jean Maillard）、瓦西利斯·普拉乔拉斯（Vassilis Plachouras）、蒂姆·罗克塔舍尔（Tim Rocktäschel）和塞巴斯蒂安·里德尔（Sebastian Riedel）。2021年。KILT：知识密集型语言任务基准。见《2021年计算语言学协会北美分会人类语言技术会议论文集》，第2523 - 2544页，线上会议。计算语言学协会。

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou,

科林·拉菲尔（Colin Raffel）、诺姆·沙泽尔（Noam Shazeer）、亚当·罗伯茨（Adam Roberts）、凯瑟琳·李（Katherine Lee）、沙兰·纳朗（Sharan Narang）、迈克尔·马泰纳（Michael Matena）、周燕琪（Yanqi Zhou）

Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21:1-67.

李伟（Wei Li）和彼得·J·刘（Peter J Liu）。2020年。用统一的文本到文本转换器探索迁移学习的极限。《机器学习研究杂志》，21:1 - 67。

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. SQuAD: 100,000+ questions for machine comprehension of text. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 2383-2392, Austin, Texas. Association for Computational Linguistics.

普拉纳夫·拉杰普尔卡（Pranav Rajpurkar）、张健（Jian Zhang）、康斯坦丁·洛皮列夫（Konstantin Lopyrev）和珀西·梁（Percy Liang）。2016年。SQuAD：用于文本机器理解的10万多个问题。见《2016年自然语言处理经验方法会议论文集》，第2383 - 2392页，美国得克萨斯州奥斯汀市。计算语言学协会。

Stephen Robertson and Hugo Zaragoza. 2009. The probabilistic relevance framework: $\mathrm{{Bm}}{25}$ and beyond. Foundations and Trends $\circledast$ in Information Retrieval, 3(4):333-389.

斯蒂芬·罗伯逊（Stephen Robertson）和雨果·萨拉戈萨（Hugo Zaragoza）。2009年。概率相关性框架：$\mathrm{{Bm}}{25}$及超越。《信息检索基础与趋势》$\circledast$，3(4)：333 - 389。

Minjoon Seo, Tom Kwiatkowski, Ankur Parikh, Ali Farhadi, and Hannaneh Hajishirzi. 2018. Phrase-indexed question answering: A new challenge for scalable document comprehension. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 559-564, Brussels, Belgium. Association for Computational Linguistics.

闵俊勋（Minjoon Seo）、汤姆·克维亚特科夫斯基（Tom Kwiatkowski）、安库尔·帕里克（Ankur Parikh）、阿里·法尔哈迪（Ali Farhadi）和汉娜内·哈吉希尔齐（Hannaneh Hajishirzi）。2018年。短语索引问答：可扩展文档理解的新挑战。见《2018年自然语言处理经验方法会议论文集》，第559 - 564页，比利时布鲁塞尔。计算语言学协会。

Minjoon Seo, Jinhyuk Lee, Tom Kwiatkowski, Ankur Parikh, Ali Farhadi, and Hannaneh Hajishirzi. 2019. Real-time open-domain question answering with dense-sparse phrase index. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4430-4441, Florence, Italy. Association for Computational Linguistics.

徐敏俊（Minjoon Seo）、李镇赫（Jinhyuk Lee）、汤姆·夸特科夫斯基（Tom Kwiatkowski）、安库尔·帕里克（Ankur Parikh）、阿里·法尔哈迪（Ali Farhadi）和汉娜内·哈吉希尔齐（Hannaneh Hajishirzi）。2019年。基于密集-稀疏短语索引的实时开放域问答。见《第57届计算语言学协会年会论文集》，第4430 - 4441页，意大利佛罗伦萨。计算语言学协会。

Haitian Sun, Tania Bedrax-Weiss, and William Cohen. 2019. PullNet: Open domain question answering with iterative retrieval on knowledge bases and text. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2380- 2390, Hong Kong, China. Association for Computational Linguistics.

孙海天（Haitian Sun）、塔尼亚·贝德拉克斯 - 韦斯（Tania Bedrax-Weiss）和威廉·科恩（William Cohen）。2019年。PullNet：基于知识库和文本迭代检索的开放域问答。见《2019年自然语言处理经验方法会议和第9届自然语言处理国际联合会议（EMNLP - IJCNLP）论文集》，第2380 - 2390页，中国香港。计算语言学协会。

Haitian Sun, Bhuwan Dhingra, Manzil Zaheer, Kathryn Mazaitis, Ruslan Salakhutdinov, and William Cohen. 2018. Open domain question answering using early fusion of knowledge bases and text. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 4231-4242, Brussels, Belgium. Association for Computational Linguistics.

海田·孙（Haitian Sun）、布万·丁格拉（Bhuwan Dhingra）、曼齐尔·扎希尔（Manzil Zaheer）、凯瑟琳·马扎蒂斯（Kathryn Mazaitis）、鲁斯兰·萨拉胡季诺夫（Ruslan Salakhutdinov）和威廉·科恩（William Cohen）。2018年。利用知识库和文本的早期融合进行开放领域问答。见《2018年自然语言处理经验方法会议论文集》，第4231 - 4242页，比利时布鲁塞尔。计算语言学协会。

James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. 2018. FEVER: a large-scale dataset for fact extraction and VERification. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 809-819, New Orleans, Louisiana. Association for Computational Linguistics.

詹姆斯·索恩（James Thorne）、安德里亚斯·弗拉乔斯（Andreas Vlachos）、克里斯托斯·克里斯托杜洛普洛斯（Christos Christodoulopoulos）和阿尔皮特·米塔尔（Arpit Mittal）。2018年。FEVER：一个用于事实提取和验证的大规模数据集。见《2018年计算语言学协会北美分会会议：人类语言技术》论文集，第1卷（长论文），第809 - 819页，美国路易斯安那州新奥尔良。计算语言学协会。

Ledell Wu, Fabio Petroni, Martin Josifoski, Sebastian Riedel, and Luke Zettlemoyer. 2020. Scalable zero-shot entity linking with dense entity retrieval. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6397-6407, Online. Association for Computational Linguistics.

莱德尔·吴（Ledell Wu）、法比奥·彼得罗尼（Fabio Petroni）、马丁·约西福斯基（Martin Josifoski）、塞巴斯蒂安·里德尔（Sebastian Riedel）和卢克·泽特尔莫耶（Luke Zettlemoyer）。2020年。基于密集实体检索的可扩展零样本实体链接。见《2020年自然语言处理经验方法会议（EMNLP）论文集》，第6397 - 6407页，线上会议。计算语言学协会。

Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul N. Bennett, Junaid Ahmed, and Arnold Overwijk. 2021. Approximate nearest neighbor negative contrastive learning for dense text retrieval. In International Conference on Learning Representations.

李雄（Lee Xiong）、熊晨彦（Chenyan Xiong）、李晔（Ye Li）、邓国峰（Kwok - Fung Tang）、刘佳琳（Jialin Liu）、保罗·N·贝内特（Paul N. Bennett）、朱奈德·艾哈迈德（Junaid Ahmed）和阿诺德·奥弗维克（Arnold Overwijk）。2021年。用于密集文本检索的近似最近邻负对比学习。见国际学习表征会议。

Sohee Yang and Minjoon Seo. 2020. Is retriever merely an approximator of reader? ArXiv preprint, abs/2010.10999.

杨素熙（Sohee Yang）和徐敏俊（Minjoon Seo）。2020年。检索器仅仅是阅读器的近似器吗？预印本，arXiv:2010.10999。

Wei Yang, Yuqing Xie, Aileen Lin, Xingyu Li, Luchen Tan, Kun Xiong, Ming Li, and Jimmy Lin. 2019. End-to-end open-domain question answering with BERTserini. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations), pages 72-77, Minneapolis, Minnesota. Association for Computational Linguistics.

魏洋（Wei Yang）、谢雨青（Yuqing Xie）、艾琳·林（Aileen Lin）、李星宇（Xingyu Li）、谭璐晨（Luchen Tan）、熊坤（Kun Xiong）、李明（Ming Li）和吉米·林（Jimmy Lin）。2019年。使用BERTserini实现端到端开放域问答。见《2019年北美计算语言学协会分会会议论文集（演示）》，第72 - 77页，美国明尼苏达州明尼阿波利斯市。计算语言学协会。