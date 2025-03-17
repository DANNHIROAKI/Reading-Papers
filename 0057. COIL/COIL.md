# COIL: Revisit Exact Lexical Match in Information Retrieval with Contextualized Inverted List

# COIL：借助上下文倒排列表重新审视信息检索中的精确词汇匹配

Luyu Gao, Zhuyun Dai, Jamie Callan

高璐宇（Luyu Gao）、戴竹云（Zhuyun Dai）、杰米·卡兰（Jamie Callan）

Language Technologies Institute

语言技术研究所

Carnegie Mellon University

卡内基梅隆大学（Carnegie Mellon University）

\{luyug, zhuyund, callan\}@cs.cmu.edu

\{luyug, zhuyund, callan\}@cs.cmu.edu

## Abstract

## 摘要

Classical information retrieval systems such as BM25 rely on exact lexical match and carry out search efficiently with inverted list index. Recent neural IR models shifts towards soft semantic matching all query document terms, but they lose the computation efficiency of exact match systems. This paper presents COIL, a contextualized exact match retrieval architecture that brings semantic lexical matching. COIL scoring is based on overlapping query document tokens' contextualized representations. The new architecture stores con-textualized token representations in inverted lists, bringing together the efficiency of exact match and the representation power of deep language models. Our experimental results show COIL outperforms classical lexical retrievers and state-of-the-art deep LM retrievers with similar or smaller latency. ${}^{1}$

像BM25（最佳匹配25算法）这样的经典信息检索系统依赖于精确的词法匹配，并通过倒排索引列表高效地执行搜索。最近的神经信息检索（IR）模型转向对所有查询文档术语进行软语义匹配，但它们失去了精确匹配系统的计算效率。本文提出了COIL（上下文感知精确匹配检索架构），这是一种引入语义词法匹配的上下文感知精确匹配检索架构。COIL评分基于查询文档标记的上下文表示的重叠。新架构将上下文标记表示存储在倒排列表中，结合了精确匹配的效率和深度语言模型的表示能力。我们的实验结果表明，COIL在延迟相似或更低的情况下，性能优于经典的词法检索器和最先进的深度语言模型（LM）检索器。${}^{1}$

## 1 Introduction

## 1 引言

Widely used, bag-of-words (BOW) information retrieval (IR) systems such as BM25 rely on exact lexical match ${}^{2}$ between query and document terms. Recent study in neural IR takes a different approach and compute soft matching between all query and document terms to model complex matching.

广泛使用的词袋（BOW）信息检索（IR）系统，如BM25（最佳匹配25算法），依赖于查询和文档术语之间的精确词法匹配${}^{2}$。最近对神经信息检索的研究采用了不同的方法，计算所有查询和文档术语之间的软匹配，以对复杂匹配进行建模。

The shift to soft matching in neural IR models attempts to address vocabulary mismatch problems, that query and the relevant documents use different terms, e.g. cat v.s. kitty, for the same concept (Huang et al., 2013; Guo et al., 2016; Xiong et al., 2017). Later introduction of contextualized representations (Peters et al., 2018) from deep language models (LM) further address semantic mismatch, that the same term can refer to different concepts, e.g., bank of river vs. bank in finance. Fine-tuned deep LM rerankers produce token representations based on context and achieve state-of-the-art in text ranking with huge performance leap (Nogueira and Cho, 2019; Dai and Callan, 2019b).

神经信息检索（IR）模型向软匹配的转变旨在解决词汇不匹配问题，即查询和相关文档针对同一概念使用不同的术语，例如“cat（猫）”和“kitty（小猫）”（Huang等人，2013年；Guo等人，2016年；Xiong等人，2017年）。后来，从深度语言模型（LM）引入上下文表示（Peters等人，2018年）进一步解决了语义不匹配问题，即同一术语可能指代不同的概念，例如“river bank（河岸）”和“finance bank（金融银行）”。经过微调的深度语言模型重排器基于上下文生成词元表示，并在文本排序方面取得了巨大的性能飞跃，达到了当前的最优水平（Nogueira和Cho，2019年；Dai和Callan，2019b）。

Though the idea of soft matching all tokens is carried through the development of neural IR models, seeing the success brought by deep LMs, we take a step back and ask: how much gain can we get if we introduce contextualized representations back to lexical exact match systems? In other words, can we build a system that still performs exact query-document token matching but compute matching signals with contextualized token representations instead of heuristics? This may seem a constraint on the model, but exact lexical match produce more explainable and controlled patterns than soft matching. It also allows search to focus on only the subset of documents that have overlapping terms with query, which can be done efficiently with inverted list index. Meanwhile, using dense contex-tualized token representations enables the model to handle semantic mismatch, which has been a long-standing problem in classic lexical systems.

尽管在神经信息检索（IR）模型的发展过程中始终贯彻着对所有词元进行软匹配的理念，但鉴于深度语言模型（LM）所带来的成功，我们退一步思考：如果将上下文相关的表示重新引入词法精确匹配系统，我们能获得多大的收益呢？换句话说，我们能否构建一个仍然执行查询 - 文档词元精确匹配的系统，但使用上下文相关的词元表示而非启发式方法来计算匹配信号呢？这看似是对模型的一种限制，但精确的词法匹配比软匹配能产生更具可解释性和可控性的模式。它还能让搜索仅聚焦于与查询有重叠词项的文档子集，这可以通过倒排索引高效地实现。同时，使用密集的上下文相关词元表示能使模型处理语义不匹配问题，而这一直是经典词法系统中的一个长期难题。

To answer the question, we propose a new lexical matching scheme that uses vector similarities between query-document overlapping term contex-tualized representations to replace heuristic scoring used in classical systems. We present COn-textualized Inverted List (COIL), a new exact lexical match retrieval architecture armed with deep LM representations. COIL processes documents with deep LM offline and produces representations for each document token. The representations are grouped by their surface tokens into inverted lists. At search time, we build representation vectors for query tokens and perform contextualized exact match: use each query token to look up its own inverted list and compute vector similarity with document vectors stored in the inverted list as matching scores. COIL enables efficient search with rich-in-semantic matching between query and document.

为了回答这个问题，我们提出了一种新的词法匹配方案，该方案使用查询 - 文档重叠词的上下文表示之间的向量相似度，来替代经典系统中使用的启发式评分。我们介绍了上下文倒排表（Contextualized Inverted List，COIL），这是一种配备了深度语言模型（LM）表示的新型精确词法匹配检索架构。COIL 离线使用深度语言模型处理文档，并为每个文档标记生成表示。这些表示按其表面标记分组到倒排表中。在搜索时，我们为查询标记构建表示向量，并执行上下文精确匹配：使用每个查询标记查找其自己的倒排表，并计算与存储在倒排表中的文档向量的向量相似度，作为匹配分数。COIL 能够在查询和文档之间进行高效搜索，并实现丰富的语义匹配。

Our contributions include 1) introduce a novel retrieval architecture, contextualized inverted lists (COIL) that brings semantic matching into lexical IR systems, 2) show matching signals induced from exact lexical match can capture complicated matching patterns, 3) demonstrate COIL significantly outperform classical and deep LM augmented lexical retrievers as well as state-of-the-art dense retrievers on two retrieval tasks.

我们的贡献包括：1）引入一种新颖的检索架构，即上下文倒排表（COIL），将语义匹配引入词法信息检索（IR）系统；2）表明从精确词法匹配中得出的匹配信号可以捕捉复杂的匹配模式；3）证明在两项检索任务中，COIL 显著优于经典和深度语言模型增强的词法检索器以及最先进的密集检索器。

---

<!-- Footnote -->

${}^{1}$ Our code is available at https://github.com/ luyug/COIL.

${}^{1}$ 我们的代码可在 https://github.com/ luyug/COIL 上获取。

${}^{2}$ Exact match up to morphological changes.

${}^{2}$ 直至词法变化的精确匹配。

<!-- Footnote -->

---

## 2 Related Work

## 2 相关工作

Lexical Retriever Classical IR systems rely on exact lexical match retrievers such as Boolean Retrieval, BM25 (Robertson and Walker, 1994) and statistical language models (Lafferty and Zhai, 2001). This type of retrieval model can process queries very quickly by organizing the documents into inverted index, where each distinct term has an inverted list that stores information about documents it appears in. Nowadays, they are still widely used in production systems. However, these retrieval models fall short of matching related terms (vocabulary mismatch) or modeling context of the terms (semantic mismatch). Much early effort was put into improving exact lexical match retrievers, such as matching n-grams (Metzler and Croft, 2005) or expanding queries with terms from related documents (Lavrenko and Croft, 2001). However, these methods still use BOW framework and have limited capability of modeling human languages.

词法检索器 经典的信息检索（IR）系统依赖于精确的词法匹配检索器，如布尔检索、BM25算法（罗伯逊和沃克，1994年）和统计语言模型（拉弗蒂和翟成祥，2001年）。这类检索模型通过将文档组织成倒排索引，可以非常快速地处理查询，其中每个不同的词项都有一个倒排列表，用于存储该词项出现的文档信息。如今，它们仍然广泛应用于生产系统中。然而，这些检索模型在匹配相关词项（词汇不匹配）或对词项的上下文进行建模（语义不匹配）方面存在不足。早期有很多工作致力于改进精确的词法匹配检索器，例如匹配n元语法（梅茨勒和克罗夫特，2005年）或用相关文档中的词项扩展查询（拉夫连科和克罗夫特，2001年）。然而，这些方法仍然使用词袋（BOW）框架，对人类语言进行建模的能力有限。

Neural Ranker In order to deal with vocabulary mismatch, neural retrievers that rely on soft matching between numerical text representations are introduced. Early attempts compute similarity between pre-trained word embedding such as word2vec (Mikolov et al., 2013) and GLoVe (Pennington et al., 2014) to produce matching score (Ganguly et al., 2015; Diaz et al., 2016). One more recent approach encodes query and document each into a vector and computes vector similarity (Huang et al., 2013). Later researches realized the limited capacity of a single vector to encode fine-grained information and introduced full interaction models to perform soft matching between all term vectors (Guo et al., 2016; Xiong et al., 2017). In these approaches, scoring is based on learned neural networks and the hugely increased computation cost limited their use to reranking a top candidate list generated by a lexical retriever.

神经排序器 为了处理词汇不匹配的问题，引入了依赖于数字文本表示之间软匹配的神经检索器。早期的尝试是计算预训练词嵌入（如word2vec（米科洛夫等人，2013年）和GLoVe（彭宁顿等人，2014年））之间的相似度，以得出匹配分数（甘古利等人，2015年；迪亚兹等人，2016年）。一种较新的方法是将查询和文档分别编码为向量，并计算向量相似度（黄等人，2013年）。后来的研究意识到单个向量在编码细粒度信息方面的能力有限，于是引入了全交互模型，以在所有词项向量之间进行软匹配（郭等人，2016年；熊等人，2017年）。在这些方法中，评分基于学习到的神经网络，而大幅增加的计算成本限制了它们仅用于对词法检索器生成的候选列表进行重排序。

Deep LM Based Ranker and Retriever Deep LM made a huge impact on neural IR. Fine-tuned Transformer (Vaswani et al., 2017) LM BERT (Devlin et al., 2019) achieved state-of-the-art reranking performance for passages and documents (Nogueira and Cho, 2019; Dai and Callan, 2019b). As illustrated in Figure 1a, the common approach is to feed the concatenated query document text through BERT and use BERT's [CLS] output token to produce a relevance score. The deep LM rerankers addressed both vocabulary and semantic mismatch by computing full cross attention between contextualized token representations. Lighter deep LM rankers are developed (MacA-vaney et al., 2020; Gao et al., 2020), but their cross attention operations are still too expensive for full-collection retrieval.

基于深度语言模型的排序器和检索器 深度语言模型（Deep LM）对神经信息检索（neural IR）产生了巨大影响。经过微调的Transformer（瓦希尼等人，2017年）语言模型BERT（德夫林等人，2019年）在段落和文档的重排序性能方面达到了最先进水平（诺盖拉和赵，2019年；戴和卡兰，2019b）。如图1a所示，常见的方法是将拼接后的查询文档文本输入BERT，并使用BERT的[CLS]输出标记来生成相关性得分。深度语言模型重排序器通过计算上下文标记表示之间的全交叉注意力，解决了词汇和语义不匹配的问题。人们开发了更轻量级的深度语言模型排序器（麦卡瓦尼等人，2020年；高等人，2020年），但它们的交叉注意力操作对于全集合检索来说仍然成本过高。

Later research therefore resorted to augmenting lexical retrieval with deep LMs by expanding the document surface form to narrow the vocabulary gap, e.g., DocT5Query (Nogueira and Lin, 2019), or altering term weights to emphasize important terms, e.g., DeepCT (Dai and Callan, 2019a). Smartly combining deep LM retriever and reranker can offer additive gain for end performance (Gao et al., 2021a). These retrievers however still suffer from vocabulary and semantic mismatch as traditional lexical retrievers.

因此，后续研究通过扩展文档的表层形式以缩小词汇差距，借助深度语言模型（LM）增强词法检索，例如DocT5Query（诺盖拉和林，2019年）；或者改变词项权重以强调重要词项，例如DeepCT（戴和卡兰，2019a）。巧妙地将深度语言模型检索器和重排器结合起来，可以为最终性能带来额外的提升（高等人，2021a）。然而，这些检索器和传统的词法检索器一样，仍然存在词汇和语义不匹配的问题。

Another line of research continues the work on single vector representation and build dense retrievers, as illustrated in Figure 1b. They store document vectors in a dense index and retrieve them through Nearest Neighbours search. Using deep LMs, dense retrievers have achieved promising results on several retrieval tasks (Karpukhin et al., 2020). Later researches show that dense retrieval systems can be further improved by better training (Xiong et al., 2020; Gao et al., 2021b).

另一类研究延续了单向量表示的工作，并构建了密集检索器，如图1b所示。它们将文档向量存储在一个密集索引中，并通过最近邻搜索来检索这些向量。利用深度语言模型，密集检索器在多个检索任务中取得了有前景的结果（卡尔普欣等人，2020年）。后续研究表明，通过更好的训练可以进一步改进密集检索系统（熊等人，2020年；高等人，2021b）。

Single vector systems have also been extended to multi-vector representation systems. Poly-encoder (Humeau et al., 2020) encodes queries into a set of vectors. Similarly, Me-BERT (Luan et al., 2020) represents documents with a set of vectors. A concurrent work ColBERT (Figure 1c) use multiple vectors to encode both queries and documents (Khattab and Zaharia, 2020). In particular, it represents a documents with all its terms' vectors and a query with an expanded set of term vectors. It then computes all-to-all (Cartesian) soft match between the tokens. ColBERT performs interaction as dot product followed pooling operations, which allows it to also leverage a dense index to do full corpus retrieval. However, since ColBERT encodes a document with all tokens, it adds another order of magnitude of index complexity to all aforementioned methods: document tokens in the collection need to be stored in a single huge index and considered at query time. Consequently, ColBERT is engineering and hardware demanding.

单向量系统也已扩展到多向量表示系统。多编码器（Poly-encoder，休莫等人，2020年）将查询编码为一组向量。同样，Me-BERT（栾等人，2020年）用一组向量表示文档。同期的一项工作ColBERT（图1c）使用多个向量对查询和文档进行编码（卡塔布和扎哈里亚，2020年）。具体而言，它用文档所有词项的向量表示文档，用一组扩展的词项向量表示查询。然后，它计算标记之间的全对全（笛卡尔积）软匹配。ColBERT通过点积和池化操作进行交互，这使其也能利用密集索引进行全语料库检索。然而，由于ColBERT用所有标记对文档进行编码，它比上述所有方法的索引复杂度又增加了一个数量级：集合中的文档标记需要存储在一个巨大的单一索引中，并在查询时进行考虑。因此，ColBERT对工程和硬件要求较高。

<!-- Media -->

<!-- figureText: score CLS bank account bank bank CLS account CLS bank CLS bank account bank river bank (b) Dense Retrievers (e.g., DPR) account CLS bank river bank CLS bank account bank river bank CLS bank account CLS bank (d) COIL: Contextualized Exact Match CLS bank account SEP bank bank CLS bank account SEP bank bank CLS account SEP bank CLS bank account SEP bank bank (a) Cross-Attention Model (e.g., BERT reranker) account CLS bank bank CLS bank account CLS (c) ColBERT: All-to-All Match -->

<img src="https://cdn.noedgeai.com/01957d2e-dacf-7190-a6f0-cc115f91897c_2.jpg?x=232&y=182&w=1183&h=622&r=0"/>

Figure 1: An illustration of reranking/retrieval mechanisms with deep LM, including our proposed model, COIL.

图1：使用深度语言模型（LM）进行重排序/检索机制的示意图，包括我们提出的模型COIL。

<!-- figureText: Bank BM25 tf: 1 Bank tf: 1 tf: 2 Account scoring Query River Account- Traditional Inverted Lists -->

<img src="https://cdn.noedgeai.com/01957d2e-dacf-7190-a6f0-cc115f91897c_2.jpg?x=186&y=904&w=580&h=361&r=0"/>

Figure 2: An illustration of traditional inverted lists. The inverted list maps a term to the list of documents where the term occurs. Retriever looks up query terms' inverted lists and scores those documents with stored statistics such as term frequency (tf).

图2：传统倒排索引列表的示意图。倒排索引列表将一个词项映射到该词项出现的文档列表。检索器会查找查询词项的倒排索引列表，并使用存储的统计信息（如词频（tf））对这些文档进行打分。

<!-- Media -->

## 3 Methodologies

## 3 方法

In this section, we first provide some preliminaries on exact lexical match systems. Then we discuss COIL's contextualized exact match design and how its search index is organized. We also give a comparison between COIL and other popular retrievers.

在本节中，我们首先介绍精确词汇匹配系统的一些预备知识。然后讨论COIL（上下文感知倒排索引，Contextualized Inverted List）的上下文精确匹配设计以及其搜索索引的组织方式。我们还将对COIL和其他流行的检索器进行比较。

<!-- Media -->

<!-- figureText: CLS vectors matrix product CLS product Bank Account matrix product docid [1367] Bank $\rightarrow$ vectors docid [124 5 5 9] River vectors docid [339] Contextualized Inverted Lists -->

<img src="https://cdn.noedgeai.com/01957d2e-dacf-7190-a6f0-cc115f91897c_2.jpg?x=851&y=904&w=608&h=512&r=0"/>

Figure 3: COIL's index and retrieval architecture. COIL-tok relies on the exact token matching (lower). COIL-full includes in addition CLS matching (upper).

图3：COIL的索引和检索架构。COIL-tok依赖于精确的词元匹配（下方）。COIL-full还包括CLS匹配（上方）。

<!-- Media -->

### 3.1 Preliminaries

### 3.1 预备知识

Classic lexical retrieval system relies on overlapping query document terms under morphological generalization like stemming, in other words, exact lexical match, to score query document pair. A scoring function is defined as a sum of matched term scores. The scores are usually based on statistics like term frequency(tf). Generally,we can write,

经典的词汇检索系统依赖于在词法泛化（如词干提取）下查询文档词项的重叠，换句话说，即精确的词汇匹配，来对查询 - 文档对进行打分。打分函数定义为匹配词项得分的总和。这些得分通常基于词频（tf）等统计信息。一般来说，我们可以写成：

$$
s = \mathop{\sum }\limits_{{t \in  q \cap  d}}{\sigma }_{t}\left( {{h}_{q}\left( {q,t}\right) ,{h}_{d}\left( {d,t}\right) }\right)  \tag{1}
$$

where for each overlapping term $t$ between query $q$ and document $d$ ,functions ${h}_{q}$ and ${h}_{d}$ extract term information and a term scoring function ${\sigma }_{t}$ combines them. A popular example is BM25, which computes,

其中，对于查询 $q$ 和文档 $d$ 之间的每个重叠词项 $t$，函数 ${h}_{q}$ 和 ${h}_{d}$ 提取词项信息，词项评分函数 ${\sigma }_{t}$ 将这些信息进行组合。一个常见的例子是BM25（最佳匹配25算法，Best Matching 25），它的计算方式如下，

$$
{s}_{\mathrm{{BM}}{25}} = \mathop{\sum }\limits_{{t \in  q \cap  d}}{idf}\left( t\right) {h}_{q}^{\mathrm{{BM}}{25}}\left( {q,t}\right) {h}_{d}^{\mathrm{{BM}}{25}}\left( {d,t}\right) 
$$

$$
{h}_{q}^{\mathrm{{BM}}{25}}\left( {q,t}\right)  = \frac{t{f}_{t,q}\left( {1 + {k}_{2}}\right) }{t{f}_{t,q} + {k}_{2}} \tag{2}
$$

$$
{h}_{d}^{\mathrm{{BM}}{25}}\left( {d,t}\right)  = \frac{t{f}_{t,d}\left( {1 + {k}_{1}}\right) }{t{f}_{t,d} + {k}_{1}\left( {1 - b + b\frac{\left| d\right| }{\text{ avgdl }}}\right) }
$$

where $t{f}_{t,d}$ refers to term frequency of term $t$ in document $d,t{f}_{t,q}$ refers to the term frequency in query, ${idf}\left( t\right)$ is inverse document frequency,and $b$ , ${k}_{1},{k}_{2}$ are hyper-parameters.

其中 $t{f}_{t,d}$ 指的是术语 $t$ 在文档 $d,t{f}_{t,q}$ 中的词频，${idf}\left( t\right)$ 指的是查询中的词频，${idf}\left( t\right)$ 是逆文档频率，$b$、${k}_{1},{k}_{2}$ 是超参数。

One key advantage of exact lexical match systems lies in efficiency. With summation over exact matches, scoring of each query term only goes to documents that contain matching terms. This can be done efficiently using inverted list indexing (Figure 2). The inverted list maps back from a term to a list of documents where the term occurs. To compute Equation 1, the retriever only needs to traverse the subset of documents in query terms' inverted lists instead of going over the entire document collection.

精确词汇匹配系统的一个关键优势在于效率。通过对精确匹配进行求和，每个查询词的评分仅针对包含匹配词的文档。这可以使用倒排索引列表高效完成（图2）。倒排索引列表将一个词映射回该词出现的文档列表。为了计算公式1，检索器只需遍历查询词倒排索引列表中的文档子集，而无需遍历整个文档集合。

While recent neural IR research mainly focuses on breaking the exact match bottleneck with soft matching of text, we hypothesize that exact match itself can be improved by replacing semantic independent frequency-based scoring with semantic rich scoring. In the rest of this section, we show how to modify the exact lexical match framework with contextualized term representations to build effective and efficient retrieval systems.

虽然近期的神经信息检索（IR）研究主要侧重于通过文本的软匹配来打破精确匹配的瓶颈，但我们假设，通过用富含语义的评分取代基于语义无关频率的评分，可以改进精确匹配本身。在本节的其余部分，我们将展示如何使用上下文相关的词表示来修改精确词汇匹配框架，以构建高效且有效的检索系统。

### 3.2 Contextualized Exact Lexical Match

### 3.2 上下文精确词汇匹配

Instead of term frequency, we desire to encode the semantics of terms to facilitate more effective matching. Inspired by recent advancements in deep LM, we encode both query and document tokens into contextualized vector representations and carry out matching between exact lexical matched tokens. Figure 1d illustrates the scoring model of COIL.

我们希望对术语的语义进行编码，以促进更有效的匹配，而不是使用词频。受深度语言模型（LM）近期进展的启发，我们将查询词和文档标记都编码为上下文向量表示，并对精确词汇匹配的标记进行匹配。图1d展示了COIL（上下文感知的精确词汇匹配）的评分模型。

In this work, we use a Transformer language ${\text{model}}^{3}$ as the contextualization function. We encode a query $q$ with the language model (LM) and represent its $i$ -th token by projecting the corresponding output:

在这项工作中，我们使用Transformer语言模型${\text{model}}^{3}$作为上下文函数。我们使用语言模型（LM）对查询词$q$进行编码，并通过投影相应的输出来表示其第$i$个标记：

$$
{\mathbf{v}}_{i}^{q} = {\mathbf{W}}_{\text{tok }}\operatorname{LM}\left( {q,i}\right)  + {\mathbf{b}}_{\text{tok }} \tag{3}
$$

where ${\mathbf{W}}_{\text{tok }}^{{n}_{t} \times  {n}_{lm}}$ is a matrix that maps the LM’s ${n}_{lm}$ dimension output into a vector of lower dimension ${n}_{t}$ . We down project the vectors as we hypothesize that it suffices to use lower dimension token vectors. We confirm this in section 5. Similarly,we encode a document $d$ ’s $j$ -th token ${d}_{j}$ with:

其中${\mathbf{W}}_{\text{tok }}^{{n}_{t} \times  {n}_{lm}}$是一个矩阵，它将语言模型的${n}_{lm}$维输出映射到较低维度${n}_{t}$的向量。我们对向量进行降维投影，因为我们假设使用较低维度的标记向量就足够了。我们将在第5节中证实这一点。同样，我们使用以下方式对文档$d$的第$j$个标记${d}_{j}$进行编码：

$$
{\mathbf{v}}_{j}^{d} = {\mathbf{W}}_{tok}\operatorname{LM}\left( {d,j}\right)  + {\mathbf{b}}_{tok} \tag{4}
$$

We then define the contextualized exact lexical match scoring function between query document based on vector similarities between exact matched query document token pairs:

然后，我们基于精确匹配的查询文档词元对之间的向量相似度，定义查询文档之间的上下文精确词汇匹配评分函数：

$$
{s}_{\text{tok }}\left( {q,d}\right)  = \mathop{\sum }\limits_{{{q}_{i} \in  q \cap  d}}\mathop{\max }\limits_{{{d}_{j} = {q}_{i}}}\left( {{\mathbf{v}}_{i}^{q \intercal  }{\mathbf{v}}_{j}^{d}}\right)  \tag{5}
$$

Note that, importantly, the summation goes through only overlapping terms, ${q}_{i} \in  q \cap  d$ . For each query token ${q}_{i}$ ,we finds all same tokens ${d}_{j}$ in the document,computes their similarity with ${q}_{i}$ using the contextualized token vectors. The maximum similarities are picked for query token ${q}_{i}$ . Max operator is adopted to capture the most important signal (Kim, 2014). This fits in the general lexical match formulation,with ${h}_{q}$ giving representation for ${q}_{i},{h}_{t}$ giving representations for all ${d}_{j} = {q}_{i}$ ,and ${\sigma }_{t}$ compute dot similarities between query vector with document vectors and max pool the scores.

重要的是，请注意，求和仅针对重叠的术语 ${q}_{i} \in  q \cap  d$ 进行。对于每个查询词元 ${q}_{i}$，我们在文档中找到所有相同的词元 ${d}_{j}$，使用上下文词元向量计算它们与 ${q}_{i}$ 的相似度。为查询词元 ${q}_{i}$ 选取最大相似度。采用最大值运算符来捕捉最重要的信号（金（Kim），2014 年）。这符合一般的词汇匹配公式，其中 ${h}_{q}$ 表示 ${q}_{i},{h}_{t}$ 表示所有 ${d}_{j} = {q}_{i}$，并且 ${\sigma }_{t}$ 计算查询向量与文档向量之间的点积相似度，并对得分进行最大池化。

As with classic lexical systems, ${s}_{tok}$ defined in Equation 5 does not take into account similarities between lexical-different terms, thus faces vocabulary mismatch. Many popular LMs (Devlin et al., 2019; Yang et al., 2019; Liu et al., 2019) use a special CLS token to aggregate sequence representation. We project the CLS vectos with ${\mathbf{W}}_{cls}^{{n}_{c} \times  {n}_{lm}}$ to represent the entire query or document,

与经典的词法系统一样，公式5中定义的${s}_{tok}$没有考虑词汇不同的术语之间的相似性，因此面临词汇不匹配的问题。许多流行的语言模型（Devlin等人，2019年；Yang等人，2019年；Liu等人，2019年）使用特殊的CLS标记来聚合序列表示。我们用${\mathbf{W}}_{cls}^{{n}_{c} \times  {n}_{lm}}$对CLS向量进行投影，以表示整个查询或文档。

$$
{\mathbf{v}}_{cls}^{q} = {\mathbf{W}}_{cls}\operatorname{LM}\left( {q,\mathrm{{CLS}}}\right)  + {\mathbf{b}}_{cls} \tag{6}
$$

$$
{\mathbf{v}}_{cls}^{d} = {\mathbf{W}}_{cls}\operatorname{LM}\left( {d,\mathrm{{CLS}}}\right)  + {\mathbf{b}}_{cls}
$$

The similarity between ${\mathbf{v}}_{cls}^{q}$ and ${\mathbf{v}}_{cls}^{d}$ provides high-level semantic matching and mitigates the issue of vocabulary mismatch. The full form of COIL is:

${\mathbf{v}}_{cls}^{q}$和${\mathbf{v}}_{cls}^{d}$之间的相似性提供了高层次的语义匹配，并缓解了词汇不匹配的问题。COIL的完整形式是：

$$
{s}_{\text{full }}\left( {q,d}\right)  = {s}_{\text{tok }}\left( {q,d}\right)  + {\mathbf{v}}_{cls}^{q}{}^{\top }{\mathbf{v}}_{cls}^{d} \tag{7}
$$

In the rest of the paper, we refer to systems with CLS matching COIL-full and without COIL-tok.

在本文的其余部分，我们将具有CLS匹配的系统称为COIL-full，而将没有CLS匹配的系统称为COIL-tok。

COIL's scoring model (Figure 1d) is fully differentiable. Following earlier work (Karpukhin et al., 2020), we train COIL with negative log likelihood defined over query $q$ ,a positive document ${d}^{ + }$ and a set of negative documents $\left\{  {{d}_{1}^{ - },{d}_{2}^{ - },..{d}_{l}^{ - }..}\right\}$ as loss.

COIL的评分模型（图1d）是完全可微的。遵循早期的研究（卡尔普欣等人，2020年），我们使用在查询 $q$、一个正文档 ${d}^{ + }$ 和一组负文档 $\left\{  {{d}_{1}^{ - },{d}_{2}^{ - },..{d}_{l}^{ - }..}\right\}$ 上定义的负对数似然作为损失来训练COIL。

$$
\mathcal{L} =  - \log \frac{\exp \left( {s\left( {q,{d}^{ + }}\right) }\right) }{\exp \left( {s\left( {q,{d}^{ + }}\right) }\right)  + \mathop{\sum }\limits_{l}\exp \left( {s\left( {q,{d}_{l}^{ - }}\right) }\right) }
$$

(8)

---

<!-- Footnote -->

${}^{3}$ We used the base,uncased variant of BERT.

${}^{3}$ 我们使用了BERT的基础无大小写变体。

<!-- Footnote -->

---

Following Karpukhin et al. (2020), we use in batch negatives and hard negatives generated by BM25. Details are discussed in implementation, section 4.

遵循卡尔普欣等人（2020年）的方法，我们使用批次内负样本和由BM25生成的难负样本。具体细节将在第4节“实现”中讨论。

### 3.3 Index and Retrieval with COIL

### 3.3 使用COIL进行索引和检索

COIL pre-computes the document representations and builds up a search index, which is illustrated in Figure 3. Documents in the collection are encoded offline into token and CLS vectors. Formally, for a unique token $t$ in the vocabulary $V$ ,we collect its contextualized vectors from all of its mentions from documents in collection $C$ ,building token $t$ ’s contextualized inverted list:

COIL（上下文感知倒排列表，Contextualized Inverted List）预先计算文档表示并构建搜索索引，如图3所示。集合中的文档会离线编码为词元（token）和CLS向量。形式上，对于词汇表$V$中的唯一词元$t$，我们从集合$C$中文档里该词元的所有提及处收集其上下文向量，构建词元$t$的上下文倒排列表：

$$
{I}^{t} = \left\{  {{\mathbf{v}}_{j}^{d} \mid  {d}_{j} = t,d \in  C}\right\}  , \tag{9}
$$

where ${\mathbf{v}}_{j}^{d}$ is the BERT-based token encoding defined in Equation 4. We define search index to store inverted lists for all tokens in vocabulary, $\mathbb{I} = \left\{  {{I}^{t} \mid  t \in  V}\right\}$ . For COIL-full,we also build an index for the CLS token ${I}^{cls} = \left\{  {{\mathbf{v}}_{cls}^{d} \mid  d \in  C}\right\}$ .

其中${\mathbf{v}}_{j}^{d}$是公式4中定义的基于BERT的词元编码。我们定义搜索索引来存储词汇表$\mathbb{I} = \left\{  {{I}^{t} \mid  t \in  V}\right\}$中所有词元的倒排列表。对于COIL-full，我们还会为CLS词元${I}^{cls} = \left\{  {{\mathbf{v}}_{cls}^{d} \mid  d \in  C}\right\}$构建一个索引。

As shown in Figure 3, in this work we implement COIL's by stacking vectors in each inverted list ${I}^{t}$ into a matrix ${M}^{{n}_{t} \times  \left| {I}^{k}\right| }$ ,so that similarity computation that traverses an inverted list and computes vector dot product can be done efficiently as one matrix-vector product with optimized BLAS (Blackford et al., 2002) routines on CPU or GPU. All ${\mathbf{v}}_{cls}^{d}$ vectors can also be organized in a similar fashion into matrix ${M}_{cls}$ and queried with matrix product. The matrix implementation here is an exhaustive approach that involves all vectors in an inverted list. As a collection of dense vectors, it is also possible to organize each inverted list as an approximate search index (Johnson et al., 2017; Guo et al., 2019) to further speed up search.

如图3所示，在这项工作中，我们通过将每个倒排列表${I}^{t}$中的向量堆叠成矩阵${M}^{{n}_{t} \times  \left| {I}^{k}\right| }$来实现COIL（上下文感知倒排列表，Contextualized Inverted Lists），这样，遍历倒排列表并计算向量点积的相似度计算就可以作为一次矩阵 - 向量乘法高效完成，利用CPU或GPU上经过优化的BLAS（基础线性代数子程序库，Basic Linear Algebra Subprograms）例程（Blackford等人，2002）。所有${\mathbf{v}}_{cls}^{d}$向量也可以以类似的方式组织成矩阵${M}_{cls}$，并通过矩阵乘法进行查询。这里的矩阵实现是一种穷举方法，涉及倒排列表中的所有向量。作为密集向量的集合，也可以将每个倒排列表组织成近似搜索索引（Johnson等人，2017；Guo等人，2019），以进一步加快搜索速度。

When a query $q$ comes in,we encode every of its token into vectors ${\mathbf{v}}_{i}^{q}$ . The vectors are sent to the subset of COIL inverted lists that corresponds query tokens $\mathbb{J} = \left\{  {{I}^{t} \mid  t \in  q}\right\}$ . where the matrix product described above is carried out. This is efficient as $\left| \mathbb{J}\right|  <  < \left| \mathbb{I}\right|$ ,having only a small subset of all inverted lists to be involved in search. For COIL-full,we also use encoded CLS vectors ${\mathbf{v}}_{cls}^{q}$ to query the CLS index to get the CLS matching scores. The scoring over different inverted lists can serve in parallel. The scores are then combined by Equation 5 to rank the documents.

当查询 $q$ 到来时，我们将其每个标记编码为向量 ${\mathbf{v}}_{i}^{q}$。这些向量被发送到与查询标记 $\mathbb{J} = \left\{  {{I}^{t} \mid  t \in  q}\right\}$ 对应的 COIL（上下文感知倒排列表，Contextualized Inverted Lists）倒排列表子集，在那里执行上述矩阵乘法。这很高效，因为 $\left| \mathbb{J}\right|  <  < \left| \mathbb{I}\right|$，搜索时只涉及所有倒排列表的一小部分。对于 COIL-full，我们还使用编码后的 CLS（分类标记，Classification Token）向量 ${\mathbf{v}}_{cls}^{q}$ 查询 CLS 索引以获得 CLS 匹配分数。不同倒排列表的评分可以并行进行。然后通过公式 5 组合这些分数对文档进行排名。

Readers can find detailed illustration figures in the Appendix A, for index building and querying, Figure 4 and Figure 5, respectively.

读者可以在附录 A 中分别找到关于索引构建和查询的详细说明图，即图 4 和图 5。

### 3.4 Connection to Other Retrievers

### 3.4 与其他检索器的联系

Deep LM based Lexical Index Models like DeepCT (Dai and Callan, 2019a, 2020) and DocT5Query (Nogueira and Lin,2019) alter $t{f}_{t,d}$ in documents with deep LM BERT or T5. This is similar to a COIL-tok with token dimension ${n}_{t} = 1$ . A single degree of freedom however measures more of a term importance than semantic agreement.

基于深度语言模型（Deep LM）的词法索引模型，如DeepCT（戴和卡兰，2019a，2020年）和DocT5Query（诺盖拉和林，2019年），使用深度语言模型BERT或T5来改变文档中的$t{f}_{t,d}$。这类似于具有标记维度${n}_{t} = 1$的COIL标记。然而，单一自由度衡量的更多是词项重要性，而非语义一致性。

Dense Retriever Dense retrievers (Figure 1b) are equivalent to COIL-full's CLS matching. COIL makes up for the lost token-level interactions in dense retriever with exact matching signals.

密集检索器 密集检索器（图1b）等同于COIL-full的CLS匹配。COIL通过精确匹配信号弥补了密集检索器中丢失的标记级交互。

ColBERT ColBERT (Figure 1c) computes relevance by soft matching all query and document term's contextualized vectors.

ColBERT ColBERT（图1c）通过对所有查询词和文档词项的上下文向量进行软匹配来计算相关性。

$$
s\left( {q,d}\right)  = \mathop{\sum }\limits_{{{q}_{i} \in  \left\lbrack  {{cls};q;{exp}}\right\rbrack  }}\mathop{\max }\limits_{{{d}_{j} \in  \left\lbrack  {{cls};d}\right\rbrack  }}\left( {{\mathbf{v}}_{i}^{q\top }{\mathbf{v}}_{j}^{d}}\right)  \tag{10}
$$

where interactions happen among query $q$ ,document $d,{cls}$ and set of query expansion tokens exp. The all-to-all match contrasts COIL that only uses exact match. It requires a dense retrieval over all document tokens' representations as opposed to COIL which only considers query's overlapping tokens, and are therefore much more computationally expensive than COIL.

其中，交互发生在查询$q$、文档$d,{cls}$和查询扩展标记集exp之间。全对全匹配与仅使用精确匹配的COIL形成对比。与仅考虑查询重叠标记的COIL不同，它需要对所有文档标记的表示进行密集检索，因此在计算上比COIL昂贵得多。

## 4 Experiment Methodologies

## 4 实验方法

Datasets We experiment with two large scale ad hoc retrieval benchmarks from the TREC 2019 Deep Learning (DL) shared task: MSMARCO passage ( $8\mathrm{M}$ English passages of average length around 60 tokens) and MSMARCO document (3M English documents of average length around 900 tokens ${)}^{4}$ . For each,we train models with the MSMARCO Train queries, and record results on MSMARCO Dev queries and TREC DL 2019 test queries. We report mainly full-corpus retrieval results but also include the rerank task on MSMARCO Dev queries where we use neural scores to reorder BM25 retrieval results provided by MSMARO organizers. Official metrics include MRR@1K and NDCG@10 on test and MRR@10 on MSMARCO Dev. We also report recall for the dev queries following prior work (Dai and Callan, 2019a; Nogueira and Lin, 2019).

数据集 我们使用来自2019年文本检索会议（TREC）深度学习（DL）共享任务的两个大规模即席检索基准进行实验：微软机器阅读理解数据集段落（MSMARCO passage，约$8\mathrm{M}$个平均长度约为60个词元的英文段落）和微软机器阅读理解数据集文档（MSMARCO document，300万个平均长度约为900个词元的英文文档${)}^{4}$）。对于每个数据集，我们使用微软机器阅读理解数据集训练查询（MSMARCO Train queries）来训练模型，并记录在微软机器阅读理解数据集开发查询（MSMARCO Dev queries）和2019年文本检索会议深度学习测试查询（TREC DL 2019 test queries）上的结果。我们主要报告全量语料库检索结果，但也包括微软机器阅读理解数据集开发查询的重排序任务，在该任务中，我们使用神经得分对微软机器阅读理解数据集组织者提供的BM25检索结果进行重新排序。官方指标包括测试集上的前1000个结果的平均倒数排名（MRR@1K）和前10个结果的归一化折损累积增益（NDCG@10），以及微软机器阅读理解数据集开发集上的前10个结果的平均倒数排名（MRR@10）。我们还按照先前的研究（戴和卡兰，2019a；诺盖拉和林，2019）报告了开发查询的召回率。

---

<!-- Footnote -->

${}^{4}$ Both datasets can be downloaded from https:// microsoft.github.io/msmarco/

${}^{4}$ 这两个数据集均可从https:// microsoft.github.io/msmarco/下载

<!-- Footnote -->

---

Compared Systems Baselines include 1) traditional exact match system $\mathrm{{BM}}{25},2)$ deep $\mathrm{{LM}}$ augmented BM25 systems DeepCT (Dai and Callan, 2019a) and DocT5Query (Nogueira and Lin, 2019), 3) dense retrievers, and 4) soft all-to-all retriever ColBERT. For DeepCT and DocT5Query, we use the rankings provided by the authors. For dense retrievers, we report two dense retrievers trained with BM25 negatives or with mixed BM25 and random negatives, published in Xiong et al. (2020). However since these systems use a robust version of BERT, RoBERTa (Liu et al., 2019) as the LM and train document retriever also on MSMARCO passage set, we in addition reproduce a third dense retriever, that uses the exact same training setup as COIL. All dense retrievers use 768 dimension embedding. For ColBERT, we report its published results (available only on passage collection). BERT reranker is added in the rerank task.

对比系统基线包括：1) 传统精确匹配系统；2) 深度增强的BM25系统，如DeepCT（戴和卡兰，2019a）和DocT5Query（诺盖拉和林，2019）；3) 密集检索器；4) 软全对全检索器ColBERT。对于DeepCT和DocT5Query，我们使用作者提供的排名。对于密集检索器，我们报告了两个分别用BM25负样本或混合BM25和随机负样本训练的密集检索器，相关内容发表于熊等人（2020）的研究中。然而，由于这些系统使用了BERT的增强版本RoBERTa（刘等人，2019）作为语言模型（LM），并且也在MSMARCO段落集上训练文档检索器，因此我们额外复现了第三个密集检索器，其训练设置与COIL完全相同。所有密集检索器都使用768维嵌入。对于ColBERT，我们报告其已发表的结果（仅适用于段落集合）。在重排任务中加入了BERT重排器。

We include 2 COIL systems: 1) COIL-tok, the exact token match only system, and 2) COLL-full, the model with both token match and CLS match.

我们纳入了两个COIL（上下文无关信息检索）系统：1）COIL-tok，即仅进行精确词元匹配的系统；2）COIL-full，即同时进行词元匹配和CLS（分类）匹配的模型。

Implementation We build our models with Py-torch (Paszke et al., 2019) based on huggingface transformers (Wolf et al., 2019). COIL's LM is based on BERT's base variant. COIL systems use token dimension ${n}_{t} = {32}$ and COIL-full use CLS dimension ${n}_{c} = {768}$ as default,leading to ${110}\mathrm{M}$ parameters. We add a Layer Normalization to CLS vector when useful. All models are trained for 5 epochs with AdamW optimizer, a learning rate of $3\mathrm{e} - 6,{0.1}$ warm-up ratio,and linear learning rate decay, which takes around 12 hours. Hard negatives are sampled from top 1000 BM25 results. Each query uses 1 positive and 7 hard negatives; each batch uses 8 queries on MSMARCO passage and 4 on MSMARCO document. Documents are truncated to the first 512 tokens to fit in BERT. We conduct validation on randomly selected 512 queries from corresponding train set. Latency numbers are measured on dual Xeon E5-2630 v3 for CPU and RTX 2080 ti for GPU. We implement COIL's inverted lists as matrices as described in subsection 3.3, using NumPy (Harris et al., 2020) on CPU and Pytorch on GPU. We perform a) a set of matrix products to compute token similarities over contextualized inverted lists, b) scatter to map token scores back to documents, and c) sort to rank the documents. Illustration can be found in the appendix, Figure 5.

实现 我们基于Hugging Face的Transformers库（Wolf等人，2019年），使用PyTorch（Paszke等人，2019年）构建模型。COIL的语言模型（LM）基于BERT的基础变体。COIL系统默认使用词元维度${n}_{t} = {32}$，而COIL-full默认使用CLS维度${n}_{c} = {768}$，这会产生${110}\mathrm{M}$个参数。必要时，我们会对CLS向量添加层归一化（Layer Normalization）。所有模型使用AdamW优化器进行5个轮次（epoch）的训练，学习率为$3\mathrm{e} - 6,{0.1}$，采用热身比例（warm-up ratio）和线性学习率衰减策略，训练大约需要12小时。难负样本（hard negatives）从BM25的前1000个结果中采样。每个查询使用1个正样本和7个难负样本；在MSMARCO段落数据集上，每个批次使用8个查询，在MSMARCO文档数据集上则使用4个查询。为了适应BERT，文档会被截断为前512个词元。我们从相应的训练集中随机选择512个查询进行验证。延迟数据在配备双至强E5 - 2630 v3 CPU和RTX 2080 ti GPU的设备上进行测量。我们按照3.3小节的描述，将COIL的倒排表实现为矩阵，在CPU上使用NumPy（Harris等人，2020年），在GPU上使用PyTorch。我们执行以下操作：a) 进行一组矩阵乘法，以计算上下文倒排表上的词元相似度；b) 进行散射操作，将词元得分映射回文档；c) 进行排序操作，对文档进行排名。具体示例可在附录图5中找到。

## 5 Results

## 5 结果

This section studies the effectiveness of COIL and how vector dimension in COIL affects the effectiveness-efficiency tradeoff. We also provide qualitative analysis on contextualized exact match.

本节研究了COIL（上下文感知词交互学习，Contextualized Overlap Interaction Learning）的有效性，以及COIL中的向量维度如何影响有效性 - 效率权衡。我们还对上下文精确匹配进行了定性分析。

### 5.1 Main Results

### 5.1 主要结果

Table 1 reports various systems' performance on the MARCO passage collection. COIL-tok exact lexical match only system significantly outperforms all previous lexical retrieval systems. With contextualized term similarities, COIL-tok achieves a MRR of 0.34 compared to BM25's MRR 0.18. DeepCT and DocT5Query, which also use deep LMs like BERT and T5, are able to break the limit of heuristic term frequencies but are still limited by semantic mismatch issues. We see COIL-tok outperforms both systems by a large margin.

表1报告了各种系统在MARCO段落数据集上的性能。仅使用COIL - tok精确词汇匹配的系统显著优于所有先前的词汇检索系统。通过上下文词相似度，COIL - tok的平均倒数排名（MRR）达到了0.34，而BM25的MRR为0.18。DeepCT和DocT5Query也使用了像BERT和T5这样的深度语言模型，它们能够突破启发式词频的限制，但仍然受语义不匹配问题的限制。我们发现COIL - tok大幅优于这两个系统。

COIL-tok also ranks top of the candidate list better than dense retrieves. It prevails in MRR and NDCG while performs on par in recall with the best dense system, indicating that COIL's token level interaction can improve precision. With the CLS matching added, COIL-full gains the ability to handle mismatched vocabulary and enjoys another performance leap, outperforming all dense retrievers.

COIL - tok在候选列表中的排名也比密集检索器更好。它在平均倒数排名（MRR）和归一化折损累积增益（NDCG）方面表现出色，同时在召回率方面与最佳的密集系统相当，这表明COIL的词元级交互可以提高精度。添加了CLS匹配后，COIL - full具备了处理不匹配词汇的能力，性能再次跃升，优于所有密集检索器。

COIL-full achieves a very narrow performance gap to ColBERT. Recall that ColBERT computes all-to-all soft matches between all token pairs. For retrieval, it needs to consider for each query token all mentions of all tokens in the collection (MS-MARCO passage collection has around ${500}\mathrm{M}$ token mentions). COIL-full is able to capture matching patterns as effectively with exact match signals from only query tokens' mentions and a single CLS matching to bridge the vocabulary gap.

COIL-full与ColBERT（一种检索模型）的性能差距非常小。请记住，ColBERT会计算所有词元对之间的全对全软匹配。在检索时，它需要为每个查询词元考虑集合中所有词元的所有提及情况（MS-MARCO段落集合大约有${500}\mathrm{M}$个词元提及）。COIL-full仅通过查询词元的提及的精确匹配信号以及单个CLS匹配来弥合词汇差距，就能有效地捕捉匹配模式。

We observe a similar pattern in the rerank task. COIL-tok is already able to outperform dense retriever and COIL-full further adds up to performance with CLS matching, being on-par with ColBERT. Meanwhile, previous BERT rerankers have little performance advantage over COIL ${}^{5}$ . In practice, we found BERT rerankers to be much more expensive,requiring over ${2700}\mathrm{\;{ms}}$ for reranking compared to around ${10}\mathrm{\;{ms}}$ in the case of COIL.

我们在重排序任务中观察到了类似的模式。COIL-tok已经能够超越密集检索器，而COIL-full通过CLS匹配进一步提升了性能，与ColBERT相当。与此同时，之前的BERT重排序器相比COIL ${}^{5}$ 几乎没有性能优势。在实践中，我们发现BERT重排序器的成本要高得多，重排序需要超过${2700}\mathrm{\;{ms}}$，而COIL大约只需要${10}\mathrm{\;{ms}}$。

---

<!-- Footnote -->

${}^{5}$ Close performance between COIL and BERT rerankers is partially due to the bottleneck of BM25 candidates.

${}^{5}$ COIL和BERT重排序器性能相近，部分原因是BM25候选集的瓶颈。

<!-- Footnote -->

---

<!-- Media -->

Table 1: MSMARCO passage collection results. Results not applicable are denoted '-' and no available 'n.a.'.

表1：MS MARCO段落数据集结果。不适用的结果用“-”表示，无可用结果用“n.a.”表示。

<table><tr><td/><td colspan="5">MS MARCO Passage Ranking</td></tr><tr><td rowspan="2">Model</td><td rowspan="2">Dev Rerank MRR@10</td><td colspan="2">Dev Retrieval</td><td colspan="2">DL2019 Retrieval</td></tr><tr><td>MRR@10</td><td>Recall@1K</td><td>NDCG@10</td><td>MRR@1K</td></tr><tr><td colspan="6">Lexical Retriever</td></tr><tr><td>BM25</td><td>-</td><td>0.184</td><td>0.853</td><td>0.506</td><td>0.825</td></tr><tr><td>DeepCT</td><td>-</td><td>0.243</td><td>0.909</td><td>0.572</td><td>0.883</td></tr><tr><td>DocT5Query</td><td>-</td><td>0.278</td><td>0.945</td><td>0.642</td><td>0.888</td></tr><tr><td>BM25+BERT reranker</td><td>0.347</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="6">Dense Retriever</td></tr><tr><td>Dense (BM25 neg)</td><td>n.a.</td><td>0.299</td><td>0.928</td><td>0.600</td><td>n.a.</td></tr><tr><td>Dense (rand + BM25 neg)</td><td>n.a.</td><td>0.311</td><td>0.952</td><td>0.576</td><td>n.a.</td></tr><tr><td>Dense (our train)</td><td>0.312</td><td>0.304</td><td>0.932</td><td>0.635</td><td>0.898</td></tr><tr><td>ColBERT</td><td>0.349</td><td>0.360</td><td>0.968</td><td>n.a.</td><td>n.a.</td></tr><tr><td>COIL-tok</td><td>0.336</td><td>0.341</td><td>0.949</td><td>0.660</td><td>0.915</td></tr><tr><td>COIL-full</td><td>0.348</td><td>0.355</td><td>0.963</td><td>0.704</td><td>0.924</td></tr></table>

<table><tbody><tr><td></td><td colspan="5">微软机器阅读理解数据集段落排序（MS MARCO Passage Ranking）</td></tr><tr><td rowspan="2">模型</td><td rowspan="2">开发集重排序的前10名平均倒数排名（Dev Rerank MRR@10）</td><td colspan="2">开发集检索（Dev Retrieval）</td><td colspan="2">2019年深度学习挑战赛检索（DL2019 Retrieval）</td></tr><tr><td>前10名平均倒数排名（MRR@10）</td><td>前1000召回率（Recall@1K）</td><td>前10归一化折损累积增益（NDCG@10）</td><td>前1000平均倒数排名（MRR@1K）</td></tr><tr><td colspan="6">词法检索器（Lexical Retriever）</td></tr><tr><td>二元独立模型（BM25）</td><td>-</td><td>0.184</td><td>0.853</td><td>0.506</td><td>0.825</td></tr><tr><td>深度上下文词项（DeepCT）</td><td>-</td><td>0.243</td><td>0.909</td><td>0.572</td><td>0.883</td></tr><tr><td>文档T5查询（DocT5Query）</td><td>-</td><td>0.278</td><td>0.945</td><td>0.642</td><td>0.888</td></tr><tr><td>BM25+BERT重排序器</td><td>0.347</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="6">密集检索器（Dense Retriever）</td></tr><tr><td>密集（BM25负样本）</td><td>不适用（n.a.）</td><td>0.299</td><td>0.928</td><td>0.600</td><td>不适用（n.a.）</td></tr><tr><td>密集（随机+BM25负样本）</td><td>不适用（n.a.）</td><td>0.311</td><td>0.952</td><td>0.576</td><td>不适用（n.a.）</td></tr><tr><td>密集型（我们的模型）</td><td>0.312</td><td>0.304</td><td>0.932</td><td>0.635</td><td>0.898</td></tr><tr><td>ColBERT（一种模型）</td><td>0.349</td><td>0.360</td><td>0.968</td><td>不适用（n.a.）</td><td>不适用（n.a.）</td></tr><tr><td>COIL词元版（COIL-tok）</td><td>0.336</td><td>0.341</td><td>0.949</td><td>0.660</td><td>0.915</td></tr><tr><td>COIL完整版（COIL-full）</td><td>0.348</td><td>0.355</td><td>0.963</td><td>0.704</td><td>0.924</td></tr></tbody></table>

Table 2: MSMARCO document collection results. Results not applicable are denoted ‘-’ and no available ‘n.a.’.

表2：MS MARCO文档集结果。不适用的结果标记为“-”，无可用结果标记为“n.a.”。

<table><tr><td/><td colspan="5">MS MARCO Document Ranking</td></tr><tr><td rowspan="2">Model</td><td rowspan="2">Dev Rerank MRR@10</td><td colspan="2">Dev Retrieval</td><td colspan="2">DL2019 Retrieval</td></tr><tr><td>MRR@10</td><td>Recall@1K</td><td>NDCG@10</td><td>MRR@1K</td></tr><tr><td colspan="6">Lexical Retriever</td></tr><tr><td>BM25</td><td>-</td><td>0.230</td><td>0.886</td><td>0.519</td><td>0.805</td></tr><tr><td>DeepCT</td><td>-</td><td>0.320</td><td>0.942</td><td>0.544</td><td>0.891</td></tr><tr><td>DocT5Query</td><td>-</td><td>0.288</td><td>0.926</td><td>0.597</td><td>0.837</td></tr><tr><td>BM25+BERT reranker</td><td>0.383</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="6">Dense Retriever</td></tr><tr><td>Dense (BM25 neg)</td><td>n.a.</td><td>0.299</td><td>0.928</td><td>0.600</td><td>n.a.</td></tr><tr><td>Dense (rand + BM25 neg)</td><td>n.a.</td><td>0.311</td><td>0.952</td><td>0.576</td><td>n.a.</td></tr><tr><td>Dense (our train)</td><td>0.358</td><td>0.340</td><td>0.883</td><td>0.546</td><td>0.785</td></tr><tr><td>COIL-tok</td><td>0.381</td><td>0.385</td><td>0.952</td><td>0.626</td><td>0.921</td></tr><tr><td>COIL-full</td><td>0.388</td><td>0.397</td><td>0.962</td><td>0.636</td><td>0.913</td></tr></table>

<table><tbody><tr><td></td><td colspan="5">MS MARCO文档排名（MS MARCO Document Ranking）</td></tr><tr><td rowspan="2">模型</td><td rowspan="2">开发集重排序的前10名平均倒数排名（Dev Rerank MRR@10）</td><td colspan="2">开发集检索（Dev Retrieval）</td><td colspan="2">2019年深度学习挑战赛检索（DL2019 Retrieval）</td></tr><tr><td>前10名平均倒数排名（MRR@10）</td><td>前1000召回率（Recall@1K）</td><td>前10归一化折损累积增益（NDCG@10）</td><td>前1000平均倒数排名（MRR@1K）</td></tr><tr><td colspan="6">词法检索器（Lexical Retriever）</td></tr><tr><td>二元独立模型（BM25）</td><td>-</td><td>0.230</td><td>0.886</td><td>0.519</td><td>0.805</td></tr><tr><td>深度上下文词项（DeepCT）</td><td>-</td><td>0.320</td><td>0.942</td><td>0.544</td><td>0.891</td></tr><tr><td>文档T5查询模型（DocT5Query）</td><td>-</td><td>0.288</td><td>0.926</td><td>0.597</td><td>0.837</td></tr><tr><td>BM25算法+BERT重排器</td><td>0.383</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="6">密集检索器（Dense Retriever）</td></tr><tr><td>密集检索（BM25负样本）</td><td>不适用</td><td>0.299</td><td>0.928</td><td>0.600</td><td>不适用</td></tr><tr><td>密集检索（随机样本+BM25负样本）</td><td>不适用</td><td>0.311</td><td>0.952</td><td>0.576</td><td>不适用</td></tr><tr><td>密集（我们的训练）</td><td>0.358</td><td>0.340</td><td>0.883</td><td>0.546</td><td>0.785</td></tr><tr><td>COIL - tok（原文未变，因无更多信息难以准确翻译）</td><td>0.381</td><td>0.385</td><td>0.952</td><td>0.626</td><td>0.921</td></tr><tr><td>COIL - 完整（原文未变，因无更多信息难以准确翻译）</td><td>0.388</td><td>0.397</td><td>0.962</td><td>0.636</td><td>0.913</td></tr></tbody></table>

<!-- Media -->

Table 2 reports the results on MSMARCO document collection. In general, we observe a similar pattern as with the passage case. COIL systems significantly outperform both lexical and dense systems in MRR and NDCG and retain a small advantage measured in recall. The results suggest that COIL can be applicable to longer documents with a consistent advantage in effectiveness.

表2报告了在MS MARCO文档集上的实验结果。总体而言，我们观察到的模式与段落情况类似。在平均倒数排名（MRR）和归一化折损累积增益（NDCG）指标上，COIL系统显著优于词法系统和稠密向量系统，并且在召回率方面也保持着微弱优势。这些结果表明，COIL可以应用于较长的文档，并且在有效性方面具有持续的优势。

The results indicate exact lexical match mechanism can be greatly improved with the introduction of contextualized representation in COIL. COIL's token-level match also yields better fine-grained signals than dense retriever's global match signal. COIL-full further combines the lexical signals with dense CLS match, forming a system that can deal with both vocabulary and semantic mismatch, being as effective as all-to-all system.

结果表明，通过在COIL中引入上下文表示，可以极大地改进精确词法匹配机制。与稠密检索器的全局匹配信号相比，COIL的词元级匹配还能产生更好的细粒度信号。COIL - full进一步将词法信号与稠密CLS匹配相结合，形成了一个既能处理词汇不匹配又能处理语义不匹配的系统，其效果与全对全系统相当。

### 5.2 Analysis of Dimensionality

### 5.2 维度分析

The second experiment tests how varying COIL's token dimension ${n}_{t}$ and CLS dimension ${n}_{c}$ affect model effectiveness and efficiency. We record retrieval performance and latency on MARCO passage collection in Table 3.

第二个实验测试了改变COIL的词元维度${n}_{t}$和CLS维度${n}_{c}$如何影响模型的有效性和效率。我们在表3中记录了在MARCO段落集上的检索性能和延迟。

In COIL-full systems, reducing CLS dimension from 768 to 128 leads to a small drop in performance on the Dev set, indicating that a full 768 dimension may not be necessary for COIL. Keeping CLS dimension at 128, systems with token dimension 32 and 8 have very small performance difference, suggesting that token-specific semantic consumes much fewer dimensions. Similar pattern in ${n}_{t}$ is also observed in COIL-tok $\left( {{n}_{c} = 0}\right)$ .

在COIL-full系统中，将CLS维度从768降至128会导致开发集上的性能略有下降，这表明对于COIL而言，完整的768维可能并非必要。将CLS维度保持在128时，词元维度为32和8的系统在性能上差异极小，这表明词元特定的语义所需的维度要少得多。在COIL-tok $\left( {{n}_{c} = 0}\right)$ 中也观察到了与 ${n}_{t}$ 类似的模式。

On the DL2019 queries, we observe that reducing dimension actually achieves better MRR. We believe this is due to a regulatory effect, as the test queries were labeled differently from the MS-MARCO train/dev queries (Craswell et al., 2020).

在DL2019查询上，我们发现降低维度实际上能实现更好的平均倒数排名（MRR）。我们认为这是由于一种正则化效应，因为测试查询的标注方式与MS-MARCO训练/开发查询不同（Craswell等人，2020）。

<!-- Media -->

Table 3: Performance and latency of COIL systems with different representation dimensions. Results not applicable are denoted ’-’ and no available ’ $\mathrm{n}$ .a.’. Here ${n}_{c}$ denotes COIL CLS dimension and ${n}_{t}$ token vector dimension. *: ColBERT use approximate search and quantization. We exclude I/O time from measurements.

表3：不同表示维度的COIL系统的性能和延迟。不适用的结果用“-”表示，无可用结果用“ $\mathrm{n}$ .a.”表示。这里 ${n}_{c}$ 表示COIL的CLS维度， ${n}_{t}$ 表示词元向量维度。*：ColBERT使用近似搜索和量化。我们在测量中排除了输入/输出（I/O）时间。

<table><tr><td rowspan="2">Model</td><td colspan="2">Dev Retrieval</td><td colspan="2">DL2019 Retrieval</td><td colspan="2">Latency/ms</td></tr><tr><td>MRR@10</td><td>Recall@1K</td><td>NDCG@10</td><td>MRR</td><td>CPU</td><td>GPU</td></tr><tr><td>BM25</td><td>0.184</td><td>0.853</td><td>0.506</td><td>0.825</td><td>36</td><td>n.a.</td></tr><tr><td>Dense</td><td>0.304</td><td>0.932</td><td>0.635</td><td>0.898</td><td>293</td><td>32</td></tr><tr><td>ColBERT</td><td>0.360</td><td>0.968</td><td>n.a.</td><td>n.a.</td><td>458*</td><td>-</td></tr><tr><td>COIL</td><td/><td/><td/><td/><td/><td/></tr><tr><td>${n}_{c}$${n}_{t}$</td><td/><td/><td/><td/><td/><td/></tr><tr><td>76832</td><td>0.355</td><td>0.963</td><td>0.704</td><td>0.924</td><td>380</td><td>41</td></tr><tr><td>12832</td><td>0.350</td><td>0.953</td><td>0.692</td><td>0.956</td><td>125</td><td>23</td></tr><tr><td>1288</td><td>0.347</td><td>0.956</td><td>0.694</td><td>0.977</td><td>113</td><td>21</td></tr><tr><td>032</td><td>0.341</td><td>0.949</td><td>0.660</td><td>0.915</td><td>67</td><td>18</td></tr><tr><td>08</td><td>0.336</td><td>0.940</td><td>0.678</td><td>0.953</td><td>55</td><td>16</td></tr></table>

<table><tbody><tr><td rowspan="2">模型</td><td colspan="2">开发集检索</td><td colspan="2">DL2019检索</td><td colspan="2">延迟/毫秒</td></tr><tr><td>前10名平均倒数排名（MRR@10）</td><td>前1000名召回率（Recall@1K）</td><td>归一化折损累计增益@10（Normalized Discounted Cumulative Gain@10）</td><td>平均倒数排名（Mean Reciprocal Rank）</td><td>中央处理器（Central Processing Unit）</td><td>图形处理器（Graphics Processing Unit）</td></tr><tr><td>二元独立模型25（Best Matching 25）</td><td>0.184</td><td>0.853</td><td>0.506</td><td>0.825</td><td>36</td><td>不适用（not applicable）</td></tr><tr><td>密集的</td><td>0.304</td><td>0.932</td><td>0.635</td><td>0.898</td><td>293</td><td>32</td></tr><tr><td>ColBERT（原词）</td><td>0.360</td><td>0.968</td><td>不适用（not applicable）</td><td>不适用（not applicable）</td><td>458*</td><td>-</td></tr><tr><td>COIL（原词）</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>${n}_{c}$${n}_{t}$</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>76832</td><td>0.355</td><td>0.963</td><td>0.704</td><td>0.924</td><td>380</td><td>41</td></tr><tr><td>12832</td><td>0.350</td><td>0.953</td><td>0.692</td><td>0.956</td><td>125</td><td>23</td></tr><tr><td>1288</td><td>0.347</td><td>0.956</td><td>0.694</td><td>0.977</td><td>113</td><td>21</td></tr><tr><td>032</td><td>0.341</td><td>0.949</td><td>0.660</td><td>0.915</td><td>67</td><td>18</td></tr><tr><td>08</td><td>0.336</td><td>0.940</td><td>0.678</td><td>0.953</td><td>55</td><td>16</td></tr></tbody></table>

Table 4: Sample query document pairs with similarity scores produced by COIL. Tokens in examination are colored blue. Numbers in brackets are query-document vector similarities computed with vectors generated by COIL.

表4：由COIL生成的带有相似度得分的样本查询文档对。待检查的Token标记为蓝色。括号中的数字是使用COIL生成的向量计算出的查询 - 文档向量相似度。

<table><tr><td>Query Token</td><td>COIL Contextualized Exact Match Score</td><td>Relevance</td></tr><tr><td rowspan="2">what is a cabinet in govt</td><td>Cabinet [16.28] (government) A cabinet [16.75] is a body of high- ranking state officials, typically consisting of the top leaders of the ...</td><td>+</td></tr><tr><td>Cabinet [7.23] is ${20} \times  {60}$ and top is ${28} \times  {72}$ . .... I had a $2\mathrm{\;{cm}}$ granite counter- top installed with a 10 inch overhang on one side and a 14 inch...</td><td>-</td></tr><tr><td rowspan="2">what is priority pass</td><td>Priority Pass [11.61] is an independent airport lounge access program. A membership provides you with access to their network of over 700 ....</td><td>+</td></tr><tr><td>Snoqualmie Pass [7.98] is a mountain pass [6.83] that carries Interstate 90 through the Cascade Range in the U.S. State of Washington....</td><td>-</td></tr><tr><td rowspan="2">what is njstart</td><td>NJSTART is [1.25] a self-service online platform that allows vendors to manage forms, certifications, submit proposals, access training ...</td><td>+</td></tr><tr><td>Contract awardees will receive their Blanket P.O. once it is [-0.10] con- verted, and details regarding that process will also be sent...</td><td>-</td></tr></table>

<table><tbody><tr><td>查询Token</td><td>COIL上下文精确匹配得分</td><td>相关性</td></tr><tr><td rowspan="2">政府中的内阁是什么</td><td>内阁 [16.28]（政府） 内阁 [16.75] 是一个高级国家官员团体，通常由……的最高领导人组成</td><td>+</td></tr><tr><td>橱柜 [7.23] 是 ${20} \times  {60}$，台面是 ${28} \times  {72}$。……我安装了一块 $2\mathrm{\;{cm}}$ 花岗岩台面，一侧有10英寸的悬挑，另一侧有14英寸……</td><td>-</td></tr><tr><td rowspan="2">什么是优先通行卡</td><td>优先通行卡（Priority Pass）[11.61]是一个独立的机场贵宾休息室准入计划。会员资格可让您使用其覆盖700多个……的网络。</td><td>+</td></tr><tr><td>斯诺夸尔米山口（Snoqualmie Pass）[7.98]是一条位于美国华盛顿州，承载着90号州际公路穿越喀斯喀特山脉（Cascade Range）的山口[6.83]……</td><td>-</td></tr><tr><td rowspan="2">什么是新泽西州供应商技术援助与注册培训系统（NJSTART）</td><td>新泽西州供应商技术援助与注册培训系统（NJSTART）[1.25]是一个自助式在线平台，允许供应商管理表格、认证、提交提案、获取培训……</td><td>+</td></tr><tr><td>合同中标方将在总订单（Blanket P.O.）转换完成后收到该订单，有关该流程的详细信息也将一并发送……</td><td>-</td></tr></tbody></table>

<!-- Media -->

We also record CPU and GPU search latency in Table 3. Lowering COIL-full's CLS dimension from 768 to 128 gives a big speedup, making COIL faster than DPR system. Further dropping token dimensions provide some extra speedup. The COIL-tok systems run faster than COIL-full, with a latency of the same order of magnitude as the traditional BM25 system. Importantly, lower dimension COIL systems still retain a performance advantage over dense systems while being much faster. We include ColBERT's latency reported in the original paper, which was optimized by approximate search and quantization. All COIL systems have lower latency than ColBERT even though our current implementation does not use those optimization techniques. We however note that approximate search and quantization are applicable to COIL, and leave the study of speeding up COIL to future work.

我们还在表3中记录了CPU和GPU的搜索延迟。将COIL-full的CLS维度从768降低到128可以显著提高速度，使COIL比DPR系统更快。进一步降低词元维度还能带来一些额外的速度提升。COIL-tok系统的运行速度比COIL-full快，其延迟与传统的BM25系统处于同一数量级。重要的是，低维度的COIL系统在速度快得多的同时，仍比密集型系统具有性能优势。我们纳入了原论文中报告的ColBERT的延迟，该系统通过近似搜索和量化进行了优化。即使我们目前的实现没有使用这些优化技术，所有的COIL系统的延迟都比ColBERT低。不过，我们注意到近似搜索和量化适用于COIL，并将加速COIL的研究留待未来工作。

### 5.3 Case Study

### 5.3 案例研究

COIL differs from all previous embedding-based models in that it does not use a single unified embedding space. Instead, for a specific token, COIL learns an embedding space to encode and measure the semantic similarity of the token in different contexts. In this section, we show examples where COIL differentiates different senses of a word under different contexts. In Table 4, we show how the token similarity scores differ across contexts in relevant and irrelevant query document pairs.

COIL（上下文感知的开放词汇嵌入学习，Contextualized Open-vocabulary Inference Learning）与以往所有基于嵌入的模型不同，它不使用单一统一的嵌入空间。相反，对于特定的标记，COIL学习一个嵌入空间，以编码和衡量该标记在不同上下文中的语义相似度。在本节中，我们将展示COIL如何区分一个单词在不同上下文中的不同含义。在表4中，我们展示了在相关和不相关的查询 - 文档对中，标记相似度得分如何随上下文而变化。

The first query looks for "cabinet" in the context of "govt" (abbreviation for "government"). The two documents both include query token "cabinet" but of a different concept. The first one refers to the government cabinet and the second to a case or cupboard. COIL manages to match "cabinet" in the query to "cabinet" in the first document with a much higher score. In the second query, "pass" in both documents refer to the concept of permission. However, through contextualization, COIL captures the variation of the same concept and assigns a higher score to "pass" in the first document.

第一个查询是在“govt”（“government”的缩写）的上下文中查找“cabinet”。这两篇文档都包含查询标记“cabinet”，但概念不同。第一篇文档中的“cabinet”指的是政府内阁，而第二篇文档中的“cabinet”指的是箱子或橱柜。COIL能够将查询中的“cabinet”与第一篇文档中的“cabinet”进行匹配，并给出了更高的分数。在第二个查询中，两篇文档中的“pass”都指的是“许可”的概念。然而，通过上下文分析，COIL捕捉到了同一概念的差异，并为第一篇文档中的“pass”赋予了更高的分数。

Stop words like "it", "a", and "the" are commonly removed in classic exact match IR systems as they are not informative on their own. In the third query, on the other hand, we observe that COIL is able to differentiate "is" in an explanatory sentence and "is" in a passive form, assigning the first higher score to match query context.

像“it”、“a”和“the”这样的停用词在经典的精确匹配信息检索（IR）系统中通常会被去除，因为它们本身并无信息价值。另一方面，在第三个查询中，我们观察到上下文感知精确匹配索引（COIL）能够区分解释性句子中的“is”和被动语态中的“is”，并为前者赋予更高的分数以匹配查询上下文。

All examples here show that COIL can go beyond matching token surface form and introduce rich context information to estimate matching. Differences in similarity scores across mentions under different contexts demonstrate how COIL systems gain strength over lexical systems.

这里的所有示例都表明，上下文感知精确匹配索引（COIL）可以超越对词元表面形式的匹配，并引入丰富的上下文信息来评估匹配度。不同上下文中提及内容的相似度得分差异表明了上下文感知精确匹配索引（COIL）系统相对于词法系统的优势所在。

## 6 Conclusion and Future Work

## 6 结论与未来工作

Exact lexical match systems have been widely used for decades in classical IR systems and prove to be effective and efficient. In this paper, we point out a critical problem, semantic mismatch, that generally limits all IR systems based on surface token for matching. To fix semantic mismatch, we introduce contextualized exact match to differentiate the same token in different contexts, providing effective semantic-aware token match signals. We further propose contextualized inverted list (COIL) search index which swaps token statistics in inverted lists with contextualized vector representations to perform effective search.

精确词法匹配系统在经典信息检索（IR）系统中已被广泛使用了数十年，并且被证明是有效且高效的。在本文中，我们指出了一个关键问题——语义不匹配，这一问题普遍限制了所有基于表面词元进行匹配的信息检索系统。为了解决语义不匹配问题，我们引入了上下文感知精确匹配，以区分不同上下文中的相同词元，从而提供有效的语义感知词元匹配信号。我们进一步提出了上下文感知倒排索引（COIL）搜索索引，该索引用上下文向量表示替换倒排表中的词元统计信息，以实现有效的搜索。

On two large-scale ad hoc retrieval benchmarks, we find COIL substantially improves lexical retrieval and outperforms state-of-the-art dense retrieval systems. These results indicate large headroom of the simple-but-efficient exact lexical match scheme. When the introduction of contextualiza-tion handles the issue of semantic mismatch, exact match system gains the capability of modeling complicated matching patterns that were not captured by classical systems.

在两个大规模的临时检索基准测试中，我们发现COIL（上下文感知倒排索引，Contextualized Inverted Indexing）显著提高了词法检索性能，并且优于最先进的密集检索系统。这些结果表明，这种简单而高效的精确词法匹配方案仍有很大的提升空间。当引入上下文处理语义不匹配问题时，精确匹配系统获得了对经典系统无法捕捉的复杂匹配模式进行建模的能力。

Vocabulary mismatch in COIL can also be largely mitigated with a high-level CLS vector matching. The full system performs on par with more expensive and complex all-to-all match retrievers. The success of the full system also shows that dense retrieval and COIL's exact token matching give complementary effects, with COIL making up dense system's lost token level matching signals and dense solving the vocabulary mismatch probably for COIL.

通过高级的CLS（分类器，Classifier）向量匹配，COIL中的词汇不匹配问题也能在很大程度上得到缓解。完整的系统与更昂贵、更复杂的全对全匹配检索器表现相当。完整系统的成功还表明，密集检索和COIL的精确词元匹配具有互补效应，COIL弥补了密集系统丢失的词元级匹配信号，而密集检索可能解决了COIL的词汇不匹配问题。

With our COIL systems showing viable search latency, we believe this paper makes a solid step towards building next-generation index that stores semantics. At the intersection of lexical and neural systems, efficient algorithms proposed for both can push COIL towards real-world systems.

由于我们的COIL系统显示出可行的搜索延迟，我们相信本文为构建存储语义的下一代索引迈出了坚实的一步。在词法系统和神经网络系统的交叉点上，为两者提出的高效算法可以推动COIL走向实际应用系统。

## References

## 参考文献

S. Blackford, J. Demmel, J. Dongarra, I. Duff, S. Ham-marling, Greg Henry, M. Héroux, L. Kaufman, Andrew Lumsdaine, A. Petitet, R. Pozo, K. Remington, and C. Whaley. 2002. An updated set of basic linear algebra subprograms (blas). ACM Transactions on Mathematical Software, 28.

S. 布莱克福德（S. Blackford）、J. 德梅尔（J. Demmel）、J. 东加拉（J. Dongarra）、I. 达夫（I. Duff）、S. 哈马林（S. Hammarling）、格雷格·亨利（Greg Henry）、M. 埃鲁（M. Héroux）、L. 考夫曼（L. Kaufman）、安德鲁·拉姆斯代恩（Andrew Lumsdaine）、A. 佩蒂特（A. Petitet）、R. 波佐（R. Pozo）、K. 雷明顿（K. Remington）和 C. 惠利（C. Whaley）。2002 年。一组更新的基本线性代数子程序（BLAS）。《美国计算机协会数学软件汇刊》（ACM Transactions on Mathematical Software），28。

Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, and Ellen M Voorhees. 2020. Overview of the trec 2019 deep learning track. arXiv preprint arXiv:2003.07820.

尼克·克拉斯韦尔（Nick Craswell）、巴斯卡尔·米特拉（Bhaskar Mitra）、埃米内·伊尔马兹（Emine Yilmaz）、丹尼尔·坎波斯（Daniel Campos）和埃伦·M·沃里斯（Ellen M Voorhees）。2020 年。TREC 2019 深度学习赛道概述。预印本 arXiv:2003.07820。

Zhuyun Dai and J. Callan. 2019a. Context-aware sentence/passage term importance estimation for first stage retrieval. ArXiv, abs/1910.10687.

戴竹云（Zhuyun Dai）和 J. 卡兰（J. Callan）。2019a。用于第一阶段检索的上下文感知句子/段落术语重要性估计。预印本 arXiv:1910.10687。

Zhuyun Dai and J. Callan. 2020. Context-aware document term weighting for ad-hoc search. Proceedings of The Web Conference 2020.

戴竹云（Zhuyun Dai）和 J. 卡兰（J. Callan）。2020 年。用于临时搜索的上下文感知文档术语加权。2020 年万维网会议论文集。

Zhuyun Dai and Jamie Callan. 2019b. Deeper text understanding for IR with contextual neural language modeling. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR 2019, Paris, France, July 21-25, 2019, pages 985-988. ACM.

戴竹云（Zhuyun Dai）和杰米·卡兰（Jamie Callan）。2019b。利用上下文神经语言建模实现信息检索（IR）中的深度文本理解。见《第42届ACM SIGIR国际信息检索研究与发展会议论文集》（Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval），SIGIR 2019，法国巴黎，2019年7月21 - 25日，第985 - 988页。美国计算机协会（ACM）。

J. Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In NAACL-HLT.

J. 德夫林（J. Devlin）、张明伟（Ming-Wei Chang）、肯顿·李（Kenton Lee）和克里斯蒂娜·图托纳娃（Kristina Toutanova）。2019年。BERT：用于语言理解的深度双向变换器预训练。见《北美计算语言学协会 - 人机语言技术会议》（NAACL - HLT）。

Fernando Diaz, Bhaskar Mitra, and Nick Craswell. 2016. Query expansion with locally-trained word embeddings. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics.

费尔南多·迪亚兹（Fernando Diaz）、巴斯卡尔·米特拉（Bhaskar Mitra）和尼克·克拉斯韦尔（Nick Craswell）。2016年。使用局部训练的词嵌入进行查询扩展。见《计算语言学协会第54届年会论文集》（Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics）。

Debasis Ganguly, Dwaipayan Roy, Mandar Mitra, and Gareth J. F. Jones. 2015. Word embedding based generalized language model for information retrieval. In Proceedings of the 38th International ACM SIGIR Conference on Research and Development in Information Retrieval.

德巴西斯·甘古利（Debasis Ganguly）、德维帕扬·罗伊（Dwaipayan Roy）、曼达尔·米特拉（Mandar Mitra）和加雷斯·J. F. 琼斯（Gareth J. F. Jones）。2015年。基于词嵌入的通用信息检索语言模型。见《第38届ACM SIGIR国际信息检索研究与发展会议论文集》（Proceedings of the 38th International ACM SIGIR Conference on Research and Development in Information Retrieval）。

Luyu Gao, Zhuyun Dai, and Jamie Callan. 2020. Modularized transfomer-based ranking framework. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, Online, November 16-20, 2020. Association for Computational Linguistics.

高璐宇（Luyu Gao）、戴竹云（Zhuyun Dai）和杰米·卡兰（Jamie Callan）。2020年。基于模块化Transformer的排序框架。见《2020年自然语言处理经验方法会议论文集》（Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing），EMNLP 2020，线上会议，2020年11月16 - 20日。计算语言学协会。

Luyu Gao, Zhuyun Dai, and Jamie Callan. 2021a. Rethink training of BERT rerankers in multi-stage retrieval pipeline. In Advances in Information Retrieval - 43rd European Conference on IR Research, ECIR 2021, Virtual Event, March 28 - April 1, 2021, Proceedings, Part II.

高璐宇（Luyu Gao）、戴竹云（Zhuyun Dai）和杰米·卡兰（Jamie Callan）。2021a。重新思考多阶段检索流程中BERT重排器的训练。见《信息检索进展——第43届欧洲信息检索研究会议》（Advances in Information Retrieval - 43rd European Conference on IR Research），ECIR 2021，线上活动，2021年3月28日 - 4月1日，论文集，第二部分。

Luyu Gao, Zhuyun Dai, Tongfei Chen, Zhen Fan, Benjamin Van Durme, and Jamie Callan. 2021b. Complement lexical retrieval model with semantic residual embeddings. In Advances in Information Retrieval - 43rd European Conference on IR Research, ECIR 2021, Virtual Event, March 28 - April 1, 2021, Proceedings, Part I.

高璐宇（Luyu Gao）、戴竹云（Zhuyun Dai）、陈同飞（Tongfei Chen）、樊震（Zhen Fan）、本杰明·范·杜尔姆（Benjamin Van Durme）和杰米·卡兰（Jamie Callan）。2021b。用语义残差嵌入补充词法检索模型。见《信息检索进展——第43届欧洲信息检索研究会议》（Advances in Information Retrieval - 43rd European Conference on IR Research），ECIR 2021，线上活动，2021年3月28日 - 4月1日，论文集，第一部分。

J. Guo, Y. Fan, Qingyao Ai, and W. Croft. 2016. A deep relevance matching model for ad-hoc retrieval. Proceedings of the 25th ACM International on Conference on Information and Knowledge Management.

郭（Guo）、范（Fan）、艾清瑶（Qingyao Ai）和克罗夫特（W. Croft）。2016年。用于临时检索的深度相关性匹配模型。第25届ACM国际信息与知识管理会议论文集。

R. Guo, Philip Y. Sun, E. Lindgren, Quan Geng, David Simcha, Felix Chern, and S. Kumar. 2019. Accelerating large-scale inference with anisotropic vector quantization. arXiv: Learning.

郭（R. Guo）、孙（Philip Y. Sun）、林德格伦（E. Lindgren）、耿权（Quan Geng）、大卫·辛查（David Simcha）、陈（Felix Chern）和库马尔（S. Kumar）。2019年。使用各向异性向量量化加速大规模推理。预印本：学习。

Charles R. Harris, K. Jarrod Millman, Stéfan J van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, and Travis E. Oliphant. 2020. Array programming with NumPy. Nature.

查尔斯·R·哈里斯（Charles R. Harris）、K·贾罗德·米尔曼（K. Jarrod Millman）、斯特凡·J·范德沃尔特（Stéfan J van der Walt）、拉尔夫·戈默斯（Ralf Gommers）、保利·维尔塔宁（Pauli Virtanen）、大卫·库尔纳波（David Cournapeau）、埃里克·维瑟（Eric Wieser）、朱利安·泰勒（Julian Taylor）、塞巴斯蒂安·伯格（Sebastian Berg）、纳撒尼尔·J·史密斯（Nathaniel J. Smith）、罗伯特·克恩（Robert Kern）、马蒂·皮库斯（Matti Picus）、斯蒂芬·霍耶（Stephan Hoyer）、马滕·H·范克尔克维克（Marten H. van Kerkwijk）、马修·布雷特（Matthew Brett）、艾伦·霍尔丹（Allan Haldane）、海梅·费尔南德斯·德尔里奥（Jaime Fernández del Río）、马克·威比（Mark Wiebe）、佩鲁·彼得森（Pearu Peterson）、皮埃尔·热拉尔 - 马尔尚（Pierre Gérard - Marchant）、凯文·谢泼德（Kevin Sheppard）、泰勒·雷迪（Tyler Reddy）、沃伦·韦克塞瑟（Warren Weckesser）、哈米尔·阿巴西（Hameer Abbasi）、克里斯托夫·戈尔克（Christoph Gohlke）和特拉维斯·E·奥利芬特（Travis E. Oliphant）。2020年。使用NumPy进行数组编程。《自然》杂志。

Po-Sen Huang, Xiaodong He, Jianfeng Gao, Li Deng, Alex Acero, and Larry Heck. 2013. Learning deep structured semantic models for web search using clickthrough data. In Proceedings of the 22nd ACM international conference on Information & Knowledge Management.

黄伯森（Po - Sen Huang）、何晓东（Xiaodong He）、高剑锋（Jianfeng Gao）、邓力（Li Deng）、亚历克斯·阿塞罗（Alex Acero）和拉里·赫克（Larry Heck）。2013年。利用点击数据学习用于网络搜索的深度结构化语义模型。见第22届ACM国际信息与知识管理会议论文集。

Samuel Humeau, Kurt Shuster, Marie-Anne Lachaux, and J. Weston. 2020. Poly-encoders: Architectures and pre-training strategies for fast and accurate multi-sentence scoring. In ICLR.

塞缪尔·于莫（Samuel Humeau）、库尔特·舒斯特（Kurt Shuster）、玛丽 - 安妮·拉肖（Marie - Anne Lachaux）和J. 韦斯顿（J. Weston）。2020年。多编码器：用于快速准确多句子评分的架构和预训练策略。发表于国际学习表征会议（ICLR）。

J. Johnson, M. Douze, and H. Jégou. 2017. Billion-scale similarity search with gpus. ArXiv, abs/1702.08734.

J. 约翰逊（J. Johnson）、M. 杜兹（M. Douze）和H. 杰古（H. Jégou）。2017年。基于GPU的十亿级相似度搜索。预印本平台（ArXiv），论文编号：abs/1702.08734。

V. Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Yu Wu, Sergey Edunov, Danqi Chen, and W. Yih. 2020. Dense passage retrieval for open-domain question answering. ArXiv, abs/2004.04906.

V. 卡尔普欣（V. Karpukhin）、巴拉斯·奥古兹（Barlas Oğuz）、闵世元（Sewon Min）、帕特里克·刘易斯（Patrick Lewis）、余武乐德尔（Ledell Yu Wu）、谢尔盖·叶杜诺夫（Sergey Edunov）、陈丹琦（Danqi Chen）和W. 易（W. Yih）。2020年。开放域问答的密集段落检索。预印本平台（ArXiv），论文编号：abs/2004.04906。

O. Khattab and M. Zaharia. 2020. Colbert: Efficient and effective passage search via contextualized late interaction over bert. Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval.

奥·哈塔布（O. Khattab）和马·扎哈里亚（M. Zaharia）。2020年。科尔伯特（Colbert）：通过基于BERT的上下文延迟交互实现高效有效的段落搜索。第43届国际计算机协会信息检索研究与发展会议（ACM SIGIR）论文集。

Yoon Kim. 2014. Convolutional neural networks for sentence classification. In EMNLP.

尹基姆（Yoon Kim）。2014年。用于句子分类的卷积神经网络。发表于自然语言处理经验方法会议（EMNLP）。

John Lafferty and Chengxiang Zhai. 2001. Document language models, query models, and risk minimization for information retrieval. In Proceedings of the 24th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval.

约翰·拉弗蒂（John Lafferty）和翟成祥（Chengxiang Zhai）。2001年。文档语言模型、查询模型以及信息检索中的风险最小化。发表于第24届国际计算机协会信息检索研究与发展会议（ACM SIGIR）论文集。

Victor Lavrenko and W. Bruce Croft. 2001. Relevance-based language models. In Proceedings of the 24th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval.

维克多·拉夫连科（Victor Lavrenko）和W.布鲁斯·克罗夫特（W. Bruce Croft）。2001年。基于相关性的语言模型。发表于第24届国际计算机协会信息检索研究与发展会议（ACM SIGIR）论文集。

Y. Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, M. Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. ArXiv, abs/1907.11692.

刘（Y. Liu）、迈尔·奥特（Myle Ott）、纳曼·戈亚尔（Naman Goyal）、杜静菲（Jingfei Du）、曼达尔·乔希（Mandar Joshi）、陈丹琦（Danqi Chen）、奥默·利维（Omer Levy）、M.刘易斯（M. Lewis）、卢克·泽特尔莫尔（Luke Zettlemoyer）和韦塞林·斯托亚诺夫（Veselin Stoyanov）。2019年。罗伯塔（Roberta）：一种对BERT预训练方法的稳健优化。预印本平台arXiv，论文编号：abs/1907.11692。

Yi Luan, Jacob Eisenstein, Kristina Toutanova, and M. Collins. 2020. Sparse, dense, and attentional representations for text retrieval. ArXiv, abs/2005.00181.

栾轶（Yi Luan）、雅各布·艾森斯坦（Jacob Eisenstein）、克里斯蒂娜·图塔诺娃（Kristina Toutanova）和M. 柯林斯（M. Collins）。2020年。用于文本检索的稀疏、密集和注意力表示。预印本平台arXiv，论文编号：abs/2005.00181。

Sean MacAvaney, F. Nardini, R. Perego, N. Tonellotto, Nazli Goharian, and O. Frieder. 2020. Efficient document re-ranking for transformers by precomputing term representations. Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval.

肖恩·麦卡瓦尼（Sean MacAvaney）、F. 纳尔迪尼（F. Nardini）、R. 佩雷戈（R. Perego）、N. 托内洛托（N. Tonellotto）、纳兹利·戈哈里安（Nazli Goharian）和O. 弗里德（O. Frieder）。2020年。通过预计算词项表示实现变压器模型的高效文档重排序。第43届美国计算机协会信息检索研究与发展国际会议论文集。

Donald Metzler and W. Bruce Croft. 2005. A markov random field model for term dependencies. In SIGIR 2005: Proceedings of the 28th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval.

唐纳德·梅茨勒（Donald Metzler）和W. 布鲁斯·克罗夫特（W. Bruce Croft）。2005年。一种用于词项依赖关系的马尔可夫随机场模型。收录于SIGIR 2005：第28届美国计算机协会信息检索研究与发展年度国际会议论文集。

Tomas Mikolov, Ilya Sutskever, Kai Chen, G. S. Corrado, and J. Dean. 2013. Distributed representations of words and phrases and their compositionality. In NIPS.

托马斯·米科洛夫（Tomas Mikolov）、伊利亚·苏茨克维（Ilya Sutskever）、凯·陈（Kai Chen）、G. S. 科拉多（G. S. Corrado）和J. 迪恩（J. Dean）。2013年。单词和短语的分布式表示及其组合性。收录于神经信息处理系统大会（NIPS）。

Rodrigo Nogueira and Kyunghyun Cho. 2019. Passage re-ranking with bert. ArXiv, abs/1901.04085.

罗德里戈·诺盖拉（Rodrigo Nogueira）和赵京焕（Kyunghyun Cho）。2019年。使用BERT进行段落重排序。预印本平台arXiv，论文编号：abs/1901.04085。

Rodrigo Nogueira and Jimmy Lin. 2019. From doc2query to doctttttquery.

罗德里戈·诺盖拉（Rodrigo Nogueira）和吉米·林（Jimmy Lin）。2019年。从doc2query到doctttttquery。

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Te-jani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. 2019. Py-torch: An imperative style, high-performance deep learning library. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems 32. Curran Associates, Inc.

亚当·帕斯兹克（Adam Paszke）、山姆·格罗斯（Sam Gross）、弗朗西斯科·马萨（Francisco Massa）、亚当·勒雷尔（Adam Lerer）、詹姆斯·布拉德伯里（James Bradbury）、格雷戈里·查南（Gregory Chanan）、特雷弗·基林（Trevor Killeen）、林泽明（Zeming Lin）、娜塔莉亚·吉梅尔申（Natalia Gimelshein）、卢卡·安蒂加（Luca Antiga）、阿尔班·德斯梅森（Alban Desmaison）、安德里亚斯·科普夫（Andreas Kopf）、爱德华·杨（Edward Yang）、扎卡里·德维托（Zachary DeVito）、马丁·赖森（Martin Raison）、阿利汗·特贾尼（Alykhan Te - jani）、萨桑克·奇拉姆库尔蒂（Sasank Chilamkurthy）、贝努瓦·施泰纳（Benoit Steiner）、方璐（Lu Fang）、白俊杰（Junjie Bai）和苏米思·钦塔拉（Soumith Chintala）。2019年。PyTorch：一种命令式风格的高性能深度学习库。收录于H. 沃拉赫（H. Wallach）、H. 拉罗谢尔（H. Larochelle）、A. 贝格尔齐默（A. Beygelzimer）、F. 达尔谢 - 布克（F. d'Alché - Buc）、E. 福克斯（E. Fox）和R. 加内特（R. Garnett）主编的《神经信息处理系统进展32》。柯伦联合公司（Curran Associates, Inc.）。

Jeffrey Pennington, R. Socher, and Christopher D. Manning. 2014. Glove: Global vectors for word representation. In ${EMNLP}$ .

杰弗里·彭宁顿（Jeffrey Pennington）、R. 索切尔（R. Socher）和克里斯托弗·D·曼宁（Christopher D. Manning）。2014年。GloVe：用于词表示的全局向量。收录于${EMNLP}$。

Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. 2018. Deep contextualized word representations. ArXiv, abs/1802.05365.

马修·E·彼得斯（Matthew E. Peters）、马克·诺伊曼（Mark Neumann）、莫希特·伊耶尔（Mohit Iyyer）、马特·加德纳（Matt Gardner）、克里斯托弗·克拉克（Christopher Clark）、肯顿·李（Kenton Lee）和卢克·泽特尔莫耶（Luke Zettlemoyer）。2018年。深度上下文词表示。预印本库（ArXiv），编号：abs/1802.05365。

Stephen E Robertson and Steve Walker. 1994. Some simple effective approximations to the 2-poisson model for probabilistic weighted retrieval. In Proceedings of the 17th Annual International ACM-SIGIR Conference on Research and Development in Information Retrieval.

斯蒂芬·E·罗伯逊（Stephen E Robertson）和史蒂夫·沃克（Steve Walker）。1994年。对概率加权检索的双泊松模型的一些简单有效近似方法。发表于第17届ACM信息检索研究与发展国际会议论文集。

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, L. Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In ${NIPS}$ .

阿什什·瓦斯瓦尼（Ashish Vaswani）、诺姆·沙泽尔（Noam Shazeer）、尼基·帕尔马（Niki Parmar）、雅各布·乌斯库雷特（Jakob Uszkoreit）、利昂·琼斯（Llion Jones）、艾丹·N·戈麦斯（Aidan N. Gomez）、L·凯泽（L. Kaiser）和伊利亚·波洛苏金（Illia Polosukhin）。2017年。注意力就是你所需要的一切。发表于${NIPS}$。

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pier-ric Cistac, Tim Rault, Rémi Louf, Morgan Funtow-icz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. 2019. Huggingface's transformers: State-of-the-art natural language processing. ArXiv, abs/1910.03771.

托马斯·沃尔夫（Thomas Wolf）、利桑德尔·迪比（Lysandre Debut）、维克多·桑（Victor Sanh）、朱利安·肖蒙（Julien Chaumond）、克莱门特·德朗格（Clement Delangue）、安东尼·莫伊（Anthony Moi）、皮埃尔 - 里克·西斯塔克（Pier - ric Cistac）、蒂姆·劳（Tim Rault）、雷米·卢夫（Rémi Louf）、摩根·丰托维茨（Morgan Funtowicz）、乔·戴维森（Joe Davison）、山姆·施莱弗（Sam Shleifer）、帕特里克·冯·普拉滕（Patrick von Platen）、克拉拉·马（Clara Ma）、亚辛·杰尔尼（Yacine Jernite）、朱利安·普鲁（Julien Plu）、徐灿文（Canwen Xu）、特文·勒·斯考（Teven Le Scao）、西尔万·古热（Sylvain Gugger）、玛丽亚玛·德拉梅（Mariama Drame）、昆汀·勒斯特（Quentin Lhoest）和亚历山大·M·拉什（Alexander M. Rush）。2019年。Huggingface的Transformer：最先进的自然语言处理技术。《arXiv》，论文编号：abs/1910.03771。

Chenyan Xiong, Zhuyun Dai, J. Callan, Zhiyuan Liu, and R. Power. 2017. End-to-end neural ad-hoc ranking with kernel pooling. Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval.

熊晨彦（Chenyan Xiong）、戴珠云（Zhuyun Dai）、J·卡兰（J. Callan）、刘志远（Zhiyuan Liu）和R·鲍尔（R. Power）。2017年。基于核池化的端到端神经特别排序。第40届国际计算机协会信息检索研究与发展会议论文集。

Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, J. Liu, P. Bennett, Junaid Ahmed, and Arnold Over-wijk. 2020. Approximate nearest neighbor negative contrastive learning for dense text retrieval. ArXiv, abs/2007.00808.

李雄（Lee Xiong）、熊晨燕（Chenyan Xiong）、李烨（Ye Li）、邓国峰（Kwok-Fung Tang）、刘（J. Liu）、贝内特（P. Bennett）、朱奈德·艾哈迈德（Junaid Ahmed）和阿诺德·奥弗维克（Arnold Over-wijk）。2020年。用于密集文本检索的近似最近邻负对比学习。预印本平台arXiv，论文编号：abs/2007.00808。

Z. Yang, Zihang Dai, Yiming Yang, J. Carbonell, R. Salakhutdinov, and Quoc V. Le. 2019. Xlnet: Generalized autoregressive pretraining for language understanding. In NeurIPS.

杨（Z. Yang）、戴子航（Zihang Dai）、杨一鸣（Yiming Yang）、卡博内尔（J. Carbonell）、萨拉胡丁诺夫（R. Salakhutdinov）和乐国伟（Quoc V. Le）。2019年。XLNet：用于语言理解的广义自回归预训练。发表于神经信息处理系统大会（NeurIPS）。

## A Appendix

## 附录A

### A.1 Index Building Illustration

### A.1 索引构建说明

The following figure demonstrates how the document "apple pie baked ..." is indexed by COIL. The document is first processed by a fine-tuned deep LM to produce for each token a contextualized vector. The vectors of each term "apple" and "juice" are collected to the corresponding inverted list index along with the document id for lookup.

下图展示了COIL如何对文档“烤苹果派……”进行索引。首先，该文档由经过微调的深度语言模型进行处理，为每个标记生成上下文向量。每个术语“苹果”和“果汁”的向量会与文档ID一起收集到相应的倒排列表索引中，以便进行查找。

## Document #10 - apple pie baked ...

## 文档#10 - 烤苹果派……

<!-- Media -->

<!-- figureText: apple pie LM baked baked apple -->

<img src="https://cdn.noedgeai.com/01957d2e-dacf-7190-a6f0-cc115f91897c_11.jpg?x=316&y=599&w=1026&h=1191&r=0"/>

Figure 4: COIL Index Building of document "apple pie baked..."

图4：文档“烤苹果派……”的COIL索引构建

<!-- Media -->

### A.2 Search Illustration

### A.2 搜索示例

The following figure demonstrates how the query "apple juice" is processed by COIL. Contextualized vectors of each term "apple" and "juice" go to the corresponding inverted list index consisting of a lookup id array and a matrix stacked from document term vectors. For each index, a matrix vector product is run to produce an array of scores. Afterwards a max-scatter of scores followed by a sort produces the final ranking. Note for each index, we show only operations for a subset of vectors ( 3 vectors) in the index matrix.

下图展示了COIL如何处理查询“苹果汁”。每个词“苹果”和“汁”的上下文向量会进入相应的倒排列表索引，该索引由查找ID数组和从文档词向量堆叠而成的矩阵组成。对于每个索引，会进行矩阵向量乘法以生成一个得分数组。然后对得分进行最大分散处理，接着排序，得出最终排名。请注意，对于每个索引，我们仅展示了索引矩阵中部分向量（3个向量）的操作。

<!-- Media -->

<!-- figureText: Query: apple juice Matrix Vector Product Max Scatter Sorting -->

<img src="https://cdn.noedgeai.com/01957d2e-dacf-7190-a6f0-cc115f91897c_12.jpg?x=184&y=492&w=1284&h=888&r=0"/>

Figure 5: COIL Search of query "apple juice".

图5：对查询“苹果汁”的COIL搜索。

<!-- Media -->