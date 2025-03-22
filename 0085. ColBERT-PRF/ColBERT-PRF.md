# ColBERT-PRF: Semantic Pseudo-Relevance Feedback for Dense Passage and Document Retrieval

# ColBERT-PRF：用于密集段落和文档检索的语义伪相关反馈

XIAO WANG and CRAIG MACDONALD, University of Glasgow, UK

王晓（XIAO WANG）和克雷格·麦克唐纳（CRAIG MACDONALD），英国格拉斯哥大学

NICOLA TONELLOTTO, University of Pisa, Italy

尼古拉·托内洛托（NICOLA TONELLOTTO），意大利比萨大学

IADH OUNIS, University of Glasgow, UK

伊阿德·乌尼斯（IADH OUNIS），英国格拉斯哥大学

Pseudo-relevance feedback mechanisms, from Rocchio to the relevance models, have shown the usefulness of expanding and reweighting the users' initial queries using information occurring in an initial set of retrieved documents, known as the pseudo-relevant set. Recently, dense retrieval - through the use of neural contextual language models such as BERT for analysing the documents' and queries' contents and computing their relevance scores - has shown a promising performance on several information retrieval tasks still relying on the traditional inverted index for identifying documents relevant to a query. Two different dense retrieval families have emerged: the use of single embedded representations for each passage and query, e.g., using BERT's [CLS] token, or via multiple representations, e.g., using an embedding for each token of the query and document (exemplified by ColBERT). In this work, we conduct the first study into the potential for multiple representation dense retrieval to be enhanced using pseudo-relevance feedback and present our proposed approach ColBERT-PRF. In particular, based on the pseudo-relevant set of documents identified using a first-pass dense retrieval, ColBERT-PRF extracts the representative feedback embeddings from the document embeddings of the pseudo-relevant set. Among the representative feedback embeddings, the em-beddings that most highly discriminate among documents are employed as the expansion embeddings, which are then added to the original query representation. We show that these additional expansion embeddings both enhance the effectiveness of a reranking of the initial query results as well as an additional dense retrieval operation. Indeed, experiments on the MSMARCO passage ranking dataset show that MAP can be improved by up to 26% on the TREC 2019 query set and 10% on the TREC 2020 query set by the application of our proposed ColBERT-PRF method on a ColBERT dense retrieval approach. We further validate the effectiveness of our proposed pseudo-relevance feedback technique for a dense retrieval model on MSMARCO document ranking and TREC Robust04 document ranking tasks. For instance, ColBERT-PRF exhibits up to 21% and 14% improvement in MAP over the ColBERT E2E model on the MSMARCO document ranking TREC 2019 and TREC 2020 query sets, respectively. Additionally, we study the effectiveness of variants of the ColBERT-PRF model with different weighting methods. Finally, we show that ColBERT-PRF can be made more efficient,

从罗基奥算法（Rocchio）到相关模型，伪相关反馈机制已证明，利用初始检索到的文档集合（即伪相关集合）中的信息来扩展和重新加权用户的初始查询是有用的。最近，密集检索——通过使用如BERT这样的神经上下文语言模型来分析文档和查询的内容并计算它们的相关性得分——在几个仍依赖传统倒排索引来识别与查询相关文档的信息检索任务中表现出了良好的性能。出现了两种不同的密集检索类型：为每个段落和查询使用单一嵌入表示，例如使用BERT的[CLS]标记；或者通过多种表示，例如为查询和文档的每个标记使用一个嵌入（以ColBERT为例）。在这项工作中，我们首次研究了使用伪相关反馈来增强多表示密集检索的潜力，并提出了我们的方法ColBERT-PRF。具体而言，基于通过首轮密集检索识别出的伪相关文档集合，ColBERT-PRF从伪相关集合的文档嵌入中提取代表性反馈嵌入。在代表性反馈嵌入中，对文档区分度最高的嵌入被用作扩展嵌入，然后将其添加到原始查询表示中。我们表明，这些额外的扩展嵌入既提高了对初始查询结果重新排序的有效性，也提高了额外的密集检索操作的有效性。实际上，在MSMARCO段落排名数据集上的实验表明，通过在ColBERT密集检索方法上应用我们提出的ColBERT-PRF方法，在TREC 2019查询集上平均准确率均值（MAP）可提高多达26%，在TREC 2020查询集上可提高10%。我们进一步验证了我们提出的伪相关反馈技术在MSMARCO文档排名和TREC Robust04文档排名任务的密集检索模型中的有效性。例如，在MSMARCO文档排名的TREC 2019和TREC 2020查询集上，ColBERT-PRF相较于ColBERT端到端（E2E）模型，平均准确率均值（MAP）分别提高了多达21%和14%。此外，我们研究了使用不同加权方法的ColBERT-PRF模型变体的有效性。最后，我们表明，通过应用近似评分和不同的聚类方法，ColBERT-PRF可以提高效率

## This manuscript extends an earlier ICTIR 2021 publication [44].

## 本手稿扩展了之前在2021年信息与通信技术信息检索会议（ICTIR 2021）上发表的论文[44]。

Nicola Tonellotto was partially supported by the Italian government in the framework of the Progetto PNRR "CN1 - Simu-lazioni, calcolo e analisi dei dati ad alte prestazioni - Spoke 1 - Future HPC & Big Data". Xiao Wang acknowledges support by the China Scholarship Council (CSC) from the Ministry of Education of P.R. China. Craig Macdonald and Iadh Ou-nis acknowledge EPSRC grant EP/R018634/1: Closed-Loop Data Science for Complex, Computationally- & Data-Intensive Analytics.

尼古拉·托内洛托（Nicola Tonellotto）部分得到了意大利政府在“国家复苏与韧性计划项目（Progetto PNRR）‘CN1 - 高性能数据模拟、计算与分析 - 分支1 - 未来高性能计算与大数据’”框架下的支持。王晓感谢中国教育部国家留学基金管理委员会（CSC）的资助。克雷格·麦克唐纳（Craig Macdonald）和伊阿德·乌尼斯（Iadh Ounis）感谢英国工程与物理科学研究委员会（EPSRC）的资助（项目编号：EP/R018634/1）：用于复杂、计算和数据密集型分析的闭环数据科学。

Authors' addresses: X. Wang, C. Macdonald, and I. Ounis, University of Glasgow, School of Computing Science, Lilybank Gardens, G12 8QQ, Glasgow, United Kingdom; emails: x.wang.8@research.gla.ac.uk, \{craig.macdonald, iadh.ounis\}@glasgow.ac.uk; N. Tonellotto, University of Pisa, Department of Information Engineering, Via G. Caruso 16 - 56122 - Pisa, Italy; email: nicola.tonellotto@unipi.it.

作者地址：王晓（X. Wang）、克雷格·麦克唐纳（C. Macdonald）和伊阿德·乌尼斯（I. Ounis），英国格拉斯哥大学计算科学学院，利利班克花园，格拉斯哥G12 8QQ；电子邮件：x.wang.8@research.gla.ac.uk，{craig.macdonald, iadh.ounis}@glasgow.ac.uk；尼古拉·托内洛托（N. Tonellotto），意大利比萨大学信息工程系，加埃塔诺·卡鲁索路16号 - 56122 - 比萨；电子邮件：nicola.tonellotto@unipi.it。

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires

允许个人或课堂使用本作品的全部或部分内容制作数字或硬拷贝，无需付费，但前提是这些拷贝不得用于盈利或商业目的，并且拷贝必须带有此声明和首页的完整引用。必须尊重本作品中不属于美国计算机协会（ACM）的组件的版权。允许进行带引用的摘要。否则，如需复制、重新发布、上传到服务器或分发给列表，则需要

prior specific permission and/or a fee. Request permissions from permissions@acm.org.

事先获得特定许可和/或支付费用。请向permissions@acm.org请求许可。

© 2023 Association for Computing Machinery.

© 2023美国计算机协会。

1559-1131/2023/01-ART3 \$15.00

1559 - 1131/2023/01 - ART3 15.00美元

https://doi.org/10.1145/3572405 attaining up to ${4.54} \times$ speedup over the default ColBERT-PRF model,and with little impact on effectiveness, through the application of approximate scoring and different clustering methods.

https://doi.org/10.1145/3572405 通过应用近似评分和不同的聚类方法，与默认的ColBERT - PRF模型相比，可实现高达${4.54} \times$的加速，且对有效性影响很小。

CCS Concepts: - Information systems $\rightarrow$ Information retrieval query processing; Information retrieval;

计算机协会概念分类体系（CCS Concepts）： - 信息系统 $\rightarrow$ 信息检索查询处理；信息检索

Additional Key Words and Phrases: Query expansion, pseudo-relevance feedback, BERT, dense retrieval

其他关键词和短语：查询扩展、伪相关反馈、BERT（双向编码器表征变换器）、密集检索

## ACM Reference format:

## 美国计算机协会（ACM）引用格式：

Xiao Wang, Craig Macdonald, Nicola Tonellotto, and Iadh Ounis. 2023. ColBERT-PRF: Semantic Pseudo-Relevance Feedback for Dense Passage and Document Retrieval. ACM Trans. Web 17, 1, Article 3 (January 2023), 39 pages.

小王、克雷格·麦克唐纳、尼古拉·托内洛托和伊阿德·乌尼斯。2023 年。ColBERT - PRF：用于密集段落和文档检索的语义伪相关反馈。《ACM 网络汇刊》17 卷 1 期，文章编号 3（2023 年 1 月），39 页。

https://doi.org/10.1145/3572405

## 1 INTRODUCTION

## 1 引言

When searching for information, users often formulate queries in a different way to the relevant documents. For instance, a user may search for information about "surname meaning" using a query "where do last names come from". However, a relevant document may describe the "last name" using "family name" or "surname" and may use terms such as "originate" or "history" instead of "come from". Thus, a relevant document and the user query might form a lexical mismatch gap during retrieval, which must be bridged for effective retrieval.

在搜索信息时，用户表述查询的方式通常与相关文档不同。例如，用户可能使用“姓氏从何而来”这样的查询来搜索关于“姓氏含义”的信息。然而，相关文档可能用“家族姓氏”或“姓氏”来描述“姓氏”，并且可能使用“起源”或“历史”等术语，而不是“从何而来”。因此，在检索过程中，相关文档和用户查询可能会形成词汇不匹配的差距，为了实现有效检索，必须弥合这一差距。

Query expansion approaches, which rewrite the user's query, have been shown to be an effective approach to alleviate the vocabulary discrepancies between the user query and the relevant documents, by modifying the user's original query to improve the retrieval effectiveness. Many approaches follow the pseudo-relevance feedback (PRF) paradigm - such as Rocchio's algorithm [37], the RM3 relevance language model [1], or the DFR query expansion models [4] - where terms appearing in the top-ranked documents for the initial query are used to expand it. Query expansion (QE) approaches have also found a useful role when integrated with effective BERT-based neural reranking models, by providing a high quality set of candidate documents obtained using the expanded query,which can then be reranked [35, 42, 47].

查询扩展方法通过重写用户的查询，已被证明是一种有效的方法，可以缓解用户查询和相关文档之间的词汇差异。该方法通过修改用户的原始查询来提高检索效果。许多方法遵循伪相关反馈（PRF）范式，如罗基奥算法 [37]、RM3 相关语言模型 [1] 或基于文档频率排名（DFR）的查询扩展模型 [4]，这些方法使用初始查询的排名靠前的文档中出现的术语来扩展查询。查询扩展（QE）方法在与基于 BERT 的有效神经重排序模型集成时也发挥了有用的作用，它通过使用扩展后的查询获得高质量的候选文档集，然后对这些文档进行重排序 [35, 42, 47]。

On the other hand, many studies have focused on the use of static word embeddings, such as Word2Vec,within query expansion methods $\left\lbrack  {{12},{19},{39},{40}}\right\rbrack$ . Indeed,most of the existing embedding-based QE methods $\left\lbrack  {{12},{19},{39},{40},{49}}\right\rbrack$ are based on static embeddings,where a word embedding is always the same within different sentences, and hence they do not address contex-tualised language models such as BERT. Recently, CEQE [29] was proposed, which makes use of contextualised BERT embeddings for query expansion. The resulting refined query representation is then used for a further round of retrieval using a traditional (sparse) inverted index. In contrast, in this paper, we focus on implementing contextualised embedding-based query expansion for dense retrieval.

另一方面，许多研究专注于在查询扩展方法中使用静态词嵌入，如 Word2Vec $\left\lbrack  {{12},{19},{39},{40}}\right\rbrack$。实际上，大多数现有的基于嵌入的查询扩展方法 $\left\lbrack  {{12},{19},{39},{40},{49}}\right\rbrack$ 都基于静态嵌入，其中一个词的嵌入在不同句子中始终相同，因此它们无法处理像 BERT 这样的上下文语言模型。最近，有人提出了 CEQE [29] 方法，该方法利用上下文相关的 BERT 嵌入进行查询扩展。然后，将得到的精炼查询表示用于使用传统（稀疏）倒排索引进行的下一轮检索。相比之下，在本文中，我们专注于为密集检索实现基于上下文嵌入的查询扩展。

Indeed, the BERT models have demonstrated further promise in being a suitable basis for dense retrieval. In particular, instead of using a classical inverted index, in dense retrieval, the documents and queries are represented using embeddings. Then, the documents can be retrieved using an approximate nearest neighbour algorithm - as exemplified by the FAISS toolkit [15]. Two distinct families of approaches have emerged: single representation dense retrieval and multiple representation dense retrieval. In single representation dense retrieval, as used by DPR [16] and ANCE [46], each query or document is represented entirely by the single embedding of the [CLS] (classification) token computed by BERT. Query-document relevance is estimated in terms of the similarity of the corresponding [CLS] embeddings. In contrast, in multiple representation dense retrieval - as proposed by ColBERT [17] - each term of the queries and documents is represented by a single embedding. For each query embedding, one per query term, the nearest document token embed-dings are identified using an approximate nearest neighbour search, before a final re-scoring to obtain exact relevance estimations. Although it has been found that performing information retrieval based on the contextualised representation of the query and document can alleviate both the lexical mismatch, for instance, "last name" and "surname" and the semantic mismatch, for instance, "I like an apple" and "I like Apple airpods" [36]. We argue that, as users issue the query prior to the access to the relevant documents, the users' queries can still be insufficiently well represented within the dense retrieval paradigm, and as a consequence, this representation can be improved by access to a pseudo-relevant set.

实际上，BERT 模型已显示出作为密集检索合适基础的进一步潜力。特别是，在密集检索中，文档和查询使用嵌入来表示，而不是使用传统的倒排索引。然后，可以使用近似最近邻算法来检索文档，如 FAISS 工具包 [15] 所示。目前出现了两种不同的方法家族：单表示密集检索和多表示密集检索。在单表示密集检索中，如分布式段落表示（DPR）[16] 和自适应神经上下文编码器（ANCE）[46] 所采用的方法，每个查询或文档完全由 BERT 计算的 [CLS]（分类）标记的单个嵌入表示。查询 - 文档相关性根据相应 [CLS] 嵌入的相似度来估计。相比之下，在多表示密集检索中，如 ColBERT [17] 所提出的方法，查询和文档的每个术语都由单个嵌入表示。对于每个查询嵌入（每个查询术语一个），在进行最终重新评分以获得精确相关性估计之前，使用近似最近邻搜索来识别最近的文档标记嵌入。尽管已经发现基于查询和文档的上下文表示进行信息检索可以缓解词汇不匹配问题（例如“姓氏”和“surname”）和语义不匹配问题（例如“我喜欢苹果”和“我喜欢苹果耳机”）[36]。但我们认为，由于用户在访问相关文档之前就发出查询，因此在密集检索范式中，用户的查询仍然可能无法得到充分的表示，因此可以通过访问伪相关集来改进这种表示。

Indeed, in this work, we are concerned with applying pseudo-relevance feedback in a multiple representation dense retrieval setting. Indeed, as retrieval uses multiple representations, this allows additional useful embeddings to be appended to the query representation. Furthermore, the exact scoring stage provides the document embeddings in response to the original query, which can be used as pseudo-relevance information.

实际上，在这项工作中，我们关注的是在多表示密集检索环境中应用伪相关反馈。实际上，由于检索使用多表示，这允许将额外有用的嵌入附加到查询表示中。此外，精确评分阶段会根据原始查询提供文档嵌入，这些嵌入可以用作伪相关信息。

Thus, in this work, we propose a pseudo-relevance feedback mechanism called ColBERT-PRF for dense retrieval. In particular, as embeddings cannot be counted, ColBERT-PRF applies clustering to the embeddings occurring in the pseudo-relevant set, and then identifies the most discriminative embeddings among the cluster centroids. These centroids are then appended to the embeddings of the original query. ColBERT-PRF is focussed on multiple representation dense retrieval settings; However, compared to existing work, our approach is the first work to apply pseudo-relevance feedback to any form of dense retrieval setting; moreover, among the existing approaches applying deep learning for pseudo-relevance feedback, our work in this paper is the first that can improve the recall of the candidate set by re-executing the expanded query representation upon the dense retrieval index, and thereby identify more relevant documents that can be highly ranked for the user. In summary, a preliminary version of this paper appeared in ICTIR 2021 [44] which made the following contributions: (1) we propose a novel contextualised pseudo-relevance feedback mechanism for multiple representation dense retrieval; (2) we cluster and rank the feedback document embeddings for selecting candidate expansion embeddings; (3) we evaluate our proposed contex-tualised PRF model in both ranking and reranking settings. In this work, we extend our previous work and thus make the following additional contributions: (4) we demonstrate the effectiveness of ColBERT-PRF model on document ranking tasks, using the MSMARCO document test collection and the TREC Robust04 test collections; (5) We further investigate the effectiveness of ColBERT-PRF by varying the selection of the expansion embeddings. (6) We thoroughly investigate the trade-off between the effectiveness and the efficiency of ColBERT-PRF.

因此，在这项工作中，我们提出了一种名为ColBERT - PRF的伪相关反馈机制，用于密集检索。具体而言，由于嵌入向量无法直接计数，ColBERT - PRF对伪相关集中出现的嵌入向量进行聚类，然后在聚类中心中识别出最具区分性的嵌入向量。这些中心向量随后会被添加到原始查询的嵌入向量中。ColBERT - PRF专注于多表示密集检索设置；然而，与现有工作相比，我们的方法是首个将伪相关反馈应用于任何形式的密集检索设置的工作；此外，在现有的应用深度学习进行伪相关反馈的方法中，本文的工作是首个能够通过在密集检索索引上重新执行扩展后的查询表示来提高候选集召回率的，从而为用户识别出更多可以获得高排名的相关文档。总之，本文的初步版本发表于ICTIR 2021 [44]，做出了以下贡献：(1) 我们为多表示密集检索提出了一种新颖的上下文伪相关反馈机制；(2) 我们对反馈文档嵌入向量进行聚类和排序，以选择候选扩展嵌入向量；(3) 我们在排序和重排序设置中评估了我们提出的上下文伪相关反馈（PRF）模型。在这项工作中，我们扩展了之前的工作，因此做出了以下额外贡献：(4) 我们使用MSMARCO文档测试集和TREC Robust04测试集，证明了ColBERT - PRF模型在文档排序任务中的有效性；(5) 我们通过改变扩展嵌入向量的选择，进一步研究了ColBERT - PRF的有效性。(6) 我们深入研究了ColBERT - PRF的有效性和效率之间的权衡。

The remainder of this paper is as follows: Section 2 positions this work among existing approaches to pseudo-relevance feedback; Section 3 describes a multi-representation dense retrieval, while Section 4 presents our proposed dense PRF method. Next, we discuss the effectiveness of ColBERT-PRF for passage ranking task and for document ranking task in Sections 5 and 6, respectively. Next, we discuss the usefulness of different weighting methods for measuring the informativeness of the expansion embeddings of ColBERT-PRF in Section 7. In Section 8, we study efficient variants of ColBERT-PRF. Finally, we provide concluding remarks and a discussion of future directions in Section 9.

本文的其余部分安排如下：第2节将这项工作置于现有的伪相关反馈方法之中；第3节描述了多表示密集检索，而第4节介绍了我们提出的密集伪相关反馈（PRF）方法。接下来，我们分别在第5节和第6节讨论ColBERT - PRF在段落排序任务和文档排序任务中的有效性。然后，我们在第7节讨论不同加权方法对于衡量ColBERT - PRF扩展嵌入向量信息量的有用性。在第8节，我们研究ColBERT - PRF的高效变体。最后，我们在第9节给出结论并讨论未来的研究方向。

## 2 RELATED WORK

## 2 相关工作

Pseudo-relevance feedback approaches have a long history in Information Retrieval (IR) going back to Rocchio [37] who generated refined query reformulations through linear combinations of the sparse vectors, e.g., containing term frequency information representing the query and the top-ranked feedback documents. Refined classical PRF models, such as Divergence from Randomness's Bo1 [4], KL [2], and RM3 relevance models [1] have demonstrated their effectiveness on many test collections. Typically, these models identify and weight feedback terms that are frequent in the feedback documents and infrequent in the corpus, by exploiting statistical information about the occurrence of terms in the documents and in the whole collection. In all cases, the reformulated query is then re-executed on the traditional (so-called sparse) inverted index.

伪相关反馈方法在信息检索（IR）领域有着悠久的历史，可以追溯到Rocchio [37]，他通过稀疏向量（例如包含表示查询和排名靠前的反馈文档的词频信息）的线性组合生成精炼的查询重写。精炼的经典伪相关反馈（PRF）模型，如偏离随机性的Bo1 [4]、KL [2]和RM3相关模型 [1]，已经在许多测试集上证明了它们的有效性。通常，这些模型通过利用文档和整个集合中词项出现的统计信息，识别并加权在反馈文档中频繁出现但在语料库中不常出现的反馈词项。在所有情况下，重写后的查询随后会在传统的（所谓的稀疏）倒排索引上重新执行。

Recently, deep learning solutions based on transformer networks have been used to enrich the statistical information about terms by rewriting or expanding the collection of documents. For instance, DeepCT [10] reweights terms occurring in the documents according to a fine-tuned BERT model to highlight important terms. This results in augmented document representations, which can be indexed using a traditional inverted indexer. Similarly, doc2query [33] and its more modern variant docT5query [32] apply text-to-text translation models to each document in the collection to suggest queries that may be relevant to the document. When the suggested queries are indexed along with the original document, the retrieval effectiveness is enhanced.

最近，基于Transformer网络的深度学习解决方案已被用于通过重写或扩展文档集合来丰富关于词项的统计信息。例如，DeepCT [10]根据微调后的BERT模型对文档中出现的词项重新加权，以突出重要的词项。这会产生增强的文档表示，可以使用传统的倒排索引器进行索引。类似地，doc2query [33]及其更现代的变体docT5query [32]对集合中的每个文档应用文本到文本的翻译模型，以提出可能与该文档相关的查询。当建议的查询与原始文档一起被索引时，检索效果会得到增强。

More recently, instead of leveraging (augmented) statistical information such as the in-document and collection frequency of terms to model a query or a document, dense representations, also known as embeddings, are becoming commonplace. Embeddings encode terms in queries and documents by learning a vector representation for each term, which takes into account the word semantic and context. Instead of identifying the related terms in the pseudo-relevance feedback documents using statistical methods, embedding-based query expansion methods $\left\lbrack  {{12},{19},{39},{40},{49}}\right\rbrack$ expand a query with terms that are closest to the query terms in the word embedding space. However, the expansion terms may not be sufficiently informative to distinguish relevant documents from non-relevant documents - for instance, the embedding of "grows" may be closest to "grow" in the embedding space, but adding "grows" to the query may not help to identify more relevant documents. Moreover, all these embedding-based method are based on non-contextualised embeddings, where a word embedding is always the same within different sentences, and hence they do not address contextualised language models. Pre-trained contextualised language models such as BERT [11] have brought large effectiveness improvements over prior art in information retrieval tasks. In particular, deep learning is able to successfully exploit general language features in order to capture the contextual semantic signals allowing to better estimate the relevance of documents w.r.t. a given query.

最近，不再利用（增强的）统计信息（如词项在文档内和集合中的频率）来对查询或文档进行建模，密集表示（也称为嵌入向量）正变得越来越普遍。嵌入向量通过为每个词项学习一个向量表示来对查询和文档中的词项进行编码，该表示考虑了词的语义和上下文。基于嵌入向量的查询扩展方法$\left\lbrack  {{12},{19},{39},{40},{49}}\right\rbrack$不再使用统计方法在伪相关反馈文档中识别相关词项，而是用在词嵌入空间中与查询词项最接近的词项来扩展查询。然而，扩展词项可能不足以区分相关文档和非相关文档——例如，在嵌入空间中，“grows”的嵌入向量可能与“grow”最接近，但将“grows”添加到查询中可能无助于识别更多相关文档。此外，所有这些基于嵌入向量的方法都基于非上下文嵌入，即一个词的嵌入向量在不同句子中始终相同，因此它们没有考虑上下文语言模型。像BERT [11]这样的预训练上下文语言模型在信息检索任务中比现有技术有了很大的效果提升。特别是，深度学习能够成功利用通用语言特征来捕捉上下文语义信号，从而更好地估计文档相对于给定查询的相关性。

Query expansion approaches have been used for generating a high quality pool of candidate documents to be reranked by effective BERT-based neural reranking models [35, 42, 47]. However, the use of BERT models directly within the pseudo-relevance feedback mechanism has seen comparatively little use in the literature. The current approaches leveraging the BERT contextualised embeddings for PRF are Neural PRF [20], BERT-QE [51], and CEQE [29].

查询扩展方法已被用于生成高质量的候选文档池，以供基于BERT的有效神经重排模型[35, 42, 47]进行重排。然而，在文献中，将BERT模型直接用于伪相关反馈机制的情况相对较少。目前利用BERT上下文嵌入进行伪相关反馈（PRF）的方法有神经伪相关反馈（Neural PRF）[20]、BERT查询扩展（BERT - QE）[51]和上下文嵌入查询扩展（CEQE）[29]。

In particular, Neural PRF uses neural ranking models, such as DRMM [14] and KNRM [45], to score the similarity of a document to a top-ranked feedback document. BERT-QE is conceptually similar to Neural PRF, but it measures the similarity of each document w.r.t. feedback chunks that are extracted from the top-ranked feedback documents. This results in an expensive application of many BERT computations - approximately ${11} \times$ as many GPU operations than a simple BERT reranker [51]. Both Neural PRF and BERT-QE approaches leverage contextualised language models to rerank an initial ranking of documents retrieved by a preliminary sparse retrieval system. However, they cannot identify any new relevant documents from the collection that were not retrieved in the initial ranking.

具体来说，神经伪相关反馈（Neural PRF）使用神经排序模型，如深度交互匹配模型（DRMM）[14]和核化神经匹配模型（KNRM）[45]，来对文档与排名靠前的反馈文档的相似度进行评分。BERT查询扩展（BERT - QE）在概念上与神经伪相关反馈（Neural PRF）类似，但它衡量每个文档相对于从排名靠前的反馈文档中提取的反馈块的相似度。这导致需要进行大量的BERT计算——大约比简单的BERT重排器多${11} \times$倍的GPU操作[51]。神经伪相关反馈（Neural PRF）和BERT查询扩展（BERT - QE）这两种方法都利用上下文语言模型对初步稀疏检索系统检索到的文档初始排名进行重排。然而，它们无法从集合中识别出在初始排名中未被检索到的任何新的相关文档。

Meanwhile, Rocchio's relevance feedback algorithm has also been implemented for a learned sparse index by SNRM [50]. However, this model relies on a sparse index representation, which loses the advantages of dense retrieval. CEQE exploits BERT to compute contextualised representations for the query as well as for the terms in the top-ranked feedback documents, and then selects as expansion terms those which are the closest to the query embeddings according to some similarity measure. In contrast to Neural PRF and BERT-QE, CEQE is used to generate a new query of terms for execution upon a traditional (sparse) inverted index. This means that the contextual meaning of an expansion term is lost - for instance, a polysemous word added to the query can result in a topic drift.

同时，SNRM [50]也为学习到的稀疏索引实现了罗基奥（Rocchio）相关反馈算法。然而，该模型依赖于稀疏索引表示，这失去了密集检索的优势。上下文嵌入查询扩展（CEQE）利用BERT为查询以及排名靠前的反馈文档中的词项计算上下文表示，然后根据某种相似度度量选择与查询嵌入最接近的词项作为扩展词项。与神经伪相关反馈（Neural PRF）和BERT查询扩展（BERT - QE）不同，上下文嵌入查询扩展（CEQE）用于生成一个新的词项查询，以便在传统（稀疏）倒排索引上执行。这意味着扩展词项的上下文含义丢失了——例如，将一个多义词添加到查询中可能会导致主题漂移。

In contrast to the aforementioned approaches, our proposed ColBERT-PRF approach can be exploited in a dense retrieval system, both in end-to-end ranking and reranking scenarios. Dense retrieval approaches, exemplified by ANCE [46] and ColBERT [17], are of increasing interest, due to their use of the BERT embedding(s) for representing queries and documents. By using directly the BERT embeddings for retrieval, topic drifts for polysemous words can be avoided. Concurrently to our work, ANCE-PRF [22, 48] has been proposed to improve the effectiveness for a single representation ANCE model by retraining the query encoder using pseudo-relevance feedback information. In contrast, our work doesn't require any further training. To the best of our knowledge, ColBERT-PRF is the first work investigating PRF for a multiple representation dense retrieval setting.

与上述方法不同，我们提出的ColBERT - PRF方法可用于密集检索系统，适用于端到端排序和重排场景。以ANCE [46]和ColBERT [17]为代表的密集检索方法越来越受到关注，因为它们使用BERT嵌入来表示查询和文档。通过直接使用BERT嵌入进行检索，可以避免多义词的主题漂移。与我们的工作同时，有人提出了ANCE - PRF [22, 48]，通过使用伪相关反馈信息重新训练查询编码器来提高单表示ANCE模型的有效性。相比之下，我们的工作不需要任何进一步的训练。据我们所知，ColBERT - PRF是第一项针对多表示密集检索设置研究伪相关反馈（PRF）的工作。

## 3 MULTI REPRESENTATION DENSE RETRIEVAL

## 3 多表示密集检索

The queries and documents are represented by tokens from a vocabulary $V$ . Each token occurrence has a contextualised real-valued vector with dimension $d$ ,called an embedding. More formally,let $f : {V}^{n} \rightarrow  {\mathbb{R}}^{n \times  d}$ be a function mapping a sequence of terms $\left\{  {{t}_{1},\ldots ,{t}_{n}}\right\}$ ,representing a query $q$ , composed by $\left| q\right|$ tokens into a set of embeddings $\left\{  {{\phi }_{{q}_{1}},\ldots ,{\phi }_{{q}_{\left| q\right| }}}\right\}$ and a document composed by $\left| d\right|$ tokens into a set of embeddings $\left\{  {{\phi }_{{d}_{1}},\ldots ,{\phi }_{{d}_{\left| d\right| }}}\right\}$ .

查询和文档由词汇表 $V$ 中的词元（token）表示。每个词元的出现都有一个维度为 $d$ 的上下文实值向量，称为嵌入（embedding）。更正式地说，设 $f : {V}^{n} \rightarrow  {\mathbb{R}}^{n \times  d}$ 是一个函数，它将表示查询 $q$ 的词项序列 $\left\{  {{t}_{1},\ldots ,{t}_{n}}\right\}$（由 $\left| q\right|$ 个词元组成）映射到一组嵌入 $\left\{  {{\phi }_{{q}_{1}},\ldots ,{\phi }_{{q}_{\left| q\right| }}}\right\}$，并将由 $\left| d\right|$ 个词元组成的文档映射到一组嵌入 $\left\{  {{\phi }_{{d}_{1}},\ldots ,{\phi }_{{d}_{\left| d\right| }}}\right\}$。

Khattab & Zaharia [17] recommended that the number of query embeddings be 32 , with extra [MASK] tokens being used as query augmentation. Indeed, these mask tokens are a differentiable mechanism that allows documents to gain score contributions from embeddings that do not actually occur in the query, but which the model assumes could be present in the query. In practice, as we later show in Section 4.4, the [MASK] embeddings are very similar to embeddings of the existing query tokens, and hence cannot be considered as a form of query expansion. Moreover, they do not make use of pseudo-relevance feedback information obtained from the top-ranked documents of the original query, which has repeatedly been shown to be an effective source to improve query representations.

卡塔布（Khattab）和扎哈里亚（Zaharia）[17] 建议查询嵌入的数量为 32 个，并使用额外的 [MASK] 词元进行查询增强。实际上，这些掩码词元是一种可微机制，它允许文档从实际上未在查询中出现但模型假设可能存在于查询中的嵌入中获得得分贡献。实际上，正如我们在后面的 4.4 节中所示，[MASK] 嵌入与现有查询词元的嵌入非常相似，因此不能被视为一种查询扩展形式。此外，它们没有利用从原始查询的排名靠前的文档中获得的伪相关反馈信息，而该信息已多次被证明是改进查询表示的有效来源。

The similarity of two embeddings is computed by the dot product. Hence,for a query $q$ and a document $d$ ,their similarity score $s\left( {q,d}\right)$ is obtained by summing the maximum similarity between the query token embeddings and the document token embeddings [17]:

两个嵌入的相似度通过点积计算。因此，对于查询 $q$ 和文档 $d$，它们的相似度得分 $s\left( {q,d}\right)$ 通过对查询词元嵌入和文档词元嵌入之间的最大相似度求和得到 [17]：

$$
s\left( {q,d}\right)  = \mathop{\sum }\limits_{{i = 1}}^{\left| q\right| }\mathop{\max }\limits_{{j = 1,\ldots ,\left| d\right| }}{\phi }_{{q}_{i}}^{T}{\phi }_{{d}_{j}} \tag{1}
$$

Indeed,Formal et al. [13] showed that the dot product ${\phi }_{{q}_{i}}^{T}{\phi }_{{d}_{j}}$ used by ColBERT implicitly encapsulates token importance, by giving higher scores to tokens that have higher IDF values.

实际上，福尔马尔（Formal）等人 [13] 表明，ColBERT 使用的点积 ${\phi }_{{q}_{i}}^{T}{\phi }_{{d}_{j}}$ 通过给具有较高逆文档频率（IDF）值的词元赋予更高的分数，隐式地封装了词元的重要性。

To obtain a first set of candidate documents, Khattab & Zaharia [17] make use of FAISS, an approximate nearest neighbour search library, on the pre-computed document embeddings. Conceptually,FAISS allows to retrieve the ${k}^{\prime }$ documents containing the nearest neighbour document embeddings to a query embedding ${\phi }_{{q}_{i}}$ ,i.e.,it provides a function ${\mathcal{F}}_{d}\left( {{\phi }_{{q}_{i}},{k}^{\prime }}\right)  \rightarrow  \left( {d,\ldots }\right)$ that returns a list of ${k}^{\prime }$ documents,sorted in decreasing approximate scores.

为了获得第一组候选文档，卡塔布（Khattab）和扎哈里亚（Zaharia）[17] 在预先计算的文档嵌入上使用了 FAISS（一种近似最近邻搜索库）。从概念上讲，FAISS 允许检索包含与查询嵌入 ${\phi }_{{q}_{i}}$ 最接近的文档嵌入的 ${k}^{\prime }$ 个文档，即它提供了一个函数 ${\mathcal{F}}_{d}\left( {{\phi }_{{q}_{i}},{k}^{\prime }}\right)  \rightarrow  \left( {d,\ldots }\right)$，该函数返回一个按近似得分降序排序的 ${k}^{\prime }$ 个文档的列表。

However, these approximate scores are insufficient for accurately depicting the similarity scores of the documents, hence the accurate final document scores are computed using Equation (1) in a second pass. Typically,for each query embedding,the nearest ${k}^{\prime } = 1,{000}$ documents are identified.

然而，这些近似得分不足以准确描述文档的相似度得分，因此在第二轮中使用公式 (1) 计算准确的最终文档得分。通常，对于每个查询嵌入，会识别出最接近的 ${k}^{\prime } = 1,{000}$ 个文档。

<!-- Media -->

Table 1. Summary of Notation - Top Group for ColBERT Dense Retrieval; Bottom Group for ColBERT-PRF

表 1. 符号总结 - 上方为 ColBERT 密集检索；下方为 ColBERT - PRF

<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td>${\phi }_{{q}_{i}},{\phi }_{{d}_{j}}$</td><td>An embedding for a query token ${q}_{i}$ or a doc- ument token ${d}_{i}$</td></tr><tr><td>${\mathcal{F}}_{d}\left( {{\phi }_{{q}_{i}},{k}^{\prime }}\right)$</td><td>Function returning a list of the ${k}^{\prime }$ documents closest to embedding ${\phi }_{{q}_{i}}$</td></tr><tr><td>$\Phi$</td><td>Set of feedback embeddings from ${f}_{b}$ top- ranked feedback documents</td></tr><tr><td>${v}_{i}$</td><td>A representative (centroid) embedding se- lected by applying KMeans among $\Phi$</td></tr><tr><td>$K$</td><td>Number of representative embeddings to se- lect, i.e., number of clusters for KMeans</td></tr><tr><td>${\mathcal{F}}_{t}\left( {{v}_{i},r}\right)$</td><td>Function returning the $r$ token ids corre- sponding to the $r$ closest document embed- dings to embedding ${v}_{i}$</td></tr><tr><td>${\sigma }_{i}$</td><td>Importance score of ${v}_{i}$</td></tr><tr><td>${F}_{e}$</td><td>Set of expansion embeddings</td></tr><tr><td>${f}_{e}$</td><td>Number of expansion embeddings selected from $K$ representative embeddings</td></tr><tr><td>${f}_{b}$</td><td>Number of feedback documents</td></tr><tr><td>$\beta$</td><td>Parameter weighting the contribution of the expansion embeddings</td></tr></table>

<table><tbody><tr><td>符号</td><td>含义</td></tr><tr><td>${\phi }_{{q}_{i}},{\phi }_{{d}_{j}}$</td><td>查询标记 ${q}_{i}$ 或文档标记 ${d}_{i}$ 的嵌入向量</td></tr><tr><td>${\mathcal{F}}_{d}\left( {{\phi }_{{q}_{i}},{k}^{\prime }}\right)$</td><td>返回与嵌入向量 ${\phi }_{{q}_{i}}$ 最接近的 ${k}^{\prime }$ 个文档列表的函数</td></tr><tr><td>$\Phi$</td><td>来自 ${f}_{b}$ 个排名最高的反馈文档的反馈嵌入向量集合</td></tr><tr><td>${v}_{i}$</td><td>通过在 $\Phi$ 中应用 K 均值算法（KMeans）选择的代表性（质心）嵌入向量</td></tr><tr><td>$K$</td><td>要选择的代表性嵌入向量的数量，即 K 均值算法（KMeans）的聚类数量</td></tr><tr><td>${\mathcal{F}}_{t}\left( {{v}_{i},r}\right)$</td><td>返回与嵌入向量 ${v}_{i}$ 最接近的 $r$ 个文档嵌入向量对应的 $r$ 个标记 ID 的函数</td></tr><tr><td>${\sigma }_{i}$</td><td>${v}_{i}$ 的重要性得分</td></tr><tr><td>${F}_{e}$</td><td>扩展嵌入向量集合</td></tr><tr><td>${f}_{e}$</td><td>从 $K$ 个代表性嵌入向量中选择的扩展嵌入向量的数量</td></tr><tr><td>${f}_{b}$</td><td>反馈文档的数量</td></tr><tr><td>$\beta$</td><td>对扩展嵌入向量的贡献进行加权的参数</td></tr></tbody></table>

<!-- Media -->

The set formed by the union of these documents are reranked ${}^{1}$ using Equation (1). A separate index data structure (typically in memory) is used to store the uncompressed embeddings for each document. To the best of our knowledge, ColBERT [17] exemplifies the implementation of an end-to-end IR system that uses multiple representation. Algorithm 1 summarises the ColBERT retrieval algorithm for the end-to-end dense retrieval approach proposed by Khattab & Zaharia, while the top part of Table 1 summarises the notation for the main components of the algorithm.

通过合并这些文档形成的集合使用公式 (1) 进行重新排序 ${}^{1}$。使用一个单独的索引数据结构（通常在内存中）来存储每个文档的未压缩嵌入向量。据我们所知，ColBERT [17] 是使用多种表示的端到端信息检索（IR）系统实现的范例。算法 1 总结了 Khattab 和 Zaharia 提出的端到端密集检索方法的 ColBERT 检索算法，而表 1 的上半部分总结了该算法主要组件的符号表示。

The easy access to the document embeddings used by ColBERT provides an excellent basis for our dense retrieval pseudo-relevance feedback approach. Indeed, while the use of embeddings in ColBERT addresses the vocabulary mismatch problem, we argue that identifying more related embeddings from the top-ranked documents may help to further refine the document ranking. In particular, as we will show, this permits representative embeddings from a set of pseudo-relevance documents to be used to refine the query representation $\phi$ .

ColBERT 所使用的文档嵌入向量易于访问，这为我们的密集检索伪相关反馈方法提供了良好的基础。实际上，虽然 ColBERT 中嵌入向量的使用解决了词汇不匹配问题，但我们认为从排名靠前的文档中识别更多相关的嵌入向量可能有助于进一步优化文档排名。特别是，正如我们将展示的，这允许使用一组伪相关文档的代表性嵌入向量来优化查询表示 $\phi$。

## 4 DENSE PSEUDO-RELEVANCE FEEDBACK

## 4 密集伪相关反馈

The aim of a pseudo-relevance feedback approach is typically to generate a refined query representation by analysing the text of the feedback documents. In our proposed ColBERT-PRF approach, we are inspired by conventional PRF approaches such as Bo1 [4] and RM3 [1], which assume that good expansion terms will occur frequently in the feedback set (and hence are somehow representative of the information need underlying the query), but infrequent in the collection as a whole (therefore are sufficiently discriminative). Therefore, we aim to encapsulate these intuitions while operating in the contextualised embedding space ${\mathbb{R}}^{d}$ ,where the exact counting of frequencies is not actually possible. In particular, by operating entirely in the embedding space rather than directly on tokens, we conjecture that we can identify similar embeddings (corresponding to tokens with similar contexts),which can be added to the query representation for improved effectiveness. ${}^{2}$ The bottom part of Table 1 summarises the main notations that we use in describing ColBERT-PRF.

伪相关反馈方法的目标通常是通过分析反馈文档的文本生成一个优化后的查询表示。在我们提出的 ColBERT - PRF 方法中，我们受到了诸如 Bo1 [4] 和 RM3 [1] 等传统伪相关反馈（PRF）方法的启发，这些方法假设好的扩展词项会在反馈集中频繁出现（因此在某种程度上代表了查询背后的信息需求），但在整个文档集合中出现频率较低（因此具有足够的区分度）。因此，我们的目标是在上下文嵌入空间 ${\mathbb{R}}^{d}$ 中操作时体现这些直觉，在这个空间中实际上无法精确统计频率。特别是，通过完全在嵌入空间中操作而不是直接对词元进行操作，我们推测可以识别出相似的嵌入向量（对应于具有相似上下文的词元），可以将其添加到查询表示中以提高检索效果。${}^{2}$ 表 1 的下半部分总结了我们在描述 ColBERT - PRF 时使用的主要符号。

---

<!-- Footnote -->

${}^{1}$ In this way,any notion of similarity from the ANN stage is discarded - the entire set of retrieved documents is reranked; we return to this detail later in Section 8.

${}^{1}$ 通过这种方式，抛弃了近似最近邻（ANN）阶段的任何相似性概念——对整个检索到的文档集合进行重新排序；我们将在第 8 节后面再详细讨论这个细节。

<!-- Footnote -->

---

<!-- Media -->

ALGORITHM 1: The ColBERT E2E algorithm

算法 1：ColBERT 端到端算法

---

Input : A query $Q$

Output: A set $A$ of (docid,score) pairs

CoLBERT E2E(Q):

	${\phi }_{{q}_{1}},\ldots ,{\phi }_{{q}_{n}} \leftarrow  \operatorname{Encode}\left( Q\right)$

	$D \leftarrow  \varnothing$

	for ${\phi }_{{q}_{i}}$ in ${\phi }_{{q}_{1}},\ldots ,{\phi }_{{q}_{n}}$ do

		$D \leftarrow  D \cup  {\mathcal{F}}_{d}\left( {{\phi }_{{q}_{i}},{k}^{\prime }}\right)$

	$A \leftarrow  \varnothing$

	for $d$ in $D$ do

		$s \leftarrow  \mathop{\sum }\limits_{{i = 1}}^{\left| q\right| }\mathop{\max }\limits_{{j = 1,\ldots ,\left| d\right| }}{\phi }_{{q}_{i}}^{T}{\phi }_{{d}_{j}}$

		$A \leftarrow  A \cup  \{ \left( {d,s}\right) \}$

	return $A$

---

<!-- Media -->

In this section, we detail how we identify representative (centroid) embeddings from the feedback documents (Section 4.1), how we ensure that those centroid embeddings are sufficiently discriminative (Section 4.2), and how we apply these discriminative representative centroid embed-dings for (re)ranking (Section 4.3). We conclude with an illustrative example (Section 4.4) and a discussion of the novelty of ColBERT-PRF (Section 4.5).

在本节中，我们详细介绍如何从反馈文档中识别代表性（质心）嵌入向量（第 4.1 节），如何确保这些质心嵌入向量具有足够的区分度（第 4.2 节），以及如何应用这些具有区分度的代表性质心嵌入向量进行（重新）排名（第 4.3 节）。最后，我们给出一个示例（第 4.4 节）并讨论 ColBERT - PRF 的新颖性（第 4.5 节）。

### 4.1 Representative Embeddings in Feedback Documents

### 4.1 反馈文档中的代表性嵌入向量

First,we need to identify representative embeddings $\left\{  {{v}_{1},\ldots ,{v}_{K}}\right\}$ among all embeddings in the feedback documents set. A typical "sparse" PRF approach - such as RM3 - would count the frequency of terms occurring in the feedback set to identify representative ones. However, in a dense embedded setting, the document embeddings are not countable. Instead, we resort to clustering to identify patterns in the embedding space that are representative of embeddings.

首先，我们需要在反馈文档集合的所有嵌入向量中识别代表性嵌入向量 $\left\{  {{v}_{1},\ldots ,{v}_{K}}\right\}$。典型的“稀疏”伪相关反馈方法（如 RM3）会统计反馈集中词项的出现频率以识别代表性词项。然而，在密集嵌入的环境中，文档嵌入向量是不可计数的。相反，我们采用聚类方法来识别嵌入空间中代表嵌入向量的模式。

Specifically,let $\Phi \left( {q,{f}_{b}}\right)$ be the set of all document embeddings from the ${f}_{b}$ top-ranked feedback documents. Then, we apply a clustering approach, e.g., the KMeans clustering algorithm, to $\Phi \left( {q,{f}_{b}}\right)  :$

具体来说，设 $\Phi \left( {q,{f}_{b}}\right)$ 是来自 ${f}_{b}$ 个排名靠前的反馈文档的所有文档嵌入向量的集合。然后，我们对 $\Phi \left( {q,{f}_{b}}\right)  :$ 应用聚类方法，例如 K 均值聚类算法。

$$
\left\{  {{v}_{1},\ldots ,{v}_{K}}\right\}   = \operatorname{Clustering}\left( {K,\Phi \left( {q,{f}_{b}}\right) }\right) . \tag{2}
$$

By applying the clustering algorithm,we obtain $K$ representative centroid embeddings of the feedback documents. The embeddings forming each cluster may or may not correspond to the exact same tokens spread across the feedback documents. In this way, a cluster can represent one or more tokens that appear in similar contexts, rather than a particular exact token. This is a key advantage of ColBERT-PRF. To further demonstrate the choice of clustering technique for ColBERT-PRF, we have compared ColBERT-PRF implemented using KMeans clustering and ColBERT-PRF with traditional query expansion methods, namely Bo1 and RM3 techniques in Appendix A.1. Later, in Section 8, we propose and evaluate other approaches for clustering. Next, we determine how well these centroids discriminate among the documents in the corpus.

通过应用聚类算法，我们得到反馈文档的 $K$ 个代表性质心嵌入向量。形成每个聚类的嵌入向量可能对应也可能不对应于分布在反馈文档中的完全相同的词元。通过这种方式，一个聚类可以代表出现在相似上下文中的一个或多个词元，而不是特定的某个词元。这是 ColBERT - PRF 的一个关键优势。为了进一步证明为 ColBERT - PRF 选择聚类技术的合理性，我们在附录 A.1 中比较了使用 K 均值聚类实现的 ColBERT - PRF 和采用传统查询扩展方法（即 Bo1 和 RM3 技术）的 ColBERT - PRF。稍后，在第 8 节中，我们将提出并评估其他聚类方法。接下来，我们确定这些质心在语料库文档之间的区分能力如何。

---

<!-- Footnote -->

${}^{2}$ In Appendix A.1,we provide experiments that use Bo1 and RM3 to select tokens and their corresponding embeddings that verify this conjecture.

${}^{2}$ 在附录 A.1 中，我们提供了使用 Bo1 和 RM3 选择词元及其相应嵌入向量的实验，这些实验验证了这一推测。

<!-- Footnote -->

---

### 4.2 Identifying Discriminative Embeddings among Representative Embeddings

### 4.2 在代表性嵌入中识别有区分性的嵌入

Many of the $K$ representative embeddings may represent stopwords and therefore are not sufficiently informative when retrieving documents. Typically, identifying informative and discriminative expansion terms from feedback documents would involve examining the collection frequency or the document frequency of the constituent terms [6, 38]. However, there may not be a one-toone relationship between query/centroid embeddings and actual tokens, hence we seek to map each centroid ${v}_{i}$ to a possible token $t$ .

许多$K$代表性嵌入可能代表停用词，因此在检索文档时信息不够丰富。通常，从反馈文档中识别信息丰富且有区分性的扩展词需要检查组成词的集合频率或文档频率[6, 38]。然而，查询/质心嵌入与实际标记之间可能不存在一一对应关系，因此我们试图将每个质心${v}_{i}$映射到一个可能的标记$t$。

We resort to FAISS to achieve this,through the function ${\mathcal{F}}_{t}\left( {{v}_{i},r}\right)  \rightarrow  \left( {t,\ldots }\right)$ that,given the centroid embedding ${v}_{i}$ and $r$ ,returns the list of the $r$ token ids corresponding to the $r$ closest document embeddings to the centroid. ${}^{3}$ From a probabilistic viewpoint,the likelihood $P\left( {t \mid  {v}_{i}}\right)$ of a token $t$ given an embedding ${v}_{i}$ can be obtained as:

我们借助FAISS（快速近似最近邻搜索库）来实现这一点，通过函数${\mathcal{F}}_{t}\left( {{v}_{i},r}\right)  \rightarrow  \left( {t,\ldots }\right)$，给定质心嵌入${v}_{i}$和$r$，该函数返回与质心最接近的$r$个文档嵌入对应的$r$个标记ID列表。${}^{3}$从概率的角度来看，给定嵌入${v}_{i}$时标记$t$的似然性$P\left( {t \mid  {v}_{i}}\right)$可以通过以下方式获得：

$$
P\left( {t \mid  {v}_{i}}\right)  = \frac{1}{r}\mathop{\sum }\limits_{{\tau  \in  {\mathcal{F}}_{t}\left( {{v}_{i},r}\right) }}\mathbb{1}\left\lbrack  {\tau  = t}\right\rbrack   \tag{3}
$$

where $\mathbb{1}\left\lbrack  \right\rbrack$ is the indicator function.

其中$\mathbb{1}\left\lbrack  \right\rbrack$是指示函数。

For simplicity,we choose the most likely token id,i.e., ${t}_{i} = \arg \mathop{\max }\limits_{t}P\left( {t \mid  {v}_{i}}\right)$ . Mapping back to a token id allows us to make use of Inverse Document Frequency (IDF), which can be prerecorded for each token id. The importance ${\sigma }_{i}$ of a centroid embedding ${v}_{i}$ is obtained using a traditional IDF formula: ${}^{4}{\sigma }_{i} = \log \left( \frac{N + 1}{{N}_{i} + 1}\right)$ ,where ${N}_{i}$ is the number of passages containing the token ${t}_{i}$ and $N$ is the total number of passages in the collection. While this approximation of embedding informativeness is obtained by mapping back to tokens, as we shall show, it is very effective. In addition, we will discuss different derivations of a tailored informativeness measure in Section 7, including Inverse Collection Term Frequency and Mean Cosine Similarity methods. Finally, we select the ${f}_{e}$ most informative centroids as expansion embeddings based on the ${\sigma }_{i}$ importance scores as follows:

为了简单起见，我们选择最可能的标记ID，即${t}_{i} = \arg \mathop{\max }\limits_{t}P\left( {t \mid  {v}_{i}}\right)$。映射回标记ID使我们能够利用逆文档频率（IDF，Inverse Document Frequency），该频率可以为每个标记ID预先记录。质心嵌入${v}_{i}$的重要性${\sigma }_{i}$使用传统的IDF公式获得：${}^{4}{\sigma }_{i} = \log \left( \frac{N + 1}{{N}_{i} + 1}\right)$，其中${N}_{i}$是包含标记${t}_{i}$的段落数量，$N$是集合中段落的总数。虽然这种嵌入信息性的近似是通过映射回标记获得的，但正如我们将展示的，它非常有效。此外，我们将在第7节讨论定制信息性度量的不同推导方法，包括逆集合词频和平均余弦相似度方法。最后，我们根据${\sigma }_{i}$重要性得分选择${f}_{e}$个信息最丰富的质心作为扩展嵌入，如下所示：

$$
{F}_{e} = \operatorname{TopScoring}\left( {\left\{  {\left( {{v}_{1},{\sigma }_{1}}\right) ,\ldots ,\left( {{v}_{K},{\sigma }_{K}}\right) }\right\}  ,{f}_{e}}\right)  \tag{4}
$$

where $\operatorname{TopScoring}\left( {A,c}\right)$ returns the $c$ elements of $A$ with the highest importance score.

其中$\operatorname{TopScoring}\left( {A,c}\right)$返回$A$中重要性得分最高的$c$个元素。

### 4.3 Ranking and Reranking with ColBERT-PRF

### 4.3 使用ColBERT - PRF进行排序和重排序

Given the original $\left| q\right|$ query embeddings and the ${f}_{e}$ expansion embeddings,we incorporate the score contributions of the expansion embeddings in Equation (1) as follows:

给定原始$\left| q\right|$查询嵌入和${f}_{e}$扩展嵌入，我们将扩展嵌入的得分贡献纳入方程(1)，如下所示：

$$
s\left( {q,d}\right)  = \mathop{\sum }\limits_{{i = 1}}^{\left| q\right| }\mathop{\max }\limits_{{j = 1,\ldots ,\left| d\right| }}{\phi }_{{q}_{i}}^{T}{\phi }_{{d}_{j}} + \beta \mathop{\sum }\limits_{{\left( {{v}_{i},{\sigma }_{i}}\right)  \in  {F}_{e}}}\mathop{\max }\limits_{{j = 1,\ldots ,\left| d\right| }}{\sigma }_{i}{v}_{i}^{T}{\phi }_{{d}_{j}}, \tag{5}
$$

where $\beta  > 0$ is a parameter weighting the contribution of the expansion embeddings,and the score produced by each expansion embedding is further weighted by the IDF weight of its most likely token, ${\sigma }_{i}$ . Note that Equation (5) can be applied to rerank the documents obtained from the initial query, or as part of a full re-execution of the full dense retrieval operation including the additional ${f}_{e}$ expansion embeddings. In both ranking and reranking,ColBERT-PRF has four parameters: ${f}_{b}$ , the number of feedback documents; $K$ ,the number of clusters; ${f}_{e} \leq  K$ ,the number of expansion embeddings; and $\beta$ ,the importance of the expansion embeddings during scoring. Figure 1 presents the five stages of ColBERT-PRF in its ranking configuration.

其中$\beta  > 0$是一个参数，用于权衡扩展嵌入的贡献，每个扩展嵌入产生的得分进一步由其最可能标记的IDF权重${\sigma }_{i}$加权。请注意，方程(5)可用于对从初始查询中获得的文档进行重排序，或者作为包括额外${f}_{e}$扩展嵌入的全密集检索操作的完整重新执行的一部分。在排序和重排序中，ColBERT - PRF有四个参数：${f}_{b}$，反馈文档的数量；$K$，聚类的数量；${f}_{e} \leq  K$，扩展嵌入的数量；以及$\beta$，扩展嵌入在评分时的重要性。图1展示了ColBERT - PRF在其排序配置中的五个阶段。

---

<!-- Footnote -->

${}^{3}$ This additional mapping can be recorded at indexing time,using the same FAISS index as for dense retrieval,increasing the index size by $3\%$ .

${}^{3}$可以在索引时使用与密集检索相同的FAISS索引记录此额外映射，使索引大小增加$3\%$。

${}^{4}$ We have observed no marked empirical benefits in using other IDF formulations.

${}^{4}$我们没有观察到使用其他IDF公式有明显的实证益处。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: Query Text Q ${\phi }_{{q}_{1}}$ Document embeddings ${k}^{\prime }$ MaxSim reranking ${D}_{{q}_{1}}$ ColBERT-PRF Stage 2 ${D}_{{q}_{2}}$ Stage 3 ${D}_{{q}_{1}}$ MaxSim reranking Return top Stage 5 ranked documents Query ANN search embeddings Stage 1 Expansion embedding ANN search Stage 4 -->

<img src="https://cdn.noedgeai.com/0195b372-2fb6-7dbd-a6e2-deb0bb3cad1c_8.jpg?x=222&y=237&w=1119&h=311&r=0"/>

Fig. 1. Workflow of ColBERT-PRF ranker.

图1. ColBERT - PRF排序器的工作流程。

<!-- Media -->

Furthermore, we provide the pseudo-code of our proposed ColBERT PRF ReRanker in Algorithm 2. The ColBERT-PRF Ranker can be easily obtained by inserting lines 3-4 of Algorithm 1 at line 10 of Algorithm 2 to perform retrieval using both the original query embeddings and the expansion embeddings, and similarly adapting the max-sim scoring in Equation (1) to encapsulate the original query embeddings as well as the expansion embeddings.

此外，我们在算法2中提供了所提出的ColBERT伪相关反馈重排序器（ColBERT PRF ReRanker）的伪代码。通过将算法1的第3 - 4行插入到算法2的第10行，就可以轻松得到ColBERT - PRF排序器，从而使用原始查询嵌入和扩展嵌入进行检索，并类似地调整公式（1）中的最大相似度评分，以同时封装原始查询嵌入和扩展嵌入。

<!-- Media -->

ALGORITHM 2: The ColBERT PRF (reranking) algorithm

算法2：ColBERT伪相关反馈（重排序）算法

---

Input : A query $Q$ ,

		number of feedback documents ${f}_{b}$ ,

		number of representative embeddings $K$ ,

		number of expansion embeddings ${f}_{e}$

Output: A set $B$ of (docid,score) pairs

ColBERT PRF(Q):

	$A \leftarrow$ ColBERT E2E(Q)

	$\Phi \left( {Q,{f}_{b}}\right)  \leftarrow$ set of all document embeddings from

				the ${f}_{b}$ top-scored documents in $A$

	$V \leftarrow  \varnothing$

	${v}_{1},\ldots ,{v}_{K} = \operatorname{KMeans}\left( {K,\Phi \left( {Q,{f}_{b}}\right) }\right)$

	for ${v}_{i}$ in ${v}_{1},\ldots ,{v}_{K}$ do

		${t}_{i} \leftarrow  {\operatorname{argmax}}_{t}\frac{1}{r}\mathop{\sum }\limits_{{\tau  \in  {\mathcal{F}}_{t}\left( {{v}_{i},r}\right) }}\mathbb{1}\left\lbrack  {\tau  = t}\right\rbrack$

		${\sigma }_{i} \leftarrow  \log \left( \frac{N + 1}{{N}_{i} + 1}\right)$

		$V \leftarrow  V \cup  \left\{  \left( {{v}_{i},{\sigma }_{i}}\right) \right\}$

	${F}_{e} \leftarrow  \operatorname{TopScoring}\left( {V,{f}_{e}}\right)$

	$B \leftarrow  \varnothing$

	for(d,s)in $A$ do

		$s \leftarrow  s + \beta \mathop{\sum }\limits_{{\left( {{v}_{i},{\sigma }_{i}}\right)  \in  {F}_{e}}}\mathop{\max }\limits_{{j = 1,\ldots ,\left| d\right| }}{\sigma }_{i}{v}_{i}^{T}{\phi }_{{d}_{j}}$

		$B \leftarrow  B \cup  \{ \left( {d,s}\right) \}$

	return $B$

---

<!-- Media -->

### 4.4 Illustrative Example

### 4.4 示例说明

We now illustrate the effect of ColBERT-PRF upon one query from the TREC 2019 Deep Learning track, 'do goldfish grow'. We use PCA to quantize the 128-dimension embeddings into two dimensions purely to allow visualisation. Firstly, Figure 2(a) shows the embeddings of the original query (black ellipses); the red [MASK] tokens are also visible, clustered around the original query terms (##fish, gold, grow). Meanwhile, document embeddings extracted from 10 feedback documents are shown as light blue ellipses in Figure 2(a). There appear to be visible clusters of document embeddings near the query embeddings, but also other document embeddings exhibit some clustering. The mass of embeddings near the origin is not distinguishable in PCA. Figure 2(b) demonstrates the application of KMeans clustering upon the document embeddings; we map back to the original tokens by virtue of Equation (3). In Figure 2(b), the point size is indicative of the IDF of the corresponding token. We can see that the cluster centroids with high IDF correspond to the original query tokens ('gold', '##fish', 'grow'), as well as the related terms ('tank', 'size'). In contrast, a centroid with low IDF is 'the'. This illustrates the utility of our proposed ColBERT-PRF approach in using KMeans to identify representative clusters of embeddings, as well as using IDF to differentiate useful clusters.

现在，我们以TREC 2019深度学习赛道中的一个查询“金鱼会长大吗”为例，来说明ColBERT - PRF的效果。我们使用主成分分析（PCA）将128维的嵌入量化为二维，纯粹是为了便于可视化。首先，图2（a）展示了原始查询的嵌入（黑色椭圆）；红色的[MASK]标记也清晰可见，它们聚集在原始查询词（##fish、gold、grow）周围。同时，从10篇反馈文档中提取的文档嵌入在图2（a）中显示为浅蓝色椭圆。在查询嵌入附近似乎有明显的文档嵌入簇，但其他文档嵌入也呈现出一定的聚类现象。在主成分分析中，原点附近的大量嵌入无法区分。图2（b）展示了对文档嵌入应用K均值聚类的情况；我们通过公式（3）将其映射回原始标记。在图2（b）中，点的大小表示相应标记的逆文档频率（IDF）。我们可以看到，逆文档频率较高的聚类质心对应于原始查询标记（“gold”、“##fish”、“grow”）以及相关术语（“tank”、“size”）。相比之下，逆文档频率较低的质心是“the”。这说明了我们提出的ColBERT - PRF方法在使用K均值识别代表性嵌入簇以及使用逆文档频率区分有用簇方面的实用性。

<!-- Media -->

<!-- figureText: 0.6 Doc. embs. ###fish 0.6 0.4 ###fish 0.2 0.0 the tanktank (war) -0.2 -0.4 gold -0.50 -0.25 0.00 0.25 0.50 0.75 (b) Cluster centroids, $K = {24}$ . Query embs. 0.4 Mask embs. 0.2 0.0 [CLS] O[SEP] -0.2 -0.4 gold -0.50 -0.25 0.00 0.25 0.50 0.75 (a) Query & doc. embeddings. -->

<img src="https://cdn.noedgeai.com/0195b372-2fb6-7dbd-a6e2-deb0bb3cad1c_9.jpg?x=231&y=235&w=1093&h=435&r=0"/>

Fig. 2. Example showing how ColBERT-PRF operates for the query 'do goldfish grow' in a 2D PCA space. In Figure 2(b), the point size is representative of IDF; five high IDF and one low IDF centroids are shown. For contrast, $\times$ ‘tank (war)’ denotes the embedding of ‘tank’ occurring in a non-fish context.

图2. 展示了在二维主成分分析空间中，ColBERT - PRF如何处理查询“金鱼会长大吗”的示例。在图2（b）中，点的大小代表逆文档频率；展示了五个高逆文档频率和一个低逆文档频率的质心。为作对比，$\times$ ‘tank (war)’ 表示“tank”在非鱼类语境中的嵌入。

<!-- Media -->

Furthermore,Figure 2(b) also includes,marked by an $\times$ and denoted ’tank (war)’,the embedding for the word 'tank' when placed in the passage "While the soldiers advanced, the tank bombarded the troops with artillery". It can be seen that, even in the highly compressed PCA space, the 'tank' centroid embedding is distinct from the embedding of 'tank (war)'. This shows the utility of ColBERT-PRF when operating in the embedding space, as the PRF process for the query 'do goldfish grow' will not retrieve documents containing 'tank (war)', but will focus on a fish-related context, thereby dealing with the polysemous nature of a word such as 'tank'. To the best of our knowledge, this is a unique feature of ColBERT-PRF among PRF approaches.

此外，图2（b）还包括一个标记为$\times$且表示为“tank (war)”的内容，即单词“tank”在段落“当士兵们推进时，坦克用火炮轰击部队”中的嵌入。可以看出，即使在高度压缩的主成分分析空间中，“tank”的质心嵌入也与“tank (war)”的嵌入不同。这显示了ColBERT - PRF在嵌入空间中操作时的实用性，因为针对查询“金鱼会长大吗”的伪相关反馈过程不会检索包含“tank (war)”的文档，而是会聚焦于与鱼类相关的语境，从而处理像“tank”这样的多义词的问题。据我们所知，这是ColBERT - PRF在伪相关反馈方法中的一个独特特征。

### 4.5 Discussion

### 4.5 讨论

To the best of our knowledge ColBERT-PRF is the first investigation of pseudo-relevance feedback for multiple representation dense retrieval. Existing works on neural pseudo-relevance feedback, such as Neural PRF [20] and BERT-QE [51] only function as rerankers. Other approaches such as DeepCT [10] and doc2query [32, 33] use neural models to augment documents before indexing using a traditional inverted index. CEQE [29] generates words to expand the initial query, which is then executed on the inverted index. However, returning the BERT embeddings back to textual word forms can result in polysemous words negatively affecting retrieval. In contrast, ColBERT-PRF operates entirely on an existing dense index representation (without augmenting documents), and can function for both ranking as well as reranking. By retrieving using feedback embeddings directly, ColBERT-PRF addresses polysemous words (such as 'tank', illustrated above). It is also of note that it also requires no additional neural network training beyond that of ColBERT. Indeed, while ANCE-PRF requires further training of the refined query encoder, ColBERT-PRF does not require any further retraining. Furthermore, compared to the single embedding of ANCE-PRF, ColBERT-PRF is also more explainable in nature, as the expansion embeddings can be mapped to tokens (as shown in Figure 2), and their contribution to document scoring can be examined, as we will show in Section 5.3.4.

据我们所知，ColBERT - PRF是首次针对多表示密集检索进行伪相关反馈的研究。现有的关于神经伪相关反馈的工作，如神经伪相关反馈（Neural PRF）[20]和BERT查询扩展（BERT - QE）[51]仅作为重排序器发挥作用。其他方法，如深度上下文词项加权（DeepCT）[10]和文档到查询（doc2query）[32, 33]，使用神经模型在使用传统倒排索引进行索引之前对文档进行增强。上下文嵌入查询扩展（CEQE）[29]生成单词来扩展初始查询，然后在倒排索引上执行该查询。然而，将BERT嵌入转换回文本单词形式可能会导致多义词对检索产生负面影响。相比之下，ColBERT - PRF完全在现有的密集索引表示上操作（不增强文档），并且既可以用于排序也可以用于重排序。通过直接使用反馈嵌入进行检索，ColBERT - PRF解决了多义词（如上述示例中的“tank”）的问题。值得注意的是，除了ColBERT的训练之外，它不需要额外的神经网络训练。实际上，虽然自适应负对比增强伪相关反馈（ANCE - PRF）需要对精炼查询编码器进行进一步训练，但ColBERT - PRF不需要任何进一步的再训练。此外，与ANCE - PRF的单一嵌入相比，ColBERT - PRF本质上更具可解释性，因为扩展嵌入可以映射到标记（如图2所示），并且我们将在5.3.4节中展示，可以检查它们对文档评分的贡献。

In the following, we first show the retrieval effectiveness of ColBERT-PRF for passage ranking and document ranking tasks in Sections 5 and 6, respectively. In particular, in Section 5, we examine the characteristics of ColBERT-PRF, including how ColBERT-PRF addresses polysemous words, how ColBERT-PRF demonstrates compared with the traditional query expansion techniques and how to quantify the extent of the semantic matching ability of ColBERT-PRF. Next, we discuss three variants of ColBERT-PRF with different discriminative power measure methods in Section 7, and we address the effectiveness and efficiency trade-off of ColBERT-PRF in Section 8.

接下来，我们将分别在第5节和第6节中展示ColBERT-PRF（基于伪相关反馈的ColBERT模型）在段落排序和文档排序任务中的检索效果。具体而言，在第5节中，我们将研究ColBERT-PRF的特性，包括ColBERT-PRF如何处理多义词、与传统查询扩展技术相比ColBERT-PRF的表现如何，以及如何量化ColBERT-PRF的语义匹配能力的程度。接下来，我们将在第7节中讨论具有不同判别力度量方法的ColBERT-PRF的三种变体，并在第8节中探讨ColBERT-PRF在有效性和效率之间的权衡。

## 5 PASSAGE RANKING EFFECTIVENESS OF COLBERT-PRF

## 5 ColBERT-PRF的段落排序效果

In this section, we analyse the performance of ColBERT-PRF for passage ranking. In particular, we evaluated the performance of ColBERT-PRF on TREC 2019 and TREC 2020 query sets. Section 5.1 describes the research question addressed by our passage ranking experiments. The experimental setup and the obtained results are detailed in Sections 5.2 and 5.3, respectively.

在本节中，我们分析ColBERT-PRF在段落排序方面的性能。具体而言，我们评估了ColBERT-PRF在TREC 2019和TREC 2020查询集上的性能。第5.1节描述了我们的段落排序实验所解决的研究问题。实验设置和所得结果分别在第5.2节和第5.3节中详细介绍。

### 5.1 Research Questions

### 5.1 研究问题

Our passage ranking experiments address the four following research questions:

我们的段落排序实验解决了以下四个研究问题：

- RQ1: Can a multiple representation dense retrieval approach be enhanced by pseudo-relevance feedback, i.e., can ColBERT-PRF outperform ColBERT dense retrieval?

- RQ1：多表示密集检索方法能否通过伪相关反馈得到增强，即ColBERT-PRF能否优于ColBERT密集检索？

- RQ2: How does ColBERT-PRF compare to other existing baselines and state-of-the-art approaches, namely:

- RQ2：ColBERT-PRF与其他现有的基线方法和最先进的方法相比表现如何，具体如下：

(a) lexical (sparse) baselines, including using PRF,

(a) 词法（稀疏）基线方法，包括使用伪相关反馈（PRF）的方法

(b) neural augmentation approaches, namely DeepCT and docT5query,

(b) 神经增强方法，即DeepCT和docT5query

(c) BERT-QE Reranking models, and

(c) BERT-QE重排序模型

(d) embedding based query expansion models, namely the three variants of CEQE models: CEQE-Max, CEQE-Centroid, and CEQE-Mul?

(d) 基于嵌入的查询扩展模型，即CEQE模型的三种变体：CEQE-Max、CEQE-Centroid和CEQE-Mul

- RQ3: What is the impact of the parameters of ColBERT-PRF, namely the number of clusters and expansion embeddings,the number of feedback passages and the $\beta$ parameter controlling the influence of the expansion embeddings?

- RQ3：ColBERT-PRF的参数，即聚类数量和扩展嵌入数量、反馈段落数量以及控制扩展嵌入影响的$\beta$参数，会产生什么影响？

- RQ4: To what extent does ColBERT-PRF perform semantic matching?

- RQ4：ColBERT-PRF在多大程度上进行语义匹配？

### 5.2 Experimental Setup

### 5.2 实验设置

5.2.1 Dataset & Measures. Experiments are conducted on the MSMARCO passage corpus, using the TREC 2019 Deep Learning track topics (43 topics with an average of 215.35 relevance judgements per query) and the TREC 2020 Deep Learning track topics (54 topics with an average of 210.85 relevance judgements per query) from TRECDL passage ranking task. We omit topics from the MSMARCO Dev set,which have only sparse judgements, $\sim  {1.1}$ per query. Indeed,pseudo-relevance feedback approaches are known to be not effective on test collections with few judged passages [3].

5.2.1 数据集与度量指标。实验在MSMARCO段落语料库上进行，使用来自TRECDL段落排序任务的TREC 2019深度学习赛道主题（43个主题，每个查询平均有215.35个相关性判断）和TREC 2020深度学习赛道主题（54个主题，每个查询平均有210.85个相关性判断）。我们省略了MSMARCO开发集中每个查询只有稀疏判断（$\sim  {1.1}$）的主题。实际上，已知伪相关反馈方法在判断段落较少的测试集上效果不佳 [3]。

We report the commonly used metrics for the TREC 2019 and TREC 2020 query sets following the corresponding track overview papers [7, 8]: we report mean reciprocal rank (MRR) and normalised discounted cumulative gain (NDCG) calculated at rank 10 , as well as Recall and Mean Average Precision (MAP) at rank 1000 [8]. For the MRR, MAP and Recall metrics, we treat passages with label grade 1 as non-relevant, following [7, 8]. In addition, we also report the Mean Response Time (MRT) for each retrieval system. For significance testing, we use the paired t-test $\left( {p < {0.05}}\right)$ and apply the Holm-Bonferroni multiple testing correction.

我们按照相应的赛道概述论文 [7, 8] 报告了TREC 2019和TREC 2020查询集常用的指标：我们报告了在排名第10时计算的平均倒数排名（MRR）和归一化折损累积增益（NDCG），以及在排名第1000时的召回率和平均准确率均值（MAP） [8]。对于MRR、MAP和召回率指标，我们按照 [7, 8] 的做法，将标签等级为1的段落视为不相关。此外，我们还报告了每个检索系统的平均响应时间（MRT）。对于显著性检验，我们使用配对t检验 $\left( {p < {0.05}}\right)$ 并应用霍尔姆 - 邦费罗尼多重检验校正。

5.2.2 Implementation and Settings. We conduct experiments using PyTerrier [25] and, in particular using our PyTerrier_ColBERT plugin, ${}^{5}$ which includes ColBERT-PRF as well as our adaptations of the ColBERT source code. ColBERT and ColBERT-PRF are expressed as PyTerrier transformer operations - the source code of the ColBERF-PRF ranker and re-ranker pipelines is shown in the Appendix A.2.

5.2.2 实现与设置。我们使用PyTerrier [25]进行实验，特别是使用我们的PyTerrier_ColBERT插件${}^{5}$，该插件包含ColBERT-PRF以及我们对ColBERT源代码的改编。ColBERT和ColBERT-PRF被表示为PyTerrier转换器操作——ColBERF-PRF排序器和重排序器管道的源代码见附录A.2。

In terms of the ColBERT configuration, we train ColBERT upon the MSMARCO passage ranking triples file for 44,000 batches,applying the parameters specified by Khattab & Zaharia in [17]: Maximum document length is set to 180 tokens and queries are encoded into 32 query embeddings (including [MASK] tokens); We encode all passages to a FAISS index that has been trained using 5% of all embeddings; At retrieval time,FAISS retrieves ${k}^{\prime } = {1000}$ passage embeddings for every query embedding. ColBERT-PRF is implemented using the KMeans implementation [5] of sci-kit learn (sklearn). For query expansion settings, we follow the default settings of Terrier [34], which is 10 expansion terms obtained from three feedback passages; we follow the same default setting for ColBERT-PRF,additionally using representative values,namely $K = {24}$ clusters, ${}^{6}$ and $\beta  = \{ {0.5},1\}$ for the weight of the expansion embeddings. We later show the impact of these parameters when we address RQ3.

在ColBERT配置方面，我们在MSMARCO段落排名三元组文件上对ColBERT进行44000批次的训练，应用Khattab和Zaharia在[17]中指定的参数：最大文档长度设置为180个词元，查询被编码为32个查询嵌入（包括[MASK]词元）；我们将所有段落编码到一个FAISS索引中，该索引使用所有嵌入的5%进行训练；在检索时，FAISS为每个查询嵌入检索${k}^{\prime } = {1000}$个段落嵌入。ColBERT-PRF使用scikit-learn（sklearn）的KMeans实现[5]。对于查询扩展设置，我们遵循Terrier [34]的默认设置，即从三个反馈段落中获取10个扩展词；我们对ColBERT-PRF采用相同的默认设置，另外使用代表性值，即$K = {24}$个聚类，以及扩展嵌入权重的${}^{6}$和$\beta  = \{ {0.5},1\}$。我们稍后在解决研究问题3（RQ3）时展示这些参数的影响。

5.2.3 Baselines. To test the effectiveness of our proposed dense PRF approach, we compare with five families of baseline models, for which we vary the use of a BERT-based reranker (namely BERT or ColBERT). For the BERT reranker, we use OpenNIR [24] and capreolus/ bert-base-msmarco fine-tuned model from [21]. For the ColBERT reranker, unless otherwise noted, we use the existing pre-indexed ColBERT representation of passages for efficient reranking. The five families are: Lexical Retrieval Approaches: These are traditional retrieval models using a sparse inverted index, with and without BERT and ColBERT rerankers, namely: (i) BM25 (ii) BM25+BERT (iii) BM25+ColBERT, (iv) BM25+RM3, (v) BM25+RM3+BERT, and (vi) BM25+RM3+ColBERT.

5.2.3 基线模型。为了测试我们提出的密集伪相关反馈（PRF）方法的有效性，我们与五类基线模型进行比较，对于这些模型，我们会改变基于BERT的重排序器（即BERT或ColBERT）的使用情况。对于BERT重排序器，我们使用OpenNIR [24]和来自[21]的capreolus/bert-base-msmarco微调模型。对于ColBERT重排序器，除非另有说明，我们使用现有的预索引段落的ColBERT表示进行高效重排序。这五类模型是：词法检索方法：这些是使用稀疏倒排索引的传统检索模型，有或没有BERT和ColBERT重排序器，即：（i）BM25（ii）BM25 + BERT（iii）BM25 + ColBERT（iv）BM25 + RM3（v）BM25 + RM3 + BERT（vi）BM25 + RM3 + ColBERT。

Neural Augmentation Approaches: These use neural components to augment the (sparse) inverted index: (i) BM25+DeepCT and (ii) BM25+docT5query, both without and with BERT and ColBERT rerankers. For BM25+docT5query+ColBERT, the ColBERT reranker is applied on expanded passage texts encoded at querying time, rather than the indexed ColBERT representation. The response time for BM25+docT5query+ColBERT reflects this difference.

神经增强方法：这些方法使用神经组件来增强（稀疏）倒排索引：（i）BM25 + DeepCT和（ii）BM25 + docT5query，都有或没有BERT和ColBERT重排序器。对于BM25 + docT5query + ColBERT，ColBERT重排序器应用于查询时编码的扩展段落文本，而不是索引的ColBERT表示。BM25 + docT5query + ColBERT的响应时间反映了这种差异。

Dense Retrieval Models: This family consists of the dense retrieval approaches: (i) ANCE: The ANCE [46] model is a single representation dense retrieval model. We use the trained models provided by the authors trained on MSMARCO training data. (ii) ANCE-PRF: The ANCE-PRF [48] is a PRF variant of ANCE model - we use the results released by the authors. (iii) ColBERT E2E: ColBERT end-to-end (E2E) [17] is the dense retrieval version of ColBERT, as defined in Section 3. BERT-QE Models: We apply BERT-QE [51] on top of a strong sparse baseline and our dense retrieval baseline, ColBERT E2E, i.e., (i) BM25+RM3+ColBERT+BERT-QE and (ii) ColBERT E2E+BERT-QE; Where possible, we use the ColBERT index for scoring passages; for identifying the top scoring chunks within passages, we use ColBERT in a slower "text" mode, i.e., without using the index. For the BERT-QE parameters, we follow the settings in [51], in particular using the recommended settings of $\alpha  = {0.4}$ and $\beta  = {0.9}$ ,which are also the most effective on MSMARCO. Indeed,to the best our knowledge, this is the first application of BERT-QE upon dense retrieval, the first application of BERT-QE on MSMARCO and the first application using ColBERT. We did attempt to apply BERT-QE using the BERT re-ranker, but we found it to be ineffective on MSMARCO, and exhibiting a response time exceeding 30 seconds per query, hence we omit it from our experiments.

密集检索模型：此类模型包含密集检索方法：（i）ANCE：ANCE [46] 模型是一种单表示密集检索模型。我们使用作者在MS MARCO训练数据上训练得到的模型。（ii）ANCE - PRF：ANCE - PRF [48] 是ANCE模型的一种伪相关反馈（PRF）变体——我们使用作者发布的结果。（iii）ColBERT端到端（E2E）：ColBERT端到端（E2E）[17] 是ColBERT的密集检索版本，如第3节所定义。BERT查询扩展（QE）模型：我们在强大的稀疏基线模型和我们的密集检索基线模型ColBERT E2E之上应用BERT - QE [51]，即（i）BM25 + RM3 + ColBERT + BERT - QE和（ii）ColBERT E2E + BERT - QE；在可能的情况下，我们使用ColBERT索引对段落进行评分；为了识别段落中得分最高的块，我们以较慢的“文本”模式使用ColBERT，即不使用索引。对于BERT - QE参数，我们遵循文献[51]中的设置，特别是使用 $\alpha  = {0.4}$ 和 $\beta  = {0.9}$ 推荐的设置，这些设置在MS MARCO上也是最有效的。事实上，据我们所知，这是BERT - QE在密集检索上的首次应用，是BERT - QE在MS MARCO上的首次应用，也是使用ColBERT的首次应用。我们确实尝试过使用BERT重排器应用BERT - QE，但我们发现它在MS MARCO上效果不佳，并且每个查询的响应时间超过30秒，因此我们在实验中省略了它。

---

<!-- Footnote -->

${}^{5}$ https://github.com/terrierteam/pyterrier_colbert.

${}^{5}$ https://github.com/terrierteam/pyterrier_colbert.

${}^{6}$ Indeed, $K = {24}$ gave reasonable looking clusters in our initial investigations,and,as we shall see in Section 6.3,is an effective setting for the TREC 2019 query set.

${}^{6}$ 实际上，在我们的初步研究中，$K = {24}$ 给出了看起来合理的聚类，并且，正如我们将在第6.3节中看到的，它是TREC 2019查询集的有效设置。

<!-- Footnote -->

---

CEQE Models: This family consists of three CEQE variants [29], i.e., CEQE-Max, CEQE-Centroid, and CEQE-Mul. We apply each CEQE query expansion variant on top of the documents retrieved by BM25. Compared with the original CEQE, we apply the pipeline BM25 + RM3 + BM25 rather than the Dirichlet LM + RM3 + BM25 pipeline for generating the expansion terms.

上下文嵌入查询扩展（CEQE）模型：此类模型包含三种CEQE变体 [29]，即CEQE - Max、CEQE - Centroid和CEQE - Mul。我们将每种CEQE查询扩展变体应用于BM25检索到的文档之上。与原始的CEQE相比，我们使用BM25 + RM3 + BM25管道而不是狄利克雷语言模型（Dirichlet LM）+ RM3 + BM25管道来生成扩展词项。

For reproducibility, ColBERT-PRF and the baselines results are available in our virtual appendix. ${}^{7}$

为了可重复性，ColBERT - PRF和基线结果可在我们的虚拟附录中获取。${}^{7}$

### 5.3 Passage Ranking Results

### 5.3 段落排序结果

5.3.1 Results for RQ1 - Overall Effectiveness of ColBERT-PRF. In this section, we examine the effectiveness of a pseudo-relevance feedback technique for the ColBERT dense retrieval model on passage ranking task. On analysing Table 2, we first note that the ColBERT dense retrieval approach outperforms the single representation based dense retrieval models, i.e., ANCE and its PRF variant ANCE-PRF for all metrics on both test query sets, probably because the single representation used in ANCE provides limited information for matching queries and documents [23]. In particular, compared with ANCE-PRF, ColBERT-PRF shows markedly improvement on all metrics for both query sets and shows significant improvement in terms of MAP on TREC 2019 and NDCG@10 on TREC 2020. This indicates that the PRF mechanism that explicitly expands query with expansion embeddings to refine the query representation is superior to implicitly learning from PRF information to form a better query representation.

5.3.1 研究问题1的结果——ColBERT - PRF的整体有效性。在本节中，我们研究了伪相关反馈技术在ColBERT密集检索模型的段落排序任务中的有效性。通过分析表2，我们首先注意到，在两个测试查询集的所有指标上，ColBERT密集检索方法优于基于单表示的密集检索模型，即ANCE及其PRF变体ANCE - PRF，这可能是因为ANCE中使用的单表示为匹配查询和文档提供的信息有限 [23]。特别是，与ANCE - PRF相比，ColBERT - PRF在两个查询集的所有指标上都有显著改进，并且在TREC 2019的平均准确率均值（MAP）和TREC 2020的归一化折损累积增益（NDCG@10）方面有显著提升。这表明，通过扩展嵌入显式扩展查询以优化查询表示的PRF机制，优于从PRF信息中隐式学习以形成更好查询表示的机制。

Based on this, we then compare the performances of our proposed ColBERT-PRF models, instantiated as ColBERT-PRF Ranker & ColBERT-PRF ReRanker, with the more effective ColBERT E2E model. We find that both the Ranker and ReRanker models outperform ColBERT E2E on all the metrics for both used query sets. Typically, on the TREC 2019 test queries, both the Ranker and ReRanker models exhibit significant improvements in terms of MAP over the ColBERT E2E model. In particular,we observe a ${26}\%$ increase in MAP on TREC 2019 ${}^{8}$ and ${10}\%$ for TREC 2020 over ColBERT E2E for the ColBERT-PRF Ranker. In addition, both ColBERT-PRF Ranker and ReRanker exhibit significant improvements over ColBERT E2E in terms of NDCG@10 on TREC 2019 queries.

基于此，我们随后将我们提出的ColBERT - PRF模型（具体为ColBERT - PRF排序器和ColBERT - PRF重排器）的性能与更有效的ColBERT E2E模型进行比较。我们发现，在两个使用的查询集的所有指标上，排序器和重排器模型都优于ColBERT E2E。通常，在TREC 2019测试查询上，排序器和重排器模型在MAP方面相对于ColBERT E2E模型都有显著改进。特别是，我们观察到ColBERT - PRF排序器在TREC 2019上的MAP相对于ColBERT E2E有 ${26}\%$ 的提升 ${}^{8}$，在TREC 2020上有 ${10}\%$ 的提升。此外，在TREC 2019查询的NDCG@10方面，ColBERT - PRF排序器和重排器相对于ColBERT E2E都有显著改进。

The high effectiveness of ColBERT-PRF anker (which is indeed higher than ColBERT-PRF ReRanker) can be explained in that the expanded query obtained using the PRF process introduces more relevant passages, thus it increases recall after re-executing the query on the dense index. As can be seen from Table 2, ColBERT-PRF Ranker exhibits significant improvements over both ANCE and ColBERT E2E models on Recall. On the other hand, the effectiveness of ColBERT-PRF ReRanker also suggests that the expanded query provides a better query representation, which can which can better rank documents in the existing candidate set. Overall, in response to RQ1, we conclude that our proposed ColBERT-PRF model is effective compared to the ColBERT E2E dense retrieval model.

ColBERT - PRF锚点（ColBERT - PRF anker，其效果确实高于ColBERT - PRF重排器）的高效性可以解释为，使用PRF过程获得的扩展查询引入了更多相关段落，因此在密集索引上重新执行查询后提高了召回率。从表2可以看出，ColBERT - PRF排序器在召回率方面比ANCE和ColBERT端到端（E2E）模型都有显著改进。另一方面，ColBERT - PRF重排器的有效性也表明，扩展查询提供了更好的查询表示，能够在现有候选集中更好地对文档进行排序。总体而言，针对研究问题1（RQ1），我们得出结论：与ColBERT端到端密集检索模型相比，我们提出的ColBERT - PRF模型是有效的。

5.3.2 Results for RQ2 - Comparison to Baselines. Next, to address RQ2(a)-(c), we analyse the performances of the ColBERT-PRF Ranker and ColBERT-PRF ReRanker approaches in comparison to different groups of baselines, namely sparse (lexical) retrieval approaches, neural augmented baselines, and BERT-QE.

5.3.2 研究问题2（RQ2）的结果 - 与基线模型的比较。接下来，为了解决研究问题2（a） - （c），我们分析了ColBERT - PRF排序器和ColBERT - PRF重排器方法与不同组基线模型的性能对比，即稀疏（词法）检索方法、神经增强基线模型和BERT - QE。

---

<!-- Footnote -->

${}^{7}$ https://github.com/Xiao0728/ColBERT-PRF-VirtualAppendix.

${}^{7}$ https://github.com/Xiao0728/ColBERT-PRF-VirtualAppendix.

${}^{8}$ Indeed,this is $8\%$ higher than the highest MAP among all TREC 2019 participants [8].

${}^{8}$ 实际上，这$8\%$高于2019年所有TREC参与者中的最高平均准确率均值（MAP）[8]。

<!-- Footnote -->

---

<!-- Media -->

Table 2. Comparison with Baselines

表2. 与基线模型的比较

<table><tr><td rowspan="2"/><td colspan="5">TREC 2019 (43 queries)</td><td colspan="5">TREC 2020 (54 queries)</td></tr><tr><td>MAP</td><td>NDCG@10</td><td>MRR@10</td><td>Recall</td><td>MRT</td><td>MAP</td><td>NDCG@10</td><td>MRR@10</td><td>Recall</td><td>MRT</td></tr><tr><td colspan="11">Lexical Retrieval Approaches</td></tr><tr><td>BM25 (a)</td><td>0.2864</td><td>0.4795</td><td>0.6416</td><td>0.7553</td><td>133</td><td>0.2930</td><td>0.4936</td><td>0.5912</td><td>0.8103</td><td>129</td></tr><tr><td>BM25+BERT (b)</td><td>0.4441</td><td>0.6855</td><td>0.8295</td><td>0.7553</td><td>3589</td><td>0.4699</td><td>0.6716</td><td>0.8069</td><td>0.8103</td><td>3554</td></tr><tr><td>BM25+ColBERT (c)</td><td>0.4582</td><td>0.6950</td><td>0.8580</td><td>0.7553</td><td>202</td><td>0.4752</td><td>0.6931</td><td>0.8546</td><td>0.8103</td><td>203</td></tr><tr><td>BM25+RM3 (d)</td><td>0.3108</td><td>0.5156</td><td>0.6093</td><td>0.7756</td><td>201</td><td>0.3203</td><td>0.5043</td><td>0.5912</td><td>0.8423</td><td>248</td></tr><tr><td>BM25+RM3+BERT (e)</td><td>0.4531</td><td>0.6862</td><td>0.8275</td><td>0.7756</td><td>4035</td><td>0.4739</td><td>0.6704</td><td>0.8079</td><td>0.8423</td><td>4003</td></tr><tr><td>BM25+RM3+ColBERT (f)</td><td>0.4709</td><td>0.7055</td><td>0.8651</td><td>0.7756</td><td>320</td><td>0.4800</td><td>0.6877</td><td>0.8560</td><td>0.8423</td><td>228</td></tr><tr><td colspan="11">Neural Augmentation Approaches</td></tr><tr><td>BM25+DeepCT (g)</td><td>0.3169</td><td>0.5599</td><td>0.7155</td><td>0.7321</td><td>54</td><td>0.3570</td><td>0.5603</td><td>0.7090</td><td>0.8008</td><td>64</td></tr><tr><td>BM25+DeepCT+BERT (h)</td><td>0.4308</td><td>0.7011</td><td>0.8483</td><td>0.7321</td><td>3737</td><td>0.4671</td><td>0.6852</td><td>0.8068</td><td>0.8008</td><td>3719</td></tr><tr><td>BM25+DeepCT+ColBERT (i)</td><td>0.4416</td><td>0.7004</td><td>0.8541</td><td>0.7321</td><td>129</td><td>0.4757</td><td>0.7071</td><td>0.8549</td><td>0.8008</td><td>141</td></tr><tr><td>BM25+docT5query (j)</td><td>0.4044</td><td>0.6308</td><td>0.7614</td><td>0.8263</td><td>282</td><td>0.4082</td><td>0.6228</td><td>0.7434</td><td>0.8456</td><td>295</td></tr><tr><td>BM25+docT5query+BERT (k)</td><td>0.4802</td><td>0.7123</td><td>0.8483</td><td>0.8263</td><td>8025</td><td>0.4714</td><td>0.6810</td><td>0.8160</td><td>0.8456</td><td>3888</td></tr><tr><td>BM25+docT5query+ColBERT (l)</td><td>0.5009</td><td>0.7136</td><td>0.8367</td><td>0.8263</td><td>2362</td><td>0.4733</td><td>0.6934</td><td>0.8021</td><td>0.8456</td><td>2381</td></tr><tr><td colspan="11">Dense Retrieval Models</td></tr><tr><td>ANCE (m)</td><td>0.3715</td><td>0.6537</td><td>0.8590</td><td>0.7571</td><td>199</td><td>0.4070</td><td>0.6447</td><td>0.7898</td><td>0.7737</td><td>179</td></tr><tr><td>ANCE-PRF (n)</td><td>0.4253</td><td>0.6807</td><td>0.8492</td><td>0.7912</td><td>-</td><td>0.4452</td><td>0.6948</td><td>0.8371</td><td>0.8148</td><td>-</td></tr><tr><td>ColBERT E2E (o)</td><td>0.4318</td><td>0.6934</td><td>0.8529</td><td>0.7892</td><td>581</td><td>0.4654</td><td>0.6871</td><td>0.8525</td><td>0.8245</td><td>600</td></tr><tr><td colspan="11">BERT-QE Reranking Models</td></tr><tr><td>BM25 + RM3 + ColBERT + BERT-OE (p)</td><td>0.4832</td><td>0.7179</td><td>0.8754</td><td>0.7756</td><td>1130</td><td>0.4842</td><td>0.6909</td><td>0.8315</td><td>0.8423</td><td>1595</td></tr><tr><td>ColBERT E2E + BERT-QE (q)</td><td>0.4423</td><td>0.7013</td><td>0.8683</td><td>0.7892</td><td>1261</td><td>0.4749</td><td>0.6911</td><td>0.8315</td><td>0.8245</td><td>1328</td></tr><tr><td colspan="11">Embedding-based Query Expansion Models</td></tr><tr><td>BM25 + CEQE-Max (r)</td><td>0.3453</td><td>0.5382</td><td>0.6605</td><td>0.8277</td><td>15656</td><td>0.3380</td><td>0.5094</td><td>0.6132</td><td>0.8561</td><td>16103</td></tr><tr><td>BM25 + CEOE-Centroid (s)</td><td>0.3425</td><td>0.5345</td><td>0.6595</td><td>0.8234</td><td>14230</td><td>0.3302</td><td>0.5099</td><td>0.6270</td><td>0.8540</td><td>15432</td></tr><tr><td>BM25 + CEQE-Mul (t)</td><td>0.3203</td><td>0.4987</td><td>0.5941</td><td>0.8097</td><td>15612</td><td>0.2999</td><td>0.4749</td><td>0.5825</td><td>0.8447</td><td>14887</td></tr><tr><td colspan="11">ColBERT-PRF Models</td></tr><tr><td>ColBERT-PRF Ranker $\left( {\beta  = 1}\right)$</td><td>$\mathbf{{0.5431}}{abcdghijmnoqrst}$</td><td>${0.7352}^{adgrst}$</td><td>${0.8858}^{adt}$</td><td>${0.8706}^{abhmo}$</td><td>4103</td><td>${0.4962}^{adgjmrst}$</td><td>${0.6993}^{adgrst}$</td><td>${0.8396}^{ad}$</td><td>${\mathbf{{0.8892}}}^{abghlmno}$</td><td>4150</td></tr><tr><td>ColBERT-PRF ReRanker $\left( {\beta  = 1}\right)$</td><td>${0.5040}^{adgmnoqrst}$</td><td>${0.7369}^{adgrst}$</td><td>${0.8858}^{adt}$</td><td>0.7961</td><td>3543</td><td>${0.4919}^{adgjrst}$</td><td>${0.7006}^{adgrst}$</td><td>${0.8396}^{ad}$</td><td>${0.8431}^{m}$</td><td>3600</td></tr><tr><td>ColBERT-PRF Ranker $\left( {\beta  = {0.5}}\right)$</td><td>${0.5427}^{abcdghijmnoqrst}$</td><td>${0.7395}^{adgjmrst}$</td><td>${\mathbf{{0.8897}}}^{adt}$</td><td>${\mathbf{{0.8711}}}^{abhmo}$</td><td>4111</td><td>${\mathbf{{0.5116}}}^{adgjmnorst}$</td><td>${0.7153}^{adgjrst}$</td><td>${0.8439}^{ad}$</td><td>${0.8837}^{{agh}{lmno}}$</td><td>4155</td></tr><tr><td>ColBERT-PRF ReRanker $\left( {\beta  = {0.5}}\right)$</td><td>${0.5026}^{adgmnoqrst}$</td><td>${\mathbf{{0.7409}}}^{adgjmrst}$</td><td>${\mathbf{{0.8897}}}^{adt}$</td><td>0.7977</td><td>3470</td><td>${0.5063}^{adgjmrst}$</td><td>${\mathbf{{0.7161}}}^{\text{adgjrst }}$</td><td>${0.8439}^{ad}$</td><td>${0.8443}^{m}$</td><td>3477</td></tr></table>

<table><tbody><tr><td rowspan="2"></td><td colspan="5">2019年文本检索会议（TREC 2019，43个查询）</td><td colspan="5">2020年文本检索会议（TREC 2020，54个查询）</td></tr><tr><td>平均准确率均值（MAP）</td><td>前10名归一化折损累计增益（NDCG@10）</td><td>前10名平均倒数排名（MRR@10）</td><td>召回率（Recall）</td><td>平均响应时间（MRT）</td><td>平均准确率均值（MAP）</td><td>前10名归一化折损累计增益（NDCG@10）</td><td>前10名平均倒数排名（MRR@10）</td><td>召回率（Recall）</td><td>平均响应时间（MRT）</td></tr><tr><td colspan="11">词法检索方法（Lexical Retrieval Approaches）</td></tr><tr><td>二元独立模型25（BM25 (a)）</td><td>0.2864</td><td>0.4795</td><td>0.6416</td><td>0.7553</td><td>133</td><td>0.2930</td><td>0.4936</td><td>0.5912</td><td>0.8103</td><td>129</td></tr><tr><td>二元独立模型25+双向编码器表征变换器（BM25+BERT (b)）</td><td>0.4441</td><td>0.6855</td><td>0.8295</td><td>0.7553</td><td>3589</td><td>0.4699</td><td>0.6716</td><td>0.8069</td><td>0.8103</td><td>3554</td></tr><tr><td>二元独立模型25+ColBERT模型（BM25+ColBERT (c)）</td><td>0.4582</td><td>0.6950</td><td>0.8580</td><td>0.7553</td><td>202</td><td>0.4752</td><td>0.6931</td><td>0.8546</td><td>0.8103</td><td>203</td></tr><tr><td>二元独立模型25+相关反馈模型3（BM25+RM3 (d)）</td><td>0.3108</td><td>0.5156</td><td>0.6093</td><td>0.7756</td><td>201</td><td>0.3203</td><td>0.5043</td><td>0.5912</td><td>0.8423</td><td>248</td></tr><tr><td>二元独立模型25+相关反馈模型3+双向编码器表征变换器（BM25+RM3+BERT (e)）</td><td>0.4531</td><td>0.6862</td><td>0.8275</td><td>0.7756</td><td>4035</td><td>0.4739</td><td>0.6704</td><td>0.8079</td><td>0.8423</td><td>4003</td></tr><tr><td>二元独立模型25+相关反馈模型3+ColBERT模型（BM25+RM3+ColBERT (f)）</td><td>0.4709</td><td>0.7055</td><td>0.8651</td><td>0.7756</td><td>320</td><td>0.4800</td><td>0.6877</td><td>0.8560</td><td>0.8423</td><td>228</td></tr><tr><td colspan="11">神经增强方法（Neural Augmentation Approaches）</td></tr><tr><td>二元独立模型25+深度上下文词项（BM25+DeepCT (g)）</td><td>0.3169</td><td>0.5599</td><td>0.7155</td><td>0.7321</td><td>54</td><td>0.3570</td><td>0.5603</td><td>0.7090</td><td>0.8008</td><td>64</td></tr><tr><td>二元独立模型25+深度上下文词项+双向编码器表征变换器（BM25+DeepCT+BERT (h)）</td><td>0.4308</td><td>0.7011</td><td>0.8483</td><td>0.7321</td><td>3737</td><td>0.4671</td><td>0.6852</td><td>0.8068</td><td>0.8008</td><td>3719</td></tr><tr><td>二元独立模型25+深度上下文词项+ColBERT模型（BM25+DeepCT+ColBERT (i)）</td><td>0.4416</td><td>0.7004</td><td>0.8541</td><td>0.7321</td><td>129</td><td>0.4757</td><td>0.7071</td><td>0.8549</td><td>0.8008</td><td>141</td></tr><tr><td>二元独立模型25+文档T5查询（BM25+docT5query (j)）</td><td>0.4044</td><td>0.6308</td><td>0.7614</td><td>0.8263</td><td>282</td><td>0.4082</td><td>0.6228</td><td>0.7434</td><td>0.8456</td><td>295</td></tr><tr><td>二元独立模型25+文档T5查询+双向编码器表征变换器（BM25+docT5query+BERT (k)）</td><td>0.4802</td><td>0.7123</td><td>0.8483</td><td>0.8263</td><td>8025</td><td>0.4714</td><td>0.6810</td><td>0.8160</td><td>0.8456</td><td>3888</td></tr><tr><td>二元独立模型25+文档T5查询+ColBERT模型（BM25+docT5query+ColBERT (l)）</td><td>0.5009</td><td>0.7136</td><td>0.8367</td><td>0.8263</td><td>2362</td><td>0.4733</td><td>0.6934</td><td>0.8021</td><td>0.8456</td><td>2381</td></tr><tr><td colspan="11">密集检索模型（Dense Retrieval Models）</td></tr><tr><td>自适应神经上下文编码器（ANCE (m)）</td><td>0.3715</td><td>0.6537</td><td>0.8590</td><td>0.7571</td><td>199</td><td>0.4070</td><td>0.6447</td><td>0.7898</td><td>0.7737</td><td>179</td></tr><tr><td>自适应神经上下文编码器-伪相关反馈（ANCE-PRF (n)）</td><td>0.4253</td><td>0.6807</td><td>0.8492</td><td>0.7912</td><td>-</td><td>0.4452</td><td>0.6948</td><td>0.8371</td><td>0.8148</td><td>-</td></tr><tr><td>ColBERT端到端模型（ColBERT E2E (o)）</td><td>0.4318</td><td>0.6934</td><td>0.8529</td><td>0.7892</td><td>581</td><td>0.4654</td><td>0.6871</td><td>0.8525</td><td>0.8245</td><td>600</td></tr><tr><td colspan="11">双向编码器表征变换器-查询扩展重排模型（BERT-QE Reranking Models）</td></tr><tr><td>二元独立模型25 + 相关反馈模型3 + ColBERT模型 + 双向编码器表征变换器-输出扩展（BM25 + RM3 + ColBERT + BERT-OE (p)）</td><td>0.4832</td><td>0.7179</td><td>0.8754</td><td>0.7756</td><td>1130</td><td>0.4842</td><td>0.6909</td><td>0.8315</td><td>0.8423</td><td>1595</td></tr><tr><td>ColBERT端到端模型 + 双向编码器表征变换器-查询扩展（ColBERT E2E + BERT-QE (q)）</td><td>0.4423</td><td>0.7013</td><td>0.8683</td><td>0.7892</td><td>1261</td><td>0.4749</td><td>0.6911</td><td>0.8315</td><td>0.8245</td><td>1328</td></tr><tr><td colspan="11">基于嵌入的查询扩展模型（Embedding-based Query Expansion Models）</td></tr><tr><td>二元独立模型25 + 上下文嵌入查询扩展-最大值（BM25 + CEQE-Max (r)）</td><td>0.3453</td><td>0.5382</td><td>0.6605</td><td>0.8277</td><td>15656</td><td>0.3380</td><td>0.5094</td><td>0.6132</td><td>0.8561</td><td>16103</td></tr><tr><td>二元独立模型25 + 上下文嵌入输出扩展-质心（BM25 + CEOE-Centroid (s)）</td><td>0.3425</td><td>0.5345</td><td>0.6595</td><td>0.8234</td><td>14230</td><td>0.3302</td><td>0.5099</td><td>0.6270</td><td>0.8540</td><td>15432</td></tr><tr><td>二元独立模型25 + 上下文嵌入查询扩展-乘法（BM25 + CEQE-Mul (t)）</td><td>0.3203</td><td>0.4987</td><td>0.5941</td><td>0.8097</td><td>15612</td><td>0.2999</td><td>0.4749</td><td>0.5825</td><td>0.8447</td><td>14887</td></tr><tr><td colspan="11">ColBERT伪相关反馈模型（ColBERT-PRF Models）</td></tr><tr><td>ColBERT伪相关反馈排序器 $\left( {\beta  = 1}\right)$</td><td>$\mathbf{{0.5431}}{abcdghijmnoqrst}$</td><td>${0.7352}^{adgrst}$</td><td>${0.8858}^{adt}$</td><td>${0.8706}^{abhmo}$</td><td>4103</td><td>${0.4962}^{adgjmrst}$</td><td>${0.6993}^{adgrst}$</td><td>${0.8396}^{ad}$</td><td>${\mathbf{{0.8892}}}^{abghlmno}$</td><td>4150</td></tr><tr><td>ColBERT伪相关反馈重排序器 $\left( {\beta  = 1}\right)$</td><td>${0.5040}^{adgmnoqrst}$</td><td>${0.7369}^{adgrst}$</td><td>${0.8858}^{adt}$</td><td>0.7961</td><td>3543</td><td>${0.4919}^{adgjrst}$</td><td>${0.7006}^{adgrst}$</td><td>${0.8396}^{ad}$</td><td>${0.8431}^{m}$</td><td>3600</td></tr><tr><td>ColBERT伪相关反馈排序器 $\left( {\beta  = {0.5}}\right)$</td><td>${0.5427}^{abcdghijmnoqrst}$</td><td>${0.7395}^{adgjmrst}$</td><td>${\mathbf{{0.8897}}}^{adt}$</td><td>${\mathbf{{0.8711}}}^{abhmo}$</td><td>4111</td><td>${\mathbf{{0.5116}}}^{adgjmnorst}$</td><td>${0.7153}^{adgjrst}$</td><td>${0.8439}^{ad}$</td><td>${0.8837}^{{agh}{lmno}}$</td><td>4155</td></tr><tr><td>ColBERT伪相关反馈重排序器 $\left( {\beta  = {0.5}}\right)$</td><td>${0.5026}^{adgmnoqrst}$</td><td>${\mathbf{{0.7409}}}^{adgjmrst}$</td><td>${\mathbf{{0.8897}}}^{adt}$</td><td>0.7977</td><td>3470</td><td>${0.5063}^{adgjmrst}$</td><td>${\mathbf{{0.7161}}}^{\text{adgjrst }}$</td><td>${0.8439}^{ad}$</td><td>${0.8443}^{m}$</td><td>3477</td></tr></tbody></table>

Superscripts a...p denote significant improvements over the indicated baseline model(s). The highest value in each column is boldfaced. The higher MRT of BM25+ docT5query+ColBERT is expected, as we do not have a ColBERT index for the docT5query representation.

上标a...p表示相对于指定的基线模型有显著改进。每列中的最高值用粗体显示。BM25 + docT5query + ColBERT的平均响应时间（MRT）较高是可以预期的，因为我们没有针对docT5query表示的ColBERT索引。

<!-- Media -->

<!-- Media -->

Table 3. Comparison of Different PRF Mechanisms: (i) Numbers of Queries Improved, Unchanged or Degraded Compared to their Respective Baselines; (ii) Performance Improvement Correlation (Spearman’s $\rho$ Correlation Coefficient) between Pairs of PRF Mechanisms

表3. 不同伪相关反馈（PRF）机制的比较：（i）与各自基线相比，查询得到改进、不变或变差的数量；（ii）成对的伪相关反馈（PRF）机制之间的性能改进相关性（斯皮尔曼$\rho$相关系数）

<table><tr><td/><td>BM25+RM3 vs. BM25</td><td>ANCE-PRF vs. ANCE</td><td>ColBERT-PRF vs. ColBERT E2E</td></tr><tr><td/><td>Improved/Unchanged/Degraded 23/1/19</td><td>Improved/Unchanged/Degraded 26/1/16</td><td>Improved/Unchanged/Degraded 30/0/13</td></tr><tr><td>BM25+RM3 vs. BM25</td><td>1.00</td><td>0.37</td><td>0.34</td></tr><tr><td>ANCE-PRF vs. ANCE</td><td>0.37</td><td>1.00</td><td>0.41</td></tr><tr><td>ColBERT-PRF vs. ColBERT E2E</td><td>0.34</td><td>0.41</td><td>1.00</td></tr></table>

<table><tbody><tr><td></td><td>BM25+RM3与BM25对比</td><td>ANCE-PRF与ANCE对比</td><td>ColBERT-PRF与ColBERT端到端（E2E）对比</td></tr><tr><td></td><td>提升/不变/下降 23/1/19</td><td>提升/不变/下降 26/1/16</td><td>提升/不变/下降 30/0/13</td></tr><tr><td>BM25+RM3与BM25对比</td><td>1.00</td><td>0.37</td><td>0.34</td></tr><tr><td>ANCE-PRF与ANCE对比</td><td>0.37</td><td>1.00</td><td>0.41</td></tr><tr><td>ColBERT-PRF与ColBERT端到端（E2E）对比</td><td>0.34</td><td>0.41</td><td>1.00</td></tr></tbody></table>

<!-- Media -->

For RQ2(a), we compare the ColBERT-PRF Ranker and ReRanker models with the lexical retrieval approaches. For both query sets, both Ranker and ReRanker provide significant improvements on all evaluation measures compared to the BM25 and BM25+RM3 models. This is mainly due to the more effective contexualised representation employed in the ColBERT-PRF models than the traditional sparse representation used in the lexical retrieval approaches. Furthermore, both ColBERT-PRF Ranker and ReRanker outperform the sparse retrieval approaches when reranked by either the BERT or the ColBERT models - e.g., BM25+(Col)BERT and BM25+RM3+(Col)BERT - on all metrics. In particular, ColBERT-PRF Ranker exhibits marked improvements over the BM25 with BERT or ColBERT reranking approach for MAP on the TREC 2019 queries. This indicates that our query expansion in the contextualised embedding space produces query representations that result in improved retrieval effectiveness. Hence, in answer to RQ2(a), we find that our proposed ColBERT-PRF models show significant improvements in retrieval effectiveness over sparse baselines.

对于研究问题2(a)，我们将ColBERT-PRF排序器（Ranker）和重排序器（ReRanker）模型与词法检索方法进行了比较。对于两个查询集，与BM25和BM25+RM3模型相比，排序器和重排序器在所有评估指标上都有显著提升。这主要是因为ColBERT-PRF模型采用了比词法检索方法中使用的传统稀疏表示更有效的上下文表示。此外，当通过BERT或ColBERT模型进行重排序时，ColBERT-PRF排序器和重排序器在所有指标上都优于稀疏检索方法，例如BM25+(Col)BERT和BM25+RM3+(Col)BERT。特别是，在TREC 2019查询的平均准确率均值（MAP）方面，ColBERT-PRF排序器相对于使用BERT或ColBERT重排序的BM25方法有显著改进。这表明我们在上下文嵌入空间中的查询扩展产生了能够提高检索效果的查询表示。因此，针对研究问题2(a)，我们发现我们提出的ColBERT-PRF模型在检索效果上比稀疏基线有显著提升。

To further gauge the extent of improvements brought by the PRF additional information in the sparse retrieval and the dense retrieval paradigms, we compare the amount of performance improvements in terms of MAP for ColBERT-PRF vs. ColBERT, ANCE-PRF vs. ANCE, and BM25+RM3 vs. BM25 in Figure 3. We observe that more queries improved, and by a larger margin, by ColBERT-PRF compared to both RM3 and ANCE-PRF. Furthermore, from Figure 3, we find that among the failed queries for ColBERT-PRF, most of these queries also failed for the ANCE-PRF and RM3 approaches. These queries are hard queries that may struggle to be improved by a PRF technique. On the other hand, in Table 3, we present the number of queries whose performances are improved, unchanged and degraded when comparing a retrieval system with and without a PRF mechanism applied. We find that ColBERT-PRF has the highest number of improved queries and the lowest number of degraded queries. In the bottom half of Table 3, we compute Spearman’s $\rho$ correlation coefficient between the performance improvements of different PRF methods - a high positive correlation coefficient would be indicative that the two methods demonstrate a similar effect on different types of queries. From Table 3, we see that the correlation coefficient between ColBERT-PRF vs. ColBERT and ANCE-PRF vs. ANCE is highest among all the compared pairs (0.41). Overall, this tells us that while there is no strong correlations between the queries improved by applying PRF to each baseline, ColBERT-PRF and ANCE-PRF are the most correlated pair. Indeed, only moderate correlations are observed, showing that the approaches improve different queries. Moreover, from Figure 3 we see that ColBERT-PRF improves more queries and with further margin than ANCE-PRF.

为了进一步衡量伪相关反馈（PRF）额外信息在稀疏检索和密集检索范式中带来的改进程度，我们在图3中比较了ColBERT-PRF与ColBERT、ANCE-PRF与ANCE以及BM25+RM3与BM25在平均准确率均值（MAP）方面的性能提升幅度。我们观察到，与RM3和ANCE-PRF相比，ColBERT-PRF改进的查询更多，且改进幅度更大。此外，从图3中我们发现，在ColBERT-PRF失败的查询中，大多数查询在ANCE-PRF和RM3方法中也失败了。这些查询是难以通过PRF技术改进的难题。另一方面，在表3中，我们列出了应用和未应用PRF机制的检索系统在性能上得到改进、保持不变和下降的查询数量。我们发现ColBERT-PRF改进的查询数量最多，性能下降的查询数量最少。在表3的下半部分，我们计算了不同PRF方法性能提升之间的斯皮尔曼$\rho$相关系数——较高的正相关系数表明这两种方法对不同类型的查询表现出相似的效果。从表3中我们可以看到，ColBERT-PRF与ColBERT以及ANCE-PRF与ANCE之间的相关系数在所有比较对中最高（0.41）。总体而言，这告诉我们，虽然对每个基线应用PRF改进的查询之间没有很强的相关性，但ColBERT-PRF和ANCE-PRF是相关性最强的一对。实际上，只观察到中等相关性，这表明这些方法改进了不同的查询。此外，从图3中我们可以看到，ColBERT-PRF改进的查询更多，且改进幅度比ANCE-PRF更大。

For RQ2(b), on analysing the neural augmentation approaches, we observe that both the DeepCT and docT5query neural components could lead to effectiveness improvements over the corresponding lexical retrieval models without neural augmentation. However, despite their improved effectiveness, our proposed ColBERT-PRF models exhibit marked improvements over the neural augmentation approaches. Specifically, on the TREC 2019 query set, ColBERT-PRF Ranker significantly outperforms four out of six neural augmentation baselines and the BM25+DeepCT baseline on MAP. Meanwhile, both ColBERT-PRF Ranker and ReRanker exhibit significant improvements over BM25+DeepCT and BM25+docT5query on MAP for TREC 2020 queries, and exhibit improvements up to 9.5% improvements over neural augmentation approaches with neural re-ranking (e.g., MAP 0.4671 $\rightarrow  {0.5116}$ ). On analysing these comparisons,the effectiveness of the ColBERT-PRF models indicates that the query representation enrichment in a contextualised embedding space leads to a higher effectiveness performance than the sparse representation passage enrichment. Thus, in response to RQ2(b), the ColBERT-PRF models exhibit markedly higher performances than the neural augmentation approaches.

对于研究问题2(b)，在分析神经增强方法时，我们观察到DeepCT和docT5query神经组件都可以使相应的无神经增强的词法检索模型的效果得到提升。然而，尽管这些神经增强方法的效果有所改善，但我们提出的ColBERT-PRF模型相对于神经增强方法有显著改进。具体来说，在TREC 2019查询集上，ColBERT-PRF排序器在平均准确率均值（MAP）方面显著优于六个神经增强基线中的四个以及BM25+DeepCT基线。同时，对于TREC 2020查询，ColBERT-PRF排序器和重排序器在平均准确率均值（MAP）方面相对于BM25+DeepCT和BM25+docT5query有显著改进，并且相对于采用神经重排序的神经增强方法（例如，平均准确率均值为0.4671 $\rightarrow  {0.5116}$）的改进幅度高达9.5%。通过分析这些比较结果，ColBERT-PRF模型的有效性表明，在上下文嵌入空间中对查询表示进行丰富比稀疏表示段落丰富能带来更高的检索效果。因此，针对研究问题2(b)，ColBERT-PRF模型的性能明显高于神经增强方法。

<!-- Media -->

<!-- figureText: 0.6 ANCE-PRF vs. ANCE BM25+RM3 vs. BM25 면네네버펜əʌ leɔ!ueyɔәw jo uo!quyәp s әuɛɔ!pəw :zogɪstr әp!ɔ!ns ^ueɪ!!!w.jo sәsneɔ :zst/ - Asped level about eviques/p jo sadA- Estatz as/jeue @puell!@Auns jíve de s! \{eym :9LLSITT -ˌleɪəɔsɪʌ əuɪʃəp : Loo90TT eəy jo әp/> әjli s! buo! moy : http://t рэроо|q шцем sy.eus әwos ә.ne моч :98L 궐uəw6pn! Kuozeuepəp uo1!u!yəp :0ISOET p!ozәdeɪŋ e jo quəw'bəsp!w әy] puy oː moy :6 [MM J혁겐넥 Keliaәunioʌ sn ayi pip Kym :0SLE90I [ˈuəˌwɪlˌʌuə ʃo uɒlˈuɪʃəp lɛɔ:ˈbɔ:ˌbɔ:ˌbɔ:ˌdɔ:ˌdʌuəˌdʌkɪzˌeɪ ˈsɛɛsɪz'ɪ əɔnuds jo uondupsəp iеɔls/qd s! Jeym : Letz ETIT səsneɔ uʃed ɔ!ʌləd 146μ :tɒz68t эр!ʌ snos yoo> no< ue> poo! jo sad/q 2eum :E6S9 usq pue u.uәәmɪəq əɔuəJəjj!p :8ES8tT 【ydouquәdʌy ueinɔ:uәuәʌ 겨희 jo səsneɔ :ɪst еɔ!eweʃ u! uəqıeəm əqɪ s! mou : 29IEETI ADX WORLULLO P ESU :S6S06D -jәb.unqәsәәyp әjqnop e pue ә(qnopɔw e uəəm'əq əɔl uo.plu!Jap qou.jp/adeu/s Jo sleu!WJA LOXE :EZ6LD - әjdoәd !euɪ jo әj!! ^!!ep әuʒ s! \{e\\M : It yìoqìànjq sʌ yìm sì qìeum :66TOIII y겨eəy jo squeu!Wuāzap jepos ayi ave jeym :trZLELS 0.4 Delta MAP 0.2 0.0 -0.2 jo qued әue spiәqɔeɪ :0ɪztzɪt - upper and added the ?әwais e jo uon!uyәp :ɛt8tɛt бщ↓○○は ヨネシ↓○○ ↓○↓→↓↓ ↓○ qs○○ : [98t:01 Гецб 긴Эод sqо」 sqм :86LLE01 ABO|O!Q UO!1!U!J|BP SUOXQ :8LEZ8T lo』 pəqµɔsəud 』!ʌwe」 s! \{eum :9†9†TI speed upward todate bps s! : $L$ TLSOD suəˌuəɔuəd əəˌuʒiəuə reum:60LIZI pәpuno] Kwue uoдеʌея əyı sem uəчм :6LTZ9( әɔnpәu bu:moid unoquoo ueo qeum :zotIZII 4.0!u!u!Jap SMEI :96EET səˌə:s quəpuədəpu! jo yɔlēәmuowwoɔ әyː pəwɒɡ o4m :zt8E0T ә.ɪnɪeɪədwəŋ reqM mojəq p!nb! e s! uәbo.jp/y : LEZ6ZIT e!uownəud əsneɔ el!ydownəud elləuo!bəl səop :9ɪZ89I jo asisuoɔ quəwdinbə jeɔ!pəw ə!qe.inp s! \{eym : 6[8])TIT риердэдмв цероодлепndоd эош эчдечм :098ɛɛ8 молб чя!jpюб ор :ɛбtря ⑩дɔun」 ɔ!u0q0u0w ,b әjdwexa :6ESZ8T -->

<img src="https://cdn.noedgeai.com/0195b372-2fb6-7dbd-a6e2-deb0bb3cad1c_15.jpg?x=142&y=228&w=1277&h=1128&r=0"/>

Fig. 3. Per-query analysis on the TREC 2019 query set.

图3. TREC 2019查询集的逐查询分析。

<!-- Media -->

We further compare the ColBERT-PRF models with the recently proposed BERT-QE Reranking model. In particular, we provide results when using BERT-QE to rerank both BM25+RM3 as well as ColBERT E2E. Before comparing the ColBERT-PRF models with the BERT-QE rerankers, we first note that BERT-QE doesn't provide benefit to MAP on either query set, but can lead to a marginal improvement for NDCG@10 and MRR@10.However, the BERT-QE reranker models still underperform compared to our ColBERT-PRF models. Indeed, ColBERT E2E+BERT-QE exhibits a performance significantly lower than both ColBERT-PRF Ranker and ReRanker on the TREC 2019 query set. Hence, in response to RQ2(c), we find that the ColBERT-PRF models significantly outperform the BERT-QE reranking models.

我们进一步将ColBERT-PRF模型与最近提出的BERT-QE重排序模型进行比较。具体而言，我们给出了使用BERT-QE对BM25+RM3以及ColBERT端到端（E2E）结果进行重排序时的结果。在将ColBERT-PRF模型与BERT-QE重排序器进行比较之前，我们首先注意到，BERT-QE在任何一个查询集上都没有提升平均准确率均值（MAP，Mean Average Precision），但可以使前10名归一化折损累积增益（NDCG@10，Normalized Discounted Cumulative Gain at 10）和前10名平均倒数排名（MRR@10，Mean Reciprocal Rank at 10）略有提升。然而，与我们的ColBERT-PRF模型相比，BERT-QE重排序器模型的性能仍然较差。实际上，在TREC 2019查询集上，ColBERT E2E+BERT-QE的性能明显低于ColBERT-PRF排序器和重排序器。因此，针对研究问题2（c），我们发现ColBERT-PRF模型的性能明显优于BERT-QE重排序模型。

<!-- Media -->

<!-- figureText: 0.6 0.6 ##fish 0.4 0.2 innocent 0.0 -0.2 cared stunt -0.4 -0.50 -0.25 0.00 0.25 0.50 0.75 (b) Cluster centroids, $K = {64}$ . ###fish 0.4 0.2 0.0 only in -0.2 -0.4 gold -0.50 -0.25 0.00 0.25 0.50 0.75 (a) Cluster centroids, $K = 8$ . -->

<img src="https://cdn.noedgeai.com/0195b372-2fb6-7dbd-a6e2-deb0bb3cad1c_16.jpg?x=194&y=234&w=1169&h=474&r=0"/>

Fig. 4. Embeddings selected using different number of clustering centroids $K$ for the query ’do goldfish grow’; point size is representative of the magnitude of IDF.

图4. 针对查询“金鱼会长大吗”，使用不同数量的聚类质心$K$选择的嵌入；点的大小代表逆文档频率（IDF，Inverse Document Frequency）的大小。

<!-- Media -->

Finally, we consider the mean response times reported in Table 2, noting that ColBERT PRF exhibits higher response times than other ColBERT-based baselines, and similar to BERT-based re-rankers. There are several reasons for ColBERT PRF's speed: Firstly, the KMeans clustering of the feedback embeddings is conducted online, and the scikit-learn implementation we used is fairly slow - we tried other markedly faster KMeans implementations, but they were limited in terms of effectiveness (particularly for MAP), perhaps due to the lack of the KMeans++ initialisation procedure [5], which scikit-learn adopts; Secondly ColBERT PRF adds more expansion embeddings to the query - for the ranking setup, each feedback embedding can potentially cause a further ${k}^{\prime } = {1000}$ passages to be scored - further tuning of ColBERT’s ${k}^{\prime }$ parameter may allow efficiency improvements for ColBERT-PRF without much loss of effectiveness, at least for the first retrieval stage. Based on this, we further investigate how to attain more of a balance between the effectiveness and the efficiency in leveraging techniques such as approximate scoring technique [26] and other clustering algorithms.

最后，我们考虑表2中报告的平均响应时间，注意到ColBERT PRF的响应时间比其他基于ColBERT的基线模型更长，与基于BERT的重排序器相似。ColBERT PRF速度较慢有几个原因：首先，反馈嵌入的K均值（KMeans）聚类是在线进行的，我们使用的scikit-learn实现相当慢——我们尝试了其他明显更快的KMeans实现，但它们在有效性（特别是对于MAP）方面受到限制，这可能是由于缺乏scikit-learn采用的KMeans++初始化过程[5]；其次，ColBERT PRF为查询添加了更多的扩展嵌入——在排序设置中，每个反馈嵌入都可能导致进一步对${k}^{\prime } = {1000}$个段落进行评分——进一步调整ColBERT的${k}^{\prime }$参数可能会在不显著损失有效性的情况下提高ColBERT-PRF的效率，至少在第一个检索阶段是这样。基于此，我们进一步研究如何在利用近似评分技术[26]和其他聚类算法等技术时，在有效性和效率之间取得更好的平衡。

5.3.3 Results for RQ3 - Impact of ColBERT-PRF Parameters. To address RQ3, we investigate the impact of the parameters of ColBERT-PRF. In particular, when varying the values of a specific hyper-parameter type,we fix all the other hyper-parameters to their default setting,i.e., ${f}_{b} = 3$ , ${f}_{e} = {10},\beta  = 1$ and $k = {24}$ . Firstly,concerning the number of clusters, $K$ ,and the number of expansion embeddings ${f}_{e}$ selected from those clusters $\left( {{f}_{e} \leq  K}\right)$ ,Figures 5(a) and (b) report,for ColBERT-PRF Ranker and ColBERT-PRF ReRanker, respectively, the MAP (y-axis) performance for different ${f}_{e}$ (x-axis) selected from $K$ clusters (different curves). We observe that,with the same number of clusters and expansion embeddings, ColBERT-PRF Ranker exhibits a higher MAP performance than ColBERT-PRF ReRanker - as we also observed in Section 5.3.1.

5.3.3 研究问题3的结果——ColBERT-PRF参数的影响。为了解决研究问题3，我们研究了ColBERT-PRF参数的影响。具体而言，当改变特定超参数类型的值时，我们将所有其他超参数固定为其默认设置，即${f}_{b} = 3$、${f}_{e} = {10},\beta  = 1$和$k = {24}$。首先，关于聚类数量$K$以及从这些聚类$\left( {{f}_{e} \leq  K}\right)$中选择的扩展嵌入数量${f}_{e}$，图5（a）和（b）分别报告了ColBERT-PRF排序器和ColBERT-PRF重排序器在从$K$个聚类（不同曲线）中选择不同${f}_{e}$（x轴）时的MAP（y轴）性能。我们观察到，在聚类数量和扩展嵌入数量相同的情况下，ColBERT-PRF排序器的MAP性能高于ColBERT-PRF重排序器——这与我们在5.3.1节中观察到的结果一致。

Then,for a given ${f}_{e}$ value,Figures 5(a) and (b) show that the best performance is achieved by ColBERT-PRF when using $K = {24}$ . To explain this,we refer to Figure 4 together with Figure 2(b), which both show the centroid embeddings obtained using different numbers of clusters $K$ . Indeed, if the number of clusters $K$ is too small,the informativeness of the returned embeddings would be limited. For instance, in Figure 4(a), the centroid embeddings represent stopwords such as 'in' and '##' are included, which are unlikely to be helpful for retrieving more relevant passages. However, if $K$ is too large,the returned embeddings contain more noise,and hence are not suitable for expansion - for instance,using $K = {64}$ ,feedback embeddings representing ’innocent’ and ’stunt’ are identified in Figure 4(b), which could cause a topic drift.

然后，对于给定的${f}_{e}$值，图5（a）和（b）表明，ColBERT-PRF在使用$K = {24}$时实现了最佳性能。为了解释这一点，我们参考图4和图2（b），这两个图都显示了使用不同数量的聚类$K$获得的质心嵌入。实际上，如果聚类数量$K$太小，返回的嵌入的信息量将受到限制。例如，在图4（a）中，质心嵌入包含了诸如“in”和“##”等停用词，这些词不太可能有助于检索更相关的段落。然而，如果$K$太大，返回的嵌入包含更多的噪声，因此不适合用于扩展——例如，使用$K = {64}$时，在图4（b）中识别出了代表“无辜的”和“特技”的反馈嵌入，这可能会导致主题漂移。

Next,we analyse the impact of the number of feedback passages, ${f}_{b}$ . Figure 5(c) reports the MAP performance in response to different number of ${f}_{b}$ for both ColBERT-PRF Ranker and ReRanker. We observe that,when ${f}_{b} = 3$ ,both Ranker and ReRanker obtain their peak MAP values. In addition,for a given ${f}_{b}$ value,the Ranker exhibits a higher performance than the ReRanker. Similar to existing PRF models, we also find that considering too many feedback passages causes a query drift, in this case by identifying unrelated embeddings.

接下来，我们分析反馈段落数量 ${f}_{b}$ 的影响。图 5(c) 展示了 ColBERT - PRF 排序器（Ranker）和重排序器（ReRanker）在不同 ${f}_{b}$ 数量下的平均准确率均值（Mean Average Precision，MAP）性能。我们观察到，当 ${f}_{b} = 3$ 时，排序器和重排序器均达到其 MAP 值的峰值。此外，对于给定的 ${f}_{b}$ 值，排序器的性能优于重排序器。与现有的伪相关反馈（Pseudo - Relevance Feedback，PRF）模型类似，我们还发现考虑过多的反馈段落会导致查询漂移，在这种情况下是通过识别不相关的嵌入来实现的。

<!-- Media -->

<!-- figureText: 0.550 K=16 K=16 0.50 K=24 K=36 0.48 K=48 MAP K=64 0.46 0.44 0.42 60 10 35 (b) Impact of $K$ and ${f}_{e}$ on ColBERT-PRF ReRanker. 0.550 0.525 0.500 MAP 0.475 0.450 ColBERT-PRF Ranker ColBERT-PRF ReRan 0 0.4 0.8 1.2 1.6 2.0 2.4 2.8 $\beta$ (d) Impact of expansion embedding weight $\beta$ . K=24 0.525 K=36 K=48 MAP 0.500 K=64 0.450 60 (a) Impact of $K$ and ${f}_{e}$ on ColBERT-PRF Ranker. 0.525 0.500 MAP 0.475 0.450 ColBERT-PRF Ranker ColBERT-PRF ReRanke 0.425 3 8 10 15 20 ${f}_{b}$ (c) Impact of pseudo-relevance feedback size ${f}_{b}$ . -->

<img src="https://cdn.noedgeai.com/0195b372-2fb6-7dbd-a6e2-deb0bb3cad1c_17.jpg?x=162&y=233&w=1225&h=940&r=0"/>

Fig. 5. MAP on the TREC 2019 query set while varying the number of clusters(K),number of expansion embeddings $\left( {f}_{e}\right)$ ,as well as the feedback set size ${f}_{b}$ and expansion embedding weight $\beta .\beta  = 0\& {f}_{e} = 0$ correspond to the original ColBERT.

图 5. 在 TREC 2019 查询集上，改变聚类数量（K）、扩展嵌入数量 $\left( {f}_{e}\right)$、反馈集大小 ${f}_{b}$ 以及扩展嵌入权重 $\beta .\beta  = 0\& {f}_{e} = 0$ 时的 MAP，这些参数对应于原始的 ColBERT。

<!-- Media -->

Finally,we analyse the impact of the $\beta$ parameter,which controls the emphasis of the expansion embeddings during the final passage scoring. Figure 5(d) reports MAP as $\beta$ is varied for ColBERT-PRF Ranker and ReRanker. From the figure, we observe that in both scenarios, the highest MAP is obtained for $\beta  \in  \left\lbrack  {{0.6},{0.8}}\right\rbrack$ ,but good effectiveness is maintained for higher values of $\beta$ ,which emphasises the high utility of the centroid embeddings for effective retrieval.

最后，我们分析 $\beta$ 参数的影响，该参数控制在最终段落评分过程中扩展嵌入的权重。图 5(d) 展示了 ColBERT - PRF 排序器和重排序器在 $\beta$ 变化时的 MAP。从图中可以看出，在两种情况下，当 $\beta  \in  \left\lbrack  {{0.6},{0.8}}\right\rbrack$ 时获得最高的 MAP，但对于较高的 $\beta$ 值，仍能保持较好的效果，这强调了质心嵌入在有效检索中的高实用性。

Overall, in response to RQ3, we find that ColBERT-PRF, similar to existing PRF approaches, is sensitive to the number of feedback passages and the number of expansion embeddings that are added to the query $\left( {{f}_{b}\& {f}_{e}}\right)$ as well as their relative importance during scoring (c.f. $\beta$ ). However,going further,the $K$ parameter of KMeans has a notable impact on performance: if too high, noisy clusters can be obtained; too low and the obtained centroids can represent stopwords. Yet, the stable and effective results across the hyperparameters demonstrate the overall promise of ColBERT-PRF.

总体而言，针对研究问题 3（RQ3），我们发现 ColBERT - PRF 与现有的 PRF 方法类似，对反馈段落的数量、添加到查询 $\left( {{f}_{b}\& {f}_{e}}\right)$ 中的扩展嵌入数量以及它们在评分过程中的相对重要性（参见 $\beta$）较为敏感。然而，进一步来看，K 均值（KMeans）的 $K$ 参数对性能有显著影响：如果该参数值过高，可能会得到噪声聚类；如果过低，得到的质心可能代表停用词。不过，在各个超参数下稳定且有效的结果表明了 ColBERT - PRF 的整体潜力。

5.3.4 Results for RQ4 - Semantic Matching by ColBERT-PRF. We now analyse the expansion embeddings and the retrieved passages in order to better understand the behaviour of ColBERT-PRF, and why it demonstrates advantages over traditional (sparse) QE techniques.

5.3.4 研究问题 4（RQ4）的结果 - ColBERT - PRF 的语义匹配。我们现在分析扩展嵌入和检索到的段落，以便更好地理解 ColBERT - PRF 的行为，以及它为何比传统（稀疏）的查询扩展（Query Expansion，QE）技术具有优势。

<!-- Media -->

Table 4. Examples of the Expanded Queries by the ColBERT PRF Model on the TREC 2019 & 2020 Query Sets

表 4. ColBERT PRF 模型在 TREC 2019 和 2020 查询集上的扩展查询示例

<table><tr><td>Original query terms</td><td>Original query tokens</td><td colspan="4">Most likely tokens for expansion embeddings</td></tr><tr><td colspan="6">TREC 2019 queries</td></tr><tr><td rowspan="3">what is an active margin</td><td rowspan="3">what is an active margin</td><td/><td>(by|opposition)</td><td>oceanic</td><td>volcanoes##cton</td></tr><tr><td colspan="4">(margin margins)(breeds|##kshi)continentalplate</td></tr><tr><td colspan="4">an each</td></tr><tr><td rowspan="2">what is wifi vs bluetooth</td><td rowspan="2">what is wi ##fi vs blue ##tooth</td><td colspan="4">##toothphonesdevicesblue</td></tr><tr><td>systems</td><td/><td>point</td><td/></tr><tr><td rowspan="3">what is the most popular food in switzerland</td><td rowspan="3">what is the most popular food in switzerland</td><td>##hs</td><td colspan="3">(swiss|switzerland)(influences|includes)</td></tr><tr><td colspan="2">(breeds|##kshi)</td><td>potato(dishes|food)</td><td>(bologna|hog)</td></tr><tr><td>cheese</td><td colspan="2">(italians french)</td><td/></tr><tr><td colspan="6">TREC 2020 queries</td></tr><tr><td rowspan="2">what is mamey</td><td rowspan="2">what is ma ##me ##</td><td>(is upset)</td><td/><td>(breeds|##kshi)</td><td/></tr><tr><td/><td/><td>##me (larger|more) central</td><td/></tr><tr><td rowspan="2">average annual income data an- alyst</td><td rowspan="2">average annual income data an- alyst</td><td colspan="2">(analyst analysts)</td><td>(breeds|##kshi)</td><td>(grow|growth)</td></tr><tr><td/><td/><td/><td/></tr><tr><td rowspan="2">do google docs auto save</td><td rowspan="2">do google doc ##s auto save</td><td colspan="2"/><td>(breeds|##kshi) doc (to|automatically) google docı</td><td/></tr><tr><td>save</td><td>(saves|saved)</td><td>drive(changes|revisions)</td><td>(back|to)</td></tr></table>

<table><tbody><tr><td>原始查询词</td><td>原始查询标记</td><td colspan="4">扩展嵌入最可能的标记</td></tr><tr><td colspan="6">TREC 2019查询</td></tr><tr><td rowspan="3">什么是活动边缘（Active Margin）</td><td rowspan="3">什么是活动边缘（Active Margin）</td><td></td><td>(通过|对立)</td><td>海洋的</td><td>火山（Volcanoes）##cton</td></tr><tr><td colspan="4">(边缘 边缘地带)(品种|##kshi)大陆板块（Continental Plate）</td></tr><tr><td colspan="4">每一个</td></tr><tr><td rowspan="2">Wi-Fi和蓝牙（Bluetooth）有什么区别</td><td rowspan="2">Wi ##Fi和Blue ##Tooth有什么区别</td><td colspan="4">##Tooth手机设备蓝牙</td></tr><tr><td>系统</td><td></td><td>点</td><td></td></tr><tr><td rowspan="3">瑞士最受欢迎的食物是什么</td><td rowspan="3">瑞士最受欢迎的食物是什么</td><td>##hs</td><td colspan="3">(瑞士的|瑞士)(影响|包括)</td></tr><tr><td colspan="2">(品种|##kshi)</td><td>土豆(菜肴|食物)</td><td>(博洛尼亚香肠|猪)</td></tr><tr><td>奶酪</td><td colspan="2">(意大利人 法国人)</td><td></td></tr><tr><td colspan="6">TREC 2020查询</td></tr><tr><td rowspan="2">什么是曼密苹果（Mamey）</td><td rowspan="2">什么是Ma ##Me ##</td><td>(心烦意乱)</td><td></td><td>(品种|##kshi)</td><td></td></tr><tr><td></td><td></td><td>##Me (更大|更)中心的</td><td></td></tr><tr><td rowspan="2">平均年收入数据分析师</td><td rowspan="2">平均年收入数据分析师</td><td colspan="2">(分析师 分析师们)</td><td>(品种|##kshi)</td><td>(增长|成长)</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">谷歌文档（Google Docs）会自动保存吗</td><td rowspan="2">谷歌Doc ##s会自动保存吗</td><td colspan="2"></td><td>(品种|##kshi)文档(到|自动地)谷歌Docı</td><td></td></tr><tr><td>保存</td><td>(保存 已保存)</td><td>云端硬盘(更改|修订)</td><td>(返回|到)</td></tr></tbody></table>

The symbol denotes that there are multiple tokens that are highly likely for a particular expansion embedding. Token with darker red colour indicate its higher effectiveness contribution.

该符号表示对于特定的扩展嵌入，有多个Token极有可能被使用。颜色越深的Token表示其有效性贡献越高。

<!-- Media -->

Firstly,it is useful to inspect tokens corresponding to the expansion embeddings. Table ${4}^{9}$ lists three example queries from both the TREC 2019 and 2020 query sets and their tokenised forms as well as the expansion tokens generated by the ColBERT-PRF model. For a given query, we used our default setting for the ColBERT-PRF model, i.e., selecting ten expansion embeddings; Equation (3) is used to resolve embeddings to tokens. On examination of Table 4, it is clear to see the relation of the expansion embeddings to the original query - for instance, we observe that expansion embeddings for the tectonic concept of active margin relate to 'oceanic', 'volcanoes' and 'continental' 'plate'. Overall, we find that most of the expansion tokens identified are credible supplementary information for each user query and can indeed clarify the information needs.

首先，检查与扩展嵌入对应的Token是很有用的。表${4}^{9}$列出了来自TREC 2019和2020查询集的三个示例查询及其分词形式，以及ColBERT - PRF模型生成的扩展Token。对于给定的查询，我们对ColBERT - PRF模型使用默认设置，即选择十个扩展嵌入；使用公式(3)将嵌入解析为Token。通过检查表4，可以清楚地看到扩展嵌入与原始查询之间的关系——例如，我们观察到活动边缘构造概念的扩展嵌入与“海洋”、“火山”和“大陆板块”相关。总体而言，我们发现识别出的大多数扩展Token是每个用户查询的可靠补充信息，确实可以明确信息需求。

To answer RQ4, we further conduct analysis to measure the ability to perform semantic matching within the ColBERT Max-Sim operation. In particular, we examine which of the query embed-dings match most strongly with a passage embedding that corresponds to exactly the same token - a so called exact match; in contrast a semantic match is a query embedding matching with a passage embedding which has a different token id. Indeed, in [13], the authors concluded that ColBERT is able to conduct exact matches for important terms based on their embedded representations. In contrast, little work has considered the extent that ColBERT-based models perform semantic (i.e., non-exact) matching. Thus, firstly, following [28], we look into the interaction matrix between the query and passage embeddings. Figure 6 describes the interaction matrix between the query "why did the us voluntarily enter ww1" expanded with 10 expansion embeddings and its top returned passage embeddings. ${}^{10}$ From Figure 6,we see that some query tokens,such as ’the’,’us’,’,and '##w', experience exact matching as these tokens are in the same form with their corresponding X. Wang et al. returned highest Max-Sim scored passage tokens. In contract, the remaining query tokens are performing semantic matching to the passage as their corresponding passage tokens with the highest Max-Sim score are in different lexical forms, for instance, query token 'why' matches with passage token 'reason'. In particular, the expansion token 'revolution' and 'entered', which does not exist in the original token but expanded using ColBERT-PRF, also performs the exact matching. In addition, the expansion tokens such as 'attacked' and 'harbour' further perform semantic matching to the passages. This further indicates the usefulness of the expansion tokens to improve the matching performance between query and passage pairs.

为了回答研究问题4，我们进一步进行分析，以衡量ColBERT最大相似度（Max - Sim）操作中的语义匹配能力。具体来说，我们研究哪些查询嵌入与对应于完全相同Token的段落嵌入匹配得最强烈——即所谓的精确匹配；相反，语义匹配是指查询嵌入与具有不同Token ID的段落嵌入匹配。实际上，在文献[13]中，作者得出结论，ColBERT能够基于其嵌入表示对重要术语进行精确匹配。相比之下，很少有研究考虑基于ColBERT的模型进行语义（即非精确）匹配的程度。因此，首先，按照文献[28]的方法，我们研究查询和段落嵌入之间的交互矩阵。图6描述了扩展了10个扩展嵌入的查询“美国为什么自愿参加第一次世界大战”与其返回的排名最高的段落嵌入之间的交互矩阵。${}^{10}$从图6中可以看出，一些查询Token，如“the”、“us”、“,”和“##w”，经历了精确匹配，因为这些Token与其对应的X. Wang等人返回的具有最高Max - Sim得分的段落Token形式相同。相反，其余的查询Token与段落进行语义匹配，因为它们对应的具有最高Max - Sim得分的段落Token在词法形式上不同，例如，查询Token“why”与段落Token“reason”匹配。特别是，扩展Token“revolution”和“entered”（原始Token中不存在，但使用ColBERT - PRF进行了扩展）也进行了精确匹配。此外，像“attacked”和“harbour”这样的扩展Token进一步与段落进行语义匹配。这进一步表明扩展Token有助于提高查询和段落对之间的匹配性能。

---

<!-- Footnote -->

${}^{9}$ In Table 4,the expansion embedding ’(breeds|#kkshi)’,which appears for each query,is projected to be close to the embedding of the [D] token, which ColBERT places in each passage.

${}^{9}$在表4中，每个查询中出现的扩展嵌入’(breeds|#kkshi)’被投影到接近ColBERT放置在每个段落中的[D] Token的嵌入。

${}^{10}$ We use a FAISS index to map embeddings back to most likely token.

${}^{10}$我们使用FAISS索引将嵌入映射回最可能的Token。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: enter ##1 [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] inning ## ✘ [CLS] [D] the reasons that the entered ##w ##1 the sinking of the ##sit ✘ ##ania the tel ##eg ✘ ##ra the russian revolution [SEP] -->

<img src="https://cdn.noedgeai.com/0195b372-2fb6-7dbd-a6e2-deb0bb3cad1c_19.jpg?x=319&y=232&w=921&h=755&r=0"/>

Fig. 6. ColBERT-PRF interaction matrix between query (qid: 106375) and passage (docid: 4337532) embed-dings. The darker shading indicate a higher similarity. The highest similarity among all the passage embed-dings for a given query embedding is highlighted with an X symbol. The histogram depicts the magnitude of contribution for each query embedding to the final score of the passage.

图6. 查询（查询ID：106375）和段落（文档ID：4337532）嵌入之间的ColBERT - PRF交互矩阵。颜色越深表示相似度越高。对于给定的查询嵌入，所有段落嵌入中相似度最高的用X符号突出显示。直方图描绘了每个查询嵌入对段落最终得分的贡献大小。

<!-- Media -->

To quantify the extent that semantic matching takes place, we follow [43] and employ a recent measure that inspects the Max-Sim, and determines whether each query embedding is matched with the same token (exact match) vs. an inexact (semantic) match with a different token. Formally, let ${t}_{i}$ and ${t}_{j}$ respectively denote the token id of the $i$ -th query embedding and $j$ -th passage embedding,respectively. Given a query $q$ and the set ${R}_{k}$ of the top ranked $k$ passages,the Semantic Match Proportion (SMP) at rank cutoff $k$ w.r.t. $q$ and ${R}_{k}$ is defined as:

为了量化语义匹配发生的程度，我们遵循文献[43]，采用一种最近的度量方法，该方法检查最大相似度（Max - Sim），并确定每个查询嵌入是与相同的Token匹配（精确匹配）还是与不同的Token进行不精确（语义）匹配。形式上，设${t}_{i}$和${t}_{j}$分别表示第$i$个查询嵌入和第$j$个段落嵌入的Token ID。给定一个查询$q$和排名最高的$k$个段落的集合${R}_{k}$，相对于$q$和${R}_{k}$，在排名截断值$k$处的语义匹配比例（SMP）定义为：

$$
{SMP}\left( {q,{R}_{k}}\right)  = \mathop{\sum }\limits_{{d \in  {R}_{k}}}\frac{\mathop{\sum }\limits_{{i \in  \operatorname{toks}\left( q\right) }}\mathbb{1}\left\lbrack  {{t}_{i} \neq  {t}_{j}}\right\rbrack   \cdot  \mathop{\max }\limits_{{j = 1,\ldots ,\left| d\right| }}{\phi }_{{q}_{i}}^{T}{\phi }_{{d}_{j}}}{\mathop{\sum }\limits_{{i \in  \operatorname{toks}\left( q\right) }}\mathop{\max }\limits_{{j = 1,\ldots ,\left| d\right| }}{\phi }_{{q}_{i}}^{T}{\phi }_{{d}_{j}}}, \tag{6}
$$

where toks(q)returns the indices of the query embeddings that correspond to BERT tokens,i.e.,not [CLS],[Q],or [MASK] tokens, ${}^{11}{R}_{k}$ is the top ranked $k$ passages,and $\mathbb{1}\left\lbrack  \right\rbrack$ is the indicator function.

其中toks(q)返回对应于BERT Token（即不是[CLS]、[Q]或[MASK] Token）的查询嵌入的索引，${}^{11}{R}_{k}$是排名最高的$k$个段落，$\mathbb{1}\left\lbrack  \right\rbrack$是指示函数。

Figure 7 depicts the per-query semantic matching proportion calculated at the rank cutoff 10 for the ColBERT-PRF and ColBERT E2E models on the TREC 2019 query set. From the figure, we observe that, when the expansion embeddings are added to the original query by ColBERT-PRF, SMP is increased for most of the queries over the original ColBERT E2E model. Next, on both TREC 2019 and TREC 2020 query sets,we investigate the impact of the rank cutoff $k$ to the semantic match proportion on ColBERT-PRF model instantiated as Ranker and ReRanker models as well as the ColBERT E2E model, which is portrayed in Figure 8. In general, from Figure 8, we can see that Mean SMP grows as the rank cutoff $k$ increases - this is expected,as we know that ColBERT prefers exact matches, and the number of exact matches will decreased by rank (resulting in increasing SMP). However, that ColBERT-PRF (both Ranker and Reranker) have, in general, higher SMP than the original ColBERT ranking. This verifies the results from Figure 7. The interesting exception is at the very highest ranks, where both ColBERT-PRF approaches exhibit lower SMP than the baseline. This suggests that at the very top ranks ColBERT-PRF exhibits higher preference for exact token matches than the E2E baseline. However, overall, the higher SMPs exhibited by ColBERT-PRF indicates that, at deeper ranks, the embedding-based query expansion has the ability to retrieve passages with less lexical exact match between the query and passage embeddings.

图7展示了在TREC 2019查询集上，ColBERT - PRF和ColBERT端到端（E2E）模型在排名截断值为10时计算得到的每个查询的语义匹配比例。从图中我们可以观察到，当ColBERT - PRF将扩展嵌入添加到原始查询中时，与原始的ColBERT E2E模型相比，大多数查询的语义匹配比例（SMP）有所提高。接下来，在TREC 2019和TREC 2020查询集上，我们研究了排名截断值$k$对ColBERT - PRF模型（实例化为排序器（Ranker）和重排序器（ReRanker）模型）以及ColBERT E2E模型的语义匹配比例的影响，如图8所示。总体而言，从图8中我们可以看到，平均语义匹配比例（Mean SMP）随着排名截断值$k$的增加而增长——这是符合预期的，因为我们知道ColBERT更倾向于精确匹配，而精确匹配的数量会随着排名下降（从而导致语义匹配比例增加）。然而，总体而言，ColBERT - PRF（包括排序器和重排序器）的语义匹配比例通常高于原始的ColBERT排名。这验证了图7的结果。有趣的例外是在最高排名处，两种ColBERT - PRF方法的语义匹配比例都低于基线。这表明在最高排名处，ColBERT - PRF比端到端基线更倾向于精确的词元匹配。然而，总体而言，ColBERT - PRF表现出的较高语义匹配比例表明，在更深的排名中，基于嵌入的查询扩展能够检索到查询和段落嵌入之间词汇精确匹配较少的段落。

---

<!-- Footnote -->

${}^{11}$ Indeed,[CLS],[Q],and [MASK] do not correspond to actual WordPiece tokens originating from the user's query and hence can never have exact matches, so we exclude them from this calculation.

${}^{11}$ 实际上，[CLS]、[Q]和[MASK]并不对应于源自用户查询的实际词块（WordPiece）词元，因此永远不会有精确匹配，所以我们将它们排除在此次计算之外。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: 489204: right pelvic pain causes ColBERT-PRF 0.1 573724: what are the social determinants of health 87181: causes of left ventricular hypertroph 168216: does legionella pneumophila cause pneumoniá 148538: difference between rn and bsn 1124210: tracheids are part of 130510: definition declaratory judgment 527433: types of dvsarthria from cerebral palsv 156493: do goldfish grow 490595: rsa definition key 182539: example of monotonic function 87452: causes of military suicide 1129237: hydrogen is a liquid below what temperature 1103812: who formed the commonwealth of independent states 1106007: define visceral 405717: is cdg airport in main paris 19335: anthropological definition of environment 207786: how are some sharks warm blooded 47923: axon terminals or synaptic knob definition 131843: definition of a sigmet 1117099: what is a active margin 104861: cost of interior concrete flooring 855410: what is theraderm used for 1114819: what is durable medical equipment consist of 1037798: who is robert grav 1133167: how is the weather in jamäicá 1121402: what can contour plowing reduce 1121709: what are the three percenters 915593: what types of food can you cook sous vide "1110199: what is wifi vs bluetooth 264014: how lona is life cvcle of flea 1115776: what is an aml surveillance analyst 1112341: what is the daily life of thai people 1113437: what is physical description of spruce 443396: Ips laws definition 183378: exons definition biology 1063750: why did the us volunterilay enter ww1 -->

<img src="https://cdn.noedgeai.com/0195b372-2fb6-7dbd-a6e2-deb0bb3cad1c_20.jpg?x=135&y=237&w=1276&h=723&r=0"/>

Fig. 7. Per-query semantic matching proportion measurements (measured to rank 10) for the CoIBERT E2E (shown as red bars) and ColBERT-PRF (shown as cyan bars) models on the TREC 2019 passage ranking query set.

图7. 在TREC 2019段落排名查询集上，ColBERT端到端（E2E）模型（以红色条形图表示）和ColBERT - PRF模型（以青色条形图表示）的每个查询的语义匹配比例测量结果（测量到排名10）。

<!-- Media -->

In addition, we further investigate the potential for topic drift when applying ColBERT-PRF with different number of expansion embeddings on the TREC 2019 queries. In particular,in Figure 9(a) ${}^{12}$ we measure retrieval effectiveness (MAP) as the number of expansion embeddings is varied and, in Figure 9(b), we present Mean SMP (y-axis) calculated upon the retrieved results after PRF, at different rank cutoffs (curves), also as the number of expansion embeddings is varied (x-axis).

此外，我们进一步研究了在TREC 2019查询上应用具有不同数量扩展嵌入的ColBERT - PRF时出现主题漂移的可能性。具体而言，在图9(a) ${}^{12}$中，我们测量了随着扩展嵌入数量的变化检索效果（平均准确率均值，MAP）；在图9(b)中，我们展示了在伪相关反馈（PRF）后，根据检索结果计算的平均语义匹配比例（Mean SMP，y轴），该比例是在不同排名截断值（曲线）下，随着扩展嵌入数量的变化（x轴）而计算得到的。

From Figure 9(a),we can see that ${f}_{e} = 8$ gives the highest (MAP) effectiveness (as also shown earlier in Figure 5(b)). At the same time,from Figure 9(b),we observe that (1) for $2 \leq  {f}_{e} \leq  8$ , Mean SMP falls; (2) however,for ${f}_{e} > 8$ ,Mean SMP rises again. This trend is apparent when Mean SMP is analysed for five or more retrieved passages. This suggests that when more than eight expansion embeddings are selected, excessive semantic matching occurs (Figure 9(b)) and effectiveness approaches MAP 0.50 (Figure 9(a)). As expansion embeddings are selected by using the IDF of the corresponding token, this suggests that given the size of the feedback set (three passages, with length up to 180 tokens and on average 77 tokens), for more than eight embeddings we are starting to select non-informative expansion embeddings that can only be semantically matched in the retrieved passages, and hence there is no further positive benefit in terms of effectiveness. However,as effectiveness does not markedly decrease for ${f}_{e} > 8$ ,this indicates that there is little risk of topic drift with ColBERT-PRF, due to the contextualised nature of the expansion embeddings. Overall, these analyses answer RQ4.

从图9(a)中，我们可以看到${f}_{e} = 8$给出了最高的（平均准确率均值，MAP）检索效果（如图5(b)中先前所示）。同时，从图9(b)中我们观察到：(1) 对于$2 \leq  {f}_{e} \leq  8$，平均语义匹配比例下降；(2) 然而，对于${f}_{e} > 8$，平均语义匹配比例再次上升。当分析五个或更多检索段落的平均语义匹配比例时，这种趋势很明显。这表明当选择超过八个扩展嵌入时，会出现过度的语义匹配（图9(b)），并且检索效果接近平均准确率均值0.50（图9(a)）。由于扩展嵌入是通过使用相应词元的逆文档频率（IDF）来选择的，这表明在给定反馈集的大小（三个段落，长度最长为180个词元，平均为77个词元）的情况下，当选择超过八个嵌入时，我们开始选择那些只能在检索段落中进行语义匹配的非信息性扩展嵌入，因此在检索效果方面没有进一步的积极益处。然而，由于对于${f}_{e} > 8$检索效果并没有显著下降，这表明由于扩展嵌入的上下文特性，ColBERT - PRF出现主题漂移的风险很小。总体而言，这些分析回答了研究问题4（RQ4）。

---

<!-- Footnote -->

${}^{12}$ This is a subset of the curves presented earlier in Figure 5(b),repeated here for ease of reference.

${}^{12}$ 这是图5(b)中先前展示的曲线的一个子集，在此重复以便参考。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: 0.35 0.45 0.40 Mean SMP 0.35 0.30 0.25 ColBERT E2E 0.20 ColBERT-PRF Ranker ColBERT-PRF ReRanker 0.15 1 5 9 13 17 21 23 25 27 29 Rank Cutoff (b) Mean SMP on TREC 2020 Mean SMP 0.30 0.25 ColBERT E2E 0.20 ColBERT-PRF Ranker ColBERT-PRF ReRanker 3 7 11 15 19 21 23 25 27 29 Rank Cutoff (a) Mean SMP on TREC 2019 -->

<img src="https://cdn.noedgeai.com/0195b372-2fb6-7dbd-a6e2-deb0bb3cad1c_21.jpg?x=176&y=234&w=1204&h=422&r=0"/>

Fig. 8. Mean Semantic Matching Proportion (Mean SMP) as rank varies.

图8. 平均语义匹配比例（Mean SMP）随排名的变化情况。

<!-- figureText: 0.51 Exp RankCutoff 1 0.40 Exp_RankCutoff_10 Exp_RankCutoff_15 Mean SMP 0.35 Exp_RankCutoff_20 0.30 0.25 0.20 0.15 10 20 ${f}_{e}$ (a) Retrieval effectiveness (MAP) in terms of dif-(b) Mean SMP performance in terms of different PRF ReRanker. 0.50 0.49 0.48 MAP 0.47 0.46 0.44 0.43 5 15 20 ${f}_{e}$ ferent number of expansion embeddings ${f}_{e}$ on number of expansion embeddings ${f}_{e}$ on ColBERT ColBERT-PRF ReRanker. -->

<img src="https://cdn.noedgeai.com/0195b372-2fb6-7dbd-a6e2-deb0bb3cad1c_21.jpg?x=166&y=761&w=1213&h=471&r=0"/>

Fig. 9. Potential topic drift analysis for ColBERT-PRF ReRanker on the TREC 2019 query set.

图9. 在TREC 2019查询集上对ColBERT - PRF重排序器进行的潜在主题漂移分析。

<!-- Media -->

## 6 DOCUMENT RANKING EFFECTIVENESS OF COLBERT-PRF

## 6 ColBERT - PRF的文档排名效果

After assessing the effectiveness of ColBERT-PRF on passage ranking in the previous section, we further demonstrate the performance of ColBERT-PRF on document ranking. In this task, documents are longer than passages, hence they need to be divided into smaller chuncks, with lengths comparable to those of passages. Moreover, in document ranking we do not fine tune the ColBERT model on the new collection due to the limited number of queries available; hence, we leverage the ColBERT model trained on the MSMARCO as detailed in Section 5.2, e.g., in a zero shot setting. Thus, in this section, we focus on testing the effectiveness of our proposed ColBERT-PRF for MSMARCO document retrieval task and the TREC Robust04 document retrieval task. Research questions and experimental setup for document ranking experiments are detailed in Sections 6.1 and 6.2, respectively. Results and analysis are discussed in Section 6.3.

在上一节评估了ColBERT-PRF在段落排序上的有效性之后，我们进一步展示ColBERT-PRF在文档排序上的性能。在这项任务中，文档比段落长，因此需要将它们分割成较小的块，其长度与段落相当。此外，在文档排序中，由于可用查询数量有限，我们不会在新的文档集合上微调ColBERT模型；因此，我们利用在MSMARCO上训练的ColBERT模型，如第5.2节所述，例如，在零样本设置下。因此，在本节中，我们专注于测试我们提出的ColBERT-PRF在MSMARCO文档检索任务和TREC Robust04文档检索任务中的有效性。文档排序实验的研究问题和实验设置分别在第6.1节和第6.2节详细介绍。结果和分析在第6.3节讨论。

### 6.1 Research Questions

### 6.1 研究问题

Our document ranking experiments address the following research questions:

我们的文档排序实验解决了以下研究问题：

- RQ5: Can our pseudo-relevance feedback mechanism enhance over the retrieval effectiveness of dense retrieval models, i.e.,can the ColBERT-PRF model outperform ColBERT, ANCE and ANCE-PRF dense retrieval models for document retrieval task?

- RQ5：我们的伪相关反馈机制能否提高密集检索模型的检索效果，即ColBERT-PRF模型在文档检索任务中能否优于ColBERT、ANCE和ANCE-PRF密集检索模型？

- RQ6: How does ColBERT-PRF compare to other existing baseline and state-of-the-art approaches for document retrieval task, namely:

- RQ6：在文档检索任务中，ColBERT-PRF与其他现有的基线方法和最先进的方法相比表现如何，即：

(a) lexical (sparse) baselines, including using PRF,

(a) 词法（稀疏）基线，包括使用PRF（伪相关反馈）

(b) BERT-QE Reranking models, and

(b) BERT-QE重排序模型

(c) embedding based query expansion models, namely the three variants of the CEQE model: CEQE-Max, CEQE-Centroid, and CEQE-Mul?

(c) 基于嵌入的查询扩展模型，即CEQE模型的三种变体：CEQE-Max、CEQE-Centroid和CEQE-Mul？

### 6.2 Experimental Setup

### 6.2 实验设置

6.2.1 Dataset & Measures. In this section, we evaluate our ColBERT-PRF on document ranking task using MSMARCO document and Robust04 document datasets. The MSMARCO training dataset contains $\sim  {3.2}\mathrm{M}$ documents along with ${367}\mathrm{\;K}$ training queries,each with 1-2 labelled relevant documents. The Robust04 collection contains 528K newswire articles from TREC disks 4 & 5. To test retrieval effectiveness of ColBERT-PRF model, we use the 43 test queries from the TREC Deep Learning Track 2019 and 45 test queries from the TREC Deep Learning Track 2020 with an average of 153.4 and 39.26 relevant documents per query, respectively. In addition, we also conduct the evaluation using 250 title-only and description-only query sets from TREC Robust04 document ranking task.

6.2.1 数据集与指标。在本节中，我们使用MSMARCO文档和Robust04文档数据集评估我们的ColBERT-PRF在文档排序任务上的性能。MSMARCO训练数据集包含$\sim  {3.2}\mathrm{M}$个文档以及${367}\mathrm{\;K}$个训练查询，每个查询有1 - 2个标注的相关文档。Robust04文档集合包含来自TREC磁盘4和5的52.8万篇新闻文章。为了测试ColBERT-PRF模型的检索效果，我们使用TREC深度学习赛道2019年的43个测试查询和TREC深度学习赛道2020年的45个测试查询，每个查询平均分别有153.4个和39.26个相关文档。此外，我们还使用TREC Robust04文档排序任务中的250个仅标题和仅描述的查询集进行评估。

We report the following metrics for MSMARCO document ranking tasks, namely the normalised discounted cumulative gain (NDCG) calculated at rank 10 , Mean Average Precision (MAP) at rank 1000 as well as Recall calculated at ranks 100 and 1000. For the Robust04 experiments, we use the same metrics used for passage ranking tasks in Section 5.2. For significance testing,we use the paired t-test $\left( {p < {0.05}}\right)$ and apply the Holm-Bonferroni multiple testing correction.

我们报告了MSMARCO文档排序任务的以下指标，即在排名第10位计算的归一化折损累积增益（NDCG）、在排名第1000位的平均准确率均值（MAP）以及在排名第100位和第1000位计算的召回率。对于Robust04实验，我们使用第5.2节中段落排序任务使用的相同指标。为了进行显著性检验，我们使用配对t检验$\left( {p < {0.05}}\right)$并应用霍尔姆 - 邦费罗尼多重检验校正。

6.2.2 Implementation and Settings. As the length of documents in these corpora are too long to be fitted into the BERT [11] model,and in particular our trained ColBERT model ${}^{13}$ (limited to 512 and 180 BERT WordPiece tokens, respectively), we split long documents into smaller passages and index the generated passages following [9]. In particular, when building the index for each document corpora, a sliding window of 150 tokens with a stride of 75 tokens is applied to split the documents into passages. All the passages are encoded into a FAISS index. At retrieval time, FAISS retrieves ${k}^{\prime } = {1000}$ document embeddings for every query embedding. The final score for each document is obtained by taking its highest ranking passage, a.k.a., its max passage.

6.2.2 实现与设置。由于这些语料库中的文档长度太长，无法直接输入到BERT [11]模型中，特别是我们训练的ColBERT模型${}^{13}$（分别限制为512和180个BERT词片标记），我们将长文档分割成较小的段落，并按照文献[9]的方法对生成的段落进行索引。具体来说，在为每个文档语料库构建索引时，应用一个步长为75个标记、窗口大小为150个标记的滑动窗口将文档分割成段落。所有段落都被编码到一个FAISS索引中。在检索时，FAISS为每个查询嵌入检索${k}^{\prime } = {1000}$个文档嵌入。每个文档的最终得分通过其排名最高的段落（即其最大段落）获得。

To ensure a fair comparison, we apply passaging for all other indices used in this section, including the Terrier inverted index,i.e.,the ANCE dense index. ${}^{14}$ Similarly,all PRF methods are applied on feedback passages, and max passage applied on the final ranking of passages.

为了确保公平比较，我们对本节中使用的所有其他索引应用段落分割，包括Terrier倒排索引，即ANCE密集索引。${}^{14}$同样，所有PRF方法都应用于反馈段落，并且在段落的最终排名中应用最大段落。

---

<!-- Footnote -->

${}^{13}$ It is a common practice to use models trained on the MSMARCO passage corpus [30] for document retrieval (e.g., [21,31]). ${}^{14}$ While this is necessary for a fair comparison,it results in a small degradation in effectiveness for the sparse baselines - this has also been observed by the authors of Anserini - see https://github.com/castorini/anserini/blob/master/src/main/ python/passage_retrieval/example/robust04.md.

${}^{13}$ 在文档检索中使用在MS MARCO段落语料库（MS MARCO passage corpus）[30]上训练的模型是一种常见做法（例如[21,31]）。${}^{14}$ 虽然这对于公平比较是必要的，但它会导致稀疏基线的有效性略有下降——Anserini的作者也观察到了这一点——请参阅https://github.com/castorini/anserini/blob/master/src/main/python/passage_retrieval/example/robust04.md。

<!-- Footnote -->

---

Finally, we follow the same ColBERT-PRF implementation as introduced in Section 5.2. For query expansion settings, we follow the default settings for passage ranking task in Section 5.3, which is 10 expansion terms obtained from three feedback passages ${}^{15}$ and $K = {24}$ clusters.

最后，我们采用与5.2节中介绍的相同的ColBERT - PRF实现。对于查询扩展设置，我们遵循5.3节中段落排序任务的默认设置，即从三个反馈段落${}^{15}$和$K = {24}$聚类中获取10个扩展词。

6.2.3 Baselines. To test the effectiveness of our ColBERT-PRF model on document ranking task, we compare with the all the baseline models we used for passage ranking task except the Neural Augmentation Approaches, due to the high GPU indexing time require for performing the doc2query and DeepCT processing for these large document corpora.

6.2.3 基线模型。为了测试我们的ColBERT - PRF模型在文档排序任务上的有效性，由于对这些大型文档语料库执行doc2query和DeepCT处理需要较长的GPU索引时间，我们将其与用于段落排序任务的所有基线模型进行比较，但不包括神经增强方法。

### 6.3 Document Ranking Results

### 6.3 文档排序结果

In this section, we further investigate the effectiveness of our proposed CoIBERT-PRF for document ranking task. Tables 5 and 6 present the performance of ColBERT-PRF models as well as the baselines on the MSMARCO document dataset and the Robust04 dataset, respectively.

在本节中，我们进一步研究我们提出的CoIBERT - PRF在文档排序任务上的有效性。表5和表6分别展示了ColBERT - PRF模型以及基线模型在MS MARCO文档数据集和Robust04数据集上的性能。

6.3.1 Results for RQ5. Similar to the passage retrieval task, in this section we validate the effectiveness of the pseudo-relevance feedback technique for the ColBERT dense retrieval model on the document retrieval task. On analysing Table 5, we found that both ColBERT-PRF Ranker and ReRanker models significantly outperform both the single representation dense retrieval, namely ANCE, and the multiple representation dense retrieval model, namely ColBERT E2E, in terms of MAP and Recall on both TREC 2019 and TREC 2020 query sets. In particular, the application of ColBERT-PRF leads to up to 21% and 14% improvements over ColBERT E2E in terms of MAP for TREC 2019 and TREC 2020 query sets, respectively.

6.3.1 RQ5的结果。与段落检索任务类似，在本节中，我们验证了伪相关反馈技术在ColBERT密集检索模型的文档检索任务中的有效性。通过分析表5，我们发现，在TREC 2019和TREC 2020查询集的平均准确率均值（MAP）和召回率方面，ColBERT - PRF排序器和重排序器模型都显著优于单表示密集检索模型（即ANCE）和多表示密集检索模型（即ColBERT E2E）。特别是，在TREC 2019和TREC 2020查询集的MAP方面，ColBERT - PRF的应用分别比ColBERT E2E提高了多达21%和14%。

Indeed, ColBERT-PRF outperforms all document retrieval runs to the TREC 2019 Deep Learning track,exceeding the highest observed MAP by 23% in terms of MAP. Similarly,on the TREC 2020 query set, the MAP observed is markedly above that attained by the second-ranked group on the leaderboard [7]. ${}^{16}$ In terms of NDCG@10,ColBERT-PRF outperforms over both the ANCE and ColBERT E2E models on both MSMARCO query sets. Moreover, both the ColBERT-PRF Ranker and ReRanker models significantly outperform the ColBERT and ANCE models w.r.t. Recall@100, indicating the effectiveness of the ColBERT-PRF refined query representations.

实际上，ColBERT - PRF在TREC 2019深度学习赛道的所有文档检索运行中表现更优，在MAP方面比观察到的最高MAP高出23%。同样，在TREC 2020查询集上，观察到的MAP明显高于排行榜上排名第二的组所达到的MAP [7]。${}^{16}$ 在归一化折损累积增益（NDCG@10）方面，ColBERT - PRF在两个MS MARCO查询集上都优于ANCE和ColBERT E2E模型。此外，ColBERT - PRF排序器和重排序器模型在召回率（Recall@100）方面都显著优于ColBERT和ANCE模型，这表明ColBERT - PRF改进后的查询表示是有效的。

Similarly, when comparing the performances of ColBERT-PRF with the dense retrieval models without pseudo-relevance feedback on Robust04 in Table 6, we note that both ColBERT-PRF Ranker and ReRanker models are markedly improved over the ANCE and ColBERT E2E models on MAP, NDCG@10, and Recall on both title-only and description-only type of queries. Overall, between the Ranker and ReRanker ColBERT-PRF models, we find that ColBERT-PRF Ranker is more effective than ColBERT-PRF ReRanker, likely due to its increased Recall, consistent with those obtained from the passage ranking task (Section 5). Thus, in response to RQ5, we conclude that our ColBERT-PRF is effective at improving ColBERT E2E on document ranking tasks, similar to the improvements observed in Section 5.

同样，当在表6中比较ColBERT - PRF与没有伪相关反馈的密集检索模型在Robust04上的性能时，我们注意到，在仅标题和仅描述类型的查询的MAP、NDCG@10和召回率方面，ColBERT - PRF排序器和重排序器模型都明显优于ANCE和ColBERT E2E模型。总体而言，在ColBERT - PRF排序器和重排序器模型之间，我们发现ColBERT - PRF排序器比ColBERT - PRF重排序器更有效，这可能是由于其召回率更高，这与在段落排序任务（第5节）中得到的结果一致。因此，针对RQ5，我们得出结论，我们的ColBERT - PRF在文档排序任务上能有效改进ColBERT E2E，与第5节中观察到的改进情况类似。

6.3.2 Results for RQ6. In the following, we compare the effectiveness of the ColBERT-PRF model with various baselines. From Table 5, we find that ColBERT-PRF instantiated as the Ranker model significantly improves over the BM25-based lexical retrieval baselines and the ColBERT E2E with BERT-QE as the reranker, as well as all the CEQE variants models in terms of the NDCG@10 and Recall@100 metrics on the TREC 2019 query set. In addition, for the TREC 2020 query set, ColBERT-PRF significantly improves over all the baselines except those with BERT-based neural reranking models, namely BERT, ColBERT and BERT-QE, in terms of the MAP and Recall@100 metrics.

6.3.2 RQ6的结果。接下来，我们将ColBERT - PRF模型与各种基线模型的有效性进行比较。从表5中我们发现，作为排序器模型实例化的ColBERT - PRF在TREC 2019查询集的NDCG@10和Recall@100指标方面，显著优于基于BM25的词法检索基线模型、以BERT - QE作为重排序器的ColBERT E2E模型以及所有CEQE变体模型。此外，对于TREC 2020查询集，在MAP和Recall@100指标方面，ColBERT - PRF显著优于除基于BERT的神经重排序模型（即BERT、ColBERT和BERT - QE）之外的所有基线模型。

---

<!-- Footnote -->

${}^{15}$ We also tried filtering passages from the same document before applying PRF. We observed no significant improvements across multiple measures.

${}^{15}$ 我们还尝试在应用伪相关反馈（PRF）之前过滤来自同一文档的段落。我们发现多项指标均无显著改善。

${}^{16}$ The first ranked group used expensive document expansion techniques.

${}^{16}$ 排名第一的小组使用了昂贵的文档扩展技术。

<!-- Footnote -->

---

<!-- Media -->

Table 5. Results for the MSMARCO Document Corpus

表5. MSMARCO文档语料库的结果

<table><tr><td rowspan="2"/><td colspan="4">TREC 2019 (43 queries)</td><td colspan="4">TREC 2020 (45 queries)</td></tr><tr><td>MAP</td><td>NDCG@10</td><td>Recall@100</td><td>Recall@1000</td><td>MAP</td><td>NDCG@10</td><td>Recall@100</td><td>Recall@1000</td></tr><tr><td colspan="9">Lexical Retrieval Approaches</td></tr><tr><td>BM25 (a)</td><td>0.3145</td><td>0.5048</td><td>0.3891</td><td>0.6975</td><td>0.3650</td><td>0.4709</td><td>0.6095</td><td>0.8143</td></tr><tr><td>BM25+BERT (b)</td><td>0.3797</td><td>0.6279</td><td>0.4363</td><td>0.6977</td><td>0.4387</td><td>0.5993</td><td>0.6646</td><td>0.8147</td></tr><tr><td>BM25+ColBERT (c)</td><td>0.3862</td><td>0.6503</td><td>0.4378</td><td>0.6970</td><td>0.4390</td><td>0.6144</td><td>0.6580</td><td>0.8155</td></tr><tr><td>BM25+RM3 (d)</td><td>0.3650</td><td>0.5411</td><td>0.4203</td><td>0.7304</td><td>0.3822</td><td>0.4770</td><td>0.6380</td><td>0.8311</td></tr><tr><td>BM25+RM3+BERT (e)</td><td>0.3973</td><td>0.6330</td><td>0.4466</td><td>0.7304</td><td>0.4470</td><td>0.5981</td><td>0.6646</td><td>0.8305</td></tr><tr><td>BM25+RM3+ColBERT (f)</td><td>0.4083</td><td>0.6633</td><td>0.4506</td><td>0.7300</td><td>0.4467</td><td>0.6074</td><td>0.6580</td><td>0.8305</td></tr><tr><td colspan="9">Dense Retrieval Models</td></tr><tr><td>ANCE (g)</td><td>0.2708</td><td>0.6468</td><td>0.3443</td><td>0.5349</td><td>0.4050</td><td>0.6256</td><td>0.5682</td><td>0.7197</td></tr><tr><td>ColBERT E2E (h)</td><td>0.3195</td><td>0.6342</td><td>0.3880</td><td>0.5642</td><td>0.4290</td><td>0.6113</td><td>0.6351</td><td>0.7951</td></tr><tr><td colspan="9">BERT-OE Reranking Models</td></tr><tr><td>BM25 + RM3 + ColBERT + BERT-OE (i)</td><td>0.4340</td><td>0.6850</td><td>0.4626</td><td>0.7298</td><td>0.4728</td><td>0.6268</td><td>0.6848</td><td>0.8310</td></tr><tr><td>ColBERT E2E + BERT-OE (j)</td><td>0.3358</td><td>0.6668</td><td>0.3953</td><td>0.5642</td><td>0.4478</td><td>0.6244</td><td>0.7141</td><td>0.7951</td></tr><tr><td colspan="9">Embedding based Ouery Expansion</td></tr><tr><td>CEOE-Max (k)</td><td>0.3778</td><td>0.5176</td><td>0.4313</td><td>0.7462</td><td>0.3956</td><td>0.4729</td><td>0.6546</td><td>0.8410</td></tr><tr><td>CEOE-Centroid (l)</td><td>0.3765</td><td>0.5103</td><td>0.4312</td><td>0.7432</td><td>0.3968</td><td>0.4746</td><td>0.6540</td><td>0.8390</td></tr><tr><td>CEOE-Mul (m)</td><td>0.3680</td><td>0.4959</td><td>0.4207</td><td>0.7360</td><td>0.3937</td><td>0.4809</td><td>0.6467</td><td>0.8351</td></tr><tr><td colspan="9">ColBERT-PRF Models</td></tr><tr><td>ColBERT-PRF Ranker $\left( {\beta  = 1}\right)$</td><td>${0.3851}^{ghj}$</td><td>${0.6681}^{{ad}{klm}}$</td><td>${0.4467}^{aghj}$</td><td>${0.6252}^{g}$</td><td>${\mathbf{{0.4885}}}^{adghklm}$</td><td>${0.6146}^{adklm}$</td><td>${0.7120}^{acdfghm}$</td><td>${0.8128}^{g}$</td></tr><tr><td>ColBERT-PRF ReRanker $\left( {\beta  = 1}\right)$</td><td>${0.3473}^{gh}$</td><td>${0.6688}^{ad}{}^{al}{}^{pl}{}^{em}$</td><td>${0.4283}^{ghj}$</td><td>0.5459</td><td>${0.4739}^{adgklm}$</td><td>${0.6171}^{adklm}$</td><td>${0.6933}^{agh}$</td><td>${0.7782}^{g}$</td></tr></table>

<table><tbody><tr><td rowspan="2"></td><td colspan="4">2019年文本检索会议（TREC 2019，43个查询）</td><td colspan="4">2020年文本检索会议（TREC 2020，45个查询）</td></tr><tr><td>平均准确率均值（MAP）</td><td>前10名归一化折损累积增益（NDCG@10）</td><td>前100名召回率（Recall@100）</td><td>前1000名召回率（Recall@1000）</td><td>平均准确率均值（MAP）</td><td>前10名归一化折损累积增益（NDCG@10）</td><td>前100名召回率（Recall@100）</td><td>前1000名召回率（Recall@1000）</td></tr><tr><td colspan="9">词法检索方法</td></tr><tr><td>二元独立模型25（BM25 (a)）</td><td>0.3145</td><td>0.5048</td><td>0.3891</td><td>0.6975</td><td>0.3650</td><td>0.4709</td><td>0.6095</td><td>0.8143</td></tr><tr><td>二元独立模型25+双向编码器表征变换器（BM25+BERT (b)）</td><td>0.3797</td><td>0.6279</td><td>0.4363</td><td>0.6977</td><td>0.4387</td><td>0.5993</td><td>0.6646</td><td>0.8147</td></tr><tr><td>二元独立模型25+ColBERT模型（BM25+ColBERT (c)）</td><td>0.3862</td><td>0.6503</td><td>0.4378</td><td>0.6970</td><td>0.4390</td><td>0.6144</td><td>0.6580</td><td>0.8155</td></tr><tr><td>二元独立模型25+相关反馈模型3（BM25+RM3 (d)）</td><td>0.3650</td><td>0.5411</td><td>0.4203</td><td>0.7304</td><td>0.3822</td><td>0.4770</td><td>0.6380</td><td>0.8311</td></tr><tr><td>二元独立模型25+相关反馈模型3+双向编码器表征变换器（BM25+RM3+BERT (e)）</td><td>0.3973</td><td>0.6330</td><td>0.4466</td><td>0.7304</td><td>0.4470</td><td>0.5981</td><td>0.6646</td><td>0.8305</td></tr><tr><td>二元独立模型25+相关反馈模型3+ColBERT模型（BM25+RM3+ColBERT (f)）</td><td>0.4083</td><td>0.6633</td><td>0.4506</td><td>0.7300</td><td>0.4467</td><td>0.6074</td><td>0.6580</td><td>0.8305</td></tr><tr><td colspan="9">密集检索模型</td></tr><tr><td>自适应神经上下文编码器（ANCE (g)）</td><td>0.2708</td><td>0.6468</td><td>0.3443</td><td>0.5349</td><td>0.4050</td><td>0.6256</td><td>0.5682</td><td>0.7197</td></tr><tr><td>端到端ColBERT模型（ColBERT E2E (h)）</td><td>0.3195</td><td>0.6342</td><td>0.3880</td><td>0.5642</td><td>0.4290</td><td>0.6113</td><td>0.6351</td><td>0.7951</td></tr><tr><td colspan="9">基于双向编码器表征变换器的开放域实体重排序模型（BERT - OE重排序模型）</td></tr><tr><td>二元独立模型25 + 相关反馈模型3 + ColBERT模型 + 基于双向编码器表征变换器的开放域实体（BM25 + RM3 + ColBERT + BERT - OE (i)）</td><td>0.4340</td><td>0.6850</td><td>0.4626</td><td>0.7298</td><td>0.4728</td><td>0.6268</td><td>0.6848</td><td>0.8310</td></tr><tr><td>端到端ColBERT模型 + 基于双向编码器表征变换器的开放域实体（ColBERT E2E + BERT - OE (j)）</td><td>0.3358</td><td>0.6668</td><td>0.3953</td><td>0.5642</td><td>0.4478</td><td>0.6244</td><td>0.7141</td><td>0.7951</td></tr><tr><td colspan="9">基于嵌入的查询扩展</td></tr><tr><td>基于嵌入的查询扩展 - 最大值（CEOE - Max (k)）</td><td>0.3778</td><td>0.5176</td><td>0.4313</td><td>0.7462</td><td>0.3956</td><td>0.4729</td><td>0.6546</td><td>0.8410</td></tr><tr><td>基于嵌入的查询扩展 - 质心（CEOE - Centroid (l)）</td><td>0.3765</td><td>0.5103</td><td>0.4312</td><td>0.7432</td><td>0.3968</td><td>0.4746</td><td>0.6540</td><td>0.8390</td></tr><tr><td>基于嵌入的查询扩展 - 乘法（CEOE - Mul (m)）</td><td>0.3680</td><td>0.4959</td><td>0.4207</td><td>0.7360</td><td>0.3937</td><td>0.4809</td><td>0.6467</td><td>0.8351</td></tr><tr><td colspan="9">ColBERT伪相关反馈模型</td></tr><tr><td>ColBERT伪相关反馈排序器 $\left( {\beta  = 1}\right)$</td><td>${0.3851}^{ghj}$</td><td>${0.6681}^{{ad}{klm}}$</td><td>${0.4467}^{aghj}$</td><td>${0.6252}^{g}$</td><td>${\mathbf{{0.4885}}}^{adghklm}$</td><td>${0.6146}^{adklm}$</td><td>${0.7120}^{acdfghm}$</td><td>${0.8128}^{g}$</td></tr><tr><td>ColBERT伪相关反馈重排序器 $\left( {\beta  = 1}\right)$</td><td>${0.3473}^{gh}$</td><td>${0.6688}^{ad}{}^{al}{}^{pl}{}^{em}$</td><td>${0.4283}^{ghj}$</td><td>0.5459</td><td>${0.4739}^{adgklm}$</td><td>${0.6171}^{adklm}$</td><td>${0.6933}^{agh}$</td><td>${0.7782}^{g}$</td></tr></tbody></table>

Comparison with baselines. Superscripts a...p denote significant improvements over the indicated baseline model(s). The highest value in each column is boldfaced.

与基线模型的比较。上标a...p表示相较于指定的基线模型有显著改进。每列中的最高值用粗体显示。

Table 6. Results for the Robust Corpus

表6. 鲁棒语料库的结果

<table><tr><td rowspan="2"/><td colspan="4">Robust title (250 queries)</td><td colspan="4">Robust description (250 queries)</td></tr><tr><td>MAP</td><td>NDCG@10</td><td>MRR@10</td><td>Recall</td><td>MAP</td><td>NDCG@10</td><td>MRR@10</td><td>Recall</td></tr><tr><td colspan="9">Lexical Retrieval Approaches</td></tr><tr><td>BM25 (a)</td><td>0.2319</td><td>0.4163</td><td>0.6330</td><td>0.6758</td><td>0.2193</td><td>0.3966</td><td>0.6570</td><td>0.6584</td></tr><tr><td>BM25+BERT (b)</td><td>0.2550</td><td>0.4820</td><td>0.7290</td><td>0.6819</td><td>0.2723</td><td>0.4709</td><td>0.7293</td><td>0.6721</td></tr><tr><td>BM25+ColBERT (c)</td><td>0.2770</td><td>0.4753</td><td>0.7307</td><td>0.6821</td><td>0.2658</td><td>0.4728</td><td>0.7349</td><td>0.6684</td></tr><tr><td>BM25+RM3 (d)</td><td>0.2542</td><td>0.4244</td><td>0.6139</td><td>0.7007</td><td>0.2619</td><td>0.4182</td><td>0.6277</td><td>0.7008</td></tr><tr><td>BM25+RM3+BERT (e)</td><td>0.2884</td><td>0.4839</td><td>0.7343</td><td>0.7037</td><td>0.2814</td><td>0.4708</td><td>0.7251</td><td>0.7081</td></tr><tr><td>BM25+RM3+ColBERT (f)</td><td>0.2840</td><td>0.4758</td><td>0.7277</td><td>0.7058</td><td>0.2766</td><td>0.4739</td><td>0.7419</td><td>0.7068</td></tr><tr><td colspan="9">Dense Retrieval Models</td></tr><tr><td>ANCE (g)</td><td>0.1605</td><td>0.3713</td><td>0.6096</td><td>0.5410</td><td>0.1919</td><td>0.4242</td><td>0.7002</td><td>0.5794</td></tr><tr><td>ColBERT E2E (h)</td><td>0.2327</td><td>0.4446</td><td>0.7011</td><td>0.6076</td><td>0.2175</td><td>0.4352</td><td>0.6853</td><td>0.6054</td></tr><tr><td colspan="9">BERT-OE Reranking Models</td></tr><tr><td>BM25 + RM3 + ColBERT + BERT-OE (i)</td><td>0.2762</td><td>0.4407</td><td>0.6302</td><td>0.7072</td><td>0.2926</td><td>0.4857</td><td>0.7369</td><td>0.7076</td></tr><tr><td>ColBERT E2E + BERT-OE (j)</td><td>0.2395</td><td>0.4523</td><td>0.6973</td><td>0.6078</td><td>0.2289</td><td>0.4468</td><td>0.6904</td><td>0.6055</td></tr><tr><td colspan="9">Embedding based Query Expansion</td></tr><tr><td>CEQE-Max (l)</td><td>0.2829</td><td>0.4318</td><td>0.6334</td><td>0.7494</td><td>0.2745</td><td>0.4224</td><td>0.6461</td><td>0.7232</td></tr><tr><td>CEOE-Centroid (m)</td><td>0.2818</td><td>0.4299</td><td>0.6305</td><td>0.7457</td><td>0.2746</td><td>0.4217</td><td>0.6475</td><td>0.7278</td></tr><tr><td>CEOE-Mul (n)</td><td>0.2764</td><td>0.4267</td><td>0.6225</td><td>0.7375</td><td>0.2672</td><td>0.4076</td><td>0.6146</td><td>0.7256</td></tr><tr><td colspan="9">ColBERT-PRF Models</td></tr><tr><td>ColBERT-PRF Ranker $\left( {\beta  = 1}\right)$</td><td>${0.2715}^{adghj}$</td><td>${0.4670}^{adgh}$</td><td>${0.6836}^{dglmn}$</td><td>${0.6476}^{ghj}$</td><td>${0.2627}^{aghj}$</td><td>${0.4605}^{ah}$</td><td>0.6678</td><td>${0.6347}^{ghj}$</td></tr><tr><td>ColBERT-PRF ReRanker $\left( {\beta  = 1}\right)$</td><td>${0.2642}^{adghj}$</td><td>${0.4682}^{adgh}$</td><td>${0.6837}^{dg}$</td><td>${0.6158}^{g}$</td><td>${0.2592}^{aghj}$</td><td>${0.4624}^{ah}$</td><td>0.6681</td><td>${0.6289}^{ghj}$</td></tr></table>

<table><tbody><tr><td rowspan="2"></td><td colspan="4">稳健标题（250次查询）</td><td colspan="4">稳健描述（250次查询）</td></tr><tr><td>平均准确率均值（Mean Average Precision，MAP）</td><td>前10名归一化折损累积增益（Normalized Discounted Cumulative Gain at 10，NDCG@10）</td><td>前10名平均倒数排名（Mean Reciprocal Rank at 10，MRR@10）</td><td>召回率</td><td>平均准确率均值（Mean Average Precision，MAP）</td><td>前10名归一化折损累积增益（Normalized Discounted Cumulative Gain at 10，NDCG@10）</td><td>前10名平均倒数排名（Mean Reciprocal Rank at 10，MRR@10）</td><td>召回率</td></tr><tr><td colspan="9">词法检索方法</td></tr><tr><td>二元独立模型25（BM25）（a）</td><td>0.2319</td><td>0.4163</td><td>0.6330</td><td>0.6758</td><td>0.2193</td><td>0.3966</td><td>0.6570</td><td>0.6584</td></tr><tr><td>二元独立模型25+双向编码器表征变换器（BM25+BERT）（b）</td><td>0.2550</td><td>0.4820</td><td>0.7290</td><td>0.6819</td><td>0.2723</td><td>0.4709</td><td>0.7293</td><td>0.6721</td></tr><tr><td>二元独立模型25+ColBERT（BM25+ColBERT）（c）</td><td>0.2770</td><td>0.4753</td><td>0.7307</td><td>0.6821</td><td>0.2658</td><td>0.4728</td><td>0.7349</td><td>0.6684</td></tr><tr><td>二元独立模型25+相关反馈模型3（BM25+RM3）（d）</td><td>0.2542</td><td>0.4244</td><td>0.6139</td><td>0.7007</td><td>0.2619</td><td>0.4182</td><td>0.6277</td><td>0.7008</td></tr><tr><td>二元独立模型25+相关反馈模型3+双向编码器表征变换器（BM25+RM3+BERT）（e）</td><td>0.2884</td><td>0.4839</td><td>0.7343</td><td>0.7037</td><td>0.2814</td><td>0.4708</td><td>0.7251</td><td>0.7081</td></tr><tr><td>二元独立模型25+相关反馈模型3+ColBERT（BM25+RM3+ColBERT）（f）</td><td>0.2840</td><td>0.4758</td><td>0.7277</td><td>0.7058</td><td>0.2766</td><td>0.4739</td><td>0.7419</td><td>0.7068</td></tr><tr><td colspan="9">密集检索模型</td></tr><tr><td>自适应神经上下文编码器（ANCE）（g）</td><td>0.1605</td><td>0.3713</td><td>0.6096</td><td>0.5410</td><td>0.1919</td><td>0.4242</td><td>0.7002</td><td>0.5794</td></tr><tr><td>端到端ColBERT（ColBERT E2E）（h）</td><td>0.2327</td><td>0.4446</td><td>0.7011</td><td>0.6076</td><td>0.2175</td><td>0.4352</td><td>0.6853</td><td>0.6054</td></tr><tr><td colspan="9">双向编码器表征变换器开放域实体（BERT-OE）重排序模型</td></tr><tr><td>二元独立模型25 + 相关反馈模型3 + ColBERT + 双向编码器表征变换器开放域实体（BM25 + RM3 + ColBERT + BERT-OE）（i）</td><td>0.2762</td><td>0.4407</td><td>0.6302</td><td>0.7072</td><td>0.2926</td><td>0.4857</td><td>0.7369</td><td>0.7076</td></tr><tr><td>端到端ColBERT + 双向编码器表征变换器开放域实体（ColBERT E2E + BERT-OE）（j）</td><td>0.2395</td><td>0.4523</td><td>0.6973</td><td>0.6078</td><td>0.2289</td><td>0.4468</td><td>0.6904</td><td>0.6055</td></tr><tr><td colspan="9">基于嵌入的查询扩展</td></tr><tr><td>基于上下文嵌入的查询扩展-最大值（CEQE-Max）（l）</td><td>0.2829</td><td>0.4318</td><td>0.6334</td><td>0.7494</td><td>0.2745</td><td>0.4224</td><td>0.6461</td><td>0.7232</td></tr><tr><td>基于上下文嵌入的查询扩展-质心（CEOE-Centroid）（m）</td><td>0.2818</td><td>0.4299</td><td>0.6305</td><td>0.7457</td><td>0.2746</td><td>0.4217</td><td>0.6475</td><td>0.7278</td></tr><tr><td>基于上下文嵌入的查询扩展-乘法（CEOE-Mul）（n）</td><td>0.2764</td><td>0.4267</td><td>0.6225</td><td>0.7375</td><td>0.2672</td><td>0.4076</td><td>0.6146</td><td>0.7256</td></tr><tr><td colspan="9">ColBERT伪相关反馈（PRF）模型</td></tr><tr><td>ColBERT伪相关反馈排序器 $\left( {\beta  = 1}\right)$</td><td>${0.2715}^{adghj}$</td><td>${0.4670}^{adgh}$</td><td>${0.6836}^{dglmn}$</td><td>${0.6476}^{ghj}$</td><td>${0.2627}^{aghj}$</td><td>${0.4605}^{ah}$</td><td>0.6678</td><td>${0.6347}^{ghj}$</td></tr><tr><td>ColBERT伪相关反馈重排序器 $\left( {\beta  = 1}\right)$</td><td>${0.2642}^{adghj}$</td><td>${0.4682}^{adgh}$</td><td>${0.6837}^{dg}$</td><td>${0.6158}^{g}$</td><td>${0.2592}^{aghj}$</td><td>${0.4624}^{ah}$</td><td>0.6681</td><td>${0.6289}^{ghj}$</td></tr></tbody></table>

Comparison with baselines. Superscripts a...p denote significant improvements over the indicated baseline model(s). The highest value in each column is boldfaced.

与基线模型的比较。上标a...p表示相较于指定的基线模型有显著改进。每列中的最高值用粗体表示。

<!-- Media -->

Now let's analyse the performance of ColBERT-PRF models on Robust04 query sets. From Table 6, we observe that ColBERT-PRF models significantly outperforms the BM25 on both query sets and markedly outperforms over BM25 + RM3 on title-only queries. In addition, ColBERT-PRF show the similar performance with CEQE models in terms of MAP but exhibit markedly improvements in terms of NDCG@10 and MRR@10.Moreover, when comparing with the models with neural rerankers, both ColBERT-PRF Ranker and ReRanker models significantly outperform the ColBERT E2E + BERT-QE baseline and exhibits comparable performance than the other neural reranker models. However, we argue that the limited performance of ColBERT-PRF compared with the BERT-based reranking models for the Robust04 query sets comes from the two following aspects: firstly, we used a zero-shot setting of ColBERT model for the document ranking tasks, in that the ColBERT model was not trained on the larger document datasets; second, we didn't perform further parameter tuning for ColBERT-PRF on the document ranking task. Thus, in response to RQ6, we find that ColBERT-PRF is more effective than most of the baseline models and comparable to the BERT based neural reranking models.

现在让我们分析ColBERT - PRF模型在Robust04查询集上的性能。从表6中我们可以观察到，ColBERT - PRF模型在两个查询集上都显著优于BM25，并且在仅标题查询上明显优于BM25 + RM3。此外，ColBERT - PRF在平均准确率均值（MAP，Mean Average Precision）方面与CEQE模型表现相似，但在归一化折损累积增益（NDCG@10，Normalized Discounted Cumulative Gain at 10）和平均倒数排名（MRR@10，Mean Reciprocal Rank at 10）方面有显著改进。此外，与使用神经重排器的模型相比，ColBERT - PRF排序器和重排器模型都显著优于ColBERT端到端（E2E，End - to - End）+ BERT查询扩展（QE，Query Expansion）基线模型，并且与其他神经重排器模型表现相当。然而，我们认为与基于BERT的重排模型相比，ColBERT - PRF在Robust04查询集上性能有限，主要来自以下两个方面：首先，我们在文档排名任务中使用了ColBERT模型的零样本设置，即ColBERT模型没有在更大的文档数据集上进行训练；其次，我们没有在文档排名任务上对ColBERT - PRF进行进一步的参数调整。因此，针对研究问题6（RQ6），我们发现ColBERT - PRF比大多数基线模型更有效，并且与基于BERT的神经重排模型相当。

## 7 MEASURING THE INFORMATIVENESS OF EXPANSION EMBEDDINGS OF COLBERT-PRF

## 7 衡量ColBERT - PRF扩展嵌入的信息性

In this section, we investigate the effectiveness of the three variants of the ColBERT-PRF model using different techniques to measure the informativeness of the expansion embeddings. The strategies are detailed in Section 7.1. Accordingly, a research question is posed in Section 7.2, with a corresponding experimental setup. Finally, Section 7.3 presents the performance and analysis of the three ColBERT-PRF variants.

在本节中，我们使用不同的技术来衡量扩展嵌入的信息性，以此研究ColBERT - PRF模型的三种变体的有效性。具体策略在7.1节详细介绍。相应地，7.2节提出了一个研究问题，并给出了对应的实验设置。最后，7.3节展示了三种ColBERT - PRF变体的性能和分析结果。

### 7.1 Methodology

### 7.1 方法

In Section 4.2 we proposed to map each expansion embedding back to its most likely token, and use the IDF of that token to measure the importance $\sigma$ of each expansion embedding ${v}_{i}$ generated by ColBERT-PRF. This results in a weight, $\sigma \left( {v}_{i}\right)$ ,that is used in the expanded max-sim calculation (Equation (5)). Indeed, notions of document frequency or collection frequency are commonly used in PRF models to measure expansion terms [2]. The intuition behind this is that if a term appears more frequently in the feedback documents than in the whole corpus, the term is taken as an informative term. In contrast, terms that occur frequently in the corpus will not discriminate well relevant documents from other documents in the collection. In this section, we revisit the use of IDF in ColBERT-PRF, by additionally using collection frequency of the token, while also examining the corresponding embeddings of the tokens.

在4.2节中，我们提出将每个扩展嵌入映射回其最可能的词元，并使用该词元的逆文档频率（IDF，Inverse Document Frequency）来衡量ColBERT - PRF生成的每个扩展嵌入 ${v}_{i}$ 的重要性 $\sigma$。这会得到一个权重 $\sigma \left( {v}_{i}\right)$，用于扩展的最大相似度计算（公式（5））。实际上，文档频率或集合频率的概念在伪相关反馈（PRF，Pseudo - Relevant Feedback）模型中常用于衡量扩展词项 [2]。其背后的直觉是，如果一个词项在反馈文档中出现的频率比在整个语料库中更高，那么该词项就被视为一个有信息价值的词项。相反，在语料库中频繁出现的词项不能很好地区分相关文档和集合中的其他文档。在本节中，我们重新审视了IDF在ColBERT - PRF中的应用，除了使用词元的IDF，还考虑了词元的集合频率，同时检查词元的相应嵌入。

Indeed, in addition to the document frequency focus of IDF, the collection frequency is also useful to reflect the informativeness of a term within the whole collection, measured as follows:

实际上，除了IDF关注的文档频率外，集合频率也有助于反映一个词项在整个集合中的信息性，测量方法如下：

$$
{\sigma }_{ICTF}\left( t\right)  = \log \left( \frac{\left| D\right|  + 1}{{tf}\left( {t,D}\right)  + 1}\right)  \tag{7}
$$

where $\left| D\right|$ is the number of terms in the collection $D$ and ${tf}\left( {t,D}\right)$ is the number of occurrences of expansion term $t$ in the whole collection $D$ .

其中 $\left| D\right|$ 是集合 $D$ 中的词项数量，${tf}\left( {t,D}\right)$ 是扩展词项 $t$ 在整个集合 $D$ 中出现的次数。

However using either IDF or ICTF as expansion embedding weights does not consider the con-textualised nature of the embeddings - that different tokens can have distinct meanings, and these may be more or less useful for retrieval. Use of IDF or ICTF can mask such distinctions.

然而，使用IDF或逆集合词频（ICTF，Inverse Collection Term Frequency）作为扩展嵌入的权重并没有考虑到嵌入的上下文性质——不同的词元可能有不同的含义，这些含义对检索的有用程度可能不同。使用IDF或ICTF可能会掩盖这些差异。

Hence, we examine a further method based directly on the embedded representations. In particular, for each token, we examine all corresponding embeddings in the index, and determine how 'focused' these are - we postulate that a token with more focused embeddings will only have a single meaning (and therefore less polysemous), and hence is more likely to be a good expansion embedding. Specifically, we measure the Mean Cosine similarity (MCos) for the embeddings of each token compared to the mean of all those embeddings:

因此，我们研究了一种直接基于嵌入表示的进一步方法。具体来说，对于每个词元，我们检查索引中所有相应的嵌入，并确定它们的“聚焦”程度——我们假设具有更聚焦嵌入的词元只有单一含义（因此多义性较低），因此更有可能是一个好的扩展嵌入。具体而言，我们测量每个词元的嵌入与所有这些嵌入的均值之间的平均余弦相似度（MCos，Mean Cosine similarity）：

$$
{\sigma }_{MCos}\left( t\right)  = \frac{1}{{tf}\left( {t,D}\right) }\mathop{\sum }\limits_{{j = 1}}^{{{tf}\left( {t,D}\right) }}\cos \left( {\Upsilon ,{\phi }_{{c}_{j}}}\right)  \tag{8}
$$

where $\Upsilon$ is the element-wise average embedding of all embeddings in the index for token $t$ . MCos is intended to approximate the semantic coherence of the embeddings for a given token. The expansion embeddings of more coherent tokens are given a higher weight in ColBERT-PRF.

其中 $\Upsilon$ 是索引中词元 $t$ 的所有嵌入的逐元素平均嵌入。MCos旨在近似给定词元的嵌入的语义连贯性。在ColBERT - PRF中，连贯性更强的词元的扩展嵌入被赋予更高的权重。

<!-- Media -->

<!-- figureText: 0.54 (IDF) Ranker (ICTF) Ranker 0.50 (Mcos) Ranke 0.48 MAP 0.46 0.44 0.42 0.4 (b) MSMARCO passage TREC 2020 query set (IDF) Ranker (ICTF) Ranker (Mcos) Ranker 0.25 MAP 0.24 0.22 0.4 0.8 10 20 (d) Robust04 desciption query set. 0.52 MAP 0.50 (IDF) Ranker (ICTF) Ranker (Mcos) Ranker 0.48 0.46 0.44 0.4 0.8 10 a) MSMARCO passage TREC 2019 query set. (IDF) Ranker 0.27 (ICTF) Ranker (Mcos) Ranker 0.26 MAP 0.25 0.24 0.23 0.4 10 (c) Robust04 title query set. -->

<img src="https://cdn.noedgeai.com/0195b372-2fb6-7dbd-a6e2-deb0bb3cad1c_26.jpg?x=194&y=234&w=1171&h=856&r=0"/>

Fig. 10. Influence of different weighting methods. $\beta  = 0$ corresponds to the original ColBERT.

图10. 不同加权方法的影响。$\beta  = 0$ 对应于原始的ColBERT。

<!-- Media -->

### 7.2 Research Question & Experimental Setup

### 7.2 研究问题与实验设置

Our informativeness measurement experiments address the following research question:

我们的信息性测量实验旨在解决以下研究问题：

- RQ7: What is the impact of the effectiveness ColBERT-PRF using different informativeness of expansion embedding measurements methods, namely the IDF weighting method, ICTF weighting method, and the MCos weighting method?

- RQ7：使用不同信息性的扩展嵌入度量方法（即逆文档频率（IDF）加权方法、逆集合词频（ICTF）加权方法和最大余弦（MCos）加权方法）时，ColBERT-PRF的有效性会受到怎样的影响？

In our experiments addressing RQ7, while testing IDF, ICTF and MCos importance measures, we vary the parameter of ColBERT-PRF that controls the overall weight of the expansion embeddings, $\beta$ . We do not normalise the various importance measures ${\sigma }_{IDF}\left( t\right) ,{\sigma }_{ICTF}\left( t\right)$ and ${\sigma }_{MCos}\left( t\right)$ - their inherent differences in scales are addressed by varying $\beta$ .

在我们针对RQ7的实验中，在测试IDF、ICTF和MCos重要性度量时，我们改变了控制扩展嵌入整体权重的ColBERT-PRF参数$\beta$。我们不对各种重要性度量${\sigma }_{IDF}\left( t\right) ,{\sigma }_{ICTF}\left( t\right)$和${\sigma }_{MCos}\left( t\right)$进行归一化处理——它们在尺度上的固有差异通过改变$\beta$来解决。

Dataset: The query sets we used to demonstrate the effectiveness of the three variants of ColBERT-PRF proposed are the MSMARCO passage TREC 2019 and TREC 2020 passage query sets for passage retrieval task and the Robust title and description query sets for document retrieval task. Measures: Mean Average Precision (MAP) is used as the main metric.

数据集：我们用于证明所提出的ColBERT-PRF三种变体有效性的查询集包括用于段落检索任务的MSMARCO段落TREC 2019和TREC 2020段落查询集，以及用于文档检索任务的Robust标题和描述查询集。度量指标：平均准确率均值（MAP）被用作主要指标。

### 7.3 Results

### 7.3 结果

Figure 10 shows the impacts of the retrieval effectiveness of the different weighting methods while $\beta$ is varied,in terms of MAP,for ColBERT-PRF for both the MSMARCO passage ranking task and the Robust04 document ranking task. Specifically, for the passage ranking task, we measure the retrieval effectiveness on both the TREC 2019 and TREC 2020 passage ranking queries, and using title-only and description-only types of queries of Robust04.

图10展示了在改变$\beta$的情况下，不同加权方法对ColBERT-PRF在MSMARCO段落排序任务和Robust04文档排序任务中的检索有效性的影响，以MAP衡量。具体而言，对于段落排序任务，我们在TREC 2019和TREC 2020段落排序查询上测量检索有效性，并使用Robust04的仅标题和仅描述类型的查询。

On analysing the figure, we see that, for both TREC 2019 and TREC 2020 query sets, the peak MAP scores for all the three weighting methods are the same,approximately with MAP=0.54 and $\mathrm{{MAP}} = {0.51}$ ,respectively. In addition,according to the Figures ${10}\left( \mathrm{a}\right)$ and ${10}\left( \mathrm{\;b}\right)$ ,the overall trend for IDF and ICTF weighting methods are the same and both reaches the highest MAP score with $\beta  \in  \left\lbrack  {{0.4},{0.8}}\right\rbrack$ . When we compare with IDF and ICTF,we see that MCos with $\beta  \in  \left\lbrack  {{4.0},{6.0}}\right\rbrack$ exhibits the highest MAP performance. These trends allow us to draw the following observations: the lines for IDF and ICTF are very similar,varying only in terms of the $\beta$ value needed to obtain the highest MAP; In contrast, the MCos weighting method achieves a similar maximum MAP, but at a larger $\beta$ value - this is due to the lack of common normalisation. Indeed,as the maximum MAP values obtained are similar for IDF, ICTF and MCos, this suggests that the MCos is correlated with IDF, and that the statistical approaches are sufficient for measuring expansion embedding importance. A closer analysis of IDF and ICTF, as calculated on the BERT tokens, found that they exhibit a very high correlation (Spearman’s $\rho$ of $\sim  {1.00}$ on the MSMARCO passage corpus). This is indeed higher than the correlation observed on a traditional sparse Terrier inverted index (which uses a more conventional tokeniser) of 0.95 on the MSMARCO document index. The differences in correlations can be explained as follows: firstly, due to the use of WordPieces by the BERT tokeniser, which reduces the presence of long-tail tokens (which are tokenised to smaller WordPieces); secondly, passage corpora use smaller indexing units than document corpora, so it is less likely for terms to occur multiple times - this results in collection frequency being more correlated to document frequency.

通过分析该图，我们发现，对于TREC 2019和TREC 2020查询集，三种加权方法的MAP峰值得分相同，分别约为MAP = 0.54和$\mathrm{{MAP}} = {0.51}$。此外，根据图${10}\left( \mathrm{a}\right)$和${10}\left( \mathrm{\;b}\right)$，IDF和ICTF加权方法的总体趋势相同，并且在$\beta  \in  \left\lbrack  {{0.4},{0.8}}\right\rbrack$时都达到了最高的MAP得分。当我们将其与IDF和ICTF进行比较时，我们发现具有$\beta  \in  \left\lbrack  {{4.0},{6.0}}\right\rbrack$的MCos表现出最高的MAP性能。这些趋势使我们能够得出以下观察结果：IDF和ICTF的曲线非常相似，仅在获得最高MAP所需的$\beta$值方面有所不同；相比之下，MCos加权方法实现了相似的最大MAP，但在更大的$\beta$值处——这是由于缺乏共同的归一化处理。实际上，由于IDF、ICTF和MCos获得的最大MAP值相似，这表明MCos与IDF相关，并且统计方法足以衡量扩展嵌入的重要性。对基于BERT标记计算的IDF和ICTF进行更深入的分析发现，它们表现出非常高的相关性（在MSMARCO段落语料库上的斯皮尔曼相关系数$\rho$为$\sim  {1.00}$）。这确实高于在传统稀疏Terrier倒排索引（使用更传统的分词器）上观察到的相关性，在MSMARCO文档索引上为0.95。相关性的差异可以解释如下：首先，由于BERT分词器使用了词片（WordPieces），这减少了长尾标记的存在（这些标记被分词为更小的词片）；其次，段落语料库使用的索引单元比文档语料库小，因此术语多次出现的可能性较小——这导致集合频率与文档频率的相关性更高。

For the Robust04 queryset (Figures 10(c) and 10(d)), we see that while the peak MAP values for IDF and ICTF are again similar, the MCos weighting method gives lower MAP scores on the Robust04 title and description query sets. This suggests that using the coherence of a token's embeddings may not well indicate the utility of the expansion embedding. Indeed, some tokens with high embedding coherence could be stopword-like in nature. This motivates the continued use of IDF and ICTF for identifying important expansion embeddings.

对于Robust04查询集（图10(c)和10(d)），我们发现虽然IDF和ICTF的MAP峰值再次相似，但MCos加权方法在Robust04标题和描述查询集上的MAP得分较低。这表明使用标记嵌入的连贯性可能不能很好地表明扩展嵌入的实用性。实际上，一些具有高嵌入连贯性的标记本质上可能类似于停用词。这促使我们继续使用IDF和ICTF来识别重要的扩展嵌入。

Overall, to address RQ7, we find that the statistical information, based IDF and ICTF weighting methods, is more stable than the MCos weighting method for different retrieval tasks. Use of IDF and ICTF were shown to be equivalent, due to the higher correlation between document frequency and collection frequency on passage corpora.

总体而言，为了回答RQ7，我们发现基于统计信息的IDF和ICTF加权方法在不同的检索任务中比MCos加权方法更稳定。由于段落语料库中文档频率和集合频率之间的相关性较高，IDF和ICTF的使用被证明是等效的。

## 8 EFFICIENT VARIANTS OF COLBERT-PRF

## 8 高效的ColBERT-PRF变体

In Section 5.3, we noted the high mean response time of the ColBERT PRF approach. Higher response times are a feature of many PRF approaches, due to the need to analyse the contents of the feedback documents, and decide upon the expansion terms/embeddings. In this section, we investigate several efficient variants of our ColBERT-PRF model, by experimenting with different clustering approaches, as well as different retrieval configurations of ColBERT.

在第5.3节中，我们指出了ColBERT PRF方法的平均响应时间较长。由于需要分析反馈文档的内容并确定扩展词项/嵌入，许多PRF方法都存在响应时间较长的问题。在本节中，我们通过试验不同的聚类方法以及ColBERT的不同检索配置，研究了我们的ColBERT-PRF模型的几种高效变体。

In particular, we describe different variants in Section 8.1. Two research questions and the implementation setup are detailed in Section 8.2. Results and analysis are discussed in Section 8.3.

具体而言，我们在第8.1节中描述了不同的变体。两个研究问题和实现设置在第8.2节中详细介绍。结果和分析在第8.3节中讨论。

### 8.1 ColBERT-PRF Variants

### 8.1 ColBERT-PRF变体

The overall workflow of a ColBERT-PRF Ranker model can be described in five stages, as shown in Figure 1. These stages can be summarised as follows (for the ColBERT-PRF ReRanker model, the fourth stage ANN retrieval is omitted):

ColBERT-PRF排序器模型的整体工作流程可以分为五个阶段，如图1所示。这些阶段可以总结如下（对于ColBERT-PRF重排序器模型，省略了第四阶段的近似最近邻（ANN）检索）：

- Stage 1: First-pass FAISS ANN Retrieval

- 阶段1：首轮FAISS ANN检索

- Stage 2: First-pass exact ColBERT MaxSim re-ranking

- 阶段2：首轮精确ColBERT最大相似度重排序

- Stage 3: Clustering of Feedback Documents and Expansion Embedding Weighting

- 阶段3：反馈文档聚类和扩展嵌入加权

- Stage 4: Second-pass FAISS ANN Retrieval

- 阶段4：次轮FAISS ANN检索

- Stage 5: Second-pass exact ColBERT MaxSim re-ranking

- 阶段5：次轮精确ColBERT最大相似度重排序

<!-- Media -->

<!-- figureText: Doc. embs. Doc. embs. Doc. embs Explansion embs Expansion embs Indicative embs Indicative embs. 2 1 1.5 2.0 2.5 3.0 0.0 0.5 1.0 1.5 2.0 2.5 3.0 (c) KMedoid clustering. Expansion embs. Indicative embs. 2 1 0 0.0 0.5 1.0 1.5 2.0 2.5 3.0 0.0 0.5 1.0 (a) KMeans clustering. (b) KMeans-Closest clustering. -->

<img src="https://cdn.noedgeai.com/0195b372-2fb6-7dbd-a6e2-deb0bb3cad1c_28.jpg?x=196&y=236&w=1165&h=321&r=0"/>

Fig. 11. The illustration of different clustering methods. Dots in different colours indicate the document embeddings belonging to different clusters. Blue stars represents the expansion embedding, while the red diamond represents the indicative embedding used to measure the informativeness of the expansion embed-dings.

图11. 不同聚类方法的示意图。不同颜色的点表示属于不同聚类的文档嵌入。蓝色星星表示扩展嵌入，而红色菱形表示用于衡量扩展嵌入信息量的指示性嵌入。

<!-- Media -->

In the following, we discuss changes to the clustering (Stage 3 above, Section 8.1.1) and ANN retrieval (Stages 1 & 4, Section 8.1.2).

下面，我们讨论对聚类（上述阶段3，第8.1.1节）和ANN检索（阶段1和4，第8.1.2节）的更改。

8.1.1 Clustering. The default clustering technique in Stage 3 is the KMeans clustering algorithm. KMeans clustering is a widely used clustering method,which groups the samples into $k$ clusters according to their Euclidean distance to each other. Hence, in ColBERT-PRF, given a set of document embeddings and the number of clusters expected to be returned, KMeans clustering is employed to return a list of representative centroid embeddings. Figure 11(a) provides an illustration of the KMeans clustering method. Indeed, as shown in Figure 11(a), we notice that both cluster centroids (which can be applied as expansion embeddings for PRF) are distinct from the input embeddings. As a consequence, while measuring the importance and selecting the most informative ones among the representative centroid embeddings using IDF (or ICTF or MCos), we require to map each centroid embedding to a corresponding token id. As the representative centroid embedding, by definition, is not an actual document embedding, we turn to the FAISS ANN index and apply Equation (3) to obtain a list of token ids (see Section 4.2).

8.1.1 聚类。阶段3中的默认聚类技术是K均值（KMeans）聚类算法。K均值聚类是一种广泛使用的聚类方法，它根据样本之间的欧几里得距离将样本分组为$k$个聚类。因此，在ColBERT-PRF中，给定一组文档嵌入和预期返回的聚类数量，采用K均值聚类来返回一个代表性质心嵌入列表。图11(a)展示了K均值聚类方法。实际上，如图11(a)所示，我们注意到两个聚类质心（可作为PRF的扩展嵌入）都与输入嵌入不同。因此，在使用逆文档频率（IDF）（或逆聚类词频（ICTF）或最大余弦相似度（MCos））衡量代表性质心嵌入的重要性并选择最具信息量的嵌入时，我们需要将每个质心嵌入映射到相应的词元ID。由于代表性质心嵌入根据定义不是实际的文档嵌入，我们借助FAISS ANN索引并应用公式(3)来获取一个词元ID列表（见第4.2节）。

However, the main drawback of the above KMeans clustering method in ColBERT-PRF is that the procedure of looking up the most likely token for each of the $K$ centroid embeddings requires another $K$ FAISS lookups. To address this issue,we propose variants that avoid these additional FAISS lookups, by using the most likely token within each cluster - to do so, we recognise that the expansion embedding (which is added to the query) needs not perfectly alignment with the embedding used to measure informativeness, which we call the indicative embedding.

然而，上述ColBERT-PRF中的K均值聚类方法的主要缺点是，为$K$个质心嵌入中的每一个查找最可能的词元的过程需要另外进行$K$次FAISS查找。为了解决这个问题，我们提出了避免这些额外FAISS查找的变体，方法是使用每个聚类中最可能的词元——为此，我们认识到添加到查询中的扩展嵌入不需要与用于衡量信息量的嵌入（我们称之为指示性嵌入）完全对齐。

Our first proposed alternative strategy is called KMeans-Closest, which is still based on KMeans clustering but does not rely on additional FAISS lookups to obtain the most likely tokens. Once the $K$ centroid embeddings are computed,for each centroid we identify the closest feedback document embedding in the corresponding cluster - the indicative embedding for each cluster - and we use its token id to measure the importance score, such as IDF of the expansion embeddings. As shown in Figure 11(b), the indicative embeddings (the diamonds) are the closest actual document embeddings to the KMeans centroid embeddings (the blue stars).

我们提出的第一种替代策略称为KMeans-Closest，它仍然基于K均值聚类，但不依赖额外的FAISS查找来获取最可能的词元。一旦计算出$K$个质心嵌入，对于每个质心，我们在相应的聚类中识别出最接近的反馈文档嵌入——每个聚类的指示性嵌入——并使用其词元ID来衡量扩展嵌入的重要性得分，如IDF。如图11(b)所示，指示性嵌入（菱形）是最接近K均值质心嵌入（蓝色星星）的实际文档嵌入。

Our second proposed clustering strategy is KMedoids [18]. The KMedoids algorithm returns the medoid of each cluster - the medoid is the most centrally located embedding of the input document embeddings. Thus, after applying clustering upon the feedback document embeddings, for each cluster, we obtain the medoid (an indicative embedding for the cluster) that is also an actual document embedding, and hence can be mapped back to a token id, without requiring additional FAISS lookups for each centroid. Figure 11(c) depicts both the expansion embeddings and the indicative embeddings are the returned medoid embeddings of the KMedoids clustering algorithm.

我们提出的第二种聚类策略是K中心点（KMedoids）[18]。K中心点算法返回每个聚类的中心点——中心点是输入文档嵌入中位于最中心位置的嵌入。因此，在对反馈文档嵌入进行聚类后，对于每个聚类，我们得到一个中心点（该聚类的指示性嵌入），它也是一个实际的文档嵌入，因此可以映射回一个词元ID，而不需要为每个质心进行额外的FAISS查找。图11(c)展示了扩展嵌入和指示性嵌入都是K中心点聚类算法返回的中心点嵌入。

Overall, while the use of the KMeans-Closest and KMedoids methods can speed up the third stage of ColBERT-PRF, there might exist some potential risks (e.g., token id mismatch), thus hindering the effectiveness - hence, we report effectiveness as well as efficiency in our experiments.

总体而言，虽然使用K均值最近邻（KMeans - Closest）和K中心点（KMedoids）方法可以加快ColBERT - PRF第三阶段的速度，但可能存在一些潜在风险（例如，词元ID不匹配），从而影响其有效性。因此，我们在实验中同时报告了有效性和效率。

8.1.2 ANN Retrieval. The overall ColBERT-PRF Ranker process encapsulates a total of five stages, as shown in Figure 1. An ANN retrieval stage is used in both stages 1 & 4, and hence forms a significant part of the workflow. Indeed, as highlighted in Section 3, for each given query embedding,the approximate nearest neighbour search produces ${k}^{\prime }$ document embeddings for each query embedding, which are then mapped to the corresponding documents, thereby forming an unordered set of candidate documents. However, the contribution of the different query embeddings to the final score of the document varies (c.f. the contribution histogram in Figure 6). ${}^{17}$ Therefore, it is not efficient to take upto ${k}^{\prime } = {1000}$ documents for each query embedding forward to the 2nd stage for accurate MaxSim scoring, as not all of these documents will likely receive high scores.

8.1.2 近似最近邻（ANN）检索。如图1所示，整个ColBERT - PRF排序器（Ranker）过程总共包含五个阶段。在阶段1和阶段4都使用了ANN检索阶段，因此它构成了工作流程的重要部分。实际上，正如第3节所强调的，对于每个给定的查询嵌入，近似最近邻搜索会为每个查询嵌入生成${k}^{\prime }$个文档嵌入，然后将这些嵌入映射到相应的文档，从而形成一个无序的候选文档集。然而，不同查询嵌入对文档最终得分的贡献是不同的（参见图6中的贡献直方图）。${}^{17}$ 因此，将每个查询嵌入对应的多达${k}^{\prime } = {1000}$个文档推进到第二阶段进行精确的最大相似度（MaxSim）评分是低效的，因为并非所有这些文档都可能获得高分。

To this end, we experiment with using Approximate Scoring [26] at the first stage, as well as in the later stage 4 retrieval. In particular, this approach makes use of the MaxSim operator applied on the approximate cosine scores of the ANN algorithm, to generate a ranking of candidates from the first stage. Indeed,as this is a ranking,rather than a set,then the number of the candidates $k$ can be directly controlled,rather than indirectly through ${k}^{\prime }$ . While this requires more computation in stage 1 (and has a small negative impact on the response time of that stage), its has marked overall efficiency benefits [26] for ColBERT dense retrieval, as a smaller number of candidates can be passed to MaxSim without loss of recall.

为此，我们尝试在第一阶段以及后续的阶段4检索中使用近似评分（Approximate Scoring）[26]。具体而言，这种方法利用了应用于ANN算法近似余弦得分的最大相似度运算符，从第一阶段生成候选文档的排名。实际上，由于这是一个排名，而不是一个集合，因此可以直接控制候选文档的数量$k$，而不是通过${k}^{\prime }$间接控制。虽然这在阶段1需要更多的计算（并且对该阶段的响应时间有轻微的负面影响），但它对ColBERT密集检索具有显著的整体效率优势[26]，因为可以在不损失召回率的情况下将较少数量的候选文档传递给最大相似度评分。

More specifically, for the ColBERT-PRF instantiated as Ranker model, we apply the Approximate Scoring technique only in the first stage or in both the first and fourth stage of the ColBERT-PRF-Ranker model. Indeed, as we only require the most relevant three feedback passages for effective PRF, accurately scoring thousands of passages retrieved by the 1st ANN stage is superfluous. For the ColBERT-PRF instantiated as the ReRanker model, we apply Approximate Scoring in the first stage. In addition, we further investigate the efficiency and effectiveness trade-off when implementing the different clustering technique and the Approximate Scoring technique in the various ColBERT stages.

更具体地说，对于实例化为排序器模型的ColBERT - PRF，我们仅在ColBERT - PRF - 排序器模型的第一阶段或第一阶段和第四阶段应用近似评分技术。实际上，由于我们仅需要最相关的三个反馈段落来实现有效的伪相关反馈（PRF），因此对第一阶段ANN检索到的数千个段落进行精确评分是多余的。对于实例化为重排序器（ReRanker）模型的ColBERT - PRF，我们在第一阶段应用近似评分。此外，我们进一步研究了在ColBERT的各个阶段实施不同聚类技术和近似评分技术时的效率和有效性权衡。

### 8.2 Research Question & Experimental Setup

### 8.2 研究问题与实验设置

- RQ8: What is the impact on efficiency and effectiveness of the ColBERT-PRF model using different clustering methods, namely the KMeans and KMeans-Closest clustering methods and the KMedoids clustering method?

- 研究问题8（RQ8）：使用不同的聚类方法，即K均值（KMeans）、K均值最近邻（KMeans - Closest）聚类方法和K中心点（KMedoids）聚类方法，对ColBERT - PRF模型的效率和有效性有何影响？

- RQ9: What is the impact on efficiency and effectiveness of the ColBERT-PRF model when instantiated using Approximate Scoring?

- 研究问题9（RQ9）：当使用近似评分实例化ColBERT - PRF模型时，对其效率和有效性有何影响？

Dataset: We compare the efficiency and the effectiveness of ColBERT-PRF model efficient variants on TREC 2019 and TREC 2020 query sets from MSMARCO passage.

数据集：我们比较了ColBERT - PRF模型高效变体在来自MSMARCO段落的2019年文本检索会议（TREC 2019）和2020年文本检索会议（TREC 2020）查询集上的效率和有效性。

Measures: For measuring the performance in terms of efficiency, we report the Mean Response Time (MRT) for each stage of the ColBERT-PRF model (described in Figure 1) and its overall MRT. Mean response times are measured with one Nvidia Titan RTX GPU (using a single thread for retrieval). In addition, we report the effectiveness performance with the metrics used in Section 5.2, namely MAP, NDCG@10, MRR and Recall. For significance testing, we use the paired t-test ( $p <$ 0.05 and apply the Holm-Bonferroni multiple testing correction technique. Experimental setup: For both KMeans-Closest and KMedoids clustering, we reuse the default setting of the KMeans clustering algorithm,i.e.,the number of clusters $K = {24}$ ,the number of feedback documents ${f}_{b} = 3$ ,and the number of expansion embeddings ${f}_{e} = {10}$ . As for $\beta$ ,based on the conclusions obtained from Section 5,we pick the appropriate $\beta$ for each query set,namely $\beta  = 1$ and $\beta  = {0.5}$ for the TREC 2019 and TREC 2020 passage ranking query sets,respectively. For the Approximate Scoring experiments,let ${k}_{1}$ denote the number of passages retrieved in the Stage 1 ANN,and ${k}_{4}$ denote the number of passages retrieved in the Stage 4 ANN. Then,for (i) the ColBERT-PRF Ranker model,we apply with rank cutoff of ${k}_{1} = {300}$ and ${k}_{4} = {1000},{}^{18}$ and for (ii) the ReRanker model,we apply with rank cutoff ${k}_{1} = {1000}$ in the first stage only,to ensure sufficient recall of relevant passages to be upranked after applying PRF. We later vary ${k}_{1}$ and ${k}_{4}$ to demonstrate their impact upon efficiency and effectiveness.

衡量指标：为了从效率方面衡量性能，我们报告了ColBERT - PRF模型（如图1所示）每个阶段的平均响应时间（Mean Response Time，MRT）及其整体MRT。平均响应时间是使用一块英伟达Titan RTX GPU（使用单线程进行检索）进行测量的。此外，我们使用5.2节中使用的指标报告有效性性能，即平均准确率均值（Mean Average Precision，MAP）、前10名归一化折损累积增益（Normalized Discounted Cumulative Gain at 10，NDCG@10）、平均倒数排名（Mean Reciprocal Rank，MRR）和召回率（Recall）。对于显著性检验，我们使用配对t检验（$p <$ 0.05）并应用霍尔姆 - 邦费罗尼多重检验校正技术。实验设置：对于K均值最近邻（KMeans - Closest）和K中心点（KMedoids）聚类，我们复用K均值聚类算法的默认设置，即聚类数量$K = {24}$、反馈文档数量${f}_{b} = 3$和扩展嵌入数量${f}_{e} = {10}$。至于$\beta$，根据从第5节得出的结论，我们为每个查询集选择合适的$\beta$，即分别为2019年文本检索会议（TREC 2019）和2020年文本检索会议（TREC 2020）段落排名查询集选择$\beta  = 1$和$\beta  = {0.5}$。对于近似评分实验，令${k}_{1}$表示在第一阶段近似最近邻搜索（ANN）中检索到的段落数量，${k}_{4}$表示在第四阶段ANN中检索到的段落数量。然后，（i）对于ColBERT - PRF排序器模型，我们应用排名截断值${k}_{1} = {300}$和${k}_{4} = {1000},{}^{18}$；（ii）对于重排序器模型，我们仅在第一阶段应用排名截断值${k}_{1} = {1000}$，以确保在应用伪相关反馈（PRF）后有足够的相关段落召回率以提升排名。我们随后改变${k}_{1}$和${k}_{4}$的值，以展示它们对效率和有效性的影响。

---

<!-- Footnote -->

${}^{17}$ Indeed,in separate but orthogonal work [41],we show that query embeddings vary in their ability to recall relevant documents, and some can even be discarded (pruned) from the ANN search phase without significant loss of effectiveness.

${}^{17}$ 实际上，在另一项但相互独立的工作[41]中，我们表明查询嵌入在召回相关文档的能力方面存在差异，并且有些查询嵌入甚至可以在近似最近邻搜索阶段被丢弃（修剪）而不会显著降低有效性。

<!-- Footnote -->

---

### 8.3 Results

### 8.3 结果

8.3.1 RQ8 - Clustering Variants. Table 7 lists the effectiveness and the efficiency performance for ColBERT E2E and the ColBERT-PRF instantiated as Ranker and ReRanker models on both the TREC 2019 and TREC 2020 passage ranking query sets. In terms of efficiency, we measure the MRT of the different ColBERT-PRF stages as well as the overall MRT for each model variant. From Table 7, we note that, for both the TREC 2019 and TREC 2020 query sets, both the ColBERT-PRF Ranker and ReRanker model variants implemented with KMeans-Closest and KMedoids clustering methods are much faster than the KMeans clustering method model, without markedly compromising their effectiveness. In particular, both KMeans-Closest and KMedoids still exhibit enhanced NDCG@10 and MAP (significantly so) over the ColBERT E2E baseline. Moreover, this speed benefit is obtained by omitting the FAISS lookup step in the default ColBERT-PRF with KMeans-Closest and KMedoids clustering algorithms, as large efficiency improvements can be observed in the Stage 3 column of Table 7 (e.g.,on TREC 2019, $\sim  {900}\mathrm{\;{ms}}$ for KMeans-Closest vs. $\sim  {3000}\mathrm{\;{ms}}$ for KMeans). Going further, KMedoids is faster still (218ms on TREC 2019), demonstrating the benefit of a fast clustering algorithm, with no further loss of effectiveness compared to KMeans-Closest.

8.3.1 研究问题8 - 聚类变体。表7列出了ColBERT端到端（E2E）模型以及实例化为排序器和重排序器模型的ColBERT - PRF在2019年文本检索会议（TREC 2019）和2020年文本检索会议（TREC 2020）段落排名查询集上的有效性和效率性能。在效率方面，我们测量了不同ColBERT - PRF阶段的MRT以及每个模型变体的整体MRT。从表7中我们注意到，对于TREC 2019和TREC 2020查询集，使用K均值最近邻（KMeans - Closest）和K中心点（KMedoids）聚类方法实现的ColBERT - PRF排序器和重排序器模型变体都比K均值聚类方法模型快得多，并且在有效性方面没有明显损失。特别是，KMeans - Closest和KMedoids在ColBERT E2E基线之上仍表现出增强的NDCG@10和MAP（显著增强）。此外，通过在使用KMeans - Closest和KMedoids聚类算法的默认ColBERT - PRF中省略快速近似最近邻搜索（FAISS）查找步骤获得了这种速度优势，如表7的第三阶段列中可以观察到显著的效率提升（例如，在TREC 2019上，KMeans - Closest的$\sim  {900}\mathrm{\;{ms}}$对比K均值的$\sim  {3000}\mathrm{\;{ms}}$）。更进一步，KMedoids更快（在TREC 2019上为218毫秒），展示了快速聚类算法的优势，与KMeans - Closest相比没有进一步的有效性损失。

Overall, in a reranking scenario, KMeans-Closest and KMedoids clustering methods experience upto 2.48 $\times$ and ${4.54} \times$ speedups,respectively. Indeed,the mean response times of KMedoids of 766ms (TREC 2020) is very respectable compared to the ColBERT E2E baseline,despite the normally expensive application of a PRF technique. Thus, in response to RQ8, we conclude that for both the ColBERT-PRF Ranker and ReRanker models with KMeans-Closest or KMedoids clustering are more efficient than the KMeans clustering method without compromising the effectiveness.

总体而言，在重排序场景中，K均值最近邻（KMeans - Closest）和K中心点（KMedoids）聚类方法分别实现了高达2.48 $\times$ 和 ${4.54} \times$ 的加速。实际上，尽管通常应用伪相关反馈（PRF）技术的成本较高，但K中心点聚类方法在2020年文本检索会议（TREC 2020）中的平均响应时间为766毫秒，与ColBERT端到端（E2E）基线相比表现相当不错。因此，针对研究问题8（RQ8），我们得出结论：对于采用K均值最近邻或K中心点聚类的ColBERT - PRF排序器和重排序器模型，在不影响有效性的前提下，比未采用聚类的K均值聚类方法更高效。

8.3.2 RQ9 - Variants using Approximate Scoring. Next, we consider the application of Approximate Scoring within ColBERT-PRF. Again, efficiency and effectiveness results are reported in Table 7. We report response times only for KMeans. Firstly, on examining the table, we find that Approximate Scoring applied in both the first stage and the fourth stage of the ColBERT-PRF Ranker model exhibits similar effectiveness performance but much more efficient than the original ColBERT-PRF Ranker model. In addition, deploying Approximate Scoring within the ColBERT-PRF ReRanker model also reduces the response time while still outperforming the ColBERT E2E model (but not by a significant margin). From Table 7, we see that rows with Approximate Scoring techniques applied exhibit increased Stage 1 times ( ${43}\mathrm{\;{ms}} \rightarrow  {95}\mathrm{\;{ms}}/{90}\mathrm{\;{ms}}$ for Ranker,as MaxSim takes time to compute), but are much faster in Stage 2, as the exact scoring only occurs in the selected high quality candidates ( ${344}\mathrm{\;{ms}} \rightarrow  {22}\mathrm{\;{ms}}/{23}\mathrm{\;{ms}}$ for Ranker). The next effect of replacing both of the set retrieval ANN stages with Approximate Scoring in Ranker is an up to 18% speedup in response times $\left( {{4103}\mathrm{\;{ms}} \rightarrow  {3466}\mathrm{\;{ms}}}\right)$ ,while still maintaining high effectiveness,e.g.,significant improvements in MAP over the baseline ColBERT E2E.

8.3.2 研究问题9（RQ9） - 使用近似评分的变体。接下来，我们考虑在ColBERT - PRF中应用近似评分。同样，表7报告了效率和有效性结果。我们仅报告K均值聚类的响应时间。首先，查看该表可知，在ColBERT - PRF排序器模型的第一阶段和第四阶段应用近似评分，其有效性表现相似，但比原始的ColBERT - PRF排序器模型高效得多。此外，在ColBERT - PRF重排序器模型中部署近似评分也能减少响应时间，同时仍优于ColBERT端到端模型（但优势并不显著）。从表7中可以看出，应用近似评分技术的行在第一阶段的时间有所增加（排序器为 ${43}\mathrm{\;{ms}} \rightarrow  {95}\mathrm{\;{ms}}/{90}\mathrm{\;{ms}}$，因为最大相似度（MaxSim）计算需要时间），但在第二阶段快得多，因为精确评分仅在选定的高质量候选对象中进行（排序器为 ${344}\mathrm{\;{ms}} \rightarrow  {22}\mathrm{\;{ms}}/{23}\mathrm{\;{ms}}$）。在排序器中用近似评分替换两个集合检索近似最近邻（ANN）阶段的另一个效果是，响应时间最多可加快18% $\left( {{4103}\mathrm{\;{ms}} \rightarrow  {3466}\mathrm{\;{ms}}}\right)$，同时仍保持较高的有效性，例如，与基线ColBERT端到端模型相比，平均准确率均值（MAP）有显著提高。

---

<!-- Footnote -->

${}^{18}$ Indeed,[26] suggest $k = {300}$ is sufficient for high precision retrieval.

${}^{18}$ 实际上，[26] 表明 $k = {300}$ 足以实现高精度检索。

<!-- Footnote -->

---

<!-- Media -->

Table 7. Mean Response Time and the Effectiveness on Both TREC 2019 and TREC 2020 Passage Ranking Query Sets

表7. 2019年文本检索会议（TREC 2019）和2020年文本检索会议（TREC 2020）段落排序查询集的平均响应时间和有效性

<table><tr><td rowspan="2">Models</td><td rowspan="2">PRF Description</td><td colspan="6">Mean Response Time (ms)</td><td rowspan="2">MAP</td><td rowspan="2">NDCG@10</td><td rowspan="2">MRR</td><td rowspan="2">Recall</td></tr><tr><td>Stage 1</td><td>Stage 2</td><td>Stage 3</td><td>Stage 4</td><td>Stage 5</td><td>Overall</td></tr><tr><td colspan="12">TREC 2019 query set</td></tr><tr><td>ColBERT E2E</td><td>-</td><td>47</td><td>318</td><td>-</td><td>-</td><td>-</td><td>365</td><td>0.4318</td><td>0.6934</td><td>0.8529</td><td>0.7892</td></tr><tr><td rowspan="5">ColBERT-PRF Ranker</td><td>KMeans</td><td>43</td><td>344</td><td>2997</td><td>61</td><td>658</td><td>4103</td><td>${0.5431} \dagger$</td><td>0.7352</td><td>0.8858</td><td>${\mathbf{{0.8706}}}^{ + }$</td></tr><tr><td>KMeans-Closest</td><td>45</td><td>333</td><td>903</td><td>116</td><td>641</td><td>2038 (2.01x)</td><td>${0.5075} \dagger$</td><td>0.7289</td><td>0.8497</td><td>0.8507†</td></tr><tr><td>KMedoids</td><td>45</td><td>327</td><td>218</td><td>134</td><td>610</td><td>1334 (3.07×)</td><td>0.5073+</td><td>0.7200</td><td>0.8723</td><td>0.8681†</td></tr><tr><td>Approximate Scoring (Stage 1)</td><td>95</td><td>22</td><td>3011</td><td>56</td><td>684</td><td>3868 (1.06%)</td><td>0.5478+</td><td>0.7314</td><td>0.8649</td><td>${0.8649} \dagger$</td></tr><tr><td>Approximate Scoring (Stages 1 & 4 )</td><td>90</td><td>23</td><td>3158</td><td>129</td><td>66</td><td>3466 (1.18%)</td><td>${0.5196} \dagger$</td><td>0.7314</td><td>0.8042</td><td>0.8646+</td></tr><tr><td rowspan="4">ColBERT-PRF ReRanker</td><td>KMeans</td><td>47</td><td>374</td><td>3047</td><td>-</td><td>75</td><td>3543</td><td>${0.5040} \dagger$</td><td>0.7369</td><td>0.8858</td><td>0.7961</td></tr><tr><td>KMeans-Closest</td><td>47</td><td>352</td><td>921</td><td>-</td><td>110</td><td>1430 (2.48x)</td><td>${0.4700} \dagger$</td><td>0.7062</td><td>0.8497</td><td>0.7890</td></tr><tr><td>KMedoids</td><td>47</td><td>351</td><td>257</td><td>-</td><td>139</td><td>794 (4.46%)</td><td>${0.4744} \dagger$</td><td>0.7235</td><td>0.8723</td><td>0.7892</td></tr><tr><td>Approximate Scoring (Stage 1)</td><td>93</td><td>56</td><td>3214</td><td>-</td><td>68</td><td>3431 (1.03%)</td><td>0.4565</td><td>0.7336</td><td>0.8858</td><td>0.6953</td></tr><tr><td colspan="12">TREC 2020 query set</td></tr><tr><td>ColBERT E2E</td><td>-</td><td>44</td><td>346</td><td>-</td><td>-</td><td>-</td><td>390</td><td>0.4654</td><td>0.6871</td><td>0.8525</td><td>0.8245</td></tr><tr><td rowspan="5">ColBERT-PRF Ranker</td><td>KMeans</td><td>45</td><td>346</td><td>3033</td><td>54</td><td>677</td><td>4155</td><td>${0.5116} \dagger$</td><td>0.7152</td><td>0.8439</td><td>0.8837†</td></tr><tr><td>KMeans-Closest</td><td>45</td><td>348</td><td>945</td><td>120</td><td>647</td><td>2105 (1.97%)</td><td>0.4920†</td><td>0.7054</td><td>0.7850</td><td>${0.8670} \dagger$</td></tr><tr><td>KMedoids</td><td>45</td><td>338</td><td>222</td><td>134</td><td>609</td><td>1348 (3.08x)</td><td>${0.4970} \dagger$</td><td>0.7065</td><td>0.8363</td><td>${0.8787} \dagger$</td></tr><tr><td>Approximate Scoring (Stage 1)</td><td>91</td><td>22</td><td>3030</td><td>60</td><td>711</td><td>3914 (1.06%)</td><td>${0.5062} \dagger$</td><td>0.7108</td><td>0.8417</td><td>0.8802†</td></tr><tr><td>Approximate Scoring (Stages 1 & 4 )</td><td>89</td><td>22</td><td>3086</td><td>137</td><td>63</td><td>3397 (1.22x)</td><td>0.4954†</td><td>0.7091</td><td>0.8019</td><td>0.8419</td></tr><tr><td rowspan="4">ColBERT-PRF ReRanker</td><td>KMeans</td><td>47</td><td>374</td><td>2922</td><td>-</td><td>64</td><td>3477</td><td>0.5049 ${}^{ + }$</td><td>0.7165</td><td>0.8439</td><td>0.8246</td></tr><tr><td>KMeans-Closest</td><td>46</td><td>352</td><td>987</td><td>-</td><td>106</td><td>1491 (2.33%)</td><td>0.4908</td><td>0.7061</td><td>0.7850</td><td>0.8255</td></tr><tr><td>KMedoids</td><td>47</td><td>341</td><td>251</td><td>-</td><td>127</td><td>766 (4.54x)</td><td>${0.4927} \dagger$</td><td>0.7077</td><td>0.8363</td><td>0.8245</td></tr><tr><td>Approximate Scoring (Stage 1)</td><td>96</td><td>54</td><td>3110</td><td>-</td><td>72</td><td>3332 (1.04x)</td><td>0.4858</td><td>0.7127</td><td>0.8464</td><td>0.7550</td></tr></table>

<table><tbody><tr><td rowspan="2">模型</td><td rowspan="2">PRF描述</td><td colspan="6">平均响应时间（毫秒）</td><td rowspan="2">平均准确率均值（Mean Average Precision，MAP）</td><td rowspan="2">前10名归一化折损累积增益（Normalized Discounted Cumulative Gain at 10，NDCG@10）</td><td rowspan="2">平均倒数排名（Mean Reciprocal Rank，MRR）</td><td rowspan="2">召回率</td></tr><tr><td>阶段1</td><td>阶段2</td><td>阶段3</td><td>阶段4</td><td>阶段5</td><td>总体</td></tr><tr><td colspan="12">2019年文本检索会议（TREC）查询集</td></tr><tr><td>ColBERT端到端模型</td><td>-</td><td>47</td><td>318</td><td>-</td><td>-</td><td>-</td><td>365</td><td>0.4318</td><td>0.6934</td><td>0.8529</td><td>0.7892</td></tr><tr><td rowspan="5">ColBERT - PRF排序器</td><td>K均值聚类算法（KMeans）</td><td>43</td><td>344</td><td>2997</td><td>61</td><td>658</td><td>4103</td><td>${0.5431} \dagger$</td><td>0.7352</td><td>0.8858</td><td>${\mathbf{{0.8706}}}^{ + }$</td></tr><tr><td>最近邻K均值聚类算法（KMeans - Closest）</td><td>45</td><td>333</td><td>903</td><td>116</td><td>641</td><td>2038 (2.01x)</td><td>${0.5075} \dagger$</td><td>0.7289</td><td>0.8497</td><td>0.8507†</td></tr><tr><td>K中心点聚类算法（KMedoids）</td><td>45</td><td>327</td><td>218</td><td>134</td><td>610</td><td>1334（3.07倍）</td><td>0.5073+</td><td>0.7200</td><td>0.8723</td><td>0.8681†</td></tr><tr><td>近似评分（阶段1）</td><td>95</td><td>22</td><td>3011</td><td>56</td><td>684</td><td>3868 (1.06%)</td><td>0.5478+</td><td>0.7314</td><td>0.8649</td><td>${0.8649} \dagger$</td></tr><tr><td>近似评分（阶段1和阶段4）</td><td>90</td><td>23</td><td>3158</td><td>129</td><td>66</td><td>3466 (1.18%)</td><td>${0.5196} \dagger$</td><td>0.7314</td><td>0.8042</td><td>0.8646+</td></tr><tr><td rowspan="4">ColBERT - PRF重排序器</td><td>K均值聚类算法（KMeans）</td><td>47</td><td>374</td><td>3047</td><td>-</td><td>75</td><td>3543</td><td>${0.5040} \dagger$</td><td>0.7369</td><td>0.8858</td><td>0.7961</td></tr><tr><td>最近邻K均值聚类算法（KMeans - Closest）</td><td>47</td><td>352</td><td>921</td><td>-</td><td>110</td><td>1430 (2.48x)</td><td>${0.4700} \dagger$</td><td>0.7062</td><td>0.8497</td><td>0.7890</td></tr><tr><td>K中心点聚类算法（KMedoids）</td><td>47</td><td>351</td><td>257</td><td>-</td><td>139</td><td>794 (4.46%)</td><td>${0.4744} \dagger$</td><td>0.7235</td><td>0.8723</td><td>0.7892</td></tr><tr><td>近似评分（阶段1）</td><td>93</td><td>56</td><td>3214</td><td>-</td><td>68</td><td>3431 (1.03%)</td><td>0.4565</td><td>0.7336</td><td>0.8858</td><td>0.6953</td></tr><tr><td colspan="12">2020年文本检索会议（TREC）查询集</td></tr><tr><td>ColBERT端到端模型</td><td>-</td><td>44</td><td>346</td><td>-</td><td>-</td><td>-</td><td>390</td><td>0.4654</td><td>0.6871</td><td>0.8525</td><td>0.8245</td></tr><tr><td rowspan="5">ColBERT - PRF排序器</td><td>K均值聚类算法（KMeans）</td><td>45</td><td>346</td><td>3033</td><td>54</td><td>677</td><td>4155</td><td>${0.5116} \dagger$</td><td>0.7152</td><td>0.8439</td><td>0.8837†</td></tr><tr><td>最近邻K均值聚类算法（KMeans - Closest）</td><td>45</td><td>348</td><td>945</td><td>120</td><td>647</td><td>2105 (1.97%)</td><td>0.4920†</td><td>0.7054</td><td>0.7850</td><td>${0.8670} \dagger$</td></tr><tr><td>K中心点聚类算法（KMedoids）</td><td>45</td><td>338</td><td>222</td><td>134</td><td>609</td><td>1348 (3.08x)</td><td>${0.4970} \dagger$</td><td>0.7065</td><td>0.8363</td><td>${0.8787} \dagger$</td></tr><tr><td>近似评分（阶段1）</td><td>91</td><td>22</td><td>3030</td><td>60</td><td>711</td><td>3914 (1.06%)</td><td>${0.5062} \dagger$</td><td>0.7108</td><td>0.8417</td><td>0.8802†</td></tr><tr><td>近似评分（阶段1和阶段4）</td><td>89</td><td>22</td><td>3086</td><td>137</td><td>63</td><td>3397 (1.22x)</td><td>0.4954†</td><td>0.7091</td><td>0.8019</td><td>0.8419</td></tr><tr><td rowspan="4">ColBERT - PRF重排序器</td><td>K均值聚类算法（KMeans）</td><td>47</td><td>374</td><td>2922</td><td>-</td><td>64</td><td>3477</td><td>0.5049 ${}^{ + }$</td><td>0.7165</td><td>0.8439</td><td>0.8246</td></tr><tr><td>最近邻K均值聚类算法（KMeans - Closest）</td><td>46</td><td>352</td><td>987</td><td>-</td><td>106</td><td>1491 (2.33%)</td><td>0.4908</td><td>0.7061</td><td>0.7850</td><td>0.8255</td></tr><tr><td>K中心点聚类算法（KMedoids）</td><td>47</td><td>341</td><td>251</td><td>-</td><td>127</td><td>766 (4.54x)</td><td>${0.4927} \dagger$</td><td>0.7077</td><td>0.8363</td><td>0.8245</td></tr><tr><td>近似评分（阶段1）</td><td>96</td><td>54</td><td>3110</td><td>-</td><td>72</td><td>3332 (1.04x)</td><td>0.4858</td><td>0.7127</td><td>0.8464</td><td>0.7550</td></tr></tbody></table>

$\dagger$ indicates significant improvement over the ColBERT-E2E model. The highest effectiveness and lowest response time value in each scenario is boldfaced.

$\dagger$ 表示相较于 ColBERT - E2E 模型有显著改进。每个场景中最高的有效性和最低的响应时间值用粗体表示。

<!-- Media -->

Next, we further study the trade-off between the efficiency and the effectiveness of ColBERT-PRF applied with Approximate Scoring, as well as the benefits brought by the different clustering techniques. Aligned with the table, Figure 12 presents both the effectiveness and efficiency of the following three strategies on the TREC 2019 query set: (i) ColBERT-PRF Ranker applied with Approximate Scoring in stage 1 using three different clustering techniques; (ii) ColBERT-PRF Ranker applied with Approximate Scoring in both stage 1 and stage 4 using three different clustering techniques, and (iii) ColBERT-PRF ReRanker applied with Approximate Scoring in stage 1 using three different clustering techniques. In each figure,we vary the cutoff, ${k}_{1}$ or ${k}_{4}$ ,of Approximate Scoring to produce curves for each setting $\left( {{100} \leq  \left\{  {{k}_{1},{k}_{3}}\right\}   \leq  {7300}^{19}}\right)$ . We provide separate figures for MAP and NDCG@10.Each figure has two asterisk points (★) denoting the performance of ColBERT E2E, and the ColBERT-PRF default setting (KMeans, ANN set retrieval). For the points in each curve,the marker $\bullet$ indicates the corresponding performance is significantly improved (and $\times$ indicates not significantly) over the ColBERT E2E baseline.

接下来，我们进一步研究应用近似评分的 ColBERT - PRF 在效率和有效性之间的权衡，以及不同聚类技术带来的益处。与表格一致，图 12 展示了以下三种策略在 TREC 2019 查询集上的有效性和效率：（i）在阶段 1 应用近似评分并使用三种不同聚类技术的 ColBERT - PRF 排序器；（ii）在阶段 1 和阶段 4 都应用近似评分并使用三种不同聚类技术的 ColBERT - PRF 排序器；（iii）在阶段 1 应用近似评分并使用三种不同聚类技术的 ColBERT - PRF 重排序器。在每个图中，我们改变近似评分的截断值 ${k}_{1}$ 或 ${k}_{4}$，为每个设置 $\left( {{100} \leq  \left\{  {{k}_{1},{k}_{3}}\right\}   \leq  {7300}^{19}}\right)$ 生成曲线。我们分别给出了平均准确率均值（MAP）和前 10 名归一化折损累积增益（NDCG@10）的图。每个图中有两个星号点（★）表示 ColBERT E2E 的性能以及 ColBERT - PRF 默认设置（K 均值聚类，近似最近邻集检索）的性能。对于每条曲线中的点，标记 $\bullet$ 表示相应性能相较于 ColBERT E2E 基线有显著提升（$\times$ 表示无显著提升）。

Firstly, we analyse ColBERT-PRF Ranker when only the Stage 1 Approximate Scoring is applied. From Figure 12(b),we observe that,for the smaller ${k}_{1}$ ,there is some minor degradation of NDCG@10; but the impact on MAP (Figure 12(a)) is indistinguishable. In terms of efficiency, it can easily be seen that KMedoids is the most efficient technique, followed by the KMeans-Closest technique and finally the KMeans clustering technique.

首先，我们分析仅应用阶段 1 近似评分时的 ColBERT - PRF 排序器。从图 12（b）中我们观察到，对于较小的 ${k}_{1}$，NDCG@10 有一些轻微下降；但对平均准确率均值（MAP，图 12（a））的影响不明显。在效率方面，很容易看出 K 中心点聚类（KMedoids）是最有效的技术，其次是最近 K 均值（KMeans - Closest）技术，最后是 K 均值（KMeans）聚类技术。

We next consider Figures 12(c) and 12(d), where we applied the Approximate Scoring technique for both the first and fourth stages for ColBERT-PRF Ranker model, with the different clustering methods. More specifically, ${k}_{1}$ ,the rank cutoff of first stage Approximate Scoring is fixed to 300,while ${k}_{4}$ is varied. From Figure 12(c),we find that all of the three clustering techniques exhibit correlations between efficiency and effectiveness, in that increased MRT also exhibits increased effectiveness. Moreover,reducing ${k}_{4}$ results in more marked degradations for MAP than for NDCG@10, and, for each of the three clustering methods, stable effectiveness can be achieved with large enough ${k}_{4}$ .

接下来，我们考虑图 12（c）和 12（d），在这两个图中，我们对 ColBERT - PRF 排序器模型的第一阶段和第四阶段都应用了近似评分技术，并使用了不同的聚类方法。更具体地说，第一阶段近似评分的排名截断值 ${k}_{1}$ 固定为 300，而 ${k}_{4}$ 是变化的。从图 12（c）中我们发现，三种聚类技术在效率和有效性之间都呈现出相关性，即平均响应时间（MRT）增加时，有效性也会增加。此外，减小 ${k}_{4}$ 对平均准确率均值（MAP）的影响比对前 10 名归一化折损累积增益（NDCG@10）的影响更明显，并且对于三种聚类方法中的每一种，当 ${k}_{4}$ 足够大时都可以实现稳定的有效性。

---

<!-- Footnote -->

${}^{19}{7300}$ is the average number of passages retrieved by ColBERT E2E for ${k}^{\prime } = {1000}$ .

${}^{19}{7300}$ 是 ColBERT E2E 为 ${k}^{\prime } = {1000}$ 检索到的段落的平均数量。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: 0.54 0.73 NDCG@10 0.72 0.70 ColBERT-PRF Ranker KMeans ColBERT-PRF Ranker KMeans-Closest ColBERT-PRF Ranker KMedoids 1000 2000 3000 4000 Mean Response Time (ms) (b) ANN 1 stage ColBERT-PRF Ranker 0.73 NDCG@10 0.72 0.71 0.70 KMeans ANN 1stage=300 KMeans-Closest ANN 1stage=300 KMedoids ANN 1stage=300 500 1000 1500 2000 2500 3000 3500 4000 Mean Response Time (ms (d) ANN 1&4 stage ColBERT-PRF Ranker ColBERT-PRF ReRanker KMeans ColBERT-PRF ReRanker KMedoids 0.73 NDCG@10 0.72 0.71 0.70 2000 2500 Mean Response Time (ms) (f) ANN 1 stage ColBERT-PRF ReRanker 0.52 MAP 0.50 0.48 0.44 ColBERT-PRF Ranker KMeans-Closest ColBERT-PRF Ranker KMedoids 1000 2000 3000 4000 Mean Response Time (ms) (a) ANN 1 stage ColBERT-PRF Ranker 0.54 KMeans ANN 1stage=300 KMeans-Closest ANN 1stage $= {300}$ KMedoids ANN 1stage=300 0.52 0.50 MAP 0.48 0.46 0.44 0.42 0.40 500 1000 1500 2000 2500 3000 3500 4000 Mean Response Time (m: c) ANN 1&4 stage ColBERT-PRF Ranker 0.50 ColBERT-PRF ReRanker KMeans 0.48 ColBERT-PRF ReRanker KMedoids 0.46 0.44 MAP 0.42 0.38 0.36 0.34 1000 1500 3000 3500 Mean Response Time (ms) (e) ANN 1 stage ColBERT-PRF ReRanker -->

<img src="https://cdn.noedgeai.com/0195b372-2fb6-7dbd-a6e2-deb0bb3cad1c_32.jpg?x=194&y=231&w=1174&h=1331&r=0"/>

Fig. 12. Trade-off between efficiency and effectiveness for ColBERT-PRF implemented with different clustering methods and the Approximate Scoring technique. The star coloured with purple and the red represents the ColBERT E2E and the default ColBERT PRF Ranker or ReRanker performance. A point marker of $\bullet$ indicates the corresponding performance is significantly improved (and $\times$ indicates not significantly) over the ColBERT-E2E baseline.

图 12. 使用不同聚类方法和近似评分技术实现的 ColBERT - PRF 在效率和有效性之间的权衡。紫色和红色的星号分别表示 ColBERT E2E 和默认的 ColBERT PRF 排序器或重排序器的性能。标记 $\bullet$ 表示相应性能相较于 ColBERT - E2E 基线有显著提升（$\times$ 表示无显著提升）。

<!-- Media -->

Finally, we analyse the efficiency/effectiveness trade-off for ColBERT-PRF ReRanker. From Figures 12(e) and 12(f), we observe that the trade-off curves for ColBERT-PRF ReRanker model with different clustering technique exhibits similar trend with Figures 12(c) and 12(d), with the slightly lower MAP values typically exhibited by ReRanker in comparison to the Ranker setting of ColBERT-PRF. Overall,reducing ${k}_{1}$ here can markedly impact both MAP and NDCG@10.However,using sufficiently large ${k}_{1}$ can still result in significantly enhanced MAP (denoted using $\bullet$ ),even with response times around ${1000}\mathrm{\;{ms}}$ . This is markedly faster than the default ColBERT-PRF ReRanker setting,which attains ${3500}\mathrm{\;{ms}}$ (shown as $\star$ ) and much closer to the default response time of ColBERT E2E (★).

最后，我们分析 ColBERT - PRF 重排序器在效率/有效性方面的权衡。从图 12（e）和 12（f）中我们观察到，使用不同聚类技术的 ColBERT - PRF 重排序器模型的权衡曲线与图 12（c）和 12（d）呈现出相似的趋势，与 ColBERT - PRF 的排序器设置相比，重排序器通常显示出略低的平均准确率均值（MAP）值。总体而言，减小这里的 ${k}_{1}$ 会显著影响平均准确率均值（MAP）和前 10 名归一化折损累积增益（NDCG@10）。然而，使用足够大的 ${k}_{1}$ 仍然可以显著提高平均准确率均值（MAP，用 $\bullet$ 表示），即使响应时间约为 ${1000}\mathrm{\;{ms}}$。这比默认的 ColBERT - PRF 重排序器设置明显更快，默认设置达到 ${3500}\mathrm{\;{ms}}$（显示为 $\star$），并且更接近 ColBERT E2E 的默认响应时间（★）。

Overall, in response to RQ9, we conclude that the Approximate Scoring technique is useful to attain a better balance of effectiveness and efficiency for ColBERT-PRF model, by reducing the number of documents being re-ranked, and can also be combined with the more efficient clustering techniques.

总体而言，针对研究问题9（RQ9），我们得出结论：近似评分技术（Approximate Scoring technique）有助于为ColBERT-PRF模型在有效性和效率之间实现更好的平衡，它通过减少需要重新排序的文档数量来实现这一点，并且还可以与更高效的聚类技术相结合。

## 9 CONCLUSIONS

## 9 结论

This work is the first to propose a contextualised pseudo-relevance feedback mechanism for multiple representation dense retrieval. Based on the feedback documents obtained from the first-pass retrieval, our proposed ColBERT-PRF approach extracts representative feedback embeddings using a clustering technique. It then identifies discriminative embeddings among these representative embeddings and appends them to the query representation. ColBERT-PRF can be effectively applied in both ranking and reranking scenarios, and requires no further neural network training beyond that of ColBERT. Indeed, our passage ranking experimental results - on the TREC 2019 and 2020 Deep Learning track passage ranking query sets - show that our proposed approach can significantly improve the retrieval effectiveness of the state-of-the-art ColBERT dense retrieval approach. In particular, our ColBERT-PRF outperforms ColBERT E2E model by 26% and 10% on TREC 2019 and TREC 2020 passage ranking query sets. Our proposed ColBERT-PRF is a novel and extremely promising approach into applying PRF in dense retrieval. It may also be adaptable to further multiple representation dense retrieval approaches beyond ColBERT. We further validate the effectiveness of the proposed ColBERT-PRF approach on the MSMARCO document ranking task and TREC Robust04 document ranking task, where ColBERT-PRF is observed to exhibit up to 21% and 14% improvements over ColBERT E2E model on TREC 2019 and TREC 2020 document ranking query sets, respectively. Moreover, we investigate ColBERT-PRF variants with different weighting approaches for measuring the usefulness of the expansion embeddings. Finally, in order to trade-off the efficiency and the effectiveness, we explore the efficient variants of ColBERT-PRF using the approximate scoring technique and/or different clustering algorithms, bringing up to ${4.54} \times$ speedup without compromising the retrieval effectiveness.

这项工作首次为多表示密集检索提出了一种上下文伪相关反馈机制。基于首轮检索得到的反馈文档，我们提出的ColBERT-PRF方法使用聚类技术提取具有代表性的反馈嵌入。然后，它在这些代表性嵌入中识别出有区分性的嵌入，并将其附加到查询表示中。ColBERT-PRF可以有效地应用于排序和重排序场景，并且除了ColBERT的训练之外，无需进一步训练神经网络。实际上，我们在TREC 2019和2020深度学习赛道段落排序查询集上的段落排序实验结果表明，我们提出的方法可以显著提高最先进的ColBERT密集检索方法的检索有效性。特别是，我们的ColBERT-PRF在TREC 2019和TREC 2020段落排序查询集上分别比ColBERT端到端（E2E）模型的性能高出26%和10%。我们提出的ColBERT-PRF是一种将伪相关反馈（PRF）应用于密集检索的新颖且极具前景的方法。它也可能适用于除ColBERT之外的其他多表示密集检索方法。我们进一步在MSMARCO文档排序任务和TREC Robust04文档排序任务上验证了所提出的ColBERT-PRF方法的有效性，在TREC 2019和TREC 2020文档排序查询集上，观察到ColBERT-PRF分别比ColBERT E2E模型的性能提高了21%和14%。此外，我们研究了使用不同加权方法来衡量扩展嵌入有用性的ColBERT-PRF变体。最后，为了在效率和有效性之间进行权衡，我们使用近似评分技术和/或不同的聚类算法探索了ColBERT-PRF的高效变体，在不影响检索有效性的情况下实现了高达${4.54} \times$的加速。

In conclusion, the main findings of this work can be summarised as follows:

总之，这项工作的主要发现可以总结如下：

- The pseudo-relevance feedback information from the top-returned documents in multiple representation dense retrieval is beneficial for improving the retrieval effectiveness on passage retrieval (Section 5) and document retrieval (Section 6). Indeed, our proposed pseudo-relevance feedback mechanism can significantly improve the retrieval effectiveness over than ColBERT end-to-end model, the single representation dense retrieval models, as well as most of the baselines for both passage ranking and document ranking tasks;

- 多表示密集检索中返回的前几名文档的伪相关反馈信息有助于提高段落检索（第5节）和文档检索（第6节）的有效性。实际上，我们提出的伪相关反馈机制相比ColBERT端到端模型、单表示密集检索模型以及段落排序和文档排序任务的大多数基线模型，可以显著提高检索有效性；

- Techniques based on statistical information, namely IDF and ICTF, and on embedding coherency, namely Mean Cosine Similarity, can be used to measure the informativeness of expansion embeddings of ColBERT-PRF (Section 7);

- 基于统计信息（即逆文档频率（IDF）和逆簇词频（ICTF））以及嵌入一致性（即平均余弦相似度）的技术可用于衡量ColBERT-PRF扩展嵌入的信息量（第7节）；

- The trade-off of the retrieval effectiveness and efficiency of ColBERT-PRF can be attained using different clustering techniques and/or candidate selection techniques based on approximate scoring (Section 8).

- 可以使用不同的聚类技术和/或基于近似评分的候选选择技术来实现ColBERT-PRF检索有效性和效率之间的权衡（第8节）。

Overall, our work makes it feasible to implement the pseudo-relevance feedback technique in a multiple-representation dense retrieval setting. In particular, the provided extensive experimental results demonstrate the effectiveness of our proposed ColBERT-PRF model. However, how this proposed dense PRF technique can be applied to the single-representation dense retrieval models remains an open problem. In addition, while the performance of most of the queries can benefit from the expansion embeddings, the performance of some of the queries is still degraded. Thus, a more cautious design that applies selective query embedding expansion will likely alleviate this issue. We leave this as one of our future works.

总体而言，我们的工作使得在多表示密集检索环境中实现伪相关反馈技术成为可能。特别是，所提供的大量实验结果证明了我们提出的ColBERT-PRF模型的有效性。然而，所提出的密集伪相关反馈技术如何应用于单表示密集检索模型仍是一个悬而未决的问题。此外，虽然大多数查询的性能可以从扩展嵌入中受益，但一些查询的性能仍然会下降。因此，采用选择性查询嵌入扩展的更谨慎设计可能会缓解这个问题。我们将此作为我们未来的工作之一。

## A APPENDIX

## A 附录

In the following, Appendix A. 1 firstly details how variants of ColBERT-PRF can be implemented with weight token occurrences, specifically using the Bo1 and RM3 query expansion models. These are compared with ColBERT-PRF when implemented using the KMeans clustering technique. Next, in Appendix A.2, we demonstrate the experimental pipelines for ColBERT-PRF.

以下，附录A.1首先详细介绍如何使用加权词元出现次数来实现ColBERT-PRF的变体，具体使用Bo1和RM3查询扩展模型。将这些变体与使用KMeans聚类技术实现的ColBERT-PRF进行比较。接下来，在附录A.2中，我们展示ColBERT-PRF的实验流程。

### A.1 ColBERT-PRF (Bo1 or RM3) Variants

### A.1 ColBERT-PRF（Bo1或RM3）变体

As discussed in Section 4.1, in ColBERT-PRF, embeddings are clustered, rather than the frequency of the corresponding tokens. In this section, we analyse this choice, by separating the clustering from the embedded representation. In particular, we use traditional token counting to measure the informativeness of tokens in the feedback documents, but then expand the query using the corresponding embedded representation of the selected token. Therefore, for these variants, the informativeness of each feedback embedding is measured using the Bo1 or RM3 technique, then the highest informativeness feedback embeddings are selected as the expansion embeddings. For instance, for the ColBERT-PRF (Bo1) implementation, the expansion embeddings are weighted according to the following equation:

如第4.1节所述，在ColBERT-PRF中，是对嵌入进行聚类，而不是对相应词元的频率进行聚类。在本节中，我们通过将聚类与嵌入表示分离来分析这一选择。具体来说，我们使用传统的词元计数来衡量反馈文档中词元的信息量，但随后使用所选词元的相应嵌入表示来扩展查询。因此，对于这些变体，使用Bo1或RM3技术来衡量每个反馈嵌入的信息量，然后选择信息量最高的反馈嵌入作为扩展嵌入。例如，对于ColBERT-PRF（Bo1）实现，扩展嵌入根据以下公式进行加权：

$$
{W}_{\mathrm{{Bo}}1}\left( t\right)  = t{f}_{x}{\log }_{2}\frac{1 + \lambda }{\lambda } + {\log }_{2}\left( {1 + \lambda }\right) , \tag{9}
$$

where $\lambda  = t{f}_{rel}/{N}_{rel}$ , $t{f}_{rel}$ denotes the frequency of (BERT WordPiece) token $t$ in the pseudo-relevant feedback documents and ${N}_{\text{rel }}$ denotes the number of feedback documents. $t{f}_{x}$ denotes the number of unique tokens in the pseudo-relevant document set.

其中 $\lambda  = t{f}_{rel}/{N}_{rel}$ ，$t{f}_{rel}$ 表示（BERT词块，BERT WordPiece）标记 $t$ 在伪相关反馈文档中的频率，${N}_{\text{rel }}$ 表示反馈文档的数量。$t{f}_{x}$ 表示伪相关文档集中唯一标记的数量。

Similarly, for the ColBERT-PRF (RM3) variant, the expansion embeddings are selected using:

类似地，对于ColBERT-PRF（RM3）变体，使用以下方法选择扩展嵌入：

$$
{W}_{\mathrm{{RM}}3}\left( t\right)  = \lambda {\text{ score }}_{\text{exp }}\left( t\right)  + \left( {1 - \lambda }\right) {\text{ score }}_{\text{orig }}\left( t\right) , \tag{10}
$$

where $0 \leq  \lambda  \leq  1$ . score ${}_{\text{orig }}\left( t\right)  = 1$ denotes the weights for the original query embeddings and ${\operatorname{score}}_{\exp }\left( t\right)$ denotes the weights for the expansion embeddings. ${\operatorname{score}}_{\exp }\left( t\right)  = \frac{S\left( t\right) }{\mathop{\sum }\limits_{{d \in  {PRD}}}\mathop{\sum }\limits_{{{t}^{\prime } \in  d}}S\left( {t}^{\prime }\right) }$ , where $S\left( t\right)$ is calculated as follows:

其中 $0 \leq  \lambda  \leq  1$ . score ${}_{\text{orig }}\left( t\right)  = 1$ 表示原始查询嵌入的权重，${\operatorname{score}}_{\exp }\left( t\right)$ 表示扩展嵌入的权重。${\operatorname{score}}_{\exp }\left( t\right)  = \frac{S\left( t\right) }{\mathop{\sum }\limits_{{d \in  {PRD}}}\mathop{\sum }\limits_{{{t}^{\prime } \in  d}}S\left( {t}^{\prime }\right) }$ ，其中 $S\left( t\right)$ 计算如下：

$$
S\left( t\right)  = P\left( {t,{q}_{1},\ldots ,{q}_{\left| q\right| }}\right) 
$$

$$
 = \mathop{\sum }\limits_{{M \in  \mathcal{M}}}P\left( M\right) P\left( {t \mid  M}\right) \mathop{\prod }\limits_{{i = 1}}^{\left| q\right| }P\left( {{q}_{i} \mid  M}\right)  \tag{11}
$$

$$
 = \frac{1}{\# {PRD}}\mathop{\sum }\limits_{{d \in  {PRD}}}\left( {\frac{{tf}\left( {t,d}\right) }{\left| d\right| } \times  \operatorname{MaxSim}\left( {q,d}\right) }\right) 
$$

where $\operatorname{MaxSim}\left( {q,d}\right)  = \mathop{\sum }\limits_{{i = 1}}^{\left| q\right| }\mathop{\max }\limits_{{j = 1,\ldots ,\left| d\right| }}{\phi }_{{q}_{i}}^{T}{\phi }_{{d}_{j}}$ denotes the maximum similarity score and ${PRD}$ is the set of pseudo-relevant feedback documents.

其中 $\operatorname{MaxSim}\left( {q,d}\right)  = \mathop{\sum }\limits_{{i = 1}}^{\left| q\right| }\mathop{\max }\limits_{{j = 1,\ldots ,\left| d\right| }}{\phi }_{{q}_{i}}^{T}{\phi }_{{d}_{j}}$ 表示最大相似度得分，${PRD}$ 是伪相关反馈文档集。

In Table 8, we report the ColBERT-PRF models with various expansion embedding selection techniques on both the TREC 2019 and 2020 query sets. From Table 8, we find that both ColBERT-PRF Ranker and ReRanker with the Bo1 selection technique can outperform the ColBERT E2E model in terms of NDCG@10, MRR@10 and Recall on TREC 2019 while the improvements are not observed on the TREC 2020 query set. In particular, on the TREC 2020 query set, the ColBERT-PRF models with the RM3 expansion embeddings selection approach exhibit lower performance than the ColBERT E2E model. More importantly, we observe that, by comparing the ColBERT-PRF model with the KMeans clustering technique with the Bo1 and RM3 selection variants, the KMeans clustering technique significantly outperforms both the Bo1 and RM3 variants. Figure 13 below shows the impact of $\beta$ for the Bo1 and KMeans variants,in both the ranking and reranking settings, for MAP and NDCG@10.From the figures, it is clear that KMeans always outperforms Bo1,regardless of the setting of $\beta$ .

在表8中，我们报告了在TREC 2019和2020查询集上使用各种扩展嵌入选择技术的ColBERT-PRF模型。从表8中我们发现，采用Bo1选择技术的ColBERT-PRF排序器和重排序器在TREC 2019的NDCG@10、MRR@10和召回率方面都能优于ColBERT端到端（E2E）模型，但在TREC 2020查询集上未观察到这种改进。特别是在TREC 2020查询集上，采用RM3扩展嵌入选择方法的ColBERT-PRF模型的性能低于ColBERT端到端模型。更重要的是，我们观察到，通过将采用KMeans聚类技术的ColBERT-PRF模型与Bo1和RM3选择变体进行比较，KMeans聚类技术明显优于Bo1和RM3变体。下图13显示了 $\beta$ 对Bo1和KMeans变体在排序和重排序设置下的平均准确率均值（MAP）和NDCG@10的影响。从图中可以明显看出，无论 $\beta$ 如何设置，KMeans始终优于Bo1。

<!-- Media -->

Table 8. Comparison of Applying Clustering-based Expansion Embedding Selection vs. Bo1 and RM3 based Expansion Embedding Selection for ColBERT-PRF

表8. ColBERT-PRF基于聚类的扩展嵌入选择与基于Bo1和RM3的扩展嵌入选择的比较

<table><tr><td rowspan="2"/><td colspan="4">TREC 2019 (43 queries)</td><td colspan="4">TREC 2020 (45 queries)</td></tr><tr><td>MAP</td><td>NDCG@10</td><td>MRR@10</td><td>Recall</td><td>MAP</td><td>NDCG@10</td><td>MRR@10</td><td>Recall</td></tr><tr><td>ColBERT E2E (a)</td><td>0.4318</td><td>0.6934</td><td>0.8529</td><td>0.7892</td><td>0.4654</td><td>0.6871</td><td>0.8525</td><td>0.8245</td></tr><tr><td>ColBERT-PRF Ranker (Bo1) (b)</td><td>0.4720</td><td>0.7068</td><td>0.8420</td><td>0.8561</td><td>0.4609</td><td>0.6811</td><td>0.8374</td><td>${0.8589}^{c}$</td></tr><tr><td>ColBERT-PRF Ranker (RM3) (c)</td><td>0.4036</td><td>0.7352</td><td>0.8858</td><td>0.8706</td><td>0.4289</td><td>0.6483</td><td>0.8417</td><td>0.7852</td></tr><tr><td>ColBERT-PRF Ranker (KMeans)</td><td>${\mathbf{{0.5427}}}^{abc}$</td><td>0.7395</td><td>0.8897</td><td>${\mathbf{{0.8711}}}^{ac}$</td><td>${\mathbf{{0.5116}}}^{c}$</td><td>0.7153</td><td>0.8439</td><td>${\mathbf{{0.8837}}}^{abc}$</td></tr><tr><td>ColBERT-PRF ReRanker (Bo1) (d)</td><td>0.4382</td><td>0.7096</td><td>0.8411</td><td>0.7891</td><td>0.4650</td><td>0.6819</td><td>0.8374</td><td>${0.8278}^{c}$</td></tr><tr><td>ColBERT-PRF ReRanker (RM3) (e)</td><td>0.3943</td><td>0.6686</td><td>0.8624</td><td>0.7281</td><td>0.4313</td><td>0.6502</td><td>0.8417</td><td>0.7882</td></tr><tr><td>ColBERT-PRF ReRanker (KMeans)</td><td>${\mathbf{{0.5026}}}^{abc}$</td><td>0.7409</td><td>0.8897</td><td>${\mathbf{{0.7977}}}^{c}$</td><td>0.5063</td><td>0.7161</td><td>0.8439</td><td>${\mathbf{{0.8443}}}^{c}$</td></tr></table>

<table><tbody><tr><td rowspan="2"></td><td colspan="4">2019年文本检索会议（TREC 2019）（43个查询）</td><td colspan="4">2020年文本检索会议（TREC 2020）（45个查询）</td></tr><tr><td>平均准确率均值（MAP）</td><td>前10名归一化折损累积增益（NDCG@10）</td><td>前10名平均倒数排名（MRR@10）</td><td>召回率（Recall）</td><td>平均准确率均值（MAP）</td><td>前10名归一化折损累积增益（NDCG@10）</td><td>前10名平均倒数排名（MRR@10）</td><td>召回率（Recall）</td></tr><tr><td>ColBERT端到端模型（a）</td><td>0.4318</td><td>0.6934</td><td>0.8529</td><td>0.7892</td><td>0.4654</td><td>0.6871</td><td>0.8525</td><td>0.8245</td></tr><tr><td>基于Bo1的ColBERT伪相关反馈排序器（ColBERT - PRF Ranker (Bo1)）（b）</td><td>0.4720</td><td>0.7068</td><td>0.8420</td><td>0.8561</td><td>0.4609</td><td>0.6811</td><td>0.8374</td><td>${0.8589}^{c}$</td></tr><tr><td>基于RM3的ColBERT伪相关反馈排序器（ColBERT - PRF Ranker (RM3)）（c）</td><td>0.4036</td><td>0.7352</td><td>0.8858</td><td>0.8706</td><td>0.4289</td><td>0.6483</td><td>0.8417</td><td>0.7852</td></tr><tr><td>基于K均值的ColBERT伪相关反馈排序器（ColBERT - PRF Ranker (KMeans)）</td><td>${\mathbf{{0.5427}}}^{abc}$</td><td>0.7395</td><td>0.8897</td><td>${\mathbf{{0.8711}}}^{ac}$</td><td>${\mathbf{{0.5116}}}^{c}$</td><td>0.7153</td><td>0.8439</td><td>${\mathbf{{0.8837}}}^{abc}$</td></tr><tr><td>基于Bo1的ColBERT伪相关反馈重排序器（ColBERT - PRF ReRanker (Bo1)）（d）</td><td>0.4382</td><td>0.7096</td><td>0.8411</td><td>0.7891</td><td>0.4650</td><td>0.6819</td><td>0.8374</td><td>${0.8278}^{c}$</td></tr><tr><td>基于RM3的ColBERT伪相关反馈重排序器（ColBERT - PRF ReRanker (RM3)）（e）</td><td>0.3943</td><td>0.6686</td><td>0.8624</td><td>0.7281</td><td>0.4313</td><td>0.6502</td><td>0.8417</td><td>0.7882</td></tr><tr><td>基于K均值的ColBERT伪相关反馈重排序器（ColBERT - PRF ReRanker (KMeans)）</td><td>${\mathbf{{0.5026}}}^{abc}$</td><td>0.7409</td><td>0.8897</td><td>${\mathbf{{0.7977}}}^{c}$</td><td>0.5063</td><td>0.7161</td><td>0.8439</td><td>${\mathbf{{0.8443}}}^{c}$</td></tr></tbody></table>

Superscripts a...e denote significant improvements over the indicated baseline model(s). The highest value in each column is boldfaced.

上标a...e表示相对于指定的基线模型有显著改进。每列中的最高值用粗体表示。

<!-- figureText: ColBERT-PRF Ranker (Bo1) 0.74 ColBERT-PRF Ranker (Bo1) ColBERT-PRF Ranker (KMeans) NDCG@10 0.72 0.70 0.68 0.0 0.8 1.6 2.4 3.2 4.0 4.8 5.6 $\beta$ 0.74 ColBERT-PRF ReRanker (Bo1) ColBERT-PRF ReRanker (KMeans) NDCG@10 0.72 0.70 0.68 0.0 0.8 1.6 2.4 3.2 4.0 4.8 5.6 ColBERT-PRF Ranker (KMeans) 0.50 0.45 0.40 0.0 0.8 1.6 2.4 3.2 4.0 4.8 5.6 $\beta$ 0.500 0.475 0.450 ColBERT-PRF ReRanker (Bo1) ColBERT-PRF ReRanker (KMeans) 0.425 0.400 0.0 0.8 1.6 2.4 3.2 4.0 4.8 5.6 $\beta$ -->

<img src="https://cdn.noedgeai.com/0195b372-2fb6-7dbd-a6e2-deb0bb3cad1c_35.jpg?x=177&y=701&w=1199&h=745&r=0"/>

Fig. 13. Impact of $\beta$ on TREC 2019 query set for ColBERT-PRF Ranker and ReRanker models with Bo1 and RM3 expansion embedding selection techniques in terms of MAP and NDCG@10 performances.

图13. $\beta$对ColBERT - PRF排序器和重排序器模型在TREC 2019查询集上的影响，采用Bo1和RM3扩展嵌入选择技术，以平均准确率均值（MAP）和前10名归一化折损累积增益（NDCG@10）性能衡量。

<!-- Media -->

Thus, we conclude that the KMeans clustering selection technique is more effective than the traditional Bo1 and RM3 selection approaches for the ColBERT-PRF model. This is because the Bo1 and RM3 query expansion techniques rely solely on the word occurrence statistics for selecting expansion embeddings rather than the semantic coherence of the embeddings, and hence select embeddings for tokens that occur frequently, rather than for frequently occurring semantic concepts. Selecting semantically coherent concepts is a key advantage of ColBERT-PRF for a dense retrieval environment.

因此，我们得出结论，对于ColBERT - PRF模型，KMeans聚类选择技术比传统的Bo1和RM3选择方法更有效。这是因为Bo1和RM3查询扩展技术仅依赖于单词出现统计信息来选择扩展嵌入，而不是嵌入的语义连贯性，因此选择的是频繁出现的词元的嵌入，而不是频繁出现的语义概念的嵌入。选择语义连贯的概念是ColBERT - PRF在密集检索环境中的一个关键优势。

### A.2 ColBERT-PRF Pipeline

### A.2 ColBERT - PRF管道

In this appendix, we demonstrate the stages of ColBERT-PRF, when defined as PyTerrier [25, 27] pipelines. In particular, in PyTerrier, the >> operator is used to delineate different stages of a retrieval pipeline. In Listing 1, we portray the experimental pipelines for ColBERT E2E and ColBERT-PRF. The original source code can be found in the PyTerrier_ColBERT repository. ${}^{20}$

在本附录中，我们展示了将ColBERT - PRF定义为PyTerrier [25, 27]管道时的各个阶段。特别是在PyTerrier中，>>运算符用于划分检索管道的不同阶段。在清单1中，我们描绘了ColBERT端到端（E2E）和ColBERT - PRF的实验管道。原始源代码可以在PyTerrier_ColBERT仓库中找到。${}^{20}$

<!-- Media -->

---

#Loading the ColBERT index

from pyterrier_colbert.ranking import ColBERTFactory

pytcolbert = ColBERTFactory("/path/to/checkpoint.dnn", "/path/to/index", "index_name")

#Build the experimental pipeline

def prf(pytcolbert, rerank, fb_docs=3, fb_embs=10, beta=1.0, k=24) -> Transformer:

		#Pipeline for ColBERT E2E: dense_e2e

		dense_e2e = (pytcolbert.set_retrieve(   )

							>> pytcolbert.index_scorer(query_encoded=True, add_ranks=True,

																			batch_size=10000))

		if rerank:

				#Build pipeline for ColBERT-PRF ReRanker

				prf_pipe = (   )

								dense_e2e

								>> ColbertPRF(pytcolbert, k=k, fb_docs=fb_docs,

															fb_embs=fb_embs, beta=beta, return_docs=True)

								>> (pytcolbert.index_scorer(query_encoded=True,

																					add_ranks=True,

																					batch_size=5000) %1000)

						)

		else:

				#Build pipeline for ColBERT-PRF Ranker

				prf_pipe = (   )

								dense_e2e

								>> ColbertPRF(pytcolbert, k=k, fb_docs=fb_docs,

															fb_embs=fb_embs, beta=beta, return_docs=False)

								>> pytcolbert.set_retrieve(query_encoded=True)

								>> (pytcolbert.index_scorer(query_encoded=True,

																					add_ranks=True,

																					batch_size=5000) % 1000)

						)

		return prf_pipe

---

<!-- Media -->

## ACKNOWLEDGMENTS

## 致谢

The authors are thankful to reviewers for their constructive suggestions and comments, as well as Sean MacAvaney and Sasha Petrov for insightful comments and assistance with implementations.

作者感谢审稿人提出的建设性建议和意见，以及肖恩·麦卡瓦尼（Sean MacAvaney）和萨沙·彼得罗夫（Sasha Petrov）提出的深刻见解和在实现方面提供的帮助。

## REFERENCES

## 参考文献

[1] Nasreen Abdul-Jaleel, James Allan, W. Bruce Croft, Fernando Diaz, Leah Larkey, Xiaoyan Li, Mark D. Smucker, and Courtney Wade. 2004. UMass at TREC 2004: Novelty and HARD. In Proceedings of TREC.

[1] 纳斯林·阿卜杜勒 - 贾利勒（Nasreen Abdul - Jaleel）、詹姆斯·艾伦（James Allan）、W. 布鲁斯·克罗夫特（W. Bruce Croft）、费尔南多·迪亚兹（Fernando Diaz）、利亚·拉基（Leah Larkey）、李 Xiaoyan、马克·D. 斯穆克（Mark D. Smucker）和考特尼·韦德（Courtney Wade）。2004年。马萨诸塞大学在TREC 2004：新颖性和HARD。见TREC会议论文集。

[2] Giambattista Amati. 2003. Probability Models for Information Retrieval Based on Divergence from Randomness Ph.D. thesis. University of Glasgow (2003).

[2] 詹巴蒂斯塔·阿马蒂（Giambattista Amati）。2003年。基于与随机性偏离的信息检索概率模型。博士论文。格拉斯哥大学（2003年）。

[3] Giambattista Amati, Claudio Carpineto, and Giovanni Romano. 2004. Query difficulty, robustness, and selective application of query expansion. In Proceedings of ECIR. 127-137.

[3] 詹巴蒂斯塔·阿马蒂（Giambattista Amati）、克劳迪奥·卡尔皮内托（Claudio Carpineto）和乔瓦尼·罗曼诺（Giovanni Romano）。2004年。查询难度、鲁棒性和查询扩展的选择性应用。见ECIR会议论文集。127 - 137页。

[4] Gianni Amati and Cornelis Joost Van Rijsbergen. 2002. Probabilistic models of information retrieval based on measuring the divergence from randomness. ACM Transactions on Information Systems (TOIS) 20, 4 (2002), 357-389.

[4] 詹尼·阿马蒂（Gianni Amati）和科内利斯·约斯特·范·里斯伯根（Cornelis Joost Van Rijsbergen）。2002年。基于测量与随机性偏离的信息检索概率模型。《ACM信息系统汇刊》（TOIS）20卷，第4期（2002年），357 - 389页。

[5] David Arthur and Sergei Vassilvitskii. 2007. K-Means++: The advantages of careful seeding. In Proceedings of SODA. 1027-1035.

[5] 大卫·阿瑟（David Arthur）和谢尔盖·瓦西里维茨基（Sergei Vassilvitskii）。2007年。K - Means++：精心初始化的优势。见SODA会议论文集。1027 - 1035页。

---

<!-- Footnote -->

${}^{20}$ http://github.com/terrierteam/pyterrier_colbert.

${}^{20}$ http://github.com/terrierteam/pyterrier_colbert.

<!-- Footnote -->

---

[6] Guihong Cao, Jian-Yun Nie, Jianfeng Gao, and Stephen Robertson. 2008. Selecting good expansion terms for pseudo-relevance feedback. In Proceedings of SIGIR. 243-250.

[6] 曹桂红（Guihong Cao）、聂建云（Jian - Yun Nie）、高剑锋（Jianfeng Gao）和斯蒂芬·罗伯逊（Stephen Robertson）。2008年。为伪相关反馈选择优质扩展词。见SIGIR会议论文集。243 - 250页。

[7] Nick Craswell, Bhaskar Mitra, Emine Yilmaz, and Daniel Campos. 2021. Overview of the TREC 2020 deep learning track. In Proceedings of TREC.

[7] 尼克·克拉斯韦尔（Nick Craswell）、巴斯卡尔·米特拉（Bhaskar Mitra）、埃米内·伊尔马兹（Emine Yilmaz）和丹尼尔·坎波斯（Daniel Campos）。2021年。TREC 2020深度学习赛道概述。见TREC会议论文集。

[8] Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, and Ellen M. Voorhees. 2020. Overview of the TREC 2019 deep learning track. In Proceedings of TREC.

[8] 尼克·克拉斯韦尔（Nick Craswell）、巴斯卡尔·米特拉（Bhaskar Mitra）、埃米内·伊尔马兹（Emine Yilmaz）、丹尼尔·坎波斯（Daniel Campos）和埃伦·M·沃里斯（Ellen M. Voorhees）。2020年。2019年文本检索会议（TREC）深度学习赛道综述。收录于《文本检索会议论文集》。

[9] Zhuyun Dai and Jamie Callan. 2019. Deeper text understanding for IR with contextual neural language modeling. In Proceedings of SIGIR. 985-988.

[9] 戴竹云（Zhuyun Dai）和杰米·卡伦（Jamie Callan）。2019年。利用上下文神经语言建模实现信息检索（IR）中更深入的文本理解。收录于《信息检索研究与发展会议论文集》（SIGIR）。第985 - 988页。

[10] Zhuyun Dai and Jamie Callan. 2020. Context-aware document term weighting for ad-hoc search. In Proceedings of ${WWW}$ . 1897-1907.

[10] 戴竹云（Zhuyun Dai）和杰米·卡伦（Jamie Callan）。2020年。即席搜索中的上下文感知文档术语加权。收录于${WWW}$会议论文集。第1897 - 1907页。

[11] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of ACL. 4171-4186.

[11] 雅各布·德夫林（Jacob Devlin）、张明伟（Ming-Wei Chang）、肯顿·李（Kenton Lee）和克里斯蒂娜·图托纳娃（Kristina Toutanova）。2019年。BERT：用于语言理解的深度双向变换器预训练。收录于《计算语言学协会年会论文集》（ACL）。第4171 - 4186页。

[12] Fernando Diaz, Bhaskar Mitra, and Nick Craswell. 2016. Query expansion with locally-trained word embeddings. In Proceedings of ACL. 367-377.

[12] 费尔南多·迪亚兹（Fernando Diaz）、巴斯卡尔·米特拉（Bhaskar Mitra）和尼克·克拉斯韦尔（Nick Craswell）。2016年。使用局部训练的词嵌入进行查询扩展。收录于《计算语言学协会年会论文集》（ACL）。第367 - 377页。

[13] Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant. 2021. A white box analysis of ColBERT. In Proceedings of ECIR. 257-263.

[13] 蒂博·福尔马尔（Thibault Formal）、本杰明·皮沃瓦尔斯基（Benjamin Piwowarski）和斯特凡·克兰尚（Stéphane Clinchant）。2021年。ColBERT的白盒分析。收录于《欧洲信息检索会议论文集》（ECIR）。第257 - 263页。

[14] Jiafeng Guo, Yixing Fan, Qingyao Ai, and W. Bruce Croft. 2016. A deep relevance matching model for ad-hoc retrieval. In Proceedings of CIKM. 55-64.

[14] 郭佳峰（Jiafeng Guo）、范宜兴（Yixing Fan）、艾清瑶（Qingyao Ai）和W·布鲁斯·克罗夫特（W. Bruce Croft）。2016年。即席检索的深度相关性匹配模型。收录于《信息与知识管理会议论文集》（CIKM）。第55 - 64页。

[15] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2017. Billion-scale similarity search with GPUs. arXiv preprint arXiv:1702.08734 (2017).

[15] 杰夫·约翰逊（Jeff Johnson）、马蒂亚斯·杜泽（Matthijs Douze）和埃尔韦·热古（Hervé Jégou）。2017年。使用GPU进行十亿级相似度搜索。预印本arXiv:1702.08734（2017年）。

[16] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-domain question answering. In Proceedings of EMNLP. 6769-6781.

[16] 弗拉基米尔·卡尔普欣（Vladimir Karpukhin）、巴拉斯·奥古兹（Barlas Oguz）、闵世文（Sewon Min）、帕特里克·刘易斯（Patrick Lewis）、莱德尔·吴（Ledell Wu）、谢尔盖·叶杜诺夫（Sergey Edunov）、陈丹琦（Danqi Chen）和易文涛（Wen-tau Yih）。2020年。开放域问答的密集段落检索。收录于《自然语言处理经验方法会议论文集》（EMNLP）。第6769 - 6781页。

[17] Omar Khattab and Matei Zaharia. 2020. ColBERT: Efficient and effective passage search via contextualized late interaction over BERT. In Proceedings of SIGIR. 39-48.

[17] 奥马尔·哈塔卜（Omar Khattab）和马特·扎哈里亚（Matei Zaharia）。2020年。ColBERT：通过基于BERT的上下文后期交互实现高效有效的段落搜索。收录于《信息检索研究与发展会议论文集》（SIGIR）。第39 - 48页。

[18] Ilyes Khennak, Habiba Drias, Amine Kechid, and Hadjer Moulai. 2019. Clustering algorithms for query expansion based information retrieval. In Proceedings of ICCI. 261-272.

[18] 伊莱斯·肯纳克（Ilyes Khennak）、哈比巴·德里亚斯（Habiba Drias）、阿明·凯希德（Amine Kechid）和哈杰尔·穆莱（Hadjer Moulai）。2019年。基于查询扩展的信息检索聚类算法。收录于《国际计算智能会议论文集》（ICCI）。第261 - 272页。

[19] Saar Kuzi, Anna Shtok, and Oren Kurland. 2016. Query expansion using word embeddings. In Proceedings of CIKM. 1929-1932.

[19] 萨尔·库齐（Saar Kuzi）、安娜·什托克（Anna Shtok）和奥伦·库兰德（Oren Kurland）。2016年。使用词嵌入进行查询扩展。收录于《信息与知识管理会议论文集》（CIKM）。第1929 - 1932页。

[20] Canjia Li, Yingfei Sun, Ben He, Le Wang, Kai Hui, Andrew Yates, Le Sun, and Jungang Xu. 2018. NPRF: A neural pseudo relevance feedback framework for ad-hoc information retrieval. In Proceedings of EMNLP. 4482-4491.

[20] 李灿佳（Canjia Li）、孙迎飞（Yingfei Sun）、何本（Ben He）、王乐（Le Wang）、惠凯（Kai Hui）、安德鲁·耶茨（Andrew Yates）、孙乐（Le Sun）和徐俊刚（Jungang Xu）。2018年。NPRF：即席信息检索的神经伪相关反馈框架。收录于《自然语言处理经验方法会议论文集》（EMNLP）。第4482 - 4491页。

[21] Canjia Li, Andrew Yates, Sean MacAvaney, Ben He, and Yingfei Sun. 2021. PARADE: Passage Representation Aggregation for Document Reranking. arXiv:2008.09093 [cs.IR].

[21] 李灿佳（Canjia Li）、安德鲁·耶茨（Andrew Yates）、肖恩·麦卡瓦尼（Sean MacAvaney）、何本（Ben He）和孙迎飞（Yingfei Sun）。2021年。PARADE：用于文档重排序的段落表示聚合。预印本arXiv:2008.09093 [计算机科学 - 信息检索（cs.IR）]。

[22] Hang Li, Shengyao Zhuang, Ahmed Mourad, Xueguang Ma, Jimmy Lin, and Guido Zuccon. 2021. Improving query representations for dense retrieval with pseudo relevance feedback: A reproducibility study. In Proceedings of ECIR.

[22] 李航（Hang Li）、庄圣耀（Shengyao Zhuang）、艾哈迈德·穆拉德（Ahmed Mourad）、马学光（Xueguang Ma）、吉米·林（Jimmy Lin）和圭多·祖科恩（Guido Zuccon）。2021年。利用伪相关反馈改进密集检索的查询表示：一项可重复性研究。收录于《欧洲信息检索会议论文集》（ECIR）。

[23] Yi Luan, Jacob Eisenstein, Kristina Toutanova, and Michael Collins. 2021. Sparse, dense, and attentional representations for text retrieval. Transactions of the Association for Computational Linguistics 9 (2021), 329-345.

[23] 栾义（Yi Luan）、雅各布·艾森斯坦（Jacob Eisenstein）、克里斯蒂娜·图托纳娃（Kristina Toutanova）和迈克尔·柯林斯（Michael Collins）。2021年。文本检索的稀疏、密集和注意力表示。《计算语言学协会汇刊》9（2021年），第329 - 345页。

[24] Sean MacAvaney. 2020. OpenNIR: A complete neural ad-hoc ranking pipeline. In Proceedings of WSDM. 845-848.

[24] 肖恩·麦卡瓦尼（Sean MacAvaney）。2020年。OpenNIR：一个完整的神经即席排名管道。见《网络搜索与数据挖掘会议论文集》（Proceedings of WSDM）。第845 - 848页。

[25] Craig Macdonald and Nicola Tonellotto. 2020. Declarative experimentation in information retrieval using PyTerrier. In Proceedings of ICTIR. 161-168.

[25] 克雷格·麦克唐纳（Craig Macdonald）和尼科拉·托内洛托（Nicola Tonellotto）。2020年。使用PyTerrier进行信息检索的声明式实验。见《信息与知识管理国际会议论文集》（Proceedings of ICTIR）。第161 - 168页。

[26] Craig Macdonald and Nicola Tonellotto. 2021. On approximate nearest neighbour selection for multi-stage dense retrieval. In Proceedings of CIKM. 3318-3322.

[26] 克雷格·麦克唐纳（Craig Macdonald）和尼科拉·托内洛托（Nicola Tonellotto）。2021年。关于多级密集检索的近似最近邻选择。见《信息与知识管理国际会议论文集》（Proceedings of CIKM）。第3318 - 3322页。

[27] Craig Macdonald, Nicola Tonellotto, Sean MacAvaney, and Iadh Ounis. 2021. PyTerrier: Declarative experimentation in Python from BM25 to dense retrieval. In Proceedings of CIKM. 4526-4533.

[27] 克雷格·麦克唐纳（Craig Macdonald）、尼科拉·托内洛托（Nicola Tonellotto）、肖恩·麦卡瓦尼（Sean MacAvaney）和伊阿德·乌尼斯（Iadh Ounis）。2021年。PyTerrier：从BM25到密集检索的Python声明式实验。见《信息与知识管理国际会议论文集》（Proceedings of CIKM）。第4526 - 4533页。

[28] Craig Macdonald, Nicola Tonellotto, and Iadh Ounis. 2021. On single and multiple representations in dense passage retrieval. IIR 2021 Workshop (2021).

[28] 克雷格·麦克唐纳（Craig Macdonald）、尼科拉·托内洛托（Nicola Tonellotto）和伊阿德·乌尼斯（Iadh Ounis）。2021年。关于密集段落检索中的单一和多种表示。《信息检索进展国际会议2021研讨会》（IIR 2021 Workshop）（2021年）。

[29] Shahrzad Naseri, Jeffrey Dalton, Andrew Yates, and James Allan. 2021. CEQE: Contextualized embeddings for query expansion. Proceedings of ECIR (2021), 467-482.

[29] 沙赫扎德·纳塞里（Shahrzad Naseri）、杰弗里·道尔顿（Jeffrey Dalton）、安德鲁·耶茨（Andrew Yates）和詹姆斯·艾伦（James Allan）。2021年。CEQE：用于查询扩展的上下文嵌入。《欧洲信息检索会议论文集》（Proceedings of ECIR）（2021年），第467 - 482页。

[30] Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, and Li Deng. 2016. MS MARCO: A human generated machine reading comprehension dataset. In CoCo@ NIPs.

[30] 特里·阮（Tri Nguyen）、米尔·罗森伯格（Mir Rosenberg）、夏·宋（Xia Song）、简峰·高（Jianfeng Gao）、索拉布·蒂瓦里（Saurabh Tiwary）、兰甘·马朱姆德（Rangan Majumder）和李·邓（Li Deng）。2016年。MS MARCO：一个人工生成的机器阅读理解数据集。见《神经信息处理系统大会计算与认知研讨会》（CoCo@ NIPs）。

[31] Rodrigo Nogueira, Zhiying Jiang, and Jimmy Lin. 2020. Document ranking with a pretrained sequence-to-sequence model. arXiv preprint arXiv:2003.06713 (2020).

[31] 罗德里戈·诺盖拉（Rodrigo Nogueira）、蒋志英（Zhiying Jiang）和吉米·林（Jimmy Lin）。2020年。使用预训练的序列到序列模型进行文档排名。预印本arXiv：2003.06713（2020年）。

[32] Rodrigo Nogueira, Jimmy Lin, and AI Epistemic. 2019. From doc2query to docTTTTTquery. Online preprint (2019).

[32] 罗德里戈·诺盖拉（Rodrigo Nogueira）、吉米·林（Jimmy Lin）和AI认知公司（AI Epistemic）。2019年。从doc2query到docTTTTTquery。在线预印本（2019年）。

[33] Rodrigo Nogueira, Wei Yang, Jimmy Lin, and Kyunghyun Cho. 2019. Document expansion by query prediction. arXiv preprint arXiv:1904.08375 (2019).

[33] 罗德里戈·诺盖拉（Rodrigo Nogueira）、杨威（Wei Yang）、吉米·林（Jimmy Lin）和赵京焕（Kyunghyun Cho）。2019年。通过查询预测进行文档扩展。预印本arXiv：1904.08375（2019年）。

[34] Iadh Ounis, Gianni Amati, Vassilis Plachouras, Ben He, Craig Macdonald, and Douglas Johnson. 2005. Terrier information retrieval platform. In Proceedings of ECIR. 517-519.

[34] 伊阿德·乌尼斯（Iadh Ounis）、詹尼·阿马蒂（Gianni Amati）、瓦西利斯·普拉乔拉斯（Vassilis Plachouras）、何本（Ben He）、克雷格·麦克唐纳（Craig Macdonald）和道格拉斯·约翰逊（Douglas Johnson）。2005年。Terrier信息检索平台。见《欧洲信息检索会议论文集》（Proceedings of ECIR）。第517 - 519页。

ColBERT-PRF: Semantic Pseudo-Relevance Feedback

ColBERT - PRF：语义伪相关反馈

[35] Ramith Padaki, Zhuyun Dai, and Jamie Callan. 2020. Rethinking query expansion for BERT reranking. In Proceedings

[35] 拉米斯·帕达基（Ramith Padaki）、戴竹云（Zhuyun Dai）和杰米·卡兰（Jamie Callan）。2020年。重新思考用于BERT重排序的查询扩展。见《会议论文集》

of ECIR. 297-304.

《欧洲信息检索会议论文集》（Proceedings of ECIR）。第297 - 304页。

[36] Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. 2018. Deep contextualized word representations. In Proceedings of NAACL-HLT. 2227-2237.

[36] 马修·E·彼得斯（Matthew E. Peters）、马克·诺伊曼（Mark Neumann）、莫希特·伊耶尔（Mohit Iyyer）、马特·加德纳（Matt Gardner）、克里斯托弗·克拉克（Christopher Clark）、肯顿·李（Kenton Lee）和卢克·泽特尔莫耶（Luke Zettlemoyer）。2018年。深度上下文词表示。见《北美计算语言学协会人类语言技术会议论文集》（Proceedings of NAACL - HLT）。第2227 - 2237页。

[37] Joseph Rocchio. 1971. Relevance feedback in information retrieval. The Smart Retrieval System-experiments in Automatic Document Processing (1971), 313-323.

[37] 约瑟夫·罗基奥（Joseph Rocchio）。1971年。信息检索中的相关反馈。《智能检索系统 - 自动文档处理实验》（The Smart Retrieval System - experiments in Automatic Document Processing）（1971年），第313 - 323页。

[38] Dwaipayan Roy, Sumit Bhatia, and Mandar Mitra. 2019. Selecting discriminative terms for relevance model. In Proceedings of SIGIR. 1253-1256.

[38] 德瓦帕扬·罗伊（Dwaipayan Roy）、苏米特·巴蒂亚（Sumit Bhatia）和曼达尔·米特拉（Mandar Mitra）。2019年。为相关性模型选择有区分性的术语。见《SIGIR会议论文集》。第1253 - 1256页。

[39] Dwaipayan Roy, Debasis Ganguly, Sumit Bhatia, Srikanta Bedathur, and Mandar Mitra. 2018. Using word embeddings for information retrieval: How collection and term normalization choices affect performance. In Proceedings of CIKM. 1835-1838.

[39] 德瓦帕扬·罗伊（Dwaipayan Roy）、德巴西斯·冈古利（Debasis Ganguly）、苏米特·巴蒂亚（Sumit Bhatia）、斯里坎塔·贝达图尔（Srikanta Bedathur）和曼达尔·米特拉（Mandar Mitra）。2018年。使用词嵌入进行信息检索：语料库和术语归一化选择如何影响性能。见《CIKM会议论文集》。第1835 - 1838页。

[40] Dwaipayan Roy, Debjyoti Paul, Mandar Mitra, and Utpal Garain. 2016. Using word embeddings for automatic query expansion. In Proceedings of SIGIR Workshop on Neural Information Retrieval. arXiv:1606.07608.

[40] 德瓦帕扬·罗伊（Dwaipayan Roy）、德布乔蒂·保罗（Debjyoti Paul）、曼达尔·米特拉（Mandar Mitra）和乌特帕尔·加赖恩（Utpal Garain）。2016年。使用词嵌入进行自动查询扩展。见《SIGIR神经信息检索研讨会论文集》。预印本编号：arXiv:1606.07608。

[41] Nicola Tonellotto and Craig Macdonald. 2021. Query embedding pruning for dense retrieval. In Proceedings of CIKM. 3453-3457.

[41] 尼古拉·托内洛托（Nicola Tonellotto）和克雷格·麦克唐纳（Craig Macdonald）。2021年。用于密集检索的查询嵌入剪枝。见《CIKM会议论文集》。第3453 - 3457页。

[42] Junmei Wang, Min Pan, Tingting He, Xiang Huang, Xueyan Wang, and Xinhui Tu. 2020. A pseudo-relevance feedback framework combining relevance matching and semantic matching for information retrieval. Information Processing & Management 57, 6 (2020), 102342.

[42] 王君梅、潘敏、何婷婷、黄翔、王雪燕和涂新会。2020年。一种结合相关性匹配和语义匹配的信息检索伪相关反馈框架。《信息处理与管理》，2020年，第57卷，第6期，第102342页。

[43] Xiao Wang, Craig Macdonald, and Iadh Ounis. 2022. Improving zero-shot retrieval using dense external expansion. Information Processing & Management 59, 5 (2022), 103026.

[43] 王潇、克雷格·麦克唐纳（Craig Macdonald）和伊阿德·乌尼斯（Iadh Ounis）。2022年。使用密集外部扩展改进零样本检索。《信息处理与管理》，2022年，第59卷，第5期，第103026页。

[44] Xiao Wang, Craig Macdonald, Nicola Tonellotto, and Iadh Ounis. 2021. Pseudo-relevance feedback for multiple representation dense retrieval. In Proceedings of ICTIR. 297-306.

[44] 王潇、克雷格·麦克唐纳（Craig Macdonald）、尼古拉·托内洛托（Nicola Tonellotto）和伊阿德·乌尼斯（Iadh Ounis）。2021年。用于多表示密集检索的伪相关反馈。见《ICTIR会议论文集》。第297 - 306页。

[45] Chenyan Xiong, Zhuyun Dai, Jamie Callan, Zhiyuan Liu, and Russell Power. 2017. End-to-end neural ad-hoc ranking with kernel pooling. In Proceedings of SIGIR. 55-64.

[45] 熊晨彦、戴珠云、杰米·卡兰（Jamie Callan）、刘志远和拉塞尔·鲍尔（Russell Power）。2017年。基于核池化的端到端神经临时排序。见《SIGIR会议论文集》。第55 - 64页。

[46] Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed, and Arnold Overwijk. 2021. Approximate nearest neighbor negative contrastive learning for dense text retrieval. In Proceedings of ICLR.

[46] 熊磊、熊晨彦、李烨、邓国峰、刘佳琳、保罗·贝内特（Paul Bennett）、朱奈德·艾哈迈德（Junaid Ahmed）和阿诺德·奥弗维克（Arnold Overwijk）。2021年。用于密集文本检索的近似最近邻负对比学习。见《ICLR会议论文集》。

[47] HongChien Yu, Zhuyun Dai, and Jamie Callan. 2021. PGT: Pseudo relevance feedback using a graph-based transformer. In Proceedings of ECIR. 440-447.

[47] 余宏谦、戴珠云、杰米·卡兰（Jamie Callan）。2021年。PGT：使用基于图的Transformer的伪相关反馈。见《ECIR会议论文集》。第440 - 447页。

[48] HongChien Yu, Chenyan Xiong, and Jamie Callan. 2021. Improving query representations for dense retrieval with pseudo relevance feedback. In Proceedings of CIKM. 3592-3596.

[48] 余宏谦、熊晨彦和杰米·卡兰（Jamie Callan）。2021年。使用伪相关反馈改进密集检索的查询表示。见《CIKM会议论文集》。第3592 - 3596页。

[49] Hamed Zamani and W. Bruce Croft. 2016. Embedding-based query language models. In Proceedings of ICTIR. 147-156.

[49] 哈米德·扎马尼（Hamed Zamani）和W. 布鲁斯·克罗夫特（W. Bruce Croft）。2016年。基于嵌入的查询语言模型。见《ICTIR会议论文集》。第147 - 156页。

[50] Hamed Zamani, Mostafa Dehghani, W. Bruce Croft, Erik Learned-Miller, and Jaap Kamps. 2018. From neural re-ranking to neural ranking: Learning a sparse representation for inverted indexing. In Proceedings of CIKM. 497-506.

[50] 哈米德·扎马尼（Hamed Zamani）、莫斯塔法·德赫加尼（Mostafa Dehghani）、W. 布鲁斯·克罗夫特（W. Bruce Croft）、埃里克·勒纳 - 米勒（Erik Learned - Miller）和雅普·坎普斯（Jaap Kamps）。2018年。从神经重排序到神经排序：学习用于倒排索引的稀疏表示。见《CIKM会议论文集》。第497 - 506页。

[51] Zhi Zheng, Kai Hui, Ben He, Xianpei Han, Le Sun, and Andrew Yates. 2020. BERT-QE: Contextualized query expansion for document re-ranking. In Proceedings of EMNLP: Findings. 4718-4728.

[51] 郑智、惠凯、何本、韩先培、孙乐和安德鲁·耶茨（Andrew Yates）。2020年。BERT - QE：用于文档重排序的上下文查询扩展。见《EMNLP：研究成果》。第4718 - 4728页。

Received 22 February 2022; revised 4 August 2022; accepted 19 September 2022

2022年2月22日收到；2022年8月4日修订；2022年9月19日接受