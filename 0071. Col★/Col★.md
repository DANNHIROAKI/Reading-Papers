# Reproducibility, Replicability, and Insights into Dense Multi-Representation Retrieval Models: from ColBERT to Col★

# 密集多表示检索模型的可复现性、可重复性及见解：从ColBERT到Col★

Xiao Wang

小王

University of Glasgow, UK

英国格拉斯哥大学

x.wang.8@research.gla.ac.uk

Craig Macdonald

克雷格·麦克唐纳

University of Glasgow, UK

英国格拉斯哥大学

craig.macdonald@glasgow.ac.uk

Nicola Tonellotto

尼古拉·托内洛托

University of Pisa, Italy

意大利比萨大学

nicola.tonellotto@unipi.it

Iadh Ounis

伊阿德·乌尼斯

University of Glasgow, UK

英国格拉斯哥大学

iadh.ounis@glasgow.ac.uk

## ABSTRACT

## 摘要

Dense multi-representation retrieval models, exemplified as ColBERT, estimate the relevance between a query and a document based on the similarity of their contextualised token-level embed-dings. Indeed, by using contextualised token embeddings, dense retrieval, conducted as either exact or semantic matches, can result in increased effectiveness for both in-domain and out-of-domain retrieval tasks, indicating that it is an important model to study. However, the exact role that these semantic matches play is not yet well investigated. For instance, although tokenisation is one of the crucial design choices for various pretrained language models, its impact on the matching behaviour has not been examined in detail. In this work, we inspect the reproducibility and replicability of the contextualised late interaction mechanism by extending ColBERT to $\mathrm{{Col}} \star$ ,which implements the late interaction mechanism across various pretrained models and different types of tokenisers. As different tokenisation methods can directly impact the matching behaviour within the late interaction mechanism, we study the nature of matches occurring in different $\mathrm{{Col}} \star$ models,and further quantify the contribution of lexical and semantic matching on retrieval effectiveness. Overall, our experiments successfully reproduce the performance of ColBERT on various query sets, and replicate the late interaction mechanism upon different pretrained models with different tokenisers. Moreover, our experimental results yield new insights, such as: (i) semantic matching behaviour varies across different tokenisers; (ii) more specifically, high-frequency tokens tend to perform semantic matching than other token families; (iii) late interaction mechanism benefits more from lexical matching than semantic matching; (iv) special tokens, such as [CLS], play a very important role in late interaction.

以ColBERT为代表的密集多表示检索模型，基于查询和文档的上下文词元级嵌入的相似度来估计它们之间的相关性。实际上，通过使用上下文词元嵌入，以精确匹配或语义匹配方式进行的密集检索，可以提高领域内和领域外检索任务的有效性，这表明它是一个值得研究的重要模型。然而，这些语义匹配所起的确切作用尚未得到充分研究。例如，尽管词元化是各种预训练语言模型的关键设计选择之一，但其对匹配行为的影响尚未详细研究。在这项工作中，我们通过将ColBERT扩展到$\mathrm{{Col}} \star$来检查上下文后期交互机制的可复现性和可重复性，$\mathrm{{Col}} \star$在各种预训练模型和不同类型的词元化器上实现了后期交互机制。由于不同的词元化方法会直接影响后期交互机制中的匹配行为，我们研究了不同$\mathrm{{Col}} \star$模型中发生的匹配的性质，并进一步量化了词法匹配和语义匹配对检索有效性的贡献。总体而言，我们的实验成功复现了ColBERT在各种查询集上的性能，并在使用不同词元化器的不同预训练模型上复现了后期交互机制。此外，我们的实验结果产生了新的见解，例如：（i）语义匹配行为因不同的词元化器而异；（ii）更具体地说，高频词元比其他词元类别更倾向于进行语义匹配；（iii）后期交互机制从词法匹配中获得的益处比从语义匹配中更多；（iv）特殊词元，如[CLS]，在后期交互中起着非常重要的作用。

## CCS CONCEPTS

## 计算机协会概念分类

- Information systems $\rightarrow$ Retrieval models and ranking.

- 信息系统 $\rightarrow$ 检索模型与排序。

## KEYWORDS

## 关键词

Dense Retrieval; Semantic Matching; Reproducibility

密集检索；语义匹配；可复现性

## ACM Reference Format:

## 美国计算机协会引用格式：

Xiao Wang, Craig Macdonald, Nicola Tonellotto, and Iadh Ounis. 2023. Reproducibility, Replicability, and Insights into Dense Multi-Representation Retrieval Models: from ColBERT to Col★. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '23), July 23-27, 2023, Taipei, Taiwan. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3539618.3591916

小王、克雷格·麦克唐纳（Craig Macdonald）、尼古拉·托内洛托（Nicola Tonellotto）和伊阿德·乌尼斯（Iadh Ounis）。2023年。密集多表示检索模型的可复现性、可重复性及见解：从ColBERT到Col★。收录于第46届国际计算机协会信息检索研究与发展会议（SIGIR '23）论文集，2023年7月23 - 27日，中国台湾台北。美国纽约州纽约市美国计算机协会（ACM），10页。https://doi.org/10.1145/3539618.3591916

## 1 INTRODUCTION

## 1 引言

Ranking is a core task in information retrieval systems. Traditional lexical retrieval models, such as BM25, focus upon exact term matching, where the relevance of a document to a search query is measured by their precise term overlap. However, this causes both vocabulary and semantic mismatch problems during retrieval. On the one hand, there is a high chance that users would formulate their queries using different terms than used in the relevant document(s), as users have no access to the relevant documents prior to the search. For instance, a document describing a 'vehicle' may not be scored highly for a query 'car' - an example of the vocabulary mismatch problem. On the other hand, precise lexical matching cannot distinguish the same word with different senses. For instance, the word 'case' exhibits different meanings when used in phrases like "phone case" or "case study" phrases. Such a polysemous word can cause problems for retrieval.

排序是信息检索系统中的核心任务。传统的词法检索模型，如BM25，侧重于精确的词项匹配，即通过文档与搜索查询的精确词项重叠来衡量文档与查询的相关性。然而，这在检索过程中会导致词汇和语义不匹配的问题。一方面，由于用户在搜索前无法访问相关文档，他们很可能使用与相关文档不同的词项来表述查询。例如，一篇描述“车辆（vehicle）”的文档可能不会因查询“汽车（car）”而获得高分，这是词汇不匹配问题的一个例子。另一方面，精确的词法匹配无法区分具有不同含义的同一个词。例如，“case”一词在“手机壳（phone case）”或“案例研究（case study）”等短语中具有不同的含义。这种多义词会给检索带来问题。

Recently proposed dense retrieval models alleviate the above limitations by encoding the query and document into contextualised embeddings, and have yielded significant improvements over lexical retrieval $\left\lbrack  {{12},{13},{18},{38},{39}}\right\rbrack$ . In dense retrieval,the relevance of a document to a query is estimated according to the inner product of the corresponding contextualised embeddings in the same vector space. Most dense retrieval models encode queries and documents as single-representation embeddings, i.e., a single vector to represent a document or a query. Differently, ColBERT [13] encodes queries and documents into multiple representations, one vector per token. Then, ColBERT employs a late interaction scoring mechanism to estimate a similarity score between the query and document. ColBERT and its late interaction mechanism is an important dense retrieval paradigm as it shows high retrieval effectiveness on in-domain and zero-shot out-of-domain retrieval tasks, while also being flexible to perform other tasks, such as question answering and document retrieval $\left\lbrack  {{13},{17},{29}}\right\rbrack$ . Thus,in this paper,we take a closer look at ColBERT in terms of its "complete" [35] reproducibility (different team, same artefacts) and replicability (different team, different artefacts) [24] and evaluate its performance not only on the original paper used MSMARCO Dev query set but also on both TREC DL 2019 & 2020 query sets. In addition, we conduct several ablation studies to further explain the performance of the model.

最近提出的密集检索模型通过将查询和文档编码为上下文嵌入来缓解上述限制，并且相对于词法检索取得了显著改进 $\left\lbrack  {{12},{13},{18},{38},{39}}\right\rbrack$ 。在密集检索中，根据同一向量空间中相应上下文嵌入的内积来估计文档与查询的相关性。大多数密集检索模型将查询和文档编码为单表示嵌入，即使用单个向量来表示一个文档或一个查询。不同的是，ColBERT [13] 将查询和文档编码为多个表示，每个词元对应一个向量。然后，ColBERT采用后期交互评分机制来估计查询和文档之间的相似度得分。ColBERT及其后期交互机制是一种重要的密集检索范式，因为它在领域内和零样本领域外检索任务中显示出较高的检索效率，同时还能灵活地执行其他任务，如问答和文档检索 $\left\lbrack  {{13},{17},{29}}\right\rbrack$ 。因此，在本文中，我们从其“完全” [35] 可复现性（不同团队，相同制品）和可重复性（不同团队，不同制品） [24] 的角度仔细研究ColBERT，并不仅在原论文使用的MSMARCO开发查询集上评估其性能，还在TREC DL 2019和2020查询集上进行评估。此外，我们进行了几项消融研究，以进一步解释该模型的性能。

---

<!-- Footnote -->

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.SIGIR '23, July 23-27, 2023, Taipei, Taiwan. © 2023 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 978-1-4503-9408-6/23/07...\$15.00 https://doi.org/10.1145/3539618.3591916

允许个人或课堂使用本作品的全部或部分内容制作数字或硬拷贝，无需付费，但前提是这些拷贝不得用于盈利或商业目的，并且拷贝必须带有此声明和第一页的完整引用。必须尊重本作品中除作者之外其他人拥有版权的组件。允许进行带引用的摘要。否则，如需复制、重新发布、上传到服务器或分发给列表，则需要事先获得特定许可和/或支付费用。请向permissions@acm.org请求许可。SIGIR '23，2023年7月23 - 27日，中国台湾台北。© 2023版权归所有者/作者所有。出版权授予美国计算机协会（ACM）。ACM ISBN 978 - 1 - 4503 - 9408 - 6/23/07... 15.00美元https://doi.org/10.1145/3539618.3591916

<!-- Footnote -->

---

Moreover, the matching behaviour performed within the late interaction mechanism includes both lexical and semantic matching, both of which depend on the vocabulary. Sub-word tokenisa-tion is the de-facto standard tokenisation approach in neural IR, due to the advantages of a limited-size vocabulary. Tokenisation algorithms used by common contextualised models include Word-Piece [31], used by BERT [4] and ELECTRA [3], Byte-Pair Encoding (BPE) [32], used by RoBERTa [20], and SentencePiece [14], used by ALBERT [15] and T5 [27]. Different pretrained models, and their different tokenisation algorithms, lead to different embeddings in different representation spaces. In addition, the same type of pre-trained model can often be instantiated in differing sizes (number of layers, etc.), where larger models can be more effective. Going further,we extend ColBERT to $\operatorname{Col} \star$ ,instantiating the late interaction mechanism with various pretrained models using different types of tokenisation techniques. By doing so, we further inspect the replicability [24] of the contextualised late interaction mechanism.

此外，后期交互机制内执行的匹配行为包括词法匹配和语义匹配，这两种匹配都依赖于词汇表。由于有限大小词汇表的优势，子词分词（Sub-word tokenisation）是神经信息检索（neural IR）中事实上的标准分词方法。常见上下文模型使用的分词算法包括BERT [4]和ELECTRA [3]使用的Word-Piece [31]、RoBERTa [20]使用的字节对编码（Byte-Pair Encoding，BPE） [32]，以及ALBERT [15]和T5 [27]使用的SentencePiece [14]。不同的预训练模型及其不同的分词算法会在不同的表示空间中产生不同的嵌入。此外，同一类型的预训练模型通常可以实例化为不同的规模（层数等），其中较大的模型可能更有效。更进一步，我们将ColBERT扩展到$\operatorname{Col} \star$，使用不同类型的分词技术，用各种预训练模型实例化后期交互机制。通过这样做，我们进一步考察上下文后期交互机制的可复制性 [24]。

In practice, when scoring a query-document pair using the late interaction mechanism, some of the matched embeddings will represent the same token - a lexical match - while others may match at a semantic level ('car' vs. 'vehicle'). However, the extent that such semantic matching behaviour occurs among the contextualised representations is still under-investigated. Indeed, it is difficult to disentangle the semantic matching from the dot product operation on the single-representation dense retrieval models, while the multiple-representation dense retrieval paradigm provides more transparency in its ranking mechanism. Thus, in this work, we further inspect the dense matching behaviour in ColBERT as well as the Col★ models with different types of tokenisation methods, attempting to generate new insights behind the late interaction mechanism based retrieval. In particular, we investigate the matching behaviour for different token families and further quantify the contribution of different types of matching behaviour to the retrieval effectiveness.

实际上，在使用后期交互机制对查询 - 文档对进行评分时，一些匹配的嵌入可能表示相同的Token（词法匹配），而另一些可能在语义层面上匹配（例如“汽车（car）”与“车辆（vehicle）”）。然而，在上下文表示中这种语义匹配行为发生的程度仍有待研究。实际上，在单表示密集检索模型的点积运算中很难区分出语义匹配，而多表示密集检索范式在其排序机制上提供了更高的透明度。因此，在这项工作中，我们进一步考察ColBERT以及采用不同分词方法的Col★模型中的密集匹配行为，试图为基于后期交互机制的检索背后产生新的见解。具体而言，我们研究不同Token族的匹配行为，并进一步量化不同类型的匹配行为对检索效果的贡献。

In summary, this work studies the reproducibility and replicabil-ity of the ColBERT model. In addition, it provides new insights by explaining the semantic matching behaviour of the contextualised late interaction mechanism. Our main findings can be summarised in terms of Reproducibility, Replicability and Insights aspects of the contextualised late interaction mechanism, as follows:

综上所述，这项工作研究了ColBERT模型的可重复性和可复制性。此外，它通过解释上下文后期交互机制的语义匹配行为提供了新的见解。我们的主要发现可以从上下文后期交互机制的可重复性、可复制性和见解方面总结如下：

Reproducibility: We investigate the reproducibility of ColBERT by training our own ColBERT models and: (i) we find that we are able to reproduce the results of ColBERT on MSMARCO Dev query set; (ii) in terms of the similarity function for ColBERT, we find that there is no difference between L2-based and Cosine similarity methods for reranking, but the L2 similarity method benefits more in end-to-end settings; (iii) regarding the number of training iterations,we find that ColBERT training becomes stable at around ${150}\mathrm{k}$ iterations with a batch size of 32 However, further training beyond this point still results in a modest increase in retrieval effectiveness.

可重复性：我们通过训练自己的ColBERT模型来研究ColBERT的可重复性，并且：（i）我们发现能够在MSMARCO开发查询集上重现ColBERT的结果；（ii）就ColBERT的相似度函数而言，我们发现基于L2和余弦相似度的重排序方法没有差异，但L2相似度方法在端到端设置中更有益；（iii）关于训练迭代次数，我们发现当批量大小为32时，ColBERT训练在大约${150}\mathrm{k}$次迭代时变得稳定。然而，在此之后继续训练仍会使检索效果有适度提升。

Replicability: We study the effectiveness of multi-representation dense retrieval with different pretrained models with different to-kenisation algorithms and (iv) we find that ColBERT can generalise upon various pretrained language models. (v) in terms of retrieval effectiveness, we find that applying the late interaction mechanism upon a RoBERTa model (which employs BPE tokenisation) exhibits competitive retrieval effectiveness to ColBERT.

可复制性：我们研究了使用不同分词算法的不同预训练模型进行多表示密集检索的有效性，并且（iv）我们发现ColBERT可以在各种预训练语言模型上进行泛化。（v）在检索效果方面，我们发现将后期交互机制应用于RoBERTa模型（采用BPE分词）表现出与ColBERT相当的检索效果。

Insights: Extensive experimental analysis on semantic matching behaviour yields the following new findings: (vi) applying the late interaction mechanism with the BPE tokeniser is more likely to perform semantic matching than the more common ColBERT model; (vii) among various salient token families, all of the contextualised late interaction models perform semantic matching, particularly for low IDF tokens and stopwords tokens; (viii) performing only exact matching and the special token matching contribute more than only semantic matching to the overall retrieval effectiveness. These insights help explain the matching behaviour in contextualised late interaction retrieval and can shed light on the more effective dense retrieval model design and retrieval.

见解：对语义匹配行为进行的广泛实验分析得出以下新发现：（vi）与更常见的ColBERT模型相比，使用BPE分词器的后期交互机制更有可能执行语义匹配；（vii）在各种显著的Token族中，所有上下文后期交互模型都进行语义匹配，特别是对于低逆文档频率（IDF）Token和停用词Token；（viii）仅进行精确匹配和特殊Token匹配对整体检索效果的贡献比仅进行语义匹配更大。这些见解有助于解释上下文后期交互检索中的匹配行为，并为更有效的密集检索模型设计和检索提供启示。

The remainder of this paper is organised as follows: Section 2 describes related work about dense retrieval and tokenisation. We detail the reproducibility and replicability experiments in Section 3 and Section 4, respectively. Next, we explain the semantic matching behaviour of the contextualised late interaction mechanism and generate new insights in Section 5. Finally, we summarise our findings and provide future work directions in Section 6.

本文的其余部分组织如下：第2节描述了关于密集检索和分词的相关工作。我们分别在第3节和第4节详细介绍可重复性和可复制性实验。接下来，我们在第5节解释上下文后期交互机制的语义匹配行为并产生新的见解。最后，我们在第6节总结我们的发现并提供未来的工作方向。

## 2 RELATED WORK

## 2 相关工作

Dense retrieval performs relevance scoring through the encoded contextualised representations of queries and documents. According to the way the queries and the documents are encoded, dense retrieval models can be divided into two families [22]: single representation and multiple representation dense retrieval models. In single representation models, such as DPR [12], ANCE [38] and TCT-ColBERT [19], each query or document as a whole is encoded into a single dense representation. Then the relevance between the query and document is estimated using the dot-product of the encoded vectors. In contrast, in multiple representation dense retrieval models, exemplified by ColBERT [13], each token of the query or document is encoded into a dense representation. To estimate the relevance score of a document to a query, ColBERT implements a two-stage scoring pipeline: in the first stage, an approximate nearest neighbour search produces a set of candidate documents, and in the second stage, these documents are re-ranked with a late interaction mechanism. Recent research has focused on various aspects to improve the quality of dense retrieval models. For instance, some researchers have studied the effect of the negative samples for training more effective dense retrieval models $\left\lbrack  {{19},{25},{38},{39}}\right\rbrack$ ; other researchers have observed that distilling the knowledge from a more effective model, for instance, ColBERT, can result in a more effective single-representation dense retrieval model $\left\lbrack  {9,{10},{18},{19},{35}}\right\rbrack$ . Another thread of work involves reducing the index size of ColBERT, for instance by pruning [16] or compressing embeddings [1, 29]. Most relevant to this work, Formal et al. investigated the term importance captured by ColBERT in the exact and semantic matches [5]. However, they did not investigate the importance of semantic matching, nor the impact of different base models and tokenisation methods. Therefore, our replicability study is important as it examines the generalisation of ColBERT to other pretrained language models and tokenisers. Later, Formal et al. attempted to quantify the importance of lexical matching by examining the frequency of important query tokens in the top-k returned documents [6]. However, it is unclear how strongly this metric correlates with the actual importance of lexical matching. In this work, we propose a semantic matching proportion method that directly measures the extent a query token performs exact or semantic matching to the document.

密集检索通过对查询和文档进行编码后的上下文表示来进行相关性评分。根据对查询和文档的编码方式，密集检索模型可分为两类[22]：单表示和多表示密集检索模型。在单表示模型中，如DPR[12]、ANCE[38]和TCT - ColBERT[19]，每个查询或文档整体被编码为单个密集表示。然后使用编码向量的点积来估计查询和文档之间的相关性。相比之下，在多表示密集检索模型中，以ColBERT[13]为例，查询或文档的每个Token（词元）都被编码为一个密集表示。为了估计文档与查询的相关性得分，ColBERT实现了一个两阶段评分流程：在第一阶段，近似最近邻搜索产生一组候选文档，在第二阶段，使用后期交互机制对这些文档进行重新排序。近期的研究集中在各个方面以提高密集检索模型的质量。例如，一些研究人员研究了负样本对训练更有效的密集检索模型的影响$\left\lbrack  {{19},{25},{38},{39}}\right\rbrack$；其他研究人员观察到，从更有效的模型（例如ColBERT）中提炼知识可以得到更有效的单表示密集检索模型$\left\lbrack  {9,{10},{18},{19},{35}}\right\rbrack$。另一类工作涉及减小ColBERT的索引大小，例如通过剪枝[16]或压缩嵌入[1, 29]。与本工作最相关的是，Formal等人研究了ColBERT在精确匹配和语义匹配中捕获的词项重要性[5]。然而，他们没有研究语义匹配的重要性，也没有研究不同基础模型和分词方法的影响。因此，我们的可重复性研究很重要，因为它考察了ColBERT对其他预训练语言模型和分词器的泛化能力。后来，Formal等人试图通过检查前k个返回文档中重要查询Token（词元）的频率来量化词法匹配的重要性[6]。然而，尚不清楚该指标与词法匹配的实际重要性之间的关联程度有多强。在这项工作中，我们提出了一种语义匹配比例方法，该方法直接测量查询Token（词元）与文档进行精确或语义匹配的程度。

<!-- Media -->

Table 1: Tokenisation for example inputs for 3 tokenisers, corresponding to BERT, ALBERT and RoBERTa respectively.

表1：分别对应BERT、ALBERT和RoBERTa的3种分词器对示例输入的分词结果。

<table><tr><td>Technique</td><td>Example 1</td><td>Example 2</td></tr><tr><td>Sample Text</td><td>casualties in ww2</td><td>Casualties</td></tr><tr><td>WordPiece</td><td>[CLS] casualties in w, ##w ##2 [SEP]</td><td>[CLS] casualties [SEP]</td></tr><tr><td>SentencePiece</td><td>[CLS] _casualties _in _ ww 2 [SEP]</td><td>[CLS] _casualties [SEP]</td></tr><tr><td>BPE</td><td><s> Gcasualties Gin Gw w 2 </s></td><td><s> Cas ual ties</s></td></tr></table>

<table><tbody><tr><td>技术</td><td>示例1</td><td>示例2</td></tr><tr><td>示例文本</td><td>二战（World War II）中的伤亡情况</td><td>伤亡人员；伤亡情况</td></tr><tr><td>词块切分法（WordPiece）</td><td>[CLS] 二战（w, ##w ##2）中的伤亡情况 [SEP]</td><td>[CLS] 伤亡情况 [SEP]</td></tr><tr><td>句子切分法（SentencePiece）</td><td>[CLS] 二战（_ ww 2）中的伤亡情况 [SEP]</td><td>[CLS] 伤亡情况 [SEP]</td></tr><tr><td>字节对编码（BPE）</td><td><s> 二战（Gcasualties Gin Gw w 2）中的伤亡情况 </s></td><td><s> 伤亡人员 </s></td></tr></tbody></table>

<!-- Media -->

Indeed, tokenisation is an important technique to preprocess the input text before input to a contextualised language model. In particular, as transformer-based models learn representations for each unique token, a limited-size vocabulary is important. A large vocabulary size would cause increased memory and time complexity, and difficulty of learning accurate representations for rare tokens. For these reasons, sub-word tokenisation is usually used to split the input text into small chunks of text. Thus, frequently-used words are given unique ids, while rare words will be processed into sub-words. Prevalent tokenisation techniques used by large pretrained language models include WordPiece [31], Byte-Pair Encoding (BPE) [32] and SentencePiece [14] tokenisation techniques. For instance, WordPiece [31] is used by BERT [4] and miniLM [34]; BPE [32] is used by RoBERTa [20] and GPT [26] models; Sentence-Piece [14] is used by ALBERT [15] and T5 [27] models. In particular, the BPE [32] and WordPiece [31] tokenisation technique merge the characters into larger tokens but control the vocabulary size using different algorithms to maximise the likelihood of the training data. In contrast, SentencePiece treats the whole sentence as one large token and learns to split it into sub-words.

实际上，分词（tokenisation）是在将输入文本输入到上下文语言模型之前对其进行预处理的一项重要技术。特别是，由于基于Transformer的模型会为每个唯一的词元（token）学习表示，因此词汇表大小有限很重要。词汇表规模过大会导致内存和时间复杂度增加，并且难以学习稀有词元的准确表示。出于这些原因，子词分词通常用于将输入文本分割成小的文本块。因此，常用词会被赋予唯一的ID，而稀有词则会被处理成子词。大型预训练语言模型常用的分词技术包括WordPiece [31]、字节对编码（Byte-Pair Encoding，BPE） [32] 和SentencePiece [14] 分词技术。例如，BERT [4] 和miniLM [34] 使用了WordPiece [31]；RoBERTa [20] 和GPT [26] 模型使用了BPE [32]；ALBERT [15] 和T5 [27] 模型使用了SentencePiece [14]。特别是，BPE [32] 和WordPiece [31] 分词技术将字符合并成更大的词元，但使用不同的算法控制词汇表大小，以最大化训练数据的似然性。相比之下，SentencePiece将整个句子视为一个大词元，并学习将其分割成子词。

Table 1 compares the outputs of the different tokenisation approaches for the example texts "casualties in ww2" and "Casualties". Firstly, each tokeniser has its own rule to mark the begin and end of the sentence and whether the token is sub-word token or not (## vs. - vs. G). Moreover, we see that all three compared tokenisation techniques can produce tokens of the more frequent words with their surface word form, such as in. However, for the rarer words (ww2), the various tokenisers differ in how they split these words into sub-words and encode as tokens. For instance, WordPiece and BPE produce separate the $\mathrm{w},\mathrm{w}$ and 2 in ww2,while SentencePiece has a token for ww. Notably, RoBERTa's BPE tokeniser is case-sensitive (see also Table 3), and while the vocabulary contains the surface form of casulaties, the uppercase word is broken into three sub-word tokens. This can directly impact the matching behaviour within the late interaction mechanism, as further discussed in Section 5.1.

表1比较了不同分词方法对示例文本“casualties in ww2”和“Casualties”的输出。首先，每个分词器都有自己的规则来标记句子的开头和结尾，以及该词元是否为子词词元（## 与 - 与 G）。此外，我们发现，所比较的三种分词技术都可以将较常用的词以其表面词形式生成为词元，例如“in”。然而，对于较罕见的词（如“ww2”），不同的分词器在如何将这些词分割成子词并编码为词元方面存在差异。例如，WordPiece和BPE会将“ww2”中的 $\mathrm{w},\mathrm{w}$ 和 2 分开，而SentencePiece有一个表示“ww”的词元。值得注意的是，RoBERTa的BPE分词器区分大小写（另见表3），虽然词汇表中包含“casulaties”的表面形式，但大写的“Casualties”一词被拆分成三个子词词元。这会直接影响后期交互机制中的匹配行为，如第5.1节进一步讨论的那样。

Indeed, different tokenisers will directly affect the generated embeddings thus affecting the model performance. For instance, studies have examined different tokenisation techniques for language model pretraining $\left\lbrack  {2,8}\right\rbrack$ and for low-resource language models $\left\lbrack  {{28},{33}}\right\rbrack$ . However,the impact of differing tokenisers for dense retrieval has not been previously investigated. Most recently, ColBERT-X [23] has replaced the BERT pretrained model with the XLM-RoBERTa pretrained model of ColBERT for the cross-language retrieval task. However, ColBERT-X is motivated by the cross-language abilities of the XLM-RoBERTa model and made no conclusions on the effect of the different tokenisation techniques. In this work, we not only investigate the effect of the different pretrained models in ColBERT but also study the effect of using different tokenisation techniques upon English dense retrieval. In addition, we further inspect their impact on the contextualised matching pattern occurring in the dense retrieval models.

实际上，不同的分词器会直接影响生成的嵌入（embeddings），从而影响模型性能。例如，已有研究考察了用于语言模型预训练 $\left\lbrack  {2,8}\right\rbrack$ 和低资源语言模型 $\left\lbrack  {{28},{33}}\right\rbrack$ 的不同分词技术。然而，不同分词器对密集检索的影响此前尚未得到研究。最近，ColBERT - X [23] 在跨语言检索任务中用ColBERT的XLM - RoBERTa预训练模型取代了BERT预训练模型。然而，ColBERT - X的动机是XLM - RoBERTa模型的跨语言能力，并未对不同分词技术的效果得出结论。在这项工作中，我们不仅研究了ColBERT中不同预训练模型的效果，还研究了使用不同分词技术对英语密集检索的影响。此外，我们还进一步考察了它们对密集检索模型中上下文匹配模式的影响。

## 3 REPRODUCIBILITY OF COLBERT

## 3 ColBERT的可复现性

In this section, we first illustrate the late interaction mechanism implemented by ColBERT in Section 3.1, then detail the reproduction results from Section 3.2 to Section 3.4. In particular, the reproduction results address the following research questions: RQ1.1: Can we reproduce the training of ColBERT? (Section 3.2) Going further, we conduct ablations of ColBERT, including: RQ1.2: What is the impact of the similarity function for ColBERT? (Section 3.3) and RQ1.3 Does the model really need to train with full ${200}\mathrm{k}$ iterations (with batch size set as 32)? (Section 3.4). The source code, runs and model checkpoints for all of our experiments are provided in our virtual appendix. ${}^{1}$

在本节中，我们首先在3.1节阐述ColBERT实现的后期交互机制，然后在3.2节至3.4节详细介绍复现结果。特别是，复现结果解决了以下研究问题：RQ1.1：我们能否复现ColBERT的训练过程？（3.2节）进一步地，我们对ColBERT进行了消融实验，包括：RQ1.2：ColBERT的相似度函数有什么影响？（3.3节）以及RQ1.3：模型真的需要以完整的 ${200}\mathrm{k}$ 次迭代（批量大小设置为32）进行训练吗？（3.4节）。我们所有实验的源代码、运行记录和模型检查点都在我们的虚拟附录中提供。${}^{1}$

### 3.1 Contextualised Late Interaction

### 3.1 上下文后期交互

ColBERT consists of a query encoder ${E}_{Q}$ and a document encoder ${E}_{D}$ ,which are fined-tuned based on the pretrained BERT model. For each query $q$ and document $d$ ,the WordPiece tokeniser splits the query text into $\left\{  {{t}_{{q}_{1}},{t}_{{q}_{2}},\cdots {t}_{{q}_{\left| q\right| }}}\right\}$ tokens and the document text into $\left\{  {{t}_{{d}_{1}},{t}_{{d}_{2}},\cdots {t}_{{d}_{\left| d\right| }}}\right\}$ tokens. Then,the series of query and document tokens are encoded by the corresponding encoder into a bag of dense representations $\left\{  {{\phi }_{{q}_{1}},\ldots ,{\phi }_{{q}_{\left| q\right| }}}\right\}$ and $\left\{  {{\phi }_{{d}_{1}},\ldots ,{\phi }_{{d}_{\left| d\right| }}}\right\}$ , respectively. In particular, the number of encoded query tokens is fixed to $\left| q\right|  = {32}$ and filled with the special token ’[MASK]’ if the original query contains less than 32 tokens. Moreover, a linear layer is used to map the BERT representations into a low-dimensional vector with $m$ components,typically $m = {128}$ [13].

ColBERT（列伯特）由一个查询编码器 ${E}_{Q}$ 和一个文档编码器 ${E}_{D}$ 组成，它们基于预训练的BERT（双向编码器表征变换器）模型进行微调。对于每个查询 $q$ 和文档 $d$，WordPiece分词器将查询文本拆分为 $\left\{  {{t}_{{q}_{1}},{t}_{{q}_{2}},\cdots {t}_{{q}_{\left| q\right| }}}\right\}$ 个词元，将文档文本拆分为 $\left\{  {{t}_{{d}_{1}},{t}_{{d}_{2}},\cdots {t}_{{d}_{\left| d\right| }}}\right\}$ 个词元。然后，查询和文档词元序列分别由相应的编码器编码为一组密集表示 $\left\{  {{\phi }_{{q}_{1}},\ldots ,{\phi }_{{q}_{\left| q\right| }}}\right\}$ 和 $\left\{  {{\phi }_{{d}_{1}},\ldots ,{\phi }_{{d}_{\left| d\right| }}}\right\}$。特别地，编码后的查询词元数量固定为 $\left| q\right|  = {32}$，如果原始查询包含的词元少于32个，则用特殊词元 “[MASK]” 填充。此外，使用一个线性层将BERT表示映射到一个具有 $m$ 个分量的低维向量，通常为 $m = {128}$ [13]。

The relevance score of a document $d$ to a query $q$ ,denoted as $S\left( {q,d}\right)$ ,is calculated using a late interaction matching mechanism. The late interaction mechanism is based on the bag of encoded query and document representations, where the maximum similarity score among all the document representations for each query token representation is calculated and then summed to obtain the final relevance score:

文档 $d$ 与查询 $q$ 的相关性得分，记为 $S\left( {q,d}\right)$，使用后期交互匹配机制进行计算。后期交互机制基于编码后的查询和文档表示集合，其中计算每个查询词元表示与所有文档表示之间的最大相似度得分，然后将这些得分相加得到最终的相关性得分：

$$
S\left( {q,d}\right)  = \mathop{\sum }\limits_{{i = 1}}^{\left| q\right| }\mathop{\max }\limits_{{j = 1,\ldots ,\left| d\right| }}\operatorname{Sim}\left( {{\phi }_{{q}_{i}}^{T},{\phi }_{{d}_{j}}}\right) , \tag{1}
$$

where $\operatorname{Sim}\left( {.,.}\right)$ denotes the similarity function used to measure the similarity between query and document embeddings. There are several commonly used similarity functions used for dense retrieval models,namely the L2-based and Cosine similarity ${}^{2}$ functions.

其中 $\operatorname{Sim}\left( {.,.}\right)$ 表示用于衡量查询和文档嵌入之间相似度的相似性函数。有几种常用的相似性函数可用于密集检索模型，即基于L2范数和余弦相似度 ${}^{2}$ 函数。

### 3.2 RQ1.1: Reproduce the Training of ColBERT

### 3.2 研究问题1.1：复现ColBERT（列伯特）的训练过程

The aim of this section is to study the reproducibility of the BERT-based late interaction model, in particular, the ColBERT-v1 [13] model. We note that ColBERT-v2 [29] also uses the same late interaction mechanism of ColBERT-v1 while boosting its retrieval effectiveness by leveraging a number of tricks during training, including periodically mining hard-negative samples from the ColBERT-v2 indices [38], in-batch negative training and performing knowledge distillation from a MiniLM [34] based cross-encoder model. As the efficacy of the above training tricks has been studied in $\left\lbrack  {{18},{19},{25},{35},{38},{39}}\right\rbrack$ ,we focus upon the contextualised late interaction mechanism.

本节的目的是研究基于BERT（双向编码器表征变换器）的后期交互模型的可复现性，特别是ColBERT - v1 [13] 模型。我们注意到，ColBERT - v2 [29] 也使用了与ColBERT - v1相同的后期交互机制，同时在训练过程中利用了一些技巧来提高其检索效果，包括定期从ColBERT - v2索引中挖掘难负样本 [38]、批内负样本训练以及从基于MiniLM [34] 的交叉编码器模型进行知识蒸馏。由于上述训练技巧的有效性已在 $\left\lbrack  {{18},{19},{25},{35},{38},{39}}\right\rbrack$ 中进行了研究，我们将重点关注上下文后期交互机制。

---

<!-- Footnote -->

${}^{1}$ https://github.com/Xiao0728/ColStar_VirtualAppendix ${}^{2}$ This is implemented using the inner product, as the embeddings have been normalised to unit length.

${}^{1}$ https://github.com/Xiao0728/ColStar_VirtualAppendix ${}^{2}$ 这是使用内积实现的，因为嵌入已经被归一化为单位长度。

<!-- Footnote -->

---

<!-- Media -->

Table 2: Reproduction results of ColBERT reranking (rerank on the official top-1000 results produced by BM25) and end-to-end retrieval on MSMARCO Dev Small as well as the TREC DL 2019 and 2020 query sets. The $\dagger$ symbol denotes statistically significant differences over BM25. The highest value in each column is boldfaced.

表2：ColBERT（列伯特）重排序（在BM25生成的官方前1000个结果上进行重排序）以及在MSMARCO Dev Small以及TREC DL 2019和2020查询集上的端到端检索的复现结果。$\dagger$ 符号表示与BM25相比具有统计学显著差异。每列中的最高值用粗体表示。

<table><tr><td rowspan="2">Models</td><td colspan="4">MSMARCO (Dev Small)</td><td colspan="4">TREC DL 19</td><td colspan="4">TREC DL 20</td></tr><tr><td>MRR@10</td><td>R@50</td><td>R@100</td><td>R@1k</td><td>nDCG@10</td><td>MRR@10</td><td>MAP@1k</td><td>R@1k</td><td>nDCG@10</td><td>MRR@10</td><td>MAP@1k</td><td>R@1k</td></tr><tr><td/><td colspan="4">BM25 (official) » Late Interaction</td><td colspan="4">BM25 (PyTerrier) » Late Interaction</td><td colspan="4">BM25 (PyTerrier) » Late Interaction</td></tr><tr><td>BM25 (official)</td><td>0.167</td><td>-</td><td>-</td><td>0.814</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>BM25 (PyTerrier)</td><td>0.196</td><td>0.604</td><td>0.755</td><td>0.871</td><td>0.480</td><td>0.640</td><td>0.286</td><td>0.755</td><td>0.494</td><td>0.615</td><td>0.293</td><td>0.807</td></tr><tr><td>ColBERT-L2 (reported)</td><td>0.348</td><td>0.753</td><td>0.805</td><td>0.814</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>ColBERT-Cosine (reported)</td><td>0.349</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>ColBERT-L2 (ours)</td><td>0.349</td><td>0.754</td><td>0.805</td><td>0.814</td><td>0.713</td><td>0.862</td><td>0.470 †</td><td>0.755</td><td>${0.698} \dagger$</td><td>${0.828} \dagger$</td><td>${0.483} \dagger$</td><td>0.807</td></tr><tr><td>ColBERT-Cosine (ours)</td><td>0.348</td><td>0.753</td><td>0.804</td><td>0.814</td><td>0.713</td><td>${0.847}\dot{7}$</td><td>${0.459} \dagger$</td><td>0.755</td><td>0.707 †</td><td>0.835</td><td>0.484†</td><td>0.807</td></tr><tr><td colspan="13">ANN Search » Late Interaction</td></tr><tr><td>ColBERT-L2 (reported)</td><td>0.367</td><td>0.829</td><td>0.923</td><td>0.968</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>ColBERT-L2 (ours)</td><td>0.361</td><td>0.832</td><td>0.923</td><td>0.965</td><td>0.722 $\dagger$</td><td>0.870 †</td><td>0.462†</td><td>0.823</td><td>${0.685} \dagger$</td><td>${0.823} \dagger$</td><td>$\mathbf{{0.475}} \dagger$</td><td>0.839</td></tr><tr><td>ColBERT-Cosine (ours)</td><td>0.358</td><td>0.823</td><td>0.911</td><td>0.952</td><td>${0.708} \dagger$</td><td>${0.857}\dot{7}$</td><td>${0.445} \dagger$</td><td>0.773</td><td>0.690 †</td><td>0.832</td><td>${0.473} \dagger$</td><td>0.806</td></tr></table>

<table><tbody><tr><td rowspan="2">模型</td><td colspan="4">MSMARCO（开发小数据集）</td><td colspan="4">TREC DL 19（文本检索会议深度学习任务2019年版）</td><td colspan="4">TREC DL 20（文本检索会议深度学习任务2020年版）</td></tr><tr><td>前10名平均倒数排名（MRR@10）</td><td>R@50</td><td>R@100</td><td>前1000名召回率（R@1k）</td><td>前10名归一化折损累积增益（nDCG@10）</td><td>前10名平均倒数排名（MRR@10）</td><td>前1000名平均准确率均值（MAP@1k）</td><td>前1000名召回率（R@1k）</td><td>前10名归一化折损累积增益（nDCG@10）</td><td>前10名平均倒数排名（MRR@10）</td><td>前1000名平均准确率均值（MAP@1k）</td><td>前1000名召回率（R@1k）</td></tr><tr><td></td><td colspan="4">BM25（官方版） » 后期交互</td><td colspan="4">BM25（PyTerrier版） » 后期交互</td><td colspan="4">BM25（PyTerrier版） » 后期交互</td></tr><tr><td>BM25（官方版）</td><td>0.167</td><td>-</td><td>-</td><td>0.814</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>BM25（PyTerrier版）</td><td>0.196</td><td>0.604</td><td>0.755</td><td>0.871</td><td>0.480</td><td>0.640</td><td>0.286</td><td>0.755</td><td>0.494</td><td>0.615</td><td>0.293</td><td>0.807</td></tr><tr><td>ColBERT - L2（报告值）</td><td>0.348</td><td>0.753</td><td>0.805</td><td>0.814</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>ColBERT - 余弦相似度（报告值）</td><td>0.349</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>ColBERT - L2（我们的结果）</td><td>0.349</td><td>0.754</td><td>0.805</td><td>0.814</td><td>0.713</td><td>0.862</td><td>0.470 †</td><td>0.755</td><td>${0.698} \dagger$</td><td>${0.828} \dagger$</td><td>${0.483} \dagger$</td><td>0.807</td></tr><tr><td>ColBERT - 余弦相似度（我们的结果）</td><td>0.348</td><td>0.753</td><td>0.804</td><td>0.814</td><td>0.713</td><td>${0.847}\dot{7}$</td><td>${0.459} \dagger$</td><td>0.755</td><td>0.707 †</td><td>0.835</td><td>0.484†</td><td>0.807</td></tr><tr><td colspan="13">近似最近邻搜索 » 后期交互</td></tr><tr><td>ColBERT - L2（报告值）</td><td>0.367</td><td>0.829</td><td>0.923</td><td>0.968</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>ColBERT - L2（我们的结果）</td><td>0.361</td><td>0.832</td><td>0.923</td><td>0.965</td><td>0.722 $\dagger$</td><td>0.870 †</td><td>0.462†</td><td>0.823</td><td>${0.685} \dagger$</td><td>${0.823} \dagger$</td><td>$\mathbf{{0.475}} \dagger$</td><td>0.839</td></tr><tr><td>ColBERT - 余弦相似度（我们的结果）</td><td>0.358</td><td>0.823</td><td>0.911</td><td>0.952</td><td>${0.708} \dagger$</td><td>${0.857}\dot{7}$</td><td>${0.445} \dagger$</td><td>0.773</td><td>0.690 †</td><td>0.832</td><td>${0.473} \dagger$</td><td>0.806</td></tr></tbody></table>

<!-- Media -->

As the original authors did not release the trained query or document vectors nor a trained ColBERT model, thus we skip the "Last-Meter" and "Last-Mile" [35] reproduction stages. Building upon the original ColBERT paper [13], we conduct a "complete" reproduction [35] of ColBERT by training from scratch using the MSMARCO passage training dataset. ${}^{3}$ The MSMARCO training dataset contains $\sim  {8.8}\mathrm{M}$ passages along with ${0.5}\mathrm{M}$ training queries, each with 1-2 labelled relevant passages. We train ColBERT using both similarity methods for ${200}\mathrm{k}$ iterations with batch size set as 32. Following ColBERT, the retrieval effectiveness is measured on the MSMARCO Dev small query set, which contains 6980 queries with an average of 1.1 relevance judgements per query. We report MRR@10 and Recall with various rank cutoff values (R@50, R@200, R@1k) for MSMARCO Dev Small query set.

由于原作者未发布训练好的查询或文档向量，也未发布训练好的ColBERT模型，因此我们跳过“最后一米”和“最后一英里”[35]的复现阶段。基于原ColBERT论文[13]，我们使用MSMARCO段落训练数据集从头开始训练，对ColBERT进行了“完整”复现[35]。${}^{3}$ MSMARCO训练数据集包含$\sim  {8.8}\mathrm{M}$个段落以及${0.5}\mathrm{M}$个训练查询，每个查询有1 - 2个标注的相关段落。我们使用两种相似度方法对ColBERT进行${200}\mathrm{k}$次迭代训练，批量大小设置为32。按照ColBERT的做法，在MSMARCO Dev小查询集上衡量检索效果，该查询集包含6980个查询，每个查询平均有1.1个相关性判断。我们报告了MSMARCO Dev小查询集的MRR@10和不同排名截断值（R@50、R@200、R@1k）下的召回率。

Results of RQ1.1: Table 2 reports the reproducibility results by training ColBERT using both Cosine and L2 similarity on MS-MARCO Dev Small query set. From the table, we see that for the reranking setting, where ColBERT is applied to rerank the official top 1000 BM25 results, our trained ColBERT-L2 model exhibits negligible differences in terms of all the reported metrics (with a difference up to 0.012 on MRR@10) compared to the reported ColBERT-L2 model. Similarly, results for our trained ColBERT-Cosine model show that we can reproduce the original training of ColBERT for reranking. For the end-to-end setting, our trained ColBERT-L2 model can reproduce the results reported in the original paper.

RQ1.1的结果：表2报告了在MS - MARCO Dev小查询集上使用余弦相似度和L2相似度训练ColBERT的复现结果。从表中可以看出，在重排序设置下（即使用ColBERT对官方前1000个BM25结果进行重排序），与已报告的ColBERT - L2模型相比，我们训练的ColBERT - L2模型在所有报告的指标上差异可忽略不计（MRR@10的差异最大为0.012）。同样，我们训练的ColBERT - 余弦模型的结果表明，我们可以复现ColBERT用于重排序的原始训练。在端到端设置下，我们训练的ColBERT - L2模型可以复现原论文中报告的结果。

Answer to RQ1.1: In summary, we find that we are able to reproduce the results of ColBERT on the MSMARCO Dev Small query set for both reranking and end-to-end scenarios by training our own ColBERT models.

对RQ1.1的回答：总之，我们发现通过训练我们自己的ColBERT模型，能够在MSMARCO Dev小查询集上复现ColBERT在重排序和端到端场景下的结果。

### 3.3 RQ1.2: ColBERT Similarity Functions

### 3.3 RQ1.2：ColBERT相似度函数

For the choice of the vector-similarity method used by the Max-Sim operator in Equation (1), the original ColBERT paper uses Cosine similarity for the reranking setting but uses L2 similarity for the end-to-end setting. The choice of the similarity function, Cosine or L2-based, is important for dense retrieval, as it directly affects the scoring of the query and document vectors. However, the original paper makes no experimental assessment of their impact on retrieval effectiveness. In this section, we evaluate the performance of our trained ColBERT models with different similarity functions not only on MSMARCO Dev Small query set but also on the TREC 2019 & 2020 query sets.

对于公式(1)中Max - Sim算子使用的向量相似度方法的选择，原ColBERT论文在重排序设置中使用余弦相似度，而在端到端设置中使用L2相似度。基于余弦或L2的相似度函数的选择对于密集检索很重要，因为它直接影响查询和文档向量的评分。然而，原论文没有对它们对检索效果的影响进行实验评估。在本节中，我们不仅在MSMARCO Dev小查询集上，还在TREC 2019和2020查询集上评估了我们训练的使用不同相似度函数的ColBERT模型的性能。

<!-- Media -->

<!-- figureText: - ColBERT-L2 —*— ColBERT-Cosine -->

<img src="https://cdn.noedgeai.com/0195a573-0afc-70a3-9912-674101fa6b3f_3.jpg?x=992&y=798&w=588&h=300&r=0"/>

Figure 1: Validation of ColBERT checkpoints on MSMARCO Dev Small in terms of MRR@10.

图1：基于MRR@10对MSMARCO Dev小数据集上的ColBERT检查点进行验证。

<!-- Media -->

Results of RQ1.2: Table 2 and Figure 1 compare the retrieval performance of our trained ColBERT models with L2 similarity as well as Cosine similarity functions. During training, from Figure 1, we see that no marked differences in the Dev Small validation queries were observed. During inference, the evaluation results from Table 2 on MSMARCO Dev Small show comparable results for ColBERT models with different similarity methods for reranking but ColBERT-L2 gives slightly higher performance than ColBERT-Cosine under end-to-end settings. We also observe that there is no significant difference between the two similarity methods for reranking TREC 2019 and 2020 queries. However, for end-to-end retrieval results, we find that L2-based similarity exhibits higher performance than Cosine similarity across all metrics on TREC 2019 queries, as well as on Recall@1k on TREC 2020 queries.

RQ1.2的结果：表2和图1比较了我们训练的使用L2相似度和余弦相似度函数的ColBERT模型的检索性能。从图1可以看出，在训练期间，Dev小验证查询中未观察到明显差异。在推理期间，表2中MSMARCO Dev小数据集的评估结果显示，不同相似度方法的ColBERT模型在重排序方面的结果相当，但在端到端设置下，ColBERT - L2的性能略高于ColBERT - 余弦。我们还观察到，在对TREC 2019和2020查询进行重排序时，两种相似度方法之间没有显著差异。然而，对于端到端检索结果，我们发现基于L2的相似度在TREC 2019查询的所有指标上以及TREC 2020查询的Recall@1k上都表现出比余弦相似度更高的性能。

Answer to RQ1.2: Overall, we find that (i) there is no significant difference between the two similarity functions for reranking; (ii) for end-to-end retrieval, the L2 similarity shows higher performance than the Cosine similarity function.

对RQ1.2的回答：总体而言，我们发现：(i) 两种相似度函数在重排序方面没有显著差异；(ii) 对于端到端检索，L2相似度的性能高于余弦相似度函数。

### 3.4 RQ1.3: Training Iterations (with batch size set as 32) Needed for ColBERT Training

### 3.4 RQ1.3：ColBERT训练所需的训练迭代次数（批量大小设置为32）

The ColBERT paper [13] suggests training all the ColBERT models to ${200}\mathrm{k}$ iterations with a batch size of 32 ,but provides no validation guidance for the early stop of the training. The ColBERT code repository ${}^{4}$ suggests performing validation on the saved checkpoints based on their top $k$ reranking performance before indexing. Here, we perform the validation in the reranking setup for both models the official top-1000 BM25 documents using 1000 MSMARCO Dev Small queries, using the MRR@10 metric.

ColBERT论文[13]建议以32的批量大小对所有ColBERT模型进行${200}\mathrm{k}$次迭代训练，但没有为训练的提前停止提供验证指导。ColBERT代码库${}^{4}$建议在索引之前根据保存的检查点的前$k$个重排序性能对其进行验证。在这里，我们使用1000个MSMARCO Dev小查询，基于MRR@10指标，在重排序设置中对两种模型的官方前1000个BM25文档进行验证。

---

<!-- Footnote -->

3 https://microsoft.github.io/msmarco/

3 https://microsoft.github.io/msmarco/

4 https://github.com/stanford-futuredata/ColBERT/tree/colbertv1

4 https://github.com/stanford-futuredata/ColBERT/tree/colbertv1

<!-- Footnote -->

---

Results of RQ1.3: Figure 1 presents the validation results on the 1000 MSMARCO Dev Small queries. We find that the training process for ColBERT models becomes steady after ${150}\mathrm{k}$ iterations but ColBERT-Cosine achieves the highest validation performance at around ${240}\mathrm{k}$ iterations. However,for ColBERT-L2,the validation curve reaches its highest validation performance at ${280}\mathrm{k}$ iterations.

RQ1.3的结果：图1展示了在1000个MSMARCO开发小查询集上的验证结果。我们发现，ColBERT模型的训练过程在${150}\mathrm{k}$次迭代后趋于稳定，但ColBERT - 余弦（ColBERT - Cosine）在大约${240}\mathrm{k}$次迭代时达到最高验证性能。然而，对于ColBERT - L2，验证曲线在${280}\mathrm{k}$次迭代时达到最高验证性能。

Answer to RQ1.3: We find that ColBERT training converges at around ${150}\mathrm{k}$ iterations with batch size as 32 . Further training leads to slight improvements in terms of MRR@10 validation scores for both ColBERT-L2 and ColBERT-Cosine models. However, to make a fair comparison to the original paper, we report the performance of all ColBERT models trained with ${200}\mathrm{k}$ iterations. To this end,we conclude that we can reproduce the results of ColBERT and obtain new findings from our ablation studies. In the next section, we will begin to examine the replicability of ColBERT.

RQ1.3的答案：我们发现，当批量大小为32时，ColBERT训练在大约${150}\mathrm{k}$次迭代时收敛。进一步训练会使ColBERT - L2和ColBERT - 余弦模型的MRR@10验证分数略有提高。然而，为了与原论文进行公平比较，我们报告了所有经过${200}\mathrm{k}$次迭代训练的ColBERT模型的性能。为此，我们得出结论，我们可以复现ColBERT的结果，并从我们的消融研究中获得新发现。在下一节中，我们将开始研究ColBERT的可复现性。

## 4 REPLICABILITY: FROM COLBERT TO COL★

## 4 可复现性：从ColBERT到Col★

Besides the reproducibility study of ColBERT, we further look into the replicability of ColBERT by generalising the BERT-based contextualised late interaction mechanism upon various different pretrained models,thus forming $\operatorname{Col} \star$ models. Accordingly,we pose our research question for the replicability study as follows: RQ2: How does the retrieval effectiveness vary across different contextualised late interaction models?

除了对ColBERT进行可重复性研究外，我们还通过将基于BERT的上下文后期交互机制推广到各种不同的预训练模型上，进一步研究ColBERT的可复现性，从而形成$\operatorname{Col} \star$模型。因此，我们为可复现性研究提出的研究问题如下：RQ2：不同的上下文后期交互模型的检索效果有何不同？

More specifically,the characteristics of the Col★ models we introduce are summarised in Table 3. The models can be classified within three families, according to the tokenisation technique each model uses, namely WordPiece, BPE and SentencePiece. From the table, we can see that the different base models have different vocabulary sizes and the number of parameters. Moreover, their corresponding ColBERT-like dense indices vary considerably in size.

更具体地说，我们引入的Col★模型的特征总结在表3中。根据每个模型使用的分词技术，这些模型可以分为三类，即WordPiece、BPE和SentencePiece。从表中可以看出，不同的基础模型具有不同的词汇量和参数数量。此外，它们相应的类似ColBERT的密集索引在大小上差异很大。

For the models with the WordPiece tokeniser, we apply the late interaction upon six BERT models with various sized pretrained models, from BERT-Tiny to BERT-Large. The aim of training these variants is to investigate the impact of the number of parameters of the base model that ColBERT encoders are initialised from. In addition, for WordPiece tokeniser models, we also apply ColminiLM and ColELECTRA models. miniLM [34] is a distilled variant of BERT, which aims to reduce the huge number of parameters while retaining BERT's performance. In our work, we use miniLM as a base model for the late interaction dense retrieval mechanism and use $m = {32}$ component embeddings. This thus represents a ColBERT-like setting with minimal time- and space-efficiency overheads [1]. We denote this as ColminiLM. Moreover, ELECTRA [3] has been shown to achieve higher performance than a similar-sized BERT on certain NLP tasks and can be implemented as an effective cross-encoder for reranking $\left\lbrack  {7,{21}}\right\rbrack$ ,but its performance has yet to be ascertained for dense retrieval. We implement the late interaction based on ELECTRA and denote this as ColELECTRA.

对于使用WordPiece分词器的模型，我们在六个不同大小的预训练BERT模型（从BERT - Tiny到BERT - Large）上应用后期交互。训练这些变体的目的是研究ColBERT编码器初始化所基于的基础模型的参数数量的影响。此外，对于WordPiece分词器模型，我们还应用了ColminiLM和ColELECTRA模型。miniLM [34]是BERT的蒸馏变体，旨在减少大量参数，同时保留BERT的性能。在我们的工作中，我们使用miniLM作为后期交互密集检索机制的基础模型，并使用$m = {32}$组件嵌入。这因此代表了一种具有最小时间和空间效率开销的类似ColBERT的设置[1]。我们将其表示为ColminiLM。此外，ELECTRA [3]已被证明在某些自然语言处理任务上比类似大小的BERT具有更高的性能，并且可以作为一种有效的交叉编码器用于重排序$\left\lbrack  {7,{21}}\right\rbrack$，但其在密集检索中的性能尚未确定。我们基于ELECTRA实现后期交互，并将其表示为ColELECTRA。

Secondly, to consider the BPE tokeniser, we train ColRoBERTa with both Base and Large sizes. RoBERTa [20] employs the same model architecture as BERT but exploits the BPE tokeniser, with an increased vocabulary size wrt. ColBERT and ColminiLM. We note that RoBERTa is used as the base model for the ANCE dense retrieval model [38]. We extend the ColBERT model using RoBERTa base model within its BPE tokeniser, denoted as ColRoBERTa.

其次，考虑到BPE分词器，我们训练了基础版和大型版的ColRoBERTa。RoBERTa [20]采用了与BERT相同的模型架构，但使用了BPE分词器，与ColBERT和ColminiLM相比，其词汇量有所增加。我们注意到，RoBERTa被用作ANCE密集检索模型的基础模型[38]。我们在其BPE分词器中使用RoBERTa基础模型扩展了ColBERT模型，记为ColRoBERTa。

<!-- Media -->

<!-- figureText: 0.72 ColBERT-Large ColRoBERTa-Large ColALBERT-XXLarge WordPiece Tokeniser ColBERT-Base ColBERT-Medium ColBERT-Small ColBERT-Mini ColALBERT-Large - ColALBERT-XLarge ColALBERT-Base ColBERT-Tiny -->

<img src="https://cdn.noedgeai.com/0195a573-0afc-70a3-9912-674101fa6b3f_4.jpg?x=963&y=238&w=645&h=389&r=0"/>

Figure 2: The retrieval effectiveness (y-axis: nDCG@10) of Col★ models on TREC 2020 query set. The x-axis shows the number of parameters of the Col★ models. Different markers indicate the tokenisation technique used by the Col★ models.

图2：Col★模型在TREC 2020查询集上的检索效果（y轴：nDCG@10）。x轴显示了Col★模型的参数数量。不同的标记表示Col★模型使用的分词技术。

<!-- Media -->

Similar to miniLM [34], ALBERT [15] aims at reducing the number of parameters of BERT by sharing parameters across transformer layers. In our replicability experiments, we train four ColAL-BERT models by fine-tuning various sized base models, including 'Base', 'Large', 'XLarge' and 'XXLarge' ALBERT models. ColAL-BERT models employ the SentencePiece tokeniser, which allows us a third tokeniser setting.

与miniLM [34]类似，ALBERT [15]旨在通过在Transformer层之间共享参数来减少BERT的参数数量。在我们的可复现性实验中，我们通过微调各种大小的基础模型（包括“基础版”、“大型版”、“超大型版”和“超大版”ALBERT模型）训练了四个ColAL - BERT模型。ColAL - BERT模型使用SentencePiece分词器，这为我们提供了第三种分词器设置。

All the Col★ models listed in Table 3 are trained following the original ColBERT training setup, with a batch size of 32 and the query length and document length are set as 32 and 180 , respectively. Table 3 also provides salient details and statistics of the models and their corresponding indices. In addition, for all the Col★ models, except ColminiLM, we fine-tune the models upto 300k iterations, selecting the final model based on reranking effectiveness on the 2019 queries. For ColminiLM, we use the checkpoint vespa-engine/col-minilm provided by the author of [1] which was trained similarly. Since using the MSMARCO Dev query set for validation is computationally expensive, we used a smaller set of TREC 2019 queries for validation instead. All the Col★ models are trained with the Cosine similarity method.

表3中列出的所有Col★模型均按照原始ColBERT训练设置进行训练，批量大小为32，查询长度和文档长度分别设置为32和180。表3还提供了这些模型及其对应索引的显著细节和统计信息。此外，对于除ColminiLM之外的所有Col★模型，我们将模型微调至300k次迭代，并根据2019年查询的重排序效果选择最终模型。对于ColminiLM，我们使用了文献[1]作者提供的检查点vespa - engine/col - minilm，该检查点采用了类似的训练方式。由于使用MSMARCO开发查询集进行验证的计算成本较高，因此我们改用较小的TREC 2019查询集进行验证。所有Col★模型均使用余弦相似度方法进行训练。

Figure 2 shows the number of parameters and the tokeniser's impact on the retrieval effectiveness of various $\mathrm{{Col}} \star$ models. An ANOVA study indicates that both the number of parameters and the type of tokeniser used have a significant impact on the nDCG@10 scores,at a significance level of $p < {0.05}$ . The performance of the models on natural language understanding tasks tends to improve with an increase in the number of trainable parameters [11], although this is not always the case [40]. Our findings, as displayed in Figure 2, indicate that for BERT-based, ALBERT-based and RoBERTa-based Col★ models, retrieval effectiveness tends to increase with an increase in the number of parameters. It should be noted that larger parameterised models may be more prone to over-fitting and require more computational resources for both training and inference. Additionally, the quality of the training data and the model architecture can also impact the retrieval performance of Col★ models. More importantly, considering the environmentally friendly information retrieval [30],we focus on the Col★ models with different tokenisation techniques and investigate the impact of the tokenisation techniques on semantic matching behaviour. To this end, we select to index ColBERT-Base, ColRoBERTa-Base and ColALBERT-Base models. We also compare with the ColminiLM model, which reduces the embedding dimension from 128 to 32 .

图2展示了参数数量和分词器对各种$\mathrm{{Col}} \star$模型检索效果的影响。一项方差分析研究表明，参数数量和所使用的分词器类型在显著性水平为$p < {0.05}$时，均对nDCG@10分数有显著影响。虽然并非总是如此[40]，但可训练参数数量的增加往往会提升模型在自然语言理解任务上的性能[11]。如图2所示，我们的研究结果表明，对于基于BERT、基于ALBERT和基于RoBERTa的Col★模型，检索效果往往会随着参数数量的增加而提高。需要注意的是，参数更多的模型可能更容易过拟合，并且在训练和推理过程中都需要更多的计算资源。此外，训练数据的质量和模型架构也会影响Col★模型的检索性能。更重要的是，考虑到环保信息检索[30]，我们重点研究了采用不同分词技术的Col★模型，并探究了分词技术对语义匹配行为的影响。为此，我们选择对ColBERT - Base、ColRoBERTa - Base和ColALBERT - Base模型进行索引。我们还将其与ColminiLM模型进行了比较，该模型将嵌入维度从128降低到了32。

<!-- Media -->

Table 3: Characteristics for different $\mathrm{{Col}} \star$ models with contextualised late interaction.

表3：具有上下文延迟交互的不同$\mathrm{{Col}} \star$模型的特征。

<table><tr><td>Col★ Model</td><td>Tokeniser</td><td>Vocab. Size</td><td>Index size</td><td>Embedding Dim.</td><td>Number of Parameters</td><td>HF Base Model</td></tr><tr><td>ColBERT-Base</td><td>WordPiece</td><td>30,522</td><td>373G</td><td>128</td><td>1095M</td><td>bert-base-uncased</td></tr><tr><td>ColBERT-Large</td><td>WordPiece</td><td>30,522</td><td>-</td><td>128</td><td>3353M</td><td>bert-large-uncased</td></tr><tr><td>ColBERT-Tiny</td><td>WordPiece</td><td>30,522</td><td>-</td><td>128</td><td>44M</td><td>bert-tiny</td></tr><tr><td>ColBERT-Mini</td><td>WordPiece</td><td>30,522</td><td>-</td><td>128</td><td>112M</td><td>bert-mini</td></tr><tr><td>ColBERT-Small</td><td>WordPiece</td><td>30,522</td><td>-</td><td>128</td><td>288M</td><td>bert-small</td></tr><tr><td>ColBERT-Medium</td><td>WordPiece</td><td>30,522</td><td>-</td><td>128</td><td>414M</td><td>bert-medium</td></tr><tr><td>ColELECTRA-Base</td><td>WordPiece</td><td>30,522</td><td>-</td><td>128</td><td>1090M</td><td>electra-small-discriminator</td></tr><tr><td>ColminiLM</td><td>WordPiece</td><td>30,522</td><td>64G</td><td>32</td><td>227M</td><td>-</td></tr><tr><td>ColRoBERTa-Base</td><td>BPE</td><td>50,267</td><td>356G</td><td>128</td><td>1247M</td><td>roberta-base</td></tr><tr><td>ColRoBERTa-Large</td><td>BPE</td><td>50,267</td><td>-</td><td>128</td><td>3555M</td><td>roberta-large</td></tr><tr><td>ColALBERT-Base</td><td>SentencePiece</td><td>30,002</td><td>199G</td><td>128</td><td>119M</td><td>albert-base-v2</td></tr><tr><td>ColALBERT-Large</td><td>SentencePiece</td><td>30,002</td><td>-</td><td>128</td><td>218M</td><td>albert-large-v2</td></tr><tr><td>ColALBERT-XLarge</td><td>SentencePiece</td><td>30,002</td><td>-</td><td>128</td><td>631M</td><td>albert-xlarge-v2</td></tr><tr><td>ColALBERT-XXLarge</td><td>SentencePiece</td><td>30,002</td><td>-</td><td>128</td><td>${2275}\mathrm{M}$</td><td>albert-xxlarge-v2</td></tr></table>

<table><tbody><tr><td>Col★ 模型</td><td>分词器（Tokeniser）</td><td>词汇表大小</td><td>索引大小</td><td>嵌入维度</td><td>参数数量</td><td>HF 基础模型</td></tr><tr><td>ColBERT 基础版</td><td>词块切分（WordPiece）</td><td>30,522</td><td>373G</td><td>128</td><td>1095M</td><td>bert-base-uncased</td></tr><tr><td>ColBERT 大模型版</td><td>词块切分（WordPiece）</td><td>30,522</td><td>-</td><td>128</td><td>3353M</td><td>bert-large-uncased</td></tr><tr><td>ColBERT 微小版</td><td>词块切分（WordPiece）</td><td>30,522</td><td>-</td><td>128</td><td>44M</td><td>bert-tiny</td></tr><tr><td>ColBERT 迷你版</td><td>词块切分（WordPiece）</td><td>30,522</td><td>-</td><td>128</td><td>112M</td><td>bert-mini</td></tr><tr><td>ColBERT 小模型版</td><td>词块切分（WordPiece）</td><td>30,522</td><td>-</td><td>128</td><td>288M</td><td>bert-small</td></tr><tr><td>ColBERT 中模型版</td><td>词块切分（WordPiece）</td><td>30,522</td><td>-</td><td>128</td><td>414M</td><td>bert-medium</td></tr><tr><td>ColELECTRA 基础版</td><td>词块切分（WordPiece）</td><td>30,522</td><td>-</td><td>128</td><td>1090M</td><td>electra-small-discriminator</td></tr><tr><td>ColminiLM</td><td>词块切分（WordPiece）</td><td>30,522</td><td>64G</td><td>32</td><td>227M</td><td>-</td></tr><tr><td>ColRoBERTa 基础版</td><td>字节对编码（BPE）</td><td>50,267</td><td>356G</td><td>128</td><td>1247M</td><td>roberta-base</td></tr><tr><td>ColRoBERTa 大模型版</td><td>字节对编码（BPE）</td><td>50,267</td><td>-</td><td>128</td><td>3555M</td><td>roberta-large</td></tr><tr><td>ColALBERT 基础版</td><td>句子切分（SentencePiece）</td><td>30,002</td><td>199G</td><td>128</td><td>119M</td><td>albert-base-v2</td></tr><tr><td>ColALBERT 大模型版</td><td>句子切分（SentencePiece）</td><td>30,002</td><td>-</td><td>128</td><td>218M</td><td>albert-large-v2</td></tr><tr><td>ColALBERT 超大模型版</td><td>句子切分（SentencePiece）</td><td>30,002</td><td>-</td><td>128</td><td>631M</td><td>albert-xlarge-v2</td></tr><tr><td>ColALBERT 超超大模型版</td><td>句子切分（SentencePiece）</td><td>30,002</td><td>-</td><td>128</td><td>${2275}\mathrm{M}$</td><td>albert-xxlarge-v2</td></tr></tbody></table>

<!-- Media -->

### 4.1 RQ2: Retrieval Effectiveness across Col★?

### 4.1 研究问题2：不同Col★模型的检索效果如何？

To understand if the de-facto BERT base model can be replaced for implementing the late interaction mechanism, we deploy the late interaction technique on various contextualised pretrained language models (which also use varying tokenisers).

为了了解在实现后期交互机制时，事实上的BERT基础模型是否可以被替代，我们在各种上下文预训练语言模型（这些模型也使用不同的分词器）上部署了后期交互技术。

Results of RQ2: Table 4 reports the replication results for the selected $\mathrm{{Col}} \star$ models for both the reranking and end-to-end dense retrieval scenarios on both TREC DL 2019 and 2020 query sets.

研究问题2的结果：表4报告了在TREC DL 2019和2020查询集上，针对所选$\mathrm{{Col}} \star$模型在重排序和端到端密集检索场景下的复现结果。

First, we analyse the ColminiLM model, which exploits a lightweight BERT model and uses the identical WordPiece tokeniser as ColBERT. From the reranking results in Table 4, we see that ColminiLM significantly outperforms BM25 and shows comparable performance to ColBERT across the metrics on both TREC 2019 and 2020 query sets, except markedly lower than ColBERT in terms of nDCG@10 on TREC 2019 queries. Similarly, for the end-to-end retrieval experiments, ColminiLM exhibits significant improvements over BM25. However, compared to ColBERT, ColminiLM shows significantly lower MAP, nDCG@10 and Recall on TREC 2019 and significantly lower MAP and Recall on TREC 2020 queries. The lower performance of ColminiLM can be explained in that, as shown in Table 3,it requires much fewer parameters (only ${20}\%$ of the ColBERT parameters). ColminiLM remains promising as it shows comparable nDCG@10 performance on the test queries (TREC 2020) and it has a smaller index size (~17% of the ColBERT index size).

首先，我们分析ColminiLM模型，它采用了轻量级的BERT模型，并使用与ColBERT相同的WordPiece分词器。从表4中的重排序结果可以看出，在TREC 2019和2020查询集的各项指标上，ColminiLM明显优于BM25，并且与ColBERT表现相当，但在TREC 2019查询的nDCG@10指标上明显低于ColBERT。同样，在端到端检索实验中，ColminiLM相较于BM25有显著改进。然而，与ColBERT相比，ColminiLM在TREC 2019查询上的平均准确率均值（MAP）、nDCG@10和召回率显著较低，在TREC 2020查询上的MAP和召回率也显著较低。ColminiLM性能较低的原因可以解释为，如表3所示，它所需的参数要少得多（仅为ColBERT参数的${20}\%$）。ColminiLM仍然很有前景，因为它在测试查询（TREC 2020）上的nDCG@10性能相当，并且索引大小更小（约为ColBERT索引大小的17%）。

Next, we analyse ColRoBERTa. We find that ColRoBERTa exhibits comparable retrieval effectiveness to ColBERT and markedly improvements over BM25 when employed as a reranker on top of the BM25 sparse retrieval across all the reported metrics. In addition, it shows comparable performance wrt. ColBERT in the dense end-to-end retrieval scenario on TREC 2019 and 2020 queries, except MAP on TREC 2020 query set. Overall, we find that ColRoBERTa is a good replacement of ColBERT.

接下来，我们分析ColRoBERTa。我们发现，当ColRoBERTa作为BM25稀疏检索之上的重排序器时，在所有报告的指标上，它的检索效果与ColBERT相当，并且相较于BM25有显著改进。此外，在TREC 2019和2020查询的端到端密集检索场景中，除了TREC 2020查询集的MAP指标外，它与ColBERT的表现相当。总体而言，我们发现ColRoBERTa是ColBERT的一个很好的替代模型。

For ColALBERT, we observe that it shows lower performance than ColBERT across all the reported metrics on both reranking and end-to-end dense retrieval implementations on both query sets. Similar to ColminiLM, ColALBERT has significantly fewer parameters and a simplified model structure than ColBERT. Overall, ColALBERT has low performance in terms of the precision metrics: MAP, nDCG@10 and MRR@10, and surprisingly high performance in terms of the Recall@1k.

对于ColALBERT，我们观察到，在两个查询集的重排序和端到端密集检索实现中，在所有报告的指标上，它的性能都低于ColBERT。与ColminiLM类似，ColALBERT的参数明显少于ColBERT，并且模型结构更简单。总体而言，ColALBERT在精度指标（如MAP、nDCG@10和MRR@10）方面表现较低，但在Recall@1k指标上表现出奇地高。

Finally, it is notable that, at least on this query set, the other model families consistently do not outperform the BERT family. This suggests that more recent families of pretrained language model (ALBERT, RoBERTa) have not equated to improvements in a downstream retrieval task compared to the original BERT model.

最后，值得注意的是，至少在这个查询集上，其他模型家族始终没有超过BERT家族。这表明，与原始的BERT模型相比，更新的预训练语言模型家族（如ALBERT、RoBERTa）在下游检索任务中并没有带来性能提升。

Answer to RQ2: We conclude that we can replicate the con-textualised late interaction mechanism upon various pretrained models. More specifically, we find that, when compared to the ColBERT model, ColRoBERTa exhibits a competitive performance to ColBERT. However, consistent with the findings from Figure 2, we find that the ColminiLM and ColALBERT models show slightly lower retrieval effectiveness than ColBERT due to their lightweight model structures. Notably, no model family exceeds BERT in terms of effectiveness for a comparable number of parameters.

研究问题2的答案：我们得出结论，我们可以在各种预训练模型上复现上下文后期交互机制。更具体地说，我们发现，与ColBERT模型相比，ColRoBERTa的性能与ColBERT具有竞争力。然而，与图2的结果一致，我们发现ColminiLM和ColALBERT模型由于其轻量级的模型结构，检索效果略低于ColBERT。值得注意的是，在参数数量相当的情况下，没有一个模型家族在效果上超过BERT。

## 5 INSIGHTS: SEMANTIC MATCHING

## 5 洞察：语义匹配

The success of reproducibility and replicability of ColBERT motivates us to investigate the semantic matching behaviour to generate more insights. Thus, to examine more deeply how the different con-textualised late interaction models perform retrieval, we turn to investigate their semantic matching behaviour. In particular, in this section, we introduce an approach to measure the semantic contribution to relevance scoring of documents with contextualised late interaction models. Then, we conduct experiments to address the following research questions: RQ3.1: How does the semantic matching behaviour vary across different contextualised late interaction models? (Section 5.1) RQ3.2: Can we characterise the salient token families of matches, i.e., which type of tokens contribute the most to semantic matching? (Section 5.2) and RQ3.3: Can we quantify the contribution of different types of matching behaviour, namely the lexical match and semantic match as well as special token match, to the retrieval effectiveness? (Section 5.3)

ColBERT的可重复性和可复现性的成功促使我们研究语义匹配行为，以获得更多洞察。因此，为了更深入地研究不同的上下文后期交互模型如何进行检索，我们转而研究它们的语义匹配行为。具体而言，在本节中，我们介绍一种方法来衡量上下文后期交互模型对文档相关性评分的语义贡献。然后，我们进行实验来解决以下研究问题：研究问题3.1：不同的上下文后期交互模型的语义匹配行为有何不同？（第5.1节）研究问题3.2：我们能否刻画显著匹配的词元家族，即哪种类型的词元对语义匹配贡献最大？（第5.2节）以及研究问题3.3：我们能否量化不同类型的匹配行为（即词汇匹配、语义匹配以及特殊词元匹配）对检索效果的贡献？（第5.3节）

Figure 3 illustrates the contextualised late interaction mechanism among a query and a document for ColBERT (left) and ColRoBERTa (right) models. For every query token,on the columns,a X marks the matching document tokens with the highest similarity score, hence contributing to the final relevance score, as in Equation (1). For ColBERT, query tokens such as the, w, and ##w exact match with lexically identical document tokens. At the same time, semantic matching behaviour occurs with the query tokens why and enter, matching with document tokens because and entered, respectively. However, the late interaction for ColRoBERTa produces different token forms and different lexical and semantic matches with document tokens and some query tokens. Thus, we observe that the base model and the tokenisation algorithm not only affect the model size (c.f. Table 3), but, more importantly, they impact the way the matching between queries and documents is conducted within the late interaction mechanism.

图3展示了ColBERT（左）和ColRoBERTa（右）模型中查询与文档之间的上下文相关后期交互机制。对于每一个查询词元（位于列中），“X”标记出了相似度得分最高的匹配文档词元，因此这些词元会对最终的相关性得分产生影响，如公式（1）所示。对于ColBERT而言，像“the”“w”和“##w”这样的查询词元会与词法上相同的文档词元进行精确匹配。同时，查询词元“why”和“enter”分别与文档词元“because”和“entered”发生语义匹配行为。然而，ColRoBERTa的后期交互会产生不同的词元形式，以及与文档词元和部分查询词元不同的词法和语义匹配。因此，我们观察到基础模型和词元化算法不仅会影响模型大小（参见表3），更重要的是，它们会影响后期交互机制中查询与文档之间的匹配方式。

<!-- Media -->

Table 4: Performance of contextualised late interaction models. The ${}^{ + }\left( \diamond \right)$ symbol denotes statistically significant differences compared to BM25 (ColBERT). The highest value in each column is boldfaced.

表4：上下文相关后期交互模型的性能。${}^{ + }\left( \diamond \right)$符号表示与BM25（ColBERT）相比具有统计学上的显著差异。每列中的最高值用粗体表示。

<table><tr><td rowspan="2">Models</td><td colspan="5">TREC DL 2019</td><td colspan="5">TREC DL 2020</td></tr><tr><td>MAP@1k</td><td>nDCG@10</td><td>MRR@10</td><td>R@1k</td><td>Mean SMP</td><td>MAP@1k</td><td>nDCG@10</td><td>MRR@10</td><td>R@1k</td><td>Mean SMP</td></tr><tr><td>BM25 (PyTerrier)</td><td>0.286</td><td>0.480</td><td>0.640</td><td>0.755</td><td>-</td><td>0.293</td><td>0.494</td><td>0.615</td><td>0.807</td><td>-</td></tr><tr><td colspan="11">BM25 » Late Interaction</td></tr><tr><td>ColBERT</td><td>0.459</td><td>${0.713} \dagger$</td><td>${0.847}\dot{ \dagger  }$</td><td>0.755</td><td>0.375</td><td>0.484†</td><td>0.707 †</td><td>${0.835} \dagger$</td><td>0.807</td><td>0.387</td></tr><tr><td>ColminiLM</td><td>${0.431} \dagger$</td><td>0.654</td><td>${0.811}\dot{1}$</td><td>0.755</td><td>0.362</td><td>${0.458} \dagger$</td><td>${0.685} \dagger$</td><td>0.866 $\dagger$</td><td>0.807</td><td>0.363</td></tr><tr><td>ColRoBERTa</td><td>0.458</td><td>0.695</td><td>0.865†</td><td>0.755</td><td>0.599</td><td>${0.462}\dot{ \dagger  }$</td><td>${0.695}\dot{ + }$</td><td>0.844 $\dagger$</td><td>0.807</td><td>0.607</td></tr><tr><td>ColALBERT</td><td>0.412%</td><td>0.634</td><td>${0.821} \dagger$</td><td>0.755</td><td>0.367</td><td>0.401%</td><td>0.630 $\dagger  \diamond$</td><td>0.751 $\dagger$</td><td>0.807</td><td>0.390</td></tr><tr><td colspan="11">ANN Search » Late Interaction</td></tr><tr><td>ColBERT</td><td>0.4451</td><td>0.708</td><td>${0.857} \dagger$</td><td>0.773</td><td>0.390</td><td>0.473†</td><td>0.690 †</td><td>0.832†</td><td>0.806</td><td>0.406</td></tr><tr><td>ColminiLM</td><td>0.388%</td><td>0.631†◇</td><td>${0.811} \dagger$</td><td>0.698 %</td><td>0.382</td><td>0.434</td><td>0.672†</td><td>0.860+</td><td>0.762◥</td><td>0.388</td></tr><tr><td>ColRoBERTa</td><td>0.426</td><td>${0.684} \dagger$</td><td>0.866†</td><td>0.738</td><td>0.610</td><td>0.423</td><td>0.666†</td><td>${0.828}\overset{ + }{7}$</td><td>0.760</td><td>0.622</td></tr><tr><td>ColALBERT</td><td>0.356</td><td>0.613</td><td>0.769</td><td>0.772</td><td>0.381</td><td>0.367†◇</td><td>0.604 $\dagger  \diamond$</td><td>${0.745} \dagger$</td><td>0.792</td><td>0.413</td></tr></table>

<table><tbody><tr><td rowspan="2">模型</td><td colspan="5">文本检索会议深度学习任务2019年版（TREC DL 2019）</td><td colspan="5">文本检索会议深度学习任务2020年版（TREC DL 2020）</td></tr><tr><td>前1000个结果的平均准确率均值（MAP@1k）</td><td>前10个结果的归一化折损累积增益（nDCG@10）</td><td>前10个结果的平均倒数排名（MRR@10）</td><td>前1000个结果的召回率（R@1k）</td><td>平均语义匹配精度（Mean SMP）</td><td>前1000个结果的平均准确率均值（MAP@1k）</td><td>前10个结果的归一化折损累积增益（nDCG@10）</td><td>前10个结果的平均倒数排名（MRR@10）</td><td>前1000个结果的召回率（R@1k）</td><td>平均语义匹配精度（Mean SMP）</td></tr><tr><td>二元独立模型25（BM25，基于PyTerrier）</td><td>0.286</td><td>0.480</td><td>0.640</td><td>0.755</td><td>-</td><td>0.293</td><td>0.494</td><td>0.615</td><td>0.807</td><td>-</td></tr><tr><td colspan="11">二元独立模型25（BM25） » 后期交互</td></tr><tr><td>ColBERT模型</td><td>0.459</td><td>${0.713} \dagger$</td><td>${0.847}\dot{ \dagger  }$</td><td>0.755</td><td>0.375</td><td>0.484†</td><td>0.707 †</td><td>${0.835} \dagger$</td><td>0.807</td><td>0.387</td></tr><tr><td>ColminiLM模型</td><td>${0.431} \dagger$</td><td>0.654</td><td>${0.811}\dot{1}$</td><td>0.755</td><td>0.362</td><td>${0.458} \dagger$</td><td>${0.685} \dagger$</td><td>0.866 $\dagger$</td><td>0.807</td><td>0.363</td></tr><tr><td>ColRoBERTa模型</td><td>0.458</td><td>0.695</td><td>0.865†</td><td>0.755</td><td>0.599</td><td>${0.462}\dot{ \dagger  }$</td><td>${0.695}\dot{ + }$</td><td>0.844 $\dagger$</td><td>0.807</td><td>0.607</td></tr><tr><td>ColALBERT模型</td><td>0.412%</td><td>0.634</td><td>${0.821} \dagger$</td><td>0.755</td><td>0.367</td><td>0.401%</td><td>0.630 $\dagger  \diamond$</td><td>0.751 $\dagger$</td><td>0.807</td><td>0.390</td></tr><tr><td colspan="11">近似最近邻搜索（ANN Search） » 后期交互</td></tr><tr><td>ColBERT模型</td><td>0.4451</td><td>0.708</td><td>${0.857} \dagger$</td><td>0.773</td><td>0.390</td><td>0.473†</td><td>0.690 †</td><td>0.832†</td><td>0.806</td><td>0.406</td></tr><tr><td>ColminiLM模型</td><td>0.388%</td><td>0.631†◇</td><td>${0.811} \dagger$</td><td>0.698 %</td><td>0.382</td><td>0.434</td><td>0.672†</td><td>0.860+</td><td>0.762◥</td><td>0.388</td></tr><tr><td>ColRoBERTa模型</td><td>0.426</td><td>${0.684} \dagger$</td><td>0.866†</td><td>0.738</td><td>0.610</td><td>0.423</td><td>0.666†</td><td>${0.828}\overset{ + }{7}$</td><td>0.760</td><td>0.622</td></tr><tr><td>ColALBERT模型</td><td>0.356</td><td>0.613</td><td>0.769</td><td>0.772</td><td>0.381</td><td>0.367†◇</td><td>0.604 $\dagger  \diamond$</td><td>${0.745} \dagger$</td><td>0.792</td><td>0.413</td></tr></tbody></table>

<!-- figureText: [CLS] [D] G Gthe GUSA Gentered Gbecause Gpearl </s> because [SEP] -->

<img src="https://cdn.noedgeai.com/0195a573-0afc-70a3-9912-674101fa6b3f_6.jpg?x=176&y=815&w=676&h=439&r=0"/>

Figure 3: Late interaction diagrams for ColBERT and Col-RoBERTa models between the query: why did the us voluntarily enter ww1 and the document: the usa entered ww2 because of pearl harbor. For each column, the heatmap indicates the similarity scores among all the document embeddings for each query embedding, where the highest similarity score is highlighted with the symbol X. The top histogram depicts the magnitude of the contribution of the maximum similarity of each query embedding for the final relevance score between the query and document. The [MASK] tokens are omitted.

图3：ColBERT和Col - RoBERTa模型在查询“美国为何自愿参加第一次世界大战”与文档“美国因珍珠港事件参加了第二次世界大战”之间的后期交互图。对于每一列，热力图表示每个查询嵌入与所有文档嵌入之间的相似度得分，其中最高相似度得分用符号X突出显示。顶部的直方图描绘了每个查询嵌入的最大相似度对查询与文档之间最终相关性得分的贡献程度。[MASK]标记（Token）被省略。

<!-- Media -->

Indeed, a benefit of late interaction over multiple contextualised dense representations is that we can investigate the lexical and semantic matching behaviour. To do so, we use a recently proposed technique to quantify the extent of the semantic matching during the contextualised late interaction mechanism [36, 37]. More specifically, among the query-document token pairs contributing to the similarity score computation in late interaction, lexical matching corresponds to the contributing pairs with identical tokens, while semantic matching corresponds to different contributing query-document tokens pairs (e.g. why with because). Formally, given a query $q$ and the list ${R}_{k}$ of the top-ranked $k$ passages,the Semantic Match Proportion (SMP) at rank cutoff $k$ wrt. $q$ and ${R}_{k}$ is defined as:

实际上，基于多个上下文密集表示的后期交互的一个好处是，我们可以研究词汇和语义匹配行为。为此，我们使用最近提出的一种技术来量化上下文后期交互机制中的语义匹配程度[36, 37]。更具体地说，在后期交互中对相似度得分计算有贡献的查询 - 文档标记（Token）对中，词汇匹配对应于具有相同标记（Token）的贡献对，而语义匹配对应于不同的贡献查询 - 文档标记（Token）对（例如，“why”与“because”）。形式上，给定一个查询$q$和排名前$k$的段落列表${R}_{k}$，相对于$q$和${R}_{k}$，在排名截断$k$处的语义匹配比例（SMP）定义为：

$$
\operatorname{SMP}\left( {q,{R}_{k}}\right)  = \mathop{\sum }\limits_{{d \in  {R}_{k}}}\frac{\mathop{\sum }\limits_{{i \in  \operatorname{toks}\left( q\right) }}\mathbb{1}\left\lbrack  {{t}_{i} \neq  {t}_{j}}\right\rbrack   \cdot  \mathop{\max }\limits_{{j = 1,\ldots ,\left| d\right| }}{\phi }_{{q}_{i}}^{T}{\phi }_{{d}_{j}}}{\mathop{\sum }\limits_{{i \in  \operatorname{toks}\left( q\right) }}\mathop{\max }\limits_{{j = 1,\ldots ,\left| d\right| }}{\phi }_{{q}_{i}}^{T}{\phi }_{{d}_{j}}}, \tag{2}
$$

where toks(q)contains the indices of the query embeddings that correspond to the tokens produced by the tokeniser,and ${t}_{i}$ and ${t}_{j}$ denote the token ids of the $i$ -th query embedding and $j$ -th passage embedding, respectively. Special tokens such as [SEP], [CLS], [Q] and [MASK] for the WordPiece-based models always match semantically. Therefore, we exclude these special tokens when measuring the SMP value for a model. However, we revisit and quantify their contribution to the retrieval effectiveness in Section 5.3.

其中toks(q)包含对应于分词器生成的标记（Token）的查询嵌入的索引，${t}_{i}$和${t}_{j}$分别表示第$i$个查询嵌入和第$j$个段落嵌入的标记（Token）ID。基于WordPiece的模型中的特殊标记（Token），如[SEP]、[CLS]、[Q]和[MASK]，在语义上总是匹配的。因此，我们在测量模型的SMP值时排除这些特殊标记（Token）。然而，我们将在第5.3节中重新审视并量化它们对检索效果的贡献。

### 5.1 RQ3.1: Semantic Matching Behaviour of Col★

### 5.1 RQ3.1：Col★的语义匹配行为

Results of RQ3.1: Now, we analyse the semantic matching proportion scores for the selected $\mathrm{{Col}} \star$ models presented in Table 4. For all the compared models, we report the Mean SMP values computed at rank cutoff $k = {10}$ . From the Mean-SMP columns in Table 4, we find that ColminiLM, with the same tokenisation and vocabulary size of ColBERT, shows a similar, but slightly reduce semantic matching behaviour to ColBERT. In addition, the SentencePiece tokeniser also shows comparable semantic matching scores. Finally, ColRoBERTa performs more of its matching in the semantic space, both for the reranking and dense retrieval scenarios. This is actually not in line with our expectations - indeed, with a larger vocabulary, we expected to see more exact matches by ColRoBERTa. We explain further the ColRoBERTa's behaviour in the next section.

RQ3.1的结果：现在，我们分析表4中所选$\mathrm{{Col}} \star$模型的语义匹配比例得分。对于所有比较的模型，我们报告在排名截断$k = {10}$处计算的平均SMP值。从表4的平均SMP列中，我们发现与ColBERT具有相同分词和词汇量的ColminiLM，表现出与ColBERT相似但略有降低的语义匹配行为。此外，SentencePiece分词器也显示出相当的语义匹配得分。最后，无论是在重排序还是密集检索场景中，ColRoBERTa在语义空间中的匹配更多。这实际上与我们的预期不符 - 实际上，我们原本期望词汇量更大的ColRoBERTa能有更多的精确匹配。我们将在下一节进一步解释ColRoBERTa的行为。

Answer to RQ3.1: Overall, we observe that using different to-kenisers, $\operatorname{Col} \star$ exhibits different amounts of semantic matching. In particular, the BPE tokeniser based ColRoBERTa model exhibits a stronger preference for semantic matching compared to WordPiece and SentencePiece tokeniser based models. Based on the findings of RQ3.1, we next inspect how semantic matching proportion values can be attributed to different families of tokens (Section 5.2), and to determine the contribution of lexical vs. semantic matching types to retrieval effectiveness (Section 5.3).

对RQ3.1的回答：总体而言，我们观察到使用不同的分词器（Tokeniser），$\operatorname{Col} \star$表现出不同程度的语义匹配。特别是，基于BPE分词器的ColRoBERTa模型与基于WordPiece和SentencePiece分词器的模型相比，对语义匹配表现出更强的偏好。基于RQ3.1的发现，我们接下来将检查语义匹配比例值如何归因于不同类型的标记（Token）（第5.2节），并确定词汇匹配与语义匹配类型对检索效果的贡献（第5.3节）。

### 5.2 RQ3.2: SMP on Salient Token Families

### 5.2 RQ3.2：显著标记（Token）类型的SMP

We now further deepen our analysis on the internals of the late interaction mechanism, by investigating the semantic matching contribution of individual query and document tokens. To this end, we identify salient families of tokens in queries and documents, based on our intuitions about how contextualised embeddings are matched. Table 5 summarises the identified token families.

现在，我们通过研究单个查询和文档标记（Token）的语义匹配贡献，进一步深入分析后期交互机制的内部原理。为此，我们根据对上下文嵌入如何匹配的直觉，确定查询和文档中显著的标记（Token）类型。表5总结了所确定的标记（Token）类型。

<!-- Media -->

Table 5: Salient token families of query (Q) and document (Doc) tokens.

表5：查询（Q）和文档（Doc）标记（Token）的显著类型。

<table><tr><td/><td>Notation</td><td>Type of Tokens</td><td>Example</td></tr><tr><td/><td>QuesToken</td><td>Question tokens</td><td>who, what, where, when, why, which, and how</td></tr><tr><td rowspan="7">Doc</td><td>SubToken</td><td>Sub-word tokens</td><td>Tokens beginning with ## for ColBERT and ColminiLM, not beginning with space for Col- RoBERTa, and not beginning with _ for ColALBERT</td></tr><tr><td>SwToken</td><td>Stopwords tokens</td><td>Terrier stopwords such as is and a</td></tr><tr><td>NumToken</td><td>Numeric tokens</td><td>Token corresponding to single-digit numbers</td></tr><tr><td>StemToken</td><td>Stemmed tokens</td><td>Tokens in the same form as the matching query token after applying Porter stemming</td></tr><tr><td>Low ${}_{idf}$ Token</td><td>Low IDF tokens</td><td>Tokens with IDF below the 25th percentile of IDF distribution</td></tr><tr><td>${\text{Med}}_{idf}$ Token</td><td>Medium IDF tokens</td><td>Tokens with IDF between the 25th and the 75th percentiles of IDF distribution</td></tr><tr><td>${\text{High}}_{idf}$ Token</td><td>High IDF tokens</td><td>Tokens with IDF above the 75th percentile of IDF distribution</td></tr></table>

<table><tbody><tr><td></td><td>符号表示</td><td>Token类型</td><td>示例</td></tr><tr><td></td><td>问题Token（QuesToken）</td><td>问题Token</td><td>谁、什么、哪里、何时、为什么、哪个以及如何</td></tr><tr><td rowspan="7">文档（Doc）</td><td>子Token（SubToken）</td><td>子词Token</td><td>对于ColBERT和ColminiLM，以##开头的Token；对于Col - RoBERTa，不以空格开头的Token；对于ColALBERT，不以_开头的Token</td></tr><tr><td>停用词Token（SwToken）</td><td>停用词Token</td><td>如“is”和“a”这样的Terrier停用词</td></tr><tr><td>数字Token（NumToken）</td><td>数字Token</td><td>对应个位数的Token</td></tr><tr><td>词干Token（StemToken）</td><td>词干化后的Token</td><td>应用Porter词干提取算法后，与匹配查询Token形式相同的Token</td></tr><tr><td>低${}_{idf}$ Token</td><td>低逆文档频率（IDF）Token</td><td>逆文档频率低于逆文档频率分布第25百分位数的Token</td></tr><tr><td>${\text{Med}}_{idf}$ Token</td><td>中等逆文档频率（IDF）Token</td><td>逆文档频率介于逆文档频率分布第25百分位数和第75百分位数之间的Token</td></tr><tr><td>${\text{High}}_{idf}$ Token</td><td>高逆文档频率（IDF）Token</td><td>逆文档频率高于逆文档频率分布第75百分位数的Token</td></tr></tbody></table>

Table 6: Mean semantic matching proportion for the salient document token families in query and document on TREC DL 2020. The highest value among the salient token families in each column is boldfaced.

表6：TREC DL 2020上查询和文档中显著文档词元族的平均语义匹配比例。每列中显著词元族的最高值用粗体显示。

<table><tr><td/><td colspan="5">BM25 (PyTerrier) » Late Interaction</td><td colspan="4">ANN Search » Late Interaction</td></tr><tr><td/><td/><td>ColBERT</td><td>ColminiLM</td><td>ColRoBERTa</td><td>ColALBERT</td><td>ColBERT</td><td>ColminiLM</td><td>ColRoBERTa</td><td>ColALBERT</td></tr><tr><td/><td>All Types</td><td>0.387</td><td>0.363</td><td>0.607</td><td>0.390</td><td>0.406</td><td>0.388</td><td>0.622</td><td>0.413</td></tr><tr><td>OV</td><td>QuesToken</td><td>0.085</td><td>0.087</td><td>0.090</td><td>0.067</td><td>0.087</td><td>0.089</td><td>0.091</td><td>0.070</td></tr><tr><td rowspan="7">Doc</td><td>SubToken</td><td>0.009</td><td>0.011</td><td>0.126</td><td>0.179</td><td>0.013</td><td>0.020</td><td>0.133</td><td>0.190</td></tr><tr><td>SwToken</td><td>0.163</td><td>0.127</td><td>0.159</td><td>0.125</td><td>0.169</td><td>0.134</td><td>0.165</td><td>0.130</td></tr><tr><td>NumToken</td><td>0.017</td><td>0.018</td><td>0.003</td><td>0.001</td><td>0.019</td><td>0.018</td><td>0.004</td><td>0.001</td></tr><tr><td>StemToken</td><td>0.022</td><td>0.024</td><td>0.025</td><td>0.019</td><td>0.023</td><td>0.022</td><td>0.026</td><td>0.020</td></tr><tr><td>LowidfToken</td><td>0.365</td><td>0.344</td><td>0.517</td><td>0.270</td><td>0.381</td><td>0.361</td><td>0.523</td><td>0.289</td></tr><tr><td>${\text{Med}}_{idf}$ Token</td><td>0.021</td><td>0.018</td><td>0.068</td><td>0.018</td><td>0.025</td><td>0.026</td><td>0.074</td><td>0.018</td></tr><tr><td>${\text{High}}_{idf}$ Token</td><td>0.001</td><td>0.001</td><td>0.005</td><td>0.004</td><td>0.001</td><td>0.001</td><td>0.006</td><td>0.005</td></tr></table>

<table><tbody><tr><td></td><td colspan="5">BM25（PyTerrier） » 后期交互</td><td colspan="4">ANN搜索 » 后期交互</td></tr><tr><td></td><td></td><td>ColBERT</td><td>ColminiLM</td><td>ColRoBERTa</td><td>ColALBERT</td><td>ColBERT</td><td>ColminiLM</td><td>ColRoBERTa</td><td>ColALBERT</td></tr><tr><td></td><td>所有类型</td><td>0.387</td><td>0.363</td><td>0.607</td><td>0.390</td><td>0.406</td><td>0.388</td><td>0.622</td><td>0.413</td></tr><tr><td>OV</td><td>问题Token</td><td>0.085</td><td>0.087</td><td>0.090</td><td>0.067</td><td>0.087</td><td>0.089</td><td>0.091</td><td>0.070</td></tr><tr><td rowspan="7">文档</td><td>子Token</td><td>0.009</td><td>0.011</td><td>0.126</td><td>0.179</td><td>0.013</td><td>0.020</td><td>0.133</td><td>0.190</td></tr><tr><td>SwToken</td><td>0.163</td><td>0.127</td><td>0.159</td><td>0.125</td><td>0.169</td><td>0.134</td><td>0.165</td><td>0.130</td></tr><tr><td>数字Token</td><td>0.017</td><td>0.018</td><td>0.003</td><td>0.001</td><td>0.019</td><td>0.018</td><td>0.004</td><td>0.001</td></tr><tr><td>词干Token</td><td>0.022</td><td>0.024</td><td>0.025</td><td>0.019</td><td>0.023</td><td>0.022</td><td>0.026</td><td>0.020</td></tr><tr><td>低逆文档频率Token</td><td>0.365</td><td>0.344</td><td>0.517</td><td>0.270</td><td>0.381</td><td>0.361</td><td>0.523</td><td>0.289</td></tr><tr><td>${\text{Med}}_{idf}$ Token</td><td>0.021</td><td>0.018</td><td>0.068</td><td>0.018</td><td>0.025</td><td>0.026</td><td>0.074</td><td>0.018</td></tr><tr><td>${\text{High}}_{idf}$ Token</td><td>0.001</td><td>0.001</td><td>0.005</td><td>0.004</td><td>0.001</td><td>0.001</td><td>0.006</td><td>0.005</td></tr></tbody></table>

<!-- Media -->

Results of RQ3.2: We inspect the semantic matching behaviour for different contextualised late interaction models with various salient token families listed in Table 5. More specifically, we are more concerned about what matching behaviour is performed for the question tokens in the query and seven families of salient tokens in the document. Table 6 presents the semantic matching proportion scores for the above salient token families for all four contextu-alised late interaction models. We examine the semantic matching behaviour for both the reranking and end-to-end dense retrieval scenarios on the TREC DL 2020 query set. From Table 6, we find that question tokens occurring in the query exhibit low semantic matching scores. Among all the families of salient tokens from documents, semantic matching prefers the low IDF (i.e. frequent) tokens, followed by the family of stopwords tokens. However, semantic matching seldom occurs in the medium and high IDF tokens, which means such rare tokens are more likely to exactly, match during scoring. In addition, token families include stemmed, numeric and sub-word tokens all exhibiting low semantic matching proportion values. Finally,comparing the different $\mathrm{{Col}} \star$ models,we find that ColRoBERTa exhibits the highest semantic matching proportion scores, which is consistent with the findings obtained in Table 4. More interestingly, although ColBERT, ColminiLM and ColALBERT show similar SMP values overall for all types of tokens in Table 4, results in Table 6 indicate that for their semantic matching occurs for different types of tokens. For instance, ColBERT and ColminiLM tend to perform semantic matching for the tokens with relatively low IDF scores and sub-word tokens. ColALBERT (SentencePiece) behaves more similarly to the WordPiece-based models (ColBERT & ColminiLM), except that it more semantic matching comes from sub-word tokens and less from low-IDF tokens.

RQ3.2的结果：我们研究了表5中列出的各种显著词元族在不同上下文延迟交互模型下的语义匹配行为。更具体地说，我们更关注查询中的问题词元和文档中七个显著词元族的匹配行为。表6展示了上述显著词元族在所有四种上下文延迟交互模型下的语义匹配比例得分。我们在TREC DL 2020查询集上研究了重排序和端到端密集检索场景下的语义匹配行为。从表6中我们发现，查询中出现的问题词元的语义匹配得分较低。在文档中的所有显著词元族中，语义匹配更倾向于低逆文档频率（IDF，即频繁出现）的词元，其次是停用词词元族。然而，语义匹配很少发生在中高IDF的词元上，这意味着这类稀有词元在评分时更有可能进行精确匹配。此外，词元族包括词干化、数字和子词词元，它们的语义匹配比例值都较低。最后，比较不同的$\mathrm{{Col}} \star$模型，我们发现ColRoBERTa的语义匹配比例得分最高，这与表4中的结果一致。更有趣的是，尽管表4中ColBERT、ColminiLM和ColALBERT对于所有类型的词元总体上显示出相似的语义匹配比例（SMP）值，但表6的结果表明，它们的语义匹配发生在不同类型的词元上。例如，ColBERT和ColminiLM倾向于对IDF得分相对较低的词元和子词词元进行语义匹配。ColALBERT（SentencePiece）的行为与基于WordPiece的模型（ColBERT和ColminiLM）更为相似，只是它更多的语义匹配来自子词词元，而来自低IDF词元的语义匹配较少。

Interestingly, ColRoBERTa exhibits the highest semantic matching, mostly on low IDF (i.e. frequent) tokens. For different models, we inspect queries returning the same documents, and we focus on those with different matching proportions for the same document, and we explain these differences as follows: as RoBERTa's vocabulary is case-sensitive, some words can be represented by a whole token when occurring in lower-case, but resort to sub-word tokens when starting with an uppercase letter (see Casualties vs. casualties examples in the last row of Table 1). To make a match between these words requires a semantic match (involving relatively frequent sub-word tokens), where a case-insensitive model would have made an exact match (that would likely have been easier to learn). Indeed, the original RoBERTa authors acknowledged that their tokenisation configuration choice might not be the most effective [20]. This analysis indicates the challenges for the search with case-sensitive contextualised language models.

有趣的是，ColRoBERTa的语义匹配程度最高，主要体现在低IDF（即频繁出现）的词元上。对于不同的模型，我们检查返回相同文档的查询，并关注那些对同一文档有不同匹配比例的查询，我们对这些差异的解释如下：由于RoBERTa的词汇表区分大小写，一些单词在小写出现时可以用一个完整的词元表示，但在首字母大写时则需要使用子词词元（见表1最后一行中的“Casualties”与“casualties”示例）。要使这些单词之间进行匹配需要进行语义匹配（涉及相对频繁的子词词元），而不区分大小写的模型则会进行精确匹配（这可能更容易学习）。实际上，原始的RoBERTa作者承认他们的词元化配置选择可能不是最有效的[20]。这一分析表明了使用区分大小写的上下文语言模型进行搜索时面临的挑战。

Answer to RQ3.2: Overall, in quantifying the extent of semantic matching for various token families, we find that low IDF tokens are most likely to exhibit semantic matching. In the next section, we conduct further experiments to quantify the contribution of different types of matching to retrieval effectiveness.

RQ3.2的答案：总体而言，在量化各种词元族的语义匹配程度时，我们发现低IDF词元最有可能表现出语义匹配。在下一节中，我们将进行进一步的实验，以量化不同类型的匹配对检索效果的贡献。

### 5.3 RQ3.3: Contribution of Matching Types to Retrieval Effectiveness

### 5.3 RQ3.3：匹配类型对检索效果的贡献

Finally, as the final outcome of matching behaviour is the ranking of the document, we analyse how the final retrieval effectiveness correlates with the lexical matches and the semantic matches. To conduct this ablation, we also consider retrieval using only "special" tokens, such as [CLS] and [Q], which always match semantically.

最后，由于匹配行为的最终结果是文档的排序，我们分析了最终的检索效果与词汇匹配和语义匹配之间的相关性。为了进行这项消融实验，我们还考虑仅使用“特殊”词元（如[CLS]和[Q]）进行检索，这些词元总是进行语义匹配。

<!-- Media -->

Table 7: Impact of different types of matching behaviour for TREC DL 2020 on nDCG@10, and relative decrease from All (Δ). The $\dagger$ and $\diamond$ symbols denote statistically significant differences compared to the BM25 and the all types matching of a model. The highest nDCG@10 value in each column is boldfaced.

表7：TREC DL 2020不同类型的匹配行为对nDCG@10的影响，以及与所有类型匹配情况相比的相对下降（Δ）。$\dagger$和$\diamond$符号表示与BM25和模型的所有类型匹配情况相比具有统计学显著差异。每列中最高的nDCG@10值用粗体表示。

<table><tr><td rowspan="2">Models</td><td>All Types</td><td colspan="2">Lexical Matching</td><td colspan="2">Semantic Matching</td><td colspan="2">Special Token Matching</td></tr><tr><td>nDCG@10</td><td>nDCG@10</td><td>$\Delta$</td><td>nDCG@10</td><td>$\Delta$</td><td>nDCG@10</td><td>$\Delta$</td></tr><tr><td>BM25 (PyTerrier)</td><td>0.4936</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="8">BM25 (PyTerrier) » Late Interaction</td></tr><tr><td>ColBERT</td><td>0.707 †</td><td>0.527 %</td><td>-25.5%</td><td>0.139%</td><td>-80.3%</td><td>0.519◥</td><td>-26.6%</td></tr><tr><td>ColminiLM</td><td>${0.685} \dagger$</td><td>0.487⋄</td><td>-28.8%</td><td>0.074†◇</td><td>-89.1%</td><td>0.523 %</td><td>-23.7%</td></tr><tr><td>ColRoBERTa</td><td>0.695</td><td>0.397%</td><td>-42.9%</td><td>0.261%</td><td>-62.5%</td><td>0.635 $\dagger  \diamond$</td><td>-8.6%</td></tr><tr><td>ColALBERT</td><td>${0.630}\dot{ + }$</td><td>0.505</td><td>-19.8%</td><td>0.074</td><td>-88.2%</td><td>0.460 %</td><td>-27.1%</td></tr><tr><td colspan="8">ANN Search » Late Interaction</td></tr><tr><td>ColBERT</td><td>0.690 †</td><td>0.492 %</td><td>-28.7%</td><td>0.002†◇</td><td>-99.7%</td><td>0.384⋄</td><td>-44.4%</td></tr><tr><td>ColminiLM</td><td>0.672</td><td>0.426</td><td>-36.6%</td><td>0.001†◇</td><td>-99.9%</td><td>0.347†◇</td><td>-48.4%</td></tr><tr><td>ColRoBERTa</td><td>0.666</td><td>0.350%</td><td>-47.5%</td><td>0.157 to</td><td>-76.4%</td><td>0.574 %</td><td>-13.8%</td></tr><tr><td>ColALBERT</td><td>0.604 $\dagger$</td><td>0.411 %</td><td>-32.0%</td><td>0.007†◇</td><td>-98.8%</td><td>0.341%</td><td>-43.4%</td></tr></table>

<table><tbody><tr><td rowspan="2">模型</td><td>所有类型</td><td colspan="2">词法匹配</td><td colspan="2">语义匹配</td><td colspan="2">特殊Token匹配</td></tr><tr><td>nDCG@10</td><td>nDCG@10</td><td>$\Delta$</td><td>nDCG@10</td><td>$\Delta$</td><td>nDCG@10</td><td>$\Delta$</td></tr><tr><td>BM25（PyTerrier）</td><td>0.4936</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="8">BM25（PyTerrier） » 后期交互</td></tr><tr><td>ColBERT</td><td>0.707 †</td><td>0.527 %</td><td>-25.5%</td><td>0.139%</td><td>-80.3%</td><td>0.519◥</td><td>-26.6%</td></tr><tr><td>ColminiLM</td><td>${0.685} \dagger$</td><td>0.487⋄</td><td>-28.8%</td><td>0.074†◇</td><td>-89.1%</td><td>0.523 %</td><td>-23.7%</td></tr><tr><td>ColRoBERTa</td><td>0.695</td><td>0.397%</td><td>-42.9%</td><td>0.261%</td><td>-62.5%</td><td>0.635 $\dagger  \diamond$</td><td>-8.6%</td></tr><tr><td>ColALBERT</td><td>${0.630}\dot{ + }$</td><td>0.505</td><td>-19.8%</td><td>0.074</td><td>-88.2%</td><td>0.460 %</td><td>-27.1%</td></tr><tr><td colspan="8">ANN搜索 » 后期交互</td></tr><tr><td>ColBERT</td><td>0.690 †</td><td>0.492 %</td><td>-28.7%</td><td>0.002†◇</td><td>-99.7%</td><td>0.384⋄</td><td>-44.4%</td></tr><tr><td>ColminiLM</td><td>0.672</td><td>0.426</td><td>-36.6%</td><td>0.001†◇</td><td>-99.9%</td><td>0.347†◇</td><td>-48.4%</td></tr><tr><td>ColRoBERTa</td><td>0.666</td><td>0.350%</td><td>-47.5%</td><td>0.157 至</td><td>-76.4%</td><td>0.574 %</td><td>-13.8%</td></tr><tr><td>ColALBERT</td><td>0.604 $\dagger$</td><td>0.411 %</td><td>-32.0%</td><td>0.007†◇</td><td>-98.8%</td><td>0.341%</td><td>-43.4%</td></tr></tbody></table>

<!-- Media -->

Results of RQ3.3: Now, we examine the retrieval effectiveness by conducting only special matching, only semantic matching, as well as special token matching (e.g., [CLS], [Q], [SEP] and [MASK] tokens for WordPiece tokeniser), in response to the input queries of the TREC DL 2020 query set. Table 7 presents the impact of performing a particular type of matching on the retrieval effectiveness (measured by nDCG@10) as well as the reduction percentage compared to all types of matching. From Table 7, we find that performing each type of matching alone results in significant reductions in effectiveness compared to all types of matching, for both the reranking and end-to-end dense retrieval scenarios. In particular, for all models except ColRoBERTa, lexical matching contributes to the highest retrieval effectiveness; for ColRoBERTa, the special tokens have excellent effectiveness (contributing 80-90% of the full effectiveness). Similarly, semantic matching alone exhibits low effectiveness but is strongest for ColRoBERTa (this again demonstrates the strong semantic properties of the ColRoBERTa embeddings). Moreover, Table 5 tells us that this semantic matching is mostly concentrated on frequent (low IDF) tokens. Finally, the high performance of lexical matching is mostly related to medium and high IDF tokens - indeed, this observation echoes the finding of [5] that ColBERT is able to capture more important terms by performing exact matches. Our work systematically quantifies and generalises this finding to various contextualised late interaction models. However, different types of matching need to work together to achieve optimal retrieval effectiveness, as performing any type of matching alone will result in a significant drop in retrieval effectiveness compared to performing all types of matching.

RQ3.3的结果：现在，针对TREC DL 2020查询集的输入查询，我们仅进行特殊匹配、仅进行语义匹配以及特殊标记匹配（例如，WordPiece分词器的[CLS]、[Q]、[SEP]和[MASK]标记），以此来检验检索效果。表7展示了执行特定类型的匹配对检索效果（通过nDCG@10衡量）的影响，以及与所有类型匹配相比的降低百分比。从表7中我们发现，与所有类型的匹配相比，在重排序和端到端密集检索场景中，单独执行每种类型的匹配都会导致效果显著降低。特别是，除了ColRoBERTa之外的所有模型，词法匹配对检索效果的贡献最大；对于ColRoBERTa，特殊标记具有出色的效果（贡献了全部效果的80 - 90%）。同样，仅进行语义匹配的效果较低，但对于ColRoBERTa来说是最强的（这再次证明了ColRoBERTa嵌入的强大语义特性）。此外，表5告诉我们，这种语义匹配主要集中在高频（低逆文档频率，IDF）标记上。最后，词法匹配的高性能主要与中高IDF标记有关——实际上，这一观察结果与文献[5]的发现相呼应，即ColBERT能够通过精确匹配捕获更重要的术语。我们的工作系统地量化并将这一发现推广到各种上下文后期交互模型。然而，不同类型的匹配需要协同工作才能实现最佳检索效果，因为与执行所有类型的匹配相比，单独执行任何类型的匹配都会导致检索效果显著下降。

Answer to RQ3.3: We find that the late interaction mechanism benefits more from lexical matching than semantic matching. In addition, special tokens, such as the [CLS] token, play a very important role in matching, especially for the ColRoBERTa model.

对RQ3.3的回答：我们发现后期交互机制从词法匹配中获得的益处比语义匹配更多。此外，特殊标记，如[CLS]标记，在匹配中起着非常重要的作用，特别是对于ColRoBERTa模型。

## 6 CONCLUSIONS

## 6 结论

This work provides a comprehensive study that investigates the reproducibility and replicability of ColBERT and sheds insights into the semantic matching behaviour in multiple representation dense retrieval. Our main findings and insights are summarised as follows:

这项工作提供了一项全面的研究，调查了ColBERT的可重复性和可复现性，并深入了解了多表示密集检索中的语义匹配行为。我们的主要发现和见解总结如下：

Based on the Reproducibility experiments of ColBERT, we are able to successfully reproduce the performance of ColBERT on various query sets. In addition, several ablation studies show that more training interactions still help improve the retrieval effectiveness of ColBERT. The L2 similarity function gives higher performance than Cosine for the end-to-end setting and exhibits comparable performance for the reranking retrieval.

基于ColBERT的可重复性实验，我们能够成功复现ColBERT在各种查询集上的性能。此外，几项消融研究表明，更多的训练交互仍然有助于提高ColBERT的检索效果。在端到端设置中，L2相似度函数的性能比余弦相似度函数更高，并且在重排序检索中表现出相当的性能。

For Replicability,we extend ColBERT to Col $\star$ by implementing the contextualised late interaction mechanism upon various pre-trained models with different tokenisers. We find that the base pre-trained model used for ColBERT can greatly impact the retrieval performance, but models from the BERT family are the most effective.

对于可复现性，我们通过在各种使用不同分词器的预训练模型上实现上下文后期交互机制，将ColBERT扩展到Col $\star$。我们发现，用于ColBERT的基础预训练模型会极大地影响检索性能，但BERT家族的模型最为有效。

Finally, we conduct the Insights experiments to explore more useful insights behind the contextualised late interaction. In particular, we introduce a metric to quantify semantic matching for dense retrieval. Extensive experimental results reveal that: (i) $\mathrm{{Col}} \star$ models with different tokenisation methods show different semantic matching values, in particular, the ColRoBERTa model exhibits higher SMP values due to its case-sensitive tokeniser; (ii) among various salient families of tokens, low IDF and stopwords tokens are more likely to perform semantic matching; (iii) performing only exact matching and only special token matching contribute more than only semantic matching to all types matching retrieval effectiveness. Overall, our experimental results explain how ColBERT-like models perform retrieval, and can shed insight into more effective dense retrieval model design.

最后，我们进行了洞察实验，以探索上下文后期交互背后更有用的见解。特别是，我们引入了一个指标来量化密集检索的语义匹配。大量实验结果表明：（i）使用不同分词方法的$\mathrm{{Col}} \star$模型表现出不同的语义匹配值，特别是，由于其区分大小写的分词器，ColRoBERTa模型表现出更高的语义匹配百分比（SMP）值；（ii）在各种显著的标记族中，低IDF标记和停用词标记更有可能进行语义匹配；（iii）仅进行精确匹配和仅进行特殊标记匹配对所有类型匹配的检索效果的贡献比仅进行语义匹配更大。总体而言，我们的实验结果解释了类似ColBERT的模型如何进行检索，并为更有效的密集检索模型设计提供了见解。

## ACKNOWLEDGMENTS

## 致谢

This work is supported, in part, by the spoke "FutureHPC & BigData" of the ICSC - Centro Nazionale di Ricerca in High-Performance Computing, Big Data and Quantum Computing funded by European Union - NextGenerationEU, and the FoReLab project (Departments of Excellence). Xiao Wang acknowledges support by the China Scholarship Council (CSC) from the Ministry of Education of P.R. China. We thank Jo Kristian Bergum for assistance with the ColminiLM checkpoint.

这项工作部分得到了由欧盟 - 下一代欧盟资助的意大利国家高性能计算、大数据和量子计算研究中心（ICSC）的“未来高性能计算与大数据”分支以及卓越部门项目FoReLab的支持。王晓感谢中国教育部国家留学基金管理委员会（CSC）的资助。我们感谢乔·克里斯蒂安·伯古姆（Jo Kristian Bergum）在ColminiLM检查点方面提供的帮助。

## REFERENCES

## 参考文献

[1] Jo Kristian Bergum. 2021. Pretrained Transformer Language Models for Search - part 4. https://blog.vespa.ai/pretrained-transformer-language-models-for-search-part-4/

[1] 乔·克里斯蒂安·伯古姆（Jo Kristian Bergum）. 2021. 用于搜索的预训练Transformer语言模型 - 第4部分. https://blog.vespa.ai/pretrained-transformer-language-models-for-search-part-4/

[2] Kaj Bostrom and Greg Durrett. 2020. Byte Pair Encoding is Suboptimal for Language Model Pretraining. In Proceedings of EMNLP: Findings. 4617-4624.

[2] 卡伊·博斯特伦（Kaj Bostrom）和格雷格·达雷特（Greg Durrett）. 2020. 字节对编码对于语言模型预训练并非最优. 见《自然语言处理经验方法会议：研究成果》论文集. 4617 - 4624.

[3] Kevin Clark, Minh-Thang Luong, Quoc V Le, and Christopher D Manning. 2020. ELECTRA: Pre-training text encoders as discriminators rather than generators. In Proceddings of ICLR.

[3] 凯文·克拉克（Kevin Clark）、明 - 唐·卢昂（Minh - Thang Luong）、奎克·V·勒（Quoc V Le）和克里斯托弗·D·曼宁（Christopher D Manning）. 2020. ELECTRA：将文本编码器预训练为判别器而非生成器. 见《国际学习表征会议》论文集.

[4] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of ACL. 4171-4186.

[4] 雅各布·德夫林（Jacob Devlin）、张明伟（Ming-Wei Chang）、肯顿·李（Kenton Lee）和克里斯蒂娜·图托纳娃（Kristina Toutanova）。2019年。BERT：用于语言理解的深度双向变换器预训练。见《计算语言学协会会议录》（Proceedings of ACL）。4171 - 4186页。

[5] Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant. 2021. A White Box Analysis of ColBERT. In Proceedings of ECIR. 257-263.

[5] 蒂博·福尔马尔（Thibault Formal）、本杰明·皮沃瓦尔斯基（Benjamin Piwowarski）和斯特凡·克兰尚（Stéphane Clinchant）。2021年。ColBERT的白盒分析。见《欧洲信息检索会议录》（Proceedings of ECIR）。257 - 263页。

[6] Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant. 2022. Match your words! a study of lexical matching in neural information retrieval. In Proceedings of ECIR. 120-127.

[6] 蒂博·福尔马尔（Thibault Formal）、本杰明·皮沃瓦尔斯基（Benjamin Piwowarski）和斯特凡·克兰尚（Stéphane Clinchant）。2022年。匹配你的词汇！神经信息检索中的词汇匹配研究。见《欧洲信息检索会议录》（Proceedings of ECIR）。120 - 127页。

[7] Mitko Gospodinov, Sean MacAvaney, and Craig Macdonald. 2023. Doc2Query: When Less is More. In Proceedings of ECIR.

[7] 米特科·戈斯波迪诺夫（Mitko Gospodinov）、肖恩·麦卡瓦尼（Sean MacAvaney）和克雷格·麦克唐纳（Craig Macdonald）。2023年。Doc2Query：少即是多。见《欧洲信息检索会议录》（Proceedings of ECIR）。

[8] Weidong Guo, Mingjun Zhao, Lusheng Zhang, Di Niu, Jinwen Luo, Zhenhua Liu, Zhenyang Li, and Jianbo Tang. 2021. LICHEE: Improving Language Model Pre-training with Multi-grained Tokenization. In Proceedings of ACL-IJCNLP: Findings. 1383-1392.

[8] 郭卫东、赵明君、张鲁生、牛迪、罗金文、刘振华、李振阳和唐建波。2021年。荔枝模型（LICHEE）：通过多粒度分词改进语言模型预训练。见《计算语言学协会 - 国际自然语言处理联合会议：研究成果》（Proceedings of ACL - IJCNLP: Findings）。1383 - 1392页。

[9] Sebastian Hofstätter, Sophia Althammer, Michael Schröder, Mete Sertkan, and Allan Hanbury. 2020. Improving efficient neural ranking models with cross-architecture knowledge distillation. In arXiv preprint arXiv:2010.02666.

[9] 塞巴斯蒂安·霍夫施泰特（Sebastian Hofstätter）、索菲娅·阿尔塔默（Sophia Althammer）、迈克尔·施罗德（Michael Schröder）、梅特·塞尔特坎（Mete Sertkan）和艾伦·汉伯里（Allan Hanbury）。2020年。通过跨架构知识蒸馏改进高效神经排序模型。见arXiv预印本arXiv:2010.02666。

[10] Sebastian Hofstätter, Sheng-Chieh Lin, Jheng-Hong Yang, Jimmy Lin, and Allan Hanbury. 2021. Efficiently teaching an effective dense retriever with balanced topic aware sampling. In Proceedings of SIGIR. 113-122.

[10] 塞巴斯蒂安·霍夫施泰特（Sebastian Hofstätter）、林圣杰（Sheng - Chieh Lin）、杨政宏（Jheng - Hong Yang）、林吉米（Jimmy Lin）和艾伦·汉伯里（Allan Hanbury）。2021年。通过平衡主题感知采样有效训练高效密集检索器。见《信息检索研究与发展会议录》（Proceedings of SIGIR）。113 - 122页。

[11] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361.

[11] 贾里德·卡普兰（Jared Kaplan）、山姆·麦坎德利什（Sam McCandlish）、汤姆·亨尼根（Tom Henighan）、汤姆·B·布朗（Tom B Brown）、本杰明·切斯（Benjamin Chess）、雷翁·蔡尔德（Rewon Child）、斯科特·格雷（Scott Gray）、亚历克·拉德福德（Alec Radford）、杰弗里·吴（Jeffrey Wu）和达里奥·阿莫迪（Dario Amodei）。2020年。神经语言模型的缩放定律。arXiv预印本arXiv:2001.08361。

[12] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for Open-Domain Question Answering. In Proceedings of EMNLP. 6769-6781.

[12] 弗拉基米尔·卡尔普欣（Vladimir Karpukhin）、巴拉斯·奥古兹（Barlas Oguz）、闵世元（Sewon Min）、帕特里克·刘易斯（Patrick Lewis）、莱德尔·吴（Ledell Wu）、谢尔盖·叶杜诺夫（Sergey Edunov）、陈丹琦和易文涛。2020年。用于开放域问答的密集段落检索。见《自然语言处理经验方法会议录》（Proceedings of EMNLP）。6769 - 6781页。

[13] Omar Khattab and Matei Zaharia. 2020. ColBERT: Efficient and effective passage search via contextualized late interaction over BERT. In Proceedings of SIGIR. 39-48.

[13] 奥马尔·哈塔卜（Omar Khattab）和马特·扎哈里亚（Matei Zaharia）。2020年。ColBERT：通过基于BERT的上下文后期交互实现高效有效的段落搜索。见《信息检索研究与发展会议录》（Proceedings of SIGIR）。39 - 48页。

[14] Taku Kudo and John Richardson. 2018. SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing. In Proceedings of EMNLP. 66-71.

[14] 工藤拓（Taku Kudo）和约翰·理查森（John Richardson）。2018年。SentencePiece：一种用于神经文本处理的简单且与语言无关的子词分词器和解分词器。见《自然语言处理经验方法会议录》（Proceedings of EMNLP）。66 - 71页。

[15] Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. 2020. ALBERT: A lite BERT for self-supervised learning of language representations. In Proceddings of ICLR.

[15] 蓝振中、陈明达、塞巴斯蒂安·古德曼（Sebastian Goodman）、凯文·金佩尔（Kevin Gimpel）、皮尤什·夏尔马（Piyush Sharma）和拉杜·索里库特（Radu Soricut）。2020年。ALBERT：用于语言表示自监督学习的轻量级BERT。见《国际学习表征会议录》（Proceddings of ICLR）。

[16] Carlos Lassance, Maroua Maachou, Joohee Park, and Stéphane Clinchant. 2022. Learned Token Pruning in Contextualized Late Interaction over BERT (ColBERT). In Proceedings of SIGIR. 2232-2236.

[16] 卡洛斯·拉桑斯（Carlos Lassance）、马鲁阿·马乔（Maroua Maachou）、朴珠熙（Joohee Park）和斯特凡·克兰尚（Stéphane Clinchant）。2022年。基于BERT的上下文后期交互（ColBERT）中的学习型Token剪枝。见《信息检索研究与发展会议录》（Proceedings of SIGIR）。2232 - 2236页。

[17] Sheng-Chieh Lin, Akari Asai, Minghan Li, Barlas Oguz, Jimmy Lin, Yashar Mehdad, Wen-tau Yih, and Xilun Chen. 2023. How to Train Your DRAGON: Diverse Augmentation Towards Generalizable Dense Retrieval. arXiv preprint arXiv:2302.07452 (2023).

[17] 林圣杰（Sheng - Chieh Lin）、浅井朱里（Akari Asai）、李明翰、巴拉斯·奥古兹（Barlas Oguz）、林吉米（Jimmy Lin）、亚沙尔·梅赫达德（Yashar Mehdad）、易文涛和陈希伦。2023年。如何训练你的DRAGON：迈向可泛化密集检索的多样化增强。arXiv预印本arXiv:2302.07452（2023年）。

[18] Sheng-Chieh Lin, Jheng-Hong Yang, and Jimmy Lin. 2020. Distilling dense representations for ranking using tightly-coupled teachers. arXiv preprint arXiv:2010.11386 (2020).

[18] 林圣杰（Sheng - Chieh Lin）、杨政宏（Jheng - Hong Yang）和林吉米（Jimmy Lin）。2020年。使用紧密耦合教师蒸馏用于排序的密集表示。arXiv预印本arXiv:2010.11386（2020年）。

[19] Sheng-Chieh Lin, Jheng-Hong Yang, and Jimmy Lin. 2021. In-batch negatives for knowledge distillation with tightly-coupled teachers for dense retrieval. In Proceedings of the 6th Workshop on RepL4NLP. 163-173.

[19] 林圣杰（Sheng - Chieh Lin）、杨政宏（Jheng - Hong Yang）和林吉米（Jimmy Lin）。2021年。使用紧密耦合教师进行密集检索的知识蒸馏中的批内负样本。见《第6届自然语言处理表示学习研讨会会议录》（Proceedings of the 6th Workshop on RepL4NLP）。163 - 173页。

[20] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2020. RoBERTa: A

[20] 刘音涵（Yinhan Liu）、迈尔·奥特（Myle Ott）、纳曼·戈亚尔（Naman Goyal）、杜静飞（Jingfei Du）、曼达尔·乔希（Mandar Joshi）、陈丹琦（Danqi Chen）、奥默·利维（Omer Levy）、迈克·刘易斯（Mike Lewis）、卢克·泽特尔莫耶（Luke Zettlemoyer）和韦塞林·斯托亚诺夫（Veselin Stoyanov）。2020年。RoBERTa：一种

Robustly Optimized BERT Pretraining Approach. In Proceddings of ICLR.

鲁棒优化的BERT预训练方法。发表于国际学习表征会议（ICLR）论文集。

[21] Sean MacAvaney, Nicola Tonellotto, and Craig Macdonald. 2022. Adaptive re-ranking with a corpus graph. In Proceedings of CIKM. 1491-1500.

[21] 肖恩·麦卡瓦尼（Sean MacAvaney）、尼古拉·托内洛托（Nicola Tonellotto）和克雷格·麦克唐纳（Craig Macdonald）。2022年。基于语料库图的自适应重排序。发表于信息与知识管理国际会议（CIKM）论文集，第1491 - 1500页。

[22] Craig Macdonald, Nicola Tonellotto, and Iadh Ounis. 2021. On Single and Multiple Representations in Dense Passage Retrieval. IIR 2021 Workshop (2021).

[22] 克雷格·麦克唐纳（Craig Macdonald）、尼古拉·托内洛托（Nicola Tonellotto）和伊阿德·乌尼斯（Iadh Ounis）。2021年。密集段落检索中的单表示和多表示研究。发表于信息检索进展国际会议（IIR）2021研讨会（2021年）。

[23] Suraj Nair, Eugene Yang, Dawn Lawrie, Kevin Duh, Paul McNamee, Kenton Murray, James Mayfield, and Douglas W Oard. 2022. Transfer learning approaches for building cross-language dense retrieval models. In roceddings of ECIR. 382- 396.

[23] 苏拉杰·奈尔（Suraj Nair）、尤金·杨（Eugene Yang）、道恩·劳里（Dawn Lawrie）、凯文·杜（Kevin Duh）、保罗·麦克纳米（Paul McNamee）、肯顿·默里（Kenton Murray）、詹姆斯·梅菲尔德（James Mayfield）和道格拉斯·W·奥尔德（Douglas W Oard）。2022年。构建跨语言密集检索模型的迁移学习方法。发表于欧洲信息检索会议（ECIR）论文集，第382 - 396页。

[24] Association of Computing Machinery. 2020. Artifact Review and Badging. https://www.acm.org/publications/policies/artifact-review-and-badging-current.

[24] 美国计算机协会（Association of Computing Machinery）。2020年。人工制品评审与徽章制度。https://www.acm.org/publications/policies/artifact-review-and-badging-current。

[25] Yingqi Qu, Yuchen Ding, Jing Liu, Kai Liu, Ruiyang Ren, Wayne Xin Zhao, Daxi-ang Dong, Hua Wu, and Haifeng Wang. 2021. RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering. In Proceedings of NAACL. 5835-5847.

[25] 曲英琦（Yingqi Qu）、丁雨晨（Yuchen Ding）、刘静（Jing Liu）、刘凯（Kai Liu）、任瑞阳（Ruiyang Ren）、赵鑫（Wayne Xin Zhao）、董大祥（Daxi-ang Dong）、吴华（Hua Wu）和王海峰（Haifeng Wang）。2021年。RocketQA：面向开放域问答的密集段落检索优化训练方法。发表于北美计算语言学协会会议（NAACL）论文集，第5835 - 5847页。

[26] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language Models are Unsupervised Multitask Learners.

[26] 亚历克·拉德福德（Alec Radford）、杰夫·吴（Jeff Wu）、雷翁·蔡尔德（Rewon Child）、大卫·栾（David Luan）、达里奥·阿莫迪（Dario Amodei）和伊利亚·苏茨克维（Ilya Sutskever）。2019年。语言模型是无监督多任务学习者。

[27] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. Journal of Machine Learning Research (2020), 1-67.

[27] 科林·拉菲尔（Colin Raffel）、诺姆·沙泽尔（Noam Shazeer）、亚当·罗伯茨（Adam Roberts）、凯瑟琳·李（Katherine Lee）、沙兰·纳朗（Sharan Narang）、迈克尔·马特纳（Michael Matena）、周燕琪（Yanqi Zhou）、李伟（Wei Li）和彼得·J·刘（Peter J Liu）。2020年。使用统一的文本到文本转换器探索迁移学习的极限。《机器学习研究杂志》（2020年），第1 - 67页。

[28] Jenalea Rajab. 2022. Effect of Tokenisation Strategies for Low-Resourced Southern African Languages. In Proceedings of Workshop on African Natural Language Processing.

[28] 杰纳利娅·拉贾布（Jenalea Rajab）。2022年。低资源南部非洲语言分词策略的影响。发表于非洲自然语言处理研讨会论文集。

[29] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. 2021. ColBERTv2: Effective and efficient retrieval via lightweight late interaction. In Proceedings of NAACL.

[29] 凯沙夫·桑塔南姆（Keshav Santhanam）、奥马尔·哈塔卜（Omar Khattab）、乔恩·萨德 - 法尔孔（Jon Saad-Falcon）、克里斯托弗·波茨（Christopher Potts）和马特·扎哈里亚（Matei Zaharia）。2021年。ColBERTv2：通过轻量级后期交互实现高效检索。发表于北美计算语言学协会会议（NAACL）论文集。

[30] Harrisen Scells, Shengyao Zhuang, and Guido Zuccon. 2022. Reduce, Reuse, Recycle: Green Information Retrieval Research. In Proceedings of SIGIR. 2825- 2837.

[30] 哈里森·斯凯尔斯（Harrisen Scells）、庄圣耀（Shengyao Zhuang）和圭多·祖科恩（Guido Zuccon）。2022年。减少、再利用、回收：绿色信息检索研究。发表于信息检索研究与发展国际会议（SIGIR）论文集，第2825 - 2837页。

[31] Mike Schuster and Kaisuke Nakajima. 2012. Japanese and korean voice search. In Proceddings of ICASSP. IEEE, 5149-5152.

[31] 迈克·舒斯特（Mike Schuster）和中岛佳介（Kaisuke Nakajima）。2012年。日语和韩语语音搜索。发表于国际声学、语音和信号处理会议（ICASSP）论文集。电气与电子工程师协会（IEEE），第5149 - 5152页。

[32] Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. Neural Machine Translation of Rare Words with Subword Units. In Proceedings of ACL. 1715- 1725.

[32] 里科·森里奇（Rico Sennrich）、巴里·哈多（Barry Haddow）和亚历山德拉·伯奇（Alexandra Birch）。2016年。使用子词单元进行稀有词的神经机器翻译。发表于计算语言学协会会议（ACL）论文集，第1715 - 1725页。

[33] Cagri Toraman, Eyup Halit Yilmaz, Furkan Şahinuç, and Oguzhan Ozcelik. 2022. Impact of Tokenization on Language Models: An Analysis for Turkish. arXiv preprint arXiv:2204.08832 (2022).

[33] 卡格里·托拉曼（Cagri Toraman）、埃尤普·哈利特·伊尔马兹（Eyup Halit Yilmaz）、富尔坎·萨希努奇（Furkan Şahinuç）和奥古赞·奥兹切利克（Oguzhan Ozcelik）。2022年。分词对语言模型的影响：土耳其语分析。预印本arXiv:2204.08832（2022年）。

[34] Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. 2020. MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers. In Proceddings of NeurIPS, Vol. 33. 5776-5788.

[34] 王文慧（Wenhui Wang）、魏富如（Furu Wei）、李东（Li Dong）、鲍航波（Hangbo Bao）、杨楠（Nan Yang）和周明（Ming Zhou）。2020年。MiniLM：用于预训练变压器与任务无关压缩的深度自注意力蒸馏。发表于神经信息处理系统大会（NeurIPS）论文集，第33卷，第5776 - 5788页。

[35] Xiao Wang, Sean MacAvaney, Craig Macdonald, and Iadh Ounis. 2022. An Inspection of the Reproducibility and Replicability of TCT-ColBERT. In Proceedings of SIGIR. 2790-2800.

[35] 小王、肖恩·麦卡瓦尼（Sean MacAvaney）、克雷格·麦克唐纳（Craig Macdonald）和伊阿德·乌尼斯（Iadh Ounis）。2022年。对TCT - ColBERT可重复性和可复制性的考察。收录于《SIGIR会议论文集》。第2790 - 2800页。

[36] Xiao Wang, Craig Macdonald, and Iadh Ounis. 2022. Improving zero-shot retrieval using dense external expansion. Information Processing & Management 59, 5 (2022), 103026.

[36] 小王、克雷格·麦克唐纳（Craig Macdonald）和伊阿德·乌尼斯（Iadh Ounis）。2022年。利用密集外部扩展改进零样本检索。《信息处理与管理》，2022年第59卷第5期，第103026页。

[37] Xiao Wang, Craig Macdonald, Nicola Tonellotto, and Iadh Ounis. 2022. ColBERT-PRF: Semantic Pseudo-Relevance Feedback for Dense Passage and Document Retrieval. ACM Transactions on the Web (2022).

[37] 小王、克雷格·麦克唐纳（Craig Macdonald）、尼古拉·托内洛托（Nicola Tonellotto）和伊阿德·乌尼斯（Iadh Ounis）。2022年。ColBERT - PRF：用于密集段落和文档检索的语义伪相关反馈。《ACM网络汇刊》（2022年）。

[38] Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed, and Arnold Overwijk. 2021. Approximate nearest neighbor negative contrastive learning for dense text retrieval. In Proceedings of ICLR.

[38] 李雄（Lee Xiong）、熊晨彦（Chenyan Xiong）、李烨（Ye Li）、邓国峰（Kwok - Fung Tang）、刘佳琳（Jialin Liu）、保罗·贝内特（Paul Bennett）、朱奈德·艾哈迈德（Junaid Ahmed）和阿诺德·奥弗维克（Arnold Overwijk）。2021年。用于密集文本检索的近似最近邻负对比学习。收录于《ICLR会议论文集》。

[39] Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Jiafeng Guo, Min Zhang, and Shaoping Ma. 2021. Optimizing dense retrieval model training with hard negatives. In Proceedings of SIGIR. 1503-1512.

[39] 詹景涛、毛佳欣、刘奕群、郭佳峰、张敏和马少平。2021年。利用难负样本优化密集检索模型训练。收录于《SIGIR会议论文集》。第1503 - 1512页。

[40] Ruiqi Zhong, Dhruba Ghosh, Dan Klein, and Jacob Steinhardt. 2021. Are Larger Pretrained Language Models Uniformly Better? Comparing Performance at the Instance Level. In Proceedings of ACL: Findings. 3813-3827.

[40] 钟瑞琪、德鲁巴·戈什（Dhruba Ghosh）、丹·克莱因（Dan Klein）和雅各布·施泰因哈特（Jacob Steinhardt）。2021年。更大的预训练语言模型是否始终更优？在实例层面比较性能。收录于《ACL：研究成果》。第3813 - 3827页。