# Overlap Set Similarity Joins with Theoretical Guarantees

# 具有理论保证的重叠集相似度连接

Dong Deng

邓东

MIT CSAIL

dongdeng@csail.mit.edu

Yufei Tao

陶宇飞

Chinese University of Hong Kong

香港中文大学

taoyf@cse.cuhk.edu.hk

Guoliang Li

李国良

Tsinghua University

清华大学

liguoliang@tsinghua.edu.cn

## Abstract

## 摘要

This paper studies the set similarity join problem with overlap constraints which,given two collections of sets and a constant $c$ ,finds all the set pairs in the datasets that share at least $c$ common elements. This is a fundamental operation in many fields, such as information retrieval, data mining, and machine learning. The time complexity of all existing methods is $O\left( {n}^{2}\right)$ where $n$ is the total size of all the sets. In this paper, we present a size-aware algorithm with the time complexity of $O\left( {{n}^{2 - \frac{1}{c}}{k}^{\frac{1}{2c}}}\right)  = o\left( {n}^{2}\right)  + O\left( k\right)$ ,where $k$ is the number of results. The size-aware algorithm divides all the sets into small and large ones based on their sizes and processes them separately. We can use existing methods to process the large sets and focus on the small sets in this paper. We develop several optimization heuristics for the small sets to improve the practical performance significantly. As the size boundary between the small sets and the large sets is crucial to the efficiency, we propose an effective size boundary selection algorithm to judiciously choose an appropriate size boundary, which works very well in practice. Experimental results on real-world datasets show that our methods achieve high performance and outperform the state-of-the-art approaches by up to an order of magnitude.

本文研究具有重叠约束的集合相似度连接问题，即给定两个集合集合和一个常数 $c$ ，找出数据集中所有共享至少 $c$ 个公共元素的集合对。这是许多领域（如信息检索、数据挖掘和机器学习）中的一项基本操作。所有现有方法的时间复杂度为 $O\left( {n}^{2}\right)$ ，其中 $n$ 是所有集合的总大小。在本文中，我们提出了一种大小感知算法，其时间复杂度为 $O\left( {{n}^{2 - \frac{1}{c}}{k}^{\frac{1}{2c}}}\right)  = o\left( {n}^{2}\right)  + O\left( k\right)$ ，其中 $k$ 是结果的数量。该大小感知算法根据集合的大小将所有集合分为小集合和大集合，并分别进行处理。我们可以使用现有方法处理大集合，而本文主要关注小集合。我们为小集合开发了几种优化启发式方法，以显著提高实际性能。由于小集合和大集合之间的大小边界对效率至关重要，我们提出了一种有效的大小边界选择算法，以明智地选择合适的大小边界，该算法在实践中效果非常好。在真实世界数据集上的实验结果表明，我们的方法具有高性能，并且比现有最先进的方法高出一个数量级。

## CCS CONCEPTS

## 计算机协会概念分类

- Information systems $\rightarrow$ Join algorithms; Information integration;

- 信息系统 $\rightarrow$ 连接算法；信息集成；

## KEYWORDS

## 关键词

Similarity Join; Overlap; Set; Scalable; Sub-quadratic; Theoretical Guarantee

相似度连接；重叠；集合；可扩展；亚二次；理论保证

## ACM Reference format:

## ACM引用格式:

Dong Deng, Yufei Tao, and Guoliang Li. 2018. Overlap Set Similarity Joins with Theoretical Guarantees. In Proceedings of 2018 International Conference on Management of Data, Houston, TX, USA, June 10-15, 2018 (SIGMOD'18), 16 pages.

邓东、陶宇飞和李国良。2018年。具有理论保证的重叠集相似度连接。收录于《2018年国际数据管理会议论文集》，美国德克萨斯州休斯顿，2018年6月10 - 15日（SIGMOD'18），共16页。

https://doi.org/http://dx.doi.org/10.1145/XXXXXX.XXXXXX

## 1 INTRODUCTION

## 1 引言

Set similarity join with overlap constraints, which, given two collections of sets (e.g.,the topic set of a document) and a constant $c$ , finds all the set pairs that share at least $c$ common elements,is a

具有重叠约束的集合相似度连接，即给定两个集合集合（例如，文档的主题集合）和一个常数 $c$ ，找出所有共享至少 $c$ 个公共元素的集合对，是一种

fundamental operation in many applications, such as word embedding [23], recommender systems [35], and matrix factorization [29]. Given a collection of documents, where each document contains a bag of words, for each word we can get a set of documents containing this word. Many prestige word embedding models, such as the deep learning based models GloVe [23] and Word2Vec [20] and the classical matrix factorization based models HAL (Hyperspace Analogue to Language) [16] and PPMI (Positive Pointwise Mutual Information) [6], make use of the number of documents in which a pair of words co-occurs. The overlap set similarity join can build the word co-occurrence matrix for these models, as the co-occurrence of two words is the same as the overlap size of their corresponding document sets. In addition, the recommender systems [35] often use the overlap (e.g.,"sharing $c$ common friends" in Facebook) to explain the recommendations for better transparency and user experience.

在许多应用中是一项基础操作，例如词嵌入（word embedding）[23]、推荐系统[35]和矩阵分解（matrix factorization）[29]。给定一组文档，其中每个文档包含一组词，对于每个词，我们可以得到一组包含该词的文档。许多著名的词嵌入模型，如基于深度学习的模型GloVe [23]和Word2Vec [20]，以及基于经典矩阵分解的模型HAL（超空间语言类比，Hyperspace Analogue to Language）[16]和PPMI（正点互信息，Positive Pointwise Mutual Information）[6]，都会利用一对词共同出现的文档数量。重叠集相似度连接可以为这些模型构建词共现矩阵，因为两个词的共现等同于它们对应文档集的重叠大小。此外，推荐系统[35]通常利用重叠（例如，脸书（Facebook）中的“共享$c$个共同好友”）来解释推荐内容，以提高透明度和用户体验。

In terms of algorithm design, this problem is interesting from the perspectives of both theory and practice. Theoretically speaking, it admits a naive solution that simply compares all pairs of sets, and finishes in $O\left( {n}^{2}\right)$ time,where $n$ is the total size of all the sets. Existing approaches $\left\lbrack  {3,{13},{30}}\right\rbrack$ utilize various heuristics to improve efficiency. However, unfortunately, all of them are still captured by the $O\left( {n}^{2}\right)$ bound,namely,asymptotically as bad as the naive solution. Practically speaking, it has been observed that the "realistic" inputs to the problem appear much easier than the theoretical "worst case", which explains the community's enthusiasm for purely heuristic solutions so far.

在算法设计方面，这个问题从理论和实践的角度来看都很有趣。从理论上讲，它有一个简单的解决方案，即简单地比较所有的集合对，并在$O\left( {n}^{2}\right)$时间内完成，其中$n$是所有集合的总大小。现有的方法$\left\lbrack  {3,{13},{30}}\right\rbrack$利用各种启发式方法来提高效率。然而，不幸的是，所有这些方法仍然受到$O\left( {n}^{2}\right)$界限的限制，即渐近性能与简单解决方案一样差。从实践上讲，已经观察到该问题的“实际”输入似乎比理论上的“最坏情况”容易得多，这解释了到目前为止学术界对纯启发式解决方案的热情。

This paper makes progress on both fronts simultaneously. At the philosophical level, we show that there does not need to be a fine line between theory and practice, as opposed to what was conceived previously. For this purpose, we propose a framework that (i) in theory, gives the first algorithm that escapes the quadratic trap, and (ii) in practice, can be easily integrated with clever heuristics to yield new solutions that improve the efficiency of the state-of-the-art. Specifically, our contributions can be summarized as follows.

本文同时在这两个方面取得了进展。从理念层面上讲，我们表明理论和实践之间不需要有明确的界限，这与之前的设想相反。为此，我们提出了一个框架，（i）在理论上，给出了第一个摆脱二次复杂度陷阱的算法；（ii）在实践中，可以很容易地与巧妙的启发式方法相结合，产生新的解决方案，提高现有技术的效率。具体来说，我们的贡献可以总结如下。

Theoretical Guarantee: We present a size-aware algorithm that has the time complexity of $O\left( {{n}^{2 - \frac{1}{c}}{k}^{\frac{1}{2c}}}\right)  = o\left( {n}^{2}\right)  + O\left( k\right)$ ,where $k$ is the number of results. This is $o\left( {n}^{2}\right)$ as long as $k = o\left( {n}^{2}\right)$ ; on the other hand,if $k = \Omega \left( {n}^{2}\right)$ ,then any algorithm must incur $\Omega \left( {n}^{2}\right)$ time just to output the results. Therefore, our algorithm beats the quadratic complexity, whenever possible.

理论保证：我们提出了一种考虑大小的算法，其时间复杂度为$O\left( {{n}^{2 - \frac{1}{c}}{k}^{\frac{1}{2c}}}\right)  = o\left( {n}^{2}\right)  + O\left( k\right)$，其中$k$是结果的数量。只要$k = o\left( {n}^{2}\right)$，该复杂度就是$o\left( {n}^{2}\right)$；另一方面，如果$k = \Omega \left( {n}^{2}\right)$，那么任何算法仅输出结果就必须花费$\Omega \left( {n}^{2}\right)$时间。因此，只要有可能，我们的算法就能突破二次复杂度。

Practical Performance: The size-aware algorithm divides all the sets into small sets and large sets based on their sizes and processes them separately using two different methods. The two methods are size sensitive, i.e., one method is more efficient for small sets and the other one is more effective for large sets. We can utilize existing studies to process large sets, and focus on the small sets in this paper. For the small sets, we enumerate all their subsets with size $c$ and take any two small sets sharing a common subset as a result. We develop optimization techniques to avoid enumerating a huge number of unnecessary subsets and improve the practical performance dramatically. Furthermore, as the size boundary between the small sets and the large sets is crucial to the efficiency, we propose an effective size boundary selection algorithm to judiciously choose a size boundary. Our optimization techniques can improve the practical performance dramatically. We have conducted extensive experiments on real datasets and the experimental results show that our method outperforms state-of-the-art methods by up to an order of magnitude.

实际性能：考虑大小的算法根据集合的大小将所有集合分为小集合和大集合，并使用两种不同的方法分别处理它们。这两种方法对大小敏感，即一种方法对小集合更有效，另一种方法对大集合更有效。我们可以利用现有的研究来处理大集合，而本文主要关注小集合。对于小集合，我们枚举它们所有大小为$c$的子集，并将任何共享一个共同子集的两个小集合作为一个结果。我们开发了优化技术，以避免枚举大量不必要的子集，并显著提高实际性能。此外，由于小集合和大集合之间的大小边界对效率至关重要，我们提出了一种有效的大小边界选择算法，以明智地选择一个大小边界。我们的优化技术可以显著提高实际性能。我们在真实数据集上进行了广泛的实验，实验结果表明，我们的方法比现有技术方法的性能高出一个数量级。

---

<!-- Footnote -->

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.SIGMOD'18, June 10-15, 2018, Houston, TX, USA © 2018 Association for Computing Machinery. ACM ISBN ISBN 978-1-4503-4703-7/18/06...\$15.00 https://doi.org/http://dx.doi.org/10.1145/XXXXXX.XXXXXX

允许个人或课堂使用本作品的全部或部分内容制作数字或硬拷贝，无需付费，但前提是这些拷贝不得用于盈利或商业目的，且必须在首页注明此声明和完整引用信息。对于本作品中不属于美国计算机协会（ACM）的部分，其版权必须得到尊重。允许在注明出处的情况下进行摘要引用。若要以其他方式复制、重新发布、上传到服务器或分发给列表，则需要事先获得特定许可和/或支付费用。请向permissions@acm.org申请许可。SIGMOD'18，2018年6月10 - 15日，美国得克萨斯州休斯顿市 © 2018美国计算机协会。ACM国际标准书号ISBN 978 - 1 - 4503 - 4703 - 7/18/06... 15.00美元 https://doi.org/http://dx.doi.org/10.1145/XXXXXX.XXXXXX

<!-- Footnote -->

---

<!-- Media -->

<table><tr><td/><td>${id}$</td><td>set</td><td>size</td></tr><tr><td rowspan="4">${\mathcal{R}}_{s}$</td><td>${R}_{1}$</td><td>$\left\{  {{e}_{1},{e}_{2},{e}_{3}}\right\}$</td><td>3</td></tr><tr><td>${R}_{2}$</td><td>$\left\{  {{e}_{1},{e}_{3},{e}_{4},{e}_{7}}\right\}$</td><td>4</td></tr><tr><td>${R}_{3}$</td><td>$\left\{  {{e}_{1},{e}_{3},{e}_{5},{e}_{7}}\right\}$</td><td>4</td></tr><tr><td>${R}_{4}$</td><td>$\left\{  {{e}_{2},{e}_{4},{e}_{5},{e}_{6}}\right\}$</td><td>4</td></tr><tr><td rowspan="3">${\mathcal{R}}_{l}$</td><td>${R}_{5}$</td><td>$\left\{  {{e}_{2},{e}_{4},{e}_{5},{e}_{6},{e}_{8},{e}_{9},{e}_{10},{e}_{11}}\right\}$</td><td>8</td></tr><tr><td>${R}_{6}$</td><td>$\left\{  {{e}_{11},{e}_{12},{e}_{13},{e}_{14},{e}_{15},{e}_{16},{e}_{17},{e}_{18}}\right\}$</td><td>8</td></tr><tr><td>${R}_{7}$</td><td>$\left\{  {{e}_{11},{e}_{12},{e}_{13},{e}_{14},{e}_{15},{e}_{16},{e}_{17},{e}_{18},{e}_{19}}\right\}$</td><td>9</td></tr></table>

<table><tbody><tr><td></td><td>${id}$</td><td>集合（set）</td><td>大小；规模；尺寸（size）</td></tr><tr><td rowspan="4">${\mathcal{R}}_{s}$</td><td>${R}_{1}$</td><td>$\left\{  {{e}_{1},{e}_{2},{e}_{3}}\right\}$</td><td>3</td></tr><tr><td>${R}_{2}$</td><td>$\left\{  {{e}_{1},{e}_{3},{e}_{4},{e}_{7}}\right\}$</td><td>4</td></tr><tr><td>${R}_{3}$</td><td>$\left\{  {{e}_{1},{e}_{3},{e}_{5},{e}_{7}}\right\}$</td><td>4</td></tr><tr><td>${R}_{4}$</td><td>$\left\{  {{e}_{2},{e}_{4},{e}_{5},{e}_{6}}\right\}$</td><td>4</td></tr><tr><td rowspan="3">${\mathcal{R}}_{l}$</td><td>${R}_{5}$</td><td>$\left\{  {{e}_{2},{e}_{4},{e}_{5},{e}_{6},{e}_{8},{e}_{9},{e}_{10},{e}_{11}}\right\}$</td><td>8</td></tr><tr><td>${R}_{6}$</td><td>$\left\{  {{e}_{11},{e}_{12},{e}_{13},{e}_{14},{e}_{15},{e}_{16},{e}_{17},{e}_{18}}\right\}$</td><td>8</td></tr><tr><td>${R}_{7}$</td><td>$\left\{  {{e}_{11},{e}_{12},{e}_{13},{e}_{14},{e}_{15},{e}_{16},{e}_{17},{e}_{18},{e}_{19}}\right\}$</td><td>9</td></tr></tbody></table>

Table 1: A collection $\mathcal{R}$ of sets

表1：集合的集合$\mathcal{R}$

<!-- Media -->

The rest of the paper is organized as follows. We formulate the problem in Section 2. Section 3 introduces the size-aware algorithm. We develop the optimization heuristics for the small sets in Section 4. We propose the size boundary selection algorithm in Section 5. Section 6 reports the experimental results. We review related work in Section 7 and conclude in Section 8.

本文的其余部分组织如下。我们在第2节中阐述问题。第3节介绍考虑大小的算法。我们在第4节中为小集合开发优化启发式方法。我们在第5节中提出大小边界选择算法。第6节报告实验结果。我们在第7节回顾相关工作，并在第8节进行总结。

## 2 PROBLEM DEFINITION

## 2 问题定义

Given two collections of sets, the set similarity join problem aims to find all the similar set pairs from the two collections. We use the overlap similarity to measure the similarity between two sets in this paper. Given two sets $R$ and $S$ ,their overlap similarity is their intersection size,namely $\left| {R \cap  S}\right|$ . Two sets are said to be similar if and only if their overlap similarity is at least a given threshold $c$ ,i.e., $\left| {R \cap  S}\right|  \geq  c$ . Next we formally define the problem of set similarity joins with overlap constraints.

给定两个集合的集合，集合相似性连接问题旨在从这两个集合中找出所有相似的集合对。在本文中，我们使用重叠相似性来衡量两个集合之间的相似性。给定两个集合$R$和$S$，它们的重叠相似性是它们的交集大小，即$\left| {R \cap  S}\right|$。当且仅当两个集合的重叠相似性至少为给定的阈值$c$时，即$\left| {R \cap  S}\right|  \geq  c$，这两个集合才被认为是相似的。接下来，我们正式定义具有重叠约束的集合相似性连接问题。

DEFINITION 1. Given two collections of sets $\mathcal{R}$ and $\mathcal{S}$ and a constant $c$ ,the set similarity join with overlap constraints reports all the set pairs $\langle R,S\rangle  \in  \mathcal{R} \times  \mathcal{S}$ such that $\left| {R \cap  S}\right|  \geq  c$ .

定义1。给定两个集合的集合$\mathcal{R}$和$\mathcal{S}$以及一个常数$c$，具有重叠约束的集合相似性连接报告所有满足$\left| {R \cap  S}\right|  \geq  c$的集合对$\langle R,S\rangle  \in  \mathcal{R} \times  \mathcal{S}$。

We first focus on the self-join case in this paper,i.e., $\mathcal{R} = \mathcal{S}$ . Our technical and theoretical results can be seamlessly extended to the case of $\mathcal{R} \neq  \mathcal{S}$ ,which are discussed in Appendix B. For example, consider the dataset $\mathcal{R}$ in Table 1. Suppose the overlap similarity threshold $c$ is 2. ${R}_{1}$ and ${R}_{2}$ make a similar set pair as $\left| {{R}_{1} \cap  {R}_{2}}\right|  = 2 \geq  c$ . In our running example, $\left\langle  {{R}_{1},{R}_{3}}\right\rangle  ,\left\langle  {{R}_{2},{R}_{3}}\right\rangle  ,\left\langle  {{R}_{4},{R}_{5}}\right\rangle$ and $\left\langle  {{R}_{6},{R}_{7}}\right\rangle$ are also similar pairs.

在本文中，我们首先关注自连接的情况，即$\mathcal{R} = \mathcal{S}$。我们的技术和理论结果可以无缝扩展到$\mathcal{R} \neq  \mathcal{S}$的情况，这将在附录B中讨论。例如，考虑表1中的数据集$\mathcal{R}$。假设重叠相似性阈值$c$为2。由于$\left| {{R}_{1} \cap  {R}_{2}}\right|  = 2 \geq  c$，${R}_{1}$和${R}_{2}$构成一个相似的集合对。在我们的运行示例中，$\left\langle  {{R}_{1},{R}_{3}}\right\rangle  ,\left\langle  {{R}_{2},{R}_{3}}\right\rangle  ,\left\langle  {{R}_{4},{R}_{5}}\right\rangle$和$\left\langle  {{R}_{6},{R}_{7}}\right\rangle$也是相似的对。

A brute-force method enumerates every set pair in $\mathcal{R} \times  \mathcal{R}$ and calculates their overlap size. Let $n = \mathop{\sum }\limits_{{{R}_{i} \in  \mathcal{R}}}\left| {R}_{i}\right|$ be the total size of all sets. The brute-force method has a time complexity of $O\left( {n}^{2}\right)$ .

一种暴力方法会枚举$\mathcal{R} \times  \mathcal{R}$中的每一个集合对，并计算它们的重叠大小。设$n = \mathop{\sum }\limits_{{{R}_{i} \in  \mathcal{R}}}\left| {R}_{i}\right|$为所有集合的总大小。暴力方法的时间复杂度为$O\left( {n}^{2}\right)$。

## 3 A SIZE-AWARE ALGORITHM

## 3 考虑大小的算法

In this section, we present an algorithm that solves the set similarity join problem with running time $o\left( {n}^{2}\right)  + O\left( k\right)$ ,where $k$ is the number of pairs in the result. This is the first algorithm that beats the quadratic time complexity of this problem whenever it is possible. Section 3.1 will describe the overall framework of our solution, but leave open the choice of a crucial parameter. Section 3.2 will explain how to set that parameter to achieve the best time complexity. We will discuss how to improve the practical performance of this algorithm in Sections 4 and 5.

在本节中，我们提出一种算法，该算法解决集合相似性连接问题的运行时间为$o\left( {n}^{2}\right)  + O\left( k\right)$，其中$k$是结果中的对的数量。这是第一种在可能的情况下打破该问题二次时间复杂度的算法。第3.1节将描述我们解决方案的总体框架，但不确定一个关键参数的选择。第3.2节将解释如何设置该参数以实现最佳时间复杂度。我们将在第4节和第5节讨论如何提高该算法的实际性能。

<!-- Media -->

Algorithm 1: SizeAwareAlgorithm

算法1：考虑大小的算法

---

Input: $\mathcal{R}$ : the dataset $\left\{  {{R}_{1},{R}_{2},\ldots ,{R}_{m}}\right\}  ;c$ : threshold;

输入：$\mathcal{R}$：数据集；$\left\{  {{R}_{1},{R}_{2},\ldots ,{R}_{m}}\right\}  ;c$：阈值；

Output: $\mathcal{A} = \left\{  {\left\langle  {{R}_{i},{R}_{j}}\right\rangle  \left| {{R}_{i} \cap  {R}_{j}}\right|  \geq  c}\right\}$ ;

输出：$\mathcal{A} = \left\{  {\left\langle  {{R}_{i},{R}_{j}}\right\rangle  \left| {{R}_{i} \cap  {R}_{j}}\right|  \geq  c}\right\}$；

$x =$ GetSizeBoundary(R,c);

$x =$ 获取大小边界(R,c);

divide $\mathcal{R}$ into small sets ${\mathcal{R}}_{s}$ and large sets ${\mathcal{R}}_{l}$ by $x$ ;

通过$x$将$\mathcal{R}$划分为小集合${\mathcal{R}}_{s}$和大集合${\mathcal{R}}_{l}$；

foreach large set ${R}_{i} \in  {\mathcal{R}}_{l}$ do

对每个大集合${R}_{i} \in  {\mathcal{R}}_{l}$执行

	foreach ${R}_{j} \in  \mathcal{R}$ do

	  对${R}_{j} \in  \mathcal{R}$执行

		if $\left| {{R}_{i} \cap  {R}_{j}}\right|  \geq  c$ then insert $\left\langle  {{R}_{i},{R}_{j}}\right\rangle$ into $\mathcal{A}$

		  如果$\left| {{R}_{i} \cap  {R}_{j}}\right|  \geq  c$，则将$\left\langle  {{R}_{i},{R}_{j}}\right\rangle$插入到$\mathcal{A}$中

foreach small set ${R}_{j} \in  {\mathcal{R}}_{s}$ do

对每个小集合${R}_{j} \in  {\mathcal{R}}_{s}$执行

	foreach $c$ -subset ${r}_{c}$ of ${R}_{j}$ do

	  对${R}_{j}$的$c$ - 子集${r}_{c}$执行

		append ${R}_{j}$ to $\mathcal{L}\left\lbrack  {\mathbf{r}}_{c}\right\rbrack$ ;

		  将${R}_{j}$追加到$\mathcal{L}\left\lbrack  {\mathbf{r}}_{c}\right\rbrack$中；

foreach inverted list $\mathcal{L}\left\lbrack  {\mathrm{r}}_{c}\right\rbrack$ in $\mathcal{L}$ do

对$\mathcal{L}$中的每个倒排表$\mathcal{L}\left\lbrack  {\mathrm{r}}_{c}\right\rbrack$执行

	add every set pair in $\mathcal{L}\left\lbrack  {r}_{c}\right\rbrack$ into $\mathcal{A}$ ;

	将$\mathcal{L}\left\lbrack  {r}_{c}\right\rbrack$中的每个集合对添加到$\mathcal{A}$中；

return $\mathcal{A}$ ;

返回$\mathcal{A}$；

---

<!-- Media -->

### 3.1 The Framework

### 3.1 框架

We will use the term $c$ -subset to refer to any set of $c$ elements (drawn from the sets in $\mathcal{R}$ and $\mathcal{S}$ ). It is easy to see that,two sets are similar if and only if they share a common $c$ -subset. The observation motivates us to build an inverted index on all the $c$ -subsets to aggregate those sets sharing common $c$ -subsets,and compute the join result by examining each inverted list in turn. Thus, we avoid the enumeration of dissimilar set pairs, i.e., set pairs that do not share any common $c$ -subsets. This approach,however,works well only for sets with small sizes,as they have a small number of $c$ - subsets. On the other hand, the number of large sets cannot be very large, such that we can afford to apply even a "brute-force" method on them. Next, we develop these ideas into a formal algorithm.

我们将使用术语$c$ - 子集来指代任何由$c$个元素组成的集合（这些元素从$\mathcal{R}$和$\mathcal{S}$中的集合中选取）。很容易看出，当且仅当两个集合共享一个共同的$c$ - 子集时，它们才相似。这一观察结果促使我们在所有$c$ - 子集上构建一个倒排索引，以聚合那些共享共同$c$ - 子集的集合，并通过依次检查每个倒排表来计算连接结果。因此，我们避免了对不相似集合对（即不共享任何共同$c$ - 子集的集合对）的枚举。然而，这种方法仅适用于小尺寸的集合，因为它们的$c$ - 子集数量较少。另一方面，大集合的数量不能太多，这样我们甚至可以对它们应用“暴力”方法。接下来，我们将这些想法发展成一个正式的算法。

Given a collection $\mathcal{R}$ of sets ${R}_{1},{R}_{2},\ldots ,{R}_{m}$ and an overlap similarity threshold $c$ ,we divide all the sets into two categories based on their sizes. The first category ${\mathcal{R}}_{l}$ contains all the sets with sizes at least $x$ -the selection of the size boundary $x$ will be discussed later-which we refer to as the large sets. The second category ${\mathcal{R}}_{s}$ contains all the sets with sizes smaller than $x$ ,which we refer to as the small sets. Obviously,any similar set pair in $\mathcal{R} \times  \mathcal{R}$ can be found in either ${\mathcal{R}}_{l} \times  \mathcal{R}$ or ${\mathcal{R}}_{s} \times  {\mathcal{R}}_{s}$ .

给定集合${R}_{1},{R}_{2},\ldots ,{R}_{m}$的集合族$\mathcal{R}$和重叠相似度阈值$c$，我们根据集合的大小将所有集合分为两类。第一类${\mathcal{R}}_{l}$包含所有大小至少为$x$的集合（大小边界$x$的选择将在后面讨论），我们将其称为大集合。第二类${\mathcal{R}}_{s}$包含所有大小小于$x$的集合，我们将其称为小集合。显然，$\mathcal{R} \times  \mathcal{R}$中的任何相似集合对都可以在${\mathcal{R}}_{l} \times  \mathcal{R}$或${\mathcal{R}}_{s} \times  {\mathcal{R}}_{s}$中找到。

We obtain the similar set pairs in ${\mathcal{R}}_{l} \times  \mathcal{R}$ and ${\mathcal{R}}_{s} \times  {\mathcal{R}}_{s}$ in different ways:

我们以不同的方式获得${\mathcal{R}}_{l} \times  \mathcal{R}$和${\mathcal{R}}_{s} \times  {\mathcal{R}}_{s}$中的相似集合对：

- For ${\mathcal{R}}_{l} \times  \mathcal{R}$ ,simply enumerate every set pair in ${\mathcal{R}}_{l} \times  \mathcal{R}$ ,and calculate their intersection size.

- 对于${\mathcal{R}}_{l} \times  \mathcal{R}$，只需枚举${\mathcal{R}}_{l} \times  \mathcal{R}$中的每个集合对，并计算它们的交集大小。

- To find all the similar set pairs from ${\mathcal{R}}_{s} \times  {\mathcal{R}}_{s}$ ,we first build a $c$ -subset inverted index $\mathcal{L}$ for all the $c$ -subsets in the small sets. The inverted list $\mathcal{L}\left\lbrack  {\mathbf{r}}_{c}\right\rbrack$ consists of all the small sets that contain the $c$ -subset ${r}_{c}$ . Then,we access each inverted list, and add every set pair in it into the result set (i.e., for any two distinct sets $R$ and ${R}^{\prime }$ in the inverted list,add $\left\langle  {R,{R}^{\prime }}\right\rangle$ to the result). This produces all the similar set pairs in ${\mathcal{R}}_{s} \times  {\mathcal{R}}_{s}$ .

- 为了从${\mathcal{R}}_{s} \times  {\mathcal{R}}_{s}$中找出所有相似的集合对，我们首先为小集合中的所有$c$ -子集构建一个$c$ -子集倒排索引$\mathcal{L}$。倒排列表$\mathcal{L}\left\lbrack  {\mathbf{r}}_{c}\right\rbrack$由包含$c$ -子集${r}_{c}$的所有小集合组成。然后，我们访问每个倒排列表，并将其中的每个集合对添加到结果集中（即，对于倒排列表中任意两个不同的集合$R$和${R}^{\prime }$，将$\left\langle  {R,{R}^{\prime }}\right\rangle$添加到结果中）。这样就能得到${\mathcal{R}}_{s} \times  {\mathcal{R}}_{s}$中所有相似的集合对。

The pseudo-code of the above algorithm is shown in Algorithm 1. It takes a collection of sets $\mathcal{R} = \left\{  {{R}_{1},{R}_{2},\ldots ,{R}_{m}}\right\}$ and a constant threshold $c$ as input,and outputs all the similar set pairs. It first calculates the size boundary $x$ ,and then divides all the sets in $\mathcal{R}$ into two categories,the small sets ${\mathcal{R}}_{s}$ and the large sets ${\mathcal{R}}_{l}$ ,based on $x$ (Lines 1 to 2). For each set pair $\left\langle  {{R}_{i},{R}_{j}}\right\rangle$ in ${\mathcal{R}}_{l} \times  \mathcal{R}$ ,it adds the pair to the result set $\mathcal{A}$ if their intersection size is at least $c$ (Lines 3 to 5). Next,for each small set ${R}_{j} \in  {\mathcal{R}}_{s}$ ,it enumerates all its $c$ -subsets, and inserts them to the inverted index $\mathcal{L}$ (Lines 6 to 8). For each inverted list in $\mathcal{L}$ ,it adds every set pair in it to $\mathcal{A}$ (Lines 9 to 10). Finally,the algorithm returns $\mathcal{A}$ (Line 11).

上述算法的伪代码如算法1所示。该算法以一组集合$\mathcal{R} = \left\{  {{R}_{1},{R}_{2},\ldots ,{R}_{m}}\right\}$和一个常量阈值$c$作为输入，并输出所有相似的集合对。它首先计算大小边界$x$，然后根据$x$将$\mathcal{R}$中的所有集合分为两类，即小集合${\mathcal{R}}_{s}$和大集合${\mathcal{R}}_{l}$（第1行到第2行）。对于${\mathcal{R}}_{l} \times  \mathcal{R}$中的每个集合对$\left\langle  {{R}_{i},{R}_{j}}\right\rangle$，如果它们的交集大小至少为$c$，则将该对添加到结果集$\mathcal{A}$中（第3行到第5行）。接下来，对于每个小集合${R}_{j} \in  {\mathcal{R}}_{s}$，枚举其所有的$c$ -子集，并将它们插入到倒排索引$\mathcal{L}$中（第6行到第8行）。对于$\mathcal{L}$中的每个倒排列表，将其中的每个集合对添加到$\mathcal{A}$中（第9行到第10行）。最后，算法返回$\mathcal{A}$（第11行）。

<!-- Media -->

<table><tr><td/><td>${R}_{1}$</td><td>${R}_{2}$</td><td>${R}_{3}$</td><td>${R}_{4}$</td><td>${R}_{5}$</td><td>${R}_{6}$</td><td>${R}_{7}$</td></tr><tr><td>${R}_{5}$</td><td>1</td><td>1</td><td>1</td><td>4</td><td>-</td><td>-</td><td>-</td></tr><tr><td>${R}_{6}$</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>-</td><td>-</td></tr><tr><td>${R}_{7}$</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>8</td><td>-</td></tr></table>

<table><tr><td/><td>${R}_{1}$</td><td>${R}_{2}$</td><td>${R}_{3}$</td><td>${R}_{4}$</td><td>${R}_{5}$</td><td>${R}_{6}$</td><td>${R}_{7}$</td></tr><tr><td>${R}_{5}$</td><td>1</td><td>1</td><td>1</td><td>4</td><td>-</td><td>-</td><td>-</td></tr><tr><td>${R}_{6}$</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>-</td><td>-</td></tr><tr><td>${R}_{7}$</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>8</td><td>-</td></tr></table>

Figure 1: ${\mathcal{R}}_{l} \times  \mathcal{R}$

图1：${\mathcal{R}}_{l} \times  \mathcal{R}$

<!-- figureText: ${e}_{1}{e}_{2}$ ${e}_{I}{e}_{3}$ ${e}_{1}{e}_{4}$ ${e}_{l}{e}_{j}$ ${e}_{1}{e}_{7}$ ${e}_{2}{e}_{3}$ ${e}_{2}{e}_{4}$ ${e}_{2}{e}_{5}$ block ${e}_{3}$ block ${e}_{4}$ block ${e}_{5}$ ${e}_{2}{e}_{6}$ ${e}_{3}{e}_{4}$ ${e}_{3}{e}_{5}$ ${e}_{3}{e}_{7}$ ${e}_{4}{e}_{5}$ ${e}_{4}{e}_{6}$ ${e}_{4}{e}_{7}$ ${e}_{5}{e}_{6}$ ${e}_{5}{e}_{7}$ unique c-subsets ${R}_{l}$ ${R}_{2}$ ${R}_{3}$ ${R}_{4}$ redundants unique c-subsets -->

<img src="https://cdn.noedgeai.com/0195ccc7-1611-78aa-a97d-afb7fc00df51_2.jpg?x=661&y=159&w=973&h=223&r=0"/>

Figure 2: The $c$ -subset inverted index for small sets.

图2：小集合的$c$ -子集倒排索引。

<!-- Media -->

EXAMPLE 1. Consider the dataset $\mathcal{R}$ in Table 1,and suppose that the threshold is $c = 2$ . As explained in the next section,the size boundary is $x = 5$ . Thus,we have ${\mathcal{R}}_{s} = \left\{  {{R}_{1},{R}_{2},{R}_{3},{R}_{4}}\right\}$ and ${\mathcal{R}}_{l} = \left\{  {{R}_{5},{R}_{6},{R}_{7}}\right\}$ . As shown in Figure 1,we enumerate every set pair in ${\mathcal{R}}_{l} \times  \mathcal{R}$ ,and calculate their intersection size,which yields two similar pairs $\left\langle  {{R}_{4},{R}_{5}}\right\rangle$ and $\left\langle  {{R}_{6},{R}_{7}}\right\rangle$ . Then,we build the inverted index for all the 2-subsets found in the small sets. The index is shown in Figure 2, where a black block indicates the existence of this c-subset in the corresponding small set. The inverted list $\mathcal{L}\left\lbrack  \left\{  {{e}_{1},{e}_{3}}\right\}  \right\rbrack$ has three sets ${R}_{1},{R}_{2}$ and ${R}_{3}$ , according to which we obtain three similar pairs $\left\langle  {{R}_{1},{R}_{2}}\right\rangle  ,\left\langle  {{R}_{1},{R}_{3}}\right\rangle$ , and $\left\langle  {{R}_{2},{R}_{3}}\right\rangle$ . Similarly,a similar pair is spawned from the inverted list $\mathcal{L}\left\lbrack  \left\{  {{e}_{1},{e}_{7}}\right\}  \right\rbrack$ ,and another from $\mathcal{L}\left\lbrack  \left\{  {{e}_{3},{e}_{7}}\right\}  \right\rbrack$ . However,these two pairs have appeared earlier, and hence, are duplicates. In total, the join result consists of 5 pairs.

示例1。考虑表1中的数据集$\mathcal{R}$，并假设阈值为$c = 2$。如下一节所述，大小边界为$x = 5$。因此，我们有${\mathcal{R}}_{s} = \left\{  {{R}_{1},{R}_{2},{R}_{3},{R}_{4}}\right\}$和${\mathcal{R}}_{l} = \left\{  {{R}_{5},{R}_{6},{R}_{7}}\right\}$。如图1所示，我们枚举${\mathcal{R}}_{l} \times  \mathcal{R}$中的每一对集合，并计算它们的交集大小，得到两对相似的集合对$\left\langle  {{R}_{4},{R}_{5}}\right\rangle$和$\left\langle  {{R}_{6},{R}_{7}}\right\rangle$。然后，我们为在小集合中找到的所有2 -子集构建倒排索引。该索引如图2所示，其中黑色方块表示该c -子集在相应小集合中存在。倒排列表$\mathcal{L}\left\lbrack  \left\{  {{e}_{1},{e}_{3}}\right\}  \right\rbrack$包含三个集合${R}_{1},{R}_{2}$和${R}_{3}$，据此我们得到三对相似的集合对$\left\langle  {{R}_{1},{R}_{2}}\right\rangle  ,\left\langle  {{R}_{1},{R}_{3}}\right\rangle$和$\left\langle  {{R}_{2},{R}_{3}}\right\rangle$。类似地，从倒排列表$\mathcal{L}\left\lbrack  \left\{  {{e}_{1},{e}_{7}}\right\}  \right\rbrack$产生一对相似的集合对，从$\mathcal{L}\left\lbrack  \left\{  {{e}_{3},{e}_{7}}\right\}  \right\rbrack$产生另一对。然而，这两对之前已经出现过，因此是重复的。总的来说，连接结果由5对组成。

Intuition Behind the Size Aware Algorithm. Existing approaches build an inverted index $\mathcal{I}$ for the elements in the sets. As will be discussed in Section 6.1,each inverted list $\mathcal{I}\left\lbrack  e\right\rbrack$ is scanned $\left| {\mathcal{I}\left\lbrack  e\right\rbrack  }\right|$ times where $\left| {\mathcal{I}\left\lbrack  e\right\rbrack  }\right|$ is the inverted list length. Thus they need $O\left( {\left| \mathcal{I}\left\lbrack  e\right\rbrack  \right| }^{2}\right)$ time to process each inverted list $\mathcal{I}\left\lbrack  e\right\rbrack$ . Notice that,as there are $n$ elements in total,the number of large sets cannot exceed $\frac{n}{x}$ . Thus the large sets contribute at most $\frac{n}{x}$ to the inverted list length. On the contrary, there is no bound for the number of small sets and they could contribute up to $O\left( n\right)$ to the inverted list length which results in a time complexity of $O\left( {n}^{2}\right)$ . This is why existing methods fail to perform effectively over small sets. This is also why we can afford using any existing methods to process large sets while have to design a new method for the small sets.

大小感知算法的原理。现有方法为集合中的元素构建倒排索引$\mathcal{I}$。正如将在6.1节中讨论的，每个倒排列表$\mathcal{I}\left\lbrack  e\right\rbrack$会被扫描$\left| {\mathcal{I}\left\lbrack  e\right\rbrack  }\right|$次，其中$\left| {\mathcal{I}\left\lbrack  e\right\rbrack  }\right|$是倒排列表的长度。因此，处理每个倒排列表$\mathcal{I}\left\lbrack  e\right\rbrack$需要$O\left( {\left| \mathcal{I}\left\lbrack  e\right\rbrack  \right| }^{2}\right)$的时间。注意，由于总共有$n$个元素，大集合的数量不能超过$\frac{n}{x}$。因此，大集合对倒排列表长度的贡献最多为$\frac{n}{x}$。相反，小集合的数量没有上限，它们对倒排列表长度的贡献最多可达$O\left( n\right)$，这导致时间复杂度为$O\left( {n}^{2}\right)$。这就是为什么现有方法在处理小集合时不能有效执行的原因。这也是为什么我们可以使用任何现有方法来处理大集合，而必须为小集合设计一种新方法的原因。

Remark. Obviously it is expensive to enumerate all $c$ -subsets in every small set, especially when the small sets have large sizes. To address this issue, we propose various techniques to avoid enumerating a large number of them in Section 4 and our method has both theoretical and practical guarantees. In addition, any existing method can be plugged in our framework to process the large sets for better practical performance. In particular, we use ScanCount [30] as described in Section 6.1 in our implementation.

备注：显然，枚举每个小集合中的所有 $c$ -子集的代价很高，尤其是当小集合的规模较大时。为解决这一问题，我们在第4节中提出了各种技术来避免枚举大量的此类子集，并且我们的方法在理论和实践上都有保障。此外，任何现有的方法都可以嵌入到我们的框架中，以处理大集合，从而获得更好的实际性能。具体而言，我们在实现中使用了第6.1节中描述的ScanCount [30] 方法。

### 3.2 Size Boundary Selection in Theory

### 3.2 理论上的规模边界选择

It remains to clarify the setting of the size boundary $x$ (which divides the small sets from the large ones). We will adopt an analytic approach: bounding the running time of our algorithm as a function of $x$ ,and then finding the best $x$ to minimize the cost.

仍需明确规模边界 $x$ 的设置（该边界将小集合与大集合区分开来）。我们将采用一种解析方法：将我们算法的运行时间表示为 $x$ 的函数，然后找到使成本最小化的最优 $x$ 。

Running Time as a Function of $x$ : Let us first analyze the time complexity of finding the similar pairs in ${\mathcal{R}}_{l} \times  \mathcal{R}$ . Recall that our algorithm calculates the intersection size $\left| {R \cap  {R}^{\prime }}\right|$ for every pair of $\left\langle  {R,{R}^{\prime }}\right\rangle   \in  {\mathcal{R}}_{l} \times  \mathcal{R}$ . To do so efficiently,we create a hash table on every large set $R \in  {\mathcal{R}}_{l}$ so that whether an element $e$ belongs to $R$ can be determined in constant time. Then, $\left| {R \cap  {R}^{\prime }}\right|$ can be computed in $O\left( \left| {R}^{\prime }\right| \right)$ time,by probing the hash table of $R$ with every element in ${R}^{\prime }$ . In other words,the computation of $\left| {R \cap  {R}^{\prime }}\right|$ for the same ${R}^{\prime }$ but all $R \in  {\mathcal{R}}_{l}$ can be accomplished in $O\left( {\left| {R}^{\prime }\right|  \cdot  \frac{n}{x}}\right)$ time. Therefore, the size $\left| {R \cap  {R}^{\prime }}\right|$ of all $\left\langle  {R,{R}^{\prime }}\right\rangle   \in  {\mathcal{R}}_{l} \times  \mathcal{R}$ can be obtained in time

作为 $x$ 函数的运行时间：让我们首先分析在 ${\mathcal{R}}_{l} \times  \mathcal{R}$ 中查找相似对的时间复杂度。回顾一下，我们的算法会计算每对 $\left\langle  {R,{R}^{\prime }}\right\rangle   \in  {\mathcal{R}}_{l} \times  \mathcal{R}$ 的交集大小 $\left| {R \cap  {R}^{\prime }}\right|$ 。为了高效地完成这一任务，我们在每个大集合 $R \in  {\mathcal{R}}_{l}$ 上创建一个哈希表，这样就可以在常数时间内确定一个元素 $e$ 是否属于 $R$ 。然后，可以在 $O\left( \left| {R}^{\prime }\right| \right)$ 时间内计算出 $\left| {R \cap  {R}^{\prime }}\right|$ ，方法是用 ${R}^{\prime }$ 中的每个元素去探查 $R$ 的哈希表。换句话说，对于相同的 ${R}^{\prime }$ 但所有的 $R \in  {\mathcal{R}}_{l}$ ， $\left| {R \cap  {R}^{\prime }}\right|$ 的计算可以在 $O\left( {\left| {R}^{\prime }\right|  \cdot  \frac{n}{x}}\right)$ 时间内完成。因此，所有 $\left\langle  {R,{R}^{\prime }}\right\rangle   \in  {\mathcal{R}}_{l} \times  \mathcal{R}$ 的大小 $\left| {R \cap  {R}^{\prime }}\right|$ 可以在以下时间内获得

$$
\mathop{\sum }\limits_{{{R}^{\prime } \in  \mathcal{R}}}O\left( {\left| {R}^{\prime }\right|  \cdot  \frac{n}{x}}\right)  = O\left( \frac{{n}^{2}}{x}\right) .
$$

We now proceed to discuss the time complexity of finding similar pairs in ${\mathcal{R}}_{s} \times  {\mathcal{R}}_{s}$ . There are two steps: (i) the first enumerates all the $c$ -subsets to build the inverted index,and (ii) the second generates similar pairs from the inverted lists. A small set $R$ has $\left( \begin{matrix} \left| R\right| \\  c \end{matrix}\right) c$ -subsets. Since $\left| R\right|  \leq  x$ (as $R$ is a small set),the total number of $c$ -subsets from all small sets is at most:

现在我们来讨论在 ${\mathcal{R}}_{s} \times  {\mathcal{R}}_{s}$ 中查找相似对的时间复杂度。有两个步骤：（i）第一步枚举所有的 $c$ -子集以构建倒排索引，（ii）第二步从倒排列表中生成相似对。一个小集合 $R$ 有 $\left( \begin{matrix} \left| R\right| \\  c \end{matrix}\right) c$ -子集。由于 $\left| R\right|  \leq  x$ （因为 $R$ 是小集合），所有小集合中的 $c$ -子集的总数最多为：

$$
\mathop{\sum }\limits_{{R \in  {\mathcal{R}}_{s}}}\left( \begin{matrix} \left| R\right| \\  c \end{matrix}\right)  \leq  \mathop{\sum }\limits_{{R \in  {\mathcal{R}}_{s}}}{\left| R\right| }^{c} \leq  {x}^{c - 1}\mathop{\sum }\limits_{{R \in  {\mathcal{R}}_{s}}}\left| R\right|  \leq  {x}^{c - 1}n.
$$

The enumeration cost in the first step is asymptotically the same as the above number,i.e.,the cost is bounded by $O\left( {{x}^{c - 1}n}\right)$ .

第一步中的枚举成本在渐近意义上与上述数量相同，即成本受 $O\left( {{x}^{c - 1}n}\right)$ 限制。

The cost of the second step comes from generating all the set pairs in each inverted list. Let ${\mathcal{L}}_{1},{\mathcal{L}}_{2},\ldots ,{\mathcal{L}}_{l}$ be all the inverted lists in $\mathcal{L}$ ,and $\left| {\mathcal{L}}_{i}\right|$ be the length of ${\mathcal{L}}_{i}$ . The time complexity of the second step is $O\left( {\mathop{\sum }\limits_{{i = 1}}^{l}{\left| {\mathcal{L}}_{i}\right| }^{2}}\right)$ . As the total length of all the inverted lists is exactly the number of $c$ -subsets in all the small sets,it holds that $\mathop{\sum }\limits_{{i = 1}}^{l}\left| {\mathcal{L}}_{i}\right|  \leq  {x}^{c - 1}n$ . Moreover,for any inverted list ${\mathcal{L}}_{i}$ ,we have $\frac{\left| {\mathcal{L}}_{i}\right| \left( {\left| {\mathcal{L}}_{i}\right|  - 1}\right) }{2} \leq  k$ (remember that $k$ is the total number of similar set pairs in $\mathcal{R} \times  \mathcal{R}$ ) because the number of similar set pairs generated in ${\mathcal{L}}_{i}$ obviously cannot exceed $k$ . It thus follows that $\left| {\mathcal{L}}_{i}\right|  = O\left( \sqrt{k}\right)$ . Hence, the second step runs in time

第二步的成本来自于生成每个倒排列表中的所有集合对。设${\mathcal{L}}_{1},{\mathcal{L}}_{2},\ldots ,{\mathcal{L}}_{l}$为$\mathcal{L}$中的所有倒排列表，$\left| {\mathcal{L}}_{i}\right|$为${\mathcal{L}}_{i}$的长度。第二步的时间复杂度为$O\left( {\mathop{\sum }\limits_{{i = 1}}^{l}{\left| {\mathcal{L}}_{i}\right| }^{2}}\right)$。由于所有倒排列表的总长度恰好是所有小集合中$c$ -子集的数量，因此有$\mathop{\sum }\limits_{{i = 1}}^{l}\left| {\mathcal{L}}_{i}\right|  \leq  {x}^{c - 1}n$。此外，对于任何倒排列表${\mathcal{L}}_{i}$，我们有$\frac{\left| {\mathcal{L}}_{i}\right| \left( {\left| {\mathcal{L}}_{i}\right|  - 1}\right) }{2} \leq  k$（记住$k$是$\mathcal{R} \times  \mathcal{R}$中相似集合对的总数），因为在${\mathcal{L}}_{i}$中生成的相似集合对的数量显然不能超过$k$。因此可得$\left| {\mathcal{L}}_{i}\right|  = O\left( \sqrt{k}\right)$。所以，第二步的运行时间为

$$
O\left( {\mathop{\sum }\limits_{{i = 1}}^{l}{\left| {\mathcal{L}}_{i}\right| }^{2}}\right)  = O\left( {\sqrt{k}\mathop{\sum }\limits_{{i = 1}}^{l}\left| {\mathcal{L}}_{i}\right| }\right)  = O\left( {{x}^{c - 1}n\sqrt{k}}\right) .
$$

Choosing $x$ When $k$ is Known: Let us first make an (unrealistic) assumption that we know in advance the value of $k$ (the assumption will be removed shortly). In this scenario,the best value of $x$ results directly from the earlier analysis. Specifically, as shown above, the overall running time of our algorithm is

已知$k$时选择$x$：让我们首先做一个（不切实际的）假设，即我们预先知道$k$的值（这个假设很快会被去掉）。在这种情况下，$x$的最优值可直接从前面的分析得出。具体来说，如上所示，我们算法的总体运行时间为

$$
O\left( {\frac{{n}^{2}}{x} + {x}^{c - 1}n\sqrt{k}}\right) \text{.}
$$

To minimize the time complexity,we set $x = {\left( n/\sqrt{k}\right) }^{1/c}$ . In this case, the time complexity of our algorithm is

为了使时间复杂度最小化，我们令$x = {\left( n/\sqrt{k}\right) }^{1/c}$。在这种情况下，我们算法的时间复杂度为

$$
O\left( {{n}^{2 - \frac{1}{c}}{k}^{\frac{1}{2c}}}\right) \text{.} \tag{1}
$$

As an example, consider the dataset in Table 1 again with the threshold $c = 2$ . Here, $n = \mathop{\sum }\limits_{{i = 1}}^{7}\left| {R}_{i}\right|  = {40}$ ,and $k = 5$ . Hence,we set $x = {\left( {40}/\sqrt{5}\right) }^{\frac{1}{2}} \approx  5$

例如，再次考虑表1中的数据集，阈值为$c = 2$。这里，$n = \mathop{\sum }\limits_{{i = 1}}^{7}\left| {R}_{i}\right|  = {40}$，且$k = 5$。因此，我们令$x = {\left( {40}/\sqrt{5}\right) }^{\frac{1}{2}} \approx  5$

When $k$ is Not Known-The Doubling Trick: Now we return to the reality where one does not have the precise value of $k$ . Interestingly, even in this case, it is still possible to achieve the same time complexity as (1) with a technique often known as the doubling trick, which is a commonly used technique in theory for analyzing the complexity. The main idea is to guess $k$ starting from a small value. Then, we run the algorithm as if our guess was accurate. If it is, then the algorithm indeed achieves the desired cost; otherwise, we are able to detect the fact that our guess is too low. In the former case, the join problem has already been solved, whereas in the latter, we double our guess for $k$ and repeat. The algorithm eventually terminates; and when it does so, it is guaranteed that (i) our final guess is at most twice the real $k$ ,and that (ii) the total execution time is dominated by that of the last run (with the final guess). ${}^{1}$

未知$k$时——倍增技巧：现在我们回到现实情况，即人们没有$k$的确切值。有趣的是，即使在这种情况下，仍然可以使用一种通常称为倍增技巧的技术达到与(1)相同的时间复杂度，这是理论中用于分析复杂度的常用技术。主要思想是从一个小值开始猜测$k$。然后，我们就像猜测准确一样运行算法。如果猜测准确，那么算法确实能达到预期的成本；否则，我们能够检测到猜测值过低的情况。在前一种情况下，连接问题已经解决，而在后一种情况下，我们将对$k$的猜测值加倍并重复上述过程。算法最终会终止；当算法终止时，可以保证(i)我们的最终猜测值最多是真实$k$的两倍，并且(ii)总执行时间主要由最后一次运行（使用最终猜测值）决定。${}^{1}$

Next, we give the details of the above solution. The solution has multiple rounds. In each round we guess a $k$ and execute the size aware algorithm (Algorithm 1). Let $\widehat{k}$ be our guess of $k$ in the current round. If the guess is accurate, we know from the earlier analysis that, the size aware algorithm must terminate by performing at most $\alpha  \cdot  {n}^{2 - \frac{1}{c}}{\widehat{k}}^{\frac{1}{2c}}$ "micro steps" ( $\alpha$ is the hidden constant in the big- $O$ of (1)),each of which takes $O\left( 1\right)$ time,and can be tracked easily-more specifically, a micro step in our algorithm is one probe in a hash table in processing ${\mathcal{R}}_{l} \times  \mathcal{R}$ ,or the enumeration of one set pair in processing ${\mathcal{R}}_{s} \times  {\mathcal{R}}_{s}$ . Therefore,as soon as $1 + \alpha  \cdot  {n}^{2 - \frac{1}{c}}{\widehat{k}}^{\frac{1}{2c}}$ micro steps have been performed,we know that our guess $\widehat{k}$ is smaller than the real $k$ . Hence,the size aware algorithm can now terminate itself-in which case, we say that the current round has finished. Then,we double $\widehat{k}$ and perform another round until our guess $\widehat{k} \geq  k$ . Note in each round it takes $O\left( {{n}^{2 - \frac{1}{c}}{\widehat{k}}^{\frac{1}{2c}}}\right)$ time.

接下来，我们详细介绍上述解决方案。该解决方案包含多个轮次。在每一轮中，我们猜测一个 $k$ 并执行大小感知算法（算法 1）。设 $\widehat{k}$ 为当前轮次中我们对 $k$ 的猜测值。如果猜测准确，根据前面的分析可知，大小感知算法最多执行 $\alpha  \cdot  {n}^{2 - \frac{1}{c}}{\widehat{k}}^{\frac{1}{2c}}$ 个“微步骤”（ $\alpha$ 是式 (1) 中大 $O$ 记号里的隐藏常数）就一定会终止，每个微步骤耗时 $O\left( 1\right)$ ，并且易于跟踪——更具体地说，我们算法中的一个微步骤是处理 ${\mathcal{R}}_{l} \times  \mathcal{R}$ 时在哈希表中的一次探查，或者是处理 ${\mathcal{R}}_{s} \times  {\mathcal{R}}_{s}$ 时对一组集合对的枚举。因此，一旦执行了 $1 + \alpha  \cdot  {n}^{2 - \frac{1}{c}}{\widehat{k}}^{\frac{1}{2c}}$ 个微步骤，我们就知道我们的猜测值 $\widehat{k}$ 小于真实的 $k$ 。因此，大小感知算法现在可以自行终止——在这种情况下，我们称当前轮次结束。然后，我们将 $\widehat{k}$ 翻倍并进行下一轮，直到我们的猜测 $\widehat{k} \geq  k$ 。注意，每一轮耗时 $O\left( {{n}^{2 - \frac{1}{c}}{\widehat{k}}^{\frac{1}{2c}}}\right)$ 。

To achieve the desired complexity (1),we start with $\widehat{k} = 1$ . At its termination,the value of $\widehat{k}$ is at most ${2k}$ (otherwise,the algorithm would have terminated in the previous round). Therefore, the overall running time of all the rounds is bounded by

为了达到期望的复杂度 (1)，我们从 $\widehat{k} = 1$ 开始。在算法终止时， $\widehat{k}$ 的值至多为 ${2k}$ （否则，算法会在上一轮终止）。因此，所有轮次的总运行时间受限于

$$
O\left( {\mathop{\sum }\limits_{{i = 0}}^{{{\log }_{2}\left( {2k}\right) }}{n}^{2 - \frac{1}{c}}{\left( {2}^{i}\right) }^{\frac{1}{2c}}}\right)  = O\left( {{n}^{2 - \frac{1}{c}}{k}^{\frac{1}{2c}}}\right) .
$$

We thus have proved:

因此，我们证明了：

THEOREM 1. There exists an algorithm with the time complexity $O\left( {{n}^{2 - \frac{1}{c}}{k}^{\frac{1}{2c}}}\right)$ for the set similarity join with overlap constraints problem,where $n$ is the total size of all the sets,constant $c$ is the similarity threshold,and $k$ is the number of similar set pairs in the result.

定理 1. 对于带重叠约束的集合相似度连接问题，存在一个时间复杂度为 $O\left( {{n}^{2 - \frac{1}{c}}{k}^{\frac{1}{2c}}}\right)$ 的算法，其中 $n$ 是所有集合的总大小，常数 $c$ 是相似度阈值， $k$ 是结果中相似集合对的数量。

Beating the Quadratic Barrier: The value of $k$ ranges from 0 to $\left( \begin{array}{l} n \\  2 \end{array}\right)$ . As explained in Section 1,we achieve sub-quadratic time whenever this is possible. That is,for $k = o\left( {n}^{2}\right)$ ,it always holds that $O\left( {{n}^{2 - \frac{1}{c}}{k}^{\frac{1}{2c}}}\right)  = o\left( {n}^{2}\right)$ ,whereas for $k = \Omega \left( {n}^{2}\right)$ ,any algorithm must spend $\Omega \left( {n}^{2}\right)$ time just to output all the similar pairs.

突破二次复杂度障碍： $k$ 的取值范围是从 0 到 $\left( \begin{array}{l} n \\  2 \end{array}\right)$ 。如第 1 节所述，只要有可能，我们就能实现亚二次时间复杂度。也就是说，对于 $k = o\left( {n}^{2}\right)$ ，始终有 $O\left( {{n}^{2 - \frac{1}{c}}{k}^{\frac{1}{2c}}}\right)  = o\left( {n}^{2}\right)$ ，而对于 $k = \Omega \left( {n}^{2}\right)$ ，任何算法仅输出所有相似对就必须花费 $\Omega \left( {n}^{2}\right)$ 的时间。

Remark: Note we make no assumptions about the distribution of the set sizes. In the extreme case where all the sets have exactly the same size $\ell$ ,either all of them are classified as small sets,or all of them are classified as large sets, depending on the comparison between $\ell$ and ${\left( n/\sqrt{k}\right) }^{1/c}$ (i.e.,the size boundary). In both cases, the time complexity is as claimed-our proof holds in general.

备注：注意，我们对集合大小的分布不做任何假设。在极端情况下，所有集合的大小恰好都为 $\ell$ ，根据 $\ell$ 与 ${\left( n/\sqrt{k}\right) }^{1/c}$ 的比较结果（即大小边界），它们要么都被归类为小集合，要么都被归类为大集合。在这两种情况下，时间复杂度都如所声称的那样——我们的证明具有一般性。

Having proved the theoretical guarantee of our algorithm, in the subsequent sections, we will strive to improve its practical performance dramatically with careful optimization heuristics. Focus will be placed on processing ${\mathcal{R}}_{s} \times  {\mathcal{R}}_{s}$ ,as any existing approach can be used to process ${\mathcal{R}}_{l} \times  \mathcal{R}$ . As a serious challenge,our current algorithm needs to enumerate all the $c$ -subsets of a small set,the number of which can be huge, thus causing significant overhead. We will remedy this issue with novel ideas, as presented below.

在证明了我们算法的理论保证之后，在后续章节中，我们将努力通过精心设计的优化启发式方法显著提高其实际性能。我们将重点处理${\mathcal{R}}_{s} \times  {\mathcal{R}}_{s}$，因为任何现有方法都可用于处理${\mathcal{R}}_{l} \times  \mathcal{R}$。一个严峻的挑战是，我们当前的算法需要枚举一个小集合的所有$c$ - 子集，其数量可能非常庞大，从而导致显著的开销。我们将用下面介绍的新方法来解决这个问题。

## 4 HEAP-BASED METHODS ON SMALL SETS

## 4 小集合上基于堆的方法

In this section,we focus on building the inverted index ${\mathcal{L}}_{\text{slim }}$ for $c$ - subsets in ${\mathcal{R}}_{s}$ that can generate all the results in ${\mathcal{R}}_{s} \times  {\mathcal{R}}_{s}$ ,which we shall call a slimmed inverted index, instead of the full inverted index $\mathcal{L}$ . It is possible to skip some unnecessary $c$ -subsets in ${\mathcal{R}}_{s}$ when we construct a slimmed inverted index,which includes unique $c$ - subsets and redundant $c$ -subsets. We propose heap-based methods to skip unique and redundant $c$ -subsets in Section 4.1 and Section 4.2 respectively. As it is expensive to maintain the heap, especially when the heap is wide, we propose a blocking-based method to shrink the heap in Section 4.3.

在本节中，我们专注于为${\mathcal{R}}_{s}$中的$c$ - 子集构建倒排索引${\mathcal{L}}_{\text{slim }}$，该索引可以在${\mathcal{R}}_{s} \times  {\mathcal{R}}_{s}$中生成所有结果，我们将其称为精简倒排索引，而不是完整倒排索引$\mathcal{L}$。在构建精简倒排索引时，可以跳过${\mathcal{R}}_{s}$中一些不必要的$c$ - 子集，该索引包括唯一的$c$ - 子集和冗余的$c$ - 子集。我们分别在4.1节和4.2节中提出基于堆的方法来跳过唯一和冗余的$c$ - 子集。由于维护堆的成本很高，尤其是当堆很宽时，我们在4.3节中提出一种基于分块的方法来缩小堆的规模。

### 4.1 Skipping Unique c-subsets

### 4.1 跳过唯一的c - 子集

For each small set, the size-aware algorithm needs to enumerate all its $c$ -subsets to build the full inverted index $\mathcal{L}$ and generate the results based on it. If a $c$ -subset is unique,i.e.,it appears only once in all the small sets, we can avoid generating it and get a slimmed inverted index that can generate all the results as the unique $c$ -subset cannot produce any result.

对于每个小集合，考虑大小的算法需要枚举其所有的$c$ - 子集来构建完整倒排索引$\mathcal{L}$，并基于此生成结果。如果一个$c$ - 子集是唯一的，即它在所有小集合中只出现一次，我们可以避免生成它，从而得到一个精简倒排索引，该索引可以生成所有结果，因为唯一的$c$ - 子集不会产生任何结果。

Definition 2 (Unique $c$ -subset). A $c$ -subset ${r}_{c}$ is called a unique c-subset if $\left| {\mathcal{L}\left\lbrack  {\mathbf{r}}_{c}\right\rbrack  }\right|  = 1$ .

定义2（唯一的$c$ - 子集）。如果$\left| {\mathcal{L}\left\lbrack  {\mathbf{r}}_{c}\right\rbrack  }\right|  = 1$，则$c$ - 子集${r}_{c}$被称为唯一的c - 子集。

As there are a large number of unique $c$ -subsets,it is important to avoid generating them. For example,in Figure 2, ${R}_{4}$ has ${6c}$ - subsets and all of them are unique $c$ -subsets. Thus we do not need to generate them. Given a set $R$ ,it has $\left( \begin{matrix} \left| R\right| \\  c \end{matrix}\right) c$ -subsets and it is prohibitively expensive to generate all of them. Fortunately, most of them are unique $c$ -subsets and next we discuss how to skip them. Skip Unique c-subsets. We first give the basic idea of skipping unique $c$ -subsets. We fix a global order for the $c$ -subsets in all the small sets and visit the $c$ -subsets in ascending order. As shown in Figure 3,consider a $c$ -subset ${r}_{c}$ in a small set $R$ . Let ${r}_{c}^{\prime }$ be the smallest $c$ -subset that is larger than ${r}_{c}$ in ${\mathcal{R}}_{s} \smallsetminus  \{ R\}$ (i.e.,not in $R$ ). Then all the $c$ -subsets between ${r}_{c}$ and ${r}_{c}^{\prime }$ (the gray ones in the figure) must only appear in $R$ and must be unique $c$ -subsets (this is based on the definition of $\left. {r}_{c}^{\prime }\right)$ . Thus we can skip the $c$ -subsets in $R$ which are larger than ${r}_{c}$ and smaller than ${r}_{c}^{\prime }$ . Next we discuss how to utilize this idea to skip unique $c$ -subsets.

由于存在大量唯一的$c$ - 子集，避免生成它们非常重要。例如，在图2中，${R}_{4}$有${6c}$ - 子集，并且它们都是唯一的$c$ - 子集。因此，我们不需要生成它们。给定一个集合$R$，它有$\left( \begin{matrix} \left| R\right| \\  c \end{matrix}\right) c$ - 子集，生成所有这些子集的成本高得令人望而却步。幸运的是，其中大多数是唯一的$c$ - 子集，接下来我们讨论如何跳过它们。跳过唯一的c - 子集。我们首先给出跳过唯一的$c$ - 子集的基本思路。我们为所有小集合中的$c$ - 子集确定一个全局顺序，并按升序访问这些$c$ - 子集。如图3所示，考虑小集合$R$中的一个$c$ - 子集${r}_{c}$。设${r}_{c}^{\prime }$是${\mathcal{R}}_{s} \smallsetminus  \{ R\}$中（即不在$R$中）比${r}_{c}$大的最小$c$ - 子集。那么${r}_{c}$和${r}_{c}^{\prime }$之间的所有$c$ - 子集（图中灰色的那些）一定只出现在$R$中，并且一定是唯一的$c$ - 子集（这是基于$\left. {r}_{c}^{\prime }\right)$的定义）。因此，我们可以跳过$R$中比${r}_{c}$大且比${r}_{c}^{\prime }$小的$c$ - 子集。接下来我们讨论如何利用这个思路来跳过唯一的$c$ - 子集。

Global Ordering. We fix a global order for all the elements in the small sets and sort the elements in each small set by this global order. Then we can order the $c$ -subset based on the order of elements,i.e., first by the smallest element, then by the second smallest element and finally by the largest element. For example, consider the four small sets in Table 1 and suppose that we order the elements by their subscripts,i.e.,the order of ${e}_{1},{e}_{2},\ldots ,{e}_{7}$ . Then the order of the 2-subsets is shown in Figure 2 from left to right.

全局排序。我们为小集合中的所有元素确定一个全局顺序，并按照这个全局顺序对每个小集合中的元素进行排序。然后，我们可以根据元素的顺序对 $c$ -子集进行排序，即首先按最小元素排序，然后按次小元素排序，最后按最大元素排序。例如，考虑表 1 中的四个小集合，并假设我们按元素的下标对元素进行排序，即 ${e}_{1},{e}_{2},\ldots ,{e}_{7}$ 的顺序。那么 2 -子集的顺序如图 2 所示，从左到右排列。

Heap-based Method. We first give a naive heap-based method to construct the entire inverted index $\mathcal{L}$ . For each small set,we visit its $c$ -subsets in ascending order and denote the smallest unvisited $c$ -subset as its min-subset. A min-heap $\mathcal{H}$ is used to manage all the min-subsets of the small sets. We pop $\mathcal{H}$ to get the globally smallest min-subset,which is denoted as ${r}_{c}^{\min }$ . Suppose that ${r}_{c}^{\min }$ comes from the set $R$ . We append $R$ to the inverted list $\mathcal{L}\left\lbrack  {r}_{c}^{\min }\right\rbrack$ ,mark ${r}_{c}^{\min }$ as visited and reinsert the next min-subset of $R$ to the heap. Iteratively, we can build all the inverted lists and get the entire inverted index $\mathcal{L}$ .

基于堆的方法。我们首先给出一种简单的基于堆的方法来构建整个倒排索引 $\mathcal{L}$。对于每个小集合，我们按升序访问其 $c$ -子集，并将未访问的最小 $c$ -子集记为其最小子集。使用一个最小堆 $\mathcal{H}$ 来管理所有小集合的最小子集。我们从 $\mathcal{H}$ 中弹出元素以获取全局最小的最小子集，记为 ${r}_{c}^{\min }$。假设 ${r}_{c}^{\min }$ 来自集合 $R$。我们将 $R$ 追加到倒排列表 $\mathcal{L}\left\lbrack  {r}_{c}^{\min }\right\rbrack$ 中，将 ${r}_{c}^{\min }$ 标记为已访问，并将 $R$ 的下一个最小子集重新插入堆中。通过迭代，我们可以构建所有的倒排列表并得到整个倒排索引 $\mathcal{L}$。

---

<!-- Footnote -->

${}^{1}$ Note the doubling trick is only for the complexity analysis. In practice,we use the approach later proposed in Section 5 to determine the size boundary.

${}^{1}$ 注意，倍增技巧仅用于复杂度分析。在实践中，我们使用后面第 5 节提出的方法来确定大小边界。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: visited c-subsets unique c-subsets unvisited c-subsets ${\mathcal{R}}_{s} \smallsetminus  \{ R\}$ the global order of c-subsets $R$ ${r}_{c}$ the smallest c-subset that is larger than ${r}_{\mathrm{c}}$ in ${\mathcal{R}}_{s} \smallsetminus  \{ R\}$ -->

<img src="https://cdn.noedgeai.com/0195ccc7-1611-78aa-a97d-afb7fc00df51_4.jpg?x=154&y=160&w=754&h=388&r=0"/>

Figure 3: Skip the unique $c$ -subsets.

图 3：跳过唯一的 $c$ -子集。

<!-- Media -->

Next we construct a slimmed inverted index by excluding the inverted lists of the unique $c$ -subsets from $\mathcal{L}$ . For this purpose,every time we pop the heap and get the smallest min-subset ${r}_{c}^{\min }$ from a small set $R$ ,we can again compare ${r}_{c}^{\min }$ with the min-subset that currently tops the heap,which is denoted as ${r}_{c}^{top}$ . If ${r}_{c}^{top} \neq  {r}_{c}^{min}$ , instead of reinserting the next min-subset of $R$ to the heap,we can jump directly to the smallest $c$ -subset in $R$ that is no smaller than ${r}_{c}^{top}$ and reinsert it to the heap as the skipped $c$ -subsets must only appear in $R$ and must be unique $c$ -subsets (recall the basic idea above,in which case ${r}_{c}^{\min }$ corresponds to ${r}_{c}$ and ${r}_{c}^{top}$ corresponds to $\left. {r}_{c}^{\prime }\right)$ . We can achieve this by a binary search as the elements and $c$ -subsets in $R$ are ordered. The details of the binary search are described in Appendix A.

接下来，我们通过从 $\mathcal{L}$ 中排除唯一的 $c$ -子集的倒排列表来构建一个精简的倒排索引。为此，每次我们从堆中弹出元素并从小集合 $R$ 中获取最小的最小子集 ${r}_{c}^{\min }$ 时，我们可以再次将 ${r}_{c}^{\min }$ 与当前堆顶的最小子集（记为 ${r}_{c}^{top}$）进行比较。如果 ${r}_{c}^{top} \neq  {r}_{c}^{min}$，我们可以直接跳到 $R$ 中不小于 ${r}_{c}^{top}$ 的最小 $c$ -子集，并将其作为跳过的 $c$ -子集重新插入堆中，因为跳过的 $c$ -子集必定仅出现在 $R$ 中，并且必定是唯一的 $c$ -子集（回顾上述基本思想，在这种情况下，${r}_{c}^{\min }$ 对应于 ${r}_{c}$，${r}_{c}^{top}$ 对应于 $\left. {r}_{c}^{\prime }\right)$）。由于 $R$ 中的元素和 $c$ -子集是有序的，我们可以通过二分查找来实现这一点。二分查找的详细信息在附录 A 中描述。

The pseudo code of the HeapSkip method is shown in Algorithm 2. Instead of enumerating every $c$ -subset in each small set, it first fixes a global order for all the elements and builds a min-heap $\mathcal{H}$ by inserting all the min-subsets of the small sets to $\mathcal{H}$ (Lines 1 to 2). It keeps popping $\mathcal{H}$ until it is empty (Lines 3 to 9). Suppose that the smallest popped out min-subset ${r}_{c}^{\min }$ comes from $R$ ,it appends $R$ to the inverted list ${\mathcal{L}}_{\text{slim }}\left\lbrack  {r}_{c}^{\min }\right\rbrack$ and compares ${r}_{c}^{\min }$ with ${r}_{c}^{top}$ which is the current top element of $\mathcal{H}$ . If ${r}_{c}^{min}$ and ${r}_{c}^{top}$ are different,it binary searches the first $c$ -subset in $R$ that is no smaller than ${r}_{c}^{top}$ and reinserts it to $\mathcal{H}$ (Lines 6 to 7); otherwise it reinserts the next min-subset in $R$ into $\mathcal{H}$ (Line 9). Finally,it returns a slimmed inverted index ${\mathcal{L}}_{\text{slim }}$ (Line 10).

HeapSkip方法的伪代码如算法2所示。该方法并非枚举每个小集合中的所有$c$ -子集，而是首先为所有元素确定一个全局顺序，并通过将小集合的所有最小子集插入到最小堆$\mathcal{H}$ 中来构建该最小堆（第1至2行）。它会持续从$\mathcal{H}$ 中弹出元素，直到堆为空（第3至9行）。假设弹出的最小最小子集${r}_{c}^{\min }$ 来自$R$ ，则将$R$ 追加到倒排表${\mathcal{L}}_{\text{slim }}\left\lbrack  {r}_{c}^{\min }\right\rbrack$ 中，并将${r}_{c}^{\min }$ 与$\mathcal{H}$ 的当前堆顶元素${r}_{c}^{top}$ 进行比较。如果${r}_{c}^{min}$ 和${r}_{c}^{top}$ 不同，则在$R$ 中二分查找第一个不小于${r}_{c}^{top}$ 的$c$ -子集，并将其重新插入到$\mathcal{H}$ 中（第6至7行）；否则，将$R$ 中的下一个最小子集重新插入到$\mathcal{H}$ 中（第9行）。最后，返回一个精简的倒排索引${\mathcal{L}}_{\text{slim }}$ （第10行）。

Example 2. Consider the dataset $\mathcal{R}$ in Table 1 and suppose the threshold is $c = 2$ . There are 4 small sets, ${R}_{1},{R}_{2},{R}_{3}$ ,and ${R}_{4}$ . As illustrate in Figure 2, HeapSkip first orders the elements in them by their subscripts. Then it inserts the min-subsets $\left\{  {{e}_{1},{e}_{2}}\right\}  ,\left\{  {{e}_{1},{e}_{3}}\right\}$ , $\left\{  {{e}_{1},{e}_{3}}\right\}$ ,and $\left\{  {{e}_{2},{e}_{4}}\right\}$ of the four small sets into a min-heap $\mathcal{H}$ . Next it pops $\mathcal{H}$ and has ${r}_{c}^{\text{min }} = \left\{  {{e}_{1},{e}_{2}}\right\}$ from ${R}_{1}$ and ${r}_{c}^{\text{top }} = \left\{  {{e}_{1},{e}_{3}}\right\}$ . It appends ${R}_{1}$ to the inverted list ${\mathcal{L}}_{\text{slim }}\left\lbrack  \left\{  {{e}_{1},{e}_{2}}\right\}  \right\rbrack$ . As ${r}_{c}^{\min } \neq  {r}_{c}^{top}$ , it binary searches the first $c$ -subset in ${R}_{1}$ that is no smaller than ${r}_{c}^{top}$ . It gets $\left\{  {{e}_{1},{e}_{3}}\right\}$ and reinserts this $c$ -subset to $\mathcal{H}$ . Then it pops $\mathcal{H}$ and has ${r}_{c}^{\text{min }} = \left\{  {{e}_{1},{e}_{3}}\right\}$ from ${R}_{1}$ and ${r}_{c}^{\text{top }} = \left\{  {{e}_{1},{e}_{3}}\right\}$ . It appends ${R}_{1}$ to ${\mathcal{L}}_{\text{slim }}\left\lbrack  \left\{  {{e}_{1},{e}_{3}}\right\}  \right\rbrack$ . As ${r}_{c}^{\min } = {r}_{c}^{\text{top }}$ ,it reinserts the next min-subset $\left\{  {{e}_{2},{e}_{3}}\right\}$ of ${R}_{1}$ to $\mathcal{H}$ . Iteratively,it can build a slimmed inverted index ${\mathcal{L}}_{\text{slim }}$ . Note it can skip the unique $c$ -subsets $\left\{  {{e}_{2},{e}_{5}}\right\}$ and $\left\{  {{e}_{2},{e}_{6}}\right\}$ of ${R}_{4}$ by binary searching the smallest $c$ -subset in ${R}_{4}$ that is no smaller than ${r}_{c}^{top} = \left\{  {{e}_{3},{e}_{4}}\right\}$ when ${r}_{c}^{min} = \left\{  {{e}_{2},{e}_{4}}\right\}$ comes from ${R}_{4}$ . Similarly it can also skip the $c$ -subset $\left\{  {{e}_{4},{e}_{6}}\right\}$ of ${R}_{4}$ when ${r}_{c}^{\min } = \left\{  {{e}_{4},{e}_{5}}\right\}$ and ${r}_{c}^{top} = \left\{  {{e}_{4},{e}_{7}}\right\}$

示例2. 考虑表1中的数据集$\mathcal{R}$，并假设阈值为$c = 2$。有4个小集合，${R}_{1},{R}_{2},{R}_{3}$和${R}_{4}$。如图2所示，HeapSkip算法首先根据元素的下标对这些小集合中的元素进行排序。然后，它将这四个小集合的最小子集$\left\{  {{e}_{1},{e}_{2}}\right\}  ,\left\{  {{e}_{1},{e}_{3}}\right\}$、$\left\{  {{e}_{1},{e}_{3}}\right\}$和$\left\{  {{e}_{2},{e}_{4}}\right\}$插入到一个最小堆$\mathcal{H}$中。接下来，它从$\mathcal{H}$中弹出元素，从${R}_{1}$和${r}_{c}^{\text{top }} = \left\{  {{e}_{1},{e}_{3}}\right\}$中得到${r}_{c}^{\text{min }} = \left\{  {{e}_{1},{e}_{2}}\right\}$。它将${R}_{1}$添加到倒排表${\mathcal{L}}_{\text{slim }}\left\lbrack  \left\{  {{e}_{1},{e}_{2}}\right\}  \right\rbrack$中。由于${r}_{c}^{\min } \neq  {r}_{c}^{top}$，它在${R}_{1}$中二分查找第一个不小于${r}_{c}^{top}$的$c$ - 子集。它得到$\left\{  {{e}_{1},{e}_{3}}\right\}$，并将这个$c$ - 子集重新插入到$\mathcal{H}$中。然后，它从$\mathcal{H}$中弹出元素，从${R}_{1}$和${r}_{c}^{\text{top }} = \left\{  {{e}_{1},{e}_{3}}\right\}$中得到${r}_{c}^{\text{min }} = \left\{  {{e}_{1},{e}_{3}}\right\}$。它将${R}_{1}$添加到${\mathcal{L}}_{\text{slim }}\left\lbrack  \left\{  {{e}_{1},{e}_{3}}\right\}  \right\rbrack$中。由于${r}_{c}^{\min } = {r}_{c}^{\text{top }}$，它将${R}_{1}$的下一个最小子集$\left\{  {{e}_{2},{e}_{3}}\right\}$重新插入到$\mathcal{H}$中。通过迭代，它可以构建一个精简的倒排索引${\mathcal{L}}_{\text{slim }}$。注意，当${r}_{c}^{min} = \left\{  {{e}_{2},{e}_{4}}\right\}$来自${R}_{4}$时，通过在${R}_{4}$中二分查找不小于${r}_{c}^{top} = \left\{  {{e}_{3},{e}_{4}}\right\}$的最小$c$ - 子集，它可以跳过${R}_{4}$的唯一$c$ - 子集$\left\{  {{e}_{2},{e}_{5}}\right\}$和$\left\{  {{e}_{2},{e}_{6}}\right\}$。类似地，当${r}_{c}^{\min } = \left\{  {{e}_{4},{e}_{5}}\right\}$和${r}_{c}^{top} = \left\{  {{e}_{4},{e}_{7}}\right\}$时，它也可以跳过${R}_{4}$的$c$ - 子集$\left\{  {{e}_{4},{e}_{6}}\right\}$

<!-- Media -->

Algorithm 2: HEAPSKIP

算法2：HeapSkip算法

---

Input: ${\mathcal{R}}_{s}$ : all the small sets; $c$ : threshold;

输入：${\mathcal{R}}_{s}$：所有小集合；$c$：阈值；

Output: ${\mathcal{L}}_{\text{slim }}$ : a slimmed inverted index for ${\mathcal{R}}_{s}$ ;

输出：${\mathcal{L}}_{\text{slim }}$ ：${\mathcal{R}}_{s}$ 的精简倒排索引；

Fix a global order for all the elements in ${\mathcal{R}}_{s}$ ;

为 ${\mathcal{R}}_{s}$ 中的所有元素确定一个全局顺序；

Insert all the min-subsets of small sets to a heap $\mathcal{H}$ ;

将小集合的所有最小子集插入到一个堆 $\mathcal{H}$ 中；

while $\mathcal{H}$ is not empty do

当 $\mathcal{H}$ 不为空时执行

	pop $\mathcal{H}$ to get ${r}_{c}^{\min }$ and suppose it is from $R$ ;

	从 $\mathcal{H}$ 中弹出元素以获取 ${r}_{c}^{\min }$ ，并假设它来自 $R$ ；

	append $R$ to ${\mathcal{L}}_{\text{slim }}\left\lbrack  {\mathrm{r}}_{c}^{\min }\right\rbrack$ ;

	将 $R$ 追加到 ${\mathcal{L}}_{\text{slim }}\left\lbrack  {\mathrm{r}}_{c}^{\min }\right\rbrack$ 中；

	if ${r}_{c}^{top} \neq  {r}_{c}^{min}$ then

	如果 ${r}_{c}^{top} \neq  {r}_{c}^{min}$ 则

		binary search for the first $c$ -subset in $R$ that is no

		在 $R$ 中二分查找第一个不小于 $c$ 的 $c$ -子集

		smaller than ${r}_{c}^{top}$ and reinsert it into $\mathcal{H}$ ;

		并将其重新插入到 $\mathcal{H}$ 中；

	else

	否则

		reinsert the next min-subset in $R$ into $\mathcal{H}$ ;

		将 $R$ 中的下一个最小子集重新插入到 $\mathcal{H}$ 中；

return ${\mathcal{L}}_{\text{slim }}$

返回 ${\mathcal{L}}_{\text{slim }}$

---

<!-- Media -->

### 4.2 Skipping Redundant c-subsets

### 4.2 跳过冗余的 c-子集

For small sets, the size-aware algorithm may produce duplicate results as some set pairs may share multiple common $c$ -subsets. If a $c$ -subset only generates duplicate results,we can skip enumerating it and still get a slimmed inverted index. Obviously,given two $c$ - subsets ${r}_{c}$ and ${r}_{c}^{\prime }$ ,if $\mathcal{L}\left\lbrack  {r}_{c}\right\rbrack   \subseteq  \mathcal{L}\left\lbrack  {r}_{c}^{\prime }\right\rbrack$ ,then ${r}_{c}$ is redundant,because the result generated by ${r}_{c}$ (i.e., $\mathcal{L}\left\lbrack  {r}_{c}\right\rbrack   \times  \mathcal{L}\left\lbrack  {r}_{c}\right\rbrack$ ) is a subset of that generated by ${r}_{c}^{\prime }$ (i.e., $\mathcal{L}\left\lbrack  {r}_{c}^{\prime }\right\rbrack   \times  \mathcal{L}\left\lbrack  {r}_{c}^{\prime }\right\rbrack$ ).

对于小集合，考虑大小的算法可能会产生重复结果，因为某些集合对可能共享多个公共 $c$ -子集。如果一个 $c$ -子集仅生成重复结果，我们可以跳过对其进行枚举，仍然可以得到一个精简的倒排索引。显然，给定两个 $c$ -子集 ${r}_{c}$ 和 ${r}_{c}^{\prime }$ ，如果 $\mathcal{L}\left\lbrack  {r}_{c}\right\rbrack   \subseteq  \mathcal{L}\left\lbrack  {r}_{c}^{\prime }\right\rbrack$ ，那么 ${r}_{c}$ 是冗余的，因为 ${r}_{c}$ 生成的结果（即 $\mathcal{L}\left\lbrack  {r}_{c}\right\rbrack   \times  \mathcal{L}\left\lbrack  {r}_{c}\right\rbrack$ ）是 ${r}_{c}^{\prime }$ 生成的结果（即 $\mathcal{L}\left\lbrack  {r}_{c}^{\prime }\right\rbrack   \times  \mathcal{L}\left\lbrack  {r}_{c}^{\prime }\right\rbrack$ ）的子集。

DEFINITION 3 (REDUNDANT $c$ -SUBSET). A $c$ -subset ${r}_{c}$ is a redundant c-subset of another c-subset ${r}_{c}^{\prime }$ if $\mathcal{L}\left\lbrack  {r}_{c}\right\rbrack   \subseteq  \mathcal{L}\left\lbrack  {r}_{c}^{\prime }\right\rbrack$ .

定义 3（冗余 $c$ -子集）。如果 $\mathcal{L}\left\lbrack  {r}_{c}\right\rbrack   \subseteq  \mathcal{L}\left\lbrack  {r}_{c}^{\prime }\right\rbrack$ ，则 $c$ -子集 ${r}_{c}$ 是另一个 c-子集 ${r}_{c}^{\prime }$ 的冗余 c-子集。

Note that the duplicate results are generated whenever $\mid  \mathcal{L}\left\lbrack  {r}_{c}\right\rbrack   \cap$ $\mathcal{L}\left\lbrack  {\mathrm{r}}_{c}^{\prime }\right\rbrack   \mid   \geq  2$ . However,it is expensive to eliminate all the duplicate results. In fact,it remains expensive to detect all redundant $c$ -subsets, as it requires to enumerate every two $c$ -subsets and checks whether the inverted list of one $c$ -subset is a subset of the other. To address this issue, we propose an efficient algorithm that can detect all adjacent redundant $c$ -subsets.

请注意，只要 $\mid  \mathcal{L}\left\lbrack  {r}_{c}\right\rbrack   \cap$ $\mathcal{L}\left\lbrack  {\mathrm{r}}_{c}^{\prime }\right\rbrack   \mid   \geq  2$ 就会生成重复结果。然而，消除所有重复结果的代价很高。实际上，检测所有冗余的 $c$ -子集的代价仍然很高，因为这需要枚举每两个 $c$ -子集，并检查一个 $c$ -子集的倒排列表是否是另一个的子集。为了解决这个问题，我们提出了一种高效的算法，该算法可以检测所有相邻的冗余 $c$ -子集。

Definition 4 (Adjacent Redundant c-subset). The c-subset ${r}_{c}$ is an adjacent redundant $c$ -subset of another $c$ -subset ${r}_{c}^{\prime }$ if ${r}_{c}^{\prime } < {r}_{c}$ , where $<$ denotes the order of $c$ -subsets,and the $c$ -subsets between ${r}_{c}^{\prime }$ and ${r}_{c}$ ,including ${r}_{c}$ ,are all redundant $c$ -subsets of ${r}_{c}^{\prime }$ .

定义4（相邻冗余c-子集）。若${r}_{c}^{\prime } < {r}_{c}$ ，则c-子集${r}_{c}$ 是另一个$c$ -子集${r}_{c}^{\prime }$ 的相邻冗余$c$ -子集，其中$<$ 表示$c$ -子集的顺序，且${r}_{c}^{\prime }$ 和${r}_{c}$ 之间（包括${r}_{c}$ ）的所有$c$ -子集都是${r}_{c}^{\prime }$ 的冗余$c$ -子集。

For example,in Figure 2,we have $\mathcal{L}\left\lbrack  \left\{  {{e}_{1}{e}_{3}}\right\}  \right\rbrack   = \left\{  {{R}_{1},{R}_{2},{R}_{3}}\right\}$ , $\mathcal{L}\left\lbrack  \left\{  {{e}_{1}{e}_{4}}\right\}  \right\rbrack   = \left\{  {R}_{2}\right\}  ,\mathcal{L}\left\lbrack  \left\{  {{e}_{1}{e}_{5}}\right\}  \right\rbrack   = \left\{  {R}_{3}\right\}  ,\mathcal{L}\left\lbrack  \left\{  {{e}_{1}{e}_{7}}\right\}  \right\rbrack   = \left\{  {{R}_{2},{R}_{3}}\right\}$ ,and $\mathcal{L}\left\lbrack  \left\{  {{e}_{2}{e}_{3}}\right\}  \right\rbrack   = \left\{  {R}_{1}\right\}$ . Thus,based on the definition, $\left\{  {{e}_{1}{e}_{4}}\right\}  ,\left\{  {{e}_{1}{e}_{5}}\right\}$ , $\left\{  {{e}_{1}{e}_{7}}\right\}$ ,and $\left\{  {{e}_{2}{e}_{3}}\right\}$ are all adjacent redundant $c$ -subsets of $\left\{  {{e}_{1}{e}_{3}}\right\}$ . If we skip them, we can build a smaller slimmed inverted index.

例如，在图2中，我们有$\mathcal{L}\left\lbrack  \left\{  {{e}_{1}{e}_{3}}\right\}  \right\rbrack   = \left\{  {{R}_{1},{R}_{2},{R}_{3}}\right\}$ 、$\mathcal{L}\left\lbrack  \left\{  {{e}_{1}{e}_{4}}\right\}  \right\rbrack   = \left\{  {R}_{2}\right\}  ,\mathcal{L}\left\lbrack  \left\{  {{e}_{1}{e}_{5}}\right\}  \right\rbrack   = \left\{  {R}_{3}\right\}  ,\mathcal{L}\left\lbrack  \left\{  {{e}_{1}{e}_{7}}\right\}  \right\rbrack   = \left\{  {{R}_{2},{R}_{3}}\right\}$ 和$\mathcal{L}\left\lbrack  \left\{  {{e}_{2}{e}_{3}}\right\}  \right\rbrack   = \left\{  {R}_{1}\right\}$ 。因此，根据定义，$\left\{  {{e}_{1}{e}_{4}}\right\}  ,\left\{  {{e}_{1}{e}_{5}}\right\}$ 、$\left\{  {{e}_{1}{e}_{7}}\right\}$ 和$\left\{  {{e}_{2}{e}_{3}}\right\}$ 都是$\left\{  {{e}_{1}{e}_{3}}\right\}$ 的相邻冗余$c$ -子集。如果我们跳过它们，就可以构建一个更小的精简倒排索引。

Skip Adjacent Redundant c-subsets. We first give the basic idea of skipping adjacent redundant $c$ -subsets. As shown in Figure 4,we still visit the $c$ -subsets in ascending order. Let ${r}_{c}^{\prime \prime }$ be the smallest $c$ -subset that is larger than ${r}_{c}$ in ${\mathcal{R}}_{s} \smallsetminus  \mathcal{L}\left\lbrack  {r}_{c}\right\rbrack$ (i.e.,sets not containing $\left. {r}_{c}\right)$ . We find that all the $c$ -subsets in the small sets in $\mathcal{L}\left\lbrack  {r}_{c}\right\rbrack$ that are larger than ${r}_{c}$ and smaller than ${r}_{c}^{\prime \prime }$ (the gray ones in the figure, e.g., ${r}_{c}^{\prime }$ ) are adjacent redundant $c$ -subsets of ${r}_{c}$ as their inverted lists are all sub-lists of $\mathcal{L}\left\lbrack  {r}_{c}\right\rbrack$ . Next we discuss how to utilize this idea to skip adjacent redundant $c$ -subsets.

跳过相邻冗余c-子集。我们首先给出跳过相邻冗余$c$ -子集的基本思路。如图4所示，我们仍然按升序访问$c$ -子集。设${r}_{c}^{\prime \prime }$ 是${\mathcal{R}}_{s} \smallsetminus  \mathcal{L}\left\lbrack  {r}_{c}\right\rbrack$ 中比${r}_{c}$ 大的最小$c$ -子集（即不包含$\left. {r}_{c}\right)$ 的集合）。我们发现，$\mathcal{L}\left\lbrack  {r}_{c}\right\rbrack$ 中比${r}_{c}$ 大且比${r}_{c}^{\prime \prime }$ 小的小集合中的所有$c$ -子集（图中灰色部分，例如${r}_{c}^{\prime }$ ）都是${r}_{c}$ 的相邻冗余$c$ -子集，因为它们的倒排列表都是$\mathcal{L}\left\lbrack  {r}_{c}\right\rbrack$ 的子列表。接下来我们讨论如何利用这一思路跳过相邻冗余$c$ -子集。

<!-- Media -->

<!-- figureText: visited c-subsets adjacent redundant c-subsets ι ${\mathcal{R}}_{s} \smallsetminus  \mathcal{L}\left\lbrack  {r}_{c}\right\rbrack$ the global order of c-subsets ... ... $\mathcal{L}\left\lbrack  {r}_{c}\right\rbrack   <$ the smallest c-subset that is larger than ${r}_{c}$ in ${\mathcal{R}}_{s} \smallsetminus  \mathcal{L}\left\lbrack  {r}_{c}\right\rbrack$ . -->

<img src="https://cdn.noedgeai.com/0195ccc7-1611-78aa-a97d-afb7fc00df51_5.jpg?x=155&y=163&w=743&h=408&r=0"/>

Figure 4: Skip the adjacent redundant $c$ -subsets.

图4：跳过相邻冗余$c$ -子集。

<!-- Media -->

Heap-based Method. We still fix the order of the $c$ -subsets by the order of elements,access the $c$ -subsets of each set in order, utilize a min-heap $\mathcal{H}$ to manage the min-subsets,and iteratively pop the min-heap to build the inverted lists. Every time we pop $\mathcal{H}$ and get ${r}_{c}^{\min }$ from a set $R$ ,we first append $R$ to ${\mathcal{L}}_{\text{slim }}\left\lbrack  {r}_{c}^{\min }\right\rbrack$ . Then we compare ${r}_{c}^{\min }$ with ${r}_{c}^{top}$ . However,if ${r}_{c}^{\min } = {r}_{c}^{top}$ ,we do not reinsert the next min-subset of $R$ to $\mathcal{H}$ . Only if ${r}_{c}^{\min } \neq  {r}_{c}^{\text{top }}$ we reinsert the min-subsets of all the sets in ${\mathcal{L}}_{\text{slim }}\left\lbrack  {r}_{c}^{\min }\right\rbrack$ to $\mathcal{H}$ by binary searching the first $c$ -subsets that are no smaller than ${r}_{c}^{top}$ . In this way,we can skip those $c$ -subsets larger than ${r}_{c}^{\min }$ and smaller than ${r}_{c}^{top}$ which must be adjacent redundant $c$ -subsets of ${r}_{c}^{min}$ as the inverted lists of these $c$ -subsets are all sub-lists of ${\mathcal{L}}_{\text{slim }}\left\lbrack  {r}_{c}^{\text{min }}\right\rbrack$ (recall the basic idea above,in which case ${r}_{c}^{\min }$ corresponds to ${r}_{c}$ and ${r}_{c}^{top}$ corresponds to ${r}_{c}^{\prime \prime }$ ).

基于堆的方法。我们仍然按照元素顺序固定$c$ -子集的顺序，依次访问每个集合的$c$ -子集，利用一个最小堆$\mathcal{H}$来管理最小子集，并迭代地从最小堆中弹出元素以构建倒排表。每次我们从最小堆$\mathcal{H}$中弹出元素并从集合$R$中得到${r}_{c}^{\min }$时，我们首先将$R$追加到${\mathcal{L}}_{\text{slim }}\left\lbrack  {r}_{c}^{\min }\right\rbrack$中。然后我们将${r}_{c}^{\min }$与${r}_{c}^{top}$进行比较。然而，如果${r}_{c}^{\min } = {r}_{c}^{top}$，我们不会将$R$的下一个最小子集重新插入到$\mathcal{H}$中。只有当${r}_{c}^{\min } \neq  {r}_{c}^{\text{top }}$时，我们才通过二分查找不小于${r}_{c}^{top}$的第一个$c$ -子集，将${\mathcal{L}}_{\text{slim }}\left\lbrack  {r}_{c}^{\min }\right\rbrack$中所有集合的最小子集重新插入到$\mathcal{H}$中。通过这种方式，我们可以跳过那些大于${r}_{c}^{\min }$且小于${r}_{c}^{top}$的$c$ -子集，这些子集一定是${r}_{c}^{min}$的相邻冗余$c$ -子集，因为这些$c$ -子集的倒排表都是${\mathcal{L}}_{\text{slim }}\left\lbrack  {r}_{c}^{\text{min }}\right\rbrack$的子表（回顾上述基本思想，在这种情况下${r}_{c}^{\min }$对应于${r}_{c}$，${r}_{c}^{top}$对应于${r}_{c}^{\prime \prime }$）。

The pseudo code of the HeapDedup method is shown in Algorithm 3. HeapDedup improves on HeapSkip by lazily reinserting the min-subsets. Instead of reinserting a min-subset to the min-heap every time, HeapDedup reinserts a batch of min-subsets to the min-heap by binary searching the min-subsets no smaller than ${r}_{c}^{top}$ when ${r}_{c}^{min} \neq  {r}_{c}^{top}$ (Lines 1 to 3) and does nothing when ${r}_{c}^{\min } = {r}_{c}^{top}$ .

HeapDedup方法的伪代码如算法3所示。HeapDedup通过延迟重新插入最小子集对HeapSkip进行了改进。与每次将一个最小子集重新插入到最小堆不同，HeapDedup在${r}_{c}^{min} \neq  {r}_{c}^{top}$时（第1到3行）通过二分查找不小于${r}_{c}^{top}$的最小子集，将一批最小子集重新插入到最小堆中，而在${r}_{c}^{\min } = {r}_{c}^{top}$时不做任何操作。

Example 3. Consider the four small sets ${R}_{1},{R}_{2},{R}_{3}$ and ${R}_{4}$ in Table 1 and suppose that the threshold is $c = 2$ . As illustrate in Figure 2, HeapDedup first inserts the min-subsets $\left\{  {{e}_{1},{e}_{2}}\right\}  ,\left\{  {{e}_{1},{e}_{3}}\right\}  ,\left\{  {{e}_{1},{e}_{3}}\right\}$ and $\left\{  {{e}_{2},{e}_{4}}\right\}$ of the four small sets into a min-heap $\mathcal{H}$ . Next it pops $\mathcal{H}$ ,gets ${r}_{c}^{\min } = \left\{  {{e}_{1},{e}_{2}}\right\}$ from ${R}_{1}$ and reinserts the next ${r}_{c}^{\min }\left\{  {{e}_{1},{e}_{3}}\right\}$ of ${R}_{1}$ to $\mathcal{H}$ . Then it pops $\mathcal{H}$ again and has ${r}_{c}^{\min } = \left\{  {{e}_{1},{e}_{3}}\right\}$ from ${R}_{1}$ and ${r}_{c}^{\text{top }} = \left\{  {{e}_{1},{e}_{3}}\right\}$ . It appends ${R}_{1}$ to ${\mathcal{L}}_{\text{slim }}\left\lbrack  \left\{  {{e}_{1},{e}_{3}}\right\}  \right\rbrack$ . As ${r}_{c}^{\text{min }} = {r}_{c}^{\text{top }}$ , it keeps popping $\mathcal{H}$ and has ${r}_{c}^{\min } = \left\{  {{e}_{1},{e}_{3}}\right\}$ from ${R}_{2}$ and ${r}_{c}^{\text{top }} =$ $\left\{  {{e}_{1},{e}_{3}}\right\}$ . It appends ${R}_{2}$ to ${\mathcal{L}}_{\text{slim }}\left\lbrack  \left\{  {{e}_{1},{e}_{3}}\right\}  \right\rbrack$ . As ${r}_{c}^{\min } = {r}_{c}^{\text{top }}$ ,it pops $\mathcal{H}$ and has ${r}_{c}^{\min } = \left\{  {{e}_{1},{e}_{3}}\right\}$ from ${R}_{3}$ and ${r}_{c}^{\text{top }} = \left\{  {{e}_{2},{e}_{4}}\right\}$ . It appends ${R}_{3}$ to ${\mathcal{L}}_{\text{slim }}\left\lbrack  \left\{  {{e}_{1},{e}_{3}}\right\}  \right\rbrack$ . As ${r}_{c}^{\min } \neq  {r}_{c}^{\text{top }}$ ,for the sets ${R}_{1},{R}_{2}$ ,and ${R}_{3}$ in ${\mathcal{L}}_{\text{slim }}\left\lbrack  \left\{  {{e}_{1},{e}_{3}}\right\}  \right\rbrack$ ,it binary searches the first min-subsets in them that are no smaller than ${r}_{c}^{\text{top }}$ . It gets $\left\{  {{e}_{3},{e}_{4}}\right\}$ and $\left\{  {{e}_{3},{e}_{5}}\right\}$ for ${R}_{2}$ and ${R}_{3}$ and reinserts them to $\mathcal{H}$ . It reaches the end of ${R}_{1}$ and does not reinsert any $c$ -subset for ${R}_{1}$ to $\mathcal{H}$ . Iteratively it can build a slimmed inverted index without adjacent redundant c-subsets.

示例3. 考虑表1中的四个小集合${R}_{1},{R}_{2},{R}_{3}$和${R}_{4}$，并假设阈值为$c = 2$。如图2所示，堆去重（HeapDedup）算法首先将这四个小集合的最小子集$\left\{  {{e}_{1},{e}_{2}}\right\}  ,\left\{  {{e}_{1},{e}_{3}}\right\}  ,\left\{  {{e}_{1},{e}_{3}}\right\}$和$\left\{  {{e}_{2},{e}_{4}}\right\}$插入到一个最小堆$\mathcal{H}$中。接下来，它从$\mathcal{H}$中弹出元素，从${R}_{1}$中获取${r}_{c}^{\min } = \left\{  {{e}_{1},{e}_{2}}\right\}$，并将${R}_{1}$的下一个${r}_{c}^{\min }\left\{  {{e}_{1},{e}_{3}}\right\}$重新插入到$\mathcal{H}$中。然后，它再次从$\mathcal{H}$中弹出元素，从${R}_{1}$和${r}_{c}^{\text{top }} = \left\{  {{e}_{1},{e}_{3}}\right\}$中得到${r}_{c}^{\min } = \left\{  {{e}_{1},{e}_{3}}\right\}$。它将${R}_{1}$追加到${\mathcal{L}}_{\text{slim }}\left\lbrack  \left\{  {{e}_{1},{e}_{3}}\right\}  \right\rbrack$中。由于${r}_{c}^{\text{min }} = {r}_{c}^{\text{top }}$，它继续从$\mathcal{H}$中弹出元素，从${R}_{2}$和${r}_{c}^{\text{top }} =$ $\left\{  {{e}_{1},{e}_{3}}\right\}$中得到${r}_{c}^{\min } = \left\{  {{e}_{1},{e}_{3}}\right\}$。它将${R}_{2}$追加到${\mathcal{L}}_{\text{slim }}\left\lbrack  \left\{  {{e}_{1},{e}_{3}}\right\}  \right\rbrack$中。由于${r}_{c}^{\min } = {r}_{c}^{\text{top }}$，它从$\mathcal{H}$中弹出元素，从${R}_{3}$和${r}_{c}^{\text{top }} = \left\{  {{e}_{2},{e}_{4}}\right\}$中得到${r}_{c}^{\min } = \left\{  {{e}_{1},{e}_{3}}\right\}$。它将${R}_{3}$追加到${\mathcal{L}}_{\text{slim }}\left\lbrack  \left\{  {{e}_{1},{e}_{3}}\right\}  \right\rbrack$中。由于${r}_{c}^{\min } \neq  {r}_{c}^{\text{top }}$，对于${\mathcal{L}}_{\text{slim }}\left\lbrack  \left\{  {{e}_{1},{e}_{3}}\right\}  \right\rbrack$中的集合${R}_{1},{R}_{2}$和${R}_{3}$，它对其中不小于${r}_{c}^{\text{top }}$的第一个最小子集进行二分查找。它为${R}_{2}$和${R}_{3}$分别找到$\left\{  {{e}_{3},{e}_{4}}\right\}$和$\left\{  {{e}_{3},{e}_{5}}\right\}$，并将它们重新插入到$\mathcal{H}$中。它到达了${R}_{1}$的末尾，并且没有将${R}_{1}$的任何$c$ - 子集重新插入到$\mathcal{H}$中。通过迭代，它可以构建一个没有相邻冗余c - 子集的精简倒排索引。

### 4.3 Blocking c-subsets

### 4.3 阻塞c-子集

For each small set, the heap-based methods need to maintain a min-subset in the min-heap. Thus the heap size is $\left| {\mathcal{R}}_{s}\right|$ ,which is rather large and leads a high heap adjusting cost (the time cost for each heap adjusting operation is $c \times  \log \left| {\mathcal{R}}_{s}\right|$ as each $c$ -subset comparison takes $c$ cost).

对于每个小集合，基于堆的方法需要在最小堆中维护一个最小子集。因此堆的大小为$\left| {\mathcal{R}}_{s}\right|$，这相当大，会导致较高的堆调整成本（每次堆调整操作的时间成本为$c \times  \log \left| {\mathcal{R}}_{s}\right|$，因为每次$c$ -子集比较需要$c$的成本）。

<!-- Media -->

Algorithm 3: HeapDEDUP

算法3：HeapDEDUP

---

	Input: ${\mathcal{R}}_{s}$ : all the small sets; $c$ : threshold;

	输入：${\mathcal{R}}_{s}$：所有小集合；$c$：阈值；

	Output: ${\mathcal{L}}_{\text{slim }}$ : a slimmed inverted index for ${\mathcal{R}}_{s}$ ;

	输出：${\mathcal{L}}_{\text{slim }}$：${\mathcal{R}}_{s}$的精简倒排索引；

	// replace lines 6 to 9 of Algorithm 2

	// 替换算法2的第6行到第9行

	if ${r}_{c}^{top} \neq  {r}_{c}^{\min }$ then

	if ${r}_{c}^{top} \neq  {r}_{c}^{\min }$ 则

			foreach $R$ in ${\mathcal{L}}_{\text{slim }}\left\lbrack  {\mathbf{r}}_{c}^{\text{min }}\right\rbrack$ do

					对于${\mathcal{L}}_{\text{slim }}\left\lbrack  {\mathbf{r}}_{c}^{\text{min }}\right\rbrack$中的每个$R$ 执行

3 binary search the first $c$ -subset in $R$ that is no smaller

		二分查找$R$中第一个不小于$c$的$c$ -子集

					than ${r}_{c}^{top}$ and reinsert it to $\mathcal{H}$ ;

							并将其重新插入到$\mathcal{H}$中；

	else

	else

			continue; // lazy reinsertion

					继续； // 惰性重新插入

---

<!-- Media -->

To address this issue,we propose to block the $c$ -subsets by their smallest elements. As shown in Figure 5,consider the block ${\mathcal{B}}_{e}$ with smallest element $e$ . As the other $c$ -subsets either have the smallest elements larger than $e$ or smaller than $e$ ,they must be different from the $c$ -subsets in the block ${\mathcal{B}}_{e}$ . Thus we can independently utilize the heap-based methods to build a part of the slimmed inverted index for the $c$ -subsets in ${\mathcal{B}}_{e}$ with a smaller heap (as we do not need to maintain the min-subsets for those small sets without $c$ -subsets in ${\mathcal{B}}_{e}$ ,such as ${R}_{a}$ and ${R}_{b}$ in the figure). Next we formalize our idea.

为了解决这个问题，我们提议根据$c$ -子集的最小元素对其进行分块。如图5所示，考虑最小元素为$e$的块${\mathcal{B}}_{e}$。由于其他$c$ -子集的最小元素要么大于$e$，要么小于$e$，它们一定与块${\mathcal{B}}_{e}$中的$c$ -子集不同。因此，我们可以独立地使用基于堆的方法，用一个较小的堆为块${\mathcal{B}}_{e}$中的$c$ -子集构建部分精简倒排索引（因为我们不需要为那些在${\mathcal{B}}_{e}$中没有$c$ -子集的小集合维护最小子集，如图中的${R}_{a}$和${R}_{b}$）。接下来，我们将我们的想法形式化。

We first fix a global order for all the elements. Then we build an inverted index $\mathcal{I}$ for all the elements in ${\mathcal{R}}_{s}$ to facilitate blocking the $c$ -subsets. The inverted list $\mathcal{I}\left\lbrack  e\right\rbrack$ of the element $e$ consists of all the small sets containing $e$ . As all the $c$ -subsets in a block ${\mathcal{B}}_{e}$ must contain the element $e$ while all the small sets having element $e$ are in $\mathcal{I}\left\lbrack  e\right\rbrack$ ,the $c$ -subsets in the block ${\mathcal{B}}_{e}$ are from and only from the sets in $\mathcal{I}\left\lbrack  e\right\rbrack$ . Thus for each inverted list $\mathcal{I}\left\lbrack  e\right\rbrack$ ,we apply the heap-based method on all the sets in $\mathcal{I}\left\lbrack  e\right\rbrack$ to construct the inverted list ${\mathcal{L}}_{\text{slim }}\left\lbrack  {r}_{c}\right\rbrack$ for every $c$ -subset ${r}_{c} \in  {\mathcal{B}}_{e}$ . Note we only need to access those $c$ -subsets with the smallest element $e$ in the sets in $I\left\lbrack  e\right\rbrack$ . To achieve this,we can perform a simulation by removing those elements no larger than $e$ in the sets in $\mathcal{I}\left\lbrack  e\right\rbrack$ and decreasing the threshold by 1 when applying the heap-based methods. ${}^{2}$

我们首先为所有元素确定一个全局顺序。然后，我们为${\mathcal{R}}_{s}$中的所有元素构建一个倒排索引$\mathcal{I}$，以便对$c$ -子集进行分块。元素$e$的倒排列表$\mathcal{I}\left\lbrack  e\right\rbrack$由所有包含$e$的小集合组成。由于块${\mathcal{B}}_{e}$中的所有$c$ -子集必须包含元素$e$，而所有包含元素$e$的小集合都在$\mathcal{I}\left\lbrack  e\right\rbrack$中，块${\mathcal{B}}_{e}$中的$c$ -子集仅来自$\mathcal{I}\left\lbrack  e\right\rbrack$中的集合。因此，对于每个倒排列表$\mathcal{I}\left\lbrack  e\right\rbrack$，我们对$\mathcal{I}\left\lbrack  e\right\rbrack$中的所有集合应用基于堆的方法，为每个$c$ -子集${r}_{c} \in  {\mathcal{B}}_{e}$构建倒排列表${\mathcal{L}}_{\text{slim }}\left\lbrack  {r}_{c}\right\rbrack$。注意，我们只需要访问$I\left\lbrack  e\right\rbrack$中最小元素为$e$的那些$c$ -子集。为了实现这一点，我们可以通过移除$\mathcal{I}\left\lbrack  e\right\rbrack$中不大于$e$的元素，并在应用基于堆的方法时将阈值减1来进行模拟。${}^{2}$

The pseudo code of the BlockDedup is shown in Algorithm 4. It first fixes a global order for elements and then builds the element inverted index $\mathcal{I}$ (Lines 1 to 2). Next for each inverted list $\mathcal{I}\left\lbrack  e\right\rbrack   \in$ $I$ ,it generates a temporary set ${R}_{tmp}$ of sets by removing all the elements no larger than $e$ in the sets in $\mathcal{I}\left\lbrack  e\right\rbrack  {\left( \text{ Line }4\right) }^{3}$ . Then it applies the HeapDedup method on ${R}_{tmp}$ with the threshold $c - 1$ to build a part of the slimmed inverted index ${\mathcal{L}}_{\text{slim }}$ (Line 5). Note the blocking-based method can also work with HeapSkip here and is named as BlockSkip in the experiment. Finally it returns a slimmed inverted index ${\mathcal{L}}_{\text{slim }}$ (Line 6).

块去重（BlockDedup）的伪代码如算法4所示。它首先为元素确定一个全局顺序，然后构建元素倒排索引 $\mathcal{I}$（第1至2行）。接下来，对于每个倒排列表 $\mathcal{I}\left\lbrack  e\right\rbrack   \in$ $I$，它通过移除 $\mathcal{I}\left\lbrack  e\right\rbrack  {\left( \text{ Line }4\right) }^{3}$ 中集合里所有不大于 $e$ 的元素，生成一个临时集合 ${R}_{tmp}$。然后，它以阈值 $c - 1$ 对 ${R}_{tmp}$ 应用堆去重（HeapDedup）方法，以构建精简倒排索引 ${\mathcal{L}}_{\text{slim }}$ 的一部分（第5行）。注意，基于分块的方法在这里也可以与堆跳过（HeapSkip）方法配合使用，在实验中被称为块跳过（BlockSkip）。最后，它返回一个精简倒排索引 ${\mathcal{L}}_{\text{slim }}$（第6行）。

EXAMPLE 4. Consider the small sets in Table 1 and suppose that the threshold $c = 2$ . In Figure 2,we can group all their $c$ -subsets to 5 blocks. The $c$ -subsets with ${e}_{1},{e}_{2},{e}_{3},{e}_{4}$ ,and ${e}_{5}$ as their smallest element respectively. The block of ${e}_{3}$ contains three $c$ -subsets, $\left\{  {{e}_{3},{e}_{4}}\right\}$ , $\left\{  {{e}_{3},{e}_{5},}\right\}$ ,and $\left\{  {{e}_{3},{e}_{7}}\right\}$ . Note the $c$ -subset $\left\{  {{e}_{1},{e}_{3}}\right\}$ is not belong to this block as its minimum element is ${e}_{1}$ rather than ${e}_{3}$ . The block of ${e}_{3}$ only has c-subsets from two small sets, ${R}_{2}$ and ${R}_{3}$ . We can utilize the

示例4. 考虑表1中的小集合，并假设阈值为 $c = 2$。在图2中，我们可以将它们所有的 $c$ -子集分组为5个块。这些 $c$ -子集分别以 ${e}_{1},{e}_{2},{e}_{3},{e}_{4}$ 和 ${e}_{5}$ 作为其最小元素。以 ${e}_{3}$ 为最小元素的块包含三个 $c$ -子集，即 $\left\{  {{e}_{3},{e}_{4}}\right\}$、$\left\{  {{e}_{3},{e}_{5},}\right\}$ 和 $\left\{  {{e}_{3},{e}_{7}}\right\}$。注意，$c$ -子集 $\left\{  {{e}_{1},{e}_{3}}\right\}$ 不属于这个块，因为它的最小元素是 ${e}_{1}$ 而不是 ${e}_{3}$。以 ${e}_{3}$ 为最小元素的块仅包含来自两个小集合 ${R}_{2}$ 和 ${R}_{3}$ 的 $c$ -子集。我们可以利用

---

<!-- Footnote -->

${}^{2}$ In our implementation,we do not remove the elements and copy all the sets. Instead we omit the elements no larger than $e$ when accessing the $c$ -subsets in heap-based methods.

${}^{2}$ 在我们的实现中，我们不会移除元素并复制所有集合。相反，在基于堆的方法中访问 $c$ -子集时，我们会忽略不大于 $e$ 的元素。

${}^{3}$ If the size of a set is smaller than $c - 1$ after removing the elements no larger than $e$ , we do not need to add it into ${R}_{tmp}$ . Instead,we drop it.

${}^{3}$ 如果一个集合在移除不大于 $e$ 的元素后，其大小小于 $c - 1$，我们不需要将其添加到 ${R}_{tmp}$ 中。相反，我们将其舍弃。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: the global order of c-subsets All the c-subsets with smallest elements $> e$ ... ... nverted lists for these c-subsets ... All the c-subsets with All the c-subsets with smallest elements $< e$ smallest elements $= e$ ${R}_{a}$ ... ... ${R}_{b}$ use a smaller heap to build -->

<img src="https://cdn.noedgeai.com/0195ccc7-1611-78aa-a97d-afb7fc00df51_6.jpg?x=160&y=162&w=706&h=387&r=0"/>

Figure 5: Block $c$ -subsets with smallest element $e$ . heap-based methods on this block with a smaller heap size of 2 to build a part of a slimmed inverted index.

图5：以 $e$ 为最小元素的块 $c$ -子集。在这个块上使用堆大小为2的基于堆的方法来构建精简倒排索引的一部分。

<!-- Media -->

## 5 SIZE BOUNDARY SELECTION IN PRACTICE

## 5 实际中的大小边界选择

The complexity analysis in Section 3 gives us the insight that we need to process the small and large sets separately. It gives the size boundary by equating the time complexities of the small and large sets. However there is a gap between the time complexity and the actual time cost. In practice,the number of the enumerated $c$ - subsets is much smaller than ${x}^{c - 1}n$ due to the pruning techniques in Section 4 and the upper bounds used in analyzing the total number of $c$ -subsets under the worst case in Section 3. Moreover,the lengths of the inverted lists of the $c$ -subsets are much shorter than $\sqrt{k}$ in practice and the time cost for generating the results is far smaller than ${x}^{c - 1}n\sqrt{k}$ . Thus the time complexity largely overestimates the time cost for processing the small sets and the suggested size boundary ${\left( \frac{n}{\sqrt{k}}\right) }^{\frac{1}{c}}$ is too small in practice.

第3节中的复杂度分析让我们明白，需要分别处理小集合和大集合。它通过使小集合和大集合的时间复杂度相等来确定大小边界。然而，时间复杂度和实际时间成本之间存在差距。实际上，由于第4节中的剪枝技术以及第3节中在最坏情况下分析 $c$ -子集总数时使用的上界，枚举的 $c$ -子集的数量远小于 ${x}^{c - 1}n$。此外，实际上 $c$ -子集的倒排列表的长度远短于 $\sqrt{k}$，生成结果的时间成本也远小于 ${x}^{c - 1}n\sqrt{k}$。因此，时间复杂度在很大程度上高估了处理小集合的时间成本，并且建议的大小边界 ${\left( \frac{n}{\sqrt{k}}\right) }^{\frac{1}{c}}$ 在实际中太小了。

Next we give the basic idea of our size boundary selection method. Based on the time complexity analysis, with the increasing of the size boundary $x$ ,the time complexity of the small sets $O\left( {{x}^{c - 1}n\sqrt{k}}\right)$ grows more and more sharply while the time complexity of the large sets $O\left( \frac{{n}^{2}}{x}\right)$ falls less and less precipitously. Thus we can increase the size boundary from the smallest set size in $\mathcal{R}$ and estimate the time costs for the small sets and the large sets. We stop increasing the boundary when the time cost for small sets grows more than the decrease of the time cost for large sets, and partition all the sets by the current size boundary. To this end, we show how to estimate the time costs for the small sets and the large sets in Section 5.1 and propose an effective size boundary selection method in Section 5.2.

接下来，我们介绍我们的大小边界选择方法的基本思路。基于时间复杂度分析，随着大小边界 $x$ 的增加，小集合 $O\left( {{x}^{c - 1}n\sqrt{k}}\right)$ 的时间复杂度增长越来越快，而大集合 $O\left( \frac{{n}^{2}}{x}\right)$ 的时间复杂度下降越来越慢。因此，我们可以从 $\mathcal{R}$ 中的最小集合大小开始增加大小边界，并估算小集合和大集合的时间成本。当小集合的时间成本增长超过大集合时间成本的减少时，我们停止增加边界，并根据当前的大小边界对所有集合进行划分。为此，我们将在 5.1 节中展示如何估算小集合和大集合的时间成本，并在 5.2 节中提出一种有效的大小边界选择方法。

### 5.1 Estimating the Time Costs

### 5.1 估算时间成本

Next we estimate the costs for processing large sets and small sets. Estimating the time cost for large sets: In our implementation, we use the ScanCount [13] method to process the large sets. For the large sets,we build an inverted index $I$ for all the elements in the sets in $\mathcal{R}$ . For each large set $R \in  {\mathcal{R}}_{l}$ ,we scan the corresponding inverted lists of its elements and count the occurrences of the other sets in the inverted lists. All the sets with occurrence times no smaller than $c$ are similar to $R$ . We can estimate the time cost for processing the large set $R$ by adding up the lengths of all its inverted lists. Thus the time cost for all the large sets is proportional to $\mathop{\sum }\limits_{{R \in  {\mathcal{R}}_{l}}}\mathop{\sum }\limits_{{e \in  R}}\left| {\mathcal{I}\left\lbrack  e\right\rbrack  }\right|$ . We can get this cost by scanning the entire dataset for one pass.

接下来，我们估算处理大集合和小集合的成本。估算大集合的时间成本：在我们的实现中，我们使用 ScanCount [13] 方法来处理大集合。对于大集合，我们为 $\mathcal{R}$ 中集合的所有元素构建一个倒排索引 $I$。对于每个大集合 $R \in  {\mathcal{R}}_{l}$，我们扫描其元素对应的倒排列表，并统计倒排列表中其他集合的出现次数。所有出现次数不小于 $c$ 的集合都与 $R$ 相似。我们可以通过累加其所有倒排列表的长度来估算处理大集合 $R$ 的时间成本。因此，所有大集合的时间成本与 $\mathop{\sum }\limits_{{R \in  {\mathcal{R}}_{l}}}\mathop{\sum }\limits_{{e \in  R}}\left| {\mathcal{I}\left\lbrack  e\right\rbrack  }\right|$ 成正比。我们可以通过对整个数据集进行一次扫描来得到这个成本。

Estimating the time cost for small sets. For the small sets, the size-aware method uses a heap to manage the min-subsets, accesses the $c$ -subsets of each small set by binary searching,and generates the results by scanning the $c$ -subset inverted index. Thus there are three major costs, the heap adjusting cost, the binary searching cost, and the result generation cost. Next we estimate them.

估算小集合的时间成本。对于小集合，大小感知方法使用一个堆来管理最小子集，通过二分查找访问每个小集合的 $c$ -子集，并通过扫描 $c$ -子集倒排索引生成结果。因此，主要有三种成本，即堆调整成本、二分查找成本和结果生成成本。接下来我们对它们进行估算。

<!-- Media -->

Algorithm 4: BLOCKDEDUP

算法 4：BLOCKDEDUP

---

Input: ${\mathcal{R}}_{s}$ : all the small sets; $c$ : threshold;

输入：${\mathcal{R}}_{s}$：所有小集合；$c$：阈值；

Output: ${\mathcal{L}}_{\text{slim }}$ : a slimmed inverted index for ${\mathcal{R}}_{s}$ ;

输出：${\mathcal{L}}_{\text{slim }}$：${\mathcal{R}}_{s}$ 的精简倒排索引；

Fix a global order for all the elements in ${\mathcal{R}}_{s}$ ;

为 ${\mathcal{R}}_{s}$ 中的所有元素确定一个全局顺序；

Build an inverted index $\mathcal{I}$ for all the elements in ${\mathcal{R}}_{s}$ ;

为 ${\mathcal{R}}_{s}$ 中的所有元素构建一个倒排索引 $\mathcal{I}$；

foreach $\mathcal{I}\left\lbrack  e\right\rbrack$ in $\mathcal{I}$ do

对于 $\mathcal{I}$ 中的每个 $\mathcal{I}\left\lbrack  e\right\rbrack$ 执行

	${\mathcal{R}}_{\text{tmp }} =$ sets in $\mathcal{I}\left\lbrack  e\right\rbrack$ with elements $\leq  e$ removed;

	${\mathcal{R}}_{\text{tmp }} =$ 移除元素 $\leq  e$ 后的 $\mathcal{I}\left\lbrack  e\right\rbrack$ 中的集合；

	${\mathcal{L}}_{\text{slim }} = {\mathcal{L}}_{\text{slim }} \cup$ HeapDedup $\left( {{\mathcal{R}}_{\text{tmp }},c - 1}\right)$ ;

	${\mathcal{L}}_{\text{slim }} = {\mathcal{L}}_{\text{slim }} \cup$ 堆去重 $\left( {{\mathcal{R}}_{\text{tmp }},c - 1}\right)$；

	// ${\mathcal{L}}_{\text{slim }} = {\mathcal{L}}_{\text{slim }} \cup$ HeapSkip $\left( {{\mathcal{R}}_{\text{tmp }},c - 1}\right)$ for

	// ${\mathcal{L}}_{\text{slim }} = {\mathcal{L}}_{\text{slim }} \cup$ 堆跳过 $\left( {{\mathcal{R}}_{\text{tmp }},c - 1}\right)$ 用于

		BlockSkip

		块跳过

return ${\mathcal{L}}_{\text{slim }}$

返回 ${\mathcal{L}}_{\text{slim }}$

---

<!-- Media -->

We first estimate the result generation cost. Obviously the result generation cost is proportional to the number of $c$ -subsets shared by the small sets as each set pair generated from the $c$ -subset inverted index corresponds to a $c$ -subset shared by the set pair and any small set pair sharing a $c$ -subset corresponds to two entries in the inverted list of this $c$ -subset. Thus we can randomly sample $y$ small set pairs from all the $Y = \left( \begin{matrix} \left| {\mathcal{R}}_{s}\right| \\  2 \end{matrix}\right)$ small set pairs. Suppose that for the ${i}^{th}$ sampling set pair,they share ${p}_{i}$ common elements; then they share $\left( \begin{matrix} {p}_{i} \\  c \end{matrix}\right) c$ -subsets. Based on the law of the large numbers,we can estimate the result generation cost as proportional to $\frac{Y}{y}\mathop{\sum }\limits_{{i = 1}}^{y}\left( \begin{array}{l} {p}_{i} \\  c \end{array}\right)$ as the total number of small set pairs $Y$ is large.

我们首先估算结果生成成本。显然，结果生成成本与小集合共享的$c$ -子集的数量成正比，因为从$c$ -子集倒排索引生成的每个集合对都对应于该集合对共享的一个$c$ -子集，并且任何共享一个$c$ -子集的小集合对都对应于这个$c$ -子集倒排列表中的两个条目。因此，我们可以从所有$Y = \left( \begin{matrix} \left| {\mathcal{R}}_{s}\right| \\  2 \end{matrix}\right)$个小集合对中随机抽取$y$个小集合对。假设对于${i}^{th}$个抽样集合对，它们共享${p}_{i}$个公共元素；那么它们共享$\left( \begin{matrix} {p}_{i} \\  c \end{matrix}\right) c$ -子集。根据大数定律，由于小集合对的总数$Y$很大，我们可以估算结果生成成本与$\frac{Y}{y}\mathop{\sum }\limits_{{i = 1}}^{y}\left( \begin{array}{l} {p}_{i} \\  c \end{array}\right)$成正比。

Next we estimate the heap adjusting cost and the binary search cost. The size-aware method blocks the $c$ -subsets based on their smallest elements and utilizes the heap-based methods to process each block. There are a large number of distinct elements, and the number of blocks is also large (the number of blocks is the same as the number of distinct elements). We can randomly sample a number of blocks to estimate the heap adjusting cost and the binary searching cost for all the blocks. More specifically, for each sample block, we run the heap-based method. For each heap adjusting operation,we estimate its cost as proportional to $c\log h$ where $h$ is the current heap size. For each binary search operation, we estimate its cost as proportional to $c\log t$ where $t$ is the size of the set on which we do a binary search. Suppose that we randomly sample $z$ blocks out of all $Z = \left| \mathcal{I}\right|$ blocks and have the heap adjusting cost for the ${i}^{th}$ block is proportional to ${H}_{i}$ and the binary search cost is proportional to ${T}_{i}$ ; then based on the law of large numbers,we can estimate that the heap adjusting cost and binary searching cost for all the blocks are proportional to $\frac{Z}{z}\mathop{\sum }\limits_{{i = 1}}^{z}\left( {{H}_{i} + {T}_{i}}\right)$ as the number of blocks $Z$ is quite large.

接下来，我们估算堆调整成本和二分查找成本。考虑大小的方法根据$c$ -子集的最小元素对其进行分块，并利用基于堆的方法处理每个块。存在大量不同的元素，块的数量也很大（块的数量与不同元素的数量相同）。我们可以随机抽取一定数量的块来估算所有块的堆调整成本和二分查找成本。更具体地说，对于每个抽样块，我们运行基于堆的方法。对于每个堆调整操作，我们估算其成本与$c\log h$成正比，其中$h$是当前堆的大小。对于每个二分查找操作，我们估算其成本与$c\log t$成正比，其中$t$是我们进行二分查找的集合的大小。假设我们从所有$Z = \left| \mathcal{I}\right|$个块中随机抽取$z$个块，并且第${i}^{th}$个块的堆调整成本与${H}_{i}$成正比，二分查找成本与${T}_{i}$成正比；那么根据大数定律，由于块的数量$Z$相当大，我们可以估算所有块的堆调整成本和二分查找成本与$\frac{Z}{z}\mathop{\sum }\limits_{{i = 1}}^{z}\left( {{H}_{i} + {T}_{i}}\right)$成正比。

### 5.2 The Size Boundary Selection Method

### 5.2 大小边界选择方法

In this section, we propose a size boundary selection method. Based on Section 3.2, the time complexities of the small sets and the large sets are respectively $O\left( {{x}^{c - 1}n\sqrt{k}}\right)$ and $O\left( \frac{{n}^{2}}{x}\right)$ . The slope of ${x}^{c - 1}n\sqrt{k}$ is always positive and monotonically increasing ${}^{4}$ w.r.t. the size boundary $x$ while the slope of $\frac{{n}^{2}}{x}$ is always negative and also monotonically increasing w.r.t. the size boundary $x$ . This means with the increasing of the size boundary $x$ ,the time complexity of the small sets grows first slowly and then sharply while the time complexity of the large sets falls first precipitously and then slowly. Based on this idea, we propose a cost model which uses the decrease of the time cost for the large sets as the benefit and the increase of the time cost for the small sets as the cost. More specifically,we first set the size boundary $x$ as the smallest set size in $\mathcal{R}$ or the threshold $c$ ,whichever is larger and try to increase $x$ by 1 each time. Let ${\mathcal{R}}_{s}^{x}$ and ${\mathcal{R}}_{l}^{x}$ respectively be the sets of small sets and large sets achieved by the size boundary $x$ . Then we estimate the time costs for ${\mathcal{R}}_{s}^{x},{\mathcal{R}}_{s}^{x + 1},{\mathcal{R}}_{l}^{x}$ ,and ${\mathcal{R}}_{l}^{x + 1}$ as proportional to ${\mathcal{Z}}^{\prime }$ , $Z,{\mathcal{Y}}^{\prime }$ ,and $\mathcal{Y}$ . Next we compare the benefit,which is proportional to ${\mathcal{Y}}^{\prime } - \mathcal{Y}$ ,with the cost,which is proportional to $\mathcal{Z} - {\mathcal{Z}}^{\prime }$ . If the benefit is larger than the cost,we increase the size boundary $x$ by 1 and repeat this procedure. Otherwise,we stop increasing $x$ and dichotomize $\mathcal{R}$ by this size boundary. Note this method makes no assumptions about the distribution of the set sizes. If all the sets have the same size, it will classify all the sets either as small or large, depending on the estimations.

在本节中，我们提出一种大小边界选择方法。基于3.2节，小集合和大集合的时间复杂度分别为$O\left( {{x}^{c - 1}n\sqrt{k}}\right)$和$O\left( \frac{{n}^{2}}{x}\right)$。${x}^{c - 1}n\sqrt{k}$的斜率始终为正，并且相对于大小边界$x$单调递增${}^{4}$，而$\frac{{n}^{2}}{x}$的斜率始终为负，并且相对于大小边界$x$也单调递增。这意味着随着大小边界$x$的增加，小集合的时间复杂度先缓慢增长，然后急剧增长，而大集合的时间复杂度先急剧下降，然后缓慢下降。基于这一思路，我们提出一个成本模型，该模型将大集合时间成本的降低视为收益，将小集合时间成本的增加视为成本。更具体地说，我们首先将大小边界$x$设置为$\mathcal{R}$中的最小集合大小或阈值$c$中的较大值，并尝试每次将$x$增加1。设${\mathcal{R}}_{s}^{x}$和${\mathcal{R}}_{l}^{x}$分别为通过大小边界$x$得到的小集合和大集合。然后我们估计${\mathcal{R}}_{s}^{x},{\mathcal{R}}_{s}^{x + 1},{\mathcal{R}}_{l}^{x}$和${\mathcal{R}}_{l}^{x + 1}$的时间成本与${\mathcal{Z}}^{\prime }$、$Z,{\mathcal{Y}}^{\prime }$和$\mathcal{Y}$成正比。接下来，我们将与${\mathcal{Y}}^{\prime } - \mathcal{Y}$成正比的收益与与$\mathcal{Z} - {\mathcal{Z}}^{\prime }$成正比的成本进行比较。如果收益大于成本，我们将大小边界$x$增加1并重复此过程。否则，我们停止增加$x$，并通过此大小边界对$\mathcal{R}$进行二分。注意，此方法对集合大小的分布不做任何假设。如果所有集合的大小相同，它将根据估计结果将所有集合分类为小集合或大集合。

---

<!-- Footnote -->

${}^{4}$ When $c = 2$ ,though the slope is a constant,the size boundary selection method proposed presently still works.

${}^{4}$ 当$c = 2$时，尽管斜率是一个常数，但目前提出的大小边界选择方法仍然有效。

<!-- Footnote -->

---

<!-- Media -->

Algorithm 5: GETSIZEBOUNDARY

算法5：获取大小边界

---

Input: $\mathcal{R}$ : the dataset; $c$ : the threshold;

输入：$\mathcal{R}$：数据集；$c$：阈值；

Output: $x$ : a size boundary for dichotomizing $\mathcal{R}$ ;

输出：$x$：用于对$\mathcal{R}$进行二分的大小边界；

Set $x$ as the larger of the smallest set size in $\mathcal{R}$ and $c$ ;

将$x$设置为$\mathcal{R}$中的最小集合大小和$c$中的较大值；

Estimate the time cost for ${\mathcal{R}}_{s}^{x}$ as ${\mathcal{Z}}^{\prime }$ and for ${\mathcal{R}}_{l}^{x}$ as ${\mathcal{Y}}^{\prime }$ ;

估计${\mathcal{R}}_{s}^{x}$的时间成本为${\mathcal{Z}}^{\prime }$，${\mathcal{R}}_{l}^{x}$的时间成本为${\mathcal{Y}}^{\prime }$；

while $x$ is no larger than the largest set size in $\mathcal{R}$ do

当$x$不大于$\mathcal{R}$中的最大集合大小时

	Estimate the time cost $\mathcal{Z}$ for the small sets ${\mathcal{R}}_{s}^{x + 1}$ ;

	估计小集合${\mathcal{R}}_{s}^{x + 1}$的时间成本$\mathcal{Z}$；

	Estimate the time cost $\mathcal{Y}$ for the large sets ${\mathcal{R}}_{l}^{x + 1}$ ;

	估计大集合${\mathcal{R}}_{l}^{x + 1}$的时间成本$\mathcal{Y}$；

	if benefit $= {\mathcal{Y}}^{\prime } - \mathcal{Y} \leq  \operatorname{cost} = \mathcal{Z} - {\mathcal{Z}}^{\prime }$ then break

	如果收益$= {\mathcal{Y}}^{\prime } - \mathcal{Y} \leq  \operatorname{cost} = \mathcal{Z} - {\mathcal{Z}}^{\prime }$，则跳出循环

	$x = x + 1,{\mathcal{Y}}^{\prime } = \mathcal{Y}$ and ${\mathcal{Z}}^{\prime } = \mathcal{Z}$ ;

	$x = x + 1,{\mathcal{Y}}^{\prime } = \mathcal{Y}$和${\mathcal{Z}}^{\prime } = \mathcal{Z}$；

return $x$

返回$x$

---

<!-- Media -->

The pseudo-code of the cost-based method is shown in Algorithm 5 . It takes a dataset $\mathcal{R}$ and a threshold $c$ as input and outputs a size boundary $x$ for dichotomizing $\mathcal{R}$ . It first sets $x$ to the larger one of the smallest set size in $\mathcal{R}$ and $c$ (Line 1). Then it estimates the time costs ${\mathcal{Z}}^{\prime }$ and ${\mathcal{Y}}^{\prime }$ for ${\mathcal{R}}_{s}^{x}$ and ${\mathcal{R}}_{l}^{x}$ (Line 2),and the time costs $\mathcal{Z}$ and $\mathcal{Y}$ for ${\mathcal{R}}_{s}^{x + 1}$ and ${\mathcal{R}}_{l}^{x + 1}$ (Lines 4 to 5). If the benefit ${\mathcal{Y}}^{\prime } - \mathcal{Y}$ is smaller than the cost $\mathcal{Z} - {\mathcal{Z}}^{\prime }$ ,it stops and returns $x$ as the size boundary (Line 6). Otherwise it increases $x$ by 1,sets ${\mathcal{Y}}^{\prime }$ to $\mathcal{Y}$ and ${Z}^{\prime }$ to $Z$ (Line 6) and repeats the estimation until $x$ is larger than the largest set size in $\mathcal{R}$ (Line 3).

基于成本的方法的伪代码如算法5所示。该方法将数据集$\mathcal{R}$和阈值$c$作为输入，并输出用于对$\mathcal{R}$进行二分的大小边界$x$。它首先将$x$设置为$\mathcal{R}$中最小集合大小和$c$中的较大值（第1行）。然后，它估计${\mathcal{R}}_{s}^{x}$和${\mathcal{R}}_{l}^{x}$的时间成本${\mathcal{Z}}^{\prime }$和${\mathcal{Y}}^{\prime }$（第2行），以及${\mathcal{R}}_{s}^{x + 1}$和${\mathcal{R}}_{l}^{x + 1}$的时间成本$\mathcal{Z}$和$\mathcal{Y}$（第4至5行）。如果收益${\mathcal{Y}}^{\prime } - \mathcal{Y}$小于成本$\mathcal{Z} - {\mathcal{Z}}^{\prime }$，则停止并返回$x$作为大小边界（第6行）。否则，将$x$加1，将${\mathcal{Y}}^{\prime }$设置为$\mathcal{Y}$，将${Z}^{\prime }$设置为$Z$（第6行），并重复进行估计，直到$x$大于$\mathcal{R}$中最大集合的大小（第3行）。

EXAMPLE 5. Consider the dataset $\mathcal{R}$ in Table 1 and suppose that the threshold is $c = 3$ . The complexity analysis suggests the size boundary as ${\left( \frac{40}{\sqrt{3}}\right) }^{\frac{1}{3}} = {2.8}$ and then all the sets are large sets. Our method first sets the boundary size $x$ as the smallest set size $3.{\mathcal{R}}_{s}^{3}$ is empty and ${\mathcal{R}}_{l}^{3}$ has all the sets. The time cost ${\mathcal{Z}}^{\prime }$ for ${\mathcal{R}}_{s}^{3}$ is 0 while the cost ${\mathcal{Y}}^{\prime }$ for ${\mathcal{R}}_{l}^{3}$ is ${67}.{\mathcal{R}}_{s}^{4}$ contains ${R}_{1}$ and ${\mathcal{R}}_{l}^{4}$ has the other sets. The time cost $\mathcal{Z}$ for ${\mathcal{R}}_{s}^{4}$ is still 0 while the cost $\mathcal{Y}$ for ${\mathcal{R}}_{l}^{4}$ is 64 . As the benefit ${\mathcal{Y}}^{\prime } - \mathcal{Y} = 3$ is larger than the cost $\mathcal{Z} - {\mathcal{Z}}^{\prime } = 0$ ,we increase $x$ to 4 and set ${\mathcal{Y}}^{\prime } = {64}$ and ${\mathcal{Z}}^{\prime } = 0$ . Then ${\mathcal{R}}_{s}^{5} = \left\{  {{R}_{1},{R}_{2},{R}_{3},{R}_{4}}\right\}$ and ${\mathcal{R}}_{l}^{5} = \left\{  {{R}_{5},{R}_{6},{R}_{7}}\right\}$ . The $\operatorname{cost}\mathcal{Y}$ for ${\mathcal{R}}_{1}^{5}$ is 42 while the $\operatorname{cost}\mathcal{Z}$ for ${\mathcal{R}}_{s}^{5}$ is 55 . As the benefit $\mathcal{Y} - {\mathcal{Y}}^{\prime } = {22}$ is smaller than the cost ${\mathcal{Z}}^{\prime } - \mathcal{Z} = {55}$ ,we stop increasing $x$ and set $x = 4$ .

示例5. 考虑表1中的数据集$\mathcal{R}$，并假设阈值为$c = 3$。复杂度分析表明大小边界为${\left( \frac{40}{\sqrt{3}}\right) }^{\frac{1}{3}} = {2.8}$，那么所有集合都是大集合。我们的方法首先将边界大小$x$设为最小集合大小，因为$3.{\mathcal{R}}_{s}^{3}$为空集，而${\mathcal{R}}_{l}^{3}$包含所有集合。对于${\mathcal{R}}_{s}^{3}$的时间成本${\mathcal{Z}}^{\prime }$为0，而对于${\mathcal{R}}_{l}^{3}$的成本${\mathcal{Y}}^{\prime }$为${67}.{\mathcal{R}}_{s}^{4}$，其中包含${R}_{1}$，${\mathcal{R}}_{l}^{4}$包含其他集合。对于${\mathcal{R}}_{s}^{4}$的时间成本$\mathcal{Z}$仍然为0，而对于${\mathcal{R}}_{l}^{4}$的成本$\mathcal{Y}$为64。由于效益${\mathcal{Y}}^{\prime } - \mathcal{Y} = 3$大于成本$\mathcal{Z} - {\mathcal{Z}}^{\prime } = 0$，我们将$x$增加到4，并设置${\mathcal{Y}}^{\prime } = {64}$和${\mathcal{Z}}^{\prime } = 0$。然后得到${\mathcal{R}}_{s}^{5} = \left\{  {{R}_{1},{R}_{2},{R}_{3},{R}_{4}}\right\}$和${\mathcal{R}}_{l}^{5} = \left\{  {{R}_{5},{R}_{6},{R}_{7}}\right\}$。对于${\mathcal{R}}_{1}^{5}$的$\operatorname{cost}\mathcal{Y}$为42，而对于${\mathcal{R}}_{s}^{5}$的$\operatorname{cost}\mathcal{Z}$为55。由于效益$\mathcal{Y} - {\mathcal{Y}}^{\prime } = {22}$小于成本${\mathcal{Z}}^{\prime } - \mathcal{Z} = {55}$，我们停止增加$x$并设置$x = 4$。

<!-- Media -->

Table 2: The dataset details

表2：数据集详情

<table><tr><td/><td>R</td><td>$n$</td><td colspan="3">avg, $\min ,\max \left| R\right|$</td><td>I</td><td colspan="3">avg, $\min ,\max \left| {\mathcal{I}\left\lbrack  e\right\rbrack  }\right|$</td></tr><tr><td>DBLP</td><td>1M</td><td>10M</td><td>10.1</td><td>1</td><td>304</td><td>183K</td><td>55228</td><td>1</td><td>183226</td></tr><tr><td>CLICK</td><td>0.99M</td><td>8M</td><td>8.1</td><td>1</td><td>2,498</td><td>41K</td><td>194</td><td>1</td><td>601374</td></tr><tr><td>ORKUT</td><td>1M</td><td>77M</td><td>77.1</td><td>1</td><td>27,317</td><td>2.9M</td><td>2822</td><td>1</td><td>10785</td></tr><tr><td>ADDRESS</td><td>1M</td><td>7M</td><td>7</td><td>7</td><td>7</td><td>657K</td><td>10.65</td><td>1</td><td>223321</td></tr></table>

<table><tbody><tr><td></td><td>R</td><td>$n$</td><td colspan="3">平均值，$\min ,\max \left| R\right|$</td><td>I</td><td colspan="3">平均值，$\min ,\max \left| {\mathcal{I}\left\lbrack  e\right\rbrack  }\right|$</td></tr><tr><td>计算机科学文献数据库（DBLP）</td><td>1M</td><td>10M</td><td>10.1</td><td>1</td><td>304</td><td>183K</td><td>55228</td><td>1</td><td>183226</td></tr><tr><td>点击（CLICK）</td><td>0.99M</td><td>8M</td><td>8.1</td><td>1</td><td>2,498</td><td>41K</td><td>194</td><td>1</td><td>601374</td></tr><tr><td>聚友网（ORKUT）</td><td>1M</td><td>77M</td><td>77.1</td><td>1</td><td>27,317</td><td>2.9M</td><td>2822</td><td>1</td><td>10785</td></tr><tr><td>地址（ADDRESS）</td><td>1M</td><td>7M</td><td>7</td><td>7</td><td>7</td><td>657K</td><td>10.65</td><td>1</td><td>223321</td></tr></tbody></table>

<!-- Media -->

## 6 EXPERIMENTS

## 6 实验

This section evaluates the efficiency and scalability of our methods.

本节评估我们方法的效率和可扩展性。

### 6.1 Setup

### 6.1 实验设置

We implemented all our proposed techniques and conducted experiments on four real-world datasets ${\mathrm{{DBLP}}}^{5},{\mathrm{{CLICK}}}^{6},{\mathrm{{ORKUT}}}^{7}$ ,and ADDRESS ${}^{8}$ . DBLP is a bibliography dataset from DBLP where each title is a set and each word is an element. CLICK is an anonymized click-stream dataset from a Hungarian on-line news portal where each set is a user and each click record is an element. ORKUT is a social network dataset from Orkut, where each set corresponds to a user and each element is a friend of the user. The friendship relation is undirected such that if two users are friends, they appear in each other's set. We randomly selected 1 million sets from DBLP and ORKUT and almost 1 million sets from CLICK as our datasets. ADDRESS is a collection of addresses crawled from the CSV tables on www.data.gov, where each element is a whitespace-delimited word. We randomly chose 1 million sets with exactly 7 elements to verify that our method worked well for the sets with the same sizes. In the experiment, our size boundary selection method classified all the sets in ADDRESS as small sets. We did not use a dataset in which the sets are of the same size and are all classified as large since SizeAware will use an existing method to process them. In this case SizeAware is the same as the existing method and the results are less interesting. The detailed information of all the datasets are shown in Table 2. We show their size distributions in Appendix E.

我们实现了所有提出的技术，并在四个真实世界的数据集上进行了实验，分别是${\mathrm{{DBLP}}}^{5},{\mathrm{{CLICK}}}^{6},{\mathrm{{ORKUT}}}^{7}$和地址数据集（ADDRESS）${}^{8}$。DBLP是来自计算机科学 bibliography 网站（DBLP）的文献数据集，其中每个标题是一个集合，每个单词是一个元素。CLICK是来自匈牙利一个在线新闻门户的匿名点击流数据集，其中每个集合代表一个用户，每个点击记录是一个元素。ORKUT是来自社交网站（Orkut）的社交网络数据集，其中每个集合对应一个用户，每个元素是该用户的一个朋友。友谊关系是无向的，即如果两个用户是朋友，他们会出现在彼此的集合中。我们从DBLP和ORKUT中随机选择了100万个集合，从CLICK中选择了近100万个集合作为我们的数据集。ADDRESS是从www.data.gov上的CSV表格中爬取的地址集合，其中每个元素是一个由空格分隔的单词。我们随机选择了100万个恰好包含7个元素的集合，以验证我们的方法对相同大小的集合也能很好地工作。在实验中，我们的大小边界选择方法将ADDRESS中的所有集合都分类为小集合。我们没有使用集合大小相同且都被分类为大集合的数据集，因为大小感知算法（SizeAware）会使用现有方法来处理它们。在这种情况下，SizeAware与现有方法相同，结果的趣味性较低。所有数据集的详细信息如表2所示。我们在附录E中展示了它们的大小分布。

We compared our size-aware algorithm with the following state-of-the-art approaches for set similarity join with overlap constraints. ScanCount [13]: It first builds an inverted index $\mathcal{I}$ for all the sets in a given dataset $\mathcal{R}$ ,where each entry is an element in the sets and is associated with an inverted list, which keeps all the sets that contain the element. Let $\mathcal{I}\left\lbrack  e\right\rbrack$ denote the inverted list of the element $e.\mathcal{I}\left\lbrack  e\right\rbrack$ consists of all the sets containing $e$ . For example,for the dataset $\mathcal{R}$ in Table 1,we have $I\left\lbrack  {e}_{2}\right\rbrack   = \left( {{R}_{1},{R}_{4},{R}_{5}}\right)$ . For each set $R$ ,the ScanCount method scans all the corresponding inverted lists of its elements and counts the occurrence of each set in these inverted lists. Then it outputs all the sets with occurrences no smaller than $c$ as similar sets of $R$ . Let $\left| {\mathcal{I}\left\lbrack  e\right\rbrack  }\right|$ denote the length of the inverted list $\mathcal{I}\left\lbrack  e\right\rbrack$ . For each set in $\mathcal{I}\left\lbrack  e\right\rbrack$ ,the set contains element $e$ and this method needs to scan $\mathcal{I}\left\lbrack  e\right\rbrack$ . As $\mathcal{I}\left\lbrack  e\right\rbrack$ has $\left| {\mathcal{I}\left\lbrack  e\right\rbrack  }\right|$ elements, $\mathcal{I}\left\lbrack  e\right\rbrack$ is scanned $\left| {\mathcal{I}\left\lbrack  e\right\rbrack  }\right|$ times. Thus the time complexity is $O\left( {\mathop{\sum }\limits_{{\mathcal{I}\left\lbrack  e\right\rbrack   \in  \mathcal{I}}}{\left| \mathcal{I}\left\lbrack  e\right\rbrack  \right| }^{2}}\right)  = O\left( {n}^{2}\right)$ . Note that our SizeAware uses this method to process large sets.

我们将我们的大小感知算法与以下具有重叠约束的集合相似性连接的最先进方法进行了比较。扫描计数法（ScanCount）[13]：它首先为给定数据集$\mathcal{R}$中的所有集合构建一个倒排索引$\mathcal{I}$，其中每个条目是集合中的一个元素，并与一个倒排列表相关联，该列表保存了包含该元素的所有集合。用$\mathcal{I}\left\lbrack  e\right\rbrack$表示元素$e.\mathcal{I}\left\lbrack  e\right\rbrack$的倒排列表，它由包含$e$的所有集合组成。例如，对于表1中的数据集$\mathcal{R}$，我们有$I\left\lbrack  {e}_{2}\right\rbrack   = \left( {{R}_{1},{R}_{4},{R}_{5}}\right)$。对于每个集合$R$，ScanCount方法扫描其元素对应的所有倒排列表，并统计这些倒排列表中每个集合的出现次数。然后，它输出所有出现次数不小于$c$的集合作为$R$的相似集合。用$\left| {\mathcal{I}\left\lbrack  e\right\rbrack  }\right|$表示倒排列表$\mathcal{I}\left\lbrack  e\right\rbrack$的长度。对于$\mathcal{I}\left\lbrack  e\right\rbrack$中的每个集合，该集合包含元素$e$，并且该方法需要扫描$\mathcal{I}\left\lbrack  e\right\rbrack$。由于$\mathcal{I}\left\lbrack  e\right\rbrack$有$\left| {\mathcal{I}\left\lbrack  e\right\rbrack  }\right|$个元素，$\mathcal{I}\left\lbrack  e\right\rbrack$被扫描$\left| {\mathcal{I}\left\lbrack  e\right\rbrack  }\right|$次。因此，时间复杂度为$O\left( {\mathop{\sum }\limits_{{\mathcal{I}\left\lbrack  e\right\rbrack   \in  \mathcal{I}}}{\left| \mathcal{I}\left\lbrack  e\right\rbrack  \right| }^{2}}\right)  = O\left( {n}^{2}\right)$。请注意，我们的大小感知算法（SizeAware）使用此方法来处理大集合。

DivideSkip [13]: Same as the ScanCount method, DivideSkip also builds an inverted index for the elements in the sets. However, for each set $R$ ,instead of scanning all the corresponding inverted lists of its elements, DivideSkip first scans some relatively shorter inverted lists to generate candidates and then binary searches the other longer inverted lists to get the finally results ${}^{9}$ . This method still has the worst-case time complexity of $O\left( {n}^{2}\right)$ .

DivideSkip [13]：与ScanCount方法相同，DivideSkip也会为集合中的元素构建倒排索引。然而，对于每个集合$R$，DivideSkip并非扫描其元素对应的所有倒排列表，而是首先扫描一些相对较短的倒排列表以生成候选集，然后对其他较长的倒排列表进行二分查找以获得最终结果${}^{9}$。该方法的最坏情况时间复杂度仍为$O\left( {n}^{2}\right)$。

---

<!-- Footnote -->

${}^{5}$ http://dblp.uni-trier.de/

${}^{5}$ http://dblp.uni-trier.de/

${}^{6}$ http://fimi.cs.helsinki.fi/data/

${}^{6}$ http://fimi.cs.helsinki.fi/data/

${}^{7}$ https://snap.stanford.edu/data/com-Orkut.html

${}^{7}$ https://snap.stanford.edu/data/com-Orkut.html

${}^{8}$ http://www.data.gov

${}^{8}$ http://www.data.gov

${}^{9}$ It divides the short and long inverted lists based on a heuristic [13].

${}^{9}$ 它基于一种启发式方法[13]划分短倒排列表和长倒排列表。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: #of enumed c-subset: ${10}^{14}$ VIII of enumed c-subset ${10}^{14}$ of enumed c-subsets ${10}^{18}$ #of enumed c-subsets ${10}^{10}$ naive heapskif heapdedup ${10}^{8}$ blockskip BSSSSS ${10}^{6}$ ${10}^{4}$ naive 1222222 ${10}^{15}$ heapdedup blockskip DISSSS ${10}^{12}$ blockdedup ${10}^{9}$ ${10}^{6}$ 12 20 6 Threshold c Threshold c (c) ORKUT (set size $\leq  {30}$ only) (d) ADDRESS 150 200 naive VZZZZZ heapskip Elapsed Time (s) heapdedue Elapsed Time ( s ) heapskip 150 blockskip SSSSSS blockdedup EXXXXXX 100 50 blockskip 100 blockdedup 0 0 20 Threshold c Threshold c (g) ORKUT (set size $\leq  {30}$ only) (h) ADDRESS naive VIII heapskip ${10}^{12}$ heapdedup blockskip ESSSSS blockdedup ${10}^{10}$ ${10}^{6}$ ${10}^{6}$ naive heapskip ${10}^{12}$ heapdedup blockskip DESSSS blockdedup ${10}^{10}$ ${10}^{8}$ ${10}^{6}$ 12 12 Threshold c Threshold c (a) DBLP (set size $\leq  {30}$ only) (b) CLICK (set size $\leq  {25}$ only) Elapsed Time (s) 1000 heapskip 250 heapskip heapdedup Elapsed Time (s) heapdedup 200 blockskip blockdedup 100 50 800 blockdedup 600 400 200 0 12 Threshold c Threshold c (e) DBLP (set size $\leq  {30}$ only) (f) CLICK (set size $\leq  {25}$ only) -->

<img src="https://cdn.noedgeai.com/0195ccc7-1611-78aa-a97d-afb7fc00df51_8.jpg?x=142&y=180&w=1458&h=627&r=0"/>

Figure 6: Evaluating the Heap-based Methods

图6：评估基于堆的方法

<!-- Media -->

All-Pairs [3]: All-Pairs first fixes a global order for all the distinct elements in $\mathcal{R}$ ,such as the alphabetical order or the frequency order. Then it sorts the elements in each set by this global order and generates the prefix of each set,where the prefix of the set $R$ consists of its first $\left| R\right|  - c + 1$ elements. It can guarantee that two sets are similar only if their prefixes share at least one common element. Next All-Pairs builds an inverted index for all the elements in the prefixes. For each set $R$ ,it unions all the inverted lists of the elements in the prefix of $R$ as candidates and verifies them by calculating their real similarity to $R$ . Its time complexity is $O\left( {n}^{2}\right)$ . ${}^{10}$ AdaptJoin [30]: PPJoin [33] first proposes a fixed-length prefix scheme where the $l$ -prefix scheme takes the first $\left| R\right|  - c + l$ elements of the set $R$ as its prefix. PPJoin proves that two sets are similar only if their $l$ -prefixes share at least $l$ common elements. AdaptJoin further proposes an adaptive prefix scheme to improve the fixed length prefix scheme. It develops a cost model to select an appropriate prefix scheme for each set. It builds an incremental inverted index for all the elements with position information, i.e., the inverted list of an element consists of all the sets containing this element and its positions in the sets. For the set $R$ with $l$ -prefix scheme, AdaptJoin retrieves all the inverted lists of the elements in its prefix, scans those elements in the prefix of some sets using the position,outputs all the sets sharing at least $l$ common elements in their prefixes as candidates, and verifies them. Nevertheless, its worst-case time complexity is still $O\left( {n}^{2}\right)$ .

All - Pairs [3]：All - Pairs首先为$\mathcal{R}$中所有不同的元素确定一个全局顺序，例如字母顺序或频率顺序。然后，它按照这个全局顺序对每个集合中的元素进行排序，并生成每个集合的前缀，其中集合$R$的前缀由其前$\left| R\right|  - c + 1$个元素组成。它可以保证只有当两个集合的前缀至少共享一个公共元素时，这两个集合才相似。接下来，All - Pairs为前缀中的所有元素构建倒排索引。对于每个集合$R$，它将$R$前缀中元素的所有倒排列表合并为候选集，并通过计算它们与$R$的实际相似度来验证这些候选集。其时间复杂度为$O\left( {n}^{2}\right)$。${}^{10}$ AdaptJoin [30]：PPJoin [33]首先提出了一种固定长度前缀方案，其中$l$ - 前缀方案将集合$R$的前$\left| R\right|  - c + l$个元素作为其前缀。PPJoin证明，只有当两个集合的$l$ - 前缀至少共享$l$个公共元素时，这两个集合才相似。AdaptJoin进一步提出了一种自适应前缀方案来改进固定长度前缀方案。它开发了一个成本模型，为每个集合选择合适的前缀方案。它为所有带有位置信息的元素构建增量倒排索引，即一个元素的倒排列表由包含该元素的所有集合及其在集合中的位置组成。对于采用$l$ - 前缀方案的集合$R$，AdaptJoin检索其前缀中元素的所有倒排列表，利用位置信息扫描某些集合前缀中的元素，输出前缀中至少共享$l$个公共元素的所有集合作为候选集，并对其进行验证。然而，其最坏情况时间复杂度仍为$O\left( {n}^{2}\right)$。

Note ScanCount and DivideSkip were original designated for search queries. We adapted them to do joins by conducting a search query for each set. For all the experiments,we varied $c$ from 4 to 12 for DBLP and CLICK, 4 to 20 for ORKUT, and 2 to 6 for ADDRESS. The thresholds were ${40}\%$ to ${120}\%$ of the average set size on DBLP, ${50}\%$ to ${150}\%$ on CLICK, $5\%$ to ${25}\%$ on ORKUT,and 30% to ${85}\%$ on ADDRESS, which are wide in relation to the average set size.

注意，ScanCount和DivideSkip最初是为搜索查询设计的。我们通过为每个集合执行搜索查询，将它们应用于连接操作。在所有实验中，对于DBLP和CLICK数据集，我们将$c$的值从4变化到12；对于ORKUT数据集，从4变化到20；对于ADDRESS数据集，从2变化到6。阈值方面，DBLP数据集为平均集合大小的${40}\%$到${120}\%$，CLICK数据集为${50}\%$到${150}\%$，ORKUT数据集为$5\%$到${25}\%$，ADDRESS数据集为30%到${85}\%$，这些阈值相对于平均集合大小而言范围较广。

We implemented All-Pairs by ourselves and obtained the source code from the corresponding authors for the rest. All the methods were implemented in C++ and compiled using g++ 4.8.4 with -O3 flag. All experiments were conducted on a machine with Ubuntu 14.04 LTS, an Intel(R) Xeon(R) CPU E7-4830 @ 2.13GHz processor, and 256 GB memory.

我们自己实现了All - Pairs方法，并从相应作者处获取了其余方法的源代码。所有方法均用C++实现，并使用g++ 4.8.4编译器以 - O3标志进行编译。所有实验均在一台运行Ubuntu 14.04 LTS操作系统、配备Intel(R) Xeon(R) CPU E7 - 4830 @ 2.13GHz处理器和256 GB内存的机器上进行。

### 6.2 Evaluating The Heap-based Methods

### 6.2 评估基于堆的方法

The first set of experiments aimed to identify the best heap-based method for processing small sets. For this purpose, we used all the sets with sizes no larger than 30, 25, and 30 from DBLP, CLICK, and ORKUT to conduct the experiment, which results in 998618, 934203, and 359124 small sets respectively. As all the sets in ADDRESS are quite small, we used all of them in this set of experiments.

第一组实验旨在确定处理小集合的最佳基于堆的方法。为此，我们分别使用DBLP、CLICK和ORKUT数据集中大小不超过30、25和30的所有集合进行实验，分别得到998618、934203和359124个小集合。由于ADDRESS数据集中的所有集合都相当小，我们在这组实验中使用了该数据集中的所有集合。

We implemented the following five methods: (1) Naive, which enumerates all $c$ -subsets for each small set; (2) HeapSkip,which utilizes a min-heap to skip unique $c$ -subsets; (3) HeapDedup,which utilizes a min-heap to skip both unique $c$ -subsets and adjacent redundant $c$ -subsets; (4) BlockSkip,which first blocks the $c$ -subsets and then utilizes a min-heap for each block to skip unique $c$ -subsets; (5) BlockDedup,which first blocks the $c$ -subsets and then utilizes a min-heap for each block to skip both unique $c$ -subsets and adjacent redundant $c$ -subsets.

我们实现了以下五种方法：（1）朴素法（Naive），为每个小集合枚举所有 $c$ -子集；（2）堆跳过法（HeapSkip），利用最小堆跳过唯一的 $c$ -子集；（3）堆去重法（HeapDedup），利用最小堆跳过唯一的 $c$ -子集和相邻的冗余 $c$ -子集；（4）块跳过法（BlockSkip），先对 $c$ -子集进行分块，然后对每个块利用最小堆跳过唯一的 $c$ -子集；（5）块去重法（BlockDedup），先对 $c$ -子集进行分块，然后对每个块利用最小堆跳过唯一的 $c$ -子集和相邻的冗余 $c$ -子集。

We first varied the threshold and reported the number of enumerated $c$ -subsets (which is equal to the number of heap popping operations). Figure 6(a)-(d) gives the results. We observed that BlockDedup and HeapDedup enumerated the least number of $c$ - subsets, and reduced that of Naive by up to 6 orders of magnitudes. For example,on ORKUT dataset when $c = {12}$ ,the numbers of enumerated $c$ -subsets for Naive,HeapSkip,BlockSkip,HeapDedup, and BlockDedup were respectively 2.2 trillion, 123 million, 122 million, 4.3 million, and 3.5 million. The reason behind the effectiveness of BlockDedup and HeapDedup is two-fold. First, they can skip all the adjacent redundant $c$ -subsets. Second,their lazy insertion technique can skip more unique $c$ -subsets than HeapSkip and BlockSkip. We can also see that BlockDedup and BlockSkip enumerated a little fewer $c$ -subsets than HeapDedup and HeapSkip. This is because after blocking, some small sets could be directly dropped as they have less than $c$ elements that are larger than the one used for blocking. For ADDRESS dataset, as the set sizes are quite small,the number of $c$ -subsets for each set is limited (no larger than 35 for any $c \in  \left\lbrack  {2,6}\right\rbrack$ ),which leads the gap between different methods much smaller than those of the other datasets. Nevertheless, we still observed that the heap-based methods beat the naive method, and HeapDedup and BlockDedup enumerated less number of $c$ -subsets than HeapSkip and BlockSkip.

我们首先改变阈值，并报告枚举的 $c$ -子集的数量（这等于堆弹出操作的数量）。图 6(a)-(d) 给出了结果。我们观察到，块去重法（BlockDedup）和堆去重法（HeapDedup）枚举的 $c$ -子集数量最少，与朴素法（Naive）相比最多可减少 6 个数量级。例如，在 ORKUT 数据集上，当 $c = {12}$ 时，朴素法（Naive）、堆跳过法（HeapSkip）、块跳过法（BlockSkip）、堆去重法（HeapDedup）和块去重法（BlockDedup）枚举的 $c$ -子集数量分别为 2.2 万亿、1.23 亿、1.22 亿、430 万和 350 万。块去重法（BlockDedup）和堆去重法（HeapDedup）有效的原因有两个方面。首先，它们可以跳过所有相邻的冗余 $c$ -子集。其次，它们的惰性插入技术比堆跳过法（HeapSkip）和块跳过法（BlockSkip）能跳过更多唯一的 $c$ -子集。我们还可以看到，块去重法（BlockDedup）和块跳过法（BlockSkip）枚举的 $c$ -子集比堆去重法（HeapDedup）和堆跳过法（HeapSkip）略少。这是因为分块后，一些小集合可能会因为元素数量少于 $c$ 且大于用于分块的元素而被直接丢弃。对于 ADDRESS 数据集，由于集合大小非常小，每个集合的 $c$ -子集数量有限（对于任何 $c \in  \left\lbrack  {2,6}\right\rbrack$ 都不超过 35 个），这使得不同方法之间的差距比其他数据集小得多。尽管如此，我们仍然观察到基于堆的方法优于朴素法，并且堆去重法（HeapDedup）和块去重法（BlockDedup）枚举的 $c$ -子集数量比堆跳过法（HeapSkip）和块跳过法（BlockSkip）少。

---

<!-- Footnote -->

${}^{10}$ A proof sketch is presented in Appendix C.

${}^{10}$ 附录 C 中给出了证明概要。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: 1000 3000 5000 c=6 Elapsed Time(s) 4000 c=8 c=12 -----目---- c=16 3000 c=20 2000 1000 c=8 -----目---- c=10 c=12 15 20 25 30 35 10 20 30 50 60 Size Boundary X Size Boundary X (b) CLICK (small sets) (c) ORKUT (small sets) 100 c=6 Elapsed Time(s) 95 c=8 90 c=16 c=20 85 80 75 c=10 c=12 15 20 25 30 35 0 30 40 Size Boundary X Size Boundary x (e) CLICK (large sets) (f) ORKUT (large sets) 5000 c=4 Elapsed Time ( s ) c=4 4000 c=8 c=16 3000 2000 1000 c=6 c=10 15 20 25 30 35 0 10 20 30 40 50 60 Size Boundary X Size Boundary x (h) CLICK (total running time) (i) ORKUT (total running time) Elapsed Time(s) c=4 Elapsed Time ( s ) 2500 2000 1500 1000 500 800 c=8 c=10 600 c=12 400 200 10 100 5 10 Size Boundary X (a) DBLP (small sets) Elapsed Time(s) 2000 Elapsed Time(s) 1000 800 600 400 200 c=6 1500 c=10 c=12 1000 500 100 10 Size Boundary x (d) DBLP (large sets) _____ 3000 c=4 Elapsed Time(s) 2500 2000 1500 1000 500 Elapsed Time(s) c=6 $- \infty  -$ 1500 c=10 _____。 1000 500 1 10 100 0 10 Size Boundary x (g) DBLP (total running time) -->

<img src="https://cdn.noedgeai.com/0195ccc7-1611-78aa-a97d-afb7fc00df51_9.jpg?x=186&y=184&w=1380&h=1095&r=0"/>

Figure 7: Evaluating the Size Boundary Selection Method

图 7：评估大小边界选择方法

<!-- Media -->

We also measured the total running time for the heap-based methods by varying the thresholds. The results are shown in Figure 6(e)- (h). We have the following observations. Firstly, BlockDedup and HeapDedup respectively outperformed BlockSkip and HeapSkip all the time as the former had less number of heap popping operations. Secondly, BlockDedup and BlockSkip respectively beat HeapDedup and HeapSkip in all the cases as the former had a smaller unit heap popping cost. Thirdly, BlockDedup consistently achieved the best performance as it not only required fewer popping operations but also had a smaller unit heap popping cost. The following experiments utilized BlockDedup as the designated method to process the small sets, due to its best overall efficiency. We also measured the elapsed time for Naive method. However, for DBLP, CLICK, and ORKUT, Naive reported the out-of-memory error after a long time (>1000s) in almost all the cases. Thus we only reported the results on ADDRESS dataset, as shown in Figure 6(h). We can see that on the only dataset that Naive can handle, BlockDedup still outperformed Naive by several times when $c \in  \left\lbrack  {3,6}\right\rbrack$ . However, Naive beat BlockDedup when $c = 2$ . This is because the sets in ADDRESS are very small,which leads a small total number of $c$ - subsets. In addition,the chance for BlockDedup to skip $c$ -subsets decreased when $c$ becomes small while BlockDedup needs more time to enumerate a $c$ -subset than Naive as it used a heap to do so.

我们还通过改变阈值来测量基于堆的方法的总运行时间。结果如图 6(e)-(h) 所示。我们有以下观察结果。首先，块去重法（BlockDedup）和堆去重法（HeapDedup）始终分别优于块跳过法（BlockSkip）和堆跳过法（HeapSkip），因为前者的堆弹出操作数量更少。其次，在所有情况下，块去重法（BlockDedup）和块跳过法（BlockSkip）分别优于堆去重法（HeapDedup）和堆跳过法（HeapSkip），因为前者的单位堆弹出成本更小。第三，块去重法（BlockDedup）始终实现了最佳性能，因为它不仅需要的弹出操作更少，而且单位堆弹出成本也更小。由于块去重法（BlockDedup）的整体效率最高，以下实验将其作为处理小集合的指定方法。我们还测量了朴素法（Naive）的运行时间。然而，对于 DBLP、CLICK 和 ORKUT 数据集，朴素法（Naive）在几乎所有情况下经过很长时间（>1000 秒）后都会报告内存不足错误。因此，我们仅报告了 ADDRESS 数据集的结果，如图 6(h) 所示。我们可以看到，在朴素法（Naive）唯一能处理的数据集上，当 $c \in  \left\lbrack  {3,6}\right\rbrack$ 时，块去重法（BlockDedup）仍然比朴素法（Naive）快几倍。然而，当 $c = 2$ 时，朴素法（Naive）优于块去重法（BlockDedup）。这是因为 ADDRESS 数据集中的集合非常小，导致 $c$ -子集的总数较少。此外，当 $c$ 变小时，块去重法（BlockDedup）跳过 $c$ -子集的机会减少，而块去重法（BlockDedup）使用堆来枚举一个 $c$ -子集比朴素法（Naive）需要更多时间。

<!-- Media -->

<table><tr><td rowspan="2"/><td colspan="2">by complexity</td><td colspan="3">by our method</td><td colspan="2">the best</td></tr><tr><td>$x$</td><td>time (sec)</td><td>$x$</td><td>time (sec)</td><td>accuracy</td><td>$x$</td><td>time (sec)</td></tr><tr><td>DBLP, $c = 4$</td><td>4</td><td>2042.5</td><td>30</td><td>174.5</td><td>112.6%</td><td>36</td><td>172</td></tr><tr><td>DBLP, $c = 6$</td><td>4</td><td>1894.9</td><td>34</td><td>85.99</td><td>80.9%</td><td>30</td><td>85.56</td></tr><tr><td>DBLP, $c = 8$</td><td>4</td><td>1455.6</td><td>32</td><td>38.41</td><td>98.2%</td><td>32</td><td>38.41</td></tr><tr><td>DBLP, $c = {10}$</td><td>3</td><td>873.2</td><td>29</td><td>20.07</td><td>83.8%</td><td>31</td><td>19.71</td></tr><tr><td>DBLP, $c = {12}$</td><td>3</td><td>392.6</td><td>29</td><td>11.04</td><td>112.7%</td><td>31</td><td>10.78</td></tr><tr><td>CLICK, $c = 4$</td><td>4</td><td>1000</td><td>30</td><td>358.12</td><td>89.1%</td><td>28</td><td>357.5</td></tr><tr><td>CLICK, $c = 6$</td><td>3</td><td>516.4</td><td>23</td><td>270.29</td><td>131.9%</td><td>24</td><td>269.4</td></tr><tr><td>CLICK, $c = 8$</td><td>3</td><td>329.1</td><td>24</td><td>224.44</td><td>80.1%</td><td>23</td><td>222</td></tr><tr><td>CLICK, $c = {10}$</td><td>2</td><td>238.2</td><td>21</td><td>193.4</td><td>91.6%</td><td>21</td><td>193.4</td></tr><tr><td>CLICK, $c = {12}$</td><td>2</td><td>182.3</td><td>25</td><td>182.96</td><td>78.6%</td><td>21</td><td>162.7</td></tr><tr><td>ORKUT, $c = 4$</td><td>5</td><td>149.4</td><td>8</td><td>149.5</td><td>97.0%</td><td>4</td><td>149.4</td></tr><tr><td>ORKUT, $c = 8$</td><td>3</td><td>146.9</td><td>11</td><td>146.8</td><td>115.8%</td><td>13</td><td>146.7</td></tr><tr><td>ORKUT, $c = {12}$</td><td>2</td><td>145.1</td><td>15</td><td>144.8</td><td>97.3%</td><td>16</td><td>144.7</td></tr><tr><td>ORKUT, $c = {16}$</td><td>2</td><td>142.9</td><td>18</td><td>142.7</td><td>84.9%</td><td>13</td><td>142.1</td></tr><tr><td>ORKUT, $c = {20}$</td><td>2</td><td>139.7</td><td>22</td><td>139.4</td><td>91.6%</td><td>22</td><td>139.4</td></tr></table>

<table><tbody><tr><td rowspan="2"></td><td colspan="2">按复杂度</td><td colspan="3">通过我们的方法</td><td colspan="2">最佳</td></tr><tr><td>$x$</td><td>时间（秒）</td><td>$x$</td><td>时间（秒）</td><td>准确率</td><td>$x$</td><td>时间（秒）</td></tr><tr><td>计算机科学文献数据库（DBLP）, $c = 4$</td><td>4</td><td>2042.5</td><td>30</td><td>174.5</td><td>112.6%</td><td>36</td><td>172</td></tr><tr><td>计算机科学文献数据库（DBLP）, $c = 6$</td><td>4</td><td>1894.9</td><td>34</td><td>85.99</td><td>80.9%</td><td>30</td><td>85.56</td></tr><tr><td>计算机科学文献数据库（DBLP）, $c = 8$</td><td>4</td><td>1455.6</td><td>32</td><td>38.41</td><td>98.2%</td><td>32</td><td>38.41</td></tr><tr><td>计算机科学文献数据库（DBLP）, $c = {10}$</td><td>3</td><td>873.2</td><td>29</td><td>20.07</td><td>83.8%</td><td>31</td><td>19.71</td></tr><tr><td>计算机科学文献数据库（DBLP）, $c = {12}$</td><td>3</td><td>392.6</td><td>29</td><td>11.04</td><td>112.7%</td><td>31</td><td>10.78</td></tr><tr><td>点击数据集（CLICK）, $c = 4$</td><td>4</td><td>1000</td><td>30</td><td>358.12</td><td>89.1%</td><td>28</td><td>357.5</td></tr><tr><td>点击数据集（CLICK）, $c = 6$</td><td>3</td><td>516.4</td><td>23</td><td>270.29</td><td>131.9%</td><td>24</td><td>269.4</td></tr><tr><td>点击数据集（CLICK）, $c = 8$</td><td>3</td><td>329.1</td><td>24</td><td>224.44</td><td>80.1%</td><td>23</td><td>222</td></tr><tr><td>点击数据集（CLICK）, $c = {10}$</td><td>2</td><td>238.2</td><td>21</td><td>193.4</td><td>91.6%</td><td>21</td><td>193.4</td></tr><tr><td>点击数据集（CLICK）, $c = {12}$</td><td>2</td><td>182.3</td><td>25</td><td>182.96</td><td>78.6%</td><td>21</td><td>162.7</td></tr><tr><td>社交网络数据集（ORKUT）, $c = 4$</td><td>5</td><td>149.4</td><td>8</td><td>149.5</td><td>97.0%</td><td>4</td><td>149.4</td></tr><tr><td>社交网络数据集（ORKUT）, $c = 8$</td><td>3</td><td>146.9</td><td>11</td><td>146.8</td><td>115.8%</td><td>13</td><td>146.7</td></tr><tr><td>社交网络数据集（ORKUT）, $c = {12}$</td><td>2</td><td>145.1</td><td>15</td><td>144.8</td><td>97.3%</td><td>16</td><td>144.7</td></tr><tr><td>社交网络数据集（ORKUT）, $c = {16}$</td><td>2</td><td>142.9</td><td>18</td><td>142.7</td><td>84.9%</td><td>13</td><td>142.1</td></tr><tr><td>社交网络数据集（ORKUT）, $c = {20}$</td><td>2</td><td>139.7</td><td>22</td><td>139.4</td><td>91.6%</td><td>22</td><td>139.4</td></tr></tbody></table>

Table 3: The selected size boundaries

表3：所选的大小边界

<!-- Media -->

### 6.3 Evaluating The Size Boundary Selection

### 6.3 评估大小边界选择

The experiments in this subsection focused on the behavior of our size-boundary selection method. Note for ADDRESS dataset, as all the sets have exactly the same size, our size boundary selection method classified all of them as small sets. The experiment results were less interesting for ADDRESS than the other three datasets and thus we omitted them here. We first enumerated a number of size boundaries and evaluated the processing time for small sets and large sets in the size-aware algorithm. Figures 7(a)-(c) present the processing time on the small sets, Figures 7(d)-(f) show the results for the large ones, and Figures 7(g)-(i) give the total running time. As the size boundary increased, the cost reduction for the large sets was considerable initially but then became insignificant later; while for the small sets, the cost growth was slow at the beginning, and then dramatically accelerated. For example, on DBLP dataset when $c = {12}$ ,on size boundary 7,17,27,and 37,the elapsed time for small sets was ${0.03}\mathrm{\;s},{0.98}\mathrm{\;s},{4.4}\mathrm{\;s}$ ,and 11.1 s respectively and 390 s, 111s, 7.8s, and 2.2s for large sets. This is consistent with our time complexity analysis in Section 3.2. Due to this tradeoff between small and large sets, the overall cost of the size-aware algorithm first decreased and then increased. For example, on DBLP dataset when $c = {12}$ ,the elapsed time for size boundary12,21,31,61,and 101 was respectively 377s, 35s, 10.8s, 25.4s, and 270s.

本小节的实验重点关注我们的大小边界选择方法的性能。对于ADDRESS数据集，由于所有集合的大小完全相同，我们的大小边界选择方法将它们都归类为小集合。ADDRESS的实验结果不如其他三个数据集有趣，因此我们在此省略。我们首先列举了一些大小边界，并评估了大小感知算法中小集合和大集合的处理时间。图7(a)-(c)展示了小集合的处理时间，图7(d)-(f)展示了大集合的结果，图7(g)-(i)给出了总运行时间。随着大小边界的增加，大集合的成本降低起初相当显著，但后来变得不明显；而对于小集合，成本增长起初缓慢，然后急剧加速。例如，在DBLP数据集上，当$c = {12}$ 时，在大小边界为7、17、27和37的情况下，小集合的耗时分别为${0.03}\mathrm{\;s},{0.98}\mathrm{\;s},{4.4}\mathrm{\;s}$ 和11.1秒，大集合的耗时分别为390秒、111秒、7.8秒和2.2秒。这与我们在3.2节中的时间复杂度分析一致。由于小集合和大集合之间的这种权衡，大小感知算法的总体成本先降低后增加。例如，在DBLP数据集上，当$c = {12}$ 时，大小边界为12、21、31、61和101时的耗时分别为377秒、35秒、10.8秒、25.4秒和270秒。

<!-- Media -->

<!-- figureText: 4000 3000 3000 6000 Elapsed Time ( s 2500 divide = Elapsed Time ( s ) 5000 divide adaptioin all-pairs --x- 3000 2000 1000 adaptioin all-pairs -> 1500 1000 500 0 0 12 16 20 4 Threshold c Threshold c (c) ORKUT (d) ADDRESS Elapsed Time (s) divide - Elapsed Time (s) divide 2500 adaptivir all-pairs - X- 1500 1000 500 3000 adaptioin all-pairs -x- 2000 1000 0 0 8 10 12 8 10 12 Threshold c Threshold c (a) DBLP (b) CLICK -->

<img src="https://cdn.noedgeai.com/0195ccc7-1611-78aa-a97d-afb7fc00df51_10.jpg?x=144&y=184&w=1450&h=310&r=0"/>

Figure 8: Comparison with Existing Methods: Overlap Threshold

图8：与现有方法的比较：重叠阈值

<!-- Media -->

Table 3 shows the size boundaries that (i) were produced by the time complexity analysis, (ii) were chosen by our size-boundary selection method, and (iii) actually gave the best performance. We also reported the running time under each size boundary. We can see that our size boundary selection method was quite effective, and picked fairly good values that were close to the optimal ones. For example,on DBLP when $c = 8$ ,our size boundary selection method selected $x = {32}$ as the boundary which achieved the optimal performance (38.41s) among all the enumerated boundaries. However, the time complexity analysis suggested $x = 4$ as the boundary which led a much worse running time of 1455s. This evidenced that the cost model in our method is accurate. Note the sixth column gives the estimation accuracy for the small sets, which was the ratio of the estimated costs to the real costs for processing small sets. We can see the cost estimation is accurate. The cost estimation for large sets is always accurate as it does not use sampling techniques.

表3展示了（i）通过时间复杂度分析得出的、（ii）由我们的大小边界选择方法选择的以及（iii）实际表现最佳的大小边界。我们还报告了每个大小边界下的运行时间。我们可以看到，我们的大小边界选择方法非常有效，选择的值相当接近最优值。例如，在DBLP数据集上，当$c = 8$ 时，我们的大小边界选择方法选择$x = {32}$ 作为边界，在所有列举的边界中实现了最优性能（38.41秒）。然而，时间复杂度分析建议$x = 4$ 作为边界，导致运行时间长达1455秒，这证明了我们方法中的成本模型是准确的。注意，第六列给出了小集合的估计准确率，即处理小集合的估计成本与实际成本的比率。我们可以看到成本估计是准确的。大集合的成本估计总是准确的，因为它不使用采样技术。

### 6.4 Comparison with Existing Methods: Overlap Threshold

### 6.4 与现有方法的比较：重叠阈值

We compared our size-aware algorithm with four existing methods DivideSkip, All-Pairs, AdaptJoin, and ScanCount by varying the threshold $c$ . Figure 8 reports the total running time as a function of $c$ . Our size-aware method always achieved the best performance and outperformed the others by up to an order of magnitude. For example,on DBLP dataset when $c = 4$ ,the elapsed time for ScanCount, DivideSkip, AdaptJoin, All-Pairs, and SizeAware was respectively ${2585}\mathrm{\;s},{3358}\mathrm{\;s},{2612}\mathrm{\;s},{3509}\mathrm{\;s}$ ,and ${161}\mathrm{\;s}$ . The main reason for this is the existing methods spent considerable time scanning the element inverted lists, while our size-aware method avoided this by separately processing the small sets and the large sets. Moreover, the size boundary selection method can select a good size boundary for the size-aware algorithm. In addition, with the increase of the threshold $c$ ,the running time decreased because there were fewer answers and fewer sets with sizes no smaller than $c$ . We have the same observation on ADDRESS, because SizeAware generates the results directly from the $c$ -subsets,whose total number is small as the sets in ADDRESS are very small. However, scanning the element inverted index took a long time for existing methods. Moreover, when all the sets have the same size, the overlap similarity is equivalent to Jaccard similarity (see details in Appendix D). We compared SizeAware with some additional existing methods for set similarity join with Jaccard constraint on ADDRESS dataset in Appendix E.

我们通过改变阈值$c$ ，将我们的大小感知算法与四种现有方法DivideSkip、All - Pairs、AdaptJoin和ScanCount进行了比较。图8报告了总运行时间与$c$ 的函数关系。我们的大小感知方法始终表现最佳，比其他方法的性能高出一个数量级。例如，在DBLP数据集上，当$c = 4$ 时，ScanCount、DivideSkip、AdaptJoin、All - Pairs和SizeAware的耗时分别为${2585}\mathrm{\;s},{3358}\mathrm{\;s},{2612}\mathrm{\;s},{3509}\mathrm{\;s}$ 和${161}\mathrm{\;s}$ 。主要原因是现有方法花费大量时间扫描元素倒排列表，而我们的大小感知方法通过分别处理小集合和大集合避免了这一点。此外，大小边界选择方法可以为大小感知算法选择一个合适的大小边界。另外，随着阈值$c$ 的增加，运行时间减少，因为答案更少，且大小不小于$c$ 的集合也更少。在ADDRESS数据集上我们有相同的观察结果，因为SizeAware直接从$c$ -子集生成结果，由于ADDRESS中的集合非常小，$c$ -子集的总数也很小。然而，现有方法扫描元素倒排索引需要很长时间。此外，当所有集合的大小相同时，重叠相似度等同于杰卡德相似度（详见附录D）。我们在附录E中比较了SizeAware与ADDRESS数据集上一些具有杰卡德约束的集合相似度连接的其他现有方法。

### 6.5 Comparison with Existing Methods: Scalability

### 6.5 与现有方法的比较：可扩展性

The last set of experiments studied the scalability of our method. We varied the dataset sizes from 1 million to 3 million for DBLP dataset, 250,000 to around 1 million for CLICK, 1 million to 3 million for ORKUT, and 1 million to 3 million for ADDRESS dataset. The elapsed time of SizeAware under different thresholds is reported in Figure 9. We can see that our methods achieved sub-quadratic scalability, which is consistent with our time complexity analysis. For example,on DBLP dataset,when the threshold $c = 4$ ,the elapsed time for 1 million, 1.5 million, 2 million, 2.5 million, and 3 million sets was respectively 200s, 362s, 569s, 788s, and 1044s. This is because SizeAware processes small sets and large sets separately using two methods that are scalable to sets with different sizes. In addition, the size boundary selection method can properly dichotomize the input dataset. We also evaluated the scalability of SizeAware in $\mathcal{R} - \mathcal{S}$ join case and report the results in Appendix E.

最后一组实验研究了我们方法的可扩展性。对于DBLP数据集（数字图书馆计算机科学文献数据集），我们将数据集大小从100万变化到300万；对于CLICK数据集，从25万变化到约100万；对于ORKUT数据集，从100万变化到300万；对于ADDRESS数据集，从100万变化到300万。图9报告了SizeAware在不同阈值下的运行时间。我们可以看到，我们的方法实现了亚二次可扩展性，这与我们的时间复杂度分析一致。例如，在DBLP数据集上，当阈值为$c = 4$时，处理100万、150万、200万、250万和300万集合的运行时间分别为200秒、362秒、569秒、788秒和1044秒。这是因为SizeAware使用两种可扩展到不同大小集合的方法分别处理小集合和大集合。此外，大小边界选择方法可以适当地对输入数据集进行二分。我们还评估了SizeAware在$\mathcal{R} - \mathcal{S}$连接情况下的可扩展性，并在附录E中报告了结果。

We also compared our scalability with the existing work. Figures 10 gives the results. We varied the dataset sizes and reported the elapsed time for different methods under specific thresholds. We can see that our method achieved the best scalability. For example, on ORKUT dataset,under the threshold $c = {12}$ ,when there were 1 million sets, the elapsed time for ScanCount, DivideSkip, AdaptJoin, All-Pairs, and SizeAware was respectively 1130s, 520s, 1600s, ${520}\mathrm{\;s}$ ,and ${150}\mathrm{\;s}$ ; while it was ${9885}\mathrm{\;s},{4585}\mathrm{\;s},{15500}\mathrm{\;s},{4400}\mathrm{\;s}$ ,and ${875}\mathrm{\;s}$ when there were 3 million sets. The elapsed time increased 8.75 , 8.82,9.69,8.46,and 5.83 times when the dataset size increased 3 times. This is because all existing methods had a quadratic worst-case time complexity and their filtering techniques had little effect when the threshold was relatively small compared to the set sizes.

我们还将我们的可扩展性与现有工作进行了比较。图10给出了结果。我们改变了数据集大小，并报告了不同方法在特定阈值下的运行时间。我们可以看到，我们的方法实现了最佳的可扩展性。例如，在ORKUT数据集上，在阈值为$c = {12}$时，当有100万集合时，ScanCount、DivideSkip、AdaptJoin、All - Pairs和SizeAware的运行时间分别为1130秒、520秒、1600秒、${520}\mathrm{\;s}$和${150}\mathrm{\;s}$；而当有300万集合时，运行时间分别为${9885}\mathrm{\;s},{4585}\mathrm{\;s},{15500}\mathrm{\;s},{4400}\mathrm{\;s}$和${875}\mathrm{\;s}$。当数据集大小增加3倍时，运行时间分别增加了8.75倍、8.82倍、9.69倍、8.46倍和5.83倍。这是因为所有现有方法的最坏情况时间复杂度都是二次的，并且当阈值相对于集合大小较小时，它们的过滤技术效果甚微。

## 7 RELATED WORK

## 7 相关工作

Set Similarity Join and Search with Overlap Constraints. Broder et al. [5] proposed to build an inverted index for the elements and enumerate every set pair in each inverted list to find the set pairs with enough overlap. This is different from our method for small sets where we resort to element subsets with size $c$ . Sarawagi et al. [25] proposed a threshold sensitive list merge algorithm for set similarity join. Li et al. [13] improved the list merge algorithm and adapted it to set similarity search. Chaudhuri et al. [7] proposed the prefix filter technique and used it as a primitive operator in a database system for similarity join. Bayardo et al. [3] proposed a similar approach for solving the same problem under in-memory setting. Wang et al. [30] improved the prefix filter by proposing a cost model to create adaptive prefix filters. Teflioudi et al. [29] studied the inner product join problem which takes vectors instead of sets as the input and utilizes the vector inner product in the join constraint (instead of the overlap as in our problem).

带重叠约束的集合相似度连接与搜索。布罗德（Broder）等人[5]提出为元素构建倒排索引，并枚举每个倒排列表中的每个集合对，以找到具有足够重叠的集合对。这与我们处理小集合的方法不同，我们的方法是借助大小为$c$的元素子集。萨拉瓦吉（Sarawagi）等人[25]提出了一种用于集合相似度连接的阈值敏感列表合并算法。李（Li）等人[13]改进了列表合并算法，并将其应用于集合相似度搜索。乔杜里（Chaudhuri）等人[7]提出了前缀过滤技术，并将其用作数据库系统中进行相似度连接的基本操作符。巴亚尔多（Bayardo）等人[3]提出了一种在内存环境下解决相同问题的类似方法。王（Wang）等人[30]通过提出一个成本模型来创建自适应前缀过滤器，改进了前缀过滤技术。泰夫柳迪（Teflioudi）等人[29]研究了内积连接问题，该问题以向量而非集合作为输入，并在连接约束中利用向量内积（而不是像我们的问题中那样使用重叠）。

<!-- Media -->

<!-- figureText: 400 1000 1500 Elapsed Time ( S ) c=8 一日一 Elapsed Time (s) 1000 c=6 500 800 c=12 --------- 600 400 200 1m 1.5m 2m 2.5r $3\mathrm{m}$ 1m 1.5m 2m 2.5r 3m Dataset Sizes Dataset Sizes (c) ORKUT (d) ADDRESS Figure 9: Scalability under Different Overlap Threshold scancount scancount Elapsed Time ( s ) 15000 divide 一日一 Elapsed Time (s) 20000 divide 一日一 adapt joir 15000 all-pairs sizeaware 10000 5000 adaptioir all-pairs 10000 sizeaware 5000 1m 1.5m 2m 2.5n 3m 1m 1.5m 2m 2.5m 3m Dataset Sizes Dataset Sizes (c) ORKUT (c=12) (d) ADDRESS (c=4) Elapsed Time (s) c=6 --日--- Elapsed Time (s) 300 200 100 800 c=8 --------- 600 400 200 1m 2.5 3m ${250}\mathrm{k}$ 500k 750k 1m Dataset Sizes (a) DBLP (b) CLICK 30000 3500 scancount scancount Elapsed Time (s) divide 一日一 Elapsed Time (s) 3000 divide 日一 2500 adapt join all-pairs 2000 sizeaware 1500 500 adaptjoin 20000 all-pairs " *- 10000 0 1m . . 5m 2m 2.5m 3m ${250}\mathrm{k}$ 500k 750k 1m Dataset Sizes Dataset Sizes (a) DBLP (c=6) (b) CLICK (c=4) Figure 10: Comparison with Existir -->

<img src="https://cdn.noedgeai.com/0195ccc7-1611-78aa-a97d-afb7fc00df51_11.jpg?x=140&y=191&w=1460&h=668&r=0"/>

<!-- Media -->

Similarity Join and Search with Other Constraints. Similarity join and search with other constraints, such as Jaccard, Cosine, Hamming, Edit Distance and Containment, are extensively studied $\left\lbrack  {2,3,7 - 9,{15},{19},{31},{33}}\right\rbrack$ . Xiao et al. [33] proposed PPJoin and PPJoin+ for set similarity join with Jaccard, Cosine and Dice constraints which improve the prefix filter by considering the element positions. Bouros et al. [4] designed GroupJoin to group the same prefixes to share computation. Wang et al. [31] developed SKJ which can skip scanning a part of the inverted lists. Mann et al. [17] proposed PEL to improved the length filter using the position information. Deng et al. $\left\lbrack  {9,{32}}\right\rbrack$ proposed a partition-based method. Arasu et al. [2] developed a partition-and-enumeration method for the set similarity join with Jaccard and Hamming constraints. All of them use the filter-and-refine framework [9]. Melnik et al. [19] proposed partition-based algorithms for set containment join. Note our work is different from the set containment join works $\left\lbrack  {{24},{34}}\right\rbrack$ as they aim to find set pairs with the containment relationship. Li et al. [14] proposed a partition-based method for string similarity join with the edit distance constraint. Deng et al. [8] proposed a pivotal prefix filter for string similarity search. $\left\lbrack  {{12},{18}}\right\rbrack$ conducted experimental evaluations on the similarity join problem. We discuss more details about the relationship between the set similarity join with overlap constraint and the other constraints and experimentally compare SizeAware with them in Appendixes D and E.

带其他约束的相似性连接与搜索。带其他约束（如杰卡德（Jaccard）、余弦（Cosine）、汉明（Hamming）、编辑距离（Edit Distance）和包含关系）的相似性连接与搜索已得到广泛研究 $\left\lbrack  {2,3,7 - 9,{15},{19},{31},{33}}\right\rbrack$。肖等人 [33] 提出了 PPJoin 和 PPJoin+ 用于带杰卡德、余弦和骰子（Dice）约束的集合相似性连接，它们通过考虑元素位置改进了前缀过滤。布罗斯等人 [4] 设计了 GroupJoin 来对相同前缀进行分组以共享计算。王等人 [31] 开发了 SKJ，它可以跳过对部分倒排列表的扫描。曼等人 [17] 提出了 PEL，利用位置信息改进了长度过滤。邓等人 $\left\lbrack  {9,{32}}\right\rbrack$ 提出了一种基于划分的方法。阿拉苏等人 [2] 开发了一种用于带杰卡德和汉明约束的集合相似性连接的划分 - 枚举方法。所有这些方法都使用了过滤 - 细化框架 [9]。梅尔尼克等人 [19] 提出了用于集合包含连接的基于划分的算法。注意，我们的工作与集合包含连接工作 $\left\lbrack  {{24},{34}}\right\rbrack$ 不同，因为它们的目标是找到具有包含关系的集合对。李等人 [14] 提出了一种用于带编辑距离约束的字符串相似性连接的基于划分的方法。邓等人 [8] 提出了一种用于字符串相似性搜索的关键前缀过滤方法。$\left\lbrack  {{12},{18}}\right\rbrack$ 对相似性连接问题进行了实验评估。我们在附录 D 和 E 中更详细地讨论了带重叠约束的集合相似性连接与其他约束之间的关系，并通过实验将 SizeAware 与它们进行了比较。

Approximate Similarity Join and Search Algorithms. There is a rich literature $\left\lbrack  {1,{21},{22},{26} - {28}}\right\rbrack$ on approximate algorithms for set similarity join and search. Most of them are related to locality sensitive hashing (LSH) $\left\lbrack  {{10},{11}}\right\rbrack$ . The idea behind LSH is to partition the input sets into buckets such that the more similar two sets are, the higher probability they are hashed to the same bucket. Pagh [22] proposed the LSH for hamming distance without false negatives. The traditional LSH cannot support the non-metric space distance function. To address this issue, Shrivastava et al. [27] proposed an asymmetric LSH which pre-processes the vectors by asymmetric transformation to make them fit in the classic LSH technique. However, it is non-trivial to extend these techniques to support the threshold-based overlap set similarity join query, because the overlap between two similar sets can be vanishingly small compared to the size of the sets and the tricks like picking a random element and expecting it to be in both sets do not work.

近似相似性连接与搜索算法。关于集合相似性连接与搜索的近似算法有大量文献 $\left\lbrack  {1,{21},{22},{26} - {28}}\right\rbrack$。其中大多数与局部敏感哈希（Locality Sensitive Hashing，LSH）$\left\lbrack  {{10},{11}}\right\rbrack$ 相关。LSH 的基本思想是将输入集合划分为多个桶，使得两个集合越相似，它们被哈希到同一个桶的概率就越高。帕格 [22] 提出了用于汉明距离且无假阴性的 LSH。传统的 LSH 无法支持非度量空间距离函数。为了解决这个问题，什里瓦斯塔瓦等人 [27] 提出了一种非对称 LSH，它通过非对称变换对向量进行预处理，使其适用于经典的 LSH 技术。然而，将这些技术扩展以支持基于阈值的重叠集合相似性连接查询并非易事，因为与集合的大小相比，两个相似集合之间的重叠可能非常小，而且像随机选择一个元素并期望它同时存在于两个集合中的技巧并不适用。

## 8 CONCLUSION

## 8 结论

In this paper, we study the set similarity join problem with overlap constraints. We propose a size-aware algorithm with the time complexity of $O\left( {{n}^{2 - \frac{1}{c}}{k}^{\frac{1}{2c}}}\right)$ where $n$ is the total size of all the sets and $k$ is the number of results. We divide all the sets into small sets and large sets and process them separately. For the small sets, we enumerate all their $c$ -subsets and take any set pair sharing at least one $c$ -subset as a result. To avoid enumerating unnecessary $c$ -subsets, we develop a heap-based method to avoid the unique $c$ -subsets that cannot generate any result and the redundant $c$ -subsets that only generate duplicate results. We propose to block the $c$ -subsets to reduce the heap size and the heap-adjusting cost. We design an effective method to select an appropriate size boundary. Experimental results show that our algorithm outperforms state-of-the-art studies by up to an order of magnitude.

在本文中，我们研究了带重叠约束的集合相似性连接问题。我们提出了一种时间复杂度为 $O\left( {{n}^{2 - \frac{1}{c}}{k}^{\frac{1}{2c}}}\right)$ 的大小感知算法，其中 $n$ 是所有集合的总大小，$k$ 是结果的数量。我们将所有集合分为小集合和大集合，并分别对它们进行处理。对于小集合，我们枚举它们所有的 $c$ - 子集，并将共享至少一个 $c$ - 子集的任意集合对作为结果。为了避免枚举不必要的 $c$ - 子集，我们开发了一种基于堆的方法，以避免那些无法产生任何结果的唯一 $c$ - 子集和仅产生重复结果的冗余 $c$ - 子集。我们提议对 $c$ - 子集进行分块，以减小堆的大小和堆调整成本。我们设计了一种有效的方法来选择合适的大小边界。实验结果表明，我们的算法比现有最先进的研究成果性能高出一个数量级。

Acknowledgment: The research of Yufei Tao was partially supported by a direct grant (Project Number: 4055079) from CUHK and by a Faculty Research Award from Google. Guoliang Li was supported by the 973 Program of China (2015CB358700), NSF of China (61632016,61472198,61521002,61661166012), and TAL education.

致谢：陶宇飞的研究部分得到了香港中文大学的直接资助（项目编号：4055079）和谷歌的教师研究奖的支持。李国良得到了中国 973 计划（2015CB358700）、国家自然科学基金（61632016、61472198、61521002、61661166012）和好未来教育的支持。

## REFERENCES

## 参考文献

[1] T. D. Ahle, R. Pagh, I. P. Razenshteyn, and F. Silvestri. On the complexity of inner product similarity join. In PODS, pages 151-164, 2016.

[2] A. Arasu, V. Ganti, and R. Kaushik. Efficient exact set-similarity joins. In VLDB, pages 918-929, 2006.

[3] R. J. Bayardo, Y. Ma, and R. Srikant. Scaling up all pairs similarity search. In ${WWW}$ ,pages 131-140,2007.

[4] P. Bouros, S. Ge, and N. Mamoulis. Spatio-textual similarity joins. PVLDB, 6(1):1-12,2012.

[5] A. Z. Broder, S. C. Glassman, M. S. Manasse, and G. Zweig. Syntactic clustering of the web. Computer Networks, 29(8-13):1157-1166, 1997.

[6] J. A. Bullinaria and J. P. Levy. Extracting semantic representations from word co-occurrence statistics: A computational study. Behavior Research Methods, 39(3):510-526, Aug 2007.

[7] S. Chaudhuri, V. Ganti, and R. Kaushik. A primitive operator for similarity joins in data cleaning. In ICDE, pages 5-16, 2006.

[8] D. Deng, G. Li, and J. Feng. A pivotal prefix based filtering algorithm for string similarity search. In SIGMOD, pages 673-684, 2014.

[9] D. Deng, G. Li, H. Wen, and J. Feng. An efficient partition based method for exact set similarity joins. PVLDB, 9(4):360-371, 2015.

[10] A. Gionis, P. Indyk, and R. Motwani. Similarity search in high dimensions via hashing. In VLDB, pages 518-529, 1999.

[11] S. Har-Peled, P. Indyk, and R. Motwani. Approximate nearest neighbor: Towards removing the curse of dimensionality. Theory of Computing, 8(1):321-350, 2012.

[12] Y. Jiang, G. Li, J. Feng, and W.-S. Li. String similarity joins: An experimental evaluation. PVLDB, 7(8):625-636, 2014.

[13] C. Li, J. Lu, and Y. Lu. Efficient merging and filtering algorithms for approximate string searches. In ICDE, pages 257-266, 2008.

[14] G. Li, D. Deng, and J. Feng. A partition-based method for string similarity joins with edit-distance constraints. ACM Trans. Database Syst., 38(2):9, 2013.

[15] G. Li, D. Deng, J. Wang, and J. Feng. Pass-join: A partition-based method for similarity joins. PVLDB, 5(3):253-264, 2011.

[16] K. Lund and C. Burgess. Producing high-dimensional semantic spaces from lexical co-occurrence. Behavior Research Methods, Instruments, & Computers, 28(2):203-208, Jun 1996.

[17] W. Mann and N. Augsten. PEL: position-enhanced length filter for set similarity joins. In ${GVD}$ ,pages ${89} - {94},{2014}$ .

[18] W. Mann, N. Augsten, and P. Bouros. An empirical evaluation of set similarity join techniques. PVLDB, 9(9):636-647, 2016.

[19] S. Melnik and H. Garcia-Molina. Adaptive algorithms for set containment joins. ACM Trans. Database Syst., 28:56-99, 2003.

[20] T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, and J. Dean. Distributed representations of words and phrases and their compositionality. In NIPS, pages 3111-3119, 2013.

[21] B. Neyshabur and N. Srebro. On symmetric and asymmetric lshs for inner product search. In ICML, pages 1926-1934, 2015.

[22] R. Pagh. Locality-sensitive hashing without false negatives. In SODA, pages 1-9, 2016.

[23] J. Pennington, R. Socher, and C. D. Manning. Glove: Global vectors for word representation. In ${EMNLP}$ ,pages 1532-1543,2014.

[24] K. Ramasamy, J. M. Patel, J. F. Naughton, and R. Kaushik. Set containment joins: The good, the bad and the ugly. In VLDB, pages 351-362, 2000.

[25] S. Sarawagi and A. Kirpal. Efficient set joins on similarity predicates. In SIGMOD, pages 743-754, 2004.

[26] V. Satuluri and S. Parthasarathy. Bayesian locality sensitive hashing for fast similarity search. PVLDB, 5(5):430-441, 2012.

[27] A. Shrivastava and P. Li. Asymmetric LSH (ALSH) for sublinear time maximum inner product search (MIPS). In NIPS, pages 2321-2329, 2014.

[28] A. Shrivastava and P. Li. Asymmetric minwise hashing for indexing binary inner products and set containment. In ${WWW}$ ,pages 981-991,2015.

[29] C. Teflioudi, R. Gemulla, and O. Mykytiuk. LEMP: fast retrieval of large entries in a matrix product. In SIGMOD, pages 107-122, 2015.

[30] J. Wang, G. Li, and J. Feng. Can we beat the prefix filtering?: an adaptive framework for similarity join and search. In SIGMOD, pages 85-96, 2012.

[31] X. Wang, L. Qin, X. Lin, Y. Zhang, and L. Chang. Leveraging set relations in exact set similarity join. PVLDB, 10(9):925-936, 2017.

[32] C. Xiao, W. Wang, X. Lin, and H. Shang. Top-k set similarity joins. In ICDE, pages 916-927, 2009.

[33] C. Xiao, W. Wang, X. Lin, J. X. Yu, and G. Wang. Efficient similarity joins for near-duplicate detection. ACM Trans. Database Syst., 36(3):15, 2011.

[34] J. Yang, W. Zhang, S. Yang, Y. Zhang, and X. Lin. Tt-join: Efficient set containment join. In ICDE, pages 509-520, 2017.

[35] Y. Zhang, G. Lai, M. Zhang, Y. Zhang, Y. Liu, and S. Ma. Explicit factor models for explainable recommendation based on phrase-level sentiment analysis. In SIGIR, pages 83-92, 2014.

## A BINARY SEARCHING FOR C-SUBSETS

## A C - 子集的二分查找

Given the ${r}_{c}^{min}$ from $R$ and the ${r}_{c}^{top}$ ,when ${r}_{c}^{min} \neq  {r}_{c}^{top}$ we require to find the smallest $c$ -subset in $R$ that is not smaller than ${r}_{c}^{top}$ . We can achieve this by binary searching $R$ . Suppose that $R = \left\{  {{e}_{1}^{\prime },{e}_{2}^{\prime },\cdots ,{e}_{\left| R\right| }^{\prime }}\right\}$ and ${\mathrm{r}}_{c}^{\text{top }} = {e}_{1}{e}_{2}\cdots {e}_{c}$ where ${e}_{i} < {e}_{j}$ and ${e}_{i}^{\prime } < {e}_{j}^{\prime }$ for any $i < j$ . For each element ${e}_{i}$ in increasing order, we binary search the smallest element ${e}_{{a}_{i}}^{\prime }$ in $R$ that is not smaller than ${e}_{i}$ until we first meet ${e}_{i} \neq  {e}_{{a}_{i}}^{\prime }$ . Then we reinsert the $c$ -subset ${e}_{{a}_{1}}^{\prime }{e}_{{a}_{2}}^{\prime }\cdots {e}_{{a}_{i}}^{\prime }{e}_{{a}_{i} + 1}^{\prime }\cdots {e}_{{a}_{i} + c - i}^{\prime }$ into the heap where ${e}_{{a}_{1}}^{\prime } = {e}_{1},{e}_{{a}_{2}}^{\prime } =$ ${e}_{2},\cdots ,{e}_{{a}_{i - 1}}^{\prime } = {e}_{i - 1}$ and ${e}_{{a}_{i}}^{\prime } \neq  {e}_{i}$ and ${e}_{{a}_{i} + 1}^{\prime }\cdots {e}_{{a}_{i} + c - 1}^{\prime }$ are the elements right after ${e}_{{a}_{i}}^{\prime }$ in $R$ .

给定来自$R$的${r}_{c}^{min}$和${r}_{c}^{top}$，当${r}_{c}^{min} \neq  {r}_{c}^{top}$时，我们需要在$R$中找到不小于${r}_{c}^{top}$的最小$c$ - 子集。我们可以通过对$R$进行二分查找来实现这一点。假设$R = \left\{  {{e}_{1}^{\prime },{e}_{2}^{\prime },\cdots ,{e}_{\left| R\right| }^{\prime }}\right\}$且${\mathrm{r}}_{c}^{\text{top }} = {e}_{1}{e}_{2}\cdots {e}_{c}$，其中对于任意$i < j$有${e}_{i} < {e}_{j}$和${e}_{i}^{\prime } < {e}_{j}^{\prime }$。对于按升序排列的每个元素${e}_{i}$，我们在$R$中二分查找不小于${e}_{i}$的最小元素${e}_{{a}_{i}}^{\prime }$，直到首次遇到${e}_{i} \neq  {e}_{{a}_{i}}^{\prime }$。然后我们将$c$ - 子集${e}_{{a}_{1}}^{\prime }{e}_{{a}_{2}}^{\prime }\cdots {e}_{{a}_{i}}^{\prime }{e}_{{a}_{i} + 1}^{\prime }\cdots {e}_{{a}_{i} + c - i}^{\prime }$重新插入堆中，其中${e}_{{a}_{1}}^{\prime } = {e}_{1},{e}_{{a}_{2}}^{\prime } =$ ${e}_{2},\cdots ,{e}_{{a}_{i - 1}}^{\prime } = {e}_{i - 1}$，并且${e}_{{a}_{i}}^{\prime } \neq  {e}_{i}$和${e}_{{a}_{i} + 1}^{\prime }\cdots {e}_{{a}_{i} + c - 1}^{\prime }$是$R$中${e}_{{a}_{i}}^{\prime }$之后的元素。

## B ADAPTATION FOR R-S JOIN

## B 对R - S连接的适配

In this section, we extend our theoretical results and techniques to the $\mathcal{R} - \mathcal{S}$ join case (where $\mathcal{R} \neq  \mathcal{S}$ ).

在本节中，我们将我们的理论结果和技术扩展到$\mathcal{R} - \mathcal{S}$连接的情况（其中$\mathcal{R} \neq  \mathcal{S}$）。

The size-aware algorithm for the $\mathcal{R} - \mathcal{S}$ join. Given two collections of sets $\mathcal{R}$ and $\mathcal{S}$ and a threshold $c$ ,the size-aware algorithm divides all the sets into large sets ${\mathcal{R}}_{l}$ and ${\mathcal{S}}_{l}$ and small sets ${\mathcal{R}}_{s}$ and ${\mathcal{S}}_{s}$ with the size boundary $x$ and processes them separately. For each large set $R \in  {\mathcal{R}}_{l}$ (or $S \in  {\mathcal{S}}_{l}$ ),it compares $R$ (or $S$ ) with every set in $\mathcal{S}$ (or $\mathcal{R}$ ). As there are totally at most $\frac{n}{x}$ large sets where $n$ is the total size of all the sets, the time complexity of processing the large sets is $O\left( \frac{{n}^{2}}{x}\right)$ . For each small set,the size-aware algorithm enumerates all its $c$ -subsets. As the size of a small set is no larger than $x$ and the number of $c$ -subsets for a small set $R \in  {\mathcal{R}}_{s}$ (or $\left. {S \in  {\mathcal{S}}_{s}}\right)$ is within ${\left| R\right| }^{c}$ (or ${\left| S\right| }^{c}$ ),the number of all $c$ -subsets cannot exceed ${x}^{c - 1}n$ and the time complexity of enumerating $c$ -subsets is $O\left( {{x}^{c - 1}n}\right)$ . Next it generates all the results from the $c$ -subset inverted index. Suppose that the $c$ -subset inverted lists generated by ${\mathcal{R}}_{s}$ and ${\mathcal{S}}_{s}$ are respectively ${\mathcal{L}}_{1},{\mathcal{L}}_{2},\ldots {\mathcal{L}}_{l}$ and ${\mathcal{L}}_{1}^{\prime },{\mathcal{L}}_{2}^{\prime }\ldots {\mathcal{L}}_{l}^{\prime }$ where ${\mathcal{L}}_{i}$ and ${\mathcal{L}}_{i}^{\prime }$ associated with the same $c$ -subset. The time complexity of generating the results is

$\mathcal{R} - \mathcal{S}$连接的大小感知算法。给定两个集合集合$\mathcal{R}$和$\mathcal{S}$以及一个阈值$c$，大小感知算法将所有集合划分为大集合${\mathcal{R}}_{l}$和${\mathcal{S}}_{l}$以及小集合${\mathcal{R}}_{s}$和${\mathcal{S}}_{s}$，其大小边界为$x$，并分别对它们进行处理。对于每个大集合$R \in  {\mathcal{R}}_{l}$（或$S \in  {\mathcal{S}}_{l}$），它将$R$（或$S$）与$\mathcal{S}$（或$\mathcal{R}$）中的每个集合进行比较。由于总共最多有$\frac{n}{x}$个大集合，其中$n$是所有集合的总大小，因此处理大集合的时间复杂度为$O\left( \frac{{n}^{2}}{x}\right)$。对于每个小集合，大小感知算法枚举其所有的$c$ - 子集。由于小集合的大小不大于$x$，并且小集合$R \in  {\mathcal{R}}_{s}$（或$\left. {S \in  {\mathcal{S}}_{s}}\right)$）的$c$ - 子集的数量在${\left| R\right| }^{c}$（或${\left| S\right| }^{c}$）范围内，所有$c$ - 子集的数量不会超过${x}^{c - 1}n$，枚举$c$ - 子集的时间复杂度为$O\left( {{x}^{c - 1}n}\right)$。接下来，它从$c$ - 子集倒排索引中生成所有结果。假设由${\mathcal{R}}_{s}$和${\mathcal{S}}_{s}$生成的$c$ - 子集倒排列表分别为${\mathcal{L}}_{1},{\mathcal{L}}_{2},\ldots {\mathcal{L}}_{l}$和${\mathcal{L}}_{1}^{\prime },{\mathcal{L}}_{2}^{\prime }\ldots {\mathcal{L}}_{l}^{\prime }$，其中${\mathcal{L}}_{i}$和${\mathcal{L}}_{i}^{\prime }$与同一个$c$ - 子集相关联。生成结果的时间复杂度为

$$
O\left( {\mathop{\sum }\limits_{{i = 1}}^{l}\left| {\mathcal{L}}_{i}\right|  \times  \left| {\mathcal{L}}_{i}^{\prime }\right| }\right) 
$$

As the number of results generated from any inverted list cannot exceed $k$ ,we have

由于从任何倒排列表生成的结果数量不会超过$k$，我们有

$$
\min \left( {\frac{\left| {\mathcal{L}}_{i}\right| \left( {\left| {\mathcal{L}}_{i}\right|  - 1}\right) }{2},\frac{\left| {\mathcal{L}}_{i}^{\prime }\right| \left( {\left| {\mathcal{L}}_{i}^{\prime }\right|  - 1}\right) }{2}}\right)  \leq  k.
$$

It thus follows that $\min \left( {\left| {\mathcal{L}}_{i}\right| ,\left| {\mathcal{L}}_{i}^{\prime }\right| }\right)  = O\left( \sqrt{k}\right)$ . Moreover,as the total size of all inverted lists,which is exactly the number of all $c$ -subsets, cannot be larger than ${x}^{c - 1}n$ ,we have

因此可得$\min \left( {\left| {\mathcal{L}}_{i}\right| ,\left| {\mathcal{L}}_{i}^{\prime }\right| }\right)  = O\left( \sqrt{k}\right)$。此外，由于所有倒排列表的总大小（恰好是所有$c$ - 子集的数量）不会大于${x}^{c - 1}n$，我们有

$$
\mathop{\sum }\limits_{{i = 1}}^{l}\max \left( {\left| {\mathcal{L}}_{i}\right| ,\left| {\mathcal{L}}_{i}^{\prime }\right| }\right)  \leq  \mathop{\sum }\limits_{{i = 1}}^{l}\left( {\left| {\mathcal{L}}_{i}\right|  + \left| {\mathcal{L}}_{i}^{\prime }\right| }\right)  \leq  {x}^{c - 1}n.
$$

Thus the time complexity of generating the results is

因此，生成结果的时间复杂度为

$$
O\left( {\mathop{\sum }\limits_{{i = 1}}^{l}\left| {\mathcal{L}}_{i}\right|  \times  \left| {\mathcal{L}}_{i}^{\prime }\right| }\right)  = O\left( {\mathop{\sum }\limits_{{i = 1}}^{l}\min \left( {\left| {\mathcal{L}}_{i}\right| ,\left| {\mathcal{L}}_{i}^{\prime }\right| }\right)  \times  \max \left( {\left| {\mathcal{L}}_{i}\right| ,\left| {\mathcal{L}}_{i}^{\prime }\right| }\right) }\right) 
$$

$$
 = O\left( {\sqrt{k}\mathop{\sum }\limits_{{i = 1}}^{l}\max \left( {\left| {\mathcal{L}}_{i}\right| ,\left| {\mathcal{L}}_{i}^{\prime }\right| }\right) }\right)  = O\left( {{x}^{c - 1}n\sqrt{k}}\right) .
$$

<!-- Media -->

Algorithm 6: SIZEAWARERSJOIN

算法6：大小感知RS连接（SIZEAWARERSJOIN）

---

Input: $\mathcal{R}$ : a dataset; $\mathcal{S}$ : another dataset;

输入：$\mathcal{R}$：一个数据集；$\mathcal{S}$：另一个数据集；

$c$ : a threshold.

$c$：一个阈值。

Output: $\mathcal{A} = \{ \langle R,S\rangle  \mid  \left| {R \cap  S}\right|  \geq  c,R \in  \mathcal{R},S \in  \mathcal{S}\}$ .

输出：$\mathcal{A} = \{ \langle R,S\rangle  \mid  \left| {R \cap  S}\right|  \geq  c,R \in  \mathcal{R},S \in  \mathcal{S}\}$。

$x =$ GetSizeBoundary(R,S,c);

$x =$获取大小边界（GetSizeBoundary）(R,S,c)；

divide $\mathcal{R}$ and $\mathcal{S}$ into small sets ${\mathcal{R}}_{s}$ and ${\mathcal{S}}_{s}$ and large sets ${\mathcal{R}}_{l}$

根据大小边界$\mathcal{S}$将$\mathcal{R}$和$\mathcal{S}$划分为小集合${\mathcal{R}}_{s}$和${\mathcal{S}}_{s}$以及大集合${\mathcal{R}}_{l}$

and ${\mathcal{S}}_{l}$ by the size boundary $x$ ;

和${\mathcal{S}}_{l}$；

5 Using ScanCount to find all the similar set pairs in ${\mathcal{R}}_{l} \times  \mathcal{S}$ and

5 使用扫描计数（ScanCount）方法在${\mathcal{R}}_{l} \times  \mathcal{S}$中查找所有相似的集合对以及

${\mathcal{S}}_{l} \times  \mathcal{R}$ and add them into $\mathcal{A}$ ;

${\mathcal{S}}_{l} \times  \mathcal{R}$ 并将它们添加到 $\mathcal{A}$ 中；

$\left\langle  {{\mathcal{L}}_{\text{slim }},{\mathcal{L}}_{\text{slim }}^{\prime }}\right\rangle   = \operatorname{BlockDedup}\left( {{\mathcal{R}}_{s},{\mathcal{S}}_{s},c}\right)$ ;

foreach ${r}_{c}$ s.t. ${\mathcal{L}}_{\text{slim }}\left\lbrack  {r}_{c}\right\rbrack   \neq  \phi \& {\mathcal{L}}_{\text{slim }}^{\prime }\left\lbrack  {r}_{c}\right\rbrack   \neq  \phi$ do

对于满足 ${\mathcal{L}}_{\text{slim }}\left\lbrack  {r}_{c}\right\rbrack   \neq  \phi \& {\mathcal{L}}_{\text{slim }}^{\prime }\left\lbrack  {r}_{c}\right\rbrack   \neq  \phi$ 的每个 ${r}_{c}$ 执行

	add every set pair in ${\mathcal{L}}_{\text{slim }}\left\lbrack  {r}_{c}\right\rbrack   \times  {\mathcal{L}}_{\text{slim }}^{\prime }\left\lbrack  {r}_{c}\right\rbrack$ into $\mathcal{A}$ ;

	将 ${\mathcal{L}}_{\text{slim }}\left\lbrack  {r}_{c}\right\rbrack   \times  {\mathcal{L}}_{\text{slim }}^{\prime }\left\lbrack  {r}_{c}\right\rbrack$ 中的每个集合对添加到 $\mathcal{A}$ 中；

return $\mathcal{A}$ ;

返回 $\mathcal{A}$ ；

---

Algorithm 7: HeapDedup $\left( {{\mathcal{R}}_{s},{\mathcal{S}}_{s},\mathrm{c}}\right)$

算法 7：堆去重 $\left( {{\mathcal{R}}_{s},{\mathcal{S}}_{s},\mathrm{c}}\right)$

---

Input: ${\mathcal{R}}_{s}$ : a collection of small sets; $c$ : a threshold;

输入：${\mathcal{R}}_{s}$ ：小集合的集合；$c$ ：阈值；

${\mathcal{S}}_{s}$ : another collection of small sets.

${\mathcal{S}}_{s}$ ：另一个小集合的集合。

Output: $\left\langle  {{\mathcal{L}}_{\text{slim }},{\mathcal{L}}_{\text{slim }}^{\prime }}\right\rangle$ : slimmed inverted indexes.

输出：$\left\langle  {{\mathcal{L}}_{\text{slim }},{\mathcal{L}}_{\text{slim }}^{\prime }}\right\rangle$ ：精简后的倒排索引。

Insert all the min-subsets in ${\mathcal{R}}_{s}$ and ${\mathcal{S}}_{s}$ to heaps $\mathcal{H}$ and ${\mathcal{H}}^{\prime }$ ;

将 ${\mathcal{R}}_{s}$ 和 ${\mathcal{S}}_{s}$ 中的所有最小子集插入到堆 $\mathcal{H}$ 和 ${\mathcal{H}}^{\prime }$ 中；

- Pop $\mathcal{H}$ and ${\mathcal{H}}^{\prime }$ to get the smallest $c$ -subsets ${\mathrm{r}}_{c}^{\min }$ and ${\mathrm{s}}_{c}^{\min }$ ;

- 弹出 $\mathcal{H}$ 和 ${\mathcal{H}}^{\prime }$ 以获取最小的 $c$ -子集 ${\mathrm{r}}_{c}^{\min }$ 和 ${\mathrm{s}}_{c}^{\min }$ ；

while neither $\mathcal{H}$ nor ${\mathcal{H}}^{\prime }$ is empty do

当 $\mathcal{H}$ 和 ${\mathcal{H}}^{\prime }$ 都不为空时执行

	Suppose ${r}_{c}^{\min }$ and ${s}_{c}^{\min }$ are from $R$ and $S$ respectively;

	假设 ${r}_{c}^{\min }$ 和 ${s}_{c}^{\min }$ 分别来自 $R$ 和 $S$ ；

	if ${r}_{c}^{\min } > {s}_{c}^{\min }$ then

	如果 ${r}_{c}^{\min } > {s}_{c}^{\min }$ 则

		append $S$ to ${\mathcal{L}}_{\text{slim }}^{\prime }\left\lbrack  {\mathrm{s}}_{c}^{\min }\right\rbrack$ ,binary search $S$ for the first

		将 $S$ 追加到 ${\mathcal{L}}_{\text{slim }}^{\prime }\left\lbrack  {\mathrm{s}}_{c}^{\min }\right\rbrack$ ，对 $S$ 进行二分查找以找到第一个

		$c$ -subset that is no smaller than ${r}_{c}^{\min }$ ,reinsert it into

		不小于 ${r}_{c}^{\min }$ 的 $c$ -子集，将其重新插入到

		${\mathcal{H}}^{\prime }$ ,and pop ${\mathcal{H}}^{\prime }$ to get the next ${\mathbf{s}}_{c}^{\min }$ ;

		${\mathcal{H}}^{\prime }$ 中，并弹出 ${\mathcal{H}}^{\prime }$ 以获取下一个 ${\mathbf{s}}_{c}^{\min }$ ；

	else if ${r}_{c}^{\min } < {s}_{c}^{\min }$ then

	否则，如果 ${r}_{c}^{\min } < {s}_{c}^{\min }$ 成立，则

		append $R$ to ${\mathcal{L}}_{\text{slim }}\left\lbrack  {r}_{c}^{\min }\right\rbrack$ ,binary search $R$ for the first

		将 $R$ 追加到 ${\mathcal{L}}_{\text{slim }}\left\lbrack  {r}_{c}^{\min }\right\rbrack$ 中，对 $R$ 进行二分查找以找到第一个

		$c$ -subset that is no smaller than ${\mathrm{s}}_{c}^{\min }$ ,reinsert it into

		不小于 ${\mathrm{s}}_{c}^{\min }$ 的 $c$ -子集，将其重新插入到

		$\mathcal{H}$ ,and pop $\mathcal{H}$ to get the next ${r}_{c}^{\min }$ ;

		$\mathcal{H}$ 中，并弹出 $\mathcal{H}$ 以获取下一个 ${r}_{c}^{\min }$；

	else if ${r}_{c}^{\min } = {s}_{c}^{\min }$ then

	否则，如果 ${r}_{c}^{\min } = {s}_{c}^{\min }$ 成立，则

		while ${r}_{c}^{\min } \neq  {r}_{c}^{top}$ do

		当 ${r}_{c}^{\min } \neq  {r}_{c}^{top}$ 条件满足时执行

			append $R$ to ${\mathcal{L}}_{\text{slim }}\left\lbrack  {r}_{c}^{\min }\right\rbrack$ ,pop $\mathcal{H}$ to get next ${r}_{c}^{\min }$ ;

			将 $R$ 追加到 ${\mathcal{L}}_{\text{slim }}\left\lbrack  {r}_{c}^{\min }\right\rbrack$ 中，弹出 $\mathcal{H}$ 以获取下一个 ${r}_{c}^{\min }$；

		while ${\mathrm{s}}_{c}^{\min } \neq  {\mathrm{s}}_{c}^{\text{top }}$ do

		当 ${\mathrm{s}}_{c}^{\min } \neq  {\mathrm{s}}_{c}^{\text{top }}$ 条件满足时执行

			append $S$ to ${\mathcal{L}}_{\text{slim }}^{\prime }\left\lbrack  {\mathrm{s}}_{c}^{\min }\right\rbrack$ ,pop ${\mathcal{H}}^{\prime }$ to get next ${\mathrm{s}}_{c}^{\min }$ ;

			将 $S$ 追加到 ${\mathcal{L}}_{\text{slim }}^{\prime }\left\lbrack  {\mathrm{s}}_{c}^{\min }\right\rbrack$ 中，弹出 ${\mathcal{H}}^{\prime }$ 以获取下一个 ${\mathrm{s}}_{c}^{\min }$；

return $\left\langle  {{\mathcal{L}}_{\text{slim }},{\mathcal{L}}_{\text{slim }}^{\prime }}\right\rangle$ ;

返回 $\left\langle  {{\mathcal{L}}_{\text{slim }},{\mathcal{L}}_{\text{slim }}^{\prime }}\right\rangle$；

---

<!-- Media -->

The rest is the same as the self-join case and the time complexity of the size-aware algorithm is still

其余部分与自连接情况相同，且大小感知算法的时间复杂度仍然是

$$
O\left( {{n}^{2 - \frac{1}{c}}{k}^{\frac{1}{2c}}}\right)  = o\left( {n}^{2}\right)  + O\left( k\right) .
$$

The pseudo codes of the size-aware algorithm for the $\mathcal{R} - \mathcal{S}$ join case is shown in Algorithm 6. It first utilizes the size boundary selection method to choose a size boundary $x$ (Line 1). Then it divides the two datasets using $x$ (Line 2). It uses ScanCount to find the results in ${\mathcal{R}}_{l} \times  \mathcal{S}$ and ${\mathcal{S}}_{l} \times  \mathcal{R}$ and add them to the result set $\mathcal{A}$ (Lines 3). Next it utilizes the BlockDedup method to generate the slimmed inverted indexes ${\mathcal{L}}_{\text{slim }}$ and ${\mathcal{L}}_{\text{slim }}^{\prime }$ for ${\mathcal{R}}_{s}$ and ${\mathcal{S}}_{s}$ respectively (Line 4). Finally,for each $c$ -subset ${r}_{c}$ ,it adds every pair in ${\mathcal{L}}_{\text{slim }}\left\lbrack  {r}_{c}\right\rbrack   \times  {\mathcal{L}}_{\text{slim }}^{\prime }\left\lbrack  {r}_{c}\right\rbrack$ to $\mathcal{A}$ and returns $\mathcal{A}$ (Lines 5 to 7).

用于 $\mathcal{R} - \mathcal{S}$ 连接情况的大小感知算法的伪代码如算法 6 所示。它首先利用大小边界选择方法选择一个大小边界 $x$（第 1 行）。然后使用 $x$ 划分两个数据集（第 2 行）。它使用 ScanCount 在 ${\mathcal{R}}_{l} \times  \mathcal{S}$ 和 ${\mathcal{S}}_{l} \times  \mathcal{R}$ 中查找结果并将其添加到结果集 $\mathcal{A}$ 中（第 3 行）。接下来，它利用 BlockDedup 方法分别为 ${\mathcal{R}}_{s}$ 和 ${\mathcal{S}}_{s}$ 生成精简的倒排索引 ${\mathcal{L}}_{\text{slim }}$ 和 ${\mathcal{L}}_{\text{slim }}^{\prime }$（第 4 行）。最后，对于每个 $c$ -子集 ${r}_{c}$，它将 ${\mathcal{L}}_{\text{slim }}\left\lbrack  {r}_{c}\right\rbrack   \times  {\mathcal{L}}_{\text{slim }}^{\prime }\left\lbrack  {r}_{c}\right\rbrack$ 中的每一对添加到 $\mathcal{A}$ 中并返回 $\mathcal{A}$（第 5 行到第 7 行）。

The HeapSkip method for the $\mathcal{R} - \mathcal{S}$ join. Suppose $\mathcal{L}$ and ${\mathcal{L}}^{\prime }$ are the $c$ -subset inverted indexes constructed from ${\mathcal{R}}_{s}$ and ${\mathcal{S}}_{s}$ . For any $c$ -subset ${r}_{c}$ ,if $\mathcal{L}\left\lbrack  {r}_{c}\right\rbrack   = \phi$ or ${\mathcal{L}}^{\prime }\left\lbrack  {r}_{c}\right\rbrack   = \phi$ ,it cannot generate any result and we call it a unique $c$ -subset. To skip the unique $c$ -subsets, we fix a global order for all the $c$ -subsets and access the $c$ -subsets in each set in order. We build two min-heaps $\mathcal{H}$ and ${\mathcal{H}}^{\prime }$ to maintain the min-subsets of the sets in ${\mathcal{R}}_{s}$ and ${\mathcal{S}}_{s}$ respectively. We pop $\mathcal{H}$ and ${\mathcal{H}}^{\prime }$ and get the smallest min-subsets ${r}_{c}^{\min }$ and ${s}_{c}^{\min }$ in $\mathcal{H}$ and ${\mathcal{H}}^{\prime }$ . Suppose that they come from $R$ and $S$ respectively. We compare ${\mathrm{r}}_{c}^{\min }$ with ${\mathrm{s}}_{c}^{\min }$ . If ${\mathrm{r}}_{c}^{\min } = {\mathrm{s}}_{c}^{\min }$ ,we first append $R$ to $\mathcal{L}\left\lbrack  {\mathrm{r}}_{c}^{\min }\right\rbrack$ and $S$ to ${\mathcal{L}}^{\prime }\left\lbrack  {\mathrm{s}}_{c}^{\min }\right\rbrack$ and then reinsert the next min-subsets in $R$ and $S$ to $\mathcal{H}$ and ${\mathcal{H}}^{\prime }$ respectively. If ${r}_{c}^{\min } > {s}_{c}^{\min }$ ,we first append $S$ to ${\mathcal{L}}^{\prime }\left\lbrack  {s}_{c}^{\min }\right\rbrack$ and then reinsert the smallest $c$ -subset that is no smaller than ${r}_{c}^{\min }$ in $S$ to ${\mathcal{H}}^{\prime }$ by binary searching. Otherwise ${\mathrm{s}}_{c}^{\min } > {\mathrm{r}}_{c}^{\min }$ ,we first append $R$ to $\mathcal{L}\left\lbrack  {\mathrm{r}}_{c}^{\min }\right\rbrack$ and then reinsert the smallest $c$ -subset that is no smaller than ${\mathrm{s}}_{c}^{\min }$ in $R$ to $\mathcal{H}$ by binary searching. We repeat this until $\mathcal{H}$ or ${\mathcal{H}}^{\prime }$ is empty.

用于$\mathcal{R} - \mathcal{S}$连接的堆跳过（HeapSkip）方法。假设$\mathcal{L}$和${\mathcal{L}}^{\prime }$是从${\mathcal{R}}_{s}$和${\mathcal{S}}_{s}$构建的$c$ -子集倒排索引。对于任何$c$ -子集${r}_{c}$，如果$\mathcal{L}\left\lbrack  {r}_{c}\right\rbrack   = \phi$或${\mathcal{L}}^{\prime }\left\lbrack  {r}_{c}\right\rbrack   = \phi$，则它无法生成任何结果，我们称其为唯一的$c$ -子集。为了跳过唯一的$c$ -子集，我们为所有$c$ -子集确定一个全局顺序，并按顺序访问每个集合中的$c$ -子集。我们构建两个最小堆$\mathcal{H}$和${\mathcal{H}}^{\prime }$，分别维护${\mathcal{R}}_{s}$和${\mathcal{S}}_{s}$中集合的最小子集。我们从$\mathcal{H}$和${\mathcal{H}}^{\prime }$中弹出元素，得到$\mathcal{H}$和${\mathcal{H}}^{\prime }$中的最小子集${r}_{c}^{\min }$和${s}_{c}^{\min }$。假设它们分别来自$R$和$S$。我们比较${\mathrm{r}}_{c}^{\min }$和${\mathrm{s}}_{c}^{\min }$。如果${\mathrm{r}}_{c}^{\min } = {\mathrm{s}}_{c}^{\min }$，我们首先将$R$追加到$\mathcal{L}\left\lbrack  {\mathrm{r}}_{c}^{\min }\right\rbrack$，将$S$追加到${\mathcal{L}}^{\prime }\left\lbrack  {\mathrm{s}}_{c}^{\min }\right\rbrack$，然后分别将$R$和$S$中的下一个最小子集重新插入到$\mathcal{H}$和${\mathcal{H}}^{\prime }$中。如果${r}_{c}^{\min } > {s}_{c}^{\min }$，我们首先将$S$追加到${\mathcal{L}}^{\prime }\left\lbrack  {s}_{c}^{\min }\right\rbrack$，然后通过二分查找将$S$中不小于${r}_{c}^{\min }$的最小$c$ -子集重新插入到${\mathcal{H}}^{\prime }$中。否则${\mathrm{s}}_{c}^{\min } > {\mathrm{r}}_{c}^{\min }$，我们首先将$R$追加到$\mathcal{L}\left\lbrack  {\mathrm{r}}_{c}^{\min }\right\rbrack$，然后通过二分查找将$R$中不小于${\mathrm{s}}_{c}^{\min }$的最小$c$ -子集重新插入到$\mathcal{H}$中。我们重复这个过程，直到$\mathcal{H}$或${\mathcal{H}}^{\prime }$为空。

<!-- Media -->

Algorithm 8: BlockDedup $\left( {{\mathcal{R}}_{s},{\mathcal{S}}_{s},\mathrm{c}}\right)$

算法8：块去重（BlockDedup）$\left( {{\mathcal{R}}_{s},{\mathcal{S}}_{s},\mathrm{c}}\right)$

---

Input: ${\mathcal{R}}_{s}$ : a collection of small sets; $c$ : a threshold;

输入：${\mathcal{R}}_{s}$：小集合的集合；$c$：一个阈值；

${\mathcal{S}}_{s}$ : another collection of small sets.

${\mathcal{S}}_{s}$：另一个小集合的集合。

Output: $\left\langle  {{\mathcal{L}}_{\text{slim }},{\mathcal{L}}_{\text{slim }}^{\prime }}\right\rangle$ : slimmed inverted indexes.

输出：$\left\langle  {{\mathcal{L}}_{\text{slim }},{\mathcal{L}}_{\text{slim }}^{\prime }}\right\rangle$：精简后的倒排索引。

Fix a global order for all the elements in ${\mathcal{R}}_{s}$ and ${\mathcal{S}}_{s}$ ;

为${\mathcal{R}}_{s}$和${\mathcal{S}}_{s}$中的所有元素确定一个全局顺序；

Build element inverted indexes $\mathcal{I}$ and ${\mathcal{I}}^{\prime }$ for ${\mathcal{R}}_{s}$ and ${\mathcal{S}}_{s}$ ;

为${\mathcal{R}}_{s}$和${\mathcal{S}}_{s}$构建元素倒排索引$\mathcal{I}$和${\mathcal{I}}^{\prime }$；

foreach element e s.t. $I\left\lbrack  e\right\rbrack   \neq  \phi$ and ${I}^{\prime }\left\lbrack  e\right\rbrack   \neq  \phi$ do

对于满足$I\left\lbrack  e\right\rbrack   \neq  \phi$和${I}^{\prime }\left\lbrack  e\right\rbrack   \neq  \phi$的每个元素e执行

	${\mathcal{R}}_{\text{tmp }} =$ sets in $\mathcal{I}\left\lbrack  e\right\rbrack$ with elements $\leq  e$ removed;

	移除元素$\leq  e$后的$\mathcal{I}\left\lbrack  e\right\rbrack$中的${\mathcal{R}}_{\text{tmp }} =$集合；

	${\mathcal{S}}_{\text{tmp }} =$ sets in ${\mathcal{I}}^{\prime }\left\lbrack  e\right\rbrack$ with elements $\leq  e$ removed;

	移除元素$\leq  e$后的${\mathcal{I}}^{\prime }\left\lbrack  e\right\rbrack$中的${\mathcal{S}}_{\text{tmp }} =$集合；

	$\left\langle  {{\mathcal{L}}_{\text{slim }},{\mathcal{L}}_{\text{slim }}^{\prime }}\right\rangle   = \left\langle  {{\mathcal{L}}_{\text{slim }},{\mathcal{L}}_{\text{slim }}^{\prime }}\right\rangle   \cup  \operatorname{HeapDedup}\left( {{\mathcal{R}}_{\text{tmp }},{\mathcal{S}}_{\text{tmp }}}\right.$ ,

	$c - 1)$ ;

return $\left\langle  {{\mathcal{L}}_{\text{slim }},{\mathcal{L}}_{\text{slim }}^{\prime }}\right\rangle$ ;

返回$\left\langle  {{\mathcal{L}}_{\text{slim }},{\mathcal{L}}_{\text{slim }}^{\prime }}\right\rangle$；

---

<!-- Media -->

The HeapDedup method for the $\mathcal{R}$ - $\mathcal{S}$ join. If the two inverted lists $\mathcal{L}\left\lbrack  {r}_{c}\right\rbrack$ and ${\mathcal{L}}^{\prime }\left\lbrack  {r}_{c}\right\rbrack$ of a $c$ -subset ${r}_{c}$ are sub-lists of those of another $c$ -subset ${r}_{c}^{\prime }$ ,i.e., $\mathcal{L}\left\lbrack  {r}_{c}\right\rbrack   \subseteq  \mathcal{L}\left\lbrack  {r}_{c}^{\prime }\right\rbrack$ and ${\mathcal{L}}^{\prime }\left\lbrack  {r}_{c}\right\rbrack   \subseteq  {\mathcal{L}}^{\prime }\left\lbrack  {r}_{c}^{\prime }\right\rbrack  ,{r}_{c}$ can only generate duplicate results and we call it a redundant $c$ -subset. To skip the adjacent redundant $c$ -subsets,we delay the reinsertion of min-subsets to the heaps when ${r}_{c}^{\min } = {s}_{c}^{\min }$ . More specifically, when ${r}_{c}^{\min } = {\mathrm{s}}_{c}^{\min }$ ,we keep popping $\mathcal{H}\left( {\mathcal{H}}^{\prime }\right)$ until ${r}_{c}^{\min } \neq  {r}_{c}^{top}$ $\left( {{\mathrm{s}}_{c}^{\text{min }} \neq  {\mathrm{s}}_{c}^{\text{top }}}\right)$ where ${\mathrm{r}}_{c}^{\text{top }}\left( {\mathrm{s}}_{c}^{\text{top }}\right)$ is the $c$ -subset currently tops $\mathcal{H}$ $\left( {\mathcal{H}}^{\prime }\right)$ . Then for each set in $\mathcal{L}\left\lbrack  {r}_{c}^{\min }\right\rbrack  \left( {{\mathcal{L}}^{\prime }\left\lbrack  {s}_{c}^{\min }\right\rbrack  }\right)$ ,we reinsert the smallest $c$ -subsets in it that is no smaller than ${\mathrm{s}}_{c}^{\text{top }}\left( {\mathrm{r}}_{c}^{\text{top }}\right)$ to $\mathcal{H}$ $\left( {\mathcal{H}}^{\prime }\right)$ by binary searching. This is because the $c$ -subsets between ${r}_{c}^{min}$ and ${s}_{c}^{top}$ in $\mathcal{R}$ and between ${s}_{c}^{min}$ and ${r}_{c}^{top}$ in $\mathcal{S}$ do not appear in the other sets except those in $\mathcal{L}\left\lbrack  {r}_{c}^{\min }\right\rbrack$ and ${\mathcal{L}}^{\prime }\left\lbrack  {s}_{c}^{\min }\right\rbrack$ and must be redundant $c$ -subsets. The rest is the same as the self-join case.

$\mathcal{R}$ - $\mathcal{S}$连接的堆去重（HeapDedup）方法。如果一个$c$ -子集${r}_{c}$的两个倒排列表$\mathcal{L}\left\lbrack  {r}_{c}\right\rbrack$和${\mathcal{L}}^{\prime }\left\lbrack  {r}_{c}\right\rbrack$是另一个$c$ -子集${r}_{c}^{\prime }$的倒排列表的子列表，即$\mathcal{L}\left\lbrack  {r}_{c}\right\rbrack   \subseteq  \mathcal{L}\left\lbrack  {r}_{c}^{\prime }\right\rbrack$和${\mathcal{L}}^{\prime }\left\lbrack  {r}_{c}\right\rbrack   \subseteq  {\mathcal{L}}^{\prime }\left\lbrack  {r}_{c}^{\prime }\right\rbrack  ,{r}_{c}$只会生成重复的结果，我们称其为冗余$c$ -子集。为了跳过相邻的冗余$c$ -子集，当${r}_{c}^{\min } = {s}_{c}^{\min }$时，我们延迟将最小子集重新插入堆中。更具体地说，当${r}_{c}^{\min } = {\mathrm{s}}_{c}^{\min }$时，我们持续弹出$\mathcal{H}\left( {\mathcal{H}}^{\prime }\right)$，直到${r}_{c}^{\min } \neq  {r}_{c}^{top}$ $\left( {{\mathrm{s}}_{c}^{\text{min }} \neq  {\mathrm{s}}_{c}^{\text{top }}}\right)$，其中${\mathrm{r}}_{c}^{\text{top }}\left( {\mathrm{s}}_{c}^{\text{top }}\right)$是当前位于$\mathcal{H}$ $\left( {\mathcal{H}}^{\prime }\right)$顶部的$c$ -子集。然后，对于$\mathcal{L}\left\lbrack  {r}_{c}^{\min }\right\rbrack  \left( {{\mathcal{L}}^{\prime }\left\lbrack  {s}_{c}^{\min }\right\rbrack  }\right)$中的每个集合，我们通过二分查找将其中不小于${\mathrm{s}}_{c}^{\text{top }}\left( {\mathrm{r}}_{c}^{\text{top }}\right)$的最小$c$ -子集重新插入到$\mathcal{H}$ $\left( {\mathcal{H}}^{\prime }\right)$中。这是因为$\mathcal{R}$中${r}_{c}^{min}$和${s}_{c}^{top}$之间以及$\mathcal{S}$中${s}_{c}^{min}$和${r}_{c}^{top}$之间的$c$ -子集，除了$\mathcal{L}\left\lbrack  {r}_{c}^{\min }\right\rbrack$和${\mathcal{L}}^{\prime }\left\lbrack  {s}_{c}^{\min }\right\rbrack$中的集合外，不会出现在其他集合中，并且必定是冗余$c$ -子集。其余部分与自连接的情况相同。

The pseudo-code of the HeapDedup method for $\mathcal{R}$ - $\mathcal{S}$ join is shown in Algorithm 7. It takes two collections of small sets ${\mathcal{R}}_{s}$ and ${\mathcal{S}}_{s}$ as input and outputs two slimmed inverted indexes ${\mathcal{L}}_{\text{slim }}$ and ${\mathcal{L}}_{\text{slim }}^{\prime }$ for ${\mathcal{R}}_{s}$ and ${\mathcal{S}}_{s}$ respectively. It first initializes two min-heaps $\mathcal{H}$ and ${\mathcal{H}}^{\prime }$ for ${\mathcal{R}}_{s}$ and $\mathcal{S}$ and pops out the smallest min-subsets ${r}_{c}^{\min }$ and ${s}_{c}^{\min }$ from $\mathcal{H}$ and ${\mathcal{H}}^{\prime }$ (Lines 1 to 2). Suppose that ${r}_{c}^{\min }$ and ${s}_{c}^{\min }$ are come from $R$ and $S$ respectively. It keeps comparing ${\mathrm{r}}_{c}^{\min }$ and ${\mathrm{s}}_{c}^{\min }$ until either $\mathcal{H}$ or ${\mathcal{H}}^{\prime }$ is empty (Line 3). If ${r}_{c}^{\min } > {s}_{c}^{\min }$ ,it first appends $S$ to ${\mathcal{L}}_{\text{slim }}^{\prime }\left\lbrack  {\mathrm{s}}_{c}^{\min }\right\rbrack$ ,then binary searches $S$ for the first $c$ -subset that is no smaller than ${r}_{c}^{\min }$ ,next reinserts it to ${\mathcal{H}}^{\prime }$ ,and finally pops ${\mathcal{H}}^{\prime }$ to get the next ${\mathrm{s}}_{c}^{\min }$ (Line 6). If ${\mathrm{r}}_{c}^{\min } < {\mathrm{s}}_{c}^{\min }$ ,it first appends $R$ to ${\mathcal{L}}_{\text{slim }}\left\lbrack  {r}_{c}^{\text{min }}\right\rbrack$ ,then binary searches $R$ for the first $c$ -subset that is no smaller than ${\mathrm{s}}_{c}^{\min }$ ,next reinserts it to $\mathcal{H}$ ,and finally pops $\mathcal{H}$ to get the next ${r}_{c}^{\min }$ (Line 8). If ${r}_{c}^{\min } = {s}_{c}^{\min }$ ,it keeps popping $\mathcal{H}$ until ${r}_{c}^{\min } \neq  {r}_{c}^{\text{top }}$ and builds the inverted list ${\mathcal{L}}_{\text{slim }}\left\lbrack  {r}_{c}^{\min }\right\rbrack$ (Line 11). Similarly it keeps popping ${\mathcal{H}}^{\prime }$ until ${\mathrm{s}}_{c}^{\min } \neq  {\mathrm{s}}_{c}^{\text{top }}$ and builds the inverted list ${\mathcal{L}}_{\text{slim }}^{\prime }\left\lbrack  {\mathrm{s}}_{c}^{\text{min }}\right\rbrack$ (Line 13). In this way,it can construct two slimmed inverted indexes.

$\mathcal{R}$ - $\mathcal{S}$连接的HeapDedup方法的伪代码如算法7所示。它将两个小集合${\mathcal{R}}_{s}$和${\mathcal{S}}_{s}$作为输入，并分别输出${\mathcal{R}}_{s}$和${\mathcal{S}}_{s}$的两个精简倒排索引${\mathcal{L}}_{\text{slim }}$和${\mathcal{L}}_{\text{slim }}^{\prime }$。它首先为${\mathcal{R}}_{s}$和$\mathcal{S}$初始化两个最小堆$\mathcal{H}$和${\mathcal{H}}^{\prime }$，并从$\mathcal{H}$和${\mathcal{H}}^{\prime }$中弹出最小的最小子集${r}_{c}^{\min }$和${s}_{c}^{\min }$（第1行到第2行）。假设${r}_{c}^{\min }$和${s}_{c}^{\min }$分别来自$R$和$S$。它会持续比较${\mathrm{r}}_{c}^{\min }$和${\mathrm{s}}_{c}^{\min }$，直到$\mathcal{H}$或${\mathcal{H}}^{\prime }$为空（第3行）。如果${r}_{c}^{\min } > {s}_{c}^{\min }$，它首先将$S$追加到${\mathcal{L}}_{\text{slim }}^{\prime }\left\lbrack  {\mathrm{s}}_{c}^{\min }\right\rbrack$，然后对$S$进行二分查找，找到第一个不小于${r}_{c}^{\min }$的$c$ - 子集，接着将其重新插入到${\mathcal{H}}^{\prime }$中，最后弹出${\mathcal{H}}^{\prime }$以获取下一个${\mathrm{s}}_{c}^{\min }$（第6行）。如果${\mathrm{r}}_{c}^{\min } < {\mathrm{s}}_{c}^{\min }$，它首先将$R$追加到${\mathcal{L}}_{\text{slim }}\left\lbrack  {r}_{c}^{\text{min }}\right\rbrack$，然后对$R$进行二分查找，找到第一个不小于${\mathrm{s}}_{c}^{\min }$的$c$ - 子集，接着将其重新插入到$\mathcal{H}$中，最后弹出$\mathcal{H}$以获取下一个${r}_{c}^{\min }$（第8行）。如果${r}_{c}^{\min } = {s}_{c}^{\min }$，它会持续弹出$\mathcal{H}$，直到${r}_{c}^{\min } \neq  {r}_{c}^{\text{top }}$，并构建倒排列表${\mathcal{L}}_{\text{slim }}\left\lbrack  {r}_{c}^{\min }\right\rbrack$（第11行）。类似地，它会持续弹出${\mathcal{H}}^{\prime }$，直到${\mathrm{s}}_{c}^{\min } \neq  {\mathrm{s}}_{c}^{\text{top }}$，并构建倒排列表${\mathcal{L}}_{\text{slim }}^{\prime }\left\lbrack  {\mathrm{s}}_{c}^{\text{min }}\right\rbrack$（第13行）。通过这种方式，它可以构建两个精简倒排索引。

The blocking-based methods for the $\mathcal{R} - \mathcal{S}$ join. For the $\mathcal{R} - \mathcal{S}$ join case,we still block all the $c$ -subsets based on the smallest elements. We build two element inverted indexes $I$ and ${I}^{\prime }$ for the two datasets. Then for each element $e$ ,if $\mathcal{I}\left\lbrack  e\right\rbrack   \neq  \phi$ and ${\mathcal{I}}^{\prime }\left\lbrack  e\right\rbrack   \neq  \phi$ , we can independently apply the heap-based methods on the sets in $\mathcal{I}\left\lbrack  e\right\rbrack$ and ${\mathcal{I}}^{\prime }\left\lbrack  e\right\rbrack$ by only inserting those $c$ -subsets with the smallest element as $e$ to the heaps.

基于分块的$\mathcal{R} - \mathcal{S}$连接方法。对于$\mathcal{R} - \mathcal{S}$连接的情况，我们仍然基于最小元素对所有$c$ - 子集进行分块。我们为两个数据集构建两个元素倒排索引$I$和${I}^{\prime }$。然后，对于每个元素$e$，如果$\mathcal{I}\left\lbrack  e\right\rbrack   \neq  \phi$且${\mathcal{I}}^{\prime }\left\lbrack  e\right\rbrack   \neq  \phi$，我们可以仅将最小元素为$e$的$c$ - 子集插入堆中，从而独立地对$\mathcal{I}\left\lbrack  e\right\rbrack$和${\mathcal{I}}^{\prime }\left\lbrack  e\right\rbrack$中的集合应用基于堆的方法。

The pseudo-code of the BlockDedup method for $\mathcal{R} - \mathcal{S}$ join is shown in Algorithm 7. It first fixes a global order for all the elements and builds two inverted indexes $I$ and ${I}^{\prime }$ for the elements in ${\mathcal{R}}_{s}$ and ${\mathcal{S}}_{s}$ respectively (Lines 1 to 2). Then for each element $e$ such that $\mathcal{I}\left\lbrack  e\right\rbrack   \neq  \phi$ and ${\mathcal{I}}^{\prime }\left\lbrack  e\right\rbrack   \neq  \phi$ ,it builds two temporary collections of sets ${\mathcal{R}}_{tmp}$ and ${\mathcal{S}}_{tmp}$ by removing the elements no larger than $e$ in $\mathcal{I}\left\lbrack  e\right\rbrack$ and ${\mathcal{I}}^{\prime }\left\lbrack  e\right\rbrack$ respectively (Lines 4 to 5). It utilizes the HeapDedup method to construct the parts of the slimmed inverted indexes for the two temporary sets with the threshold $c - 1$ (Line 6). Finally it can get two slimmed inverted indexes and return them (Line 7).

$\mathcal{R} - \mathcal{S}$连接的BlockDedup方法的伪代码如算法7所示。它首先为所有元素确定一个全局顺序，并分别为${\mathcal{R}}_{s}$和${\mathcal{S}}_{s}$中的元素构建两个倒排索引$I$和${I}^{\prime }$（第1行到第2行）。然后，对于每个满足$\mathcal{I}\left\lbrack  e\right\rbrack   \neq  \phi$且${\mathcal{I}}^{\prime }\left\lbrack  e\right\rbrack   \neq  \phi$的元素$e$，它分别通过移除$\mathcal{I}\left\lbrack  e\right\rbrack$和${\mathcal{I}}^{\prime }\left\lbrack  e\right\rbrack$中不大于$e$的元素来构建两个临时集合${\mathcal{R}}_{tmp}$和${\mathcal{S}}_{tmp}$（第4行到第5行）。它利用HeapDedup方法，以阈值$c - 1$为两个临时集合构建精简倒排索引的部分（第6行）。最后，它可以得到两个精简倒排索引并返回它们（第7行）。

The boundary size selection method for the $\mathcal{R} - \mathcal{S}$ join. The boundary size selection method for the $\mathcal{R} - \mathcal{S}$ join is basically the same as that for the self-join case. The time cost for the large set is proportional to $\mathop{\sum }\limits_{{R \in  {\mathcal{R}}_{I}}}\mathop{\sum }\limits_{{e \in  R}}\left| {{\mathcal{I}}^{\prime }\left\lbrack  e\right\rbrack  }\right|  + \mathop{\sum }\limits_{{S \in  {\mathcal{S}}_{I}}}\mathop{\sum }\limits_{{e \in  S}}\left| {\mathcal{I}\left\lbrack  e\right\rbrack  }\right|$ . We randomly sample small set pairs from ${\mathcal{R}}_{s} \times  {\mathcal{S}}_{s}$ to estimate the result generation cost. We randomly sample blocks to estimate the heap adjusting cost and the binary searching cost. We have the same observation as that of the self-join case on the trends of the time complexities of small sets and large sets with the increase of the size boundary $x$ . Thus the cost model is all the same. We first set the size boundary $x$ as the smallest set size in both datasets or $c$ , whichever is larger and try to increase $x$ by 1 each time. We use the increasing of the time cost for small sets as the cost and the decreasing of the time cost for large sets as the benefit. We stop increasing $x$ when the benefit is smaller than the cost.

$\mathcal{R} - \mathcal{S}$连接的边界大小选择方法。$\mathcal{R} - \mathcal{S}$连接的边界大小选择方法与自连接情况基本相同。大集合的时间成本与$\mathop{\sum }\limits_{{R \in  {\mathcal{R}}_{I}}}\mathop{\sum }\limits_{{e \in  R}}\left| {{\mathcal{I}}^{\prime }\left\lbrack  e\right\rbrack  }\right|  + \mathop{\sum }\limits_{{S \in  {\mathcal{S}}_{I}}}\mathop{\sum }\limits_{{e \in  S}}\left| {\mathcal{I}\left\lbrack  e\right\rbrack  }\right|$成正比。我们从${\mathcal{R}}_{s} \times  {\mathcal{S}}_{s}$中随机采样小集合对来估计结果生成成本。我们随机采样分块来估计堆调整成本和二分查找成本。随着大小边界$x$的增加，我们对小集合和大集合的时间复杂度趋势的观察与自连接情况相同。因此，成本模型完全相同。我们首先将大小边界$x$设置为两个数据集中最小的集合大小或$c$，取较大者，然后每次尝试将$x$增加1。我们将小集合时间成本的增加作为成本，将大集合时间成本的减少作为收益。当收益小于成本时，我们停止增加$x$。

## C THE TIME COMPLEXITY OF PREFIX FILTER

## C 前缀过滤的时间复杂度

Here is an example to show that the prefix filter has a worst case time complexity of $O\left( {n}^{2}\right)$ . Suppose there is a constant number $p$ of distinct elements in the sets, all the elements have the same frequency $\frac{n}{p}$ ,and the sizes of all the sets are larger than $c$ . There exists at least one element (the first element in the global order used by All-Pairs) whose corresponding inverted list has a length of $\frac{n}{p}$ and is scanned $\frac{n}{p}$ times. Thus the complexity is $O\left( {\frac{n}{p} \times  \frac{n}{p}}\right)  = O\left( {n}^{2}\right)$ .

这里有一个例子表明前缀过滤器的最坏情况时间复杂度为$O\left( {n}^{2}\right)$。假设集合中有常数个$p$不同元素，所有元素的频率都相同，均为$\frac{n}{p}$，并且所有集合的大小都大于$c$。至少存在一个元素（全对算法所使用的全局顺序中的第一个元素），其对应的倒排列表长度为$\frac{n}{p}$，并且会被扫描$\frac{n}{p}$次。因此，复杂度为$O\left( {\frac{n}{p} \times  \frac{n}{p}}\right)  = O\left( {n}^{2}\right)$。

## D RELATIONSHIP WITH THE OTHER SIMILARITY FUNCTIONS

## D 与其他相似度函数的关系

Just like all the similarity problems, set similarity join can also be studied under other functions, such as Jaccard similarity, Cosine similarity, Dice similarity, edit distance, and the normalized overlap similarity. Every metric has its pros and cons, such that no metric serves as a one-size-fits-all approach that cures all the issues coming up in practice. For example, in some applications, the set sizes may differ a lot, in which case most metrics will give low similarities to the "lop-sided" set pairs (where one set out-sizes the other significantly), whereas the overlap similarity is known to be much less sensitive to such an issue.

和所有相似度问题一样，集合相似度连接也可以在其他函数下进行研究，例如杰卡德相似度（Jaccard similarity）、余弦相似度（Cosine similarity）、戴斯相似度（Dice similarity）、编辑距离（edit distance）和归一化重叠相似度（normalized overlap similarity）。每个度量都有其优缺点，因此没有一种度量可以作为解决实际中出现的所有问题的通用方法。例如，在某些应用中，集合的大小可能差异很大，在这种情况下，大多数度量会给“不均衡”的集合对（其中一个集合的大小明显大于另一个集合）赋予较低的相似度，而重叠相似度对这种问题的敏感度要低得多。

Nevertheless, when all the sets in the given dataset have the same sizes (e.g., the sets in ADDRESS), the overlap similarity can be transformed to Jaccard similarity, Cosine similarity, Dice similarity, and the normalized overlap similarity and vice versa. For example, suppose all the set sizes are $m$ . Then the set pairs with overlap similarity no smaller than $c$ are exactly the set pairs with Jaccard similarity no smaller than $\frac{c}{{2m} - c}$ . This is because for any sets $r$ and $s$ in the given dataset,we have

然而，当给定数据集中的所有集合大小相同时（例如，地址数据集中的集合），重叠相似度可以转换为杰卡德相似度、余弦相似度、戴斯相似度和归一化重叠相似度，反之亦然。例如，假设所有集合的大小均为$m$。那么重叠相似度不小于$c$的集合对恰好是杰卡德相似度不小于$\frac{c}{{2m} - c}$的集合对。这是因为对于给定数据集中的任何集合$r$和$s$，我们有

$$
\operatorname{Jaccard}\left( {r,s}\right)  = \frac{\left| r \cap  s\right| }{\left| r \cup  s\right| } = \frac{\left| r \cap  s\right| }{\left| r\right|  + \left| s\right|  - \left| {r \cap  s}\right| } = \frac{\left| r \cap  s\right| }{{2m} - \left| {r \cap  s}\right| }
$$

and thus $\left| {r \cap  s}\right|  \geq  c$ iff $\operatorname{Jaccard}\left( {r,s}\right)  \geq  \frac{c}{{2m} - c}$ . In this case,SizeAware and the methods for set similarity joins under the other similarity functions can be used to solve each other's problems. We compared SizeAware with five existing methods for set similarity joins under Jaccard similarity constraint on ADDRESS in Appendix E.

因此，当且仅当$\operatorname{Jaccard}\left( {r,s}\right)  \geq  \frac{c}{{2m} - c}$时，$\left| {r \cap  s}\right|  \geq  c$。在这种情况下，SizeAware算法和其他相似度函数下的集合相似度连接方法可以相互解决对方的问题。我们在附录E中，在地址数据集上，将SizeAware算法与五种现有的在杰卡德相似度约束下的集合相似度连接方法进行了比较。

In general, when the set sizes are not all the same in the given dataset, all the alternative similarity functions mentioned earlier can actually be transformed to overlap similarity (but not vice versa), after which our SizeAware algorithm can be used to solve the set similarity join under those metrics as well. The transformation can be achieved using an existing technique called AdaptJoin [30]. For each set $R$ ,given a constant $l$ ,the fixed-length prefix schema of AdaptJoin takes the first $\left| R\right|  - f\left( \left| R\right| \right)  + l$ elements as prefixes,and guarantees that two sets are similar only if their prefixes share at least $l$ elements,where $f\left( \left| R\right| \right)$ is a function that depends on the similarity metric. By taking all the $l$ -prefixes as input and setting the overlap threshold $c = l$ ,our size-aware algorithm can efficiently identify the candidates for the set similarity join problem. The candidates can then be fed into a verification step to produce the final results. In practice, we can use the advance length filter [17], prefix filter [33], and position filter [31] to process the large sets and the sets $R$ with $f\left( \left| R\right| \right)  < c$ . We can also use the estimation in AdaptJoin to select a good $c$ .

一般来说，当给定数据集中的集合大小并不都相同时，前面提到的所有替代相似度函数实际上都可以转换为重叠相似度（但反之则不成立），之后我们的SizeAware算法也可以用于解决这些度量下的集合相似度连接问题。可以使用一种名为AdaptJoin [30]的现有技术来实现这种转换。对于每个集合$R$，给定一个常数$l$，AdaptJoin的固定长度前缀模式会选取前$\left| R\right|  - f\left( \left| R\right| \right)  + l$个元素作为前缀，并保证只有当两个集合的前缀至少共享$l$个元素时，这两个集合才是相似的，其中$f\left( \left| R\right| \right)$是一个依赖于相似度度量的函数。通过将所有的$l$ - 前缀作为输入，并设置重叠阈值$c = l$，我们的大小感知算法可以有效地识别集合相似度连接问题的候选集。然后可以将这些候选集输入到验证步骤中以产生最终结果。在实践中，我们可以使用提前长度过滤器 [17]、前缀过滤器 [33]和位置过滤器 [31]来处理大型集合和$f\left( \left| R\right| \right)  < c$的集合$R$。我们还可以使用AdaptJoin中的估计方法来选择一个合适的$c$。

The reverse transformation, on the other hand, does not always appear to be possible when the set sizes are not all the same. This fundamental nature of overlap similarity provides further motivation for our algorithmic study of this metric.

另一方面，当集合大小并不都相同时，反向转换似乎并不总是可行的。重叠相似度的这种基本特性为我们对该度量进行算法研究提供了进一步的动力。

## E MORE EXPERIMENTS

## E 更多实验

The Set Size Distributions. The set size distributions of DBLP, CLICK, and ORKUT are shown in Figure 11.

集合大小分布。图11展示了DBLP、CLICK和ORKUT数据集的集合大小分布情况。

Memory Usage. We also compared the memory usage with existing methods and Table 4 gives the numbers. We can see that the memory usage of SizeAware is comparable to the existing approaches. In our heap-based methods,the smallest $c$ -subsets ${r}_{c}$ are first popped out from the heap and the inverted list $\mathcal{L}\left\lbrack  {r}_{c}\right\rbrack$ is constructed. We can enumerate the results in this inverted list and drop it immediately as it will never be used again later in the algorithm. Thus it only needs a small amount of memory to keep the results,the element inverted index $\mathcal{I}$ ,and the heap. However,the heap-based methods may generate duplicate results. In our implementation, we keep the whole slimmed inverted index for efficient result deduplication.

内存使用情况。我们还将内存使用情况与现有方法进行了比较，表4给出了具体数据。我们可以看到，SizeAware的内存使用情况与现有方法相当。在我们基于堆的方法中，最小的 $c$ -子集 ${r}_{c}$ 首先从堆中弹出，并构建倒排列表 $\mathcal{L}\left\lbrack  {r}_{c}\right\rbrack$。我们可以枚举这个倒排列表中的结果，并立即丢弃它，因为在算法的后续过程中它不会再被使用。因此，它只需要少量的内存来保存结果、元素倒排索引 $\mathcal{I}$ 和堆。然而，基于堆的方法可能会产生重复的结果。在我们的实现中，我们保留了整个精简后的倒排索引，以实现高效的结果去重。

<!-- Media -->

<!-- figureText: Number of sets 20000 2000 9000 6000 3000 0 10 100 1000 100 1000 10000 Set sizes Set sizes (b) CLICK (c) ORKUT Figure 11: Set Size Distributions 700 800 Elapsed Time (s 600 Elapsed Time (s) c=2 c=3 600 c=4 ...※…… c=6 400 . 日 200 ------------ 500 c=20 -A- 400 300 200 100 1m 1.5r 2m 2.5m 3m 1m 1.5m 2m 2.5m 3m Dataset Sizes Dataset Si (c) ORKUT (d) ADDRESS 300 Elapsed Time (s _____③ 一 Elapsed Time (s) SKJ --- X- - ppjoin - I-sizeaware 200 100 15 1 + 一日一 0.8 0.85 0.95 0.8 0.85 0.95 Threshold $\delta$ Threshold $\delta$ (c) CLICK (d) ORKUT 60000 Number of sets 60000 20000 80000 40000 30000 1 100 Set sizes (a) DBLP 800 Elapsed Time (s) 一日一 Clapsed Time ( s 200 ------------ 600 ………… c=12 400 200 1m 1.5m 2m 2.5m 3m 250k 500k 750k 1m Dataset Sizes Dataset thises (a) DBLP (b) CLICK 30 Elapsed Time (s) 5000 SKJ Elapsed Time ( s ) SKJ ---Q· -- X- - ppjoin 20 + 一日一 10 4000 ppjoin ppjoin+ 一日 3000 2000 1000 2 (.16) (.55 6 (.75) 0.8 0.85 0.95 Threshold c $\left( \delta \right)$ Threshold $\delta$ (a) ADDRESS (b) DBLP -->

<img src="https://cdn.noedgeai.com/0195ccc7-1611-78aa-a97d-afb7fc00df51_15.jpg?x=142&y=200&w=1461&h=977&r=0"/>

Figure 13: Comparison with Existing Methods for Jaccard Set Similarity Join

图13：Jaccard集合相似度连接与现有方法的比较

<table><tr><td/><td>ScanCount</td><td>DivideSkip</td><td>AdaptJoin</td><td>AllPair</td><td>SizeAware</td></tr><tr><td>DBLP, $c = 8$</td><td>${180}\mathrm{{MB}}$</td><td>199 MB</td><td>308 MB</td><td>309 MB</td><td>579 MB</td></tr><tr><td>CLICK, $c = 8$</td><td>142 MB</td><td>2251 MB</td><td>304 MB</td><td>241 MB</td><td>940 MB</td></tr><tr><td>ORKUT, $c = {12}$</td><td>1578 MB</td><td>1835 MB</td><td>2974 MB</td><td>2592 MB</td><td>3584 MB</td></tr><tr><td>ADDRESS, $c = 4$</td><td>202 MB</td><td>4400 MB</td><td>364 MB</td><td>336 MB</td><td>534 MB</td></tr><tr><td>DBLP</td><td>$c = 4$</td><td>$c = 6$</td><td>$c = 8$</td><td>$c = {10}$</td><td>$c = {12}$</td></tr><tr><td>SizeAware</td><td>2712 MB</td><td>1448 MB</td><td>579 MB</td><td>393 MB</td><td>${320}\mathrm{{MB}}$</td></tr></table>

<table><tbody><tr><td></td><td>扫描计数</td><td>分割跳过</td><td>自适应连接</td><td>全对</td><td>大小感知</td></tr><tr><td>计算机科学 bibliography（DBLP）, $c = 8$</td><td>${180}\mathrm{{MB}}$</td><td>199兆字节</td><td>308兆字节</td><td>309兆字节</td><td>579兆字节</td></tr><tr><td>点击数据集（CLICK）, $c = 8$</td><td>142兆字节</td><td>2251兆字节</td><td>304兆字节</td><td>241兆字节</td><td>940兆字节</td></tr><tr><td>社交网络数据集（ORKUT）, $c = {12}$</td><td>1578兆字节</td><td>1835兆字节</td><td>2974兆字节</td><td>2592兆字节</td><td>3584兆字节</td></tr><tr><td>地址数据集（ADDRESS）, $c = 4$</td><td>202兆字节</td><td>4400兆字节</td><td>364兆字节</td><td>336兆字节</td><td>534兆字节</td></tr><tr><td>计算机科学 bibliography（DBLP）</td><td>$c = 4$</td><td>$c = 6$</td><td>$c = 8$</td><td>$c = {10}$</td><td>$c = {12}$</td></tr><tr><td>大小感知</td><td>2712兆字节</td><td>1448兆字节</td><td>579兆字节</td><td>393兆字节</td><td>${320}\mathrm{{MB}}$</td></tr></tbody></table>

Table 4: The Memory Usage

表4：内存使用情况

<!-- Media -->

The Scalability of R-S Join. We report the scalability of our SizeAware method for $\mathcal{R} - \mathcal{S}$ join in this section. We still varied the sizes of the datasets from 1 million to 3 millions for DBLP, ORKUT, and ADDRESS and from 250 thousand to almost 1 million for CLICK. We equally and randomly divided all the datasets into two parts for the $\mathcal{R} - \mathcal{S}$ join case and reported the elapsed time. The results are shown in Figure 12. We can see that the scalability of our method on $\mathcal{R} - \mathcal{S}$ join case was fairly good. For example,for DBLP dataset,when the threshold $c = 4$ ,the elapsed time for 1 million,1.5 million, 2 million, 2.5 million, and 3 million sets were respectively 102 seconds, 180 seconds, 278 seconds, 405 seconds, and 518 seconds, which is consistent with our time complexity analysis. This is because the effectiveness of our proposed heuristics. In addition, the size boundary selection method can choose a good boundary for the SizeAware algorithm.

R - S连接的可扩展性。在本节中，我们报告了我们的SizeAware方法在$\mathcal{R} - \mathcal{S}$连接上的可扩展性。对于DBLP、ORKUT和ADDRESS数据集，我们仍然将数据集的大小从100万变化到300万；对于CLICK数据集，从25万变化到近100万。对于$\mathcal{R} - \mathcal{S}$连接情况，我们将所有数据集均匀且随机地分成两部分，并报告了运行时间。结果如图12所示。我们可以看到，我们的方法在$\mathcal{R} - \mathcal{S}$连接情况下的可扩展性相当好。例如，对于DBLP数据集，当阈值为$c = 4$时，100万、150万、200万、250万和300万集合的运行时间分别为102秒、180秒、278秒、405秒和518秒，这与我们的时间复杂度分析一致。这是因为我们提出的启发式方法有效。此外，大小边界选择方法可以为SizeAware算法选择一个合适的边界。

Comparing with Jaccard Set Similarity Join Methods. We compared SizeAware with six existing methods for set similarity join under Jaccard similarity constraint on the special ADDRESS dataset, which are AdaptJoin [30], GroupJoin [4], PPJoin [33], PPJoin+ [33], PEL [17], SKJ [31] (see Section 7 for the details of these methods). Figure 13(a) gives the results. Note PPJoin and PPJoin+ ran out of memory when $c = 2$ (which is the same as the threshold $\delta  = {0.16}$ for Jaccard similarity). We can see that SizeAware consistently outperformed the existing methods in all cases by up to 1-2 orders of magnitude. For example,when $c = 3$ (that is $\delta  = {0.27}$ ),the elapsed time for AdaptJoin, SKJ, PPJoin, PPJoin+, GroupJoin, PEL, and SizeAware was 1898s, 1318s, 2840s, 3024s, 1412s, 1608s, and 59s respectively. The reason was two-fold. First, when all the set sizes are exactly the same, the most effective length filter in all the existing work does not work. Second, all the existing methods use a filter-and-refine framework, which first generates candidates by some filtering conditions and then verifies the survived pairs. However, on ADDRESS dataset, the corresponding thresholds of the Jaccard similarity are low which limits the pruning power of the filtering conditions and makes the existing methods performed rather bad. Our SizeAware algorithm, however, directly generated all the result pairs without generating candidates. In other words, our method does not require a verification step, which is one of the reasons behind its efficiency. We also implemented the extensions for SizeAware as discussed in Appendix D to support Jaccard similarity and compared it with the six methods above on the other three datasets which have different set sizes. We varied the Jaccard similarity threshold from 0.8 to 0.95 and reported the elapsing time. Figure 13(b)-13(d) shows the results. Our SizeAware algorithm consistently outperformed the other methods. This is attributed to our proposed SizeAware algorithm and heap-based methods for reducing the filtering time and the cost-model in AdaptJoin for choosing the right value of $c$ .

与杰卡德集合相似度连接方法的比较。我们在特殊的ADDRESS数据集上，将SizeAware与六种现有的在杰卡德相似度约束下的集合相似度连接方法进行了比较，这些方法分别是AdaptJoin [30]、GroupJoin [4]、PPJoin [33]、PPJoin+ [33]、PEL [17]、SKJ [31]（这些方法的详细信息见第7节）。图13(a)给出了结果。注意，当$c = 2$（这与杰卡德相似度的阈值$\delta  = {0.16}$相同）时，PPJoin和PPJoin+内存不足。我们可以看到，在所有情况下，SizeAware始终比现有方法性能高出1 - 2个数量级。例如，当$c = 3$（即$\delta  = {0.27}$）时，AdaptJoin、SKJ、PPJoin、PPJoin+、GroupJoin、PEL和SizeAware的运行时间分别为1898秒、1318秒、2840秒、3024秒、1412秒、1608秒和59秒。原因有两方面。首先，当所有集合的大小完全相同时，现有工作中最有效的长度过滤方法不起作用。其次，所有现有方法都使用过滤 - 细化框架，即首先通过一些过滤条件生成候选对，然后验证存活的对。然而，在ADDRESS数据集上，杰卡德相似度的相应阈值较低，这限制了过滤条件的剪枝能力，使得现有方法表现相当糟糕。然而，我们的SizeAware算法直接生成所有结果对，而不生成候选对。换句话说，我们的方法不需要验证步骤，这是其高效的原因之一。我们还实现了附录D中讨论的SizeAware的扩展，以支持杰卡德相似度，并在其他三个具有不同集合大小的数据集上与上述六种方法进行了比较。我们将杰卡德相似度阈值从0.8变化到0.95，并报告了运行时间。图13(b) - 13(d)显示了结果。我们的SizeAware算法始终优于其他方法。这归因于我们提出的SizeAware算法和基于堆的方法来减少过滤时间，以及AdaptJoin中的成本模型来选择$c$的正确值。