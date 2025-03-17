# An Introduction to Johnson-Lindenstrauss Transforms

# 约翰逊 - 林登施特劳斯变换（Johnson-Lindenstrauss Transforms）简介

Casper Benjamin Freksen*

卡斯珀·本杰明·弗雷克森（Casper Benjamin Freksen）*

2nd March 2021

2021年3月2日

## Abstract

## 摘要

Johnson-Lindenstrauss Transforms are powerful tools for reducing the dimensionality of data while preserving key characteristics of that data, and they have found use in many fields from machine learning to differential privacy and more. This note explains what they are; it gives an overview of their use and their development since they were introduced in the 1980s; and it provides many references should the reader wish to explore these topics more deeply.

约翰逊 - 林登施特劳斯变换（Johnson-Lindenstrauss Transforms）是在保留数据关键特征的同时降低数据维度的强大工具，它们已在从机器学习到差分隐私等众多领域得到应用。本说明解释了它们是什么；概述了自20世纪80年代引入以来它们的用途和发展；如果读者希望更深入地探索这些主题，还提供了许多参考文献。

The text was previously a main part of the introduction of my PhD thesis [Fre20], but it has been adapted to be self contained and serve as a (hopefully good) starting point for readers interested in the topic.

本文曾是我博士论文 [Fre20] 引言的主要部分，但现在已改编为独立内容，希望能为对该主题感兴趣的读者提供一个（有望是不错的）起点。

## 1 The Why, What, and How

## 1 原因、内容与方法

### 1.1 The Problem

### 1.1 问题

Consider the following scenario: We have some data that we wish to process but the data is too large, e.g. processing the data takes too much time, or storing the data takes too much space. A solution would be to compress the data such that the valuable parts of the data are kept and the other parts discarded. Of course, what is considered valuable is defined by the data processing we wish to apply to our data. To make our scenario more concrete let us say that our data consists of vectors in a high dimensional Euclidean space, ${\mathbb{R}}^{d}$ ,and we wish to find a transform to embed these vectors into a lower dimensional space, ${\mathbb{R}}^{m}$ ,where $m \ll  d$ ,so that we can apply our data processing in this lower dimensional space and still get meaningful results. The problem in this more concrete scenario is known as dimensionality reduction.

考虑以下场景：我们有一些想要处理的数据，但这些数据量太大，例如处理这些数据需要花费太多时间，或者存储这些数据需要占用太多空间。一种解决方案是对数据进行压缩，保留数据中有价值的部分，丢弃其他部分。当然，什么是有价值的，是由我们想要对数据进行的处理来定义的。为了让这个场景更具体，假设我们的数据由高维欧几里得空间（Euclidean space）${\mathbb{R}}^{d}$ 中的向量组成，我们希望找到一种变换，将这些向量嵌入到一个低维空间 ${\mathbb{R}}^{m}$ 中，其中 $m \ll  d$ ，这样我们就可以在这个低维空间中进行数据处理，并且仍然能得到有意义的结果。在这个更具体的场景中，这个问题被称为降维（dimensionality reduction）。

As an example, let us pretend to be a library with a large corpus of texts and whenever a person returns a novel that they liked, we would like to recommend some similar novels for them to read next. Or perhaps we wish to be able to automatically categorise the texts into groups such as fiction/non-fiction or child/young-adult/adult literature. To be able to use the wealth of research on similarity search and classification we need a suitable representation of our texts, and here a common choice is called bag-of-words. For a language or vocabulary with $d$ different words,the bag-of-words representation of a text $t$ is a vector $x \in  {\mathbb{R}}^{d}$ whose $i$ th entry is the number of times the $i$ th word occurs in $t$ . For example,if the language is just ["be","is", "not", "or", "question", "that", "the", "to"] then the text "to be or not to be" is represented as ${\left( 2,0,1,1,0,0,0,2\right) }^{\top }$ . To capture some of the context of the words,we can instead represent a text as the count of so-called $n$ -grams ${}^{1}$ ,which are sequences of $n$ consecutive words,e.g. the 2-grams of "to be or not to be" are ["to be", "be or", "or not", "not to", "to be"], and we represent such a bag-of- $n$ -grams as a vector in ${\mathbb{R}}^{\left( {d}^{n}\right) }$ . To compare two texts we compute the distance between the vectors of those texts, because the distance between vectors of texts with mostly the same words (or $n$ -grams) is small ${}^{2}$ . For a more realistic language such as English with $d \approx  {171000}$ words [SW89] or that of the "English" speaking internet at $d \gtrsim  {4790000}$ words [WZ05],the dimension quickly becomes infeasable. While we only need to store the nonzero counts of words (or $n$ -grams) to represent a vector,many data processing algorithms have a dependency on the vector dimension $d$ (or ${d}^{n}$ ),e.g. using nearest-neighbour search to find similar novels to recommend [AI17] or using neural networks to classify our texts [Sch18]. These algorithms would be infeasible for our library use case if we do not first reduce the dimension of our data.

举个例子，假设我们是一个拥有大量文本语料库的图书馆，每当有人归还一本他们喜欢的小说时，我们希望为他们推荐一些类似的小说供其接下来阅读。或者，我们可能希望能够自动将文本分类，例如分为虚构类/非虚构类，或者儿童文学/青少年文学/成人文学。为了能够利用关于相似性搜索和分类的大量研究成果，我们需要对文本进行合适的表示，这里常用的一种表示方法叫做词袋模型。对于包含 $d$ 个不同单词的语言或词汇表，文本 $t$ 的词袋表示是一个向量 $x \in  {\mathbb{R}}^{d}$，其第 $i$ 个元素是第 $i$ 个单词在 $t$ 中出现的次数。例如，如果词汇表仅包含 ["be","is", "not", "or", "question", "that", "the", "to"]，那么文本 "to be or not to be" 就表示为 ${\left( 2,0,1,1,0,0,0,2\right) }^{\top }$。为了捕捉单词的一些上下文信息，我们可以将文本表示为所谓的 $n$ -元语法（${}^{1}$）的计数，$n$ -元语法是由 $n$ 个连续单词组成的序列，例如 "to be or not to be" 的 2 -元语法是 ["to be", "be or", "or not", "not to", "to be"]，我们将这样的 $n$ -元语法词袋表示为 ${\mathbb{R}}^{\left( {d}^{n}\right) }$ 中的一个向量。为了比较两个文本，我们计算这些文本向量之间的距离，因为大部分单词（或 $n$ -元语法）相同的文本向量之间的距离较小 ${}^{2}$。对于像英语这样更现实的语言，有 $d \approx  {171000}$ 个单词 [SW89]，或者在互联网“英语”语境中有 $d \gtrsim  {4790000}$ 个单词 [WZ05]，向量的维度会迅速变得不可行。虽然我们只需要存储单词（或 $n$ -元语法）的非零计数来表示一个向量，但许多数据处理算法依赖于向量的维度 $d$（或 ${d}^{n}$），例如，使用最近邻搜索来查找类似的小说进行推荐 [AI17]，或者使用神经网络对我们的文本进行分类 [Sch18]。如果我们不首先降低数据的维度，这些算法对于我们图书馆的应用场景将不可行。

---

<!-- Footnote -->

*Work done while at the Computer Science Department, Aarhus University. casper@freksen.dk

*在奥胡斯大学（Aarhus University）计算机科学系工作期间完成。casper@freksen.dk

<!-- Footnote -->

---

A seemingly simple approach would be to select a subset of the coordinates, say if the data contained redundant or irrelevant coordinates. This is known as feature selection [JS08; HTF17; Jam+13a],and can be seen as projecting ${}^{3}$ onto an axis aligned subspace,i.e. a subspace whose basis is a subset of $\left\{  {{e}_{1},\ldots ,{e}_{d}}\right\}$ .

一种看似简单的方法是选择坐标的一个子集，例如当数据包含冗余或无关坐标时。这被称为特征选择 [JS08; HTF17; Jam+13a]，并且可以看作是将 ${}^{3}$ 投影到一个轴对齐子空间上，即一个基是 $\left\{  {{e}_{1},\ldots ,{e}_{d}}\right\}$ 的子集的子空间。

We can build upon feature selection by choosing the basis from a richer set of vectors. For instance, in principal component analysis as dimensionality reduction (PCA) [Pea01: Hot33] we let the basis of the subspace be the $m$ first eigenvectors (ordered decreasingly by eigenvalue) of ${X}^{\top }X$ ,where the rows of $X \in  {\mathbb{R}}^{n \times  d}$ are our $n$ high dimensional vectors ${}^{4}$ . This subspace maximises the variance of the data in the sense that the first eigenvector is the axis that maximises variance and subsequent eigenvectors are the axes that maximise variance subject to being orthogonal to all previous eigenvectors [LRU20b; HTF17; Jam+13b].

我们可以通过从更丰富的向量集中选择基来改进特征选择。例如，在作为降维方法的主成分分析（PCA）[Pea01: Hot33]中，我们让子空间的基为$m$个${X}^{\top }X$的前特征向量（按特征值降序排列），其中$X \in  {\mathbb{R}}^{n \times  d}$的行是我们的$n$个高维向量${}^{4}$。从这个意义上说，这个子空间使数据的方差最大化：第一个特征向量是使方差最大化的轴，后续的特征向量是在与所有先前特征向量正交的条件下使方差最大化的轴[LRU20b; HTF17; Jam+13b]。

But what happens if we choose a basis randomly?

但如果我们随机选择一个基会怎样呢？

### 1.2 The Johnson-Lindenstrauss Lemma(s)

### 1.2 约翰逊 - 林登斯特劳斯引理

In 1984 ${}^{5}$ it was discovered that projecting onto a random basis approximately preserves pairwise distances with high probability. In order to prove a theorem regarding Lipschitz extensions of functions from metric spaces into ${\ell }_{2}$ ,Johnson and Lindenstrauss [JL84] proved the following lemma.

1984年${}^{5}$，人们发现，以高概率将向量投影到随机基上可以近似保持成对距离。为了证明关于从度量空间到${\ell }_{2}$的函数的利普希茨（Lipschitz）延拓的一个定理，约翰逊（Johnson）和林登施特劳斯（Lindenstrauss）[JL84]证明了以下引理。

---

<!-- Footnote -->

${}^{1}$ These are sometimes referred to as shingles.

${}^{1}$ 这些有时被称为子串（shingles）。

${}^{2}$ We might wish to apply some normalisation to the vectors,e.g. tf-idf [LRU20a],so that rare words are weighted more and text length is less significant.

${}^{2}$ 我们可能希望对向量应用某种归一化方法，例如词频 - 逆文档频率（tf - idf）[LRU20a]，这样罕见词的权重会更高，而文本长度的影响会更小。

${}^{3}$ Here we slightly bend the definition of projection in the sense that we represent a projected vector as the coefficients of the of the linear combination of the (chosen) basis of the subspace we project onto, rather than as the result of that linear combination. If $A \in  {\mathbb{R}}^{m \times  d}$ is a matrix with the subspace basis vectors as rows,we represent the projection of a vector $x \in  {\mathbb{R}}^{d}$ as the result of ${Ax} \in  {\mathbb{R}}^{m}$ rather than ${A}^{\top }{Ax} \in  {\mathbb{R}}^{d}$ .

${}^{3}$ 在此，我们对投影的定义做了一点变通，即我们将投影向量表示为投影所在子空间（选定）基的线性组合的系数，而非该线性组合的结果。如果 $A \in  {\mathbb{R}}^{m \times  d}$ 是一个以子空间基向量为行的矩阵，我们将向量 $x \in  {\mathbb{R}}^{d}$ 的投影表示为 ${Ax} \in  {\mathbb{R}}^{m}$ 的结果，而非 ${A}^{\top }{Ax} \in  {\mathbb{R}}^{d}$ 的结果。

${}^{4}$ Here it is assumed that the mean of our vectors is 0,otherwise the mean vector of our vectors should be subtracted from each of the rows of $X$ .

${}^{4}$ 这里假设我们的向量均值为 0，否则应从 $X$ 的每一行中减去我们向量的均值向量。

${}^{5}$ Or rather in 1982 as that was when a particular "Conference in Modern Analysis and Probability" was held at Yale University, but the proceedings were published in 1984.

${}^{5}$ 更准确地说，是在 1982 年，因为那一年耶鲁大学（Yale University）举办了一场特别的“现代分析与概率论会议（Conference in Modern Analysis and Probability）”，但其会议记录于 1984 年出版。

<!-- Footnote -->

---

Lemma 1.1 (Johnson-Lindenstrauss lemma [JL84]). For every $d \in  {\mathbb{N}}_{1},\varepsilon  \in  \left( {0,1}\right)$ ,and $X \subset  {\mathbb{R}}^{d}$ , there exists a function $f : X \rightarrow  {\mathbb{R}}^{m}$ ,where $m = \Theta \left( {{\varepsilon }^{-2}\log \left| X\right| }\right)$ such that for every $x,y \in  X$ ,

引理1.1（约翰逊 - 林登斯特劳斯引理 [JL84]）。对于任意的$d \in  {\mathbb{N}}_{1},\varepsilon  \in  \left( {0,1}\right)$和$X \subset  {\mathbb{R}}^{d}$，存在一个函数$f : X \rightarrow  {\mathbb{R}}^{m}$，其中$m = \Theta \left( {{\varepsilon }^{-2}\log \left| X\right| }\right)$，使得对于任意的$x,y \in  X$，

$$
\left| {\parallel f\left( x\right)  - f\left( y\right) {\parallel }_{2}^{2} - \parallel x - y{\parallel }_{2}^{2}}\right|  \leq  \varepsilon \parallel x - y{\parallel }_{2}^{2}. \tag{1}
$$

Proof. The gist of the proof is to first define $f\left( x\right)  \mathrel{\text{:=}} {\left( d/m\right) }^{1/2}{Ax}$ ,where $A \in  {\mathbb{R}}^{m \times  d}$ are the first $m$ rows of a random orthogonal matrix. They then showed that $f$ preserves the norm of any vector with high probability,or more formally that the distribution of $f$ satisfies the following lemma.

证明。该证明的要点是首先定义$f\left( x\right)  \mathrel{\text{:=}} {\left( d/m\right) }^{1/2}{Ax}$，其中$A \in  {\mathbb{R}}^{m \times  d}$是一个随机正交矩阵的前$m$行。然后他们证明了$f$以高概率保持任何向量的范数，或者更正式地说，$f$的分布满足以下引理。

Lemma 1.2 (Distributional Johnson-Lindenstrauss lemma [JL84]). For every $d \in  {\mathbb{N}}_{1}$ and $\varepsilon ,\delta  \in$ (0,1),there exists a probability distribution $\mathcal{F}$ over linear functions $f : {\mathbb{R}}^{d} \rightarrow  {\mathbb{R}}^{m}$ ,where $m =$ $\Theta \left( {{\varepsilon }^{-2}\log \frac{1}{\delta }}\right)$ such that for every $x \in  {\mathbb{R}}^{d}$ ,

引理1.2（分布型约翰逊 - 林登施特劳斯引理 [JL84]）。对于每一个$d \in  {\mathbb{N}}_{1}$和$\varepsilon,\delta  \in$∈(0,1)，存在一个关于线性函数$f : {\mathbb{R}}^{d} \rightarrow  {\mathbb{R}}^{m}$的概率分布$\mathcal{F}$，其中$m =\Theta \left( {{\varepsilon }^{-2}\log \frac{1}{\delta }}\right)$，使得对于每一个$x \in  {\mathbb{R}}^{d}$，

$$
\mathop{\Pr }\limits_{{f \sim  \mathcal{F}}}\left\lbrack  {\left| {\parallel f\left( x\right) {\parallel }_{2}^{2} - \parallel x{\parallel }_{2}^{2}}\right|  \leq  \varepsilon \parallel x{\parallel }_{2}^{2}}\right\rbrack   \geq  1 - \delta . \tag{2}
$$

By choosing $\delta  = 1/{\left| X\right| }^{2}$ ,we can union bound over all pairs of vectors $x,y \in  X$ and show that their distance (i.e. the ${\ell }_{2}$ norm of the vector $x - y$ ) is preserved simultaneously for all pairs with probability at least $1 - \left( \begin{matrix} \left| X\right| \\  2 \end{matrix}\right) /{\left| X\right| }^{2} > 1/2$ .

通过选择 $\delta  = 1/{\left| X\right| }^{2}$，我们可以对所有向量对 $x,y \in  X$ 进行联合界估计，并证明它们的距离（即向量 $x - y$ 的 ${\ell }_{2}$ 范数）以至少 $1 - \left( \begin{matrix} \left| X\right| \\  2 \end{matrix}\right) /{\left| X\right| }^{2} > 1/2$ 的概率同时得到保留。

We will use the term Johnson-Lindenstrauss distribution (JLD) to refer to a distribution $\mathcal{F}$ that is a witness to lemma 1.2,and the term Johnson-Lindenstrauss transform (JLT) to a function $f$ witnessing lemma 1.1,e.g. a sample of a JLD.

我们将使用术语约翰逊 - 林登施特劳斯分布（Johnson - Lindenstrauss distribution，JLD）来指代引理 1.2 所对应的分布 $\mathcal{F}$，使用术语约翰逊 - 林登施特劳斯变换（Johnson - Lindenstrauss transform，JLT）来指代引理 1.1 所对应的函数 $f$，例如 JLD 的一个样本。

A few things to note about these lemmas are that when sampling a JLT from a JLD it is independent of the input vectors themselves; the JLT is only dependendent on the source dimension $d$ ,number of vectors $\left| X\right|$ ,and distortion $\varepsilon$ . This allows us to sample a JLT without having access to the input data, e.g. to compute the JLT before the data exists, or to compute the JLT in settings where the data is too large to store on or move to a single machine ${}^{6}$ . Secondly,the target dimension $m$ is independent from the source dimension $d$ ,meaning there are potentially very significant savings in terms of dimensionality, which will become more apparent shortly.

关于这些引理，有几点需要注意：从约翰逊 - 林登施特劳斯分布（JLD）中采样约翰逊 - 林登施特劳斯变换（JLT）时，它与输入向量本身无关；JLT 仅取决于源维度 $d$、向量数量 $\left| X\right|$ 和失真度 $\varepsilon$。这使我们在无法访问输入数据的情况下也能采样 JLT，例如在数据存在之前计算 JLT，或者在数据太大而无法存储在单台机器上或移动到单台机器的情况下计算 JLT ${}^{6}$。其次，目标维度 $m$ 与源维度 $d$ 无关，这意味着在维度方面可能会有非常显著的节省，这一点很快会变得更加明显。

Compared to PCA, the guarantees that JLTs give are different: PCA finds an embedding with optimal average distortion of distances between the original and the embedded vectors, i.e. ${A}_{\mathrm{{PCA}}} = \arg \mathop{\min }\limits_{{A \in  {\mathbb{R}}^{m \times  d}}}\mathop{\sum }\limits_{{x \in  X}}{\begin{Vmatrix}{A}^{\top }Ax - x\end{Vmatrix}}_{2}^{2}\left\lbrack  \text{Jol02}\right\rbrack$ ,whereas a JLT bounds the worst case distortion between the distances within the original space and distances within the embedded space. As for computing the transformations, a common 7 way of performing PCA is done by computing the covariance matrix and then performing eigenvalue decomposition, which results in a running time ${}^{8}$ of $O\left( {\left| X\right| {d}^{2} + {d}^{\omega }}\right)$ [DDH07],compared to $\Theta \left( {\left| X\right| d\log d}\right)$ and ${}^{9} \mid  \Theta \left( {\parallel X{\parallel }_{0}{\varepsilon }^{-1}\log \left| X\right| }\right)$ that can be achieved by the JLDs "FJLT" and "Block SparseJL", respectively, which will be introduced in section 1.4. As such, PCA and JLDs are different tools appropriate for different scenarios (see e.g. [Bre+19] where the two techniques are compared empirically in the domain of medicinal imaging; see also [Das00; BM01; FB03; FM03; Tan+05; DB06; Arp+14; Woj+16; Bre+20]). That is not to say the two are mutually exclusive, as one could apply JL to quickly shave off some dimensions followed by PCA to more carefully reduce the remaining dimensions [e.g. RST09] HMT11: Xie+16: Yan+20]. For more on PCA, we refer the interested reader to [Jol02], which provides an excellent in depth treatment of the topic.

与主成分分析（PCA）相比，约翰逊 - 林登施特劳斯变换（JLT）提供的保证有所不同：主成分分析会找到一种嵌入方式，使原始向量与嵌入向量之间距离的平均失真达到最优，即${A}_{\mathrm{{PCA}}} = \arg \mathop{\min }\limits_{{A \in  {\mathbb{R}}^{m \times  d}}}\mathop{\sum }\limits_{{x \in  X}}{\begin{Vmatrix}{A}^{\top }Ax - x\end{Vmatrix}}_{2}^{2}\left\lbrack  \text{Jol02}\right\rbrack$，而约翰逊 - 林登施特劳斯变换会限制原始空间内距离与嵌入空间内距离之间的最坏情况失真。至于计算变换，执行主成分分析的一种常见方法是计算协方差矩阵，然后进行特征值分解，这会导致运行时间${}^{8}$为$O\left( {\left| X\right| {d}^{2} + {d}^{\omega }}\right)$ [DDH07]，相比之下，约翰逊 - 林登施特劳斯降维（JLD）中的“快速约翰逊 - 林登施特劳斯变换（FJLT）”和“块稀疏约翰逊 - 林登施特劳斯变换（Block SparseJL）”分别可以达到$\Theta \left( {\left| X\right| d\log d}\right)$和${}^{9} \mid  \Theta \left( {\parallel X{\parallel }_{0}{\varepsilon }^{-1}\log \left| X\right| }\right)$的运行时间，这两种方法将在1.4节中介绍。因此，主成分分析和约翰逊 - 林登施特劳斯降维是适用于不同场景的不同工具（例如，参见[Bre+19]，其中在医学成像领域对这两种技术进行了实证比较；另见[Das00; BM01; FB03; FM03; Tan+05; DB06; Arp+14; Woj+16; Bre+20]）。这并不是说这两种方法相互排斥，因为可以先应用约翰逊 - 林登施特劳斯变换快速去除一些维度，然后再应用主成分分析更精细地减少剩余维度[例如，RST09] [HMT11: Xie+16: Yan+20]。关于主成分分析的更多内容，我们建议感兴趣的读者参考[Jol02]，该文献对该主题进行了出色的深入探讨。

---

<!-- Footnote -->

${}^{6}$ Note that the sampled transform only satisfies lemma 1.1 with some (high) probability. In the setting where we have access to the data, we can avoid this by resampling the transform until it satisfies lemma 1.1 for our specific dataset.

${}^{6}$ 请注意，采样变换仅以一定（较高）概率满足引理1.1。在我们能够获取数据的情况下，我们可以通过重新采样变换来避免这种情况，直到它针对我们的特定数据集满足引理1.1。

${}^{7}$ There are also other approaches that compute an approximate PCA more efficiently than this,e.g. [RST09 AH14|.

${}^{7}$ 还有其他一些方法比这种方法更有效地计算近似主成分分析（PCA），例如 [RST09 AH14]。

<!-- Footnote -->

---

One natural question to ask with respect to JLDs and JLTs is if the target dimension is optimal. This is indeed the case as Kane, Meka, and Nelson [KMN11] and Jayram and Woodruff [JW13] independently give a matching lower bound of $m = \Omega \left( {{\varepsilon }^{-2}\log \frac{1}{\delta }}\right)$ for any JLD that satisfies lemma 1.2, and Larsen and Nelson [LN17] showed that the bound in lemma 1.1 is optimal up constant factors for almost the entire range of $\varepsilon$ with the following theorem.

关于约翰逊 - 林登施特劳斯引理（JLDs）和约翰逊 - 林登施特劳斯变换（JLTs），一个自然会提出的问题是目标维度是否为最优。事实的确如此，因为凯恩（Kane）、梅卡（Meka）和纳尔逊（Nelson）[KMN11]以及杰拉姆（Jayram）和伍德拉夫（Woodruff）[JW13]独立地给出了任何满足引理1.2的约翰逊 - 林登施特劳斯引理的匹配下界$m = \Omega \left( {{\varepsilon }^{-2}\log \frac{1}{\delta }}\right)$，并且拉森（Larsen）和纳尔逊（Nelson）[LN17]通过以下定理表明，引理1.1中的界在常数因子范围内对于几乎整个$\varepsilon$范围都是最优的。

Theorem 1.3 ([LN17]). For any integers $n,d \geq  2$ and ${\lg }^{0.5001}n/\sqrt{\min \{ d,n\} } < \varepsilon  < 1$ there exists a set of points $X \subset  {\mathbb{R}}^{d}$ of size $n$ such that any function $f : X \rightarrow  {\mathbb{R}}^{m}$ satisfying eq. (1) must have

定理1.3（[LN17]）。对于任意整数$n,d \geq  2$和${\lg }^{0.5001}n/\sqrt{\min \{ d,n\} } < \varepsilon  < 1$，存在一个大小为$n$的点集$X \subset  {\mathbb{R}}^{d}$，使得任何满足等式(1)的函数$f : X \rightarrow  {\mathbb{R}}^{m}$都必须满足

$$
m = \Omega \left( {{\varepsilon }^{-2}\log \left( {{\varepsilon }^{2}n}\right) }\right) . \tag{3}
$$

Note that if $\varepsilon  \leq  \sqrt{\lg n/\min \{ d,n\} }$ then ${\varepsilon }^{-2}\lg n \geq  \min \{ d,n\}$ ,and embedding $X$ into dimension $\min \{ d,\left| X\right| \}$ can be done isometrically by the identity function or by projecting onto $\operatorname{span}\left( X\right)$ ,respectively.

注意，如果$\varepsilon  \leq  \sqrt{\lg n/\min \{ d,n\} }$，那么${\varepsilon }^{-2}\lg n \geq  \min \{ d,n\}$，并且分别通过恒等函数或将其投影到$\operatorname{span}\left( X\right)$上，可以等距地将$X$嵌入到维度$\min \{ d,\left| X\right| \}$中。

Alon and Klartag [AK17] extended the result in [LN17] by providing a lower bound for the gap in the range of $\varepsilon$ .

阿隆（Alon）和克拉塔尔格（Klartag）[AK17]通过给出$\varepsilon$范围内差距的下界，扩展了[LN17]中的结果。

Theorem 1.4 ([AK17]). There exists an absolute positive constant $0 < c < 1$ so that for any $n \geq  d > {cd} \geq  m$ and for all $\varepsilon  \geq  2/\sqrt{n}$ ,there exists a set of points $X \subset  {\mathbb{R}}^{d}$ of size $n$ such that any function $f : X \rightarrow  {\mathbb{R}}^{m}$ satisfying eq. (1) must have

定理1.4（[AK17]）。存在一个绝对正常数$0 < c < 1$，使得对于任何$n \geq  d > {cd} \geq  m$以及所有$\varepsilon  \geq  2/\sqrt{n}$，存在一个大小为$n$的点集$X \subset  {\mathbb{R}}^{d}$，使得任何满足等式（1）的函数$f : X \rightarrow  {\mathbb{R}}^{m}$必定满足

$$
m = \Omega \left( {{\varepsilon }^{-2}\log \left( {2 + {\varepsilon }^{2}n}\right) }\right) . \tag{4}
$$

It is, however, possible to circumvent these lower bounds by restricting the set of input vectors we apply the JLTs to. For instance, Klartag and Mendelson [KM05], Dirksen [Dir16], and Bourgain, Dirksen, and Nelson [BDN15b] provide target dimension upper bounds for JLTs that are dependent on statistical properties of the input set $X$ . Similarly,JLTs can be used to approximately preserve pairwise distances simultaneously for an entire subspace using $m = \Theta \left( {{\varepsilon }^{-2}t\log \left( {t/\varepsilon }\right) }\right)$ ,where $t$ denotes the dimension of the subspace [Sar06],which is a great improvement when $t \ll  \left| X\right| ,d$ .

然而，通过限制我们对其应用约翰逊 - 林登施特劳斯引理（JLTs）的输入向量集，可以规避这些下界。例如，克拉尔塔格（Klartag）和门德尔松（Mendelson）[KM05]、迪克森（Dirksen）[Dir16]，以及布尔甘（Bourgain）、迪克森（Dirksen）和纳尔逊（Nelson）[BDN15b]为依赖于输入集统计特性 $X$ 的约翰逊 - 林登施特劳斯引理（JLTs）提供了目标维度上界。类似地，使用 $m = \Theta \left( {{\varepsilon }^{-2}t\log \left( {t/\varepsilon }\right) }\right)$，约翰逊 - 林登施特劳斯引理（JLTs）可用于同时近似保留整个子空间的成对距离，其中 $t$ 表示子空间的维度 [Sar06]，当 $t \ll  \left| X\right| ,d$ 时，这是一个很大的改进。

Another useful property of JLTs is that they approximately preserve dot products. Corollary 1.5 formalises this property in terms of lemma 1.1, though it is sometimes [Sar06, AV06] stated in terms of lemma 1.2. Corollary 1.5 has a few extra requirements on $f$ and $X$ compared to lemma 1.1, but these are not an issue if the JLT is sampled from a JLD, or if we add the negations of all our vectors to $X$ ,which only slightly increases the target dimension.

约翰逊 - 林登施特劳斯变换（JLT）的另一个有用性质是，它们能近似保留点积。推论1.5依据引理1.1对这一性质进行了形式化表述，不过有时[Sar06, AV06]会依据引理1.2来表述。与引理1.1相比，推论1.5对$f$和$X$有一些额外要求，但如果约翰逊 - 林登施特劳斯变换（JLT）是从约翰逊 - 林登施特劳斯分布（JLD）中采样得到的，或者如果我们将所有向量的负向量添加到$X$中（这只会略微增加目标维度），那么这些要求就不成问题。

---

<!-- Footnote -->

${}^{8}$ Here $\omega  \lesssim  {2.373}$ is the exponent from the running time of squared matrix multiplication [Wil12,Le 14].

${}^{8}$ 这里的$\omega  \lesssim  {2.373}$是来自方阵乘法运行时间的指数[Wil12,Le 14]。

${}^{9}$ Here $\parallel X{\parallel }_{0}$ is the total number of nonzero entries in the set of vectors $X$ ,i.e. $\parallel X{\parallel }_{0} \mathrel{\text{:=}} \mathop{\sum }\limits_{{x \in  X}}\parallel x{\parallel }_{0}$ where $\parallel x{\parallel }_{0} \mathrel{\text{:=}} \left| \left\{  {i \mid  {x}_{i} \neq  0}\right\}  \right|$ for a vector $x$ .

${}^{9}$ 这里 $\parallel X{\parallel }_{0}$ 是向量集 $X$ 中非零元素的总数，即 $\parallel X{\parallel }_{0} \mathrel{\text{:=}} \mathop{\sum }\limits_{{x \in  X}}\parallel x{\parallel }_{0}$ ，其中对于向量 $x$ 有 $\parallel x{\parallel }_{0} \mathrel{\text{:=}} \left| \left\{  {i \mid  {x}_{i} \neq  0}\right\}  \right|$ 。

<!-- Footnote -->

---

Corollary 1.5. Let $d,\varepsilon ,X$ and $f$ be as defined in lemma 1.1,and furthermore let $f$ be linear. Then for every $x,y \in  X$ ,if $- y \in  X$ then

推论1.5。设 $d,\varepsilon ,X$ 和 $f$ 如引理1.1中所定义，此外，设 $f$ 是线性的。那么对于每个 $x,y \in  X$ ，如果 $- y \in  X$ ，则

$$
\left| {\langle f\left( x\right) ,f\left( y\right) \rangle -\langle x,y\rangle }\right|  \leq  \varepsilon \parallel x{\parallel }_{2}\parallel y{\parallel }_{2}. \tag{5}
$$

Proof. If at least one of $x$ and $y$ is the 0 -vector,then eq. (5) is trivially satisfied as $f$ is linear. If $x$ and $y$ are both unit vectors then we assume w.l.o.g. that $\parallel x + y{\parallel }_{2} \geq  \parallel x - y{\parallel }_{2}$ and we proceed as follows,utilising the polarisation identity: $4\langle u,v\rangle  = \parallel u + v{\parallel }_{2}^{2} - \parallel u - v{\parallel }_{2}^{2}$ .

证明。如果 $x$ 和 $y$ 中至少有一个是零向量，那么由于 $f$ 是线性的，等式 (5) 显然成立。如果 $x$ 和 $y$ 都是单位向量，那么不失一般性，我们假设 $\parallel x + y{\parallel }_{2} \geq  \parallel x - y{\parallel }_{2}$，然后利用极化恒等式 $4\langle u,v\rangle  = \parallel u + v{\parallel }_{2}^{2} - \parallel u - v{\parallel }_{2}^{2}$ 按如下步骤进行。

$$
4\left| {\langle f\left( x\right) ,f\left( y\right) \rangle -\langle x,y\rangle }\right|  = \left| {\parallel f\left( x\right)  + f\left( y\right) {\parallel }_{2}^{2} - \parallel f\left( x\right)  - f\left( y\right) {\parallel }_{2}^{2} - 4\langle x,y\rangle }\right| 
$$

$$
 \leq  \left| {\left( {1 + \varepsilon }\right) \parallel x + y{\parallel }_{2}^{2} - \left( {1 - \varepsilon }\right) \parallel x - y{\parallel }_{2}^{2} - 4\langle x,y\rangle }\right| 
$$

$$
 = \left| {4\langle x,y\rangle  + \varepsilon \left( {\parallel x + y{\parallel }_{2}^{2} + \parallel x - y{\parallel }_{2}^{2}}\right)  - 4\langle x,y\rangle }\right| 
$$

$$
 = \varepsilon \left( {2\parallel x{\parallel }_{2}^{2} + 2\parallel y{\parallel }_{2}^{2}}\right) 
$$

$$
 = {4\varepsilon }\text{.}
$$

Otherwise we can reduce to the unit vector case.

否则，我们可以将其简化为单位向量的情况。

$$
\left| {\langle f\left( x\right) ,f\left( y\right) \rangle -\langle x,y\rangle }\right|  = \left| {\left\langle  {f\left( \frac{x}{\parallel x{\parallel }_{2}}\right) ,f\left( \frac{y}{\parallel y{\parallel }_{2}}\right) }\right\rangle   - \left\langle  {\frac{x}{\parallel x{\parallel }_{2}},\frac{y}{\parallel y{\parallel }_{2}}}\right\rangle  }\right| \parallel x{\parallel }_{2}\parallel y{\parallel }_{2}
$$

$$
 \leq  \varepsilon \parallel x{\parallel }_{2}\parallel y{\parallel }_{2}.
$$

Before giving an overview of the development of JLDs in section 1.4, let us return to our scenario and example in section 1.1 and show the wide variety of fields where dimensionality reduction via JLTs have found use. Furthermore, to make us more familiar with lemma 1.1 and its related concepts, we will pick a few examples of how the lemma is used.

在第 1.4 节概述联合低维嵌入（JLDs）的发展之前，让我们回到第 1.1 节的场景和示例，展示通过约翰逊 - 林登施特劳斯引理（JLTs）进行降维已得到应用的广泛领域。此外，为了让我们更熟悉引理 1.1 及其相关概念，我们将选取几个该引理的应用示例。

### 1.3 The Use(fulness) of Johnson-Lindenstrauss

### 1.3 约翰逊 - 林登斯特劳斯（Johnson - Lindenstrauss）的用途（实用性）

JLDs and JLTs have found uses and parallels in many fields and tasks, some of which we will list below. Note that there are some overlap between the following categories, as e.g. [FB03] uses a JLD for an ensemble of weak learners to learn a mixture of Gaussians clustering, and [PW15] solves convex optimisation problems in a way that gives differential privacy guarantees.

约翰逊 - 林登斯特劳斯变换（JLDs）和约翰逊 - 林登斯特劳斯引理（JLTs）在许多领域和任务中得到了应用，并存在相似之处，下面我们将列举其中一些。请注意，以下类别之间存在一些重叠，例如，文献[FB03]使用约翰逊 - 林登斯特劳斯变换（JLD）为一组弱学习器学习高斯混合聚类，而文献[PW15]以一种能提供差分隐私保证的方式解决凸优化问题。

Nearest-neighbour search have benefited from the Johnson-Lindenstrauss lemmas on multiple occasions, including [Kle97; KOR00], which used JL to randomly partition space rather than reduce the dimension, while others [AC09; HIM12] used the dimensionality reduction properties of JL more directly. Variations on these results include consructing locality sensitive hashing schemes [Dat+04] and finding nearest neighbours without false negatives [SW17].

最近邻搜索多次受益于约翰逊 - 林登斯特劳斯引理，包括文献[Kle97; KOR00]，它们使用约翰逊 - 林登斯特劳斯引理（JL）对空间进行随机划分而非降维，而其他文献[AC09; HIM12]则更直接地利用了约翰逊 - 林登斯特劳斯引理（JL）的降维特性。这些结果的变体包括构建局部敏感哈希方案[Dat+04]以及在无漏检的情况下查找最近邻[SW17]。

Clustering with results in various sub-areas such as mixture of Gaussians [Das99; FB03; UDS07], subspace clustering [HTB17], graph clustering [SI09; Guo+20], self-organising maps [RK89; Kas98], and k-means [Bec+19; Coh+15; Bou+14; Liu+17; SF18], which will be explained in more detail in section 1.3.1.

聚类在多个子领域产生了成果，如高斯混合模型（mixture of Gaussians）[Das99; FB03; UDS07]、子空间聚类（subspace clustering）[HTB17]、图聚类（graph clustering）[SI09; Guo+20]、自组织映射（self-organising maps）[RK89; Kas98]和k均值聚类（k-means）[Bec+19; Coh+15; Bou+14; Liu+17; SF18]，这些将在1.3.1节中详细解释。

Outlier detection where there have been works for various settings of outliers, including approximate nearest-neighbours [dVCH10; SZK15] and Gaussian vectors [NC20], while Zha+20 uses JL as a preprocessor for a range of outlier detection algorithms in a distributed computational model, and [AP12] evaluates the use of JLTs for outlier detection of text documents.

离群点检测方面，针对各种离群点设置已有相关研究，包括近似最近邻（approximate nearest-neighbours）[dVCH10; SZK15]和高斯向量（Gaussian vectors）[NC20]。同时，Zha等人在2020年的研究（Zha+20）将约翰逊 - 林登斯特劳斯引理（JL）用作分布式计算模型中一系列离群点检测算法的预处理器，而[AP12]评估了约翰逊 - 林登斯特劳斯变换（JLTs）在文本文档离群点检测中的应用。

Ensemble learning where independent JLTs can be used to generate training sets for weak learners for bagging [SR09] and with the voting among the learners weighted by how well a given JLT projects the data [CS17; Can20]. The combination of JLTs with multiple learners have also found use in the regime of learning high-dimensional distributions from few datapoints (i.e. $\left| X\right|  \ll  d$ ) [DK13; ZK19; Niy+20].

集成学习中，独立的约翰逊 - 林登斯特劳斯变换（Johnson - Lindenstrauss Transform，JLT）可用于为装袋法（bagging）的弱学习器生成训练集[SR09]，并且学习器之间的投票会根据给定的JLT对数据的投影效果进行加权[CS17; Can20]。JLT与多个学习器的结合也被用于从少量数据点中学习高维分布的场景（即$\left| X\right|  \ll  d$）[DK13; ZK19; Niy+20]。

Adversarial machine learning where Johnson-Lindenstrauss can both be used to defend against adversarial input $\left\lbrack  {\mathrm{{Ngu}} + {16};\text{Wee} + {19};\text{Tar} + {19}}\right\rbrack$ as well as help craft such attacks [Li+20].

对抗式机器学习中，约翰逊 - 林登斯特劳斯变换（Johnson - Lindenstrauss）既可以用于防御对抗性输入$\left\lbrack  {\mathrm{{Ngu}} + {16};\text{Wee} + {19};\text{Tar} + {19}}\right\rbrack$，也可以帮助设计此类攻击[Li+20]。

Miscellaneous machine learning where, in addition to the more specific machine learning topics mentioned above, Johnson-Lindenstrauss has been used together with support vector machines [CJS09; Pau+14; LL20], Fisher's linear discriminant [DK10], and neural networks [Sch18], while [KY20] uses JL to facilitate stochastic gradient descent in a distributed setting.

其他机器学习领域中，除了上述更具体的机器学习主题外，约翰逊 - 林登斯特劳斯引理（Johnson - Lindenstrauss）已与支持向量机[CJS09; Pau + 14; LL20]、费舍尔线性判别法（Fisher's linear discriminant）[DK10]和神经网络[Sch18]结合使用，而[KY20]则使用JL引理来促进分布式环境中的随机梯度下降。

Numerical linear algebra with work focusing on low rank approximation [Coh+15; MM20], canonical correlation analysis [Avr+14], and regression in a local [THM17; MM09; Kab14; Sla17] and a distributed [HMM16] computational model. Futhermore, as many of these subfields are related some papers tackle multiple numerical linear algebra problems at once, e.g. low rank approximation, regression, and approximate matrix multiplication [Sar06], and a line of work [MM13; CW17; NN13a] have used JLDs to perform subspace embeddings which in turn gives algorithms for ${\ell }_{p}$ regression,low rank approximation,and leverage scores.

数值线性代数领域的研究主要集中在低秩近似[Coh + 15; MM20]、典型相关分析（canonical correlation analysis）[Avr + 14]，以及局部[THM17; MM09; Kab14; Sla17]和分布式[HMM16]计算模型中的回归问题。此外，由于这些子领域相互关联，一些论文会同时处理多个数值线性代数问题，例如低秩近似、回归和近似矩阵乘法[Sar06]，并且一系列研究[MM13; CW17; NN13a]使用约翰逊 - 林登斯特劳斯嵌入（JLDs）进行子空间嵌入，进而得到了用于${\ell }_{p}$回归、低秩近似和杠杆得分的算法。

For further reading, there are surveys [Mah11; HMT11; Woo14] covering much of JLDs' use in numerical linear algebra.

如需进一步阅读，有综述文献[Mah11; HMT11; Woo14]涵盖了约翰逊 - 林登斯特劳斯变换（Johnson - Lindenstrauss Transform，JLD）在数值线性代数中的大部分应用。

Convex optimisation in which Johnson-Lindenstrauss has been used for (integer) linear programming [VPL15] and to improve a cutting plane method for finding a point in a convex set using a separation oracle [TSC15; Jia+20]. Additionally, [Zha+13] studies how to recover a high-dimensional optimisation solution from a JL dimensionality reduced one.

凸优化领域中，约翰逊 - 林登斯特劳斯变换已被用于（整数）线性规划[VPL15]，并用于改进一种使用分离预言机在凸集中寻找点的割平面法[TSC15; Jia + 20]。此外，文献[Zha + 13]研究了如何从经约翰逊 - 林登斯特劳斯降维后的解中恢复高维优化解。

Differential privacy have utilised Johnson-Lindenstrauss to provide sanitised solutions to the linear algebra problems of variance estimation [Blo+12], regression [She19; SKD19;

差分隐私利用约翰逊 - 林登斯特劳斯变换为方差估计[Blo + 12]、回归[She19; SKD19;

ZLW09], Euclidean distance estimation [Ken+13: LKR06: GLK13: Tur+08: Xu+17], and low-rank factorisation [Upa18], as well as convex optimisation [PW15; KJ16], collaborative filtering [Yan+17] and solutions to graph-cut queries [Blo+12; Upa13]. Furthermore, [Upa15] analysis various JLDs with respect to differential privacy and introduces a novel one designed for this purpose.

ZLW09]、欧几里得距离估计[Ken + 13: LKR06: GLK13: Tur + 08: Xu + 17]和低秩分解[Upa18]等线性代数问题提供净化后的解，以及用于凸优化[PW15; KJ16]、协同过滤[Yan + 17]和图割查询的解[Blo + 12; Upa13]。此外，文献[Upa15]针对差分隐私分析了各种约翰逊 - 林登斯特劳斯变换，并为此目的引入了一种新颖的变换。

Neuroscience where it is used as a tool to process data in computational neuroscience [GS12] ALG13], but also as a way of modelling neurological processes [GS12: ALG13: All+14: PP14]. Interestingly, there is some evidence [MFL08; SA09; Car+13] to suggest that JL-like operations occur in nature, as a large set of olifactory sensory inputs (projection neurons) map onto a smaller set of neurons (Kenyon cells) in the brains of fruit flies, where each Kenyon cell is connected to a small and seemingly random subset of the projection neurons. This is reminiscent of sparse JL constructions, which will be introduced in section 1.4.1. though I am not neuroscientifically adept enough to judge how far these similarities between biological constructs and randomised linear algebra extend.

神经科学领域，它被用作计算神经科学中处理数据的工具[GS12][ALG13]，同时也作为对神经过程进行建模的一种方式[GS12; ALG13; All+14; PP14]。有趣的是，有一些证据[MFL08; SA09; Car+13]表明，类似约翰逊 - 林登斯特劳斯（Johnson - Lindenstrauss，JL）的操作在自然界中也会出现，因为果蝇大脑中大量的嗅觉感觉输入（投射神经元）会映射到数量较少的一组神经元（肯扬细胞）上，其中每个肯扬细胞都与投射神经元的一个小的、看似随机的子集相连。这让人想起稀疏的JL构造，相关内容将在1.4.1节中介绍。不过，我在神经科学方面的能力还不足以判断生物构造与随机线性代数之间的这些相似性能延伸到何种程度。

Other topics where Johnson-Lindenstrauss have found use include graph sparsification [SS11], graph embeddings in Euclidean spaces [FM88], integrated circuit design [Vem98], biometric authentication [Arp+14], and approximating minimum spanning trees [HI00].

约翰逊 - 林登斯特劳斯（Johnson - Lindenstrauss）已得到应用的其他主题包括图稀疏化[SS11]、图在欧几里得空间中的嵌入[FM88]、集成电路设计[Vem98]、生物特征认证[Arp+14]以及近似最小生成树[HI00]。

For further examples of Johnson-Lindenstrauss use cases, please see [Ind01: Vem04].

有关约翰逊 - 林登斯特劳斯（Johnson - Lindenstrauss）用例的更多示例，请参阅[Ind01; Vem04]。

Now, let us dive deeper into the areas of clustering and streaming algorithms to see how Johnson-Lindenstrauss can be used there.

现在，让我们更深入地探讨聚类和流算法领域，看看约翰逊 - 林登斯特劳斯（Johnson - Lindenstrauss）引理在这些领域如何应用。

#### 1.3.1 Clustering

#### 1.3.1 聚类

Clustering can be defined as partitioning a dataset such that elements are similar to elements in the same partition while being dissimilar to elements in other partitions. A classic clustering problem is the so-called $k$ -means clustering where the dataset $X \subset  {\mathbb{R}}^{d}$ consists of points in Euclidean space. The task is to choose $k$ cluster centers ${c}_{1},\ldots ,{c}_{k}$ such that they minimise the sum of squared distances from datapoints to their nearest cluster center, i.e.

聚类可以定义为对数据集进行划分，使得同一分区内的元素彼此相似，而与其他分区内的元素不同。一个经典的聚类问题是所谓的 $k$ - 均值聚类，其中数据集 $X \subset  {\mathbb{R}}^{d}$ 由欧几里得空间中的点组成。任务是选择 $k$ 个聚类中心 ${c}_{1},\ldots ,{c}_{k}$，使得数据点到其最近聚类中心的距离平方和最小，即

$$
\underset{{c}_{1},\ldots ,{c}_{k}}{\arg \min }\mathop{\sum }\limits_{{x \in  X}}\mathop{\min }\limits_{i}{\begin{Vmatrix}x - {c}_{i}\end{Vmatrix}}_{2}^{2} \tag{6}
$$

This creates a Voronoi partition, as each datapoint is assigned to the partition corresponding to its nearest cluster center. We let ${X}_{i} \subseteq  X$ denote the set of points that have ${c}_{i}$ as their closest center. It is well known that for an optimal choice of centers, the centers are the means of their corresponding partitions, and furthermore, the cost of any choice of centers is never lower than the sum of squared distances from datapoints to the mean of their assigned partition, i.e.

这会创建一个沃罗诺伊分割（Voronoi partition），因为每个数据点都会被分配到与其最近的聚类中心相对应的分割中。我们用 ${X}_{i} \subseteq  X$ 表示那些以 ${c}_{i}$ 为其最近中心的点的集合。众所周知，对于中心的最优选择，这些中心是其对应分割的均值，此外，任何中心选择的代价都不会低于数据点到其分配分割的均值的距离平方和，即

$$
\mathop{\sum }\limits_{{x \in  X}}\mathop{\min }\limits_{i}{\begin{Vmatrix}x - {c}_{i}\end{Vmatrix}}_{2}^{2} \geq  \mathop{\sum }\limits_{{i = 1}}^{k}\mathop{\sum }\limits_{{x \in  {X}_{i}}}{\begin{Vmatrix}x - \frac{1}{\left| {X}_{i}\right| }\mathop{\sum }\limits_{{y \in  {X}_{i}}}y\end{Vmatrix}}_{2}^{2}. \tag{7}
$$

It has been shown that finding the optimal centers,even for $k = 2$ ,is NP-hard Alo+09; Das08]; however, various heuristic approaches have found success such as the commonly used Lloyd's algorithm [Llo82]. In Lloyd's algorithm, after initialising the centers in some way we iteratively improve the choice of centers by assigning each datapoint to its nearest center and then updating the center to be the mean of the datapoints assigned to it. These two steps can then be repeated until some termination criterion is met, e.g. when the centers have converged. If we let $t$ denote the number of iterations,then the running time becomes $O\left( {t\left| X\right| {kd}}\right)$ ,as we use $O\left( {\left| X\right| {kd}}\right)$ time per iteration to assign each data point to its nearest center. We can improve this running time by quickly embedding the datapoints into a lower dimensional space using a JLT and then running Lloyd's algorithm in this smaller space. The Fast Johnson-Lindenstrauss Transform, which we will introduce later, can for many sets of parameters embed a vector in $O\left( {d\log d}\right)$ time reducing the total running time to $O\left( {\left| X\right| d\log d + t\left| X\right| k{\varepsilon }^{-2}\log \left| X\right| }\right)$ . However,for this to be useful we need the partitioning of points in the lower dimensional space to correspond to an (almost) equally good partition in the original higher dimensional space.

研究表明，即使对于$k = 2$，寻找最优中心也是NP难问题[阿洛等人09年；达斯08年]；然而，各种启发式方法已取得成功，例如常用的劳埃德算法[劳埃德82年]。在劳埃德算法中，以某种方式初始化中心后，我们通过将每个数据点分配给其最近的中心，然后将中心更新为分配给它的数据点的均值，来迭代地改进中心的选择。然后可以重复这两个步骤，直到满足某个终止条件，例如当中心收敛时。如果我们用$t$表示迭代次数，那么运行时间变为$O\left( {t\left| X\right| {kd}}\right)$，因为我们每次迭代使用$O\left( {\left| X\right| {kd}}\right)$的时间将每个数据点分配给其最近的中心。我们可以通过使用约翰逊 - 林登施特劳斯变换（JLT）将数据点快速嵌入到低维空间，然后在这个较小的空间中运行劳埃德算法来改善这个运行时间。我们稍后将介绍的快速约翰逊 - 林登施特劳斯变换，对于许多参数集，可以在$O\left( {d\log d}\right)$的时间内嵌入一个向量，从而将总运行时间减少到$O\left( {\left| X\right| d\log d + t\left| X\right| k{\varepsilon }^{-2}\log \left| X\right| }\right)$。然而，要使这一方法有用，我们需要低维空间中点的划分与原始高维空间中（几乎）同样好的划分相对应。

In order to prove such a result we will use the following lemma, which shows that the cost of a partitioning, with its centers chosen as the means of the partitions, can be written in terms of pairwise distances between datapoints in the partitions.

为了证明这一结果，我们将使用以下引理，该引理表明，若将分区的中心选为各分区的均值，则分区的代价可以用分区中数据点之间的成对距离来表示。

Lemma 1.6. Let $k,d \in  {\mathbb{N}}_{1}$ and ${X}_{i} \subset  {\mathbb{R}}^{d}$ for $\overline{10}i \in  \left\lbrack  k\right\rbrack$ .

引理1.6。设$k,d \in  {\mathbb{N}}_{1}$且对于$\overline{10}i \in  \left\lbrack  k\right\rbrack$有${X}_{i} \subset  {\mathbb{R}}^{d}$。

$$
\mathop{\sum }\limits_{{i = 1}}^{k}\mathop{\sum }\limits_{{x \in  {X}_{i}}}{\begin{Vmatrix}x - \frac{1}{\left| {X}_{i}\right| }\mathop{\sum }\limits_{{y \in  {X}_{i}}}y\end{Vmatrix}}_{2}^{2} = \frac{1}{2}\mathop{\sum }\limits_{{i = 1}}^{k}\frac{1}{\left| {X}_{i}\right| }\mathop{\sum }\limits_{{x,y \in  {X}_{i}}}\parallel x - y{\parallel }_{2}^{2}. \tag{8}
$$

The proof of lemma 1.6 consists of various linear algebra manipulations and can be found in section 2.1. Now we are ready to prove the following proposition, which states that if we find a partitioning whose cost is within $\left( {1 + \gamma }\right)$ of the optimal cost in low dimensional space,that partitioning when moving back to the high dimensional space is within $\left( {1 + {4\varepsilon }}\right) \left( {1 + \gamma }\right)$ of the optimal cost there.

引理1.6的证明涉及各种线性代数运算，具体证明过程可在2.1节中找到。现在我们准备证明以下命题，该命题指出，如果我们在低维空间中找到一个代价在最优代价的$\left( {1 + \gamma }\right)$范围内的分区，那么当将该分区移回到高维空间时，其代价在高维空间最优代价的$\left( {1 + {4\varepsilon }}\right) \left( {1 + \gamma }\right)$范围内。

Proposition 1.7. Let $k,d \in  {\mathbb{N}}_{1},X \subset  {\mathbb{R}}^{d},\varepsilon  \leq  1/2,m = \Theta \left( {{\varepsilon }^{-2}\log \left| X\right| }\right)$ ,and $f : X \rightarrow  {\mathbb{R}}^{m}$ be a JLT. Let $Y \subset  {\mathbb{R}}^{m}$ be the embedding of $X$ . Let ${\kappa }_{m}^{ * }$ denote the optimal cost of a partitioning of $Y$ ,with respect to eq. (6). Let ${Y}_{1},\ldots ,{Y}_{k} \subseteq  Y$ be a partitioning of $Y$ with cost ${\kappa }_{m}$ such that ${\kappa }_{m} \leq  \left( {1 + \gamma }\right) {\kappa }_{m}^{ * }$ for some $\gamma  \in  \mathbb{R}$ . Let ${\kappa }_{d}^{ * }$ be the cost of an optimal partitioning of $X$ and ${\kappa }_{d}$ be the cost of the partitioning ${X}_{1},\ldots ,{X}_{k} \subseteq  X$ , satisfying ${Y}_{i} = \left\{  {f\left( x\right)  \mid  x \in  {X}_{i}}\right\}$ . Then

命题1.7。设$k,d \in  {\mathbb{N}}_{1},X \subset  {\mathbb{R}}^{d},\varepsilon  \leq  1/2,m = \Theta \left( {{\varepsilon }^{-2}\log \left| X\right| }\right)$ ，且$f : X \rightarrow  {\mathbb{R}}^{m}$ 为一个JLT（联合线性变换，Joint Linear Transformation）。设$Y \subset  {\mathbb{R}}^{m}$ 为$X$ 的嵌入。设${\kappa }_{m}^{ * }$ 表示关于等式(6)对$Y$ 进行划分的最优成本。设${Y}_{1},\ldots ,{Y}_{k} \subseteq  Y$ 为$Y$ 的一个成本为${\kappa }_{m}$ 的划分，使得对于某个$\gamma  \in  \mathbb{R}$ 有${\kappa }_{m} \leq  \left( {1 + \gamma }\right) {\kappa }_{m}^{ * }$ 。设${\kappa }_{d}^{ * }$ 为对$X$ 进行最优划分的成本，${\kappa }_{d}$ 为划分${X}_{1},\ldots ,{X}_{k} \subseteq  X$ 的成本，满足${Y}_{i} = \left\{  {f\left( x\right)  \mid  x \in  {X}_{i}}\right\}$ 。那么

$$
{\kappa }_{d} \leq  \left( {1 + {4\varepsilon }}\right) \left( {1 + \gamma }\right) {\kappa }_{d}^{ * }. \tag{9}
$$

Proof. Due to lemma 1.6 and the fact that $f$ is a JLT we know that the cost of our partitioning is approximately preserved when going back to the high dimensional space,i.e. ${\kappa }_{d} \leq  {\kappa }_{m}/\left( {1 - \varepsilon }\right)$ . Furthermore,since the cost of ${X}^{\prime }$ s optimal partitioning when embedded down to $Y$ cannot be lower than the optimal cost of partitioning $Y$ ,we can conclude ${\kappa }_{m}^{ * } \leq  \left( {1 + \varepsilon }\right) {\kappa }_{d}^{ * }$ . Since $\varepsilon  \leq  1/2$ , we have $1/\left( {1 - \varepsilon }\right)  = 1 + \varepsilon /\left( {1 - \varepsilon }\right)  \leq  1 + {2\varepsilon }$ and also $\left( {1 + \varepsilon }\right) \left( {1 + {2\varepsilon }}\right)  = \left( {1 + {3\varepsilon } + 2{\varepsilon }^{2}}\right)  \leq  1 + {4\varepsilon }$ .

证明。根据引理1.6以及$f$是一个约翰逊 - 林登施特劳斯变换（Johnson - Lindenstrauss Transform，JLT）这一事实，我们知道当回到高维空间时，我们的划分成本大致保持不变，即${\kappa }_{d} \leq  {\kappa }_{m}/\left( {1 - \varepsilon }\right)$。此外，由于${X}^{\prime }$嵌入到$Y$时的最优划分成本不能低于划分$Y$的最优成本，我们可以得出${\kappa }_{m}^{ * } \leq  \left( {1 + \varepsilon }\right) {\kappa }_{d}^{ * }$。因为$\varepsilon  \leq  1/2$，我们有$1/\left( {1 - \varepsilon }\right)  = 1 + \varepsilon /\left( {1 - \varepsilon }\right)  \leq  1 + {2\varepsilon }$，并且还有$\left( {1 + \varepsilon }\right) \left( {1 + {2\varepsilon }}\right)  = \left( {1 + {3\varepsilon } + 2{\varepsilon }^{2}}\right)  \leq  1 + {4\varepsilon }$。

---

<!-- Footnote -->

${}^{10}$ We use $\left\lbrack  k\right\rbrack$ to denote the set $\{ 1,\ldots ,k\}$ .

${}^{10}$ 我们使用 $\left\lbrack  k\right\rbrack$ 来表示集合 $\{ 1,\ldots ,k\}$。

<!-- Footnote -->

---

Combining these inequalities we get

将这些不等式组合起来，我们得到

$$
{\kappa }_{d} \leq  \frac{1}{1 - \varepsilon }{\kappa }_{m}
$$

$$
 \leq  \left( {1 + {2\varepsilon }}\right) \left( {1 + \gamma }\right) {\kappa }_{m}^{ * }
$$

$$
 \leq  \left( {1 + {2\varepsilon }}\right) \left( {1 + \gamma }\right) \left( {1 + \varepsilon }\right) {\kappa }_{d}^{ * }
$$

$$
 \leq  \left( {1 + {4\varepsilon }}\right) \left( {1 + \gamma }\right) {\kappa }_{d}^{ * }\text{.}
$$

By pushing the constant inside the $\Theta$ -notation,proposition 1.7 shows that we can achieve a $\left( {1 + \varepsilon }\right)$ approximation ${}^{11}$ of $k$ -means with $m = \Theta \left( {{\varepsilon }^{-2}\log \left| X\right| }\right)$ . However,by more carefully analysing which properties are needed,we can improve upon this for the case where $k \ll  \left| X\right|$ . Boutsidis et al. [Bou+14] showed that projecting down to a target dimension of $m = \Theta \left( {{\varepsilon }^{-2}k}\right)$ suffices for a slightly worse $k$ -means approximation factor of $\left( {2 + \varepsilon }\right)$ . This result was expanded upon in two ways by Cohen et al. [Coh+15],who showed that projecting down to $m = \Theta \left( {{\varepsilon }^{-2}k}\right)$ achieves a $\left( {1 + \varepsilon }\right)$ approximation ratio,while projecting all the way down to $m = \Theta \left( {{\varepsilon }^{-2}\log k}\right)$ still suffices for a $\left( {9 + \varepsilon }\right)$ approximation ratio. The $\left( {1 + \varepsilon }\right)$ case has recently been further improved upon by both Becchetti et al. [Bec+19],who have shown that one can achieve the $\left( {1 + \varepsilon }\right)$ approximation ratio for $k$ -means when projecting down to $m = \Theta \left( {{\varepsilon }^{-6}\left( {\log k + \log \log \left| X\right| }\right) \log {\varepsilon }^{-1}}\right)$ ,and by Makarychev, Makarychev, and Razenshteyn [MMR19], who independently have proven an even better bound of $m = \Theta \left( {{\varepsilon }^{-2}\log k/\varepsilon }\right)$ ,essentially giving a "best of worlds" result with respect to [Coh+15].

通过将常数移到$\Theta$符号内，命题1.7表明，我们可以用$m = \Theta \left( {{\varepsilon }^{-2}\log \left| X\right| }\right)$实现对$k$ -均值的$\left( {1 + \varepsilon }\right)$近似（${}^{11}$）。然而，通过更仔细地分析所需的性质，我们可以在$k \ll  \left| X\right|$的情况下对这一结果进行改进。布蒂西迪斯等人（Boutsidis et al. [Bou+14]）表明，将维度投影到目标维度$m = \Theta \left( {{\varepsilon }^{-2}k}\right)$足以实现稍差的$k$ -均值近似因子$\left( {2 + \varepsilon }\right)$。科恩等人（Cohen et al. [Coh+15]）从两个方面扩展了这一结果，他们表明，投影到$m = \Theta \left( {{\varepsilon }^{-2}k}\right)$可实现$\left( {1 + \varepsilon }\right)$近似比，而一直投影到$m = \Theta \left( {{\varepsilon }^{-2}\log k}\right)$仍足以实现$\left( {9 + \varepsilon }\right)$近似比。最近，贝凯蒂等人（Becchetti et al. [Bec+19]）进一步改进了$\left( {1 + \varepsilon }\right)$的情况，他们表明，当投影到$m = \Theta \left( {{\varepsilon }^{-6}\left( {\log k + \log \log \left| X\right| }\right) \log {\varepsilon }^{-1}}\right)$时，可以实现$k$ -均值的$\left( {1 + \varepsilon }\right)$近似比；马卡里切夫、马卡里切夫和拉赞施泰因（Makarychev, Makarychev, and Razenshteyn [MMR19]）也独立证明了一个更好的界$m = \Theta \left( {{\varepsilon }^{-2}\log k/\varepsilon }\right)$，本质上相对于[Coh+15]给出了一个“两全其美”的结果。

For an overview of the history of $k$ -means clustering,we refer the interested reader to Boc08].

关于$k$均值聚类（$k$ -means clustering）的历史概述，我们建议感兴趣的读者参考[Boc08]。

#### 1.3.2 Streaming

#### 1.3.2 流式处理

The field of streaming algorithms is characterised by problems where we receive a sequence (or stream) of items and are queried on the items received so far. The main constraint is usually that we only have limited access to the sequence, e.g. that we are only allowed one pass over it, and that we have very limited space, e.g. polylogarithmic in the length of the stream. To make up for these constraints we are allowed to give approximate answers to the queries. The subclass of streaming problems we will look at here are those where we are only allowed a single pass over the sequence and the items are updates to a vector and a query is some statistic on that vector, e.g. the ${\ell }_{2}$ norm of the vector. More formally,and to introduce the notation,let $d \in  {\mathbb{N}}_{1}$ be the number of different items and let $T \in  {\mathbb{N}}_{1}$ be the length of the stream of updates $\left( {{i}_{j},{v}_{j}}\right)  \in  \left\lbrack  d\right\rbrack   \times  \mathbb{R}$ for $j \in  \left\lbrack  T\right\rbrack$ ,and define the vector $x$ at time $t$ as ${x}^{\left( t\right) } \mathrel{\text{:=}} \mathop{\sum }\limits_{{j = 1}}^{t}{v}_{j}{e}_{{i}_{j}}$ . A query $q$ at time $t$ is then a function of ${x}^{\left( t\right) }$ ,and we will omit the ${}^{\left( t\right) }$ superscript when referring to the current time.

流算法领域的特点在于处理这样一类问题：我们会接收到一个项目序列（或流），并需要对目前已接收的项目进行查询。主要限制通常是我们只能有限地访问该序列，例如，只允许对其进行一次遍历，并且我们的可用空间非常有限，例如，空间复杂度为流长度的多项式对数级。为了弥补这些限制，我们可以对查询给出近似答案。我们在这里要研究的流问题子类是那些只允许对序列进行一次遍历的问题，其中项目是对向量的更新，而查询是关于该向量的某种统计信息，例如向量的${\ell }_{2}$范数。更正式地说，为了引入相关符号，设$d \in  {\mathbb{N}}_{1}$为不同项目的数量，设$T \in  {\mathbb{N}}_{1}$为更新流$\left( {{i}_{j},{v}_{j}}\right)  \in  \left\lbrack  d\right\rbrack   \times  \mathbb{R}$（其中$j \in  \left\lbrack  T\right\rbrack$ ）的长度，并将时间$t$时的向量$x$定义为${x}^{\left( t\right) } \mathrel{\text{:=}} \mathop{\sum }\limits_{{j = 1}}^{t}{v}_{j}{e}_{{i}_{j}}$ 。那么在时间$t$的查询$q$就是${x}^{\left( t\right) }$的一个函数，并且在提及当前时间时，我们将省略${}^{\left( t\right) }$上标。

There are a few common variations on this model with respect to the updates. In the cash register model or insertion only model $x$ is only incremented by bounded integers,i.e. ${v}_{j} \in  \left\lbrack  M\right\rbrack$ , for some $M \in  {\mathbb{N}}_{1}$ . In the turnstile model, $x$ can only be incremented or decremented by bounded integers,i.e. ${v}_{j} \in  \{  - M,\ldots ,M\}$ for some $M \in  {\mathbb{N}}_{1}$ ,and the strict turnstile model is as the turnstile model with the additional constraint that the entries of $x$ are always non-negative,i.e. ${x}_{i}^{\left( t\right) } \geq  0$ , for all $t \in  \left\lbrack  T\right\rbrack$ and $i \in  \left\lbrack  d\right\rbrack$ .

关于更新，此模型有几种常见的变体。在收银机模型（cash register model）或仅插入模型中，$x$ 仅通过有界整数递增，即对于某个 $M \in  {\mathbb{N}}_{1}$，有 ${v}_{j} \in  \left\lbrack  M\right\rbrack$。在旋转门模型（turnstile model）中，$x$ 只能通过有界整数递增或递减，即对于某个 $M \in  {\mathbb{N}}_{1}$，有 ${v}_{j} \in  \{  - M,\ldots ,M\}$，而严格旋转门模型与旋转门模型类似，但有额外的约束条件，即 $x$ 的元素始终是非负的，即对于所有的 $t \in  \left\lbrack  T\right\rbrack$ 和 $i \in  \left\lbrack  d\right\rbrack$，有 ${x}_{i}^{\left( t\right) } \geq  0$。

---

<!-- Footnote -->

${}^{11}$ Here the approximation ratio is between any $k$ -means algorithm running on the high dimensional original data and on the low dimensional projected data.

${}^{11}$ 这里的近似比是在高维原始数据上运行的任何 $k$ -均值算法与在低维投影数据上运行的该算法之间的比值。

<!-- Footnote -->

---

As mentioned above, we are usually space constrained so that we cannot explicitely store $x$ and the key idea to overcome this limitation is to store a linear sketch of $x$ ,that is storing $y \mathrel{\text{:=}} f\left( x\right)$ ,where $f : {\mathbb{R}}^{d} \rightarrow  {\mathbb{R}}^{m}$ is a linear function and $m \ll  d$ ,and then answering queries by applying some function on $y$ rather than $x$ . Note that since $f$ is linear,we can apply it to each update individually and compute $y$ as the sum of the sketched updates. Furthermore,we can aggregate results from different streams by adding the different sketches, allowing us to distribute the computation of the streaming algorithm.

如上所述，我们通常会受到空间限制，因此无法显式存储 $x$ 。克服这一限制的关键思路是存储 $x$ 的线性概要，即存储 $y \mathrel{\text{:=}} f\left( x\right)$ ，其中 $f : {\mathbb{R}}^{d} \rightarrow  {\mathbb{R}}^{m}$ 是一个线性函数且 $m \ll  d$ ，然后通过对 $y$ 而非 $x$ 应用某个函数来回答查询。请注意，由于 $f$ 是线性的，我们可以分别对每个更新应用它，并将 $y$ 计算为概要更新的总和。此外，我们可以通过将不同的概要相加来聚合来自不同流的结果，从而使我们能够对流式算法进行分布式计算。

The relevant Johnson-Lindenstrauss lemma in this setting is lemma 1.2 as with a JLD we get linearity and are able to sample a JLT before seeing any data at the cost of introducing some failure probability.

在这种情况下，相关的约翰逊 - 林登斯特劳斯引理（Johnson - Lindenstrauss lemma）是引理1.2，因为使用约翰逊 - 林登斯特劳斯变换（JLD）可以获得线性性质，并且能够在查看任何数据之前对约翰逊 - 林登斯特劳斯变换（JLT）进行采样，但代价是引入了一定的失败概率。

Based on JLDs, the most natural streaming problem to tackle is second frequency moment estimation in the turnstile model,i.e. approximating $\parallel x{\parallel }_{2}^{2}$ ,which has found use in database query optimisation [Alo+02 WDJ91; DeW+92] and network data analysis [Gil+01; CG05] among other areas. Simply letting $f$ be a sample from a JLD and returning $\parallel f\left( x\right) {\parallel }_{2}^{2}$ on queries,gives a factor $\left( {1 \pm  \varepsilon }\right)$ approximation with failure probability $\delta$ using $O\left( {{\varepsilon }^{-2}\log \frac{1}{\delta } + \left| f\right| }\right)$ words ${}^{12}$ of space, where $\left| f\right|$ denotes the words of space needed to store and apply $f$ . However,the approach taken by the streaming literature is to estimate $\parallel x{\parallel }_{2}^{2}$ with constant error probability using $O\left( {{\varepsilon }^{-2} + \left| f\right| }\right)$ words of space,and then sampling $O\left( {\log \frac{1}{\delta }}\right)$ JLTs ${f}_{1},\ldots ,{f}_{O\left( {\log 1/\delta }\right) } : {\mathbb{R}}^{d} \rightarrow  {\mathbb{R}}^{O\left( {\varepsilon }^{-2}\right) }$ and responding to a query with ${\operatorname{median}}_{k}{\begin{Vmatrix}{f}_{k}\left( x\right) \end{Vmatrix}}_{2}^{2}$ ,which reduces the error probability to $\delta$ . This allows for simpler analyses as well more efficient embeddings (in the case of Count Sketch) compared to using a single bigger JLT,but it comes at the cost of not embedding into ${\ell }_{2}$ ,which is needed for some applications outside of streaming. With this setup the task lies in constructing space efficient JLTs and a seminal work here is the AMS Sketch a.k.a. AGMS Sketch a.k.a. Tug-of-War Sketch [AMS99; Alo+02],whose JLTs can be defined as ${f}_{i} \mathrel{\text{:=}} {m}^{-1/2}{Ax}$ ,where $A \in  \{  - 1,1{\} }^{m \times  d}$ is a random matrix. The key idea is that each row $r$ of $A$ can be backed by a hash function ${\sigma }_{r} : \left\lbrack  d\right\rbrack   \rightarrow  \{  - 1,1\}$ that need only be 4 -wise independent,meaning that for any set of 4 distinct keys $\left\{  {{k}_{1},\ldots {k}_{4}}\right\}   \subset  \left\lbrack  d\right\rbrack$ and 4 (not necessarily distinct) values ${v}_{1},\ldots {v}_{4} \in  \{  - 1,1\}$ ,the probability that the keys hash to those values is $\mathop{\Pr }\limits_{{\sigma }_{r}}\left\lbrack  {\mathop{\bigwedge }\limits_{i}{\sigma }_{r}\left( {k}_{i}\right)  = {v}_{i}}\right\rbrack   = {\left| \{  - 1,1\} \right| }^{-4}$ . This can for instance ${}^{13}$ be attained by implementing ${\sigma }_{r}$ as 3rd degree polynomial modulus a sufficiently large prime with random coefficients [WC81],and so such a JLT need only use $\mathcal{O}\left( {\varepsilon }^{-2}\right)$ words of space. Embedding a scaled standard unit vector with such a JLT takes $O\left( {\varepsilon }^{-2}\right)$ time leading to an overall update time of the AMS Sketch of $O\left( {{\varepsilon }^{-2}\log \frac{1}{\delta }}\right)$ .

基于JLD（联合拉普拉斯分布，Joint Laplace Distribution），要解决的最自然的流式问题是旋转门模型（turnstile model）中的二阶频率矩估计，即近似$\parallel x{\parallel }_{2}^{2}$，这在数据库查询优化[阿洛等人2002年；WDJ 1991年；德韦等人1992年]和网络数据分析[吉尔等人2001年；CG 2005年]等领域有应用。简单地让$f$为来自JLD的一个样本，并在查询时返回$\parallel f\left( x\right) {\parallel }_{2}^{2}$，使用$O\left( {{\varepsilon }^{-2}\log \frac{1}{\delta } + \left| f\right| }\right)$个存储单元（单词，words）${}^{12}$的空间，可得到一个因子为$\left( {1 \pm  \varepsilon }\right)$的近似结果，失败概率为$\delta$，其中$\left| f\right|$表示存储和应用$f$所需的存储单元数量。然而，流式文献中采用的方法是使用$O\left( {{\varepsilon }^{-2} + \left| f\right| }\right)$个存储单元的空间以恒定误差概率估计$\parallel x{\parallel }_{2}^{2}$，然后对$O\left( {\log \frac{1}{\delta }}\right)$个JLT（约翰逊 - 林登施特劳斯变换，Johnson - Lindenstrauss Transform）${f}_{1},\ldots ,{f}_{O\left( {\log 1/\delta }\right) } : {\mathbb{R}}^{d} \rightarrow  {\mathbb{R}}^{O\left( {\varepsilon }^{-2}\right) }$进行采样，并以${\operatorname{median}}_{k}{\begin{Vmatrix}{f}_{k}\left( x\right) \end{Vmatrix}}_{2}^{2}$响应查询，这将误差概率降低到$\delta$。与使用单个更大的JLT相比，这种方法在分析上更简单，并且（在计数草图，Count Sketch的情况下）嵌入效率更高，但代价是不能嵌入到${\ell }_{2}$中，而这在一些流式处理之外的应用中是需要的。在这种设置下，任务在于构建空间高效的JLT，这里的开创性工作是AMS草图（也称为AGMS草图、拔河草图，Tug - of - War Sketch）[AMS 1999年；阿洛等人2002年]，其JLT可定义为${f}_{i} \mathrel{\text{:=}} {m}^{-1/2}{Ax}$，其中$A \in  \{  - 1,1{\} }^{m \times  d}$是一个随机矩阵。关键思想是$A$的每一行$r$可以由一个只需4 - 独立的哈希函数${\sigma }_{r} : \left\lbrack  d\right\rbrack   \rightarrow  \{  - 1,1\}$支持，这意味着对于任意4个不同的键$\left\{  {{k}_{1},\ldots {k}_{4}}\right\}   \subset  \left\lbrack  d\right\rbrack$和4个（不一定不同）的值${v}_{1},\ldots {v}_{4} \in  \{  - 1,1\}$，这些键哈希到那些值的概率为$\mathop{\Pr }\limits_{{\sigma }_{r}}\left\lbrack  {\mathop{\bigwedge }\limits_{i}{\sigma }_{r}\left( {k}_{i}\right)  = {v}_{i}}\right\rbrack   = {\left| \{  - 1,1\} \right| }^{-4}$。例如${}^{13}$，可以通过将${\sigma }_{r}$实现为具有随机系数的三次多项式模一个足够大的素数来实现[WC 1981年]，因此这样的JLT只需要使用$\mathcal{O}\left( {\varepsilon }^{-2}\right)$个存储单元的空间。用这样的JLT嵌入一个缩放的标准单位向量需要$O\left( {\varepsilon }^{-2}\right)$的时间，导致AMS草图的总体更新时间为$O\left( {{\varepsilon }^{-2}\log \frac{1}{\delta }}\right)$。

A later improvement of the AMS Sketch is the so-called Fast-AGMS Sketch [CG05] a.k.a. Count Sketch [CCF04; TZ12], which sparsifies the JLTs such that each column in their matrix representations only has one non-zero entry. Each JLT can be represented by a pairwise independent hash function $h : \left\lbrack  d\right\rbrack   \rightarrow  \left\lbrack  {O\left( {\varepsilon }^{-2}\right) }\right\rbrack$ to choose the position of each nonzero entry and a 4-wise independent hash function $\sigma  : \left\lbrack  d\right\rbrack   \rightarrow  \{  - 1,1\}$ to choose random signs as before. This reduces the standard unit vector embedding time to $O\left( 1\right)$ and so the overall update time becomes $O\left( {\log \frac{1}{\delta }}\right)$ for Count Sketch. It should be noted that the JLD inside Count Sketch is also known as Feature Hashing, which we will return to in section 1.4.1.

AMS草图（AMS Sketch）的一个后续改进是所谓的快速AGMS草图（Fast-AGMS Sketch）[CG05]，也称为计数草图（Count Sketch）[CCF04; TZ12]，它对约翰逊 - 林登施特劳斯变换（JLTs）进行了稀疏化处理，使得它们矩阵表示中的每一列只有一个非零元素。每个JLT可以由一个两两独立的哈希函数$h : \left\lbrack  d\right\rbrack   \rightarrow  \left\lbrack  {O\left( {\varepsilon }^{-2}\right) }\right\rbrack$来选择每个非零元素的位置，以及一个4 - 独立的哈希函数$\sigma  : \left\lbrack  d\right\rbrack   \rightarrow  \{  - 1,1\}$像之前一样来选择随机符号。这将标准单位向量嵌入时间减少到$O\left( 1\right)$，因此计数草图的总体更新时间变为$O\left( {\log \frac{1}{\delta }}\right)$。需要注意的是，计数草图中的约翰逊 - 林登施特劳斯维数约减（JLD）也被称为特征哈希（Feature Hashing），我们将在1.4.1节中再讨论它。

---

<!-- Footnote -->

${}^{12}$ Here we assume that a word is large enough to hold a sufficient approximation of any real number we use and to hold a number from the stream,i.e. if $w$ denotes the number of bits in a word then $w = \Omega \left( {\log d + \log M}\right)$ .

${}^{12}$ 这里我们假设一个字（word）足够大，能够对我们使用的任何实数进行充分近似，并且能够存储数据流中的一个数字，即如果 $w$ 表示一个字中的位数，那么 $w = \Omega \left( {\log d + \log M}\right)$ 。

${}^{13}$ See e.g. [TZ12] for other families of $k$ -wise independent hash functions.

${}^{13}$ 例如，关于 $k$ -wise 独立哈希函数的其他族，请参阅 [TZ12]。

<!-- Footnote -->

---

Despite not embedding into ${\ell }_{2}$ ,due to the use of the non-linear median,AMS Sketch and Count Sketch approximately preserve dot products similarly to corollary 1.5 [CG05, Theorem 2.1 and Theorem 3.5]. This allows us to query for the (approximate) frequency of any particular item as

尽管由于使用了非线性中位数而无法嵌入到 ${\ell }_{2}$ 中，但 AMS 草图（AMS Sketch）和计数草图（Count Sketch）与推论 1.5 [CG05，定理 2.1 和定理 3.5] 类似，近似地保留了点积。这使我们能够查询任何特定项的（近似）频率，如下所示

$$
\mathop{\operatorname{median}}\limits_{k}\left\langle  {{f}_{k}\left( x\right) ,{f}_{k}\left( {e}_{i}\right) }\right\rangle   = \left\langle  {x,{e}_{i}}\right\rangle   \pm  \varepsilon \parallel x{\parallel }_{2}{\begin{Vmatrix}{e}_{i}\end{Vmatrix}}_{2} = {x}_{i} \pm  \varepsilon \parallel x{\parallel }_{2}
$$

with probability at least $1 - \delta$ .

概率至少为 $1 - \delta$ 。

This can be extended to finding frequent items in an insertion only stream [CCF04]. The idea is to use a slightly larger ${}^{14}$ Count Sketch instance to maintain a heap of the $k$ approximately most frequent items of the stream so far. That is,if we let ${i}_{k}$ denote the $k$ th most frequent item (i.e. $\left| \left\{  {j \mid  {x}_{j} \geq  {x}_{{i}_{k}}}\right\}  \right|  = k$ ),then with probability $1 - \delta$ we have ${x}_{j} > \left( {1 - \varepsilon }\right) {x}_{{i}_{k}}$ for every item $j$ in our heap.

这可以扩展到在仅插入流中查找频繁项[CCF04]。其思路是使用一个稍大的${}^{14}$计数草图（Count Sketch）实例来维护一个堆，该堆包含到目前为止流中$k$个近似最频繁的项。也就是说，如果我们用${i}_{k}$表示第$k$个最频繁的项（即$\left| \left\{  {j \mid  {x}_{j} \geq  {x}_{{i}_{k}}}\right\}  \right|  = k$），那么对于我们堆中的每个项$j$，以概率$1 - \delta$有${x}_{j} > \left( {1 - \varepsilon }\right) {x}_{{i}_{k}}$。

For more on streaming algorithms, we refer the reader to [Mut05] and [Nel11], which also relates streaming to Johnson-Lindenstrauss.

关于流算法的更多内容，我们建议读者参考[Mut05]和[Nel11]，它们还将流与约翰逊 - 林登斯特劳斯（Johnson - Lindenstrauss）变换联系起来。

### 1.4 The Tapestry of Johnson-Lindenstrauss Transforms

### 1.4 约翰逊 - 林登斯特劳斯变换的图景

Isti mirant stella

他们惊讶于星星

—Scene 32, The Bayeux Tapestry [Unk70]

—第32幕，贝叶挂毯（Bayeux Tapestry）[Unk70]

As mentioned in section 1.2, the original JLD from [JL84] is a distribution over functions $f : {\mathbb{R}}^{d} \rightarrow  {\mathbb{R}}^{m}$ ,where ${}^{15}f\left( x\right)  = {\left( d/m\right) }^{1/2}{Ax}$ and $A$ is a random $m \times  d$ matrix whose rows form an orthonormal basis of some $m$ -dimensional subspace of ${\mathbb{R}}^{d}$ ,i.e. the rows are unit vectors and pairwise orthogonal. While Johnson and Lindenstrauss [JL84] showed that $m = \Theta \left( {{\varepsilon }^{-2}\log \left| X\right| }\right)$ suffices to prove lemma 1.1,they did not give any bounds on the constant in the big- $O$ expression. This was remedied in $\left| {\overline{\mathrm{{FM}}}{88}}\right|$ ,which proved that $m = \left\lceil  {9{\left( {\varepsilon }^{2} - 2{\varepsilon }^{3}/3\right) }^{-1}\ln \left| X\right| }\right\rceil   + 1$ suffices for the same JLD if $m < \sqrt{\left| X\right| }$ . This bound was further improved in [FM90] by removing the $m < \sqrt{\left| X\right| }$ restriction and lowering the bound to $m = \left\lceil  {8{\left( {\varepsilon }^{2} - 2{\varepsilon }^{3}/3\right) }^{-1}\ln \left| X\right| }\right\rceil$ .

如1.2节所述，文献[JL84]中最初的约翰逊 - 林登斯特劳斯分布（Johnson - Lindenstrauss distribution，JLD）是函数$f : {\mathbb{R}}^{d} \rightarrow  {\mathbb{R}}^{m}$上的一个分布，其中${}^{15}f\left( x\right)  = {\left( d/m\right) }^{1/2}{Ax}$，且$A$是一个随机$m \times  d$矩阵，其行构成${\mathbb{R}}^{d}$的某个$m$维子空间的一组标准正交基，即这些行向量是单位向量且两两正交。虽然约翰逊（Johnson）和林登斯特劳斯（Lindenstrauss）[JL84]证明了$m = \Theta \left( {{\varepsilon }^{-2}\log \left| X\right| }\right)$足以证明引理1.1，但他们并未给出大$O$表达式中常数的任何界。文献$\left| {\overline{\mathrm{{FM}}}{88}}\right|$弥补了这一不足，证明了若$m < \sqrt{\left| X\right| }$，则$m = \left\lceil  {9{\left( {\varepsilon }^{2} - 2{\varepsilon }^{3}/3\right) }^{-1}\ln \left| X\right| }\right\rceil   + 1$对于相同的约翰逊 - 林登斯特劳斯分布（JLD）是足够的。文献[FM90]进一步改进了这个界，去掉了$m < \sqrt{\left| X\right| }$的限制并将界降低到$m = \left\lceil  {8{\left( {\varepsilon }^{2} - 2{\varepsilon }^{3}/3\right) }^{-1}\ln \left| X\right| }\right\rceil$。

The next thread of JL research worked on simplifying the JLD constructions as Indyk and Motwani [HIM12] showed that sampling each entry in the matrix i.i.d. from a properly scaled Gaussian distribution is a JLD. The rows of such a matrix do not form a basis as they are with high probability not orthogonal; however, the literature still refer to this and most other JLDs as random projections. Shortly thereafter Arriaga and Vempala [AV06] constructed a JLD by sampling i.i.d. from a Rademacher ${}^{16}$ distribution,and Achlioptas [Ach03] sparsified the Rademacher construction such that the entries ${a}_{ij}$ are sampled i.i.d. with $\Pr \left\lbrack  {{a}_{ij} = 0}\right\rbrack   =$ $2/3$ and $\Pr \left\lbrack  {{a}_{ij} =  - 1}\right\rbrack   = \Pr \left\lbrack  {{a}_{ij} = 1}\right\rbrack   = 1/6$ . We will refer to such sparse i.i.d. Rademacher constructions as Achlioptas constructions. The Gaussian and Rademacher results have later been generalised [Mat08; IN07; KM05] to show that a JLD can be constructed by sampling each entry in a $m \times  d$ matrix i.i.d. from any distribution with mean 0,variance 1,and a subgaussian tail ${}^{17}$ . It should be noted that these developments have a parallel in the streaming literature as the previously mentioned AMS Sketch [AMS99; Alo+02] is identical to the Rademacher construction [AV06], albeit with constant error probability.

JL研究的下一个方向致力于简化JLD构造，因为英迪克（Indyk）和莫特瓦尼（Motwani）[HIM12]证明了从适当缩放的高斯分布中独立同分布地对矩阵的每个元素进行采样可得到一个JLD。这种矩阵的行并不构成一个基，因为它们大概率不是正交的；然而，文献中仍然将这种以及其他大多数JLD称为随机投影。此后不久，阿里亚加（Arriaga）和文帕拉（Vempala）[AV06]通过从拉德马赫（Rademacher）${}^{16}$分布中独立同分布地采样构造了一个JLD，阿赫利奥普塔斯（Achlioptas）[Ach03]对拉德马赫构造进行了稀疏化处理，使得元素${a}_{ij}$以$\Pr \left\lbrack  {{a}_{ij} = 0}\right\rbrack   =$ $2/3$和$\Pr \left\lbrack  {{a}_{ij} =  - 1}\right\rbrack   = \Pr \left\lbrack  {{a}_{ij} = 1}\right\rbrack   = 1/6$的概率独立同分布地采样。我们将这种稀疏的独立同分布的拉德马赫构造称为阿赫利奥普塔斯构造。高斯和拉德马赫的结果后来得到了推广[Mat08; IN07; KM05]，表明可以通过从任何均值为0、方差为1且具有次高斯尾部${}^{17}$的分布中独立同分布地对$m \times  d$矩阵的每个元素进行采样来构造一个JLD。值得注意的是，这些进展在流式数据文献中有类似情况，因为前面提到的AMS草图[AMS99; Alo+02]与拉德马赫构造[AV06]相同，尽管误差概率是恒定的。

---

<!-- Footnote -->

${}^{14}$ Rather than each JLT having a target dimension of $O\left( {\varepsilon }^{-2}\right)$ ,the analysis needs the target dimension to be $O\left( \frac{{\begin{Vmatrix}{\operatorname{tail}}_{k}\left( x\right) \end{Vmatrix}}_{2}^{2}}{{\left( \varepsilon {x}_{{i}_{k}}\right) }^{2}}\right)$ ,where ${\operatorname{tail}}_{k}\left( x\right)$ denotes $x$ with its $k$ largest entries zeroed out.

${}^{14}$ 分析所需的目标维度应为 $O\left( \frac{{\begin{Vmatrix}{\operatorname{tail}}_{k}\left( x\right) \end{Vmatrix}}_{2}^{2}}{{\left( \varepsilon {x}_{{i}_{k}}\right) }^{2}}\right)$，而非每个联合局部变换（JLT）的目标维度为 $O\left( {\varepsilon }^{-2}\right)$，其中 ${\operatorname{tail}}_{k}\left( x\right)$ 表示将 $x$ 中 $k$ 个最大元素置零后的结果。

${}^{15}$ We will usually omit the normalisation or scaling factor (the ${\left( d/m\right) }^{1/2}$ for this JLD) when discussing JLDs as they are textually noisy, not that interesting, and independent of randomness and input data.

${}^{15}$ 在讨论联合局部离散化（JLD）时，我们通常会省略归一化或缩放因子（此 JLD 的 ${\left( d/m\right) }^{1/2}$），因为它们在文本上造成干扰，不太有趣，且与随机性和输入数据无关。

${}^{16}$ The Rademacher distribution is the uniform distribution on $\{  - 1,1\}$ .

${}^{16}$ 拉德马赫分布（Rademacher distribution）是 $\{  - 1,1\}$ 上的均匀分布。

<!-- Footnote -->

---

As for the target dimension for these constructions, [HIM12] proved that the Gaussian construction is a JLD if $m \geq  8{\left( {\varepsilon }^{2} - 2{\varepsilon }^{3}/3\right) }^{-1}\left( {\ln \left| X\right|  + \mathcal{O}\left( {\log m}\right) }\right)$ ,which roughly corresponds to an additional additive $O\left( {{\varepsilon }^{-2}\log \log \left| X\right| }\right)$ term over the original construction. This additive $\log \log$ term was shaved off by the proof in [DG02],which concerns itself with the original JLD construction but can easily ${}^{18}$ be adapted to the Gaussian construction,and the proof in [AV06], which also give the same log log free bound for the dense Rademacher construction. Achlioptas [Ach03] showed that his construction also achieves $m = \left\lceil  {8{\left( {\varepsilon }^{2} - 2{\varepsilon }^{3}/3\right) }^{-1}\ln \left| X\right| }\right\rceil$ . The constant of 8 has been improved for the Gaussian and dense Rademacher constructions in the sense that Rojo and Nguyen [RN10; Ngu09] have been able to replace the bound with more intricate ${}^{19}$ expressions,which yield a 10 to 40% improvement for many sets of parameters. However, in the distributional setting it has been shown in [BGK18] that $m \geq  4{\varepsilon }^{-2}\ln \frac{1}{\delta }\left( {1 - o\left( 1\right) }\right)$ is necessary for any JLD to satisfy lemma 1.2, which corresponds to a constant of 8 if we prove lemma 1.1 the usual way by setting $\delta  = {n}^{-2}$ and union bounding over all pairs of vectors.

对于这些构造的目标维度，文献[HIM12]证明了，如果$m \geq  8{\left( {\varepsilon }^{2} - 2{\varepsilon }^{3}/3\right) }^{-1}\left( {\ln \left| X\right|  + \mathcal{O}\left( {\log m}\right) }\right)$，则高斯构造是一个约翰逊 - 林登施特劳斯引理（JLD）构造，这大致对应于在原始构造基础上增加一个附加项$O\left( {{\varepsilon }^{-2}\log \log \left| X\right| }\right)$。文献[DG02]中的证明去掉了这个附加项$\log \log$，该证明针对的是原始的约翰逊 - 林登施特劳斯引理构造，但可以很容易地${}^{18}$应用到高斯构造上；文献[AV06]中的证明也为稠密拉德马赫构造给出了相同的无双重对数界。阿赫利奥普塔斯（Achlioptas）[Ach03]表明他的构造也能达到$m = \left\lceil  {8{\left( {\varepsilon }^{2} - 2{\varepsilon }^{3}/3\right) }^{-1}\ln \left| X\right| }\right\rceil$。对于高斯构造和稠密拉德马赫构造，常数 8 已经得到了改进，具体来说，罗霍（Rojo）和阮（Nguyen）[RN10; Ngu09]能够用更复杂的表达式${}^{19}$来替代这个界，对于许多参数集，这能带来 10% 到 40% 的改进。然而，在分布设置下，文献[BGK18]表明，任何约翰逊 - 林登施特劳斯引理构造要满足引理 1.2 都需要$m \geq  4{\varepsilon }^{-2}\ln \frac{1}{\delta }\left( {1 - o\left( 1\right) }\right)$，如果我们按照通常的方式通过设置$\delta  = {n}^{-2}$并对所有向量对进行并集界估计来证明引理 1.1，这对应于常数 8。

There seems to have been some confusion in the literature regarding the improvements in target dimension. The main pitfall was that some papers [e.g. Ach03; HIM12; DG02; RN10; BGK18] were only referring to [FM88] when referring to the target dimension bound of the original construction. As such, [Ach03, HIM12] mistakenly claim to improve the constant for the target dimension with their constructions. Furthermore, [Ach01] is sometimes [e.g. in AC09: Mat08: Sch18] the only work credited for the Rademacher construction, despite it being developed independently and published 2 years prior in [AV99].

在文献中，关于目标维度的改进似乎存在一些混淆。主要的问题在于，一些论文（例如Ach03；HIM12；DG02；RN10；BGK18）在提及原始构造的目标维度界时，仅参考了[FM88]。因此，[Ach03, HIM12]错误地声称其构造改进了目标维度的常数。此外，尽管拉德马赫（Rademacher）构造是在[AV99]中独立开发并于两年前发表的，但有时（例如在AC09、Mat08、Sch18中）只有[Ach01]被认为是该构造的相关研究。

All the constructions that have been mentioned so far in this section, embed a vector by performing a relatively dense and unstructured matrix-vector multiplication, which takes $\Theta \left( {m\parallel x{\parallel }_{0}}\right)  = O\left( {md}\right)$ time ${}^{20}$ to compute. This sparked two distinct but intertwined strands of research seeking to reduce the embedding time, namely the sparsity-based JLDs which dealt with the density of the embedding matrix and the fast Fourier transform-based which introduced more structure to the matrix.

到目前为止，本节中提到的所有构造方法都是通过执行相对密集且无结构的矩阵 - 向量乘法来嵌入向量，该计算需要$\Theta \left( {m\parallel x{\parallel }_{0}}\right)  = O\left( {md}\right)$时间${}^{20}$。这引发了两条不同但相互交织的研究方向，旨在减少嵌入时间，即基于稀疏性的联合低维表示（JLDs，Joint Low - Dimensional Representations），它处理嵌入矩阵的密度问题；以及基于快速傅里叶变换的方法，它为矩阵引入了更多结构。

---

<!-- Footnote -->

${}^{17}$ A real random variable $X$ with mean 0 has a subgaussian tail if there exists constants $\alpha ,\beta  > 0$ such that for all $\lambda  > 0,\Pr \left\lbrack  {\left| X\right|  > \lambda }\right\rbrack   \leq  \beta {e}^{-\alpha {\lambda }^{2}}.$

${}^{17}$ 若存在常数$\alpha ,\beta  > 0$，使得对于所有的$\lambda  > 0,\Pr \left\lbrack  {\left| X\right|  > \lambda }\right\rbrack   \leq  \beta {e}^{-\alpha {\lambda }^{2}}.$，均值为 0 的实随机变量$X$具有次高斯尾部。

${}^{18}$ The main part of the proof in [DG02] is showing that the ${\ell }_{2}$ norm of a vector of i.i.d. Gaussians is concentrated around the expected value. A vector projected with the Gaussian construction is distributed as a vector of i.i.d. Gaussians.

${}^{18}$ [DG02]中证明的主要部分是表明独立同分布高斯变量构成的向量的${\ell }_{2}$范数集中在期望值附近。通过高斯构造进行投影的向量的分布与独立同分布高斯变量构成的向量的分布相同。

${}^{19}$ For example,one of the bounds for the Rademacher construction is $m \geq  \frac{2\left( {d - 1}\right) {\alpha }^{2}}{d{\varepsilon }^{2}}$ ,where $\alpha  \mathrel{\text{:=}} \frac{Q + \sqrt{{Q}^{2} + {5.98}}}{2}$ , $Q \mathrel{\text{:=}} {\Phi }^{-1}\left( {1 - 1/{\left| X\right| }^{2}}\right)$ ,and ${\Phi }^{-1}$ is the quantile function of the standard Gaussian random variable.

${}^{19}$ 例如，拉德马赫构造（Rademacher construction）的一个边界是$m \geq  \frac{2\left( {d - 1}\right) {\alpha }^{2}}{d{\varepsilon }^{2}}$ ，其中$\alpha  \mathrel{\text{:=}} \frac{Q + \sqrt{{Q}^{2} + {5.98}}}{2}$ ，$Q \mathrel{\text{:=}} {\Phi }^{-1}\left( {1 - 1/{\left| X\right| }^{2}}\right)$ ，并且${\Phi }^{-1}$ 是标准高斯随机变量的分位数函数。

${}^{20}\parallel x{\parallel }_{0}$ is the number of nonzero entries in the vector $x$ .

${}^{20}\parallel x{\parallel }_{0}$ 是向量 $x$ 中非零元素的数量。

<!-- Footnote -->

---

#### 1.4.1 Sparse Johnson-Lindenstrauss Transforms

#### 1.4.1 稀疏约翰逊 - 林登斯特劳斯变换（Sparse Johnson-Lindenstrauss Transforms）

The simple fact underlying the following string of results is that if $A$ has $s$ nonzero entries per column,then $f\left( x\right)  = {Ax}$ can be computed in $\Theta \left( {s\parallel x{\parallel }_{0}}\right)$ time. The first result here is the Achlioptas construction [Ach03] mentioned above,whose column sparsity $s$ is $m/3$ in expectancy,which leads to an embedding time that is a third of the full Rademacher construction ${}^{21}$ . However, the first superconstant improvement is due to Dasgupta, Kumar, and Sarlós [DKS10], who based on heuristic approaches [Wei+09; Shi+09b; LLS07; GD08] constructed a JLD with $s = O\left( {{\varepsilon }^{-1}\log \frac{1}{\delta }{\log }^{2}\frac{m}{\delta }}\right)$ . Their construction,which we will refer to as the DKS construction, works by sampling $s$ hash functions ${h}_{1},\ldots ,{h}_{s} : \left\lbrack  d\right\rbrack   \rightarrow  \{  - 1,1\}  \times  \left\lbrack  m\right\rbrack$ independently,such that each source entry ${x}_{i}$ will be hashed to $s$ random signs ${\sigma }_{i,1},\ldots ,{\sigma }_{i,s}$ and $s$ target coordinates ${j}_{i,1},\ldots ,{j}_{i,s}$ (with replacement). The embedding can then be defined as $f\left( x\right)  \mathrel{\text{:=}} \mathop{\sum }\limits_{i}\mathop{\sum }\limits_{k}{e}_{{j}_{i,k}}{\sigma }_{i,k}{x}_{i}$ , which is to say that every source coordinate is hashed to $s$ output coordinates and randomly added to or subtracted from those output coordinates. The sparsity analysis was later tightened to show that $s = O\left( {{\varepsilon }^{-1}\log \frac{1}{\delta }\log \frac{m}{\delta }}\right)$ suffices $\left\lbrack  {\mathrm{{KN}}{10} : \mathrm{{KN}}{14}}\right\rbrack$ and even $s = O\left( {{\varepsilon }^{-1}{\left( \frac{\log \frac{1}{\delta }\log \log \log \frac{1}{\delta }}{\log \log \frac{1}{\delta }}\right) }^{2}}\right)$ suffices for the DKS construction assuming $\varepsilon  < {\log }^{-2}\frac{1}{\delta }\left\lbrack  \mathrm{{BOR10}}\right\rbrack$ ,while $\left\lbrack  \mathrm{{KN14}}\right\rbrack$ showed that $s = \Omega \left( {{\varepsilon }^{-1}{\log }^{2}\frac{1}{\delta }/{\log }^{2}\frac{1}{\varepsilon }}\right)$ is neccessary for the DKS construction.

以下一系列结果背后的简单事实是，如果$A$每列有$s$个非零元素，那么$f\left( x\right)  = {Ax}$可以在$\Theta \left( {s\parallel x{\parallel }_{0}}\right)$时间内计算得出。这里的第一个结果是上述提到的阿赫利奥普塔斯构造（Achlioptas construction，[Ach03]），其列稀疏度$s$的期望值为$m/3$，这使得嵌入时间是完整拉德马赫构造（Rademacher construction，${}^{21}$）的三分之一。然而，首次实现超常数改进归功于达斯古普塔（Dasgupta）、库马尔（Kumar）和萨尔洛斯（Sarlós）（[DKS10]），他们基于启发式方法（[Wei+09; Shi+09b; LLS07; GD08]）构造了一个具有$s = O\left( {{\varepsilon }^{-1}\log \frac{1}{\delta }{\log }^{2}\frac{m}{\delta }}\right)$的约翰逊 - 林登施特劳斯引理（JLD）。他们的构造（我们将其称为DKS构造）通过独立采样$s$个哈希函数${h}_{1},\ldots ,{h}_{s} : \left\lbrack  d\right\rbrack   \rightarrow  \{  - 1,1\}  \times  \left\lbrack  m\right\rbrack$来实现，使得每个源元素${x}_{i}$将被哈希到$s$个随机符号${\sigma }_{i,1},\ldots ,{\sigma }_{i,s}$和$s$个目标坐标${j}_{i,1},\ldots ,{j}_{i,s}$（可重复）。然后嵌入可以定义为$f\left( x\right)  \mathrel{\text{:=}} \mathop{\sum }\limits_{i}\mathop{\sum }\limits_{k}{e}_{{j}_{i,k}}{\sigma }_{i,k}{x}_{i}$，也就是说，每个源坐标被哈希到$s$个输出坐标，并随机地加到或从这些输出坐标中减去。后来对稀疏性分析进行了优化，结果表明对于DKS构造，$s = O\left( {{\varepsilon }^{-1}\log \frac{1}{\delta }\log \frac{m}{\delta }}\right)$就足够了（$\left\lbrack  {\mathrm{{KN}}{10} : \mathrm{{KN}}{14}}\right\rbrack$），假设$\varepsilon  < {\log }^{-2}\frac{1}{\delta }\left\lbrack  \mathrm{{BOR10}}\right\rbrack$，甚至$s = O\left( {{\varepsilon }^{-1}{\left( \frac{\log \frac{1}{\delta }\log \log \log \frac{1}{\delta }}{\log \log \frac{1}{\delta }}\right) }^{2}}\right)$也足够，而$\left\lbrack  \mathrm{{KN14}}\right\rbrack$表明对于DKS构造，$s = \Omega \left( {{\varepsilon }^{-1}{\log }^{2}\frac{1}{\delta }/{\log }^{2}\frac{1}{\varepsilon }}\right)$是必要的。

Kane and Nelson [KN14] present two constructions that circumvent the DKS lower bound by ensuring that the hash functions do not collide within a column,i.e. ${j}_{i,a} \neq  {j}_{i,b}$ for all $i,a$ , and $b$ . The first construction,which we will refer to as the graph construction,simply samples the $s$ coordinates without replacement. The second construction,which we will refer to as the block construction,partitions the output vector into $s$ consecutive blocks of length $m/s$ and samples one output coordinate per block. Note that the block construction is the same as Count Sketch from the streaming literature [CG05; CCF04], though the hash functions differ and the output is interpreted differently. Kane and Nelson [KN14] prove that $s = \Theta \left( {{\varepsilon }^{-1}\log \frac{1}{\delta }}\right)$ is both neccessary and sufficient in order for their two constructions to satisfy lemma 1.2. Note that while Count Sketch is even sparser than the lower bound for the block construction, it does not contradict it as Count sketch does not embed into ${\ell }_{2}^{m}$ as it computes the median,which is nonlinear. As far as general sparsity lower bounds go, [DKS10] shows that an average column sparsity of ${s}_{\text{avg }} = \Omega \left( {\min \left\{  {{\varepsilon }^{-2},{\varepsilon }^{-1}\sqrt{{\log }_{m}d}}\right\}  }\right)$ is neccessary for a sparse JLD,while Nelson and Nguyên [NN13b] improves upon this by showing that there exists a set of points $X \in  {\mathbb{R}}^{d}$ such that any JLT for that set must have column sparsity $s = \Omega \left( {{\varepsilon }^{-1}\log \left| X\right| /\log \frac{1}{\varepsilon }}\right)$ in order to satisfy lemma 1.1. And so it seems that we have almost reached the limit of the sparse JL approach, but why should theory be in the way of a good result? Let us massage the definitions so as to get around these lower bounds.

凯恩（Kane）和纳尔逊（Nelson）[KN14]提出了两种构造方法，通过确保哈希函数在列内不发生冲突，即对于所有$i,a$和$b$满足${j}_{i,a} \neq  {j}_{i,b}$，从而绕过了DKS下界。第一种构造方法，我们称之为图构造（graph construction），它只是无放回地对$s$个坐标进行采样。第二种构造方法，我们称之为块构造（block construction），它将输出向量划分为$s$个长度为$m/s$的连续块，并从每个块中采样一个输出坐标。请注意，块构造与流式文献[CG05; CCF04]中的计数草图（Count Sketch）相同，不过哈希函数不同，输出的解释也不同。凯恩和纳尔逊[KN14]证明，为了使他们的两种构造方法满足引理1.2，$s = \Theta \left( {{\varepsilon }^{-1}\log \frac{1}{\delta }}\right)$既是必要条件也是充分条件。请注意，虽然计数草图比块构造的下界还要稀疏，但这并不矛盾，因为计数草图计算的是中位数（这是非线性的），它并不嵌入到${\ell }_{2}^{m}$中。就一般的稀疏性下界而言，[DKS10]表明，对于稀疏JLD，平均列稀疏度为${s}_{\text{avg }} = \Omega \left( {\min \left\{  {{\varepsilon }^{-2},{\varepsilon }^{-1}\sqrt{{\log }_{m}d}}\right\}  }\right)$是必要的，而纳尔逊和阮（Nguyên）[NN13b]对此进行了改进，他们证明存在一组点$X \in  {\mathbb{R}}^{d}$，使得该集合的任何JLT为了满足引理1.1，其列稀疏度必须为$s = \Omega \left( {{\varepsilon }^{-1}\log \left| X\right| /\log \frac{1}{\varepsilon }}\right)$。因此，似乎我们已经几乎达到了稀疏约翰逊 - 林登施特劳斯（JL）方法的极限，但为什么理论要阻碍取得好的结果呢？让我们对定义进行调整，以绕过这些下界。

The hard instances used to prove the lower bounds [NN13b; KN14] consist of very sparse vectors,e.g. $x = {\left( 1/\sqrt{2},1/\sqrt{2},0,\ldots ,0\right) }^{\mathrm{T}}$ ,but the vectors we are interested in applying a JLT to might not be so unpleasant, and so by restricting the input vectors to be sufficiently "nice", we can get meaningful result that perform better than what the pessimistic lower bound would indicate. The formal formulation of this niceness is bounding the ${\ell }_{\infty }/{\ell }_{2}$ ratio of the vectors lemmas 1.1 and 1.2 need apply to. Let us denote this norm ratio as $v \in  \left\lbrack  {1/\sqrt{d},1}\right\rbrack$ ,and revisit some of the sparse JLDs. The Achlioptas construction [Ach03] can be generalised so that the expected number of nonzero entries per column is ${qm}$ rather than $\frac{1}{3}m$ for a parameter $q \in  \left\lbrack  {0,1}\right\rbrack$ . Ailon and Chazelle [AC09] show that if $v = O\left( {\sqrt{\log \frac{1}{\delta }}/\sqrt{d}}\right)$ then choosing $q = \Theta \left( \frac{{\log }^{2}1/\delta }{d}\right)$ and sampling the nonzero entries from a Gaussian distribution suffices. This result is generalised in [Mat08] by proving that for all $v \in  \left\lbrack  {1/\sqrt{d},1}\right\rbrack$ choosing $q = \Theta \left( {{v}^{2}\log \frac{d}{\varepsilon \delta }}\right)$ and sampling the nonzero entries from a Rademacher distribution is a JLD for the vectors constrained by that $v$ .

用于证明下界的困难实例[NN13b; KN14]由非常稀疏的向量组成，例如$x = {\left( 1/\sqrt{2},1/\sqrt{2},0,\ldots ,0\right) }^{\mathrm{T}}$，但我们有兴趣应用约翰逊 - 林登施特劳斯引理（JLT）的向量可能并非如此糟糕。因此，通过将输入向量限制为足够“良好”，我们可以得到比悲观下界所表明的结果更优的有意义结果。这种“良好性”的正式表述是对向量的${\ell }_{\infty }/{\ell }_{2}$比率进行界定，引理1.1和1.2需要应用于此。我们将这个范数比率记为$v \in  \left\lbrack  {1/\sqrt{d},1}\right\rbrack$，并重新审视一些稀疏的约翰逊 - 林登施特劳斯分布（JLD）。阿赫利奥普塔斯构造（Achlioptas construction）[Ach03]可以进行推广，使得对于参数$q \in  \left\lbrack  {0,1}\right\rbrack$，每列的非零元素的期望数量为${qm}$而非$\frac{1}{3}m$。艾隆和查泽尔（Ailon and Chazelle）[AC09]表明，如果$v = O\left( {\sqrt{\log \frac{1}{\delta }}/\sqrt{d}}\right)$，那么选择$q = \Theta \left( \frac{{\log }^{2}1/\delta }{d}\right)$并从高斯分布中采样非零元素就足够了。在[Mat08]中，通过证明对于所有的$v \in  \left\lbrack  {1/\sqrt{d},1}\right\rbrack$，选择$q = \Theta \left( {{v}^{2}\log \frac{d}{\varepsilon \delta }}\right)$并从拉德马赫分布中采样非零元素，对于受该$v$约束的向量而言是一种约翰逊 - 林登施特劳斯分布（JLD），从而将这一结果进行了推广。

---

<!-- Footnote -->

${}^{21}$ Here we ignore any overhead that switching to a sparse matrix representation would introduce.

${}^{21}$ 在此，我们忽略切换到稀疏矩阵表示法会带来的任何开销。

<!-- Footnote -->

---

Be aware that sometimes [e.g. in DKS10; BOR10] this bound ${}^{22}$ on $q$ is misinterpreted as a lower bound stating that ${qm} = \widetilde{\Omega }\left( {\varepsilon }^{-2}\right)$ is neccessary for the Achlioptas construction when $v = 1$ . However,Matoušek $\left\lbrack  \text{Mat08}\right\rbrack$ only loosely argues that his bound is tight for $v \leq  {d}^{-{0.1}}$ ,and if it indeed was tight at $v = 1$ ,the factors hidden by the $\widetilde{\Omega }$ would lead to the contradiction that $m \geq  {qm} = \Omega \left( {{\varepsilon }^{-2}\log \frac{1}{\delta }\log \frac{d}{\varepsilon \delta }}\right)  = \omega \left( m\right) .$

请注意，有时（例如在DKS10；BOR10中），关于$q$的这个界${}^{22}$会被错误地解读为一个下界，即声称当$v = 1$时，${qm} = \widetilde{\Omega }\left( {\varepsilon }^{-2}\right)$对于阿赫利奥普塔斯构造（Achlioptas construction）是必要的。然而，马托谢克（Matoušek）$\left\lbrack  \text{Mat08}\right\rbrack$只是大致论证了他的界对于$v \leq  {d}^{-{0.1}}$是紧的，并且如果它在$v = 1$处确实是紧的，那么由$\widetilde{\Omega }$所隐藏的因子将会导致$m \geq  {qm} = \Omega \left( {{\varepsilon }^{-2}\log \frac{1}{\delta }\log \frac{d}{\varepsilon \delta }}\right)  = \omega \left( m\right) .$这一矛盾结果

The heuristic [Wei+09; Shi+09b; LLS07; GD08] that [DKS10] is based on is called Feature Hashing a.k.a. the hashing trick a.k.a. the hashing kernel and is a sparse JL construction with exactly 1 nonzero entry per column ${}^{23}$ . The block construction can then be viewed as the concatenation of $s = \Theta \left( {{\varepsilon }^{-1}\log \frac{1}{\delta }}\right)$ feature hashing instances,and the DKS construction can be viewed as the sum of $s = O\left( {{\varepsilon }^{-1}\log \frac{1}{\delta }\log \frac{m}{\delta }}\right)$ Feature Hashing instances or alternatively as first duplicating each entry of $x \in  {\mathbb{R}}^{d}s$ times before applying Feature Hashing to the enlarged vector ${x}^{\prime } \in  {\mathbb{R}}^{sd}$ : Let ${f}_{\text{dup }} : {\mathbb{R}}^{d} \rightarrow  {\mathbb{R}}^{sd}$ be a function that duplicates each entry in its input $s$ times,i.e. ${f}_{\mathrm{{dup}}}{\left( x\right) }_{\left( {i - 1}\right) s + j} = {x}_{\left( {i - 1}\right) s + j}^{\prime } \mathrel{\text{:=}} {x}_{i}$ for $i \in  \left\lbrack  d\right\rbrack  ,j \in  \left\lbrack  s\right\rbrack$ ,then ${f}_{\mathrm{{DKS}}} = {f}_{\mathrm{{FH}}} \circ  {f}_{\mathrm{{dup}}}$ .

[DKS10]所基于的启发式方法[Wei+09; Shi+09b; LLS07; GD08]被称为特征哈希（Feature Hashing，又名哈希技巧（hashing trick）或哈希核（hashing kernel）），它是一种稀疏的约翰逊 - 林登斯特劳斯（JL）构造，每列恰好有1个非零元素${}^{23}$。块构造可以看作是$s = \Theta \left( {{\varepsilon }^{-1}\log \frac{1}{\delta }}\right)$个特征哈希实例的串联，而DKS构造可以看作是$s = O\left( {{\varepsilon }^{-1}\log \frac{1}{\delta }\log \frac{m}{\delta }}\right)$个特征哈希实例的和，或者也可以看作是先将$x \in  {\mathbb{R}}^{d}s$中的每个元素复制若干次，然后对扩展后的向量${x}^{\prime } \in  {\mathbb{R}}^{sd}$应用特征哈希：设${f}_{\text{dup }} : {\mathbb{R}}^{d} \rightarrow  {\mathbb{R}}^{sd}$是一个将其输入中的每个元素复制$s$次的函数，即对于$i \in  \left\lbrack  d\right\rbrack  ,j \in  \left\lbrack  s\right\rbrack$有${f}_{\mathrm{{dup}}}{\left( x\right) }_{\left( {i - 1}\right) s + j} = {x}_{\left( {i - 1}\right) s + j}^{\prime } \mathrel{\text{:=}} {x}_{i}$，那么${f}_{\mathrm{{DKS}}} = {f}_{\mathrm{{FH}}} \circ  {f}_{\mathrm{{dup}}}$。

This duplication is the key to the analysis in [DKS10] as ${f}_{\text{dup }}$ is isometric (up to normalisation) and it ensures that the ${\ell }_{\infty }/{\ell }_{2}$ ratio of ${x}^{\prime }$ is small,i.e. $v \leq  1/\sqrt{s}$ from the point of view of the Feature Hashing data structure $\left( {f}_{\mathrm{{FH}}}\right)$ . And so,any lower bound on the sparsity of the DKS construction (e.g. the one given in [KN14]) gives an upper bound on the values of $v$ for which Feature Hashing is a JLD: If $u$ is a unit vector such that a DKS instance with sparsity $\widehat{s}$ fails to preserve $u$ s norm within $1 \pm  \varepsilon$ with probability $\delta$ ,then it must be the case that Feature Hashing fails to preserve the norm of ${f}_{\text{dup }}\left( u\right)$ within $1 \pm  \varepsilon$ with probability $\delta$ ,and therefore the ${\ell }_{\infty }/{\ell }_{2}$ ratio for which Feature Hashing can handle all vectors is strictly less than $1/\sqrt{\widehat{s}}$ .

这种重复是[DKS10]中分析的关键，因为${f}_{\text{dup }}$是等距的（直至归一化），并且它确保了${x}^{\prime }$的${\ell }_{\infty }/{\ell }_{2}$比率较小，即从特征哈希（Feature Hashing）数据结构$\left( {f}_{\mathrm{{FH}}}\right)$的角度来看为$v \leq  1/\sqrt{s}$。因此，DKS构造的稀疏性的任何下界（例如[KN14]中给出的下界）都给出了特征哈希成为JLD时$v$值的上界：如果$u$是一个单位向量，使得稀疏性为$\widehat{s}$的DKS实例以概率$\delta$无法将$u$的范数保持在$1 \pm  \varepsilon$范围内，那么特征哈希必定以概率$\delta$无法将${f}_{\text{dup }}\left( u\right)$的范数保持在$1 \pm  \varepsilon$范围内，因此特征哈希能够处理所有向量时的${\ell }_{\infty }/{\ell }_{2}$比率严格小于$1/\sqrt{\widehat{s}}$。

Written more concisely the statement is ${s}_{\mathrm{{DKS}}} = \Omega \left( a\right)  \Rightarrow  {v}_{\mathrm{{FH}}} = O\left( {1/\sqrt{a}}\right)$ and by contraposition ${}^{24}{v}_{\mathrm{{FH}}} = \Omega \left( {1/\sqrt{a}}\right)  \Rightarrow  {s}_{\mathrm{{DKS}}} = O\left( a\right)$ ,where ${s}_{\mathrm{{DKS}}}$ is the minimum column sparsity of a DKS construction that is a JLD, ${v}_{\mathrm{{FH}}}$ is the maximum ${\ell }_{\infty }/{\ell }_{2}$ constraint for which Feature Hashing is a JLD,and $a$ is any positive expression. Furthermore,if we prove an upper bound on ${v}_{\mathrm{{FH}}}$ using a hard instance that is identical to an ${x}^{\prime }$ that the DKS construction can generate after duplication, we can replace the previous two implications with bi-implications.

更简洁地表述，该陈述为${s}_{\mathrm{{DKS}}} = \Omega \left( a\right)  \Rightarrow  {v}_{\mathrm{{FH}}} = O\left( {1/\sqrt{a}}\right)$，通过逆否命题为${}^{24}{v}_{\mathrm{{FH}}} = \Omega \left( {1/\sqrt{a}}\right)  \Rightarrow  {s}_{\mathrm{{DKS}}} = O\left( a\right)$，其中${s}_{\mathrm{{DKS}}}$是作为联合线性鉴别器（JLD）的DKS构造的最小列稀疏度，${v}_{\mathrm{{FH}}}$是特征哈希（Feature Hashing）作为联合线性鉴别器（JLD）时的最大${\ell }_{\infty }/{\ell }_{2}$约束，$a$是任意正表达式。此外，如果我们使用一个与DKS构造在复制后所能生成的${x}^{\prime }$相同的困难实例来证明${v}_{\mathrm{{FH}}}$的上界，我们可以将上述两个蕴含关系替换为双向蕴含关系。

[Wei+09] claims to give a bound on ${v}_{\mathrm{{FH}}}$ ,but it sadly contains an error in its proof of this bound [DKS10; Wei+10]. Dahlgaard,Knudsen,and Thorup [DKT17] improve the ${v}_{\mathrm{{FH}}}$ lower bound to ${v}_{\mathrm{{FH}}} = \Omega \left( \sqrt{\frac{\varepsilon \log \left( {1 + \frac{\delta }{\varepsilon }}\right) }{\log \frac{1}{\delta }\log \frac{m}{\delta }}}\right)$ ,and Freksen,Kamma,and Larsen [FKL18] give an intricate but tight bound for ${v}_{\mathrm{{FH}}}$ shown in theorem 1.8,where the hard instance used to prove the upper bound is identical to an ${x}^{\prime }$ from the DKS construction.

[魏（Wei）等人，2009年]声称给出了${v}_{\mathrm{{FH}}}$的一个界，但遗憾的是，其对该界的证明存在错误[达尔加德（Dahlgaard）等人，2010年；魏（Wei）等人，2010年]。达尔加德（Dahlgaard）、克努森（Knudsen）和索鲁普（Thorup）[DKT17]将${v}_{\mathrm{{FH}}}$的下界改进为${v}_{\mathrm{{FH}}} = \Omega \left( \sqrt{\frac{\varepsilon \log \left( {1 + \frac{\delta }{\varepsilon }}\right) }{\log \frac{1}{\delta }\log \frac{m}{\delta }}}\right)$，弗雷克森（Freksen）、卡马（Kamma）和拉森（Larsen）[FKL18]给出了定理1.8中所示的关于${v}_{\mathrm{{FH}}}$的一个复杂但精确的界，其中用于证明上界的困难实例与DKS构造中的一个${x}^{\prime }$相同。

---

<!-- Footnote -->

${}^{22}$ Which seems to be the only thing in [Mat08] related to a bound on $q$ .

${}^{22}$ 这似乎是[马特（Mat），2008年]中唯一与$q$的界相关的内容。

${}^{23}$ i.e. the DKS,graph,or block construction with $s = 1$ .

${}^{23}$ 即具有 $s = 1$ 的DKS、图或块构造。

${}^{24}$ Contraposition is $\left( {P \Rightarrow  Q}\right)  \Rightarrow  \left( {\neg Q \Rightarrow  \neg P}\right)$ and it does not quite prove what was just claimed without some assumptions that ${s}_{\mathrm{{DKS}}},{v}_{\mathrm{{FH}}}$ ,and $a$ do not behave too erratically.

${}^{24}$ 逆否命题是 $\left( {P \Rightarrow  Q}\right)  \Rightarrow  \left( {\neg Q \Rightarrow  \neg P}\right)$ ，并且如果没有一些关于 ${s}_{\mathrm{{DKS}}},{v}_{\mathrm{{FH}}}$ 和 $a$ 不会表现得过于不规则的假设，它并不能完全证明刚才所声称的内容。

<!-- Footnote -->

---

Theorem 1.8 ([FKL18]). There exist constants $C \geq  D > 0$ such that for every $\varepsilon ,\delta  \in  \left( {0,1}\right)$ and $m \in  {\mathbb{N}}_{1}$ the following holds. If $\frac{C\lg \frac{1}{\delta }}{{\varepsilon }^{2}} \leq  m < \frac{2}{{\varepsilon }^{2}\delta }$ then

定理1.8（[FKL18]）。存在常数 $C \geq  D > 0$ ，使得对于每个 $\varepsilon ,\delta  \in  \left( {0,1}\right)$ 和 $m \in  {\mathbb{N}}_{1}$ ，以下结论成立。如果 $\frac{C\lg \frac{1}{\delta }}{{\varepsilon }^{2}} \leq  m < \frac{2}{{\varepsilon }^{2}\delta }$ ，那么

$$
{v}_{\mathrm{{FH}}}\left( {m,\varepsilon ,\delta }\right)  = \Theta \left( {\sqrt{\varepsilon }\min \left\{  {\frac{\log \frac{\varepsilon m}{\log \frac{1}{\delta }}}{\log \frac{1}{\delta }},\sqrt{\frac{\log \frac{{\varepsilon }^{2}m}{\log \frac{1}{\delta }}}{\log \frac{1}{\delta }}}}\right\}  }\right) .
$$

Otherwise,if $m \geq  \frac{2}{{\varepsilon }^{2}\delta }$ then ${v}_{\mathrm{{FH}}}\left( {m,\varepsilon ,\delta }\right)  = 1$ . Moreover if $m < \frac{D\lg \frac{1}{\delta }}{{\varepsilon }^{2}}$ then ${v}_{\mathrm{{FH}}}\left( {m,\varepsilon ,\delta }\right)  = 0$ .

否则，如果 $m \geq  \frac{2}{{\varepsilon }^{2}\delta }$ 成立，那么 ${v}_{\mathrm{{FH}}}\left( {m,\varepsilon ,\delta }\right)  = 1$ 成立。此外，如果 $m < \frac{D\lg \frac{1}{\delta }}{{\varepsilon }^{2}}$ 成立，那么 ${v}_{\mathrm{{FH}}}\left( {m,\varepsilon ,\delta }\right)  = 0$ 成立。

Furthermore,if an $x \in  \{ 0,1{\} }^{d}$ satisfies ${v}_{\mathrm{{FH}}} < \parallel x{\parallel }_{2}^{-1} < 1$ then

此外，如果一个 $x \in  \{ 0,1{\} }^{d}$ 满足 ${v}_{\mathrm{{FH}}} < \parallel x{\parallel }_{2}^{-1} < 1$，那么

$$
\mathop{\Pr }\limits_{{f \sim  \mathrm{{FH}}}}\left\lbrack  {\left| {\parallel f\left( x\right) {\parallel }_{2}^{2} - \parallel x{\parallel }_{2}^{2}}\right|  > \varepsilon \parallel x{\parallel }_{2}^{2}}\right\rbrack   > \delta .
$$

This bound gives a tight tradeoff between target dimension $m$ ,distortion $\varepsilon$ ,error probability $\delta$ ,and ${\ell }_{\infty }/{\ell }_{2}$ constraint $v$ for Feature Hashing,while showing how to construct hard instances for Feature Hashing: Vectors with the shape $x = {\left( 1,\ldots ,1,0,\ldots ,0\right) }^{\top }$ are hard instances if they contain few 1s,meaning that Feature Hashing cannot preserve their norms within $1 \pm  \varepsilon$ with probability $\delta$ . Theorem 1.8 is used in corollary 1.9 to provide a tight tradeoff between $m,\varepsilon ,\delta ,v$ , and column sparsity $s$ for the DKS construction.

该边界给出了特征哈希（Feature Hashing）的目标维度 $m$、失真度 $\varepsilon$、错误概率 $\delta$ 以及 ${\ell }_{\infty }/{\ell }_{2}$ 约束 $v$ 之间的紧密权衡关系，同时展示了如何为特征哈希构造困难实例：如果形状为 $x = {\left( 1,\ldots ,1,0,\ldots ,0\right) }^{\top }$ 的向量中 1 的数量很少，那么它们就是困难实例，这意味着特征哈希无法以概率 $\delta$ 将它们的范数保持在 $1 \pm  \varepsilon$ 范围内。推论 1.9 中使用定理 1.8 给出了 DKS 构造中 $m,\varepsilon ,\delta ,v$ 和列稀疏性 $s$ 之间的紧密权衡关系。

Corollary 1.9. Let ${v}_{\mathrm{{DKS}}} \in  \left\lbrack  {1/\sqrt{d},1}\right\rbrack$ denote the largest ${\ell }_{\infty }/{\ell }_{2}$ ratio required, ${v}_{\mathrm{{FH}}}$ denote the ${\ell }_{\infty }/{\ell }_{2}$ constraint for Feature Hashing as defined in theorem 1.8,and ${s}_{\mathrm{{DKS}}} \in  \left\lbrack  m\right\rbrack$ as the minimum column sparsity such that the DKS construction with that sparsity is a JLD for the subset of vectors $x \in  {\mathbb{R}}^{d}$ that satisfy $\parallel x{\parallel }_{\infty }/\parallel x{\parallel }_{2} \leq  {v}_{\mathrm{{DKS}}}$ . Then

推论1.9。设${v}_{\mathrm{{DKS}}} \in  \left\lbrack  {1/\sqrt{d},1}\right\rbrack$表示所需的最大${\ell }_{\infty }/{\ell }_{2}$比率，${v}_{\mathrm{{FH}}}$表示定理1.8中所定义的特征哈希（Feature Hashing）的${\ell }_{\infty }/{\ell }_{2}$约束，${s}_{\mathrm{{DKS}}} \in  \left\lbrack  m\right\rbrack$表示最小列稀疏度，使得具有该稀疏度的DKS构造是满足$\parallel x{\parallel }_{\infty }/\parallel x{\parallel }_{2} \leq  {v}_{\mathrm{{DKS}}}$的向量子集$x \in  {\mathbb{R}}^{d}$的JLD。那么

$$
{s}_{\mathrm{{DKS}}} = \Theta \left( \frac{{v}_{\mathrm{{DKS}}}^{2}}{{v}_{\mathrm{{FH}}}^{2}}\right) .
$$

The proof of this corollary is deferred to section 2.2

该推论的证明推迟到2.2节

Jagadeesan [Jag19] generalised the result from [FKL18] to give a lower bound ${}^{25}$ on the $m,\varepsilon ,\delta$ , $v$ ,and $s$ tradeoff for any sparse Rademacher construction with a chosen column sparsity,e.g. the block and graph constructions, and gives a matching upper bound for the graph construction.

贾加迪桑（Jagadeesan）[Jag19]将[FKL18]的结果进行了推广，针对任意具有选定列稀疏性的稀疏拉德马赫构造（例如块构造和图构造），给出了$m,\varepsilon ,\delta$、$v$和$s$权衡的下界${}^{25}$，并为图构造给出了匹配的上界。

#### 1.4.2 Structured Johnson-Lindenstrauss Transforms

#### 1.4.2 结构化约翰逊 - 林登斯特劳斯变换

As we move away from the sparse JLDs we will slightly change our idea of what an efficient JLD is. In the previous section the JLDs were especially fast when the vectors were sparse, as the running time scaled with $\parallel x{\parallel }_{0}$ ,whereas we in this section will optimise for dense input vectors such that an embedding time of $O\left( {d\log d}\right)$ is a satisfying result.

当我们不再关注稀疏的约翰逊 - 林登斯特劳斯降维（JLD）时，我们对高效 JLD 的定义会稍有改变。在上一节中，当向量是稀疏的时候，JLD 特别快，因为运行时间与$\parallel x{\parallel }_{0}$成比例，而在本节中，我们将针对密集输入向量进行优化，使得嵌入时间为$O\left( {d\log d}\right)$是一个令人满意的结果。

The chronologically first asymptotic improvement over the original JLD construction is due to Ailon and Chazelle [AC09] who introduced the so-called Fast Johnson-Lindenstrauss Transform (FJLT). As mentioned in the previous section, [AC09] showed that we can use a very sparse (and therefore very fast) embedding matrix as long as the vectors have a low ${\ell }_{\infty }/{\ell }_{2}$ ratio, and furthermore that applying a randomised Walsh-Hadamard transform to a vector results in a low ${\ell }_{\infty }/{\ell }_{2}$ ratio with high probability. And so,the FJLT is defined as $f\left( x\right)  \mathrel{\text{:=}} {PHDx}$ ,where $P \in  {\mathbb{R}}^{m \times  d}$ is a sparse Achlioptas matrix with Gaussian entries and $q = \Theta \left( \frac{{\log }^{2}1/\delta }{d}\right) ,H \in  \{  - 1,1{\} }^{d \times  d}$ is a Walsh-Hadamard matrix ${}^{26}$ , and $D \in  \{  - 1,0,1{\} }^{d \times  d}$ is a random diagonal matrix with i.i.d. Rademachers on the diagonal. As the Walsh-Hadamard transform can be computed using a simple recursive formula,the expected embedding time becomes $O\left( {d\log d + m{\log }^{2}\frac{1}{\delta }}\right)$ . And as mentioned, [Mat08] showed that we can sample from a Rademacher rather than a Gaussian distribution when constructing the matrix $P$ . The embedding time improvement of FJLT over previous constructions depends on the relationship between $m$ and $d$ . If $m = \Theta \left( {{\varepsilon }^{-2}\log \frac{1}{\delta }}\right)$ and $m = O\left( {{\varepsilon }^{-4/3}{d}^{1/3}}\right)$ ,FJLT’s embedding time becomes bounded by the Walsh-Hadamard transform at $O\left( {d\log d}\right)$ ,but at $m = \Theta \left( {d}^{1/2}\right)$ FJLT is only barely faster than the original construction.

在时间顺序上，对原始JLD构造的首次渐进式改进归功于艾隆（Ailon）和查泽尔（Chazelle）[AC09]，他们引入了所谓的快速约翰逊 - 林登施特劳斯变换（Fast Johnson - Lindenstrauss Transform，FJLT）。如前一节所述，[AC09]表明，只要向量的${\ell }_{\infty }/{\ell }_{2}$比率较低，我们就可以使用非常稀疏（因此速度非常快）的嵌入矩阵，此外，对向量应用随机化的沃尔什 - 哈达玛变换（Walsh - Hadamard transform）很可能会得到较低的${\ell }_{\infty }/{\ell }_{2}$比率。因此，FJLT定义为$f\left( x\right)  \mathrel{\text{:=}} {PHDx}$，其中$P \in  {\mathbb{R}}^{m \times  d}$是一个具有高斯元素的稀疏阿赫利奥普塔斯矩阵（Achlioptas matrix），$q = \Theta \left( \frac{{\log }^{2}1/\delta }{d}\right) ,H \in  \{  - 1,1{\} }^{d \times  d}$是一个沃尔什 - 哈达玛矩阵${}^{26}$，$D \in  \{  - 1,0,1{\} }^{d \times  d}$是一个对角线上具有独立同分布拉德马赫变量（Rademachers）的随机对角矩阵。由于可以使用简单的递归公式计算沃尔什 - 哈达玛变换，因此预期的嵌入时间变为$O\left( {d\log d + m{\log }^{2}\frac{1}{\delta }}\right)$。如前所述，[Mat08]表明，在构造矩阵$P$时，我们可以从拉德马赫分布而非高斯分布中采样。与之前的构造相比，FJLT的嵌入时间改进取决于$m$和$d$之间的关系。如果$m = \Theta \left( {{\varepsilon }^{-2}\log \frac{1}{\delta }}\right)$且$m = O\left( {{\varepsilon }^{-4/3}{d}^{1/3}}\right)$，FJLT的嵌入时间受限于沃尔什 - 哈达玛变换，为$O\left( {d\log d}\right)$，但在$m = \Theta \left( {d}^{1/2}\right)$时，FJLT仅比原始构造略快。

---

<!-- Footnote -->

${}^{25}$ Here a lower bound refers to a lower bound on $v$ as a function of $m,\varepsilon ,\delta$ ,and $s$ .

${}^{25}$ 这里，下界指的是作为 $m,\varepsilon ,\delta$ 和 $s$ 的函数的 $v$ 的下界。

<!-- Footnote -->

---

Ailon and Liberty [AL09] improved the running time of the FJLT construction to $O\left( {d\log m}\right)$ for $m = O\left( {d}^{1/2 - \gamma }\right)$ for any fixed $\gamma  > 0$ . The increased applicable range of $m$ was achieved by applying multiple randomised Walsh-Hadamard transformations,i.e. replacing ${HD}$ with $\mathop{\prod }\limits_{i}H{D}^{\left( i\right) }$ ,where the ${D}^{\left( i\right) }$ s are a constant number of independent diagonal Rademacher matrices, as well as by replacing $P$ with ${BD}$ where $D$ is yet another diagonal matrix with Rademacher entries and $B$ is consecutive blocks of specific partial Walsh-Hadamard matrices (based on so-called binary dual BCH codes [see e.g. MS77]). The reduction in running time comes from altering the transform slightly by partitioning the input into consecutive blocks of length poly(m) and applying the randomised Walsh-Hadamard transforms to each of them independently. We will refer to this variant of FJLT as the BCHJL construction.

艾隆（Ailon）和利伯蒂（Liberty）[AL09]将FJLT构造的运行时间改进为$O\left( {d\log m}\right)$，其中$m = O\left( {d}^{1/2 - \gamma }\right)$对于任何固定的$\gamma  > 0$成立。通过应用多次随机沃尔什 - 哈达玛变换（Walsh - Hadamard transformations），即把${HD}$替换为$\mathop{\prod }\limits_{i}H{D}^{\left( i\right) }$（其中${D}^{\left( i\right) }$是固定数量的独立对角拉德马赫矩阵（Rademacher matrices）），以及把$P$替换为${BD}$（其中$D$是另一个具有拉德马赫元素的对角矩阵，$B$是特定部分沃尔什 - 哈达玛矩阵的连续块（基于所谓的二元对偶BCH码（binary dual BCH codes）[例如参见MS77]）），实现了$m$适用范围的扩大。运行时间的减少源于对变换进行了轻微修改，即将输入划分为长度为poly(m)的连续块，并对每个块独立应用随机沃尔什 - 哈达玛变换。我们将这种FJLT的变体称为BCHJL构造。

The next pattern of results has roots in compressed sensing and approaches the problem from another angle: Rather than being fast only when $m \ll  d$ ,they achieve $O\left( {d\log d}\right)$ embedding time even when $m$ is close to $d$ ,at the cost of $m$ being suboptimal. Before describing these constructions, let us set the scene by briefly introducing some concepts from compressed sensing.

下一组结果模式源于压缩感知（compressed sensing），并从另一个角度来处理这个问题：它们并非仅在$m \ll  d$时速度快，即使$m$接近$d$，也能实现$O\left( {d\log d}\right)$的嵌入时间，代价是$m$并非最优。在描述这些构造之前，让我们通过简要介绍压缩感知中的一些概念来做个铺垫。

Roughly speaking, compressed sensing concerns itself with recovering a sparse signal via a small number of linear measurements and a key concept here is the Restricted Isometry Property [CT05; CT06; CRT06; Don06].

大致来说，压缩感知关注的是通过少量线性测量来恢复稀疏信号，这里的一个关键概念是受限等距性质（Restricted Isometry Property）[CT05; CT06; CRT06; Don06]。

Definition 1.10 (Restricted Isometry Property). Let $d,m,k \in  {\mathbb{N}}_{1}$ with $m,k < d$ and $\varepsilon  \in  \left( {0,1}\right)$ . $A$ linear function $f : {\mathbb{R}}^{d} \rightarrow  {\mathbb{R}}^{m}$ is said to have the Restricted Isometry Property of order $k$ and level $\varepsilon$ (which we will denote as $\left( {k,\varepsilon }\right)$ -RIP) if for all $x \in  {\mathbb{R}}^{d}$ with $\parallel x{\parallel }_{0} \leq  k$ ,

定义1.10（受限等距性质）。设$d,m,k \in  {\mathbb{N}}_{1}$，其中$m,k < d$且$\varepsilon  \in  \left( {0,1}\right)$。若对于所有满足$\parallel x{\parallel }_{0} \leq  k$的$x \in  {\mathbb{R}}^{d}$，线性函数$f : {\mathbb{R}}^{d} \rightarrow  {\mathbb{R}}^{m}$都满足，则称该线性函数具有阶为$k$、水平为$\varepsilon$的受限等距性质（我们将其记为$\left( {k,\varepsilon }\right)$ -RIP）。

$$
\left| {\parallel f\left( x\right) {\parallel }_{2}^{2} - \parallel x{\parallel }_{2}^{2}}\right|  \leq  \varepsilon \parallel x{\parallel }_{2}^{2}. \tag{10}
$$

In the compressed sensing literature it has been shown [CT06; RV08] that the subsampled Hadamard transform (SHT) defined as $f\left( x\right)  \mathrel{\text{:=}} {SHx}$ ,has the $\left( {k,\varepsilon }\right)$ -RIP with high probability for $m = \Omega \left( {{\varepsilon }^{-2}k{\log }^{4}d}\right)$ while allowing a vector to be embedded in $O\left( {d\log d}\right)$ time. Here $H \in  \{  - 1,1{\} }^{d \times  d}$ is the Walsh-Hadamard matrix and $S \in  \{ 0,1{\} }^{m \times  d}$ samples $m$ entries of ${Hx}$ with replacement,i.e. each row in $S$ has one non-zero entry per row,which is chosen uniformly and independently,i.e. $S$ is a uniformly random feature selection matrix. Inspired by this transform and the FJLT mentioned previously, Ailon and Liberty [AL13] were able to show that the subsampled randomised Hadamard transform (SRHT) defined as $\bar{f}\left( x\right)  \mathrel{\text{:=}} {SHDx}$ ,is a JLT if $m = \Theta \left( {{\varepsilon }^{-4}\log \left| X\right| {\log }^{4}d}\right)$ . Once again $D$ denotes a random diagonal matrix with Rademacher entries,and $S$ and $H$ is as in the SHT. Some related results include Do et al. [Do+09] who before [AL13] were able to get a bound of $m = \Theta \left( {{\varepsilon }^{-2}{\log }^{3}\left| X\right| }\right)$ in the large set case where $\left| X\right|  \geq  d$ , Tro11 which showed how the SRHT construction approximately preserves the norms of a subspace of vectors,and [LL20] which modified the sampling matrix $S$ to improve precision when used as a preprocessor for support vector machines (SVMs) by sacrificing input data independence.

在压缩感知文献中，已有研究[CT06; RV08]表明，定义为$f\left( x\right)  \mathrel{\text{:=}} {SHx}$的下采样哈达玛变换（Subsampled Hadamard Transform，SHT），对于$m = \Omega \left( {{\varepsilon }^{-2}k{\log }^{4}d}\right)$而言，大概率具有$\left( {k,\varepsilon }\right)$ - 受限等距性质（Restricted Isometry Property，RIP），同时允许向量在$O\left( {d\log d}\right)$时间内完成嵌入。这里$H \in  \{  - 1,1{\} }^{d \times  d}$是沃尔什 - 哈达玛矩阵（Walsh - Hadamard matrix），$S \in  \{ 0,1{\} }^{m \times  d}$对${Hx}$的$m$个元素进行有放回采样，即$S$中的每一行都有一个非零元素，该元素是均匀且独立选取的，也就是说，$S$是一个均匀随机特征选择矩阵。受此变换以及前文提到的FJLT启发，艾隆（Ailon）和利伯蒂（Liberty）[AL13]证明了，定义为$\bar{f}\left( x\right)  \mathrel{\text{:=}} {SHDx}$的下采样随机哈达玛变换（Subsampled Randomised Hadamard Transform，SRHT），若满足$m = \Theta \left( {{\varepsilon }^{-4}\log \left| X\right| {\log }^{4}d}\right)$，则是一个约翰逊 - 林登斯特劳斯引理（Johnson - Lindenstrauss Lemma，JLT）变换。同样，$D$表示一个具有拉德马赫（Rademacher）元素的随机对角矩阵，$S$和$H$的定义与SHT中的相同。一些相关结果包括：多（Do）等人[Do + 09]在[AL13]之前，在$\left| X\right|  \geq  d$的大集合情况下得到了$m = \Theta \left( {{\varepsilon }^{-2}{\log }^{3}\left| X\right| }\right)$的界；特罗（Tro）[Tro11]展示了SRHT构造如何近似保留向量子空间的范数；以及林（Lin）和李（Li）[LL20]对采样矩阵$S$进行修改，在作为支持向量机（Support Vector Machines，SVMs）的预处理器使用时，通过牺牲输入数据的独立性来提高精度。

---

<!-- Footnote -->

${}^{26}$ One definition of the Walsh-Hadamard matrices is that the entries are ${H}_{ij} = {\left( -1\right) }^{\langle i - 1,j - 1\rangle }$ for all $i,j \in  \left\lbrack  d\right\rbrack$ ,where $\langle a,b\rangle$ denote the dot product of the $\left( {\lg d}\right)$ -bit vectors corresponding to the binary representation of the numbers $a,b \in  \{ 0,\ldots ,d - 1\}$ ,and $d$ is a power of two. To illustrate its recursive nature,a large Walsh-Hadamard matrix can be described as a Kronecker product of smaller Walsh-Hadamard matrices,i.e. if $d > 2$ and ${H}^{\left( n\right) }$ refers to a $n \times  n$ Walsh-Hadamard matrix,then ${H}^{\left( d\right) } = {H}^{\left( 2\right) } \otimes  {H}^{\left( d/2\right) } = \left( \begin{matrix} {H}^{\left( d/2\right) } & {H}^{\left( d/2\right) } \\  {H}^{\left( d/2\right) } &  - {H}^{\left( d/2\right) } \end{matrix}\right)$ .

${}^{26}$ 沃尔什 - 哈达玛矩阵（Walsh - Hadamard matrices）的一种定义是，对于所有的 $i,j \in  \left\lbrack  d\right\rbrack$，其元素为 ${H}_{ij} = {\left( -1\right) }^{\langle i - 1,j - 1\rangle }$，其中 $\langle a,b\rangle$ 表示对应于数字 $a,b \in  \{ 0,\ldots ,d - 1\}$ 的二进制表示的 $\left( {\lg d}\right)$ 位向量的点积，并且 $d$ 是 2 的幂。为了说明其递归性质，一个大的沃尔什 - 哈达玛矩阵可以描述为较小的沃尔什 - 哈达玛矩阵的克罗内克积（Kronecker product），即如果 $d > 2$ 且 ${H}^{\left( n\right) }$ 指的是一个 $n \times  n$ 沃尔什 - 哈达玛矩阵，那么 ${H}^{\left( d\right) } = {H}^{\left( 2\right) } \otimes  {H}^{\left( d/2\right) } = \left( \begin{matrix} {H}^{\left( d/2\right) } & {H}^{\left( d/2\right) } \\  {H}^{\left( d/2\right) } &  - {H}^{\left( d/2\right) } \end{matrix}\right)$。

<!-- Footnote -->

---

<!-- Media -->

$$
\left( \begin{matrix} {t}_{0} & {t}_{1} & {t}_{2} & \cdots & {t}_{d - 1} \\  {t}_{-1} & {t}_{0} & {t}_{1} & \cdots & {t}_{d - 2} \\  {t}_{-2} & {t}_{-1} & {t}_{0} & \cdots & {t}_{d - 3} \\  \vdots & \vdots & \vdots &  \ddots  & \vdots \\  {t}_{-\left( {m - 1}\right) } & {t}_{-\left( {m - 2}\right) } & {t}_{-\left( {m - 3}\right) } & \cdots & {t}_{d - m} \end{matrix}\right) 
$$

Figure 1: The structure of a Toeplitz matrix.

图1：托普利茨矩阵（Toeplitz matrix）的结构。

<!-- Media -->

This target dimension bound of [AL13] was later tightened by Krahmer and Ward [KW11], who showed that $m = \Theta \left( {{\varepsilon }^{-2}\log \left| X\right| {\log }^{4}d}\right)$ suffices for the SRHT. This was a corollary of a more general result,namely that if $\sigma  : {\mathbb{R}}^{d} \rightarrow  {\mathbb{R}}^{d}$ applies random signs equivalently to the $D$ matrices mentioned previously and $f : {\mathbb{R}}^{d} \rightarrow  {\mathbb{R}}^{m}$ has the $\left( {\Omega \left( {\log \left| X\right| }\right) ,\varepsilon /4}\right)$ -RIP then $f \circ  \sigma$ is a JLT with high probability. An earlier result by Baraniuk et al. [Bar+08] showed that a transform sampled from a JLD has the $\left( {\mathcal{O}\left( {{\varepsilon }^{2}m/\log d}\right) ,\varepsilon }\right)$ -RIP with high probability. And so,as one might have expected from their appearance, the Johnson-Lindenstrauss Lemma and the Restricted Isometry Property are indeed cut from the same cloth.

[AL13]的这一目标维度界后来被克拉默（Krahmer）和沃德（Ward）[KW11]收紧，他们证明了对于子采样随机哈达玛变换（SRHT），$m = \Theta \left( {{\varepsilon }^{-2}\log \left| X\right| {\log }^{4}d}\right)$就足够了。这是一个更一般结果的推论，即如果$\sigma  : {\mathbb{R}}^{d} \rightarrow  {\mathbb{R}}^{d}$对前面提到的$D$矩阵等效地应用随机符号，并且$f : {\mathbb{R}}^{d} \rightarrow  {\mathbb{R}}^{m}$具有$\left( {\Omega \left( {\log \left| X\right| }\right) ,\varepsilon /4}\right)$ - 受限等距性质（RIP），那么$f \circ  \sigma$大概率是一个约翰逊 - 林登斯特劳斯变换（JLT）。巴拉纽克（Baraniuk）等人[Bar + 08]的早期结果表明，从联合拉普拉斯分布（JLD）中采样的变换大概率具有$\left( {\mathcal{O}\left( {{\varepsilon }^{2}m/\log d}\right) ,\varepsilon }\right)$ - 受限等距性质（RIP）。因此，正如从它们的表现所预期的那样，约翰逊 - 林登斯特劳斯引理（Johnson - Lindenstrauss Lemma）和受限等距性质（Restricted Isometry Property）确实是一脉相承的。

Another transform from the compressed sensing literature uses so-called Toeplitz or partial circulant matrices [Baj+07: Rau09: Rom09: Hau+10: RRT12: Baj12: DJR19], which can be defined in the following way. For $m,d \in  {\mathbb{N}}_{1}$ we say that $T \in  {\mathbb{R}}^{m \times  d}$ is a real Toeplitz matrix if there exists ${t}_{-\left( {m - 1}\right) },{t}_{-\left( {m - 2}\right) }\ldots ,{t}_{d - 1} \in  \mathbb{R}$ such that ${T}_{ij} = {t}_{j - i}$ . This has the effect that the entries on any one diagonal are the same (see fig. 1) and computing the matrix-vector product corresponds to computing the convolution with a vector of the $t$ s. Partial circulant matrices are special cases of Toeplitz matrices where the diagonals "wrap around" the ends of the matrix,i.e. ${t}_{-i} = {t}_{d - i}$ for all $i \in  \left\lbrack  {m - 1}\right\rbrack$ .

压缩感知文献中的另一种变换使用了所谓的托普利茨（Toeplitz）矩阵或部分循环矩阵[Baj+07: Rau09: Rom09: Hau+10: RRT12: Baj12: DJR19]，它们可以按以下方式定义。对于$m,d \in  {\mathbb{N}}_{1}$，若存在${t}_{-\left( {m - 1}\right) },{t}_{-\left( {m - 2}\right) }\ldots ,{t}_{d - 1} \in  \mathbb{R}$使得${T}_{ij} = {t}_{j - i}$ ，则称$T \in  {\mathbb{R}}^{m \times  d}$为实托普利茨矩阵。这意味着任意一条对角线上的元素都相同（见图1），并且计算矩阵 - 向量乘积相当于计算与由$t$组成的向量的卷积。部分循环矩阵是托普利茨矩阵的特殊情况，其中对角线会“环绕”矩阵的两端，即对于所有的$i \in  \left\lbrack  {m - 1}\right\rbrack$ ，有${t}_{-i} = {t}_{d - i}$ 。

As a JLT,the Toeplitz construction is $f\left( x\right)  \mathrel{\text{:=}} {TDx}$ ,where $T \in  \{  - 1,1{\} }^{m \times  d}$ is a Toeplitz matrix with i.i.d. Rademacher entries and $D \in  \{  - 1,0,1{\} }^{d \times  d}$ is a diagonal matrix with Rademacher entries as usual. Note that the convolution of two vectors corresponds to the entrywise product in Fourier space, and we can therefore employ fast Fourier transform (FFT) to embed a vector with the Toeplitz construction in time $O\left( {d\log d}\right)$ . This time can even be reduced to $O\left( {d\log m}\right)$ as we realise that by partitioning $T$ into $\frac{d}{m}$ consecutive blocks of size $m \times  m$ ,each block is also a Toeplitz matrix,and by applying each individually the embedding time becomes $O\left( {\frac{d}{m}m\log m}\right)$ .

作为一个联合限制特征值（JLT），托普利茨（Toeplitz）构造为$f\left( x\right)  \mathrel{\text{:=}} {TDx}$，其中$T \in  \{  - 1,1{\} }^{m \times  d}$是一个具有独立同分布拉德马赫（Rademacher）元素的托普利茨矩阵，而$D \in  \{  - 1,0,1{\} }^{d \times  d}$是一个像往常一样具有拉德马赫元素的对角矩阵。请注意，两个向量的卷积对应于傅里叶（Fourier）空间中的逐元素乘积，因此我们可以采用快速傅里叶变换（FFT）在时间$O\left( {d\log d}\right)$内用托普利茨构造嵌入一个向量。当我们意识到将$T$划分为$\frac{d}{m}$个大小为$m \times  m$的连续块时，每个块也是一个托普利茨矩阵，并且通过分别应用，嵌入时间变为$O\left( {\frac{d}{m}m\log m}\right)$，此时这个时间甚至可以减少到$O\left( {d\log m}\right)$。

Combining the result from [KW11] with RIP bounds for Toeplitz matrices [RRT12] gives that $m = \Theta \left( {{\varepsilon }^{-1}{\log }^{3/2}\left| X\right| {\log }^{3/2}d + {\varepsilon }^{-2}\log \left| X\right| {\log }^{4}d}\right)$ is sufficient for the Toeplitz construction to be a JLT with high probability. However, the Toeplitz construction has also been studied directly as a JLD without going via its RIP bounds. Hinrichs and Vybíral [HV11] showed that $m = \Theta \left( {{\varepsilon }^{-2}{\log }^{3}\frac{1}{\delta }}\right)$ is sufficient for the Toeplitz construction,and this bound was improved shortly thereafter in [Vyb11] to $m = \Theta \left( {{\varepsilon }^{-2}{\log }^{2}\frac{1}{\delta }}\right)$ . The question then is if we can tighten the analysis to shave off the last log factor and get the elusive result of a JLD with optimal target dimension and $O\left( {d\log d}\right)$ embedding time even when $m$ is close to $d$ . Sadly,this is not the case as Freksen and Larsen [FL20] showed that there exists vectors ${}^{27}$ that necessitates $m = \Omega \left( {{\varepsilon }^{-2}{\log }^{2}\frac{1}{\delta }}\right)$ for the Toeplitz construction.

将[KW11]的结果与托普利茨矩阵（Toeplitz matrices）的受限等距性质（RIP）界[RRT12]相结合，可以得出，$m = \Theta \left( {{\varepsilon }^{-1}{\log }^{3/2}\left| X\right| {\log }^{3/2}d + {\varepsilon }^{-2}\log \left| X\right| {\log }^{4}d}\right)$足以使托普利茨构造以高概率成为约翰逊 - 林登斯特劳斯引理（JLT）。然而，托普利茨构造也被直接作为约翰逊 - 林登斯特劳斯维数约简（JLD）进行研究，而无需通过其受限等距性质界。欣里克斯（Hinrichs）和维比拉尔（Vybíral）[HV11]表明，$m = \Theta \left( {{\varepsilon }^{-2}{\log }^{3}\frac{1}{\delta }}\right)$足以用于托普利茨构造，此后不久，[Vyb11]将此界改进为$m = \Theta \left( {{\varepsilon }^{-2}{\log }^{2}\frac{1}{\delta }}\right)$。那么问题是，我们是否可以加强分析，去掉最后的对数因子，从而得到一个具有最优目标维度和$O\left( {d\log d}\right)$嵌入时间的约翰逊 - 林登斯特劳斯维数约简的难以捉摸的结果，即使当$m$接近$d$时也是如此。遗憾的是，情况并非如此，因为弗雷克森（Freksen）和拉森（Larsen）[FL20]表明，存在向量${}^{27}$，使得托普利茨构造必须满足$m = \Omega \left( {{\varepsilon }^{-2}{\log }^{2}\frac{1}{\delta }}\right)$。

Just as JLTs are used as preprocessors to speed up algorithms that solve the problems we actually care about, we can also use a JLT to speed up other JLTs in what one could refer to as compound JLTs. More explicitely if ${f}_{1} : {\mathbb{R}}^{d} \rightarrow  {\mathbb{R}}^{{d}^{\prime }}$ and ${f}_{2} : {\mathbb{R}}^{{d}^{\prime }} \rightarrow  {\mathbb{R}}^{m}$ with $m \ll  {d}^{\prime } \ll  d$ are two JLTs and computing ${f}_{1}\left( x\right)$ is fast,we could hope that computing $\left( {{f}_{2} \circ  {f}_{1}}\right) \left( x\right)$ is fast as well as ${f}_{2}$ only need to handle ${d}^{\prime }$ dimensional vectors and hope that $\left( {{f}_{2} \circ  {f}_{1}}\right)$ preserves the norm sufficiently well since both ${f}_{1}$ and ${f}_{2}$ approximately preserve norms individually. As presented here,the obvious candidate for ${f}_{1}$ is one of the RIP-based JLDs,which was succesfully applied in [BK17]. In their construction,which we will refer to as ${\mathrm{{GRHD}}}^{28},{f}_{1}$ is the SRHT and ${f}_{2}$ is the dense Rademacher construction (i.e. $f\left( x\right)  \mathrel{\text{:=}} {A}_{\text{Rad }}{SHDx}$ ),and it can embed a vector in time $O\left( {d\log m}\right)$ for $m = O\left( {d}^{1/2 - \gamma }\right)$ for any fixed $\gamma  > 0$ . This is a similar result to the construction of Ailon and Liberty [AL09], but unlike that construction, GRHD handles the remaining range of $m$ more gracefully as for any $r \in  \left\lbrack  {1/2,1}\right\rbrack$ and $m = O\left( {d}^{r}\right)$ ,the embedding time for GRHD becomes $O\left( {{d}^{2r}{\log }^{4}d}\right)$ . However the main selling point of the GRHD construction is that it allows the simultaneous embedding of sufficiently large sets of points $X$ to be computed in total time $O\left( {\left| X\right| d\log m}\right)$ ,even when $m = \Theta \left( {d}^{1 - \gamma }\right)$ for any fixed $\gamma  > 0$ ,by utilising fast matrix-matrix multiplication techniques [LR83].

正如约翰逊 - 林登施特劳斯变换（JLTs）被用作预处理器来加速解决我们实际关心问题的算法一样，我们还可以使用一个JLT来加速其他JLT，这可以称为复合JLT。更明确地说，如果${f}_{1} : {\mathbb{R}}^{d} \rightarrow  {\mathbb{R}}^{{d}^{\prime }}$和${f}_{2} : {\mathbb{R}}^{{d}^{\prime }} \rightarrow  {\mathbb{R}}^{m}$（其中$m \ll  {d}^{\prime } \ll  d$）是两个JLT，并且计算${f}_{1}\left( x\right)$的速度很快，我们可以期望计算$\left( {{f}_{2} \circ  {f}_{1}}\right) \left( x\right)$的速度也很快，因为${f}_{2}$只需要处理${d}^{\prime }$维向量，并且希望$\left( {{f}_{2} \circ  {f}_{1}}\right)$能充分保留向量的范数，因为${f}_{1}$和${f}_{2}$各自都能近似地保留范数。如这里所介绍的，${f}_{1}$的一个明显候选者是基于受限等距性质（RIP）的约翰逊 - 林登施特劳斯降维（JLD）方法之一，该方法已在[BK17]中成功应用。在他们的构造中，我们将其称为${\mathrm{{GRHD}}}^{28},{f}_{1}$，其中${\mathrm{{GRHD}}}^{28},{f}_{1}$是稀疏随机哈达玛变换（SRHT），${f}_{2}$是稠密拉德马赫构造（即$f\left( x\right)  \mathrel{\text{:=}} {A}_{\text{Rad }}{SHDx}$），并且对于任何固定的$\gamma  > 0$，它可以在时间$O\left( {d\log m}\right)$内嵌入一个向量，其中$m = O\left( {d}^{1/2 - \gamma }\right)$。这与艾隆（Ailon）和利伯蒂（Liberty）[AL09]的构造结果类似，但与该构造不同的是，广义随机哈达玛降维（GRHD）能更优雅地处理$m$的剩余范围，因为对于任何$r \in  \left\lbrack  {1/2,1}\right\rbrack$和$m = O\left( {d}^{r}\right)$，GRHD的嵌入时间变为$O\left( {{d}^{2r}{\log }^{4}d}\right)$。然而，GRHD构造的主要卖点在于，通过利用快速矩阵 - 矩阵乘法技术[LR83]，即使对于任何固定的$\gamma  > 0$有$m = \Theta \left( {d}^{1 - \gamma }\right)$，它也允许在总时间$O\left( {\left| X\right| d\log m}\right)$内同时嵌入足够大的点集$X$。

Another compound JLD is based on the so-called lean Walsh transforms (LWT) [LAS11], which are defined based on so-called seed matrices. For $r,c \in  {\mathbb{N}}_{1}$ we say that ${A}_{1} \in  {\mathbb{C}}^{r \times  c}$ is a seed matrix if $r < c$ ,its columns are of unit length,and its rows are pairwise orthogonal and have the same ${\ell }_{2}$ norm. As such,partial Walsh-Hadamard matrices and partial Fourier matrices are seed matrices (up to normalisation); however, for simplicity's sake we will keep it real by focusing on partial Walsh-Hadamard matrices. We can then define a LWT of order $l \in  {\mathbb{N}}_{1}$ based on this seed as ${A}_{l} \mathrel{\text{:=}} {A}_{1}^{\otimes l} = {A}_{1} \otimes  \cdots  \otimes  {A}_{1}$ ,where $\otimes$ denotes the Kronecker product,which we will quickly define. Let $A$ be a $m \times  n$ matrix and $B$ be a $p \times  q$ matrix,then the Kronecker product $A \otimes  B$ is the ${mp} \times  {nq}$ block matrix defined as

另一种化合物JLD基于所谓的精简沃尔什变换（LWT）[LAS11]，该变换是基于所谓的种子矩阵定义的。对于$r,c \in  {\mathbb{N}}_{1}$，若$r < c$，其列向量长度为单位长度，且其行向量两两正交并具有相同的${\ell }_{2}$范数，则称${A}_{1} \in  {\mathbb{C}}^{r \times  c}$为种子矩阵。因此，部分沃尔什 - 哈达玛矩阵（partial Walsh - Hadamard matrices）和部分傅里叶矩阵（partial Fourier matrices）（经归一化处理后）均为种子矩阵；不过，为简便起见，我们将专注于部分沃尔什 - 哈达玛矩阵以保持实数运算。然后，我们可以基于此种子定义阶数为$l \in  {\mathbb{N}}_{1}$的精简沃尔什变换为${A}_{l} \mathrel{\text{:=}} {A}_{1}^{\otimes l} = {A}_{1} \otimes  \cdots  \otimes  {A}_{1}$，其中$\otimes$表示克罗内克积（Kronecker product），下面我们将快速给出其定义。设$A$为一个$m \times  n$矩阵，$B$为一个$p \times  q$矩阵，则克罗内克积$A \otimes  B$是定义为如下形式的${mp} \times  {nq}$分块矩阵

$$
A \otimes  B \mathrel{\text{:=}} \left( \begin{matrix} {A}_{11}B & \cdots & {A}_{1n}B \\  \vdots &  \ddots  & \vdots \\  {A}_{m1}B & \cdots & {A}_{mn}B \end{matrix}\right) .
$$

---

<!-- Footnote -->

${}^{27}$ Curiously,the hard instances for the Toeplitz construction are very similar to the hard instances for Feature Hashing used in [FKL18].

${}^{27}$ 奇怪的是，托普利茨构造（Toeplitz construction）的困难实例与[FKL18]中使用的特征哈希（Feature Hashing）的困难实例非常相似。

${}^{28}$ Due to the choice of matrix names in [BK17].

${}^{28}$ 这是由于[BK17]中矩阵名称的选择。

<!-- Footnote -->

---

Note that ${A}_{l}$ is a ${r}^{l} \times  {c}^{l}$ matrix and that any Walsh-Hadamard matrix can be written as ${A}_{l}$ for some $l$ and the $2 \times  2$ Walsh-Hadamard matrix ${}^{29}$ as ${A}_{1}$ . Furthermore,for a constant sized seed the time complexity of applying ${A}_{l}$ to a vector is $\bar{O}\left( {c}^{l}\right)$ by using an algorithm similar to FFT. We can then define the compound transform which we will refer to as LWTJL,as $f\left( x\right)  \mathrel{\text{:=}} G{A}_{l}{Dx}$ ,where $D \in  \{  - 1,1{\} }^{d \times  d}$ is a diagonal matrix with Rademacher entries, ${A}_{l} \in  {\mathbb{R}}^{{r}^{l} \times  d}$ is a LWT,and $G \in  {\mathbb{R}}^{m \times  {r}^{l}}$ is a JLT,and $r$ and $c$ are constants. One way to view LWTJL is as a variant of GRHD where the subsampling occurs on the seed matrix rather than the final Walsh-Hadamard matrix. If $G$ can be applied in $O\left( {{r}^{l}\log {r}^{l}}\right)$ time,e.g. if $G$ is the BCHJL construction [AL09] and $m = O\left( {r}^{l\left( {1/2 - \gamma }\right) }\right)$ , the total embedding time becomes $O\left( d\right)$ ,as ${r}^{l} = {d}^{\alpha }$ for some $\alpha  < 1$ . However,in order to prove that LWTJL satisfies lemma 1.2 the analysis of [LAS11] imposes a few requirements on $r,c$ ,and the vectors we wish to embed,namely that $\log r/\log c \geq  1 - {2\delta }$ and $v = O\left( {{m}^{-1/2}{d}^{-\delta }}\right)$ ,where $v$ is an upper bound on the ${\ell }_{\infty }/{\ell }_{2}$ ratio as introduced at the end of section 1.4.1. The bound on $v$ is somewhat tight as shown in proposition 1.11

请注意，${A}_{l}$ 是一个 ${r}^{l} \times  {c}^{l}$ 矩阵，并且任何沃尔什 - 哈达玛矩阵（Walsh - Hadamard matrix）都可以写成 ${A}_{l}$ 的形式，其中 $l$ 为某个值，而 $2 \times  2$ 沃尔什 - 哈达玛矩阵 ${}^{29}$ 可写成 ${A}_{1}$ 的形式。此外，对于固定大小的种子，使用类似于快速傅里叶变换（FFT）的算法将 ${A}_{l}$ 应用于向量的时间复杂度为 $\bar{O}\left( {c}^{l}\right)$。然后，我们可以定义复合变换，我们将其称为 LWTJL，即 $f\left( x\right)  \mathrel{\text{:=}} G{A}_{l}{Dx}$，其中 $D \in  \{  - 1,1{\} }^{d \times  d}$ 是一个具有拉德马赫（Rademacher）元素的对角矩阵，${A}_{l} \in  {\mathbb{R}}^{{r}^{l} \times  d}$ 是一个 LWT，$G \in  {\mathbb{R}}^{m \times  {r}^{l}}$ 是一个 JLT，并且 $r$ 和 $c$ 是常数。一种看待 LWTJL 的方式是将其视为广义随机哈达玛设计（GRHD）的一种变体，其中子采样发生在种子矩阵上，而不是最终的沃尔什 - 哈达玛矩阵上。如果 $G$ 可以在 $O\left( {{r}^{l}\log {r}^{l}}\right)$ 时间内应用，例如，如果 $G$ 是 BCHJL 构造 [AL09] 且 $m = O\left( {r}^{l\left( {1/2 - \gamma }\right) }\right)$，则总嵌入时间变为 $O\left( d\right)$，因为对于某个 $\alpha  < 1$ 有 ${r}^{l} = {d}^{\alpha }$。然而，为了证明 LWTJL 满足引理 1.2，[LAS11] 的分析对 $r,c$ 以及我们希望嵌入的向量施加了一些要求，即 $\log r/\log c \geq  1 - {2\delta }$ 和 $v = O\left( {{m}^{-1/2}{d}^{-\delta }}\right)$，其中 $v$ 是第 1.4.1 节末尾引入的 ${\ell }_{\infty }/{\ell }_{2}$ 比率的上界。如命题 1.11 所示，$v$ 的界有些严格

Proposition 1.11. For any seed matrix define LWT as the LWTJL distribution seeded with that matrix. Then for all $\delta  \in  \left( {0,1}\right)$ ,there exists a vector $x \in  {\mathbb{C}}^{d}$ (or $x \in  {\mathbb{R}}^{d}$ ,if the seed matrix is a real matrix) satisfying $\parallel x{\parallel }_{\infty }/\parallel x{\parallel }_{2} = \Theta \left( {{\log }^{-1/2}\frac{1}{\delta }}\right)$ such that

命题1.11。对于任何种子矩阵，将LWT定义为以该矩阵为种子的LWTJL分布。那么对于所有$\delta  \in  \left( {0,1}\right)$，存在一个向量$x \in  {\mathbb{C}}^{d}$（如果种子矩阵是实矩阵，则为$x \in  {\mathbb{R}}^{d}$）满足$\parallel x{\parallel }_{\infty }/\parallel x{\parallel }_{2} = \Theta \left( {{\log }^{-1/2}\frac{1}{\delta }}\right)$，使得

$$
\mathop{\Pr }\limits_{{f \sim  \mathrm{{LWT}}}}\left\lbrack  {f\left( x\right)  = \mathbf{0}}\right\rbrack   > \delta  \tag{11}
$$

The proof of proposition 1.11 can be found in section 2.3,and it is based on constructing $x$ as a few copies of a vector that is orthogonal to the rows of the seed matrix.

命题1.11的证明可在第2.3节中找到，它基于将$x$构造为与种子矩阵的行正交的向量的若干副本。

The last JLD we will cover is based on so-called Kac random walks, and despite Ailon and Chazelle [AC09] conjecturing that such a construction could satisfy lemma 1.1, it was not until Jain et al. [Jai+20] that a proof was finally at hand. As with the lean Walsh transforms above, let us first define Kac random walks before describing how they can be used to construct JLDs. A Kac random walk is a Markov chain of linear transformations, where for each step we choose two coordinates at random and perform a random rotation on the plane spanned by these two coordinates, or more formally:

我们将介绍的最后一种联合线性设计（JLD）基于所谓的卡茨随机游走（Kac random walks）。尽管艾隆（Ailon）和查泽尔（Chazelle）[AC09]猜想这种构造可能满足引理1.1，但直到贾恩（Jain）等人[Jai+20]才最终给出了证明。与上述精简沃尔什变换（lean Walsh transforms）一样，在描述如何用它们构造联合线性设计之前，让我们先定义卡茨随机游走。卡茨随机游走是一个线性变换的马尔可夫链，在每一步中，我们随机选择两个坐标，并对这两个坐标所张成的平面进行随机旋转，更正式的定义如下：

Definition 1.12 (Kac random walk [Kac56]). For a given dimention $d \in  {\mathbb{N}}_{1}$ ,let ${K}^{\left( 0\right) } \mathrel{\text{:=}} I \in  \{ 0,1{\} }^{d \times  d}$ be the identity matrix,and for each $t > 0$ sample $\left( {{i}_{t},{j}_{t}}\right)  \in  \left( \begin{matrix} \left\lbrack  d\right\rbrack  \\  2 \end{matrix}\right)$ and ${\theta }_{t} \in  \lbrack 0,{2\pi })$ independently and uniformly at random. Then define the Kac random walk of length $t$ as ${K}^{\left( t\right) } \mathrel{\text{:=}} {R}^{\left( {i}_{t},{j}_{t},{\theta }_{t}\right) }{K}^{\left( t - 1\right) }$ ,where ${R}^{\left( i,j,\theta \right) } \in  {\mathbb{R}}^{d \times  d}$ is the rotation in the(i,j)plane by $\theta$ and is given by

定义1.12（卡茨随机游走 [Kac56]）。对于给定的维度 $d \in  {\mathbb{N}}_{1}$，设 ${K}^{\left( 0\right) } \mathrel{\text{:=}} I \in  \{ 0,1{\} }^{d \times  d}$ 为单位矩阵，并且对于每个 $t > 0$，独立且均匀地随机采样 $\left( {{i}_{t},{j}_{t}}\right)  \in  \left( \begin{matrix} \left\lbrack  d\right\rbrack  \\  2 \end{matrix}\right)$ 和 ${\theta }_{t} \in  \lbrack 0,{2\pi })$。然后将长度为 $t$ 的卡茨随机游走定义为 ${K}^{\left( t\right) } \mathrel{\text{:=}} {R}^{\left( {i}_{t},{j}_{t},{\theta }_{t}\right) }{K}^{\left( t - 1\right) }$，其中 ${R}^{\left( i,j,\theta \right) } \in  {\mathbb{R}}^{d \times  d}$ 是在 (i,j) 平面上旋转 $\theta$ 角度，其表达式为

$$
{R}^{\left( i,j,\theta \right) }{e}_{k} \mathrel{\text{:=}} {e}_{k}
$$

$\forall k \notin  \{ i,j\}$

$$
{R}^{\left( i,j,\theta \right) }\left( {a{e}_{i} + b{e}_{j}}\right)  \mathrel{\text{:=}} \left( {a\cos \theta  - b\sin \theta }\right) {e}_{i} + \left( {a\sin \theta  + b\cos \theta }\right) {e}_{j}.
$$

---

<!-- Footnote -->

${}^{29}$ Here we ignore the $r < c$ requirement of seed matrices.

${}^{29}$ 这里我们忽略种子矩阵的$r < c$要求。

<!-- Footnote -->

---

The main JLD introduced in [Jai+20], which we will refer to as KacJL, is a compound JLD where both ${f}_{1}$ and ${f}_{2}$ consists of a Kac random walk followed by subsampling,which can be defined more formally in the following way. Let ${T}_{1} \mathrel{\text{:=}} \Theta \left( {d\log d}\right)$ be the length of the first Kac random walk, ${d}^{\prime } \mathrel{\text{:=}} \min \left\{  {d,\Theta \left( {{\varepsilon }^{-2}\log \left| X\right| {\log }^{2}\log \left| X\right| {\log }^{3}d}\right) }\right\}$ be the intermediate dimension, ${T}_{2} \mathrel{\text{:=}} \Theta \left( {{d}^{\prime }\log \left| X\right| }\right)$ be the length of the second Kac random walk,and $m \mathrel{\text{:=}} \Theta \left( {{\varepsilon }^{-2}\log \left| X\right| }\right)$ be the target dimension,and then define the JLT as $f\left( x\right)  = \left( {{f}_{2} \circ  {f}_{1}}\right) \left( x\right)  \mathrel{\text{:=}} {S}^{\left( m,{d}^{\prime }\right) }{K}^{\left( {T}_{2}\right) }{S}^{\left( {d}^{\prime },d\right) }{K}^{\left( {T}_{1}\right) }x$ ,where ${K}^{\left( {T}_{1}\right) } \in  {\mathbb{R}}^{d \times  d}$ and ${K}^{\left( {T}_{2}\right) } \in  {\mathbb{R}}^{{d}^{\prime } \times  {d}^{\prime }}$ are independent Kac random walks of length ${T}_{1}$ and ${T}_{2}$ ,respectively,and ${S}^{\left( {d}^{\prime },d\right) } \in  \{ 0,1{\} }^{{d}^{\prime } \times  d}$ and ${S}^{\left( m,{d}^{\prime }\right) } \in  \{ 0,1{\} }^{m \times  {d}^{\prime }}$ projects onto the first ${d}^{\prime }$ and $m$ coordinates ${}^{30}$ , respectively. Since ${K}^{\left( T\right) }$ can be applied in time $O\left( T\right)$ ,the KacJL construction is JLD with embedding time $O\left( {d\log d + \min \left\{  {d\log \left| X\right| ,{\varepsilon }^{-2}{\log }^{2}\left| X\right| {\log }^{2}\log \left| X\right| {\log }^{3}d}\right\}  }\right)$ with asymptotically optimal target dimension,and by only applying the first part $\left( {f}_{1}\right)$ ,KacJL achieves an embedding time of $O\left( {d\log d}\right)$ but with a suboptimal target dimension of $O\left( {{\varepsilon }^{-2}\log \left| X\right| {\log }^{2}\log \left| X\right| {\log }^{3}d}\right)$ .

文献[Jai+20]中引入的主要联合线性降维（JLD）方法，我们将其称为KacJL，是一种复合JLD，其中${f}_{1}$和${f}_{2}$均由Kac随机游走（Kac random walk）后接子采样组成，可按以下方式更正式地定义。设${T}_{1} \mathrel{\text{:=}} \Theta \left( {d\log d}\right)$为第一次Kac随机游走的长度，${d}^{\prime } \mathrel{\text{:=}} \min \left\{  {d,\Theta \left( {{\varepsilon }^{-2}\log \left| X\right| {\log }^{2}\log \left| X\right| {\log }^{3}d}\right) }\right\}$为中间维度，${T}_{2} \mathrel{\text{:=}} \Theta \left( {{d}^{\prime }\log \left| X\right| }\right)$为第二次Kac随机游走的长度，$m \mathrel{\text{:=}} \Theta \left( {{\varepsilon }^{-2}\log \left| X\right| }\right)$为目标维度，然后将联合线性变换（JLT）定义为$f\left( x\right)  = \left( {{f}_{2} \circ  {f}_{1}}\right) \left( x\right)  \mathrel{\text{:=}} {S}^{\left( m,{d}^{\prime }\right) }{K}^{\left( {T}_{2}\right) }{S}^{\left( {d}^{\prime },d\right) }{K}^{\left( {T}_{1}\right) }x$，其中${K}^{\left( {T}_{1}\right) } \in  {\mathbb{R}}^{d \times  d}$和${K}^{\left( {T}_{2}\right) } \in  {\mathbb{R}}^{{d}^{\prime } \times  {d}^{\prime }}$分别是长度为${T}_{1}$和${T}_{2}$的独立Kac随机游走，${S}^{\left( {d}^{\prime },d\right) } \in  \{ 0,1{\} }^{{d}^{\prime } \times  d}$和${S}^{\left( m,{d}^{\prime }\right) } \in  \{ 0,1{\} }^{m \times  {d}^{\prime }}$分别投影到前${d}^{\prime }$和$m$个坐标${}^{30}$上。由于${K}^{\left( T\right) }$可以在时间$O\left( T\right)$内应用，KacJL构造是一种嵌入时间为$O\left( {d\log d + \min \left\{  {d\log \left| X\right| ,{\varepsilon }^{-2}{\log }^{2}\left| X\right| {\log }^{2}\log \left| X\right| {\log }^{3}d}\right\}  }\right)$且目标维度渐近最优的JLD，并且仅应用第一部分$\left( {f}_{1}\right)$时，KacJL的嵌入时间为$O\left( {d\log d}\right)$，但目标维度为$O\left( {{\varepsilon }^{-2}\log \left| X\right| {\log }^{2}\log \left| X\right| {\log }^{3}d}\right)$，并非最优。

Jain et al. [Jai+20] also proposes a version of their JLD construction that avoids computing trigonometric functions ${}^{31}$ by choosing the angles ${\theta }_{t}$ uniformly at random from the set $\{ \pi /4,{3\pi }/4,{5\pi }/4,{7\pi }/4\}$ or even the singleton set $\{ \pi /4\}$ . This comes at the ${\cos }^{32}$ of increasing ${T}_{1}$ by a factor of $\log \log d$ and ${T}_{2}$ by a factor of $\log d$ ,and for the singleton case multiplying with random signs (as we have done with the $D$ matrices in many of the previous constructions) and projecting down onto a random subset of coordinates rather than the ${d}^{\prime }$ or $m$ first.

贾恩等人（Jain et al.）[Jai+20]还提出了他们的JLD构造的一个版本，该版本通过从集合$\{ \pi /4,{3\pi }/4,{5\pi }/4,{7\pi }/4\}$中甚至是单元素集合$\{ \pi /4\}$中均匀随机地选择角度${\theta }_{t}$，避免了计算三角函数${}^{31}$。这样做的${\cos }^{32}$是将${T}_{1}$增加$\log \log d$倍，将${T}_{2}$增加$\log d$倍，并且在单元素集合的情况下，与随机符号相乘（就像我们在之前的许多构造中对$D$矩阵所做的那样），并投影到坐标的随机子集上，而不是投影到前${d}^{\prime }$个或前$m$个坐标上。

$$
\text{-} \rightarrow  
$$

This concludes the overview of Johnson-Lindenstrauss distributions and transforms, though there are many aspects we did not cover such as space usage, preprocessing time, randomness usage,and norms other than ${\ell }_{2}$ . However,a summary of the main aspects we did cover (embedding times and target dimensions of the JLDs) can be found in table 1.

至此，我们完成了对约翰逊 - 林登斯特劳斯分布（Johnson - Lindenstrauss distributions）和变换的概述，不过仍有许多方面我们未作讨论，比如空间使用、预处理时间、随机性使用以及除 ${\ell }_{2}$ 之外的范数。然而，我们所涵盖的主要方面（JLD 的嵌入时间和目标维度）的总结可在表 1 中找到。

---

<!-- Footnote -->

${}^{30}$ The paper lets ${d}^{\prime }$ and $m$ be random variables,but with the way the JLD is presented here a deterministic projection suffices,though it may affect constants hiding in the big- $O$ expressions.

${}^{30}$ 论文将 ${d}^{\prime }$ 和 $m$ 设为随机变量，但就此处所呈现的 JLD 而言，确定性投影就足够了，尽管这可能会影响隐藏在大 $O$ 表达式中的常数。

${}^{31}$ Recall that $\sin \left( {\pi /4}\right)  = {2}^{-1/2}$ and that similar results holds for cosine and for the other angles.

${}^{31}$ 回顾 $\sin \left( {\pi /4}\right)  = {2}^{-1/2}$，并且类似的结果对于余弦以及其他角度也成立。

${}^{32}$ Note that the various Kac walk lengths are only shown to be sufficient,and so tighter analysis might shorten them and perhaps remove the cost of using simpler angles.

${}^{32}$ 请注意，各种卡茨游走（Kac walk）长度仅被证明是足够的，因此更严谨的分析可能会缩短它们，并且或许能消除使用更简单角度的成本。

<!-- Footnote -->

---

## 2 Deferred Proofs

## 2 延迟证明

### 2.1 k-Means Cost is Pairwise Distances

### 2.1 k - 均值成本即成对距离

Let us first repeat the lemma to remind ourselves what we need to show.

让我们先重复一下引理，以提醒自己需要证明什么。

Lemma 1.6. Let $k,d \in  {\mathbb{N}}_{1}$ and ${X}_{i} \subset  {\mathbb{R}}^{d}$ for $i \in  \{ 1,\ldots ,k\}$ ,then

引理1.6。设$k,d \in  {\mathbb{N}}_{1}$且对于$i \in  \{ 1,\ldots ,k\}$有${X}_{i} \subset  {\mathbb{R}}^{d}$，则

$$
\mathop{\sum }\limits_{{i = 1}}^{k}\mathop{\sum }\limits_{{x \in  {X}_{i}}}{\begin{Vmatrix}x - \frac{1}{\left| {X}_{i}\right| }\mathop{\sum }\limits_{{y \in  {X}_{i}}}y\end{Vmatrix}}_{2}^{2} = \frac{1}{2}\mathop{\sum }\limits_{{i = 1}}^{k}\frac{1}{\left| {X}_{i}\right| }\mathop{\sum }\limits_{{x,y \in  {X}_{i}}}\parallel x - y{\parallel }_{2}^{2}.
$$

In order to prove lemma 1.6 we will need the following lemma.

为了证明引理1.6，我们需要以下引理。

Lemma 2.1. Let $d \in  {\mathbb{N}}_{1}$ and $X \subset  {\mathbb{R}}^{d}$ and define $\mu  \mathrel{\text{:=}} \frac{1}{\left| X\right| }\mathop{\sum }\limits_{{x \in  X}}x$ as the mean of $X$ ,then it holds that

引理2.1。设$d \in  {\mathbb{N}}_{1}$和$X \subset  {\mathbb{R}}^{d}$，并将$\mu  \mathrel{\text{:=}} \frac{1}{\left| X\right| }\mathop{\sum }\limits_{{x \in  X}}x$定义为$X$的均值，则有如下结论成立

$$
\mathop{\sum }\limits_{{x,y \in  X}}\langle x - \mu ,y - \mu \rangle  = 0.
$$

Proof of lemma 2.1. The lemma follows from the definition of $\mu$ and the linearity of the real inner product.

引理2.1的证明。该引理可由$\mu$的定义以及实内积的线性性质推出。

$$
\mathop{\sum }\limits_{{x,y \in  X}}\langle x - \mu ,y - \mu \rangle  = \mathop{\sum }\limits_{{x,y \in  X}}\left( {\langle x,y\rangle -\langle x,\mu \rangle -\langle y,\mu \rangle +\langle \mu ,\mu \rangle }\right) 
$$

$$
 = \mathop{\sum }\limits_{{x,y \in  X}}\langle x,y\rangle  - \mathop{\sum }\limits_{{x \in  X}}2\left| X\right| \langle x,\mu \rangle  + {\left| X\right| }^{2}\langle \mu ,\mu \rangle 
$$

$$
 = \mathop{\sum }\limits_{{x,y \in  X}}\langle x,y\rangle  - 2\mathop{\sum }\limits_{{x \in  X}}\left\langle  {x,\mathop{\sum }\limits_{{y \in  X}}y}\right\rangle   + \left\langle  {\mathop{\sum }\limits_{{x \in  X}}x,\mathop{\sum }\limits_{{y \in  X}}y}\right\rangle  
$$

$$
 = \mathop{\sum }\limits_{{x,y \in  X}}\langle x,y\rangle  - 2\mathop{\sum }\limits_{{x,y \in  X}}\langle x,y\rangle  + \mathop{\sum }\limits_{{x,y \in  X}}\langle x,y\rangle 
$$

$$
 = 0\text{.}
$$

Proof of lemma 1.6. We will first prove an identity for each partition,so let ${X}_{i} \subseteq  X \subset  {\mathbb{R}}^{d}$ be any partition of the dataset $X$ and define ${\mu }_{i} \mathrel{\text{:=}} \frac{1}{\left| {X}_{i}\right| }\mathop{\sum }\limits_{{x \in  {X}_{i}}}x$ as the mean of ${X}_{i}$ .

引理1.6的证明。我们首先将证明每个划分的一个恒等式，因此，设${X}_{i} \subseteq  X \subset  {\mathbb{R}}^{d}$为数据集$X$的任意划分，并将${\mu }_{i} \mathrel{\text{:=}} \frac{1}{\left| {X}_{i}\right| }\mathop{\sum }\limits_{{x \in  {X}_{i}}}x$定义为${X}_{i}$的均值。

$$
\frac{1}{2\left| {X}_{i}\right| }\mathop{\sum }\limits_{{x,y \in  {X}_{i}}}\parallel x - y{\parallel }_{2}^{2} = \frac{1}{2\left| {X}_{i}\right| }\mathop{\sum }\limits_{{x,y \in  {X}_{i}}}{\begin{Vmatrix}\left( x - {\mu }_{i}\right)  - \left( y - {\mu }_{i}\right) \end{Vmatrix}}_{2}^{2}
$$

$$
 = \frac{1}{2\left| {X}_{i}\right| }\mathop{\sum }\limits_{{x,y \in  {X}_{i}}}\left( {{\begin{Vmatrix}x - {\mu }_{i}\end{Vmatrix}}_{2}^{2} + {\begin{Vmatrix}y - {\mu }_{i}\end{Vmatrix}}_{2}^{2} - 2\left\langle  {x - {\mu }_{i},y - {\mu }_{i}}\right\rangle  }\right) 
$$

$$
 = \mathop{\sum }\limits_{{x \in  {X}_{i}}}{\begin{Vmatrix}x - {\mu }_{i}\end{Vmatrix}}_{2}^{2} - \frac{1}{2\left| {X}_{i}\right| }\mathop{\sum }\limits_{{x,y \in  {X}_{i}}}2\left\langle  {x - {\mu }_{i},y - {\mu }_{i}}\right\rangle  
$$

$$
 = \mathop{\sum }\limits_{{x \in  {X}_{i}}}{\begin{Vmatrix}x - {\mu }_{i}\end{Vmatrix}}_{2}^{2}
$$

where the last equality holds by lemma 2.1. We now substitute each term in the sum in lemma 1.6 using the just derived identity:

最后一个等式由引理2.1成立。我们现在使用刚刚推导出的恒等式来替换引理1.6中求和式里的每一项：

$$
\mathop{\sum }\limits_{{i = 1}}^{k}\mathop{\sum }\limits_{{x \in  {X}_{i}}}{\begin{Vmatrix}x - \frac{1}{\left| {X}_{i}\right| }\mathop{\sum }\limits_{{y \in  {X}_{i}}}y\end{Vmatrix}}_{2}^{2} = \frac{1}{2}\mathop{\sum }\limits_{{i = 1}}^{k}\frac{1}{\left| {X}_{i}\right| }\mathop{\sum }\limits_{{x,y \in  {X}_{i}}}\parallel x - y{\parallel }_{2}^{2}.
$$

### 2.2 Super Sparse DKS

### 2.2 超稀疏DKS（Super Sparse DKS）

The tight bounds on the performance of feature hashing presented in theorem 1.8 can be extended to tight performance bounds for the DKS construction. Recall that the DKS construction, parameterised by a so-called column sparsity $s \in  {\mathbb{N}}_{1}$ ,works by first mapping a vector $x \in  {\mathbb{R}}^{d}$ to an ${x}^{\prime } \in  {\mathbb{R}}^{sd}$ by duplicating each entry in ${xs}$ times and then scaling with $1/\sqrt{s}$ ,before applying feature hashing to ${x}^{\prime }$ ,as ${x}^{\prime }$ has a more palatable ${\ell }_{\infty }/{\ell }_{2}$ ratio compared to $x$ . The setting for the extended result is that if we wish to use the DKS construction but we only need to handle vectors with a small $\parallel x{\parallel }_{\infty }/\parallel x{\parallel }_{2}$ ratio,we can choose a column sparsity smaller than the usual $\Theta \left( {{\varepsilon }^{-1}\log \frac{1}{\delta }\log \frac{m}{\delta }}\right)$ and still get the Johnson-Lindenstrauss guarantees. This is formalised in corollary 1.9. The two pillars of theorem 1.8 we use in the proof of corollary 1.9 is that the feature hashing tradeoff is tight and that we can force the DKS construction to create hard instances for feature hashing.

定理1.8中给出的特征哈希性能的紧界可以扩展到DKS构造（DKS construction）的紧性能界。回顾一下，由所谓的列稀疏度 $s \in  {\mathbb{N}}_{1}$ 参数化的DKS构造的工作方式是，首先将向量 $x \in  {\mathbb{R}}^{d}$ 映射到 ${x}^{\prime } \in  {\mathbb{R}}^{sd}$，方法是将 ${xs}$ 中的每个元素复制若干次，然后用 $1/\sqrt{s}$ 进行缩放，再对 ${x}^{\prime }$ 应用特征哈希，因为与 $x$ 相比，${x}^{\prime }$ 具有更合适的 ${\ell }_{\infty }/{\ell }_{2}$ 比率。扩展结果的设定是，如果我们希望使用DKS构造，但只需要处理 $\parallel x{\parallel }_{\infty }/\parallel x{\parallel }_{2}$ 比率较小的向量，我们可以选择比通常的 $\Theta \left( {{\varepsilon }^{-1}\log \frac{1}{\delta }\log \frac{m}{\delta }}\right)$ 更小的列稀疏度，仍然能获得约翰逊 - 林登斯特劳斯（Johnson - Lindenstrauss）保证。这在推论1.9中得到了形式化表述。我们在推论1.9的证明中使用的定理1.8的两个支柱是，特征哈希的权衡是紧的，并且我们可以迫使DKS构造为特征哈希创建困难实例。

Corollary 1.9. Let ${v}_{\mathrm{{DKS}}} \in  \left\lbrack  {1/\sqrt{d},1}\right\rbrack$ denote the largest ${\ell }_{\infty }/{\ell }_{2}$ ratio required, ${v}_{\mathrm{{FH}}}$ denote the ${\ell }_{\infty }/{\ell }_{2}$ constraint for feature hashing as defined in theorem 1.8,and ${s}_{\mathrm{{DKS}}} \in  \left\lbrack  m\right\rbrack$ as the minimum column sparsity such that the DKS construction with that sparsity is a JLD for the subset of vectors $x \in  {\mathbb{R}}^{d}$ that satisfy $\parallel x{\parallel }_{\infty }/\parallel x{\parallel }_{2} \leq  {v}_{\mathrm{{DKS}}}$ . Then

推论1.9。设${v}_{\mathrm{{DKS}}} \in  \left\lbrack  {1/\sqrt{d},1}\right\rbrack$表示所需的最大${\ell }_{\infty }/{\ell }_{2}$比率，${v}_{\mathrm{{FH}}}$表示定理1.8中所定义的特征哈希的${\ell }_{\infty }/{\ell }_{2}$约束，${s}_{\mathrm{{DKS}}} \in  \left\lbrack  m\right\rbrack$表示最小列稀疏度，使得具有该稀疏度的DKS构造是满足$\parallel x{\parallel }_{\infty }/\parallel x{\parallel }_{2} \leq  {v}_{\mathrm{{DKS}}}$的向量子集$x \in  {\mathbb{R}}^{d}$的JLD。那么

$$
{s}_{\mathrm{{DKS}}} = \Theta \left( \frac{{v}_{\mathrm{{DKS}}}^{2}}{{v}_{\mathrm{{FH}}}^{2}}\right) . \tag{12}
$$

The upper bound part of the $\Theta$ in corollary 1.9 shows how sparse we can choose the DKS construction to be and still get Johnson-Lindenstrauss guarantees for the data we care about, while the lower bound shows that if we choose a sparsity below this bound, there exists vectors who get distorted too much too often despite having an ${\ell }_{\infty }/{\ell }_{2}$ ratio of at most ${v}_{\mathrm{{DKS}}}$ .

推论1.9中$\Theta$的上界部分表明，我们可以将DKS构造（DKS construction）选择得多么稀疏，同时仍能为我们关注的数据提供约翰逊 - 林登斯特劳斯（Johnson - Lindenstrauss）保证；而下界则表明，如果我们选择的稀疏度低于此界限，那么存在一些向量，尽管其${\ell }_{\infty }/{\ell }_{2}$比率至多为${v}_{\mathrm{{DKS}}}$，但它们仍会经常出现过度失真的情况。

Proof of corollary 1.9 Let us first prove the upper bound: ${s}_{\mathrm{{DKS}}} = O\left( \frac{{v}_{\mathrm{{DKS}}}^{2}}{{v}_{\mathrm{{FH}}}^{2}}\right)$ .

推论1.9的证明 让我们首先证明上界：${s}_{\mathrm{{DKS}}} = O\left( \frac{{v}_{\mathrm{{DKS}}}^{2}}{{v}_{\mathrm{{FH}}}^{2}}\right)$。

Let $s \mathrel{\text{:=}} \Theta \left( \frac{{v}_{\mathrm{{DES}}}^{2}}{{v}_{\mathrm{{FH}}}^{2}}\right)  \in  \left\lbrack  m\right\rbrack$ be the column sparsity,and let $x \in  {\mathbb{R}}^{d}$ be a unit vector with

设 $s \mathrel{\text{:=}} \Theta \left( \frac{{v}_{\mathrm{{DES}}}^{2}}{{v}_{\mathrm{{FH}}}^{2}}\right)  \in  \left\lbrack  m\right\rbrack$ 为列稀疏度，并设 $x \in  {\mathbb{R}}^{d}$ 为一个单位向量，且

$\parallel x{\parallel }_{\infty } \leq  {v}_{\mathrm{{DKS}}}$ . The goal is now to show that a DKS construction with sparsity $s$ can embed $x$ while preserving its norm within $1 \pm  \varepsilon$ with probability at least $1 - \delta$ (as defined in lemma 1.2). Let ${x}^{\prime } \in  {\mathbb{R}}^{sd}$ be the unit vector constructed by duplicating each entry in ${xs}$ times and scaling with $1/\sqrt{s}$ as in the DKS construction. We now have

$\parallel x{\parallel }_{\infty } \leq  {v}_{\mathrm{{DKS}}}$ 。现在的目标是证明，稀疏度为 $s$ 的 DKS 构造（DKS construction）能够嵌入 $x$ ，同时以至少 $1 - \delta$ 的概率（如引理 1.2 所定义）将其范数保持在 $1 \pm  \varepsilon$ 范围内。设 ${x}^{\prime } \in  {\mathbb{R}}^{sd}$ 为按照 DKS 构造方式，将 ${xs}$ 中的每个元素复制若干次并按 $1/\sqrt{s}$ 进行缩放后得到的单位向量。现在我们有

$$
{\begin{Vmatrix}{x}^{\prime }\end{Vmatrix}}_{\infty } \leq  \frac{{v}_{\mathrm{{DKS}}}}{\sqrt{s}} = \Theta \left( {v}_{\mathrm{{FH}}}\right) . \tag{13}
$$

Let DKS denote the JLD from the DKS construction with column sparsity $s$ ,and let FH denote the feature hashing JLD. Then we can conclude

让DKS表示通过DKS构造得到的列稀疏度为$s$的联合低密度奇偶校验码（JLD），并让FH表示特征哈希JLD。然后我们可以得出结论

$$
\mathop{\Pr }\limits_{{f \sim  \mathrm{{DKS}}}}\left\lbrack  {\left| {\parallel f\left( x\right) {\parallel }_{2}^{2} - 1}\right|  \leq  \varepsilon }\right\rbrack   = \mathop{\Pr }\limits_{{g \sim  \mathrm{{FH}}}}\left\lbrack  {\left| {{\begin{Vmatrix}g\left( {x}^{\prime }\right) \end{Vmatrix}}_{2}^{2} - 1}\right|  \leq  \varepsilon }\right\rbrack   \geq  1 - \delta ,
$$

where the inequality is implied by eq. (13) and theorem 1.8

其中该不等式由等式(13)和定理1.8推出

Now let us prove the lower bound: ${s}_{\mathrm{{DKS}}} = \Omega \left( \frac{{v}_{\mathrm{{DKS}}}^{2}}{{v}_{\mathrm{{FH}}}^{2}}\right)$ .

现在让我们证明下界：${s}_{\mathrm{{DKS}}} = \Omega \left( \frac{{v}_{\mathrm{{DKS}}}^{2}}{{v}_{\mathrm{{FH}}}^{2}}\right)$ 。

Let $s \mathrel{\text{:=}} o\left( \frac{{v}_{\mathrm{{DKS}}}^{2}}{{v}_{\mathrm{{FH}}}^{2}}\right)$ ,and let $x = {\left( {v}_{\mathrm{{DKS}}},\ldots ,{v}_{\mathrm{{DKS}}},0,\ldots ,0\right) }^{\top } \in  {\mathbb{R}}^{d}$ be a unit vector. We now wish to show that a DKS construction with sparsity $s$ will preserve the norm of $x$ to within $1 \pm  \varepsilon$ with probability strictly less than $1 - \delta$ . As before,define ${x}^{\prime } \in  {\mathbb{R}}^{sd}$ as the unit vector the DKS construction computes when duplicating every entry in $xs$ times and scaling with $1/\sqrt{s}$ . This

设$s \mathrel{\text{:=}} o\left( \frac{{v}_{\mathrm{{DKS}}}^{2}}{{v}_{\mathrm{{FH}}}^{2}}\right)$，并设$x = {\left( {v}_{\mathrm{{DKS}}},\ldots ,{v}_{\mathrm{{DKS}}},0,\ldots ,0\right) }^{\top } \in  {\mathbb{R}}^{d}$为单位向量。我们现在要证明，稀疏度为$s$的DKS构造（DKS construction）将以严格小于$1 - \delta$的概率把$x$的范数保持在$1 \pm  \varepsilon$的范围内。和之前一样，将${x}^{\prime } \in  {\mathbb{R}}^{sd}$定义为DKS构造在把$xs$中的每个元素复制$1/\sqrt{s}$倍并进行缩放时所计算出的单位向量。这

gives

给出

$$
{\begin{Vmatrix}{x}^{\prime }\end{Vmatrix}}_{\infty } = \frac{{v}_{\mathrm{{DKS}}}}{\sqrt{s}} = \omega \left( {v}_{\mathrm{{FH}}}\right) . \tag{14}
$$

Finally,let DKS denote the JLD from the DKS construction with column sparsity $s$ ,and let FH denote the feature hashing JLD. Then we can conclude

最后，令DKS表示通过DKS构造得到的具有列稀疏性$s$的联合局部敏感哈希函数族（JLD），令FH表示特征哈希JLD。然后我们可以得出结论

$$
\mathop{\Pr }\limits_{{f \sim  \mathrm{{DKS}}}}\left\lbrack  {\left| {\parallel f\left( x\right) {\parallel }_{2}^{2} - 1}\right|  \leq  \varepsilon }\right\rbrack   = \mathop{\Pr }\limits_{{g \sim  \mathrm{{FH}}}}\left\lbrack  {\left| {{\begin{Vmatrix}g\left( {x}^{\prime }\right) \end{Vmatrix}}_{2}^{2} - 1}\right|  \leq  \varepsilon }\right\rbrack   < 1 - \delta ,
$$

where the inequality is implied by eq. (14) and theorem 1.8,and the fact that ${x}^{\prime }$ has the shape of an asymptotically worst case instance for feature hashing.

其中该不等式由等式(14)和定理1.8以及${x}^{\prime }$具有特征哈希的渐近最坏情况实例的形式这一事实所蕴含。

### 2.3 LWTJL Fails for Too Sparse Vectors

### 2.3 轻量级张量联合局部敏感哈希（LWTJL）对过于稀疏的向量失效

Proposition 1.11. For any seed matrix define LWT as the LWTJL distribution seeded with that matrix. Then for all $\delta  \in  \left( {0,1}\right)$ ,there exists a vector $x \in  {\mathbb{C}}^{d}$ (or $x \in  {\mathbb{R}}^{d}$ ,if the seed matrix is a real matrix) satisfying $\parallel x{\parallel }_{\infty }/\parallel x{\parallel }_{2} = \Theta \left( {{\log }^{-1/2}\frac{1}{\delta }}\right)$ such that

命题1.11。对于任何种子矩阵，将LWT定义为以该矩阵为种子的LWTJL分布。那么对于所有的$\delta  \in  \left( {0,1}\right)$，存在一个向量$x \in  {\mathbb{C}}^{d}$（或者如果种子矩阵是实矩阵，则为$x \in  {\mathbb{R}}^{d}$）满足$\parallel x{\parallel }_{\infty }/\parallel x{\parallel }_{2} = \Theta \left( {{\log }^{-1/2}\frac{1}{\delta }}\right)$，使得

$$
\mathop{\Pr }\limits_{{f \sim  \mathrm{{LWT}}}}\left\lbrack  {f\left( x\right)  = \mathbf{0}}\right\rbrack   > \delta 
$$

Proof. The main idea is to construct the vector $x$ out of segments that are orthogonal to the seed matrix with some probability,and then show that $x$ is orthogonal to all copies of the seed matrix simultaneously with probability larger than $\delta$ .

证明。主要思路是从以一定概率与种子矩阵正交的线段中构建向量 $x$，然后证明向量 $x$ 同时与种子矩阵的所有副本正交的概率大于 $\delta$。

Let $r,c \in  {\mathbb{N}}_{1}$ be constants and ${A}_{1} \in  {\mathbb{C}}^{r \times  c}$ be a seed matrix. Let $d$ be the source dimension of the LWTJL construction, $D \in  \{  - 1,0,1{\} }^{d \times  d}$ be the random diagonal matrix with i.i.d. Rademachers, $l \in  {\mathbb{N}}_{1}$ such that ${c}^{l} = d$ ,and ${A}_{l} \in  {\mathbb{C}}^{{r}^{l} \times  {c}^{l}}$ be the the LWT,i.e. ${A}_{l} \mathrel{\text{:=}} {A}_{1}^{\otimes l}$ . Since $r < c$ there exists a nontrivial vector $z \in  {\mathbb{C}}^{c} \smallsetminus  \{ \mathbf{0}\}$ that is orthogonal to all $r$ rows of ${A}_{1}$ and $\parallel z{\parallel }_{\infty } = \Theta \left( 1\right)$ . Now define $x \in  {\mathbb{C}}^{d}$ as $k \in  {\mathbb{N}}_{1}$ copies of $z$ followed by a padding of $0\mathrm{\;s}$ ,where $k = \left\lfloor  {\frac{1}{c}\lg \frac{1}{\delta } - 1}\right\rfloor$ . Note that if the seed matrix is real,we can choose $z$ and therefore $x$ to be real as well.

设 $r,c \in  {\mathbb{N}}_{1}$ 为常数，${A}_{1} \in  {\mathbb{C}}^{r \times  c}$ 为种子矩阵。设 $d$ 为 LWTJL 构造的源维度，$D \in  \{  - 1,0,1{\} }^{d \times  d}$ 为具有独立同分布拉德马赫（Rademacher）变量的随机对角矩阵，$l \in  {\mathbb{N}}_{1}$ 使得 ${c}^{l} = d$ ，且 ${A}_{l} \in  {\mathbb{C}}^{{r}^{l} \times  {c}^{l}}$ 为 LWT，即 ${A}_{l} \mathrel{\text{:=}} {A}_{1}^{\otimes l}$ 。由于 $r < c$ ，存在一个非平凡向量 $z \in  {\mathbb{C}}^{c} \smallsetminus  \{ \mathbf{0}\}$ 与 ${A}_{1}$ 的所有 $r$ 行以及 $\parallel z{\parallel }_{\infty } = \Theta \left( 1\right)$ 正交。现在将 $x \in  {\mathbb{C}}^{d}$ 定义为 $z$ 的 $k \in  {\mathbb{N}}_{1}$ 个副本，后面再填充 $0\mathrm{\;s}$ ，其中 $k = \left\lfloor  {\frac{1}{c}\lg \frac{1}{\delta } - 1}\right\rfloor$ 。注意，如果种子矩阵是实矩阵，我们也可以选择 $z$ ，从而 $x$ 也为实矩阵。

The first thing to note is that

首先需要注意的是

$$
\parallel x{\parallel }_{0} \leq  {ck} < \lg \frac{1}{\delta },
$$

which implies that

这意味着

$$
\mathop{\Pr }\limits_{D}\left\lbrack  {{Dx} = x}\right\rbrack   = {2}^{-\parallel x{\parallel }_{0}} > \delta .
$$

Secondly,due to the Kronecker structure of ${A}_{l}$ and the fact that $z$ is orthogonal to the rows of ${A}_{1}$ ,we have

其次，由于${A}_{l}$的克罗内克结构（Kronecker structure）以及$z$与${A}_{1}$的行正交这一事实，我们有

$$
{Ax} = \mathbf{0}\text{.}
$$

Taken together, we can conclude

综上所述，我们可以得出结论

$$
\mathop{\Pr }\limits_{{f \sim  \mathrm{{LWT}}}}\left\lbrack  {f\left( x\right)  = \mathbf{0}}\right\rbrack   \geq  \mathop{\Pr }\limits_{D}\left\lbrack  {{A}_{l}{Dx} = \mathbf{0}}\right\rbrack   \geq  \mathop{\Pr }\limits_{D}\left\lbrack  {{Dx} = x}\right\rbrack   > \delta .
$$

Now we just need to show that $\parallel x{\parallel }_{\infty }/\parallel x{\parallel }_{2} = \Theta \left( {{\log }^{-1/2}\frac{1}{\delta }}\right)$ . Since $c$ is a constant and $x$ is consists of $k = \Theta \left( {\log \frac{1}{\delta }}\right)$ copies of $z$ followed by zeroes,

现在我们只需证明$\parallel x{\parallel }_{\infty }/\parallel x{\parallel }_{2} = \Theta \left( {{\log }^{-1/2}\frac{1}{\delta }}\right)$。由于$c$是一个常数，且$x$由$k = \Theta \left( {\log \frac{1}{\delta }}\right)$个$z$的副本后面跟着零组成，

$$
\parallel x{\parallel }_{\infty } = \parallel z{\parallel }_{\infty } = \Theta \left( 1\right) ,
$$

$$
\parallel z{\parallel }_{2} = \Theta \left( 1\right) 
$$

$$
\parallel x{\parallel }_{2} = \sqrt{k}\parallel z{\parallel }_{2} = \Theta \left( \sqrt{\log \frac{1}{\delta }}\right) ,
$$

which implies the claimed ratio,

这意味着所声称的比率

$$
\frac{\parallel x{\parallel }_{\infty }}{\parallel x{\parallel }_{2}} = \Theta \left( {{\log }^{-1/2}\frac{1}{\delta }}\right) .
$$

The following corollary is just a restatement of proposition 1.11 in terms of lemma 1.2 and the proof therefore follows immediately from proposition 1.11

以下推论只是根据引理1.2对命题1.11的重述，因此其证明可直接从命题1.11得出

Corollary 2.2. For every $m,d, \in  {\mathbb{N}}_{1}$ ,and $\delta ,\varepsilon  \in  \left( {0,1}\right)$ ,and LWTJL distribution LWT over $f : {\mathbb{K}}^{d} \rightarrow  {\mathbb{K}}^{m}$ , where $\mathbb{K} \in  \{ \mathbb{R},\mathbb{C}\}$ and $m < d$ there exists a vector $x \in  {\mathbb{K}}^{d}$ with $\parallel x{\parallel }_{\infty }/\parallel x{\parallel }_{2} = \Theta \left( {{\log }^{-1/2}\frac{1}{\delta }}\right)$ such that

推论2.2。对于每一个$m,d, \in  {\mathbb{N}}_{1}$、$\delta ,\varepsilon  \in  \left( {0,1}\right)$，以及在$f : {\mathbb{K}}^{d} \rightarrow  {\mathbb{K}}^{m}$上的LWTJL分布LWT（其中$\mathbb{K} \in  \{ \mathbb{R},\mathbb{C}\}$且$m < d$），存在一个向量$x \in  {\mathbb{K}}^{d}$，满足$\parallel x{\parallel }_{\infty }/\parallel x{\parallel }_{2} = \Theta \left( {{\log }^{-1/2}\frac{1}{\delta }}\right)$，使得

$$
\mathop{\Pr }\limits_{{f \sim  \operatorname{LWT}}}\left\lbrack  {\left| {\parallel f\left( x\right) {\parallel }_{2}^{2} - \parallel x{\parallel }_{2}^{2}}\right|  \leq  \varepsilon \parallel x{\parallel }_{2}^{2}}\right\rbrack   < 1 - \delta .
$$

## References

## 参考文献

[AC06] Nir Ailon and Bernard Chazelle. "Approximate nearest neighbors and the fast Johnson-Lindenstrauss transform". In: Proceedings of the 38th Symposium on Theory of Computing. STOC '06. ACM, 2006, pp. 557-563. DOI: 10.1145/1132516. 1132597 (cit. on p. 25).

[AC06] 尼尔·艾隆（Nir Ailon）和伯纳德·查泽尔（Bernard Chazelle）。《近似最近邻与快速约翰逊 - 林登斯特劳斯变换》。载于：第38届计算理论研讨会会议录。STOC '06。美国计算机协会（ACM），2006年，第557 - 563页。DOI: 10.1145/1132516. 1132597（见第25页引用）。

[AC09] Nir Ailon and Bernard Chazelle. "The fast Johnson-Lindenstrauss transform and approximate nearest neighbors". In: Journal on Computing 39.1 (2009), pp. 302-322. DOI: 10.1137/060673096. Previously published as [AC06] (cit. on pp. 5, 12.14, 15. 19, 21).

[AC09] 尼尔·艾隆（Nir Ailon）和伯纳德·查泽尔（Bernard Chazelle）。《快速约翰逊 - 林登斯特劳斯变换与近似最近邻》。载于：《计算期刊》（Journal on Computing）第39卷第1期（2009年），第302 - 322页。DOI: 10.1137/060673096。之前以[AC06]发表（见第5、12.14、15、19、21页引用）。

[Ach01] Dimitris Achlioptas. "Database-friendly Random Projections". In: Proceedings of the 20th Symposium on Principles of Database Systems. PODS '01. ACM, 2001, pp. 274-281. DOI: 10.1145/375551.375608 (cit. on pp. 12, 26).

[Ach01] 迪米特里斯·阿赫利奥普塔斯（Dimitris Achlioptas）。《适合数据库的随机投影》。载于：第20届数据库系统原理研讨会会议录。PODS '01。美国计算机协会（ACM），2001年，第274 - 281页。DOI: 10.1145/375551.375608（见第12、26页引用）。

[Ach03] Dimitris Achlioptas. "Database-friendly random projections: Johnson-Lindenstrauss with binary coins". In: Journal of Computer and System Sciences 66.4 (2003), pp. 671- 687. DOI: 10.1016/S0022-0000(03) 00025-4. Previously published as [Ach01] (cit. on pp. 11-14, 21).

[Ach03] 迪米特里斯·阿赫利奥普塔斯（Dimitris Achlioptas）。“对数据库友好的随机投影：使用二元硬币的约翰逊 - 林登斯特劳斯引理”。载于：《计算机与系统科学杂志》（Journal of Computer and System Sciences）66.4（2003 年），第 671 - 687 页。DOI：10.1016/S0022 - 0000(03) 00025 - 4。此前以 [Ach01] 发表（见第 11 - 14、21 页引用）。

[AH14]

Farhad Pourkamali Anaraki and Shannon Hughes. "Memory and Computation Efficient PCA via Very Sparse Random Projections". In: Proceedings of the 31st International Conference on Machine Learning (ICML '14). Vol. 32. Proceedings of Machine Learning Research (PMLR) 2. PMLR, 2014, pp. 1341-1349 (cit. on p. 3).

法尔哈德·普尔卡马利·阿纳拉基（Farhad Pourkamali Anaraki）和香农·休斯（Shannon Hughes）。“通过极稀疏随机投影实现内存和计算高效的主成分分析”。载于：第 31 届国际机器学习会议（ICML '14）论文集。第 32 卷。机器学习研究会议录（Proceedings of Machine Learning Research，PMLR）2。机器学习研究会议录出版（PMLR），2014 年，第 1341 - 1349 页（见第 3 页引用）。

[AI17]

Alexandr Andoni and Piotr Indyk. "Nearest Neighbors in High-Dimensional Spaces". In: Handbook of Discrete and Computational Geometry. 3rd ed. CRC Press, 2017. Chap. 43, pp. 1135-1155. ISBN: 978-1-4987-1139-5 (cit. on p. 2).

亚历山大·安多尼（Alexandr Andoni）和彼得·因迪克（Piotr Indyk）。《高维空间中的最近邻》。载于《离散与计算几何手册》（Handbook of Discrete and Computational Geometry）。第3版。CRC出版社，2017年。第43章，第1135 - 1155页。国际标准书号：978 - 1 - 4987 - 1139 - 5（见第2页引用）。

[AK17]

Noga Alon and Bo'Az B. Klartag. "Optimal Compression of Approximate Inner Products and Dimension Reduction". In: Proceedings of the 58th Symposium on Foundations of Computer Science. FOCS '17. IEEE, 2017, pp. 639-650. DOI: 10.1109/ FOCS. 2017.65 (cit. on p. 4).

诺加·阿隆（Noga Alon）和博阿兹·B·克拉塔格（Bo'Az B. Klartag）。《近似内积的最优压缩与降维》。载于《第58届计算机科学基础研讨会会议录》（Proceedings of the 58th Symposium on Foundations of Computer Science）。FOCS '17。电气与电子工程师协会（IEEE），2017年，第639 - 650页。数字对象标识符：10.1109/ FOCS. 2017.65（见第4页引用）。

[AL08] Nir Ailon and Edo Liberty. "Fast dimension reduction using Rademacher series on dual BCH codes". In: Proceedings of the 19th Symposium on Discrete Algorithms. SODA '08. SIAM, 2008, pp. 1-9 (cit. on p. 26).

[AL08] 尼尔·艾隆（Nir Ailon）和埃多·利伯蒂（Edo Liberty）。《在对偶BCH码上使用拉德马赫级数进行快速降维》。载于《第19届离散算法研讨会会议录》（Proceedings of the 19th Symposium on Discrete Algorithms）。SODA '08。工业与应用数学学会（SIAM），2008年，第1 - 9页（见第26页引用）。

[AL09] Nir Ailon and Edo Liberty. "Fast Dimension Reduction Using Rademacher Series on Dual BCH Codes". In: Discrete & Computational Geometry 42.4 (2009), pp. 615-630. DOI: 10.1007/s00454-008-9110-x. Previously published as [AL08] (cit. on pp. 16, 18, 19, 21).

[AL09] 尼尔·艾隆（Nir Ailon）和埃多·利伯蒂（Edo Liberty）。《利用对偶BCH码上的拉德马赫级数进行快速降维》。载于《离散与计算几何》（Discrete & Computational Geometry）42.4（2009年），第615 - 630页。DOI: 10.1007/s00454-008-9110-x。此前以[AL08]发表（见第16、18、19、21页引用）。

[AL11]

Nir Ailon and Edo Liberty. "An Almost Optimal Unrestricted Fast Johnson-Lindenstrauss Transform". In: Proceedings of the 22nd Symposium on Discrete Algorithms. SODA '11. SIAM, 2011, pp. 185-191. DOI: 10.1137/1. 9781611973082. 17 (cit. on p. 26).

尼尔·艾隆（Nir Ailon）和埃多·利伯蒂（Edo Liberty）。《一种近乎最优的无约束快速约翰逊 - 林登施特劳斯变换》。载于《第22届离散算法研讨会论文集》。SODA '11。工业与应用数学学会（SIAM），2011年，第185 - 191页。DOI: 10.1137/1. 9781611973082. 17（见第26页引用）。

[AL13]

Nir Ailon and Edo Liberty. "An Almost Optimal Unrestricted Fast Johnson-Lindenstrauss Transform". In: Transactions on Algorithms 9.3 (2013), 21:1-21:12. DOI: 10.1145/2483699.2483701. Previously published as [AL11] (cit. on pp. 17.21).

尼尔·艾隆（Nir Ailon）和埃多·利伯蒂（Edo Liberty）。《一种近乎最优的无约束快速约翰逊 - 林登施特劳斯变换》。载于《算法汇刊》（Transactions on Algorithms）9.3（2013年），21:1 - 21:12。DOI: 10.1145/2483699.2483701。此前以[AL11]发表（见第17、21页引用）。

[ALG13]

Madhu Advani, Subhaneil Lahiri, and Surya Ganguli. "Statistical mechanics of complex neural systems and high dimensional data". In: Journal of Statistical Mechanics: Theory and Experiment 2013.03 (2013), P03014. DOI: 10.1088/1742- 5468/2013/03/p03014 (cit. on p. 7).

马杜·阿瓦尼（Madhu Advani）、苏巴内尔·拉希里（Subhaneil Lahiri）和苏里亚·甘古利（Surya Ganguli）。《复杂神经系统与高维数据的统计力学》。载于《统计力学杂志：理论与实验》2013年第3期（2013年），P03014。DOI: 10.1088/1742 - 5468/2013/03/p03014（见第7页引用）。

[All+14]

Zeyuan Allen-Zhu, Rati Gelashvili, Silvio Micali, and Nir Shavit. "Sparse sign-consistent Johnson-Lindenstrauss matrices: Compression with neuroscience-based constraints". In: Proceedings of the National Academy of Sciences (PNAS) 111.47 (2014), pp. 16872-16876. DOI: 10.1073/pnas. 1419100111 (cit. on p. 7).

泽远·艾伦 - 朱（Zeyuan Allen - Zhu）、拉蒂·格拉什维利（Rati Gelashvili）、西尔维奥·米凯利（Silvio Micali）和尼尔·沙维特（Nir Shavit）。《稀疏符号一致的约翰逊 - 林登施特劳斯矩阵：基于神经科学约束的压缩》。载于《美国国家科学院院刊》（PNAS）第111卷第47期（2014年），第16872 - 16876页。DOI: 10.1073/pnas. 1419100111（见第7页引用）。

<!-- Media -->

[Alo+02] Noga Alon, Phillip B. Gibbons, Yossi Matias, and Mario Szegedy. "Tracking Join and Self-Join Sizes in Limited Storage". In: Journal of Computer and System Sciences 64.3 (2002), pp. 719-747. DOI: 10.1006/jcss. 2001.1813 Previously published as Alo+99] (cit. on pp. 10, 12).

[Alo+02] 诺加·阿隆（Noga Alon）、菲利普·B·吉本斯（Phillip B. Gibbons）、约西·马蒂亚斯（Yossi Matias）和马里奥·塞格迪（Mario Szegedy）。“在有限存储中跟踪连接和自连接大小”。载于：《计算机与系统科学杂志》（Journal of Computer and System Sciences）64.3（2002年），第719 - 747页。DOI: 10.1006/jcss. 2001.1813 此前以[Alo+99]发表（见第10、12页引用）。

[Alo+09] Daniel Aloise, Amit Deshpande, Pierre Hansen, and Preyas Popat. "NP-hardness of Euclidean sum-of-squares clustering". In: Machine Learning 75 (2009), pp. 245-248. DOI: 10.1007/s10994-009-5103-0 (cit. on p. 7).

[Alo+09] 丹尼尔·阿洛伊斯（Daniel Aloise）、阿米特·德什潘德（Amit Deshpande）、皮埃尔·汉森（Pierre Hansen）和普雷亚斯·波帕特（Preyas Popat）。“欧几里得平方和聚类的NP难问题”。载于：《机器学习》（Machine Learning）75（2009年），第245 - 248页。DOI: 10.1007/s10994-009-5103-0（见第7页引用）。

[Alo+99] 9] Noga Alon, Phillip B. Gibbons, Yossi Matias, and Mario Szegedy. "Tracking Join and Self-Join Sizes in Limited Storage". In: Proceedings of the 18th Symposium on Principles of Database Systems. PODS '99. ACM, 1999, pp. 10-20. DOI: 10.1145/303976. 303978 (cit. on p. 27).

[阿洛+99] 9] 诺加·阿隆（Noga Alon）、菲利普·B·吉本斯（Phillip B. Gibbons）、约西·马蒂亚斯（Yossi Matias）和马里奥·塞格迪（Mario Szegedy）。“在有限存储中跟踪连接和自连接大小”。见：第18届数据库系统原理研讨会论文集。PODS '99。美国计算机协会（ACM），1999年，第10 - 20页。DOI: 10.1145/303976. 303978（引自第27页）。

[AMS96] | Noga Alon, Yossi Matias, and Mario Szegedy. "The Space Complexity of Approximating the Frequency Moments". In: Proceedings of the 28th Symposium on Theory of Computing. STOC '96. ACM, 1996, pp. 20-29. DOI: 10.1145/237814.237823 (cit. on p. 27).

[AMS96] | 诺加·阿隆（Noga Alon）、约西·马蒂亚斯（Yossi Matias）和马里奥·塞格迪（Mario Szegedy）。“近似频率矩的空间复杂度”。见：第28届计算理论研讨会论文集。STOC '96。美国计算机协会（ACM），1996年，第20 - 29页。DOI: 10.1145/237814.237823（引自第27页）。

[AMS99] Noga Alon, Yossi Matias, and Mario Szegedy. "The Space Complexity of Approximating the Frequency Moments". In: Journal of Computer and System Sciences 58.1 (1999), pp. 137-147. DOI: 10.1006/jcss. 1997.1545 Previously published as AMS96] (cit. on pp. 10, 12).

[AMS99] 诺加·阿隆（Noga Alon）、约西·马蒂亚斯（Yossi Matias）和马里奥·塞格迪（Mario Szegedy）。《近似频率矩的空间复杂度》。载于《计算机与系统科学杂志》（Journal of Computer and System Sciences）第58卷第1期（1999年），第137 - 147页。DOI: 10.1006/jcss. 1997.1545 此前以[AMS96]发表（见第10、12页引用）。

[AP12]

Mazin Aouf and Laurence Anthony F. Park. "Approximate Document Outlier Detection Using Random Spectral Projection". In: Proceedings of the 25th Australasian Joint Conference on Artificial Intelligence (AI'12). Vol. 7691. Lecture Notes in Computer Science (LNCS). Springer, 2012, pp. 579-590. DOI: 10.1007/978-3-642-35101-3_49 (cit. on p. 6).

马赞·奥夫（Mazin Aouf）和劳伦斯·安东尼·F·帕克（Laurence Anthony F. Park）。《使用随机谱投影的近似文档离群值检测》。载于《第25届澳大利亚人工智能联合会议论文集》（Proceedings of the 25th Australasian Joint Conference on Artificial Intelligence (AI'12)）。第7691卷。《计算机科学讲义》（Lecture Notes in Computer Science (LNCS)）。施普林格出版社（Springer），2012年，第579 - 590页。DOI: 10.1007/978-3-642-35101-3_49（见第6页引用）。

[Arp+14] 4] Devansh Arpit, Ifeoma Nwogu, Gaurav Srivastava, and Venu Govindaraju. "An Analysis of Random Projections in Cancelable Biometrics". In: arXiv e-prints (2014). arXiv: 1401.4489 [cs.CV] (cit. on pp. 4, 7).

[Arp+14] 4] 德万什·阿尔皮特（Devansh Arpit）、伊费奥马·恩沃古（Ifeoma Nwogu）、高拉夫·斯里瓦斯塔瓦（Gaurav Srivastava）和维努·戈文达拉朱（Venu Govindaraju）。《可撤销生物识别中随机投影的分析》。载于：arXiv预印本（2014年）。arXiv: 1401.4489 [计算机科学 - 计算机视觉]（见第4、7页引用）。

[AV06] Rosa I. Arriaga and Santosh S. Vempala. "An algorithmic theory of learning: Robust concepts and random projection". In: Machine Learning 63.2 (2006), pp. 161-182. DOI: 10.1007/s10994-006-6265-7. Previously published as [AV99] (cit. on pp. 4, 11,12.21

[AV06] 罗莎·I·阿里亚加（Rosa I. Arriaga）和桑托什·S·文帕拉（Santosh S. Vempala）。《学习的算法理论：鲁棒概念与随机投影》。载于：《机器学习》63.2（2006年），第161 - 182页。DOI: 10.1007/s10994 - 006 - 6265 - 7。之前以[AV99]发表（见第4、11、12、21页引用）

[AV99]

Rosa I. Arriaga and Santosh S. Vempala. "An Algorithmic Theory of Learning: Robust Concepts and Random Projection". In: Proceedings of the 40th Symposium on Foundations of Computer Science. FOCS '99. IEEE, 1999, pp. 616-623. DOI: 10.1109/ SFFCS. 1999.814637 (cit. on pp. 12, 27).

罗莎·I·阿里亚加（Rosa I. Arriaga）和桑托什·S·文帕拉（Santosh S. Vempala）。《学习的算法理论：鲁棒概念与随机投影》。载于《第40届计算机科学基础研讨会会议录》。FOCS '99。电气与电子工程师协会（IEEE），1999年，第616 - 623页。DOI: 10.1109/ SFFCS. 1999.814637（见第12、27页引用）。

[Avr+14] Haim Avron, Christos Boutsidis, Sivan Toledo, and Anastasios Zouzias. "Efficient Dimensionality Reduction for Canonical Correlation Analysis". In: Journal on Scientific Computing 36.5 (2014), S111-S131. DOI: 10.1137/130919222 (cit. on p. 6).

[Avr+14] 海姆·阿夫隆（Haim Avron）、克里斯托斯·布蒂西迪斯（Christos Boutsidis）、西万·托莱多（Sivan Toledo）和阿纳斯塔西奥斯·祖齐亚斯（Anastasios Zouzias）。《典型相关分析的高效降维方法》。载于《科学计算杂志》第36卷第5期（2014年），第S111 - S131页。DOI: 10.1137/130919222（见第6页引用）。

[Baj+07] Waheed Uz Zaman Bajwa, Jarvis D. Haupt, Gil M. Raz, Stephen J. Wright, and Robert D. Nowak. "Toeplitz-Structured Compressed Sensing Matrices". In: Proceedings of the 14th Workshop on Statistical Signal Processing. SSP '07. IEEE, 2007, pp. 294-298. DOI: 10.1109/SSP. 2007.430126 (cit. on p. 17).

[Baj+07] 瓦希德·乌兹·扎曼·巴杰瓦（Waheed Uz Zaman Bajwa）、贾维斯·D·豪普特（Jarvis D. Haupt）、吉尔·M·拉兹（Gil M. Raz）、斯蒂芬·J·赖特（Stephen J. Wright）和罗伯特·D·诺瓦克（Robert D. Nowak）。《托普利茨结构压缩感知矩阵》。载于：第14届统计信号处理研讨会论文集。SSP '07。电气与电子工程师协会（IEEE），2007年，第294 - 298页。DOI: 10.1109/SSP. 2007.430126（见第17页引用）。

[Baj12] Waheed Uz Zaman Bajwa. "Geometry of random Toeplitz-block sensing matrices: bounds and implications for sparse signal processing". In: Proceedings of the Compressed Sensing track of the 7th Conference on Defense, Security, and Sensing (DSS '12). Vol. 8365. Proceedings of SPIE. SPIE, 2012, pp. 16-22. DOI: 10.1117/12.919475 (cit. on p. 17).

[Baj12] 瓦希德·乌兹·扎曼·巴杰瓦（Waheed Uz Zaman Bajwa）。《随机托普利茨块感知矩阵的几何性质：对稀疏信号处理的边界和影响》。载于：第7届国防、安全与传感会议（DSS '12）压缩感知专题会议论文集。第8365卷。国际光学工程学会（SPIE）会议论文集。国际光学工程学会（SPIE），2012年，第16 - 22页。DOI: 10.1117/12.919475（见第17页引用）。

[Bar+08] Richard Baraniuk, Mark Davenport, Ronald DeVore, and Michael Wakin. "A Simple Proof of the Restricted Isometry Property for Random Matrices". In: Constructive Approximation 28.3 (2008), pp. 253-263. DOI: 10.1007/s00365-007-9003-x (cit. on p. 17).

[Bar+08] 理查德·巴拉纽克（Richard Baraniuk）、马克·达文波特（Mark Davenport）、罗纳德·德沃尔（Ronald DeVore）和迈克尔·韦金（Michael Wakin）。《随机矩阵受限等距性质的简单证明》。载于《构造性逼近》（Constructive Approximation）第28卷第3期（2008年），第253 - 263页。DOI: 10.1007/s00365 - 007 - 9003 - x（见第17页引用）。

[BDN15a] N15a] Jean Bourgain, Sjoerd Dirksen, and Jelani Nelson. "Toward a Unified Theory of Sparse Dimensionality Reduction in Euclidean Space". In: Proceedings of the 47th Symposium on Theory of Computing. STOC '15. ACM, 2015, pp. 499-508. DOI: 10.1145/2746539.2746541 (cit. on p. 28).

[BDN15a] 让·布尔甘（Jean Bourgain）、斯约德·迪尔克斯（Sjoerd Dirksen）和杰拉尼·纳尔逊（Jelani Nelson）。《迈向欧几里得空间稀疏降维的统一理论》。载于《第47届计算理论研讨会会议录》。STOC '15。美国计算机协会（ACM），2015年，第499 - 508页。DOI: 10.1145/2746539.2746541（见第28页引用）。

DN15b] Jean Bourgain, Sjoerd Dirksen, and Jelani Nelson. "Toward a unified theory of sparse dimensionality reduction in Euclidean space". In: Geometric and Functional Analysis (GAFA) 25.4 (2015), pp. 1009-1088. DOI: 10.1007/s00039-015-0332-9. Previously published as [BDN15a] (cit. on p. 4).

[DN15b] 让·布尔甘（Jean Bourgain）、斯约德·迪尔克斯（Sjoerd Dirksen）和杰拉尼·纳尔逊（Jelani Nelson）。“迈向欧几里得空间中稀疏降维的统一理论”。载于《几何与泛函分析》（Geometric and Functional Analysis，GAFA）第25卷第4期（2015年），第1009 - 1088页。DOI: 10.1007/s00039 - 015 - 0332 - 9。此前以[BDN15a]发表（见第4页引用）。

[Bec+19] Luca Becchetti, Marc Bury, Vincent Cohen-Addad, Fabrizio Grandoni, and Chris Schwiegelshohn. "Oblivious dimension reduction for $k$ -means: beyond subspaces and the Johnson-Lindenstrauss lemma". In: Proceedings of the 51st Symposium on Theory of Computing. STOC '19. ACM, 2019, pp. 1039-1050. DOI: 10.1145/3313276. 3316318 (cit. on pp. 6.9).

[Bec + 19] 卢卡·贝凯蒂（Luca Becchetti）、马克·伯里（Marc Bury）、文森特·科恩 - 阿达德（Vincent Cohen - Addad）、法布里齐奥·格兰多尼（Fabrizio Grandoni）和克里斯·施维格尔绍恩（Chris Schwiegelshohn）。“$k$ - 均值的无感知降维：超越子空间和约翰逊 - 林登施特劳斯引理”。载于《第51届计算理论研讨会会议录》。STOC '19。美国计算机协会（ACM），2019年，第1039 - 1050页。DOI: 10.1145/3313276. 3316318（见第6、9页引用）。

[BGK18]

Michael A. Burr, Shuhong Gao, and Fiona Knoll. "Optimal Bounds for Johnson-Lindenstrauss Transformations". In: Journal of Machine Learning Research 19.73 (2018), pp. 1-22 (cit. on p. 12).

迈克尔·A·伯尔（Michael A. Burr）、高树红（Shuhong Gao）和菲奥娜·诺尔（Fiona Knoll）。《约翰逊 - 林登施特劳斯变换的最优界》。载于《机器学习研究杂志》（Journal of Machine Learning Research）第19卷第73期（2018年），第1 - 22页（见第12页引用）。

[BK17] Stefan Bamberger and Felix Krahmer. "Optimal Fast Johnson-Lindenstrauss Em-beddings for Large Data Sets". In: arXiv e-prints (2017). arXiv: 1712.01774 [cs.DS] (cit. on pp. 18, 21).

[BK17] 斯特凡·班贝格（Stefan Bamberger）和费利克斯·克拉默（Felix Krahmer）。《大数据集的最优快速约翰逊 - 林登施特劳斯嵌入》。载于arXiv预印本（2017年）。arXiv: 1712.01774 [计算机科学 - 数据结构]（见第18、21页引用）。

[Blo+12]

Jeremiah Blocki, Avrim Blum, Anupam Datta, and Or Sheffet. "The Johnson-Lindenstrauss Transform Itself Preserves Differential Privacy". In: Proceedings of the 53rd Symposium on Foundations of Computer Science. FOCS '12. IEEE, 2012, pp. 410-419. DOI: 10.1109/FOCS. 2012.67 (cit. on pp. 6.7).

杰里迈亚·布洛基（Jeremiah Blocki）、阿夫里姆·布卢姆（Avrim Blum）、阿努帕姆·达塔（Anupam Datta）和奥尔·谢菲特（Or Sheffet）。《约翰逊 - 林登施特劳斯变换本身可保持差分隐私》。载于第53届计算机科学基础研讨会论文集。FOCS '12。电气与电子工程师协会（IEEE），2012年，第410 - 419页。DOI: 10.1109/FOCS. 2012.67（见第6、7页引用）。

[BM01]

Ella Bingham and Heikki Mannila. "Random Projection in Dimensionality Reduction: Applications to Image and Text Data". In: Proceedings of the 7th International Conference on Knowledge Discovery and Data Mining. KDD '01. ACM, 2001, pp. 245- 250 (cit. on p. 4).

埃拉·宾厄姆（Ella Bingham）和海基·曼尼拉（Heikki Mannila）。《降维中的随机投影：在图像和文本数据中的应用》。载于：第7届知识发现与数据挖掘国际会议论文集。KDD '01。美国计算机协会（ACM），2001年，第245 - 250页（见第4页引用）。

[Boc08] Hans-Hermann Bock. "Origins and Extensions of the $k$ -Means Algorithm in Cluster Analysis". In: Electronic Journal for History of Probability and Statistics 4.2 (2008), 9:1-9:18 (cit. on p. 9).

[Boc08] 汉斯 - 赫尔曼·博克（Hans - Hermann Bock）。《聚类分析中$k$均值算法的起源与扩展》。载于：《概率与统计史电子期刊》4.2（2008年），9:1 - 9:18（见第9页引用）。

[BOR10] Vladimir Braverman, Rafail Ostrovsky, and Yuval Rabani. "Rademacher Chaos, Random Eulerian Graphs and The Sparse Johnson-Lindenstrauss Transform". In: arXiv e-prints (2010). Presented at the Embedding worshop (DANW01) at Isaac Newton Institute for Mathematical Sciences. arXiv: 1011.2590 [cs.DS] (cit. on pp.13.14.

[BOR10] 弗拉基米尔·布拉弗曼（Vladimir Braverman）、拉斐尔·奥斯特罗夫斯基（Rafail Ostrovsky）和尤瓦尔·拉巴尼（Yuval Rabani）。《拉德马赫混沌、随机欧拉图与稀疏约翰逊 - 林登施特劳斯变换》。载于：arXiv预印本（2010年）。在艾萨克·牛顿数学科学研究所的嵌入研讨会（DANW01）上发表。arXiv: 1011.2590 [计算机科学 - 数据结构]（见第13、14页引用）。

[Bou+14] Christos Boutsidis, Anastasios Zouzias, Michael W. Mahoney, and Petros Drineas. "Randomized Dimensionality Reduction for $k$ -Means Clustering". In: Transactions on Information Theory 61.2 (2014), pp. 1045-1062. DOI: 10.1109/TIT. 2014.2375327. Previously published as [BZD10] (cit. on pp. 6,9).

[Bou+14] 克里斯托斯·布特西迪斯（Christos Boutsidis）、阿纳斯塔西奥斯·祖齐亚斯（Anastasios Zouzias）、迈克尔·W·马奥尼（Michael W. Mahoney）和佩特罗斯·德里尼亚斯（Petros Drineas）。“$k$ - 均值聚类的随机降维”。载于《信息论汇刊》（Transactions on Information Theory）第61卷第2期（2014年），第1045 - 1062页。DOI: 10.1109/TIT. 2014.2375327。此前以[BZD10]发表（见第6、9页引用）。

[Bre+19] Anna Breger, Gabriel Ramos Llorden, Gonzalo Vegas Sanchez-Ferrero, W. Scott Hoge, Martin Ehler, and Carl-Fredrik Westin. "On the Reconstruction Accuracy of Multi-Coil MRI with Orthogonal Projections". In: arXiv e-prints (2019). arXiv: 1910.13422 [physics.med-ph] (cit. on p. 4).

[Bre+19] 安娜·布雷格（Anna Breger）、加布里埃尔·拉莫斯·洛登（Gabriel Ramos Llorden）、贡萨洛·维加斯·桑切斯 - 费雷罗（Gonzalo Vegas Sanchez-Ferrero）、W·斯科特·霍奇（W. Scott Hoge）、马丁·埃勒（Martin Ehler）和卡尔 - 弗雷德里克·韦斯汀（Carl-Fredrik Westin）。“正交投影下多线圈磁共振成像的重建精度”。载于arXiv预印本（2019年）。arXiv: 1910.13422 [物理学.医学物理]（见第4页引用）。

[Bre+20] Anna Breger, José Ignacio Orlando, Pavol Harár, Monika Dörfler, Sophie Klimscha, Christoph Grechenig, Bianca S. Gerendas, Ursula Schmidt-Erfurth, and Martin Ehler. "On Orthogonal Projections for Dimension Reduction and Applications in Augmented Target Loss Functions for Learning Problems". In: Journal of Mathematical Imaging and Vision 62.3 (2020), pp. 376-394. DOI: 10.1007/s10851-019-00902-2 (cit. on $p.\left( 4\right)$ .

[布雷+20] 安娜·布雷格（Anna Breger）、何塞·伊格纳西奥·奥兰多（José Ignacio Orlando）、帕沃尔·哈拉尔（Pavol Harár）、莫妮卡·多尔弗勒（Monika Dörfler）、索菲·克利姆沙（Sophie Klimscha）、克里斯托夫·格雷切尼格（Christoph Grechenig）、比安卡·S·格伦达斯（Bianca S. Gerendas）、乌尔苏拉·施密特 - 埃尔富特（Ursula Schmidt-Erfurth）和马丁·埃勒（Martin Ehler）。“关于降维的正交投影及其在学习问题增强目标损失函数中的应用”。载于：《数学成像与视觉杂志》（Journal of Mathematical Imaging and Vision）62.3（2020年），第376 - 394页。DOI: 10.1007/s10851 - 019 - 00902 - 2（在$p.\left( 4\right)$处引用）。

[BZD10] Christos Boutsidis, Anastasios Zouzias, and Petros Drineas. "Random Projections for $k$ -means Clustering". In: Advances in Neural Information Processing Systems 23. NIPS '10. Curran Associates, Inc., 2010, pp. 298-306 (cit. on p. 29).

[布茨迪迪斯+10] 克里斯托斯·布茨迪迪斯（Christos Boutsidis）、阿纳斯塔西奥斯·祖齐亚斯（Anastasios Zouzias）和彼得罗斯·德里尼亚斯（Petros Drineas）。“用于$k$均值聚类的随机投影”。载于：《神经信息处理系统进展》（Advances in Neural Information Processing Systems）23。NIPS '10。柯伦联合公司（Curran Associates, Inc.），2010年，第298 - 306页（在第29页引用）。

[Can20] Timothy I. Cannings. "Random projections: Data perturbation for classification problems". In: WIREs Computational Statistics 13.1 (2020), e1499. DOI: 10.1002/wics. 1499 (cit. on p. 6).

[Can20] 蒂莫西·I·坎宁斯（Timothy I. Cannings）。《随机投影：分类问题的数据扰动》。载于《威立跨学科评论：计算统计学》（WIREs Computational Statistics）第13卷第1期（2020年），e1499。DOI: 10.1002/wics. 1499（见第6页引用）。

[Car+13] Sophie J. C. Caron, Vanessa Ruta, L. F. Abbott, and Richard Axel. "Random convergence of olfactory inputs in the Drosophila mushroom body". In: Nature 497.7447 (2013), pp. 113-117. DOI: 10.1038/nature12063 (cit. on p. 7).

[Car+13] 索菲·J·C·卡龙（Sophie J. C. Caron）、凡妮莎·鲁塔（Vanessa Ruta）、L·F·阿博特（L. F. Abbott）和理查德·阿克塞尔（Richard Axel）。《果蝇蘑菇体中嗅觉输入的随机汇聚》。载于《自然》（Nature）第497卷第7447期（2013年），第113 - 117页。DOI: 10.1038/nature12063（见第7页引用）。

[CCF02] Moses Charikar, Kevin C. Chen, and Martin Farach-Colton. "Finding Frequent Items in Data Streams". In: Proceedings of the 29th International Colloquium on Automata, Languages and Programming (ICALP '02). Vol. 2380. Lecture Notes in Computer Science (LNCS). Springer, 2002, pp. 693-703. DOI: 10.1007/3-540-45465-9_59 (cit. on p. 29).

[CCF02] 摩西·查里卡尔（Moses Charikar）、凯文·C·陈（Kevin C. Chen）和马丁·法拉奇 - 科尔托（Martin Farach - Colton）。《在数据流中查找频繁项》。载于《第29届自动机、语言和程序设计国际学术讨论会论文集》（Proceedings of the 29th International Colloquium on Automata, Languages and Programming，ICALP '02）。第2380卷。《计算机科学讲义》（Lecture Notes in Computer Science，LNCS）。施普林格出版社，2002年，第693 - 703页。DOI: 10.1007/3 - 540 - 45465 - 9_59（见第29页引用）。

[CCF04]

Moses Charikar, Kevin C. Chen, and Martin Farach-Colton. "Finding Frequent Items in Data Streams". In: Theoretical Computer Science 312.1 (2004), pp. 3-15. DOI: 10.1016/S0304-3975(03)00400-6. Previously published as [CCF02] (cit. on pp. 10, 11,13).

摩西·查里卡尔（Moses Charikar）、凯文·C·陈（Kevin C. Chen）和马丁·法拉奇 - 科尔顿（Martin Farach - Colton）。《在数据流中查找频繁项》。载于《理论计算机科学》（Theoretical Computer Science）312.1（2004年），第3 - 15页。DOI: 10.1016/S0304 - 3975(03)00400 - 6。之前以[CCF02]发表（见第10、11、13页引用）。

[CG05] Graham Cormode and Minos N. Garofalakis. "Sketching Streams Through the Net: Distributed Approximate Query Tracking". In: Proceedings of the 31st International Conference on Very Large Data Bases. VLDB '05. ACM, 2005, pp. 13-24 (cit. on pp. 10, 11,13).

[CG05]格雷厄姆·科莫德（Graham Cormode）和米诺斯·N·加罗法拉克基斯（Minos N. Garofalakis）。《通过网络对数据流进行概要分析：分布式近似查询跟踪》。载于《第31届超大型数据库国际会议论文集》。VLDB '05。美国计算机协会（ACM），2005年，第13 - 24页（见第10、11、13页引用）。

[CJS09] Robert Calderbank, Sina Jafarpour, and Robert Schapire. "Compressed Learning: Universal Sparse Dimensionality Reduction and Learning in the Measurement Domain". Manuscript. 2009. URL: https://core.ac.uk/display/21147568 (cit. on p. 6).

[CJS09]罗伯特·卡尔德班克（Robert Calderbank）、西纳·贾法普尔（Sina Jafarpour）和罗伯特·沙皮尔（Robert Schapire）。《压缩学习：测量域中的通用稀疏降维和学习》。手稿。2009年。网址：https://core.ac.uk/display/21147568（见第6页引用）。

[CMM17] Michael B. Cohen, Cameron Musco, and Christopher Musco. "Input Sparsity Time Low-rank Approximation via Ridge Leverage Score Sampling". In: Proceedings of the 28th Symposium on Discrete Algorithms. SODA '17. SIAM, 2017, pp. 1758-1777. DOI: 10.1137/1.9781611974782.115 (cit. on p. 38).

[CMM17] 迈克尔·B·科恩（Michael B. Cohen）、卡梅伦·马斯科（Cameron Musco）和克里斯托弗·马斯科（Christopher Musco）。“通过岭杠杆得分采样实现输入稀疏时间低秩逼近”。见：第28届离散算法研讨会会议录。SODA '17。工业与应用数学学会（SIAM），2017年，第1758 - 1777页。DOI: 10.1137/1.9781611974782.115（见第38页引用）。

[Coh+15] Michael B. Cohen, Sam Elder, Cameron Musco, Christopher Musco, and Madalina Persu. "Dimensionality Reduction for $k$ -Means Clustering and Low Rank Approximation". In: Proceedings of the 47th Symposium on Theory of Computing. STOC '15. ACM, 2015, pp. 163-172. DOI: 10.1145/2746539. 2746569 (cit. on pp. 6, 9.38).

[Coh+15] 迈克尔·B·科恩（Michael B. Cohen）、山姆·埃尔德（Sam Elder）、卡梅伦·马斯科（Cameron Musco）、克里斯托弗·马斯科（Christopher Musco）和马达利娜·佩尔苏（Madalina Persu）。“$k$ - 均值聚类和低秩逼近的降维”。见：第47届计算理论研讨会会议录。STOC '15。美国计算机协会（ACM），2015年，第163 - 172页。DOI: 10.1145/2746539. 2746569（见第6、9、38页引用）。

[CRT06] Emmanuel J. Candès, Justin K. Romberg, and Terence Tao. "Robust uncertainty principles: exact signal reconstruction from highly incomplete frequency information". In: Transactions on Information Theory 52.2 (2006), pp. 489-509. DOI: 10.1109/TIT.2005.862083 (cit. on p.16).

[CRT06] 伊曼纽尔·J·坎德斯（Emmanuel J. Candès）、贾斯汀·K·龙伯格（Justin K. Romberg）和陶哲轩（Terence Tao）。《稳健的不确定性原理：从高度不完整的频率信息中精确重建信号》。载于《信息论汇刊》（Transactions on Information Theory）2006年第52卷第2期，第489 - 509页。DOI: 10.1109/TIT.2005.862083（见第16页引用）。

[CS17] Timothy I. Cannings and Richard J. Samworth. "Random Projection Ensemble Classification". In: Journal of the Royal Statistical Society: Series B (Statistical Methodology) 79.4 (2017), pp. 959-1035. DOI: 10.1111/rssb. 12228 (cit. on p. 6).

[CS17] 蒂莫西·I·坎宁斯（Timothy I. Cannings）和理查德·J·萨姆沃思（Richard J. Samworth）。《随机投影集成分类》。载于《皇家统计学会学报：B辑（统计方法）》（Journal of the Royal Statistical Society: Series B (Statistical Methodology)）2017年第79卷第4期，第959 - 1035页。DOI: 10.1111/rssb. 12228（见第6页引用）。

[CT05]

Emmanuel J. Candès and Terence Tao. "Decoding by linear programming". In: Transactions on Information Theory 51.12 (2005), pp. 4203-4215. DOI: 10.1109/TIT, 2005.858979 (cit. on p. 16).

伊曼纽尔·J·坎德斯（Emmanuel J. Candès）和陶哲轩（Terence Tao）。《通过线性规划解码》。载于《信息论汇刊》（Transactions on Information Theory）2005年第51卷第12期，第4203 - 4215页。DOI: 10.1109/TIT, 2005.858979（见第16页引用）。

[CT06]

Emmanuel J. Candès and Terence Tao. "Near-Optimal Signal Recovery From Random Projections: Universal Encoding Strategies?" In: Transactions on Information Theory 52.12 (2006), pp. 5406-5425. DOI: 10.1109/TIT. 2006.885507 (cit. on p. 16).

伊曼纽尔·J·坎德斯（Emmanuel J. Candès）和陶哲轩（Terence Tao）。《从随机投影中近乎最优地恢复信号：通用编码策略？》，载于《信息论汇刊》（Transactions on Information Theory）第52卷第12期（2006年），第5406 - 5425页。DOI: 10.1109/TIT. 2006.885507（见第16页引用）。

[CW13]

Kenneth L. Clarkson and David P. Woodruff. "Low rank approximation and regression in input sparsity time". In: Proceedings of the 45th Symposium on Theory of Computing. STOC '13. ACM, 2013, pp. 81-90. DOI: 10.1145/2488608. 2488620 (cit. on p. 30).

肯尼斯·L·克拉克森（Kenneth L. Clarkson）和大卫·P·伍德拉夫（David P. Woodruff）。《输入稀疏时间下的低秩逼近与回归》，载于《第45届计算理论研讨会论文集》（Proceedings of the 45th Symposium on Theory of Computing）。STOC '13。美国计算机协会（ACM），2013年，第81 - 90页。DOI: 10.1145/2488608. 2488620（见第30页引用）。

[CW17]

Kenneth L. Clarkson and David P. Woodruff. "Low-Rank Approximation and Regression in Input Sparsity Time". In: Journal of the ACM 63.6 (2017), 54:1-54:45. DOI: 10.1145/3019134. Previously published as [CW13] (cit. on p. 6).

肯尼斯·L·克拉克森（Kenneth L. Clarkson）和大卫·P·伍德拉夫（David P. Woodruff）。《输入稀疏时间下的低秩逼近与回归》，载于《美国计算机协会期刊》（Journal of the ACM）第63卷第6期（2017年），54:1 - 54:45。DOI: 10.1145/3019134。此前以[CW13]发表（见第6页引用）。

[Das00]

Sanjoy Dasgupta. "Experiments with Random Projection". In: Proceedings of the 16th Conference on Uncertainty in Artificial Intelligence. UAI '00. Morgan Kaufmann, 2000, pp. 143-151 (cit. on p. 4).

桑乔伊·达斯古普塔（Sanjoy Dasgupta）。《随机投影实验》。载于：第16届人工智能不确定性会议论文集。UAI '00。摩根·考夫曼出版社（Morgan Kaufmann），2000年，第143 - 151页（见第4页引用）。

[Das08]

Sanjoy Dasgupta. The hardness of k-means clustering. Tech. rep. CS2008-0916. University of California San Diego, 2008 (cit. on p. 7).

桑乔伊·达斯古普塔（Sanjoy Dasgupta）。《k - 均值聚类的难度》。技术报告CS2008 - 0916。加利福尼亚大学圣地亚哥分校（University of California San Diego），2008年（见第7页引用）。

[Das99] Sanjoy Dasgupta. "Learning Mixtures of Gaussians". In: Proceedings of the 40th Symposium on Foundations of Computer Science. FOCS '99. IEEE, 1999, pp. 634-644. DOI: 10.1109/SFFCS.1999.814639 (cit. on p. 6).

[Das99] 桑乔伊·达斯古普塔（Sanjoy Dasgupta）。《学习高斯混合模型》。载于：第40届计算机科学基础研讨会论文集。FOCS '99。电气与电子工程师协会（IEEE），1999年，第634 - 644页。DOI: 10.1109/SFFCS.1999.814639（见第6页引用）。

[Dat+04] Mayur Datar, Nicole Immorlica, Piotr Indyk, and Vahab Seyed Mirrokni. "Locality-Sensitive Hashing Scheme Based on p-Stable Distributions". In: Proceedings of the 20th Symposium on Computational Geometry. SoCG '04. ACM, 2004, pp. 253-262. DOI: 10.1145/997817.997857 (cit. on p. 5).

[Dat+04] 马尤尔·达塔尔（Mayur Datar）、妮可·伊莫利卡（Nicole Immorlica）、彼得·因迪克（Piotr Indyk）和瓦哈布·赛义德·米罗克尼（Vahab Seyed Mirrokni）。“基于p-稳定分布的局部敏感哈希方案”。载于：第20届计算几何研讨会论文集。SoCG '04。美国计算机协会（ACM），2004年，第253 - 262页。DOI: 10.1145/997817.997857（见第5页引用）。

[DB06]

Sampath Deegalla and Henrik Boström. "Reducing High-Dimensional Data by Principal Component Analysis vs. Random Projection for Nearest Neighbor Classification". In: Proceedings of the 5th International Conference on Machine Learning and Applications. ICMLA '06. IEEE, 2006, pp. 245-250. DOI: 10.1109/ICMLA. 2006.43 (cit. on p. 4).

桑帕特·迪加拉（Sampath Deegalla）和亨里克·博斯特伦（Henrik Boström）。“主成分分析与随机投影在最近邻分类中对高维数据的降维比较”。载于：第5届机器学习与应用国际会议论文集。ICMLA '06。电气和电子工程师协会（IEEE），2006年，第245 - 250页。DOI: 10.1109/ICMLA. 2006.43（见第4页引用）。

[DDH07] 107] James Demmel, Ioana Dumitriu, and Olga Holtz. "Fast linear algebra is stable". In: Numerische Mathematik 108.1 (2007), pp. 59-91. DOI: 10.1007/s00211-007-0114-x (cit. on p. 4).

[DDH07] 107] 詹姆斯·德梅尔（James Demmel）、伊奥娜·杜米特留（Ioana Dumitriu）和奥尔加·霍尔茨（Olga Holtz）。《快速线性代数是稳定的》。载于《数值数学》（Numerische Mathematik）第108卷第1期（2007年），第59 - 91页。DOI: 10.1007/s00211-007-0114-x（见第4页引用）。

[DeW+92] David J. DeWitt, Jeffrey F. Naughton, Donovan A. Schneider, and S. Seshadri. "Practical Skew Handling in Parallel Joins". In: Proceedings of the 18th International Conference on Very Large Data Bases. VLDB '92. Morgan Kaufmann, 1992, pp. 27-40 (cit. on p. 10).

[DeW+92] 大卫·J·德维特（David J. DeWitt）、杰弗里·F·诺顿（Jeffrey F. Naughton）、多诺万·A·施耐德（Donovan A. Schneider）和S·塞沙德里（S. Seshadri）。《并行连接中的实用倾斜处理》。载于《第18届超大型数据库国际会议论文集》。VLDB '92。摩根·考夫曼出版社（Morgan Kaufmann），1992年，第27 - 40页（见第10页引用）。

[DG02] Sanjoy Dasgupta and Anupam Gupta. "An Elementary Proof of a Theorem of Johnson and Lindenstrauss". In: Random Structures & Algorithms 22.1 (2002), pp. 60- 65. DOI: 10.1002/rsa. 10073 (cit. on p. 12).

[DG02] 桑乔伊·达斯古普塔（Sanjoy Dasgupta）和阿努帕姆·古普塔（Anupam Gupta）。《约翰逊 - 林登施特劳斯定理的初等证明》。载于《随机结构与算法》（Random Structures & Algorithms）第22卷第1期（2002年），第60 - 65页。DOI: 10.1002/rsa. 10073（见第12页引用）。

[Dir16]

Sjoerd Dirksen. "Dimensionality Reduction with Subgaussian Matrices: A Unified Theory". In: Foundations of Computational Mathematics volume 16.5 (2016), pp. 1367- 1396. DOI: 10.1007/s10208-015-9280-x (cit. on p. 4).

斯约德·迪尔克斯（Sjoerd Dirksen）。《亚高斯矩阵的降维：统一理论》。载于《计算数学基础》第16卷第5期（2016年），第1367 - 1396页。DOI: 10.1007/s10208-015-9280-x（见第4页引用）。

[DJR19]

Sjoerd Dirksen, Hans Christian Jung, and Holger Rauhut. "One-bit compressed sensing with partial Gaussian circulant matrices". In: Information and Inference: A Journal of the IMA (2019), iaz017. DOI: 10.1093/imaiai/iaz017 (cit. on p. 17).

斯约德·迪尔克斯（Sjoerd Dirksen）、汉斯·克里斯蒂安·荣格（Hans Christian Jung）和霍尔格·劳胡特（Holger Rauhut）。《基于部分高斯循环矩阵的一位压缩感知》。载于《信息与推理：IMA期刊》（2019年），iaz017。DOI: 10.1093/imaiai/iaz017（见第17页引用）。

[DK10]

Robert J. Durrant and Ata Kabán. "Compressed Fisher Linear Discriminant Analysis: Classification of Randomly Projected Data". In: Proceedings of the 16th International Conference on Knowledge Discovery and Data Mining. KDD '10. ACM, 2010, pp. 1119- 1128. DOI: 10.1145/1835804.1835945 (cit. on p. 6).

罗伯特·J·杜兰特（Robert J. Durrant）和阿塔·卡班（Ata Kabán）。《压缩费舍尔线性判别分析：随机投影数据的分类》。载于《第16届知识发现与数据挖掘国际会议论文集》。KDD '10。美国计算机协会（ACM），2010年，第1119 - 1128页。DOI: 10.1145/1835804.1835945（见第6页引用）。

[DK13]

Robert Durrant and Ata Kabán. "Random Projections as Regularizers: Learning a Linear Discriminant Ensemble from Fewer Observations than Dimensions". In: Proceedings of the 5th Asian Conference on Machine Learning (ACML '13). Vol. 29. Proceedings of Machine Learning Research (PMLR). PMLR, 2013, pp. 17-32 (cit. on p. 6).

罗伯特·杜兰特（Robert Durrant）和阿塔·卡班（Ata Kabán）。《随机投影作为正则化器：从少于维度数的观测值中学习线性判别集成》。载于：第五届亚洲机器学习会议（ACML '13）论文集。第29卷。机器学习研究会议录（PMLR）。机器学习研究会议录出版社（PMLR），2013年，第17 - 32页（见第6页引用）。

[DKS10]

Anirban Dasgupta, Ravi Kumar, and Tamás Sarlós. "A Sparse Johnson-Lindenstrauss Transform". In: Proceedings of the 42nd Symposium on Theory of Computing. STOC '10. ACM, 2010, pp. 341-350. DOI: 10.1145/1806689. 1806737 (cit. on pp. 13, 14.21).

阿尼尔班·达斯古普塔（Anirban Dasgupta）、拉维·库马尔（Ravi Kumar）和塔马斯·萨尔洛斯（Tamás Sarlós）。《一种稀疏约翰逊 - 林登施特劳斯变换》。载于：第四十二届计算理论研讨会论文集。STOC '10。美国计算机协会（ACM），2010年，第341 - 350页。DOI: 10.1145/1806689. 1806737（见第13、14、21页引用）。

[DKT17] Søren Dahlgaard, Mathias Bæk Tejs Knudsen, and Mikkel Thorup. "Practical Hash Functions for Similarity Estimation and Dimensionality Reduction". In: Advances in Neural Information Processing Systems 30. NIPS '17. Curran Associates, Inc., 2017, pp. 6615-6625 (cit. on p. 14).

[DKT17] 索伦·达尔加德（Søren Dahlgaard）、马蒂亚斯·贝克·泰斯·克努森（Mathias Bæk Tejs Knudsen）和米克尔·索鲁普（Mikkel Thorup）。“用于相似度估计和降维的实用哈希函数”。见：《神经信息处理系统进展30》。神经信息处理系统大会（NIPS）2017年会议论文集。柯伦联合公司（Curran Associates, Inc.），2017年，第6615 - 6625页（见第14页引用）。

[Do+09] Thong T. Do, Lu Gan, Yi Chen, Nam Hoai Nguyen, and Trac D. Tran. "Fast and efficient dimensionality reduction using Structurally Random Matrices". In: Proceedings of the 34th International Conference on Acoustics, Speech, and Signal Processing. ICASSP '09. IEEE, 2009, pp. 1821-1824. DOI: 10.1109/ICASSP. 2009. 4959960 (cit. on pp. 17, 21).

[Do+09] 杜通·T（Thong T. Do）、甘璐（Lu Gan）、陈怡（Yi Chen）、阮南怀（Nam Hoai Nguyen）和陈德·T（Trac D. Tran）。“使用结构随机矩阵进行快速高效的降维”。见：第34届国际声学、语音和信号处理会议论文集。国际声学、语音和信号处理会议（ICASSP）2009年会议论文集。电气和电子工程师协会（IEEE），2009年，第1821 - 1824页。DOI: 10.1109/ICASSP. 2009. 4959960（见第17、21页引用）。

[Don06] David L. Donoho. "For Most Large Underdetermined Systems of Equations, the Minimal ${\ell }_{1}$ -norm Near-Solution Approximates the Sparsest Near-Solution". In: Communications on Pure and Applied Mathematics 59.7 (2006), pp. 907-934. DOI: 10.1002/cpa.20131 (cit. on p.16).

[Don06] 大卫·L·多诺霍（David L. Donoho）。“对于大多数大型欠定方程组，最小 ${\ell }_{1}$ -范数近似解逼近最稀疏近似解”。载于《纯粹与应用数学通讯》（Communications on Pure and Applied Mathematics）第59卷第7期（2006年），第907 - 934页。DOI: 10.1002/cpa.20131（见第16页引用）。

[dVCH10] Timothy de Vries, Sanjay Chawla, and Michael E. Houle. "Finding Local Anomalies in Very High Dimensional Space". In: Proceedings of the 10th International Conference on Data Mining. ICDM '03. IEEE, 2010, pp. 128-137. DOI: 10.1109/ICDM. 2010.151 (cit. on p. 6).

[dVCH10] 蒂莫西·德弗里斯（Timothy de Vries）、桑杰·乔拉（Sanjay Chawla）和迈克尔·E·霍尔（Michael E. Houle）。“在超高维空间中寻找局部异常值”。载于《第10届国际数据挖掘会议论文集》（Proceedings of the 10th International Conference on Data Mining）。ICDM '03。电气与电子工程师协会（IEEE），2010年，第128 - 137页。DOI: 10.1109/ICDM. 2010.151（见第6页引用）。

[FB03] Xiaoli Zhang Fern and Carla E. Brodley. "Random Projection for High Dimensional Data Clustering: A Cluster Ensemble Approach". In: Proceedings of the 20th International Conference on Machine Learning. ICML '03. AAAI Press, 2003, pp. 186-193 (cit. on pp. 4-6).

[FB03] 张晓丽（Xiaoli Zhang Fern）和卡拉·E·布罗德利（Carla E. Brodley）。“高维数据聚类的随机投影：一种聚类集成方法”。见：第20届国际机器学习会议论文集。ICML '03。美国人工智能协会出版社（AAAI Press），2003年，第186 - 193页（引用见第4 - 6页）。

[FKL18]

Casper Benjamin Freksen, Lior Kamma, and Kasper Green Larsen. "Fully Understanding the Hashing Trick". In: Advances in Neural Information Processing Systems 31. NeurIPS '18. Curran Associates, Inc., 2018, pp. 5394-5404 (cit. on pp. 14, 15. 18, 21).

卡斯珀·本杰明·弗雷克森（Casper Benjamin Freksen）、利奥尔·卡马（Lior Kamma）和卡斯珀·格林·拉森（Kasper Green Larsen）。“全面理解哈希技巧”。见：《神经信息处理系统进展31》。NeurIPS '18。柯伦联合公司（Curran Associates, Inc.），2018年，第5394 - 5404页（引用见第14、15、18、21页）。

[FL17]

Casper Benjamin Freksen and Kasper Green Larsen. "On Using Toeplitz and Circulant Matrices for Johnson-Lindenstrauss Transforms". In: Proceedings of the 28th International Symposium on Algorithms and Computation (ISAAC '17). Vol. 92. Leibniz International Proceedings in Informatics (LIPIcs). Schloss Dagstuhl, 2017, 32:1-32:12. DOI: 10.4230/LIPIcs.ISAAC. 2017.32 (cit. on p. 32).

卡斯珀·本杰明·弗雷克森（Casper Benjamin Freksen）和卡斯珀·格林·拉森（Kasper Green Larsen）。《关于使用托普利茨矩阵（Toeplitz）和循环矩阵（Circulant）进行约翰逊 - 林登斯特劳斯变换（Johnson - Lindenstrauss Transforms）》。见：第28届国际算法与计算研讨会（ISAAC '17）论文集。第92卷。莱布尼茨国际信息学会议录（LIPIcs）。德国达格斯图尔城堡出版社，2017年，32:1 - 32:12。DOI: 10.4230/LIPIcs.ISAAC. 2017.32（见第32页引用）。

[FL20]

Casper Benjamin Freksen and Kasper Green Larsen. "On Using Toeplitz and Circulant Matrices for Johnson-Lindenstrauss Transforms". In: Algorithmica 82.2 (2020), pp. 338-354. DOI: 10.1007/s00453-019-00644-y. Previously published as [FL17] (cit. on pp. 18, 21).

卡斯珀·本杰明·弗雷克森（Casper Benjamin Freksen）和卡斯珀·格林·拉森（Kasper Green Larsen）。《关于使用托普利茨矩阵（Toeplitz）和循环矩阵（Circulant）进行约翰逊 - 林登斯特劳斯变换（Johnson - Lindenstrauss Transforms）》。见：《算法理论》（Algorithmica）82.2（2020年），第338 - 354页。DOI: 10.1007/s00453 - 019 - 00644 - y。之前以[FL17]发表（见第18、21页引用）。

[FM03]

Dmitriy Fradkin and David Madigan. "Experiments with Random Projections for Machine Learning". In: Proceedings of the 9th International Conference on Knowledge Discovery and Data Mining. KDD '03. ACM, 2003, pp. 517-522. DOI: 10.1145/956750. 956812 (cit. on p. 4).

德米特里·弗拉德金（Dmitriy Fradkin）和大卫·马迪根（David Madigan）。《机器学习随机投影实验》。载于：第9届知识发现与数据挖掘国际会议论文集。KDD '03。美国计算机协会（ACM），2003年，第517 - 522页。DOI: 10.1145/956750. 956812（见第4页引用）。

[FM88]

Peter Frankl and Hiroshi Maehara. "The Johnson-Lindenstrauss lemma and the sphericity of some graphs". In: Journal of Combinatorial Theory, Series B 44.3 (1988), pp. 355-362. DOI: 10.1016/0095-8956(88) 90043-3 (cit. on pp. 7.11, 12).

彼得·弗兰克尔（Peter Frankl）和前原浩（Hiroshi Maehara）。《约翰逊 - 林登施特劳斯引理与某些图的球形性》。载于：《组合理论杂志，B辑》44.3（1988年），第355 - 362页。DOI: 10.1016/0095-8956(88) 90043-3（见第7、11、12页引用）。

[FM90] Peter Frankl and Hiroshi Maehara. "Some geometric applications of the beta distribution". In: Annals of the Institute of Statistical Mathematics (AISM) 42 (1990), pp. 463-474. DOI: 10.1007/BF00049302 (cit. on p. 11).

[FM90] 彼得·弗兰克尔（Peter Frankl）和前原浩（Hiroshi Maehara）。《β分布的一些几何应用》。载于：《统计数学研究所年报》（AISM）42（1990年），第463 - 474页。DOI: 10.1007/BF00049302（见第11页引用）。

[Fre20] Casper Benjamin Freksen. "A Song of Johnson and Lindenstrauss". PhD thesis. Aarhus, Denmark: Aarhus University, 2020 (cit. on p. 1).

[Fre20] 卡斯珀·本杰明·弗雷克森（Casper Benjamin Freksen）。《约翰逊与林登施特劳斯之歌》（"A Song of Johnson and Lindenstrauss"）。博士论文。丹麦奥胡斯（Aarhus）：奥胡斯大学（Aarhus University），2020年（见第1页引用）。

[GD08] Kuzman Ganchev and Mark Dredze. "Small Statistical Models by Random Feature Mixing". In: Mobile NLP workshop at the 46th Annual Meeting of the Association for Computational Linguistics (ACL '08). ACL08-Mobile-NLP. 2008, pp. 19-20 (cit. on pp.13.14.

[GD08] 库兹曼·甘切夫（Kuzman Ganchev）和马克·德雷兹（Mark Dredze）。《通过随机特征混合构建小型统计模型》（"Small Statistical Models by Random Feature Mixing"）。载于：第46届计算语言学协会年会（ACL '08）移动自然语言处理研讨会。ACL08 - 移动自然语言处理。2008年，第19 - 20页（见第13、14页引用）。

[Gil+01]

Anna C. Gilbert, Yannis Kotidis, Shanmugavelayutham Muthukrishnan, and M. J. Strauss. QuickSAND: Quick Summary and Analysis of Network Data. Tech. rep. 2001-43. DIMACS, 2001 (cit. on p. 10).

安娜·C·吉尔伯特（Anna C. Gilbert）、扬尼斯·科蒂迪斯（Yannis Kotidis）、尚穆加韦拉尤坦·穆图克里什南（Shanmugavelayutham Muthukrishnan）和M. J. 施特劳斯（M. J. Strauss）。《QuickSAND：网络数据的快速总结与分析》（QuickSAND: Quick Summary and Analysis of Network Data）。技术报告2001 - 43。离散数学与理论计算机科学中心（DIMACS），2001年（见第10页引用）。

[GLK13] Chris Giannella, Kun Liu, and Hillol Kargupta. "Breaching Euclidean Distance-Preserving Data Perturbation using few Known Inputs". In: Data & Knowledge Engineering 83 (2013), pp. 93-110. DOI: 10.1016/j. datak. 2012.10.004 (cit. on p. $7)$ .

[GLK13] 克里斯·詹内拉（Chris Giannella）、刘坤（Kun Liu）和希洛尔·卡尔古普塔（Hillol Kargupta）。“利用少量已知输入突破欧几里得距离保持数据扰动”。载于《数据与知识工程》（Data & Knowledge Engineering）83 (2013)，第93 - 110页。DOI: 10.1016/j. datak. 2012.10.004（见第$7)$页引用）。

[GS12] Surya Ganguli and Haim Sompolinsky. "Compressed Sensing, Sparsity, and Dimensionality in Neuronal Information Processing and Data Analysis". In: Annual Review of Neuroscience 35.1 (2012), pp. 485-508. DOI: 10.1146/annurev-neuro- 062111-150410 (cit. on p.7).

[GS12] 苏里亚·甘古利（Surya Ganguli）和海姆·桑波林斯基（Haim Sompolinsky）。“神经元信息处理和数据分析中的压缩感知、稀疏性和维度”。载于《神经科学年度评论》（Annual Review of Neuroscience）35.1 (2012)，第485 - 508页。DOI: 10.1146/annurev - neuro - 062111 - 150410（见第7页引用）。

Guo+20] Xiao Guo, Yixuan Qiu, Hai Zhang, and Xiangyu Chang. "Randomized spectral co-clustering for large-scale directed networks". In: arXiv e-prints (2020). arXiv: 2004.12164 [stat.ML] (cit. on p. 6).

[Guo+20] 郭晓（Xiao Guo）、邱逸轩（Yixuan Qiu）、张海（Hai Zhang）和常翔宇（Xiangyu Chang）。“大规模有向网络的随机谱共聚类”。载于arXiv预印本 (2020)。arXiv: 2004.12164 [stat.ML]（见第6页引用）。

[Har01] Sariel Har-Peled. "A Replacement for Voronoi Diagrams of Near Linear Size". In: Proceedings of the 42nd Symposium on Foundations of Computer Science. FOCS '01. IEEE, 2001, pp. 94-103. DOI: 10.1109/SFCS. 2001.959884 (cit. on p. 33).

[Har01] 萨里尔·哈尔 - 佩莱德（Sariel Har-Peled）。“近线性规模的沃罗诺伊图替代方案”。见：第42届计算机科学基础研讨会论文集。FOCS '01。电气与电子工程师协会（IEEE），2001年，第94 - 103页。DOI: 10.1109/SFCS. 2001.959884（引自第33页）。

[Hau+10] Jarvis D. Haupt, Waheed Uz Zaman Bajwa, Gil M. Raz, and Robert D. Nowak. "Toeplitz Compressed Sensing Matrices With Applications to Sparse Channel Estimation". In: Transactions on Information Theory 56.11 (2010), pp. 5862-5875. DOI: 10.1109/TIT.2010.2070191 (cit. on p. 17).

[Hau+10] 贾维斯·D·豪普特（Jarvis D. Haupt）、瓦希德·乌兹·扎曼·巴杰瓦（Waheed Uz Zaman Bajwa）、吉尔·M·拉兹（Gil M. Raz）和罗伯特·D·诺瓦克（Robert D. Nowak）。“托普利茨压缩感知矩阵及其在稀疏信道估计中的应用”。见：《信息论汇刊》（Transactions on Information Theory）2010年第56卷第11期，第5862 - 5875页。DOI: 10.1109/TIT.2010.2070191（引自第17页）。

[HI00] Sariel Har-Peled and Piotr Indyk. "When Crossings Count - Approximating the Minimum Spanning Tree". In: Proceedings of the 16th Symposium on Computational Geometry. SoCG '00. ACM, 2000, pp. 166-175. DOI: 10.1145/336154.336197 (cit. on p.7

[HI00] 萨里尔·哈尔 - 佩莱德（Sariel Har-Peled）和彼得·因迪克（Piotr Indyk）。“当交叉点起作用时——近似最小生成树”。载于：第16届计算几何研讨会会议录。SoCG '00。美国计算机协会（ACM），2000年，第166 - 175页。DOI: 10.1145/336154.336197（见第7页引用）

[HIM12]

Sariel Har-Peled, Piotr Indyk, and Rajeev Motwani. "Approximate Nearest Neighbor: Towards Removing the Curse of Dimensionality". In: Theory of Computing 8.14 (2012), pp. 321-350. DOI: 10.4086/toc. 2012. v008a014. Previously published as [IM98; Har01] (cit. on pp. 5, 11, 12, 21).

萨里尔·哈尔 - 佩莱德（Sariel Har-Peled）、彼得·因迪克（Piotr Indyk）和拉杰夫·莫特瓦尼（Rajeev Motwani）。“近似最近邻：消除维度灾难的方向”。载于：《计算理论》（Theory of Computing）8.14（2012年），第321 - 350页。DOI: 10.4086/toc. 2012. v008a014。先前以[IM98; Har01]形式发表（见第5、11、12、21页引用）。

[HMM16] M16] Christina Heinze, Brian McWilliams, and Nicolai Meinshausen. "Dual-Loco: Distributing Statistical Estimation Using Random Projections". In: Proceedings of the 19th International Conference on Artificial Intelligence and Statistics (AISTATS '16). Vol. 51. Proceedings of Machine Learning Research (PMLR). PMLR, 2016, pp. 875-883 (cit. on p. 6).

[HMM16] M16] 克里斯蒂娜·海因策（Christina Heinze）、布莱恩·麦克威廉姆斯（Brian McWilliams）和尼科莱·迈恩绍森（Nicolai Meinshausen）。《双局部：使用随机投影进行统计估计的分布》。载于：第19届人工智能与统计国际会议论文集（AISTATS '16）。第51卷。机器学习研究会议录（PMLR）。机器学习研究会议录出版社（PMLR），2016年，第875 - 883页（见第6页引用）。

[HMT11] N. Halko, P. G. Martinsson, and Joel Aaron Tropp. "Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions". In: SIAM Review 53.2 (2011), pp. 217-288. DOI: 10.1137/090771806 (cit. on pp. 4.6).

[HMT11] N. 哈尔科（N. Halko）、P. G. 马丁松（P. G. Martinsson）和乔尔·亚伦·特罗普（Joel Aaron Tropp）。《利用随机性寻找结构：构建近似矩阵分解的概率算法》。载于：《工业与应用数学学会评论》（SIAM Review）53.2（2011年），第217 - 288页。DOI: 10.1137/090771806（见第4.6页引用）。

[Hot33] Harold Hotelling. "Analysis of a Complex of Statistical Variables into Principal Components". In: Journal of Educational Psychology 24.6 (1933), pp. 417-441. DOI: 10.1037/h0071325 (cit. on p.2).

[Hot33] 哈罗德·霍特林（Harold Hotelling）。《将一组复杂统计变量分解为主成分》。载于：《教育心理学杂志》（Journal of Educational Psychology）24.6（1933年），第417 - 441页。DOI: 10.1037/h0071325（见第2页引用）。

[HTB14] Reinhard Heckel, Michael Tschannen, and Helmut Bölcskei. "Subspace clustering of dimensionality-reduced data". In: Proceedings of the 47th International Symposium on Information Theory. ISIT '14. IEEE, 2014, pp. 2997-3001. DOI: 10.1109/ISIT. 2014. 6875384 (cit. on p. 34).

[HTB14] 莱因哈德·赫克尔（Reinhard Heckel）、迈克尔·察南（Michael Tschannen）和赫尔穆特·布尔克斯凯（Helmut Bölcskei）。“降维数据的子空间聚类”。见：第47届信息论国际研讨会论文集。ISIT '14。电气与电子工程师协会（IEEE），2014年，第2997 - 3001页。DOI: 10.1109/ISIT. 2014. 6875384（见第34页引用）。

[HTB17] Reinhard Heckel, Michael Tschannen, and Helmut Bölcskei. "Dimensionality-reduced subspace clustering". In: Information and Inference: A Journal of the IMA 6.3 (2017), pp. 246-283. DOI: 10.1093/imaiai/iaw021. Previously published as [HTB14] (cit. on p. 6).

[HTB17] 莱因哈德·赫克尔（Reinhard Heckel）、迈克尔·察南（Michael Tschannen）和赫尔穆特·布尔克斯凯（Helmut Bölcskei）。“降维子空间聚类”。见：《信息与推理：IMA期刊》（Information and Inference: A Journal of the IMA）第6卷第3期（2017年），第246 - 283页。DOI: 10.1093/imaiai/iaw021。之前以[HTB14]发表（见第6页引用）。

[HTF17] Trevor Hastie, Robert Tibshirani, and Jerome Friedman. "The Elements of Statistical Learning: Data Mining, Inference, and Prediction". In: 2nd ed. Springer Series in Statistics (SSS). 12th printing. Springer, 2017. Chap. 3.4. DOI: 10.1007/b94608 (cit. on p. 2).

[HTF17] 特雷弗·哈斯蒂（Trevor Hastie）、罗伯特·蒂布希拉尼（Robert Tibshirani）和杰罗姆·弗里德曼（Jerome Friedman）。《统计学习基础：数据挖掘、推理与预测》（The Elements of Statistical Learning: Data Mining, Inference, and Prediction）。见：第2版。施普林格统计学丛书（Springer Series in Statistics，SSS）。第12次印刷。施普林格出版社（Springer），2017年。第3.4章。DOI: 10.1007/b94608（引用见第2页）。

[HV11]

Aicke Hinrichs and Jan Vybíral. "Johnson-Lindenstrauss lemma for circulant matrices". In: Random Structures & Algorithms 39.3 (2011), pp. 391-398. DOI: 10, 1002/rsa.20360 (cit. on pp. 18, 21).

艾克·欣里克斯（Aicke Hinrichs）和扬·维比拉尔（Jan Vybíral）。《循环矩阵的约翰逊 - 林登施特劳斯引理》（Johnson-Lindenstrauss lemma for circulant matrices）。见：《随机结构与算法》（Random Structures & Algorithms）39.3（2011年），第391 - 398页。DOI: 10, 1002/rsa.20360（引用见第18、21页）。

[IM98]

Piotr Indyk and Rajeev Motwani. "Approximate Nearest Neighbors: Towards Removing the Curse of Dimensionality". In: Proceedings of the 30th Symposium on Theory of Computing. STOC '98. ACM, 1998, pp. 604-613. DOI: 10.1145/276698 276876 (cit. on pp. 11, 33).

彼得·因迪克（Piotr Indyk）和拉杰夫·莫特瓦尼（Rajeev Motwani）。《近似最近邻：消除维度灾难之路》（Approximate Nearest Neighbors: Towards Removing the Curse of Dimensionality）。见：第30届计算理论研讨会会议录。STOC '98。美国计算机协会（ACM），1998年，第604 - 613页。DOI: 10.1145/276698 276876（引用见第11、33页）。

[IN07]

Piotr Indyk and Assaf Naor. "Nearest-Neighbor-Preserving embeddings". In: Transactions on Algorithms 3.3 (2007), 31:1-31:12. DOI: 10.1145/1273340. 1273347 (cit. on p. 12).

皮奥特·因迪克（Piotr Indyk）和阿萨夫·诺尔（Assaf Naor）。《保持最近邻的嵌入》。载于《算法汇刊》（Transactions on Algorithms）第3卷第3期（2007年），第31:1 - 31:12页。DOI: 10.1145/1273340. 1273347（见第12页引用）。

[Ind01]

Piotr Indyk. "Algorithmic Applications of Low-Distortion Geometric Embeddings". In: Proceedings of the 42nd Symposium on Foundations of Computer Science. FOCS '01. IEEE, 2001, pp. 10-33. DOI: 10.1109/SFCS. 2001.959878 (cit. on p. 7).

皮奥特·因迪克（Piotr Indyk）。《低失真几何嵌入的算法应用》。载于《第42届计算机科学基础研讨会论文集》（Proceedings of the 42nd Symposium on Foundations of Computer Science）。FOCS '01。电气与电子工程师协会（IEEE），2001年，第10 - 33页。DOI: 10.1109/SFCS. 2001.959878（见第7页引用）。

[Jag19] Meena Jagadeesan. "Understanding Sparse JL for Feature Hashing". In: Advances in Neural Information Processing Systems 32. NeurIPS '19. Curran Associates, Inc., 2019, pp. 15203-15213 (cit. on p. 15).

[Jag19] 米娜·贾加迪桑（Meena Jagadeesan）。《理解用于特征哈希的稀疏约翰逊 - 林登施特劳斯引理》。载于《神经信息处理系统进展》（Advances in Neural Information Processing Systems）第32卷。NeurIPS '19。柯伦联合公司（Curran Associates, Inc.），2019年，第15203 - 15213页（见第15页引用）。

[Jai+20] Vishesh Jain, Natesh S. Pillai, Ashwin Sah, Mehtaab Sawhney, and Aaron Smith. "Fast and memory-optimal dimension reduction using Kac's walk". In: arXiv e-prints (2020). arXiv: 2003.10069 [cs.DS] (cit. on pp. 19-21).

[贾伊+20] 维什什·贾伊（Vishesh Jain）、纳泰什·S·皮莱（Natesh S. Pillai）、阿什温·萨赫（Ashwin Sah）、梅塔布·索哈尼（Mehtaab Sawhney）和亚伦·史密斯（Aaron Smith）。“利用卡茨游走实现快速且内存最优的降维”。载于：arXiv预印本（2020年）。arXiv: 2003.10069 [计算机科学.数据结构]（见第19 - 21页）。

[Jam+13a] Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani. "An Introduction to Statistical Learning". In: 1st ed. Springer Texts in Statistics (STS). 7th printing. Springer, 2013. Chap. 6. DOI: 10.1007/978-1-4614-7138-7 (cit. on p. 2).

[詹姆斯+13a] 加雷斯·詹姆斯（Gareth James）、丹妮拉·维滕（Daniela Witten）、特雷弗·哈斯蒂（Trevor Hastie）和罗伯特·蒂布希拉尼（Robert Tibshirani）。“统计学习导论”。载于：第1版。施普林格统计学教材（Springer Texts in Statistics，STS）。第7次印刷。施普林格出版社，2013年。第6章。DOI: 10.1007/978 - 1 - 4614 - 7138 - 7（见第2页）。

[Jam+13b] Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani. "An Introduction to Statistical Learning". In: 1st ed. Springer Texts in Statistics (STS). 7th printing. Springer, 2013. Chap. 10. DOI: 10.1007/978-1-4614-7138-7 (cit. on p. 2).

[Jam+13b] 加雷斯·詹姆斯（Gareth James）、丹妮拉·维滕（Daniela Witten）、特雷弗·哈斯蒂（Trevor Hastie）和罗伯特·蒂布希拉尼（Robert Tibshirani）。《统计学习导论》（"An Introduction to Statistical Learning"）。见：第1版。施普林格统计学教材（Springer Texts in Statistics，STS）。第7次印刷。施普林格出版社（Springer），2013年。第10章。DOI: 10.1007/978-1-4614-7138-7（引自第2页）。

[Jia+20] Haotian Jiang, Yin Tat Lee, Zhao Song, and Sam Chiu-wai Wong. "An Improved Cutting Plane Method for Convex Optimization, Convex-Concave Games and its Applications". In: Proceedings of the 52nd Symposium on Theory of Computing. STOC '20. ACM, 2020, pp. 944-953. DOI: 10.1145/3357713. 3384284 (cit. on p. 6).

[Jia+20] 江浩天（Haotian Jiang）、李尹达（Yin Tat Lee）、宋钊（Zhao Song）和黄兆辉（Sam Chiu-wai Wong）。《凸优化、凸 - 凹博弈的改进割平面法及其应用》（"An Improved Cutting Plane Method for Convex Optimization, Convex-Concave Games and its Applications"）。见：第52届计算理论研讨会论文集。STOC '20。美国计算机协会（ACM），2020年，第944 - 953页。DOI: 10.1145/3357713. 3384284（引自第6页）。

[JL84] William B Johnson and Joram Lindenstrauss. "Extensions of Lipschitz mappings into a Hilbert space". In: Proceedings of the 1982 Conference in Modern Analysis and Probability. Vol. 26. Contemporary Mathematics. AMS, 1984, pp. 189-206. DOI: 10.1090/conm/026/737400 (cit. on pp. 2,3,11,21).

[JL84] 威廉·B·约翰逊（William B Johnson）和约拉姆·林登施特劳斯（Joram Lindenstrauss）。《利普希茨映射到希尔伯特空间的扩展》。载于：《1982年现代分析与概率论会议论文集》。第26卷。《当代数学》。美国数学学会（AMS），1984年，第189 - 206页。DOI: 10.1090/conm/026/737400（引用见第2、3、11、21页）。

[Jol02] Ian T. Jolliffe. Principal Component Analysis. 2nd ed. Springer Series in Statistics (SSS). Springer, 2002. DOI: 10.1007/b98835 (cit. on pp. 3, 4).

[Jol02] 伊恩·T·乔利夫（Ian T. Jolliffe）。《主成分分析》。第2版。施普林格统计学丛书（Springer Series in Statistics，SSS）。施普林格出版社（Springer），2002年。DOI: 10.1007/b98835（引用见第3、4页）。

[JS08] Richard Jensen and Qiang Shen. Computational Intelligence and Feature Selection - Rough and Fuzzy Approaches. IEEE Press series on computational intelligence. IEEE, 2008. DOI: 10.1002/9780470377888 (cit. on p. 2).

[JS08] 理查德·詹森（Richard Jensen）和沈强（Qiang Shen）。《计算智能与特征选择——粗糙与模糊方法》。电气与电子工程师协会出版社（IEEE Press）计算智能系列。电气与电子工程师协会（IEEE），2008年。DOI: 10.1002/9780470377888（引用见第2页）。

[JW11] T. S. Jayram and David P. Woodruff. "Optimal Bounds for Johnson-Lindenstrauss Transforms and Streaming Problems with Sub-Constant Error". In: Proceedings of the 22nd Symposium on Discrete Algorithms. SODA '11. SIAM, 2011, pp. 1-10. DOI: 10.1137/1.9781611973082.1 (cit. on p. 35).

[JW11] T. S. 杰拉姆（T. S. Jayram）和大卫·P·伍德拉夫（David P. Woodruff）。“约翰逊 - 林登施特劳斯变换（Johnson-Lindenstrauss Transforms）和具有次常数误差的流数据问题的最优界”。收录于：第22届离散算法研讨会论文集。SODA '11。工业与应用数学学会（SIAM），2011年，第1 - 10页。DOI: 10.1137/1.9781611973082.1（见第35页引用）。

[JW13] T. S. Jayram and David P. Woodruff. "Optimal Bounds for Johnson-Lindenstrauss Transforms and Streaming Problems with Subconstant Error". In: Transactions on Algorithms 9.3 (2013), 26:1-26:17. DOI: 10.1145/2483699.2483706. Previously published as [JW11] (cit. on p. 4).

[JW13] T. S. 杰拉姆（T. S. Jayram）和大卫·P·伍德拉夫（David P. Woodruff）。“约翰逊 - 林登施特劳斯变换（Johnson-Lindenstrauss Transforms）和具有次常数误差的流数据问题的最优界”。收录于：《算法汇刊》（Transactions on Algorithms）第9卷第3期（2013年），26:1 - 26:17。DOI: 10.1145/2483699.2483706。之前以[JW11]发表（见第4页引用）。

[Kab14]

Ata Kabán. "New Bounds on Compressive Linear Least Squares Regression". In: Proceedings of the Seventeenth International Conference on Artificial Intelligence and Statistics, (AISTATS '14). Vol. 33. Proceedings of Machine Learning Research (PMLR). PMLR, 2014, pp. 448-456 (cit. on p. 6).

阿塔·卡班（Ata Kabán）。《压缩线性最小二乘回归的新边界》。载于：《第十七届人工智能与统计国际会议论文集》（AISTATS '14）。第33卷。机器学习研究会议录（PMLR）。机器学习研究会议录出版社，2014年，第448 - 456页（引用见第6页）。

[Kac56]

Mark Kac. "Foundations of Kinetic Theory". In: Proceedings of the 3rd Berkeley Symposium on Mathematical Statistics and Probability. Vol. 3. University of California Press, 1956, pp. 171-197 (cit. on p. 19).

马克·卡茨（Mark Kac）。《动力学理论基础》。载于：《第三届伯克利数学统计与概率研讨会论文集》。第3卷。加利福尼亚大学出版社，1956年，第171 - 197页（引用见第19页）。

[Kas98]

Samuel Kaski. "Dimensionality Reduction by Random Mapping: Fast Similarity Computation for Clustering". In: Proceedings of the 8th International Joint Conference on Neural Networks. Vol. 1. IJCNN '98. IEEE, 1998, pp. 413-418. DOI: 10.1109/IJCNN. 1998.682302 (cit. on p. 6).

塞缪尔·卡斯基（Samuel Kaski）。《通过随机映射进行降维：用于聚类的快速相似度计算》。载于：《第八届国际神经网络联合会议论文集》。第1卷。IJCNN '98。电气与电子工程师协会（IEEE），1998年，第413 - 418页。DOI: 10.1109/IJCNN. 1998.682302（引用见第6页）。

[Ken+13] Krishnaram Kenthapadi, Aleksandra Korolova, Ilya Mironov, and Nina Mishra. "Privacy via the Johnson-Lindenstrauss Transform". In: Journal of Privacy and Confidentiality 5.1 (2013). DOI: 10.29@12/jpc.v5i1.625 (cit. on p. 7).

[Ken+13] 克里什纳拉姆·肯塔帕迪（Krishnaram Kenthapadi）、亚历山德拉·科罗洛娃（Aleksandra Korolova）、伊利亚·米罗诺夫（Ilya Mironov）和妮娜·米什拉（Nina Mishra）。《通过约翰逊 - 林登施特劳斯变换实现隐私保护》。载于《隐私与保密期刊》（Journal of Privacy and Confidentiality）第5卷第1期（2013年）。DOI: 10.29@12/jpc.v5i1.625（见第7页引用）。

[KJ16] Shiva Prasad Kasiviswanathan and Hongxia Jin. "Efficient Private Empirical Risk Minimization for High-dimensional Learning". In: Proceedings of the 33st International Conference on Machine Learning (ICML '16). Vol. 48. Proceedings of Machine Learning Research (PMLR). PMLR, 2016, pp. 488-497 (cit. on p. 7).

[KJ16] 希瓦·普拉萨德·卡西维斯瓦纳坦（Shiva Prasad Kasiviswanathan）和金红霞（Hongxia Jin）。《高维学习中的高效隐私经验风险最小化》。载于《第33届国际机器学习会议论文集》（Proceedings of the 33st International Conference on Machine Learning (ICML '16)）。第48卷。机器学习研究会议录（Proceedings of Machine Learning Research (PMLR)）。机器学习研究会议录出版社（PMLR），2016年，第488 - 497页（见第7页引用）。

[Kle97]

Jon M. Kleinberg. "Two Algorithms for Nearest-Neighbor Search in High Dimensions". In: Proceedings of the 29th Symposium on the Theory of Computing. STOC '97. ACM, 1997, pp. 599-608. DOI: 10.1145/258533. 258653 (cit. on p. 5).

乔恩·M·克莱因伯格（Jon M. Kleinberg）。《高维最近邻搜索的两种算法》。载于《第29届计算理论研讨会论文集》（Proceedings of the 29th Symposium on the Theory of Computing）。STOC '97。美国计算机协会（ACM），1997年，第599 - 608页。DOI: 10.1145/258533. 258653（见第5页引用）。

[KM05] Bo'Az B. Klartag and Shahar Mendelson. "Empirical processes and random projections". In: Journal of Functional Analysis 225.1 (2005), pp. 229-245. DOI: 10.1016/j jfa.2004.10.009 (cit. on pp.4.12).

[KM05] 博阿兹·B·克拉尔塔格（Bo'Az B. Klartag）和沙哈尔·门德尔松（Shahar Mendelson）。《经验过程与随机投影》。载于《泛函分析杂志》（Journal of Functional Analysis）225.1（2005年），第229 - 245页。DOI: 10.1016/j jfa.2004.10.009（见第4、12页引用）。

[KMN11] Daniel M. Kane, Raghu Meka, and Jelani Nelson. "Almost Optimal Explicit Johnson-Lindenstrauss Families". In: Proceedings of the 14th International Workshop on Approximation Algorithms for Combinatorial Optimization Problems (APPROX '11) and the 15th International Workshop on Randomization and Computation (RANDOM '11). Vol. 6845. Lecture Notes in Computer Science (LNCS). Springer, 2011, pp. 628-639. DOI: 10.1007/978-3-642-22935-0_53 (cit. on p. 4).

[KMN11] 丹尼尔·M·凯恩（Daniel M. Kane）、拉古·梅卡（Raghu Meka）和杰拉尼·纳尔逊（Jelani Nelson）。《近乎最优的显式约翰逊 - 林登施特劳斯族》。载于《第14届组合优化问题近似算法国际研讨会（APPROX '11）和第15届随机化与计算国际研讨会（RANDOM '11）论文集》。第6845卷。《计算机科学讲义》（Lecture Notes in Computer Science，LNCS）。施普林格出版社（Springer），2011年，第628 - 639页。DOI: 10.1007/978 - 3 - 642 - 22935 - 0_53（见第4页引用）。

[KN10] Daniel M. Kane and Jelani Nelson. "A Derandomized Sparse Johnson-Lindenstrauss Transform". In: arXiv e-prints (2010). arXiv: 1006.3585 [cs.DS] (cit. on p. 13).

[KN10] 丹尼尔·M·凯恩（Daniel M. Kane）和杰拉尼·纳尔逊（Jelani Nelson）。《一种去随机化的稀疏约翰逊 - 林登施特劳斯变换》。载于：arXiv预印本（2010年）。arXiv: 1006.3585 [计算机科学 - 数据结构]（见第13页引用）。

[KN12] Daniel M. Kane and Jelani Nelson. "Sparser Johnson-Lindenstrauss Transforms". In: Proceedings of the 23rd Symposium on Discrete Algorithms. SODA '12. SIAM, 2012, pp. 1195-1206. DOI: 10.1137/1.9781611973099. 94 (cit. on p. 36).

[KN12] 丹尼尔·M·凯恩（Daniel M. Kane）和杰拉尼·纳尔逊（Jelani Nelson）。《更稀疏的约翰逊 - 林登施特劳斯变换》。载于：第23届离散算法研讨会论文集。SODA '12。工业与应用数学学会（SIAM），2012年，第1195 - 1206页。DOI: 10.1137/1.9781611973099. 94（见第36页引用）。

[KN14] Daniel M. Kane and Jelani Nelson. "Sparser Johnson-Lindenstrauss Transforms". In: Journal of the ACM 61.1 (2014), 4:1-4:23. DOI: 10.1145/2559902. Previously published as [KN12] (cit. on pp. 13, 14, 21).

[KN14] 丹尼尔·M·凯恩（Daniel M. Kane）和杰拉尼·纳尔逊（Jelani Nelson）。《更稀疏的约翰逊 - 林登施特劳斯变换》。载于：《美国计算机协会期刊》61.1（2014年），4:1 - 4:23。DOI: 10.1145/2559902。之前以[KN12]发表（见第13、14、21页引用）。

[KOR00]

Eyal Kushilevitz, Rafail Ostrovsky, and Yuval Rabani. "Efficient Search for Approximate Nearest Neighbor in High Dimensional Spaces". In: Journal on Computing 30.2 (2000), pp. 457-474. DOI: 10.1137/S0097539798347177. Previously published as [KOR98] (cit. on p. 5).

埃亚尔·库什列维茨（Eyal Kushilevitz）、拉斐尔·奥斯特罗夫斯基（Rafail Ostrovsky）和尤瓦尔·拉巴尼（Yuval Rabani）。《高维空间中近似最近邻的高效搜索》。载于《计算期刊》（Journal on Computing）第30卷第2期（2000年），第457 - 474页。DOI: 10.1137/S0097539798347177。此前以[KOR98]形式发表（见第5页引用）。

[KOR98]

Eyal Kushilevitz, Rafail Ostrovsky, and Yuval Rabani. "Efficient Search for Approximate Nearest Neighbor in High Dimensional Spaces". In: Proceedings of the 30th Symposium on the Theory of Computing. STOC '98. ACM, 1998, pp. 614-623. DOI: 10.1145/276698.276877 (cit. on p. 36).

埃亚尔·库什列维茨（Eyal Kushilevitz）、拉斐尔·奥斯特罗夫斯基（Rafail Ostrovsky）和尤瓦尔·拉巴尼（Yuval Rabani）。《高维空间中近似最近邻的高效搜索》。载于《第30届计算理论研讨会会议录》（Proceedings of the 30th Symposium on the Theory of Computing）。STOC '98。美国计算机协会（ACM），1998年，第614 - 623页。DOI: 10.1145/276698.276877（见第36页引用）。

[KW11]

Felix Krahmer and Rachel Ward. "New and improved Johnson-Lindenstrauss embeddings via the Restricted Isometry Property". In: Journal on Mathematical Analysis 43.3 (2011), pp. 1269-1281. DOI: 10.1137/100810447 (cit. on pp. 17, 18.21).

费利克斯·克拉默（Felix Krahmer）和雷切尔·沃德（Rachel Ward）。《通过受限等距性质实现新的改进的约翰逊 - 林登施特劳斯嵌入》。载于《数学分析期刊》（Journal on Mathematical Analysis）第43卷第3期（2011年），第1269 - 1281页。DOI: 10.1137/100810447（见第17、18、21页引用）。

[KY20] SeongYoon Kim and SeYoung Yun. "Accelerating Randomly Projected Gradient with Variance Reduction". In: Proceedings of the 7th International Conference on Big Data and Smart Computing. BigComp '20. IEEE, 2020, pp. 531-534. DOI: 10.1109/ BigComp48618.2020.00-11 (cit. on p. 6).

[KY20] 金成允（SeongYoon Kim）和尹世英（SeYoung Yun）。“通过方差缩减加速随机投影梯度”。载于：第七届大数据与智能计算国际会议论文集。BigComp '20。电气与电子工程师协会（IEEE），2020年，第531 - 534页。DOI: 10.1109/ BigComp48618.2020.00 - 11（引自第6页）。

[LAS08]

Edo Liberty, Nir Ailon, and Amit Singer. "Dense Fast Random Projections and Lean Walsh Transforms". In: Proceedings of the 11th International Workshop on Approximation Algorithms for Combinatorial Optimization Problems (APPROX '08) and the 12th International Workshop on Randomization and Computation (RANDOM '08). Vol. 5171. Lecture Notes in Computer Science (LNCS). Springer, 2008, pp. 512-522. DOI: 10.1007/978-3-540-85363-3_40 (cit. on p. 37).

埃多·利伯蒂（Edo Liberty）、尼尔·艾隆（Nir Ailon）和阿米特·辛格（Amit Singer）。“密集快速随机投影与精简沃尔什变换”。载于：第十一届组合优化问题近似算法国际研讨会（APPROX '08）和第十二届随机化与计算国际研讨会（RANDOM '08）论文集。第5171卷。计算机科学讲义（LNCS）。施普林格出版社（Springer），2008年，第512 - 522页。DOI: 10.1007/978 - 3 - 540 - 85363 - 3_40（引自第37页）。

[LAS11]

Edo Liberty, Nir Ailon, and Amit Singer. "Dense Fast Random Projections and Lean Walsh Transforms". In: Discrete & Computational Geometry 45.1 (2011), pp. 34-44. DOI: 10.1007/s00454-010-9309-5. Previously published as [LAS08] (cit. on pp. 18, 19, 21).

埃多·利伯蒂（Edo Liberty）、尼尔·艾隆（Nir Ailon）和阿米特·辛格（Amit Singer）。《密集快速随机投影与精简沃尔什变换》。载于《离散与计算几何》（Discrete & Computational Geometry）第45卷第1期（2011年），第34 - 44页。DOI: 10.1007/s00454-010-9309-5。此前以[LAS08]发表（见第18、19、21页引用）。

[Le 14]

François Le Gall. "Powers of Tensors and Fast Matrix Multiplication". In: Proceedings of the 39th International Symposium on Symbolic and Algebraic Computation. ISSAC '14. ACM, 2014, pp. 296-303. DOI: 10.1145/2608628.2608664 (cit. on p. 4).

弗朗索瓦·勒加尔（François Le Gall）。《张量的幂与快速矩阵乘法》。载于《第39届国际符号与代数计算研讨会论文集》（Proceedings of the 39th International Symposium on Symbolic and Algebraic Computation）。ISSAC '14。美国计算机协会（ACM），2014年，第296 - 303页。DOI: 10.1145/2608628.2608664（见第4页引用）。

[Li+20]

Jie Li, Rongrong Ji, Hong Liu, Jianzhuang Liu, Bineng Zhong, Cheng Deng, and Qi Tian. "Projection & Probability-Driven Black-Box Attack". In: arXiv e-prints (2020). arXiv: 2005.03837 [cs.CV] (cit. on p. 6).

李杰、计荣荣、刘宏、刘建庄、钟必能、邓程和齐天。《投影与概率驱动的黑盒攻击》。载于arXiv预印本（2020年）。arXiv: 2005.03837 [计算机视觉（cs.CV）]（见第6页引用）。

[Liu+17]

Wenfen Liu, Mao Ye, Jianghong Wei, and Xuexian Hu. "Fast Constrained Spectral Clustering and Cluster Ensemble with Random Projection". In: Computational Intelligence and Neuroscience 2017 (2017). DOI: 10.1155/2017/2658707 (cit. on p. 6).

刘文芬（Wenfen Liu）、叶茂（Mao Ye）、魏江洪（Jianghong Wei）和胡学先（Xuexian Hu）。《基于随机投影的快速约束谱聚类与聚类集成》。载于《计算智能与神经科学》2017年（2017年）。DOI: 10.1155/2017/2658707（见第6页引用）。

[LKR06]

Kun Liu, Hillol Kargupta, and Jessica Ryan. "Random Projection-Based Multiplicative Data Perturbation for Privacy Preserving Distributed Data Mining". In: Transactions on Knowledge and Data Engineering 18.1 (2006), pp. 92-106. DOI: 10.1109/TKDE.2006.14 (cit. on p. 7).

刘坤（Kun Liu）、希洛尔·卡尔古普塔（Hillol Kargupta）和杰西卡·瑞安（Jessica Ryan）。《基于随机投影的乘法数据扰动用于隐私保护的分布式数据挖掘》。载于《知识与数据工程汇刊》第18卷第1期（2006年），第92 - 106页。DOI: 10.1109/TKDE.2006.14（见第7页引用）。

[LL20]

Zijian Lei and Liang Lan. "Improved Subsampled Randomized Hadamard Transform for Linear SVM". In: Proceedings of the 34th AAAI Conference on Artificial Intelligence (AAAI '20) and the 32nd Conference on Innovative Applications of Artificial Intelligence (IAAI '20) and the 10th Symposium on Educational Advances in Artificial Intelligence (EAAI '20). Vol. 34. 4. AAAI Press, 2020, pp. 4519-4526. DOI: 10.1609/aaai.v34i04.5880(cit.onpp.6,17).

雷子健（Zijian Lei）和兰亮（Liang Lan）。《用于线性支持向量机的改进子采样随机哈达玛变换》。收录于：第34届人工智能协会会议（AAAI '20）、第32届人工智能创新应用会议（IAAI '20）和第10届人工智能教育进展研讨会（EAAI '20）论文集。第34卷，第4期。美国人工智能协会出版社（AAAI Press），2020年，第4519 - 4526页。DOI: 10.1609/aaai.v34i04.5880（引用见第6、17页）。

[Llo82]

Stuart P. Lloyd. "Least squares quantization in PCM". In: Transactions on Information Theory 28.2 (1982), pp. 129-137. DOI: 10.1109/TIT. 1982.1056489 (cit. on p. 8).

斯图尔特·P·劳埃德（Stuart P. Lloyd）。《脉冲编码调制中的最小二乘量化》。收录于：《信息论汇刊》28卷2期（1982年），第129 - 137页。DOI: 10.1109/TIT. 1982.1056489（引用见第8页）。

[LLS07]

John Langford, Lihong Li, and Alexander L. Strehl. Vowpal Wabbit Code Release. 2007. URL: https://hunch.net/?p=309 (visited on 16/06/2020) (cit. on pp. 13, 14).

约翰·兰福德（John Langford）、李立宏（Lihong Li）和亚历山大·L·施特尔（Alexander L. Strehl）。Vowpal Wabbit代码发布。2007年。网址：https://hunch.net/?p=309（2020年6月16日访问）（引用见第13、14页）。

[LN17]

Kasper Green Larsen and Jelani Nelson. "Optimality of the Johnson-Lindenstrauss Lemma". In: Proceedings of the 58th Symposium on Foundations of Computer Science. FOCS '17. IEEE, 2017, pp. 633-638. DOI: 10.1109/FOCS. 2017.64 (cit. on p. 4).

卡斯珀·格林·拉森（Kasper Green Larsen）和杰拉尼·纳尔逊（Jelani Nelson）。《约翰逊 - 林登施特劳斯引理的最优性》。载于：第58届计算机科学基础研讨会会议录。FOCS '17。电气与电子工程师协会（IEEE），2017年，第633 - 638页。DOI: 10.1109/FOCS. 2017.64（见第4页引用）。

[LR83] Grazia Lotti and Francesco Romani. "On the asymptotic complexity of rectangular matrix multiplication". In: Theoretical Computer Science 23.2 (1983), pp. 171-185. DOI: 10.1016/0304-3975(83)90054-3 (cit. on p. 18).

[LR83] 格拉齐亚·洛蒂（Grazia Lotti）和弗朗切斯科·罗马尼（Francesco Romani）。《矩形矩阵乘法的渐近复杂度》。载于：《理论计算机科学》23.2（1983年），第171 - 185页。DOI: 10.1016/0304 - 3975(83)90054 - 3（见第18页引用）。

[LRU20a] Jure Leskovec, Anand Rajaraman, and Jeffrey David Ullman. "Mining of Massive Datasets". In: 3rd ed. Cambridge University Press, 2020. Chap. 1. ISBN: 978-1-108- 47634-8 (cit. on p. 2).

[LRU20a] 尤雷·莱斯科维奇（Jure Leskovec）、阿南德·拉贾拉曼（Anand Rajaraman）和杰弗里·大卫·厄尔曼（Jeffrey David Ullman）。《大规模数据集挖掘》。第3版。剑桥大学出版社（Cambridge University Press），2020年。第1章。ISBN: 978 - 1 - 108 - 47634 - 8（见第2页引用）。

[LRU20b] Jure Leskovec, Anand Rajaraman, and Jeffrey David Ullman. "Mining of Massive Datasets". In: 3rd ed. Cambridge University Press, 2020. Chap. 11. ISBN: 978-1-108- 47634-8 (cit. on p. 2).

[LRU20b] 尤雷·莱斯克夫塞克（Jure Leskovec）、阿南德·拉贾拉曼（Anand Rajaraman）和杰弗里·大卫·厄尔曼（Jeffrey David Ullman）。《大规模数据集挖掘》（“Mining of Massive Datasets”）。见：第3版。剑桥大学出版社（Cambridge University Press），2020年。第11章。国际标准书号：978 - 1 - 108 - 47634 - 8（见第2页引用）。

[Mah11] Michael W. Mahoney. "Randomized Algorithms for Matrices and Data". In: Foundations and Trends® in Machine Learning 3.2 (2011), pp. 123-224. DOI: 10.1561/ 2200000035 (cit. on p. 6).

[Mah11] 迈克尔·W·马奥尼（Michael W. Mahoney）。《矩阵和数据的随机算法》（“Randomized Algorithms for Matrices and Data”）。见：《机器学习的基础与趋势》（Foundations and Trends® in Machine Learning）3.2（2011年），第123 - 224页。数字对象标识符：10.1561 / 2200000035（见第6页引用）。

[Mat08] Jiří Matoušek. "On variants of the Johnson-Lindenstrauss lemma". In: Random Structures & Algorithms 33.2 (2008), pp. 142-156. DOI: 10.1002/rsa.20218 (cit. on pp. ${12},{14},{16})$ .

[Mat08] 吉里·马托谢克（Jiří Matoušek）。《关于约翰逊 - 林登施特劳斯引理的变体》（“On variants of the Johnson - Lindenstrauss lemma”）。见：《随机结构与算法》（Random Structures & Algorithms）33.2（2008年），第142 - 156页。数字对象标识符：10.1002/rsa.20218（见${12},{14},{16})$页引用）。

[MFL08] Mala Murthy, Ila Fiete, and Gilles Laurent. "Testing Odor Response Stereotypy in the Drosophila Mushroom Body". In: Neuron 59.6 (2008), pp. 1009-1023. DOI: 10.1016/j.neuron.2008.07.040(cit.on p.7).

[MFL08] 马拉·穆尔蒂（Mala Murthy）、伊拉·菲耶特（Ila Fiete）和吉勒斯·洛朗（Gilles Laurent）。《测试果蝇蘑菇体中的气味反应刻板性》。载于《神经元》（Neuron）第59卷第6期（2008年），第1009 - 1023页。DOI: 10.1016/j.neuron.2008.07.040（引自第7页）。

[MM09] Odalric-Ambrym Maillard and Rémi Munos. "Compressed Least-Squares Regression". In: Advances in Neural Information Processing Systems 22. NIPS '09. Curran Associates, Inc., 2009, pp. 1213-1221 (cit. on p. 6).

[MM09] 奥达利克 - 安布里姆·梅拉德（Odalric - Ambrym Maillard）和雷米·穆诺斯（Rémi Munos）。《压缩最小二乘回归》。载于《神经信息处理系统进展22》（Advances in Neural Information Processing Systems 22）。NIPS '09。柯伦联合公司（Curran Associates, Inc.），2009年，第1213 - 1221页（引自第6页）。

[MM13] Xiangrui Meng and Michael W. Mahoney. "Low-Distortion Subspace Embeddings in Input-Sparsity Time and Applications to Robust Linear Regression". In: Proceedings of the 45th Symposium on Theory of Computing. STOC '13. ACM, 2013, pp. 91-100. DOI: 10.1145/2488608.2488621 (cit. on p. 6).

[MM13] 孟祥瑞（Xiangrui Meng）和迈克尔·W·马奥尼（Michael W. Mahoney）。《输入稀疏时间下的低失真子空间嵌入及其在稳健线性回归中的应用》。载于《第45届计算理论研讨会会议录》（Proceedings of the 45th Symposium on Theory of Computing）。STOC '13。美国计算机协会（ACM），2013年，第91 - 100页。DOI: 10.1145/2488608.2488621（引自第6页）。

[MM20]

Cameron Musco and Christopher Musco. "Projection-Cost-Preserving Sketches: Proof Strategies and Constructions". In: arXiv e-prints (2020). arXiv: 2004.08434 [cs.DS]. Previously published as [Coh+15; CMM17] (cit. on p. 6).

卡梅隆·马斯科（Cameron Musco）和克里斯托弗·马斯科（Christopher Musco）。《保留投影成本的草图：证明策略与构造》。载于：arXiv预印本（2020年）。arXiv: 2004.08434 [计算机科学 - 数据结构]。此前曾以[Coh+15; CMM17]形式发表（见第6页引用）。

[MMR19] Konstantin Makarychev, Yury Makarychev, and Ilya Razenshteyn. "Performance of Johnson-Lindenstrauss Transform for $k$ -Means and $k$ -Medians Clustering". In: Proceedings of the 51st Symposium on Theory of Computing. STOC '19. ACM, 2019, pp. 1027-1038. DOI: 10.1145/3313276.3316350 (cit. on p. 9).

[MMR19] 康斯坦丁·马卡里切夫（Konstantin Makarychev）、尤里·马卡里切夫（Yury Makarychev）和伊利亚·拉赞施泰因（Ilya Razenshteyn）。《约翰逊 - 林登施特劳斯变换在$k$ - 均值和$k$ - 中位数聚类中的性能》。载于：第51届计算理论研讨会会议录。STOC '19。美国计算机协会（ACM），2019年，第1027 - 1038页。DOI: 10.1145/3313276.3316350（见第9页引用）。

[MS77]

Florence Jessie MacWilliams and Neil James Alexander Sloane. The Theory of Error-Correcting Codes. Vol. 16. North-Holland Mathematical Library. Elsevier, 1977. ISBN: 978-0-444-85193-2 (cit. on p. 16).

弗洛伦斯·杰西·麦克威廉姆斯（Florence Jessie MacWilliams）和尼尔·詹姆斯·亚历山大·斯隆（Neil James Alexander Sloane）。《纠错码理论》。第16卷。北荷兰数学图书馆。爱思唯尔（Elsevier），1977年。ISBN: 978 - 0 - 444 - 85193 - 2（见第16页引用）。

[Mut05]

S. Muthukrishnan. "Data Streams: Algorithms and Applications". In: Foundations and Trends® in Theoretical Computer Science 1.2 (2005), pp. 117-236. DOI: 10.1561/ 0400000002 (cit. on p. 11).

 S. 穆图克里什南（S. Muthukrishnan）。《数据流：算法与应用》。载于《理论计算机科学基础与趋势》（Foundations and Trends® in Theoretical Computer Science）第1卷第2期（2005年），第117 - 236页。DOI: 10.1561/ 0400000002（见第11页引用）。

[NC20] Paula Navarro-Esteban and Juan Antonio Cuesta-Albertos. "High-dimensional outlier detection using random projections". In: arXiv e-prints (2020). arXiv: 2005. 08923 [stat.ME] (cit. on p.6).

[NC20] 葆拉·纳瓦罗 - 埃斯特万（Paula Navarro - Esteban）和胡安·安东尼奥·库埃斯塔 - 阿尔韦托斯（Juan Antonio Cuesta - Albertos）。《使用随机投影进行高维离群点检测》。载于arXiv预印本（2020年）。arXiv: 2005. 08923 [统计方法（stat.ME）]（见第6页引用）。

[Nel11] Jelani Nelson. "Sketching and Streaming High-Dimensional Vectors". PhD thesis. Cambridge, MA: Massachusetts Institute of Technology, 2011 (cit. on p. 11).

[Nel11] 杰拉尼·纳尔逊（Jelani Nelson）。《高维向量的草图绘制与流式处理》。博士学位论文。马萨诸塞州剑桥市：麻省理工学院（Massachusetts Institute of Technology），2011年（见第11页引用）。

[Ngu+16] u+16] Xuan Vinh Nguyen, Sarah M. Erfani, Sakrapee Paisitkriangkrai, James Bailey, Christopher Leckie, and Kotagiri Ramamohanarao. "Training robust models using Random Projection". In: Proceedings of the 23rd International Conference on Pattern Recognition. ICPR '16. IEEE, 2016, pp. 531-536. DOI: 10.1109/ICPR. 2016.7899688 (cit. on p. 6).

[Ngu+16] 阮春荣（Xuan Vinh Nguyen）、莎拉·M·埃尔法尼（Sarah M. Erfani）、萨克拉皮·派西特克良克莱（Sakrapee Paisitkriangkrai）、詹姆斯·贝利（James Bailey）、克里斯托弗·莱基（Christopher Leckie）和科塔吉里·拉马莫哈拉奥（Kotagiri Ramamohanarao）。“使用随机投影训练鲁棒模型”。见：第23届国际模式识别会议论文集。ICPR '16。电气与电子工程师协会（IEEE），2016年，第531 - 536页。DOI: 10.1109/ICPR. 2016.7899688（引自第6页）。

[Ngu09] Tuan S. Nguyen. "Dimension Reduction Methods with Applications to High Dimensional Data with a Censored Response". PhD thesis. Houston, TX: Rice University, 2009 (cit. on p. 12).

[Ngu09] 阮团（Tuan S. Nguyen）。“应用于带删失响应的高维数据的降维方法”。博士学位论文。得克萨斯州休斯顿：莱斯大学（Rice University），2009年（引自第12页）。

[Niy+20] D] Lama B. Niyazi, Abla Kammoun, Hayssam Dahrouj, Mohamed-Slim Alouini, and Tareq Y. Al-Naffouri. "Asymptotic Analysis of an Ensemble of Randomly Projected Linear Discriminants". In: arXiv e-prints (2020). arXiv: 2004.08217 [stat.ML] (cit. on p. 6).

[尼亚齐+20] 喇嘛·B·尼亚齐（Lama B. Niyazi）、阿布拉·卡蒙（Abla Kammoun）、海萨姆·达鲁杰（Hayssam Dahrouj）、穆罕默德 - 斯利姆·阿卢伊尼（Mohamed-Slim Alouini）和塔里克·Y·纳富里（Tareq Y. Al-Naffouri）。“随机投影线性判别式集合的渐近分析”。载于：arXiv预印本（2020年）。arXiv: 2004.08217 [统计机器学习]（引自第6页）。

[NN13a] Jelani Nelson and Huy Lê Nguyên. "OSNAP: Faster Numerical Linear Algebra Algorithms via Sparser Subspace Embeddings". In: Proceedings of the 54th Symposium on Foundations of Computer Science. FOCS '13. IEEE, 2013, pp. 117-126. DOI: 10.1109/ FOCS. 2013.21 (cit. on p. 6).

[纳尔逊和阮13a] 杰拉尼·纳尔逊（Jelani Nelson）和胡伊·勒·阮（Huy Lê Nguyên）。“OSNAP：通过更稀疏子空间嵌入实现更快的数值线性代数算法”。载于：第54届计算机科学基础研讨会会议录。FOCS '13。电气与电子工程师协会（IEEE），2013年，第117 - 126页。DOI: 10.1109/ FOCS. 2013.21（引自第6页）。

[NN13b] Jelani Nelson and Huy Lê Nguyěn. "Sparsity Lower Bounds for Dimensionality Reducing Maps". In: Proceedings of the 45th Symposium on Theory of Computing. STOC '13. ACM, 2013, pp. 101-110. DOI: 10.1145/2488608. 2488622 (cit. on p. 13).

[NN13b] 杰拉尼·尼尔森（Jelani Nelson）和胡·黎·阮（Huy Lê Nguyěn）。《降维映射的稀疏性下界》。载于：第45届计算理论研讨会会议录。STOC '13。美国计算机协会（ACM），2013年，第101 - 110页。DOI: 10.1145/2488608. 2488622（见第13页引用）。

[Pau+13] Saurabh Paul, Christos Boutsidis, Malik Magdon-Ismail, and Petros Drineas. "Random Projections for Support Vector Machines". In: Proceedings of the 16th International Conference on Artificial Intelligence and Statistics (AISTATS '13). Vol. 31. Proceedings of Machine Learning Research (PMLR). PMLR, 2013, pp. 498-506 (cit. on p. 39).

[Pau+13] 索拉布·保罗（Saurabh Paul）、克里斯托斯·布蒂西迪斯（Christos Boutsidis）、马利克·马格登 - 伊斯梅尔（Malik Magdon-Ismail）和彼得罗斯·德里尼亚斯（Petros Drineas）。《支持向量机的随机投影》。载于：第16届人工智能与统计国际会议（AISTATS '13）会议录。第31卷。机器学习研究会议录（PMLR）。机器学习研究会议录（PMLR），2013年，第498 - 506页（见第39页引用）。

[Pau+14] Saurabh Paul, Christos Boutsidis, Malik Magdon-Ismail, and Petros Drineas. "Random Projections for Linear Support Vector Machines". In: Transactions on Knowledge Discovery from Data 8.4 (2014), 22:1-22:25. DOI: 10.1145/2641760. Previously published as [Pau+13] (cit. on p. 6).

[Pau+14] 索拉布·保罗（Saurabh Paul）、克里斯托斯·布蒂西迪斯（Christos Boutsidis）、马利克·马格登 - 伊斯梅尔（Malik Magdon-Ismail）和彼得罗斯·德里尼亚斯（Petros Drineas）。《线性支持向量机的随机投影》。载于《数据知识发现汇刊》（Transactions on Knowledge Discovery from Data）第8卷第4期（2014年），第22:1 - 22:25页。DOI: 10.1145/2641760。此前以[Pau+13]形式发表（见第6页引用）。

[Pea01] Karl Pearson. "On lines and planes of closest fit to systems of points in space". In: The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science 2.11 (1901), pp. 559-572. DOI: 10.1080/14786440109462720 (cit. on p. 2).

[Pea01] 卡尔·皮尔逊（Karl Pearson）。《关于空间点集的最优拟合直线和平面》。载于《伦敦、爱丁堡和都柏林哲学杂志与科学期刊》（The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science）第2卷第11期（1901年），第559 - 572页。DOI: 10.1080/14786440109462720（见第2页引用）。

[PP14] Panagiotis C. Petrantonakis and Panayiota Poirazi. "A compressed sensing perspective of hippocampal function". In: Frontiers in Systems Neuroscience 8 (2014), p. 141. DOI: 10.3389/fnsys. 2014.00141 (cit. on p. 7).

[PP14] 帕纳约蒂斯·C·彼得兰托纳基斯（Panagiotis C. Petrantonakis）和帕纳约塔·波伊拉齐（Panayiota Poirazi）。《从压缩感知角度看海马体功能》。载于《系统神经科学前沿》（Frontiers in Systems Neuroscience）第8卷（2014年），第141页。DOI: 10.3389/fnsys. 2014.00141（见第7页引用）。

[PW14] Mert Pilanci and Martin J. Wainwright. "Randomized Sketches of Convex Programs with Sharp Guarantees". In: Proceedings of the 47th International Symposium on Information Theory. ISIT '14. IEEE, 2014, pp. 921-925. DOI: 10.1109/ISIT. 2014 6874967 (cit. on p. 40).

[PW14] 梅尔特·皮兰西（Mert Pilanci）和马丁·J·温赖特（Martin J. Wainwright）。“具有精确保证的凸规划随机草图”。载于：第47届信息论国际研讨会论文集。ISIT '14。电气与电子工程师协会（IEEE），2014年，第921 - 925页。DOI: 10.1109/ISIT. 2014 6874967（见第40页引用）。

[PW15] Mert Pilanci and Martin J. Wainwright. "Randomized Sketches of Convex Programs with Sharp Guarantees". In: Transactions on Information Theory 61.9 (2015), pp. 5096- 5115. DOI: 10.1109/TIT. 2015.2450722. Previously published as [PW14] (cit. on pp. 5.7).

[PW15] 梅尔特·皮兰西（Mert Pilanci）和马丁·J·温赖特（Martin J. Wainwright）。“具有精确保证的凸规划随机草图”。载于：《信息论汇刊》（Transactions on Information Theory）第61卷第9期（2015年），第5096 - 5115页。DOI: 10.1109/TIT. 2015.2450722。之前以[PW14]发表（见第5、7页引用）。

[Rau09]

Holger Rauhut. "Circulant and Toeplitz matrices in compressed sensing". In: Proceedings of the 2nd Workshop on Signal Processing with Adaptive Sparse Structured Representations. SPARS '09. CCSD, 2009, 32:1-32:6 (cit. on p. 17).

霍尔格·劳胡特（Holger Rauhut）。“压缩感知中的循环矩阵和托普利茨矩阵”。载于：第二届自适应稀疏结构表示信号处理研讨会论文集。SPARS '09。法国国家科研中心文档服务中心（CCSD），2009年，32:1 - 32:6（见第17页引用）。

[RK89]

Helge J. Ritter and Teuvo Kohonen. "Self-Organizing Semantic Maps". In: Biological Cybernetics 61 (1989), pp. 241-254. DOI: 10.1007/BF00203171 (cit. on p. 6).

赫尔格·J·里特（Helge J. Ritter）和特沃·科霍宁（Teuvo Kohonen）。《自组织语义地图》。载于《生物控制论》（Biological Cybernetics）第61卷（1989年），第241 - 254页。DOI: 10.1007/BF00203171（见第6页引用）。

[RN10]

Javier Rojo and Tuan S. Nguyen. "Improving the Johnson-Lindenstrauss Lemma". In: arXiv e-prints (2010). arXiv: 1005.1440 [stat.ML] (cit. on p. 12).

哈维尔·罗霍（Javier Rojo）和阮团·S（Tuan S. Nguyen）。《改进约翰逊 - 林登施特劳斯引理》。载于arXiv预印本（2010年）。arXiv: 1005.1440 [统计机器学习（stat.ML）]（见第12页引用）。

[Rom09] Justin K. Romberg. "Compressive Sensing by Random Convolution". In: Journal on Imaging Sciences 2.4 (2009), pp. 1098-1128. DOI: 10.1137/08072975X (cit. on p. 17).

[Rom09] 贾斯汀·K·龙伯格（Justin K. Romberg）。《通过随机卷积进行压缩感知》。载于《成像科学杂志》（Journal on Imaging Sciences）第2卷第4期（2009年），第1098 - 1128页。DOI: 10.1137/08072975X（见第17页引用）。

[RRT12]

Holger Rauhut, Justin K. Romberg, and Joel Aaron Tropp. "Restricted isometries for partial random circulant matrices". In: Applied and Computational Harmonic Analysis 32.2 (2012), pp. 242-254. DOI: 10.1016/j. acha. 2011.05.001 (cit. on pp. 17.18).

霍尔格·劳胡特（Holger Rauhut）、贾斯汀·K·龙伯格（Justin K. Romberg）和乔尔·亚伦·特罗普（Joel Aaron Tropp）。《部分随机循环矩阵的受限等距性》。载于《应用与计算调和分析》（Applied and Computational Harmonic Analysis）第32卷第2期（2012年），第242 - 254页。DOI: 10.1016/j. acha. 2011.05.001（见第17、18页引用）。

[RST09]

Vladimir Rokhlin, Arthur Szlam, and Mark Tygert. "A Randomized Algorithm for Principal Component Analysis". In: Journal on Matrix Analysis and Applications 31.3 (2009), pp. 1100-1124. DOI: 10.1137/080736417 (cit. on pp. 3, 4).

弗拉基米尔·罗赫林（Vladimir Rokhlin）、亚瑟·斯兹拉姆（Arthur Szlam）和马克·泰格特（Mark Tygert）。《主成分分析的随机算法》。载于《矩阵分析与应用杂志》（Journal on Matrix Analysis and Applications）第31卷第3期（2009年），第1100 - 1124页。DOI: 10.1137/080736417（见第3、4页引用）。

[RV08]

Mark Rudelson and Roman Vershynin. "On Sparse Reconstruction from Fourier and Gaussian Measurements". In: Communications on Pure and Applied Mathematics 61.8 (2008), pp. 1025-1045. DOI: 10.1002/cpa. 20227 (cit. on p. 16).

马克·鲁德尔森（Mark Rudelson）和罗曼·韦申宁（Roman Vershynin）。《基于傅里叶和高斯测量的稀疏重建》。载于《纯粹与应用数学通讯》（Communications on Pure and Applied Mathematics）第61卷第8期（2008年），第1025 - 1045页。DOI: 10.1002/cpa. 20227（见第16页引用）。

[SA09]

Dan D. Stettler and Richard Axel. "Representations of Odor in the Piriform Cortex". In: Neuron 63.6 (2009), pp. 854-864. DOI: 10.1016/j. neuron. 2009.09.005 (cit. on p. 7).

丹·D·斯泰特勒（Dan D. Stettler）和理查德·阿克塞尔（Richard Axel）。《梨状皮质中气味的表征》。载于《神经元》（Neuron）第63卷第6期（2009年），第854 - 864页。DOI: 10.1016/j. neuron. 2009.09.005（见第7页引用）。

[Sar06]

Tamás Sarlós. "Improved Approximation Algorithms for Large Matrices via Random Projections". In: Proceedings of the 47th Symposium on Foundations of Computer Science. FOCS '06. IEEE, 2006, pp. 143-152. DOI: 10.1109/FOCS. 2006.37 (cit. on pp.4.

塔马斯·萨尔洛斯（Tamás Sarlós）。《通过随机投影改进大型矩阵的近似算法》。载于《第47届计算机科学基础研讨会会议录》。FOCS '06。电气与电子工程师协会（IEEE），2006年，第143 - 152页。DOI: 10.1109/FOCS. 2006.37（见第4页引用）。

[Sch18]

Benjamin Schmidt. "Stable Random Projection: Lightweight, General-Purpose Dimensionality Reduction for Digitized Libraries". In: Journal of Cultural Analytics (2018). DOI: 10.22148/16.025 (cit. on pp. 2, 6, 12).

本杰明·施密特（Benjamin Schmidt）。《稳定随机投影：数字化图书馆的轻量级通用降维方法》。载于《文化分析杂志》（Journal of Cultural Analytics）（2018年）。DOI: 10.22148/16.025（见第2、6、12页引用）。

[SF18]

Sami Sieranoja and Pasi Fränti. "Random Projection for k-means Clustering". In: Proceedings of the 17th International Conference on Artificial Intelligence and Soft Computing (ICAISC '18). Vol. 10841. Lecture Notes in Computer Science (LNCS). Springer, 2018, pp. 680-689. DOI: 10.1007/978-3-319-91253-0_63 (cit. on p. 6).

萨米·谢拉诺亚（Sami Sieranoja）和帕西·弗兰蒂（Pasi Fränti）。《用于k均值聚类的随机投影》。载于：第17届人工智能与软计算国际会议（ICAISC '18）论文集。第10841卷。计算机科学讲义（LNCS）。施普林格出版社，2018年，第680 - 689页。DOI: 10.1007/978 - 3 - 319 - 91253 - 0_63（见第6页引用）。

[She17] Or Sheffet. "Differentially Private Ordinary Least Squares". In: Proceedings of the 34st International Conference on Machine Learning (ICML '17). Vol. 70. Proceedings of Machine Learning Research (PMLR). PMLR, 2017, pp. 3105-3114 (cit. on p. 41).

[She17] 奥尔·谢菲特（Or Sheffet）。《差分隐私普通最小二乘法》。载于：第34届机器学习国际会议（ICML '17）论文集。第70卷。机器学习研究会议录（PMLR）。机器学习研究会议录出版社，2017年，第3105 - 3114页（见第41页引用）。

[She19] Or Sheffet. "Differentially Private Ordinary Least Squares". In: Journal of Privacy and Confidentiality 9.1 (2019). DOI: 10.29012/jpc. 654. Previously published as [She17] (cit. on p. 6).

[She19] 奥尔·谢菲特（Or Sheffet）。《差分隐私普通最小二乘法》。载于：《隐私与保密期刊》9.1（2019年）。DOI: 10.29012/jpc. 654。之前以[She17]发表（见第6页引用）。

[Shi+09a] Qinfeng Shi, James Petterson, Gideon Dror, John Langford, Alexander J. Smola, Alexander L. Strehl, and S. V. N. Vishwanathan. "Hash Kernels". In: Proceedings of the 12th International Conference on Artificial Intelligence and Statistics (AISTATS '09). Vol. 5. Proceedings of Machine Learning Research (PMLR). PMLR, 2009, pp. 496-503 (cit. on p. 41).

[Shi+09a] 史钦峰（Qinfeng Shi）、詹姆斯·彼得森（James Petterson）、吉迪恩·德罗尔（Gideon Dror）、约翰·兰福德（John Langford）、亚历山大·J·斯莫拉（Alexander J. Smola）、亚历山大·L·施特尔（Alexander L. Strehl）和S. V. N. 维什瓦纳坦（S. V. N. Vishwanathan）。《哈希核函数》。载于：第12届人工智能与统计国际会议论文集（AISTATS '09）。第5卷。机器学习研究会议录（PMLR）。机器学习研究会议录出版社（PMLR），2009年，第496 - 503页（见第41页引用）。

[Shi+09b] Qinfeng Shi, James Petterson, Gideon Dror, John Langford, Alexander J. Smola, and S. V. N. Vishwanathan. "Hash Kernels for Structured Data". In: Journal of Machine Learning Research 10.90 (2009), pp. 2615-2637. Previously published as [Shi+09a] (cit. on pp. 13, 14).

[Shi+09b] 史钦峰（Qinfeng Shi）、詹姆斯·彼得森（James Petterson）、吉迪恩·德罗尔（Gideon Dror）、约翰·兰福德（John Langford）、亚历山大·J·斯莫拉（Alexander J. Smola）和S. V. N. 维什瓦纳坦（S. V. N. Vishwanathan）。《用于结构化数据的哈希核函数》。载于：《机器学习研究杂志》第10卷第90期（2009年），第2615 - 2637页。之前以[Shi+09a]发表（见第13、14页引用）。

[SI09] Tomoya Sakai and Atsushi Imiya. "Fast Spectral Clustering with Random Projection and Sampling". In: Proceedings of the 6th International Conference on Machine Learning and Data Mining in Pattern Recognition (MLDM '09). Vol. 5632. Lecture Notes in Computer Science (LNCS). Springer, 2009, pp. 372-384. DOI: 10.1007/978-3-642- 03070-3_28 (cit. on p. 6).

[SI09] 酒井智也（Tomoya Sakai）和饭宫敦（Atsushi Imiya）。“基于随机投影和采样的快速谱聚类”。载于：第六届模式识别机器学习与数据挖掘国际会议（MLDM '09）论文集。第5632卷。《计算机科学讲义》（LNCS）。施普林格出版社，2009年，第372 - 384页。DOI: 10.1007/978 - 3 - 642 - 03070 - 3_28（见第6页引用）。

[SKD19] Mehrdad Showkatbakhsh, Can Karakus, and Suhas N. Diggavi. "Privacy-Utility Trade-off of Linear Regression under Random Projections and Additive Noise". In: arXiv e-prints (2019). arXiv: 1902.04688 [cs.LG] (cit. on p. 6).

[SKD19] 梅赫拉德·肖卡特巴赫什（Mehrdad Showkatbakhsh）、坎·卡拉库斯（Can Karakus）和苏哈斯·N·迪加维（Suhas N. Diggavi）。“随机投影和加性噪声下线性回归的隐私 - 效用权衡”。载于：arXiv预印本（2019年）。arXiv: 1902.04688 [计算机科学. 机器学习]（见第6页引用）。

[Sla17]

Martin Slawski. "Compressed Least Squares Regression revisited". In: Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS '17). Vol. 54. Proceedings of Machine Learning Research (PMLR). PMLR, 2017, pp. 1207-1215 (cit. on p. 6).

马丁·斯拉夫斯基（Martin Slawski）。“重温压缩最小二乘回归”。载于：第20届人工智能与统计国际会议（AISTATS '17）论文集。第54卷。机器学习研究会议录（PMLR）。机器学习研究会议录出版社（PMLR），2017年，第1207 - 1215页（见第6页引用）。

[SR09]

Alon Schclar and Lior Rokach. "Random Projection Ensemble Classifiers". In: Proceedings of the 11th International Conference on Enterprise Information Systems (ICEIS '09). Vol. 24. Lecture Notes in Business Information Processing (LNBIP). Springer, 2009, pp. 309-316. DOI: 10.1007/978-3-642-01347-8_26 (cit. on p. 6).

阿隆·施克拉尔（Alon Schclar）和利奥尔·罗卡赫（Lior Rokach）。“随机投影集成分类器”。载于：第11届企业信息系统国际会议（ICEIS '09）论文集。第24卷。商务信息处理讲义（LNBIP）。施普林格出版社（Springer），2009年，第309 - 316页。DOI: 10.1007/978 - 3 - 642 - 01347 - 8_26（见第6页引用）。

[SS08]

Daniel A. Spielman and Nikhil Srivastava. "Graph Sparsification by Effective Resistances". In: Proceedings of the 40th Symposium on Theory of Computing. STOC '08. ACM, 2008, pp. 563-568. DOI: 10.1145/1374376. 1374456 (cit. on p. 41).

丹尼尔·A·斯皮尔曼（Daniel A. Spielman）和尼基尔·斯里瓦斯塔瓦（Nikhil Srivastava）。《基于有效电阻的图稀疏化》。载于：第40届计算理论研讨会会议录。STOC '08。美国计算机协会（ACM），2008年，第563 - 568页。DOI: 10.1145/1374376. 1374456（见第41页引用）。

[SS11]

Daniel A. Spielman and Nikhil Srivastava. "Graph Sparsification by Effective Resistances". In: Journal on Computing 40.6 (2011), pp. 1913-1926. DOI: 10.1137/ 080734029. Previously published as [SS08] (cit. on p. 7).

丹尼尔·A·斯皮尔曼（Daniel A. Spielman）和尼基尔·斯里瓦斯塔瓦（Nikhil Srivastava）。《基于有效电阻的图稀疏化》。载于：《计算期刊》（Journal on Computing）第40卷第6期（2011年），第1913 - 1926页。DOI: 10.1137/ 080734029。之前以[SS08]发表（见第7页引用）。

[SW17] Piotr Sankowski and Piotr Wygocki. "Approximate Nearest Neighbors Search Without False Negatives For ${\ell }_{2}$ For $c > \sqrt{\log \log n}$ ". In: Proceedings of the 28th International Symposium on Algorithms and Computation (ISAAC '17). Vol. 92. Leibniz International Proceedings in Informatics (LIPIcs). Schloss Dagstuhl, 2017, 63:1- 63:12. DOI: 10.4230/LIPIcs.ISAAC. 2017.63 (cit. on p. 5).

[SW17] 彼得·桑科夫斯基（Piotr Sankowski）和彼得·维戈茨基（Piotr Wygocki）。“针对${\ell }_{2}$和$c > \sqrt{\log \log n}$无漏检的近似最近邻搜索”。载于：第28届国际算法与计算研讨会（ISAAC '17）论文集。第92卷。莱布尼茨国际信息学会议录（LIPIcs）。德国达格施塔特城堡出版社，2017年，63:1 - 63:12。DOI: 10.4230/LIPIcs.ISAAC. 2017.63（见第5页引用）。

[SW89] John Simpson and Edmund Weiner, eds. Oxford English Dictionary. 2nd ed. 20 vols. Oxford University Press, 1989. ISBN: 978-0-19-861186-8 (cit. on p. 2).

[SW89] 约翰·辛普森（John Simpson）和埃德蒙·韦纳（Edmund Weiner）编。《牛津英语词典》。第2版。共20卷。牛津大学出版社，1989年。ISBN: 978 - 0 - 19 - 861186 - 8（见第2页引用）。

[SZK15] Erich Schubert, Arthur Zimek, and Hans-Peter Kriegel. "Fast and Scalable Outlier Detection with Approximate Nearest Neighbor Ensembles". In: Proceedings of the 20th International Conference on Database Systems for Advanced Applications (DASFAA '15). Vol. 9050. Lecture Notes in Computer Science (LNCS). Springer, 2015, pp. 19-36. DOI: 10.1007/978-3-319-18123-3_2 (cit. on p. 6).

[SZK15] 埃里希·舒伯特（Erich Schubert）、亚瑟·齐梅克（Arthur Zimek）和汉斯 - 彼得·克里格尔（Hans - Peter Kriegel）。“基于近似最近邻集成的快速可扩展离群点检测”。载于：第20届高级应用数据库系统国际会议（DASFAA '15）论文集。第9050卷。计算机科学讲义（LNCS）。施普林格出版社，2015年，第19 - 36页。DOI: 10.1007/978 - 3 - 319 - 18123 - 3_2（见第6页引用）。

[Tan+05] Bin Tang, Michael A. Shepherd, Malcolm I. Heywood, and Xiao Luo. "Comparing Dimension Reduction Techniques for Document Clustering". In: Proceedings of the 18th Canadian Conference on AI (AI '05). Vol. 3501. Lecture Notes in Computer Science (LNCS). Springer, 2005, pp. 292-296. DOI: 10.1007/11424918_30 (cit. on p. 4).

[Tan+05] 唐斌（Bin Tang）、迈克尔·A·谢泼德（Michael A. Shepherd）、马尔科姆·I·海伍德（Malcolm I. Heywood）和肖罗（Xiao Luo）。“文档聚类中降维技术的比较”。载于：第18届加拿大人工智能会议（AI '05）论文集。第3501卷。计算机科学讲义（LNCS）。施普林格出版社，2005年，第292 - 296页。DOI: 10.1007/11424918_30（见第4页引用）。

[Tar+19] Olga Taran, Shideh Rezaeifar, Taras Holotyak, and Slava Voloshynovskiy. "Defending Against Adversarial Attacks by Randomized Diversification". In: Proceedings of the 32nd Conference on Computer Vision and Pattern Recognition. CVPR '19. IEEE, 2019, pp. 11218-11225. DOI: 10.1109/CVPR. 2019.01148 (cit. on p. 6).

[Tar+19] 奥尔加·塔兰（Olga Taran）、希德·礼萨伊法尔（Shideh Rezaeifar）、塔拉斯·霍洛蒂亚克（Taras Holotyak）和斯拉瓦·沃洛申诺夫斯基（Slava Voloshynovskiy）。《通过随机多样化抵御对抗性攻击》。见：第32届计算机视觉与模式识别会议论文集。CVPR '19。电气与电子工程师协会（IEEE），2019年，第11218 - 11225页。DOI: 10.1109/CVPR. 2019.01148（引自第6页）。

[THM17] Gian-Andrea Thanei, Christina Heinze, and Nicolai Meinshausen. "Random Projections for Large-Scale Regression". In: Big and Complex Data Analysis: Methodologies and Applications. Contributions to Statistics. Springer, 2017, pp. 51-68. DOI: 10.1007/978-3-319-41573-4_3 (cit. on p.6).

[THM17] 吉安 - 安德烈亚·塔内伊（Gian - Andrea Thanei）、克里斯蒂娜·海因策（Christina Heinze）和尼科莱·迈因豪森（Nicolai Meinshausen）。《大规模回归的随机投影》。见：《大数据与复杂数据分析：方法与应用》。《统计学贡献》。施普林格出版社（Springer），2017年，第51 - 68页。DOI: 10.1007/978 - 3 - 319 - 41573 - 4_3（引自第6页）。

[Tro11] Joel Aaron Tropp. "Improved Analysis of the subsampled Randomized Hadamard Transform". In: Advances in Adaptive Data Analysis 3.1-2 (2011), pp. 115-126. DOI: 10.1142/S1793536911000787 (cit. on p. 17).

[特罗普11] 乔尔·亚伦·特罗普（Joel Aaron Tropp）。《对下采样随机哈达玛变换的改进分析》。载于《自适应数据分析进展》第3卷第1 - 2期（2011年），第115 - 126页。DOI: 10.1142/S1793536911000787（见第17页引用）。

[TSC15]

Yin Tat Lee, Aaron Sidford, and Sam Chiu-Wai Wong. "A Faster Cutting Plane Method and its Implications for Combinatorial and Convex Optimization". In: Proceedings of the 56th Symposium on Foundations of Computer Science. FOCS '15. IEEE, 2015, pp. 1049-1065. DOI: 10.1109/FOCS. 2015.68 (cit. on p. 6).

李尹达（Yin Tat Lee）、亚伦·西德福德（Aaron Sidford）和黄兆辉（Sam Chiu - Wai Wong）。《一种更快的割平面法及其对组合优化和凸优化的意义》。载于《第56届计算机科学基础研讨会会议录》。FOCS '15。电气与电子工程师协会（IEEE），2015年，第1049 - 1065页。DOI: 10.1109/FOCS. 2015.68（见第6页引用）。

[Tur+08]

E. Onur Turgay, Thomas Brochmann Pedersen, Yücel Saygin, Erkay Savas, and Albert Levi. "Disclosure Risks of Distance Preserving Data Transformations". In: Proceedings of the 20th International Conference on Scientific and Statistical Database Management (SSDBM '08). Vol. 5069. Lecture Notes in Computer Science (LNCS). Springer, 2008, pp. 79-94. DOI: 10.1007/978-3-540-69497-7_8 (cit. on p.7).

厄努尔·图尔盖（E. Onur Turgay）、托马斯·布罗克曼·佩德森（Thomas Brochmann Pedersen）、于塞尔·赛金（Yücel Saygin）、埃尔凯·萨瓦斯（Erkay Savas）和阿尔伯特·利维（Albert Levi）。《距离保持数据转换的披露风险》。载于：第20届科学与统计数据库管理国际会议论文集（SSDBM '08）。第5069卷。计算机科学讲义（LNCS）。施普林格出版社，2008年，第79 - 94页。DOI: 10.1007/978 - 3 - 540 - 69497 - 7_8（引自第7页）。

[TZ04]

Mikkel Thorup and Yin Zhang. "Tabulation based 4-universal hashing with applications to second moment estimation". In: Proceedings of the 15th Symposium on Discrete Algorithms. SODA '04. SIAM, 2004, pp. 615-624 (cit. on p. 43).

米克尔·索鲁普（Mikkel Thorup）和张寅（Yin Zhang）。《基于制表的4 - 通用哈希及其在二阶矩估计中的应用》。载于：第15届离散算法研讨会论文集。SODA '04。工业与应用数学学会（SIAM），2004年，第615 - 624页（引自第43页）。

[TZ10] Mikkel Thorup and Yin Zhang. "Tabulation Based 5-Universal Hashing and Linear Probing". In: Proceedings of the 12th Workshop on Algorithm Engineering and Experiments. ALENEX '10. SIAM, 2010, pp. 62-76. DOI: 10.1137/1. 9781611972900. 7 (cit. on p. 43).

[TZ10] 米克尔·索鲁普（Mikkel Thorup）和张寅（Yin Zhang）。“基于制表的5-通用哈希与线性探测”。见：第12届算法工程与实验研讨会论文集。ALENEX '10。美国工业与应用数学学会（SIAM），2010年，第62 - 76页。DOI: 10.1137/1. 9781611972900. 7（引自第43页）。

[TZ12] Mikkel Thorup and Yin Zhang. "Tabulation-Based 5-Independent Hashing with Applications to Linear Probing and Second Moment Estimation". In: Journal on Computing 41.2 (2012), pp. 293-331. DOI: 10.1137/100800774. Previously published as [TZ10; TZ04] (cit. on p. 10).

[TZ12] 米克尔·索鲁普（Mikkel Thorup）和张寅（Yin Zhang）。“基于制表的5-独立哈希及其在线性探测和二阶矩估计中的应用”。见：《计算期刊》（Journal on Computing）第41卷第2期（2012年），第293 - 331页。DOI: 10.1137/100800774。先前以 [TZ10; TZ04] 形式发表（引自第10页）。

[UDS07]

Thierry Urruty, Chabane Djeraba, and Dan A. Simovici. "Clustering by Random Projections". In: Proceedings of the 3rd Industrial Conference on Data Mining (ICDM '07). Vol. 4597. Lecture Notes in Computer Science (LNCS). Springer, 2007, pp. 107-119. DOI: 10.1007/978-3-540-73435-2_9 (cit. on p. 6).

蒂埃里·于尔吕蒂（Thierry Urruty）、沙巴内·杰拉巴（Chabane Djeraba）和丹·A·西莫维奇（Dan A. Simovici）。《通过随机投影进行聚类》。载于：第三届数据挖掘工业会议（ICDM '07）论文集。第4597卷。《计算机科学讲义》（LNCS）。施普林格出版社，2007年，第107 - 119页。DOI: 10.1007/978 - 3 - 540 - 73435 - 2_9（见第6页引用）。

[Unk70] Unknown embroiderer(s). Bayeux Tapestry. Embroidery. Musée de la Tapisserie de Bayeux, Bayeux, France, ca. 1070 (cit. on p. 11).

[Unk70] 未知刺绣者。《贝叶挂毯》。刺绣作品。法国贝叶市贝叶挂毯博物馆藏，约1070年（见第11页引用）。

[Upa13] Jalaj Upadhyay. "Random Projections, Graph Sparsification, and Differential Privacy". In: Proceedings of the 19th International Conference on the Theory and Application of Cryptology and Information Security (ASIACRYPT '13). Vol. 8269. Lecture Notes in Computer Science (LNCS). Springer, 2013, pp. 276-295. DOI: 10.1007/978-3-642- 42033-7_15 (cit. on p.7).

[Upa13] 贾拉杰·乌帕德亚伊（Jalaj Upadhyay）。《随机投影、图稀疏化与差分隐私》。载于：第19届密码学与信息安全理论与应用国际会议（ASIACRYPT '13）论文集。第8269卷。《计算机科学讲义》（LNCS）。施普林格出版社，2013年，第276 - 295页。DOI: 10.1007/978 - 3 - 642 - 42033 - 7_15（引自第7页）。

[Upa15] Jalaj Upadhyay. "Randomness Efficient Fast-Johnson-Lindenstrauss Transform with Applications in Differential Privacy and Compressed Sensing". In: arXiv e-prints (2015). arXiv: 1410.2470 [cs.DS] (cit. on p. 7).

[Upa15] 贾拉杰·乌帕德亚伊（Jalaj Upadhyay）。《具有差分隐私和压缩感知应用的随机高效快速约翰逊 - 林登施特劳斯变换》。载于：arXiv预印本（2015年）。arXiv: 1410.2470 [计算机科学 - 数据结构]（引自第7页）。

[Upa18] Jalaj Upadhyay. "The Price of Privacy for Low-rank Factorization". In: Advances in Neural Information Processing Systems 31. NeurIPS '18. Curran Associates, 2018, pp. 4176-4187 (cit. on p. 7).

[Upa18] 贾拉杰·乌帕德亚伊（Jalaj Upadhyay）。《低秩分解的隐私代价》。载于：《神经信息处理系统进展31》。神经信息处理系统大会（NeurIPS '18）。柯伦联合出版社，2018年，第4176 - 4187页（引自第7页）。

[Vem04] Santosh S. Vempala. The Random Projection Method. Vol. 65. DIMACS Series in Discrete Mathematics and Theoretical Computer Science. AMS, 2004. DOI: 10.1090/ dimacs/065 (cit. on p.7).

[Vem04] 桑托什·S·文帕拉（Santosh S. Vempala）。《随机投影方法》。第65卷。离散数学与理论计算机科学的DIMACS系列。美国数学学会（AMS），2004年。DOI: 10.1090/ dimacs/065（见第7页引用）。

[Vem98] Santosh S. Vempala. "Random Projection: A New Approach to VLSI Layout". In: Proceedings of the 39th Symposium on Foundations of Computer Science. FOCS '98. IEEE, 1998, pp. 389-395. DOI: 10.1109/SFCS. 1998.743489 (cit. on p. 7).

[Vem98] 桑托什·S·文帕拉（Santosh S. Vempala）。《随机投影：超大规模集成电路布局的新方法》。载于：第39届计算机科学基础研讨会会议录。FOCS '98。电气与电子工程师协会（IEEE），1998年，第389 - 395页。DOI: 10.1109/SFCS. 1998.743489（见第7页引用）。

[VPL15] Ky Khac Vu, Pierre-Louis Poirion, and Leo Liberti. "Using the Johnson-Lindenstrauss Lemma in Linear and Integer Programming". In: arXiv e-prints (2015). arXiv: 1507.00990 [math.OC] (cit. on p. 6).

[VPL15] 阮克启（Ky Khac Vu）、皮埃尔 - 路易·普里翁（Pierre-Louis Poirion）和利奥·利伯蒂（Leo Liberti）。《在线性和整数规划中使用约翰逊 - 林登施特劳斯引理》。载于：arXiv预印本（2015年）。arXiv: 1507.00990 [数学.运筹学与控制论]（见第6页引用）。

[Vyb11] Jan Vybíral. "A variant of the Johnson-Lindenstrauss lemma for circulant matrices". In: Journal of Functional Analysis 260.4 (2011), pp. 1096-1105. DOI: 10.1016/j. jfa 2010.11.014 (cit. on pp. 18.21).

[Vyb11] 扬·维比拉尔（Jan Vybíral）。“循环矩阵的约翰逊 - 林登施特劳斯引理（Johnson-Lindenstrauss lemma）的一个变体”。载于：《泛函分析杂志》（Journal of Functional Analysis）260.4（2011 年），第 1096 - 1105 页。DOI: 10.1016/j. jfa 2010.11.014（见第 18、21 页引用）。

[WC81]

Mark N. Wegman and J. Lawrence Carter. "New Hash Functions and Their Use in Authentication and Set Equality". In: Journal of Computer and System Sciences 22.3 (1981), pp. 265-279. DOI: 10.1016/0022-0000(81) 90033-7 (cit. on p. 10).

马克·N·韦格曼（Mark N. Wegman）和 J·劳伦斯·卡特（J. Lawrence Carter）。“新的哈希函数及其在认证和集合相等性中的应用”。载于：《计算机与系统科学杂志》（Journal of Computer and System Sciences）22.3（1981 年），第 265 - 279 页。DOI: 10.1016/0022-0000(81) 90033 - 7（见第 10 页引用）。

[WDJ91] Christopher B. Walton, Alfred G. Dale, and Roy M. Jenevein. "A Taxonomy and Performance Model of Data Skew Effects in Parallel Joins". In: Proceedings of the 17th International Conference on Very Large Data Bases. VLDB '91. Morgan Kaufmann, 1991, pp. 537-548 (cit. on p. 10).

[WDJ91] 克里斯托弗·B·沃尔顿（Christopher B. Walton）、阿尔弗雷德·G·戴尔（Alfred G. Dale）和罗伊·M·杰内文（Roy M. Jenevein）。“并行连接中数据倾斜效应的分类法和性能模型”。载于：第 17 届超大型数据库国际会议论文集。VLDB '91。摩根·考夫曼出版社（Morgan Kaufmann），1991 年，第 537 - 548 页（见第 10 页引用）。

[Wee+19] Sandamal Weerasinghe, Sarah Monazam Erfani, Tansu Alpcan, and Christopher Leckie. "Support vector machines resilient against training data integrity attacks". In: Pattern Recognition 96 (2019), p. 106985. DOI: 10.1016/j.patcog. 2019.106985 (cit. on p. 6).

[Wee+19] 桑达马尔·韦拉辛哈（Sandamal Weerasinghe）、莎拉·莫纳扎姆·埃尔法尼（Sarah Monazam Erfani）、坦苏·阿尔普坎（Tansu Alpcan）和克里斯托弗·莱基（Christopher Leckie）。“抗训练数据完整性攻击的支持向量机”。见：《模式识别》（Pattern Recognition）96（2019），第106985页。DOI: 10.1016/j.patcog. 2019.106985（引自第6页）。

[Wei+09] Kilian Weinberger, Anirban Dasgupta, John Langford, Alex Smola, and Josh Attenberg. "Feature Hashing for Large Scale Multitask Learning". In: Proceedings of the 26th International Conference on Machine Learning. ICML '09. ACM, 2009, pp. 1113-1120. DOI: 10.1145/1553374. 1553516 (cit. on pp. 13, 14.21).

[Wei+09] 基利安·温伯格（Kilian Weinberger）、阿尼尔班·达斯古普塔（Anirban Dasgupta）、约翰·兰福德（John Langford）、亚历克斯·斯莫拉（Alex Smola）和约什·阿滕伯格（Josh Attenberg）。“用于大规模多任务学习的特征哈希”。见：《第26届国际机器学习会议论文集》（Proceedings of the 26th International Conference on Machine Learning）。ICML '09。美国计算机协会（ACM），2009年，第1113 - 1120页。DOI: 10.1145/1553374. 1553516（引自第13、14、21页）。

[Wei+10] Kilian Weinberger, Anirban Dasgupta, Josh Attenberg, John Langford, and Alex Smola. "Feature Hashing for Large Scale Multitask Learning". In: arXiv e-prints (2010). arXiv: 0902.2206v5 [cs.AI] (cit. on p. 14).

[Wei+10] 基利安·温伯格（Kilian Weinberger）、阿尼尔班·达斯古普塔（Anirban Dasgupta）、乔希·阿滕伯格（Josh Attenberg）、约翰·兰福德（John Langford）和亚历克斯·斯莫拉（Alex Smola）。“大规模多任务学习的特征哈希”。载于：arXiv预印本（2010年）。arXiv: 0902.2206v5 [计算机科学.人工智能]（见第14页引用）。

[Wil12] Virginia Vassilevska Williams. "Multiplying Matrices faster than Coppersmith-Winograd". In: Proceedings of the 44th Symposium on Theory of Computing. STOC '12. ACM, 2012, pp. 887-898. DOI: 10.1145/2213977.2214056 (cit. on p. 4).

[Wil12] 弗吉尼亚·瓦西列夫斯卡·威廉姆斯（Virginia Vassilevska Williams）。“比铜匠 - 温诺格拉德算法更快的矩阵乘法”。载于：第44届计算理论研讨会会议录。STOC '12。美国计算机协会（ACM），2012年，第887 - 898页。DOI: 10.1145/2213977.2214056（见第4页引用）。

[Woj+16] Michael Wojnowicz, Di Zhang, Glenn Chisholm, Xuan Zhao, and Matt Wolff. "Projecting "Better Than Randomly": How to Reduce the Dimensionality of Very Large Datasets in a Way that Outperforms Random Projections". In: Proceedings of the 3rd International Conference on Data Science and Advanced Analytics. DSAA '16. IEEE, 2016, pp. 184-193. DOI: 10.1109/DSAA. 2016.26 (cit. on p. 4).

[Woj+16] 迈克尔·沃伊诺维奇（Michael Wojnowicz）、张迪（Di Zhang）、格伦·奇泽姆（Glenn Chisholm）、赵轩（Xuan Zhao）和马特·沃尔夫（Matt Wolff）。《“优于随机投影”：如何以超越随机投影的方式对超大型数据集进行降维》。载于：第三届数据科学与高级分析国际会议论文集。DSAA '16。电气与电子工程师协会（IEEE），2016年，第184 - 193页。DOI: 10.1109/DSAA. 2016.26（见第4页引用）。

[Woo14] David P. Woodruff. "Sketching as a Tool for Numerical Linear Algebra". In: Foundations and Trends® in Theoretical Computer Science 10.1-2 (2014), pp. 1-157. DOI: 10.1561/0400000060 (cit. on p. 6).

[Woo14] 大卫·P·伍德拉夫（David P. Woodruff）。《草图法：数值线性代数的工具》。载于：《理论计算机科学基础与趋势》（Foundations and Trends® in Theoretical Computer Science）第10卷第1 - 2期（2014年），第1 - 157页。DOI: 10.1561/0400000060（见第6页引用）。

[WZ05] Hugh E. Williams and Justin Zobel. "Searchable words on the Web". In: International Journal on Digital Libraries 5 (2005), pp. 99-105. DOI: 10.1007/s00799-003-0050-z (cit. on p. 2).

[WZ05] 休·E·威廉姆斯（Hugh E. Williams）和贾斯汀·佐贝尔（Justin Zobel）。《网络上的可搜索词汇》。载于：《数字图书馆国际期刊》（International Journal on Digital Libraries）第5卷（2005年），第99 - 105页。DOI: 10.1007/s00799-003-0050-z（见第2页引用）。

[Xie+16] Haozhe Xie, Jie Li, Qiaosheng Zhang, and Yadong Wang. "Comparison among dimensionality reduction techniques based on Random Projection for cancer classification". In: Computational Biology and Chemistry 65 (2016), pp. 165-172. DOI: 10.1016/j.compbiolchem.2016.09.010(cit.onp.4).

[谢+16] 谢浩哲（Haozhe Xie）、李杰（Jie Li）、张巧生（Qiaosheng Zhang）和王亚东（Yadong Wang）。“基于随机投影的降维技术在癌症分类中的比较”。载于《计算生物学与化学》（Computational Biology and Chemistry）65 卷（2016 年），第 165 - 172 页。DOI: 10.1016/j.compbiolchem.2016.09.010（引自第 4 页）。

$\left\lbrack  {\mathrm{{Xu}} + {17}}\right\rbrack$ Chugui Xu, Ju Ren, Yaoxue Zhang, Zhan Qin, and Kui Ren. "DPPro: Differentially Private High-Dimensional Data Release via Random Projection". In: Transactions on Information Forensics and Security 12.12 (2017), pp. 3081-3093. DOI: 10.1109/TIFS 2017.2737966 (cit. on p. 7).

$\left\lbrack  {\mathrm{{Xu}} + {17}}\right\rbrack$ 徐楚贵（Chugui Xu）、任菊（Ju Ren）、张耀学（Yaoxue Zhang）、覃展（Zhan Qin）和任奎（Kui Ren）。“DPPro：通过随机投影实现差分隐私的高维数据发布”。载于《信息取证与安全汇刊》（Transactions on Information Forensics and Security）12 卷 12 期（2017 年），第 3081 - 3093 页。DOI: 10.1109/TIFS 2017.2737966（引自第 7 页）。

[Yan+17] Mengmeng Yang, Tianqing Zhu, Lichuan Ma, Yang Xiang, and Wanlei Zhou. "Privacy Preserving Collaborative Filtering via the Johnson-Lindenstrauss Transform". In: Proceedings of the 16th International Conference on Trust, Security and Privacy in Computing and Communications (TrustCom '17) and the 11th International Conference on Big Data Science and Engineering (BigDataSE '17) and the 14th International Conference on Embedded Software and Systems (ICESS'17). IEEE, 2017, pp. 417-424. DOI: 10.1109/Trustcom/BigDataSE/ICESS.2017.266 (cit. on p. 7).

[Yan+17] 杨萌萌（Mengmeng Yang）、朱天庆（Tianqing Zhu）、马立川（Lichuan Ma）、向洋（Yang Xiang）和周万雷（Wanlei Zhou）。“通过约翰逊 - 林登斯特劳斯变换实现隐私保护的协同过滤”。见：第16届计算与通信中的信任、安全和隐私国际会议（TrustCom '17）、第11届大数据科学与工程国际会议（BigDataSE '17）和第14届嵌入式软件与系统国际会议（ICESS'17）论文集。电气和电子工程师协会（IEEE），2017年，第417 - 424页。DOI: 10.1109/Trustcom/BigDataSE/ICESS.2017.266（引自第7页）。

[Yan+20] Fan Yang, Sifan Liu, Edgar Dobriban, and David P. Woodruff. "How to reduce dimension with PCA and random projections?" In: arXiv e-prints (2020). arXiv: 2005.00511 [math.ST] (cit. on p.4).

[Yan+20] 杨帆（Fan Yang）、刘思凡（Sifan Liu）、埃德加·多布里班（Edgar Dobriban）和大卫·P·伍德拉夫（David P. Woodruff）。“如何使用主成分分析（PCA）和随机投影进行降维？”见：预印本平台arXiv（2020年）。arXiv: 2005.00511 [数学统计学（math.ST）]（引自第4页）。

[Zha+13] 3] Lijun Zhang, Mehrdad Mahdavi, Rong Jin, Tianbao Yang, and Shenghuo Zhu. "Recovering the Optimal Solution by Dual Random Projection". In: Proceedings of the 26th Annual Conference on Learning Theory (COLT '13). Vol. 30. Proceedings of Machine Learning Research (PMLR). PMLR, 2013, pp. 135-157 (cit. on p. 6).

[Zha+13] 3] 张立军（Lijun Zhang）、梅赫拉德·马赫达维（Mehrdad Mahdavi）、金榕（Rong Jin）、杨天保（Tianbao Yang）和朱圣火（Shenghuo Zhu）。“通过双重随机投影恢复最优解”。见：第26届学习理论年度会议（COLT '13）论文集。第30卷。机器学习研究会议录（PMLR）。机器学习研究会议录（PMLR），2013年，第135 - 157页（引自第6页）。

[Zha+20] Yue Zhao, Xueying Ding, Jianing Yang, and Haoping Bai. "SUOD: Toward Scalable Unsupervised Outlier Detection". In: Proceedings of the AAAI-20 Workshop on Artificial Intelligence for Cyber Security. AICS '20. 2020. arXiv: 2002.03222 [cs.LG] (cit. on p. 6).

[Zha+20] 赵越（Yue Zhao）、丁雪莹（Xueying Ding）、杨佳宁（Jianing Yang）和白浩平（Haoping Bai）。“SUOD：迈向可扩展的无监督离群点检测”。见：AAAI - 20网络安全人工智能研讨会论文集。AICS '20。2020年。预印本：2002.03222 [计算机科学.机器学习]（引自第6页）。

[ZK19] Xi Zhang and Ata Kabán. "Experiments with Random Projections Ensembles: Linear Versus Quadratic Discriminants". In: Proceedings of the 2019 International Conference on Data Mining Workshops. ICDMW '19. IEEE, 2019, pp. 719-726. DOI: 10.1109/ICDMW.2019.00108 (cit. on p.6).

[ZK19] 张曦（Xi Zhang）和阿塔·卡班（Ata Kabán）。《随机投影集成实验：线性与二次判别》。见：《2019 年国际数据挖掘研讨会论文集》。ICDMW '19。电气与电子工程师协会（IEEE），2019 年，第 719 - 726 页。DOI: 10.1109/ICDMW.2019.00108（引自第 6 页）。

[ZLW07] Shuheng Zhou, John D. Lafferty, and Larry A. Wasserman. "Compressed Regression". In: Advances in Neural Information Processing Systems 20. NIPS '07. Curran Associates, Inc., 2007, pp. 1713-1720 (cit. on p. 45).

[ZLW07] 周树heng（Shuheng Zhou）、约翰·D·拉弗蒂（John D. Lafferty）和拉里·A·沃瑟曼（Larry A. Wasserman）。《压缩回归》。见：《神经信息处理系统进展 20》。NIPS '07。柯伦联合公司（Curran Associates, Inc.），2007 年，第 1713 - 1720 页（引自第 45 页）。

[ZLW09] Shuheng Zhou, John D. Lafferty, and Larry A. Wasserman. "Compressed and Privacy-Sensitive Sparse Regression". In: Transactions on Information Theory 55.2 (2009), pp. 846-866. DOI: 10.1109/TIT. 2008. 2009605. Previously published as [ZLW07] (cit. on p. 7).

[ZLW09] 周树heng（Shuheng Zhou）、约翰·D·拉弗蒂（John D. Lafferty）和拉里·A·沃瑟曼（Larry A. Wasserman）。《压缩且隐私敏感的稀疏回归》。见：《信息论汇刊》第 55 卷第 2 期（2009 年），第 846 - 866 页。DOI: 10.1109/TIT. 2008. 2009605。之前以 [ZLW07] 发表（引自第 7 页）。

<!-- Media -->