# The X-tree: An Index Structure for High-Dimensional Data

# X树：一种用于高维数据的索引结构

Stefan Berchtold

斯特凡·贝希托尔德

Daniel A. Keim

丹尼尔·A·凯姆

Hans-Peter Kriegel

汉斯 - 彼得·克里格尔

Institute for Computer Science, University of Munich, Oettingenstr. 67, D-80538 Munich, Germany \{berchtol, keim, kriegel\} @informatik.uni-muenchen.de

慕尼黑大学计算机科学研究所，奥廷根大街67号，德国慕尼黑80538 \{berchtol, keim, kriegel\} @informatik.uni - muenchen.de

## Abstract

## 摘要

In this paper, we propose a new method for indexing large amounts of point and spatial data in high-dimensional space. An analysis shows that index structures such as the R*-tree are not adequate for indexing high-dimensional data sets. The major problem of R-tree-based index structures is the overlap of the bounding boxes in the directory, which increases with growing dimension. To avoid this problem, we introduce a new organization of the directory which uses a split algorithm minimizing overlap and additionally utilizes the concept of supernodes. The basic idea of overlap-minimizing split and supernodes is to keep the directory as hierarchical as possible, and at the same time to avoid splits in the directory that would result in high overlap. Our experiments show that for high-dimensional data,the $\mathrm{X}$ -tree outperforms the well-known R*-tree and the TV-tree by up to two orders of magnitude.

在本文中，我们提出了一种在高维空间中对大量点数据和空间数据进行索引的新方法。分析表明，诸如R*树之类的索引结构不足以对高维数据集进行索引。基于R树的索引结构的主要问题是目录中边界框的重叠，这种重叠会随着维度的增加而加剧。为避免这一问题，我们引入了一种新的目录组织方式，它使用一种最小化重叠的分裂算法，并额外利用了超级节点的概念。最小化重叠分裂和超级节点的基本思想是使目录尽可能保持层次结构，同时避免在目录中进行会导致高度重叠的分裂。我们的实验表明，对于高维数据，$\mathrm{X}$树的性能比著名的R*树和TV树高出多达两个数量级。

## 1. Introduction

## 1. 引言

In many applications, indexing of high-dimensional data has become increasingly important. In multimedia databases, for example, the multimedia objects are usually mapped to feature vectors in some high-dimensional space and queries are processed against a database of those feature vectors [Fal 94]. Similar approaches are taken in many other areas including CAD [MG 93], molecular biology (for the docking of molecules) [SBK 92], string matching and sequence alignment [AGMM 90], etc. Examples of feature vectors are color histograms [SH 94], shape descriptors [Jag 91, MG 95], Fourier vectors [WW 80], text descriptors [Kuk 92], etc. In some applications, the mapping process does not yield point objects, but extended spatial objects in high-dimensional space [MN 95]. In many of the mentioned applications, the databases are very large and consist of millions of data objects with several tens to a few hundreds of dimensions. For querying these databases, it is essential to use appropriate indexing techniques which provide an efficient access to high-dimensional data. The goal of this paper is to demonstrate the limits of currently available index structures, and present a new index structure which considerably improves the performance in indexing high-dimensional data.

在许多应用中，高维数据的索引变得越来越重要。例如，在多媒体数据库中，多媒体对象通常被映射到某个高维空间中的特征向量，并且针对这些特征向量的数据库进行查询处理[Fal 94]。许多其他领域也采用了类似的方法，包括计算机辅助设计（CAD）[MG 93]、分子生物学（用于分子对接）[SBK 92]、字符串匹配和序列比对[AGMM 90]等。特征向量的例子有色直方图[SH 94]、形状描述符[Jag 91, MG 95]、傅里叶向量[WW 80]、文本描述符[Kuk 92]等。在一些应用中，映射过程不会产生点对象，而是高维空间中的扩展空间对象[MN 95]。在上述许多应用中，数据库非常大，由数百万个具有几十到几百个维度的数据对象组成。为了查询这些数据库，必须使用适当的索引技术，以便能够高效地访问高维数据。本文的目标是展示当前可用索引结构的局限性，并提出一种新的索引结构，该结构能显著提高高维数据索引的性能。

Our approach is motivated by an examination of R-tree-based index structures. One major reason for using R-tree-based index structures is that we have to index not only point data but also extended spatial data, and R-tree-based index structures are well suited for both types of data. In contrast to most other index structures (such as kdB-trees [Rob 81], grid files [NHS 84], and their variants [see e.g. SK 90]), R-tree-based index structures do not need point transformations to store spatial data and therefore provide a better spatial clustering.

我们的方法是基于对基于R树的索引结构的研究。使用基于R树的索引结构的一个主要原因是，我们不仅要对点位数据进行索引，还要对扩展空间数据进行索引，而基于R树的索引结构非常适合这两种类型的数据。与大多数其他索引结构（如kdB树[Rob 81]、网格文件[NHS 84]及其变体[例如见SK 90]）不同，基于R树的索引结构不需要进行点位变换来存储空间数据，因此能提供更好的空间聚类。

Some previous work on indexing high-dimensional data has been done, mainly focussing on two different approaches. The first approach is based on the observation that real data in high-dimensional space are highly correlated and clustered, and therefore the data occupy only some subspace of the high-dimensional space. Algorithms such as Fastmap [FL 95], multidimensional scaling [KW 78], principal component analysis [DE 82], and factor analysis [Har 67] take advantage of this fact and transform data objects into some lower dimensional space which can be efficiently indexed using traditional multidimensional index structures. A similar approach is proposed in the SS-tree [WJ 96] which is an R-tree-based index structure. The SS-tree uses ellipsoid bounding regions in a lower dimensional space applying a different transformation in each of the directory nodes. The second approach is based on the observation that in most high-dimensional data sets, a small number of the dimensions bears most of the information. The TV-tree [LJF 94], for example, organizes the directory in a way that only the information needed to distinguish between data objects is stored in the directory. This leads to a higher fanout and a smaller directory, resulting in a better query performance.

之前已经有一些关于高维数据索引的工作，主要集中在两种不同的方法上。第一种方法基于这样的观察：高维空间中的实际数据具有高度相关性和聚类性，因此数据仅占据高维空间的某个子空间。诸如快速映射（Fastmap）[FL 95]、多维标度法[KW 78]、主成分分析[DE 82]和因子分析[Har 67]等算法利用了这一事实，将数据对象转换到某个低维空间，然后可以使用传统的多维索引结构对其进行高效索引。SS树[WJ 96]是一种基于R树的索引结构，它也提出了类似的方法。SS树在低维空间中使用椭球边界区域，在每个目录节点中应用不同的变换。第二种方法基于这样的观察：在大多数高维数据集中，少数维度承载了大部分信息。例如，TV树[LJF 94]以一种方式组织目录，使得目录中仅存储区分数据对象所需的信息。这导致了更高的扇出和更小的目录，从而提高了查询性能。

---

<!-- Footnote -->

Permission to copy without fee all or part of this material is granted provided that the copies are not made or distributed for direct commercial advantage, the VLDB copyright notice and the title of the publication and its date appear, and notice is given that copying is by permission of the Very Large Data Base Endowment. To copy otherwise, or to republish, requires a fee and/or special permission from the Endowment. Proceedings ot the 22nd VLDB Conference Mumbai (Bombay), India, 1996

允许免费复制本材料的全部或部分内容，前提是复制的内容不用于直接商业利益，要显示VLDB版权声明、出版物标题及其日期，并注明复制获得了超大型数据库基金会的许可。否则，进行复制或重新发布需要向该基金会支付费用和/或获得特别许可。第22届VLDB会议论文集，印度孟买，1996年

<!-- Footnote -->

---

For high-dimensional data sets, reducing the dimensionality is an obvious and important possibility for diminishing the dimensionality problem and should be performed whenever possible. In many cases, the data sets resulting from reducing the dimensionality will still have a quite large dimensionality. The remaining dimensions are all relatively important which means that any efficient indexing method must guarantee a good selectivity on all those dimensions. Unfortunately, as we will see in section 2, currently available index structures for spatial data such as the R*-tree ${}^{1}$ do not adequately support an effective indexing of more than five dimensions. Our experiments show that the performance of the R*-tree is rapidly deteriorating when going to higher dimensions. To understand the reason for the performance problems, we carry out a detailed evaluation of the overlap of the bounding boxes in the directory of the R*-tree. Our experiments show that the overlap of the bounding boxes in the directory is rapidly increasing to about ${90}\%$ when increasing the dimensionality to 5 . In subsection 3.3, we provide a detailed explanation of the increasing overlap and show that the high overlap is not an $\mathrm{R}$ -tree specific problem, but a general problem in indexing high-dimensional data.

对于高维数据集，降维是缓解维度问题的一种显而易见且重要的方法，应尽可能进行降维操作。在许多情况下，降维后的数据集维度仍然相当高。剩余的维度都相对重要，这意味着任何高效的索引方法都必须保证在所有这些维度上具有良好的选择性。不幸的是，正如我们将在第2节中看到的，目前用于空间数据的索引结构，如R*树 ${}^{1}$ ，不能充分支持对超过五维数据的有效索引。我们的实验表明，当数据维度增加时，R*树的性能会迅速下降。为了理解性能问题的原因，我们对R*树目录中边界框的重叠情况进行了详细评估。我们的实验表明，当维度增加到5时，目录中边界框的重叠率会迅速增加到约 ${90}\%$ 。在3.3小节中，我们详细解释了重叠率增加的原因，并表明高重叠率不是 $\mathrm{R}$ 树特有的问题，而是高维数据索引中的一个普遍问题。

Based on our observations, we then develop an improved index structure for high-dimensional data,the X-tree (cf. section 3). The main idea of the X-tree is to avoid overlap of bounding boxes in the directory by using a new organization of the directory which is optimized for high-dimensional space. The X-tree avoids splits which would result in a high degree of overlap in the directory. Instead of allowing splits that introduce high overlaps, directory nodes are extended over the usual block size, resulting in so-called supernodes. The supernodes may become large and the linear scan of the large supernodes might seem to be a problem. The alternative, however, would be to introduce high overlap in the directory which leads to a fast degeneration of the filtering selectivity and also makes a sequential search of all subnodes necessary with the additional penalty of many random page accesses instead of a much faster sequential read. The concept of supernodes has some similarity to the idea of oversize shelves [GN 91]. In contrast to supernodes, oversize shelves are data nodes which are attached to internal nodes in order to avoid excessive clipping of large objects. Additionally, oversize shelves are organized as chains of disk pages which cannot be read sequentially.

基于我们的观察，我们为高维数据开发了一种改进的索引结构——X树（参见第3节）。X树的主要思想是通过使用一种针对高维空间优化的目录组织方式，避免目录中边界框的重叠。X树避免了会导致目录中高度重叠的分裂操作。与允许引入高重叠的分裂操作不同，目录节点会扩展到超过通常的块大小，从而形成所谓的超节点。超节点可能会变得很大，对大型超节点进行线性扫描似乎会成为一个问题。然而，另一种选择是在目录中引入高重叠，这会导致过滤选择性迅速下降，并且还需要对所有子节点进行顺序搜索，同时会带来许多随机页面访问的额外开销，而不是更快的顺序读取。超节点的概念与超大货架 [GN 91] 的思想有一些相似之处。与超节点不同的是，超大货架是附加到内部节点的数据节点，以避免对大型对象进行过度裁剪。此外，超大货架被组织成磁盘页面链，无法顺序读取。

We implemented the X-tree index structure and performed a detailed performance evaluation using very large amounts (up to 100 MBytes) of randomly generated as well as real data (point data and extended spatial data). Our experiments show that on high-dimensional data,the X-tree outperforms the TV-tree and the R*-tree by orders of magnitude (cf. section 4). For dimensionality larger than 2, the $\mathrm{X}$ -tree is up to 450 times faster than the ${\mathrm{R}}^{ * }$ -tree and between 4 and 12 times faster than the TV-tree. The X-tree also provides much faster insertion times (about 8 times faster than the R*-tree and about 30 times faster than the TV-tree).

我们实现了X树索引结构，并使用大量（高达100兆字节）随机生成的数据以及真实数据（点数据和扩展空间数据）进行了详细的性能评估。我们的实验表明，在高维数据上，X树的性能比TV树和R*树高出几个数量级（参见第4节）。对于维度大于2的数据， $\mathrm{X}$ 树比 ${\mathrm{R}}^{ * }$ 树快达450倍，比TV树快4到12倍。X树的插入时间也快得多（比R*树快约8倍，比TV树快约30倍）。

<!-- Media -->

<!-- figureText: 40 8 10 12 14 16 dimension Joal Search Time (sec) 35 30 25 20 10 0 2 4 -->

<img src="https://cdn.noedgeai.com/0195c90f-ed1a-7cf0-ae28-0ea5e38bb72d_1.jpg?x=941&y=210&w=667&h=488&r=0"/>

Figure 1: Performance of the R-tree Depending on the Dimension (Real Data)

图1：R树性能随维度的变化（真实数据）

<!-- Media -->

## 2. Problems of (R-tree-based) Index Structures in High-Dimensional Space

## 2. 高维空间中（基于R树）索引结构的问题

In our performance evaluation of the R*-tree, we found that the performance deteriorates rapidly when going to higher dimensions (cf. Figure 1). Effects such as a lower fanout in higher dimensions do not explain this fact. In trying to understand the effects that lead to the performance problems, we performed a detailed evaluation of important characteristics of the R*-tree and found that the overlap in the directory is increasing very rapidly with growing dimensionality of the data. Overlap in the directory directly corresponds to the query performance since even for simple point queries multiple paths have to be followed. Overlap in the directory is a relatively imprecise term and there is no generally accepted definition especially for the high-dimensional case. In the following, we therefore provide definitions of overlap.

在对R*树的性能评估中，我们发现当数据维度增加时，其性能会迅速下降（参见图1）。高维情况下扇出较低等因素并不能解释这一现象。为了理解导致性能问题的因素，我们对R*树的重要特征进行了详细评估，发现随着数据维度的增加，目录中的重叠率会迅速上升。目录中的重叠率直接影响查询性能，因为即使是简单的点查询也需要遍历多条路径。目录中的重叠是一个相对不精确的术语，尤其是在高维情况下，目前还没有普遍接受的定义。因此，下面我们将给出重叠的定义。

### 2.1 Definition of Overlap

### 2.1 重叠的定义

Intuitively, overlap is the percentage of the volume that is covered by more than one directory hyperrectangle. This intuitive definition of overlap is directly correlated to the query performance since in processing queries, overlap of directory nodes results in the necessity to follow multiple paths, even for point queries.

直观地说，重叠是指被多个目录超矩形覆盖的体积百分比。这种直观的重叠定义与查询性能直接相关，因为在处理查询时，目录节点的重叠会导致即使是点查询也需要遍历多条路径。

---

<!-- Footnote -->

1. According to [BKSS 90], the R*-tree provides a consistently better performance than the R-tree [Gut 84] and ${\mathrm{R}}^{\top }$ -tree [SRF 87] over a wide range of data sets and query types. In the rest of this paper, we therefore restrict ourselves to the R*-tree.

1. 根据 [BKSS 90] 的研究，在广泛的数据集和查询类型中，R*树的性能始终优于R树 [Gut 84] 和 ${\mathrm{R}}^{\top }$ 树 [SRF 87] 。因此，在本文的其余部分，我们将主要关注R*树。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: 100% 100% 90% 80% 50% 40% dimension b. Weighted Overlap (Real Data) 80% 20% 0% 6 8 10 12 14 16 dimension a. Overlap (Uniformly Distributed Data) -->

<img src="https://cdn.noedgeai.com/0195c90f-ed1a-7cf0-ae28-0ea5e38bb72d_2.jpg?x=186&y=208&w=1407&h=471&r=0"/>

Figure 2: Overlap of R*-tree Directory Nodes depending on the Dimensionality

图2：R*树目录节点的重叠率随维度的变化

<!-- Media -->

## Definition 1a (Overlap)

## 定义1a（重叠）

The overlap of an R-tree node is the percentage of space covered by more than one hyperrectangle. If the R-tree node contains $n$ hyperrectangles $\left\{  {{R}_{1},\ldots {R}_{n}}\right\}$ ,the overlap may formally be defined as

R树节点的重叠度是被一个以上超矩形覆盖的空间所占的百分比。如果R树节点包含$n$个超矩形$\left\{  {{R}_{1},\ldots {R}_{n}}\right\}$，则重叠度可以正式定义为

$$
\text{ Overlap } = \frac{\begin{Vmatrix}\mathop{\bigcup }\limits_{{i,j \in  \{ 1\ldots n\} ,i \neq  j}}\left( {R}_{i} \cap  {R}_{j}\right) \end{Vmatrix}}{\begin{Vmatrix}\mathop{\bigcup }\limits_{{i \in  \{ 1\ldots n\} }}{R}_{i}\end{Vmatrix}}.1
$$

The amount of overlap measured in definition 1a is related to the expected query performance only if the query objects (points, hyperrectangles) are distributed uniformly. A more accurate definition of overlap needs to take the actual distribution of queries into account. Since it is impossible to determine the distribution of queries in advance, in the following we will use the distribution of the data as an estimation for the query distribution. This seems to be reasonable for high-dimensional data since data and queries are often clustered in some areas, whereas other areas are virtually empty. Overlap in highly populated areas is much more critical than overlap in areas with a low population. In our second definition of overlap, the overlapping areas are therefore weighted with the number of data objects that are located in the area.

只有当查询对象（点、超矩形）均匀分布时，定义1a中测量的重叠量才与预期的查询性能相关。更准确的重叠度定义需要考虑查询的实际分布。由于无法提前确定查询的分布，在下面我们将使用数据的分布来估计查询分布。对于高维数据来说，这似乎是合理的，因为数据和查询通常会聚集在某些区域，而其他区域实际上是空的。人口密集区域的重叠比人口稀少区域的重叠更为关键。因此，在我们的第二个重叠度定义中，重叠区域会根据该区域内的数据对象数量进行加权。

## Definition 1b (Weighted Overlap)

## 定义1b（加权重叠度）

The weighted overlap of an $\mathrm{R}$ -tree node is the percentage of data objects that fall in the overlapping portion of the space. More formally,

$\mathrm{R}$树节点的加权重叠度是落在空间重叠部分的数据对象所占的百分比。更正式地说，

<!-- Media -->

<!-- figureText: WeightedOverlap $= \frac{\left| \left\{  p \mid  p \in  \mathop{\bigcup }\limits_{{i,j \in  \{ 1\ldots n\} ,i \neq  j}}\left( {R}_{i} \cap  {R}_{j}\right) \right\}  \right| }{\left| \left\{  p \mid  p \in  \mathop{\bigcup }\limits_{{i \in  \{ 1\ldots n\} }}{R}_{i}\right\}  \right| }.$ -->

<img src="https://cdn.noedgeai.com/0195c90f-ed1a-7cf0-ae28-0ea5e38bb72d_2.jpg?x=179&y=1819&w=687&h=208&r=0"/>

<!-- Media -->

In definition 1a, overlap occurring at any point of space equally contributes to the overall overlap even if only few data objects fall within the overlapping area. If the query points are expected to be uniformly distributed over the data space, definition 1a is an appropriate measure which determines the expected query performance. If the distribution of queries corresponds to the distribution of the data and is nonuniform, definition 1b corresponds to the expected query performance and is therefore more appropriate. Depending on the query distribution, we have to choose the appropriate definition.

在定义1a中，即使只有少数数据对象落在重叠区域内，空间中任何一点的重叠对总体重叠度的贡献都是相等的。如果预计查询点在数据空间中均匀分布，定义1a是一种合适的度量方法，它可以确定预期的查询性能。如果查询的分布与数据的分布相对应且是非均匀的，定义1b与预期的查询性能相对应，因此更为合适。根据查询分布，我们必须选择合适的定义。

So far, we have only considered overlap to be any portion of space that is covered by more than one hyperrectan-gle. In practice however, it is very important how many hyperrectangles overlap at a certain portion of the space. The so-called multi-overlap of an $R$ -tree node is defined as the sum of overlapping volumes multiplied by the number of overlapping hyperrectangles relative to the overall volume of the considered space.

到目前为止，我们只考虑了重叠是指被一个以上超矩形覆盖的任何空间部分。然而在实践中，在空间的某一特定部分有多少个超矩形重叠是非常重要的。所谓的$R$树节点的多重重叠度定义为重叠体积之和乘以重叠超矩形的数量，再相对于所考虑空间的总体积。

In Figure 3, we show a two-dimensional example of the overlap according to definition 1a and the corresponding multi-overlap. The weighted overlap and weighted multi-overlap (not shown in the figure) would correspond to the areas weighted by the number of data objects that fall within the areas.

在图3中，我们展示了一个根据定义1a的二维重叠示例以及相应的多重重叠。加权重叠和加权多重重叠（图中未显示）将对应于由落在这些区域内的数据对象数量加权的区域。

<!-- Media -->

<img src="https://cdn.noedgeai.com/0195c90f-ed1a-7cf0-ae28-0ea5e38bb72d_2.jpg?x=911&y=1750&w=685&h=258&r=0"/>

Figure 3: Overlap and Multi-Overlap of 2-dimensional data

图3：二维数据的重叠和多重重叠

<!-- Media -->

---

<!-- Footnote -->

1. $\parallel A\parallel$ denotes the volume covered by $A$ .

1. $\parallel A\parallel$表示$A$所覆盖的体积。

2. $\left| A\right|$ denotes the number of data elements contained in $\mathrm{A}$

2. $\left| A\right|$表示$\mathrm{A}$中包含的数据元素数量

<!-- Footnote -->

---

### 2.2 Experimental Evaluation of Overlap in R*-tree Directories

### 2.2 R*树目录中重叠度的实验评估

In this subsection, we empirically evaluate the development of the overlap in the R*-tree depending on the dimensionality. For the experiments, we use the implementation of the R*-tree according to [BKSS 90]. The data used for the experiments are constant size databases of uniformly distributed and real data. The real data are Fourier vectors which are used in searching for similarly shaped polygons. The overlap curves presented in Figure 2 show the average overlap of directory nodes according to definition 1 . In averaging the node overlaps, we used all directory levels except the root level since the root page may only contain a few hy-perrectangles, which causes a high variance of the overlap in the root node.

在本小节中，我们根据维度对R*树中重叠度的发展进行实证评估。对于实验，我们使用根据[BKSS 90]实现的R*树。实验使用的数据是均匀分布的常量大小数据库和真实数据。真实数据是用于搜索形状相似多边形的傅里叶向量。图2中呈现的重叠曲线显示了根据定义1的目录节点的平均重叠度。在对节点重叠度进行平均时，我们使用了除根级别之外的所有目录级别，因为根页面可能只包含几个超矩形，这会导致根节点的重叠度有很大的方差。

In Figure 2a, we present the overlap curves of R*-trees generated from 6 MBytes of uniformly distributed point data. As expected, for a uniform distribution overlap and weighted overlap (definition 1a and 1b) provide the same results. For dimensionality larger than two, the overlap (cf. Figure 2a) increases rapidly to approach ${100}\%$ for dimensionality larger than ten. This means that even for point queries on ten or higher dimensional data in almost every directory node at least two subnodes have to be accessed. For real data (cf. Figure 2b), the increase of the overlap is even more remarkable. The weighted overlap increases to about ${80}\%$ for dimensionality 4 and approaches ${100}\%$ for dimensionality larger than 6 .

在图2a中，我们展示了由6兆字节均匀分布的点数据生成的R*树的重叠曲线。正如预期的那样，对于均匀分布，重叠度和加权重叠度（定义1a和1b）提供了相同的结果。对于维度大于2的情况，重叠度（参见图2a）迅速增加，在维度大于10时接近${100}\%$。这意味着即使对于十维或更高维数据的点查询，几乎每个目录节点中至少要访问两个子节点。对于真实数据（参见图2b），重叠度的增加更为显著。加权重叠度在维度为4时增加到约${80}\%$，在维度大于6时接近${100}\%$。

### 3.The $X$ -tree

### 3. $X$树

The X-tree (eXtended node tree) is a new index structure supporting efficient query processing of high-dimensional data. The goal is to support not only point data but also extended spatial data and therefore, the X-tree uses the concept of overlapping regions. From the insight obtained in the previous section, it is clear that we have to avoid overlap in the directory in order to improve the indexing of high-dimensional data. The X-tree therefore avoids overlap whenever it is possible without allowing the tree to degenerate; otherwise, the X-tree uses extended variable size directory nodes, so-called supernodes. In addition to providing a directory organization which is suitable for high-dimensional data, the X-tree uses the available main memory more efficiently (in comparison to using a cache).

X树（扩展节点树，eXtended node tree）是一种支持高效处理高维数据查询的新型索引结构。其目标不仅是支持点数据，还支持扩展空间数据，因此，X树采用了重叠区域的概念。从上一节的分析可知，为了改进高维数据的索引，我们必须避免目录中的重叠。因此，X树会在不使树退化的前提下尽可能避免重叠；否则，X树会使用扩展可变大小的目录节点，即所谓的超级节点。除了提供适合高维数据的目录组织方式外，X树还能更有效地利用可用的主内存（与使用缓存相比）。

The X-tree may be seen as a hybrid of a linear array-like and a hierarchical R-tree-like directory. It is well established that in low dimensions the most efficient organization of the directory is a hierarchical organization. The reason is that the selectivity in the directory is very high which means that, e.g. for point queries, the number of required page accesses directly corresponds to the height of the tree. This, however, is only true if there is no overlap between directory rectangles which is the case for a low dimensionality. It is also reasonable, that for very high dimensionality a linear organization of the directory is more efficient. The reason is that due to the high overlap, most of the directory if not the whole directory has to be searched anyway. If the whole directory has to be searched, a linearly organized directory needs less space ${}^{1}$ and may be read much faster from disk than a block-wise reading of the directory. For medium dimensionality, an efficient organization of the directory would probably be partially hierarchical and partially linear. The problem is to dynamically organize the tree such that portions of the data which would produce high overlap are organized linearly and those which can be organized hierarchically without too much overlap are dynamically organized in a hierarchical form. The algorithms used in the $\mathrm{X}$ -tree are designed to automatically organize the directory as hierarchical as possible, resulting in a very efficient hybrid organization of the directory.

X树可以看作是类似线性数组和类似分层R树的目录的混合体。众所周知，在低维度下，最有效的目录组织方式是分层组织。原因在于目录的选择性非常高，这意味着，例如对于点查询，所需的页面访问次数直接对应于树的高度。然而，这仅在目录矩形之间没有重叠的情况下成立，而低维度时就是这种情况。同样合理的是，对于非常高的维度，目录的线性组织方式更为高效。原因是由于高度重叠，即使不是整个目录，大部分目录也必须进行搜索。如果必须搜索整个目录，线性组织的目录所需的空间更少${}^{1}$，并且从磁盘读取的速度比按块读取目录要快得多。对于中等维度，目录的有效组织方式可能是部分分层和部分线性的。问题在于如何动态地组织树，使得会产生高度重叠的数据部分采用线性组织，而那些可以在不过多重叠的情况下进行分层组织的数据部分则动态地采用分层形式组织。$\mathrm{X}$树中使用的算法旨在自动将目录组织成尽可能分层的形式，从而实现非常高效的混合目录组织。

<!-- Media -->

<table><tr><td>MBR ${}_{0}$ | SplitHistory ${}_{0}$ | Ptr ${}_{0}$ )MBR ${}_{n - 1}$ SplitHistory ${}_{n - 1}$ Pt ${}_{n - 1}$</td></tr></table>

<table><tbody><tr><td>最小边界矩形（MBR） ${}_{0}$ | 分割历史（SplitHistory） ${}_{0}$ | 指针（Ptr） ${}_{0}$ )最小边界矩形（MBR） ${}_{n - 1}$ 分割历史（SplitHistory） ${}_{n - 1}$ 指针（Pt） ${}_{n - 1}$</td></tr></tbody></table>

Figure 4: Structure of a Directory Node

图4：目录节点的结构

<!-- figureText: root -->

<img src="https://cdn.noedgeai.com/0195c90f-ed1a-7cf0-ae28-0ea5e38bb72d_3.jpg?x=962&y=193&w=594&h=300&r=0"/>

CNormal Directory Nodes

普通目录节点

Figure 5: Structure of the X-tree

图5：X树的结构

<!-- Media -->

### 3.1 Structure of the $X$ -tree

### 3.1 $X$树的结构

The overall structure of the X-tree is presented in Figure 5. The data nodes of the X-tree contain rectilinear minimum bounding rectangles (MBRs) together with pointers to the actual data objects, and the directory nodes contain MBRs together with pointers to sub-MBRs (cf. Figure 5). The $\mathrm{X}$ -tree consists of three different types of nodes: data nodes, normal directory nodes, and supernodes. Supernodes are large directory nodes of variable size (a multiple of the usual block size). The basic goal of supernodes is to avoid splits in the directory that would result in an inefficient directory structure. The alternative to using larger node sizes are highly overlapping directory nodes which would require to access most of the son nodes during the search process. This, however, is more inefficient than linearly scanning the larger supernode. Note that the X-tree is completely different from an R-tree with a larger block size since the X-tree only consists of larger nodes where actually necessary. As a result, the structure of the X-tree may be rather heterogeneous as indicated in Figure 5. Due to the fact that the overlap is increasing with the dimension, the internal structure of the X-tree is also changing with increasing dimension. In Figure 5, three examples of X-trees containing data of different dimensionality are shown. As expected, the number and size of supernodes increases with the dimension. For generating the examples, the block size has been artificially reduced to obtain a drawable fanout. Due to the increasing number and size of supernodes, the height of the X-tree which corresponds to the number of page accesses necessary for point queries is decreasing with increasing dimension.

图5展示了X树的整体结构。X树的数据节点包含直线最小边界矩形（MBRs，Minimum Bounding Rectangles）以及指向实际数据对象的指针，而目录节点包含MBR以及指向子MBR的指针（参见图5）。$\mathrm{X}$树由三种不同类型的节点组成：数据节点、普通目录节点和超级节点。超级节点是可变大小的大型目录节点（通常是常规块大小的倍数）。超级节点的基本目标是避免目录中出现分裂，因为分裂会导致目录结构效率低下。使用更大节点大小的替代方案是高度重叠的目录节点，这在搜索过程中需要访问大多数子节点。然而，这比线性扫描更大的超级节点效率更低。请注意，X树与块大小更大的R树完全不同，因为X树仅在实际需要的地方使用更大的节点。因此，如图5所示，X树的结构可能相当不均匀。由于重叠会随着维度的增加而增加，X树的内部结构也会随着维度的增加而变化。图5展示了三个包含不同维度数据的X树示例。正如预期的那样，超级节点的数量和大小会随着维度的增加而增加。为了生成这些示例，人为地减小了块大小以获得可绘制的扇出。由于超级节点的数量和大小不断增加，与点查询所需的页面访问次数相对应的X树高度会随着维度的增加而降低。

---

<!-- Footnote -->

1. In comparison to a hierarchically organized directory, a linearly organized directory only consists of the concatenation of the nodes on the lowest level of the corresponding hierarchical directory and is therefore much smaller.

1. 与分层组织的目录相比，线性组织的目录仅由相应分层目录最低层的节点串联组成，因此要小得多。

<!-- Footnote -->

---

Supernodes are created during insertion only if there is no other possibility to avoid overlap. In many cases, the creation or extension of supernodes may be avoided by choosing an overlap-minimal split axis (cf. subsection 3.3). For a fast determination of the overlap-minimal split, additional information is necessary which is stored in each of the directory nodes (cf. Figure 4). If enough main memory is available, supernodes are kept in main memory. Otherwise, the nodes which have to be replaced are determined by a priority function which depends on level, type (normal node or su-pernode), and size of the nodes. According to our experience,the priority function ${c}_{t} \cdot$ type $+ {c}_{l} \cdot$ level $+ {c}_{s} \cdot$ size with ${c}_{t} \gg  {c}_{l} \gg  {c}_{s}$ is a good choice for practical purposes. Note that the storage utilization of supernodes is higher than the storage utilization of normal directory nodes. For normal directory nodes, the expected storage utilization for uniformly distributed data is about ${66}\%$ . For supernodes of size $m \cdot$ BlockSize,the expected storage utilization can be determined as the average of the following two extreme cases: Assuming a certain amount of data occupies $X \cdot  m$ blocks for a maximally filled node. Then the same amount of data requires $X \cdot  \frac{{m}^{2}}{m - 1}$ blocks when using a minimally filled node. On the average, a supernode storing the same amount of data requires $\left( {X \cdot  m + X \cdot  \frac{{m}^{2}}{m - 1}}\right) /2 = X\left( \frac{m\left( {{2m} - 1}\right) }{{2m} - 2}\right)$ blocks. From that, we obtain a storage utilization of $m/\left( \frac{m\left( {{2m} - 1}\right) }{{2m} - 2}\right)  = \frac{2 \cdot  m - 2}{2 \cdot  m - 1}$ which for large $m$ is considerably higher than ${66}\%$ . For $m = 5$ ,for example,the storage utilization is about ${88}\%$ .

只有在没有其他方法可以避免重叠时，才会在插入过程中创建超级节点。在许多情况下，可以通过选择重叠最小的分割轴来避免创建或扩展超级节点（参见3.3小节）。为了快速确定重叠最小的分割，需要额外的信息，这些信息存储在每个目录节点中（参见图4）。如果有足够的主内存，超级节点会保存在主内存中。否则，需要替换的节点由一个优先级函数确定，该函数取决于节点的级别、类型（普通节点或超级节点）和大小。根据我们的经验，优先级函数${c}_{t} \cdot$类型 $+ {c}_{l} \cdot$级别 $+ {c}_{s} \cdot$大小（其中${c}_{t} \gg  {c}_{l} \gg  {c}_{s}$）在实际应用中是一个不错的选择。请注意，超级节点的存储利用率高于普通目录节点的存储利用率。对于普通目录节点，均匀分布数据的预期存储利用率约为${66}\%$。对于大小为$m \cdot$块大小的超级节点，预期存储利用率可以通过以下两种极端情况的平均值来确定：假设一定量的数据在最大填充节点中占用$X \cdot  m$个块。那么，当使用最小填充节点时，相同数量的数据需要$X \cdot  \frac{{m}^{2}}{m - 1}$个块。平均而言，存储相同数量数据的超级节点需要$\left( {X \cdot  m + X \cdot  \frac{{m}^{2}}{m - 1}}\right) /2 = X\left( \frac{m\left( {{2m} - 1}\right) }{{2m} - 2}\right)$个块。由此，我们得到存储利用率为$m/\left( \frac{m\left( {{2m} - 1}\right) }{{2m} - 2}\right)  = \frac{2 \cdot  m - 2}{2 \cdot  m - 1}$，对于较大的$m$，该值明显高于${66}\%$。例如，对于$m = 5$，存储利用率约为${88}\%$。

<!-- Media -->

<!-- figureText: D=4: D=8: D=32: -->

<img src="https://cdn.noedgeai.com/0195c90f-ed1a-7cf0-ae28-0ea5e38bb72d_4.jpg?x=209&y=1684&w=628&h=360&r=0"/>

Figure 6: Various Shapes of the X-tree in different dimensions

图6：不同维度下X树的各种形状

<!-- Media -->

There are two interesting special cases of the X-tree: (1) none of the directory nodes is a supernode and (2) the directory consists of only one large supernode (root). In the first case, the X-tree has a completely hierarchical organization of the directory and is therefore similar to an R-tree. This case may occur for low dimensional and non-overlapping data. In the second case, the directory of the X-tree is basically one root-supernode which contains the lowest directory level of the corresponding R-tree. The performance therefore corresponds to the performance of a linear directory scan. This case will only occur for high-dimensional or highly overlapping data where the directory would have to be completely searched anyway. The two cases also correspond to the two extremes for the height of the tree and the directory size. In case of a completely hierarchical organization, the height and size of the directory basically correspond to that of an R-tree. In the root-supernode case, the size of the directory linearly depends on the dimension

X树有两种有趣的特殊情况：（1）没有一个目录节点是超级节点；（2）目录仅由一个大的超级节点（根节点）组成。在第一种情况下，X树的目录具有完全分层的组织结构，因此与R树类似。这种情况可能发生在低维且不重叠的数据中。在第二种情况下，X树的目录基本上是一个根超级节点，它包含了相应R树的最低目录层。因此，其性能与线性目录扫描的性能相当。这种情况只会发生在高维或高度重叠的数据中，在这种情况下，无论如何都必须对目录进行全面搜索。这两种情况也对应于树的高度和目录大小的两个极端。在完全分层组织的情况下，目录的高度和大小基本上与R树的高度和大小相对应。在根超级节点的情况下，目录的大小线性依赖于维度

$$
\operatorname{DirSize}\left( D\right)  = \frac{\text{ Database Size }}{\text{ Block Size } \cdot  \text{ Storage Util }} \cdot  2 \cdot  \text{ Bytes Float } \cdot  L
$$

For 1 GBytes of 16-dimensional data, a block size of 4 KBytes,a storage utilization of ${66}\%$ for data nodes,and 4 bytes per float, the size of the directory is about 44 MBytes for the root-supernode in contrast to about ${72}\mathrm{{MBytes}}$ for the completely hierarchical directory.

对于1GB的16维数据，块大小为4KB，数据节点的存储利用率为${66}\%$，每个浮点数为4字节，根超级节点的目录大小约为44MB，而完全分层目录的大小约为${72}\mathrm{{MBytes}}$。

### 3.2 Algorithms

### 3.2 算法

The most important algorithm of the X-tree is the insertion algorithm. The insertion algorithm determines the structure of the X-tree which is a suitable combination of a hierarchical and a linear structure. The main objective of the algorithm is to avoid splits which would produce overlap. The algorithm (cf. Figure 7) first determines the MBR in which to insert the data object and recursively calls the insertion algorithm to actually insert the data object into the corresponding node. If no split occurs in the recursive insert, only the size of the corresponding MBRs has to be updated. In case of a split of the subnode, however, an additional MBR has to be added to the current node which might cause an overflow of the node. In this case, the current node calls the split algorithm (cf. Figure 8) which first tries to find a split of the node based on the topological and geometric properties of the MBRs. Topological and geometric properties of the MBRs are for example dead-space partitioning, extension of MBRs, etc. The heuristics of the R*-tree [BKSS 90] split algorithm are an example for a topological split to be used in this step. If the topological split however results in high overlap, the split algorithm tries next to find an overlap-minimal split which can be determined based on the split history (cf. subsection 3.3). In subsection 3.3, we show that for point data there always exists an overlap-free split. The partitioning of the MBRs resulting from the overlap-minimal split, however, may result in underfilled nodes which is unacceptable since it leads to a degeneration of the tree and also deteriorates the space utilization. If the number of MBRs in one of the partitions is below a given threshold, the split algorithm terminates without providing a split. In this case, the current node is extended to become a super-node of twice the standard block size. If the same case occurs for an already existing supernode, the supernode is extended by one additional block. Obviously, supernodes are only created or extended if there is no possibility to find a suitable hierarchical structure of the directory. If a supernode is created or extended, there may be not enough contiguous space on disk to sequentially store the supernode. In this case, the disk manager has to perform a local reorganization. Since supernodes are created or extended in main memory, the local reorganization is only necessary when writing back the supernodes on secondary storage which does not occur frequently.

X树最重要的算法是插入算法。插入算法决定了X树的结构，它是分层结构和线性结构的合适组合。该算法的主要目标是避免产生重叠的分裂。该算法（参见图7）首先确定要插入数据对象的最小边界矩形（MBR），并递归调用插入算法将数据对象实际插入到相应的节点中。如果在递归插入过程中没有发生分裂，只需更新相应MBR的大小。然而，如果子节点发生分裂，则必须向当前节点添加一个额外的MBR，这可能会导致节点溢出。在这种情况下，当前节点调用分裂算法（参见图8），该算法首先尝试根据MBR的拓扑和几何属性找到节点的分裂方式。MBR的拓扑和几何属性例如有死空间划分、MBR的扩展等。R*树[BKSS 90]分裂算法的启发式方法就是在这一步中使用的拓扑分裂的一个例子。然而，如果拓扑分裂导致高度重叠，分裂算法接下来会尝试找到一个基于分裂历史确定的最小重叠分裂（参见3.3小节）。在3.3小节中，我们表明对于点数据，总是存在无重叠的分裂。然而，由最小重叠分裂产生的MBR划分可能会导致节点填充不足，这是不可接受的，因为这会导致树的退化并降低空间利用率。如果其中一个分区中的MBR数量低于给定阈值，分裂算法将终止，不提供分裂。在这种情况下，当前节点将扩展为标准块大小两倍的超级节点。如果对于已经存在的超级节点发生同样的情况，超级节点将增加一个额外的块。显然，只有在无法找到合适的目录分层结构时才会创建或扩展超级节点。如果创建或扩展了超级节点，磁盘上可能没有足够的连续空间来顺序存储超级节点。在这种情况下，磁盘管理器必须进行局部重组。由于超级节点是在主内存中创建或扩展的，局部重组仅在将超级节点写回二级存储时才需要，这种情况并不经常发生。

<!-- Media -->

int X_DirectoryNode: : insert(DataObject obj, X_Node **new_node)

int X_DirectoryNode::insert(DataObject obj, X_Node **new_node)

---

\{

			SET_OF_MBR *s1, *s2;

			SET_OF_MBR *s1, *s2;

			x_Node *follow, *new_son;

			x_Node *follow, *new_son;

			int return value;

			int return value;

			follow = choose_subtree(obj); // choose a son node to insert obi into

			follow = choose_subtree(obj); // 选择一个子节点来插入对象

			return value = follow->insert(obj, &new_son); // insert obj into subtree

			return value = follow->insert(obj, &new_son); // 将对象插入子树

			update_mbr(follow->calc_mbr(   )); // update MBR of old son node

			update_mbr(follow->calc_mbr(   )); // 更新旧子节点的MBR

			if (return_value == SPLIT) \{

			if (return_value == SPLIT) {

							add_mbr(new_son->calc_mbr(   )); // insert mbr of new son node into current node

							add_mbr(new_son->calc_mbr(   )); // 将新子节点的MBR插入当前节点

							if (num_of_mbrs(   ) > CAPACITY) \{ // overflow occurs

							if (num_of_mbrs(   ) > CAPACITY) { // 发生溢出

												if (split(mbrs, s1, s2) == TRUE)\{

												if (split(mbrs, s1, s2) == TRUE){

																	// topological or overlap-minimal split was successfull

																	// 拓扑或最小重叠分裂成功

																	set_mbrs(s1);

																	设置成员(s1);

																	*new_node = new X_DirectoryNode(s2);

																	*新节点 = new X_目录节点(s2);

																	return SPLIT;

																	返回 分裂;

												\}

													! Ise // there is no good split

													! Ise // 没有合适的分裂

												\{

																	*new_node = new X_SuperNode(   );

																	*新节点 = new X_超级节点(   );

																	(*new_node) ->set_mbrs(mbrs) ;

																	(*新节点) ->设置成员(成员) ;

																	return SUPERNODE;

																	返回 超级节点;

							\} \}

			\} else if (return_value == SUPERNODE) \{ // node 'follow' becomes a supernode

			} 否则如果 (返回值 == 超级节点) { // 节点 '跟随' 变为超级节点

			- remove_son(follow);

			- 移除子节点(跟随);

							insert_son(new_son);

							插入子节点(新子节点);

			\}

			return NO_SPLIT;

			返回 不分裂;

\}

---

Figure 7: X-tree Insertion Algorithm for Directory Nodes

图 7：X 树目录节点的插入算法

<!-- Media -->

For point data, overlap in the X-tree directory may only occur if the overlap induced by the topological split is below a threshold overlap value (MAX_OVERLAP). In that case, the overlap-minimal split and the possible creation of a su-pernode do not make sense. The maximum overlap value which is acceptable is basically a system constant and depends on the page access time $\left( {T}_{IO}\right)$ ,the time to transfer a block from disk into main memory $\left( {T}_{Tr}\right)$ ,and the CPU time necessary to process a block $\left( {T}_{CPU}\right)$ . The maximum overlap value $\left( {\operatorname{Max}{O}^{1}}\right)$ may be determined approximately by the balance between reading a supernode of size 2*BlockSize and reading 2 blocks with a probability of MaxO and one block with a probability of (1-MaxO). This estimation is only correct for the most simple case of initially creating a supernode. It does not take the effect of further splits into account. Nevertheless, for practical purposes the following equation provides a good estimation:

对于点数据，只有当拓扑分裂引起的重叠低于阈值重叠值（最大重叠）时，X 树目录中才可能出现重叠。在这种情况下，最小重叠分裂和可能创建超级节点就没有意义了。可接受的最大重叠值基本上是一个系统常量，它取决于页面访问时间 $\left( {T}_{IO}\right)$、将一个块从磁盘传输到主内存的时间 $\left( {T}_{Tr}\right)$ 以及处理一个块所需的 CPU 时间 $\left( {T}_{CPU}\right)$。最大重叠值 $\left( {\operatorname{Max}{O}^{1}}\right)$ 可以通过读取大小为 2 * 块大小的超级节点和以 MaxO 的概率读取 2 个块以及以 (1 - MaxO) 的概率读取 1 个块之间的平衡来近似确定。这种估计仅在最初创建超级节点的最简单情况下是正确的。它没有考虑进一步分裂的影响。尽管如此，出于实际目的，以下方程提供了一个很好的估计：

$$
{MaxO} \cdot  2 \cdot  \left( {{T}_{IO} + {T}_{Tr} + {T}_{CPU}}\right)  + \left( {1 - {MaxO}}\right)  \cdot  \left( {{T}_{IO} + {T}_{Tr} + {T}_{CPU}}\right) 
$$

$$
 = {T}_{IO} + 2 \cdot  \left( {{T}_{Tr} + {T}_{CPU}}\right) 
$$

$$
 \Rightarrow  \operatorname{Max}O = \frac{{T}_{Tr} + {T}_{CPU}}{{T}_{IO} + {T}_{Tr} + {T}_{CPU}}
$$

For realistic system values measured in our experiments $\left( {{T}_{IO} = {20}\mathrm{\;{ms}},{T}_{Tr} = 4\mathrm{\;{ms}},{T}_{CPU} = 1\mathrm{\;{ms}}}\right)$ ,the resulting $\operatorname{Max}O$ value is ${20}\%$ . Note that in the above formula,the fact that the probability of a node being in main memory is increasing due to the decreasing directory size in case of using the supernode has not yet been considered. The other constant of our algorithm (MIN_FANOUT) is the usual minimum fanout value of a node which is similar to the corresponding value used in other index structures. An appropriate value of MIN_FANOUT is between 35% and 45%.

对于我们实验中测量的实际系统值 $\left( {{T}_{IO} = {20}\mathrm{\;{ms}},{T}_{Tr} = 4\mathrm{\;{ms}},{T}_{CPU} = 1\mathrm{\;{ms}}}\right)$，得到的 $\operatorname{Max}O$ 值为 ${20}\%$。请注意，在上述公式中，尚未考虑在使用超级节点的情况下，由于目录大小减小，节点位于主内存中的概率增加这一事实。我们算法的另一个常量（最小扇出）是节点的通常最小扇出值，这与其他索引结构中使用的相应值类似。最小扇出的合适值在 35% 到 45% 之间。

The algorithms to query the X-tree (point, range, and nearest neighbor queries) are similar to the algorithms used in the R*-tree since only minor changes are necessary in accessing supernodes. The delete and update operations are also simple modifications of the corresponding ${\mathrm{R}}^{ * }$ -tree algorithms. The only difference occurs in case of an underflow of a supernode. If the supernode consists of two blocks, it is converted to a normal directory node. Otherwise, that is if the supernode consists of more than two blocks, we reduce the size of the supernode by one block. The update operation can be seen as a combination of a delete and an insert operation and is therefore straightforward.

查询 X 树的算法（点查询、范围查询和最近邻查询）与 R* 树中使用的算法类似，因为访问超级节点时只需要进行微小的更改。删除和更新操作也是相应 ${\mathrm{R}}^{ * }$ 树算法的简单修改。唯一的区别发生在超级节点下溢的情况下。如果超级节点由两个块组成，则将其转换为普通目录节点。否则，即如果超级节点由两个以上的块组成，我们将超级节点的大小减少一个块。更新操作可以看作是删除和插入操作的组合，因此很直接。

---

<!-- Footnote -->

1. Max $O$ is the probability that we have to access both son nodes because of overlap during the search.

1. 最大 $O$ 是在搜索过程中由于重叠而必须访问两个子节点的概率。

<!-- Footnote -->

---

<!-- Media -->

---

bool X_DirectoryNode::split(SET_OF_MBR *in, SET_OF_MBR *out1, SET_OF_MBR *out2)

布尔型 X_DirectoryNode::split(最小边界矩形集合（SET_OF_MBR） *in, 最小边界矩形集合（SET_OF_MBR） *out1, 最小边界矩形集合（SET_OF_MBR） *out2)

\{

	SET_OF_MBR t1, t2;

	最小边界矩形集合（SET_OF_MBR） t1, t2;

	MBR r1, r2;

	最小边界矩形（MBR） r1, r2;

	// first try topological split, resulting in two sets of MBRs t1 and t2

	// 首先尝试拓扑分割，得到两组最小边界矩形集合 t1 和 t2

	topological_split(in, t1, t2);

	拓扑分割（topological_split）(in, t1, t2);

	r1 = t1->calc_mbr(   ); r2 = t2->calc_mbr(   );

	r1 = t1->计算最小边界矩形（calc_mbr）(   ); r2 = t2->计算最小边界矩形（calc_mbr）(   );

	// test for overlap

	// 测试重叠情况

	if (overlap(r1,r2) > MAX_OVERLAP)

	if (重叠（overlap）(r1,r2) > 最大重叠值（MAX_OVERLAP）)

	\{

		// topological split fails -> try overlap minimal split

		// 拓扑分割失败 -> 尝试最小重叠分割

		overlap_minimal_split(in, t1, t2);

		最小重叠分割（overlap_minimal_split）(in, t1, t2);

		// test for unbalanced nodes

		// 测试节点是否不平衡

		if (t1->num_of_mbrs(   ) < MIN_FANOUT || t2->num_of_mbrs(   ) < MIN_FANOUT)

		if (t1->最小边界矩形数量（num_of_mbrs）(   ) < 最小扇出（MIN_FANOUT） || t2->最小边界矩形数量（num_of_mbrs）(   ) < 最小扇出（MIN_FANOUT）)

				// overlap-minimal split also fails (-> caller has to create supernode)

				// 最小重叠分割也失败（-> 调用者必须创建超级节点）

				return FALSE;

				return 假（FALSE）;

	\}

	*out1 = t1; *out2 = t2;

	*out1 = t1; *out2 = t2;

	return TRUE;

	return 真（TRUE）;

\}

---

Figure 8: X-tree Split Algorithm for Directory Nodes

图8：目录节点的X树分裂算法

<!-- Media -->

### 3.3 Determining the Overlap-Minimal Split

### 3.3 确定最小重叠分裂

For determining an overlap-minimal split of a directory node, we have to find a partitioning of the MBRs in the node into two subsets such that the overlap of the minimum bounding hyperrectangles of the two sets is minimal. In case of point data, it is always possible to find an overlap-free split, but in general it is not possible to guarantee that the two sets are balanced, i.e. have about the same cardinality.

为了确定目录节点的最小重叠分裂，我们必须将节点中的最小边界矩形（MBR）划分为两个子集，使得这两个子集的最小边界超矩形的重叠部分最小。对于点数据，总是可以找到无重叠的分裂，但一般来说，无法保证这两个子集是平衡的，即它们的基数大致相同。

## Definition 2 (Split)

## 定义2（分裂）

The split of a node $S = \left\{  {{mb}{r}_{1},\ldots ,{mb}{r}_{n}}\right\}$ into two subnodes ${S}_{1} = \left\{  {{mb}{r}_{{i}_{1}},\ldots ,{mb}{r}_{{i}_{{s}_{1}}}}\right\}  \;$ and $\;{S}_{2} = \left\{  {{mb}{r}_{{i}_{1}},\ldots ,{mb}{r}_{{i}_{{s}_{2}}}}\right\}$ $\left( {{S}_{1} \neq  \varnothing \text{and}{S}_{2} \neq  \varnothing }\right)$ is defined as

将节点 $S = \left\{  {{mb}{r}_{1},\ldots ,{mb}{r}_{n}}\right\}$ 分裂为两个子节点 ${S}_{1} = \left\{  {{mb}{r}_{{i}_{1}},\ldots ,{mb}{r}_{{i}_{{s}_{1}}}}\right\}  \;$ 和 $\;{S}_{2} = \left\{  {{mb}{r}_{{i}_{1}},\ldots ,{mb}{r}_{{i}_{{s}_{2}}}}\right\}$ $\left( {{S}_{1} \neq  \varnothing \text{and}{S}_{2} \neq  \varnothing }\right)$ 定义为

$$
\operatorname{Split}\left( S\right)  = \left\{  {\left. \left( {{S}_{1},{S}_{2}}\right) \right| \;S = {S}_{1} \cup  {S}_{2} \land  {S}_{1} \cap  {S}_{2} = \varnothing }\right\}  .
$$

The split is called

这种分裂称为

(1) overlap-minimal iff $\begin{Vmatrix}{{MBR}\left( {S}_{1}\right)  \cap  {MBR}\left( {S}_{2}\right) }\end{Vmatrix}$ is minimal

(1) 最小重叠分裂，当且仅当 $\begin{Vmatrix}{{MBR}\left( {S}_{1}\right)  \cap  {MBR}\left( {S}_{2}\right) }\end{Vmatrix}$ 最小

(2) overlap-free iff $\parallel {MBR}\left( {S}_{1}\right)  \cap  {MBR}\left( {S}_{2}\right) \parallel  = 0$

(2) 无重叠分裂，当且仅当 $\parallel {MBR}\left( {S}_{1}\right)  \cap  {MBR}\left( {S}_{2}\right) \parallel  = 0$

(3) balanced iff $- \varepsilon  \leq  \left| {S}_{1}\right|  - \left| {S}_{2}\right|  \leq  \varepsilon$ .

(3) 平衡分裂，当且仅当 $- \varepsilon  \leq  \left| {S}_{1}\right|  - \left| {S}_{2}\right|  \leq  \varepsilon$ 。

For obtaining a suitable directory structure, we are interested in overlap-minimal (overlap-free) splits which are balanced. For simplification, in the following we focus on overlap-free splits and assume to have high-dimensional uniformly distributed point data. ${}^{1}$ It is an interesting observation that an overlap-free split is only possible if there is a dimension according to which all MBRs have been split since otherwise at least one of the MBRs will span the full range of values in that dimension, resulting in some overlap.

为了获得合适的目录结构，我们关注的是平衡的最小重叠（无重叠）分裂。为了简化，在下面我们将重点讨论无重叠分裂，并假设我们有高维均匀分布的点数据。 ${}^{1}$ 一个有趣的观察结果是，只有当存在一个维度，使得所有的最小边界矩形（MBR）都已根据该维度进行了分裂时，才可能进行无重叠分裂，否则至少有一个最小边界矩形（MBR）将跨越该维度的整个取值范围，从而导致一些重叠。

## Lemma 1

## 引理1

For uniformly distributed point data, an overlap-free split is only possible iff there is a dimension according to which all MBRs in the node have been previously split. More formally,

对于均匀分布的点数据，只有当存在一个维度，使得节点中的所有最小边界矩形（MBR）之前都已根据该维度进行了分裂时，才可能进行无重叠分裂。更正式地说，

$\operatorname{Split}\left( S\right)$ is overlap-free $\Leftrightarrow$

$\operatorname{Split}\left( S\right)$ 是无重叠的 $\Leftrightarrow$

$\exists d \in  \{ 1,\ldots ,D\} \forall {mbr} \in  S$ :

mbr has been split according to d

最小边界矩形（MBR）已根据维度d进行了分裂

## Proof (by contradiction):

## 证明（反证法）：

" $\Rightarrow$ ": Assume that for all dimensions there is at least one MBR which has not been split in that dimension. This means for uniformly distributed data that the MBRs span the full range of values of the corresponding dimensions. Without loss of generality,we assume that the ${mbr}$ which spans the full range of values of dimension $d$ is assigned to ${S}_{l}$ . As a consequence, ${MBR}\left( {S}_{1}\right)$ spans the full range for dimension $d$ . Since the extension of ${MBR}\left( {S}_{2}\right)$ cannot be zero in dimension $d$ ,a split using dimension $d$ as split axis cannot be overlap-free (i.e., ${MBR}\left( {S}_{1}\right)  \cap  {MBR}\left( {S}_{2}\right)  \neq  0$ ). Since for all dimensions there is at least one MBR which has not been split in that dimension, we cannot find an overlap-free split.

" $\Rightarrow$ "：假设对于所有维度，至少有一个最小边界矩形（MBR）在该维度上未被分裂。对于均匀分布的数据，这意味着最小边界矩形（MBR）跨越了相应维度的整个取值范围。不失一般性，我们假设跨越维度 $d$ 整个取值范围的 ${mbr}$ 被分配给了 ${S}_{l}$ 。因此， ${MBR}\left( {S}_{1}\right)$ 跨越了维度 $d$ 的整个取值范围。由于 ${MBR}\left( {S}_{2}\right)$ 在维度 $d$ 上的扩展不能为零，使用维度 $d$ 作为分裂轴的分裂不可能是无重叠的（即 ${MBR}\left( {S}_{1}\right)  \cap  {MBR}\left( {S}_{2}\right)  \neq  0$ ）。由于对于所有维度，至少有一个最小边界矩形（MBR）在该维度上未被分裂，我们无法找到无重叠的分裂。

" $\Leftarrow$ ": Assume that an overlap-free split of the node is not possible. This means that there is no dimension which can be used to partition the MBRs into two subsets ${S}_{1}$ and ${S}_{2}$ . This however is in contradiction to the fact that there is a dimension $d$ for which all MBRs have been split. For uniformly distributed point data, the split may be assumed to be in the middle of the range of dimension $d$ and therefore,an overlap-free split is possible using dimension $d{.}^{1}$ I

" $\Leftarrow$ ": 假设无法对该节点进行无重叠分割。这意味着不存在一个维度可用于将最小边界矩形（MBR，Minimum Bounding Rectangle）划分为两个子集 ${S}_{1}$ 和 ${S}_{2}$ 。然而，这与存在一个维度 $d$ 使得所有 MBR 都已被分割这一事实相矛盾。对于均匀分布的点数据，可以假设分割发生在维度 $d$ 范围的中间，因此，使用维度 $d{.}^{1}$ 进行无重叠分割是可行的。

---

<!-- Footnote -->

1. According to our experiments, the results generalize to real data and even to spatial data (cf. section 4).

1. 根据我们的实验，这些结果可推广到真实数据，甚至空间数据（参见第 4 节）。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: split tree A A' B" C I A" B" C D I 囚 Node S A’B A' B' C -->

<img src="https://cdn.noedgeai.com/0195c90f-ed1a-7cf0-ae28-0ea5e38bb72d_7.jpg?x=191&y=204&w=694&h=328&r=0"/>

Figure 9: Example for the Split History

图 9：分割历史示例

<!-- Media -->

According to Lemma 1, for finding an overlap-free split we have to determine a dimension according to which all MBRs of $S$ have been split previously. The split history provides the necessary information, in particular the dimensions according to which an MBR has been split and which new MBRs have been created by this split. Since a split creates two new MBRs from one, the split history may be represented as a binary tree, called the split tree. Each leaf node of the split tree corresponds to an MBR in $S$ . The internal nodes of the split tree correspond to MBRs which do not exist any more since they have been split into new MBRs previously. Internal nodes of the split tree are labeled by the split axis that has been used; leaf nodes are labeled by the MBR they are related to. All MBRs related to leaves in the left subtree of an internal node have lower values in the split dimension of the node than the MBRs related to those in the right subtree.

根据引理 1，为了找到无重叠分割，我们必须确定一个维度，根据该维度，$S$ 中的所有 MBR 之前已被分割。分割历史提供了必要的信息，特别是 MBR 被分割所依据的维度以及通过该分割创建的新 MBR。由于一次分割从一个 MBR 创建出两个新 MBR，分割历史可以表示为一棵二叉树，称为分割树。分割树的每个叶节点对应于 $S$ 中的一个 MBR。分割树的内部节点对应于那些由于之前已被分割成新 MBR 而不再存在的 MBR。分割树的内部节点用所使用的分割轴进行标记；叶节点用它们所关联的 MBR 进行标记。与内部节点左子树中的叶节点相关联的所有 MBR 在该节点的分割维度上的值低于与右子树中的叶节点相关联的 MBR。

Figure 9 shows an example for the split history of a node $S$ and the respective split tree. The process starts with a single MBR A corresponding to a split tree which consists of only one leaf node labeled by A. For uniformly distributed data, A spans the full range of values in all dimensions. The split of A using dimension 2 as split axis produces new MBRs A' and B. Note that A' and B are disjoint because any point in MBR A' has a lower coordinate value in dimension 2 than all points in MBR B. The split tree now has one internal node (marked with dimension 2) and two leaf nodes (A' and B). Splitting MBR B using dimension 5 as split axis creates the nodes B' and C. After splitting B' and A' again, we finally reach the situation depicted in the right most tree of Figure 9 where $\mathrm{S}$ is completely filled with the MBRs $\mathrm{A}$ ", $\mathrm{B}$ ", C, D and E.

图 9 展示了一个节点 $S$ 的分割历史以及相应的分割树示例。该过程从一个单一的 MBR A 开始，对应于一棵仅由一个标记为 A 的叶节点组成的分割树。对于均匀分布的数据，A 在所有维度上跨越了完整的值范围。使用维度 2 作为分割轴对 A 进行分割会产生新的 MBR A' 和 B。请注意，A' 和 B 是不相交的，因为 MBR A' 中的任何点在维度 2 上的坐标值都低于 MBR B 中的所有点。此时分割树有一个内部节点（标记为维度 2）和两个叶节点（A' 和 B）。使用维度 5 作为分割轴对 MBR B 进行分割会创建节点 B' 和 C。再次分割 B' 和 A' 后，我们最终到达图 9 最右侧树所描绘的情况，其中 $\mathrm{S}$ 完全被 MBR $\mathrm{A}$ "、$\mathrm{B}$ "、C、D 和 E 填充。

According to Lemma 1, we may find an overlap-free split if there is a dimension according to which all MBRs of S have been split. To obtain the information according to which dimensions an MBR X in S has been split,we only have to traverse the split tree from the root node to the leaf that corresponds to X. For example, MBR C has been split according to dimension 2 and 5 , since the path from the root node to the leaf $\mathrm{C}$ is labeled with 2 and 5 . Obviously,all MBRs of the split tree in Figure 9 have been split according to dimension 2, the split axis used in the root of the split tree. In general, all MBRs in any split tree have one split dimension in common, namely the split axis used in the root node of the split tree.

根据引理 1，如果存在一个维度使得 S 中的所有 MBR 都已被分割，我们就可以找到无重叠分割。为了获取 S 中的 MBR X 是根据哪些维度被分割的信息，我们只需从分割树的根节点遍历到对应于 X 的叶节点。例如，MBR C 是根据维度 2 和 5 被分割的，因为从根节点到叶节点 $\mathrm{C}$ 的路径标记为 2 和 5。显然，图 9 中分割树的所有 MBR 都根据维度 2 进行了分割，维度 2 是分割树的根节点所使用的分割轴。一般来说，任何分割树中的所有 MBR 都有一个共同的分割维度，即分割树的根节点所使用的分割轴。

## Lemma 2 (Existence of an Overlap-free Split) For point data, an overlap-free split always exists.

## 引理 2（无重叠分割的存在性） 对于点数据，无重叠分割总是存在的。

## Proof (using the split history):

## 证明（使用分割历史）：

From the description of the split tree it is clear that all MBRs of a directory node $S$ have one split dimension in common, namely the dimension used as split axis in the root node of the split tree. Let $\mathrm{{SD}}$ be this dimension. We are able to partition $S$ such that all MBRs related to leaves in the left subtree of the root node are contained in ${S}_{1}$ and all other MBRs contained in ${\mathrm{S}}_{2}$ . Since any point belonging to ${\mathrm{S}}_{1}$ has a lower value in dimension SD than all points belonging to ${\mathrm{S}}_{2}$ ,the split is overlap-free ${}^{2}$ . ∎

从分割树的描述中可以清楚地看出，目录节点 $S$ 的所有 MBR 有一个共同的分割维度，即分割树的根节点中用作分割轴的维度。设 $\mathrm{{SD}}$ 为该维度。我们能够对 $S$ 进行划分，使得与根节点左子树中的叶节点相关联的所有 MBR 都包含在 ${S}_{1}$ 中，而所有其他 MBR 包含在 ${\mathrm{S}}_{2}$ 中。由于属于 ${\mathrm{S}}_{1}$ 的任何点在维度 SD 上的值都低于属于 ${\mathrm{S}}_{2}$ 的所有点，因此该分割是无重叠的 ${}^{2}$ 。 ∎

One may argue that there may exist more than one overlap-free split dimension which is part of the split history of all data pages. This is true in most cases for low dimensionality, but the probability that a second split dimension exists which is part of the split history of all MBRs is decreasing rapidly with increasing dimensionality (cf. Figure 10). If there is no dimension which is in the split history of all MBRs, the resulting overlap of the newly created directory entries is on the average about ${50}\%$ . This can be explained as follows: Since at least one MBR has not been split in the split dimension $d$ ,one of the partitions (without loss of generality: ${S}_{1}$ ) spans the full range of values in that dimension. The other partition ${\mathrm{S}}_{2}$ spans at least half the range of values of the split dimension $\bar{d}$ . Since the MBRs are only partitioned with respect to dimension $d,{S}_{1}$ and ${S}_{2}$ span the full range of values of all other dimensions, resulting in a total overlap of about ${50}\%$ .

有人可能会认为，可能存在不止一个无重叠的分割维度，且该维度是所有数据页分割历史的一部分。在大多数低维情况下确实如此，但存在第二个无重叠分割维度且该维度是所有最小边界矩形（MBR）分割历史一部分的概率会随着维度的增加而迅速降低（参见图10）。如果不存在一个维度是所有MBR分割历史的一部分，那么新创建的目录项的平均重叠率约为${50}\%$。这可以解释如下：由于至少有一个MBR在分割维度$d$上未被分割，其中一个分区（不失一般性：${S}_{1}$）在该维度上涵盖了完整的值范围。另一个分区${\mathrm{S}}_{2}$在分割维度$\bar{d}$上至少涵盖了一半的值范围。由于MBR仅在维度$d,{S}_{1}$上进行分区，且${S}_{2}$涵盖了所有其他维度的完整值范围，因此总重叠率约为${50}\%$。

<!-- Media -->

<!-- figureText: 0.80 20.00 26.00 32.00 dimension 0.70 0.60 0.50 0.40 0.30 0.10 0.00 2.00 8.00 14.00 -->

<img src="https://cdn.noedgeai.com/0195c90f-ed1a-7cf0-ae28-0ea5e38bb72d_7.jpg?x=932&y=1334&w=665&h=556&r=0"/>

Figure 10: Probability of the Existence of a Second Overlap-free Split Dimension

图10：存在第二个无重叠分割维度的概率

<!-- Media -->

---

<!-- Footnote -->

1. If the splits have not been performed exactly in the middle of the data space, at least an overlap-minimal split is obtained.

1. 如果分割不是精确地在数据空间的中间进行，至少可以获得一个重叠最小的分割。

2. Note that the resulting split is not necessarily balanced since sorted input data, for example, will result in an unbalanced split tree.

2. 请注意，得到的分割不一定是平衡的，因为例如排序后的输入数据会导致分割树不平衡。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: 500.00 25.00 20.00 Speed-Up Factor 15.00 10.00 0.00 2 4 6 8 10 14 16 dimension b. 10 Nearest-Neighbor Query 400.00 Speed-Up Factor 300.00 200.00 100.00 0.00 2 6 dimension a. Point Query -->

<img src="https://cdn.noedgeai.com/0195c90f-ed1a-7cf0-ae28-0ea5e38bb72d_8.jpg?x=313&y=219&w=1184&h=509&r=0"/>

Figure 11: Speed-Up of X-tree over R*-tree on Real Point Data (70 MBytes)

图11：X树在真实点数据（70兆字节）上相对于R*树的加速比

<!-- Media -->

The probability that a split algorithm which arbitrarily chooses the split axis coincidentally selects the right split axis for an overlap-free split is very low in high-dimensional space. As our analysis of the R*-tree shows, the behavior of the topological R*-tree split algorithm in high-dimensional space is similar to a random choice of the split axis since it optimizes different criteria. If the topological split fails, our split algorithm tries to perform an overlap-free split. This is done by determining the dimension for the overlap-free split as described above, determining the split value, and partitioning the MBRs with respect to the split value. If the resulting split is unbalanced,the insert algorithm of the X-tree initiates the creation/extension of a supernode (cf. subsection 3.2). Note that for the overlap-minimal split, information about the split history has to be stored in the directory nodes. The space needed for this purpose, however, is very small since the split history may be coded by a few bits.

在高维空间中，一个随机选择分割轴的分割算法恰好选择到无重叠分割的正确分割轴的概率非常低。正如我们对R*树的分析所示，拓扑R*树分割算法在高维空间中的行为类似于随机选择分割轴，因为它优化了不同的标准。如果拓扑分割失败，我们的分割算法会尝试进行无重叠分割。这通过如上所述确定无重叠分割的维度、确定分割值，并根据分割值对MBR进行分区来实现。如果得到的分割不平衡，X树的插入算法会启动超节点的创建/扩展（参见3.2小节）。请注意，对于重叠最小的分割，分割历史信息必须存储在目录节点中。然而，为此所需的空间非常小，因为分割历史可以用几位来编码。

## 4. Performance Evaluation

## 4. 性能评估

To show the practical relevance of our method, we performed an extensive experimental evaluation of the X-tree and compared it to the TV-tree as well to as the R*-tree. All experimental results presented in this sections are computed on an HP735 workstation with 64 MBytes of main memory and several GBytes of secondary storage. All programs have been implemented in $\mathrm{C} +  +$ as templates to support different types of data objects. The X-tree and R*-tree support different types of queries such as point queries and nearest neighbor queries; the implementation of the TV-tree ${}^{1}$ only supports point queries. We use the original implementation of the TV-tree by K. Lin, H. V. Jagadish, and C. Faloutsos [LJF 94].

为了展示我们方法的实际相关性，我们对X树进行了广泛的实验评估，并将其与TV树以及R*树进行了比较。本节中呈现的所有实验结果都是在一台配备64兆字节主内存和数吉字节二级存储的HP735工作站上计算得出的。所有程序都用$\mathrm{C} +  +$作为模板实现，以支持不同类型的数据对象。X树和R*树支持不同类型的查询，如点查询和最近邻查询；TV树${}^{1}$的实现仅支持点查询。我们使用了K. Lin、H. V. Jagadish和C. Faloutsos [LJF 94]对TV树的原始实现。

The test data used for the experiments are real point data consisting of Fourier points in high-dimensional space $\left( {D = 2,4,8,{16}}\right)$ ,spatial data $\left( {D = 2,4,8,{16}}\right)$ consisting of manifolds in high-dimensional space describing regions of real CAD-objects, and synthetic data consisting of uniformly distributed points in high-dimensional space $(\mathrm{D} = 2,3$ , $4,6,8,{10},{12},{14},{16})$ . The block size used for our experiments is 4 KByte, and all index structures were allowed to use the same amount of cache. For a realistic evaluation, we used very large amounts of data in our experiments. The total amount of disk space occupied by the created index structures of TV-trees, R*-trees, and X-trees is about 10 GByte and the CPU time for inserting the data adds up to about four weeks of CPU time. As one expects, the insertion times increase with increasing dimension. For all experiments, the insertion into the X-tree was much faster than the insertion into the TV-tree and the R*-tree (up to a factor of 10.45 faster than the R*-tree). The X-tree reached a rate of about 170 insertions per second for a 150 MBytes index containing 16- dimensional point data.

实验所用的测试数据包括：高维空间 $\left( {D = 2,4,8,{16}}\right)$ 中的傅里叶点构成的真实点数据、描述真实计算机辅助设计（CAD）对象区域的高维空间流形构成的空间数据 $\left( {D = 2,4,8,{16}}\right)$，以及高维空间 $(\mathrm{D} = 2,3$、$4,6,8,{10},{12},{14},{16})$ 中均匀分布点构成的合成数据。我们实验使用的块大小为 4 千字节，并且允许所有索引结构使用相同大小的缓存。为了进行真实评估，我们在实验中使用了大量数据。TV 树、R* 树和 X 树所创建的索引结构占用的磁盘总空间约为 10 吉字节，插入数据的 CPU 时间总计约为四周。正如预期的那样，插入时间随维度的增加而增加。在所有实验中，插入 X 树的速度比插入 TV 树和 R* 树快得多（比 R* 树快达 10.45 倍）。对于一个包含 16 维点数据、大小为 150 兆字节的索引，X 树每秒大约能插入 170 条数据。

First, we evaluated the X-tree on synthetic databases with varying dimensionality. Using the same number of data items over the different dimensions implies that the size of the database is linearly increasing with the dimension. This however has an important drawback, namely that in low dimensions, we would obtain only very small databases, whereas in high dimensions the databases would become very large. It is more realistic to assume that the amount of data which is stored in the database is constant. This means, however, that the number of data items needs to be varied accordingly. For the experiment presented in Figure 13, we used 100 MByte databases containing uniformly distributed point data. The number of data items varied between 8.3 million for $\mathrm{D} = 2$ and 1.5 million for $\mathrm{D} = {16}$ . Figure 13,shows the speed-up of the search time for point queries of the X-tree over the R*-tree. As expected, the speed-up increases with growing dimension,reaching values of about270for $\mathrm{D} = {16}$ . For lower dimensions, the speed-up is still higher than one order of magnitude (e.g., for D=8 the speed-up is about 30). The high speed-up factors are caused by the fact that, due to the high overlap in high dimensions, the R*-tree needs to access most of the directory pages. The total query time turned out to be clearly dominated by the I/O-time, i.e. the number of page accesses (see also Figure 12).

首先，我们在不同维度的合成数据库上对 X 树进行了评估。在不同维度上使用相同数量的数据项意味着数据库的大小随维度线性增加。然而，这有一个重要的缺点，即在低维度下，我们只能得到非常小的数据库，而在高维度下，数据库会变得非常大。更现实的做法是假设数据库中存储的数据量是恒定的。但这意味着数据项的数量需要相应地变化。在图 13 所示的实验中，我们使用了包含均匀分布点数据、大小为 100 兆字节的数据库。数据项的数量在 $\mathrm{D} = 2$ 时为 830 万，在 $\mathrm{D} = {16}$ 时为 150 万。图 13 展示了 X 树相对于 R* 树在点查询搜索时间上的加速比。正如预期的那样，加速比随维度的增加而增加，在 $\mathrm{D} = {16}$ 时达到约 270。对于较低维度，加速比仍高于一个数量级（例如，当 D = 8 时，加速比约为 30）。高加速比的原因是，由于高维度中的高重叠性，R* 树需要访问大部分目录页。事实证明，总查询时间明显由 I/O 时间（即页面访问次数）决定（另见图 12）。

---

<!-- Footnote -->

1 We use the original implementation of the TV-tree by K. Lin, H. V. Jagadish, and C. Faloutsos [LJF 94].

1 我们使用了 K. Lin、H. V. Jagadish 和 C. Faloutsos [LJF 94] 对 TV 树的原始实现。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: 14000 40.00 35.00 30.00 CPU (sec) 25.00 20.00 - R*-tree 15.00 X-tree 10.00 5.00 0.00 8 10 12 14 16 dimension b. CPU-Time 12000 page accesses 8000 6000 X-tree 2000 2 4 6 10 12 dimension a. Page Accesses -->

<img src="https://cdn.noedgeai.com/0195c90f-ed1a-7cf0-ae28-0ea5e38bb72d_9.jpg?x=270&y=263&w=1270&h=529&r=0"/>

Figure 12: Number of Page Accesses versus CPU-Time on Real Point Data (70 MBytes)

图 12：真实点数据（70 兆字节）的页面访问次数与 CPU 时间的关系

<!-- Media -->

Since one may argue that synthetic databases with uniformly distributed data are not realistic in high-dimensional space, we also used real data in our experiments. We had access to large Fourier databases of variable dimensionality containing about ${70}\mathrm{M}$ byte of Fourier data representing shapes of polygons. The results of our experiments (cf. Figure 11) show that the speed-up of the total search time for point queries is even higher (about 90 for $\mathrm{D} = 4$ and about 320for $\mathrm{D} = 8$ ) than the speed-up of uniformly distributed data. This result was surprising but corresponds to the higher overlap of real data found in the overlap curves (cf. Figure 2). Additionally to point queries, in applications with high-dimensional data nearest neighbor queries are also important. We therefore also compared the performance of nearest neighbor queries searching for the 10 nearest neighbors. The nearest neighbor algorithm supported in the X-tree and R*-tree is the algorithm presented in [RKV 95]. The results of our comparisons show that the speed-up for nearest neighbor queries is still between about 10 for $\mathrm{D} = 6$ and about 20 for $\mathrm{D} = {16}$ . Since the nearest neighbor algorithm requires sorting the nodes according to the min-max distance, the CPU-time needed for nearest neighbor queries is much higher. In Figure 12, we therefore present the number of page accesses and the CPU-time of the $X$ -tree and the ${R}^{ * }$ -tree for nearest-neighbor queries. The figure shows that the X-tree provides a consistently better performance than the R*-tree. Note that, in counting page accesses,accesses to supernodes of size $s$ are counted as $s$ page accesses. In most practical cases,however,the su-pernodes will be cached due to the better main memory utilization of the X-tree. For practically relevant buffer sizes ( 1 MByte to 10 MBytes) there is no significant change of page accesses. For extreme buffer sizes of more than 10 MBytes or less than 1 MByte, the speed-up may decrease. The better CPU-times of the X-tree may be explained by the fact that due to the overlap the R*-tree has to search a large portion of the directory which in addition is larger than the X-tree directory.

由于有人可能会认为，在高维空间中，具有均匀分布数据的合成数据库并不现实，因此我们在实验中也使用了真实数据。我们可以访问可变维度的大型傅里叶数据库，其中包含约 ${70}\mathrm{M}$ 字节的傅里叶数据，这些数据代表多边形的形状。我们的实验结果（参见图 11）表明，点查询的总搜索时间加速比甚至比均匀分布数据的加速比更高（对于 $\mathrm{D} = 4$ 约为 90，对于 $\mathrm{D} = 8$ 约为 320）。这一结果令人惊讶，但与重叠曲线中发现的真实数据的更高重叠度相符（参见图 2）。除了点查询之外，在处理高维数据的应用中，最近邻查询也很重要。因此，我们还比较了搜索 10 个最近邻的最近邻查询的性能。X 树和 R* 树所支持的最近邻算法是文献 [RKV 95] 中提出的算法。我们的比较结果表明，最近邻查询的加速比仍然在对于 $\mathrm{D} = 6$ 约为 10 到对于 $\mathrm{D} = {16}$ 约为 20 之间。由于最近邻算法需要根据最小 - 最大距离对节点进行排序，因此最近邻查询所需的 CPU 时间要高得多。因此，在图 12 中，我们展示了 $X$ 树和 ${R}^{ * }$ 树在最近邻查询时的页面访问次数和 CPU 时间。该图显示，X 树的性能始终优于 R* 树。请注意，在计算页面访问次数时，对大小为 $s$ 的超级节点的访问被计为 $s$ 次页面访问。然而，在大多数实际情况下，由于 X 树对主内存的利用率更高，超级节点将被缓存。对于实际相关的缓冲区大小（1 兆字节到 10 兆字节），页面访问次数没有显著变化。对于超过 10 兆字节或小于 1 兆字节的极端缓冲区大小，加速比可能会降低。X 树具有更好的 CPU 时间这一现象可以解释为：由于重叠问题，R* 树必须搜索目录的很大一部分，而且该目录比 X 树的目录更大。

<!-- Media -->

<!-- figureText: 300.00 12 14 dimension 250.00 Speed-Up Factor 200.00 150.00 100.00 50.00 0.00 2 4 6 -->

<img src="https://cdn.noedgeai.com/0195c90f-ed1a-7cf0-ae28-0ea5e38bb72d_9.jpg?x=219&y=1507&w=639&h=515&r=0"/>

Figure 13: Speed-Up of X-tree over R*-tree on Point Queries (100 MBytes of Synthetic Point Data)

图 13：在点查询上 X 树相对于 R* 树的加速比（100 兆字节的合成点数据）

<!-- Media -->

Figure 14 shows the total search time of point queries depending on the size of the database $\left( {\mathrm{D} = {16}}\right)$ . Note that in this figure we use a logarithmic scale of the y-axis, since otherwise the development of the times for the X-tree would not be visible (identical with the x-axis). Figure 14 shows that the search times of the X-tree are consistently about two orders of magnitude faster than those of the R*-tree (for $\mathrm{D} = {16}$ ). The speed-up slightly increases with the database size from about 100 for 20 MBytes to about 270 for 100 MBytes. Also, as expected, the total search time of the X-tree grows logarithmically with the database size which means that the X-tree scales well to very large database sizes.

图 14 展示了点查询的总搜索时间与数据库大小 $\left( {\mathrm{D} = {16}}\right)$ 的关系。请注意，在该图中，我们使用了 y 轴的对数刻度，因为否则 X 树的时间变化情况将不可见（与 x 轴重合）。图 14 显示，X 树的搜索时间始终比 R* 树的搜索时间快约两个数量级（对于 $\mathrm{D} = {16}$）。加速比随数据库大小略有增加，从 20 兆字节时的约 100 增加到 100 兆字节时的约 270。此外，正如预期的那样，X 树的总搜索时间随数据库大小呈对数增长，这意味着 X 树能够很好地适应非常大的数据库规模。

<!-- Media -->

<!-- figureText: 10.00 R*-tree x-tree log(search time) (sec) 1.00 0.10 0.00 Amount of Data (MBytes) -->

<img src="https://cdn.noedgeai.com/0195c90f-ed1a-7cf0-ae28-0ea5e38bb72d_10.jpg?x=199&y=218&w=667&h=498&r=0"/>

Figure 14: Total Search Time of Point Queries for Varying Database Size (Synthetic Point Data)

图14：不同数据库大小下点查询的总搜索时间（合成点数据）

<!-- Media -->

We also performed a comparison of the X-tree with the TV-tree and the R*-tree. With the implementation of the TV-tree made available to us by the authors of the TV-tree, we only managed to insert up to 25.000 data items which is slightly higher than the number of data items used in the original paper [LJF 94]. For the comparisons, we were therefore not able to use our large databases. The results of our comparisons are presented in Figure 16. The speed-up of the X-tree over the TV-tree ranges between 4 and 12 , even for the rather small databases. It is interesting to note that the performance of the R*-tree is better than the performance of the TV-tree for D smaller than 16.

我们还对X树与TV树和R*树进行了比较。借助TV树作者提供给我们的TV树实现，我们最多只能插入25000个数据项，这略高于原论文[LJF 94]中使用的数据项数量。因此，在进行比较时，我们无法使用大型数据库。我们的比较结果如图16所示。即使对于相当小的数据库，X树相对于TV树的加速比也在4到12之间。有趣的是，当D小于16时，R*树的性能优于TV树。

<!-- Media -->

<!-- figureText: 8.00 8 10 12 14 16 dimension 7.00 Speed-Up Factor 6.00 5.00 4.00 3.00 2.00 1.00 0.00 4 6 -->

<img src="https://cdn.noedgeai.com/0195c90f-ed1a-7cf0-ae28-0ea5e38bb72d_10.jpg?x=216&y=1453&w=641&h=572&r=0"/>

Figure 15: Speed-Up of X-tree over R*-tree on Real Extended Spatial Data

图15：在真实扩展空间数据上X树相对于R*树的加速比

<!-- figureText: 28.00 TV-tree Rev-tree 24 24.00 page accesses 20.00 16.00 12.00 8.00 4.00 0.00 12 16 dimension -->

<img src="https://cdn.noedgeai.com/0195c90f-ed1a-7cf0-ae28-0ea5e38bb72d_10.jpg?x=938&y=222&w=658&h=486&r=0"/>

Figure 16: Comparison of X-tree, TV-tree, and R*-tree on Synthetic Data

图16：在合成数据上X树、TV树和R*树的比较

<!-- Media -->

In addition to using point data, we also examined the performance of the X-tree for extended data objects in high-dimensional space. The results of our experiments are shown in Figure 15. Since the extended spatial data objects induce some overlap in the X-tree as well, the speed-up of the X-tree over the R*-tree is lower than for point data. Still, we achieve a speed-up factor of about 8 for $\mathrm{D} = {16}$ .

除了使用点数据，我们还研究了X树在高维空间中对扩展数据对象的性能。我们的实验结果如图15所示。由于扩展空间数据对象也会在X树中导致一些重叠，因此X树相对于R*树的加速比低于点数据的情况。不过，对于$\mathrm{D} = {16}$，我们仍实现了约8倍的加速比。

## 5. Conclusions

## 5. 结论

In this paper, we propose a new indexing method for high-dimensional data. We investigate the effects that occur in high dimensions and show that R-tree-based index structures do not behave well for indexing high-dimensional spaces. We introduce formal definitions of overlap and show the correlation between overlap in the directory and poor query performance. We then propose a new index structure, the X-tree, which uses - in addition to the concept of supernodes - a new split algorithm minimizing overlap. Supernodes are directory nodes which are extended over the usual block size in order to avoid a degeneration of the index. We carry out an extensive performance evaluation of the $\mathrm{X}$ -tree and compare the $\mathrm{X}$ -tree with the $\mathrm{{TV}}$ -tree and the R*-tree using up to 100 MBytes of point and spatial data. The experiments show that the X-tree outperforms the TV-tree and R*-tree up to orders of magnitude for point queries and nearest neighbor queries on both synthetic and real data.

在本文中，我们提出了一种新的高维数据索引方法。我们研究了在高维情况下出现的影响，并表明基于R树的索引结构在对高维空间进行索引时表现不佳。我们引入了重叠的正式定义，并展示了目录中的重叠与查询性能不佳之间的相关性。然后，我们提出了一种新的索引结构——X树，它除了使用超节点的概念外，还采用了一种新的分裂算法来最小化重叠。超节点是目录节点，其大小超出了通常的块大小，以避免索引退化。我们对$\mathrm{X}$树进行了广泛的性能评估，并使用多达100兆字节的点数据和空间数据将$\mathrm{X}$树与$\mathrm{{TV}}$树和R*树进行了比较。实验表明，在合成数据和真实数据上进行点查询和最近邻查询时，X树的性能比TV树和R*树高出几个数量级。

Since for very high dimensionality the supernodes may become rather large, we currently work on a parallel version of the X-tree which is expected to provide a good performance even for larger data sets and the more time consuming nearest neighbor queries. We also develop a novel nearest neighbor algorithm for high-dimensional data which is adapted to the X-tree.

由于在非常高的维度下，超节点可能会变得相当大，我们目前正在研究X树的并行版本，预计该版本即使对于更大的数据集和更耗时的最近邻查询也能提供良好的性能。我们还为高维数据开发了一种新的最近邻算法，该算法适用于X树。

## Acknowledgment

## 致谢

We are thankful to K. Lin, C. Faloutsos, and H. V. Jag-adish for making the implementation of the TV-tree available to us.

我们感谢K. Lin、C. Faloutsos和H. V. Jag - adish为我们提供TV树的实现。

## References

## 参考文献

[AFS 93] Agrawal R., Faloutsos C., Swami A.: 'Efficient Similarity Search in Sequence Databases', Proc. 4th Int. Conf. on Foundations of Data Organization and Algorithms, Evanston, ILL, 1993, in: Lecture Notes in Computer Science, Vol. 730, Springer, 1993, pp. 69-84.

MM 90] Altschul S. F., Gish W., Miller W., Myers E. W., Lipman D. J.: 'A Basic Local Alignment Search Tool', Journal of Molecular Biology, Vol. 215, No. 3, 1990, pp. 403-410.

[BKSS 90] Beckmann N., Kriegel H.-P., Schneider R., Seeger B.: 'The R*-tree: An Efficient and Robust Access Method for Points and Rectangles', Proc. ACM SIGMOD Int. Conf. on Management of Data, Atlantic City, NJ, 1990, pp. 322-331.

[DE 82] Dunn G., Everitt B.: 'An Introduction to Mathematical Taxonomy', Cambridge University Press, Cambridge, MA, 1982.

[Fal 94] Faloutsos C., Barber R., Flickner M., Hafner J., et al.: 'Efficient and Effective Querying by Image Content', Journal of Intelligent Information Systems, 1994, Vol. 3, pp. 231-262.

[FL 95] Faloutsos C., Lin K.: 'Fastmap: A fast Algorithm for Indexing, Data-Mining and Visualization of Traditional and Multimedia Datasets', Proc. ACM SIGMOD Int. Conf. on Management of Data, San Jose, CA, 1995, pp. 163-174.

[Gut 84] Guttman A.: 'R-trees: A Dynamic Index Structure for Spatial Searching', Proc. ACM SIGMOD Int. Conf. on Management of Data, Boston, MA, 1984, pp. 47-57.

[GN 91] Günther O., Noltemeier H.: 'Spatial Database Indices For Large Extended Objects', Proc. 7 th Int. Conf. on Data Engineering, 1991, pp. 520-527.

[Har 67] Harman'H. H.: 'Modern Factor Analysis', University of Chicago Press, 1967.

[Jag 91] Jagadish H. V.: 'A Retrieval Technique for Similar Shapes', Proc. ACM SIGMOD Int. Conf. on Management of Data, 1991, pp. 208-217.

[Kuk 92] Kukich K.: 'Techniques for Automatically Correcting Words in Text', ACM Computing Surveys, Vol. 24, No. 4, 1992, pp. 377-440.

[KW 78] Kruskal J. B., Wish M.: 'Multidimensional Scaling', SAGE publications, Beverly Hills, 1978.

[LJF 94] Lin K., Jagadish H. V., Faloutsos C.: 'The TV-tree: An Index Structure for High-Dimensional Data', VLDB Journal, Vol. 3, 1995, pp. 517-542.

[MG 93] Mehrotra R., Gary J. E.: 'Feature-Based Retrieval of Similar Shapes', Proc. 9th Int. Conf. on Data Engineering, Vienna, Austria, 1993, pp. 108-115.

[MG 95] Mehrotra R., Gary J. E.: 'Feature-Index-Based Similar Shape retrieval', Proc. of the 3rd Working Conf. on Visual Database Systems, 1995, pp. 46-65.

[MN 95] Murase H., Nayar S. K: 'Three-Dimensional Object Recognition from Appearance-Parametric Eigenspace Method', Systems and Computers in Japan, Vol. 26, No. 8, 1995, pp. 45-54.

[NHS 84] Nievergelt J., Hinterberger H., Sevcik K. C.: 'The Grid File: An Adaptable, Symmetric Multikey File Structure', ACM Trans. on Database Systems, Vol. 9, No. 1, 1984, pp. 38-71.

[RKV 95] Roussopoulos N., Kelley S., Vincent F.: 'Nearest Neighbor Queries', Proc. ACM SIGMOD Int. Conf. on Management of Data, San Jose, CA, 1995, pp. 71-79.

[Rob 81] Robinson J. T.: The K-D-B-tree: A Search Structure for Large Multidimensional Dynamic Indexes', Proc. ACM SIGMOD Int. Conf. on Management of Data, 1981, pp. 10-18.

[SBK 92] Shoichet B. K., Bodian D. L., Kuntz I. D.: 'Molecular Docking Using Shape Descriptors', Journal of Computational Chemistry, Vol. 13, No. 3, 1992, pp. 380-397.

[SH 94] Shawney H., Hafner J.: 'Efficient Color Histogram Indexing', Proc. Int. Conf. on Image Processing, 1994, pp. 66-70.

[SK 90] Seeger B., Kriegel H.-P.: 'The Buddy Tree: An Efficient and Robust Access Method for Spatial Data Base Systems', Proc. 16th Int. Conf. on Very Large Data Bases, Brisbane, Australia, 1990, pp. 590-601.

[SRF 87] Sellis T., Roussopoulos N., Faloutsos C.: 'The ${R}^{ + }$ -Tree: A Dynamic Index for Multi-Dimensional Objects', Proc. 13th Int. Conf. on Very Large Databases, Brighton, England, 1987, pp 507-518.

[WJ 96] White, D., Jain R.: 'Similarity Indexing with the SS-tree', Proc. 12th Int. Conf. on Data Engineering, New Orleans, LA, 1996.

[WW 80] Wallace T., Wintz P.: 'An Efficient Three-Dimensional Aircraft Recognition Algorithm Using Normalized Fourier Descriptors', Computer Graphics and Image Processing, Vol. 13, 1980, pp. 99-126.