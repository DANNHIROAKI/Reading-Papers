# LOF: Identifying Density-Based Local Outliers

# 局部离群因子（LOF）：识别基于密度的局部离群点

Markus M. Breunig ${}^{ \dagger  }$ ,Hans-Peter Kriegel ${}^{ \dagger  }$ ,Raymond T. Ng ${}^{ \ddagger  }$ ,Jörg Sander ${}^{ \dagger  }$

马库斯·M·布罗伊尼格 ${}^{ \dagger  }$ ，汉斯 - 彼得·克里格尔 ${}^{ \dagger  }$ ，雷蒙德·T·吴 ${}^{ \ddagger  }$ ，约尔格·桑德 ${}^{ \dagger  }$

† Institute for Computer Science University of Munich

† 慕尼黑大学计算机科学研究所

Oettingenstr. 67, D-80538 Munich, Germany

奥廷根大街67号，德国慕尼黑80538

$\{$ breunig $\mid$ kriegel $\mid$ sander $\}$

$\{$ breunig $\mid$ kriegel $\mid$ sander $\}$

@dbs.informatik.uni-muenchen.de

@dbs.informatik.uni - muenchen.de

rng@cs.ubc.ca

Department of Computer Science

计算机科学系

University of British Columbia

英属哥伦比亚大学

Vancouver, BC V6T 1Z4 Canada

加拿大不列颠哥伦比亚省温哥华市V6T 1Z4

## ABSTRACT

## 摘要

For many KDD applications, such as detecting criminal activities in E-commerce, finding the rare instances or the outliers, can be more interesting than finding the common patterns. Existing work in outlier detection regards being an outlier as a binary property. In this paper, we contend that for many scenarios, it is more meaningful to assign to each object a degree of being an outlier. This degree is called the local outlier factor (LOF) of an object. It is local in that the degree depends on how isolated the object is with respect to the surrounding neighborhood. We give a detailed formal analysis showing that LOF enjoys many desirable properties. Using real-world datasets, we demonstrate that LOF can be used to find outliers which appear to be meaningful, but can otherwise not be identified with existing approaches. Finally, a careful performance evaluation of our algorithm confirms we show that our approach of finding local outliers can be practical.

对于许多知识发现与数据挖掘（KDD）应用，如检测电子商务中的犯罪活动，发现罕见实例或离群点可能比发现常见模式更有意义。现有的离群点检测工作将是否为离群点视为一个二元属性。在本文中，我们认为在许多场景下，为每个对象赋予一个离群程度更为合理。这个程度被称为对象的局部离群因子（LOF）。它是局部的，因为该程度取决于对象相对于其周围邻域的孤立程度。我们进行了详细的形式化分析，表明LOF具有许多理想的性质。通过使用真实世界的数据集，我们证明了LOF可用于发现看似有意义但用现有方法无法识别的离群点。最后，对我们算法的仔细性能评估证实了我们的局部离群点发现方法是可行的。

## Keywords

## 关键词

Outlier Detection, Database Mining.

离群点检测，数据库挖掘。

## 1. INTRODUCTION

## 1. 引言

Larger and larger amounts of data are collected and stored in databases, increasing the need for efficient and effective analysis methods to make use of the information contained implicitly in the data. Knowledge discovery in databases (KDD) has been defined as the non-trivial process of identifying valid, novel, potentially useful, and ultimately understandable knowledge from the data [9].

越来越多的数据被收集并存储在数据库中，这增加了对高效且有效的分析方法的需求，以便利用数据中隐含的信息。数据库中的知识发现（KDD）被定义为从数据中识别有效、新颖、潜在有用且最终可理解的知识的非平凡过程 [9]。

Most studies in KDD focus on finding patterns applicable to a considerable portion of objects in a dataset. However, for applications such as detecting criminal activities of various kinds (e.g. in electronic commerce), rare events, deviations from the majority, or exceptional cases may be more interesting and useful than the com-

大多数KDD研究侧重于发现适用于数据集中相当一部分对象的模式。然而，对于诸如检测各种犯罪活动（例如电子商务中的犯罪活动）等应用，罕见事件、与大多数情况的偏差或特殊情况可能比普

mon cases. Finding such exceptions and outliers, however, has not yet received as much attention in the KDD community as some other topics have, e.g. association rules.

在许多情况下。然而，在知识发现与数据挖掘（KDD）领域，寻找此类异常和离群值的研究尚未像关联规则等其他主题那样受到广泛关注。

Recently, a few studies have been conducted on outlier detection for large datasets (e.g. [18], [1], [13], [14]). While a more detailed discussion on these studies will be given in section 2 , it suffices to point out here that most of these studies consider being an outlier as a binary property. That is, either an object in the dataset is an outlier or not. For many applications, the situation is more complex. And it becomes more meaningful to assign to each object a degree of being an outlier.

最近，已经针对大型数据集的离群值检测开展了一些研究（例如 [18]、[1]、[13]、[14]）。虽然将在第 2 节对这些研究进行更详细的讨论，但在此指出以下这点就足够了：这些研究大多将离群值视为一个二元属性。也就是说，数据集中的某个对象要么是离群值，要么不是。对于许多应用而言，情况更为复杂。为每个对象赋予一个离群程度会更有意义。

Also related to outlier detection is an extensive body of work on clustering algorithms. From the viewpoint of a clustering algorithm, outliers are objects not located in clusters of a dataset, usually called noise. The set of noise produced by a clustering algorithm, however, is highly dependent on the particular algorithm and on its clustering parameters. Only a few approaches are directly concerned with outlier detection. These algorithms, in general, consider outliers from a more global perspective, which also has some major drawbacks. These drawbacks are discussed in detail in section 2 and section 3. Furthermore, based on these clustering algorithms, the property of being an outlier is again binary.

与离群值检测相关的还有大量关于聚类算法的研究。从聚类算法的角度来看，离群值是指不在数据集聚类中的对象，通常被称为噪声。然而，聚类算法产生的噪声集高度依赖于特定的算法及其聚类参数。只有少数方法直接关注离群值检测。一般来说，这些算法从更全局的视角考虑离群值，这也存在一些主要缺点。这些缺点将在第 2 节和第 3 节详细讨论。此外，基于这些聚类算法，离群值属性仍然是二元的。

In this paper, we introduce a new method for finding outliers in a multidimensional dataset. We introduce a local outlier (LOF) for each object in the dataset, indicating its degree of outlier-ness. This is, to the best of our knowledge, the first concept of an outlier which also quantifies how outlying an object is. The outlier factor is local in the sense that only a restricted neighborhood of each object is taken into account. Our approach is loosely related to density-based clustering. However, we do not require any explicit or implicit notion of clusters for our method. Specifically, our technical contributions in this paper are as follow:

在本文中，我们介绍了一种在多维数据集中寻找离群值的新方法。我们为数据集中的每个对象引入了局部离群因子（LOF），以表明其离群程度。据我们所知，这是首个对对象离群程度进行量化的离群值概念。离群因子是局部的，因为只考虑每个对象的有限邻域。我们的方法与基于密度的聚类有一定关联。然而，我们的方法不需要任何显式或隐式的聚类概念。具体而言，本文的技术贡献如下：

- After introducing the concept of LOF, we analyze the formal properties of LOF. We show that for most objects in a cluster their LOF are approximately equal to 1 . For any other object, we give a lower and upper bound on its LOF. These bounds highlight the local nature of LOF. Furthermore, we analyze when these bounds are tight. We identify classes of objects for which the bounds are tight. Finally, for those objects for which the bounds are not tight, we provide sharper bounds.

- 在引入局部离群因子（LOF）的概念后，我们分析了 LOF 的形式属性。我们表明，对于聚类中的大多数对象，其 LOF 近似等于 1。对于其他任何对象，我们给出了其 LOF 的上下界。这些界限凸显了 LOF 的局部性质。此外，我们分析了这些界限何时是紧的。我们确定了界限为紧的对象类别。最后，对于那些界限不紧的对象，我们提供了更精确的界限。

- The LOF of an object is based on the single parameter of MinPts, which is the number of nearest neighbors used in defining the local neighborhood of the object. We study how this parameter affects the LOF value, and we present practical guidelines for choosing the MinPts values for finding local outliers.

- 对象的局部离群因子（LOF）基于单一参数 MinPts，它是定义对象局部邻域时使用的最近邻数量。我们研究了该参数如何影响 LOF 值，并给出了为寻找局部离群值选择 MinPts 值的实用指南。

---

<!-- Footnote -->

Permission to make digital or hard copies of part or all of this work or personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. To copy otherwise, to republish, to post on servers, or to redistribute to lists, requires prior specific permission and/or a fee.

允许在不收取费用的情况下，为个人使用或课堂教学目的制作本作品部分或全部内容的数字或硬拷贝，但前提是这些拷贝不得用于盈利或商业目的，并且必须在首页注明此声明和完整的引用信息。如需以其他方式复制、重新发布、上传到服务器或分发给列表，则需要事先获得特定许可和/或支付费用。

MOD 2000, Dallas, TX USA

2000 年数据挖掘与知识发现国际会议（MOD 2000），美国得克萨斯州达拉斯市

© ACM 2000 1-58113-218-2/00/05 . . .\$5.00

© 美国计算机协会（ACM）2000 1 - 58113 - 218 - 2/00/05... 5.00 美元

<!-- Footnote -->

---

- Last but not least, we present experimental results which show both the capability and the performance of finding local outliers. We conclude that finding local outliers using LOF is meaningful and efficient.

- 最后但同样重要的是，我们展示了实验结果，这些结果表明了寻找局部离群值的能力和性能。我们得出结论，使用局部离群因子（LOF）寻找局部离群值是有意义且高效的。

The paper is organized as follows. In section 2, we discuss related work on outlier detection and their drawbacks. In section 3 we discuss in detail the motivation of our notion of outliers, especially, the advantage of a local instead of a global view on outliers. In section 4 we introduce LOF and define other auxiliary notions. In section 5 we analyze thoroughly the formal properties of LOF. Since LOF requires the single parameter MinPts, in section 6 we analyze the impact of the parameter, and discuss ways to choose MinPts values for LOF computation. In section 7 we perform an extensive experimental evaluation.

本文的组织结构如下。在第 2 节中，我们讨论离群值检测的相关工作及其缺点。在第 3 节中，我们详细讨论我们的离群值概念的动机，特别是从局部而非全局视角看待离群值的优势。在第 4 节中，我们引入局部离群因子（LOF）并定义其他辅助概念。在第 5 节中，我们全面分析 LOF 的形式属性。由于 LOF 需要单一参数 MinPts，在第 6 节中，我们分析该参数的影响，并讨论为 LOF 计算选择 MinPts 值的方法。在第 7 节中，我们进行广泛的实验评估。

## 2. RELATED WORK

## 2. 相关工作

Most of the previous studies on outlier detection were conducted in the field of statistics. These studies can be broadly classified into two categories. The first category is distribution-based, where a standard distribution (e.g. Normal, Poisson, etc.) is used to fit the data best. Outliers are defined based on the probability distribution. Over one hundred tests of this category, called discordancy tests, have been developed for different scenarios (see [5]). A key drawback of this category of tests is that most of the distributions used are univariate. There are some tests that are multivariate (e.g. multivariate normal outliers). But for many KDD applications, the underlying distribution is unknown. Fitting the data with standard distributions is costly, and may not produce satisfactory results.

以往大多数关于离群值检测的研究是在统计学领域进行的。这些研究大致可分为两类。第一类是基于分布的，即使用标准分布（例如正态分布、泊松分布等）来最佳拟合数据。离群值是根据概率分布来定义的。针对不同场景，已经开发了超过一百种这类测试，称为不一致性测试（见 [5]）。这类测试的一个关键缺点是，所使用的大多数分布是单变量的。有一些测试是多变量的（例如多元正态离群值）。但对于许多知识发现与数据挖掘（KDD）应用而言，底层分布是未知的。用标准分布拟合数据成本高昂，并且可能无法产生令人满意的结果。

The second category of outlier studies in statistics is depth-based. Each data object is represented as a point in a k-d space,and is assigned a depth. With respect to outlier detection, outliers are more likely to be data objects with smaller depths. There are many definitions of depth that have been proposed (e.g. [20], [16]). In theory, depth-based approaches could work for large values of $\mathrm{k}$ . However, in practice,while there exist efficient algorithms for $\mathrm{k} = 2$ or 3 ([16], [18], [12]), depth-based approaches become inefficient for large datasets for $k \geq  4$ . This is because depth-based approaches rely on the computation of $\mathrm{k}$ -d convex hulls which has a lower bound complexity of $\Omega \left( {\mathrm{n}}^{\mathrm{k}/2}\right)$ for $\mathrm{n}$ objects.

统计学中离群点研究的第二类是基于深度的。每个数据对象在k维空间中表示为一个点，并被赋予一个深度值。在离群点检测方面，离群点更有可能是深度值较小的数据对象。目前已经提出了许多关于深度的定义（例如[20]、[16]）。理论上，基于深度的方法可以用于较大的$\mathrm{k}$值。然而在实践中，虽然存在适用于$\mathrm{k} = 2$维或3维的高效算法（[16]、[18]、[12]），但对于较大的$k \geq  4$值和大型数据集，基于深度的方法效率较低。这是因为基于深度的方法依赖于$\mathrm{k}$维凸包的计算，对于$\mathrm{n}$个对象，其复杂度下限为$\Omega \left( {\mathrm{n}}^{\mathrm{k}/2}\right)$。

Recently, $\mathrm{{Knorr}}$ and $\mathrm{{Ng}}$ proposed the notion of distance-based outliers [13], [14]. Their notion generalizes many notions from the distribution-based approaches, and enjoys better computational complexity than the depth-based approaches for larger values of $\mathrm{k}$ . Later in section 3 , we will discuss in detail how their notion is different from the notion of local outliers proposed in this paper. In [17] the notion of distance based outliers is extended by using the distance to the k-nearest neighbor to rank the outliers. A very efficient algorithms to compute the top $n$ outliers in this ranking is given,but their notion of an outlier is still distance-based.

最近，$\mathrm{{Knorr}}$和$\mathrm{{Ng}}$提出了基于距离的离群点概念[13]、[14]。他们的概念对基于分布的方法中的许多概念进行了推广，并且对于较大的$\mathrm{k}$值，其计算复杂度比基于深度的方法更优。在第3节中，我们将详细讨论他们的概念与本文提出的局部离群点概念有何不同。在文献[17]中，基于距离的离群点概念通过使用到k近邻的距离对离群点进行排序得到了扩展。文中给出了一种非常高效的算法来计算该排序中的前$n$个离群点，但他们的离群点概念仍然是基于距离的。

Given the importance of the area, fraud detection has received more attention than the general area of outlier detection. Depending on the specifics of the application domains, elaborate fraud models and fraud detection algorithms have been developed (e.g. [8], [6]). In contrast to fraud detection, the kinds of outlier detection work discussed so far are more exploratory in nature. Outlier detection may indeed lead to the construction of fraud models.

鉴于该领域的重要性，欺诈检测比一般的离群点检测领域受到了更多关注。根据应用领域的具体情况，已经开发出了精细的欺诈模型和欺诈检测算法（例如[8]、[6]）。与欺诈检测不同，到目前为止所讨论的离群点检测工作本质上更具探索性。离群点检测确实可能会促使欺诈模型的构建。

Finally, most clustering algorithms, especially those developed in the context of KDD (e.g. CLARANS [15], DBSCAN [7], BIRCH [23], STING [22], WaveCluster [19], DenClue [11], CLIQUE [3]), are to some extent capable of handling exceptions. However, since the main objective of a clustering algorithm is to find clusters, they are developed to optimize clustering, and not to optimize outlier detection. The exceptions (called "noise" in the context of clustering) are typically just tolerated or ignored when producing the clustering result. Even if the outliers are not ignored, the notions of outliers are essentially binary, and there are no quantification as to how outlying an object is. Our notion of local outliers share a few fundamental concepts with density-based clustering approaches. However, our outlier detection method does not require any explicit or implicit notion of clusters.

最后，大多数聚类算法，尤其是那些在知识发现与数据挖掘（KDD）背景下开发的算法（例如CLARANS [15]、DBSCAN [7]、BIRCH [23]、STING [22]、WaveCluster [19]、DenClue [11]、CLIQUE [3]），在一定程度上能够处理异常值。然而，由于聚类算法的主要目标是寻找聚类，它们是为优化聚类而开发的，而不是为了优化离群点检测。在生成聚类结果时，异常值（在聚类上下文中称为“噪声”）通常只是被容忍或忽略。即使不忽略离群点，离群点的概念本质上也是二元的，并且没有对一个对象的离群程度进行量化。我们提出的局部离群点概念与基于密度的聚类方法有一些基本概念是相同的。然而，我们的离群点检测方法不需要任何显式或隐式的聚类概念。

## 3. PROBLEMS OF EXISTING (NON-LOCAL) APPROACHES

## 3. 现有（非局部）方法的问题

As we have seen in section 2, most of the existing work in outlier detection lies in the field of statistics. Intuitively, outliers can be defined as given by Hawkins [10].

正如我们在第2节中所看到的，现有的离群点检测工作大多属于统计学领域。直观地说，离群点可以按照霍金斯（Hawkins）[10]给出的方式来定义。

Definition 1: (Hawkins-Outlier)

定义1：（霍金斯离群点）

An outlier is an observation that deviates so much from other observations as to arouse suspicion that it was generated by a different mechanism.

离群点是指与其他观测值偏差极大，以至于让人怀疑它是由不同机制生成的观测值。

This notion is formalized by Knorr and Ng [13] in the following definition of outliers. Throughout this paper,we use o, $\mathrm{p},\mathrm{q}$ to denote objects in a dataset. We use the notation $\mathrm{d}\left( {\mathrm{p},\mathrm{q}}\right)$ to denote the distance between objects $\mathrm{p}$ and $\mathrm{q}$ . For a set of objects,we use $\mathrm{C}$ (sometimes with the intuition that $\mathrm{C}$ forms a cluster). To simplify our notation,we use $\mathrm{d}\left( {\mathrm{p},\mathrm{C}}\right)$ to denote the minimum distance between $p$ and object $q$ in $C$ ,i.e. $d\left( {p,C}\right)  = \min \{ d\left( {p,q}\right)  \mid  q \in  C\}$ .

诺尔（Knorr）和吴（Ng）[13]在以下离群点定义中对这一概念进行了形式化。在本文中，我们用o、$\mathrm{p},\mathrm{q}$表示数据集中的对象。我们用符号$\mathrm{d}\left( {\mathrm{p},\mathrm{q}}\right)$表示对象$\mathrm{p}$和$\mathrm{q}$之间的距离。对于一组对象，我们用$\mathrm{C}$表示（有时可以直观地认为$\mathrm{C}$构成一个聚类）。为了简化符号，我们用$\mathrm{d}\left( {\mathrm{p},\mathrm{C}}\right)$表示$p$与$C$中对象$q$之间的最小距离，即$d\left( {p,C}\right)  = \min \{ d\left( {p,q}\right)  \mid  q \in  C\}$。

Definition 2: (DB(pct, dmin)-Outlier)

定义2：（DB(pct, dmin) - 离群点）

An object p in a dataset $\mathrm{D}$ is a $\mathrm{{DB}}\left( {\mathrm{{pct}},\mathrm{{dmin}}}\right)$ -outlier if at least percentage pct of the objects in D lies greater than distance dmin from $\mathrm{p}$ ,i.e.,the cardinality of the set $\{ \mathrm{q} \in  \mathrm{D} \mid  \mathrm{d}\left( {\mathrm{p},\mathrm{q}}\right)  \leq$ $\mathrm{{dmin}}\}$ is less than or equal to $\left( {{100} - \mathrm{{pct}}}\right) \%$ of the size of $\mathrm{D}$ .

数据集中 $\mathrm{D}$ 的对象 p 是一个 $\mathrm{{DB}}\left( {\mathrm{{pct}},\mathrm{{dmin}}}\right)$ -离群点（outlier），如果数据集中至少 pct 百分比的对象与 $\mathrm{p}$ 的距离大于 dmin，即集合 $\{ \mathrm{q} \in  \mathrm{D} \mid  \mathrm{d}\left( {\mathrm{p},\mathrm{q}}\right)  \leq$ $\mathrm{{dmin}}\}$ 的基数小于或等于 $\mathrm{D}$ 大小的 $\left( {{100} - \mathrm{{pct}}}\right) \%$ 。

The above definition captures only certain kinds of outliers. Because the definition takes a global view of the dataset, these outliers can be viewed as "global" outliers. However, for many interesting real-world datasets which exhibit a more complex structure, there is another kind of outliers. These can be objects that are outlying relative to their local neighborhoods, particularly with respect to the densities of the neighborhoods. These outliers are regarded as "local" outliers.

上述定义仅涵盖了某些类型的离群点。由于该定义从全局视角审视数据集，这些离群点可被视为“全局”离群点。然而，对于许多具有更复杂结构的现实世界数据集，还存在另一种离群点。这些离群点可能是相对于其局部邻域而言偏离的对象，特别是在邻域密度方面。这些离群点被视为“局部”离群点。

<!-- Media -->

<!-- figureText: 92 . ${}^{0}1$ -->

<img src="https://cdn.noedgeai.com/0195c8fe-047f-7311-b009-9bd967908604_1.jpg?x=989&y=1731&w=394&h=349&r=0"/>

Figure 1: 2-d dataset DS1

图 1：二维数据集 DS1

<!-- Media -->

To illustrate, consider the example given in Figure 1. This is a simple 2-dimensional dataset containing 502 objects. There are 400 objects in the first cluster ${\mathrm{C}}_{1},{100}$ objects in the cluster ${\mathrm{C}}_{2}$ ,and two additional objects ${\mathrm{o}}_{1}$ and ${\mathrm{o}}_{2}$ . In this example, ${\mathrm{C}}_{2}$ forms a denser cluster than ${\mathrm{C}}_{1}$ . According to Hawkins’ definition,both o1 and o2 can be called outliers,whereas objects in ${\mathrm{C}}_{1}$ and ${\mathrm{C}}_{2}$ should not be. With our notion of a "local" outlier,we wish to label both ${\mathrm{o}}_{1}$ and ${\mathrm{o}}_{2}$ as outliers. In contrast, within the framework of distance-based outliers,only ${\mathrm{o}}_{1}$ is a reasonable $\mathrm{{DB}}\left( {\mathrm{{pct}},\mathrm{{dmin}}}\right)$ -outlier in the following sense. If for every object $q$ in ${C}_{1}$ ,the distance between $q$ and its nearest neighbor is greater than the distance between ${\mathrm{o}}_{2}$ and ${\mathrm{C}}_{2}$ (i.e., $\left. \left( {{\mathrm{o}}_{2},{\mathrm{C}}_{2}}\right) \right)$ ,we can in fact show that there is no appropriate value of pct and dmin such that ${\mathrm{o}}_{2}$ is a $\mathrm{{DB}}\left( {\mathrm{{pct}},\mathrm{{dmin}}}\right)$ -outlier but the the objects in ${\mathrm{C}}_{1}$ are not.

为了说明这一点，考虑图 1 中的示例。这是一个包含 502 个对象的简单二维数据集。第一个簇中有 400 个对象，簇 ${\mathrm{C}}_{2}$ 中有 ${\mathrm{C}}_{1},{100}$ 个对象，还有两个额外的对象 ${\mathrm{o}}_{1}$ 和 ${\mathrm{o}}_{2}$ 。在这个例子中，${\mathrm{C}}_{2}$ 形成的簇比 ${\mathrm{C}}_{1}$ 更密集。根据霍金斯（Hawkins）的定义，${\mathrm{o}}_{1}$ 和 ${\mathrm{o}}_{2}$ 都可以被称为离群点，而 ${\mathrm{C}}_{1}$ 和 ${\mathrm{C}}_{2}$ 中的对象则不应被称为离群点。根据我们“局部”离群点的概念，我们希望将 ${\mathrm{o}}_{1}$ 和 ${\mathrm{o}}_{2}$ 都标记为离群点。相比之下，在基于距离的离群点框架下，只有 ${\mathrm{o}}_{1}$ 是合理的 $\mathrm{{DB}}\left( {\mathrm{{pct}},\mathrm{{dmin}}}\right)$ -离群点，具体含义如下。如果对于 ${C}_{1}$ 中的每个对象 $q$ ，$q$ 与其最近邻的距离大于 ${\mathrm{o}}_{2}$ 与 ${\mathrm{C}}_{2}$ 之间的距离（即 $\left. \left( {{\mathrm{o}}_{2},{\mathrm{C}}_{2}}\right) \right)$ ），我们实际上可以证明，不存在合适的 pct 和 dmin 值，使得 ${\mathrm{o}}_{2}$ 是 $\mathrm{{DB}}\left( {\mathrm{{pct}},\mathrm{{dmin}}}\right)$ -离群点，而 ${\mathrm{C}}_{1}$ 中的对象不是。

The reason is as follows. If the dmin value is less than the distance $\mathrm{d}\left( {{\mathrm{o}}_{2},{\mathrm{C}}_{2}}\right)$ ,then all 501 objects (pct $= {100} * {501}/{502}$ ) are further away from ${\mathrm{o}}_{2}$ than dmin. But the same condition holds also for every object $q$ in ${\mathrm{C}}_{1}$ . Thus,in this case, ${\mathrm{o}}_{2}$ and all objects in ${\mathrm{C}}_{1}$ are $\mathrm{{DB}}(\mathrm{{pct}}$ , dmin)-outliers.

原因如下。如果 dmin 值小于距离 $\mathrm{d}\left( {{\mathrm{o}}_{2},{\mathrm{C}}_{2}}\right)$ ，那么所有 501 个对象（pct $= {100} * {501}/{502}$ ）与 ${\mathrm{o}}_{2}$ 的距离都大于 dmin。但对于 ${\mathrm{C}}_{1}$ 中的每个对象 $q$ ，同样的条件也成立。因此，在这种情况下，${\mathrm{o}}_{2}$ 和 ${\mathrm{C}}_{1}$ 中的所有对象都是（$\mathrm{{DB}}(\mathrm{{pct}}$ ，dmin）-离群点。

Otherwise,if the dmin value is greater than the distance $\mathrm{d}\left( {{\mathrm{o}}_{2},{\mathrm{C}}_{2}}\right)$ , then it is easy to see that: ${\mathrm{o}}_{2}$ is a $\mathrm{{DB}}\left( {\mathrm{{pct}},\mathrm{{dmin}}}\right)$ -outlier implies that there are many objects q in ${\mathrm{C}}_{1}$ such that q is also a $\mathrm{{DB}}\left( {\mathrm{{pct}},\mathrm{{dmin}}}\right)$ - outlier. This is because the cardinality of the set $\left\{  {\mathrm{p} \in  \mathrm{D} \mid  \mathrm{d}\left( {\mathrm{p},{\mathrm{o}}_{2}}\right)  \leq  \mathrm{{dmin}}}\right\}$ is always bigger than the cardinality of the set $\{ \mathrm{p} \in  \mathrm{D} \mid  \mathrm{d}\left( {\mathrm{p},\mathrm{q}}\right)  \leq  \mathrm{{dmin}}\}$ . Thus,in this case,if ${\mathrm{o}}_{2}$ is a $\mathrm{{DB}}\left( {\mathrm{{pct}},\mathrm{{dmin}}}\right)$ -outlier,so are many objects $\mathrm{q}$ in ${\mathrm{C}}_{1}$ . Worse still, there are values of pct and dmin such that while ${\mathrm{o}}_{2}$ is not an outlier, some q in ${\mathrm{C}}_{1}$ are.

否则，如果dmin值大于距离$\mathrm{d}\left( {{\mathrm{o}}_{2},{\mathrm{C}}_{2}}\right)$，那么很容易看出：${\mathrm{o}}_{2}$是一个$\mathrm{{DB}}\left( {\mathrm{{pct}},\mathrm{{dmin}}}\right)$ - 离群点意味着在${\mathrm{C}}_{1}$中有许多对象q也是$\mathrm{{DB}}\left( {\mathrm{{pct}},\mathrm{{dmin}}}\right)$ - 离群点。这是因为集合$\left\{  {\mathrm{p} \in  \mathrm{D} \mid  \mathrm{d}\left( {\mathrm{p},{\mathrm{o}}_{2}}\right)  \leq  \mathrm{{dmin}}}\right\}$的基数总是大于集合$\{ \mathrm{p} \in  \mathrm{D} \mid  \mathrm{d}\left( {\mathrm{p},\mathrm{q}}\right)  \leq  \mathrm{{dmin}}\}$的基数。因此，在这种情况下，如果${\mathrm{o}}_{2}$是一个$\mathrm{{DB}}\left( {\mathrm{{pct}},\mathrm{{dmin}}}\right)$ - 离群点，那么${\mathrm{C}}_{1}$中的许多对象$\mathrm{q}$也是。更糟糕的是，存在一些pct和dmin的值，使得虽然${\mathrm{o}}_{2}$不是离群点，但${\mathrm{C}}_{1}$中的一些q是离群点。

## 4. FORMAL DEFINITION OF LOCAL OUTLIERS

## 4. 局部离群点的形式化定义

The above example shows that the global view taken by DB(pct, dmin)-outliers is meaningful and adequate under certain conditions, but not satisfactory for the general case when clusters of different densities exist. In this section, we develop a formal definition of local outliers, which avoids the shortcomings presented in the previous section. The key difference between our notion and existing notions of outliers is that being outlying is not a binary property. Instead, we assign to each object an outlier factor, which is the degree the object is being outlying.

上述示例表明，DB(pct, dmin) - 离群点所采用的全局视角在某些条件下是有意义且合适的，但当存在不同密度的聚类时，对于一般情况并不令人满意。在本节中，我们给出局部离群点的形式化定义，以避免上一节中出现的缺点。我们的概念与现有的离群点概念之间的关键区别在于，离群性不是一个二元属性。相反，我们为每个对象分配一个离群因子，它表示该对象的离群程度。

We begin with the notions of the k -distance of object p, and, correspondingly,the $\mathrm{k}$ -distance neighborhood of $\mathrm{p}$ .

我们从对象p的k - 距离的概念开始，相应地，还有$\mathrm{p}$的$\mathrm{k}$ - 距离邻域。

Definition 3: (k-distance of an object p)

定义3：（对象p的k - 距离）

For any positive integer $\mathrm{k}$ ,the $\mathrm{k}$ -distance of object $\mathrm{p}$ ,denoted as $\mathrm{k}$ -distance(p),is defined as the distance $\mathrm{d}\left( {\mathrm{p},\mathrm{o}}\right)$ between $\mathrm{p}$ and an object $\mathrm{o} \in  \mathrm{D}$ such that:

对于任何正整数$\mathrm{k}$，对象$\mathrm{p}$的$\mathrm{k}$ - 距离，记为$\mathrm{k}$ - distance(p)，定义为$\mathrm{p}$与对象$\mathrm{o} \in  \mathrm{D}$之间的距离$\mathrm{d}\left( {\mathrm{p},\mathrm{o}}\right)$，使得：

(i) for at least $\mathrm{k}$ objects o’ $\in  \mathrm{D} \smallsetminus  \{ \mathrm{p}\}$ it holds that $\mathrm{d}\left( {\mathrm{p},{\mathrm{o}}^{\prime }}\right)  \leq  \mathrm{d}\left( {\mathrm{p},\mathrm{o}}\right)$ ,and

（i）对于至少$\mathrm{k}$个对象o’ $\in  \mathrm{D} \smallsetminus  \{ \mathrm{p}\}$，有$\mathrm{d}\left( {\mathrm{p},{\mathrm{o}}^{\prime }}\right)  \leq  \mathrm{d}\left( {\mathrm{p},\mathrm{o}}\right)$，并且

(ii) for at most $\mathrm{k} - 1$ objects $\mathrm{o}$ ’ $\in  \mathrm{D} \smallsetminus  \{ \mathrm{p}\}$ it holds that $\mathrm{d}\left( {\mathrm{p},{\mathrm{o}}^{\prime }}\right)  < \mathrm{d}\left( {\mathrm{p},\mathrm{o}}\right)$ .

（ii）对于至多$\mathrm{k} - 1$个对象$\mathrm{o}$’ $\in  \mathrm{D} \smallsetminus  \{ \mathrm{p}\}$，有$\mathrm{d}\left( {\mathrm{p},{\mathrm{o}}^{\prime }}\right)  < \mathrm{d}\left( {\mathrm{p},\mathrm{o}}\right)$。

<!-- Media -->

<!-- figureText: reach-dis ${\mathrm{t}}_{\mathrm{k}}\left( {{\mathrm{p}}_{2},\mathrm{o}}\right)$ y ${}_{2}$ -->

<img src="https://cdn.noedgeai.com/0195c8fe-047f-7311-b009-9bd967908604_2.jpg?x=938&y=259&w=487&h=350&r=0"/>

Figure 2: reach-dist $\left( {{\mathrm{p}}_{1},\mathrm{o}}\right)$ and reach-dist $\left( {{\mathrm{p}}_{2},\mathrm{o}}\right)$ ,for $\mathrm{k} = 4$

图2：对于$\mathrm{k} = 4$的可达距离$\left( {{\mathrm{p}}_{1},\mathrm{o}}\right)$和可达距离$\left( {{\mathrm{p}}_{2},\mathrm{o}}\right)$

<!-- Media -->

Definition 4: (k-distance neighborhood of an object p)

定义4：（对象p的k - 距离邻域）

Given the k-distance of $\mathrm{p}$ ,the k-distance neighborhood of $\mathrm{p}$ contains every object whose distance from $\mathrm{p}$ is not greater than the $\mathrm{k}$ -distance,i.e. ${\mathrm{N}}_{\mathrm{k}\text{-distance}\left( \mathrm{p}\right) }\left( \mathrm{p}\right)  = \{ \mathrm{q} \in  \mathrm{D} \smallsetminus  \{ \mathrm{p}\}  \mid  \mathrm{d}\left( {\mathrm{p},\mathrm{q}}\right)  \leq  \mathrm{k}$ - distance(p) \}.

给定$\mathrm{p}$的k-距离，$\mathrm{p}$的k-距离邻域包含所有与$\mathrm{p}$的距离不大于$\mathrm{k}$-距离的对象，即${\mathrm{N}}_{\mathrm{k}\text{-distance}\left( \mathrm{p}\right) }\left( \mathrm{p}\right)  = \{ \mathrm{q} \in  \mathrm{D} \smallsetminus  \{ \mathrm{p}\}  \mid  \mathrm{d}\left( {\mathrm{p},\mathrm{q}}\right)  \leq  \mathrm{k}$ - distance(p) \}。

These objects $q$ are called the $k$ -nearest neighbors of $p$ .

这些对象$q$被称为$p$的$k$-最近邻。

Whenever no confusion arises, we simplify our notation to use ${\mathrm{N}}_{\mathrm{k}}\left( \mathrm{p}\right)$ as a shorthand for ${\mathrm{N}}_{\mathrm{k}\text{-distance}\left( \mathrm{p}\right) }\left( \mathrm{p}\right)$ . Note that in definition 3, the $\mathrm{k}$ -distance(p) is well defined for any positive integer $\mathrm{k}$ ,although the object o may not be unique. In this case,the cardinality of ${\mathrm{N}}_{\mathrm{k}}\left( \mathrm{p}\right)$ is greater than k. For example,suppose that there are: (i) 1 object with distance 1 unit from p; (ii) 2 objects with distance 2 units from p; and (iii) 3 objects with distance 3 units from p. Then 2-distance(p) is identical to 3-distance(p). And there are 3 objects of 4- distance(p) from p. Thus,the cardinality of ${\mathrm{N}}_{4}\left( \mathrm{p}\right)$ can be greater than 4 , in this case 6 .

只要不会引起混淆，我们简化符号，用${\mathrm{N}}_{\mathrm{k}}\left( \mathrm{p}\right)$作为${\mathrm{N}}_{\mathrm{k}\text{-distance}\left( \mathrm{p}\right) }\left( \mathrm{p}\right)$的简写。注意，在定义3中，对于任何正整数$\mathrm{k}$，$\mathrm{k}$ - distance(p)都有明确定义，尽管对象o可能不唯一。在这种情况下，${\mathrm{N}}_{\mathrm{k}}\left( \mathrm{p}\right)$的基数大于k。例如，假设存在：(i) 1个与p的距离为1个单位的对象；(ii) 2个与p的距离为2个单位的对象；以及(iii) 3个与p的距离为3个单位的对象。那么2 - distance(p)与3 - distance(p)相同。并且有3个与p的距离为4 - distance(p)的对象。因此，${\mathrm{N}}_{4}\left( \mathrm{p}\right)$的基数可以大于4，在这种情况下为6。

Definition 5: (reachability distance of an object p w.r.t. object o)

定义5：（对象p相对于对象o的可达距离）

Let $k$ be a natural number. The reachability distance of object $\mathrm{p}$ with respect to object $\mathrm{o}$ is defined as

设$k$为自然数。对象$\mathrm{p}$相对于对象$\mathrm{o}$的可达距离定义为

reach-dis ${\mathrm{t}}_{\mathrm{k}}\left( {\mathrm{p},\mathrm{o}}\right)  = \max \{ \mathrm{k}$ -distance $\left( \mathrm{o}\right) ,\mathrm{d}\left( {\mathrm{p},\mathrm{o}}\right) \}$ .

可达距离 ${\mathrm{t}}_{\mathrm{k}}\left( {\mathrm{p},\mathrm{o}}\right)  = \max \{ \mathrm{k}$ - 距离 $\left( \mathrm{o}\right) ,\mathrm{d}\left( {\mathrm{p},\mathrm{o}}\right) \}$。

Figure 2 illustrates the idea of reachability distance with $\mathrm{k} = 4$ . Intuitively,if object $\mathrm{p}$ is far away from $\mathrm{o}$ (e.g. ${\mathrm{p}}_{2}$ in the figure),then the reachability distance between the two is simply their actual distance. However,if they are "sufficiently" close (e.g., ${p}_{1}$ in the figure),the actual distance is replaced by the k - distance of o. The reason is that in so doing,the statistical fluctuations of $\mathrm{d}\left( {\mathrm{p},\mathrm{o}}\right)$ for all the p's close to o can be significantly reduced. The strength of this smoothing effect can be controlled by the parameter $\mathrm{k}$ . The higher the value of $\mathrm{k}$ ,the more similar the reachability distances for objects within the same neighborhood.

图2说明了$\mathrm{k} = 4$的可达距离的概念。直观地说，如果对象$\mathrm{p}$离$\mathrm{o}$很远（例如图中的${\mathrm{p}}_{2}$），那么这两个对象之间的可达距离就是它们的实际距离。然而，如果它们“足够”接近（例如图中的${p}_{1}$），实际距离就会被o的k - 距离所取代。原因是这样做可以显著减少所有接近o的p的$\mathrm{d}\left( {\mathrm{p},\mathrm{o}}\right)$的统计波动。这种平滑效果的强度可以通过参数$\mathrm{k}$来控制。$\mathrm{k}$的值越高，同一邻域内对象的可达距离就越相似。

So far,we have defined $\mathrm{k}$ -distance(p) and reach-dist ${}_{\mathrm{k}}\left( \mathrm{p}\right)$ for any positive integer $\mathrm{k}$ . But for the purpose of defining outliers,we focus on a specific instantiation of $\mathrm{k}$ which links us back to density-based clustering. In a typical density-based clustering algorithm, such as $\left\lbrack  7\right\rbrack  ,\left\lbrack  3\right\rbrack  ,\left\lbrack  {22}\right\rbrack$ ,or [11],there are two parameters that define the notion of density: (i) a parameter MinPts specifying a minimum number of objects; (ii) a parameter specifying a volume. These two parameters determine a density threshold for the clustering algorithms to operate. That is, objects or regions are connected if their neighborhood densities exceed the given density threshold. To detect density-based outliers, however, it is necessary to compare the densities of different sets of objects, which means that we have to determine the density of sets of objects dynamically. Therefore, we keep MinPts as the only parameter and use the values reach-dist ${}_{\text{MinPts }}\left( {\mathrm{p},\mathrm{o}}\right)$ ,for $\mathrm{o} \in  {\mathrm{N}}_{\text{MinPts }}\left( \mathrm{p}\right)$ ,as a measure of the volume to determine the density in the neighborhood of an object p.

到目前为止，我们已经为任意正整数 $\mathrm{k}$ 定义了 $\mathrm{k}$ -距离(p)和可达距离 ${}_{\mathrm{k}}\left( \mathrm{p}\right)$。但为了定义离群点，我们关注 $\mathrm{k}$ 的一个特定实例，它将我们与基于密度的聚类联系起来。在典型的基于密度的聚类算法中，如 $\left\lbrack  7\right\rbrack  ,\left\lbrack  3\right\rbrack  ,\left\lbrack  {22}\right\rbrack$ 或文献[11]，有两个参数定义了密度的概念：(i) 参数 MinPts，指定对象的最小数量；(ii) 一个指定体积的参数。这两个参数为聚类算法的运行确定了一个密度阈值。也就是说，如果对象或区域的邻域密度超过给定的密度阈值，它们就会被连接起来。然而，为了检测基于密度的离群点，有必要比较不同对象集的密度，这意味着我们必须动态地确定对象集的密度。因此，我们将 MinPts 作为唯一的参数，并使用可达距离 ${}_{\text{MinPts }}\left( {\mathrm{p},\mathrm{o}}\right)$（其中 $\mathrm{o} \in  {\mathrm{N}}_{\text{MinPts }}\left( \mathrm{p}\right)$）的值作为体积的度量，来确定对象 p 邻域的密度。

Definition 6: (local reachability density of an object p)

定义 6：（对象 p 的局部可达密度）

The local reachability density of $\mathrm{p}$ is defined as

$\mathrm{p}$ 的局部可达密度定义为

$$
{\operatorname{lrd}}_{\text{MinPts }}\left( p\right)  = 1/\left( \frac{\mathop{\sum }\limits_{{o \in  {N}_{\text{MinPts }}\left( p\right) }}{\operatorname{reach}}_{\text{MinPts }}\left( {p,o}\right) }{\left| {N}_{\text{MinPts }}\left( p\right) \right| }\right) 
$$

Intuitively,the local reachability density of an object $\mathrm{p}$ is the inverse of the average reachability distance based on the MinPts-nearest neighbors of p. Note that the local density can be $\infty$ if all the reachability distances in the summation are 0 . This may occur for an object $\mathrm{p}$ if there are at least MinPts objects,different from $\mathrm{p}$ ,but sharing the same spatial coordinates, i.e. if there are at least MinPts duplicates of $\mathrm{p}$ in the dataset. For simplicity,we will not handle this case explicitly but simply assume that there are no duplicates. (To deal with duplicates, we can base our notion of neighborhood on a $\mathrm{k}$ -distinct-distance,defined analogously to $\mathrm{k}$ -distance in definition 3 ,with the additional requirement that there be at least $\mathrm{k}$ objects with different spatial coordinates.)

直观地说，对象 $\mathrm{p}$ 的局部可达密度是基于 p 的 MinPts 最近邻的平均可达距离的倒数。请注意，如果求和中的所有可达距离都为 0，则局部密度可以为 $\infty$。如果至少有 MinPts 个不同于 $\mathrm{p}$ 但具有相同空间坐标的对象，即数据集中至少有 MinPts 个 $\mathrm{p}$ 的重复对象，那么对于对象 $\mathrm{p}$ 可能会出现这种情况。为了简单起见，我们不会明确处理这种情况，而是简单地假设没有重复对象。（为了处理重复对象，我们可以基于 $\mathrm{k}$ -相异距离来定义邻域的概念，其定义类似于定义 3 中的 $\mathrm{k}$ -距离，额外要求至少有 $\mathrm{k}$ 个具有不同空间坐标的对象。）

Definition 7: ((local) outlier factor of an object p)

定义 7：（对象 p 的（局部）离群因子）

The (local) outlier factor of $p$ is defined as

$p$ 的（局部）离群因子定义为

$$
{\mathrm{{LOF}}}_{\text{MinPts }}\left( \mathrm{p}\right)  = \frac{1}{\rho }\frac{1}{{\log }_{\text{MinPts }}\left( \mathrm{p}\right) }\frac{{\operatorname{lrd}}_{\text{MinPts }}\left( \mathrm{o}\right) }{{\operatorname{lrd}}_{\text{MinPts }}\left( \mathrm{p}\right) }
$$

The outlier factor of object p captures the degree to which we call $\mathrm{p}$ an outlier. It is the average of the ratio of the local reachability density of p and those of p's MinPts-nearest neighbors. It is easy to see that the lower p's local reachability density is, and the higher the local reachability densities of p's MinPts-nearest neighbors are, the higher is the LOF value of p. In the following section, the formal properties of LOF are made precise. To simplify notation, we drop the subscript MinPts from reach-dist, lrd and LOF, if no confusion arises.

对象 p 的离群因子反映了我们将 $\mathrm{p}$ 称为离群点的程度。它是 p 的局部可达密度与 p 的 MinPts 最近邻的局部可达密度之比的平均值。很容易看出，p 的局部可达密度越低，p 的 MinPts 最近邻的局部可达密度越高，p 的 LOF 值就越高。在下一节中，将精确阐述 LOF 的形式化性质。为了简化表示，如果不会引起混淆，我们将从可达距离、局部可达密度（lrd）和局部离群因子（LOF）中去掉下标 MinPts。

## 5. PROPERTIES OF LOCAL OUTLIERS

## 5. 局部离群点的性质

In this section, we conduct a detailed analysis on the properties of LOF. The goal is to show that our definition of LOF captures the spirit of local outliers, and enjoys many desirable properties. Specifically,we show that for most objects $\mathrm{p}$ in a cluster,the LOF of $\mathrm{p}$ is approximately equal to 1 . As for other objects, including those outside of a cluster, we give a general theorem giving a lower and upper bound on the LOF. Furthermore, we analyze the tightness of our bounds. We show that the bounds are tight for important classes of objects. However, for other classes of objects, the bounds may not be as tight. For the latter, we give another theorem specifying better bounds.

在本节中，我们对局部离群因子（LOF）的性质进行详细分析。目标是证明我们对 LOF 的定义抓住了局部离群点的本质，并具有许多理想的性质。具体来说，我们证明对于聚类中的大多数对象 $\mathrm{p}$，$\mathrm{p}$ 的 LOF 近似等于 1。对于其他对象，包括聚类外的对象，我们给出一个一般性定理，给出 LOF 的上下界。此外，我们分析这些界的紧性。我们证明对于重要的对象类，这些界是紧的。然而，对于其他类别的对象，这些界可能不那么紧。对于后者，我们给出另一个定理，指定更好的界。

### 5.1 LOF for Objects Deep in a Cluster

### 5.1 聚类内部深处对象的 LOF

In section 3, we motivate the notion of a local outlier using figure 1 . In particular,we hope to label ${\mathrm{o}}_{2}$ as outlying,but label all objects in the cluster ${\mathrm{C}}_{1}$ as non-outlying. Below,we show that for most objects in ${\mathrm{C}}_{1}$ its LOF is approximately 1,indicating that they cannot be labeled as outlying.

在第3节中，我们使用图1引出局部离群点的概念。具体来说，我们希望将${\mathrm{o}}_{2}$标记为离群点，但将簇${\mathrm{C}}_{1}$中的所有对象标记为非离群点。下面，我们将证明对于${\mathrm{C}}_{1}$中的大多数对象，其局部离群因子（LOF）近似为1，这表明它们不能被标记为离群点。

Lemma 1: Let $\mathrm{C}$ be a collection of objects. Let reach-dist-min denote the minimum reachability distance of objects in C, i.e., reach-dist-min $= \min \{$ reach-dist $\left( {\mathrm{p},\mathrm{q}}\right)  \mid  \mathrm{p},\mathrm{q} \in  \mathrm{C}\}$ . Similarly,let reach-dist-max denote the maximum reachability distance of objects in $\mathrm{C}$ . Let $\varepsilon$ be defined as (reach-dist-max/reach-dist-min -1). Then for all objects $\mathrm{p} \in  \mathrm{C}$ ,such that:

引理1：设$\mathrm{C}$为一组对象。令reach - dist - min表示$\mathrm{C}$中对象的最小可达距离，即reach - dist - min $= \min \{$ reach - dist $\left( {\mathrm{p},\mathrm{q}}\right)  \mid  \mathrm{p},\mathrm{q} \in  \mathrm{C}\}$ 。类似地，令reach - dist - max表示$\mathrm{C}$中对象的最大可达距离。设$\varepsilon$定义为（reach - dist - max / reach - dist - min - 1）。那么对于所有对象$\mathrm{p} \in  \mathrm{C}$，满足：

(i) all the MinPts-nearest neighbors q of p are in C, and

（i）[latex4]的所有MinPts最近邻[latex1]都在[latex0]中，并且

(ii) all the MinPts-nearest neighbors o of q are also in C, it holds that $1/\left( {1 + \varepsilon }\right)  \leq  \operatorname{LOF}\left( \mathrm{p}\right)  \leq  \left( {1 + \varepsilon }\right)$ .

（ii）[latex1]的所有MinPts最近邻[latex2]也都在$1/\left( {1 + \varepsilon }\right)  \leq  \operatorname{LOF}\left( \mathrm{p}\right)  \leq  \left( {1 + \varepsilon }\right)$中，则有$1/\left( {1 + \varepsilon }\right)  \leq  \operatorname{LOF}\left( \mathrm{p}\right)  \leq  \left( {1 + \varepsilon }\right)$ 。

Proof (Sketch): For all MinPts-nearest neighbors q of p, reach- $\operatorname{dist}\left( {\mathrm{p},\mathrm{q}}\right)  \geq$ reach-dist-min. Then the local reachability density of $\mathrm{p}$ ,as per definition 6,is $\leq  1/$ reach-dist-min. On the other hand,reach-dist $\left( {\mathrm{p},\mathrm{q}}\right)  \leq$ reach-dist-max. Thus,the local reachability density of $p$ is $\geq  1/$ reach-dist-max.

证明（概要）：对于$p$的所有MinPts最近邻$\mathrm{p}$，可达距离reach - $\operatorname{dist}\left( {\mathrm{p},\mathrm{q}}\right)  \geq$ 为reach - dist - min。根据定义6，$p$的局部可达密度为$\leq  1/$ reach - dist - min。另一方面，可达距离reach - dist $\left( {\mathrm{p},\mathrm{q}}\right)  \leq$ 为reach - dist - max。因此，$\mathrm{p}$的局部可达密度为$\geq  1/$ reach - dist - max。

Let $q$ be a MinPts-nearest neighbor of $p$ . By an argument identical to the one for $p$ above,the local reachability density of $q$ is also between 1/reach-dist-max and 1/reach-dist-min.

设$q$是$p$的一个MinPts最近邻。通过与上述$p$相同的论证，$q$的局部可达密度也在1 / reach - dist - max和1 / reach - dist - min之间。

Thus, by definition 7, we have reach-dist-min/reach-dist-max $\leq  \operatorname{LOF}\left( \mathrm{p}\right)  \leq$ reach-dist-max/reach-dist-min. Hence,we establish $1/\left( {1 + \varepsilon }\right)  \leq  \operatorname{LOF}\left( \mathrm{p}\right)  \leq  \left( {1 + \varepsilon }\right)$ .

因此，根据定义7，我们有reach - dist - min / reach - dist - max $\leq  \operatorname{LOF}\left( \mathrm{p}\right)  \leq$ reach - dist - max / reach - dist - min。由此，我们证明了$1/\left( {1 + \varepsilon }\right)  \leq  \operatorname{LOF}\left( \mathrm{p}\right)  \leq  \left( {1 + \varepsilon }\right)$ 。

The interpretation of lemma 1 is as follows. Intuitively, $\mathrm{C}$ corresponds to a "cluster". Let us consider the objects p that are "deep" inside the cluster, which means that all the MinPts-nearest neighbors $q$ of $p$ are in $C$ ,and that,in turn,all the MinPts-nearest neighbors of $q$ are also in $C$ . For such deep objects $p$ ,the LOF of $p$ is bounded. If $\mathrm{C}$ is a "tight" cluster,the $\varepsilon$ value in lemma 1 can be quite small,thus forcing the LOF of $p$ to be quite close to 1 .

引理1的解释如下。直观地说，$\mathrm{C}$对应于一个“簇”。让我们考虑那些位于簇“深处”的对象$p$，这意味着$p$的所有MinPts最近邻$q$都在$\mathrm{C}$中，并且反过来，$q$的所有MinPts最近邻也都在$\mathrm{C}$中。对于这样位于深处的对象$p$，其局部离群因子（LOF）是有界的。如果$\mathrm{C}$是一个“紧凑”的簇，引理1中的$\varepsilon$值可能会非常小，从而使得$p$的局部离群因子（LOF）非常接近1。

To return to the example in figure 1, we can apply lemma 1 to conclude that the LOFs of most objects in cluster ${\mathrm{C}}_{1}$ are close to 1 .

回到图1中的例子，我们可以应用引理1得出结论：簇${\mathrm{C}}_{1}$中大多数对象的局部离群因子（LOF）接近1。

### 5.2 A General Upper and Lower Bound on LOF

### 5.2 局部离群因子（LOF）的一般上下界

Lemma 1 above shows a basic property of LOF, namely that for objects deep inside a cluster, their LOFs are close to 1, and should not be labeled as a local outlier. A few immediate questions come to mind. What about those objects that are near the periphery of the cluster? And what about those objects that are outside the cluster, such as ${\mathrm{o}}_{2}$ in figure 1? Can we get an upper and lower bound on the LOF of these objects?

上述引理1展示了局部离群因子（LOF）的一个基本性质，即对于位于簇深处的对象，其局部离群因子（LOF）接近1，不应被标记为局部离群点。我们会立刻想到几个问题。那些位于簇边缘附近的对象呢？那些位于簇外部的对象呢，如图1中的${\mathrm{o}}_{2}$？我们能否得到这些对象的局部离群因子（LOF）的上下界？

Theorem 1 below shows a general upper and lower bound on $\mathrm{{LOF}}\left( \mathrm{p}\right)$ for any object p. As such,theorem 1 generalizes lemma 1 along two dimensions. First, theorem 1 applies to any object p, and is not restricted to objects deep inside a cluster. Second, even for objects deep inside a cluster, the bound given by theorem 1 can be tighter than the bound given by lemma 1, implying that the epsilon defined in lemma 1 can be made closer to zero.. This is because in lemma 1, the values of reach-dist-min and reach-dist-max are obtained based on a larger set of reachability distances. In contrast, in theorem 1, this minimum and maximum are based on just the MinPts-nearest neighborhoods of the objects under consideration, giving rise to tighter bounds. In section 5.3, we will analyze in greater details the tightness of the bounds given in theorem 1 .

下面的定理 1 展示了对于任意对象 p 的 $\mathrm{{LOF}}\left( \mathrm{p}\right)$ 的一般上下界。因此，定理 1 在两个维度上推广了引理 1。首先，定理 1 适用于任意对象 p，并不局限于聚类内部深处的对象。其次，即使对于聚类内部深处的对象，定理 1 给出的界可能比引理 1 给出的界更紧，这意味着引理 1 中定义的 ε 可以更接近零。这是因为在引理 1 中，可达距离最小值（reach-dist-min）和可达距离最大值（reach-dist-max）的值是基于更大的可达距离集合得到的。相比之下，在定理 1 中，这个最小值和最大值仅基于所考虑对象的 MinPts 最近邻域，从而产生更紧的界。在 5.3 节中，我们将更详细地分析定理 1 中给出的界的紧度。

<!-- Media -->

<!-- figureText: MinPts = 3 1min ${\mathrm{d}}_{\min } = 4 * {\mathrm{i}}_{\max }$ -->

<img src="https://cdn.noedgeai.com/0195c8fe-047f-7311-b009-9bd967908604_4.jpg?x=134&y=261&w=629&h=517&r=0"/>

Figure 3: Illustration of theorem 1

图 3：定理 1 的图示

<!-- Media -->

Before we present theorem 1, we define the following terms. For any object $\mathrm{p}$ ,let ${\operatorname{direct}}_{\min }\left( \mathrm{p}\right)$ denote the minimum reachability distance between $\mathrm{p}$ and a MinPts-nearest neighbor of $\mathrm{p}$ ,i.e.,

在给出定理 1 之前，我们定义以下术语。对于任意对象 $\mathrm{p}$，令 ${\operatorname{direct}}_{\min }\left( \mathrm{p}\right)$ 表示 $\mathrm{p}$ 与其 MinPts 最近邻之间的最小可达距离，即

${\operatorname{direct}}_{\min }\left( \mathrm{p}\right)  = \min \{$ reach-dist $\left( {\mathrm{p},\mathrm{q}}\right)  \mid  \mathrm{q} \in  {\mathrm{N}}_{\text{MinPts }}\left( \mathrm{p}\right) \} .$

${\operatorname{direct}}_{\min }\left( \mathrm{p}\right)  = \min \{$ 可达距离 $\left( {\mathrm{p},\mathrm{q}}\right)  \mid  \mathrm{q} \in  {\mathrm{N}}_{\text{MinPts }}\left( \mathrm{p}\right) \} .$

Similarly, let direct_max(p) denote the corresponding maximum, i.e. ${\operatorname{direct}}_{\max }\left( \mathrm{p}\right)  = \max \{$ reach-dist $\left( {\mathrm{p},\mathrm{q}}\right)  \mid  \mathrm{q} \in  {\mathrm{N}}_{\text{MinPts }}\left( \mathrm{p}\right) \} .$

类似地，令 direct_max(p) 表示相应的最大值，即 ${\operatorname{direct}}_{\max }\left( \mathrm{p}\right)  = \max \{$ 可达距离 $\left( {\mathrm{p},\mathrm{q}}\right)  \mid  \mathrm{q} \in  {\mathrm{N}}_{\text{MinPts }}\left( \mathrm{p}\right) \} .$

Furthermore, to generalize these definitions to the MinPts-nearest neighbor $q$ of $p$ ,let indirec ${t}_{\min }\left( p\right)$ denote the minimum reachability distance between $\mathrm{q}$ and a MinPts-nearest neighbor of $\mathrm{q}$ ,i.e.,

此外，为了将这些定义推广到 $p$ 的 MinPts 最近邻 $q$，令 indirec ${t}_{\min }\left( p\right)$ 表示 $\mathrm{q}$ 与其 MinPts 最近邻之间的最小可达距离，即

${\text{indirect}}_{\min }\left( \mathrm{p}\right)  = \min \{$ reach-dist $\left( {\mathrm{q},\mathrm{o}}\right)  \mid  \mathrm{q} \in  {\mathrm{N}}_{\text{MinPts }}\left( \mathrm{p}\right)$ and $\mathrm{o}$ $\in  {\mathrm{N}}_{\text{MinPts }}\left( \mathrm{q}\right) \}$ .

${\text{indirect}}_{\min }\left( \mathrm{p}\right)  = \min \{$ 可达距离 $\left( {\mathrm{q},\mathrm{o}}\right)  \mid  \mathrm{q} \in  {\mathrm{N}}_{\text{MinPts }}\left( \mathrm{p}\right)$ 且 $\mathrm{o}$ $\in  {\mathrm{N}}_{\text{MinPts }}\left( \mathrm{q}\right) \}$。

Similarly,let indirect ${}_{\max }\left( \mathrm{p}\right)$ denote the corresponding maximum. In the sequel, we refer to p's MinPts-nearest neighborhood as p's direct neighborhood, and refer to q's MinPts-nearest neighbors as p's indirect neighbors,whenever $\mathrm{q}$ is a MinPts-nearest neighbor of $\mathrm{p}$ .

类似地，令 indirect ${}_{\max }\left( \mathrm{p}\right)$ 表示相应的最大值。在接下来的内容中，只要 $\mathrm{q}$ 是 $\mathrm{p}$ 的 MinPts 最近邻，我们就将 p 的 MinPts 最近邻域称为 p 的直接邻域，将 q 的 MinPts 最近邻称为 p 的间接邻域。

Figure 3 gives a simple example to illustrate these definitions. In this example,object p lies some distance away from a cluster of objects C. For ease of understanding,let MinPts $= 3$ . The ${\operatorname{direct}}_{\min }\left( \mathrm{p}\right)$ value is marked as ${\mathrm{d}}_{\min }$ in the figure; the ${\operatorname{direct}}_{\max }\left( \mathrm{p}\right)$ value is marked as ${\mathrm{d}}_{\max }$ . Because $\mathrm{p}$ is relatively far away from $\mathrm{C}$ ,the 3-distance of every object $q$ in $C$ is much smaller than the actual distance between $\mathrm{p}$ and $\mathrm{q}$ . Thus,from definition 5,the reachability distance of $\mathrm{p}$ w.r.t. $\mathrm{q}$ is given by the actual distance between $\mathrm{p}$ and $\mathrm{q}$ . Now among the 3-nearest neighbors of $\mathrm{p}$ ,we in turn find their minimum and maximum reachability distances to their 3-nearest neighbors. In the figure,the ${\operatorname{indirect}}_{\min }\left( \mathrm{p}\right)$ and ${\operatorname{indirect}}_{\max }\left( \mathrm{p}\right)$ values are marked as ${\mathrm{i}}_{\min }$ and ${\mathrm{i}}_{\max }$ respectively.

图3给出了一个简单的例子来说明这些定义。在这个例子中，对象p与一组对象C相距一定距离。为便于理解，设最小点数为$= 3$。${\operatorname{direct}}_{\min }\left( \mathrm{p}\right)$值在图中标记为${\mathrm{d}}_{\min }$；${\operatorname{direct}}_{\max }\left( \mathrm{p}\right)$值标记为${\mathrm{d}}_{\max }$。由于$\mathrm{p}$与$\mathrm{C}$相对较远，$C$中每个对象$q$的3 - 距离远小于$\mathrm{p}$与$\mathrm{q}$之间的实际距离。因此，根据定义5，$\mathrm{p}$相对于$\mathrm{q}$的可达距离由$\mathrm{p}$与$\mathrm{q}$之间的实际距离给出。现在，在$\mathrm{p}$的3 - 最近邻中，我们依次找出它们到各自3 - 最近邻的最小和最大可达距离。在图中，${\operatorname{indirect}}_{\min }\left( \mathrm{p}\right)$和${\operatorname{indirect}}_{\max }\left( \mathrm{p}\right)$值分别标记为${\mathrm{i}}_{\min }$和${\mathrm{i}}_{\max }$。

Theorem 1: Let $\mathrm{p}$ be an object from the database $\mathrm{D}$ ,and 1 ≤ MinPts ≤ | D |.

定理1：设$\mathrm{p}$是数据库$\mathrm{D}$中的一个对象，且1 ≤ 最小点数 ≤ | D |。

Then, it is the case that

那么，情况如下

$$
\frac{{\text{direct}}_{\text{min}}\left( \mathrm{p}\right) }{{\text{indirect}}_{\text{max}}\left( \mathrm{p}\right) } \leq  \operatorname{LOF}\left( \mathrm{p}\right)  \leq  \frac{{\text{direct}}_{\text{max}}\left( \mathrm{p}\right) }{{\text{indirect}}_{\text{min}}\left( \mathrm{p}\right) }
$$

Proof (Sketch): (a) $\frac{{\operatorname{direct}}_{\min }\left( \mathrm{p}\right) }{{\operatorname{indirect}}_{\max }\left( \mathrm{p}\right) } \leq  \operatorname{LOF}\left( \mathrm{p}\right)$ :

证明（概要）：(a) $\frac{{\operatorname{direct}}_{\min }\left( \mathrm{p}\right) }{{\operatorname{indirect}}_{\max }\left( \mathrm{p}\right) } \leq  \operatorname{LOF}\left( \mathrm{p}\right)$：

$$
\forall \mathrm{o} \in  {\mathrm{N}}_{\text{MinPts }}\left( \mathrm{p}\right)  : \text{ reach-dist }\left( {\mathrm{p},\mathrm{o}}\right)  \geq  {\operatorname{direct}}_{\min }\left( \mathrm{p}\right) ,
$$

by definition of direct ${}_{\min }\left( \mathrm{p}\right)$ .

根据直接${}_{\min }\left( \mathrm{p}\right)$的定义。

$$
 \Rightarrow  1/\frac{\mathop{\sum }\limits_{{\mathrm{o} \in  {\mathrm{N}}_{\text{MinPts }}\left( \mathrm{p}\right) }}\text{ reach-dist(p,o) }}{\left| {\mathrm{N}}_{\text{MinPts }}\left( \mathrm{p}\right) \right| } \leq  \frac{1}{{\operatorname{direct}}_{\min }\left( \mathrm{p}\right) }\text{,i.e. }
$$

$$
\operatorname{lrd}\left( \mathrm{p}\right)  \leq  \frac{1}{{\operatorname{direct}}_{\min }\left( \mathrm{p}\right) }
$$

$$
\forall \mathrm{q} \in  {\mathrm{N}}_{\text{MinPts }}\left( \mathrm{o}\right)  : \text{ reach-dist }\left( {\mathrm{o},\mathrm{q}}\right)  \leq  {\text{ indirect }}_{\max }\left( \mathrm{p}\right) ,
$$

by definition of indirect ${}_{\max }\left( \mathrm{p}\right)$ .

根据间接${}_{\max }\left( \mathrm{p}\right)$的定义。

$$
 \Rightarrow  1/\frac{\mathrm{q} \in  {\mathrm{N}}_{\text{MinPts }}\left( \mathrm{o}\right) }{\left| {\mathrm{N}}_{\text{MinPts }}\left( \mathrm{o}\right) \right| } \geq  \frac{1}{{\mathrm{{indirect}}}_{\max }\left( \mathrm{o}\right) }\text{,i.e.}
$$

$$
\operatorname{lrd}\left( \mathrm{o}\right)  \geq  \frac{1}{{\operatorname{indirect}}_{\max }\left( \mathrm{p}\right) }
$$

Thus, it follows that

因此，由此可得

$$
\operatorname{LOF}\left( p\right)  = \frac{1}{0}\mathop{\sum }\limits_{\left| {{N}_{\text{MinPts }}\left( p\right) }\right| }\frac{\operatorname{lrd}\left( o\right) }{\operatorname{lrd}\left( p\right) } \geq  
$$

$$
o \in  {N}_{MinPts}\frac{o \in  {N}_{MinPts}\left( p\right) }{o \in  {N}_{MinPts}\left( p\right) } = \frac{{direc}{t}_{min}\left( p\right) }{{direc}{t}_{minPts}\left( p\right) }
$$

(b) $\operatorname{LOF}\left( p\right)  \leq  \frac{{\operatorname{direct}}_{\max }\left( p\right) }{{\operatorname{indirect}}_{\min }\left( p\right) }$ : analogously.

(b) $\operatorname{LOF}\left( p\right)  \leq  \frac{{\operatorname{direct}}_{\max }\left( p\right) }{{\operatorname{indirect}}_{\min }\left( p\right) }$ ：同理。

To illustrate the theorem using the example in figure 3, suppose that ${\mathrm{d}}_{\min }$ is 4 times that of ${\mathrm{i}}_{\max }$ ,and ${\mathrm{d}}_{\max }$ is 6 times that of ${\mathrm{i}}_{\min }$ . Then by theorem 1,the LOF of $\mathrm{p}$ is between 4 and 6 . It should also be clear from theorem 1 that $\operatorname{LOF}\left( \mathrm{p}\right)$ has an easy-to-understand interpretation. It is simply a function of the reachability distances in p's direct neighborhood relative to those in p's indirect neighborhood.

为了用图3中的示例说明该定理，假设${\mathrm{d}}_{\min }$ 是${\mathrm{i}}_{\max }$ 的4倍，且${\mathrm{d}}_{\max }$ 是${\mathrm{i}}_{\min }$ 的6倍。那么根据定理1，$\mathrm{p}$ 的局部离群因子（LOF）介于4和6之间。从定理1还可以清楚地看出，$\operatorname{LOF}\left( \mathrm{p}\right)$ 有一个易于理解的解释。它仅仅是点p的直接邻域中的可达距离相对于点p的间接邻域中的可达距离的一个函数。

### 5.3 The Tightness of the Bounds

### 5.3 边界的紧密性

As discussed before, theorem 1 is a general result with the specified upper and lower bounds for LOF applicable to any object p. An immediate question comes to mind. How good or tight are these bounds? In other words,if we use ${\mathrm{{LOF}}}_{\max }$ to denote the upper bound direc ${\mathrm{t}}_{\max }/{\text{indirect}}_{\min }$ ,and use ${\mathrm{{LOF}}}_{\min }$ to denote the lower bound direct ${}_{\min }/{\text{indirect}}_{\max }$ ,how large is the spread or difference between ${\mathrm{{LOF}}}_{\max }$ and ${\mathrm{{LOF}}}_{\min }$ ? In the following we study this issue. A key part of the following analysis is to show that the spread LOF- ${\mathrm{{max}}}^{ - }{\mathrm{{LOF}}}_{\min }$ is dependent on the ratio of direct/indirect. It turns out that the spread is small under some conditions, but not so small under other conditions.

如前所述，定理1是一个通用结果，为局部离群因子（LOF）指定了适用于任何对象p的上下界。我们会立刻想到一个问题。这些边界的优劣或紧密程度如何？换句话说，如果我们用${\mathrm{{LOF}}}_{\max }$ 表示上界直接${\mathrm{t}}_{\max }/{\text{indirect}}_{\min }$ ，用${\mathrm{{LOF}}}_{\min }$ 表示下界直接${}_{\min }/{\text{indirect}}_{\max }$ ，那么${\mathrm{{LOF}}}_{\max }$ 和${\mathrm{{LOF}}}_{\min }$ 之间的差距或差值有多大？接下来我们研究这个问题。以下分析的一个关键部分是表明局部离群因子（LOF） - ${\mathrm{{max}}}^{ - }{\mathrm{{LOF}}}_{\min }$ 的差距取决于直接/间接的比率。结果表明，在某些条件下差距较小，但在其他条件下差距并非如此小。

<!-- Media -->

<!-- figureText: 20.00 ${\mathrm{{LOF}}}_{\max } :$ pct $= 1\%$ ${\mathrm{{LOF}}}_{\max } :$ pct $= 5\%$ ${\mathrm{{LOF}}}_{\min } :$ pct $= 1\%$ ${\mathrm{{LOF}}}_{\min } :$ pct $= 5\%$ ${\mathrm{{LOF}}}_{\min } :$ pct $= {10}\%$ 10 15 proportion direct/indirect outlier factor LOF 15.00 10.00 ${\mathrm{{LOF}}}_{\max } :$ pct $= {10}\%$ 5.00 0.00 5 -->

<img src="https://cdn.noedgeai.com/0195c8fe-047f-7311-b009-9bd967908604_5.jpg?x=119&y=262&w=686&h=419&r=0"/>

Figure 4: Upper and lower bound on LOF depending on direct/ indirect for different values of pct

图4：不同百分比（pct）值下，局部离群因子（LOF）的上下界取决于直接/间接

<!-- Media -->

Given ${\operatorname{direct}}_{\min }\left( \mathrm{p}\right)$ and ${\operatorname{direct}}_{\max }\left( \mathrm{p}\right)$ as defined above,we use direct(p) to denote the mean value of ${\operatorname{direct}}_{\min }\left( \mathrm{p}\right)$ and ${\operatorname{direct}}_{\max }\left( \mathrm{p}\right)$ . Similarly, we use indirect(p) to denote the mean value of indirect- ${}_{\min }\left( \mathrm{p}\right)$ and indirect ${}_{\max }\left( \mathrm{p}\right)$ . In the sequel,whenever no confusion arises, we drop the parameter p, e.g., direct as a shorthand of direct(p).

给定如上定义的${\operatorname{direct}}_{\min }\left( \mathrm{p}\right)$ 和${\operatorname{direct}}_{\max }\left( \mathrm{p}\right)$ ，我们用direct(p) 表示${\operatorname{direct}}_{\min }\left( \mathrm{p}\right)$ 和${\operatorname{direct}}_{\max }\left( \mathrm{p}\right)$ 的平均值。类似地，我们用indirect(p) 表示间接 - ${}_{\min }\left( \mathrm{p}\right)$ 和间接${}_{\max }\left( \mathrm{p}\right)$ 的平均值。在后续内容中，只要不会引起混淆，我们就省略参数p，例如，用direct作为direct(p) 的简写。

Now to make our following analysis easier to understand, we simplify our discussion by requiring that

现在，为了使我们接下来的分析更易于理解，我们通过要求以下条件来简化讨论

$\left( {{\text{direct}}_{\max } - {\text{direct}}_{\min }}\right) /$ direct $= \left( {{\text{indirect}}_{\max } - {\text{indirect}}_{\min }}\right) /{\text{indirect}}_{\text{and }}$ That is, we assume that the reachability distances in the direct and indirect neighborhoods fluctuate by the same amount. Because of this simplification, we can use a single parameter pct in the sequel to control the fluctuation. More specifically,in figure 4,pct $= \mathrm{x}\%$ corresponds to the situation where ${\operatorname{direct}}_{\max } = \operatorname{direct} * \left( {1 + \mathrm{x}\% }\right)$ ,di- ${\text{rect}}_{\text{min }} =$ direct* $\left( {1 - \mathrm{x}\% }\right)$ ,indirect ${}_{\text{max }} =$ indirect* $\left( {1 + \mathrm{x}\% }\right)$ and indi- ${\text{rect}}_{\text{min }} =$ indirect* $\left( {1 - \mathrm{x}\% }\right)$ . Figure 4 shows the situations when pct is set to $1\% ,5\%$ and ${10}\%$ . The spread between ${\mathrm{{LOF}}}_{\max }$ and ${\mathrm{{LOF}}}_{\min }$ increases as pct increases.

$\left( {{\text{direct}}_{\max } - {\text{direct}}_{\min }}\right) /$ 直接 $= \left( {{\text{indirect}}_{\max } - {\text{indirect}}_{\min }}\right) /{\text{indirect}}_{\text{and }}$ 也就是说，我们假设直接邻域和间接邻域中的可达距离以相同的幅度波动。由于这种简化，我们在后续可以使用单个参数 pct 来控制波动。更具体地说，在图 4 中，pct $= \mathrm{x}\%$ 对应于 ${\operatorname{direct}}_{\max } = \operatorname{direct} * \left( {1 + \mathrm{x}\% }\right)$ 、di - ${\text{rect}}_{\text{min }} =$ 直接* $\left( {1 - \mathrm{x}\% }\right)$ 、间接 ${}_{\text{max }} =$ 间接* $\left( {1 + \mathrm{x}\% }\right)$ 以及 indi - ${\text{rect}}_{\text{min }} =$ 间接* $\left( {1 - \mathrm{x}\% }\right)$ 的情况。图 4 展示了 pct 设置为 $1\% ,5\%$ 和 ${10}\%$ 时的情况。${\mathrm{{LOF}}}_{\max }$ 和 ${\mathrm{{LOF}}}_{\min }$ 之间的差距随着 pct 的增加而增大。

More importantly, figure 4 shows that, for a fixed percentage pct $= x\%$ ,the spread between ${\mathrm{{LOF}}}_{\max }$ and ${\mathrm{{LOF}}}_{\min }$ grows linearly with respect to the ratio direct/indirect. This means that the relative span $\left( {{\mathrm{{LOF}}}_{\max } - {\mathrm{{LOF}}}_{\min }}\right) /$ (direct/indirect) is constant. Stated differently, the relative fluctuation of the LOF depends only on the ratios of the underlying reachability distances and not on their absolute values. This highlights the spirit of local outliers.

更重要的是，图 4 表明，对于固定的百分比 pct $= x\%$ ，${\mathrm{{LOF}}}_{\max }$ 和 ${\mathrm{{LOF}}}_{\min }$ 之间的差距相对于直接/间接比率呈线性增长。这意味着相对跨度 $\left( {{\mathrm{{LOF}}}_{\max } - {\mathrm{{LOF}}}_{\min }}\right) /$ （直接/间接）是恒定的。换句话说，局部离群因子（LOF）的相对波动仅取决于底层可达距离的比率，而不取决于它们的绝对值。这凸显了局部离群点的本质。

To be more precise, in fact, the whole situation is best captured in the 3-dimensional space where the three dimensions are: $\left( {\mathrm{{LOF}}}_{\max }\right.$ - ${\mathrm{{LOF}}}_{\min }$ ),(direct/indirect),and pct. Figure 4 then represents a series of 2-D projections on the first two dimensions. But figure 4 does not show the strength of the dependency between the relative fluctuation of the LOF and the relative fluctuation of pct. For this purpose, figure 5 is useful. The y-axis of the figure shows the ratio between the two dimensions $\left( {{\mathrm{{LOF}}}_{\max } - {\mathrm{{LOF}}}_{\min }}\right)$ and (direct/indirect) in the 3-dimensional space mentioned above,and the x-axis corresponds to the other dimension pct. To understand the shape of the curve in figure 5 , we have to take a closer look at the ratio (LOF-max $- {\mathrm{{LOF}}}_{\min }$ )/(direct/indirect):

更准确地说，实际上，整个情况在三维空间中能得到最好的体现，这三个维度分别是：$\left( {\mathrm{{LOF}}}_{\max }\right.$ - ${\mathrm{{LOF}}}_{\min }$ ）、（直接/间接）和 pct。图 4 则代表了在前两个维度上的一系列二维投影。但图 4 并未展示局部离群因子（LOF）的相对波动与 pct 的相对波动之间的依赖强度。为此，图 5 很有用。该图的 y 轴表示上述三维空间中两个维度 $\left( {{\mathrm{{LOF}}}_{\max } - {\mathrm{{LOF}}}_{\min }}\right)$ 和（直接/间接）之间的比率，x 轴对应另一个维度 pct。为了理解图 5 中曲线的形状，我们必须更仔细地研究比率（LOF - 最大值 $- {\mathrm{{LOF}}}_{\min }$ ）/（直接/间接）：

$$
\frac{{\mathrm{{LOF}}}_{\max } - {\mathrm{{LOF}}}_{\min }}{\frac{\text{ direct }}{\text{ indirect }}} = \frac{\text{ indirect }}{\text{ direct }}.
$$

$$
\left( {\frac{\text{direct} + \frac{\text{direct} \cdot  \text{pct}}{100}}{\text{indirect} - \frac{\text{indirect} \cdot  \text{pct}}{100}} - \frac{\text{direct} - \frac{\text{direct} \cdot  \text{pct}}{100}}{\text{indirect} + \frac{\text{indirect} \cdot  \text{pct}}{100}}}\right)  = 
$$

$$
 = \left( {\frac{1 + \frac{\text{pct}}{100}}{1 - \frac{\text{pct}}{100}} - \frac{1 - \frac{\text{pct}}{100}}{1 + \frac{\text{pct}}{100}}}\right)  = \frac{4 \times  \frac{\text{pct}}{100}}{1 - {\left( \frac{\text{pct}}{100}\right) }^{2}}
$$

Figure 5 shows that $\left( {{\mathrm{{LOF}}}_{\max } - {\mathrm{{LOF}}}_{\min }}\right) /\left( \text{direct/indirect) is only de-}\right)$ pendent on the percentage value pct. Its value approaches infinity if pct approaches 100 , but it is very small for reasonable values of pct. This also verifies that the relative fluctuation of the LOF is constant for a fixed percentage pct, as we have seen in figure 4.

图 5 表明 $\left( {{\mathrm{{LOF}}}_{\max } - {\mathrm{{LOF}}}_{\min }}\right) /\left( \text{direct/indirect) is only de-}\right)$ 取决于百分比值 pct。如果 pct 接近 100，其值趋近于无穷大，但对于合理的 pct 值，它非常小。这也验证了如我们在图 4 中所见，对于固定的百分比 pct，局部离群因子（LOF）的相对波动是恒定的。

To summarize, if the fluctuation of the average reachability distances in the direct and indirect neighborhoods is small (i.e., pct is low), theorem 1 estimates the LOF very well, as the minimum and maximum LOF bounds are close to each other. There are two important cases for which this is true.

综上所述，如果直接邻域和间接邻域中平均可达距离的波动较小（即 pct 较低），定理 1 能很好地估计局部离群因子（LOF），因为最小和最大 LOF 边界彼此接近。有两种重要情况符合这一点。

- The percentage pct is very low for an object $\mathrm{p}$ ,if the fluctuation of the reachability distances is rather homogeneous, i.e., if the MinPts-nearest neighbors of $p$ belong to the same cluster. In this case,the values direct ${}_{\min },{\text{direct}}_{\max }$ ,indirect ${}_{\min }$ and indirect ${}_{\max }$ are almost identical, resulting in the LOF being close to 1 . This is consistent with the result established in lemma 1.

- 对于对象 $\mathrm{p}$，如果可达距离的波动相当均匀，即如果 $p$ 的 MinPts 最近邻属于同一个聚类，则百分比 pct 非常低。在这种情况下，直接 ${}_{\min },{\text{direct}}_{\max }$、间接 ${}_{\min }$ 和间接 ${}_{\max }$ 的值几乎相同，导致局部离群因子（LOF）接近 1。这与引理 1 中确立的结果一致。

- The argument above can be generalized to an object p which is not located deep inside a cluster, but whose MinPts-nearest neighbors all belong to the same cluster (as depicted in figure 3). In this case, even though LOF may not be close to 1, the bounds on LOF as predicted by theorem 1 are tight.

- 上述论证可以推广到对象 p，该对象并非位于聚类的深处，但其 MinPts 最近邻都属于同一个聚类（如图 3 所示）。在这种情况下，即使局部离群因子（LOF）可能不接近 1，定理 1 所预测的局部离群因子（LOF）的边界仍然是紧密的。

### 5.4 Bounds for Objects whose Direct Neighbor- hoods Overlap Multiple Clusters

### 5.4 直接邻域与多个聚类重叠的对象的边界

So far we have analyzed the tightness of the bounds given in theorem 1, and have given two conditions under which the bounds are tight. An immediate question that comes to mind is: under what condition are the bounds not tight? Based on figure 5 , if the MinPts-nearest neighbors of an object $\mathrm{p}$ belong to different clusters having different densities, the value for pct may be very large. Then based on figure 5,the spread between ${\mathrm{{LOF}}}_{\max }$ and ${\mathrm{{LOF}}}_{\min }$ value can be large. In this case, the bounds given in theorem 1 do not work well.

到目前为止，我们已经分析了定理 1 中给出的边界的紧密性，并给出了边界紧密的两个条件。我们脑海中立即浮现的一个问题是：在什么条件下边界不紧密？根据图 5，如果对象 $\mathrm{p}$ 的 MinPts 最近邻属于具有不同密度的不同聚类，则 pct 的值可能非常大。然后根据图 5，${\mathrm{{LOF}}}_{\max }$ 和 ${\mathrm{{LOF}}}_{\min }$ 值之间的差距可能很大。在这种情况下，定理 1 中给出的边界效果不佳。

<!-- Media -->

<!-- figureText: 100.00 40 50 60 70 80 90 100 percentage of fluctuation pct ${\mathrm{{LOF}}}_{\max } - {\mathrm{{LOF}}}_{\min }$ direct/indirect 50.00 0.00 0 20 30 -->

<img src="https://cdn.noedgeai.com/0195c8fe-047f-7311-b009-9bd967908604_5.jpg?x=849&y=1602&w=680&h=416&r=0"/>

Figure 5: Relative span for LOF depending on percentage of fluctuation for $\mathrm{d}$ and $\mathrm{w}$

图 5：局部离群因子（LOF）的相对跨度取决于 $\mathrm{d}$ 和 $\mathrm{w}$ 的波动百分比

<!-- figureText: MinPts = 6 ${\mathrm{{d2}}}_{\text{min }}$ ${\mathrm{{d1}}}_{\mathrm{{min}}}$ max -->

<img src="https://cdn.noedgeai.com/0195c8fe-047f-7311-b009-9bd967908604_6.jpg?x=133&y=261&w=635&h=394&r=0"/>

Figure 6: Illustration of theorem 2

图 6：定理 2 的图示

<!-- Media -->

As an example, let us consider the situation shown in figure 1 again. For object ${\mathrm{o}}_{2}$ ,because all its MinPts-nearest neighbors come from the same cluster ${\mathrm{C}}_{2}$ ,the bounds given by theorem 1 on the LOF of ${\mathrm{o}}_{2}$ is expected to be tight. In contrast,the MinPts-nearest neighbors of ${\mathrm{o}}_{1}$ come from both clusters ${\mathrm{C}}_{1}$ and ${\mathrm{C}}_{2}$ . In this case,the given bounds on the LOF of ${\mathrm{o}}_{1}$ may not be as good.

作为一个例子，让我们再次考虑图 1 所示的情况。对于对象 ${\mathrm{o}}_{2}$，因为其所有 MinPts 最近邻都来自同一个聚类 ${\mathrm{C}}_{2}$，所以定理 1 给出的关于 ${\mathrm{o}}_{2}$ 的局部离群因子（LOF）的边界预计是紧密的。相比之下，${\mathrm{o}}_{1}$ 的 MinPts 最近邻来自聚类 ${\mathrm{C}}_{1}$ 和 ${\mathrm{C}}_{2}$。在这种情况下，给定的关于 ${\mathrm{o}}_{1}$ 的局部离群因子（LOF）的边界可能没有那么好。

Theorem 2 below intends to give better bounds on the LOF of object p when p's MinPts-nearest neighborhood overlaps with more than one cluster. The intuitive meaning of theorem 2 is that, when we partition the MinPts-nearest neighbors of p into several groups, each group contributes proportionally to the LOF of p.

下面的定理 2 旨在当对象 p 的 MinPts 最近邻域与多个聚类重叠时，给出关于对象 p 的局部离群因子（LOF）的更好边界。定理 2 的直观含义是，当我们将 p 的 MinPts 最近邻划分为几个组时，每个组对 p 的局部离群因子（LOF）按比例贡献。

An example is shown in figure 6 for MinPts=6 . In this case, 3 of object p’s 6-nearest neighbors come from cluster ${\mathrm{C}}_{1}$ ,and the other 3 come from cluster ${\mathrm{C}}_{2}$ . Then according to theorem 2, ${\mathrm{{LOF}}}_{\min }$ is given by $\left( {{0.5} * {\mathrm{{d1}}}_{\min } + {0.5} * {\mathrm{{d2}}}_{\min }}\right) /\left( {{0.5}/{\mathrm{{i1}}}_{\max } + {0.5}/{\mathrm{{i2}}}_{\max }}\right)$ ,where ${\mathrm{{d1}}}_{\min }$ and ${\mathrm{{d2}}}_{\text{min }}$ give the minimum reachability distances between $\mathrm{p}$ and the 6-nearest neighbors of $\mathrm{p}$ in ${\mathrm{C}}_{1}$ and ${\mathrm{C}}_{2}$ respectively,and ${\mathrm{{i1}}}_{\max }$ and $\mathrm{i}{2}_{\max }$ give the maximum reachability distances between $\mathrm{q}$ and q’s 6-nearest neighbors,where $\mathrm{q}$ is a 6-nearest neighbor of $\mathrm{p}$ from ${\mathrm{C}}_{1}$ and ${\mathrm{C}}_{2}$ respectively. For simplicity,figure 6 does not show the case for the upper bound ${\mathrm{{LOF}}}_{\max }$ .

图 6 展示了一个 MinPts = 6 的例子。在这种情况下，对象 p 的 6 个最近邻中有 3 个来自聚类 ${\mathrm{C}}_{1}$，另外 3 个来自聚类 ${\mathrm{C}}_{2}$。然后根据定理 2，${\mathrm{{LOF}}}_{\min }$ 由 $\left( {{0.5} * {\mathrm{{d1}}}_{\min } + {0.5} * {\mathrm{{d2}}}_{\min }}\right) /\left( {{0.5}/{\mathrm{{i1}}}_{\max } + {0.5}/{\mathrm{{i2}}}_{\max }}\right)$ 给出，其中 ${\mathrm{{d1}}}_{\min }$ 和 ${\mathrm{{d2}}}_{\text{min }}$ 分别给出 $\mathrm{p}$ 与 ${\mathrm{C}}_{1}$ 和 ${\mathrm{C}}_{2}$ 中 $\mathrm{p}$ 的 6 个最近邻之间的最小可达距离，${\mathrm{{i1}}}_{\max }$ 和 $\mathrm{i}{2}_{\max }$ 给出 $\mathrm{q}$ 与 q 的 6 个最近邻之间的最大可达距离，其中 $\mathrm{q}$ 分别是来自 ${\mathrm{C}}_{1}$ 和 ${\mathrm{C}}_{2}$ 的 $\mathrm{p}$ 的 6 个最近邻之一。为简单起见，图 6 未显示上界 ${\mathrm{{LOF}}}_{\max }$ 的情况。

Theorem 2: Let $\mathrm{p}$ be an object from the database $\mathrm{D}$ , $1 \leq$ MinPts $\leq  \left| D\right|$ ,and ${C}_{1},{C}_{2},\ldots ,{C}_{n}$ be a partition of ${N}_{\text{MinPts }}\left( p\right)$ , i.e. ${\mathrm{N}}_{\text{MinPts }}\left( \mathrm{p}\right)  = {\mathrm{C}}_{1} \cup  {\mathrm{C}}_{2} \cup  \ldots  \cup  {\mathrm{C}}_{\mathrm{n}} \cup  \{ \mathrm{p}\}$ with ${\mathrm{C}}_{\mathrm{i}} \cap  {\mathrm{C}}_{\mathrm{j}} = \varnothing$ , ${C}_{i} \neq  \varnothing$ for $1 \leq  i,j \leq  n,i \neq  j$ .

定理2：设$\mathrm{p}$为数据库$\mathrm{D}$中的一个对象，$1 \leq$为最小点数（MinPts）$\leq  \left| D\right|$，且${C}_{1},{C}_{2},\ldots ,{C}_{n}$为${N}_{\text{MinPts }}\left( p\right)$的一个划分，即${\mathrm{N}}_{\text{MinPts }}\left( \mathrm{p}\right)  = {\mathrm{C}}_{1} \cup  {\mathrm{C}}_{2} \cup  \ldots  \cup  {\mathrm{C}}_{\mathrm{n}} \cup  \{ \mathrm{p}\}$，其中${\mathrm{C}}_{\mathrm{i}} \cap  {\mathrm{C}}_{\mathrm{j}} = \varnothing$，对于$1 \leq  i,j \leq  n,i \neq  j$有${C}_{i} \neq  \varnothing$。

Furthermore,let ${\xi }_{\mathrm{i}} = \left| {\mathrm{C}}_{\mathrm{i}}\right| /\left| {{\mathrm{N}}_{\text{MinPts }}\left( \mathrm{p}\right) }\right|$ be the percentage of objects in p’s neighborhood,which are also in ${\mathrm{C}}_{\mathrm{i}}$ . Let the notions ${\text{direct}}_{\min }^{\mathrm{i}}\left( \mathrm{p}\right) ,\;{\text{direct}}_{\max }^{\mathrm{i}}\left( \mathrm{p}\right) ,\;$ indirect ${}_{\min }^{\mathrm{i}}\left( \mathrm{p}\right)$ ,and indirect ${}^{\mathrm{i}}\max \left( \mathrm{p}\right)$ be defined analogously to ${\operatorname{direct}}_{\min }\left( \mathrm{p}\right)$ ,direct- ${}_{\max }\left( \mathrm{p}\right)$ ,indirect ${}_{\min }\left( \mathrm{p}\right)$ ,and indirect ${}_{\max }\left( \mathrm{p}\right)$ but restricted to the set ${\mathrm{C}}_{\mathrm{i}}$ (e.g.,direct ${}_{\min }^{\mathrm{i}}\left( \mathrm{p}\right)$ denotes the minimum reachability distance between $\mathrm{p}$ and a MinPts-nearest neighbor of $\mathrm{p}$ in the set ${\mathrm{C}}_{\mathrm{i}}$ ).

此外，设${\xi }_{\mathrm{i}} = \left| {\mathrm{C}}_{\mathrm{i}}\right| /\left| {{\mathrm{N}}_{\text{MinPts }}\left( \mathrm{p}\right) }\right|$为p的邻域中同时也在${\mathrm{C}}_{\mathrm{i}}$中的对象的百分比。设概念${\text{direct}}_{\min }^{\mathrm{i}}\left( \mathrm{p}\right) ,\;{\text{direct}}_{\max }^{\mathrm{i}}\left( \mathrm{p}\right) ,\;$间接${}_{\min }^{\mathrm{i}}\left( \mathrm{p}\right)$和间接${}^{\mathrm{i}}\max \left( \mathrm{p}\right)$的定义与${\operatorname{direct}}_{\min }\left( \mathrm{p}\right)$、直接 - ${}_{\max }\left( \mathrm{p}\right)$、间接${}_{\min }\left( \mathrm{p}\right)$和间接${}_{\max }\left( \mathrm{p}\right)$类似，但限制在集合${\mathrm{C}}_{\mathrm{i}}$中（例如，直接${}_{\min }^{\mathrm{i}}\left( \mathrm{p}\right)$表示$\mathrm{p}$与集合${\mathrm{C}}_{\mathrm{i}}$中$\mathrm{p}$的最小点数（MinPts）最近邻之间的最小可达距离）。

Then, it holds that (a)

那么，有如下结论成立：(a)

$$
\operatorname{LOF}\left( p\right)  \geq  \left( {\mathop{\sum }\limits_{{i = 1}}^{n}{\xi }_{i} \cdot  \operatorname{direct}{\min }_{i}\left( p\right) }\right)  \cdot  \left( {\mathop{\sum }\limits_{{i = 1}}^{n}\frac{{\xi }_{i}}{\operatorname{indirect}{\max }_{i}^{i}\left( p\right) }}\right) 
$$

and (b)

以及 (b)

$$
\operatorname{LOF}\left( p\right)  \leq  \left( {\mathop{\sum }\limits_{{i = 1}}^{n}{\xi }_{i} \cdot  {\operatorname{direct}}_{\max }^{i}\left( p\right) }\right)  \cdot  \left( {\mathop{\sum }\limits_{{i = 1}}^{n}\frac{{\xi }_{i}}{{\operatorname{indirect}}_{\min }^{i}\left( p\right) }}\right) 
$$

We give a proof sketch of theorem 2 in the appendix. Theorem 2 generalizes theorem 1 in taking into consideration the ratios of the MinPts-nearest neighbors coming from multiple clusters. As such, there is the following corollary.

我们在附录中给出定理2的证明概要。定理2在考虑来自多个簇的最小点数（MinPts）最近邻的比例方面对定理1进行了推广。因此，有如下推论。

Corollary 1: If the number of partitions in theorem 2 is 1, then ${\mathrm{{LOF}}}_{\min }$ and ${\mathrm{{LOF}}}_{\max }$ given in theorem 2 are exactly the same corresponding bounds given in theorem 1 .

推论1：如果定理2中的划分数为1，那么定理2中给出的${\mathrm{{LOF}}}_{\min }$和${\mathrm{{LOF}}}_{\max }$与定理1中给出的相应边界完全相同。

### 6.THE IMPACT OF THE PARAMETER MINPTS

### 6. 参数最小点数（MinPts）的影响

In the previous section, we have analyzed the formal properties of LOF. For objects deep inside a cluster, we have shown that the LOF is approximately equal to 1 . For other objects, we have established two sets of upper and lower bounds on the LOF, depending on whether the MinPts-nearest neighbors come from one or more clusters. It is important to note that all the previous results are based on a given MinPts value. In this section, we discuss how the LOF value is influenced by the choice of the MinPts value, and how to determine the right MinPts values for the LOF computation.

在上一节中，我们分析了局部离群因子（LOF）的形式性质。对于处于簇内部深处的对象，我们已经证明局部离群因子（LOF）近似等于1。对于其他对象，我们根据最小点数（MinPts）最近邻是来自一个簇还是多个簇，建立了两组局部离群因子（LOF）的上下界。需要注意的是，之前的所有结果都是基于给定的最小点数（MinPts）值。在本节中，我们讨论局部离群因子（LOF）值如何受到最小点数（MinPts）值选择的影响，以及如何为局部离群因子（LOF）计算确定合适的最小点数（MinPts）值。

### 6.1 How LOF Varies according to Changing MinPts Values

### 6.1 局部离群因子（LOF）如何随最小点数（MinPts）值的变化而变化

Given the analytic results established in the previous section, several interesting questions come to mind. How does the LOF value change when the MinPts value is adjusted? Given an increasing sequence of MinPts values, is there a corresponding monotonic sequence of changes to LOF? That is, does LOF decrease or increase monotonically?

根据上一节建立的分析结果，我们会想到几个有趣的问题。当调整最小点数（MinPts）值时，局部离群因子（LOF）值会如何变化？给定一个递增的最小点数（MinPts）值序列，局部离群因子（LOF）是否有相应的单调变化序列？也就是说，局部离群因子（LOF）是单调递减还是单调递增？

Unfortunately, the reality is that LOF neither decreases nor increases monotonically. Figure 7 shows a simple scenario where all the objects are distributed following a Gaussian distribution. For each MinPts value between 2 and 50 , the minimum, maximum and mean LOF values, as well as the standard deviation, are shown.

不幸的是，现实情况是局部离群因子（LOF）既不会单调递减，也不会单调递增。图7展示了一个简单的场景，其中所有对象都遵循高斯分布。对于2到50之间的每个最小邻域点数（MinPts）值，都展示了最小、最大和平均局部离群因子（LOF）值以及标准差。

Let us consider the maximum LOF as an example. Initially, when the MinPts value is set to 2 , this reduces to using the actual inter-object distance $\mathrm{d}\left( {\mathrm{p},\mathrm{o}}\right)$ in definition 5 . By increasing the MinPts value, the statistical fluctuations in reachability distances and in LOF are weakened. Thus, there is an initial drop on the maximum LOF value. However, as the MinPts value continues to increase, the maximum LOF value goes up and down, and eventually stabilizes to some value.

让我们以最大局部离群因子（LOF）为例。最初，当最小邻域点数（MinPts）值设置为2时，这相当于在定义5中使用实际的对象间距离$\mathrm{d}\left( {\mathrm{p},\mathrm{o}}\right)$。通过增加最小邻域点数（MinPts）值，可达距离和局部离群因子（LOF）的统计波动会减弱。因此，最大局部离群因子（LOF）值会有一个初始下降。然而，随着最小邻域点数（MinPts）值继续增加，最大局部离群因子（LOF）值会上下波动，最终稳定在某个值。

If the LOF value changes non-monotonically even for such a pure distribution like the Gaussian distribution, the LOF value changes more wildly for more complex situations. Figure 8 shows a two-dimensional dataset containing three clusters,where ${\mathrm{S}}_{1}$ consists of 10 objects, ${\mathrm{S}}_{2}$ of 35 objects and ${\mathrm{S}}_{3}$ of 500 objects. On the right side are representative plots for one object from each of these clusters. The plots show the LOF over MinPts for the range from 10 to 50 . While the LOF of an object in ${\mathrm{S}}_{3}$ is very stable around 1,the LOFs of the objects in ${\mathrm{S}}_{1}$ and ${\mathrm{S}}_{3}$ change more wildly.

如果即使对于像高斯分布这样纯粹的分布，局部离群因子（LOF）值也会非单调变化，那么在更复杂的情况下，局部离群因子（LOF）值的变化会更加剧烈。图8展示了一个包含三个簇的二维数据集，其中${\mathrm{S}}_{1}$由10个对象组成，${\mathrm{S}}_{2}$由35个对象组成，${\mathrm{S}}_{3}$由500个对象组成。右侧是这些簇中每个簇的一个对象的代表性图表。图表展示了最小邻域点数（MinPts）在10到50范围内的局部离群因子（LOF）。虽然${\mathrm{S}}_{3}$中对象的局部离群因子（LOF）在1附近非常稳定，但${\mathrm{S}}_{1}$和${\mathrm{S}}_{3}$中对象的局部离群因子（LOF）变化更为剧烈。

<!-- Media -->

<!-- figureText: 2.5 Gaussian Distribution max mean with std.dev - min 6 11 16 21 26 31 36 41 46 51 MinPts outlier factor LOF 1.5 0.5 -->

<img src="https://cdn.noedgeai.com/0195c8fe-047f-7311-b009-9bd967908604_7.jpg?x=140&y=260&w=1361&h=518&r=0"/>

Figure 7: Fluctuation of the outlier-factors within a Gaussian cluster

图7：高斯簇内离群因子的波动

<!-- Media -->

### 6.2 Determining a Range of MinPts Values

### 6.2 确定最小邻域点数（MinPts）值的范围

Because the LOF value can go up and down, we propose as a heuristic that we use a range of MinPts values. In the following, we provide guidelines as to how this range can be picked. We use MinPtsLB and MinPtsUB to denote the "lower bound" and the "upper bound" of the range.

由于局部离群因子（LOF）值可能会上下波动，我们提出一个启发式方法，即使用一个最小邻域点数（MinPts）值的范围。下面，我们将提供关于如何选择这个范围的指导原则。我们使用最小邻域点数下限（MinPtsLB）和最小邻域点数上限（MinPtsUB）来表示该范围的“下限”和“上限”。

Let us first determine a reasonable value of MinPtsLB. Clearly, MinPtsLB can be as small as 2. However, as explained above and before definition 5 , it is wise to remove unwanted statistical fluctuations due to MinPts being too small. As an example, for the Gaussian distribution shown in figure 7, the standard deviation of LOF only stabilizes when MinPtsLB is at least 10 . As another extreme example, suppose we turn the Gaussian distribution in figure 7 to a uniform distribution. It turns out that for MinPts less than 10, there can be objects whose LOF are significant greater than 1 . This is counter-intuitive because in a uniform distribution, no object should be labeled as outlying. Thus, the first guideline we provide for picking MinPtsLB is that it should be at least 10 to remove unwanted statistical fluctuations.

让我们首先确定最小邻域点数下限（MinPtsLB）的合理值。显然，最小邻域点数下限（MinPtsLB）可以小到2。然而，正如上面和定义5之前所解释的，消除由于最小邻域点数（MinPts）过小而产生的不必要的统计波动是明智的。例如，对于图7所示的高斯分布，只有当最小邻域点数下限（MinPtsLB）至少为10时，局部离群因子（LOF）的标准差才会稳定。作为另一个极端的例子，假设我们将图7中的高斯分布转变为均匀分布。结果表明，当最小邻域点数（MinPts）小于10时，可能会有局部离群因子（LOF）显著大于1的对象。这是违反直觉的，因为在均匀分布中，不应将任何对象标记为离群点。因此，我们提供的选择最小邻域点数下限（MinPtsLB）的第一条指导原则是，它至少应该为10，以消除不必要的统计波动。

The second guideline we provide for picking MinPtsLB is based on a more subtle observation. Consider a simple situation of one object $\mathrm{p}$ and a set/cluster $\mathrm{C}$ of objects. If $\mathrm{C}$ contains fewer than MinPtsLB objects, then the set of MinPts-nearest neighbors of each object in $\mathrm{C}$ will include $\mathrm{p}$ ,and vice versa. Thus,by applying theorem 1,the LOF of $\mathrm{p}$ and all the objects in $\mathrm{C}$ will be quite similar,thus making $\mathrm{p}$ indistinguishable from the objects in $\mathrm{C}$ .

我们提供的选择最小邻域点数下限（MinPtsLB）的第二条指导原则基于一个更微妙的观察。考虑一个对象$\mathrm{p}$和一组/簇对象$\mathrm{C}$的简单情况。如果$\mathrm{C}$包含的对象少于最小邻域点数下限（MinPtsLB），那么$\mathrm{C}$中每个对象的最小邻域点数（MinPts）最近邻集合将包括$\mathrm{p}$，反之亦然。因此，通过应用定理1，$\mathrm{p}$和$\mathrm{C}$中所有对象的局部离群因子（LOF）将非常相似，从而使$\mathrm{p}$与$\mathrm{C}$中的对象难以区分。

If, on the other hand, C contains more than MinPtsLB objects, the MinPts-nearest neighborhoods of the objects deep in $\mathrm{C}$ will not contain $\mathrm{p}$ ,but some objects of $\mathrm{C}$ will be included in ${\mathrm{p}}^{\prime }$ s neighborhood. Thus,depending on the distance between $\mathrm{p}$ and $\mathrm{C}$ and the density of $\mathrm{C}$ ,the LOF of $\mathrm{p}$ can be quite different from that of an object in C. The key observation here is that MinPtsLB can be regarded as the minimum number of objects a "cluster" (like C above) has to contain,so that other objects (like p above) can be local outliers relative to this cluster. This value could be application-dependent. For most of the datasets we experimented with, picking 10 to 20 appears to work well in general.

另一方面，如果C包含的对象数量超过MinPtsLB，那么$\mathrm{C}$深处对象的MinPts最近邻域将不包含$\mathrm{p}$，但$\mathrm{C}$的一些对象将被包含在${\mathrm{p}}^{\prime }$的邻域中。因此，根据$\mathrm{p}$和$\mathrm{C}$之间的距离以及$\mathrm{C}$的密度，$\mathrm{p}$的局部离群因子（LOF）可能与C中对象的LOF有很大不同。这里的关键观察结果是，MinPtsLB可以被视为一个“簇”（如上面的C）必须包含的最小对象数量，这样其他对象（如上面的p）相对于这个簇就可以是局部离群点。这个值可能取决于具体应用。对于我们实验过的大多数数据集，一般选择10到20似乎效果不错。

Next, we turn to the selection of a reasonable value of MinPtsUB, the upper bound value of the range of MinPts values. Like the lower bound MinPtsLB, the upper bound also has an associated meaning. Let $\mathrm{C}$ be a set/cluster of "close by" objects. Then MinPtsUB can be regarded as the maximum cardinality of $\mathrm{C}$ for all objects in $\mathrm{C}$ to potentially be local outliers. By "close by" we mean, that the direct- ${}_{\min }$ ,direct ${}_{\max }$ ,indirect ${}_{\min }$ and indirect ${}_{\max }$ values are all very similar. In this case, for MinPts values exceeding MinPtsUB, theorem 1 requires that the LOF of all objects in $\mathrm{C}$ be close to 1 . Hence,the guideline we provide for picking MinPtsUB is the maximum number of "close by" objects that can potentially be local outliers.

接下来，我们讨论MinPtsUB（MinPts值范围的上限值）合理取值的选择。与下限MinPtsLB类似，上限也有相关的含义。设$\mathrm{C}$是一组“相邻”对象的集合/簇。那么MinPtsUB可以被视为对于$\mathrm{C}$中所有对象都有可能成为局部离群点时$\mathrm{C}$的最大基数。这里的“相邻”是指，直接${}_{\min }$、直接${}_{\max }$、间接${}_{\min }$和间接${}_{\max }$的值都非常相似。在这种情况下，对于超过MinPtsUB的MinPts值，定理1要求$\mathrm{C}$中所有对象的局部离群因子（LOF）接近1。因此，我们提供的选择MinPtsUB的准则是可能成为局部离群点的“相邻”对象的最大数量。

As an example, let us consider the situation shown in figure 8 again. Recall that ${\mathrm{S}}_{1}$ consists of 10 objects, ${\mathrm{S}}_{2}$ of 35 objects and ${\mathrm{S}}_{3}$ of 500 objects. From the plots,it is clear that the objects in ${\mathrm{S}}_{3}$ are never outliers, always having their LOF values close to 1 . In contrast,the objects in ${\mathrm{S}}_{1}$ are strong outliers for MinPts values between 10 and 35 . The objects in ${\mathrm{S}}_{2}$ are outliers starting at MinPts = 45 . The reason for the last two effects is that,beginning at MinPts $= {36}$ , the MinPts-nearest neighborhoods of the objects in ${\mathrm{S}}_{2}$ start to include some object(s) from ${\mathrm{S}}_{1}$ . From there on,the objects in ${\mathrm{S}}_{1}$ and ${\mathrm{S}}_{2}$ exhibit roughly the same behavior. Now at MinPts $= {45}$ ,the members of this "combined" set of objects ${\mathrm{S}}_{1}$ and ${\mathrm{S}}_{2}$ start to include object(s) from ${\mathrm{S}}_{3}$ in their neighborhoods,and thus starting to become outliers relative to ${\mathrm{S}}_{3}$ . Depending on the application domain, we may want to consider a group of 35 objects (like ${\mathrm{S}}_{2}$ ) a cluster or a bunch of "close by" local outliers. To facilitate this, we can choose a MinPtsUB value accordingly, that is either smaller than 35 or larger than 35. A similar argument can be made for MinPtsLB with respect to the minimum number of objects relative to which other objects can be considered local outliers.

作为一个例子，让我们再次考虑图8所示的情况。回顾一下，${\mathrm{S}}_{1}$由10个对象组成，${\mathrm{S}}_{2}$由35个对象组成，${\mathrm{S}}_{3}$由500个对象组成。从图中可以明显看出，${\mathrm{S}}_{3}$中的对象永远不是离群点，它们的局部离群因子（LOF）值总是接近1。相反，对于MinPts值在10到35之间，${\mathrm{S}}_{1}$中的对象是强离群点。从MinPts = 45开始，${\mathrm{S}}_{2}$中的对象成为离群点。后两种情况的原因是，从MinPts $= {36}$开始，${\mathrm{S}}_{2}$中对象的MinPts最近邻域开始包含${\mathrm{S}}_{1}$中的一些对象。从那时起，${\mathrm{S}}_{1}$和${\mathrm{S}}_{2}$中的对象表现出大致相同的行为。现在，在MinPts $= {45}$时，${\mathrm{S}}_{1}$和${\mathrm{S}}_{2}$这个“组合”对象集的成员开始在其邻域中包含${\mathrm{S}}_{3}$中的对象，从而相对于${\mathrm{S}}_{3}$开始成为离群点。根据应用领域的不同，我们可能希望将一组35个对象（如${\mathrm{S}}_{2}$）视为一个簇或一群“相邻”的局部离群点。为了便于处理这种情况，我们可以相应地选择一个MinPtsUB值，即小于35或大于35。对于MinPtsLB，关于其他对象相对于其可以被视为局部离群点的最小对象数量，也可以进行类似的论证。

Having determined MinPtsLB and MinPtsUB, we can compute for each object its LOF values within this range. We propose the heuristic of ranking all objects with respect to the maximum LOF value within the specified range. That is,the ranking of an object $\mathrm{p}$ is based on: $\max \left\{  {{\mathrm{{LOF}}}_{\text{MinPts }}\left( \mathrm{p}\right)  \mid  \text{ MinPtsLB } \leq  \text{ MinPts } \leq  \text{ MinPtsUB }\} }\right.$ .

确定了MinPtsLB和MinPtsUB之后，我们可以计算每个对象在这个范围内的局部离群因子（LOF）值。我们提出一种启发式方法，根据指定范围内的最大LOF值对所有对象进行排序。也就是说，对象$\mathrm{p}$的排序基于：$\max \left\{  {{\mathrm{{LOF}}}_{\text{MinPts }}\left( \mathrm{p}\right)  \mid  \text{ MinPtsLB } \leq  \text{ MinPts } \leq  \text{ MinPtsUB }\} }\right.$。

<!-- Media -->

<!-- figureText: ${\mathrm{S}}_{1}$ ${\mathrm{S}}_{2}$ LOF point in ${\mathrm{S}}_{1}$ point in ${\mathrm{S}}_{2}$ point in ${\mathrm{S}}_{3}$ MinPts (10 to 50) MinPts (10 to 50) MinPts (10 to 50) 1 ${\mathrm{S}}_{3}$ Example dataset -->

<img src="https://cdn.noedgeai.com/0195c8fe-047f-7311-b009-9bd967908604_8.jpg?x=156&y=255&w=1311&h=563&r=0"/>

Figure 8: Ranges of LOF values for different objects in a sample dataset

图8：样本数据集中不同对象的局部离群因子（LOF）值范围

<!-- Media -->

Given all the LOF values within the range, instead of taking the maximum, we could take other aggregates, such as the minimum or the mean. The situation in figure 8 shows that taking the minimum could be inappropriate as the minimum may erase the outlying nature of an object completely. Taking the mean may also have the effect of diluting the outlying nature of the object. We propose to take the maximum to highlight the instance at which the object is the most outlying.

给定该范围内的所有局部离群因子（LOF）值，我们可以采用其他聚合方式，而非取最大值，例如取最小值或平均值。图8中的情况表明，取最小值可能不合适，因为最小值可能会完全消除对象的离群特性。取平均值也可能会削弱对象的离群特性。我们建议取最大值，以突出对象最具离群性的情况。

## 7. EXPERIMENTS

## 7. 实验

In this section, with the proposed heuristic of taking the maximum LOF value within the range, we show that our ideas can be used to successfully identify outliers which appear to be meaningful but cannot be identified by other methods. We start with a synthetical 2-dimensional dataset, for which we show the outlier factors for all objects, in order to give an intuitive notion of the LOF values computed. The second example uses the real-world dataset that has been used in [KN98] to evaluate the DB(pct, dmin) outliers. We repeat their experiments to validate our method. In the third example, we identify meaningful outliers in a database of german soccer players, for which we happen to have a "domain expert" handy, who confirmed the meaningfulness of the outliers found. The last subsection contains performance experiments showing the practicability of our approach even for large, high-dimensional datasets.

在本节中，通过采用所提出的在范围内取最大局部离群因子（LOF）值的启发式方法，我们展示了我们的想法可用于成功识别那些看似有意义但其他方法无法识别的离群点。我们从一个合成的二维数据集开始，展示所有对象的离群因子，以便直观了解计算得到的局部离群因子（LOF）值。第二个示例使用了[KN98]中用于评估DB（百分比，最小距离）离群点的真实数据集。我们重复他们的实验以验证我们的方法。在第三个示例中，我们在一个德国足球运动员数据库中识别出有意义的离群点，碰巧我们有一位“领域专家”，他确认了所发现离群点的意义。最后一个小节包含性能实验，展示了我们的方法即使对于大型高维数据集也具有实用性。

Additionally, we conducted experiments with a 64-dimensional dataset, to demonstrate that our definitions are reasonable in very high dimensional spaces. The feature vectors used are color histograms extracted from tv snapshots [2]. We indentified multiple clusters, e.g. a cluster of pictures from a tennis match, and reasonable local outliers with LOF values of up to 7 .

此外，我们使用一个64维的数据集进行了实验，以证明我们的定义在非常高维的空间中是合理的。所使用的特征向量是从电视快照中提取的颜色直方图[2]。我们识别出了多个聚类，例如一场网球比赛图片的聚类，以及局部离群因子（LOF）值高达7的合理局部离群点。

### 7.1 A Synthetic Example

### 7.1 一个合成示例

The left side of figure 9 shows a 2-dimensional dataset containing one low density Gaussian cluster of 200 objects and three large clusters of 500 objects each. Among these three, one is a dense Gaussian cluster and the other two are uniform clusters of different densities. Furthermore, it contains a couple of outliers. On the right side of figure 9 we plot the LOF of all the objects for MinPts = 40 as a third dimension. We see that the objects in the uniform clusters all have their LOF equal to 1 . Most objects in the Gaussian clusters also have 1 as their LOF values. Slightly outside the Gaussian clusters, there are several weak outliers, i.e., those with relatively low, but larger than 1, LOF values. The remaining seven objects all have significantly larger LOF values. Furthermore, it is clear from the figure that the value of the LOF for each of these outliers depends on the density of the cluster(s) relative to which the object is an outlier, and the distance of the outlier to the cluster(s).

图9的左侧展示了一个二维数据集，其中包含一个由200个对象组成的低密度高斯聚类和三个各由500个对象组成的大聚类。在这三个聚类中，一个是密集的高斯聚类，另外两个是不同密度的均匀聚类。此外，数据集中还包含几个离群点。在图9的右侧，我们将MinPts = 40时所有对象的局部离群因子（LOF）作为第三维进行绘制。我们看到，均匀聚类中的对象其局部离群因子（LOF）值均为1。高斯聚类中的大多数对象其局部离群因子（LOF）值也为1。在高斯聚类的稍外侧，有几个弱离群点，即那些局部离群因子（LOF）值相对较低但大于1的点。其余七个对象的局部离群因子（LOF）值明显更大。此外，从图中可以清楚地看出，每个离群点的局部离群因子（LOF）值取决于该对象相对于其为离群点的聚类的密度，以及离群点到聚类的距离。

<!-- Media -->

<!-- figureText: outlier factor OF 16 14 12 10 8 6 4 2 0 -->

<img src="https://cdn.noedgeai.com/0195c8fe-047f-7311-b009-9bd967908604_8.jpg?x=177&y=1523&w=1304&h=540&r=0"/>

Figure 9: Outlier-factors for points in a sample dataset (MinPts=40)

图9：样本数据集中点的离群因子（MinPts = 40）

<!-- Media -->

### 7.2 Hockey Data

### 7.2 曲棍球数据

In [13], the authors conducted a number of experiments on historical NHL player data; see [13] for a more detailed explanation of the attributes used. We repeat their experiments on the NHL96 dataset, computing the maximum LOF in the MinPts range of 30 to 50 .

在[13]中，作者对历史国家冰球联盟（NHL）球员数据进行了一系列实验；有关所使用属性的更详细解释，请参阅[13]。我们在NHL96数据集上重复他们的实验，计算MinPts范围在30到50之间的最大局部离群因子（LOF）值。

For the first test, on the 3-dimensional subspace of points scored, plus-minus statistics and penalty-minutes, they identified Vladimir Konstantinov as the only DB(0.998, 26.3044) outlier. He was also our top outlier with the LOF value of 2.4. The second strongest local outlier, with the LOF of 2.0, is Matthew Barnaby. For most outliers found, we do not explain why they are outliers from a domain-expert standpoint here; the interested reader can find this information in [13]. The point here is that by ranking outliers with their maximum LOF value, we get almost identical results. In the next subsection, we show how this approach can identify some outliers that [13] cannot find.

在第一次测试中，在得分、正负值统计和犯规分钟数的三维子空间中，他们确定弗拉基米尔·康斯坦丁诺夫（Vladimir Konstantinov）是唯一的DB（0.998，26.3044）离群点。他也是我们的顶级离群点，局部离群因子（LOF）值为2.4。第二强的局部离群点是马修·巴纳比（Matthew Barnaby），局部离群因子（LOF）值为2.0。对于所发现的大多数离群点，我们在此不从领域专家的角度解释它们为何是离群点；感兴趣的读者可以在[13]中找到相关信息。这里的关键是，通过根据最大局部离群因子（LOF）值对离群点进行排名，我们得到了几乎相同的结果。在下一小节中，我们将展示这种方法如何识别出[13]无法找到的一些离群点。

In the second test,they identified the DB(0.997,5)outliers in the 3- dimensional subspace of games played, goals scored and shooting percentage, finding Chris Osgood and Mario Lemieux as outliers. Again, they are our top outliers, Chris Osgood with the LOF of 6.0 and Mario Lemieux with the LOF of 2.8. On our ranked list based on LOF, Steve Poapst, ranked third with the LOF of 2.5, played only three games, scored once and had a shooting percentage of ${50}\%$ .

在第二次测试中，他们在参赛场次、进球数和射门命中率的三维子空间中确定了DB（0.997，5）离群点，发现克里斯·奥斯古德（Chris Osgood）和马里奥·勒米厄（Mario Lemieux）为离群点。同样，他们也是我们的顶级离群点，克里斯·奥斯古德的局部离群因子（LOF）值为6.0，马里奥·勒米厄的局部离群因子（LOF）值为2.8。在我们基于局部离群因子（LOF）的排名列表中，史蒂夫·波普斯特（Steve Poapst）以2.5的局部离群因子（LOF）值排名第三，他只参加了三场比赛，进了一个球，射门命中率为${50}\%$。

### 7.3 Soccer Data

### 7.3 足球数据

In the following experiment, we computed the local outliers for a database of soccer-player information from the "Fußball 1. Bundes-liga" (the German national soccer league) for the season 1998/99. The database consists of 375 players, containing the name, the number of games played, the number of goals scored and the position of the player (goalie, defense, center, offense). From these we derived the average number of goals scored per game, and performed outlier detection on the three-dimensional subspace of number of games, average number of goals per game and position (coded as an integer). In general, this dataset can be partitioned into four clusters corresponding to the positions of the players. We computed the LOF values in the MinPts range of 30 to 50 . Below we discuss all the local outliers with LOF $> {1.5}$ (see table 3),and explain why they are exceptional.

在接下来的实验中，我们计算了1998/99赛季“德国足球甲级联赛”（Fußball 1. Bundes - liga，德国国家足球联赛）球员信息数据库中的局部离群点。该数据库包含375名球员的信息，包括球员姓名、参赛场次、进球数以及球员位置（守门员、后卫、中场、前锋）。我们从这些数据中计算出每场比赛的平均进球数，并对参赛场次、场均进球数和位置（编码为整数）构成的三维子空间进行离群点检测。一般来说，这个数据集可以根据球员位置划分为四个聚类。我们在MinPts值为30到50的范围内计算了局部离群因子（LOF）值。下面我们将讨论所有局部离群因子$> {1.5}$的局部离群点（见表3），并解释它们为何异常。

The strongest outlier is Michael Preetz, who played the maximum number of games and also scored the maximum number of goals, which made him the top scorer in the league ("Torschützenkönig"). He was an outlier relative to the cluster of offensive players. The second strongest outlier is Michael Schjönberg. He played an average number of games, but he was an outlier because most other defense players had a much lower average number of goals scored per game. The reason for this is that he kicked the penalty shots ("Elf-meter") for his team. The player that was ranked third is Hans-Jörg Butt, a goalie who played the maximum number of games possible and scored 7 goals. He was the only goalie to score any goal; he too kicked the penalty shots for his team. On rank positions four and five, we found Ulf Kirsten and Giovane Elber, two offensive players with very high scoring averages.

最显著的离群点是迈克尔·普雷茨（Michael Preetz），他参赛场次最多，进球数也最多，这使他成为联赛最佳射手（“Torschützenkönig”）。相对于前锋球员聚类而言，他是一个离群点。第二显著的离群点是迈克尔·舍恩贝格（Michael Schjönberg）。他的参赛场次处于平均水平，但他是离群点，因为其他大多数后卫球员的场均进球数要低得多。原因是他为球队主罚点球（“Elf - meter”）。排名第三的球员是汉斯 - 约尔格·布特（Hans - Jörg Butt），他是一名守门员，参赛场次达到了可能的最大值，并且打进了7个进球。他是唯一进球的守门员；他同样为球队主罚点球。排名第四和第五的是乌尔夫·柯尔斯滕（Ulf Kirsten）和吉奥瓦内·埃尔伯（Giovane Elber），这两名前锋球员的场均进球数非常高。

<!-- Media -->

<table><tr><td>Rank</td><td>Outlier Factor</td><td>Player Name</td><td>Games Played</td><td>Goals Scored</td><td>Position</td></tr><tr><td>1</td><td>1.87</td><td>Michael Preetz</td><td>34</td><td>23</td><td>Offense</td></tr><tr><td>2</td><td>1.70</td><td>Michael Schjönberg</td><td>15</td><td>6</td><td>Defense</td></tr><tr><td>3</td><td>1.67</td><td>Hans-Jörg Butt</td><td>34</td><td>7</td><td>Goalie</td></tr><tr><td>4</td><td>1.63</td><td>Ulf Kirsten</td><td>31</td><td>19</td><td>Offense</td></tr><tr><td>5</td><td>1.55</td><td>Giovane Elber</td><td>21</td><td>13</td><td>Offense</td></tr><tr><td colspan="3">minimum</td><td>0</td><td>0</td><td/></tr><tr><td colspan="3">median</td><td>21</td><td>1</td><td/></tr><tr><td colspan="3">maximum</td><td>34</td><td>23</td><td/></tr><tr><td colspan="3">mean</td><td>18.0</td><td>1.9</td><td/></tr><tr><td colspan="3">standard deviation</td><td>11.0</td><td>3.0</td><td/></tr></table>

<table><tbody><tr><td>排名</td><td>离群因子</td><td>球员姓名</td><td>参赛场次</td><td>进球数</td><td>位置</td></tr><tr><td>1</td><td>1.87</td><td>迈克尔·普雷茨</td><td>34</td><td>23</td><td>进攻（Offense）</td></tr><tr><td>2</td><td>1.70</td><td>迈克尔·舍恩贝格</td><td>15</td><td>6</td><td>防守（Defense）</td></tr><tr><td>3</td><td>1.67</td><td>汉斯 - 约尔格·布特</td><td>34</td><td>7</td><td>守门员</td></tr><tr><td>4</td><td>1.63</td><td>乌尔夫·柯尔斯滕</td><td>31</td><td>19</td><td>进攻（Offense）</td></tr><tr><td>5</td><td>1.55</td><td>吉奥瓦内·埃尔伯</td><td>21</td><td>13</td><td>进攻（Offense）</td></tr><tr><td colspan="3">最小值</td><td>0</td><td>0</td><td></td></tr><tr><td colspan="3">中位数</td><td>21</td><td>1</td><td></td></tr><tr><td colspan="3">最大值</td><td>34</td><td>23</td><td></td></tr><tr><td colspan="3">均值</td><td>18.0</td><td>1.9</td><td></td></tr><tr><td colspan="3">标准差</td><td>11.0</td><td>3.0</td><td></td></tr></tbody></table>

Table 3: Results of the soccer player dataset

表3：足球运动员数据集的结果

<!-- Media -->

### 7.4 Performance

### 7.4 性能

In this section, we evaluate the performance of the computation of LOF. The following experiments were conducted on an Pentium III-450 workstation with 256 MB main memory running Linux 2.2. All algorithms were implemented in Java and executed on the IBM JVM 1.1.8. The datasets used were generated randomly, containing different numbers of Gaussian clusters of different sizes and densities. All times are wall-clock times, i.e. include CPU-time and I/O.

在本节中，我们评估局部离群因子（LOF，Local Outlier Factor）计算的性能。以下实验是在一台配备256MB主内存、运行Linux 2.2系统的奔腾III - 450工作站上进行的。所有算法均用Java实现，并在IBM JVM 1.1.8上执行。所使用的数据集是随机生成的，包含不同数量、不同大小和密度的高斯簇。所有时间均为实际时间，即包括CPU时间和I/O时间。

To compute the LOF values within the range between MinPtsLB and MinPtsUB,for all the $\mathrm{n}$ objects in the database D,we implemented a two-step algorithm. In the first step, the MinPtsUB-nearest neighborhoods are found, and in the second step the LOFs are computed. Let us look at these two steps in detail.

为了计算数据库D中所有$\mathrm{n}$个对象在MinPtsLB和MinPtsUB范围内的局部离群因子（LOF）值，我们实现了一个两步算法。第一步，找出MinPtsUB最近邻；第二步，计算局部离群因子（LOF）。让我们详细了解这两个步骤。

In the first step, the MinPtsUB-nearest neighbors for every point p are materialized, together with their distances to p. The result of this step is a materialization database M of size n*MinPtsUB distances. Note that the size of this intermediate result is independent of the dimension of the original data. The runtime complexity of this step

在第一步中，为每个点p找出其MinPtsUB最近邻，并记录它们到点p的距离。此步骤的结果是一个大小为n * MinPtsUB个距离的物化数据库M。请注意，这个中间结果的大小与原始数据的维度无关。此步骤的运行时复杂度

<!-- Media -->

<!-- figureText: 18000 500 600 700 800 900 n [*1000] 200 10d 14000 5d 2d 12000 10000 time [sec] 8000 6000 4000 2000 100 200 300 400 -->

<img src="https://cdn.noedgeai.com/0195c8fe-047f-7311-b009-9bd967908604_9.jpg?x=841&y=1565&w=685&h=439&r=0"/>

Figure 10: Runtime of the materialization of the 50-nn queries for different dataset sizes and different dimensions using an index

图10：使用索引对不同数据集大小和不同维度进行50近邻（nn，nearest neighbor）查询的物化运行时间

<!-- figureText: 800 500 600 700 800 900 n [*1000] 20d 700 100 600 time [sec] 500 300 200 100 100 200 300 400 -->

<img src="https://cdn.noedgeai.com/0195c8fe-047f-7311-b009-9bd967908604_10.jpg?x=110&y=262&w=683&h=442&r=0"/>

Figure 11: Runtime for the computation of the LOFs for different dataset sizes

图11：不同数据集大小下计算局部离群因子（LOF）的运行时间

<!-- Media -->

is $\mathrm{O}$ (n*time for a k-nn query). For the k-nn queries,we have a choice among different methods. For low-dimensional data, we can use a grid based approach which can answer k-nn queries in constant time,leading to a complexity of $\mathrm{O}\left( \mathrm{n}\right)$ for the materialization step. For medium to medium high-dimensional data, we can use an index,which provides an average complexity of $\mathrm{O}\left( {\log \mathrm{n}}\right)$ for $\mathrm{k}$ -nn queries,leading to a complexity of $\mathrm{O}\left( {\mathrm{n}\log \mathrm{n}}\right)$ for the materialization. For extremely high-dimensional data, we need to use a sequential scan or some variant of it, e.g. the VA-file ([21]), with a complexity of $\mathrm{O}\left( \mathrm{n}\right)$ ,leading to a complexity of $\mathrm{O}\left( {\mathrm{n}}^{2}\right)$ for the materialization step. In our experiments,we used a variant of the X-tree ([4]),leading to the complexity of $\mathrm{O}\left( {\mathrm{n}\log \mathrm{n}}\right)$ . Figure 10 shows performance experiments for different dimensional datasets and MinPtsUB $= {50}$ . The times shown do include the time to build the index. Obviously, the index works very well for 2-dimensional and 5- dimensional dataset, leading to a near linear performance, but degenerates for the 10-dimensional and 20-dimensionsal dataset. It is a well known effect of index structures, that their effectivity decreases with increasing dimension.

为$\mathrm{O}$（n * k近邻查询的时间）。对于k近邻查询，我们有多种方法可供选择。对于低维数据，我们可以使用基于网格的方法，该方法可以在恒定时间内回答k近邻查询，从而使物化步骤的复杂度为$\mathrm{O}\left( \mathrm{n}\right)$。对于中等到中高维数据，我们可以使用索引，该索引为$\mathrm{k}$近邻查询提供的平均复杂度为$\mathrm{O}\left( {\log \mathrm{n}}\right)$，从而使物化的复杂度为$\mathrm{O}\left( {\mathrm{n}\log \mathrm{n}}\right)$。对于极高维数据，我们需要使用顺序扫描或其某种变体，例如VA文件（[21]），其复杂度为$\mathrm{O}\left( \mathrm{n}\right)$，从而使物化步骤的复杂度为$\mathrm{O}\left( {\mathrm{n}}^{2}\right)$。在我们的实验中，我们使用了X树的一种变体（[4]），使复杂度为$\mathrm{O}\left( {\mathrm{n}\log \mathrm{n}}\right)$。图10展示了不同维度数据集和MinPtsUB $= {50}$的性能实验。所示时间包括构建索引的时间。显然，该索引对于二维和五维数据集效果很好，接近线性性能，但对于十维和二十维数据集效果不佳。众所周知，索引结构的有效性会随着维度的增加而降低。

In the second step, the LOF values are computed using the materialization database M. The original database D is not needed for this step, as M contains sufficient information to compute the LOFs. The database $\mathrm{M}$ is scanned twice for every value of MinPts between MinPtsLB and MinPtsUB. In the first scan, the local reachability densities of every object are computed. In the second step, the final LOF values are computed and written to a file. These values can then be used to rank the objects according to their maximum LOF value in the interval of MinPtsLB and MinPtsUB. The time complexity of this step is $\mathrm{O}\left( \mathrm{n}\right)$ . This is confirmed by the graph shown in figure 11,where the LOF values for MinPtsLB $= {10}$ to MinPtsUB $= {50}$ were computed.

在第二步中，使用物化数据库M计算局部离群因子（LOF）值。此步骤不需要原始数据库D，因为M包含了计算局部离群因子（LOF）的足够信息。对于MinPtsLB和MinPtsUB之间的每个MinPts值，数据库$\mathrm{M}$会被扫描两次。第一次扫描时，计算每个对象的局部可达密度。第二步，计算最终的局部离群因子（LOF）值并写入文件。然后可以根据这些值在MinPtsLB和MinPtsUB区间内的最大局部离群因子（LOF）值对对象进行排序。此步骤的时间复杂度为$\mathrm{O}\left( \mathrm{n}\right)$。图11中的图表证实了这一点，该图计算了MinPtsLB $= {10}$到MinPtsUB $= {50}$的局部离群因子（LOF）值。

## 8. CONCLUSIONS

## 8. 结论

Finding outliers is an important task for many KDD applications. Existing proposals consider being an outlier as a binary property. In this paper, we show that for many situations, it is meaningful to consider being an outlier not as a binary property, but as the degree to which the object is isolated from its surrounding neighborhood. We introduce the notion of the local outlier factor LOF, which captures exactly this relative degree of isolation. We show that our definition of LOF enjoys many desirable properties. For objects deep inside a cluster, the LOF value is approximately 1 . For other objects, we give tight lower and upper bounds on the LOF value, regardless of whether the MinPts-nearest neighbors come from one or more clusters. Furthermore, we analyze how the LOF value depends on the MinPts parameter. We give practical guidelines on how to select a range of MinPts values to use, and propose the heuristic of ranking objects by their maximum LOF value within the selected range. Experimental results demonstrate that our heuristic appears to be very promising in that it can identify meaningful local outliers that previous approaches cannot find. Last but not least, we show that our approach of finding local outliers is efficient for datasets where the nearest neighbor queries are supported by index structures and still practical for very large datasets.

对于许多知识发现与数据挖掘（KDD）应用而言，寻找离群点是一项重要任务。现有方案将离群点视为一种二元属性。在本文中，我们表明，在许多情况下，将离群点视为对象与其周围邻域的隔离程度，而非二元属性，是有意义的。我们引入了局部离群因子（LOF）的概念，它精确地捕捉了这种相对隔离程度。我们证明了我们对LOF的定义具有许多理想的性质。对于处于聚类内部深处的对象，LOF值近似为1。对于其他对象，无论MinPts近邻是来自一个还是多个聚类，我们都给出了LOF值的严格上下界。此外，我们分析了LOF值如何依赖于MinPts参数。我们给出了如何选择MinPts值范围的实用指南，并提出了根据对象在所选范围内的最大LOF值对其进行排序的启发式方法。实验结果表明，我们的启发式方法似乎非常有前景，因为它能够识别出先前方法无法发现的有意义的局部离群点。最后但同样重要的是，我们表明，对于由索引结构支持最近邻查询的数据集，我们寻找局部离群点的方法是高效的，并且对于非常大的数据集仍然实用。

There are two directions for ongoing work. The first one is on how to describe or explain why the identified local outliers are exceptional. This is particularly important for high-dimensional datasets, because a local outlier may be outlying only on some, but not on all, dimensions (cf. [14]). The second one is to further improve the performance of LOF computation. For both of these directions, it is interesting to investigate how LOF computation can "handshake" with a hierarchical clustering algorithm, like OPTICS [2]. On the one hand, such an algorithm may provide more detailed information about the local outliers, e.g., by analyzing the clusters relative to which they are outlying. On the other hand, computation may be shared between LOF processing and clustering. The shared computation may include k-nn queries and reachability distances.

目前有两个研究方向。第一个方向是如何描述或解释所识别的局部离群点为何异常。这对于高维数据集尤为重要，因为局部离群点可能仅在某些维度上是离群的，而非在所有维度上（参见[14]）。第二个方向是进一步提高LOF计算的性能。对于这两个方向，研究LOF计算如何与像OPTICS [2]这样的层次聚类算法“握手”是很有趣的。一方面，这样的算法可以通过分析局部离群点相对于哪些聚类是离群的，来提供关于局部离群点的更详细信息。另一方面，LOF处理和聚类之间可以共享计算。共享计算可能包括k近邻查询和可达距离。

## References

## 参考文献

[1] Arning, A., Agrawal R., Raghavan P.: "A Linear Method for Deviation Detection in Large Databases", Proc. 2nd Int. Conf. on Knowledge Discovery and Data Mining, Portland, OR, AAAI Press, 1996, p. 164-169.

[2] Ankerst M., Breunig M. M., Kriegel H.-P., Sander J.: "OPTICS: Ordering Points To Identify the Clustering Structure", Proc. ACM SIGMOD Int. Conf. on Management of Data, Philadelphia, PA, 1999.

[3] Agrawal R., Gehrke J., Gunopulos D., Raghavan P.: "Automatic Subspace Clustering of High Dimensional Data for Data Mining Applications", Proc. ACM SIGMOD Int. Conf. on Management of Data, Seattle, WA, 1998, pp. 94-105.

[4] Berchthold S., Keim D. A., Kriegel H.-P.: "The X-Tree: An Index Structure for High-Dimensional Data", 22nd Conf. on Very Large Data Bases, Bombay, India, 1996, pp. 28-39.

[5] Barnett V., Lewis T.: "Outliers in statistical data", John Wiley, 1994.

[6] DuMouchel W., Schonlau M.: "A Fast Computer Intrusion Detection Algorithm based on Hypothesis Testing of Command Transition Probabilities", Proc. 4th Int. Conf. on Knowledge Discovery and Data Mining, New York, NY, AAAI Press, 1998, pp. 189-193.

[7] Ester M., Kriegel H.-P., Sander J., Xu X.: "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise", Proc. 2nd Int. Conf. on Knowledge Discovery and Data Mining, Portland, OR, AAAI Press, 1996, pp. 226-231.

[8] Fawcett T., Provost F.: "Adaptive Fraud Detection", Data Mining and Knowledge Discovery Journal, Kluwer Academic Publishers, Vol. 1, No. 3, 1997, pp. 291-316.

[9] Fayyad U., Piatetsky-Shapiro G., Smyth P.: "Knowledge

Discovery and Data Mining: Towards a Unifying Framework", Proc. 2nd Int. Conf. on Knowledge Discovery and Data Mining, Portland, OR, 1996, pp. 82-88.

[10] Hawkins, D.: "Identification of Outliers", Chapman and Hall, London, 1980.

[11] Hinneburg A., Keim D. A.: "An Efficient Approach to Clustering in Large Multimedia Databases with Noise", Proc. 4th Int. Conf. on Knowledge Discovery and Data Mining, New York City, NY, 1998,pp. 58-65.

[12] Johnson T., Kwok I., Ng R.: "Fast Computation of 2- Dimensional Depth Contours", Proc. 4th Int. Conf. on Knowledge Discovery and Data Mining, New York, NY, AAAI Press, 1998, pp. 224-228.

[13] Knorr E. M., Ng R. T.: "Algorithms for Mining Distance-Based Outliers in Large Datasets", Proc. 24th Int. Conf. on Very Large Data Bases, New York, NY, 1998, pp. 392-403.

[14] Knorr E. M., Ng R. T.: "Finding Intensional Knowledge of Distance-based Outliers", Proc. 25th Int. Conf. on Very Large Data Bases, Edinburgh, Scotland, 1999, pp. 211-222.

[15] Ng R. T., Han J.: "Efficient and Effective Clustering Methods for Spatial Data Mining", Proc. 20th Int. Conf. on Very Large Data Bases, Santiago, Chile, Morgan Kaufmann Publishers, San Francisco, CA, 1994, pp. 144-155.

[16] Preparata F., Shamos M.: "Computational Geometry: an Introduction", Springer, 1988.

[17] Ramaswamy S., Rastogi R., Kyuseok S.: "Efficient Algorithms for Mining Outliers from Large Data Sets", Proc. ACM SIDMOD Int. Conf. on Management of Data, 2000.

[18] Ruts I., Rousseeuw P.: "Computing Depth Contours of Bivariate Point Clouds, Journal of Computational Statistics and Data Analysis, 23, 1996, pp. 153-168.

[19] Sheikholeslami G., Chatterjee S., Zhang A.: "WaveCluster: A Multi-Resolution Clustering Approach for Very Large Spatial Databases", Proc. Int. Conf. on Very Large Data Bases, New York, NY, 1998, pp. 428-439.

[20] Tukey J. W.: "Exploratory Data Analysis", Addison-Wesley, 1977.

[21] Weber R., Schek Hans-J., Blott S.: "A Quantitative Analysis and Performance Study for Similarity-Search Methods in High-Dimensional Spaces", Proc. Int. Conf. on Very Large Data Bases, New York, NY, 1998, pp. 194-205.

[22] Wang W., Yang J., Muntz R.: "STING: A Statistical Information Grid Approach to Spatial Data Mining", Proc. 23th Int. Conf. on Very Large Data Bases, Athens, Greece, Morgan Kaufmann Publishers, San Francisco, CA, 1997, pp. 186-195.

[23] Zhang T., Ramakrishnan R., Linvy M.: "BIRCH: An Efficient Data Clustering Method for Very Large Databases", Proc. ACM SIGMOD Int. Conf. on Management of Data, ACM Press, New York, 1996, pp.103-114.

## Appendix

## 附录

Proof of Theorem 2 (Sketch): Let $p$ be an object from the database $\mathrm{D},1 \leq$ MinPts $\leq  \left| \mathrm{D}\right|$ ,and ${\mathrm{C}}_{1},{\mathrm{C}}_{2},\ldots ,{\mathrm{C}}_{\mathrm{n}}$ be a partition of ${\mathrm{N}}_{\text{MinPts }}\left( \mathrm{p}\right)$ , i.e. ${\mathrm{N}}_{\text{MinPts }}\left( \mathrm{p}\right)  = {\mathrm{C}}_{1} \cup  {\mathrm{C}}_{2} \cup  \ldots  \cup  {\mathrm{C}}_{\mathrm{n}} \cup  \{ \mathrm{p}\}$ with ${\mathrm{C}}_{\mathrm{i}} \cap  {\mathrm{C}}_{\mathrm{j}} = \varnothing$ , ${\mathrm{C}}_{\mathrm{i}} \neq  \varnothing \;$ for $1 \leq  \mathrm{i},\mathrm{j} \leq  \mathrm{n},\;\mathrm{i} \neq  \mathrm{j}$ . Furthermore,let ${\xi }_{i} = \left| {C}_{i}\right| /\left| {{N}_{\text{MinPts }}\left( p\right) }\right|$ be the percentage of objects in p’s neighborhood which are in the set ${\mathrm{C}}_{\mathrm{i}}$ . Let the notions direct ${}_{\min }^{\mathrm{i}}\left( \mathrm{p}\right)$ , direct ${}^{i}{}_{\max }\left( p\right)$ ,indirect ${}_{\min }^{i}\left( p\right)$ ,and indirect ${}_{\max }^{i}\left( p\right)$ be defined analogously to ${\operatorname{direct}}_{\min }\left( \mathrm{p}\right) ,{\operatorname{direct}}_{\max }\left( \mathrm{p}\right)$ ,indirect ${\mathrm{t}}_{\min }\left( \mathrm{p}\right)$ ,and indi- ${\text{rect}}_{\max }\left( \mathrm{p}\right)$ but restricted to the set ${\mathrm{C}}_{\mathrm{i}}$ .

定理2的证明（概要）：设$p$是数据库$\mathrm{D},1 \leq$ MinPts $\leq  \left| \mathrm{D}\right|$中的一个对象，${\mathrm{C}}_{1},{\mathrm{C}}_{2},\ldots ,{\mathrm{C}}_{\mathrm{n}}$是${\mathrm{N}}_{\text{MinPts }}\left( \mathrm{p}\right)$的一个划分，即${\mathrm{N}}_{\text{MinPts }}\left( \mathrm{p}\right)  = {\mathrm{C}}_{1} \cup  {\mathrm{C}}_{2} \cup  \ldots  \cup  {\mathrm{C}}_{\mathrm{n}} \cup  \{ \mathrm{p}\}$且${\mathrm{C}}_{\mathrm{i}} \cap  {\mathrm{C}}_{\mathrm{j}} = \varnothing$，对于$1 \leq  \mathrm{i},\mathrm{j} \leq  \mathrm{n},\;\mathrm{i} \neq  \mathrm{j}$有${\mathrm{C}}_{\mathrm{i}} \neq  \varnothing \;$。此外，设${\xi }_{i} = \left| {C}_{i}\right| /\left| {{N}_{\text{MinPts }}\left( p\right) }\right|$是p的邻域中属于集合${\mathrm{C}}_{\mathrm{i}}$的对象的百分比。设直接${}_{\min }^{\mathrm{i}}\left( \mathrm{p}\right)$、直接${}^{i}{}_{\max }\left( p\right)$、间接${}_{\min }^{i}\left( p\right)$和间接${}_{\max }^{i}\left( p\right)$的概念与${\operatorname{direct}}_{\min }\left( \mathrm{p}\right) ,{\operatorname{direct}}_{\max }\left( \mathrm{p}\right)$、间接${\mathrm{t}}_{\min }\left( \mathrm{p}\right)$和间接${\text{rect}}_{\max }\left( \mathrm{p}\right)$类似，但仅限于集合${\mathrm{C}}_{\mathrm{i}}$。

(a)

$$
\operatorname{LOF}\left( p\right)  \geq  \left( {\mathop{\sum }\limits_{{i = 1}}^{n}{\xi }_{i} \cdot  \operatorname{direct}{\min }_{i}\left( p\right) }\right)  \cdot  \left( {\mathop{\sum }\limits_{{i = 1}}^{n}\frac{{\xi }_{i}}{\operatorname{indirect}{\max }_{i}^{i}\left( p\right) }}\right) 
$$

$\forall \mathrm{o} \in  {\mathrm{C}}_{\mathrm{i}}$ : reach-dist $\left( {\mathrm{p},\mathrm{o}}\right)  \geq  {\operatorname{direct}}_{\min }^{\mathrm{i}}\left( \mathrm{p}\right)$ ,by definition of

$\forall \mathrm{o} \in  {\mathrm{C}}_{\mathrm{i}}$ ：可达距离（reach-dist） $\left( {\mathrm{p},\mathrm{o}}\right)  \geq  {\operatorname{direct}}_{\min }^{\mathrm{i}}\left( \mathrm{p}\right)$ ，根据……的定义

direct ${}_{\min }^{\mathrm{i}}\left( \mathrm{p}\right)$ . $\Rightarrow$

直接 ${}_{\min }^{\mathrm{i}}\left( \mathrm{p}\right)$ 。 $\Rightarrow$

$$
1\mathop{\sum }\limits_{{\rho  \in  {N}_{MinPts}\left( p\right) }}\frac{\text{ reach - dist }\left( {p,\rho }\right) }{\left| {N}_{MinPts}\left( p\right) \right| } = {\left( \mathop{\sum }\limits_{{i = 1}}^{n}\mathop{\sum }\limits_{{o \in  {C}_{i}}}\frac{\text{ reach - dist }\left( {p,o}\right) }{\left| {N}_{MinPts}\left( p\right) \right| }\right) }^{-1}
$$

$$
 \leq  {\left( \mathop{\sum }\limits_{{i = 1}}^{n}\mathop{\sum }\limits_{{o \in  {C}_{i}}}\frac{\operatorname{direct}\min \left( p\right) }{\left| {N}_{MinPts}\left( p\right) \right| }\right) }^{-1} = 
$$

$$
 = {\left( \mathop{\sum }\limits_{{i = 1}}^{n}\frac{\left| {C}_{i}\right|  \cdot  \operatorname{direct}{m}_{i\min }^{i}\left( p\right) }{\left| {N}_{MinPts}\left( p\right) \right| }\right) }^{-1} = {\left( \mathop{\sum }\limits_{{i = 1}}^{n}{\xi }_{i} \cdot  \operatorname{direct}{m}_{iin}^{i}\left( p\right) \right) }^{-1}
$$

$$
\text{i.e.}\operatorname{lrd}\left( p\right)  \leq  {\left( \mathop{\sum }\limits_{{i = 1}}^{n}{\xi }_{i} \cdot  \operatorname{direct}{\min }_{i}\left( p\right) \right) }^{-1}
$$

$$
\forall \mathrm{q} \in  {\mathrm{N}}_{\text{MinPts }}\left( \mathrm{o}\right)  : \;\text{ reach-dist }\left( {\mathrm{o},\mathrm{q}}\right)  \leq  {\text{ indirect }}_{\text{max }}^{\mathrm{i}}\left( \mathrm{p}\right) 
$$

$$
 \Rightarrow  \operatorname{lrd}\left( \mathrm{o}\right)  \geq  \frac{1}{{\operatorname{indirect}}_{\max }^{\mathrm{i}}\left( \mathrm{p}\right) }\text{. Thus,it follows that}
$$

$$
\operatorname{LOF}\left( p\right)  = \frac{\sigma {N}_{\operatorname{MinPts}}\left( p\right) }{\left| {N}_{\operatorname{MinPts}}\left( p\right) \right| } = \frac{1}{\operatorname{lrd}\left( p\right) } \cdot  \mathop{\sum }\limits_{{o \in  {N}_{\operatorname{MinPts}}\left( p\right) }}\frac{\operatorname{lrd}\left( o\right) }{\left| {N}_{\operatorname{MinPts}}\left( p\right) \right| }
$$

$$
 \geq  \left( {\mathop{\sum }\limits_{{i = 1}}^{n}{\xi }_{i} \cdot  {\operatorname{direct}}^{i}\min \left( p\right) }\right)  \cdot  \left( {\mathop{\sum }\limits_{{i = 1}}^{n}\mathop{\sum }\limits_{{o \in  {C}_{i}}}\frac{\frac{1}{{\operatorname{indirect}}_{\max }^{i}\left( p\right) }}{\left| {N}_{MinPts}\left( p\right) \right| }}\right) 
$$

$$
 = \left( {\mathop{\sum }\limits_{{i = 1}}^{n}{\xi }_{i} \cdot  {\operatorname{direct}}^{i}\min \left( p\right) }\right)  \cdot  \left( {\mathop{\sum }\limits_{{i = 1}}^{n}\frac{{\xi }_{i}}{{\operatorname{indirect}}^{i}\max \left( p\right) }}\right) 
$$

(b)

$$
\operatorname{LOF}\left( p\right)  \leq  \left( {\mathop{\sum }\limits_{{i = 1}}^{n}{\xi }_{i} \cdot  {\operatorname{direct}}_{\max }^{i}\left( p\right) }\right)  \cdot  \left( {\mathop{\sum }\limits_{{i = 1}}^{n}\frac{{\xi }_{i}}{{\operatorname{indirect}}_{\min }^{i}\left( p\right) }}\right) 
$$

: analogously.

：类似地。