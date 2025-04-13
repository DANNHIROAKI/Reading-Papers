# Algorithms for Mining Distance-Based Outliers in Large Datasets

# 大型数据集中基于距离的离群点挖掘算法

Edwin M. Knorr and Raymond T. Ng

埃德温·M·诺尔（Edwin M. Knorr）和雷蒙德·T·吴（Raymond T. Ng）

Department of Computer Science

计算机科学系

University of British Columbia

英属哥伦比亚大学

Vancouver, BC V6T 1Z4 Canada

加拿大不列颠哥伦比亚省温哥华市V6T 1Z4

\{knorr,rng\}@cs.ubc.ca

\{knorr,rng\}@cs.ubc.ca

## Abstract

## 摘要

This paper deals with finding outliers (exceptions) in large, multidimensional datasets. The identification of outliers can lead to the discovery of truly unexpected knowledge in areas such as electronic commerce, credit card fraud, and even the analysis of performance statistics of professional athletes. Existing methods that we have seen for finding outliers in large datasets can only deal efficiently with two dimensions/attributes of a dataset. Here,we study the notion of ${DB}$ - (Distance-Based) outliers. While we provide formal and empirical evidence showing the usefulness of ${DB}$ -outliers,we focus on the development of algorithms for computing such outliers.

本文探讨在大型多维数据集中寻找离群点（异常值）的问题。离群点的识别有助于在电子商务、信用卡欺诈甚至职业运动员表现统计分析等领域发现真正意想不到的知识。我们所了解的现有大型数据集离群点查找方法只能有效处理数据集的两个维度/属性。在此，我们研究${DB}$ -（基于距离的）离群点的概念。在提供正式和实证证据证明${DB}$ -离群点的实用性的同时，我们专注于开发计算此类离群点的算法。

First, we present two simple algorithms, both having a complexity of $O\left( {k{N}^{2}}\right) ,k$ being the dimensionality and $N$ being the number of objects in the dataset. These algorithms readily support datasets with many more than two attributes. Second, we present an optimized cell-based algorithm that has a complexity that is linear wrt $N$ ,but exponential wrt $k$ . Third,for datasets that are mainly disk-resident, we present another version of the cell-based algorithm that guarantees at most 3 passes over a dataset. We provide

首先，我们提出两种简单算法，其复杂度均为$O\left( {k{N}^{2}}\right) ,k$ （$O\left( {k{N}^{2}}\right) ,k$ 为数据集的维度，$N$ 为数据集中对象的数量）。这些算法能够轻松支持具有两个以上属性的数据集。其次，我们提出一种优化的基于单元格的算法，其复杂度相对于$N$ 是线性的，但相对于$k$ 是指数级的。第三，对于主要存储在磁盘上的数据集，我们提出了基于单元格算法的另一个版本，该版本保证对数据集最多进行3次遍历。我们提供

experimental results showing that these cell-based algorithms are by far the best for $k \leq  4$ .

实验结果表明，到目前为止，这些基于单元格的算法对于$k \leq  4$ 是最优的。

## 1 Introduction

## 1 引言

Knowledge discovery tasks can be classified into four general categories: (a) dependency detection, (b) class identification, (c) class description, and (d) exception/outlier detection. The first three categories of tasks correspond to patterns that apply to many objects, or to a large percentage of objects, in the dataset. Most research in data mining (e.g., association rules [AIS93, MTV95, MT96], classification [AGI+92], data clustering [EKSX96, NH94, ZRL96], and concept generalization [HCC92, KN96]) belongs to these 3 categories. The fourth category, in contrast, focuses on a very small percentage of data objects, which are often ignored or discarded as noise. For example, some existing algorithms in machine learning and data mining have considered outliers, but only to the extent of tolerating them in whatever the algorithms are supposed to do [AL88, EKSX96, NH94, ZRL96].

知识发现任务可分为四大类：（a）依赖关系检测，（b）类别识别，（c）类别描述，以及（d）异常/离群点检测。前三类任务对应于适用于数据集中许多对象或大部分对象的模式。数据挖掘领域的大多数研究（例如，关联规则 [AIS93, MTV95, MT96]、分类 [AGI+92]、数据聚类 [EKSX96, NH94, ZRL96] 和概念泛化 [HCC92, KN96]）都属于这三类。相比之下，第四类任务关注的是数据对象中极小的一部分，这些对象通常被视为噪声而被忽略或丢弃。例如，机器学习和数据挖掘中的一些现有算法考虑过离群点，但只是在算法的预期操作中容忍它们的存在 [AL88, EKSX96, NH94, ZRL96]。

"One person's noise is another person's signal." Indeed, for some applications, the rare events are often more interesting than the common ones, from a knowledge discovery standpoint. Sample applications include the detection of credit card fraud and the monitoring of criminal activities in electronic commerce [Kno97]. For example, in Internet commerce or smart card applications, we expect many low-value transactions to occur. However, it is the exceptional cases-exceptional perhaps in monetary amount, type of purchase, timeframe, location, or some combination thereof-that may interest us, either for fraud detection or for marketing reasons.

“一个人的噪声可能是另一个人的信号。” 实际上，从知识发现的角度来看，对于某些应用而言，罕见事件往往比常见事件更有趣。示例应用包括信用卡欺诈检测和电子商务中的犯罪活动监控 [Kno97]。例如，在互联网商务或智能卡应用中，我们预计会发生许多小额交易。然而，真正可能引起我们兴趣的是那些异常情况——可能在金额、购买类型、时间范围、地点或这些因素的某种组合上存在异常——无论是出于欺诈检测还是营销目的。

IBM's Advanced Scout data mining system has shown that data mining tools can be used to discover knowledge for strategic advantage in National Basketball Association games $\left\lbrack  {{\mathrm{{BCP}}}^{ + }{97}}\right\rbrack$ . In Section 2.2,we give some concrete examples of detecting outliers in National Hockey League (NHL) data.

IBM的高级侦察数据挖掘系统已表明，数据挖掘工具可用于在国家篮球协会（NBA）比赛中发现具有战略优势的知识 $\left\lbrack  {{\mathrm{{BCP}}}^{ + }{97}}\right\rbrack$。在2.2节中，我们给出一些在国家冰球联盟（NHL）数据中检测离群点的具体示例。

---

<!-- Footnote -->

Permission to copy without fee all or part of this material is granted provided that the copies are not made or distributed for direct commercial advantage, the VLDB copyright notice and the title of the publication and its date appear, and notice is given that copying is by permission of the Very Large Data Base Endowment. To copy otherwise, or to republish, requires a fee and/or special permission from the Endowment.

允许免费复制本材料的全部或部分内容，但前提是复制件不得用于直接商业利益，必须保留VLDB版权声明、出版物标题及其日期，并注明复制获得了超大型数据库基金会的许可。否则，如需复制或重新发布，则需向该基金会支付费用和/或获得特别许可。

Proceedings of the 24th VLDB Conference

第24届VLDB会议论文集

New York, USA, 1998

美国纽约，1998年

<!-- Footnote -->

---

### 1.1 Related Work

### 1.1 相关工作

Most of the existing work on outlier detection lies in the field of statistics [BL94, Haw80]. While there is no single, generally accepted, formal definition of an outlier, Hawkins' definition captures the spirit: "an outlier is an observation that deviates so much from other observations as to arouse suspicions that it was generated by a different mechanism" [Haw80]. Accordingly, over one hundred discordancy/outlier tests have been developed for different circumstances, depending on: (i) the data distribution, (ii) whether or not the distribution parameters (e.g., mean and variance) are known, (iii) the number of expected outliers, and even (iv) the types of expected outliers (e.g., upper or lower outliers in an ordered sample) [BL94]. However, those tests suffer from two serious problems. First, almost all of them are univariate (i.e., single attribute). This restriction makes them unsuitable for multidimensional datasets. Second, all of them are distribution-based. In numerous situations where we do not know whether a particular attribute follows a normal distribution, a gamma distribution, etc., we have to perform extensive testing to find a distribution that fits the attribute.

现有的大多数关于离群值检测的工作都属于统计学领域 [BL94, Haw80]。虽然对于离群值没有一个统一的、被普遍接受的正式定义，但霍金斯（Hawkins）的定义抓住了其核心：“离群值是指与其他观测值偏差极大，以至于让人怀疑它是由不同机制产生的观测值” [Haw80]。因此，针对不同情况已经开发了一百多种不一致性/离群值检验方法，这些方法取决于：（i）数据分布；（ii）分布参数（例如均值和方差）是否已知；（iii）预期离群值的数量；甚至（iv）预期离群值的类型（例如有序样本中的上离群值或下离群值） [BL94]。然而，这些检验方法存在两个严重问题。首先，几乎所有方法都是单变量的（即单个属性）。这种限制使得它们不适用于多维数据集。其次，所有方法都是基于分布的。在许多情况下，我们并不知道某个特定属性是服从正态分布、伽马分布还是其他分布，这时就必须进行大量测试来找到适合该属性的分布。

To improve the situation, some methods in computational statistics have been developed, which can be best described as depth-based. Based on some definition of depth, data objects are organized in layers in the data space, with the expectation that shallow layers are more likely to contain outlying data objects than the deep layers. Peeling and depth contours are two notions of depth studied in [PS88, RR96]. These depth-based methods avoid the problem of distribution fitting, and conceptually allow multidimensional data objects to be processed. However, in practice, the computation of $k$ -dimensional layers relies on the computation of $k$ -dimensional convex hulls. Because the lower bound complexity of computing a $k$ -dimensional convex hull is $\Omega \left( {N}^{\left\lceil  \frac{k}{2}\right\rceil  }\right)$ ,depth-based methods are not expected to be practical for more than 4 dimensions for large datasets. In fact, existing depth-based methods only give acceptable performance for $k \leq  2$ [RR96].

为了改善这种情况，计算统计学中已经开发了一些方法，这些方法可以最好地描述为基于深度的方法。基于某种深度定义，数据对象在数据空间中按层组织，预期浅层比深层更有可能包含离群数据对象。剥皮法（Peeling）和深度轮廓（depth contours）是 [PS88, RR96] 中研究的两种深度概念。这些基于深度的方法避免了分布拟合的问题，并且从概念上允许处理多维数据对象。然而，在实践中，$k$ 维层的计算依赖于 $k$ 维凸包的计算。由于计算 $k$ 维凸包的复杂度下限是 $\Omega \left( {N}^{\left\lceil  \frac{k}{2}\right\rceil  }\right)$，因此对于大型数据集，基于深度的方法在超过 4 维的情况下预计不实用。实际上，现有的基于深度的方法仅在 $k \leq  2$ 时才能给出可接受的性能 [RR96]。

Arning, et al. [AAR96] search a dataset for implicit redundancies, and extract data objects called sequential exceptions that maximize the reduction in Kolmogorov complexity. This notion of outliers is very different from the aforementioned statistical definitions of outliers. As will be seen shortly, it is also very different from the notion of outliers considered here, primarily because there is not an associated notion of distance and similarity measure between objects.

阿宁（Arning）等人 [AAR96] 在数据集中搜索隐式冗余，并提取称为顺序异常的数据对象，这些对象能最大程度地降低柯尔莫哥洛夫复杂度（Kolmogorov complexity）。这种离群值的概念与上述统计学中离群值的定义非常不同。正如我们很快会看到的，它也与本文所考虑的离群值概念非常不同，主要是因为对象之间没有相关的距离和相似度度量概念。

Finally, a few clustering algorithms, such as CLARANS [NH94], DBSCAN [EKSX96] and BIRCH [ZRL96], are developed with exception-handling capabilities. However, their main objective is to find clusters in a dataset. As such, their notions of outliers are defined indirectly through the notion of clusters, and they are developed only to optimize clustering, but not to optimize outlier detection.

最后，一些聚类算法，如 CLARANS [NH94]、DBSCAN [EKSX96] 和 BIRCH [ZRL96]，被开发出来并具备异常处理能力。然而，它们的主要目标是在数据集中找到聚类。因此，它们对离群值的定义是通过聚类的概念间接定义的，并且它们的开发只是为了优化聚类，而不是优化离群值检测。

### 1.2 Distance-Based Outliers and Contribu- tions of this Paper

### 1.2 基于距离的离群值及本文的贡献

The notion of outliers studied here is defined as follows:

本文研究的离群值概念定义如下：

An object $O$ in a dataset $T$ is a ${DB}\left( {p,D}\right)$ -outlier if at least fraction $p$ of the objects in $T$ lies greater than distance $D$ from $O$ .

如果数据集中 $T$ 至少有 $p$ 比例的对象与对象 $O$ 的距离大于 $D$，则对象 $O$ 是一个 ${DB}\left( {p,D}\right)$ -离群值。

We use the term ${DB}\left( {p,D}\right)$ -outlier as shorthand notation for a $D$ istance- $B$ ased outlier (detected using parameters $p$ and $D$ ). This intuitive notion of outliers is consistent with Hawkins' definition. It is suitable for situations where the observed distribution does not fit any standard distribution. More importantly, it is well-defined for $k$ -dimensional datasets for any value of $k$ . Unlike the depth-based methods, ${DB}$ -outliers are not restricted computationally to small values of $k$ . While depth-based methods rely on the computation of layers in the data space, ${DB}$ -outliers go beyond the data space and rely on the computation of distance values based on a metric distance function. ${}^{1}$

我们使用术语 ${DB}\left( {p,D}\right)$ -离群值作为基于 $D$ 距离（使用参数 $p$ 和 $D$ 检测）的离群值的简写。这种直观的离群值概念与霍金斯的定义一致。它适用于观测分布不适合任何标准分布的情况。更重要的是，对于任意 $k$ 值的 $k$ 维数据集，它都有明确的定义。与基于深度的方法不同，${DB}$ -离群值在计算上不受 $k$ 小值的限制。基于深度的方法依赖于数据空间中层的计算，而 ${DB}$ -离群值超越了数据空间，依赖于基于度量距离函数的距离值计算。${}^{1}$

We do not claim that ${DB}$ -outliers can replace all existing notions of outliers and can be used universally. Indeed, depth-based outliers would be more applicable than ${DB}$ -outliers to situations where no reasonable metric distance function can be used. However, for numerous applications that are not readily supported by existing methods, defining a distance function is not hard. Our work builds on the premise that knowledge discovery is best facilitated by keeping a human user involved. The choice of $p$ and $D$ ,and validity checking (i.e.,deciding whether each ${DB}\left( {p,D}\right)$ -outlier is a "real" outlier of any significance), is left to a human expert. ${}^{2}$

我们并不声称 ${DB}$ -离群值（${DB}$ -outliers）可以取代所有现有的离群值概念并能普遍适用。实际上，在无法使用合理的度量距离函数的情况下，基于深度的离群值比 ${DB}$ -离群值更适用。然而，对于许多现有方法难以支持的应用，定义一个距离函数并不困难。我们的工作基于这样一个前提：让人类用户参与其中最有助于知识发现。$p$ 和 $D$ 的选择以及有效性检查（即，确定每个 ${DB}\left( {p,D}\right)$ -离群值是否是具有任何意义的“真正”离群值）留给人类专家来完成。${}^{2}$

The specific parts and contributions of this paper are as follows:

本文的具体内容和贡献如下：

- We show that the notion of distance-based outliers generalizes the notions of outliers supported by statistical outlier tests for standard distributions. Because this material appears in our preliminary work [KN97], Section 2.1 only provides a brief summary. Algorithms, optimizations, and disk-residency were not the focus of our previous work, but are the focus here.

- 我们表明，基于距离的离群值概念推广了标准分布的统计离群值检验所支持的离群值概念。由于这部分内容出现在我们的前期工作 [KN97] 中，第 2.1 节仅提供简要总结。算法、优化和磁盘驻留不是我们之前工作的重点，但却是本文的重点。

- We present two simple algorithms in Section 3 having a complexity of $O\left( {k{N}^{2}}\right)$ ,where $k$ and $N$ are the dimensionality and size of the dataset, respectively. The detection of ${DB}$ -outliers,unlike the depth-based approaches, is computationally tractable for values of $k > 2$ .

- 我们在第 3 节中提出了两种复杂度为 $O\left( {k{N}^{2}}\right)$ 的简单算法，其中 $k$ 和 $N$ 分别是数据集的维度和大小。与基于深度的方法不同，对于 $k > 2$ 的值，${DB}$ -离群值的检测在计算上是可行的。

---

<!-- Footnote -->

${}^{1}$ Algorithms presented here assume that the distance function is (weighted) Euclidean.

${}^{1}$ 这里提出的算法假设距离函数是（加权）欧几里得距离。

${}^{2}$ The values of $p$ and $D$ do provide indications on how "strong" an identified outlier is. Many existing approaches for finding outliers, including the depth-based approaches, do not provide such indications.

${}^{2}$ $p$ 和 $D$ 的值确实能表明所识别出的离群值的“强度”如何。许多现有的寻找离群值的方法，包括基于深度的方法，都没有提供这样的指示。

<!-- Footnote -->

---

- We present a partitioning-based algorithm in Section 4 that,for a given dimensionality $k$ ,has a complexity of $O\left( N\right)$ . The algorithm,however,is exponential on $k$ . We show that,in some cases, this algorithm outperforms the two simple algorithms by at least an order of magnitude for $k \leq  4$ .

- 我们在第 4 节中提出了一种基于分区的算法，对于给定的维度 $k$，其复杂度为 $O\left( N\right)$。然而，该算法在 $k$ 上是指数级的。我们表明，在某些情况下，对于 $k \leq  4$，该算法的性能比两种简单算法至少高出一个数量级。

- We present a different version of the partitioning-based algorithm in Section 5 for large, disk-resident datasets. We show that the algorithm guarantees at most 3 passes over the dataset. Again, experimental results indicate that this algorithm is by far the best for $k \leq  4$ .

- 我们在第 5 节中针对大型磁盘驻留数据集提出了基于分区算法的另一个版本。我们表明，该算法保证对数据集最多进行 3 次遍历。同样，实验结果表明，对于 $k \leq  4$，该算法目前是最好的。

## 2 Justification for ${DB}$ -Outliers

## 2 ${DB}$ -离群值的合理性

In this section, we provide two justifications for finding ${DB}$ -outliers. The first is formal,and the second is empirical. In Section 2.1,we show how ${DB}$ -outliers generalize certain statistical outlier tests. In Section 2.2,we show a few sample runs of our ${DB}$ -outlier detection package using actual NHL data.

在本节中，我们为寻找 ${DB}$ -离群值提供两个合理性依据。第一个是理论上的，第二个是实证上的。在第 2.1 节中，我们展示了 ${DB}$ -离群值如何推广某些统计离群值检验。在第 2.2 节中，我们展示了使用实际的美国国家冰球联盟（NHL）数据运行我们的 ${DB}$ -离群值检测程序的几个示例。

### 2.1 Properties of ${DB}\left( {p,D}\right)$ -Outliers

### 2.1 ${DB}\left( {p,D}\right)$ -离群值的性质

Definition ${1DB}\left( {p,D}\right)$ unifies ${}^{3}$ or generalizes another definition Def for outliers, if there exist specific values ${p}_{0},{D}_{0}$ such that: object $O$ is an outlier according to $\operatorname{Def}$ iff $O$ is a ${DB}\left( {{p}_{0},{D}_{0}}\right)$ -outlier.

如果存在特定的值 ${p}_{0},{D}_{0}$ 使得：对象 $O$ 根据 $\operatorname{Def}$ 是离群值当且仅当 $O$ 是 ${DB}\left( {{p}_{0},{D}_{0}}\right)$ -离群值，则定义 ${1DB}\left( {p,D}\right)$ 统一了 ${}^{3}$ 或推广了另一个离群值定义 Def。

For a normal distribution, outliers can be considered to be observations that lie 3 or more standard deviations (i.e.,≥3σ) from the mean $\mu$ [FPP78].

对于正态分布，离群值可以被认为是距离均值 $\mu$ 3 个或更多标准差（即，≥3σ）的观测值 [FPP78]。

Definition 2 Define ${De}{f}_{\text{Normal }}$ as follows: $t$ is an outlier in a normal distribution with mean $\mu$ and standard deviation $\sigma$ iff $\left| \frac{t - \mu }{\sigma }\right|  \geq  3$ .

定义 2 定义 ${De}{f}_{\text{Normal }}$ 如下：在均值为 $\mu$ 且标准差为 $\sigma$ 的正态分布中，$t$ 是离群值当且仅当 $\left| \frac{t - \mu }{\sigma }\right|  \geq  3$。

Lemma 1 ${DB}\left( {p,D}\right)$ unifies ${\operatorname{Def}}_{\text{Normal }}$ with ${p}_{0} =$ 0.9988 and ${D}_{0} = {0.13\sigma }$ ,that is, $t$ is an outlier according to ${De}{f}_{\text{Normal }}$ iff $t$ is a ${DB}\left( {{0.9988},{0.13\sigma }}\right)$ -outlier.

引理1 ${DB}\left( {p,D}\right)$将${\operatorname{Def}}_{\text{Normal }}$与${p}_{0} =$以0.9988的程度进行统一，并且与${D}_{0} = {0.13\sigma }$统一，也就是说，根据${De}{f}_{\text{Normal }}$，$t$是离群值（outlier）当且仅当$t$是一个${DB}\left( {{0.9988},{0.13\sigma }}\right)$ - 离群值。

Proofs of the lemmas in this section have already been documented [KN97]. Note that if the value ${3\sigma }$ in ${\text{Def}}_{\text{Normal }}$ is changed to some other value,such as ${4\sigma }$ , the above lemma can easily be modified with the corresponding ${p}_{0}$ and ${D}_{0}$ to show that ${DB}\left( {p,D}\right)$ still unifies the new definition of ${De}{f}_{\text{Normal }}$ . The same general approach applies to a Student $t$ -distribution,which has fatter tails than a normal distribution. The principle of using a tail to identify outliers can also be applied to a Poisson distribution.

本节中引理的证明已在文献[KN97]中记录。请注意，如果将${\text{Def}}_{\text{Normal }}$中的值${3\sigma }$更改为其他值，例如${4\sigma }$，则可以轻松地用相应的${p}_{0}$和${D}_{0}$对上述引理进行修改，以表明${DB}\left( {p,D}\right)$仍然能统一${De}{f}_{\text{Normal }}$的新定义。同样的通用方法适用于学生$t$ - 分布，该分布的尾部比正态分布更厚。使用尾部来识别离群值的原则也可以应用于泊松分布。

Definition 3 Define Define we get follows: the enout lier in a Poisson distribution with parameter $\mu  = {3.0}$ iff $t \geq  8$ .

定义3 定义如下：在参数为$\mu  = {3.0}$的泊松分布中，当且仅当$t \geq  8$时，定义为离群值。

Lemma ${2DB}\left( {p,D}\right)$ unifies ${\operatorname{Def}}_{\text{Poisson }}$ with ${p}_{0} =$ 0.9892 and ${D}_{0} = 1$ .

引理 ${2DB}\left( {p,D}\right)$将${\operatorname{Def}}_{\text{Poisson }}$与${p}_{0} =$以0.9892的程度进行统一，并且与${D}_{0} = 1$统一。

Finally, for a class of regression models, we can define an outlier criterion ${De}{f}_{\text{Regression }}$ ,and show that ${DB}\left( {p,D}\right)$ unifies ${De}{f}_{\text{Regression }}\left\lbrack  \mathrm{{KN97}}\right\rbrack$ .

最后，对于一类回归模型，我们可以定义一个离群值准则${De}{f}_{\text{Regression }}$，并证明${DB}\left( {p,D}\right)$能统一${De}{f}_{\text{Regression }}\left\lbrack  \mathrm{{KN97}}\right\rbrack$。

### 2.2 Sample Runs Using NHL Statistics

### 2.2 使用国家冰球联盟（NHL）统计数据的样本运行

During in-lab experiments on historical NHL data, we have identified outliers among players having perhaps "ordinary looking" statistics which suddenly stand out as being non-ordinary when combined with other attributes. Portions of sample runs of these experiments are documented in Figure 1.

在对国家冰球联盟（NHL）历史数据进行的实验室实验中，我们发现了一些球员数据中的离群值，这些球员的数据单独看可能“平平无奇”，但与其他属性结合起来时就显得与众不同。这些实验的部分样本运行结果记录在图1中。

The first example shows that, in 1994, Wayne Gretzky and Sergei Fedorov were outliers when 3 attributes-points scored,plus-minus statistic, ${}^{4}$ and number of penalty minutes-were used. Fedorov was an outlier because his point and plus-minus figures were much higher than those of almost all other players. (As a reference note, the "average" NHL player has fewer than 20 points, a plus-minus statistic of 0 , and fewer than 100 penalty minutes.) Gretzky was an outlier because of his high point total and low plus-minus figure. In fact, we were surprised that Gretzky's plus-minus figure was so poor, especially since he was the highest scorer in the league that year, and since high scorers usually have positive plus-minus values (as confirmed by Fedorov in the same example). ${}^{5}$ Using the same 3 attributes for 1996 data (see the second example), we note that Vladimir Konstantinov had an astonishingly high plus-minus statistic (i.e., +60) despite having a rather mediocre point total.

第一个例子表明，在1994年，当使用三个属性——得分、正负值统计、${}^{4}$和犯规分钟数时，韦恩·格雷茨基（Wayne Gretzky）和谢尔盖·费多罗夫（Sergei Fedorov）是离群值。费多罗夫是离群值，因为他的得分和正负值数据远高于几乎所有其他球员。（作为参考，NHL“普通”球员得分少于20分，正负值统计为0，犯规分钟数少于100分钟。）格雷茨基是离群值，因为他的总得分很高，但正负值数据很低。事实上，我们很惊讶格雷茨基的正负值数据如此糟糕，特别是因为他是当年联盟的得分王，而且高分球员通常正负值为正（同一例子中的费多罗夫也证实了这一点）。${}^{5}$对1996年的数据使用同样的三个属性（见第二个例子），我们注意到弗拉基米尔·康斯坦丁诺夫（Vladimir Konstantinov）尽管总得分相当平庸，但正负值统计却高得惊人（即 +60）。

Our third example shows that Chris Osgood and Mario Lemieux were outliers when 3 attributes-games played, goals scored, and shooting percentage (i.e., goals scored, divided by shots taken)-were used. Few NHL players had shooting percentages much beyond ${20}\%$ ,but Osgood’s shooting percentage really stood out. Despite playing 50 games, he only took 1 shot, on which he scored. Osgood's outlier status is explained by the fact that he is a goalie, and that goalies have rarely scored in the history of the NHL. Lemieux was an outlier,not because he scored on ${20.4}\%$ of his shots, but because no other player in the 20% shooting range had anywhere near the same number of goals and games played.

我们的第三个例子表明，当使用三个属性——参赛场次、进球数和射门命中率（即进球数除以射门次数）时，克里斯·奥斯古德（Chris Osgood）和马里奥·勒米厄（Mario Lemieux）是离群值。很少有NHL球员的射门命中率远超过${20}\%$，但奥斯古德的射门命中率非常突出。尽管他参加了50场比赛，但只射门1次并命中。奥斯古德成为离群值是因为他是一名守门员，而在NHL历史上守门员很少进球。勒米厄是离群值，不是因为他的射门命中率达到${20.4}\%$，而是因为在射门命中率达到20%的球员中，没有其他球员的进球数和参赛场次与他接近。

Whereas the above 3 examples contain outliers that are extreme in some dimension, the 4th example is different. None of Alexander Mogilny's statistics was extreme in any of the 5 dimensions. The NHL range

上述三个例子中的离群值在某些维度上表现极端，而第四个例子则不同。亚历山大·莫吉尔尼（Alexander Mogilny）的各项数据在五个维度中都没有表现出极端情况。NHL数据范围

---

<!-- Footnote -->

${}^{4}$ The plus-minus statistic indicates how many more even-strength goals were scored by the player's team-as opposed to the opposition's team-when this particular player was on the ice. For example, a plus-minus statistic of +2 indicates that this player was on the ice for 2 more goals scored by his team than against his team.

${}^{4}$ 正负值统计数据表明，当这名特定球员在冰面上时，其所在球队（相对于对手球队）在均势情况下多进了多少球。例如，正负值统计数据为 +2 表示，当这名球员在冰面上时，他所在球队比对手球队多进了 2 个球。

${}^{5}$ The next highest scorer in negative double digits did not occur until the 23rd position overall. Perhaps Gretzky's plus-minus can be explained by the fact that he played for the Los Angeles Kings that year-a team not known for its strong defensive play.

${}^{5}$ 直到总排名第 23 位才出现下一个负两位数得分的球员。也许格雷茨基的正负值可以用他那年效力于洛杉矶国王队（Los Angeles Kings）这一事实来解释，该球队并不以强大的防守著称。

${}^{3}{DB}$ -outliers are called unified outliers in our preliminary work [KN97].

${}^{3}{DB}$ 离群值在我们的前期工作 [KN97] 中被称为统一离群值。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: FindOutliers nhl94.data p=0.998 D=29.6985 POINTS PLUSMINUS PENALTY_MINUTES 1) Name = VLAD KONSTANTINOV, POINTS = 34, PLUSMINUS = 60, PENALTY_MINUTES = 139 2) Name = MARIO LEMIEUX, GAMES_PLAYED = 70, GOALS = 69, SHOOTING_PERCENTAGE = 20.4 indOutliers nh196.normalized. 0to1 p=0.996 D=0.447214 GAMES_PLAYED POWER_PLAY_GOALS SHORTHANDED_GOALS = 5, GAME_WINNING_GOALS = 6, GAME_TIEING_GOALS = 3 SHORTHANDED_GOALS = 8, GAME_WINNING_GOALS = 8, GAME_TIEING_GOALS = 0 1) Name = WAYNE GRETZKY, POINTS = 130, PLUSMINUS = -25, PENALTY_MINUTES = 20 2) Name = SERGEI FEDOROV, POINTS = 120, PLUSMINUS = 48, PENALTY_MINUTES = 34 indOutliers nh196.data p=0.998 D=26.3044 POINTS PLUSMINUS PENALTY_MINUTES findOutliers nhl96.data p=0.997 D=5 GAMES_PLAYED GOALS SHOOTING_PERCENTAGE 1) Name = CHRIS OSGOOD, GAMES_PLAYED = 50, GOALS = 1, SHOOTING_PERCENTAGE = 100. SHORTHANDED_GOALS GAME_WINNING_GOALS GAME_TIEING_GOALS 1) Name = ALEXANDER MOGILNY, GAMES_PLAYED = 79, POWER_PLAY_GOALS = 10, 2 Name = MARIO LEMIEUX, GAMES_PLAYED = 70, POWER_PLAY_GOALS = 31, -->

<img src="https://cdn.noedgeai.com/0195c913-c64b-73be-a45d-3920f48f6845_3.jpg?x=280&y=193&w=1120&h=477&r=0"/>

<!-- Media -->

Figure 1: Sample Output Involving NHL Players' Statistics for each of the 5 attributes (with Mogilny's statistic in parentheses) was as follows: 1-84 games played (79), 0-31 power play goals (10), 0-8 shorthanded goals (5), 0-12 game winning goals (6), and 0-4 game tieing goals (3). In contrast, 3 of 5 statistics for Mario Lemieux were extreme.

图 1：涉及 NHL 球员 5 项属性统计数据的示例输出（括号内为莫吉尔尼的数据）如下：比赛场次 1 - 84 场（79 场）， power play 进球数 0 - 31 个（10 个）， shorthanded 进球数 0 - 8 个（5 个），制胜球数 0 - 12 个（6 个），扳平球数 0 - 4 个（3 个）。相比之下，马里奥·勒米厄（Mario Lemieux）的 5 项统计数据中有 3 项非常突出。

In all of these examples, a user chose suitable values for $p$ and $D$ to define the "strength" of the outliers requested. These values depend on the attributes being analyzed and each attribute's distribution. The quest for suitable values for $p$ and $D$ may involve trial and error and numerous iterations; therefore, in future work, we will use sampling techniques to estimate a suitable starting value for $D$ ,given some value of $p$ close to unity (e.g., $p = {0.999}$ ). We also plan on supporting user-defined distance functions (including statistical distance functions which account for variability among attributes [JW92]).

在所有这些示例中，用户为 $p$ 和 $D$ 选择了合适的值来定义所请求离群值的“强度”。这些值取决于所分析的属性以及每个属性的分布。为 $p$ 和 $D$ 寻找合适的值可能需要反复试验和多次迭代；因此，在未来的工作中，给定接近 1 的 $p$ 值（例如 $p = {0.999}$ ），我们将使用抽样技术来估计 $D$ 的合适起始值。我们还计划支持用户定义的距离函数（包括考虑属性间变异性的统计距离函数 [JW92]）。

## 3 Simple Algorithms for Finding All ${DB}\left( {p,D}\right)$ -Outliers

## 3 寻找所有 ${DB}\left( {p,D}\right)$ -离群值的简单算法

### 3.1 Index-Based Algorithms

### 3.1 基于索引的算法

Let $N$ be the number of objects in dataset $T$ ,and let $F$ be the underlying distance function that gives the distance between any pair of objects in $T$ . For an object $O$ ,the $D$ -neighbourhood of $O$ contains the set of objects $Q \in  T$ that are within distance $D$ of $O$ (i.e., $\{ Q \in  T \mid  F\left( {O,Q}\right)  \leq  D\}$ ). The fraction $p$ is the minimum fraction of objects in $T$ that must be outside the $D$ -neighbourhood of an outlier. For simplicity of discussion,let $M$ be the maximum number of objects within the $D$ -neighbourhood of an outlier,i.e., $M =$ $N\left( {1 - p}\right)$ .

设 $N$ 为数据集 $T$ 中的对象数量，设 $F$ 为给出 $T$ 中任意一对对象之间距离的底层距离函数。对于对象 $O$，$O$ 的 $D$ -邻域包含距离 $O$ 在 $D$ 范围内的对象集合 $Q \in  T$（即 $\{ Q \in  T \mid  F\left( {O,Q}\right)  \leq  D\}$ ）。分数 $p$ 是离群值的 $D$ -邻域之外必须包含的 $T$ 中对象的最小比例。为便于讨论，设 $M$ 为离群值的 $D$ -邻域内对象的最大数量，即 $M =$ $N\left( {1 - p}\right)$ 。

From the formulation above, it is obvious that given $p$ and $D$ ,the problem of finding all ${DB}\left( {p,D}\right)$ -outliers can be solved by answering a nearest neighbour or range query centred at each object $O$ . More specifically, based on a standard multidimensional indexing structure,we execute a range search with radius $D$ for each object $O$ . As soon as $\left( {M + 1}\right)$ neighbours are found in the $D$ -neighbourhood,the search stops,and $O$ is declared a non-outlier; otherwise, $O$ is an outlier.

从上述公式可以明显看出，给定 $p$ 和 $D$，寻找所有 ${DB}\left( {p,D}\right)$ -离群值的问题可以通过回答以每个对象 $O$ 为中心的最近邻或范围查询来解决。更具体地说，基于标准的多维索引结构，我们对每个对象 $O$ 执行半径为 $D$ 的范围搜索。一旦在 $D$ -邻域中找到 $\left( {M + 1}\right)$ 个邻居，搜索就停止，并且 $O$ 被判定为非离群值；否则，$O$ 是离群值。

Analyses of multidimensional indexing schemes [HKP97] reveal that, for variants of R-trees [Gut84] and $k$ -d trees [Ben75,Sam90],the lower bound complexity for a range search is $\Omega \left( {N}^{1 - 1/k}\right)$ ,where $k$ is the number of dimensions or attributes and $N$ is the number of data objects. As $k$ increases,a range search quickly reduces to $O\left( N\right)$ ,giving at best a constant time improvement reflecting sequential search. Thus, the above procedure for finding all ${DB}\left( {p,D}\right)$ -outliers has a worst case complexity of $O\left( {k{N}^{2}}\right)$ . Two points are worth noting:

对多维索引方案的分析 [HKP97] 表明，对于 R - 树 [Gut84] 和 $k$ -d 树 [Ben75,Sam90] 的变体，范围搜索的下界复杂度为 $\Omega \left( {N}^{1 - 1/k}\right)$，其中 $k$ 是维度或属性的数量，$N$ 是数据对象的数量。随着 $k$ 的增加，范围搜索很快退化为 $O\left( N\right)$，最多只能实现反映顺序搜索的常数时间改进。因此，上述寻找所有 ${DB}\left( {p,D}\right)$ -离群值的过程的最坏情况复杂度为 $O\left( {k{N}^{2}}\right)$ 。有两点值得注意：

- Compared to the depth-based approaches, which have a lower bound complexity of $\Omega \left( {N}^{\left\lceil  \frac{k}{2}\right\rceil  }\right) ,{DB}$ - outliers scale much better with dimensionality. The framework of ${DB}$ -outliers is applicable and computationally feasible for datasets that have many attributes,i.e., $k \geq  5$ . This is a significant improvement on the current state-of-the-art, where existing methods can only realistically deal with two attributes [RR96].

- 与基于深度的方法相比，基于深度的方法的复杂度有一个下界 $\Omega \left( {N}^{\left\lceil  \frac{k}{2}\right\rceil  }\right) ,{DB}$ ——离群值（outliers）在维度方面的扩展性要好得多。${DB}$ -离群值的框架适用于具有许多属性的数据集，即 $k \geq  5$ ，并且在计算上是可行的。这是对当前最先进技术的重大改进，目前现有的方法实际上只能处理两个属性 [RR96]。

- The above analysis only considers search time. When it comes to using an index-based algorithm, most often for the kinds of data mining applications under consideration, it is a very strong assumption that the right index exists. As will be shown in Section 6, the index building cost alone, even without counting the search cost, almost always renders the index-based algorithms uncompetitive.

- 上述分析仅考虑了搜索时间。当涉及到使用基于索引的算法时，对于所考虑的数据挖掘应用类型而言，通常假设存在合适的索引是一个非常强的假设。正如第6节将展示的那样，仅索引构建成本，即使不计算搜索成本，几乎总是使基于索引的算法缺乏竞争力。

### 3.2 A Nested-Loop Algorithm

### 3.2 嵌套循环算法

To avoid the cost of building an index for finding all ${DB}\left( {p,D}\right)$ -outliers,Algorithm NL shown in Figure 3.2 uses a block-oriented, nested-loop design. Assuming a total buffer size of $B\%$ of the dataset size,the algorithm divides the entire buffer space into two halves, called the first and second arrays. It reads the dataset into the arrays, and directly computes the distance between each pair of objects or tuples. ${}^{6}$ For each object $t$ in the first array,a count of its $D$ -neighbours is maintained. Counting stops for a particular tuple whenever the number of $D$ -neighbours exceeds $M$ .

为了避免为查找所有 ${DB}\left( {p,D}\right)$ -离群值而构建索引的成本，图3.2所示的算法NL采用了面向块的嵌套循环设计。假设数据集的总缓冲区大小为 $B\%$ ，该算法将整个缓冲区空间分为两半，分别称为第一数组和第二数组。它将数据集读入数组，并直接计算每对对象或元组之间的距离。 ${}^{6}$ 对于第一数组中的每个对象 $t$ ，维护其 $D$ -邻居的计数。当某个元组的 $D$ -邻居数量超过 $M$ 时，停止计数。

## Algorithm NL

## 算法NL

1. Fill the first array (of size $\frac{B}{2}\%$ of the dataset) with a block of tuples from $T$ .

1. 用来自 $T$ 的一组元组填充第一数组（大小为数据集的 $\frac{B}{2}\%$ ）。

2. For each tuple ${t}_{i}$ in the first array,do:

2. 对于第一数组中的每个元组 ${t}_{i}$ ，执行以下操作：

a. count ${t}_{i} \leftarrow  0$

a. 计数 ${t}_{i} \leftarrow  0$

b. For each tuple ${t}_{j}$ in the first array,if $\operatorname{dist}\left( {{t}_{i},{t}_{j}}\right)  \leq  D$ : Increment count ${t}_{i}$ by 1 . If ${\operatorname{count}}_{i} > M$ ,mark ${t}_{i}$ as a non-outlier and proceed to next ${t}_{i}$ .

b. 对于第一数组中的每个元组 ${t}_{j}$ ，如果 $\operatorname{dist}\left( {{t}_{i},{t}_{j}}\right)  \leq  D$ ：将计数 ${t}_{i}$ 加1。如果 ${\operatorname{count}}_{i} > M$ ，将 ${t}_{i}$ 标记为非离群值并处理下一个 ${t}_{i}$ 。

3. While blocks remain to be compared to the first array, do:

3. 当仍有块需要与第一数组进行比较时，执行以下操作：

a. Fill the second array with another block (but save a block which has never served as the first array, for last).

a. 用另一块填充第二数组（但保留一个从未作为第一数组的块，留到最后）。

b. For each unmarked tuple ${t}_{i}$ in the first array do:

b. 对于第一数组中每个未标记的元组 ${t}_{i}$ ，执行以下操作：

For each tuple ${t}_{j}$ in the second array,if $\operatorname{dist}\left( {t}_{i}\right.$ , $\left. {t}_{j}\right)  \leq  D$ :

对于第二数组中的每个元组 ${t}_{j}$ ，如果 $\operatorname{dist}\left( {t}_{i}\right.$ ， $\left. {t}_{j}\right)  \leq  D$ ：

Increment count ${t}_{i}$ by1. If ${\operatorname{count}}_{i} > M$ , mark ${t}_{i}$ as a non-outlier and proceed to next ${t}_{i}$ .

将计数 ${t}_{i}$ 加1。如果 ${\operatorname{count}}_{i} > M$ ，将 ${t}_{i}$ 标记为非离群值并处理下一个 ${t}_{i}$ 。

4. For each unmarked tuple ${t}_{i}$ in the first array,report ${t}_{i}$ as an outlier.

4. 对于第一数组中每个未标记的元组 ${t}_{i}$ ，将 ${t}_{i}$ 报告为离群值。

5. If the second array has served as the first array anytime before, stop; otherwise, swap the names of the first and second arrays and goto step 2 .

5. 如果第二数组之前曾作为第一数组使用过，则停止；否则，交换第一数组和第二数组的名称并转到步骤2。

Figure 2: Pseudo-Code for Algorithm NL

图2：算法NL的伪代码

As an example, consider Algorithm NL with 50% buffering, and denote the 4 logical blocks of the dataset by $A,B,C,D$ ,with each block/array containing $\frac{1}{4}$ of the dataset. Let us follow the algorithm, filling the arrays in the following order, and comparing:

作为一个示例，考虑具有50%缓冲的算法NL，并将数据集的4个逻辑块表示为$A,B,C,D$，每个块/数组包含数据集的$\frac{1}{4}$。让我们按照该算法，按以下顺序填充数组并进行比较：

1. $A$ with $A$ ,then with $B,C,D$ -for a total of 4 block reads;

1. $A$与$A$比较，然后与$B,C,D$比较——总共进行4次块读取；

2. $D$ with $D$ (no read required),then with $A$ (no read), $B,C$ -for a total of 2 block reads;

2. $D$与$D$比较（无需读取），然后与$A$比较（无需读取），与$B,C$比较——总共进行2次块读取；

3. $C$ with $C$ ,then with $D,A,B$ -for a total of 2 blocks reads; and

3. $C$与$C$比较，然后与$D,A,B$比较——总共进行2次块读取；以及

4. $B$ with $B$ ,then with $C,A,D$ -for a total of 2 block reads.

4. $B$与$B$比较，然后与$C,A,D$比较——总共进行2次块读取。

Thus, in this example, a grand total of 10 blocks are read,amounting to $\frac{10}{4} = {2.5}$ passes over the entire dataset. Later, in Section 5.3, we compute the number of passes required in the general case.

因此，在这个示例中，总共读取了10个块，相当于对整个数据集进行了$\frac{10}{4} = {2.5}$次遍历。稍后，在5.3节中，我们将计算一般情况下所需的遍历次数。

Algorithm NL avoids the explicit construction of any indexing structure,and its complexity is $O\left( {k{N}^{2}}\right)$ . Compared to a tuple-by-tuple brute force algorithm that pays no attention to I/O's, Algorithm NL is superior because it tries to minimize I/O's.

算法NL避免了显式构建任何索引结构，其复杂度为$O\left( {k{N}^{2}}\right)$。与不考虑I/O的逐元组暴力算法相比，算法NL更优，因为它试图最小化I/O操作。

In the following two sections, we present two versions of a cell-based algorithm that has a complexity linear with respect to $N$ ,but exponential with respect to $k$ . This algorithm is therefore intended only for small values of $k$ . The key idea is to gain efficiency by using cell-by-cell processing instead of tuple-by-tuple processing,thereby avoiding the ${N}^{2}$ complexity term.

在接下来的两节中，我们将介绍一种基于单元格的算法的两个版本，该算法相对于$N$具有线性复杂度，但相对于$k$具有指数复杂度。因此，该算法仅适用于$k$的较小值。关键思想是通过使用逐单元格处理而不是逐元组处理来提高效率，从而避免${N}^{2}$复杂度项。

## 4 A Cell-Based Approach

## 4 基于单元格的方法

Let us begin with a simplified version of the algorithm, which assumes that both the multidimensional cell structure and the entire dataset fit into main memory. For ease of presentation, we begin with 2-D, and then proceed to $k$ -D. In Section 5,we give the full version of the algorithm for handling disk-resident datasets.

让我们从该算法的简化版本开始，该版本假设多维单元格结构和整个数据集都能装入主内存。为了便于表述，我们从二维情况开始，然后扩展到$k$维。在第5节中，我们将给出处理磁盘驻留数据集的完整算法版本。

### 4.1 Cell Structure and Properties in 2-D

### 4.1 二维中的单元格结构和属性

Suppose our data objects are 2-D. We quantize each of the data objects into a 2-D space that has been partitioned into cells or squares of length $l = \frac{D}{2\sqrt{2}}$ . Let ${C}_{x,y}$ denote the cell that is at the intersection of row $x$ and column $y$ . The Layer $1\left( {L}_{1}\right)$ neighbours of ${C}_{x,y}$ are the immediately neighbouring cells of ${C}_{x,y}$ ,defined in the usual sense, that is,

假设我们的数据对象是二维的。我们将每个数据对象量化到一个二维空间中，该空间已被划分为长度为$l = \frac{D}{2\sqrt{2}}$的单元格或正方形。用${C}_{x,y}$表示位于第$x$行和第$y$列交叉处的单元格。${C}_{x,y}$的第$1\left( {L}_{1}\right)$层邻居是${C}_{x,y}$的直接相邻单元格，按照通常的定义，即

$$
{L}_{1}\left( {C}_{x,y}\right)  = 
$$

$$
\left\{  {{C}_{u,v} \mid  u = x \pm  1,v = y \pm  1,{C}_{u,v} \neq  {C}_{x,y}}\right\}  . \tag{1}
$$

A typical cell (except for any cell on the boundary of the cell structure) has $8{L}_{1}$ neighbours.

一个典型的单元格（除了单元格结构边界上的任何单元格）有$8{L}_{1}$个邻居。

Property 1 Any pair of objects within the same cell is at most distance $\frac{D}{2}$ apart.

属性1：同一单元格内的任意一对对象之间的距离至多为$\frac{D}{2}$。

Property 2 If ${C}_{u,v}$ is an ${L}_{1}$ neighbour of ${C}_{x,y}$ ,then any object $P \in  {C}_{u,v}$ and any object $Q \in  {C}_{x,y}$ are at most distance $D$ apart.

属性2：如果${C}_{u,v}$是${C}_{x,y}$的${L}_{1}$邻居，那么任意对象$P \in  {C}_{u,v}$和任意对象$Q \in  {C}_{x,y}$之间的距离至多为$D$。

Property 1 is valid because the length of a cell's diagonal is $\sqrt{2}l = \sqrt{2}\frac{D}{2\sqrt{2}} = \frac{D}{2}$ . Property 2 is valid because the distance between any pair of objects in the two cells cannot exceed twice the length of a cell's diagonal. We will see that these two properties are useful in ruling out many objects as outlier candidates. The Layer $2\left( {L}_{2}\right)$ neighbours of ${C}_{x,y}$ are those additional cells within 3 cells of ${C}_{x,y}$ ,i.e.,

性质1成立是因为单元格对角线的长度为$\sqrt{2}l = \sqrt{2}\frac{D}{2\sqrt{2}} = \frac{D}{2}$。性质2成立是因为两个单元格中任意一对对象之间的距离不能超过单元格对角线长度的两倍。我们将看到，这两个性质有助于排除许多对象作为离群值候选。${C}_{x,y}$的第$2\left( {L}_{2}\right)$层邻域是指距离${C}_{x,y}$不超过3个单元格的其他单元格，即

$$
{L}_{2}\left( {C}_{x,y}\right)  = \left\{  {{C}_{u,v} \mid  u = x \pm  3,v = y \pm  3,}\right. 
$$

$$
\left. {{C}_{u,v} \notin  {L}_{1}\left( {C}_{x,y}\right) ,{C}_{u,v} \neq  {C}_{x,y}}\right\}  \text{.} \tag{2}
$$

A typical cell (except for any cell on or near a boundary) has ${7}^{2} - {3}^{2} = {40}{L}_{2}$ cells. Note that Layer 1 is 1 cell thick and Layer 2 is 2 cells thick. ${L}_{2}$ was chosen in this way to satisfy the following property.

一个典型的单元格（边界上或边界附近的单元格除外）有${7}^{2} - {3}^{2} = {40}{L}_{2}$个单元格。请注意，第1层的厚度为1个单元格，第2层的厚度为2个单元格。以这种方式选择${L}_{2}$是为了满足以下性质。

Property 3 If ${C}_{u,v} \neq  {C}_{x,y}$ is neither an ${L}_{1}$ nor an ${L}_{2}$ neighbour of ${C}_{x,y}$ ,then any object $P \in  {C}_{u,v}$ and any object $Q \in  {C}_{x,y}$ must be $>$ distance $D$ apart. Since the combined thickness of ${L}_{1}$ and ${L}_{2}$ is 3 cells, the distance between $P$ and $Q$ must exceed ${3l} = \frac{3D}{2\sqrt{2}} >$ $D$ .

性质3 如果${C}_{u,v} \neq  {C}_{x,y}$既不是${C}_{x,y}$的${L}_{1}$邻域也不是${L}_{2}$邻域，那么任何对象$P \in  {C}_{u,v}$和任何对象$Q \in  {C}_{x,y}$之间的距离必定为$>$ $D$。由于${L}_{1}$和${L}_{2}$的总厚度为3个单元格，因此$P$和$Q$之间的距离必定超过${3l} = \frac{3D}{2\sqrt{2}} >$ $D$。

---

<!-- Footnote -->

${}^{6}$ From here on in,we use the terms object and tuple interchangeably.

${}^{6}$ 从这里开始，我们可以互换使用“对象”和“元组”这两个术语。

<!-- Footnote -->

---

Property 4 (a) If there are $> M$ objects in ${C}_{x,y}$ , none of the objects in ${C}_{x,y}$ is an outlier.

性质4 (a) 如果在${C}_{x,y}$中有$> M$个对象，那么${C}_{x,y}$中的任何对象都不是离群点（outlier）。

(b) If there are $> M$ objects in ${C}_{x,y} \cup  {L}_{1}\left( {C}_{x,y}\right)$ ,none of the objects in ${C}_{x,y}$ is an outlier.

(b) 如果在${C}_{x,y} \cup  {L}_{1}\left( {C}_{x,y}\right)$中有$> M$个对象，那么${C}_{x,y}$中的任何对象都不是离群点。

(c) If there are $\leq  M$ objects in ${C}_{x,y} \cup  {L}_{1}\left( {C}_{x,y}\right)  \cup$ ${L}_{2}\left( {C}_{x,y}\right)$ ,every object in ${C}_{x,y}$ is an outlier.

(c) 如果在${C}_{x,y} \cup  {L}_{1}\left( {C}_{x,y}\right)  \cup$ ${L}_{2}\left( {C}_{x,y}\right)$中有$\leq  M$个对象，那么${C}_{x,y}$中的每个对象都是离群点。

Properties 4(a) and 4(b) are direct consequences of Properties 1 and 2, and 4(c) is due to Property 3.

性质4(a)和4(b)是性质1和性质2的直接推论，而性质4(c)则归因于性质3。

### 4.2 Algorithm FindAllOutsM for Memory- Resident Datasets

### 4.2 用于内存驻留数据集的FindAllOutsM算法

Figure 3 presents Algorithm FindAllOutsM to detect all ${DB}\left( {p,D}\right)$ -outliers in memory-resident datasets. Later, in Section 5.2, we present an enhanced algorithm to handle disk-resident datasets.

图3展示了用于检测内存驻留数据集中所有${DB}\left( {p,D}\right)$ - 离群点的FindAllOutsM算法。稍后，在5.2节中，我们将介绍一种用于处理磁盘驻留数据集的增强算法。

## Algorithm FindAllOutsM

## FindAllOutsM算法

1. For $q \leftarrow  1,2,\ldots m,\;{\text{Count}}_{q} \leftarrow  0$

1. 对于$q \leftarrow  1,2,\ldots m,\;{\text{Count}}_{q} \leftarrow  0$

2. For each object $P$ ,map $P$ to an appropriate cell ${C}_{q}$ ,store $P$ ,and increment ${Coun}{t}_{q}$ by 1 .

2. 对于每个对象$P$，将$P$映射到一个合适的单元格${C}_{q}$，存储$P$，并将${Coun}{t}_{q}$加1。

3. For $q \leftarrow  1,2,\ldots ,m$ ,if ${\operatorname{Count}}_{q} > M$ ,label ${C}_{q}$ red.

3. 对于$q \leftarrow  1,2,\ldots ,m$，如果${\operatorname{Count}}_{q} > M$，则将${C}_{q}$标记为红色。

4. For each red cell ${C}_{r}$ ,label each of the ${L}_{1}$ neighbours of ${C}_{r}$ pink, provided the neighbour has not already been labelled red.

4. 对于每个红色单元格${C}_{r}$，将${C}_{r}$的每个${L}_{1}$邻域单元格标记为粉色，前提是该邻域单元格尚未被标记为红色。

5. For each non-empty white (i.e.,uncoloured) cell ${C}_{w}$ ,do:

5. 对于每个非空的白色（即未着色）单元格${C}_{w}$，执行以下操作：

a. ${\operatorname{Count}}_{w2} \leftarrow  {\operatorname{Count}}_{w} + \mathop{\sum }\limits_{{i \in  {L}_{1}\left( {C}_{w}\right) }}{\operatorname{Count}}_{i}$

b. If ${\operatorname{Count}}_{w2} > M$ ,label ${C}_{w}$ pink.

b. 如果${\operatorname{Count}}_{w2} > M$，则将${C}_{w}$标记为粉色。

c. else

c. 否则

1. Coun ${t}_{w3} \leftarrow$ Coun ${t}_{w2} + \mathop{\sum }\limits_{{i \in  {L}_{2}\left( {C}_{w}\right) }}$ Coun ${t}_{i}$

1. 计数${t}_{w3} \leftarrow$ 计数${t}_{w2} + \mathop{\sum }\limits_{{i \in  {L}_{2}\left( {C}_{w}\right) }}$ 计数${t}_{i}$

2. If ${\text{Count}}_{w3} \leq  M$ ,mark all objects in ${C}_{w}$ as outliers.

2. 如果${\text{Count}}_{w3} \leq  M$，则将${C}_{w}$中的所有对象标记为离群点。

3. else for each object $P \in  {C}_{w}$ ,do:

3. 否则，对于每个对象 $P \in  {C}_{w}$，执行以下操作：

i. Count ${t}_{P} \leftarrow$ Count ${}_{w2}$

i. 计数 ${t}_{P} \leftarrow$ 计数 ${}_{w2}$

ii. For each object $Q \in  {L}_{2}\left( {C}_{w}\right)$ ,if $\operatorname{dist}\left( {P,Q}\right)  \leq  D$ :

ii. 对于每个对象 $Q \in  {L}_{2}\left( {C}_{w}\right)$，如果 $\operatorname{dist}\left( {P,Q}\right)  \leq  D$：

Increment ${\text{Count}}_{P}$ by 1 . If ${\text{Count}}_{P} >$ $M,P$ cannot be an outlier,so goto $5\left( \mathrm{c}\right) \left( 3\right)$ .

将 ${\text{Count}}_{P}$ 加 1。如果 ${\text{Count}}_{P} >$ $M,P$ 不可能是离群值，因此跳转到 $5\left( \mathrm{c}\right) \left( 3\right)$。

iii. Mark $P$ as an outlier.

iii. 将 $P$ 标记为离群值。

## Figure 3: Pseudo-Code for Algorithm FindAllOutsM

## 图 3：算法 FindAllOutsM 的伪代码

Step 2 of Algorithm FindAllOutsM quantizes each tuple to its appropriate cell. Step 3 labels all cells containing $> M$ tuples,red. This corresponds to Property 4(a). Cells that are ${L}_{1}$ neighbours of a red cell are labelled pink in step 4 , and cannot contain outliers because of Property 4(b). Other cells satisfying Property $4\left( \mathrm{\;b}\right)$ are labelled pink in step $5\left( \mathrm{\;b}\right)$ . Finally,in step $5\left( c\right) \left( 2\right)$ of the algorithm,cells satisfying Property 4(c) are identified.

算法 FindAllOutsM 的步骤 2 将每个元组量化到其合适的单元格中。步骤 3 将所有包含 $> M$ 个元组的单元格标记为红色。这对应于属性 4(a)。在步骤 4 中，作为红色单元格 ${L}_{1}$ 邻域的单元格被标记为粉色，并且由于属性 4(b)，这些单元格不可能包含离群值。在步骤 $5\left( \mathrm{\;b}\right)$ 中，满足属性 $4\left( \mathrm{\;b}\right)$ 的其他单元格被标记为粉色。最后，在算法的步骤 $5\left( c\right) \left( 2\right)$ 中，识别出满足属性 4(c) 的单元格。

All of the properties mentioned in Section 4.1 are used to help determine outliers and non-outliers on a cell-by-cell basis, rather than on an object-by-object basis. This helps to reduce execution time significantly because we quickly rule out a large number of objects that cannot be outliers. For cells not satisfying any of Properties 4(a)-(c), we resort to object-by-object processing. Such cells are denoted as white cells $\left( {C}_{w}\right)$ . In step 5(c)(3) of Algorithm FindAllOutsM, each object $P \in  {C}_{w}$ must be compared with every object $Q$ lying in a cell that is an ${L}_{2}$ neighbour of ${C}_{w}$ in order to determine how many $Q$ ’s are inside the $D$ -neighbourhood of $P$ . As soon as the number of $D$ -neighbours exceeds $M,P$ is declared a non-outlier. If,after examining all $Q$ ’s,the count remains $\leq  M$ ,then $P$ is an outlier.

第 4.1 节中提到的所有属性都用于逐个单元格地帮助确定离群值和非离群值，而不是逐个对象地进行。这有助于显著减少执行时间，因为我们可以快速排除大量不可能是离群值的对象。对于不满足属性 4(a) - (c) 中任何一个的单元格，我们采用逐个对象的处理方式。此类单元格表示为白色单元格 $\left( {C}_{w}\right)$。在算法 FindAllOutsM 的步骤 5(c)(3) 中，为了确定有多少个 $Q$ 在 $P$ 的 $D$ 邻域内，每个对象 $P \in  {C}_{w}$ 必须与位于 ${C}_{w}$ 的 ${L}_{2}$ 邻域单元格中的每个对象 $Q$ 进行比较。一旦 $D$ 邻域的数量超过 $M,P$，则 $P$ 被声明为非离群值。如果在检查完所有 $Q$ 后，计数仍为 $\leq  M$，则 $P$ 是离群值。

### 4.3 Complexity Analysis: The 2-D Case

### 4.3 复杂度分析：二维情况

Let us analyze the complexity of Algorithm FindAll-OutsM for the 2-D case. Step 1 takes $O\left( m\right)$ time, where $m \ll  N$ is the total number of cells. Steps 2 and 3 take $O\left( N\right)$ and $O\left( m\right)$ time respectively. Since $M + 1$ is the minimum number of objects that can appear in a red cell,there are at most $\frac{N}{M + 1}$ red cells. Thus,step 4 takes $O\left( \frac{N}{M + 1}\right)$ time. The time complexity of step 5 is the most complicated. In the worst case, (i) no cell is labelled red or pink in the previous steps, and (ii) step 5(c) is necessary for all cells. If no cell is coloured, then each cell contains at most $M$ objects. Thus,in step $5\left( \mathrm{c}\right)$ ,each of the objects in a cell can require the checking of up to $M$ objects in each of the ${40}{L}_{2}$ neighbours of the cell; therefore, $O\left( {{40}{M}^{2}}\right)$ time is required for each cell. Hence,step 5 takes $O\left( {m{M}^{2}}\right)$ time. Since, by definition, $M = N\left( {1 - p}\right)$ ,we equate $O\left( {m{M}^{2}}\right)$ to $O\left( {m{N}^{2}{\left( 1 - p\right) }^{2}}\right)$ . In practice,we expect $p$ to be extremely close to 1 , especially for large datasets, so $O\left( {m{N}^{2}{\left( 1 - p\right) }^{2}}\right)$ can be approximated by $O\left( m\right)$ . Thus, the time complexity of Algorithm FindAllOutsM in 2- D is $O\left( {m + N}\right)$ . Note that this complexity figure is very conservative because, in practice, we expect many red and pink cells. As soon as this happens, there are fewer object-to-object comparisons. Thus, step 5(c) becomes less dominant, and the algorithm requires less time. In Section 6.2, we show experimental results on the efficiency of Algorithm FindAllOutsM.

让我们分析二维情况下算法FindAll - OutsM的复杂度。步骤1需要$O\left( m\right)$的时间，其中$m \ll  N$是单元格的总数。步骤2和步骤3分别需要$O\left( N\right)$和$O\left( m\right)$的时间。由于$M + 1$是红色单元格中可能出现的对象的最小数量，因此最多有$\frac{N}{M + 1}$个红色单元格。因此，步骤4需要$O\left( \frac{N}{M + 1}\right)$的时间。步骤5的时间复杂度最为复杂。在最坏的情况下，（i）在前面的步骤中没有单元格被标记为红色或粉色，并且（ii）步骤5(c)对所有单元格都是必要的。如果没有单元格被着色，那么每个单元格最多包含$M$个对象。因此，在步骤$5\left( \mathrm{c}\right)$中，一个单元格中的每个对象可能需要检查该单元格的${40}{L}_{2}$个相邻单元格中每个单元格里的多达$M$个对象；因此，每个单元格需要$O\left( {{40}{M}^{2}}\right)$的时间。因此，步骤5需要$O\left( {m{M}^{2}}\right)$的时间。根据定义，由于$M = N\left( {1 - p}\right)$，我们将$O\left( {m{M}^{2}}\right)$等同于$O\left( {m{N}^{2}{\left( 1 - p\right) }^{2}}\right)$。在实际应用中，我们预计$p$非常接近1，特别是对于大型数据集，因此$O\left( {m{N}^{2}{\left( 1 - p\right) }^{2}}\right)$可以近似为$O\left( m\right)$。因此，二维情况下算法FindAllOutsM的时间复杂度为$O\left( {m + N}\right)$。请注意，这个复杂度数字非常保守，因为在实际应用中，我们预计会有许多红色和粉色单元格。一旦出现这种情况，对象与对象之间的比较就会减少。因此，步骤5(c)的影响变小，算法所需的时间也会减少。在6.2节中，我们展示了算法FindAllOutsM效率的实验结果。

### 4.4 Generalization to Higher Dimensions

### 4.4 向更高维度的推广

When moving from $k = 2$ dimensions to $k > 2$ ,Algorithm FindAllOutsM requires only one change to incorporate a general $k$ -D cell structure. That change involves the cell length $l$ . Recall that in 2-D, $l = \frac{D}{2\sqrt{2}}$ . Since the length of a diagonal of a $k$ -D hypercube/cell of length $l$ is $\sqrt{k}l$ ,the length $l$ in a $k$ -D setting must be changed to $\frac{D}{2\sqrt{k}}$ to preserve Properties 1 and 2 .

当从$k = 2$维转换到$k > 2$维时，算法FindAllOutsM只需进行一处更改即可纳入一般的$k$维单元格结构。这一更改涉及单元格长度$l$。回顾一下，在二维情况下，$l = \frac{D}{2\sqrt{2}}$。由于长度为$l$的$k$维超立方体/单元格的对角线长度为$\sqrt{k}l$，为了保留属性1和属性2，在$k$维环境中长度$l$必须更改为$\frac{D}{2\sqrt{k}}$。

Although the following comments do not change Algorithm FindAllOutsM, an understanding of which cells appear in Layer 2 is important in correctly applying the algorithm. First,we note that the ${L}_{1}$ neighbours of cell ${C}_{{x}_{1},\ldots ,{x}_{k}}$ are:

尽管以下注释不会改变算法FindAllOutsM，但理解哪些单元格出现在第2层对于正确应用该算法很重要。首先，我们注意到单元格${C}_{{x}_{1},\ldots ,{x}_{k}}$的${L}_{1}$个相邻单元格为：

$$
{L}_{1}\left( {C}_{{x}_{1},\ldots ,{x}_{k}}\right)  = \left\{  {{C}_{{u}_{1},\ldots ,{u}_{k}} \mid  {u}_{i} = {x}_{i} \pm  1}\right. 
$$

$$
\forall 1 \leq  i \leq  k,{C}_{{u}_{1},\ldots ,{u}_{k}} \neq  {C}_{{x}_{1},\ldots ,{x}_{k}}\}  \tag{3}
$$

which generalizes the definition given in (1). However, to preserve Property 3,the definition of ${L}_{2}$ neighbours needs to be modified. Specifically,since $l = \frac{D}{2\sqrt{k}}$ ,Layer 2 needs to be thicker than it is for $k = 2$ . Let $x$ denote the thickness of Layer 2. Then, the combined thickness of Layers 1 and 2 is $x + 1$ . So,for Property 3 to hold, we require that $\left( {x + 1}\right) l > D$ ; consequently,we pick $x$ as $\left\lbrack  {2\sqrt{k} - 1}\right\rbrack$ . The ${L}_{2}$ neighbours of ${C}_{{x}_{1},\ldots ,{x}_{k}}$ are therefore:

这推广了(1)中给出的定义。然而，为了保留属性3，${L}_{2}$个相邻单元格的定义需要修改。具体来说，由于$l = \frac{D}{2\sqrt{k}}$，第2层需要比$k = 2$维时更厚。设$x$表示第2层的厚度。那么，第1层和第2层的总厚度为$x + 1$。因此，为了使属性3成立，我们要求$\left( {x + 1}\right) l > D$；因此，我们将$x$选取为$\left\lbrack  {2\sqrt{k} - 1}\right\rbrack$。因此，${C}_{{x}_{1},\ldots ,{x}_{k}}$的${L}_{2}$个相邻单元格为：

$$
{L}_{2}\left( {C}_{{x}_{1},\ldots ,{x}_{k}}\right)  = \left\{  {{C}_{{u}_{1},\ldots ,{u}_{k}} \mid  {u}_{i} = {x}_{i} \pm  \left\lceil  {2\sqrt{k}}\right\rceil  }\right. 
$$

$$
\forall 1 \leq  i \leq  k,\;{C}_{{u}_{1},\ldots ,{u}_{k}} \notin  {L}_{1}\left( {C}_{{x}_{1},\ldots ,{x}_{k}}\right) ,
$$

$$
\left. {{C}_{{u}_{1},\ldots ,{u}_{k}} \neq  {C}_{{x}_{1},\ldots ,{x}_{k}}}\right\}   \tag{4}
$$

which generalizes (2). In this way, Properties 1 to 4 listed in Section 4.1 are preserved.

这推广了式(2)。通过这种方式，第4.1节中列出的性质1至性质4得以保留。

### 4.5 Complexity Analysis: The Case for Higher Dimensions

### 4.5 复杂度分析：高维情形

For $k > 2$ ,the complexities of steps 1 to 4 in Algorithm FindAllOutsM remain the same. However, we note that $m$ is exponential with respect to $k$ ,and may not necessarily be much less than $N$ . Also,the complexity of step 5 is no longer $O\left( m\right)$ ,but $O\left( {m{\left( 2\lceil 2\sqrt{k}\rceil  + 1\right) }^{k}}\right)  \approx$ $O\left( {c}^{k}\right)$ ,where $c$ is some constant depending on $\sqrt{k}$ and on ${m}^{1/k}$ (which roughly corresponds to the number of cells along each dimension). Thus, the complexity of the entire algorithm is $O\left( {{c}^{k} + N}\right)$ .

对于$k > 2$ ，算法FindAllOutsM中步骤1至步骤4的复杂度保持不变。然而，我们注意到$m$ 相对于$k$ 是指数级的，并且不一定比$N$ 小很多。此外，步骤5的复杂度不再是$O\left( m\right)$ ，而是$O\left( {m{\left( 2\lceil 2\sqrt{k}\rceil  + 1\right) }^{k}}\right)  \approx$ $O\left( {c}^{k}\right)$ ，其中$c$ 是某个取决于$\sqrt{k}$ 和${m}^{1/k}$ 的常数（${m}^{1/k}$ 大致对应于每个维度上的单元格数量）。因此，整个算法的复杂度为$O\left( {{c}^{k} + N}\right)$ 。

While this complexity figure represents a worst case scenario, the question to ask is how efficient Algorithm FindAllOutsM is in practice for the general $k$ -D case. We defer the answer to Section 6.4, but make the following preliminary comments. First, for the identification of "strong" outliers, the number of outliers to be found is intended to be small. This is achieved by having a relatively large value of $D$ ,and a value of $p$ very close to unity. A large value of $D$ corresponds to a small number of cells along each dimension. Thus, the constant $c$ is small,but it is $> 1$ . Second,for values of $p$ very close to unity, $M$ is small,implying that there will be numerous red and pink cells. This means that the savings realized by skipping red and pink cells is enormous, and that the number of objects requiring pairwise comparisons is relatively small.

虽然这个复杂度数字代表了最坏情况下的场景，但要问的问题是，对于一般的$k$ 维情形，算法FindAllOutsM在实际应用中的效率如何。我们将答案留到第6.4节讨论，但先给出以下初步评论。首先，对于识别“强”离群点，要找到的离群点数量通常较少。这可以通过让$D$ 取相对较大的值，以及让$p$ 的值非常接近1来实现。$D$ 取较大的值对应于每个维度上的单元格数量较少。因此，常数$c$ 较小，但它是$> 1$ 。其次，当$p$ 的值非常接近1时，$M$ 较小，这意味着会有大量的红色和粉色单元格。这意味着跳过红色和粉色单元格所节省的计算量非常大，并且需要进行两两比较的对象数量相对较少。

## 5 DB-Outliers in Large, Disk-Resident Datasets

## 5 大型磁盘驻留数据集中的DB离群点

In the last section, we presented a cell-based algorithm that was simplified for memory-resident datasets. Here, we extend the simplified version to handle disk-resident datasets. This new version preserves the property of being linear with respect to $N$ . It also provides the guarantee that no more than 3 , if not 2, passes over the dataset are required.

在上一节中，我们提出了一种针对内存驻留数据集简化的基于单元格的算法。在这里，我们将简化版本扩展以处理磁盘驻留数据集。这个新版本保留了相对于$N$ 呈线性的性质。它还保证对数据集的遍历次数不超过3次，如果不是2次的话。

### 5.1 Handling Large, Disk-Resident Datasets: An Example

### 5.1 处理大型磁盘驻留数据集：一个示例

In handling a large, disk-resident dataset, the goal is to minimize the number of page reads or passes over the dataset. In the cell-based algorithm, there are two places where page reads are needed:

在处理大型磁盘驻留数据集时，目标是最小化页面读取次数或对数据集的遍历次数。在基于单元格的算法中，有两个地方需要进行页面读取：

- the initial mapping phase

- 初始映射阶段

In step 2 of Algorithm FindAllOutsM, each object is mapped to an appropriate cell. This unavoidable step requires one pass over the dataset.

在算法FindAllOutsM的步骤2中，每个对象都被映射到一个合适的单元格。这个不可避免的步骤需要对数据集进行一次遍历。

## the object-pairwise phase

## 对象两两比较阶段

In step $5\left( \mathrm{c}\right) \left( 3\right)$ ,for each object $P$ in a white cell ${C}_{w}$ ,each object $Q$ in a cell that is an ${L}_{2}$ neighbour of ${C}_{w}$ needs to be read to perform the object-by-object distance calculation. Since objects mapped to the same cell, or to nearby cells, are not necessarily physically clustered on disk, each pair of objects(P,Q)may require a page to be read,thereby causing a large number of I/O's. The point here is that if object-by-object distance calculations are to be done exactly as described in step $5\left( \mathrm{c}\right) \left( 3\right)$ , then a page may be read many times.

在步骤$5\left( \mathrm{c}\right) \left( 3\right)$ 中，对于白色单元格${C}_{w}$ 中的每个对象$P$ ，需要读取${C}_{w}$ 的${L}_{2}$ 邻域单元格中的每个对象$Q$ ，以进行对象间的距离计算。由于映射到同一单元格或相邻单元格的对象不一定在磁盘上物理聚集，每对对象(P,Q)可能需要读取一个页面，从而导致大量的I/O操作。这里的关键是，如果要按照步骤$5\left( \mathrm{c}\right) \left( 3\right)$ 中所描述的方式精确地进行对象间的距离计算，那么一个页面可能会被多次读取。

The above scenario is an extreme case whereby no tuples/pages are stored in main memory. A natural question to ask is if, in the object-pairwise phase, it is possible to read each page only once. Let $\operatorname{Page}\left( C\right)$ denote the set of pages that store at least one tuple mapped to cell $C$ . Then,for a particular white cell ${C}_{w}$ ,we need to read the pages in $\operatorname{Page}\left( {C}_{w}\right)$ . Because we also need the tuples mapped to a cell ${C}_{v}$ that is an ${L}_{2}$ neighbour of ${C}_{w}$ (i.e., ${C}_{v} \in  {L}_{2}\left( {C}_{w}\right)$ ),we need the pages in $\operatorname{Page}\left( {{L}_{2}\left( {C}_{w}\right) }\right)  = \mathop{\bigcup }\limits_{{{C}_{v} \in  {L}_{2}\left( {C}_{w}\right) }}\operatorname{Page}\left( {C}_{v}\right)$ . Also,if we want to ensure that pages in $\operatorname{Page}\left( {{L}_{2}\left( {C}_{w}\right) }\right)$ are only read once, we need to read the pages: (i) that are needed by ${C}_{v}$ ,because ${C}_{v}$ itself may be a white cell,and (ii) that use ${C}_{v}$ ,because ${C}_{v}$ may be an ${L}_{2}$ neighbour of yet another white cell. In general, the "transitive closure" of this page cascading may include every page in the dataset. Hence, the only way to ensure that a page is read at most once in the object-pairwise phase is to use a buffer the size of the dataset, which is clearly a strong assumption for large datasets.

上述场景是一种极端情况，即主存中未存储任何元组/页面。一个自然会提出的问题是，在对象逐对阶段，是否有可能仅读取每个页面一次。令 $\operatorname{Page}\left( C\right)$ 表示存储至少一个映射到单元格 $C$ 的元组的页面集合。那么，对于特定的白色单元格 ${C}_{w}$，我们需要读取 $\operatorname{Page}\left( {C}_{w}\right)$ 中的页面。因为我们还需要映射到单元格 ${C}_{v}$ 的元组，该单元格是 ${C}_{w}$ 的 ${L}_{2}$ 邻接单元格（即 ${C}_{v} \in  {L}_{2}\left( {C}_{w}\right)$），所以我们需要 $\operatorname{Page}\left( {{L}_{2}\left( {C}_{w}\right) }\right)  = \mathop{\bigcup }\limits_{{{C}_{v} \in  {L}_{2}\left( {C}_{w}\right) }}\operatorname{Page}\left( {C}_{v}\right)$ 中的页面。此外，如果我们想确保 $\operatorname{Page}\left( {{L}_{2}\left( {C}_{w}\right) }\right)$ 中的页面仅被读取一次，我们需要读取以下页面：(i) ${C}_{v}$ 所需的页面，因为 ${C}_{v}$ 本身可能是一个白色单元格；(ii) 使用 ${C}_{v}$ 的页面，因为 ${C}_{v}$ 可能是另一个白色单元格的 ${L}_{2}$ 邻接单元格。一般来说，这种页面级联的“传递闭包”可能会包含数据集中的每个页面。因此，要确保在对象逐对阶段每个页面最多被读取一次，唯一的方法是使用一个大小与数据集相同的缓冲区，对于大型数据集而言，这显然是一个很强的假设。

Our approach is a "middle ground" scenario whereby only a selected subset of tuples is kept in main memory. This subset is the set of all tuples mapped to white cells. Hereinafter, we refer to such tuples as white tuples. This is our choice partly because these are the very tuples that need object-by-object calculations, and partly because the number of tuples in a white cell,by definition,is bounded above by $M$ . Furthermore, we classify all pages into three categories:

我们的方法是一种“折衷”场景，即仅将选定的元组子集保存在主存中。这个子集是所有映射到白色单元格的元组的集合。此后，我们将这些元组称为白色元组。我们做出这样的选择，部分原因是这些正是需要进行逐个对象计算的元组，部分原因是根据定义，白色单元格中的元组数量上限为 $M$。此外，我们将所有页面分为三类：

A. Pages that contain some white tuple(s)

A. 包含一些白色元组的页面

B. Pages that do not contain any white tuple, but contain tuple(s) mapped to a non-white cell which is an ${L}_{2}$ neighbour of some white cell

B. 不包含任何白色元组，但包含映射到某个白色单元格的 ${L}_{2}$ 邻接非白色单元格的元组的页面

C. All other pages

C. 所有其他页面

To minimize page reads, our algorithm first reads Class A pages, and then Class B pages. Following this, it suffices to re-read Class A pages to complete the object-pairwise phase. Class $\mathrm{C}$ pages are not needed here.

为了最小化页面读取次数，我们的算法首先读取 A 类页面，然后读取 B 类页面。在此之后，重新读取 A 类页面就足以完成对象逐对阶段。这里不需要 $\mathrm{C}$ 类页面。

Consider a simple example where there are 600 pages in a dataset. Suppose pages 1 to 200 are Class A pages, pages 201 to 400 are Class B pages, and pages 401 to 600 are Class $\mathrm{C}$ pages. Suppose tuple $P$ is mapped to white cell ${C}_{w}$ ,and is stored in (Class A) page $i$ . For $P$ to complete its object-by-object distance calculations, these three kinds of tuples are needed:

考虑一个简单的例子，数据集中有 600 个页面。假设页面 1 到 200 是 A 类页面，页面 201 到 400 是 B 类页面，页面 401 到 600 是 $\mathrm{C}$ 类页面。假设元组 $P$ 映射到白色单元格 ${C}_{w}$，并存储在（A 类）页面 $i$ 中。为了让 $P$ 完成其逐个对象的距离计算，需要以下三种元组：

- white tuples $Q$ mapped to a white ${L}_{2}$ neighbour of ${C}_{w}$

- 映射到 ${C}_{w}$ 的白色 ${L}_{2}$ 邻接单元格的白色元组 $Q$

- non-white tuples $Q$ mapped to a non-white ${L}_{2}$ neighbour of ${C}_{w}$ ,and stored in page $j \geq  i$

- 映射到 ${C}_{w}$ 的非白色 ${L}_{2}$ 邻接单元格，并存储在页面 $j \geq  i$ 中的非白色元组 $Q$

- non-white tuples $Q$ mapped to a non-white ${L}_{2}$ neighbour of ${C}_{w}$ ,but stored in page $j < i$

- 映射到 ${C}_{w}$ 的非白色 ${L}_{2}$ 邻接单元格，但存储在页面 $j < i$ 中的非白色元组 $Q$

For the first kind of tuple,the pair(P,Q)is kept in main memory after the first 200 pages have been read, because both tuples are white. Thus, their separation can be computed and the appropriate counters (i.e., both $P$ ’s and $Q$ ’s) may be updated after all Class $\mathrm{A}$ pages have been read. For the second kind of tuple, the distance between the pair(P,Q)can be processed when page $j$ is read into main memory,because $P$ is already in main memory by then since $i \leq  j$ . The fact that $Q$ is not kept around afterwards does not affect $P$ at all. Thus, after the first 400 pages have been read (i.e., all class A and B pages), the second kind of tuples for $P$ have been checked. The only problem concerns the third kind of tuples. In this case,when $Q$ (which is stored in page $j$ ) is in main memory, $P$ (which is stored in page $i > j$ ) has not been read. Since $Q$ is a non-white tuple and is not kept around,then when $P$ eventually becomes available, $Q$ is gone. To deal with this situation,page $j$ needs to be re-read. In general, all Class A pages (except one) may need to be re-read. But it should be clear that because all white tuples are kept in main memory, it is sufficient to read Class A pages a second time.

对于第一种元组，在读取前200页后，对(P, Q)会被保留在主存中，因为这两个元组都是“白色”的。因此，可以计算它们之间的距离，并且在读取完所有$\mathrm{A}$类页面后，可以更新相应的计数器（即$P$和$Q$的计数器）。对于第二种元组，当页面$j$被读入主存时，可以处理对(P, Q)之间的距离，因为到那时$P$已经在主存中了，原因是$i \leq  j$。之后$Q$没有被保留下来这一事实根本不会影响$P$。因此，在读取前400页（即所有A类和B类页面）后，已经检查了$P$的第二种元组。唯一的问题涉及第三种元组。在这种情况下，当$Q$（存储在页面$j$中）在主存中时，$P$（存储在页面$i > j$中）还未被读取。由于$Q$是非“白色”元组且不会被保留，那么当$P$最终可用时，$Q$已经不在了。为了处理这种情况，需要重新读取页面$j$。一般来说，所有A类页面（除了一个）可能都需要重新读取。但应该清楚的是，由于所有“白色”元组都被保留在主存中，所以第二次读取A类页面就足够了。

Before presenting the formal algorithm, we offer two generalizations to the above example. First, the example assumes that all Class A pages precede all Class B pages in page numbering, and that pages are read in ascending order. Our argument above applies equally well if these assumptions are not made-so long as all Class A pages are read first, followed by all Class B pages, and the necessary (re-reading of) Class A pages. Second, Class A pages can be further divided into two subclasses: (A.1) pages that do not store any nonwhite tuple that is needed, and (A.2) pages that store some non-white tuple(s) that are needed. If this subdivision is made, it should be obvious from the above analysis that in re-reading all Class A pages, it suffices to re-read only the Class A. 2 pages. For simplicity, this optimization is not described in the algorithm below.

在给出正式算法之前，我们对上述示例进行两点推广。首先，该示例假设在页面编号中所有A类页面都在所有B类页面之前，并且页面是按升序读取的。如果不做这些假设，我们上面的论证同样适用——只要先读取所有A类页面，接着读取所有B类页面，并且必要时重新读取A类页面。其次，A类页面可以进一步分为两个子类：(A.1) 不存储任何所需非“白色”元组的页面，以及(A.2) 存储一些所需非“白色”元组的页面。如果进行了这种细分，从上述分析中应该可以明显看出，在重新读取所有A类页面时，只需要重新读取A.2类页面就足够了。为了简单起见，下面的算法中没有描述这种优化。

### 5.2 Algorithm FindAllOutsD for Disk-Resident Datasets

### 5.2 适用于磁盘驻留数据集的FindAllOutsD算法

Figure 4 presents Algorithm FindAllOutsD for mining outliers in large, disk-resident datasets. Much of the processing in the first 5 steps of Algorithm Find-AllOutsD is similar to that described for Algorithm FindAllOutsM shown in Figure 3. We draw attention to step 2 of Algorithm FindAllOutsD, which no longer stores $P$ but makes a note of the fact that $P$ ’s page contains some tuple(s) mapped to ${C}_{q}$ . This is important because (i) we may need to access a given cell's tuples later in the algorithm, and (ii) we need to know which cells have tuples from a particular page.

图4展示了用于挖掘大型磁盘驻留数据集中离群点的FindAllOutsD算法。FindAllOutsD算法的前5个步骤中的大部分处理与图3所示的FindAllOutsM算法的处理类似。我们注意到FindAllOutsD算法的步骤2，它不再存储$P$，而是记录下$P$所在页面包含一些映射到${C}_{q}$的元组这一事实。这很重要，因为(i) 在算法的后续过程中我们可能需要访问给定单元格的元组，并且(ii) 我们需要知道哪些单元格有来自特定页面的元组。

Step 5(c)(2) colours a white cell yellow if it has been determined that every tuple in a given cell is an outlier. Its tuples will be identified in step 8 after they have been read from their pages in step 6. Step 6 reads only those pages containing at least one white or yellow tuple. With respect to Section 5.1, this corresponds to reading all Class A pages. The white and yellow tuples from these pages are stored with the cell ${C}_{w}$ to which they have been quantized. ${C}_{w}$ stores exactly ${\operatorname{Count}}_{w}$ tuples,and this count is $\leq  M$ . To prepare for ${L}_{2}$ processing,step $6\left( b\right) \left( 1\right) \left( {ii}\right)$ initializes the $D$ -neighbour counter to the number of tuples in ${C}_{w} \cup  {L}_{1}\left( {C}_{w}\right)$ .

步骤5(c)(2) 如果确定给定单元格中的每个元组都是离群点，则将“白色”单元格标记为“黄色”。在步骤6从其页面读取这些元组后，将在步骤8中识别它们。步骤6仅读取那些至少包含一个“白色”或“黄色”元组的页面。关于5.1节，这对应于读取所有A类页面。这些页面中的“白色”和“黄色”元组将与它们被量化到的单元格${C}_{w}$一起存储。${C}_{w}$恰好存储${\operatorname{Count}}_{w}$个元组，这个计数是$\leq  M$。为了为${L}_{2}$处理做准备，步骤$6\left( b\right) \left( 1\right) \left( {ii}\right)$将$D$ - 邻域计数器初始化为${C}_{w} \cup  {L}_{1}\left( {C}_{w}\right)$中的元组数量。

In step 7,for each non-empty white cell ${C}_{w}$ ,we determine how many more $D$ -neighbours each tuple $P \in  {C}_{w}$ has,using (as potential neighbours) just the tuples read and stored in step 6 . As soon as we find that $P$ has $> {MD}$ -neighbours,we mark $P$ as a non-outlier. After step 7, it is possible that some (or all) of the non-empty white cells need no further comparisons, thereby reducing the number of reads in step 9 .

在步骤7中，对于每个非空的白色单元格${C}_{w}$，我们使用步骤6中读取并存储的元组（作为潜在邻居），确定每个元组$P \in  {C}_{w}$还需要多少个$D$ - 邻居。一旦我们发现$P$有$> {MD}$ - 邻居，就将$P$标记为非离群点。在步骤7之后，某些（或所有）非空白色单元格可能无需进一步比较，从而减少步骤9中的读取次数。

Necessary disk reads for cells that are both nonwhite and non-yellow are performed in step 9 . With respect to Section 5.1, this corresponds to reading all Class B pages, and re-reading (some) Class A pages. Again,we determine how many more $D$ -neighbours that each tuple $P$ in each white cell has,using only the newly read tuples from disk. If $P$ has $> {MD}$ - neighbours,then $P$ is marked as a non-outlier,and no further comparisons involving $P$ are necessary.

在步骤9中，对既不是白色也不是黄色的单元格进行必要的磁盘读取。参照第5.1节，这对应于读取所有B类页面，并重新读取（部分）A类页面。同样，我们仅使用从磁盘新读取的元组，确定每个白色单元格中的每个元组$P$还需要多少个$D$ - 邻居。如果$P$有$> {MD}$ - 邻居，则将$P$标记为非离群点，并且无需再对$P$进行进一步比较。

## Algorithm FindAllOutsD

## 算法FindAllOutsD

1. For $q \leftarrow  1,2,\ldots m,\;{\text{Count}}_{q} \leftarrow  0$

1. 对于$q \leftarrow  1,2,\ldots m,\;{\text{Count}}_{q} \leftarrow  0$

2. For each object $P$ in the dataset,do:

2. 对于数据集中的每个对象$P$，执行以下操作：

a. Map $P$ to its appropriate cell ${C}_{q}$ but do not store $P$ .

a. 将$P$映射到其合适的单元格${C}_{q}$，但不存储$P$。

b. Increment ${\operatorname{Count}}_{q}$ by 1 .

b. 将${\operatorname{Count}}_{q}$加1。

c. Note that ${C}_{q}$ references $P$ ’s page.

c. 注意，${C}_{q}$引用$P$的页面。

3. For $q \leftarrow  1,2,\ldots ,m$ ,if ${\operatorname{Count}}_{q} > M$ ,label ${C}_{q}$ red.

3. 对于$q \leftarrow  1,2,\ldots ,m$，如果${\operatorname{Count}}_{q} > M$，将${C}_{q}$标记为红色。

4. For each red cell ${C}_{r}$ ,label each of the ${L}_{1}$ neighbours of ${C}_{r}$ pink, provided the neighbour has not already been labelled red.

4. 对于每个红色单元格${C}_{r}$，将${C}_{r}$的每个${L}_{1}$邻居标记为粉色，前提是该邻居尚未被标记为红色。

5. For each non-empty white (i.e.,uncoloured) cell ${C}_{w}$ ,do:

5. 对于每个非空的白色（即未着色）单元格${C}_{w}$，执行以下操作：

a. Coun ${t}_{w2} \leftarrow  {\text{Count}}_{w} + \mathop{\sum }\limits_{{i \in  {L}_{1}\left( {C}_{w}\right) }}{\text{Count}}_{i}$

a. 计数${t}_{w2} \leftarrow  {\text{Count}}_{w} + \mathop{\sum }\limits_{{i \in  {L}_{1}\left( {C}_{w}\right) }}{\text{Count}}_{i}$

b. If ${\operatorname{Count}}_{w2} > M$ ,label ${C}_{w}$ pink.

b. 如果${\operatorname{Count}}_{w2} > M$，将${C}_{w}$标记为粉色。

c. else

c. 否则

1. Count ${t}_{w3} \leftarrow$ Coun ${t}_{w2} + \mathop{\sum }\limits_{{i \in  {L}_{2}\left( {C}_{w}\right) }}$ Coun ${t}_{i}$

1. 计数${t}_{w3} \leftarrow$ 计数${t}_{w2} + \mathop{\sum }\limits_{{i \in  {L}_{2}\left( {C}_{w}\right) }}$ 计数${t}_{i}$

2. If ${\text{Count}}_{w3} \leq  M$ ,label ${C}_{w}$ yellow to indicate that all tuples mapping to ${C}_{w}$ are outliers.

2. 如果${\text{Count}}_{w3} \leq  M$，将${C}_{w}$标记为黄色，以表明映射到${C}_{w}$的所有元组都是离群点。

3. else ${\operatorname{Sum}}_{w} \leftarrow  {\operatorname{Count}}_{w2}$

3. 否则 ${\operatorname{Sum}}_{w} \leftarrow  {\operatorname{Count}}_{w2}$

6. For each page $i$ containing at least 1 white or yellow tuple, do:

6. 对于每个至少包含1个白色或黄色元组的页面 $i$，执行以下操作：

a. Read page $i$ .

a. 读取页面 $i$。

b. For each white or yellow cell ${C}_{q}$ having tuples in page $i$ ,do:

b. 对于页面 $i$ 中每个包含元组的白色或黄色单元格 ${C}_{q}$，执行以下操作：

1. For each object $P$ in page $i$ mapped to ${C}_{q}$ ,do: i. Store $P$ in ${C}_{q}$ .

1. 对于页面 $i$ 中映射到 ${C}_{q}$ 的每个对象 $P$，执行以下操作：i. 将 $P$ 存储在 ${C}_{q}$ 中。

ii. Kount ${}_{P} \leftarrow  {\operatorname{Sum}}_{q}$

ii. 计数 ${}_{P} \leftarrow  {\operatorname{Sum}}_{q}$

7. For each object $P$ in each non-empty white cell ${C}_{w}$ ,do:

7. 对于每个非空白色单元格 ${C}_{w}$ 中的每个对象 $P$，执行以下操作：

a. For each white or yellow cell ${C}_{L} \in  {L}_{2}\left( {C}_{w}\right)$ ,do:

a. 对于每个白色或黄色单元格 ${C}_{L} \in  {L}_{2}\left( {C}_{w}\right)$，执行以下操作：

1. For each object $Q \in  {C}_{L}$ ,if $\operatorname{dist}\left( {P,Q}\right)  \leq  D$ : Increment ${\text{Kount}}_{P}$ by 1 . If ${\text{Kount}}_{P} > M$ , mark $P$ as a non-outlier,and goto next $P$ .

1. 对于每个对象 $Q \in  {C}_{L}$，如果 $\operatorname{dist}\left( {P,Q}\right)  \leq  D$：将 ${\text{Kount}}_{P}$ 加1。如果 ${\text{Kount}}_{P} > M$，将 $P$ 标记为非离群值，并转到下一个 $P$。

8. For each object $Q$ in each yellow cell,report $Q$ as an outlier.

8. 对于每个黄色单元格中的每个对象 $Q$，将 $Q$ 报告为离群值。

9. For each page $i$ containing at least 1 tuple that (i) is both non-white and non-yellow,and (ii) maps to an ${L}_{2}$ neighbour of some white cell $C$ ,do:

9. 对于每个至少包含1个元组的页面 $i$，该元组（i）既不是白色也不是黄色，并且（ii）映射到某个白色单元格 $C$ 的 ${L}_{2}$ 邻域，执行以下操作：

a. Read page $i$ .

a. 读取页面 $i$。

b. For each cell ${C}_{q} \in  {L}_{2}\left( C\right)$ that is both non-white and non-yellow,and has tuples in page $i$ ,do:

b. 对于页面 $i$ 中每个既不是白色也不是黄色且包含元组的单元格 ${C}_{q} \in  {L}_{2}\left( C\right)$，执行以下操作：

1. For each object $Q$ in page $i$ mapped to ${C}_{q}$ ,do:

1. 对于页面 $i$ 中映射到 ${C}_{q}$ 的每个对象 $Q$，执行以下操作：

i. For each non-empty white cell ${C}_{w} \in$ ${L}_{2}\left( {C}_{q}\right)$ ,do:

i. 对于每个非空白色单元格 ${C}_{w} \in$ ${L}_{2}\left( {C}_{q}\right)$，执行以下操作：

For each object $P \in  {C}_{w}$ ,if $\operatorname{dist}\left( {P,Q}\right)  \leq$ $D$ :

对于每个对象 $P \in  {C}_{w}$，如果 $\operatorname{dist}\left( {P,Q}\right)  \leq$ $D$：

Increment ${\text{Kount}}_{P}$ by 1 . If ${\text{Kount}}_{P} >$

将${\text{Kount}}_{P}$加1。如果${\text{Kount}}_{P} >$

$M$ ,mark $P$ as a non-outlier.

$M$，将$P$标记为非离群值。

10. For each object $P$ in each non-empty white cell,if $P$ has not been marked as a non-outlier,then report $P$ as an outlier.

10. 对于每个非空白色单元格中的每个对象$P$，如果$P$尚未被标记为非离群值，则将$P$报告为离群值。

Figure 4: Pseudo-Code for Algorithm FindAllOutsD

图4：算法FindAllOutsD的伪代码

### 5.3 Analysis of Algorithm FindAllOutsD and Comparison with Algorithm NL

### 5.3 算法FindAllOutsD的分析以及与算法NL的比较

Algorithm FindAllOutsD has a linear complexity wrt $N$ for the same reasons explained for Algorithm Find-AllOutsM (cf: Section 4.3), but by design, Algorithm FindAllOutsD has the following important advantage over Algorithm FindAllOutsM wrt I/O performance.

出于与算法Find - AllOutsM相同的原因（参见：第4.3节），算法FindAllOutsD相对于$N$具有线性复杂度，但从设计上看，算法FindAllOutsD在I/O性能方面相对于算法FindAllOutsM具有以下重要优势。

Lemma 3 Algorithm FindAllOutsD requires at most 3 passes over the dataset.

引理3 算法FindAllOutsD最多需要对数据集进行3次遍历。

Proof Outline: The initial mapping phase requires one pass over the dataset. Let $n$ be the total number of pages in the dataset. Then if ${n}_{1},{n}_{2},{n}_{3}$ denote the number of pages in Classes $A,B$ ,and $C$ respectively (cf: Section 5.1),it is necessary that $n = {n}_{1} + {n}_{2} + {n}_{3}$ . As shown in Section 5.1, the maximum total number of pages read in the object-pairwise phase is given by ${n}_{1} + {n}_{2} + {n}_{1}$ ,which is obviously $\leq  {2n}$ . Thus,the entire algorithm requires no more than 3 passes.

证明概要：初始映射阶段需要对数据集进行一次遍历。设 $n$ 为数据集中页面的总数。若 ${n}_{1},{n}_{2},{n}_{3}$ 分别表示类别 $A,B$ 和 $C$ 中的页面数量（参见：第 5.1 节），则必须满足 $n = {n}_{1} + {n}_{2} + {n}_{3}$。如第 5.1 节所示，对象成对阶段读取的页面总数的最大值由 ${n}_{1} + {n}_{2} + {n}_{1}$ 给出，显然为 $\leq  {2n}$。因此，整个算法最多需要三次遍历。

The above guarantee is conservative for two reasons. First,the sum ${n}_{1} + {n}_{2} + {n}_{1}$ can be smaller than $n$ . For example,if ${n}_{1} \leq  {n}_{3}$ ,then the sum is $\leq  n$ ,implying that while some page may be read 3 times, the total number of pages read is equivalent to no more than 2 passes over the dataset. Second, the above guarantee assumes that: (i) there is enough buffer space for storing the white tuples (as will be shown later, this is not a strong assumption because typically there are not too many non-empty white cells), and (ii) there is only one page remaining in the buffer space for Class A and B pages. More buffer space can be dedicated to keep more Class A pages around, thereby reducing the number of page re-reads for Class A pages.

上述保证较为保守，原因有二。其一，总和 ${n}_{1} + {n}_{2} + {n}_{1}$ 可能小于 $n$。例如，若 ${n}_{1} \leq  {n}_{3}$，则总和为 $\leq  n$，这意味着虽然某些页面可能被读取三次，但读取的页面总数相当于对数据集的遍历次数不超过两次。其二，上述保证假设：(i) 有足够的缓冲区空间来存储白色元组（后文将表明，这并非强假设，因为通常非空白色单元格的数量并不多）；(ii) 缓冲区空间中仅为 A 类和 B 类页面各保留了一个页面。可以分配更多的缓冲区空间来保留更多的 A 类页面，从而减少 A 类页面的重读次数。

At this point, let us revisit Algorithm NL, used for block-oriented, nested-loop processing of disk-resident datasets (cf: Section 3.2). We will show that Algorithm FindAllOutsD guarantees fewer dataset passes than Algorithm NL does, for sufficiently large datasets.

此时，让我们回顾一下算法 NL，它用于对磁盘驻留数据集进行面向块的嵌套循环处理（参见：第 3.2 节）。我们将证明，对于足够大的数据集，算法 FindAllOutsD 保证的数据集遍历次数比算法 NL 少。

Lemma 4 If a dataset is divided into $n = \left\lceil  \frac{200}{B}\right\rceil$ blocks ( $B$ is the percentage of buffering),then (i) the total number of block reads required by Algorithm NL is $n + \left( {n - 2}\right) \left( {n - 1}\right)$ ,and (ii) the number of passes over the dataset is $\geq  n - 2$ .

引理 4：若将数据集划分为 $n = \left\lceil  \frac{200}{B}\right\rceil$ 个块（$B$ 为缓冲百分比），则 (i) 算法 NL 所需的块读取总数为 $n + \left( {n - 2}\right) \left( {n - 1}\right)$；(ii) 对数据集的遍历次数为 $\geq  n - 2$。

Proof Outline: (i) Each of the $n$ blocks must be read exactly once during the first dataset pass. At the end of each pass, we retain 2 blocks in memory, so only $n - 2$ additional blocks need to be read during passes $2,3,\ldots ,n$ . Thus, $n + \left( {n - 2}\right) \left( {n - 1}\right)$ blocks are read. (ii) The number of dataset passes is: $\frac{n + \left( {n - 2}\right) \left( {n - 1}\right) }{n} =$ $\frac{{n}^{2} - {2n} + 2}{n} = n - 2 + \frac{2}{n} \geq  n - 2$

证明概要：(i) 在第一次数据集遍历期间，$n$ 个块中的每一个都必须恰好读取一次。每次遍历结束时，我们在内存中保留 2 个块，因此在第 $2,3,\ldots ,n$ 次遍历期间仅需读取 $n - 2$ 个额外的块。因此，总共读取 $n + \left( {n - 2}\right) \left( {n - 1}\right)$ 个块。(ii) 数据集的遍历次数为：$\frac{n + \left( {n - 2}\right) \left( {n - 1}\right) }{n} =$ $\frac{{n}^{2} - {2n} + 2}{n} = n - 2 + \frac{2}{n} \geq  n - 2$

In general, Algorithm NL may require many more passes than Algorithm FindAllOutsD. For example, if a large dataset is split into $n = {10}$ pieces,Lemma 4 states that Algorithm NL requires $n - 2 = {10} -$ $2 = 8$ passes,which is 5 more passes than Algorithm FindAllOutsD may need.

一般而言，算法 NL 所需的遍历次数可能比算法 FindAllOutsD 多得多。例如，若将一个大型数据集拆分为 $n = {10}$ 个部分，引理 4 表明算法 NL 需要 $n - 2 = {10} -$ $2 = 8$ 次遍历，比算法 FindAllOutsD 可能需要的遍历次数多 5 次。

## 6 Empirical Behaviour of the Algo- rithms

## 6 算法的实证表现

### 6.1 Experimental Setup

### 6.1 实验设置

Our base dataset is an 855-record dataset consisting of 1995-96 NHL player performance statistics. These statistics include numbers of goals, assists, points, penalty minutes, shots on goal, games played, power play goals, etc. Since this real-life dataset is quite small, and since we want to test our algorithms on large, disk-resident datasets, we created a number of synthetic datasets mirroring the distribution of statistics within the NHL dataset. More specifically, we determined the distribution of the attributes in the original dataset by using 10-partition histograms. Then, we generated datasets containing between 10,000 and 2,000,000 tuples, whose distributions mirrored that of the base dataset. Each page held up to 13 tuples.

我们的基础数据集是一个包含 855 条记录的数据集，这些记录是 1995 - 1996 年美国国家冰球联盟（NHL）球员的表现统计数据。这些统计数据包括进球数、助攻数、得分、犯规分钟数、射门次数、参赛场次、强力进攻进球数等。由于这个现实生活中的数据集相当小，并且我们希望在大型磁盘驻留数据集上测试我们的算法，因此我们创建了一些模拟 NHL 数据集中统计数据分布的合成数据集。更具体地说，我们使用 10 分区直方图确定了原始数据集中属性的分布。然后，我们生成了包含 10,000 到 2,000,000 个元组的数据集，其分布与基础数据集的分布一致。每个页面最多可容纳 13 个元组。

All of our tests were run on a Sun Microsystems UltraSPARC-1 machine having 128 MB of main memory. Unless otherwise indicated, all times shown in this paper are CPU times plus I/O times. ${}^{7}$ Our code was written in $\mathrm{C} +  +$ and was processed by an optimizing compiler. The modes of operation that we used, and their acronyms, are as follows:

我们所有的测试均在一台配备128MB主内存的太阳微系统公司（Sun Microsystems）UltraSPARC - 1机器上运行。除非另有说明，本文中显示的所有时间均为CPU时间加上I/O时间。${}^{7}$ 我们的代码用$\mathrm{C} +  +$编写，并由一个优化编译器进行处理。我们使用的操作模式及其缩写如下：

1. CS is a multidimensional cell structure implementation as described by either Algorithm FindAll-OutsM or FindAllOutsD. The context makes it clear which algorithm is being evaluated.

1. CS是由算法FindAll - OutsM或FindAllOutsD所描述的多维单元格结构（Multidimensional Cell Structure）实现。上下文会明确指出正在评估的是哪个算法。

2. NL is an implementation of Algorithm NL. The amount of memory permitted for buffering in each NL case is the same amount of memory required by CS. For example, if CS uses 10 MB of main memory, then 10 MB is also available for NL.

2. NL是算法NL的实现。在每个NL案例中允许用于缓冲的内存量与CS所需的内存量相同。例如，如果CS使用10MB主内存，那么NL也有10MB可用。

3. KD is a memory-based $k$ -d tree implementation.

3. KD是基于内存的$k$ - d树（$k$ - d Tree）实现。

4. RT is a disk-based R-tree implementation.

4. RT是基于磁盘的R树（R - Tree）实现。

Range query processing in KD and RT modes has been optimized to terminate as soon as the number of $D$ - neighbours exceeds $M$ .

KD和RT模式下的范围查询处理已进行优化，一旦$D$ - 近邻的数量超过$M$就会终止。

### 6.2 Varying the Dataset Size

### 6.2 改变数据集大小

Figure 5 shows results for various modes and various dataset sizes for 3-D,using $p = {0.9999}$ . Specifically,it shows how CPU + I/O time is affected by the number of tuples. (The $x$ -axis is measured in millions of tuples.) For example, CS takes 256.00 seconds in total time to find all the appropriate ${DB}$ -outliers in a 2 million tuple dataset. In contrast, NL takes 2332.10 seconds, about 9 times as long. RT mode is even less competitive. Unlike CS,RT is not linear wrt $N$ . In fact, just building the R-tree can take at least 10 times as long as CS, let alone searching the tree.

图5展示了在三维情况下，使用$p = {0.9999}$时各种模式和不同数据集大小的结果。具体而言，它显示了CPU + I/O时间如何受元组数量的影响。（$x$轴以百万个元组为单位进行度量。）例如，在一个包含200万个元组的数据集中，CS总共花费256.00秒来查找所有合适的${DB}$ - 离群值。相比之下，NL花费2332.10秒，大约是CS的9倍。RT模式的竞争力更弱。与CS不同，RT相对于$N$不是线性的。实际上，仅构建R树所需的时间就至少是CS的10倍，更不用说搜索树了。

<!-- Media -->

<!-- figureText: 2000 CPU + I/O Time versus Number of Tuples, for 3-D Modes AT (total) AT (build) CS 1.8 Number of Tuples 1800 CPU + I/O Time in Seconds -->

<img src="https://cdn.noedgeai.com/0195c913-c64b-73be-a45d-3920f48f6845_9.jpg?x=940&y=214&w=607&h=496&r=0"/>

Figure 5: How CPU + I/O Time Scales with $N$ for 3-D Disk-Resident Datasets,Using $p = {0.9999}$

图5：对于三维磁盘驻留数据集，使用$p = {0.9999}$时CPU + I/O时间随$N$的变化情况

Table 1: CPU Times (in Seconds) for 3-D, Memory-Resident Datasets,using $p = {0.9995}$

表1：对于三维内存驻留数据集，使用$p = {0.9995}$时的CPU时间（以秒为单位）

<table><tr><td>$N$</td><td>${CS}$</td><td>${NL}$</td><td>${KD}$</td></tr><tr><td>20000</td><td>0.32</td><td>1.02</td><td>3.14</td></tr><tr><td>40000</td><td>0.54</td><td>4.26</td><td>20.49</td></tr><tr><td>60000</td><td>0.74</td><td>9.64</td><td>33.08</td></tr><tr><td>80000</td><td>1.04</td><td>17.58</td><td>54.66</td></tr><tr><td>100000</td><td>1.43</td><td>27.67</td><td>104.28</td></tr></table>

<table><tr><td>$N$</td><td>${CS}$</td><td>${NL}$</td><td>${KD}$</td></tr><tr><td>20000</td><td>0.32</td><td>1.02</td><td>3.14</td></tr><tr><td>40000</td><td>0.54</td><td>4.26</td><td>20.49</td></tr><tr><td>60000</td><td>0.74</td><td>9.64</td><td>33.08</td></tr><tr><td>80000</td><td>1.04</td><td>17.58</td><td>54.66</td></tr><tr><td>100000</td><td>1.43</td><td>27.67</td><td>104.28</td></tr></table>

<!-- Media -->

While the preceding paragraph concerns disk-resident datasets, Table 1 summarizes the results for memory-resident datasets in 3-D,using $p = {0.9995}$ . Again, CS can outperform NL by an order of magnitude,and the index-based algorithm-a $k$ -d tree (KD) in this case-takes much longer, even if we just consider the time it takes to build the index.

虽然上一段讨论的是磁盘驻留数据集，但表1总结了使用$p = {0.9995}$时三维内存驻留数据集的结果。同样，CS（Cell-based Scoring，基于单元格的评分算法）的性能可以比NL（Nearest Neighbor，最近邻算法）高出一个数量级，并且基于索引的算法——在这种情况下是$k$-d树（KD树）——所需的时间要长得多，即使我们只考虑构建索引所需的时间。

### 6.3 Varying the Value of $p$

### 6.3 改变$p$的值

Figure 6 shows how the percentages of white, pink, red,and non-empty white cells vary with $p.{}^{8}$ The total number of cells is simply the sum of the red, pink, and white cells. Processing time is less when there is a greater percentage of $\mathit{{red}}$ and $\mathit{{pink}}$ cells because we can quickly eliminate a larger number of tuples from being considered as outliers. The success of the cell-based algorithms is largely due to the fact that many of the cells may be red or pink (and there may be relatively few non-empty white cells). Recall that non-empty white cells require the most computational effort.

图6展示了白色、粉色、红色和非空白色单元格的百分比如何随$p.{}^{8}$变化。单元格的总数就是红色、粉色和白色单元格的数量之和。当$\mathit{{red}}$和$\mathit{{pink}}$单元格的百分比更高时，处理时间更短，因为我们可以迅速排除大量元组作为离群值的可能性。基于单元格的算法的成功很大程度上是因为许多单元格可能是红色或粉色（并且非空白色单元格可能相对较少）。请记住，非空白色单元格需要最多的计算量。

---

<!-- Footnote -->

${}^{7}$ Our CPU timer wraps around after 2147 seconds; hence, times beyond this are unreliable. Where we have quoted a CPU +I/O figure > 2147 , it is because the CPU time was < 2147 , but the I/O time actually caused the sum to exceed 2147. (CPU time and $\mathrm{I}/\mathrm{O}$ time were measured separately.)

${}^{7}$ 我们的CPU计时器在2147秒后会溢出；因此，超过这个时间的计时是不可靠的。当我们引用的CPU + I/O时间大于2147时，是因为CPU时间小于2147，但I/O时间实际上导致总和超过了2147。（CPU时间和$\mathrm{I}/\mathrm{O}$时间是分别测量的。）

${}^{8}$ We include yellow cells with the white cell population since yellow cells are just a special type of white cell.

${}^{8}$ 我们将黄色单元格归为白色单元格，因为黄色单元格只是白色单元格的一种特殊类型。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: 80 Percentage of Total Number of Celle versus p Type of Cell: White Non-Empty White Pink Red 0.9975 0.998 0.9985 0.999 0.9995 Percentage of Total Number of Cells 70 80 30 0.995 0.9955 0.996 0.9965 0.997 -->

<img src="https://cdn.noedgeai.com/0195c913-c64b-73be-a45d-3920f48f6845_10.jpg?x=231&y=235&w=608&h=497&r=0"/>

Figure 6: 3-D Cell Colouring Statistics for Variable $p$ , for 500,000 Tuples

图6：500,000个元组在可变$p$下的三维单元格着色统计

<!-- Media -->

### 6.4 Varying the Number of Dimensions and Cells

### 6.4 改变维度和单元格的数量

In this section, we see how the number of dimensions and cells affects performance. We omit the trivial 1-D and 2-D results, but show results for 3-D, 4-D, and 5-D. Beyond 5-D, we believe that NL will be the clear winner. Table 2 shows how CS and NL compare in different dimensions for disk-resident datasets of various sizes,using $p = {0.9999}$ .

在本节中，我们将探讨维度和单元格的数量如何影响性能。我们省略了平凡的一维和二维结果，但展示了三维、四维和五维的结果。超过五维后，我们认为NL将明显胜出。表2展示了在使用$p = {0.9999}$的情况下，CS和NL在不同维度的各种大小磁盘驻留数据集上的比较情况。

For CS mode in 5-D, we varied the number of partitions ${m}_{i}$ in a given dimension $i$ . We chose ${m}_{i} = {10}$ , 8,and 6 for each dimension. The columns $\operatorname{CS}\left( {10}^{5}\right)$ , $\mathrm{{CS}}\left( {8}^{5}\right)$ ,and $\mathrm{{CS}}\left( {6}^{5}\right)$ stand for cases where the cell structure contains $m = {\Pi }_{i = 1}^{k}{m}_{i} = {10}^{5},{8}^{5}$ ,and ${6}^{5}$ cells, respectively. The table shows that (i) CS outperforms NL in 3-D by almost an order of magnitude, (ii) CS clearly outperforms NL in 4-D, and (iii) NL is the clear winner in 5-D. Due to the exponential growth in the number of cells, CS is uncompetitive with NL in 5-D.

对于五维的CS模式，我们改变了给定维度$i$中的分区数量${m}_{i}$。我们为每个维度选择了${m}_{i} = {10}$、8和6。列$\operatorname{CS}\left( {10}^{5}\right)$、$\mathrm{{CS}}\left( {8}^{5}\right)$和$\mathrm{{CS}}\left( {6}^{5}\right)$分别代表单元格结构包含$m = {\Pi }_{i = 1}^{k}{m}_{i} = {10}^{5},{8}^{5}$和${6}^{5}$个单元格的情况。该表显示：（i）在三维中，CS的性能几乎比NL高出一个数量级；（ii）在四维中，CS明显优于NL；（iii）在五维中，NL明显胜出。由于单元格数量呈指数级增长，在五维中CS与NL相比没有竞争力。

Even when the number of cells is greatly reduced in 5-D, CS generally cannot beat NL. In fact, of all the 5-D tests we ran (in addition to those shown in Table 2), there was only one case where CS actually beat NL. Perhaps surprisingly, the table shows that a reduction in the number of cells in 5-D does not necessarily result in a reduction in total time. This is due to an $\mathrm{I}/\mathrm{O}$ optimization that we included in our implementation, whereby dramatic savings are achieved for larger numbers of cells (for values of $p$ close to unity). ${}^{9}$ Without the optimization, however, it is true that reducing the number of cells normally results in a reduction in total time. Our implementation uses a hybrid strategy whereby we turn off the optimization at certain thresholds of $p$ and $m$ .

即使在五维中单元格数量大幅减少，CS通常也无法击败NL。事实上，在我们进行的所有五维测试中（除了表2中显示的那些），只有一种情况CS实际上击败了NL。也许令人惊讶的是，该表显示在五维中减少单元格数量并不一定会导致总时间减少。这是由于我们在实现中包含的一个$\mathrm{I}/\mathrm{O}$优化，通过该优化，对于大量单元格（当$p$的值接近1时）可以实现显著的节省。${}^{9}$ 然而，如果没有这个优化，通常减少单元格数量确实会导致总时间减少。我们的实现采用了一种混合策略，即在$p$和$m$的某些阈值处关闭优化。

Finally, because we made CS and NL use the same amount of memory for fair comparison, the amount of buffer space available to NL increased as $k$ increased. This explains why the execution time of NL shown in Table 2 often dropped with increasing dimensions.

最后，为了进行公平比较，我们让CS和NL使用相同的内存量，因此随着$k$的增加，NL可用的缓冲区空间也增加了。这就解释了为什么表2中显示的NL执行时间通常会随着维度的增加而下降。

## 7 Conclusions

## 7 结论

We believe that identifying ${DB}$ -outliers is an important and useful data mining activity. In this paper, we proposed and analyzed several algorithms for finding ${DB}$ -outliers. In addition to two simple $O\left( {k{N}^{2}}\right)$ algorithms, we developed cell-based algorithms that are linear with respect to $N$ and are suitable for $k \leq  4$ . The cell-based algorithm developed for large, disk-resident datasets also guarantees that no data page is read more than 3 times, if not once or twice. Our empirical results suggest that (i) the cell-based algorithms are far superior to the other algorithms for $k \leq  4$ (in some cases, by at least an order of magnitude), (ii) the nested-loop algorithm is the choice for $k \geq  5$ dimensions,and (iii) finding all ${DB}$ -outliers is computationally very feasible for large, multidimensional datasets (e.g., 2.5 minutes total time for 500,000 tuples in 5-D). Using Algorithm NL, there is no practical limit on the size of the dataset or on the number of dimensions.

我们认为，识别 ${DB}$ 离群点是一项重要且有用的数据挖掘活动。在本文中，我们提出并分析了几种查找 ${DB}$ 离群点的算法。除了两种简单的 $O\left( {k{N}^{2}}\right)$ 算法外，我们还开发了基于单元格的算法，这些算法相对于 $N$ 是线性的，适用于 $k \leq  4$ 。为大型磁盘驻留数据集开发的基于单元格的算法还保证，如果不是读取一次或两次，任何数据页的读取次数不会超过 3 次。我们的实验结果表明：（i）对于 $k \leq  4$ ，基于单元格的算法远优于其他算法（在某些情况下，至少高出一个数量级）；（ii）嵌套循环算法适用于 $k \geq  5$ 维；（iii）对于大型多维数据集，查找所有 ${DB}$ 离群点在计算上是非常可行的（例如，对于 5 维的 500,000 个元组，总时间为 2.5 分钟）。使用算法 NL，数据集的大小或维度数量没有实际限制。

In ongoing work, we are developing incremental techniques that allow the user to freely experiment with $p$ and $D$ ,but do not require the cell structure to be recomputed from scratch for every change of the parameters. We are also looking for ways to incorporate user-defined distance functions, and to provide incremental support for changing distance functions.

在正在进行的工作中，我们正在开发增量技术，允许用户自由地对 $p$ 和 $D$ 进行实验，但不需要为每次参数更改都从头重新计算单元格结构。我们还在寻找整合用户定义的距离函数的方法，并为更改距离函数提供增量支持。

## Acknowledgements

## 致谢

This research has been partially sponsored by NSERC Grant OGP0138055 and IRIS-2 Grants HMI-5 & IC-5.

这项研究部分由加拿大自然科学与工程研究委员会（NSERC）的 OGP0138055 资助以及 IRIS - 2 的 HMI - 5 和 IC - 5 资助。

## References

## 参考文献

[AAR96] A. Arning, R. Agrawal, and P. Raghavan. A linear method for deviation detection in large databases. In Proc. KDD, pages 164- 169, 1996.

[AAR96] A. Arning、R. Agrawal 和 P. Raghavan。一种用于大型数据库中偏差检测的线性方法。见《知识发现与数据挖掘会议论文集》，第 164 - 169 页，1996 年。

$\left\lbrack  {{\mathrm{{AGI}}}^{ + }{92}}\right\rbrack  \mathrm{R}$ . Agrawal, $\mathrm{S}$ . Ghosh, $\mathrm{T}$ . Imielinski, B. Iyer, and A. Swami. An interval clas-

$\left\lbrack  {{\mathrm{{AGI}}}^{ + }{92}}\right\rbrack  \mathrm{R}$ . Agrawal、$\mathrm{S}$ . Ghosh、$\mathrm{T}$ . Imielinski、B. Iyer 和 A. Swami。一个区间分类

---

<!-- Footnote -->

${}^{9}$ In particular,for a non-empty white cell ${C}_{w}$ ,as soon as the number of $D$ -neighbours exceeds $M$ for all of ${C}_{w}$ ’s tuples,we explicitly search through ${L}_{2}\left( {C}_{w}\right)$ ’s cells looking for red or pink cells. For each such red or pink cell, we determine the block ID's (pages) that contain tuples mapped to it, and then we subtract 1 from the respective block ID counters. Later in the program,

${}^{9}$ 特别地，对于一个非空的白色单元格 ${C}_{w}$ ，一旦 ${C}_{w}$ 中所有元组的 $D$ 邻域数量超过 $M$ ，我们就会明确地在 ${L}_{2}\left( {C}_{w}\right)$ 的单元格中搜索红色或粉色单元格。对于每个这样的红色或粉色单元格，我们确定包含映射到该单元格的元组的块 ID（页面），然后从相应的块 ID 计数器中减去 1。在程序的后续阶段，

when it comes time to read the block, if the block ID counter is no longer positive, we avoid reading and processing the page because we know that this page is no longer needed. Thus, for the additional expense of searching ${L}_{2}\left( {C}_{w}\right)$ cells,we may realize substantial savings; however,this advantage is lost as $p$ becomes smaller or as the overall number of cells becomes fewer.

当需要读取该块时，如果块 ID 计数器不再为正，我们就避免读取和处理该页面，因为我们知道不再需要该页面。因此，虽然搜索 ${L}_{2}\left( {C}_{w}\right)$ 单元格会产生额外的开销，但我们可能会实现大量的节省；然而，随着 $p$ 变小或单元格总数变少，这种优势就会丧失。

<!-- Footnote -->

---

<!-- Media -->

Table 2: CPU + I/O Times (in Seconds) for a Variable Number of Tuples,Dimensions,and Cells—for $p = {0.9999}$ .

表 2：对于可变数量的元组、维度和单元格，$p = {0.9999}$ 的 CPU + I/O 时间（以秒为单位）。

<table><tr><td/><td colspan="2">3-D</td><td colspan="2">4-D</td><td colspan="4">5-D</td></tr><tr><td>$N$</td><td>CS $\left( {10}^{3}\right)$</td><td>NL</td><td>CS $\left( {10}^{4}\right)$</td><td>NL</td><td>$\operatorname{CS}\left( {10}^{5}\right)$</td><td>CS(8)</td><td>CS(6°)</td><td>NL</td></tr><tr><td>100,000</td><td>10.77</td><td>93.96</td><td>23.32</td><td>45.79</td><td>93.40</td><td>217.04</td><td>205.63</td><td>17.30</td></tr><tr><td>500,000</td><td>57.10</td><td>490.62</td><td>114.00</td><td>223.51</td><td>695.37</td><td>997.11</td><td>1061.33</td><td>148.44</td></tr><tr><td>2,000,000</td><td>253.90</td><td>2332.10</td><td>606.56</td><td>1421.16</td><td>>2147</td><td>>2147</td><td>>2147</td><td>1555.78</td></tr></table>

<table><tbody><tr><td></td><td colspan="2">3-D</td><td colspan="2">4-D</td><td colspan="4">5-D</td></tr><tr><td>$N$</td><td>计算机科学 $\left( {10}^{3}\right)$</td><td>自然语言</td><td>计算机科学 $\left( {10}^{4}\right)$</td><td>自然语言</td><td>$\operatorname{CS}\left( {10}^{5}\right)$</td><td>计算机科学(8)</td><td>计算机科学(6°)</td><td>自然语言</td></tr><tr><td>100,000</td><td>10.77</td><td>93.96</td><td>23.32</td><td>45.79</td><td>93.40</td><td>217.04</td><td>205.63</td><td>17.30</td></tr><tr><td>500,000</td><td>57.10</td><td>490.62</td><td>114.00</td><td>223.51</td><td>695.37</td><td>997.11</td><td>1061.33</td><td>148.44</td></tr><tr><td>2,000,000</td><td>253.90</td><td>2332.10</td><td>606.56</td><td>1421.16</td><td>>2147</td><td>>2147</td><td>>2147</td><td>1555.78</td></tr></tbody></table>

<!-- Media -->

sifier for database mining applications. In Proc. 18th VLDB, pages 560-573, 1992.

[AIS93] R. Agrawal, T. Imielinski, and A. Swami. Mining association rules between sets of items in large databases. In Proc. ACM SIGMOD, pages 207-216, 1993.

[AL88] D. Angluin and P. Laird. Learning from noisy examples. Machine Learning, 2(4):343-370, 1988.

[BCP+97] I. S. Bhandari, E. Colet, J. Parker, Z. Pines, R. Pratap, and K. Ramanujam. Advanced scout: Data mining and knowledge discovery in NBA data. Data Mining and Knowledge Discovery, 1(1):121- 125, 1997.

[Ben75] J. L. Bentley. Multidimensional binary search trees used for associative searching. ${CACM},{18}\left( 9\right)  : {509} - {517},{1975}$ .

[BL94] V. Barnett and T. Lewis. Outliers in Statistical Data. John Wiley, 3rd edition, 1994.

[EKSX96] M. Ester, H.-P. Kriegel, J. Sander, and X. Xu. A density-based algorithm for discovering clusters in large spatial databases with noise. In Proc. KDD, pages 226-231, 1996.

[FPP78] D. Freedman, R. Pisani, and R. Purves. Statistics. W.W. Norton, New York, 1978.

[Gut84] R. Guttmann. A dynamic index structure for spatial searching. In Proc. ACM SIG- ${MOD}$ ,pages ${47} - {57},{1984}$ .

[Haw80] D. Hawkins. Identification of Outliers. Chapman and Hall, London, 1980.

[HCC92] J. Han, Y. Cai, and N. Cercone. Knowledge discovery in databases: An attribute-oriented approach. In Proc. 18th VLDB, pages 547-559, 1992.

[HKP97] J. Hellerstein, E. Koutsoupias, and C. Pa-padimitriou. On the analysis of indexing schemes. In Proc. PODS, pages 249-256, 1997.

[JW92] R. A. Johnson and D. W. Wichern. Applied Multivariate Statistical Analysis. Prentice-Hall, 3rd edition, 1992.

[KN96] E. M. Knorr and R. T. Ng. Finding aggregate proximity relationships and commonalities in spatial data mining. IEEE Transactions on Knowledge and Data Engineering, 8(6):884-897, 1996.

[KN97] E. M. Knorr and R. T. Ng. A unified notion of outliers: Properties and computation. In Proc. KDD, pages 219-222, 1997. An extended version of this paper appears as: E. M. Knorr and R.T. Ng. A Unified Approach for Mining Outliers. In Proc. 7th CASCON, pages 236-248, 1997.

[Kno97] E. M. Knorr. On digital money and card technologies. Technical Report 97-02, University of British Columbia, 1997.

[MT96] H. Mannila and H. Toivonen. Discovering generalized episodes using minimal occurrences. In Proc. KDD, pages 146-151, 1996.

[MTV95] H. Mannila, H. Toivonen, and A. Verkamo. Discovering frequent episodes in sequences. In Proc. KDD, pages 210-215, 1995.

[NH94] R. Ng and J. Han. Efficient and effective clustering methods for spatial data mining. In Proc. 20th VLDB, pages 144-155, 1994.

[PS88] F. Preparata and M. Shamos. Computational Geometry: an Introduction. Springer-Verlag, 1988.

[RR96] I. Ruts and P. Rousseeuw. Computing depth contours of bivariate point clouds. Computational Statistics and Data Analysis, 23:153-168, 1996.

[Sam90] H. Samet. The Design and Analysis of Spatial Data Structures. Addison-Wesley, 1990.

[ZRL96] T. Zhang, R. Ramakrishnan, and M. Livny. BIRCH: An efficient data clustering method for very large databases. In Proc. ACM SIGMOD, pages 103-114, 1996.