# Influence Sets Based on Rev erse Nearest Neiglbor Queries

# 基于反向最近邻查询的影响集

Flip Korn

弗利普·科恩

AT&T Labs-Research

美国电话电报公司实验室 - 研究部

flip@research.att.com

S. Muth ukrishnan

S. 穆图克里什南

AT&T Labs-Research

美国电话电报公司实验室 - 研究部

muthu@research.att.com

## Abstract

## 摘要

Inherent in the operation of many decision support and continuous referral systems is the notion of the "influence" of a data point on the database. This notion arises in examples such as finding the set of customers affected by the opening of a new store outlet location, notifying the subset of subscribers to a digital library who will find a newly added document most relevant, etc. Standard approaches to determining the influence set of a data point involve range searching and nearest neighbor queries.

在许多决策支持和连续推荐系统的运行中，数据点对数据库的“影响”概念是固有的。这一概念在很多例子中都会出现，比如找出受新店铺开业影响的客户群体，通知数字图书馆中最可能对新添加文档感兴趣的订阅用户子集等。确定数据点影响集的标准方法涉及范围搜索和最近邻查询。

In this paper, we formalize a novel notion of influence based on reverse neighbor queries and its variants. Since the nearest neighbor relation is not symmetric, the set of points that are closest to a query point (i.e., the nearest neighbors) differs from the set of points that have the query point as their nearest neighbor (called the reverse nearest neighbors). Influence sets based on reverse nearest neighbor (RNN) queries seem to capture the intuitive notion of influence from our motivating examples.

在本文中，我们基于反向最近邻查询及其变体，对一种新颖的影响概念进行了形式化定义。由于最近邻关系是非对称的，与查询点距离最近的点集（即最近邻）不同于以查询点为其最近邻的点集（称为反向最近邻）。基于反向最近邻（RNN）查询的影响集似乎能捕捉到我们动机示例中直观的影响概念。

We present a general approach for solving RNN queries and an efficient R-tree based method for large data sets, based on this approach. Although the RNN query appears to be natural, it has not been studied previously. RNN queries are of independent interest, and as such should be part of the suite of available queries for processing spatial and multimedia data. In our experiments with real geographical data, the proposed method appears to scale logarithmically, whereas straightforward sequential scan scales linearly. Our experimental study also shows that approaches based on range searching or nearest neighbors are ineffective at finding influence sets of our interest.

我们提出了一种解决RNN查询的通用方法，并基于此方法为大数据集提出了一种高效的基于R树的方法。尽管RNN查询看似很自然，但此前尚未有人对其进行研究。RNN查询本身就很有研究价值，因此应该成为处理空间和多媒体数据可用查询套件的一部分。在我们使用真实地理数据进行的实验中，所提出的方法似乎具有对数级的扩展性，而直接的顺序扫描则是线性扩展。我们的实验研究还表明，基于范围搜索或最近邻的方法在寻找我们感兴趣的影响集方面效果不佳。

## 1 Introduction

## 1 引言

A fundamental task that arises in various marketing and decision support systems is to determine the "influence" of a data point on the database, for example, the influence of a new store outlet or the influence of a new document to a repository. The concept of influence depends on the application at hand and is often difficult to formalize. We first develop an intuitive notion of influence sets through examples to motivate our formalization of it. The following two examples are drawn from spatial domains.

在各种营销和决策支持系统中出现的一项基本任务是确定数据点对数据库的“影响”，例如，新店铺的影响或新文档对知识库的影响。影响的概念取决于具体的应用，并且通常难以进行形式化定义。我们首先通过示例来形成影响集的直观概念，以此为我们对其进行形式化定义提供动机。以下两个示例来自空间领域。

Example 1 (Decision Support Systems): There are many factors that may contribute to a clientele adopting one outlet over another, but a simple premise is to base it on the geographical proximity to the customers. Consider a marketing application in which the issue is to determine the business impact of opening an outlet of Company $A$ at a given location. A simple task is to determine the segment of $A$ ’s customers who would be likely to use this new facility. Alternatively, one may wish to determine the segment of customers of Company $B$ (say $A$ ’s competitor) who are likely to find the new facility more convenient than the locations of $B$ . Such segments of customers are loosely what we would like to refer to as influence sets.

示例1（决策支持系统）：有许多因素可能导致客户选择一家店铺而非另一家，但一个简单的前提是基于与客户的地理接近程度。考虑一个营销应用，其问题是确定在给定位置开设公司$A$的一家店铺所产生的商业影响。一个简单的任务是确定$A$的哪些客户可能会使用这家新店铺。或者，人们可能希望确定公司$B$（假设是$A$的竞争对手）的哪些客户可能会觉得新店铺比$B$的店铺更方便。我们大致将这些客户群体称为影响集。

Example 2 (Continuous Referral Systems): Consider a referral service wherein a user can specify a street address, and the system returns a list of the five closest FedEx ${}^{TM}$ drop-off locations. ${}^{1}$ A responsible referral service may wish to give the option (e.g., by clicking a button) to make this a continuous query, that is, to request the system to notify the user when this list changes. The referral service will then notify those users whose list changes due to the opening of a closer FedEx drop-off location or the closing of an existing one. When such an event happens, the users who need to be updated correspond to our notion of the influence set of the added or dropped location.

示例2（连续推荐系统）：考虑一个推荐服务，用户可以指定一个街道地址，系统会返回五个最近的联邦快递${}^{TM}$投递点列表。${}^{1}$一个负责的推荐服务可能会提供一个选项（例如，通过点击一个按钮）将其设置为连续查询，即请求系统在该列表发生变化时通知用户。当有更近的联邦快递投递点开业或现有投递点关闭时，推荐服务将通知那些列表发生变化的用户。当发生此类事件时，需要更新信息的用户对应于我们所定义的新增或关闭投递点的影响集。

Both examples above reinforce the notion of the influence set of a data point in terms of geographical proximity. This concept of influence sets is inherent in many other decision support situations and referral services for which there is no underlying spatial or geographical distance, but for which there is a notion of similarity based on the vector space model (in which "distance" between vectors is taken as a measure of dissimilarity). The following two examples provide illustration.

上述两个示例都强化了基于地理接近程度的数据点影响集的概念。这种影响集的概念在许多其他决策支持场景和推荐服务中也很常见，这些场景和服务可能没有潜在的空间或地理距离，但基于向量空间模型存在相似性概念（其中向量之间的“距离”被用作不相似性的度量）。以下两个示例对此进行了说明。

---

<!-- Footnote -->

Permission to make digital or hard copies of part or all of this work or personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. To copy otherwise, to republish, to post on servers, or to redistribute to lists, requires prior specific permission and/or a fee.

允许个人或课堂使用本作品的部分或全部内容制作数字或硬拷贝，无需付费，但前提是这些拷贝不得用于盈利或商业目的，并且拷贝必须带有此声明和第一页的完整引用信息。否则，如需复制、重新发布、在服务器上发布或分发给列表，需要事先获得特定许可和/或支付费用。

MOD 2000, Dallas, TX USA

2000年多媒体与网络数据管理国际会议，美国得克萨斯州达拉斯市

© ACM 2000 1-58113-218-2/00/05 . . .\$5.00

© 美国计算机协会（ACM）2000 1-58113-218-2/00/05 ... 5.00美元

${}^{1}$ See http://www.fedex.com/us/dropoff for a realization of this.

${}^{1}$ 有关此实现，请访问 http://www.fedex.com/us/dropoff。

<!-- Footnote -->

---

Example 3 (Profile-based Marketing): A company may wish to keep profiles of its customers' interests so that it can gear a new service towards most customers. For example, suppose AT&T launches a new wireless service. The service may be abstracted a feature vector (e.g., covers New England area, free local calling on weekends, best for $\$ {100}$ -per-month users). The issue is which customers will find this the most suitable plan for their calling patterns; these customers form the influence set of the new service. One approach is to identify such users based on the distance between their profiles and the feature vector representing the new service.

示例 3（基于用户画像的营销）：一家公司可能希望保留其客户的兴趣画像，以便能够针对大多数客户推出新服务。例如，假设美国电话电报公司（AT&T）推出了一项新的无线服务。该服务可以抽象为一个特征向量（例如，覆盖新英格兰地区，周末本地通话免费，最适合每月$\$ {100}$费用的用户）。问题在于哪些客户会认为这个套餐最适合他们的通话模式；这些客户构成了新服务的影响集。一种方法是根据客户画像与代表新服务的特征向量之间的距离来识别这些用户。

Example 4 (Maintaining Document Repositories): Consider a repository of technical reports. When a new report is filed, it may be desirable to alert the authors of other TRs who would likely find the document interesting based on similarity to their publications; the set of all such authors corresponds to the notion of influence set we have been developing so far. Here, the influence set is defined based on the similarity between text documents which has been well-explored in the Information Retrieval community. Other similar scenarios abound, such as in a repository of Web pages, precendent legal cases, etc.

示例 4（维护文档库）：考虑一个技术报告库。当提交一份新报告时，可能希望提醒其他技术报告（TR）的作者，这些作者可能会基于与他们出版物的相似性而对该文档感兴趣；所有这些作者的集合对应于我们到目前为止所发展的影响集概念。在这里，影响集是基于文本文档之间的相似性来定义的，这在信息检索领域已经得到了深入研究。其他类似的场景比比皆是，例如网页库、先例法律案件等。

Let us now make the notion of an influence set more precise. We start with a data set $S$ ,some suitable definition of distance between points in $S$ ,and a query point $q$ ; the goal is to find the subset of points in $S$ influenced by $q$ . Two suggestions present themselves immediately. The first is to use range queries wherein one specifies a threshold radius $\epsilon$ from $q$ ,and all points within $\epsilon$ are returned. The second is to use the well known concept of nearest neighbors (NN), or, more generally, $k$ -nearest neighbors wherein one specifies $k$ , and the $k$ closest points to $q$ are returned.

现在让我们更精确地定义影响集的概念。我们从一个数据集$S$、$S$中各点之间某种合适的距离定义以及一个查询点$q$开始；目标是找到$S$中受$q$影响的点的子集。有两个建议立即浮现出来。第一个是使用范围查询，其中指定一个以$q$为中心的阈值半径$\epsilon$，并返回$\epsilon$范围内的所有点。第二个是使用众所周知的最近邻（NN）概念，或者更一般地，$k$ -最近邻，其中指定$k$，并返回距离$q$最近的$k$个点。

Both of these suggestions fall short of capturing the intuitive notion of influence we have so far developed. In both cases, parameters have to be engineered to yield an appropriate result size, and it is not obvious how to choose a value without a priori knowledge of the local density of points. Range queries may be appropriate for other notions of influence (e.g., the opening of a toxic waste dump on its surrounding population) but not for what is required in the examples given above. NN queries are commonly used in domains which call for searching based on proximity; however, they are not appropriate in this context for similar reasons. Consider Example 1, in which one wants to find potential customers for a new store outlet $q$ . The deciding factor is not how close a customer is to $q$ ,but rather if the customer is further from every other store than from $q$ . Thus,it may very well be the case that potential customers lie outside a small radius from $q$ ,or are further from $q$ than the first few nearest neighbors. Expanding the search radius will not necessarily work around this problem. Although it may encompass more customers who are likely to be influenced by $q$ ,it may do so at the trade-off of introducing many customers who are not in the influence set (i.e., customers whose closest store is not $q$ ). Later,we will make these discussions more concrete and present quantitative measures of comparison (see Section 6).

这两个建议都未能捕捉到我们到目前为止所发展的直观影响概念。在这两种情况下，都必须设计参数以产生合适的结果规模，并且在没有点的局部密度先验知识的情况下，不清楚如何选择值。范围查询可能适用于其他影响概念（例如，有毒废物倾倒场对其周边人口的影响），但不适用于上述示例中的需求。最近邻查询通常用于需要基于接近度进行搜索的领域；然而，出于类似的原因，它们在这种情况下并不合适。考虑示例 1，其中希望为新的商店分店$q$找到潜在客户。决定因素不是客户与$q$的距离有多近，而是客户与$q$的距离是否比与其他任何商店的距离更近。因此，很可能潜在客户位于距离$q$较小半径之外，或者比前几个最近邻距离$q$更远。扩大搜索半径不一定能解决这个问题。虽然它可能会涵盖更多可能受$q$影响的客户，但这样做可能会引入许多不在影响集内的客户（即，距离他们最近的商店不是$q$的客户）。稍后，我们将使这些讨论更加具体，并提出定量的比较指标（见第 6 节）。

We address these shortcomings and develop a notion of influence set with broad applications. A fundamental observation which is the basis for our work here is that the nearest neighbor relation is not symmetric. For example,if $p$ is the nearest neighbor of $q$ ,then $q$ need not be the nearest neighbor of $p$ (see Figure 1). ${}^{2}$ Note that this is the case even though the underlying distance function is Euclidean and, hence, symmetric. Similarly, the $k$ -nearest neighbor relation is not symmetric. It follows that,for a given query point $q$ ,the nearest neighbors of $q$ may differ substantially from the set of all points for which $q$ is a nearest neighbor. We call these points the reverse nearest neighbors of $q$ .

我们解决了这些缺点，并发展了一个具有广泛应用的影响集概念。作为我们在此工作基础的一个基本观察结果是，最近邻关系不是对称的。例如，如果$p$是$q$的最近邻，那么$q$不一定是$p$的最近邻（见图 1）。${}^{2}$ 请注意，即使底层距离函数是欧几里得距离，因此是对称的，情况也是如此。类似地，$k$ -最近邻关系也不是对称的。由此可知，对于给定的查询点$q$，$q$的最近邻可能与所有以$q$为最近邻的点的集合有很大不同。我们将这些点称为$q$的反向最近邻。

<!-- Media -->

<img src="https://cdn.noedgeai.com/0195c909-9cbb-7185-8180-c4de2b24249b_1.jpg?x=1058&y=1276&w=445&h=255&r=0"/>

Figure 1: Nearest neighbors need not be symmetric: the NN of $q$ is $p$ ,whereas the NN of $p$ is $r$ . (An arrow from point $i$ to point $j$ indicates that $j$ is the nearest neighbor of ${i}_{ \cdot  }$ )

图 1：最近邻不一定是对称的：$q$的最近邻是$p$，而$p$的最近邻是$r$。（从点$i$指向点$j$的箭头表示$j$是${i}_{ \cdot  }$的最近邻）

<!-- Media -->

We now summarize our contributions:

现在我们总结一下我们的贡献：

- We identify a natural and broadly applicable notion for the "influence" of a data point on the database (namely, the influence set), and formalize it based on reverse nearest neighbors (RNN) and its variants (such as reverse $k$ -nearest neighbors,reverse furthest neighbor, etc.);

- 我们确定了一个自然且广泛适用的关于数据点对数据库“影响”的概念（即影响集），并基于反向最近邻（RNN）及其变体（如反向 $k$ -最近邻、反向最远邻等）对其进行形式化定义；

---

<!-- Footnote -->

${}^{2}$ That is,provided there are other points in the collection.

${}^{2}$ 即，假设集合中存在其他点。

<!-- Footnote -->

---

- We present a general approach for determining reverse nearest neighbors. Our approach is geometric, reducing the problem to that of testing the enclosure of points in geometric objects; it works for different distance functions and variants of RNNs. Although the RNN query appears to be natural, it has not been studied previously. RNN queries are of independent interest, and as such should be part of the suite of available queries for processing spatial and multimedia data;

- 我们提出了一种确定反向最近邻的通用方法。我们的方法是基于几何的，将问题简化为测试点是否包含在几何对象中；它适用于不同的距离函数和反向最近邻的变体。尽管反向最近邻查询看似自然，但此前尚未被研究过。反向最近邻查询具有独立的研究价值，因此应成为处理空间和多媒体数据可用查询套件的一部分；

- Based on our approach, we propose efficient and scalable R-tree based methods for implementing reverse nearest neighbor queries. We also perform an experimental study of the I/O-efficiency of the proposed R-tree based methods. Using our approach, we show in terms of standard precision and recall measures to assess the output quality, that well known database queries (range and nearest neighbor queries) are not effective in finding influence sets.

- 基于我们的方法，我们提出了基于 R - 树的高效可扩展方法来实现反向最近邻查询。我们还对所提出的基于 R - 树的方法的 I/O 效率进行了实验研究。使用我们的方法，通过标准的精确率和召回率指标来评估输出质量，我们表明已知的数据库查询（范围查询和最近邻查询）在查找影响集方面并不有效。

The structure of the paper is as follows. Section 2 defines RNN queries and describes its relationship to NN queries. Section 3 presents an approach and algorithmic framework for answering RNN queries; we also propose a scalable method for implementating this framework using R-trees in Section 4. Section 5 gives empirical results from experiments for RNN queries. In Section 6, we formalize the basic notion of influence sets based on RNN queries and give results from a qualitative study of the effectiveness of well known queries to substitute for RNN queries. Then we develop the variants of RNN queries needed for generalized notions of influence sets. Section 7 reviews the related work. Section 8 lists the conclusions and gives directions for future work.

本文的结构如下。第 2 节定义了反向最近邻查询，并描述了它与最近邻查询的关系。第 3 节提出了一种回答反向最近邻查询的方法和算法框架；我们还在第 4 节中提出了一种使用 R - 树实现该框架的可扩展方法。第 5 节给出了反向最近邻查询的实验结果。在第 6 节中，我们基于反向最近邻查询对影响集的基本概念进行形式化定义，并给出了关于用已知查询替代反向最近邻查询有效性的定性研究结果。然后我们开发了广义影响集概念所需的反向最近邻查询的变体。第 7 节回顾了相关工作。第 8 节列出了结论并给出了未来工作的方向。

## 2 Reverse Nearest Neighbor Queries

## 2 反向最近邻查询

Reverse nearest neighbor (RNN) queries are the basis for influence sets, and are also of independent interest. We define and develop them in this section. We start from the definition of the nearest neighbor (NN) query, a standard query in spatial and multimedia databases and define the RNN query and its variants based on this. We will develop the underlying concepts in two dimensions for simplicity; there will be no difficulty in extending them to higher dimensions. In our discussion, we shall assume the distance between any two points $p = \left( {{p}_{x},{p}_{y}}\right)$ and $q = \left( {{q}_{x},{q}_{y}}\right)$ is $d\left( {p,q}\right)  = {\left( {q}_{x} - {p}_{x}\right) }^{2} +$ ${\left( {q}_{y} - {p}_{y}\right) }^{2}$ ,known as the Euclidean,or ${L}_{2}$ ,distance. ${}^{3}$

反向最近邻（RNN）查询是影响集的基础，也具有独立的研究价值。我们在本节中对其进行定义和研究。我们从最近邻（NN）查询的定义开始，最近邻查询是空间和多媒体数据库中的标准查询，并在此基础上定义反向最近邻查询及其变体。为了简单起见，我们将在二维空间中阐述相关概念；将其扩展到更高维度不会有困难。在我们的讨论中，我们假设任意两点 $p = \left( {{p}_{x},{p}_{y}}\right)$ 和 $q = \left( {{q}_{x},{q}_{y}}\right)$ 之间的距离为 $d\left( {p,q}\right)  = {\left( {q}_{x} - {p}_{x}\right) }^{2} +$ ${\left( {q}_{y} - {p}_{y}\right) }^{2}$ ，即欧几里得距离，或 ${L}_{2}$ 距离。${}^{3}$

### 2.1 Formal Definitions

### 2.1 形式化定义

Suppose we have a collection $S$ of points in the plane. For a nearest neighbor query, we are given a query point $q$ ,and the goal is to determine the nearest neighbor set $\mathcal{N}\mathcal{N}\left( q\right)$ defined as

假设我们有一个平面上的点集 $S$ 。对于最近邻查询，我们给定一个查询点 $q$ ，目标是确定最近邻集 $\mathcal{N}\mathcal{N}\left( q\right)$ ，其定义为

$$
\mathcal{N}\mathcal{N}\left( q\right)  = \{ r \in  S \mid  \forall p \in  S : d\left( {q,r}\right)  \leq  d\left( {q,p}\right) \} .
$$

Our focus here is on the inverse relation among the points. Given any query point $q$ ,we need to determine the set $\mathcal{{RNN}}\left( q\right)$ of reverse nearest neighbors,defined as

我们这里关注的是点之间的逆关系。给定任意查询点 $q$ ，我们需要确定反向最近邻集 $\mathcal{{RNN}}\left( q\right)$ ，其定义为

$$
\mathcal{R}\mathcal{N}\mathcal{N}\left( q\right)  = \{ r \in  S \mid  \forall p \in  S : d\left( {r,q}\right)  \leq  d\left( {r,p}\right) \} .
$$

$\mathcal{{RNN}}\left( q\right)$ may be empty,or have one or more elements, and we may wish to return any one of them, or the entire list.

$\mathcal{{RNN}}\left( q\right)$ 可能为空，也可能有一个或多个元素，我们可能希望返回其中任意一个元素，或者整个列表。

### 2.2 Variants

### 2.2 变体

There are two variants of this basic scenario that are of interest to us. We will define only the variants for RNN queries, although the corresponding variants of NN queries may also be of interest.

我们感兴趣的这个基本场景有两个变体。我们仅定义反向最近邻查询的变体，不过最近邻查询的相应变体可能也有研究价值。

- Monochromatic vs Bichromatic. In some applications, the points in $S$ are of two different categories,such as clients and servers; the points may therefore be thought of as being colored red or blue. The RNN query now consists of a point in one of the categories, say blue, and must determine the red points for which the query point is the closest blue point. Formally,let $B$ denote the set of blue points and $R$ the set of red points. Consider a blue query point $q$ . We have,

- 单色与双色。在某些应用中，$S$ 中的点分为两种不同类别，例如客户端和服务器；因此这些点可以被视为红色或蓝色。现在反向最近邻查询由其中一个类别中的一个点（例如蓝色点）组成，并且必须确定对于该查询点是最近蓝色点的红色点。形式上，设 $B$ 表示蓝色点集，$R$ 表示红色点集。考虑一个蓝色查询点 $q$ 。我们有，

$$
\mathcal{R}\mathcal{N}\mathcal{N}\left( q\right)  = \{ r \in  R \mid  \forall p \in  B : d\left( {r,q}\right)  \leq  d\left( {r,p}\right) \} .
$$

We call this the bichromatic version; in contrast, the basic scenario above wherein all points were of the same category is the monochromatic version. Both versions of the problem are of interest.

我们称这个为双色版本；相比之下，上述所有点都属于同一类别的基本场景是单色版本。这两个版本的问题都值得研究。

At first look, the mono and bichromatic versions of the RNN problem seem very similar. For a blue query point, we consider only the red points and their distance to the closest blue point (vice versa for the red query points). However, at a deeper level, there is a fundamental difference. Let us focus on the ${L}_{2}$ case.

乍一看，RNN问题的单色和双色版本似乎非常相似。对于一个蓝色查询点，我们只考虑红色点以及它们到最近蓝色点的距离（对于红色查询点则反之）。然而，从更深层次来看，存在一个根本的区别。让我们聚焦于${L}_{2}$的情况。

Proposition 1 For any query point, $\mathcal{{RNN}}\left( q\right)$ may have at most 6 points in the monochromatic case; in the bichromatic case,the size of the set $\mathcal{{RNN}}\left( q\right)$ may be unbounded.

命题1 对于任何查询点，在单色情况下，$\mathcal{{RNN}}\left( q\right)$最多可能有6个点；在双色情况下，集合$\mathcal{{RNN}}\left( q\right)$的大小可能是无界的。

A proof of this may be found in [17]. From a combinatorial viewpoint, the output of RNN queries is bounded; this in turn affects the efficiency because a RNN query is output-sensitive. This entire phenomenon is not restricted to the plane (e.g., in three dimensions, the $\mathcal{{RNN}}\left( q\right)$ contains at most 12 points under ${L}_{2}$ distance and so on),or the distance function (e.g.,in the ${L}_{\infty }$ case,the cardinality of $\mathcal{R}\mathcal{N}\mathcal{N}\left( q\right)$ is at most ${3}^{d} - 1$ in $d$ dimensions).

此命题的证明可在文献[17]中找到。从组合学的角度来看，RNN查询的输出是有界的；这反过来又会影响效率，因为RNN查询对输出敏感。这一整个现象并不局限于平面（例如，在三维空间中，在${L}_{2}$距离下，$\mathcal{{RNN}}\left( q\right)$最多包含12个点，依此类推），也不局限于距离函数（例如，在${L}_{\infty }$的情况下，在$d$维空间中，$\mathcal{R}\mathcal{N}\mathcal{N}\left( q\right)$的基数最多为${3}^{d} - 1$）。

---

<!-- Footnote -->

${}^{3}$ Other ${L}_{p}$ distances may also be interest,for example ${L}_{1}$ where $d\left( {p,q}\right)  = \left| {{q}_{x} - {p}_{x}}\right|  + \left| {{q}_{y} - {p}_{y}}\right|$ or ${L}_{\infty }$ where $d\left( {p,q}\right)  =$ $\max \left\{  {\left| {{q}_{x} - {p}_{x}}\right| ,\left| {{q}_{y} - {p}_{y}}\right| }\right\}  .$

${}^{3}$ 其他${L}_{p}$距离可能也会令人感兴趣，例如${L}_{1}$，其中$d\left( {p,q}\right)  = \left| {{q}_{x} - {p}_{x}}\right|  + \left| {{q}_{y} - {p}_{y}}\right|$ 或者${L}_{\infty }$，其中$d\left( {p,q}\right)  =$ $\max \left\{  {\left| {{q}_{x} - {p}_{x}}\right| ,\left| {{q}_{y} - {p}_{y}}\right| }\right\}  .$

<!-- Footnote -->

---

- Static vs Dynamic. Sometimes we wish to insert or delete points from the set $S$ and still support the RNN query; we refer to this as the dynamic case. In contrast, the case when set $S$ is not modified is called the static case. The dynamic case is relevant in most applications. The crux here, as in all dynamic problems, is to be able to handle insertions and deletions efficiently without rebuilding the entire data structure.

- 静态与动态。有时我们希望从集合$S$中插入或删除点，同时仍能支持RNN查询；我们将这种情况称为动态情况。相比之下，集合$S$不被修改的情况称为静态情况。动态情况在大多数应用中都很重要。与所有动态问题一样，这里的关键在于能够高效地处理插入和删除操作，而无需重建整个数据结构。

## 3 Our Approach to RNN Queries

## 3 我们解决RNN查询的方法

Our approach for solving the reverse nearest neighbors query problem is quite general, and it applies also to its variants as we shall see.

我们解决反向最近邻查询问题的方法相当通用，并且正如我们将看到的，它也适用于该问题的变体。

### 3.1 Static Case

### 3.1 静态情况

For exposition, let us consider a basic version of the problem. We are given a set $S$ of points which is not updated, and the distance between any two points is measured using Euclidean distance. Our approach involves two steps.

为了便于阐述，让我们考虑该问题的一个基本版本。我们给定一个不更新的点集$S$，并且使用欧几里得距离来度量任意两点之间的距离。我们的方法包括两个步骤。

Step 1. For each point $p \in  S$ ,determine the distance to the nearest neighbor of $p$ in $S$ ,denoted $N\left( p\right)$ . Formally, $N\left( p\right)  = \mathop{\min }\limits_{{q \in  S-\{ p\} }}d\left( {p,q}\right)$ . For each $p \in  S$ ,generate a circle $\left( {p,N\left( p\right) }\right)$ where $p$ is its center and $N\left( p\right)$ its radius. (See Figure 2(a) for an illustration.)

步骤1. 对于每个点$p \in  S$，确定$p$在$S$中到其最近邻的距离，记为$N\left( p\right)$。形式上，$N\left( p\right)  = \mathop{\min }\limits_{{q \in  S-\{ p\} }}d\left( {p,q}\right)$。对于每个$p \in  S$，生成一个圆$\left( {p,N\left( p\right) }\right)$，其中$p$是圆心，$N\left( p\right)$是半径。（如图2(a)所示。）

Step 2. For any query $q$ ,determine all the circles $\left( {p,N\left( p\right) }\right)$ that contain $q$ and return their centers $p$ .

步骤2. 对于任何查询$q$，确定所有包含$q$的圆$\left( {p,N\left( p\right) }\right)$，并返回它们的圆心$p$。

We have not yet described how to perform the two steps above, but we will first prove that they suffice.

我们尚未描述如何执行上述两个步骤，但我们将首先证明这两个步骤就足够了。

Lemma 1 Step 2 determines precisely all the reverse nearest neighbors of $q$ .

引理1 步骤2精确地确定了$q$的所有反向最近邻。

Proof. If point $p$ is returned from Step 2,then $q$ falls within the circle $\left( {p,N\left( p\right) }\right)$ . Therefore,the distance $d\left( {p,q}\right)$ is smaller than the radius $N\left( p\right)$ . In other words, $d\left( {p,q}\right)  \leq  N\left( p\right)$ and hence $q$ is the nearest neighbor of $p$ (equivalently, $p$ is a reverse nearest neighbor of $q$ ). Conversely,if $p$ is the reverse nearest neighbor of $q$ , $d\left( {p,q}\right)  \leq  N\left( p\right)$ and,therefore, $q$ lies within the circle $\left( {p,N\left( p\right) }\right)$ . Hence, $p$ will be found in Step 2.

证明。如果点 $p$ 是从步骤 2 返回的，那么 $q$ 落在圆 $\left( {p,N\left( p\right) }\right)$ 内。因此，距离 $d\left( {p,q}\right)$ 小于半径 $N\left( p\right)$。换句话说，$d\left( {p,q}\right)  \leq  N\left( p\right)$，因此 $q$ 是 $p$ 的最近邻（等价地，$p$ 是 $q$ 的反向最近邻）。相反，如果 $p$ 是 $q$ 的反向最近邻，那么 $d\left( {p,q}\right)  \leq  N\left( p\right)$，因此 $q$ 位于圆 $\left( {p,N\left( p\right) }\right)$ 内。因此，$p$ 将在步骤 2 中被找到。

What our approach has achieved is to reduce the problem of answering the reverse nearest neighbor query to the problem of finding all nearest neighbors (Step 1) and then to what is known in the literature as point enclosure problems wherein we need to determine all the objects that contain a query point (Step 2).

我们的方法所实现的是将回答反向最近邻查询的问题简化为找到所有最近邻的问题（步骤 1），然后简化为文献中已知的点包含问题，在该问题中我们需要确定所有包含查询点的对象（步骤 2）。

Our approach is attractive for two reasons. First, both steps are of independent interest and have been studied in the literature. They have efficient solutions, as we will see later. Second, our approach extends to the variants of our interest as we show below. Other distance functions. If the distance function is ${L}_{\infty }$ rather than ${L}_{2}$ ,we generate squares $\left( {p,N\left( p\right) }\right)$ in Step 1 with center $p$ and sides ${2N}\left( p\right)$ . (See Figure 2(b) for an illustration.) Similarly,for other ${L}_{p}$ distance functions, we will have suitable geometric shapes.

我们的方法有两个吸引人的原因。首先，这两个步骤本身都很有趣，并且在文献中已经得到了研究。正如我们稍后将看到的，它们有高效的解决方案。其次，正如我们下面所示，我们的方法可以扩展到我们感兴趣的变体。其他距离函数。如果距离函数是 ${L}_{\infty }$ 而不是 ${L}_{2}$，我们在步骤 1 中生成以 $p$ 为中心、边长为 ${2N}\left( p\right)$ 的正方形 $\left( {p,N\left( p\right) }\right)$。（如图 2(b) 所示。）类似地，对于其他 ${L}_{p}$ 距离函数，我们将有合适的几何形状。

Bichromatic version. Consider only blue query points for now. We perform the two steps above only for the red points in set $S$ . For each red point $p \in  S$ , we determine $N\left( p\right)$ ,the distance to the nearest blue neighbor. The rest of the description above remains unchanged. We also process for red query points analogously.

双色版本。目前仅考虑蓝色查询点。我们仅对集合 $S$ 中的红色点执行上述两个步骤。对于每个红色点 $p \in  S$，我们确定 $N\left( p\right)$，即到最近蓝色邻居的距离。上述描述的其余部分保持不变。我们也以类似的方式处理红色查询点。

### 3.2 Dynamic Case

### 3.2 动态情况

Our description above was for the static case only. For the dynamic case, we need to make some modifications. Below we assume the presence of a (dynamically maintained) data structure for answering NN queries. Recall the definition of $N\left( p\right)$ for point $p$ from the previous section. Consider an insertion of a point $q$ (as illustrated in Figure 3(a)):

我们上面的描述仅针对静态情况。对于动态情况，我们需要进行一些修改。下面我们假设存在一个（动态维护的）用于回答最近邻（NN）查询的数据结构。回顾上一节中关于点 $p$ 的 $N\left( p\right)$ 的定义。考虑插入一个点 $q$（如图 3(a) 所示）：

1. Determine the reverse nearest neighbors $p$ of $q$ . For each such point $p$ ,we replace circle $\left( {p,N\left( p\right) }\right)$ with $\left( {p,d\left( {p,q}\right) }\right)$ ,and update $N\left( p\right)$ to equal $d\left( {p,q}\right)$ ;

1. 确定 $q$ 的反向最近邻 $p$。对于每个这样的点 $p$，我们用 $\left( {p,d\left( {p,q}\right) }\right)$ 替换圆 $\left( {p,N\left( p\right) }\right)$，并将 $N\left( p\right)$ 更新为等于 $d\left( {p,q}\right)$；

2. Find $N\left( q\right)$ ,the distance of $q$ from its nearest neighbor,and add $\left( {q,N\left( q\right) }\right)$ to the collection of circles.

2. 找到 $N\left( q\right)$，即 $q$ 到其最近邻的距离，并将 $\left( {q,N\left( q\right) }\right)$ 添加到圆的集合中。

Lemma 2 The insertion procedure is correct.

引理 2 插入过程是正确的。

Proof. It suffices to argue that,for each point $p,N\left( p\right)$ is the correct distance of $p$ to its nearest neighbor after an insertion. This clearly holds for the inserted point $q$ from Step 2. Among the rest of the points, the only ones which will be affected are those which have $q$ as their nearest neighbor, in other words, the reverse nearest neighbors of $q$ . For all such points $p$ ,we update their $N\left( p\right)$ ’s appropriately in Step 1. The remaining points $p$ do not change $N\left( p\right)$ as a result of inserting $q$ . Hence, all points $p$ have the correct value of $N\left( p\right)$ .

证明。只需证明对于每个点 $p,N\left( p\right)$ 是插入后 $p$ 到其最近邻的正确距离即可。对于步骤 2 中插入的点 $q$，这显然成立。在其余的点中，唯一会受到影响的是那些以 $q$ 为最近邻的点，换句话说，就是 $q$ 的反向最近邻。对于所有这样的点 $p$，我们在步骤 1 中适当地更新它们的 $N\left( p\right)$。由于插入 $q$，其余的点 $p$ 的 $N\left( p\right)$ 不会改变。因此，所有点 $p$ 的 $N\left( p\right)$ 都有正确的值。

Step 1 is shown in Figure 3(b) where we shrink all circles $\left( {p,N\left( p\right) }\right)$ for which $q$ is the nearest neighbor of $p$ to $\left( {p,d\left( {p,q}\right) }\right)$ . Step 2 is shown in Figure 3(c).

步骤1如图3(b)所示，我们将所有满足$q$是$p$的最近邻的圆$\left( {p,N\left( p\right) }\right)$缩小到$\left( {p,d\left( {p,q}\right) }\right)$。步骤2如图3(c)所示。

Now consider an deletion of a point $q$ (as illustrated in Figure 4(a)):

现在考虑删除一个点$q$（如图4(a)所示）：

1. We need to remove the circle $\left( {q,N\left( q\right) }\right)$ from the collection of circles (see Figure 4(b));

1. 我们需要从圆的集合中移除圆$\left( {q,N\left( q\right) }\right)$（见图4(b)）；

2. Determine all the reverse nearest neighbors $p$ of $q$ . For each such point $p$ ,determine its current $N\left( p\right)$ and replace its existing circle with $\left( {p,N\left( p\right) }\right)$ .

2. 确定$q$的所有反向最近邻$p$。对于每个这样的点$p$，确定其当前的$N\left( p\right)$，并用$\left( {p,N\left( p\right) }\right)$替换其现有的圆。

We can argue much as before that the deletion procedure is correct. The crucial observation is that the only existing circles $\left( {p,N\left( p\right) }\right)$ that get affected are those that have $q$ on the circumference,that is,those associated with the reverse nearest neighbors of $q$ ; their circles get expanded in Step 1 (see Figure 4(c)). The details for how to extend these algorithms to other distance functions and to the bichromatic version are similar to those given in the previous section.

我们可以像之前一样论证删除过程是正确的。关键的观察结果是，唯一受影响的现有圆$\left( {p,N\left( p\right) }\right)$是那些圆周上有$q$的圆，即与$q$的反向最近邻相关的圆；它们的圆在步骤1中会扩大（见图4(c)）。如何将这些算法扩展到其他距离函数以及双色版本的细节与上一节给出的类似。

<!-- Media -->

<!-- figureText: (a) ${L}_{2}$ case (b) ${L}_{\infty }$ case -->

<img src="https://cdn.noedgeai.com/0195c909-9cbb-7185-8180-c4de2b24249b_4.jpg?x=340&y=179&w=1114&h=509&r=0"/>

Figure 2: A point set and its nearest neighborhoods.

图2：一个点集及其最近邻域。

<!-- figureText: (a) find $\mathcal{{RNN}}\left( q\right)$ (c) find $\mathcal{N}\mathcal{N}\left( q\right)$ (b) shrink circles -->

<img src="https://cdn.noedgeai.com/0195c909-9cbb-7185-8180-c4de2b24249b_4.jpg?x=184&y=821&w=1424&h=437&r=0"/>

Figure 3: A geometrical illustration of the insertion algorithm.

图3：插入算法的几何图示。

<!-- Media -->

## 4 Scalable RNN Queries

## 4 可扩展的反向最近邻（RNN）查询

In this section we propose a scalable method for implementing RNN queries on large, out-of-core data sets, based on our approach from Section 3. Like NN queries, RNN queries are I/O-bound (as opposed to, e.g., spatial joins which are CPU-bound), and thus the focus is on I/O performance. Because R-trees [7, 2, 16] have been successfully deployed in spatial databases and because of their generality to support a variety of norms via bounding boxes, we use them in the proposed method. However, note that any spatial access method could be employed (see [6] for a recent survey of spatial access methods). Our deployment of R-trees is standard, but requires some elaboration. First we describe static RNN search; we then present details of the algorithms and data structures for the dynamic case.

在本节中，我们基于第3节的方法，提出一种可扩展的方法，用于在大型核外数据集上实现反向最近邻（RNN）查询。与最近邻（NN）查询一样，RNN查询是I/O受限的（与例如CPU受限的空间连接相反），因此重点在于I/O性能。由于R树[7, 2, 16]已成功应用于空间数据库，并且它们具有通过边界框支持各种范数的通用性，我们在提出的方法中使用它们。然而，请注意，可以采用任何空间访问方法（有关空间访问方法的最新综述，请参阅[6]）。我们对R树的部署是标准的，但需要一些详细说明。首先，我们描述静态RNN搜索；然后介绍动态情况下的算法和数据结构的细节。

### 4.1 Static Case

### 4.1 静态情况

The first step in being able to efficiently answer RNN queries is to precompute the nearest neighbor for each and every point. The problem of efficiently computing all-nearest neighbors in large data sets has been studied in $\left\lbrack  {3,8}\right\rbrack$ ,and thus we do not investigate it further in this paper. ${}^{4}$

能够高效回答RNN查询的第一步是为每个点预先计算最近邻。在大型数据集中高效计算所有最近邻的问题已在$\left\lbrack  {3,8}\right\rbrack$中进行了研究，因此我们在本文中不再进一步研究。${}^{4}$

Given a query point $q$ ,a straightforward but naive approach for finding reverse nearest neighbors is to sequentially scan through the entries $\left( {{p}_{i} \rightarrow  {p}_{j}}\right)$ of a precomputed all-NN list in order to determine which points ${p}_{i}$ are closer to $q$ than to ${p}_{i}$ ’s current nearest neighbor ${p}_{j}$ . Ideally,one would like to avoid having to sequentially scan through the data.

给定一个查询点$q$，一种直接但简单的查找反向最近邻的方法是顺序扫描预先计算的所有最近邻（all - NN）列表的条目$\left( {{p}_{i} \rightarrow  {p}_{j}}\right)$，以确定哪些点${p}_{i}$比${p}_{i}$当前的最近邻${p}_{j}$更接近$q$。理想情况下，人们希望避免顺序扫描数据。

---

<!-- Footnote -->

${}^{4}$ All-nearest neighbors is a special case of a spatial join.

${}^{4}$ 所有最近邻是空间连接的一个特殊情况。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: (a) remove $\mathcal{N}\mathcal{N}\left( q\right)$ (b) find $\mathcal{{RNN}}\left( q\right)$ (c) expand circles -->

<img src="https://cdn.noedgeai.com/0195c909-9cbb-7185-8180-c4de2b24249b_5.jpg?x=182&y=175&w=1426&h=443&r=0"/>

Figure 4: A geometrical illustration of the deletion algorithm.

图4：删除算法的几何图示。

<!-- Media -->

Based on the approach in Section 3, a RNN query reduces to a point enclosure query in a database of nearest neighborhood objects (e.g.,circles for ${L}_{2}$ distance in the plane); these objects can be obtained from the all-nearest neighbor distances. We propose to store the objects explicitly in an $\mathrm{R}$ -tree. Henceforth,we shall refer to this instantiation of an $\mathrm{R}$ -tree as an ${RNN}$ - tree. Thus, we can answer RNN queries by a simple search in the R-tree for those objects enclosing $q$ .

基于第3节的方法，RNN查询可简化为在最近邻域对象（例如，平面上${L}_{2}$距离的圆）数据库中的点包含查询；这些对象可以从所有最近邻距离中获得。我们建议将这些对象显式存储在$\mathrm{R}$树中。此后，我们将这种$\mathrm{R}$树的实例称为${RNN}$树。因此，我们可以通过在R树中简单搜索包含$q$的对象来回答RNN查询。

### 4.2 Dynamic Case

### 4.2 动态情况

As mentioned in Section 4.1, a sequential scan of a precomputed all-NN list can be used to determine the reverse nearest neighbors of a given point query $q$ . Insertion and deletion can be handled similarly. Even if this list were inverted, enabling deletion to be achieved in constant time by looking up the corresponding entry $\left( {{p}_{j} \rightarrow  \left\{  {{p}_{{i}_{1}},{p}_{{i}_{2}},\ldots ,{p}_{{i}_{k}}}\right\}  }\right)$ ,queries and insertions would still require a pass over the data. We would like to avoid having to do this.

如4.1节所述，对预计算的全最近邻（all-NN）列表进行顺序扫描可用于确定给定点查询 $q$ 的反向最近邻。插入和删除操作也可以类似地处理。即使该列表被反转，通过查找相应条目 $\left( {{p}_{j} \rightarrow  \left\{  {{p}_{{i}_{1}},{p}_{{i}_{2}},\ldots ,{p}_{{i}_{k}}}\right\}  }\right)$ 能在常数时间内完成删除操作，但查询和插入操作仍需要遍历数据。我们希望避免这样做。

We describe how to incrementally maintain the RNN-tree in the presence of insertions and deletions. To do this will require a supporting access method that can find nearest neighbors of points efficiently. At this point, one may wonder if a single R-tree will suffice for finding reverse nearest neighbors as well as nearest neighbors, in other words, if our RNN-tree can be used for this purpose, This turns out to be not the case since geometric objects rather than points are stored in the RNN-tree, and thus the bounding boxes are not optimized for nearest neighbor search performance on points. Therefore, we propose to use a separate R-tree for NN queries,henceforth referred to as an ${NN}$ -tree. Note that the NN-tree is not needed for static RNN queries, only for insertions and deletions, and that, in addition to the RNN-tree, it must be dynamically maintained.

我们将描述如何在存在插入和删除操作的情况下增量式维护反向最近邻树（RNN-tree）。为此，需要一种支持性的访问方法，能够高效地查找点的最近邻。此时，有人可能会想，单个R树是否足以同时用于查找反向最近邻和最近邻，换句话说，我们的反向最近邻树是否可用于此目的。事实并非如此，因为反向最近邻树中存储的是几何对象而非点，因此其边界框并非针对点的最近邻搜索性能进行优化。因此，我们建议使用一个单独的R树进行最近邻（NN）查询，此后将其称为 ${NN}$ 树。请注意，静态反向最近邻查询不需要最近邻树，仅在插入和删除操作时需要，并且除了反向最近邻树之外，它也必须进行动态维护。

<!-- Media -->

---

Algorithm Insert:

插入算法：

Input: point $q$

输入：点 $q$

		$\left\{  {{p}_{1},{p}_{2},\ldots ,{p}_{k}}\right\}   \leftarrow$ query $q$ in RNN-tree;

		在反向最近邻树中进行 $\left\{  {{p}_{1},{p}_{2},\ldots ,{p}_{k}}\right\}   \leftarrow$ 查询 $q$；

		for each ${p}_{i}$ (with corresponding ${R}_{i}$ ) do

		对于每个 ${p}_{i}$（对应 ${R}_{i}$）执行以下操作

				shrink ${R}_{i}$ to $\left( {{p}_{i},d\left( {{p}_{i},q}\right) }\right)$ ;

				将 ${R}_{i}$ 收缩为 $\left( {{p}_{i},d\left( {{p}_{i},q}\right) }\right)$；

		find $N\left( q\right)$ from NN-tree;

		从最近邻树中查找 $N\left( q\right)$；

		insert $q$ in NN-tree;

		将 $q$ 插入最近邻树；

		insert $\left( {q,N\left( q\right) }\right)$ in RNN-tree;

		将 $\left( {q,N\left( q\right) }\right)$ 插入反向最近邻树；

Algorithm Delete:

删除算法：

Input: point $q$

输入：点 $q$

1. delete $q$ from NN-tree;

1. 从最近邻树中删除 $q$；

2. $\left\{  {{p}_{1},{p}_{2},\ldots ,{p}_{k}}\right\}   \leftarrow$ query $q$ in RNN-tree;

2. 在反向最近邻树中进行 $\left\{  {{p}_{1},{p}_{2},\ldots ,{p}_{k}}\right\}   \leftarrow$ 查询 $q$；

3. delete $\left( {q,N\left( q\right) }\right)$ from RNN-tree;

3. 从反向最近邻树中删除 $\left( {q,N\left( q\right) }\right)$；

	for each ${p}_{i}$ (with corresponding ${R}_{i}$ ) do

		对于每个 ${p}_{i}$（对应 ${R}_{i}$）执行以下操作

		find $N\left( {p}_{i}\right)$ from NN-tree;

		从最近邻树（NN-tree）中查找$N\left( {p}_{i}\right)$;

				grow ${R}_{i}$ to $\left( {{p}_{i},N\left( {p}_{i}\right) }\right)$ ;

				从 ${R}_{i}$ 增长到 $\left( {{p}_{i},N\left( {p}_{i}\right) }\right)$；

---

Figure 5: Proposed Algorithms for Insertion and Deletion.

图 5：插入和删除的提议算法。

<!-- Media -->

Figure 5 presents pseudocode for insertion and deletion. The algorithm for insertion retrieves (from the RNN-tree) the reverse nearest neighbors ${p}_{i}$ of $q$ ,and their corresponding neighborhood objects ${R}_{i}$ ,without having to scan; each ${R}_{i}$ is then reduced in size to $\left( {{p}_{i},d\left( {{p}_{i},q}\right) }\right)$ . The algorithm for deletion works similarly,using the RNN-tree to find the points ${p}_{i}$ affected by the deletion; each corresponding ${R}_{i}$ is then expanded to $\left( {{p}_{i},d\left( {{p}_{i},N\left( {p}_{i}\right) }\right) }\right.$ .

图 5 展示了插入和删除的伪代码。插入算法（从 RNN 树中）检索 $q$ 的反向最近邻 ${p}_{i}$ 及其对应的邻域对象 ${R}_{i}$，无需进行扫描；然后将每个 ${R}_{i}$ 的大小缩减至 $\left( {{p}_{i},d\left( {{p}_{i},q}\right) }\right)$。删除算法的工作方式类似，使用 RNN 树查找受删除影响的点 ${p}_{i}$；然后将每个对应的 ${R}_{i}$ 扩展至 $\left( {{p}_{i},d\left( {{p}_{i},N\left( {p}_{i}\right) }\right) }\right.$。

## 5 Experiments on RNN queries

## 5 RNN 查询实验

We designed a set of experiments to test the I/O performance of our proposed method on large data sets. Our goal was to determine the scale-up trend of both static and dynamic queries. We also examined the performance of bichromatic versus monochromatic data. Below we present results from two batches of experiments, for static and dynamic RNN queries.

我们设计了一组实验，以测试我们提出的方法在大型数据集上的 I/O 性能。我们的目标是确定静态和动态查询的扩展趋势。我们还研究了双色数据与单色数据的性能。下面我们展示两批实验的结果，分别针对静态和动态 RNN 查询。

Methods: We compared the proposed algorithms given in Section 4 to the basic scanning approach. In the static case, the scanning approach precomputes an all-NN list and makes a pass through it to determine the reverse nearest neighbors. In the dynamic case, the scanning approach precomputes and maintains an inverted all-NN list. Each entry in the all-NN list corresponds to a point in the data set, and thus requires storing two items for nearest neighbor information: the point coordinates and nearest neighbor distances. Similarly, the RNN-tree used in the proposed method requires storing each point and its associated nearest neighborhood. Both also use an NN-tree for nearest neighbor search. Thus, the methods require the same storage space.

方法：我们将第 4 节中给出的提议算法与基本扫描方法进行了比较。在静态情况下，扫描方法预先计算一个全最近邻（all - NN）列表，并遍历该列表以确定反向最近邻。在动态情况下，扫描方法预先计算并维护一个反向全最近邻列表。全最近邻列表中的每个条目对应数据集中的一个点，因此需要存储两个最近邻信息项：点坐标和最近邻距离。同样，提议方法中使用的 RNN 树需要存储每个点及其关联的最近邻域。两者还使用一个最近邻树（NN - tree）进行最近邻搜索。因此，这些方法需要相同的存储空间。

Data Sets: Our testbed includes two real data sets. The first is mono and the second is bichromatic:

数据集：我们的测试平台包括两个真实数据集。第一个是单色的，第二个是双色的：

- cities 1 - Centers of ${100}\mathrm{\;K}$ cities and small towns in the USA (chosen at random from a larger data set of ${132}\mathrm{\;K}$ cities),represented as latitude and longitude coordinates;

- cities 1 - 美国 ${100}\mathrm{\;K}$ 个城市和小镇的中心（从包含 ${132}\mathrm{\;K}$ 个城市的更大数据集中随机选择），表示为经纬度坐标；

- cities2 - Coordinates of ${100}\mathrm{\;K}$ red cities (i.e., clients) and 400 black cities (i.e., servers). The red cities are mutually disjoint from the black cities, and points from both colors were chosen at random from the same source.

- cities2 - ${100}\mathrm{\;K}$ 个红色城市（即客户端）和 400 个黑色城市（即服务器）的坐标。红色城市与黑色城市相互不相交，两种颜色的点均从同一数据源中随机选择。

Queries: We assume the so-called 'biased' query model, in which queries are more likely to come from dense regions [13]. We chose 500 query points at random (without replacement) from the same source that the data sets were chosen; note that these points are external to the data sets. For dynamic queries, we simulated a mixed workload of insertions by randomly choosing between insertions and deletions. In the case of insertions, one of the 500 query points were inserted; for deletions, an existing point was chosen at random. We report the average $\mathrm{I}/\mathrm{O}$ per query,that is,the cumulative number of page accesses divided by the number of queries.

查询：我们假设采用所谓的“有偏”查询模型，在该模型中，查询更有可能来自密集区域 [13]。我们从选择数据集的同一数据源中随机（无放回）选择了 500 个查询点；请注意，这些点在数据集之外。对于动态查询，我们通过在插入和删除之间随机选择来模拟混合插入工作负载。在插入情况下，插入 500 个查询点中的一个；在删除情况下，随机选择一个现有点。我们报告每个查询的平均 $\mathrm{I}/\mathrm{O}$，即页面访问的累积次数除以查询次数。

Software: The code for our experiments was implemented in $\mathrm{C}$ on a Sun SparcWorkstation. To implement RNN queries, we extended DR-tree, a disk-resident R*- tree package; to implement NN queries (which were used for the second batch of experiments), we used the DR-tree as is. ${}^{5}$ The page size was set to $4\mathrm{\;K}$ .

软件：我们的实验代码在 $\mathrm{C}$ 中于 Sun Sparc 工作站上实现。为了实现 RNN 查询，我们扩展了 DR - 树，这是一个驻留在磁盘上的 R* 树包；为了实现最近邻查询（用于第二批实验），我们直接使用了 DR - 树。${}^{5}$ 页面大小设置为 $4\mathrm{\;K}$。

### 5.1 Static Case

### 5.1 静态情况

We uniformly sampled the cities 1 data set to get subsets of varying sizes,between ${10}\mathrm{\;K}$ and ${100}\mathrm{\;K}$ points. Figure 6(a) shows the I/O performance of the proposed method compared to sequential scan. Each query took roughly between 9-28 I/Os for the data sets we tried with our approach; in contrast, the performance of the scanning approach increased from 40 to ${4001}/\mathrm{{Os}}$ with increasing data set size(n). The gap between the two curves clearly widens as $n$ increases,and the proposed method appears to scale logarithmically, whereas the scanning approach scales linearly.

我们对 cities 1 数据集进行均匀采样，以获得大小在 ${10}\mathrm{\;K}$ 到 ${100}\mathrm{\;K}$ 个点之间的不同子集。图 6(a) 显示了提议方法与顺序扫描相比的 I/O 性能。使用我们的方法，对于我们尝试的数据集，每个查询大约需要 9 - 28 次 I/O；相比之下，扫描方法的性能随着数据集大小 (n) 的增加从 40 次增加到 ${4001}/\mathrm{{Os}}$ 次。随着 $n$ 的增加，两条曲线之间的差距明显扩大，提议方法似乎呈对数扩展，而扫描方法呈线性扩展。

We performed the same experiment for cities2. Figure 6(b) plots the I/O performance. It is interesting to note that the performance degrades more with increasing $n$ (from 12-65 1/Os) with bichromatic data; this is primarily because the output size is larger in bichromatic case than in the monochromatic case as remarked earlier. However, this increase again appears to be logarithmic.

我们对城市2数据集进行了相同的实验。图6(b)展示了输入/输出（I/O）性能。有趣的是，对于双色数据，随着$n$的增加（从12 - 65次输入/输出操作），性能下降得更多；这主要是因为如前文所述，双色情况下的输出大小比单色情况下更大。然而，这种增长似乎再次呈对数关系。

### 5.2 Dynamic Case

### 5.2 动态情况

Again, we used the cities 1 data set and uniformly sampled it to get subsets of varying sizes,between ${10}\mathrm{\;K}$ and ${100}\mathrm{\;K}$ points. As shown in Figure 7,the I/O cost for an even workload of insertions and deletions appears to scale logarithmically, whereas the scanning method scales linearly. It is interesting to note that the average $\mathrm{I}/\mathrm{O}$ is up to four times worse than in the static case, although this factor decreases for larger data sets. We broke down the I/O into four categories - RNN queries, NN queries, insertions and deletions - and found that each took approximately the same number of $\mathrm{I}/\mathrm{{Os}}$ . Thus, the maintenance of the NN-tree accounts for the extra $\mathrm{I}/\mathrm{O}$ compared to the static queries.

同样，我们使用城市1数据集并对其进行均匀采样，以获得大小在${10}\mathrm{\;K}$到${100}\mathrm{\;K}$个点之间的不同子集。如图7所示，对于插入和删除操作均匀分布的工作负载，I/O成本似乎呈对数增长，而扫描方法则呈线性增长。有趣的是，平均$\mathrm{I}/\mathrm{O}$比静态情况下差达四倍，不过对于更大的数据集，这个系数会减小。我们将I/O操作分为四类——反向最近邻（RNN）查询、最近邻（NN）查询、插入和删除——并发现每类操作大约需要相同数量的$\mathrm{I}/\mathrm{{Os}}$。因此，与静态查询相比，NN树的维护导致了额外的$\mathrm{I}/\mathrm{O}$。

<!-- Media -->

<!-- figureText: 220 cities: Dynamic Performance "insdel.proposed.cities" $\rightarrow   \rightarrow$ "insdel.scan.cities" -- 50000 70000 90000 Data set size (n) 200 180 160 Avg I/Os 140 120 100 80 60 40 20 10000 30000 -->

<img src="https://cdn.noedgeai.com/0195c909-9cbb-7185-8180-c4de2b24249b_6.jpg?x=944&y=1463&w=690&h=496&r=0"/>

Figure 7: The I/O performance of dynamic RNN queries (proposed method vs. scanning) in the presence of an even mix of insertions and deletions.

图7：在插入和删除操作均匀混合的情况下，动态RNN查询的I/O性能（所提出的方法与扫描方法对比）。

<!-- Media -->

---

<!-- Footnote -->

${}^{5}$ available at ftp://ftp.olympos.umd.edu.

${}^{5}$可从ftp://ftp.olympos.umd.edu获取。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: cities: Static Performance cities.bipartite: Static RNN Performance 400 "static.proposed.cities.bipartite" 350 300 Avg Leaf I/Os 250 200 150 100 50 10000 30000 50000 70000 90000 Data set size (n) (b) cities2 (bipartite) 400 "static.proposed.cities" - 350 300 Avg I/Os 250 200 150 100 50 0 10000 30000 50000 70000 90000 Data set size (n) (a) cities 1 (monochromatic) -->

<img src="https://cdn.noedgeai.com/0195c909-9cbb-7185-8180-c4de2b24249b_7.jpg?x=173&y=184&w=1433&h=539&r=0"/>

Figure 6: The I/O performance of static RNN queries (proposed method vs. scanning) for (a) cities1 (monochromatic) and (b) cities 2 (bipartite).

图6：静态RNN查询的I/O性能（所提出的方法与扫描方法对比），(a) 城市1（单色）和 (b) 城市2（双色）。

<!-- Media -->

## 6 Influence Sets

## 6 影响集

### 6.1 Basic notion and applications

### 6.1 基本概念和应用

Our first, and most basic, definition of the influence set of a point $q$ is simply that it is the set of all reverse nearest neighbors of $q$ ,that is, $\mathcal{{RNN}}\left( q\right)$ . This may be mono or bichromatic reverse nearest neighbors, depending on the application.

我们对一个点$q$的影响集的第一个也是最基本的定义是，它是$q$的所有反向最近邻的集合，即$\mathcal{{RNN}}\left( q\right)$。根据应用场景的不同，这可能是单色或双色的反向最近邻。

Before exploring this notion further, let us briefly reexamine the motivating examples from Section 1. In Examples 1 and 2, the influence set of the new location of a store outlet is indeed the set of customers who find the new location the closest amongst all locations of stores. This is an instance of bichromatic RNN. In Example 3, the customers who are influenced by a new service are those whose profiles have the feature vector of the new service closest amongst all service feature vectors. Again, the influence set of the new service corresponds to our basic definition above. In Example 4, the influence set of a new document is the set of all documents in the database that find it the closest under a suitable measure of similarity; here, the definition of an influence set based on monochromatic RNNs applies.

在进一步探讨这个概念之前，让我们简要回顾一下第1节中的动机示例。在示例1和2中，一家商店新位置的影响集实际上是那些认为该新位置是所有商店位置中最近的顾客集合。这是双色RNN的一个实例。在示例3中，受一项新服务影响的顾客是那些其特征向量认为新服务的特征向量在所有服务特征向量中最近的顾客。同样，新服务的影响集符合我们上面的基本定义。在示例4中，一篇新文档的影响集是数据库中所有在合适的相似度度量下认为该新文档最近的文档集合；这里，基于单色RNN的影响集定义适用。

We can think of many other applications where the basic notion of influence set arises. What is perhaps more interesting is that this notion of influence sets implicitly arises in many computational tasks.

我们可以想到许多其他会出现影响集基本概念的应用。也许更有趣的是，影响集的这个概念在许多计算任务中会隐式出现。

For example, many problems of interest in Operations Research and Combinatorial Optimization have greedy solutions with good performance. One such example is the facility location problem. Here we are given many points and the goal is to designate some as facilities and others as non-facilities. There is a cost to designating a point as a facility, and a cost for non-facilities which equals the cost of accessing the closest facility. This problem is known to be NP-hard, and thus the focus is on designing approximation algorithms for this problem. The method of choice in practice for this problem is the greedy method - it is simple, and is a provably small approximation [14]. ${}^{6}$ The greedy algorithm involves repeatedly adding a facility, deleting one, or swapping a facility with a non-facility. In order to implement this algorithm, we need to determine the enhanced cost when a new facility is added which involves looking at precisely those locations whose NN distance is changed when a new facility is added (or deleted, swapped). The set of all such locations is indeed our basic definition of a influence set; these have been implicitly computed in this context for a long time. Another example is that of computing the shortest path from a single point to every other point in the database. When a point is added to a partial solution that greedy algorithms maintain, the distance of remaining points to the partial solution has to be updated and this will again be given by the influence set of the point added to the partial solution. Many other implicit uses of influence sets exist in Combinatorial Optimization.

例如，运筹学和组合优化中许多感兴趣的问题都有性能良好的贪心算法解决方案。设施选址问题就是这样一个例子。在这个问题中，我们给定了许多点，目标是指定其中一些为设施点，另一些为非设施点。指定一个点为设施点有成本，非设施点的成本等于访问最近设施点的成本。已知这个问题是NP难问题，因此重点在于为这个问题设计近似算法。在实践中，解决这个问题的首选方法是贪心算法——它简单，并且是一种经证明的小近似算法[14]。${}^{6}$贪心算法涉及反复添加一个设施点、删除一个设施点或交换一个设施点和一个非设施点。为了实现这个算法，我们需要确定添加一个新设施点时增加的成本，这需要精确查看那些在添加（或删除、交换）一个新设施点时其最近邻距离发生变化的位置。所有这些位置的集合实际上就是我们对影响集的基本定义；在这种情况下，这些影响集已经被隐式计算了很长时间。另一个例子是计算从一个单点到数据库中其他每个点的最短路径。当一个点被添加到贪心算法维护的部分解中时，剩余点到该部分解的距离必须更新，而这又将由添加到部分解中的点的影响集给出。在组合优化中，影响集还有许多其他隐式用途。

### 6.2 Using existing methods

### 6.2 使用现有方法

There are two potential problems with the effectiveness of any approach to finding influence sets. One is the precision problem wherein a large portion of the retrieved set contains irrelevant points. Conversely, there is the recall problem wherein the retrieved set misses some of the relevant points. An effective approach would achieve high precision at high recall (ideally, ${100}\%$ precision at ${100}\%$ recall). In this section we present results from an experiment to demonstrate that nearest neighbor queries and range queries are not effective "engineering" substitutes for RNN queries in finding influence sets; we use standard precision and recall metrics from information retrieval to assess their quality.

任何寻找影响集的方法在有效性方面都存在两个潜在问题。一是精度问题，即检索到的集合中有很大一部分包含不相关的点。相反，还有召回率问题，即检索到的集合遗漏了一些相关的点。一种有效的方法应该在高召回率的情况下实现高精度（理想情况下，在${100}\%$召回率时达到${100}\%$精度）。在本节中，我们展示一个实验的结果，以证明在寻找影响集时，最近邻查询和范围查询并不是反向最近邻（RNN）查询的有效“工程”替代方案；我们使用信息检索中的标准精度和召回率指标来评估它们的质量。

---

<!-- Footnote -->

${}^{6}$ Better approximations exist,but they are based on Linear Programming [10].

${}^{6}$ 存在更好的近似方法，但它们基于线性规划 [10]。

<!-- Footnote -->

---

The first issue that arises in finding influence sets is what region to search in. Two possibilities immediately present themselves: find the closest points (i.e., the $k$ -nearest neighbors) or all points within some radius (i.e., $\epsilon$ -range search). Of course,there are many variants of these basic queries, such as searching with weighted distances, searching over polygonal or elliptical regions, etc. To demonstrate the ineffectiveness of these approaches, it shall suffice to consider the most basic version. The question then is how to engineer the parameter value (namely $k$ or $\epsilon$ ) that will contain the desired information. Without a priori knowledge of the density of points near the query point $q$ ,it is not clear how to choose these values. Regardless, we show that any clever strategy to engineer parameter values (be it from histograms, etc.) would still fall short.

寻找影响集时出现的第一个问题是要搜索的区域。有两种可能性立即浮现出来：找到最近的点（即$k$ -最近邻）或某个半径内的所有点（即$\epsilon$ -范围搜索）。当然，这些基本查询有很多变体，比如使用加权距离进行搜索、在多边形或椭圆形区域上进行搜索等等。为了证明这些方法的无效性，考虑最基本的版本就足够了。那么问题就变成了如何设计包含所需信息的参数值（即$k$ 或$\epsilon$ ）。在没有查询点$q$ 附近点密度的先验知识的情况下，不清楚如何选择这些值。无论如何，我们表明，任何巧妙的参数值设计策略（无论是基于直方图等）仍然是不够的。

Figure 8 illustrates this concept. The black points represent servers and the white points represent clients. In this example, we wish to find all the clients for which $q$ is their closest server. The example illustrates that a $\epsilon$ -range (alternatively, $k$ -NN) query cannot find the desired information in this case, regardless of which value of $\epsilon$ (or $k$ ) is chosen. Figure 8(a) shows a ’safe’ radius ${\epsilon }_{l}$ in which all points are reverse nearest neighbors of $q$ ; however,there exist reverse nearest neighbors of $q$ outside ${\epsilon }_{l}$ . Figure 8(b) shows a wider radius ${\epsilon }_{h}$ that includes all of the reverse nearest neighbors of $q$ but also includes points which are not. In this example,it is possible to achieve ${100}\%$ precision or 100% recall, but not both simultaneously.

图 8 说明了这一概念。黑点代表服务器，白点代表客户端。在这个例子中，我们希望找到所有以$q$ 为其最近服务器的客户端。这个例子说明，在这种情况下，无论选择$\epsilon$ （或$k$ ）的哪个值，$\epsilon$ -范围（或者$k$ -最近邻）查询都无法找到所需信息。图 8(a) 显示了一个“安全”半径${\epsilon }_{l}$ ，其中所有点都是$q$ 的反向最近邻；然而，$q$ 的反向最近邻存在于${\epsilon }_{l}$ 之外。图 8(b) 显示了一个更宽的半径${\epsilon }_{h}$ ，它包含了$q$ 的所有反向最近邻，但也包含了一些不是的点。在这个例子中，可以实现${100}\%$ 精度或 100% 召回率，但不能同时实现两者。

We ran an experiment to investigate how often this trade-off occurs in practice. The experiment was carried out as follows. Suppose we had an oracle to suggest the largest radius ${\epsilon }_{l}$ admitting no false-positives,i.e.,whose neighborhood contains only points in the influence set. For this scenario, we assess the quality of the retrieved set from the number of false-negatives within this radius. More specifically, we measured the recall at ${100}\%$ precision,that is,the cardinality of the retrieved set divided by that of the influence set. Further suppose we had an oracle to suggest the smallest radius ${\epsilon }_{h}$ allowing no false-negatives, i.e., whose neighborhood contains the full influence set (equivalently, reverse nearest neighbors). For this scenario, we assess the quality of the retrieved set from the number of false-positives within this radius. More specficially, we measured the precision at ${100}\%$ recall,that is,the cardinality of the influence set divided by that of the retrieved set.

我们进行了一个实验，以研究这种权衡在实际中出现的频率。实验如下进行。假设我们有一个神谕可以建议最大半径${\epsilon }_{l}$ ，该半径不产生误报，即其邻域仅包含影响集中的点。对于这种情况，我们从该半径内的漏报数量来评估检索到的集合的质量。更具体地说，我们测量了在${100}\%$ 精度下的召回率，即检索到的集合的基数除以影响集的基数。进一步假设我们有一个神谕可以建议最小半径${\epsilon }_{h}$ ，该半径不产生漏报，即其邻域包含完整的影响集（等价于反向最近邻）。对于这种情况，我们从该半径内的误报数量来评估检索到的集合的质量。更具体地说，我们测量了在${100}\%$ 召回率下的精度，即影响集的基数除以检索到的集合的基数。

We used the cities 2 data set in the our experiment and averaged over 100 queries. The results are summarized in Table 1. The quality of the retrieved set at radius ${\epsilon }_{l}$ is poor,containing a small fraction of the full influence set. The quality of the retrieved set at radius ${\epsilon }_{h}$ is also poor,containing a lot of ’garbage’ in addition to the influenced points.

我们在实验中使用了城市 2 数据集，并对 100 个查询进行了平均。结果总结在表 1 中。半径${\epsilon }_{l}$ 下检索到的集合质量很差，只包含完整影响集的一小部分。半径${\epsilon }_{h}$ 下检索到的集合质量也很差，除了受影响的点之外还包含很多“无用信息”。

<!-- Media -->

<table><tr><td>measure</td><td>radius</td><td>value</td></tr><tr><td>precision (at 100% recall)</td><td>${\epsilon }_{h}$</td><td>44.3%</td></tr><tr><td>recall (at 100% precision)</td><td>${\epsilon }_{l}$</td><td>40.2%</td></tr></table>

<table><tbody><tr><td>测量；度量</td><td>半径</td><td>数值；值</td></tr><tr><td>精确率（召回率为100%时）</td><td>${\epsilon }_{h}$</td><td>44.3%</td></tr><tr><td>召回率（精确率为100%时）</td><td>${\epsilon }_{l}$</td><td>40.2%</td></tr></tbody></table>

Table 1: The effectiveness of range queries in finding influence sets. Quality is measured by precision at ${100}\%$ recall and recall at ${100}\%$ precision.

表1：范围查询在查找影响集方面的有效性。质量通过${100}\%$召回率下的精确率和${100}\%$精确率下的召回率来衡量。

<!-- Media -->

### 6.3 Extended notions of influence sets

### 6.3 影响集的扩展概念

In this section, we extend the notion of influence sets from the previous section. We do not explore these notions in depth here using experiments; instead we focus on sketching how our approach for finding the basic influence sets can be modified to find these extended influence sets. Some of these modifications will be straightforward, others less so.

在本节中，我们将扩展上一节中影响集的概念。我们在这里不会通过实验深入探讨这些概念；相反，我们将重点描述如何修改我们查找基本影响集的方法来查找这些扩展的影响集。其中一些修改很直接，而另一些则不然。

Reverse $k$ -nearest neighbors. A rather simple extension of the influence set of point $q$ is to define it to be the set of all points that have $q$ as one of their $k$ nearest neighbors. Here, $k$ is fixed and specified a priori. For static queries, the only difference in our solution is that we store the neighborhood of $k$ th neighbor rather than nearest neighbor. (Note that we do not explicitly store the $k$ nearest neighbors.) Each query is an enclosure problem on these objects as in the basic case. For insertions and deletions, we update the neighborhood of the $k$ th nearest neighbor of each affected point as follows. When inserting or deleting $q$ ,we first find the set of affected points using the enclosure problem as done for answering queries. For insertion, we perform a range query to determine the $k$ nearest neighbors of each such affected point and do necessary updates. For deletion, the neighborhood radius of the affected points is expanded to the distance of the $\left( {k + 1}\right)$ th neighbor,which can be found by a modified NN search on R-trees.

反向$k$ -最近邻。点$q$的影响集的一个相当简单的扩展是将其定义为所有将$q$作为其$k$个最近邻之一的点的集合。这里，$k$是预先固定并指定的。对于静态查询，我们的解决方案的唯一区别在于，我们存储的是第$k$个邻居的邻域，而不是最近邻。（请注意，我们不会显式存储$k$个最近邻。）与基本情况一样，每个查询都是这些对象上的包含问题。对于插入和删除操作，我们按如下方式更新每个受影响点的第$k$个最近邻的邻域。当插入或删除$q$时，我们首先使用包含问题来找到受影响点的集合，就像回答查询时那样。对于插入操作，我们执行范围查询以确定每个受影响点的$k$个最近邻，并进行必要的更新。对于删除操作，受影响点的邻域半径扩展到第$\left( {k + 1}\right)$个邻居的距离，这可以通过对R树进行修改后的最近邻搜索来找到。

Influence sets with predicates. The basic notion of influence sets can be enhanced with predicates. Some examples of predicates involve bounding the search distance (find reverse nearest neighbors within a specified region of interest) and providing multiple facilities (find the reverse nearest neighbors to any, some,or all of multiple points in the set $\left. \left\{  {{q}_{1},\ldots ,{q}_{m}}\right\}  \right)$ . For such queries, we can push the predicates inside the R-tree search.

带谓词的影响集。影响集的基本概念可以通过谓词来增强。谓词的一些示例包括限制搜索距离（在指定的感兴趣区域内查找反向最近邻）和提供多个设施（查找集合$\left. \left\{  {{q}_{1},\ldots ,{q}_{m}}\right\}  \right)$中任意、某些或所有多个点的反向最近邻）。对于此类查询，我们可以将谓词推送到R树搜索中。

<!-- Media -->

<!-- figureText: Og O O Og O 0 Og ⑥ O Og O O O Og ${\varepsilon }_{\mathrm{h}}$ 0 O O (b) range ${\epsilon }_{h}$ Og 0 0 O O Og O O (a) range ${\epsilon }_{l}$ -->

<img src="https://cdn.noedgeai.com/0195c909-9cbb-7185-8180-c4de2b24249b_9.jpg?x=381&y=178&w=1028&h=529&r=0"/>

Figure 8: In many cases,any $\epsilon$ -range query or $k$ -NN query will be either (a) too small or (b) too big.

图8：在许多情况下，任何$\epsilon$ -范围查询或$k$ -最近邻查询要么（a）太小，要么（b）太大。

<!-- Media -->

Reverse furthest neighbors. An interesting variation of influence sets is to base it on dissimilarity rather than similarity, in other words, on furthest neighbors rather than nearest neighbors. More formally, define the influence set of a point $q$ to be the set of all points $r$ such that $q$ is farther from $r$ than any other point of the database is from $r$ . This notion of influence has a solution that differs from the basic notion in an interesting way. We sketch the solution here only for the static case, but all modifications to convert this into a dynamic solution are based on ideas we already described before. We will also only describe the solution for the two dimensional case, but extending it to the multi-dimensional case is straightforward.

反向最远邻。影响集的一个有趣变体是基于相异性而不是相似性，换句话说，基于最远邻而不是最近邻。更正式地说，将点$q$的影响集定义为所有点$r$的集合，使得$q$到$r$的距离比数据库中任何其他点到$r$的距离都远。这种影响概念的解决方案与基本概念有有趣的不同。我们这里仅针对静态情况概述解决方案，但将其转换为动态解决方案的所有修改都基于我们之前已经描述过的想法。我们也仅描述二维情况的解决方案，但将其扩展到多维情况很直接。

Say $S$ is the set of points which will be fixed. A query point is denoted $q$ . For simplicity,we will first describe our solution for the ${L}_{\infty }$ distance.

假设$S$是固定的点集。查询点用$q$表示。为简单起见，我们首先描述针对${L}_{\infty }$距离的解决方案。

Preprocessing: We first determine the furthest point for each point $p \in  S$ and denote it as $f\left( p\right)$ . We will put a square with center $p$ and sides ${2d}\left( {p,f\left( p\right) }\right)$ for each $p$ ; say this square is ${R}_{p}$ .

预处理：我们首先为每个点$p \in  S$确定最远点，并将其表示为$f\left( p\right)$。我们将为每个$p$放置一个以$p$为中心、边长为${2d}\left( {p,f\left( p\right) }\right)$的正方形；假设这个正方形是${R}_{p}$。

Query processing: The simple observation is that for any query $q$ ,the reverse furthest neighbors $r$ are those for which the ${R}_{r}$ does not include $q$ . Thus the problem we have is square non-enclosure problem. (Recall that, in contrast, the reverse nearest neighbors problem led to square enclosure problem.)

查询处理：简单的观察是，对于任何查询$q$，反向最远邻$r$是那些其${R}_{r}$不包含$q$的点。因此，我们面临的问题是正方形不包含问题。（回想一下，相比之下，反向最近邻问题导致的是正方形包含问题。）

The following observation is the key to solving the square non-enclosure problem.

以下观察是解决正方形不包含问题的关键。

Lemma 3 Consider the intervals ${x}_{r}$ and ${y}_{r}$ obtained by projecting the square ${R}_{r}$ on $x$ and $y$ axis respectively. A point $q = \left( {x,y}\right)$ is not contained in ${R}_{r}$ if and only if either ${x}_{r}$ does not contain $x$ or ${y}_{r}$ does not contain $y$ .

引理3 考虑分别将正方形${R}_{r}$投影到$x$轴和$y$轴上得到的区间${x}_{r}$和${y}_{r}$。点$q = \left( {x,y}\right)$不包含在${R}_{r}$中，当且仅当${x}_{r}$不包含$x$或者${y}_{r}$不包含$y$。

Therefore,if we return all the ${x}_{r}$ ’s that do not contain $x$ as well as those ${y}_{r}$ ’s that do not contain $y$ ’s,each square $r$ in the output is repeated atmost twice. So the problem can be reduced to a one dimensional problem on intervals without losing much efficiency. Let us restate the one dimensional problem formally: we are given a set of intervals,say $N$ of them. Each query is a one dimensional point,say $p$ ,and the goal is to return all interval that do not contain $p$ .

因此，如果我们返回所有不包含$x$的${x}_{r}$以及所有不包含$y$的${y}_{r}$，输出中的每个正方形$r$最多重复两次。所以该问题可以在不损失太多效率的情况下简化为一个关于区间的一维问题。让我们正式地重述这个一维问题：给定一组区间，设为$N$个。每次查询是一个一维点，设为$p$，目标是返回所有不包含$p$的区间。

For solving this problem, we maintain two sorted arrays, one of the right endpoints of the intervals and the other of their left endpoints. The following observation is easy:

为了解决这个问题，我们维护两个有序数组，一个是区间的右端点数组，另一个是区间的左端点数组。以下观察结果很容易得出：

Proposition 2 Any interval with its right endpoint to the left of $p$ does not contain $p$ . Likewise,any interval with its left endpoint to the right of $p$ does not contain ${it}$ .

命题2 任何右端点在$p$左侧的区间都不包含$p$。同样，任何左端点在$p$右侧的区间都不包含${it}$。

Hence, it suffices to perform two binary searches with $p$ in the two arrays,to determine the intervals that do not contain $p$ . Notice that from a practical viewpoint, the only data structure we need is a B-tree to keep these two arrays.

因此，在这两个数组中对$p$进行两次二分查找，就足以确定不包含$p$的区间。从实际角度来看，我们唯一需要的数据结构是一个B树来存储这两个数组。

Theorem 1 There exists an $\Theta \left( {{\log }_{B}N + t}\right)$ time algorithm to answer each query in the square non-enclosure problem,where $t$ is the output size, $N = n/B$ is the number of disk blocks, and $B$ is the block size; space used is $\Theta \left( N\right)$ .

定理1 存在一种$\Theta \left( {{\log }_{B}N + t}\right)$时间复杂度的算法来回答正方形非包含问题中的每个查询，其中$t$是输出规模，$N = n/B$是磁盘块的数量，$B$是块大小；使用的空间为$\Theta \left( N\right)$。

Although the solution above is simple, we are not aware of any published claim of this result for square non-disclosure problem. As mentioned before, this immediately gives a solution for finding the set of all reverse furthest neighbors for a query point under the ${L}_{\infty }$ distance. While a R-tree may be used to solve this problem, our solution above shows that the only data structure we need is to a B-tree. Hence, the solution is very efficient. For other distance functions, we still have non-enclosure problem, but with different shapes (e.g., circles for Euclidean distance). Practical approaches for solving such problems would be to either use bounding boxes to reduce the problem to square non-enclosure with some false positives, or to use R-tree based search.

虽然上述解决方案很简单，但我们尚未发现有任何已发表的关于正方形非包含问题这一结果的声明。如前所述，这立即为在${L}_{\infty }$距离下查找查询点的所有反向最远邻集合提供了一个解决方案。虽然可以使用R树来解决这个问题，但我们上述的解决方案表明，我们唯一需要的数据结构是一个B树。因此，该解决方案非常高效。对于其他距离函数，我们仍然会遇到非包含问题，但形状不同（例如，欧几里得距离对应的是圆形）。解决此类问题的实际方法要么是使用边界框将问题简化为正方形非包含问题，但会有一些误报，要么是使用基于R树的搜索。

## 7 Related Work

## 7 相关工作

There has been a lot of work on nearest neighbor queries $\left\lbrack  {5,{11},4,{15},9}\right\rbrack$ . NN queries are useful in many applications: GIS (’Find the $k$ nearest hospitals from the place of an accident.'), information retrieval ('Find the most similar web page to mine.'), multimedia databases ('Find the tumor shape that looks the most similar to the query shape.' [12]), etc. Conceptually, a RNN query is different from a NN query; it is the inverse.

关于最近邻查询$\left\lbrack  {5,{11},4,{15},9}\right\rbrack$已经有很多研究工作。最近邻查询在许多应用中都很有用：地理信息系统（“查找距离事故地点最近的$k$家医院。”）、信息检索（“查找与我的网页最相似的网页。”）、多媒体数据库（“查找与查询形状最相似的肿瘤形状。” [12]）等等。从概念上讲，反向最近邻查询与最近邻查询不同；它是相反的。

There has been work in the area of spatial joins, and more specifically with all-nearest neighbor queries [8]. To the best of our knowledge, none of the previous work has addressed the issue of incremental maintenance. While reverse nearest neighbors is conceptually different, RNN queries provide an efficient means to incrementally maintain all-nearest neighbors.

在空间连接领域已经有相关研究工作，更具体地说是关于全最近邻查询[8]。据我们所知，之前的工作都没有涉及增量维护的问题。虽然反向最近邻在概念上不同，但反向最近邻查询为增量维护全最近邻提供了一种高效的方法。

Both incremental and random incremental Delaunay triangulations could be used for answering RNN queries, as the update step involves identifying points (and their circumcircles) local to the query point whose edges are affected by insertion/deletion, a superset of reverse nearest neighbors. ${}^{7}$ However,these algorithms rely on being able to efficiently locate the simplex containing the query point, a problem for which there is no efficient solution in large data sets. In addition, the algorithms make the general position assumption and do not work well for the bipartite case.

增量和随机增量德劳内三角剖分都可以用于回答反向最近邻查询，因为更新步骤涉及识别查询点局部的点（及其外接圆），这些点的边会受到插入/删除操作的影响，而这些点是反向最近邻的超集。${}^{7}$ 然而，这些算法依赖于能够高效地定位包含查询点的单纯形，而对于大数据集，这个问题没有高效的解决方案。此外，这些算法做了一般位置假设，在二分情况中效果不佳。

Our approach for RNN queries relied on solving point enclosure problems with different shapes. Point enclosure problems with $n$ rectangles can be solved after $O\left( {n{\log }^{d - 1}n}\right)$ time preprocessing in $O\left( {{\log }^{d - 1}n + t}\right)$ time per query where $t$ is the output size [1]. Such efficient algorithms are not known for other shapes, for dynamic cases, or for external memory datasets. Our R-tree approach is simple and applies to all the variants of RNN queries.

我们针对反向最近邻（RNN）查询的方法依赖于解决不同形状的点包含问题。对于$n$个矩形的点包含问题，在经过$O\left( {n{\log }^{d - 1}n}\right)$时间的预处理后，每次查询可以在$O\left( {{\log }^{d - 1}n + t}\right)$时间内解决，其中$t$是输出大小[1]。对于其他形状、动态情况或外部内存数据集，目前还没有已知的此类高效算法。我们的R树方法简单，并且适用于RNN查询的所有变体。

## 8 Conclusions and Future Work

## 8 结论与未来工作

The "influence" of a point in a database is a useful concept. In this paper, we introduce an intuitive notion of influence based on reverse nearest neighbors, and illustrate through examples that it has broad appeal in many application domains. The basic notion of influence sets depends on reverse nearest neighbor queries, which are also of independent interest. We provide the first solution to this problem, and validate its I/O-efficiency through experiments. We also demonstrate using experiments that standard database queries such as range searching and NN queries are ineffective at finding influence sets. Finally, we further extend the notion of influence based on variants of RNN queries and provide efficient solutions for these variants as well.

数据库中一个点的“影响力”是一个有用的概念。在本文中，我们引入了一种基于反向最近邻的直观影响力概念，并通过示例说明它在许多应用领域具有广泛的吸引力。影响力集合的基本概念依赖于反向最近邻查询，而反向最近邻查询本身也具有独立的研究价值。我们首次为这个问题提供了解决方案，并通过实验验证了其I/O效率。我们还通过实验证明，诸如范围搜索和最近邻（NN）查询等标准数据库查询在查找影响力集合方面是无效的。最后，我们基于RNN查询的变体进一步扩展了影响力的概念，并为这些变体提供了高效的解决方案。

We have initiated the study of influence sets using reverse nearest neighbors. Many issues remain to be explored, for example, the notion of influence outside of the typical query-response context, such as in data mining. It is often desirable to process the data set to suggest a region in which a query point should lie so as to exert maximal influence. What is the appropriate notion of influence sets in this context? From a technical point of view, efficient solutions for RNN queries are needed in high dimensions. Extensions of our approach to higher dimensions is straightforward; however, in very high dimensions, alternative approaches may be needed, as is the case for high dimensional NN queries. Also, the role of RNN queries can be explored further, such as in other proximity-based problems. It is our belief that the RNN query is a fundamental query, deserving to be a standard tool for data processing.

我们已经启动了使用反向最近邻对影响力集合的研究。许多问题仍有待探索，例如，在典型查询 - 响应上下文之外的影响力概念，如在数据挖掘中。通常希望对数据集进行处理，以建议查询点应位于的区域，从而发挥最大影响力。在这种情况下，影响力集合的合适概念是什么？从技术角度来看，高维情况下需要RNN查询的高效解决方案。将我们的方法扩展到更高维度很直接；然而，在非常高的维度中，可能需要替代方法，就像高维NN查询的情况一样。此外，RNN查询的作用可以进一步探索，例如在其他基于邻近性的问题中。我们相信RNN查询是一个基本查询，值得成为数据处理的标准工具。

## Acknowledgements

## 致谢

The authors wish to thank Christos Faloutsos, Dimitris Gunopoulos, H.V. Jagadish, Nick Koudas, and Dennis Shasha for their comments.

作者感谢克里斯托斯·法洛索斯（Christos Faloutsos）、迪米特里斯·古诺普洛斯（Dimitris Gunopoulos）、H.V.贾加迪什（H.V. Jagadish）、尼克·库达斯（Nick Koudas）和丹尼斯·沙莎（Dennis Shasha）的评论。

## References

## 参考文献

[1] P. Agrawal. Range searching. In E. Goodman and J. O'Rourke, editors, Handbook of Discrete and Computational Geometry, pages 575-598. CRC Press, Boca Raton, FL, 1997.

[2] N. Beckmann, H.-P. Kriegel, R. Schneider, and B. Seeger. The R*-tree: An efficient and robust access method for points and rectangles. ${ACM}$ SIGMOD, pages 322-331, May 23-25 1990.

[3] T. Brinkhoff, H.-P. Kriegel, and B. Seeger. Efficient processing of spatial joins using R-trees. In Proc. of ACM SIGMOD, pages 237-246, Washington, D.C., May 26-28 1993.

[4] B. Chazelle and L. J. Guibas. Fractional cascading: I. A data structuring technique. Algorithmica, $1 : {133} - {162},{1986}$ .

[5] K. Fukunaga and P. M. Narendra. A branch and bound algorithm for computing k-nearest

---

<!-- Footnote -->

${}^{7}$ Alternatively,one could use the dual data structure,Voronoi diagrams.

${}^{7}$ 或者，也可以使用对偶数据结构，即 Voronoi 图。

<!-- Footnote -->

---

neighbors. IEEE Trans. on Computers (TOC), C-

邻居。《IEEE 计算机汇刊》（TOC），C -

24(7):750-753, July 1975.

[6] V. Gaede and O. Gunther. Multidimensional access methods. ACM Computing Surveys, 30(2):170- 231, June 1998.

[7] A. Guttman. R-trees: A dynamic index structure for spatial searching. In Proc. ACM SIGMOD, pages 47-57, Boston, Mass, June 1984.

[8] G. R. Hjaltason and H. Samet. Incremental distance join algorithms for spatial databases. ACM SIGMOD '98, pages 237-248, June 1998.

[9] G. R. Hjaltason and H. Samet. Distance browsing in spatial databases. ACM TODS, 24(2):265-318, June 1999.

[10] K. Jain and V. Vazirani. Primal-dual approximation algorithms for metric facility location and $k$ - median problems. Proc. 40th IEEE Foundations of Computer Science (FOCS '99), pages 2-13, 1999.

[11] D. G. Kirkpatrick. Optimal search in planar subdivisions. SIAM J. Comput., 12:28-35, 1983.

[12] F. Korn, N. Sidiropoulos, C. Faloutsos, E. Siegel, and Z. Protopapasa. Fast nearest-neighbor search in medical image databases. Conf. on Very Large Data Bases (VLDB), pages 215-226, September 1996.

[13] B. Pagel, H. Six, H. Toben, and P. Widmayer. Towards an analysis of range query performance. In Proc. of ACM SIGACT-SIGMOD-SIGART Symposium on Principles of Database Systems (PODS), pages 214-221, Washington, D.C., May 1993.

[14] R. Rajaraman, M. Korupolu, and G. Plaxton. Analysis of a local search heuristic for facility location problems. Proceedings of ACM-SIAM Symposium on Discrete Algorithms (SODA '98), pages $1 - {10},{1998}$ .

[15] N. Roussopoulos, S. Kelley, and F. Vincent. Nearest neighbor queries. In Proc. of ACM-SIGMOD, pages 71-79, San Jose, CA, May 1995.

[16] T. Sellis, N. Roussopoulos, and C. Faloutsos. The R+ tree: A dynamic index for multi-dimensional objects. In Proc. 13th International Conference on VLDB, pages 507-518, England,, September 1987.

[17] M. Smid. Closest point problems in computational geometry. In J.-R. Sack and J. Urrutia, editors, Handbook on Computational Geometry. Elsevier Science Publishing, 1997.