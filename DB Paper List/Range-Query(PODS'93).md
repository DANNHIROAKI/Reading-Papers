# Towards an Analysis of Range Query Performance in Spatial Data Structures*

# 空间数据结构中范围查询性能分析研究*

Bernd-Uwe Pagel** Hans-Werner Six**

伯恩德 - 乌韦·帕格尔** 汉斯 - 维尔纳·西克斯**

## Abstract

## 摘要

In this paper, we motivate four different user defined window query classes and derive a probabilistic model for each of them. For each model, we characterize the efficiency of spatial data structures in terms of the expected number of data bucket accesses needed to perform a window query. Our analytical approach exhibits the performance phenomena independent of data structure and implementation details and whether the objects are points or non-point objects.

在本文中，我们提出了四种不同的用户定义窗口查询类别，并为每一类推导了一个概率模型。对于每个模型，我们根据执行窗口查询所需的数据桶访问的预期次数来描述空间数据结构的效率。我们的分析方法展示了与数据结构和实现细节无关的性能现象，也与对象是点对象还是非点对象无关。

## 1 Introduction

## 1 引言

In recent years, various efficient data structures for maintaining large sets of multidimensional geometric objects have been developed. Most of these structures have been designed for multidimensional points (see e.g. $\left\lbrack  {2,5,7}\right\rbrack  )$ . In typical applications,however,objects are arbitrarily geometric, i.e. non-point objects. In many situations, it has been proven to be useful to characterize non-point objects by their bounding boxes, i.e. minimal enclosing multidimensional intervals, serving as simple geometric keys. Hence, non-point data structures deal with multidimensional intervals (see e.g. $\left\lbrack  {4,5,6}\right\rbrack  )$ . Only the cell tree [3] does not use this approximation.

近年来，人们开发了各种用于维护大量多维几何对象的高效数据结构。这些结构大多是为多维点设计的（例如参见 $\left\lbrack  {2,5,7}\right\rbrack  )$）。然而，在典型应用中，对象是任意几何形状的，即非点对象。在许多情况下，用对象的边界框（即最小的多维包围区间）来表征非点对象已被证明是有用的，这些边界框可作为简单的几何键。因此，非点数据结构处理的是多维区间（例如参见 $\left\lbrack  {4,5,6}\right\rbrack  )$）。只有单元树 [3] 不使用这种近似方法。

All proposals claim to improve the performance of spatial accesses and provide performance evaluations for range queries, which are the most popular spatial access operation. However, up to now, all evaluations have been carried out by simulations using the only assumption that ranges are windows of certain areas (e.g. $1\% ,{0.1}\%$ and ${0.01}\%$ relatively to the area of the data space) and that window centers are uniformly distributed.

所有的提议都声称可以提高空间访问的性能，并为范围查询（最流行的空间访问操作）提供了性能评估。然而，到目前为止，所有的评估都是通过模拟进行的，唯一的假设是范围是具有一定面积的窗口（例如相对于数据空间面积的 $1\% ,{0.1}\%$ 和 ${0.01}\%$），并且窗口中心是均匀分布的。

In this paper, we try to get one step further towards an understanding of window query performance of spatial data structures. We sketch four different classes of user-defined window queries and motivate their practical relevance. For each class, we derive a probabilistic model, which we use as basis for our analytical investigations. For each window query model, we derive a performance measure which characterizes arbitrary data space organizations in terms of the expected number of bucket accesses needed to perform a window query. Since it is well-known that in practical applications data bucket accesses exceed by far external accesses to the paged parts of the corresponding directory concerning frequency and execution time, the data space organization of a spatial data structure is essential for the window query performance. Hence, although our considerations are restricted to data bucket accesses, the real situation is still sufficiently reflected.

在本文中，我们试图进一步理解空间数据结构的窗口查询性能。我们概述了四种不同类别的用户定义窗口查询，并说明了它们的实际相关性。对于每一类，我们推导了一个概率模型，并将其作为我们分析研究的基础。对于每个窗口查询模型，我们推导了一个性能度量，该度量根据执行窗口查询所需的桶访问的预期次数来描述任意数据空间组织。由于众所周知，在实际应用中，数据桶访问在频率和执行时间上远远超过对相应目录分页部分的外部访问，因此空间数据结构的数据空间组织对于窗口查询性能至关重要。因此，尽管我们的考虑仅限于数据桶访问，但仍然充分反映了实际情况。

In contrast to purely experimental investigations which present the corresponding performance phenomena for specific implementations, specific object sets and specific window query patterns, our analytical approach exhibits the effects on a conceptual level, independent of data structure and implementation details, and even independent of whether the objects are points or non-point objects. We claim that our analysis sheds some new light on the complex spatial data structure field.

与纯粹的实验研究不同，实验研究展示了特定实现、特定对象集和特定窗口查询模式的相应性能现象，我们的分析方法在概念层面上展示了这些影响，与数据结构和实现细节无关，甚至与对象是点对象还是非点对象无关。我们声称，我们的分析为复杂的空间数据结构领域带来了新的启示。

This paper is organized as follows. The next section reviews the basic concepts of spatial data management and introduces four different classes of user-driven window queries. In section 3, we provide a more formal look at the query classes by deriving a probabilistic model for each of them. In section 4, for each query model and arbitrary data space organizations, we present an analytical evaluation of the window query performance in terms of the expected number of bucket accesses needed to perform a window query. Section 5 states some open problems which are direct implications of the analytical investigations. Section 6 reports on experiments we have performed to assess the performance of well-known split strategies under the four query models. Some further open problems conclude the paper.

本文的组织如下。下一节回顾空间数据管理的基本概念，并介绍四种不同类别的用户驱动窗口查询。在第 3 节中，我们通过为每一类查询推导一个概率模型，对查询类别进行更正式的研究。在第 4 节中，对于每个查询模型和任意数据空间组织，我们根据执行窗口查询所需的桶访问的预期次数对窗口查询性能进行分析评估。第 5 节陈述了一些由分析研究直接引出的开放问题。第 6 节报告了我们为评估四种查询模型下著名的分割策略的性能而进行的实验。本文最后提出了一些进一步的开放问题。

---

<!-- Footnote -->

-This work has been supported by the Deutsche Forschungs-gemeinschaft DFG, and by the ESPRIT II Basic Research Actions Program of the European Community under contract No. 6881 (AMUSING).

- 这项工作得到了德国研究基金会（Deutsche Forschungs - gemeinschaft DFG）以及欧洲共同体 ESPRIT II 基础研究行动项目（合同编号 6881，AMUSING）的支持。

**FernUniversität Hagen, D-5800 Hagen

** 哈根远程大学，德国哈根 5800

§ETH Zürich, CH-8092 Zürich

§ 苏黎世联邦理工学院，瑞士苏黎世 8092

Permission to copy without fee all or part of this material is granted provided that the copies are not made or distributed for direct commercial advantage, the ACM copyright notice and the title of the publication and its date appear, and notice is given that copying is by permission of the Association for Computing Machinery. To copy otherwise, or to republish, requires a fee and/or specific permission.

允许免费复制本材料的全部或部分内容，但前提是复制件不得用于直接商业利益，必须保留 ACM 版权声明、出版物标题及其日期，并注明复制获得了美国计算机协会的许可。否则，复制或重新发布需要支付费用和/或获得特定许可。

ACM-PODS-5/93/Washington, D.C.

ACM - PODS - 5/93/华盛顿特区

© 1993 ACM 0-89791-593-3/93/0005/0214...\$1.50

© 1993 ACM 0 - 89791 - 593 - 3/93/0005/0214... 1.50 美元

<!-- Footnote -->

---

## 2 The Setting

## 2 背景设定

Let us shortly review the basic concepts of spatial data management. Data structures which efficiently support spatial access to geometric objects cluster these objects in data buckets according to their spatial locations. With each data bucket a subspace of the data space, the so-called bucket region, is associated containing all objects of the corresponding bucket. Except for the BANG-File [2] and the cell tree [3], a bucket region is a multidimensional interval.

让我们简要回顾一下空间数据管理的基本概念。能够有效支持对几何对象进行空间访问的数据结构会根据这些对象的空间位置将它们聚类到数据桶中。每个数据桶都关联着数据空间的一个子空间，即所谓的桶区域（bucket region），其中包含了相应桶中的所有对象。除了BANG文件 [2] 和单元树 [3] 之外，桶区域是一个多维区间。

Point data structures usually create bucket regions which form a partition of the data space (see e.g. $\left\lbrack  {2,5,7}\right\rbrack  )$ . Data structures for bounding boxes generate bucket regions which may overlap and do not necessarily cover the entire data space (see e.g. $\left\lbrack  {4,6,8}\right\rbrack$ ).

点数据结构通常会创建能构成数据空间划分的桶区域（例如参见 $\left\lbrack  {2,5,7}\right\rbrack  )$）。用于边界框的数据结构生成的桶区域可能会重叠，并且不一定会覆盖整个数据空间（例如参见 $\left\lbrack  {4,6,8}\right\rbrack$）。

We are interested in the range query performance of a spatial data structure ${DS}\left( G\right)$ ,which is currently storing a set $G$ of geometric objects. We restrict our investigations to so-called window queries where the query window forms a $d$ -dimensional interval. The operation window-query $\left( {w,{DS}\left( G\right) }\right)$ retrieves for the query window $w$ each point object which is located in $w$ ,respectively each bounding box which intersects $w$ .

我们关注的是空间数据结构 ${DS}\left( G\right)$ 的范围查询性能，该结构目前存储着一组几何对象 $G$。我们将研究范围限制在所谓的窗口查询上，其中查询窗口构成一个 $d$ 维区间。窗口查询操作 $\left( {w,{DS}\left( G\right) }\right)$ 会为查询窗口 $w$ 检索位于 $w$ 中的每个点对象，或者与 $w$ 相交的每个边界框。

The window query performance of a data structure heavily depends on the kind of the window queries to be performed, and hence depends on the expected user behavior - which is usually unspecified.

数据结构的窗口查询性能在很大程度上取决于要执行的窗口查询的类型，因此也取决于预期的用户行为 —— 而用户行为通常是未明确指定的。

A closer look turns out that the user may vary the aspect ratio, the location of the query window and the query value which can be the area of the window or the size (cardinality) of the answer set. Let us assume that there is no direct correlation between these parameters. So we can discuss each parameter in turn.

仔细观察会发现，用户可能会改变查询窗口的纵横比、位置以及查询值，查询值可以是窗口的面积或答案集的大小（基数）。我们假设这些参数之间没有直接关联。因此，我们可以依次讨论每个参数。

We assume the aspect ratio to be 1 , i.e. square windows. This seems to be appropriate unless some slope bias is known beforehand, since the expected value of the aspect ratio is 1 if all aspect ratios are equally likely.

我们假设纵横比为 1，即方形窗口。除非事先知道存在某种倾斜偏差，否则这似乎是合适的，因为如果所有纵横比的可能性相等，那么纵横比的期望值就是 1。

Concerning the window location, two possible user behaviors seem to be likely:

关于窗口位置，有两种可能的用户行为似乎比较常见：

- Every part of the data space is equally likely to be requested, i.e. a uniform distribution for the window center is assumed. This assumption models the situation where no user preference is known beforehand, respectively reflects the behavior of novice and occasional users.

- 数据空间的每个部分被请求的可能性相等，即假设窗口中心服从均匀分布。这一假设模拟了事先不知道用户偏好的情况，或者反映了新手和偶尔使用的用户的行为。

- Each object is equally likely to be requested. This situation where queries prefer densely populated parts of the data space can be observed in many applications.

- 每个对象被请求的可能性相等。在许多应用中可以观察到这种查询倾向于数据空间中人口密集部分的情况。

Let us now turn to the specification of the query value. In a user-driven query, the requested part of the data space usually has to be represented on a screen. Then two possible variants of user behavior seem to be likely:

现在让我们来讨论查询值的指定。在用户驱动的查询中，数据空间中被请求的部分通常需要显示在屏幕上。那么似乎有两种可能的用户行为变体：

- The user specifies the query value in terms of the window area. We assume the area to be a constant which is typical for situations where the user specifies queries such that the requested part covers (more or less) the entire screen (zooming facilities neglected).

- 用户根据窗口面积指定查询值。我们假设面积是一个常数，这在用户指定查询使得被请求的部分（或多或少）覆盖整个屏幕（忽略缩放功能）的情况下是很典型的。

- The user always determines the query value with the intention to retrieve the same (constant) number of objects. Here, the cardinality of the answer set is assumed to be constant. This is typical for an experienced user who tends to request an amount of information which is neither overloaded nor insufficient and satisfies his personal needs. In this model, the window area depends on the underlying object population.

- 用户总是以检索相同（恒定）数量的对象为目的来确定查询值。在这里，假设答案集的基数是恒定的。这对于有经验的用户来说是很典型的，他们倾向于请求既不过载也不不足且满足其个人需求的信息量。在这个模型中，窗口面积取决于底层的对象分布。

Combining the two proposed variants of query value, respectively location, results in four different models for user-defined window queries. In the next section, we render them in precisely defined probabilistic models. Such a framework allows for an analytical characterization of arbitrary data space organizations with respect to the underlying query model.

将所提出的两种查询值变体（分别对应位置）进行组合，会得到四种不同的用户定义窗口查询模型。在下一节中，我们将把它们转化为精确定义的概率模型。这样的框架允许针对底层查询模型对任意数据空间组织进行分析性描述。

## 3 Probabilistic models for user-defined window queries

## 3 用户定义窗口查询的概率模型

Let us resume by defining the introduced problem more precisely as follows. Let $d$ be the dimension of the data space we consider, ${S}_{i} = \lbrack 0,1),1 \leq  i \leq  d$ ,and $S = {S}_{1} \times  {S}_{2} \times  \ldots  \times  {S}_{d}$ the $d$ -dimensional data space in which all geometric objects are defined. A geometric object is either a point $p$ given by its coordinates,i.e. $p = \left( {p.{x}_{1},p.{x}_{2},\ldots ,p.{x}_{d}}\right) ,p.{x}_{i} \in  {S}_{i}$ ,or a $d$ -dimensional interval $v = \left\lbrack  {v.{l}_{1},v.{r}_{1}}\right\rbrack   \times  \ldots  \times  \left\lbrack  {v.{l}_{d},v.{r}_{d}}\right\rbrack  ,v.{l}_{i},v.{r}_{i} \in$ ${S}_{i},v \cdot  {l}_{i} \leq  v \cdot  {r}_{i}$ . An interval $v$ can be interpreted as bounding box of an arbitrary non-point object.

让我们通过更精确地定义所引入的问题来继续，具体如下。设 $d$ 为我们所考虑的数据空间的维度，${S}_{i} = \lbrack 0,1),1 \leq  i \leq  d$ ，以及 $S = {S}_{1} \times  {S}_{2} \times  \ldots  \times  {S}_{d}$ 为定义所有几何对象的 $d$ 维数据空间。一个几何对象要么是由其坐标给出的点 $p$ ，即 $p = \left( {p.{x}_{1},p.{x}_{2},\ldots ,p.{x}_{d}}\right) ,p.{x}_{i} \in  {S}_{i}$ ，要么是一个 $d$ 维区间 $v = \left\lbrack  {v.{l}_{1},v.{r}_{1}}\right\rbrack   \times  \ldots  \times  \left\lbrack  {v.{l}_{d},v.{r}_{d}}\right\rbrack  ,v.{l}_{i},v.{r}_{i} \in$ ${S}_{i},v \cdot  {l}_{i} \leq  v \cdot  {r}_{i}$ 。区间 $v$ 可以被解释为任意非点对象的边界框。

Let us assume that for storing the set $G$ of objects the data structure ${DS}\left( G\right)$ currently consumes $m$ consecutive blocks ${B}_{1},{B}_{2},\ldots ,{B}_{m}$ ,the so-called data buckets. Each bucket has a capacity of $c$ objects. With each object $g$ ,a bucket is uniquely associated. The bucket region $R\left( {B}_{i}\right)  \subseteq  S$ of a bucket ${B}_{i}$ is a $d$ -dimensional interval enclosing (the bounding boxes of) all objects in ${B}_{i}$ . For a bucket set $B = \left\{  {{B}_{1},\ldots ,{B}_{m}}\right\}$ we call $R\left( B\right)  = \left\{  {R\left( {B}_{1}\right) ,\ldots ,R\left( {B}_{m}\right) }\right\}$ the corresponding organization of the data space.

让我们假设为了存储对象集合 $G$ ，数据结构 ${DS}\left( G\right)$ 当前占用 $m$ 个连续的块 ${B}_{1},{B}_{2},\ldots ,{B}_{m}$ ，即所谓的数据桶。每个桶的容量为 $c$ 个对象。每个对象 $g$ 都唯一地关联一个桶。桶 ${B}_{i}$ 的桶区域 $R\left( {B}_{i}\right)  \subseteq  S$ 是一个 $d$ 维区间，它包围了 ${B}_{i}$ 中所有对象（的边界框）。对于桶集合 $B = \left\{  {{B}_{1},\ldots ,{B}_{m}}\right\}$ ，我们称 $R\left( B\right)  = \left\{  {R\left( {B}_{1}\right) ,\ldots ,R\left( {B}_{m}\right) }\right\}$ 为数据空间的相应组织方式。

Let ${\mathbb{R}}^{d}$ be the space in which all query windows are defined. The location of a query window $w$ is specified by its center $w.c = \left( {w.{l}_{1} + w.{l}_{2}}\right) /2$ componentwise. We call a query window $w$ legal iff $w.c \in  S$ . Let $W$ be the set of all legal windows. Let ${\overrightarrow{f}}_{c} : S \rightarrow  {\left( {\mathbb{R}}^{ + }\right) }^{d}$ be the (componentwise continuous) density function of the center distribution of legal windows and ${F}_{c}$ : $S \rightarrow  \left\lbrack  {0,1}\right\rbrack$ be the corresponding distribution function. Let $\overrightarrow{{f}_{G}} : S \rightarrow  {\left( {\mathbb{R}}^{ + }\right) }^{d}$ ,resp. ${F}_{G} : S \rightarrow  \left\lbrack  {0,1}\right\rbrack$ ,be the (componentwise continuous) density function, resp. (continuous) distribution function, of the location of the geometric objects. We assume that the location of a non-point object $g$ is uniquely determined by a point belonging to $g$ ,e.g. the center of $g$ . Let $\mathcal{M} : W \rightarrow  \left\lbrack  {0,1}\right\rbrack$ be a probability measure for legal windows.

设 ${\mathbb{R}}^{d}$ 为定义所有查询窗口的空间。查询窗口 $w$ 的位置由其中心 $w.c = \left( {w.{l}_{1} + w.{l}_{2}}\right) /2$ 按分量指定。当且仅当 $w.c \in  S$ 时，我们称查询窗口 $w$ 为合法的。设 $W$ 为所有合法窗口的集合。设 ${\overrightarrow{f}}_{c} : S \rightarrow  {\left( {\mathbb{R}}^{ + }\right) }^{d}$ 为合法窗口中心分布的（按分量连续的）密度函数，${F}_{c}$ : $S \rightarrow  \left\lbrack  {0,1}\right\rbrack$ 为相应的分布函数。设 $\overrightarrow{{f}_{G}} : S \rightarrow  {\left( {\mathbb{R}}^{ + }\right) }^{d}$ ，分别地 ${F}_{G} : S \rightarrow  \left\lbrack  {0,1}\right\rbrack$ ，为几何对象位置的（按分量连续的）密度函数，分别地（连续的）分布函数。我们假设非点对象 $g$ 的位置由属于 $g$ 的一个点唯一确定，例如 $g$ 的中心。设 $\mathcal{M} : W \rightarrow  \left\lbrack  {0,1}\right\rbrack$ 为合法窗口的一个概率测度。

A window query model $\mathcal{{WQM}}$ is a 4-tuple with the components aspect ratio ar (which is 1:1 for all models), window measure $\mathcal{M}$ ,the window value $\mathcal{M}\left( w\right)$ which is a constant ${c}_{\mathcal{M}}$ for all legal windows $w$ ,and center distribution ${F}_{c}$ :

窗口查询模型 $\mathcal{{WQM}}$ 是一个四元组，其组成部分包括宽高比 ar（所有模型的宽高比均为 1:1）、窗口度量 $\mathcal{M}$、窗口值 $\mathcal{M}\left( w\right)$（对于所有合法窗口 $w$ 而言，窗口值是一个常数 ${c}_{\mathcal{M}}$）以及中心分布 ${F}_{c}$：

$$
\mathcal{{WQM}} = \left( {{ar},\mathcal{M},{c}_{\mathcal{M}},{F}_{c}}\right) .
$$

Model 1 is characterized by choosing the conventional area function $A$ as window measure $\mathcal{M}$ ,constant window area ${c}_{A}$ for the window value and a uniformly distributed window center:

模型 1 的特征是选择传统的面积函数 $A$ 作为窗口度量 $\mathcal{M}$，选择恒定的窗口面积 ${c}_{A}$ 作为窗口值，并采用均匀分布的窗口中心：

$$
\mathcal{{WQ}}{\mathcal{M}}_{1} = \left( {1 : 1,A,{c}_{A},U\left\lbrack  S\right\rbrack  }\right) .
$$

In model 2, the window measure is retained but the center distribution equals the object distribution:

在模型 2 中，保留窗口度量，但中心分布等同于对象分布：

$$
\mathcal{W}\mathcal{Q}{\mathcal{M}}_{2} = \left( {1 : 1,A,{c}_{A},{F}_{G}}\right) .
$$

In model 3,we assume the window measure $\mathcal{M}$ to be the answer size of the query. Hence, the window measure is ${F}_{W} : W \rightarrow  \left\lbrack  {0,1}\right\rbrack$ where ${F}_{W}\left( w\right)  =$ ${\int }_{S \cap  w}{\overrightarrow{f}}_{G}\left( p\right) {dp}$ ,the window value is some constant ${c}_{{F}_{W}}$ , and the window center is uniformly distributed (Note that the window areas depend on the object distribution $\left. {{F}_{W}\text{.}}\right)$ ,

在模型 3 中，我们假设窗口度量 $\mathcal{M}$ 为查询的答案大小。因此，窗口度量为 ${F}_{W} : W \rightarrow  \left\lbrack  {0,1}\right\rbrack$，其中 ${F}_{W}\left( w\right)  =$ ${\int }_{S \cap  w}{\overrightarrow{f}}_{G}\left( p\right) {dp}$，窗口值为某个常数 ${c}_{{F}_{W}}$，并且窗口中心呈均匀分布（请注意，窗口面积取决于对象分布 $\left. {{F}_{W}\text{.}}\right)$）

$$
\mathcal{W}\mathcal{Q}{\mathcal{M}}_{3} = \left( {1 : 1,{F}_{W},{c}_{{F}_{W}},U\left\lbrack  S\right\rbrack  }\right) .
$$

Model 4 is similar to model 3 except for the window center distribution ${F}_{c}$ which equals the object distribution ${F}_{G}$ :

模型 4 与模型 3 类似，只是窗口中心分布 ${F}_{c}$ 等同于对象分布 ${F}_{G}$：

$$
\mathcal{W}\mathcal{Q}{\mathcal{M}}_{4} = \left( {1 : 1,{F}_{W},{c}_{{F}_{W}},{F}_{G}}\right) .
$$

The four query models presented above provide a framework for the following investigations on window query performance.

上述四种查询模型为后续关于窗口查询性能的研究提供了一个框架。

## 4 Analytical results on window query performance

## 4 窗口查询性能的分析结果

For each of the four window query models, we characterize data space organizations in terms of the expected number of bucket accesses needed to perform a window query. Without loss of generality and only for simplicity reasons,we choose $d = 2$ for further considerations. This reduces bounding boxes, bucket regions, and query windows to two-dimensional rectangles.

对于这四种窗口查询模型中的每一种，我们根据执行窗口查询所需的预期桶访问次数来描述数据空间组织。为了不失一般性且仅为简化起见，我们选择 $d = 2$ 进行进一步的考虑。这将边界框、桶区域和查询窗口简化为二维矩形。

For $\mathcal{{WQ}}{\mathcal{M}}_{k},1 \leq  k \leq  4$ ,and data space organization $R\left( B\right)  = \left\{  {R\left( {B}_{1}\right) ,\ldots ,R\left( {B}_{m}\right) }\right\}$ ,let ${P}_{k}\left( {w \cap  R\left( {B}_{i}\right)  \neq  \varnothing }\right)$ be the probability that the window $w$ intersects bucket region $R\left( {B}_{i}\right)$ ,and ${P}_{k}\left( {w \cap  R\left( B\right) ;j}\right)$ be the probability that window $w$ intersects exactly $j$ bucket regions in $R\left( B\right)$ . Then for a data space organization $R\left( B\right)$ ,the expected number of buckets intersecting a query window in model $\mathrm{k}$ - we call it the performance measure for model $\mathrm{k}$ - is given by

对于 $\mathcal{{WQ}}{\mathcal{M}}_{k},1 \leq  k \leq  4$ 和数据空间组织 $R\left( B\right)  = \left\{  {R\left( {B}_{1}\right) ,\ldots ,R\left( {B}_{m}\right) }\right\}$，设 ${P}_{k}\left( {w \cap  R\left( {B}_{i}\right)  \neq  \varnothing }\right)$ 为窗口 $w$ 与桶区域 $R\left( {B}_{i}\right)$ 相交的概率，设 ${P}_{k}\left( {w \cap  R\left( B\right) ;j}\right)$ 为窗口 $w$ 恰好与 $R\left( B\right)$ 中的 $j$ 个桶区域相交的概率。那么对于数据空间组织 $R\left( B\right)$，模型 $\mathrm{k}$ 中与查询窗口相交的桶的预期数量——我们称之为模型 $\mathrm{k}$ 的性能度量——由下式给出

$$
\mathcal{{PM}}\left( {\mathcal{{WQ}}{\mathcal{M}}_{k},R\left( B\right) }\right)  = \mathop{\sum }\limits_{{j = 0}}^{m}j \cdot  {P}_{k}\left( {w \cap  R\left( B\right) ;j}\right) .
$$

The following Lemma facilitates the computation of $\mathcal{{PM}}\left( {\mathcal{{WQ}}{\mathcal{M}}_{k},R\left( B\right) }\right)$ .

以下引理有助于计算 $\mathcal{{PM}}\left( {\mathcal{{WQ}}{\mathcal{M}}_{k},R\left( B\right) }\right)$。

Lemma.

引理。

$$
\mathop{\sum }\limits_{{j = 0}}^{m}j \cdot  {P}_{k}\left( {w \cap  R\left( B\right) ;j}\right)  = \mathop{\sum }\limits_{{i = 1}}^{m}{P}_{k}\left( {w \cap  R\left( {B}_{i}\right)  \neq  \varnothing }\right) 
$$

Proof (by induction on $m$ ). To improve readability we omit index $k$ . We extend the notation $R\left( B\right)$ of a data space organization to $R\left( {B}^{\left( m\right) }\right) ,m \in  {\mathbb{N}}_{0}$ ,to indicate the cardinality of $R\left( B\right)$ .

证明（对 $m$ 进行归纳）。为了提高可读性，我们省略索引 $k$。我们将数据空间组织的符号 $R\left( B\right)$ 扩展为 $R\left( {B}^{\left( m\right) }\right) ,m \in  {\mathbb{N}}_{0}$，以表示 $R\left( B\right)$ 的基数。

Basis $m = 0$ : We get

基础情况 $m = 0$：我们得到

$$
\mathop{\sum }\limits_{{j = 0}}^{0}j \cdot  P\left( {w \cap  R\left( {B}^{\left( 0\right) }\right) ;j}\right)  = 0 = \mathop{\sum }\limits_{{i = 1}}^{0}P\left( {w \cap  R\left( {B}_{i}\right)  \neq  \varnothing }\right) .
$$

By induction hypothesis

根据归纳假设

$$
\mathop{\sum }\limits_{{j = 0}}^{m}j \cdot  P\left( {w \cap  R\left( {B}^{\left( m\right) }\right) ;j}\right)  = \mathop{\sum }\limits_{{i = 1}}^{m}P\left( {w \cap  R\left( {B}_{i}\right)  \neq  \varnothing }\right) 
$$

holds for all $m \in  \mathbb{N}$ .

对于所有 $m \in  \mathbb{N}$ 都成立。

Induction step $m \rightarrow  m + 1$ :

归纳步骤 $m \rightarrow  m + 1$：

We focus on region $R\left( {B}_{m + 1}\right)  \in  R\left( {B}^{\left( m + 1\right) }\right)$ . Considering $R\left( {B}_{m + 1}\right)$ separatly and removing it from $R\left( {B}^{\left( m + 1\right) }\right)$ yields the decreased set

我们关注区域 $R\left( {B}_{m + 1}\right)  \in  R\left( {B}^{\left( m + 1\right) }\right)$。单独考虑 $R\left( {B}_{m + 1}\right)$ 并将其从 $R\left( {B}^{\left( m + 1\right) }\right)$ 中移除，得到缩减后的集合

$R\left( {B}^{\left( m\right) }\right)  = R\left( {B}^{\left( m + 1\right) }\right)  \smallsetminus  \left\{  {R\left( {B}_{m + 1}\right) }\right\}$ . We get

$R\left( {B}^{\left( m\right) }\right)  = R\left( {B}^{\left( m + 1\right) }\right)  \smallsetminus  \left\{  {R\left( {B}_{m + 1}\right) }\right\}$。我们得到

$$
\mathop{\sum }\limits_{{j = 0}}^{{m + 1}}j \cdot  P\left( {w \cap  R\left( {B}^{\left( m + 1\right) }\right) ;j}\right) 
$$

$$
 = \mathop{\sum }\limits_{{j = 0}}^{{m + 1}}j \cdot  \left\lbrack  {P\left( {w \cap  R\left( {B}^{\left( m\right) }\right) ;j}\right)  \cdot  P\left( {w \cap  R\left( {B}_{m + 1}\right)  = \varnothing }\right) }\right. 
$$

$$
\left. {+P\left( {w \cap  R\left( {B}^{\left( m\right) }\right) ;j - 1}\right)  \cdot  P\left( {w \cap  R\left( {B}_{m + 1}\right)  \neq  \varnothing }\right) }\right\rbrack  
$$

$$
 = \mathop{\sum }\limits_{{j = 0}}^{{m + 1}}j \cdot  \left\lbrack  \left( {P\left( {w \cap  R\left( {B}^{\left( m\right) }\right) ;j - 1}\right)  - P\left( {w \cap  R\left( {B}^{\left( m\right) }\right) ;j}\right) }\right) \right. 
$$

$$
\left. {\cdot P\left( {w \cap  R\left( {B}_{m + 1}\right)  \neq  \varnothing }\right)  + P\left( {w \cap  R\left( {B}^{\left( m\right) }\right) ;j}\right) }\right\rbrack  
$$

$$
 = P\left( {w \cap  R\left( {B}_{m + 1}\right)  \neq  \varnothing }\right) 
$$

$$
 \cdot  \mathop{\sum }\limits_{{j = 0}}^{{m + 1}}j \cdot  \left\lbrack  {P\left( {w \cap  R\left( {B}^{\left( m\right) }\right) ;j - 1}\right)  - P\left( {w \cap  R\left( {B}^{\left( m\right) }\right) ;j}\right) }\right\rbrack  
$$

$$
 + \mathop{\sum }\limits_{{j = 0}}^{{m + 1}}j \cdot  P\left( {w \cap  R\left( {B}^{\left( m\right) }\right) ;j}\right) 
$$

The second sum meets the induction hypothesis, hence it remains to prove

第二个求和满足归纳假设，因此只需证明

$$
\mathop{\sum }\limits_{{j = 0}}^{{m + 1}}j \cdot  \left\lbrack  {P\left( {w \cap  R\left( {B}^{\left( m\right) }\right) ;j - 1}\right)  - P\left( {w \cap  R\left( {B}^{\left( m\right) }\right) ;j}\right) }\right\rbrack   = 1.
$$

We have

我们有

$$
\mathop{\sum }\limits_{{j = 0}}^{{m + 1}}j \cdot  \left\lbrack  {P\left( {w \cap  R\left( {B}^{\left( m\right) }\right) ;j - 1}\right)  - P\left( {w \cap  R\left( {B}^{\left( m\right) }\right) ;j}\right) }\right\rbrack  
$$

$$
 = \mathop{\sum }\limits_{{j = 0}}^{m}\left( {j + 1}\right)  \cdot  P\left( {w \cap  R\left( {B}^{\left( m\right) }\right) ;j}\right) 
$$

$$
 - \mathop{\sum }\limits_{{j = 0}}^{{m + 1}}j \cdot  P\left( {w \cap  R\left( {B}^{\left( m\right) }\right) ;j}\right) 
$$

$$
 = \mathop{\sum }\limits_{{j = 0}}^{m}P\left( {w \cap  R\left( {B}^{\left( m\right) }\right) ;j}\right) 
$$

$$
 = 1
$$

<!-- Media -->

<!-- figureText: $S$ -->

<img src="https://cdn.noedgeai.com/0195c908-c558-75bd-bf47-88151d336725_3.jpg?x=258&y=1507&w=513&h=308&r=0"/>

Figure 1: Representatives of query windows according to a bucket region.

图 1：根据桶区域划分的查询窗口代表。

<!-- Media -->

The Lemma tells us that for the computation of $\mathcal{{PM}}\left( {{\mathcal{{WQM}}}_{k},R\left( B\right) }\right)$ ,it remains to compute ${P}_{k}(w \cap$ $\left. {R\left( {B}_{i}\right)  \neq  \varnothing }\right)$ for every bucket region $R\left( {B}_{i}\right)  \in  R\left( B\right)$ . For this purpose, let us consider a single bucket region $R\left( {B}_{i}\right)$ . Every legal window belongs to one of the following three classes:

引理告诉我们，为了计算 $\mathcal{{PM}}\left( {{\mathcal{{WQM}}}_{k},R\left( B\right) }\right)$，只需为每个桶区域 $R\left( {B}_{i}\right)  \in  R\left( B\right)$ 计算 ${P}_{k}(w \cap$ $\left. {R\left( {B}_{i}\right)  \neq  \varnothing }\right)$。为此，让我们考虑单个桶区域 $R\left( {B}_{i}\right)$。每个合法窗口属于以下三类之一：

- Queries with centers inside $R\left( {B}_{i}\right)$ ,

- 中心位于 $R\left( {B}_{i}\right)$ 内的查询，

- Queries with centers outside $R\left( {B}_{i}\right)$ ,but intersecting $R\left( {B}_{i}\right)$ ,

- 中心位于 $R\left( {B}_{i}\right)$ 外，但与 $R\left( {B}_{i}\right)$ 相交的查询，

- Queries which do not intersect $R\left( {B}_{i}\right)$ .

- 与 $R\left( {B}_{i}\right)$ 不相交的查询。

Figure 1 depicts a representative of each class.

图 1 展示了每一类的代表。

For bucket region $R\left( {B}_{i}\right)$ ,let ${R}_{c}\left( {B}_{i}\right)$ be the domain in which the centers of all windows intersecting $R\left( {B}_{i}\right)$ are located. Hence, the probability that a random window intersects $R\left( {B}_{i}\right)$ equals the probability that the window center falls into domain ${R}_{c}\left( {B}_{i}\right)$ . Obviously, ${R}_{c}\left( {B}_{i}\right)$ depends on the underlying query model. To be precise, ${R}_{c}\left( {B}_{i}\right)$ depends on the window value ${c}_{\mathcal{M}}$ and so,of course,on the window measure $\mathcal{M}$ .

对于桶区域 $R\left( {B}_{i}\right)$，设 ${R}_{c}\left( {B}_{i}\right)$ 为所有与 $R\left( {B}_{i}\right)$ 相交的窗口的中心所在的区域。因此，随机一个窗口与 $R\left( {B}_{i}\right)$ 相交的概率等于窗口中心落入区域 ${R}_{c}\left( {B}_{i}\right)$ 的概率。显然，${R}_{c}\left( {B}_{i}\right)$ 取决于底层的查询模型。确切地说，${R}_{c}\left( {B}_{i}\right)$ 取决于窗口值 ${c}_{\mathcal{M}}$，当然也取决于窗口度量 $\mathcal{M}$。

We exemplarily explain the computation of ${P}_{k}(w \cap$ $\left. {R\left( {B}_{i}\right)  \neq  \varnothing }\right)$ for the first model because analogous considerations hold for the remaining models.

我们以第一个模型为例解释 ${P}_{k}(w \cap$ $\left. {R\left( {B}_{i}\right)  \neq  \varnothing }\right)$ 的计算，因为对其余模型也有类似的考虑。

In this model, query windows are squares with fixed width and height $\sqrt{{c}_{A}}$ ,and their centers occur at each possible position in the data space with equal probability.

在这个模型中，查询窗口是宽度和高度固定为 $\sqrt{{c}_{A}}$ 的正方形，并且它们的中心在数据空间的每个可能位置出现的概率相等。

<!-- Media -->

<!-- figureText: $S$ $R\left( B\right)$ -->

<img src="https://cdn.noedgeai.com/0195c908-c558-75bd-bf47-88151d336725_3.jpg?x=1024&y=1171&w=511&h=312&r=0"/>

Figure 2: A domain ${R}_{c}\left( B\right)$ in model 1 .

图 2：模型 1 中的区域 ${R}_{c}\left( B\right)$。

<!-- Media -->

For simplicity, let us assume for a while that each region $R\left( {B}_{i}\right)$ is far enough off the data space boundaries such that domain ${R}_{c}\left( {B}_{i}\right)$ is just the region $R\left( {B}_{i}\right)$ inflated by a frame of width $\sqrt{{c}_{A}}/2$ . If $R\left( {B}_{i}\right)$ has width $R\left( {B}_{i}\right) .L$ and height $R\left( {B}_{i}\right) .H$ then the probability that $w$ intersects $R\left( {B}_{i}\right)$ is determined by the area of ${R}_{c}\left( {B}_{i}\right)$ which is $\left( {R\left( {B}_{i}\right) .L + \sqrt{{c}_{A}}}\right)  \cdot  \left( {R\left( {B}_{i}\right) .H + \sqrt{{c}_{A}}}\right)$ . So we get (see figure 2)

为简单起见，让我们暂时假设每个区域 $R\left( {B}_{i}\right)$ 离数据空间边界足够远，使得域 ${R}_{c}\left( {B}_{i}\right)$ 恰好是区域 $R\left( {B}_{i}\right)$ 向外扩展宽度为 $\sqrt{{c}_{A}}/2$ 的边框后的区域。如果 $R\left( {B}_{i}\right)$ 的宽度为 $R\left( {B}_{i}\right) .L$ 且高度为 $R\left( {B}_{i}\right) .H$，那么 $w$ 与 $R\left( {B}_{i}\right)$ 相交的概率由 ${R}_{c}\left( {B}_{i}\right)$ 的面积决定，该面积为 $\left( {R\left( {B}_{i}\right) .L + \sqrt{{c}_{A}}}\right)  \cdot  \left( {R\left( {B}_{i}\right) .H + \sqrt{{c}_{A}}}\right)$。因此我们得到（见图 2）

$$
\overline{\mathcal{{PM}}}\left( {\mathcal{{WQ}}{\mathcal{M}}_{1},R\left( B\right) }\right)  = 
$$

$$
 = \mathop{\sum }\limits_{{i = 1}}^{m}\left( {R\left( {B}_{i}\right) .L + \sqrt{{c}_{A}}}\right)  \cdot  \left( {R\left( {B}_{i}\right) .H + \sqrt{{c}_{A}}}\right) 
$$

$$
 = \mathop{\sum }\limits_{{i = 1}}^{m}R\left( {B}_{i}\right)  \cdot  L \cdot  R\left( {B}_{i}\right)  \cdot  H
$$

$$
 + \sqrt{{c}_{A}} \cdot  \mathop{\sum }\limits_{{i = 1}}^{m}\left( {R\left( {B}_{i}\right)  \cdot  L + R\left( {B}_{i}\right)  \cdot  H}\right) 
$$

$$
 + {c}_{A} \cdot  m\text{.}
$$

What do we gain from this simple performance measure? In geometric terms, this function combines the sum of all region areas, the weighted sum of all region perimeters, and the weighted number of regions. It should be mentioned that for the first time the strong influence of the region perimeters is revealed. To our knowledge,so far only in the ${\mathrm{R}}^{ * }$ -tree simulations to a certain extent region perimeters have been taken into account [1]. Besides this observation, a number of plausibility arguments can now quantitatively be illustrated. For instance,the term ${c}_{A} \cdot  m$ tells us that high bucket utilization is a more important factor if query windows are larger. On the other hand, very small query windows make the term $\mathop{\sum }\limits_{{i = 1}}^{m}R\left( {B}_{i}\right) .L$ . $R\left( {B}_{i}\right) .H$ dominate the others. Whenever the data space organization partitions the data space, $\mathop{\sum }\limits_{{i = 1}}^{m}R\left( {B}_{i}\right) .L$ . $R\left( {B}_{i}\right) .H$ equals 1,no matter how regions are chosen. Then for query windows with ${c}_{A} \ll  R\left( {B}_{i}\right) .L + R\left( {B}_{i}\right) .H$ , for any region $R\left( {B}_{i}\right)  \in  R\left( B\right)$ the term ${c}_{A} \cdot  m$ is negligible, and the sum of the perimeters determines the efficiency. For query windows ensuring ${c}_{A} \gg  R\left( {B}_{i}\right) .L + R\left( {B}_{i}\right) .H$ , the term ${c}_{A} \cdot  m$ ,or in other words the number of buckets, respectively the bucket storage utilization, is the significant part of the formula. Note that the latter arguments substantiate common opinions and experiments in the spatial data field.

我们从这个简单的性能度量中能得到什么呢？从几何角度来看，这个函数综合了所有区域面积之和、所有区域周长的加权和以及区域的加权数量。值得一提的是，首次揭示了区域周长的强大影响。据我们所知，到目前为止，仅在 ${\mathrm{R}}^{ * }$ -树模拟中在一定程度上考虑了区域周长 [1]。除了这一观察结果之外，现在可以定量地说明一些合理性论据。例如，项 ${c}_{A} \cdot  m$ 告诉我们，如果查询窗口较大，那么高桶利用率是一个更重要的因素。另一方面，非常小的查询窗口会使项 $\mathop{\sum }\limits_{{i = 1}}^{m}R\left( {B}_{i}\right) .L$ . $R\left( {B}_{i}\right) .H$ 比其他项更占主导地位。每当数据空间组织对数据空间进行划分时，无论如何选择区域，$\mathop{\sum }\limits_{{i = 1}}^{m}R\left( {B}_{i}\right) .L$ . $R\left( {B}_{i}\right) .H$ 都等于 1。那么对于满足 ${c}_{A} \ll  R\left( {B}_{i}\right) .L + R\left( {B}_{i}\right) .H$ 的查询窗口，对于任何区域 $R\left( {B}_{i}\right)  \in  R\left( B\right)$，项 ${c}_{A} \cdot  m$ 都可以忽略不计，此时周长之和决定了效率。对于确保 ${c}_{A} \gg  R\left( {B}_{i}\right) .L + R\left( {B}_{i}\right) .H$ 的查询窗口，项 ${c}_{A} \cdot  m$，换句话说就是桶的数量，或者说桶存储利用率，是该公式的重要部分。请注意，后面这些论据证实了空间数据领域的常见观点和实验结果。

<!-- Media -->

<!-- figureText: $S$ ${R}_{c}\left( B\right)$ -->

<img src="https://cdn.noedgeai.com/0195c908-c558-75bd-bf47-88151d336725_4.jpg?x=256&y=1397&w=524&h=339&r=0"/>

Figure 3: Boundary considerations in model 1.

图 3：模型 1 中的边界考虑。

<!-- Media -->

To get the exact performance measure for model 1 we must take the data space boundaries into account. If a region $R\left( {B}_{i}\right)$ comes close enough to a data space boundary then domain ${R}_{c}\left( {B}_{i}\right)$ is not just region $R\left( {B}_{i}\right)$ inflated by a frame of width $\sqrt{{c}_{A}}/2$ but the restriction of the inflated $R\left( {B}_{i}\right)$ to $S$ (see figure 3). (Remember that this holds for each ${R}_{c}\left( {B}_{i}\right)$ by definition.) This leads to

为了得到模型 1 的精确性能度量，我们必须考虑数据空间边界。如果一个区域 $R\left( {B}_{i}\right)$ 足够接近数据空间边界，那么域 ${R}_{c}\left( {B}_{i}\right)$ 就不只是区域 $R\left( {B}_{i}\right)$ 向外扩展宽度为 $\sqrt{{c}_{A}}/2$ 的边框后的区域，而是扩展后的 $R\left( {B}_{i}\right)$ 在 $S$ 上的限制区域（见图 3）。（请记住，根据定义，这对每个 ${R}_{c}\left( {B}_{i}\right)$ 都成立。）这导致

$$
\mathcal{{PM}}\left( {\mathcal{{WQ}}{\mathcal{M}}_{1},R\left( B\right) }\right)  = \mathop{\sum }\limits_{{i = 1}}^{m}A\left( {{R}_{c}\left( {B}_{i}\right) }\right) .
$$

Analogous considerations lead to the performance measures of the remaining models. In model 2, the domains ${R}_{c}\left( {B}_{i}\right)$ are identical to those in model 1 . But instead of simply taking the area of ${R}_{c}\left( {B}_{i}\right)$ - an implication of the uniform window center distribution $- {R}_{c}\left( {B}_{i}\right)$ must now be valued by the window measure ${F}_{W} :$

类似的考虑会得出其余模型的性能度量。在模型2中，区域 ${R}_{c}\left( {B}_{i}\right)$ 与模型1中的区域相同。但现在不能简单地取 ${R}_{c}\left( {B}_{i}\right)$ 的面积，而是必须根据窗口度量 ${F}_{W} :$ 对均匀窗口中心分布 $- {R}_{c}\left( {B}_{i}\right)$ 的含义进行评估

$$
\mathcal{{PM}}\left( {\mathcal{W}\mathcal{Q}{\mathcal{M}}_{2},R\left( B\right) }\right)  = \mathop{\sum }\limits_{{i = 1}}^{m}{F}_{W}\left( {{R}_{c}\left( {B}_{i}\right) }\right) .
$$

Along these lines, the performance measure for model 3 can easily be written down as

按照这些思路，模型3的性能度量可以很容易地写成

$$
\mathcal{{PM}}\left( {\mathcal{{WQ}}{\mathcal{M}}_{3},R\left( B\right) }\right)  = \mathop{\sum }\limits_{{i = 1}}^{m}A\left( {{R}_{c}\left( {B}_{i}\right) }\right) .
$$

However, it is not a trivial task to determine the domains ${R}_{c}\left( {B}_{i}\right)$ in $\mathcal{{WQ}}{\mathcal{M}}_{3}$ . By definition,domain ${R}_{c}\left( {B}_{i}\right)$ is the set of the centers of all legal windows $w$ such that window value $\mathcal{M}\left( w\right)  = {c}_{\mathcal{M}}$ and $w$ intersects region $R\left( {B}_{i}\right)$ . Since ${c}_{\mathcal{M}} = {c}_{{F}_{W}}$ ,i.e. the answer size is assumed to be constant, the window area depends on the location of the window center,and domain ${R}_{c}\left( {B}_{i}\right)$ has a non-rectilinear shape depending on ${F}_{G}$ ,although bucket region $R\left( {B}_{i}\right)$ is always a rectangle. To get an impression of how complicated the determination of the domain ${R}_{c}\left( {B}_{i}\right)$ for a region $R\left( {B}_{i}\right)$ can be,we provide the following example.

然而，确定 $\mathcal{{WQ}}{\mathcal{M}}_{3}$ 中的区域 ${R}_{c}\left( {B}_{i}\right)$ 并非易事。根据定义，区域 ${R}_{c}\left( {B}_{i}\right)$ 是所有合法窗口 $w$ 的中心的集合，使得窗口值 $\mathcal{M}\left( w\right)  = {c}_{\mathcal{M}}$ 与 $w$ 与区域 $R\left( {B}_{i}\right)$ 相交。由于 ${c}_{\mathcal{M}} = {c}_{{F}_{W}}$ ，即假设答案大小是恒定的，窗口面积取决于窗口中心的位置，并且尽管桶区域 $R\left( {B}_{i}\right)$ 始终是一个矩形，但区域 ${R}_{c}\left( {B}_{i}\right)$ 具有取决于 ${F}_{G}$ 的非直线形状。为了了解确定区域 $R\left( {B}_{i}\right)$ 的区域 ${R}_{c}\left( {B}_{i}\right)$ 有多复杂，我们提供以下示例。

Example. For model 3, we assume a non-uniform but still simple object distribution given by the vector-valued density function ${\overrightarrow{f}}_{G}\left( p\right)  = \left( {1,{2p}.{x}_{2}}\right)$ for $p =$ $\left( {p.{x}_{1},p.{x}_{2}}\right)$ ,and a window value ${c}_{{F}_{W}} = {0.01}$ . To avoid problems incured by data space boundaries we choose the bucket region $R\left( {B}_{i}\right)  = \left\lbrack  {{0.4},{0.6}}\right\rbrack   \times  \left\lbrack  {{0.6},{0.7}}\right\rbrack$ . The area of a query square $w$ depends on the location of its center w.c. After some calculations we yield for window $w$ the area $A\left( w\right)  = \frac{001}{{2w}.c.{x}_{2}}$ and the side length $l\left( w\right)  = \sqrt{A\left( w\right) }$ . To get the lower boundary of ${R}_{c}\left( {B}_{i}\right)$ we compute the curve of all window centers whose associated window just touch the lower boundary of region $R\left( {B}_{i}\right)$ . Hence, we have to solve for $w.c.{x}_{2}$ the equation ${0.6} - w.c.{x}_{2} =$ $l\left( w\right) /2$ . The equations $c.w.{x}_{2} - {0.7} = l\left( w\right) /2,{0.4} -$ w.c. ${x}_{1} = l\left( w\right) /2,$ w.c. ${x}_{1} - {0.6} = l\left( w\right) /2$ ,respectively, for the upper, left and right boundaries, respectively, are treated analogously. The resulting region ${R}_{c}\left( {B}_{i}\right)$ is depicted in figure 4.

示例。对于模型3，我们假设由向量值密度函数 ${\overrightarrow{f}}_{G}\left( p\right)  = \left( {1,{2p}.{x}_{2}}\right)$ 给出的非均匀但仍然简单的对象分布，其中 $p =$ $\left( {p.{x}_{1},p.{x}_{2}}\right)$ ，以及一个窗口值 ${c}_{{F}_{W}} = {0.01}$ 。为了避免数据空间边界带来的问题，我们选择桶区域 $R\left( {B}_{i}\right)  = \left\lbrack  {{0.4},{0.6}}\right\rbrack   \times  \left\lbrack  {{0.6},{0.7}}\right\rbrack$ 。查询正方形 $w$ 的面积取决于其中心w.c.的位置。经过一些计算，我们得出窗口 $w$ 的面积为 $A\left( w\right)  = \frac{001}{{2w}.c.{x}_{2}}$ ，边长为 $l\left( w\right)  = \sqrt{A\left( w\right) }$ 。为了得到 ${R}_{c}\left( {B}_{i}\right)$ 的下边界，我们计算所有相关窗口刚好触及区域 $R\left( {B}_{i}\right)$ 下边界的窗口中心的曲线。因此，我们必须求解 $w.c.{x}_{2}$ 的方程 ${0.6} - w.c.{x}_{2} =$ $l\left( w\right) /2$ 。分别针对上、左和右边界的方程 $c.w.{x}_{2} - {0.7} = l\left( w\right) /2,{0.4} -$ w.c. ${x}_{1} = l\left( w\right) /2,$ w.c. ${x}_{1} - {0.6} = l\left( w\right) /2$ ，处理方式类似。得到的区域 ${R}_{c}\left( {B}_{i}\right)$ 如图4所示。

The transition from model 3 to model 4 is analogous to the transition from model 1 to model 2. Domains ${R}_{c}\left( {B}_{i}\right)$ in model 4 are identical to those in model 3 . But as before,instead of taking the areas of the ${R}_{c}\left( {B}_{i}\right)$ as values,in this case the ${R}_{c}\left( {B}_{i}\right)$ must be valued by ${F}_{W}$ before summing them up. Altogether we get

从模型3到模型4的转变类似于从模型1到模型2的转变。模型4中的区域${R}_{c}\left( {B}_{i}\right)$与模型3中的区域相同。但和之前一样，在这种情况下，在对${R}_{c}\left( {B}_{i}\right)$求和之前，必须用${F}_{W}$对${R}_{c}\left( {B}_{i}\right)$进行赋值，而不是将${R}_{c}\left( {B}_{i}\right)$的面积作为值。总体上我们得到

$$
\mathcal{{PM}}\left( {\mathcal{{WQ}}{\mathcal{M}}_{4},R\left( B\right) }\right)  = \mathop{\sum }\limits_{{i = 1}}^{m}{F}_{W}\left( {{R}_{c}\left( {B}_{i}\right) }\right) .
$$

<!-- Media -->

<!-- figureText: $S = \lbrack 0,1{)}^{2}$ -->

<img src="https://cdn.noedgeai.com/0195c908-c558-75bd-bf47-88151d336725_5.jpg?x=294&y=202&w=434&h=434&r=0"/>

Figure 4: Non-rectilinear domain ${R}_{c}\left( {B}_{i}\right)$ from the example.

图4：示例中的非直线区域${R}_{c}\left( {B}_{i}\right)$。

<!-- Media -->

## 5 Obvious questions

## 5 明显的问题

In the preceeding section, for each of the four query models we derived a performance measure which characterizes data space organizations in terms of the expected number of bucket accesses needed to perform a random window query. Our investigations give rise to some obvious questions.

在上一节中，对于四个查询模型中的每一个，我们都推导出了一个性能度量，该度量根据执行随机窗口查询所需的预期桶访问次数来描述数据空间组织。我们的研究引发了一些明显的问题。

For example,for an object set $G$ and a window query model WQM

例如，对于一个对象集$G$和一个窗口查询模型WQM

What is an optimal data space organization? and Which data structure, resp. corresponding insertion algorithm, achieves an optimal data space organization?

什么是最优的数据空间组织？哪种数据结构，或者相应的插入算法，能够实现最优的数据空间组织？

We must admit that we have no answers yet. After all, when addressing the second question we have gained some further insight into spatial data structures we want to report on in the following.

我们必须承认，我们目前还没有答案。毕竟，在解决第二个问题时，我们对空间数据结构有了一些进一步的了解，我们将在下面进行报告。

For sake of simplicity, we concentrate on data structures for points. These data structures usually partition the data space, i.e. when an object insertion causes a data bucket overflow the corresponding bucket region $R\left( B\right)$ is partitioned by a splitline into two bucket regions $R\left( {B}_{1}\right)$ and $R\left( {B}_{2}\right)$ and the objects in $R\left( B\right)$ are distributed over the corresponding two buckets. It is important for the efficiency of a spatial data structure that for any data bucket to be split the choice of the split line depends only on the corresponding bucket region, i.e. can be chosen independently of all other bucket regions. This locality criterion for binary splits naturally leads to binary tree directories for the bookkeeping of the binary splits, resp. data space organization.

为了简单起见，我们专注于点的数据结构。这些数据结构通常会对数据空间进行划分，即当对象插入导致数据桶溢出时，相应的桶区域$R\left( B\right)$会被一条分割线划分为两个桶区域$R\left( {B}_{1}\right)$和$R\left( {B}_{2}\right)$，并且$R\left( B\right)$中的对象会分布到相应的两个桶中。对于空间数据结构的效率来说，重要的是，对于任何要分割的数据桶，分割线的选择仅取决于相应的桶区域，即可以独立于所有其他桶区域进行选择。这种二进制分割的局部性准则自然会导致使用二叉树目录来记录二进制分割，或者说数据空间组织。

The following observation is crucial for the data structure performance. As long as the data structure is flexible enough to support arbitrary split strategies, the choice of the split strategy determines the performance efficiency. This perception immediately poses the next question:

以下观察对于数据结构的性能至关重要。只要数据结构足够灵活以支持任意分割策略，分割策略的选择就决定了性能效率。这种认识立即引出了下一个问题：

For query model $k$ ,what is the best binary split strategy?

对于查询模型$k$，什么是最佳的二进制分割策略？

Unfortunately, we again cannot provide an answer. It is clear, that carrying the optimality criterion of the global situation over to the local situation of a bucket split will not achieve the desired effect. A sound solution will be based on stochastic optimization theory for dynamic processes, and still remains an open problem.

不幸的是，我们再次无法提供答案。很明显，将全局情况的最优准则应用到桶分割的局部情况不会达到预期的效果。一个合理的解决方案将基于动态过程的随机优化理论，并且仍然是一个悬而未决的问题。

Finally, we end up with the question

最后，我们得到了这个问题

How do well-known split strategies perform according to the four query models?

根据四个查询模型，知名的分割策略表现如何？

Remember that up to now all performance evaluations have been based only on the first query model. Because of the practical relevance of the models2,3, and 4 , such an investigation is meaningful and the next section deals with the experiences we gained from our experiments.

请记住，到目前为止，所有的性能评估都仅基于第一个查询模型。由于模型2、3和4具有实际相关性，这样的研究是有意义的，下一节将讨论我们从实验中获得的经验。

## 6 Experimental results

## 6 实验结果

In order to assess the effects of split strategies used in data structures for points, we have chosen the radix split, the median split and the mean split. As underlying data structure we have taken the LSD-tree [5], whose binary tree directory allows for the realization of arbitrary split strategies, and implemented it in Eiffel on a SUN SPARCstation.

为了评估用于点的数据结构中使用的分割策略的效果，我们选择了基数分割、中位数分割和均值分割。作为底层数据结构，我们采用了LSD树[5]，其二叉树目录允许实现任意分割策略，并在SUN SPARC工作站上用Eiffel语言实现了它。

During each test run 50.000 2-dimensional points from the data space $\lbrack 0,1) \times  \lbrack 0,1)$ have been inserted into the initially empty LSD-tree. In order to achieve statistically significant results (a small confidence interval) the bucket capacity was set to $c = {500}$ objects. Whenever a split has to be performed, the split line is chosen such that it hits the longer bucket side and the hit position is defined by the underlying split strategy. The three split strategies were evaluated for each query model assuming two constants ${c}_{\mathcal{M}} = {0.01}$ and ${c}_{\mathcal{M}} = {0.0001}$ . For models 3 and 4, the performance measures are computed by an approximation procedure. For each bucket split, the number of objects currently being stored and the according performance measures are reported.

在每次测试运行期间，已将来自数据空间 $\lbrack 0,1) \times  \lbrack 0,1)$ 的 50,000 个二维点插入到最初为空的 LSD 树（LSD-tree）中。为了获得具有统计学意义的结果（较小的置信区间），桶容量设置为 $c = {500}$ 个对象。每当需要进行分裂时，选择的分裂线要使其与较长的桶边相交，并且相交位置由底层的分裂策略确定。针对每个查询模型，假设两个常数 ${c}_{\mathcal{M}} = {0.01}$ 和 ${c}_{\mathcal{M}} = {0.0001}$ 对三种分裂策略进行了评估。对于模型 3 和 4，性能指标通过近似过程计算得出。对于每个桶分裂，会报告当前存储的对象数量以及相应的性能指标。

A $\beta$ -distribution randomly generates different object distributions, namely a uniform, a 1-heap and a 2- heap distribution. The relatively extreme population of the 1-heap distribution usually exhibits certain effects very clearly, while the 2-heap distribution is a suitable abstraction of cluster patterns typically occuring in real applications. A representative pattern of each of the heap distributions is depicted in figures 5 and 6 , respectively.

$\beta$ 分布会随机生成不同的对象分布，即均匀分布、单堆分布和双堆分布。单堆分布相对极端的分布情况通常能非常清晰地展现出某些效果，而双堆分布是对实际应用中典型出现的聚类模式的一种合适抽象。图 5 和图 6 分别描绘了每种堆分布的代表性模式。

<!-- Media -->

<img src="https://cdn.noedgeai.com/0195c908-c558-75bd-bf47-88151d336725_6.jpg?x=200&y=165&w=617&h=618&r=0"/>

Figure 5: 1-heap distribution.

图 5：单堆分布。

<!-- Media -->

Let us start the discussion of the experimental results with the main outcome of our extensive simulations. The efficiencies of the data space organizations created by the three split strategies differ only marginally. Differences mainly depend on the point of time the snapshot was performed and never exceed more than ten percent of the absolute values. Hence, we restrict our further discussion to radix splits.

让我们从广泛模拟的主要结果开始讨论实验结果。三种分裂策略创建的数据空间组织效率仅有微小差异。差异主要取决于进行快照的时间点，且绝对不会超过绝对值的百分之十。因此，我们将进一步的讨论限制在基数分裂（radix splits）上。

Figure 7, respecively 8, depicts the different performance measures for the 1-heap, resp. 2-heap, distribution with respect to each model and for ${c}_{\mathcal{M}} = {0.01}$ . It turns out that the different model assumptions lead to rather different evaluations of the same data space partition. This effect is mainly observed for distributions with a zero population in wide parts of the data space like e.g. the 1-heap distribution. Note, however, that for a direct comparison the absolute values must be related to the answer size.

图 7 和图 8 分别描绘了单堆分布和双堆分布在每个模型下以及 ${c}_{\mathcal{M}} = {0.01}$ 时的不同性能指标。结果表明，不同的模型假设会导致对同一数据空间划分的评估有相当大的差异。这种效果主要在数据空间大部分区域人口为零的分布中观察到，例如单堆分布。然而，需要注意的是，为了进行直接比较，绝对值必须与答案大小相关联。

A second bunch of simulations deals with the situation where the insertion sequence of objects is somewhat "presorted". Such a presorting often occurs in real applications. For example, whenever we have used real geographic data in what application so ever, the data file was "sorted" according to counties, municipalities or districts, while each data pile itself was almost random. In order to cover this situation by our experiments, we take the 2-heap distribution and completely insert the one heap first and then the other heap, both in random order.

第二组模拟处理的是对象插入顺序有一定“预排序”的情况。这种预排序在实际应用中经常出现。例如，无论在何种应用中使用真实地理数据时，数据文件都是按照县、市或区“排序”的，而每个数据堆本身几乎是随机的。为了在实验中涵盖这种情况，我们采用双堆分布，先完全插入一个堆，然后再插入另一个堆，两者均按随机顺序插入。

<!-- Media -->

<img src="https://cdn.noedgeai.com/0195c908-c558-75bd-bf47-88151d336725_6.jpg?x=975&y=170&w=621&h=613&r=0"/>

Figure 6: 2-heap distribution.

图 6：双堆分布。

<!-- Media -->

Again, our experiments do not exhibit significant differences for the different split strategies. This result is somewhat unexpected because the radix split is well known for its robustness while especially the median split is known to be order sensitive. Even in the situation when the first heap has been inserted and the procedure switches to the second heap, for none of the three split strategies a significant deterioration can be observed. For an entire discussion, however, it should be mentioned, that in case of the median split the directory tends to a certain degeneration.

同样，我们的实验对于不同的分裂策略没有显示出显著差异。这个结果有点出乎意料，因为基数分裂以其鲁棒性而闻名，而尤其是中位数分裂（median split）以对顺序敏感而著称。即使在插入第一个堆后程序切换到插入第二个堆的情况下，也没有观察到三种分裂策略中的任何一种有显著的性能恶化。然而，为了进行全面讨论，应该提到，在中位数分裂的情况下，目录会有一定程度的退化趋势。

Although all split strategies create data space organizations of more or less the same efficiency, our personal choice is the radix split. Besides the robustness of the directory against insertion ordering, the entries in the directory, i.e. the split position, can be encoded with short bitstrings thus keeping the directory small.

尽管所有的分割策略创建的数据空间组织效率或多或少都相同，但我们个人选择基数分割（radix split）。除了目录对插入顺序具有鲁棒性之外，目录中的条目，即分割位置，可以用短位串进行编码，从而使目录保持较小规模。

Another outcome of our experiments not mentioned so far is the effect of using minimal bucket regions. These regions are not bounded by split lines or data space boundaries but are just the bounding boxes of the objects actually stored in the corresponding buckets. It turns out that for small window values ${c}_{\mathcal{M}}$ ,minimal bucket regions can improve the performance up to 50 percent.

到目前为止，我们的实验还有一个尚未提及的结果，即使用最小桶区域的效果。这些区域不受分割线或数据空间边界的限制，而仅仅是实际存储在相应桶中的对象的边界框。结果表明，对于较小的窗口值 ${c}_{\mathcal{M}}$，最小桶区域可以将性能提高多达 50%。

## 7 Open problems

## 7 开放性问题

The following open problems can be added to the list presented in section 5 .

以下开放性问题可以添加到第 5 节列出的列表中。

It seems to be natural to extend the search for efficient split strategies to data structures for non-point geometric objects. These data structures generate bucket regions which may overlap and do not necessarily cover the entire data space. For example, it should be worthwile to use the knowledge gained from our analytical investigations for an improvement of the split strategies of the R-tree which are not well understood yet.

将对高效分割策略的搜索扩展到用于非点几何对象的数据结构似乎是很自然的。这些数据结构生成的桶区域可能会重叠，并且不一定覆盖整个数据空间。例如，利用我们的分析研究所得的知识来改进目前尚未被充分理解的 R 树（R-tree）的分割策略可能是值得的。

<!-- Media -->

<!-- figureText: expected number of bucket accesses madel l medel madel3 madel4 25.0 30.0 35.0 40.0 45.0 50.0 number of inserted objects $\times  {10}^{3}$ 0.0 5.0 10.0 15.0 20.0 -->

<img src="https://cdn.noedgeai.com/0195c908-c558-75bd-bf47-88151d336725_7.jpg?x=153&y=214&w=677&h=673&r=0"/>

Figure 7: The four performance measures for 1-heap distribution,radix splits and ${c}_{\mathcal{M}} = {0.01}$ .

图 7：1 堆分布（1-heap distribution）、基数分割（radix splits）和 ${c}_{\mathcal{M}} = {0.01}$ 的四项性能指标。

<!-- Media -->

Although the time penalty incured by external directory accesses is small compared to data bucket accesses, it would be desirable (at least from a theoretical viewpoint) to extend the performance measures to cover external directory accesses as well. Usually, with each directory page a directory page region is associated which is the bounding box of all data bucket regions pointed at from the directory page (see e.g. $\left\lbrack  {4,5}\right\rbrack$ ). Since directory page regions again form a data space organization, such an integrated analysis of range query performance seems to be feasible.

尽管与数据桶访问相比，外部目录访问所产生的时间开销较小，但（至少从理论角度来看）将性能指标扩展到涵盖外部目录访问也是可取的。通常，每个目录页都关联一个目录页区域，该区域是从该目录页指向的所有数据桶区域的边界框（例如，参见 $\left\lbrack  {4,5}\right\rbrack$）。由于目录页区域再次形成了一种数据空间组织，因此对范围查询性能进行这样的综合分析似乎是可行的。

Finally, the development of analogous performance measures for other query types, like e.g. nearest neighbor queries or queries partly focussing on the volume (area) of the objects, would improve the understanding of spatial data structures even more.

最后，为其他查询类型（如最近邻查询或部分关注对象体积（面积）的查询）开发类似的性能指标，将进一步增进对空间数据结构的理解。

## References

## 参考文献

[1] N. Beckmann, H.-P. Kriegel, R. Schneider, and B. Seeger. The ${\mathrm{R}}^{ * }$ -tree: an efficient and robust access method for points and rectangles. In Proc. ACM SIGMOD Int. Conf. on Management of Data, Atlantic City, 1990.

<!-- Media -->

<!-- figureText: UN madel 1 made(2 model3 madel4 25.0 30.0 35.0 40.0 45.0 number of inserted objects $\times  {10}^{2}$ 品 spected number of bucket accesses 千港元 9 63 12 又 9 0.0 5.0 10.0 15.0 20.0 -->

<img src="https://cdn.noedgeai.com/0195c908-c558-75bd-bf47-88151d336725_7.jpg?x=920&y=216&w=676&h=671&r=0"/>

Figure 8: The four performance measures for 2-heap distribution,radix splits and ${c}_{\mathcal{M}} = {0.01}$ .

图 8：2 堆分布（2-heap distribution）、基数分割（radix splits）和 ${c}_{\mathcal{M}} = {0.01}$ 的四项性能指标。

<!-- Media -->

[2] M.W. Freeston. The BANG file: a new kind of grid file. In Proc. ACM SIGMOD Int. Conf. on the Management of Data, pages 260-169, San Francisco, 1987.

[3] O. Günther. Efficient structures for geometric data management, volume 337 of Lecture Notes in Computer Science. Springer, Berlin, 1988.

[4] A. Guttman. R-trees: a dynamic index structure for spatial searching. In Proc. ACM SIGMOD Int. Conf. on Management of Data, pages 47-57, Boston, 1984.

[5] A. Henrich, H.-W. Six, and P. Widmayer. The LSD-tree: spatial access to multidimensional point- and non-point objects. In 15th Int. Conf. on VLDB, pages 45-53, Amsterdam, 1989.

[6] A. Hutflesz, H.-W. Six, and P. Widmayer. The R-file: an efficient access structure for proximity queries. In Proc. 6th Int. Conf. on Data Engineering, pages 372-379, Los Angeles, 1990.

[7] J. Nievergelt, H. Hinterberger, and K.C. Sevcik. The grid file: an adaptable, symmetric multikey file structure. ACM Transactions on Database Systems, 9(1):38-71, 1984.

[8] B. Seeger and H.-P. Kriegel. The buddy-tree: an efficient and robust access method for spatial data base systems. In 16th Int. Conf. on VLDB, pages 590-601, Brisbane, 1990.