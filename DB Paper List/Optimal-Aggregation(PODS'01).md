# Optimal Aggregation Algorithms for Middleware

# 中间件的最优聚合算法

[Extended Abstract] ${}^{ * }$

[扩展摘要] ${}^{ * }$

Ronald Fagin

罗纳德·法金

IBM Almaden Research Center

IBM阿尔马登研究中心

650 Harry Road

哈里路650号

San Jose, CA 95120

加利福尼亚州圣何塞市，邮编95120

fagin@almaden.ibm.com

Amnon Lotem

阿姆农·洛特姆

University of Maryland-College Park Dept. of Computer Science College Park, Maryland 20742

马里兰大学帕克分校计算机科学系，马里兰州帕克市，邮编20742

lotem@cs.umd.edu

Moni Naor ${}^{ \dagger  }$

莫尼·纳奥尔 ${}^{ \dagger  }$

Weizmann Institute of Science

魏茨曼科学研究所

Dept. of Computer Science

计算机科学系

and Applied Mathematics

与应用数学系

Rehovot 76100, Israel

以色列雷霍沃特市，邮编76100

naor@wisdom.weizmann.ac.il

## ABSTRACT

## 摘要

Assume that each object in a database has $m$ grades,or scores,one for each of $m$ attributes. For example,an object can have a color grade, that tells how red it is, and a shape grade, that tells how round it is. For each attribute, there is a sorted list, which lists each object and its grade under that attribute, sorted by grade (highest grade first). There is some monotone aggregation function, or combining rule, such as min or average, that combines the individual grades to obtain an overall grade.

假设数据库中的每个对象都有 $m$ 个等级或分数，每个属性对应一个等级或分数。例如，一个对象可以有一个颜色等级，用于表示它的红色程度，还有一个形状等级，用于表示它的圆形程度。对于每个属性，都有一个排序列表，该列表列出了每个对象及其在该属性下的等级，并按等级排序（等级最高的排在最前面）。存在某种单调聚合函数或组合规则，如最小值或平均值，用于将各个等级组合起来以获得一个总体等级。

To determine objects that have the best overall grades, the naive algorithm must access every object in the database, to find its grade under each attribute. Fagin has given an algorithm ("Fagin's Algorithm", or FA) that is much more efficient. For some distributions on grades, and for some monotone aggregation functions, FA is optimal in a high-probability sense.

为了确定总体等级最高的对象，朴素算法必须访问数据库中的每个对象，以找到它在每个属性下的等级。法金提出了一种算法（“法金算法”，简称FA），该算法的效率要高得多。对于某些等级分布和某些单调聚合函数，FA在高概率意义上是最优的。

We analyze an elegant and remarkably simple algorithm ( "the threshold algorithm", or TA) that is optimal in a much stronger sense than FA. We show that TA is essentially optimal, not just for some monotone aggregation functions, but for all of them, and not just in a high-probability sense, but over every database. Unlike FA, which requires large buffers (whose size may grow unboundedly as the database size grows), TA requires only a small, constant-size buffer.

我们分析了一种优雅且极其简单的算法（“阈值算法”，简称TA），它在比FA更强的意义上是最优的。我们证明，TA本质上是最优的，不仅适用于某些单调聚合函数，而且适用于所有单调聚合函数；不仅在高概率意义上是最优的，而且在每个数据库上都是最优的。与FA不同，FA需要大缓冲区（其大小可能会随着数据库大小的增长而无限制地增加），而TA只需要一个小的、固定大小的缓冲区。

We distinguish two types of access: sorted access (where the middleware system obtains the grade of an object in some sorted list by proceeding through the list sequentially from the top), and random access (where the middleware system requests the grade of object in a list, and obtains it in one step). We consider the scenarios where random access is either impossible, or expensive relative to sorted access, and provide algorithms that are essentially optimal for these cases as well.

我们区分了两种访问类型：排序访问（中间件系统通过从顶部开始顺序遍历某个排序列表来获取列表中对象的等级）和随机访问（中间件系统请求列表中对象的等级，并一步获取该等级）。我们考虑了随机访问要么不可能，要么相对于排序访问成本较高的场景，并为这些情况提供了本质上最优的算法。

## Categories and Subject Descriptors

## 类别与主题描述

H.2.4 [Database Management]: Systems-multimedia systems, query processing; E.5 [Files]: Sorting/searching; F.2 [Theory of Computation]: Analysis of Algorithms and Problem Complexity; H.3.3 [Information Storage and Retrieval]: Information Search and Retrieval-search process

H.2.4 [数据库管理]：系统 - 多媒体系统、查询处理；E.5 [文件]：排序/搜索；F.2 [计算理论]：算法分析与问题复杂度；H.3.3 [信息存储与检索]：信息搜索与检索 - 搜索过程

## General Terms

## 通用术语

Algorithms, Performance, Theory

算法、性能、理论

## Keywords

## 关键词

middleware, instance optimality, competitive analysis

中间件、实例最优性、竞争分析

## 1. INTRODUCTION

## 1. 引言

Early database systems were required to store only small character strings, such as the entries in a tuple in a traditional relational database. Thus, the data was quite homogeneous. Today, we wish for our database systems to be able to deal not only with character strings (both small and large), but also with a heterogeneous variety of multimedia data (such as images, video, and audio). Furthermore, the data that we wish to access and combine may reside in a variety of data repositories, and we may want our database system to serve as middleware that can access such data.

早期的数据库系统只需要存储小字符串，例如传统关系数据库中元组的条目。因此，数据相当同质。如今，我们希望我们的数据库系统不仅能够处理字符串（包括小字符串和大字符串），还能处理各种异构的多媒体数据（如图像、视频和音频）。此外，我们希望访问和组合的数据可能存在于各种数据存储库中，我们可能希望我们的数据库系统作为中间件来访问这些数据。

One fundamental difference between small character strings and multimedia data is that multimedia data may have attributes that are inherently fuzzy. For example, we do not say that a given image is simply either "red" or "not red". Instead, there is a degree of redness, which ranges between 0 (not at all red) and 1 (totally red).

小字符串和多媒体数据之间的一个根本区别是，多媒体数据可能具有本质上模糊的属性。例如，我们不会简单地说给定的图像是“红色”或“非红色”。相反，存在一个红色程度，其范围在0（完全不是红色）到1（完全是红色）之间。

One approach [4] to deal with such fuzzy data is to make use of an aggregation function $t$ . If ${x}_{1},\ldots ,{x}_{m}$ (each in the interval $\left\lbrack  {0,1}\right\rbrack$ ) are the grades of object $R$ under the $m$ attributes,then $t\left( {{x}_{1},\ldots ,{x}_{m}}\right)$ is the overall grade of object $R$ . As we shall discuss, such aggregation functions are useful in other contexts as well. There is a large literature on choices for the aggregation function (see Zimmermann's textbook [15] and the discussion in [4]).

处理此类模糊数据的一种方法[4]是使用聚合函数$t$。如果${x}_{1},\ldots ,{x}_{m}$（每个都在区间$\left\lbrack  {0,1}\right\rbrack$内）是对象$R$在$m$个属性下的等级，那么$t\left( {{x}_{1},\ldots ,{x}_{m}}\right)$就是对象$R$的总体等级。正如我们将讨论的，这种聚合函数在其他场景中也很有用。关于聚合函数的选择有大量文献（见齐默尔曼的教科书[15]和[4]中的讨论）。

---

<!-- Footnote -->

*A full version of this paper is available on http://www.almaden.ibm.com/cs/people/fagin/

* 本文的完整版本可在http://www.almaden.ibm.com/cs/people/fagin/上获取

${}^{ \dagger  }$ The work of this author was performed while a Visiting Scientist at the IBM Almaden Research Center.

${}^{ \dagger  }$ 本文作者的工作是在IBM阿尔马登研究中心担任访问科学家期间完成的。

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. To copy otherwise, to republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee.

允许个人或课堂使用免费制作本作品全部或部分的数字或硬拷贝，前提是这些拷贝不是为了盈利或商业利益而制作或分发，并且拷贝上带有此声明和第一页的完整引用。否则，如需复制、重新发布、在服务器上发布或分发给列表，需要事先获得特定许可和/或支付费用。

PODS '01 Santa Barbara, California USA

2001年数据库系统原理研讨会（PODS '01），美国加利福尼亚州圣巴巴拉

Copyright 2001 ACM 0-89791-88-6/97/05 ...\$5.00.

版权所有2001 美国计算机协会 0 - 89791 - 88 - 6/97/05 ... 5.00美元。

<!-- Footnote -->

---

One popular choice for the aggregation function is min. In fact, under the standard rules of fuzzy logic [14], if object $R$ has grade ${x}_{1}$ under attribute ${A}_{1}$ and ${x}_{2}$ under attribute ${A}_{2}$ ,then the grade under the fuzzy conjunction ${A}_{1} \land  {A}_{2}$ is $\min \left( {{x}_{1},{x}_{2}}\right)$ . Another popular aggregation function is the average (or the sum, in contexts where we do not care if the resulting overall grade no longer lies in the interval $\left\lbrack  {0,1}\right\rbrack$ ).

聚合函数的一个常用选择是最小值函数（min）。事实上，在模糊逻辑的标准规则下[14]，如果对象$R$在属性${A}_{1}$下的隶属度为${x}_{1}$，在属性${A}_{2}$下的隶属度为${x}_{2}$，那么在模糊合取${A}_{1} \land  {A}_{2}$下的隶属度为$\min \left( {{x}_{1},{x}_{2}}\right)$。另一个常用的聚合函数是平均值函数（或者在我们不关心最终的总体隶属度是否仍在区间$\left\lbrack  {0,1}\right\rbrack$内的情况下使用求和函数）。

We say that an aggregation function $t$ is monotone if $t\left( {{x}_{1},\ldots ,{x}_{m}}\right)  \leq  t\left( {{x}_{1}^{\prime },\ldots ,{x}_{m}^{\prime }}\right)$ whenever ${x}_{i} \leq  {x}_{i}^{\prime }$ for every $i$ . Certainly monotonicity is a reasonable property to demand of an aggregation function: if for every attribute, the grade of object ${R}^{\prime }$ is at least as high as that of object $R$ ,then we would expect the overall grade of ${R}^{\prime }$ to be at least as high as that of $R$ .

我们称一个聚合函数$t$是单调的，如果只要对于每个$i$都有${x}_{i} \leq  {x}_{i}^{\prime }$成立，就有$t\left( {{x}_{1},\ldots ,{x}_{m}}\right)  \leq  t\left( {{x}_{1}^{\prime },\ldots ,{x}_{m}^{\prime }}\right)$成立。当然，单调性是对聚合函数的一个合理要求：如果对于每个属性，对象${R}^{\prime }$的隶属度至少和对象$R$的隶属度一样高，那么我们期望${R}^{\prime }$的总体隶属度至少和$R$的总体隶属度一样高。

The notion of a query is different in a multimedia database system than in a traditional database system. Given a query in a traditional database system (such as a relational database system),there is an unordered set of answers. ${}^{1}$ By contrast, in a multimedia database system, the answer to a query can be thought of as a sorted list, with the answers sorted by grade. As in [4], we shall identify a query with a choice of the aggregation function $t$ . The user is typically interested in finding the top $k$ answers,where $k$ is a given parameter (such as $k = 1,k = {10}$ ,or $k = {100}$ ). This means that we want to obtain $k$ objects (which we may refer to as the "top $k$ objects") with the highest grades on this query, along with their grades (ties are broken arbitrarily). For convenience,throughout this paper we will think of $k$ as a constant value, and we will consider algorithms for obtaining the top $k$ answers.

多媒体数据库系统中查询的概念与传统数据库系统不同。在传统数据库系统（如关系数据库系统）中给定一个查询，会得到一个无序的答案集合${}^{1}$。相比之下，在多媒体数据库系统中，查询的答案可以看作是一个排序好的列表，答案按照隶属度排序。正如文献[4]中所述，我们将查询与聚合函数$t$的选择关联起来。用户通常对找出前$k$个答案感兴趣，其中$k$是一个给定的参数（如$k = 1,k = {10}$或$k = {100}$）。这意味着我们希望获得在该查询中隶属度最高的$k$个对象（我们可以称其为“前$k$个对象”）及其隶属度（平局情况任意处理）。为方便起见，在本文中我们将$k$视为一个常量值，并考虑用于获取前$k$个答案的算法。

Other applications: There are other applications besides multimedia databases where we make use of an aggregation function to combine grades, and where we want to find the top $k$ answers. One important example is information retrieval [11],where the objects $R$ of interest are documents,the $m$ attributes are search terms ${s}_{1},\ldots ,{s}_{m}$ ,and the grade ${x}_{i}$ measures the relevance of document $R$ for search term ${s}_{i}$ ,for $1 \leq  i \leq  m$ . It is common to take the aggregation function $t$ to be the sum. That is,the total relevance score of document $R$ when the query consists of the search terms ${s}_{1},\ldots ,{s}_{m}$ is taken to be $t\left( {{x}_{1},\ldots ,{x}_{m}}\right)  = {x}_{1} + \cdots  + {x}_{m}$ .

其他应用：除了多媒体数据库之外，还有其他应用场景需要使用聚合函数来组合隶属度，并且我们希望找出前$k$个答案。一个重要的例子是信息检索[11]，其中感兴趣的对象$R$是文档，$m$个属性是搜索词${s}_{1},\ldots ,{s}_{m}$，隶属度${x}_{i}$衡量文档$R$与搜索词${s}_{i}$的相关性，其中$1 \leq  i \leq  m$。通常将聚合函数$t$取为求和函数。也就是说，当查询由搜索词${s}_{1},\ldots ,{s}_{m}$组成时，文档$R$的总相关度得分取为$t\left( {{x}_{1},\ldots ,{x}_{m}}\right)  = {x}_{1} + \cdots  + {x}_{m}$。

Another application arises in a paper by Aksoy and Franklin [1] on scheduling large-scale on-demand data broadcast. In this case each object is a page, and there are two fields. The first field represents the amount of time waited by the earliest user requesting a page, and the second field represents the number of users requesting a page. They make use of the product function $t$ with $t\left( {{x}_{1},{x}_{2}}\right)  = {x}_{1}{x}_{2}$ ,and they wish to broadcast next the page with the top score.

另一个应用出现在阿克索伊（Aksoy）和富兰克林（Franklin）的一篇关于大规模按需数据广播调度的论文[1]中。在这种情况下，每个对象是一个页面，并且有两个字段。第一个字段表示最早请求该页面的用户等待的时间，第二个字段表示请求该页面的用户数量。他们使用乘积函数$t$，其中$t\left( {{x}_{1},{x}_{2}}\right)  = {x}_{1}{x}_{2}$，并且他们希望接下来广播得分最高的页面。

The model: We assume that each database consists of a finite set of objects. We shall typically take $N$ to represent the number of objects. Associated with each object $R$ are $m$ fields ${x}_{1},\ldots ,{x}_{m}$ ,where ${x}_{i} \in  \left\lbrack  {0,1}\right\rbrack$ for each $i$ . We may refer to ${x}_{i}$ as the $i$ th field of $R$ . The database is thought of as consisting of $m$ sorted lists ${L}_{1},\ldots ,{L}_{m}$ ,each of length $N$ (there is one entry in each list for each of the $N$ objects). We may refer to ${L}_{i}$ as list $i$ . Each entry of ${L}_{i}$ is of the form $\left( {R,{x}_{i}}\right)$ ,where ${x}_{i}$ is the $i$ th field of $R$ . Each list ${L}_{i}$ is sorted in descending order by the ${x}_{i}$ value. We take this simple view of a database, since this view is all that is relevant, as far as our algorithms are concerned. We are completely ignoring computational issues. For example, in practice it might well be expensive to compute the field values, but we ignore this issue here, and take the field values as being given.

模型：我们假设每个数据库由一组有限的对象组成。我们通常用$N$表示对象的数量。与每个对象$R$相关联的是$m$个字段${x}_{1},\ldots ,{x}_{m}$，其中对于每个$i$都有${x}_{i} \in  \left\lbrack  {0,1}\right\rbrack$。我们可以将${x}_{i}$称为$R$的第$i$个字段。数据库被认为由$m$个排序列表${L}_{1},\ldots ,{L}_{m}$组成，每个列表的长度为$N$（每个列表中每个$N$对象都有一个条目）。我们可以将${L}_{i}$称为列表$i$。${L}_{i}$的每个条目形式为$\left( {R,{x}_{i}}\right)$，其中${x}_{i}$是$R$的第$i$个字段。每个列表${L}_{i}$按${x}_{i}$的值降序排列。我们采用这种对数据库的简单看法，因为就我们的算法而言，这种看法是唯一相关的。我们完全忽略计算问题。例如，在实践中计算字段值可能成本很高，但我们在这里忽略这个问题，并假设字段值是已知的。

We consider two modes of access to data. The first mode of access is sorted (or sequential) access. Here the middle-ware system obtains the grade of an object in one of the sorted lists by proceeding through the list sequentially from the top. Thus,if object $R$ has the $\ell$ th highest grade in the $i$ th list,then $\ell$ sorted accesses to the $i$ th list are required to see this grade under sorted access. The second mode of access is random access. Here, the middleware system requests the grade of object $R$ in the $i$ th list,and obtains it in one random access. If there are $s$ sorted accesses and $r$ random accesses, then the middleware cost is taken to be $s{c}_{S} + r{c}_{R}$ ,for some positive constants ${c}_{S}$ and ${c}_{R}$ .

我们考虑两种数据访问模式。第一种访问模式是排序（或顺序）访问。在这里，中间件系统通过从排序列表的顶部开始顺序遍历列表来获取列表中某个对象的等级。因此，如果对象$R$在第$i$个列表中具有第$\ell$高的等级，那么在排序访问模式下，需要对第$i$个列表进行$\ell$次排序访问才能看到这个等级。第二种访问模式是随机访问。在这里，中间件系统请求第$i$个列表中对象$R$的等级，并通过一次随机访问获取该等级。如果有$s$次排序访问和$r$次随机访问，那么中间件成本为$s{c}_{S} + r{c}_{R}$，其中${c}_{S}$和${c}_{R}$是一些正常数。

Algorithms: There is an obvious naive algorithm for obtaining the top $k$ answers. It looks at every entry in each of the $m$ sorted lists,computes (using $t$ ) the overall grade of every object,and returns the top $k$ answers. The naive algorithm has linear middleware cost (linear in the database size), and thus is not efficient for a large database.

算法：有一种明显的朴素算法可用于获取前$k$个答案。它查看$m$个排序列表中的每个条目，（使用$t$）计算每个对象的总体等级，并返回前$k$个答案。朴素算法的中间件成本是线性的（与数据库大小呈线性关系），因此对于大型数据库而言效率不高。

Fagin [4] introduced an algorithm ("Fagin's Algorithm", or FA), which often does much better than the naive algorithm. In the case where the orderings in the sorted lists are probabilistically independent,FA finds the top $k$ answers,over a database with $N$ objects,with middleware cost $O\left( {{N}^{\left( {m - 1}\right) /m}{k}^{1/m}}\right)$ ,with arbitrarily high probability. ${}^{2}$ Fagin also proved that under this independence assumption, along with an assumption on the aggregation function, every correct algorithm must, with high probability, incur a similar middleware cost.

法金（Fagin）[4]引入了一种算法（“法金算法”，简称FA），该算法通常比朴素算法表现好得多。在排序列表中的排序在概率上相互独立的情况下，FA可以在具有$N$个对象的数据库中找到前$k$个答案，其中间件成本为$O\left( {{N}^{\left( {m - 1}\right) /m}{k}^{1/m}}\right)$，且具有任意高的概率。${}^{2}$法金还证明了在这种独立性假设以及对聚合函数的假设下，每个正确的算法很可能都会产生类似的中间件成本。

We shall present the "threshold algorithm", or TA. This algorithm has been defined and studied by (at least) three groups, including Nepal and Ramakrishna [9] (who were the first to publish), Güntzer, Balke, and Kiessling [5], and ourselves. ${}^{3}$ For more information and comparison,see Section 6 on related work.

我们将介绍“阈值算法”，简称TA。该算法至少由三个研究小组定义和研究过，其中包括尼泊尔（Nepal）和拉马克里希纳（Ramakrishna）[9]（他们是第一个发表相关研究的）、京策尔（Güntzer）、巴尔克（Balke）和基斯林（Kiessling）[5]，以及我们自己。${}^{3}$有关更多信息和比较，请参阅第6节相关工作。

We shall show that TA is optimal in a much stronger sense than FA. We now define this notion of optimality, which we consider to be interesting in its own right.

我们将证明TA在比FA更强的意义上是最优的。我们现在定义这种最优性的概念，我们认为这个概念本身就很有趣。

Instance optimality: Let $\mathbf{A}$ be a class of algorithms, and let $\mathbf{D}$ be a class of legal inputs to the algorithms. We assume that we are considering a particular nonnegative cost measure $\operatorname{cost}\left( {\mathcal{A},\mathcal{D}}\right)$ of running algorithm $\mathcal{A}$ over input $\mathcal{D}$ . This cost could be the running time of algorithm $\mathcal{A}$ on input $\mathcal{D}$ ,or in this paper,the middleware cost incurred by running algorithm $\mathcal{A}$ over database $\mathcal{D}$ . We shall mention examples later where $\operatorname{cost}\left( {\mathcal{A},\mathcal{D}}\right)$ has an interpretation other than being the amount of a resource consumed by running the algorithm $\mathcal{A}$ on input $\mathcal{D}$ .

实例最优性：设$\mathbf{A}$为一类算法，$\mathbf{D}$为这些算法的合法输入类。我们假设正在考虑在输入$\mathcal{D}$上运行算法$\mathcal{A}$的特定非负成本度量$\operatorname{cost}\left( {\mathcal{A},\mathcal{D}}\right)$。此成本可以是算法$\mathcal{A}$在输入$\mathcal{D}$上的运行时间，或者在本文中，是在数据库$\mathcal{D}$上运行算法$\mathcal{A}$所产生的中间件成本。稍后我们将提及$\operatorname{cost}\left( {\mathcal{A},\mathcal{D}}\right)$并非表示在输入$\mathcal{D}$上运行算法$\mathcal{A}$所消耗资源量的其他解释示例。

We say that an algorithm $\mathcal{B} \in  \mathbf{A}$ is instance optimal over $\mathbf{A}$ and $\mathbf{D}$ if $\mathcal{B} \in  \mathbf{A}$ and if for every $\mathcal{A} \in  \mathbf{A}$ and every $\mathcal{D} \in  \mathbf{D}$ we have

我们称算法$\mathcal{B} \in  \mathbf{A}$在$\mathbf{A}$和$\mathbf{D}$上是实例最优的，如果$\mathcal{B} \in  \mathbf{A}$且对于每个$\mathcal{A} \in  \mathbf{A}$和每个$\mathcal{D} \in  \mathbf{D}$，我们有

$$
\operatorname{cost}\left( {\mathcal{B},\mathcal{D}}\right)  = O\left( {\operatorname{cost}\left( {\mathcal{A},\mathcal{D}}\right) }\right) . \tag{1}
$$

---

<!-- Footnote -->

${}^{2}$ We shall not discuss the probability model here,including the notion of "independence", since it is off track. For details, see [4].

${}^{2}$我们这里不讨论概率模型，包括“独立性”的概念，因为这偏离了主题。详情见[4]。

${}^{3}$ Our second author first defined TA,and did extensive simulations comparing it to FA, as a project in a database course taught by Michael Franklin at the University of Maryland-College Park, in the Fall of 1997.

${}^{3}$我们的第二位作者首次定义了TA，并将其与FA进行了广泛的模拟比较，这是1997年秋季在马里兰大学帕克分校由迈克尔·富兰克林（Michael Franklin）教授的数据库课程中的一个项目。

${}^{1}$ Of course,in a relational database,the result to a query may be sorted in some way for convenience in presentation, such as sorting department members by salary, but logically speaking, the result is still simply a set, with a crisply-defined collection of members.

${}^{1}$当然，在关系数据库中，查询结果可能会为了方便展示而以某种方式排序，例如按工资对部门成员进行排序，但从逻辑上讲，结果仍然只是一个集合，具有明确定义的成员集合。

<!-- Footnote -->

---

Equation (1) means that there are constants $c$ and ${c}^{\prime }$ such that $\operatorname{cost}\left( {\mathcal{B},\mathcal{D}}\right)  \leq  c \cdot  \operatorname{cost}\left( {\mathcal{A},\mathcal{D}}\right)  + {c}^{\prime }$ for every choice of $\mathcal{A}$ and $\mathcal{D}$ . We refer to $c$ as the optimality ratio. It is similar to the competitive ratio in competitive analysis (we shall discuss competitive analysis shortly). We use the word "optimal" to reflect that fact that $\mathcal{B}$ is essentially the best algorithm in $\mathbf{A}$ .

方程(1)意味着存在常数$c$和${c}^{\prime }$，使得对于$\mathcal{A}$和$\mathcal{D}$的每一种选择都有$\operatorname{cost}\left( {\mathcal{B},\mathcal{D}}\right)  \leq  c \cdot  \operatorname{cost}\left( {\mathcal{A},\mathcal{D}}\right)  + {c}^{\prime }$。我们将$c$称为最优比。它类似于竞争分析中的竞争比（我们稍后将讨论竞争分析）。我们使用“最优”一词来反映这样一个事实，即$\mathcal{B}$本质上是$\mathbf{A}$中最好的算法。

Intuitively, instance optimality corresponds to optimality in every instance, as opposed to just the worst case or the average case. There are many algorithms that are optimal in a worst-case sense, but are not instance optimal. An example is binary search: in the worst case, binary search is guaranteed to require no more than $\log N$ probes,for $N$ data items. However, for each instance, a positive answer can be obtained in one probe, and a negative answer in two probes.

直观地说，实例最优性对应于每个实例中的最优性，而不仅仅是最坏情况或平均情况。有许多算法在最坏情况下是最优的，但并非实例最优。一个例子是二分查找：在最坏情况下，对于$N$个数据项，二分查找保证最多需要$\log N$次探查。然而，对于每个实例，一次探查可以得到肯定答案，两次探查可以得到否定答案。

We consider a nondeterministic algorithm correct if on no branch does it make a mistake. We take the middleware cost of a nondeterministic algorithm to be the minimal cost over all branches where it halts with the top $k$ answers. We take the middleware cost of a probabilistic algorithm to be the expected cost (over all probabilistic choices by the algorithm). When we say that a deterministic algorithm $\mathcal{B}$ is instance optimal over $\mathbf{A}$ and $\mathbf{D}$ ,then we are really comparing $\mathcal{B}$ against the best nondeterministic algorithm,even if A contains only deterministic algorithms. This is because for each $\mathcal{D} \in  \mathbf{D}$ ,there is always a deterministic algorithm that makes the same choices on $\mathcal{D}$ as the nondeterministic algorithm. We can view the cost of the best nondeterministic algorithm that produces the top $k$ answers over a given database as the cost of the shortest proof for that database that these are really the top $k$ answers. So instance optimality is quite strong: the cost of an instance optimal algorithm is essentially the cost of the shortest proof. Similarly, we can view $\mathbf{A}$ as if it contains also probabilistic algorithms that never make a mistake. For convenience, in our proofs we shall always assume that $\mathbf{A}$ contains only deterministic algorithms, since the results carry over automatically to nondeterministic algorithms and to probabilistic algorithms that never make a mistake.

如果一个非确定性算法在任何分支上都不会出错，我们就认为该算法是正确的。我们将非确定性算法的中间件成本定义为：在所有以排名前 $k$ 的答案终止的分支中，成本的最小值。我们将概率算法的中间件成本定义为（算法所有概率选择的）期望成本。当我们说确定性算法 $\mathcal{B}$ 在 $\mathbf{A}$ 和 $\mathbf{D}$ 上是实例最优的，实际上是将 $\mathcal{B}$ 与最优的非确定性算法进行比较，即使集合 A 中仅包含确定性算法。这是因为对于每个 $\mathcal{D} \in  \mathbf{D}$，总有一个确定性算法在 $\mathcal{D}$ 上的选择与非确定性算法相同。我们可以将在给定数据库上生成排名前 $k$ 答案的最优非确定性算法的成本，视为该数据库中证明这些答案确实是排名前 $k$ 答案的最短证明的成本。因此，实例最优性的要求相当高：实例最优算法的成本本质上就是最短证明的成本。类似地，我们可以将 $\mathbf{A}$ 视为也包含从不犯错的概率算法。为方便起见，在我们的证明中，我们总是假设 $\mathbf{A}$ 仅包含确定性算法，因为这些结果可自动推广到非确定性算法和从不犯错的概率算法。

FA is optimal in a high-probability sense (actually, in a way that involves both high probabilities and worst cases; see [4]), under certain assumptions. TA is optimal in a much stronger sense: it is instance optimal, for several natural choices of $\mathbf{A}$ and $\mathbf{D}$ . In particular,instance optimality holds when $\mathbf{A}$ is taken to be the class of algorithms that would normally be implemented in practice (since the only algorithms that are excluded are those that make very lucky guesses), and when $\mathbf{D}$ is taken to be the class of all databases. Instance optimality of TA holds in this case for all monotone aggregation functions. By contrast, high-probability optimality of FA holds only under the assumption of "strictness" (we shall define strictness later; intuitively, it means that the aggregation function is representing some notion of conjunction).

在某些假设下，FA 在高概率意义上是最优的（实际上，这种最优性既涉及高概率情况，也涉及最坏情况；参见 [4]）。TA 在更强的意义上是最优的：对于 $\mathbf{A}$ 和 $\mathbf{D}$ 的几种自然选择，它是实例最优的。具体而言，当 $\mathbf{A}$ 被定义为实际中通常会实现的算法类（因为被排除的算法仅为那些做出非常幸运猜测的算法），并且 $\mathbf{D}$ 被定义为所有数据库的类时，实例最优性成立。在这种情况下，对于所有单调聚合函数，TA 的实例最优性都成立。相比之下，FA 的高概率最优性仅在“严格性”假设下成立（我们稍后将定义严格性；直观地说，它意味着聚合函数代表某种合取概念）。

The definition we have given for instance optimality is formally the same definition as is used in competitive analysis $\left\lbrack  {2,{12}}\right\rbrack$ ,except that in competitive analysis we do not assume that $\mathcal{B} \in  \mathbf{A}$ . In competitive analysis,typically (a) $\mathbf{A}$ is taken to be the class of offline algorithms that solve a particular problem,(b) $\operatorname{cost}\left( {\mathcal{A},\mathcal{D}}\right)$ is taken to be a number that represents performance (where bigger numbers correspond to worse performance),and (c) $\mathcal{B}$ is a particular online algorithm. In this case,the online algorithm $\mathcal{B}$ is said to be competitive. The intuition is that a competitive online algorithm may perform poorly in some instances, but only on instances where every offline algorithm would also perform poorly.

我们给出的实例最优性定义在形式上与竞争分析 $\left\lbrack  {2,{12}}\right\rbrack$ 中使用的定义相同，只是在竞争分析中我们不假设 $\mathcal{B} \in  \mathbf{A}$。在竞争分析中，通常 (a) $\mathbf{A}$ 被定义为解决特定问题的离线算法类，(b) $\operatorname{cost}\left( {\mathcal{A},\mathcal{D}}\right)$ 被定义为表示性能的一个数值（数值越大表示性能越差），并且 (c) $\mathcal{B}$ 是一个特定的在线算法。在这种情况下，在线算法 $\mathcal{B}$ 被称为具有竞争性。其直观含义是，具有竞争性的在线算法在某些实例中可能表现不佳，但仅在所有离线算法表现都不佳的实例中才会如此。

Another example where the framework of instance optimality appears, but again without the assumption that $\mathcal{B} \in  \mathbf{A}$ ,is in the context of approximation algorithms [7]. In this case, (a) $\mathbf{A}$ is taken to contain algorithms that solve a particular problem exactly (in cases of interest, these algorithms are not polynomial-time algorithms), (b) $\operatorname{cost}\left( {\mathcal{A},\mathcal{D}}\right)$ is taken to be the resulting answer when algorithm $\mathcal{A}$ is applied to input $\mathcal{D}$ ,and (c) $\mathcal{B}$ is a particular polynomial-time algorithm.

实例最优性框架出现的另一个例子，但同样没有假设 $\mathcal{B} \in  \mathbf{A}$，是在近似算法的背景下 [7]。在这种情况下，(a) $\mathbf{A}$ 被定义为包含能精确解决特定问题的算法（在感兴趣的情况下，这些算法不是多项式时间算法），(b) $\operatorname{cost}\left( {\mathcal{A},\mathcal{D}}\right)$ 被定义为将算法 $\mathcal{A}$ 应用于输入 $\mathcal{D}$ 时得到的答案，并且 (c) $\mathcal{B}$ 是一个特定的多项式时间算法。

Restricting random access: As we shall discuss in Section 2, there are some systems where random access is impossible. To deal with such situations, we show in Section 5.1 how to modify TA to obtain an algorithm NRA ("no random accesses") that does no random accesses. We prove that NRA is instance optimal over all algorithms that do not make random accesses and over all databases.

限制随机访问：正如我们将在第 2 节中讨论的，有些系统无法进行随机访问。为了处理这种情况，我们在第 5.1 节中展示了如何修改 TA 以获得一个不进行随机访问的算法 NRA（“无随机访问”）。我们证明了 NRA 在所有不进行随机访问的算法和所有数据库上是实例最优的。

What about situations where random access is not forbidden, but simply expensive? Wimmers et al. [13] discuss a number of systems issues that can cause random access to be expensive. Although TA is instance optimal, the optimality ratio depends on the ratio ${c}_{R}/{c}_{S}$ of the cost of a single random access to the cost of a single sorted access. We define another algorithm that is a combination of TA and NRA, and call it CA ("combined algorithm"). The definition of the algorithm depends on ${c}_{R}/{c}_{S}$ . The motivation is to obtain an algorithm that is not only instance optimal, but whose optimality ratio is independent of ${c}_{R}/{c}_{S}$ . Our original hope was that CA would be instance optimal (with optimality ratio independent of ${c}_{R}/{c}_{S}$ ) in those scenarios where TA is instance optimal. Not only does this hope fail, but interestingly enough, we prove that there does not exist any deterministic algorithm, or even probabilistic algorithm that does not make a mistake, that is instance optimal (with optimality ratio independent of ${c}_{R}/{c}_{S}$ ) in these scenarios! However, we find a new natural scenario where CA is instance optimal,with optimality ratio independent of ${c}_{R}/{c}_{S}$ .

如果随机访问并非被禁止，只是代价高昂的情况会怎样呢？温默斯（Wimmers）等人 [13] 讨论了一些会导致随机访问代价高昂的系统问题。尽管 TA 算法是实例最优的，但最优比率取决于单次随机访问成本与单次有序访问成本之比 ${c}_{R}/{c}_{S}$。我们定义了另一种结合了 TA 算法和 NRA 算法的算法，并将其称为 CA 算法（“组合算法”）。该算法的定义取决于 ${c}_{R}/{c}_{S}$。其动机是获得一种不仅是实例最优，而且最优比率与 ${c}_{R}/{c}_{S}$ 无关的算法。我们最初希望在 TA 算法是实例最优的那些场景中，CA 算法也能是实例最优的（且最优比率与 ${c}_{R}/{c}_{S}$ 无关）。然而，这个希望不仅落空了，而且有趣的是，我们证明了在这些场景中，不存在任何确定性算法，甚至不存在不会出错的概率算法能够是实例最优的（且最优比率与 ${c}_{R}/{c}_{S}$ 无关）！不过，我们发现了一个新的自然场景，在该场景中 CA 算法是实例最优的，且最优比率与 ${c}_{R}/{c}_{S}$ 无关。

## 2. MODES OF ACCESS TO DATA

## 2. 数据访问模式

Issues of efficient query evaluation in a middleware system are very different from those in a traditional database system. This is because the middleware system receives answers to queries from various subsystems, which can be accessed only in limited ways. What do we assume about the interface between a middleware system and a subsystem? Let us consider ${\mathrm{{QBIC}}}^{4}\left\lbrack  {10}\right\rbrack$ ("Query By Image Content") as a subsystem. QBIC can search for images by various visual characteristics such as color and texture (and an experimental version can search also by shape). In response to a query, such as Color='red', the subsystem will output the graded set consisting of all objects, one by one, along with their grades under the query, in sorted order based on grade, until the middleware system tells the subsystem to halt. Then the middleware system could later tell the subsystem to resume outputting the graded set where it left off. Alternatively, the middleware system could ask the subsystem for, say, the top 10 objects in sorted order, along with their grades, then request the next 10, etc. In both cases, this corresponds to what we have referred to as "sorted access".

中间件系统中高效查询评估的问题与传统数据库系统中的问题有很大不同。这是因为中间件系统从各个子系统获取查询答案，而这些子系统只能以有限的方式进行访问。我们对中间件系统和子系统之间的接口做了哪些假设呢？让我们以 ${\mathrm{{QBIC}}}^{4}\left\lbrack  {10}\right\rbrack$（“基于图像内容的查询”）作为一个子系统来考虑。QBIC 可以通过各种视觉特征（如颜色和纹理，实验版本还可以通过形状）来搜索图像。针对一个查询，例如“颜色 = '红色'”，子系统将按照评分顺序逐个输出由所有对象及其在该查询下的评分组成的分级集合，直到中间件系统告知子系统停止。然后，中间件系统稍后可以告知子系统从停止的位置继续输出分级集合。或者，中间件系统可以要求子系统按排序顺序输出前 10 个对象及其评分，然后再请求接下来的 10 个对象，依此类推。在这两种情况下，这都对应于我们所说的“有序访问”。

---

<!-- Footnote -->

${}^{4}$ QBIC is a trademark of IBM Corporation.

${}^{4}$ QBIC 是国际商业机器公司（IBM Corporation）的商标。

<!-- Footnote -->

---

There is another way that we might expect the middleware system to interact with the subsystem. The middleware system might ask the subsystem for the grade (with respect to a query) of any given object. This corresponds to what we have referred to as "random access". In fact, QBIC allows both sorted and random access.

我们可能期望中间件系统与子系统交互的另一种方式是，中间件系统可能会向子系统询问任何给定对象（相对于某个查询）的评分。这对应于我们所说的“随机访问”。实际上，QBIC 既支持有序访问，也支持随机访问。

There are some situations where the middleware system is not allowed random access to some subsystem. An example might occur when the middleware system is a text retrieval system, and the subsystems are search engines. Thus, there does not seem to be a way to ask a major search engine on the web for its internal score on some document of our choice under a query.

在某些情况下，中间件系统不被允许对某些子系统进行随机访问。例如，当中间件系统是一个文本检索系统，而子系统是搜索引擎时，就可能出现这种情况。因此，似乎没有办法向网络上的主要搜索引擎询问其对我们选择的某个文档在某个查询下的内部评分。

Our measure of cost corresponds intuitively to the cost incurred by the middleware system in processing information passed to it from a subsystem such as QBIC. As before, if there are $s$ sorted accesses and $r$ random accesses,then the middleware cost is taken to be $s{c}_{S} + r{c}_{R}$ ,for some positive constants ${c}_{S}$ and ${c}_{R}$ . The fact that ${c}_{S}$ and ${c}_{R}$ may be different reflects the fact that the cost to a middleware system of a sorted access and of a random access may be different.

我们的成本度量直观上对应于中间件系统处理从诸如 QBIC 这样的子系统传递给它的信息时所产生的成本。和之前一样，如果有 $s$ 次有序访问和 $r$ 次随机访问，那么中间件成本被认为是 $s{c}_{S} + r{c}_{R}$，其中 ${c}_{S}$ 和 ${c}_{R}$ 是一些正常数。${c}_{S}$ 和 ${c}_{R}$ 可能不同这一事实反映了中间件系统进行有序访问和随机访问的成本可能不同。

## 3. FAGIN'S ALGORITHM

## 3. 法金算法

In this section, we discuss FA (Fagin's Algorithm) [4]. This algorithm is implemented in Garlic [3], an experimental IBM middleware system; see [13] for interesting details about the implementation and performance in practice. FA works as follows.

在本节中，我们将讨论 FA 算法（法金算法）[4]。该算法在大蒜（Garlic）[3] 这个实验性的 IBM 中间件系统中实现；有关该算法在实际中的实现和性能的有趣细节，请参阅 [13]。FA 算法的工作方式如下。

1. Do sorted access in parallel to each of the $m$ sorted lists ${L}_{i}$ . (By "in parallel",we mean that we access the top member of each of the lists under sorted access, then we access the second member of each of the lists, and so on. ${)}^{5}$ Wait until there are at least $k$ "matches", that is,wait until there is a set $H$ of at least $k$ objects such that each of these objects has been seen in each of the $m$ lists.

1. 并行地对每个 $m$ 排序列表 ${L}_{i}$ 进行有序访问。（“并行”是指，我们在有序访问下访问每个列表的首个元素，然后访问每个列表的第二个元素，依此类推。 ${)}^{5}$ 等待直到至少有 $k$ 个“匹配项”，即，等待直到存在一个集合 $H$，其中至少有 $k$ 个对象，且这些对象中的每一个都在 $m$ 个列表的每一个中出现过。

2. For each object $R$ that has been seen,do random access to each of the lists ${L}_{i}$ to find the $i$ th field ${x}_{i}$ of $R$ .

2. 对于每个已出现的对象 $R$，对每个列表 ${L}_{i}$ 进行随机访问，以找到 $R$ 的第 $i$ 个字段 ${x}_{i}$。

3. Compute the grade $t\left( R\right)  = t\left( {{x}_{1},\ldots ,{x}_{m}}\right)$ for each object $R$ that has been seen. Let $Y$ be a set containing the $k$ objects that have been seen with the highest grades (ties are broken arbitrarily). The output is then the graded set $\{ \left( {R,t\left( R\right) }\right)  \mid  R \in  Y\}$ . ${}^{7}$

3. 为每个已出现的对象 $R$ 计算得分 $t\left( R\right)  = t\left( {{x}_{1},\ldots ,{x}_{m}}\right)$。设 $Y$ 是一个包含得分最高的 $k$ 个已出现对象的集合（若有平局则任意打破）。输出即为带得分的集合 $\{ \left( {R,t\left( R\right) }\right)  \mid  R \in  Y\}$。 ${}^{7}$

It is fairly easy to show [4] that this algorithm is correct for monotone aggregation functions $t$ (that is,that the algorithm successfully finds the top $k$ answers). If there are $N$ objects in the database, and if the orderings in the sorted lists are probabilistically independent, then the middleware cost of FA is $O\left( {{N}^{\left( {m - 1}\right) /m}{k}^{1/m}}\right)$ ,with arbitrarily high probability [4].

很容易证明 [4]，对于单调聚合函数 $t$，该算法是正确的（即，该算法能成功找到前 $k$ 个答案）。如果数据库中有 $N$ 个对象，并且排序列表中的排序在概率上是相互独立的，那么 FA 的中间件成本为 $O\left( {{N}^{\left( {m - 1}\right) /m}{k}^{1/m}}\right)$，且具有任意高的概率 [4]。

An aggregation function $t$ is strict [4] if $t\left( {{x}_{1},\ldots ,{x}_{m}}\right)  = 1$ holds precisely when ${x}_{i} = 1$ for every $i$ . Thus,an aggregation function is strict if it takes on the maximal value of 1 precisely when each argument takes on this maximal value. We would certainly expect an aggregation function representing the conjunction to be strict. In fact, it is reasonable to think of strictness as being a key characterizing feature of the conjunction.

如果对于每个 $i$，当且仅当 ${x}_{i} = 1$ 时 $t\left( {{x}_{1},\ldots ,{x}_{m}}\right)  = 1$ 成立，则聚合函数 $t$ 是严格的 [4]。因此，如果一个聚合函数仅当每个参数都取到最大值 1 时才取到最大值 1，那么该聚合函数就是严格的。我们当然期望表示合取的聚合函数是严格的。事实上，将严格性视为合取的一个关键特征是合理的。

Fagin shows that his algorithm is optimal (in a high-probability sense) if the aggregation function is strict (so that, intuitively, we are dealing with a notion of conjunction), and if the orderings in the sorted lists are probabilistically independent. In fact, under the assumption that the sorted lists are probabilistically independent, the mid-dleware cost of FA is $\Theta \left( {{N}^{\left( {m - 1}\right) /m}{k}^{1/m}}\right)$ ,with arbitrarily high probability, no matter what the aggregation function is. This is true even for a constant aggregation function; in this case, of course, there is a trivial algorithm that gives us the top $k$ answers (any $k$ objects will do) with $O\left( 1\right)$ mid-dleware cost. So FA is not optimal in any sense for some monotone aggregation functions $t$ . By contrast,as we shall see, the algorithm TA is instance optimal for every monotone aggregation function, under very weak assumptions.

法金（Fagin）表明，如果聚合函数是严格的（直观地说，我们处理的是合取的概念），并且排序列表中的排序在概率上是相互独立的，那么他的算法是最优的（在高概率意义上）。事实上，在排序列表在概率上相互独立的假设下，无论聚合函数是什么，FA 的中间件成本为 $\Theta \left( {{N}^{\left( {m - 1}\right) /m}{k}^{1/m}}\right)$，且具有任意高的概率。即使对于常量聚合函数也是如此；当然，在这种情况下，有一个简单的算法可以以 $O\left( 1\right)$ 的中间件成本为我们提供前 $k$ 个答案（任意 $k$ 个对象都可以）。因此，对于某些单调聚合函数 $t$，FA 在任何意义上都不是最优的。相比之下，正如我们将看到的，在非常弱的假设下，算法 TA 对于每个单调聚合函数都是实例最优的。

Even in the cases where FA is optimal, this optimality holds only in a high-probability sense. This leaves open the possibility that there are some algorithms that have much better middleware cost than FA over certain databases. The algorithm TA, which we now discuss, is such an algorithm.

即使在 FA 是最优的情况下，这种最优性也仅在高概率意义上成立。这就留下了一种可能性，即在某些数据库上，存在一些算法的中间件成本比 FA 低得多。我们现在讨论的算法 TA 就是这样一种算法。

### 4.THE THRESHOLD ALGORITHM

### 4. 阈值算法

We now present the threshold algorithm (TA).

我们现在介绍阈值算法（TA）。

1. Do sorted access in parallel to each of the $m$ sorted lists ${L}_{i}$ . As an object $R$ is seen under sorted access in some list, do random access to the other lists to find the grade ${x}_{i}$ of object $R$ in every list ${L}_{i}$ . Then compute the grade $t\left( R\right)  = t\left( {{x}_{1},\ldots ,{x}_{m}}\right)$ of object $R$ . If this grade is one of the $k$ highest we have seen,then remember object $R$ and its grade $t\left( R\right)$ (ties are broken arbitrarily,so that only $k$ objects and their grades need to be remembered at any time).

1. 并行地对每个 $m$ 已排序列表 ${L}_{i}$ 进行有序访问。当在某个列表的有序访问中看到一个对象 $R$ 时，对其他列表进行随机访问，以找到对象 $R$ 在每个列表 ${L}_{i}$ 中的等级 ${x}_{i}$。然后计算对象 $R$ 的等级 $t\left( R\right)  = t\left( {{x}_{1},\ldots ,{x}_{m}}\right)$。如果这个等级是我们所见过的 $k$ 个最高等级之一，那么记录下对象 $R$ 及其等级 $t\left( R\right)$（任意打破平局，这样在任何时候只需记录 $k$ 个对象及其等级）。

2. For each list ${L}_{i}$ ,let ${\underline{x}}_{i}$ be the grade of the last object seen under sorted access. Define the threshold value $\tau$ to be $t\left( {{\underline{x}}_{1},\ldots ,{\underline{x}}_{m}}\right)$ . As soon as at least $k$ objects have been seen whose grade is at least equal to $\tau$ ,then halt.

2. 对于每个列表 ${L}_{i}$，设 ${\underline{x}}_{i}$ 为在有序访问中最后看到的对象的等级。将阈值 $\tau$ 定义为 $t\left( {{\underline{x}}_{1},\ldots ,{\underline{x}}_{m}}\right)$。一旦至少看到 $k$ 个对象，其等级至少等于 $\tau$，则停止。

3. Let $Y$ be a set containing the $k$ objects that have been seen with the highest grades. The output is then the graded set $\{ \left( {R,t\left( R\right) }\right)  \mid  R \in  Y\}$ .

3. 设 $Y$ 是一个包含已看到的具有最高等级的 $k$ 个对象的集合。那么输出就是分级集合 $\{ \left( {R,t\left( R\right) }\right)  \mid  R \in  Y\}$。

We now show that TA is correct for each monotone aggregation function $t$ .

我们现在证明，对于每个单调聚合函数 $t$，TA 算法是正确的。

THEOREM 4.1. If the aggregation function $t$ is monotone, then TA correctly finds the top $k$ answers. Proof: Let $Y$ be as in Part 3 of TA. We need only show that every member of $Y$ has at least as high a grade as every object $z$ not in $Y$ . By definition of $Y$ ,this is the case for each object $z$ that has been seen in running TA. So assume that $z$ was not seen. Assume that the fields of $z$ are ${x}_{1},\ldots ,{x}_{m}$ . Therefore, ${x}_{i} \leq  {\underline{x}}_{i}$ ,for every $i$ . Hence, $t\left( z\right)  =$ $t\left( {{x}_{1},\ldots ,{x}_{m}}\right)  \leq  t\left( {{\underline{x}}_{1},\ldots ,{\underline{x}}_{m}}\right)  = \tau$ ,where the inequality follows by monotonicity of $t$ . But by definition of $Y$ ,for every $y$ in $Y$ we have $t\left( y\right)  \geq  \tau$ . Therefore,for every $y$ in $Y$ we have $t\left( y\right)  \geq  \tau  \geq  t\left( z\right)$ ,as desired. [

定理 4.1。如果聚合函数 $t$ 是单调的，那么 TA 算法能正确找出前 $k$ 个答案。证明：设 $Y$ 如 TA 算法第 3 部分所定义。我们只需证明 $Y$ 中的每个成员的等级至少和不在 $Y$ 中的每个对象 $z$ 的等级一样高。根据 $Y$ 的定义，对于在运行 TA 算法时看到的每个对象 $z$，情况确实如此。因此假设 $z$ 未被看到。假设 $z$ 的字段为 ${x}_{1},\ldots ,{x}_{m}$。因此，对于每个 $i$，有 ${x}_{i} \leq  {\underline{x}}_{i}$。因此，$t\left( z\right)  =$ $t\left( {{x}_{1},\ldots ,{x}_{m}}\right)  \leq  t\left( {{\underline{x}}_{1},\ldots ,{\underline{x}}_{m}}\right)  = \tau$，其中不等式由 $t$ 的单调性得出。但根据 $Y$ 的定义，对于 $Y$ 中的每个 $y$，我们有 $t\left( y\right)  \geq  \tau$。因此，对于 $Y$ 中的每个 $y$，我们有 $t\left( y\right)  \geq  \tau  \geq  t\left( z\right)$，正如所期望的。

---

<!-- Footnote -->

${}^{5}$ It is not actually important that the lists be accessed "in lockstep". In practice, it may be convenient to allow the sorted lists to be accessed at different rates, in batches, etc. Throughout this paper, whenever we speak of "sorted access in parallel", all of our instance optimality results continue to hold even when sorted access is not in lockstep, as long as the rates of sorted access of the lists are within constant multiples of each other.

${}^{5}$ 实际上，列表是否“同步”访问并不重要。在实践中，允许以不同速率、批量等方式访问已排序列表可能会更方便。在本文中，无论何时我们提到“并行有序访问”，只要列表的有序访问速率相互之间是常数倍关系，即使有序访问不是同步的，我们所有的实例最优性结果仍然成立。

${}^{6}$ We shall often abuse notation and write $t\left( R\right)$ for the grade $t\left( {{x}_{1},\ldots ,{x}_{m}}\right)$ of $R$ .

${}^{6}$ 我们经常会简化符号，用 $t\left( R\right)$ 表示 $R$ 的等级 $t\left( {{x}_{1},\ldots ,{x}_{m}}\right)$。

${}^{7}$ Graded sets are often presented in sorted order,sorted by grade.

${}^{7}$ 分级集合通常按等级排序呈现。

<!-- Footnote -->

---

We now show that the stopping rule for TA always occurs at least as early as the stopping rule for FA (that is, with no more sorted accesses than FA). In FA,if $R$ is an object that has appeared under sorted access in every list, then by monotonicity,the grade of $R$ is at least equal to the threshold value. Therefore,when there are at least $k$ objects, each of which has appeared under sorted access in every list (the stopping rule for FA),there are at least $k$ objects whose grade is at least equal to the threshold value (the stopping rule for TA).

我们现在证明，TA（阈值算法，Threshold Algorithm）的停止规则至少与FA（某种算法，文中未明确全称）的停止规则一样早发生（即，排序访问次数不多于FA）。在FA中，如果$R$是在每个列表的排序访问中都出现过的对象，那么根据单调性，$R$的等级至少等于阈值。因此，当至少有$k$个对象，且每个对象都在每个列表的排序访问中出现过（FA的停止规则）时，至少有$k$个对象的等级至少等于阈值（TA的停止规则）。

This implies that for every database, the sorted access cost for TA is at most that of FA. This does not imply that the middleware cost for TA is always at most that of FA, since TA may do more random accesses than FA. However, since the middleware cost of TA is at most the sorted access cost times a constant (independent of the database size), it does follow that the middleware cost of TA is at most a constant times that of FA. In fact, we shall show that TA is instance optimal, under natural assumptions.

这意味着对于每个数据库，TA的排序访问成本至多与FA的排序访问成本相同。但这并不意味着TA的中间件成本总是至多与FA的中间件成本相同，因为TA可能比FA进行更多的随机访问。然而，由于TA的中间件成本至多是排序访问成本乘以一个常数（与数据库大小无关），因此可以得出TA的中间件成本至多是FA的中间件成本乘以一个常数。事实上，我们将证明，在自然假设下，TA是实例最优的。

The next simple theorem gives a useful property of TA, that further distinguishes TA from FA.

下一个简单的定理给出了TA的一个有用性质，这进一步将TA与FA区分开来。

THEOREM 4.2. TA requires only bounded buffers, whose size is independent of the size of the database.

定理4.2：TA只需要有界缓冲区，其大小与数据库的大小无关。

By contrast, FA requires buffers that grow arbitrarily large as the database grows, since FA must remember every object it has seen in sorted order in every list, in order to check for matching objects in the various lists.

相比之下，FA需要的缓冲区会随着数据库的增长而任意增大，因为FA必须按排序顺序记住它在每个列表中看到的每个对象，以便检查各个列表中的匹配对象。

There is a price to pay for the bounded buffers. Thus, for every time an object is found under sorted access, TA may do $m - 1$ random accesses (where $m$ is the number of lists), to find the grade of the object in the other lists. This is in spite of the fact that this object may have already been seen under sorted or random access in one of the other lists.

使用有界缓冲区是有代价的。因此，每次在排序访问中找到一个对象时，TA可能会进行$m - 1$次随机访问（其中$m$是列表的数量），以找到该对象在其他列表中的等级。尽管这个对象可能已经在其他某个列表的排序或随机访问中被看到过。

### 4.1 Instance Optimality of the Threshold Al- gorithm

### 4.1 阈值算法的实例最优性

In this section, we investigate the instance optimality of TA. We would have liked to be able to simply state that for every monotone aggregation function, TA is instance optimal over all algorithms that correctly find the top $k$ answers, over the class of all databases. However, it turns out that the situation is more delicate than this. We first make a distinction between algorithms that "make wild guesses" (that is, perform random access on elements not previously encountered by sorted access) and those that do not. (Neither FA nor TA make wild guesses, and neither does any "natural" algorithm.) Our first theorem (Theorem 4.3) says that for every monotone aggregation function, TA is instance optimal over all algorithms that correctly find the top $k$ answers and that do not make wild guesses, over the class of all databases. We then show that this distinction (wild guesses vs. no wild guesses) is essential: if algorithms that make wild guesses are allowed in the class $\mathbf{A}$ of algorithms that an instance optimal algorithm must compete against, then no algorithm is instance optimal (Example 4.4 and Theorem 4.5). The heart of this example (and the corresponding theorem) is the fact that there may be multiple objects with the same grade in some list. Indeed, once we restrict our attention to databases where no two objects have the same value in the same list, and make a slight, natural additional restriction on the aggregation function beyond monotonicity, then TA is instance optimal over all algorithms that correctly find the top $k$ answers (Theorem 4.6).

在本节中，我们研究TA的实例最优性。我们原本希望能够简单地声明，对于每个单调聚合函数，在所有数据库类中，TA在所有能正确找到前$k$个答案的算法中是实例最优的。然而，事实证明情况比这更微妙。我们首先区分“进行盲目猜测”的算法（即，对之前在排序访问中未遇到的元素进行随机访问）和不进行盲目猜测的算法。（FA和TA都不进行盲目猜测，任何“自然”的算法也都不进行盲目猜测）。我们的第一个定理（定理4.3）表明，对于每个单调聚合函数，在所有数据库类中，TA在所有能正确找到前$k$个答案且不进行盲目猜测的算法中是实例最优的。然后我们证明这种区分（盲目猜测与不盲目猜测）是至关重要的：如果允许进行盲目猜测的算法出现在实例最优算法必须与之竞争的算法类$\mathbf{A}$中，那么就没有算法是实例最优的（示例4.4和定理4.5）。这个示例（以及相应的定理）的核心在于，在某些列表中可能存在多个具有相同等级的对象。实际上，一旦我们将注意力限制在没有两个对象在同一列表中具有相同值的数据库上，并对聚合函数在单调性之外做一个轻微的、自然的额外限制，那么TA在所有能正确找到前$k$个答案的算法中是实例最优的（定理4.6）。

In Section 4.3 we consider instance optimality in the situation where we relax the problem of finding the top $k$ objects into finding approximately the top $k$ .

在4.3节中，我们考虑将寻找前$k$个对象的问题放宽为近似寻找前$k$个对象的情况下的实例最优性。

We now give our first positive result on instance optimality of TA. We say that an algorithm makes wild guesses if it does random access to find the grade of some object $R$ in some list before the algorithm has seen $R$ under sorted access. That is, an algorithm makes wild guesses if the first grade that it obtains for some object $R$ is under random access. We would not normally implement algorithms that make wild guesses. In fact, there are some contexts where it would not even be possible to make wild guesses (such as a database context where the algorithm could not know the name of an object it has not already seen). However, making a lucky wild guess can help, as we show later (Example 4.4).

我们现在给出关于TA实例最优性的第一个正面结果。我们说一个算法进行盲目猜测，如果它在通过排序访问看到某个对象$R$之前，就通过随机访问来查找该对象在某个列表中的等级。也就是说，如果一个算法首次获得某个对象$R$的等级是通过随机访问，那么该算法就进行了盲目猜测。我们通常不会实现进行盲目猜测的算法。事实上，在某些情况下甚至不可能进行盲目猜测（例如，在数据库环境中，算法可能不知道它尚未看到的对象的名称）。然而，正如我们稍后将展示的（示例4.4），一次幸运的盲目猜测可能会有所帮助。

We now show instance optimality of TA among algorithms that do not make wild guesses. In this theorem, when we take $\mathbf{D}$ to be the class of all databases,we really mean that $\mathbf{D}$ is the class of all databases that involve sorted lists corresponding to the arguments of the aggregation function $t$ . We are taking $k$ (where we are trying to find the top $k$ answers) and the aggregation function $t$ to be fixed. Since we are taking $t$ to be fixed,we are thereby taking the number $m$ of arguments of $t$ (that is,the number of sorted lists) to be fixed. In Section 4.2,we discuss the assumptions that $k$ and $m$ are constant.

我们现在证明，在不进行胡乱猜测的算法中，TA算法具有实例最优性。在这个定理中，当我们将$\mathbf{D}$视为所有数据库的类时，实际上是指$\mathbf{D}$是所有涉及与聚合函数$t$的参数相对应的排序列表的数据库的类。我们将$k$（即我们试图找出前$k$个答案）和聚合函数$t$视为固定的。由于我们将$t$视为固定的，因此我们也将$t$的参数数量$m$（即排序列表的数量）视为固定的。在4.2节中，我们讨论了$k$和$m$为常数的假设。

THEOREM 4.3. Assume that the aggregation function $t$ is monotone. Let $\mathbf{D}$ be the class of all databases. Let $\mathbf{A}$ be the class of all algorithms that correctly find the top $k$ answers for $t$ for every database and that do not make wild guesses. Then ${TA}$ is instance optimal over $\mathbf{A}$ and $\mathbf{D}$ .

定理4.3。假设聚合函数$t$是单调的。设$\mathbf{D}$为所有数据库的类。设$\mathbf{A}$为所有能为每个数据库正确找出$t$的前$k$个答案且不进行胡乱猜测的算法的类。那么${TA}$在$\mathbf{A}$和$\mathbf{D}$上具有实例最优性。

Proof: Assume that $\mathcal{A} \in  \mathbf{A}$ ,and that algorithm $\mathcal{A}$ is run over database $\mathcal{D}$ . Assume that algorithm $\mathcal{A}$ halts at depth $d$ (that is,if ${d}_{i}$ is the number of objects seen under sorted access to list $i$ ,for $1 \leq  i \leq  m$ ,then $\left. {d = \mathop{\max }\limits_{i}{d}_{i}}\right)$ . Assume that $\mathcal{A}$ sees $a$ distinct objects (some possibly multiple times). In particular, $a \geq  d$ . We shall show that TA halts on $\mathcal{D}$ by depth $a + k$ . Hence,TA makes at most $m\left( {a + k}\right)$ accesses, which is ${ma}$ plus an additive constant of ${mk}$ . It follows easily that the optimality ratio of TA is at most ${cm}$ ,where $c = \max \left\{  {{c}_{R}/{c}_{S},{c}_{S}/{c}_{R}}\right\}  .$

证明：假设$\mathcal{A} \in  \mathbf{A}$，并且算法$\mathcal{A}$在数据库$\mathcal{D}$上运行。假设算法$\mathcal{A}$在深度$d$处停止（即，如果${d}_{i}$是在对列表$i$进行排序访问时看到的对象数量，对于$1 \leq  i \leq  m$，则$\left. {d = \mathop{\max }\limits_{i}{d}_{i}}\right)$。假设$\mathcal{A}$看到了$a$个不同的对象（有些对象可能被看到多次）。特别地，$a \geq  d$。我们将证明TA算法在数据库$\mathcal{D}$上在深度$a + k$处停止。因此，TA算法最多进行$m\left( {a + k}\right)$次访问，这是${ma}$加上一个常数${mk}$。由此很容易得出，TA算法的最优比至多为${cm}$，其中$c = \max \left\{  {{c}_{R}/{c}_{S},{c}_{S}/{c}_{R}}\right\}  .$

Note that for each choice of ${d}^{\prime }$ ,the algorithm TA sees at least ${d}^{\prime }$ objects by depth ${d}^{\prime }$ (this is because by depth ${d}^{\prime }$ it has made $m{d}^{\prime }$ sorted accesses,and each object is accessed at most $m$ times under sorted access). Let $Y$ be the output set of $\mathcal{A}$ (consisting of the top $k$ objects). If there are at most $k$ objects that $\mathcal{A}$ does not see,then TA halts by depth $a + k$ (after having seen every object),and we are done. So assume that there are at least $k + 1$ objects that $\mathcal{A}$ does not see. Since $Y$ is of size $k$ ,there is some object $V$ that $\mathcal{A}$ does not see and that is not in $Y$ .

注意，对于${d}^{\prime }$的每个选择，TA算法在深度${d}^{\prime }$处至少看到${d}^{\prime }$个对象（这是因为在深度${d}^{\prime }$处，它已经进行了$m{d}^{\prime }$次排序访问，并且在排序访问下每个对象最多被访问$m$次）。设$Y$是$\mathcal{A}$的输出集（由前$k$个对象组成）。如果$\mathcal{A}$未看到的对象至多有$k$个，那么TA算法在深度$a + k$处停止（在看到每个对象之后），我们就完成了证明。因此，假设$\mathcal{A}$未看到的对象至少有$k + 1$个。由于$Y$的大小为$k$，所以存在某个对象$V$，$\mathcal{A}$未看到该对象且该对象不在$Y$中。

Let ${\tau }_{\mathcal{A}}$ be the threshold value when algorithm $\mathcal{A}$ halts. This means that if ${\underline{x}}_{i}$ is the grade of the last object seen under sorted access to list $i$ for algorithm $\mathcal{A}$ ,for $1 \leq  i \leq  m$ , then ${\tau }_{\mathcal{A}} = t\left( {{\underline{x}}_{1},\ldots ,{\underline{x}}_{m}}\right)$ . (For convenience,let us assume that algorithm $\mathcal{A}$ makes at least one sorted access to each list; this introduces at most $m$ more sorted accesses.) Let us call an object $R$ big if $t\left( R\right)  \geq  {\tau }_{\mathcal{A}}$ ,and otherwise call object $R$ small.

设${\tau }_{\mathcal{A}}$为算法$\mathcal{A}$停止时的阈值。这意味着，如果${\underline{x}}_{i}$是算法$\mathcal{A}$按排序访问列表$i$时所看到的最后一个对象的等级，对于$1 \leq  i \leq  m$，则有${\tau }_{\mathcal{A}} = t\left( {{\underline{x}}_{1},\ldots ,{\underline{x}}_{m}}\right)$。（为方便起见，我们假设算法$\mathcal{A}$对每个列表至少进行一次排序访问；这最多会引入$m$次额外的排序访问。）如果$t\left( R\right)  \geq  {\tau }_{\mathcal{A}}$，我们称对象$R$为大对象，否则称对象$R$为小对象。

We now show that every member $R$ of $Y$ is big. Define a database ${\mathcal{D}}^{\prime }$ to be just like $\mathcal{D}$ ,except that object $V$ has grade ${\underline{x}}_{i}$ in the $i$ th list,for $1 \leq  i \leq  m$ . Put $V$ in list $i$ below all other objects with grade ${\underline{x}}_{i}$ in list $i$ (for $1 \leq  i \leq  m$ ). Algorithm $\mathcal{A}$ performs exactly the same,and in particular gives the same output,for databases $\mathcal{D}$ and ${\mathcal{D}}^{\prime }$ . Therefore, algorithm $\mathcal{A}$ has $R$ ,but not $V$ ,in its output for database ${\mathcal{D}}^{\prime }$ . Since the grade of $V$ in ${\mathcal{D}}^{\prime }$ is ${\tau }_{\mathcal{A}}$ ,it follows by correctness of $\mathcal{A}$ that $R$ is big,as desired.

我们现在证明$Y$的每个成员$R$都是大对象。定义一个数据库${\mathcal{D}}^{\prime }$，它与$\mathcal{D}$完全相同，除了对象$V$在第$i$个列表中的等级为${\underline{x}}_{i}$（对于$1 \leq  i \leq  m$）。将对象$V$放在列表$i$中所有等级为${\underline{x}}_{i}$的其他对象之下（对于$1 \leq  i \leq  m$）。算法$\mathcal{A}$对数据库$\mathcal{D}$和${\mathcal{D}}^{\prime }$的执行过程完全相同，特别是输出也相同。因此，算法$\mathcal{A}$在数据库${\mathcal{D}}^{\prime }$的输出中包含$R$，但不包含$V$。由于$V$在${\mathcal{D}}^{\prime }$中的等级为${\tau }_{\mathcal{A}}$，根据$\mathcal{A}$的正确性可知，$R$是大对象，符合要求。

There are now two cases, depending on whether or not algorithm $\mathcal{A}$ sees every member of its output set $Y{.}^{8}$

现在有两种情况，取决于算法$\mathcal{A}$是否看到其输出集$Y{.}^{8}$的每个成员

Case 1: Algorithm $\mathcal{A}$ sees every member of $Y$ . Then by depth $d$ ,TA will see every member of $Y$ . Since,as we showed,each member of $Y$ is big,it follows that TA halts by depth $d \leq  a < a + k$ ,as desired.

情况1：算法$\mathcal{A}$看到$Y$的每个成员。那么根据深度$d$，TA将看到$Y$的每个成员。由于正如我们所证明的，$Y$的每个成员都是大对象，因此TA在深度$d \leq  a < a + k$时停止，符合要求。

Case 2: Algorithm $\mathcal{A}$ does not see some member $R$ of $Y$ . We now show that every object ${R}^{\prime }$ that is not seen by $\mathcal{A}$ must be big. Define a database ${\mathcal{D}}^{\prime }$ that is just like $\mathcal{D}$ on every object seen by $\mathcal{A}$ . Let the grade of $V$ in list $i$ be ${\underline{x}}_{i}$ , and put $V$ in list $i$ below all other objects with grade ${\underline{x}}_{i}$ in list $i$ (for $1 \leq  i \leq  m$ ). Therefore,the grade of $V$ in database ${\mathcal{D}}^{\prime }$ is ${\tau }_{\mathcal{A}}$ . Since $\mathcal{A}$ cannot distinguish between $\mathcal{D}$ and ${\mathcal{D}}^{\prime }$ ,it has the same output on $\mathcal{D}$ and ${\mathcal{D}}^{\prime }$ . Since $\mathcal{A}$ does not see $R$ and does not see ${R}^{\prime }$ ,it has no information to distinguish between $R$ and ${R}^{\prime }$ . Therefore,it must have been able to give ${R}^{\prime }$ in its output without making a mistake. But if ${R}^{\prime }$ is in the output and not $V$ ,then by correctness of $\mathcal{A}$ ,it follows that ${R}^{\prime }$ is big. So ${R}^{\prime }$ is big,as desired.

情况2：算法$\mathcal{A}$未看到$Y$的某个成员$R$。我们现在证明，$\mathcal{A}$未看到的每个对象${R}^{\prime }$必定是大对象。定义一个数据库${\mathcal{D}}^{\prime }$，它在$\mathcal{A}$看到的每个对象上都与$\mathcal{D}$相同。设$V$在列表$i$中的等级为${\underline{x}}_{i}$，并将$V$放在列表$i$中等级为${\underline{x}}_{i}$的所有其他对象之下（对于$1 \leq  i \leq  m$）。因此，$V$在数据库${\mathcal{D}}^{\prime }$中的等级为${\tau }_{\mathcal{A}}$。由于$\mathcal{A}$无法区分$\mathcal{D}$和${\mathcal{D}}^{\prime }$，所以它在$\mathcal{D}$和${\mathcal{D}}^{\prime }$上的输出相同。由于$\mathcal{A}$既未看到$R$也未看到${R}^{\prime }$，所以它没有信息来区分$R$和${R}^{\prime }$。因此，它必定能够在其输出中给出${R}^{\prime }$而不会出错。但如果${R}^{\prime }$在输出中而$V$不在，那么根据$\mathcal{A}$的正确性，可以得出${R}^{\prime }$是大对象。所以，正如我们所期望的，${R}^{\prime }$是大对象。

Since $\mathcal{A}$ sees $a$ objects,and since TA sees at least $a + k$ objects by depth $a + k$ ,it follows that by depth $a + k$ ,TA sees at least $k$ objects not seen by $\mathcal{A}$ . We have shown that every object that is not seen by $\mathcal{A}$ is big. Therefore,by depth $a + k$ ,TA sees at least $k$ big objects. So TA halts by depth $a + k$ ,as desired.

由于$\mathcal{A}$看到了$a$个对象，并且由于TA在深度$a + k$时至少看到$a + k$个对象，所以可以得出，在深度$a + k$时，TA至少看到$k$个$\mathcal{A}$未看到的对象。我们已经证明，$\mathcal{A}$未看到的每个对象都是大对象。因此，在深度$a + k$时，TA至少看到$k$个大对象。所以，正如我们所期望的，TA在深度$a + k$时停止。

We now show that making a lucky wild guess can help.

我们现在证明，进行一次幸运的大胆猜测会有所帮助。

EXAMPLE 4.4. Assume that there are ${2n} + 1$ objects,which we will call simply $1,2,\ldots ,{2n} + 1$ ,and there are two lists ${L}_{1}$ and ${L}_{2}$ . Assume that in list ${L}_{1}$ ,the objects are in the order $1,2,\ldots ,{2n} + 1$ ,where the top $n + 1$ objects $1,2,\ldots ,n + 1$ all have grade 1,and the remaining $n$ objects $n + 2,n +$ $3,\ldots ,{2n} + 1$ all have grade 0 . Assume that in list ${L}_{2}$ ,the objects are in the reverse order ${2n} + 1,{2n},\ldots ,1$ ,where the bottom $n$ objects $1,\ldots ,n$ all have grade 0,and the remaining $n + 1$ objects $n + 1,n + 2,\ldots ,{2n} + 1$ all have grade 1 . Assume that the aggregation function is $\min$ ,and that we are interested in finding the top answer (i.e., $k = 1$ ). It is clear that the top answer is object $n + 1$ with overall grade 1 (every object except object $n + 1$ has overall grade 0 ).

示例4.4。假设存在${2n} + 1$个对象，我们简单地将其称为$1,2,\ldots ,{2n} + 1$，并且有两个列表${L}_{1}$和${L}_{2}$。假设在列表${L}_{1}$中，对象的顺序为$1,2,\ldots ,{2n} + 1$，其中顶部的$n + 1$个对象$1,2,\ldots ,n + 1$的等级均为1，其余的$n$个对象$n + 2,n +$$3,\ldots ,{2n} + 1$的等级均为0。假设在列表${L}_{2}$中，对象的顺序为逆序${2n} + 1,{2n},\ldots ,1$，其中底部的$n$个对象$1,\ldots ,n$的等级均为0，其余的$n + 1$个对象$n + 1,n + 2,\ldots ,{2n} + 1$的等级均为1。假设聚合函数为$\min$，并且我们感兴趣的是找到顶级答案（即$k = 1$）。显然，顶级答案是总等级为1的对象$n + 1$（除对象$n + 1$之外的每个对象的总等级均为0）。

An algorithm that makes a wild guess and asks for the grade of object $n + 1$ in both lists would determine the correct answer and be able to halt safely after two random accesses and no sorted accesses. ${}^{9}$ However,let $\mathcal{A}$ be any algorithm (such as TA) that does not make wild guesses. Since the winning object $n + 1$ is in the middle of both sorted lists,it follows that at least $n + 1$ sorted accesses would be required before algorithm $\mathcal{A}$ would even see the winning object.

有一种算法会进行大胆猜测，并询问对象 $n + 1$ 在两个列表中的等级，该算法可以确定正确答案，并且在进行两次随机访问且不进行有序访问后能够安全停止。 ${}^{9}$ 然而，设 $\mathcal{A}$ 为任何不进行大胆猜测的算法（例如TA算法）。由于获胜对象 $n + 1$ 位于两个排序列表的中间，因此在算法 $\mathcal{A}$ 甚至还未看到获胜对象之前，至少需要进行 $n + 1$ 次有序访问。

Example 4.4 shows that TA is not instance optimal over the class $\mathbf{A}$ of all algorithms that find the top answer for min (with two arguments) and the class $\mathbf{D}$ of all databases. The next theorem says that no algorithm is instance optimal. The proof (and other missing proofs) appear in the full paper.

示例4.4表明，对于所有能找出最小值（两个参数）的最优答案的算法类 $\mathbf{A}$ 和所有数据库类 $\mathbf{D}$ 而言，TA算法并非实例最优的。下一个定理指出，不存在实例最优的算法。证明（以及其他缺失的证明）见完整论文。

THEOREM 4.5. Let $\mathbf{D}$ be the class of all databases. Let A be the class of all algorithms that correctly find the top answer for min (with two arguments) for every database. There is no deterministic algorithm (or even probabilistic algorithm that never makes a mistake) that is instance optimal over $\mathbf{A}$ and $\mathbf{D}$ .

定理4.5。设 $\mathbf{D}$ 为所有数据库的类。设A为所有能为每个数据库正确找出最小值（两个参数）的最优答案的算法类。不存在在 $\mathbf{A}$ 和 $\mathbf{D}$ 上实例最优的确定性算法（甚至不存在从不犯错的概率算法）。

Although, as we noted earlier, algorithms that make wild guesses would not normally be implemented in practice, it is still interesting to consider them. This is because of our interpretation of instance optimality of an algorithm $\mathcal{A}$ as saying that its cost is essentially the same as the cost of the shortest proof for that database that these are really the top $k$ answers. If we consider algorithms that allow wild guesses, then we are allowing a larger class of proofs. Thus,in Example 4.4,the fact that object $n + 1$ has (overall) grade 1 is a proof that it is the top answer.

尽管如我们之前所指出的，进行大胆猜测的算法通常在实践中不会被实现，但考虑它们仍然很有趣。这是因为我们将算法 $\mathcal{A}$ 的实例最优性解释为，其成本本质上与针对该数据库证明这些确实是前 $k$ 个最优答案的最短证明的成本相同。如果我们考虑允许进行大胆猜测的算法，那么我们就是在允许更大类别的证明。因此，在示例4.4中，对象 $n + 1$ 的（总体）等级为1这一事实就是它是最优答案的一个证明。

We say that an aggregation function $t$ is strictly monotone if $t\left( {{x}_{1},\ldots ,{x}_{m}}\right)  < t\left( {{x}_{1}^{\prime },\ldots ,{x}_{m}^{\prime }}\right)$ whenever ${x}_{i} < {x}_{i}^{\prime }$ for every $i$ . Although average and min are strictly monotone, there are aggregation functions suggested in the literature for representing conjunction and disjunction that are monotone but not strictly monotone (see [4] and [15] for examples). We say that a database $\mathcal{D}$ satisfies the uniqueness property if for each $i$ ,no two objects in $\mathcal{D}$ have the same grade in list ${L}_{i}$ ,that is,if the grades in list ${L}_{i}$ are distinct. We now show that these conditions guarantee optimality of TA even among algorithms that make wild guesses.

我们称聚合函数 $t$ 是严格单调的，如果只要对于每个 $i$ 都有 ${x}_{i} < {x}_{i}^{\prime }$ ，就有 $t\left( {{x}_{1},\ldots ,{x}_{m}}\right)  < t\left( {{x}_{1}^{\prime },\ldots ,{x}_{m}^{\prime }}\right)$ 。尽管平均值和最小值函数是严格单调的，但文献中提出的用于表示合取和析取的一些聚合函数是单调的，但不是严格单调的（例如，见文献[4]和[15]）。我们称数据库 $\mathcal{D}$ 满足唯一性属性，如果对于每个 $i$ ， $\mathcal{D}$ 中没有两个对象在列表 ${L}_{i}$ 中的等级相同，即列表 ${L}_{i}$ 中的等级是不同的。现在我们证明，这些条件保证了即使在进行大胆猜测的算法中，TA算法也是最优的。

THEOREM 4.6. Assume that the aggregation function $t$ is strictly monotone. Let $\mathbf{D}$ be the class of all databases that satisfy the uniqueness property. Let $\mathbf{A}$ be the class of all algorithms that correctly find the top $k$ answers for $t$ for every database in $\mathbf{D}$ . Then TA is instance optimal over $\mathbf{A}$ ${and}\;\mathbf{D}$ .

定理4.6。假设聚合函数 $t$ 是严格单调的。设 $\mathbf{D}$ 为所有满足唯一性属性的数据库的类。设 $\mathbf{A}$ 为所有能为 $\mathbf{D}$ 中的每个数据库正确找出 $t$ 的前 $k$ 个最优答案的算法的类。那么TA算法在 $\mathbf{A}$ ${and}\;\mathbf{D}$ 上是实例最优的。

Proof: Assume that $\mathcal{A} \in  \mathbf{A}$ ,and that algorithm $\mathcal{A}$ is run over database $\mathcal{D} \in  \mathbf{D}$ . Assume that $\mathcal{A}$ sees $a$ distinct objects (some possibly multiple times). We shall show that TA halts on $\mathcal{D}$ by depth $a + k$ . As before,this shows that the optimality ratio of TA is at most ${cm}$ ,where $c =$ $\max \left\{  {{c}_{R}/{c}_{S},{c}_{S}/{c}_{R}}\right\}$ .

证明：假设 $\mathcal{A} \in  \mathbf{A}$ ，并且算法 $\mathcal{A}$ 在数据库 $\mathcal{D} \in  \mathbf{D}$ 上运行。假设 $\mathcal{A}$ 看到了 $a$ 个不同的对象（有些对象可能被看到多次）。我们将证明TA算法在深度 $a + k$ 时在 $\mathcal{D}$ 上停止。如前所述，这表明TA算法的最优比至多为 ${cm}$ ，其中 $c =$ $\max \left\{  {{c}_{R}/{c}_{S},{c}_{S}/{c}_{R}}\right\}$ 。

If there are at most $k$ objects that $\mathcal{A}$ does not see,then TA halts by depth $a + k$ (after having seen every object), and we are done. So assume that there are at least $k + 1$ objects that $\mathcal{A}$ does not see. Since $Y$ is of size $k$ ,there is some object $V$ that $\mathcal{A}$ does not see and that is not in $Y$ . We shall show that TA halts on $\mathcal{D}$ by depth $a + 1$ .

如果$\mathcal{A}$看不到的对象最多有$k$个，那么TA在深度$a + k$处停止（在查看完每个对象之后），我们就完成了。因此，假设$\mathcal{A}$看不到的对象至少有$k + 1$个。由于$Y$的大小为$k$，所以存在某个对象$V$，$\mathcal{A}$看不到它，并且它不在$Y$中。我们将证明TA在深度$a + 1$处对$\mathcal{D}$停止。

---

<!-- Footnote -->

${}^{9}$ The algorithm could halt safely,since it "knows" that it has found an object with the maximal possible grade of 1 (this grade is maximal, since we are assuming that all grades lie between 0 and 1). Even if we did not assume that all grades lie between 0 and 1 , one additional sorted access would provide the information that each overall grade in the database is at most 1 .

${}^{9}$该算法可以安全地停止，因为它“知道”自己已经找到了一个等级最高为1的对象（这个等级是最高的，因为我们假设所有等级都在0到1之间）。即使我们不假设所有等级都在0到1之间，再进行一次排序访问也能提供数据库中每个总体等级至多为1的信息。

${}^{8}$ For the sake of generality,we are allowing the possibility that algorithm $\mathcal{A}$ can output an object that it has not seen. We discuss this issue more in Section 4.2.

${}^{8}$为了具有一般性，我们允许算法$\mathcal{A}$输出一个它未见过的对象的可能性。我们将在4.2节更详细地讨论这个问题。

<!-- Footnote -->

---

Let $\tau$ be the threshold value of TA at depth $a + 1$ . Thus, if ${\underline{x}}_{i}$ is the grade of the $\left( {a + 1}\right)$ th highest object in list $i$ ,then $\tau  = t\left( {{\underline{x}}_{1},\ldots ,{\underline{x}}_{m}}\right)$ . Let us call an object $R$ big if $t\left( R\right)  \geq  \tau$ , and otherwise call object $R$ small. (Note that these definitions of "big" and "small" are different from those in the proof of Theorem 4.3.)

设$\tau$为TA在深度$a + 1$处的阈值。因此，如果${\underline{x}}_{i}$是列表$i$中第$\left( {a + 1}\right)$高对象的等级，那么$\tau  = t\left( {{\underline{x}}_{1},\ldots ,{\underline{x}}_{m}}\right)$。如果$t\left( R\right)  \geq  \tau$，我们称对象$R$为“大对象”，否则称对象$R$为“小对象”。（注意，这里“大”和“小”的定义与定理4.3证明中的定义不同。）

We now show that every member $R$ of $Y$ is big. Let ${x}_{i}^{\prime }$ be some grade in the top $a + 1$ grades in list $i$ that is not the grade in list $i$ of any object seen by $\mathcal{A}$ . There is such a grade,since all grades in list $i$ are distinct,and $\mathcal{A}$ sees at most $a$ objects. Let ${\mathcal{D}}^{\prime }$ agree with $\mathcal{D}$ on all objects seen by A,and let object $V$ have grade ${x}_{i}^{\prime }$ in the $i$ th list of ${\mathcal{D}}^{\prime }$ ,for $1 \leq  i \leq  m$ . Hence,the grade of $V$ in ${\mathcal{D}}^{\prime }$ is $t\left( {{x}_{1}^{\prime },\ldots ,{x}_{m}^{\prime }}\right)  \geq  \tau$ . Since $V$ was unseen,and since $V$ is assigned grades in each list in ${\mathcal{D}}^{\prime }$ below the level that $\mathcal{A}$ reached by sorted access,it follows that algorithm $\mathcal{A}$ performs exactly the same,and in particular gives the same output,for databases $\mathcal{D}$ and ${\mathcal{D}}^{\prime }$ . Therefore,algorithm $\mathcal{A}$ has $R$ ,but not $V$ ,in its output for database ${\mathcal{D}}^{\prime }$ . By correctness of $\mathcal{A}$ ,it follows that $R$ is big, as desired.

我们现在证明$Y$的每个成员$R$都是大对象。设${x}_{i}^{\prime }$是列表$i$中前$a + 1$个等级中的某个等级，且它不是$\mathcal{A}$所看到的任何对象在列表$i$中的等级。存在这样的等级，因为列表$i$中的所有等级都是不同的，并且$\mathcal{A}$最多看到$a$个对象。设${\mathcal{D}}^{\prime }$在$\mathcal{A}$看到的所有对象上与$\mathcal{D}$一致，并且设对象$V$在${\mathcal{D}}^{\prime }$的第$i$个列表中的等级为${x}_{i}^{\prime }$，其中$1 \leq  i \leq  m$。因此，$V$在${\mathcal{D}}^{\prime }$中的等级为$t\left( {{x}_{1}^{\prime },\ldots ,{x}_{m}^{\prime }}\right)  \geq  \tau$。由于$V$未被看到，并且由于$V$在${\mathcal{D}}^{\prime }$的每个列表中被分配的等级都低于$\mathcal{A}$通过排序访问所达到的水平，所以算法$\mathcal{A}$对于数据库$\mathcal{D}$和${\mathcal{D}}^{\prime }$的操作完全相同，特别是给出相同的输出。因此，算法$\mathcal{A}$在数据库${\mathcal{D}}^{\prime }$的输出中包含$R$，但不包含$V$。根据$\mathcal{A}$的正确性，可知$R$是大对象，正如我们所期望的。

We claim that every member $R$ of $Y$ is one of the top $a + 1$ members of some list $i$ (and so is seen by TA by depth $a + 1)$ . Assume by way of contradiction that $R$ is not one of the top $a + 1$ members of list $i$ ,for $1 \leq  i \leq  m$ . By our assumptions that the aggregation function $t$ is strictly monotone. and that $\mathcal{D}$ satisfies the uniqueness property,it follows easily that $R$ is small. We already showed that every member of $Y$ is big. This contradiction proves the claim. It follows that TA halts by depth $a + 1$ ,as desired.

我们断言，$Y$中的每个元素$R$都是某个列表$i$的前$a + 1$个元素之一（因此在深度$a + 1)$时会被TA算法看到）。通过反证法假设，对于$1 \leq  i \leq  m$，$R$不是列表$i$的前$a + 1$个元素之一。根据我们的假设，聚合函数$t$是严格单调的，并且$\mathcal{D}$满足唯一性属性，很容易得出$R$是小元素。我们已经证明了$Y$中的每个元素都是大元素。这个矛盾证明了该断言。由此可知，TA算法如预期的那样在深度$a + 1$时停止。

The proofs of Theorems 4.3 and 4.6 have several nice properties:

定理4.3和4.6的证明有几个不错的性质：

- The proofs would still go through if we were in a scenario where,whenever a random access of object $R$ in list $i$ takes place,we learn not only the grade of $R$ in list $i$ ,but also the relative rank.

- 如果在我们所处的场景中，每当对列表$i$中的对象$R$进行随机访问时，我们不仅能得知$R$在列表$i$中的评分，还能得知其相对排名，那么这些证明仍然成立。

- The proofs would still go through if we were to restrict the class of databases to those where each list $i$ has a certain fixed domain.

- 如果我们将数据库的类别限制为每个列表$i$都有特定固定域的数据库，那么这些证明仍然成立。

- As we shall see, we can prove the instance optimality among approximation algorithms of an approximation version of TA, under the assumptions of Theorem 4.3, with only a small change to the proof (such a theorem does not hold under the assumptions of Theorem 4.6).

- 正如我们将看到的，在定理4.3的假设下，我们只需对证明做一个小的改动，就能证明TA算法的近似版本在近似算法中的实例最优性（在定理4.6的假设下，这样的定理不成立）。

### 4.2 Treating $\mathrm{k}$ and $\mathrm{m}$ as Constants

### 4.2 将$\mathrm{k}$和$\mathrm{m}$视为常数

In Theorems 4.3 and 4.6 about the instance optimality of TA,we are treating $k$ (where we are trying to find the top $k$ answers) and $m$ (the number of sorted lists) as constants. We now discuss these assumptions.

在关于TA算法实例最优性的定理4.3和4.6中，我们将$k$（我们试图找到前$k$个答案）和$m$（排序列表的数量）视为常数。现在我们来讨论这些假设。

We begin first with the assumption that $k$ is constant. As in the proofs of Theorems 4.3 and 4.6,let $a$ be the number of accesses by an algorithm $\mathcal{A} \in  \mathbf{A}$ . If $a \geq  k$ ,then there is no need to treat $k$ as a constant. Thus,if we were to restrict the class $\mathbf{A}$ of algorithms to contain only algorithms that make at least $k$ accesses to find the top $k$ answers,then there would be no need to assume that $k$ is constant. How can it arise that an algorithm $\mathcal{A}$ can find the top $k$ answers without making at least $k$ accesses,and in particular without accessing at least $k$ objects? It must then happen that either there are at most $k$ objects in the database,or else every object $R$ that $\mathcal{A}$ has not seen has the same overall grade $t\left( R\right)$ . The latter will occur,for example,if $t$ is a constant function. Even under these circumstances, it is still not reasonable in some contexts (such as certain database contexts) to allow an algorithm $\mathcal{A}$ to output an object as a member of the top $k$ objects without ever having seen it: how would the algorithm even know the name of the object? This is similar to an issue we raised earlier about wild guesses.

我们首先讨论$k$是常数这一假设。正如定理4.3和4.6的证明中那样，设$a$是算法$\mathcal{A} \in  \mathbf{A}$的访问次数。如果$a \geq  k$，那么就没有必要将$k$视为常数。因此，如果我们将算法类$\mathbf{A}$限制为仅包含那些为了找到前$k$个答案至少进行$k$次访问的算法，那么就没有必要假设$k$是常数。一个算法$\mathcal{A}$如何能在不进行至少$k$次访问，特别是不访问至少$k$个对象的情况下找到前$k$个答案呢？那么必然会出现以下两种情况之一：要么数据库中最多有$k$个对象，要么$\mathcal{A}$未访问过的每个对象$R$都有相同的总体评分$t\left( R\right)$。例如，如果$t$是一个常数函数，就会出现后一种情况。即使在这些情况下，在某些上下文（如某些数据库上下文）中，允许算法$\mathcal{A}$在从未见过某个对象的情况下将其作为前$k$个对象之一输出仍然是不合理的：算法怎么会知道该对象的名称呢？这与我们之前提出的关于胡乱猜测的问题类似。

We see from the proofs of Theorems 4.3 and 4.6 that the optimality ratio depends only on $m$ ,and is in fact linear in $m$ . The next theorem shows that the linear dependence of the optimality ratio of TA on $m$ in these theorems is essential. In fact, the next theorem shows that a dependence that is at least linear holds not just for TA, but for every correct deterministic algorithm (or even probabilistic algorithm that never makes a mistake). This dependence holds even when the aggregation function is min,and when $k = 1$ (so that we are interested only in the top answer). An analogous theorem about the dependence of the optimality ratio on $m$ holds also under the scenario of Theorem 4.6.

从定理4.3和4.6的证明中我们可以看出，最优性比率仅取决于$m$，并且实际上与$m$呈线性关系。下一个定理表明，在这些定理中TA算法的最优性比率对$m$的线性依赖是必不可少的。事实上，下一个定理表明，至少是线性的依赖关系不仅适用于TA算法，而且适用于每个正确的确定性算法（甚至是从不犯错的概率算法）。即使当聚合函数是最小值函数，并且$k = 1$（这样我们只对第一个答案感兴趣）时，这种依赖关系仍然成立。在定理4.6的场景下，关于最优性比率对$m$的依赖关系也有一个类似的定理。

THEOREM 4.7. Let $\mathbf{D}$ be the class of all databases. Let A be the class of all algorithms that correctly find the top answer for min for every database and that do not make wild guesses. There is no deterministic algorithm (or even probabilistic algorithm that never makes a mistake) with an optimality ratio over $\mathbf{A}$ and $\mathbf{D}$ that is less than $m/2$ .

定理4.7。设$\mathbf{D}$为所有数据库的类。设A为所有能为每个数据库正确找出最小值的最优答案且不进行胡乱猜测的算法的类。不存在一个确定性算法（甚至是从不犯错的概率算法），其在$\mathbf{A}$和$\mathbf{D}$上的最优比小于$m/2$。

### 4.3 Turning TA into an Approximation Algo- rithm

### 4.3 将TA转化为近似算法

TA can easily be modified to be an approximation algorithm. It can then be used in situations where we care only about the approximately top $k$ answers. Thus,let $\theta  > 1$ be given. Let us say that an algorithm finds a $\theta$ -approximation to the top $k$ answers for $t$ over database $\mathcal{D}$ if it gives as output $k$ objects (and their grades) such that for each $y$ among these $k$ objects and each $z$ not among these $k$ objects, ${\theta t}\left( y\right)  \geq  t\left( z\right)$ . We can modify TA to work under these requirements by modifying the stopping rule in Part 2 to say "As soon as at least $k$ objects have been seen whose grade, when multiplied by $\theta$ ,is at least equal to $\tau$ ,then halt." Let us call this approximation algorithm ${\mathrm{{TA}}}_{\theta }$ . A straightforward modification of the proof of Theorem 4.1 shows that ${\mathrm{{TA}}}_{\theta }$ is correct. We now show that if no wild guesses are allowed,then ${\mathrm{{TA}}}_{\theta }$ is instance optimal.

可以很容易地将TA修改为一个近似算法。然后它可以用于我们只关心近似最优的$k$个答案的情况。因此，给定$\theta  > 1$。如果一个算法针对数据库$\mathcal{D}$给出$k$个对象（及其得分）作为输出，使得对于这$k$个对象中的每个$y$以及不在这$k$个对象中的每个$z$，都有${\theta t}\left( y\right)  \geq  t\left( z\right)$，那么我们称该算法找到了$t$的最优$k$个答案的$\theta$ - 近似。我们可以通过修改第2部分中的停止规则，使其表述为“一旦至少看到$k$个对象，其得分乘以$\theta$后至少等于$\tau$，则停止”，来让TA在这些要求下工作。我们将这个近似算法称为${\mathrm{{TA}}}_{\theta }$。对定理4.1证明的直接修改表明${\mathrm{{TA}}}_{\theta }$是正确的。现在我们证明，如果不允许进行胡乱猜测，那么${\mathrm{{TA}}}_{\theta }$是实例最优的。

THEOREM 4.8. Assume that $\theta  > 1$ and that the aggregation function $t$ is monotone. Let $\mathbf{D}$ be the class of all databases. Let $\mathbf{A}$ be the class of all algorithms that find ${a\theta }$ - approximation to the top $k$ answers for $t$ for every database and that do not make wild guesses. Then $T{A}_{\theta }$ is instance optimal over $\mathbf{A}$ and $\mathbf{D}$ .

定理4.8。假设$\theta  > 1$且聚合函数$t$是单调的。设$\mathbf{D}$为所有数据库的类。设$\mathbf{A}$为所有能为每个数据库找出$t$的最优$k$个答案的${a\theta }$ - 近似且不进行胡乱猜测的算法的类。那么$T{A}_{\theta }$在$\mathbf{A}$和$\mathbf{D}$上是实例最优的。

Proof: The proof of Theorem 4.3 carries over verbatim provided we modify the definition of an object $R$ being "big" to be that ${\theta t}\left( R\right)  \geq  {\tau }_{\mathcal{A}}$ . $\square$

证明：只要我们将对象$R$为“大”的定义修改为${\theta t}\left( R\right)  \geq  {\tau }_{\mathcal{A}}$，定理4.3的证明就可以逐字照搬。$\square$

Theorem 4.8 shows that the analog of Theorem 4.3 holds for ${\mathrm{{TA}}}_{\theta }$ . The next example,which is a modification of Example 4.4, shows that the analog of Theorem 4.6 does not hold for ${\mathrm{{TA}}}_{\theta }$ . One interpretation of these results is that Theorem 4.3 is sufficiently robust that it can survive the perturbation of allowing approximations, whereas Theorem 4.6 is not.

定理4.8表明定理4.3的类似结论对于${\mathrm{{TA}}}_{\theta }$成立。下一个例子是对例4.4的修改，它表明定理4.6的类似结论对于${\mathrm{{TA}}}_{\theta }$不成立。对这些结果的一种解释是，定理4.3足够稳健，能够承受允许近似带来的扰动，而定理4.6则不能。

EXAMPLE 4.9. Assume that $\theta  > 1$ ,that there are ${2n} + 1$ objects,which we will call simply $1,2,\ldots ,{2n} + 1$ ,and that there are two lists ${L}_{1}$ and ${L}_{2}$ . Assume that in list ${L}_{1}$ ,the grades are assigned so that all grades are different, the ordering of the objects by grade is $1,2,\ldots ,{2n} + 1$ ,object $n + 1$ has the grade $1/\theta$ ,and object $n + 2$ has the grade $1/\left( {2{\theta }^{2}}\right)$ . Assume that in list ${L}_{2}$ ,the grades are assigned so that all grades are different, the ordering of the objects by grade is ${2n} + 1,{2n},\ldots ,1$ (the reverse of the ordering in ${L}_{1}$ ),object $n + 1$ has the grade $1/\theta$ ,and object $n + 2$ has the grade $1/\left( {2{\theta }^{2}}\right)$ . Assume that the aggregation function is $\min$ ,and that $k = 1$ (so that we are interested in finding a $\theta$ -approximation to the top answer). The (overall) grade of each object other than object $n + 1$ is at most $\alpha  = 1/\left( {2{\theta }^{2}}\right)$ . Since ${\theta \alpha } = 1/\left( {2\theta }\right)$ ,which is less than the grade $1/\theta$ of object $n + 1$ ,it follows that the unique object that can be returned by an algorithm such as ${\mathrm{{TA}}}_{\theta }$ that correctly finds a $\theta$ -approximation to the top answer is the object $n + 1$ .

示例4.9。假设$\theta  > 1$，存在${2n} + 1$个对象，我们简单地将其称为$1,2,\ldots ,{2n} + 1$，并且有两个列表${L}_{1}$和${L}_{2}$。假设在列表${L}_{1}$中，所分配的等级各不相同，按等级对对象进行排序的结果为$1,2,\ldots ,{2n} + 1$，对象$n + 1$的等级为$1/\theta$，对象$n + 2$的等级为$1/\left( {2{\theta }^{2}}\right)$。假设在列表${L}_{2}$中，所分配的等级也各不相同，按等级对对象进行排序的结果为${2n} + 1,{2n},\ldots ,1$（与列表${L}_{1}$中的排序相反），对象$n + 1$的等级为$1/\theta$，对象$n + 2$的等级为$1/\left( {2{\theta }^{2}}\right)$。假设聚合函数为$\min$，并且$k = 1$（这样我们就有兴趣找到对最优答案的$\theta$ - 近似解）。除对象$n + 1$之外的每个对象的（总体）等级至多为$\alpha  = 1/\left( {2{\theta }^{2}}\right)$。由于${\theta \alpha } = 1/\left( {2\theta }\right)$小于对象$n + 1$的等级$1/\theta$，因此像${\mathrm{{TA}}}_{\theta }$这样能正确找到对最优答案的$\theta$ - 近似解的算法所能返回的唯一对象就是对象$n + 1$。

An algorithm that makes a wild guess and asks for the grade of object $n + 1$ in both lists would determine the correct answer and be able to halt safely after two random accesses and no sorted accesses. The algorithm could halt safely, since it "knows" that it has found an object $R$ such that ${\theta t}\left( R\right)  = 1$ ,and so ${\theta t}\left( R\right)$ is at least as big as every possible grade. However,under sorted access for list ${L}_{1},{\mathrm{{TA}}}_{\theta }$ would see the objects in the order $1,2,\ldots ,{2n} + 1$ ,and under sorted access for list ${L}_{2},{\mathrm{{TA}}}_{\theta }$ would see the objects in the reverse order. Since the winning object $n + 1$ is in the middle of both sorted lists,it follows that at least $n + 1$ sorted accesses would be required before ${\mathrm{{TA}}}_{\theta }$ would even see the winning object.

一个随意猜测并询问对象$n + 1$在两个列表中等级的算法能够确定正确答案，并且在进行两次随机访问且不进行排序访问之后就可以安全地停止。该算法可以安全停止，因为它“知道”已经找到了一个对象$R$，使得${\theta t}\left( R\right)  = 1$，因此${\theta t}\left( R\right)$至少和每个可能的等级一样大。然而，对于列表${L}_{1},{\mathrm{{TA}}}_{\theta }$进行排序访问时会按$1,2,\ldots ,{2n} + 1$的顺序查看对象，对于列表${L}_{2},{\mathrm{{TA}}}_{\theta }$进行排序访问时会按相反的顺序查看对象。由于获胜对象$n + 1$位于两个排序列表的中间，因此在${\mathrm{{TA}}}_{\theta }$甚至看到获胜对象之前，至少需要进行$n + 1$次排序访问。

Just as Example 4.4 was generalized into Theorem 4.5, we can generalize Example 4.9 into the following theorem.

正如示例4.4被推广为定理4.5一样，我们可以将示例4.9推广为以下定理。

THEOREM 4.10. Assume that $\theta  > 1$ . Let $\mathbf{D}$ be the class of all databases that satisfy the uniqueness condition. Let $\mathbf{A}$ be the class of all algorithms that find a $\theta$ -approximation to the top answer for $\min$ for every database in $\mathbf{D}$ . There is no deterministic algorithm (or even probabilistic algorithm that never makes a mistake) that is instance optimal over $\mathbf{A}$ and D.

定理4.10。假设$\theta  > 1$。设$\mathbf{D}$为满足唯一性条件的所有数据库的类。设$\mathbf{A}$为对于$\mathbf{D}$中的每个数据库，能找到$\min$的最优答案的$\theta$ - 近似解的所有算法的类。不存在在$\mathbf{A}$和D上实例最优的确定性算法（甚至是从不犯错的概率算法）。

## 5. MINIMIZING RANDOM ACCESS

## 5. 最小化随机访问

Thus far in this paper, we have not been especially concerned about the number of random accesses. In our algorithms we have discussed so far (namely, FA and TA), for every sorted access,up to $m - 1$ random accesses take place. Recall that if $s$ is the number of sorted accesses,and $r$ is the number of random accesses, then the middleware cost is $s{c}_{S} + r{c}_{R}$ ,for some positive constants ${c}_{S}$ and ${c}_{R}$ . Our notion of optimality ignores constant factors like $m$ and ${c}_{R}$ (they are simply multiplicative factors in the optimality ratio). Hence, there has been no motivation so far to concern ourself with the number of random accesses.

到目前为止，在本文中，我们并未特别关注随机访问的次数。在我们目前讨论过的算法（即FA和TA）中，对于每一次有序访问，最多会进行$m - 1$次随机访问。回想一下，如果$s$是有序访问的次数，$r$是随机访问的次数，那么中间件成本为$s{c}_{S} + r{c}_{R}$，其中${c}_{S}$和${c}_{R}$是一些正常数。我们的最优性概念忽略了像$m$和${c}_{R}$这样的常数因子（它们只是最优比中的乘法因子）。因此，到目前为止，没有理由让我们关注随机访问的次数。

There are, however, some scenarios where we must pay attention to the number of random accesses. The first scenario is where random accesses are impossible (which corresponds to ${c}_{R} = \infty$ ). As we discussed in Section 2,an example of this first scenario arises when the middleware system is a text retrieval system, and the sorted lists correspond to the results of search engines. Another scenario is where random accesses are not impossible, but simply expensive, relative to sorted access. An example of this second scenario arises when the costs correspond to disk access (sequential versus random). Then we would like the optimality ratio to be independent of ${c}_{R}/{c}_{S}$ . That is,if instead of treating ${c}_{S}$ and ${c}_{R}$ as constants,we allow them to vary,we would still like the optimality ratio to be bounded.

然而，在某些场景下，我们必须关注随机访问的次数。第一种场景是随机访问无法进行的情况（对应于${c}_{R} = \infty$）。正如我们在第2节中所讨论的，当中间件系统是一个文本检索系统，并且有序列表对应于搜索引擎的搜索结果时，就会出现这种第一种场景的例子。另一种场景是随机访问并非不可能，但相对于有序访问而言成本较高。当成本对应于磁盘访问（顺序访问与随机访问）时，就会出现这种第二种场景的例子。那么我们希望最优比与${c}_{R}/{c}_{S}$无关。也就是说，如果我们不将${c}_{S}$和${c}_{R}$视为常数，而是允许它们变化，我们仍然希望最优比是有界的。

In this section we describe algorithms that do not use random access frivolously. We give two algorithms. One uses no random accesses at all, and hence is called NRA ( "No Random Access"). The second algorithm takes into account the cost of a random access. It is a combination of NRA and TA, and so we call it CA ("Combined Algorithm").

在本节中，我们描述了不会随意使用随机访问的算法。我们给出两种算法。一种完全不使用随机访问，因此被称为NRA（“无随机访问”）。第二种算法考虑了随机访问的成本。它是NRA和TA的组合，因此我们称其为CA（“组合算法”）。

Both algorithms access the information in a natural way, and intuitively, halt when they know that no improvement can take place. In general, at each point in an execution of these algorithms where a number of sorted and random accesses have taken place,for each object $R$ there is a subset $S\left( R\right)  = \left\{  {{i}_{1},{i}_{2},\ldots ,{i}_{\ell }}\right\}   \subseteq  \{ 1,\ldots ,m\}$ of the fields of $R$ where the algorithm has determined the values ${x}_{{i}_{1}},{x}_{{i}_{2}},\ldots ,{x}_{{i}_{\ell }}$ of these fields. Given this information, we define functions of this information that are lower and upper bounds on the value $t\left( R\right)$ can obtain. The algorithm proceeds until there are no more candidates whose current upper bound is better than the current $k$ th largest lower bound.

这两种算法都以自然的方式访问信息，并且直观地说，当它们知道无法再进行改进时就会停止。一般来说，在这些算法执行过程中的每一个点，当已经进行了一定数量的有序和随机访问时，对于每个对象$R$，存在$R$的字段的一个子集$S\left( R\right)  = \left\{  {{i}_{1},{i}_{2},\ldots ,{i}_{\ell }}\right\}   \subseteq  \{ 1,\ldots ,m\}$，算法已经确定了这些字段的值${x}_{{i}_{1}},{x}_{{i}_{2}},\ldots ,{x}_{{i}_{\ell }}$。根据这些信息，我们定义关于这些信息的函数，它们是对象$t\left( R\right)$所能获得的值的下界和上界。算法会一直执行，直到没有更多候选对象的当前上界优于当前第$k$大的下界。

Lower Bound: Given an object $R$ and subset $S\left( R\right)  =$ $\left\{  {{i}_{1},{i}_{2},\ldots ,{i}_{\ell }}\right\}   \subseteq  \{ 1,\ldots ,m\}$ of known fields of $R$ ,with values ${x}_{{i}_{1}},{x}_{{i}_{2}},\ldots ,{x}_{{i}_{\ell }}$ for these known fields,we define ${W}_{S}\left( R\right)$ (or $\mathrm{W}\left( \mathrm{R}\right)$ if the subset $S = S\left( R\right)$ is clear) as the minimum (or worst) value the aggregation function $t$ can attain for object $R$ . When $t$ is monotone,this minimum value is obtained by substituting for each missing field $i \in  \{ 1,\ldots ,m\}  \smallsetminus  S$ the value 0,and applying $t$ to the result. For example,if $S =$ $\{ 1,\ldots ,\ell \}$ ,then ${W}_{S}\left( R\right)  = t\left( {{x}_{1},{x}_{2},\ldots ,{x}_{\ell },0,\ldots ,0}\right)$ . The following property is immediate from the definition:

下界：给定一个对象$R$和$R$的已知字段的子集$S\left( R\right)  =$ $\left\{  {{i}_{1},{i}_{2},\ldots ,{i}_{\ell }}\right\}   \subseteq  \{ 1,\ldots ,m\}$，这些已知字段的值为${x}_{{i}_{1}},{x}_{{i}_{2}},\ldots ,{x}_{{i}_{\ell }}$，我们将${W}_{S}\left( R\right)$（如果子集$S = S\left( R\right)$明确，则为$\mathrm{W}\left( \mathrm{R}\right)$）定义为聚合函数$t$对于对象$R$所能达到的最小值（或最差值）。当$t$是单调的时，这个最小值是通过将每个缺失字段$i \in  \{ 1,\ldots ,m\}  \smallsetminus  S$的值替换为0，并将$t$应用于结果得到的。例如，如果$S =$ $\{ 1,\ldots ,\ell \}$，那么${W}_{S}\left( R\right)  = t\left( {{x}_{1},{x}_{2},\ldots ,{x}_{\ell },0,\ldots ,0}\right)$。根据定义，以下性质是显而易见的：

Proposition 5.1. If $S$ is the set of known fields of object $R$ ,then $t\left( R\right)  \geq  {W}_{S}\left( R\right)$ .

命题5.1。若$S$是对象$R$的已知字段集合，则$t\left( R\right)  \geq  {W}_{S}\left( R\right)$。

In other words, $W\left( R\right)$ represents a lower bound on $t\left( R\right)$ . Is it the best possible? Yes, unless we have additional information, such as that the value 0 does not appear in the lists. In general, as an algorithm progresses and we learn more fields of an object $R$ ,its $W$ value becomes larger (or at least not smaller). For some aggregation functions $t$ the value $W\left( R\right)$ yields no knowledge until $S$ includes all fields: for instance if $t$ is min,then $W\left( R\right)$ is 0 until all values are discovered. For other functions it is more meaningful. For instance,when $t$ is the median of three fields, then as soon as two of them are known $W\left( R\right)$ is at least the smaller of the two.

换句话说，$W\left( R\right)$表示$t\left( R\right)$的一个下界。它是可能的最优下界吗？是的，除非我们有额外信息，例如值0不在列表中出现。一般来说，随着算法推进，我们了解到对象$R$的更多字段，其$W$值会变大（或者至少不会变小）。对于某些聚合函数$t$，在$S$包含所有字段之前，$W\left( R\right)$值不提供任何信息：例如，如果$t$是最小值函数，那么在所有值都被发现之前，$W\left( R\right)$为0。对于其他函数，它更有意义。例如，当$t$是三个字段的中位数时，一旦其中两个字段已知，$W\left( R\right)$至少是这两个值中的较小值。

Upper Bound: The best value an object can attain depends on other information we have. We will use only the bottom values in each field,defined as in TA: ${\underline{x}}_{i}$ is the last (smallest) value obtained via sorted access in list ${L}_{i}$ . Given an object $R$ and subset $S\left( R\right)  = \left\{  {{i}_{1},{i}_{2},\ldots ,{i}_{\ell }}\right\}   \subseteq  \{ 1,\ldots ,m\}$ of known fields of $R$ ,with values ${x}_{{i}_{1}},{x}_{{i}_{2}},\ldots ,{x}_{{i}_{\ell }}$ for these known fields,we define ${B}_{S}\left( R\right)$ (or $\mathrm{B}\left( \mathrm{R}\right)$ if the subset $S$ is clear) as the maximum (or best) value the aggregation function $t$ can attain for object $R$ . When $t$ is monotone,this maximum value is obtained by substituting for each missing field $i \in  \{ 1,\ldots ,m\}  \smallsetminus  S$ the value ${\underline{x}}_{i}$ ,and applying $t$ to the result. For example,if $S = \{ 1,\ldots ,\ell \}$ ,then ${B}_{S}\left( R\right)  =$ $\left. {t\left( {{x}_{1},{x}_{2},\ldots ,{x}_{\ell },{\underline{x}}_{\ell  + 1},\ldots ,{\underline{x}}_{m}}\right) }\right)$ . The following property is immediate from the definition:

上界：一个对象所能达到的最优值取决于我们拥有的其他信息。我们仅使用每个字段中的底部值，其定义与TA中相同：${\underline{x}}_{i}$是通过对列表${L}_{i}$进行排序访问得到的最后一个（最小的）值。给定一个对象$R$以及$R$的已知字段子集$S\left( R\right)  = \left\{  {{i}_{1},{i}_{2},\ldots ,{i}_{\ell }}\right\}   \subseteq  \{ 1,\ldots ,m\}$，这些已知字段的值为${x}_{{i}_{1}},{x}_{{i}_{2}},\ldots ,{x}_{{i}_{\ell }}$，我们将${B}_{S}\left( R\right)$（如果子集$S$明确，则为$\mathrm{B}\left( \mathrm{R}\right)$）定义为聚合函数$t$对对象$R$所能达到的最大值（或最优值）。当$t$是单调函数时，这个最大值是通过用值${\underline{x}}_{i}$替代每个缺失字段$i \in  \{ 1,\ldots ,m\}  \smallsetminus  S$，并将$t$应用于结果得到的。例如，如果$S = \{ 1,\ldots ,\ell \}$，那么${B}_{S}\left( R\right)  =$ $\left. {t\left( {{x}_{1},{x}_{2},\ldots ,{x}_{\ell },{\underline{x}}_{\ell  + 1},\ldots ,{\underline{x}}_{m}}\right) }\right)$。根据定义，以下性质是显然的：

Proposition 5.2. If $S$ is the set of known fields of object $R$ ,then $t\left( R\right)  \leq  {B}_{S}\left( R\right)$ .

命题5.2。若$S$是对象$R$的已知字段集合，则$t\left( R\right)  \leq  {B}_{S}\left( R\right)$。

In other words, $B\left( R\right)$ represents an upper bound on the value $t\left( R\right)$ (or the best value $t\left( R\right)$ can be),given the information we have so far. Is it the best upper bound? If the lists may each contain equal values (which in general we assume they can), then given the information we have it is possible that $t\left( R\right)  = {B}_{S}\left( R\right)$ . If the uniqueness property holds (equalities are not allowed in a list), then for continuous aggregation functions $t$ it is the case that $B\left( R\right)$ is the best upper bound on the value $t$ can have on $R$ . In general, as an algorithm progresses and we learn more fields of an object $R$ and the bottom values ${\underline{x}}_{i}$ decrease, $B\left( R\right)$ can only decrease (or remain the same).

换句话说，给定我们目前所拥有的信息，$B\left( R\right)$表示值$t\left( R\right)$的一个上界（或者$t\left( R\right)$所能达到的最优值）。它是最优上界吗？如果列表中可能各自包含相等的值（一般我们假设可以），那么根据我们所拥有的信息，有可能$t\left( R\right)  = {B}_{S}\left( R\right)$。如果唯一性属性成立（列表中不允许有相等的值），那么对于连续聚合函数$t$，$B\left( R\right)$是$t$在$R$上所能具有的值的最优上界。一般来说，随着算法推进，我们了解到对象$R$的更多字段，并且底部值${\underline{x}}_{i}$减小，$B\left( R\right)$只会减小（或者保持不变）。

An important special case is an object $R$ that has not been encountered at all. In this case $B\left( R\right)  = t\left( {{\underline{x}}_{1},{\underline{x}}_{2},\ldots ,{\underline{x}}_{m}}\right)$ . Note that this is the same as the threshold value in TA.

一个重要的特殊情况是一个完全未被访问过的对象 $R$。在这种情况下 $B\left( R\right)  = t\left( {{\underline{x}}_{1},{\underline{x}}_{2},\ldots ,{\underline{x}}_{m}}\right)$。请注意，这与TA中的阈值相同。

### 5.1 No Random Access Algorithm-NRA

### 5.1 无随机访问算法 - NRA

As we have discussed, there are situations where random accesses are forbidden. We now consider algorithms that make no random accesses. Since random accesses are forbidden, in this section we change our criterion for the desired output. In earlier sections, we demanded that the output be the "top $k$ answers",which consists of the top $k$ objects, along with their (overall) grades. In this section, we make the weaker requirement that the output consist of the top $k$ objects,without their grades. The reason is that,since random access is impossible, it may be much cheaper (that is,require many fewer accesses) to find the top $k$ answers without their grades. This is because, as we now show by example, we can sometimes obtain enough partial information about grades to know that an object is in the top $k$ objects without knowing its exact grade.

正如我们所讨论的，存在禁止随机访问的情况。现在我们考虑不进行随机访问的算法。由于禁止随机访问，在本节中我们改变对期望输出的评判标准。在前面的章节中，我们要求输出是“前 $k$ 个答案”，它由前 $k$ 个对象及其（总体）评分组成。在本节中，我们提出较弱的要求，即输出仅包含前 $k$ 个对象，而不包含它们的评分。原因是，由于无法进行随机访问，在不获取评分的情况下找到前 $k$ 个答案可能成本更低（即需要的访问次数少得多）。正如我们现在通过示例所示，有时我们可以获得足够的关于评分的部分信息，从而在不知道对象确切评分的情况下知道该对象在前 $k$ 个对象之中。

EXAMPLE 5.3. Consider the following scenario, where the aggregation function is the average,and where $k = 1$ (so that we are interested only in the top object). There are only two sorted lists ${L}_{1}$ and ${L}_{2}$ ,and the grade of every object in both ${L}_{1}$ and ${L}_{2}$ is $1/3$ ,except that object $R$ has grade 1 in ${L}_{1}$ and grade 0 in ${L}_{2}$ . After two sorted accesses to ${L}_{1}$ and one sorted access to ${L}_{2}$ ,there is enough information to know that object $R$ is the top object (its average grade is at least $1/2$ ,and every other object has average grade at most $1/3$ ). If we wished to find the grade of object $R$ ,we would need to do sorted access to all of ${L}_{2}$ .

示例5.3。考虑以下场景，其中聚合函数是平均值，且 $k = 1$（因此我们只关注排名最高的对象）。只有两个排序列表 ${L}_{1}$ 和 ${L}_{2}$，并且 ${L}_{1}$ 和 ${L}_{2}$ 中每个对象的评分都是 $1/3$，除了对象 $R$ 在 ${L}_{1}$ 中的评分为1，在 ${L}_{2}$ 中的评分为0。在对 ${L}_{1}$ 进行两次有序访问并对 ${L}_{2}$ 进行一次有序访问后，有足够的信息可以知道对象 $R$ 是排名最高的对象（其平均评分至少为 $1/2$，而其他每个对象的平均评分至多为 $1/3$）。如果我们希望找到对象 $R$ 的评分，我们需要对 ${L}_{2}$ 进行全量有序访问。

Note that we are requiring only that the output consist of the top $k$ objects,with no information being given about the sorted order (sorted by grade). If we wish to know the sorted order, this can easily be determined by finding the top object,the top 2 objects,etc. Let ${C}_{i}$ be the cost of finding the top $i$ objects. It is interesting to note that there is no necessary relationship between ${C}_{i}$ and ${C}_{j}$ for $i < j$ . For example,in Example 5.3,we have ${C}_{1} < {C}_{2}$ . If we were to modify Example 5.3 so that there are two objects $R$ and ${R}^{\prime }$ with grade 1 in ${L}_{1}$ ,where the grade of $R$ in ${L}_{2}$ is 0, and the grade of ${R}^{\prime }$ in ${L}_{2}$ is $1/4$ (and so that,as before, all remaining grades of all objects in both lists is $1/3$ ),then ${C}_{2} < {C}_{1}$ .

请注意，我们只要求输出包含前 $k$ 个对象，而不提供（按评分排序的）排序顺序信息。如果我们想知道排序顺序，可以通过依次找出排名最高的对象、前2个对象等来轻松确定。设 ${C}_{i}$ 为找出前 $i$ 个对象的成本。有趣的是，对于 $i < j$，${C}_{i}$ 和 ${C}_{j}$ 之间没有必然的关系。例如，在示例5.3中，我们有 ${C}_{1} < {C}_{2}$。如果我们修改示例5.3，使得在 ${L}_{1}$ 中有两个对象 $R$ 和 ${R}^{\prime }$ 的评分为1，其中 $R$ 在 ${L}_{2}$ 中的评分为0，${R}^{\prime }$ 在 ${L}_{2}$ 中的评分为 $1/4$（并且和之前一样，两个列表中所有对象的其余评分均为 $1/3$），那么 ${C}_{2} < {C}_{1}$。

The cost of finding the top $k$ objects in sorted order is at most $k\mathop{\max }\limits_{i}{C}_{i}$ . Since we are treating $k$ as a constant, it follows easily that we can convert our instance optimal algorithm (which we shall give shortly) for finding the top $k$ objects into an instance optimal algorithm for finding the top $k$ objects in sorted order. In practice,it is usually good enough to know the top $k$ objects in sorted order,without knowing the grades. In fact, the major search engines on the web no longer give grades (possibly to prevent reverse engineering).

按排序顺序找出前 $k$ 个对象的成本至多为 $k\mathop{\max }\limits_{i}{C}_{i}$。由于我们将 $k$ 视为常数，因此很容易得出，我们可以将即将给出的用于找出前 $k$ 个对象的实例最优算法转换为用于按排序顺序找出前 $k$ 个对象的实例最优算法。在实践中，通常知道按排序顺序排列的前 $k$ 个对象就足够了，而无需知道评分。事实上，网络上的主要搜索引擎不再提供评分（可能是为了防止逆向工程）。

The algorithm NRA is as follows.

算法NRA如下。

1. Do sorted access in parallel to each of the $m$ sorted lists ${L}_{i}$ . At each depth $d$ (when $d$ objects have been accessed under sorted access in each list) maintain the following:

1. 并行地对 $m$ 个排序列表 ${L}_{i}$ 进行有序访问。在每个深度 $d$（当每个列表在有序访问下已访问了 $d$ 个对象时），维护以下内容：

- The bottom values ${\underline{x}}_{1}^{\left( d\right) },{\underline{x}}_{2}^{\left( d\right) },\ldots ,{\underline{x}}_{m}^{\left( d\right) }$ encountered in the lists.

- 列表中遇到的最小值 ${\underline{x}}_{1}^{\left( d\right) },{\underline{x}}_{2}^{\left( d\right) },\ldots ,{\underline{x}}_{m}^{\left( d\right) }$。

- For every object $R$ with discovered fields $S =$ ${S}^{\left( d\right) }\left( R\right)  \subseteq  \{ 1,\ldots ,m\}$ the values ${W}^{\left( d\right) }\left( R\right)  = {W}_{S}\left( R\right)$ and ${B}^{\left( d\right) }\left( R\right)  = {B}_{S}\left( R\right)$ .

- 对于每个具有已发现字段 $S =$ ${S}^{\left( d\right) }\left( R\right)  \subseteq  \{ 1,\ldots ,m\}$ 的对象 $R$，其值为 ${W}^{\left( d\right) }\left( R\right)  = {W}_{S}\left( R\right)$ 和 ${B}^{\left( d\right) }\left( R\right)  = {B}_{S}\left( R\right)$。

- The $k$ objects with the largest ${W}^{\left( d\right) }$ values seen so far (and their grades); if two objects have the same ${W}^{\left( d\right) }$ value,then ties are broken using the ${B}^{\left( d\right) }$ values,such that the object with the highest ${B}^{\left( d\right) }$ value wins (and arbitrarily if there is a tie for the highest ${B}^{\left( d\right) }$ value). Denote this top $k$ list by ${T}_{k}^{\left( d\right) }$ . Let ${M}_{k}^{\left( d\right) }$ be the $k$ th largest ${W}^{\left( d\right) }$ value in ${T}_{k}^{\left( d\right) }$ .

- 到目前为止所见到的具有最大 ${W}^{\left( d\right) }$ 值的 $k$ 个对象（及其等级）；如果两个对象具有相同的 ${W}^{\left( d\right) }$ 值，则使用 ${B}^{\left( d\right) }$ 值来打破平局，使得 ${B}^{\left( d\right) }$ 值最高的对象获胜（如果 ${B}^{\left( d\right) }$ 值最高的对象存在平局，则任意选择）。将这个前 $k$ 个对象的列表记为 ${T}_{k}^{\left( d\right) }$。设 ${M}_{k}^{\left( d\right) }$ 为 ${T}_{k}^{\left( d\right) }$ 中第 $k$ 大的 ${W}^{\left( d\right) }$ 值。

2. Call an object $R$ viable if ${B}^{\left( d\right) }\left( R\right)  > {M}_{k}^{\left( d\right) }$ . Halt when (a) at least $k$ distinct objects have been seen (so that in particular ${T}_{k}^{\left( d\right) }$ contains $k$ objects) and (b) there are no viable objects left outside ${T}_{k}^{\left( d\right) }$ ,that is,when ${B}^{\left( d\right) }\left( R\right)  \leq  {M}_{k}^{\left( d\right) }$ for all $R \notin  {T}_{k}^{\left( d\right) }$ . Return the objects in ${T}_{k}^{\left( d\right) }$ .

2. 如果 ${B}^{\left( d\right) }\left( R\right)  > {M}_{k}^{\left( d\right) }$，则称对象 $R$ 是可行的。当 (a) 至少已经看到 $k$ 个不同的对象（这样特别地，${T}_{k}^{\left( d\right) }$ 包含 $k$ 个对象）并且 (b) ${T}_{k}^{\left( d\right) }$ 之外没有可行的对象了，即对于所有的 $R \notin  {T}_{k}^{\left( d\right) }$ 都有 ${B}^{\left( d\right) }\left( R\right)  \leq  {M}_{k}^{\left( d\right) }$ 时，停止。返回 ${T}_{k}^{\left( d\right) }$ 中的对象。

We now show that NRA is correct for each monotone aggregation function $t$ .

我们现在证明，对于每个单调聚合函数 $t$，NRA（自然排序算法，Natural Ranking Algorithm）都是正确的。

THEOREM 5.4. If the aggregation function $t$ is monotone, then NRA correctly finds the top $k$ objects.

定理 5.4。如果聚合函数 $t$ 是单调的，那么 NRA 能正确地找出前 $k$ 个对象。

Proof: Assume that NRA halts after $d$ sorted accesses to each list,and that ${T}_{k}^{\left( d\right) } = \left\{  {{R}_{1},{R}_{2},\ldots ,{R}_{k}}\right\}$ . Thus,the objects output by NRA are ${R}_{1},{R}_{2},\ldots ,{R}_{k}$ . Let $R$ be an object not among ${R}_{1},{R}_{2},\ldots ,{R}_{k}$ . We must show that $t\left( R\right)  \leq  t\left( {R}_{i}\right)$ for each $i$ .

证明：假设 NRA 在对每个列表进行 $d$ 次排序访问后停止，并且 ${T}_{k}^{\left( d\right) } = \left\{  {{R}_{1},{R}_{2},\ldots ,{R}_{k}}\right\}$。因此，NRA 输出的对象是 ${R}_{1},{R}_{2},\ldots ,{R}_{k}$。设 $R$ 是一个不在 ${R}_{1},{R}_{2},\ldots ,{R}_{k}$ 中的对象。我们必须证明对于每个 $i$ 都有 $t\left( R\right)  \leq  t\left( {R}_{i}\right)$。

Since the algorithm halts at depth $d$ ,we know that $R$ is nonviable at depth $d$ ,that is, ${B}^{\left( d\right) }\left( R\right)  \leq  {M}_{k}^{\left( d\right) }$ . Now $t\left( R\right)  \leq$ ${B}^{\left( d\right) }\left( R\right)$ (Proposition 5.2). Also for each of the $k$ objects ${R}_{i}$ we have ${M}_{k}^{\left( d\right) } \leq  {W}^{\left( d\right) }\left( {R}_{i}\right)  \leq  t\left( {R}_{i}\right)$ (from Proposition 5.1 and the definition of $\left. {M}_{k}^{\left( d\right) }\right)$ . Combining the inequalities we have shown, we have

由于该算法在深度 $d$ 处停止，我们知道 $R$ 在深度 $d$ 处不可行，即 ${B}^{\left( d\right) }\left( R\right)  \leq  {M}_{k}^{\left( d\right) }$。现在 $t\left( R\right)  \leq$ ${B}^{\left( d\right) }\left( R\right)$（命题 5.2）。此外，对于 $k$ 个对象 ${R}_{i}$ 中的每一个，我们有 ${M}_{k}^{\left( d\right) } \leq  {W}^{\left( d\right) }\left( {R}_{i}\right)  \leq  t\left( {R}_{i}\right)$（根据命题 5.1 和 $\left. {M}_{k}^{\left( d\right) }\right)$ 的定义）。结合我们已证明的不等式，我们得到

$$
t\left( R\right)  \leq  {B}^{\left( d\right) }\left( R\right)  \leq  {M}_{k}^{\left( d\right) } \leq  {W}^{\left( d\right) }\left( {R}_{i}\right)  \leq  t\left( {R}_{i}\right) 
$$

for each $i$ ,as desired.

对于每个 $i$，正如我们所期望的。

Note that the tie-breaking mechanism was not significant for correctness. We claim instance optimality of NRA over all algorithms that do not use random access:

注意，平局决胜机制对于正确性而言并不重要。我们声称，在所有不使用随机访问的算法中，NRA（非随机访问算法，Non - Random Access Algorithm）具有实例最优性：

THEOREM 5.5. Assume that the aggregation function $t$ is monotone. Let $\mathbf{D}$ be the class of all databases. Let $\mathbf{A}$ be the class of all algorithms that correctly find the top $k$ objects for t for every database and that do not make random accesses. Then NRA is instance optimal over $\mathbf{A}$ and $\mathbf{D}$ .

定理 5.5。假设聚合函数 $t$ 是单调的。设 $\mathbf{D}$ 为所有数据库的类。设 $\mathbf{A}$ 为所有能为每个数据库正确找出前 $k$ 个对象且不进行随机访问的算法的类。那么，NRA 在 $\mathbf{A}$ 和 $\mathbf{D}$ 上具有实例最优性。

Note that the issue of "wild guesses" is not relevant here, since no algorithm that makes no random access can get any information about an object except via sorted access.

注意，“胡乱猜测”的问题在此处并不相关，因为任何不进行随机访问的算法，除了通过有序访问之外，无法获取关于某个对象的任何信息。

Implementation of NRA: Unfortunately, the execution of NRA may require a lot of bookkeeping at each step, since when NRA does sorted access at depth $t$ (for $1 \leq  t \leq  d$ ),the value of ${B}^{\left( t\right) }\left( R\right)$ must be updated for every object $R$ seen so far. This may be up to ${dm}$ updates for each depth $t$ ,which yields a total of $\Omega \left( {d}^{2}\right)$ updates by depth $d$ . Furthermore, unlike TA, it no longer suffices to have bounded buffers. However, for a specific function like min it is possible that by using appropriate data structures the computation can be greatly simplified. This is an issue for further investigation.

NRA 的实现：不幸的是，NRA 的执行在每一步可能都需要大量的记录工作，因为当 NRA 在深度 $t$ 处进行有序访问时（对于 $1 \leq  t \leq  d$），到目前为止所见到的每个对象 $R$ 的 ${B}^{\left( t\right) }\left( R\right)$ 值都必须更新。对于每个深度 $t$，这可能最多需要 ${dm}$ 次更新，到深度 $d$ 时总共需要 $\Omega \left( {d}^{2}\right)$ 次更新。此外，与 TA（某种算法，具体指代需结合前文）不同，有界缓冲区不再足够。然而，对于像最小值这样的特定函数，通过使用适当的数据结构，计算可能会大大简化。这是一个有待进一步研究的问题。

### 5.2 Taking into Account the Random Access Cost

### 5.2 考虑随机访问成本

We now present the combined algorithm CA that does use random accesses, but takes their cost (relative to sorted access) into account. As before,let ${c}_{S}$ be the cost of a sorted access and ${c}_{R}$ be the cost of a random access. The middleware cost of an algorithm that makes $s$ sorted accesses and $r$ random ones is $s{c}_{S} + r{c}_{R}$ . We know that TA is instance optimal; however, the optimality ratio is a function of the relative cost of a random access to a sorted access, that is ${c}_{R}/{c}_{S}$ . Our goal in this section is to find an algorithm that is instance optimal and where the optimality ratio is independent of ${c}_{R}/{c}_{S}$ . One can view CA as a merge between TA and NRA. Let $h = \left\lfloor  {{c}_{R}/{c}_{S}}\right\rfloor$ . We assume in this section that ${c}_{R} \geq  {c}_{S}$ ,so that $h \geq  1$ . The idea of CA is to run NRA,but every $h$ steps to run a random access phase and update the information (the upper and lower bounds $B$ and $W)$ accordingly. As in Section 5.1,in this section we require only that the output consist of the top $k$ objects,without their grades. If we wish to obtain the grades, this requires only a constant number of additional random accesses, and so has no effect on instance optimality.

我们现在介绍结合算法 CA（Combined Algorithm），它确实使用随机访问，但会考虑其成本（相对于有序访问）。和之前一样，设 ${c}_{S}$ 为一次有序访问的成本，${c}_{R}$ 为一次随机访问的成本。一个进行了 $s$ 次有序访问和 $r$ 次随机访问的算法的中间件成本为 $s{c}_{S} + r{c}_{R}$。我们知道 TA 具有实例最优性；然而，最优比率是随机访问相对于有序访问成本的函数，即 ${c}_{R}/{c}_{S}$。本节的目标是找到一个具有实例最优性且最优比率与 ${c}_{R}/{c}_{S}$ 无关的算法。可以将 CA 视为 TA 和 NRA 的合并。设 $h = \left\lfloor  {{c}_{R}/{c}_{S}}\right\rfloor$。在本节中，我们假设 ${c}_{R} \geq  {c}_{S}$，使得 $h \geq  1$。CA 的思路是运行 NRA，但每 $h$ 步运行一个随机访问阶段并相应地更新信息（上下界 $B$ 和 $W)$）。和 5.1 节一样，在本节中，我们只要求输出包含前 $k$ 个对象，而不要求输出它们的等级。如果我们希望获取等级，这只需要额外进行常数次数的随机访问，因此对实例最优性没有影响。

The algorithm CA is as follows.

算法 CA 如下。

1. Do sorted access in parallel to each of the $m$ sorted lists ${L}_{i}$ . At each depth $d$ (when $d$ objects have been accessed under sorted access in each list) maintain the following:

1. 并行地对每个 $m$ 排序列表 ${L}_{i}$ 进行有序访问。在每个深度 $d$（当在每个列表中通过有序访问访问了 $d$ 个对象时），维护以下信息：

- The bottom values ${\underline{x}}_{1}^{\left( d\right) },{\underline{x}}_{2}^{\left( d\right) },\ldots ,{\underline{x}}_{m}^{\left( d\right) }$ encountered in the lists.

- 列表中遇到的底部值 ${\underline{x}}_{1}^{\left( d\right) },{\underline{x}}_{2}^{\left( d\right) },\ldots ,{\underline{x}}_{m}^{\left( d\right) }$。

- For every object $R$ with discovered fields $S =$ ${S}^{\left( d\right) }\left( R\right)  \subseteq  \{ 1,\ldots ,m\}$ the values ${W}^{\left( d\right) }\left( R\right)  = {W}_{S}\left( R\right)$ and ${B}^{\left( d\right) }\left( R\right)  = {B}_{S}\left( R\right)$ .

- 对于每个具有已发现字段 $S =$ ${S}^{\left( d\right) }\left( R\right)  \subseteq  \{ 1,\ldots ,m\}$ 的对象 $R$，记录值 ${W}^{\left( d\right) }\left( R\right)  = {W}_{S}\left( R\right)$ 和 ${B}^{\left( d\right) }\left( R\right)  = {B}_{S}\left( R\right)$。

- The $k$ objects with the largest ${W}^{\left( d\right) }$ values seen so far (and their grades); if two objects have the same ${W}^{\left( d\right) }$ value,then ties are broken using the ${B}^{\left( d\right) }$ values,such that the object with the highest ${B}^{\left( d\right) }$ value wins (and arbitrarily if there is a tie for the highest ${B}^{\left( d\right) }$ value). Denote this top $k$ list by ${T}_{k}^{\left( d\right) }$ . Let ${M}_{k}^{\left( d\right) }$ be the $k$ th largest ${W}^{\left( d\right) }$ value in ${T}_{k}^{\left( d\right) }$ .

- 到目前为止所见到的具有最大 ${W}^{\left( d\right) }$ 值的 $k$ 个对象（及其等级）；如果两个对象具有相同的 ${W}^{\left( d\right) }$ 值，则使用 ${B}^{\left( d\right) }$ 值来打破平局，使得具有最高 ${B}^{\left( d\right) }$ 值的对象胜出（如果最高 ${B}^{\left( d\right) }$ 值出现平局，则任意选择）。用 ${T}_{k}^{\left( d\right) }$ 表示这个顶部 $k$ 列表。设 ${M}_{k}^{\left( d\right) }$ 为 ${T}_{k}^{\left( d\right) }$ 中第 $k$ 大的 ${W}^{\left( d\right) }$ 值。

2. Call an object $R$ viable if ${B}^{\left( d\right) }\left( R\right)  > {M}_{k}^{\left( d\right) }$ . Every $h$ steps (that is,every time the depth of sorted access increases by $h$ ),do the following: pick the viable object $R$ whose ${B}^{\left( d\right) }$ value is the maximum and for which not all fields are known. Perform random accesses for all the (at most $m - 1$ ) missing fields.

2. 如果 ${B}^{\left( d\right) }\left( R\right)  > {M}_{k}^{\left( d\right) }$，则称对象 $R$ 为可行对象。每 $h$ 步（即每次有序访问的深度增加 $h$ 时），执行以下操作：选择 ${B}^{\left( d\right) }$ 值最大且并非所有字段都已知的可行对象 $R$。对所有（最多 $m - 1$ 个）缺失字段进行随机访问。

3. Halt when (a) at least $k$ distinct objects have been seen (so that in particular ${T}_{k}^{\left( d\right) }$ contains $k$ objects) and (b) there are no viable objects left outside ${T}_{k}^{\left( d\right) }$ ,that is, when ${B}^{\left( d\right) }\left( R\right)  \leq  {M}_{k}^{\left( d\right) }$ for all $R \notin  {T}_{k}^{\left( d\right) }$ . Return the objects in ${T}_{k}^{\left( d\right) }$ .

3. 当 (a) 至少看到了 $k$ 个不同的对象（使得特别是 ${T}_{k}^{\left( d\right) }$ 包含 $k$ 个对象）并且 (b) ${T}_{k}^{\left( d\right) }$ 之外没有剩余的可行对象时停止，即当对于所有 $R \notin  {T}_{k}^{\left( d\right) }$ 都有 ${B}^{\left( d\right) }\left( R\right)  \leq  {M}_{k}^{\left( d\right) }$ 时停止。返回 ${T}_{k}^{\left( d\right) }$ 中的对象。

Note that if $h$ is very large (say larger than the number of objects in the database), then algorithm CA is the same as NRA, since no random access is performed. Similarly, if $h$ is very small,say $h = 1$ ,then algorithm CA is essentially the same as TA, since for each step of doing sorted access in parallel we perform random accesses for all of the missing fields of some object. If instead of performing random accesses for all of the missing fields of some object, we performed random accesses for all of the missing fields of each object seen in sorted access, then the resulting algorithm would be identical to TA. However, for moderate values of $h$ it is not the case that CA is equivalent to the intermittent algorithm that executes $h$ steps of NRA and then one step of TA. In the full paper, we give an example where the intermittent algorithm performs much worse than CA. The difference between the algorithms is that CA picks "wisely" on which objects to perform the random access, namely, according to their ${B}^{\left( d\right) }$ values.

注意，如果 $h$ 非常大（例如大于数据库中对象的数量），那么算法 CA 与 NRA 相同，因为不执行随机访问。类似地，如果 $h$ 非常小，例如 $h = 1$，那么算法 CA 本质上与 TA 相同，因为对于每次并行进行有序访问的步骤，我们都会对某个对象的所有缺失字段进行随机访问。如果我们不是对某个对象的所有缺失字段进行随机访问，而是对有序访问中看到的每个对象的所有缺失字段进行随机访问，那么得到的算法将与 TA 相同。然而，对于适中的 $h$ 值，CA 并不等同于执行 $h$ 步 NRA 然后执行一步 TA 的间歇算法。在完整论文中，我们给出了一个间歇算法的性能比 CA 差很多的例子。这些算法之间的区别在于，CA 会“明智地”选择对哪些对象执行随机访问，即根据它们的 ${B}^{\left( d\right) }$ 值。

Correctness of CA is essentially the same as for NRA, since the same upper and lower bounds are maintained:

CA 的正确性本质上与 NRA 相同，因为维护了相同的上下界：

THEOREM 5.6. If the aggregation function $t$ is monotone, then ${CA}$ correctly finds the top $k$ objects.

定理5.6。如果聚合函数$t$是单调的，那么${CA}$能正确找出前$k$个对象。

In the next section, we consider scenarios under which CA is instance optimal, with the optimality ratio independent of ${c}_{R}/{c}_{S}$ .

在下一节中，我们将考虑CA是实例最优的场景，其最优比率与${c}_{R}/{c}_{S}$无关。

### 5.3 Instance Optimality of CA: Positive and Negative Results

### 5.3 CA的实例最优性：正面和负面结果

In Section 4, we gave two scenarios under which TA is instance optimal over $\mathbf{A}$ and $\mathbf{D}$ . In the first scenario (from Theorem 4.3),(1) the aggregation function $t$ is monotone; (2) $\mathbf{D}$ is the class of all databases; and (c) $\mathbf{A}$ is the class of all algorithms that correctly find the top $k$ objects for $t$ for every database and that do not make wild guesses. In the second scenario (from Theorem 4.6), (1) the aggregation function $t$ is strictly monotone; (2) $\mathbf{D}$ is the class of all databases that satisfy the uniqueness property; and (3) $\mathbf{A}$ is the class of all algorithms that correctly find the top $k$ objects for $t$ for every database in $\mathbf{D}$ . We might hope that under either of these two scenarios, $\mathrm{{CA}}$ is instance optimal, with optimality ratio independent of ${c}_{R}/{c}_{S}$ . Unfortunately, this hope is false, in both scenarios. In fact, we shall give theorems that say that not only does CA fail to fulfill this hope, but so does every algorithm! In other words, neither of these scenarios is enough to guarantee the existence of an algorithm that is instance optimal, with optimality ratio independent of ${c}_{R}/{c}_{S}$ .

在第4节中，我们给出了两种场景，在这两种场景下TA相对于$\mathbf{A}$和$\mathbf{D}$是实例最优的。在第一种场景（来自定理4.3）中，(1) 聚合函数$t$是单调的；(2) $\mathbf{D}$是所有数据库的类；并且(c) $\mathbf{A}$是所有能为每个数据库正确找出$t$的前$k$个对象且不做随意猜测的算法的类。在第二种场景（来自定理4.6）中，(1) 聚合函数$t$是严格单调的；(2) $\mathbf{D}$是所有满足唯一性属性的数据库的类；并且(3) $\mathbf{A}$是所有能为$\mathbf{D}$中的每个数据库正确找出$t$的前$k$个对象的算法的类。我们可能希望在这两种场景中的任何一种下，$\mathrm{{CA}}$都是实例最优的，其最优比率与${c}_{R}/{c}_{S}$无关。不幸的是，在这两种场景下，这个希望都是错误的。事实上，我们将给出定理表明，不仅CA无法实现这个希望，而且所有算法都无法实现！换句话说，这两种场景都不足以保证存在一个实例最优且最优比率与${c}_{R}/{c}_{S}$无关的算法。

However, we shall see that by slightly strengthening the assumption on $t$ in the second scenario,CA becomes instance optimal,with optimality ratio independent of ${c}_{R}/{c}_{S}$ . Let us say that the aggregation function $t$ is strictly monotone in each argument if whenever one argument is strictly increased and the remaining arguments are held fixed, then the value of the aggregation function is strictly increased. That is, $t$ is strictly monotone in each argument if ${x}_{i} < {x}_{i}^{\prime }$ implies that

然而，我们将看到，通过稍微加强第二种场景中对$t$的假设，CA将成为实例最优的，其最优比率与${c}_{R}/{c}_{S}$无关。我们说，如果每当一个参数严格增加而其余参数保持固定时，聚合函数的值严格增加，那么聚合函数$t$在每个参数上都是严格单调的。也就是说，如果${x}_{i} < {x}_{i}^{\prime }$意味着

$$
t\left( {{x}_{1},\ldots ,{x}_{i - 1},{x}_{i},{x}_{i + 1},\ldots ,{x}_{m}}\right) 
$$

$$
 < t\left( {{x}_{1},\ldots ,{x}_{i - 1},{x}_{i}^{\prime },{x}_{i + 1},\ldots ,{x}_{m}}\right) .
$$

The average (or sum) is strictly monotone in each argument, whereas min is not.

平均值（或总和）在每个参数上都是严格单调的，而最小值函数则不是。

We shall see (Section 5.4) that in the second scenario above,if we replace "The aggregation function $t$ is strictly monotone" by "The aggregation function $t$ is strictly monotone in each argument", then CA is instance optimal, with optimality ratio independent of ${c}_{R}/{c}_{S}$ . We shall also see that the same result holds if instead,we simply take $t$ to be min, even though min is not strictly monotone in each argument.

我们将看到（第5.4节），在上述第二种场景中，如果我们将“聚合函数$t$是严格单调的”替换为“聚合函数$t$在每个参数上都是严格单调的”，那么CA是实例最优的，其最优比率与${c}_{R}/{c}_{S}$无关。我们还将看到，如果我们简单地将$t$设为最小值函数，即使最小值函数在每个参数上不是严格单调的，同样的结果仍然成立。

### 5.4 Positive Results about CA

### 5.4 关于CA的正面结果

The next theorem says that in the second scenario above, if we replace "The aggregation function $t$ is strictly monotone" by "The aggregation function $t$ is strictly monotone in each argument", then CA is instance optimal, with optimality ratio independent of ${c}_{R}/{c}_{S}$ .

下一个定理表明，在上述第二种场景中，如果我们将“聚合函数$t$是严格单调的”替换为“聚合函数$t$在每个参数上都是严格单调的”，那么CA是实例最优的，其最优比率与${c}_{R}/{c}_{S}$无关。

THEOREM 5.7. Assume that the aggregation function $t$ is strictly monotone in each argument. Let $\mathbf{D}$ be the class of all databases with the uniqueness property. Let $\mathbf{A}$ be the class of all algorithms that correctly find the top $k$ objects for $t$ for every database in $\mathbf{D}$ . Then ${CA}$ is instance optimal over $\mathbf{A}$ and $\mathbf{D}$ ,with optimality ratio independent of ${c}_{R}/{c}_{S}$ .

定理5.7。假设聚合函数$t$在每个参数上都是严格单调的。设$\mathbf{D}$是所有具有唯一性属性的数据库的类。设$\mathbf{A}$是所有能为$\mathbf{D}$中的每个数据库正确找出$t$的前$k$个对象的算法的类。那么${CA}$相对于$\mathbf{A}$和$\mathbf{D}$是实例最优的，其最优比率与${c}_{R}/{c}_{S}$无关。

The next theorem says that for the function min (which is not strictly monotone in each argument), algorithm CA is still instance optimal.

下一个定理表明，对于最小值函数（它在每个参数上不是严格单调的），算法CA仍然是实例最优的。

THEOREM 5.8. Let $\mathbf{D}$ be the class of all databases with the uniqueness property. Let $\mathbf{A}$ be the class of all algorithms that correctly find the top $k$ objects when the aggregation function is min for every database in $\mathbf{D}$ . Then ${CA}$ is instance optimal over $\mathbf{A}$ and $\mathbf{D}$ ,with optimality ratio independent of ${c}_{R}/{c}_{S}$ .

定理5.8。设$\mathbf{D}$为所有具有唯一性属性的数据库的类。设$\mathbf{A}$为所有在聚合函数为最小值时，能为$\mathbf{D}$中的每个数据库正确找出前$k$个对象的算法的类。那么${CA}$在$\mathbf{A}$和$\mathbf{D}$上是实例最优的，且最优比与${c}_{R}/{c}_{S}$无关。

### 5.5 Negative Results about CA

### 5.5 关于CA的负面结果

In this section, we see that even under the scenarios of Theorems 4.3 and 4.6, there is no algorithm that is instance optimal,with optimality ratio independent of ${c}_{R}/{c}_{S}$ .

在本节中，我们发现，即使在定理4.3和4.6的场景下，也不存在最优比与${c}_{R}/{c}_{S}$无关的实例最优算法。

We begin with a theorem that says that the conditions of Theorem 4.3 (i.e., not allowing wild guesses) are not sufficient to guarantee the existence of an instance optimal algorithm with optimality ratio independent of ${c}_{R}/{c}_{S}$ ,even when the aggregation function is $\min$ ,and when $k = 1$ (so that we are interested only in the top object).

我们从一个定理开始，该定理表明，定理4.3的条件（即不允许随意猜测）不足以保证存在一个最优比与${c}_{R}/{c}_{S}$无关的实例最优算法，即使聚合函数为$\min$，且当$k = 1$时（即我们只关注排名最高的对象）。

THEOREM 5.9. Let $\mathbf{D}$ be the class of all databases. Let A be the class of all algorithms that correctly find the top object for $\min$ for every database and that do not make wild guesses. There is no deterministic algorithm (or even probabilistic algorithm that never makes a mistake) that is instance optimal over $\mathbf{A}$ and $\mathbf{D}$ ,where the optimality ratio is independent of ${c}_{R}/{c}_{S}$ .

定理5.9。设$\mathbf{D}$为所有数据库的类。设A为所有能为每个数据库针对$\min$正确找到排名最高的对象且不进行随意猜测的算法的类。不存在在$\mathbf{A}$和$\mathbf{D}$上实例最优的确定性算法（甚至是从不犯错的概率算法），其中最优比与${c}_{R}/{c}_{S}$无关。

We now give a theorem that says that the conditions of Theorem 4.6 (i.e., strict monotonicity and the uniqueness property) are not sufficient to guarantee the existence of an instance optimal algorithm with optimality ratio independent of ${c}_{R}/{c}_{S}$ ,even when $k = 1$ (so that we are interested only in the top object). In this counterexample, we take the aggregation function $t$ to be given by $t\left( {{x}_{1},{x}_{2},{x}_{3}}\right)  =$ $\min \left( {{x}_{1} + {x}_{2},{x}_{3}}\right)$ . Note that $t$ is strictly monotone,although it is not strictly monotone in each argument. This shows that in Theorem 5.7,we needed to assume that $t$ is strictly monotone in each argument, rather than simply assuming that $t$ is strictly monotone.

现在我们给出一个定理，该定理表明，定理4.6的条件（即严格单调性和唯一性属性）不足以保证存在一个最优比与${c}_{R}/{c}_{S}$无关的实例最优算法，即使当$k = 1$时（即我们只关注排名最高的对象）。在这个反例中，我们取聚合函数$t$为$t\left( {{x}_{1},{x}_{2},{x}_{3}}\right)  =$ $\min \left( {{x}_{1} + {x}_{2},{x}_{3}}\right)$。注意，$t$是严格单调的，尽管它在每个参数上并非严格单调。这表明在定理5.7中，我们需要假设$t$在每个参数上都是严格单调的，而不是简单地假设$t$是严格单调的。

THEOREM 5.10. Let the aggregation function $t$ be given by $t\left( {{x}_{1},{x}_{2},{x}_{3}}\right)  = \min \left( {{x}_{1} + {x}_{2},{x}_{3}}\right)$ . Let $\mathbf{D}$ be the class of all databases that satisfy the uniqueness property. Let $\mathbf{A}$ be the class of all algorithms that correctly find the top object for $t$ for every database in $\mathbf{D}$ . There is no deterministic algorithm (or even probabilistic algorithm that never makes a mistake) that is instance optimal over $\mathbf{A}$ and $\mathbf{D}$ ,where the optimality ratio is independent of ${c}_{R}/{c}_{S}$ .

定理5.10。设聚合函数$t$为$t\left( {{x}_{1},{x}_{2},{x}_{3}}\right)  = \min \left( {{x}_{1} + {x}_{2},{x}_{3}}\right)$。设$\mathbf{D}$为所有满足唯一性属性的数据库的类。设$\mathbf{A}$为所有能为$\mathbf{D}$中的每个数据库针对$t$正确找到排名最高的对象的算法的类。不存在在$\mathbf{A}$和$\mathbf{D}$上实例最优的确定性算法（甚至是从不犯错的概率算法），其中最优比与${c}_{R}/{c}_{S}$无关。

## 6. RELATED WORK

## 6. 相关工作

Nepal and Ramakrishna [9] define an algorithm that is equivalent to TA. Their notion of optimality is weaker than ours. Further, they make an assumption that is essentially equivalent to the aggregation function being the min. ${}^{10}$

尼泊尔（Nepal）和拉马克里希纳（Ramakrishna）[9]定义了一个与TA等价的算法。他们的最优性概念比我们的弱。此外，他们做了一个本质上等同于聚合函数为最小值的假设。${}^{10}$

Güntzer, Balke, and Kiessling [5] also define an algorithm that is equivalent to TA. They call this algorithm "Quick-Combine (basic version)" to distinguish it from their algorithm of interest, which they call "Quick-Combine". The difference between these two algorithms is that Quick-Combine provides a heuristic rule that determines which sorted list ${L}_{i}$ to do the next sorted access on. The intuitive idea is that they wish to speed up TA by taking advantage of skewed distributions of grades. ${}^{11}$ They make no claims of optimality. Instead, they do extensive simulations to compare Quick-Combine against FA (but they do not compare Quick-Combine against TA).

京策尔（Güntzer）、巴尔克（Balke）和基斯林（Kiessling）[5]也定义了一个与TA等价的算法。他们将这个算法称为“快速合并（基本版本）”，以区别于他们感兴趣的算法，即“快速合并”。这两种算法的区别在于，快速合并提供了一个启发式规则，用于确定接下来对哪个排序列表${L}_{i}$进行排序访问。直观的想法是，他们希望通过利用成绩的偏态分布来加速TA。${}^{11}$他们没有声称具有最优性。相反，他们进行了大量的模拟，以将快速合并与FA进行比较（但他们没有将快速合并与TA进行比较）。

We feel that it is an interesting problem to find good heuristics as to which list should be accessed next under sorted access. Such heuristics can potentially lead to some speedup of TA (but the number of sorted accesses can decrease by a factor of at most $m$ ,the number of lists). Unfortunately, there are several problems with the heuristic used by Quick-Combine. The first problem is that it involves a partial derivative, which is not defined for certain aggregation functions (such as min). Even more seriously, it is easy to find a family of examples that shows that as a result of using the heuristic, Quick-Combine is not instance optimal. We note that heuristics that modify TA by deciding which list should be accessed next under sorted access can be forced to be instance optimal simply by insuring that each list is accessed under sorted access at least every $u$ steps,for some constant $u$ .

我们认为，找到在排序访问下接下来应该访问哪个列表的良好启发式方法是一个有趣的问题。这样的启发式方法有可能使TA的速度有所提升（但排序访问的次数最多可减少$m$倍，即列表的数量）。不幸的是，快速合并所使用的启发式方法存在几个问题。第一个问题是它涉及偏导数，而对于某些聚合函数（如最小值），偏导数是未定义的。更严重的是，很容易找到一系列示例，表明由于使用了该启发式方法，快速合并并非实例最优的。我们注意到，通过决定在排序访问下接下来应该访问哪个列表来修改TA的启发式方法，只需确保每个列表在排序访问下至少每$u$步被访问一次（对于某个常数$u$），就可以使其成为实例最优的。

---

<!-- Footnote -->

${}^{10}$ The assumption that Nepal and Ramakrishna make is that the aggregation function $t$ satisfies the lower bounding property. This property says that whenever there is some $i$ such that ${x}_{i} \leq  {x}_{j}^{\prime }$ for every $j$ ,then $t\left( {{x}_{1},\ldots ,{x}_{m}}\right)  \leq$ $t\left( {{x}_{1}^{\prime },\ldots ,{x}_{m}^{\prime }}\right)$ . It is not hard to see that if an aggregation function $t$ satisfies the lower bounding property, then $t\left( {{x}_{1},\ldots ,{x}_{m}}\right)  = f\left( {\min \left\{  {{x}_{1},\ldots ,{x}_{m}}\right\}  }\right)$ ,where $f\left( x\right)  =$ $t\left( {x,\ldots ,x}\right)$ . Note in particular that under the natural assumption that $t\left( {x,\ldots ,x}\right)  = x$ ,so that $f\left( x\right)  = x$ ,we have $t\left( {{x}_{1},\ldots ,{x}_{m}}\right)  = \min \left\{  {{x}_{1},\ldots ,{x}_{m}}\right\}  .$

${}^{10}$ 尼泊尔（Nepal）和罗摩克里希纳（Ramakrishna）所做的假设是，聚合函数 $t$ 满足下界属性。该属性表明，只要存在某个 $i$ 使得对于每个 $j$ 都有 ${x}_{i} \leq  {x}_{j}^{\prime }$，那么 $t\left( {{x}_{1},\ldots ,{x}_{m}}\right)  \leq$ $t\left( {{x}_{1}^{\prime },\ldots ,{x}_{m}^{\prime }}\right)$。不难看出，如果聚合函数 $t$ 满足下界属性，那么 $t\left( {{x}_{1},\ldots ,{x}_{m}}\right)  = f\left( {\min \left\{  {{x}_{1},\ldots ,{x}_{m}}\right\}  }\right)$，其中 $f\left( x\right)  =$ $t\left( {x,\ldots ,x}\right)$。特别要注意的是，在自然假设 $t\left( {x,\ldots ,x}\right)  = x$ 下，即 $f\left( x\right)  = x$，我们有 $t\left( {{x}_{1},\ldots ,{x}_{m}}\right)  = \min \left\{  {{x}_{1},\ldots ,{x}_{m}}\right\}  .$

${}^{11}$ They make the claim that the optimality results proven in [4] about FA do not hold for a skewed distribution of grades, but only for a uniform distribution. This claim is incorrect: the only probabilistic assumption in [4] is that the orderings given by the sorted lists are probabilistically independent.

${}^{11}$ 他们声称，文献 [4] 中关于 FA（公平分配，Fair Allocation）所证明的最优性结果对于成绩的偏态分布不成立，而仅适用于均匀分布。这一说法是错误的：文献 [4] 中唯一的概率假设是排序列表给出的排序在概率上是相互独立的。

<!-- Footnote -->

---

In another paper, Güntzer, Balke, and Kiessling [6] consider the situation where random accesses are impossible. Once again, they define a basic algorithm, called "Stream-Combine (basic version)" and a modified algorithm ("Stream-Combine") that incorporates a heuristic rule that tells which sorted list ${L}_{i}$ to do a sorted access on next. Neither version of Stream-Combine is instance optimal. The reason that the basic version of Stream-Combine is not instance optimal is that it considers only upper bounds on overall grades of objects, unlike our algorithm NRA, which considers both upper and lower bounds. They require that the top $k$ objects be given with their grades (whereas as we discussed, we do not require the grades to be given in the case where random accesses are impossible). Their algorithm cannot say that an object is in the top $k$ unless that object has been seen in every sorted list. Note that there are monotone aggregation functions (such as max, or more interestingly, median) where it is possible to determine the overall grade of an object without knowing its grade in each sorted list.

在另一篇论文中，京策（Güntzer）、巴尔克（Balke）和基斯林（Kiessling）[6] 考虑了无法进行随机访问的情况。他们再次定义了一个基本算法，称为“流合并（基本版本）”（Stream - Combine (basic version)），以及一个改进算法（“流合并”（Stream - Combine）），该改进算法纳入了一条启发式规则，用于指示接下来对哪个排序列表 ${L}_{i}$ 进行排序访问。流合并的两个版本都不是实例最优的。流合并基本版本不是实例最优的原因在于，它仅考虑对象总体成绩的上界，而我们的算法 NRA 则同时考虑上界和下界。他们要求给出前 $k$ 个对象及其成绩（而正如我们所讨论的，在无法进行随机访问的情况下，我们不要求给出成绩）。他们的算法只有在每个排序列表中都看到某个对象时，才能判定该对象在前 $k$ 个对象之中。请注意，存在一些单调聚合函数（如最大值函数，或者更有趣的中位数函数），在不知道对象在每个排序列表中的成绩的情况下，也有可能确定该对象的总体成绩。

## 7. CONCLUSIONS

## 7. 结论

We studied the elegant and remarkably simple algorithm TA, as well as algorithms for the scenario where random access is forbidden or expensive relative to sorted access (NRA and CA). To study these algorithms, we introduced the instance optimality framework in the context of aggregation algorithms, and provided both positive and negative results. This framework is appropriate for analyzing and comparing the performance of algorithms, and provides a very strong notion of optimality. We also considered approximation algorithms, and provided positive and negative results about instance optimality there as well.

我们研究了优雅且极其简单的算法 TA，以及在随机访问被禁止或相对于排序访问成本较高的场景下的算法（NRA 和 CA）。为了研究这些算法，我们在聚合算法的背景下引入了实例最优性框架，并给出了正面和负面的结果。该框架适用于分析和比较算法的性能，并提供了一个非常强的最优性概念。我们还考虑了近似算法，并给出了关于其实例最优性的正面和负面结果。

Two interesting lines of investigation are: (i) finding other scenarios where instance optimality can yield meaningful results, and (ii) finding other applications of our algorithms, such as in information retrieval.

两个有趣的研究方向是：（i）寻找实例最优性能够产生有意义结果的其他场景；（ii）寻找我们算法的其他应用，例如在信息检索领域。

## 8. ACKNOWLEDGMENTS

## 8. 致谢

We are grateful to Michael Franklin for discussions that led to this research, and to Larry Stockmeyer for helpful comments that improved readability.

我们感谢迈克尔·富兰克林（Michael Franklin）的讨论，这些讨论促成了这项研究；也感谢拉里·斯托克迈尔（Larry Stockmeyer）提出的有益意见，这些意见提高了文章的可读性。

## 9. REFERENCES

## 9. 参考文献

[1] D. Aksoy and M. Franklin. RxW: A scheduling approach for large-scale on-demand data broadcast. IEEE/ACM Transactions On Networking, 7(6):846-880, December 1999.

[2] A. Borodin and R. El-Yaniv. Online Computation and Competitive Analysis. Cambridge University Press, New York, 1998.

[3] M. J. Carey, L. M. Haas, P. M. Schwarz, M. Arya, W. F. Cody, R. Fagin, M. Flickner, A. W. Luniewski, W. Niblack, D. Petkovic, J. Thomas, J. H. Williams, and E. L. Wimmers. Towards heterogeneous multimedia information systems: the Garlic approach. In RIDE-DOM '95 (5th Int'l Workshop on Research Issues in Data Engineering: Distributed Object Management), pages 124-131, 1995.

[4] R. Fagin. Combining fuzzy information from multiple systems. J. Comput. System Sci., 58:83-99, 1999.

[5] U. Güntzer, W.-T. Balke, and W. Kiessling. Optimizing multi-feature queries in image databases. In Proc. 26th Very Large Databases (VLDB) Conference, pages 419-428, Cairo, Egypt, 2000.

[6] U. Güntzer, W.-T. Balke, and W. Kiessling. Towards efficient multi-feature queries in heterogeneous environments. In Proc. of the IEEE International Conference on Information Technology: Coding and Computing (ITCC 2001), Las Vegas, USA, April 2001.

[7] D. S. Hochbaum, editor. Approximation Algorithms for NP-Hard Problems. PWS Publishing Company, Boston, MA, 1997.

[8] R. Motwani and P. Raghavan. Randomized Algorithms. Cambridge University Press, Cambridge, U.K., 1995.

[9] S. Nepal and M. V. Ramakrishna. Query processing issues in image (multimedia) databases. In Proc. 15th International Conference on Data Engineering (ICDE), pages 22-29, March 1999.

[10] W. Niblack, R. Barber, W. Equitz, M. Flickner, E. Glasman, D. Petkovic, and P. Yanker. The QBIC project: Querying images by content using color, texture and shape. In SPIE Conference on Storage and Retrieval for Image and Video Databases, volume 1908, pages 173-187, 1993. QBIC Web server is http://wwwqbic.almaden.ibm.com/.

[11] G. Salton. Automatic Text Processing, the Transformation, Analysis and Retrieval of Information by Computer. Addison-Wesley, Reading, MA, 1989.

[12] D. Sleator and R. E. Tarjan. Amortized efficiency of list update and paging rules. Comm. ${ACM}$ , 28: ${202} - {208},{1985}$ .

[13] E. L. Wimmers, L. M. Haas, M. T. Roth, and C. Braendli. Using Fagin's algorithm for merging ranked results in multimedia middleware. In Fourth IFCIS International Conference on Cooperative Information Systems, pages 267-278. IEEE Computer Society Press, September 1999.

[14] L. A. Zadeh. Fuzzy sets. Information and Control, 8:338-353, 1969.

[15] H. J. Zimmermann. Fuzzy Set Theory. Kluwer Academic Publishers, Boston, 3rd edition, 1996.