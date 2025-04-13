# A Survey of Top- $k$ Query Processing Techniques in Relational Database Systems

# 关系数据库系统中前 $k$ 查询处理技术综述

IHAB F. ILYAS, GEORGE BESKALES, and MOHAMED A. SOLIMAN

伊哈布·F·伊利亚斯（Ihab F. Ilyas）、乔治·贝斯凯莱斯（George Beskales）和穆罕默德·A·索利曼（Mohamed A. Soliman）

University of Waterloo

滑铁卢大学

Efficient processing of top- $k$ queries is a crucial requirement in many interactive environments that involve massive amounts of data. In particular,efficient top- $k$ processing in domains such as the Web,multimedia search, and distributed systems has shown a great impact on performance. In this survey, we describe and classify top- $k$ processing techniques in relational databases. We discuss different design dimensions in the current techniques including query models, data access methods, implementation levels, data and query certainty, and supported scoring functions. We show the implications of each dimension on the design of the underlying techniques. We also discuss top- $k$ queries in XML domain,and show their connections to relational approaches.

在涉及海量数据的许多交互式环境中，高效处理前 $k$ 查询是一项关键需求。特别是在网络、多媒体搜索和分布式系统等领域，高效的前 $k$ 处理对性能产生了重大影响。在本综述中，我们描述并分类了关系数据库中的前 $k$ 处理技术。我们讨论了当前技术中的不同设计维度，包括查询模型、数据访问方法、实现级别、数据和查询的确定性以及支持的评分函数。我们展示了每个维度对底层技术设计的影响。我们还讨论了 XML 领域中的前 $k$ 查询，并展示了它们与关系方法的联系。

Categories and Subject Descriptors: H.2.4 [Database Management]: Systems

类别和主题描述符：H.2.4 [数据库管理]：系统

General Terms: Algorithms, Design, Experimentation, Performance

通用术语：算法、设计、实验、性能

Additional Key Words and Phrases: Top- $k$ ,rank-aware processing,rank aggregation,voting

其他关键词和短语：前 $k$、感知排名处理、排名聚合、投票

## ACM Reference Format:

## ACM 引用格式：

Ilyas,I. F.,Beskales,G.,and Soliman,M. A. 2008. A survey of top- $k$ query processing techniques in relational database systems. ACM Comput. Surv. 40, 4, Article 11 (October 2008), 58 pages DOI = 10.1145/1391729.1391730 http://doi.acm.org/10.1145/1391729.1391730

伊利亚斯（Ilyas），I. F.，贝斯凯莱斯（Beskales），G.，和索利曼（Soliman），M. A. 2008 年。关系数据库系统中前 $k$ 查询处理技术综述。《ACM 计算调查》40 卷，第 4 期，文章编号 11（2008 年 10 月），58 页。DOI = 10.1145/1391729.1391730 http://doi.acm.org/10.1145/1391729.1391730

## 1. INTRODUCTION

## 1. 引言

Information systems of different types use various techniques to rank query answers. In many application domains, end-users are more interested in the most important (top- $k$ ) query answers in the potentially huge answer space. Different emerging applications warrant efficient support for top- $k$ queries. For instance,in the context of the Web, the effectiveness and efficiency of metasearch engines, which combine rankings from different search engines, are highly related to efficient rank aggregation methods. Similar applications exist in the context of information retrieval [Salton and McGill 1983] and data mining [Getoor and Diehl 2005]. Most of these applications compute queries that involve joining and aggregating multiple inputs to provide users with the top- $k$ results.

不同类型的信息系统使用各种技术对查询答案进行排名。在许多应用领域中，最终用户对潜在巨大答案空间中最重要的（前 $k$）查询答案更感兴趣。不同的新兴应用需要对前 $k$ 查询提供高效支持。例如，在网络环境中，将不同搜索引擎的排名结果相结合的元搜索引擎的有效性和效率与高效的排名聚合方法高度相关。类似的应用也存在于信息检索 [萨尔通（Salton）和麦吉尔（McGill）1983 年] 和数据挖掘 [格托尔（Getoor）和迪尔（Diehl）2005 年] 领域。这些应用中的大多数计算涉及连接和聚合多个输入的查询，以向用户提供前 $k$ 结果。

---

<!-- Footnote -->

Support was provided in part by the Natural Sciences and Engineering Research Council of Canada through Grant 311671-05.

本研究部分得到了加拿大自然科学与工程研究委员会（Natural Sciences and Engineering Research Council of Canada）通过编号为 311671 - 05 的资助项目的支持。

Authors' Address: University of Waterloo, 200 University Ave. West, Waterloo, ON, Canada N2L 3G1; email: \{ilyas,gbeskale,m2ali\}@uwaterloo.ca.

作者地址：滑铁卢大学，大学西路 200 号，滑铁卢，安大略省，加拿大 N2L 3G1；电子邮件：{ilyas,gbeskale,m2ali}@uwaterloo.ca。

Permission to make digital or hard copies of part or all of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or direct commercial advantage and that copies show this notice on the first page or initial screen of a display along with the full citation. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, to republish, to post on servers, to redistribute to lists, or to use any component of this work in other works requires prior specific permission and/or a fee. Permissions may be requested from Publications Dept., ACM, Inc., 2 Penn Plaza, Suite 701, New York, NY 10121-0701 USA, fax $+ 1\left( {212}\right) {869} - {0481}$ ,or permissions@acm.org.

允许个人或课堂使用本作品的部分或全部内容制作数字或硬拷贝，无需付费，但前提是这些拷贝不得用于盈利或直接商业利益，并且在显示的第一页或初始屏幕上要显示此通知以及完整的引用信息。必须尊重本作品中除 ACM 之外的其他所有者的版权。允许进行带引用的摘要。否则，如需复制、重新发布、在服务器上发布、分发给列表或在其他作品中使用本作品的任何组件，则需要事先获得特定许可和/或支付费用。许可申请可向 ACM 公司出版部提出，地址为美国纽约州纽约市宾夕法尼亚广场 2 号 701 室，邮编 10121 - 0701，传真 $+ 1\left( {212}\right) {869} - {0481}$，或发送电子邮件至 permissions@acm.org。

©2008 ACM 0360-0300/2008/10-ART11 \$5.00. DOI 10.1145/1391729.1391730 http://doi.acm.org/10.1145/ 1391729.1391730

©2008 ACM 0360 - 0300/2008/10 - ART11 5 美元。DOI 10.1145/1391729.1391730 http://doi.acm.org/10.1145/ 1391729.1391730

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: HID Location Price SID Location Tuition HID SID Price + 10 x Tuition 1 3 150000 3500 1 152000 6000 2 145000 6200 141000 7000 7900 Join Result Schools Lafayette 90,000 Indianapolis 2 W.Lafayette 110,000 2 W.Lafayette 3 Indianapolis 111,000 Lafayette 118,000 Lafayette Lafayette 125,000 5 Indianapolis Kokomo 154,000 Indianapolis 7 Kokomo ...... Kokomo Houses -->

<img src="https://cdn.noedgeai.com/0195c915-c519-7d98-b2ff-761eddc2c676_1.jpg?x=329&y=293&w=1094&h=459&r=0"/>

Fig. 1. A top- $k$ query example.

图 1. 一个前 $k$ 查询示例。

<!-- Media -->

One common way to identify the top- $k$ objects is scoring all objects based on some scoring function. An object score acts as a valuation for that object according to its characteristics (e.g., price and size of house objects in a real estate database, or color and texture of images in a multimedia database). Data objects are usually evaluated by multiple scoring predicates that contribute to the total object score. A scoring function is therefore usually defined as an aggregation over partial scores.

识别前 $k$ 个对象的一种常用方法是根据某种评分函数对所有对象进行评分。对象得分根据对象的特征（例如，房地产数据库中房屋对象的价格和面积，或多媒体数据库中图像的颜色和纹理）对该对象进行评估。数据对象通常由多个评分谓词进行评估，这些谓词共同构成对象的总得分。因此，评分函数通常被定义为对部分得分的聚合。

Top- $k$ processing connects to many database research areas including query optimization, indexing methods, and query languages. As a consequence, the impact of efficient top- $k$ processing is becoming evident in an increasing number of applications. The following examples illustrate real-world scenarios where efficient top- $k$ processing is crucial. The examples highlight the importance of adopting efficient top- $k$ processing techniques in traditional database environments.

前 $k$ 处理与许多数据库研究领域相关，包括查询优化、索引方法和查询语言。因此，高效的前 $k$ 处理在越来越多的应用中的影响日益明显。以下示例说明了在现实场景中高效的前 $k$ 处理至关重要的情况。这些示例凸显了在传统数据库环境中采用高效的前 $k$ 处理技术的重要性。

Example 1.1. Consider a user interested in finding a location (e.g., city) where the combined cost of buying a house and paying school tuition for 10 years at that location is minimum. The user is interested in the five least expensive places. Assume that there are two external sources (databases), Houses and Schools, that can provide information on houses and schools, respectively. The Houses database provides a ranked list of the cheapest houses and their locations. Similarly, the Schools database provides a ranked list of the least expensive schools and their locations. Figure 1 gives an example of the Houses and Schools databases.

示例 1.1。假设有一个用户想找到一个地点（例如城市），在该地点购买房屋和支付 10 年学费的总费用最低。用户对五个最便宜的地方感兴趣。假设存在两个外部数据源（数据库），即房屋数据库（Houses）和学校数据库（Schools），它们分别可以提供有关房屋和学校的信息。房屋数据库提供了最便宜房屋及其位置的排名列表。同样，学校数据库提供了最便宜学校及其位置的排名列表。图 1 给出了房屋数据库和学校数据库的一个示例。

A naïve way to answer the query described in Example 1.1 is to retrieve two lists: a list of the cheapest houses from Houses, and a list of the cheapest schools from Schools. These two lists are then joined based on location such that a valid join result is comprised of a house and a school at the same location. For all join results, the total cost of each house-school pair is computed, for example, by adding the house price and the school tuition for 10 years. The five cheapest pairs constitute the final answer to this query. Figure 1 shows an illustration for the join process between houses and schools lists, and partial join results. Note that the top five results cannot be returned to the user until all the join results are generated. For large numbers of colocated houses and schools, the processing of such a query, in the traditional manner, is very expensive as it requires expensive join and sort operations for large amounts of data.

回答示例 1.1 中所述查询的一种简单方法是检索两个列表：一个是从房屋数据库中获取的最便宜房屋列表，另一个是从学校数据库中获取的最便宜学校列表。然后根据位置对这两个列表进行连接，使得有效的连接结果由位于同一位置的房屋和学校组成。对于所有连接结果，计算每对房屋 - 学校的总费用，例如，将房屋价格和 10 年的学校学费相加。五个最便宜的配对构成了该查询的最终答案。图 1 展示了房屋列表和学校列表之间的连接过程以及部分连接结果。请注意，在生成所有连接结果之前，无法将前五个结果返回给用户。对于大量位于同一地点的房屋和学校，以传统方式处理此类查询的成本非常高，因为它需要对大量数据进行昂贵的连接和排序操作。

Example 1.2. Consider a video database system where several visual features are extracted from each video object (frame or segment). Example features include color histograms, color layout, texture, and edge orientation. Features are stored in separate relations indexed using high-dimensional indexes that support similarity queries. Suppose that a user is interested in the top 10 video frames most similar to a given query image based on a set of visual features.

示例 1.2。考虑一个视频数据库系统，其中从每个视频对象（帧或片段）中提取了多个视觉特征。示例特征包括颜色直方图、颜色布局、纹理和边缘方向。这些特征存储在使用支持相似性查询的高维索引进行索引的单独关系中。假设用户对基于一组视觉特征与给定查询图像最相似的前 10 个视频帧感兴趣。

<!-- Media -->

<!-- figureText: Query _____ Color Histogram Edge Histogram Texture Video Color Histogram Database Edge Histogram Texture -->

<img src="https://cdn.noedgeai.com/0195c915-c519-7d98-b2ff-761eddc2c676_2.jpg?x=446&y=282&w=834&h=443&r=0"/>

Fig. 2. Single and multifeature queries in video database.

图 2. 视频数据库中的单特征和多特征查询。

<!-- Media -->

Example 1.2 draws attention to the importance of efficient top- $k$ processing in similarity queries. In video databases [Aref et al. 2004], hours of video data are stored inside the database producing huge amounts of data. Top- $k$ similarity queries are traditionally answered using high-dimensional indexes built on individual video features, and a nearest-neighbor scan operator on top of these indexes. A database system that supports approximate matching ranks objects depending on how well they match the query example. Figure 2 presents an example of single-feature similarity query based on color histogram, texture, and edge orientation. More useful similarity queries could involve multiple features. For example, suppose that a user is interested in the top 10 video frames most similar to a given query image based on color and texture combined. The user could provide a function that combines similarity scores in both features into an overall similarity score. For example,the global score of a frame $f$ with respect to a query image $q$ could be computed as ${0.5} \times$ ColorSimilarity $\left( {f,q}\right)  + {0.5} \times$ TextureSimilarity (f,q).

示例 1.2 凸显了在相似性查询中高效的前 $k$ 处理的重要性。在视频数据库 [Aref 等人，2004] 中，数据库中存储了数小时的视频数据，产生了大量的数据。传统上，前 $k$ 相似性查询是使用基于单个视频特征构建的高维索引以及在这些索引之上的最近邻扫描运算符来回答的。支持近似匹配的数据库系统根据对象与查询示例的匹配程度对对象进行排名。图 2 展示了一个基于颜色直方图、纹理和边缘方向的单特征相似性查询示例。更有用的相似性查询可能涉及多个特征。例如，假设用户对基于颜色和纹理组合与给定查询图像最相似的前 10 个视频帧感兴趣。用户可以提供一个函数，将两个特征的相似性得分组合成一个总体相似性得分。例如，帧 $f$ 相对于查询图像 $q$ 的全局得分可以计算为 ${0.5} \times$ 颜色相似度 $\left( {f,q}\right)  + {0.5} \times$ 纹理相似度 (f,q)。

One way to answer such a multifeature query is by sequentially scanning all database objects, computing the score of each object according to each feature, and combining the scores into a total score for each object. This approach suffers from scalability problems with respect to database size and the number of features. An alternative way is to map the query into a join query that joins the output of multiple single-feature queries, and then sorts the joined results based on combined score. This approach also does not scale with respect to both number of features and database size since all join results have to be computed then sorted.

回答此类多特征查询的一种方法是顺序扫描所有数据库对象，根据每个特征计算每个对象的得分，并将这些得分组合成每个对象的总得分。这种方法在数据库大小和特征数量方面存在可扩展性问题。另一种方法是将查询映射为一个连接查询，该查询连接多个单特征查询的输出，然后根据组合得分对连接结果进行排序。这种方法在特征数量和数据库大小方面也无法扩展，因为必须先计算所有连接结果，然后再进行排序。

The main problem with sort-based approaches is that sorting is a blocking operation that requires full computation of the join results. Although the input to the join operation is sorted on individual features, this order is not exploited by conventional join algorithms. Hence,sorting the join results becomes necessary to produce the top- $k$ answers. Embedding rank-awareness in query processing techniques provides a more efficient and scalable solution.

基于排序的方法的主要问题在于，排序是一种阻塞操作，需要对连接结果进行完整计算。尽管连接操作的输入是按各个特征排序的，但传统连接算法并未利用这种顺序。因此，为了生成前 $k$ 个答案，对连接结果进行排序就变得很有必要。在查询处理技术中融入排名感知功能，可提供一种更高效且可扩展的解决方案。

In this survey,we discuss the state-of-the-art top- $k$ query processing techniques in relational database systems. We give a detailed coverage for most of the recently presented techniques focusing primarily on their integration into relational database environments. We also introduce a taxonomy to classify top- $k$ query processing techniques based on multiple design dimensions, described in the following:

在本次综述中，我们将探讨关系数据库系统中最先进的前 $k$ 查询处理技术。我们会详细介绍近期提出的大多数技术，主要关注它们如何集成到关系数据库环境中。我们还将引入一种分类法，根据多个设计维度对前 $k$ 查询处理技术进行分类，具体如下：

<!-- Media -->

Table I. Frequently Used Notations

表 I. 常用符号

<table><tr><td>Notation</td><td>Description</td></tr><tr><td>$m$</td><td>Number of sources (lists)</td></tr><tr><td>${L}_{i}$</td><td>Ranked source (list) number $i$</td></tr><tr><td>$t$ or $o$</td><td>A tuple or object to be scored</td></tr><tr><td>$g$</td><td>A group of tuples based on some grouping attributes</td></tr><tr><td>$F$</td><td>Scoring (ranking) Function</td></tr><tr><td>$F\left( t\right)$ or $F\left( o\right)$</td><td>Score lower bound of $t$ (or $o$ )</td></tr><tr><td>$\overline{F}\left( t\right)$ or $\overline{F}\left( o\right)$</td><td>Score upper bound of $t$ (or $o$ )</td></tr><tr><td>${p}_{i}\left( t\right)$ or ${p}_{i}\left( o\right)$</td><td>The value of scoring predicate ${p}_{i}$ applied to $t$ (or $o$ ); predicate ${p}_{i}$ determines objects order in ${L}_{i}$</td></tr><tr><td>${p}_{i}^{max}$</td><td>The maximum score of predicate ${p}_{i}$</td></tr><tr><td>${p}_{i}^{min}$</td><td>The minimum score of predicate ${p}_{i}$</td></tr><tr><td>${p}_{i}$</td><td>The score upper bound of predicate ${p}_{i}$ (mostly refers to the score of the last seen object in ${L}_{i}$ )</td></tr><tr><td>$T$</td><td>Score threshold (cutoff value)</td></tr><tr><td>${A}_{k}$</td><td>The current top- $k$ set</td></tr><tr><td>${M}_{k}$</td><td>The minimum score in the current top- $k$ set</td></tr></table>

<table><tbody><tr><td>符号表示</td><td>描述</td></tr><tr><td>$m$</td><td>源（列表）的数量</td></tr><tr><td>${L}_{i}$</td><td>排名后的源（列表）编号 $i$</td></tr><tr><td>$t$ 或 $o$</td><td>待评分的元组或对象</td></tr><tr><td>$g$</td><td>基于某些分组属性的一组元组</td></tr><tr><td>$F$</td><td>评分（排名）函数</td></tr><tr><td>$F\left( t\right)$ 或 $F\left( o\right)$</td><td>$t$（或 $o$）的分数下限</td></tr><tr><td>$\overline{F}\left( t\right)$ 或 $\overline{F}\left( o\right)$</td><td>$t$（或 $o$）的分数上限</td></tr><tr><td>${p}_{i}\left( t\right)$ 或 ${p}_{i}\left( o\right)$</td><td>应用于 $t$（或 $o$）的评分谓词 ${p}_{i}$ 的值；谓词 ${p}_{i}$ 确定 ${L}_{i}$ 中对象的顺序</td></tr><tr><td>${p}_{i}^{max}$</td><td>谓词 ${p}_{i}$ 的最大分数</td></tr><tr><td>${p}_{i}^{min}$</td><td>谓词 ${p}_{i}$ 的最小分数</td></tr><tr><td>${p}_{i}$</td><td>谓词 ${p}_{i}$ 的分数上限（主要指 ${L}_{i}$ 中最后看到的对象的分数）</td></tr><tr><td>$T$</td><td>分数阈值（截断值）</td></tr><tr><td>${A}_{k}$</td><td>当前的前 $k$ 集合</td></tr><tr><td>${M}_{k}$</td><td>当前前 $k$ 集合中的最小分数</td></tr></tbody></table>

<!-- Media -->

-Query model. Top- $k$ processing techniques are classified according to the query model they assume. Some techniques assume a selection query model, where scores are attached directly to base tuples. Other techniques assume a join query model, where scores are computed over join results. A third category assumes an aggregate query model, where we are interested in ranking groups of tuples.

-查询模型。前 $k$ 处理技术根据其所假定的查询模型进行分类。一些技术假定为选择查询模型，在该模型中，分数直接与基本元组相关联。其他技术假定为连接查询模型，在该模型中，分数是在连接结果上计算得出的。第三类假定为聚合查询模型，在该模型中，我们关注的是对元组组进行排名。

-Data access methods. Top- $k$ processing techniques are classified according to the data access methods they assume to be available in the underlying data sources. For example, some techniques assume the availability of random access, while others are restricted to only sorted access.

-数据访问方法。前 $k$ 处理技术根据其所假定的底层数据源中可用的数据访问方法进行分类。例如，一些技术假定可以进行随机访问，而其他技术则仅限于有序访问。

-Implementation level. Top- $k$ processing techniques are classified according to their level of integration with database systems. For example, some techniques are implemented in an application layer on top of the database system, while others are implemented as query operators.

-实现级别。前 $k$ 处理技术根据其与数据库系统的集成级别进行分类。例如，一些技术在数据库系统之上的应用层实现，而其他技术则作为查询运算符实现。

-Data and query uncertainty. Top- $k$ processing techniques are classified based on the uncertainty involved in their data and query models. Some techniques produce exact answers, while others allow for approximate answers, or deal with uncertain data.

-数据和查询的不确定性。前 $k$ 处理技术根据其数据和查询模型中涉及的不确定性进行分类。一些技术产生精确答案，而其他技术允许近似答案，或者处理不确定数据。

- Ranking function. Top- $k$ processing techniques are classified based on the restrictions they impose on the underlying ranking (scoring) function. Most proposed techniques assume monotone scoring functions. Few proposals address general functions.

-排名函数。前 $k$ 处理技术根据其对底层排名（评分）函数施加的限制进行分类。大多数提出的技术假定为单调评分函数。很少有提议涉及通用函数。

### 1.1. Notations

### 1.1. 符号说明

The working environments, of most of the techniques we describe, assume a scoring (ranking) function used to score objects (tuples) by aggregating the values of their partial scores (scoring predicates). Table I lists the frequently used notations in this survey.

我们所描述的大多数技术的工作环境假定使用一个评分（排名）函数，该函数通过聚合对象（元组）的部分分数（评分谓词）的值来对对象进行评分。表 I 列出了本调查中常用的符号。

### 1.2. Outline

### 1.2. 大纲

The remainder of this survey is organized as follows. Section 2 introduces the taxonomy we adopt in this survey to classify top- $k$ query processing methods. Sections 3,4,5, and 6 discuss the different design dimensions of top- $k$ processing techniques,and give the details of multiple techniques in each dimension. Section 7 discusses related top- $k$ processing techniques for XML data. Section 8 presents related background from voting theory,which forms the basis of many current top- $k$ processing techniques. Section 9 concludes this survey, and describes future research directions.

本调查的其余部分组织如下。第 2 节介绍了我们在本调查中采用的用于对前 $k$ 查询处理方法进行分类的分类法。第 3、4、5 和 6 节讨论了前 $k$ 处理技术的不同设计维度，并详细介绍了每个维度中的多种技术。第 7 节讨论了用于 XML 数据的相关前 $k$ 处理技术。第 8 节介绍了来自投票理论的相关背景，这构成了许多当前前 $k$ 处理技术的基础。第 9 节总结了本调查，并描述了未来的研究方向。

<!-- Media -->

<!-- figureText: Top- $k$ Processing Techniques Implementation Level Ranking Function Sorted + Controlled Random Probes Monotone Unspecified Generic Query Engine Application Level Indexes / Materialized Views Filter-Restart Query Model Data & Query Data Access Certainty Top-k Top-k No Random Selection Aggregate Top-k Join Both Sorted and Random Certain Data, Exact Methods Uncertain Data Certain Data, Approximate Methods -->

<img src="https://cdn.noedgeai.com/0195c915-c519-7d98-b2ff-761eddc2c676_4.jpg?x=354&y=280&w=1023&h=654&r=0"/>

Fig. 3. Classification of top- $k$ query processing techniques.

图 3. 前 $k$ 查询处理技术的分类。

<!-- Media -->

We assume the reader of this survey has a general familiarity with relational database concepts.

我们假定本调查的读者对关系数据库概念有一般的了解。

## 2. TAXONOMY OF TOP- $K$ QUERY PROCESSING TECHNIQUES

## 2. 前 $K$ 查询处理技术的分类法

Supporting efficient top- $k$ processing in database systems is a relatively recent and active line of research. Top- $k$ processing has been addressed from different perspectives in the current literature. Figure 3 depicts the classification we adopt in this survey to categorize different top- $k$ processing techniques based on their capabilities and assumptions. In the following sections, we discuss our classification dimensions, and their impact on the design of the underlying top- $k$ processing techniques. For each dimension, we give a detailed description for one or more example techniques.

在数据库系统中支持高效的前 $k$ 处理是一个相对较新且活跃的研究领域。当前文献从不同角度探讨了前 $k$ 处理问题。图 3 展示了我们在本调查中采用的分类方法，用于根据不同前 $k$ 处理技术的能力和假设对其进行分类。在以下各节中，我们将讨论我们的分类维度及其对底层前 $k$ 处理技术设计的影响。对于每个维度，我们将详细描述一种或多种示例技术。

### 2.1. Query Model Dimension

### 2.1. 查询模型维度

Current top- $k$ processing techniques adopt different query models to specify the data objects to be scored. We discuss three different models: (1) top- $k$ selection query, (2) top- $k$ join query,and (3) top- $k$ aggregate query. We formally define these query models in the following.

当前的前 $k$ 处理技术采用不同的查询模型来指定要评分的数据对象。我们讨论三种不同的模型：(1) 前 $k$ 选择查询，(2) 前 $k$ 连接查询，以及 (3) 前 $k$ 聚合查询。我们将在下面正式定义这些查询模型。

2.1.1. Top-k Selection Query Model. In this model, the scores are assumed to be attached to base tuples. A top- $k$ selection query is required to report the $k$ tuples with the highest scores. Scores might not be readily available since they could be the outcome of some user-defined scoring function that aggregates information coming from different tuple attributes.

2.1.1. 前 k 选择查询模型。在该模型中，假定分数与基本元组相关联。前 $k$ 选择查询需要报告得分最高的 $k$ 个元组。分数可能并非随时可用，因为它们可能是某个用户定义的评分函数的结果，该函数聚合了来自不同元组属性的信息。

Definition 2.1 (Top- $k$ Selection Query). Consider a relation $R$ ,where each tuple in $R$ has $n$ attributes. Consider $m$ scoring predicates, ${p}_{1}\cdots {p}_{m}$ defined on these attributes. Let $F\left( t\right)  = F\left( {{p}_{1}\left( t\right) ,\ldots ,{p}_{m}\left( t\right) }\right)$ be the overall score of tuple $t \in  R$ . A top- $k$ selection query selects the $k$ tuples in $R$ with the largest $F$ values.

定义2.1（前 $k$ 选择查询）。考虑一个关系 $R$，其中 $R$ 中的每个元组有 $n$ 个属性。考虑 $m$ 个评分谓词，${p}_{1}\cdots {p}_{m}$ 定义在这些属性上。设 $F\left( t\right)  = F\left( {{p}_{1}\left( t\right) ,\ldots ,{p}_{m}\left( t\right) }\right)$ 为元组 $t \in  R$ 的总体得分。前 $k$ 选择查询会选择 $R$ 中 $F$ 值最大的 $k$ 个元组。

A SQL template for top- $k$ selection query is the following:

前 $k$ 选择查询的SQL模板如下：

SELECT some_attributes

选择某些属性

FROM $R$

从 $R$ 中

WHERE selection_condition

其中选择条件

ORDER BY $F\left( {{p}_{1},\ldots ,{p}_{m}}\right)$

按 $F\left( {{p}_{1},\ldots ,{p}_{m}}\right)$ 排序

LIMIT ${k}^{1}$

限制 ${k}^{1}$ 条

Consider Example 1.2. Assume the user is interested in finding the top-10 video objects that are most similar to a given query image $q$ ,based on color and texture,and whose release date is after $1/1/{2008}$ . This query could be written as follows:

考虑示例1.2。假设用户有兴趣找出与给定查询图像 $q$ 在颜色和纹理上最相似，且发布日期在 $1/1/{2008}$ 之后的前10个视频对象。此查询可以写成如下形式：

SELECT v.id

选择视频对象的ID

FROM VideoObject $v$

从视频对象 $v$ 中

WHERE v.date > '01/01/2008'

其中视频对象的日期 > '2008年1月1日'

ORDER BY ${0.5} *$ Color Similarity $\left( {q,v}\right)  + {0.5} *$ TextureSimilarity(q,v)

按 ${0.5} *$ 颜色相似度 $\left( {q,v}\right)  + {0.5} *$ 纹理相似度(查询图像,视频对象) 排序

LIMIT 10

The NRA algorithm [Fagin et al. 2001] is one example of top- $k$ techniques that adopt the top- $k$ selection model. The input to the NRA algorithm is a set of sorted lists; each ranks the "same" set of objects based on different attributes. The output is a ranked list of these objects ordered on the aggregate input scores. We give the full details of this algorithm in Section 3.2.

NRA算法（法金等人，2001年）是采用前 $k$ 选择模型的前 $k$ 技术的一个示例。NRA算法的输入是一组排序列表；每个列表根据不同属性对“相同”的一组对象进行排名。输出是根据聚合输入得分排序的这些对象的排名列表。我们将在3.2节中给出该算法的详细信息。

2.1.2. Top-k Join Query Model. In this model, scores are assumed to be attached to join results rather than base tuples. A top- $k$ join query joins a set of relations based on some arbitrary join condition, assigns scores to join results based on some scoring function, and reports the top- $k$ join results.

2.1.2. 前k连接查询模型。在这个模型中，假设分数是附加到连接结果上，而不是基本元组上。前 $k$ 连接查询根据某些任意连接条件连接一组关系，根据某个评分函数为连接结果分配分数，并报告前 $k$ 个连接结果。

Definition 2.2 (Top- $k$ Join Query). Consider a set of relations ${R}_{1}\cdots {R}_{n}$ . A top- $k$ join query joins ${R}_{1}\cdots {R}_{n}$ ,and returns the $k$ join results with the largest combined scores. The combined score of each join result is computed according to some function $F\left( {{p}_{1},\ldots ,{p}_{m}}\right)$ ,where ${p}_{1},\ldots ,{p}_{m}$ are scoring predicates defined over the join results.

定义2.2（前 $k$ 连接查询）。考虑一组关系 ${R}_{1}\cdots {R}_{n}$。前 $k$ 连接查询对 ${R}_{1}\cdots {R}_{n}$ 进行连接，并返回组合得分最大的 $k$ 个连接结果。每个连接结果的组合得分根据某个函数 $F\left( {{p}_{1},\ldots ,{p}_{m}}\right)$ 计算，其中 ${p}_{1},\ldots ,{p}_{m}$ 是定义在连接结果上的评分谓词。

A possible SQL template for a top- $k$ join query is

前 $k$ 连接查询的一个可能的SQL模板是

SELECT *

FROM ${R}_{1},\ldots ,{R}_{n}$

从 ${R}_{1},\ldots ,{R}_{n}$

WHERE join_condition $\left( {{R}_{1},\ldots ,{R}_{n}}\right)$

其中连接条件为 $\left( {{R}_{1},\ldots ,{R}_{n}}\right)$

ORDER BY $F\left( {{p}_{1},\ldots ,{p}_{m}}\right)$

按 $F\left( {{p}_{1},\ldots ,{p}_{m}}\right)$ 排序

LIMIT $k$

限制 $k$

For example,the top- $k$ join query in Example 1.1 could be formulated as follows:

例如，示例 1.1 中的前 $k$ 连接查询可以表述如下：

SELECT h.id, s.id

选择 h.id，s.id

FROM House $h,{School}s$

从房屋表 $h,{School}s$

WHERE h.location=s.location

其中 h.location = s.location

ORDER BY h.price + 10 * s.tuition

按 h.price + 10 * s.tuition 排序

LIMIT 5

A top- $k$ selection query can be formulated as a special top- $k$ join query by partitioning $R$ into $n$ vertical relations ${R}_{1},\ldots ,{R}_{n}$ ,such that each relation ${R}_{i}$ has the necessary attributes to compute the score ${p}_{i}$ . For example,Let $R$ contains the attributes ${tid},{A}_{1}$ , ${A}_{2}$ ,and ${A}_{3}$ . Then, $R$ can be partitioned into ${R}_{1} = \left( {{tid},{A}_{1}}\right)$ and ${R}_{2} = \left( {{tid},{A}_{2},{A}_{3}}\right)$ ,where ${p}_{1} = {A}_{1}$ and ${p}_{2} = {A}_{2} + {A}_{3}$ . In this case,the join condition is an equality condition on key attributes. The NRA-RJ algorithm [Ilyas et al. 2002] is one example of top- $k$ processing techniques that formulate top- $k$ selection queries as top- $k$ join queries based on tuples’ keys.

通过将 $R$ 划分为 $n$ 个垂直关系 ${R}_{1},\ldots ,{R}_{n}$，可以将前 $k$ 选择查询表述为一个特殊的前 $k$ 连接查询，使得每个关系 ${R}_{i}$ 都具有计算得分 ${p}_{i}$ 所需的属性。例如，假设 $R$ 包含属性 ${tid},{A}_{1}$、${A}_{2}$ 和 ${A}_{3}$。那么，$R$ 可以划分为 ${R}_{1} = \left( {{tid},{A}_{1}}\right)$ 和 ${R}_{2} = \left( {{tid},{A}_{2},{A}_{3}}\right)$，其中 ${p}_{1} = {A}_{1}$ 和 ${p}_{2} = {A}_{2} + {A}_{3}$。在这种情况下，连接条件是键属性上的相等条件。NRA - RJ 算法（伊利亚斯等人，2002 年）是将前 $k$ 选择查询表述为基于元组键的前 $k$ 连接查询的前 $k$ 处理技术的一个示例。

---

<!-- Footnote -->

${}^{1}$ Other keywords,for example,Stop After $k$ ,are also used in other SQL dialects.

${}^{1}$ 其他关键字，例如，在 $k$ 之后停止，也在其他 SQL 方言中使用。

<!-- Footnote -->

---

Many top- $k$ join techniques address the interaction between computing the join results and producing the top- $k$ answers. Examples are the ${J}^{ * }$ algorithm [Natsev et al. 2001] (Section 3.2), and the Rank-Join algorithm [Ilyas et al. 2004a] (Section 4.2). Some techniques, for example, PREFER [Hristidis et al. 2001] (Section 4.1.2), process top- $k$ join queries using auxiliary structures that materialize join results, or by ranking the join results after they are generated.

许多前 $k$ 连接技术解决了计算连接结果和生成前 $k$ 答案之间的交互问题。例如 ${J}^{ * }$ 算法（纳采夫等人，2001 年）（第 3.2 节）和排名连接算法（伊利亚斯等人，2004a）（第 4.2 节）。一些技术，例如 PREFER（赫里斯蒂迪斯等人，2001 年）（第 4.1.2 节），使用物化连接结果的辅助结构来处理前 $k$ 连接查询，或者在生成连接结果后对其进行排名。

2.1.3. Top-k Aggregate Query Model. In this model, scores are computed for tuple groups,rather than individual tuples. A top- $k$ aggregate query reports the $k$ groups with the largest scores. Group scores are computed using a group aggregate function such as sum.

2.1.3. 前 k 聚合查询模型。在这个模型中，分数是为元组组而不是单个元组计算的。前 $k$ 聚合查询报告得分最高的 $k$ 个组。组分数使用诸如求和之类的组聚合函数计算。

Definition 2.3 (Top-k Aggregate Query). Consider a set of grouping attributes $\mathcal{G} = \left\{  {{g}_{1},\ldots ,{g}_{r}}\right\}$ ,and an aggregate function $F$ that is evaluated on each group. A top- $k$ aggregate query returns the $k$ groups,based on $\mathcal{G}$ ,with the highest $F$ values.

定义 2.3（前 k 聚合查询）。考虑一组分组属性 $\mathcal{G} = \left\{  {{g}_{1},\ldots ,{g}_{r}}\right\}$ 和一个在每个组上求值的聚合函数 $F$。前 $k$ 聚合查询根据 $\mathcal{G}$ 返回具有最高 $F$ 值的 $k$ 个组。

A SQL formulation for a top- $k$ aggregate query is

前 $k$ 聚合查询的 SQL 表述为

SELECT ${g}_{1},\ldots ,{g}_{r},F$

选择 ${g}_{1},\ldots ,{g}_{r},F$

FROM ${R}_{1},\ldots ,{R}_{n}$

从 ${R}_{1},\ldots ,{R}_{n}$

WHERE join_condition $\left( {{R}_{1},\ldots ,{R}_{n}}\right)$

其中连接条件为 $\left( {{R}_{1},\ldots ,{R}_{n}}\right)$

GROUP BY ${g}_{1},\ldots ,{g}_{r}$

按 ${g}_{1},\ldots ,{g}_{r}$ 分组

ORDER BY $F$

按 $F$ 排序

LIMIT $k$

限制 $k$

An example top- $k$ aggregate query is to find the best five areas to advertise student insurance product, based on the score of each area, which is a function of student's income, age, and credit.

一个示例的前 $k$ 聚合查询是根据每个地区的得分（该得分是学生收入、年龄和信用的函数）找出最适合宣传学生保险产品的五个地区。

SELECT zipcode,Average(income*w1 + age*w2 + credit*w3) as score

选择邮政编码，将（收入 * w1 + 年龄 * w2 + 信用 * w3）的平均值作为得分

FROM customer

从客户表

WHERE occupation $=$ ’student’

其中职业 $=$ 为'学生'

GROUP BY zipcode

按邮政编码分组

ORDER BY score

按得分排序

LIMIT 5

Top- $k$ aggregate queries add additional challenges to top- $k$ join queries: (1) interaction of grouping, joining, and scoring of query results, and (2) nontrivial estimation of the scores of candidate top- $k$ groups. A few recent techniques,for example,Li et al. [2006],address these challenges to efficiently compute top- $k$ aggregate queries. We discuss these techniques in Section 4.2.

前 $k$ 聚合查询给前 $k$ 连接查询带来了额外的挑战：（1）查询结果的分组、连接和评分之间的相互作用；（2）对候选前 $k$ 组得分的非平凡估计。最近的一些技术，例如 Li 等人 [2006] 的方法，解决了这些挑战以高效计算前 $k$ 聚合查询。我们将在 4.2 节讨论这些技术。

### 2.2. Data Access Dimension

### 2.2. 数据访问维度

Many top- $k$ processing techniques involve accessing multiple data sources with different valuations of the underlying data objects. A typical example is a metasearcher that aggregates the rankings of search hits produced by different search engines. The hits produced by each search engine can be seen as a ranked list of Web pages based on some score, for example, relevance to query keywords. The manner in which these lists are accessed largely affects the design of the underlying top- $k$ processing techniques. For example, ranked lists could be scanned sequentially in score order. We refer to this access method as sorted access. Sorted access is supported by a DBMS if, for example, a B-Tree index is built on objects' scores. In this case, scanning the sequence set (leaf level) of the B-Tree index provides a sorted access of objects based on their scores. On the other hand, the score of some object might be required directly without traversing the objects with higher/smaller scores. We refer to this access method as random access. Random access could be provided through index lookup operations if an index is built on object keys.

许多前 $k$ 处理技术涉及访问多个对底层数据对象有不同估值的数据源。一个典型的例子是元搜索引擎，它聚合了不同搜索引擎产生的搜索结果排名。每个搜索引擎产生的搜索结果可以看作是基于某种得分（例如与查询关键词的相关性）对网页进行排名的列表。访问这些列表的方式在很大程度上影响了底层前 $k$ 处理技术的设计。例如，可以按得分顺序依次扫描排名列表。我们将这种访问方法称为排序访问。如果数据库管理系统（DBMS）在对象的得分上建立了 B - 树索引，就支持排序访问。在这种情况下，扫描 B - 树索引的序列集（叶节点层）可以根据对象的得分对其进行排序访问。另一方面，可能需要直接获取某个对象的得分而无需遍历得分更高/更低的对象。我们将这种访问方法称为随机访问。如果在对象键上建立了索引，就可以通过索引查找操作实现随机访问。

We classify top- $k$ processing techniques,based on the assumptions they make about available data access methods in the underlying data sources, as follows:

我们根据前 $k$ 处理技术对底层数据源中可用数据访问方法的假设，将其分类如下：

-Both sorted and random access. In this category,top- $k$ processing techniques assume the availability of both sorted and random access in all the underlying data sources. Examples are TA [Fagin et al. 2001], and the Quick-Combine algorithm [Güntzer et al. 2000]. We discuss the details of these techniques in Section 3.1.

- 排序访问和随机访问均可。在这一类中，前 $k$ 处理技术假设所有底层数据源都同时支持排序访问和随机访问。例如 TA [Fagin 等人 2001] 和快速合并算法 [Güntzer 等人 2000]。我们将在 3.1 节详细讨论这些技术。

-No random access. In this category,top- $k$ processing techniques assume the underlying sources provide only sorted access to data objects based on their scores. Examples are the NRA algorithm [Fagin et al. 2001], and the Stream-Combine algorithm [Güntzer et al. 2001]. We discuss the details of these techniques in Section 3.2.

-无随机访问。在这一类别中，前$k$处理技术假定底层数据源仅根据数据对象的得分提供对其的排序访问。示例包括NRA算法（[法金等人，2001年]）和流合并算法（[京策等人，2001年]）。我们将在3.2节讨论这些技术的细节。

-Sorted access with controlled random probes. In this category,top- $k$ processing techniques assume the availability of at least one sorted access source. Random accesses are used in a controlled manner to reveal the overall scores of candidate answers. Examples are the Rank-Join algorithm [Ilyas et al. 2004a], the MPro algorithm [Chang and Hwang 2002], and the Upper and Pick algorithms [Bruno et al. 2002b]. We discuss the details of these techniques in Section 3.3.

-带受控随机探查的排序访问。在这一类别中，前$k$处理技术假定至少有一个排序访问源可用。随机访问以受控方式使用，以揭示候选答案的总体得分。示例包括Rank - Join算法（[伊利亚斯等人，2004a]）、MPro算法（[张和黄，2002年]）以及Upper和Pick算法（[布鲁诺等人，2002b]）。我们将在3.3节讨论这些技术的细节。

### 2.3. Implementation-Level Dimension

### 2.3. 实现层面维度

Integrating top- $k$ processing with database systems is addressed in different ways by current techniques. One approach is to embed top- $k$ processing in an outer layer on top of the database engine. This approach allows for easy extensibility of top- $k$ techniques, since they are decoupled from query engines. The capabilities of database engines (e.g., storage, indexing, and query processing) are leveraged to allow for efficient top- $k$ processing. New data access methods or specialized data structures could also be built to support top- $k$ processing. However,the core of query engines remains unchanged.

当前技术以不同方式解决了将前$k$处理与数据库系统集成的问题。一种方法是将前$k$处理嵌入到数据库引擎之上的外层。这种方法允许前$k$技术易于扩展，因为它们与查询引擎解耦。利用数据库引擎的功能（例如，存储、索引和查询处理）来实现高效的前$k$处理。还可以构建新的数据访问方法或专用数据结构来支持前$k$处理。然而，查询引擎的核心保持不变。

Another approach is to modify the core of query engines to recognize the ranking requirements of top- $k$ queries during query planning and execution. This approach has a direct impact on query processing and optimization. Specifically, query operators are modified to be rank-aware. For example, a join operator is required to produce ranked join results to support pipelining top- $k$ query answers. Moreover,available access methods for ranking predicates are taken into account while optimizing a query plan.

另一种方法是修改查询引擎的核心，以便在查询规划和执行期间识别前$k$查询的排序要求。这种方法对查询处理和优化有直接影响。具体而言，查询操作符被修改为具有排序感知能力。例如，要求连接操作符生成排序的连接结果，以支持前$k$查询答案的流水线处理。此外，在优化查询计划时会考虑可用的排序谓词访问方法。

We classify top- $k$ processing techniques based on their level of integration with database engines as follows:

我们根据前$k$处理技术与数据库引擎的集成级别对其进行如下分类：

-Application level. This category includes top- $k$ processing techniques that work outside the database engine. Some of the techniques in this category rely on the support of specialized top- $k$ indexes or materialized views. However,the main top- $k$ processing remains outside the engine. Examples are Chang et al. [2000], and Hristidis et al. [2001]. Another group of techniques formulate top- $k$ queries as range queries that are repeatedly executed until the top- $k$ objects are obtained. We refer to this group of techniques as filter-restart. One example is Donjerkovic and Ramakrishnan [1999]. We discuss the details of these techniques in Section 4.1. -Query engine level. This category includes techniques that involve modifications to the query engine to allow for rank-aware processing and optimization. Some of these techniques introduce new query operators to support efficient top- $k$ processing. For example, Ilyas et al. [2004a] introduced rank-aware join operators. Other techniques, for example, Li et al. [2005, 2006], extend rank-awareness to query algebra to allow for extensive query optimization. We discuss the details of these techniques in Section 4.2.

-应用程序级别。这一类别包括在数据库引擎之外工作的前$k$处理技术。该类别中的一些技术依赖于专用的前$k$索引或物化视图的支持。然而，主要的前$k$处理仍在引擎之外。示例有[张等人，2000年]和[赫里斯蒂迪斯等人，2001年]。另一组技术将前$k$查询表述为范围查询，反复执行这些查询直到获得前$k$对象。我们将这组技术称为过滤 - 重启。一个示例是[东耶尔科维奇和拉马克里什南，1999年]。我们将在4.1节讨论这些技术的细节。 -查询引擎级别。这一类别包括涉及对查询引擎进行修改以实现排序感知处理和优化的技术。其中一些技术引入了新的查询操作符以支持高效的前$k$处理。例如，[伊利亚斯等人，2004a]引入了排序感知连接操作符。其他技术，例如[李等人，2005年，2006年]，将排序感知扩展到查询代数以实现广泛的查询优化。我们将在4.2节讨论这些技术的细节。

### 2.4. Query and Data Uncertainty Dimension

### 2.4. 查询和数据不确定性维度

In some query processing environments, for example, decision support or OLAP, obtaining exact query answers efficiently may be overwhelming to the database engine because of the interactive nature of such environments, and the sheer amounts of data they usually handle. Such environments could sacrifice the accuracy of query answers in favor of performance. In these settings,it may be acceptable for a top- $k$ query to report approximate answers.

在某些查询处理环境中，例如决策支持或联机分析处理（OLAP），由于此类环境的交互性质以及它们通常处理的数据量巨大，高效获取精确的查询答案可能会使数据库引擎不堪重负。此类环境可以为了性能而牺牲查询答案的准确性。在这些情况下，前$k$查询报告近似答案可能是可以接受的。

The uncertainty in top- $k$ query answers might alternatively arise due to the nature of the underlying data itself. Applications in domains such as sensor networks, data cleaning, and moving objects tracking involve processing data that is probabilistic in nature. For example, the temperature reading of some sensor could be represented as a probability distribution over a continuous interval, or a customer name in a dirty database could be a represented as a set of possible names. In these settings,top- $k$ queries, as well as other query types, need to be formulated and processed while taking data uncertainty into account.

前$k$查询答案的不确定性也可能源于底层数据本身的性质。传感器网络、数据清理和移动对象跟踪等领域的应用涉及处理本质上具有概率性的数据。例如，某个传感器的温度读数可以表示为连续区间上的概率分布，或者脏数据库中的客户姓名可以表示为一组可能的姓名。在这些情况下，前$k$查询以及其他类型的查询在制定和处理时需要考虑数据的不确定性。

We classify top- $k$ processing techniques based on query and data certainty as follows:

我们根据查询和数据的确定性对前$k$处理技术进行如下分类：

-Exact methods over certain data. This category includes the majority of current top- $k$ processing techniques,where deterministic top- $k$ queries are processed over deterministic data.

-对确定数据的精确方法。这一类别包括当前大多数前$k$处理技术，其中对确定性数据处理确定性的前$k$查询。

-Approximate methods over certain data. This category includes top- $k$ processing techniques that operate on deterministic data, but report approximate answers in favor of performance. The approximate answers are usually associated with probabilistic guarantees indicating how far they are from the exact answer. Examples include Theobald et al. [2005] and Amato et al. [2003]. We discuss the details of these techniques in Section 5.1.

- 特定数据上的近似方法。这一类别包括对确定性数据进行操作的前 $k$ 处理技术，但为了提高性能而报告近似答案。近似答案通常与概率保证相关联，表明它们与精确答案的差距。示例包括西奥博尔德（Theobald）等人 [2005] 和阿马托（Amato）等人 [2003]。我们将在第 5.1 节讨论这些技术的细节。

-Uncertain data. This category includes top- $k$ processing techniques that work on probabilistic data. The research proposals in this category formulate top- $k$ queries based on different uncertainty models. Some approaches treat probabilities as the only scoring dimension,where a top- $k$ query is a Boolean query that reports the $k$ most probable query answers. Other approaches study the interplay between the scoring and probability dimensions. Examples are Ré et al. [2007] and Soliman et al. [2007]. We discuss the details of these techniques in Section 5.2.

- 不确定数据。这一类别包括对概率数据进行操作的前 $k$ 处理技术。该类别中的研究提案基于不同的不确定性模型来制定前 $k$ 查询。一些方法将概率视为唯一的评分维度，其中前 $k$ 查询是一个布尔查询，报告 $k$ 个最可能的查询答案。其他方法研究评分和概率维度之间的相互作用。示例有雷（Ré）等人 [2007] 和索利曼（Soliman）等人 [2007]。我们将在第 5.2 节讨论这些技术的细节。

### 2.5. Ranking Function Dimension

### 2.5. 排名函数维度

The properties of the ranking function largely influence the design of top- $k$ processing techniques. One important property is the ability to upper bound objects' scores. This property allows early pruning of certain objects without exactly knowing their scores. A monotone ranking function can largely facilitate upper bound computation. A function $F$ ,defined on predicates ${p}_{1},\ldots ,{p}_{n}$ ,is monotone if $F\left( {{p}_{1},\ldots ,{p}_{n}}\right)  \leq  F\left( {\dot{{p}_{1}},\ldots ,\dot{{p}_{n}}}\right)$ whenever ${p}_{i} \leq  {p}_{i}$ for every $i$ . We elaborate on the impact of function monotonicity on top- $k$ processing in Section 6.1.

排名函数的属性在很大程度上影响前 $k$ 处理技术的设计。一个重要的属性是对对象分数进行上界估计的能力。这一属性允许在不完全了解某些对象分数的情况下对其进行早期剪枝。单调排名函数可以极大地促进上界计算。如果对于每个 $i$ ，当 ${p}_{i} \leq  {p}_{i}$ 时都有 $F\left( {{p}_{1},\ldots ,{p}_{n}}\right)  \leq  F\left( {\dot{{p}_{1}},\ldots ,\dot{{p}_{n}}}\right)$ ，那么定义在谓词 ${p}_{1},\ldots ,{p}_{n}$ 上的函数 $F$ 是单调的。我们将在第 6.1 节详细阐述函数单调性对前 $k$ 处理的影响。

In more complex applications, a ranking function might need to be expressed as a numeric expression to be optimized. In this setting, the monotonicity restriction of the ranking function is relaxed to allow for more generic functions. Numerical optimization tools as well as indexes are used to overcome the processing challenges imposed by such ranking functions.

在更复杂的应用中，排名函数可能需要表示为一个待优化的数值表达式。在这种情况下，排名函数的单调性限制会被放宽，以允许使用更通用的函数。数值优化工具以及索引被用于克服此类排名函数带来的处理挑战。

Another group of applications address ranking objects without specifying a ranking function. In some environments, such as data exploration or decision making, it might not be important to rank objects based on a specific ranking function. Instead, objects with high quality based on different data attributes need to be reported for further analysis. These objects could possibly be among the top- $k$ objects of some unspecified ranking function. The set of objects that are not dominated by any other objects, based on some given attributes, are usually referred to as the skyline.

另一类应用在不指定排名函数的情况下对对象进行排名。在某些环境中，如数据探索或决策制定，基于特定的排名函数对对象进行排名可能并不重要。相反，需要报告基于不同数据属性的高质量对象以供进一步分析。这些对象可能是某些未指定排名函数的前 $k$ 对象之一。基于某些给定属性，不被任何其他对象支配的对象集通常被称为天际线（skyline）。

We classify top- $k$ processing techniques based on the restrictions they impose on the underlying ranking function as follows:

我们根据前 $k$ 处理技术对底层排名函数施加的限制，将其分类如下：

-Monotone ranking function. Most of the current top- $k$ processing techniques assume monotone ranking functions since they fit in many practical scenarios, and have appealing properties allowing for efficient top- $k$ processing. One example is Fagin et al. [2001]. We discuss the properties of monotone ranking functions in Section 6.1.

- 单调排名函数。目前大多数前 $k$ 处理技术都假设使用单调排名函数，因为它们适用于许多实际场景，并且具有允许高效进行前 $k$ 处理的吸引人的属性。一个例子是法金（Fagin）等人 [2001]。我们将在第 6.1 节讨论单调排名函数的属性。

—Generic ranking function. A few recent techniques, for example, Zhang et al. [2006], address top- $k$ queries in the context of constrained function optimization. The ranking function in this case is allowed to take a generic form. We discuss the details of these techniques in Section 6.2.

—通用排名函数。最近的一些技术，例如张（Zhang）等人 [2006]，在约束函数优化的背景下处理前 $k$ 查询。在这种情况下，排名函数可以采用通用形式。我们将在第 6.2 节讨论这些技术的细节。

- No ranking function. Many techniques have been proposed to answer skyline-related queries, for example, Börzsönyi et al. [2001] and Yuan et al. [2005]. Covering current skyline literature in detail is beyond the scope of this survey. We believe it worth a dedicated survey by itself. However, we briefly show the connection between skyline and top- $k$ queries in Section 6.3.

- 无排名函数。已经提出了许多技术来回答与天际线相关的查询，例如博尔佐尼（Börzsönyi）等人 [2001] 和袁（Yuan）等人 [2005]。详细涵盖当前的天际线文献超出了本综述的范围。我们认为它本身值得进行专门的综述。然而，我们将在第 6.3 节简要展示天际线和前 $k$ 查询之间的联系。

### 2.6. Impact of Design Dimensions on Top- $k$ Processing Techniques

### 2.6. 设计维度对前 $k$ 处理技术的影响

Figure 4 shows the properties of a sample of different top- $k$ processing techniques that we describe in this survey. The applicable categories under each taxonomy dimension are marked for each technique. For example, TA [Fagin et al. 2001] is an exact method that assumes top- $k$ selection query model,and operates on certain data,exploiting both sorted and random access methods. TA integrates with database systems at the application level, and supports monotone ranking functions.

图 4 展示了本综述中描述的不同前 $k$ 处理技术示例的属性。对于每种技术，标记了每个分类维度下适用的类别。例如，TA [法金（Fagin）等人 2001] 是一种精确方法，它假设采用前 $k$ 选择查询模型，对特定数据进行操作，同时利用排序和随机访问方法。TA 在应用层与数据库系统集成，并支持单调排名函数。

Our taxonomy encapsulates different perspectives to understand the processing requirements of current top- $k$ processing techniques. The taxonomy dimensions,discussed in the previous sections, can be viewed as design dimensions that impact the capabilities and the assumptions of the underlying top- $k$ algorithms. In the following, we give some examples of the impact of each design dimension on the underlying top- $k$ processing techniques:

我们的分类法涵盖了不同的视角，以理解当前前$k$处理技术的处理需求。前几节讨论的分类维度可以看作是影响底层前$k$算法能力和假设的设计维度。下面，我们给出每个设计维度对底层前$k$处理技术影响的一些示例：

-Impact of query model. The query model significantly affects the solution space of the top- $k$ algorithms. For example,the top- $k$ join query model (Definition 2.2) imposes tight integration with the query engine and physical join operators to efficiently navigate the Cartesian space of join results.

-查询模型的影响。查询模型显著影响前$k$算法的解空间。例如，前$k$连接查询模型（定义2.2）要求与查询引擎和物理连接运算符紧密集成，以便有效地遍历连接结果的笛卡尔空间。

-Impact of data access. Available access methods affect how different algorithms compute bounds on object scores and hence affect the termination condition. For example, the NRA algorithm [Fagin et al. 2001], discussed in Section 3.2, has to compute a "range" of possible scores for each object since the lack of random access prevents computing an exact score for each seen object. On the other hand, allowing random access to the underlying data sources triggers the need for cost models to optimize the number of random and sorted accesses. One example is the CA algorithm [Fagin et al. 2001], discussed in Section 3.1.

-数据访问的影响。可用的访问方法会影响不同算法如何计算对象得分的边界，从而影响终止条件。例如，第3.2节讨论的NRA算法[Fagin等人，2001]必须为每个对象计算一个可能得分的“范围”，因为缺乏随机访问功能，无法为每个已查看的对象计算精确得分。另一方面，允许对底层数据源进行随机访问会引发对成本模型的需求，以优化随机访问和排序访问的次数。一个例子是第3.1节讨论的CA算法[Fagin等人，2001]。

<!-- Media -->

<table><tr><td rowspan="2"/><td colspan="3">Query model</td><td colspan="3">Data & query certainty</td><td colspan="3">Data access</td><td colspan="2">$\mathbf{{Implement}.}$ level</td><td colspan="2">Ranking function</td></tr><tr><td>uonəəəs y-doL</td><td>urof $y$ -dot</td><td>эевэлзове у-do」</td><td>spou10ux0exa(elep uterio)</td><td>spoqiәuuxo.idde (eye) usual</td><td>етер иедэоц</td><td>uopue.i ON</td><td>mopue.i pue partos que</td><td>Saqoudиориелрэцолиоэтрэдо𝚂</td><td>$\left\lbrack  {\partial \Lambda \partial }\right\rbrack$ 可以取 $\partial L$ ,</td><td>[əʌə] uoneopidd $\forall$</td><td>QUOLOUON</td><td>OLIOUDD</td></tr><tr><td>TA [Fagin et al. 2001], Quick-Combine [Güntzer et al. 2000]</td><td>✓</td><td/><td/><td>✓</td><td/><td/><td/><td>✓</td><td/><td/><td>✓</td><td>✓</td><td/></tr><tr><td>TA-Θ approx [Fagin et al. 2003]</td><td>✓</td><td/><td/><td/><td>✓</td><td/><td/><td>✓</td><td/><td/><td>✓</td><td>✓</td><td/></tr><tr><td>NRA [Fagin et al. 2001], Stream-Combine [Güntzer et al. 2001]</td><td>✓</td><td/><td/><td>✓</td><td/><td/><td>✓</td><td/><td/><td/><td>✓</td><td>✓</td><td/></tr><tr><td>CA [Fagin et al. 2001]</td><td>✓</td><td/><td/><td>✓</td><td/><td/><td/><td>✓</td><td/><td/><td>✓</td><td>✓</td><td/></tr><tr><td>Upper/Pick [Bruno et al. 2002b]</td><td>✓</td><td/><td/><td>✓</td><td/><td/><td/><td/><td>✓</td><td/><td>✓</td><td>✓</td><td/></tr><tr><td>Mpro [Chang and Hwang 2002]</td><td>✓</td><td/><td/><td>✓</td><td/><td/><td/><td/><td>✓</td><td/><td>✓</td><td>✓</td><td/></tr><tr><td>J* [Natsev et al. 2001]</td><td/><td>✓</td><td/><td>✓</td><td/><td/><td>✓</td><td/><td/><td/><td>✓</td><td>✓</td><td/></tr><tr><td>J* e-approx. [Natsev et al. 2001]</td><td/><td>✓</td><td/><td/><td>✓</td><td/><td>✓</td><td/><td/><td/><td>✓</td><td>✓</td><td/></tr><tr><td>PREFER [Hristidis et al. 2001], Filter-Restart [Bruno et al. 2002a], Onion Indices [Chang et al. 2000], LPTA [Das et al. 2006]</td><td colspan="2">✓</td><td/><td>✓</td><td/><td/><td colspan="3">N/A</td><td/><td>✓</td><td>✓</td><td/></tr><tr><td>NRA-RJ [Ilyas et al. 2002]</td><td>✓</td><td/><td/><td>✓</td><td/><td/><td>✓</td><td/><td/><td>✓</td><td/><td>✓</td><td/></tr><tr><td>Rank-Join [Ilyas et al. 2003]</td><td/><td>✓</td><td/><td>✓</td><td/><td/><td/><td/><td>✓</td><td>✓</td><td/><td>✓</td><td/></tr><tr><td>RankSQL - μ operator [Li et al. 2005]</td><td>✓</td><td/><td/><td>✓</td><td/><td/><td/><td/><td>✓</td><td>✓</td><td/><td>✓</td><td/></tr><tr><td>rankaggr Operator [Li et al. 2006]</td><td/><td/><td>✓</td><td>✓</td><td/><td/><td>✓</td><td/><td/><td>✓</td><td/><td>✓</td><td/></tr><tr><td>TopX [Theobald et al. 2005]</td><td>✓</td><td/><td/><td/><td>✓</td><td/><td/><td>✓</td><td/><td/><td>✓</td><td>✓</td><td/></tr><tr><td>KLEE [Michel et al. 2005]</td><td>✓</td><td/><td/><td/><td>✓</td><td/><td>✓</td><td/><td/><td/><td>✓</td><td>✓</td><td/></tr><tr><td>OPT* [Zhang et al. 2006]</td><td colspan="2">✓</td><td/><td>✓</td><td/><td/><td colspan="3">N/A</td><td/><td>✓</td><td/><td>✓</td></tr><tr><td>OPTU-Topk [Soliman et al. 2007]</td><td colspan="2">✓</td><td/><td/><td/><td>✓</td><td>✓</td><td/><td/><td/><td>✓</td><td>✓</td><td/></tr><tr><td>MS_Topk [Ré et al. 2007]</td><td colspan="3">✓</td><td/><td/><td>✓</td><td colspan="3">N/A</td><td/><td>✓</td><td>✓</td><td/></tr></table>

<table><tbody><tr><td rowspan="2"></td><td colspan="3">查询模型</td><td colspan="3">数据与查询确定性</td><td colspan="3">数据访问</td><td colspan="2">$\mathbf{{Implement}.}$ 级别</td><td colspan="2">排序函数</td></tr><tr><td>uonəəəs y-doL</td><td>urof $y$ -点积</td><td>эевэлзове у-do」</td><td>spou10ux0exa(elep uterio)</td><td>spoqiәuuxo.idde (眼睛) 通常情况</td><td>етер иедэоц</td><td>uopue.i ON</td><td>mopue.i pue partos que</td><td>Saqoudиориелрэцолиоэтрэдо𝚂</td><td>$\left\lbrack  {\partial \Lambda \partial }\right\rbrack$ 可以取 $\partial L$ ,</td><td>[əʌə] uoneopidd $\forall$</td><td>QUOLOUON</td><td>OLIOUDD</td></tr><tr><td>TA [法金（Fagin）等人，2001年]，快速合并（Quick-Combine） [京策尔（Güntzer）等人，2000年]</td><td>✓</td><td></td><td></td><td>✓</td><td></td><td></td><td></td><td>✓</td><td></td><td></td><td>✓</td><td>✓</td><td></td></tr><tr><td>TA - Θ 近似算法 [法金（Fagin）等人，2003年]</td><td>✓</td><td></td><td></td><td></td><td>✓</td><td></td><td></td><td>✓</td><td></td><td></td><td>✓</td><td>✓</td><td></td></tr><tr><td>NRA [法金（Fagin）等人，2001年]，流合并（Stream - Combine） [京策尔（Güntzer）等人，2001年]</td><td>✓</td><td></td><td></td><td>✓</td><td></td><td></td><td>✓</td><td></td><td></td><td></td><td>✓</td><td>✓</td><td></td></tr><tr><td>CA [法金（Fagin）等人，2001年]</td><td>✓</td><td></td><td></td><td>✓</td><td></td><td></td><td></td><td>✓</td><td></td><td></td><td>✓</td><td>✓</td><td></td></tr><tr><td>上限/选择（Upper/Pick） [布鲁诺（Bruno）等人，2002b]</td><td>✓</td><td></td><td></td><td>✓</td><td></td><td></td><td></td><td></td><td>✓</td><td></td><td>✓</td><td>✓</td><td></td></tr><tr><td>Mpro [张（Chang）和黄（Hwang），2002年]</td><td>✓</td><td></td><td></td><td>✓</td><td></td><td></td><td></td><td></td><td>✓</td><td></td><td>✓</td><td>✓</td><td></td></tr><tr><td>J* [纳采夫（Natsev）等人，2001年]</td><td></td><td>✓</td><td></td><td>✓</td><td></td><td></td><td>✓</td><td></td><td></td><td></td><td>✓</td><td>✓</td><td></td></tr><tr><td>J* e - 近似算法 [纳采夫（Natsev）等人，2001年]</td><td></td><td>✓</td><td></td><td></td><td>✓</td><td></td><td>✓</td><td></td><td></td><td></td><td>✓</td><td>✓</td><td></td></tr><tr><td>偏好算法（PREFER） [赫里斯蒂迪斯（Hristidis）等人，2001年]，过滤重启（Filter - Restart） [布鲁诺（Bruno）等人，2002a]，洋葱索引（Onion Indices） [张（Chang）等人，2000年]，LPTA [达斯（Das）等人，2006年]</td><td colspan="2">✓</td><td></td><td>✓</td><td></td><td></td><td colspan="3">不适用</td><td></td><td>✓</td><td>✓</td><td></td></tr><tr><td>NRA - RJ [伊利亚斯（Ilyas）等人，2002年]</td><td>✓</td><td></td><td></td><td>✓</td><td></td><td></td><td>✓</td><td></td><td></td><td>✓</td><td></td><td>✓</td><td></td></tr><tr><td>排名连接（Rank - Join） [伊利亚斯（Ilyas）等人，2003年]</td><td></td><td>✓</td><td></td><td>✓</td><td></td><td></td><td></td><td></td><td>✓</td><td>✓</td><td></td><td>✓</td><td></td></tr><tr><td>RankSQL - μ 运算符 [李（Li）等人，2005年]</td><td>✓</td><td></td><td></td><td>✓</td><td></td><td></td><td></td><td></td><td>✓</td><td>✓</td><td></td><td>✓</td><td></td></tr><tr><td>排名聚合运算符（rankaggr Operator） [李（Li）等人，2006年]</td><td></td><td></td><td>✓</td><td>✓</td><td></td><td></td><td>✓</td><td></td><td></td><td>✓</td><td></td><td>✓</td><td></td></tr><tr><td>TopX [西奥博尔德（Theobald）等人，2005年]</td><td>✓</td><td></td><td></td><td></td><td>✓</td><td></td><td></td><td>✓</td><td></td><td></td><td>✓</td><td>✓</td><td></td></tr><tr><td>克莱（KLEE） [米歇尔（Michel）等人，2005年]</td><td>✓</td><td></td><td></td><td></td><td>✓</td><td></td><td>✓</td><td></td><td></td><td></td><td>✓</td><td>✓</td><td></td></tr><tr><td>OPT* [张（Zhang）等人，2006年]</td><td colspan="2">✓</td><td></td><td>✓</td><td></td><td></td><td colspan="3">不适用</td><td></td><td>✓</td><td></td><td>✓</td></tr><tr><td>OPTU - Topk [索利曼（Soliman）等人，2007年]</td><td colspan="2">✓</td><td></td><td></td><td></td><td>✓</td><td>✓</td><td></td><td></td><td></td><td>✓</td><td>✓</td><td></td></tr><tr><td>MS_Topk [雷（Ré）等人，2007年]</td><td colspan="3">✓</td><td></td><td></td><td>✓</td><td colspan="3">不适用</td><td></td><td>✓</td><td>✓</td><td></td></tr></tbody></table>

Fig. 4. Properties of different top- $k$ processing techniques.

图4. 不同的前$k$处理技术的特性。

<!-- Media -->

-Impact of data and query uncertainty. Supporting approximate query answers requires building probabilistic models to fit the score distributions of the underlying data, as proposed in Theobald et al. [2004] (Section 5.1.2). Uncertainty in the underlying data adds further significant computational complexity because of the huge space of possible answers that needs to be explored. Building efficient search algorithms to explore such space is crucial, as addressed in Soliman et al. [2007].

-数据和查询不确定性的影响。支持近似查询答案需要构建概率模型以拟合底层数据的得分分布，如西奥博尔德（Theobald）等人在2004年所提出的（第5.1.2节）。由于需要探索的可能答案空间巨大，底层数据的不确定性进一步增加了显著的计算复杂度。如索利曼（Soliman）等人在2007年所论述的，构建高效的搜索算法来探索这样的空间至关重要。

-Impact of implementation level. The implementation level greatly affects the requirements of the top- $k$ algorithm. For example,implementing top- $k$ pipelined query operator necessitates using algorithms that require no random access to their inputs to fit in pipelined query models; it also requires the output of the top- $k$ algorithm to be a valid input to another instance of the algorithm [Ilyas et al. 2004a]. On the other hand, implementation on the application level does not have these requirements. More details are given in Section 4.2.1.

-实现层面的影响。实现层面极大地影响着前$k$算法的要求。例如，实现前$k$流水线查询操作符需要使用那些不需要随机访问其输入的算法，以适应流水线查询模型；它还要求前$k$算法的输出是该算法另一个实例的有效输入[伊利亚斯（Ilyas）等人，2004a]。另一方面，在应用层面的实现则没有这些要求。更多细节在第4.2.1节给出。

-Impact of ranking function. Assuming monotone ranking functions allows top- $k$ processing techniques to benefit from the monotonicity property to guarantee early-out of query answers. Dealing with nonmonotone functions requires more sophisticated bounding for the scores of unexplored answers. Existing indexes in the database are currently used to provide such bounding, as addressed in Xin et al. [2007] (Section 6).

-排序函数的影响。假设排序函数是单调的，前$k$处理技术可以利用单调性来保证查询答案的提前输出。处理非单调函数需要对未探索答案的得分进行更复杂的界定。如辛（Xin）等人在2007年所论述的（第6节），目前数据库中现有的索引被用于提供这样的界定。

## 3. DATA ACCESS

## 3. 数据访问

In this section,we discuss top- $k$ processing techniques that make different assumptions about available access methods supported by data sources. The primary data access methods are sorted access, random access, and a combination of both methods. In sorted access, objects are accessed sequentially ordered by some scoring predicate, while for random access, objects are directly accessed by their identifiers.

在本节中，我们讨论对数据源支持的可用访问方法做出不同假设的前$k$处理技术。主要的数据访问方法有排序访问、随机访问以及这两种方法的组合。在排序访问中，对象按照某个评分谓词顺序访问，而对于随机访问，对象通过其标识符直接访问。

The techniques presented in this section assume multiple lists (possibly located at separate sources) that rank the same set of objects based on different scoring predicates. A score aggregation function is used to aggregate partial objects' scores, obtained from the different lists,to find the top- $k$ answers.

本节介绍的技术假设存在多个列表（可能位于不同的源），这些列表基于不同的评分谓词对同一组对象进行排序。使用得分聚合函数来聚合从不同列表中获得的部分对象的得分，以找到前$k$答案。

The cost of executing a top- $k$ query,in such environments,is largely influenced by the supported data access methods. For example, random access is generally more expensive than sorted access. A common assumption in all of the techniques discussed in this section is the existence of at least one source that supports sorted access. We categorize top- $k$ processing techniques,according to the assumed source capabilities, into the three categories described in the next sections.

在这样的环境中，执行前$k$查询的成本在很大程度上受支持的数据访问方法的影响。例如，随机访问通常比排序访问成本更高。本节讨论的所有技术的一个常见假设是至少存在一个支持排序访问的源。我们根据假设的源能力将前$k$处理技术分为下一节描述的三类。

### 3.1. Both Sorted and Random Access

### 3.1. 排序访问和随机访问

Top- $k$ processing techniques in this category assume data sources that support both access methods, sorted and random. Random access allows for obtaining the overall score of some object right after it appears in one of the data sources. The Threshold Algorithm (TA) and Combined Algorithm (CA) [Fagin et al. 2001] belong to this category.

这一类的前$k$处理技术假设数据源同时支持排序访问和随机访问这两种访问方法。随机访问允许在某个对象出现在其中一个数据源后立即获得该对象的总体得分。阈值算法（TA）和组合算法（CA）[法金（Fagin）等人，2001]属于这一类。

Algorithm 1 describes the details of TA. The algorithm scans multiple lists, representing different rankings of the same set of objects. An upper bound $T$ is maintained for the overall score of unseen objects. The upper bound is computed by applying the scoring function to the partial scores of the last seen objects in different lists. Notice that the last seen objects in different lists could be different. The upper bound is updated every time a new object appears in one of the lists. The overall score of some seen object is computed by applying the scoring function to object's partial scores, obtained from different lists. To obtain such partial scores, each newly seen object in one of the lists is looked up in all other lists, and its scores are aggregated using the scoring function to obtain the overall score. All objects with total scores that are greater than or equal to $T$ can be reported. The algorithm terminates after returning the $k$ th output. Example 3.1 illustrates the processing of TA.

算法1描述了TA的详细信息。该算法扫描多个列表，这些列表代表同一组对象的不同排序。为未见过的对象的总体得分维护一个上界$T$。上界是通过对不同列表中最后看到的对象的部分得分应用评分函数来计算的。注意，不同列表中最后看到的对象可能不同。每次有新对象出现在其中一个列表中时，上界都会更新。通过对从不同列表中获得的对象的部分得分应用评分函数来计算某个已见对象的总体得分。为了获得这样的部分得分，在其中一个列表中每次看到的新对象都会在所有其他列表中查找，并使用评分函数聚合其得分以获得总体得分。所有总得分大于或等于$T$的对象都可以被报告。该算法在返回第$k$个输出后终止。示例3.1说明了TA的处理过程。

<!-- Media -->

<!-- figureText: First Step OID P2 $\mathbf{T} = {100}$ 3 50 2 40 3: (80) 30 5: (60) 20 10 $\mathrm{T} = {75}$ OID $\mathbf{{P2}}$ 3 50 3: (80) 2 40 1: (65) 30 5: (60) 4 20 2: (60) 5 10 ${\mathrm{L}}_{1}$ Buffer OID P1 5 50 1 35 3 30 2 20 10 Second Step OID P1 5 50 1 35 3 30 2 20 4 10 ${\mathrm{L}}_{2}$ -->

<img src="https://cdn.noedgeai.com/0195c915-c519-7d98-b2ff-761eddc2c676_12.jpg?x=468&y=283&w=789&h=596&r=0"/>

Fig. 5. The Threshold Algorithm (TA).

图5. 阈值算法（TA）。

Algorithm 1. TA [Fagin et al. 2001]

算法1. TA [法金（Fagin）等人，2001]

---

(1) Do sorted access in parallel to each of the $m$ sorted lists ${L}_{i}$ . As a new object $o$ is seen under

(1) 并行地对每个$m$个排序列表${L}_{i}$进行排序访问。当在某个列表的排序访问中看到一个新对象$o$时

	sorted access in some list,do random access to the other lists to find ${p}_{i}\left( o\right)$ in every other list

	对其他列表进行随机访问，以在每个其他列表中找到${p}_{i}\left( o\right)$

	${L}_{i}$ . Compute the score $F\left( o\right)  = F\left( {{p}_{1},\ldots ,{p}_{m}}\right)$ of object $o$ . If this score is among the $k$ highest

	${L}_{i}$。计算对象$o$的得分$F\left( o\right)  = F\left( {{p}_{1},\ldots ,{p}_{m}}\right)$。如果这个得分是$k$个最高得分之一

	scores seen so far,then remember object $o$ and its score $F\left( o\right)$ (ties are broken arbitrarily,so

	到目前为止所看到的分数，然后记住对象 $o$ 及其分数 $F\left( o\right)$（若分数相同则随机打破平局，因此

	that only $k$ objects and their scores are remembered at any time).

	任何时候仅会记录$k$对象及其得分。

(2) For each list ${L}_{i}$ ,let ${\bar{p}}_{i}$ be the score of the last object seen under sorted access. Define the

(2) 对于每个列表 ${L}_{i}$，设 ${\bar{p}}_{i}$ 为按排序访问时最后看到的对象的得分。定义

	threshold value $T$ to be $F\left( {{\bar{p}}_{1},\ldots ,{\bar{p}}_{m}}\right)$ . As soon as at least $k$ objects have been seen with

	阈值 $T$ 设为 $F\left( {{\bar{p}}_{1},\ldots ,{\bar{p}}_{m}}\right)$。一旦至少观察到 $k$ 个对象，且

	scores at least equal to $T$ ,halt.

	得分至少等于$T$，停止。

(3) Let ${A}_{k}$ be a set containing the $k$ seen objects with the highest scores. The output is the sorted

(3) 设 ${A}_{k}$ 为一个包含得分最高的 $k$ 个已见对象的集合。输出为排序后的

	set $\left\{  {\left( {o,F\left( o\right) }\right)  \mid  o \in  {A}_{k}}\right\}$ .

	设置 $\left\{  {\left( {o,F\left( o\right) }\right)  \mid  o \in  {A}_{k}}\right\}$。

---

<!-- Media -->

Example 3.1 (TA Example). Consider two data sources ${L}_{1}$ and ${L}_{2}$ holding different rankings for the same set of objects based on two different scoring predicates ${p}_{1}$ and ${p}_{2}$ ,respectively. Each of ${p}_{1}$ and ${p}_{2}$ produces score values in the range $\left\lbrack  {0,{50}}\right\rbrack$ . Assume each source supports sorted and random access to their ranked lists. Consider a score aggregation function $F = {p}_{1} + {p}_{2}$ . Figure 5 depicts the first two steps of TA. In the first step, retrieving the top object from each list, and probing the value of its other scoring predicate in the other list, result in revealing the exact scores for the top objects. The seen objects are buffered in the order of their scores. A threshold value, $T$ ,for the scores of unseen objects is computed by applying $F$ to the last seen scores in both lists,which results in ${50} + {50} = {100}$ . Since both seen objects have scores less than $T$ ,no results can be reported. In the second step, $T$ drops to 75,and object 3 can be safely reported since its score is above $T$ . The algorithm continues until $k$ objects are reported,or sources are exhausted.

示例3.1（TA示例）。考虑两个数据源${L}_{1}$和${L}_{2}$，它们分别基于两个不同的评分谓词${p}_{1}$和${p}_{2}$，对同一组对象持有不同的排名。${p}_{1}$和${p}_{2}$各自产生的分数值范围为$\left\lbrack  {0,{50}}\right\rbrack$。假设每个数据源都支持对其排序列表进行排序访问和随机访问。考虑一个分数聚合函数$F = {p}_{1} + {p}_{2}$。图5展示了TA的前两个步骤。在第一步中，从每个列表中检索排名最高的对象，并在另一个列表中探查该对象的另一个评分谓词的值，从而得出排名最高对象的确切分数。已查看的对象按其分数顺序进行缓冲。通过对两个列表中最后查看的分数应用$F$，计算出未查看对象分数的阈值$T$，结果为${50} + {50} = {100}$。由于两个已查看对象的分数都小于$T$，因此无法报告任何结果。在第二步中，$T$降至75，由于对象3的分数高于$T$，因此可以安全地报告该对象。该算法会继续执行，直到报告了$k$个对象，或者数据源耗尽为止。

TA assumes that the costs of different access methods are the same. In addition, TA does not have a restriction on the number of random accesses to be performed. Every sorted access in TA results in up to $m - 1$ random accesses,where $m$ is the number of lists. Such a large number of random accesses might be very expensive. The CA algorithm [Fagin et al. 2001] alternatively assumes that the costs of different access methods are different. The CA algorithm defines a ratio between the costs of the two access methods to control the number of random accesses, since they usually have higher costs than sorted accesses.

TA算法（TA）假设不同访问方法的成本相同。此外，TA算法对要执行的随机访问次数没有限制。在TA算法中，每次有序访问最多会导致$m - 1$次随机访问，其中$m$是列表的数量。如此大量的随机访问可能成本很高。相比之下，CA算法（[Fagin等人，2001年]）假设不同访问方法的成本不同。CA算法定义了两种访问方法成本之间的比率，以控制随机访问的次数，因为随机访问的成本通常比有序访问高。

The CA algorithm periodically performs random accesses to collect unknown partial scores for objects with the highest score lower bounds (ties are broken using score upper bounds). A score lower bound is computed by applying the scoring function to object's known partial scores, and the worst possible unknown partial scores. On the other hand, a score upper bound is computed by applying the scoring function to object's known partial scores, and the best possible unknown partial scores. The worst unknown partial scores are the lowest values in the score range, while the best unknown partial scores are the last seen scores in different lists. One random access is performed periodically every $\Delta$ sorted accesses,where $\Delta$ is the floor of the ratio between random access cost and sorted access cost.

CA算法会定期进行随机访问，以收集得分下界最高的对象的未知部分得分（若得分下界相同，则使用得分上界来打破平局）。得分下界是通过将评分函数应用于对象的已知部分得分和可能的最差未知部分得分来计算的。另一方面，得分上界是通过将评分函数应用于对象的已知部分得分和可能的最佳未知部分得分来计算的。最差的未知部分得分是得分范围内的最低值，而最佳的未知部分得分是不同列表中最后看到的得分。每进行$\Delta$次有序访问后会定期进行一次随机访问，其中$\Delta$是随机访问成本与有序访问成本之比的向下取整值。

Although CA minimizes the number of random accesses compared to TA, it assumes that all sources support random access at the same cost, which may not be true in practice. This problem is addressed in Bruno et al. [2002b] and Marian et al. [2004], and we discuss it in more detail in Section 3.3.

尽管与TA（试探性访问）相比，CA（确定性访问）最大限度地减少了随机访问的次数，但它假设所有数据源都能以相同的成本支持随机访问，而这在实际中可能并不成立。Bruno等人[2002b]和Marian等人[2004]探讨了这个问题，我们将在3.3节中更详细地讨论它。

In TA, tuples are retrieved from sorted lists in a round-robin style. For instance, if there are $m$ sorted access sources,tuples are retrieved from sources in this order: $\left( {{L}_{1},{L}_{2},\ldots ,{L}_{m},{L}_{1},\ldots }\right)$ . Two observations can possibly minimize the number of retrieved tuples. First, sources with rapidly decreasing scores can help decrease the upper bound of unseen objects’ scores(T)at a faster rate. Second, favoring sources with considerable influence on the overall scores could lead to identifying the top answers quickly. Based on these two observations, a variation of TA, named Quick-Combine, is introduced in Güntzer et al. [2000]. The Quick-Combine algorithm uses an indicator ${\Delta }_{i}$ ,expressing the effectiveness of reading from source $i$ ,defined as follows:

在TA（Top-k Aggregation，前k个聚合）算法中，元组以轮询的方式从排序列表中检索。例如，如果有$m$个排序的访问源，元组将按以下顺序从这些源中检索：$\left( {{L}_{1},{L}_{2},\ldots ,{L}_{m},{L}_{1},\ldots }\right)$。有两个观察结果可能会使检索的元组数量最小化。首先，得分快速下降的源可以更快地降低未查看对象得分（T）的上限。其次，优先选择对总体得分有显著影响的源可能会快速识别出前几名答案。基于这两个观察结果，Güntzer等人在2000年提出了TA算法的一个变体，名为Quick - Combine（快速合并）。Quick - Combine算法使用一个指标${\Delta }_{i}$，表示从源$i$读取的有效性，定义如下：

$$
{\Delta }_{i} = \frac{\partial F}{\partial {p}_{i}} \cdot  \left( {{S}_{i}\left( {{d}_{i} - c}\right)  - {S}_{i}\left( {d}_{i}\right) }\right) , \tag{1}
$$

where ${S}_{i}\left( x\right)$ refers to the score of the tuple at depth $x$ in source $i$ ,and ${d}_{i}$ is the current depth reached at source $i$ . The rate at which score decays in source $i$ is computed as the difference between its last seen score ${S}_{i}\left( {d}_{i}\right)$ ,and the score of the tuple $c$ steps above in the ranked list, ${S}_{i}\left( {{d}_{i} - c}\right)$ . The influence of source $i$ on the scoring function $F$ is captured using the partial derivative of $F$ with respect to source’s predicate ${p}_{i}$ . The source with the maximum ${\Delta }_{i}$ is selected,at each step,to get the next object. It has been shown that the proposed algorithm is particularly efficient when the data exhibits tangible skewness.

其中${S}_{i}\left( x\right)$指的是源$i$中深度为$x$的元组的得分，${d}_{i}$是源$i$当前达到的深度。源$i$中得分的衰减率计算为其最后一次看到的得分${S}_{i}\left( {d}_{i}\right)$与排序列表中上方$c$步的元组得分${S}_{i}\left( {{d}_{i} - c}\right)$之间的差值。源$i$对评分函数$F$的影响通过$F$关于源的谓词${p}_{i}$的偏导数来衡量。在每一步，选择${\Delta }_{i}$值最大的源来获取下一个对象。研究表明，当数据呈现明显的偏态时，所提出的算法特别有效。

#### 3.2.No Random Access

#### 3.2. 不支持随机访问

The techniques we discuss in this section assume random access is not supported by the underlying sources. The No Random Access (NRA) algorithm [Fagin et al. 2001] and the Stream-Combine algorithm [Güntzer et al. 2001] are two examples of the techniques that belong to this category.

我们在本节讨论的技术假设底层源不支持随机访问。不支持随机访问（NRA，No Random Access）算法[Fagin等人，2001年]和流合并（Stream - Combine）算法[Güntzer等人，2001年]是属于这一类技术的两个例子。

The NRA algorithm finds the top- $k$ answers by exploiting only sorted accesses. The NRA algorithm may not report the exact object scores,as it produces the top- $k$ answers using bounds computed over their exact scores. The score lower bound of some object $t$ is obtained by applying the score aggregation function on $t$ ’s known scores and the minimum possible values of $t$ ’s unknown scores. On the other hand,the score upper bound of $t$ is obtained by applying the score aggregation function on $t$ ’s known scores and the maximum possible values of $t$ ’s unknown scores,which are the same as the last seen scores in the corresponding ranked lists. This allows the algorithm to report a top- $k$ object even if its score is not precisely known. Specifically,if the score lower bound of an object $t$ is not below the score upper bounds of all other objects (including unseen objects),then $t$ can be safely reported as the next top- $k$ object. The details of the NRA algorithm are given in Algorithm 2.

NRA算法仅通过利用排序访问来找出前$k$个答案。NRA算法可能不会报告对象的确切得分，因为它使用基于对象确切得分计算出的边界来生成前$k$个答案。某个对象$t$的得分下限是通过对$t$的已知得分和$t$未知得分的最小可能值应用得分聚合函数得到的。另一方面，$t$的得分上限是通过对$t$的已知得分和$t$未知得分的最大可能值（与相应排序列表中最后一次看到的得分相同）应用得分聚合函数得到的。这使得该算法即使在对象得分不精确已知的情况下也能报告前$k$个对象。具体来说，如果一个对象$t$的得分下限不低于所有其他对象（包括未查看对象）的得分上限，那么可以安全地将$t$报告为下一个前$k$个对象。NRA算法的详细信息见算法2。

<!-- Media -->

## Algorithm 2. NRA [Fagin et al. 2001]

## 算法2. NRA [Fagin等人，2001年]

---

(1) Let ${p}_{1}^{\min },\ldots ,{p}_{m}^{\min }$ be the smallest possible values in lists ${L}_{1},\ldots ,{L}_{m}$ .

(1) 设${p}_{1}^{\min },\ldots ,{p}_{m}^{\min }$为列表${L}_{1},\ldots ,{L}_{m}$中的最小可能值。

(2) Do sorted access in parallel to lists ${L}_{1},\ldots ,{L}_{m}$ ,and at each step do the following:

(2) 并行对列表${L}_{1},\ldots ,{L}_{m}$进行排序访问，并在每一步执行以下操作：

		-Maintain the last seen predicate values ${\bar{p}}_{1},\ldots ,{\bar{p}}_{m}$ in the $m$ lists.

		- - 维护$m$个列表中最后一次看到的谓词值${\bar{p}}_{1},\ldots ,{\bar{p}}_{m}$。

		-For every object $o$ with some unknown predicate values,compute a lower bound for $F\left( o\right)$ ,

		- - 对于每个有一些未知谓词值的对象$o$，计算$F\left( o\right)$的下限，

			denoted $\underline{F}\left( o\right)$ ,by substituting each unknown predicate ${p}_{i}$ with ${p}_{i}^{\min }$ . Similarly,Compute an

			 用${p}_{i}^{\min }$替换每个未知谓词${p}_{i}$，记为$\underline{F}\left( o\right)$。同样，计算一个

			upper bound $\bar{F}\left( o\right)$ by substituting each unknown predicate ${p}_{i}$ with ${\bar{p}}_{i}$ . For object $o$ that has

			 上限$\bar{F}\left( o\right)$，用${\bar{p}}_{i}$替换每个未知谓词${p}_{i}$。对于具有

			not been seen at all, $\underline{F}\left( o\right)  = F\left( {{p}_{1}^{\min },\ldots ,{p}_{m}^{\min }}\right)$ ,and $\bar{F}\left( o\right)  = F\left( {{\bar{p}}_{1},\ldots ,{\bar{p}}_{m}}\right)$ .

			根本未被观察到，$\underline{F}\left( o\right)  = F\left( {{p}_{1}^{\min },\ldots ,{p}_{m}^{\min }}\right)$ ，以及 $\bar{F}\left( o\right)  = F\left( {{\bar{p}}_{1},\ldots ,{\bar{p}}_{m}}\right)$ 。

	-Let ${A}_{k}$ be the set of $k$ objects with the largest lower bound values $\underline{F}\left( \text{.}\right) {seensofar}.{Iftwo}$

	- - 令 ${A}_{k}$ 为具有最大下界值 $\underline{F}\left( \text{.}\right) {seensofar}.{Iftwo}$ 的 $k$ 个对象的集合

			objects have the same lower bound,then ties are broken using their upper bounds $\bar{F}\left( \text{.}\right) S$ ,and

			 如果对象具有相同的下界，则使用它们的上界 $\bar{F}\left( \text{.}\right) S$ 来打破平局，并且

			arbitrarily among objects that additionally tie in $\overline{F}$ (.).

			 对于在 $\overline{F}$ 中额外平局的对象任意选择（.）。

		-Let ${M}_{k}$ be the $k$ th largest $\underline{F}\left( \text{.}\right) {valuein}{A}_{k}$ .

		- - 令 ${M}_{k}$ 为第 $k$ 大的 $\underline{F}\left( \text{.}\right) {valuein}{A}_{k}$ 。

(3) Call an object o viable if $\bar{F}\left( o\right)  > {M}_{k}$ . Halt when (a) at least $k$ distinct objects have been seen,

(3) 如果 $\bar{F}\left( o\right)  > {M}_{k}$ ，则称对象 o 是可行的。当 (a) 至少观察到 $k$ 个不同的对象时停止，

		and (b) there are no viable objects outside ${A}_{k}$ . That is,if $\bar{F}\left( o\right)  \leq  {M}_{k}$ for all $o \notin  {A}_{k}$ ,return ${A}_{k}$ .

		 并且 (b) 在 ${A}_{k}$ 之外没有可行的对象。也就是说，如果对于所有的 $o \notin  {A}_{k}$ 都有 $\bar{F}\left( o\right)  \leq  {M}_{k}$ ，则返回 ${A}_{k}$ 。

---

<!-- Media -->

Example 3.2 illustrates the processing of the NRA algorithm.

示例 3.2 说明了 NRA 算法（NRA Algorithm）的处理过程。

Example 3.2 (NRA Example). Consider two data sources ${L}_{1}$ and ${L}_{2}$ ,where each source holds a different ranking of the same set of objects based on scoring predicates ${p}_{1}$ and ${p}_{2}$ ,respectively. Both ${p}_{1}$ and ${p}_{2}$ produce score values in the range $\left\lbrack  {0,{50}}\right\rbrack$ . Assume both sources support only sorted access to their ranked lists. Consider a score aggregation function $F = {p}_{1} + {p}_{2}$ . Figure 6 depicts the first three steps of the NRA algorithm. In the first step, retrieving the first object in each list gives lower and upper bounds for objects’ scores. For example,object 5 has a score range of $\left\lbrack  {{50},{100}}\right\rbrack$ ,since the value of its known scoring predicate ${p}_{1}$ is 50,while the value of its unknown scoring predicate ${p}_{2}$ cannot exceed 50 . An upper bound for the scores of unseen objects is computed as ${50} + {50} = {100}$ ,which is the result of applying $F$ to the last seen scores in both sorted lists. The seen objects are buffered in the order of their score lower bounds. Since the score lower bound of object 5 , the top buffered object, does not exceed the score upper bound of other objects, nothing can be reported. The second step adds two more objects to the buffer, and updates the score bounds of other buffered objects. In the third step, the scores of objects 1 and 3 are completely known. However, since the score lower bound of object 3 is not below the score upper bound of any other object (including the unseen ones), object 3 can be reported as the top-1 object. Note that at this step object 1 cannot be additionally reported, since the score upper bound of object 5 is 80 , which is larger than the score lower bound of object 1 .

示例 3.2（NRA 示例）。考虑两个数据源 ${L}_{1}$ 和 ${L}_{2}$ ，其中每个数据源分别基于评分谓词 ${p}_{1}$ 和 ${p}_{2}$ 对同一组对象进行不同的排序。${p}_{1}$ 和 ${p}_{2}$ 都产生范围在 $\left\lbrack  {0,{50}}\right\rbrack$ 内的得分值。假设两个数据源都只支持对其排序列表进行有序访问。考虑一个得分聚合函数 $F = {p}_{1} + {p}_{2}$ 。图 6 描绘了 NRA 算法的前三个步骤。在第一步中，检索每个列表中的第一个对象会得到对象得分的下界和上界。例如，对象 5 的得分范围是 $\left\lbrack  {{50},{100}}\right\rbrack$ ，因为其已知评分谓词 ${p}_{1}$ 的值为 50 ，而其未知评分谓词 ${p}_{2}$ 的值不能超过 50 。未观察到的对象的得分上界计算为 ${50} + {50} = {100}$ ，这是将 $F$ 应用于两个排序列表中最后观察到的得分的结果。观察到的对象按其得分下界的顺序进行缓冲。由于缓冲的顶部对象（对象 5）的得分下界不超过其他对象的得分上界，因此没有对象可以被报告。第二步向缓冲区中添加了另外两个对象，并更新了其他缓冲对象的得分边界。在第三步中，对象 1 和对象 3 的得分完全已知。然而，由于对象 3 的得分下界不低于任何其他对象（包括未观察到的对象）的得分上界，因此对象 3 可以被报告为排名第一的对象。请注意，在这一步中，对象 1 不能被额外报告，因为对象 5 的得分上界为 80 ，大于对象 1 的得分下界。

The Stream-Combine algorithm [Güntzer et al. 2001] is based on the same general idea of the NRA algorithm. The Stream-Combine algorithm prioritizes reading from sorted lists to give more chance to the lists that might lead to the earliest termination. To choose which sorted list (stream) to access next,an effectiveness indicator ${\Delta }_{i}$ is computed for each stream $i$ ,similar to the Quick-Combine algorithm. The definition of ${\Delta }_{i}$ in this case captures three properties of stream $i$ that may lead to early termination: (1) how rapidly scores decrease in stream $i$ ,(2) what is the influence of stream $i$ on the total aggregated score,and (3) how many top- $k$ objects would have their score bounds tightened by reading from stream $i$ . The indicator ${\Delta }_{i}$ is defined as follows:

流合并算法 [Güntzer 等人，2001 年] 基于与 NRA 算法相同的总体思路。流合并算法优先从排序列表中读取数据，以便给那些可能导致最早终止的列表更多机会。为了选择接下来要访问的排序列表（流），会为每个流 $i$ 计算一个有效性指标 ${\Delta }_{i}$，这与快速合并算法类似。在这种情况下，${\Delta }_{i}$ 的定义捕捉了流 $i$ 可能导致提前终止的三个属性：（1）流 $i$ 中的分数下降速度有多快；（2）流 $i$ 对总聚合分数有什么影响；（3）从流 $i$ 中读取数据会使多少个前 $k$ 对象的分数边界收紧。指标 ${\Delta }_{i}$ 的定义如下：

$$
{\Delta }_{i} = \# {M}_{i} \cdot  \frac{\partial F}{\partial {p}_{i}} \cdot  \left( {{S}_{i}\left( {{d}_{i} - c}\right)  - {S}_{i}\left( {d}_{i}\right) }\right)  \tag{2}
$$

<!-- Media -->

<!-- figureText: First Step OID OID $\mathbf{{P2}}$ 5:(50 - 100) 50 3:(50 - 100) 40 20 10 P2 5 :(50 - 90) 50 3:(50 - 90) 40 1:(40 - 80) 30 2:(40 - 80) 20 10 $\mathbf{{P2}}$ 3:(80 - 80) 50 1:(70 - 70) 40 5:(50 - 80) 30 2:(40 - 70) 20 10 ${\mathrm{L}}_{2}$ Buffer 5 50 40 1 2 20 10 Second Step P1 OID 5 50 3 40 2 30 2 20 10 Third Step OID $\mathbf{{P1}}$ OID 5 50 3 40 3 30 2 20 10 ${\mathrm{L}}_{1}$ -->

<img src="https://cdn.noedgeai.com/0195c915-c519-7d98-b2ff-761eddc2c676_15.jpg?x=449&y=283&w=843&h=647&r=0"/>

Fig. 6. The three first steps of the NRA algorithm.

图 6. NRA 算法的前三个步骤。

<!-- Media -->

where ${S}_{i}\left( x\right)$ refers to the score of the tuple at depth $x$ in stream $i$ ,and ${d}_{i}$ is the current depth reached at stream $i$ .

其中 ${S}_{i}\left( x\right)$ 指的是流 $i$ 中深度为 $x$ 的元组的分数，${d}_{i}$ 是流 $i$ 当前达到的深度。

The term $\left( {{S}_{i}\left( {{d}_{i} - c}\right)  - {S}_{i}\left( {d}_{i}\right) }\right)$ captures the rate of score decay in stream $i$ ,while the term $\frac{\partial F}{\partial {n}_{i}}$ captures how much the stream’s scoring predicate contributes to the total score, similar to the Quick-Combine algorithm. The term $\# {M}_{i}$ is the number of top- $k$ objects whose score bounds may be affected when reading from stream $i$ ,by reducing their score upper bounds,or knowing their precise scores. The stream with the maximum ${\Delta }_{i}$ is selected, at each step, to get the next object.

项 $\left( {{S}_{i}\left( {{d}_{i} - c}\right)  - {S}_{i}\left( {d}_{i}\right) }\right)$ 捕捉了流 $i$ 中分数衰减的速率，而项 $\frac{\partial F}{\partial {n}_{i}}$ 捕捉了流的评分谓词对总分数的贡献程度，这与快速合并算法类似。项 $\# {M}_{i}$ 是从流 $i$ 中读取数据时，其分数边界可能会受到影响的前 $k$ 对象的数量，这种影响是通过降低它们的分数上限或了解它们的精确分数来实现的。在每一步，都会选择 ${\Delta }_{i}$ 最大的流来获取下一个对象。

The NRA algorithm has been also studied in Mamoulis et al. [2006] under various application requirements. The presented techniques rely on the observation that, at some stage during NRA processing, it is not useful to keep track of up-to-date score upper bounds. Instead, the updates to these upper bounds can be deferred to a later step, or can be reduced to a much more compact set of necessary updates for more efficient computation. An NRA variant, called LARA, has been introduced based on a lattice structure that keeps a leader object for each subset of the ranked inputs. These leader objects provide score upper bounds for objects that have not been seen yet on their corresponding inputs. The top- $k$ processing algorithm proceeds in two successive phases:

Mamoulis 等人 [2006 年] 还在各种应用需求下对 NRA 算法进行了研究。所提出的技术基于这样的观察：在 NRA 处理的某个阶段，跟踪最新的分数上限是没有用的。相反，可以将这些上限的更新推迟到后续步骤，或者将其简化为一组更紧凑的必要更新，以实现更高效的计算。基于一种格结构引入了一种名为 LARA 的 NRA 变体，该结构为每个排序输入的子集保留一个领导者对象。这些领导者对象为其相应输入上尚未出现的对象提供分数上限。前 $k$ 处理算法分两个连续阶段进行：

- A growing phase. Ranked inputs are sequentially scanned to compose a candidate set. The seen objects in different inputs are added to the candidate set. A set ${W}_{k}$ , containing the $k$ objects with highest score lower bounds,is remembered at each step. The candidate set construction is complete when the threshold value (the score upper bound of any unseen object) is below the minimum score of ${W}_{k}$ . At this point, we are sure that the top- $k$ query answer belongs to the candidate set.

- 增长阶段。按顺序扫描排序输入以组成候选集。将不同输入中已出现的对象添加到候选集中。在每一步都会记住一个集合 ${W}_{k}$，该集合包含分数下限最高的 $k$ 个对象。当阈值（任何未出现对象的分数上限）低于 ${W}_{k}$ 的最小分数时，候选集的构建完成。此时，我们可以确定前 $k$ 查询的答案属于候选集。

—A shrinking phase. Materialized top- $k$ candidates are pruned gradually,by computing their score upper bounds,until the final top- $k$ answer is obtained.

—收缩阶段。通过计算物化的前 $k$ 候选对象的分数上限，逐步对其进行修剪，直到获得最终的前 $k$ 答案。

Score upper bound computation makes use of the lattice to minimize the number of required accesses to the ranked inputs by eliminating the need to access some inputs once they become useless for future operations. Different adaptation of LARA in various settings have been proposed including providing answers online or incrementally, processing rank join queries, and working with different rank aggregation functions.

分数上限计算利用格结构，通过消除对某些输入的访问需求（一旦这些输入对未来操作无用）来最小化对排序输入的所需访问次数。已经提出了 LARA 在各种设置下的不同变体，包括在线或增量式提供答案、处理排名连接查询以及使用不同的排名聚合函数。

Another example of no random access top- $k$ algorithms is the ${J}^{ * }$ algorithm [Natsev et al. 2001]. The ${J}^{ * }$ algorithm adopts a top- $k$ join query model (Section 2.1),where the top- $k$ join results are computed by joining multiple ranked inputs based on a join condition, and scoring the outcome join results based on a monotone score aggregation function. The ${J}^{ * }$ algorithm is based on the ${\mathcal{A}}^{ * }$ search algorithm. The idea is to maintain a priority queue of partial and complete join combinations, ordered on the upper bounds of their total scores. At each step, the algorithm tries to complete the join combination at queue top by selecting the next input stream to join with the partial join result, and retrieving the next object from that stream. The algorithm reports the next top join result as soon as the join result at queue top includes an object from each ranked input.

无随机访问的前 $k$ 算法的另一个例子是 ${J}^{ * }$ 算法（[Natsev 等人，2001 年]）。${J}^{ * }$ 算法采用前 $k$ 连接查询模型（第 2.1 节），其中前 $k$ 连接结果是通过基于连接条件连接多个排序输入，并根据单调得分聚合函数对结果连接进行评分来计算的。${J}^{ * }$ 算法基于 ${\mathcal{A}}^{ * }$ 搜索算法。其思路是维护一个部分和完整连接组合的优先队列，按其总得分的上限排序。在每一步，算法尝试通过选择下一个输入流与部分连接结果进行连接，并从该流中检索下一个对象，来完成队列顶部的连接组合。一旦队列顶部的连接结果包含每个排序输入中的一个对象，算法就会报告下一个前连接结果。

For each input stream, a variable is defined whose possible assignments are the set of stream objects. A state is defined as a set of variable assignments, and a state is complete if it instantiates all variables. The problem of finding a valid join combination with maximum score reduces to finding an assignment for all the variables, based on join condition, that maximizes the overall score. The score of a state is computed by exploiting the monotonicity of the score aggregation function. That is, the scores of complete states are computed by aggregating the scores of their instantiated variables, while the scores of incomplete states are computed by aggregating the scores of their instantiated variables, and the score upper bounds of their noninstantiated variables. The score upper bounds of noninstantiated variables are equal to the last seen scores in the corresponding ranked inputs.

对于每个输入流，定义一个变量，其可能的赋值是流对象的集合。一个状态被定义为一组变量赋值，如果它实例化了所有变量，则该状态是完整的。找到具有最大得分的有效连接组合的问题可以简化为基于连接条件为所有变量找到一个赋值，以使总体得分最大化。状态的得分是通过利用得分聚合函数的单调性来计算的。也就是说，完整状态的得分是通过聚合其实例化变量的得分来计算的，而不完整状态的得分是通过聚合其实例化变量的得分以及其未实例化变量的得分上限来计算的。未实例化变量的得分上限等于相应排序输入中最后看到的得分。

### 3.3. Sorted Access with Controlled Random Probes

### 3.3. 带受控随机探查的有序访问

Top- $k$ processing methods in this category assume that at least one source provides sorted access, while random accesses are scheduled to be performed only when needed. The Upper and Pick algorithms [Bruno et al. 2002b; Marian et al. 2004] are examples of these methods.

此类中的前 $k$ 处理方法假设至少有一个源提供有序访问，而随机访问仅在需要时才安排执行。Upper 和 Pick 算法（[Bruno 等人，2002b；Marian 等人，2004]）就是这些方法的例子。

The Upper and Pick algorithms are proposed in the context of Web-accessible sources. Such sources usually have large variation in the allowed access methods, and their costs. Upper and Pick assume that each source can provide a sorted and/or random access to its ranked input, and that at least one source supports sorted access. The main purpose of having at least one sorted-access source is to obtain an initial set of candidate objects. Random accesses are controlled by selecting the best candidates, based on score upper bounds, to complete their scores.

Upper 和 Pick 算法是在可通过 Web 访问的源的背景下提出的。此类源在允许的访问方法及其成本方面通常有很大差异。Upper 和 Pick 假设每个源可以对其排序输入提供有序和/或随机访问，并且至少有一个源支持有序访问。至少有一个有序访问源的主要目的是获得一组初始候选对象。随机访问通过基于得分上限选择最佳候选对象来完成其得分进行控制。

Three different types of sources are defined based on the supported access method: (1) $S$ -Source that provides sorted access,(2) $R$ -Source that provides random access,and (3) SR-Source that provides both access methods. The initial candidate set is obtained using at least one S-Source. Other R-Sources are probed to get the required partial scores as required.

根据支持的访问方法定义了三种不同类型的源：（1）提供有序访问的 $S$ 源，（2）提供随机访问的 $R$ 源，以及（3）提供两种访问方法的 SR 源。使用至少一个 S 源获得初始候选集。根据需要探查其他 R 源以获得所需的部分得分。

<!-- Media -->

Algorithm 3. Upper [Bruno et al. 2002b]

算法 3. Upper（[Bruno 等人，2002b]）

---

1: Define Candidates as priority queue based on $\overline{F}\left( .\right)$

1: 将候选对象定义为基于 $\overline{F}\left( .\right)$ 的优先队列

$: T = 1$ \{Score upper bound for all unseen tuples\}

$: T = 1$ {所有未见过的元组的得分上限}

: returned = 0

: 返回数量 = 0

while returned $< k$ do

当返回数量 $< k$ 时

			if Candidates $\neq  \phi$ then

			if 候选对象 $\neq  \phi$ 则

				select from Candidates the object ${t}_{\text{top }}$ with the maximum $\bar{F}\left( \text{.}\right)$

				从候选对象中选择具有最大 $\bar{F}\left( \text{.}\right)$ 的对象 ${t}_{\text{top }}$

			else

			else

				${t}_{\text{top }}$ is undefined

				${t}_{\text{top }}$ 未定义

		end if

		end if

			if ${t}_{top}$ is undefined or $\bar{F}\left( {t}_{top}\right)  < T$ then

			如果 ${t}_{top}$ 未定义或 $\bar{F}\left( {t}_{top}\right)  < T$，则

				Use a round-robin policy to choose the next sorted list ${L}_{i}$ to access.

				使用轮询策略选择下一个要访问的已排序列表 ${L}_{i}$。

				$t = {L}_{i}$ .getNext(   )

				$t = {L}_{i}$ .getNext(   )

				if $t$ is new object then

				如果 $t$ 是新对象，则

							Add $t$ to Candidates

							将 $t$ 添加到候选列表中

				else

				否则

							Update $\bar{F}\left( t\right)$ ,and update Candidates accordingly

							更新 $\bar{F}\left( t\right)$，并相应地更新候选列表

			end if

			结束条件判断

				$T = F\left( {{\bar{p}}_{1},\ldots ,{\bar{p}}_{m}}\right)$

		else if $F\left( {t}_{\text{top }}\right)$ is completely known then

		否则，如果 $F\left( {t}_{\text{top }}\right)$ 已完全明确，则

				Report $\left( {{t}_{top},F\left( {t}_{top}\right) }\right)$

				报告 $\left( {{t}_{top},F\left( {t}_{top}\right) }\right)$

				Remove ${t}_{\text{top }}$ from Candidates

				从候选列表中移除 ${t}_{\text{top }}$

				returned = returned + 1

				返回数量 = 返回数量 + 1

		else

		否则

				${L}_{i} =$ SelectBestSource $\left( {t}_{\text{top }}\right)$

				${L}_{i} =$ 选择最佳源 $\left( {t}_{\text{top }}\right)$

				Update $\bar{F}\left( {t}_{\text{top }}\right)$ with the value of predicate ${p}_{i}$ via random probe to ${L}_{i}$

				通过对 ${L}_{i}$ 进行随机探测，用谓词 ${p}_{i}$ 的值更新 $\bar{F}\left( {t}_{\text{top }}\right)$

		end if

		结束条件判断

	end while

	结束循环

---

<!-- Media -->

The Upper algorithm, as illustrated by Algorithm 3, probes objects that have considerable chances to be among the top- $k$ objects. In Algorithm 3,it is assumed that objects’ scores are normalized in the range $\left\lbrack  {0,1}\right\rbrack$ . Candidate objects are retrieved first from sorted sources, and inserted into a priority queue based on their score upper bounds. The upper bound of unseen objects is updated when new objects are retrieved from sorted sources. An object is reported and removed from the queue when its exact score is higher than the score upper bound of unseen objects. The algorithm repeatedly selects the best source to probe next to obtain additional information for candidate objects. The selection is performed by the SelectBestSource function. This function could have several implementations. For example, the source to be probed next can be the one which contributes the most in decreasing the uncertainty about the top candidates.

如算法3所示，上限算法（Upper algorithm）会探查那些极有可能跻身前 $k$ 个的对象。在算法3中，假设对象的分数已被归一化到范围 $\left\lbrack  {0,1}\right\rbrack$ 内。首先从排序好的数据源中检索候选对象，并根据它们的分数上限将其插入优先队列。当从排序好的数据源中检索到新对象时，未探查对象的上限会被更新。当一个对象的精确分数高于未探查对象的分数上限时，该对象会被报告并从队列中移除。该算法会反复选择下一个要探查的最佳数据源，以获取候选对象的额外信息。选择操作由SelectBestSource函数执行。这个函数可以有多种实现方式。例如，下一个要探查的数据源可以是在降低前几名候选对象的不确定性方面贡献最大的那个。

In the Pick algorithm, the next object to be probed is selected so that it minimizes a distance function, which is defined as the sum of the differences between the upper and lower bounds of all objects. The source to be probed next is selected at random from all sources that need to be probed to complete the score of the selected object.

在挑选算法（Pick algorithm）中，选择下一个要探查的对象，使其能最小化一个距离函数，该函数被定义为所有对象的上限和下限之差的总和。下一个要探查的数据源是从所有需要探查以完成所选对象分数计算的数据源中随机选择的。

A related issue to controlling the number of random accesses is the potentially expensive evaluation of ranking predicates. The full evaluation of user-defined ranking predicates is not always tolerable. One reason is that user-defined ranking predicates are usually defined only at at query time, limiting the chances of benefiting from precompu-tations. Another reason is that ranking predicates might access external autonomous sources. In these settings, optimizing the number of times the ranking predicates are invoked is necessary for efficient query processing.

控制随机访问次数的一个相关问题是对排序谓词的评估可能代价高昂。对用户定义的排序谓词进行全面评估并不总是可行的。一个原因是用户定义的排序谓词通常只在查询时定义，这限制了从预计算中受益的机会。另一个原因是排序谓词可能会访问外部自主数据源。在这些情况下，为了实现高效的查询处理，有必要优化排序谓词的调用次数。

<!-- Media -->

Table II. Object Scores Based on Different Ranking Predicates

表二. 基于不同排序谓词的对象分数

<table><tr><td>Object</td><td>${p}_{1}$</td><td>${p}_{2}$</td><td>${p}_{3}$</td><td>$F = {p}_{1} + {p}_{2} + {p}_{3}$</td></tr><tr><td>a</td><td>0.9</td><td>1.0</td><td>0.5</td><td>2.4</td></tr><tr><td>b</td><td>0.6</td><td>0.4</td><td>0.4</td><td>1.4</td></tr><tr><td>C</td><td>0.4</td><td>0.7</td><td>0.9</td><td>2.0</td></tr><tr><td>d</td><td>0.3</td><td>0.3</td><td>0.5</td><td>1.1</td></tr><tr><td>e</td><td>0.2</td><td>0.4</td><td>0.2</td><td>0.8</td></tr></table>

<table><tbody><tr><td>对象</td><td>${p}_{1}$</td><td>${p}_{2}$</td><td>${p}_{3}$</td><td>$F = {p}_{1} + {p}_{2} + {p}_{3}$</td></tr><tr><td>a</td><td>0.9</td><td>1.0</td><td>0.5</td><td>2.4</td></tr><tr><td>b</td><td>0.6</td><td>0.4</td><td>0.4</td><td>1.4</td></tr><tr><td>C</td><td>0.4</td><td>0.7</td><td>0.9</td><td>2.0</td></tr><tr><td>d</td><td>0.3</td><td>0.3</td><td>0.5</td><td>1.1</td></tr><tr><td>e</td><td>0.2</td><td>0.4</td><td>0.2</td><td>0.8</td></tr></tbody></table>

Table III. Finding the Top-2 Objects in MPro

表三。在MPro中找出前两名对象

<table><tr><td>Step</td><td>Action</td><td>Priority queue</td><td>Output</td></tr><tr><td>1</td><td>Initialize</td><td>$\left\lbrack  {a : {2.9},b : {2.6},c : {2.4},d : {2.3},e : {2.2}}\right\rbrack$</td><td>\{\}</td></tr><tr><td>2</td><td>After probe $\left( {a,{p}_{2}}\right)$</td><td>$\left\lbrack  {a : {2.9},b : {2.6},c : {2.4},d : {2.3},e : {2.2}}\right\rbrack$</td><td>\{\}</td></tr><tr><td>3</td><td>After probe $\left( {a,{p}_{3}}\right)$</td><td>$\left\lbrack  {b : {2.6},a : {2.4},c : {2.4},d : {2.3},e : {2.2}}\right\rbrack$</td><td>\{\}</td></tr><tr><td>4</td><td>After probe $\left( {b,{p}_{2}}\right)$</td><td>$\left\lbrack  {c : {2.4},d : {2.3},e : {2.2},b : {2.0}}\right\rbrack$</td><td>$\{$ a:2.4 $\}$</td></tr><tr><td>5</td><td>After probe $\left( {c,{p}_{2}}\right)$</td><td>$\left\lbrack  {d : {2.3},e : {2.2},c : {2.1},b : {2.0}}\right\rbrack$</td><td>\{a:2.4\}</td></tr><tr><td>6</td><td>After probe $\left( {d,{p}_{2}}\right)$</td><td>$\left\lbrack  {e : {2.2},c : {2.1},b : {2.0},d : {1.6}}\right\rbrack$</td><td>\{a:2.4\}</td></tr><tr><td>7</td><td>After probe $\left( {e,{p}_{2}}\right)$</td><td>$\left\lbrack  {c : {2.1},b : {2.0},e : {1.6},d : {1.6}}\right\rbrack$</td><td>$\{$ a:2.4 $\}$</td></tr><tr><td>8</td><td>After probe $\left( {c,{p}_{3}}\right)$</td><td>$\left\lbrack  {b : {2.0},e : {1.6},d : {1.6}}\right\rbrack$</td><td>$\{ \mathrm{a} : {2.4},\mathrm{c} : {2.0}\}$</td></tr></table>

<table><tbody><tr><td>步骤</td><td>操作</td><td>优先队列</td><td>输出</td></tr><tr><td>1</td><td>初始化</td><td>$\left\lbrack  {a : {2.9},b : {2.6},c : {2.4},d : {2.3},e : {2.2}}\right\rbrack$</td><td>\{\}</td></tr><tr><td>2</td><td>探测 $\left( {a,{p}_{2}}\right)$ 之后</td><td>$\left\lbrack  {a : {2.9},b : {2.6},c : {2.4},d : {2.3},e : {2.2}}\right\rbrack$</td><td>\{\}</td></tr><tr><td>3</td><td>探测 $\left( {a,{p}_{3}}\right)$ 之后</td><td>$\left\lbrack  {b : {2.6},a : {2.4},c : {2.4},d : {2.3},e : {2.2}}\right\rbrack$</td><td>\{\}</td></tr><tr><td>4</td><td>探测 $\left( {b,{p}_{2}}\right)$ 之后</td><td>$\left\lbrack  {c : {2.4},d : {2.3},e : {2.2},b : {2.0}}\right\rbrack$</td><td>$\{$ a:2.4 $\}$</td></tr><tr><td>5</td><td>探测 $\left( {c,{p}_{2}}\right)$ 之后</td><td>$\left\lbrack  {d : {2.3},e : {2.2},c : {2.1},b : {2.0}}\right\rbrack$</td><td>\{a:2.4\}</td></tr><tr><td>6</td><td>探测 $\left( {d,{p}_{2}}\right)$ 之后</td><td>$\left\lbrack  {e : {2.2},c : {2.1},b : {2.0},d : {1.6}}\right\rbrack$</td><td>\{a:2.4\}</td></tr><tr><td>7</td><td>探测 $\left( {e,{p}_{2}}\right)$ 之后</td><td>$\left\lbrack  {c : {2.1},b : {2.0},e : {1.6},d : {1.6}}\right\rbrack$</td><td>$\{$ a:2.4 $\}$</td></tr><tr><td>8</td><td>探测 $\left( {c,{p}_{3}}\right)$ 之后</td><td>$\left\lbrack  {b : {2.0},e : {1.6},d : {1.6}}\right\rbrack$</td><td>$\{ \mathrm{a} : {2.4},\mathrm{c} : {2.0}\}$</td></tr></tbody></table>

<!-- Media -->

These observations motivated the work of Chang and Hwang [2002] and Hwang and Chang [2007b] which introduced the concept of "necessary probes," to indicate whether a predicate probe is absolutely required or not. The proposed Minimal Probing (MPro) algorithm adopts this concept to minimize the predicate evaluation cost. The authors considered a top- $k$ query with a scoring function $F$ defined on a set of predicates ${p}_{1}\cdots {p}_{n}$ . The score upper bound of an object $t$ ,denoted $\bar{F}\left( t\right)$ ,is computed by aggregating the scores produced by each of the evaluated predicates and assuming the maximum possible score for unevaluated predicates. The aggregation is done using a monotonic function,for example,weighted summation. Let $\operatorname{probe}\left( {t,p}\right)$ denote probing predicate $p$ of object $t$ . It has been shown that $\operatorname{probe}\left( {t,p}\right)$ is necessary if $t$ is among the current top- $k$ objects based on the score upper bounds. This observation has been exploited to construct probing schedules as sequences of necessary predicate probes.

这些观察结果推动了张（Chang）和黄（Hwang）[2002]以及黄和张[2007b]的研究工作，他们引入了“必要探测”的概念，用于表明一个谓词探测是否是绝对必要的。所提出的最小探测（MPro）算法采用了这一概念，以最小化谓词评估成本。作者考虑了一个前$k$查询，其评分函数$F$定义在一组谓词${p}_{1}\cdots {p}_{n}$上。对象$t$的得分上限，记为$\bar{F}\left( t\right)$，是通过聚合每个已评估谓词产生的得分，并假设未评估谓词的最大可能得分来计算的。聚合使用单调函数完成，例如加权求和。用$\operatorname{probe}\left( {t,p}\right)$表示对对象$t$的谓词$p$进行探测。已经证明，如果基于得分上限，$t$位于当前前$k$个对象之中，那么$\operatorname{probe}\left( {t,p}\right)$就是必要的。这一观察结果已被用于构建探测调度，将其作为一系列必要的谓词探测。

The MPro algorithm works in two phases. First, in the initialization phase, a priority queue is initialized based on the score upper bounds of different objects. The algorithm assumes the existence of at least one cheap predicate where sorted access is available. The initial score upper bound of each object is computed by aggregating the scores of the cheap predicates and the maximum scores of expensive predicates. Second, in the probing phase, the object at queue top is removed, its next unevaluated predicate is probed, its score upper bound is updated, and the object is reinserted back to the queue. If the score of the object at queue top is already complete,the object is among the top- $k$ answers and it is moved to the output queue. Finding the optimal probing schedule for each object is shown to be in NP-Hard complexity class. Optimal probing schedules are thus approximated using a greedy heuristic defined using the benefit and cost of each predicate, which are computed by sampling the ranked lists.

MPro算法分两个阶段工作。首先，在初始化阶段，根据不同对象的得分上限初始化一个优先队列。该算法假设至少存在一个可进行有序访问的低成本谓词。每个对象的初始得分上限是通过聚合低成本谓词的得分和高成本谓词的最大得分来计算的。其次，在探测阶段，移除队列顶部的对象，对其下一个未评估的谓词进行探测，更新其得分上限，然后将该对象重新插入队列。如果队列顶部对象的得分已经完整，那么该对象就是前$k$个答案之一，并将其移至输出队列。结果表明，为每个对象找到最优的探测调度属于NP难复杂度类。因此，使用基于每个谓词的收益和成本定义的贪心启发式方法来近似最优探测调度，这些收益和成本是通过对排序列表进行采样计算得到的。

We illustrate the processing of MPro using the next example. Table II shows the scores of one cheap predicate, ${p}_{1}$ ,and two expensive predicates, ${p}_{2}$ and ${p}_{3}$ ,for a list of objects $\{ a,b,c,d,e\}$ . Table III illustrates how MPro operates,based a scoring function $F = {p}_{1} + {p}_{2} + {p}_{3}$ ,to find the top-2 objects. We use the notation object:score to refer to the score upper bound of the object.

我们通过下一个示例来说明MPro的处理过程。表二显示了一个低成本谓词${p}_{1}$和两个高成本谓词${p}_{2}$和${p}_{3}$对于对象列表$\{ a,b,c,d,e\}$的得分。表三说明了MPro如何基于评分函数$F = {p}_{1} + {p}_{2} + {p}_{3}$来找出前2个对象。我们使用“对象:得分”的表示法来指代对象的得分上限。

The goal of the MPro algorithm is to minimize the cost of random access, while assuming the cost of sorted access is cheap. A generalized algorithm,named ${NC}$ ,has been introduced in Hwang and Chang [2007a] to include the cost of sorted access while scheduling predicates probing. The algorithm maintains the current top- $k$ objects based on their scores upper bounds. At each step, the algorithm identifies the set of necessary probing alternatives to determine the top- $k$ objects. These alternatives are probing the unknown predicates for the current top- $k$ objects using either sorted or random access. The authors proved that it is sufficient to consider the family of algorithms that perform sorted access before random access in order to achieve the optimal scheduling. Additionally, to provide practical scheduling of alternatives, the NC algorithm restricts the space of considered schedules to those that only perform sorted accesses up to certain depth $\Delta$ ,and follow a fixed schedule $\mathcal{H}$ for random accesses. The algorithm first attempts to perform a sorted access. If there is no sorted access among the probing alternatives,or all sorted accesses are beyond the depth $\Delta$ ,a random access takes place based on the schedule $\mathcal{H}$ . The parameters $\Delta$ and $\mathcal{H}$ are estimated using sampling.

MPro算法的目标是在假设有序访问成本较低的情况下，最小化随机访问的成本。黄和张[2007a]引入了一种名为${NC}$的通用算法，在调度谓词探测时考虑了有序访问的成本。该算法根据对象的得分上限维护当前的前$k$个对象。在每一步，算法确定一组必要的探测选项，以确定前$k$个对象。这些选项是使用有序访问或随机访问对当前前$k$个对象的未知谓词进行探测。作者证明，为了实现最优调度，考虑先进行有序访问再进行随机访问的算法族就足够了。此外，为了提供实际的选项调度，NC算法将考虑的调度空间限制为那些仅在达到一定深度$\Delta$之前进行有序访问，并遵循固定的随机访问调度$\mathcal{H}$的调度。该算法首先尝试进行有序访问。如果在探测选项中没有有序访问，或者所有有序访问都超出了深度$\Delta$，则根据调度$\mathcal{H}$进行随机访问。参数$\Delta$和$\mathcal{H}$是通过采样估计的。

## 4. IMPLEMENTATION LEVEL

## 4. 实现层面

In this section,we discuss top- $k$ processing methods based on the design choices they make regarding integration with database systems. Some techniques are designed as application-level solutions that work outside the database engine, while others involve low level modifications to the query engine. We describe techniques belonging to these two categories in the next sections.

在本节中，我们将根据与数据库系统集成的设计选择，讨论前$k$处理方法。一些技术被设计为在数据库引擎之外工作的应用级解决方案，而另一些则涉及对查询引擎的底层修改。我们将在接下来的小节中描述属于这两类的技术。

### 4.1. Application Level

### 4.1. 应用级别

Top- $k$ query processing techniques that are implemented at the application level,or middleware, provide a ranked retrieval of database objects, without major modification to the underlying database system, particularly the query engine. We classify application level top- $k$ techniques into Filter-Restart methods,and Indexes/Materialized Views methods.

在应用层或中间件实现的前 $k$ 查询处理技术，无需对底层数据库系统（尤其是查询引擎）进行重大修改，即可对数据库对象进行排序检索。我们将应用层的前 $k$ 技术分为过滤 - 重启方法和索引/物化视图方法。

4.1.1. Filter-Restart. Filter-Restart techniques formulate top- $k$ queries as range selection queries to limit the number of retrieved objects. That is,a top- $k$ query that ranks objects based on a scoring function $F$ ,defined on a set of scoring predicates ${p}_{1},\ldots ,{p}_{m}$ , is formulated as a range query of the form "find objects with ${p}_{1} > {T}_{1}$ and $\cdots$ and ${p}_{m} >$ ${T}_{m}$ ",where ${T}_{i}$ is an estimated cutoff threshold for predicate ${p}_{i}$ . Using a range query aims at limiting the retrieved set of objects to the necessary objects to answer the top- $k$ query. The retrieved objects have to be ranked based on $F$ to find the top- $k$ answers.

4.1.1. 过滤 - 重启。过滤 - 重启技术将前 $k$ 查询表述为范围选择查询，以限制检索对象的数量。也就是说，一个基于评分函数 $F$ 对对象进行排序的前 $k$ 查询（评分函数 $F$ 定义在一组评分谓词 ${p}_{1},\ldots ,{p}_{m}$ 上），被表述为形式为“查找满足 ${p}_{1} > {T}_{1}$ 且 $\cdots$ 且 ${p}_{m} >$ ${T}_{m}$ 的对象”的范围查询，其中 ${T}_{i}$ 是谓词 ${p}_{i}$ 的估计截断阈值。使用范围查询的目的是将检索到的对象集限制为回答前 $k$ 查询所需的对象。检索到的对象必须根据 $F$ 进行排序，以找到前 $k$ 个答案。

Incorrect estimation of cutoff threshold yields one of two possibilities: (1) if the cutoff is overestimated,the retrieved objects may not be sufficient to answer the top- $k$ query and the range query has to be restarted with looser thresholds, or (2) if the cutoff is under-estimated, the number of retrieved objects will be more than necessary to answer the top- $k$ query. In both cases,the performance of query processing degrades.

截断阈值估计错误会产生两种可能性之一：（1）如果高估了截断阈值，检索到的对象可能不足以回答前 $k$ 查询，必须以更宽松的阈值重新启动范围查询；或者（2）如果低估了截断阈值，检索到的对象数量将超过回答前 $k$ 查询所需的数量。在这两种情况下，查询处理的性能都会下降。

One proposed method to estimate the cutoff threshold is using the available statistics such as histograms [Bruno et al. 2002a], where the scoring function is taken as the distance between database objects and a given query point $q$ . Multidimensional histograms on objects' attributes (dimensions) are used to identify the cutoff distance from $q$ to the potential top- $k$ set. Two extreme strategies can be used to select such cutoff distance. The first strategy, named the restarts strategy, is to select the search distance as tight as possible to just enclose the potential top- $k$ objects. Such a strategy might retrieve less objects than the required number(k),necessitating restarting the search with a larger distance. The second strategy, named the no-restarts strategy, is to choose the search distance large enough to include all potential top- $k$ objects. However, this strategy may end up retrieving a large number of unnecessary objects.

一种估计截断阈值的建议方法是使用可用的统计信息，如直方图 [Bruno 等人，2002a]，其中评分函数被视为数据库对象与给定查询点 $q$ 之间的距离。对象属性（维度）上的多维直方图用于确定从 $q$ 到潜在前 $k$ 集的截断距离。可以使用两种极端策略来选择这样的截断距离。第一种策略称为重启策略，是尽可能紧密地选择搜索距离，刚好包围潜在的前 $k$ 个对象。这种策略可能检索到的对象数量少于所需数量（k），因此需要以更大的距离重新启动搜索。第二种策略称为无重启策略，是选择足够大的搜索距离，以包含所有潜在的前 $k$ 个对象。然而，这种策略最终可能会检索到大量不必要的对象。

<!-- Media -->

<!-- figureText: (0,50) (50,50) b3(15) (50,0) b1(40) b2(5) Restarts (0,0) No-restarts -->

<img src="https://cdn.noedgeai.com/0195c915-c519-7d98-b2ff-761eddc2c676_20.jpg?x=585&y=284&w=561&h=572&r=0"/>

Fig. 7. An example of restarts and no-restarts strategies in the Filter-Restart approach [Bruno et al. 2002a].

图 7. 过滤 - 重启方法中重启和无重启策略的示例 [Bruno 等人，2002a]。

<!-- Media -->

We illustrate the two strategies using Figure 7, where it is required to find the 10 closest objects to $q$ . The rectangular cells are two-dimensional histogram bins annotated with the number of data objects in each bin. The inner circle represents the restarts strategy where,hopefully,exactly 10 objects will be retrieved: five objects from bin ${b3}$ , and five objects from bin ${b2}$ . This strategy will result in restarts if less than 10 objects are retrieved. On the other hand, the no-restarts strategy uses the outer circle, which completely encloses bins ${b2}$ and ${b3}$ ,and thus ensures that at least 20 objects will be retrieved. However, this strategy will retrieve unnecessary objects.

我们使用图 7 来说明这两种策略，图中需要找到距离 $q$ 最近的 10 个对象。矩形单元格是二维直方图区间，每个区间标注了其中的数据对象数量。内圆表示重启策略，希望恰好能检索到 10 个对象：从区间 ${b3}$ 中检索 5 个对象，从区间 ${b2}$ 中检索 5 个对象。如果检索到的对象少于 10 个，这种策略将导致重启。另一方面，无重启策略使用外圆，它完全包围了区间 ${b2}$ 和 ${b3}$，因此确保至少能检索到 20 个对象。然而，这种策略会检索到不必要的对象。

To find the optimal search distance, query workload is used as a training set to determine the number of returned objects for different search distances and $q$ locations. The optimal search distance is approximated using an optimization algorithm running over all the queries in the workload. The outcome of the optimization algorithm is a search distance that is expected to minimize the overall number of retrieved objects, and the number of restarts.

为了找到最优搜索距离，将查询工作负载用作训练集，以确定不同搜索距离和 $q$ 位置下返回的对象数量。使用在工作负载中的所有查询上运行的优化算法来近似最优搜索距离。优化算法的结果是一个预计能使检索到的对象总数和重启次数最小化的搜索距离。

A probabilistic approach to estimate cutoff threshold was proposed by Donjerkovic and Ramakrishnan [1999]. A top- $k$ query based on an attribute $X$ is mapped into a selection predicate ${\sigma }_{X > T}$ ,where $T$ is the estimated cutoff threshold. A probabilistic model is used to search for the selection predicate that would minimize the overall expected cost of restarts. This is done by constructing a probability distribution over the cardinalities of possible selection predicates in the form of (cardinality-value, probability) pairs, where the cardinality-value represents the number of database tuples satisfying the predicate, and the probability represents the degree of certainty in the correctness of the cardinality-value, which reflects the potentially imperfect statistics on the underlying data.

Donjerkovic 和 Ramakrishnan [1999] 提出了一种估计截断阈值的概率方法。基于属性 $X$ 的前 $k$ 查询被映射到一个选择谓词 ${\sigma }_{X > T}$，其中 $T$ 是估计的截断阈值。使用概率模型来搜索能使重启的总体预期成本最小化的选择谓词。这是通过以（基数 - 值，概率）对的形式构建可能的选择谓词的基数上的概率分布来实现的，其中基数 - 值表示满足谓词的数据库元组的数量，概率表示基数 - 值正确性的确定程度，这反映了底层数据上可能不完美的统计信息。

The probability distribution is represented as a vector of equi-probable cardinality points to avoid materializing the whole space. Every cardinality point is associated with a cost estimate representing the initial query processing cost, and the cost of possible restart. The goal is to find a query plan that minimizes the expected cost over all cardinality values. To be consistent with the existing cardinality estimates, the cardinality distribution has an average equal to the average cardinality obtained from existing histograms. The maintained cardinality values are selected at equi-distant steps to the left and right of the average cardinality, with a predetermined total number of points. The stepping distance, to the left and right of average point, is estimated based on the worst case estimation error of the histogram.

为避免对整个空间进行实例化，概率分布表示为等概率基数点的向量。每个基数点都与一个成本估计相关联，该成本估计代表初始查询处理成本以及可能的重启成本。目标是找到一个查询计划，使所有基数取值上的预期成本最小化。为了与现有的基数估计保持一致，基数分布的平均值等于从现有直方图中获得的平均基数。所维护的基数取值是在平均基数左右以等距步长选取的，且点的总数是预先确定的。平均点左右的步长距离是根据直方图的最坏情况估计误差来估算的。

<!-- Media -->

<!-- figureText: ${p}_{2}$ Layer 1 Layer 2 Layer 3 p1 -->

<img src="https://cdn.noedgeai.com/0195c915-c519-7d98-b2ff-761eddc2c676_21.jpg?x=584&y=282&w=569&h=457&r=0"/>

Fig. 8. Convex hulls in two-dimensional space.

图8. 二维空间中的凸包。

<!-- Media -->

4.1.2. Using Indexes and Materialized Views. Another group of application level top- $k$ processing techniques use specialized indexes and materialized views to improve the query response time at the expense of additional storage space. Top- $k$ indexes usually make use of special geometric properties of the scoring function to index the underlying data objects. On the other hand, materialized views maintain information that is expensive to gather online, for example, a sorting for the underlying objects based on some scoring function,to help compute top- $k$ queries efficiently.

4.1.2. 使用索引和物化视图。另一组应用层的前$k$处理技术使用专门的索引和物化视图，以额外的存储空间为代价来提高查询响应时间。前$k$索引通常利用评分函数的特殊几何性质对底层数据对象进行索引。另一方面，物化视图维护在线收集成本较高的信息，例如，基于某个评分函数对底层对象进行排序，以帮助高效地计算前$k$查询。

4.1.2.1. Specialized Top-k Indexes. One example of specialized top- $k$ indexes is the Onion Indices [Chang et al. 2000]. Assume that tuples are represented as $n$ -dimensional points where each dimension represents the value of one scoring predicate. The convex $\begin{matrix} {hull} & {{of}\;{these}\;{points}\;{is}\;{defined}\;{as}\;{the}\;{boundary}\;{of}\;{the}\;{smallest}\;{convex}\;{region}\;{that}\;{encloses}} \end{matrix}$ them. The geometrical properties of the convex hull guarantee that it includes the top-1 object (assuming a linear scoring function defined on the dimensions). Onion Indices extend this observation by constructing layered convex hulls, shown in Figure 8, to index the underlying objects for efficient top- $k$ processing.

4.1.2.1. 专门的前k索引。专门的前$k$索引的一个例子是洋葱索引（Onion Indices）[Chang等人，2000年]。假设元组表示为$n$维点，其中每个维度代表一个评分谓词的值。它们的凸$\begin{matrix} {hull} & {{of}\;{these}\;{points}\;{is}\;{defined}\;{as}\;{the}\;{boundary}\;{of}\;{the}\;{smallest}\;{convex}\;{region}\;{that}\;{encloses}} \end{matrix}$。凸包的几何性质保证它包含前1对象（假设在这些维度上定义了线性评分函数）。洋葱索引通过构建分层凸包（如图8所示）来扩展这一观察结果，以便对底层对象进行索引，从而实现高效的前$k$处理。

The Onion Indices return the top-1 object by searching the points of the outmost convex hull. The next result (the top-2 object) is found by searching the remaining points of the outmost convex hull, and the points belonging to the next layer. This procedure continues until all of the top- $k$ results are found. Although this indexing scheme provides performance gain,it becomes inefficient when the top- $k$ query involves additional constraints on the required data, such as range predicates on attribute values. This is because the convex hull structure will be different for each set of constraints. A proposed work-around is to divide data objects into smaller clusters, index them, and merge these clusters into larger ones progressively. The result is a hierarchical structure of clusters, each of which has its own Onion Indices. A constrained query can probably be answered by indices of smaller clusters in the hierarchy. The construction of Onion Indices has an asymptotic complexity of $O\left( {n}^{d/2}\right)$ ,where $d$ is the number of dimensions and $n$ is the number of data objects.

洋葱索引通过搜索最外层凸包的点来返回前1对象。下一个结果（前2对象）是通过搜索最外层凸包的剩余点以及属于下一层的点来找到的。这个过程一直持续到找到所有的前$k$结果。尽管这种索引方案能带来性能提升，但当前$k$查询对所需数据有额外约束时，例如对属性值的范围谓词，它就会变得低效。这是因为对于每组约束，凸包结构都会不同。一种建议的解决方法是将数据对象划分为较小的簇，对它们进行索引，然后逐步将这些簇合并成更大的簇。结果是形成一个簇的层次结构，每个簇都有自己的洋葱索引。有约束的查询可能可以通过层次结构中较小簇的索引来回答。洋葱索引的构建具有$O\left( {n}^{d/2}\right)$的渐近复杂度，其中$d$是维度数，$n$是数据对象的数量。

The idea of multilayer indexing has been also adopted by Xin et al. [2006] to provide robust indexing for top- $k$ queries. Robustness is defined in terms of providing the best possible performance in worst case scenario,which is fully scanning the first $k$ layers to find the top- $k$ answers. The main idea is that if each object ${o}_{i}$ is pushed to the deepest possible layer, its retrieval can be avoided if it is unnecessary. This is accomplished by searching for the minimum rank of each object ${o}_{i}$ in all linear scoring functions. Such rank represents the layer number,denoted ${l}^{ * }\left( {o}_{i}\right)$ ,where object ${o}_{i}$ is pushed to. For $n$ objects having $d$ scoring predicates,computing the exact layer numbers for all objects has a complexity of $O\left( {{n}^{d}\log n}\right.$ ),which is an overkill when $n$ or $d$ are large. Approximation is used to reduce the computation cost. An approximate layer number, denoted $l\left( {o}_{i}\right)$ ,is computed such that $l\left( {o}_{i}\right)  \leq  {l}^{ * }\left( {o}_{i}\right)$ ,which ensures that no false positives are produced in the top- $k$ query answer. The complexity of the approximation algorithm is $O\left( {{2}^{d}n{\left( \log n\right) }^{r\left( d\right)  - 1}}\right)$ ,where $r\left( d\right)  = \lceil \frac{d}{2}\rceil  + \lfloor \frac{d}{2}\rfloor \lceil \frac{d}{2}\rceil$ .

多层索引的思想也被辛等人 [2006] 采用，用于为前 $k$ 查询提供鲁棒的索引。鲁棒性是根据在最坏情况下提供尽可能好的性能来定义的，即完全扫描前 $k$ 层以找到前 $k$ 个答案。主要思想是，如果将每个对象 ${o}_{i}$ 推到尽可能深的层，那么在不必要的情况下可以避免对其进行检索。这是通过在所有线性评分函数中搜索每个对象 ${o}_{i}$ 的最小排名来实现的。这样的排名代表了层号，用 ${l}^{ * }\left( {o}_{i}\right)$ 表示，对象 ${o}_{i}$ 被推到该层。对于具有 $d$ 个评分谓词的 $n$ 个对象，计算所有对象的确切层号的复杂度为 $O\left( {{n}^{d}\log n}\right.$ )，当 $n$ 或 $d$ 很大时，这是过度计算。使用近似方法来降低计算成本。计算一个近似层号，用 $l\left( {o}_{i}\right)$ 表示，使得 $l\left( {o}_{i}\right)  \leq  {l}^{ * }\left( {o}_{i}\right)$ ，这确保了在前 $k$ 查询答案中不会产生误报。近似算法的复杂度为 $O\left( {{2}^{d}n{\left( \log n\right) }^{r\left( d\right)  - 1}}\right)$ ，其中 $r\left( d\right)  = \lceil \frac{d}{2}\rceil  + \lfloor \frac{d}{2}\rfloor \lceil \frac{d}{2}\rceil$ 。

<!-- Media -->

<!-- figureText: ${p}_{2}$ ${P}_{2}$ (b) ${P}_{1}$ + W ${\mathrm{X}}_{1}$ ${P}_{1}$ (a) -->

<img src="https://cdn.noedgeai.com/0195c915-c519-7d98-b2ff-761eddc2c676_22.jpg?x=445&y=286&w=841&h=425&r=0"/>

Fig. 9. Geometric representation of tuples and scoring function: (a) projection of tuple $t = \left( {{x}_{1},{x}_{2}}\right)$ on scoring function vector $\left( {{w}_{1},{w}_{2}}\right)$ ; (b) order based on obtained scores.

图 9. 元组和评分函数的几何表示：(a) 元组 $t = \left( {{x}_{1},{x}_{2}}\right)$ 在评分函数向量 $\left( {{w}_{1},{w}_{2}}\right)$ 上的投影；(b) 基于获得的分数的顺序。

<!-- Media -->

Ranked Join Indices [Tsaparas et al. 2003] is another top- $k$ index structure,based on the observation that the projection of a vector representing a tuple $t$ on the normalized scoring function vector $\overrightarrow{F}$ reveals $t$ ’s rank based on $F$ . This observation applies to any scoring function that is defined as a linear combination of the scoring predicates. For example,Figure 9 shows a scoring function $F = {w}_{1}.{p}_{1} + {w}_{2}.{p}_{2}$ ,where ${p}_{1}$ and ${p}_{2}$ are scoring predicates,and ${w}_{1}$ and ${w}_{2}$ are their corresponding weights. In this case,we have $\overrightarrow{F} = \left( {{w}_{1},{w}_{2}}\right)$ . Without loss of generality,assume that $\parallel \overrightarrow{F}\parallel  = 1$ . We can obtain the score of $t = \left( {{x}_{1},{x}_{2}}\right)$ by computing the length of its projection on $\overrightarrow{F}$ ,which is equivalent to the dot product $\left( {{w}_{1},{w}_{2}}\right)  \odot  \left( {{x}_{1},{x}_{2}}\right)  = {w}_{1}.{x}_{1} + {w}_{2}.{x}_{2}$ . By changing the values of ${w}_{1}$ and ${w}_{2}$ ,we can sweep the space using a vector of increasing angle to represent any possible linear scoring function. The tuple scores given by an arbitrary linear scoring function can thus be materialized.

排名连接索引 [察帕拉斯等人 2003] 是另一种前 $k$ 索引结构，其基于这样的观察：表示元组 $t$ 的向量在归一化评分函数向量 $\overrightarrow{F}$ 上的投影揭示了 $t$ 基于 $F$ 的排名。这一观察适用于任何定义为评分谓词线性组合的评分函数。例如，图 9 展示了一个评分函数 $F = {w}_{1}.{p}_{1} + {w}_{2}.{p}_{2}$ ，其中 ${p}_{1}$ 和 ${p}_{2}$ 是评分谓词， ${w}_{1}$ 和 ${w}_{2}$ 是它们对应的权重。在这种情况下，我们有 $\overrightarrow{F} = \left( {{w}_{1},{w}_{2}}\right)$ 。不失一般性，假设 $\parallel \overrightarrow{F}\parallel  = 1$ 。我们可以通过计算 $t = \left( {{x}_{1},{x}_{2}}\right)$ 在 $\overrightarrow{F}$ 上的投影长度来获得其分数，这等同于点积 $\left( {{w}_{1},{w}_{2}}\right)  \odot  \left( {{x}_{1},{x}_{2}}\right)  = {w}_{1}.{x}_{1} + {w}_{2}.{x}_{2}$ 。通过改变 ${w}_{1}$ 和 ${w}_{2}$ 的值，我们可以使用一个角度逐渐增大的向量来扫描空间，以表示任何可能的线性评分函数。因此，可以实现由任意线性评分函数给出的元组分数。

Before materialization,tuples that are dominated by more than $k$ tuples are discarded because they do not belong to the top- $k$ query answer of any linear scoring function. The remaining tuples,denoted as the dominating set ${\mathcal{D}}_{k}$ ,include all possible top- $k$ answers for any possible linear scoring function. Algorithm 4 describes how to construct the dominating set ${\mathcal{D}}_{k}$ with respect to a scoring function that is defined as a linear combination of predicates ${p}_{1}$ and ${p}_{2}$ . The algorithm starts by first sorting all the tuples based on ${p}_{1}$ ,and then scanning the sorted tuples. A priority queue $Q$ is maintained to keep the top- $k$ tuples,encountered so far,based on predicate ${p}_{2}$ . The first $k$ tuples are directly copied to ${\mathcal{D}}_{k}$ ,while subsequent tuples are examined against the minimum value of ${p}_{2}$ in $Q$ . If the ${p}_{2}$ value of some tuple $t$ is less than the minimum ${p}_{2}$ value in $Q$ ,then $t$ is discarded,since there are at least $k$ objects with greater ${p}_{1}$ and ${p}_{2}$ values.

在物化之前，被超过 $k$ 个元组支配的元组会被丢弃，因为它们不属于任何线性评分函数的前 $k$ 查询答案。其余的元组，记为支配集 ${\mathcal{D}}_{k}$，包含了任何可能的线性评分函数的所有可能的前 $k$ 答案。算法 4 描述了如何针对一个被定义为谓词 ${p}_{1}$ 和 ${p}_{2}$ 的线性组合的评分函数来构建支配集 ${\mathcal{D}}_{k}$。该算法首先根据 ${p}_{1}$ 对所有元组进行排序，然后扫描排序后的元组。维护一个优先队列 $Q$ 以基于谓词 ${p}_{2}$ 保存到目前为止遇到的前 $k$ 个元组。前 $k$ 个元组直接复制到 ${\mathcal{D}}_{k}$ 中，而后续的元组会与 $Q$ 中 ${p}_{2}$ 的最小值进行比较。如果某个元组 $t$ 的 ${p}_{2}$ 值小于 $Q$ 中的 ${p}_{2}$ 最小值，那么 $t$ 会被丢弃，因为至少有 $k$ 个对象的 ${p}_{1}$ 和 ${p}_{2}$ 值更大。

<!-- Media -->

Algorithm 4. Ranked Join Indices: GetDominatingSet [Tsaparas et al. 2003]

算法 4. 排序连接索引：获取支配集 [察帕拉斯（Tsaparas）等人，2003 年]

---

1: Define $Q$ as a priority queue based on ${p}_{2}$ values

1: 将 $Q$ 定义为一个基于 ${p}_{2}$ 值的优先队列

2: Define ${\mathcal{D}}_{k}$ as the dominating set. Initially,set ${\mathcal{D}}_{k} = \phi$

2: 将 ${\mathcal{D}}_{k}$ 定义为支配集。初始时，设置 ${\mathcal{D}}_{k} = \phi$

3: Sort tuples in non-increasing order of ${p}_{1}$ values.

3: 按照 ${p}_{1}$ 值的非递增顺序对元组进行排序。

	: for each tuple ${t}_{i}$ do

	: 对每个元组 ${t}_{i}$ 执行以下操作

					if $\left| Q\right|  < k$ then

					if $\left| Q\right|  < k$ 则

									${\mathcal{D}}_{k} = {\mathcal{D}}_{k} \cup  {t}_{i}$

									insert $\left( {{t}_{i},{p}_{2}\left( {t}_{i}\right) }\right)$ in $Q$

																	 将 $\left( {{t}_{i},{p}_{2}\left( {t}_{i}\right) }\right)$ 插入 $Q$ 中

			else if ${p}_{2}\left( {t}_{i}\right)  \leq  \left( \right.$ minimum $\left. {{p}_{2}\text{value in}Q}\right)$ then

			else if ${p}_{2}\left( {t}_{i}\right)  \leq  \left( \right.$ 小于 $\left. {{p}_{2}\text{value in}Q}\right)$ 的最小值 则

							skip ${t}_{i}$

													 跳过 ${t}_{i}$

			else

			else

							${\mathcal{D}}_{k} = {\mathcal{D}}_{k} \cup  {t}_{i}$

							insert $\left( {{t}_{i},{p}_{2}\left( {t}_{i}\right) }\right)$ in $Q$

													 将 $\left( {{t}_{i},{p}_{2}\left( {t}_{i}\right) }\right)$ 插入 $Q$ 中

							if $\left| Q\right|  > k$ then

													 if $\left| Q\right|  > k$ 则

													delete the minimum element of $Q$

																									 删除 $Q$ 中的最小元素

						end if

											 结束 if

					end if

									 结束 if

	end for

	结束循环

8: Return ${\mathcal{D}}_{k}$

8: 返回 ${\mathcal{D}}_{k}$

---

Algorithm 5. Ranked Join Indices: ConstructRJI $\left( {\mathcal{D}}_{k}\right)$ [Tsaparas et al. 2003]

算法5. 排序连接索引：ConstructRJI $\left( {\mathcal{D}}_{k}\right)$ [察帕拉斯（Tsaparas）等人，2003年]

---

Require: ${\mathcal{D}}_{k}$ : The dominating tuple set

要求: ${\mathcal{D}}_{k}$ : 支配元组集

	1: $V = \phi \;\{$ the separating vector set $\}$

	1: $V = \phi \;\{$ 分离向量集 $\}$

	: for each ${t}_{i},{t}_{j} \in  {\mathcal{D}}_{k},{t}_{i} \neq  {t}_{j}$ do

	: 对每个 ${t}_{i},{t}_{j} \in  {\mathcal{D}}_{k},{t}_{i} \neq  {t}_{j}$ 执行以下操作

			${e}_{{s}_{ij}} =$ separating vector for $\left( {{t}_{i},{t}_{j}}\right)$

					 ${e}_{{s}_{ij}} =$ 是 $\left( {{t}_{i},{t}_{j}}\right)$ 的分离向量

			insert $\left( {{t}_{i},{t}_{j}}\right)$ and their corresponding separating vector ${e}_{{s}_{ij}}$ in $V$

					 将 $\left( {{t}_{i},{t}_{j}}\right)$ 及其对应的分离向量 ${e}_{{s}_{ij}}$ 插入 $V$

		end for

			 结束循环

	6: Sort $V$ in non-decreasing order of vector angles $a\left( {e}_{{s}_{ij}}\right)$

	6: 按向量角度 $a\left( {e}_{{s}_{ij}}\right)$ 非降序对 $V$ 进行排序

	7: ${A}_{k} =$ top- $k$ tuples in ${\mathcal{D}}_{k}$ with respect to predicate ${p}_{1}$

	7: ${A}_{k} =$ 是 ${\mathcal{D}}_{k}$ 中关于谓词 ${p}_{1}$ 的前 $k$ 个元组

		for each $\left( {{t}_{i},{t}_{j}}\right)  \in  V$ do

			 对每个 $\left( {{t}_{i},{t}_{j}}\right)  \in  V$ 执行以下操作

			if both ${t}_{i},{t}_{j} \in  {A}_{k} \vee$ both ${t}_{i},{t}_{j} \notin  {A}_{k}$ then

					 如果 ${t}_{i},{t}_{j} \in  {A}_{k} \vee$ 和 ${t}_{i},{t}_{j} \notin  {A}_{k}$ 都成立

					No change in ${A}_{k}$ by ${e}_{{s}_{ij}}$

									 ${e}_{{s}_{ij}}$ 对 ${A}_{k}$ 无影响

			else if ${t}_{i} \in  {A}_{k} \land  {t}_{j} \notin  {A}_{k}$ then

					 否则，如果 ${t}_{i} \in  {A}_{k} \land  {t}_{j} \notin  {A}_{k}$ 成立

					Store $\left( {a\left( {e}_{{s}_{ij}}\right) ,{A}_{k}}\right)$ into index

									 将 $\left( {a\left( {e}_{{s}_{ij}}\right) ,{A}_{k}}\right)$ 存储到索引中

					Replace ${t}_{i}$ with ${t}_{j}$ in ${A}_{k}$

					在${A}_{k}$中用${t}_{j}$替换${t}_{i}$

			else if ${t}_{i} \notin  {A}_{k} \land  {t}_{j} \in  {A}_{k}$ then

			否则，如果${t}_{i} \notin  {A}_{k} \land  {t}_{j} \in  {A}_{k}$，则

					Store $\left( {a\left( {e}_{{s}_{ij}}\right) ,{A}_{k}}\right)$ into index

					将$\left( {a\left( {e}_{{s}_{ij}}\right) ,{A}_{k}}\right)$存储到索引中

					Replace ${t}_{j}$ with ${t}_{i}$ in ${A}_{k}$

					在${A}_{k}$中用${t}_{i}$替换${t}_{j}$

			end if

			结束条件判断

		end for

		结束循环

		Store $\left( {a\left( \overrightarrow{{p}_{2}}\right) ,{A}_{k}}\right)$

		存储$\left( {a\left( \overrightarrow{{p}_{2}}\right) ,{A}_{k}}\right)$

---

<!-- Media -->

We now describe how Ranked Join Indices materialize top- $k$ answers for different scoring functions. Figure 10 shows how the order of tuples change based on the scoring function vector. For two tuples ${t}_{1}$ and ${t}_{2}$ ,there are two possible cases regarding their relative positions:

我们现在描述排序连接索引（Ranked Join Indices）如何为不同的评分函数实现前$k$个答案。图10展示了元组的顺序如何根据评分函数向量而变化。对于两个元组${t}_{1}$和${t}_{2}$，关于它们的相对位置有两种可能的情况：

Case 1. The line connecting the two tuples has a positive slope. In this case, their relative ranks are the same for any scoring function $e$ . This case is illustrated in Figure 10(a).

情况1. 连接这两个元组的直线斜率为正。在这种情况下，对于任何评分函数$e$，它们的相对排名都是相同的。这种情况如图10(a)所示。

<!-- Media -->

<!-- figureText: order : ${t}_{1},{t}_{2}$ (b) ${\mathrm{t}}_{2}$ ${\mathrm{e}}_{\mathrm{s}}$ (a) -->

<img src="https://cdn.noedgeai.com/0195c915-c519-7d98-b2ff-761eddc2c676_24.jpg?x=448&y=286&w=827&h=454&r=0"/>

Fig. 10. Possible relative positions of tuples ${t}_{1}$ and ${t}_{2}$ : (a) positive slope of the line connecting ${t1}$ and ${t2}$ ; (b) negative slope of the line connecting ${t1}$ and ${t2}$ .

图10. 元组${t}_{1}$和${t}_{2}$可能的相对位置：(a) 连接${t1}$和${t2}$的直线斜率为正；(b) 连接${t1}$和${t2}$的直线斜率为负。

<!-- Media -->

Case 2. The line connecting the two tuples has a negative slope. In this case, there is a vector that separates the space into two subspaces, where the tuples' order in one of them is the inverse of the other. This vector,denoted as ${e}_{s}$ ,is perpendicular to the line connecting the two tuples ${t}_{1}$ and ${t}_{2}$ . This case is illustrated in Figure 10(b).

情况2. 连接这两个元组的直线斜率为负。在这种情况下，存在一个向量将空间划分为两个子空间，其中一个子空间中元组的顺序与另一个子空间相反。这个向量，记为${e}_{s}$，与连接两个元组${t}_{1}$和${t}_{2}$的直线垂直。这种情况如图10(b)所示。

Based on the above observation, we can index the order of tuples by keeping a list of all scoring functions vectors (along with their angles) that switch tuples' order. Algorithm 5 describes the details of constructing Ranked Join Indices. The algorithm initially finds the separating vectors between each pair of tuples and sorts these vectors based on their angles. Then,it ranks tuples based on some scoring function (e.g., $F = {p}_{1}$ ) and starts scanning the separating vectors in order. Whenever a vector ${e}_{{s}_{ij}}$ is found such that it changes the order of tuples ${t}_{i}$ and ${t}_{j}$ (case 2),the vector’s angle and its corresponding top- $k$ set are stored in a B-tree index,using angle as the index key. The index construction algorithm has a time complexity of $O\left( {{\left| {\mathcal{D}}_{k}\right| }^{2}\log \left| {\mathcal{D}}_{k}\right| }\right)$ and a space complexity of $O\left( {\left| {\mathcal{D}}_{k}\right| {k}^{2}}\right)$ .

基于上述观察，我们可以通过保留所有改变元组顺序的评分函数向量（以及它们的角度）列表来对元组的顺序进行索引。算法5描述了构建排序连接索引（Ranked Join Indices）的详细过程。该算法首先找出每对元组之间的分隔向量，并根据它们的角度对这些向量进行排序。然后，它根据某个评分函数（例如$F = {p}_{1}$）对元组进行排名，并开始按顺序扫描分隔向量。每当找到一个向量${e}_{{s}_{ij}}$，它改变了元组${t}_{i}$和${t}_{j}$的顺序（情况2）时，该向量的角度及其对应的前$k$个集合将存储在一个B树索引中，使用角度作为索引键。索引构建算法的时间复杂度为$O\left( {{\left| {\mathcal{D}}_{k}\right| }^{2}\log \left| {\mathcal{D}}_{k}\right| }\right)$，空间复杂度为$O\left( {\left| {\mathcal{D}}_{k}\right| {k}^{2}}\right)$。

At query time, the vector corresponding to the scoring function, specified in the query, is determined, and its angle is used to search the B-tree index for the corresponding top- $k$ set. The exact ranking of tuples in the retrieved top- $k$ set is computed,and returned to the user. The query processing algorithm has a complexity of $O\left( {\log \left| {\mathcal{D}}_{k}\right|  + k\log k}\right)$ .

在查询时，确定查询中指定的评分函数对应的向量，并使用其角度在B树索引中搜索对应的前$k$个集合。计算检索到的前$k$个集合中元组的确切排名，并返回给用户。查询处理算法的复杂度为$O\left( {\log \left| {\mathcal{D}}_{k}\right|  + k\log k}\right)$。

4.1.2.2. Top- $k$ Materialized Views. Materialized views have been studied in the context of top- $k$ processing as a means to provide efficient access to scoring and ordering information that is expensive to gather during query execution. Using materialized views for top- $k$ processing has been studied in the PREFER system [Hristidis et al. 2001; Hristidis and Papakonstantinou 2004], which answers preference queries using materialized views. Preference queries are represented as ORDER BY queries that return sorted answers based on predefined scoring predicates. The user preference of a certain tuple is captured by an arbitrary weighted summation of the scoring predicates. The objective is to answer such preference queries using a reasonable number of materialized views.

4.1.2.2. 前$k$个物化视图（Top-$k$ Materialized Views）。物化视图已在处理前$k$个结果的上下文中进行了研究，作为一种有效访问评分和排序信息的手段，这些信息在查询执行期间收集成本较高。在PREFER系统[Hristidis等人，2001；Hristidis和Papakonstantinou，2004]中研究了使用物化视图进行前$k$个结果的处理，该系统使用物化视图回答偏好查询。偏好查询表示为ORDER BY查询，根据预定义的评分谓词返回排序后的答案。某个元组的用户偏好通过评分谓词的任意加权求和来体现。目标是使用合理数量的物化视图来回答此类偏好查询。

The proposed method keeps a number of materialized views based on different weight assignments of the scoring predicates. Specifically,each view $v$ ranks the entire set of underlying tuples based on a scoring function ${F}_{v}$ defined as a weighted summation of the scoring predicates using some weight vector $\overrightarrow{v}$ . For a top- $k$ query with an arbitrary weight vector $\overrightarrow{q}$ ,the materialized view that best matches $\overrightarrow{q}$ is selected to find query answer. Such view is found by computing a position marker for each view to determine the number of tuples that need to be fetched from that view to find query answer. The best view is the one with the least number of fetched tuples.

所提出的方法基于评分谓词的不同权重分配保留了多个物化视图。具体而言，每个视图$v$根据一个评分函数${F}_{v}$对底层元组的整个集合进行排序，该评分函数被定义为使用某个权重向量$\overrightarrow{v}$对评分谓词进行加权求和。对于具有任意权重向量$\overrightarrow{q}$的前$k$查询，选择与$\overrightarrow{q}$最匹配的物化视图来查找查询答案。通过为每个视图计算一个位置标记来确定需要从该视图中提取多少元组以找到查询答案，从而找到这样的视图。最佳视图是提取元组数量最少的视图。

<!-- Media -->

<!-- figureText: $\forall t \in  R,{F}_{v}\left( t\right)  < {T}_{v,q} \Rightarrow  {F}_{q}\left( t\right)  < {F}_{q}\left( {t}_{v}^{1}\right)$ Maximize ${F}_{v}\left( t\right)$ while maintaining inequality ${F}_{a}\left( t\right)$ 17.2 $\overrightarrow{v} = \left( {{0.2},{0.4},{0.4}}\right)$ 17.3 $\overrightarrow{q} = \left( {{0.1},{0.6},{0.3}}\right)$ 16.1 9.9 ${T}_{v,q} = {14.26}$ 10.1 9 ID ${P}_{1}$ ${p}_{2}$ ${P}_{3}$ ${F}_{v}\left( t\right)$ 1 10 17 20 16.8 2 20 20 11 16.4 3 17 18 12 15.4 4 15 10 8 10.2 5 5 10 12 9.8 6 15 10 5 9 -->

<img src="https://cdn.noedgeai.com/0195c915-c519-7d98-b2ff-761eddc2c676_25.jpg?x=529&y=282&w=687&h=423&r=0"/>

Fig. 11. Finding top-1 object based on some materialized view.

图11. 基于某个物化视图查找排名第一的对象。

<!-- Media -->

Top- $k$ query answers in the PREFER system are pipelined. Let $n$ be the number of tuples fetched from view $v$ after computing $v$ ’s position marker. If $n \geq  k$ ,then processing terminates. Otherwise,the $n$ tuples are reported,and a new position marker is computed for $v$ . The process is repeated until $k$ tuples are reported. Computing position markers follows the next procedure. Assume a top- $k$ query $q$ is executed against a relation $R$ . The first marker position for view $v$ is the maximum value ${T}_{v,a}^{1}$ with the following property $\forall t \in  R : {F}_{v}\left( t\right)  < {T}_{v,q}^{1} \Rightarrow  {F}_{q}\left( t\right)  < {F}_{q}\left( {t}_{v}^{1}\right)$ ,where ${t}_{v}^{1}$ is the top tuple in $v$ . At next iterations ${t}_{v}^{top}$ ,the unreported $v$ tuple with the highest score in $v$ ,replaces ${t}_{v}^{1}$ in computing the marker position of $v$ .

PREFER系统中的前$k$查询答案是流水线式处理的。设$n$为在计算视图$v$的位置标记后从该视图中提取的元组数量。如果$n \geq  k$，则处理终止。否则，报告这$n$个元组，并为$v$计算一个新的位置标记。重复该过程，直到报告了$k$个元组。计算位置标记遵循以下步骤。假设针对关系$R$执行一个前$k$查询$q$。视图$v$的第一个标记位置是具有以下属性$\forall t \in  R : {F}_{v}\left( t\right)  < {T}_{v,q}^{1} \Rightarrow  {F}_{q}\left( t\right)  < {F}_{q}\left( {t}_{v}^{1}\right)$的最大值${T}_{v,a}^{1}$，其中${t}_{v}^{1}$是$v$中的顶级元组。在接下来的迭代${t}_{v}^{top}$中，$v$中得分最高的未报告元组在计算$v$的标记位置时替换${t}_{v}^{1}$。

We illustrate how PREFER works using Figure 11, which depicts the procedure followed to find the top-1 object. The depicted view materializes a sorting of the relation $R$ based on weighted summation of the scoring predicates ${p}_{1},{p}_{2}$ ,and ${p}_{3}$ using the weight vector $\overrightarrow{v}$ . However,we are interested in the top-1 object based on another weight vector $\overrightarrow{q}$ . An optimization problem is solved over the set of materialized views to find the view with the shortest prefix that needs to be fetched to answer this top-1 query. The value ${T}_{v,q}$ is computed for each view $v$ ,such that every tuple in $v$ with ${F}_{v}\left( t\right)  < {T}_{v,q}$ cannot be the top-1 answer (i.e.,there exists another tuple with higher ${F}_{q}$ (.) value). This can be verified using the first tuple in the view ${t}_{v}^{1}$ . Once ${T}_{v,q}$ is determined,the prefix from $v$ above ${T}_{v,q}$ is fetched and sorted based on ${F}_{q}$ to find the top-1 object. For example, in Figure 11, a prefix of length 3 needs to be fetched from the depicted view to find out the top-1 tuple based on ${F}_{q}$ . Among the retrieved three tuples,the second tuple is the required top-1 answer. Finding the top- $i$ tuple operates similarly,in a pipelined fashion,by looking for a new ${T}_{v,q}$ value for each $i$ value,such that there exist at least $i$ tuples with larger ${F}_{q}\left( \text{.}\right) {valuesthananyothertuplebelow}{T}_{v,q}$ .

我们使用图11来说明PREFER的工作原理，该图描绘了查找排名第一的对象所遵循的过程。所描绘的视图基于使用权重向量$\overrightarrow{v}$对评分谓词${p}_{1},{p}_{2}$和${p}_{3}$进行加权求和，对关系$R$进行排序并物化。然而，我们感兴趣的是基于另一个权重向量$\overrightarrow{q}$的排名第一的对象。在物化视图集合上解决一个优化问题，以找到需要提取最短前缀来回答这个排名第一查询的视图。为每个视图$v$计算值${T}_{v,q}$，使得$v$中${F}_{v}\left( t\right)  < {T}_{v,q}$的每个元组都不可能是排名第一的答案（即，存在另一个具有更高${F}_{q}$(.)值的元组）。这可以使用视图${t}_{v}^{1}$中的第一个元组来验证。一旦确定了${T}_{v,q}$，就提取$v$中高于${T}_{v,q}$的前缀，并基于${F}_{q}$进行排序以找到排名第一的对象。例如，在图11中，需要从所描绘的视图中提取长度为3的前缀，以找出基于${F}_{q}$的排名第一的元组。在检索到的三个元组中，第二个元组是所需的排名第一的答案。查找前$i$个元组的操作类似，以流水线方式进行，通过为每个$i$值寻找一个新的${T}_{v,q}$值，使得至少存在$i$个具有更大${F}_{q}\left( \text{.}\right) {valuesthananyothertuplebelow}{T}_{v,q}$的元组。

Using materialized views for top- $k$ processing,with linear scoring functions,has been also studied in the LPTA technique proposed by Das et al. [2006]. Top- $k$ answers are obtained by sequentially reading from materialized views, built using the answers of previous queries, and probing the scoring predicates of the retrieved tuples to compute their total scores. The main idea is to choose an optimal subset among all available views to minimize the number of accessed tuples. For example, in the case of linear scoring functions defined on two scoring predicates, only the views with the closest vectors to the query vector in anticlockwise and clockwise directions need to be considered to efficiently answer the query. For example in Figure 12, only the views whose vectors are ${v}_{2}$ and ${v}_{3}$ are considered to compute the query $Q$ . The authors showed that selecting further views in this case is suboptimal. The LPTA algorithm finds the top- $k$ answers by scanning both views, and computing the scores of the retrieved tuples while maintaining a set of top- $k$ candidates. The stoping criterion is similar to TA; the algorithm terminates when the minimum score in the candidate set has a score greater than the maximum score of the unseen tuples,denoted $T$ . The value of $T$ is computed using linear programming. Specifically, each view provides a linear constraint that bounds the space where the non-retrieved tuples reside. Constraints of different views form a convex region, and the maximum score of the unseen tuples is obtained by searching this region.

Das等人[2006]提出的LPTA技术也研究了使用物化视图进行前$k$处理，并采用线性评分函数。前$k$答案是通过顺序读取物化视图获得的，这些视图是使用先前查询的答案构建的，并对检索到的元组的评分谓词进行探测以计算它们的总分数。主要思想是在所有可用视图中选择一个最优子集，以最小化访问的元组数量。例如，在基于两个评分谓词定义的线性评分函数的情况下，为了高效地回答查询，只需要考虑在逆时针和顺时针方向上与查询向量最接近的向量所对应的视图。例如，在图12中，仅考虑向量为${v}_{2}$和${v}_{3}$的视图来计算查询$Q$。作者表明，在这种情况下选择更多的视图并非最优。LPTA算法通过扫描两个视图来找到前$k$答案，并在计算检索到的元组的分数的同时维护一组前$k$候选元组。停止准则与TA类似；当候选集中的最小分数大于未查看元组的最大分数（表示为$T$）时，算法终止。$T$的值是使用线性规划计算的。具体来说，每个视图提供一个线性约束，该约束界定了未检索到的元组所在的空间。不同视图的约束形成一个凸区域，通过搜索该区域可以获得未查看元组的最大分数。

<!-- Media -->

<!-- figureText: ${V}_{Y}^{\prime }$ -->

<img src="https://cdn.noedgeai.com/0195c915-c519-7d98-b2ff-761eddc2c676_26.jpg?x=543&y=285&w=641&h=477&r=0"/>

Fig. 12. Choosing the optimal subset of materialized views in the LPTA algorithm [Das et al. 2006].

图12. LPTA算法中选择物化视图的最优子集[Das等人2006]。

<!-- Media -->

To answer top- $k$ queries in higher dimensions,the authors proved that it is sufficient to use a subset of the available views with size less than or equal to the number of dimensions. An approximate method is used to determine which subset of views is selected based on estimates of the execution cost for each subset. To estimate the cost of a specific subset of views, a histogram of tuples' scores is built using the available histograms of scoring predicates. A greedy algorithm is used to determine the optimal subset of views by incrementally adding the view that provides the minimum estimated cost.

为了在更高维度上回答前$k$查询，作者证明使用大小小于或等于维度数的可用视图子集就足够了。采用一种近似方法，根据每个子集的执行成本估计来确定选择哪些视图子集。为了估计特定视图子集的成本，使用可用的评分谓词直方图构建元组分数的直方图。使用贪心算法通过逐步添加提供最小估计成本的视图来确定视图的最优子集。

### 4.2. Engine Level

### 4.2. 引擎级别

The main theme of the techniques discussed in this section is their tight coupling with the query engine. This tight coupling has been realized through multiple approaches. Some approaches focus on the design of efficient specialized rank-aware query operators. Other approaches introduce an algebra to formalize the interaction between ranking and other relational operations (e.g., joins and selections). A third category addresses modifying query optimizers, for example, changing optimizers' plan enumeration and cost estimation procedures,to recognize the ranking requirements of top- $k$ queries. Treating ranking as a first-class citizen in the query engine provides significant potential for efficient execution of top- $k$ queries. We discuss the techniques that belong to the above categories in the next sections.

本节讨论的技术的主要主题是它们与查询引擎的紧密耦合。这种紧密耦合通过多种方法实现。一些方法专注于设计高效的专门的感知排名的查询运算符。其他方法引入一种代数来形式化排名与其他关系操作（例如，连接和选择）之间的交互。第三类方法涉及修改查询优化器，例如，更改优化器的计划枚举和成本估计过程，以识别前$k$查询的排名要求。将排名作为查询引擎中的一等公民对待，为前$k$查询的高效执行提供了巨大的潜力。我们将在接下来的部分讨论属于上述类别的技术。

4.2.1. Query Operators. The techniques presented in this section provide solutions that embed rank-awareness within query operators. One important property that is satisfied by many of these operators is pipelining. Pipelining allows for reporting query answers without processing all of the underlying data if possible, and thus minimizing query response time. In pipelining, the next object produced by one query operator is fed into a subsequent operator upon request. Generally, algorithms that require random access are unsuitable for pipelining. The reason is that requesting objects by their identifiers breaks the pipeline by materializing specific objects, which could incur a large overhead. TA and CA (discussed in Section 3.1) are thus generally unsuitable for pipelining. Although the NRA algorithm (discussed in Section 3.2) does not require random access, it is not also capable of pipelining since the reported objects do not have associated exact scores. Hence, the output of one NRA process cannot serve as a valid input to another NRA process.

4.2.1. 查询运算符。本节介绍的技术提供了将排名感知嵌入查询运算符的解决方案。许多这些运算符满足的一个重要属性是流水线处理。流水线处理允许在可能的情况下在不处理所有底层数据的情况下报告查询答案，从而最小化查询响应时间。在流水线处理中，一个查询运算符产生的下一个对象会根据请求被输入到后续运算符中。一般来说，需要随机访问的算法不适合流水线处理。原因是通过标识符请求对象会通过物化特定对象来打破流水线，这可能会产生很大的开销。因此，TA和CA（在第3.1节讨论）通常不适合流水线处理。虽然NRA算法（在第3.2节讨论）不需要随机访问，但它也不能进行流水线处理，因为报告的对象没有关联的精确分数。因此，一个NRA过程的输出不能作为另一个NRA过程的有效输入。

One example of rank-aware query operators that support pipelining is the Rank-Join operator [Ilyas et al. 2004a], which integrates the joining and ranking tasks in one efficient operator. Algorithm 6 describes the main Rank-Join procedure. The algorithm has common properties with the NRA algorithm [Fagin et al. 2001] (described in Section 3.2). Both algorithms perform sorted access to get tuples from each data source. The main difference is that the NRA algorithm assumes that each partially seen tuple has a valid score that can be completely computed if the values of the currently unknown tuple's scoring predicates are obtained. This assumption cannot be made for the case of joining tuples from multiple sources, since arbitrary subsets of the Cartesian product of tuples may end up in the join result based on the join condition. For this reason, the Rank-Join algorithm maintains the scores of the completely seen join combinations only. As a result, the Rank-Join algorithm reports the exact scores of the top- $k$ tuples,while the NRA algorithm reports bounds on tuples’ scores. Another difference is that the NRA algorithm has strict access pattern that requires retrieval of a new tuple from each source at each iteration. The Rank-Join algorithm does not impose any constraints on tuples retrieval, leaving the access pattern to be specified by the underlying join algorithm.

支持流水线操作的感知排名查询运算符的一个示例是排名连接运算符（Rank-Join operator）[Ilyas等人，2004a]，它将连接和排名任务集成到一个高效的运算符中。算法6描述了主要的排名连接过程。该算法与NRA算法[Fagin等人，2001]（在3.2节中描述）有共同的特性。这两种算法都执行排序访问以从每个数据源获取元组。主要区别在于，NRA算法假设每个部分可见的元组都有一个有效的分数，如果获得当前未知元组的评分谓词的值，该分数可以完全计算出来。对于从多个源连接元组的情况，不能做出这种假设，因为基于连接条件，元组的笛卡尔积的任意子集最终可能会出现在连接结果中。因此，排名连接算法仅维护完全可见的连接组合的分数。结果，排名连接算法报告前$k$个元组的确切分数，而NRA算法报告元组分数的边界。另一个区别是，NRA算法有严格的访问模式，要求在每次迭代时从每个源检索一个新的元组。排名连接算法对元组检索不施加任何约束，将访问模式留给底层连接算法指定。

Similarly to NRA algorithm, the Rank-Join algorithm scans input lists (the joined relations) in the order of their scoring predicates. Join results are discovered incrementally as the algorithm moves down the ranked input relations. For each join result $j$ , the algorithm computes a score for $j$ using a score aggregation function $F$ ,following the top- $k$ join query model (Section 2.1). The algorithm maintains a threshold $T$ bounding the scores of join results that are not discovered yet. The top- $k$ join results are obtained when the minimum score of the $k$ join results with the maximum $F$ (.) values is not below the threshold $T$ .

与NRA算法类似，排名连接算法按照输入列表（被连接的关系）的评分谓词顺序扫描它们。随着算法在排序后的输入关系中向下移动，连接结果会逐步被发现。对于每个连接结果$j$，算法使用分数聚合函数$F$为$j$计算一个分数，遵循前$k$个连接查询模型（2.1节）。该算法维护一个阈值$T$，用于界定尚未发现的连接结果的分数。当前$k$个连接结果中具有最大$F$(.)值的$k$个连接结果的最小分数不低于阈值$T$时，就得到了前$k$个连接结果。

A two-way hash join implementation of the Rank-Join algorithm, called Hash Rank Join Operator (HRJN), was introduced in Ilyas et al. [2004a]. HRJN is based on symmetrical hash join. The operator maintains a hash table for each relation involved in the join process, and a priority queue to buffer the join results in the order of their scores. The hash tables hold input tuples seen so far and are used to compute the valid join results. The HRJN operator implements the traditional iterator interface of query operators. The details of the Open and GetNext methods are given by Algorithms 7 and 8, respectively. The Open method is responsible for initializing the necessary data structure; the priority queue $Q$ ,and the left and right hash tables. It also sets $T$ ,the score upper bound of unseen join results, to the maximum possible value.

Ilyas等人[2004a]引入了排名连接算法的一种双向哈希连接实现，称为哈希排名连接运算符（Hash Rank Join Operator，HRJN）。HRJN基于对称哈希连接。该运算符为连接过程中涉及的每个关系维护一个哈希表，并使用一个优先队列按分数顺序缓冲连接结果。哈希表保存到目前为止看到的输入元组，并用于计算有效的连接结果。HRJN运算符实现了查询运算符的传统迭代器接口。Open和GetNext方法的详细信息分别由算法7和8给出。Open方法负责初始化必要的数据结构；优先队列$Q$以及左右哈希表。它还将$T$（即未看到的连接结果的分数上限）设置为可能的最大值。

<!-- Media -->

Algorithm 6. Rank Join [Ilyas et al. 2004a]

算法6. 排名连接 [Ilyas等人，2004a]

---

-Retrieve tuples from input relations in descending order of their individual scores ${p}_{i}$ ’s. For each

-按照输入关系中各个元组的分数${p}_{i}$的降序从输入关系中检索元组。对于每个

	new retrieved tuple $t$ :

	新检索到的元组$t$：

	(1) Generate new valid join combinations between $t$ and seen tuples in other relations.

	(1) 在$t$和其他关系中已看到的元组之间生成新的有效连接组合。

	(2) For each resulting join combination $j$ ,compute $F\left( j\right)$ .

	(2) 对于每个得到的连接组合$j$，计算$F\left( j\right)$。

	(3) Let ${p}_{i}^{\left( \max \right) }$ be the top score in relation $i$ ,that is,the score of the first tuple retrieved from

	(3) 设${p}_{i}^{\left( \max \right) }$为关系$i$中的最高分数，即从关系${p}_{i}^{\left( \max \right) }$中检索到的第一个元组的分数。设$i$为关系${p}_{i}^{\left( \max \right) }$中最后看到的分数。设[latex2]为以下${p}_{i}^{\left( \max \right) }$值中的最大值：

			relation $i$ . Let ${\bar{p}}_{i}$ be the last seen score in relation $i$ . Let $T$ be the maximum of the following

			$i$值：

			$m$ values:

			$m$值：

			$F\left( {{\bar{p}}_{1},{p}_{2}^{\max },\ldots ,{p}_{m}^{\max }}\right) ,$

			$F\left( {{p}_{1}^{\max },{\bar{p}}_{2},\ldots ,{p}_{m}^{\max }}\right)$ ,

			...

			$F\left( {{p}_{1}^{\max },{p}_{2}^{\max },\ldots ,{\bar{p}}_{m}}\right)$ .

	(4) Let ${A}_{k}$ be a set of $k$ join results with the maximum $F$ (.) values,and ${M}_{k}$ be the lowest score

	(4) 设${A}_{k}$是具有最大$F$(.)值的$k$个连接结果的集合，${M}_{k}$是${A}_{k}$中的最低分数。当$k$时停止。

			in ${A}_{k}$ . Halt when ${M}_{k} \geq  T$ .

			当${M}_{k} \geq  T$时停止。

-Report the join results in ${A}_{k}$ ordered on their $F$ (.) values.

-按照$F$(.)值对${A}_{k}$中的连接结果进行排序并报告。

---

Algorithm 7. HRJN: Open $\left( {{L}_{1},{L}_{2}}\right)$ [Ilyas et al. 2004a]

算法7. HRJN：Open $\left( {{L}_{1},{L}_{2}}\right)$ [Ilyas等人，2004a]

---

Require: ${L}_{1}$ : Left ranked input, ${L}_{2}$ : Right ranked input

要求：${L}_{1}$：左排序输入，${L}_{2}$：右排序输入

	1: Create a priority queue $Q$ to order join results based on $F$ (.) values

	1: 创建一个优先队列 $Q$，以根据 $F$ (.) 值对连接结果进行排序

	2: Build two hash tables for ${L}_{1}$ and ${L}_{2}$

	2: 为${L}_{1}$和${L}_{2}$构建两个哈希表

	3: Set threshold $T$ to the maximum possible value of $F$

	3: 将阈值$T$设置为$F$的最大可能值

	4: Initialize ${\bar{p}}_{1}$ and ${p}_{1}^{\max }$ with the maximum score in ${L}_{1}$

	4: 用${L}_{1}$中的最大分数初始化${\bar{p}}_{1}$和${p}_{1}^{\max }$

	5: Initialize ${\bar{p}}_{2}$ and ${p}_{2}^{\max }$ with the maximum score in ${L}_{2}$

	5: 用${L}_{2}$中的最大分数初始化${\bar{p}}_{2}$和${p}_{2}^{\max }$

	6: ${L}_{1}$ .Open(   )

	6: ${L}_{1}$ .Open(   )

	$7 : {L}_{2}$ .Open(   )

	$7 : {L}_{2}$ .Open(   )

---

Algorithm 8. HRJN: GetNext [Ilyas et al. 2004a]

算法8. HRJN：获取下一个 [伊利亚斯等人，2004a]

---

	: if $\left| Q\right|  > 0$ then

	: 如果$\left| Q\right|  > 0$，则

						${j}_{\text{top }} =$ peek at top element in $Q$

						${j}_{\text{top }} =$ 查看$Q$中的顶部元素

						if $F\left( {j}_{\text{top }}\right)  \geq  T$ then

						如果$F\left( {j}_{\text{top }}\right)  \geq  T$，则

										Remove ${j}_{\text{top }}$ from $Q$

										从$Q$中移除${j}_{\text{top }}$

										Return Jtop

										返回Jtop

				end if

				结束条件判断

	end if

	end if

	loop

	循环

							Determine next input to access, ${L}_{i}$

							确定下一个要访问的输入，${L}_{i}$

							$t = {L}_{i}$ . GetNext(   )

							$t = {L}_{i}$ . 获取下一个(   )

							if $t$ is the first seen tuple in ${L}_{i}$ then

							如果 $t$ 是 ${L}_{i}$ 中首次出现的元组，则

									${p}_{i}^{\max } = {p}_{i}\left( t\right)$

			end if

			结束条件判断

						${\bar{p}}_{i} = {p}_{i}\left( t\right)$

						$T = \operatorname{MAX}\left( {F\left( {{p}_{1}^{\max },{\bar{p}}_{2}}\right) ,F\left( {{\bar{p}}_{1},{p}_{2}^{\max }}\right) }\right)$

						insert $t$ in ${L}_{i}$ Hash table

						将 $t$ 插入 ${L}_{i}$ 哈希表

						probe the other hash table using $t$ ’s join key

						使用 $t$ 的连接键探查另一个哈希表

						for all valid join combination $j$ do

						对于所有有效的连接组合 $j$ 执行

											Compute $F\left( j\right)$

											计算 $F\left( j\right)$

											Insert $j$ in $Q$

											将 $j$ 插入 $Q$

			end for

			结束循环

							if $\left| Q\right|  > 0$ then

							如果 $\left| Q\right|  > 0$ 则

													${j}_{\text{top }} =$ peek at top element in $Q$

													${j}_{\text{top }} =$ 查看 $Q$ 中的顶部元素

												if $F\left( {j}_{top}\right)  \geq  T$ then

												如果 $F\left( {j}_{top}\right)  \geq  T$ 则

																				break loop

																				跳出循环

								end if

								结束条件判断

				end if

				结束条件判断

		end loop

		结束循环

		Remove top tuple ${j}_{\text{top }}$ from $Q$

		从 $Q$ 中移除顶部元组 ${j}_{\text{top }}$

30: Return ${j}_{\text{top }}$

30: 返回 ${j}_{\text{top }}$

---

<!-- Media -->

The GetNext method remembers the two top scores, ${p}_{1}^{\max }$ and ${p}_{2}^{\max }$ ,and the last seen scores, ${\bar{p}}_{1}$ and ${\bar{p}}_{2}$ of its left and right inputs. Notice that ${\bar{p}}_{1}$ and ${\bar{p}}_{2}$ are continuously updated as new tuples are retrieved from the input relations. At any time during query execution,the threshold $T$ is computed as the maximum of $F\left( {{p}_{1}^{\max },{\bar{p}}_{2}}\right)$ and $F\left( {{\bar{p}}_{1},{p}_{2}^{\max }}\right)$ . At each step, the algorithm reads tuples from either the left or right inputs, and probes the hash table of the other input to generate join results. The algorithm decides which input to poll at each step, which gives flexibility to optimize the operator for fast generation of join results based on the joined data. A simplistic strategy is accessing the inputs in a round-robin fashion. A join result is reported if its score is not below the scores of all discovered join results,and the threshold $T$ .

GetNext 方法会记录其左右输入的两个最高分数 ${p}_{1}^{\max }$ 和 ${p}_{2}^{\max }$，以及最后看到的分数 ${\bar{p}}_{1}$ 和 ${\bar{p}}_{2}$。请注意，随着从输入关系中检索到新的元组，${\bar{p}}_{1}$ 和 ${\bar{p}}_{2}$ 会不断更新。在查询执行的任何时候，阈值 $T$ 被计算为 $F\left( {{p}_{1}^{\max },{\bar{p}}_{2}}\right)$ 和 $F\left( {{\bar{p}}_{1},{p}_{2}^{\max }}\right)$ 中的最大值。在每一步，算法从左输入或右输入中读取元组，并探查另一个输入的哈希表以生成连接结果。该算法决定在每一步轮询哪个输入，这为基于连接数据快速生成连接结果来优化操作符提供了灵活性。一种简单的策略是以轮询的方式访问输入。如果一个连接结果的分数不低于所有已发现的连接结果的分数以及阈值 $T$，则报告该连接结果。

Other examples of top- $k$ operators that are suitable for pipelining are the NRA-RJ operator [Ilyas et al. 2002],and the ${J}^{ * }$ algorithm [Natsev et al. 2001]. The NRA-RJ operator extends the NRA algorithm [Fagin et al. 2001] using an efficient query operator that can serve valid input to other NRA-RJ operators in the query pipeline. The ${J}^{ * }$ algorithm [Natsev et al. 2001] (discussed in Section 3.2) supports pipelining since it does not require random access to its inputs, and produces join results with complete scores.

适合流水线处理的前 $k$ 操作符的其他示例包括 NRA - RJ 操作符（[Ilyas 等人，2002 年]）和 ${J}^{ * }$ 算法（[Natsev 等人，2001 年]）。NRA - RJ 操作符使用一个高效的查询操作符扩展了 NRA 算法（[Fagin 等人，2001 年]），该操作符可以为查询流水线中的其他 NRA - RJ 操作符提供有效的输入。${J}^{ * }$ 算法（[Natsev 等人，2001 年]，在第 3.2 节讨论）支持流水线处理，因为它不需要随机访问其输入，并且能生成具有完整分数的连接结果。

Li et al. [2006] introduced rank-aware query operators that work under the top- $k$ aggregate query model (Section 2.1). Top- $k$ aggregate queries report the $k$ groups (based on some grouping columns) with the highest aggregate values (e.g., sum). The conventional processing of such queries follows a materialize-group-sort scheme, which can be inefficient if only the top- $k$ groups are required. Moreover,it is common,in this kind of queries, to use ad hoc aggregate functions that are specified only at query time for data exploration purposes. Supporting such ad hoc aggregate functions is challenging since they cannot benefit from any existing precomputations.

Li 等人 [2006 年] 引入了在 top - $k$ 聚合查询模型（第 2.1 节）下工作的排名感知查询操作符。top - $k$ 聚合查询报告具有最高聚合值（例如，总和）的 $k$ 个组（基于某些分组列）。此类查询的传统处理遵循物化 - 分组 - 排序方案，如果只需要前 $k$ 个组，这种方案可能效率低下。此外，在这类查询中，通常会使用仅在查询时为数据探索目的指定的临时聚合函数。支持此类临时聚合函数具有挑战性，因为它们无法从任何现有的预计算中受益。

Two fundamental principles have been proposed in Li et al. [2006] to address the above challenges. The first principle, Group-Ranking, dictates the order in which groups are probed during top- $k$ processing. The authors proposed prioritizing group access by incrementally consuming tuples from the groups with the maximum possible aggregate values. This means that it might not be necessary to complete the evaluation of some groups not included in the current top- $k$ . Knowing the maximum possible aggregate values beforehand is possible if information regarding the cardinality of each group can be obtained. This information is typically available in environments such as OLAP,where aggregation and top- $k$ queries are dominant.

Li 等人 [2006 年] 提出了两个基本原则来应对上述挑战。第一个原则，组排名（Group - Ranking），规定了在 top - $k$ 处理期间探查组的顺序。作者提出通过逐步消耗具有最大可能聚合值的组中的元组来对组访问进行优先级排序。这意味着可能不需要完成对当前 top - $k$ 中未包含的某些组的评估。如果可以获得关于每个组的基数的信息，那么事先知道最大可能的聚合值是可行的。此类信息通常在诸如 OLAP 之类的环境中可用，在这些环境中，聚合和 top - $k$ 查询占主导地位。

The second principle, Tuple-Ranking, dictates the order in which tuples should be accessed from each group. In aggregate queries, each tuple has a scoring attribute, usually referred to as the measure attribute, which contributes to the aggregate score of tuple's group, for example, the salary attribute for the aggregate function sum(salary). The authors showed that the tuple order that results in the minimum tuple depth (the number of accessed tuples from each group), is among three tuple orders, out of all possible permutations: Descending Tuple Score Order, Ascending Tuple Score Order, and Hybrid Order, which chooses the tuple with either the highest or lowest score among unseen tuples.

第二个原则，元组排名（Tuple - Ranking），规定了应从每个组中访问元组的顺序。在聚合查询中，每个元组都有一个计分属性，通常称为度量属性，它对元组所在组的聚合分数有贡献，例如，聚合函数 sum(salary) 中的工资属性。作者表明，导致最小元组深度（从每个组中访问的元组数量）的元组顺序是所有可能排列中的三种元组顺序之一：元组分数降序、元组分数升序和混合顺序，混合顺序从尚未查看的元组中选择分数最高或最低的元组。

The two above principles were encapsulated in a query operator, called rankaggr. The new operator eagerly probes the groups instead of waiting for all groups to be materialized by the underlying query subtree. The next group to probe is determined according to the maximum possible scores of all valid groups, and the next tuple is drawn from this group. As a heuristic, tuples are accessed from any group in descending score order. When a group is exhausted, its aggregate value is reported. This guarantees pipelining the resulting groups in the correct order with respect to their aggregate values.

上述两个原则被封装在一个名为 rankaggr 的查询操作符中。新操作符会主动探查组，而不是等待底层查询子树将所有组物化。要探查的下一个组是根据所有有效组的最大可能分数确定的，并且从该组中提取下一个元组。作为一种启发式方法，以分数降序从任何组中访问元组。当一个组耗尽时，报告其聚合值。这保证了根据聚合值以正确的顺序对结果组进行流水线处理。

<!-- Media -->

<!-- figureText: Rank: $\mu$ ,with a ranking predicate $p$ - ${t}_{1}{ < }_{{R}_{{\mathcal{P}}_{1}} - {S}_{{\mathcal{P}}_{2}}}{t}_{2}$ iff ${t}_{1}{ < }_{{R}_{{\mathcal{P}}_{1}}}{t}_{2}$ ,i.e., ${\overline{\mathcal{F}}}_{{\mathcal{P}}_{1}}\left\lbrack  {t}_{1}\right\rbrack   < {\overline{\mathcal{F}}}_{{\mathcal{P}}_{1}}\left\lbrack  {t}_{2}\right\rbrack$ - $t \in  {\mu }_{p}\left( {R}_{\mathcal{P}}\right)$ iff $t \in  {R}_{\mathcal{P}}$ - ${t}_{1}{ < }_{{\mu }_{p}\left( {R}_{\mathcal{P}}\right) }{t}_{2}$ iff ${\overline{\mathcal{F}}}_{\mathcal{P}\cup \{ p\} }\left\lbrack  {t}_{1}\right\rbrack   < {\overline{\mathcal{F}}}_{\mathcal{P}\cup \{ p\} }\left\lbrack  {t}_{2}\right\rbrack$ Selection: $\sigma$ ,with a Boolean condition $c$ - $t \in  {\sigma }_{c}\left( {R}_{\mathcal{P}}\right)$ iff $t \in  {R}_{\mathcal{P}}$ and $t$ satisfies $c$ - ${t}_{1}{ < }_{{\sigma }_{c}\left( {R}_{\mathcal{P}}\right) }{t}_{2}$ iff ${t}_{1}{ < }_{{R}_{\mathcal{P}}}{t}_{2}$ ,i.e., ${\overline{\mathcal{F}}}_{\mathcal{P}}\left\lbrack  {t}_{1}\right\rbrack   < {\overline{\mathcal{F}}}_{\mathcal{P}}\left\lbrack  {t}_{2}\right\rbrack$ Union: $\cup$ - $t \in  {R}_{{\mathcal{P}}_{1}} \cup  {S}_{{\mathcal{P}}_{2}}$ iff $t \in  {R}_{{\mathcal{P}}_{1}}$ or $t \in  {S}_{{\mathcal{P}}_{2}}$ - ${t}_{1}{ < }_{{R}_{{\mathcal{P}}_{1}} \cup  {S}_{{\mathcal{P}}_{2}}}{t}_{2}$ iff ${\overline{\mathcal{F}}}_{{\mathcal{P}}_{1} \cup  {\mathcal{P}}_{2}}\left\lbrack  {t}_{1}\right\rbrack   < {\overline{\mathcal{F}}}_{{\mathcal{P}}_{1} \cup  {\mathcal{P}}_{2}}\left\lbrack  {t}_{2}\right\rbrack$ Intersection: $\cap$ - $t \in  {R}_{{\mathcal{P}}_{1}} \cap  {S}_{{\mathcal{P}}_{2}}$ iff $t \in  {R}_{{\mathcal{P}}_{1}}$ and $t \in  {S}_{{\mathcal{P}}_{2}}$ $\bullet  \;{t}_{1}{ < }_{{R}_{{\mathcal{P}}_{1}} \cap  {S}_{{\mathcal{P}}_{2}}}{t}_{2}\;$ iff $\;{\overline{\mathcal{F}}}_{{\mathcal{P}}_{1} \cup  {\mathcal{P}}_{2}}\lbrack {t}_{1}\rbrack  < {\overline{\mathcal{F}}}_{{\mathcal{P}}_{1} \cup  {\mathcal{P}}_{2}}\lbrack {t}_{2}\rbrack$ Difference: - - $t \in  {R}_{{\mathcal{P}}_{1}} - {S}_{{\mathcal{P}}_{2}}$ iff $t \in  {R}_{{\mathcal{P}}_{1}}$ and $t \notin  {S}_{{\mathcal{P}}_{2}}$ Join: $\bowtie$ ,with a join condition $c$ $\bullet  t \in  {R}_{{\mathcal{P}}_{1}}{ \bowtie  }_{c}{S}_{{\mathcal{P}}_{2}}$ iff $t \in  {R}_{{\mathcal{P}}_{1}} \times  {S}_{{\mathcal{P}}_{2}}$ and satisfies $c$ $\bullet  {t}_{1}{ < }_{{R}_{{\mathcal{P}}_{1}}{ \bowtie  }_{c}{S}_{{\mathcal{P}}_{2}}}{t}_{2}$ iff ${\overline{\mathcal{F}}}_{{\mathcal{P}}_{1} \cup  {\mathcal{P}}_{2}}\left\lbrack  {t}_{1}\right\rbrack   < {\overline{\mathcal{F}}}_{{\mathcal{P}}_{1} \cup  {\mathcal{P}}_{2}}\left\lbrack  {t}_{2}\right\rbrack$ -->

<img src="https://cdn.noedgeai.com/0195c915-c519-7d98-b2ff-761eddc2c676_30.jpg?x=529&y=288&w=667&h=757&r=0"/>

Fig. 13. Operators defined in RankSQL algebra [Li et al. 2005].

图13. RankSQL代数中定义的运算符 [Li等人，2005年]。

<!-- Media -->

4.2.2. Query Algebra. Formalizing the interaction between ranking and other relational operations, for example, selections and joins, through an algebraic framework, gives more potential to optimize top- $k$ queries. Taking the ranking requirements into account,while building a top- $k$ query plan,has been shown to yield significant performance gains compared to the conventional materialize-then-sort techniques [Ilyas et al. 2004b]. These ideas are the foundations of the RankSQL system [Li et al. 2005], which introduces the first algebraic framework to support efficient evaluations of top- $k$ queries in relational database systems.

4.2.2. 查询代数。通过代数框架将排序与其他关系操作（例如选择和连接）之间的交互形式化，为优化前 $k$ 查询提供了更多潜力。在构建前 $k$ 查询计划时考虑排序要求，与传统的先物化再排序技术相比，已被证明能显著提高性能 [Ilyas等人，2004b]。这些思想是RankSQL系统 [Li等人，2005年] 的基础，该系统引入了第一个代数框架，以支持在关系数据库系统中高效评估前 $k$ 查询。

RankSQL views query ranking requirements as a logical property, similar to the conventional membership property. That is, each base or intermediate relation (the relation generated by a query operator during query execution) is attributed with the base relations it covers (the membership property), as well as the tuple orders it provides (the order property). RankSQL extends traditional relational algebra by introducing a new algebra that embeds rank-awareness into different query operators. The extended rank-relational algebra operates on rank-relations, which are conventional relations augmented with an order property.

RankSQL将查询排序要求视为一种逻辑属性，类似于传统的成员属性。也就是说，每个基本或中间关系（查询执行期间由查询运算符生成的关系）都被赋予它所涵盖的基本关系（成员属性），以及它提供的元组顺序（顺序属性）。RankSQL通过引入一种新的代数，将排序感知嵌入到不同的查询运算符中，从而扩展了传统的关系代数。扩展后的排序 - 关系代数对排序关系进行操作，排序关系是具有顺序属性的传统关系。

Figure 13 summarizes the definitions of the rank-aware operators in RankSQL. The symbol ${R}_{\mathcal{P}}$ denotes a rank-relation $R$ whose order property is the set of ranking predicates $\mathcal{P}$ . The notation ${\overline{\mathcal{F}}}_{\mathcal{P}}\left\lbrack  t\right\rbrack$ denotes the upper bound of the scoring function $\mathcal{F}$ for a tuple $t$ ,based on a set of ranking predicates $\mathcal{P}$ . This upper bound is obtained by applying $\mathcal{F}$ to $\mathcal{P}$ and the maximum values of all other scoring predicates not included in $\mathcal{P}$ .

图13总结了RankSQL中具有排序感知的运算符的定义。符号 ${R}_{\mathcal{P}}$ 表示一个排序关系 $R$，其顺序属性是排序谓词集合 $\mathcal{P}$。符号 ${\overline{\mathcal{F}}}_{\mathcal{P}}\left\lbrack  t\right\rbrack$ 表示基于一组排序谓词 $\mathcal{P}$，元组 $t$ 的评分函数 $\mathcal{F}$ 的上界。这个上界是通过将 $\mathcal{F}$ 应用于 $\mathcal{P}$ 以及 $\mathcal{P}$ 中未包含的所有其他评分谓词的最大值而得到的。

A new rank-augment operator ${\mu }_{p}$ is defined in RankSQL to allow for augmenting the order property of some rank-relation with a new scoring predicate $p$ . That is, ${\mu }_{p}\left( {R}_{\mathcal{P}}\right)  = {R}_{\mathcal{P}\cup \{ p\} }$ . The semantics of the relational operators $\pi ,\sigma , \cup  , \cap  , -$ ,and $\times$ are extended to add awareness of the orders they support. Figure 13 shows how the membership and order properties are computed for different operators. For unary operators,such as $\pi$ and $\sigma$ ,the same tuple order in their inputs is maintained. That is, only the membership property can change, for example, based on a Boolean predicate, while the order property remains the same. On the other hand, binary operators, such as $\cap$ and $\boxtimes$ ,involve both Boolean predicates and order aggregation over their inputs, which can change both the membership and order properties. For example, ${R}_{{\mathcal{P}}_{1}} \cap  {S}_{{\mathcal{P}}_{2}} \equiv  {\left( R \cap  S\right) }_{{\mathcal{P}}_{1} \cup  {\mathcal{P}}_{2}}$ . We discuss the optimization issues of such rank-aware operators in Section 4.2.3.

RankSQL中定义了一个新的排序增强运算符 ${\mu }_{p}$，用于用新的评分谓词 $p$ 增强某个排序关系的顺序属性。即 ${\mu }_{p}\left( {R}_{\mathcal{P}}\right)  = {R}_{\mathcal{P}\cup \{ p\} }$。关系运算符 $\pi ,\sigma , \cup  , \cap  , -$ 和 $\times$ 的语义被扩展，以增加对它们所支持顺序的感知。图13展示了如何为不同的运算符计算成员属性和顺序属性。对于一元运算符，如 $\pi$ 和 $\sigma$，会保持其输入中的相同元组顺序。也就是说，只有成员属性可以改变，例如基于布尔谓词，而顺序属性保持不变。另一方面，二元运算符，如 $\cap$ 和 $\boxtimes$，涉及布尔谓词和对其输入的顺序聚合，这可以同时改变成员属性和顺序属性。例如 ${R}_{{\mathcal{P}}_{1}} \cap  {S}_{{\mathcal{P}}_{2}} \equiv  {\left( R \cap  S\right) }_{{\mathcal{P}}_{1} \cup  {\mathcal{P}}_{2}}$。我们将在4.2.3节讨论此类具有排序感知的运算符的优化问题。

4.2.3. Query Optimization. Rank-aware operators need to be integrated with query optimizers to be practically useful. Top- $k$ queries often involve different relational operations such as joins, selections, and aggregations. Building a query optimizer that generates efficient query plans satisfying the requirements of such operations, as well as the query ranking requirements, is crucial for efficient processing. An observation that motivates the need for integrating rank-aware operators within query optimizers is that using a rank-aware operator may not be always the best way to produce the required ranked results [Ilyas et al. 2004b, 2006]. The reason is that there are many parameters (e.g., join selectivity, available access paths, and memory size) that need to be taken into account while comparing rank-aware and conventional query operators.

4.2.3. 查询优化。具有排序感知的运算符需要与查询优化器集成才能在实际中发挥作用。前 $k$ 查询通常涉及不同的关系操作，如连接、选择和聚合。构建一个能生成满足这些操作要求以及查询排序要求的高效查询计划的查询优化器，对于高效处理至关重要。促使将具有排序感知的运算符集成到查询优化器中的一个观察结果是，使用具有排序感知的运算符可能并不总是产生所需排序结果的最佳方式 [Ilyas等人，2004b，2006]。原因是在比较具有排序感知的查询运算符和传统查询运算符时，需要考虑许多参数（例如连接选择性、可用的访问路径和内存大小）。

Integrating rank-aware and conventional operators in query optimization has been mainly addressed form two perspectives. In the first perspective, the plan enumeration phase of the query optimizer is extended to allow for mixing and interleaving rank-aware operators with convectional operators,creating a rich space of different top- $k$ query plans. In the second perspective, cost models for rank-aware operators are used by the query optimizer to compare the expected cost of rank-aware operators against conventional operators, and hence construct efficient query plans.

在查询优化中集成感知排名的运算符和传统运算符主要从两个角度进行处理。在第一个角度中，查询优化器的计划枚举阶段得到扩展，以允许将感知排名的运算符与传统运算符混合和交错，从而创建一个丰富的不同前 $k$ 查询计划空间。在第二个角度中，查询优化器使用感知排名运算符的成本模型，将感知排名运算符的预期成本与传统运算符进行比较，从而构建高效的查询计划。

4.2.3.1. Plan Enumeration. The RankSQL system [Li et al. 2005] extends the dynamic programming plan enumeration algorithm, adopted by most RDBMSs, to treat ranking as a logical property. This extension adds an extra enumeration dimension to the conventional membership dimension. In a ranking query plan, the set of scoring predicates in a query subplan determines the order property, just like how the join conditions (together with other operations) determine the membership property. This allows the enumerator to produce different equivalent plans, for the same logical algebra expression, where each plan interleaves ranking operators with other operators in a different way. The two-dimensional (order + membership) enumeration algorithm of RankSQL maintains membership and order properties for each query subplan. Subplans with the same membership and order properties are compared against each other, and the inferior ones, based on cost estimation, are pruned.

4.2.3.1. 计划枚举。RankSQL系统[Li等人，2005年]扩展了大多数关系数据库管理系统（RDBMS）采用的动态规划计划枚举算法，将排名视为一种逻辑属性。这种扩展为传统的成员维度增加了一个额外的枚举维度。在排名查询计划中，查询子计划中的评分谓词集决定了顺序属性，就像连接条件（与其他操作一起）决定成员属性一样。这使得枚举器能够为相同的逻辑代数表达式生成不同的等效计划，其中每个计划以不同的方式将排名运算符与其他运算符交错。RankSQL的二维（顺序 + 成员）枚举算法为每个查询子计划维护成员和顺序属性。具有相同成员和顺序属性的子计划相互比较，并根据成本估计修剪掉较差的子计划。

Figure 14 illustrates the transformation laws of different query operators in RankSQL. The transformations encapsulate two basic principles: (1) splitting (Proposition 1), where ranking is evaluated in stages, predicate by predicate, and (2) interleaving (Propositions 4 and 5), where Ranking is interleaved with other operators. Splitting rank expression allows embedding rank-aware operators in the appropriate places within the query plan, while interleaving guarantees that tuples are pipelined in the correct order such that tuple flow can be stopped as soon as the top- $k$ results are generated. The next example illustrates the above principles.

图14展示了RankSQL中不同查询运算符的转换规则。这些转换封装了两个基本原则：（1）拆分（命题1），即按阶段逐个谓词评估排名；（2）交错（命题4和5），即将排名与其他运算符交错。拆分排名表达式允许将感知排名的运算符嵌入到查询计划中的适当位置，而交错则保证元组以正确的顺序进行流水线处理，以便在生成前 $k$ 结果后立即停止元组流。下一个示例说明了上述原则。

<!-- Media -->

Example 4.1 (RankSQL Example). Consider the following top- $k$ query:

示例4.1（RankSQL示例）。考虑以下前 $k$ 查询：

---

SELECT * FROM $R,S$

从 $R,S$ 中选择所有列

WHERE $R.j = S.j$

其中 $R.j = S.j$

	ORDER BY $R.{p}_{1} + S.{p}_{1} + R.{p}_{2}$

	按 $R.{p}_{1} + S.{p}_{1} + R.{p}_{2}$ 排序

	LIMIT 10

---

<!-- figureText: Proposition 1: Splitting law for $\mu$ $\equiv  {\mu }_{p}\left( {R}_{{\mathcal{P}}_{1}}\right) { \bowtie  }_{c}{S}_{{\mathcal{P}}_{2}}$ ,if only $R$ has attributes in $p$ $\equiv  {\mu }_{p}\left( {R}_{{\mathcal{P}}_{1}}\right) { \bowtie  }_{c}{\mu }_{p}\left( {S}_{{\mathcal{P}}_{2}}\right)$ ,if both $R$ and $S$ have - ${R}_{\left\{  {p}_{1},{p}_{2},\ldots ,{p}_{n}\right\}  } \equiv  {\mu }_{{p}_{1}}\left( {{\mu }_{{p}_{2}}\left( {\ldots \left( {{\mu }_{{p}_{n}}\left( R\right) }\right) \ldots }\right) }\right)$ Proposition 2: Commutative law for binary operator ${R}_{{\mathcal{P}}_{1}}\Theta {S}_{{\mathcal{P}}_{2}} \equiv  {S}_{{\mathcal{P}}_{2}}\Theta {R}_{{\mathcal{P}}_{1}},\forall \Theta  \in  \{  \cap  , \cup  ,{ \bowtie  }_{c}\}$ Proposition 3: Associative law - $\left( {{R}_{{\mathcal{P}}_{1}}\Theta {S}_{{\mathcal{P}}_{2}}}\right) \Theta {T}_{{\mathcal{P}}_{3}} \equiv  {R}_{{\mathcal{P}}_{1}}\Theta \left( {{S}_{{\mathcal{P}}_{2}}\Theta {T}_{{\mathcal{P}}_{3}}}\right) ,\forall \Theta  \in  \left\{  {\cap ,\cup ,{{ \bowtie  }_{c}}^{a}}\right\}$ Proposition 4: Commutative laws for $\mu$ $\bullet  {\mu }_{{p}_{1}}\left( {{\mu }_{{p}_{2}}\left( {R}_{\mathcal{P}}\right) }\right)  \equiv  {\mu }_{{p}_{2}}\left( {{\mu }_{{p}_{1}}\left( {R}_{\mathcal{P}}\right) }\right)$ - ${\sigma }_{c}\left( {{\mu }_{p}\left( {R}_{\mathcal{P}}\right) }\right)  \equiv  {\mu }_{p}\left( {{\sigma }_{c}\left( {R}_{\mathcal{P}}\right) }\right)$ Proposition 5: Pushing $\mu$ over binary operators $\bullet  {\mu }_{p}\left( {{R}_{{\mathcal{P}}_{1}}{ \bowtie  }_{c}{S}_{{\mathcal{P}}_{2}}}\right)$ $\bullet  {\mu }_{p}\left( {{R}_{{\mathcal{P}}_{1}} \cup  {S}_{{\mathcal{P}}_{2}}}\right)  \equiv  {\mu }_{p}\left( {R}_{{\mathcal{P}}_{1}}\right)  \cup  {\mu }_{p}\left( {S}_{{\mathcal{P}}_{2}}\right)  \equiv  {\mu }_{p}\left( {R}_{{\mathcal{P}}_{1}}\right)  \cup  {S}_{{\mathcal{P}}_{2}}$ $\bullet  {\mu }_{p}\left( {{R}_{{\mathcal{P}}_{1}} \cap  {S}_{{\mathcal{P}}_{2}}}\right)  \equiv  {\mu }_{p}\left( {R}_{{\mathcal{P}}_{1}}\right)  \cap  {\mu }_{p}\left( {S}_{{\mathcal{P}}_{2}}\right)  \equiv  {\mu }_{p}\left( {R}_{{\mathcal{P}}_{1}}\right)  \cap  {S}_{{\mathcal{P}}_{2}}$ ${\mu }_{p}\left( {{R}_{{\mathcal{P}}_{1}} - {S}_{{\mathcal{P}}_{2}}}\right)  \equiv  {\mu }_{p}\left( {R}_{{\mathcal{P}}_{1}}\right)  - {S}_{{\mathcal{P}}_{2}} \equiv  {\mu }_{p}\left( {R}_{{\mathcal{P}}_{1}}\right)  - {\mu }_{p}\left( {S}_{{\mathcal{P}}_{2}}\right)$ Proposition 6: Multiple-scan of $\mu$ $\bullet  {\mu }_{{p}_{1}}\left( {{\mu }_{{p}_{2}}\left( {R}_{\phi }\right) }\right)  \equiv  {\mu }_{{p}_{1}}\left( {R}_{\phi }\right) { \cap  }_{r}{\mu }_{{p}_{2}}\left( {R}_{\phi }\right)$ ${}^{a}$ When join columns are available. -->

<img src="https://cdn.noedgeai.com/0195c915-c519-7d98-b2ff-761eddc2c676_32.jpg?x=524&y=288&w=681&h=749&r=0"/>

Fig. 14. RankSQL algebraic laws [Li et al. 2005].

图14. RankSQL代数规则[Li等人，2005年]。

<!-- figureText: Limit 10 ${\text{Rank}}_{\mathrm{R},\mathrm{p}2}$ Hash Rank Join R. j.=S.j -->

<img src="https://cdn.noedgeai.com/0195c915-c519-7d98-b2ff-761eddc2c676_32.jpg?x=660&y=1112&w=409&h=453&r=0"/>

Fig. 15. A query plan generated by RankSQL.

图15. RankSQL生成的查询计划。

<!-- Media -->

Figure 15 depicts a query plan generated by RankSQL, where ranking expression is split and rank-aware operators are interleaved with conventional query operators. The plan shows that the query optimizer considered using existing indexes for $R.{p}_{1}$ and $S.{p}_{1}$ to access base tables,which might be cheaper for the given query,since they generate tuples in the orders of scoring predicates $R.{p}_{1}$ and $S.{p}_{1}$ . The Hash Rank Join operator is a rank-aware join operator that aggregates the orders of the joined relations, while the Rank operator augments the remaining scoring predicate $R . {p2}$ to produce tuples in the required order. The Hash Rank Join and Rank operators pipeline their outputs by upper bounding the scores of their unseen inputs, allowing for consuming a small number of tuples in order to find the top-10 query results. Figure 16 depicts an equivalent conventional plan, where the entire query result is materialized and sorted on the given ranking expression, then the top-10 results are reported.

图15描绘了RankSQL生成的一个查询计划，其中排名表达式被拆分，并且感知排名的运算符与传统查询运算符交错。该计划表明，查询优化器考虑使用 $R.{p}_{1}$ 和 $S.{p}_{1}$ 的现有索引来访问基表，对于给定的查询，这可能更便宜，因为它们按评分谓词 $R.{p}_{1}$ 和 $S.{p}_{1}$ 的顺序生成元组。哈希排名连接运算符是一种感知排名的连接运算符，它聚合连接关系的顺序，而排名运算符则增强剩余的评分谓词 $R . {p2}$ 以按所需顺序生成元组。哈希排名连接运算符和排名运算符通过对其未见过的输入的分数设置上限来对其输出进行流水线处理，从而只需处理少量元组即可找到前10个查询结果。图16描绘了一个等效的传统计划，其中整个查询结果被物化并按给定的排名表达式排序，然后报告前10个结果。

<!-- Media -->

<!-- figureText: SeqScan (R) Limit 10 ${\text{Sort}}_{\mathrm{R} \cdot  \mathrm{p}1 + \mathrm{S} \cdot  \mathrm{p}1 + \mathrm{R} \cdot  \mathrm{p}2}$ Hash Join SeqScan (S) -->

<img src="https://cdn.noedgeai.com/0195c915-c519-7d98-b2ff-761eddc2c676_33.jpg?x=686&y=283&w=362&h=445&r=0"/>

Fig. 16. Conventional query plan.

图16. 传统查询计划。

<!-- figureText: SELECT A.c1, B.c1, C.c1 ORDER BY (0.3*A.c1+0.3*B.c1+0.3*C.c1) ABC BC B. c1 0.3*B.c1 + 0.3*C.c1 AB B.c2 C. c1 C.c2 DC B.c1 B. c2 DC (b) FROM A,B,C LIMIT 5 ABC DC BC B.c1 DC AB B.c2 DC DC B.c1 B. c2 (a) -->

<img src="https://cdn.noedgeai.com/0195c915-c519-7d98-b2ff-761eddc2c676_33.jpg?x=506&y=816&w=727&h=526&r=0"/>

Fig. 17. Plan enumeration (a) in conventional optimizer, and (b) with ordering as an interesting property [Ilyas et al. 2004b].

图17. （a）传统优化器中的计划枚举，以及（b）将排序作为一个有趣属性的计划枚举[Ilyas等人，2004b]。

<!-- Media -->

Treating ranking requirements as a physical property is another approach to extend the plan enumerator. A physical property is a plan characteristic that can be different in different plans with the same membership property, but impacts the cost of subsequent operations. Ilyas et al. [2004b, 2006] defined ranking requirements as an interesting physical order, which triggers the generation of new query plans to optimize the use of Rank-Join operators. An interesting order is an order of query intermediate results, based on an expression of database columns, that can be beneficial to subsequent query operations. An interesting order can thus be the order of a join column or the order of a scoring predicate. To illustrate, Figure 17(a) shows the plan space maintained by a conventional optimizer for the shown three-way join query. In the shown MEMO structure, each row represents a set of plans having the same membership property, while each plan maintains a different interesting order. The plan space is generated by comparing plans with the same interesting order at each row,and keeping the plan with the lowest cost. The node DC refers to a don't care property value,which corresponds to no order. For example,plans in the $B$ row provide tuples of the $B$ relation (the membership property) ordered by $B.{c1},B.{c2}$ , and no order. These orders are interesting since they are included in the query join conditions.

将排序要求视为一种物理属性是扩展计划枚举器的另一种方法。物理属性是一种计划特征，在具有相同成员属性的不同计划中可能有所不同，但会影响后续操作的成本。伊利亚斯等人 [2004b, 2006] 将排序要求定义为一种有趣的物理顺序，这会触发新查询计划的生成，以优化对排名连接（Rank-Join）运算符的使用。有趣的顺序是查询中间结果的一种顺序，它基于数据库列的表达式，可能对后续查询操作有益。因此，有趣的顺序可以是连接列的顺序或评分谓词的顺序。为了说明这一点，图 17(a) 展示了传统优化器为所示的三元连接查询维护的计划空间。在所示的 MEMO 结构中，每行代表一组具有相同成员属性的计划，而每个计划维护不同的有趣顺序。计划空间是通过比较每行中具有相同有趣顺序的计划，并保留成本最低的计划来生成的。节点 DC 表示不关心的属性值，对应于无顺序。例如，$B$ 行中的计划提供 $B$ 关系（成员属性）的元组，这些元组按 $B.{c1},B.{c2}$ 排序，以及无顺序。这些顺序很有趣，因为它们包含在查询连接条件中。

<!-- Media -->

<!-- figureText: Rank-Join ${\delta }_{R}\left( {c}_{R}\right)$ ${d}_{R}$ Score $= {S}_{R}\left( {c}_{R}\right)$ $R$ Score $= {S}_{R}\left( 1\right)  - \delta$ ${\delta }_{L}\left( {c}_{L}\right)$ Score $= {S}_{L}\left( {c}_{L}\right)$ Score $= {S}_{L}\left( 1\right)  - \delta$ L $\delta  = {\delta }_{\mathrm{L}}\left( {\mathrm{c}}_{\mathrm{L}}\right)  + {\delta }_{\mathrm{R}}\left( {\mathrm{c}}_{\mathrm{R}}\right)$ -->

<img src="https://cdn.noedgeai.com/0195c915-c519-7d98-b2ff-761eddc2c676_34.jpg?x=469&y=283&w=793&h=592&r=0"/>

Fig. 18. Estimating depth of Rank-Join inputs.

图 18. 估计排名连接（Rank-Join）输入的深度。

<!-- Media -->

Consider alternatively Figure 17 (b), which shows the plan space generated by treating the orders of scoring predicates as additional interesting orders. The shaded nodes indicate the interesting orders defined by scoring predicates. Enumerating plans satisfying the orders of scoring predicates allows generating rank-aware join choices at each step of the plan enumeration phase. For example, on the level of base tables, the optimizer enforces the generation of table scans and index scans that satisfy the orders of the scoring predicates. For higher join levels, the enumerated access paths for base tables make it feasible to use Rank-Join operators as join choices.

或者考虑图 17(b)，它展示了通过将评分谓词的顺序视为额外的有趣顺序而生成的计划空间。阴影节点表示由评分谓词定义的有趣顺序。枚举满足评分谓词顺序的计划允许在计划枚举阶段的每个步骤生成感知排名的连接选择。例如，在基表级别，优化器强制生成满足评分谓词顺序的表扫描和索引扫描。对于更高的连接级别，为基表枚举的访问路径使得使用排名连接（Rank-Join）运算符作为连接选择成为可能。

4.2.3.2. Cost Estimation. The performance gain obtained by using rank-aware query operators triggers thinking of more principled approaches to integrate ranking with conventional relational operators. This raises the issue of cost estimation. Query optimizers need to be able to enumerate and cost plans with rank-aware operators as well as conventional query operators. The cost model of query operators depends on many parameters including input size, memory buffers and access paths. The optimization study of the Rank-Join operator [Ilyas et al. 2004b, 2006] shows that costing a rank-aware operator is quite different from costing other traditional query operators. A fundamental difference stems from the fact that a rank-aware operator is expected to consume only part of its input, while a traditional operator consumes its input completely. The size of the consumed input depends on the operator implementation rather than the input itself.

4.2.3.2. 成本估计。使用感知排名的查询运算符所获得的性能提升引发了对将排名与传统关系运算符更具原则性地集成的方法的思考。这就提出了成本估计的问题。查询优化器需要能够枚举并计算具有感知排名的运算符以及传统查询运算符的计划的成本。查询运算符的成本模型取决于许多参数，包括输入大小、内存缓冲区和访问路径。对排名连接（Rank-Join）运算符的优化研究 [伊利亚斯等人 2004b, 2006] 表明，计算感知排名的运算符的成本与计算其他传统查询运算符的成本有很大不同。一个根本的区别在于，感知排名的运算符预计只消耗其输入的一部分，而传统运算符会完全消耗其输入。消耗的输入大小取决于运算符的实现，而不是输入本身。

A probabilistic model has been proposed in Ilyas et al. [2004b] to estimate the rank-join inputs' depths, that is, how many tuples are consumed from each input to produce the top- $k$ join results. Figure 18 depicts the depth estimation procedure. For inputs $L$ and $R$ ,the objective is to get the estimates ${d}_{L}$ and ${d}_{R}$ such that it is sufficient to retrieve only up to ${d}_{L}$ and ${d}_{R}$ tuples from $L$ and $R$ ,respectively,to produce the top- $k$ join results. The definitions of other used notations are listed in the following:

伊利亚斯等人 [2004b] 提出了一种概率模型来估计排名连接（Rank-Join）输入的深度，即从每个输入中消耗多少元组以产生前 $k$ 个连接结果。图 18 描述了深度估计过程。对于输入 $L$ 和 $R$，目标是获得估计值 ${d}_{L}$ 和 ${d}_{R}$，使得分别从 $L$ 和 $R$ 中仅检索最多 ${d}_{L}$ 和 ${d}_{R}$ 个元组就足以产生前 $k$ 个连接结果。其他使用的符号定义如下：

$- {c}_{L}$ and ${c}_{R}$ are depths in $L$ and $R$ ,respectively,that are sufficient to find any $k$ valid join results. ${c}_{L}$ and ${c}_{R}$ can be selected arbitrarily such that $s.{c}_{L}.{c}_{R} \geq  k$ ,where $s$ is the join selectivity of $L$ joined with $R$ .

$- {c}_{L}$ 和 ${c}_{R}$ 分别是 $L$ 和 $R$ 中的深度，足以找到任何 $k$ 个有效的连接结果。${c}_{L}$ 和 ${c}_{R}$ 可以任意选择，使得 $s.{c}_{L}.{c}_{R} \geq  k$，其中 $s$ 是 $L$ 与 $R$ 连接的连接选择性。

$- {S}_{L}\left( i\right)$ and ${S}_{R}\left( i\right)$ are the scores at depth $i$ in $L$ and $R$ ,respectively.

$- {S}_{L}\left( i\right)$ 和 ${S}_{R}\left( i\right)$ 分别是 $L$ 和 $R$ 中深度为 $i$ 处的分数。

$- {\sigma }_{L}\left( i\right)$ and ${\sigma }_{R}\left( i\right)$ are the score differences between the top ranked tuple and the the tuple at depth $i$ in $L$ and $R$ ,respectively.

$- {\sigma }_{L}\left( i\right)$ 和 ${\sigma }_{R}\left( i\right)$ 分别是 $L$ 和 $R$ 中排名最高的元组与深度为 $i$ 处的元组之间的分数差。

$- \sigma  = {\sigma }_{L}\left( {c}_{L}\right)  + {\sigma }_{R}\left( {c}_{R}\right) .$

The main result of Ilyas et al. [2004b] is the following: if ${d}_{L}$ and ${d}_{R}$ are selected such that ${\sigma }_{L}\left( {d}_{L}\right)  \geq  \sigma$ and ${\sigma }_{R}\left( {d}_{R}\right)  \geq  \sigma$ ,then the top- $k$ join results can be obtained by joining $L\left( {d}_{L}\right)$ and $R\left( {d}_{R}\right)$ . Further analysis based on the score distribution in $L$ and $R$ has been conducted to reach the minimum possible values for ${d}_{L}$ and ${d}_{R}$ . For uniformly distributed scores of $L$ and $R$ with average score slabs (average distance between two consecutive scores) of $x$ and $y$ ,respectively,the expected value of ${\sigma }_{L}\left( {c}_{L}\right)  = x.{c}_{L}$ and the expected value of ${\sigma }_{R}\left( {c}_{R}\right)  = y.{c}_{R}$ . Hence,to minimize ${d}_{L}$ and ${d}_{R}$ ,we need to minimize $\sigma  = {\sigma }_{L}\left( {c}_{L}\right)  + {\sigma }_{R}\left( {c}_{R}\right)  = x.{c}_{L} + y.{c}_{R}$ subject to $s.{c}_{L}.{c}_{R} \geq  k$ . A direct way to minimize this expression is to select ${c}_{L} = \sqrt{\left( {yk}\right) /\left( {xs}\right) }$ and ${c}_{R} = \sqrt{\left( {xk}\right) /\left( {ys}\right) }$ .

伊利亚斯（Ilyas）等人 [2004b] 的主要结果如下：如果选择 ${d}_{L}$ 和 ${d}_{R}$ 使得 ${\sigma }_{L}\left( {d}_{L}\right)  \geq  \sigma$ 和 ${\sigma }_{R}\left( {d}_{R}\right)  \geq  \sigma$ 成立，那么前 $k$ 个连接结果可以通过连接 $L\left( {d}_{L}\right)$ 和 $R\left( {d}_{R}\right)$ 得到。基于 $L$ 和 $R$ 中的分数分布进行了进一步分析，以得出 ${d}_{L}$ 和 ${d}_{R}$ 的最小可能值。对于 $L$ 和 $R$ 的分数均匀分布，其平均分数区间（两个连续分数之间的平均距离）分别为 $x$ 和 $y$，${\sigma }_{L}\left( {c}_{L}\right)  = x.{c}_{L}$ 的期望值和 ${\sigma }_{R}\left( {c}_{R}\right)  = y.{c}_{R}$ 的期望值。因此，为了最小化 ${d}_{L}$ 和 ${d}_{R}$，我们需要在满足 $s.{c}_{L}.{c}_{R} \geq  k$ 的条件下最小化 $\sigma  = {\sigma }_{L}\left( {c}_{L}\right)  + {\sigma }_{R}\left( {c}_{R}\right)  = x.{c}_{L} + y.{c}_{R}$。最小化这个表达式的直接方法是选择 ${c}_{L} = \sqrt{\left( {yk}\right) /\left( {xs}\right) }$ 和 ${c}_{R} = \sqrt{\left( {xk}\right) /\left( {ys}\right) }$。

## 5. QUERY AND DATA UNCERTAINTY

## 5. 查询与数据的不确定性

In this section,we discuss top- $k$ processing techniques that report approximate answers while operating on deterministic data (Section 5.1), followed by techniques that operate on probabilistic data (Section 5.2).

在本节中，我们将讨论在处理确定性数据时报告近似答案的前 $k$ 个处理技术（5.1 节），接着是处理概率性数据的技术（5.2 节）。

### 5.1. Deterministic Data, Approximate Methods

### 5.1. 确定性数据的近似方法

Reporting the exact top- $k$ query answers could be neither cheap,nor necessary for some applications. For example, decision support and data analysis applications usually process huge volumes of data, which may cause significant delays if the application requires exact query answers. Users in such environments may thus sacrifice the accuracy of query answers in return of savings in time and resources. In these settings, reporting approximate query answers could be sufficient.

对于某些应用程序来说，报告精确的前 $k$ 个查询答案既不划算，也没有必要。例如，决策支持和数据分析应用程序通常会处理大量数据，如果应用程序需要精确的查询答案，可能会导致显著的延迟。因此，在这种环境下的用户可能会牺牲查询答案的准确性，以节省时间和资源。在这些情况下，报告近似查询答案可能就足够了。

We start by describing some of the general approximate query processing techniques, followed by approximate top- $k$ processing techniques.

我们首先描述一些通用的近似查询处理技术，接着是近似的前 $k$ 个处理技术。

5.1.1. Approximate Query Processing. The work of Vrbsky and Liu [1993] is one of the earliest attempts made to approximate query answers. The proposed system, called ${APPROXIMATE}$ , is a query processor that produces approximate answers that enhance monotonically. In this context, an operation is monotone if its results are more accurate when its operands are more accurate. APPROXIMATE works on approximate relations that are defined as subsets of the cartesian product of attribute domains. The tuples in these subsets are partitioned into certain and possible tuples. Approximate relations are used during query processing instead of the standard relations generated by different query tree nodes.

5.1.1. 近似查询处理。弗布斯基（Vrbsky）和刘（Liu） [1993] 的工作是最早尝试近似查询答案的工作之一。所提出的系统名为 ${APPROXIMATE}$，它是一个查询处理器，能产生单调增强的近似答案。在这种情况下，如果一个操作的操作数越精确，其结果就越准确，那么该操作就是单调的。APPROXIMATE 处理的是近似关系，这些关系被定义为属性域笛卡尔积的子集。这些子集中的元组被划分为确定元组和可能元组。在查询处理过程中，使用近似关系代替不同查询树节点生成的标准关系。

Based on the semantics of the data stored in the base tables, an initial approximation to query answers is driven. This initial approximation is assigned to each node in the query tree. As more certain data is read from base tables, more certain tuples are inserted into approximate relations while more possible tuples are deleted. If a query is stopped before completion, a superset of the exact answer is returned. One problem with the proposed method is that it mainly depends on detailed metadata describing the semantics of database relations, which might not be easy to construct in practice.

根据基表中存储的数据的语义，得出查询答案的初始近似值。这个初始近似值被分配给查询树中的每个节点。随着从基表中读取更多的确定数据，更多的确定元组被插入到近似关系中，同时更多的可能元组被删除。如果查询在完成之前停止，则返回精确答案的超集。所提出的方法的一个问题是，它主要依赖于描述数据库关系语义的详细元数据，而在实践中构建这些元数据可能并不容易。

Multiple approximate query processing methods have addressed aggregate queries. Aggregate queries are widely used in OLAP environments for data exploration and analysis purposes. In this kind of queries, the precision of the answer to the last decimal digit is not usually needed. The framework of Online Aggregation [Hellerstein et al. 1997] addresses generating approximate aggregate values whose accuracy enhances progressively. The work of Hellerstein et al. [1997] provides important insights on using statistical methods in aggregate computation. The adopted statistical methods allow computing aggregate values along with a correctness probability and confidence interval derived based on the expected aggregate value.

多种近似查询处理方法已经解决了聚合查询的问题。聚合查询在联机分析处理（OLAP）环境中被广泛用于数据探索和分析目的。在这类查询中，通常不需要答案精确到最后一位小数。在线聚合 [赫勒斯坦（Hellerstein）等人 1997] 框架解决了生成精度逐步提高的近似聚合值的问题。赫勒斯坦等人 [1997] 的工作为在聚合计算中使用统计方法提供了重要的见解。所采用的统计方法允许计算聚合值，并根据预期聚合值得出正确性概率和置信区间。

Other approximate query processing methods involve building a database summary or synopsis, and using it to answer queries approximately. Sampling is probably the most studied methodology for building such summaries. The basic settings involve drawing a sample from the underlying data, answering the incoming queries based on the sample, and scaling the results to approximate the exact answers. Sampling can be done either uniformly at random or in a biased manner to select the best samples that minimize the approximation errors [Chaudhuri et al. 2001b]. Other possible summaries include histograms [Babcock et al. 2003; Chaudhuri et al. 1998] and wavelets Chakrabarti et al. [2001].

其他近似查询处理方法包括构建数据库摘要或大纲，并使用它来近似回答查询。抽样可能是构建此类摘要研究最多的方法。基本设置包括从基础数据中抽取样本，根据样本回答传入的查询，并对结果进行缩放以近似精确答案。抽样可以是均匀随机进行，也可以采用有偏的方式来选择能使近似误差最小化的最佳样本[乔杜里等人，2001b]。其他可能的摘要包括直方图[巴布科克等人，2003；乔杜里等人，1998]和小波[查克拉巴蒂等人，2001]。

Approximation based on sampling has inherent problems that negatively effect the accuracy in many cases. One problem is the presence of skewness in data distributions. Skewed data items, usually called outliers, are those items which deviate significantly from other data items in their values. Outliers cause a large variance in the distribution of aggregate values, which leads to large approximation errors. Separating outliers using a special index is one approach to deal with this problem [Chaudhuri et al. 2001]. In this setting, queries are considered as the union of two subqueries, one of which is answered exactly using an outlier index, while the other is answered approximately by sampling the nonoutliers. The two results are then combined to give the full approximate query answer.

基于抽样的近似方法存在固有问题，在许多情况下会对准确性产生负面影响。一个问题是数据分布中存在偏态。偏态数据项，通常称为离群值，是那些在数值上与其他数据项有显著偏差的项。离群值会导致聚合值分布的方差较大，从而导致较大的近似误差。使用特殊索引分离离群值是处理这个问题的一种方法[乔杜里等人，2001]。在这种设置下，查询被视为两个子查询的并集，其中一个子查询使用离群值索引精确回答，而另一个子查询通过对非离群值进行抽样近似回答。然后将两个结果组合起来，得到完整的近似查询答案。

Another problem is the potentially low selectivity of selection queries. Uniform samples contain only a small fraction of the answers of highly selective queries. Nonuniform sampling is proposed to work around this problem. The idea is to use weighted sampling, where tuples are tagged by their frequency-the number of workload queries whose answers contain the tuple. Tuples that are accessed frequently by previous queries would therefore have higher probability to be included in the sample. The underlying assumption is that tuples that are part of the answers to previous queries are likely to be part of the answers to similar incoming queries. Collecting samples offline based on previous queries, and rewriting incoming queries to use these samples, were proposed in Chaudhuri et al. [2001a]. Self-tuning the samples by refreshing them after queries are processed was further studied by Ganti et al. [2000].

另一个问题是选择查询的选择性可能较低。均匀样本只包含高选择性查询答案的一小部分。为解决这个问题，有人提出了非均匀抽样。其思路是使用加权抽样，其中元组用它们的频率（即答案包含该元组的工作负载查询的数量）进行标记。因此，先前查询频繁访问的元组被包含在样本中的概率会更高。其基本假设是，作为先前查询答案一部分的元组很可能也是类似传入查询答案的一部分。乔杜里等人[2001a]提出了基于先前查询离线收集样本，并重写传入查询以使用这些样本的方法。甘蒂等人[2000]进一步研究了在查询处理后刷新样本以实现样本的自调整。

The presence of small groups in group-by queries could also lead to large approximation errors when using sampling techniques. Congressional samples [Acharya et al. 2000] addressed this problem by introducing a hybrid of uniform and nonuniform samples. The proposed strategy is to divide the available sample space equally among the groups, and take a uniform random sample within each group. This guarantees that both large and small groups will have a reasonable number of samples. The problem with this approach is that it does not deal with the data variance caused by outliers.

在使用抽样技术时，分组查询中存在小分组也可能导致较大的近似误差。国会样本[阿查里雅等人，2000]通过引入均匀样本和非均匀样本的混合来解决这个问题。所提出的策略是在各分组之间平均分配可用的样本空间，并在每个分组内进行均匀随机抽样。这保证了大分组和小分组都将有合理数量的样本。这种方法的问题是它没有处理由离群值引起的数据方差。

5.1.2. Approximate Top-k Query Processing. Adopting the concepts and techniques of approximate query processing in the context of top- $k$ queries is a natural extension of the previously described works. Some general top- $k$ algorithms,for example,TA [Fagin et al. 2001], have approximate variants. For example, an approximate variant of TA defines a parameter $\theta  > 1$ denoting the required level of approximation,such that an item $z$ not in the top- $k$ set satisfies the condition score $\left( z\right)  \leq  \theta$ score(y)for every other item $y$ inside the top- $k$ set. This approximation relaxes the threshold test of TA, making it possible to return approximate answers. However, the problem with this approach is that the selection of the approximation parameter $\theta$ is mostly application-oriented and that no general scheme is devised to decide its value. Another example is the ${J}^{ * }$ algorithm [Natsev et al. 2001] that allows approximating tuple ranks to a factor $0 < \epsilon  < 1$ . The $\epsilon$ -approximate top- $k$ answer, $X$ ,is defined as $\forall x \in  X,y \notin  X$ : $\left( {1 + \epsilon }\right) x$ . score $\geq  y$ . score. When $\epsilon  = 0$ ,the approximation algorithm reduces to the exact one.

5.1.2. 近似Top - k查询处理。在Top - $k$查询的上下文中采用近似查询处理的概念和技术是前面所述工作的自然延伸。一些通用的Top - $k$算法，例如TA[法金等人，2001]，有近似变体。例如，TA的一个近似变体定义了一个参数$\theta  > 1$，表示所需的近似水平，使得不在Top - $k$集合中的项$z$满足条件：对于Top - $k$集合内的每个其他项$y$，有score $\left( z\right)  \leq  \theta$ score(y)。这种近似放宽了TA的阈值测试，使得返回近似答案成为可能。然而，这种方法的问题是近似参数$\theta$的选择大多是面向应用的，并且没有设计出通用的方案来确定其值。另一个例子是${J}^{ * }$算法[纳采夫等人，2001]，它允许将元组排名近似到一个因子$0 < \epsilon  < 1$。$\epsilon$ -近似Top - $k$答案$X$定义为$\forall x \in  X,y \notin  X$：$\left( {1 + \epsilon }\right) x$ . score $\geq  y$ . score。当$\epsilon  = 0$时，近似算法退化为精确算法。

Approximate answers are more useful when they are associated with some accuracy guarantees. This issue has been addressed by another approximate adaptation of TA [Theobald et al. 2004], where a scheme was introduced to associate probabilistic guarantees with approximate top- $k$ answers. The proposed scheme works in the context of information retrieval systems, assuming multiple lists each holding a different ranking of an underlying document set based on different criteria, such as query keywords. The conservative TA threshold test is replaced by a probabilistic test to estimate the probability that a candidate document would eventually be in the top- $k$ set. Specifically,the probability of document $d$ to have a score above ${M}_{k}$ ,the minimum score of the current top- $k$ set,is computed as follows:

当近似答案与一定的准确性保证相关联时，它们会更有用。TA（Theobald等人，2004年）的另一种近似适配方法解决了这个问题，该方法引入了一种方案，将概率保证与近似前 $k$ 答案相关联。所提出的方案在信息检索系统的背景下工作，假设存在多个列表，每个列表基于不同的标准（如查询关键词）对底层文档集进行不同的排序。保守的TA阈值测试被概率测试所取代，以估计候选文档最终进入前 $k$ 集合的概率。具体而言，文档 $d$ 的得分高于当前前 $k$ 集合的最低得分 ${M}_{k}$ 的概率计算如下：

$$
\Pr \left( {\mathop{\sum }\limits_{{i \in  E\left( d\right) }}{p}_{i}\left( d\right)  + \mathop{\sum }\limits_{{j \notin  E\left( d\right) }}{\widehat{p}}_{j}\left( d\right)  > {M}_{k}}\right) , \tag{3}
$$

where $E\left( d\right)$ is the set of lists in which $d$ has been encountered, ${p}_{i}\left( d\right)$ is the score of document $d$ in list $i$ ,and ${\widehat{p}}_{j}\left( d\right)$ is an estimator for the score of $d$ in list $j$ ,where it has not been encountered yet. If this probability is below a threshold $\epsilon$ ,the document $d$ is discarded from the candidate set. Setting $\epsilon  = 0$ corresponds to the conservative TA test. The main departure from standard TA is that unseen partial scores are estimated probabilistically instead of setting them, loosely, to the best possible unseen scores. Computing the estimators of unseen partial scores is based on the score distributions of the underlying ranked lists. Three different distributions are considered: uniform, Poisson, and a generic distribution derived from existing histograms.

其中 $E\left( d\right)$ 是遇到文档 $d$ 的列表集合，${p}_{i}\left( d\right)$ 是文档 $d$ 在列表 $i$ 中的得分，${\widehat{p}}_{j}\left( d\right)$ 是文档 $d$ 在尚未遇到它的列表 $j$ 中的得分估计值。如果这个概率低于阈值 $\epsilon$，则将文档 $d$ 从候选集合中剔除。设置 $\epsilon  = 0$ 对应于保守的TA测试。与标准TA的主要区别在于，未见过的部分得分是通过概率估计的，而不是大致将它们设置为可能的最佳未见过得分。计算未见过部分得分的估计值基于底层排序列表的得分分布。考虑了三种不同的分布：均匀分布、泊松分布以及从现有直方图导出的通用分布。

The above approximation technique is illustrated by Algorithm 9. For each index list,the current position and last seen score are maintained. For each item $d$ that is encountered in at least one of the lists,a worst score $\underline{F}\left( d\right)$ is computed by assuming zero scores for $d$ in lists where it has not been encountered yet,and a best score $\bar{F}\left( d\right)$ is computed by assuming the highest possible scores of unseen items in the lists where $d$ has not been encountered yet. The minimum score among the current top- $k$ items, based on items’ worst scores,is stored in ${M}_{k}$ . An item is considered a candidate to join the top- $k$ set if its best score is above ${M}_{k}$ . Periodically,the candidate set is filtered from items whose best scores cannot exceed ${M}_{k}$ anymore,or items that fail the probabilistic test of being in the top- $k$ . This test is performed by computing the probability that item’s total score is above ${M}_{k}$ . If this probability is below the approximation threshold $\epsilon$ ,then the item is discarded from the candidates without computing its score.

上述近似技术由算法9进行说明。对于每个索引列表，维护当前位置和最后看到的得分。对于至少在一个列表中遇到的每个项目 $d$，通过假设 $d$ 在尚未遇到它的列表中的得分为零来计算最差得分 $\underline{F}\left( d\right)$，并通过假设 $d$ 在尚未遇到它的列表中未见过项目的最高可能得分来计算最佳得分 $\bar{F}\left( d\right)$。基于项目的最差得分，当前前 $k$ 项目中的最低得分存储在 ${M}_{k}$ 中。如果一个项目的最佳得分高于 ${M}_{k}$，则将其视为加入前 $k$ 集合的候选项目。定期从候选集合中过滤掉最佳得分不再能超过 ${M}_{k}$ 的项目，或者未能通过进入前 $k$ 的概率测试的项目。通过计算项目的总得分高于 ${M}_{k}$ 的概率来执行此测试。如果这个概率低于近似阈值 $\epsilon$，则在不计算其得分的情况下将该项目从候选项目中剔除。

<!-- Media -->

Algorithm 9. Top-k with Probabilistic Guarantees [Theobold et al. 2004]

算法9. 具有概率保证的前k项 [Theobold等人，2004年]

---

1: $\operatorname{topk} = \left\{  {{\text{dummy }}_{1},\ldots {\text{dummy }}_{k}}\right\}$ with $F\left( {\text{dummy }}_{i}\right)  = 0$

1: $\operatorname{topk} = \left\{  {{\text{dummy }}_{1},\ldots {\text{dummy }}_{k}}\right\}$ 且 $F\left( {\text{dummy }}_{i}\right)  = 0$

2: ${M}_{k} = 0\;\{$ the minimum score in the current topk set $\}$

2: ${M}_{k} = 0\;\{$ 当前前k集合 $\}$ 中的最低得分

	: candidates $= \phi$

	: 候选项目 $= \phi$

	: while index lists ${L}_{i}\left( {\mathrm{i} = 1\text{to}m}\right)$ are not exhausted do

	: 当索引列表 ${L}_{i}\left( {\mathrm{i} = 1\text{to}m}\right)$ 未遍历完时

						$d =$ next item from ${L}_{i}$

						$d =$ 从 ${L}_{i}$ 中获取下一个项目

						${\bar{p}}_{i} = {p}_{i}\left( d\right)$

						add $i$ to $E\left( d\right)$

											 将 $i$ 添加到 $E\left( d\right)$ 中

						set $\underline{F}\left( d\right)  = \mathop{\sum }\limits_{{r \in  E\left( d\right) }}{p}_{r}\left( d\right)$ ,and $\bar{F}\left( d\right)  = \underline{F}\left( d\right)  + \mathop{\sum }\limits_{{r \notin  E\left( d\right) }}{\bar{p}}_{r}$

											 设置 $\underline{F}\left( d\right)  = \mathop{\sum }\limits_{{r \in  E\left( d\right) }}{p}_{r}\left( d\right)$ 和 $\bar{F}\left( d\right)  = \underline{F}\left( d\right)  + \mathop{\sum }\limits_{{r \notin  E\left( d\right) }}{\bar{p}}_{r}$

						if $\bar{F}\left( d\right)  > {M}_{k}$ then

											 如果 $\bar{F}\left( d\right)  > {M}_{k}$ 则

											add $d$ to candidates

																					 将 $d$ 添加到候选项目中

						else

											 否则

											drop $d$ from candidates if present

																					 如果 $d$ 在候选项目中则将其剔除

				end if

							 结束条件判断

						if $\underline{F}\left( d\right)  > {M}_{k}$ then

						如果 $\underline{F}\left( d\right)  > {M}_{k}$ 成立，则

												if $d \notin$ topk then

												如果 $d \notin$ 满足前 k 个条件，则

																				remove the item $d$ with the $\min \underline{F}\left( \text{.}\right)$ in topk

																				从排名前 k 的元素中移除具有 $\min \underline{F}\left( \text{.}\right)$ 的元素 $d$

																				add $d$ to topk

																				将 $d$ 添加到排名前 k 的元素中

																				add $d$ to candidates

																				将 $d$ 添加到候选元素中

							end if

							结束条件判断

							${M}_{k} = \min \{ \underline{F}\left( \acute{d}\right)  \mid  \acute{d} \in  {topk}\}$

				end if

				结束条件判断

						periodically do the following loop

						定期执行以下循环

						for all $d \in$ candidates do

						对所有 $d \in$ 候选元素执行以下操作

										update $\bar{F}\left( d\right)$

										更新 $\bar{F}\left( d\right)$

										$P = \Pr \left( {\mathop{\sum }\limits_{{i \in  E\left( \widehat{d}\right) }}{p}_{i}\left( \widehat{d}\right)  + \mathop{\sum }\limits_{{j \notin  E\left( \widehat{d}\right) }}{\widehat{p}}_{j}\left( \widehat{d}\right)  > {M}_{k}}\right)$

										if $\bar{F}\left( \widehat{d}\right)  < {M}_{k}$ or $P < \epsilon$ then

										如果 $\bar{F}\left( \widehat{d}\right)  < {M}_{k}$ 或 $P < \epsilon$ 成立，则

																drop $\acute{d}$ from candidates

																从候选元素中移除 $\acute{d}$

							end if

							结束条件判断

				end for

				结束循环

						$T = \max \{ \bar{F}\left( \acute{d}\right)  \mid  \acute{d} \in$ candidates $\}$

						$T = \max \{ \bar{F}\left( \acute{d}\right)  \mid  \acute{d} \in$ 候选元素 $\}$

						if candidates $= \phi$ or $T \leq  {M}_{k}$ then

						如果候选元素满足 $= \phi$ 或 $T \leq  {M}_{k}$ 条件，则

										return topk

										返回前k个元素

				end if

				结束条件判断

		end while

		结束循环

---

<!-- Media -->

Reporting approximate top- $k$ answers is also considered in similarity search in multimedia databases. A similarity search algorithm usually uses a distance metric to rank objects according to their distance from a target query object. In higher dimensions, a query can be seen as a hypersphere, centered at the target object. The number of data objects that intersect the query sphere depends on the data distribution, which makes it possible to apply probabilistic methods to provide approximate answers [Amato et al. 2003]. In many cases, even if the query region overlaps a data region, no actual data points (or a small number of data points) appear in the intersection, depending on data distribution. The problem is that there is no way to precisely determine the useless regions, whose intersection with the query region is empty, without accessing such regions. The basic idea in Amato et al. [2003] is to use a proximity measure to decide if a data region should be inspected or not. Only data regions whose proximity to the query region is greater than a specified threshold are accessed. This method is used to rank the nearest neighbors to some target data object in an approximate manner.

在多媒体数据库的相似性搜索中，也会考虑报告近似的前 $k$ 个答案。相似性搜索算法通常使用距离度量，根据对象与目标查询对象的距离对其进行排序。在高维空间中，查询可以看作是以目标对象为中心的超球体。与查询球体相交的数据对象数量取决于数据分布，这使得应用概率方法来提供近似答案成为可能 [Amato等人，2003年]。在许多情况下，即使查询区域与数据区域重叠，根据数据分布，交集处可能没有实际的数据点（或只有少量数据点）。问题在于，在不访问某些区域的情况下，无法精确确定与查询区域交集为空的无用区域。Amato等人 [2003年] 的基本思想是使用接近度度量来决定是否应该检查某个数据区域。只访问与查询区域的接近度大于指定阈值的数据区域。这种方法用于以近似方式对某个目标数据对象的最近邻进行排序。

Approximate top- $k$ query processing has been also studied in peer-to-peer environments. The basic settings involve a query initiator submitting a query (mostly in the form of keywords) to a number of sites that respond back with the top- $k$ answers,based on their local scores. The major problem is how to efficiently coordinate the communication among the respondents in order to aggregate their rankings. The KLEE system [Michel et al. 2005] addresses this problem, where distributed aggregation queries are processed based on index lists located at isolated sites. KLEE assumes no random accesses are made to index lists located at each peer. Message transfers among peers are reduced by encoding messages into lightweight Bloom filters representing data summaries.

在对等网络环境中，也对近似前 $k$ 个查询处理进行了研究。基本设置包括查询发起者向多个站点提交查询（大多以关键词形式），这些站点根据其本地得分返回前 $k$ 个答案。主要问题是如何有效地协调响应者之间的通信，以便汇总他们的排名。KLEE系统 [Michel等人，2005年] 解决了这个问题，该系统基于位于孤立站点的索引列表处理分布式聚合查询。KLEE假设不会对每个对等节点上的索引列表进行随机访问。通过将消息编码为表示数据摘要的轻量级布隆过滤器，减少了对等节点之间的消息传输。

The query processing in KLEE starts by exploring the network to find an initial approximation for the minimum score of the top- $k$ set $\left( {M}_{k}\right)$ . The query initiator sends to each peer ${M}_{k}/k$ as an estimate for the minimum score of the top- $k$ results expected from that peer. When a peer receives a query,it finds the top- $k$ items locally,builds a histogram on their scores, and hashes the items' IDs in each histogram cell into a Bloom filter. Each peer returns back to the initiator a Bloom filter containing the IDs of all items with scores above ${M}_{k}/k$ . The query initiator combines the received answers,extracts the top- $k$ set,and inserts the remaining items into a candidate set. Candidate items are filtered by identifying the IDs of items with high scores. The structure of Bloom filters facilitates spotting the items that are included in the answers of a large number of peers. The good candidates are finally identified and requested from the peers.

KLEE中的查询处理首先探索网络，为前 $k$ 个集合 $\left( {M}_{k}\right)$ 的最小得分找到初始近似值。查询发起者向每个对等节点 ${M}_{k}/k$ 发送一个估计值，作为该对等节点预期的前 $k$ 个结果的最小得分。当一个对等节点收到查询时，它在本地找到前 $k$ 个项目，根据它们的得分构建一个直方图，并将每个直方图单元格中项目的ID哈希到一个布隆过滤器中。每个对等节点向发起者返回一个布隆过滤器，其中包含所有得分高于 ${M}_{k}/k$ 的项目的ID。查询发起者合并收到的答案，提取前 $k$ 个集合，并将其余项目插入候选集合。通过识别高分项目的ID来过滤候选项目。布隆过滤器的结构便于找出包含在大量对等节点答案中的项目。最终确定好的候选项目并向对等节点请求获取。

### 5.2. Uncertain Data

### 5.2. 不确定数据

Uncertain data management has gained more visibility with the emergence of many practical applications in domains like sensor networks, data cleaning, and location tracking, where data is intrinsically uncertain. Many uncertain (probabilistic) data models, for example, Fuhr [1990], Barbará et al. [1992], and Imielinski and Lipski Jr. [1984], have been proposed to capture data uncertainty on different levels. According to many of these models, tuples have membership probability, for example, based on data source reliability, expressing the belief that they should belong to the database. A tuple attribute could also be defined probabilistically as multiple possible values drawn from discrete or continuous domains, for example, a set of possible customer names in a dirty database, or an interval of possible sensor readings.

随着传感器网络、数据清洗和位置跟踪等领域出现许多实际应用，不确定数据管理受到了更多关注，这些领域的数据本质上是不确定的。已经提出了许多不确定（概率）数据模型，例如Fuhr [1990年]、Barbará等人 [1992年] 以及Imielinski和Lipski Jr. [1984年] 的模型，用于在不同层面捕捉数据的不确定性。根据许多此类模型，元组具有成员概率，例如，基于数据源的可靠性，表示它们属于数据库的可信度。元组属性也可以概率性地定义为从离散或连续域中抽取的多个可能值，例如，脏数据库中可能的客户姓名集合，或传感器可能读数的区间。

Many uncertain data models adopt possible worlds semantics, where an uncertain database is viewed as a set of possible instances (worlds) associated with probabilities. Each possible world represents a valid combination of database tuples. The validity of some tuple combination is determined based on the underlying tuple dependencies. For example, two tuples might never appear together in the same world if they represent the same real-world entity. Alternatively, the existence of one tuple in some world might imply the existence of another tuple.

许多不确定数据模型采用可能世界语义，即将不确定数据库视为与概率相关联的一组可能实例（世界）。每个可能世界代表数据库元组的一种有效组合。某些元组组合的有效性是根据底层元组依赖关系确定的。例如，如果两个元组代表同一个现实世界实体，它们可能永远不会出现在同一个世界中。或者，某个世界中一个元组的存在可能意味着另一个元组的存在。

Top- $k$ queries in deterministic databases assume a single ranking dimension,namely, tuples' scores. In probabilistic databases, tuples' probabilities arise as an additional ranking dimension that interacts with tuples' scores. Both tuple probabilities and scores need to be factored in the interpretation of top- $k$ queries in probabilistic databases. For example, it is not meaningful to report a top-scored tuple with insignificant probability. Alternatively, it is not accepted to order tuples by probability, while ignoring their scores. Moreover, combining scores and probabilities using some score aggregation function eliminates uncertainty completely, which may not be meaningful in some cases, and does not conform with the currently adopted probabilistic query models.

确定性数据库中的前 $k$ 查询假定只有一个排序维度，即元组的分数。在概率数据库中，元组的概率成为一个额外的排序维度，它与元组的分数相互作用。在解释概率数据库中的前 $k$ 查询时，元组的概率和分数都需要考虑在内。例如，报告一个概率微不足道但分数最高的元组是没有意义的。或者，只按概率对元组进行排序而忽略它们的分数也是不可接受的。此外，使用某种分数聚合函数将分数和概率结合起来会完全消除不确定性，这在某些情况下可能没有意义，并且不符合目前采用的概率查询模型。

The scores-uncertainty interaction of top- $k$ queries in probabilistic databases was addressed in Soliman et al. [2007], where a processing framework was introduced to find the most probable top- $k$ answers in uncertain databases. The interaction between the concepts of "most probable" and "top- $k$ " is materialized using two new top- $k$ query semantics: (1) U-Top $k$ query: a top- $k$ query that reports a $k$ -length tuple vector with the maximum probability of being top- $k$ across all database possible worlds; and (2) U- $k$ Ranks query: a top- $k$ query that reports a set of $k$ tuples,where each tuple is the most probable tuple to appear at some rank $1\cdots k$ across all database possible worlds. These two interpretations involve both ranking and aggregation of possible worlds.

Soliman 等人（2007 年）探讨了概率数据库中前 $k$ 查询的分数 - 不确定性交互问题，他们引入了一个处理框架来在不确定数据库中找到最可能的前 $k$ 答案。“最可能”和“前 $k$ ”这两个概念之间的交互通过两种新的前 $k$ 查询语义实现：（1）U - 前 $k$ 查询：一种前 $k$ 查询，它报告一个长度为 $k$ 的元组向量，该向量在所有数据库可能世界中成为前 $k$ 的概率最大；（2）U - $k$ 排名查询：一种前 $k$ 查询，它报告一组 $k$ 个元组，其中每个元组是在所有数据库可能世界中某个排名 $1\cdots k$ 上出现概率最大的元组。这两种解释都涉及可能世界的排序和聚合。

<!-- Media -->

<!-- figureText: Most Probable Top-k Answer Rule Engine Prob. Access Space Navigation Processing Layer Tuple Requests Tuples State Formulation Tuple Access Layer Relational Query Engine Access Methods Random Access Score Access Physical Data and Rules Store -->

<img src="https://cdn.noedgeai.com/0195c915-c519-7d98-b2ff-761eddc2c676_40.jpg?x=515&y=280&w=701&h=583&r=0"/>

Fig. 19. Processing framework for uncertain top- $k$ processing [Soliman et al. 2007].

图 19. 不确定前 $k$ 处理的处理框架 [Soliman 等人，2007 年]。

<!-- Media -->

Figure 19 depicts the processing framework introduced in Soliman et al. [2007] to answer top- $k$ queries in uncertain databases. The framework leverages RDBMS storage, indexing, and query processing techniques, in addition to probabilistic inference tools,to compute the most probable top- $k$ answers. The framework contains two main layers, described in the following:

图 19 描绘了 Soliman 等人（2007 年）引入的用于回答不确定数据库中前 $k$ 查询的处理框架。该框架除了利用概率推理工具外，还利用关系数据库管理系统（RDBMS）的存储、索引和查询处理技术来计算最可能的前 $k$ 答案。该框架包含两个主要层，如下所述：

-Tuple access layer. Tuple retrieval, indexing, and traditional query processing (including score-based ranking) are the main functionalities provided by this layer. Uncertain data and probabilistic dependencies are stored in a relational database with different access methods provided to allow the processing layer to retrieve the uncertain tuples.

- 元组访问层。该层提供的主要功能包括元组检索、索引和传统查询处理（包括基于分数的排序）。不确定数据和概率依赖关系存储在关系数据库中，并提供不同的访问方法，以便处理层检索不确定元组。

-Processing layer. The processing layer retrieves uncertain tuples from the underlying storage layer, and efficiently navigates the space of possible worlds to compute the most probable top- $k$ answers.

- 处理层。处理层从底层存储层检索不确定元组，并有效地遍历可能世界的空间，以计算最可能的前 $k$ 答案。

The problem of finding top- $k$ query answers in uncertain databases is formulated as searching the space of states that represent possible top- $k$ answers,where a state is a possible prefix of one or more worlds ordered on score. Each state has a probability equal to the aggregate probability of the possible worlds prefixed by this state. The Rule Engine component in Figure 19 is responsible for computing such probabilities using probabilistic inference tools, for example, Bayesian networks. The search for uncertain top- $k$ answers starts from an empty state with length 0 and ends at a goal state with length $k$ ,having the maximum probability. The proposed search algorithms minimize the number of accessed tuples, and the number of visited space states.

在不确定数据库中寻找前 $k$ 查询答案的问题被表述为搜索表示可能的前 $k$ 答案的状态空间，其中一个状态是按分数排序的一个或多个世界的可能前缀。每个状态的概率等于以该状态为前缀的可能世界的总概率。图 19 中的规则引擎组件负责使用概率推理工具（例如贝叶斯网络）计算此类概率。对不确定前 $k$ 答案的搜索从长度为 0 的空状态开始，到长度为 $k$ 且概率最大的目标状态结束。所提出的搜索算法将访问的元组数量和访问的空间状态数量降至最低。

The problem of finding the $k$ most probable query answers in probabilistic databases was addressed by Ré et al. [2007]. In this setting, probability is the only ranking dimension, since tuples are not scored by a scoring function. Tuples are treated as probabilistic events with the assumption that base tuples correspond to independent events. However, when relational operations, for example, joins and projections, are conducted on base tuples, the independence assumption does not hold any more on the output tuples. Computing the exact probabilities of the answers in this case is generally in #P-complete, which is the class of counting algorithms corresponding to the NP-complete complexity class.

Ré 等人（2007 年）探讨了在概率数据库中寻找 $k$ 个最可能查询答案的问题。在这种情况下，概率是唯一的排序维度，因为元组没有通过评分函数进行评分。元组被视为概率事件，假设基本元组对应于独立事件。然而，当对基本元组进行关系操作（例如连接和投影）时，输出元组不再满足独立性假设。在这种情况下计算答案的精确概率通常是 #P 完全问题，这是与 NP 完全复杂度类相对应的计数算法类。

To address the above challenges, Ré et al. [2007] proposed a multisimulation algorithm (MS_Topk) based on Monte-Carlo simulation. In MS_Topk, computing the exact probability of an answer is relaxed in favor of computing the correct ranking efficiently. ${\mathrm{{MS}}}_{ - }$ Topk maintains a probability interval for each candidate top- $k$ answer enclosing its exact probability. In each step, the probability intervals of some candidates are tightened by generating random possible worlds, and testing whether these candidates belong to such worlds or not. The probability intervals of candidate answers are progressively tightened until there are $k$ intervals with no other intervals overlapping with their minimum bounds,and hence the top- $k$ answers are obtained. Ré et al. [2007] also proposed other variants of the MS_Topk algorithm to sort the top- $k$ answers and incrementally report the top- $k$ answers one by one.

为应对上述挑战，雷（Ré）等人 [2007] 提出了一种基于蒙特卡罗模拟的多模拟算法（MS_Topk）。在 MS_Topk 算法中，放宽了对答案精确概率的计算要求，转而注重高效地计算正确排名。${\mathrm{{MS}}}_{ - }$ Topk 为每个候选的前 $k$ 个答案维护一个包含其精确概率的概率区间。在每一步中，通过生成随机的可能世界并测试这些候选答案是否属于这些世界，来收紧某些候选答案的概率区间。候选答案的概率区间会逐步收紧，直到有 $k$ 个区间的最小边界不与其他任何区间重叠，从而得到前 $k$ 个答案。雷（Ré）等人 [2007] 还提出了 MS_Topk 算法的其他变体，用于对前 $k$ 个答案进行排序，并逐个递增地报告前 $k$ 个答案。

## 6. RANKING FUNCTION

## 6. 排名函数

In this section, we discuss the properties of different ranking functions adopted by top- $k$ processing techniques. Top- $k$ processing techniques are classified into three categories based on the type of ranking functions they assume. The first category, which includes the majority of current techniques, assumes monotone ranking functions. The second category allows for generic ranking functions. The third category leaves the ranking function unspecified.

在本节中，我们讨论前 $k$ 处理技术所采用的不同排名函数的性质。前 $k$ 处理技术根据其所假定的排名函数类型分为三类。第一类包括当前的大多数技术，假定使用单调排名函数。第二类允许使用通用排名函数。第三类则不指定排名函数。

### 6.1. Monotone Ranking Functions

### 6.1. 单调排名函数

The majority of top- $k$ techniques assumes monotone scoring functions,for example, TA [Fagin et al. 2001], and UPPER [Bruno et al. 2002b]. Using monotone ranking functions is common in many practical applications, especially in Web settings [Marian et al. 2004]. For example,many top- $k$ processing scenarios involve linear combinations of multiple scoring predicates, or maximum/minimum functions, which are all monotone.

大多数前 $k$ 技术假定使用单调评分函数，例如 TA [法金（Fagin）等人 2001] 和 UPPER [布鲁诺（Bruno）等人 2002b]。在许多实际应用中，尤其是在网络环境中 [玛丽安（Marian）等人 2004]，使用单调排名函数很常见。例如，许多前 $k$ 处理场景涉及多个评分谓词的线性组合，或最大/最小函数，这些都是单调的。

Monotone ranking functions have special properties that can be exploited for efficient processing of top- $k$ queries. As demonstrated by our previous discussion of many top- $k$ processing techniques,when aggregating objects’ scores from multiple ranked lists using a monotone score aggregation function, an upper bound of the scores of unseen objects is easily derived. To illustrate,assume that we sequentially scan $m$ sorted lists, ${L}_{1},\ldots ,{L}_{m}$ ,for the same set of objects based on different ranking predicates. If the scores of the last retrieved objects from these lists are ${\bar{p}}_{1},{\bar{p}}_{2},\ldots ,{\bar{p}}_{m}$ ,then an upper bound, $\bar{F}$ ,over the scores of all unseen objects is computed as $\bar{F} = F\left( {{\bar{p}}_{1},{\bar{p}}_{2},\ldots ,{\bar{p}}_{m}}\right)$ . It is easy to verify that no unseen object can possibly have a score greater than the above upper bound $\bar{F}$ ; by contradiction,assume that an unseen object ${o}_{u}$ has a score greater than $\bar{F}$ . Then,based on $F$ ’s monotonicity, ${o}_{u}$ must have at least one ranking predicate ${p}_{i}$ with score greater than the last seen score ${\bar{p}}_{i}$ . This implies that the retrieval from list ${L}_{i}$ is not score-ordered,which contradicts with the original assumptions. Additionally, monotone scoring functions allow computing a score upper bound for each seen object $o$ by substituting the values of unknown ranking predicates of $o$ with the last seen scores in the corresponding lists. A top- $k$ processing algorithm halts when there are $k$ objects whose score lower bounds are not below the score upper bounds of all other objects, including the unseen objects.

单调排名函数具有一些特殊性质，可用于高效处理前 $k$ 查询。正如我们之前对许多前 $k$ 处理技术的讨论所示，当使用单调分数聚合函数从多个排序列表中聚合对象的分数时，很容易得出未见过对象分数的上界。为了说明这一点，假设我们根据不同的排名谓词依次扫描 $m$ 个排序列表 ${L}_{1},\ldots ,{L}_{m}$，针对同一组对象。如果从这些列表中最后检索到的对象的分数是 ${\bar{p}}_{1},{\bar{p}}_{2},\ldots ,{\bar{p}}_{m}$，那么所有未见过对象分数的上界 $\bar{F}$ 计算为 $\bar{F} = F\left( {{\bar{p}}_{1},{\bar{p}}_{2},\ldots ,{\bar{p}}_{m}}\right)$。很容易验证，没有未见过的对象的分数可能大于上述上界 $\bar{F}$；通过反证法，假设一个未见过的对象 ${o}_{u}$ 的分数大于 $\bar{F}$。那么，基于 $F$ 的单调性，${o}_{u}$ 必须至少有一个排名谓词 ${p}_{i}$ 的分数大于最后看到的分数 ${\bar{p}}_{i}$。这意味着从列表 ${L}_{i}$ 中的检索不是按分数排序的，这与原始假设相矛盾。此外，单调评分函数允许通过用相应列表中最后看到的分数替换对象 $o$ 未知排名谓词的值，为每个已见过的对象 $o$ 计算分数上界。当前 $k$ 处理算法中，当有 $k$ 个对象的分数下界不低于所有其他对象（包括未见过的对象）的分数上界时，算法停止。

The above properties are exploited in various algorithms, for example, TA [Fagin et al. 2001] and Quick-Combine [Güntzer et al. 2001] (discussed in Section 3.1), to guarantee early termination of top- $k$ processing. An important property that follows from monotonicity is that these algorithms are instance-optimal, within some bounds, in the sense that, for any database instance, there is no other algorithm that can retrieve less number of objects and return a correct answer. For proof of instance optimality, we refer the reader to Fagin et al. [2001].

上述性质在各种算法中得到了应用，例如 TA [法金（Fagin）等人 2001] 和 Quick - Combine [京策尔（Güntzer）等人 2001]（在第 3.1 节中讨论），以保证前 $k$ 处理的提前终止。从单调性得出的一个重要性质是，这些算法在一定范围内是实例最优的，也就是说，对于任何数据库实例，没有其他算法能够检索更少的对象并返回正确答案。关于实例最优性的证明，请读者参考法金（Fagin）等人 [2001] 的文献。

Linear ranking functions constitute a subset of monotone functions. Linear ranking functions define the aggregated score of ranking predicates as a weighted sum. Several top- $k$ techniques exploit the geometrical properties of linear functions to efficiently retrieve the top- $k$ answers,such as the Onion Indices [Chang et al. 2000] and Ranked Join Indices [Tsaparas et al. 2003], discussed in Section 4.1.2. Linear functions can be represented as vectors based on the weights associated with the ranking predicates. Such representation can be used to geometrically compute objects' scores as projections on the ranking function vector. Other properties such as the relation between linear functions and convex hulls of the data points are also exploited as in Chang et al. [2000].

线性排名函数是单调函数的一个子集。线性排名函数将排名谓词的聚合得分定义为加权和。几种前 $k$ 技术利用线性函数的几何性质来高效检索前 $k$ 个答案，例如第4.1.2节中讨论的洋葱索引（Onion Indices）[Chang等人，2000年]和排名连接索引（Ranked Join Indices）[Tsaparas等人，2003年]。线性函数可以基于与排名谓词相关联的权重表示为向量。这种表示可用于通过几何方式将对象的得分计算为排名函数向量上的投影。其他性质，如线性函数与数据点凸包之间的关系，也如Chang等人[2000年]那样被利用。

### 6.2. Generic Ranking Functions

### 6.2. 通用排名函数

Using nonmonotone ranking functions in top- $k$ queries is challenging. One reason is that it is not straightforward to prune objects that do not qualify to query answer at an early stage. The methods used to upper-bound the scores of unseen (or partially seen) objects are not applicable to nonmonotone functions. For example, assume two sorted lists based on ranking predicates ${p}_{1}$ and ${p}_{2}$ ,and a scoring function $F\left( t\right)  = {p}_{1}\left( t\right) /{p}_{2}\left( t\right)$ . The function $F$ in this case is obviously nonmonotone. The last seen scores ${\bar{p}}_{1}$ and ${\bar{p}}_{2}$ cannot be used to derive a score upper bound for the unseen objects. The reason is that it is possible to find objects with scores above ${\bar{p}}_{1}/{\bar{p}}_{2}$ later on if the unseen scores in the first list are all equal to ${\bar{p}}_{1}$ ,while the unseen scores in the second list, ${p}_{2}$ ,decrease.

在前 $k$ 查询中使用非单调排名函数具有挑战性。一个原因是在早期阶段修剪不符合查询答案条件的对象并不直接。用于对未见过（或部分见过）对象的得分进行上界估计的方法不适用于非单调函数。例如，假设基于排名谓词 ${p}_{1}$ 和 ${p}_{2}$ 的两个排序列表，以及一个评分函数 $F\left( t\right)  = {p}_{1}\left( t\right) /{p}_{2}\left( t\right)$。在这种情况下，函数 $F$ 显然是非单调的。最后看到的得分 ${\bar{p}}_{1}$ 和 ${\bar{p}}_{2}$ 不能用于推导未见过对象的得分上界。原因是，如果第一个列表中未见过的得分都等于 ${\bar{p}}_{1}$，而第二个列表 ${p}_{2}$ 中未见过的得分降低，那么之后有可能找到得分高于 ${\bar{p}}_{1}/{\bar{p}}_{2}$ 的对象。

Some recent proposals have addressed the challenges imposed by generic (not necessarily monotone) ranking functions. The technique proposed in Zhang et al. [2006] supports arbitrary ranking functions by modeling top- $k$ query as an optimization problem. The optimization goal function consists of a Boolean expression that filters tuples based on query predicates, and a ranking function that determines the score of each tuple. The goal function is equal to zero whenever a tuple does not satisfy the Boolean expression,and it is equal to the tuple’s score otherwise. The answer to the top- $k$ query is the set of $k$ tuples with the highest values of the goal function.

最近的一些提议已经解决了通用（不一定是单调的）排名函数带来的挑战。Zhang等人[2006年]提出的技术通过将前 $k$ 查询建模为一个优化问题来支持任意排名函数。优化目标函数由一个基于查询谓词过滤元组的布尔表达式和一个确定每个元组得分的排名函数组成。当元组不满足布尔表达式时，目标函数等于零，否则等于元组的得分。前 $k$ 查询的答案是目标函数值最高的 $k$ 个元组的集合。

In order to efficiently search for the top- $k$ answers,existing indexes of scoring predicates are used. The optimization problem is solved using an ${\mathcal{A}}^{ * }$ search algorithm,named ${OP}{T}^{ * }$ ,by transforming the problem into a shortest path problem as follows. Each state in the search space is created by joining index nodes. A state thus covers subsets of predicate domains. Two types of states are defined at different index levels. The first type is region states, which represents internal index nodes, while the second type is tuple states, which represents index leaf nodes (i.e., the tuple level of the index). A transition between two states is achieved by traversing the indexes. A dummy goal state is created so that it can be reached from any tuple state, where the distance from a tuple state to the goal state is equal to the inverse of tuple's score. Distances between other state pairs are set to zero. Based on this formulation, the shortest path to the goal state involves the tuples with the maximum scores. To reach the goal state in a small number of state transitions, a heuristic function is used to guide transitions using the maximum possible score of the current state. The authors proved that the OPT* algorithm is optimal in the number of visited leaves, tuples, and internal states under certain constraints over the weights associated with each term in the optimization function.

为了高效地搜索前 $k$ 个答案，使用了评分谓词的现有索引。通过将问题转换为如下的最短路径问题，使用一个 ${\mathcal{A}}^{ * }$ 搜索算法（名为 ${OP}{T}^{ * }$）来解决优化问题。搜索空间中的每个状态是通过连接索引节点创建的。因此，一个状态覆盖了谓词域的子集。在不同的索引级别定义了两种类型的状态。第一种类型是区域状态，它表示内部索引节点，而第二种类型是元组状态，它表示索引叶节点（即索引的元组级别）。通过遍历索引实现两个状态之间的转换。创建一个虚拟目标状态，以便可以从任何元组状态到达它，其中从元组状态到目标状态的距离等于元组得分的倒数。其他状态对之间的距离设置为零。基于这种表述，到目标状态的最短路径涉及得分最高的元组。为了在少量状态转换中到达目标状态，使用一个启发式函数，根据当前状态的最大可能得分来引导转换。作者证明，在优化函数中与每个项相关联的权重的某些约束下，OPT* 算法在访问的叶节点、元组和内部状态的数量方面是最优的。

<!-- Media -->

<!-- figureText: A.root (a1, a2, a3) (A.root, B.root) Joined states omitted (a1,b1) (a1,b2) (a1,b3) (a2,b1) (a3,b3) (a) Tree-structured joined states State: (a2,b2) (a3,b1) f(S): 0 0 25 1296 (b) States sorted by $f\left( S\right)$ [10-30] [50-54] [72-85] 10,t1 20,t2 30,t3 50,t4 54,t5 72,t6 75,t7 80,48 B.root (b1, b2, b3) [10-36] [40-45] [60-65] 10,15 30,16 36,17 40,t1 45,t4 60,t2 62,48 65,13 -->

<img src="https://cdn.noedgeai.com/0195c915-c519-7d98-b2ff-761eddc2c676_43.jpg?x=331&y=283&w=1078&h=320&r=0"/>

Fig. 20. Index Merge [Xin et al. 2007]. (a) Indices on predicates A and B; (b) space of joint states.

图20. 索引合并 [Xin等人，2007年]。(a) 谓词A和B上的索引；(b) 联合状态空间。

<!-- Media -->

Ad hoc ranking functions were addressed by Xin et al. [2007], with the restriction that the function is lower-bounded. A ranking function $F$ is lower-bounded in a region $\Omega$ of its variables domains,if the lower bound of $F$ in $\Omega$ can be derived. Examples include $F = {\left( x - y\right) }^{2}$ . The authors presented an index-merge framework that performs progressive search over a space of states composed by joining index nodes. The main idea is to exploit existing B-Tree and R-Tree indexes of ranking predicates to create a search space of possible query answers. A state in this space is composed by joining multiple index nodes. Promising search states, in terms of their involved scores, are progressively materialized,while the states with no chances of yielding the top- $k$ answers are early-pruned. The search algorithm prioritizes visiting space states based on their score lower bounds. For each state, an internal heap is maintained to prioritize traversing the state's children (subspaces).

Xin等人[2007]研究了特定的排序函数（ad hoc ranking functions），并对函数设置了下界限制。如果能推导出排序函数$F$在其变量域的某个区域$\Omega$内的下界，则称该函数在该区域内是有下界的。示例包括$F = {\left( x - y\right) }^{2}$。作者提出了一个索引合并框架，该框架对由连接索引节点组成的状态空间进行渐进式搜索。其主要思想是利用现有的排序谓词的B树和R树索引来创建一个可能的查询答案搜索空间。该空间中的一个状态由多个索引节点连接而成。根据所涉及的分数，有希望的搜索状态会逐步实现，而那些不可能产生前$k$个答案的状态则会被提前剪枝。搜索算法根据状态的分数下界来优先访问空间状态。对于每个状态，会维护一个内部堆来对遍历该状态的子状态（子空间）进行优先级排序。

To understand the search space of the above technique, consider Figure 20, which shows the search space for a ranking function $f = {\left( A - B\right) }^{2}$ ,where $A$ and $B$ are two ranking predicates. Figure 20(a) shows two existing indices on predicates $A$ and $B$ , while Figure 20(b) shows the generated search space, and the computed score lower bounds of different states. Let ${I1}$ and ${I2}$ be the two indices defined on predicates $A$ and $B$ ,respectively. The search space formed by joining ${I1}$ and ${I2}$ is constructed based on the hierarchical index structure. That is, for any joint state (I1.n1, I2.n2), its child states are created as the Cartesian products of child nodes of ${I1}.{n1}$ and ${I1}.{n2}$ . For example, in Figure 20(a), the joint state (A.root, B.root) has the following children: $\{ \left( {{a1},{b1}}\right) ,\left( {{a1},{b2}}\right) ,\left( {{a1},{b3}}\right) ,\left( {{a2},{b1}}\right) ,\left( {{a2},{b2}}\right) ,\left( {{a2},{b3}}\right) ,\left( {{a3},{b1}}\right) ,\left( {{a3},{b2}}\right) ,\left( {{a3},{b3}}\right) \}$ . Score lower bounds of different states are used to order state traversal. For example, the state (a2,b2)has a score lower bound of 25,which is computed based on the square of the difference between 50 and 45,which are possible tuple scores in index nodes ${a2}$ and ${b2}$ , respectively.

为了理解上述技术的搜索空间，请参考图20，该图展示了排序函数$f = {\left( A - B\right) }^{2}$的搜索空间，其中$A$和$B$是两个排序谓词。图20(a)展示了谓词$A$和$B$上的两个现有索引，而图20(b)展示了生成的搜索空间以及不同状态的计算得分下界。设${I1}$和${I2}$分别是定义在谓词$A$和$B$上的两个索引。通过连接${I1}$和${I2}$形成的搜索空间是基于分层索引结构构建的。也就是说，对于任何联合状态 (I1.n1, I2.n2)，其子状态是${I1}.{n1}$和${I1}.{n2}$的子节点的笛卡尔积。例如，在图20(a)中，联合状态 (A.root, B.root) 有以下子状态：$\{ \left( {{a1},{b1}}\right) ,\left( {{a1},{b2}}\right) ,\left( {{a1},{b3}}\right) ,\left( {{a2},{b1}}\right) ,\left( {{a2},{b2}}\right) ,\left( {{a2},{b3}}\right) ,\left( {{a3},{b1}}\right) ,\left( {{a3},{b2}}\right) ,\left( {{a3},{b3}}\right) \}$。不同状态的得分下界用于对状态遍历进行排序。例如，状态 (a2,b2) 的得分下界为25，这是根据索引节点${a2}$和${b2}$中可能的元组得分50和45之间的差值的平方计算得出的。

#### 6.3.No Ranking Function (Skyline Queries)

#### 6.3.无排序函数（天际线查询）

In some applications, it might not be straightforward to define a ranking function. For example, when query answers are to be returned to more than one user, it might be more meaningful to report all "interesting" answers, rather than a strict ranking according to some specified ranking function.

在某些应用中，定义排序函数可能并不直接。例如，当查询答案要返回给多个用户时，报告所有“有趣”的答案可能比根据某个指定的排序函数进行严格排序更有意义。

A skyline query returns the objects that are not dominated by any other objects restricted to a set of dimensions (predicates). Figure 21 shows the skyline of a number of objects based on two predicates ${p}_{1}$ and ${p}_{2}$ ,where larger values of both predicates are more favored. An object $X$ dominates object $Y$ if $\left( {X.{p}_{1} > Y.{p}_{1}}\right.$ and $\left. {X.{p}_{2} \geq  Y.{p}_{2}}\right)$ or $\left( {X.{p}_{1} \geq  Y.{p}_{1}}\right.$ and $\left. {X.{p}_{2} > Y.{p}_{2}}\right)$ . For example ${t}_{4}$ dominates all objects in the shaded rectangle in Figure 21. All the objects that lie on the skyline are not dominated by any other object. An interesting property of the skyline is that the top-1 object, based on any monotone ranking function, must be one of the skyline objects. The Onion Indices [Chang et al. 2000] make use of this property by materializing and indexing the $k$ first skylines and then answering the query for some ranking function by only searching the skyline objects.

天际线查询（skyline query）返回的对象在一组维度（谓词）上不被其他任何对象所支配。图21展示了基于两个谓词 ${p}_{1}$ 和 ${p}_{2}$ 的多个对象的天际线，其中两个谓词的值越大越受青睐。如果 $\left( {X.{p}_{1} > Y.{p}_{1}}\right.$ 且 $\left. {X.{p}_{2} \geq  Y.{p}_{2}}\right)$ ，或者 $\left( {X.{p}_{1} \geq  Y.{p}_{1}}\right.$ 且 $\left. {X.{p}_{2} > Y.{p}_{2}}\right)$ ，则对象 $X$ 支配对象 $Y$ 。例如， ${t}_{4}$ 支配图21中阴影矩形内的所有对象。所有位于天际线上的对象都不被其他任何对象所支配。天际线的一个有趣特性是，基于任何单调排序函数的排名第一的对象，必定是天际线对象之一。洋葱索引（Onion Indices）[Chang等人，2000]利用了这一特性，通过物化并索引 $k$ 个初始天际线，然后仅通过搜索天际线对象来回答某些排序函数的查询。

<!-- Media -->

<!-- figureText: ${p}_{2}$ ${P}_{1}$ Area dominated by ${\mathrm{t}}_{4}$ -->

<img src="https://cdn.noedgeai.com/0195c915-c519-7d98-b2ff-761eddc2c676_44.jpg?x=588&y=285&w=545&h=483&r=0"/>

Fig. 21. Skyline query.

图21. 天际线查询。

<!-- Media -->

There is a large body of research that addresses skyline related queries. Details of skyline query processing techniques are out of the scope of this survey.

有大量研究致力于解决与天际线相关的查询问题。天际线查询处理技术的详细内容超出了本综述的范围。

## 7. TOP- $K$ QUERY PROCESSING IN XML DATABASES

## 7. XML数据库中的前 $K$ 查询处理

Top- $k$ processing in XML databases has recently gained more attention since XML has become the preferred medium for formatting and exchanging data in many domains such as the Web and e-commerce. Top- $k$ queries are dominant type of queries in such domains.

由于XML已成为许多领域（如Web和电子商务）中数据格式化和交换的首选媒介，XML数据库中的前 $k$ 处理最近受到了更多关注。前 $k$ 查询是这些领域中占主导地位的查询类型。

In XML queries, structure and content constraints are usually specified to express the characteristics of the required query answers. XML elements that (partially) match query conditions are usually associated with relevance scores capturing their similarity to the query conditions. It is often required to rank query matches based on their relevance scores,or report only the top- $k$ matches. There are two main issues that need to be addressed in this type of queries. The first issue is what kind of scoring functions can be used to measure the relevance of answers with respect to query conditions. The second issue is how to exploit scoring functions to prune the less relevant elements as early as possible. In this section we discuss these two issues and give an overview for multiple examples of top- $k$ techniques in XML databases.

在XML查询中，通常会指定结构和内容约束来表达所需查询答案的特征。与查询条件（部分）匹配的XML元素通常会关联一个相关性得分，以反映它们与查询条件的相似度。通常需要根据相关性得分对查询匹配结果进行排序，或者仅报告前 $k$ 个匹配结果。这类查询需要解决两个主要问题。第一个问题是可以使用哪种评分函数来衡量答案相对于查询条件的相关性。第二个问题是如何利用评分函数尽早地剔除相关性较低的元素。在本节中，我们将讨论这两个问题，并概述XML数据库中前 $k$ 技术的多个示例。

#### 7.1.The TopX System

#### 7.1. TopX系统

It is typical in XML databases that multiple indexes are built for different content and/or structure conditions. Processing multiple indexes to identify the top- $k$ objects is extensively studied by top- $k$ algorithms in relational databases,for example,the TA family of algorithms [Fagin et al. 2001]. This has motivated approaches that extend relational top- $k$ algorithms in XML settings. The TopX system [Theobald et al. 2005] is one example of such approaches. TopX builds on previous work [Theobald et al. 2004], discussed in Section 5.1.2. The proposed techniques focus on inexpensive sorted accesses to ranked document lists, with scheduled random accesses. For each document, local scores coming from ranked lists are aggregated into global scores based a monotonic score aggregation function such as weighted summation.

在XML数据库中，通常会针对不同的内容和/或结构条件构建多个索引。关系数据库中的前 $k$ 算法（例如TA算法家族 [Fagin等人，2001]）对处理多个索引以识别前 $k$ 个对象进行了广泛研究。这促使人们提出了在XML环境中扩展关系型前 $k$ 算法的方法。TopX系统 [Theobald等人，2005] 就是这类方法的一个示例。TopX基于第5.1.2节中讨论的先前工作 [Theobald等人，2004] 构建。所提出的技术侧重于对排序后的文档列表进行低成本的顺序访问，并安排随机访问。对于每个文档，来自排序列表的局部得分会基于单调得分聚合函数（如加权求和）聚合为全局得分。

Ranked lists in TopX maintain different orders for the documents corpus based on different content and structure conditions. These lists are implemented as relational tables indexed using various B+-tree indexes. The ranked lists are used primarily to evaluate content conditions in block-scan fashion. The evaluation of expensive structure conditions is postponed or avoided by scheduling random accesses only when they are cost-beneficial.

TopX中的排序列表根据不同的内容和结构条件为文档语料库维护不同的顺序。这些列表以关系表的形式实现，并使用各种B + 树索引进行索引。排序列表主要用于以块扫描的方式评估内容条件。通过仅在成本效益高时安排随机访问，可以推迟或避免对昂贵的结构条件的评估。

The computational model of TopX is based on the traditional XML element-tree model, where each tree node has a tag and content. The full content of a node is defined as the concatenation of the contents of all node's descendants. Different content scoring measures are adopted based on the contents,or the full contents of a node $n$ with $\operatorname{tag}A$ :

TopX的计算模型基于传统的XML元素树模型，其中每个树节点都有一个标签和内容。节点的完整内容定义为该节点所有后代节点内容的串联。根据节点 $n$ 与 $\operatorname{tag}A$ 的内容或完整内容，采用不同的内容评分度量：

-Term frequency, ${tf}\left( {t,n}\right)$ ,of term $t$ in node $n$ ,is the number of occurrences of $t$ in the content of $n$ .

-词项 $t$ 在节点 $n$ 中的词频 ${tf}\left( {t,n}\right)$ ，是 $t$ 在 $n$ 内容中出现的次数。

-Full term frequency, ${ftf}\left( {t,n}\right)$ ,of term $t$ in node $n$ ,is the number of occurrences of $t$ in the full content of $n$ .

-词项 $t$ 在节点 $n$ 中的全词频 ${ftf}\left( {t,n}\right)$ ，是 $t$ 在 $n$ 完整内容中出现的次数。

-Tag frequency, ${N}_{A}$ ,of $\operatorname{tag}A$ ,is the number of nodes with $\operatorname{tag}A$ in the entire document corpus.

-标签频率 ${N}_{A}$（即 $\operatorname{tag}A$ 的标签频率）是整个文档语料库中带有 $\operatorname{tag}A$ 的节点数量。

-Element frequency, $e{f}_{A}\left( t\right)$ ,of term $t$ with respect to $\operatorname{tag}A$ ,is the number of nodes with tag $A$ that contain $t$ in their full contents in the entire document corpus.

-元素频率 $e{f}_{A}\left( t\right)$（即术语 $t$ 相对于 $\operatorname{tag}A$ 的元素频率）是整个文档语料库中标签为 $A$ 且完整内容包含 $t$ 的节点数量。

For example,consider the content condition " $A//{t}_{1},{t}_{2}\ldots {t}_{m}$ ",where $A$ is a tag name and ${t}_{1}\cdots {t}_{m}$ are terms that occur in the full contents of $A$ . The score of node $n$ with tag $A$ ,with respect to the above condition,is computed as follows:

例如，考虑内容条件 “ $A//{t}_{1},{t}_{2}\ldots {t}_{m}$ ”，其中 $A$ 是标签名，${t}_{1}\cdots {t}_{m}$ 是出现在 $A$ 完整内容中的术语。标签为 $A$ 的节点 $n$ 相对于上述条件的得分计算如下：

$$
\operatorname{score}\left( {n,A//{t}_{1}\ldots {t}_{m}}\right)  = \frac{\mathop{\sum }\limits_{{i = 1}}^{m}{\text{ relevance }}_{i} \cdot  {\text{ specificity }}_{i}}{\text{ compactness }\left( n\right) }, \tag{4}
$$

where relevance ${}_{i}$ reflects ftf values,specificity ${}_{i}$ reflects ${N}_{A}$ and $e{f}_{A}\left( t\right)$ values,and compactness(n) considers the subtree or element size for normalization. This content scoring function is based on Okapi BM25 scoring model [Robertson and Walker 1994]. The following example illustrates the processing of TopX to answer top- $k$ queries.

其中相关性 ${}_{i}$ 反映了 ftf 值，特异性 ${}_{i}$ 反映了 ${N}_{A}$ 和 $e{f}_{A}\left( t\right)$ 的值，紧凑性(n) 考虑子树或元素大小以进行归一化处理。此内容评分函数基于 Okapi BM25 评分模型（[罗伯逊和沃克，1994 年]）。以下示例说明了 TopX 处理前 $k$ 个查询的过程。

Example 7.1 (TopX Example). We consider the example illustrated by Theobald et al. [2004] and depicted by Figure 22, where a sample document set composed of three documents is used for illustration. The numbers beside elements refer to their order in pre-order tree traversal. Consider the following query: //A[.//B[.//"b"] and .//C[./"c"]]. This query requests elements with tag name $A$ and two child tags $B$ and $C$ containing terms $b$ and $c$ ,respectively. Table IV shows the element scores of different content conditions computed as ftf scores normalized by the number of terms in a subtree, for simplicity. For instance,the (tag:term) pair $\left( {A : a}\right)$ for element 10 in document ${d}_{3}$ has a score of $1/4$ because the term $a$ occurs twice among the eight terms under element 10 subtree.

示例 7.1（TopX 示例）。我们考虑西奥博尔德等人 [2004 年] 所举并由图 22 所示的示例，其中使用由三个文档组成的样本文档集进行说明。元素旁边的数字指的是它们在先序树遍历中的顺序。考虑以下查询：//A[.//B[.//"b"] and .//C[./"c"]]。此查询请求标签名为 $A$ 且有两个子标签 $B$ 和 $C$ 分别包含术语 $b$ 和 $c$ 的元素。为简单起见，表 IV 显示了不同内容条件下的元素得分，这些得分是通过将 ftf 得分除以子树中的术语数量进行归一化计算得到的。例如，文档 ${d}_{3}$ 中元素 10 的（标签:术语）对 $\left( {A : a}\right)$ 的得分为 $1/4$，因为术语 $a$ 在元素 10 子树下的八个术语中出现了两次。

In Example 7.1,finding the top- $k$ matches is done by opening index scans for the two tag:term conditions $\left( {B : b\text{,and}C : c}\right)$ ,and block-fetching the best document for each of these two conditions. For $\left( {B : b}\right)$ ,the first three entries of index list 2 that belong to the same document ${d}_{1}$ are fetched in a block-scan based on document IDs. Scanning the indexes proceeds in a round-robin fashion among all indexes. A score interval $\left\lbrack  {\text{worstscore}\left( d\right) ,\text{bestscore}\left( d\right) }\right\rbrack$ is computed for each candidate document $d$ ,and is updated periodically based on the current candidate scores and the score upper bound of the unseen candidate elements in each list.

在示例 7.1 中，查找前 $k$ 个匹配项的方法是为两个标签:术语条件 $\left( {B : b\text{,and}C : c}\right)$ 打开索引扫描，并为这两个条件分别批量提取最佳文档。对于 $\left( {B : b}\right)$，基于文档 ID 以块扫描的方式提取索引列表 2 中属于同一文档 ${d}_{1}$ 的前三个条目。所有索引以轮询方式进行扫描。为每个候选文档 $d$ 计算得分区间 $\left\lbrack  {\text{worstscore}\left( d\right) ,\text{bestscore}\left( d\right) }\right\rbrack$，并根据当前候选得分和每个列表中未查看候选元素的得分上限定期更新该区间。

<!-- Media -->

<!-- figureText: ${d}_{1}$ ${d}_{2}$ ${d}_{3}$ 1:Z 6:B 2:B 6:B 8:X 7:C 3:X 7:C 9:B 10:A abc 4:C 5:A acc bb 11:C 12:C bb aaaa aabbc xyz 1:A 1:A 2:A 6:B 2:X 3:X 7:X 3:B 4:B 5:C 8:B 9:C 4:B 5:C ab aacc bbb CCCXY CCC -->

<img src="https://cdn.noedgeai.com/0195c915-c519-7d98-b2ff-761eddc2c676_46.jpg?x=366&y=282&w=999&h=407&r=0"/>

Fig. 22. XML document set.

图 22. XML 文档集。

Table IV. Elements Scores for Different Content

表 IV. 不同内容的元素得分

<table><tr><td colspan="6">Conditions</td></tr><tr><td>Index</td><td>Tag</td><td>Term</td><td>Score</td><td>DocID</td><td>Preorder</td></tr><tr><td>1</td><td>A</td><td>a</td><td>1</td><td>d3</td><td>5</td></tr><tr><td>1</td><td>A</td><td>a</td><td>1/4</td><td>d3</td><td>10</td></tr><tr><td>1</td><td>A</td><td>a</td><td>1/2</td><td>d1</td><td>2</td></tr><tr><td>1</td><td>A</td><td>a</td><td>2/9</td><td>d2</td><td>1</td></tr><tr><td>2</td><td>B</td><td>b</td><td>1</td><td>d1</td><td>8</td></tr><tr><td>2</td><td>B</td><td>b</td><td>1/2</td><td>d1</td><td>4</td></tr><tr><td>2</td><td>B</td><td>b</td><td>3/7</td><td>d1</td><td>6</td></tr><tr><td>2</td><td>B</td><td>b</td><td>1</td><td>d3</td><td>9</td></tr><tr><td>2</td><td>B</td><td>b</td><td>1/3</td><td>d3</td><td>2</td></tr><tr><td>2</td><td>B</td><td>b</td><td>2/3</td><td>d2</td><td>4</td></tr><tr><td>2</td><td>B</td><td>b</td><td>1/3</td><td>d2</td><td>3</td></tr><tr><td>2</td><td>B</td><td>b</td><td>1/3</td><td>d2</td><td>6</td></tr><tr><td>3</td><td>C</td><td>C</td><td>1</td><td>d2</td><td>5</td></tr><tr><td>3</td><td>C</td><td>C</td><td>1/3</td><td>d2</td><td>7</td></tr><tr><td>3</td><td>C</td><td>C</td><td>2/3</td><td>d3</td><td>7</td></tr><tr><td>3</td><td>C</td><td>C</td><td>1/5</td><td>d3</td><td>11</td></tr><tr><td>3</td><td>C</td><td>C</td><td>3/5</td><td>d1</td><td>9</td></tr><tr><td>3</td><td>C</td><td>C</td><td>1/2</td><td>d1</td><td>5</td></tr></table>

<table><tbody><tr><td colspan="6">条件</td></tr><tr><td>索引</td><td>标签</td><td>术语</td><td>得分</td><td>文档编号</td><td>前序遍历</td></tr><tr><td>1</td><td>A</td><td>a</td><td>1</td><td>d3</td><td>5</td></tr><tr><td>1</td><td>A</td><td>a</td><td>1/4</td><td>d3</td><td>10</td></tr><tr><td>1</td><td>A</td><td>a</td><td>1/2</td><td>d1</td><td>2</td></tr><tr><td>1</td><td>A</td><td>a</td><td>2/9</td><td>d2</td><td>1</td></tr><tr><td>2</td><td>B</td><td>b</td><td>1</td><td>d1</td><td>8</td></tr><tr><td>2</td><td>B</td><td>b</td><td>1/2</td><td>d1</td><td>4</td></tr><tr><td>2</td><td>B</td><td>b</td><td>3/7</td><td>d1</td><td>6</td></tr><tr><td>2</td><td>B</td><td>b</td><td>1</td><td>d3</td><td>9</td></tr><tr><td>2</td><td>B</td><td>b</td><td>1/3</td><td>d3</td><td>2</td></tr><tr><td>2</td><td>B</td><td>b</td><td>2/3</td><td>d2</td><td>4</td></tr><tr><td>2</td><td>B</td><td>b</td><td>1/3</td><td>d2</td><td>3</td></tr><tr><td>2</td><td>B</td><td>b</td><td>1/3</td><td>d2</td><td>6</td></tr><tr><td>3</td><td>C</td><td>C</td><td>1</td><td>d2</td><td>5</td></tr><tr><td>3</td><td>C</td><td>C</td><td>1/3</td><td>d2</td><td>7</td></tr><tr><td>3</td><td>C</td><td>C</td><td>2/3</td><td>d3</td><td>7</td></tr><tr><td>3</td><td>C</td><td>C</td><td>1/5</td><td>d3</td><td>11</td></tr><tr><td>3</td><td>C</td><td>C</td><td>3/5</td><td>d1</td><td>9</td></tr><tr><td>3</td><td>C</td><td>C</td><td>1/2</td><td>d1</td><td>5</td></tr></tbody></table>

<!-- Media -->

Since the first round of block-scan yields two different documents, ${d}_{1}$ and ${d}_{2}$ ,the second-best document for each condition need to be fetched. After the second round, all ${d}_{3}$ ’s relevant elements for both content conditions are in memory. At this point,a random access for all $A$ elements in ${d}_{3}$ can be triggered,if it is cost-beneficial. This operation efficiently tests the query structure conditions for ${d}_{3}$ ,that is,whether the $B : b$ and $C : c$ elements are descendants of the same $A$ element. This is done by comparing the preorder and postorder of the respective element pairs. The result is that none of ${d}_{3}$ ’s element satisfies both structure conditions. Notice that the same test cannot be fully performed for document ${d}_{1}$ unless its $C$ tags are fetched first. If worstscore(d)is greater than the bottom score of the current top- $k$ set, $\mathop{\min }\limits_{k}$ ,then $d$ is inserted into the current top- $k$ set. On the other hand,if bestscore(d)is greater than ${mi}{n}_{k}$ ,then $d$ is inserted into a candidate queue and a probabilistic threshold test is periodically performed to examine whether $d$ still has a good chance to qualify for the top- $k$ documents or not.

由于第一轮块扫描产生了两个不同的文档，即${d}_{1}$和${d}_{2}$，因此需要获取每种条件下的次优文档。第二轮扫描后，${d}_{3}$中与两个内容条件相关的所有元素都已载入内存。此时，如果成本效益合适，可以触发对${d}_{3}$中所有$A$元素的随机访问。此操作可高效测试${d}_{3}$的查询结构条件，即$B : b$和$C : c$元素是否为同一个$A$元素的后代。这通过比较各个元素对的前序和后序来实现。结果是，${d}_{3}$的元素均不满足这两个结构条件。请注意，除非先获取文档${d}_{1}$的$C$标签，否则无法对该文档完全执行相同的测试。如果worstscore(d)大于当前前$k$集合的最低分数$\mathop{\min }\limits_{k}$，则将$d$插入到当前前$k$集合中。另一方面，如果bestscore(d)大于${mi}{n}_{k}$，则将$d$插入候选队列，并定期进行概率阈值测试，以检查$d$是否仍有很大机会符合前$k$文档的条件。

#### 7.2.The XRank System

#### 7.2. XRank系统

As illustrated by Example 7.1,top- $k$ queries in XML databases report the top- $k$ documents (or elements) based on their relevance scores to the given query conditions. A related issue is the notion of keyword proximity, which is the distance between keywords inside documents (elements). Proximity is rather complex to deal with in the hierarchical XML data models because it has to take into account the structure information of elements containing the required keywords. For instance, when keywords are located inside hierarchically distant elements, the proximity should be penalized even if the keywords are physically located near to each other inside the document text. The XRank system [Guo et al. 2003] addresses these issues. XRank considers the problem of producing ranked results for keyword queries over hyperlinked XML documents. The approach of XRank retains the simple keyword search query interface (similar to traditional HTML search engines), while exploiting XML tagged structure during query processing. The adopted scoring function favors more specific results to general results with respect to keyword matching. For instance, the occurrence of all keywords in the same element is favored to their occurrence in distinct elements. The score of an element with respect to one keyword is derived based on its popularity (similar to Google's PageRank), and its specificity with respect to the keyword. The element score with respect to all query keywords is the sum of its scores with respect to different keywords, weighted by a measure of keyword proximity.

如示例7.1所示，XML数据库中的前$k$查询会根据文档（或元素）与给定查询条件的相关性得分，报告前$k$个文档（或元素）。一个相关问题是关键词邻近度的概念，即文档（元素）内关键词之间的距离。在分层XML数据模型中处理邻近度相当复杂，因为必须考虑包含所需关键词的元素的结构信息。例如，当关键词位于层次上距离较远的元素中时，即使这些关键词在文档文本中实际位置相近，也应对邻近度进行惩罚。XRank系统[Guo等人，2003年]解决了这些问题。XRank考虑了对超链接XML文档的关键词查询生成排序结果的问题。XRank的方法保留了简单的关键词搜索查询界面（类似于传统的HTML搜索引擎），同时在查询处理过程中利用XML标签结构。所采用的评分函数在关键词匹配方面更倾向于更具体的结果而非通用结果。例如，所有关键词出现在同一元素中比出现在不同元素中更受青睐。元素相对于一个关键词的得分是基于其受欢迎程度（类似于谷歌的PageRank）及其相对于该关键词的特异性得出的。元素相对于所有查询关键词的得分是其相对于不同关键词的得分之和，并根据关键词邻近度的度量进行加权。

#### 7.3.XML Structure Scoring

#### 7.3. XML结构评分

Scoring XML elements based on structure conditions is addressed by Amer-Yahia et al. [2005] with a focus on twig queries. A twig query is a rooted tree with string-labeled nodes and two types of edges, / (a child edge) and // (a descendant edge). Three different heuristics of structure relaxation were considered, while scoring XML elements:

Amer - Yahia等人[2005年]针对基于结构条件对XML元素进行评分的问题进行了研究，重点关注树枝查询。树枝查询是一种带有字符串标签节点和两种类型边的有根树，这两种边分别是/（子边）和//（后代边）。在对XML元素进行评分时，考虑了三种不同的结构松弛启发式方法：

- Edge generalization. A / edge in query $Q$ is replaced by a // edge to obtain an approximate query $Q$ .

- 边泛化。将查询$Q$中的/边替换为//边，以获得近似查询$Q$。

-Subtree promotion. A pattern $a\left\lbrack  {b\left\lbrack  {Q1}\right\rbrack  //{Q2}}\right\rbrack$ is replaced by $a\left\lbrack  {b\left\lbrack  {Q1}\right\rbrack  \text{and}.//{Q2}}\right\rbrack$

- 子树提升。将模式$a\left\lbrack  {b\left\lbrack  {Q1}\right\rbrack  //{Q2}}\right\rbrack$替换为$a\left\lbrack  {b\left\lbrack  {Q1}\right\rbrack  \text{and}.//{Q2}}\right\rbrack$

-Leaf node deletion. A pattern $a\left\lbrack  {{Q1}\text{and}.//b}\right\rbrack$ where $a$ is the root of the query tree and $b$ is a leaf node is replaced by $a\left\lbrack  {Q1}\right\rbrack$ .

- 叶节点删除。将模式$a\left\lbrack  {{Q1}\text{and}.//b}\right\rbrack$（其中$a$是查询树的根节点，$b$是叶节点）替换为$a\left\lbrack  {Q1}\right\rbrack$。

These relaxations lead to approximate results that do not necessarily satisfy the original query. Based on these approximations, a directed acyclic graph (DAG), called a relaxation DAG, is constructed. The relaxation DAG contains one source node, which is the original query, and one sink node which is the most general relaxed query. An example of relaxation DAG is shown in Figure 23. In this example, the most constraining query is shown in the top node, while its children represent more relaxed queries that are obtained by replacing / constraints by // constraints. Further relaxation is obtained by incrementally removing constraints. The most relaxed query appears at the bottom of the DAG, which only contains the root element.

这些松弛操作会产生不一定满足原始查询的近似结果。基于这些近似结果，会构建一个有向无环图（DAG），称为松弛有向无环图。松弛有向无环图包含一个源节点（即原始查询）和一个汇节点（即最通用的松弛查询）。图23展示了一个松弛有向无环图的示例。在这个示例中，最具约束性的查询显示在顶部节点，而其子节点表示通过将 / 约束替换为 // 约束而得到的更宽松的查询。通过逐步移除约束可以实现进一步的松弛。最宽松的查询出现在有向无环图的底部，它只包含根元素。

Elements' scores depend on how close the elements are to the given query. The wellknown tf.idf measure [Salton and McGill 1983] is adopted to compute elements’ scores. The inverse document frequency (idf),with respect to a query $Q$ ,quantifies how many of the elements that satisfy the sink node in the relaxation DAG additionally satisfy $Q$ . The term frequency (tf) score quantifies the number of distinct ways in which an element matches a query and its relaxations. The ${tf}$ and ${idf}$ scores are used to compute the overall elements’ scores. A typical top- $k$ algorithm is used to find the top- $k$ elements based on these scores.

元素的得分取决于元素与给定查询的接近程度。采用著名的tf.idf度量方法[Salton和McGill 1983]来计算元素的得分。相对于查询 $Q$ 的逆文档频率（idf）量化了在松弛有向无环图中满足汇节点的元素中有多少还额外满足 $Q$。词频（tf）得分量化了元素匹配查询及其松弛形式的不同方式的数量。${tf}$ 和 ${idf}$ 得分用于计算元素的总体得分。使用典型的前 $k$ 算法根据这些得分找出前 $k$ 个元素。

<!-- Media -->

<!-- figureText: DVD Air Conditioner Car DVD Air Player Conditioner Car Air Conditioner Car Air Conditioner Player Car DVD Air Player Conditioner DVD Player Car DVD Player Car -->

<img src="https://cdn.noedgeai.com/0195c915-c519-7d98-b2ff-761eddc2c676_48.jpg?x=543&y=284&w=642&h=882&r=0"/>

Fig. 23. A query relaxations DAG.

图23. 查询松弛有向无环图。

<!-- Media -->

## 8. VOTING SYSTEMS: THEORETICAL BACKGROUND

## 8. 投票系统：理论背景

In this section,we give a theoretical background for ranking and top- $k$ processing problems in voting theory. The problem of combining different rankings of a list of candidates has deep roots in social and political sciences. Different voting systems have been designed to allow voters to express their preferences regarding a number of candidates and select the winning candidate(s) in a way that satisfies the majority of voters. Voting systems are based on one or a combination of rank aggregation techniques. The underlying rank aggregation methods specify rules that determine how votes are counted and how candidates are ordered. The study of formal voting systems is called voting theory.

在本节中，我们为投票理论中的排序和前 $k$ 处理问题提供理论背景。组合候选人列表的不同排名问题在社会和政治科学中有着深厚的根源。已经设计了不同的投票系统，以允许选民表达他们对多个候选人的偏好，并以满足大多数选民的方式选出获胜候选人。投票系统基于一种或多种排名聚合技术的组合。底层的排名聚合方法指定了确定如何计票以及如何对候选人进行排序的规则。对正式投票系统的研究称为投票理论。

Choosing a voting system to rank a list of candidates, based on the votes received from several parties, is usually not straightforward. It has been shown that no preference aggregation method could always yield a fair result when combining votes for three or more candidates [Arrow 1951]. The problem with choosing a fair voting system is mainly attributed to the required, but sometimes inevitably violated, properties that a voting system should exhibit [Dwork et al. 2001; Cranor 1996], summarized in the following:

根据从多个方面收到的选票选择一个投票系统来对候选人列表进行排名通常并非易事。已经证明，当对三个或更多候选人的选票进行组合时，没有一种偏好聚合方法能够始终产生公平的结果[Arrow 1951]。选择公平投票系统的问题主要归因于投票系统应具备但有时不可避免会被违反的属性[Dwork等人2001；Cranor 1996]，总结如下：

-Monotonicity. A voting system is called monotonic when raising the valuation for a winning candidate allows it to remain a winner, while lowering the valuation for the candidate lowers its rank in the candidate list. All voting systems that eliminate candidates prior to selecting a winner violate monotonicity.

-单调性。当提高获胜候选人的估值能使其保持获胜，而降低该候选人的估值会降低其在候选人列表中的排名时，这样的投票系统被称为具有单调性。所有在选出获胜者之前淘汰候选人的投票系统都违反了单调性。

-Transitivity. A voting system is transitive if,whenever the rank of $x$ is over the rank of $y$ and $y$ is over $z$ ,then it should always be the case that $x$ is ranked over $z$ .

-传递性。如果每当 $x$ 的排名高于 $y$ 且 $y$ 的排名高于 $z$ 时，$x$ 的排名总是高于 $z$，则该投票系统具有传递性。

-Neutrality. A voting system is neutral if it does not favor any specific candidates. Systems that have tie-breaking rules, other than random selection, violate neutrality.

-中立性。如果投票系统不偏袒任何特定候选人，则称其具有中立性。除随机选择外具有打破平局规则的系统违反了中立性。

-Pareto-optimality. A voting system is pareto-optimal if, when every voter prefers candidate $x$ to candidate $y$ ,candidate $y$ is not selected.

-帕累托最优性。如果当每个选民都更喜欢候选人 $x$ 而不是候选人 $y$ 时，候选人 $y$ 不会被选中，则该投票系统具有帕累托最优性。

-Consistency. A voting system is consistent if dividing the candidate list into two arbitrary parts, and conducting separate voting for each part, results in the same candidate being selected as if the entire candidate list was subjected to voting. If a voting system is consistent, then voting could be conducted in voting domains.

-一致性。如果将候选人列表任意分成两部分，并对每部分分别进行投票，选出的候选人与对整个候选人列表进行投票选出的候选人相同，则该投票系统具有一致性。如果一个投票系统具有一致性，则可以在投票域中进行投票。

A detailed taxonomy of various voting strategies from the perspectives of social and political sciences can be found in Cranor [1996]. In the following sections, we introduce two of the most widely adopted voting systems. The first system is based on the majority of votes, while the second is based on candidates' relative positions.

从社会和政治科学的角度对各种投票策略进行的详细分类可以在Cranor [1996] 中找到。在以下各节中，我们将介绍两种应用最广泛的投票系统。第一种系统基于多数选票，而第二种系统基于候选人的相对位置。

### 8.1. Majority-Based Procedures

### 8.1. 基于多数的程序

Ranking by majority of votes is probably the simplest form of aggregating ranks from different sources. The basic principle is that a group of more than half of the voters should be able to get the outcome they prefer. Ranking by majority for two candidates possesses all desirable properties in voting systems. However, when three or more candidates are under consideration, there may not be a single candidate that is preferred by the majority. In this case, voters compare each pair of candidates in different rounds and another procedure must be used to ensure that the selected candidate is preferred by the majority.

按多数选票进行排名可能是聚合不同来源排名的最简单形式。基本原则是超过半数的选民群体应该能够获得他们偏好的结果。对两名候选人按多数进行排名具备投票系统中所有理想的属性。然而，当考虑三名或更多候选人时，可能没有一个候选人能得到多数人的偏好。在这种情况下，选民会在不同轮次中比较每对候选人，并且必须使用另一种程序来确保选出的候选人得到多数人的偏好。

8.1.1. Condorcet Criterion. Ranking candidates through pairwise voting was discovered over 200 years ago by the mathematician and social scientist M. Condorcet in 1785. The criterion was based on a majority rule, but instead of voting for only one candidate, candidates are ranked in order of preference. This can be seen as series of pairwise comparisons between the members of each candidate pair. The end result is a winner who is favored by the majority of voters. Ballots are counted by considering all possible sets of two-candidate elections from all available candidates. That is, each candidate is considered against each and every other candidate. A candidate wins against an opponent in a single ballot if the candidate is ranked higher than its opponent. Whoever has the most votes based on all these one-on-one elections wins.

8.1.1. 孔多塞准则（Condorcet Criterion）。通过两两投票对候选人进行排名的方法是200多年前，即1785年由数学家兼社会科学家M. 孔多塞（M. Condorcet）发现的。该准则基于多数决规则，但选民不是只对一名候选人投票，而是按照偏好对候选人进行排序。这可以看作是对每对候选人进行的一系列两两比较。最终结果是选出一位受到大多数选民青睐的获胜者。计票时会考虑从所有候选人中可能产生的两两竞选组合。也就是说，会将每位候选人和其他所有候选人进行比较。在一张选票中，如果一名候选人的排名高于对手，那么该候选人就在此次两两比较中获胜。在所有这些一对一的竞选比较中获得最多票数的候选人将赢得选举。

If a candidate is preferred over all other candidates, that candidate is the Condorcet winner. Condorcet winners may not always exist, due to a fundamental paradox: it is possible for the voters to prefer $A$ over $B,B$ over $C$ ,and $C$ over $A$ simultaneously. This is called majority rule cycle, and it must be resolved by some other mechanism. A voting system that is able to produce the Condorcet winner is said to attain the Condorcet property [Young and Levenglick 1978].

如果一名候选人比其他所有候选人都更受青睐，那么该候选人就是孔多塞获胜者（Condorcet winner）。由于一个基本悖论，孔多塞获胜者并非总是存在：选民有可能同时更偏好$A$胜过$B,B$，偏好$B,B$胜过$C$，又偏好$C$胜过$A$。这被称为多数决循环（majority rule cycle），必须通过其他机制来解决。能够选出孔多塞获胜者的投票系统被认为具有孔多塞属性 [扬和莱文格利克（Young and Levenglick），1978年]。

Example 8.1 (Condorcet Criterion ${}^{2}$ ). This example illustrates the basic counting method of Condorcet criterion. Consider an election for the candidates $A,B,C$ ,and $D$ . The election ballot can be represented as a matrix,where the row is the runner under consideration, and the column is the opponent. The cell at (runner, opponent) contains 1 if runner is preferred, and 0 if not. Cells marked "-" are logically zero as a candidate can not be defeated by himself. This binary matrix is inversely symmetric: (runner,opponent) is $\neg$ (opponent,runner); however,the sum of all ballot matrices is not symmetric. When the sum matrix is found, the contest between each candidate pair is considered. The number of votes for runner over opponent (runner, opponent) is compared to the number of votes for opponent over runner (opponent, runner). The one-on-one winner has the most votes. If one candidate wins against all other candidates, that candidate wins the election. The sum matrix is the primary piece of data used to resolve majority rule cycles.

示例8.1（孔多塞准则${}^{2}$）。本示例说明了孔多塞准则的基本计票方法。假设有一场候选人分别为$A,B,C$和$D$的选举。选举选票可以用一个矩阵来表示，其中行代表被考虑的参选者，列代表对手。如果参选者更受青睐，那么（参选者，对手）对应的单元格值为1，否则为0。标记为“ - ”的单元格在逻辑上为0，因为候选人不可能输给自己。这个二进制矩阵是反对称的：（参选者，对手）的值等于$\neg$（对手，参选者）的值；然而，所有选票矩阵的总和并不对称。得到总和矩阵后，会对每对候选人之间的竞争情况进行考量。会比较参选者胜过对手（参选者，对手）的票数与对手胜过参选者（对手，参选者）的票数。在一对一比较中获得最多票数的一方获胜。如果一名候选人在与其他所有候选人的比较中都获胜，那么该候选人赢得选举。总和矩阵是解决多数决循环问题的主要数据依据。

---

<!-- Footnote -->

${}^{2}$ Available online at the Election Methods Web site http://www.electionmethods.org.

${}^{2}$ 可在选举方法网站http://www.electionmethods.org上在线获取。

<!-- Footnote -->

---

<!-- Media -->

<table><tr><td/><td>A</td><td>B</td><td>C</td><td>D</td></tr><tr><td>A</td><td>-</td><td>0</td><td>0</td><td>0</td></tr><tr><td>B</td><td>1</td><td>-</td><td>1</td><td>1</td></tr><tr><td>C</td><td>1</td><td>0</td><td>-</td><td>0</td></tr><tr><td>D</td><td>1</td><td>0</td><td>1</td><td>-</td></tr></table>

<table><tr><td/><td>A</td><td>B</td><td>C</td><td>D</td></tr><tr><td>A</td><td>-</td><td>0</td><td>0</td><td>0</td></tr><tr><td>B</td><td>1</td><td>-</td><td>1</td><td>1</td></tr><tr><td>C</td><td>1</td><td>0</td><td>-</td><td>0</td></tr><tr><td>D</td><td>1</td><td>0</td><td>1</td><td>-</td></tr></table>

Fig. 24. Voter 1 ballot.

图24. 选民1的选票。

<table><tr><td/><td>A</td><td>B</td><td>C</td><td>D</td></tr><tr><td>A</td><td>-</td><td>0</td><td>0</td><td>0</td></tr><tr><td>B</td><td>1</td><td>-</td><td>1</td><td>0</td></tr><tr><td>C</td><td>0</td><td>0</td><td>-</td><td>0</td></tr><tr><td>D</td><td>1</td><td>1</td><td>1</td><td>-</td></tr></table>

<table><tr><td/><td>A</td><td>B</td><td>C</td><td>D</td></tr><tr><td>A</td><td>-</td><td>0</td><td>0</td><td>0</td></tr><tr><td>B</td><td>1</td><td>-</td><td>1</td><td>0</td></tr><tr><td>C</td><td>0</td><td>0</td><td>-</td><td>0</td></tr><tr><td>D</td><td>1</td><td>1</td><td>1</td><td>-</td></tr></table>

Fig. 25. Voter 2 ballot.

图25. 选民2的选票。

<table><tr><td/><td>A</td><td>B</td><td>C</td><td>D</td></tr><tr><td>A</td><td>-</td><td>0</td><td>0</td><td>0</td></tr><tr><td>B</td><td>2</td><td>-</td><td>2</td><td>1</td></tr><tr><td>C</td><td>1</td><td>0</td><td>-</td><td>0</td></tr><tr><td>D</td><td>2</td><td>1</td><td>2</td><td>-</td></tr></table>

<table><tr><td/><td>A</td><td>B</td><td>C</td><td>D</td></tr><tr><td>A</td><td>-</td><td>0</td><td>0</td><td>0</td></tr><tr><td>B</td><td>2</td><td>-</td><td>2</td><td>1</td></tr><tr><td>C</td><td>1</td><td>0</td><td>-</td><td>0</td></tr><tr><td>D</td><td>2</td><td>1</td><td>2</td><td>-</td></tr></table>

Fig. 26. Sum ballot.

图26. 汇总选票。

<!-- Media -->

Figures 24 and 25 show two sample voter ballots for four candidates $A,B,C$ ,and $D$ ,while Figure 26 shows the sum ballot. The sum ballot shows that $B$ defeats $A$ with score 2-0, $B$ defeats $C$ with score 2-0,and $B$ and $D$ have equal scores 1-1. In this case, no Condorcet winner could be reached.

图24和图25展示了针对四位候选人$A,B,C$和$D$的两张选民选票样本，而图26展示了汇总选票。汇总选票显示，$B$以2 - 0的比分击败$A$，$B$以2 - 0的比分击败$C$，并且$B$和$D$的比分相同，均为1 - 1。在这种情况下，无法选出孔多塞胜者（Condorcet winner）。

An extension of the Condorect criterion has been introduced by Truchon [1998]: if a candidate is consistently ranked ahead of another candidate by an absolute majority of voters, it should be ahead in the final ranking. The term consistently refers to the absence of cycles in the majority relation involving these two candidates.

特鲁雄（Truchon）[1998年]引入了孔多塞准则（Condorect criterion）的一个扩展：如果绝对多数选民始终将一位候选人排在另一位候选人之前，那么该候选人在最终排名中也应靠前。“始终”一词指的是在涉及这两位候选人的多数关系中不存在循环。

It has been shown that the extended Condorect criterion has excellent spam filtering properties when aggregating the rankings generated by different search engines [Dwork et al. 2001]. Specifically, if a spam page is ranked high by fewer than half of the search engines, then the majority of search engines prefer a good page to this spam page. In this case, spam pages will occupy a late rank based on any rank aggregation method that satisfies the extended Condorect criterion.

研究表明，在聚合不同搜索引擎生成的排名时，扩展的孔多塞准则具有出色的垃圾邮件过滤特性[德沃克（Dwork）等人，2001年]。具体而言，如果不到一半的搜索引擎将某个垃圾页面排在较高位置，那么大多数搜索引擎会更倾向于一个优质页面而非这个垃圾页面。在这种情况下，基于任何满足扩展孔多塞准则的排名聚合方法，垃圾页面将占据靠后的排名。

8.1.2. Dodgson Ranking. In 1876, C. Dodgson devised a nonbinary procedure to overcome the problem of voting cycles [Black 1958]. He suggested that one should always choose the candidate that is "closest" to being a Condorcet winner. His concept of distance was based on the number of inversions of pairs in the individual preferences. A natural extension of Dodgson's method is to rank all candidates with respect to the minimum number of inversions necessary to make each of them a Condorcet winner.

8.1.2. 道奇森排名（Dodgson Ranking）。1876年，C. 道奇森（C. Dodgson）设计了一种非二元程序来解决投票循环问题[布莱克（Black），1958年]。他建议应始终选择最“接近”成为孔多塞胜者的候选人。他的距离概念基于个体偏好中配对反转的数量。道奇森方法的一个自然扩展是根据使每个候选人成为孔多塞胜者所需的最小反转次数对所有候选人进行排名。

8.1.3. Copeland Rule. Copeland's [1951] method is a Condorcet method in which the winner is determined by finding the candidate with the most pairwise victories. This rule selects the candidate with the largest Copeland index, which is the number of times a candidate beats other candidates minus the number of times that candidate loses to other candidates when the candidates are considered in pairwise comparisons. However, this method often leads to ties when there is no Condorcet winner. For example, if there is a three-candidate majority rule cycle, each candidate will have exactly one loss, and there will be an unresolved tie among the three. The Copeland rule is Pareto-optimal.

8.1.3. 科普兰规则（Copeland Rule）。科普兰（Copeland）[1951年]的方法是一种孔多塞方法，其中胜者是通过找出获得最多两两对决胜利的候选人来确定的。该规则选择科普兰指数最大的候选人，科普兰指数是指在两两比较候选人时，一位候选人击败其他候选人的次数减去该候选人输给其他候选人的次数。然而，当不存在孔多塞胜者时，这种方法往往会导致平局。例如，如果存在一个三位候选人的多数规则循环，每位候选人将恰好输掉一场比赛，并且这三位候选人之间将出现无法解决的平局。科普兰规则是帕累托最优的（Pareto - optimal）。

The main underlying idea of both Dodgson's and Copeland's methods, namely, the closer a candidate is to be a Condorcet winner the better, is actually identical. However, some anomalies indicate that they can result in different outcomes. Klamler [2003] gave a detailed comparison of the Dodgson ranking and the Copeland rule showing that the Copeland winner can occur at any position in the Dodgson ranking. For some settings of individual preferences of candidates, the Dodgson ranking and the Copeland ranking are exactly the opposite.

道奇森方法和科普兰方法的主要潜在思想实际上是相同的，即候选人越接近成为孔多塞胜者越好。然而，一些异常情况表明，它们可能会导致不同的结果。克拉姆勒（Klamler）[2003年]对道奇森排名和科普兰规则进行了详细比较，结果表明科普兰胜者在道奇森排名中可以处于任何位置。对于候选人个体偏好的某些设定，道奇森排名和科普兰排名恰好相反。

### 8.2. Positional Procedures

### 8.2. 位置程序

Unlike majority-based methods, which are usually based on information from binary comparisons, positional methods take into account the information about voters' preference orderings. These methods do not necessarily select the Condorcet winner when one exists. In addition, for some settings of voters' preferences, each method produces a different outcome. The three positional methods discussed here are all monotonic and Pareto-optimal.

与通常基于二元比较信息的多数方法不同，位置方法考虑了选民偏好排序的信息。当存在孔多塞胜者时，这些方法不一定会选择该胜者。此外，对于选民偏好的某些设定，每种方法会产生不同的结果。这里讨论的三种位置方法都是单调的且是帕累托最优的。

8.2.1. Approval Voting. In this method voters give a single vote to each candidate they approve [Brams and Fishburn 1983]. This is equivalent to giving each candidate a score of +1 if the voter approves the candidate, and a score of 0 if the voter does not approve the candidate. Votes are tallied, and the candidates are ranked based on the total number of votes they receive.

8.2.1. 认可投票（Approval Voting）。在这种方法中，选民为他们认可的每位候选人投一票[布拉姆斯（Brams）和菲什伯恩（Fishburn），1983年]。这相当于如果选民认可某位候选人，则给该候选人打+1分，如果不认可，则打0分。对选票进行统计，并根据候选人获得的总票数对他们进行排名。

8.2.2. Plurality Voting. This method is also referred to as first-past-the-post [Nurmi 1987]. The method is based on the plurality criterion, which can be simply described as follows: if the number of ballots ranking $A$ as the first preference is greater than the number of ballots on which another candidate $B$ is the first preference,then $A$ ’s probability of winning must be no less than $B$ ’s. In other words,plurality voting mainly uses information about each voter’s most preferred candidate. When $n$ candidates are to be selected,voters vote for their $n$ most preferred candidates. Because this procedure only takes into account the first preference,or first $n$ preferences,of each voter,it often fails to select the Condorcet winner. As a result, it sometimes produces results which appear illogical when three or more candidates compete. It sometimes even selects the Condorcet loser, the candidate that is defeated by all others in pairwise comparisons.

8.2.2. 多数投票制。这种方法也被称为简单多数制（first-past-the-post）[努尔米（Nurmi）1987年]。该方法基于多数标准，可简单描述如下：如果将$A$列为首选的选票数量多于将另一位候选人$B$列为首选的选票数量，那么$A$的获胜概率必须不低于$B$的获胜概率。换句话说，多数投票制主要利用每位选民最偏好候选人的信息。当要选出$n$位候选人时，选民为他们最偏好的$n$位候选人投票。由于这一程序仅考虑每位选民的第一偏好，或前$n$个偏好，因此它往往无法选出孔多塞胜者（Condorcet winner）。结果，当有三位或更多候选人竞争时，它有时会产生看似不合逻辑的结果。它有时甚至会选出孔多塞败者（Condorcet loser），即在两两比较中被其他所有候选人击败的候选人。

Example 8.2 (Plurality Voting). Imagine an election held to select the winner among four candidates $A,B,C$ ,and $D$ . Assume voters have cast their ballots strictly as follows:

示例8.2（多数投票制）。假设有一场选举，要从四位候选人$A,B,C$和$D$中选出获胜者。假设选民严格按照以下方式投票：

<!-- Media -->

<table><tr><td>${42}\%$ of voters</td><td>26% of voters</td><td>15% of voters</td><td>17% of voters</td></tr><tr><td>1. $A$</td><td>1. $B$</td><td>1. $C$</td><td>1. $D$</td></tr><tr><td>2. $B$</td><td>2. $C$</td><td>2. $D$</td><td>2. $C$</td></tr><tr><td>3. $C$</td><td>3. $D$</td><td>3. $B$</td><td>3. $B$</td></tr><tr><td>4. $D$</td><td>4. $A$</td><td>4. $A$</td><td>4. $A$</td></tr></table>

<table><tbody><tr><td>${42}\%$的选民</td><td>26%的选民</td><td>15%的选民</td><td>17%的选民</td></tr><tr><td>1. $A$</td><td>1. $B$</td><td>1. $C$</td><td>1. $D$</td></tr><tr><td>2. $B$</td><td>2. $C$</td><td>2. $D$</td><td>2. $C$</td></tr><tr><td>3. $C$</td><td>3. $D$</td><td>3. $B$</td><td>3. $B$</td></tr><tr><td>4. $D$</td><td>4. $A$</td><td>4. $A$</td><td>4. $A$</td></tr></tbody></table>

<!-- Media -->

Candidate $A$ is the winner based on the plurality of votes,even though the majority of voters (58%) did not select $A$ as the winner.

候选人 $A$ 基于最多票数成为获胜者，尽管大多数选民（58%）并未选择 $A$ 为获胜者。

8.2.3. Borda Count. Borda count voting , proposed in 1770 by the French mathematician Jean-Charles de Borda, is a procedure in which each voter forms a preference ranking for all candidates. The Borda count is a voting system used for single-winner elections in which each voter orders the candidates. The Borda count is classified as a positional voting system because each rank on the ballot is worth a certain number of points. Given $n$ candidates,each voter orders all the candidates such that the first-place candidate on a ballot receives $n - 1$ points,the second-place candidate receives $n - 2$ points,and in general the candidate in the $i$ th place receives $n - i$ points. The candidate ranked last on the ballot therefore receives zero points. The points are added up across all the ballots, and the candidate with the most points is the winner.

8.2.3. 博尔达计数法（Borda Count）。博尔达计数投票法由法国数学家让 - 夏尔·德·博尔达（Jean - Charles de Borda）于1770年提出，在该程序中，每位选民要对所有候选人形成偏好排序。博尔达计数法是一种用于单获胜者选举的投票系统，每位选民对候选人进行排序。博尔达计数法被归类为一种位次投票系统，因为选票上的每个位次对应一定数量的分数。假设有 $n$ 位候选人，每位选民对所有候选人进行排序，使得选票上排名第一的候选人获得 $n - 1$ 分，排名第二的候选人获得 $n - 2$ 分，一般来说，排名第 $i$ 位的候选人获得 $n - i$ 分。因此，选票上排名最后的候选人得零分。将所有选票上的分数相加，得分最多的候选人即为获胜者。

This system is similar to "scoring" methods which incorporate all voter preference information into the vote aggregation. However, as with the other positional voting methods, this does not always produce a logical result. In fact, the result of a Borda count is often a function of the number of candidates considered. However, the Borda count will never choose the Condorcet loser. Borda's method is the basis for most current rank aggregation algorithms.

该系统类似于将所有选民偏好信息纳入投票汇总的“计分”方法。然而，与其他位次投票方法一样，这并不总是能产生合理的结果。事实上，博尔达计数法的结果通常取决于所考虑的候选人数量。不过，博尔达计数法永远不会选择孔多塞失败者（Condorcet loser）。博尔达方法是当前大多数排名汇总算法的基础。

Example 8.3 (Borda Count). In Example 8.2, $B$ is the Borda winner in this election, as it has the most points (see the table below). $B$ also happens to be the Condorcet winner in this case. While the Borda count does not always select the Condorcet winner as the Borda Count winner, it always ranks the Condorcet winner above the Condorcet loser. No other positional method can guarantee such a relationship.

例8.3（博尔达计数法）。在例8.2中，$B$ 是此次选举中的博尔达获胜者，因为它的得分最高（见下表）。在这种情况下，$B$ 恰好也是孔多塞获胜者（Condorcet winner）。虽然博尔达计数法并不总是将孔多塞获胜者选为博尔达计数获胜者，但它总是将孔多塞获胜者的排名排在孔多塞失败者之上。没有其他位次方法能保证这种关系。

<!-- Media -->

<table><tr><td>Candidate</td><td>First</td><td>Second</td><td>Third</td><td>Fourth</td><td>Total Points</td></tr><tr><td>$A$</td><td>42</td><td>0</td><td>0</td><td>58</td><td>126</td></tr><tr><td>$B$</td><td>26</td><td>42</td><td>32</td><td>0</td><td>194</td></tr><tr><td>$C$</td><td>15</td><td>43</td><td>42</td><td>0</td><td>173</td></tr><tr><td>$D$</td><td>17</td><td>15</td><td>26</td><td>42</td><td>107</td></tr></table>

<table><tbody><tr><td>候选人</td><td>第一</td><td>第二</td><td>第三</td><td>第四</td><td>总积分</td></tr><tr><td>$A$</td><td>42</td><td>0</td><td>0</td><td>58</td><td>126</td></tr><tr><td>$B$</td><td>26</td><td>42</td><td>32</td><td>0</td><td>194</td></tr><tr><td>$C$</td><td>15</td><td>43</td><td>42</td><td>0</td><td>173</td></tr><tr><td>$D$</td><td>17</td><td>15</td><td>26</td><td>42</td><td>107</td></tr></tbody></table>

<!-- Media -->

The Borda count is vulnerable to compromising. That is, voters can help avoid the election of some candidate by raising the position of a more-preferred candidate on their ballot. The Borda count method can be extended to include tie-breaking methods. Moreover, ballots that do not rank all the candidates can be allowed by giving unranked candidates zero points,or alternatively by assigning the points up to $k$ ,where $k$ is the number of candidates ranked on a ballot. For example,a ballot that ranks candidate $A$ first and candidate $B$ second,leaving everyone else unranked,would give 2 points to $A,1$ point to $B$ ,and zero points to all other candidates.

博尔达计数法（Borda count）容易受到妥协策略的影响。也就是说，选民可以通过在选票上提高更偏好候选人的排名，来帮助避免某些候选人当选。博尔达计数法可以扩展到包含打破平局的方法。此外，对于没有对所有候选人进行排名的选票，可以给未排名的候选人计零分，或者将分数分配到$k$ ，其中$k$ 是选票上排名的候选人数。例如，一张选票将候选人$A$ 排在第一位，候选人$B$ 排在第二位，其余候选人未排名，那么将给$A,1$ 2分，给$B$ 1分，给其他所有候选人0分。

Nanson's method, due to Edward John Nanson (1850-1936), is a procedure for finding the Condorcet winner of a Borda count. In this method, the Borda count is computed for each candidate and the candidate with the lowest Borda count is eliminated, and a new election is held using the Borda count until a single winner emerges. If there is a candidate that is the Condorcet winner, this method chooses that candidate. If there is no Condorcet winner, then some candidate, not necessarily the same as the Borda count winner, is chosen.

南森方法（Nanson's method）由爱德华·约翰·南森（Edward John Nanson，1850 - 1936）提出，是一种用于找出博尔达计数法中的孔多塞胜者（Condorcet winner）的程序。在这种方法中，为每个候选人计算博尔达计数，淘汰博尔达计数最低的候选人，然后使用博尔达计数法举行新的选举，直到选出唯一的胜者。如果存在孔多塞胜者，该方法将选择该候选人。如果不存在孔多塞胜者，则会选择某个候选人，但不一定是博尔达计数法的胜者。

### 8.3. Measuring Distance Between Rankings

### 8.3. 测量排名之间的距离

An alternative procedure to aggregate the rankings obtained from different sources is using a distance measure to quantify the disagreements among different rankings. An optimal rank aggregation method is the one that induces an overall ranking with minimum distance to the different rankings obtained from different sources. Diaconis [1998] and Fagin et al. $\left\lbrack  {{2003},{2004}}\right\rbrack$ described different distance measures. We describe in the following two widely adopted measures.

另一种聚合来自不同来源的排名的方法是使用距离度量来量化不同排名之间的差异。最优的排名聚合方法是诱导出一个与来自不同来源的不同排名的总距离最小的整体排名。迪亚科尼斯（Diaconis）[1998]和法金（Fagin）等人$\left\lbrack  {{2003},{2004}}\right\rbrack$描述了不同的距离度量方法。我们在下面描述两种广泛采用的度量方法。

8.3.1. Footrule Distance. The Footrule, also called the Spearman, distance is an absolute distance between two ranked vectors. Given two ranked vectors $\alpha$ and $\beta$ ,for a list of $n$ candidates,the Footrule distance is defined as

8.3.1. 英尺规则距离（Footrule Distance）。英尺规则距离，也称为斯皮尔曼（Spearman）距离，是两个排名向量之间的绝对距离。对于包含$n$ 个候选人的列表，给定两个排名向量$\alpha$ 和$\beta$ ，英尺规则距离定义为

$$
F\left( {\alpha ,\beta }\right)  = \mathop{\sum }\limits_{{i = 1}}^{n}\left| {\alpha \left( i\right)  - \beta \left( i\right) }\right| , \tag{5}
$$

where $\alpha \left( i\right)$ and $\beta \left( i\right)$ denote the position of candidate $i$ in $\alpha$ and $\beta$ ,respectively. The maximum value of $F\left( {\alpha ,\beta }\right)$ is $\frac{{n}^{2}}{2}$ when $n$ is even,and $\frac{\left( {n + 1}\right) \left( {n - 1}\right) }{2}$ when $n$ is odd.

其中$\alpha \left( i\right)$ 和$\beta \left( i\right)$ 分别表示候选人$i$ 在$\alpha$ 和$\beta$ 中的位置。当$n$ 为偶数时，$F\left( {\alpha ,\beta }\right)$ 的最大值为$\frac{{n}^{2}}{2}$ ；当$n$ 为奇数时，最大值为$\frac{\left( {n + 1}\right) \left( {n - 1}\right) }{2}$ 。

For example,consider a list of three candidates, $\{ A,B,C\}$ ; if two voters, $\alpha$ and $\beta$ ,order candidates as $\alpha  = \{ A,B,C\}$ and $\beta  = \{ C,B,A\}$ ,then the Footrule distance between $\alpha$ and $\beta$ is $\left| {1 - 3}\right|  + \left| {2 - 2}\right|  + \left| {3 - 1}\right|  = 4$ . The computed distance is the maximum possible distance for lists of three candidates, since one vector ranks the candidates in the reverse order of the other. The Footrule distance can be computed in linear time.

例如，考虑一个包含三个候选人的列表$\{ A,B,C\}$ ；如果两个选民$\alpha$ 和$\beta$ 对候选人的排序分别为$\alpha  = \{ A,B,C\}$ 和$\beta  = \{ C,B,A\}$ ，那么$\alpha$ 和$\beta$ 之间的英尺规则距离为$\left| {1 - 3}\right|  + \left| {2 - 2}\right|  + \left| {3 - 1}\right|  = 4$ 。计算得到的距离是三个候选人列表可能的最大距离，因为一个向量对候选人的排名顺序与另一个向量相反。英尺规则距离可以在线性时间内计算。

8.3.2. Kendall Tau Distance. The Kendall tau distance measures the amount of disagreement between two rankings by counting the number of pairwise disagreements [Kendall 1945]. Consider two rankings $\alpha$ and $\beta$ ,of a list of $n$ candidates; the Kendall tau distance is defined as

8.3.2. 肯德尔 tau 距离（Kendall Tau Distance）。肯德尔 tau 距离通过计算成对差异的数量来衡量两个排名之间的差异程度[肯德尔（Kendall）1945]。考虑对包含$n$ 个候选人的列表的两个排名$\alpha$ 和$\beta$ ；肯德尔 tau 距离定义为

$$
K\left( {\alpha ,\beta }\right)  = \left| {\{ \left( {i,j}\right) }\right| i < j,\alpha \left( i\right)  < \alpha \left( j\right) \text{while}\beta \left( i\right)  > \beta \left( j\right) \}  \mid  \text{.} \tag{6}
$$

The maximum value of $K\left( {\alpha ,\beta }\right)$ is $\frac{n\left( {n - 1}\right) }{2}$ ,which occurs if $\alpha$ is the reverse of $\beta$ . Kendall tau distance can be computed by finding the minimum number of pairwise swaps of adjacent elements needed to convert one ranking to the other. For example,if two voters, $\alpha$ and $\beta$ ,order candidates as $\alpha  = \{ A,B,C\}$ and $\beta  = \{ B,C,A\}$ ,the Kendall tau distance between the two rankings is 2 since it takes two swaps to convert one of the rankings to the other.

$K\left( {\alpha ,\beta }\right)$ 的最大值为$\frac{n\left( {n - 1}\right) }{2}$ ，当$\alpha$ 是$\beta$ 的逆序时会出现这种情况。肯德尔 tau 距离可以通过找出将一个排名转换为另一个排名所需的相邻元素的最小成对交换次数来计算。例如，如果两个选民$\alpha$ 和$\beta$ 对候选人的排序分别为$\alpha  = \{ A,B,C\}$ 和$\beta  = \{ B,C,A\}$ ，那么这两个排名之间的肯德尔 tau 距离为 2，因为需要两次交换才能将其中一个排名转换为另一个排名。

Diaconis and Graham [1977] showed that the Kendall tau distance can be approximated using the Footrule distance as $K\left( {\alpha ,\beta }\right)  \leq  F\left( {\alpha ,\beta }\right)  \leq  {2K}\left( {\alpha ,\beta }\right)$ ,which shows that Footrule and Kendall tau distance measures are equivalent to each other to some extent.

迪亚科尼斯（Diaconis）和格雷厄姆（Graham）[1977]表明，可以使用足尺距离（Footrule distance）来近似肯德尔 tau 距离（Kendall tau distance），即$K\left( {\alpha ,\beta }\right)  \leq  F\left( {\alpha ,\beta }\right)  \leq  {2K}\left( {\alpha ,\beta }\right)$ ，这表明足尺距离和肯德尔 tau 距离度量在一定程度上是等价的。

8.3.3. Kemeny Proposal. The optimal aggregation of different rankings produces a overall ranking with minimum distance with respect to the given rankings. Given $n$ different rankings ${\alpha }_{1},{\alpha }_{2},\ldots ,{\alpha }_{n}$ for a list of candidates,the normalized Kendall tau distance between the aggregated overall ranking $\rho$ and the given rankings is defined as

8.3.3. 凯米尼（Kemeny）提议。不同排名的最优聚合会产生一个与给定排名距离最小的总体排名。对于一组候选人的$n$ 个不同排名${\alpha }_{1},{\alpha }_{2},\ldots ,{\alpha }_{n}$ ，聚合后的总体排名$\rho$ 与给定排名之间的归一化肯德尔 tau 距离定义为

$$
\text{ Distance }\left( {\rho ,{\alpha }_{1},{\alpha }_{2},\ldots {\alpha }_{n}}\right)  = \mathop{\sum }\limits_{{i = 1}}^{n}\left( {K\left( {\rho ,\alpha \left( i\right) }\right) }\right) /n. \tag{7}
$$

Kemeny's proposal has been adopted for performing rank aggregation, based on the Kendall tau distance, in Dwork et al. [2001]. It was shown that computing Kemeny optimal aggregation is NP-hard even for $n = 4$ . Kemeny aggregation satisfies neutrality and consistency properties of voting methods.

德怀尔（Dwork）等人[2001]基于肯德尔 tau 距离，采用了凯米尼的提议来进行排名聚合。研究表明，即使对于$n = 4$ ，计算凯米尼最优聚合也是 NP 难问题。凯米尼聚合满足投票方法的中立性和一致性属性。

The hardness of solving the problem of distance-based rank aggregation is related to the chosen distance measure. In some settings, measures could be adopted to solve the problem in polynomial time,such as measuring the distance between only the top- $k$ lists rather than fully ranked lists. Applying the traditional measures in this case is not possible since it requires accessing the full lists. To work around this problem, extended versions of Kendall tau and Footrule distance measures were adopted by Fagin et al. [2003]. The main idea is to truncate the top- $k$ lists at various points $i \leq  k$ ,compute the symmetric difference metric between the resulting top $i$ lists,and take a suitable combination of them. Different cases are considered such as an item pair that appears in all top- $k$ lists as opposed to one or both items missing from one of the lists. A penalty is assigned to each case while computing the distance measures.

解决基于距离的排名聚合问题的难度与所选的距离度量有关。在某些情况下，可以采用一些度量方法在多项式时间内解决该问题，例如仅测量前$k$ 列表之间的距离，而不是全排名列表之间的距离。在这种情况下，无法应用传统的度量方法，因为它需要访问完整的列表。为了解决这个问题，法金（Fagin）等人[2003]采用了肯德尔 tau 距离和足尺距离度量的扩展版本。主要思想是在不同点$i \leq  k$ 截断前$k$ 列表，计算得到的前$i$ 列表之间的对称差度量，并对它们进行适当的组合。考虑了不同的情况，例如一个项目对出现在所有前$k$ 列表中，与一个或两个项目从其中一个列表中缺失的情况。在计算距离度量时，会为每种情况分配一个惩罚值。

## 9. CONCLUSIONS AND FUTURE RESEARCH DIRECTIONS

## 9. 结论与未来研究方向

We have surveyed top- $k$ processing techniques in relational databases. We provided a classification for top- $k$ techniques based on several dimensions such as the adopted query model, data access, implementation level, and supported ranking functions. We discussed the details of several algorithms to illustrate the different challenges and data management problems they address. We also discussed related top- $k$ processing techniques in the XML domain, as well as methods used for scoring XML elements. Finally, we provided a theoretical background of the ranking and top- $k$ processing problems from voting theory.

我们对关系数据库中的前$k$ 处理技术进行了综述。我们基于几个维度，如采用的查询模型、数据访问、实现级别和支持的排名函数，对前$k$ 技术进行了分类。我们讨论了几种算法的细节，以说明它们所解决的不同挑战和数据管理问题。我们还讨论了 XML 领域中相关的前$k$ 处理技术，以及用于对 XML 元素进行评分的方法。最后，我们从投票理论的角度提供了排名和前$k$ 处理问题的理论背景。

We envision the following research directions to be important to pursue:

我们认为以下研究方向值得重点关注：

-Dealing with uncertainty. Efficient processing of top- $k$ queries that deal with different sources of uncertainty and fuzziness, in both data and queries, is a challenging task. Designing uncertainty models to meet the needs of practical applications, as well as extending relational processing to conform with different probabilistic models, are two important issues with many unsolved problems. Exploiting the semantics of top- $k$ queries to identify optimization chances in these settings is another important question.

- 处理不确定性。高效处理涉及数据和查询中不同不确定性和模糊性来源的前$k$ 查询是一项具有挑战性的任务。设计满足实际应用需求的不确定性模型，以及扩展关系处理以符合不同的概率模型，是两个重要的问题，还有许多未解决的难题。在这些情况下，利用前$k$ 查询的语义来识别优化机会是另一个重要问题。

-Cost models. Building generic cost models for top- $k$ queries with different ranking functions is still in its primitive stages. Leveraging the concept of rank-awareness in query optimizers, and making use of rank-aware cost models is an important related direction.

- 成本模型。为具有不同排名函数的前$k$ 查询构建通用成本模型仍处于初级阶段。在查询优化器中利用排名感知的概念，并使用排名感知成本模型是一个重要的相关方向。

-Learning ranking functions. Learning ranking functions from users' profiles or feedback is an interesting research direction that involves many practical applications, especially in Web environments. Building intelligent systems that recognize user's preferences by interaction, and optimizing data storage and retrieval for efficient query processing, are two important problems.

- 学习排名函数。从用户的个人资料或反馈中学习排名函数是一个有趣的研究方向，涉及许多实际应用，特别是在 Web 环境中。构建通过交互识别用户偏好的智能系统，并优化数据存储和检索以实现高效查询处理，是两个重要的问题。

-Privacy and anonymization. Most current top- $k$ processing techniques assume that the rankings obtained from different sources are readily available. In some settings, revealing such rankings might be restricted or anonymized to protect private data. Processing top- $k$ queries using partially disclosed data is an interesting research topic.

- 隐私和匿名化。目前大多数前$k$ 处理技术假设从不同来源获得的排名是现成可用的。在某些情况下，为了保护隐私数据，可能会限制或匿名化此类排名的披露。使用部分披露的数据处理前$k$ 查询是一个有趣的研究课题。

## REFERENCES

## 参考文献

Acharya, S., Gibbons, P. B., AND PoosALA, V. 2000. Congressional samples for approximate answering of group-by queries. In Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data. 487-498.

Aмято, G., RabiĭтI, F., Savino, P., AND Zezula, P. 2003. Region proximity in metric spaces and its use for approximate similarity search. ACM Trans. Inform. Syst. 21, 2, 192-227.

Amer-Yahla, S., Koudas, N., Marian, A., Srivastava, D., and Toman, D. 2005. Structure and content scoring for xml. In Proceedings of the 31st International Conference on Very Large Data Bases. 361-372.

Aref, W. G., Catlin, A. C., Elmagarmin, A. K., Fan, J., Hammad, M. A., Ilyas, I. F., Marzouk, M., Prabsukaka, S., AND ZHU, X. 2004. VDBMS: A testbed facility for research in video database benchmarking. ACM Multimed. Syst. J. (Special Issue on Multimedia Document Management Systems) 9, 6, 575-585.

Arrow, K. 1951. Social Choice and Individual Values. Wiley, New York, NY.

Babcock, B., Chaudhuri, S., and Das, G. 2003. Dynamic sample selection for approximate query processing. In Proceedings of the 2003 ACM SIGMOD International Conference on Management of Data. 539- 550.

Barbara, D., Garcia-Molina, H., and Portre, D. 1992. The management of probabilistic data. IEEE Trans. Knowl. Data Eng. 4, 5, 487-502.

BLACK, D. 1958. The Theory of Committees and Elections. Cambridge University Press, London, U.K.

Börzsönyı, S., Kosshann, D., AND Stocker, K. 2001. The skyline operator. In Proceedings of the 17th International Conference on Data Engineering. 421.

Brams, S. J. and Fishburn, P. C. 1983. Approval Voting. Birkhauser, Boston, MA.

Bruno, N., Chaubhuri, S., AND Gravano, L. 2002a. Top-k selection queries over relational databases: Mapping strategies and performance evaluation. ACM Trans. Database Syst. 27, 2, 153-187.

Bruno, N., Gravano, L., and Marian, A. 2002b. Evaluating top- $k$ queries over Web-accessible databases. In Proceedings of the 18th International Conference on Data Engineering. 369.

Chakrabarti, K., Garofalakis, M., Rastogi, R., AND SHM, K. 2001. Approximate query processing using wavelets. VLDB J. 10, 2-3, 199-223.

Chang, K. C. and Hwang, S. 2002. Minimal probing: supporting expensive predicates for top- $k$ queries. In Proceedings of the 2001 ACM SIGMOD International Conference on Management of Data. 346- 357.

Chang, Y., Bergman, L. D., Castelli, V., Li, C., Lo, M., and Smitth, J. R. 2000. The Onion Technique: Indexing for linear optimization queries. In Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data. 391-402.

Chaudhura, S., Das, G., Datar, M., Morwani, R., and Narasakyуа, V. R. 2001a. Overcoming limitations of sampling for aggregation queries. In Proceedings of the 17th International Conference on Data Engineering. 534-542.

Chaudhuri, S., Das, G., AND Narasayyı, V. 2001b. A robust, optimization-based approach for approximate answering of aggregate queries. In Proceedings of the 2001 ACM SIGMOD International Conference on Management of Data. 295-306.

Chaudhuri, S., Motwani, R., and Narasasysa, V. 1998. Random sampling for histogram construction: How much is enough? In Proceedings of the 1998 ACM SIGMOD International Conference on Management of Data. 436-447.

Copelann, A. H. 1951. A reasonable social welfare function. Mimeo. Univer sity of Michigan, Ann Arbor, MI.

Cranor, L. 1996. Declared-strategy voting: An instrument for group decision-making. Ph.D. dissertation. Washington University, St. Louis, MO.

DAS,G.,GuNOPULOS,D.,Koudas,N.,AND TSIROGIANNIS,D. 2006. Answering top- $k$ queries using views. In Proceedings of the 32nd International Conference on Very Large Data Bases. 451-462.

DIACONIS, P. 1998. Group Representation in Probability and Statistics. Institute of Mathematical Statistics. Web site: www.imstat.org.

DIACONIS, P. AND GRAHAM, R. 1977. Spearman's footrule as a measure of disarray. J. Roy. Statist. Soc. Series ${B39},2,{262} - {268}$ .

DONJERKOVIC,D. AND RAMAKRISHNAN,R. 1999. Probabilistic optimization of top $N$ queries. In Proceedings of the 25th International Conference on Very Large Data Bases. 411-422.

Dwork, C., Kumar, S. R., Naor, M., and Sivakumar, D. 2001. Rank aggregation methods for the Web. In Proceedings of the 10th International Conference on World Wide Web. 613-622.

Fagin, R., Kumar, R., Mahdian, M., Sivakumar, D., and Vee, E. 2004. Comparing and aggregating rankings with ties. In Proceedings of the Twenty-Third ACM SIGMOD-SIGACT-SIGART Symposium on Principles of Database Systems. 47-58.

Fagin,R.,Kumak,R.,and Sivakumak,D. 2003. Comparing top $k$ lists. In Proceedings of the Fourteenth Annual ACM-SIAM Symposium on Discrete Algorithms. 28-36.

Fagin, R., Lorem, A., AND Naor, M. 2001. Optimal aggregation algorithms for middleware. J. Comput. Syst. Sci. 1, 1, 614-656.

FUHR, N. 1990. A probabilistic framework for vague queries and imprecise information in databases. In Proceedings of the 16th International Conference on Very Large Data Bases. 696-707.

GANTI, V., LEE, M., AND RAMAKRISHNAN, R. 2000. ICICLES: Self-tuning samples for approximate query answering. In Proceedings of the 26th International Conference on Very Large Data Bases. 176-187.

GETOOR, L. AND DIEHL, C. P. 2005. Link mining: A survey. ACM SIGKDD Explor. Newsl. 7, 2, 3-12.

Güntzer, U., Balke, W., AND Kleßling, W. 2000. Optimizing multi-feature queries for image databases. In Proceedings of the 26th International Conference on Very Large Data Bases. 419-428.

Güntzer, U., Balke, W., AND KIEßliNG, W. 2001. Towards efficient multi-feature queries in heterogeneous environments. In Proceedings of the International Conference on Information Technology: Coding and Computing. 622.

Guo, L., Shao, F., Borev, C., and Shannugassundaram, J. 2003. XRANK: Ranked keyword search over XML documents. In Proceedings of the 2003 ACM SIGMOD International Conference on Management of Data. 16-27.

HELLERSTEIN, J. M., HAAS, P. J., AND WANG, H. J. 1997. Online aggregation. In Proceedings of the 1997 ACM SIGMOD International Conference on Management of Data. 171-182.

Hoeffding, W. 1963. Probability inequalities for sums of bounded random variables. American Statistical Association Journal, 13-30.

HRISTIDIS, V., KOUDAS, N., AND PAPAKONSTANTINOU, Y. 2001. PREFER: A system for the efficient execution of multi-parametric ranked queries. In Proceedings of the 2001 ACM SIGMOD International Conference on Management of Data. 259-270.

HRISTIDIS, V. AND PAPAKONSTANTINOU, Y. 2004. Algorithms and applications for answering ranked queries using ranked views. VLDB J. 13, 1, 49-70.

Hwang,S. and Chang,K. C. 2007a. Optimizing top- $k$ queries for middleware access: A unified cost-based approach. ACM Trans. Database Syst. 32, 1, 5.

Hwang, S. and Chang, K. C. 2007b. Probe minimization by schedule optimization: Supporting top- $k$ queries with expensive predicates. IEEE Trans. Knowl. Data Eng. 19, 5, 646-662.

Ixyas, I. F., Aref, W. G., and Elmagarmin, A. K. 2002. Joining ranked inputs in practice. In Proceedings of the 28th International Conference on Very Large Data Bases. 950-961.

ILYAS, I. F., AREF, W. G., AND ELMAGARMID, A. K. 2004a. Supporting top- $k$ join queries in relational databases. VLDB J. 13, 3, 207-221.

Ixyas, I. F., Aref, W. G., Elmagarmin, A. K., Elmongui, H. G., Shah, R., And Vitт𝙴r, J. S. 2006. Adaptive rank-aware query optimization in relational databases. ACM Trans. Database Syst. 31, 4, 1257- 1304.

Ilyas, I. F., SнАн, R., AREF, W. G., VirTER, J. S., AND ELMAGARMID, A. K. 2004b. Rank-aware query optimization. In Proceedings of the 2004 ACM SIGMOD International Conference on Management of Data. 203- 214.

Ilyas, F. I., Walid, G. A., and ElhagarMID, A. K. 2003. Supporting top- $k$ join queries in relational databases. In Proceedings of the 29th International Conference on Very Large Databases. 754-765.

IMIELINSKI, T. AND LIPSKI JR., W. 1984. Incomplete information in relational databases. J. ACM 31, 4, 761- 791.

KENDALL, M. G. 1945. The treatment of ties in ranking problems. Biometrika 33, 3, 239-251.

KLAMLER, C. 2003. A comparison of the dodgson method and the copeland rule. Econ. Bull. 4, 8, 1-7.

LI, C., CHANG, K. C., AND ILYAS, I. F. 2006. Supporting ad-hoc ranking aggregates. In Proceedings of the 2006 ACM SIGMOD International Conference on Management of Data. 61-72.

Li, C., Chang, K. C., ILyas, I. F., AND Song, S. 2005. RankSQL: Query algebra and optimization for relational top- $k$ queries. In Proceedings of the 2005 ACM SIGMOD International Conference on Management of Data. 131-142.

MahouLis, N., Cheng, K. H., Yru, M. L., and Cheung, D. W. 2006. Efficient aggregation of ranked inputs. In Proceedings of the 22rd International Conference on Data Engineering. 72.

Marian,A.,Bruno,N.,AND Gravano,L. 2004. Evaluating top- $k$ queries over Web-accessible databases. ${ACM}$ Trans. Database Syst. 29, 2, 319-362.

Michel, S., Triantafillou, P., and Weikum, G. 2005. KLEE: A framework for distributed top- $k$ query algorithms. In Proceedings of the 31st International Conference on Very Large Data Bases. 637-648.

Natsev, A., Chang, Y., SmirrH, J. R., Li, C., AND VirTER, J. S. 2001. Supporting incremental join queries on ranked inputs. In Proceedings of the 27th International Conference on Very Large Data Bases. 281-290.

Nural, H. 1987. Comparing Voting Systems. D. Reidel Publishing Company, Dordrecht, Germany.

Ré, C., DALVI, N. N., AND SUCIU, D. 2007. Efficient top- $k$ query evaluation on probabilistic data. In Proceedings of the 23rd International Conference on Data Engineering. 886-895.

Robertson, S. E. AND WALKER, S. 1994. Some simple effective approximations to the 2-Poisson model for probabilistic weighted retrieval. In Proceedings of the 17th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval. 232-241.

Salton, G. and McGilli, M. J. 1983. Introduction to Modern IR. McGraw-Hill, New York, NY.

Silbergtein, A., Braynard, R., ELLIS, C. S., Munagala, K., and Yang, J. 2006. A sampling-based approach to optimizing top- $k$ queries in sensor networks. In Proceedings of the 22nd International Conference on Data Engineering. 68.

Solıman,M. A.,ILyas,I. F.,and Chang,K. C.-C. 2007. Top- $k$ query processing in uncertain databases. In Proceedings of the 23rd International Conference on Data Engineering. 896-905.

THEOBALD, M., SCHENKEL, R., AND WEIKUM, G. 2005. An efficient and versatile query engine for TopX search. In Proceedings of the 31st International Conference on Very Large Data Bases. 625-636.

THEOBALD, M., WEIKUM, G., AND SCHENKEL, R. 2004. Top- $k$ query evaluation with probabilistic guarantees. In Proceedings of the 30th International Conference on Very Large Data Bases. 648-659.

TRUCHON, M. 1998. An extension of the Condorcet criterion and Kemeny orders. Cahier 98-15. Centre de Recherche en Economie et Finance Appliquees, Université Laval, Québec, Canada.

Tsaparas, P., Pal.panas, T., Korinis, Y., Koudas, N., and Srivastava, D. 2003. Ranked join indices. In Proceedings of the 19th International Conference on Data Engineering. 277.

VRBSKY, S. V. AND LIU, J. W. S. 1993. APPROXIMATE—a query processor that produces monotonically improving approximate answers. IEEE Trans. Knowl. Data Eng. 5, 6, 1056-1068.

XIN, D., CHEN, C., AND HAN, J. 2006. Towards robust indexing for ranked queries. In Proceedings of the 32nd International Conference on Very Large Data Bases. 235-246.

XIN, D., HAN, J., AND CHANG, K. C. 2007. Progressive and selective merge: computing top- $k$ with ad-hoc ranking functions. In Proceedings of the 2007 ACM SIGMOD International Conference on Management of Data. 103-114.

Young, H. P. AND LevenGLICK, A. 1978. A consistent extension of condorcet’s election principle. SIAM J. Appl. Math. 35, 2, 285-300.

Yuan, Y., Lin, X., Liu, Q., Wang, W., Yu, J. X., AND Zhang, Q. 2005. Efficient computation of the skyline cube. In Proceedings of the 31st International Conference on Very Large Data Bases. 241-252.

Zhang, Z., Hwang, S., Chang, K. C., Wang, M., Lang, C. A., and Chang, Y. 2006. Boolean + ranking: Querying a database by $k$ -constrained optimization. In Proceedings of the 2006 ACM SIGMOD International Conference on Management of Data. 359-370.