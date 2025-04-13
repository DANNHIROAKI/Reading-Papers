# Comparison of Access Methods for Time-Evolving Data

# 时间演化数据访问方法的比较

BETTY SALZBERG

贝蒂·萨尔茨伯格

Northeastern University

东北大学

AND

VASSILIS J. TSOTRAS

瓦西里斯·J·索特拉斯

University of California, Riverside

加州大学河滨分校

This paper compares different indexing techniques proposed for supporting efficient access to temporal data. The comparison is based on a collection of important performance criteria, including the space consumed, update processing, and query time for representative queries. The comparison is based on worst-case analysis, hence no assumptions on data distribution or query frequencies are made. When a number of methods have the same asymptotic worst-case behavior, features in the methods that affect average case behavior are discussed. Additional criteria examined are the pagination of an index, the ability to cluster related data together, and the ability to efficiently separate old from current data (so that larger archival storage media such as write-once optical disks can be used). The purpose of the paper is to identify the difficult problems in accessing temporal data and describe how the different methods aim to solve them. A general lower bound for answering basic temporal queries is also introduced.

本文比较了为支持高效访问时态数据而提出的不同索引技术。该比较基于一系列重要的性能标准，包括所占用的空间、更新处理以及代表性查询的查询时间。该比较基于最坏情况分析，因此不对数据分布或查询频率做任何假设。当多种方法具有相同的渐近最坏情况行为时，将讨论这些方法中影响平均情况行为的特征。所考察的其他标准包括索引的分页、将相关数据聚类在一起的能力，以及有效分离旧数据和当前数据的能力（以便可以使用诸如一次写入光盘等较大的存档存储介质）。本文的目的是识别访问时态数据中的难题，并描述不同方法如何旨在解决这些问题。还引入了回答基本时态查询的一般下界。

Categories and Subject Descriptors: H.2.2 [Database Management]: Physical Design; Access methods; H.3.1 [Information Storage and Retrieval]: Content Analysis and Indexing-Indexing methods

类别和主题描述符：H.2.2 [数据库管理]：物理设计；访问方法；H.3.1 [信息存储与检索]：内容分析与索引 - 索引方法

General Terms: Management, Performance

通用术语：管理、性能

Additional Key Words and Phrases: Access methods, I/O performance, structures, temporal databases

其他关键词和短语：访问方法、I/O 性能、结构、时态数据库

## 1. INTRODUCTION

## 1. 引言

Conventional database systems capture only a single logical state of the modeled reality (usually the most current). Using transactions, the database evolves from one consistent state to the next, while the previous state is discarded after a transaction commits. As a result, there is no memory with respect to prior

传统数据库系统仅捕获所建模现实的单一逻辑状态（通常是最新状态）。通过事务，数据库从一个一致状态演变为下一个一致状态，而前一个状态在事务提交后被丢弃。因此，对于先前状态没有记忆

---

<!-- Footnote -->

Betty Salzberg's work was supported by NSF grants IRI-9303403 and IRI-9610001. Vassilis Tsotras' work was performed while the author was with the Department of Computer Science, Polytechnic University, Brooklyn, NY 11201; it was supported by NSF grants IRI-9111271, IRI-9509527, and by the New York State Science and Technology Foundation as part of its Center for Advanced Technology program.

贝蒂·萨尔茨伯格的工作得到了美国国家科学基金会（NSF）资助项目 IRI - 9303403 和 IRI - 9610001 的支持。瓦西里斯·索特拉斯的工作是在作者任职于纽约布鲁克林理工大学（邮编 11201）计算机科学系期间完成的；该工作得到了美国国家科学基金会资助项目 IRI - 9111271、IRI - 9509527 的支持，以及纽约州科学与技术基金会作为其先进技术中心项目的一部分的支持。

Authors' addresses: B. Salzberg, College of Computer Science, Northeastern University, Boston, MA 02115; email: salzberg@ccs.neu.edu; V. J. Tsotras, Department of Computer Science and Engineering, University of California, Riverside, Riverside, CA 92521; email: tsotras@cs.ucr.edu.

作者地址：B. 萨尔茨伯格，东北大学计算机科学学院，马萨诸塞州波士顿市 02115；电子邮件：salzberg@ccs.neu.edu；V. J. 索特拉斯，加州大学河滨分校计算机科学与工程系，加利福尼亚州河滨市 92521；电子邮件：tsotras@cs.ucr.edu。

Permission to make digital/hard copy of part or all of this work for personal or classroom use is granted without fee provided that the copies are not made or distributed for profit or commercial advantage, the copyright notice, the title of the publication, and its date appear, and notice is given that copying is by permission of the ACM, Inc. To copy otherwise, to republish, to post on servers, or to redistribute to lists, requires prior specific permission and/or a fee.

允许个人或课堂使用免费制作本作品部分或全部内容的数字/硬拷贝，前提是这些拷贝不是为了盈利或商业利益而制作或分发，并且要显示版权声明、出版物标题及其日期，并注明复制需获得美国计算机协会（ACM）的许可。否则，若要复制、重新发布、上传到服务器或分发给列表，则需要事先获得特定许可和/或支付费用。

© 1999 ACM 0360-0300/99/0600-0158 \$5.00

© 1999 美国计算机协会 0360 - 0300/99/0600 - 0158 5 美元

<!-- Footnote -->

---

## CONTENTS

## 目录

1. Introduction

1. 引言

2. Problem Specification

2. 问题说明

3. Items for comparison

3. 比较项

3.1 Queries

3.1 查询

3.2 Access Method Costs

3.2 访问方法成本

3.3 Index Pagination and Data Clustering

3.3 索引分页与数据聚类

3.4 Migration of Past Data to Another Location

3.4 历史数据迁移至其他位置

3.5 Lower Bounds on I/O Complexity

3.5 I/O复杂度的下界

4. Efficient Method Design For Transaction/Bitemporal

4. 事务/双时态数据的高效方法设计

Data

数据

4.1 The Transaction Pure-Timeslice Query

4.1 事务纯时间片查询

4.2 The Transaction Pure-Key Query

4.2 事务纯键查询

4.3 The Transaction Range-Timeslice Query

4.3 事务范围时间片查询

4.4 Bitemporal Queries

4.4 双时态查询

4.5 Separating Past from Current Data and Use of

4.5 分离历史数据与当前数据并使用

WORM disks

一次写入多次读取（WORM）磁盘

5. Method Classification and Comparison

5. 方法分类与比较

5.1 Transaction-Time Methods

5.1 事务时间方法

5.2 Valid-Time Methods

5.2 有效时间方法

5.3 Bitemporal Methods

5.3 双时态方法

6. Conclusions

6. 结论

states of the data. Such database systems capture a single snapshot of reality (also called snapshot databases), and are insufficient for those applications that require the support of past, current, or even future data. What is needed is a temporal database system [Snodgrass and Ahn 1986]. The term "temporal database" refers in general to a database system that supports some time domain, and is thus able to manage time-varying data. (Note that this definition excludes user-defined time, which is an uninterpreted time domain directly managed by the user and not by the database system.)

数据的状态。此类数据库系统只能捕捉现实的单一快照（也称为快照数据库），对于那些需要支持过去、当前甚至未来数据的应用来说是不够的。我们需要的是一个时态数据库系统 [Snodgrass 和 Ahn 1986]。“时态数据库” 这一术语通常指支持某个时间域的数据库系统，因此能够管理随时间变化的数据。（请注意，此定义不包括用户定义的时间，用户定义的时间是由用户直接管理而非由数据库系统管理的未解释时间域。）

Research in temporal databases has grown immensly in recent years [Tso-tras and Kumar 1996]. Various aspects of temporal databases have been examined [Ozsoyoglu and Snodgrass 1995], including temporal data models, query languages, access methods, etc. Prototype efforts appear in Böhlen [1995]. In this paper we provide a comparison of proposed temporal access methods, i.e., indexing techniques for temporal data. We attempt to identify the problems in the area, together with the solutions given by each method.

近年来，时态数据库的研究取得了巨大进展 [Tso - tras 和 Kumar 1996]。时态数据库的各个方面都得到了研究 [Ozsoyoglu 和 Snodgrass 1995]，包括时态数据模型、查询语言、访问方法等。Böhlen [1995] 中展示了一些原型工作。在本文中，我们对已提出的时态访问方法（即时态数据的索引技术）进行了比较。我们试图找出该领域存在的问题，以及每种方法给出的解决方案。

A taxonomy of time in databases was developed in Snodgrass and Ahn [1995]. Specifically, transaction time and valid time have been proposed. Transaction and valid time are two orthogonal time dimensions. Transaction time is defined as the time when a fact is stored in the database. It is consistent with the transaction serialization order (i.e., it is monotonically increasing), and can be implemented using the commit times of transactions [Salzberg 1994]. Valid time is defined as the time when a fact becomes effective (valid) in reality. Depending on the time dimension(s) supported, there are three kinds of temporal databases: transaction-time, valid-time, and bitemporal [Dyreson et al. 1994].

Snodgrass 和 Ahn [1995] 提出了数据库中时间的分类方法。具体来说，提出了事务时间和有效时间的概念。事务时间和有效时间是两个正交的时间维度。事务时间定义为事实存储到数据库中的时间。它与事务序列化顺序一致（即单调递增），可以使用事务的提交时间来实现 [Salzberg 1994]。有效时间定义为事实在现实中变得有效（合法）的时间。根据所支持的时间维度，时态数据库有三种类型：事务时间数据库、有效时间数据库和双时态数据库 [Dyreson 等人 1994]。

A transaction-time database records the history of a database activity rather than real-world history. As such, it can "rollback" to one of its previous states. Since previous transaction times cannot be changed (every change is stamped with a new transaction time), there is no way to change the past. This is useful for applications in auditing, billing, etc. A valid-time database maintains the entire temporal behavior of an enterprise as best known now. It stores our current knowledge about the enterprise's past, current, or even future behavior. If errors are discovered in this temporal behavior, they are corrected by modifying the database. When a correction is applied, previous values are not retained. It is thus not possible to view the database as it was before the correction. A bitemporal database combines the features of the other two types. It more accurately represents reality and allows for retroactive as well as postactive changes.

事务时间数据库记录的是数据库活动的历史，而非现实世界的历史。因此，它可以 “回滚” 到其之前的某个状态。由于之前的事务时间无法更改（每次更改都会标记一个新的事务时间），所以无法改变过去。这对于审计、计费等应用很有用。有效时间数据库尽可能完整地维护企业的整个时态行为。它存储了我们目前对企业过去、当前甚至未来行为的了解。如果发现这种时态行为存在错误，则通过修改数据库来进行纠正。当进行纠正时，不会保留之前的值。因此，无法查看纠正之前的数据库状态。双时态数据库结合了前两种类型的特点。它能更准确地表示现实，并且允许进行追溯性和事后性的更改。

The tuple-versioning temporal model [Lorentzos and Johnson 1988; Navathe and Ahmed 1987] is used in this paper. Under this model, the database is a set of records (tuples) that store the versions of real-life objects. Each such record has a time-invariant key (surrogate) and, in general, a number of time-variant attributes; for simplicity, we assume that each record has exactly one time-varying attribute. In addition, it has one or two intervals, depending on which types of time are supported. Each interval is represented by two attributes: start_time and end_time.

本文使用了元组版本化时态模型 [Lorentzos 和 Johnson 1988；Navathe 和 Ahmed 1987]。在该模型下，数据库是一组记录（元组），用于存储现实生活中对象的版本。每个这样的记录都有一个时间不变的键（代理键），并且通常有多个随时间变化的属性；为了简单起见，我们假设每个记录恰好有一个随时间变化的属性。此外，根据所支持的时间类型，它有一个或两个时间间隔。每个时间间隔由两个属性表示：开始时间和结束时间。

Accurate specification of the problem that needs to be solved is critical in the design of any access method. This is particularly important in temporal databases, since problem specification depends dramatically on the time dimension(s) supported. Whether valid and/or transaction times are supported directly affects the way records are created or updated. In the past, this resulted in much confusion in the design of temporal access methods. To exemplify the distinct characteristics of the transaction and valid time dimensions, we use a separate abstraction to describe the central underlying problem for each kind of temporal database.

准确地定义需要解决的问题对于任何访问方法的设计都至关重要。这在时态数据库中尤为重要，因为问题的定义在很大程度上取决于所支持的时间维度。是否支持有效时间和/或事务时间会直接影响记录的创建或更新方式。过去，这在时态访问方法的设计中造成了很多混淆。为了说明事务时间和有效时间维度的不同特点，我们使用单独的抽象来描述每种时态数据库的核心底层问题。

The query performance of the methods examined is compared in the context of various temporal queries. In order to distinguish among the various kinds of queries, we use a general temporal query classification scheme [Tso-tras et al. 1998]. The paper also introduces lower bounds for answering basic temporal queries. Each lower bound assumes a disk-oriented environment and describes the minimal $\mathrm{I}/\mathrm{O}$ for solving the query if space consumption is kept minimal. We also show access methods that achieve a matching upper bound for a temporal query.

在各种时态查询的上下文中比较了所研究方法的查询性能。为了区分各种类型的查询，我们使用了一种通用的时态查询分类方案 [Tso - tras 等人 1998]。本文还给出了回答基本时态查询的下界。每个下界都假设是面向磁盘的环境，并描述了在空间消耗最小的情况下解决查询所需的最小 $\mathrm{I}/\mathrm{O}$。我们还展示了能达到时态查询匹配上界的访问方法。

Among the methods discussed, the ones that support transaction time (either in a transaction time or in a bitemporal environment) assume a linear transaction-time evolution [Ozsoyoglu and Snodgrass 1995]. This implies that a new database state is created by updating only the current database state. Another option is the so-called branching transaction time [Ozsoyoglu and Snodgrass 1995], where evolutions can be created from any past database state. Such branched evolutions form a tree-of-evolution that resembles the version-trees found in versioning environments. A version-tree is formed as new versions emanate from any previous version (assume that no version merging is allowed). There is however a distinct difference that makes branched evolutions a more difficult problem. In a version-tree, every new version is uniquely identified by a successive version number that can be used to access it directly [Driscoll et al. 1989; Lanka and Mays 1991]. In contrast, branched evolutions use timestamps. These timestamps enable queries on the evolution on a given branch. However, timestamps are not unique. The same time instant can exist as a timestamp in many branches in a tree-of-evolution simply because many updates could have been recorded at that time in various branches. We are aware of only two works that address problems related to indexing branching transaction time, namely Salzberg and Lomet [1995] and Landau et al. [1995]. In the ongoing work of Salzberg and Lomet [1995], version identifiers are replaced by (branch identifier, timestamp) pairs. Both a tree access method and a forest access method are proposed for these branched versions. Landau et al. [1995] provides data structures for (a) locating the relative position of a timestamp on the evolution of a given branch and (b) locating the same timestamp among sibling branches. Clearly, more work is needed in this area.

在讨论的方法中，支持事务时间（无论是在事务时间环境还是双时态环境中）的方法都假定事务时间呈线性演变[奥兹索约格鲁（Ozsoyoglu）和斯诺德格拉斯（Snodgrass），1995年]。这意味着新的数据库状态仅通过更新当前数据库状态来创建。另一种选择是所谓的分支事务时间[奥兹索约格鲁（Ozsoyoglu）和斯诺德格拉斯（Snodgrass），1995年]，在这种情况下，可以从任何过去的数据库状态创建演变。这种分支演变形成了一个演变树，类似于版本控制环境中的版本树。版本树是随着新版本从任何先前版本派生出来而形成的（假设不允许版本合并）。然而，有一个明显的区别使得分支演变成为一个更难的问题。在版本树中，每个新版本都由一个连续的版本号唯一标识，该版本号可用于直接访问它[德里斯科尔（Driscoll）等人，1989年；兰卡（Lanka）和梅斯（Mays），1991年]。相比之下，分支演变使用时间戳。这些时间戳可以对给定分支上的演变进行查询。然而，时间戳不是唯一的。同一个时间点可以作为时间戳存在于演变树的许多分支中，仅仅是因为在那个时间可能在各个分支中记录了许多更新。我们只知道有两项研究涉及与分支事务时间索引相关的问题，即萨尔茨伯格（Salzberg）和洛梅特（Lomet）[1995年]以及兰道（Landau）等人[1995年]。在萨尔茨伯格（Salzberg）和洛梅特（Lomet）[1995年]正在进行的工作中，版本标识符被（分支标识符，时间戳）对所取代。针对这些分支版本，提出了一种树访问方法和一种森林访问方法。兰道（Landau）等人[1995年]提供了用于（a）定位给定分支演变上时间戳的相对位置以及（b）在兄弟分支中定位相同时间戳的数据结构。显然，这一领域还需要更多的研究工作。

Other kinds of temporal, in particular time-series, queries have recently appeared: [Agrawal and Swami 1993; Fa-loutsos et al. 1994; Jagadish et al. 1995; Seshadri et al. 1996; Motakis and Za-niolo 1997]. A pattern and a time-series (an evolution) are given, and the typical query asks for all times that a similar pattern appeared in the series. The search involves some distance criterion that qualifies when a pattern is similar to the given pattern. The distance criterion guarantees no false dismissals (false alarms are eliminated afterwards). Whole pattern-matching [Agrawal et al. 1993] and submatching [Faloutsos et al. 1994] queries have been examined. Such time-series queries are reciprocal in nature to the temporal queries addressed here (which usually provide a time instant and ask for the pattern at that time), and are not covered in this paper.

其他类型的时态查询，特别是时间序列查询，最近已经出现：[阿格拉瓦尔（Agrawal）和斯瓦米（Swami），1993年；法洛托斯（Faloutsos）等人，1994年；贾加迪什（Jagadish）等人，1995年；塞沙德里（Seshadri）等人，1996年；莫塔基斯（Motakis）和扎尼奥洛（Zaniolo），1997年]。给定一个模式和一个时间序列（一种演变），典型的查询是询问该序列中所有出现类似模式的时间。搜索涉及某种距离准则，用于判断一个模式何时与给定模式相似。该距离准则保证不会有漏判（误报随后会被消除）。已经研究了全模式匹配[阿格拉瓦尔（Agrawal）等人，1993年]和子匹配[法洛托斯（Faloutsos）等人，1994年]查询。这种时间序列查询在本质上与本文所讨论的时态查询是相反的（本文的时态查询通常提供一个时间点并询问该时间点的模式），本文不涉及此类查询。

<!-- Media -->

<!-- figureText: $h$ ${t}_{7}$ ${t}_{10}$ $b$ $f$ -->

<img src="https://cdn.noedgeai.com/0195c90d-0a33-750f-877f-c1cec48583bb_3.jpg?x=423&y=236&w=805&h=208&r=0"/>

Figure 1. An example evolution where changes occur in increasing time order. The evolution is depicted at time ${t}_{10}$ . Lines ending in ’ $>$ ’ correspond to objects that have not yet been deleted. At ${t}_{10}$ , state $s\left( {t}_{9}\right)  = \{ a,f,g\}$ is updated by the addition of object $e$ to create state $s\left( {t}_{10}\right)  = \left\{  {a,f,g,e}\right\}$ .

图1. 一个变化按时间递增顺序发生的演变示例。该演变在时间${t}_{10}$ 时进行了描绘。以’ $>$ ’结尾的线对应尚未被删除的对象。在${t}_{10}$ 时，状态$s\left( {t}_{9}\right)  = \{ a,f,g\}$ 通过添加对象$e$ 进行更新，从而创建状态$s\left( {t}_{10}\right)  = \left\{  {a,f,g,e}\right\}$ 。

<!-- Media -->

The rest of the paper is organized as follows: Section 2 specifies the basic problem underlying each of the three temporal databases. We categorize a method as transaction-time, valid-time, and bitemporal, depending on which time dimension(s) it most efficiently supports. Section 3 presents the items on which our comparison was based, including the lower bounds. Section 4 discusses in more detail the basic characteristics that a good transaction or bitemporal access method should have. The examined methods are presented in Section 5. The majority falls in the transaction-time category, which comprises the bulk of this paper (Section 5.1). Within the transaction-time category, we further classify methods according to what queries they support more efficiently (key-only, time-only, or time-key methods). A table summarizing the worst-case performance characteristics of the transaction-time methods is also included. For completeness, we also cover valid-time and bitemporal methods in Sections 5.2 and 5.3, respectively. We conclude the paper with a discussion on the remaining open problems.

本文的其余部分组织如下：第2节详细说明了三种时态数据库各自所基于的基本问题。我们根据一种方法最有效地支持的时间维度，将其分类为事务时间、有效时间和双时态方法。第3节介绍了我们进行比较所依据的项目，包括下界。第4节更详细地讨论了一个好的事务或双时态访问方法应具备的基本特征。第5节介绍了所研究的方法。大多数方法属于事务时间类别，这也是本文的主要内容（第5.1节）。在事务时间类别中，我们进一步根据方法更有效地支持的查询类型（仅键查询、仅时间查询或时间 - 键查询）对其进行分类。还包括一个总结事务时间方法最坏情况性能特征的表格。为了完整起见，我们还分别在第5.2节和第5.3节中介绍了有效时间和双时态方法。最后，我们对仍未解决的问题进行了讨论，以此结束本文。

## 2. PROBLEM SPECIFICATION

## 2. 问题说明

The following discussion is influenced by Snodgrass and Ahn [1986], where the differences between valid and transaction times were introduced and illustrated by various examples. Here we attempt to identify the implications for access method design from support of each time dimension.

以下讨论受到了斯诺德格拉斯（Snodgrass）和安（Ahn）[1986年]的影响，他们在文中介绍了有效时间和事务时间之间的差异，并通过各种示例进行了说明。在这里，我们试图确定支持每个时间维度对访问方法设计的影响。

To visualize a transaction-time database, consider first an initially empty set of objects that evolves over time as follows. Time is assumed to be discrete and is described by a succession of consecutive nonnegative integers. Any change is assumed to occur at a time indicated by one of these integers. A change is the addition or deletion of an object or the value change (modification) of the object's attribute. A real life example is the evolution of the employees in a company. Each employee has a surrogate (ssn) and a salary attribute. The changes include additions of new employees (as they are hired or rehired), salary changes or employee deletions (as they retire or leave the company). Since an attribute value change can be represented by the artificial deletion of the object, followed by the simultaneous rebirth of the object with the modified attribute, we may concentrate on object additions or deletions. Such an evolution appears in Figure 1. An object is alive from the time it is added in the set and until (if ever) it is deleted from the set. The state of the evolving set at time $t$ ,namely $s\left( t\right)$ ,consists of all the alive objects at $t$ . Note that changes are always applied to the most current state $s\left( t\right)$ ,i.e.,past states cannot be changed.

为了直观呈现事务时间数据库，首先考虑一个最初为空的对象集合，它随时间的演变如下。假设时间是离散的，用一系列连续的非负整数来描述。假设任何变化都发生在这些整数所表示的某个时间点。变化包括对象的添加或删除，或者对象属性的值变更（修改）。一个现实生活中的例子是公司员工的变动情况。每个员工都有一个代理键（社保号码，ssn）和一个薪资属性。这些变化包括新员工的入职（新聘或重新聘用）、薪资调整或员工离职（退休或离开公司）。由于属性值的变更可以通过人为删除对象，然后立即以修改后的属性重新创建该对象来表示，因此我们可以专注于对象的添加或删除。这种演变如图1所示。一个对象从被添加到集合中开始存活，直到（如果有的话）从集合中被删除。在时间$t$时，不断演变的集合的状态，即$s\left( t\right)$，由在$t$时所有存活的对象组成。请注意，变化总是应用于最新状态$s\left( t\right)$，即过去的状态不能被更改。

Assume that the history of the above evolution is to be stored in a database. Since time is always increasing and the past is unchanged, a transaction time database can be utilized with the implicit updating assumption: that when an object is added or deleted from the evolving set at time $t$ ,a transaction updates the database system about this change at the same time, i.e., this transaction has commit timestamp $t$ .

假设上述演变的历史记录要存储在数据库中。由于时间总是在增加且过去的记录不会改变，因此可以使用事务时间数据库，并采用隐式更新假设：当在时间$t$从不断演变的集合中添加或删除一个对象时，一个事务会同时更新数据库系统关于此变化的信息，即该事务的提交时间戳为$t$。

When a new object is added on the evolving set at time $t$ ,a record representing this object is stored in the database accompanied by a transaction-time interval of the form $\lbrack t,{now})$ . now is a variable representing the current transaction time, used because at the time the object is born its deletion time is yet unknown. If this object is later deleted at time ${t}^{\prime }$ the transaction-time interval of the corresponding record is updated to $\left\lbrack  {t,{t}^{\prime }}\right)$ . Thus,an object deletion in the evolving set is represented as a "logical" deletion in the database (the record of the deleted object is still retained in the database, but with a different transaction end_time).

当在时间$t$向不断演变的集合中添加一个新对象时，代表该对象的一条记录会存储在数据库中，并附带一个形式为$\lbrack t,{now})$的事务时间间隔。now是一个表示当前事务时间的变量，使用它是因为在对象创建时，其删除时间尚不清楚。如果该对象后来在时间${t}^{\prime }$被删除，则相应记录的事务时间间隔会更新为$\left\lbrack  {t,{t}^{\prime }}\right)$。因此，不断演变的集合中对象的删除在数据库中表现为“逻辑”删除（被删除对象的记录仍保留在数据库中，但事务结束时间不同）。

Since a transaction-time database system keeps both current and past data, it is natural to introduce the notion of a logical database state as a function of time. We therefore distinguish between the database system and the logical database state. (This is not required in traditional database systems because there always exists exactly one logical database state-the current one.) The logical database state at time $t$ consists of those records whose transaction time interval contains $t$ . Under the implicit updating assumption, the logical database state is equivalent to the state $s\left( t\right)$ of the observed evolving set. Since an object can be reborn, there may be many records (or versions) that are stored in the database system representing the history of the same object. But all these records correspond to disjoint transaction-time intervals in the object's history, and each such record can belong to a single logical database state.

由于事务时间数据库系统同时保存当前数据和过去的数据，因此很自然地会引入逻辑数据库状态作为时间的函数这一概念。因此，我们区分了数据库系统和逻辑数据库状态。（在传统数据库系统中不需要这样做，因为始终只有一个逻辑数据库状态——当前状态。）在时间$t$的逻辑数据库状态由那些事务时间间隔包含$t$的记录组成。在隐式更新假设下，逻辑数据库状态等同于所观察到的不断演变的集合的状态$s\left( t\right)$。由于一个对象可以重新创建，因此数据库系统中可能存储有许多代表同一对象历史的记录（或版本）。但所有这些记录对应于该对象历史中不相交的事务时间间隔，并且每个这样的记录只能属于一个逻辑数据库状态。

To summarize, an access method for a transaction-time database needs to (a) store its past logical states, (b) support addition/deletion/modification changes on the objects of its current logical state, and (c) efficiently access and query the objects in any of its states.

综上所述，事务时间数据库的访问方法需要：(a) 存储其过去的逻辑状态；(b) 支持对其当前逻辑状态中的对象进行添加/删除/修改操作；(c) 高效地访问和查询其任何状态下的对象。

In general, a fact can be entered in the database at a different time than when it happened in reality. This implies that the transaction-time interval associated with a record is actually related to the process of updating the database (the database activity), and may not accurately represent the period the corresponding object was alive in reality.

一般来说，一个事实录入数据库的时间可能与它在现实中发生的时间不同。这意味着与一条记录关联的事务时间间隔实际上与更新数据库的过程（数据库活动）相关，可能无法准确表示相应对象在现实中存在的时间段。

A valid-time database has a different abstraction. To visualize it, consider a dynamic collection of interval-objects. We use the term interval-object to emphasize that the object carries a valid-time interval to represent the validity period of some object property. (In contrast, and to emphasize that transaction-time represents the database activity rather than reality, we term the objects in the transaction-time abstraction as plain-objects.) The allowable changes are the addition/deletion/ modification of an interval-object, but the collection's evolution (past states) is not kept. An example of a dynamic collection of object-intervals appears in Figure 2.

有效时间数据库有不同的抽象概念。为了直观理解它，考虑一个动态的区间对象集合。我们使用“区间对象”这个术语来强调该对象带有一个有效时间间隔，以表示某个对象属性的有效时间段。（相比之下，为了强调事务时间代表的是数据库活动而非现实情况，我们将事务时间抽象中的对象称为普通对象。）允许的操作是对区间对象进行添加/删除/修改，但不保留集合的演变历史（过去的状态）。对象区间的动态集合示例见图2。

As a real-life example, consider the collection of contracts in a company. Each contract has an identity (contract_no), an amount attribute, and an interval representing the contract's duration or validity. Assume that when a correction is applied only the corrected contract is kept.

举一个现实生活中的例子，考虑公司的合同集合。每份合同都有一个标识（合同编号，contract_no）、一个金额属性，以及一个表示合同期限或有效期的区间。假设进行更正时，只保留更正后的合同。

A valid-time database is suitable for this environment. When an object is added to the collection, it is stored in the database as a record that contains the object's attributes (including its valid-time interval). The time of the record's insertion in the database is not kept. When an object deletion occurs, the corresponding record is physically deleted from the database. If an object attribute is modified, its corresponding record attribute is updated, but the previous attribute value is not retained. The valid-time database keeps only the latest "snapshot" of the collection of interval-objects. Querying a valid-time database cannot give any information on the past states of the database or how the collection evolved. Note that the database may store records with the same surrogate but with nonintersecting valid-time intervals.

有效时间数据库适用于这种环境。当一个对象被添加到集合中时，它会作为一条包含该对象属性（包括其有效时间间隔）的记录存储在数据库中。记录插入数据库的时间不会被保留。当发生对象删除操作时，相应的记录会从数据库中物理删除。如果修改了对象的属性，其对应的记录属性会被更新，但之前的属性值不会保留。有效时间数据库只保留区间对象集合的最新“快照”。查询有效时间数据库无法获取有关数据库过去状态或集合如何演变的任何信息。请注意，数据库可能会存储具有相同代理键但有效时间间隔不相交的记录。

<!-- Media -->

<!-- figureText: previous Collection ${l}_{2}$ valid-time axis (b) ${I}_{1}$ valid-time axis new Collection -->

<img src="https://cdn.noedgeai.com/0195c90d-0a33-750f-877f-c1cec48583bb_5.jpg?x=311&y=224&w=362&h=649&r=0"/>

Figure 2. Two states of a dynamic collection of interval-objects. Only the valid-time intervals of the objects are shown. The new collection (b) is created from the previous collection (a) after deleting object ${I}_{1}$ and adding object ${I}_{2}$ . Only the new (latest) collection is retained.

图2. 区间对象动态集合的两种状态。仅显示了对象的有效时间间隔。新集合（b）是在删除对象${I}_{1}$并添加对象${I}_{2}$后，从先前的集合（a）创建的。仅保留新的（最新的）集合。

<!-- Media -->

The notion of time is now related to the valid-time axis. Given a valid-time point, interval-objects can be classified as past, future, or current (alive), as related to this point, if their valid-time interval is before, after, or contains the given point. Valid-time databases are said to correct errors anywhere in the valid-time domain (past, current, or future) because the record of any interval-object in the collection can be changed independently of its position on the valid-time axis.

现在，时间的概念与有效时间轴相关。给定一个有效时间点，如果区间对象的有效时间间隔在该点之前、之后或包含该点，则可以将区间对象相对于该点分类为过去、未来或当前（存活）。据说有效时间数据库可以纠正有效时间域中任何位置（过去、当前或未来）的错误，因为集合中任何区间对象的记录都可以独立于其在有效时间轴上的位置进行更改。

An access method for a valid-time database should (a) store the latest collection of interval-objects, (b) support addition/deletion/modification changes to this collection, and (c) efficiently query the interval-objects contained in the collection when the query is asked.

有效时间数据库的访问方法应（a）存储区间对象的最新集合，（b）支持对该集合进行添加/删除/修改操作，并且（c）在进行查询时能高效地查询集合中包含的区间对象。

Reality is more accurately represented if both time dimensions are supported. The abstraction of a bitemporal database can be viewed as keeping the evolution (through the support of transaction-time) of a dynamic collection of (valid-time) interval-objects. Figure 3 (taken from Kumar et al. [1998]) offers a conceptual view of a bitemporal database. Instead of a single collection of interval-objects, there is a sequence of collections indexed by transaction time. If each interval-object represents a company contract, we can now represent how our knowledge about such contracts evolved. When an interval-object is inserted in the database at transaction-time $t$ ,a record is created with the object's surrogate (contract_no), attribute (contract amount) and valid-time interval (contract duration), and an initial transaction-time interval $\lbrack t$ , now). The transaction-time interval endpoint is changed to another transaction time if this object is updated later. For example, the record for interval-object ${I}_{2}$ has transaction-time interval $\left\lbrack  {{t}_{2},{t}_{4}}\right)$ ,since it was inserted in the database at transaction-time ${t}_{2}$ and "deleted" at ${t}_{4}$ . Such a scenario occurs if at time ${t}_{4}$ we realize that a contract was wrongly inserted at the database.

如果同时支持两个时间维度，现实将得到更准确的表示。双时态数据库的抽象可以看作是保留（有效时间）区间对象动态集合的演变（通过支持事务时间）。图3（取自Kumar等人[1998]）提供了双时态数据库的概念视图。不是单个区间对象集合，而是有一个按事务时间索引的集合序列。如果每个区间对象代表一份公司合同，我们现在就可以表示我们对这些合同的了解是如何演变的。当在事务时间$t$将一个区间对象插入数据库时，会创建一条记录，其中包含对象的代理键（合同编号）、属性（合同金额）和有效时间间隔（合同期限），以及一个初始事务时间间隔$\lbrack t$，现在）。如果该对象后来被更新，事务时间间隔的端点将更改为另一个事务时间。例如，区间对象${I}_{2}$的记录的事务时间间隔为$\left\lbrack  {{t}_{2},{t}_{4}}\right)$，因为它在事务时间${t}_{2}$插入数据库，并在${t}_{4}$“删除”。如果在时间${t}_{4}$我们意识到一份合同被错误地插入了数据库，就会出现这种情况。

A bitemporal access method should (a) store its past logical states, (b) support addition/deletion/modification changes on the interval-objects of its current logical state, and (c) efficiently access and query the interval-objects on any of its states.

双时态访问方法应（a）存储其过去的逻辑状态，（b）支持对其当前逻辑状态的区间对象进行添加/删除/修改操作，并且（c）能高效地访问和查询其任何状态下的区间对象。

Figure 3 is helpful in summarizing the differences among the underlying problems of the various database types. A transaction-time database differs from a bitemporal database in that it maintains the history of an evolving set of plain-objects instead of interval-objects. A valid-time database differs from a bitemporal since it keeps only one collection of interval-objects (the latest). Each collection $C\left( {t}_{i}\right)$ can be thought of on its own as a separate valid-time database. A transaction-time database differs from a (traditional) snapshot database in that it also keeps its past states instead of only the latest state. Finally, the difference between a valid-time and a snapshot database is that the former keeps interval-objects (and these intervals can be queried).

图3有助于总结各种数据库类型的潜在问题之间的差异。事务时间数据库与双时态数据库的不同之处在于，它维护的是一组普通对象的演变历史，而不是区间对象。有效时间数据库与双时态数据库不同，因为它只保留一个区间对象集合（最新的）。每个集合$C\left( {t}_{i}\right)$可以单独看作一个独立的有效时间数据库。事务时间数据库与（传统的）快照数据库的不同之处在于，它还保留其过去的状态，而不仅仅是最新状态。最后，有效时间数据库和快照数据库的区别在于，前者保留区间对象（并且可以查询这些区间）。

<!-- Media -->

<!-- figureText: $C\left( {t}_{1}\right)$ $C\left( {t}_{2}\right)$ $C\left( {t}_{3}\right)$ $C\left( {t}_{4}\right)$ $C\left( {t}_{5}\right)$ ${t}_{5}$ ${t}_{2}$ ${t}_{3}$ -->

<img src="https://cdn.noedgeai.com/0195c90d-0a33-750f-877f-c1cec48583bb_6.jpg?x=349&y=232&w=924&h=371&r=0"/>

Figure 3. A conceptual view of a bitemporal database. The t-axis (v-axis) corresponds to transaction (valid) times. Only the valid-time interval is shown from each interval-object. At transaction time ${t}_{1}$ the database recorded that interval-object ${I}_{1}$ is added on collection $C\left( {t}_{1}\right)$ . At ${t}_{5}$ the valid-time interval of object ${I}_{1}$ is modified to a new length.

图3. 双时态数据库的概念视图。t轴（v轴）对应于事务（有效）时间。每个区间对象仅显示其有效时间间隔。在事务时间${t}_{1}$，数据库记录到区间对象${I}_{1}$被添加到集合$C\left( {t}_{1}\right)$中。在${t}_{5}$，对象${I}_{1}$的有效时间间隔被修改为新的长度。

<!-- Media -->

Most of the methods directly support a single time-dimension. We categorize methods that take advantage of increasing time-ordered changes as transaction-time access methods, since this is the main characteristic of transaction-time. (The bulk of this paper deals with transaction-time methods.) Few approaches deal with valid-time access methods, and even fewer with the bitemporal methods category (methods that support both time dimensions on the same index).

大多数方法直接支持单一时间维度。我们将利用按时间顺序递增变化的方法归类为事务时间访问方法，因为这是事务时间的主要特征。（本文主要讨论事务时间方法。）很少有方法处理有效时间访问方法，处理双时态方法类别（在同一索引上支持两个时间维度的方法）的方法更少。

## 3. ITEMS FOR COMPARISON

## 3. 比较项目

This section elaborates on the items used in comparing the various access methods. We start with the kinds of queries examined and proceed with other criteria.

本节详细阐述了用于比较各种访问方法的项目。我们从所研究的查询类型开始，然后讨论其他标准。

### 3.1 Queries

### 3.1 查询

From a query perspective, valid-time and a transaction-time databases are simply collections of intervals. Figures $1,2\left( a\right)$ ,and $2\left( b\right)$ differ on how these intervals were created (which is important to the access method's update and space performance) and their meaning (which is important to the application). Hence, for single-time databases, (valid or transaction) queries are of similar form. First, we discuss queries in the transaction-time domain,i.e.,interval $T$ below corresponds to a transaction-time interval and "history" is on the transaction-time axis. The queries can be categorized into the following classes:

从查询的角度来看，有效时间数据库和事务时间数据库只是区间的集合。图$1,2\left( a\right)$和$2\left( b\right)$的不同之处在于这些区间的创建方式（这对访问方法的更新和空间性能很重要）及其含义（这对应用程序很重要）。因此，对于单时间数据库（有效时间或事务时间），查询的形式相似。首先，我们讨论事务时间域中的查询，即下面的区间$T$对应一个事务时间区间，“历史”位于事务时间轴上。查询可以分为以下几类：

(I) Given a contiguous interval $T$ ,find all objects alive during this interval.

(I) 给定一个连续区间$T$，找出在该区间内存在的所有对象。

(II) Given a key range and a contiguous time interval $T$ ,find the objects with keys in the given range that are alive during interval $T$ .

(II) 给定一个键范围和一个连续时间区间$T$，找出键在给定范围内且在区间$T$内存在的对象。

(III) Given a key range, find the history of the objects in this range.

(III) 给定一个键范围，找出该范围内对象的历史记录。

A special case of class (I) occurs when interval $T$ is reduced to a single transaction time instant $t$ . This query is termed the transaction pure-timeslice. In the company employee example, this query is "find all employees working at the company at time $t$ ." It is usually the case that an access method that efficiently solves the timeslice query is also efficient for the more general interval query; so we consider the timeslice query as a good representative of class (I) queries.

当区间$T$缩减为单个事务时间点$t$时，会出现类别 (I) 的一种特殊情况。这种查询称为事务纯时间片查询。在公司员工的示例中，此查询为“找出在时间$t$在公司工作的所有员工”。通常情况下，能有效解决时间片查询的访问方法对于更一般的区间查询也很有效；因此，我们认为时间片查询是类别 (I) 查询的一个很好的代表。

Similarly for class (II). Special cases include combinations where the key range, and/or the transaction time interval, contain a single key and a single time instant, respectively. For simplicity, we consider the representative case where the time interval is reduced to a single transaction time instant; this is the transaction range-timeslice query ("find the employees working at the company at time $t$ and whose ssn belongs in range $K$ ").

类别 (II) 同理。特殊情况包括键范围和/或事务时间区间分别包含单个键和单个时间点的组合。为简单起见，我们考虑时间区间缩减为单个事务时间点的代表性情况；这就是事务范围 - 时间片查询（“找出在时间$t$在公司工作且社保号属于范围$K$的员工”）。

From class (III), we chose the special case where the key range is reduced to a single key, as in: "find the salary history of employee with ssn $k$ ." This is the transaction pure-key query. If employee $k$ ever existed,the answer would be the history of salaries of that employee, else the answer is empty. In some methods, an instance of an employee object must be provided in the query and its previous salary history found (this is because these methods need to include a time predicate in their search). This special pure-key query (termed the pure-key with time predicate) is of the form: "find the salary history of employee $k$ who existed at time $t$ ."

从类别 (III) 中，我们选择键范围缩减为单个键的特殊情况，例如：“找出社保号为$k$的员工的薪资历史记录”。这就是事务纯键查询。如果员工$k$曾经存在过，答案将是该员工的薪资历史记录，否则答案为空。在某些方法中，查询中必须提供员工对象的实例，并找出其先前的薪资历史记录（这是因为这些方法需要在搜索中包含时间谓词）。这种特殊的纯键查询（称为带时间谓词的纯键查询）的形式为：“找出在时间$t$存在的员工$k$的薪资历史记录”。

The range and pure timeslices and pure key with time predicate correspond to "range" queries in Rivest's categorization [Rivest 1976], since being "alive" corresponds to two range queries, namely start.time $\leq  t$ and end.time $>$ $t$ . The pure-key without time predicate is an example of the "exact-match" query, as all objects with the given key should be retrieved.

范围和纯时间片以及带时间谓词的纯键查询对应于里维斯特分类法[Rivest 1976]中的“范围”查询，因为“存在”对应于两个范围查询，即开始时间$\leq  t$和结束时间$>$ $t$。不带时间谓词的纯键查询是“精确匹配”查询的一个示例，因为应检索具有给定键的所有对象。

When no key range is specified, query class (I) can be thought as a special case of class (II), and class (III) a special case of (II) when no interval is specified (rather, all times in history are of interest). As some of the proposed methods are better suited for answering queries from a particular class, we discuss all three classes separately. We indicate when an access method, as originally presented, does not address queries from a given class, but feel that such queries could be addressed with a slight modification that does not affect the method's behavior.

当未指定键范围时，查询类别 (I) 可以被视为类别 (II) 的特殊情况，而当未指定区间时（确切地说，历史上的所有时间都相关），类别 (III) 是类别 (II) 的特殊情况。由于一些提出的方法更适合回答特定类别的查询，我们将分别讨论这三个类别。我们会指出某个访问方法在最初提出时未处理给定类别的查询，但认为可以通过不影响该方法行为的轻微修改来处理此类查询。

Similarly, we can define the valid-time pure-timeslice for valid-time databases ("find all contracts valid at time $v$ "),valid-time range-timeslice ("find all contracts with numbers in range $K$ and which are valid at $v$ "),etc. A bitemporal database enables queries in both time dimensions: "find all contracts that were valid on $v =$ January 1,1994,as recorded in the database at transaction time $t =$ May 1,1993." From all contracts in the collection $C\left( t\right)$ for $t =$ May 1, 1993, the query retrieves only the contracts that would be valid on Jan. 1, 1994.

同样，我们可以为有效时间数据库定义有效时间纯时间片（“查找在时间 $v$ 有效的所有合同”）、有效时间范围时间片（“查找编号在范围 $K$ 内且在 $v$ 有效的所有合同”）等。双时态数据库支持在两个时间维度上进行查询：“查找在事务时间 $t =$ 1993 年 5 月 1 日记录在数据库中，在 $v =$ 1994 年 1 月 1 日有效的所有合同”。对于集合 $C\left( t\right)$ 中 $t =$ 1993 年 5 月 1 日的所有合同，该查询仅检索在 1994 年 1 月 1 日有效的合同。

The selection of the above query classes is definitely not complete, but contains basic, nontrivial queries. In particular, classes (I) and (II) relate to intersection-based queries, i.e., the answer consists of objects whose interval contains some query time point or in general intersects a query interval. Depending on the application, other queries may be of importance. For example, find all objects with intervals before or after a query time point/interval, or all objects with intervals contained in a given interval [Bozkaya and Ozsoyoglu 1995; Nascimento et al. 1996], etc.

上述查询类别的选择肯定并不完整，但包含了基本的、重要的查询。特别是，类别 (I) 和 (II) 与基于交集的查询相关，即答案由其时间间隔包含某个查询时间点或通常与查询时间间隔相交的对象组成。根据应用场景，其他查询可能也很重要。例如，查找所有时间间隔在查询时间点/时间间隔之前或之后的对象，或者所有时间间隔包含在给定时间间隔内的对象 [Bozkaya 和 Ozsoyoglu 1995；Nascimento 等人 1996] 等。

A three-entry notation, namely key/ valid/ transaction [Tsotras et al. 1998], to distinguish among the various temporal queries, will be used alternatively. This notation specifies which object attributes are involved in the query and in what way. Each entry is described as a "point," "range," " *," or "-". A "point" for the key entry means that the user has specified a single value to match the object key; a "point" for the valid or transaction entry implies that a single time instant is specified for the valid or transaction-time domain. A "range" indicates a specified range of object key values for the key entry, or an interval for the valid/transaction entries. A "*" means that any value is accepted in this entry, while "-" means that the entry is not applicable for this query. For example, "*/-/point" denotes the transaction pure-timeslice query, "range/point/-" is the valid range timeslice query, and "point/-/*" is the transaction pure-key query. In a bitemporal environment, the query "find all the company contracts that were valid on $v =$ January 1,1994, as recorded in the database during transaction time interval $T$ : May 1- May 20, 1993" is an example of a "*/ point/range" query. As presented, the three-entry notation deals with intersection queries, but can easily be extended through the addition of extra entry descriptions to accommodate before/after and other kinds of temporal queries.

将交替使用一种三项表示法，即键/有效/事务 [Tsotras 等人 1998]，来区分各种时态查询。这种表示法指定了查询中涉及哪些对象属性以及以何种方式涉及。每个项被描述为“点”、“范围”、“*”或“ - ”。键项的“点”表示用户指定了一个单一值来匹配对象键；有效或事务项的“点”意味着为有效或事务时间域指定了一个单一时间点。“范围”表示键项指定的对象键值范围，或者有效/事务项的时间间隔。“*”表示该项接受任何值，而“ - ”表示该项不适用于此查询。例如，“*/ - /点”表示事务纯时间片查询，“范围/点/ - ”是有效范围时间片查询，“点/ - /*”是事务纯键查询。在双时态环境中，查询“查找在事务时间间隔 $T$：1993 年 5 月 1 日 - 5 月 20 日记录在数据库中，在 $v =$ 1994 年 1 月 1 日有效的所有公司合同”是一个“*/点/范围”查询的示例。如前所述，三项表示法处理交集查询，但可以通过添加额外的项描述轻松扩展，以适应之前/之后和其他类型的时态查询。

### 3.2 Access Method Costs

### 3.2 访问方法成本

The performance of an access method is characterized by three costs: (1) the storage space to physically store the data records and the structures of the access method, (2) the update processing time (the time to update the method's data structures as a result of a change), and (3) the query time for each of the basic queries.

一种访问方法的性能由三种成本来表征：(1) 物理存储数据记录和访问方法结构所需的存储空间；(2) 更新处理时间（由于数据更改而更新该方法的数据结构所需的时间）；(3) 每个基本查询的查询时间。

An access method has two modes of operation: in the Update mode, data is inserted, altered, or deleted while in the Query mode queries are specified and answered using the access method. For a transaction-time access method, the input for an update consists of a time instant $t$ and all the changes that occurred in the data in that instant. A change is further specified by the unique key of the object it affects and the kind of change (addition, deletion, or attribute modification). The access method's data structure(s) will then be updated to include the new change. Input to a bitemporal access method where the time of the change is specified along with the changes and the affected interval-object(s) is performed similarly. The input to a valid-time access method simply contains the changes and the interval-object(s) affected.

一种访问方法有两种操作模式：在更新模式下，进行数据的插入、修改或删除；在查询模式下，使用该访问方法指定并回答查询。对于事务时间访问方法，更新的输入包括一个时间点 $t$ 以及在该时间点数据中发生的所有更改。更改进一步由其影响的对象的唯一键和更改类型（添加、删除或属性修改）来指定。然后将更新访问方法的数据结构以包含新的更改。对于双时态访问方法，在指定更改的同时指定更改时间以及受影响的间隔对象，其操作方式类似。有效时间访问方法的输入仅包含更改和受影响的间隔对象。

For a transaction or a bitemporal method,the space is a function of $n$ ,the total number of changes in the evolution,i.e., $n$ is the summation of insertions, deletions, and modification updates. If there are 1,000 updates to a database with only one record, $n$ is 1,000. If there are 1,000 insertions to an empty database and no deletions or value modifications, $n$ is also 1,000 . Similarly, for 1,000 insertions followed by 1,000 deletions, $n$ is 2,000 . Note that $n$ corresponds to the minimal information needed for storing the evolution's past. We assume that the total number of transaction instants is also $O\left( n\right)$ . This is a natural assumption, since every real computer system can process a possibly large but limited number of updates per transaction instant.

对于事务或双时态方法，所需空间是 $n$ 的函数，$n$ 表示演化过程中的总更改次数，即 $n$ 是插入、删除和修改更新操作次数的总和。如果对仅包含一条记录的数据库进行 1000 次更新，那么 $n$ 为 1000。如果向一个空数据库插入 1000 条记录，且没有删除或值修改操作，$n$ 同样为 1000。类似地，先进行 1000 次插入操作，再进行 1000 次删除操作，$n$ 则为 2000。请注意，$n$ 对应于存储演化历史所需的最少信息。我们假设事务时刻的总数也是 $O\left( n\right)$。这是一个合理的假设，因为每个实际的计算机系统在每个事务时刻所能处理的更新次数虽然可能很多，但仍然是有限的。

In a valid-time method, space is a function of $l$ ,the number of interval-objects currently stored in the method, i.e., the size of the collection. For example,in both Figures 2(a) and 2(b), $l$ is seven.

在有效时间方法中，所需空间是 $l$ 的函数，$l$ 表示该方法当前存储的区间对象的数量，即集合的大小。例如，在图 2(a) 和图 2(b) 中，$l$ 均为 7。

A method's query time is a function of the answer size $a$ . We use $a$ to denote the answer size of a query in general.

一种方法的查询时间是答案大小 $a$ 的函数。我们通常用 $a$ 来表示查询的答案大小。

Since temporal data can be large (especially in transaction and bitemporal databases), a good solution should use space efficiently. A method with fast update processing can be utilized even with a quickly changing real-world application. In addition, fast query times will greatly facilitate the use of temporal data.

由于时态数据可能非常庞大（尤其是在事务和双时态数据库中），因此一个好的解决方案应该能够高效地利用空间。一种更新处理速度快的方法，即使在现实世界应用快速变化的情况下也能发挥作用。此外，快速的查询时间将极大地促进时态数据的使用。

The basic queries that we examine can be considered as special cases of classical problems in computational geometry, for which efficient in-core (main memory) solutions have been provided [Chiang and Tamassia 1992]. It should be mentioned that general computational geometry problems support physical deletions of intervals. Hence, they are more closely related to the valid-time database environment. The valid pure-time-slice query ("*/point/-") is a special case of the dynamic interval management problem. The best in-core bounds for dynamic interval management are given by the priority-search tree data structure in Mc-Creight [1985], yielding $O\left( l\right)$ space, $O\left( {\log l}\right)$ update processing per change, and $O\left( {\log l + a}\right)$ query time (all logarithms are base-2). Here $l$ is the number of intervals in the structure when the query/update is performed. The range-timeslice query is a special case of the orthogonal segment intersection problem, for which a solution using $O\left( {l\log l}\right)$ space, $O\left( {\left( {\log l}\right) \log \log l}\right)$ update processing,and $O\left( {\left( {\log l}\right) \log \log l + a}\right)$ query time is provided in Mehlhorn[1984]; another solution that uses a combination of the priority-search tree [McCreight 1985] and the interval tree [Edelsbrunner 1983] yields $O\left( l\right)$ space, $O\left( {\log l}\right)$ update processing,and $O\left( {{\log }^{2}l + a}\right)$ query time.

我们所研究的基本查询可以被视为计算几何中经典问题的特殊情况，针对这些经典问题，已经有了高效的内存（主存）解决方案 [Chiang 和 Tamassia 1992]。需要提及的是，一般的计算几何问题支持对区间进行物理删除。因此，它们与有效时间数据库环境的关联更为紧密。有效的纯时间切片查询（“*/点/-”）是动态区间管理问题的一个特殊情况。动态区间管理的最佳内存边界由 McCreight [1985] 提出的优先搜索树数据结构给出，其空间复杂度为 $O\left( l\right)$，每次更改的更新处理复杂度为 $O\left( {\log l}\right)$，查询时间复杂度为 $O\left( {\log l + a}\right)$（所有对数均以 2 为底）。这里的 $l$ 是执行查询/更新操作时结构中的区间数量。范围时间切片查询是正交线段相交问题的一个特殊情况，Mehlhorn [1984] 针对该问题提供了一种使用 $O\left( {l\log l}\right)$ 空间、$O\left( {\left( {\log l}\right) \log \log l}\right)$ 更新处理复杂度和 $O\left( {\left( {\log l}\right) \log \log l + a}\right)$ 查询时间复杂度的解决方案；另一种结合了优先搜索树 [McCreight 1985] 和区间树 [Edelsbrunner 1983] 的解决方案，其空间复杂度为 $O\left( l\right)$，更新处理复杂度为 $O\left( {\log l}\right)$，查询时间复杂度为 $O\left( {{\log }^{2}l + a}\right)$。

The problems addressed by transaction or bitemporal methods are related to work on persistent data structures [Driscoll et al. 1989]. In particular, Driscoll et al. [1989] shows how to take an in-core "ephemeral data structure" (meaning that past states are erased when updates are made) and convert it to a "persistent data structure" (where past states are maintained). A "fully persistent" data structure allows updates to all past states. A "partially persistent" data structure allows updates only to the most recent state. Due to the properties of transaction time evolution, transaction and bitemporal access methods can be thought of as disk extensions of partially persistent data structures.

事务或双时态方法所解决的问题与持久数据结构方面的研究相关 [Driscoll 等人 1989]。具体而言，Driscoll 等人 [1989] 展示了如何将一个内存中的“临时数据结构”（即更新时会删除过去状态的数据结构）转换为“持久数据结构”（即会保留过去状态的数据结构）。“完全持久”的数据结构允许对所有过去状态进行更新。“部分持久”的数据结构仅允许对最近的状态进行更新。由于事务时间演化的特性，事务和双时态访问方法可以被视为部分持久数据结构在磁盘上的扩展。

### 3.3 Index Pagination and Data Clustering

### 3.3 索引分页与数据聚类

In a database environment the cost of a computation is not based on how many main memory slots are accessed or how many comparisons are made (as the case with in-core algorithms), but instead on how many pages are transferred between main and secondary memory. In our comparison this is very crucial, as the bulk of data is stored in secondary storage media. So it is natural to use an I/O complexity cost [Kanel-lakis et al. 1993] that measures the number of disk accesses for updating and answering queries. The need to use I/O complexity for secondary storage structures is also recognized in the "theory of indexability" [Hellerstein et al. 1997]. Index pagination and data clustering are two important aspects when considering the I/O complexity of query time. How well the index nodes of a method are paginated is dealt with by the process of index pagination. Since the index is used as a means to search for and update data, its pagination greatly affects the performance of the method. For example,a ${\mathrm{B}}^{ + }$ -tree is a well-paginated index, since it requires $O\left( {{\log }_{B}r}\right)$ page accesses for searching or updating $r$ objects,using pages of size $B$ . The reader should be careful with the notation: ${\log }_{B}r$ is itself an $O\left( {{\log }_{2}r}\right)$ function only if $B$ is considered a constant. For an $\mathrm{I}/\mathrm{O}$ environment, $B$ is another problem variable. Thus, ${\log }_{B}r$ represents a ${\log }_{2}B$ speedup over ${\log }_{2}r$ , which,for $\mathrm{I}/\mathrm{O}$ complexity,is a great improvement. Transferring a page takes about ${10}\mathrm{{msec}}$ on the fastest disk drives; in contrast, comparing two integers in main memory takes about 5 nsec. Accessing pages also uses CPU time. The CPU cost of reading a page from the disk is about 2,000 instructions [Gray and Reuter 1993].

在数据库环境中，一次计算的成本并非基于访问了多少个主存槽或进行了多少次比较（如内存内算法的情况），而是基于主存和辅存之间传输了多少页数据。在我们的比较中，这一点非常关键，因为大部分数据都存储在二级存储介质中。因此，自然会采用一种I/O复杂度成本[卡内拉基斯等人，1993年]来衡量更新和回答查询时的磁盘访问次数。在“可索引性理论”[赫勒斯坦等人，1997年]中也认识到了对二级存储结构使用I/O复杂度的必要性。在考虑查询时间的I/O复杂度时，索引分页和数据聚类是两个重要方面。索引分页过程处理的是一种方法的索引节点分页情况。由于索引用于搜索和更新数据，其分页情况会极大地影响该方法的性能。例如，一棵${\mathrm{B}}^{ + }$ -树是一种分页良好的索引，因为使用大小为$B$的页面搜索或更新$r$个对象时，它需要$O\left( {{\log }_{B}r}\right)$次页面访问。读者应注意符号表示：只有当$B$被视为常数时，${\log }_{B}r$本身才是一个$O\left( {{\log }_{2}r}\right)$函数。对于一个$\mathrm{I}/\mathrm{O}$环境，$B$是另一个问题变量。因此，${\log }_{B}r$表示相对于${\log }_{2}r$有${\log }_{2}B$的加速，对于$\mathrm{I}/\mathrm{O}$复杂度而言，这是一个很大的改进。在最快的磁盘驱动器上传输一页数据大约需要${10}\mathrm{{msec}}$；相比之下，在主存中比较两个整数大约需要5纳秒。访问页面也会消耗CPU时间。从磁盘读取一页数据的CPU成本大约是2000条指令[格雷和罗伊特，1993年]。

Data clustering can also substantially improve the performance of an access method. If data records that are "logically" related for a given query can also be stored physically close, then the query is optimized as fewer pages are accessed. Consider for example an access method that can cluster data in such a way that answering the transaction pure-timeslice query takes $O\left( {{\log }_{B}n + a/B}\right)$ page accesses. This method is more I/O efficient than another method that solves the same query in $O\left( {{\log }_{B}n + a}\right)$ page accesses. Both methods use a well-paginated index (which corresponds to the logarithmic part of the query). However, in the second method, each data record that belongs to the answer set may be stored on a separate page, thus requiring a much larger number of page accesses for solving the query.

数据聚类也可以显著提高访问方法的性能。如果对于给定查询“逻辑上”相关的数据记录也能在物理上存储得很近，那么由于访问的页面更少，查询就得到了优化。例如，考虑一种访问方法，它可以对数据进行聚类，使得回答事务纯时间片查询需要$O\left( {{\log }_{B}n + a/B}\right)$次页面访问。这种方法比另一种解决相同查询需要$O\left( {{\log }_{B}n + a}\right)$次页面访问的方法在I/O方面更高效。两种方法都使用了分页良好的索引（对应于查询的对数部分）。然而，在第二种方法中，属于答案集的每个数据记录可能存储在单独的页面上，因此解决查询需要访问的页面数量要多得多。

Data can be clustered by time dimension only, where data records "alive" for the same time periods are collocated, or by both time and key range, or by key range only. Note that a clustering strategy that optimizes a given class of queries may not work for another query class; for example, a good clustering strategy for pure-key queries stores all the versions of a particular key in the same page; however, this strategy does not work for pure-timeslice queries because the clustering objective is different.

数据可以仅按时间维度进行聚类，即将在相同时间段内“存活”的数据记录放在一起，也可以按时间和键范围进行聚类，或者仅按键范围进行聚类。请注意，一种针对某类查询进行优化的聚类策略可能不适用于另一类查询；例如，一种适用于纯键查询的良好聚类策略会将某个特定键的所有版本存储在同一页面中；然而，这种策略不适用于纯时间片查询，因为聚类目标不同。

Clustering is in general more difficult to maintain in a valid-time access method because of its dynamic behavior. The answer to a valid-time query depends on the collection of interval-objects currently contained in the access method; this collection changes as valid-time updates are applied. Even though some good clustering may have been achieved for some collection, it may not be as efficient for the next collection produced after a number of valid-time updates. In contrast, in transaction or bitemporal access methods, the past is not changed, so an efficient clustering can be retained more easily, despite updates.

由于有效时间访问方法的动态特性，通常在其中维护聚类更加困难。有效时间查询的答案取决于访问方法当前包含的区间对象集合；随着有效时间更新的应用，这个集合会发生变化。即使针对某个集合实现了良好的聚类，但对于经过多次有效时间更新后产生的下一个集合，它可能就没有那么高效了。相比之下，在事务或双时态访问方法中，过去的数据不会改变，因此尽管有更新，也更容易保留高效的聚类。

Any method that clusters data (a primary index) and uses,say, $O\left( {{\log }_{B}n + }\right.$ $a/B$ ) pages for queries can also be used (less efficiently) as a secondary index by replacing the data records with pointers to pages containing data records, thus using $O\left( {{\log }_{B}n + a}\right)$ pages for queries. The distinction between methods used as primary indexes and methods used as secondary indexes is one of efficiency, not of algorithmic properties.

任何对数据（主索引）进行聚类并使用（例如）$O\left( {{\log }_{B}n + }\right.$ $a/B$ 页进行查询的方法，也可以（效率较低地）用作辅助索引，方法是将数据记录替换为指向包含数据记录的页面的指针，从而使用 $O\left( {{\log }_{B}n + a}\right)$ 页进行查询。用作主索引的方法和用作辅助索引的方法之间的区别在于效率，而非算法特性。

We use the term "primary index" to mean that the index controls the physical placement of data only. For example, a primary ${\mathrm{B}}^{ + }$ -tree has data in the leaves. A secondary ${\mathrm{B}}^{ + }$ -tree has only keys and references to data pages (pointers) in the leaves. Primary indexes need not be on the primary keys of relations. Many of the methods do expect a unique nontime varying key for each record; we do not attempt to discuss how these methods might be modified to cluster records by nonunique keys.

我们使用“主索引”这一术语仅表示该索引控制数据的物理存储位置。例如，主 ${\mathrm{B}}^{ + }$ 树的叶子节点中存储着数据。辅助 ${\mathrm{B}}^{ + }$ 树的叶子节点中仅包含键和对数据页的引用（指针）。主索引不必基于关系的主键。许多方法确实期望每条记录有一个唯一的、不随时间变化的键；我们不打算讨论如何修改这些方法以通过非唯一键对记录进行聚类。

### 3.4 Migration of Past Data to Another Location

### 3.4 过往数据迁移至其他位置

Methods that support transaction time maintain all their past states, a property that can easily result in excessive amounts of data (even for methods that support transaction time in the most space-efficient way). In comparing such methods, it is natural to introduce two other comparison considerations: (a) whether or not past data can be separated from the current data, so that the smaller collection of current data can be accessed more efficiently, and (b) whether data is appended sequentially to the method and never changed, so that write-once read-many (WORM) devices could be used.

支持事务时间的方法会保留其所有过往状态，这一特性很容易导致数据量过大（即使是那些以最节省空间的方式支持事务时间的方法也是如此）。在比较此类方法时，很自然地会引入另外两个比较因素：(a) 过往数据是否可以与当前数据分离，以便更高效地访问较小的当前数据集；(b) 数据是否按顺序追加到该方法中且从不更改，从而可以使用一次写入多次读取（WORM）设备。

For WORMs, one must burn into the disk an entire page with a checksum (the error rate is high, so a very long error-correcting code must be appended to each page). Thus, once a page is written, it cannot be updated. Note that since WORM devices are themselves random access media, any access method that can use WORM devices can also be used with magnetic disks(only). There are no access methods restricted to the use of WORMs.

对于一次写入多次读取（WORM）设备，必须将带有校验和的整页数据写入磁盘（错误率很高，因此必须为每页追加一个很长的纠错码）。因此，一旦写入一页，就无法对其进行更新。请注意，由于 WORM 设备本身就是随机访问介质，任何可以使用 WORM 设备的访问方法也可以仅用于磁盘。没有仅限于使用 WORM 设备的访问方法。

### 3.5 Lower Bounds on I/O Complexity

### 3.5 I/O 复杂度的下界

We first establish a lower bound on the I/O complexity of basic transaction-time queries. The lower bound is obtained using a comparison-based model in a paginated environment, and applies to the transaction pure-timeslice ("*/-/ point"), range-timeslice ("range/-/ point"), and pure-key (with time predicate or a "point/-/range") query. Any method that attempts to solve such a query in linear $\left( {O\left( {n/B}\right) }\right)$ space needs at least $\Omega \left( {{\log }_{B}n + a/B}\right)$ I/Os to solve it.

我们首先确定基本事务时间查询的 I/O 复杂度的下界。该下界是在分页环境中使用基于比较的模型得出的，适用于事务纯时间片（“*/-/ 点”）、范围时间片（“范围/-/ 点”）和纯键（带有时间谓词或“点/-/ 范围”）查询。任何试图在线性 $\left( {O\left( {n/B}\right) }\right)$ 空间中解决此类查询的方法至少需要 $\Omega \left( {{\log }_{B}n + a/B}\right)$ 次 I/O 操作才能解决该查询。

Since $a$ corresponds to the query answer size, to provide the answer, no method can do better than $O\left( {a/B}\right) \mathrm{I}/\mathrm{{Os}}$ ; $a/B$ is the minimal number of pages to store this answer. Note that the lower bound discussion assumes that a query may ask for any time instant in the set of possible time instants. That is, all time instants (whether recent or past) have the same probability of being in the query predicate. This fact separates this discussion from cases where queries have special properties (for example, if most queries ask for the most recent times). While this does not affect the answer part $\left( {a/B}\right)$ of the lower bound, it does affect the logarithmic search part $\left( {{\log }_{B}n}\right)$ . We could probably locate the time of interest faster if we knew that this instant is among the most recent times. Under the above equal query probability assumption, we proceed with the justification for the logarithmic part of the lower bound. Since the range-timeslice query is more general than the pure-timeslice query, we first show that the pure-timeslice problem is reduced to the "predecessor" problem for which a lower bound is then established [Tsotras and Kangelaris 1995]. A similar reduction can be proved for the pure-key query with time predicate.

由于 $a$ 对应于查询答案的大小，为了提供答案，没有任何方法能比 $O\left( {a/B}\right) \mathrm{I}/\mathrm{{Os}}$ 做得更好；$a/B$ 是存储该答案所需的最少页数。请注意，下界讨论假设查询可能会要求获取可能的时间点集合中的任何时间点。也就是说，所有时间点（无论是近期的还是过去的）在查询谓词中出现的概率相同。这一事实将此讨论与查询具有特殊属性的情况区分开来（例如，如果大多数查询要求获取最近的时间）。虽然这不会影响下界的答案部分 $\left( {a/B}\right)$，但会影响对数搜索部分 $\left( {{\log }_{B}n}\right)$。如果我们知道感兴趣的时间点是最近的时间之一，我们可能会更快地定位该时间。在上述查询概率相等的假设下，我们继续对下界的对数部分进行论证。由于范围时间片查询比纯时间片查询更具一般性，我们首先证明纯时间片问题可以简化为“前驱”问题，然后为该问题建立下界 [Tsotras 和 Kangelaris 1995]。对于带有时间谓词的纯键查询，也可以证明类似的简化。

The predecessor problem is defined as follows: Given an ordered set $P$ of $N$ distinct items,and an item $k$ ,find the largest member of set $P$ that is less than or equal to $k$ . For the reduction of the pure-timeslice problem, assume that set $P$ contains integers ${t}_{1} < {t}_{2} < \ldots  < {t}_{N}$ and consider the following real-world evolution: at time ${t}_{1}$ ,a single real-world object with name (oid) ${t}_{1}$ is created and lives until just before time ${t}_{2}$ ,i.e.,the lifespan of object ${t}_{1}$ is $\left\lbrack  {{t}_{1},{t}_{2}}\right)$ . Then, real-world object ${t}_{2}$ is born at ${t}_{2}$ and lives for the interval $\left\lbrack  {{t}_{2},{t}_{3}}\right)$ ,and so on. So at any time instant ${t}_{i}$ the state of the real-world system is a single object with name ${t}_{i}$ . Hence the $N$ integers correspond to $n = {2N}$ changes in the above evolution. Consequently, finding the whole timeslice at time $t$ reduces to finding the largest element in set $P$ that is less or equal to $t$ ,i.e.,the predecessor of $t$ inside $P$ .

前驱问题的定义如下：给定一个由 $N$ 个不同元素组成的有序集合 $P$，以及一个元素 $k$，找出集合 $P$ 中小于或等于 $k$ 的最大元素。为了简化纯时间片问题，假设集合 $P$ 包含整数 ${t}_{1} < {t}_{2} < \ldots  < {t}_{N}$，并考虑以下现实世界的演变：在时间 ${t}_{1}$，创建了一个名为（对象标识符）${t}_{1}$ 的单个现实世界对象，该对象一直存在到时间 ${t}_{2}$ 之前，即对象 ${t}_{1}$ 的生命周期为 $\left\lbrack  {{t}_{1},{t}_{2}}\right)$。然后，现实世界对象 ${t}_{2}$ 在 ${t}_{2}$ 时刻诞生，并在区间 $\left\lbrack  {{t}_{2},{t}_{3}}\right)$ 内存在，依此类推。因此，在任何时刻 ${t}_{i}$，现实世界系统的状态是一个名为 ${t}_{i}$ 的单个对象。因此，$N$ 个整数对应于上述演变中的 $n = {2N}$ 次变化。因此，找到时间 $t$ 的整个时间片就简化为找到集合 $P$ 中小于或等于 $t$ 的最大元素，即 $t$ 在集合 $P$ 中的前驱。

We show that in the comparison-based model and in a paginated environment, the predecessor problem needs at least $\Omega \left( {{\log }_{B}N}\right)$ I/Os. The assumption is that each page contains $B$ items,and there is no charge for a comparisons within a page. Our argument is based on a decision tree proof. Let the first page be read and assume that the items read within that page are sorted (sorting inside one page is free of $\mathrm{I}/\mathrm{{Os}}$ ). By exploring the entire page using comparisons,we can only get $B + 1$ different answers concerning item $k$ . These correspond to the $B + 1$ intervals created by $B$ items. No additional information can be retrieved. Then, a new page is retrieved that is based on the outcome of the previous comparisons on the first page, i.e., a different page is read every $B + 1$ outcomes. In order to determine the predecessor of $k$ ,the decision tree must have $N$ leaves (since there are $N$ possible predecessors). As a result, the height of the tree must be ${\log }_{B}N$ . Thus any algorithm that solves the paginated version of the predecessor problem in the comparison model needs at least $\Omega \left( {{\log }_{B}N}\right)$ I/Os.

我们证明，在基于比较的模型和分页环境中，前驱问题至少需要 $\Omega \left( {{\log }_{B}N}\right)$ 次输入/输出（I/O）操作。假设每页包含 $B$ 个元素，并且在页内进行比较不产生费用。我们的论证基于决策树证明。假设读取第一页，并且该页内读取的元素已排序（在一页内排序不产生 $\mathrm{I}/\mathrm{{Os}}$ 费用）。通过使用比较操作遍历整个页面，我们关于元素 $k$ 只能得到 $B + 1$ 种不同的答案。这些答案对应于由 $B$ 个元素创建的 $B + 1$ 个区间。无法获取额外信息。然后，根据第一页上先前比较的结果检索新的一页，即每 $B + 1$ 种结果读取不同的一页。为了确定 $k$ 的前驱，决策树必须有 $N$ 个叶子节点（因为有 $N$ 种可能的前驱）。因此，树的高度必须为 ${\log }_{B}N$。因此，任何在比较模型中解决分页版本前驱问题的算法至少需要 $\Omega \left( {{\log }_{B}N}\right)$ 次 I/O 操作。

If there were a faster than $O\left( {{\log }_{B}n}\right.$ $+ a/B)$ method for the pure-timeslice problem using $O\left( {n/B}\right)$ space,then we would have invented a method that solves the above predecessor problem in less than $O\left( {{\log }_{B}N}\right)$ I/Os.

如果存在一种使用 $O\left( {n/B}\right)$ 空间、比 $O\left( {{\log }_{B}n}\right.$ $+ a/B)$ 更快的方法来解决纯时间片问题，那么我们就发明了一种在少于 $O\left( {{\log }_{B}N}\right)$ 次 I/O 操作内解决上述前驱问题的方法。

Observe that we have shown the lower bound for the query time of methods using linear space, irrespective of update processing. If the elements of set $P$ are given in order,one after the other, $O\left( 1\right)$ time (amortized) per element is needed in order to create an index on the set that would solve the predecessor problem in $O\left( {{\log }_{B}N}\right)$ I/Os (more accurately, since no deletions are needed, we only need a fully paginated, multilevel index that increases in one direction). If these elements are given out of order, then $O\left( {{\log }_{B}N}\right)$ time is needed per insertion (B-tree index). In the transaction pure timeslice problem, ("*/-/point") time is always increasing and $O\left( 1\right)$ time for update processing per change is enough and clearly minimal. Thus we call a method $I/O$ optimal for the transaction pure-timeslice query if it achieves $O\left( {n/B}\right)$ space and $O\left( {{\log }_{B}n + a/B}\right)$ query time using constant updating.

请注意，我们已经展示了使用线性空间的方法在查询时间上的下界，而不考虑更新处理。如果集合 $P$ 的元素是按顺序逐个给出的，那么为了在该集合上创建一个能在 $O\left( {{\log }_{B}N}\right)$ 次输入/输出（I/O）操作内解决前驱问题的索引，每个元素需要 $O\left( 1\right)$ 的时间（均摊）（更准确地说，由于不需要进行删除操作，我们只需要一个完全分页的、单向增长的多级索引）。如果这些元素是无序给出的，那么每次插入操作需要 $O\left( {{\log }_{B}N}\right)$ 的时间（B 树索引）。在事务纯时间片问题中，（“*/-/点”）时间总是在增加，并且每次更改的更新处理时间为 $O\left( 1\right)$ 就足够了，而且显然是最小的。因此，如果一种方法在使用常量更新的情况下能达到 $O\left( {n/B}\right)$ 的空间和 $O\left( {{\log }_{B}n + a/B}\right)$ 的查询时间，我们就称其为事务纯时间片查询的 $I/O$ 最优方法。

Similarly, for the transaction range-timeslice problem ("range/-/point"), we call a method $I/O$ optimal if it achieves $O\left( {{\log }_{B}n + a/B}\right)$ query time, $O\left( {n/B}\right)$ space and $O\left( {{\log }_{B}m}\right)$ update processing per change. $m$ is the number of alive objects when the update takes place. Logarithmic processing is needed because the range-timeslice problem requires ordering keys by their values. Changes arrive in time order, but out of key order,and there are $m$ alive keys on the latest state from which an update has to choose.

类似地，对于事务范围 - 时间片问题（“范围/-/点”），如果一种方法能达到 $O\left( {{\log }_{B}n + a/B}\right)$ 的查询时间、$O\left( {n/B}\right)$ 的空间以及每次更改 $O\left( {{\log }_{B}m}\right)$ 的更新处理时间，我们就称其为 $I/O$ 最优方法。$m$ 是更新发生时存活对象的数量。需要对数级的处理时间，因为范围 - 时间片问题要求根据键的值对键进行排序。更改是按时间顺序到达的，但键是无序的，并且在最新状态中有 $m$ 个存活的键，更新必须从这些键中进行选择。

For the transaction pure-key with time predicate, the lower bound for query time is $\Omega \left( {{\log }_{B}n + a/B}\right)$ ,since the logarithmic part is needed to locate the time predicate in the past and $a/B\mathrm{I}/\mathrm{{Os}}$ are required to provide the answer in the output.

对于带有时间谓词的事务纯键查询，查询时间的下界是 $\Omega \left( {{\log }_{B}n + a/B}\right)$，因为需要对数级的部分来定位过去的时间谓词，并且需要 $a/B\mathrm{I}/\mathrm{{Os}}$ 来在输出中提供答案。

The same lower bound holds for bitemporal queries, since they are at least as complex as transaction queries. For example, consider the "*/point/point" query specified by a valid time $v$ and a transaction time $t$ . If the valid-time interval of each interval object extends from $- \infty$ to $\infty$ in the valid-time domain, finding all interval objects that at $t$ , where intersecting $v$ ,reduces to finding all interval-objects in collection $C\left( t\right)$ (since all of them would contain the valid instant $v$ ). However,this is the “*/-/point” query.

双时态查询也有相同的下界，因为它们至少和事务查询一样复杂。例如，考虑由有效时间 $v$ 和事务时间 $t$ 指定的“*/点/点”查询。如果每个区间对象的有效时间区间在有效时间域中从 $- \infty$ 延伸到 $\infty$，那么找到在 $t$ 时刻与 $v$ 相交的所有区间对象，就简化为在集合 $C\left( t\right)$ 中找到所有区间对象（因为它们都包含有效时刻 $v$）。然而，这就是“*/-/点”查询。

Since from a query perspective a valid and a transaction-time database are both collections of intervals, a similar lower bound applies to the corresponding valid-time queries (by replacing $n$ by $l$ ,the number of interval-objects in the collection). For example, any algorithm solving the "*/point/-" query in $O\left( {l/B}\right)$ space needs at least $\Omega \left( {{\log }_{B}l + a/B}\right)$ I/Os query time.

从查询的角度来看，有效时间数据库和事务时间数据库都是区间的集合，因此类似的下界也适用于相应的有效时间查询（通过将 $n$ 替换为 $l$，即集合中间隔对象的数量）。例如，任何在 $O\left( {l/B}\right)$ 空间内解决“*/点/-”查询的算法至少需要 $\Omega \left( {{\log }_{B}l + a/B}\right)$ 次输入/输出（I/O）查询时间。

## 4. EFFICIENT METHOD DESIGN FOR TRANSACTION/BITEMPORAL DATA

## 4. 事务/双时态数据的高效方法设计

Common to all methods that support the transaction time axis is the problem of how to efficiently store large amounts of data. We first consider the transaction pure-timeslice query and show why obvious solutions are not efficient. We also discuss the transaction pure-key and range-timeslice queries. Bitemporal queries follow. The problem of separating past from current data (and the use of WORM disks) is also examined.

所有支持事务时间轴的方法都面临着如何高效存储大量数据的问题。我们首先考虑事务纯时间片查询，并说明为什么显而易见的解决方案并不高效。我们还将讨论事务纯键查询和范围 - 时间片查询。接下来是双时态查询。我们还将研究将过去数据与当前数据分离的问题（以及一次写入多次读取（WORM）磁盘的使用）。

### 4.1 The Transaction Pure-Timeslice Query

### 4.1 事务纯时间片查询

There are two straightforward solutions to the transaction pure-timeslice query ("*/-/point") which, in our comparison, serve as two extreme cases; we denote them the "copy" and "log" approaches.

对于事务纯时间片查询（“*/-/点”）有两种直接的解决方案，在我们的比较中，它们代表了两种极端情况；我们将它们分别称为“复制”和“日志”方法。

The "copy" approach stores a copy of the transaction database state $s\left( t\right)$ (timeslice) for each transaction time that at least one change occurred. These copies are indexed by time $t$ . Access to a state $s\left( t\right)$ is performed by searching for time $t$ on a multilevel index on the time dimension. Since changes arrive in order, this multilevel index is clearly paginated. The closest time that is less or equal to $t$ is found with $O\left( {{\log }_{B}n}\right)$ page accesses. An additional $O\left( {a/B}\right)$ I/O time is needed to output the copy of the state,where $a$ denotes the number of "alive" objects in the accessed database state. The major disadvantage of the "copy" approach is with the space and update processing requirements. The space used can in the worst case be proportional to $O\left( {{n}^{2}/B}\right)$ . This happens if the evolution is mainly composed of "births" of new objects. The database state is thus enlarged continuously. If the size of the database remains relatively constant, due to deletions and insertions balancing out, and if there are $p$ records on average,the space used is $O\left( {{np}/B}\right)$ .

“复制”方法会为至少发生一次更改的每个事务时间存储事务数据库状态 $s\left( t\right)$（时间片）的副本。这些副本按时间 $t$ 进行索引。对状态 $s\left( t\right)$ 的访问是通过在时间维度的多级索引上搜索时间 $t$ 来实现的。由于更改是按顺序到来的，因此这个多级索引显然是分页的。通过 $O\left( {{\log }_{B}n}\right)$ 次页面访问可以找到小于或等于 $t$ 的最近时间。输出状态副本还需要额外的 $O\left( {a/B}\right)$ 次 I/O 时间，其中 $a$ 表示所访问数据库状态中“存活”对象的数量。“复制”方法的主要缺点在于空间和更新处理要求。在最坏的情况下，所使用的空间可能与 $O\left( {{n}^{2}/B}\right)$ 成正比。如果数据库的演变主要由新对象的“诞生”组成，就会出现这种情况。因此，数据库状态会不断增大。如果由于删除和插入相互平衡，数据库的大小保持相对恒定，并且平均有 $p$ 条记录，那么所使用的空间为 $O\left( {{np}/B}\right)$。

Update processing is $O\left( {n/B}\right)$ per change instant in a growing database and $O\left( {p/B}\right)$ per change instant in a nongrowing database, as a new copy of the database has to be stored at each change instant. The "copy" approach provides a minimal query time. However, since the information stored is much more than the actual changes, the space and update requirements suffer.

在不断增长的数据库中，每次更改瞬间的更新处理为 $O\left( {n/B}\right)$，在非增长的数据库中，每次更改瞬间的更新处理为 $O\left( {p/B}\right)$，因为在每个更改瞬间都必须存储数据库的新副本。“复制”方法提供了最短的查询时间。然而，由于存储的信息远远超过实际的更改，因此空间和更新要求较高。

A variant on the copy approach stores a list of record ADDRESSES that are "alive" each time at least one change occurs. The total amount of space used is smaller than if the records themselves were stored in each copy. However, the asymptotic space used is still $O\left( {{n}^{2}/B}\right)$ for growing databases and $O\left( {{np}/B}\right)$ for databases whose size does not increase significantly over time. This means most records have $O\left( n\right)$ references in the index. " $n$ " does not have to be very large before the index is several times the size of the record collection. In addition, by changing from a primary to a secondary unclustered structure, $O\left( a\right)$ ,not $O\left( {a/B}\right)$ ,pages must be accessed to output the copy of the a alive records (after the usual $O\left( {{\log }_{B}n}\right)$ accesses to find the correct list).

复制方法的一种变体存储了每次至少发生一次更改时“存活”的记录地址列表。与在每个副本中存储记录本身相比，所使用的总空间量更小。然而，对于不断增长的数据库，渐近空间仍然为 $O\left( {{n}^{2}/B}\right)$，对于随时间推移大小没有显著增加的数据库，渐近空间为 $O\left( {{np}/B}\right)$。这意味着大多数记录在索引中有 $O\left( n\right)$ 个引用。在索引大小达到记录集合大小的数倍之前，“ $n$ ”不必非常大。此外，通过从主结构更改为二级非聚集结构，在输出存活记录的副本时（在通常的 $O\left( {{\log }_{B}n}\right)$ 次访问以找到正确的列表之后），必须访问 $O\left( a\right)$ 个页面，而不是 $O\left( {a/B}\right)$ 个页面。

In the remainder of this paper, we will not consider any secondary indexes. In order to make a fair comparison, indexes that are described as secondary by their authors will be treated as if they were primary indexes. Secondary indexes never cluster data in disk pages, and thus always lose out in query time. Recall that by "primary" index we mean only an index that dictates the physical location of records, not an index on "primary key." Secondary indexes can only cluster references to records, not the records themselves.

在本文的其余部分，我们将不考虑任何二级索引。为了进行公平比较，作者描述为二级索引的索引将被视为一级索引。二级索引从不将数据聚集在磁盘页面中，因此在查询时间上总是处于劣势。请记住，我们所说的“一级”索引仅指决定记录物理位置的索引，而不是“主键”上的索引。二级索引只能聚集对记录的引用，而不能聚集记录本身。

In an attempt to reduce the quadratic space and linear updating of the "copy" approach, the "log" approach stores only the changes that occur in the database timestamped by the time instant on which they occurred. Update processing is clearly reduced to $O\left( 1\right)$ per change, since this history management scheme appends the sequence of inputs in a "log" without any other processing. The space is similarly reduced to the minimal $O\left( {n/B}\right)$ . Nevertheless,this straightforward approach will increase the query time to $O\left( {n/B}\right)$ ,since in order to reconstruct a past state, the whole "log" may have to be searched.

为了减少“复制”方法的二次空间和线性更新，“日志”方法仅存储数据库中发生的更改，并按其发生的时间戳进行标记。显然，每次更改的更新处理减少到 $O\left( 1\right)$，因为这种历史管理方案将输入序列追加到“日志”中，而无需进行任何其他处理。同样，空间也减少到最小的 $O\left( {n/B}\right)$。然而，这种直接的方法会将查询时间增加到 $O\left( {n/B}\right)$，因为为了重建过去的状态，可能必须搜索整个“日志”。

Combinations of the two straightforward approaches are possible; for example, a method could keep repeated time-slices of the database state and "logs" of the changes between the stored time-slices. If repeated timeslices are stored after some bounded number of changes, this solution is equivalent to the "copy" approach, since it is equivalent to using different time units (and therefore changing only the constant in the space complexity measure). If the number of changes between repeated timeslices is not bounded, the method is equivalent to the "log" approach, as it corresponds to a series of logs. We use the two extreme cases to characterize the performance of the examined transaction-time methods. Some of the proposed methods are equivalent to one of the two extremes. However, it is possible to combine the fast query time of the first approach with the space and update requirements of the second.

两种直接方法的组合是可行的；例如，一种方法可以保留数据库状态的重复时间片以及存储的时间片之间的更改“日志”。如果在有界数量的更改之后存储重复的时间片，这种解决方案等同于“复制”方法，因为这相当于使用不同的时间单位（因此仅改变空间复杂度度量中的常数）。如果重复时间片之间的更改数量无界，该方法等同于“日志”方法，因为它对应于一系列日志。我们使用这两种极端情况来描述所研究的事务时间方法的性能。一些提出的方法等同于这两种极端情况之一。然而，有可能将第一种方法的快速查询时间与第二种方法的空间和更新要求相结合。

In order for a method to answer the transaction pure-timeslice ("*/-/point") query efficiently, data must at least be clustered according to its transaction time behavior. Since this query asks for all records "alive" at a given time, this clustering can only be based on the transaction time axis, i.e., records that exist at the same time should be clustered together, independently of their key values. We call access methods that cluster by time only, (transaction) time-only methods. There are methods that cluster by both time and key; we call them (transaction) time-key methods. They optimize queries that involve both time and key predicates, such as the transaction range-timeslice query ("range/-/point"). Clustering by time only can lead to constant update processing per change; thus a good time-only method can "follow" its input changes "on-line." In contrast, clustering by time and key needs some logarithmic updating because changes arrive in time order but not in key order; some appropriate placement of change is needed based on the key it (the change) is applied on.

为了使一种方法能够高效地回答事务纯时间片（“*/-/点”）查询，数据至少必须根据其事务时间行为进行聚类。由于此查询要求获取在给定时间“存活”的所有记录，这种聚类只能基于事务时间轴，即，在同一时间存在的记录应聚集在一起，而与它们的键值无关。我们将仅按时间聚类的访问方法称为（事务）仅时间方法。有些方法同时按时间和键进行聚类；我们将它们称为（事务）时间 - 键方法。它们优化涉及时间和键谓词的查询，例如事务范围 - 时间片查询（“范围/-/点”）。仅按时间聚类可以使每次更改的更新处理保持恒定；因此，一种好的仅时间方法可以“在线”跟踪其输入更改。相比之下，按时间和键进行聚类需要一些对数级的更新，因为更改按时间顺序到达，但不按键顺序到达；需要根据应用更改的键进行一些适当的更改放置。

### 4.2 The Transaction Pure-Key Query

### 4.2 事务纯键查询

The "copy" and "log" solutions could be used for the pure-key query ("point/-/*"). But they are both very inefficient. The "copy" method uses too much space, no matter what query it is used for. In addition, finding a key in a timeslice implies either that one uses linear search or that there is some organization on each timeslice (such as an index on the key). The "log" approach requires running from the beginning of the log to the time of the query, keeping the most recent version of the record with that key. This is still in $O\left( {n/B}\right)$ time.

“复制”和“日志”解决方案可用于纯键查询（“点/-/*”）。但它们的效率都非常低。“复制”方法无论用于何种查询，都使用过多的空间。此外，在一个时间片中查找一个键意味着要么使用线性搜索，要么每个时间片有某种组织形式（如键上的索引）。“日志”方法需要从日志开头运行到查询时间，保留具有该键的记录的最新版本。这仍然需要$O\left( {n/B}\right)$时间。

A better solution is to store the history of each key separately, i.e., cluster data by key only. This creates a (transaction) key-only method. Since at each transaction time instant there exists at most one "alive" version of a given key, versions of the same key can be linked together. Access to a key's (transaction-time) history can be implemented by a hashing function (which must be dynamic hashing, as it has to support the addition of new keys) or a balanced multiway search tree (B-tree). Hashing provides constant access (in the expected amortized sense), while the B-tree provides logarithmic access. Note that hashing does not guarantee against pathological worst cases, while the B-tree does. Hashing cannot be used to obtain the history for a range of keys (as in the general class (III) query). After the queried key is identified, its whole history can be retrieved (forward or backward reconstruction using the list of versions).

更好的解决方案是分别存储每个键的历史记录，即仅按键对数据进行聚类。这就产生了一种（事务）仅键方法。由于在每个事务时间点，给定键最多存在一个“存活”版本，同一键的不同版本可以链接在一起。对键的（事务时间）历史记录的访问可以通过哈希函数（必须是动态哈希，因为它必须支持添加新键）或平衡多路搜索树（B 树）来实现。哈希提供常数级访问（在预期的平摊意义上），而 B 树提供对数级访问。请注意，哈希不能保证避免病态的最坏情况，而 B 树可以。哈希不能用于获取一系列键的历史记录（如一般类（III）查询）。在识别出查询的键之后，可以检索其整个历史记录（使用版本列表进行向前或向后重建）。

The list of versions of each key can be further organized in a separate array indexed by transaction time to answer a pure-key query with time predicate ("point/-/range"). Since updates are appended at the end of such an array, a simple paginated multilevel index can be implemented on each array to expedite searching. Then a query of the form "provide the history of key $k$ after (before) time $t$ " is addressed by first finding $k$ (using hashing or the B-tree) and then locating the version of $k$ that is closest to transaction time $t$ using the multilevel index on $k$ ’s versions. This takes $O\left( {{\log }_{B}n}\right)$ time (each array can be $O$ $\left( {n/B}\right)$ large).

每个键的版本列表可以进一步组织成一个由事务时间索引的单独数组，以回答带有时间谓词的纯键查询（“点/-/范围”）。由于更新会追加到这样一个数组的末尾，可以在每个数组上实现一个简单的分页多级索引以加快搜索速度。然后，对于“提供键$k$在时间$t$之后（之前）的历史记录”形式的查询，首先找到$k$（使用哈希或 B 树），然后使用$k$版本上的多级索引定位最接近事务时间$t$的$k$版本。这需要$O\left( {{\log }_{B}n}\right)$时间（每个数组可以是$O$$\left( {n/B}\right)$大）。

The above straightforward data clustering by key is only efficient for class III queries, but is not efficient for any of the other two classes. For example, to answer a "*/-/point" query, each key ever created in the evolution must be searched for being "alive" at the query transaction time, and it takes logarithmic time to search each key's version history.

上述直接按键进行的数据聚类仅对 III 类查询有效，但对其他两类查询均无效。例如，为了回答“*/-/点”查询，必须搜索演化过程中创建的每个键在查询事务时间是否“存活”，并且搜索每个键的版本历史记录需要对数级时间。

### 4.3 The Transaction Range-Timeslice Query

### 4.3 事务范围 - 时间片查询

If records that are "logically" related for a given query can also be stored physically close, then the query is optimized as fewer pages are accessed. Therefore, to answer a "range/-/point" query efficiently, it is best to cluster by transaction time and key within pages. This is very similar to spatial indexing; but it has some special properties.

如果针对给定查询在“逻辑上”相关的记录也能在物理上存储得相近，那么由于访问的页面更少，查询将得到优化。因此，为了高效地回答“范围/ - /点”查询，最好在页面内按交易时间和键进行聚类。这与空间索引非常相似，但它有一些特殊属性。

If the time-key space is partitioned into disjoint rectangles, one for each disk page, and only one copy of each record is kept, long-lived records (records with long transaction-time intervals) have to be collocated with many short-lived ones that cannot all fit on the same page. We cannot partition the space in this way without allowing duplicate records. So we are reduced to either making copies (data duplication), allowing overlap of time-key rectangles (data bounding), or mapping records represented by key (transaction) start_time and end_time, to points in three-dimensional space (data mapping) and using a multidimensional search method.

如果将时间 - 键空间划分为不相交的矩形，每个磁盘页对应一个矩形，并且每条记录只保留一份副本，那么长期存在的记录（交易时间间隔较长的记录）必须与许多短期存在的记录放在一起，而这些短期记录不可能全部放在同一页面上。如果不允许记录重复，我们就不能以这种方式划分空间。因此，我们只能选择复制记录（数据重复）、允许时间 - 键矩形重叠（数据边界划分），或者将由键（交易）开始时间和结束时间表示的记录映射到三维空间中的点（数据映射），并使用多维搜索方法。

Time-key spaces do not have the "density" problem of spatial indexes. Density is defined as the largest overlap of spatial objects at a point. There is only one version of each key at a given time, so time-key objects (line segments in time-key space) never overlap. This makes data duplication a more attractive option than spatial indexing, especially if the amount of duplication can be limited as in Eaton [1986], Lomet and Sal-zberg [1990], Lanka and Mays [1991], Becker et al. [1996], and Varman and Verma [1997].

时间 - 键空间不存在空间索引的“密度”问题。密度定义为空间对象在某一点的最大重叠程度。在给定时间，每个键只有一个版本，因此时间 - 键对象（时间 - 键空间中的线段）永远不会重叠。这使得数据重复比空间索引更具吸引力，特别是如果像伊顿（Eaton）[1986]、洛梅特（Lomet）和萨尔茨伯格（Sal - zberg）[1990]、兰卡（Lanka）和梅斯（Mays）[1991]、贝克尔（Becker）等人[1996]以及瓦尔曼（Varman）和维尔马（Verma）[1997]所做的那样，能够限制重复的数量。

Data bounding may force single-point queries to use backtracking, since there is not a unique path to a given time-key point. In general, for the data-bounding approach, temporal indexing has worse problems than spatial indexing because long-lived records are likely to be common. In a data-bounding structure, such a record is stored in a page with a long time-span and some key range. Every timeslice query in that timespan must access that page, even though the long-lived record may be the only one alive at search time (the other records in the page are alive in another part of the timespan). The R-tree-based-methods use data bounding [Stonebraker 1987; Kolovson and Stonebraker 1989; 1991].

数据边界划分可能会迫使单点查询使用回溯法，因为到给定时间 - 键点没有唯一的路径。一般来说，对于数据边界划分方法，时态索引比空间索引存在更严重的问题，因为长期存在的记录可能很常见。在数据边界划分结构中，这样的记录存储在具有较长时间跨度和一定键范围的页面中。该时间跨度内的每个时间片查询都必须访问该页面，即使在搜索时长期存在的记录可能是唯一存活的记录（页面中的其他记录在时间跨度的其他部分存活）。基于R树的方法使用数据边界划分[斯通布雷克（Stonebraker）1987；科洛夫森（Kolovson）和斯通布雷克（Stonebraker）1989；1991]。

The third possibility, data mapping, maps a record to three (or more) coordinates-transaction start_time, end- _time, and key(s)—and then uses a mul-tiattribute point index. Here records with long transaction-time intervals are clustered with other records with long intervals because their start and end times are close. If they were alive at nearby times, records with short transaction-time intervals are clustered with other records with short intervals. This is efficient for most queries, as the long-lived records are the answers to many queries. The pages with short-lived records effectively partition the answers to different queries; most such pages are not touched for a given timeslice query. However, there are special problems because many records may still be current and have growing lifetimes (i.e., transaction-time intervals extending to now). This approach is discussed further at the end of Section 5.1.3.

第三种可能性，即数据映射，将一条记录映射到三个（或更多）坐标——交易开始时间、结束时间和键——然后使用多属性点索引。在这里，交易时间间隔较长的记录会与其他时间间隔较长的记录聚类，因为它们的开始和结束时间相近。如果短期交易时间间隔的记录在相近时间存活，它们会与其他短期间隔的记录聚类。这对大多数查询来说是高效的，因为长期存在的记录是许多查询的答案。包含短期记录的页面有效地划分了不同查询的答案；对于给定的时间片查询，大多数这样的页面不会被访问。然而，存在一些特殊问题，因为许多记录可能仍然是当前有效的，并且其生命周期在不断延长（即，交易时间间隔延伸到现在）。本节5.1.3末尾将进一步讨论这种方法。

Naturally, the most efficient methods for the transaction range-timeslice query are the ones that combine the time and key dimensions. In contrast, by using a (transaction) time-only method, the whole timeslice for the given transaction time is first reconstructed and then the records with keys outside the given range are eliminated. This is clearly inefficient, especially if the requested range is a small part of the whole timeslice.

自然地，用于交易范围 - 时间片查询的最有效方法是将时间和键维度结合起来的方法。相比之下，通过使用仅基于（交易）时间的方法，首先要重建给定交易时间的整个时间片，然后排除键不在给定范围内的记录。这显然是低效的，特别是如果请求的范围只是整个时间片的一小部分。

### 4.4 Bitemporal Queries

### 4.4 双时态查询

An obvious approach is to index bitemporal objects on a single time axis (transaction or valid time) and use a single time access method. For example, if a transaction access method is utilized, a bitemporal "*/point/point" query is answered in two steps. First all bitemporal objects existing at transaction time $t$ are found. Then the valid time interval of each such object is checked to see if it includes valid time $v$ . This approach is inefficient because very few of the accessed objects may actually satisfy the valid-time predicate.

一种明显的方法是在单个时间轴（交易时间或有效时间）上对双时态对象进行索引，并使用单时间访问方法。例如，如果使用交易访问方法，双时态“*/点/点”查询分两步回答。首先找到在交易时间$t$存在的所有双时态对象。然后检查每个这样的对象的有效时间间隔，看它是否包含有效时间$v$。这种方法效率低下，因为被访问的对象中实际上可能只有很少一部分满足有效时间谓词。

<!-- Media -->

<!-- figureText: ${I}_{3}$ now ${t}_{4}$ ${t}_{5}$ ${t}_{1}$ ${t}_{2}$ ${t}_{3}$ -->

<img src="https://cdn.noedgeai.com/0195c90d-0a33-750f-877f-c1cec48583bb_16.jpg?x=216&y=245&w=537&h=263&r=0"/>

Figure 4. The bounding-rectangle approach for bitemporal queries (the key dimension is not shown). The evolution of Figure 3 is depicted as of (transaction) time $t > {t}_{5}$ . Modification of interval ${I}_{1}$ at ${t}_{5}$ ends the initial rectangle for ${I}_{1}$ and inserts a new rectangle from ${t}_{5}$ to now.

图4. 双时态查询的边界矩形方法（未显示键维度）。图3的演变情况描绘的是截至（交易）时间$t > {t}_{5}$的情况。在${t}_{5}$对区间${I}_{1}$进行修改，结束了${I}_{1}$的初始矩形，并插入了一个从${t}_{5}$到现在的新矩形。

<!-- Media -->

If both axes are utilized, an obvious approach is an extended combination of the "copy" and "log" solutions. This approach stores copies of the collections $C\left( t\right)$ (Figure 3) at given transaction-time instants and a log of changes between copies. Together with each collection $C\left( t\right)$ ,an access method (for example an R-tree [Guttman 1984]) that indexes the objects of this $C\left( t\right)$ is also stored. Conceptually it is like storing snapshots of $\mathrm{R}$ -trees and the changes between them. While each R-tree enables efficient searching on a stored collection $C\left( t\right)$ ,the approach is clearly inefficient because the space or query time increases dramatically, depending on the frequency of snapshots.

如果同时使用两个轴，一种明显的方法是将“复制”和“日志”解决方案进行扩展组合。这种方法在给定的事务时间点存储集合 $C\left( t\right)$ 的副本（图 3），并记录副本之间的更改日志。除了每个集合 $C\left( t\right)$ 之外，还会存储一个对该 $C\left( t\right)$ 中的对象进行索引的访问方法（例如 R 树 [古特曼 1984 年]）。从概念上讲，这就像是存储 $\mathrm{R}$ 树的快照以及它们之间的更改。虽然每个 R 树都能在存储的集合 $C\left( t\right)$ 上实现高效搜索，但这种方法显然效率低下，因为空间或查询时间会根据快照的频率急剧增加。

The data bounding and data mapping approaches can also be used in a bitemporal environment. However, the added (valid-time) dimension provides an extra reason for inefficiency. For example, the bounding rectangle of a bitemporal object consists of two intervals (Figure 4; taken from Kumar et al. [1998]). A "*/point/point" query is translated into finding all rectangles that include the query point $\left( {{t}_{i},{v}_{j}}\right)$ An R-tree [Guttman 1984] could be used to manage these rectangles. However, the special characteristics of transaction time (many rectangles may extend up to now) and the inclusion of the valid-time dimension increase the possibility of extensive overlap, which in turn reduces the R-tree query efficiency [Kumar et al. 1998].

数据边界和数据映射方法也可用于双时态环境。然而，新增的（有效时间）维度为效率低下提供了额外的原因。例如，双时态对象的边界矩形由两个区间组成（图 4；取自库马尔等人 [1998 年]）。一个“*/点/点”查询会被转换为查找所有包含查询点 $\left( {{t}_{i},{v}_{j}}\right)$ 的矩形。可以使用 R 树 [古特曼 1984 年] 来管理这些矩形。然而，事务时间的特殊特性（许多矩形可能一直延伸到现在）以及有效时间维度的包含增加了广泛重叠的可能性，这反过来又降低了 R 树的查询效率 [库马尔等人 1998 年]。

### 4.5 Separating Past from Current Data and Use of WORM disks

### 4.5 分离过去数据和当前数据以及使用一次写入多次读取（WORM）磁盘

In transaction or bitemporal databases, it is usually the case that access to current data is more frequent than to past data (in the transaction-time sense). In addition, since the bulk of data in these databases is due to the historical part, it is advantageous to use a higher capacity, but slower access medium, for the past data such as optical disks. First, the method should provide for natural separation between current and past data. There are two ways to achieve this separation: (a) with the "manual" approach a process will vacuum all records that are "dead" (in the transaction-time sense) when the process is invoked (this vacuuming process can be invoked at any time); (b) with the "automated" approach, where such "dead" records are migrated to the optical disk due directly to the evolution process (for example during an update). The total I/O involved is likely to be smaller than in a manual method, since it is piggybacked on I/O, which, in any case, is necessary for index maintenance (such as splitting a full node).

在事务或双时态数据库中，通常对当前数据的访问比过去数据（从事务时间的角度来看）更频繁。此外，由于这些数据库中的大部分数据是历史数据，因此使用大容量但访问速度较慢的介质（如光盘）来存储过去的数据是有利的。首先，该方法应能自然地分离当前数据和过去数据。有两种方法可以实现这种分离：(a) “手动”方法，在调用该过程时，会清理所有“已失效”（从事务时间的角度来看）的记录（此清理过程可以在任何时间调用）；(b) “自动”方法，此类“已失效”的记录会直接由于数据演变过程（例如在更新期间）迁移到光盘上。涉及的总输入/输出（I/O）可能比手动方法少，因为它是搭在 I/O 操作上进行的，而无论如何，I/O 操作对于索引维护（如拆分已满节点）是必要的。

Even though write-many read-many optical disks are available,WORM optical disks are still the main choice for storing large amounts of archival data; they are less expensive, have larger capacities, and usually have faster write transfer times. Since the contents of WORM disk blocks cannot be changed after their initial writing (due to added error-correcting code), data that is to be appended on a WORM disk should not be allowed to change in the future. Since on the transaction axis the past is not changed, past data can be written on the WORM disk.

尽管有多次写入多次读取的光盘可用，但一次写入多次读取（WORM）光盘仍然是存储大量存档数据的主要选择；它们更便宜，容量更大，并且通常写入传输速度更快。由于 WORM 磁盘块的内容在首次写入后无法更改（由于添加了纠错码），因此不允许将来对要追加到 WORM 磁盘上的数据进行更改。由于在事务轴上过去的数据不会改变，因此可以将过去的数据写入 WORM 磁盘。

We emphasize again that methods that can be used on WORM disks are not "WORM methods"—they can also be used on magnetic disks. Thus the question of separation of past and current records can be considered regardless of the availability of WORM disks.

我们再次强调，可用于 WORM 磁盘的方法并非“WORM 方法” —— 它们也可用于磁盘。因此，无论是否有 WORM 磁盘，都可以考虑分离过去记录和当前记录的问题。

## 5. METHOD CLASSIFICATION AND COMPARISON

## 5. 方法分类与比较

This section provides a concise description of the methods we examine. Since it is practically impossible to run simulations for all methods on the same collections of data and queries, our analysis is based on worst-case performance. Various access method proposals provide a performance analysis that may have strong assumptions about the input data (uniform distribution of data points, etc.), and it may very well be that under those constraints the proposed method works quite well. Our purpose, however, is to categorize the methods without any assumption on the input data or the frequency of queries. Obviously, the worst-case analysis may penalize a method for some very unlikely scenarios; to distinguish against likely worst cases, we call such scenarios pathological worst cases. We also point out some features that may affect average-case behavior without necessarily affecting worst-case behavior.

本节简要描述了我们研究的方法。由于实际上不可能在相同的数据集合和查询上对所有方法进行模拟，因此我们的分析基于最坏情况性能。各种访问方法的提议提供的性能分析可能对输入数据有很强的假设（数据点均匀分布等），很可能在这些约束条件下，所提议的方法效果相当好。然而，我们的目的是在不对输入数据或查询频率做任何假设的情况下对这些方法进行分类。显然，最坏情况分析可能会因一些极不可能出现的情况而对某种方法不利；为了区分可能的最坏情况，我们将此类情况称为病态最坏情况。我们还指出了一些可能影响平均情况行为但不一定影响最坏情况行为的特征。

We first describe transaction-time access methods. These methods are further classified as key-only, time-only, and time-key, based on the way data is clustered. Among the key-only methods, we study reverse chaining, accession lists, time sequence arrays, and C-lists. Among time-only methods, we examine append-only tree, time-index and its variants (monotonic B-tree, time-index+), the differential file approach, checkpoint index, archivable time index, snapshot index, and the windows method. In the time-key category, we present the POSTGRES storage system and the use of composite indexes, segment- $R$ tree,write-once $B$ -tree,time-split $B$ -tree,persistent $B$ -tree,multiversion $B$ - tree, multiversion access structure, and the overlapping $B$ -tree. A comparison table (Table II) is included at the end of the section with a summary of each method's worst-case performance. We then proceed with the valid-time access methods, where we discuss the meta-block tree, external degment tree, external interval tree,and the MAP21 methods. The bitemporal category describes $M$ -IVTT,the bitemporal interval tree, and bitemporal $R$ -tree.

我们首先描述事务时间访问方法。这些方法根据数据的聚类方式进一步分为仅键（key-only）、仅时间（time-only）和时间 - 键（time-key）三类。在仅键方法中，我们研究反向链接（reverse chaining）、访问列表（accession lists）、时间序列数组（time sequence arrays）和 C 列表（C-lists）。在仅时间方法中，我们考察仅追加树（append-only tree）、时间索引（time-index）及其变体（单调 B 树（monotonic B-tree）、时间索引 +（time-index+））、差分文件方法（differential file approach）、检查点索引（checkpoint index）、可归档时间索引（archivable time index）、快照索引（snapshot index）和窗口方法（windows method）。在时间 - 键类别中，我们介绍 POSTGRES 存储系统以及复合索引（composite indexes）、段 - $R$ 树（segment- $R$ tree）、一次写入 $B$ 树（write-once $B$ -tree）、时间分割 $B$ 树（time-split $B$ -tree）、持久 $B$ 树（persistent $B$ -tree）、多版本 $B$ 树（multiversion $B$ - tree）、多版本访问结构（multiversion access structure）和重叠 $B$ 树（overlapping $B$ -tree）的使用。本节末尾包含一个比较表（表 II），总结了每种方法的最坏情况性能。然后我们继续讨论有效时间访问方法，其中我们讨论元块树（meta-block tree）、外部段树（external degment tree）、外部区间树（external interval tree）和 MAP21 方法。双时态类别描述了 $M$ -IVTT、双时态区间树（bitemporal interval tree）和双时态 $R$ 树（bitemporal $R$ -tree）。

### 5.1 Transaction-Time Methods

### 5.1 事务时间方法

In this category we include methods that assume that changes arrive in increasing time order, a characteristic of transaction time. This property greatly affects the update processing of the method. If "out of order" changes (a characteristic of valid-time) are to be supported, the updating cost becomes much higher (practically prohibitive).

在这一类别中，我们纳入了假设变更按时间递增顺序到来的方法，这是事务时间的一个特征。这一特性极大地影响了该方法的更新处理。如果要支持“乱序”变更（有效时间的一个特征），更新成本会变得高得多（实际上难以承受）。

5.1.1 Key-Only Methods. The basic characteristic of transaction key-only approaches is the organization of evolving data by key (surrogate), i.e., all versions that a given key assumes are "clustered" together logically or physically. Such organization makes these methods more efficient for transaction pure-key queries. In addition, the approaches considered here correspond to the earliest solutions proposed for time-evolving data.

5.1.1 仅键方法。事务仅键方法的基本特征是按键（代理键）组织演变数据，即给定键的所有版本在逻辑或物理上“聚类”在一起。这种组织方式使这些方法在处理事务纯键查询时更高效。此外，这里考虑的方法对应于最早为时间演变数据提出的解决方案。

Reverse chaining was introduced in Ben-Zvi [1982] and further developed in Lum et al. [1984]. Under this approach, previous versions of a given key are linked together in reverse chronological order. The idea of keeping separate stores for current and past data was also introduced. Current data is assumed to be queried more often, so by separating it from past data, the size of the search structure is decreased and queries for current data become faster.

反向链接由 Ben - Zvi [1982] 提出，并由 Lum 等人 [1984] 进一步发展。在这种方法下，给定键的先前版本按逆时间顺序链接在一起。还提出了为当前数据和过去数据分别设置存储的想法。假设当前数据被查询的频率更高，因此通过将其与过去数据分离，搜索结构的大小减小，对当前数据的查询变得更快。

Each version of a key is represented by a tuple (which includes the key, attribute value, and a lifespan interval) augmented with a pointer field that points to the previous version (if any) of this key. When a key is first inserted into a relation, its corresponding tuple is put into the current store with its previous-version pointer being null. When the attribute value of this key is changed, the version existing in the current store is moved to the past store, with the new tuple replacing it in the current store. The previous-version pointer of the new tuple points to the location of the previous version in the past store. Hence a chain of past versions is created out of each current key. Tuples are stored in the past store without necessarily being clustered by key.

键的每个版本由一个元组（包括键、属性值和生命周期间隔）表示，并增加了一个指针字段，该指针指向该键的前一个版本（如果有的话）。当一个键首次插入到一个关系中时，其对应的元组被放入当前存储中，其前一个版本指针为空。当该键的属性值发生变化时，当前存储中现有的版本被移动到过去存储中，新元组替换它在当前存储中的位置。新元组的前一个版本指针指向过去存储中前一个版本的位置。因此，每个当前键都会创建一个过去版本的链。元组存储在过去存储中时不一定按键聚类。

Current keys are indexed by a regular ${\mathrm{B}}^{ + }$ -tree ("front" ${\mathrm{B}}^{ + }$ -tree). The chain of past versions of a current key is accessed by following previous-version pointers starting from the current key. If a current key is deleted, it is removed from the ${\mathrm{B}}^{ + }$ -tree and is inserted in a second ${\mathrm{B}}^{ + }$ -tree ("back" ${\mathrm{B}}^{ + }$ -tree),which indexes the latest version of keys that are not current. The past version chain of the deleted key is still accessed from its latest version stored in the "back" ${\mathrm{B}}^{ + }$ -tree. If a key is "reborn" it is reinserted in the "front" ${\mathrm{B}}^{ + }$ -tree. Subsequent modifications of this current key create a new chain of past versions. It is thus possible to have two chains of past versions, one starting from its current version and one from a past version, for the same key. Hence queries about the past are directed to both ${\mathrm{B}}^{ + }$ -trees. If the key is deleted again later, its new chain of past versions is attached to its previous chain by appropriately updating the latest version stored in the "back" ${\mathrm{B}}^{ + }$ - tree.

当前键由常规的 ${\mathrm{B}}^{ + }$ 树（“前” ${\mathrm{B}}^{ + }$ 树）索引。通过从当前键开始跟随前一个版本指针来访问当前键的过去版本链。如果删除一个当前键，它将从 ${\mathrm{B}}^{ + }$ 树中移除，并插入到第二个 ${\mathrm{B}}^{ + }$ 树（“后” ${\mathrm{B}}^{ + }$ 树）中，该树索引非当前键的最新版本。已删除键的过去版本链仍然可以从存储在“后” ${\mathrm{B}}^{ + }$ 树中的其最新版本访问。如果一个键“重生”，它将被重新插入到“前” ${\mathrm{B}}^{ + }$ 树中。对这个当前键的后续修改会创建一个新的过去版本链。因此，对于同一个键，可能有两个过去版本链，一个从其当前版本开始，一个从过去版本开始。因此，关于过去的查询会指向两个 ${\mathrm{B}}^{ + }$ 树。如果该键稍后再次被删除，通过适当地更新存储在“后” ${\mathrm{B}}^{ + }$ 树中的最新版本，其新的过去版本链将附加到其先前的链上。

Clearly,this approach uses $O\left( {n/B}\right)$ space,where $n$ denotes the number of changes and $B$ is the page size. The number of changes corresponds to the number of versions for all keys ever created. When a change occurs (such as a new version of key or the deletion of key),the "front" ${\mathrm{B}}^{ + }$ -tree (current store) has first to be searched to locate the current version of key. If it is a deletion, the "back" ${\mathrm{B}}^{ + }$ -tree is also searched to locate the latest version of key, if any. So the update processing of this method is $O\left( {{\log }_{B}n}\right)$ ,since the number of different keys can be similar to the number of changes.

显然，这种方法使用$O\left( {n/B}\right)$的空间，其中$n$表示更改的数量，$B$是页面大小。更改的数量对应于所有已创建键的版本数量。当发生更改（例如键的新版本或键的删除）时，首先要搜索“前”${\mathrm{B}}^{ + }$树（当前存储）以定位键的当前版本。如果是删除操作，还会搜索“后”${\mathrm{B}}^{ + }$树以定位键的最新版本（如果有的话）。因此，由于不同键的数量可能与更改的数量相近，此方法的更新处理复杂度为$O\left( {{\log }_{B}n}\right)$。

To find all previous versions of a given key, the "front" B+-tree is first searched for the latest version of key; if a key is in the current store, its pointer will provide access to recent past versions of the key. Since version lists are in reverse chronological order, we have to follow such a list until a version number (transaction timestamp) that is less or equal to the query timestamp is found. The "back" ${\mathrm{B}}^{ + }$ -tree is then searched for older past versions. If $a$ denotes all past versions of a key, the query time is $O\left( {{\log }_{B}n + a}\right)$ ,since versions of a given key could in the worst case be stored in different pages. This can be improved if cellular chaining, clustering, or stacking is is used [Ahn and Snodgrass 1988]. If each collection of versions for a given key is clustered in a set of pages but versions of distinct keys are never on the same page, query time is $O\left( {{\log }_{B}n + a/B}\right)$ but space utilization is $O\left( n\right)$ pages (not $O\left( {n/B}\right)$ ,as the versions of a key may not be enough to justify the use of a full page.

要查找给定键的所有先前版本，首先在“前”B + 树中搜索该键的最新版本；如果一个键存在于当前存储中，其指针将提供对该键近期旧版本的访问。由于版本列表是按逆时间顺序排列的，我们必须沿着这样的列表查找，直到找到一个小于或等于查询时间戳的版本号（事务时间戳）。然后在“后”${\mathrm{B}}^{ + }$树中搜索更早的旧版本。如果$a$表示一个键的所有过去版本，查询时间为$O\left( {{\log }_{B}n + a}\right)$，因为在最坏的情况下，给定键的版本可能存储在不同的页面中。如果使用单元链接、聚类或堆叠技术，这种情况可以得到改善[Ahn和Snodgrass 1988]。如果给定键的每个版本集合都聚类在一组页面中，但不同键的版本永远不会在同一页面上，查询时间为$O\left( {{\log }_{B}n + a/B}\right)$，但空间利用率为$O\left( n\right)$页（而不是$O\left( {n/B}\right)$，因为一个键的版本可能不足以填满一整页）。

Reverse chaining can be further improved by the introduction of accession lists [Ahn and Snodgrass 1988]. An accession list clusters all version numbers (timestamps) of a given key together. Each timestamp is associated with a pointer to the accompanying tuple, which is stored in the past store (or to a cluster of tuples). Thus, instead of searching a reverse chain until a given timestamp is reached, we can search an index of the chain's timestamps. As timestamps are stored in chronological order on an accession list, finding the appropriate version of a given key takes $O\left( {{\log }_{B}n + {\log }_{B}a}\right)$ . The space and update processing remain as before.

通过引入访问列表，可以进一步改进反向链接技术[Ahn和Snodgrass 1988]。访问列表将给定键的所有版本号（时间戳）聚类在一起。每个时间戳都与一个指向伴随元组的指针相关联，该元组存储在过去存储中（或一个元组簇中）。因此，我们不必搜索反向链直到达到给定的时间戳，而是可以搜索该链的时间戳索引。由于时间戳按时间顺序存储在访问列表中，查找给定键的适当版本需要$O\left( {{\log }_{B}n + {\log }_{B}a}\right)$的时间。空间和更新处理与之前相同。

While the above structure can be efficient for a transaction pure-key query, answering pure- or range-timeslice queries is problematic. For example, to answer a "*/-/point" query that is satisfied only by some keys, we have to search the accession lists of all keys ever created.

虽然上述结构对于事务纯键查询可能是高效的，但回答纯时间片查询或范围时间片查询却存在问题。例如，要回答一个仅由某些键满足的“*/ - /点”查询，我们必须搜索所有已创建键的访问列表。

Another early approach proposed the use of time sequence arrays (TSAs) [Shoshani and Kawagoe 1986]. Conceptually, a TSA is a two-dimensional array with a row for each key ever created; each column represents a time instant. The(x,y)entry stores the value of key $x$ at time $y$ . Static (the data set has been fully collected) and dynamic (the data set is continuously growing , as in a transaction-time environment) data are examined. If this structure is implemented as a two-dimensional array, query time is minimal (just access the appropriate array entry), but update processing and space are prohibitive $(O\left( n\right)$ and $O\left( {n}^{2}\right)$ ,respectively). We could implement each row as an array, keeping only those values where there was a change; this is conceptually the same solution as reverse chaining with accession lists. A solution based on a multidimensional partitioning scheme is proposed in Rotem and Segev [1987], but the underlying assumption is that the whole temporal evolution is known in advance, before the partitioning scheme is implemented.

另一种早期方法提出使用时间序列数组（TSA）[Shoshani和Kawagoe 1986]。从概念上讲，TSA是一个二维数组，为每个已创建的键设置一行；每列代表一个时间点。(x, y)项存储键$x$在时间$y$的值。研究了静态（数据集已完全收集）和动态（数据集在不断增长，如在事务时间环境中）数据。如果将此结构实现为二维数组，查询时间最短（只需访问适当的数组项），但更新处理和空间开销过大（分别为$(O\left( n\right)$和$O\left( {n}^{2}\right)$）。我们可以将每行实现为一个数组，只保留有更改的值；从概念上讲，这与带有访问列表的反向链接是相同的解决方案。Rotem和Segev [1987]提出了一种基于多维分区方案的解决方案，但潜在的假设是，在实施分区方案之前，整个时间演变情况是已知的。

The theoretically optimal solution for the transaction pure-key query with time predicate is provided by the $C$ -lists of Varman and Verma [1997]. C-lists are similar to accession lists, in that they cluster the versions of a given key together. There are two main differences. First, access to each C-list is provided through another method, the multiversion access structure (MVAS in short; MVAS is discussed later with the time-key methods) [Varman and Verma 1997]. Second, maintenance is more complicated: splitting/ merging C-list pages is guided by page splitting/merging MVAS (for details, see Varman and Verma [1997]). If there are $m$ "alive" keys in the structure, updating takes $O\left( {{\log }_{B}m}\right)$ . The history of key $k$ before time $t$ is found in $O\left( {{\log }_{B}n + a/B}\right)$ I/Os, which is optimal. C-lists have an advantage in that they can be combined with the MVAS structure to create a method that optimally answers both the range-timeslice and pure-key with time predicate queries. However, the method needs an extra $\mathrm{B} +$ -tree,together with double pointers between the C-lists and MVAS, which adds implementation complexity.

Varman和Verma（1997年）提出的$C$列表为带有时间谓词的事务纯键查询提供了理论上的最优解决方案。C列表与访问列表类似，它们会将给定键的各个版本聚集在一起。主要有两个区别。首先，对每个C列表的访问是通过另一种方法实现的，即多版本访问结构（简称MVAS；后面会结合时间键方法讨论MVAS）[Varman和Verma 1997年]。其次，维护更为复杂：C列表页面的拆分/合并由页面拆分/合并MVAS指导（详情见Varman和Verma [1997年]）。如果结构中有$m$个“活跃”键，更新操作需要$O\left( {{\log }_{B}m}\right)$。在时间$t$之前键$k$的历史记录可以通过$O\left( {{\log }_{B}n + a/B}\right)$次I/O操作找到，这是最优的。C列表的一个优势在于，它们可以与MVAS结构相结合，创建一种方法，以最优方式回答范围时间片查询和带有时间谓词的纯键查询。然而，该方法需要额外的$\mathrm{B} +$树，以及C列表和MVAS之间的双指针，这增加了实现的复杂性。

5.1.2 Time-OnlyMethods. Mosttime-only methods timestamp changes (additions, deletions, etc.) by the transaction time they occurred and append them in some form of a "history log." Since no clustering of data according to keys is made, such methods optimize "*/-/point" or "*/-/range" queries. Because changes arrive in chronological order, ideally a time-only method can provide constant update processing (as the change is simply appended at the end of the "history log"); this advantage is important in applications where changes are frequent and the database has to "follow" these changes in an on- line fashion. For efficient query time, most methods use some index on the top of the "history log" that indexes the (transaction) timestamps of the changes. Because of the time-ordered changes, the cost of maintaining this (paginated) index on the transaction time-axis is minimal, amortized $O\left( 1\right)$ per change.

5.1.2 仅基于时间的方法。大多数仅基于时间的方法会根据事务发生的时间对更改（添加、删除等）进行时间戳标记，并以某种“历史日志”的形式将它们追加记录。由于没有根据键对数据进行聚类，此类方法对“*/-/点”或“*/-/范围”查询进行了优化。由于更改是按时间顺序到来的，理想情况下，仅基于时间的方法可以提供恒定的更新处理（因为更改只需简单地追加到“历史日志”的末尾）；在更改频繁且数据库必须以在线方式“跟踪”这些更改的应用中，这一优势非常重要。为了实现高效的查询时间，大多数方法会在“历史日志”之上使用某种索引，对更改的（事务）时间戳进行索引。由于更改是按时间顺序排列的，在事务时间轴上维护这个（分页的）索引的成本极低，每次更改的分摊成本为$O\left( 1\right)$。

While organizing data by only its time behavior provides for very fast updating, it is not efficient for answering transaction range-timeslice queries. In order to use time-only methods for such queries, one suggestion is to employ a separate key index, whose leaves point to predefined key "regions" [Elmasri et al. 1993; Gunadhi 1993]. A key region could be a single key or a collection of keys (either a subrange of the key space or a relation). The history of each "region" is organized separately, using an individual time-only access method (such as the time index or the append-only tree). The key index will direct a change of a given key to update the method that keeps the history of the key's region. However, after the region is found, the placement of this key in the region's access method is based on the key's time behavior only (and no longer on the key itself).

虽然仅根据数据的时间特性来组织数据可以实现非常快速的更新，但对于回答事务范围时间片查询并不高效。为了将仅基于时间的方法用于此类查询，一种建议是采用一个单独的键索引，其叶子节点指向预定义的键“区域”[Elmasri等人1993年；Gunadhi 1993年]。一个键区域可以是单个键，也可以是一组键（可以是键空间的一个子范围，也可以是一个关系）。每个“区域”的历史记录使用单独的仅基于时间的访问方法（如时间索引或仅追加树）进行单独组织。键索引会将给定键的更改引导至维护该键所在区域历史记录的方法进行更新。然而，在找到区域之后，该键在区域访问方法中的放置仅基于该键的时间特性（而不再基于键本身）。

To answer transaction range-time-slice queries, we have to search the history of each region that belongs to the query range. Thus the range-time-slice is constructed by creating the individual timeslices for every region in the query range. If $R$ is the number of regions,the key index adds $O\left( {R/B}\right)$ space and $O\left( {{\log }_{B}R}\right)$ update processing to the performance of the individual historical access methods. The query time for the combination of the key index and the time-only access methods is $o\left( {{Mf}\left( {{n}_{i},t}\right. }\right.$ , $\left. \left. {a}_{i}\right) \right)$ ,where $M$ is the number of regions that fall in the given query range and $f\left( {{n}_{i},t,{a}_{i}}\right)$ is the time needed in each individual region ${r}_{i}\left( {i = 1,\ldots ,m}\right)$ to perform a timeslice query for time $t\left( {n}_{i}\right.$ and ${a}_{i}$ correspond to the total number of changes in ${r}_{i}$ and the number of "alive" objects from ${r}_{i}$ at the time $t$ ,respectively). For example, if the time-index [El-masri et al. 1990] is used as the access method in each individual region, then $f\left( {{n}_{i},t,{a}_{i}}\right)  = O\left( {{\log }_{B}{n}_{i} + {a}_{i}/B}\right) .$

为了回答事务范围 - 时间片查询，我们必须搜索属于查询范围的每个区域的历史记录。因此，通过为查询范围内的每个区域创建单独的时间片来构建范围 - 时间片。如果$R$是区域的数量，键索引会为各个历史访问方法的性能增加$O\left( {R/B}\right)$的空间和$O\left( {{\log }_{B}R}\right)$的更新处理。键索引和仅时间访问方法组合的查询时间为$o\left( {{Mf}\left( {{n}_{i},t}\right. }\right.$，$\left. \left. {a}_{i}\right) \right)$，其中$M$是落在给定查询范围内的区域数量，$f\left( {{n}_{i},t,{a}_{i}}\right)$是在每个单独区域${r}_{i}\left( {i = 1,\ldots ,m}\right)$中执行时间为$t\left( {n}_{i}\right.$的时间片查询所需的时间，${a}_{i}$分别对应于${r}_{i}$中的总更改数量和在时间$t$时来自${r}_{i}$的“活跃”对象数量）。例如，如果在每个单独区域中使用时间索引[El - masri等人，1990]作为访问方法，那么$f\left( {{n}_{i},t,{a}_{i}}\right)  = O\left( {{\log }_{B}{n}_{i} + {a}_{i}/B}\right) .$

There are three drawbacks to this approach: (1) If the query key range is a small subset of a given region, the whole region's timeslice is reconstructed, even if most of its objects may not belong to the query range and thus do not contribute to the answer. (2) If the query key range contains many regions, all these regions have to be searched, even if they may contribute no "alive" objects at the transaction time of interest $t$ . (3) For every region examined, a logarithmic search, at best, is performed to locate $t$ among the changes recorded in the region. To put this in perspective, imagine replacing a multiattribute spatial search structure with a number of collections of records from predefined key ranges in one attribute and then organizing each key range by some other attribute.

这种方法有三个缺点：（1）如果查询键范围是给定区域的一个小子集，即使该区域的大多数对象可能不属于查询范围，从而对查询结果没有贡献，整个区域的时间片也会被重建。（2）如果查询键范围包含许多区域，即使这些区域在感兴趣的事务时间$t$可能没有“活跃”对象，也必须搜索所有这些区域。（3）对于检查的每个区域，最多要进行一次对数搜索，以在该区域记录的更改中定位$t$。为了更直观地理解，想象一下用一个属性中预定义键范围的多个记录集合替换多属性空间搜索结构，然后按另一个属性组织每个键范围。

To answer general pure-key queries of the form "find the salary history of employee named $k$ ," an index on the key space can be utilized. This index keeps the latest version of a key, while key versions are linked together. Since the key space is separate from the time space, such an index is easily updated. In some methods this index has the form of a ${\mathrm{B}}^{ + }$ -tree,and is also facilitated for transaction range-timeslice queries (such as the surrogate superindex used in the AP-tree [Gunadhi and Segev 1993] and the archivable time index [Verma and Varman 1994]) or it has the form of a hashing function, as in the snapshot index [Tsotras and Kangelaris 1995]. A general method is to link records to any one copy of the most recent distinct past version of the record. We continue with the presentation of various time-only methods.

为了回答“查找名为$k$的员工的薪资历史”这种形式的一般纯键查询，可以利用键空间上的索引。该索引保存键的最新版本，同时键的各个版本相互链接。由于键空间与时间空间是分离的，因此这种索引很容易更新。在某些方法中，这种索引采用${\mathrm{B}}^{ + }$ - 树的形式，并且也便于进行事务范围 - 时间片查询（例如AP树[Gunadhi和Segev 1993]中使用的代理超级索引和可存档时间索引[Verma和Varman 1994]），或者它采用哈希函数的形式，如快照索引[Tsotras和Kangelaris 1995]。一种通用方法是将记录链接到该记录最近不同过去版本的任意一个副本。我们继续介绍各种仅时间方法。

Append-Only Tree. The append-only tree (AP-Tree) is a multiway search tree that is a hybrid of an ISAM index and a ${\mathrm{B}}^{ + }$ -tree. It was proposed as a method to optimize event-joins [Segev and Gunadhi 1989; Gunadhi and Segev 1993]. Here we examine it as an access method for the query classes of section 3.1. Each tuple is associated with a (start_time, end_time) interval. The basic method indexes the start_times of tuples. Each leaf node has entries of the form:(t,b)where $t$ is a time instant and $b$ is a pointer to a bucket that contains all tuples with start_time greater than the time recorded in the previous entry (if any) and less than or equal to $t$ . Each nonleaf node indexes nodes at the next level (Figure 5).

仅追加树。仅追加树（AP树）是一种多路搜索树，它是ISAM索引和${\mathrm{B}}^{ + }$ - 树的混合体。它被提出作为一种优化事件连接的方法[Segev和Gunadhi 1989；Gunadhi和Segev 1993]。在这里，我们将其作为第3.1节查询类别的一种访问方法进行研究。每个元组都与一个（开始时间，结束时间）区间相关联。基本方法对元组的开始时间进行索引。每个叶节点都有如下形式的条目：(t,b)，其中$t$是一个时间点，$b$是一个指向桶的指针，该桶包含所有开始时间大于前一个条目（如果有）中记录的时间且小于或等于$t$的元组。每个非叶节点对下一级的节点进行索引（图5）。

In the AP-tree, insertions of new tuples arrive in increasing start_time order; on this basis, we consider it a transaction-time method. It is also assumed that the end_times of tuples are known when a tuple is inserted in the access method. In that case the update processing is $O\left( 1\right)$ ,since the tuple is inserted ("appended") on the rightmost leaf of the tree. (This is somewhat similar to the procedure used in most commercial systems for loading a sorted file to a multilevel index [Salzberg 1988], except that insertions are now successive instead of batched.) If end_times are not known at insertion but are updated later (as in a transaction-time environment), the index has to be searched for the record that is updated. If the start_time of the updated record is given in the input, then this search is $O\left( {{\log }_{B}n}\right)$ . Otherwise,we could use a hashing function that stores the alive objects only, and for each such object it points to a position in the AP-tree (this is not discussed in the original paper).

在AP树（Append-Only Tree）中，新元组的插入按照开始时间（start_time）递增的顺序进行；基于此，我们将其视为一种事务时间方法。同时假设，当一个元组插入到访问方法中时，其结束时间（end_time）是已知的。在这种情况下，更新处理的复杂度为$O\left( 1\right)$，因为元组是插入（“追加”）到树的最右侧叶子节点上的。（这与大多数商业系统将排序文件加载到多级索引中的过程有些相似[萨尔茨伯格1988]，只是现在的插入是连续的，而不是批量的。）如果插入时结束时间未知，但在后续进行更新（如在事务时间环境中），则必须在索引中搜索要更新的记录。如果输入中给出了更新记录的开始时间，那么这种搜索的复杂度为$O\left( {{\log }_{B}n}\right)$。否则，我们可以使用一个哈希函数，该函数仅存储存活对象，并且对于每个这样的对象，它指向AP树中的一个位置（原文未讨论此内容）。

<!-- Media -->

<!-- figureText: 8 20 8 45 30 ${f}_{1}$ 50, 60 51, 40 45 key, start_time, end_time key, start_time, end_time $k$ . 10 ${h}_{s}$ $m$ , 20 -->

<img src="https://cdn.noedgeai.com/0195c90d-0a33-750f-877f-c1cec48583bb_21.jpg?x=444&y=230&w=759&h=375&r=0"/>

Figure 5. The append-only tree. Leaves include the start_time fields of intervals only. Each leaf points to file pages, with records ordered according to the start_time field. New records are added only at the rightmost leaf of the tree. It is assumed that both endpoints are known for the intervals in this figure.

图5. 仅追加树。叶子节点仅包含区间的开始时间字段。每个叶子节点指向文件页，记录按照开始时间字段排序。新记录仅添加到树的最右侧叶子节点。假设本图中区间的两个端点都是已知的。

<!-- Media -->

To answer a transaction pure-time-slice query for time $t$ ,the AP-tree is first searched for the leaf that contains $t$ . All intervals on the "right" of this leaf have start_times that are larger than $t$ , and thus should not be searched further. However, all intervals on the left of this leaf (i.e., the data file from the beginning until $t$ ) have to be checked for "containing" $t$ . Such a search can be as large as $O\left( {n/B}\right)$ ,since the number of intervals in the tree is proportional to the number of changes in the evolution. Of course, if we assume that the queries are randomly distributed over the entire transaction-time range, half of the leaf nodes, on average, must be searched. The space is $O\left( {n/B}\right)$ .

为了回答针对时间$t$的事务纯时间切片查询，首先在AP树中搜索包含$t$的叶子节点。该叶子节点“右侧”的所有区间的开始时间都大于$t$，因此无需进一步搜索。然而，该叶子节点左侧的所有区间（即从开始到$t$的数据文件）都必须检查是否“包含”$t$。这种搜索的复杂度可能高达$O\left( {n/B}\right)$，因为树中区间的数量与演化过程中的变化数量成正比。当然，如果我们假设查询在整个事务时间范围内随机分布，那么平均而言，必须搜索一半的叶子节点。空间复杂度为$O\left( {n/B}\right)$。

For answering transaction pure-key and range-timeslice queries, the nested ${ST}$ -tree has been proposed [Gunadhi and Segev 1993]. This method facilitates a separate ${\mathrm{B}}^{ + }$ -tree index (called surrogate superindex) on all the keys (surrogates) ever inserted in the database. A leaf node of such a tree contains entries of the form $\left( {\text{key,}\left( {{p}_{1},{p}_{2}}\right) }\right.$ ,where ${p}_{1}$ is a pointer to an AP-tree (called the time subindex) that organizes the evolution of the particular key and ${p}_{2}$ is a pointer to the latest version of a key. This approach solves the problem of updating intervals by key (just search the surrogate superindex for the key of the interval; then this key's time subindex will provide the latest version of this interval, i.e., the version to be updated). The ${ST}$ -tree approach is conceptually equivalent to reverse chaining with an index on each accession list (however, due to its relation to the AP-tree, we include it in the time-only methods).

为了回答事务纯键和范围时间切片查询，有人提出了嵌套的${ST}$树[古纳迪和塞格夫1993]。这种方法为数据库中插入过的所有键（代理键）建立了一个单独的${\mathrm{B}}^{ + }$树索引（称为代理超级索引）。这种树的叶子节点包含形式为$\left( {\text{key,}\left( {{p}_{1},{p}_{2}}\right) }\right.$的条目，其中${p}_{1}$是指向一个AP树（称为时间子索引）的指针，该AP树组织特定键的演化，${p}_{2}$是指向键的最新版本的指针。这种方法解决了按键更新区间的问题（只需在代理超级索引中搜索区间的键；然后该键的时间子索引将提供该区间的最新版本，即要更新的版本）。${ST}$树方法在概念上等同于对每个访问列表使用索引进行反向链接（然而，由于它与AP树的关系，我们将其归入仅时间方法中）。

Update processing is now $O\left( {{\log }_{B}S}\right)$ , where $S$ denotes the total number of keys (surrogates) ever created ( $S$ is itself $O\left( n\right)$ ). Note that there may be key histories with just one record. For the space to remain $O\left( {n/B}\right)$ ,unused page portions should be shared by other key histories. This implies that the versions of a given key may reside in separate pages. Answering a pure key query then takes $O\left( {{\log }_{B}S + a}\right)$ I/Os. The given key can be found with a logarithmic search on the surrogate superindex, and then its $a$ versions are accessed but,at worst, each version may reside in a distinct page. For a transaction range-timeslice query whose range contains $K$ keys (alive or not at $t$ ),the query time is $O\left( {K{\log }_{B}n}\right)$ because each key in the range has to be searched. When the range is the whole key space, i.e., to answer a transaction pure-timeslice query for time $t$ ,we have to perform a logarithmic search on the time subindex of each key ever created. This takes time $O\left( {S{\log }_{B}n}\right)$ .

更新处理现在为 $O\left( {{\log }_{B}S}\right)$，其中 $S$ 表示曾经创建的键（代理键）的总数（$S$ 本身为 $O\left( n\right)$）。请注意，可能存在只有一条记录的键历史。为使空间保持 $O\left( {n/B}\right)$，未使用的页面部分应由其他键历史共享。这意味着给定键的各个版本可能位于不同的页面中。那么，回答一个纯键查询需要 $O\left( {{\log }_{B}S + a}\right)$ 次输入/输出操作。可以通过在代理超索引上进行对数搜索找到给定的键，然后访问其 $a$ 个版本，但在最坏的情况下，每个版本可能位于不同的页面中。对于一个事务范围 - 时间片查询，其范围包含 $K$ 个键（在 $t$ 时刻是否存活），查询时间为 $O\left( {K{\log }_{B}n}\right)$，因为必须搜索该范围内的每个键。当范围是整个键空间时，即回答在时间 $t$ 的事务纯时间片查询时，我们必须对曾经创建的每个键的时间子索引进行对数搜索。这需要 $O\left( {S{\log }_{B}n}\right)$ 的时间。

The basic AP-tree does not separate past from current data, so transferring to a write-once optical disk may be problematic. We could start transferring data to the write-once medium in start_time order, but this could also transfer long-lived tuples that are still current (alive), and may be updated later. The ST-tree does not have this problem because data are clustered by key; the history of each key represents past data that can be transferred to an optical medium.

基本的 AP 树（AP-tree）不会将过去的数据与当前数据分开，因此转移到一次写入光盘可能会有问题。我们可以按照开始时间（start_time）的顺序开始将数据转移到一次写入介质，但这也可能会转移仍然是当前（存活）的长生命周期元组，并且这些元组可能会在以后被更新。ST 树（ST-tree）没有这个问题，因为数据是按键进行聚类的；每个键的历史记录代表可以转移到光学介质的过去数据。

If the AP-tree is used in a valid-time environment, interval insertions, deletions, or updates may happen anywhere in the valid-time domain. This implies that the index will not be as compact as in the transaction domain where changes arrive in order, but it would behave as a B-Tree. If, for each update, only the key associated with the updated interval is provided, the whole index may have to be searched. If the start_time of the updated interval is given, a logarithmic search is needed. Since the $\mathrm{M}l$ valid intervals are sorted by start_time, a "*/point/-" query takes $O\left( {l/B}\right)$ I/Os. For "range/point/-" queries, the ST-tree must be combined with a B-tree as its time subindex. Updates are logarithmic (by traversing the surrogate superindex and the time subindex). A valid range timeslice query whose range contains $K$ keys takes $O\left( {K{\log }_{B}l}\right)$ I/Os, since every key in the query range must be searched for being alive at the valid query time.

如果在有效时间环境中使用 AP 树，区间插入、删除或更新可能会在有效时间域的任何位置发生。这意味着该索引不会像在事务域中那样紧凑，在事务域中更改是按顺序到达的，但它的行为会类似于 B 树（B-Tree）。如果对于每次更新，仅提供与更新区间关联的键，则可能需要搜索整个索引。如果给出了更新区间的开始时间（start_time），则需要进行对数搜索。由于 $\mathrm{M}l$ 个有效区间是按开始时间排序的，一个“*/点/-”查询需要 $O\left( {l/B}\right)$ 次输入/输出操作。对于“范围/点/-”查询，ST 树必须与 B 树结合作为其时间子索引。更新操作是对数级的（通过遍历代理超索引和时间子索引）。一个有效范围时间片查询，其范围包含 $K$ 个键，需要 $O\left( {K{\log }_{B}l}\right)$ 次输入/输出操作，因为必须搜索查询范围内的每个键在有效查询时间是否存活。

Time Index. The time index, proposed in Elmasri et al. [1990; 1991], is a ${\mathrm{B}}^{ + }$ -tree-based access method on the time axis. In the original paper the method was proposed for storing valid-times. But it makes the assumption that changes arrive in increasing time order and that physical deletions rarely occur. Since these are basic characteristics of the transaction-time dimension, we consider the time-index to be in the transaction-time category. There is a ${\mathrm{B}}^{ + }$ -tree that indexes a linearly-ordered set of time points, where a time point (also referred to as an indexing point in Elmasri et al. [1990]) is either the time instant where a new version is created or the next time instant after a version is deleted. Thus, a time point corresponds to the time instant of a change (for deletions it is the next time instant after the deletion). Each entry of a leaf node of the time index is of the form (t,b),where $t$ is a time point and $b$ is a pointer to a bucket. The pointer of a leaf's first entry points to a bucket that holds all records that are "alive" (i.e., a snapshot) at this time point; the rest of the leaf entries point to buckets that hold incremental changes (Figure 6). As a result, the time index does not need to know in advance the end_time of an object (which is an advantage over the AP-tree).

时间索引。时间索引由埃尔马斯里（Elmasri）等人在 1990 年和 1991 年提出，它是一种基于 ${\mathrm{B}}^{ + }$ 树的时间轴访问方法。在原论文中，该方法是为存储有效时间而提出的。但它假设更改是按时间递增顺序到达的，并且很少发生物理删除。由于这些是事务时间维度的基本特征，我们认为时间索引属于事务时间类别。有一个 ${\mathrm{B}}^{ + }$ 树对一组线性排序的时间点进行索引，其中一个时间点（在埃尔马斯里等人 1990 年的论文中也称为索引点）要么是创建新版本的时间点，要么是删除版本后的下一个时间点。因此，一个时间点对应于一次更改的时间点（对于删除操作，它是删除后的下一个时间点）。时间索引的叶节点的每个条目形式为 (t,b)，其中 $t$ 是一个时间点，$b$ 是一个指向桶的指针。叶节点第一个条目的指针指向一个桶，该桶保存了在这个时间点“存活”的所有记录（即一个快照）；叶节点其余条目的指针指向保存增量更改的桶（图 6）。因此，时间索引不需要预先知道对象的结束时间（这是相对于 AP 树的一个优势）。

The time index was originally proposed as a secondary index; but we shall treat it as a primary index here, in order to make a fair comparison to other methods, as explained in Section 4.1. This makes the search estimates competitive with the other methods without changing the worst-case asymptotic space and update formulas.

时间索引最初被提议作为二级索引；但为了与其他方法进行公平比较，正如4.1节所解释的，我们在这里将其视为一级索引。这使得搜索估计结果与其他方法具有竞争力，同时不改变最坏情况下的渐近空间和更新公式。

<!-- Media -->

<!-- figureText: time 324 369 464 499 598 alive insert 112387 789999 306735 723875 654783 286548 565483 update 276549 119875 119875 239765 insert 892365 332244 762341 654783 324987 654389 233388 222333 879654 127658 221 243 265 302 312 alive records insert 112387 233388 306735 723875 delete 286548 347865 565483 276549 delete 119875 987456 239765 987456 892365 insert 762341 222333 654783 324987 654389 345234 347865 879654 127658 -->

<img src="https://cdn.noedgeai.com/0195c90d-0a33-750f-877f-c1cec48583bb_23.jpg?x=220&y=227&w=545&h=715&r=0"/>

Figure 6. The time index. Each first leaf entry holds a full timeslice, while the next entries keep incremental changes.

图6. 时间索引。每个第一个叶子条目保存一个完整的时间片，而后续条目保存增量变化。

<!-- Media -->

Since in a transaction environment changes occur in increasing time order, new nodes are always added on the rightmost leaf of the index. This can produce a more compact index than the $\mathrm{B} +$ tree in the original paper. The new index is called the monotonic ${B}^{ + }$ -tree [Elmasri et al. 1993] (the monotonic ${\mathrm{B}}^{ + }$ - tree insertion algorithm is similar to that of the AP-tree).

由于在事务环境中，变化按时间递增顺序发生，因此新节点总是添加到索引的最右侧叶子节点上。这可以产生比原论文中的$\mathrm{B} +$树更紧凑的索引。新的索引称为单调${B}^{ + }$ - 树[Elmasri等人，1993年]（单调${\mathrm{B}}^{ + }$ - 树插入算法与AP - 树的插入算法类似）。

To answer a transaction pure-time-slice query for some time $t$ ,we have to search the time index for $t$ ; this leads to a leaf node that "contains" $t$ . The past state is reconstructed by accessing all the buckets of entries of this leaf node that contain timestamps that are less or equal to $t$ . If we assume that the number of changes that can occur at each time instant is bounded (by some constant), the query time of the time index is $O\left( {{\log }_{B}n + a/B}\right)$ . After the appropriate leaf node is found in logarithmic time,the answer $a$ is reconstructed by reading leaf buckets. The update processing and space can be as large as $O\left( {n/B}\right)$ and $O\left( {{n}^{2}/B}\right)$ ,respectively. Therefore, this method is conceptually equivalent to the "copy" approach of Section 4.1 (the only difference is that copies are now made after a constant number of changes).

为了回答某个时间$t$的事务纯时间片查询，我们必须在时间索引中搜索$t$；这会指向一个“包含”$t$的叶子节点。通过访问该叶子节点中所有包含小于或等于$t$的时间戳的条目桶来重建过去的状态。如果我们假设每个时间点可能发生的变化数量是有界的（由某个常数界定），则时间索引的查询时间为$O\left( {{\log }_{B}n + a/B}\right)$。在以对数时间找到合适的叶子节点后，通过读取叶子桶来重建答案$a$。更新处理和空间分别可能高达$O\left( {n/B}\right)$和$O\left( {{n}^{2}/B}\right)$。因此，这种方法在概念上等同于4.1节中的“复制”方法（唯一的区别是现在在固定数量的变化之后进行复制）。

Answering a transaction range-time-slice query with the time index requires reconstructing the whole timeslice for the time of interest and then selecting only the tuples in the given range. To answer range-timeslice queries more efficiently, the two-level attribute/time index (using predefined key regions) was proposed in Elmasri et al. [1990]. Assuming that there are $R$ predefined key regions (and $R$ is smaller than $n$ ),the update processing and space remain $O\left( {n/B}\right)$ and $O\left( {{n}^{2}/B}\right)$ ,respectively— since most of the changes can happen to a single region. Answering a "*/-/point" query means creating the timeslices for all $R$ ranges,even if a range does not contribute to the answer. Thus the pure-timeslice query time is proportional to ${\sum }_{i = 1}^{R}{\log }_{B}{n}_{i} + {a}_{i}/B$ ,where ${n}_{i}$ and ${a}_{i}$ correspond to the total number of changes in individual region ${r}_{i}$ and the number of "alive" objects from ${r}_{i}$ ,respectively,at time $t$ . This can be as high as $O\left( {R{\log }_{B}n + a}\right)$ ,since each region can contribute a single tuple to the answer. Similarly, for "range/-/point" queries the query time becomes $O\left( {M{\log }_{B}n + a}\right)$ , where $M$ is the number of regions that fall in the given query range (assuming that the query range contains a number of regions, otherwise a whole region timeslice has to be created).

使用时间索引回答事务范围时间片查询需要为感兴趣的时间重建整个时间片，然后仅选择给定范围内的元组。为了更有效地回答范围时间片查询，Elmasri等人[1990年]提出了两级属性/时间索引（使用预定义的键区域）。假设存在$R$个预定义的键区域（并且$R$小于$n$），更新处理和空间分别保持为$O\left( {n/B}\right)$和$O\left( {{n}^{2}/B}\right)$ —— 因为大多数变化可能发生在单个区域。回答“*/ - /点”查询意味着为所有$R$个范围创建时间片，即使某个范围对答案没有贡献。因此，纯时间片查询时间与${\sum }_{i = 1}^{R}{\log }_{B}{n}_{i} + {a}_{i}/B$成正比，其中${n}_{i}$和${a}_{i}$分别对应于时间$t$时单个区域${r}_{i}$中的总变化数量和来自${r}_{i}$的“活跃”对象数量。这可能高达$O\left( {R{\log }_{B}n + a}\right)$，因为每个区域可以为答案贡献一个元组。类似地，对于“范围/ - /点”查询，查询时间变为$O\left( {M{\log }_{B}n + a}\right)$，其中$M$是落在给定查询范围内的区域数量（假设查询范围包含多个区域，否则必须创建整个区域的时间片）。

Pure-key queries are not supported, as record versions of the same object are not linked (for example, to answer a query of the form: "find all past versions of a given key," we may have to search the whole history of the range to find where this key belongs).

不支持纯键查询，因为同一对象的记录版本没有关联（例如，要回答“查找给定键的所有过去版本”形式的查询，我们可能必须搜索该范围的整个历史记录以找到该键所属的位置）。

In Elmasri et al. [1993], it is suggested we move record versions to optical disk when their end times change to a time before now. This is under the assumption that the time index is being used as a secondary index and that each record version is only located in one place. So the leaf buckets contain lists of addresses of record versions.

在Elmasri等人[1993年]的研究中，建议当记录版本的结束时间变为当前时间之前的某个时间时，将这些记录版本移动到光盘上。这是基于时间索引被用作二级索引，并且每个记录版本仅位于一个位置的假设。因此，叶子桶包含记录版本的地址列表。

In order to move full pages of data to the optical disk, a buffer is used in the magnetic disk to collect records as their end times are changed. An optical disk page is reserved for the contents of each buffer page. When a record version is placed in a buffer page, all pointers to it in the time index must be changed to refer to its new page in the optical disk. This can require $O\left( {n/B}\right)$ update processing, as a record version pointer can be contained in $O\left( {n/B}\right)$ leaves of the time index. A method for finding pointers for particular record versions within the lists of addresses in the leafs first entry, in order to update them, is not given.

为了将整页数据移动到光盘上，磁盘中使用了一个缓冲区来收集记录，同时更改它们的结束时间。每个缓冲区页面的内容都会在光盘上预留一个页面。当一个记录版本被放入缓冲区页面时，时间索引中所有指向它的指针都必须更改为指向它在光盘上的新页面。这可能需要$O\left( {n/B}\right)$次更新处理，因为一个记录版本指针可能包含在时间索引的$O\left( {n/B}\right)$个叶子节点中。文中并未给出一种在叶子节点首条记录的地址列表中查找特定记录版本指针以进行更新的方法。

Index leaf pages can be migrated to the optical disk only when all their pointers are references to record versions on the optical disk or in the magnetic disk buffer used to transfer record versions to optical disk. Since each index leaf page contains the pointers to all record versions alive at the time the index page was created, it is likely that many index pages may not qualify for moving to optical disk because they contain long-lived records.

只有当索引叶子页面的所有指针都指向光盘上的记录版本，或者指向用于将记录版本传输到光盘的磁盘缓冲区中的记录版本时，这些索引叶子页面才能迁移到光盘上。由于每个索引叶子页面都包含指向在创建该索引页面时所有存活记录版本的指针，因此许多索引页面可能不符合迁移到光盘的条件，因为它们包含长期存在的记录。

It is suggested in Elmasri et al. [1993] that long-lived records inhibiting the movement of index pages also be kept in a magnetic buffer and assigned an optical address, so that the index leaf page can be moved. When all the children of an internal index page have been moved to the optical disk, an internal index page can also be moved. However, the number of long-lived record versions can also be $O\left( n\right)$ . Thus the number of empty optical pages waiting for long-lived object versions to die and having mirror buffers on magnetic disk is $O\left( {n/B}\right)$ .

埃尔马斯里（Elmasri）等人在1993年的研究中建议，将阻碍索引页面移动的长期存在的记录也保留在磁盘缓冲区中，并为其分配一个光盘地址，这样索引叶子页面就可以移动了。当内部索引页面的所有子页面都已移动到光盘上时，该内部索引页面也可以移动。然而，长期存在的记录版本的数量也可能达到$O\left( n\right)$。因此，等待长期存在的对象版本过期的空光盘页面数量，以及磁盘上的镜像缓冲区数量为$O\left( {n/B}\right)$。

In an attempt to overcome the high storage and update requirements, the time index ${}^{ + }$ [Kourmajian et al. 1994] has been proposed. There are two new structures in the time index ${}^{ + }$ : the ${SCS}$ and the ${SCI}$ buckets. In the original time index, a timeslice is stored for the first timestamp entry of each leaf node. Since sibling leaf nodes may share much of this timeslice, in the time in- ${\text{dex}}^{ + }$ ,odd-even pairs of sibling nodes store their common parts of the time-slice in a shared SCS bucket. Even though the SCS technique would in practice save considerable space (about half of what was used before), the asymptotic behavior remains the same as the original time index.

为了克服高存储和高更新要求，有人提出了时间索引${}^{ + }$（库尔马吉安（Kourmajian）等人，1994年）。时间索引${}^{ + }$中有两种新结构：${SCS}$桶和${SCI}$桶。在原始的时间索引中，每个叶子节点的第一个时间戳条目都会存储一个时间片。由于兄弟叶子节点可能共享这个时间片的大部分内容，在时间索引${\text{dex}}^{ + }$中，兄弟节点的奇偶对会将它们时间片的公共部分存储在一个共享的SCS桶中。尽管SCS技术在实践中可以节省大量空间（大约是之前使用空间的一半），但其渐近行为与原始时间索引保持相同。

Common intervals that span a number of leaf nodes are stored together on some parent index node (similar to the segment tree data structure [Bentley 1977]). Each index node in the time index ${}^{ + }$ is associated with a range,i.e., the range of time instants covered by its subtree. A time interval $I$ is stored in the highest internal node $v$ such that $I$ covers $v$ ’s range and does not cover the range of $v$ ’s parent. All such intervals are kept in the ${SCI}$ bucket of an index node.

跨越多个叶子节点的公共时间间隔会一起存储在某个父索引节点上（类似于线段树数据结构，本特利（Bentley），1977年）。时间索引${}^{ + }$中的每个索引节点都与一个范围相关联，即其对应子树所覆盖的时间点范围。一个时间间隔$I$会存储在最高级的内部节点$v$中，使得$I$覆盖$v$的范围，但不覆盖$v$父节点的范围。所有这些时间间隔都保存在索引节点的${SCI}$桶中。

By keeping the intervals in this way, quadratic space is dramatically reduced. Observe that an interval may now be stored in, at most, logarithmic many internal nodes (due to the segment tree property [Mehlhorn 1984]). This implies that the space consumption of the time index ${}^{ + }$ is reduced to $O\left( {\left( {n/B}\right) {\log }_{B}n}\right)$ space. The authors mention in the paper that in practice there is no need to associate ${SCI}$ buckets to more than two-levels of index nodes. However, if no SCI buckets are used at the higher levels, asymptotic behavior remains similar to the original time index.

通过这种方式存储时间间隔，二次空间消耗得到了显著降低。可以观察到，一个时间间隔现在最多存储在对数级数量的内部节点中（由于线段树的性质，梅尔霍恩（Mehlhorn），1984年）。这意味着时间索引${}^{ + }$的空间消耗降低到了$O\left( {\left( {n/B}\right) {\log }_{B}n}\right)$空间。论文作者提到，在实践中，不需要为超过两层的索引节点关联${SCI}$桶。然而，如果在较高级别不使用SCI桶，其渐近行为将与原始时间索引相似。

In addition, it is not clear how updates are performed when ${SCI}$ buckets are used. In order to find the actual ${SCI}$ s where a given interval is to be stored, both endpoints of the interval should be known. Otherwise, if an interval is initially inserted as(t,now),it has to be found and updated when, at a later time, the right endpoint becomes known. This implies that some search structure is needed in each ${SCI}$ ,which would, of course, affect the update behavior of the whole structure. Finally, the query time bound remains the same for the time index ${}^{ + }$ as for the original time index.

此外，目前尚不清楚使用${SCI}$桶时如何进行更新操作。为了找到存储给定时间间隔的实际${SCI}$，需要知道该时间间隔的两个端点。否则，如果一个时间间隔最初以(t,now)的形式插入，那么当稍后知道其右端点时，就必须找到并更新该时间间隔。这意味着每个${SCI}$中都需要某种搜索结构，这当然会影响整个结构的更新行为。最后，时间索引${}^{ + }$的查询时间界限与原始时间索引相同。

If the original time-index (using the regular $\mathrm{B} +$ tree) is used in a valid-time environment, physical object deletions anywhere in the (valid) time domain should be supported. However, a deleted object should be removed from all the stored (valid) snapshots. If the deleted object has a long valid-time interval, the whole structure may have to be updated, making such deletions very costly. Similarly, objects can be added anywhere in the valid domain, implying that all affected stored snapshots have to be updated.

如果在有效时间环境中使用原始时间索引（使用常规的$\mathrm{B} +$树），则应支持在（有效）时间域内任意位置对物理对象进行删除操作。然而，已删除的对象应从所有已存储的（有效）快照中移除。如果被删除对象的有效时间间隔较长，则可能需要更新整个结构，这使得此类删除操作的成本非常高。同样，对象可以在有效域内的任意位置添加，这意味着所有受影响的已存储快照都必须进行更新。

Differential File Approach. While the differential file approach [Jensen et al. 1991;1992] does not propose the creation of a new index, we discuss it, since it involves an interesting implementation of a database system based on transaction time. In practice, an index can be implemented on top of the differential file approach, however, here we assume no such index exists. Changes that occur for a base relation $r$ are stored incrementally and timestamped on the relation's log; this log is itself considered a special relation, called a backlog. In addition to the attributes of the base relation, each entry of the backlog contains a triplet: (time, key, op). Here time corresponds to the (commit) time of the transaction that updated the database about a change that was applied on the base relation tuple with key surrogate; op corresponds to the kind of change that was applied on this tuple (addition, deletion, or modification operations).

差异文件方法。虽然差异文件方法[Jensen等人，1991；1992]并未提议创建新的索引，但我们仍对其进行讨论，因为它涉及一种基于事务时间的数据库系统的有趣实现方式。实际上，可以在差异文件方法的基础上实现索引，但在此我们假设不存在这样的索引。基础关系$r$发生的更改会被增量存储，并在该关系的日志中添加时间戳；此日志本身被视为一种特殊的关系，称为积压日志。除了基础关系的属性外，积压日志的每个条目都包含一个三元组：（时间，键，操作）。这里的时间对应于更新数据库的事务的（提交）时间，该事务针对具有键代理的基础关系元组应用了更改；操作对应于对该元组应用的更改类型（添加、删除或修改操作）。

As a consequence of the use of timestamps, a base relation is a function of time; thus $r\left( t\right)$ is a timeslice of the base relation at time $t$ . A timeslice of a base relation can be stored or computed. Storing a timeslice can be implemented as either a cache (where pointers to the appropriate backlog entries are used) or as materialized data (where the actual tuples of the timeslice are kept). Using the cache avoids storing (probably) long attributes, but some time is needed to reconstruct the full timeslice (Figure 7).

由于使用了时间戳，基础关系是时间的函数；因此$r\left( t\right)$是基础关系在时间$t$的一个时间切片。基础关系的时间切片可以被存储或计算得出。存储时间切片可以实现为缓存（使用指向相应积压日志条目的指针），也可以实现为物化数据（保留时间切片的实际元组）。使用缓存可以避免存储（可能）较长的属性，但需要一些时间来重构完整的时间切片（图7）。

A timeslice can be fixed (for example, $\left. {r\left( {t}_{1}\right) }\right)$ or time-dependent $\left( {r\left( {\text{now } - {t}_{1}}\right) }\right)$ . Time-dependent stored base relations have to be updated; this is done eagerly (changes directly update such relations) or lazily (when the relation is requested, the backlog is used to bring it up in the current state). An eager current $\left( {r\left( {now}\right) }\right)$ timeslice is like a snapshot relation, that is, a collection of all current records.

时间切片可以是固定的（例如，$\left. {r\left( {t}_{1}\right) }\right)$）或与时间相关的$\left( {r\left( {\text{now } - {t}_{1}}\right) }\right)$。与时间相关的已存储基础关系必须进行更新；这可以通过主动方式（更改直接更新此类关系）或被动方式（当请求该关系时，使用积压日志将其更新到当前状态）来完成。主动的当前$\left( {r\left( {now}\right) }\right)$时间切片类似于快照关系，即所有当前记录的集合。

A time-dependent base relation can also be computed from a previous stored timeslice and the set of changes that occurred in between. These changes correspond to a differential file (instead of searching the whole backlog). Differential files are also stored as relations.

与时间相关的基础关系也可以从先前存储的时间切片以及其间发生的一组更改计算得出。这些更改对应于一个差异文件（而不是搜索整个积压日志）。差异文件也作为关系进行存储。

For answering "*/-/point" queries, this approach can be conceptually equivalent to the "log" or "copy" methods, depending on how often timeslices are stored. Consider, for example, a single base relation $r$ with backlog ${b}_{r}$ : if time-slices are infrequent or the distance (number of changes) between timeslices is not fixed, the method is equivalent to the "log" approach where ${b}_{r}$ is the history log. The space is $O\left( {n/B}\right)$ and update processing is constant (amortized) per change, but the reconstruction can also be $O\left( {n/B}\right)$ . Conversely,if timeslices are kept with fixed distance, the method will behave similarly to the "copy" approach.

对于回答“*/-/点”查询，根据时间片的存储频率，这种方法在概念上可能等同于“日志”或“复制”方法。例如，考虑一个具有积压 ${b}_{r}$ 的单个基础关系 $r$：如果时间片不频繁，或者时间片之间的距离（更改次数）不固定，那么该方法等同于“日志”方法，其中 ${b}_{r}$ 是历史日志。空间复杂度为 $O\left( {n/B}\right)$，每次更改的更新处理是常数级（均摊）的，但重建复杂度也可能是 $O\left( {n/B}\right)$。相反，如果以固定距离保存时间片，该方法的行为将类似于“复制”方法。

In order to address "range/-/point" queries, we have to produce the time-slice of the base relation and then check all of the tuples of this timeslice for being in the query range. Similarly, if the value of a given key is requested as of some time, the whole relation must first be reconstructed as of that time. The history (previous versions) of a key is not kept explicitly, as versions of a given key are not connected together.

为了处理“范围/-/点”查询，我们必须生成基础关系的时间片，然后检查该时间片的所有元组是否在查询范围内。同样，如果请求某个时间点给定键的值，则必须首先重建该时间点的整个关系。由于给定键的各个版本没有关联在一起，因此不会显式保存键的历史（先前版本）。

<!-- Media -->

<!-- figureText: insert delete delete insert update update 196534 565483 the backlog alive delete 股份 654783 19875 112387 306735 723875 286548 565483 276549 update 119875 879654 239765 762341 789999 332244 324987 345234 233388 222333 876888 127658 current database 233388 347865 987456 222333 alive records 112387 789999 306735 723875 286548 565483 276549 687654 119875 insert 239765 332244 987456 892365 762341 654783 324987 654389 345234 347865 879654 127658 answer to previous query; may be deleted in future. -->

<img src="https://cdn.noedgeai.com/0195c90d-0a33-750f-877f-c1cec48583bb_26.jpg?x=436&y=230&w=754&h=715&r=0"/>

Figure 7. Differential file approach.

图 7. 差异文件方法。

<!-- Media -->

Checkpoint Index. The checkpoint index was originally proposed for the implementation of various temporal operators (temporal joins, parallel temporal joins, snapshot/interval operators, etc.) [Leung and Muntz 1992a; 1992b; 1993]. Here we take the liberty of considering its behavior as if it was used as an access method for transaction-time queries. Timeslices (called checkpoints) are periodically taken from the state of an evolving relation. If the query operator is a join, checkpoints from two relations are taken. Partial relation checkpoints based on some key predicate have also been proposed. For simplicity, we concentrate on checkpointing a single relation.

检查点索引。检查点索引最初是为实现各种时态运算符（时态连接、并行时态连接、快照/区间运算符等）而提出的 [Leung 和 Muntz 1992a; 1992b; 1993]。在这里，我们不妨将其行为视为用于事务时间查询的访问方法。定期从不断演变的关系状态中获取时间片（称为检查点）。如果查询运算符是连接操作，则会从两个关系中获取检查点。也有人提出了基于某些键谓词的部分关系检查点。为简单起见，我们专注于对单个关系进行检查点操作。

The checkpoint index assumes that object intervals are ordered by their start_time. This is a property of the transaction-time environment (Figure 1). A stream processor follows the evolution as time proceeds. When a checkpoint is made at some (checkpoint) instant $t$ ,the objects alive at $t$ are stored in the checkpoint. A separate structure, called the data stream pointer (DSP), points to the first object born after $t$ . Conceptually, the DSP provides access to an ordered (by interval start_time) list of objects born between checkpoints. The DSP is needed, since some of these objects may end before the next checkpoint, and thus would not be recorded otherwise. The checkpoint time instants are indexed through a B+-tree-like structure (Figure 8).

检查点索引假设对象区间按其开始时间排序。这是事务时间环境的一个特性（图 1）。随着时间的推移，流处理器跟踪演变过程。当在某个（检查点）时刻 $t$ 进行检查点操作时，在 $t$ 时刻存活的对象会存储在检查点中。一个单独的结构，称为数据流指针（DSP），指向在 $t$ 之后诞生的第一个对象。从概念上讲，DSP 提供了对检查点之间诞生的对象的有序（按区间开始时间）列表的访问。需要 DSP 是因为这些对象中的一些可能在下一个检查点之前结束，否则将不会被记录。检查点时刻通过类似 B + 树的结构进行索引（图 8）。

The performance of the checkpoint index for pure-timeslice queries depends on how often checkpoints are taken. At one extreme, if very few checkpoints are taken,the space remains linear $O\left( {n/B}\right)$ . Conversely, if checkpoints are kept within a fixed distance, the method behaves similarly to the "copy" approach. In general, the DSP pointer may be "reset" backwards in time to reduce the size of a checkpoint (which is an optimization issue).

检查点索引用于纯时间片查询的性能取决于检查点的获取频率。在一种极端情况下，如果获取的检查点非常少，空间复杂度保持线性 $O\left( {n/B}\right)$。相反，如果以固定距离保存检查点，该方法的行为类似于“复制”方法。一般来说，可以将 DSP 指针在时间上向后“重置”以减小检查点的大小（这是一个优化问题）。

When an object is deleted, its record has to be found to update the end_time. The original presentation of the checkpoint index implicitly assumes that the object end_times are known (since the whole evolution or stream is known). However, a hashing function on alive objects can be used to solve this problem (as with the AP-tree). The checkpoint index resembles the differential file and the time indexes, in that all of them keep various timeslices. However, instead of simply storing the changes between timeslices, the checkpoint index keeps the DSP pointers to actual object records. Hence, in the checkpoint index, an update to an interval end_time cannot simply be added at the end of a log, but it has to update the corresponding object's record.

当删除一个对象时，必须找到其记录以更新结束时间。检查点索引的原始表述隐含地假设对象的结束时间是已知的（因为整个演变过程或流是已知的）。然而，可以使用存活对象的哈希函数来解决这个问题（与 AP 树类似）。检查点索引与差异文件和时间索引类似，因为它们都保存了各种时间片。然而，检查点索引不是简单地存储时间片之间的更改，而是保存指向实际对象记录的 DSP 指针。因此，在检查点索引中，对区间结束时间的更新不能简单地添加到日志末尾，而是必须更新相应对象的记录。

<!-- Media -->

<!-- figureText: ${t}_{1}/{t}_{4}$ ${t}_{4}$ Checkpoint( ${t}_{7}$ ) Checkpoint( ${t}_{9}$ ) $\{ a,f,c\}$ (a, f,g) $\operatorname{DSP}\left( {t}_{7}\right)$ $\operatorname{DSP}\left( {t}_{9}\right)$ \{g\} $\{ e\}$ Checkpoint( ${t}_{1}$ ) Checkpoint $\left( {t}_{4}\right)$ (a) $\{ a,h,b,f\}$ $\operatorname{DSP}\left( {t}_{1}\right)$ $\operatorname{DSP}\left( {t}_{4}\right)$ $\{ h\}$ $\{ c\}$ -->

<img src="https://cdn.noedgeai.com/0195c90d-0a33-750f-877f-c1cec48583bb_27.jpg?x=481&y=237&w=686&h=326&r=0"/>

Figure 8. The checkpoint index. The evolution at Figure 1 is assumed.

图 8. 检查点索引。假设采用图 1 中的演变过程。

<!-- Media -->

To address range-timeslice queries with the checkpoint index, the timeslice of the base relation is first produced and then all of the tuples of this timeslice are checked for being in the query range. The history (previous versions) of a given key is not kept explicitly because versions of the same key are not connected together.

为了使用检查点索引处理范围 - 时间片查询，首先生成基础关系的时间片，然后检查该时间片的所有元组是否在查询范围内。由于相同键的各个版本没有关联在一起，因此不会显式保存给定键的历史（先前版本）。

The checkpoint index could use a transfer policy to an optical medium similar to the one for the time index.

检查点索引可以使用类似于时间索引的向光介质的转移策略。

Archivable Time Index. The ar-chivable time index [Verma and Varman 1994] does not directly index actual transaction time instants, but does index version numbers. The transaction time instant of the first change takes version number 0 , and successive changes are mapped to consecutive version numbers. An interval is represented by the version numbers corresponding to its start and end times. A special structure is needed to transform versions to timestamps and vice versa. For the rest, we use the terms time instant and version number synonymously.

可存档时间索引。可存档时间索引[Verma和Varman 1994]并不直接对实际事务时间点进行索引，而是对版本号进行索引。第一次变更的事务时间点的版本号为0，后续的变更依次映射到连续的版本号。一个时间间隔由其开始和结束时间对应的版本号表示。需要一种特殊的结构来实现版本号和时间戳之间的相互转换。在其余部分，我们将时间点和版本号视为同义词。

Let ${T}_{c}$ denote the current time. The method partitions records to current and past records. For the current records (those with unknown end_time), a conventional $\mathrm{B} +$ -tree structure is used to index the start_time of their transaction intervals. For past records (records whose end_time is less or equal to ${T}_{c}$ ),a more complex structure,PVAS, is used. Conceptually, the PVAS can be viewed as a logical binary tree of size ${2}^{a}$ $\left( {{T}_{c} \leq  {2}^{a}}\right)$ . Each node in the tree represents a segment of the transaction time space. At ${T}_{c}$ ,only some of the nodes of the tree have been created; new nodes are dynamically added on the right path of the tree as time increases. A node denoted by segment $\left\lbrack  {i,j}\right\rbrack$ where $i < j$ has $\operatorname{span}\left( {j - 1}\right)$ . The root is denoted $\left\lbrack  {0,{2}^{a}}\right\rbrack$ . The left child of node $\left\lbrack  {i,j}\right\rbrack$ is node $\left\lbrack  {i,\left( {i + j}\right) /2}\right\rbrack$ and its right child is node $\left\lbrack  {\left( {i + j}\right) /2,j}\right\rbrack$ . Hence,the span of a node is the sum of the span of its two children. The span of a leaf node is two. Figure 9 (from Verma and Varman [1994]) shows an example of the PVAS tree at ${T}_{c} = {55}$ with $a = 6$ . Node segments appear inside the nodes.

设${T}_{c}$表示当前时间。该方法将记录划分为当前记录和过去记录。对于当前记录（那些结束时间未知的记录），使用传统的$\mathrm{B} +$ -树结构来索引其事务间隔的开始时间。对于过去记录（结束时间小于或等于${T}_{c}$的记录），使用一种更复杂的结构，即PVAS。从概念上讲，PVAS可以看作是一个大小为${2}^{a}$ $\left( {{T}_{c} \leq  {2}^{a}}\right)$的逻辑二叉树。树中的每个节点代表事务时间空间的一个片段。在${T}_{c}$时刻，树中只有部分节点被创建；随着时间的推移，新节点会动态地添加到树的右路径上。由片段$\left\lbrack  {i,j}\right\rbrack$（其中$i < j$）表示的节点具有$\operatorname{span}\left( {j - 1}\right)$。根节点表示为$\left\lbrack  {0,{2}^{a}}\right\rbrack$。节点$\left\lbrack  {i,j}\right\rbrack$的左子节点是节点$\left\lbrack  {i,\left( {i + j}\right) /2}\right\rbrack$，右子节点是节点$\left\lbrack  {\left( {i + j}\right) /2,j}\right\rbrack$。因此，一个节点的跨度是其两个子节点跨度之和。叶节点的跨度为2。图9（来自Verma和Varman [1994]）展示了在${T}_{c} = {55}$时刻，当$a = 6$时PVAS树的一个示例。节点片段显示在节点内部。

Past records are stored in the nodes of this tree. Each record is stored in exactly one node: the lowest node whose span contains the record's interval. For example, a record with interval [3,16] is assigned to node $\left\lbrack  {0,{16}}\right\rbrack$ . The nodes of the binary tree are partitioned into three disjoint sets: passive, active, and future nodes. A node is passive if no more records can ever be stored in that node. It is an active node if it is possible for a record with intervals ending in ${T}_{c}$ to be stored there. It is a future node if it can only store records whose intervals end after ${T}_{c}$ ,i.e.,in the future. Initially,all nodes begin as future nodes at ${T}_{c} = 0$ , then become active, and finally, as time proceeds, end up as passive nodes. Node $\left\lbrack  {i,j}\right\rbrack$ becomes ${T}_{c} = \left( {i + j}\right) /2$ active if it is a leaf,or at ${T}_{c} = \left( {i + j}\right) /2 + 1$ otherwise. For example,in Figure 9,for ${T}_{c}$ $= {55}$ ,node $\left\lbrack  {{48},{64}}\right\rbrack$ belongs to the future nodes. This is because any record with interval contained in $\left\lbrack  {{48},{55}}\right\rbrack$ will be stored somewhere in its left subtree. The only records that can be stored in $\left\lbrack  {{48},{64}}\right\rbrack$ have intervals ending after time 55, so they are future records. Future nodes need not be kept in the tree before becoming active.

过去记录存储在这棵树的节点中。每条记录恰好存储在一个节点中：其跨度包含该记录时间间隔的最低节点。例如，时间间隔为[3,16]的记录被分配到节点$\left\lbrack  {0,{16}}\right\rbrack$。二叉树的节点被划分为三个不相交的集合：被动节点、活动节点和未来节点。如果一个节点不能再存储更多记录，则该节点为被动节点。如果有可能存储结束时间在${T}_{c}$的记录，则该节点为活动节点。如果一个节点只能存储结束时间在${T}_{c}$之后（即未来）的记录，则该节点为未来节点。最初，在${T}_{c} = 0$时刻，所有节点都是未来节点，然后变为活动节点，最后，随着时间的推移，最终成为被动节点。如果节点$\left\lbrack  {i,j}\right\rbrack$是叶节点，则在${T}_{c} = \left( {i + j}\right) /2$时刻变为活动节点；否则在${T}_{c} = \left( {i + j}\right) /2 + 1$时刻变为活动节点。例如，在图9中，对于${T}_{c}$ $= {55}$，节点$\left\lbrack  {{48},{64}}\right\rbrack$属于未来节点。这是因为任何时间间隔包含在$\left\lbrack  {{48},{55}}\right\rbrack$内的记录都会存储在其左子树的某个位置。唯一可以存储在$\left\lbrack  {{48},{64}}\right\rbrack$中的记录是结束时间在55之后的记录，因此它们是未来记录。未来节点在变为活动节点之前不必保留在树中。

<!-- Media -->

<!-- figureText: Passive active $\left\lbrack  {0,{64}}\right\rbrack$ active [32,64] future [32,48 [48,64] future active [48,56] [52,56] Passive active [54,56] $\left\lbrack  {0,{32}}\right\rbrack$ Interval stored at this node [3,16] 16,32] [0,16] ... ... -->

<img src="https://cdn.noedgeai.com/0195c90d-0a33-750f-877f-c1cec48583bb_28.jpg?x=443&y=225&w=750&h=445&r=0"/>

Figure 9. The PVAS binary tree. The current logical time is 55 .

图9. PVAS二叉树。当前逻辑时间为55。

<!-- Media -->

Each interval assigned to a PVAS node is stored in two lists, one that stores the intervals in increasing start-time order and one that stores them in increasing end_time order. This is similar to the interval tree [Edelsbrunner 1983]. In Verma and Varman [1994], a different structure is used to implement these lists for the active and passive nodes by exploiting the fact that passive nodes do not get any new intervals after they become passive. In particular, all passive node lists can be stored in two sequential files (the IFILE and the JFILE), a property that provides for good pagination and record clustering. Two dynamic structures, the ITREE (a B-tree structure) and JLISTS (a collection of lists) are used for the active node lists.

分配给PVAS（可能是某种特定的数据结构，原文未明确展开）节点的每个区间都存储在两个列表中，一个列表按开始时间递增的顺序存储这些区间，另一个列表按结束时间递增的顺序存储它们。这与区间树[埃德尔布鲁纳1983年]类似。在维尔马和瓦尔曼[1994年]的研究中，利用被动节点在变为被动状态后不会再获得任何新区间这一事实，采用了一种不同的结构来为活动节点和被动节点实现这些列表。具体而言，所有被动节点列表可以存储在两个顺序文件（IFILE和JFILE）中，这一特性有利于分页和记录聚类。两个动态结构，即ITREE（一种B树结构）和JLISTS（列表集合）用于活动节点列表。

The PVAS logical binary tree and its accompanied structures can be efficiently placed in pages (details in Verma and Varman [1994]) occupying $O\left( {n/B}\right)$ space. Since the structure does not index record keys, the update assumes that the start_time of the updated record is known; then updating is $O\left( {{\log }_{B}n}\right)$ . As with most of the other time-only methods, if updates are provided by the record key only, a hashing function can be used to find the start_time of the record before the update proceeds on the PVAS.

PVAS逻辑二叉树及其附带结构可以高效地放置在页面中（具体细节见维尔马和瓦尔曼[1994年]），占用$O\left( {n/B}\right)$空间。由于该结构不索引记录键，更新操作假定已知更新记录的开始时间；那么更新操作的复杂度为$O\left( {{\log }_{B}n}\right)$。与大多数其他仅处理时间的方法一样，如果仅通过记录键进行更新，则可以在PVAS上进行更新之前使用哈希函数来查找记录的开始时间。

To answer a transaction pure-time-slice query, both the CVAS and the PVAS are searched. Since the CVAS is ordered on start_times, a logarithmic search will provide the current records born before query time $t$ . Searching the PVAS structure is more complicated. The search follows a single path down the logical binary tree, and the lists of nodes whose span contains $t$ are searched sequentially. Searching each list provides a clustered answer, but there may be $O\left( {{\log }_{2}n}\right)$ binary nodes whose lists are searched. Since every list access may be a separate $\mathrm{I}/\mathrm{O}$ ,the query time becomes $O\left( {{\log }_{2}n + a/B}\right)$ .

为了回答事务纯时间片查询，需要同时搜索CVAS（可能是某种特定的数据结构，原文未明确展开）和PVAS。由于CVAS是按开始时间排序的，因此对数搜索将提供在查询时间$t$之前创建的当前记录。搜索PVAS结构则更为复杂。搜索沿着逻辑二叉树的单一路径向下进行，并顺序搜索其跨度包含$t$的节点列表。搜索每个列表会得到一个聚类答案，但可能有$O\left( {{\log }_{2}n}\right)$个二叉节点的列表需要搜索。由于每次列表访问可能是一次单独的$\mathrm{I}/\mathrm{O}$，因此查询时间变为$O\left( {{\log }_{2}n + a/B}\right)$。

Since no record keys are indexed, the method as presented above cannot efficiently answer pure-key queries. For transaction range-timeslice queries, the whole timeslice should first be computed. Answering pure-key and range-timeslice queries [Verma and Varman 1994] assumes the existence of another index for various key regions, similarly to the time-index.

由于没有对记录键进行索引，上述方法无法高效地回答纯键查询。对于事务范围时间片查询，首先应该计算整个时间片。回答纯键和范围时间片查询[维尔马和瓦尔曼1994年]假定存在另一个用于各种键区域的索引，这与时间索引类似。

Snapshot Index. The snapshot index [Tsotras and Kangelaris 1995] achieves the I/O-optimal solution to the transaction pure-timeslice problem. Conceptually it consists of three data structures: a multilevel index that provides access to the past by time $t$ ; a multilinked structure among the leaf pages of the multilevel index that facilitates the creation of the query answer at $t$ ; and a hashing function that provides access to records by key, used for update purposes. A real-world object is represented by a record with a time invariant id (object id), a time-variant (temporal) attribute, and a semiclosed transaction-time interval of the form [start_time, end_time). When a new object is added at time $t$ ,a new record is created with interval $\left\lbrack  {t,\text{ now }}\right\rbrack$ and stored sequentially in a data page. At any given instant there is only one data page that stores (accepts) records, called the acceptor page. When an acceptor page becomes full, a new acceptor page is created. Acceptor pages are added at the end of a linked list (list $L$ ) as they are created. Up to now, the snapshot index resembles a linked "log" of pages that keeps the object records.

快照索引。快照索引[索特拉斯和坎杰拉里斯1995年]为事务纯时间片问题实现了I/O最优解决方案。从概念上讲，它由三种数据结构组成：一个多级索引，可按时间$t$访问过去的数据；多级索引的叶页面之间的多链接结构，便于在$t$时刻创建查询答案；以及一个哈希函数，可通过键访问记录，用于更新操作。现实世界中的对象由一条记录表示，该记录包含一个时间不变的ID（对象ID）、一个随时间变化的（时态）属性，以及一个半开的事务时间区间，形式为[开始时间，结束时间)。当在时间$t$添加一个新对象时，会创建一个区间为$\left\lbrack  {t,\text{ now }}\right\rbrack$的新记录，并顺序存储在一个数据页面中。在任何给定时刻，只有一个数据页面用于存储（接受）记录，称为接受页面。当一个接受页面已满时，会创建一个新的接受页面。接受页面在创建时会添加到一个链表（列表$L$）的末尾。到目前为止，快照索引类似于一个页面的链接“日志”，用于保存对象记录。

There are three main differences from a regular log: the use of the hashing function, the in-place deletion updates, and the notion of page usefulness. The hashing function is used for updating records about their "deletion." When a new record is created, the hashing function will store the id of this record, together with the address of the acceptor page that stores it. Object deletions are not added at the and of the log. Rather, they are represented by changing the end_time of the corresponding deleted record. This access is facilitated by the hashing function. All records with end_time equal to now are termed "alive" else they are called "deleted."

与常规日志有三个主要区别：哈希函数的使用、原地删除更新以及页面有用性的概念。哈希函数用于更新有关记录“删除”的信息。当创建一条新记录时，哈希函数会存储该记录的ID以及存储该记录的接受页面的地址。对象删除操作不会添加到日志末尾。相反，它们通过更改相应已删除记录的结束时间来表示。这种访问由哈希函数提供便利。所有结束时间等于当前时间的记录被称为“存活”记录，否则称为“已删除”记录。

As pointed out in Section 4.1, time-only methods need to order their data by time only, and not by time and key. Since data arrives ordered by time, a dynamic hashing function is enough for accessing a record by key (membership test) when updating it. Of course, hashing cannot guarantee against pathological worst cases (i.e., when a bad hashing function is chosen). In those cases, a $\mathrm{B} +$ tree on the keys can be used instead of hashing, leading to logarithmic worst-case update.

正如4.1节所指出的，仅基于时间的方法只需按时间对数据进行排序，而无需按时间和键进行排序。由于数据按时间顺序到达，因此在更新记录时，动态哈希函数足以通过键访问记录（成员测试）。当然，哈希不能保证避免病态的最坏情况（即，当选择了糟糕的哈希函数时）。在这些情况下，可以使用基于键的$\mathrm{B} +$树代替哈希，从而实现对数级的最坏情况更新。

A data page is defined useful for: (i) all time instants for which it was the acceptor page, or (ii) after it ceased being the acceptor page, for all time instants for which the page contains at least $u \cdot  B$ "alive" records. For all other instants, the page is called nonuseful. The useful period [u.start_time, u.end-

数据页在以下情况下被定义为有用：(i) 它作为接收页的所有时间点；或者 (ii) 在它不再是接收页之后，对于该页包含至少 $u \cdot  B$ 条“存活”记录的所有时间点。对于所有其他时间点，该页称为无用页。页的有用时间段 [u.start_time, u.end-

_time) of a page forms a "usefulness" interval for this page. The u.start_time is the time instant the page became an acceptor page. The usefulness parameter $u\left( {0 < u \leq  1}\right)$ is a constant that tunes the behavior of the snapshot index. To answer a pure-timeslice about time $t$ ,the snapshot index will only access the pages useful at $t$ (or,equivalently, those pages that have at least $u \cdot  B$ records alive at $t$ ) plus at most one additional page that was the acceptor page at $t$ . This single page may contain less than $u \cdot  B$ records from the answer.

_time) 构成了该页的“有用性”区间。u.start_time 是该页成为接收页的时间点。有用性参数 $u\left( {0 < u \leq  1}\right)$ 是一个常量，用于调整快照索引的行为。为了回答关于时间 $t$ 的纯时间切片查询，快照索引将仅访问在 $t$ 时刻有用的页（或者等效地，那些在 $t$ 时刻至少有 $u \cdot  B$ 条记录存活的页），再加上最多一个在 $t$ 时刻作为接收页的额外页。这单个页可能包含少于 $u \cdot  B$ 条来自查询结果的记录。

When a useful data page becomes nonuseful, its "alive" records are copied to the current acceptor page (this is like a time-split [Easton 1986; Lomet and Salzberg 1989]). In addition, based on its position in the linked list $L$ ,a non-useful data page is removed from $L$ and is logically placed under the previous data page in the list. This creates a multilinked structure that resembles a forest of trees of data pages, and is called the access forest (Figure 10). The root of each tree in the access forest lies in list $L$ . The access forest has the following properties: (a) u.start_time fields of the data pages in a tree are organized in a preorder fashion; (b) the usefulness interval of a page includes all the corresponding intervals of the pages in its subtree; (c) the usefulness intervals $\left\lbrack  {{d}_{i},{e}_{i}}\right\rbrack$ and $\left\lbrack  {{d}_{\mathrm{i} + 1},{e}_{\mathrm{i} + 1}}\right\rbrack$ of two consecutive children under the same parent page may have one of two orderings: ${d}_{i} < {e}_{i}$ $< {d}_{i + 1} < {e}_{i + 1}$ or ${d}_{i} < {d}_{i + 1} < {e}_{i} < {e}_{i + 1}.$

当一个有用的数据页变得无用时，其“存活”记录会被复制到当前的接收页（这类似于时间分割 [Easton 1986; Lomet 和 Salzberg 1989]）。此外，根据其在链表 $L$ 中的位置，一个无用的数据页会从 $L$ 中移除，并在逻辑上放置在链表中前一个数据页之下。这会创建一个多链表结构，类似于数据页树的森林，称为访问森林（图10）。访问森林中每棵树的根位于链表 $L$ 中。访问森林具有以下属性：(a) 树中数据页的 u.start_time 字段按前序方式组织；(b) 一个页的有用性区间包含其子树中所有页的相应区间；(c) 同一父页下两个连续子页的有用性区间 $\left\lbrack  {{d}_{i},{e}_{i}}\right\rbrack$ 和 $\left\lbrack  {{d}_{\mathrm{i} + 1},{e}_{\mathrm{i} + 1}}\right\rbrack$ 可能有两种排序方式：${d}_{i} < {e}_{i}$ $< {d}_{i + 1} < {e}_{i + 1}$ 或 ${d}_{i} < {d}_{i + 1} < {e}_{i} < {e}_{i + 1}.$

<!-- Media -->

<!-- figureText: 15 30 45 51 60 65 70 80 (a) $\phi \left( {H,{10}\mathrm{{NOW}}}\right)  \geq$ <F. $\lbrack {60},{65}) >$ (65,70)> (b) $\leq  D,\left\lbrack  {{30},{now}}\right\rbrack$ . $\xi {H,\left\lbrack  {{70},{now}}\right\rbrack  } \geq$ $< E,\lbrack {45},{80}) >$ $\left( { < G,\lbrack {65},{70}}\right)  >$ (c) $A$ $C$ $D$ List $L$ STP, $\left\lbrack  {0,\text{ now }}\right\rbrack   \geq$ $\left\lbrack  {{30},{now}}\right\rbrack   \geq$ $< A,\lbrack 1,{53}) >$ $< C,\lbrack {15},{51}) >$ List $L$ $\leq  {TP},\left\lbrack  {0,{now}}\right\rbrack   \geq$ A <A, [1,53)> <C, $\lbrack {15},{51}) > )$ <F, [60,65)>) -->

<img src="https://cdn.noedgeai.com/0195c90d-0a33-750f-877f-c1cec48583bb_30.jpg?x=340&y=234&w=946&h=799&r=0"/>

Figure 10. A schematic view of the access forest for a given collection of usefulness intervals: (a) the usefulness interval of each data page at time 80 ; an open interval at time $t = {80}$ represents a data page that is still useful at that time; (b) the access forest at time $t = {79}$ (in this figure now corresponds to time 79). Each page is represented by a tuple, <page-id, page-usefuleness.period>. Page TP denotes the top of list $L$ ,while the current acceptor page is always at the end of $L$ . (c) The access forest at time $t$ $= {80}$ . At that time page $E$ became nonuseful (because some record deletion reduced the number of "alive" records in $E$ below the ${uB}$ threshold). As a result it is removed from $L$ and placed (together with its subtree) under the previous page in the list,page $D$ . The multilevel index is not shown.

图10. 给定有用性区间集合的访问森林示意图：(a) 每个数据页在时间80的有用性区间；时间 $t = {80}$ 处的开放区间表示在该时间仍有用的数据页；(b) 时间 $t = {79}$ 时的访问森林（在该图中现在对应时间79）。每个页由一个元组 <页ID, 页有用性周期> 表示。页TP表示链表 $L$ 的顶部，而当前接收页始终位于 $L$ 的末尾。(c) 时间 $t$ $= {80}$ 时的访问森林。在那个时间，页 $E$ 变得无用（因为某些记录删除操作使 $E$ 中“存活”记录的数量低于 ${uB}$ 阈值）。结果，它从 $L$ 中移除，并（连同其子树）放置在链表中的前一个页，即页 $D$ 之下。多级索引未显示。

<!-- Media -->

Finding the timeslice as of a given time $t$ is reduced to finding the data pages that were useful at time $t$ . This is equivalent to the set-history problem of Tsotras and Gopinath [1990] and Tso-tras et al. [1995]. The acceptor page as of $t$ is found through the multilevel index that indexes the u.start_time fields of all the data pages. That is, all data pages are at the leaves of the multilevel index (the link list and the access forest are implemented among these leaf pages). Since time is increasing, the multilevel index is "packed" and increases only through its right side. After the acceptor data page at $t$ is located,the remaining useful data pages at $t$ are found by traversing the access forest. This traversing can be done very efficiently using access forest properties [Tsotras and Kangelaris 1995].

查找给定时间 $t$ 的时间片可简化为查找在时间 $t$ 时有用的数据页。这等同于 Tsotras 和 Gopinath [1990] 以及 Tsotras 等人 [1995] 提出的集合历史问题。通过对所有数据页的 u.start_time 字段进行索引的多级索引来查找截至 $t$ 的接受页。也就是说，所有数据页都位于多级索引的叶子节点（链表和访问森林在这些叶子页之间实现）。由于时间是递增的，多级索引是“紧凑的”，并且仅从其右侧增长。在定位到 $t$ 时的接受数据页后，通过遍历访问森林来查找 $t$ 时其余有用的数据页。利用访问森林的属性 [Tsotras 和 Kangelaris 1995] 可以非常高效地完成此遍历。

As a result, the snapshot index solves the "*/-/point" query optimally: $O\left( {{\log }_{B}n}\right.$ $+ a/B)\mathrm{I}/\mathrm{{Os}}$ for query time, $O\left( {n/B}\right)$ space,and $O\left( 1\right)$ update processing per change (in the expected amortized sense, assuming the use of a dynamic hashing function instead of a B-tree [Di-etzfelbinger et al. 1988]). The number of useful pages depends on the choice of parameter $u$ . Larger $u$ means faster query time (fewer accessed pages) and savings in additional space (which remains linear to $n/B$ ). Since more space is available, the answer could be contained in a smaller number of useful pages.

因此，快照索引能最优地解决“*/-/点”查询问题：查询时间为 $O\left( {{\log }_{B}n}\right.$ $+ a/B)\mathrm{I}/\mathrm{{Os}}$，空间为 $O\left( {n/B}\right)$，每次更改的更新处理为 $O\left( 1\right)$（在预期的平摊意义下，假设使用动态哈希函数而非 B 树 [Dietzfelbinger 等人 1988]）。有用页面的数量取决于参数 $u$ 的选择。$u$ 越大，查询时间越快（访问的页面越少），并且能节省额外的空间（该空间与 $n/B$ 保持线性关系）。由于有更多的空间可用，答案可能包含在数量更少的有用页面中。

Migration to a WORM disk is possible for each data page that becomes non-useful. Since the parent of a nonuseful page in the access forest may still be a useful page, an optical disk page must be reserved for the parent. Observe, however, that the snapshot index uses a "batched" migration policy that guarantees that the "reserved" space in the optical disk is limited to a controlled small fraction of the number of pages already transferred to the WORM.

对于每个变得无用的数据页，可以迁移到一次写入多次读取（WORM）磁盘。由于访问森林中无用页面的父页面可能仍然是有用页面，因此必须为父页面保留一个光盘页面。然而，请注意，快照索引使用“批量”迁移策略，该策略保证光盘中的“保留”空间限制在已转移到 WORM 的页面数量的一个可控小比例内。

Different versions of a given key can be linked together so that pure-key queries of the form "find all versions of a given key" are addressed in $O\left( a\right)$ I/Os, where $a$ now represents the number of such versions. Since the key space is separate from the transaction time space, the hashing function used to access records by key can keep the latest version of a key, if any. Each key version when updated can be linked to its previous version; thus each record representing a key contains an extra pointer to the record's previous version. If, instead of a hashing function, a B-tree is used to access the key space, the bound becomes $O\left( {{\log }_{B}S + a}\right)$ where $S$ is the number of different keys ever created.

可以将给定键的不同版本链接在一起，以便以 $O\left( a\right)$ 次输入/输出（I/O）操作处理“查找给定键的所有版本”这种形式的纯键查询，其中 $a$ 现在表示此类版本的数量。由于键空间与事务时间空间是分离的，用于按键访问记录的哈希函数可以保留键的最新版本（如果有的话）。每个键版本在更新时可以链接到其前一个版本；因此，每个表示键的记录包含一个指向该记录前一个版本的额外指针。如果使用 B 树而不是哈希函数来访问键空间，则界限变为 $O\left( {{\log }_{B}S + a}\right)$，其中 $S$ 是曾经创建的不同键的数量。

For answering "range/-/point" queries, the snapshot index has the same problem as the other time-only methods: the whole timeslice must first be computed. In general, this is the trade-off for fast update processing.

为了回答“范围/-/点”查询，快照索引与其他仅考虑时间的方法存在相同的问题：必须首先计算整个时间片。一般来说，这是实现快速更新处理所需要做出的权衡。

Windows Method. Recently, Ra-maswamy [1997] provided yet another solution to the "*/-/point" query, i.e., the windows method. This approach has the same performance as the Snapshot Index. It is a paginated version of a data-structure presented in Chazelle [1986], which optimally solved the pure time-slice query in main memory. Ra-maswamy [1994] partitions time space into contiguous "windows" and associates with each window a list of all intervals that intersect the window's interval. Windows are indexed by a B-tree structure (similar to the multilevel index of the snapshot index).

窗口方法。最近，Ramaswamy [1997] 为“*/-/点”查询提供了另一种解决方案，即窗口方法。这种方法的性能与快照索引相同。它是 Chazelle [1986] 中提出的数据结构的分页版本，该数据结构在主存中最优地解决了纯时间片查询问题。Ramaswamy [1994] 将时间空间划分为连续的“窗口”，并为每个窗口关联一个与该窗口区间相交的所有区间的列表。窗口通过 B 树结构进行索引（类似于快照索引的多级索引）。

To answer a pure timeslice query, the appropriate window that contains this timeslice is first found and then the window's list of intervals is accessed. Note that the "windows" of Ramaswamy [1997] correspond to one or more consecutive pages in the access-forest of Tso-tras and Kangelaris [1995].

为了回答纯时间片查询，首先找到包含该时间片的合适窗口，然后访问该窗口的区间列表。请注意，Ramaswamy [1997] 的“窗口”对应于 Tsotras 和 Kangelaris [1995] 的访问森林中的一个或多个连续页面。

As with the snapshot index, some objects will appear in many windows (when a new window is created it gets copies of the "alive" objects from the previous one), but the space remains $O\left( {n/B}\right)$ . The windows method uses the B-tree to also access the objects by key, hence updating is amortized $O\left( {{\log }_{B}n}\right)$ . If all copies of a given object are linked as proposed in the previous section, all versions of a given key can be found in $O\left( {{\log }_{B}n + a}\right)$ I/Os.

与快照索引一样，有些对象会出现在多个窗口中（当创建一个新窗口时，它会从上一个窗口复制“存活”的对象），但空间仍为$O\left( {n/B}\right)$。窗口方法还使用B树通过键来访问对象，因此更新的均摊时间为$O\left( {{\log }_{B}n}\right)$。如果按照上一节的提议将给定对象的所有副本进行链接，那么可以在$O\left( {{\log }_{B}n + a}\right)$次I/O操作中找到给定键的所有版本。

5.1.3 Time-Key Methods. To answer a transaction range-timeslice query efficiently, it is best to cluster data by both transaction time and key within pages. The "logically" related data for this query are then colocated, thus minimizing the number of pages accessed. Methods in this category are based on some form of a balanced tree whose leaf pages dynamically correspond to regions of the two-dimensional transaction time-key space. While changes still occur in increasing time order, the corresponding keys on which the changes are applied are not in order. Thus, there is a logarithmic update processing per change so that data is placed according to key values in the above time-key space.

5.1.3 时间 - 键方法。为了高效地回答事务范围 - 时间片查询，最好在页面内同时按事务时间和键对数据进行聚类。该查询的“逻辑”相关数据随后会被放置在一起，从而最小化访问的页面数量。此类方法基于某种形式的平衡树，其叶子页面动态对应于二维事务时间 - 键空间的区域。虽然更改仍然按时间递增顺序发生，但应用更改的相应键并非有序。因此，每次更改都有对数级的更新处理，以便数据根据上述时间 - 键空间中的键值进行放置。

<!-- Media -->

<!-- figureText: key h 9 d C b a 10 5 6 7 8 9 time -->

<img src="https://cdn.noedgeai.com/0195c90d-0a33-750f-877f-c1cec48583bb_32.jpg?x=357&y=215&w=936&h=449&r=0"/>

Figure 11. Each page is storing data from a time-key range.

图11. 每个页面存储来自一个时间 - 键范围的数据。

<!-- Media -->

An example of a page containing a time-key range is shown in Figure 11. Here, at transaction time instant 5, a new version of the record with key $b$ is created. At time 6,a record with key $g$ is inserted. At time 7, a new version of the record with key $c$ is created. At time 8,both $c$ and $f$ have new versions and record $h$ is deleted. Each line segment, whose start and end times are represented by ticks, represents one record version. Each record version takes up space in the disk page.

一个包含时间 - 键范围的页面示例如图11所示。在这里，在事务时间点5，创建了键为$b$的记录的新版本。在时间6，插入了键为$g$的记录。在时间7，创建了键为$c$的记录的新版本。在时间8，$c$和$f$都有新版本，并且记录$h$被删除。每个线段（其起始和结束时间由刻度表示）代表一个记录版本。每个记录版本在磁盘页面中占用空间。

There have been two major approaches: methods based on variants of R-trees [Stonebraker 1987; Kolovson and Stone-braker 1989; 1991] and methods based on variants of ${\mathrm{B}}^{ + }$ -trees [Easton 1986; Lomet and Salzberg 1989; Lanka and Mays 1991; Manolopoulos and Kapetan-akis 1990; Becker et al. 1996; Varman and Verma 1997]. Utilizing R-tree-based methods provides a strong advantage, in that R-trees [Guttman 1984; Sellis et al. 1987; Beckmann et al. 1990; Kamel and Faloutsos 1994] can represent additional dimensions on the same index (in principle such a method could support both time dimensions on the same index). A disadvantage of the R-tree-based methods is that they cannot guarantee a good worst-case update and query time performance. However, such worst cases are usually pathological (and do not happen often). In practice, R-trees have shown good average-case performance. Another characteristic of R-tree-based methods is that the end_time of a record's interval is assumed known when the record is inserted in the method, which is not a property of transaction time.

主要有两种方法：基于R树变体的方法[Stonebraker 1987；Kolovson和Stone - braker 1989；1991]以及基于${\mathrm{B}}^{ + }$树变体的方法[Easton 1986；Lomet和Salzberg 1989；Lanka和Mays 1991；Manolopoulos和Kapetan - akis 1990；Becker等人1996；Varman和Verma 1997]。利用基于R树的方法具有很大优势，因为R树[Guttman 1984；Sellis等人1987；Beckmann等人1990；Kamel和Faloutsos 1994]可以在同一索引上表示额外的维度（原则上，这种方法可以在同一索引上支持两个时间维度）。基于R树的方法的一个缺点是，它们不能保证在最坏情况下有良好的更新和查询时间性能。然而，这种最坏情况通常是极端情况（而且不常发生）。在实践中，R树在平均情况下表现良好。基于R树的方法的另一个特点是，在记录插入该方法时，假设已知记录区间的结束时间，而这并非事务时间的属性。

R-Tree-Based Methods. The POST-GRES database management system [Stonebraker 1987] proposed a novel storage system in which no data is ever overwritten. Rather, updates are turned into insertions. POSTGRES timestamps are timestamps of committed transactions. Thus, the POSTGRES storage system is a transaction-time access method.

基于R树的方法。POST - GRES数据库管理系统[Stonebraker 1987]提出了一种新颖的存储系统，其中数据永远不会被覆盖。相反，更新会转换为插入操作。POSTGRES时间戳是已提交事务的时间戳。因此，POSTGRES存储系统是一种事务时间访问方法。

The storage manager accommodates past states of the database on a WORM optical disk (archival system), in addition to the current state that is kept on an ordinary magnetic disk. The assumption is that users will access current data more often than past data, thus the faster magnetic disk is more appropriate for recent data. As past data keeps increasing, the magnetic disk will eventually be filled.

除了将当前状态保存在普通磁盘上之外，存储管理器还将数据库的过去状态存储在一次写入多次读取（WORM）光盘（存档系统）上。假设用户访问当前数据的频率高于过去数据，因此速度更快的磁盘更适合存储近期数据。随着过去数据的不断增加，磁盘最终会被填满。

As data becomes "old" it migrates to the archival system by means of an asynchronous process, called the vacuum cleaner. Each data record has a corresponding interval (Tmin, Tmax), where $T\min$ and $T\max$ are the commit times of the transactions that inserted and (logically) deleted this record from the database. When the vacuum cleaner operates, it transfers data whose end time is before some fixed time to the optical disk. The versions of data that reside on an optical disk page have similar end times (Tmax), but may have widely varying start times (Tmin). Thus, pages on the optical disk are as in Figure 12.

当数据变得“陈旧”时，它会通过一个称为“清理器”的异步过程迁移到存档系统。每个数据记录都有一个相应的区间(Tmin, Tmax)，其中$T\min$和$T\max$分别是将该记录插入数据库和（逻辑上）从数据库中删除该记录的事务的提交时间。当清理器运行时，它会将结束时间早于某个固定时间的数据传输到光盘上。驻留在光盘页面上的数据版本具有相似的结束时间(Tmax)，但起始时间(Tmin)可能差异很大。因此，光盘上的页面如图12所示。

<!-- Media -->

<!-- figureText: key now -->

<img src="https://cdn.noedgeai.com/0195c90d-0a33-750f-877f-c1cec48583bb_33.jpg?x=181&y=241&w=625&h=222&r=0"/>

Figure 12. A page storing data with similar end times.

图12. 一个存储结束时间相似的数据的页面。

<!-- Media -->

If such a page is accessed for a query about some "early" time $t$ ,it may contribute only a single version to the answer, i.e., the answer would not be well clustered among pages.

如果针对某个“早期”时间 $t$ 的查询访问了这样一个页面，它可能只为答案贡献一个版本，即答案在各页面间的聚类效果不佳。

Since data records can be accessed by queries that may involve both time and key predicates, a two-dimensional R-tree [Guttman 1984] access method has been proposed for archival data. POSTGRES assumes that this R-tree is a secondary access method. Pointers to data records are organized according to their key value in one dimension and to their intervals (lifespans) in the other dimension.

由于数据记录可以通过可能同时涉及时间和键谓词的查询来访问，因此有人提出了一种二维R树 [古特曼1984年] 访问方法用于存档数据。POSTGRES假定该R树是一种辅助访问方法。指向数据记录的指针在一个维度上根据其键值进行组织，在另一个维度上根据其区间（生命周期）进行组织。

The data are written sequentially to the WORM device by the vacuuming process. It is not possible to insert new records in a data page on a WORM device that already contains data. So it is not possible to have a primary R-tree with leaves on the optical disk without changing the R-tree insertion algorithm. However, we make estimates based on a primary R-tree, in keeping with our policy of Section 4.1.

数据通过清理过程按顺序写入一次写入多次读取（WORM）设备。在已经包含数据的WORM设备的数据页面中无法插入新记录。因此，如果不改变R树插入算法，就不可能在光盘上构建带有叶子节点的主R树。不过，根据我们在4.1节中的策略，我们基于主R树进行估算。

For current data, POSTGRES does not specify the indexing in use. Whatever it is, queries as of any past time before the most recent vacuum time must access both the current and the historical components of the storage structure. Current records are stored only in the current database and their start times can be arbitrarily far back in the past.

对于当前数据，POSTGRES并未指定所使用的索引。无论使用何种索引，在最近一次清理时间之前的任何过去时间的查询都必须访问存储结构的当前组件和历史组件。当前记录仅存储在当前数据库中，其开始时间可以追溯到任意久远的过去。

For archival data, (secondary) indexes spanning the magnetic and optical disk are proposed (combined media or composite indexes). There are two advantages in allowing indexes to span both media: (a) improved search and insert performance as compared to indexes that are completely on the optical medium (such as the write-once balanced tree [Easton 1986] and the allocation tree [Vitter 1985]); and (b) reduced cost per bit of disk storage as compared to indexes entirely contained on magnetic disk. Two combined media R-tree indexes are proposed in Kolovson and Stonebraker [1989]; they differ on the way index blocks are vacuumed from the magnetic to the archival medium.

对于存档数据，有人提出了跨越磁盘和光盘的（辅助）索引（组合介质或复合索引）。允许索引跨越两种介质有两个优点：（a）与完全基于光学介质的索引（如一次写入平衡树 [伊斯顿1986年] 和分配树 [维特1985年]）相比，搜索和插入性能得到提升；（b）与完全存储在磁盘上的索引相比，磁盘存储的每比特成本降低。科洛夫森和斯通布雷克 [1989年] 提出了两种组合介质R树索引；它们在索引块从磁盘清理到存档介质的方式上有所不同。

In the first approach, the R-tree is rooted on the magnetic disk, and whenever its size on the magnetic disk exceeds some preallocated threshold, the vacuuming process starts moving some of the leaf pages to the archival medium. These pages refer to records that have already been moved to the optical disk. Each such record has Tmax less than some time value. For each leaf page, the maximum Tmax is recorded. The pages with smallest maximum Tmax refer to data that was transferred longest ago. These are the pages that are transferred. Following the vacuuming of the leaf nodes, the process recursively vacuums all parent nodes that point entirely to children nodes that have already been stored on the archive. The root node, however, is never a candidate for vacuuming.

在第一种方法中，R树的根位于磁盘上，每当其在磁盘上的大小超过某个预分配的阈值时，清理过程就会开始将一些叶子页面移动到存档介质中。这些页面引用的记录已经被移动到光盘上。每个这样的记录的Tmax都小于某个时间值。对于每个叶子页面，都会记录最大的Tmax。最大Tmax最小的页面引用的是最早被转移的数据。这些就是被转移的页面。在清理叶子节点之后，该过程会递归地清理所有完全指向已经存储在存档中的子节点的父节点。不过，根节点永远不会成为清理的候选对象。

The second approach (dual $R$ -tree) maintains two $\mathrm{R}$ -trees,both rooted on the magnetic disk. The first is entirely stored on the magnetic disk, while the second is stored on the archival disk, except for its root (in general, except from the upper levels). When the first tree gains the height of the second tree, the vacuuming process vacuums all the nodes of the first tree, except its root, to the optical disk. References to the blocks below the root of the first tree are inserted in the root of the second tree. Over time, there will continue to be two R-trees, the first completely on the magnetic disk and periodically archived. Searches are performed by descending both R-trees.

第二种方法（双 $R$ 树）维护两棵 $\mathrm{R}$ 树，它们的根都位于磁盘上。第一棵树完全存储在磁盘上，而第二棵树除了其根节点（一般来说，除了上层节点）之外，都存储在存档磁盘上。当第一棵树的高度达到第二棵树的高度时，清理过程会将第一棵树除根节点之外的所有节点清理到光盘上。对第一棵树根节点以下块的引用会插入到第二棵树的根节点中。随着时间的推移，将始终存在两棵R树，第一棵树完全位于磁盘上并定期存档。搜索操作通过遍历这两棵R树来执行。

In analyzing the use of the R-tree as a temporal index, we speak of records rather than pointers to records. In both approaches, a given record is kept only once, therefore the space is clearly linear to the number of changes (the number of data records in the tree is proportional to $n$ ). Since the height of the trees is $O\left( {{\log }_{B}n}\right)$ ,each record insertion needs logarithmic time. While, on the average, searching an R-tree is also logarithmic, in the (pathological) worst case this searching can be $O\left( {n/B}\right)$ ,since the whole tree may have to be traversed due to the overlapping regions.

在分析将R树用作时间索引时，我们讨论的是记录而非指向记录的指针。在这两种方法中，给定的记录仅保存一次，因此空间显然与更改次数呈线性关系（树中的数据记录数量与 $n$ 成正比）。由于树的高度为 $O\left( {{\log }_{B}n}\right)$，每次记录插入需要对数时间。虽然平均而言，搜索R树也是对数时间，但在（病态的）最坏情况下，这种搜索可能需要 $O\left( {n/B}\right)$ 时间，因为由于区域重叠，可能需要遍历整个树。

Figure 13 shows the general R-tree method, using overlapping rectangles of time-key space.

图13展示了使用时间 - 键空间重叠矩形的通用R树方法。

R-trees are best suited for indexing data that exhibits a high degree of natural clustering in multiple dimensions; then the index can partition data into rectangles so as to minimize both the coverage and the overlap of the entire set of rectangles (i.e., rectangles corresponding to leaf pages and internal nodes). Transaction time databases, however, may consist of data whose attribute values vary independently of their transaction time intervals, thus exhibiting only one-dimensional clustering. In addition, in an R-tree that stores temporal data, page splits cause a good deal of overlap in the search regions of the nonleaf nodes. It was observed that for data records with nonuniform interval lengths (i.e., a large proportion of "short" intervals and a small proportion of "long" intervals), the overlapping is clearly increased, affecting the query and update performance of the index.

R树最适合对在多个维度上呈现高度自然聚类的数据进行索引；然后，索引可以将数据划分为矩形，以最小化整个矩形集合（即对应于叶页面和内部节点的矩形）的覆盖范围和重叠部分。然而，事务时间数据库可能包含其属性值与其事务时间间隔无关的数据，因此仅呈现一维聚类。此外，在存储时态数据的R树中，页面分裂会导致非叶节点的搜索区域出现大量重叠。据观察，对于具有不均匀间隔长度的数据记录（即，“短”间隔占很大比例，“长”间隔占很小比例），重叠明显增加，影响了索引的查询和更新性能。

<!-- Media -->

<!-- figureText: 5.6 10 12 14 20 10 12 14 20 B b 12 14 20 time Suppose a maximum capacity of 5 record versions in each page Record versions are entered as they die. At time instant 9, the records must be split into two pages as illustrated here because at instant 9 there are six dead record versions. key These are possible data page boundaries at the time record version $d$ dies. A C 5.6 time This is a possible allocation to R-tree data pages of all versions shown. given that at most 5 and and least 3 record versions must be in each page. The parent node will contain the border coordinates for each of the five children. For example, data node $\mathrm{C}$ has borders with time running between 0 and 14 and keys b and c only. The version of record c between 8 and 12 belongs to E. A time slice query at time instant 8 visits A, B, C, and E and obtains one record version from each page. -->

<img src="https://cdn.noedgeai.com/0195c90d-0a33-750f-877f-c1cec48583bb_34.jpg?x=844&y=227&w=624&h=1155&r=0"/>

Figure 13. An example of data bounding as used in R-tree based methods.

图13. 基于R树的方法中使用的数据边界示例。

<!-- Media -->

Figure 14 shows how long-lived records inhibit the performance of structures that keep only one copy of each record and which keep time-key rectangles. The problem is that a long-lived record determines the length of the time range associated with the page in which it resides. Then, even if only one other key value is present, and there are many changes to the record with the other key value in that time range, overlap is required. For example, in Figure 14, the eleventh record version (shown with a dotted line) belongs to the time-key range of this page but it cannot fit since the page has already ten record versions. It will instead be placed in a different page whose rectangle has to overlap this one. The same example also illustrates that the number of leaf pages to be retrieved for a timeslice can be large $\left( {O\left( a\right) }\right)$ since only a few records may be "alive" (contain the given time value in their interval) for any one page.

图14展示了长期存在的记录如何抑制那些只为每条记录保留一个副本并维护时间键矩形的结构的性能。问题在于，长期存在的记录决定了其所在页面关联的时间范围的长度。然后，即使只有另一个键值存在，并且在该时间范围内具有另一个键值的记录有许多更改，也会产生重叠。例如，在图14中，第11个记录版本（用虚线表示）属于该页面的时间键范围，但由于该页面已经有10个记录版本，它无法放入。相反，它将被放置在另一个矩形必须与该矩形重叠的页面中。同一个示例还表明，对于一个时间片，需要检索的叶页面数量可能很大$\left( {O\left( a\right) }\right)$，因为对于任何一个页面，可能只有少数记录是“活跃的”（其间隔包含给定的时间值）。

<!-- Media -->

<!-- figureText: time one other record, with another key. Now we have minimum boundaries versions, the eleventh version in this time-key rectangle does not fit. shows where one long-lived record and one record with a different key a) The first record version is very long-lived. Its time span is the minimum time span this time-key rectangle can have. Here we add in both time and key. key time b) The second record gets many new versions over the time span of the first (long-lived) record. If the page capacity is ten record rectangle, which holds the next version of the second record. This and many short versions forces time-key rectangles to overlap. -->

<img src="https://cdn.noedgeai.com/0195c90d-0a33-750f-877f-c1cec48583bb_35.jpg?x=187&y=224&w=587&h=474&r=0"/>

Figure 14. The effect of long-lived records on overlapping.

图14. 长期存在的记录对重叠的影响。

<!-- Media -->

In an attempt to overcome the above problems,the segment $R$ -tree (SR-tree) was proposed [Kolovson and Stone-braker 1991; Kolovson 1993]. The SR-tree combines properties of the R-tree and the segment tree, a binary tree data structure proposed in Bentley [1977] for storing line segments. A Segment Tree stores the interval endpoints on the leaf nodes; each internal node is associated with a "range" that contains all the endpoints in its subtree. An interval $I$ is stored in the highest internal node $v$ such that $I$ covers $v$ ’s range and does not cover the range of $v$ ’s parent. Observe that an interval may be stored in at most logarithmic many internal nodes; thus the space is no longer linear [Mehlhorn 1984].

为了克服上述问题，有人提出了分段$R$树（SR树）[科洛夫森（Kolovson）和斯通布雷克（Stonebraker）1991年；科洛夫森1993年]。SR树结合了R树和分段树的特性，分段树是本特利（Bentley）在1977年提出的用于存储线段的二叉树数据结构。分段树将间隔端点存储在叶节点上；每个内部节点与一个“范围”相关联，该范围包含其所有子树中的端点。一个间隔$I$存储在最高的内部节点$v$中，使得$I$覆盖$v$的范围，但不覆盖$v$父节点的范围。请注意，一个间隔最多可以存储在对数数量的内部节点中；因此，空间不再是线性的[梅尔霍恩（Mehlhorn）1984年]。

The SR-tree (Figure 15) is an R-tree where intervals can be stored in both leaf and nonleaf nodes. An interval $I$ is placed to the highest level node $X$ of the tree such that $I$ spans at least one of the intervals represented by $X$ ’s child nodes. If $I$ does not span $X$ ,spans at least one of its children but is not fully contained in $X$ ,then $I$ is fragmented.

SR树（图15）是一种R树，其中间隔可以存储在叶节点和非叶节点中。一个间隔$I$被放置在树的最高层节点$X$中，使得$I$至少跨越$X$的子节点所代表的一个间隔。如果$I$不跨越$X$，至少跨越其一个子节点，但不完全包含在$X$中，则$I$会被分割。

<!-- Media -->

<!-- figureText: Line L spans C but is root root spans D spans $E$ Spanning portion of L C Remnant portion of L not contained in $A$ spans $A$ A spans B B -->

<img src="https://cdn.noedgeai.com/0195c90d-0a33-750f-877f-c1cec48583bb_35.jpg?x=903&y=221&w=494&h=707&r=0"/>

Figure 15. The SR-tree.

图15. SR树。

<!-- Media -->

Using this idea, long intervals will be placed in higher levels of the R-Tree, thus the SR-Tree tends to decrease the overlapping in leaf nodes (in the regular R-Tree, a long interval stored in a leaf node will "elongate" the area of this node thus exacerbating the overlap problem). One risks having large numbers of spanning records or fragments of spanning records stored high up in the tree. This decreases the fan-out of the index as there is less room for pointers to children. It is suggested to vary the size of the nodes in the tree, making higher-up nodes larger. "Varying the size" of a node means that several pages are used for one node. This adds some page accesses to the search cost.

利用这个思路，长间隔将被放置在R树的较高层，因此SR树倾向于减少叶节点中的重叠（在常规R树中，存储在叶节点中的长间隔会“拉长”该节点的区域，从而加剧重叠问题）。这样做可能会导致大量跨越记录或跨越记录的片段存储在树的高层。这会降低索引的扇出，因为指向子节点的指针空间变小了。建议改变树中节点的大小，使高层节点更大。“改变节点的大小”意味着一个节点使用多个页面。这会增加一些页面访问的搜索成本。

As with the R-tree, if the record is inserted at a leaf (because it did not span anything), the boundaries of the space covered by the leaf node in which it is placed may be expanded. Expansions may be needed on all nodes on the path to the leaf that contains the new record. This may change the spanning relationships, since records may no longer span children that have been expanded. In this case, such records are reinserted in the tree, possibly being demoted to occupants of nodes they previously spanned. Splitting nodes may also cause changes in spanning relationships, as they make children smaller-former occupants of a node may be promoted to spanning records in the parent.

与R树一样，如果记录插入到叶节点（因为它不跨越任何内容），放置该记录的叶节点所覆盖的空间边界可能会扩展。在通往包含新记录的叶节点的路径上的所有节点可能都需要进行扩展。这可能会改变跨越关系，因为记录可能不再跨越已扩展的子节点。在这种情况下，这些记录会被重新插入到树中，可能会被降级到它们之前跨越的节点中。节点分裂也可能导致跨越关系的变化，因为它们会使子节点变小——节点的前占用者可能会被提升为父节点中的跨越记录。

Similarly with the segment tree, the space used by the SR-tree is no longer linear. An interval may be stored in more than one nonleaf nodes (in the spanning and remnant portions of this interval). Due to the use of the segment-tree property, the space can be as much as $O\left( {\left( {n/B}\right) {\log }_{B}n}\right)$ . Inserting an interval still takes logarithmic time. However, due to possible promotions, demotions, and fragmentation, insertion is slower than in the R-tree. Even though the segment property tends to reduce the overlapping problem, the (pathological) worst-case performance for the deletion and query times remains the same as for the R-tree organization. The average-case behavior is again logarithmic.

与线段树类似，SR树（Segment Remnant Tree，分段残差树）所使用的空间不再是线性的。一个区间可能会存储在多个非叶节点中（位于该区间的跨越部分和残余部分）。由于利用了线段树的特性，其空间使用量可能高达 $O\left( {\left( {n/B}\right) {\log }_{B}n}\right)$ 。插入一个区间仍然需要对数时间。然而，由于可能存在的提升、降级和分裂操作，插入操作比R树（R-Tree）要慢。尽管线段特性有助于减少重叠问题，但在（极端）最坏情况下，删除和查询操作的性能与R树结构相同。平均情况下的性能仍然是对数级的。

To improve the performance of their structure, the authors also proposed the use of a skeleton ${SR}$ -tree,which is an SR-tree that prepartitions the entire domain into some number of regions. This prepartition is based on some initial assumption about the distribution of data and the number of intervals to be inserted. Then the skeleton SR-tree is populated with data; if the data distribution is changed, the structure of the skeleton SR-tree can be changed, too.

为了提高其结构的性能，作者还提出使用骨架 ${SR}$ 树，它是一种SR树，会将整个定义域预先划分为若干个区域。这种预划分是基于对数据分布和待插入区间数量的一些初始假设。然后将数据填充到骨架SR树中；如果数据分布发生变化，骨架SR树的结构也可以随之改变。

An implicit assumption made by all R-tree-based methods is that when an interval is inserted,both its Tmin and Tmax values are known. In practice, however, this is not true for "current" data. One solution is to enter all such intervals as (Tmin, now), where now is a variable representing the current time. A problem with this approach is that a "deletion" update that changes the now value of an interval to Tmax is implemented by a search for the interval, a deletion of the (Tmin, now) interval, and a reinsertion as (Tmin, Tmax) interval. Since searches are not guaranteed for worst-case performance, this approach could be problematic. The deletion of (Tmin, now) is a physical deletion, which implies the physical deletion of all remnant portions of this interval. A better solutionis to keep the current records in a separate index (probably a basic R-tree). This avoids the above deletion problem, but the worst-case performance remains as before.

所有基于R树的方法都有一个隐含假设，即插入一个区间时，其Tmin和Tmax值都是已知的。然而在实际应用中，对于“当前”数据并非如此。一种解决方案是将所有此类区间表示为 (Tmin, now) ，其中now是一个表示当前时间的变量。这种方法的一个问题是，将区间的now值更新为Tmax的“删除”操作，需要先搜索该区间，删除 (Tmin, now) 区间，然后再以 (Tmin, Tmax) 区间重新插入。由于无法保证最坏情况下的搜索性能，这种方法可能会存在问题。删除 (Tmin, now) 是物理删除，这意味着要物理删除该区间的所有残余部分。更好的解决方案是将当前记录存储在一个单独的索引中（可能是一个基本的R树）。这样可以避免上述删除问题，但最坏情况下的性能仍然和之前一样。

The pure-key query is addressed as a special case of a range time-interval query, where the range is limited to a key and the time-interval is the whole time axis. Hence, all pages that contain the key in their range are accessed. However, if this key never existed, the search may go through $O\left( {n/B}\right)$ pages in (pathological) worst case. If this key has existed, the search will definitely find its appearances, but it may also access pages that do not contain any appearances of this key.

纯键查询可作为范围时间区间查询的一种特殊情况来处理，其中范围仅限于一个键，时间区间为整个时间轴。因此，会访问其范围包含该键的所有页面。然而，如果该键从未存在过，在（极端）最坏情况下，搜索可能会遍历 $O\left( {n/B}\right)$ 个页面。如果该键曾经存在过，搜索肯定能找到它的出现位置，但也可能会访问不包含该键任何出现位置的页面。

If the SR-tree is used as a valid-time method, then physical deletion of any stored interval should be supported efficiently. As above, the problem with physical deletions emanates from keeping an interval in many remnant segments, all of which have to be found and physically deleted. Actually, the original SR-tree paper [Kolovson and Stone-braker 1991] assumes that physical deletions do not happen often.

如果将SR树用作有效时间方法，那么应该能够高效地支持对任何已存储区间的物理删除。如前所述，物理删除的问题源于一个区间会存储在多个残余段中，必须找到并物理删除所有这些残余段。实际上，原始的SR树论文 [Kolovson和Stonebraker 1991] 假设物理删除操作不会频繁发生。

Write-Once B-Tree. The write-once B-tree, or WOBT, proposed in Easton [1986], was originally intended for a database that resided entirely on WORMs. However, many variations of this method (the time-split B-tree [Lomet and Salzberg 1989]; the persistent B-tree [Lanka and Mays 1991]; the multiversion B-tree [Becker et al. 1996]; and the multiversion access structure [Varman and Verma 1997]) have been proposed which may use both a WORM and a magnetic disk, or only a magnetic disk. The WOBT itself can be used either on a WORM or on a magnetic disk. The WOBT is a modification of the B+- tree, given the constraints of a WORM.

一次写入B树（Write-Once B-Tree，WOBT）。Easton [1986] 提出的一次写入B树，最初是为完全驻留在一次写入多次读取（Write-Once Read-Many，WORM）设备上的数据库设计的。然而，已经提出了该方法的许多变体（如时间分裂B树 [Lomet和Salzberg 1989]；持久B树 [Lanka和Mays 1991]；多版本B树 [Becker等人 1996]；以及多版本访问结构 [Varman和Verma 1997]），这些变体可能同时使用WORM设备和磁盘，或者仅使用磁盘。WOBT本身既可以在WORM设备上使用，也可以在磁盘上使用。WOBT是在WORM设备的限制条件下对B+树的一种改进。

The WORM characteristics imply that once a page is written, no new data can be entered or updated in the page (since a checksum is burned into the disk). As a result, each new index entry occupies an entire page; for example, if a new index entry takes 12 bytes and a page is 1,024 bytes, 99% of the page is empty. Similarly, each new record version is an entire page. Tree nodes are collections of pages-for example, a track on a disk. Record versions contain their transaction start times only. A new version with the same key is placed in the same node. Its start time is the end time of the previous version. Nodes represent a rectangle in the transaction time-key space. The nodes partition that space-each time-key point is in exactly one node.

WORM设备的特性意味着一旦写入一个页面，就不能在该页面中输入或更新新数据（因为校验和会被写入磁盘）。因此，每个新的索引项会占用整个页面；例如，如果一个新的索引项占用12字节，而一个页面为1024字节，那么该页面99%的空间都是空的。同样，每个新的记录版本也是一个完整的页面。树节点是页面的集合，例如磁盘上的一个磁道。记录版本只包含其事务开始时间。具有相同键的新版本会被放置在同一个节点中。其开始时间是前一个版本的结束时间。节点在事务时间 - 键空间中表示一个矩形。节点对该空间进行划分，每个时间 - 键点恰好位于一个节点中。

When a node fills up, it can be split by (current) transaction time or split first by current transaction time and then by key. The choice depends on how many records in the node are current versions at the time of the split. The old node is left in place. (There is no other choice.) The record versions "alive" at the current transaction time are copied to a new node, or two new nodes if it is also split by key. There is space for new versions in the new nodes. Deletions of records are handled in the only possible way: a node deletion record is written in a current node and it contains the end time. When the current node is split, the deleted record is not copied. This design enables some clustering of the records in nodes by time and key (after a node split, "alive" records are stored together in a page) but most of the space of most of the optical disk pages is empty (because most new entries occupy whole pages).

当一个节点填满时，可以按（当前）事务时间进行拆分，或者先按当前事务时间拆分，然后再按键拆分。选择哪种方式取决于拆分时节点中有多少记录是当前版本。旧节点保持原位（别无选择）。在当前事务时间“存活”的记录版本会被复制到一个新节点中，如果还按键拆分，则会复制到两个新节点中。新节点中有存放新版本的空间。记录的删除以唯一可能的方式处理：在当前节点中写入一条节点删除记录，其中包含结束时间。当当前节点拆分时，已删除的记录不会被复制。这种设计使得节点中的记录能够按时间和键进行一定程度的聚类（节点拆分后，“存活”的记录会存放在同一页面中），但大多数光盘页面的大部分空间都是空的（因为大多数新条目会占用整个页面）。

When a root node splits, a new root is created. Addresses of consecutive roots and their creation times are held in a "root log" that has the form of a variable-length append-only array. This array provides access to the appropriate root of the WOBT by time.

当根节点拆分时，会创建一个新的根节点。连续根节点的地址及其创建时间保存在一个“根日志”中，该日志采用变长的仅追加数组形式。这个数组可以按时间访问WOBT（写时复制B树）的相应根节点。

If the WOBT is implemented on a magnetic disk, space utilization is immediately improved, as it is not necessary to use an entire page for one entry. Pages can be updated, so they can be used for nodes. Space utilization is $O\left( {n/B}\right)$ and range queries are $O\left( {{\log }_{B}n}\right.$ $+ a/B)$ ,if one disregards record deletion. These bounds are for using the method exactly as described in this paper, except that each node of the tree will be a page on a magnetic disk. In particular, the old node in a split is not moved. Current records are copied to a new node or to two new nodes. Since deletions are simply handled with a deletion record (which is "seen" by the method as another updated value), the search algorithm is not able to avoid searching pages that may be full of "deleted" records. So if deletions are frequent, pages that do not contribute to the answer may be accessed.

如果WOBT在磁盘上实现，空间利用率会立即得到提高，因为不需要为一个条目使用整个页面。页面可以更新，因此可用于存储节点。如果不考虑记录删除，空间利用率为$O\left( {n/B}\right)$，范围查询为$O\left( {{\log }_{B}n}\right.$ $+ a/B)$。这些界限是按照本文所述方法精确使用时的情况，只是树的每个节点将是磁盘上的一个页面。特别是，拆分中的旧节点不会移动。当前记录会被复制到一个新节点或两个新节点中。由于删除操作只是通过一条删除记录来处理（该方法将其视为另一个更新值），搜索算法无法避免搜索可能充满“已删除”记录的页面。因此，如果删除操作频繁，可能会访问对查询结果无贡献的页面。

Since all the ${\mathrm{B}}^{ + }$ -tree-based transaction-time methods search data records by both transaction time and key, or by transaction time only, answering a pure-key query with the WOBT (or the time-split B-tree, persistent B-tree, and multiversion B-tree) requires that a given version (instance) of the key whose previous versions are requested should also be provided by the query. That is, a transaction time predicate should be provided in the pure-key query as, for example, in "find the previous salary history of employee $A$ who was alive at $t$ ."

由于所有基于${\mathrm{B}}^{ + }$ -树的事务时间方法都是通过事务时间和键，或者仅通过事务时间来搜索数据记录，因此使用WOBT（或时间拆分B树、持久B树和多版本B树）来回答纯键查询时，查询还需要提供请求其先前版本的键的给定版本（实例）。也就是说，纯键查询中应该提供一个事务时间谓词，例如“查找在$t$时在职的员工$A$的先前薪资历史”。

Different versions of a given key can be linked together so that the pure-key query (with time predicate) is addressed by the WOBT in $O\left( {{\log }_{B}n + a}\right)$ I/Os. The logarithmic part is spent finding the instance of employee $A$ in version $t$ and then its previous $a$ versions are accessed using a linked structure. Basically, the WOBT (and the time-split B-tree, persistent B-tree, and the multiversion B-tree) can have backwards links in each node to the previous historical version. This does not use much space, but for records that do not change over many copies, one needs to go back many pages before getting more information. To achieve the bound above, each record needs to keep the address of the last different version of that record.

给定键的不同版本可以链接在一起，这样WOBT就可以通过$O\left( {{\log }_{B}n + a}\right)$次I/O操作来处理（带有时间谓词的）纯键查询。对数部分的时间用于在版本$t$中查找员工$A$的实例，然后使用链接结构访问其先前的$a$个版本。基本上，WOBT（以及时间拆分B树、持久B树和多版本B树）的每个节点中都可以有指向先前历史版本的反向链接。这不会占用太多空间，但对于在多次复制中都未更改的记录，在获取更多信息之前需要回溯多个页面。为了达到上述界限，每个记录需要保存该记录最后一个不同版本的地址。

If such addresses are kept in records, the address of the last different version for each record is available at the time the data node does a time split. Then these addresses can be copied to the new node with their respective records. A record whose most recent previous version is in the node that is split must add that address. A record that is the first version with that key must have a special symbol to indicate this fact. This simple algorithm can be applied to any method that does time splits.

如果记录中保存了这些地址，那么在数据节点进行时间拆分时，每个记录的最后一个不同版本的地址就可以获取到。然后这些地址可以随各自的记录一起复制到新节点中。其最近的先前版本位于被拆分节点中的记录必须添加该地址。具有该键的第一个版本的记录必须有一个特殊符号来表明这一事实。这种简单的算法可以应用于任何进行时间拆分的方法。

To answer the general pure-key query "find the previous salary history of employee $A$ " requires finding if $A$ was ever an employee. The WOBT needs to copy "deleted" records when a time split occurs, which implies that the WOBT state carries the latest record for all keys ever created. However, this increases space consumption. Otherwise, if "deleted" records are not copied, all pages including this key in their key space may have to be searched.

要回答一般的纯键查询“查找员工$A$的先前薪资历史”，需要确定$A$是否曾经是员工。WOBT在进行时间拆分时需要复制“已删除”的记录，这意味着WOBT状态会保留所有曾经创建的键的最新记录。然而，这会增加空间消耗。否则，如果不复制“已删除”的记录，可能就需要搜索其键空间中包含该键的所有页面。

The WOBT used on a magnetic disk still makes copies of records where it does not seem necessary. The WOBT always makes a time split before making a key split. This creates one historical page and two current pages where previously there was only one current page. A B-tree split creates two current pages where there was only one. No historical pages are created. It seems like a good idea to be able to make pure key splits as well as time splits or time-and-key splits. This would make the space utilization better.

磁盘上使用的写时复制B树（WOBT）仍然会在看似不必要的情况下复制记录。写时复制B树总是在进行键分裂之前进行时间分裂。这会创建一个历史页面和两个当前页面，而之前只有一个当前页面。B树分裂会在原本只有一个页面的地方创建两个当前页面，不会创建历史页面。能够进行纯键分裂、时间分裂或时间 - 键分裂似乎是个好主意，这将提高空间利用率。

Time Split B-Tree. The time-split B-tree (or TSB-tree) [Lomet and Sal-zberg 1989; 1990; 1993] is a modification of the WOBT that allows pure key splits and keeps the current data in an erasable medium such as a magnetic disk and migrates the data to another disk (which could be magnetic or optical) when a time split is made. This partitions the data in nodes by transaction time and key (like the WOBT), but is more space efficient. It also separates the current records from most of the historical records. In addition, the TSB-tree does not keep a "root log." Instead, it creates new roots as B+-trees do, by increasing the height of the tree when the root splits.

时间分裂B树。时间分裂B树（或TSB树）[洛梅特（Lomet）和萨尔茨伯格（Sal - zberg）1989年；1990年；1993年]是对写时复制B树的一种改进，它允许进行纯键分裂，并将当前数据存储在可擦除介质（如磁盘）中，在进行时间分裂时将数据迁移到另一个磁盘（可以是磁盘或光盘）。它像写时复制B树一样按事务时间和键对节点中的数据进行分区，但更节省空间。它还将当前记录与大多数历史记录分开。此外，TSB树不保留“根日志”，而是像B +树那样，在根节点分裂时通过增加树的高度来创建新的根节点。

When a data page is full and there are fewer than some threshold value of alive distinct keys, the TSB-tree will split the page by transaction time only. This is the same as what the WOBT does, except now times other than the current time can be chosen. For example, the split time for a data page could be the "time of last update," after which there were only insertions of records with new keys and no updates creating new versions of already existing records. The new insertions, after the time chosen for the split, need not have copies in the historical node. Time splits in the WOBT and in the TSB-trees are illustrated in Figure 16.

当数据页已满，且存活的不同键的数量少于某个阈值时，TSB树将仅按事务时间对页面进行分裂。这与写时复制B树的操作相同，只是现在可以选择当前时间以外的其他时间。例如，数据页的分裂时间可以是“最后更新时间”，在此之后，只有插入具有新键的记录，而没有更新以创建现有记录的新版本。在选择的分裂时间之后插入的新记录不需要在历史节点中复制。写时复制B树和TSB树中的时间分裂如图16所示。

Time splitting, whether by current time or by time of last update, enables an automatic migration of older versions of records to a separate historical database. This is to be contrasted with POSTGRES' vacuuming, which is "manual" and is invoked as a separate background process that searches through the database for dead records.

无论是按当前时间还是按最后更新时间进行时间分裂，都能将记录的旧版本自动迁移到单独的历史数据库中。这与POSTGRES的清理操作形成对比，POSTGRES的清理操作是“手动”的，作为一个单独的后台进程调用，该进程会在数据库中搜索已删除的记录。

It can also be contrasted with methods that reserve optical pages for pages that cannot yet be moved and maintain two addresses (a magnetic page address and an optical page address) for searching for the contents. TSB-tree migration takes place when a time split is made. The current page retains the current contents and the historical records are written sequentially to the optical disk. The new optical disk address and the time of the split are posted to the parent in the TSB-tree. As with ${\mathrm{B}}^{ + }$ -tree node splitting, only the node to be split, the newly allocated node, and the parent are affected (also, but rarely, a full parent may require further splitting). Since the node is full and is obtaining new data, a split must be made anyway, whether or not the new node is on an optical disk. (This migration to an archive can also be used for media recovery, as illustrated in Lomet and Sal-zberg [1993].)

它也可以与以下方法形成对比：这些方法为尚不能移动的页面预留光盘页面，并维护两个地址（磁盘页面地址和光盘页面地址）以搜索内容。TSB树的迁移在进行时间分裂时发生。当前页面保留当前内容，历史记录按顺序写入光盘。新的光盘地址和分裂时间会发布到TSB树中的父节点。与${\mathrm{B}}^{ + }$树节点分裂一样，只有要分裂的节点、新分配的节点和父节点会受到影响（此外，很少情况下，已满的父节点可能需要进一步分裂）。由于节点已满且要获取新数据，无论新节点是否在光盘上，都必须进行分裂。（这种迁移到存档的操作也可用于介质恢复，如洛梅特和萨尔茨伯格[1993年]所示。）

<!-- Media -->

<!-- figureText: key h 9 d C b new node new node old node 10 time (a) The WOBT splits at current time, copying current records into a new node. key old node 56 10 after split time (b) The TSB tree can choose other times to split -->

<img src="https://cdn.noedgeai.com/0195c90d-0a33-750f-877f-c1cec48583bb_39.jpg?x=446&y=227&w=739&h=693&r=0"/>

Figure 16. Time splitting in the WOBT and TSB-trees.

图16. 写时复制B树和TSB树中的时间分裂。

<!-- Media -->

Time splitting by other than the current transaction time has another advantage. It can be used in distributed databases where the commit time of a transaction is not known until the commit message is sent from the coordinator. In such a database, an updated record of a PREPARED cohort may or may not have a commit time before the time when an overflowing page containing it must be split. Such a page can only be split by a time before the time voted by the cohort as the earliest time it may commit (see Salzberg [1994] and Lomet [1993] for details).

按当前事务时间以外的时间进行时间分裂还有另一个优点。它可用于分布式数据库，在这种数据库中，直到协调器发送提交消息，事务的提交时间才可知。在这样的数据库中，一个处于准备状态的参与者的更新记录，在包含它的溢出页面必须分裂时，可能有也可能没有提交时间。这样的页面只能按参与者投票确定的最早可能提交时间之前的时间进行分裂（详情见萨尔茨伯格[1994年]和洛梅特[1993年]）。

Full data pages with a large number of distinct keys currently "alive" are split by key only in the TSB-tree. The WOBT splits first by time and then by key. Similar to the WOBT, space usage for the TSB-tree is $O\left( {n/B}\right)$ . The constant factor in the asymptotic bound is smaller for the TSB-tree, since it makes fewer copies of records. Key splitting for the WOBT and the TSB-tree is shown in Figure 17.

在TSB树中，包含大量当前“存活”的不同键的满数据页仅按键进行分裂。写时复制B树先按时间分裂，然后按键分裂。与写时复制B树类似，TSB树的空间使用率为$O\left( {n/B}\right)$。由于TSB树复制的记录更少，其渐近边界中的常数因子更小。写时复制B树和TSB树的键分裂如图17所示。

An extensive average-case analysis using Markov chains, and considering various rates of update versus insertions of records with new keys, can be found in Lomet and Salzberg [1990], showing, at worst, two copies of each record, even under large update rates. The split threshold was kept at ${2B}/3$ . (If more than ${2B}/3$ distinct keys are in the page, a pure key split is made.)

洛梅特和萨尔茨伯格[1990年]使用马尔可夫链进行了广泛的平均情况分析，并考虑了各种更新率与插入具有新键的记录的比率，结果表明，即使在高更新率下，最坏情况下每个记录也只有两份副本。分裂阈值保持在${2B}/3$。（如果页面中有超过${2B}/3$个不同的键，则进行纯键分裂。）

There is, however, a problem with pure key splits. The decision on the key splits is made on the basis of alive keys at the time the key split is made. For example, in Figure 17(b), the key split is taken at time $t = {18}$ ,when there are six keys alive, separated three per new page. However, this key range division does not guarantee that the two pages will have enough alive keys for all previous times; at time $t = {15}$ ,the bottom page has only one key alive.

然而，纯键分割（key splits）存在一个问题。键分割的决策是基于进行键分割时的活跃键（alive keys）做出的。例如，在图17(b)中，键分割在时间$t = {18}$进行，此时有六个活跃键，每个新页面分配三个。然而，这种键范围划分并不能保证两个页面在所有先前时间都有足够的活跃键；在时间$t = {15}$，底部页面只有一个活跃键。

<!-- Media -->

<!-- figureText: key h 9 14 15 18 new nodes h 9 e d C b 14 15 18 old node 5 6 7 8 9 10 12 time (a) The WOBT splits data nodes first by time then sometimes also by key. key new nodes 5.6 7 8 9 10 12 time (b) The TSB-tree can split by key alone. -->

<img src="https://cdn.noedgeai.com/0195c90d-0a33-750f-877f-c1cec48583bb_40.jpg?x=220&y=248&w=1216&h=1150&r=0"/>

Figure 17. Key splitting in the WOBT and TSB-tree.

图17. WOBT和TSB树中的键分割。

<!-- Media -->

Suppose we have a database where most of the changes are insertions of records with a new key. As time goes by, in the TSB-tree, only key splits are made. After a while, queries of a past time will become inefficient. Every timeslice query will have to visit every node of the TSB-tree, since they are all current nodes. Queries as of now, or of recent time, will be efficient, since every node will have many alive records. But queries as of the distant past will be inefficient, since many of the current nodes will not contain records that were "alive" at that distant past time.

假设我们有一个数据库，其中大部分更改是插入具有新键的记录。随着时间的推移，在TSB树中，只会进行键分割。过了一段时间后，对过去某个时间的查询将变得低效。每个时间片查询都必须访问TSB树的每个节点，因为它们都是当前节点。对当前或近期的查询将是高效的，因为每个节点都会有许多活跃记录。但对遥远过去的查询将是低效的，因为许多当前节点将不包含在那个遥远过去时间“活跃”的记录。

In addition, as in the WOBT, the TSB-tree merely posts deletion markers and does not merge sparse nodes. If no merging of current nodes is done, and there are many record deletions, a current node may contain few current records. This could make current search slower than it should be.

此外，与WOBT一样，TSB树只是张贴删除标记，而不合并稀疏节点。如果不合并当前节点，并且有许多记录被删除，那么一个当前节点可能只包含很少的当前记录。这可能会使当前搜索比应有的速度更慢。

Thus the worst-case search time for the TSB-tree can be $O\left( {n/B}\right)$ for a transaction (pure or range) timeslice. Pages may be accessed that have no answers to the query. Other modifications of Easton [1986], discussed in the next section, combined with the TSB-tree modifications of the author above should solve this problem. Basically, when there are too few distinct keys at any time covered by the time-key rectangle of a node to be split, it must be split by time and then possibly by key. Node consolidation should also be supported (to deal with pages lacking in alive keys due to deletions).

因此，对于TSB树，事务（纯事务或范围事务）时间片的最坏情况搜索时间可能是$O\left( {n/B}\right)$。可能会访问那些对查询没有答案的页面。下一节将讨论的伊斯顿（Easton）[1986]的其他修改，结合上述作者对TSB树的修改，应该可以解决这个问题。基本上，当要分割的节点的时间 - 键矩形所覆盖的任何时间内，不同键的数量太少时，必须先按时间分割，然后可能再按键分割。还应该支持节点合并（以处理由于删除而缺少活跃键的页面）。

Index nodes in the TSB-tree are treated differently from data nodes. The children of index nodes are rectangles in time-key space. So making a time split or key split of an index node may cause a lower level node to be referred to by two parents.

TSB树中的索引节点与数据节点的处理方式不同。索引节点的子节点是时间 - 键空间中的矩形。因此，对索引节点进行时间分割或键分割可能会导致一个较低层级的节点被两个父节点引用。

Index node splits in the TSB-tree are restricted in ways that guarantee that current nodes (the only ones where insertions and updates occur) have only one parent. This parent is a current index node. Updates need never be made in historical index nodes, which like historical data nodes can be placed on WORM devices.

TSB树中的索引节点分割受到限制，以确保当前节点（即发生插入和更新的唯一节点）只有一个父节点。这个父节点是一个当前索引节点。永远不需要在历史索引节点中进行更新，这些历史索引节点与历史数据节点一样，可以放在一次写入多次读取（WORM）设备上。

A time split can be done at any time before the start time of the oldest current child. If time splits were allowed at current transaction times for index nodes, lower level current nodes would have more than one parent.

可以在最旧的当前子节点的开始时间之前的任何时间进行时间分割。如果允许在当前事务时间对索引节点进行时间分割，那么较低层级的当前节点将有多个父节点。

A key split can be done at any current key boundary. This also assures that lower level current nodes have only one parent. Index node splitting is illustrated in Figure 18.

可以在任何当前键边界进行键分割。这也确保了较低层级的当前节点只有一个父节点。索引节点分割如图18所示。

Unlike the WOBT (or Lanka and Mays [1991] or Becker et al. [1996]), the TSB-tree can move the contents of the historical node to another location in a separate historical database without updating more than one parent. No node which might be split in the future has more than one parent. If a node does a time split, the new address of the historical data from the old node can be placed in its unique parent and the old address can be used for the new current data. If it does a key split, the new key range for the old page can be posted along with the new key range and address.

与WOBT（或兰卡（Lanka）和梅斯（Mays）[1991]或贝克尔（Becker）等人[1996]）不同，TSB树可以将历史节点的内容移动到单独的历史数据库中的另一个位置，而无需更新多个父节点。未来可能会被分割的节点不会有多个父节点。如果一个节点进行时间分割，旧节点的历史数据的新地址可以放在其唯一的父节点中，而旧地址可以用于新的当前数据。如果进行键分割，可以将旧页面的新键范围与新键范围和地址一起张贴。

<!-- Media -->

<!-- figureText: key 9 25 40 now 18 time -->

<img src="https://cdn.noedgeai.com/0195c90d-0a33-750f-877f-c1cec48583bb_41.jpg?x=869&y=257&w=533&h=292&r=0"/>

a) original index page: rectangles represent key-time space of children

a) 原始索引页面：矩形表示子节点的键 - 时间空间

<!-- figureText: key now now time b) split by begin time of oldest current child key time c) split by a current key boundary -->

<img src="https://cdn.noedgeai.com/0195c90d-0a33-750f-877f-c1cec48583bb_41.jpg?x=841&y=612&w=600&h=663&r=0"/>

Figure 18. Index node splitting in the TSB-tree.

图18. TSB树中的索引节点分割。

<!-- Media -->

As with the WOBT, pure-key queries with time predicates are addressed in $O\left( {{\log }_{B}n + a}\right)$ I/Os,where $a$ represents the size of the answer.

与WOBT一样，带有时间谓词的纯键查询需要$O\left( {{\log }_{B}n + a}\right)$次输入/输出操作，其中$a$表示答案的大小。

Persistent B-Tree. Several methods [Lanka and Mays 1991; Becker et al. 1996; Varman and Verma 1997] were derived from a method by Driscoll et al. [1989] for general main-memory resident linked data structures. Driscoll et al. [1989] show how to take an "ephemeral data structure" (meaning that past states are erased when updates are made) and convert it to a "persistent data structure" (where past states are maintained). A "fully persistent" data structure allows updates to all past states. A "partially persistent" data structure allows updates only to the most recent state.

持久B树。几种方法[兰卡（Lanka）和梅斯（Mays）1991；贝克尔（Becker）等人1996；瓦尔曼（Varman）和维尔马（Verma）1997]源自德里斯科尔（Driscoll）等人[1989]针对一般驻留在主内存中的链接数据结构的方法。德里斯科尔等人[1989]展示了如何将一个“临时数据结构”（即进行更新时会擦除过去状态）转换为一个“持久数据结构”（即保留过去状态）。“完全持久”的数据结构允许对所有过去状态进行更新。“部分持久”的数据结构只允许对最近状态进行更新。

Consider the abstraction of a transaction time database as the "history of an evolving set of plain objects" (Figure 1). Assume that a ${\mathrm{B}}^{ + }$ -tree is used to index the initial state of this evolving set. If this ${\mathrm{B}}^{ + }$ -tree is made partially persistent, we have constructed an access method that supports transaction range-timeslice queries ("range/-/point"). Conceptually, a range-timeslice query for transaction time $t$ is answered by traversing the ${\mathrm{B}}^{ + }$ -tree as it was at $t$ . Partial persistence is nicely suited to transaction time, since only the most recent state is updated. Note that the method used to index the evolving set state affects what queries are addressed. For example, to construct a pure-timeslice method, the evolving set state is represented by a hashing function that is made partially persistent. This is another way to "visualize" the approach taken by the snapshot index.

将事务时间数据库抽象为“一组不断演变的普通对象的历史”（图1）。假设使用${\mathrm{B}}^{ + }$ -树对这一演变集合的初始状态进行索引。如果使该${\mathrm{B}}^{ + }$ -树具有部分持久性，我们就构建了一种支持事务范围时间片查询（“范围/ - /点”）的访问方法。从概念上讲，事务时间为$t$的范围时间片查询是通过遍历$t$时刻的${\mathrm{B}}^{ + }$ -树来完成的。部分持久性非常适合事务时间，因为只有最新状态会被更新。请注意，用于索引演变集合状态的方法会影响所处理的查询类型。例如，为了构建纯时间片方法，演变集合状态由一个具有部分持久性的哈希函数表示。这是“可视化”快照索引所采用方法的另一种方式。

Note that a fully persistent access structure can be restricted to the partially persistent case, which is the reason for discussing Driscoll et al. [1989] and Lanka and Mays [1991] in this survey.

请注意，完全持久化的访问结构可以限制为部分持久化的情况，这就是本调查中讨论德里斯科尔（Driscoll）等人[1989]以及兰卡（Lanka）和梅斯（Mays）[1991]研究成果的原因。

Lanka and Mays [1991] provide a fully persistent ${\mathrm{B}}^{ + }$ -tree. For our purposes, we are only interested in the methods presented in Lanka and Mays [1991] when reduced to partial persistence. Thus we call Lanka and Mays' [1991] partially persistent method the persistent B-tree. The multiversion B-tree (or MVBT) of Becker et al. [1996] and the MVAS of Varman and Verma [1997] are also partially persistent ${\mathrm{B}}^{ + }$ - trees. The Persistent B-tree and the MVBT, MVAS support node consolidation (that is, a page is consolidated with another page if it becomes sparse in alive keys due to frequent deletions). In comparison, the WOBT and the TSB-tree are partially persistent ${\mathrm{B}}^{ + }$ -trees, which do not do node consolidation (since they aim for applications where data is mainly updated and infrequently deleted). Node consolidation may result in thrashing (consolidating and splitting the same page continually), which results in more space. The MVBT, MVAS disallow thrashing, while the persistent B-tree does not.

兰卡和梅斯[1991]提出了一种完全持久化的${\mathrm{B}}^{ + }$ -树。就我们的目的而言，我们仅对兰卡和梅斯[1991]中简化为部分持久化的方法感兴趣。因此，我们将兰卡和梅斯[1991]的部分持久化方法称为持久化B树。贝克尔（Becker）等人[1996]的多版本B树（或MVBT）以及瓦尔曼（Varman）和维尔马（Verma）[1997]的MVAS也是部分持久化的${\mathrm{B}}^{ + }$ -树。持久化B树、MVBT和MVAS支持节点合并（即，如果一个页面由于频繁删除而导致存活键变得稀疏，则将其与另一个页面合并）。相比之下，WOBT和TSB -树是部分持久化的${\mathrm{B}}^{ + }$ -树，它们不进行节点合并（因为它们针对的是数据主要进行更新且很少删除的应用场景）。节点合并可能会导致颠簸（不断合并和拆分同一页面），从而占用更多空间。MVBT和MVAS不允许出现颠簸，而持久化B树则没有这一限制。

Driscoll et al. [1989]; Lanka and Mays [1991]; Becker et al. [1996]; and Varman and Verma [1997] speak of version numbers rather than timestamps. One important difference between version numbers for partially persistent data and timestamps is that timestamps as we have defined them are transaction time instants when events (changes) are stored. So timestamps are not consecutive integers; but version numbers can be consecutive integers. This has an effect on search operations, since Driscoll et al. [1989], Lanka and Mays [1991], and Becker et al. [1996] maintain an auxiliary structure called root*, which serves the same purpose as the "root log" of the WOBT.

德里斯科尔等人[1989]、兰卡和梅斯[1991]、贝克尔等人[1996]以及瓦尔曼和维尔马[1997]提到的是版本号而非时间戳。部分持久化数据的版本号和时间戳之间的一个重要区别在于，我们所定义的时间戳是存储事件（变更）的事务时间点。因此，时间戳不是连续的整数；而版本号可以是连续的整数。这会对搜索操作产生影响，因为德里斯科尔等人[1989]、兰卡和梅斯[1991]以及贝克尔等人[1996]维护了一个名为root*的辅助结构，其作用与WOBT的“根日志”相同。

In Driscoll et al. [1989], root* is an array indexed on version numbers. Each array entry has a pointer to the root of the version in question. If the version numbers are consecutive integers, search for the root is $O\left( 1\right)$ . If timestamps are used,search is $O\left( {{\log }_{B}n}\right)$ . In Lanka and Mays [1991] and Becker et al. [1996], root* only obtains entries when a root splits. Although root* is smaller than it would be if it had an entry for each timestamp, search within root* for the correct root is $O\left( {{\log }_{B}n}\right)$ .

在德里斯科尔等人[1989]的研究中，root*是一个以版本号为索引的数组。每个数组条目都有一个指向相关版本根节点的指针。如果版本号是连续的整数，对根节点的搜索复杂度为$O\left( 1\right)$。如果使用时间戳，搜索复杂度为$O\left( {{\log }_{B}n}\right)$。在兰卡和梅斯[1991]以及贝克尔等人[1996]的研究中，root*仅在根节点分裂时才会有新条目。尽管root*比为每个时间戳都设置一个条目的情况要小，但在root*中搜索正确的根节点的复杂度为$O\left( {{\log }_{B}n}\right)$。

The use of the root* structure (array) in Lanka and Mays [1991] and Becker et al. [1996] facilitates faster update processing, as the most current version of the ${\mathrm{B}}^{ + }$ -tree is separated from most of the previous versions. The most current root can have a separate pointer yielding $O\left( 1\right)$ access to that root. (Each root corresponds to a consecutive set of versions.) If the current version has size $m$ , updating is $O\left( {{\log }_{B}m}\right)$ . Methods that do not use the root* structure have $O\left( {{\log }_{B}n}\right)$ update processing.

兰卡和梅斯[1991]以及贝克尔等人[1996]使用root*结构（数组）有助于加快更新处理速度，因为${\mathrm{B}}^{ + }$ -树的最新版本与大多数先前版本是分离的。最新的根节点可以有一个单独的指针，从而以$O\left( 1\right)$的复杂度访问该根节点。（每个根节点对应一组连续的版本。）如果当前版本的大小为$m$，更新操作的复杂度为$O\left( {{\log }_{B}m}\right)$。不使用root*结构的方法的更新处理复杂度为$O\left( {{\log }_{B}n}\right)$。

Driscoll et al. [1989] explains how to make any ephemeral main-memory linked structure persistent. Two main methods are proposed: the fat node method and the node copying method. The fat node method keeps all the variations of a node in a variable-sized "fat node." When an ephemeral structure wants to update the contents of a node, the fat node method simply appends the new values to the old node, with a notation of the version number (timestamp) that does the update. When an ephemeral structure wants to create a new node, a new fat node is created.

德里斯科尔（Driscoll）等人 [1989] 阐述了如何将任何临时的主存链表结构变为持久化结构。文中提出了两种主要方法：胖节点法和节点复制法。胖节点法将节点的所有变体保存在一个可变大小的“胖节点”中。当临时结构想要更新节点内容时，胖节点法只需将新值追加到旧节点上，并标注进行更新的版本号（时间戳）。当临时结构想要创建新节点时，就会创建一个新的胖节点。

Lanka and Mays [1991] apply the fat node method of Driscoll et al. [1989] to the ${\mathrm{B}}^{ + }$ -tree. The fat nodes are collections of ${\mathrm{B}}^{ + }$ -tree pages,each corresponding to a set of versions. Versions can share ${\mathrm{B}}^{ + }$ -tree pages if the records in them are identical for each member of the sharing set. But versions with only one different data record have distinct ${\mathrm{B}}^{ + }$ -tree pages.

兰卡（Lanka）和梅斯（Mays） [1991] 将德里斯科尔等人 [1989] 提出的胖节点法应用于 ${\mathrm{B}}^{ + }$ -树。胖节点是 ${\mathrm{B}}^{ + }$ -树页面的集合，每个页面对应一组版本。如果共享集合中的每个成员的记录都相同，那么这些版本可以共享 ${\mathrm{B}}^{ + }$ -树页面。但只要有一条数据记录不同，这些版本就会有不同的 ${\mathrm{B}}^{ + }$ -树页面。

Pointers to lower levels of the structure are pointers to fat nodes, not to individual pages within fat nodes. When a record is updated, inserted, or deleted, a new leaf page is created. The new leaf is added to the old fat node. If the new leaf contents overflows, the new leaf is split, with the lower-value keys in the old fat node and the higher value keys in a new fat node. When a leaf splits, the parent node must be updated to search correctly for part of the new version that is in the new fat node. When a new page is added to a fat node, the parent need not be updated.

指向结构较低层级的指针是指向胖节点的，而不是指向胖节点内的单个页面。当记录被更新、插入或删除时，会创建一个新的叶页面。新叶会被添加到旧的胖节点中。如果新叶的内容溢出，新叶就会被拆分，较小值的键留在旧的胖节点中，较大值的键则放入新的胖节点中。当叶节点拆分时，必须更新父节点，以便正确搜索位于新胖节点中的新版本的部分内容。当新页面添加到胖节点时，无需更新父节点。

Similarly, when index nodes obtain new values because a fat node child has a split, new pages are allocated to the fat index node. When an index node wants to split, the parent of the index node obtains a new value. When roots split, a new pointer is put in the array root*, which allows access to the correct (fat) root nodes.

同样地，当索引节点因胖节点子节点拆分而获得新值时，会为胖索引节点分配新页面。当索引节点想要拆分时，索引节点的父节点会获得一个新值。当根节点拆分时，会在数组 root* 中放入一个新指针，这样就能访问正确的（胖）根节点。

Since search within a fat node means fetching all the pages in the fat node until the correct one is found (with the correct version number), Lanka and Mays [1991] suggest a version block: an auxiliary structure in each fat node of the persistent B-tree. The version block indicates which page or block in the fat node corresponds to which version number. Figure 19 shows the incremental creation of a version block with its fat node pages. In Figure 20, an update causes this version block to split. The version block is envisioned as one disk page, but there is no reason it might not become much larger. It may itself have to take the form of a multiway access tree (since new entries are always added at the end of a version block). Search in one version block for one data page could itself be $O\left( {{\log }_{B}n}\right)$ . For example, if all changes to the database were updates of one ${\mathrm{B}}^{ + }$ -tree page,the fat node would have $n{\mathrm{\;B}}^{ + }$ -tree pages in it.

由于在胖节点内进行搜索意味着要获取胖节点中的所有页面，直到找到正确的页面（具有正确的版本号），兰卡和梅斯 [1991] 提出了版本块的概念：这是持久化 B 树的每个胖节点中的一种辅助结构。版本块指示胖节点中的哪个页面或块对应哪个版本号。图 19 展示了带有胖节点页面的版本块的增量创建过程。在图 20 中，一次更新导致这个版本块拆分。版本块被设想为一个磁盘页面，但它也有可能变得更大。它本身可能必须采用多路访问树的形式（因为新条目总是添加到版本块的末尾）。在一个版本块中搜索一个数据页面本身的复杂度可能是 $O\left( {{\log }_{B}n}\right)$ 。例如，如果对数据库的所有更改都是对一个 ${\mathrm{B}}^{ + }$ -树页面的更新，那么胖节点中就会有 $n{\mathrm{\;B}}^{ + }$ -树页面。

Although search is no longer linear within the fat node, the path from the root to a leaf is at least twice as many blocks as it would be for an ephemeral structure. The height of the tree in blocks is at least twice what it would be for an ephemeral ${\mathrm{B}}^{ + }$ -tree containing the same data as one of the versions. Update processing is amortized $O\left( {{\log }_{B}m}\right)$ where $m$ is the size of the current ${\mathrm{B}}^{ + }$ - tree being updated. Range timeslice search is $O\left( {{\log }_{B}n\left( {{\log }_{B}m + a/B}\right) }\right)$ . After the correct root is found, the tree that was current at the time of interest is searched. This tree has height $O\left( {{\log }_{B}m}\right)$ ,and searching each version block in the path is $O\left( {{\log }_{B}n}\right)$ . A similar bound holds for the pure-timeslice query. Space is $O\left( n\right)$ (not $O\left( {n/B}\right)$ ),since new leaf blocks are created for each update.

尽管在胖节点内的搜索不再是线性的，但从根节点到叶节点的路径所经过的块数至少是临时结构的两倍。以块为单位衡量，树的高度至少是包含与其中一个版本相同数据的临时 ${\mathrm{B}}^{ + }$ -树的两倍。更新处理的均摊复杂度是 $O\left( {{\log }_{B}m}\right)$ ，其中 $m$ 是当前正在更新的 ${\mathrm{B}}^{ + }$ -树的大小。范围时间片搜索的复杂度是 $O\left( {{\log }_{B}n\left( {{\log }_{B}m + a/B}\right) }\right)$ 。找到正确的根节点后，会搜索在感兴趣的时间点有效的树。这棵树的高度是 $O\left( {{\log }_{B}m}\right)$ ，并且在路径中搜索每个版本块的复杂度是 $O\left( {{\log }_{B}n}\right)$ 。纯时间片查询也有类似的复杂度界限。空间复杂度是 $O\left( n\right)$ （而不是 $O\left( {n/B}\right)$ ），因为每次更新都会创建新的叶块。

To avoid creating new leaf blocks for each update, the "fat field method" is also proposed in Lanka and Mays [1991] where updates can fit in a space of nonfull pages. In the general full persistence case, each update must be marked with the version number that created it and with the version numbers of all later versions that delete it. Since we are interested in partial persistence, this corresponds to the start time and the end time of the update. Fat fields for the persistent B-tree are illustrated in Figure 21.

为避免每次更新都创建新的叶块，兰卡（Lanka）和梅斯（Mays）在1991年的论文中还提出了“胖字段方法”，该方法可使更新操作适配非满页空间。在一般的全持久化情况下，每次更新都必须标记创建它的版本号以及删除它的所有后续版本号。由于我们关注的是部分持久化，这对应于更新的开始时间和结束时间。持久化B树的胖字段如图21所示。

<!-- Media -->

<!-- figureText: key a) Every time a new version of the database is created by a transaction, a new data page is allocated to hold only those records alive after directs search to the correct data page. Here, for times before 5 , search goes to the first data page. At and after 5, it goes to the second data page. The record " ${c2}$ " is the updated version of the record " $\mathrm{c}$ ". A set of data pages with their version block is called a "fat node." b) At time 6, a new record, with key "g", is inserted. A new data page is allocated. The version block is updated to direct search at and after 6 to the third data page. c) This shows the fat node after an update is made on record "b" at time 7 . d) This shows the fat node after an update is made on records with b2,c3,f2,g keys "c" and "f" at time 8. $\begin{array}{lllll} {56} & 7 & 8 & 9 & {10} \end{array}$ time At time 0,the database contains records with keys h,f,c and b. At time 5, a new version is created which has an update to record c. At time 6 , record $g$ is inserted. Each time instant when a change occurs corresponds to version block 0.5 b,c,f,h b,c2, f,h $b,c,f,h$ fat node version block 0.567 b,c,f,h b,c2, f,h $b,{c2},f,g,h$ b2,c2,f,g,h fat node 0.5678 version block b,c,f,h $b,{c2},f,h$ b,c2,f,g,h b2,c2,f,g,h fat node -->

<img src="https://cdn.noedgeai.com/0195c90d-0a33-750f-877f-c1cec48583bb_44.jpg?x=347&y=238&w=972&h=1271&r=0"/>

Figure 19. Incremental creation of a fat node in the persistent B-tree.

图19. 持久化B树中胖节点的增量创建。

<!-- Media -->

When a page becomes full, the version creating the overflow copies all information relevant to that version to a new node. The persistent B-tree then creates a fat node and a version block. If the new copied node is still overflowing, a key split can be made and then information must be posted to the parent node regarding the key split and the new version. Thus the Persistent B-tree of Lanka and Mays [1991] does time splits and time-and-key splits just as in the WOBT. In this variation, space usage is $O\left( {n/B}\right)$ ,update processing is amortized $O\left( {{\log }_{B}m}\right)$ ,and query time (for both the range and pure-timeslice queries) is $O\left( {{\log }_{B}n\left( {{\log }_{B}m + a/B}\right) }\right)$ . The update and query time characteristics remain asymptotically the same as in the fat-node method, since the fat-field method still uses version blocks.

当一个页面满时，导致溢出的版本会将与该版本相关的所有信息复制到一个新节点。然后，持久化B树会创建一个胖节点和一个版本块。如果新复制的节点仍然溢出，可以进行键分裂，然后必须将有关键分裂和新版本的信息发布到父节点。因此，兰卡和梅斯在1991年提出的持久化B树与写时复制B树（WOBT）一样，会进行时间分裂和时间 - 键分裂。在这种变体中，空间使用率为$O\left( {n/B}\right)$，更新处理的均摊复杂度为$O\left( {{\log }_{B}m}\right)$，查询时间（包括范围查询和纯时间片查询）为$O\left( {{\log }_{B}n\left( {{\log }_{B}m + a/B}\right) }\right)$。由于胖字段方法仍然使用版本块，因此更新和查询时间特性在渐近意义上与胖节点方法相同。

<!-- Media -->

<!-- figureText: key version block b2,c2,f,g,h b2,c3,f2,g version block fat node version block version block b2,c2,f,g,h b2,c3,f2,g b2,c3,f2 8 time 0.5678 b,c,f,h b,c2, f,h b,c2,f,g,h fat node a) This fat node shows all the states of the database up to instant 8 from the last Figure. We assume a capacity of 5 records for each B+-tree page. At instant 9 records $\mathrm{i}$ and $\mathrm{j}$ are inserted,requiring a split. The old version block splits also _____ root* 0.9 b,c,f,h b,c2, f,h b,c2,f,g,h fat node b) A split. -->

<img src="https://cdn.noedgeai.com/0195c90d-0a33-750f-877f-c1cec48583bb_45.jpg?x=382&y=236&w=871&h=1128&r=0"/>

Figure 20. An example of a split on a fat node in the persistent B-tree.

图20. 持久化B树中胖节点分裂的示例。

<!-- Media -->

If a page becomes sparse from too many deletions, a node consolidation algorithm is presented. The version making the last deletion copies all information relative to that version to a new node. Then a sibling node also has its information relative to that version copied to the new node. If it is necessary, the new node is then key-split.

如果一个页面因过多删除操作而变得稀疏，则会采用节点合并算法。进行最后一次删除操作的版本会将与该版本相关的所有信息复制到一个新节点。然后，一个兄弟节点也会将与该版本相关的信息复制到新节点。如有必要，再对新节点进行键分裂。

Technically speaking, the possibility of thrashing by continually consolidating and splitting the same node could cause space usage to become $O\left( n\right)$ ,not $O\left( {n/B}\right)$ . This could happen by inserting a record in a node, causing it to time-and-key split, then deleting a record from one of the new current nodes and causing a node consolidation, which creates a new full current node, and so forth. A solution for thrashing appears in Snodgrass and Ahn [1985]. Basically, the threshold for node consolidation is made lower than half the threshold for node splitting. Since this is a rather pathological scenario, we continue to assume that space usage for the fat-fields variation of the persistent B-tree is $O\left( {n/B}\right)$ .

从技术上讲，不断对同一节点进行合并和分裂可能会导致系统颠簸，使空间使用率变为$O\left( n\right)$，而不是$O\left( {n/B}\right)$。例如，在一个节点中插入一条记录，导致其进行时间 - 键分裂，然后从其中一个新的当前节点中删除一条记录，导致节点合并，从而创建一个新的满的当前节点，依此类推。斯诺德格拉斯（Snodgrass）和安（Ahn）在1985年的论文中提出了解决颠簸问题的方法。基本上，节点合并的阈值设定为低于节点分裂阈值的一半。由于这是一种相当极端的情况，我们继续假设持久化B树的胖字段变体的空间使用率为$O\left( {n/B}\right)$。

<!-- Media -->

<!-- figureText: 10 11 13 root* i<9,#> j<9,#> version block f<0,8> f<8,#> g<6,#> h<0.8> f<8,#> g<9,13> g<13,#> A copy is made when a page overflows. Page capacity is five records. If there are four or more records in the new node, a key split is made after the copy. The root" structure points to the root for a given timestam Fat fields contain a key,a begin time,an end time and data. We shall not show the data. The "now" end time a<6,#> (<6,#> i<9,#> version block 11 b<0,#> C<0,5> c<5,#> f < 0, #> h<0,#> $b < 0,7 >$ b<7,#> c<5,8> c<8.#> e<10,#> b<7,13> c<11,#> e<10,#> The fat field method stores records from several versions as long as they fit in a page. At time 5 , a new version of record c is placed in the page. The old version gets 5 as its new end time. At time 6, overflow occurs. and $\mathbf{g}$ is updated, another copy and key split occur. This causes the new root to obtain a new index entry. At time 11, a new version of record c causes an overflow with only a copy, not a key split. Fat nodes are crea when the first copy operation is made. Search for a given time follows pointers which begin before or at the search time and do not end before the search time. Search in a version block (or in root*) follows the largest time. -->

<img src="https://cdn.noedgeai.com/0195c90d-0a33-750f-877f-c1cec48583bb_46.jpg?x=259&y=239&w=1118&h=1219&r=0"/>

Figure 21. The fat-field method of the persistent B-tree.

图21. 持久化B树的胖字段方法。

<!-- Media -->

To move historical data to another medium, observe that time splitting by current transaction time as performed in the persistent B-tree means that nodes cannot be moved once they are created, unless all the parents (not just the current ones) are updated with the new address of the historical data. Only the TSB-tree solves this problem by splitting index nodes before the time of the earliest start time of their current children. Thus, in the TSB-tree, when a current node is time-split, the historical data can be moved to another disk. In the TSB-tree, current nodes have only one parent.

要将历史数据迁移到其他介质，需注意持久化B树中按当前事务时间进行的时间分裂意味着节点一旦创建就不能移动，除非所有父节点（不仅仅是当前父节点）都用历史数据的新地址进行更新。只有时间片B树（TSB - tree）通过在当前子节点最早开始时间之前分裂索引节点解决了这个问题。因此，在时间片B树中，当一个当前节点进行时间分裂时，历史数据可以迁移到另一个磁盘。在时间片B树中，当前节点只有一个父节点。

Fat nodes are not necessary for partial persistence. This is observed in Driscoll et al. [1989], where "node-copying" for partially persistent structures is discussed.

对于部分持久化，胖节点并非必需。德里斯科尔（Driscoll）等人在1989年的论文中讨论了部分持久化结构的“节点复制”问题，其中提到了这一点。

The reason fat nodes are not needed is that although alive (current) nodes have many parents, only one of them is current. So when a current node is copied or split, only its current parent has to be updated. The other parents will correctly refer to its contents as of a previous version. The fact that new items may have been added does not affect the correctness of search. Since nodes are always time- split (most recent versions of copied items) by current transaction time, no information is erased when a time split is made.

不需要胖节点的原因是，虽然活跃（当前）节点有多个父节点，但只有一个是当前父节点。因此，当一个当前节点被复制或分裂时，只需更新其当前父节点。其他父节点仍会正确引用其先前版本的内容。新项的添加不会影响搜索的正确性。由于节点总是按当前事务时间进行时间分裂（复制项的最新版本），因此进行时间分裂时不会删除任何信息。

Both approaches of the persistent B-tree use a version block inside each fat node. If the node in question is never key-split (that is, all changes are applied to the same ephemeral ${\mathrm{B}}^{ + }$ -tree node), a new version of block pages may be created for this node, without updating the parent's version block. Thus, when answering a query, all encountered version blocks have to be searched for time $t$ . In comparison,the MVBT and MVAS we discuss next use "node-copying," and so have better asymptotic query time $\left( {O\left( {{\log }_{B}n + a/B}\right) }\right)$ .

持久B树的两种方法都在每个胖节点（fat node）内部使用一个版本块（version block）。如果所讨论的节点从未进行键分裂（即所有更改都应用于同一个临时${\mathrm{B}}^{ + }$ - 树节点），则可以为该节点创建新版本的块页面，而无需更新其父节点的版本块。因此，在回答查询时，必须在所有遇到的版本块中搜索时间$t$。相比之下，我们接下来讨论的多版本B树（MVBT）和多版本访问结构（MVAS）使用“节点复制”，因此具有更好的渐进查询时间$\left( {O\left( {{\log }_{B}n + a/B}\right) }\right)$。

Multiversion B-Tree and Multiversion Access Structure. The multiversion B-tree of Becker et al. [1996] and the multiversion access structure of Varman and Verma [1997] provide another approach to partially persistent ${\mathrm{B}}^{ + }$ -trees. Both structures have the same asymptotic behavior, but the MVAS improves the constant of MVBT's space complexity. We first discuss the $\overline{\mathrm{{MVBT}}}$ and then present its main differences with the MVAS.

多版本B树（Multiversion B - Tree）和多版本访问结构（Multiversion Access Structure）。贝克尔（Becker）等人[1996]提出的多版本B树和瓦尔曼（Varman）与维尔马（Verma）[1997]提出的多版本访问结构为部分持久${\mathrm{B}}^{ + }$ - 树提供了另一种方法。这两种结构具有相同的渐进行为，但多版本访问结构（MVAS）改善了多版本B树（MVBT）空间复杂度的常数。我们首先讨论$\overline{\mathrm{{MVBT}}}$，然后介绍它与多版本访问结构（MVAS）的主要区别。

The MVBT is similar to the WOBT; however, it efficiently supports deletions (as in Driscoll et al. [1989] and Lanka and Mays [1991]). Supporting deletions efficiently implies use of node consolidation. In addition, the MVBT uses a form of node-copying [Driscoll et al. 1989], and disallows thrashing.

多版本B树（MVBT）与无写操作B树（WOBT）类似；然而，它能有效地支持删除操作（如德里斯科尔（Driscoll）等人[1989]以及兰卡（Lanka）和梅斯（Mays）[1991]所述）。有效地支持删除操作意味着要使用节点合并。此外，多版本B树（MVBT）使用一种节点复制形式[德里斯科尔等人1989]，并且不允许颠簸现象。

As with the WOBT and the persistent B-tree, it uses a root* structure. When the root does a time-split, the sibling becomes a new root. Then a new entry is placed in the variable length array root*, pointing to the new root. If the root does a time-and-key split, the new tree has one more level. If a child of the root becomes sparse and merges with its only sibling, the newly merged node becomes a root of a new tree.

与无写操作B树（WOBT）和持久B树一样，它使用根*（root*）结构。当根节点进行时间分裂时，其兄弟节点成为新的根节点。然后在可变长度数组根*中放置一个新条目，指向新的根节点。如果根节点进行时间和键分裂，新树将多一层。如果根节点的一个子节点变得稀疏并与其唯一的兄弟节点合并，新合并的节点将成为新树的根节点。

Figures 21 and 22 illustrate some of the similarities and differences between the persistent B-tree, the MVBT, and the WOBT. To better illustrate the similarities, we picture the WOBT in Figure 22 with end times and start times in each record version. In the original WOBT, end times of records were calculated from the begin times of the next version of the record with the same key. If no such version was in the node, the end time of the record was known to be after the end time of the node.

图21和图22展示了持久B树、多版本B树（MVBT）和无写操作B树（WOBT）之间的一些相似之处和不同之处。为了更好地说明这些相似之处，我们在图22中描绘了无写操作B树（WOBT），每个记录版本都有结束时间和开始时间。在原始的无写操作B树（WOBT）中，记录的结束时间是根据具有相同键的下一个记录版本的开始时间计算得出的。如果节点中没有这样的版本，则已知该记录的结束时间在节点的结束时间之后。

In all three methods, if we have no node consolidation, the data nodes are exactly the same. In all three methods, when a node becomes full, a copy is made of all the records "alive"at the time the version makes the update that causes the overflow. If the number of distinct records in the copy is above some threshold, the copy is split into two nodes by key.

在这三种方法中，如果不进行节点合并，数据节点完全相同。在这三种方法中，当一个节点满时，会对版本进行导致溢出的更新时“存活”的所有记录进行复制。如果复制中的不同记录数量超过某个阈值，则通过键将该复制拆分为两个节点。

The persistent B-tree creates a fat node when a data node is copied. The WOBT and the MVBT do not create fat nodes. Instead, as illustrated in Figure 22, information is posted to the parent of the overflowing data node. A new index entry or two new index entries, which describe the split, are created. If there is only one new data node, the key used as the lower limit for the overflowing child is copied to the new index entry. The old child pointer gets the time of the copy as its end time and the new child pointer gets the split time as its start time. If there are two new children, they both have the same start time, but one has the key of the overflowing child and the other has the key used for the key split.

持久B树在复制数据节点时会创建一个胖节点（fat node）。无写操作B树（WOBT）和多版本B树（MVBT）不会创建胖节点。相反，如图22所示，信息会被发布到溢出数据节点的父节点。会创建一个新的索引条目或两个新的索引条目来描述拆分情况。如果只有一个新的数据节点，则将用作溢出子节点下限的键复制到新的索引条目中。旧的子节点指针将复制时间作为其结束时间，新的子节点指针将拆分时间作为其开始时间。如果有两个新的子节点，它们的开始时间相同，但一个具有溢出子节点的键，另一个具有用于键分裂的键。

A difference between the persistent B-tree, the WOBT, the MVBT, on one hand, and the TSB, on the other, is that the TSB does not have root*. When the only root in the TSB does a time split, a new level is placed on the tree to contain the information about the split. When the root in the MVBT does a time-split, root* obtains a new entry. When the root in the (fat-field) persistent B-tree does a time split, that root fat node obtains a new page and a new entry in the version block. (Only when the root fat node requires a key split or a merge, so that a new root fat node is constructed, does root* obtain a new entry in the Persistent B-tree.)

一方面，持久B树、无写操作B树（WOBT）和多版本B树（MVBT）与时间序列B树（TSB）的一个区别在于，时间序列B树（TSB）没有根*（root*）结构。当时间序列B树（TSB）中唯一的根节点进行时间分裂时，会在树上添加一个新层来包含有关分裂的信息。当多版本B树（MVBT）的根节点进行时间分裂时，根*会获得一个新条目。当（胖字段）持久B树的根节点进行时间分裂时，该根胖节点会获得一个新页面和版本块中的一个新条目。（只有当根胖节点需要进行键分裂或合并，从而构建一个新的根胖节点时，持久B树的根*才会获得一个新条目。）

<!-- Media -->

<!-- figureText: time h 9 13 -->

<img src="https://cdn.noedgeai.com/0195c90d-0a33-750f-877f-c1cec48583bb_48.jpg?x=489&y=232&w=577&h=311&r=0"/>

A copy is made when a page overflows. Page capacity is five records. If there are four or more records in the new node, a key split is made after the copy. The root* structure points to the root for a given timestamp.

当页面溢出时会进行复制。页面容量为五条记录。如果新节点中有四条或更多记录，则在复制后进行键分裂。根*结构指向给定时间戳的根节点。

Records contain a key, a begin time, an end time and data. We shall not show the data. The "now" end time is represented with a $\#$ sign.

记录包含一个键、一个开始时间、一个结束时间和数据。我们将不展示数据。“当前”结束时间用$\#$符号表示。

<!-- figureText: root* c<5,#> f < 0,#> h<0,#> f<6,9> $f < 9,\#  >$ i<9,#> f<8,#> g<6,#> $h < 0,8 >$ f<8,#> g<9,13> $g < {13},\#  >$ 0 b<0,#> C<0,5> 6 a<6,11> a<11,#> b<0,7> b<7,#> c<5,8> c<8,#> e<10,#> b<7,13> c<11,#> e<10,#> f<0.8> - -->

<img src="https://cdn.noedgeai.com/0195c90d-0a33-750f-877f-c1cec48583bb_48.jpg?x=331&y=678&w=970&h=609&r=0"/>

At time 5, a new version of record c is placed in the page. The old version gets 5 as its new end time. At time 6,

在时间5，记录c的新版本被放入页面。旧版本的结束时间更新为5。在时间6，

overflow occurs. There are four distinct keys in the new node, so a key split takes place. At time 9, when i and j are inserted and $g$ is updated,another copy and key split occur. This causes the new root to obtain two new index entries. At time 11, a new version of record c causes an overflow with only a copy, not a split. There is one new index entry.

发生溢出。新节点中有四个不同的键，因此进行了键分裂。在时间9，当插入i和j并更新$g$时，又发生了一次复制和键分裂。这使得新的根节点获得两个新的索引项。在时间11，记录c的新版本仅通过复制就导致了溢出，而没有进行分裂。有一个新的索引项。

Figure 22. The multiversion B-tree and the write-once B-tree. (For simplicity of comparison, both the end and start times appear in each record, which is not needed in the original WOBT).

图22. 多版本B树（Multiversion B-tree）和一次写入B树（Write-once B-tree）。（为了便于比较，每个记录中都显示了开始时间和结束时间，而在原始的一次写入B树中这并非必要。）

<!-- Media -->

Another difference with the WOBT is that the MVBT and the persistent B-tree use a node consolidation algorithm. When a node is sparse, it is consolidated with a sibling by time- splitting (copying) both the sparse node and its sibling and then combining the two copies, possibly key-splitting if necessary.

与一次写入B树的另一个区别是，多版本B树和持久化B树使用节点合并算法。当一个节点变得稀疏时，通过对稀疏节点及其兄弟节点进行时间分裂（复制），然后将两个副本合并，必要时可能进行键分裂，从而将其与兄弟节点合并。

In addition, the MVBT disallows thrashing (splitting and consolidating the same node continually) by suggesting that the threshold for splitting be higher than twice the threshold for consolidating. The persistent B-tree does not disallow thrashing. This is not an issue with the WOBT, since it does no node consolidation.

此外，多版本B树通过建议分裂阈值高于合并阈值的两倍来避免颠簸（不断对同一节点进行分裂和合并）。持久化B树不避免颠簸。这对于一次写入B树来说不是问题，因为它不进行节点合并。

Search in root* for the correct root in MBVT is $O\left( {{\log }_{B}n}\right)$ . Although the example illustrated in Figure 22 has a small root*, there is no reason why this should always be the case. We only need to imagine a database with one data node with records that are continually updated, causing the root (which is also the data node) to continually time-split. So if the root* becomes too large, a small index has to be created above it.

在多版本B树（MBVT）中，在根*中搜索正确的根的时间复杂度是$O\left( {{\log }_{B}n}\right)$。尽管图22所示的示例中根*较小，但没有理由认为情况总是如此。我们只需想象一个数据库，其中有一个数据节点，其记录不断更新，导致根节点（也是数据节点）不断进行时间分裂。因此，如果根*变得太大，就必须在其上方创建一个小的索引。

In MBVT, the transaction range time-slice search ("range/-/point") is $O\left( {{\log }_{B}n}\right.$ $+ a/B)$ ,since search for the root is itself $O\left( {{\log }_{B}n}\right)$ . The MVBT has $O\left( {{\log }_{B}m}\right)$ amortized update cost (where $m$ now denotes the size of the current ${\mathrm{B}}^{ + }$ -tree on which the update is performed), and $O\left( {n/B}\right)$ space usage. Thus the MVBT provides the I/O-optimal solution to the transaction range timeslice problem. The update cost is amortized due to the updating needed to maintain efficient access to the root* structure.

在多版本B树（MBVT）中，事务范围时间片搜索（“范围/-/点”）的时间复杂度是$O\left( {{\log }_{B}n}\right.$ $+ a/B)$，因为搜索根节点本身的时间复杂度是$O\left( {{\log }_{B}n}\right)$。多版本B树的平摊更新成本是$O\left( {{\log }_{B}m}\right)$（其中$m$现在表示当前进行更新操作的${\mathrm{B}}^{ + }$ -树的大小），空间使用是$O\left( {n/B}\right)$。因此，多版本B树为事务范围时间片问题提供了I/O最优解决方案。由于需要进行更新以保持对根*结构的高效访问，更新成本是平摊的。

The WOBT is not optimal because deletions of records can cause pages to be sparsely populated as of some recent time. Thus, transaction range-timeslice searches may become inefficient. Insertions in the WOBT and in the MVBT are $O\left( {{\log }_{B}m}\right)$ ,since a special pointer can be kept to the root of the tree containing all current records (that are $O\left( m\right) )$ .

一次写入B树不是最优的，因为记录的删除可能会导致页面在最近的某个时间变得稀疏。因此，事务范围时间片搜索可能会变得低效。一次写入B树和多版本B树的插入操作的时间复杂度是$O\left( {{\log }_{B}m}\right)$，因为可以保留一个特殊指针指向包含所有当前记录（即$O\left( m\right) )$）的树的根节点。

The MVBT uses more space than the $\mathrm{{WOBT}}$ ,which in turn uses more space than the TSB-tree. In order to guard against sparse nodes and thrashing, the MVBT policies create more replication (the constant in the $O\left( {n/B}\right)$ space worst-case bound of the method is about 10.)

多版本B树比$\mathrm{{WOBT}}$使用更多的空间，而$\mathrm{{WOBT}}$又比TSB树使用更多的空间。为了防止出现稀疏节点和颠簸，多版本B树的策略会产生更多的复制（该方法的最坏情况下$O\left( {n/B}\right)$空间复杂度的常数约为10）。

Probably the best variation of the WOBT is to use some parameters to decide whether to time-split, time-and-key split, key-split, time-split and merge, or time-split, merge and key-split. These parameters depend on the minimum number of versions alive at any time in the interval spanned by the page. All of the policies pit disk space usage against query time. A pure key-split creates one new page. A time-and-key split creates two new pages: one new historical page and one new current page. The historical page will have copies of the current records, so more copies are made than when pure key splits are allowed. Node consolidation creates at least two new historical pages. However, once a minimum number of records is guaranteed to be alive for any given version in all pages, range-timeslice queries are $O\left( {{\log }_{B}n + }\right.$ $a/B)$ and space usage is $O\left( {n/B}\right)$ . Different splitting policies will affect the total amount of space used and the average number of copies of record versions made.

可能WOBT（工作负载优化B树，Workload-Optimized B-Tree）的最佳变体是使用一些参数来决定是进行时间分割、时间和键分割、键分割、时间分割并合并，还是时间分割、合并再进行键分割。这些参数取决于页面所跨越的时间间隔内任意时刻存活版本的最小数量。所有这些策略都是在磁盘空间使用和查询时间之间进行权衡。纯键分割会创建一个新页面。时间和键分割会创建两个新页面：一个新的历史页面和一个新的当前页面。历史页面会有当前记录的副本，因此与允许纯键分割时相比，会创建更多的副本。节点合并至少会创建两个新的历史页面。然而，一旦保证所有页面中任何给定版本都有最少数量的记录存活，范围时间片查询的复杂度为$O\left( {{\log }_{B}n + }\right.$ $a/B)$，并且空间使用为$O\left( {n/B}\right)$。不同的分割策略会影响所使用的总空间量以及记录版本的平均副本数量。

The multiversion access structure (MVAS) [Varman and Verma 1997] is similar to the MVBT, but it achieves a smaller constant on the space bound by using better policies to handle the cases where key-splits or merges are performed. There are two main differences in the MVAS policies. The first deals with the case when a node becomes sparse after performing a record deletion. Instead of always consuming a new page (as in the MVBT), the MVAS tries to find a sibling with free space where the remaining alive entries of the time-split page can be stored. The conditions under which this step is carried out are described in detail in Varman and Verma [1997]. The second difference deals with the case when the number of entries in a just time-split node is below the prespecified threshold. If a sibling page has enough alive records, the MVBT copies all the sibling's alive records to the sparse time-split page, thus "deleting" the sibling page. Instead, the MVAS copies only as many alive records as needed from the sibling page for the time-split page to avoid violating the threshold. The above two modifications reduce the extent of duplication, hence reducing the overall space. As a result, the MVAS reduces the worst-case storage bound of MVBT by a factor of 2 .

多版本访问结构（MVAS，Multiversion Access Structure）[Varman和Verma 1997]与MVBT（多版本B树，Multiversion B-Tree）类似，但它通过使用更好的策略来处理进行键分割或合并的情况，在空间界限上实现了更小的常数。MVAS策略有两个主要区别。第一个区别处理的是在执行记录删除后节点变得稀疏的情况。与MVBT总是使用一个新页面不同，MVAS会尝试找到一个有空闲空间的兄弟节点，将时间分割页面中剩余的存活条目存储在那里。执行这一步骤的条件在Varman和Verma [1997]中有详细描述。第二个区别处理的是刚进行时间分割的节点中的条目数量低于预设阈值的情况。如果一个兄弟页面有足够多的存活记录，MVBT会将该兄弟页面的所有存活记录复制到稀疏的时间分割页面，从而“删除”该兄弟页面。相反，MVAS只会从兄弟页面复制足够数量的存活记录到时间分割页面，以避免违反阈值。上述两个修改减少了重复的程度，从而减少了总体空间。因此，MVAS将MVBT的最坏情况存储界限降低了一半。

Since the WOBT, TSB-tree, persistent B-tree, MVBT, and MVAS are similar in their approach to solving range-time-slice queries, we summarize their characteristics in Table I. The issues of time-split, key-split, time- and key-split, sparse nodes, thrashing, and history migration are closely related.

由于WOBT、TSB树（时间分割B树，Time-Split B-Tree）、持久B树（Persistent B-Tree）、MVBT和MVAS在解决范围时间片查询的方法上相似，我们在表I中总结了它们的特点。时间分割、键分割、时间和键分割、稀疏节点、颠簸和历史迁移等问题密切相关。

<!-- Media -->

Table I. Basic Characteristics of WOBT, TSB-Tree, Persistent B-Tree, MVBT, and MVAS

表I. WOBT、TSB树、持久B树、MVBT和MVAS的基本特点

<table><tr><td/><td>time split ${}^{1}$</td><td>pure key split ${}^{2}$</td><td>time/key split</td><td>sparse node merge</td><td>prevent thrashing ${}^{3}$</td><td>root** ${}^{4}$</td><td>history migrate ${}^{5}$</td></tr><tr><td>WOBT</td><td>yes</td><td>no</td><td>yes</td><td>no</td><td>N.A.</td><td>yes</td><td>no</td></tr><tr><td>TSB-Tree</td><td>yes</td><td>yes</td><td>no</td><td>no</td><td>N.A.</td><td>no</td><td>yes</td></tr><tr><td>Persistent B-Tree</td><td>yes</td><td>no</td><td>yes</td><td>yes</td><td>no</td><td>yes</td><td>no</td></tr><tr><td>MVBT/MVAS</td><td>yes</td><td>no</td><td>yes</td><td>yes</td><td>yes</td><td>yes</td><td>no</td></tr></table>

<table><tbody><tr><td></td><td>时间分割 ${}^{1}$</td><td>纯密钥分割 ${}^{2}$</td><td>时间/密钥分割</td><td>稀疏节点合并</td><td>防止颠簸 ${}^{3}$</td><td>根** ${}^{4}$</td><td>历史迁移 ${}^{5}$</td></tr><tr><td>WOBT</td><td>是</td><td>否</td><td>是</td><td>否</td><td>不适用</td><td>是</td><td>否</td></tr><tr><td>TSB树</td><td>是</td><td>是</td><td>否</td><td>否</td><td>不适用</td><td>否</td><td>是</td></tr><tr><td>持久化B树（Persistent B-Tree）</td><td>是</td><td>否</td><td>是</td><td>是</td><td>否</td><td>是</td><td>否</td></tr><tr><td>多版本B树/多版本访问结构（MVBT/MVAS）</td><td>是</td><td>否</td><td>是</td><td>是</td><td>是</td><td>是</td><td>否</td></tr></tbody></table>

1. All methods time-split (copy) data and index nodes. The TSB-tree can time-split by other than current time.

1. 所有方法都会对数据和索引节点进行时间分割（复制）。TSB树可以按当前时间以外的时间进行时间分割。

2. The TSB-Tree does pure key splits. The other methods do time-and-key splits. Pure key splits use less total space, but risk poor performance on past-time queries.

2. TSB树进行纯键分割。其他方法进行时间 - 键分割。纯键分割使用的总空间较少，但在过去时间查询时可能性能不佳。

3. Thrashing is repeated merging and splitting of the same node. Only the MBVT prevents thrashing by choice of splitting and merging thresholds. Prevention of thrashing is not needed when there is no merging.

3. 颠簸是指同一节点反复进行合并和分割。只有MBVT通过选择分割和合并阈值来防止颠簸。当没有合并操作时，不需要防止颠簸。

4. The use of root* enables the current tree search to be more efficient by keeping a separate pointer to its root. Past time queries must search within root*, so are not more efficient than methods without root*.

4. 使用根指针*（root*）通过保留一个指向根节点的单独指针，使当前树搜索更高效。过去时间查询必须在根指针*范围内搜索，因此并不比没有根指针*的方法更高效。

5. Only the TSB-tree has only one reference to current nodes, allowing historial data to migrate.

5. 只有TSB树对当前节点只有一个引用，允许历史数据迁移。

<!-- Media -->

The pure-key query is not addressed in the work of Driscoll et al. [1989]; Lanka and Mays [1991]; and Becker et al. [1996]; however, the technique that keeps the address of any one copy of the most recent distinct previous version of the record with each record can avoid going through all copies of a record. The pure-key query (with time predicate) is then addressed in $O\left( {{\log }_{B}n + a/B}\right)$ I/Os, just as proposed for the WOBT (where $a$ represents the number of different versions of the given key).

德里斯科尔（Driscoll）等人[1989]、兰卡（Lanka）和梅斯（Mays）[1991]以及贝克尔（Becker）等人[1996]的工作中未涉及纯键查询。然而，为每条记录保留该记录最近不同先前版本的任意一份副本的地址的技术可以避免遍历记录的所有副本。然后，纯键查询（带时间谓词）可以在$O\left( {{\log }_{B}n + a/B}\right)$次输入/输出（I/O）操作内完成，就像为WOBT所提出的那样（其中$a$表示给定键的不同版本数量）。

As discussed in Section 5.1.1, Varman and Verma [1997] solve the pure-key query (with time predicate) in optimal query time $O\left( {{\log }_{B}n + a/B}\right)$ using C-lists. An advantage of the C-lists, despite their extra complexity in maintenance, is that they can be combined with the main MVAS method.

如5.1.1节所述，瓦尔曼（Varman）和维尔马（Verma）[1997]使用C列表在最优查询时间$O\left( {{\log }_{B}n + a/B}\right)$内解决了纯键查询（带时间谓词）问题。尽管C列表在维护上额外复杂，但它们的一个优点是可以与主要的多版本访问结构（MVAS）方法结合使用。

Exodus and Overlapping ${B}^{ + }$ . Trees. The overlapping ${\mathrm{B}}^{ + }$ -tree [Manolopoulos and Kapetanakis 1990; Burton et al. 1985] and the Exodus large storage object [Richardson et al. 1986] are similar. We begin here with a ${\mathrm{B}}^{ + }$ -tree. When a new version makes an update in a leaf page, copies are made of the full path from the leaf to the root, changing references as necessary. Each new version has a separate root and subtrees may be shared (Figure 23).

出埃及记（Exodus）和重叠${B}^{ + }$树。重叠${\mathrm{B}}^{ + }$树[马诺洛普洛斯（Manolopoulos）和卡佩塔纳基斯（Kapetanakis）1990；伯顿（Burton）等人1985]和出埃及记大型存储对象[理查森（Richardson）等人1986]类似。我们从${\mathrm{B}}^{ + }$树开始。当新版本对叶页面进行更新时，会复制从叶节点到根节点的完整路径，并在必要时更改引用。每个新版本都有一个单独的根节点，子树可以共享（图23）。

Space usage is $O\left( {n{\log }_{B}n}\right)$ ,since new pages are created for the whole path leading to each data page updated by a new version. Update processing is $O\left( {{\log }_{B}m}\right)$ ,where $m$ is the size of the current tree being updated. Timeslice or range-timeslice query time depends on the time needed to find the correct root. If nonconsecutive transaction timestamps of events are used, it is $O\left( {{\log }_{B}n + a/B}\right)$ .

空间使用量为$O\left( {n{\log }_{B}n}\right)$，因为会为新版本更新的每个数据页面的整个路径创建新页面。更新处理复杂度为$O\left( {{\log }_{B}m}\right)$，其中$m$是正在更新的当前树的大小。时间片或范围时间片查询时间取决于找到正确根节点所需的时间。如果使用事件的非连续事务时间戳，查询时间为$O\left( {{\log }_{B}n + a/B}\right)$。

Even though pure-key queries of the form "find the previous salary history of employee $A$ who was alive at $t$ " (i.e., with time predicate) are not discussed, they can in principle be addressed in the same way as the other B+-tree-based methods by linking data records together.

尽管未讨论“查找在$t$时在职的员工$A$的先前薪资历史”这种形式的纯键查询（即带时间谓词），但原则上可以通过将数据记录链接在一起，以与其他基于B + 树的方法相同的方式处理此类查询。

Multiattribute Indexes. Suppose that the transaction start time, transaction end time, and database key are used as a triplet key for a multiat-tribute point structure. If this structure clusters records in disk pages by closeness in several attributes, one can obtain efficient transaction pure-timeslice and range-timeslice queries using only one copy of each record.

多属性索引。假设将事务开始时间、事务结束时间和数据库键用作多属性点结构的三元组键。如果该结构根据多个属性的接近程度将记录聚集在磁盘页面中，那么仅使用每条记录的一份副本就可以实现高效的事务纯时间片和范围时间片查询。

<!-- Media -->

<!-- figureText: root of root of newversion shared new old version shared shared old -->

<img src="https://cdn.noedgeai.com/0195c90d-0a33-750f-877f-c1cec48583bb_51.jpg?x=173&y=233&w=615&h=424&r=0"/>

Figure 23. The Overlapping tree/Exodus structure.

图23. 重叠树/出埃及记结构。

<!-- Media -->

Records with similar values of start time, end time, and key are clustered together in disk pages. Having both a similar start time and a similar end time means that long-lived records will be in the same page as other long-lived records. These records are answers to many timeslice queries. Short-lived records will only be on the same pages if their short lives are close in time. These contain many correct answers to time-slice queries with time values in the short interval that their entries span. Every timeslice query accesses some of the long-lived record pages and a small proportion of the short-lived record pages. Individual timeslice queries do not need to access most of the short-lived record pages, as they do not intersect the timeslice.

开始时间、结束时间和键值相似的记录会聚集在磁盘页面中。开始时间和结束时间都相似意味着长生命周期记录会与其他长生命周期记录位于同一页面。这些记录是许多时间片查询的答案。短生命周期记录只有在其短生命周期在时间上接近时才会位于同一页面。这些页面包含许多时间值在其条目所跨越的短时间间隔内的时间片查询的正确答案。每个时间片查询都会访问一些长生命周期记录页面和一小部分短生命周期记录页面。单个时间片查询不需要访问大多数短生命周期记录页面，因为它们与时间片不相交。

There are some subtle problems with this. Suppose a data page is split by start time. In one of the pages resulting from the split, all the record versions whose start time is before the split time are stored. This page has an upper bound on start time, implying that no new record versions can be inserted. All new record versions will have a start time after now, which is certainly after split time. Further, if there are current records in this page, their end time will continue to rise, so the lengths of the record time spans in this page will be variable.

这存在一些微妙的问题。假设一个数据页按开始时间进行分割。在分割产生的其中一个页面中，存储了所有开始时间早于分割时间的记录版本。这个页面的开始时间有一个上限，这意味着不能插入新的记录版本。所有新的记录版本的开始时间都将晚于当前时间，而当前时间肯定晚于分割时间。此外，如果这个页面中有当前记录，它们的结束时间将继续增加，因此这个页面中记录时间跨度的长度将是可变的。

<!-- Media -->

<!-- figureText: key -->

<img src="https://cdn.noedgeai.com/0195c90d-0a33-750f-877f-c1cec48583bb_51.jpg?x=835&y=220&w=623&h=221&r=0"/>

Figure 24. Storing data with similar start_times.

图24. 存储具有相似开始时间的数据。

<!-- Media -->

Some are long and others short. Queries as of current transaction time may only retrieve a few (or no) records from a page that is limited by an upper bound on start time. This is illustrated in Figure 24. Many such pages may have to be accessed in order to answer a query, each one contributing very little to the answer (i.e., the answer is not clustered well in pages).

有些时间跨度长，有些则短。在当前事务时间进行的查询，可能只能从一个开始时间有上限的页面中检索到少量（或没有）记录。这在图24中有所说明。为了回答一个查询，可能需要访问许多这样的页面，每个页面为查询结果贡献的内容很少（即，查询结果在页面中没有很好地聚集）。

Also, when a new version is created, its start time is often far from the start time of its predecessor (the previous version with the same key). So consecutive versions of the same record are unlikely to be on the same page if start-time splits are used.

此外，当创建一个新版本时，其开始时间往往与它的前一个版本（具有相同键的上一个版本）的开始时间相差甚远。因此，如果使用开始时间分割，同一记录的连续版本不太可能在同一页面上。

Now suppose we decide that splitting by start time is a bad idea and we split only by key or by end time. Splitting by end time enables migration of past data to a WORM disk. However, a query of a past transaction time may only retrieve a small number of records if the records are placed only by the requirement of having an end time before some cut-off value, just as in Figure 12.

现在假设我们认为按开始时间分割不是一个好主意，而只按键或结束时间进行分割。按结束时间分割可以将过去的数据迁移到一次写入多次读取（WORM）磁盘。然而，如果记录仅根据结束时间在某个截止值之前的要求进行放置，那么对过去事务时间的查询可能只能检索到少量记录，就像图12所示的那样。

Current pages (which were split by key) can contain versions whose lifetimes are very long and versions whose lifetimes are very short. This also makes past-time queries inefficient.

当前页面（按键分割的页面）可以包含生命周期非常长的版本和生命周期非常短的版本。这也会使过去时间的查询效率低下。

All of these subtle problems come from the fact that many records are still current and have growing lifetimes and all new record versions have increasing start times. Perhaps if we use a point-based multiattribute index for dead versions only, efficient clustering may be possible. Here newly dead record versions can be inserted in a page with an upper limit on start time because the start times may have been long ago. Items can be clustered in pages by key, nearby start times, and nearby end times. No guarantee can be made that a query as of a given time will hit a minimum number of record versions in a page, however. For example, imagine a page with record versions with very short lifetimes, all of which are close by but none of which overlap.

所有这些微妙的问题都源于这样一个事实：许多记录仍然是当前记录，并且其生命周期在不断增长，而所有新的记录版本的开始时间都在不断增加。也许如果我们仅对已失效的版本使用基于点的多属性索引，就有可能实现高效的聚集。在这里，新失效的记录版本可以插入到开始时间有上限的页面中，因为这些开始时间可能是很久以前的。记录项可以按键、相近的开始时间和相近的结束时间在页面中进行聚集。然而，不能保证在给定时间的查询能在一个页面中命中最少数量的记录版本。例如，想象一个页面中的记录版本生命周期都非常短，它们都很接近，但没有一个是重叠的。

Although no guarantee of worst-case search time can be made, the advantages of having only one copy of each record and no overlapping of time-key space, so that backtracking is not necessary, may make this approach worthwhile, at least for "dead" versions. Space usage is thus linear (space is $O\left( {n/B}\right)$ ,if in addition the multiat-tribute method can guarantee that index and data pages have good space utilization). A method for migrating current data to the WORM and organizing the current data for efficient temporal queries is needed if the multiat-tribute method was used for past data only.

虽然不能保证最坏情况下的搜索时间，但每个记录只有一份副本且时间 - 键空间不重叠，从而无需回溯，这些优点可能使这种方法值得一试，至少对于“已失效”的版本是如此。因此，空间使用是线性的（如果多属性方法还能保证索引和数据页有良好的空间利用率，那么空间复杂度为$O\left( {n/B}\right)$）。如果多属性方法仅用于过去的数据，那么就需要一种将当前数据迁移到WORM磁盘并组织当前数据以进行高效时态查询的方法。

5.1.4 Summary. The worst-case performance of the transaction-time methods is summarized in Table II. The reader should be cautious when interpreting worst-case performance; the notation sometimes penalizes a method for its performance on a pathological scenario. The footnotes indicate such cases.

5.1.4 总结。事务时间方法的最坏情况性能总结在表II中。读者在解释最坏情况性能时应谨慎；这种表示法有时会因为一种方法在病态场景下的性能而对其进行惩罚。脚注指出了此类情况。

5.1.5 Declustering and Bulk Loading. The query bounds presented in Table II assume that queries are processed in a uniprocessor environment. Query performance can be substantially improved if historical data is spread (declustered) across a number of disks that are then accessed in parallel. Temporal data offers an ideal declustering predicate based on time. This idea is explored in Kouramajian et al. [1994] and Muth et al. [1996]. Muth et al. [1996] present a way to decluster TSB pages. The declus-tering method, termed LoT, attempts to assign logically consecutive leaf (data) pages of a TSB tree into a number of separate disks. When a new data page is created in a TSB tree, it is allocated a disk address based on the disk addresses used by its neighboring pages. Various worst-case performance guarantees are derived. Simulation results show a large benefit over random data page allocation. Another declustering approach appears in Kourmajian et al. [1994], where time-based declustering is presented for the time-index. For details we refer to Muth et al. [1996] and Kouramajian et al. [1994].

5.1.5 分散存储和批量加载。表II中给出的查询边界假设查询是在单处理器环境中处理的。如果将历史数据分散（分散存储）到多个磁盘上，然后并行访问这些磁盘，查询性能可以得到显著提高。时态数据提供了一个基于时间的理想分散存储谓词。Kouramajian等人[1994]和Muth等人[1996]探讨了这个想法。Muth等人[1996]提出了一种分散存储TSB页面的方法。这种分散存储方法称为LoT，它试图将TSB树中逻辑上连续的叶子（数据）页面分配到多个单独的磁盘上。当在TSB树中创建一个新的数据页面时，会根据其相邻页面使用的磁盘地址为其分配一个磁盘地址。推导了各种最坏情况下的性能保证。模拟结果表明，与随机数据页分配相比，这种方法有很大的优势。另一种分散存储方法出现在Kouramajian等人[1994]的研究中，其中针对时间索引提出了基于时间的分散存储方法。有关详细信息，请参考Muth等人[1996]和Kouramajian等人[1994]的研究。

Most transaction-time access methods take advantage of the time-ordered changes to achieve good update performance. The update processing comparison presented in Table II is in terms of a single update. Faster updating can be achieved if updates are buffered and then applied to the index in bulks of work. The log-structured history data access method (LHAM) [Neil and Wei-kum 1993] aims to support extremely high update rates. It partitions data into successive components based on transaction-time timestamps. To achieve fast update rates, the most recent data is clustered together by transaction start-time as it arrives. A simple (B+-tree-like) index is used to index such data quickly; since it is based on transaction start-time and is not very efficient for the queries we examine here. As data gets older, it migrates (using an efficient technique termed rolling merge) to another component where the authors propose using other, more efficient, indexes (like the TSB-tree, etc.). Van den Bercken et al. [1997] recently proposed a generic algorithm to quickly create an index for a presumably large data set. This algorithm can be applied to various index structures (the authors have applied it to R-trees and on MVBT trees). Using the basic index structure, buffers are used at each node of the index. As changes arrive they are buffered on the root and, as this buffer fills, changes are propagated in page units to buffers in lower parts of the index, until the leaves are reached. Using this approach, the total update cost for $n$ changes becomes $O\left( {\left( {n/B}\right) {\log }_{c}n}\right)$ where $c$ is the number of pages available in main-memory, i.e., it is more efficient than inserting updates one by one (where instead the total update processing is $O\left( {n{\log }_{B}n}\right)$ .

大多数事务时间访问方法利用按时间顺序的更改来实现良好的更新性能。表II中给出的更新处理比较是基于单次更新的。如果将更新进行缓冲，然后批量应用到索引中，则可以实现更快的更新。日志结构历史数据访问方法（LHAM）[尼尔和魏库姆，1993年]旨在支持极高的更新率。它根据事务时间戳将数据划分为连续的组件。为了实现快速更新率，最新的数据在到达时会按事务开始时间聚集在一起。使用一个简单的（类似B +树的）索引来快速对这些数据进行索引；由于它基于事务开始时间，因此对于我们在此研究的查询而言效率不是很高。随着数据变旧，它会（使用一种称为滚动合并的高效技术）迁移到另一个组件，作者建议在该组件中使用其他更高效的索引（如TSB树等）。范登·贝肯等人[1997年]最近提出了一种通用算法，用于为可能很大的数据集快速创建索引。该算法可以应用于各种索引结构（作者已将其应用于R树和MVBT树）。使用基本的索引结构时，在索引的每个节点处都使用缓冲区。当更改到达时，它们会在根节点处进行缓冲，当该缓冲区填满时，更改会以页面为单位传播到索引较低部分的缓冲区，直到到达叶子节点。使用这种方法，$n$次更改的总更新成本变为$O\left( {\left( {n/B}\right) {\log }_{c}n}\right)$，其中$c$是主内存中可用的页面数，即它比逐个插入更新更高效（逐个插入更新时总更新处理成本为$O\left( {n{\log }_{B}n}\right)$）。

<!-- Media -->

Table II. Performance Characteristics of Examined Transaction-Time Methods

表II. 所研究的事务时间方法的性能特征

<table><tr><td>Access Method (related section)</td><td>Total Space</td><td>Update per change</td><td>Pure-key Query</td><td>Pure-Timeslice Query</td><td>Range-Timeslice Query</td></tr><tr><td>AP-Tree (5.1.2)</td><td>$O\left( {n/B}\right)$</td><td>$O{\left( {\log }_{B}n\right) }^{1}$</td><td>$N/A$</td><td>$O\left( {n/B}\right)$</td><td>$O\left( {n/B}\right)$</td></tr><tr><td>ST-Tree (5.1.2)</td><td>$O\left( {n/B}\right)$</td><td>$O{\left( {\log }_{B}S\right) }^{2}$</td><td>$O{\left( {\log }_{B}S + a\right) }^{2}$</td><td>$O{\left( S{\log }_{B}n\right) }^{2}$</td><td>$O{\left( K{\log }_{B}n\right) }^{3}$</td></tr><tr><td>Time-Index (5.1.2)</td><td>$O\left( {{n}^{2}/B}\right)$</td><td>$O\left( {n/B}\right)$</td><td>$N/A$</td><td>$O\left( {{\log }_{B}n + a/B}\right)$</td><td>$O{\left( {\log }_{B}n + s/B\right) }^{4}$</td></tr><tr><td>Two-level Time (5.1.2)</td><td>$O\left( {{n}^{2}/B}\right)$</td><td>$O\left( {n/B}\right)$</td><td>$N/A$</td><td>$O{\left( R{\log }_{B}n + a\right) }^{5}$</td><td>$O{\left( M{\log }_{B}n + a\right) }^{6}$</td></tr><tr><td>Checkpoint Index(5.1.2) ${}^{7}$</td><td>$O\left( {n/B}\right)$</td><td>$O\left( {n/B}\right)$</td><td>$N/A$</td><td>$O\left( {n/B}\right)$</td><td>$O\left( {n/B}\right)$</td></tr><tr><td>Archivable Time ${\left( {5.1}.2\right) }^{8}$</td><td>$O\left( {n/B}\right)$</td><td>$O\left( {{\log }_{B}n}\right)$</td><td>$N/A$</td><td>$O\left( {{\log }_{2}n + a/B}\right)$</td><td>$O{\left( {\log }_{2}n + s/B\right) }^{4}$</td></tr><tr><td>Snapshot Index (5.1.2)</td><td>$O\left( {n/B}\right)$</td><td>$O{\left( 1\right) }^{9}$</td><td>$O{\left( a\right) }^{10}$</td><td>$O\left( {{\log }_{B}n + a/B}\right)$</td><td>$O{\left( {\log }_{B}n + s/B\right) }^{4}$</td></tr><tr><td>Windows Method (5.1.2)</td><td>$O\left( {n/B}\right)$</td><td>$O\left( {{\log }_{B}n}\right)$</td><td>$O\left( {{\log }_{B}n + a}\right)$</td><td>$O\left( {{\log }_{B}n + a/B}\right)$</td><td>$O{\left( {\log }_{B}n + s/B\right) }^{4}$</td></tr><tr><td>R-Trees (5.1.3)</td><td>$O\left( {n/B}\right)$</td><td>$O\left( {{\log }_{B}n}\right)$</td><td>$O{\left( n/B\right) }^{11}$</td><td>$O{\left( n/B\right) }^{11}$</td><td>$O{\left( n/B\right) }^{11}$</td></tr><tr><td>SR-Tree (5.1.3)</td><td>$O\left( {\left( {n/B}\right) {\log }_{B}n}\right)$</td><td>$O\left( {\log {n}_{B}}\right)$</td><td>$O{\left( n/B\right) }^{11}$</td><td>$O{\left( n/B\right) }^{11}$</td><td>$O{\left( n/B\right) }^{11}$</td></tr><tr><td>WOBT(5.1.3) ${}^{12}$</td><td>$O\left( {n/B}\right)$</td><td>$O{\left( {\log }_{B}m\right) }^{13}$</td><td>$O{\left( {\log }_{B}n + a\right) }^{14}$</td><td>$O\left( {{\log }_{B}n + a/B}\right)$</td><td>$O\left( {{\log }_{B}n + a/B}\right)$</td></tr><tr><td>TSB-Tree (5.1.3)</td><td>$O\left( {n/B}\right)$</td><td>$O\left( {{\log }_{B}n}\right)$</td><td>$O{\left( {\log }_{B}n + a\right) }^{14}$</td><td>$O{\left( n/B\right) }^{15}$</td><td>$O{\left( n/B\right) }^{15}$</td></tr><tr><td>Persistent B-tree/Fat Node (5.1.3)</td><td>$O\left( n\right)$</td><td>$O{\left( {\log }_{B}m\right) }^{13}$</td><td>$O\left( {{\log }_{B}n{\log }_{B}m}\right.  +$ $a{)}^{{14},{16}}$</td><td>$O\left( {{\log }_{B}n\left( {{\log }_{B}m + }\right. }\right.$ $a{\left( B\right) }^{16}$</td><td>$O\left( {{\log }_{B}n\left( {{\log }_{B}m + }\right. }\right.$ $a{\left( B\right) }^{16}$</td></tr><tr><td>Persistent B-tree/Fat Field (5.1.3)</td><td>$O\left( {n/B}\right)$</td><td>$O{\left( {\log }_{B}m\right) }^{13}$</td><td>$O\left( {{\log }_{B}n{\log }_{B}m + }\right.$ $a{)}^{{14},{16}}$</td><td>$O\left( {{\log }_{B}n\left( {{\log }_{B}m + }\right. }\right.$ $a{\left( B\right) }^{16}$</td><td>$O\left( {{\log }_{B}n\left( {{\log }_{B}m}\right. }\right.  +$ $a{\left( B\right) }^{16}$</td></tr><tr><td>MVBT(5.1.3)</td><td>$O\left( {n/B}\right)$</td><td>$O{\left( {\log }_{B}m\right) }^{13}$</td><td>$O{\left( {\log }_{B}n + a\right) }^{14}$</td><td>$O\left( {{\log }_{B}n + a/B}\right)$</td><td>$O\left( {{\log }_{B}n + a/B}\right)$</td></tr><tr><td>MVAS(5.1.1 & 5.1.3)</td><td>$O\left( {n/B}\right)$</td><td>$O{\left( {\log }_{B}m\right) }^{13}$</td><td>$O\left( {{\log }_{B}n + a/B}\right)$</td><td>${7O}\left( {{\log }_{B}n + a/B}\right)$</td><td>$O\left( {{\log }_{B}n + a/B}\right)$</td></tr><tr><td>Overlapping B-Tree (5.1.3)</td><td>$O\left( {n{\log }_{B}n}\right)$</td><td>$O{\left( {\log }_{B}m\right) }^{13}$</td><td>$O{\left( {\log }_{B}n + a\right) }^{14}$</td><td>$O\left( {{\log }_{B}n + a/B}\right)$</td><td>$O\left( {{\log }_{B}n + a/B}\right)$</td></tr></table>

<table><tbody><tr><td>访问方法（相关章节）</td><td>总空间</td><td>每次更改时更新</td><td>纯键查询</td><td>纯时间片查询</td><td>范围时间片查询</td></tr><tr><td>AP树（5.1.2）</td><td>$O\left( {n/B}\right)$</td><td>$O{\left( {\log }_{B}n\right) }^{1}$</td><td>$N/A$</td><td>$O\left( {n/B}\right)$</td><td>$O\left( {n/B}\right)$</td></tr><tr><td>ST树（5.1.2）</td><td>$O\left( {n/B}\right)$</td><td>$O{\left( {\log }_{B}S\right) }^{2}$</td><td>$O{\left( {\log }_{B}S + a\right) }^{2}$</td><td>$O{\left( S{\log }_{B}n\right) }^{2}$</td><td>$O{\left( K{\log }_{B}n\right) }^{3}$</td></tr><tr><td>时间索引（5.1.2）</td><td>$O\left( {{n}^{2}/B}\right)$</td><td>$O\left( {n/B}\right)$</td><td>$N/A$</td><td>$O\left( {{\log }_{B}n + a/B}\right)$</td><td>$O{\left( {\log }_{B}n + s/B\right) }^{4}$</td></tr><tr><td>两级时间（5.1.2）</td><td>$O\left( {{n}^{2}/B}\right)$</td><td>$O\left( {n/B}\right)$</td><td>$N/A$</td><td>$O{\left( R{\log }_{B}n + a\right) }^{5}$</td><td>$O{\left( M{\log }_{B}n + a\right) }^{6}$</td></tr><tr><td>检查点索引（5.1.2） ${}^{7}$</td><td>$O\left( {n/B}\right)$</td><td>$O\left( {n/B}\right)$</td><td>$N/A$</td><td>$O\left( {n/B}\right)$</td><td>$O\left( {n/B}\right)$</td></tr><tr><td>可存档时间 ${\left( {5.1}.2\right) }^{8}$</td><td>$O\left( {n/B}\right)$</td><td>$O\left( {{\log }_{B}n}\right)$</td><td>$N/A$</td><td>$O\left( {{\log }_{2}n + a/B}\right)$</td><td>$O{\left( {\log }_{2}n + s/B\right) }^{4}$</td></tr><tr><td>快照索引（5.1.2）</td><td>$O\left( {n/B}\right)$</td><td>$O{\left( 1\right) }^{9}$</td><td>$O{\left( a\right) }^{10}$</td><td>$O\left( {{\log }_{B}n + a/B}\right)$</td><td>$O{\left( {\log }_{B}n + s/B\right) }^{4}$</td></tr><tr><td>窗口方法（5.1.2）</td><td>$O\left( {n/B}\right)$</td><td>$O\left( {{\log }_{B}n}\right)$</td><td>$O\left( {{\log }_{B}n + a}\right)$</td><td>$O\left( {{\log }_{B}n + a/B}\right)$</td><td>$O{\left( {\log }_{B}n + s/B\right) }^{4}$</td></tr><tr><td>R树（5.1.3）</td><td>$O\left( {n/B}\right)$</td><td>$O\left( {{\log }_{B}n}\right)$</td><td>$O{\left( n/B\right) }^{11}$</td><td>$O{\left( n/B\right) }^{11}$</td><td>$O{\left( n/B\right) }^{11}$</td></tr><tr><td>SR树（5.1.3）</td><td>$O\left( {\left( {n/B}\right) {\log }_{B}n}\right)$</td><td>$O\left( {\log {n}_{B}}\right)$</td><td>$O{\left( n/B\right) }^{11}$</td><td>$O{\left( n/B\right) }^{11}$</td><td>$O{\left( n/B\right) }^{11}$</td></tr><tr><td>WOBT（5.1.3） ${}^{12}$</td><td>$O\left( {n/B}\right)$</td><td>$O{\left( {\log }_{B}m\right) }^{13}$</td><td>$O{\left( {\log }_{B}n + a\right) }^{14}$</td><td>$O\left( {{\log }_{B}n + a/B}\right)$</td><td>$O\left( {{\log }_{B}n + a/B}\right)$</td></tr><tr><td>TSB树（5.1.3）</td><td>$O\left( {n/B}\right)$</td><td>$O\left( {{\log }_{B}n}\right)$</td><td>$O{\left( {\log }_{B}n + a\right) }^{14}$</td><td>$O{\left( n/B\right) }^{15}$</td><td>$O{\left( n/B\right) }^{15}$</td></tr><tr><td>持久B树/胖节点（5.1.3）</td><td>$O\left( n\right)$</td><td>$O{\left( {\log }_{B}m\right) }^{13}$</td><td>$O\left( {{\log }_{B}n{\log }_{B}m}\right.  +$ $a{)}^{{14},{16}}$</td><td>$O\left( {{\log }_{B}n\left( {{\log }_{B}m + }\right. }\right.$ $a{\left( B\right) }^{16}$</td><td>$O\left( {{\log }_{B}n\left( {{\log }_{B}m + }\right. }\right.$ $a{\left( B\right) }^{16}$</td></tr><tr><td>持久B树/胖字段（5.1.3）</td><td>$O\left( {n/B}\right)$</td><td>$O{\left( {\log }_{B}m\right) }^{13}$</td><td>$O\left( {{\log }_{B}n{\log }_{B}m + }\right.$ $a{)}^{{14},{16}}$</td><td>$O\left( {{\log }_{B}n\left( {{\log }_{B}m + }\right. }\right.$ $a{\left( B\right) }^{16}$</td><td>$O\left( {{\log }_{B}n\left( {{\log }_{B}m}\right. }\right.  +$ $a{\left( B\right) }^{16}$</td></tr><tr><td>MVBT（5.1.3）</td><td>$O\left( {n/B}\right)$</td><td>$O{\left( {\log }_{B}m\right) }^{13}$</td><td>$O{\left( {\log }_{B}n + a\right) }^{14}$</td><td>$O\left( {{\log }_{B}n + a/B}\right)$</td><td>$O\left( {{\log }_{B}n + a/B}\right)$</td></tr><tr><td>MVAS（5.1.1 & 5.1.3）</td><td>$O\left( {n/B}\right)$</td><td>$O{\left( {\log }_{B}m\right) }^{13}$</td><td>$O\left( {{\log }_{B}n + a/B}\right)$</td><td>${7O}\left( {{\log }_{B}n + a/B}\right)$</td><td>$O\left( {{\log }_{B}n + a/B}\right)$</td></tr><tr><td>重叠B树（5.1.3）</td><td>$O\left( {n{\log }_{B}n}\right)$</td><td>$O{\left( {\log }_{B}m\right) }^{13}$</td><td>$O{\left( {\log }_{B}n + a\right) }^{14}$</td><td>$O\left( {{\log }_{B}n + a/B}\right)$</td><td>$O\left( {{\log }_{B}n + a/B}\right)$</td></tr></tbody></table>

${}^{1}$ This is the time needed when the end_time of a stored interval is updated; it is assumed that the start_time of the updated interval is given. If intervals can be identified by some key attribute, then a hashing function could find the updated interval at $O\left( 1\right)$ expected amortized time. In the original paper it was assumed that intervals can only be added and in increasing start_time order; in that case the update time is $O\left( 1\right)$ .

${}^{1}$ 这是更新已存储区间的结束时间所需的时间；假设已给出更新区间的开始时间。如果可以通过某个键属性来识别区间，那么哈希函数可以在 $O\left( 1\right)$ 期望均摊时间内找到更新的区间。在原论文中，假设只能按递增的开始时间顺序添加区间；在这种情况下，更新时间为 $O\left( 1\right)$。

${}^{2}$ Where $S$ denotes the number of different keys (surrogates) ever created in the evolution.

${}^{2}$ 其中 $S$ 表示在演化过程中创建的不同键（代理键）的数量。

${}^{3}$ Where $K$ denotes the number of keys in the query key range (which may or may not be alive at the time of interest). ${}^{4}$ Where $s$ denotes the size of the whole timeslice for the time of interest. No separation of the key space in regions is assumed.

${}^{3}$ 其中 $K$ 表示查询键范围中的键的数量（这些键在感兴趣的时间点可能处于活跃状态，也可能不处于活跃状态）。${}^{4}$ 其中 $s$ 表示感兴趣时间点的整个时间片的大小。假设不将键空间划分为多个区域。

${}^{5}$ Where $R$ is the number of predefined key regions.

${}^{5}$ 其中 $R$ 是预定义的键区域的数量。

${}^{6}$ Assuming that a query contains a number of predefined key-regions, $M$ denotes the number of regions in the query range.

${}^{6}$ 假设一个查询包含多个预定义的键区域，$M$ 表示查询范围中的区域数量。

${}^{7}$ The performance is under the assumption that the Checkpoint Index creates very few checkpoints and the space remains linear. The update time is $O\left( {n/B}\right)$ since when the end_time of a stored interval is updated,the interval has to be found. As with the AP-Index, if intervals can be identified by some key attribute, then a hashing function could find the updated interval at $O\left( 1\right)$ . The original paper did not deal with this issue since it was implicitly assumed that interval endpoints are known at insertion. If checkpoints are often then the method will behave as the Time Index.

${}^{7}$ 该性能是在检查点索引创建的检查点非常少且空间保持线性的假设下得出的。更新时间为 $O\left( {n/B}\right)$，因为在更新已存储区间的结束时间时，必须找到该区间。与 AP 索引一样，如果可以通过某个键属性来识别区间，那么哈希函数可以在 $O\left( 1\right)$ 时间内找到更新的区间。原论文没有处理这个问题，因为隐式假设在插入时就已知区间端点。如果经常创建检查点，那么该方法的行为将类似于时间索引。

${}^{8}$ For the update it is assumed that the start_time of the updated interval is known. Otherwise,if intervals can be identified by some key, a hashing function could be used to find the start_time of the updated interval. For the range-timeslice query, we assume no extra structure is used. The original paper proposes using an approach similar to the Two-Level Time Index or the ST-Tree.

${}^{8}$ 对于更新操作，假设已知更新区间的开始时间。否则，如果可以通过某个键来识别区间，则可以使用哈希函数来查找更新区间的开始时间。对于范围 - 时间片查询，我们假设不使用额外的结构。原论文提出使用类似于两级时间索引或 ST 树的方法。

${}^{9}$ In the expected amortized sense,using a hashing function on the object key space. If no hashing but a B-tree is used then the update becomes $O\left( {{\log }_{B}m}\right)$ ,where $m$ is the size of the current state,on which the update is performed

${}^{9}$ 在期望均摊意义上，对对象键空间使用哈希函数。如果不使用哈希函数而是使用 B 树，那么更新操作的时间复杂度变为 $O\left( {{\log }_{B}m}\right)$，其中 $m$ 是执行更新操作时当前状态的大小。

${}^{10}$ Assuming as in 9 that a hashing function is used. If a B-tree is used the query becomes $O\left( {{\log }_{B}S + a}\right)$ ,where $S$ is the total number of keys ever created.

${}^{10}$ 假设如第 9 点所述使用了哈希函数。如果使用 B 树，查询操作的时间复杂度变为 $O\left( {{\log }_{B}S + a}\right)$，其中 $S$ 是曾经创建的键的总数。

${}^{11}$ This is a pathological worst case,due to the non-guaranteed search on an R-tree based structure. In most cases the avg. performance would be $O\left( {{\log }_{B}n + a}\right)$ . Note that all the R-tree related methods assume both interval endpoints are known at insertion time.

${}^{11}$ 这是一种病态的最坏情况，这是由于基于 R 树的结构的搜索不具有保证性。在大多数情况下，平均性能为 $O\left( {{\log }_{B}n + a}\right)$。请注意，所有与 R 树相关的方法都假设在插入时已知区间的两个端点。

${}^{12}$ Here we assume that the WOBT tree is implemented thoroughly on a magnetic disk,and that no (or infrequent) deletions occur, i.e., just additions and updates.

${}^{12}$ 这里我们假设 WOBT 树完全在磁盘上实现，并且不发生（或很少发生）删除操作，即只进行添加和更新操作。

${}^{13}$ In the amortized sense,where $m$ denotes the size of the current tree being updated.

${}^{13}$ 在均摊意义上，其中 $m$ 表示正在更新的当前树的大小。

${}^{14}$ For a pure-key query of the form: "find the previous salaries of employee $A$ who existed at time $t$ ".

${}^{14}$ 对于形式为“查找在时间 $t$ 存在的员工 $A$ 的过往工资”的纯键查询。

${}^{15}$ This is a pathological worst case,where only key-splits are performed. If a time-split is performed before a key-split

${}^{15}$ 这是一种病态的最坏情况，其中只进行键分裂操作。如果在键分裂之前进行时间分裂

when nodes resulting from a pure key-split would have too few records "alive" at the begin time of the node, then the query takes $O\left( {{\log }_{B}n + a}\right)$ ,also assuming infrequent deletions.

当纯键分裂产生的节点在节点开始时间“活跃”的记录太少时，那么查询时间为 $O\left( {{\log }_{B}n + a}\right)$，同样假设很少发生删除操作。

${}^{16}$ Where $m$ denotes the size of the ephemeral B+-tree at the time of interest.

${}^{16}$ 其中 $m$ 表示在感兴趣的时间点临时 B + 树的大小。

${}^{17}$ The pure-key query performance assumes the existence of the C-lists on top of the MVAS structure.

${}^{17}$ 纯键查询性能假定在多版本访问结构（MVAS）之上存在C列表。

<!-- Media -->

5.1.6 Temporal Hashing. External dynamic hashing has been used in traditional database systems as a fast access method for membership queries. Given a dynamic set $D$ of objects,a membership query asks whether an object with identity $z$ is in the most current $D$ . While such a query can still be answered by a B+-tree, hashing is on average much faster (usually one I/O instead of the usual two to three I/Os for traversing the tree index). Note, however. that the worst-case hashing performance is linear to the size of $D$ , while the B+-tree guarantees logarithmic access. Nevertheless, due to its practicality, hashing is a widely used access method. An interesting question is whether temporal hashing has an efficient solution. In this setting, changes to the dynamic set are timestamped and the membership query has a temporal predicate, as in "find whether object with identity $z$ was in the set $D$ at time $t$ ." Kollios and Tsotras [1998] present an efficient solution to this problem by extending traditional linear hashing [Litwin 1980] to a transaction-time environment.

5.1.6 时态哈希。外部动态哈希已在传统数据库系统中用作成员查询的快速访问方法。给定一个对象的动态集合$D$，成员查询会询问具有标识$z$的对象是否在最新的$D$中。虽然这样的查询仍然可以通过B +树来回答，但哈希平均而言要快得多（通常是一次I/O，而遍历树索引通常需要两到三次I/O）。然而，需要注意的是，哈希的最坏情况性能与$D$的大小呈线性关系，而B +树保证对数级的访问。尽管如此，由于其实用性，哈希是一种广泛使用的访问方法。一个有趣的问题是时态哈希是否有高效的解决方案。在这种情况下，对动态集合的更改会被打上时间戳，并且成员查询具有时态谓词，例如“查找具有标识$z$的对象在时间$t$是否在集合$D$中”。科利奥斯（Kollios）和索特拉斯（Tsotras）[1998]通过将传统的线性哈希[利特温（Litwin）1980]扩展到事务时间环境，为这个问题提出了一个高效的解决方案。

### 5.2 Valid-Time Methods

### 5.2 有效时间方法

According to the valid-time abstraction presented in Section 2, a valid-time database should maintain a dynamic collection of interval-objects. Arge and Vit-ter [1996] recently presented an I/O optimal solution for the "*/point/-" query. The solution (the external interval tree) is based on a main-memory data structure, the interval tree that is made external (disk-resident)[Edels-brunner 1983]. Valid timeslices are supported in $O\left( {l/B}\right)$ space,using $O\left( {{\log }_{B}l}\right)$ update per change (interval addition, deletion,or modification) and $O\left( {\log }_{B}\right.$ $l + a/B$ ) query time. Here $l$ is the number of interval-objects in the database when the update or query is performed. Even though it is not clear how practical the solution is (various details are not included in the original paper), the result is very interesting. To optimally support valid timeslices is a rather difficult problem because in a valid-time environment the clustering of data in pages can dramatically change the updates. Deletions are now physical and insertions can happen anywhere in the valid-time domain. In contrast, in a transaction-time environment objects are inserted in increasing time order and after their insertion they can be "logically" deleted, but they are not removed from the database.

根据第2节中提出的有效时间抽象，有效时间数据库应维护一个区间对象的动态集合。阿尔格（Arge）和维特（Vitter）[1996]最近为“*/点/-”查询提出了一种I/O最优解决方案。该解决方案（外部区间树）基于一种主存数据结构，即区间树，并将其扩展到外部（驻留在磁盘上）[埃德尔布鲁纳（Edelsbrunner）1983]。使用$O\left( {{\log }_{B}l}\right)$每次更改（区间添加、删除或修改）的更新操作和$O\left( {\log }_{B}\right.$ $l + a/B$）查询时间，在$O\left( {l/B}\right)$空间中支持有效时间片。这里$l$是执行更新或查询时数据库中区间对象的数量。尽管目前尚不清楚该解决方案的实际实用性如何（原始论文中未包含各种细节），但这个结果非常有趣。要最优地支持有效时间片是一个相当困难的问题，因为在有效时间环境中，页面中数据的聚类会显著改变更新操作。现在删除操作是物理性的，并且插入操作可以在有效时间域的任何位置发生。相比之下，在事务时间环境中，对象按时间递增顺序插入，插入后可以“逻辑上”删除，但不会从数据库中移除。

A valid timeslice query ("*/point/-") is actually a special case of a two-dimensional range query. Note that an interval contains a query point $v$ if and only if its start_time is less than or equal to $v$ and its end_time is greater than or equal to $v$ . Let us map an interval $I =$ $\left( {{x}_{1},{y}_{1}}\right)$ into a point $\left( {{x}_{1},{y}_{1}}\right)$ in the two-dimensional space. Then an interval contains query $v$ if and only if its corresponding two-dimensional point lies inside the box generated by lines $x = 0$ , $x = v,y = v$ ,and $y = \infty$ (Figure 25). Since an interval's end_time is always greater or equal than its start_time, all intervals are represented by points above the diagonal $x = y$ . This two-dimensional mapping is used in the priority search tree [McCreight 1985], the data structure that provides the main-memory optimal solution. A number of attempts have been made to externalize this structure [Kanellakis et al. 1993; Icking et al. 1987; Blankenagel and Guting 1990].

有效时间片查询（“*/点/-”）实际上是二维范围查询的一种特殊情况。请注意，一个区间包含查询点$v$当且仅当它的开始时间小于或等于$v$，并且它的结束时间大于或等于$v$。让我们将一个区间$I =$ $\left( {{x}_{1},{y}_{1}}\right)$映射到二维空间中的一个点$\left( {{x}_{1},{y}_{1}}\right)$。那么一个区间包含查询点$v$当且仅当它对应的二维点位于由直线$x = 0$、$x = v,y = v$和$y = \infty$生成的矩形内（图25）。由于一个区间的结束时间总是大于或等于它的开始时间，所有区间都由对角线$x = y$上方的点表示。这种二维映射用于优先搜索树[麦格雷特（McCreight）1985]，这是一种提供主存最优解决方案的数据结构。已经有许多尝试将这种结构扩展到外部[卡内拉基斯（Kanellakis）等人1993；伊金（Icking）等人1987；布兰克纳格尔（Blankenagel）和古廷（Guting）1990]。

Kanellakis et al. [1993] uses the above two-dimensional mapping to address two problems: indexing constraints and indexing classes in an I/O environment. Constraints are represented as intervals that can be added, modified, or deleted. The problem of indexing constraints is then reduced to the dynamic interval management problem, i.e., the "*/point/-" query! For solving the dynamic interval management problem, Kanellakis et al. [1993] introduces a new access method, the metab-lock tree, which is a B-ary access method that partitions the upper diagonal of the two-dimensional space into metablocks,each of which has ${B}^{2}$ data points (the structure is rather complex; for details see Kanellakis et al. [1993]). Note however that the metablock tree is a semidynamic structure, since it can support only interval insertions (no deletions). It uses $O\left( {l/B}\right)$ space, $O\left( {\log }_{B}\right.$ $l + a/B)$ query time,and $O\left( {{\log }_{B}l + }\right.$ $\left. {{\left( {\log }_{B}l\right) }^{2}/B}\right)$ amortized insertion time. The insertion bound is amortized because the maintenance of a metablock's internal organization is rather complex to perform after each insertion. Instead, metablock reorganizations are deferred until enough insertions have accumulated. If interval insertions are random, the expected insertion time becomes $O\left( {{\log }_{B}l}\right)$ .

卡内拉基斯（Kanellakis）等人 [1993] 使用上述二维映射来解决两个问题：在输入/输出（I/O）环境中对约束条件和类别进行索引。约束条件以区间的形式表示，这些区间可以被添加、修改或删除。这样，约束条件的索引问题就简化为动态区间管理问题，即“*/点/-”查询！为了解决动态区间管理问题，卡内拉基斯等人 [1993] 引入了一种新的访问方法——元块树（metab - lock tree），这是一种B元访问方法，它将二维空间的上对角线划分为元块，每个元块有 ${B}^{2}$ 个数据点（该结构相当复杂；详细信息请参阅卡内拉基斯等人 [1993]）。不过要注意，元块树是一种半动态结构，因为它仅支持区间插入（不支持删除）。它使用 $O\left( {l/B}\right)$ 的空间、$O\left( {\log }_{B}\right.$ $l + a/B)$ 的查询时间以及 $O\left( {{\log }_{B}l + }\right.$ $\left. {{\left( {\log }_{B}l\right) }^{2}/B}\right)$ 的平摊插入时间。插入时间的界限是平摊的，因为每次插入后维护元块的内部组织相当复杂。相反，元块的重组会推迟到积累了足够多的插入操作之后进行。如果区间插入是随机的，那么预期插入时间变为 $O\left( {{\log }_{B}l}\right)$。

<!-- Media -->

<!-- figureText: ${x}_{1}$ -->

<img src="https://cdn.noedgeai.com/0195c90d-0a33-750f-877f-c1cec48583bb_55.jpg?x=290&y=240&w=384&h=270&r=0"/>

Figure 25. An interval is translated into a point in a two-dimensional space. Axes $x$ and $y$ represent an interval's starting and ending valid-times. Intervals that intersect valid instant $v$ correspond to the points included in the shaded area.

图25。一个区间被转换为二维空间中的一个点。坐标轴 $x$ 和 $y$ 分别表示一个区间的起始和结束有效时间。与有效时刻 $v$ 相交的区间对应于阴影区域内的点。

<!-- Media -->

Icking et al. [1987] and Blankenagel and Guting [1990] present two other external implementations of the priority search tree. Both use optimal space $\left( {O\left( {l/B}\right) }\right)$ ; Icking et al. [1987] has $O\left( {{\log }_{2}l + a/B}\right)$ query time I/Os for valid timeslices, while Blankenagel and Guting [1990] has $O\left( {{\log }_{B}l + a}\right)$ query time.

伊金（Icking）等人 [1987] 以及布兰克纳格尔（Blankenagel）和古廷（Guting） [1990] 提出了优先搜索树的另外两种外部实现方式。这两种方式都使用最优空间 $\left( {O\left( {l/B}\right) }\right)$；伊金等人 [1987] 对于有效时间片的查询时间为 $O\left( {{\log }_{2}l + a/B}\right)$ 次输入/输出操作，而布兰克纳格尔和古廷 [1990] 的查询时间为 $O\left( {{\log }_{B}l + a}\right)$。

In Ramaswamy and Subramanian [1994], a new technique called path caching is introduced for solving two-dimensional range queries. This technique is used to turn various main-memory data structures, like the interval tree or the priority search tree [McCreight 1985], into external structures. With this approach, the "*/point/-" query is addressed in $O\left( {{\log }_{B}l + a/B}\right)$ query time, $O\left( {{\log }_{B}l}\right)$ amortized update time (including insertions and deletions),but in $O(\left( {l/B}\right)$ ${\log }_{2}{\log }_{2}B$ ) space.

在拉马什瓦米（Ramaswamy）和苏布拉马尼亚姆（Subramanian） [1994] 的研究中，引入了一种名为路径缓存（path caching）的新技术来解决二维范围查询问题。该技术用于将各种主存数据结构，如区间树（interval tree）或优先搜索树 [麦格雷特（McCreight）1985]，转换为外部结构。采用这种方法，“*/点/-”查询的查询时间为 $O\left( {{\log }_{B}l + a/B}\right)$，平摊更新时间（包括插入和删除）为 $O\left( {{\log }_{B}l}\right)$，但需要 $O(\left( {l/B}\right)$ ${\log }_{2}{\log }_{2}B$ 的空间。

The above approaches are aimed at good worst-case bounds, but lead to rather complex structures. Another main-memory data-structure that solves the "*/point/-" query is the segment tree [Bentley 1977] which, however, uses more than linear space. Blankenagel and Gut-ing [1994] present the external segment tree (EST), which is a paginated version of the segment tree. We first describe the worst-case performance of the EST method. If the endpoints of the valid-time intervals take values from a universe of size $V$ (i.e.,there are $V$ possible endpoint values), the EST supports "*/point/-" queries using $O\left( {\left( {l/B}\right) {\log }_{2}V}\right)$ space, $O\left( {{\log }_{2}V}\right)$ update per change, and query time $O\left( {{\log }_{2}V + a}\right)$ . Blankenagel and Guting [1994] also present an extended analysis of the expected behavior of the EST under the assumption of a uniformly distributed set of intervals of fixed length. It is shown that the expected behavior is much better; the average height of the EST is, for all practical purposes, small (this affects the logarithmic portion of the performance) and the answer is found by accessing an additional $O\left( {a/B}\right)$ pages.

上述方法旨在实现良好的最坏情况边界，但会导致结构相当复杂。另一种解决“*/点/-”查询的主存数据结构是线段树（segment tree） [本特利（Bentley）1977]，不过它使用的空间超过线性空间。布兰克纳格尔和古廷 [1994] 提出了外部线段树（EST），它是线段树的分页版本。我们首先描述EST方法的最坏情况性能。如果有效时间区间的端点取值来自大小为 $V$ 的全集（即有 $V$ 种可能的端点值），EST支持“*/点/-”查询，使用 $O\left( {\left( {l/B}\right) {\log }_{2}V}\right)$ 的空间，每次更改的更新时间为 $O\left( {{\log }_{2}V}\right)$，查询时间为 $O\left( {{\log }_{2}V + a}\right)$。布兰克纳格尔和古廷 [1994] 还在固定长度区间集均匀分布的假设下，对EST的预期行为进行了扩展分析。结果表明，预期行为要好得多；实际上，EST的平均高度较小（这会影响性能的对数部分），并且通过额外访问 $O\left( {a/B}\right)$ 个页面就能找到答案。

An advantage of the external segment tree is that the method can be modified to also address queries with key predicates (like the "range/point/-" query). This is performed by embedding B-trees in the EST. The original EST structure guides the search to a subset of intervals that contain the query valid time $v$ while an embedded B-tree allows searching this subset for whether the query key predicate is also satisfied. For details see Blankenagel and Guting [1994].

外部线段树（External Segment Tree，EST）的一个优点是，该方法可以进行修改，以处理带有键谓词的查询（如“范围/点/-”查询）。这可以通过在EST中嵌入B树来实现。原始的EST结构会引导搜索到包含查询有效时间 $v$ 的一个区间子集，而嵌入的B树则允许在这个子集中搜索查询键谓词是否也得到满足。详细信息请参阅Blankenagel和Guting [1994]。

Good average-case performance could also be achieved by using a dynamic multidimensional access method. If only multidimensional points are supported, as in the k-d-B-tree [Robinson 1984] or the h-B-tree [Lomet and Salzberg 1990], mapping an (interval, key) pair to a triplet consisting of start_time, end_time, key, as discussed above, allows the valid intervals to be represented by points in three-dimensional space.

通过使用动态多维访问方法，也可以实现良好的平均情况性能。如果只支持多维点，如在k-d-B树 [Robinson 1984] 或h-B树 [Lomet和Salzberg 1990] 中，将（区间，键）对映射为一个由开始时间、结束时间和键组成的三元组（如上文所述），就可以用三维空间中的点来表示有效区间。

If intervals are represented more naturally, as line segments in a two- dimensional key-time space, the cell-tree [Gunther 1989], the R-tree, or one of its variants, the R* [Beckmann et al. 1990], or the R+ [Sellis et al. 1987] could be used. Such solutions should provide good average-case performance, but overlapping still remains a problem, especially if the interval distribution is highly nonuniform (as observed in Kolovson and Stonebraker [1991] for R-trees). If the SR-tree [Kolovson and Stonebraker 1991] is utilized for valid-time databases, overlapping is decreased, but the method may suffer if there are many interval deletions, since all remnants (segments) of a deleted interval have to be found and physically deleted.

如果更自然地将区间表示为二维键 - 时间空间中的线段，那么可以使用单元树（cell-tree） [Gunther 1989]、R树或其变体之一，如R*树 [Beckmann等人1990] 或R+树 [Sellis等人1987]。这样的解决方案应该能提供良好的平均情况性能，但重叠问题仍然存在，特别是当区间分布高度不均匀时（如Kolovson和Stonebraker [1991] 在R树中所观察到的）。如果将SR树 [Kolovson和Stonebraker 1991] 用于有效时间数据库，重叠情况会减少，但如果有许多区间删除操作，该方法可能会出现问题，因为必须找到并物理删除已删除区间的所有残余部分（线段）。

Another possibility is to facilitate a two-level method whose top level indexes the key attribute of the interval objects (using a B+-tree), while the second level indexes the intervals that share the same key attribute. An example of such method is the ST-index [Gu-nadhi and Segev 1993]. In the ST-index there is a separate AP-tree that indexes the start_times of all valid-time intervals sharing a distinct key attribute value. The problem with this approach is that a "*/point/-" query will have to check all stored intervals to see whether they include the query valid-time $v$ .

另一种可能性是采用两级方法，其顶层对区间对象的键属性进行索引（使用B+树），而第二层对共享相同键属性的区间进行索引。这种方法的一个例子是ST索引 [Gu - nadhi和Segev 1993]。在ST索引中，有一个单独的AP树对共享不同键属性值的所有有效时间区间的开始时间进行索引。这种方法的问题在于，“*/点/-”查询必须检查所有存储的区间，以确定它们是否包含查询有效时间 $v$。

The time-index [Elmasri et al. 1990] may also be considered for storing valid-time intervals; there are, however, two drawbacks. First, changes can arrive in any order, so leaf entries anywhere in the index may have to merge or split, thus affecting their relevant timeslices. Second, updating may be problematic as deleting (or adding or modifying the length) of an interval involves updating all the stored timeslices that this interval overlaps.

时间索引（time - index） [Elmasri等人1990] 也可以考虑用于存储有效时间区间；然而，它有两个缺点。首先，更改可以以任何顺序到来，因此索引中任何位置的叶子条目可能都需要合并或拆分，从而影响它们相关的时间片。其次，更新可能会有问题，因为删除（或添加或修改长度）一个区间涉及更新该区间所重叠的所有存储时间片。

Nascimento et al. [1996] offer yet another approach to indexing valid-time databases, the MAP21 structure. A valid-time interval(x,y)is mapped to a point $z = x{10}^{s} + y$ ,where $s$ is the maximum number of digits needed to represent any time point in the valid-time domain. This is enough to map each interval to a separate point. A regular B-tree is then used to index these points. An advantage of this approach is that interval insertions/deletions are easy using the B-tree. However, to answer a valid timeslice query about time $v$ the point closer to $v$ is found in the B-tree and then a sequential search for all intervals before $v$ is performed. At worst, many intervals that do not intersect $v$ can be found (Nascimento et al. [1996] assumes that in practice the maximal interval length is known, which limits how far back the sequential search continues from $v$ ).

Nascimento等人 [1996] 提出了另一种对有效时间数据库进行索引的方法，即MAP21结构。一个有效时间区间(x,y)被映射到一个点 $z = x{10}^{s} + y$，其中 $s$ 是表示有效时间域中任何时间点所需的最大数字位数。这足以将每个区间映射到一个单独的点。然后使用常规的B树对这些点进行索引。这种方法的一个优点是，使用B树进行区间插入/删除很容易。然而，为了回答关于时间 $v$ 的有效时间片查询，需要在B树中找到最接近 $v$ 的点，然后对 $v$ 之前的所有区间进行顺序搜索。在最坏的情况下，可能会找到许多与 $v$ 不相交的区间（Nascimento等人 [1996] 假设在实践中已知最大区间长度，这限制了顺序搜索从 $v$ 回溯的距离）。

Further research is needed in this area. An interesting open problem is whether an I/O optimal solution exists for the "range/point/-" query (valid range timeslices).

该领域还需要进一步的研究。一个有趣的开放性问题是，对于“范围/点/-”查询（有效范围时间片）是否存在I/O最优解。

### 5.3 Bitemporal Methods

### 5.3 双时态方法

As mentioned in Section 4.5, one way to address bitemporal queries is to fully store some of the $C\left( {t}_{i}\right)$ collections of Figure 3, together with the changes between these collections. To exemplify searching through the intervals of a stored $C\left( {t}_{i}\right)$ ,an access method for each stored $C\left( {t}_{i}\right)$ is also included. These $C\left( {t}_{i}\right) \mathrm{s}$ (and their accompanying methods) can then be indexed by a regular B-tree on ${t}_{i}$ ,the transaction time. This is the approach taken in the M-IVTT [Nasci-mento et al. 1996]; the changes between stored methods are called "patches" and each stored $C\left( {t}_{i}\right)$ is indexed by a MAP21 method [Nascimento et al. 1996].

如4.5节所述，处理双时态查询的一种方法是完整存储图3中的一些$C\left( {t}_{i}\right)$集合，以及这些集合之间的变化。为了举例说明如何在存储的$C\left( {t}_{i}\right)$的区间中进行搜索，还为每个存储的$C\left( {t}_{i}\right)$包含了一种访问方法。然后，可以通过一个基于${t}_{i}$（事务时间）的常规B树对这些$C\left( {t}_{i}\right) \mathrm{s}$（及其附带的方法）进行索引。这是M - IVTT方法[纳西门托等人，1996年]所采用的方法；存储方法之间的变化被称为“补丁”，并且每个存储的$C\left( {t}_{i}\right)$都通过MAP21方法进行索引[纳西门托等人，1996年]。

<!-- Media -->

<!-- figureText: $\left( {{t}_{3},{v}_{2}}\right)$ $\left( {{t}_{i},{v}_{j}}\right)$ $0{t}_{1}$ ${t}_{2}$ ${t}_{5}$ $\left( {0,{v}_{j}}\right)$ $\left( {{t}_{i},{v}_{j}}\right)$ $\left( {{t}_{3},{v}_{1}}\right)$ 0 ${t}_{3}$ -->

<img src="https://cdn.noedgeai.com/0195c90d-0a33-750f-877f-c1cec48583bb_57.jpg?x=345&y=247&w=922&h=257&r=0"/>

Figure 26. In the 2-R-tree approach, bitemporal data is divided according to whether their right transaction endpoint is known. The scenario of Figure 3 is presented here (i.e.,after time ${t}_{5}$ has elapsed). The left two-dimensional space is stored in the front $\mathrm{R}$ -tree,while the right in the back $\mathrm{R}$ -tree.

图26。在2 - R树方法中，双时态数据根据其右事务端点是否已知进行划分。这里展示了图3的场景（即，在时间${t}_{5}$过去之后）。左侧的二维空间存储在前$\mathrm{R}$树中，而右侧的存储在后$\mathrm{R}$树中。

<!-- Media -->

The M-IVTT approach can be thought as an extension of the time-index [El-masri et al. 1990] to a bitemporal environment. Depending on how often $C\left( {t}_{i}\right) \mathrm{s}$ are indexed, the space/ update or the query time of the M-IVTT will increase. For example, the space can easily become quadratic if the indexed $C\left( {t}_{i}\right)$ s are every constant number of changes and each change is the addition of a new interval.

可以将M - IVTT方法视为时间索引[埃尔马斯里等人，1990年]在双时态环境下的扩展。根据$C\left( {t}_{i}\right) \mathrm{s}$被索引的频率，M - IVTT的空间/更新或查询时间将会增加。例如，如果每经过固定数量的变化就对$C\left( {t}_{i}\right)$进行索引，并且每次变化都是添加一个新的区间，那么空间很容易变为二次方增长。

In another approach, the intervals associated with a bitemporal object can be "visualized" as a bounding rectangle, which is then stored in a multidimensional index, such as the R-tree [Gutt-man 1984] (or some of its variants, like the SR-tree [Kolovson and Stonebraker 1991]). While this approach has the advantage of using a single index to support both time dimensions, the characteristics of transaction-time create a serious overlapping problem [Kumar et al. 1998]. All bitemporal objects that have not been "deleted" (in the transaction sense) are represented with a transaction-time endpoint extending to now (Figure 4).

在另一种方法中，与双时态对象相关联的区间可以被“可视化”为一个边界矩形，然后将其存储在多维索引中，例如R树[古特曼，1984年]（或者它的一些变体，如SR树[科洛夫森和斯通布雷克，1991年]）。虽然这种方法具有使用单个索引来支持两个时间维度的优点，但事务时间的特性会产生严重的重叠问题[库马尔等人，1998年]。所有尚未“删除”（从事务意义上来说）的双时态对象都用一个延伸到当前时间的事务时间端点来表示（图4）。

To avoid this overlapping, the use of two R-trees (two-R approach) is proposed [Kumar et al. 1998]. When a bitemporal object with valid-time interval $I$ is added in the database at transaction-time $t$ ,it is inserted at the front R-tree. This tree keeps bitemporal objects whose right transaction endpoint is unknown. If a bitemporal object is later "deleted" at some time ${t}^{\prime }\left( {{t}^{\prime } > t}\right)$ , it is physically deleted from the front R-tree and inserted as a rectangle of height $I$ and width from $t$ to ${t}^{\prime }$ ,in the back R-tree. The back R-tree keeps bitemporal objects with known transaction-time intervals (Figure 26, from Kumar et al. [1998]). At any given time, all bitemporal objects stored in the front R-tree share the property that they are "alive" in the transaction-time sense. The temporal information of every such object is thus represented simply by a vertical (valid-time) interval that "cuts" the transaction axis at the transaction-time this object was inserted in the database. Insertions in the front R-tree objects are in increasing transaction time, while physical deletions can happen anywhere on the transaction axis.

为了避免这种重叠，有人提出使用两个R树（双R方法）[库马尔等人，1998年]。当一个有效时间区间为$I$的双时态对象在事务时间$t$被添加到数据库中时，它会被插入到前R树中。这棵树保存右事务端点未知的双时态对象。如果一个双时态对象后来在某个时间${t}^{\prime }\left( {{t}^{\prime } > t}\right)$被“删除”，它会从前R树中物理删除，并作为一个高度为$I$、宽度从$t$到${t}^{\prime }$的矩形插入到后R树中。后R树保存具有已知事务时间区间的双时态对象（图26，来自库马尔等人[1998年]）。在任何给定时间，存储在前R树中的所有双时态对象都具有一个特性，即从事务时间的意义上来说它们是“存活”的。因此，每个这样的对象的时态信息可以简单地用一个垂直（有效时间）区间来表示，该区间在该对象插入数据库的事务时间处“切割”事务轴。前R树对象的插入是按事务时间递增的顺序进行的，而物理删除可以在事务轴的任何位置发生。

A "*/point/point" query about $\left( {{t}_{i},{v}_{i}}\right)$ is then answered with two searches. The back R-tree is searched for all rectangles that contain point $\left( {{t}_{i},{v}_{j}}\right)$ . The front R-tree is searched for all vertical intervals that intersect a horizontal interval $H$ . Interval $H$ starts from the beginning of transaction time and extends until point ${t}_{i}$ is at height ${v}_{i}$ (Figure 26). To support "range/range/range" queries, an additional third dimension for the key ranges is added in both R-trees.

关于$\left( {{t}_{i},{v}_{i}}\right)$的“*/点/点”查询可以通过两次搜索来回答。在后R树中搜索所有包含点$\left( {{t}_{i},{v}_{j}}\right)$的矩形。在前R树中搜索所有与水平区间$H$相交的垂直区间。区间$H$从事务时间的开始处开始，一直延伸到点${t}_{i}$处于高度${v}_{i}$的位置（图26）。为了支持“范围/范围/范围”查询，在两个R树中都为键范围添加了一个额外的第三维。

The usage of two $R$ -trees is reminiscent of the dual-root mixed media R-tree proposed in Kolovson and Stonebraker [1989] as a mixed-media index that stores intervals and also consists of two R-trees. There, new intervals are stored on one R-tree and are gradually moved to the second R-tree. There are, however, the following differences: (a) in the dual-root mixed media R-tree, intervals inserted have both their endpoints known in advance (which is not a characteristic of transaction-time); (b) both R-trees in Kolovson and Stonebraker [1989] store intervals with the same format; (c) the transferring of data in the dual-root mixed media R-tree is performed in a batched way. When the first R-tree reaches a threshold near its maximum allocated size, a vacuuming process completely vacuums all the nodes of the first R-tree (except its root) and inserts them into the second R-tree. In contrast, transferring a bitemporal object in the $2 - \mathrm{R}$ approach is performed whenever this object is deleted in the transaction-time sense. Such a deletion can happen to any currently "alive" object in the front R-tree.

使用两棵 $R$ 树让人想起了科洛夫森（Kolovson）和斯通布雷克（Stonebraker）在1989年提出的双根混合媒体R树，它是一种存储区间的混合媒体索引，同样由两棵R树组成。在那里，新的区间存储在一棵R树上，并逐渐转移到第二棵R树上。然而，存在以下差异：（a）在双根混合媒体R树中，插入的区间的两个端点都是预先已知的（这不是事务时间的特征）；（b）科洛夫森和斯通布雷克在1989年提出的两棵R树以相同的格式存储区间；（c）双根混合媒体R树中的数据转移是以批量方式进行的。当第一棵R树达到其最大分配大小附近的阈值时，清理过程会完全清空第一棵R树的所有节点（除了其根节点），并将它们插入到第二棵R树中。相比之下，在 $2 - \mathrm{R}$ 方法中，只要某个双时态对象在事务时间意义上被删除，就会进行该对象的转移。这种删除可能发生在前部R树中任何当前“存活”的对象上。

Bitemporal problems can also be addressed by the partial persistence approach; this solution emanates from the abstraction of a bitemporal database as a sequence of history-timeslices $C\left( t\right)$ (Figure 3) and has two steps. First, a good ephemeral structure is chosen to represent each $C\left( t\right)$ . This structure must support dynamic addition/deletion of (valid-time) interval-objects. Second, this structure is made partially persistent. The collection of queries supported by the ephemeral structure implies what queries are answered by the bitemporal structure.

双时态问题也可以通过部分持久化方法来解决；该解决方案源于将双时态数据库抽象为一系列历史时间片 $C\left( t\right)$（图3），并分为两个步骤。首先，选择一个合适的临时结构来表示每个 $C\left( t\right)$。这个结构必须支持（有效时间）区间对象的动态添加/删除。其次，使这个结构具有部分持久性。临时结构所支持的查询集合意味着双时态结构能够回答哪些查询。

The main advantage obtained by "viewing" a bitemporal query as a partial persistence problem is that the valid-time requirements are disassociated from those with transaction-time. More specifically, valid time support is provided from the properties of the ephemeral structure, while the transaction time support is achieved by making this structure partially persistent. Conceptually, this methodology provides fast access to the $C\left( t\right)$ of interest,on which the valid-time query is then performed.

将双时态查询“视为”部分持久化问题所获得的主要优势在于，有效时间要求与事务时间要求是分离的。更具体地说，有效时间支持由临时结构的属性提供，而事务时间支持则通过使该结构具有部分持久性来实现。从概念上讲，这种方法能够快速访问感兴趣的 $C\left( t\right)$，然后在其上执行有效时间查询。

The partial persistence methodology is also used in Lanka and Mays [1991]; Becker et al. [1996]; Varman and Verma [1997] for the design of transaction-time access methods. For a transaction-time environment, the ephemeral structure must support dynamic addition/deletion of plain-objects; hence a B-tree is the obvious choice. For a bitemporal environment, two access methods were proposed: the bitemporal interval tree [Kumar et al. 1995], which is created by making an interval tree [Edelsbrunner 1983] partially persistent (and well paginated) and the bitemporal $\mathrm{R}$ -tree $\lbrack \mathrm{{Ku}} -$ mar et al. 1998] created by making an R-tree partially persistent.

部分持久化方法也被兰卡（Lanka）和梅斯（Mays）在1991年；贝克尔（Becker）等人在1996年；瓦尔曼（Varman）和维尔马（Verma）在1997年用于设计事务时间访问方法。对于事务时间环境，临时结构必须支持普通对象的动态添加/删除；因此，B树是显而易见的选择。对于双时态环境，提出了两种访问方法：双时态区间树[库马尔（Kumar）等人，1995年]，它是通过使区间树[埃德尔布鲁纳（Edelsbrunner），1983年]具有部分持久性（并进行良好的分页）而创建的；以及双时态 $\mathrm{R}$ 树 $\lbrack \mathrm{{Ku}} -$ 马尔（mar）等人，1998年]，它是通过使R树具有部分持久性而创建的。

The bitemporal interval tree is designed for the "*/point/point" and “*/range/point” queries. Answering such queries implies that the ephemeral data structure should support point-enclosure and interval-intersection queries. In the absence of an external ephemeral method that optimally solves these problems [Kanellakis 1993; Ramaswamy and Sub-ramanian 1994], a main-memory data structure, the interval tree (which optimally solves the in-core versions of the above problems), was used and made partially persistent and well paginated. One constraint of the bitemporal interval tree is that the universe size $V$ on the valid domain is known in advance. The method computes "*/point/point" and "*/range/ point" queries in $O\left( {{\log }_{B}V + {\log }_{B}n + a}\right)$ I/Os. The space is $O\left( {\left( {n + V}\right) /B}\right)$ ; the update is amortized $O\left( {{\log }_{B}\left( {m + V}\right) }\right)$ I/Os per change. Here $n$ denotes the total number of changes, $a$ is the answer size, and $m$ is the number of intervals contained in the current timeslice $C\left( t\right)$ when the change is performed.

双时态区间树是为“*/点/点”和“*/范围/点”查询而设计的。回答此类查询意味着临时数据结构应支持点包含和区间相交查询。由于缺乏一种能最优解决这些问题的外部临时方法[卡内拉基斯（Kanellakis），1993年；拉马什瓦米（Ramaswamy）和苏布拉马尼亚姆（Sub - ramanian），1994年]，使用了一种内存数据结构——区间树（它能最优解决上述问题的核心版本），并使其具有部分持久性和良好的分页。双时态区间树的一个限制是，有效域上的全域大小 $V$ 是预先已知的。该方法在 $O\left( {{\log }_{B}V + {\log }_{B}n + a}\right)$ 次输入/输出操作中计算“*/点/点”和“*/范围/点”查询。空间复杂度为 $O\left( {\left( {n + V}\right) /B}\right)$；每次更改的更新操作平均为 $O\left( {{\log }_{B}\left( {m + V}\right) }\right)$ 次输入/输出操作。这里 $n$ 表示更改的总数，$a$ 是答案的大小，$m$ 是执行更改时当前时间片 $C\left( t\right)$ 中包含的区间数量。

The bitemporal R-tree does not have the valid-universe constraint. It is a method designed for the more general “range/point/point” and “range/range/ point" bitemporal queries. For that purpose, the ephemeral data structure must support range point-enclosure and range interval-intersection queries on interval-objects. Since neither a main-memory, nor an external data structure exists with good worst-case performance for this problem, the R*-tree [Beckmann et al. 1990] was used, an access method that has good average-case performance for these queries. As a result, the performance of the bitemporal R-tree is bound by the performance of the ephemeral ${\mathrm{R}}^{ * }$ -tree. This is because a method created by the partial-persistence methodology behaves asymptotically as does the original ephemeral structure.

双时态R树没有有效全域约束。它是一种为更通用的“范围/点/点”和“范围/范围/点”双时态查询而设计的方法。为此，临时数据结构必须支持对区间对象进行范围点包含和范围区间相交查询。由于对于这个问题，既没有主存数据结构，也没有外部数据结构能在最坏情况下有良好的性能，因此使用了R*树[贝克曼等人，1990年]，这是一种在这些查询的平均情况下性能良好的访问方法。因此，双时态R树的性能受限于临时${\mathrm{R}}^{ * }$树的性能。这是因为通过部分持久化方法创建的方法在渐近意义上的表现与原始临时结构相同。

Kumar et al. [1998] contains various experiments comparing the average-case performance of the 2-R methodology, the bitemporal R-tree, and the obvious approach that stores bitemporal objects in a single R-tree (the 1-R approach, as in Figure 4). Due to the limited copying introduced by partial persistence, the bitemporal R-tree uses some small extra space (about double the space used by the 1-R and 2-R methods), but it has much better update and query performance. Similarly, the 2-R approach has in general better performance than the 1-R approach.

库马尔等人[1998年]进行了各种实验，比较了2 - R方法、双时态R树和将双时态对象存储在单个R树中的明显方法（即1 - R方法，如图4所示）的平均情况性能。由于部分持久化引入的复制操作有限，双时态R树使用了一些额外的小空间（大约是1 - R和2 - R方法所用空间的两倍），但它的更新和查询性能要好得多。同样，2 - R方法通常比1 - R方法性能更好。

Recently, Bliujute et al. [1998] examine how to handle valid-time now-relative data (data whose end of valid-time is not fixed but tracks the current time) using extensions of R-trees.

最近，布柳尤特等人[1998年]研究了如何使用R树的扩展来处理有效时间相对于当前时间的数据（即有效时间的结束不是固定的，而是跟踪当前时间的数据）。

It remains an interesting open problem to find the theoretically I/O optimal solutions even for the simplest bitemporal problems, like the "*/point/point" and "*/range/point" queries.

即使对于像“*/点/点”和“*/范围/点”查询这样最简单的双时态问题，找到理论上I/O最优的解决方案仍然是一个有趣的开放性问题。

## 6. CONCLUSIONS

## 6. 结论

We presented a comparison of different indexing techniques which have been proposed for efficient access to temporal data. While we have also covered valid-time and bitemporal approaches, the bulk of this paper addresses transaction-time methods because they are in the the majority among the published approaches. Since it is practically impossible to run simulations of all methods under the same input patterns, our comparison was based on the worst-case performance of the examined methods. The items being compared include space requirements, update characteristics, and query performance. Query performance is measured against three basic transaction-time queries: the pure-key, pure-timeslice, and range-timeslice queries, or, using the three-entry notation, the "point/-/*", the "*/-/point," and the "range/-/point" queries, respectively. In addition, we addressed problems like index pagination, data clustering, and the ability of a method to efficiently migrate data to another medium (such as a WORM device). We also introduced a general lower bound for such queries. A method that achieves the lower bound for a particular query is termed I/O-optimal for that query. The worst-case performance of each transaction-time method is summarized in Table II. The reader should be cautious when interpreting worst-case performance. The notation will sometimes penalize a method for its performance on a pathological scenario; we indicate such cases. While Table II provides a good feeling for the asymptotic behavior of the examined methods, the choice of the appropriate method for the particular application also depends on the application characteristics. In addition, issues such as data clustering, index pagination, migration of data to optical disks, etc., may also be more or less important, according to the application. While I/O-optimal (and practical) solutions exist for many transaction-time queries, this is not the case for the valid and bitemporal domain. An I/O-optimal solution exists for the valid-timeslice query, but is mainly of theoretical importance; more work is needed in this area. All examined transaction-time methods support "linear" transaction time. The support of branching transaction time is another promising area of research.

我们对为高效访问时态数据而提出的不同索引技术进行了比较。虽然我们也涵盖了有效时间和双时态方法，但本文的大部分内容讨论的是事务时间方法，因为在已发表的方法中，事务时间方法占大多数。由于实际上不可能在相同的输入模式下对所有方法进行模拟，我们的比较是基于所研究方法的最坏情况性能。比较的项目包括空间需求、更新特性和查询性能。查询性能是根据三种基本的事务时间查询来衡量的：纯键查询、纯时间片查询和范围时间片查询，或者用三项表示法分别表示为“点/-/*”、“*/-/点”和“范围/-/点”查询。此外，我们还讨论了诸如索引分页、数据聚类以及一种方法将数据有效迁移到另一种介质（如一次写入多次读取设备）的能力等问题。我们还为这类查询引入了一个通用的下界。对于特定查询达到下界的方法被称为该查询的I/O最优方法。每个事务时间方法的最坏情况性能总结在表II中。读者在解释最坏情况性能时应谨慎。这种表示法有时会因为一种方法在病态情况下的性能而对其进行惩罚；我们会指出这种情况。虽然表II能很好地反映所研究方法的渐近行为，但为特定应用选择合适的方法还取决于应用的特点。此外，根据应用的不同，诸如数据聚类、索引分页、数据迁移到光盘等问题的重要性也可能有所不同。虽然对于许多事务时间查询存在I/O最优（且实用）的解决方案，但在有效时间和双时态领域并非如此。对于有效时间片查询存在I/O最优解决方案，但主要具有理论意义；该领域还需要更多的研究工作。所有研究的事务时间方法都支持“线性”事务时间。支持分支事务时间是另一个有前景的研究领域。

## ACKNOWLEDGMENTS

## 致谢

The idea of doing this survey was proposed to the authors by R. Snodgrass. We would like to thank the anonymous referees for many insightful comments that improved the presentation of this paper. The performance and the description of the methods are based on our understanding of the related papers, hence any error is entirely our own. The second author would also like to thank J.P. Schmidt for many helpful discussions on lower bounds in a paginated environment.

进行这项调查的想法是由R. 斯诺德格拉斯向作者提出的。我们要感谢匿名评审人员提出的许多有深刻见解的意见，这些意见改进了本文的表述。方法的性能和描述基于我们对相关论文的理解，因此任何错误完全是我们自己的。第二作者还要感谢J.P. 施密特就分页环境中的下界问题进行的许多有益讨论。

## REFERENCES

## 参考文献

Agrawal, R., Faloutsos, C., and Swami, A. 1993. Efficient similarity search in sequence databases. In Proceedings of FODO.

AHN, I. AND SNODGRASS, R. 1988. Partitioned storage for temporal databases. Inf. Syst. 13, 4 (May 1, 1988), 369-391.

ArgE, L. AND VITTER, J. 1996. Optimal dynamic interval management in external memory. In Proceedings of the 37th IEEE Symposium on Foundations of Computer Science (FOCS). IEEE Computer Society Press, Los Alamitos, CA.

Becker, B., Gschwind, S., Ohler, T., Seeger, B., AND WIDMAYER, P. 1996. An asymptotically optimal multiversion B-tree. VLDB J. 5,4, 264-275.

Beckmann, N., Kriegel, H.-P., Schneider, R., and Seeger,B. 1990. The ${\mathrm{R}}^{ * }$ -tree: An efficient and robust access method for points and rectangles. In Proceedings of the 1990 ACM SIG-MOD International Conference on Management of Data (SIGMOD '90, Atlantic City, NJ, May 23-25, 1990), H. Garcia-Molina, Ed. ACM Press, New York, NY, 322-331.

BENTLEY, J. 1977. Algorithms for Klee's rectangle problems. Computer Science Department, Carnegie Mellon University, Pittsburgh, PA.

BEN-ZVI, J. 1982. The time relational model. Ph.D. Dissertation. University of California at Los Angeles, Los Angeles, CA.

Blankenagel, G. and Guting, R. 1990. XP-trees, external priority search trees. Tech. Rep., Fern Universitat Hagen, Informatik-Bericht No.92.

BLANKENAGEL, G. AND GUTING, R. 1994. External segment trees. Algorithmica 12, 6, 498-532.

BLIUJUTE, R., JENSEN, C. S., SALTENIS, S., AND SLIVINSKAS, G. 1998. R-tree based indexing of now-relative bitemporal data. In Proceedings of the Conference on Very Large Data Bases.

BÖHLEN, M. H. 1995. Temporal database system implementations. SIGMOD Rec. 24, 4 (Dec.), 53-60.

Bozkaya, T. and Özsoyoglu, M. 1995. Indexing transaction-time databases. Tech. Rep. CES- 95-19. Case Western Reserve University, Cleveland, OH.

Burton, F., Huntbach, M., and Kollias, J. 1985. Multiple generation text files using

伯顿，F.，亨特巴赫，M.，和科利亚斯，J. 1985年。使用

overlapping tree structures. Comput. J. 28, 414-416.

CHAZELLE, B. 1986. Filtering search: A new approach to query answering. SIAM J. Com-put. 15, 3, 703-724.

Chiang, Y. AND TAMASSIA, R. 1992. Dynamic algorithms in computational geometry. Proc. IEEE 80, 9, 362-381.

Dietzfelbinger, M., Karlin, A., Mehlhorn, K., Meyer, F., Rohntert, H., and Tarjan, R. 1988. Dynamic perfect hashing: Upper and lower bounds. In Proceedings of the 29th IEEE Conference on Foundations of Computer Science. 524-531.

DRISCOLL, J. R., SARNAK, N., SLEATOR, D. D., AND TARJAN, R. E. 1989. Making data structures persistent. J. Comput. Syst. Sci. 38, 1 (Feb. 1989), 86-124.

Dyneson, C., Grandi, F., Käfer, W., Kline, N., Lorentzos, N., Mitsopoulos, Y., Montanari, A., Nonen, D., Peressi, E., Pernici, B., Rob-dick, J. F., Sarda, N. L., Scalas, M. R., Segev, A., Snodgrass, R. T., Soo, M. D., Tansel, A., Tiberio, P., Wiederhold, G., and Jensen, C. S, Eds. 1994. A consensus glossary of temporal database concepts. SIGMOD Rec. 23, 1 (Mar. 1994), 52-64.

EASTON, M. C 1986. Key-sequence data sets on indelible storage. IBM J. Res. Dev. 30, 3 (May 1986), 230-241.

Edelsbrunner, H. 1983. A new approach to rectangle intersections, Parts I&II. Int. J. Comput. Math. 13, 209-229.

ELMASRI, R., KIM, Y., AND WUU, G. 1991. Efficient implementation techniques for the time index. In Proceedings of the Seventh International Conference on Data Engineering (Kobe, Japan). IEEE Computer Society Press, Los Alamitos, CA, 102-111.

ELMASRI, R., WUU, G., AND KIM, Y. 1990. The time index: An access structure for temporal data. In Proceedings of the 16th VLDB Conference on Very Large Data Bases (VLDB, Brisbane, Australia). VLDB Endowment, Berkeley, CA, 1-12.

ELMASRI, R., WUU, G., AND KOURAMAJIAN, V. 1993. The time index and the monotonic $\mathrm{B} +$ - tree. In Temporal Databases: Theory, Design, and Implementation, A. Tansel, J. Clifford, S. Gadia, S. Jajodia, A. Segev, and R. Snodgrass, Eds. Benjamin/Cummings, Redwood City, CA, 433-456.

Faloutsos, C., Ranganathan, M., and Manolo-POULOS, Y. 1994. Fast subsequence matching in time-series databases. In Proceedings of the 1994 ACM SIGMOD International Conference on Management of Data (SIGMOD '94, Minneapolis, MN, May 24-27, 1994), R. T. Snodgrass and M. Winslett, Eds. ACM Press, New York, NY, 419-429.

Gray, J. AND REUTER, A. 1993. Transaction Processing: Concepts and Techniques. Morgan

Kaufmann Publishers Inc., San Francisco, CA.

Gunadhi, H. and Segev, A. 1993. Efficient indexing methods for temporal relation-s. IEEE Trans. Knowl. Data Eng. 5, 3 (June), 496-509.

GUNTHER, O. 1989. The design of the cell-tree: An object-oriented index structure for geometric databases. In Proceedings of the Fifth IEEE International Conference on Data Engineering (Los Angeles, CA, Feb. 1989). 598- 605.

GUTTMAN, A. 1984. R-trees: A dynamic index structure for spatial searching. In Proceedings of the ACM SIGMOD Conference on Management of Data. ACM Press, New York, NY, 47-57.

HELLERSTEIN, J. M., KOUTSOUPIAS, E., AND PAPAD-IMITRIOU, C. H. 1997. On the analysis of indexing schemes. In Proceedings of the 16th ACM SIGACT-SIGMOD-SIGART Symposium on Principles of Database Systems (PODS '97, Tucson, AZ, May 12-14, 1997), A. Mendelzon and Z. M. Özsoyoglu, Eds. ACM Press, New York, NY, 249-256.

ICKING, CH., KLEIN, R., AND OTTMANN, TH. 1988. Priority search trees in secondary memory (extended abstract). In Proceedings of the International Workshop WG '87 Conference on Graph-Theoretic Concepts in Computer Science (Kloster Banz/Staffelstein, Germany, June 29-July 1, 1987), H. Göttler and H.-J. Schneider, Eds. Proceedings of the Second Symposium on Advances in Spatial Databases, vol. LNCS 314. Springer-Verlag, New York, NY, 84-93.

Jagadish, H. V., Mendelzon, A. O., and Milo, T. 1995. Similarity-based queries. In Proceedings of the 14th ACM SIGACT-SIGMOD-SI-GART Symposium on Principles of Database Systems (PODS '95, San Jose, California, May 22-25, 1995), M. Yannakakis, Ed. ACM Press, New York, NY, 36-45.

Jensen, C. S., Mark, L., and Roussopoulos, N. 1991. Incremental implementation model for relational databases with transaction time. IEEE Trans. Knowl. Data Eng. 3, 4, 461-473.

Jensen, C. S., Mark, L., Roussopoulos, N., AND SELLIS, T. 1992. Using differential techniques to efficiently support transaction time- - VLDB J. 2, 1, 75-111.

KAMEL, I. AND FALOUTSOS, C. 1994. Hilbert R-tree: An improved R-tree using fractals. In Proceedings of the 20th International Conference on Very Large Data Bases (VLDB'94, Santiago, Chile, Sept.). VLDB Endowment, Berkeley, CA, 500-509.

Kanellakis, P. C., Ramaswamy, S., Vengroff, D. E., AND VITTER, J. S. 1993. Indexing for data models with constraints and classes (extended abstract). In Proceedings of the Twelfth ACM SIGACT-SIGMOD-SIGART Symposium on Principles of Database Systems (PODS, Washington, DC, May 25-28), C.

Beeri, Ed. ACM Press, New York, NY, 233-

贝里，编辑。美国计算机协会出版社，纽约，纽约州，233 -

243.

Kollios, G. AND Tsotras, V. J. 1998. Hashing methods for temporal data. TimeCenter TR- 24. Aalborg Univ., Aalborg, Denmark. http:www.cs.auc.dk/general/DBS/tdb/ TimeCenter/publications.html

Kolovson, C. 1993. Indexing techniques for historical databases. In Temporal Databases: Theory, Design, and Implementation, A. Tansel, J. Clifford, S. Gadia, S. Jajodia, A. Segev, and R. Snodgrass, Eds. Benjamin/ Cummings, Redwood City, CA, 418-432.

Kolovson, C. and Stonebraker, M. 1989. Indexing techniques for historical databases. In Proceedings of the Fifth IEEE International Conference on Data Engineering (Los Angeles, CA, Feb. 1989). 127-137.

Kolovson, C. and Stonebraker, M. 1991. Segment indexes: Dynamic indexing techniques for multi-dimensional interval data. In Proceedings of the 1991 ACM SIG-MOD International Conference on Management of Data (SIGMOD '91, Denver, CO, May 29-31, 1991), J. Clifford and R. King, Eds. ACM Press, New York, NY, 138- 147.

Kouramajian, V., Elhasri, R., and Chaubury, A. 1994. Declustering techniques for parallelizing temporal access structures. In Proceedings of the 10th IEEE Conference on Data Engineering. 232-242.

Kouramajian, V., Kamel, I., Kouramajian, V., EL-MASRI, R., AND WAHEED, S. 1994. The time index+: an incremental access structure for temporal databases. In Proceedings of the 3rd International Conference on Information and Knowledge Management (CIKM '94, Gaithersburg, Maryland, Nov. 29-Dec. 2, 1994), N. R. Adam, B. K. Bhargava, and Y. Yesha, Eds. ACM Press, New York, NY, 296-303.

Kumar, A., Tsotras, V. J., and Faloutsos, C. 1995. Access methods for bitemporal databases. In Proceedings of the international Workshop on Recent Advances in Temporal Databases (Zurich, Switzerland, Sept.), S. Clifford and A. Tuzhlin, Eds. Springer-Verlag, New York, NY, 235-254.

Kumar, A., Tsotras, V. J., and Faloutsos, C. 1998. Designing access methods for bitemporal databases. IEEE Trans. Knowl. Data Eng. 10, 1 (Jan./Feb.).

Landau, G. M., Schmidt, J. P., and Tsotras, V. J. 1995. On historical queries along multiple lines of time evolution. VLDB J. $4,4,{103} -$ 726.

LANKA, S. AND MAYS, E. 1991. Fully persistent B+trees. In Proceedings of the 1991 ACM SIGMOD International Conference on Management of Data (SIGMOD '91, Denver, CO, May 29-31, 1991), J. Clifford and R. King, Eds. ACM Press, New York, NY, 426-435.

LEUNG, T. Y. C. AND MUNTZ, R. R. 1992. Generalized data stream indexing and temporal query processing. In Proceedings of the Second International Workshop on Research Issues in Data Engineering: Transactions and Query Processing.

LEUNG, T. Y. C. AND MUNTZ, R. R. 1992. Temporal query processing and optimization in multiprocessor database machines. In Proceedings of the 18th International Conference on Very Large Data Bases (Vancouver, B.C., Aug.). VLDB Endowment, Berkeley, CA, 383-394.

LEUNG, T. Y. C. AND MUNTZ, R. R. 1993. Stream processing: Temporal query processing and optimization. In Temporal Databases: Theory, Design, and Implementation, A. Tansel, J. Clifford, S. Gadia, S. Jajodia, A. Segev, and R. Snodgrass, Eds. Benjamin/Cummings, Redwood City, CA, 329-355.

LITWIN, W. 1980. Linear hashing: A new tool for file and table addressing. In Proceedings of the 6th International Conference on Very Large Data Bases (Montreal, Ont. Canada, Oct. 1-3). ACM Press, New York, NY, 212- 223.

LOMET, D. 1993. Using timestamping to optimize commit. In Proceedings of the Second International Conference on Parallel and Distributed Systems (Dec.). 48-55.

LOMET, D. AND SALZBERG, B. 1989. Access methods for multiversion data. In Proceedings of the 1989 ACM SIGMOD International Conference on Management of Data (SIGMOD '89, Portland, OR, June 1989), J. Clifford, J. Clifford, B. Lindsay, D. Maier, and J. Clifford, Eds. ACM Press, New York, NY, 315-324.

LOMET, D. AND SALZBERG, B. 1990. The performance of a multiversion access method. In Proceedings of the 1990 ACM SIGMOD International Conference on Management of Data (SIGMOD '90, Atlantic City, NJ, May 23-25, 1990), H. Garcia-Molina, Ed. ACM Press, New York, NY, 353-363.

LOMET, D. AND SALZBERG, B. 1990. The hB-tree: a multiattribute indexing method with good guaranteed performance. ACM Trans. Database Syst. 15, 4 (Dec. 1990), 625-658.

LOMET, D. AND SALZBERG, B. 1993. Transaction-time databases. In Temporal Databases: Theory, Design, and Implementation, A. Tansel, J. Clifford, S. Gadia, S. Jajodia, A. Segev, and R. Snodgrass, Eds. Benjamin/ Cummings, Redwood City, CA.

LOMET, D. AND SALZBERG, B. 1993. Exploiting a history database for backup. In Proceedings of the 19th International Conference on Very Large Data Bases (VLDB '93, Dublin, Ireland, Aug.). Morgan Kaufmann Publishers Inc., San Francisco, CA, 380-390.

Lorentzos, N. A. and Johnson, R. G. 1988. Extending relational algebra to manipulate temporal data. Inf. Syst. 13, 3 (Oct., 1, 1988), 289-296.

Lum, V., Dabam, P., ERвЕ, R., Guenauer, J., Pis-

卢姆，V.，达巴姆，P.，埃尔夫，R.，格诺尔，J.，皮斯 -

TOR, P., WALCH, G., WERNER, H., AND WOOD-FILL, J. 1984. Designing DBMS support for the temporal database. In Proceedings of the ACM SIGMOD Conference on Management of Data. ACM Press, New York, NY, 115-130.

Manolopoulos, Y. and Kapetanakis, G. 1990. Overlapping B +trees for temporal data. In Proceedings of the Fifth Conference on JCIT (JCIT, Jerusalem, Oct. 22-25). 491-498.

McCreight, E. M. 1985. Priority search trees. SIAM J. Comput. 14, 2, 257-276.

MEHLHORN, K. 1984. Data Structures and Algorithms 3: Multi-dimensional Searching and Computational Geometry. EATCS monographs on theoretical computer science. Springer-Verlag, New York, NY.

Motakis, I. AND ZANIOLO, C. 1997. Temporal aggregation in active database rules. In Proceedings of the International ACM Conference on Management of Data (SIGMOD '97, May). ACM, New York, NY, 440-451.

Muth, P., Kraiss, A., And Weikum, G. 1996. LoT: A dynamic declustering of TSB-tree nodes for parallel access to temporal data. In Proceedings of the Conference on ${EDBT}$ . 553-572.

NASCIMENTO, M., DUNHAM, M. H., AND ELMASRI, R. 1996. M-IVTT: A practical index for bitemporal databases. In Proceedings of the Conference on DEXA (DEX '96, Zurich, Switzerland).

NASCIMENTO, M., DUNHAM, M. H., AND KOURAMA-JIAN, V. 1996. A multiple tree mapping-based approach for range indexing. $J$ . Brazilian Comput. Soc. 2, 3 (Apr.).

NAVATHE, S. B. AND AHMED, R. 1989. A temporal relational model and a query language. Inf. Sci. 49, 1, 2 & 3 (Oct./Nov./Dec. 1989), 147-175.

O'NEIL, P. AND WEIKUM, G. 1993. A log-structured history data access method (LHAM). In Proceedings of the Workshop on High Performance Transaction System (Asilomar, CA).

ÖzSOYOGLU, G. AND SNODGRASS, R. 1995. Temporal and real-time databases: A survey. IEEE Trans. Knowl. Data Eng. 7, 4 (Aug.), 513-532.

RAMASWAMY, S. 1997. Efficient indexing for constraint and temporal databases. In Proceedings of the 6th International Conference on Database Theory (ICDT '97, Delphi, Greece, Jan. 9-10). Springer-Verlag, Berlin, Germany.

Ramaswamy, S. and Subramanian, S. 1994. Path caching (extended abstract): a technique for optimal external searching. In Proceedings of the 13th ACM SIGACT-SIGMOD-SI-GART Symposium on Principles of Database Systems (PODS '94, Minneapolis, MN, May 24-26, 1994), V. Vianu, Ed. ACM Press, New York, NY, 25-35.

Richardson, J., Carey, M., DeWitt, D., and She-KITA, E. 1986. Object and file management

in the Exodus extensible system. In Proceedings of the 12th International Conference on Very Large Data Bases (Kyoto, Japan, Aug.). VLDB Endowment, Berkeley, CA, 91- 100.

RIVEST, R. 1976. Partial-match retrieval algorithms. SIAM J. Comput. 5, 1 (Mar.), 19-50.

Robinson, J. 1984. The K-D-B tree: A search structure for large multidimensional dynamic indexes. In Proceedings of the ACM SIG-MOD Conference on Management of Data. ACM Press, New York, NY, 10-18.

ROTEM, D. AND SEGEV, A. 1987. Physical organization of temporal data. In Proceedings of the Third IEEE International Conference on Data Engineering. IEEE Computer Society Press, Los Alamitos, CA, 547-553.

SALZBERG, B. 1988. File Structures: An Analytic Approach. Prentice-Hall, Inc., Upper Saddle River, NJ.

SALZBERG, B. 1994. Timestamping after commit. In Proceedings of the 3rd International Conference on Parallel and Distributed Information Systems (PDIS, Austin, TX, Sept.). 160-167.

SALZBERG, B. AND LOMET, D. 1995. Branched and Temporal Index Structures. Tech. Rep. NU-CCS-95-17. Northeastern Univ., Boston, MA.

Segev, A. and Gunabhi, H. 1989. Event-join optimization in temporal relational databases. In Proceedings of the 15th International Conference on Very Large Data Bases (VLDB '89, Amsterdam, The Netherlands, Aug 22- 25), R. P. van de Riet, Ed. Morgan Kaufmann Publishers Inc., San Francisco, CA, 205-215.

SELLIS, T., ROUSSOPOULOS, N., AND FALOUTSOS, C. 1987. The R+-tree: A dynamic index for multi-dimensional objects. In Proceedings of the 13th Confererence on Very Large Data Bases (Brighton, England, Sept., 1987). VLDB Endowment, Berkeley, CA.

Seshadri, P., Livny, M., and Rahakrishnan, R. 1996. The design and implementation of a sequence database system. In Proceedings of the 22nd International Conference on Very Large Data Bases (VLDB '96, Mumbai, India, Sept.). 99-110.

Shoshani, A. and Kawagoe, K. 1986. Temporal data management. In Proceedings of the 12th International Conference on Very Large Data Bases (Kyoto, Japan, Aug.). VLDB Endowment, Berkeley, CA, 79-88.

Snodgrags, R. T. AND AHN, I. 1985. A taxonomy of time in databases. In Proceedings of the ACM SIGMOD Conference on Management of Data. ACM Press, New York, NY, 236-246.

Snodgrags, R. T. AND AHN, I. 1986. Temporal databases. IEEE Comput. 19, 9 (Sept. 1986), 35-41 .

STONEBRAKER, M. 1987. The design of the Post-gres storage system. In Proceedings of the 13th Confererence on Very Large Data Bases (Brighton, England, Sept., 1987). VLDB Endowment, Berkeley, CA, 289-300.

Tsotras, V. J. AND Gopinath, B. 1990. Efficient algorithms for managing the history of evolving databases. In Proceedings of the Third International Conference on Database Theory (ICDT '90, Paris, France, Dec.), S. Abiteboul and P. C. Kanellakis, Eds. Proceedings of the Second Symposium on Advances in Spatial Databases, vol. LNCS 470. Springer-Verlag, New York, NY, 141-174.

Tsotras, V. J., Gopinath, B., and Hart, G. W. 1995. Efficient management of time-evolving databases. IEEE Trans. Knowl. Data Eng. 7, 4 (Aug.), 591-608.

Tsotras, V. J., Jensen, C. S., and Snodgrass, R. T. 1998. An extensible notation for spatiotemporal index queries. SIGMOD Rec. 27, 1, ${47} - {53}$ .

TSOTRAS, V. J. AND KANGELARIS, N. 1995. The snapshot index: An I/O-optimal access method for timeslice queries. Inf. Syst. 20, 3 (May 1995), 237-260.

Tsotras, V. J. AND Kumar, A. 1996. Temporal database bibliography update. SIGMOD Rec. 25, 1 (Mar.), 41-51.

Van den Bercken, J., Seeger, B., and Windayer, P. 1997. A generic approach to bulk loading multidimensional index structures. In Proceedings of the 23rd International Conference on Very Large Data Bases (VLDB '97, Athens, Greece, Aug.). 406-415.

VarмАи, P. AND VERMA, R. 1997. An efficient multiversion access structure. IEEE Trans. Knowl. Data Eng. 9, 3 (May/June), 391-409.

VERMA, R. AND VARMAN, P. 1994. Efficient ar-chivable time index: A dynamic indexing scheme for temporal data. In Proceedings of the International Conference on Computer Systems and Education. 59-72.

Vitter, J. S. 1985. An efficient I/O interface for optical disks. ACM Trans. Database Syst. 10, 2 (June 1985), 129-162.