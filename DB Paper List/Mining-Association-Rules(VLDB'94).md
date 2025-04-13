# Fast Algorithms for Mining Association Rules

# 挖掘关联规则的快速算法

Rakesh Agrawal Ramakrishnan Srikant*

拉凯什·阿格拉瓦尔 拉马克里什南·斯里坎特*

IBM Almaden Research Center

IBM阿尔马登研究中心

650 Harry Road, San Jose, CA 95120

加利福尼亚州圣何塞市哈里路650号，邮编95120

## Abstract

## 摘要

We consider the problem of discovering association rules between items in a large database of sales transactions. We present two new algorithms for solving this problem that are fundamentally different from the known algorithms. Empirical evaluation shows that these algorithms outperform the known algorithms by factors ranging from three for small problems to more than an order of magnitude for large problems. We also show how the best features of the two proposed algorithms can be combined into a hybrid algorithm, called AprioriHybrid. Scale-up experiments show that AprioriHybrid scales linearly with the number of transactions. AprioriHybrid also has excellent scale-up properties with respect to the transaction size and the number of items in the database.

我们考虑在大型销售交易数据库中发现项目之间关联规则的问题。我们提出了两种解决该问题的新算法，它们与已知算法有根本的不同。实证评估表明，这些算法在性能上优于已知算法，对于小问题，优势系数为3，对于大问题，优势系数超过一个数量级。我们还展示了如何将这两种算法的最佳特性结合成一种混合算法，称为AprioriHybrid。扩展实验表明，AprioriHybrid的性能与交易数量呈线性关系。AprioriHybrid在交易规模和数据库中的项目数量方面也具有出色的扩展性。

## 1 Introduction

## 1 引言

Progress in bar-code technology has made it possible for retail organizations to collect and store massive amounts of sales data, referred to as the basket data. A record in such data typically consists of the transaction date and the items bought in the transaction. Successful organizations view such databases as important pieces of the marketing infrastructure. They are interested in instituting information-driven marketing processes, managed by database technology, that enable marketers to develop and implement customized marketing programs and strategies [6].

条形码技术的进步使零售企业能够收集和存储大量的销售数据，即购物篮数据。此类数据中的一条记录通常包含交易日期和交易中购买的商品。成功的企业将这些数据库视为营销基础设施的重要组成部分。他们有兴趣建立由数据库技术管理的信息驱动的营销流程，使营销人员能够制定和实施定制的营销计划和策略 [6]。

The problem of mining association rules over basket data was introduced in [4]. An example of such a rule might be that ${98}\%$ of customers that purchase tires and auto accessories also get automotive services done. Finding all such rules is valuable for cross-marketing and attached mailing applications. Other applications include catalog design, add-on sales, store layout, and customer segmentation based on buying patterns. The databases involved in these applications are very large. It is imperative, therefore, to have fast algorithms for this task.

在购物篮数据中挖掘关联规则的问题在文献 [4] 中被提出。这类规则的一个例子可能是，购买轮胎和汽车配件的客户中有 ${98}\%$ 也会进行汽车维修服务。找出所有此类规则对于交叉营销和附带邮件营销应用非常有价值。其他应用包括商品目录设计、附加销售、店铺布局以及基于购买模式的客户细分。这些应用所涉及的数据库非常庞大。因此，必须有快速的算法来完成这项任务。

The following is a formal statement of the problem [4]: Let $\mathcal{I} = \left\{  {{i}_{1},{i}_{2},\ldots ,{i}_{m}}\right\}$ be a set of literals, called items. Let $\mathcal{D}$ be a set of transactions,where each transaction $T$ is a set of items such that $T \subseteq$ $\mathcal{I}$ . Associated with each transaction is a unique identifier, called its TID. We say that a transaction $T$ contains $X$ ,a set of some items in $\mathcal{I}$ ,if $X \subseteq  T$ . An association rule is an implication of the form $X \Rightarrow  Y$ ,where $X \subset  \mathcal{I},Y \subset  \mathcal{I}$ ,and $X \cap  Y = \varnothing$ . The rule $X \Rightarrow  Y$ holds in the transaction set $\mathcal{D}$ with confidence $c$ if $c\%$ of transactions in $\mathcal{D}$ that contain $X$ also contain $Y$ . The rule $X \Rightarrow  Y$ has support $s$ in the transaction set $\mathcal{D}$ if $s\%$ of transactions in $\mathcal{D}$ contain $X \cup  Y$ . Our rules are somewhat more general than in [4] in that we allow a consequent to have more than one item.

以下是该问题的正式表述 [4]：设 $\mathcal{I} = \left\{  {{i}_{1},{i}_{2},\ldots ,{i}_{m}}\right\}$ 是一组文字，称为项目。设 $\mathcal{D}$ 是一组交易，其中每个交易 $T$ 是一组项目，满足 $T \subseteq$ $\mathcal{I}$ 。每个交易都有一个唯一的标识符，称为交易ID（TID）。如果 $X \subseteq  T$ ，我们称交易 $T$ 包含 $X$ （$\mathcal{I}$ 中某些项目的集合）。关联规则是形如 $X \Rightarrow  Y$ 的蕴含式，其中 $X \subset  \mathcal{I},Y \subset  \mathcal{I}$ 且 $X \cap  Y = \varnothing$ 。如果在 $\mathcal{D}$ 中包含 $X$ 的交易中有 $c\%$ 也包含 $Y$ ，则规则 $X \Rightarrow  Y$ 在交易集 $\mathcal{D}$ 中具有置信度 $c$ 。如果在 $\mathcal{D}$ 中有 $s\%$ 的交易包含 $X \cup  Y$ ，则规则 $X \Rightarrow  Y$ 在交易集 $\mathcal{D}$ 中具有支持度 $s$ 。我们的规则比文献 [4] 中的规则更通用，因为我们允许规则的结果部分包含多个项目。

Given a set of transactions $\mathcal{D}$ ,the problem of mining association rules is to generate all association rules that have support and confidence greater than the user-specified minimum support (called minsup) and minimum confidence (called minconf) respectively. Our discussion is neutral with respect to the representation of $\mathcal{D}$ . For example, $\mathcal{D}$ could be a data file, a relational table, or the result of a relational expression.

给定一组交易 $\mathcal{D}$ ，挖掘关联规则的问题是生成所有支持度和置信度分别大于用户指定的最小支持度（称为minsup）和最小置信度（称为minconf）的关联规则。我们的讨论与 $\mathcal{D}$ 的表示形式无关。例如，$\mathcal{D}$ 可以是一个数据文件、一个关系表或一个关系表达式的结果。

An algorithm for finding all association rules, henceforth referred to as the AIS algorithm, was presented in [4]. Another algorithm for this task, called the SETM algorithm, has been proposed in [13]. In this paper, we present two new algorithms, Apriori and AprioriTid, that differ fundamentally from these algorithms. We present experimental results showing that the proposed algorithms always outperform the earlier algorithms. The performance gap is shown to increase with problem size, and ranges from a factor of three for small problems to more than an order of magnitude for large problems. We then discuss how the best features of Apriori and Apriori-Tid can be combined into a hybrid algorithm, called AprioriHybrid. Experiments show that the Apriori-Hybrid has excellent scale-up properties, opening up the feasibility of mining association rules over very large databases.

文献 [4] 中提出了一种用于找出所有关联规则的算法，此后称为AIS算法。文献 [13] 中提出了另一种用于此任务的算法，称为SETM算法。在本文中，我们提出了两种新算法，Apriori和AprioriTid，它们与这些算法有根本的不同。我们给出的实验结果表明，所提出的算法始终优于早期的算法。性能差距随着问题规模的增大而增大，对于小问题，优势系数为3，对于大问题，优势系数超过一个数量级。然后我们讨论如何将Apriori和Apriori - Tid的最佳特性结合成一种混合算法，称为AprioriHybrid。实验表明，AprioriHybrid具有出色的扩展性，这使得在非常大的数据库中挖掘关联规则成为可能。

---

<!-- Footnote -->

*Visiting from the Department of Computer Science, University of Wisconsin, Madison.

*来自威斯康星大学麦迪逊分校计算机科学系的访问学者。

Permission to copy without fee all or part of this material is granted provided that the copies are not made or distributed for direct commercial advantage, the VLDB copyright notice and the title of the publication and its date appear, and notice is given that copying is by permission of the Very Large Data Base Endowment. To copy otherwise, or to republish, requires a fee and/or special permission from the Endowment. Proceedings of the 20th VLDB Conference Santiago, Chile, 1994

允许免费复制本材料的全部或部分内容，但前提是复制的目的不是为了直接的商业利益，需保留VLDB版权声明、出版物标题及其日期，并注明复制获得了超大型数据库基金会的许可。若要以其他方式复制或重新发布，则需要向该基金会支付费用和/或获得特别许可。《第20届VLDB会议论文集》，智利圣地亚哥，1994年

<!-- Footnote -->

---

The problem of finding association rules falls within the purview of database mining [3] [12], also called knowledge discovery in databases [21]. Related, but not directly applicable, work includes the induction of classification rules [8] [11] [22], discovery of causal rules [19], learning of logical definitions [18], fitting of functions to data [15], and clustering [9] [10]. The closest work in the machine learning literature is the KID3 algorithm presented in [20]. If used for finding all association rules, this algorithm will make as many passes over the data as the number of combinations of items in the antecedent, which is exponentially large. Related work in the database literature is the work on inferring functional dependencies from data [16]. Functional dependencies are rules requiring strict satisfaction. Consequently, having determined a dependency $X \rightarrow  A$ ,the algorithms in [16] consider any other dependency of the form $X + Y \rightarrow  A$ redundant and do not generate it. The association rules we consider are probabilistic in nature. The presence of a rule $X \rightarrow  A$ does not necessarily mean that $X + Y \rightarrow  A$ also holds because the latter may not have minimum support. Similarly, the presence of rules $X \rightarrow  Y$ and $Y \rightarrow  Z$ does not necessarily mean that $X \rightarrow  Z$ holds because the latter may not have minimum confidence.

发现关联规则的问题属于数据库挖掘的范畴[3][12]，也被称为数据库中的知识发现[21]。相关但并非直接适用的工作包括分类规则的归纳[8][11][22]、因果规则的发现[19]、逻辑定义的学习[18]、函数对数据的拟合[15]以及聚类[9][10]。机器学习文献中最接近的工作是文献[20]中提出的KID3算法。如果用该算法来发现所有关联规则，它对数据的遍历次数将与前件中项的组合数相同，而这个组合数是呈指数级增长的。数据库文献中的相关工作是从数据中推断函数依赖的研究[16]。函数依赖是要求严格满足的规则。因此，在确定了一个依赖关系$X \rightarrow  A$后，文献[16]中的算法会认为任何形式为$X + Y \rightarrow  A$的其他依赖关系是冗余的，不会生成它。我们所考虑的关联规则本质上是概率性的。规则$X \rightarrow  A$的存在并不一定意味着$X + Y \rightarrow  A$也成立，因为后者可能不满足最小支持度。同样，规则$X \rightarrow  Y$和$Y \rightarrow  Z$的存在并不一定意味着$X \rightarrow  Z$成立，因为后者可能不满足最小置信度。

There has been work on quantifying the "usefulness" or "interestingness" of a rule [20]. What is useful or interesting is often application-dependent. The need for a human in the loop and providing tools to allow human guidance of the rule discovery process has been articulated, for example, in [7] [14]. We do not discuss these issues in this paper, except to point out that these are necessary features of a rule discovery system that may use our algorithms as the engine of the discovery process.

已有关于量化规则“有用性”或“趣味性”的研究[20]。什么是有用的或有趣的往往取决于具体应用。例如，文献[7][14]中明确指出了在规则发现过程中需要人工参与并提供工具以实现人工引导的必要性。在本文中，我们不讨论这些问题，只是指出这些是规则发现系统的必要特性，该系统可能会使用我们的算法作为发现过程的引擎。

### 1.1 Problem Decomposition and Paper Organization

### 1.1 问题分解与论文组织

The problem of discovering all association rules can be decomposed into two subproblems [4]:

发现所有关联规则的问题可以分解为两个子问题[4]：

1. Find all sets of items (itemsets) that have transaction support above minimum support. The support for an itemset is the number of transactions that contain the itemset. Itemsets with minimum support are called large itemsets, and all others small itemsets. In Section 2, we give new algorithms, Apriori and AprioriTid, for solving this problem.

1. 找出所有事务支持度高于最小支持度的项集（项集）。项集的支持度是包含该项集的事务数量。具有最小支持度的项集称为大项集，其他的则称为小项集。在第2节中，我们提出了新的算法，即Apriori和AprioriTid，用于解决这个问题。

2. Use the large itemsets to generate the desired rules. Here is a straightforward algorithm for this task. For every large itemset $l$ ,find all non-empty subsets of $l$ . For every such subset $a$ ,output a rule of the form $a \Rightarrow  \left( {l - a}\right)$ if the ratio of support(l) to support(a)is at least minconf. We need to consider all subsets of $l$ to generate rules with multiple consequents. Due to lack of space, we do not discuss this subproblem further, but refer the reader to [5] for a fast algorithm.

2. 使用大项集生成所需的规则。以下是完成此任务的一个简单算法。对于每个大项集$l$，找出$l$的所有非空子集。对于每个这样的子集$a$，如果支持度(l)与支持度(a)的比值至少为最小置信度，则输出形式为$a \Rightarrow  \left( {l - a}\right)$的规则。我们需要考虑$l$的所有子集，以生成具有多个后件的规则。由于篇幅限制，我们不再进一步讨论这个子问题，但建议读者参考文献[5]以获取一个快速算法。

In Section 3, we show the relative performance of the proposed Apriori and AprioriTid algorithms against the AIS [4] and SETM [13] algorithms. To make the paper self-contained, we include an overview of the AIS and SETM algorithms in this section. We also describe how the Apriori and AprioriTid algorithms can be combined into a hybrid algorithm, AprioriHybrid, and demonstrate the scaleup properties of this algorithm. We conclude by pointing out some related open problems in Section 4.

在第3节中，我们展示了所提出的Apriori和AprioriTid算法相对于AIS[4]和SETM[13]算法的相对性能。为了使本文内容完整，我们在本节中对AIS和SETM算法进行了概述。我们还描述了如何将Apriori和AprioriTid算法组合成一个混合算法AprioriHybrid，并展示了该算法的可扩展性。最后，我们在第4节中指出了一些相关的开放性问题。

## 2 Discovering Large Itemsets

## 2 发现大项集

Algorithms for discovering large itemsets make multiple passes over the data. In the first pass, we count the support of individual items and determine which of them are large, i.e. have minimum support. In each subsequent pass, we start with a seed set of itemsets found to be large in the previous pass. We use this seed set for generating new potentially large itemsets, called candidate itemsets, and count the actual support for these candidate itemsets during the pass over the data. At the end of the pass, we determine which of the candidate itemsets are actually large, and they become the seed for the next pass. This process continues until no new large itemsets are found.

发现大型项集的算法会对数据进行多次扫描。在第一次扫描中，我们统计单个项的支持度，并确定哪些项是大型项，即具有最小支持度的项。在随后的每次扫描中，我们以在上一次扫描中被发现为大型项集的种子集开始。我们使用这个种子集来生成新的潜在大型项集，称为候选项集，并在对数据进行扫描期间统计这些候选项集的实际支持度。在扫描结束时，我们确定哪些候选项集实际上是大型项集，它们将成为下一次扫描的种子。这个过程会一直持续，直到没有新的大型项集被发现为止。

The Apriori and AprioriTid algorithms we propose differ fundamentally from the AIS [4] and SETM [13] algorithms in terms of which candidate itemsets are counted in a pass and in the way that those candidates are generated. In both the AIS and SETM algorithms, candidate itemsets are generated on-the-fly during the pass as data is being read. Specifically, after reading a transaction, it is determined which of the itemsets found large in the previous pass are present in the transaction. New candidate itemsets are generated by extending these large itemsets with other items in the transaction. However, as we will see, the disadvantage is that this results in unnecessarily generating and counting too many candidate itemsets that turn out to be small.

我们提出的Apriori和AprioriTid算法与AIS [4]和SETM [13]算法在一次扫描中统计哪些候选项集以及生成这些候选项集的方式上存在根本差异。在AIS和SETM算法中，候选项集是在读取数据的扫描过程中动态生成的。具体来说，在读取一个事务后，会确定在上一次扫描中被发现为大型项集的哪些项集存在于该事务中。通过用事务中的其他项扩展这些大型项集来生成新的候选项集。然而，正如我们将看到的，缺点是这会导致不必要地生成和统计太多最终被证明是小型的候选项集。

The Apriori and AprioriTid algorithms generate the candidate itemsets to be counted in a pass by using only the itemsets found large in the previous pass - without considering the transactions in the database. The basic intuition is that any subset of a large itemset must be large. Therefore, the candidate itemsets having $k$ items can be generated by joining large itemsets having $k - 1$ items,and deleting those that contain any subset that is not large. This procedure results in generation of a much smaller number of candidate itemsets.

Apriori和AprioriTid算法仅使用在上一次扫描中被发现为大型的项集来生成在一次扫描中要统计的候选项集，而不考虑数据库中的事务。基本的直觉是，大型项集的任何子集都必须是大型的。因此，具有$k$个项的候选项集可以通过连接具有$k - 1$个项的大型项集，并删除那些包含任何非大型子集的项集来生成。这个过程会生成数量少得多的候选项集。

The AprioriTid algorithm has the additional property that the database is not used at all for counting the support of candidate itemsets after the first pass. Rather, an encoding of the candidate itemsets used in the previous pass is employed for this purpose. In later passes, the size of this encoding can become much smaller than the database, thus saving much reading effort. We will explain these points in more detail when we describe the algorithms.

AprioriTid算法还有一个额外的特性，即在第一次扫描之后，根本不使用数据库来统计候选项集的支持度。相反，为此目的使用上一次扫描中使用的候选项集的编码。在后续的扫描中，这种编码的大小可能会比数据库小得多，从而节省大量的读取工作。在描述这些算法时，我们将更详细地解释这些要点。

Notation We assume that items in each transaction are kept sorted in their lexicographic order. It is straightforward to adapt these algorithms to the case where the database $\mathcal{D}$ is kept normalized and each database record is a <TID, item> pair, where TID is the identifier of the corresponding transaction.

符号表示 我们假设每个事务中的项按字典序排序。将这些算法应用于数据库$\mathcal{D}$保持规范化且每个数据库记录是一个<TID, 项>对的情况是很直接的，其中TID是相应事务的标识符。

We call the number of items in an itemset its size, and call an itemset of size $k$ a $k$ -itemset. Items within an itemset are kept in lexicographic order. We use the notation $c\left\lbrack  1\right\rbrack   \cdot  c\left\lbrack  2\right\rbrack   \cdot  \ldots  \cdot  c\left\lbrack  k\right\rbrack$ to represent a $k$ - itemset $c$ consisting of items $c\left\lbrack  1\right\rbrack  ,c\left\lbrack  2\right\rbrack  ,\ldots c\left\lbrack  k\right\rbrack$ ,where $c\left\lbrack  1\right\rbrack   < c\left\lbrack  2\right\rbrack   < \ldots  < c\left\lbrack  k\right\rbrack$ . If $c = X \cdot  Y$ and $Y$ is an $m$ -itemset,we also call $Y$ an $m$ -extension of $X$ . Associated with each itemset is a count field to store the support for this itemset. The count field is initialized to zero when the itemset is first created.

我们将项集中的项的数量称为其大小，并将大小为$k$的项集称为$k$ -项集。项集内的项按字典序排列。我们使用符号$c\left\lbrack  1\right\rbrack   \cdot  c\left\lbrack  2\right\rbrack   \cdot  \ldots  \cdot  c\left\lbrack  k\right\rbrack$来表示由项$c\left\lbrack  1\right\rbrack  ,c\left\lbrack  2\right\rbrack  ,\ldots c\left\lbrack  k\right\rbrack$组成的$k$ -项集$c$，其中$c\left\lbrack  1\right\rbrack   < c\left\lbrack  2\right\rbrack   < \ldots  < c\left\lbrack  k\right\rbrack$。如果$c = X \cdot  Y$且$Y$是一个$m$ -项集，我们也将$Y$称为$X$的$m$ -扩展。与每个项集相关联的是一个计数字段，用于存储该项集的支持度。当项集首次创建时，计数字段初始化为零。

We summarize in Table 1 the notation used in the algorithms. The set ${\bar{C}}_{k}$ is used by AprioriTid and will be further discussed when we describe this algorithm.

我们在表1中总结了算法中使用的符号。集合${\bar{C}}_{k}$由AprioriTid使用，在描述该算法时将进一步讨论。

### 2.1 Algorithm Apriori

### 2.1 Apriori算法

Figure 1 gives the Apriori algorithm. The first pass of the algorithm simply counts item occurrences to determine the large 1-itemsets. A subsequent pass, say pass $k$ ,consists of two phases. First,the large itemsets ${L}_{k - 1}$ found in the(k - 1)th pass are used to generate the candidate itemsets ${C}_{k}$ ,using the apriori-gen function described in Section 2.1.1. Next, the database is scanned and the support of candidates in ${C}_{k}$ is counted. For fast counting,we need to efficiently determine the candidates in ${C}_{k}$ that are contained in a given transaction $t$ . Section 2.1.2 describes the subset function used for this purpose. See [5] for a discussion of buffer management.

图1展示了Apriori算法（先验算法）。该算法的第一轮只是对项的出现次数进行计数，以确定频繁1-项集。后续的某一轮，比如第$k$轮，由两个阶段组成。首先，使用第2.1.1节中描述的apriori - gen函数，利用在第(k - 1)轮中找到的频繁项集${L}_{k - 1}$来生成候选项集${C}_{k}$。接下来，扫描数据库并统计候选项集${C}_{k}$中各项的支持度。为了实现快速计数，我们需要高效地确定给定事务$t$中包含的候选项集${C}_{k}$中的项。第2.1.2节描述了用于此目的的子集函数。有关缓冲区管理的讨论请参阅文献[5]。

<!-- Media -->

Table 1: Notation

表1：符号说明

<table><tr><td>$k$ -itemset</td><td>An itemset having $k$ items.</td></tr><tr><td>${L}_{k}$</td><td>Set of large $k$ -itemsets (those with minimum support). Each member of this set has two fields: i) itemset and ii) support count.</td></tr><tr><td>${C}_{k}$</td><td>Set of candidate $k$ -itemsets (potentially large itemsets). Each member of this set has two fields: i) itemset and ii) support count.</td></tr><tr><td>${C}_{k}$</td><td>Set of candidate $k$ -itemsets when the TIDs of the generating transactions are kept associated with the candidates.</td></tr></table>

<table><tbody><tr><td>$k$项集</td><td>包含$k$个项的项集。</td></tr><tr><td>${L}_{k}$</td><td>大型$k$项集（具有最小支持度的项集）的集合。该集合的每个成员有两个字段：i) 项集；ii) 支持度计数。</td></tr><tr><td>${C}_{k}$</td><td>候选$k$项集（潜在的大型项集）的集合。该集合的每个成员有两个字段：i) 项集；ii) 支持度计数。</td></tr><tr><td>${C}_{k}$</td><td>当生成事务的事务标识符（TIDs）与候选集保持关联时的候选$k$项集的集合。</td></tr></tbody></table>

${L}_{1} = \{$ large 1-itemsets $\}$ ;

${L}_{1} = \{$ 大型 1-项集 $\}$ ;

for $\left( {k = 2;{L}_{k - 1} \neq  \varnothing ;k +  + }\right)$ do begin

对于 $\left( {k = 2;{L}_{k - 1} \neq  \varnothing ;k +  + }\right)$ 执行开始

${C}_{k} = \operatorname{apriori-gen}\left( {L}_{k - 1}\right) ;//$ New candidates

${C}_{k} = \operatorname{apriori-gen}\left( {L}_{k - 1}\right) ;//$ 新候选集

forall transactions $t \in  \mathcal{D}$ do begin

对于所有事务 $t \in  \mathcal{D}$ 执行开始

${C}_{t} = \operatorname{subset}\left( {{C}_{k},t}\right) ;//$ Candidates contained in $t$

${C}_{t} = \operatorname{subset}\left( {{C}_{k},t}\right) ;//$ 包含在 $t$ 中的候选集

forall candidates $c \in  {C}_{t}$ do

对于所有候选集 $c \in  {C}_{t}$ 执行

c.count++;

c.计数加 1;

end

结束

${L}_{k} = \left\{  {c \in  {C}_{k} \mid  c\text{.count} \geq  \text{minsup}}\right\}$

and

并且

) Answer $= \mathop{\bigcup }\limits_{k}{L}_{k}$ ;

) 答案 $= \mathop{\bigcup }\limits_{k}{L}_{k}$ ;

Figure 1: Algorithm Apriori

图 1：Apriori 算法

<!-- Media -->

#### 2.1.1 Apriori Candidate Generation

#### 2.1.1 Apriori 候选集生成

The apriori-gen function takes as argument ${L}_{k - 1}$ , the set of all large(k - 1)-itemsets. It returns a superset of the set of all large $k$ -itemsets. The function works as follows. ${}^{1}$ First,in the join step, we join ${L}_{k - 1}$ with ${L}_{k - 1}$ :

apriori - gen 函数以 ${L}_{k - 1}$（所有大型(k - 1)-项集的集合）作为参数。它返回所有大型 $k$ -项集集合的超集。该函数的工作方式如下。 ${}^{1}$ 首先，在连接步骤中，我们将 ${L}_{k - 1}$ 与 ${L}_{k - 1}$ 进行连接：

insert into ${C}_{k}$

插入到 ${C}_{k}$ 中

select $p$ .item ${}_{1},p$ .item ${}_{2},\ldots ,p$ .item ${}_{k - 1},q$ .item ${}_{k - 1}$

选择 $p$ .项 ${}_{1},p$ .项 ${}_{2},\ldots ,p$ .项 ${}_{k - 1},q$ .项 ${}_{k - 1}$

from ${L}_{k - 1}p,{L}_{k - 1}q$

从 ${L}_{k - 1}p,{L}_{k - 1}q$ 中

where $p$ .it ${\mathrm{{em}}}_{1} = q$ .it ${\mathrm{{em}}}_{1},\ldots ,p$ .it ${\mathrm{{em}}}_{k - 2} = q$ .it ${\mathrm{{em}}}_{k - 2}$ ,

其中 $p$ 。它 ${\mathrm{{em}}}_{1} = q$ 。它 ${\mathrm{{em}}}_{1},\ldots ,p$ 。它 ${\mathrm{{em}}}_{k - 2} = q$ 。它 ${\mathrm{{em}}}_{k - 2}$ ，

$p$ . item $k - 1 < q$ . item $k - 1$ ;

$p$ 。项 $k - 1 < q$ 。项 $k - 1$ ;

Next,in the prune step,we delete all itemsets $c \in  {C}_{k}$ such that some(k - 1)-subset of $c$ is not in ${L}_{k - 1}$ :

接下来，在剪枝步骤中，我们删除所有项集 $c \in  {C}_{k}$ ，使得 $c$ 的某个(k - 1) - 子集不在 ${L}_{k - 1}$ 中：

---

<!-- Footnote -->

${}^{1}$ Concurrent to our work,the following two-step candidate generation procedure has been proposed in [17]:

${}^{1}$ 在我们开展工作的同时，文献[17]中提出了以下两步候选生成过程：

$$
{C}_{k}^{\prime } = \left\{  {X \cup  {X}^{\prime } \mid  X,{X}^{\prime } \in  {L}_{k - 1},\left| {X \cap  {X}^{\prime }}\right|  = k - 2}\right\}  
$$

$$
{C}_{k} = \left\{  {X \in  {C}_{k}^{\prime } \mid  X\text{ contains }k\text{ members of }{L}_{k - 1}}\right\}  
$$

These two steps are similar to our join and prune steps respectively. However, in general, step 1 would produce a superset of the candidates produced by our join step.

这两个步骤分别与我们的连接和剪枝步骤类似。然而，一般来说，步骤1会生成我们连接步骤所生成候选集的超集。

<!-- Footnote -->

---

forall itemsets $c \in  {C}_{k}$ do

对所有项集 $c \in  {C}_{k}$ 执行

forall(k - 1)-subsets $s$ of $c$ do

对 $c$ 的所有(k - 1) - 子集 $s$ 执行

if $\left( {s \notin  {L}_{k - 1}}\right)$ then

如果 $\left( {s \notin  {L}_{k - 1}}\right)$ 则

delete $c$ from ${C}_{k}$ ;

从 ${C}_{k}$ 中删除 $c$ ;

Example Let ${L}_{3}$ be $\{ \{ {123}\} ,\{ {124}\} ,\{ {134}\} ,\{ 1$ ${35}\} ,\{ {234}\} \}$ . After the join step, ${C}_{4}$ will be $\{ \{ {123}$ $4\} ,\{ {1345}\} \}$ . The prune step will delete the itemset $\{ {1345}\}$ because the itemset $\{ {145}\}$ is not in ${L}_{3}$ . We will then be left with only $\{ {1234}\}$ in ${C}_{4}$ .

示例 设 ${L}_{3}$ 为 $\{ \{ {123}\} ,\{ {124}\} ,\{ {134}\} ,\{ 1$ ${35}\} ,\{ {234}\} \}$ 。在连接步骤之后， ${C}_{4}$ 将为 $\{ \{ {123}$ $4\} ,\{ {1345}\} \}$ 。剪枝步骤将删除项集 $\{ {1345}\}$ ，因为项集 $\{ {145}\}$ 不在 ${L}_{3}$ 中。然后， ${C}_{4}$ 中将仅剩下 $\{ {1234}\}$ 。

Contrast this candidate generation with the one used in the AIS and SETM algorithms. In pass $k$ of these algorithms,a database transaction $t$ is read and it is determined which of the large itemsets in ${L}_{k - 1}$ are present in $t$ . Each of these large itemsets $l$ is then extended with all those large items that are present in $t$ and occur later in the lexicographic ordering than any of the items in $l$ . Continuing with the previous example, consider a transaction \{12 345\}. In the fourth pass, AIS and SETM will generate two candidates, $\{ {1234}\}$ and $\{ {1235}\}$ , by extending the large itemset $\{ \begin{array}{lll} 1 & 2 & 3 \end{array}\}$ . Similarly,an additional three candidate itemsets will be generated by extending the other large itemsets in ${L}_{3}$ ,leading to a total of 5 candidates for consideration in the fourth pass. Apriori, on the other hand, generates and counts only one itemset, $\{ {1345}\}$ ,because it concludes a priori that the other combinations cannot possibly have minimum support.

将这种候选生成方法与AIS和SETM算法中使用的方法进行对比。在这些算法的第 $k$ 遍扫描中，读取一个数据库事务 $t$ ，并确定 ${L}_{k - 1}$ 中的哪些大项集存在于 $t$ 中。然后，将这些大项集 $l$ 中的每一个都用 $t$ 中存在的、且在字典序中比 $l$ 中的任何项都靠后的所有大项进行扩展。继续前面的示例，考虑一个事务{12 345}。在第四遍扫描中，AIS和SETM将通过扩展大项集 $\{ \begin{array}{lll} 1 & 2 & 3 \end{array}\}$ 生成两个候选集 $\{ {1234}\}$ 和 $\{ {1235}\}$ 。类似地，通过扩展 ${L}_{3}$ 中的其他大项集将额外生成三个候选项集，从而在第四遍扫描中总共产生5个候选集以供考虑。另一方面，Apriori算法仅生成并计数一个项集 $\{ {1345}\}$ ，因为它先验地得出结论，其他组合不可能具有最小支持度。

Correctness We need to show that ${C}_{k} \supseteq  {L}_{k}$ . Clearly, any subset of a large itemset must also have minimum support. Hence, if we extended each itemset in ${L}_{k - 1}$ with all possible items and then deleted all those whose(k - 1)-subsets were not in ${L}_{k - 1}$ ,we would be left with a superset of the itemsets in ${L}_{k}$ .

正确性 我们需要证明 ${C}_{k} \supseteq  {L}_{k}$ 。显然，大项集的任何子集也必须具有最小支持度。因此，如果我们用所有可能的项扩展 ${L}_{k - 1}$ 中的每个项集，然后删除所有其(k - 1) - 子集不在 ${L}_{k - 1}$ 中的项集，我们将得到 ${L}_{k}$ 中项集的一个超集。

The join is equivalent to extending ${L}_{k - 1}$ with each item in the database and then deleting those itemsets for which the(k - 1)-itemset obtained by deleting the (k - 1)th item is not in ${L}_{k - 1}$ . The condition $p$ .item $< q$ .item $k - 1$ simply ensures that no duplicates are generated. Thus,after the join step, ${C}_{k} \supseteq  {L}_{k}$ . By similar reasoning, the prune step, where we delete from ${C}_{k}$ all itemsets whose(k - 1)-subsets are not in ${L}_{k - 1}$ ,also does not delete any itemset that could be in ${L}_{k}$ .

连接操作等同于用数据库中的每个项扩展${L}_{k - 1}$，然后删除那些通过删除第(k - 1)项得到的(k - 1)项集不在${L}_{k - 1}$中的项集。条件$p$.项 $< q$.项 $k - 1$只是确保不会生成重复项。因此，在连接步骤之后，${C}_{k} \supseteq  {L}_{k}$。通过类似的推理，剪枝步骤（即从${C}_{k}$中删除所有(k - 1)子集不在${L}_{k - 1}$中的项集）也不会删除任何可能在${L}_{k}$中的项集。

Variation: Counting Candidates of Multiple Sizes in One Pass Rather than counting only candidates of size $k$ in the $k$ th pass,we can also count the candidates ${C}_{k + 1}^{\prime }$ ,where ${C}_{k + 1}^{\prime }$ is generated from ${C}_{k}$ ,etc. Note that ${C}_{k + 1}^{\prime } \supseteq  {C}_{k + 1}$ since ${C}_{k + 1}$ is generated from ${L}_{k}$ . This variation can pay off in the later passes when the cost of counting and keeping in memory additional ${C}_{k + 1}^{\prime } - {C}_{k + 1}$ candidates becomes less than the cost of scanning the database.

变体：一次统计多种大小的候选项集 我们不必在第$k$遍扫描时仅统计大小为$k$的候选项集，还可以统计候选项集${C}_{k + 1}^{\prime }$，其中${C}_{k + 1}^{\prime }$是从${C}_{k}$生成的，等等。注意，${C}_{k + 1}^{\prime } \supseteq  {C}_{k + 1}$，因为${C}_{k + 1}$是从${L}_{k}$生成的。当统计并在内存中保存额外的${C}_{k + 1}^{\prime } - {C}_{k + 1}$候选项集的成本低于扫描数据库的成本时，这种变体在后续扫描中会带来收益。

#### 2.1.2 Subset Function

#### 2.1.2 子集函数

Candidate itemsets ${C}_{k}$ are stored in a hash-tree. A node of the hash-tree either contains a list of itemsets (a leaf node) or a hash table (an interior node). In an interior node, each bucket of the hash table points to another node. The root of the hash-tree is defined to be at depth 1 . An interior node at depth $d$ points to nodes at depth $d + 1$ . Itemsets are stored in the leaves. When we add an itemset $c$ ,we start from the root and go down the tree until we reach a leaf. At an interior node at depth $d$ ,we decide which branch to follow by applying a hash function to the $d$ th item of the itemset. All nodes are initially created as leaf nodes. When the number of itemsets in a leaf node exceeds a specified threshold, the leaf node is converted to an interior node.

候选项集${C}_{k}$存储在哈希树中。哈希树的节点要么包含一个项集列表（叶节点），要么包含一个哈希表（内部节点）。在内部节点中，哈希表的每个桶指向另一个节点。哈希树的根节点定义为深度为1。深度为$d$的内部节点指向深度为$d + 1$的节点。项集存储在叶节点中。当我们添加一个项集$c$时，我们从根节点开始向下遍历树，直到到达一个叶节点。在深度为$d$的内部节点，我们通过对项集的第$d$项应用哈希函数来决定要跟随的分支。所有节点最初都创建为叶节点。当叶节点中的项集数量超过指定阈值时，叶节点将转换为内部节点。

Starting from the root node, the subset function finds all the candidates contained in a transaction $t$ as follows. If we are at a leaf,we find which of the itemsets in the leaf are contained in $t$ and add references to them to the answer set. If we are at an interior node and we have reached it by hashing the item $i$ ,we hash on each item that comes after $i$ in $t$ and recursively apply this procedure to the node in the corresponding bucket. For the root node, we hash on every item in $t$ .

子集函数从根节点开始，按如下方式找出事务$t$中包含的所有候选项集。如果我们处于叶节点，我们找出叶节点中的哪些项集包含在$t$中，并将它们的引用添加到答案集中。如果我们处于内部节点，并且是通过对项$i$进行哈希操作到达该节点的，我们对$t$中$i$之后的每个项进行哈希操作，并对相应桶中的节点递归应用此过程。对于根节点，我们对$t$中的每个项进行哈希操作。

To see why the subset function returns the desired set of references, consider what happens at the root node. For any itemset $c$ contained in transaction $t$ ,the first item of $c$ must be in $t$ . At the root,by hashing on every item in $t$ ,we ensure that we only ignore itemsets that start with an item not in $t$ . Similar arguments apply at lower depths. The only additional factor is that, since the items in any itemset are ordered, if we reach the current node by hashing the item $i$ ,we only need to consider the items in $t$ that occur after $i$ .

为了理解子集函数为何返回所需的引用集，考虑在根节点会发生什么。对于事务$t$中包含的任何项集$c$，$c$的第一个项必须在$t$中。在根节点，通过对$t$中的每个项进行哈希操作，我们确保只忽略以不在$t$中的项开头的项集。类似的论点适用于较低的深度。唯一额外的因素是，由于任何项集中的项是有序的，如果我们通过对项$i$进行哈希操作到达当前节点，我们只需要考虑$t$中在$i$之后出现的项。

### 2.2 Algorithm AprioriTid

### 2.2 AprioriTid算法

The AprioriTid algorithm, shown in Figure 2, also uses the apriori-gen function (given in Section 2.1.1) to determine the candidate itemsets before the pass begins. The interesting feature of this algorithm is that the database $\mathcal{D}$ is not used for counting support after the first pass. Rather,the set ${\bar{C}}_{k}$ is used for this purpose. Each member of the set ${\bar{C}}_{k}$ is of the form $< {TID},\left\{  {X}_{k}\right\}   >$ ,where each ${X}_{k}$ is a potentially large $k$ -itemset present in the transaction with identifier TID. For $k = 1,{\bar{C}}_{1}$ corresponds to the database $\mathcal{D}$ ,although conceptually each item $i$ is replaced by the itemset $\{ i\}$ . For $k > 1,{\bar{C}}_{k}$ is generated by the algorithm (step 10). The member of ${\bar{C}}_{k}$ corresponding to transaction $t$ is $< t.{TID}$ , $\left\{  {c \in  {C}_{k} \mid  c\text{contained in}t}\right\}   >$ . If a transaction does not contain any candidate $k$ -itemset,then ${\bar{C}}_{k}$ will not have an entry for this transaction. Thus, the number of entries in ${\bar{C}}_{k}$ may be smaller than the number of transactions in the database, especially for large values of $k$ . In addition,for large values of $k$ , each entry may be smaller than the corresponding transaction because very few candidates may be contained in the transaction. However, for small values for $k$ ,each entry may be larger than the corresponding transaction because an entry in ${C}_{k}$ includes all candidate $k$ -itemsets contained in the transaction.

如图2所示的AprioriTid算法（先验事务ID算法），同样使用apriori - gen函数（在2.1.1节给出）在扫描开始前确定候选项集。该算法的有趣之处在于，在第一次扫描之后，数据库$\mathcal{D}$不再用于支持度计数。相反，集合${\bar{C}}_{k}$用于此目的。集合${\bar{C}}_{k}$的每个成员形式为$< {TID},\left\{  {X}_{k}\right\}   >$，其中每个${X}_{k}$是标识符为TID的事务中可能存在的大型$k$ - 项集。对于$k = 1,{\bar{C}}_{1}$对应于数据库$\mathcal{D}$，尽管从概念上讲，每个项$i$都被项集$\{ i\}$所取代。对于$k > 1,{\bar{C}}_{k}$由算法生成（步骤10）。对应于事务$t$的${\bar{C}}_{k}$的成员是$< t.{TID}$，$\left\{  {c \in  {C}_{k} \mid  c\text{contained in}t}\right\}   >$。如果一个事务不包含任何候选$k$ - 项集，那么${\bar{C}}_{k}$中将不会有该事务的条目。因此，${\bar{C}}_{k}$中的条目数量可能小于数据库中的事务数量，特别是当$k$的值较大时。此外，对于较大的$k$值，每个条目可能小于相应的事务，因为事务中可能只包含很少的候选项。然而，对于较小的$k$值，每个条目可能大于相应的事务，因为${C}_{k}$中的一个条目包含事务中所有的候选$k$ - 项集。

In Section 2.2.1, we give the data structures used to implement the algorithm. See [5] for a proof of correctness and a discussion of buffer management.

在2.2.1节中，我们给出了用于实现该算法的数据结构。有关正确性证明和缓冲区管理的讨论，请参阅文献[5]。

<!-- Media -->

---

) ${L}_{1} = \{$ large 1-itemsets $\}$ ;

) ${L}_{1} = \{$ 大型1 - 项集 $\}$ ;

	${\bar{C}}_{1} =$ database $\mathcal{D}$ ;

	${\bar{C}}_{1} =$ 数据库 $\mathcal{D}$ ;

1) for $\left( {k = 2;{L}_{k - 1} \neq  \varnothing ;k +  + }\right)$ do begin

1) 对于 $\left( {k = 2;{L}_{k - 1} \neq  \varnothing ;k +  + }\right)$ 执行以下操作：

			${C}_{k} = \operatorname{apriori-gen}\left( {L}_{k - 1}\right) ;//$ New candidates

			${C}_{k} = \operatorname{apriori-gen}\left( {L}_{k - 1}\right) ;//$ 新候选集

			${\bar{C}}_{k} = 0;$

		forall entries $t \in  {\bar{C}}_{k - 1}$ do begin

			 对所有条目 $t \in  {\bar{C}}_{k - 1}$ 执行以下操作：

					// determine candidate itemsets in ${C}_{k}$ contained

					// 确定 ${C}_{k}$ 中包含的候选项集

					// in the transaction with identifier $t$ . TID

					// 在标识符为 $t$ 的事务中。事务ID

					${C}_{t} = \left\{  {c \in  {C}_{k} \mid  \left( {c - c\left\lbrack  k\right\rbrack  }\right)  \in  t}\right.$ .set-of-itemsets $\land$

					${C}_{t} = \left\{  {c \in  {C}_{k} \mid  \left( {c - c\left\lbrack  k\right\rbrack  }\right)  \in  t}\right.$ .项集集合 $\land$

								$\left( {c - c\left\lbrack  {k - 1}\right\rbrack  }\right)  \in  t$ .set-of-itemsets $\}$ ;

								$\left( {c - c\left\lbrack  {k - 1}\right\rbrack  }\right)  \in  t$ .项集集合 $\}$ ;

					forall candidates $c \in  {C}_{t}$ do

									 对所有候选 $c \in  {C}_{t}$ 执行以下操作

							c.count++;

													  c.计数加1;

					if $\left( {{C}_{t} \neq  \varnothing }\right)$ then ${\bar{C}}_{k} +  =  < t$ . TID, ${C}_{t} >$ ;

									 如果 $\left( {{C}_{t} \neq  \varnothing }\right)$ 则 ${\bar{C}}_{k} +  =  < t$ . 事务ID, ${C}_{t} >$ ;

			end

					 结束

			${L}_{k} = \left\{  {c \in  {C}_{k} \mid  c\text{.count} \geq  \text{minsup}}\right\}$

	end

	 结束

	Answer $= \mathop{\bigcup }\limits_{k}{L}_{k}$ ;

	答案 $= \mathop{\bigcup }\limits_{k}{L}_{k}$ ;

---

## Figure 2: Algorithm AprioriTid

## 图2：AprioriTid算法

<!-- Media -->

Example Consider the database in Figure 3 and assume that minimum support is 2 transactions. Calling apriori-gen with ${L}_{1}$ at step 4 gives the candidate itemsets ${C}_{2}$ . In steps 6 through 10,we count the support of candidates in ${C}_{2}$ by iterating over the entries in ${\bar{C}}_{1}$ and generate ${\bar{C}}_{2}$ . The first entry in ${\bar{C}}_{1}$ is $\{ \{ 1\} \{ 3\} \{ 4\} \}$ ,corresponding to transaction 100. The ${C}_{t}$ at step 7 corresponding to this entry $t$ is $\{ \{ {13}\} \}$ ,because $\{ {13}\}$ is a member of ${C}_{2}$ and both $\left( {\{ {13}\} -\{ 1\} }\right)$ and $\left( {\{ {13}\} -\{ 3\} }\right)$ are members of $t$ -set-of-itemsets.

示例 考虑图3中的数据库，并假设最小支持度为2笔交易。在步骤4中对 ${L}_{1}$ 调用apriori - gen函数，得到候选项集 ${C}_{2}$ 。在步骤6到10中，我们通过遍历 ${\bar{C}}_{1}$ 中的条目来计算 ${C}_{2}$ 中候选项集的支持度，并生成 ${\bar{C}}_{2}$ 。 ${\bar{C}}_{1}$ 中的第一个条目是 $\{ \{ 1\} \{ 3\} \{ 4\} \}$ ，对应交易100。步骤7中对应此条目 $t$ 的 ${C}_{t}$ 是 $\{ \{ {13}\} \}$ ，因为 $\{ {13}\}$ 是 ${C}_{2}$ 的成员，并且 $\left( {\{ {13}\} -\{ 1\} }\right)$ 和 $\left( {\{ {13}\} -\{ 3\} }\right)$ 都是 $t$ 项集的成员。

Calling apriori-gen with ${L}_{2}$ gives ${C}_{3}$ . Making a pass over the data with ${\bar{C}}_{2}$ and ${C}_{3}$ generates ${\bar{C}}_{3}$ . Note that there is no entry in ${\bar{C}}_{3}$ for the transactions with TIDs 100 and 400 , since they do not contain any of the itemsets in ${C}_{3}$ . The candidate $\{ {235}\}$ in ${C}_{3}$ turns out to be large and is the only member of ${L}_{3}$ . When we generate ${C}_{4}$ using ${L}_{3}$ ,it turns out to be empty, and we terminate.

对 ${L}_{2}$ 调用apriori - gen函数得到 ${C}_{3}$ 。使用 ${\bar{C}}_{2}$ 和 ${C}_{3}$ 对数据进行一次遍历生成 ${\bar{C}}_{3}$ 。请注意， ${\bar{C}}_{3}$ 中没有TID为100和400的交易的条目，因为它们不包含 ${C}_{3}$ 中的任何项集。 ${C}_{3}$ 中的候选项集 $\{ {235}\}$ 结果为频繁项集，并且是 ${L}_{3}$ 的唯一成员。当我们使用 ${L}_{3}$ 生成 ${C}_{4}$ 时，结果为空，我们终止算法。

<!-- Media -->

Database ${C}_{1}$ ${C}_{2}$

数据库 ${C}_{1}$ ${C}_{2}$

<table><tr><td>TID</td><td>Items</td></tr><tr><td>100</td><td/></tr><tr><td>200</td><td/></tr><tr><td>300</td><td>1 2 3 5</td></tr><tr><td>400</td><td/></tr></table>

<table><tbody><tr><td>交易标识符（TID）</td><td>项目；物品</td></tr><tr><td>100</td><td></td></tr><tr><td>200</td><td></td></tr><tr><td>300</td><td>1 2 3 5</td></tr><tr><td>400</td><td></td></tr></tbody></table>

<table><tr><td>TID</td><td>Set-of-Itemsets</td></tr><tr><td>100</td><td>$\{ \{ 1\} ,\{ 3\} ,\{ 4\} \}$</td></tr><tr><td>200</td><td>$\{ \begin{matrix} \{ 2\} , & \{ 3\} , & \{ 5\} \;\}  \end{matrix}$</td></tr><tr><td>300</td><td>$\{ \begin{matrix} \{ 1\} , & \{ 2\} , & \{ 3\} , & \{ 5\}  \end{matrix}$</td></tr><tr><td>400</td><td/></tr></table>

<table><tbody><tr><td>事务标识符（Transaction ID）</td><td>项集集合</td></tr><tr><td>100</td><td>$\{ \{ 1\} ,\{ 3\} ,\{ 4\} \}$</td></tr><tr><td>200</td><td>$\{ \begin{matrix} \{ 2\} , & \{ 3\} , & \{ 5\} \;\}  \end{matrix}$</td></tr><tr><td>300</td><td>$\{ \begin{matrix} \{ 1\} , & \{ 2\} , & \{ 3\} , & \{ 5\}  \end{matrix}$</td></tr><tr><td>400</td><td></td></tr></tbody></table>

<table><tr><td colspan="2">${L}_{1}$</td></tr><tr><td>Itemset</td><td>Support</td></tr><tr><td>\{1\}</td><td>2</td></tr><tr><td>$\{ 2\}$</td><td>3</td></tr><tr><td>$\{ 3\}$</td><td>3</td></tr><tr><td>\{5\}</td><td>3</td></tr></table>

<table><tbody><tr><td colspan="2">${L}_{1}$</td></tr><tr><td>项集（Itemset）</td><td>支持度（Support）</td></tr><tr><td>\{1\}</td><td>2</td></tr><tr><td>$\{ 2\}$</td><td>3</td></tr><tr><td>$\{ 3\}$</td><td>3</td></tr><tr><td>\{5\}</td><td>3</td></tr></tbody></table>

<table><tr><td>Itemset</td><td>Support</td></tr><tr><td>2\}</td><td>1</td></tr><tr><td>\{13\}</td><td>2</td></tr><tr><td>\{1</td><td>1</td></tr><tr><td>$\{ 2\;3\}$</td><td>2</td></tr><tr><td>\{25\}</td><td>3</td></tr><tr><td/><td>2</td></tr></table>

<table><tbody><tr><td>项集（Itemset）</td><td>支持度（Support）</td></tr><tr><td>2\}</td><td>1</td></tr><tr><td>\{13\}</td><td>2</td></tr><tr><td>\{1</td><td>1</td></tr><tr><td>$\{ 2\;3\}$</td><td>2</td></tr><tr><td>\{25\}</td><td>3</td></tr><tr><td></td><td>2</td></tr></tbody></table>

${C}_{2}$ ${L}_{2}$

<table><tr><td>TID</td><td>Set-of-Itemsets</td></tr><tr><td>100</td><td>$\{ \{ 1,3\} \}$</td></tr><tr><td>200</td><td>$\{ \{ 2\;3\} ,\{ 2\;5\} ,\{ 3\;5\} \}$</td></tr><tr><td>300</td><td>$\{ \{ 1\;2\} ,\;\{ 1\;3\} ,\;\{ 1\;5\} ,$</td></tr><tr><td/><td>$\{ 2\;3\} ,\;\{ 2\;5\} ,\;\{ 3\;5\} \;\}$</td></tr><tr><td>400</td><td>$\{ \{ 25\} \}$</td></tr></table>

<table><tbody><tr><td>事务标识符（TID）</td><td>项集集合（Set-of-Itemsets）</td></tr><tr><td>100</td><td>$\{ \{ 1,3\} \}$</td></tr><tr><td>200</td><td>$\{ \{ 2\;3\} ,\{ 2\;5\} ,\{ 3\;5\} \}$</td></tr><tr><td>300</td><td>$\{ \{ 1\;2\} ,\;\{ 1\;3\} ,\;\{ 1\;5\} ,$</td></tr><tr><td></td><td>$\{ 2\;3\} ,\;\{ 2\;5\} ,\;\{ 3\;5\} \;\}$</td></tr><tr><td>400</td><td>$\{ \{ 25\} \}$</td></tr></tbody></table>

Itemset Support

项集支持度

$\{ 1\;3\}$

\{2 3\} 2

\{2 5\} 3

\{3 5\}

<table><tr><td>TID</td><td>Set-of-Itemsets</td></tr><tr><td>200 300</td><td/></tr></table>

<table><tbody><tr><td>事务标识符（TID）</td><td>项集集合（Set-of-Itemsets）</td></tr><tr><td>200 300</td><td></td></tr></tbody></table>

<!-- figureText: ${C}_{3}$ Itemset Support 2 ${L}_{3}$ Support 2 $\{ 2\;3\;5\}$ Itemset $\{ {235}\}$ -->

<img src="https://cdn.noedgeai.com/0195c900-f548-7e6e-93eb-62d54d470345_4.jpg?x=948&y=1067&w=276&h=266&r=0"/>

${C}_{3}$

Figure 3: Example

图3：示例

<!-- Media -->

#### 2.2.1 Data Structures

#### 2.2.1 数据结构

We assign each candidate itemset a unique number, called its ID. Each set of candidate itemsets ${C}_{k}$ is kept in an array indexed by the IDs of the itemsets in ${C}_{k}$ . A member of ${\bar{C}}_{k}$ is now of the form $<$ TID, $\{ \mathrm{{ID}}\}  >$ . Each ${\bar{C}}_{k}$ is stored in a sequential structure.

我们为每个候选项集分配一个唯一的编号，称为其ID。每个候选项集集合 ${C}_{k}$ 保存在一个数组中，该数组由 ${C}_{k}$ 中项集的ID索引。${\bar{C}}_{k}$ 的一个成员现在的形式为 $<$ 事务ID（TID），$\{ \mathrm{{ID}}\}  >$ 。每个 ${\bar{C}}_{k}$ 存储在一个顺序结构中。

The apriori-gen function generates a candidate $k$ - itemset ${c}_{k}$ by joining two large(k - 1)-itemsets. We maintain two additional fields for each candidate itemset: i) generators and ii) extensions. The generators field of a candidate itemset ${c}_{k}$ stores the IDs of the two large(k - 1)-itemsets whose join generated ${c}_{k}$ . The extensions field of an itemset ${c}_{k}$ stores the IDs of all the $\left( {k + 1}\right)$ -candidates that are extensions of ${c}_{k}$ . Thus,when a candidate ${c}_{k}$ is generated by joining ${l}_{k - 1}^{1}$ and ${l}_{k - 1}^{2}$ ,we save the IDs of ${l}_{k - 1}^{1}$ and ${l}_{k - 1}^{2}$ in the generators field for ${c}_{k}$ . At the same time,the ID of ${c}_{k}$ is added to the extensions field of ${l}_{k - 1}^{1}$ .

apriori - gen函数通过连接两个大的(k - 1) - 项集来生成一个候选 $k$ - 项集 ${c}_{k}$ 。我们为每个候选项集维护两个额外的字段：i) 生成元（generators）和ii) 扩展项（extensions）。候选项集 ${c}_{k}$ 的生成元字段存储生成 ${c}_{k}$ 的两个大的(k - 1) - 项集的ID。项集 ${c}_{k}$ 的扩展项字段存储所有作为 ${c}_{k}$ 扩展的 $\left( {k + 1}\right)$ - 候选项集的ID。因此，当通过连接 ${l}_{k - 1}^{1}$ 和 ${l}_{k - 1}^{2}$ 生成候选 ${c}_{k}$ 时，我们将 ${l}_{k - 1}^{1}$ 和 ${l}_{k - 1}^{2}$ 的ID保存在 ${c}_{k}$ 的生成元字段中。同时，将 ${c}_{k}$ 的ID添加到 ${l}_{k - 1}^{1}$ 的扩展项字段中。

We now describe how Step 7 of Figure 2 is implemented using the above data structures. Recall that the $t$ .set-of-itemsets field of an entry $t$ in ${\bar{C}}_{k - 1}$ gives the IDs of all(k - 1)-candidates contained in transaction $t$ .TID. For each such candidate ${c}_{k - 1}$ the extensions field gives ${T}_{k}$ ,the set of IDs of all the candidate $k$ -itemsets that are extensions of ${c}_{k - 1}$ . For each ${c}_{k}$ in ${T}_{k}$ ,the generators field gives the IDs of the two itemsets that generated ${c}_{k}$ . If these itemsets are present in the entry for $t$ .set-of-itemsets,we can conclude that ${c}_{k}$ is present in transaction $t$ .TID,and add ${c}_{k}$ to ${C}_{t}$ .

我们现在描述如何使用上述数据结构实现图2的步骤7。回想一下，${\bar{C}}_{k - 1}$ 中条目 $t$ 的 $t$ .项集集合字段给出了包含在事务 $t$ .TID中的所有(k - 1) - 候选项集的ID。对于每个这样的候选 ${c}_{k - 1}$ ，扩展项字段给出 ${T}_{k}$ ，即所有作为 ${c}_{k - 1}$ 扩展的候选 $k$ - 项集的ID集合。对于 ${T}_{k}$ 中的每个 ${c}_{k}$ ，生成元字段给出生成 ${c}_{k}$ 的两个项集的ID。如果这些项集存在于 $t$ .项集集合的条目中，我们可以得出 ${c}_{k}$ 存在于事务 $t$ .TID中，并将 ${c}_{k}$ 添加到 ${C}_{t}$ 中。

## 3 Performance

## 3 性能

To assess the relative performance of the algorithms for discovering large sets, we performed several experiments on an IBM RS/6000 530H workstation with a CPU clock rate of ${33}\mathrm{{MHz}},{64}\mathrm{{MB}}$ of main memory, and running AIX 3.2. The data resided in the AIX file system and was stored on a 2GB SCSI 3.5" drive, with measured sequential throughput of about 2 MB/second.

为了评估发现大项集算法的相对性能，我们在一台IBM RS/6000 530H工作站上进行了多次实验，该工作站的CPU时钟频率为 ${33}\mathrm{{MHz}},{64}\mathrm{{MB}}$ ，主内存大小为 [latex1] ，运行AIX 3.2操作系统。数据存储在AIX文件系统中，并存储在一个2GB的SCSI 3.5英寸驱动器上，测得的顺序吞吐量约为2 MB/秒。

We first give an overview of the AIS [4] and SETM [13] algorithms against which we compare the performance of the Apriori and AprioriTid algorithms. We then describe the synthetic datasets used in the performance evaluation and show the performance results. Finally, we describe how the best performance features of Apriori and AprioriTid can be combined into an AprioriHybrid algorithm and demonstrate its scale-up properties.

我们首先概述AIS [4] 和SETM [13] 算法，我们将Apriori和AprioriTid算法的性能与它们进行比较。然后我们描述性能评估中使用的合成数据集并展示性能结果。最后，我们描述如何将Apriori和AprioriTid的最佳性能特征组合成一个AprioriHybrid算法，并展示其可扩展性。

### 3.1 The AIS Algorithm

### 3.1 AIS算法

Candidate itemsets are generated and counted on-the-fly as the database is scanned. After reading a transaction, it is determined which of the itemsets that were found to be large in the previous pass are contained in this transaction. New candidate itemsets are generated by extending these large itemsets with other items in the transaction. A large itemset $l$ is extended with only those items that are large and occur later in the lexicographic ordering of items than any of the items in $l$ . The candidates generated from a transaction are added to the set of candidate itemsets maintained for the pass, or the counts of the corresponding entries are increased if they were created by an earlier transaction. See [4] for further details of the AIS algorithm.

在扫描数据库时，候选项集会即时生成并计数。读取一个事务后，确定在前一次扫描中被发现为大项集的哪些项集包含在该事务中。通过用事务中的其他项扩展这些大项集来生成新的候选项集。一个大项集 $l$ 仅用那些是大项且在项的字典序中比 $l$ 中的任何项出现更晚的项进行扩展。从一个事务生成的候选项集被添加到为该次扫描维护的候选项集集合中，如果它们是由早期事务创建的，则相应条目的计数会增加。有关AIS算法的更多详细信息，请参阅 [4] 。

### 3.2 The SETM Algorithm

### 3.2 SETM算法

The SETM algorithm [13] was motivated by the desire to use SQL to compute large itemsets. Like AIS, the SETM algorithm also generates candidates on-the-fly based on transactions read from the database. It thus generates and counts every candidate itemset that the AIS algorithm generates. However, to use the standard SQL join operation for candidate generation, SETM separates candidate generation from couniting. It saves a copy of the candidate itemset together with the TID of the generating transaction in a sequential structure. At the end of the pass, the support count of candidate itemsets is determined by sorting and aggregating this sequential structure.

SETM算法[13]的设计初衷是希望使用SQL来计算大型项集。与AIS算法一样，SETM算法也会根据从数据库中读取的事务动态生成候选项集。因此，它会生成并统计AIS算法所生成的每个候选项集。然而，为了使用标准的SQL连接操作来生成候选项集，SETM算法将候选项集的生成与计数分开进行。它会将候选项集的副本以及生成该候选项集的事务的事务标识符（TID）存储在一个顺序结构中。在遍历结束时，通过对这个顺序结构进行排序和聚合来确定候选项集的支持度计数。

SETM remembers the TIDs of the generating transactions with the candidate itemsets. To avoid needing a subset operation, it uses this information to determine the large itemsets contained in the transaction read. ${\bar{L}}_{k} \subseteq  {\bar{C}}_{k}$ and is obtained by deleting those candidates that do not have minimum support. Assuming that the database is sorted in TID order, SETM can easily find the large itemsets contained in a transaction in the next pass by sorting ${\bar{L}}_{k}$ on TID. In fact,it needs to visit every member of ${\bar{L}}_{k}$ only once in the TID order, and the candidate generation can be performed using the relational merge-join operation [13].

SETM算法会将生成候选项集的事务的TID与候选项集关联起来。为了避免使用子集操作，它利用这些信息来确定所读取事务中包含的大型项集。${\bar{L}}_{k} \subseteq  {\bar{C}}_{k}$是通过删除那些不满足最小支持度的候选项集而得到的。假设数据库是按照TID顺序排序的，SETM算法可以通过对${\bar{L}}_{k}$按TID进行排序，轻松地在下一次遍历中找到事务中包含的大型项集。实际上，它只需要按照TID顺序对${\bar{L}}_{k}$中的每个元素访问一次，并且可以使用关系合并连接操作来进行候选项集的生成[13]。

The disadvantage of this approach is mainly due to the size of candidate sets ${\bar{C}}_{k}$ . For each candidate itemset, the candidate set now has as many entries as the number of transactions in which the candidate itemset is present. Moreover, when we are ready to count the support for candidate itemsets at the end of the pass, ${\bar{C}}_{k}$ is in the wrong order and needs to be sorted on itemsets. After counting and pruning out small candidate itemsets that do not have minimum support,the resulting set ${\bar{L}}_{k}$ needs another sort on TID before it can be used for generating candidates in the next pass.

这种方法的缺点主要源于候选项集${\bar{C}}_{k}$的规模。对于每个候选项集，候选项集现在的条目数量与该候选项集所在的事务数量相同。此外，当我们在遍历结束时准备统计候选项集的支持度时，${\bar{C}}_{k}$的顺序是错误的，需要按项集进行排序。在统计并修剪掉那些不满足最小支持度的小候选项集之后，得到的集合${\bar{L}}_{k}$需要再按TID进行一次排序，才能用于下一次遍历中生成候选项集。

### 3.3 Generation of Synthetic Data

### 3.3 合成数据的生成

We generated synthetic transactions to evaluate the performance of the algorithms over a large range of data characteristics. These transactions mimic the transactions in the retailing environment. Our model of the "real" world is that people tend to buy sets of items together. Each such set is potentially a maximal large itemset. An example of such a set might be sheets, pillow case, comforter, and ruffles. However, some people may buy only some of the items from such a set. For instance, some people might buy only sheets and pillow case, and some only sheets. A transaction may contain more than one large itemset. For example, a customer might place an order for a dress and jacket when ordering sheets and pillow cases, where the dress and jacket together form another large itemset. Transaction sizes are typically clustered around a mean and a few transactions have many items. Typical sizes of large itemsets are also clustered around a mean, with a few large itemsets having a large number of items.

我们生成了合成事务，以评估算法在各种数据特征下的性能。这些事务模拟了零售环境中的交易。我们对“现实”世界的模型假设是，人们倾向于一起购买一组商品。每一组这样的商品都有可能是一个最大的大型项集。例如，一组商品可能包括床单、枕套、被子和床罩花边。然而，有些人可能只购买这组商品中的部分商品。例如，有些人可能只购买床单和枕套，而有些人只购买床单。一个事务可能包含多个大型项集。例如，一位顾客在订购床单和枕套时可能还会订购一条连衣裙和一件夹克，其中连衣裙和夹克一起构成另一个大型项集。事务的大小通常围绕一个平均值聚类，只有少数事务包含大量商品。大型项集的典型大小也围绕一个平均值聚类，只有少数大型项集包含大量商品。

To create a dataset, our synthetic data generation program takes the parameters shown in Table 2.

为了创建一个数据集，我们的合成数据生成程序会使用表2中所示的参数。

<!-- Media -->

Table 2: Parameters

表2：参数

<table><tr><td>| D</td><td>Number of transactions</td></tr><tr><td>$\left| T\right|$</td><td>Average size of the transactions</td></tr><tr><td>|</td><td>Average size of the maximal potentially</td></tr><tr><td/><td>large itemsets</td></tr><tr><td>$\left| L\right|$</td><td>Number of maximal potentially large itemsets</td></tr><tr><td>$N$</td><td>Number of items</td></tr></table>

<table><tbody><tr><td>| D</td><td>交易数量</td></tr><tr><td>$\left| T\right|$</td><td>交易的平均规模</td></tr><tr><td>|</td><td>最大潜在的平均规模</td></tr><tr><td></td><td>大项集</td></tr><tr><td>$\left| L\right|$</td><td>最大潜在大项集的数量</td></tr><tr><td>$N$</td><td>项目数量</td></tr></tbody></table>

<!-- Media -->

We first determine the size of the next transaction. The size is picked from a Poisson distribution with mean $\mu$ equal to $\left| T\right|$ . Note that if each item is chosen with the same probability $p$ ,and there are $N$ items, the expected number of items in a transaction is given by a binomial distribution with parameters $N$ and $p$ , and is approximated by a Poisson distribution with mean ${Np}$ .

我们首先确定下一笔交易的规模。该规模从均值 $\mu$ 等于 $\left| T\right|$ 的泊松分布中选取。请注意，如果每个物品被选中的概率相同，均为 $p$，且共有 $N$ 个物品，那么一笔交易中物品的期望数量由参数为 $N$ 和 $p$ 的二项分布给出，并且可以用均值为 ${Np}$ 的泊松分布来近似。

We then assign items to the transaction. Each transaction is assigned a series of potentially large itemsets. If the large itemset on hand does not fit in the transaction, the itemset is put in the transaction anyway in half the cases, and the itemset is moved to the next transaction the rest of the cases.

然后，我们为交易分配物品。每笔交易都会被分配一系列潜在的大项集。如果手头的大项集无法完全放入当前交易中，那么在一半的情况下，仍会将该大项集放入当前交易；在另一半的情况下，则将该大项集移至下一笔交易。

Large itemsets are chosen from a set $\mathcal{T}$ of such itemsets. The number of itemsets in $\mathcal{T}$ is set to $\left| L\right|$ . There is an inverse relationship between $\left| L\right|$ and the average support for potentially large itemsets. An itemset in $\mathcal{T}$ is generated by first picking the size of the itemset from a Poisson distribution with mean $\mu$ equal to $\left| I\right|$ . Items in the first itemset are chosen randomly. To model the phenomenon that large itemsets often have common items, some fraction of items in subsequent itemsets are chosen from the previous itemset generated. We use an exponentially distributed random variable with mean equal to the correlation level to decide this fraction for each itemset. The remaining items are picked at random. In the datasets used in the experiments, the correlation level was set to 0.5 . We ran some experiments with the correlation level set to 0.25 and 0.75 but did not find much difference in the nature of our performance results.

大项集是从这样的项集集合 $\mathcal{T}$ 中选取的。$\mathcal{T}$ 中的项集数量设定为 $\left| L\right|$。$\left| L\right|$ 与潜在大项集的平均支持度之间存在反比关系。$\mathcal{T}$ 中的一个项集是这样生成的：首先从均值 $\mu$ 等于 $\left| I\right|$ 的泊松分布中选取项集的规模。第一个项集中的物品是随机选取的。为了模拟大项集通常包含共同物品这一现象，后续项集中的部分物品会从之前生成的项集中选取。我们使用均值等于相关水平的指数分布随机变量来为每个项集确定这一比例。其余物品则随机选取。在实验所用的数据集中，相关水平设定为 0.5。我们还进行了一些相关水平设定为 0.25 和 0.75 的实验，但发现性能结果的性质没有太大差异。

Each itemset in $\mathcal{T}$ has a weight associated with it, which corresponds to the probability that this itemset will be picked. This weight is picked from an exponential distribution with unit mean, and is then normalized so that the sum of the weights for all the itemsets in $\mathcal{T}$ is 1 . The next itemset to be put in the transaction is chosen from $\mathcal{T}$ by tossing an $\left| L\right|$ - sided weighted coin, where the weight for a side is the probability of picking the associated itemset.

$\mathcal{T}$ 中的每个项集都关联着一个权重，该权重对应于该项集被选中的概率。这个权重从均值为 1 的指数分布中选取，然后进行归一化处理，使得 $\mathcal{T}$ 中所有项集的权重之和为 1。要放入交易中的下一个项集是通过抛掷一个 $\left| L\right|$ 面加权硬币从 $\mathcal{T}$ 中选取的，其中每一面的权重就是选取相关项集的概率。

To model the phenomenon that all the items in a large itemset are not always bought together, we assign each itemset in $\mathcal{T}$ a corruption level $c$ . When adding an itemset to a transaction, we keep dropping an item from the itemset as long as a uniformly distributed random number between 0 and 1 is less than $c$ . Thus for an itemset of size $l$ ,we will add $l$ items to the transaction $1 - c$ of the time, $l - 1$ items $c\left( {1 - c}\right)$ of the time, $l - 2$ items ${c}^{2}\left( {1 - c}\right)$ of the time, etc. The corruption level for an itemset is fixed and is obtained from a normal distribution with mean 0.5 and variance 0.1 .

为了模拟大项集中的所有物品并非总是一起购买这一现象，我们为 $\mathcal{T}$ 中的每个项集分配一个损坏水平 $c$。当将一个项集添加到交易中时，只要在 0 到 1 之间均匀分布的随机数小于 $c$，我们就会从该项集中移除一个物品。因此，对于规模为 $l$ 的项集，我们有 $1 - c$ 的概率会将 $l$ 个物品添加到交易中，有 $c\left( {1 - c}\right)$ 的概率添加 $l - 1$ 个物品，有 ${c}^{2}\left( {1 - c}\right)$ 的概率添加 $l - 2$ 个物品，依此类推。项集的损坏水平是固定的，它从均值为 0.5、方差为 0.1 的正态分布中获取。

We generated datasets by setting $N = {1000}$ and $\left| L\right|$ $= {2000}$ . We chose 3 values for $\left| T\right|  : 5,{10}$ ,and 20 . We also chose 3 values for $\left| I\right|  : 2,4$ ,and 6 . The number of transactions was to set to 100,000 because, as we will see in Section 3.4, SETM could not be run for larger values. However, for our scale-up experiments, we generated datasets with up to 10 million transactions (838MB for T20). Table 3 summarizes the dataset parameter settings. For the same $\left| T\right|$ and $\left| D\right|$ values, the size of datasets in megabytes were roughly equal for the different values of $\left| I\right|$ .

我们通过设置 $N = {1000}$ 和 $\left| L\right|$ $= {2000}$ 来生成数据集。我们为 $\left| T\right|  : 5,{10}$ 选取了 3 个值，分别为  和 20。我们还为 $\left| I\right|  : 2,4$ 选取了 3 个值，分别为  和 6。交易数量设定为 100,000，因为正如我们将在第 3.4 节中看到的，对于更大的值，SETM 无法运行。然而，在我们的扩展实验中，我们生成了交易数量多达 1000 万的数据集（对于 T20 为 838MB）。表 3 总结了数据集的参数设置。对于相同的 $\left| T\right|$ 和 $\left| D\right|$ 值，不同 $\left| I\right|$ 值对应的数据集大小（以兆字节为单位）大致相等。

<!-- Media -->

Table 3: Parameter settings

表 3：参数设置

<table><tr><td>Name</td><td>T</td><td>I</td><td>$D$</td><td>Size in Megabytes</td></tr><tr><td>T5.12.D100K</td><td>5</td><td>2</td><td>100K</td><td>2.4</td></tr><tr><td>T10.I2.D100K</td><td>10</td><td>2</td><td>100K</td><td>4.4</td></tr><tr><td>T10.I4.D100K</td><td>10</td><td>4</td><td>100K</td><td/></tr><tr><td>T20.I2.D100K</td><td>20</td><td>2</td><td>100K</td><td rowspan="3">8.4</td></tr><tr><td>T20.I4.D100K</td><td>20</td><td>4</td><td>100K</td></tr><tr><td>T20.I6.D100K</td><td>20</td><td>6</td><td>100K</td></tr></table>

<table><tbody><tr><td>名称</td><td>T</td><td>I</td><td>$D$</td><td>大小（兆字节）</td></tr><tr><td>T5.12.D100K</td><td>5</td><td>2</td><td>100K</td><td>2.4</td></tr><tr><td>T10.I2.D100K</td><td>10</td><td>2</td><td>100K</td><td>4.4</td></tr><tr><td>T10.I4.D100K</td><td>10</td><td>4</td><td>100K</td><td></td></tr><tr><td>T20.I2.D100K</td><td>20</td><td>2</td><td>100K</td><td rowspan="3">8.4</td></tr><tr><td>T20.I4.D100K</td><td>20</td><td>4</td><td>100K</td></tr><tr><td>T20.I6.D100K</td><td>20</td><td>6</td><td>100K</td></tr></tbody></table>

<!-- Media -->

### 3.4 Relative Performance

### 3.4 相对性能

Figure 4 shows the execution times for the six synthetic datasets given in Table 3 for decreasing values of minimum support. As the minimum support decreases, the execution times of all the algorithms increase because of increases in the total number of candidate and large itemsets.

图4展示了表3中六个合成数据集在最小支持度值递减情况下的执行时间。随着最小支持度的降低，由于候选项集和大项集的总数增加，所有算法的执行时间都会增加。

For SETM, we have only plotted the execution times for the dataset T5.I2.D100K in Figure 4. The execution times for SETM for the two datasets with an average transaction size of 10 are given in Table 4. We did not plot the execution times in Table 4 on the corresponding graphs because they are too large compared to the execution times of the other algorithms. For the three datasets with transaction sizes of 20, SETM took too long to execute and we aborted those runs as the trends were clear. Clearly, Apriori beats SETM by more than an order of magnitude for large datasets.

对于SETM算法，在图4中我们仅绘制了数据集T5.I2.D100K的执行时间。表4给出了SETM算法在平均事务大小为10的两个数据集上的执行时间。我们没有将表4中的执行时间绘制在相应的图表上，因为与其他算法的执行时间相比，它们太长了。对于事务大小为20的三个数据集，SETM算法执行时间过长，由于趋势已经很明显，我们中止了这些运行。显然，对于大型数据集，Apriori算法的性能比SETM算法高出一个数量级以上。

<!-- Media -->

<!-- figureText: T5.I2.D100K T10.I2.D100K 160 AIS 140 AprioriTid Apriori 120 Time (sec) 100 80 60 40 20 0 1.5 0.75 0.5 0.33 0.25 Minimum Support T20.I2.D100K 1000 AIS 900 AprioriTid 800 Aprion 700 Time (sec) 600 500 400 300 200 100 0 1.5 0.75 0.5 0.33 0.25 Minimum Support T20.I6.D100K 3500 AIS 3000 AprioriTid Apriori 2500 Time (sec) 2000 1500 1000 500 2 1.5 0.75 0.33 0.25 Minimum Support 80 70 AIS AprioriTid 60 Aprior Time (sec) 50 40 30 20 10 0 2 1.5 0.33 0.25 Minimum Support T10.I4.D100K 350 AIS - 300 AprioriTid Apriori 250 Time (sec) 200 150 100 50 1.5 0.75 0.5 0.33 0.25 Minimum Support T20.I4.D100K 1800 AIS 1600 AprioriTid Apriori 1400 1200 Time (sec) 1000 800 600 400 200 0 1.5 0.75 0.5 0.33 0.25 Minimum Support -->

<img src="https://cdn.noedgeai.com/0195c900-f548-7e6e-93eb-62d54d470345_7.jpg?x=184&y=137&w=1359&h=1821&r=0"/>

Figure 4: Execution times

图4：执行时间

Table 4: Execution times in seconds for SETM

表4：SETM算法的执行时间（秒）

<table><tr><td rowspan="2">Algorithm</td><td colspan="5">Minimum Support</td></tr><tr><td>2.0%</td><td>1.5%</td><td>1.0%</td><td>0.75%</td><td>0.5%</td></tr><tr><td colspan="6">Dataset T10.I2.D100K</td></tr><tr><td>SETM</td><td>74</td><td>161</td><td>838</td><td>1262</td><td>1878</td></tr><tr><td>Apriori</td><td>4.4</td><td>5.3</td><td>11.0</td><td>14.5</td><td>15.3</td></tr><tr><td colspan="6">Dataset T10.I4.D100K</td></tr><tr><td>SETM</td><td>41</td><td>91</td><td>659</td><td>929</td><td>1639</td></tr><tr><td>Apriori</td><td>3.8</td><td>4.8</td><td>11.2</td><td>17.4</td><td>19.3</td></tr></table>

<table><tbody><tr><td rowspan="2">算法</td><td colspan="5">最小支持度</td></tr><tr><td>2.0%</td><td>1.5%</td><td>1.0%</td><td>0.75%</td><td>0.5%</td></tr><tr><td colspan="6">数据集T10.I2.D100K</td></tr><tr><td>SETM（原文未明确含义，保留英文）</td><td>74</td><td>161</td><td>838</td><td>1262</td><td>1878</td></tr><tr><td>先验算法（Apriori）</td><td>4.4</td><td>5.3</td><td>11.0</td><td>14.5</td><td>15.3</td></tr><tr><td colspan="6">数据集T10.I4.D100K</td></tr><tr><td>SETM（原文未明确含义，保留英文）</td><td>41</td><td>91</td><td>659</td><td>929</td><td>1639</td></tr><tr><td>先验算法（Apriori）</td><td>3.8</td><td>4.8</td><td>11.2</td><td>17.4</td><td>19.3</td></tr></tbody></table>

<!-- Media -->

Apriori beats AIS for all problem sizes, by factors ranging from 2 for high minimum support to more than an order of magnitude for low levels of support. AIS always did considerably better than SETM. For small problems, AprioriTid did about as well as Apriori, but performance degraded to about twice as slow for large problems.

对于所有问题规模，Apriori（先验算法）都优于AIS算法，在高最小支持度下优势因子为2，在低支持度下优势超过一个数量级。AIS算法的性能始终明显优于SETM算法。对于小规模问题，AprioriTid算法的性能与Apriori算法大致相当，但在大规模问题上，其性能下降至约为Apriori算法的两倍慢。

### 3.5 Explanation of the Relative Performance

### 3.5 相对性能解释

To explain these performance trends, we show in Figure 5 the sizes of the large and candidate sets in different passes for the T10.I4.D100K dataset for the minimum support of ${0.75}\%$ . Note that the $\mathrm{Y}$ -axis in this graph has a log scale.

为了解释这些性能趋势，我们在图5中展示了T10.I4.D100K数据集在最小支持度为${0.75}\%$时，不同遍次中大型项集和候选集的规模。请注意，此图中的$\mathrm{Y}$轴采用对数刻度。

<!-- Media -->

<!-- figureText: 18+07 (SETM) C-K (AIS, SETM) C-k (Apriori, AprioriTid) L-k 6 Pass Number 1e+06 Number of Itemsets 100000 10000 1000 100 10 1 2 -->

<img src="https://cdn.noedgeai.com/0195c900-f548-7e6e-93eb-62d54d470345_8.jpg?x=170&y=1062&w=621&h=515&r=0"/>

Figure 5: Sizes of the large and candidate sets (T10.I4.D100K, minsup $= {0.75}\%$ )

图5：大型项集和候选集的规模（T10.I4.D100K，最小支持度$= {0.75}\%$）

<!-- Media -->

The fundamental problem with the SETM algorithm is the size of its ${\bar{C}}_{k}$ sets. Recall that the size of the set ${\bar{C}}_{k}$ is given by

SETM算法的根本问题在于其${\bar{C}}_{k}$集的规模。回顾一下，${\bar{C}}_{k}$集的规模由下式给出

$\sum$ support-count(c).

$\sum$ 支持计数(c)。

candidate itemsets $c$

候选项集 $c$

Thus,the sets ${\bar{C}}_{k}$ are roughly $S$ times bigger than the corresponding ${C}_{k}$ sets,where $S$ is the average support count of the candidate itemsets. Unless the problem size is very small,the ${\bar{C}}_{k}$ sets have to be written to disk, and externally sorted twice, causing the SETM algorithm to perform poorly. ${}^{2}$ This explains the jump in time for SETM in Table 4 when going from 1.5% support to 1.0% support for datasets with transaction size 10 . The largest dataset in the scaleup experiments for SETM in [13] was still small enough that ${\bar{C}}_{k}$ could fit in memory; hence they did not encounter this jump in execution time. Note that for the same minimum support, the support count for candidate itemsets increases linearly with the number of transactions. Thus, as we increase the number of transactions for the same values of $\left| T\right|$ and $\left| I\right|$ ,though the size of ${C}_{k}$ does not change,the size of ${\bar{C}}_{k}$ goes up linearly. Thus, for datasets with more transactions, the performance gap between SETM and the other algorithms will become even larger.

因此，${\bar{C}}_{k}$集大约比相应的${C}_{k}$集大$S$倍，其中$S$是候选项集的平均支持计数。除非问题规模非常小，否则${\bar{C}}_{k}$集必须写入磁盘，并进行两次外部排序，这导致SETM算法性能不佳。${}^{2}$ 这解释了表4中，对于事务大小为10的数据集，当支持度从1.5%降至1.0%时，SETM算法的运行时间出现跳跃的原因。在文献[13]中SETM算法的扩展实验里，最大的数据集仍然足够小，使得${\bar{C}}_{k}$能够装入内存；因此他们没有遇到这种执行时间的跳跃。请注意，对于相同的最小支持度，候选项集的支持计数随事务数量线性增加。因此，当我们在$\left| T\right|$和$\left| I\right|$值相同的情况下增加事务数量时，尽管${C}_{k}$的规模不变，但${\bar{C}}_{k}$的规模会线性增加。因此，对于事务更多的数据集，SETM算法与其他算法之间的性能差距将变得更大。

The problem with AIS is that it generates too many candidates that later turn out to be small, causing it to waste too much effort. Apriori also counts too many small sets in the second pass (recall that ${C}_{2}$ is really a cross-product of ${L}_{1}$ with $\left. {L}_{1}\right)$ . However,this wastage decreases dramatically from the third pass onward. Note that for the example in Figure 5, after pass 3, almost every candidate itemset counted by Apriori turns out to be a large set.

AIS算法的问题在于它生成了太多后来被证明是小型的候选集，导致它浪费了过多的精力。Apriori算法在第二遍中也对太多小型项集进行计数（回顾一下，${C}_{2}$实际上是${L}_{1}$与$\left. {L}_{1}\right)$的叉积）。然而，从第三遍开始，这种浪费显著减少。请注意，对于图5中的示例，在第三遍之后，Apriori算法计数的几乎每个候选项集都被证明是大型项集。

AprioriTid also has the problem of SETM that ${\bar{C}}_{k}$ tends to be large. However, the apriori candidate generation used by AprioriTid generates significantly fewer candidates than the transaction-based candidate generation used by SETM. As a result,the ${\bar{C}}_{k}$ of AprioriTid has fewer entries than that of SETM. Apri-oriTid is also able to use a single word (ID) to store a candidate rather than requiring as many words as the number of items in the candidate. ${}^{3}$ In addition, unlike SETM,AprioriTid does not have to sort ${\bar{C}}_{k}$ . Thus, AprioriTid does not suffer as much as SETM from maintaining ${\bar{C}}_{k}$ .

AprioriTid算法也存在SETM算法的问题，即${\bar{C}}_{k}$往往规模较大。然而，AprioriTid算法使用的先验候选集生成方法比SETM算法使用的基于事务的候选集生成方法生成的候选集显著减少。结果，AprioriTid算法的${\bar{C}}_{k}$条目比SETM算法的少。AprioriTid算法还能够使用单个字（ID）来存储一个候选集，而不是像SETM算法那样需要与候选集中项的数量相同的字数。${}^{3}$ 此外，与SETM算法不同，AprioriTid算法不必对${\bar{C}}_{k}$进行排序。因此，AprioriTid算法在维护${\bar{C}}_{k}$方面不像SETM算法那样受到很大影响。

AprioriTid has the nice feature that it replaces a pass over the original dataset by a pass over the set ${\bar{C}}_{k}$ . Hence,AprioriTid is very effective in later passes when the size of ${\bar{C}}_{k}$ becomes small compared to the size of the database. Thus, we find that AprioriTid beats Apriori when its ${\bar{C}}_{k}$ sets can fit in memory and the distribution of the large itemsets has a long tail. When ${\bar{C}}_{k}$ doesn’t fit in memory,there is a jump in the execution time for AprioriTid, such as when going from ${0.75}\%$ to ${0.5}\%$ for datasets with transaction size 10 in Figure 4. In this region, Apriori starts beating AprioriTid.

先验事务标识（AprioriTid）具有一个很好的特性，即它用对集合${\bar{C}}_{k}$的一次遍历取代了对原始数据集的一次遍历。因此，当${\bar{C}}_{k}$的规模与数据库规模相比变小时，先验事务标识（AprioriTid）在后续遍历中非常有效。因此，我们发现，当先验事务标识（AprioriTid）的${\bar{C}}_{k}$集合能够放入内存，且大项集的分布具有长尾特征时，先验事务标识（AprioriTid）的性能优于先验算法（Apriori）。当${\bar{C}}_{k}$无法放入内存时，先验事务标识（AprioriTid）的执行时间会出现跳跃，例如在图4中，对于事务大小为10的数据集，从${0.75}\%$到${0.5}\%$时就会出现这种情况。在这个区域，先验算法（Apriori）的性能开始优于先验事务标识（AprioriTid）。

---

<!-- Footnote -->

${}^{2}$ The cost of external sorting in SETM can be reduced somewhat as follows. Before writing out entries in ${\bar{C}}_{k}$ to disk, we can sort them on itemsets using an internal sorting procedure, and write them as sorted runs. These sorted runs can then be merged to obtain support counts. However, given the poor performance of SETM, we do not expect this optimization to affect the algorithm choice.

${}^{2}$集合挖掘（SETM）中外部排序的成本可以通过以下方法有所降低。在将${\bar{C}}_{k}$中的条目写入磁盘之前，我们可以使用内部排序程序按项集对它们进行排序，并将它们作为有序运行段写入。然后可以合并这些有序运行段以获得支持度计数。然而，鉴于集合挖掘（SETM）的性能较差，我们预计这种优化不会影响算法的选择。

${}^{3}$ For SETM to use IDs,it would have to maintain two additional in-memory data structures: a hash table to find out whether a candidate has been generated previously, and a mapping from the IDs to candidates. However, this would destroy the set-oriented nature of the algorithm. Also, once we have the hash table which gives us the IDs of candidates, we might as well count them at the same time and avoid the two external sorts. We experimented with this variant of SETM and found that, while it did better than SETM, it still performed much worse than Apriori or AprioriTid.

${}^{3}$如果集合挖掘（SETM）要使用标识符（IDs），它必须维护两个额外的内存数据结构：一个哈希表，用于确定某个候选项集是否之前已经生成过；以及一个从标识符（IDs）到候选项集的映射。然而，这将破坏该算法的面向集合的特性。此外，一旦我们有了能给出候选项集标识符（IDs）的哈希表，我们不妨同时对它们进行计数，从而避免两次外部排序。我们对集合挖掘（SETM）的这种变体进行了实验，发现虽然它的性能比集合挖掘（SETM）好，但仍然比先验算法（Apriori）或先验事务标识（AprioriTid）差得多。

<!-- Footnote -->

---

### 3.6 Algorithm AprioriHybrid

### 3.6 先验混合算法（AprioriHybrid）

It is not necessary to use the same algorithm in all the passes over data. Figure 6 shows the execution times for Apriori and AprioriTid for different passes over the dataset T10.I4.D100K. In the earlier passes, Apriori does better than AprioriTid. However, AprioriTid beats Apriori in later passes. We observed similar relative behavior for the other datasets, the reason for which is as follows. Apriori and AprioriTid use the same candidate generation procedure and therefore count the same itemsets. In the later passes, the number of candidate itemsets reduces (see the size of ${C}_{k}$ for Apriori and AprioriTid in Figure 5). However, Apriori still examines every transaction in the database. On the other hand, rather than scanning the database, AprioriTid scans ${\bar{C}}_{k}$ for obtaining support counts,and the size of ${\bar{C}}_{k}$ has become smaller than the size of the database. When the ${\bar{C}}_{k}$ sets can fit in memory,we do not even incur the cost of writing them to disk.

在对数据的所有遍历中，并不一定非要使用相同的算法。图6显示了先验算法（Apriori）和先验事务标识（AprioriTid）在对数据集T10.I4.D100K进行不同遍历时的执行时间。在前期遍历中，先验算法（Apriori）的性能优于先验事务标识（AprioriTid）。然而，在后期遍历中，先验事务标识（AprioriTid）的性能优于先验算法（Apriori）。我们在其他数据集上也观察到了类似的相对性能表现，原因如下。先验算法（Apriori）和先验事务标识（AprioriTid）使用相同的候选项集生成过程，因此对相同的项集进行计数。在后期遍历中，候选项集的数量减少（见图5中先验算法（Apriori）和先验事务标识（AprioriTid）的${C}_{k}$的规模）。然而，先验算法（Apriori）仍然会检查数据库中的每个事务。另一方面，先验事务标识（AprioriTid）不是扫描数据库，而是扫描${\bar{C}}_{k}$来获取支持度计数，并且${\bar{C}}_{k}$的规模已经小于数据库的规模。当${\bar{C}}_{k}$集合能够放入内存时，我们甚至无需承担将它们写入磁盘的成本。

<!-- Media -->

<!-- figureText: 14 Apriori AprioriTid 12 10 8 6 2 3 -->

<img src="https://cdn.noedgeai.com/0195c900-f548-7e6e-93eb-62d54d470345_9.jpg?x=235&y=1191&w=618&h=516&r=0"/>

Figure 6: Per pass execution times of Apriori and AprioriTid (T10.I4.D100K, minsup $= {0.75}\%$ )

图6：先验算法（Apriori）和先验事务标识（AprioriTid）的每次遍历执行时间（T10.I4.D100K，最小支持度$= {0.75}\%$）

<!-- Media -->

Based on these observations, we can design a hybrid algorithm, which we call AprioriHybrid, that uses Apriori in the initial passes and switches to AprioriTid when it expects that the set ${\bar{C}}_{k}$ at the end of the pass will fit in memory. We use the following heuristic to estimate if ${\bar{C}}_{k}$ would fit in memory in the next pass. At the end of the current pass, we have the counts of the candidates in ${C}_{k}$ . From this,we estimate what the size of ${\bar{C}}_{k}$ would have been if it had been generated. This size,in words,is $\left( {\mathop{\sum }\limits_{{\text{candidates }c \in  {C}_{k}}}\operatorname{support}\left( c\right)  + }\right.$ number of transactions). If ${\bar{C}}_{k}$ in this pass was small enough to fit in memory, and there were fewer large candidates in the current pass than the previous pass, we switch to AprioriTid. The latter condition is added to avoid switching when ${\bar{C}}_{k}$ in the current pass fits in memory but ${\bar{C}}_{k}$ in the next pass may not.

基于这些观察结果，我们可以设计一种混合算法，我们称之为先验混合算法（AprioriHybrid），该算法在初始遍历中使用先验算法（Apriori），并在预计遍历结束时的集合${\bar{C}}_{k}$能够放入内存时切换到先验事务标识（AprioriTid）。我们使用以下启发式方法来估计下一次遍历中${\bar{C}}_{k}$是否能够放入内存。在当前遍历结束时，我们有${C}_{k}$中候选项集的计数。据此，我们估计如果生成了${\bar{C}}_{k}$，它的规模会是多少。用文字表述，这个规模是$\left( {\mathop{\sum }\limits_{{\text{candidates }c \in  {C}_{k}}}\operatorname{support}\left( c\right)  + }\right.$事务数量）。如果当前遍历中的${\bar{C}}_{k}$足够小，可以放入内存，并且当前遍历中的大候选项集比上一次遍历少，我们就切换到先验事务标识（AprioriTid）。添加后一个条件是为了避免在当前遍历中的${\bar{C}}_{k}$能够放入内存，但下一次遍历中的${\bar{C}}_{k}$可能无法放入内存时进行切换。

Switching from Apriori to AprioriTid does involve a cost. Assume that we decide to switch from Apriori to AprioriTid at the end of the $k$ th pass. In the $\left( {k + 1}\right)$ th pass,after finding the candidate itemsets contained in a transaction, we will also have to add their IDs to ${\bar{C}}_{k + 1}$ (see the description of AprioriTid in Section 2.2). Thus there is an extra cost incurred in this pass relative to just running Apriori. It is only in the $\left( {k + 2}\right)$ th pass that we actually start running AprioriTid. Thus,if there are no large $\left( {k + 1}\right)$ -itemsets, or no $\left( {k + 2}\right)$ -candidates,we will incur the cost of switching without getting any of the savings of using AprioriTid.

从Apriori算法切换到AprioriTid算法确实会产生一定成本。假设我们决定在第 $k$ 遍扫描结束时从Apriori算法切换到AprioriTid算法。在第 $\left( {k + 1}\right)$ 遍扫描中，在找到事务中包含的候选项集后，我们还必须将它们的ID添加到 ${\bar{C}}_{k + 1}$ 中（参见2.2节中对AprioriTid算法的描述）。因此，与仅运行Apriori算法相比，这一遍扫描会产生额外的成本。直到第 $\left( {k + 2}\right)$ 遍扫描，我们才真正开始运行AprioriTid算法。因此，如果没有大的 $\left( {k + 1}\right)$ -项集，或者没有 $\left( {k + 2}\right)$ -候选集，我们将承担切换的成本，却无法从使用AprioriTid算法中获得任何节省。

Figure 7 shows the performance of AprioriHybrid relative to Apriori and AprioriTid for three datasets. AprioriHybrid performs better than Apriori in almost all cases. For T10.I2.D100K with 1.5% support, AprioriHybrid does a little worse than Apriori since the pass in which the switch occurred was the last pass; AprioriHybrid thus incurred the cost of switching without realizing the benefits. In general, the advantage of AprioriHybrid over Apriori depends on how the size of the ${\bar{C}}_{k}$ set decline in the later passes. If ${\bar{C}}_{k}$ remains large until nearly the end and then has an abrupt drop, we will not gain much by using AprioriHybrid since we can use AprioriTid only for a short period of time after the switch. This is what happened with the T20.I6.D100K dataset. On the other hand, if there is a gradual decline in the size of ${\bar{C}}_{k}$ ,AprioriTid can be used for a while after the switch, and a significant improvement can be obtained in the execution time.

图7展示了AprioriHybrid算法相对于Apriori算法和AprioriTid算法在三个数据集上的性能表现。在几乎所有情况下，AprioriHybrid算法的性能都优于Apriori算法。对于支持度为1.5%的T10.I2.D100K数据集，AprioriHybrid算法的表现略逊于Apriori算法，因为切换发生在最后一遍扫描；因此，AprioriHybrid算法承担了切换的成本，却未实现相应的收益。一般来说，AprioriHybrid算法相对于Apriori算法的优势取决于 ${\bar{C}}_{k}$ 集的大小在后续扫描中的下降情况。如果 ${\bar{C}}_{k}$ 集的大小直到接近扫描结束时仍然很大，然后突然下降，那么使用AprioriHybrid算法并不会带来太多收益，因为切换后我们只能在短时间内使用AprioriTid算法。这就是T20.I6.D100K数据集所发生的情况。另一方面，如果 ${\bar{C}}_{k}$ 集的大小逐渐下降，切换后可以在一段时间内使用AprioriTid算法，从而显著缩短执行时间。

### 3.7 Scale-up Experiment

### 3.7 可扩展性实验

Figure 8 shows how AprioriHybrid scales up as the number of transactions is increased from 100,000 to 10 million transactions. We used the combinations (T5.I2), (T10.I4), and (T20.I6) for the average sizes of transactions and itemsets respectively. All other parameters were the same as for the data in Table 3. The sizes of these datasets for 10 million transactions were ${239}\mathrm{{MB}},{439}\mathrm{{MB}}$ and ${838}\mathrm{{MB}}$ respectively. The minimum support level was set to ${0.75}\%$ . The execution times are normalized with respect to the times for the 100,000 transaction datasets in the first graph and with respect to the 1 million transaction dataset in the second. As shown, the execution times scale quite linearly.

图8展示了随着事务数量从100,000增加到1000万，AprioriHybrid算法的可扩展性情况。我们分别使用组合 (T5.I2)、(T10.I4) 和 (T20.I6) 来表示事务和项集的平均大小。所有其他参数与表3中的数据相同。对于1000万条事务的这些数据集，其大小分别为 ${239}\mathrm{{MB}},{439}\mathrm{{MB}}$ 和 ${838}\mathrm{{MB}}$。最小支持度级别设置为 ${0.75}\%$。执行时间相对于第一个图中100,000条事务数据集的时间进行了归一化处理，相对于第二个图中100万条事务数据集的时间进行了归一化处理。如图所示，执行时间的增长具有相当好的线性关系。

<!-- Media -->

<!-- figureText: T10.I2.D100K AprioriTid. AprioriHybrid 0.75 0.5 0.33 0.25 Minimum Support Aprioritid Apriori AprioriHybrid 0.75 0.5 0.33 0.25 Minimum Support AprioriTid AprioriHybrid 0.75 0.5 0.33 0.25 Minimum Support 40 35 30 Time (sec) 25 20 15 10 2 1.5 T10.I4.D100K 55 50 45 40 35 Time (sec) 30 25 20 15 10 1.5 T20.I6.D100K 700 600 500 Time (sec) 400 300 200 100 2 1.5 -->

<img src="https://cdn.noedgeai.com/0195c900-f548-7e6e-93eb-62d54d470345_10.jpg?x=129&y=228&w=676&h=1681&r=0"/>

Figure 7: Execution times: AprioriHybrid

图7：执行时间：AprioriHybrid

<!-- figureText: 12 T20.16 T10.H TS-12-a- 500 750 1000 Number of Transactions (in ’000s) T20.16 T10.14 T5.12 5 7.5 10 Number of Transactions (in Millions) 10 Relative Time 4 0 100 250 14 12 10 Relative Time 8 2.5 -->

<img src="https://cdn.noedgeai.com/0195c900-f548-7e6e-93eb-62d54d470345_10.jpg?x=909&y=283&w=640&h=1046&r=0"/>

Figure 8: Number of transactions scale-up

图8：事务数量的可扩展性

<!-- Media -->

Next, we examined how AprioriHybrid scaled up with the number of items. We increased the number of items from 1000 to 10,000 for the three parameter settings T5.I2.D100K, T10.I4.D100K and T20.I6.D100K. All other parameters were the same as for the data in Table 3. We ran experiments for a minimum support at ${0.75}\%$ ,and obtained the results shown in Figure 9. The execution times decreased a little since the average support for an item decreased as we increased the number of items. This resulted in fewer large itemsets and, hence, faster execution times.

接下来，我们研究了AprioriHybrid算法随项数量的可扩展性情况。对于三种参数设置T5.I2.D100K、T10.I4.D100K和T20.I6.D100K，我们将项的数量从1000增加到10,000。所有其他参数与表3中的数据相同。我们针对最小支持度为 ${0.75}\%$ 进行了实验，并得到了图9所示的结果。由于随着项数量的增加，单个项的平均支持度降低，执行时间略有减少。这导致大项集的数量减少，从而加快了执行时间。

Finally, we investigated the scale-up as we increased the average transaction size. The aim of this experiment was to see how our data structures scaled with the transaction size, independent of other factors like the physical database size and the number of large itemsets. We kept the physical size of the database roughly constant by keeping the product of the average transaction size and the number of transactions constant. The number of transactions ranged from 200,000 for the database with an average transaction size of 5 to 20,000 for the database with an average transaction size 50 . Fixing the minimum support as a percentage would have led to large increases in the number of large itemsets as the transaction size increased, since the probability of a itemset being present in a transaction is roughly proportional to the transaction size. We therefore fixed the minimum support level in terms of the number of transactions. The results are shown in Figure 10. The numbers in the key (e.g. 500) refer to this minimum support. As shown, the execution times increase with the transaction size, but only gradually. The main reason for the increase was that in spite of setting the minimum support in terms of the number of transactions, the number of large itemsets increased with increasing transaction length. A secondary reason was that finding the candidates present in a transaction took a little longer time.

最后，我们研究了随着平均事务大小增加时的扩展性。该实验的目的是观察我们的数据结构如何随事务大小扩展，而不受物理数据库大小和大项集数量等其他因素的影响。我们通过保持平均事务大小和事务数量的乘积不变，使数据库的物理大小大致保持恒定。事务数量从平均事务大小为 5 的数据库的 200,000 个，到平均事务大小为 50 的数据库的 20,000 个不等。如果将最小支持度固定为百分比，那么随着事务大小的增加，大项集的数量会大幅增加，因为项集出现在事务中的概率大致与事务大小成正比。因此，我们根据事务数量来固定最小支持度级别。结果如图 10 所示。图例中的数字（例如 500）指的就是这个最小支持度。如图所示，执行时间随事务大小增加，但只是逐渐增加。增加的主要原因是，尽管根据事务数量设置了最小支持度，但大项集的数量仍随事务长度的增加而增加。次要原因是，查找事务中存在的候选集需要稍微长一点的时间。

<!-- Media -->

<!-- figureText: 40 T20.16 T10.14 T5.12 5000 7500 10000 Number of items 35 30 Time (sec) 25 20 15 10 5 0 1000 2500 -->

<img src="https://cdn.noedgeai.com/0195c900-f548-7e6e-93eb-62d54d470345_11.jpg?x=244&y=151&w=632&h=522&r=0"/>

Figure 9: Number of items scale-up

图 9：项数量的扩展性

<!-- Media -->

## 4 Conclusions and Future Work

## 4 结论与未来工作

We presented two new algorithms, Apriori and Apri-oriTid, for discovering all significant association rules between items in a large database of transactions. We compared these algorithms to the previously known algorithms, the AIS [4] and SETM [13] algorithms. We presented experimental results, showing that the proposed algorithms always outperform AIS and SETM. The performance gap increased with the problem size, and ranged from a factor of three for small problems to more than an order of magnitude for large problems.

我们提出了两种新算法，Apriori 和 Apri - oriTid，用于在大型事务数据库中发现项之间的所有重要关联规则。我们将这些算法与之前已知的算法，即 AIS [4] 和 SETM [13] 算法进行了比较。我们给出了实验结果，表明所提出的算法总是优于 AIS 和 SETM。性能差距随问题规模的增大而增大，从小问题的 3 倍到大型问题的一个数量级以上不等。

We showed how the best features of the two proposed algorithms can be combined into a hybrid algorithm, called AprioriHybrid, which then becomes the algorithm of choice for this problem. Scale-up experiments showed that AprioriHybrid scales linearly with the number of transactions. In addition, the execution time decreases a little as the number of items in the database increases. As the average transaction size increases (while keeping the database size constant), the execution time increases only gradually. These experiments demonstrate the feasibility of using AprioriHybrid in real applications involving very large databases.

我们展示了如何将这两种提出的算法的最佳特性组合成一种混合算法，称为 AprioriHybrid，它随后成为解决该问题的首选算法。扩展性实验表明，AprioriHybrid 与事务数量呈线性扩展。此外，随着数据库中项的数量增加，执行时间会略有减少。随着平均事务大小的增加（同时保持数据库大小不变），执行时间仅逐渐增加。这些实验证明了在涉及非常大型数据库的实际应用中使用 AprioriHybrid 的可行性。

<!-- Media -->

<!-- figureText: 30 500 ...... 1000 30 40 50 Transaction Size 20 Time (sec) 15 10 5 10 20 -->

<img src="https://cdn.noedgeai.com/0195c900-f548-7e6e-93eb-62d54d470345_11.jpg?x=983&y=151&w=619&h=520&r=0"/>

Figure 10: Transaction size scale-up

图 10：事务大小的扩展性

<!-- Media -->

The algorithms presented in this paper have been implemented on several data repositories, including the AIX file system, DB2/MVS, and DB2/6000. We have also tested these algorithms against real customer data, the details of which can be found in [5]. In the future, we plan to extend this work along the following dimensions:

本文提出的算法已在多个数据存储库上实现，包括 AIX 文件系统、DB2/MVS 和 DB2/6000。我们还针对实际客户数据对这些算法进行了测试，具体细节可在 [5] 中找到。未来，我们计划从以下几个方面扩展这项工作：

- Multiple taxonomies (is- $a$ hierarchies) over items are often available. An example of such a hierarchy is that a dish washer is a kitchen appliance is a heavy electric appliance, etc. We would like to be able to find association rules that use such hierarchies.

- 项上的多个分类法（is - $a$ 层次结构）通常是可用的。这种层次结构的一个例子是，洗碗机是厨房电器，厨房电器是大型电器，等等。我们希望能够找到使用这种层次结构的关联规则。

- We did not consider the quantities of the items bought in a transaction, which are useful for some applications. Finding such rules needs further work.

- 我们没有考虑事务中购买项的数量，这在某些应用中是有用的。寻找此类规则还需要进一步的工作。

The work reported in this paper has been done in the context of the Quest project at the IBM Almaden Research Center. In Quest, we are exploring the various aspects of the database mining problem. Besides the problem of discovering association rules, some other problems that we have looked into include the enhancement of the database capability with classification queries [2] and similarity queries over time sequences [1]. We believe that database mining is an important new application area for databases, combining commercial interest with intriguing research questions.

本文报告的工作是在 IBM 阿尔马登研究中心的 Quest 项目背景下完成的。在 Quest 项目中，我们正在探索数据库挖掘问题的各个方面。除了发现关联规则的问题外，我们还研究了其他一些问题，包括通过分类查询 [2] 增强数据库功能以及对时间序列进行相似性查询 [1]。我们认为，数据库挖掘是数据库的一个重要新应用领域，它将商业利益与有趣的研究问题相结合。

Acknowledgment We wish to thank Mike Carey for his insightful comments and suggestions.

致谢 我们要感谢迈克·凯里（Mike Carey）提出的深刻见解和建议。

## References

## 参考文献

[1] R. Agrawal, C. Faloutsos, and A. Swami. Efficient similarity search in sequence databases. In Proc. of the Fourth International Conference on Foundations of Data Organization and Algorithms, Chicago, October 1993.

[2] R. Agrawal, S. Ghosh, T. Imielinski, B. Iyer, and A. Swami. An interval classifier for database mining applications. In Proc. of the VLDB Conference, pages 560-573, Vancouver, British Columbia, Canada, 1992.

[3] R. Agrawal, T. Imielinski, and A. Swami. Database mining: A performance perspective. IEEE Transactions on Knowledge and Data Engineering, 5(6):914-925, December 1993. Special Issue on Learning and Discovery in Knowledge-Based Databases.

[4] R. Agrawal, T. Imielinski, and A. Swami. Mining association rules between sets of items in large databases. In Proc. of the ACM SIGMOD Conference on Management of Data, Washington, D.C., May 1993.

[5] R. Agrawal and R. Srikant. Fast algorithms for mining association rules in large databases. Research Report RJ 9839, IBM Almaden Research Center, San Jose, California, June 1994.

[6] D. S. Associates. The new direct marketing. Business One Irwin, Illinois, 1990.

[7] R. Brachman et al. Integrated support for data archeology. In AAAI-93 Workshop on Knowledge Discovery in Databases, July 1993.

[8] L. Breiman, J. H. Friedman, R. A. Olshen, and C. J. Stone. Classification and Regression Trees. Wadsworth, Belmont, 1984.

[9] P. Cheeseman et al. Autoclass: A bayesian classification system. In 5th Int'l Conf. on Machine Learning. Morgan Kaufman, June 1988.

[10] D. H. Fisher. Knowledge acquisition via incre-

[10] D. H. 费舍尔（Fisher）。通过增量式方法获取知识

mental conceptual clustering. Machine Learning, $2\left( 2\right) ,{1987}$ .

[11] J. Han, Y. Cai, and N. Cercone. Knowledge discovery in databases: An attribute oriented approach. In Proc. of the VLDB Conference, pages 547-559, Vancouver, British Columbia, Canada, 1992.

[12] M. Holsheimer and A. Siebes. Data mining: The search for knowledge in databases. Technical Report CS-R9406, CWI, Netherlands, 1994.

[13] M. Houtsma and A. Swami. Set-oriented mining of association rules. Research Report RJ 9567, IBM Almaden Research Center, San Jose, California, October 1993.

[14] R. Krishnamurthy and T. Imielinski. Practitioner problems in need of database research: Research directions in knowledge discovery. SIG-MOD RECORD, 20(3):76-78, September 1991.

[15] P. Langley, H. Simon, G. Bradshaw, and J. Zytkow. Scientific Discovery: Computational Explorations of the Creative Process. MIT Press, 1987.

[16] H. Mannila and K.-J. Raiha. Dependency inference. In Proc. of the VLDB Conference, pages 155-158, Brighton, England, 1987.

[17] H. Mannila, H. Toivonen, and A. I. Verkamo. Efficient algorithms for discovering association rules. In KDD-94: AAAI Workshop on Knowledge Discovery in Databases, July 1994.

[18] S. Muggleton and C. Feng. Efficient induction of logic programs. In S. Muggleton, editor, Inductive Logic Programming. Academic Press, 1992.

[19] J. Pearl. Probabilistic reasoning in intelligent systems: Networks of plausible inference, 1992.

[20] G. Piatestsky-Shapiro. Discovery, analysis, and presentation of strong rules. In G. Piatestsky-Shapiro, editor, Knowledge Discovery in Databases. AAAI/MIT Press, 1991.

[21] G. Piatestsky-Shapiro, editor. Knowledge Discovery in Databases. AAAI/MIT Press, 1991.

[22] J. R. Quinlan. C4.5: Programs for Machine Learning. Morgan Kaufman, 1993.