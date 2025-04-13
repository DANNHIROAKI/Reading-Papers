# iDistance: An Adaptive ${\mathrm{B}}^{ + }$ -Tree Based Indexing Method for Nearest Neighbor Search

# iDistance：一种基于自适应${\mathrm{B}}^{ + }$树的最近邻搜索索引方法

H. V. JAGADISH

H. V. 贾加迪什

University of Michigan

密歇根大学

BENG CHIN OOI and KIAN-LEE TAN

ooi 本钦和陈健利

National University of Singapore

新加坡国立大学

CUI YU

Monmouth University

蒙茅斯大学

and

和

RUI ZHANG

National University of Singapore

新加坡国立大学

In this article,we present an efficient ${\mathrm{B}}^{ + }$ -tree based indexing method,called iDistance,for K-nearest neighbor (KNN) search in a high-dimensional metric space. iDistance partitions the data based on a space- or data-partitioning strategy, and selects a reference point for each partition. The data points in each partition are transformed into a single dimensional value based on their similarity with respect to the reference point. This allows the points to be indexed using a ${\mathrm{B}}^{ + }$ -tree structure and KNN search to be performed using one-dimensional range search. The choice of partition and reference points adapts the index structure to the data distribution.

在本文中，我们提出了一种高效的基于${\mathrm{B}}^{ + }$树的索引方法，称为 iDistance，用于在高维度量空间中进行 K 近邻（KNN）搜索。iDistance 根据空间或数据划分策略对数据进行划分，并为每个分区选择一个参考点。每个分区中的数据点根据它们与参考点的相似度转换为一维值。这使得可以使用${\mathrm{B}}^{ + }$树结构对这些点进行索引，并使用一维范围搜索执行 KNN 搜索。分区和参考点的选择使索引结构适应数据分布。

We conducted extensive experiments to evaluate the iDistance technique, and report results demonstrating its effectiveness. We also present a cost model for iDistance KNN search, which can be exploited in query optimization.

我们进行了广泛的实验来评估 iDistance 技术，并报告了证明其有效性的结果。我们还提出了一个用于 iDistance KNN 搜索的成本模型，该模型可用于查询优化。

Categories and Subject Descriptors: H.3.1 [Information Storage and Retrieval]: Content Analysis and Indexing

类别和主题描述符：H.3.1 [信息存储与检索]：内容分析与索引

General Terms: Algorithms, Performance

通用术语：算法、性能

Additional Key Words and Phrases: Indexing, KNN, nearest neighbor queries

其他关键词和短语：索引、KNN、最近邻查询

---

<!-- Footnote -->

Authors' addresses: H. V. Jagadish, Department of Computer Science, University of Michigan, 1301 Beal Avenue, Ann Arbor, MI 48109; email: jag@eecs.umich.edu; B. C. Ooi, K.-L. Tan, and R. Zhang, Department of Computer Science, National University of Singapore Kent Ridge, Singapore 117543; email: \{ooibc,tankl,zhangru1\}@comp.nus.edu.sg; C. Yu, Department of Computer Science, Monmouth University, 400 Cedar Avenue, West Long Branch, NJ 07764-1898; email: cyu@ monmouth.edu.

作者地址：H. V. 贾加迪什，密歇根大学计算机科学系，比尔大道 1301 号，安阿伯，密歇根州 48109；电子邮件：jag@eecs.umich.edu；B. C. ooi、K.-L. 陈和张锐，新加坡国立大学计算机科学系，肯特岭，新加坡 117543；电子邮件：{ooibc,tankl,zhangru1}@comp.nus.edu.sg；于晨，蒙茅斯大学计算机科学系，雪松大道 400 号，西长滩，新泽西州 07764 - 1898；电子邮件：cyu@monmouth.edu。

Permission to make digital or hard copies of part or all of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or direct commercial advantage and that copies show this notice on the first page or initial screen of a display along with the full citation. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, to republish, to post on servers, to redistribute to lists, or to use any component of this work in other works requires prior specific permission and/or a fee. Permissions may be requested from Publications Dept., ACM, Inc., 1515 Broadway, New York, NY 10036 USA, fax: +1 (212) 869-0481, or permissions@acm.org. © 2005 ACM 0362-5915/05/0600-0364 \$5.00

允许个人或课堂使用本作品的部分或全部制作数字或硬拷贝，无需付费，但前提是这些拷贝不得用于盈利或直接商业利益，并且在显示的第一页或初始屏幕上要显示此通知以及完整的引用信息。必须尊重本作品中除 ACM 之外其他所有者的版权。允许进行带引用的摘要。否则，复制、重新发布、发布到服务器、分发给列表或在其他作品中使用本作品的任何组件都需要事先获得特定许可和/或支付费用。许可申请可向美国纽约百老汇 1515 号 ACM 公司出版部提出，传真：+1 (212) 869 - 0481，或发送电子邮件至 permissions@acm.org。© 2005 ACM 0362 - 5915/05/0600 - 0364 5 美元

<!-- Footnote -->

---

## 1. INTRODUCTION

## 1. 引言

Many emerging database applications such as image, time series, and scientific databases, manipulate high-dimensional data. In these applications, one of the most frequently used and yet expensive operations is to find objects in the high-dimensional database that are similar to a given query object. Nearest neighbor search is a central requirement in such cases.

许多新兴的数据库应用，如图像、时间序列和科学数据库，都需要处理高维数据。在这些应用中，最常用但代价高昂的操作之一是在高维数据库中查找与给定查询对象相似的对象。最近邻搜索在这种情况下是一项核心需求。

There is a long stream of research on solving the nearest neighbor search problem, and a large number of multidimensional indexes have been developed for this purpose. Existing multidimensional indexes such as R-trees [Guttman 1984] have been shown to be inefficient even for supporting range queries in high-dimensional databases; however, they form the basis for indexes designed for high-dimensional databases [Katamaya and Satoh 1997; White and Jain 1996]. To reduce the effect of high dimensionality, use of larger fanouts [Berchtold et al. 1996; Sakurai et al. 2000], dimensionality reduction techniques [Chakrabarti and Mehrotra 2000, 1999], and filter-and-refine methods [Berchtold et al. 1998b; Weber et al. 1998] have been proposed. Indexes have also been specifically designed to facilitate metric based query processing [Bozkaya and Ozsoyoglu 1997; Ciaccia et al. 1997; Traina et al. 2000; Filho et al. 2001]. However, linear scan remains an efficient search strategy for similarity search [Beyer et al. 1999]. This is because there is a tendency for data points to be nearly equidistant to query points in a high-dimensional space. While linear scan is effective in terms of sequential read, every point incurs expensive distance computation, when used for the nearest neighbor problem. For quick response to queries, with some tolerance for errors (i.e., answers may not necessarily be the nearest neighbors), approximate nearest neighbor (NN) search indexes such as the P-Sphere tree [Goldstein and Ramakrishnan 2000] have been proposed. The P-Sphere tree works well on static databases and provides answers with assigned accuracy. It achieves its efficiency by duplicating data points in data clusters based on a sample query set. Generally, most of these structures are not adaptive to data distributions. Consequently, they tend to perform well for some datasets and poorly for others.

关于解决最近邻搜索问题已有大量研究，并且为此开发了大量的多维索引。现有的多维索引，如R树[Guttman 1984]，即使在支持高维数据库中的范围查询时也被证明效率低下；然而，它们构成了为高维数据库设计的索引的基础[Katamaya和Satoh 1997；White和Jain 1996]。为了减少高维性的影响，有人提出使用更大的扇出[Berchtold等人1996；Sakurai等人2000]、降维技术[Chakrabarti和Mehrotra 2000，1999]以及过滤 - 细化方法[Berchtold等人1998b；Weber等人1998]。还专门设计了便于基于度量的查询处理的索引[Bozkaya和Ozsoyoglu 1997；Ciaccia等人1997；Traina等人2000；Filho等人2001]。然而，线性扫描仍然是相似性搜索的一种有效搜索策略[Beyer等人1999]。这是因为在高维空间中，数据点与查询点的距离往往近乎相等。虽然线性扫描在顺序读取方面很有效，但在用于最近邻问题时，每个点都需要进行代价高昂的距离计算。为了快速响应查询，同时允许一定的误差（即答案不一定是最近邻），有人提出了近似最近邻（NN）搜索索引，如P - 球树[Goldstein和Ramakrishnan 2000]。P - 球树在静态数据库上表现良好，并能以指定的精度提供答案。它通过基于样本查询集复制数据簇中的数据点来实现高效性。一般来说，这些结构大多不能适应数据分布。因此，它们在某些数据集上表现良好，而在其他数据集上表现不佳。

In this article, we present iDistance, a new technique for KNN search that can be adapted to different data distributions. In our technique, we first partition the data and define a reference point for each partition. Then we index the distance of each data point to the reference point of its partition. Since this distance is a simple scalar, with a small mapping effort to keep partitions distinct,a classical ${\mathrm{B}}^{ + }$ -tree can be used to index this distance. As such,it is easy to graft our technique on top of an existing commercial relational database. This is important as most commercial DBMSs today do not support indexes beyond the ${\mathrm{B}}^{ + }$ -tree and the $\mathrm{R}$ -tree (or one of its variants). The effectiveness of iDistance depends on how the data are partitioned, and how reference points are selected.

在本文中，我们提出了iDistance，这是一种用于K近邻（KNN）搜索的新技术，它可以适应不同的数据分布。在我们的技术中，我们首先对数据进行分区，并为每个分区定义一个参考点。然后，我们对每个数据点到其所在分区参考点的距离进行索引。由于这个距离是一个简单的标量，只需进行少量的映射操作就能区分不同的分区，因此可以使用经典的${\mathrm{B}}^{ + }$ - 树来对这个距离进行索引。这样，很容易将我们的技术嫁接到现有的商业关系数据库之上。这一点很重要，因为如今大多数商业数据库管理系统（DBMS）除了支持${\mathrm{B}}^{ + }$ - 树和$\mathrm{R}$ - 树（或其变体之一）之外，不支持其他索引。iDistance的有效性取决于数据的分区方式以及参考点的选择方式。

For a KNN query centered at $q$ ,a range query with radius $r$ is issued. The iDistance KNN search algorithm searches the index from the query point outwards, and for each partition that intersects with the query sphere, a range query is resulted. If the algorithm finds $K$ elements that are closer than $r$ from $q$ at the end of the search,the algorithm terminates. Otherwise,it extends the search radius by ${\Delta r}$ ,and the search continues to examine the unexplored region in the partitions that intersects with the query sphere. The process is repeated till the stopping condition is satisfied. To facilitate efficient KNN search, we propose partitioning and reference point selection strategies as well as a cost model to estimate the page access cost of iDistance KNN searching.

对于以$q$为中心的KNN查询，会发出一个半径为$r$的范围查询。iDistance KNN搜索算法从查询点向外搜索索引，对于与查询球相交的每个分区，都会产生一个范围查询。如果算法在搜索结束时找到$K$个比$r$更接近$q$的元素，则算法终止。否则，它将搜索半径扩展${\Delta r}$，并继续搜索与查询球相交的分区中未探索的区域。重复这个过程，直到满足停止条件。为了便于高效的KNN搜索，我们提出了分区和参考点选择策略，以及一个成本模型来估计iDistance KNN搜索的页面访问成本。

This article is an extended version of our earlier paper [Yu et al. 2001]. There, we present the basic iDistance method. Here, we have extended it substantially to include a more detailed discussion of the technique and algorithms, a cost model, and comprehensive experimental studies. In this article, we conducted a whole new set of experiments using different indexes for comparison. In particular, we compare iDistance against sequential scan, the M-tree [Ciaccia et al. 1997], the Omni-sequential [Filho et al. 2001] and the bd-tree structure [Arya et al. 1994] on both synthetic and real datasets. While the M-tree and the Omni-sequential schemes are disk-based structures, the bd-tree is a main memory based index. Our results showed that iDistance is superior to these techniques for a wide range of experimental setups.

本文是我们早期论文[Yu等人，2001年]的扩展版本。在那篇论文中，我们介绍了基本的iDistance方法。在此，我们对其进行了大幅扩展，包括对该技术和算法进行更详细的讨论、建立成本模型以及开展全面的实验研究。在本文中，我们使用不同的索引进行了全新的一组实验以作比较。特别是，我们在合成数据集和真实数据集上，将iDistance与顺序扫描、M树[Ciaccia等人，1997年]、全顺序索引[Filho等人，2001年]和bd树结构[Arya等人，1994年]进行了比较。虽然M树和全顺序方案是基于磁盘的结构，但bd树是基于主内存的索引。我们的结果表明，在广泛的实验设置中，iDistance优于这些技术。

The rest of this article is organized as follows. In the next section, we present the background for metric-based KNN processing, and review some related work. In Section 3, we present the iDistance indexing method and KNN search algorithm, and in Section 4, its space- and data-based partitioning strategies. In Section 5, we present the cost model for estimating the page access cost of iDistance KNN search. We present the performance studies in Section 6, and finally, we conclude in Section 7.

本文的其余部分组织如下。在下一节中，我们介绍基于度量的K近邻（KNN）处理的背景，并回顾一些相关工作。在第3节中，我们介绍iDistance索引方法和KNN搜索算法，在第4节中介绍其基于空间和数据的分区策略。在第5节中，我们介绍用于估计iDistance KNN搜索页面访问成本的成本模型。我们在第6节中介绍性能研究，最后，在第7节中得出结论。

## 2. BACKGROUND AND RELATED WORK

## 2. 背景与相关工作

In this section, we provide the background for metric-based KNN processing, and review related work.

在本节中，我们提供基于度量的KNN处理的背景，并回顾相关工作。

### 2.1 KNN Query Processing

### 2.1 KNN查询处理

In our discussion,we assume that ${DB}$ is a set of points in a $d$ -dimensional data space. A $K$ -nearest neighbor query finds the $K$ objects in the database closest in distance to a given query object. More formally, the KNN problem can be defined as follows:

在我们的讨论中，我们假设${DB}$是$d$维数据空间中的一组点。一个$K$近邻查询会在数据库中找到与给定查询对象距离最近的$K$个对象。更正式地说，KNN问题可以定义如下：

Given a set of points ${DB}$ in a $d$ -dimensional space ${DS}$ ,and a query point $q \in  {DS}$ ,find a set $S$ that contains $K$ points in ${DB}$ such that,for any $p \in  S$ and for any ${p}^{\prime } \in  {DB} - S$ ,dist $\left( {q,p}\right)  < \operatorname{dist}\left( {q,{p}^{\prime }}\right)$ .

给定$d$维空间${DS}$中的一组点${DB}$，以及一个查询点$q \in  {DS}$，找到一个集合$S$，该集合包含${DB}$中的$K$个点，使得对于任何$p \in  S$和任何${p}^{\prime } \in  {DB} - S$，有dist $\left( {q,p}\right)  < \operatorname{dist}\left( {q,{p}^{\prime }}\right)$。

Table I describes the notation used in this article.

表I描述了本文中使用的符号。

To search for the $K$ nearest neighbors of a query point $q$ ,the distance of the $K$ th nearest neighbor to $q$ defines the minimum radius required for retrieving the complete answer set. Unfortunately, such a distance cannot be predetermined with 100% accuracy. Hence, an iterative approach can be employed (see Figure 1). The search starts with a query sphere about $q$ ,with a small initial radius, which can be set according to historical records. We maintain a candidate answer set that contains points that could be the $K$ nearest neighbors of $q$ . Then the query sphere is enlarged step by step and the candidate answer set

为了搜索查询点$q$的$K$个最近邻，第$K$个最近邻到$q$的距离定义了检索完整答案集所需的最小半径。不幸的是，这样的距离无法100%准确地预先确定。因此，可以采用迭代方法（见图1）。搜索从以$q$为中心的查询球开始，初始半径较小，该半径可以根据历史记录进行设置。我们维护一个候选答案集，其中包含可能是$q$的$K$个最近邻的点。然后，查询球逐步扩大，候选答案集

<!-- Media -->

Table I. Notation

表I. 符号

<table><tr><td>Notation</td><td>Meaning</td></tr><tr><td>${C}_{eff}$</td><td>Average number of points stored in a page</td></tr><tr><td>$d$</td><td>Dimensionality of the data space</td></tr><tr><td>${DB}$</td><td>The dataset</td></tr><tr><td>${DS}$</td><td>The data space</td></tr><tr><td>$m$</td><td>Number of reference points</td></tr><tr><td>$K$</td><td>Number of nearest neighbor points required by the query</td></tr><tr><td>$p$</td><td>A data point</td></tr><tr><td>$q$</td><td>A query point</td></tr><tr><td>$S$</td><td>The set containing $K$ NNs</td></tr><tr><td>$r$</td><td>Radius of a sphere</td></tr><tr><td>${\mathit{{dist}}}_{ - }{\mathit{{max}}}_{i}$</td><td>Maximum radius of partition ${P}_{i}$</td></tr><tr><td>${O}_{i}$</td><td>The $i$ th reference point</td></tr><tr><td>${P}_{i}$</td><td>The $i$ th partition</td></tr><tr><td>$\operatorname{dist}\left( {{p}_{1},{p}_{2}}\right)$</td><td>Metric function returns the distance between points</td></tr><tr><td/><td>${p}_{1}$ and ${p}_{2}$</td></tr><tr><td>querydist(q)</td><td>Query radius of $q$</td></tr><tr><td>sphere(q,r)</td><td>Sphere of radius $r$ and center $q$</td></tr><tr><td>$\mathit{{furthest}}\left( {S,q}\right)$</td><td>Function returns the object in $\mathrm{S}$ furthest in distance from $q$</td></tr></table>

<table><tbody><tr><td>符号表示</td><td>含义</td></tr><tr><td>${C}_{eff}$</td><td>存储在一个页面中的平均点数</td></tr><tr><td>$d$</td><td>数据空间的维度</td></tr><tr><td>${DB}$</td><td>数据集</td></tr><tr><td>${DS}$</td><td>数据空间</td></tr><tr><td>$m$</td><td>参考点的数量</td></tr><tr><td>$K$</td><td>查询所需的最近邻点的数量</td></tr><tr><td>$p$</td><td>一个数据点</td></tr><tr><td>$q$</td><td>一个查询点</td></tr><tr><td>$S$</td><td>包含$K$个最近邻点的集合</td></tr><tr><td>$r$</td><td>球体的半径</td></tr><tr><td>${\mathit{{dist}}}_{ - }{\mathit{{max}}}_{i}$</td><td>分区${P}_{i}$的最大半径</td></tr><tr><td>${O}_{i}$</td><td>第$i$个参考点</td></tr><tr><td>${P}_{i}$</td><td>第$i$个分区</td></tr><tr><td>$\operatorname{dist}\left( {{p}_{1},{p}_{2}}\right)$</td><td>度量函数返回点之间的距离</td></tr><tr><td></td><td>${p}_{1}$和${p}_{2}$</td></tr><tr><td>查询距离（q）</td><td>$q$的查询半径</td></tr><tr><td>球体（q，r）</td><td>半径为$r$、中心为$q$的球体</td></tr><tr><td>$\mathit{{furthest}}\left( {S,q}\right)$</td><td>函数返回$\mathrm{S}$中与$q$距离最远的对象</td></tr></tbody></table>

<!-- Media -->

KNN Basic Search Algorithm start with a small search sphere centered at query point search and check all partitions intersecting the current query space if $K$ nearest neighbors are found exit; else 6. enlarge search sphere; 7. goto 2 ; end KNN;

KNN基本搜索算法：从以查询点为中心的小搜索球开始搜索，检查与当前查询空间相交的所有分区；如果找到$K$个最近邻，则退出；否则，6. 扩大搜索球；7. 转到步骤2；KNN算法结束。

Fig. 1. Basic KNN algorithm. is updated accordingly until we can make sure that the $K$ candidate answers are the true $K$ nearest neighbors of $q$ .

图1. 基本KNN算法。相应地更新，直到我们可以确定$K$个候选答案是$q$的真正$K$个最近邻。

### 2.2 Related Work

### 2.2 相关工作

Many multi-dimensional structures have been proposed in the literature, including various KNN algorithms [Böhm et al. 2001]. Here, we briefly describe a few relevant methods.

文献中提出了许多多维结构，包括各种KNN算法[Böhm等人，2001年]。在这里，我们简要描述几种相关方法。

In Weber et al. [1998], the authors describe a simple vector approximation scheme,called VA-file. The VA-file divides the data space into ${2}^{b}$ rectangular cells where $b$ denotes a user specified number of bits. The scheme allocates a unique bit-string of length $b$ for each cell,and approximates data points that fall into a cell by that bit-string. The VA-file itself is simply an array of these compact, geometric approximations. Nearest neighbor searches are performed by scanning the entire approximation file, and by excluding the vast majority of vectors from the search (filtering step) based only on these approximations. After the filtering step, a small set of candidates remains. These candidates are then visited and the actual distances to the query point $q$ are determined. VA-file reduces the number of disk accesses, but it incurs higher computational cost to decode the bit-string, compute all the lower and some upper bounds on the distance to the query point, and determine the actual distances of candidate points. Another problem with the VA-file is that it works well for uniform data, but for skewed data, the pruning effect of the approximation vectors becomes very bad. The IQ-tree [Berchtold et al. 2000] extends the notion of the VA-file to use a tree structure where appropriate, and the bit-encoded file structure where appropriate. It inherits many of the benefits and drawbacks of the VA-file discussed above and the M-tree discussed next.

在Weber等人[1998年]的研究中，作者描述了一种简单的向量近似方案，称为VA文件（VA-file）。VA文件将数据空间划分为${2}^{b}$个矩形单元，其中$b$表示用户指定的位数。该方案为每个单元分配一个长度为$b$的唯一位串，并通过该位串近似落入该单元的数据点。VA文件本身只是这些紧凑的几何近似的数组。最近邻搜索通过扫描整个近似文件来执行，并仅基于这些近似从搜索中排除绝大多数向量（过滤步骤）。过滤步骤之后，会留下一小部分候选对象。然后访问这些候选对象，并确定它们到查询点$q$的实际距离。VA文件减少了磁盘访问次数，但解码位串、计算到查询点距离的所有下界和一些上界以及确定候选点的实际距离会产生较高的计算成本。VA文件的另一个问题是，它对均匀数据效果很好，但对于偏斜数据，近似向量的剪枝效果会变得非常差。IQ树（IQ-tree）[Berchtold等人，2000年]将VA文件的概念扩展为在适当的地方使用树结构，在适当的地方使用位编码文件结构。它继承了上述VA文件和接下来要讨论的M树的许多优点和缺点。

In Ciaccia et al. [1997], the authors proposed the height-balanced M-tree to organize and search large datasets from a generic metric space, where object proximity is only defined by a distance function satisfying the positivity, symmetry, and triangle inequality postulates. In an M-tree, leaf nodes store all indexed (database) objects, represented by their keys or features, whereas internal nodes store the routing objects. For each routing object ${O}_{r}$ ,there is an associated pointer,denoted ptr $\left( {\mathrm{T}\left( {O}_{r}\right) }\right)$ ,that references the root of a sub-tree, $\mathrm{T}\left( {O}_{r}\right)$ ,called the covering tree of ${O}_{r}$ . All objects in the covering tree of ${O}_{r}$ , are within the distance $\mathrm{r}\left( {O}_{r}\right)$ from ${O}_{r},\mathrm{r}\left( {O}_{r}\right)  > 0$ ,which is called the covering radius of ${O}_{r}$ . Finally,a routing object ${O}_{r}$ ,is associated with a distance to $\mathrm{P}\left( {O}_{r}\right)$ , its parent object, that is, the routing object that references the node where the ${O}_{r}$ entry is stored. Obviously,this distance is not defined for entries in the root of the M-tree. An entry for a database object ${O}_{j}$ in a leaf node is quite similar to that of a routing object, but no covering radius is needed. The strength of M-tree lies in maintaining the pre-computed distance in the index structure. However, the node utilization of the M-tree tends to be low due to its splitting strategy.

在Ciaccia等人[1997年]的研究中，作者提出了高度平衡的M树（M-tree），用于组织和搜索来自通用度量空间的大型数据集，其中对象接近度仅由满足正性、对称性和三角不等式假设的距离函数定义。在M树中，叶节点存储所有索引（数据库）对象，由它们的键或特征表示，而内部节点存储路由对象。对于每个路由对象${O}_{r}$，有一个关联的指针，记为ptr $\left( {\mathrm{T}\left( {O}_{r}\right) }\right)$，它引用一个子树$\mathrm{T}\left( {O}_{r}\right)$的根，称为${O}_{r}$的覆盖树。${O}_{r}$的覆盖树中的所有对象与${O}_{r},\mathrm{r}\left( {O}_{r}\right)  > 0$的距离在$\mathrm{r}\left( {O}_{r}\right)$以内，这称为${O}_{r}$的覆盖半径。最后，路由对象${O}_{r}$与它的父对象$\mathrm{P}\left( {O}_{r}\right)$有一个关联的距离，即引用存储${O}_{r}$条目的节点的路由对象。显然，M树根节点中的条目没有定义这个距离。叶节点中数据库对象${O}_{j}$的条目与路由对象的条目非常相似，但不需要覆盖半径。M树的优势在于在索引结构中维护预先计算的距离。然而，由于其分裂策略，M树的节点利用率往往较低。

Omni-concept was proposed in Filho et al. [2001]. The scheme chooses a number of objects from a database as global 'foci' and gauges all other objects based on their distances to each focus. If there are $l$ foci,each object will have $l$ distances to all the foci. These distances are the Omni-coordinates of the object. The Omni-concept is applied in the case where the correlation behaviors of the database are known beforehand and the intrinsic dimensionality $\left( {d}_{2}\right)$ is smaller than the embedded dimensionality $d$ of the database. A good number of foci is $\left\lceil  {d}_{2}\right\rceil   + 1$ or $\left\lceil  {d}_{2}\right\rceil   \times  2 + 1$ ,and they can either be selected or efficiently generated. Omni-trees can be built on top of different indexes such as the ${\mathrm{B}}^{ + }$ -tree and the $\mathrm{R}$ - tree. Omni B-trees used $l{\mathrm{\;B}}^{ + }$ -trees to index the Omni-coordinates of the objects. When a similarity range query is conducted,on each ${\mathrm{B}}^{ + }$ -tree,a set of candidate objects is obtained and intersection of all the $l$ candidate sets will be checked for the final answer. For the KNN query, the query radius is estimated by some selectivity estimation formulas. The Omni-concept improves the performance of similarity search by reducing the number of distance calculations during search operation. However, multiple sets of ordinates for each point increases the page access cost, and searching multiple B-trees (or R-trees) also increases CPU time. Finally,the intersection of the $l$ candidate sets incurs additional cost. In iDistance,only one set of ordinates is used and also only one ${\mathrm{B}}^{ + }$ -tree is used to index them, therefore iDistance has less page accesses while still reducing the distance computation. Besides, the choice of reference points in iDistance is quite different from the choice of foci bases in Omni-family techniques.

全概念（Omni-concept）由菲略（Filho）等人于2001年提出。该方案从数据库中选取若干对象作为全局“焦点”，并根据其他所有对象到每个焦点的距离来衡量这些对象。如果有$l$个焦点，那么每个对象到所有焦点都会有$l$个距离。这些距离就是该对象的全坐标（Omni-coordinates）。全概念适用于事先已知数据库的关联行为，且数据库的内在维数$\left( {d}_{2}\right)$小于嵌入维数$d$的情况。合适的焦点数量为$\left\lceil  {d}_{2}\right\rceil   + 1$或$\left\lceil  {d}_{2}\right\rceil   \times  2 + 1$，这些焦点既可以被选择，也可以被高效生成。全树（Omni-trees）可以构建在不同的索引之上，例如${\mathrm{B}}^{ + }$树和$\mathrm{R}$树。全B树（Omni B-trees）使用$l{\mathrm{\;B}}^{ + }$树来对对象的全坐标进行索引。当进行相似范围查询时，在每棵${\mathrm{B}}^{ + }$树上都会得到一组候选对象，然后会检查所有$l$个候选集的交集以得到最终答案。对于最近邻（KNN）查询，查询半径通过一些选择性估计公式来估算。全概念通过减少搜索操作期间的距离计算次数来提高相似搜索的性能。然而，每个点的多组纵坐标会增加页面访问成本，搜索多棵B树（或R树）也会增加CPU时间。最后，$l$个候选集的交集会产生额外的成本。在iDistance中，仅使用一组纵坐标，并且也仅使用一棵${\mathrm{B}}^{ + }$树来对它们进行索引，因此iDistance在减少距离计算的同时，页面访问次数也更少。此外，iDistance中参考点的选择与全系列技术中焦点基的选择有很大不同。

The P-Sphere tree [Goldstein and Ramakrishnan 2000] is a two level structure,the root level and leaf level. The root level contains a series of $<$ sphere descriptor, leaf page pointer> pairs, while each leaf of the index corresponds to a sphere (we call it the leaf sphere in the following) and contains all data points that lie within the sphere described in the corresponding sphere descriptor. The leaf sphere centers are chosen by sampling the dataset. The NN search algorithm only searches the leaf with the sphere center closest to the query point $q$ . It searches the NN (we denote it as $p$ ) of $q$ among the points in this leaf. When finding $p$ ,if the query sphere is totally contained in the leaf sphere,then we can confirm that $p$ is the nearest neighbor of $q$ ; otherwise, a second best strategy is used (such as sequential scan). A data point can be within multiple leaf spheres, so the points are stored multiple times in the P-Sphere tree. This is how it trades space for time. A variant of the P-Sphere tree is the nondeterministic (ND) P-Sphere tree, which returns answers with some probability of being correct. The ND P-Sphere tree NN search algorithm searches $k$ leaf spheres whose centers are closest to the query point,where $k$ is a given constant (note that this $k$ is different from the $K$ in KNN). A problem arises in high-dimensional space for the deterministic P-Sphere tree search, because the nearest neighbor distance tends to be very large. It is hard for the nearest leaf sphere of $q$ to contain the whole query sphere when finding the NN of $q$ within this sphere. If the leaf sphere contains the whole query sphere, the radius of the leaf sphere must be very large, typically close to the side length of the data space. In this case, where the major portion of the whole dataset is within this leaf, scanning a leaf is not much different from scanning the whole dataset. Therefore, the authors also hinted that using deterministic P-Sphere trees for medium to high dimensionality is impractical. In Goldstein and Ramakrishnan [2000], only the experimental results of ND P-Sphere are reported, which is shown to be better than sequential scan at the cost of space. Again, iDistance only uses one set of ordinates and hence has no duplicates. iDistance is meant for high dimensional KNN search; which P-Sphere tree cannot address efficiently. The ND P-Sphere tree has better performance in high-dimensional space, but our technique, iDistance is looking for exact nearest neighbors.

P球树 [Goldstein和Ramakrishnan 2000] 是一种两层结构，包括根层和叶层。根层包含一系列 <$<$ 球体描述符，叶页指针> 对，而索引的每个叶节点对应一个球体（以下我们称其为叶球体），并包含位于相应球体描述符所描述的球体内的所有数据点。叶球体的中心是通过对数据集进行采样来选择的。最近邻（NN）搜索算法仅搜索球体中心最接近查询点 $q$ 的叶节点。它在该叶节点的点中搜索 $q$ 的最近邻（我们将其表示为 $p$ ）。在找到 $p$ 时，如果查询球体完全包含在叶球体内，那么我们可以确认 $p$ 是 $q$ 的最近邻；否则，将使用次优策略（如顺序扫描）。一个数据点可以位于多个叶球体内，因此这些点在P球树中会被多次存储。这就是它以空间换时间的方式。P球树的一种变体是非确定性（ND）P球树，它以一定的正确概率返回答案。ND P球树的最近邻搜索算法搜索中心最接近查询点的 $k$ 个叶球体，其中 $k$ 是一个给定的常数（注意，这里的 $k$ 与KNN中的 $K$ 不同）。确定性P球树搜索在高维空间中会出现一个问题，因为最近邻距离往往非常大。在该球体内寻找 $q$ 的最近邻时，$q$ 的最近叶球体很难包含整个查询球体。如果叶球体包含整个查询球体，那么叶球体的半径必须非常大，通常接近数据空间的边长。在这种情况下，整个数据集的大部分都在这个叶节点内，扫描一个叶节点与扫描整个数据集没有太大区别。因此，作者还暗示，对于中高维度使用确定性P球树是不切实际的。在Goldstein和Ramakrishnan [2000] 中，只报告了ND P球树的实验结果，结果表明它以空间为代价优于顺序扫描。同样，iDistance只使用一组坐标，因此没有重复项。iDistance适用于高维K近邻（KNN）搜索；而P球树无法有效地处理这个问题。ND P球树在高维空间中具有更好的性能，但我们的技术iDistance是在寻找精确的最近邻。

Another metric based index is the Slim-tree [Traina et al. 2000], which is a height balanced and dynamic tree structure that grows from the leaves to the root. The structure is fairly similar to that of the M-tree, and the objective of the design is to reduce the overlap between the covering regions in each level of the metric tree. The split algorithm of the Slim-tree is based on the concept of minimal spanning tree [Kruskal 1956], and it distributes the objects by cutting the longest line among all the closest connecting lines between objects. If none exits, an uneven split is accepted as a compromise. The slim-down algorithm is a post-processing step applied on an existing Slim-tree to reduce the overlaps between the regions in the tree.

另一种基于度量的索引是Slim树 [Traina等人 2000]，它是一种高度平衡的动态树结构，从叶节点向根节点生长。其结构与M树相当相似，设计目标是减少度量树每一层中覆盖区域之间的重叠。Slim树的分裂算法基于最小生成树的概念 [Kruskal 1956]，它通过切断对象之间所有最近连接线段中最长的线段来分配对象。如果不存在这样的线段，则接受不均匀分裂作为折衷方案。瘦身算法是应用于现有Slim树的后处理步骤，用于减少树中区域之间的重叠。

Due to the difficulty of processing exact KNN queries, some studies, such as Arya et al. [1994, 1998] turn to approximate KNN search. In these studies, a relative error bound $\epsilon$ is specified so that the approximate KNN distance is at most $\left( {1 + \epsilon }\right)$ times the actual KNN distance. We can specify $\epsilon$ to be 0 so that exact answers are returned. However, the algorithms in Arya et al. [1994, 1998] are based on a main memory indexing structure called bd-tree, while the problem we are considering is when the data and indexes are stored on secondary memory. Main memory indexing requires a slightly different treatment since optimization on the use of L2 cache is important for speed-up. Cui et al. $\left\lbrack  {{2003},{2004}}\right\rbrack$ show that existing indexes have to be fine-tuned for exploiting L2 cache efficiently. Approximate KNN search has recently been studied in the data stream model [Koudas et al. 2004], where the memory is constrained and each data item could be read only once.

由于处理精确K近邻（KNN）查询存在困难，一些研究，如Arya等人 [1994, 1998] 转向近似KNN搜索。在这些研究中，指定了一个相对误差界 $\epsilon$，使得近似KNN距离最多是实际KNN距离的 $\left( {1 + \epsilon }\right)$ 倍。我们可以将 $\epsilon$ 指定为0，以便返回精确答案。然而，Arya等人 [1994, 1998] 中的算法基于一种称为bd树的主存索引结构，而我们考虑的问题是数据和索引存储在二级存储器中的情况。主存索引需要稍微不同的处理方式，因为优化L2缓存的使用对于提高速度很重要。Cui等人 $\left\lbrack  {{2003},{2004}}\right\rbrack$ 表明，为了有效利用L2缓存，必须对现有索引进行微调。最近，在数据流模型中研究了近似KNN搜索 [Koudas等人 2004]，其中内存受到限制，并且每个数据项只能读取一次。

While more indexes have been proposed for high-dimensional databases, other performance speedup methods such as dimensionality reduction have also been performed. The idea of dimensionality reduction is to pick the most important features to represent the data, and an index is built on the reduced space [Chakrabarti and Mehrotra 2000; Faloutsos and Lin 1995; Lin et al. 1995; Jolliffe 1986; Pagel et al. 2000]. To answer a query, it is mapped to the reduced space and the index is searched based on the dimensions indexed. The answer set returned contains all the answers and some false positives. In general, dimensionality reduction can be performed on the datasets before they are indexed as a means to reduce the effect of the dimensionality curse on the index structure. Dimensionality reduction is lossy in nature; hence the query accuracy is affected as a result. How much information is lost, depends on the specific technique used and on the specific dataset at hand. For instance, Principal Component Analysis (PCA) [Jolliffe 1986] is a widely used method for transforming points in the original (high-dimensional) space into another (usually lower dimensional) space. Using PCA, most of the information in the original space is condensed into a few dimensions along which the variances in the data distribution are the largest. When the dataset is globally correlated, principal component analysis is an effective method for reducing the number of dimensions with little or no loss of information. However, in practice, the data points tend not to be globally correlated, and the use of global dimensionality reduction may cause a significant loss of information. As an attempt to reduce such loss of information, and also to reduce query processing due to false positives, a local dimensionality reduction (LDR) technique was proposed in Chakrabarti and Mehrotra [2000]. It exploits local correlations in data points for the purpose of indexing.

虽然已经为高维数据库提出了更多的索引，但也采用了其他性能加速方法，如降维。降维的思想是挑选最重要的特征来表示数据，并在降维后的空间上构建索引 [Chakrabarti 和 Mehrotra 2000；Faloutsos 和 Lin 1995；Lin 等人 1995；Jolliffe 1986；Pagel 等人 2000]。为了回答查询，将其映射到降维后的空间，并根据索引的维度搜索索引。返回的答案集包含所有答案和一些误报。一般来说，可以在对数据集进行索引之前对其进行降维，以减少维度诅咒对索引结构的影响。降维本质上是有损的；因此查询准确性会受到影响。损失多少信息取决于所使用的具体技术和手头的具体数据集。例如，主成分分析（Principal Component Analysis，PCA）[Jolliffe 1986] 是一种广泛使用的方法，用于将原始（高维）空间中的点转换到另一个（通常是低维）空间。使用 PCA，原始空间中的大部分信息被浓缩到几个维度上，沿着这些维度数据分布的方差最大。当数据集具有全局相关性时，主成分分析是一种减少维度数量且几乎不损失信息的有效方法。然而，在实践中，数据点往往不具有全局相关性，使用全局降维可能会导致大量信息丢失。为了减少这种信息损失，并减少由于误报导致的查询处理，Chakrabarti 和 Mehrotra [2000] 提出了一种局部降维（Local Dimensionality Reduction，LDR）技术。它利用数据点的局部相关性进行索引。

### 3.THE IDISTANCE

### 3. i距离

In this section, we describe a new KNN processing scheme, called iDistance, to facilitate efficient distance-based KNN search. The design of iDistance is motivated by the following observations. First, the (dis)similarity between data points can be derived with reference to a chosen reference or representative point. Second, data points can be ordered based on their distances to a reference point. Third, distance is essentially a single dimensional value. This allows us to represent high-dimensional data in single dimensional space, thereby enabling reuse of existing single dimensional indexes such as the ${\mathrm{B}}^{ + }$ -tree. Moreover, false drops can be efficiently filtered without incurring expensive distance computation.

在本节中，我们描述一种新的 K 近邻（KNN）处理方案，称为 i距离，以促进基于距离的高效 KNN 搜索。i距离的设计基于以下观察。首先，可以参考选定的参考点或代表点来推导数据点之间的（不）相似性。其次，可以根据数据点到参考点的距离对其进行排序。第三，距离本质上是一个一维值。这使我们能够在一维空间中表示高维数据，从而能够重用现有的一维索引，如 ${\mathrm{B}}^{ + }$ -树。此外，可以有效地过滤掉误删情况，而无需进行昂贵的距离计算。

### 3.1 An Overview

### 3.1 概述

Consider a set of data points ${DB}$ in a unit $d$ -dimensional metric space ${DS}$ , which is a set of points with an associated distance function dist. Let ${p}_{1}$ : $\left( {{x}_{0},{x}_{1},\ldots ,{x}_{d - 1}}\right) ,{p}_{2} : \left( {{y}_{0},{y}_{1},\ldots ,{y}_{d - 1}}\right)$ and ${p}_{3} : \left( {{z}_{0},{z}_{1},\ldots ,{z}_{d - 1}}\right)$ be three data points in ${DS}$ . The distance function dist has the following properties:

考虑在单位 $d$ 维度量空间 ${DS}$ 中的一组数据点 ${DB}$，该空间是具有关联距离函数 dist 的一组点。设 ${p}_{1}$：$\left( {{x}_{0},{x}_{1},\ldots ,{x}_{d - 1}}\right) ,{p}_{2} : \left( {{y}_{0},{y}_{1},\ldots ,{y}_{d - 1}}\right)$ 和 ${p}_{3} : \left( {{z}_{0},{z}_{1},\ldots ,{z}_{d - 1}}\right)$ 是 ${DS}$ 中的三个数据点。距离函数 dist 具有以下性质：

$$
\operatorname{dist}\left( {{p}_{1},{p}_{2}}\right)  = \operatorname{dist}\left( {{p}_{2},{p}_{1}}\right) \;\forall {p}_{1},{p}_{2} \in  {DB} \tag{1}
$$

$$
\operatorname{dist}\left( {{p}_{1},{p}_{1}}\right)  = 0\;\forall {p}_{1} \in  {DB} \tag{2}
$$

$$
0 < \operatorname{dist}\left( {{p}_{1},{p}_{2}}\right) \;\forall {p}_{1},{p}_{2} \in  {DB};{p}_{1} \neq  {p}_{2} \tag{3}
$$

$$
\operatorname{dist}\left( {{p}_{1},{p}_{3}}\right)  \leq  \operatorname{dist}\left( {{p}_{1},{p}_{2}}\right)  + \operatorname{dist}\left( {{p}_{2},{p}_{3}}\right) \;\forall {p}_{1},{p}_{2},{p}_{3} \in  {DB} \tag{4}
$$

The last formula defines the triangular inequality, and provides a condition for selecting candidates based on metric relationship. Without loss of generality, we use the Euclidean distance as the distance function in our article, although other distance functions also apply for iDistance. For Euclidean distance, the distance between ${p}_{1}$ and ${p}_{2}$ is defined as

最后一个公式定义了三角不等式，并为基于度量关系选择候选点提供了条件。不失一般性，在本文中我们使用欧几里得距离作为距离函数，不过其他距离函数也适用于 i距离。对于欧几里得距离，${p}_{1}$ 和 ${p}_{2}$ 之间的距离定义为

$$
\operatorname{dist}\left( {{p}_{1},{p}_{2}}\right)  = \sqrt{{\left( {x}_{0} - {y}_{0}\right) }^{2} + {\left( {x}_{1} - {y}_{1}\right) }^{2} + \cdots  + {\left( {x}_{d - 1} - {y}_{d - 1}\right) }^{2}}.
$$

As in other databases, a high-dimensional database can be split into partitions. Suppose a point,denoted as ${O}_{i}$ ,is picked as the reference point for a data partition ${P}_{i}$ . As we shall see shortly, ${O}_{i}$ need not be a data point. A data point, $p$ ,in the partition can be referenced via ${O}_{i}$ in terms of its distance (or proximity) to it, $\operatorname{dist}\left( {{O}_{i},p}\right)$ . Using the triangle inequality,it is straightforward to see that

与其他数据库一样，高维数据库可以划分为多个分区。假设选择一个点，记为 ${O}_{i}$，作为数据分区 ${P}_{i}$ 的参考点。正如我们很快将看到的，${O}_{i}$ 不必是一个数据点。分区中的一个数据点 $p$ 可以通过其到 ${O}_{i}$ 的距离（或接近程度）$\operatorname{dist}\left( {{O}_{i},p}\right)$ 来引用。使用三角不等式，可以很容易地看出

$$
\operatorname{dist}\left( {{O}_{i},q}\right)  - \operatorname{dist}\left( {p,q}\right)  \leq  \operatorname{dist}\left( {{O}_{i},p}\right)  \leq  \operatorname{dist}\left( {{O}_{i},q}\right)  + \operatorname{dist}\left( {p,q}\right) .
$$

When we are working with a search radius of querydist(q),we are interested in finding all points $p$ such that $\operatorname{dist}\left( {p,q}\right)  \leq$ querydist(q). For every such point $p$ ,by adding this inequality to the above one,we must have:

当我们使用查询距离 querydist(q) 的搜索半径时，我们感兴趣的是找到所有满足 $\operatorname{dist}\left( {p,q}\right)  \leq$ querydist(q) 的点 $p$。对于每个这样的点 $p$，将这个不等式与上面的不等式相加，我们必须有：

$$
\operatorname{dist}\left( {{O}_{i},q}\right)  - \text{ querydist }\left( q\right)  \leq  \operatorname{dist}\left( {{O}_{i},p}\right)  \leq  \operatorname{dist}\left( {{O}_{i},q}\right)  + \text{ querydist }\left( q\right) .
$$

In other words,in partition ${P}_{i}$ ,we need only examine candidate points $p$ whose distance from the reference point, $\operatorname{dist}\left( {{O}_{i},p}\right)$ ,is bounded by this inequality, which in general specifies an annulus around the reference point.

换句话说，在分区 ${P}_{i}$ 中，我们只需检查候选点 $p$，这些候选点与参考点 $\operatorname{dist}\left( {{O}_{i},p}\right)$ 的距离受此不等式约束，该不等式通常指定了一个围绕参考点的圆环区域。

Let dist_ma ${x}_{i}$ be the distance between ${O}_{i}$ and the point furthest from it in partition ${P}_{i}$ . That is,let ${P}_{i}$ have a radius of dist_ma ${x}_{i}$ . If $\operatorname{dist}\left( {{O}_{i},q}\right)$ -querydist $\left( q\right)  \leq$ dist_ma ${x}_{i}$ ,then ${P}_{i}$ has to be searched for NN points,else we can eliminate this partition from consideration altogether. The range to be searched within an affected partition in the single dimensional space is $\left\lbrack  {\operatorname{dist}\left( {{0}_{i},q}\right)  - }\right.$ querydist(q), $\min \left( {{\text{dist_max}}_{i},\text{dist}\left( {{O}_{i},q}\right)  + \text{querydist}\left( q\right) }\right) \rbrack$ . Figure 2 shows an example where the partitions are formed based on data clusters (the data partitioning strategy will be discussed in detail in Section 4.2). Here,for query point $q$ and query radius $r$ ,partitions ${P}_{1}$ and ${P}_{2}$ need to be searched,while partition ${P}_{3}$ need not.

设 dist_ma ${x}_{i}$ 为 ${O}_{i}$ 与分区 ${P}_{i}$ 中距离它最远的点之间的距离。也就是说，设 ${P}_{i}$ 的半径为 dist_ma ${x}_{i}$。如果 $\operatorname{dist}\left( {{O}_{i},q}\right)$ - 查询距离 $\left( q\right)  \leq$ > dist_ma ${x}_{i}$，则必须在 ${P}_{i}$ 中搜索最近邻（NN）点，否则我们可以完全不考虑这个分区。在一维空间中受影响的分区内需要搜索的范围是 $\left\lbrack  {\operatorname{dist}\left( {{0}_{i},q}\right)  - }\right.$ 查询距离(q)，$\min \left( {{\text{dist_max}}_{i},\text{dist}\left( {{O}_{i},q}\right)  + \text{querydist}\left( q\right) }\right) \rbrack$。图 2 展示了一个基于数据簇形成分区的示例（数据分区策略将在 4.2 节详细讨论）。这里，对于查询点 $q$ 和查询半径 $r$，需要搜索分区 ${P}_{1}$ 和 ${P}_{2}$，而无需搜索分区 ${P}_{3}$。

<!-- Media -->

Leaf nodes of ${\mathrm{B}}^{ + }$ -tree

${\mathrm{B}}^{ + }$ - 树的叶节点

<img src="https://cdn.noedgeai.com/0195c914-b8e7-76ad-b610-2cb563debcc3_8.jpg?x=540&y=337&w=900&h=551&r=0"/>

Fig. 2. Search regions for NN query $q$ .

图 2. 最近邻查询 $q$ 的搜索区域。

<!-- Media -->

From the figure, it is clear that all points along a fixed radius have the same value after transformation due to the lossy transformation of data points into distance with respect to the reference points. As such, the shaded regions are the areas that need to be checked.

从图中可以清楚地看到，由于数据点相对于参考点进行有损变换为距离，沿着固定半径的所有点在变换后具有相同的值。因此，阴影区域是需要检查的区域。

To facilitate efficient metric-based KNN search, we have identified two important issues that have to be addressed:

为了便于进行高效的基于度量的 K 近邻（KNN）搜索，我们确定了两个必须解决的重要问题：

(1) What index structure can be used to support metric-based similarity search?

(1) 可以使用什么索引结构来支持基于度量的相似性搜索？

(2) How should the data space be partitioned, and which point should be picked as the reference point for a partition?

(2) 应该如何对数据空间进行分区，以及应该选择哪个点作为分区的参考点？

We focus on the first issue here, and will turn to the second issue in the next section. In other words, for this section, we assume that the data space has been partitioned, and the reference point in each partition has been determined.

我们在这里关注第一个问题，下一节将讨论第二个问题。换句话说，在本节中，我们假设数据空间已经被分区，并且每个分区中的参考点已经确定。

### 3.2 The Data Structure

### 3.2 数据结构

In iDistance, high-dimensional points are transformed into points in a single dimensional space. This is done using a three-step algorithm.

在 iDistance 中，高维点被转换为一维空间中的点。这是通过一个三步算法实现的。

In the first step, the high-dimensional data space is split into a set of partitions. In the second step, a reference point is identified for each partition. Suppose that we have $m$ partitions, ${P}_{0},{P}_{1},\ldots ,{P}_{m - 1}$ and their corresponding reference points, ${O}_{0},{O}_{1},\ldots ,{O}_{m - 1}$ .

第一步，将高维数据空间划分为一组分区。第二步，为每个分区确定一个参考点。假设我们有 $m$ 个分区 ${P}_{0},{P}_{1},\ldots ,{P}_{m - 1}$ 以及它们对应的参考点 ${O}_{0},{O}_{1},\ldots ,{O}_{m - 1}$。

Finally, in the third step, all data points are represented in a single dimensional space as follows. A data point $p : \left( {{x}_{0},{x}_{1},\ldots ,{x}_{d - 1}}\right) ,0 \leq  {x}_{j} \leq  1,0 \leq  j < d$ , has an index key, $y$ ,based on the distance from the nearest reference point ${O}_{i}$ as follows:

最后，在第三步中，所有数据点在一维空间中表示如下。一个数据点 $p : \left( {{x}_{0},{x}_{1},\ldots ,{x}_{d - 1}}\right) ,0 \leq  {x}_{j} \leq  1,0 \leq  j < d$ 具有一个索引键 $y$，该索引键基于其与最近参考点 ${O}_{i}$ 的距离，如下所示：

$$
y = i \times  c + \operatorname{dist}\left( {p,{O}_{i}}\right)  \tag{5}
$$

where $c$ is a constant used to stretch the data ranges. Essentially, $c$ serves to partition the single dimension space into regions so that all points in partition ${P}_{i}$ will be mapped to the range $\lbrack i \times  c,\left( {i + 1}\right)  \times  c).c$ must be set sufficiently large in order to avoid the overlap between the index key ranges of different partitions. Typically, it should be larger than the length of diagonal in the hypercube data space.

其中 $c$ 是一个用于扩展数据范围的常数。本质上，$c$ 用于将一维空间划分为多个区域，以便分区 ${P}_{i}$ 中的所有点都将被映射到范围 $\lbrack i \times  c,\left( {i + 1}\right)  \times  c).c$。$c$ 必须设置得足够大，以避免不同分区的索引键范围重叠。通常，它应该大于超立方体数据空间的对角线长度。

<!-- Media -->

<!-- figureText: ${\mathrm{O}}_{0}\left( {{0.0},{0.5}}\right)$ ${\mathrm{O}}_{3}\left( {{0.5},{1.0}}\right)$ D. ${\mathrm{O}}_{2}({1.0},{0.5}$ O $1/\left( {{0.5},{0.0}}\right)$ D ${\mathrm{c}}_{3}$ ${\mathrm{c}}_{4}$ A C0 -->

<img src="https://cdn.noedgeai.com/0195c914-b8e7-76ad-b610-2cb563debcc3_9.jpg?x=489&y=334&w=850&h=973&r=0"/>

Fig. 3. Mapping of data points.

图 3. 数据点的映射。

<!-- Media -->

Figure 3 shows a mapping in a 2-dimensional space. Here, ${O}_{0},{O}_{1},{O}_{2}$ and ${O}_{3}$ are the reference points; points $\mathrm{A},\mathrm{B},\mathrm{C}$ and $\mathrm{D}$ are data points in partitions associated with the reference points; and, ${c}_{0},{c}_{1},{c}_{2},{c}_{3}$ and ${c}_{4}$ are range partitioning values that represent the reference points as well. For example ${O}_{0}$ is associated with ${c}_{0}$ ,and all data points falling in its partition (the shaded region) have their distances relative to ${c}_{0}$ . Clearly,iDistance is lossy in the sense that multiple data points in the high-dimensional space may be mapped to the same value in the single dimensional space. That is, different points within a partition that are equidistant from the reference point have the same transformed value. For example,data points $\mathrm{C}$ and $\mathrm{D}$ have the same mapping value,and as a result, false positives may exist during search.

图3展示了二维空间中的一种映射。这里，${O}_{0},{O}_{1},{O}_{2}$和${O}_{3}$是参考点；点$\mathrm{A},\mathrm{B},\mathrm{C}$和$\mathrm{D}$是与参考点相关分区中的数据点；并且，${c}_{0},{c}_{1},{c}_{2},{c}_{3}$和${c}_{4}$也是表示参考点的范围分区值。例如，${O}_{0}$与${c}_{0}$相关联，并且落在其分区（阴影区域）内的所有数据点都有相对于${c}_{0}$的距离。显然，iDistance是有损的，因为高维空间中的多个数据点可能会映射到一维空间中的同一个值。也就是说，分区内与参考点等距的不同点具有相同的转换值。例如，数据点$\mathrm{C}$和$\mathrm{D}$具有相同的映射值，因此在搜索过程中可能会存在误报。

<!-- Media -->

KNN Search Algorithm iDistanceKNN $\left( {q,{\Delta r},{\max }_{ - }r}\right)$

最近邻搜索算法iDistanceKNN $\left( {q,{\Delta r},{\max }_{ - }r}\right)$

---

1. $r = 0$ ;

2. Stopflag $=$ FALSE;

2. 停止标志 $=$ 为假;

3. initialize $\operatorname{lp}\left\lbrack  \right\rbrack  ,\operatorname{rp}\left\lbrack  \right\rbrack  ,\operatorname{oflag}\left\lbrack  \right\rbrack$ ;

3. 初始化 $\operatorname{lp}\left\lbrack  \right\rbrack  ,\operatorname{rp}\left\lbrack  \right\rbrack  ,\operatorname{oflag}\left\lbrack  \right\rbrack$ ;

4. while Stopflag $=  =$ FALSE

4. 当停止标志 $=  =$ 为假时

5. $r = r + {\Delta r}$

6. SearchO(q,r);

6. 执行SearchO(q,r);

end iDistanceKNN;

iDistanceKNN结束;

	Fig. 4. iDistance KNN main search algorithm.

	图4. iDistance最近邻主搜索算法。

---

<!-- Media -->

## SearchO(q,r)

## 执行SearchO(q,r)

<!-- Media -->

---

		${p}_{\text{furthest }} =$ furthest(S,q)

		${p}_{\text{furthest }} =$ 为S中距离q最远的点

		if $\operatorname{dist}\left( {{p}_{\text{furthest }},q}\right)  < r$ and $\left| S\right|  =  = K$

		如果 $\operatorname{dist}\left( {{p}_{\text{furthest }},q}\right)  < r$ 且 $\left| S\right|  =  = K$

				Stopflag $=$ TRUE;

				停止标志 $=$ 设为真;

					/* need to continue searching for correctness sake before stop*/

					/* 在停止前为保证正确性需要继续搜索 */

			for $i = 0$ to $m - 1$

			从 $i = 0$ 到 $m - 1$

				dis $= \operatorname{dist}\left( {{O}_{i},q}\right)$ ;

				距离 $= \operatorname{dist}\left( {{O}_{i},q}\right)$ ;

				if not of $\log \left\lbrack  i\right\rbrack  / *$ if ${O}_{i}$ has not been searched before $* /$

				如果 $\log \left\lbrack  i\right\rbrack  / *$ 不满足条件 且 ${O}_{i}$ 之前未被搜索过 $* /$

					if $\operatorname{sphere}\left( {{O}_{i},{\text{ dist_max }}_{i}}\right)$ contains $q$

					如果 $\operatorname{sphere}\left( {{O}_{i},{\text{ dist_max }}_{i}}\right)$ 包含 $q$

							of $\log \left\lbrack  i\right\rbrack   =$ TRUE;

							$\log \left\lbrack  i\right\rbrack   =$ 为真；

							Inode $=$ LocateLeaf(btree, $i * c + {dis}$ );

							索引节点 $=$ 定位叶子节点(btree, $i * c + {dis}$ );

							${lp}\left\lbrack  i\right\rbrack   =$ SearchInward $\left( {\text{lnode,}i * c + \text{dis} - r}\right)$ ;

							${lp}\left\lbrack  i\right\rbrack   =$ 向内搜索 $\left( {\text{lnode,}i * c + \text{dis} - r}\right)$ ;

							${rp}\left\lbrack  i\right\rbrack   =$ SearchOutward $\left( {\text{ lnode },i * c + \text{ dis } + r}\right) ;$

							${rp}\left\lbrack  i\right\rbrack   =$ 向外搜索 $\left( {\text{ lnode },i * c + \text{ dis } + r}\right) ;$

					else if $\operatorname{sphere}\left( {{O}_{i},{\text{ dist_max }}_{i}}\right.$ ) intersects sphere(q,r)

					否则，如果 $\operatorname{sphere}\left( {{O}_{i},{\text{ dist_max }}_{i}}\right.$ 与球体(q,r)相交

							of $\operatorname{lag}\left\lbrack  i\right\rbrack   =$ TRUE;

							$\operatorname{lag}\left\lbrack  i\right\rbrack   =$ 为真；

							Inode $=$ LocateLeaf(btree,dist_max ${}_{i}$ );

							索引节点 $=$ 定位叶子节点(btree,最大距离 ${}_{i}$ );

							${lp}\left\lbrack  i\right\rbrack   =$ SearchInward $\left( {\text{lnode,}i * c + \text{dis} - r}\right)$ ;

							${lp}\left\lbrack  i\right\rbrack   =$ 向内搜索 $\left( {\text{lnode,}i * c + \text{dis} - r}\right)$ ;

				else

				否则

					if ${lp}\left\lbrack  i\right\rbrack$ not nil

					如果 ${lp}\left\lbrack  i\right\rbrack$ 不为空

						${lp}\left\lbrack  i\right\rbrack   =$ SearchInward $\left( {{lp}\left\lbrack  i\right\rbrack   \rightarrow  \text{leftnode,}i * c + {dis} - r}\right)$ ;

						${lp}\left\lbrack  i\right\rbrack   =$ 向内搜索 $\left( {{lp}\left\lbrack  i\right\rbrack   \rightarrow  \text{leftnode,}i * c + {dis} - r}\right)$ ;

					if ${rp}\left\lbrack  i\right\rbrack$ not nil

					如果 ${rp}\left\lbrack  i\right\rbrack$ 不为空

22. ${rp}\left\lbrack  i\right\rbrack   =$ SearchOutward $\left( {{rp}\left\lbrack  i\right\rbrack   \rightarrow  \text{ rightnode },i * c + {dis} + r}\right) ;$

22. ${rp}\left\lbrack  i\right\rbrack   =$ 向外搜索 $\left( {{rp}\left\lbrack  i\right\rbrack   \rightarrow  \text{ rightnode },i * c + {dis} + r}\right) ;$

end SearchO;

搜索O结束;

					Fig. 5. iDistance KNN search algorithm: SearchO.

					图5. iDistance最近邻搜索算法：搜索O。

---

<!-- Media -->

## In iDistance, we employ two data structures:

## 在iDistance中，我们采用两种数据结构：

-A ${\mathrm{B}}^{ + }$ -tree is used to index the transformed points to facilitate speedy retrieval. We choose the ${\mathrm{B}}^{ + }$ -tree because it is an efficient indexing structure for one-dimensional data and it is also available in most commercial DBMSs. In our implementation of the ${\mathrm{B}}^{ + }$ -tree,leaf nodes are linked to both the left and right siblings [Ramakrishnan and Gehrke 2000]. This is to facilitate searching the neighboring nodes when the search region is gradually enlarged.

- 一个${\mathrm{B}}^{ + }$树用于对转换后的点进行索引，以实现快速检索。我们选择${\mathrm{B}}^{ + }$树是因为它是一种高效的一维数据索引结构，并且大多数商业数据库管理系统（DBMS）中都有提供。在我们实现的${\mathrm{B}}^{ + }$树中，叶节点与左右兄弟节点相连[拉马克里什南和格尔克2000年]。这样做是为了在搜索区域逐渐扩大时便于搜索相邻节点。

-An array is used to store the $m$ data space partitions and their respective reference points. The array is used to determine the data partitions that need to be searched during query processing.

- 一个数组用于存储$m$数据空间分区及其各自的参考点。该数组用于确定查询处理期间需要搜索的数据分区。

### 3.3 KNN Search in iDistance

### 3.3 iDistance中的K近邻搜索

Figures 4-6 summarize the algorithm for KNN search with the iDistance method. The essence of the algorithm is similar to the generalized search strategy outlined in Figure 1. It begins by searching a small 'sphere', and incrementally enlarges the search space till all $K$ nearest neighbors are found. The search stops when the distance of the furthest object in $\mathrm{S}$ (answer set) from the query point $q$ is less than or equal to the current search radius $r$ .

图4 - 6总结了使用iDistance方法进行K近邻搜索的算法。该算法的本质与图1中概述的通用搜索策略类似。它从搜索一个小的“球体”开始，然后逐步扩大搜索空间，直到找到所有$K$个最近邻。当$\mathrm{S}$（答案集）中距离查询点$q$最远的对象的距离小于或等于当前搜索半径$r$时，搜索停止。

<!-- Media -->

SearchInward(node, ivalue)

向内搜索（节点，整数值）

---

	for each entry $e$ in node $\left( {e = {e}_{j},j = 1,2,\ldots ,{Number}\_ {of}\_ \text{entries}}\right)$

	  对于节点$\left( {e = {e}_{j},j = 1,2,\ldots ,{Number}\_ {of}\_ \text{entries}}\right)$中的每个条目$e$

		if $\left| S\right|  =  = K$

		  如果$\left| S\right|  =  = K$

			${p}_{\text{furthest }} =$ furthest(S,q);

			  ${p}_{\text{furthest }} =$ 最远（S，q）；

			if $\operatorname{dist}\left( {e,q}\right)  < \operatorname{dist}\left( {{p}_{\text{furthest }},q}\right)$

			  如果$\operatorname{dist}\left( {e,q}\right)  < \operatorname{dist}\left( {{p}_{\text{furthest }},q}\right)$

					$S = S - {p}_{\text{furthest }};$

					$S = S \cup  e$

		else

		  否则

				$S = S \cup  e$

		if ${e}_{1}$ .key $>$ ivalue

		  如果${e}_{1}$.键 $>$ 整数值

			node $=$ SearchInward $\left( {\text{node} \rightarrow  \text{leftnode,}i * c + {dis} - r}\right)$ ;

			  节点 $=$ 向内搜索 $\left( {\text{node} \rightarrow  \text{leftnode,}i * c + {dis} - r}\right)$ ；

	if end of partition is reached

	  如果到达分区末尾

		node = nil;

		  节点 = 空；

		return(node);

		  返回（节点）；

and SearchInward;

以及向内搜索（SearchInward）；

		Fig. 6. iDistance KNN search algorithm: SearchInward.

		图 6. iDistance K 近邻（KNN）搜索算法：向内搜索（SearchInward）。

---

<!-- Media -->

Before we explain the main concept of the algorithm iDistanceKNN, let us discuss three important routines. Note that routines SearchInward and SearchOutward are similar to each other, so we shall only explain routine SearchInward. Given a leaf node, routine SearchInward examines the entries of the node towards the left to determine if they are among the $K$ nearest neighbors, and updates the answers accordingly. We note that because iDistance is lossy, it is possible that points with the same values are actually not close to one another-some may be closer to $q$ ,while others are far from it. If the first element (or last element for SearchOutward) of the node is contained in the query sphere, then it is likely that its predecessor with respect to distance from the reference point (or successor for SearchOutward) may also be close to $q$ . As such, the left (or right for SearchOutward) sibling is examined. In other words, SearchInward (SearchOutward) searches the space towards (away from) the reference point of the partition. Let us consider again the example shown in Figure 2. For query point $q$ ,the SearchInward search on the partition ${P}_{1}$ will search towards left sibling as shown by the direction of arrow A, while the SearchOutward will search towards right sibling as shown by the direction of arrow B. For partition ${P}_{2}$ ,we only search towards left sibling by SearchInward as shown by the direction of arrow $C$ . The routine LocateLeaf is a typical ${B}^{ + }$ -tree traversal algorithm, which locates a leaf node given a search value, hence the detailed description of the algorithm is omitted. It locates the leaf node either based on the respective value of $q$ or maximum radius of the partition being searched.

在解释 iDistanceKNN 算法的主要概念之前，我们先讨论三个重要的子程序。注意，子程序“向内搜索（SearchInward）”和“向外搜索（SearchOutward）”彼此相似，因此我们仅解释“向内搜索（SearchInward）”子程序。给定一个叶节点，“向内搜索（SearchInward）”子程序会向左检查该节点的条目，以确定它们是否属于 $K$ 近邻，并相应地更新答案。我们注意到，由于 iDistance 是有损的，具有相同值的点实际上可能彼此并不接近——有些可能更接近 $q$，而有些则离它很远。如果节点的第一个元素（对于“向外搜索（SearchOutward）”则是最后一个元素）包含在查询球内，那么相对于参考点的距离而言，它的前一个元素（对于“向外搜索（SearchOutward）”则是后一个元素）也可能接近 $q$。因此，会检查左兄弟节点（对于“向外搜索（SearchOutward）”则是右兄弟节点）。换句话说，“向内搜索（SearchInward）”（“向外搜索（SearchOutward）”）是朝着（远离）分区的参考点搜索空间。让我们再次考虑图 2 所示的示例。对于查询点 $q$，在分区 ${P}_{1}$ 上进行的“向内搜索（SearchInward）”将按照箭头 A 的方向向左兄弟节点搜索，而“向外搜索（SearchOutward）”将按照箭头 B 的方向向右兄弟节点搜索。对于分区 ${P}_{2}$，我们仅按照箭头 $C$ 的方向通过“向内搜索（SearchInward）”向左兄弟节点搜索。“定位叶节点（LocateLeaf）”子程序是一种典型的 ${B}^{ + }$ 树遍历算法，它根据搜索值定位叶节点，因此省略该算法的详细描述。它要么根据 $q$ 的相应值，要么根据正在搜索的分区的最大半径来定位叶节点。

We now explain the search algorithm. Searching in iDistance begins by scanning the auxiliary structure to identify the reference points, ${O}_{i}$ ,whose data spaces intersect the query region. For a partition that needs to be searched, the starting search point must be located. If $q$ is contained inside the data sphere, the iDistance value of $q$ (obtained based on Equation 5) is used directly,else ${\text{dist_max}}_{i}$ is used. The search starts with a small radius. In our implementation, we just use ${\Delta r}$ as the initial search radius. Then the search radius is increased by ${\Delta r}$ ,step by step,to form a larger query sphere. For each enlargement,there are three cases to consider.

现在我们来解释搜索算法。iDistance 中的搜索首先扫描辅助结构，以确定其数据空间与查询区域相交的参考点 ${O}_{i}$。对于需要搜索的分区，必须确定起始搜索点。如果 $q$ 包含在数据球内，则直接使用 $q$ 的 iDistance 值（根据公式 5 获得），否则使用 ${\text{dist_max}}_{i}$。搜索从一个小半径开始。在我们的实现中，我们仅使用 ${\Delta r}$ 作为初始搜索半径。然后，搜索半径逐步增加 ${\Delta r}$，以形成一个更大的查询球。对于每次扩大，需要考虑三种情况。

(1) The partition contains the query point, $q$ . In this case,we want to traverse the partition sufficiently to determine the $K$ nearest neighbors. This can be done by first locating the leaf node whereby $q$ may be stored (Recall that this node does not necessarily contain points whose distance is closest to $q$ compared to its sibling nodes),and searching inward or outward of the reference point accordingly. For the example shown in Figure 2,only ${P}_{1}$ is examined in the first iteration and $q$ is used to traverse down the ${\mathrm{B}}^{ + }$ -tree.

（1）分区包含查询点 $q$。在这种情况下，我们希望充分遍历该分区，以确定 $K$ 近邻。这可以通过首先定位可能存储 $q$ 的叶节点（回想一下，与它的兄弟节点相比，该节点不一定包含距离 $q$ 最近的点），并相应地朝着或远离参考点进行搜索来实现。对于图 2 所示的示例，在第一次迭代中仅检查 ${P}_{1}$，并使用 $q$ 向下遍历 ${\mathrm{B}}^{ + }$ 树。

(2) The query point is outside the partition but the query sphere intersects the partition. In this case,we only need to search inward. Partition ${P}_{2}$ (with reference point ${O}_{2}$ ) in Figure 2 is searched inward when the search sphere enlarged by ${\Delta r}$ intersects ${P}_{2}$ .

（2）查询点在分区之外，但查询球与分区相交。在这种情况下，我们只需要进行向内搜索。当搜索球扩大 ${\Delta r}$ 后与图 2 中的分区 ${P}_{2}$（参考点为 ${O}_{2}$）相交时，对该分区进行向内搜索。

(3) The partition does not intersect the query sphere. Then, we do not need to examine this partition. An example in point is ${P}_{3}$ of Figure 2.

（3）分区与查询球不相交。那么，我们不需要检查这个分区。图 2 中的 ${P}_{3}$ 就是一个例子。

The search stops when the $K$ nearest neighbors have been identified from the data partitions that intersect with the current query sphere and when further enlargement of the query sphere does not change the $K$ nearest list. In other words, all points outside the partitions intersecting with the query sphere will definitely be at a distance $D$ from the query point such that $D$ is greater than querydist. This occurs at the end of some iteration when the distance of the furthest object in the answer set, $S$ ,from query point $q$ is less than or equal to the current search radius $r$ . At this time,all the points outside the query sphere have a distance larger than querydist, while all candidate points in the answer set have distance smaller than querydist. In other words, further enlargement of the query sphere would not change the answer set. Therefore, the answers returned by iDistance are of 100% accuracy.

当从与当前查询球相交的数据分区中识别出 $K$ 个最近邻，并且进一步扩大查询球不会改变 $K$ 个最近邻列表时，搜索停止。换句话说，与查询球相交的分区之外的所有点到查询点的距离 $D$ 肯定大于查询距离（querydist）。当答案集中最远对象 $S$ 到查询点 $q$ 的距离小于或等于当前搜索半径 $r$ 时，这种情况会在某次迭代结束时出现。此时，查询球之外的所有点的距离都大于查询距离，而答案集中的所有候选点的距离都小于查询距离。换句话说，进一步扩大查询球不会改变答案集。因此，iDistance 返回的答案准确率为 100%。

## 4. SELECTION OF REFERENCE POINTS AND DATA SPACE PARTITIONING

## 4. 参考点的选择和数据空间划分

To support distance-based similarity search, we need to split the data space into partitions and for each partition, we need a reference point. In this section we look at some choices. For ease of exposition, we use 2-dimensional diagrams for illustration. However, we note that the complexity of indexing problems in a high-dimensional space is much higher; for instance, the distance between points larger than one (the full normalized range in a single dimension) could still be considered close since points are relatively sparse.

为了支持基于距离的相似性搜索，我们需要将数据空间划分为多个分区，并且为每个分区选择一个参考点。在本节中，我们将探讨一些选择。为了便于说明，我们使用二维图进行示例。然而，我们注意到，高维空间中索引问题的复杂度要高得多；例如，由于点相对稀疏，即使两点之间的距离大于 1（单维中的全归一化范围），仍可能被认为是接近的。

### 4.1 Space-Based Partitioning

### 4.1 基于空间的划分

A straightforward approach to data space partitioning is to subdivide the space into equal partitions. In a $d$ -dimensional space,we have ${2d}$ hyperplanes. The method we adopted is to partition the space into ${2d}$ pyramids with the center of the unit cube space as their top, and each hyperplane forming the base of each pyramid. ${}^{1}$ We study the following possible reference point selection and partition strategies.

一种直接的数据空间划分方法是将空间细分为相等的分区。在 $d$ 维空间中，我们有 ${2d}$ 个超平面。我们采用的方法是将空间划分为 ${2d}$ 个金字塔，以单位立方体空间的中心为顶点，每个超平面构成每个金字塔的底面。${}^{1}$ 我们研究以下可能的参考点选择和分区策略。

<!-- Media -->

<!-- figureText: ${\mathrm{O}}_{3}\left( {{0.5},{1.0}}\right)$ ${\mathrm{O}}_{3}$ ${\mathrm{O}}_{2}\left( {{1.0},{0.5}}\right)$ ${\mathrm{O}}_{0}$ (b) Effective search space ${\mathrm{O}}_{0}\left( {{0.0},{0.5}}\right)$ ${\mathrm{O}}_{1}\left( {{0.5},{0.0}}\right)$ (a) Space partitioning -->

<img src="https://cdn.noedgeai.com/0195c914-b8e7-76ad-b610-2cb563debcc3_13.jpg?x=372&y=334&w=1085&h=529&r=0"/>

Fig. 7. Using (centers of hyperplanes, closest distance) as reference point.

图 7. 使用（超平面中心，最近距离）作为参考点。

<!-- Media -->

(1) Center of Hyperplane, Closest Distance. The center of each hyperplane can be used as a reference point, and the partition associated with the point contains all points that are nearest to it. Figure 7(a) shows an example in a 2-dimensional space. Here, ${O}_{0},{O}_{1},{O}_{2}$ and ${O}_{3}$ are the reference points,and point $\mathrm{A}$ is closest to ${O}_{0}$ and so belongs to the partition associated with it (the shaded region). Moreover, as shown, the actual data space is disjoint though the hyperspheres overlap. Figure 7(b) shows an example of a query region, which is the dark shaded area, and the affected space of each pyramid, which is the shaded area bounded by the pyramid boundary and the dashed curve. For each partition, the area not contained by the query sphere does not contain any answers for the query. However, since the mapping is lossy, the corner area outside the query region has to be checked since the data points have the same mapping values as those in the area intersecting with the query region.

(1) 超平面中心，最近距离。每个超平面的中心可以用作参考点，与该点关联的分区包含所有最接近它的点。图 7(a) 展示了二维空间中的一个示例。这里，${O}_{0},{O}_{1},{O}_{2}$ 和 ${O}_{3}$ 是参考点，点 $\mathrm{A}$ 最接近 ${O}_{0}$，因此属于与之关联的分区（阴影区域）。此外，如图所示，尽管超球体重叠，但实际的数据空间是不相交的。图 7(b) 展示了一个查询区域的示例，即深色阴影区域，以及每个金字塔的受影响空间，即由金字塔边界和虚线曲线界定的阴影区域。对于每个分区，查询球未包含的区域不包含该查询的任何答案。然而，由于映射存在信息损失，查询区域之外的角落区域必须进行检查，因为这些数据点与与查询区域相交的区域中的数据点具有相同的映射值。

For reference points along the central axis, the partitions look similar to those of the Pyramid tree. When dealing with query and data points, the sets of points are however not exactly identical, due to the curvature of the hypersphere as compared to the partitioning along axial hyperplanes in the case of the Pyramid tree.

对于沿中心轴的参考点，分区看起来与金字塔树的分区相似。然而，在处理查询点和数据点时，由于超球体的曲率与金字塔树中沿轴向超平面的分区相比，这些点集并不完全相同。

(2) Center of Hyperplane, Furthest Distance. The center of each hyperplane can be used as a reference point, and the partition associated with the point contains all points that are furthest from it. Figure 8(a) shows an example in a 2-dimensional space. Figure 8(b) shows the affected search area for the given query point. The shaded search area is that required by the previous scheme, while the search area caused by the current scheme is bounded by the bold arches. As can be seen in Figure 8(b), the affected search area bounded by the bold arches is now greatly reduced as compared to the closest distance counterpart. We must however note that the query search space is dependent on the choice of reference points, partition strategy and the query point itself.

(2) 超平面中心，最远距离。每个超平面的中心可以用作参考点，与该点关联的分区包含所有离它最远的点。图 8(a) 展示了二维空间中的一个示例。图 8(b) 展示了给定查询点的受影响搜索区域。阴影搜索区域是先前方案所需的区域，而当前方案导致的搜索区域由粗拱门界定。从图 8(b) 可以看出，与最近距离对应方案相比，由粗拱门界定的受影响搜索区域现在大大减小了。然而，我们必须注意，查询搜索空间取决于参考点的选择、分区策略以及查询点本身。

---

<!-- Footnote -->

${}^{1}$ We note that the space is similar to that of the Pyramid technique [Berchtold et al. 1998a]. However, the rationales behind the design and the mapping function are different; in the Pyramid method,a $d$ -dimensional data point is associated with a pyramid based on an attribute value,and is represented as a value away from the center of the space.

${}^{1}$ 我们注意到该空间与金字塔技术 [Berchtold 等人，1998a] 的空间类似。然而，设计背后的原理和映射函数是不同的；在金字塔方法中，一个 $d$ 维数据点基于一个属性值与一个金字塔相关联，并表示为离空间中心的一个值。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: Furthest points from O 1 O. The bounding area by these arches are the affected searching area of $\mathrm{{kNN}}\left( {\mathrm{Q},\mathrm{r}}\right)$ . (b) Effect of reduction on query space are located in this area. ${\mathrm{O}}_{2}$ ${\mathrm{O}}_{0}$ ${\mathrm{O}}_{1}$ (a) Space partitioning -->

<img src="https://cdn.noedgeai.com/0195c914-b8e7-76ad-b610-2cb563debcc3_14.jpg?x=372&y=337&w=1057&h=568&r=0"/>

Fig. 8. Using (center of hyperplane, furthest distance) as reference point.

图 8. 使用（超平面中心，最远距离）作为参考点。

<!-- Media -->

(3) External Point. Any point along the line formed by the center of a hyperplane and the center of the corresponding data space can also be used as a reference point. ${}^{2}$ By external point,we refer to a reference point that falls outside the data space. This heuristic is expected to perform well when the affected area is quite large, especially when the data are uniformly distributed. We note that both closest and furthest distance can be supported. Figure 9 shows an example of the closest distance scheme for a 2-dimensional space when using external points as reference points. Again, we observe that the affected search space for the same query point is reduced under an external point scheme (compared to using the center of the hyperplane).

(3) 外部点。超平面中心与相应数据空间中心所形成直线上的任何点也可以用作参考点。${}^{2}$ 我们所说的外部点是指落在数据空间之外的参考点。当受影响区域相当大时，尤其是当数据均匀分布时，这种启发式方法预计会表现良好。我们注意到最近距离和最远距离方案都可以支持。图 9 展示了在二维空间中使用外部点作为参考点时最近距离方案的一个示例。同样，我们观察到在外部点方案下，对于相同查询点的受影响搜索空间减小了（与使用超平面中心相比）。

### 4.2 Data-Based Partitioning

### 4.2 基于数据的分区

Equi-partitioning may seem attractive for uniformly distributed data. However, data in real life are often clustered or correlated. Even when no correlation exists in all dimensions, there are usually subsets of data that are locally correlated [Chakrabarti and Mehrotra 2000; Pagel et al. 2000]. In these cases, a more appropriate partitioning strategy would be used to identify clusters from the data space. There are several existing clustering schemes in the literature such

等分区对于均匀分布的数据可能看起来很有吸引力。然而，现实生活中的数据通常是聚类的或相关的。即使在所有维度上都不存在相关性，通常也有局部相关的数据子集 [Chakrabarti 和 Mehrotra，2000；Pagel 等人，2000]。在这些情况下，更合适的分区策略将用于从数据空间中识别聚类。文献中有几种现有的聚类方案，例如

---

<!-- Footnote -->

${}^{2}$ We note that the other two reference points are actually special cases of this.

${}^{2}$ 我们注意到其他两个参考点实际上是这种情况的特殊情况。

<!-- Footnote -->

---

<!-- Media -->

<img src="https://cdn.noedgeai.com/0195c914-b8e7-76ad-b610-2cb563debcc3_15.jpg?x=452&y=332&w=925&h=900&r=0"/>

Fig. 9. Space partitioning under (external point, closest distance)-based reference point.

图 9. 基于（外部点，最近距离）参考点的空间分区。

<!-- Media -->

as K-Means [MacQueen 1967], BIRCH [Zhang et al. 1996], CURE [Guha et al. 1998], and PROCLUS [Aggarwal et al. 1999]. While our metric-based indexing is not dependent on the underlying clustering method, we expect the clustering strategy to have an influence on retrieval performance. In our implementation, we adopted the K-means clustering algorithm [MacQueen 1967]. The number of clusters affects the search area and the number of traversals from the root to the leaf nodes. We expect the number of clusters to be a tuning parameter, which may vary for different applications and domains.

如 K - 均值算法 [MacQueen，1967]、BIRCH 算法 [Zhang 等人，1996]、CURE 算法 [Guha 等人，1998] 和 PROCLUS 算法 [Aggarwal 等人，1999]。虽然我们基于度量的索引不依赖于底层的聚类方法，但我们预计聚类策略会对检索性能产生影响。在我们的实现中，我们采用了 K - 均值聚类算法 [MacQueen，1967]。聚类的数量会影响搜索区域以及从根节点到叶节点的遍历次数。我们预计聚类的数量是一个调优参数，对于不同的应用和领域可能会有所不同。

Once the clusters are obtained, we need to select the reference points. Again, we have two possible options when selecting reference points:

一旦获得了聚类，我们需要选择参考点。同样，在选择参考点时我们有两种可能的选择：

(1) Center of cluster. The center of a cluster is a natural candidate as a reference point. Figure 10 shows a 2-dimensional example. Here, we have 2 clusters, one cluster has center ${O}_{1}$ and another has center ${O}_{2}$ .

(1) 聚类中心。聚类的中心自然是参考点的候选。图 10 展示了一个二维示例。这里，我们有 2 个聚类，一个聚类的中心是 ${O}_{1}$，另一个聚类的中心是 ${O}_{2}$。

(2) Edge of cluster. As shown in Figure 10, when the cluster center is used, the sphere areas of both clusters have to be enlarged to include outlier points, leading to significant overlap in the data space. To minimize the overlap, we can select points on the edge of the partition as reference points, such as points on hyperplanes, data space corners, data points at one side of a cluster and away from other clusters, and so on. Figure 11 is an example of selecting

(2) 聚类边缘。如图 10 所示，当使用聚类中心时，两个聚类的球形区域必须扩大以包含离群点，导致数据空间中出现显著的重叠。为了最小化重叠，我们可以选择分区边缘的点作为参考点，例如超平面上的点、数据空间的角点、位于一个聚类一侧且远离其他聚类的数据点等等。图 11 是在二维数据空间中选择

<!-- Media -->

<!-- figureText: 10.70 ${\mathrm{O}}_{1} : \left( {{0.20},{0.70}}\right)$ O’:(0.67,0.31) 0.67 0.31 0.20 -->

<img src="https://cdn.noedgeai.com/0195c914-b8e7-76ad-b610-2cb563debcc3_16.jpg?x=568&y=338&w=665&h=554&r=0"/>

Fig. 10. Cluster centers and reference points.

图 10. 聚类中心和参考点。

<!-- figureText: 0.70 0.67 ${\mathrm{O}}_{2} : \left( {1,0}\right)$ 0.31 0 0.20 -->

<img src="https://cdn.noedgeai.com/0195c914-b8e7-76ad-b610-2cb563debcc3_16.jpg?x=573&y=979&w=651&h=620&r=0"/>

Fig. 11. Cluster edge points as reference points.

图 11. 聚类边缘点作为参考点。

<!-- Media -->

the edge points as the reference points in a 2-dimensional data space. There are two clusters and the edge points are ${O}_{1} : \left( {0,1}\right)$ and ${O}_{2} : \left( {1,0}\right)$ . As shown, the overlap of the two partitions is smaller than that using cluster centers as reference points.

边缘点作为参考点的一个示例。有两个聚类，边缘点是 ${O}_{1} : \left( {0,1}\right)$ 和 ${O}_{2} : \left( {1,0}\right)$。如图所示，两个分区的重叠比使用聚类中心作为参考点时要小。

In short, overlap of partitioning spheres can lead to more intersections by the query sphere, and more points having the same similarity (distance) value will cause more data points to be examined if a query region covers that area. Therefore, when we choose a partitioning strategy, it is important to avoid or reduce such partitioning sphere overlap and large number of points with close similarity, as much as possible.

简而言之，划分球的重叠会导致查询球产生更多的交集，并且如果查询区域覆盖了某个区域，那么具有相同相似度（距离）值的点越多，需要检查的数据点就越多。因此，在选择划分策略时，尽可能避免或减少这种划分球的重叠以及大量相似度接近的点是很重要的。

<!-- Media -->

<!-- figureText: ${k}_{1}$ ${k}_{2}$ ${O}_{i}$ $q$ -->

<img src="https://cdn.noedgeai.com/0195c914-b8e7-76ad-b610-2cb563debcc3_17.jpg?x=629&y=339&w=570&h=403&r=0"/>

Fig. 12. Histogram-based cost model.

图12. 基于直方图的成本模型。

<!-- Media -->

### 5.A COST MODEL FOR IDISTANCE

### 5. iDistance的成本模型

iDistance is designed to handle KNN search efficiently. However, due to the complexity of very high-dimensionality or the very large $K$ used in the query, iDistance is expected to be superior for certain (but not all) scenarios. We therefore develop cost models to estimate the page access cost of iDistance, which can be used in query optimization (for example, if the iDistance has the number of page accesses less than a certain percentage of that of sequential scan, we would use iDistance instead of sequential scan). In this section, we present a cost model based on both the Power-method [Tao et al. 2003] and a histogram of the key distribution. This histogram-based cost model applies to all partitioning strategies and any data distribution, and it predicts individual query processing cost in terms of page accesses instead of average cost. The basic idea of the Power-method is to precompute the local power law for a set of representative points and perform the estimation using the local power law of a point close to the query point. In the key distribution histogram, we divide the key values into buckets and maintain the number of points that are in each bucket.

iDistance旨在高效处理K近邻（KNN）搜索。然而，由于数据的超高维度复杂性或查询中使用的非常大的$K$，iDistance预计在某些（但并非所有）场景中表现更优。因此，我们开发了成本模型来估算iDistance的页面访问成本，该模型可用于查询优化（例如，如果iDistance的页面访问次数少于顺序扫描的一定百分比，我们将使用iDistance而非顺序扫描）。在本节中，我们提出了一种基于幂方法[陶等人，2003年]和键分布直方图的成本模型。这种基于直方图的成本模型适用于所有划分策略和任何数据分布，并且它预测的是单个查询处理的页面访问成本，而非平均成本。幂方法的基本思想是预先计算一组代表性点的局部幂律，并使用靠近查询点的某个点的局部幂律进行估算。在键分布直方图中，我们将键值划分为多个桶，并记录每个桶中的点数。

Figure 12 shows an example of how to estimate the page access cost for a partition ${P}_{i}$ ,whose reference point is ${O}_{i}.q$ is the query point and $r$ is the query radius. ${k}_{1}$ is on the line $q{O}_{i}$ and with the largest key in the partition ${P}_{i}.{k}_{2}$ is the intersection of the query sphere and the line $q{O}_{i}$ . First,we use the Power-method to estimate the $K$ th nearest neighbor distance $r$ ,which equals the query radius when the search terminates. Then we can calculate the key of ${k}_{2}$ , $\left| {q{O}_{i}}\right|  - r + i \cdot  c$ ,where $i$ is the partition number and $c$ is the constant to stretch the key values. Since we know the boundary information of each partition and hence the key of ${k}_{1}$ ,we know the range of the keys accessed in partition ${P}_{i}$ ,that is,between the keys of ${k}_{2}$ and ${k}_{1}$ . By checking the keys distribution histogram, we know the number of points accessed in this key range, ${N}_{a,i}$ ; then the number of pages accessed in the partition is $\left\lceil  {{N}_{a,i}/{C}_{\text{eff }}}\right\rceil$ . The summation of the number of page accesses of all the partitions provides us the number of page accesses for the query.

图12展示了一个如何估算分区${P}_{i}$的页面访问成本的示例，其中参考点为${O}_{i}.q$，${O}_{i}.q$是查询点，$r$是查询半径。${k}_{1}$位于直线$q{O}_{i}$上，并且是分区${P}_{i}.{k}_{2}$中键值最大的点，${P}_{i}.{k}_{2}$是查询球与直线$q{O}_{i}$的交点。首先，我们使用幂方法估算第$K$近邻距离$r$，当搜索终止时，该距离等于查询半径。然后，我们可以计算${k}_{2}$的键值$\left| {q{O}_{i}}\right|  - r + i \cdot  c$，其中$i$是分区编号，$c$是用于拉伸键值的常数。由于我们知道每个分区的边界信息，从而知道${k}_{1}$的键值，我们就知道在分区${P}_{i}$中访问的键值范围，即${k}_{2}$和${k}_{1}$的键值之间。通过查看键分布直方图，我们可以知道在这个键值范围内访问的点数${N}_{a,i}$；然后，该分区中访问的页面数为$\left\lceil  {{N}_{a,i}/{C}_{\text{eff }}}\right\rceil$。所有分区的页面访问次数之和即为该查询的页面访问次数。

<!-- Media -->

<img src="https://cdn.noedgeai.com/0195c914-b8e7-76ad-b610-2cb563debcc3_18.jpg?x=704&y=339&w=398&h=396&r=0"/>

Fig. 13. Histogram-based cost model, query sphere inside the partition.

图13. 基于直方图的成本模型，查询球在分区内部。

<!-- Media -->

Note that, if the query sphere is inside a partition as shown in Figure 13, both ${k}_{1}$ and ${k}_{2}$ are intersections of the query sphere and the line $q{O}_{i}$ . Different from the above case,the key of ${k}_{1}$ is $\left| {q{O}_{i}}\right|  + r + i \cdot  c$ here. The number of page accesses is derived in the same way as above.

请注意，如果查询球位于分区内部，如图13所示，${k}_{1}$和${k}_{2}$都是查询球与直线$q{O}_{i}$的交点。与上述情况不同的是，这里${k}_{1}$的键值为$\left| {q{O}_{i}}\right|  + r + i \cdot  c$。页面访问次数的计算方法与上述相同。

The costs estimated by the techniques described above turn out to be very close to actual costs observed, as we will show in the experimental section that follows.

正如我们将在后续的实验部分展示的那样，上述技术估算的成本与实际观察到的成本非常接近。

In Jagadish et al. [2004], we also present cost models purely based on formula derivations. They are less expensive to maintain and compute, in that no summary data structures need be maintained, but they assume uniform data distribution and therefore are not accurate for nonuniform workloads. Where data distributions are known, these or similar other formulae may be used to advantage.

在贾加迪什等人[2004年]的研究中，我们还提出了纯粹基于公式推导的成本模型。这些模型的维护和计算成本较低，因为不需要维护汇总数据结构，但它们假设数据是均匀分布的，因此对于非均匀工作负载来说并不准确。在已知数据分布的情况下，可以利用这些或其他类似的公式。

### 6.A PERFORMANCE STUDY

### 6. 性能研究

In this section, we present results of an experimental study performed to evaluate iDistance. First we compare the space-based partitioning strategy and the data-based partitioning strategy and find that the data-based partitioning strategy is much better. Then we focus our study on the behavior of iDistance using the data-based partitioning strategy with various parameters and under different workloads. At last we compare iDistance with other metric based indexing methods, the M-tree and the Omni-sequential, as well as a main memory bd-tree [Arya et al. 1994]. We have also evaluated iDistance against iMinMax [Ooi et al. 2000] and A-tree [Sakurai et al. 2000], and our results, which have been reported in Yu et al. [2001] showed the superiority of iDistance over these schemes. As such, we shall not duplicate the latter results here.

在本节中，我们展示了为评估iDistance（i距离）而进行的一项实验研究的结果。首先，我们比较了基于空间的分区策略和基于数据的分区策略，发现基于数据的分区策略要好得多。然后，我们将研究重点放在使用基于数据的分区策略、不同参数以及不同工作负载下iDistance的性能表现上。最后，我们将iDistance与其他基于度量的索引方法（M树（M-tree）和全序法（Omni-sequential））以及一种主存bd树（[Arya等人，1994年]）进行了比较。我们还将iDistance与iMinMax（[Ooi等人，2000年]）和A树（[Sakurai等人，2000年]）进行了评估，我们的结果（已在Yu等人[2001年]的报告中提及）显示了iDistance相对于这些方案的优越性。因此，我们在此不再重复后面这些结果。

We implemented the iDistance technique and associated search algorithms in $\mathrm{C}$ ,and used the ${\mathrm{B}}^{ + }$ -tree as the single dimensional index structure. We obtained the M-tree, Omni-sequential, and the bd-tree from the authors or their web sites, and standardized the codes as much as we could for fair comparison. Each index page is 4096 Bytes. Unless stated otherwise, all the experiments were performed on a computer with Pentium(R) 1.6 GHz CPU and 256 MB RAM except the comparison with bd-tree (the experimental setting for this comparison would be specified later). The operating system running on this computer is RedHat Linux 9. We conducted many experiments using various datasets. Each result we show was obtained as the average (number of page accesses or total response time) over 200 queries that follow the same distribution of the data.

我们用$\mathrm{C}$实现了iDistance技术及相关的搜索算法，并使用${\mathrm{B}}^{ + }$树作为一维索引结构。我们从作者或他们的网站获取了M树、全序法和bd树，并尽可能对代码进行了标准化处理，以进行公平比较。每个索引页为4096字节。除非另有说明，除了与bd树的比较（此比较的实验设置将在后面说明）之外，所有实验均在一台配备奔腾（R）1.6 GHz CPU和256 MB RAM的计算机上进行。该计算机运行的操作系统是RedHat Linux 9。我们使用各种数据集进行了许多实验。我们展示的每个结果都是对遵循相同数据分布的200个查询的平均值（页面访问次数或总响应时间）。

<!-- Media -->

<!-- figureText: 0.9 0.5 0.6 0.7 0.8 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.1 0.2 0.3 0.4 -->

<img src="https://cdn.noedgeai.com/0195c914-b8e7-76ad-b610-2cb563debcc3_19.jpg?x=547&y=335&w=729&h=710&r=0"/>

Fig. 14. Distribution of the clustered data.

图14. 聚类数据的分布。

<!-- Media -->

In the experiment, we generated 8, 16, 30-dimensional uniform, and clustered datasets. The dataset size ranges from 100,000 to 500,000 data points. For the clustered datasets, the default number of clusters is 20 . The cluster centers are randomly generated and in each cluster, the data follow the normal distribution with the default standard deviation of 0.05 . Figure 14 shows a 2-dimensional image of the data distribution.

在实验中，我们生成了8维、16维、30维的均匀数据集和聚类数据集。数据集大小从100,000到500,000个数据点不等。对于聚类数据集，默认的聚类数量为20。聚类中心是随机生成的，并且在每个聚类中，数据遵循默认标准差为0.05的正态分布。图14展示了数据分布的二维图像。

We also used a real dataset, the Color Histogram dataset. This dataset is obtained from http://kdd.ics.uci.edu/databases/CorelFeatures/CorelFeatures.data.html.It contains image features extracted from a Corel image collection. HSV color space is divided into 32 subspaces (32 colors: 8 ranges of $\mathrm{H}$ and 4 ranges of S). And the value in each dimension in a Color Histogram of an image is the density of each color in the entire image. The number of records is 68,040 . All the data values of each dimension are normalized to the range $\left\lbrack  {0,1}\right\rbrack$ .

我们还使用了一个真实的数据集，即颜色直方图数据集。该数据集可从http://kdd.ics.uci.edu/databases/CorelFeatures/CorelFeatures.data.html获取。它包含从Corel图像集中提取的图像特征。HSV颜色空间被划分为32个子空间（32种颜色：$\mathrm{H}$的8个范围和S的4个范围）。图像颜色直方图中每个维度的值是整个图像中每种颜色的密度。记录数量为68,040。每个维度的所有数据值都被归一化到$\left\lbrack  {0,1}\right\rbrack$范围。

In our evaluation, we use the number of page accesses and the total response time as the performance metric. Default value of ${\Delta r}$ is 0.01,that is, $1\%$ of the side length of the data space. The initial search radius is just set as ${\Delta r}$ .

在我们的评估中，我们使用页面访问次数和总响应时间作为性能指标。${\Delta r}$的默认值为0.01，即数据空间边长的$1\%$。初始搜索半径设置为${\Delta r}$。

<!-- Media -->

<!-- figureText: 4500 200 space-based Total response time (millisec) data-based ----X- seq. scan --- ※··· 150 100 5 10 15 20 25 30 Dimensionality (b) Total response time 4000 space-based data-based ----X- 3500 seq. scan ---米··· Page accesses 3000 2500 2000 1500 1000 500 10 15 20 25 30 Dimensionality (a) Page accesses -->

<img src="https://cdn.noedgeai.com/0195c914-b8e7-76ad-b610-2cb563debcc3_20.jpg?x=364&y=341&w=1078&h=412&r=0"/>

Fig. 15. Space-based partitioning vs. data-based partitioning, uniform data.

图15. 基于空间的分区与基于数据的分区，均匀数据。

<!-- Media -->

### 6.1 Comparing Space-Based and Data-Based Partitioning Strategies

### 6.1 比较基于空间的分区策略和基于数据的分区策略

We begin by investigating the relative performance of the partitioning strategies. Note that the number of reference points is always ${2d}$ for the space-based partitioning approach,and for a fair comparison,we also use ${2d}$ reference points in the data-based partitioning approach. Figure 15 shows the result of 10NN queries on the 100,000 uniform dataset. The space-based partitioning has almost the same page accesses as sequential scan when dimensionality is 8 and more page accesses than sequential scan in high dimensionality. The data-based partitioning strategy has fewer page accesses than sequential scan when dimensionality is 8 , more page accesses when dimensionality is 16 , and almost the same page accesses when dimensionality is 30 . This is because the pruning effect of the data-based strategy is better in low dimensionality than in high dimensionality. The relative decrease (compared to sequential scan) of page accesses when dimensionality is 30 is because of the larger number of reference points. While iDistance's page access performance is not attractive relative to sequential scan, the total response time performance is better because of its ability to filter data using a single dimensional key. The total response time of the space-based partitioning is about ${60}\%$ that of sequential scan when dimensionality is 8 , same as sequential scan when dimensionality is 16 , but worse than sequential scan when dimensionality is 30 . The total response time of the data-based partitioning is always less than both of the others, while its difference from the sequential scan decreases as dimensionality increases.

我们首先研究分区策略的相对性能。请注意，对于基于空间的分区方法，参考点的数量始终为${2d}$，为了进行公平比较，我们在基于数据的分区方法中也使用${2d}$个参考点。图15展示了在100,000个均匀分布数据集上进行10近邻（10NN）查询的结果。当维度为8时，基于空间的分区的页面访问次数几乎与顺序扫描相同，而在高维度时，其页面访问次数比顺序扫描更多。基于数据的分区策略在维度为8时的页面访问次数比顺序扫描少，在维度为16时页面访问次数更多，在维度为30时页面访问次数几乎相同。这是因为基于数据的策略在低维度时的剪枝效果比高维度时更好。当维度为30时，页面访问次数相对于顺序扫描的相对减少是由于参考点数量较多。虽然iDistance的页面访问性能相对于顺序扫描并不出众，但由于它能够使用一维键过滤数据，其总响应时间性能更好。当维度为8时，基于空间的分区的总响应时间约为顺序扫描的${60}\%$；当维度为16时，与顺序扫描相同；但当维度为30时，比顺序扫描更差。基于数据的分区的总响应时间始终小于其他两种方法，并且随着维度的增加，它与顺序扫描的差异减小。

Figure 16 shows the result of ${10}\mathrm{{NN}}$ queries on the 100,000 clustered dataset. Both partitioning strategies are better than sequential scan in both page accesses and total response time. This is because for clustered data,the $K$ th nearest neighbor distance is much smaller than that in the uniform data. In this case, iDistance can prune a lot of data points in the searching. The total response time of the space-based partitioning is about ${20}\%$ that of sequential scan. The total response time of data-based partitioning is less than 10% that of sequential scan. Again, the data-based partitioning is better than both of the others.

图16展示了在100,000个聚类数据集上进行${10}\mathrm{{NN}}$查询的结果。在页面访问次数和总响应时间方面，两种分区策略都优于顺序扫描。这是因为对于聚类数据，第$K$近邻距离比均匀数据中的要小得多。在这种情况下，iDistance可以在搜索过程中剪枝大量的数据点。基于空间的分区的总响应时间约为顺序扫描的${20}\%$。基于数据的分区的总响应时间小于顺序扫描的10%。同样，基于数据的分区优于其他两种方法。

In Section 4.1, we discussed using external point as the reference points of the space-based partitioning. A comparison between using external points and the center point as the reference point on the uniform datasets is shown in Figure 17.

在4.1节中，我们讨论了使用外部点作为基于空间的分区的参考点。图17展示了在均匀数据集上使用外部点和中心点作为参考点的比较。

<!-- Media -->

<!-- figureText: 4500 space-based Total response time (millisec) data-based ----X- seq. scan ...*... 10 15 20 25 30 Dimensionality (b) Total response time 4000 space-based data-based ----x- 3500 seq. scar Page accesses 3000 2500 2000 1500 1000 500 5 10 20 25 Dimensionality (a) Page accesses -->

<img src="https://cdn.noedgeai.com/0195c914-b8e7-76ad-b610-2cb563debcc3_21.jpg?x=375&y=341&w=1079&h=408&r=0"/>

Fig. 16. Space-based partitioning vs. data-based partitioning, clustered data.

图16. 基于空间的分区与基于数据的分区，聚类数据。

<!-- figureText: 4500 200 hyperplane center Total response time (millisec) external point ---x- further external point seq. scan ………… 150 100 50 0 10 15 20 25 30 Dimensionality (b) Total response time 4000 hyperplane center external 3500 further external point --- 米··· Page accesses 3000 2500 2000 1500 1000 500 5 10 20 25 Dimensionality (a) Page accesses -->

<img src="https://cdn.noedgeai.com/0195c914-b8e7-76ad-b610-2cb563debcc3_21.jpg?x=375&y=846&w=1079&h=415&r=0"/>

Fig. 17. Effect of reference points in space-based partitioning, uniform data.

图17. 基于空间的分区中参考点的影响，均匀数据。

<!-- Media -->

Using an external point as the reference point has slightly better performance than using the center point, and using a farther external point is slightly better than using the external point in turn, but the difference between them is not big, and all of them are still worse than the data-based partitioning approach (compare with Figure 15). Here, the farther external point is already very far (more than 10 times the side length of the data space) and the performance using even farther points almost does not change, therefore they are not presented.

使用外部点作为参考点的性能略优于使用中心点，而使用更远的外部点又略优于使用较近的外部点，但它们之间的差异不大，并且所有这些方法仍然比基于数据的分区方法差（与图15比较）。这里，更远的外部点已经非常远（超过数据空间边长的10倍），并且使用更远的点时性能几乎没有变化，因此未展示这些结果。

From the above results, we can see that the data-based partitioning scheme is always better than the space-based partitioning approach. Thus, for all subsequent experimental study, we will mainly focus on the data-based partitioning strategy. However, we note that the space-based partitioning is always better than sequential scan in low and medium dimensional spaces (less than 16). Thus, it is useful for these workloads. Moreover, the scheme incurs much less overhead since there is no need to cluster data to find the reference points as in the data-based partitioning.

从上述结果可以看出，基于数据的分区方案始终优于基于空间的分区方法。因此，在所有后续的实验研究中，我们将主要关注基于数据的分区策略。然而，我们注意到，基于空间的分区在低维和中维空间（小于16）中始终优于顺序扫描。因此，它对这些工作负载是有用的。此外，该方案的开销要小得多，因为不需要像基于数据的分区那样对数据进行聚类来找到参考点。

<!-- Media -->

<!-- figureText: 2000 200 Total response time (millisec) 100 iDistance $\rightarrow$ seq. scan ---x--- 0 20 40 60 80 100 120 140 Number of reference points (b) Total response time 1500 Page accesses 1000 500 iDistance — seq. scan ---x--- 20 40 60 80 100 120 140 Number of reference points (a) Page accesses -->

<img src="https://cdn.noedgeai.com/0195c914-b8e7-76ad-b610-2cb563debcc3_22.jpg?x=364&y=337&w=1079&h=414&r=0"/>

Fig. 18. Effects of number of reference points, uniform data.

图18. 参考点数量的影响，均匀数据。

<!-- Media -->

### 6.2 iDistance Using Data-Based Partitioning

### 6.2 使用基于数据的分区的iDistance

In this subsection, we further study the performance of iDistance using a data-based partitioning strategy (iDistance for short in the sequel). We study the effects of different parameters and different workloads. As reference, we compare the iDistance with sequential scan. Although iDistance is better than sequential scan for the 30-dimensional uniform dataset, the difference is small. To see more clearly the behavior of iDistance, we use 16-dimensional data when we test on uniform datasets. For clustered data, we use 30-dimensional datasets since iDistance is still much better than sequential scan for such high dimensionality.

在本小节中，我们进一步研究使用基于数据的分区策略的iDistance（以下简称iDistance）的性能。我们研究不同参数和不同工作负载的影响。作为参考，我们将iDistance与顺序扫描进行比较。虽然在30维均匀数据集上iDistance优于顺序扫描，但差异较小。为了更清楚地了解iDistance的性能，我们在均匀数据集上测试时使用16维数据。对于聚类数据，我们使用30维数据集，因为在如此高的维度下iDistance仍然比顺序扫描好得多。

## Experiments on Uniform Datasets

## 均匀数据集上的实验

In the first experiment, we study the effect of the number of reference points on the performance of iDistance. The results of 10NN queries on the 100,000 16-dimensional uniform dataset are shown in Figure 18. We can see that as the number of reference points increases, both the number of page accesses and total response time decrease. This is expected, as smaller and fewer clusters need to be examined (i.e., more data are pruned). The amount of the decrease in time also decreases as the number of reference points increases. While we can choose a very large number of reference points to improve the performance, this will increase (a) the CPU time as more reference points need to be checked, and (b) the time for clustering to find the reference points. Moreover, there will also be more fragmented pages. So a moderate number of reference points is fine. In our other experiments, we used 64 as the default number of reference points.

在第一个实验中，我们研究参考点数量对iDistance性能的影响。在100,000个16维均匀数据集上进行10近邻（10NN）查询的结果如图18所示。我们可以看到，随着参考点数量的增加，页面访问次数和总响应时间都会减少。这是符合预期的，因为需要检查的簇更小且数量更少（即，更多的数据被剪枝）。随着参考点数量的增加，时间减少的幅度也会减小。虽然我们可以选择非常多的参考点来提高性能，但这会增加（a）CPU时间，因为需要检查更多的参考点，以及（b）用于聚类以找到参考点的时间。此外，还会有更多碎片化的页面。因此，适度数量的参考点就可以了。在我们的其他实验中，我们默认使用64个参考点。

The second experiment studies the effect of $K$ on the performance of iDistance. We varied $K$ from 10 to 50 at the step of 10 . The results of queries on the 100,000 16-dimensional uniform dataset are shown in Figure 19. As expected,as $K$ increases,iDistance incurs a larger number of page accesses. However, it remains superior over sequential scan. In terms of total response time, while both iDistance's and sequential scan's response times increase linearly as $K$ increases,the rate of increase for iDistance is slower. This is because as $K$ increases,the number of distance computations also increases for both iDistance and sequential scan. But, iDistance not only has fewer distance computations, the rate of increase in the distance computation is also smaller (than sequential scan).

第二个实验研究$K$对iDistance性能的影响。我们以10为步长，将$K$从10变化到50。在100,000个16维均匀数据集上进行查询的结果如图19所示。正如预期的那样，随着$K$的增加，iDistance会产生更多的页面访问次数。然而，它仍然优于顺序扫描。就总响应时间而言，随着$K$的增加，iDistance和顺序扫描的响应时间都呈线性增加，但iDistance的增加速率较慢。这是因为随着$K$的增加，iDistance和顺序扫描的距离计算次数也会增加。但是，iDistance不仅距离计算次数更少，而且距离计算的增加速率也更小（与顺序扫描相比）。

<!-- Media -->

<!-- figureText: 2000 iDistance Total response time (millisec) seq. scan ---x-- 15 20 25 30 35 40 45 50 K (b) Total response time 1500 Page accesses 1000 iDistance seq. scan 500 10 15 20 25 30 40 45 (a) Page accesses -->

<img src="https://cdn.noedgeai.com/0195c914-b8e7-76ad-b610-2cb563debcc3_23.jpg?x=375&y=341&w=1079&h=410&r=0"/>

Fig. 19. Effects of $K$ ,uniform data.

图19. $K$的影响，均匀数据。

<!-- figureText: 8000 800 700 iDistance Total response time (millisec) 500 400 200 100 100 150 200 250 300 350 400 450 500 Dataset size (thousand) (b) Total response time 7000 iDistance seq. scan ---x--- 6000 Page accesses 5000 4000 2000 1000 100 150 200 250 300 350 400 450 500 Dataset size (thousand) (a) Page accesses -->

<img src="https://cdn.noedgeai.com/0195c914-b8e7-76ad-b610-2cb563debcc3_23.jpg?x=375&y=829&w=1078&h=418&r=0"/>

Fig. 20. Effects of dataset size, uniform data.

图20. 数据集大小的影响，均匀数据。

<!-- Media -->

The third experiment studies the effect of the dataset size. We varied the number of data points from 100,000 to 500,000 . The results of 10NN queries on five 16-dimensional uniform datasets are shown in Figure 20. The number of page accesses and the total response time of both iDistance and sequential scan increase linearly as the dataset size increases, but the increase for sequential scan is much faster. When the dataset size is 500,000, the number of page accesses and the total response time of iDistance are about half of that of sequential scan.

第三个实验研究数据集大小的影响。我们将数据点的数量从100,000变化到500,000。在五个16维均匀数据集上进行10NN查询的结果如图20所示。随着数据集大小的增加，iDistance和顺序扫描的页面访问次数和总响应时间都呈线性增加，但顺序扫描的增加速度要快得多。当数据集大小为500,000时，iDistance的页面访问次数和总响应时间约为顺序扫描的一半。

The fourth experiment examines the effect of the ${\Delta r}$ in the iDistance KNN Search Algorithm presented in Figure 4. Figure 21 shows the performance when we varied the values of ${\Delta r}$ . We can observe that,as ${\Delta r}$ increases,both the number of page accesses and total response time decrease at first but then increase. For a small ${\Delta r}$ ,there will be more iterations to reach the final query radius and consequently, more pages are accessed and more CPU time is incurred. On the other hand,if ${\Delta r}$ is too large,the query radius may exceed the KNN distance at the last iteration and redundant data pages are fetched for checking. We note that it is very difficult to derive an optimal ${\Delta r}$ since it is dependent on the data distribution and the order in which the data points are inserted into the index. Fortunately, the impact on performance is marginal (less than 10%). Considering that,in practice,small $K$ may be used in KNN search,which implies a very small KNN distance. Therefore, in all our experiments, we have safely set ${\Delta r} = {0.01}$ ,that is, $1\%$ of the side length of the data space.

第四个实验考察图4中提出的iDistance K近邻（KNN）搜索算法中${\Delta r}$的影响。图21显示了我们改变${\Delta r}$值时的性能。我们可以观察到，随着${\Delta r}$的增加，页面访问次数和总响应时间起初会减少，但随后会增加。对于较小的${\Delta r}$，需要更多的迭代才能达到最终的查询半径，因此会访问更多的页面并消耗更多的CPU时间。另一方面，如果${\Delta r}$太大，查询半径可能会在最后一次迭代时超过KNN距离，从而会获取冗余的数据页面进行检查。我们注意到，很难推导出一个最优的${\Delta r}$，因为它取决于数据分布以及数据点插入索引的顺序。幸运的是，对性能的影响很小（小于10%）。考虑到在实际应用中，KNN搜索可能会使用较小的$K$，这意味着KNN距离非常小。因此，在我们所有的实验中，我们安全地设置${\Delta r} = {0.01}$，即数据空间边长的$1\%$。

<!-- Media -->

<!-- figureText: 2000 140 Total response time (millisec) 120 100 80 K=10 60 40 20 0.02 0.04 0.06 0.08 0.1 0.12 0.14 0.16 Delta r (b) Total response time 1500 Page accesses 1000 K=10 500 0.02 0.04 0.06 0.08 0.1 0.12 0.14 0.16 Delta r (a) Page accesses -->

<img src="https://cdn.noedgeai.com/0195c914-b8e7-76ad-b610-2cb563debcc3_24.jpg?x=364&y=335&w=1079&h=418&r=0"/>

Fig. 21. Effects of ${\Delta r}$ ,uniform data.

图21. ${\Delta r}$的影响，均匀数据。

<!-- figureText: 3000 200 Total response time (millisec) 150 100 iDistance 50 20 40 60 80 100 120 140 Number of reference points (b) Total response time 2500 Page accesses 2000 1500 iDistance 1000 500 20 40 60 80 100 120 140 Number of reference points (a) Page accesses -->

<img src="https://cdn.noedgeai.com/0195c914-b8e7-76ad-b610-2cb563debcc3_24.jpg?x=362&y=831&w=1080&h=418&r=0"/>

Fig. 22. Effects of number of reference points, clustered data.

图22. 参考点数量的影响，聚类数据。

<!-- Media -->

## Experiments on Clustered Datasets

## 聚类数据集上的实验

For the clustered datasets, we also study the effect of the number of the reference points, $K$ ,and dataset size. By default,the number of reference points is ${64},K$ is 10 and dataset size is 100,000 . Dimensionality of all these datasets is 30 . The results are shown in Figures 22, 23 and 24 respectively. These results exhibit similar characteristics to those of the uniform datasets except that iDistance has much better performance compared to sequential scan. The speedup factor is as high as 10 . The reason is that for clustered data,the $K$ th nearest neighbor distance is much smaller than that in uniform data, so many more data points can be pruned from the search. Figure 22 shows that after the number of reference points exceeds 32 , the performance gain becomes almost constant. For the rest of the experiments, we use 64 reference points as default.

对于聚类数据集，我们还研究了参考点数量 $K$ 和数据集大小的影响。默认情况下，参考点数量 ${64},K$ 为 10，数据集大小为 100,000。所有这些数据集的维度均为 30。结果分别如图 22、图 23 和图 24 所示。这些结果与均匀数据集的结果具有相似的特征，只是与顺序扫描相比，iDistance 的性能要好得多。加速因子高达 10。原因在于，对于聚类数据，第 $K$ 近邻距离比均匀数据中的要小得多，因此在搜索过程中可以剔除更多的数据点。图 22 显示，当参考点数量超过 32 后，性能提升几乎保持不变。在其余的实验中，我们默认使用 64 个参考点。

<!-- Media -->

<!-- figureText: 3000 200 Total response time (millisec) 150 100 iDistance seq. scan 50 0 15 20 25 30 35 40 45 50 K (b) Total response time 2500 Page accesses 2000 1500 iDistance 1000 500 10 15 20 25 30 35 40 45 K (a) Page accesses -->

<img src="https://cdn.noedgeai.com/0195c914-b8e7-76ad-b610-2cb563debcc3_25.jpg?x=375&y=338&w=1079&h=411&r=0"/>

Fig. 23. Effects of $K$ ,clustered data.

图 23. $K$ 的影响，聚类数据。

<!-- figureText: 14000 800 Total response time (millisec) seq. scan ---x--- 600 500 400 300 200 100 150 200 250 300 350 400 450 500 Dataset size (thousand) (b) Total response time 12000 seq. scan ---x--- Page accesses 10000 8000 6000 2000 0 100 150 200 250 300 350 400 450 500 Dataset size (thousand) (a) Page accesses -->

<img src="https://cdn.noedgeai.com/0195c914-b8e7-76ad-b610-2cb563debcc3_25.jpg?x=374&y=845&w=1080&h=418&r=0"/>

Fig. 24. Effects of dataset size, clustered data.

图 24. 数据集大小的影响，聚类数据。

<!-- Media -->

Each of the above clustered datasets consists of 20 clusters, each of which has a standard deviation of 0.05 . To evaluate the performance of iDistance on different distributions, we tested three other datasets with different numbers of clusters and different standard deviations, while other settings are kept at the default values. The results are shown in Figure 25. Because all these datasets have the same number of data points but only differ in distribution, the performance of sequential scan is almost the same for all of them, hence we only plot one curve for sequential scan on these datasets. We observe that the total response time of iDistance remains very small for all the datasets with standard deviation $\sigma$ less than or equal to 0.1 but increases a lot when the standard deviation increases to 0.2 . This is because as the standard deviation increases, the distribution of the dataset becomes closer to uniform distribution, which is when iDistance becomes less efficient (but is still better than sequential scan).

上述每个聚类数据集由 20 个聚类组成，每个聚类的标准差为 0.05。为了评估 iDistance 在不同分布下的性能，我们测试了另外三个具有不同聚类数量和不同标准差的数据集，同时其他设置保持默认值。结果如图 25 所示。由于所有这些数据集的数据点数量相同，只是分布不同，因此顺序扫描在所有数据集上的性能几乎相同，所以我们在这些数据集上只绘制了一条顺序扫描的曲线。我们观察到，对于标准差 $\sigma$ 小于或等于 0.1 的所有数据集，iDistance 的总响应时间仍然非常小，但当标准差增加到 0.2 时，响应时间会大幅增加。这是因为随着标准差的增加，数据集的分布变得更接近均匀分布，此时 iDistance 的效率会降低（但仍然优于顺序扫描）。

We also studied the effect of different ${\Delta r}$ on the clustered datasets. Like the results on the uniform datasets, the performance change is very small.

我们还研究了不同 ${\Delta r}$ 对聚类数据集的影响。与均匀数据集上的结果类似，性能变化非常小。

<!-- Media -->

<!-- figureText: 3000 200 Total response time (millisec) 150 100 20 clusters, sigma=0.05 20 clusters, sigma=0.1 ---x--- 50 20 clusters, sigma=0.2 …* 50 clusters, sigma=0.1 seq. scan ---_ 10 15 20 25 30 35 40 50 (b) Total response time 2500 20 clusters. sigma $= {0.05}$ 20 clusters, sigma=0.1 Page accesses 20 clusters. sigma=0.2 2000 50 clusters. sigma=0.1 1500 1000 500 0 母 15 20 25 30 35 40 45 50 K (a) Page accesses -->

<img src="https://cdn.noedgeai.com/0195c914-b8e7-76ad-b610-2cb563debcc3_26.jpg?x=364&y=335&w=1079&h=411&r=0"/>

Fig. 25. Effects of different data distribution, clustered data.

图 25. 不同数据分布的影响，聚类数据。

<!-- figureText: 7000 Total response time (millisec) 400 iDistance ----X---- $\cdots  =  - \frac{3}{4}k - \cdots$ 300 seq. scan 200 10 15 20 25 40 45 50 (b) Total response time 6000 iDistance M-tree ----X-- Page accesses 5000 Omni seq. $\cdots  - \frac{m}{m}\cdots$ seq. scan 4000 3000 2000 1000 0 15 20 25 30 50 K (a) Page accesses -->

<img src="https://cdn.noedgeai.com/0195c914-b8e7-76ad-b610-2cb563debcc3_26.jpg?x=364&y=821&w=1079&h=414&r=0"/>

Fig. 26. Comparative study, 16-dimensional uniform data.

图 26. 对比研究，16 维均匀数据。

<!-- Media -->

### 6.3 Comparative Study of iDistance and Other Techniques

### 6.3 iDistance 与其他技术的对比研究

In this subsection, we compare iDistance with sequential scan and two other metric based indexing methods, the M-tree [Ciaccia et al. 1997] and the Omni-sequential [Filho et al. 2001]. Both the M-tree and the Omni-sequential are disk-based indexing schemes. We also compare iDistance with a main memory index, the bd-tree [Arya et al. 1994] in the environment of constrained memory. In Filho et al. [2001], several indexing schemes of the Omni-family were proposed, and the Omni-sequential was reported to have the best average performance. We therefore pick the Omni-sequential from the family for comparison. The Omni-sequential needs to select a good number of foci bases to work efficiently. In our comparative study, we tried the Omni-sequential for several numbers of foci bases and only presented the one giving the best performance in the sequel. We still use 64 reference points for iDistance. Datasets used include 100,000 16-dimensional uniformly distributed points, 100,000 30-dimensional clustered points and 68040 32-dimensional real data. We varied $K$ from 10 to 50 at the step of 10 .

在本小节中，我们将 iDistance 与顺序扫描以及另外两种基于度量的索引方法进行比较，即 M 树（M-tree，[Ciaccia 等人，1997]）和 Omni 顺序法（Omni-sequential，[Filho 等人，2001]）。M 树和 Omni 顺序法都是基于磁盘的索引方案。我们还在内存受限的环境下，将 iDistance 与一种主存索引 bd 树（bd-tree，[Arya 等人，1994]）进行了比较。在 Filho 等人 [2001] 的研究中，提出了几种 Omni 家族的索引方案，其中 Omni 顺序法被报道具有最佳的平均性能。因此，我们从该家族中选择 Omni 顺序法进行比较。Omni 顺序法需要选择合适数量的焦点基才能高效工作。在我们的对比研究中，我们尝试了不同数量焦点基的 Omni 顺序法，并在后续只展示性能最佳的一种。我们仍然为 iDistance 使用 64 个参考点。使用的数据集包括 100,000 个 16 维均匀分布的点、100,000 个 30 维聚类点和 68040 个 32 维的真实数据。我们将 $K$ 以 10 为步长从 10 变化到 50。

First we present the comparison between the disk-based methods. The results on the uniform dataset are shown in Figure 26. Both the M-tree and the Omni-sequential have more page accesses and longer total response time than sequential scan. iDistance has similar page accesses to sequential scan, but shorter total response time than sequential scan. The results on the clustered dataset are shown in Figure 27. The M-tree, the Omni-sequential and iDistance are all better than sequential scan because the smaller $K$ th nearest neighbor distance enables more effective pruning of the data space for these metric based methods. iDistance performs the best. It has a speedup factor of about 3 over the M-tree and 6 over the Omni-sequential. The results on the real dataset are shown in Figure 28. The M-tree and the Omni-sequential have similar page accesses as sequential scan while the number of page accesses of iDistance is about $1/3$ those of the other techniques. The Omni-sequential and iDistance have shorter total response times than sequential scan while the M-tree has a very long total response time. The Omni-sequential can reduce the number of distance computations, so it takes less time while having the same page accesses as sequential scan. The M-tree accesses the pages randomly, therefore it is much slower. iDistance has significantly fewer page accesses and distance computations, hence it has the least total response time.

首先，我们展示基于磁盘的方法之间的比较。均匀数据集上的结果如图26所示。与顺序扫描相比，M树（M-tree）和全序法（Omni-sequential）的页面访问次数更多，总响应时间更长。iDistance的页面访问次数与顺序扫描相似，但总响应时间比顺序扫描短。聚类数据集上的结果如图27所示。M树、全序法和iDistance都比顺序扫描好，因为较小的第$K$近邻距离使这些基于度量的方法能更有效地对数据空间进行剪枝。iDistance表现最佳。它比M树快约3倍，比全序法快约6倍。真实数据集上的结果如图28所示。M树和全序法的页面访问次数与顺序扫描相似，而iDistance的页面访问次数约为其他技术的$1/3$。全序法和iDistance的总响应时间比顺序扫描短，而M树的总响应时间非常长。全序法可以减少距离计算的次数，因此在与顺序扫描页面访问次数相同的情况下，它花费的时间更少。M树随机访问页面，因此速度要慢得多。iDistance的页面访问次数和距离计算次数明显更少，因此总响应时间最短。

<!-- Media -->

<!-- figureText: 3000 200 Total response time (millisec) 150 iDistance 100 Omni seq seq. scar 50 0 15 20 25 30 35 40 45 50 K (b) Total response time 2500 Page accesses 2000 iDistance 1500 ----X---- Omni seq. $\cdots  + x\cdots$ seq. scan 1000 500 10 15 20 25 30 35 40 45 K (a) Page accesses -->

<img src="https://cdn.noedgeai.com/0195c914-b8e7-76ad-b610-2cb563debcc3_27.jpg?x=375&y=337&w=1079&h=412&r=0"/>

Fig. 27. Comparative study, 30-dimensional clustered data.

图27. 对比研究，30维聚类数据。

<!-- figureText: Page accesses iDistance 300 Total response time (millisec) 200 M-tree ----X---- Omni seq 150 seq. scan 100 50 15 20 25 30 35 40 45 50 K (b) Total response time 1500 M-tree ----X Omni seq. --- ※··· seq. scan 1000 500 10 15 20 25 30 35 40 45 K (a) Page accesses -->

<img src="https://cdn.noedgeai.com/0195c914-b8e7-76ad-b610-2cb563debcc3_27.jpg?x=375&y=831&w=1079&h=412&r=0"/>

Fig. 28. Comparative study, 32-dimensional real data.

图28. 对比研究，32维真实数据。

<!-- Media -->

Next we compare the iDistance with the bd-tree [Arya et al. 1994]. The bd-tree was proposed to process approximate KNN queries, but it is able to return exact KNNs when the error bound $\epsilon$ is set to 0 . All other parameters used in the bd-tree are set to the values suggested by the authors. The bd-tree is a memory resident index that loads the full index and data in memory, while iDistance reads in index and data pages from disk as and when they are required. To have a sensible comparison, we conducted this set of experiments on a computer with a small memory,whose size is ${32}\mathrm{M}$ bytes. The CPU of the computer is a Pentium ${266}\mathrm{{MHz}}$ and the operating system is RedHat Linux 9. When the bd-tree runs out of memory, we let the operating system do the paging. As the performance of a main memory structure is affected more by the size of the dataset,we study the effect of dataset size instead of $K$ . Since the main memory index has no explicit page access operation, we only present the total response time as the performance measurement. Figure 29(a) shows the results on the 16-dimensional uniform datasets. When the dataset is small (less than 200,000), the bd-tree is slightly better than iDistance; however, as the dataset grows beyond certain size (greater than300,000),the total response time increases dramatically. When the dataset size is400,000,the total response time of the bd-tree is more than 4 times that of iDistance. The reason is obvious. When the whole dataset can fit into memory, its performance is better than the disk-based iDistance, but when the data size goes beyond the available memory, thrashing occurs and impairs the performance considerably. In fact, the response time deteriorates significantly when the dataset size hits 300,000 data points or ${19}\mathrm{M}$ bytes. The reason is that the operating system also uses up a fair amount of memory so the memory available for the index is less than the total. Figure 29(b) shows the results on the 30-dimensional clustered datasets. As before, the bd-tree performs well when the dataset size is small and degrades significantly when the dataset size increases. However, the trend is less intense than that of the uniform datasets, as the index takes advantage of the locality of the clustered data and hence less thrashing happens. The results on the 32-dimensional real dataset are similar to that of the 30-dimensional clustered dataset up to the point of dataset size of 50,000 . Since the real dataset has a much smaller size than the available memory, the bd-tree performs better than iDistance. However, in practice, we probably would not have so much memory available for a single query processing process. Therefore, an efficient index must be scalable in terms of data size and be main memory efficient.

接下来，我们将iDistance与bd树（bd-tree）[Arya等人，1994年]进行比较。bd树是为处理近似K近邻（KNN）查询而提出的，但当误差界$\epsilon$设置为0时，它能够返回精确的K近邻。bd树中使用的所有其他参数都设置为作者建议的值。bd树是一种内存驻留索引，它将完整的索引和数据加载到内存中，而iDistance根据需要从磁盘读取索引和数据页面。为了进行合理的比较，我们在一台内存较小（大小为${32}\mathrm{M}$字节）的计算机上进行了这组实验。该计算机的CPU是奔腾${266}\mathrm{{MHz}}$，操作系统是RedHat Linux 9。当bd树内存不足时，我们让操作系统进行页面调度。由于主存结构的性能受数据集大小的影响更大，我们研究数据集大小的影响，而不是$K$。由于主存索引没有显式的页面访问操作，我们仅将总响应时间作为性能度量。图29(a)显示了16维均匀数据集上的结果。当数据集较小时（少于200,000），bd树略优于iDistance；然而，当数据集增长到一定大小（大于300,000）时，总响应时间急剧增加。当数据集大小为400,000时，bd树的总响应时间是iDistance的4倍多。原因很明显。当整个数据集可以放入内存时，其性能优于基于磁盘的iDistance，但当数据大小超过可用内存时，会发生内存颠簸，显著影响性能。实际上，当数据集大小达到300,000个数据点或${19}\mathrm{M}$字节时，响应时间会显著恶化。原因是操作系统也会占用相当多的内存，因此可用于索引的内存小于总内存。图29(b)显示了30维聚类数据集上的结果。和之前一样，当数据集较小时，bd树表现良好，而当数据集大小增加时，性能显著下降。然而，这种趋势不如均匀数据集那么明显，因为索引利用了聚类数据的局部性，因此内存颠簸现象较少。32维真实数据集上的结果在数据集大小达到50,000之前与30维聚类数据集的结果相似。由于真实数据集的大小远小于可用内存，bd树的性能优于iDistance。然而，在实际应用中，我们可能没有那么多内存用于单个查询处理过程。因此，一个高效的索引必须在数据大小方面具有可扩展性，并且在主存使用上具有高效性。

<!-- Media -->

<!-- figureText: 200 Total response time (second) iDistance 12 bd-tree 4 2 50 100 150 200 250 300 350 400 450 500 Dataset size (thousand) (b) 30-dimensional clustered data Total response time (second) iDistance bd-tree ----X-- 150 100 50 50 100 150 200 250 300 350 400 Dataset size (thousand) (a) 16-dimensional uniform data -->

<img src="https://cdn.noedgeai.com/0195c914-b8e7-76ad-b610-2cb563debcc3_28.jpg?x=362&y=337&w=1081&h=412&r=0"/>

Fig. 29. Comparison with a main memory index: bd-tree.

图29. 与主存索引bd树的比较。

<!-- figureText: 500 70 Total response time (millisec) 60 50 40 with updates no updates 30 20 10 80 85 90 95 100 Percentage of data inserted (b) Total response time 450 400 Page accesses 350 300 250 with updates 200 no updates --- 150 100 50 80 85 90 95 100 Percentage of data inserted (a) Page accesses -->

<img src="https://cdn.noedgeai.com/0195c914-b8e7-76ad-b610-2cb563debcc3_29.jpg?x=375&y=339&w=1078&h=420&r=0"/>

Fig. 30. iDistance performance with updates.

图30. 带有更新操作的iDistance性能。

<!-- Media -->

### 6.4 On Updates

### 6.4 更新操作

We use clustering to choose the reference points from a collection of data points, and fix them from that point onwards. It is therefore important to see whether a dynamic workload would affect the performance of iDistance much. In this experiment,we first construct the index using ${80}\%$ of the data points from the real dataset. We run ${20010}\mathrm{{NN}}$ queries and record the average number of page accesses and total response time. Then we insert 5% of the data to the database and rerun the same queries. This process is repeated until the other ${20}\%$ of the data are inserted. Separately,we also run the same queries on the index built based on the reference points chosen for ${85}\% ,{90}\% ,{95}\%$ and ${100}\%$ of the dataset. We compare the average number of page accesses and total response time of the two as shown in Figure 30. The difference between them is very small. The reason is that real data from the same source tends to follow a similar distribution, so the reference points chosen at different times are similar. Of course, if the distribution of the data changes too much, we will need choose the reference points again and rebuild the index.

我们使用聚类方法从一组数据点中选择参考点，并从那时起固定这些参考点。因此，了解动态工作负载是否会对iDistance的性能产生重大影响非常重要。在这个实验中，我们首先使用真实数据集中${80}\%$的数据点构建索引。我们运行${20010}\mathrm{{NN}}$个查询，并记录平均页面访问次数和总响应时间。然后，我们向数据库中插入5%的数据，并重新运行相同的查询。重复这个过程，直到插入其余${20}\%$的数据。另外，我们还对基于为数据集的${85}\% ,{90}\% ,{95}\%$和${100}\%$选择的参考点构建的索引运行相同的查询。我们比较两者的平均页面访问次数和总响应时间，如图30所示。它们之间的差异非常小。原因是来自同一数据源的真实数据往往遵循相似的分布，因此在不同时间选择的参考点是相似的。当然，如果数据的分布变化太大，我们将需要重新选择参考点并重建索引。

### 6.5 Evaluation of the Cost Models

### 6.5 成本模型评估

Since our cost model estimates page accesses of each individual query, we show the actual number of page accesses and the estimated page accesses from 5 randomly chosen queries on the real dataset in Figure 31. Estimation of each of these 5 queries has the relative error below 20%. For all the tested queries, the estimations of more than ${95}\%$ of them achieve a relative error below ${20}\%$ . Considering that iDistance often has a speedup factor of 2 to 6 over other techniques, the 20% error will not affect the query optimization result greatly.

由于我们的成本模型估计每个单独查询的页面访问次数，我们在图31中展示了从真实数据集中随机选择的5个查询的实际页面访问次数和估计的页面访问次数。这5个查询的估计相对误差均低于20%。对于所有测试的查询，超过${95}\%$的查询估计相对误差低于${20}\%$。考虑到iDistance相对于其他技术通常有2到6倍的加速因子，20%的误差不会对查询优化结果产生太大影响。

We also measured the time needed for computing the cost model. The average computation time (including the time for retrieving the number from the histogram) is less than $3\%$ of the average KNN query processing time. So this cost model is still a practical approach for query optimization.

我们还测量了计算成本模型所需的时间。平均计算时间（包括从直方图中检索数值的时间）小于平均K近邻（KNN）查询处理时间的$3\%$。因此，这个成本模型仍然是一种实用的查询优化方法。

### 6.6 Summary of the Experimental Results

### 6.6 实验结果总结

The data-based partitioning approach is more efficient than the space-based partitioning approach. The iDistance using the data-based partitioning is always better than the other techniques in all our experiments on various workloads. For uniform data, it beats sequential scan in dimensionality as high as 30 . Of course, due to the intrinsic characteristics of the KNN problem, we expect iDistance to lose out to sequential scan in much higher dimensionality on uniform datasets. However, for more practical data distributions, where data are skew and clustered, iDistance shows much better performance compared with sequential scan. Its speedup factor over sequential scan is as high as 10 .

基于数据的分区方法比基于空间的分区方法更高效。在我们对各种工作负载进行的所有实验中，使用基于数据分区的iDistance始终优于其他技术。对于均匀数据，它在高达30维的情况下仍优于顺序扫描。当然，由于KNN问题的内在特性，我们预计在均匀数据集的更高维度上，iDistance会不如顺序扫描。然而，对于更实际的数据分布，即数据存在偏斜和聚类的情况，与顺序扫描相比，iDistance表现出更好的性能。它相对于顺序扫描的加速因子高达10。

<!-- Media -->

<!-- figureText: 1200 actual estimated 4 5 Query number 1000 Page accesses 800 600 400 200 0 2 -->

<img src="https://cdn.noedgeai.com/0195c914-b8e7-76ad-b610-2cb563debcc3_30.jpg?x=595&y=337&w=615&h=439&r=0"/>

Fig. 31. Evaluation of the histogram-based cost model.

图31. 基于直方图的成本模型评估。

<!-- Media -->

The number of reference points is an important tunable parameter for iDistance. Generally, the more the number of reference points, the better the performance, and at the same time, the longer the time needed for clustering to determine these reference points. Too many reference points also impairs performance because of higher computation overhead. Therefore, a moderate number is fine. We have used 64 as the number of reference points in most of our experiments (the others are because we need to study the effects of number of reference points) and iDistance performs better than sequential scan and other indexing techniques in these experiments. For a dataset with unknown data distribution, we suggest 60 to 80 reference points. Usually iDistance achieves a speedup factor of 2 to 6 over the other techniques. We can use a histogram-based cost model in query optimization to estimate the page access cost of iDistance, which usually has a relative error below 20%.

参考点的数量是iDistance的一个重要可调参数。一般来说，参考点的数量越多，性能越好，但同时，确定这些参考点所需的聚类时间也越长。过多的参考点也会因更高的计算开销而损害性能。因此，适中的数量就可以了。在我们的大多数实验中，我们使用64作为参考点的数量（其他情况是因为我们需要研究参考点数量的影响），并且在这些实验中，iDistance的表现优于顺序扫描和其他索引技术。对于数据分布未知的数据集，我们建议使用60到80个参考点。通常，iDistance相对于其他技术的加速因子为2到6。我们可以在查询优化中使用基于直方图的成本模型来估计iDistance的页面访问成本，该模型的相对误差通常低于20%。

The space-based partitioning is simpler and can be used in low and medium dimensional space.

基于空间的分区方法更简单，可用于低维和中等维度的空间。

## 7. CONCLUSION

## 7. 结论

Similarity search is of growing importance, and is often most useful for objects represented in a high dimensionality attribute space. A central problem in similarity search is to find the points in the dataset nearest to a given query point. In this article we have presented a simple and efficient method, called iDistance, for K-nearest neighbor (KNN) search in a high-dimensional metric space.

相似性搜索的重要性日益增加，并且对于在高维属性空间中表示的对象通常最为有用。相似性搜索的一个核心问题是在数据集中找到离给定查询点最近的点。在本文中，我们提出了一种简单而高效的方法，称为iDistance，用于在高维度量空间中进行K近邻（KNN）搜索。

Our technique partitions the data and selects one reference point for each partition. The data in each cluster can be described based on their similarity with respect to a reference point, hence they can be transformed into a single dimensional space based on such relative similarity. This allows us to index the data points using a ${\mathrm{B}}^{ + }$ -tree structure and perform KNN search using a simple one-dimensional range search. As such, the method is well suited for integration into existing DBMSs.

我们的技术对数据进行分区，并为每个分区选择一个参考点。每个聚类中的数据可以根据它们与参考点的相似性进行描述，因此可以基于这种相对相似性将它们转换到一维空间中。这使我们能够使用${\mathrm{B}}^{ + }$ -树结构对数据点进行索引，并使用简单的一维范围搜索来执行KNN搜索。因此，该方法非常适合集成到现有的数据库管理系统（DBMS）中。

The choice of partition and reference points provides the iDistance technique with degrees of freedom that most other techniques do not have. We described how appropriate choices here can effectively adapt the index structure to the data distribution. In fact, several well-known data structures can be obtained as special cases of iDistance suitable for particular classes of data distributions. A cost model was proposed for iDistance KNN search to facilitate query optimization.

分区和参考点的选择为iDistance技术提供了大多数其他技术所没有的自由度。我们描述了如何通过恰当的选择使索引结构有效地适应数据分布。事实上，一些著名的数据结构可以作为iDistance的特殊情况得到，适用于特定类型的数据分布。我们为iDistance的K近邻（KNN）搜索提出了一个成本模型，以方便查询优化。

We conducted an extensive experimental study to evaluate iDistance against two other metric based indexes, the M-tree and the Omni-sequential, and the main memory based bd-tree structure. As a reference, we also compared iDistance against sequential scan. Our experimental results showed that iDistance outperformed the other techniques in most of the cases. Moreover, iDistance can be incorporated into existing DBMS cost effectively since the method is built on top of the ${\mathrm{B}}^{ + }$ -tree. Thus,we believe iDistance is a practical and efficient indexing method for nearest neighbor search.

我们进行了广泛的实验研究，将iDistance与另外两种基于度量的索引（M树和全序索引）以及基于主存的bd树结构进行了比较。作为参考，我们还将iDistance与顺序扫描进行了对比。实验结果表明，在大多数情况下，iDistance的性能优于其他技术。此外，由于iDistance方法是建立在${\mathrm{B}}^{ + }$树之上的，因此可以经济高效地将其集成到现有的数据库管理系统（DBMS）中。因此，我们认为iDistance是一种实用且高效的最近邻搜索索引方法。

## REFERENCES

## 参考文献

Aggarwal, C., Procopruc, C., Wolf, J., Yu, P., AND Park, J. 1999. Fast algorithm for projected clustering. In Proceedings of the ACM SIGMOD International Conference on Management of Data.

Arya, S., Mount, D., NETANYAHU, N., SILVERMAN, R., AND WU, A. 1994. An optimal algorithm for approximate nearest neighbor searching. In Proceedings of the Fifth Annual ACM-SIAM Symposium on Discrete Algorithms, 573-582.

ARYA, S., Mount, D., NETANYAHU, N., SILVERMAN, R., AND WU, A. 1998. An optimal algorithm for approximate nearest neighbor searching fixed dimensions. J. ACM 45, 6, 891-923.

BERCHTOLD, S., BöHM, C., JagADISH, H., KRIEGEL, H., AND SANDER, J. 2000. Independent quantization: An index compression technique for high-dimensional data spaces. In Proceedings of the International Conference on Data Engineering. 577-588.

BERCHTOLD, S., BöHM, C., AND KRIEGEL, H.-P. 1998a. The pyramid-technique: Towards breaking the curse of dimensionality. In Proceedings of the ACM SIGMOD International Conference on Management of Data. 142-153.

BERCHTOLD, S., ERTL, B., KEIM, D., KRIEGEL, H.-P., AND SEIDL, T. 1998b. Fast nearest neighbor search in high-dimensional space. In Proceedings of the International Conference on Data Engineering. 209-218.

BERCHTOLD, S., KEIM, D., AND KRIEGEL, H. 1996. The X-tree: An index structure for high-dimensional data. In Proceedings of the International Conference on Very Large Data Bases. 28-37.

Beyer, K., Goldstein, J., Rahakrishnan, R., and Shaft, U. 1999. When is nearest neighbors meaningful? In Proceedings of the International Conference on Database Theory.

Böhu, C., Berснтоц, S., AND KEIM, D. 2001. Searching in high-dimensional spaces: Index structures for improving the performance of multimedia databases. ACM Comput. Surv. 33, 322- 373.

BOZKAYA, T. AND OZSOYOGLU, M. 1997. Distance-based indexing for high-dimensional metric spaces. In Proceedings of the ACM SIGMOD International Conference on Management of Data. 357-368.

Chakrabarti, K. and Mehrotra, S. 1999. The hybrid tree: An index structure for high dimensional feature spaces. In Proceedings of the International Conference on Data Engineering. 322-331.

Chakrabarti, K. and Mehrotra, S. 2000. Local dimensionality reduction: a new approach to indexing high dimensional spaces. In Proceedings of the International Conference on Very Large Databases. 89-100.

Claccia, P., Pattella, M., and Zezula, P. 1997. M-trees: An efficient access method for similarity search in metric space. In Proceedings of the International Conference on Very Large Data Bases. ${426} - {435}$ .

CuI, B., OoI, B. C., Su, J. W., AND TAN, K. L. 2003. Contorting high dimensional data for efficient main memory processing. In Proceedings of the ACM SIGMOD Conference. 479-490.

CuI, B., OoI, B. C., Su, J. W., AND TAN, K. L. 2004. Indexing high-dimensional data for efficient in-memory similarity search. In IEEE Trans. Knowl. Data Eng. to appear.

FALOUTSOS, C. AND LIN, K.-I. 1995. Fastmap: A fast algorithm for indexing, data-mining and visualization of traditional and multimedia datasets. In Proceedings of the ACM SIGMOD International Conference on Management of Data. 163-174.

FLHO, R. F. S., Traina, A., and FALOUTSOS, C. 2001. Similarity search without tears: The omni family of all-purpose access methods. In Proceedings of the International Conference on Data Engineering. 623-630.

Goldstein, J. and Ramakerishnan, R. 2000. Contrast plots and p-sphere trees: space vs. time in nearest neighbor searches. In Proceedings of the International Conference on Very Large Databases. 429-440.

GUHA, S., RASTOGI, R., AND SHIM, K. 1998. Cure: an efficient clustering algorithm for large databases. In Proceedings of the ACM SIGMOD International Conference on Management of Data.

GUTTMAN, A. 1984. R-trees: A dynamic index structure for spatial searching. In Proceedings of the ACM SIGMOD International Conference on Management of Data. 47-57.

Jagabi, H., OoI, B. C., Tan, K.-L., Yu, C., AND Zhang, R. 2004. iDistance: An adaptive B ${}^{ + }$ -tree based indexing method for nearest neighbor search. Tech. Rep. www.comp.nus.edu.sg/~ooibc, National University of Singapore.

JOLLIFFE, I. T. 1986. Principle Component Analysis. Springer-Verlag.

Katahaya, N. and Satoh, S. 1997. The SR-tree: An index structure for high-dimensional nearest neighbor queries. In Proceedings of the ACM SIGMOD International Conference on Management of Data.

KOUDAS, N., OoI, B. C., TAN, K.-L., AND ZHANG, R. 2004. Approximate NN queries on streams with guaranteed error/performance bounds. In Proceedings of the International Conference on Very Large Data Bases. 804-815.

KRUSKAL, J. B. 1956. On the shortest spanning subtree of a graph and the travelling salesman problem. In Proceedings of the American Mathematical Society 7, 48-50.

LIN, K., JAGADISH, H., AND FALOUTSOS, C. 1995. The TV-tree: An index structure for high-dimensional data. VLDB Journal 3, 4, 517-542.

MacQueen, J. 1967. Some methods for classification and analysis of multivariate observations. In Fifth Berkeley Symposium on Mathematical statistics and probability. University of California Press, 281-297.

OoI, B. C., TAN, K. L., Yu, C., AND BRESSAN, S. 2000. Indexing the edge: a simple and yet efficient approach to high-dimensional indexing. In Proceedings of the ACM SIGACT-SIGMOD-SIGART Symposium on Principles of Database Systems. 166-174.

PageL, B.-U., Korn, F., AND FALOUTSOS, C. 2000. Deflating the dimensionality curse using multiple fractal dimensions. In Proceedings of the International Conference on Data Engineering.

RamakRISHNAN, R. AND GEHRKE, J. 2000. Database Management Systems. McGraw-Hill.

Sakura, Y., Yoshikawa, M., and Uemura, S. 2000. The a-tree: An index structure for high-dimensional spaces using relative approximation. In Proceedings of the International Conference on Very Large Data Bases. 516-526.

TAO, Y., FALOUTSOS, C., AND PAPADIAS, D. 2003. The power-method: A comprehensive estimation technique for multi-dimensional queries. In Proceedings of the Conference on Information and Knowledge Management.

Traina, A., Seeger, B., AND FALoutsos, C. 2000. Slim-trees: high performance metric trees minimizing overlap between nodes. In Advances in Database Technology-EDBT 2000, International

Conference on Extending Database Technology, Konstanz, Germany, March 27-31, 2000, Proceedings. Lecture Notes in Computer Science, vol. 1777. Springer-Verlag, 51-65.

WEBER, R., Schek, H., AND BLOTT, S. 1998. A quantitative analysis and performance study for similarity-search methods in high-dimensional spaces. In Proceedings of the International Conference on Very Large Data Bases. 194-205.

WHITE, D. AND JAIN, R. 1996. Similarity indexing with the SS-tree. In Proceedings of the International Conference on Data Engineering. 516-523.

Yu, C., OoI, B. C., Tan, K. L., and Jagadish, H. 2001. Indexing the distance: an efficient method to knn processing. In Proceedings of the International Conference on Very Large Data Bases. ${421} - {430}$ .

Zhang, T., Rahakrishnan, R., and Livny, M. 1996. Birch: an efficient data clustering method for very large databases. In Proceedings of the ACM SIGMOD International Conference on Management of Data.