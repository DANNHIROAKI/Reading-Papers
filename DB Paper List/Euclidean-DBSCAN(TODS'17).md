# On the Hardness and Approximation of Euclidean DBSCAN

# 欧几里得DBSCAN算法的复杂度与近似求解

JUNHAO GAN, University of Queensland

甘俊豪，昆士兰大学

YUFEI TAO, Chinese University of Hong Kong

陶宇飞，香港中文大学

DBSCAN is a method proposed in 1996 for clustering multi-dimensional points, and has received extensive applications. Its computational hardness is still unsolved to this date. The original KDD'96 paper claimed an algorithm of $O\left( {n\log n}\right)$ "average runtime complexity" (where $n$ is the number of data points) without a rigorous proof. In 2013,a genuine $O\left( {n\log n}\right)$ -time algorithm was found in 2D space under Euclidean distance. The hardness of dimensionality $d \geq  3$ has remained open ever since.

DBSCAN是1996年提出的一种用于多维点聚类的方法，已得到广泛应用。至今其计算复杂度问题仍未解决。最初发表于KDD'96的论文声称有一种“平均运行时间复杂度”为$O\left( {n\log n}\right)$的算法（其中$n$是数据点的数量），但未给出严格证明。2013年，在二维欧几里得空间中发现了一种真正的$O\left( {n\log n}\right)$时间复杂度算法。从那时起，维度$d \geq  3$的复杂度问题一直悬而未决。

This article considers the problem of computing DBSCAN clusters from scratch (assuming no existing indexes) under Euclidean distance. We prove that,for $d \geq  3$ ,the problem requires $\Omega \left( {n}^{4/3}\right)$ time to solve,unless very significant breakthroughs-ones widely believed to be impossible-could be made in theoretical computer science. Motivated by this,we propose a relaxed version of the problem called $\rho$ -approximate DBSCAN, which returns the same clusters as DBSCAN, unless the clusters are "unstable" (i.e., they change once the input parameters are slightly perturbed). The $\rho$ -approximate problem can be settled in $O\left( n\right)$ expected time regardless of the constant dimensionality $d$ .

本文考虑在欧几里得距离下从头计算DBSCAN聚类的问题（假设没有现有的索引）。我们证明，对于$d \geq  3$，该问题需要$\Omega \left( {n}^{4/3}\right)$的时间来解决，除非理论计算机科学能取得重大突破（人们普遍认为这是不可能的）。基于此，我们提出了该问题的一个松弛版本，称为$\rho$ -近似DBSCAN，它返回与DBSCAN相同的聚类，除非聚类是“不稳定的”（即，一旦输入参数稍有扰动，聚类就会改变）。无论常数维度$d$是多少，$\rho$ -近似问题都可以在$O\left( n\right)$的期望时间内解决。

The article also enhances the previous result on the exact DBSCAN problem in 2D space. We show that, if the $n$ data points have been pre-sorted on each dimension (i.e.,one sorted list per dimension),the problem can be settled in $O\left( n\right)$ worst-case time. As a corollary,when all the coordinates are integers,the 2D DBSCAN problem can be solved in $O\left( {n\log \log n}\right)$ time deterministically,improving the existing $O\left( {n\log n}\right)$ bound.

本文还改进了之前关于二维空间中精确DBSCAN问题的结果。我们证明，如果$n$个数据点在每个维度上都已预先排序（即每个维度有一个排序列表），则该问题可以在$O\left( n\right)$的最坏情况下时间内解决。由此可得，当所有坐标都是整数时，二维DBSCAN问题可以在$O\left( {n\log \log n}\right)$的时间内确定性地解决，这改进了现有的$O\left( {n\log n}\right)$界。

Categories and Subject Descriptors: H3.3 [Information Search and Retrieval]: Clustering

分类与主题描述符：H3.3 [信息搜索与检索]：聚类

General Terms: Algorithms, Theory, Performance

通用术语：算法、理论、性能

Additional Key Words and Phrases: DBSCAN, density-based clustering, hopcroft hard, algorithms, computational geometry

其他关键词和短语：DBSCAN、基于密度的聚类、Hopcroft困难问题、算法、计算几何

## ACM Reference format:

## ACM引用格式：

Junhao Gan and Yufei Tao. 2017. On the Hardness and Approximation of Euclidean DBSCAN. ACM Trans. Database Syst. 42, 3, Article 14 (July 2017), 45 pages.

甘俊豪和陶宇飞。2017年。欧几里得DBSCAN算法的复杂度与近似求解。《ACM数据库系统汇刊》42卷，第3期，文章编号14（2017年7月），45页。

https://doi.org/10.1145/3083897

## 1 INTRODUCTION

## 1 引言

Density-based clustering is one of the most fundamental topics in data mining. Given a set $P$ of $n$ points in $d$ -dimensional space ${\mathbb{R}}^{d}$ ,the objective is to group the points of $P$ into subsets-called clusters-such that any two clusters are separated by "sparse regions." Figure 1 shows two classic examples taken from Ester et al. (1996): the left one contains four snake-shaped clusters, while the right one contains three clusters together with some noise. The main advantage of density-based clustering (over methods such as $k$ -means) is its capability of discovering clusters with arbitrary shapes (while $k$ -means typically returns ball-like clusters).

基于密度的聚类是数据挖掘中最基础的主题之一。给定$d$维空间${\mathbb{R}}^{d}$中的$n$个点的集合$P$，目标是将$P$中的点分组为子集（称为聚类），使得任意两个聚类被“稀疏区域”分隔开。图1展示了Ester等人（1996年）给出的两个经典示例：左边的示例包含四个蛇形聚类，而右边的示例包含三个聚类以及一些噪声点。基于密度的聚类（相对于$k$ -均值等方法）的主要优点是它能够发现任意形状的聚类（而$k$ -均值通常返回球形聚类）。

---

<!-- Footnote -->

Authors' addresses: J. Gan, School of Information Technology and Electrical Engineering, University of Queensland, St Lucia, 4067, Brisbane, Australia; email: j.gan@uq.edu.au; Y. Tao, Department of Computer Science and Engineering, Chinese University of Hong Kong, Sha Tin, New Territories, Hong Kong; email: taoyf@cse.cuhk.edu.hk.

作者地址：甘俊豪，澳大利亚布里斯班圣卢西亚区昆士兰大学信息技术与电气工程学院，邮编4067；电子邮件：j.gan@uq.edu.au；陶宇飞，中国香港新界沙田区香港中文大学计算机科学与工程系；电子邮件：taoyf@cse.cuhk.edu.hk。

Permission to make digital or hard copies of part or all of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies show this notice on the first page or initial screen of a display along with the full citation. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, to republish, to post on servers, to redistribute to lists, or to use any component of this work in other works requires prior specific permission and/or a fee. Permissions may be requested from Publications Dept., ACM, Inc., 2 Penn Plaza, Suite 701, New York, NY 10121-0701 USA, fax + 1 (212) 869-0481, or permissions@acm.org.

允许个人或课堂使用本作品的部分或全部内容制作数字或硬拷贝，无需付费，但前提是不得出于盈利或商业利益的目的制作或分发拷贝，并且拷贝应在第一页或显示的初始屏幕上显示此通知以及完整的引用信息。必须尊重本作品中除ACM之外的其他所有者的版权。允许进行带引用的摘要。否则，如需复制、重新发布、发布到服务器、分发给列表或在其他作品中使用本作品的任何组件，则需要事先获得特定许可和/或支付费用。许可申请可向美国纽约州纽约市第2宾夕法尼亚广场701室的ACM公司出版部提出，传真：+1 (212) 869 - 0481，或发送电子邮件至permissions@acm.org。

© 2017 ACM 0362-5915/2017/07-ART14 \$15.00

© 2017 美国计算机协会（ACM） 0362 - 5915/2017/07 - ART14 售价 15.00 美元

https://doi.org/10.1145/3083897

<!-- Footnote -->

---

<!-- Media -->

<img src="https://cdn.noedgeai.com/0195c91b-4734-77b1-933f-3c7cb33c7bfe_1.jpg?x=511&y=270&w=543&h=296&r=0"/>

Fig. 1. Examples of density-based clustering from Ester et al. (1996).

图 1. 埃斯特（Ester）等人（1996 年）提出的基于密度的聚类示例。

<!-- Media -->

Density-based clustering can be achieved using a variety of approaches, which differ mainly in their (i) definitions of "dense/sparse regions," and (ii) criteria of how dense regions should be connected to form clusters. In this article, we concentrate on DBSCAN, which is an approach invented by Ester et al. (1996), and received the test-of-time award in KDD'14. DBSCAN characterizes "density/sparsity" by resorting to two parameters:

基于密度的聚类可以通过多种方法实现，这些方法的主要区别在于：（i）对“密集/稀疏区域”的定义；（ii）密集区域应如何连接以形成聚类的标准。在本文中，我们主要关注 DBSCAN 算法，这是埃斯特等人（1996 年）发明的一种方法，并在 2014 年的知识发现与数据挖掘会议（KDD'14）上获得了时间检验奖。DBSCAN 算法通过两个参数来描述“密度/稀疏性”：

$- \epsilon$ : a positive real value;

$- \epsilon$ ：一个正实数值；

-MinPts: a small positive constant integer.

-最小点数（MinPts）：一个小的正整数常量。

Let $B\left( {p,\epsilon }\right)$ be the $d$ -dimensional ball centered at point $p$ with radius $\epsilon$ ,where the distance metric is Euclidean distance. $B\left( {p,\epsilon }\right)$ is "dense" if it covers at least MinPts points of $P$ .

设 $B\left( {p,\epsilon }\right)$ 是以点 $p$ 为中心、半径为 $\epsilon$ 的 $d$ 维球，其中距离度量采用欧几里得距离。如果 $B\left( {p,\epsilon }\right)$ 至少覆盖了 $P$ 中的最小点数（MinPts）个点，则称 $B\left( {p,\epsilon }\right)$ 是“密集的”。

DBSCAN forms clusters based on the following rationale. If $B\left( {p,\epsilon }\right)$ is dense,all the points in $B\left( {p,\epsilon }\right)$ should be added to the same cluster as $p$ . This creates a "chained effect": whenever a new point ${p}^{\prime }$ with a dense $B\left( {{p}^{\prime },\epsilon }\right)$ is added to the cluster of $p$ ,all the points in $B\left( {{p}^{\prime },\epsilon }\right)$ should also join the same cluster. The cluster of $p$ continues to grow in this manner to the effect’s fullest extent.

DBSCAN 算法基于以下原理形成聚类。如果 $B\left( {p,\epsilon }\right)$ 是密集的，则 $B\left( {p,\epsilon }\right)$ 中的所有点都应与 $p$ 加入同一个聚类。这会产生一种“连锁效应”：每当一个具有密集 $B\left( {{p}^{\prime },\epsilon }\right)$ 的新点 ${p}^{\prime }$ 加入到 $p$ 所在的聚类中时，$B\left( {{p}^{\prime },\epsilon }\right)$ 中的所有点也应加入同一个聚类。$p$ 所在的聚类以这种方式持续增长，直至达到该效应的最大程度。

### 1.1 Previous Description of DBSCAN's Running Time

### 1.1 先前对 DBSCAN 算法运行时间的描述

The original DBSCAN algorithm of Ester et al. (1996) performs a region query for each point $p \in  P$ , which retrieves $B\left( {p,\epsilon }\right)$ . Regarding the running time,Ester et al. (1996) wrote:

埃斯特等人（1996 年）提出的原始 DBSCAN 算法会对每个点 $p \in  P$ 执行一次区域查询，以检索 $B\left( {p,\epsilon }\right)$ 。关于运行时间，埃斯特等人（1996 年）写道：

"The height an ${R}^{ * }$ -tree is $O\left( {\log n}\right)$ for a database of $n$ points in the worst case and a query with a "small" query region has to traverse only a limited number of paths in the ${R}^{ * }$ -tree. Since the Eps-Neighborhoods are expected to be small compared to the size of the whole data space, the average run time complexity of a single region query is $O\left( {\log n}\right)$ . For each of the points of the database,we have at most one region query. Thus,the average runtime complexity of DBSCAN is $O\left( {n\log n}\right)$ ."

“对于一个包含 $n$ 个点的数据库，在最坏情况下，${R}^{ * }$ -树的高度为 $O\left( {\log n}\right)$ ，并且使用‘小’查询区域的查询只需遍历 ${R}^{ * }$ -树中的有限路径。由于预期的邻域（Eps - Neighborhoods）与整个数据空间的大小相比是较小的，因此单个区域查询的平均运行时间复杂度为 $O\left( {\log n}\right)$ 。对于数据库中的每个点，我们最多进行一次区域查询。因此，DBSCAN 算法的平均运行时间复杂度为 $O\left( {n\log n}\right)$ 。”

The underlined statement lacks scientific rigor:

下划线部分的陈述缺乏科学严谨性：

-Consider a dataset where $\Omega \left( n\right)$ points coincide at the same location. No matter how small is $\epsilon$ ,for every such point $p,B\left( {p,\epsilon }\right)$ always covers $\Omega \left( n\right)$ points. Even just reporting the points inside $B\left( {p,\epsilon }\right)$ for all such $p$ already takes $\Theta \left( {n}^{2}\right)$ time-this is true regardless of how good is the underlying ${\mathrm{R}}^{ * }$ -tree or any other index deployed.

- 考虑一个数据集，其中 $\Omega \left( n\right)$ 个点位于同一位置。无论 $\epsilon$ 有多小，对于每个这样的点，$p,B\left( {p,\epsilon }\right)$ 总是覆盖 $\Omega \left( n\right)$ 个点。即使只是报告所有这些 $p$ 对应的 $B\left( {p,\epsilon }\right)$ 内的点，也已经需要 $\Theta \left( {n}^{2}\right)$ 的时间——无论底层的 ${\mathrm{R}}^{ * }$ -树或其他任何使用的索引有多好，这都是成立的。

- The notion of "average runtime complexity" in the statement does not seem to follow any of the standard definitions in computer science (see,for example,Wikipedia ${}^{1}$ ). There was no clarification on the mathematical meaning of this notion in Ester et al. (1996), and neither was there a proof on the claimed complexity. In fact, it would have been a great result if an $O\left( {n\log n}\right)$ bound could indeed be proved under any of those definitions.

- 陈述中“平均运行时间复杂度”的概念似乎不符合计算机科学中的任何标准定义（例如，参见维基百科 ${}^{1}$ ）。埃斯特等人（1996 年）没有对这个概念的数学含义进行澄清，也没有对所声称的复杂度进行证明。事实上，如果确实能在任何这些定义下证明 $O\left( {n\log n}\right)$ 界，那将是一个了不起的结果。

The " $O\left( {n\log n}\right)$ average runtime complexity" has often been re-stated with fuzzy or even no description of the accompanying conditions. A popular textbook (Han et al. 2012), for example, comments in Chapter 10.4.1:

“ $O\left( {n\log n}\right)$ 平均运行时间复杂度”经常被重新表述，但对伴随条件的描述模糊甚至没有描述。例如，一本流行的教科书（韩（Han）等人，2012 年）在第 10.4.1 章中评论道：

If a spatial index is used,the computational complexity of DBSCAN is $O\left( {n\log n}\right)$ , where $n$ is the number of database objects. Otherwise,the complexity is $O\left( {n}^{2}\right)$ .

如果使用空间索引，DBSCAN的计算复杂度为$O\left( {n\log n}\right)$ ，其中$n$ 是数据库对象的数量。否则，复杂度为$O\left( {n}^{2}\right)$ 。

Similar statements have appeared in many papers: (Böhm et al. 2004) (Section 3.1), Chaoji et al. (2008) (Section 2), Ester (2013) (Chapter 5, Section 2), Klusch et al. (2003) (Section 2), Lu et al. (2011) (Section 5.4), Milenova and Campos (2002) (Section 1), Patwary et al. (2012) (Section 2), Sheikholeslami et al. (2000) (Section 3.3), Wang et al. (1997) (Section 2.2.3), Wen et al. (2002) (Section 5.2),mentioning just 10 papers. Several works have even utilized the $O\left( {n\log n}\right)$ bound as a building-brick lemma to derive new "results" incorrectly: see Section D.1 of Li et al. (2010), Section 3.2 of Pei et al. (2006), and Section 5.2 of Roy and Bhattacharyya (2005)).

类似的表述出现在许多论文中：(Böhm等人，2004年)（第3.1节）、Chaoji等人（2008年）（第2节）、Ester（2013年）（第5章，第2节）、Klusch等人（2003年）（第2节）、Lu等人（2011年）（第5.4节）、Milenova和Campos（2002年）（第1节）、Patwary等人（2012年）（第2节）、Sheikholeslami等人（2000年）（第3.3节）、Wang等人（1997年）（第2.2.3节）、Wen等人（2002年）（第5.2节），这里仅提及10篇论文。有几项研究甚至将$O\left( {n\log n}\right)$ 界作为一个基本引理来错误地推导新的“结果”：参见Li等人（2010年）的D.1节、Pei等人（2006年）的第3.2节以及Roy和Bhattacharyya（2005年）的第5.2节。

Gunawan (2013) also showed that all of the subsequently improved versions of the original DBSCAN algorithm either do not compute the precise DBSCAN result (e.g., see Borah and Bhattacharyya (2004),Liu (2006),and Tsai and Wu (2009)),or still suffer from $O\left( {n}^{2}\right)$ running time (Mahran and Mahar 2008). As a partial remedy, he developed a new 2D algorithm which truly runs in $O\left( {n\log n}\right)$ time,without assuming any indexes.

Gunawan（2013年）还表明，原始DBSCAN算法所有后续改进版本要么无法计算出精确的DBSCAN结果（例如，参见Borah和Bhattacharyya（2004年）、Liu（2006年）以及Tsai和Wu（2009年）），要么仍然受限于$O\left( {n}^{2}\right)$ 的运行时间（Mahran和Mahar，2008年）。作为部分补救措施，他开发了一种新的二维算法，该算法在不假设任何索引的情况下，真正能在$O\left( {n\log n}\right)$ 时间内运行。

### 1.2 Our Contributions

### 1.2 我们的贡献

This article was motivated by two questions:

本文受两个问题的启发：

-For $d \geq  3$ ,is it possible to design an algorithm that genuinely has $O\left( {n\log n}\right)$ time complexity? To make things easier,is it possible to achieve time complexity $O\left( {n{\log }^{c}n}\right)$ even for some very large constant $c$ ?

- 对于$d \geq  3$ ，是否有可能设计出一种真正具有$O\left( {n\log n}\right)$ 时间复杂度的算法？为了简化问题，即使对于某个非常大的常数$c$ ，是否有可能达到$O\left( {n{\log }^{c}n}\right)$ 的时间复杂度？

- If the answer to the previous question is no, is it possible to achieve linear or near-linear running time by sacrificing the quality of clusters slightly, while still being able to give a strong guarantee on the quality?

- 如果上一个问题的答案是否定的，是否可以通过稍微牺牲聚类质量来实现线性或接近线性的运行时间，同时仍然能对质量给出强有力的保证？

We answer the above questions with the following contributions:

我们通过以下贡献回答上述问题：

(1) We prove that the DBSCAN problem (computing the clusters from scratch, without assuming an existing index) requires $\Omega \left( {n}^{4/3}\right)$ time to solve in $d \geq  3$ ,unless very significant breakthroughs-ones widely believed to be impossible-can be made in theoretical computer science. Note that ${n}^{4/3}$ is arbitrarily larger than $n{\log }^{c}n$ ,regardless of constant $c$ .

(1) 我们证明，DBSCAN问题（从头开始计算聚类，不假设已有索引）在$d \geq  3$ 中求解需要$\Omega \left( {n}^{4/3}\right)$ 时间，除非理论计算机科学能取得重大突破（人们普遍认为这是不可能的）。请注意，无论常数$c$ 为何值，${n}^{4/3}$ 都比$n{\log }^{c}n$ 大得多。

(2) We introduce a new concept called $\rho$ -approximate DBSCAN which comes with strong assurances in both quality and efficiency. For quality, its clustering result is guaranteed to be "sandwiched" between the results of DBSCAN obtained with parameters $\left( {\epsilon ,\text{MinPts}}\right)$ and $\left( {\epsilon \left( {1 + \rho }\right) ,\text{MinPts}}\right)$ ,respectively. For efficiency,we prove that $\rho$ -approximate DBSCAN can be solved in linear $O\left( n\right)$ expected time,for any $\epsilon$ ,arbitrarily small constant $\rho$ ,and in any fixed dimensionality $d$ .

(2) 我们引入了一个名为$\rho$ -近似DBSCAN的新概念，它在质量和效率方面都有强有力的保证。在质量方面，其聚类结果保证介于分别使用参数$\left( {\epsilon ,\text{MinPts}}\right)$ 和$\left( {\epsilon \left( {1 + \rho }\right) ,\text{MinPts}}\right)$ 得到的DBSCAN结果之间。在效率方面，我们证明对于任意$\epsilon$ 、任意小的常数$\rho$ 以及任意固定维度$d$ ，$\rho$ -近似DBSCAN可以在线性$O\left( n\right)$ 期望时间内求解。

---

<!-- Footnote -->

${}^{1}$ Https://en.wikipedia.org/wiki/Average-case_complexity.

${}^{1}$ 网址：https://en.wikipedia.org/wiki/Average-case_complexity。

<!-- Footnote -->

---

(3) We give a new algorithm that solves the exact DBSCAN problem in 2D space using $O\left( {n\log n}\right)$ time,but in a way substantially simpler than the solution of Gunawan (2013). The algorithm reveals an inherent geometric connection between (exact) DBSCAN and Delaunay graphs. The connection is of independent interests.

(3) 我们给出了一种新算法，该算法使用$O\left( {n\log n}\right)$ 时间解决二维空间中的精确DBSCAN问题，并且比Gunawan（2013年）的解决方案要简单得多。该算法揭示了（精确）DBSCAN与德劳内图（Delaunay graphs）之间的内在几何联系。这种联系具有独立的研究价值。

(4) We prove that the 2D exact DBSCAN problem can actually be settled in $O\left( n\right)$ time,provided that the $n$ data points have been sorted along each dimension. In other words,the "hardest" component of the problem turns out to be sorting the coordinates, whereas the clustering part is easy. Immediately, this implies that 2D DBSCAN can be settled in $o\left( {n\log n}\right)$ time when the coordinates are integers,by utilizing fast integer-sorting algorithms (Andersson et al. 1998; Han and Thorup 2002): (i) deterministically, we achieve $O\left( {n\log \log n}\right)$ time-improving the $O\left( {n\log n}\right)$ bound of Gunawan (2013); (ii) randomly, we achieve $O\left( {n\sqrt{\log \log n}}\right)$ time in expectation.

(4) 我们证明，只要 $n$ 个数据点已按每个维度排序，二维精确DBSCAN（基于密度的空间聚类应用噪声）问题实际上可以在 $O\left( n\right)$ 时间内解决。换句话说，该问题“最难”的部分结果是对坐标进行排序，而聚类部分则很容易。由此立即可以推出，当坐标为整数时，利用快速整数排序算法（Andersson等人，1998年；Han和Thorup，2002年），二维DBSCAN问题可以在 $o\left( {n\log n}\right)$ 时间内解决：(i) 确定性地，我们实现了 $O\left( {n\log \log n}\right)$ 时间复杂度，改进了Gunawan（2013年）的 $O\left( {n\log n}\right)$ 界；(ii) 随机地，我们期望达到 $O\left( {n\sqrt{\log \log n}}\right)$ 时间复杂度。

(5) We perform an extensive experimental evaluation to explore the situations in which the original DBSCAN is adequate,and the situations in which $\rho$ -approximate DBSCAN serves as a nice alternative. In a nutshell, when the input data is "realistic" and it suffices to play with small $\epsilon$ ,existing algorithms may be used to find precise DBSCAN clusters efficiently. However,their cost escalates rapidly with $\epsilon$ . The proposed $\rho$ -approximate version is fast for a much wider parameter range. The performance advantage of $\rho$ -approximate DB-SCAN is most significant when the clusters have varying densities. In that case, a suitable $\epsilon$ is decided by the sparsest cluster,and has to be large with respect to the densest cluster, thus causing region queries in that cluster to be expensive (this will be further explained in Section 6).

(5) 我们进行了广泛的实验评估，以探究原始DBSCAN算法适用的情况，以及 $\rho$ -近似DBSCAN算法可作为不错替代方案的情况。简而言之，当输入数据“符合实际情况”且使用较小的 $\epsilon$ 就足够时，可以使用现有算法高效地找到精确的DBSCAN聚类。然而，它们的成本会随着 $\epsilon$ 的增大而迅速增加。所提出的 $\rho$ -近似版本在更广泛的参数范围内速度更快。当聚类具有不同密度时，$\rho$ -近似DBSCAN算法的性能优势最为显著。在这种情况下，合适的 $\epsilon$ 由最稀疏的聚类决定，并且相对于最密集的聚类而言必须较大，从而导致该聚类中的区域查询成本较高（这将在第6节中进一步解释）。

A short version of this article appeared in SIGMOD'15 (Gan and Tao 2015). In terms of technical contents, the current article extends that preliminary work with Contributions 3 and 4. Furthermore, the article also features revamped experiments that carry out a more complete study of the behavior of various algorithms.

本文的简短版本发表于SIGMOD'15会议（Gan和Tao，2015年）。就技术内容而言，本文通过贡献3和贡献4扩展了那项初步工作。此外，本文还进行了改进后的实验，对各种算法的性能进行了更全面的研究。

### 1.3 Organization of the Article

### 1.3 文章结构

Section 2 reviews the previous work related to ours. Section 3 provides theoretical evidence on the computational hardness of DBSCAN, and presents a sub-quadratic algorithm for solving the problem exactly. Section 4 proposes $\rho$ -approximate DBSCAN,elaborates on our algorithm,and establishes its quality and efficiency guarantees. Section 5 presents new algorithms for solving the exact DBSCAN problem in 2D space. Section 6 discusses several issues related to the practical performance of different algorithms and implementations. Section 7 evaluates all the exact and approximation algorithms with extensive experimentation. Finally, Section 8 concludes the article with a summary of findings.

第2节回顾了与我们工作相关的先前研究。第3节提供了关于DBSCAN计算复杂度的理论证据，并提出了一种用于精确解决该问题的次二次算法。第4节提出了 $\rho$ -近似DBSCAN算法，详细阐述了我们的算法，并确立了其质量和效率保证。第5节提出了用于在二维空间中精确解决DBSCAN问题的新算法。第6节讨论了与不同算法和实现的实际性能相关的几个问题。第7节通过广泛的实验评估了所有精确算法和近似算法。最后，第8节总结了研究结果，结束本文。

## 2 RELATED WORK

## 2 相关工作

Section 2.1 reviews the DBSCAN definitions as set out by Ester et al. (1996). Section 2.2 describes the 2D algorithm in Gunawan (2013) that solves the problem genuinely in $O\left( {n\log n}\right)$ time. Section 2.3 points out several results from computational geometry which will be needed to prove the intractability of DBSCAN later.

第2.1节回顾了Ester等人（1996年）提出的DBSCAN定义。第2.2节描述了Gunawan（2013年）中真正能在 $O\left( {n\log n}\right)$ 时间内解决该问题的二维算法。第2.3节指出了计算几何中的几个结果，这些结果将用于后续证明DBSCAN问题的难解性。

### 2.1 Definitions

### 2.1 定义

As before,let $P$ be a set of $n$ points in $d$ -dimensional space ${\mathbb{R}}^{d}$ . Given two points $p,q \in  {\mathbb{R}}^{d}$ ,we denote by $\operatorname{dist}\left( {p,q}\right)$ the Euclidean distance between $p$ and $q$ . Denote by $B\left( {p,r}\right)$ the ball centered at a point $p \in  {\mathbb{R}}^{d}$ with radius $r$ . Remember that DBSCAN takes two parameters: $\epsilon$ and MinPts.

如前所述，设 $P$ 是 $d$ 维空间 ${\mathbb{R}}^{d}$ 中的 $n$ 个点的集合。给定两个点 $p,q \in  {\mathbb{R}}^{d}$ ，我们用 $\operatorname{dist}\left( {p,q}\right)$ 表示 $p$ 和 $q$ 之间的欧几里得距离。用 $B\left( {p,r}\right)$ 表示以点 $p \in  {\mathbb{R}}^{d}$ 为中心、半径为 $r$ 的球。请记住，DBSCAN算法采用两个参数： $\epsilon$ 和最小点数（MinPts）。

<!-- Media -->

<!-- figureText: ${o}_{10}$ ${o}_{1}$ ${o}_{8}$ 07 Og 。 ${o}_{18}$ ${o}_{13}$ - ${o}_{11}$ ${o}_{14}$ ${o}_{12}$ ${o}_{15}$ ${o}_{16}$ ${}^{ \bullet  }{o}_{17}$ -->

<img src="https://cdn.noedgeai.com/0195c91b-4734-77b1-933f-3c7cb33c7bfe_4.jpg?x=492&y=261&w=584&h=448&r=0"/>

Fig. 2. An example dataset (the two circles have radius $\epsilon ;$ MinPts $= 4$ ).

图2. 一个示例数据集（两个圆的半径为 $\epsilon ;$ ，最小点数为 $= 4$ ）。

<!-- Media -->

Definition 2.1. A point $p \in  P$ is a core point if $B\left( {p,\epsilon }\right)$ covers at least MinPts points of $P$ (including $p$ itself).

定义2.1. 如果 $B\left( {p,\epsilon }\right)$ 覆盖了 $P$ 中的至少最小点数（MinPts）个点（包括 $p$ 本身），则点 $p \in  P$ 是一个核心点。

If $p$ is not a core point,it is said to be a non-core point. To illustrate,suppose that $P$ is the set of points in Figure 2,where MinPts $= 4$ and the two circles have radius $\epsilon$ . Core points are shown in black, and non-core points in white.

如果$p$不是核心点，则称其为非核心点。为了说明这一点，假设$P$是图2中的点集，其中最小点数为$= 4$，两个圆的半径为$\epsilon$。核心点用黑色表示，非核心点用白色表示。

Definition 2.2. A point $q \in  P$ is density-reachable from $p \in  P$ if there is a sequence of points ${p}_{1},{p}_{2},\ldots ,{p}_{t} \in  P$ (for some integer $t \geq  2$ ) such that

定义2.2。如果存在一个点序列${p}_{1},{p}_{2},\ldots ,{p}_{t} \in  P$（对于某个整数$t \geq  2$）使得点$q \in  P$从$p \in  P$是密度可达的。

$- {p}_{1} = p$ and ${p}_{t} = q$ ,

$- {p}_{1} = p$和${p}_{t} = q$。

$- {p}_{1},{p}_{2},\ldots ,{p}_{t - 1}$ are core points,

$- {p}_{1},{p}_{2},\ldots ,{p}_{t - 1}$是核心点。

$- {p}_{i + 1} \in  B\left( {{p}_{i},\epsilon }\right)$ for each $i \in  \left\lbrack  {1,t - 1}\right\rbrack$ .

对于每个$i \in  \left\lbrack  {1,t - 1}\right\rbrack$，有$- {p}_{i + 1} \in  B\left( {{p}_{i},\epsilon }\right)$。

Note that points $p$ and $q$ do not need to be different. In Figure 2,for example, ${o}_{1}$ is density-reachable from itself; ${o}_{10}$ is density-reachable from ${o}_{1}$ and from ${o}_{3}$ (through the sequence ${o}_{3},{o}_{2},{o}_{1},{o}_{10})$ . On the other hand, ${o}_{11}$ is not density-reachable from ${o}_{10}$ (recall that ${o}_{10}$ is not a core point).

注意，点$p$和$q$不必不同。例如，在图2中，${o}_{1}$从自身是密度可达的；${o}_{10}$从${o}_{1}$和${o}_{3}$是密度可达的（通过序列${o}_{3},{o}_{2},{o}_{1},{o}_{10})$）。另一方面，${o}_{11}$从${o}_{10}$不是密度可达的（回想一下，${o}_{10}$不是核心点）。

Definition 2.3. A cluster $C$ is a non-empty subset of $P$ such that

定义2.3。一个簇$C$是$P$的一个非空子集，使得

-(Maximality) If a core point $p \in  C$ ,then all the points density-reachable from $p$ also belong to $C$ .

 - （最大性）如果一个核心点$p \in  C$，那么从$p$密度可达的所有点也属于$C$。

-(Connectivity) For any points ${p}_{1},{p}_{2} \in  C$ ,there is a point $p \in  C$ such that both ${p}_{1}$ and ${p}_{2}$ are density-reachable from $p$ .

 - （连通性）对于任意点${p}_{1},{p}_{2} \in  C$，存在一个点$p \in  C$，使得${p}_{1}$和${p}_{2}$都从$p$是密度可达的。

Definition 2.3 implies that each cluster contains at least a core point (i.e., $p$ ). In Figure 2, $\left\{  {{o}_{1},{o}_{10}}\right\}$ is not a cluster because it does not involve all the points density-reachable from ${o}_{1}$ . On the other hand, $\left\{  {{o}_{1},{o}_{2},{o}_{3},\ldots ,{o}_{10}}\right\}$ is a cluster.

定义2.3意味着每个簇至少包含一个核心点（即$p$）。在图2中，$\left\{  {{o}_{1},{o}_{10}}\right\}$不是一个簇，因为它不包含从${o}_{1}$密度可达的所有点。另一方面，$\left\{  {{o}_{1},{o}_{2},{o}_{3},\ldots ,{o}_{10}}\right\}$是一个簇。

Ester et al. (1996) gave a nice proof that $P$ has a unique set of clusters,which gives rise to

埃斯特等人（1996年）给出了一个很好的证明，即$P$有唯一的簇集，这引出了

Problem 1. The DBSCAN problem is to find the unique set $\mathcal{C}$ of clusters of $P$ .

问题1。DBSCAN（基于密度的空间聚类应用于噪声，Density-Based Spatial Clustering of Applications with Noise）问题是找到$P$的唯一簇集$\mathcal{C}$。

Given the input $P$ in Figure 2,the problem should output two clusters: ${C}_{1} = \left\{  {{o}_{1},{o}_{2},\ldots ,{o}_{10}}\right\}$ and ${C}_{2} = \left\{  {{o}_{10},{o}_{11},\ldots ,{o}_{17}}\right\}  .$

给定图2中的输入$P$，该问题应输出两个簇：${C}_{1} = \left\{  {{o}_{1},{o}_{2},\ldots ,{o}_{10}}\right\}$和${C}_{2} = \left\{  {{o}_{10},{o}_{11},\ldots ,{o}_{17}}\right\}  .$

Remark. A cluster can contain both core and non-core points. Any non-core point $p$ in a cluster is called a border point. Some points may not belong to any clusters at all; they are called noise points. In Figure 2, ${o}_{10}$ is a border point,while ${o}_{18}$ is noise.

注：一个簇可以同时包含核心点和非核心点。簇中的任何非核心点$p$称为边界点。有些点可能根本不属于任何簇；它们被称为噪声点。在图2中，${o}_{10}$是一个边界点，而${o}_{18}$是噪声点。

<!-- Media -->

<!-- figureText: o10 ${C}_{4}$ ${o}_{0}$ ... (b) Graph $G$ (c) $\epsilon$ -neighbor cells (in gray) of the cell of ${o}_{10}$ 。 ${}_{1}^{1}{c}_{3}$ $\epsilon /\sqrt{2}$ $\epsilon /\sqrt{2}$ (a) Core cells are shown in gray -->

<img src="https://cdn.noedgeai.com/0195c91b-4734-77b1-933f-3c7cb33c7bfe_5.jpg?x=148&y=257&w=1269&h=498&r=0"/>

Fig. 3. DBSCAN with a grid $\left( {\text{ MinPts } = 4}\right)$ .

图3. 带有网格$\left( {\text{ MinPts } = 4}\right)$的DBSCAN算法。

<!-- Media -->

The clusters in $\mathcal{C}$ are not necessarily disjoint (e.g., ${o}_{10}$ belongs to both ${C}_{1}$ and ${C}_{2}$ in Figure 2). In general,if a point $p$ appears in more than one cluster in $\mathcal{C}$ ,then $p$ must be a border point (see Lemma 2 of Ester et al. (1996)). In other words, a core point always belongs to a unique cluster.

$\mathcal{C}$中的簇不一定是不相交的（例如，图2中${o}_{10}$同时属于${C}_{1}$和${C}_{2}$）。一般来说，如果一个点$p$出现在$\mathcal{C}$中的多个簇中，那么$p$一定是一个边界点（见Ester等人（1996）的引理2）。换句话说，一个核心点总是属于一个唯一的簇。

### 2.2 The 2D Algorithm with Genuine $O\left( {n\log n}\right)$ Time

### 2.2 具有真正$O\left( {n\log n}\right)$时间复杂度的二维算法

Next, we explain in detail the algorithm of Gunawan (2013), which solves the DBSCAN problem in $2\mathrm{D}$ space in $O\left( {n\log n}\right)$ time. The algorithm imposes an arbitrary grid $T$ in the data space ${\mathbb{R}}^{2}$ ,where each cell of $T$ is a $\left( {\epsilon /\sqrt{2}}\right)  \times  \left( {\epsilon /\sqrt{2}}\right)$ square. Without loss of generality,we assume that no point of $P$ falls on any boundary line of $T$ (otherwise,move $T$ infinitesimally to make this assumption hold). Figure 3(a) shows a grid on the data of Figure 2. Note that any two points in the same cell are at most distance $\epsilon$ apart. A cell $c$ of $T$ is non-empty if it contains at least one point of $P$ ; otherwise, $c$ is empty. Clearly,there can be at most $n$ non-empty cells.

接下来，我们详细解释Gunawan（2013）提出的算法，该算法能在$O\left( {n\log n}\right)$时间内解决$2\mathrm{D}$空间中的DBSCAN问题。该算法在数据空间${\mathbb{R}}^{2}$中施加一个任意网格$T$，其中$T$的每个单元格都是一个$\left( {\epsilon /\sqrt{2}}\right)  \times  \left( {\epsilon /\sqrt{2}}\right)$的正方形。不失一般性，我们假设$P$中的任何点都不会落在$T$的任何边界线上（否则，将$T$进行微小移动以使该假设成立）。图3（a）展示了图2数据上的一个网格。注意，同一单元格中的任意两点之间的距离至多为$\epsilon$。如果$T$的一个单元格$c$包含$P$中的至少一个点，则该单元格非空；否则，$c$为空。显然，非空单元格最多有$n$个。

The algorithm then launches a labeling process to decide for each point $p \in  P$ whether $p$ is core or non-core. Denote by $P\left( c\right)$ the set of points of $P$ covered by $c$ . A cell $c$ is a core cell if $P\left( c\right)$ contains at least one core point. Denote by ${S}_{\text{core }}$ the set of core cells in $T$ . In Figure 3(a) where $\operatorname{MinPts} = 4$ , there are six core cells as shown in gray (core points are in black, and non-core points in white). Let $G = \left( {V,E}\right)$ be a graph defined as follows:

然后，该算法启动一个标记过程，以确定每个点$p \in  P$是核心点还是非核心点。用$P\left( c\right)$表示$c$所覆盖的$P$中的点集。如果$P\left( c\right)$包含至少一个核心点，则单元格$c$是一个核心单元格。用${S}_{\text{core }}$表示$T$中的核心单元格集。在图3（a）中，当$\operatorname{MinPts} = 4$时，有六个核心单元格，如灰色所示（核心点为黑色，非核心点为白色）。定义一个图$G = \left( {V,E}\right)$如下：

- Each vertex in $V$ corresponds to a distinct core cell in ${S}_{\text{core }}$ .

- $V$中的每个顶点对应于${S}_{\text{core }}$中的一个不同的核心单元格。

- Given two different cells ${c}_{1},{c}_{2} \in  {S}_{\text{core }},E$ contains an edge between ${c}_{1}$ and ${c}_{2}$ if and only if there exist core points ${p}_{1} \in  P\left( {c}_{1}\right)$ and ${p}_{2} \in  P\left( {c}_{2}\right)$ such that $\operatorname{dist}\left( {{p}_{1},{p}_{2}}\right)  \leq  \epsilon$ .

- 给定两个不同的单元格${c}_{1},{c}_{2} \in  {S}_{\text{core }},E$，当且仅当存在核心点${p}_{1} \in  P\left( {c}_{1}\right)$和${p}_{2} \in  P\left( {c}_{2}\right)$使得$\operatorname{dist}\left( {{p}_{1},{p}_{2}}\right)  \leq  \epsilon$时，${c}_{1},{c}_{2} \in  {S}_{\text{core }},E$中存在一条连接${c}_{1}$和${c}_{2}$的边。

Figure 3(b) shows the $G$ for Figure 3(a) (note that there is no edge between cells ${c}_{4}$ and ${c}_{6}$ ). The algorithm then proceeds by finding all the connected components of $G$ . Let $k$ be the number of connected components, ${V}_{i}\left( {1 \leq  i \leq  k}\right)$ be the set of vertices in the $i$ -th connected component, and $P\left( {V}_{i}\right)$ be the set of core points covered by the cells of ${V}_{i}$ . Then:

图3（b）展示了图3（a）对应的$G$（注意，单元格${c}_{4}$和${c}_{6}$之间没有边）。然后，该算法通过找出$G$的所有连通分量来继续进行。设$k$为连通分量的数量，${V}_{i}\left( {1 \leq  i \leq  k}\right)$为第$i$个连通分量中的顶点集，$P\left( {V}_{i}\right)$为${V}_{i}$中的单元格所覆盖的核心点集。那么：

LEMMA 2.4 (GUNAWAN (2013)). The number $k$ is also the number of clusters in P. Furthermore, $P\left( {V}_{i}\right) \left( {1 \leq  i \leq  k}\right)$ is exactly the set of core points in the $i$ -th cluster.

引理2.4（GUNAWAN（2013））。数量$k$也是集合P中簇的数量。此外，$P\left( {V}_{i}\right) \left( {1 \leq  i \leq  k}\right)$恰好是第$i$个簇中的核心点集。

In Figure 3(b), $k = 2$ ,and ${V}_{1} = \left\{  {{c}_{1},{c}_{2},{c}_{3}}\right\}  ,{V}_{2} = \left\{  {{c}_{4},{c}_{5},{c}_{6}}\right\}$ . It is easy to verify the correctness of Lemma 2.4 on this example.

在图3(b)中，$k = 2$ ，以及${V}_{1} = \left\{  {{c}_{1},{c}_{2},{c}_{3}}\right\}  ,{V}_{2} = \left\{  {{c}_{4},{c}_{5},{c}_{6}}\right\}$ 。在此示例中，很容易验证引理2.4的正确性。

<!-- Media -->

<!-- figureText: closest pair (b) USEC (c) Hopcroft (a) BCP -->

<img src="https://cdn.noedgeai.com/0195c91b-4734-77b1-933f-3c7cb33c7bfe_6.jpg?x=145&y=261&w=1274&h=349&r=0"/>

Fig. 4. Three relevant geometric problems.

图4. 三个相关的几何问题。

<!-- Media -->

Labeling Process. Let ${c}_{1}$ and ${c}_{2}$ be two different cells in $T$ . They are $\epsilon$ -neighbors of each other if the minimum distance between them is less than $\epsilon$ . Figure 3(c) shows in gray all the $\epsilon$ -neighbor cells of the cell covering ${o}_{10}$ . It is easy to see that each cell has at most ${21\epsilon }$ -neighbors. If a non-empty cell $c$ contains at least MinPts points,then all those points must be core points.

标记过程。设${c}_{1}$ 和${c}_{2}$ 为$T$ 中的两个不同单元格。如果它们之间的最小距离小于$\epsilon$ ，则它们互为$\epsilon$ -邻域单元格。图3(c)以灰色显示了覆盖${o}_{10}$ 的单元格的所有$\epsilon$ -邻域单元格。很容易看出，每个单元格最多有${21\epsilon }$ 个邻域单元格。如果一个非空单元格$c$ 至少包含MinPts个点，那么所有这些点必定都是核心点。

Now consider a cell $c$ with $\left| {P\left( c\right) }\right|  <$ MinPts. Each point $p \in  P\left( c\right)$ may or may not be a core point. To find out,the algorithm simply calculates the distances between $p$ and all the points covered by each of the $\epsilon$ -neighbor cells of $c$ . This allows us to know exactly the size of $\left| {B\left( {p,\epsilon }\right) }\right|$ ,and hence, whether $p$ is core or non-core. For example,in Figure 3(c),for $p = {o}_{10}$ ,we calculate the distance between ${o}_{10}$ and all the points in the gray cells to find out that ${o}_{10}$ is a non-core point.

现在考虑一个包含$\left| {P\left( c\right) }\right|  <$ 个MinPts的单元格$c$ 。每个点$p \in  P\left( c\right)$ 可能是也可能不是核心点。为了确定这一点，算法只需计算$p$ 与$c$ 的每个$\epsilon$ -邻域单元格所覆盖的所有点之间的距离。这样我们就能确切知道$\left| {B\left( {p,\epsilon }\right) }\right|$ 的大小，从而知道$p$ 是核心点还是非核心点。例如，在图3(c)中，对于$p = {o}_{10}$ ，我们计算${o}_{10}$ 与灰色单元格中所有点之间的距离，从而确定${o}_{10}$ 是非核心点。

Computation of $G$ . Fix a core cell ${c}_{1}$ . We will explain how to obtain the edges incident on ${c}_{1}$ in $E$ . Let ${c}_{2}$ be a core cell that is an $\epsilon$ -neighbor of ${c}_{1}$ . For each core point $p \in  P\left( {c}_{1}\right)$ ,we find the core point ${p}^{\prime } \in  {c}_{2}$ that is the nearest to $p$ . If $\operatorname{dist}\left( {p,{p}^{\prime }}\right)  \leq  \epsilon$ ,an edge $\left( {{c}_{1},{c}_{2}}\right)$ is added to $G$ . On the other hand,if all such $p \in  P\left( {c}_{1}\right)$ have been tried but still no edge has been created,we conclude that $E$ has no edge between ${c}_{1},{c}_{2}$ .

$G$ 的计算。固定一个核心单元格${c}_{1}$ 。我们将解释如何在$E$ 中获取与${c}_{1}$ 相关联的边。设${c}_{2}$ 是${c}_{1}$ 的一个$\epsilon$ -邻域核心单元格。对于每个核心点$p \in  P\left( {c}_{1}\right)$ ，我们找到距离$p$ 最近的核心点${p}^{\prime } \in  {c}_{2}$ 。如果$\operatorname{dist}\left( {p,{p}^{\prime }}\right)  \leq  \epsilon$ ，则将一条边$\left( {{c}_{1},{c}_{2}}\right)$ 添加到$G$ 中。另一方面，如果已经尝试了所有这样的$p \in  P\left( {c}_{1}\right)$ ，但仍然没有创建边，我们就得出结论：在$E$ 中，${c}_{1},{c}_{2}$ 之间没有边。

As a corollary of the above,each core cell ${c}_{1}$ has $O\left( 1\right)$ incident edges in $E$ (because it has $O\left( 1\right)$ $\epsilon$ -neighbors). In other words, $E$ has only a linear number $O\left( n\right)$ of edges.

根据上述内容的推论，每个核心单元格${c}_{1}$ 在$E$ 中有$O\left( 1\right)$ 条关联边（因为它有$O\left( 1\right)$ 个$\epsilon$ -邻域单元格）。换句话说，$E$ 只有线性数量$O\left( n\right)$ 的边。

Assigning Border Points. Recall that each $P\left( {V}_{i}\right) \left( {1 \leq  i \leq  k}\right)$ includes only the core points in the $i$ -th cluster of $P$ . It is still necessary to assign each non-core point $q$ (i.e.,border point) to the appropriate clusters. The principle of doing so is simple: if $p$ is a core point and $\operatorname{dist}\left( {p,q}\right)  \leq  \epsilon$ ,then $q$ should be added to the (unique) cluster of $p$ . To find all such core points $p$ ,Gunawan (2013) adopted the following simple algorithm. Let $c$ be the cell where $q$ lies. For each $\epsilon$ -neighbor cell ${c}^{\prime }$ of $c$ ,simply calculate the distances from $q$ to all the core points in ${c}^{\prime }$ .

分配边界点。回顾一下，每个$P\left( {V}_{i}\right) \left( {1 \leq  i \leq  k}\right)$仅包含$P$中第$i$个簇的核心点。仍然需要将每个非核心点$q$（即边界点）分配到合适的簇中。这样做的原则很简单：如果$p$是一个核心点且$\operatorname{dist}\left( {p,q}\right)  \leq  \epsilon$，那么$q$应该被添加到$p$所在的（唯一）簇中。为了找到所有这样的核心点$p$，古纳万（Gunawan，2013年）采用了以下简单算法。设$c$是$q$所在的单元格。对于$c$的每个$\epsilon$邻接单元格${c}^{\prime }$，只需计算从$q$到${c}^{\prime }$中所有核心点的距离。

Running Time. Gunawan (2013) showed that,other than the computation of $G$ ,the rest of the algorithm runs in $O\left( {\text{MinPts } \cdot  n}\right)  = O\left( n\right)$ expected time or $O\left( {n\log n}\right)$ worst-case time. The computation of $G$ requires $O\left( n\right)$ nearest neighbor queries,each of which can be answered in $O\left( {\log n}\right)$ time after building a Voronoi diagram for each core cell. Therefore, the overall execution time is bounded by $O\left( {n\log n}\right)$ .

运行时间。古纳万（Gunawan，2013年）表明，除了计算$G$之外，算法的其余部分的期望运行时间为$O\left( {\text{MinPts } \cdot  n}\right)  = O\left( n\right)$，最坏情况下的运行时间为$O\left( {n\log n}\right)$。计算$G$需要进行$O\left( n\right)$次最近邻查询，在为每个核心单元格构建一个沃罗诺伊图（Voronoi diagram）之后，每次查询可以在$O\left( {\log n}\right)$时间内得到答案。因此，总体执行时间受限于$O\left( {n\log n}\right)$。

### 2.3 Some Geometric Results

### 2.3 一些几何结果

Bichromatic Closest Pair (BCP). Let ${P}_{1},{P}_{2}$ be two sets of points in ${\mathbb{R}}^{d}$ for some constant $d$ . Set ${m}_{1} = \left| {P}_{1}\right|$ and ${m}_{2} = \left| {P}_{2}\right|$ . The goal of the BCP problem is to find a pair of points $\left( {{p}_{1},{p}_{2}}\right)  \in  {P}_{1} \times  {P}_{2}$ with the smallest distance,namely, $\operatorname{dist}\left( {{p}_{1},{p}_{2}}\right)  \leq  \operatorname{dist}\left( {{p}_{1}^{\prime },{p}_{2}^{\prime }}\right)$ for any $\left( {{p}_{1}^{\prime },{p}_{2}^{\prime }}\right)  \in  {P}_{1} \times  {P}_{2}$ . Figure 4 shows the closest pair for a set of black points and a set of white points.

双色最近点对（Bichromatic Closest Pair，BCP）。对于某个常数$d$，设${P}_{1},{P}_{2}$是${\mathbb{R}}^{d}$中的两个点集。设${m}_{1} = \left| {P}_{1}\right|$和${m}_{2} = \left| {P}_{2}\right|$。BCP问题的目标是找到距离最小的一对点$\left( {{p}_{1},{p}_{2}}\right)  \in  {P}_{1} \times  {P}_{2}$，即对于任何$\left( {{p}_{1}^{\prime },{p}_{2}^{\prime }}\right)  \in  {P}_{1} \times  {P}_{2}$，有$\operatorname{dist}\left( {{p}_{1},{p}_{2}}\right)  \leq  \operatorname{dist}\left( {{p}_{1}^{\prime },{p}_{2}^{\prime }}\right)$。图4展示了一组黑点和一组白点的最近点对。

In 2D space,it is well known that BCP can be solved in $O\left( {{m}_{1}\log {m}_{1} + {m}_{2}\log {m}_{2}}\right)$ time. The problem is much more challenging for $d \geq  3$ ,for which currently the best result is due to Agarwal et al. (1991):

在二维空间中，众所周知，BCP问题可以在$O\left( {{m}_{1}\log {m}_{1} + {m}_{2}\log {m}_{2}}\right)$时间内解决。对于$d \geq  3$，这个问题更具挑战性，目前最好的结果归功于阿加瓦尔等人（Agarwal et al.，1991年）：

LEMMA 2.5 (AGARWAL ET AL. (1991)). For any fixed dimensionality $d \geq  4$ ,there is an algorithm solving the BCP problem in

引理2.5（阿加瓦尔等人（Agarwal et al.，1991年））。对于任何固定的维度$d \geq  4$，存在一种算法可以在

$$
O\left( {{\left( {m}_{1}{m}_{2}\right) }^{1 - \frac{1}{\lceil d/2\rceil  + 1} + {\delta }^{\prime }} + {m}_{1}\log {m}_{2} + {m}_{2}\log {m}_{1}}\right) 
$$

expected time,where ${\delta }^{\prime } > 0$ can be an arbitrarily small constant. For $d = 3$ ,the expected running time can be improved to

期望时间内解决BCP问题，其中${\delta }^{\prime } > 0$可以是任意小的常数。对于$d = 3$，期望运行时间可以改进为

$$
O\left( {{\left( {m}_{1}{m}_{2} \cdot  \log {m}_{1} \cdot  \log {m}_{2}\right) }^{2/3} + {m}_{1}{\log }^{2}{m}_{2} + {m}_{2}{\log }^{2}{m}_{1}}\right) ).
$$

Spherical Emptiness and Hopcroft. Let us now introduce the unit-spherical emptiness checking (USEC) problem:

球面空性与霍普克罗夫特问题。现在让我们介绍单位球面试空检查（Unit - Spherical Emptiness Checking，USEC）问题：

Let ${S}_{pt}$ be a set of points,and ${S}_{\text{ball }}$ be a set of balls with the same radius,all in data space ${\mathbb{R}}^{d}$ ,where the dimensionality $d$ is a constant. The objective of USEC is to determine whether there is a point of ${S}_{pt}$ that is covered by some ball in ${S}_{\text{ball }}$ .

设${S}_{pt}$是一个点集，${S}_{\text{ball }}$是一组具有相同半径的球，它们都在数据空间${\mathbb{R}}^{d}$中，其中维度$d$是一个常数。USEC问题的目标是确定${S}_{pt}$中是否存在一个点被${S}_{\text{ball }}$中的某个球覆盖。

For example, in Figure 4(b), the answer is yes.

例如，在图4（b）中，答案是肯定的。

Set $n = \left| {S}_{pt}\right|  + \left| {S}_{\text{ball }}\right|$ . In 3D space,the USEC problem can be solved in $O\left( {{n}^{4/3} \cdot  {\log }^{4/3}n}\right)$ expected time (Agarwal et al. 1991). Finding a 3D USEC algorithm with running time $o\left( {n}^{4/3}\right)$ is a big open problem in computational geometry, and is widely believed to be impossible (see Erickson (1995)).

设 $n = \left| {S}_{pt}\right|  + \left| {S}_{\text{ball }}\right|$ 。在三维空间中，通用最小外接圆（USEC）问题可以在 $O\left( {{n}^{4/3} \cdot  {\log }^{4/3}n}\right)$ 的期望时间内解决（阿加瓦尔等人，1991 年）。找到一个运行时间为 $o\left( {n}^{4/3}\right)$ 的三维通用最小外接圆算法是计算几何领域一个重大的开放性问题，并且人们普遍认为这是不可能的（见埃里克森（1995 年））。

Strong hardness results are known about USEC when the dimensionality $d$ is higher,owing to an established connection between the problem to the Hopcroft's problem:

当维数 $d$ 更高时，由于该问题与霍普克罗夫特问题之间存在已确立的联系，关于通用最小外接圆（USEC）已有很强的难解性结果：

Let ${S}_{pt}$ be a set of points,and ${S}_{\text{line }}$ be a set of lines,all in data space ${\mathbb{R}}^{2}$ (note that the dimensionality is always 2). The goal of the Hopcroft's problem is to determine whether there is a point in ${S}_{pt}$ that lies on some line of ${S}_{\text{line }}$ .

设 ${S}_{pt}$ 为一个点集， ${S}_{\text{line }}$ 为一个直线集，它们都位于数据空间 ${\mathbb{R}}^{2}$ 中（注意维数始终为 2）。霍普克罗夫特问题的目标是确定 ${S}_{pt}$ 中是否存在一个点位于 ${S}_{\text{line }}$ 中的某条直线上。

For example, in Figure 4(c), the answer is no.

例如，在图 4(c) 中，答案是否定的。

The Hopcroft’s problem can be settled in time slightly higher than $O\left( {n}^{4/3}\right)$ time (see Matousek (1993) for the precise bound),where $n = \left| {S}_{pt}\right|  + \left| {S}_{\text{line }}\right|$ . It is widely believed (Erickson 1995) that $\Omega \left( {n}^{4/3}\right)$ is a lower bound on how fast the problem can be solved. In fact,this lower bound has already been proved on a broad class of algorithms (Erickson 1996).

霍普克罗夫特问题可以在略高于 $O\left( {n}^{4/3}\right)$ 时间内解决（精确的界限见马托塞克（1993 年）），其中 $n = \left| {S}_{pt}\right|  + \left| {S}_{\text{line }}\right|$ 。人们普遍认为（埃里克森，1995 年） $\Omega \left( {n}^{4/3}\right)$ 是该问题可求解的最快时间下界。事实上，这个下界已经在一大类算法上得到了证明（埃里克森，1996 年）。

It turns out that the Hopcroft's problem is a key reason of difficulty for a large number of other problems (Erickson 1995). We say that a problem $X$ is Hopcroft hard if an algorithm solving $X$ in $o\left( {n}^{4/3}\right)$ time implies an algorithm solving the Hopcroft’s problem in $o\left( {n}^{4/3}\right)$ time. In other words,a lower bound $\Omega \left( {n}^{4/3}\right)$ on the time of solving the Hopcroft’s problem implies the same lower bound on $X$ .

事实证明，霍普克罗夫特问题（Hopcroft's problem）是导致大量其他问题难以解决的关键原因（埃里克森，1995 年）。如果一个能在$o\left( {n}^{4/3}\right)$时间内解决问题$X$的算法意味着存在一个能在$o\left( {n}^{4/3}\right)$时间内解决霍普克罗夫特问题的算法，我们就称问题$X$是霍普克罗夫特困难的。换句话说，解决霍普克罗夫特问题的时间下界$\Omega \left( {n}^{4/3}\right)$意味着问题$X$也有相同的时间下界。

Erickson (1996) proved the following relationship between USEC and the Hopcroft's problem:

埃里克森（1996 年）证明了无符号最短偶数圈问题（USEC）和霍普克罗夫特问题之间存在以下关系：

LEMMA 2.6 (ERICKSON (1996)). The USEC problem in any dimensionality $d \geq  5$ is Hopcroft hard.

引理 2.6（埃里克森（1996 年））。任意维度$d \geq  5$下的无符号最短偶数圈问题（USEC）都是霍普克罗夫特困难的。

## 3 DBSCAN IN DIMENSIONALITY 3 AND ABOVE

## 3 三维及更高维度下的基于密度的空间聚类应用（DBSCAN）

This section paves the way toward approximate DBSCAN, which is the topic of the next section. In Section 3.1, we establish the computational hardness of DBSCAN in practice via a novel reduction from the USEC problem (see Section 2.3). For practitioners that insist on applying this clustering method with the utmost accuracy, in Section 3.2, we present a new exact DBSCAN algorithm that terminates in a sub-quadratic time complexity.

本节为近似基于密度的空间聚类应用（DBSCAN）方法奠定基础，这也是下一节的主题。在 3.1 节中，我们通过从无符号最短偶数圈问题（USEC）进行一种新颖的归约（见 2.3 节），证明了在实际应用中基于密度的空间聚类应用（DBSCAN）问题的计算难度。对于坚持以最高精度应用这种聚类方法的从业者，在 3.2 节中，我们提出了一种新的精确基于密度的空间聚类应用（DBSCAN）算法，该算法的时间复杂度为亚二次的。

### 3.1 Hardness of DBSCAN

### 3.1 基于密度的空间聚类应用（DBSCAN）问题的难度

We will prove:

我们将证明：

THEOREM 3.1. The following statements are true about the DBSCAN problem:

定理 3.1。关于基于密度的空间聚类应用（DBSCAN）问题，以下陈述是正确的：

-It is Hopcroft hard in any dimensionality $d \geq  5$ . Namely,the problem requires $\Omega \left( {n}^{4/3}\right)$ time to solve,unless the Hopcroft problem can be settled in $o\left( {n}^{4/3}\right)$ time.

- 在任意维度$d \geq  5$下，它都是霍普克罗夫特困难的。即，除非霍普克罗夫特问题能在$o\left( {n}^{4/3}\right)$时间内解决，否则解决该问题需要$\Omega \left( {n}^{4/3}\right)$时间。

-When $d = 3$ (and hence, $d = 4$ ),the problem requires $\Omega \left( {n}^{4/3}\right)$ time to solve,unless the USEC problem can be settled in $o\left( {n}^{4/3}\right)$ time.

- 当$d = 3$（因此，$d = 4$）时，除非无符号最短偶数圈问题（USEC）能在$o\left( {n}^{4/3}\right)$时间内解决，否则解决该问题需要$\Omega \left( {n}^{4/3}\right)$时间。

As mentioned in Section 2.3, it is widely believed that neither the Hopcroft problem nor the USEC problem can be solved in $o\left( {n}^{4/3}\right)$ time-any such algorithm would be a celebrated breakthrough in theoretical computer science.

如 2.3 节所述，人们普遍认为霍普克罗夫特问题和无符号最短偶数圈问题（USEC）都无法在$o\left( {n}^{4/3}\right)$时间内解决——任何能解决这些问题的算法都将是理论计算机科学领域的一项重大突破。

Proof of Theorem 3.1. We observe a subtle connection between USEC and DBSCAN:

定理 3.1 的证明。我们观察到无符号最短偶数圈问题（USEC）和基于密度的空间聚类应用（DBSCAN）之间存在一种微妙的联系：

LEMMA 3.2. For any dimensionality $d$ ,if we can solve the DBSCAN problem in $T\left( n\right)$ time,then we can solve the USEC problem in $T\left( n\right)  + O\left( n\right)$ time.

引理 3.2。对于任意维度$d$，如果我们能在$T\left( n\right)$时间内解决基于密度的空间聚类应用（DBSCAN）问题，那么我们就能在$T\left( n\right)  + O\left( n\right)$时间内解决无符号最短偶数圈问题（USEC）。

Proof. Recall that the USEC problem is defined by a set ${S}_{pt}$ of points and a set ${S}_{\text{ball }}$ of balls with equal radii,both in ${\mathbb{R}}^{d}$ . Denote by $\mathcal{A}$ a DBSCAN algorithm in ${\mathbb{R}}^{d}$ that runs in $T\left( m\right)$ time on $m$ points. Next,we describe an algorithm that deploys $\mathcal{A}$ as a black box to solve the USEC problem in $T\left( n\right)  + O\left( n\right)$ time,where $n = \left| {S}_{pt}\right|  + \left| {S}_{\text{ball }}\right|$ .

证明。回顾一下，无符号最短偶数圈问题（USEC）是由一个点集${S}_{pt}$和一个等半径球集${S}_{\text{ball }}$定义的，它们都在${\mathbb{R}}^{d}$中。用$\mathcal{A}$表示一个在${\mathbb{R}}^{d}$中对$m$个点运行时间为$T\left( m\right)$的基于密度的空间聚类应用（DBSCAN）算法。接下来，我们描述一个将$\mathcal{A}$作为黑盒使用的算法，该算法能在$T\left( n\right)  + O\left( n\right)$时间内解决无符号最短偶数圈问题（USEC），其中$n = \left| {S}_{pt}\right|  + \left| {S}_{\text{ball }}\right|$。

Our algorithm is simple:

我们的算法很简单：

(1) Obtain $P$ ,which is the union of ${S}_{pt}$ and the set of centers of the balls in ${S}_{\text{ball }}$ .

(1) 得到$P$，它是${S}_{pt}$和${S}_{\text{ball }}$中球心集合的并集。

(2) Set $\epsilon$ to the identical radius of the balls in ${S}_{\text{ball }}$ .

(2) 将 $\epsilon$ 设置为 ${S}_{\text{ball }}$ 中球的相同半径。

(3) Run $\mathcal{A}$ to solve the DBSCAN problem on $P$ with this $\epsilon$ and $\operatorname{MinPts} = 1$ .

(3) 运行 $\mathcal{A}$ 以使用此 $\epsilon$ 和 $\operatorname{MinPts} = 1$ 解决 $P$ 上的 DBSCAN 问题。

(4) If any point in ${S}_{pt}$ and any center of ${S}_{\text{ball }}$ belong to the same cluster,then return yes for the USEC problem (namely,a point in ${S}_{pt}$ is covered by some ball in ${S}_{\text{ball }}$ ). Otherwise,return no.

(4) 如果 ${S}_{pt}$ 中的任何点和 ${S}_{\text{ball }}$ 中的任何中心属于同一聚类，则对于 USEC 问题返回“是”（即，${S}_{pt}$ 中的一个点被 ${S}_{\text{ball }}$ 中的某个球覆盖）。否则，返回“否”。

It is fundamental to implement the above algorithm in $T\left( n\right)  + O\left( n\right)$ time. Next,we prove its correctness.

在 $T\left( n\right)  + O\left( n\right)$ 时间内实现上述算法至关重要。接下来，我们证明其正确性。

Case 1: We return yes. We will show that in this case there is indeed a point of ${S}_{pt}$ that is covered by some ball in ${S}_{\text{ball }}$ .

情况 1：我们返回“是”。我们将证明在这种情况下，${S}_{pt}$ 中确实存在一个点被 ${S}_{\text{ball }}$ 中的某个球覆盖。

Recall that a yes return means a point $p \in  {S}_{pt}$ and the center $q$ of some ball in ${S}_{\text{ball }}$ have been placed in the same cluster,which we denote by $C$ . By connectivity of Definition 2.3,there exists a point $z \in  C$ such that both $p$ and $q$ are density-reachable from $z$ .

回顾一下，返回“是”意味着点 $p \in  {S}_{pt}$ 和 ${S}_{\text{ball }}$ 中某个球的中心 $q$ 已被置于同一聚类中，我们将该聚类记为 $C$。根据定义 2.3 的连通性，存在一个点 $z \in  C$，使得 $p$ 和 $q$ 都可从 $z$ 密度可达。

By setting $\operatorname{MinPts} = 1$ ,we ensure that all the points in $P$ are core points. In general,if a core point ${p}_{1}$ is density-reachable from ${p}_{2}$ (which by definition must be a core point),then ${p}_{2}$ is also density-reachable from ${p}_{1}$ (as can be verified by Definition 2.2). This means that $z$ is density-reachable from $p$ ,which-together with the fact that $q$ is density-reachable from $z$ -shows that $q$ is density-reachable from $p$ .

通过设置 $\operatorname{MinPts} = 1$，我们确保 $P$ 中的所有点都是核心点。一般来说，如果一个核心点 ${p}_{1}$ 可从 ${p}_{2}$ 密度可达（根据定义，${p}_{2}$ 必须是核心点），那么 ${p}_{2}$ 也可从 ${p}_{1}$ 密度可达（可由定义 2.2 验证）。这意味着 $z$ 可从 $p$ 密度可达，再结合 $q$ 可从 $z$ 密度可达这一事实，表明 $q$ 可从 $p$ 密度可达。

It thus follows by Definition 2.2 that there is a sequence of points ${p}_{1},{p}_{2},\ldots ,{p}_{t} \in  P$ such that (i) ${p}_{1} = p,{p}_{t} = q$ ,and (ii) dist $\left( {{p}_{i},{p}_{i + 1}}\right)  \leq  \epsilon$ for each $i \in  \left\lbrack  {1,t - 1}\right\rbrack$ . Let $k$ be the smallest $i \in  \left\lbrack  {2,t}\right\rbrack$ such that ${p}_{i}$ is the center of a ball in ${S}_{\text{ball }}$ . Note that $k$ definitely exists because ${p}_{t}$ is such a center. It thus follows that ${p}_{k - 1}$ is a point from ${S}_{pt}$ ,and that ${p}_{k - 1}$ is covered by the ball in ${S}_{\text{ball }}$ centered at ${p}_{k}$ .

因此，根据定义 2.2，存在一个点序列 ${p}_{1},{p}_{2},\ldots ,{p}_{t} \in  P$，使得 (i) ${p}_{1} = p,{p}_{t} = q$，并且 (ii) 对于每个 $i \in  \left\lbrack  {1,t - 1}\right\rbrack$，有 dist $\left( {{p}_{i},{p}_{i + 1}}\right)  \leq  \epsilon$。设 $k$ 是使得 ${p}_{i}$ 是 ${S}_{\text{ball }}$ 中某个球的中心的最小 $i \in  \left\lbrack  {2,t}\right\rbrack$。注意，$k$ 肯定存在，因为 ${p}_{t}$ 就是这样一个中心。因此，${p}_{k - 1}$ 是 ${S}_{pt}$ 中的一个点，并且 ${p}_{k - 1}$ 被 ${S}_{\text{ball }}$ 中以 ${p}_{k}$ 为中心的球覆盖。

Case 2: We return no. We will show that in this case no point of ${S}_{pt}$ is covered by any ball in ${S}_{\text{ball }}$ .

情况 2：我们返回“否”。我们将证明在这种情况下，${S}_{pt}$ 中没有点被 ${S}_{\text{ball }}$ 中的任何球覆盖。

This is in fact very easy. Suppose on the contrary that a point $p \in  {S}_{pt}$ is covered by a ball of ${S}_{\text{ball }}$ centered at $q$ . Thus, $\operatorname{dist}\left( {p,q}\right)  \leq  \epsilon$ ,namely, $q$ is density-reachable from $p$ . Then,by maximality of Definition 2.3, $q$ must be in the cluster of $p$ (recall that all the points of $P$ are core points). This contradicts the fact that we returned no.

实际上这非常简单。假设相反，点 $p \in  {S}_{pt}$ 被 ${S}_{\text{ball }}$ 中以 $q$ 为中心的球覆盖。因此，$\operatorname{dist}\left( {p,q}\right)  \leq  \epsilon$，即 $q$ 可从 $p$ 密度可达。然后，根据定义 2.3 的极大性，$q$ 必须在 $p$ 的聚类中（回顾一下，$P$ 中的所有点都是核心点）。这与我们返回“否”这一事实相矛盾。

Theorem 3.1 immediately follows from Lemmas 2.6 and 3.2.

定理 3.1 可直接由引理 2.6 和 3.2 推出。

### 3.2 A New Exact Algorithm for $d \geq  3$

### 3.2 针对 $d \geq  3$ 的一种新的精确算法

It is well known that DBSCAN can be solved in $O\left( {n}^{2}\right)$ time (e.g.,see Tan et al. (2006)) in any constant dimensionality $d$ . Next,we show that it is possible to always terminate in $o\left( {n}^{2}\right)$ time regardless of the constant $d$ . Our algorithm extends that of Gunawan (2013) with two ideas:

众所周知，在任何固定维度 $d$ 下，DBSCAN（基于密度的空间聚类应用程序与噪声）算法可以在 $O\left( {n}^{2}\right)$ 时间内求解（例如，参见 Tan 等人（2006 年）的研究）。接下来，我们将证明，无论固定维度 $d$ 为何值，该算法总能在 $o\left( {n}^{2}\right)$ 时间内终止。我们的算法基于 Gunawan（2013 年）的算法，融入了两个新思路：

-Use a $d$ -dimensional grid $T$ with an appropriate side length for its cells.

- 使用一个 $d$ 维网格 $T$，其单元格的边长设置合适。

-Compute the edges of the graph $G$ with a BCP algorithm (as opposed to nearest neighbor search).

- 使用 BCP（二元约束规划）算法来计算图 $G$ 的边（而非采用最近邻搜索方法）。

Next,we explain the details. $T$ is now a grid on ${\mathbb{R}}^{d}$ where each cell of $T$ is a $d$ -dimensional hyper-square with side length $\epsilon /\sqrt{d}$ . As before,this ensures that any two points in the same cell are within distance $\epsilon$ from each other.

接下来，我们详细解释。$T$ 现在是 ${\mathbb{R}}^{d}$ 上的一个网格，其中 $T$ 的每个单元格都是边长为 $\epsilon /\sqrt{d}$ 的 $d$ 维超立方体。和之前一样，这确保了同一单元格内的任意两点之间的距离不超过 $\epsilon$。

The algorithm description in Section 2.2 carries over to any $d \geq  3$ almost verbatim. The only difference is the way we compute the edges of $G$ . Given core cells ${c}_{1}$ and ${c}_{2}$ that are $\epsilon$ -neighbors of each other,we solve the BCP problem on the sets of core points in ${c}_{1}$ and ${c}_{2}$ ,respectively. Let $\left( {{p}_{1},{p}_{2}}\right)$ be the pair returned. We add an edge $\left( {{c}_{1},{c}_{2}}\right)$ to $G$ if and only if $\operatorname{dist}\left( {{p}_{1},{p}_{2}}\right)  \leq  \epsilon$ .

第 2.2 节中的算法描述几乎可以逐字应用于任何 $d \geq  3$ 情况。唯一的区别在于我们计算 $G$ 边的方式。给定互为 $\epsilon$ 邻域的核心单元格 ${c}_{1}$ 和 ${c}_{2}$，我们分别对 ${c}_{1}$ 和 ${c}_{2}$ 中的核心点集求解 BCP 问题。设 $\left( {{p}_{1},{p}_{2}}\right)$ 为返回的点对。当且仅当 $\operatorname{dist}\left( {{p}_{1},{p}_{2}}\right)  \leq  \epsilon$ 时，我们向 $G$ 中添加一条边 $\left( {{c}_{1},{c}_{2}}\right)$。

The adapted algorithm achieves the following efficiency guarantee:

改进后的算法实现了以下效率保证：

THEOREM 3.3. For any fixed dimensionality $d \geq  4$ ,there is an algorithm solving the DBSCAN problem in $O\left( {n}^{2 - \frac{2}{\lceil d/2\rceil  + 1} + \delta }\right)$ expected time,where $\delta  > 0$ can be an arbitrarily small constant. For $d = 3$ ,the running time can be improved to $O\left( {\left( n\log n\right) }^{4/3}\right)$ expected.

定理 3.3。对于任何固定维度 $d \geq  4$，存在一种算法能在 $O\left( {n}^{2 - \frac{2}{\lceil d/2\rceil  + 1} + \delta }\right)$ 的期望时间内解决 DBSCAN 问题，其中 $\delta  > 0$ 可以是任意小的常数。对于 $d = 3$，运行时间可以改进为 $O\left( {\left( n\log n\right) }^{4/3}\right)$ 的期望时间。

Proof. It suffices to analyze the time used by our algorithm to generate the edges of $G$ . The other parts of the algorithm use $O\left( n\right)$ expected time,following the analysis of Gunawan (2013).

证明。只需分析我们的算法生成 $G$ 边所花费的时间即可。根据 Gunawan（2013 年）的分析，算法的其他部分使用 $O\left( n\right)$ 的期望时间。

Let us consider first $d \geq  4$ . First,fix the value of $\delta$ in Theorem 3.3. Define: $\lambda  = \frac{1}{\lceil d/2\rceil  + 1} - \delta /2$ . Given a core cell $c$ ,we denote by ${m}_{c}$ the number of core points in $c$ . Then,by Lemma 2.5,the time we spend generating the edges of $G$ is

让我们首先考虑 $d \geq  4$。首先，固定定理 3.3 中 $\delta$ 的值。定义：$\lambda  = \frac{1}{\lceil d/2\rceil  + 1} - \delta /2$。给定一个核心单元格 $c$，我们用 ${m}_{c}$ 表示 $c$ 中核心点的数量。然后，根据引理 2.5，我们生成 $G$ 边所花费的时间为

$$
\mathop{\sum }\limits_{\substack{{\epsilon \text{-neighbor }} \\  {\text{ core cells }c,{c}^{\prime }} }}O\left( {{\left( {m}_{c}{m}_{{c}^{\prime }}\right) }^{1 - \lambda } + {m}_{c}\log {m}_{{c}^{\prime }} + {m}_{{c}^{\prime }}\log {m}_{c}}\right) . \tag{1}
$$

To bound the first term, we derive

为了界定第一项，我们推导

$$
\mathop{\sum }\limits_{{\epsilon \text{-neighbor core cells}}}O\left( {\left( {m}_{c}{m}_{{c}^{\prime }}\right) }^{1 - \lambda }\right) 
$$

$$
 = \mathop{\sum }\limits_{\substack{{\epsilon \text{ -neighbor }} \\  {\text{ core cells }{c}^{\prime }} \\  {\text{ some cells }{c}^{\prime }} }}O\left( {\left( {m}_{c}{m}_{{c}^{\prime }}\right) }^{1 - \lambda }\right)  + \mathop{\sum }\limits_{\substack{{\epsilon \text{ -neighbor }} \\  {\text{ core cells }{c}^{\prime }} \\  {\text{ core cells }{c}^{\prime }} }}O\left( {\left( {m}_{c}{m}_{{c}^{\prime }}\right) }^{1 - \lambda }\right) 
$$

$$
 = \mathop{\sum }\limits_{\substack{{\epsilon \text{ -neighbor }} \\  {\text{ core cells }c,{c}^{\prime }} \\  {\text{ sor cells }c,{c}^{\prime }} }}O\left( {{m}_{{c}^{\prime }} \cdot  {m}_{c}^{1 - {2\lambda }}}\right)  + \mathop{\sum }\limits_{\substack{{\epsilon \text{ -neighbor }} \\  {\text{ core cells }c,{c}^{\prime }} \\  {\text{ core cells }c,{c}^{\prime }} }}O\left( {{m}_{c} \cdot  {m}_{{c}^{\prime }}^{1 - {2\lambda }}}\right) 
$$

$$
 = \mathop{\sum }\limits_{\substack{{\epsilon \text{-neighbor }} \\  {\text{ core ells }c,{c}^{\prime }} \\  {\text{ core ells }c,{c}^{\prime }} }}O\left( {{m}_{{c}^{\prime }} \cdot  {n}^{1 - {2\lambda }}}\right)  + \mathop{\sum }\limits_{\substack{{\epsilon \text{-neighbor }} \\  {\text{ core cells }c,{c}^{\prime }} \\  {\text{ core cells }c,{c}^{\prime }} \\  {\text{ s.t. }{m}_{c} > {m}_{{c}^{\prime }}} }}O\left( {{m}_{c} \cdot  {n}^{1 - {2\lambda }}}\right) 
$$

$$
 = O\left( {{n}^{1 - {2\lambda }}\mathop{\sum }\limits_{{\epsilon \text{-neighbor core cells }c,{c}^{\prime }}}{m}_{c}}\right)  = O\left( {n}^{2 - {2\lambda }}\right) ,
$$

where the last equality used the fact that $c$ has only $O\left( 1\right) \epsilon$ -neighbor cells as long as $d$ is a constant (and hence, ${m}_{c}$ can be added only $O\left( 1\right)$ times). The other terms in Equation (1) are easy to bound:

其中最后一个等式利用了这样一个事实：只要 $d$ 是常数，$c$ 就只有 $O\left( 1\right) \epsilon$ 邻域单元格（因此，${m}_{c}$ 最多只能添加 $O\left( 1\right)$ 次）。方程（1）中的其他项很容易界定：

$$
\mathop{\sum }\limits_{{\epsilon \text{-neighbor core cells }c,{c}^{\prime }}}O\left( {{m}_{c}\log {m}_{{c}^{\prime }} + {m}_{{c}^{\prime }}\log {m}_{c}}\right) 
$$

$$
 = \mathop{\sum }\limits_{{\epsilon \text{-neighbor core cells }c,{c}^{\prime }}}O\left( {{m}_{c}\log n + {m}_{{c}^{\prime }}\log n}\right)  = O\left( {n\log n}\right) .
$$

In summary,we spend $O\left( {{n}^{2 - {2\lambda }} + n\log n}\right)  = O\left( {n}^{2 - \frac{2}{\lceil d/2\rceil  + 1} + \delta }\right)$ time generating the edges of $E$ . This proves the part of Theorem 3.3 for $d \geq  4$ . An analogous analysis based on the $d = 3$ branch of Lemma 2.5 establishes the other part of Theorem 3.3.

综上所述，我们花费 $O\left( {{n}^{2 - {2\lambda }} + n\log n}\right)  = O\left( {n}^{2 - \frac{2}{\lceil d/2\rceil  + 1} + \delta }\right)$ 的时间生成 $E$ 的边。这证明了定理 3.3 中关于 $d \geq  4$ 的部分。基于引理 2.5 的 $d = 3$ 分支进行类似分析，可证明定理 3.3 的另一部分。

It is worth pointing out that the running time of our 3D algorithm nearly matches the lower bound in Theorem 3.1.

值得指出的是，我们的三维算法的运行时间几乎达到了定理 3.1 中的下界。

## 4 $\rho$ -Approximate DBSCAN

## 4 $\rho$ -近似 DBSCAN

The hardness result in Theorem 3.1 indicates the need of resorting to approximation if one wants to achieve near-linear running time for $d \geq  3$ . In Section 4.1,we introduce the concept of $\rho$ - approximate DBSCAN designed to replace DBSCAN on large datasets. In Section 4.2, we establish a strong quality guarantee of this new form of clustering. In Sections 4.3 and 4.4, we propose an algorithm for solving the $\rho$ -approximate DBSCAN problem in time linear to the dataset size.

定理3.1中的难解性结果表明，如果想让$d \geq  3$达到近似线性的运行时间，就需要采用近似算法。在4.1节中，我们引入$\rho$ - 近似DBSCAN的概念，旨在用其替代大数据集上的DBSCAN算法。在4.2节中，我们为这种新的聚类形式建立了强大的质量保证。在4.3节和4.4节中，我们提出一种算法，用于在与数据集大小呈线性关系的时间内解决$\rho$ - 近似DBSCAN问题。

### 4.1 Definitions

### 4.1 定义

As before,let $P$ be the input set of $n$ points in ${\mathbb{R}}^{d}$ to be clustered. We still take parameters $\epsilon$ and MinPts,but in addition,also a third parameter $\rho$ ,which can be any arbitrarily small positive constant, and controls the degree of approximation.

和之前一样，设$P$为${\mathbb{R}}^{d}$中待聚类的$n$个点的输入集。我们仍然采用参数$\epsilon$和最小点数（MinPts），此外，还引入第三个参数$\rho$，它可以是任意小的正常数，用于控制近似程度。

Next, we re-visit the basic definitions of DBSCAN in Section 2, and modify some of them to their " $\rho$ -approximate versions." First,the notion of core/non-core point remains the same as Definition 2.1. The concept of density-reachability in Definition 2.2 is also inherited directly, but we will also need the following:

接下来，我们重新回顾第2节中DBSCAN的基本定义，并将其中一些定义修改为它们的“$\rho$ - 近似版本”。首先，核心点/非核心点的概念与定义2.1保持一致。定义2.2中密度可达性的概念也直接沿用，但我们还需要以下内容：

Definition 4.1. A point $q \in  P$ is $\rho$ -approximate density-reachable from $p \in  P$ if there is a sequence of points ${p}_{1},{p}_{2},\ldots ,{p}_{t} \in  P$ (for some integer $t \geq  2$ ) such that

定义4.1。如果存在一个点序列${p}_{1},{p}_{2},\ldots ,{p}_{t} \in  P$（对于某个整数$t \geq  2$），使得点$q \in  P$从$p \in  P$是$\rho$ - 近似密度可达的。

$$
 - {p}_{1} = p\text{and}{p}_{t} = q\text{,}
$$

$- {p}_{1},{p}_{2},\ldots ,{p}_{t - 1}$ are core points,and

$- {p}_{1},{p}_{2},\ldots ,{p}_{t - 1}$是核心点，并且

$$
 - {p}_{i + 1} \in  B\left( {{p}_{i},\epsilon \left( {1 + \rho }\right) }\right) \text{for each}i \in  \left\lbrack  {1,t - 1}\right\rbrack  \text{.}
$$

Note the difference between the above and Definition 2.2: in the third bullet, the radius of the ball is increased to $\epsilon \left( {1 + \rho }\right)$ . To illustrate,consider a small input set $P$ as shown in Figure 5 . Set MinPts = 4 . The inner and outer circles have radii $\epsilon$ and $\epsilon \left( {1 + \rho }\right)$ ,respectively. Core and non-core points are in black and white,respectively. Point ${o}_{5}$ is $\rho$ -approximate density-reachable from ${o}_{3}$ (via sequence: ${o}_{3},{o}_{2},{o}_{1},{o}_{5}$ ). However, ${o}_{5}$ is not density-reachable from ${o}_{3}$ .

注意上述内容与定义2.2的区别：在第三点中，球的半径增加到$\epsilon \left( {1 + \rho }\right)$。为了说明这一点，考虑如图5所示的一个小输入集$P$。设最小点数（MinPts） = 4。内圆和外圆的半径分别为$\epsilon$和$\epsilon \left( {1 + \rho }\right)$。核心点和非核心点分别用黑色和白色表示。点${o}_{5}$从${o}_{3}$是$\rho$ - 近似密度可达的（通过序列：${o}_{3},{o}_{2},{o}_{1},{o}_{5}$）。然而，${o}_{5}$从${o}_{3}$不是密度可达的。

Definition 4.2. A $\rho$ -approximate cluster $C$ is a non-empty subset of $P$ such that

定义4.2。一个$\rho$ - 近似聚类$C$是$P$的一个非空子集，使得

-(Maximality) If a core point $p \in  C$ ,then all the points density-reachable from $p$ also belong to $C$ .

-（最大性）如果一个核心点$p \in  C$，那么从$p$密度可达的所有点也都属于$C$。

<!-- Media -->

<!-- figureText: ${o}_{1}$ 0.3 ${O}_{2} \bullet$ -->

<img src="https://cdn.noedgeai.com/0195c91b-4734-77b1-933f-3c7cb33c7bfe_11.jpg?x=586&y=262&w=392&h=428&r=0"/>

Fig. 5. Density-reachability and $\rho$ -approximate density-reachability $\left( {\text{ MinPts } = 4}\right)$ .

图5. 密度可达性和$\rho$ - 近似密度可达性$\left( {\text{ MinPts } = 4}\right)$。

<!-- Media -->

-( $\rho$ -Approximate Connectivity) For any points ${p}_{1},{p}_{2} \in  C$ ,there exists a point $p \in  C$ such that both ${p}_{1}$ and ${p}_{2}$ are $\rho$ -approximate density-reachable from $p$ .

-（$\rho$ - 近似连通性）对于任意点${p}_{1},{p}_{2} \in  C$，存在一个点$p \in  C$，使得${p}_{1}$和${p}_{2}$都从$p$是$\rho$ - 近似密度可达的。

Note the difference between the above and the original cluster formulation (Definition 1): the connectivity requirement has been weakened into $\rho$ -approximate connectivity. In Figure 5,both $\left\{  {{o}_{1},{o}_{2},{o}_{3},{o}_{4}}\right\}$ and $\left\{  {{o}_{1},{o}_{2},{o}_{3},{o}_{4},{o}_{5}}\right\}$ are $\rho$ -approximate clusters.

注意上述内容与原始聚类定义（定义1）的区别：连通性要求已被弱化为$\rho$ - 近似连通性。在图5中，$\left\{  {{o}_{1},{o}_{2},{o}_{3},{o}_{4}}\right\}$和$\left\{  {{o}_{1},{o}_{2},{o}_{3},{o}_{4},{o}_{5}}\right\}$都是$\rho$ - 近似聚类。

Problem 2. The $\rho$ -approximate DBSCAN problem is to find a set $\mathcal{C}$ of $\rho$ -approximate clusters of $P$ such that every core point of $P$ appears in exactly one $\rho$ -approximate cluster.

问题2。$\rho$ - 近似DBSCAN问题是找到$P$的一个$\rho$ - 近似聚类集合$\mathcal{C}$，使得$P$的每个核心点恰好出现在一个$\rho$ - 近似聚类中。

Unlike the original DBSCAN problem,the $\rho$ -approximate version may not have a unique result. In Figure 5,for example,it is legal to return either $\left\{  {{o}_{1},{o}_{2},{o}_{3},{o}_{4}}\right\}$ or $\left\{  {{o}_{1},{o}_{2},{o}_{3},{o}_{4},{o}_{5}}\right\}$ . Nevertheless, any result of the $\rho$ -approximate problem comes with the quality guarantee to be proved next.

与原始的DBSCAN问题不同，$\rho$近似版本可能没有唯一的结果。例如，在图5中，返回$\left\{  {{o}_{1},{o}_{2},{o}_{3},{o}_{4}}\right\}$或$\left\{  {{o}_{1},{o}_{2},{o}_{3},{o}_{4},{o}_{5}}\right\}$都是合理的。尽管如此，$\rho$近似问题的任何结果都具有接下来要证明的质量保证。

### 4.2 A Sandwich Theorem

### 4.2 三明治定理

Both DBSCAN and $\rho$ -approximate DBSCAN are parameterized by $\epsilon$ and MinPts. It would be perfect if they can always return exactly the same clustering results. Of course, this is too good to be true. Nevertheless,in this subsection,we will show that this is almost true: the result of $\rho$ - approximate DBSCAN is guaranteed to be somewhere between the (exact) DBSCAN results obtained by $\left( {\epsilon ,\text{MinPts}}\right)$ and by $\left( {\epsilon \left( {1 + \rho }\right) ,\text{MinPts}}\right)$ ! It is well known that the clusters of DBSCAN rarely differ considerably when $\epsilon$ changes by just a small factor-in fact,if this really happens,it suggests that the choice of $\epsilon$ is very bad,such that the exact clusters are not stable anyway (we will come back to this issue later).

DBSCAN和$\rho$近似DBSCAN都由$\epsilon$和MinPts这两个参数确定。如果它们总能返回完全相同的聚类结果，那就再好不过了。当然，这好得不太现实。不过，在本小节中，我们将证明这几乎是可行的：$\rho$近似DBSCAN的结果必定介于通过$\left( {\epsilon ,\text{MinPts}}\right)$和$\left( {\epsilon \left( {1 + \rho }\right) ,\text{MinPts}}\right)$得到的（精确）DBSCAN结果之间！众所周知，当$\epsilon$仅发生微小变化时，DBSCAN的聚类结果很少会有显著差异——事实上，如果真的出现这种情况，就表明$\epsilon$的选择非常糟糕，以至于精确的聚类本身就不稳定（我们稍后会再讨论这个问题）。

Let us define

让我们定义

$- {\mathcal{C}}_{1}$ as the set of clusters of DBSCAN with parameters $\left( {\epsilon ,\text{ MinPts }}\right)$ ;

$- {\mathcal{C}}_{1}$为参数为$\left( {\epsilon ,\text{ MinPts }}\right)$时DBSCAN的聚类集合；

$- {\mathcal{C}}_{2}$ as the set of clusters of DBSCAN with parameters $\left( {\epsilon \left( {1 + \rho }\right) ,\text{MinPts}}\right)$ ; and

$- {\mathcal{C}}_{2}$为参数为$\left( {\epsilon \left( {1 + \rho }\right) ,\text{MinPts}}\right)$时DBSCAN的聚类集合；以及

- $\mathcal{C}$ as an arbitrary set of clusters that is a legal result of $\left( {\epsilon ,\text{MinPts,}\rho }\right)$ -approx-DBSCAN.

- $\mathcal{C}$为$\left( {\epsilon ,\text{MinPts,}\rho }\right)$近似DBSCAN的任意合法聚类结果集合。

The next theorem formalizes the quality assurance mentioned earlier:

接下来的定理将前面提到的质量保证进行了形式化表述：

THEOREM 4.3 (SANDWICH QUALITY GUARANTEE). The following statements are true:

定理4.3（三明治质量保证）。以下陈述是正确的：

(1) For any cluster ${C}_{1} \in  {\mathcal{C}}_{1}$ ,there is a cluster $C \in  \mathcal{C}$ such that ${C}_{1} \subseteq  C$ .

(1) 对于任意聚类${C}_{1} \in  {\mathcal{C}}_{1}$，存在一个聚类$C \in  \mathcal{C}$，使得${C}_{1} \subseteq  C$。

(2) For any cluster $C \in  \mathcal{C}$ ,there is a cluster ${C}_{2} \in  {\mathcal{C}}_{2}$ such that $C \subseteq  {C}_{2}$ .

(2) 对于任意聚类$C \in  \mathcal{C}$，存在一个聚类${C}_{2} \in  {\mathcal{C}}_{2}$，使得$C \subseteq  {C}_{2}$。

<!-- Media -->

<!-- figureText: ( $b\breve{a}d$ -->

<img src="https://cdn.noedgeai.com/0195c91b-4734-77b1-933f-3c7cb33c7bfe_12.jpg?x=393&y=263&w=779&h=402&r=0"/>

Fig. 6. Good and bad choices of $\epsilon$ .

图6. $\epsilon$的好坏选择。

<!-- Media -->

Proof. To prove Statement 1,let $p$ be an arbitrary core point in ${C}_{1}$ . Then, ${C}_{1}$ is precisely the set of points in $P$ density-reachable from $p.{}^{2}$ In general,if a point $q$ is density-reachable from $p$ in $\left( {\epsilon ,\text{ MinPts }}\right)$ -exact-DBSCAN, $q$ is also density-reachable from $p$ in $\left( {\epsilon ,\text{ MinPts,}\rho }\right)$ -approx-DBSCAN. By maximality of Definition 4.2,if $C$ is the cluster in $\mathcal{C}$ containing $p$ ,then all the points of ${C}_{1}$ must be in $C$ .

证明。为了证明陈述1，设$p$是${C}_{1}$中的任意一个核心点。那么，${C}_{1}$恰好是在$P$中从$p.{}^{2}$密度可达的点的集合。一般来说，如果一个点$q$在$\left( {\epsilon ,\text{ MinPts }}\right)$精确DBSCAN中从$p$密度可达，那么$q$在$\left( {\epsilon ,\text{ MinPts,}\rho }\right)$近似DBSCAN中也从$p$密度可达。根据定义4.2的极大性，如果$C$是$\mathcal{C}$中包含$p$的聚类，那么${C}_{1}$中的所有点都必定在$C$中。

To prove Statement 2,consider an arbitrary core point $p \in  C$ (there must be one by Definition 4.2). In $\left( {\epsilon \left( {1 + \rho }\right) ,\text{MinPts}}\right)$ -exact-DBSCAN, $p$ must also be a core point. We choose ${C}_{2}$ to be the cluster of ${\mathcal{C}}_{2}$ where $p$ belongs. Now,fix an arbitrary point $q \in  C$ . In $\left( {\epsilon ,\operatorname{MinPts},\rho }\right)$ -approx-DBSCAN,by $\rho$ -approximate connectivity of Definition 4.2,we know that $p$ and $q$ are both $\rho$ - approximate reachable from a point $z$ . This implies that $z$ is also $\rho$ -approximate reachable from $p$ . Hence, $q$ is $\rho$ -approximate reachable from $p$ . This means that $q$ is density-reachable from $p$ in $\left( {\epsilon \left( {1 + \rho }\right) ,\text{ MinPts }}\right)$ -exact-DBSCAN,indicating that $q \in  {C}_{2}$ .

为了证明陈述2，考虑任意一个核心点$p \in  C$（根据定义4.2，必定存在这样一个点）。在$\left( {\epsilon \left( {1 + \rho }\right) ,\text{MinPts}}\right)$精确DBSCAN算法中，$p$也必定是一个核心点。我们选择${C}_{2}$作为${\mathcal{C}}_{2}$所在的簇，其中$p$属于该簇。现在，固定任意一个点$q \in  C$。在$\left( {\epsilon ,\operatorname{MinPts},\rho }\right)$近似DBSCAN算法中，根据定义4.2中的$\rho$近似连通性，我们知道$p$和$q$都可以从点$z$ $\rho$近似可达。这意味着$z$也可以从$p$ $\rho$近似可达。因此，$q$可以从$p$ $\rho$近似可达。这表明在$\left( {\epsilon \left( {1 + \rho }\right) ,\text{ MinPts }}\right)$精确DBSCAN算法中，$q$从$p$是密度可达的，即$q \in  {C}_{2}$。

Here is an alternative, more intuitive, interpretation of Theorem 4.3:

以下是对定理4.3的另一种更直观的解释：

- Statement 1 says that if two points belong to the same cluster of DBSCAN with parameters ( $\epsilon ,$ MinPts),they are definitely in the same cluster of $\rho$ -approximate DBSCAN with the same parameters.

- 陈述1表明，如果两个点属于参数为( $\epsilon ,$ 最小点数)的DBSCAN算法的同一个簇，那么它们肯定也属于具有相同参数的$\rho$近似DBSCAN算法的同一个簇。

-On the other hand,a cluster of $\rho$ -approximate DBSCAN parameterized by $\left( {\epsilon ,\text{MinPts}}\right)$ may also contain two points ${p}_{1},{p}_{2}$ that are in different clusters of DBSCAN with the same parameters. However, this is not bad because Statement 2 says that as soon as the parameter $\epsilon$ increases to $\epsilon \left( {1 + \rho }\right) ,{p}_{1}$ and ${p}_{2}$ will fall into the same cluster of DBSCAN!

- 另一方面，参数为$\left( {\epsilon ,\text{MinPts}}\right)$的$\rho$近似DBSCAN算法的一个簇可能还包含两个点${p}_{1},{p}_{2}$，而这两个点在具有相同参数的DBSCAN算法中属于不同的簇。然而，这并非坏事，因为陈述2表明，一旦参数$\epsilon$增大到$\epsilon \left( {1 + \rho }\right) ,{p}_{1}$，${p}_{2}$将落入DBSCAN算法的同一个簇中！

Figure 6 illustrates the effects of approximation. How many clusters are there? Interestingly, the answer is it depends. As pointed out in the classic OPTICS paper (Ankerst et al. 1999), different $\epsilon$ values allow us to view the dataset from various granularities,leading to different clustering results. In Figure 6,given ${\epsilon }_{1}$ (and some MinPts say 2),DBSCAN outputs three clusters. Given ${\epsilon }_{2}$ , on the other hand, DBSCAN outputs two clusters, which makes sense because at this distance, the two clusters on the right merge into one.

图6展示了近似的效果。有多少个簇呢？有趣的是，答案是这取决于具体情况。正如经典的OPTICS论文（Ankerst等人，1999年）所指出的，不同的$\epsilon$值使我们能够从不同的粒度观察数据集，从而得到不同的聚类结果。在图6中，给定${\epsilon }_{1}$（以及某个最小点数，例如2），DBSCAN算法输出三个簇。另一方面，给定${\epsilon }_{2}$，DBSCAN算法输出两个簇，这是合理的，因为在这个距离下，右侧的两个簇合并成了一个。

Now let us consider approximation. The dashed circles illustrate the radii obtained with $\rho$ - approximation. For both ${\epsilon }_{1}$ and ${\epsilon }_{2},\rho$ -approximate DBSCAN will return exactly the same clusters,because these distances are robustly chosen by being insensitive to small perturbation. For ${\epsilon }_{3}$ , however, $\rho$ -approximate DBSCAN may return only one cluster (i.e.,all points in the same cluster), whereas exact DBSCAN will return only two (i.e.,the same two clusters as ${\epsilon }_{2}$ ). By looking at the figure closely,one can realize that this happens because the dashed circle of radius $\left( {1 + \rho }\right) {\epsilon }_{3}$ "happens" to pass a point-namely,point $o$ -which falls outside the solid circle of radius ${\epsilon }_{3}$ . Intuitively, ${\epsilon }_{3}$ is a poor parameter choice because it is too close to the distance between two clusters such that a small change to it will cause the clustering results to be altered.

现在让我们考虑近似情况。虚线圆表示通过$\rho$近似得到的半径。对于${\epsilon }_{1}$和${\epsilon }_{2},\rho$，近似DBSCAN算法将返回完全相同的簇，因为这些距离的选择具有鲁棒性，对小的扰动不敏感。然而，对于${\epsilon }_{3}$，$\rho$近似DBSCAN算法可能只返回一个簇（即，所有点都在同一个簇中），而精确DBSCAN算法将只返回两个簇（即，与${\epsilon }_{2}$相同的两个簇）。仔细观察该图，可以发现这种情况的发生是因为半径为$\left( {1 + \rho }\right) {\epsilon }_{3}$的虚线圆“恰好”经过一个点，即点$o$，而该点落在半径为${\epsilon }_{3}$的实线圆之外。直观地说，${\epsilon }_{3}$是一个糟糕的参数选择，因为它太接近两个簇之间的距离，以至于对其进行小的更改就会导致聚类结果发生改变。

---

<!-- Footnote -->

${}^{2}$ This should be folklore but here is a proof. By maximality of Definition 2.3,all the points density-reachable from $p$ are in ${C}_{1}$ . On the other hand,let $q$ be any point in ${C}_{1}$ . By connectivity, $p$ and $q$ are both density-reachable from a point $z$ . As $p$ is a core point,we know that $z$ is also density-reachable from $p$ . Hence, $q$ is density-reachable from $p$ .

${}^{2}$ 这应该是一个常识，但这里给出一个证明。根据定义2.3的极大性，所有从$p$密度可达的点都在${C}_{1}$中。另一方面，设$q$是${C}_{1}$中的任意一点。根据连通性，$p$和$q$都从某一点$z$密度可达。由于$p$是一个核心点，我们知道$z$也从$p$密度可达。因此，$q$从$p$密度可达。

<!-- Footnote -->

---

Next we present a useful corollary of the sandwich theorem:

接下来，我们给出夹逼定理的一个有用推论：

Corollary 4.4. Let ${\mathcal{C}}_{1},{\mathcal{C}}_{2}$ ,and $\mathcal{C}$ be as defined in Theorem 4.3. If a cluster $C$ appears in both ${\mathcal{C}}_{1}$ and ${\mathcal{C}}_{2}$ ,then $C$ must also be a cluster in $\mathcal{C}$ .

推论4.4。设${\mathcal{C}}_{1},{\mathcal{C}}_{2}$和$\mathcal{C}$如定理4.3所定义。如果一个簇$C$同时出现在${\mathcal{C}}_{1}$和${\mathcal{C}}_{2}$中，那么$C$也一定是$\mathcal{C}$中的一个簇。

Proof. Suppose,on the contrary,that $\mathcal{C}$ does not contain $C$ . By Theorem 4.3,(i) $\mathcal{C}$ must contain a cluster ${C}^{\prime }$ such that $C \subseteq  {C}^{\prime }$ ,and (ii) ${\mathcal{C}}_{2}$ must contain a cluster ${C}^{\prime \prime }$ such that ${C}^{\prime } \subseteq  {C}^{\prime \prime }$ . This means $C \subseteq  {C}^{\prime \prime }$ . On the other hand,as $C \in  {\mathcal{C}}_{2}$ ,it follows that,in ${\mathcal{C}}_{2}$ ,every core point in $C$ belongs also to ${C}^{\prime \prime }$ . This is impossible because a core point can belong to only one cluster.

证明。假设相反情况，即$\mathcal{C}$不包含$C$。根据定理4.3，(i) $\mathcal{C}$必须包含一个簇${C}^{\prime }$，使得$C \subseteq  {C}^{\prime }$，并且(ii) ${\mathcal{C}}_{2}$必须包含一个簇${C}^{\prime \prime }$，使得${C}^{\prime } \subseteq  {C}^{\prime \prime }$。这意味着$C \subseteq  {C}^{\prime \prime }$。另一方面，由于$C \in  {\mathcal{C}}_{2}$，由此可知，在${\mathcal{C}}_{2}$中，$C$中的每个核心点也属于${C}^{\prime \prime }$。这是不可能的，因为一个核心点只能属于一个簇。

The corollary states that,even if some exact DBSCAN clusters have changed when $\epsilon$ increases by a factor of $1 + \rho$ (i.e., $\epsilon$ is not robust),our $\rho$ -approximation still captures all those clusters that do not change. For example, imagine that the points in Figure 6 are part of a larger dataset such that the clusters on the rest of the points are unaffected as ${\epsilon }_{3}$ increases to ${\epsilon }_{3}\left( {1 + \rho }\right)$ . By Corollary 4.4, all those clusters are safely captured by $\rho$ -approximate DBSCAN under ${\epsilon }_{3}$ .

该推论表明，即使当$\epsilon$增大$1 + \rho$倍时（即$\epsilon$不具有鲁棒性），一些精确的DBSCAN（基于密度的空间聚类应用噪声）簇发生了变化，我们的$\rho$ - 近似算法仍然能捕获所有未发生变化的簇。例如，假设图6中的点是一个更大数据集的一部分，使得当${\epsilon }_{3}$增大到${\epsilon }_{3}\left( {1 + \rho }\right)$时，其余点上的簇不受影响。根据推论4.4，所有这些簇都能被${\epsilon }_{3}$下的$\rho$ - 近似DBSCAN算法安全地捕获。

### 4.3 Approximate Range Counting

### 4.3 近似范围计数

Let us now take a break from DBSCAN, and turn our attention to a different problem, whose solution is vital to our $\rho$ -approximate DBSCAN algorithm.

现在让我们暂时放下DBSCAN，将注意力转向另一个问题，该问题的解决方案对我们的$\rho$ - 近似DBSCAN算法至关重要。

Let $P$ still be a set of $n$ points in ${\mathbb{R}}^{d}$ where $d$ is a constant. Given any point $q \in  {\mathbb{R}}^{d}$ ,a distance threshold $\epsilon  > 0$ and an arbitrarily small constant $\rho  > 0$ ,an approximate range count query returns an integer that is guaranteed to be between $\left| {B\left( {q,\epsilon }\right)  \cap  P}\right|$ and $\left| {B\left( {q,\epsilon \left( {1 + \rho }\right) }\right)  \cap  P}\right|$ . For example,in Figure 5,given $q = {o}_{1}$ ,a query may return either 4 or 5 .

设$P$仍然是${\mathbb{R}}^{d}$中的一组$n$个点，其中$d$是一个常数。给定任意一点$q \in  {\mathbb{R}}^{d}$、一个距离阈值$\epsilon  > 0$和一个任意小的常数$\rho  > 0$，一个近似范围计数查询返回一个保证在$\left| {B\left( {q,\epsilon }\right)  \cap  P}\right|$和$\left| {B\left( {q,\epsilon \left( {1 + \rho }\right) }\right)  \cap  P}\right|$之间的整数。例如，在图5中，给定$q = {o}_{1}$，一个查询可能返回4或5。

Arya and Mount (2000) developed a structure of $O\left( n\right)$ space that can be built in $O\left( {n\log n}\right)$ time, and answers any such query in $O\left( {\log n}\right)$ time. Next,we design an alternative structure with better performance in our context:

阿亚（Arya）和芒特（Mount）（2000年）开发了一种$O\left( n\right)$空间的结构，该结构可以在$O\left( {n\log n}\right)$时间内构建，并能在$O\left( {\log n}\right)$时间内回答任何此类查询。接下来，我们设计一种在我们的情境中性能更好的替代结构：

LEMMA 4.5. For any fixed $\epsilon$ and $\rho$ ,there is a structure of $O\left( n\right)$ space that can be built in $O\left( n\right)$ expected time,and answers any approximate range count query in $O\left( 1\right)$ expected time.

引理4.5。对于任意固定的$\epsilon$和$\rho$，存在一种$O\left( n\right)$空间的结构，该结构可在$O\left( n\right)$期望时间内构建完成，并能在$O\left( 1\right)$期望时间内回答任何近似范围计数查询。

Structure. Our structure is a simple quadtree-like hierarchical grid partitioning of ${\mathbb{R}}^{d}$ . First, impose a regular grid on ${\mathbb{R}}^{d}$ where each cell is a $d$ -dimensional hyper-square with side length $\epsilon /\sqrt{d}$ . For each non-empty cell $c$ of the grid (i.e., $c$ covers at least 1 point of $P$ ),divide it into ${2}^{d}$ cells of the same size. For each resulting non-empty cell ${c}^{\prime }$ ,divide it recursively in the same manner, until the side length of ${c}^{\prime }$ is at most ${\epsilon \rho }/\sqrt{d}$ .

结构。我们的结构是对${\mathbb{R}}^{d}$进行的一种类似四叉树的分层网格划分。首先，在${\mathbb{R}}^{d}$上施加一个规则网格，其中每个单元格是一个边长为$\epsilon /\sqrt{d}$的$d$维超立方体。对于网格中每个非空单元格$c$（即$c$覆盖了$P$中的至少一个点），将其划分为${2}^{d}$个相同大小的单元格。对于每个得到的非空单元格${c}^{\prime }$，以相同的方式递归划分，直到${c}^{\prime }$的边长至多为${\epsilon \rho }/\sqrt{d}$。

We use $H$ to refer to the hierarchy thus obtained. We keep only the non-empty cells of $H$ ,and for each such cell $c$ ,record $\operatorname{cnt}\left( c\right)$ which is the number of points in $P$ covered by $c$ . We will refer to a cell of $H$ with side length $\epsilon /\left( {{2}^{i}\sqrt{d}}\right)$ as a level-i cell. Clearly, $H$ has only $h = \max \left\{  {1,1 + \left\lceil  {{\log }_{2}\left( {1/\rho }\right) }\right\rceil  }\right\}   =$ $O\left( 1\right)$ levels. If a level- $\left( {i + 1}\right)$ cell ${c}^{\prime }$ is inside a level- $i$ cell $c$ ,we say that ${c}^{\prime }$ is a child of $c$ ,and $c$ a parent of ${c}^{\prime }$ . A cell with no children is called a leaf cell.

我们使用$H$来表示这样得到的层次结构。我们只保留$H$中的非空单元格，并且对于每个这样的单元格$c$，记录$\operatorname{cnt}\left( c\right)$，它是$c$所覆盖的$P$中的点数。我们将边长为$\epsilon /\left( {{2}^{i}\sqrt{d}}\right)$的$H$单元格称为第i层单元格。显然，$H$只有$h = \max \left\{  {1,1 + \left\lceil  {{\log }_{2}\left( {1/\rho }\right) }\right\rceil  }\right\}   =$ $O\left( 1\right)$层。如果一个第$\left( {i + 1}\right)$层单元格${c}^{\prime }$位于一个第$i$层单元格$c$内部，我们称${c}^{\prime }$是$c$的子单元格，而$c$是${c}^{\prime }$的父单元格。没有子单元格的单元格称为叶单元格。

Figure 7 illustrates the part of the first three levels of $H$ for the dataset on the left. Note that empty cells are not stored.

图7展示了左侧数据集对应的$H$的前三层的部分情况。注意，空单元格不被存储。

Query. Given an approximate range count query with parameters $q,\epsilon ,\rho$ ,we compute its answer ans as follows. Initially,ans $= 0$ . In general,given a non-empty level- $i$ cell $c$ ,we distinguish three cases:

查询。给定一个参数为$q,\epsilon ,\rho$的近似范围计数查询，我们按如下方式计算其答案ans。初始时，ans $= 0$。一般来说，给定一个非空的第$i$层单元格$c$，我们区分三种情况：

<!-- Media -->

<!-- figureText: a level-0 cell number of points in this level-0 cell root(18) level 0 NW(2) NE(8) SW(8) level 1 NE(3) SW(5) NE(4) SW(4) $B\left( {q,\epsilon \left( {1 + \rho }\right) }\right)$ SE(2) $B\left( {q,\epsilon }\right)$ -->

<img src="https://cdn.noedgeai.com/0195c91b-4734-77b1-933f-3c7cb33c7bfe_14.jpg?x=373&y=262&w=821&h=415&r=0"/>

Fig. 7. Approximate range counting.

图7. 近似范围计数。

<!-- Media -->

-If $c$ is disjoint with $B\left( {q,\epsilon }\right)$ ,ignore it.

-如果$c$与$B\left( {q,\epsilon }\right)$不相交，则忽略它。

-If $c$ is fully covered by $B\left( {q,\epsilon \left( {1 + \rho }\right) }\right)$ ,add $\operatorname{cnt}\left( c\right)$ to ans.

-如果$c$完全被$B\left( {q,\epsilon \left( {1 + \rho }\right) }\right)$覆盖，则将$\operatorname{cnt}\left( c\right)$加到ans中。

-When neither of the above holds,check if $c$ is a leaf cell in $H$ . If not,process the child cells of $c$ in the same manner. Otherwise (i.e., $c$ is a leaf),add $\operatorname{cnt}\left( c\right)$ to ans only if $c$ intersects $B\left( {q,\epsilon }\right)$ .

-当上述两种情况都不成立时，检查$c$是否是$H$中的叶单元格。如果不是，则以相同的方式处理$c$的子单元格。否则（即$c$是叶单元格），仅当$c$与$B\left( {q,\epsilon }\right)$相交时，才将$\operatorname{cnt}\left( c\right)$加到ans中。

The algorithm starts from the level-0 non-empty cells that intersect with $B\left( {q,\epsilon }\right)$ .

该算法从与$B\left( {q,\epsilon }\right)$相交的第0层非空单元格开始。

To illustrate, consider the query shown in Figure 7. The two gray cells correspond to nodes SW(5) and $\mathrm{{NE}}\left( 4\right)$ at level 2. The subtree of neither of them is visited,but the reasons are different. For $\mathrm{{SW}}\left( 5\right)$ ,its cell is disjoint with $B\left( {q,\epsilon }\right)$ ,so we ignore it (even though it intersects $B\left( {q,\epsilon \left( {1 + \rho }\right) }\right)$ ). For $\mathrm{{NE}}\left( 4\right)$ ,its cell completely falls in $B\left( {q,\epsilon \left( {1 + \rho }\right) }\right)$ ,so we add its count 4 to the result (even though it is not covered by $B\left( {q,\epsilon }\right)$ ).

为了说明这一点，考虑图7中所示的查询。两个灰色单元格对应于第2层的节点SW(5)和$\mathrm{{NE}}\left( 4\right)$。它们的子树都未被访问，但原因不同。对于$\mathrm{{SW}}\left( 5\right)$，其单元格与$B\left( {q,\epsilon }\right)$不相交，因此我们忽略它（即使它与$B\left( {q,\epsilon \left( {1 + \rho }\right) }\right)$相交）。对于$\mathrm{{NE}}\left( 4\right)$，其单元格完全落在$B\left( {q,\epsilon \left( {1 + \rho }\right) }\right)$内，因此我们将其计数4添加到结果中（即使它未被$B\left( {q,\epsilon }\right)$覆盖）。

Correctness. The above algorithm has two guarantees. First,if a point $p \in  P$ is inside $B\left( {q,\epsilon }\right)$ ,it is definitely counted in ans. Second,if $p$ is outside $B\left( {q,\epsilon \left( {1 + \rho }\right) }\right)$ ,then it is definitely not counted in ans. These guarantees are easy to verify,utilizing the fact that if a leaf cell $c$ intersects $B\left( {p,\epsilon }\right)$ ,then $c$ must fall completely in $B\left( {p,\epsilon \left( {1 + \rho }\right) }\right)$ because any two points in a leaf cell are within distance ${\epsilon \rho }$ . It thus follows that the ans returned is a legal answer.

正确性。上述算法有两个保证。首先，如果一个点$p \in  P$在$B\left( {q,\epsilon }\right)$内部，那么它肯定会被计入答案ans中。其次，如果$p$在$B\left( {q,\epsilon \left( {1 + \rho }\right) }\right)$外部，那么它肯定不会被计入答案ans中。利用叶单元格$c$与$B\left( {p,\epsilon }\right)$相交时，$c$必定完全落在$B\left( {p,\epsilon \left( {1 + \rho }\right) }\right)$内这一事实（因为叶单元格中的任意两点之间的距离在${\epsilon \rho }$以内），很容易验证这些保证。因此，返回的答案ans是一个合法的答案。

Time Analysis. Remember that the hierarchy $H$ has $O\left( 1\right)$ levels. Since there are $O\left( n\right)$ non-empty cells at each level,the total space is $O\left( n\right)$ . With hashing,it is easy to build the structure level by level in $O\left( n\right)$ expected time.

时间分析。请记住，层次结构$H$有$O\left( 1\right)$层。由于每一层有$O\left( n\right)$个非空单元格，因此总空间为$O\left( n\right)$。通过哈希，很容易在期望时间$O\left( n\right)$内逐层构建该结构。

To analyze the running time of our query algorithm,observe that each cell $c$ visited by our algorithm must satisfy one of the following conditions: (i) $c$ is a level-0 cell,or (ii) the parent of $c$ intersects the boundary of $B\left( {q,\epsilon }\right)$ . For type (i),the $O\left( 1\right)$ level-0 cells intersecting $B\left( {q,\epsilon }\right)$ can be found in $O\left( 1\right)$ expected time using the coordinates of $q$ . For type (ii),it suffices to bound the number of cells intersecting the boundary of $B\left( {q,\epsilon }\right)$ because each such cell has ${2}^{d} = O\left( 1\right)$ child nodes.

为了分析我们查询算法的运行时间，请注意我们的算法访问的每个单元格 $c$ 必须满足以下条件之一：(i) $c$ 是 0 级单元格；或者 (ii) $c$ 的父单元格与 $B\left( {q,\epsilon }\right)$ 的边界相交。对于类型 (i)，使用 $q$ 的坐标，可以在 $O\left( 1\right)$ 的期望时间内找到与 $B\left( {q,\epsilon }\right)$ 相交的 $O\left( 1\right)$ 个 0 级单元格。对于类型 (ii)，只需限制与 $B\left( {q,\epsilon }\right)$ 的边界相交的单元格数量即可，因为每个这样的单元格都有 ${2}^{d} = O\left( 1\right)$ 个子节点。

In general,a $d$ -dimensional grid of cells with side length $l$ has $O\left( {1 + {\left( \frac{\theta }{l}\right) }^{d - 1}}\right)$ cells intersecting the boundary of a sphere with radius $\theta$ (Arya and Mount 2000). Combining this and the fact that a level- $i$ cell has side length $\epsilon /\left( {{2}^{i}\sqrt{d}}\right)$ ,we know that the total number of cells (of all levels) intersecting the boundary of $B\left( {q,\epsilon }\right)$ is bounded by

一般来说，边长为 $l$ 的 $d$ 维单元格网格中，与半径为 $\theta$ 的球体边界相交的单元格有 $O\left( {1 + {\left( \frac{\theta }{l}\right) }^{d - 1}}\right)$ 个（阿瑞亚和芒特，2000 年）。结合这一点以及 $i$ 级单元格边长为 $\epsilon /\left( {{2}^{i}\sqrt{d}}\right)$ 这一事实，我们知道与 $B\left( {q,\epsilon }\right)$ 的边界相交的（所有级别的）单元格总数受限于

$$
\mathop{\sum }\limits_{{i = 0}}^{{h - 1}}O\left( {1 + {\left( \frac{\epsilon }{\epsilon /\left( {{2}^{i}\sqrt{d}}\right) }\right) }^{d - 1}}\right)  = O\left( {\left( {2}^{h}\right) }^{d - 1}\right) 
$$

$$
 = O\left( {1 + {\left( 1/\rho \right) }^{d - 1}}\right) ,
$$

which is a constant for any fixed $\rho$ . This concludes the proof of Lemma 4.5.

对于任何固定的 $\rho$，这是一个常数。这就完成了引理 4.5 的证明。

### 4.4 Solving $\rho$ -Approximate DBSCAN

### 4.4 求解 $\rho$ -近似 DBSCAN 算法

We are now ready to solve the $\rho$ -approximate DBSCAN problem by proving:

现在我们准备通过证明以下内容来解决 $\rho$ -近似 DBSCAN 问题：

THEOREM 4.6. There is a $\rho$ -approximate DBSCAN algorithm that terminates in $O\left( n\right)$ expected time, regardless of the value of $\epsilon$ ,the constant approximation ratio $\rho$ ,and the fixed dimensionality $d$ .

定理 4.6。存在一种 $\rho$ -近似 DBSCAN 算法，无论 $\epsilon$ 的值、常数近似比 $\rho$ 以及固定维度 $d$ 如何，该算法都能在 $O\left( n\right)$ 的期望时间内终止。

Algorithm. Our $\rho$ -approximate algorithm differs from the exact algorithm we proposed in Section 3.2 only in the definition and computation of the graph $G$ . We re-define $G = \left( {V,E}\right)$ as follows:

算法。我们的 $\rho$ -近似算法与我们在 3.2 节中提出的精确算法的不同之处仅在于图 $G$ 的定义和计算。我们将 $G = \left( {V,E}\right)$ 重新定义如下：

-As before,each vertex in $V$ is a core cell of the grid $T$ (remember that the algorithm of Section 3.2 imposes a grid $T$ on ${\mathbb{R}}^{d}$ ,where a cell is a core cell if it covers at least one core point).

- 和之前一样，$V$ 中的每个顶点都是网格 $T$ 的核心单元格（请记住，3.2 节的算法在 ${\mathbb{R}}^{d}$ 上施加了一个网格 $T$，如果一个单元格至少覆盖一个核心点，则该单元格为核心单元格）。

- Given two different core cells ${c}_{1},{c}_{2}$ ,whether $E$ has an edge between ${c}_{1}$ and ${c}_{2}$ obeys the rules below:

- 给定两个不同的核心单元格 ${c}_{1},{c}_{2}$，$E$ 在 ${c}_{1}$ 和 ${c}_{2}$ 之间是否有边遵循以下规则：

-yes,if there exist core points ${p}_{1},{p}_{2}$ in ${c}_{1},{c}_{2}$ ,respectively,such that $\operatorname{dist}\left( {{p}_{1},{p}_{2}}\right)  \leq  \epsilon$ ; -no,if no core point in ${c}_{1}$ is within distance $\epsilon \left( {1 + \rho }\right)$ from any core point in ${c}_{2}$ ; - don't care, in all the other cases.

- 是，如果分别在 ${c}_{1},{c}_{2}$ 中存在核心点 ${p}_{1},{p}_{2}$，使得 $\operatorname{dist}\left( {{p}_{1},{p}_{2}}\right)  \leq  \epsilon$； - 否，如果 ${c}_{1}$ 中的任何核心点与 ${c}_{2}$ 中的任何核心点的距离都不在 $\epsilon \left( {1 + \rho }\right)$ 之内； - 其他情况则不关心。

To compute $G$ ,our algorithm starts by building,for each core cell $c$ in $T$ ,a structure of Lemma 4.5 on the set of core points in $c$ . To generate the edges of a core cell ${c}_{1}$ ,we examine each $\epsilon$ -neighbor cell ${c}_{2}$ of ${c}_{1}$ in turn. For every core point $p$ in ${c}_{1}$ ,do an approximate range count query on the set of core points in ${c}_{2}$ . If the query returns a non-zero answer,add an edge $\left( {{c}_{1},{c}_{2}}\right)$ to $G$ . If all such $p$ have been tried but still no edge has been added, we decide that there should be no edge between ${c}_{1}$ and ${c}_{2}$ .

为了计算 $G$，我们的算法首先为 $T$ 中的每个核心单元格 $c$，在 $c$ 中的核心点集上构建引理 4.5 的结构。为了生成核心单元格 ${c}_{1}$ 的边，我们依次检查 ${c}_{1}$ 的每个 $\epsilon$ -邻接单元格 ${c}_{2}$。对于 ${c}_{1}$ 中的每个核心点 $p$，对 ${c}_{2}$ 中的核心点集进行近似范围计数查询。如果查询返回非零答案，则向 $G$ 中添加一条边 $\left( {{c}_{1},{c}_{2}}\right)$。如果尝试了所有这样的 $p$ 但仍然没有添加边，我们就判定 ${c}_{1}$ 和 ${c}_{2}$ 之间不应有边。

Correctness. Let $C$ be an arbitrary cluster returned by our algorithm. We will show that $C$ satisfies Definition 4.2.

正确性。设 $C$ 为我们的算法返回的任意一个聚类。我们将证明 $C$ 满足定义 4.2。

Maximality. Let $p$ be an arbitrary core point in $C$ ,and $q$ be any point of $P$ density-reachable from $p$ . We will show that $q \in  C$ . Let us start by considering that $q$ is a core point. By Definition 2.2, there is a sequence of core points ${p}_{1},{p}_{2},\ldots ,{p}_{t}$ (for some integer $t \geq  2$ ) such that ${p}_{1} = p,{p}_{t} = q$ , and $\operatorname{dist}\left( {{p}_{i + 1},{p}_{i}}\right)  \leq  \epsilon$ for each $i \in  \left\lbrack  {1,t - 1}\right\rbrack$ . Denote by ${c}_{i}$ the cell of $T$ covering ${p}_{i}$ . By the way $G$ is defined,there must be an edge between ${c}_{i}$ and ${c}_{i + 1}$ ,for each $i \in  \left\lbrack  {1,t - 1}\right\rbrack$ . It thus follows that ${c}_{1}$ and ${c}_{t}$ must be in the same connected component of $G$ ; therefore, $p$ and $q$ must be in the same cluster. The correctness of the other scenario where $q$ is a non-core point is trivially guaranteed by the way that non-core points are assigned to clusters.

极大性。设 $p$ 为 $C$ 中的任意一个核心点，$q$ 为从 $p$ 出发 $P$ 密度可达的 $P$ 中的任意一点。我们将证明 $q \in  C$。首先考虑 $q$ 是核心点的情况。根据定义 2.2，存在一个核心点序列 ${p}_{1},{p}_{2},\ldots ,{p}_{t}$（对于某个整数 $t \geq  2$），使得 ${p}_{1} = p,{p}_{t} = q$，并且对于每个 $i \in  \left\lbrack  {1,t - 1}\right\rbrack$ 都有 $\operatorname{dist}\left( {{p}_{i + 1},{p}_{i}}\right)  \leq  \epsilon$。用 ${c}_{i}$ 表示覆盖 ${p}_{i}$ 的 $T$ 的单元格。根据 $G$ 的定义方式，对于每个 $i \in  \left\lbrack  {1,t - 1}\right\rbrack$，${c}_{i}$ 和 ${c}_{i + 1}$ 之间必定存在一条边。因此，${c}_{1}$ 和 ${c}_{t}$ 必定在 $G$ 的同一个连通分量中；所以，$p$ 和 $q$ 必定在同一个聚类中。当 $q$ 是非核心点时，根据非核心点分配到聚类的方式，这种情况的正确性显然是有保证的。

$\rho$ -Approximate Connectivity. Let $p$ be an arbitrary core point in $C$ . For any point $q \in  C$ ,we will show that $q$ is $\rho$ -approximate density-reachable from $p$ . Again,we consider first that $q$ is a core point. Let ${c}_{p}$ and ${c}_{q}$ be the cells of $T$ covering $p$ and $q$ ,respectively. Since ${c}_{p}$ and ${c}_{q}$ are in the same connected component of $G$ ,there is a path ${c}_{1},{c}_{2},\ldots ,{c}_{t}$ in $G$ (for some integer $t \geq  2$ ) such that ${c}_{1} = {c}_{p}$ and ${c}_{t} = {c}_{q}$ . Recall that any two points in the same cell are within distance $\epsilon$ . Combining this fact with how the edges of $G$ are defined,we know that there is a sequence of core points ${p}_{1},{p}_{2},\ldots ,{p}_{{t}^{\prime }}$ (for some integer ${t}^{\prime } \geq  2$ ) such that ${p}_{1} = p,{p}_{{t}^{\prime }} = q$ ,and $\operatorname{dist}\left( {{p}_{i + 1},{p}_{i}}\right)  \leq  \epsilon \left( {1 + \rho }\right)$ for each $i \in  \left\lbrack  {1,{t}^{\prime } - 1}\right\rbrack$ . Therefore, $q$ is $\rho$ -approximate density-reachable from $p$ . The correctness of the other scenario where $q$ is a non-core point is again trivial.

$\rho$ -近似连通性。设 $p$ 为 $C$ 中的任意一个核心点。对于任意点 $q \in  C$，我们将证明 $q$ 从 $p$ 出发是 $\rho$ -近似密度可达的。同样，我们首先考虑 $q$ 是核心点的情况。设 ${c}_{p}$ 和 ${c}_{q}$ 分别是覆盖 $p$ 和 $q$ 的 $T$ 的单元格。由于 ${c}_{p}$ 和 ${c}_{q}$ 在 $G$ 的同一个连通分量中，在 $G$ 中存在一条路径 ${c}_{1},{c}_{2},\ldots ,{c}_{t}$（对于某个整数 $t \geq  2$），使得 ${c}_{1} = {c}_{p}$ 和 ${c}_{t} = {c}_{q}$。回想一下，同一个单元格中的任意两点之间的距离在 $\epsilon$ 以内。结合这一事实以及 $G$ 的边的定义方式，我们知道存在一个核心点序列 ${p}_{1},{p}_{2},\ldots ,{p}_{{t}^{\prime }}$（对于某个整数 ${t}^{\prime } \geq  2$），使得 ${p}_{1} = p,{p}_{{t}^{\prime }} = q$，并且对于每个 $i \in  \left\lbrack  {1,{t}^{\prime } - 1}\right\rbrack$ 都有 $\operatorname{dist}\left( {{p}_{i + 1},{p}_{i}}\right)  \leq  \epsilon \left( {1 + \rho }\right)$。因此，$q$ 从 $p$ 出发是 $\rho$ -近似密度可达的。当 $q$ 是非核心点时，这种情况的正确性同样显而易见。

Time Analysis. It takes $O\left( n\right)$ expected time to construct the structure of Lemma 4.5 for all cells. The time of computing $G$ is proportional to the number of approximate range count queries issued. For each core point of a cell ${c}_{1}$ ,we issue $O\left( 1\right)$ queries in total (one for each $\epsilon$ -neighbor cell of ${c}_{2}$ ). Hence,the total number of queries is $O\left( n\right)$ . The rest of the $\rho$ -approximate algorithm runs in $O\left( n\right)$ expected time,following the same analysis as in Gunawan (2013). This completes the proof of Theorem 4.6. It is worth mentioning that, intuitively, the efficiency improvement of our approximate algorithm (over the exact algorithm in Section 3.2) owes to the fact that we settle for an imprecise solution to the BCP problem by using Lemma 4.5.

时间分析。为所有单元格构建引理4.5的结构需要$O\left( n\right)$的期望时间。计算$G$的时间与发出的近似范围计数查询的数量成正比。对于单元格${c}_{1}$的每个核心点，我们总共发出$O\left( 1\right)$个查询（${c}_{2}$的每个$\epsilon$ - 邻域单元格对应一个查询）。因此，查询的总数为$O\left( n\right)$。$\rho$ - 近似算法的其余部分在$O\left( n\right)$的期望时间内运行，分析过程与古纳万（Gunawan，2013年）的相同。至此，定理4.6的证明完成。值得一提的是，直观地说，我们的近似算法（相对于3.2节中的精确算法）的效率提升得益于我们通过使用引理4.5对BCP问题采用了不精确的解决方案。

<!-- Media -->

<!-- figureText: ${o}_{1}$ (b) Delaunay graph (c) Remainder graph after edge removal ${o}_{2}$ (a) Voronoi diagram -->

<img src="https://cdn.noedgeai.com/0195c91b-4734-77b1-933f-3c7cb33c7bfe_16.jpg?x=145&y=262&w=1272&h=534&r=0"/>

Fig. 8. Illustration of our Step-2 algorithm in Section 5.1.

图8. 5.1节中我们的步骤2算法的示意图。

<!-- Media -->

Remark. It should be noted that the hidden constant in $O\left( n\right)$ is at the order of ${\left( 1/\rho \right) }^{d - 1}$ ; see the proof of Lemma 4.5. As this is exponential to the dimensionality $d$ ,our techniques are suitable only when $d$ is low. Our experiments considered dimensionalities up to 7 .

备注。应当注意，$O\left( n\right)$中的隐藏常数的阶为${\left( 1/\rho \right) }^{d - 1}$；参见引理4.5的证明。由于这是维度$d$的指数函数，我们的技术仅适用于$d$较低的情况。我们的实验考虑的维度最高为7。

## 5 NEW 2D EXACT ALGORITHMS

## 5 新的二维精确算法

This section gives two new algorithms for solving the (exact) DBSCAN problem in ${\mathbb{R}}^{2}$ . These algorithms are based on different ideas, and are interesting in their own ways. The first one (Section 5.1) is conceptually simple, and establishes a close connection between DBSCAN and Delaunay graphs. The second one (Section 5.2) manages to identify coordinate sorting as the most expensive component in DBSCAN computation.

本节给出两种用于解决${\mathbb{R}}^{2}$中（精确的）DBSCAN问题的新算法。这些算法基于不同的思路，并且各自都很有趣。第一种（5.1节）在概念上很简单，并且在DBSCAN和德劳内图（Delaunay图）之间建立了紧密的联系。第二种（5.2节）成功地将坐标排序确定为DBSCAN计算中最耗时的部分。

### 5.1 DBSCAN from a Delaunay Graph

### 5.1 基于德劳内图（Delaunay图）的DBSCAN算法

Recall from Section 2.2 that Gunawan's algorithm runs in three steps:

回顾2.2节，古纳万（Gunawan）的算法分三个步骤运行：

(1) Label each point of the input set $P$ as either core or non-core.

(1) 将输入集$P$中的每个点标记为核心点或非核心点。

(2) Partition the set ${P}_{\text{core }}$ of core points into clusters.

(2) 将核心点集${P}_{\text{core }}$划分为多个簇。

(3) Assign each non-core point to the appropriate cluster(s).

(3) 将每个非核心点分配到合适的簇中。

Step 2 is the performance bottleneck. Next, we describe a new method to accomplish this step. Algorithm for Step 2. The Delaunay graph of ${P}_{\text{core }}$ can be regarded as the dual of the Voronoi diagram of ${P}_{\text{core }}$ . The latter is a subdivision of the data space ${\mathbb{R}}^{2}$ into $\left| {P}_{\text{core }}\right|$ convex polygons,each of which corresponds to a distinct $p \in  {P}_{\text{core }}$ ,and is called the Voronoi cell of $p$ ,containing every location in ${\mathbb{R}}^{2}$ that finds $p$ as its Euclidean nearest neighbor in ${P}_{\text{core }}$ . The Delaunay graph of ${P}_{\text{core }}$ is a graph ${G}_{dln} = \left( {{V}_{dln},{E}_{dln}}\right)$ defined as follows:

步骤2是性能瓶颈。接下来，我们描述一种完成此步骤的新方法。步骤2的算法。${P}_{\text{core }}$的德劳内图（Delaunay图）可以看作是${P}_{\text{core }}$的沃罗诺伊图（Voronoi图）的对偶图。后者是将数据空间${\mathbb{R}}^{2}$划分为$\left| {P}_{\text{core }}\right|$个凸多边形，每个凸多边形对应一个不同的$p \in  {P}_{\text{core }}$，并称为$p$的沃罗诺伊单元（Voronoi单元），它包含${\mathbb{R}}^{2}$中所有将$p$视为其在${P}_{\text{core }}$中的欧几里得最近邻的位置。${P}_{\text{core }}$的德劳内图（Delaunay图）是一个图${G}_{dln} = \left( {{V}_{dln},{E}_{dln}}\right)$，定义如下：

$- {V}_{dln} = {P}_{\text{core }}$ ,that is,every core point is a vertex of ${G}_{dln}$ .

$- {V}_{dln} = {P}_{\text{core }}$，即每个核心点都是${G}_{dln}$的一个顶点。

$- {E}_{dln}$ contains an edge between two core points ${p}_{1},{p}_{2}$ if and only if their Voronoi cells are adjacent (i.e., sharing a common boundary segment).

$- {E}_{dln}$在两个核心点${p}_{1},{p}_{2}$之间包含一条边，当且仅当它们的沃罗诺伊单元（Voronoi单元）相邻（即共享一条公共边界线段）。

${G}_{dln}$ ,in general,always has only a linear number of edges,i.e., $\left| {E}_{dln}\right|  = O\left( \left| {P}_{\text{core }}\right| \right)$ .

${G}_{dln}$通常总是只有线性数量的边，即$\left| {E}_{dln}\right|  = O\left( \left| {P}_{\text{core }}\right| \right)$。

Figure 8(a) demonstrates the Voronoi diagram defined by the set of black points shown. The shaded polygon is the Voronoi cell of ${o}_{1}$ ; the Voronoi cells of ${o}_{1}$ and ${o}_{2}$ are adjacent. The corresponding Delaunay graph is given in Figure 8(b).

图8(a)展示了由所示黑点集定义的沃罗诺伊图（Voronoi图）。阴影多边形是${o}_{1}$的沃罗诺伊单元（Voronoi单元）；${o}_{1}$和${o}_{2}$的沃罗诺伊单元（Voronoi单元）相邻。相应的德劳内图（Delaunay图）如图8(b)所示。

Provided that ${G}_{dln}$ is already available,we perform Step 2 using a simple strategy:

假设${G}_{dln}$已经可用，我们使用一种简单的策略执行步骤2：

(2.1) Remove all the edges $\left( {{p}_{1},{p}_{2}}\right)$ in ${E}_{\text{dln }}$ such that $\operatorname{dist}\left( {{p}_{1},{p}_{2}}\right)  > \epsilon$ . Let us refer to the resulting graph as the remainder graph.

(2.1) 移除${E}_{\text{dln }}$中所有满足$\operatorname{dist}\left( {{p}_{1},{p}_{2}}\right)  > \epsilon$的边$\left( {{p}_{1},{p}_{2}}\right)$。我们将得到的图称为剩余图。

(2.2) Compute the connected components of the remainder graph.

(2.2) 计算剩余图的连通分量。

(2.3) Put the core points in each connected component into a separate cluster.

(2.3) 将每个连通分量中的核心点放入一个单独的簇中。

Continuing the example in Figure 8(b), Figure 8(c) illustrates the remainder graph after the edge removal in Step 2.1 (the radius of the circle centered at point $p$ indicates the value of $\epsilon$ ). There are two connected components in the remainder graph; the core points in each connected component constitute a cluster.

继续图8(b)中的示例，图8(c)展示了在步骤2.1中移除边后得到的剩余图（以点$p$为圆心的圆的半径表示$\epsilon$的值）。剩余图中有两个连通分量；每个连通分量中的核心点构成一个簇。

In general,the Delaunay graph of $x$ 2D points can be computed in $O\left( {x\log x}\right)$ time (de Berg et al. 2008). Clearly,Steps 2.1-2.3 require only $O\left( \left| {P}_{\text{core }}\right| \right)  = O\left( n\right)$ time. Therefore,our Step 2 algorithm finishes in $O\left( {n\log n}\right)$ time overall.

一般来说，$x$个二维点的德劳内三角剖分图（Delaunay graph）可以在$O\left( {x\log x}\right)$时间内计算得到（de Berg等人，2008年）。显然，步骤2.1 - 2.3仅需要$O\left( \left| {P}_{\text{core }}\right| \right)  = O\left( n\right)$时间。因此，我们的步骤2算法总体上在$O\left( {n\log n}\right)$时间内完成。

Correctness of the Algorithm. It remains to explain why the above simple strategy correctly clusters the core points. Remember that a core point $p$ ought to be placed in the same cluster as another core point $q$ if and only if there is a sequence of core points ${p}_{1},{p}_{2},\ldots ,{p}_{t}$ (for some $t \geq  2$ ) such that

算法的正确性。还需要解释为什么上述简单策略能正确地对核心点进行聚类。请记住，当且仅当存在一个核心点序列${p}_{1},{p}_{2},\ldots ,{p}_{t}$（对于某个$t \geq  2$）使得

$$
 - {p}_{1} = p\text{and}{p}_{t} = q\text{,}
$$

$- \operatorname{dist}\left( {{p}_{i},{p}_{i + 1}}\right)  \leq  \epsilon$ for each $i \in  \left\lbrack  {1,t - 1}\right\rbrack$ .

$- \operatorname{dist}\left( {{p}_{i},{p}_{i + 1}}\right)  \leq  \epsilon$ 对于每个$i \in  \left\lbrack  {1,t - 1}\right\rbrack$成立。

We now prove:

我们现在证明：

LEMMA 5.1. Two core points $p,q$ belong to the same cluster if and only if our Step-2 algorithm declares so.

引理5.1。两个核心点$p,q$属于同一个簇，当且仅当我们的步骤2算法如此判定。

Proof. The If Direction. This direction is straightforward. Our algorithm declares $p,q$ to be in the same cluster only if they appear in the same connected component of the remainder graph obtained at Step 2.1. This, in turn, suggests that the connected component has a path starting from $p$ and ending at $q$ satisfying the aforementioned requirement.

证明。充分性方向。这个方向很直接。我们的算法仅当$p,q$出现在步骤2.1得到的剩余图的同一个连通分量中时，才判定它们属于同一个簇。这反过来表明，该连通分量有一条从$p$开始到$q$结束的路径，满足上述要求。

The Only-If Direction. Let $p,q$ be a pair of core points that should be placed in the same cluster. Next, we will prove that our Step-2 algorithm definitely puts them in the same connected component of the remainder graph.

必要性方向。设$p,q$是一对应该被放入同一个簇的核心点。接下来，我们将证明我们的步骤2算法肯定会将它们放入剩余图的同一个连通分量中。

We will first establish this fact by assuming $\operatorname{dist}\left( {p,q}\right)  \leq  \epsilon$ . Consider the line segment ${pq}$ . Since Voronoi cells are convex polygons,in moving on segment ${pq}$ from $p$ to $q$ ,we must be traveling through the Voronoi cells of a sequence of distinct core points-let them be ${p}_{1},{p}_{2},\ldots ,{p}_{t}$ for some $t \geq  2$ ,where ${p}_{1} = p$ and ${p}_{t} = q$ . Our goal is to show that $\operatorname{dist}\left( {{p}_{i},{p}_{i + 1}}\right)  \leq  \epsilon$ for all $i \in  \left\lbrack  {1,t - 1}\right\rbrack$ . This will indicate that the remainder graph must contain an edge between each pair of $\left( {{p}_{i},{p}_{i + 1}}\right)$ for all $i \in  \left\lbrack  {1,t - 1}\right\rbrack$ ,implying that all of ${p}_{1} = p,{p}_{2},\ldots ,{p}_{t} = q$ must be in the same connected component at Step 2.3.

我们首先通过假设$\operatorname{dist}\left( {p,q}\right)  \leq  \epsilon$来确立这一事实。考虑线段${pq}$。由于沃罗诺伊图（Voronoi diagram）的单元格是凸多边形，当我们沿着线段${pq}$从$p$移动到$q$时，我们必定会穿过一系列不同核心点的沃罗诺伊单元格 — 设它们为${p}_{1},{p}_{2},\ldots ,{p}_{t}$（对于某个$t \geq  2$），其中${p}_{1} = p$且${p}_{t} = q$。我们的目标是证明对于所有的$i \in  \left\lbrack  {1,t - 1}\right\rbrack$，$\operatorname{dist}\left( {{p}_{i},{p}_{i + 1}}\right)  \leq  \epsilon$成立。这将表明剩余图在每一对$\left( {{p}_{i},{p}_{i + 1}}\right)$（对于所有的$i \in  \left\lbrack  {1,t - 1}\right\rbrack$）之间必定包含一条边，这意味着所有的${p}_{1} = p,{p}_{2},\ldots ,{p}_{t} = q$在步骤2.3中必定处于同一个连通分量中。

We now prove $\operatorname{dist}\left( {{p}_{i},{p}_{i + 1}}\right)  \leq  \epsilon$ for an arbitrary $i \in  \left\lbrack  {1,t - 1}\right\rbrack$ . Let ${\widetilde{p}}_{i}$ (for $i \in  \left\lbrack  {1,t - 1}\right\rbrack$ ) be the intersection between ${pq}$ and the common boundary of the Voronoi cells of ${p}_{i}$ and ${p}_{i + 1}$ . Figure 9 illustrates the definition with an example where $t = 7$ . We will apply triangle inequality a number of times to arrive at our target conclusion. Let us start with

我们现在为任意的 $i \in  \left\lbrack  {1,t - 1}\right\rbrack$ 证明 $\operatorname{dist}\left( {{p}_{i},{p}_{i + 1}}\right)  \leq  \epsilon$。设 ${\widetilde{p}}_{i}$（对于 $i \in  \left\lbrack  {1,t - 1}\right\rbrack$）为 ${pq}$ 与 ${p}_{i}$ 和 ${p}_{i + 1}$ 的沃罗诺伊胞腔（Voronoi cells）公共边界的交集。图 9 用一个 $t = 7$ 的例子说明了这一定义。我们将多次应用三角不等式以得出我们的目标结论。让我们从

$$
\operatorname{dist}\left( {{p}_{i},{p}_{i + 1}}\right)  \leq  \operatorname{dist}\left( {{p}_{i},{\widetilde{p}}_{i}}\right)  + \operatorname{dist}\left( {{p}_{i + 1},{\widetilde{p}}_{i}}\right) . \tag{2}
$$

<!-- Media -->

<!-- figureText: ${p}_{2}$ ${p}_{4}$ ${\widetilde{p}}_{6}$ ${\widetilde{p}}_{4}$ $\left( {p}_{7}\right)$ ${p}_{6}$ $/{p}_{3}$ $\left( {p}_{1}\right)$ ${\widetilde{p}}_{1}$ ${\widetilde{p}}_{3}$ -->

<img src="https://cdn.noedgeai.com/0195c91b-4734-77b1-933f-3c7cb33c7bfe_18.jpg?x=357&y=263&w=855&h=314&r=0"/>

Fig. 9. Correctness proof of our Step-2 algorithm.

图 9. 我们的步骤 2 算法的正确性证明。

<!-- Media -->

Regarding dist $\left( {{p}_{i},{\widetilde{p}}_{i}}\right)$ ,we have

关于距离 $\left( {{p}_{i},{\widetilde{p}}_{i}}\right)$，我们有

$$
\operatorname{dist}\left( {{p}_{i},{\widetilde{p}}_{i}}\right)  \leq  \operatorname{dist}\left( {{p}_{i},{\widetilde{p}}_{i - 1}}\right)  + \operatorname{dist}\left( {{\widetilde{p}}_{i - 1},{\widetilde{p}}_{i}}\right) 
$$

$$
 = \operatorname{dist}\left( {{p}_{i - 1},{\widetilde{p}}_{i - 1}}\right)  + \operatorname{dist}\left( {{\widetilde{p}}_{i - 1},{\widetilde{p}}_{i}}\right) 
$$

$$
\text{(note:}\operatorname{dist}\left( {{p}_{i - 1},{\widetilde{p}}_{i - 1}}\right)  = \operatorname{dist}\left( {{p}_{i},{\widetilde{p}}_{i - 1}}\right) \text{as}{\widetilde{p}}_{i - 1}\text{is on the}
$$

$$
\text{perpendicular bisector of segment}\left. {{p}_{i}{p}_{i - 1}}\right) 
$$

$$
 \leq  \operatorname{dist}\left( {{p}_{i - 1},{\widetilde{p}}_{i - 2}}\right)  + \operatorname{dist}\left( {{\widetilde{p}}_{i - 2},{\widetilde{p}}_{i - 1}}\right)  + \operatorname{dist}\left( {{\widetilde{p}}_{i - 1},{\widetilde{p}}_{i}}\right) 
$$

$$
\text{(triangle inequality)}
$$

$$
 = \operatorname{dist}\left( {{p}_{i - 1},{\widetilde{p}}_{i - 2}}\right)  + \operatorname{dist}\left( {{\widetilde{p}}_{i - 2},{\widetilde{p}}_{i}}\right) 
$$

$$
\text{...}
$$

$$
 \leq  \operatorname{dist}\left( {{p}_{2},{\widetilde{p}}_{1}}\right)  + \operatorname{dist}\left( {{\widetilde{p}}_{1},{\widetilde{p}}_{i}}\right) 
$$

$$
 = \operatorname{dist}\left( {{p}_{1},{\widetilde{p}}_{1}}\right)  + \operatorname{dist}\left( {{\widetilde{p}}_{1},{\widetilde{p}}_{i}}\right) 
$$

$$
 = \operatorname{dist}\left( {{p}_{1},{\widetilde{p}}_{i}}\right) \text{.} \tag{3}
$$

Following a symmetric derivation, we have

通过对称推导，我们有

$$
\operatorname{dist}\left( {{p}_{i + 1},{\widetilde{p}}_{i}}\right)  \leq  \operatorname{dist}\left( {{\widetilde{p}}_{i},{p}_{t}}\right) . \tag{4}
$$

The combination of Equations (2)-(4) gives

将方程 (2) - (4) 组合可得

$$
\operatorname{dist}\left( {{p}_{i},{p}_{i + 1}}\right)  \leq  \operatorname{dist}\left( {{p}_{1},{\widetilde{p}}_{i}}\right)  + \operatorname{dist}\left( {{\widetilde{p}}_{i},{p}_{t}}\right) 
$$

$$
 = \operatorname{dist}\left( {{p}_{1},{p}_{t}}\right)  \leq  \epsilon 
$$

as claimed.

如所声称的那样。

We now get rid of the assumption that $\operatorname{dist}\left( {p,q}\right)  \leq  \epsilon$ . This is fairly easy. By the given fact that $p$ and $q$ should be placed in the same cluster,we know that there is a path ${p}_{1} = p,{p}_{2},{p}_{3},\ldots ,{p}_{t} = q$ (where $t \geq  2$ ) such that $\operatorname{dist}\left( {{p}_{i},{p}_{i + 1}}\right)  \leq  \epsilon$ for each $i \in  \left\lbrack  {1,t - 1}\right\rbrack$ . By our earlier argument,each pair of $\left( {{p}_{i},{p}_{i + 1}}\right)$ must be in the same connected component of our remainder graph. Consequently,all of ${p}_{1},{p}_{2},\ldots ,{p}_{t}$ are in the same connected component. This completes the proof.

我们现在去掉 $\operatorname{dist}\left( {p,q}\right)  \leq  \epsilon$ 这一假设。这相当容易。根据给定事实，即 $p$ 和 $q$ 应置于同一聚类中，我们知道存在一条路径 ${p}_{1} = p,{p}_{2},{p}_{3},\ldots ,{p}_{t} = q$（其中 $t \geq  2$），使得对于每个 $i \in  \left\lbrack  {1,t - 1}\right\rbrack$ 都有 $\operatorname{dist}\left( {{p}_{i},{p}_{i + 1}}\right)  \leq  \epsilon$。根据我们之前的论证，每一对 $\left( {{p}_{i},{p}_{i + 1}}\right)$ 必定位于我们剩余图的同一连通分量中。因此，所有的 ${p}_{1},{p}_{2},\ldots ,{p}_{t}$ 都位于同一连通分量中。证明完毕。

Remark. The concepts of Voronoi Diagram and Delaunay graph can both be extended to arbitrary dimensionality $d \geq  3$ . Our Step-2 algorithm also works for any $d \geq  3$ . While this may be interesting from a geometric point of view,it is not from an algorithmic perspective. Even at $d = 3$ ,a Delaunay graph on $n$ points can have $\Omega \left( {n}^{2}\right)$ edges,necessitating $\Omega \left( {n}^{2}\right)$ time for its computation. In contrast, in Section 3.2,we already showed that the exact DBSCAN problem can be solved in $o\left( {n}^{2}\right)$ time for any constant dimensionality $d$ .

注记。沃罗诺伊图（Voronoi Diagram）和德劳内图（Delaunay graph）的概念都可以扩展到任意维度 $d \geq  3$。我们的步骤 2 算法对于任何 $d \geq  3$ 也都适用。虽然从几何角度来看这可能很有趣，但从算法角度来看并非如此。即使在 $d = 3$ 时，$n$ 个点上的德劳内图可能有 $\Omega \left( {n}^{2}\right)$ 条边，这需要 $\Omega \left( {n}^{2}\right)$ 的时间来计算。相比之下，在 3.2 节中，我们已经表明对于任何恒定维度 $d$，精确的 DBSCAN 问题可以在 $o\left( {n}^{2}\right)$ 的时间内解决。

### 5.2 Separation of Sorting from DBSCAN

### 5.2 排序与 DBSCAN 的分离

We say that the $2\mathrm{D}$ input set $P$ is bi-dimensionally sorted if the points therein are given in two sorted lists:

我们称 $2\mathrm{D}$ 输入集 $P$ 是二维排序的，如果其中的点以两个排序列表给出：

$- {P}_{x}$ ,where the points are sorted by x-dimension; and

$- {P}_{x}$，其中的点按 x 维度排序；以及

$- {P}_{y}$ ,where the points are sorted by y-dimension.

$- {P}_{y}$，其中的点按 y 维度排序。

This subsection will establish the last main result of this article:

本小节将确立本文的最后一个主要结果：

THEOREM 5.2. If $P$ has been bi-dimensionally sorted,the exact DBSCAN problem (in 2D space) can be solved in $O\left( n\right)$ worst-case time.

定理 5.2。如果 $P$ 已经进行了二维排序，精确的 DBSCAN 问题（在二维空间中）可以在 $O\left( n\right)$ 的最坏情况下时间内解决。

The theorem reveals that coordinate sorting is actually the "hardest" part of the 2D DBSCAN problem! This means that we can even beat the $\Omega \left( {n\log n}\right)$ time bound for this problem in scenarios where sorting can be done fast. The corollaries below state two such scenarios:

该定理揭示了坐标排序实际上是二维 DBSCAN 问题中“最难”的部分！这意味着在排序可以快速完成的场景中，我们甚至可以突破该问题的 $\Omega \left( {n\log n}\right)$ 时间界限。以下推论陈述了两种这样的场景：

COROLLARY 5.3. If each dimension has an integer domain of size at most ${n}^{c}$ for an arbitrary positive constant $c$ ,the 2D DBSCAN problem can be solved in $O\left( n\right)$ worst-case time (even if $P$ is not bi-dimensionally sorted).

推论 5.3。如果每个维度的整数域大小至多为 ${n}^{c}$（对于任意正常数 $c$），二维 DBSCAN 问题可以在 $O\left( n\right)$ 的最坏情况下时间内解决（即使 $P$ 没有进行二维排序）。

Proof. Kirkpatrick and Reisch (1984) showed that $n$ integers drawn from a domain of size ${n}^{c}$ (regardless of the constant $c \geq  1$ ) can be sorted in $O\left( n\right)$ time,by generalizing the idea of radix sort. Using their algorithm, $P$ can be made bi-dimensionally sorted in $O\left( n\right)$ time. Then,the corollary follows from Theorem 5.2.

证明。柯克帕特里克（Kirkpatrick）和赖施（Reisch）（1984年）表明，通过推广基数排序的思想，从大小为${n}^{c}$的域中抽取的$n$个整数（无论常数$c \geq  1$的值如何）都可以在$O\left( n\right)$时间内完成排序。使用他们的算法，$P$可以在$O\left( n\right)$时间内进行二维排序。然后，该推论可由定理5.2得出。

The above corollary is important because, in real applications, (i) coordinates are always discrete (after digitalization),and (ii) when $n$ is large (e.g., ${10}^{6}$ ),the domain size of each dimension rarely exceeds ${n}^{2}$ . The 2D DBSCAN problem can be settled in linear time in all such applications.

上述推论很重要，因为在实际应用中，（i）坐标始终是离散的（数字化后），并且（ii）当$n$很大时（例如${10}^{6}$），每个维度的域大小很少超过${n}^{2}$。在所有此类应用中，二维DBSCAN问题都可以在线性时间内解决。

COROLLARY 5.4. If each dimension has an integer domain, the 2D DBSCAN problem can be solved in $O\left( {n\log \log n}\right)$ worst-case time or $O\left( {n\sqrt{\log \log n}}\right)$ expected time (even if $P$ is not bi-dimensionally sorted).

推论5.4。如果每个维度都有一个整数域，那么二维DBSCAN问题可以在最坏情况下的$O\left( {n\log \log n}\right)$时间内或期望的$O\left( {n\sqrt{\log \log n}}\right)$时间内解决（即使$P$没有进行二维排序）。

Proof. Andersson et al. (1998) gave a deterministic algorithm to sort $n$ integers in $O\left( {n\log \log n}\right)$ worst-case time. Han and Thorup (2002) gave a randomized algorithm to do so in $O\left( {n\sqrt{\log \log n}}\right)$ expected time. Plugging these results into Theorem 5.2 yields the corollary.

证明。安德森（Andersson）等人（1998年）给出了一种确定性算法，可在最坏情况下的$O\left( {n\log \log n}\right)$时间内对$n$个整数进行排序。韩（Han）和索鲁普（Thorup）（2002年）给出了一种随机算法，可在期望的$O\left( {n\sqrt{\log \log n}}\right)$时间内完成同样的排序。将这些结果代入定理5.2即可得出该推论。

Next, we provide the details of our algorithm for Theorem 5.2. The general framework is still the three-step process as shown in Section 5.1, but we will develop new methods to implement Steps 1 and 2 in linear time,utilizing the property that $P$ is bi-dimensionally sorted. Step 3 is carried out in the same manner as in Gunawan’s algorithm (Section 2.2),which demands only $O\left( n\right)$ time.

接下来，我们将详细介绍定理5.2的算法。总体框架仍然是5.1节中所示的三步过程，但我们将利用$P$已进行二维排序的特性，开发新的方法在线性时间内实现步骤1和步骤2。步骤3的执行方式与古纳万（Gunawan）的算法（2.2节）相同，只需要$O\left( n\right)$时间。

5.2.1 Step 1.. Recall that,for this step,Gunawan’s algorithm places an arbitrary grid $T$ (where each cell is a square with side length $\epsilon /\sqrt{2}$ ) in ${\mathbb{R}}^{2}$ ,and then proceeds as follows:

5.2.1 步骤1。回顾一下，对于这一步，古纳万的算法在${\mathbb{R}}^{2}$中放置一个任意网格$T$（其中每个单元格是边长为$\epsilon /\sqrt{2}$的正方形），然后按以下步骤进行：

(1.1) For each non-empty cell $c$ of $T$ ,compute the set $P\left( c\right)$ of points in $P$ that are covered by $c$ .

（1.1）对于$T$的每个非空单元格$c$，计算$P$中被$c$覆盖的点集$P\left( c\right)$。

(1.2) For each non-empty cell $c$ of $T$ ,identify all of its non-empty $\epsilon$ -neighbor cells ${c}^{\prime }$ (i.e.,the minimum distance between $c$ and ${c}^{\prime }$ is less than $\epsilon$ ).

（1.2）对于$T$的每个非空单元格$c$，找出其所有非空的$\epsilon$ - 邻接单元格${c}^{\prime }$（即$c$和${c}^{\prime }$之间的最小距离小于$\epsilon$）。

(1.3) Perform a labeling process to determine whether each point in $P$ is a core or non-core point.

（1.3）执行标记过程，以确定$P$中的每个点是核心点还是非核心点。

Our approach differs from Gunawan’s in Steps 1.1 and 1.2 (his solution to Step 1.3 takes only $O\left( n\right)$ time, and is thus sufficient for our purposes). Before continuing, note that Steps 1.1 and 1.2 can be done easily with hashing using $O\left( n\right)$ expected time,but our goal is to attain the same time complexity in the worst case.

我们的方法在步骤1.1和步骤1.2上与古纳万的方法不同（他对步骤1.3的解决方案只需要$O\left( n\right)$时间，因此足以满足我们的需求）。在继续之前，请注意，步骤1.1和步骤1.2可以使用哈希在期望的$O\left( n\right)$时间内轻松完成，但我们的目标是在最坏情况下达到相同的时间复杂度。

Step 1.1. We say that a column of $T$ (a column contains all the cells of $T$ sharing the same projection on the x-dimension) is non-empty if it has at least one non-empty cell. We label the leftmost nonempty column as 1,and the second leftmost non-empty column as 2,and so on. By scanning ${P}_{x}$ once in ascending order of $\mathrm{x}$ -coordinate,we determine,for each point $p \in  P$ ,the label of the nonempty column that contains $p$ ; the time required is $O\left( n\right)$ .

步骤1.1。我们称$T$的一列（一列包含$T$中所有在x维度上投影相同的单元格）为非空列，如果它至少有一个非空单元格。我们将最左边的非空列标记为1，第二左边的非空列标记为2，依此类推。通过按$\mathrm{x}$坐标的升序对${P}_{x}$进行一次扫描，我们可以确定每个点$p \in  P$所在的非空列的标签；所需时间为$O\left( n\right)$。

Suppose that there are ${n}_{col}$ non-empty columns. Next,for each $i \in  \left\lbrack  {1,{n}_{col}}\right\rbrack$ ,we generate a sorted list ${P}_{y}\left\lbrack  i\right\rbrack$ that arranges,in ascending of y-coordinate,the points of $P$ covered by (non-empty) column i. In other words,we aim to "distribute" ${P}_{y}$ into ${n}_{col}$ sorted lists,one for each non-empty column. This can be done in $O\left( n\right)$ time as follows. First,initialize all the ${n}_{col}$ lists to be empty. Then,scan ${P}_{y}$ in ascending order of $\mathrm{y}$ -coordinate; for each point $p$ seen,append it to ${P}_{y}\left\lbrack  i\right\rbrack$ where $i$ is the label of the column containing $p$ . The point ordering in ${P}_{y}$ ensures that each ${P}_{y}\left\lbrack  i\right\rbrack$ thus created is sorted on y-dimension.

假设存在 ${n}_{col}$ 个非空列。接下来，对于每个 $i \in  \left\lbrack  {1,{n}_{col}}\right\rbrack$，我们生成一个排序列表 ${P}_{y}\left\lbrack  i\right\rbrack$，该列表按照 y 坐标升序排列被（非空）列 i 覆盖的 $P$ 中的点。换句话说，我们的目标是将 ${P}_{y}$ “分配”到 ${n}_{col}$ 个排序列表中，每个非空列对应一个列表。这可以在 $O\left( n\right)$ 时间内完成，具体如下。首先，将所有 ${n}_{col}$ 个列表初始化为空。然后，按照 $\mathrm{y}$ 坐标升序扫描 ${P}_{y}$；对于看到的每个点 $p$，将其添加到 ${P}_{y}\left\lbrack  i\right\rbrack$ 中，其中 $i$ 是包含 $p$ 的列的标签。${P}_{y}$ 中的点排序确保了这样创建的每个 ${P}_{y}\left\lbrack  i\right\rbrack$ 在 y 维度上是有序的。

Finally,for each $i \in  \left\lbrack  {1,{n}_{col}}\right\rbrack$ ,we generate the target set $P\left( c\right)$ for every non-empty cell $c$ in column $i$ ,by simply scanning ${P}_{y}\left\lbrack  i\right\rbrack$ once in order to divide it into sub-sequences,each of which includes all the points in a distinct cell (sorted by y-coordinate). The overall cost of Step 1.1 is therefore $O\left( n\right)$ . As a side product,for every $i \in  \left\lbrack  {1,{n}_{col}}\right\rbrack$ ,we have also obtained a list ${L}_{i}$ of all the non-empty cells in column $i$ ,sorted in bottom-up order.

最后，对于每个 $i \in  \left\lbrack  {1,{n}_{col}}\right\rbrack$，我们为列 $i$ 中的每个非空单元格 $c$ 生成目标集 $P\left( c\right)$，只需扫描一次 ${P}_{y}\left\lbrack  i\right\rbrack$ 以将其划分为子序列，每个子序列包含一个不同单元格中的所有点（按 y 坐标排序）。因此，步骤 1.1 的总体成本为 $O\left( n\right)$。作为副产品，对于每个 $i \in  \left\lbrack  {1,{n}_{col}}\right\rbrack$，我们还获得了列 $i$ 中所有非空单元格的列表 ${L}_{i}$，该列表按从下到上的顺序排序。

Step 1.2. We do so by processing each non-empty column in turn. First, observe that if a cell is in column $i \in  \left\lbrack  {1,{n}_{col}}\right\rbrack$ ,all of its $\epsilon$ -neighbor cells must appear in columns $i - 2,i - 1,i,i + 1$ ,and $i + 2$ (see Figure 3(c)). Motivated by this,for each $j \in  \{ i - 2,i - 1,i,i + 1,i + 2\}  \cap  \left\lbrack  {1,{n}_{col}}\right\rbrack$ ,we scan synchronously the cells of ${L}_{i}$ and ${L}_{j}$ in bottom-up order (if two cells are at the same row,break the tie by scanning first the one from ${L}_{i}$ ). When a cell $c \in  {L}_{i}$ is encountered,we pinpoint the last cell ${c}_{0} \in  {L}_{j}$ that was scanned. Define

步骤 1.2。我们通过依次处理每个非空列来实现这一点。首先，观察到如果一个单元格位于列 $i \in  \left\lbrack  {1,{n}_{col}}\right\rbrack$ 中，其所有 $\epsilon$ 邻接单元格必须出现在列 $i - 2,i - 1,i,i + 1$ 和 $i + 2$ 中（见图 3(c)）。基于此，对于每个 $j \in  \{ i - 2,i - 1,i,i + 1,i + 2\}  \cap  \left\lbrack  {1,{n}_{col}}\right\rbrack$，我们按从下到上的顺序同步扫描 ${L}_{i}$ 和 ${L}_{j}$ 的单元格（如果两个单元格位于同一行，则先扫描来自 ${L}_{i}$ 的单元格以打破平局）。当遇到一个单元格 $c \in  {L}_{i}$ 时，我们确定最后扫描的单元格 ${c}_{0} \in  {L}_{j}$。定义

$- {c}_{-1}$ as the cell in ${L}_{j}$ immediately before ${c}_{0}$ ;

$- {c}_{-1}$ 为 ${L}_{j}$ 中紧接在 ${c}_{0}$ 之前的单元格；

$- {c}_{1}$ as the cell in ${L}_{j}$ immediately after ${c}_{0}$ ;

$- {c}_{1}$ 为 ${L}_{j}$ 中紧接在 ${c}_{0}$ 之后的单元格；

$- {c}_{2}$ as the cell in ${L}_{j}$ immediately after ${c}_{1}$ ;

$- {c}_{2}$ 为 ${L}_{j}$ 中紧接在 ${c}_{1}$ 之后的单元格；

$- {c}_{3}$ as the cell in ${L}_{j}$ immediately after ${c}_{2}$ ;

$- {c}_{3}$ 为 ${L}_{j}$ 中紧接在 ${c}_{2}$ 之后的单元格；

The five cells ${}^{3}{c}_{-1},{c}_{0},\ldots ,{c}_{3}$ are the only ones that can be $\epsilon$ -neighbors of $c$ in ${L}_{j}$ . Checking which of them are indeed $\epsilon$ -neighbors of $c$ takes $O\left( 1\right)$ time. Hence,the synchronous scan of ${L}_{i}$ and ${L}_{j}$ costs $O\left( {\left| {L}_{i}\right|  + \left| {L}_{j}\right| }\right)$ time. The total cost of Step 1.2 is,therefore, $O\left( n\right)$ ,noticing that each ${L}_{i}\left( {i \in  \left\lbrack  {1,{n}_{col}}\right\rbrack  }\right)$ will be scanned at most five times.

五个单元格 ${}^{3}{c}_{-1},{c}_{0},\ldots ,{c}_{3}$ 是在 ${L}_{j}$ 中唯一可能成为 $c$ 的 $\epsilon$ -邻域的单元格。检查其中哪些确实是 $c$ 的 $\epsilon$ -邻域需要 $O\left( 1\right)$ 的时间。因此，对 ${L}_{i}$ 和 ${L}_{j}$ 进行同步扫描需要 $O\left( {\left| {L}_{i}\right|  + \left| {L}_{j}\right| }\right)$ 的时间。注意到每个 ${L}_{i}\left( {i \in  \left\lbrack  {1,{n}_{col}}\right\rbrack  }\right)$ 最多会被扫描五次，因此步骤 1.2 的总成本为 $O\left( n\right)$。

Remark. By slightly extending the above algorithm,for each non-empty cell $c$ ,we can store the points of $P\left( c\right)$ in two sorted lists:

备注：通过对上述算法进行略微扩展，对于每个非空单元格 $c$，我们可以将 $P\left( c\right)$ 中的点存储在两个排序列表中：

$- {P}_{x}\left( c\right)$ ,where the points of $P\left( c\right)$ are sorted on $\mathrm{x}$ -dimension;

$- {P}_{x}\left( c\right)$，其中 $P\left( c\right)$ 中的点按 $\mathrm{x}$ 维度排序；

$- {P}_{y}\left( c\right)$ ,where the points are sorted on y-dimension.

$- {P}_{y}\left( c\right)$，其中的点按 y 维度排序。

To achieve this purpose, first observe that, at the end of Step 1.1, the sub-sequence obtained for each non-empty cell $c$ is precisely ${P}_{y}\left( c\right)$ . This allows us to know,for each point $p \in  P$ ,the id of the non-empty cell covering it. After this,the ${P}_{x}\left( c\right)$ of all non-empty cells $c$ can be obtained with just another scan of ${P}_{x}$ : for each point $p$ seen in ${P}_{x}$ ,append it to ${P}_{x}\left( c\right)$ ,where $c$ is the cell containing $p$ . The point ordering in ${P}_{x}$ ensures that each ${P}_{x}\left( c\right)$ is sorted by x-coordinate,as desired. The additional time required is still $O\left( n\right)$ .

为了实现这一目的，首先观察到，在步骤 1.1 结束时，为每个非空单元格 $c$ 获得的子序列恰好是 ${P}_{y}\left( c\right)$。这使我们能够为每个点 $p \in  P$ 确定覆盖它的非空单元格的 ID。在此之后，只需对 ${P}_{x}$ 进行另一次扫描，就可以获得所有非空单元格 $c$ 的 ${P}_{x}\left( c\right)$：对于在 ${P}_{x}$ 中看到的每个点 $p$，将其追加到 ${P}_{x}\left( c\right)$ 中，其中 $c$ 是包含 $p$ 的单元格。${P}_{x}$ 中的点排序确保每个 ${P}_{x}\left( c\right)$ 都按 x 坐标排序，符合要求。所需的额外时间仍然是 $O\left( n\right)$。

5.2.2 Step 2.. For this step,Gunawan’s algorithm generates a graph $G = \left( {V,E}\right)$ where each core cell in $T$ corresponds to a distinct vertex in $V$ . Between core cells (a.k.a.,vertices) ${c}_{1}$ and ${c}_{2}$ ,an edge exists in $E$ if and only if there is a core point ${p}_{1}$ in ${c}_{1}$ and a core point ${p}_{2}$ in ${c}_{2}$ such that $\operatorname{dist}\left( {{p}_{1},{p}_{2}}\right)  \leq$ $\epsilon$ . Once $G$ is available,Step 2 is accomplished in $O\left( n\right)$ time by computing the connected components of $G$ . The performance bottleneck lies in the creation of $G$ ,to which Gunawan’s solution takes $O\left( {n\log n}\right)$ time. We develop a new algorithm below that fulfills the purpose in $O\left( n\right)$ time.

5.2.2 步骤 2。对于此步骤，古纳万（Gunawan）的算法生成一个图 $G = \left( {V,E}\right)$，其中 $T$ 中的每个核心单元格对应于 $V$ 中的一个不同顶点。当且仅当在 ${c}_{1}$ 中有一个核心点 ${p}_{1}$ 且在 ${c}_{2}$ 中有一个核心点 ${p}_{2}$ 使得 $\operatorname{dist}\left( {{p}_{1},{p}_{2}}\right)  \leq$ $\epsilon$ 时，在核心单元格（也就是顶点）${c}_{1}$ 和 ${c}_{2}$ 之间，$E$ 中存在一条边。一旦 $G$ 可用，通过计算 $G$ 的连通分量，步骤 2 可以在 $O\left( n\right)$ 的时间内完成。性能瓶颈在于 $G$ 的创建，古纳万的解决方案在这方面需要 $O\left( {n\log n}\right)$ 的时间。我们下面开发一种新算法，能够在 $O\left( n\right)$ 的时间内实现这一目的。

---

<!-- Footnote -->

${}^{3}$ If ${c}_{0} = \varnothing$ (namely,no cell in ${L}_{j}$ has been scanned),set ${c}_{1},{c}_{2},{c}_{3}$ to the lowest three cells in ${L}_{j}$ .

${}^{3}$ 如果 ${c}_{0} = \varnothing$（即，${L}_{j}$ 中没有单元格被扫描），则将 ${c}_{1},{c}_{2},{c}_{3}$ 设置为 ${L}_{j}$ 中最下面的三个单元格。

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: ✘ ✘ ✘ -->

<img src="https://cdn.noedgeai.com/0195c91b-4734-77b1-933f-3c7cb33c7bfe_21.jpg?x=453&y=262&w=661&h=419&r=0"/>

Fig. 10. USEC with line separation.

图 10. 带有线分隔的 USEC（使用线分隔的 USEC）。

<!-- Media -->

USEC with Line Separation. Let us introduce a special variant of the USEC problem defined in Section 2.3,which stands at the core of our $O\left( n\right)$ -time algorithm. Recall that in the 2D USEC problem,we are given a set ${S}_{\text{ball }}$ of disks with the same radius $\epsilon$ ,and a set ${S}_{pt}$ of points,all in the data space ${\mathbb{R}}^{2}$ . The objective is to determine whether any point in ${S}_{pt}$ is covered by any disk in ${S}_{\text{ball }}$ . In our special variant,there are two extra constraints:

带线分隔的USEC（带线分隔的通用圆盘覆盖点问题）。让我们引入在2.3节中定义的USEC（通用圆盘覆盖点问题）问题的一个特殊变体，它是我们$O\left( n\right)$时间算法的核心。回顾一下，在二维USEC问题中，我们给定了一组半径均为$\epsilon$的圆盘${S}_{\text{ball }}$，以及一组点${S}_{pt}$，所有这些都位于数据空间${\mathbb{R}}^{2}$中。目标是确定${S}_{pt}$中的任何点是否被${S}_{\text{ball }}$中的任何圆盘覆盖。在我们的特殊变体中，有两个额外的约束条件：

-There is a horizontal line $\ell$ such that (i) all the centers of the disks in ${S}_{\text{ball }}$ are on or below $\ell$ ,and (ii) all the points in ${S}_{pt}$ are on or above $\ell$ .

- 存在一条水平线$\ell$，使得（i）${S}_{\text{ball }}$中所有圆盘的圆心都在$\ell$上或$\ell$下方，并且（ii）${S}_{pt}$中的所有点都在$\ell$上或$\ell$上方。

- The centers of the disks in ${S}_{\text{ball }}$ have been sorted by x-dimension,and so are the points in ${S}_{pt}$ .

- ${S}_{\text{ball }}$中圆盘的圆心已按x维度排序，${S}_{pt}$中的点也已按x维度排序。

Figure 10 illustrates an instance of the above USEC with line separation problem (where crosses indicate disk centers). The answer to this instance is yes (i.e., a point falls in a disk).

图10展示了上述带线分隔的USEC问题的一个实例（其中十字表示圆盘的圆心）。这个实例的答案是肯定的（即，有一个点落在一个圆盘中）。

LEMMA 5.5. The USEC with line separation problem can be settled in linear time, namely, with cost $O\left( {\left| {S}_{pt}\right|  + \left| {S}_{\text{ball }}\right| }\right)$ .

引理5.5。带线分隔的USEC问题可以在线性时间内解决，即，成本为$O\left( {\left| {S}_{pt}\right|  + \left| {S}_{\text{ball }}\right| }\right)$。

An algorithm for achieving the above lemma is implied in Bose et al. [2007]. However, the description in Bose et al. [2007] is rather brief, and does not provide the full details. In the Appendix, we reconstruct their algorithm, and prove its correctness (such a proof was missing in Bose et al. [2007]). Nonetheless, we believe that credits on the lemma should be attributed to Bose et al. [2007]. The reader may also see de Berg et al. (2015) for another account of the algorithm.

实现上述引理的一种算法在博斯（Bose）等人[2007]的论文中有提及。然而，博斯等人[2007]的描述相当简略，没有提供完整的细节。在附录中，我们重构了他们的算法，并证明了其正确性（博斯等人[2007]的论文中缺少这样的证明）。尽管如此，我们认为该引理的功劳应归于博斯等人[2007]。读者也可以参考德贝尔格（de Berg）等人（2015）的论文以了解该算法的另一种描述。

Generating $G$ in $O\left( n\right)$ Time. We now return to our endeavor of finding an $O\left( n\right)$ time algorithm to generate $G$ . The vertices of $G$ ,which are precisely the core cells,can obviously be collected in $O\left( n\right)$ time (there are at most $n$ core cells). It remains to discuss the creation of the edges in $G$ . Now,focus on any two core cells ${c}_{1}$ and ${c}_{2}$ that are $\epsilon$ -neighbors of each other. Our mission is to determine whether there should be an edge between them. It turns out that this requires solving at most two instances of USEC with line separation. Following our earlier terminology,let $P\left( {c}_{1}\right)$ be the set of points of $P$ that fall in ${c}_{1}$ . Recall that we have already obtained two sorted lists of $P\left( {c}_{1}\right)$ , that is, ${P}_{x}\left( {c}_{1}\right)$ and ${P}_{y}\left( {c}_{1}\right)$ that are sorted by x- and y-dimension,respectively. Define $P\left( {c}_{2}\right) ,{P}_{x}\left( {c}_{2}\right)$ , and ${P}_{y}\left( {c}_{2}\right)$ similarly for ${c}_{2}$ . Depending on the relative positions of ${c}_{1}$ and ${c}_{2}$ ,we proceed differently in the following two cases (which essentially have represented all possible cases by symmetry):

在$O\left( n\right)$时间内生成$G$。现在我们回到寻找一个$O\left( n\right)$时间算法来生成$G$的工作上。$G$的顶点，也就是核心单元，显然可以在$O\left( n\right)$时间内收集到（最多有$n$个核心单元）。接下来需要讨论$G$中边的创建。现在，关注任意两个互为$\epsilon$ - 邻接的核心单元${c}_{1}$和${c}_{2}$。我们的任务是确定它们之间是否应该有一条边。结果表明，这最多需要解决两个带线分隔的USEC问题实例。按照我们之前的术语，设$P\left( {c}_{1}\right)$为$P$中落在${c}_{1}$内的点的集合。回顾一下，我们已经得到了$P\left( {c}_{1}\right)$的两个排序列表，即分别按x维度和y维度排序的${P}_{x}\left( {c}_{1}\right)$和${P}_{y}\left( {c}_{1}\right)$。类似地，为${c}_{2}$定义$P\left( {c}_{2}\right) ,{P}_{x}\left( {c}_{2}\right)$和${P}_{y}\left( {c}_{2}\right)$。根据${c}_{1}$和${c}_{2}$的相对位置，我们在以下两种情况（通过对称性，这两种情况本质上代表了所有可能的情况）中采取不同的处理方式：

<!-- Media -->

<!-- figureText: ${c}_{2}$ ${c}_{2}$ $\ell$ ${c}_{1}$ (b) Case 2 ${c}_{1}$ (a) Case 1 -->

<img src="https://cdn.noedgeai.com/0195c91b-4734-77b1-933f-3c7cb33c7bfe_22.jpg?x=511&y=264&w=546&h=302&r=0"/>

Fig. 11. Deciding the existence of an edge by USEC with line separation.

图11. 通过带线分隔的USEC确定边的存在性。

<!-- Media -->

- Case 1: ${c}_{2}$ is in the same column as ${c}_{1}$ ,and is above ${c}_{1}$ ,as in Figure 11(a). Imagine placing a disk centered at each core point in $P\left( {c}_{1}\right)$ . All these discs constitute ${S}_{\text{ball }}$ . Set ${S}_{pt}$ to the set of core points in $P\left( {c}_{2}\right)$ . Together with the horizontal line $\ell$ shown,this defines an instance of USEC with line separation. There is an edge between ${c}_{1},{c}_{2}$ if and only if the instance has a yes answer.

- 情况1：${c}_{2}$与${c}_{1}$在同一列，且位于${c}_{1}$上方，如图11(a)所示。设想在$P\left( {c}_{1}\right)$中的每个核心点放置一个圆盘。所有这些圆盘构成${S}_{\text{ball }}$。将${S}_{pt}$设为$P\left( {c}_{2}\right)$中的核心点集合。结合所示的水平线$\ell$，这定义了一个具有线分隔的USEC（无向分离边覆盖，Undirected Separated Edge Cover）实例。当且仅当该实例的答案为“是”时，${c}_{1},{c}_{2}$之间存在一条边。

- Case 2: ${c}_{2}$ is to the northeast of ${c}_{1}$ ,as in Figure 11(b). Define ${S}_{\text{ball }}$ and ${S}_{pt}$ in the same manner as before. They define an instance of USEC with line separation based on $\ell$ . There is an edge between ${c}_{1},{c}_{2}$ if and only if the instance has a yes answer.

- 情况2：${c}_{2}$在${c}_{1}$的东北方向，如图11(b)所示。以与之前相同的方式定义${S}_{\text{ball }}$和${S}_{pt}$。它们基于$\ell$定义了一个具有线分隔的USEC实例。当且仅当该实例的答案为“是”时，${c}_{1},{c}_{2}$之间存在一条边。

It is immediately clear from Lemma 5.5 that we can make the correct decision about the edge existence between ${c}_{1},{c}_{2}$ using $O\left( {\left| {P\left( {c}_{1}\right) }\right|  + \left| {P\left( {c}_{2}\right) }\right| }\right)$ time. Therefore,the total cost of generating all the edges in $G$ is bounded by

从引理5.5可以立即清楚地看出，我们可以在$O\left( {\left| {P\left( {c}_{1}\right) }\right|  + \left| {P\left( {c}_{2}\right) }\right| }\right)$时间内对${c}_{1},{c}_{2}$之间是否存在边做出正确的决策。因此，在$G$中生成所有边的总成本受限于

$$
\mathop{\sum }\limits_{{\text{core cell }{c}_{1}}}\left( {\mathop{\sum }\limits_{{\epsilon \text{-neighbor }{c}_{2}\text{ of }{c}_{1}}}O\left( {\left| {P\left( {c}_{1}\right) }\right|  + \left| {P\left( {c}_{2}\right) }\right| }\right) }\right)  = \mathop{\sum }\limits_{{\text{core cell }{c}_{1}}}O\left( \left| {P\left( {c}_{1}\right) }\right| \right)  = O\left( n\right) ,
$$

where the first equality used the fact that each core cell has $O\left( 1\right) \epsilon$ -neighbors,and hence,can participate in only $O\left( 1\right)$ instances of USEC with line separation.

其中第一个等式利用了每个核心单元格有$O\left( 1\right) \epsilon$个邻居这一事实，因此，它只能参与$O\left( 1\right)$个具有线分隔的USEC实例。

## 6 DISCUSSION ON PRACTICAL EFFICIENCY

## 6 实际效率讨论

Besides our theoretical findings, we have developed a software prototype based on the proposed algorithms. Our implementation has evolved beyond that of Gan and Tao (2015) by incorporating new heuristics (note also that Gan and Tao (2015) focused on $d \geq  3$ ). Next,we will explain the most crucial heuristics adopted which apply to all of our algorithms (since they are based on the same grid-based framework). Then, we will discuss when the original DBSCAN algorithm of Ester et al. (1996) is or is not expected to work well in practice. Finally, a qualitative comparison of the precise and $\rho$ -approximate DBSCAN algorithms will be presented.

除了我们的理论研究结果，我们还基于所提出的算法开发了一个软件原型。我们的实现比 Gan 和 Tao（2015 年）的实现更先进，融入了新的启发式方法（还需注意，Gan 和 Tao（2015 年）主要关注$d \geq  3$）。接下来，我们将解释所采用的适用于我们所有算法的最关键的启发式方法（因为它们基于相同的基于网格的框架）。然后，我们将讨论 Ester 等人（1996 年）提出的原始 DBSCAN 算法在实际应用中何时效果好，何时效果不好。最后，将对精确的和$\rho$ - 近似的 DBSCAN 算法进行定性比较。

Heuristics. The three most effective heuristics in our implementation can be summarized as follows:

启发式方法。我们实现中最有效的三种启发式方法可总结如下：

-Recall that our $\rho$ -approximate algorithm imposes a grid $T$ on ${\mathbb{R}}^{d}$ . We manage all the nonempty cells in a (main memory) R-tree which is constructed by bulkloading. This R-tree allows us to efficiently find,for any cell $c$ ,all its $\epsilon$ -neighbor non-empty cells ${c}^{\prime }$ . Recall that such an operation is useful in a number of scenarios: (i) in the labeling process when a point $p$ falls in a cell covering less than MinPts points,(ii) in deciding the edges of $c$ in $G$ ,and (iii) assigning a non-core point in $c$ to appropriate clusters.

- 回顾一下，我们的$\rho$ - 近似算法在${\mathbb{R}}^{d}$上施加了一个网格$T$。我们使用通过批量加载构建的（主存）R - 树来管理所有非空单元格。这个 R - 树使我们能够有效地为任何单元格$c$找到其所有的$\epsilon$ - 邻接非空单元格${c}^{\prime }$。回想一下，这样的操作在许多场景中都很有用：（i）在标记过程中，当点$p$落在覆盖少于 MinPts 个点的单元格中时；（ii）在确定$G$中$c$的边时；（iii）将$c$中的非核心点分配到合适的聚类中。

-For every non-empty cell $c$ ,we store all its $\epsilon$ -neighbor non-empty cells in a list,after they have been computed for the first time. As each list has length $O\left( 1\right)$ ,the total space of all the lists is $O\left( n\right)$ (recall that at most $n$ non-empty cells exist). The lists allow us to avoid re-computing $\epsilon$ -neighbor non-empty cells of $c$ . -Theoretically speaking,we achieve $O\left( n\right)$ expected time by first generating the edges of $G$ and then computing its connected components (CC). In reality, it is faster not to produce the edges, but instead, maintain the CCs using a union-find structure (Tarjan 1979).

- 对于每个非空单元格$c$，在首次计算出其所有的$\epsilon$ - 邻接非空单元格后，我们将它们存储在一个列表中。由于每个列表的长度为$O\left( 1\right)$，所有列表的总空间为$O\left( n\right)$（回想一下，最多存在$n$个非空单元格）。这些列表使我们能够避免重新计算$c$的$\epsilon$ - 邻接非空单元格。 - 从理论上讲，我们通过先生成$G$的边，然后计算其连通分量（CC）来达到$O\left( n\right)$的期望时间。实际上，不生成边，而是使用并查集结构（Tarjan 1979 年）来维护连通分量会更快。

<!-- Media -->

<img src="https://cdn.noedgeai.com/0195c91b-4734-77b1-933f-3c7cb33c7bfe_23.jpg?x=380&y=262&w=776&h=310&r=0"/>

Fig. 12. A small $\epsilon$ for the left cluster is large for the other two clusters.

图 12. 对于左侧聚类来说较小的$\epsilon$，对于另外两个聚类来说则较大。

<!-- Media -->

Specifically,whenever an edge between non-empty cells $c$ and ${c}^{\prime }$ is found,we perform a "union" operation using $c$ and ${c}^{\prime }$ on the structure. After all the edges have been processed like this, the final CCs can be easily determined by issuing a "find" operation on every nonempty cell. In theory,this approach entails $O\left( {n \cdot  \alpha \left( n\right) }\right)$ time,where $\alpha \left( n\right)$ is the inverse of the Ackermann which is extremely slow growing such that $\alpha \left( n\right)$ is very small for all practical $n$ .

具体来说，每当在非空单元格$c$和${c}^{\prime }$之间找到一条边时，我们就在该结构上使用$c$和${c}^{\prime }$执行“合并”操作。在所有边都以这种方式处理后，通过对每个非空单元格执行“查找”操作，就可以轻松确定最终的连通分量。从理论上讲，这种方法需要$O\left( {n \cdot  \alpha \left( n\right) }\right)$的时间，其中$\alpha \left( n\right)$是阿克曼函数的反函数，它的增长极其缓慢，以至于对于所有实际的$n$，$\alpha \left( n\right)$的值都非常小。

An advantage of this approach is that, it avoids a large amount of edge detection that was needed in Gan and Tao (2015). Before, such detection was performed for each pair of non-empty cells $c$ and ${c}^{\prime }$ that were $\epsilon$ -neighbors of each other. Now,we can safely skip the detection if these cells are already found to be in the same CC.

这种方法的一个优点是，它避免了 Gan 和 Tao（2015 年）中所需的大量边检测。以前，对于每对互为$\epsilon$ - 邻接的非空单元格$c$和${c}^{\prime }$都要进行这样的检测。现在，如果发现这些单元格已经在同一个连通分量中，我们就可以安全地跳过检测。

Characteristics of the KDD'96 Algorithm. As mentioned in Section 1.1, the running time of the algorithm in Ester et al. (1996) is determined by the total cost of $n$ region queries,each of which retrieves $B\left( {p,\epsilon }\right)$ for each $p \in  P$ . Our hardness result in Theorem 3.1 implies that,even if each $B\left( {p,\epsilon }\right)$ returns just $p$ itself,the cost of all $n$ queries must still sum up to $\Omega \left( {n}^{4/3}\right)$ for a hard dataset.

KDD'96 算法的特点。如第 1.1 节所述，Ester 等人（1996 年）提出的算法的运行时间由$n$区域查询的总成本决定，每次查询为每个$p \in  P$检索$B\left( {p,\epsilon }\right)$。我们在定理 3.1 中的困难性结果表明，即使每个$B\left( {p,\epsilon }\right)$只返回$p$本身，对于一个困难的数据集，所有$n$查询的成本仍然必须总计达到$\Omega \left( {n}^{4/3}\right)$。

As reasonably argued by Ester et al. (1996),on practical data,the cost of a region query $B\left( {p,\epsilon }\right)$ depends on how many points are in $B\left( {p,\epsilon }\right)$ . The KDD’96 algorithm may have acceptable efficiency when $\epsilon$ is small such that the total number of points returned by all the region queries is near linear. Such a value of $\epsilon$ ,however,may not exist when the clusters have varying densities. Consider the example in Figure 12 where there are three clusters. Suppose that $\operatorname{MinPts} = 4$ . To discover the sparsest cluster on the left, $\epsilon$ needs to be at least the radius of the circles illustrated. For each point $p$ from the right (i.e.,the densest) cluster,however,the $B\left( {p,\epsilon }\right)$ under such an $\epsilon$ covers a big fraction of the cluster. On this dataset, therefore, the algorithm of Ester et al. (1996) either does not discover all three clusters, or must do so with expensive cost.

正如埃斯特（Ester）等人（1996年）合理论证的那样，在实际数据中，区域查询$B\left( {p,\epsilon }\right)$的成本取决于$B\left( {p,\epsilon }\right)$中有多少个点。当$\epsilon$较小时，KDD’96算法可能具有可接受的效率，这样所有区域查询返回的点的总数接近线性。然而，当聚类具有不同密度时，可能不存在这样的$\epsilon$值。考虑图12中的示例，其中有三个聚类。假设$\operatorname{MinPts} = 4$。为了发现左侧最稀疏的聚类，$\epsilon$至少需要是所示圆圈的半径。然而，对于右侧（即最密集）聚类中的每个点$p$，在这样的$\epsilon$下的$B\left( {p,\epsilon }\right)$覆盖了该聚类的很大一部分。因此，在这个数据集上，埃斯特等人（1996年）的算法要么无法发现所有三个聚类，要么必须以高昂的成本来实现。

A Comparison. The preceding discussion suggests that the relative superiority between the KDD’96 algorithm and our proposed $\rho$ -approximate algorithm depends primarily on two factors: (i) whether the cluster densities are similar or varying,and (ii) whether the value of $\epsilon$ is small or large. For a dataset with varying-density clusters, our algorithm is expected to perform better because,as explained,a good $\epsilon$ that finds all clusters must be relatively large for the dense clusters, forcing the KDD'96 algorithm to entail high cost on those clusters.

比较。前面的讨论表明，KDD’96算法和我们提出的$\rho$ - 近似算法之间的相对优势主要取决于两个因素：（i）聚类密度是相似还是不同，以及（ii）$\epsilon$的值是小还是大。对于具有不同密度聚类的数据集，我们的算法预计表现更好，因为正如所解释的，要找到所有聚类的合适$\epsilon$对于密集聚类来说必须相对较大，这迫使KDD'96算法在这些聚类上产生高成本。

For a dataset with similar-density clusters,the KDD’96 algorithm can be faster when $\epsilon$ is sufficiently small. In fact,our empirical experience indicates a pattern: when the $\rho$ -approximate algorithm is slower,the grid $T$ it imposes on ${\mathbb{R}}^{d}$ has $\Omega \left( n\right)$ non-empty cells-more specifically,we observe that the cutoff threshold is roughly $n/\sqrt{2}$ cells,regardless of $d$ . This makes sense because, in such a case, most non-empty cells have very few points (e.g., one or two), thus the extra overhead of creating and processing the grid no longer pays off.

对于具有相似密度聚类的数据集，当$\epsilon$足够小时，KDD’96算法可能更快。事实上，我们的经验表明了一种模式：当$\rho$ - 近似算法较慢时，它在${\mathbb{R}}^{d}$上施加的网格$T$有$\Omega \left( n\right)$个非空单元格——更具体地说，我们观察到，无论$d$如何，截止阈值大约是$n/\sqrt{2}$个单元格。这是有道理的，因为在这种情况下，大多数非空单元格中的点非常少（例如，一个或两个），因此创建和处理网格的额外开销不再值得。

The above observations will be verified in the next section.

上述观察结果将在下一节中得到验证。

## 7 EXPERIMENTS

## 7 实验

The philosophy of the following experiments differs from that in the short version (Gan and Tao 2015). Specifically, Gan and Tao (2015) treated DBSCAN clustering as a computer science problem, and aimed to demonstrate the quadratic nature of the previous DBSCAN algorithms for $d \geq  3$ . In this work, we regard DBSCAN as an application, and will focus on parameter values that are more important in practice.

以下实验的理念与简短版本（甘（Gan）和陶（Tao）2015年）中的不同。具体来说，甘和陶（2015年）将DBSCAN聚类视为一个计算机科学问题，旨在证明之前针对$d \geq  3$的DBSCAN算法的二次性质。在这项工作中，我们将DBSCAN视为一个应用程序，并将专注于在实践中更重要的参数值。

All the experiments were run on a machine equipped with ${3.4}\mathrm{{GHz}}\mathrm{{CPU}}$ and ${16}\mathrm{{GB}}$ memory. The operating system was Linux (Ubuntu 14.04). All the programs were coded in C++, and compiled using g++ with -o3 turned on.

所有实验都在配备了${3.4}\mathrm{{GHz}}\mathrm{{CPU}}$和${16}\mathrm{{GB}}$内存的机器上运行。操作系统是Linux（Ubuntu 14.04）。所有程序都用C++编写，并使用g++编译，同时开启了 - o3选项。

Section 7.1 describes the datasets in our experimentation, after which Section 7.2 seeks parameter values that lead to meaningful clusters on those data. The evaluation of the proposed techniques will then proceed in three parts. First, Section 7.3 assesses the clustering precision of $\rho$ -approximate DBSCAN. Section 7.4 demonstrates the efficiency gain achieved by our approximation algorithm compared to exact DBSCAN in dimensionality $d \geq  3$ . Finally,Section 7.5 examines the performance of exact DBSCAN algorithms for $d = 2$ .

第7.1节描述了我们实验中的数据集，之后第7.2节将寻找能在这些数据上形成有意义聚类的参数值。然后，对所提出技术的评估将分三个部分进行。首先，第7.3节评估$\rho$ - 近似DBSCAN的聚类精度。第7.4节展示了我们的近似算法与精确DBSCAN在维度$d \geq  3$上相比所实现的效率提升。最后，第7.5节考察针对$d = 2$的精确DBSCAN算法的性能。

### 7.1 Datasets

### 7.1 数据集

In all datasets,the underlying data space had a normalized integer domain of $\left\lbrack  {0,{10}^{5}}\right\rbrack$ for every dimension. We deployed both synthetic and real datasets whose details are explained next.

在所有数据集中，基础数据空间的每个维度都有一个归一化的整数域$\left\lbrack  {0,{10}^{5}}\right\rbrack$。我们同时使用了合成数据集和真实数据集，其详细信息将在下面解释。

Synthetic: Seed Spreader (SS). A synthetic dataset was generated in a "random walk with restart" fashion. First,fix the dimensionality $d$ ,take the target cardinality $n$ ,a restart probability ${\rho }_{\text{restart }}$ ,and a noise percentage ${\rho }_{\text{noise }}$ . Then,we simulate a seed spreader that moves about in the space, and spits out data points around its current location. The spreader carries a local counter such that whenever the counter reaches 0,the spreader moves a distance of ${r}_{\text{shift }}$ toward a random direction,after which the counter is reset to ${c}_{\text{reset }}$ . The spreader works in steps. In each step,(i) with probability ${\rho }_{\text{restart }}$ ,the spreader restarts,by jumping to a random location in the data space,and resetting its counter to ${c}_{\text{reset }}$ ; (ii) no matter if a restart has happened,the spreader produces a point uniformly at random in the ball centered at its current location with radius ${r}_{\text{vincinity }}$ ,after which the local counter decreases by 1 . Intuitively, every time a restart happens, the spreader begins to generate a new cluster. In the first step, a restart is forced so as to put the spreader at a random location. We repeat in total $n\left( {1 - {\rho }_{\text{noise }}}\right)$ steps,which generate the same number of points. Finally, we add $n \cdot  {\rho }_{\text{noise }}$ noise points,each of which is uniformly distributed in the whole space.

合成数据：种子扩散器（SS）。合成数据集以“带重启的随机游走”方式生成。首先，固定维度$d$，确定目标基数$n$、重启概率${\rho }_{\text{restart }}$和噪声百分比${\rho }_{\text{noise }}$。然后，我们模拟一个在空间中移动的种子扩散器，并在其当前位置周围生成数据点。扩散器带有一个局部计数器，每当计数器达到 0 时，扩散器会向随机方向移动${r}_{\text{shift }}$的距离，之后计数器重置为${c}_{\text{reset }}$。扩散器按步骤工作。在每一步中，(i) 以概率${\rho }_{\text{restart }}$，扩散器重启，即跳转到数据空间中的随机位置，并将其计数器重置为${c}_{\text{reset }}$；(ii) 无论是否发生重启，扩散器都会在以其当前位置为中心、半径为${r}_{\text{vincinity }}$的球内均匀随机地生成一个点，之后局部计数器减 1。直观地说，每次重启发生时，扩散器开始生成一个新的聚类。在第一步，强制进行一次重启，以便将扩散器置于随机位置。我们总共重复$n\left( {1 - {\rho }_{\text{noise }}}\right)$步，这将生成相同数量的点。最后，我们添加$n \cdot  {\rho }_{\text{noise }}$个噪声点，每个噪声点在整个空间中均匀分布。

Figure 13 shows a small 2D dataset which was generated with $n = 1,{000}$ and four restarts; the dataset will be used for visualization. The other experiments used larger datasets created with ${c}_{\text{reset }} = {100},{\rho }_{\text{noise }} = 1/{10}^{4},{\rho }_{\text{restart }} = {10}/\left( {n\left( {1 - {\rho }_{\text{noise }}}\right) }\right)$ . In expectation,around 10 restarts occur in the generation of a dataset. The values of ${r}_{\text{vincinity }}$ and ${r}_{\text{shift }}$ were set in two different ways to produce clusters with either similar or varying densities:

图 13 展示了一个小型二维数据集，该数据集使用$n = 1,{000}$生成并进行了四次重启；该数据集将用于可视化。其他实验使用了通过${c}_{\text{reset }} = {100},{\rho }_{\text{noise }} = 1/{10}^{4},{\rho }_{\text{restart }} = {10}/\left( {n\left( {1 - {\rho }_{\text{noise }}}\right) }\right)$创建的更大数据集。预计在数据集生成过程中大约会发生 10 次重启。${r}_{\text{vincinity }}$和${r}_{\text{shift }}$的值通过两种不同的方式设置，以生成具有相似或不同密度的聚类：

- Similar-density dataset: Namely, the clusters have roughly the same density. Such a dataset was obtained by fixing ${r}_{\text{vincinity }} = {100}$ and ${r}_{\text{shift }} = {50d}$ .

- 相似密度数据集：即，聚类的密度大致相同。通过固定${r}_{\text{vincinity }} = {100}$和${r}_{\text{shift }} = {50d}$可获得这样的数据集。

- Varying-density dataset: Namely, the clusters have different densities. Such a dataset was obtained by setting ${r}_{\text{vincinity }} = {100} \cdot  \left( {\left( {i{\;\operatorname{mod}\;{10}}}\right)  + 1}\right)$ and ${r}_{\text{shift }} = {r}_{\text{vincinity }} \cdot  d/2$ ,where $i$ equals

- 不同密度数据集：即，聚类具有不同的密度。通过设置${r}_{\text{vincinity }} = {100} \cdot  \left( {\left( {i{\;\operatorname{mod}\;{10}}}\right)  + 1}\right)$和${r}_{\text{shift }} = {r}_{\text{vincinity }} \cdot  d/2$可获得这样的数据集，其中$i$等于

<!-- Media -->

<img src="https://cdn.noedgeai.com/0195c91b-4734-77b1-933f-3c7cb33c7bfe_25.jpg?x=530&y=266&w=501&h=434&r=0"/>

Fig. 13. A 2D seed spreader dataset.

图 13. 二维种子扩散器数据集。

Table 1. Parameter Values (Defaults are in Bold)

表 1. 参数值（默认值用粗体表示）

<table><tr><td>Parameter</td><td>Values</td></tr><tr><td>$n$ (synthetic)</td><td>100k, 0.5m, 1m, 2m, 5m, 10m</td></tr><tr><td>$d$ (synthetic)</td><td>2,3,5,7</td></tr><tr><td>$\epsilon$</td><td>from 100 (or 40 for $d = 2$ ) to 5,000 (each dataset has its own default)</td></tr><tr><td>MinPts</td><td>10, 20, 40, 60, 100 (each dataset has its own default)</td></tr><tr><td>$\rho$</td><td>0.001, 0.01, 0.02, ..., 0.1</td></tr></table>

<table><tbody><tr><td>参数</td><td>值</td></tr><tr><td>$n$（合成的）</td><td>100k、0.5m、1m、2m、5m、10m</td></tr><tr><td>$d$（合成的）</td><td>2,3,5,7</td></tr><tr><td>$\epsilon$</td><td>从100（对于$d = 2$为40）到5000（每个数据集有其自己的默认值）</td></tr><tr><td>最小点数（MinPts）</td><td>10、20、40、60、100（每个数据集有其自己的默认值）</td></tr><tr><td>$\rho$</td><td>0.001, 0.01, 0.02, ..., 0.1</td></tr></tbody></table>

<!-- Media -->

the number of restarts that have taken place (at the beginning $i = 0$ ). Note that the "modulo 10 " ensures that there are at most 10 different cluster densities.

已发生的重启次数（开始时为 $i = 0$ ）。请注意，“模 10” 确保最多有 10 种不同的簇密度。

The value of $n$ ranged from 100k to 10 million,while $d$ from 2 to 7 (see Table 1). Hereafter,by SS-simden- ${dD}$ ,we refer to a $d$ -dimensional similar-density dataset (the default cardinality is $2\mathrm{\;m}$ ),while by ${SS}$ -varden- ${dD}$ ,we refer to a $d$ -dimensional varying-density dataset (same default on cardinality). Real. Three real datasets were employed in our experimentation:

$n$ 的值范围从 10 万到 1000 万，而 $d$ 的值范围从 2 到 7（见表 1）。此后，通过 SS - simden - ${dD}$，我们指的是一个 $d$ 维的相似密度数据集（默认基数为 $2\mathrm{\;m}$ ），而通过 ${SS}$ - varden - ${dD}$，我们指的是一个 $d$ 维的可变密度数据集（基数的默认值相同）。真实数据集。我们的实验中使用了三个真实数据集：

- The first one, PAMAP2, is a four-dimensional dataset with cardinality 3,850,505, obtained by taking the first four principle components of a PCA on the PAMAP2 database (Reiss and Stricker 2012) from the UCI machine-learning archive (Bache and Lichman 2013).

- 第一个数据集 PAMAP2 是一个四维数据集，基数为 3850505，它是通过对 UCI 机器学习存档（Bache 和 Lichman 2013）中的 PAMAP2 数据库（Reiss 和 Stricker 2012）进行主成分分析（PCA）并取前四个主成分得到的。

- The second one, Farm, is a five-dimensional dataset with cardinality 3,627,086, which contains the VZ-features (Varma and Zisserman 2003) of a satellite image of a farm in Saudi Arabia. ${}^{4}$ It is worth noting that VZ-feature clustering is a common approach to perform color segmentation of an image (Varma and Zisserman 2003).

- 第二个数据集 Farm 是一个五维数据集，基数为 3627086，它包含沙特阿拉伯一个农场卫星图像的 VZ 特征（Varma 和 Zisserman 2003）。${}^{4}$ 值得注意的是，VZ 特征聚类是进行图像颜色分割的常用方法（Varma 和 Zisserman 2003）。

- The third one, Household, is a seven-dimensional dataset with cardinality 2,049,280, which includes all the attributes of the Household database again from the UCI archive (Bache and Lichman 2013) except the temporal columns date and time. Points in the original database with missing coordinates were removed.

- 第三个数据集 Household 是一个七维数据集，基数为 2049280，它包含 UCI 存档（Bache 和 Lichman 2013）中 Household 数据库的所有属性，但不包括时间列日期和时间。原始数据库中坐标缺失的点已被移除。

---

<!-- Footnote -->

${}^{4}$ http://www.satimagingcorp.com/gallery/ikonos/ikonos-tadco-farms-saudi-arabia.

${}^{4}$ http://www.satimagingcorp.com/gallery/ikonos/ikonos - tadco - farms - saudi - arabia.

<!-- Footnote -->

---

### 7.2 Characteristics of the Datasets

### 7.2 数据集的特征

This subsection aims to study the clusters in each dataset under different parameters, and thereby, decide the values of MinPts and $\epsilon$ suitable for the subsequent efficiency experiments.

本小节旨在研究不同参数下每个数据集中的簇，从而确定适合后续效率实验的 MinPts 和 $\epsilon$ 的值。

Clustering Validation Index. We resorted to a method called clustering validation (CV) (Moulavi et al. 2014) whose objective is to quantify the quality of clustering using a real value. In general, a set of good density-based clusters should have two properties: first, the points in a cluster should be "tightly" connected; second, any two points belonging to different clusters should have a large distance. To quantify the first property for a cluster $C$ ,we compute a Euclidean minimum spanning tree (EMST) on the set of core points in $C$ ,and then,define $\operatorname{DSC}\left( C\right)$ as the maximum weight of the edges in the EMST. "DSC" stands for density sparseness of a cluster, a term used by Moulavi et al. [2014]. Intuitively,the EMST is a "backbone" of $C$ such that if $C$ is tightly connected, ${DSC}\left( C\right)$ ought to be small. Note that the border points of $C$ are excluded because they are not required to have a dense vicinity. To quantify the second property,define $\operatorname{DSPC}\left( {{C}_{i},{C}_{j}}\right)$ between two clusters ${C}_{i}$ and ${C}_{j}$ as

聚类验证指标。我们采用了一种称为聚类验证（CV）的方法（Moulavi 等人，2014），其目标是使用一个实数值来量化聚类的质量。一般来说，一组好的基于密度的簇应该具有两个属性：首先，一个簇中的点应该 “紧密” 相连；其次，属于不同簇的任意两点之间应该有较大的距离。为了量化簇 $C$ 的第一个属性，我们在 $C$ 中的核心点集上计算一个欧几里得最小生成树（EMST），然后将 $\operatorname{DSC}\left( C\right)$ 定义为 EMST 中边的最大权重。“DSC” 代表簇的密度稀疏性，这是 Moulavi 等人 [2014] 使用的术语。直观地说，EMST 是 $C$ 的 “骨干”，因此如果 $C$ 紧密相连，${DSC}\left( C\right)$ 应该较小。请注意，$C$ 的边界点被排除在外，因为它们不需要有密集的邻域。为了量化第二个属性，将两个簇 ${C}_{i}$ 和 ${C}_{j}$ 之间的 $\operatorname{DSPC}\left( {{C}_{i},{C}_{j}}\right)$ 定义为

$$
\min \left\{  {\operatorname{dist}\left( {{p}_{1},{p}_{2}}\right)  \mid  {p}_{1} \in  {C}_{1}}\right. \text{and}{p}_{2} \in  {C}_{2}\text{are core points}\} \text{,}
$$

where "DSPC" stands for density separation for a pair of clusters (Moulavi et al. 2014). Let $\mathcal{C} =$ $\left\{  {{C}_{1},{C}_{2},\ldots ,{C}_{t}}\right\}$ (where $t \geq  2$ ) be a set of clusters returned by an algorithm. For each ${C}_{i}$ ,we define (following Moulavi et al. (2014))

其中 “DSPC” 代表一对簇的密度分离（Moulavi 等人，2014）。设 $\mathcal{C} =$ $\left\{  {{C}_{1},{C}_{2},\ldots ,{C}_{t}}\right\}$（其中 $t \geq  2$ ）是一个算法返回的一组簇。对于每个 ${C}_{i}$，我们定义（遵循 Moulavi 等人（2014））

$$
{V}_{\mathcal{C}}\left( {C}_{i}\right)  = \frac{\left( {\mathop{\min }\limits_{{1 \leq  j \leq  t,j \neq  i}}\operatorname{DSPC}\left( {{C}_{i},{C}_{j}}\right) }\right)  - \operatorname{DSC}\left( {C}_{i}\right) }{\max \left\{  {\operatorname{DSC}\left( {C}_{i}\right) ,\mathop{\min }\limits_{{1 \leq  j \leq  t,j \neq  i}}\operatorname{DSPC}\left( {{C}_{i},{C}_{j}}\right) }\right\}  }.
$$

Then,the ${CV}$ index of $\mathcal{C}$ is calculated as in Moulavi et al. [2014]:

然后，$\mathcal{C}$ 的 ${CV}$ 指标按照 Moulavi 等人 [2014] 的方法计算如下：

$$
\mathop{\sum }\limits_{{i = 1}}^{t}\frac{\left| {C}_{i}\right| }{n}{V}_{\mathcal{C}}\left( {C}_{i}\right) 
$$

where $n$ is the size of the dataset. A higher validity index indicates better quality of $\mathcal{C}$ .

其中 $n$ 是数据集的大小。有效性指标越高，表明 $\mathcal{C}$ 的质量越好。

Moulavi et al. [2014] computed ${DSC}\left( {C}_{i}\right)$ and ${DSPC}\left( {{C}_{i},{C}_{j}}\right)$ differently,but their approach requires $O\left( {n}^{2}\right)$ time which is intolerably long for the values of $n$ considered here. Our proposition follows the same rationale, admits faster implementation (EMST is well studied (Agarwal et al. 1991; Arya and Mount 2016)), and worked well in our experiments as shown below.

穆拉维等人 [2014] 以不同方式计算了 ${DSC}\left( {C}_{i}\right)$ 和 ${DSPC}\left( {{C}_{i},{C}_{j}}\right)$，但他们的方法需要 $O\left( {n}^{2}\right)$ 的时间，对于此处考虑的 $n$ 值而言，这一时间长得令人难以忍受。我们的提议遵循相同的原理，允许更快的实现（欧几里得最小生成树（EMST）已得到充分研究（阿加瓦尔等人 1991 年；阿亚和蒙特 2016 年）），并且如下面所示，在我们的实验中效果良好。

Influence of MinPts and $\epsilon$ on DBSCAN Clusters. For each dataset,we examined the quality of its clusters under different combinations of MinPts and $\epsilon$ . For MinPts,we inspected values 10 and 100,while for $\epsilon$ ,we inspected a wide range starting from $\epsilon  = {40}$ and 100 for $d = 2$ and $d \geq  3$ , respectively. Only two values of MinPts were considered because (i) either 10 or 100 worked well on the synthetic and real data deployed, and (ii) the number of combinations was already huge.

最小点数（MinPts）和 $\epsilon$ 对 DBSCAN 聚类的影响。对于每个数据集，我们研究了在最小点数（MinPts）和 $\epsilon$ 的不同组合下其聚类的质量。对于最小点数（MinPts），我们考察了 10 和 100 这两个值，而对于 $\epsilon$，我们分别考察了从 $\epsilon  = {40}$ 开始的较宽范围以及针对 $d = 2$ 和 $d \geq  3$ 的 100。只考虑最小点数（MinPts）的两个值，原因在于：（i）10 或 100 在部署的合成数据和真实数据上效果都很好；（ii）组合的数量已经非常庞大。

Table 2 presents some key statistics for ${SS}$ -simden- ${dD}$ datasets with $d = 2,3,5$ and 7,while Table 3 shows the same statistics for ${SS}$ -varden- ${dD}$ . Remember that the cardinality here is $n = 2\mathrm{\;m}$ , implying that there should be around 200 noise points. The number of intended clusters should not exceed the number of restarts whose expectation is 10 . But the former number can be smaller, because the seed spreader may not necessarily create a new cluster after a restart, if it happens to jump into the region of a cluster already generated.

表 2 给出了 ${SS}$ -simden- ${dD}$ 数据集在 $d = 2,3,5$ 和 7 情况下的一些关键统计信息，而表 3 展示了 ${SS}$ -varden- ${dD}$ 的相同统计信息。请记住，此处的基数为 $n = 2\mathrm{\;m}$，这意味着应该大约有 200 个噪声点。预期聚类的数量不应超过重启次数，重启次数的期望值为 10。但前者的数量可能更小，因为如果种子传播器在重启后恰好跳入已生成聚类的区域，它不一定会创建新的聚类。

Both MinPts = 10 and 100,when coupled with an appropriate $\epsilon$ ,were able to discover all the intended clusters-observe that the CV index stabilizes soon as $\epsilon$ increases. We set 10 as the default for MinPts on the synthetic datasets, as it produced better clusters than 100 under most values of $\epsilon$ . Notice that,for varying-density datasets, $\epsilon$ needed to be larger to ensure good clustering quality (compared to similar-density datasets). This is due to the reason explained in Section 6 (cf. Figure 12). The bold $\epsilon$ values in Tables 2 and 3 were chosen as the default for the corresponding datasets (they were essentially the smallest that gave good clusters).

当最小点数（MinPts）取值为 10 和 100 并与合适的 $\epsilon$ 结合时，都能够发现所有预期的聚类 —— 可以观察到，随着 $\epsilon$ 的增加，变异系数（CV）指数很快趋于稳定。我们将 10 设为合成数据集上最小点数（MinPts）的默认值，因为在大多数 $\epsilon$ 值下，它产生的聚类效果比 100 更好。请注意，对于密度变化的数据集，$\epsilon$ 需要更大才能确保良好的聚类质量（与密度相似的数据集相比）。这是由于第 6 节中解释的原因（参见图 12）。表 2 和表 3 中加粗的 $\epsilon$ 值被选为相应数据集的默认值（它们实际上是能产生良好聚类的最小值）。

<!-- Media -->

Table 2. Cluster Quality Under Different $\left( {\text{MinPts},\epsilon }\right)$ : SS Similar Density

表 2. 不同 $\left( {\text{MinPts},\epsilon }\right)$ 下的聚类质量：SS 相似密度

<table><tr><td rowspan="2">$\epsilon$</td><td colspan="3">MinPts = 10</td><td colspan="3">MinPts = 100</td></tr><tr><td>CV Index</td><td>#clusters</td><td>#noise pts</td><td>CV Index</td><td>#clusters</td><td>#noise pts</td></tr><tr><td>40</td><td>0.978</td><td>10</td><td>325</td><td>0.555</td><td>230</td><td>309,224</td></tr><tr><td>60</td><td>0.994</td><td>9</td><td>197</td><td>0.577</td><td>72</td><td>33,489</td></tr><tr><td>80</td><td>0.994</td><td>9</td><td>197</td><td>0.994</td><td>9</td><td>506</td></tr><tr><td>100</td><td>0.994</td><td>9</td><td>197</td><td>0.994</td><td>9</td><td>197</td></tr><tr><td>200</td><td>0.994</td><td>9</td><td>197</td><td>0.994</td><td>9</td><td>197</td></tr><tr><td colspan="7">(a) SS-simden-2D</td></tr></table>

<table><tbody><tr><td rowspan="2">$\epsilon$</td><td colspan="3">最小点数（MinPts） = 10</td><td colspan="3">最小点数（MinPts） = 100</td></tr><tr><td>CV指数</td><td>聚类数量（#clusters）</td><td>噪声点数量（#noise pts）</td><td>CV指数</td><td>聚类数量（#clusters）</td><td>噪声点数量（#noise pts）</td></tr><tr><td>40</td><td>0.978</td><td>10</td><td>325</td><td>0.555</td><td>230</td><td>309,224</td></tr><tr><td>60</td><td>0.994</td><td>9</td><td>197</td><td>0.577</td><td>72</td><td>33,489</td></tr><tr><td>80</td><td>0.994</td><td>9</td><td>197</td><td>0.994</td><td>9</td><td>506</td></tr><tr><td>100</td><td>0.994</td><td>9</td><td>197</td><td>0.994</td><td>9</td><td>197</td></tr><tr><td>200</td><td>0.994</td><td>9</td><td>197</td><td>0.994</td><td>9</td><td>197</td></tr><tr><td colspan="7">(a) SS - 相似密度 - 二维（SS - simden - 2D）</td></tr></tbody></table>

<table><tr><td rowspan="2">$\epsilon$</td><td colspan="3">MinPts = 10</td><td colspan="3">MinPts = 100</td></tr><tr><td>CV Index</td><td>#clusters</td><td>#noise pts</td><td>CV Index</td><td>#clusters</td><td>#noise pts</td></tr><tr><td>100</td><td>0.996</td><td>14</td><td>200</td><td>0.205</td><td>240</td><td>467</td></tr><tr><td>200</td><td>0.996</td><td>14</td><td>200</td><td>0.996</td><td>14</td><td>200</td></tr><tr><td>400</td><td>0.996</td><td>14</td><td>200</td><td>0.996</td><td>14</td><td>200</td></tr><tr><td>800</td><td>0.996</td><td>14</td><td>200</td><td>0.996</td><td>14</td><td>200</td></tr><tr><td>1,000</td><td>0.996</td><td>14</td><td>200</td><td>0.996</td><td>14</td><td>200</td></tr><tr><td colspan="7">(b) SS-simden-3D</td></tr></table>

<table><tbody><tr><td rowspan="2">$\epsilon$</td><td colspan="3">最小点数（MinPts） = 10</td><td colspan="3">最小点数（MinPts） = 100</td></tr><tr><td>变异系数指数（CV Index）</td><td>聚类数量（#clusters）</td><td>噪声点数量（#noise pts）</td><td>变异系数指数（CV Index）</td><td>聚类数量（#clusters）</td><td>噪声点数量（#noise pts）</td></tr><tr><td>100</td><td>0.996</td><td>14</td><td>200</td><td>0.205</td><td>240</td><td>467</td></tr><tr><td>200</td><td>0.996</td><td>14</td><td>200</td><td>0.996</td><td>14</td><td>200</td></tr><tr><td>400</td><td>0.996</td><td>14</td><td>200</td><td>0.996</td><td>14</td><td>200</td></tr><tr><td>800</td><td>0.996</td><td>14</td><td>200</td><td>0.996</td><td>14</td><td>200</td></tr><tr><td>1,000</td><td>0.996</td><td>14</td><td>200</td><td>0.996</td><td>14</td><td>200</td></tr><tr><td colspan="7">(b) 基于结构相似度的三维密度聚类（SS - simden - 3D）</td></tr></tbody></table>

(b) SS-simden-3D

(b) SS-相似度密度-3D（SS-simden-3D）

MinPts $= {10}$ MinPts = 100

最小点数 $= {10}$ 最小点数 = 100

$\epsilon$ CV Index #clusters #noise pts CV Index #clusters #noise pts

$\epsilon$ 轮廓系数（CV Index） 簇数量（#clusters） 噪声点数量（#noise pts） 轮廓系数（CV Index） 簇数量（#clusters） 噪声点数量（#noise pts）

100 0.102 4721 219 0.583 19,057 632

200 0.996 13 200 0.996 13 241

400 0.996 13 200 0.996 13 200

800 0.996 13 200 0.996 13 200

1,000 0.996 13 200 0.996 13 200

(c) SS-simden-5D

(c) SS-相似度密度-5D（SS-simden-5D）

MinPts = 10 MinPts = 100

最小点数 = 10 最小点数 = 100

$\epsilon$ CV Index #clusters #noise pts CV Index #clusters #noise pts

$\epsilon$ 轮廓系数（CV Index） 簇数量（#clusters） 噪声点数量（#noise pts） 轮廓系数（CV Index） 簇数量（#clusters） 噪声点数量（#noise pts）

100 0.588 19,824 215 0.705 19,822 1,000

200 0.403 14,988 215 0.403 14,976 998

400 0.992 17 200 0.992 17 200

800 0.984 17 200 0.984 17 200

1,000 0.980 17 200 0.980 17 200

(d) SS-simden-7D

(d) SS-相似度密度-7D（SS-simden-7D）

<!-- Media -->

Figure 14 plots the OPTICS diagrams ${}^{5}$ for ${SS}$ -simden-5D and ${SS}$ -varden-5D,obtained with MinPts = 10. In an OPTICS diagram (Ankerst et al. 1999), the data points are arranged into a sequence as given along the $\mathrm{x}$ -axis. The diagram shows the area beneath a function $f\left( x\right)  : \left\lbrack  {1,n}\right\rbrack   \rightarrow  \mathbb{R}$ , where $f\left( x\right)$ can be understood roughly as follows: if $p$ is the $x$ -th point in the sequence,then $f\left( x\right)$ is the smallest $\epsilon$ value which (together with the chosen MinPts) puts $p$ into some cluster-in other words, $p$ remains as a noise point for $\epsilon  < f\left( x\right)$ . A higher/lower $f\left( x\right)$ indicates that $p$ is in a denser/sparser area. The ordering of the sequence conveys important information: each "valley"- a subsequence of points between two "walls"-corresponds to a cluster. Furthermore, the points of this valley will remain in a cluster under any $\epsilon$ greater than the maximum $f\left( x\right)$ value of those points.

图14绘制了使用最小点数（MinPts） = 10时得到的 ${SS}$ -相似度密度-5D（${SS}$ -simden-5D）和 ${SS}$ -变密度-5D（${SS}$ -varden-5D）的OPTICS图 ${}^{5}$。在OPTICS图中（Ankerst等人，1999年），数据点按照 $\mathrm{x}$ 轴的顺序排列成一个序列。该图显示了函数 $f\left( x\right)  : \left\lbrack  {1,n}\right\rbrack   \rightarrow  \mathbb{R}$ 下方的区域，其中 $f\left( x\right)$ 大致可以理解如下：如果 $p$ 是序列中的第 $x$ 个点，那么 $f\left( x\right)$ 是最小的 $\epsilon$ 值（与所选的最小点数一起），它能将 $p$ 归入某个簇——换句话说，对于 $\epsilon  < f\left( x\right)$，$p$ 仍为噪声点。较高/较低的 $f\left( x\right)$ 表示 $p$ 处于较密集/稀疏的区域。序列的排序传达了重要信息：每个“谷”——两个“壁”之间的点的子序列——对应一个簇。此外，在任何大于这些点的最大 $f\left( x\right)$ 值的 $\epsilon$ 下，这个谷中的点将仍属于一个簇。

---

<!-- Footnote -->

${}^{5}$ The OPTICS algorithm (Ankerst et al. 1999) requires a parameter called maxEps,which was set to 10,000 in our expeirments.

${}^{5}$ OPTICS算法（Ankerst等人，1999年）需要一个名为最大邻域距离（maxEps）的参数，在我们的实验中该参数设置为10,000。

<!-- Footnote -->

---

<!-- Media -->

Table 3. Cluster Quality Under Different $\left( {\text{MinPts,}\epsilon }\right)$ : SS Varying Density

表3. 不同 $\left( {\text{MinPts,}\epsilon }\right)$ 下的簇质量：SS可变密度

<table><tr><td rowspan="2">$\epsilon$</td><td colspan="3">MinPts = 10</td><td colspan="3">${MinPts} = {100}$</td></tr><tr><td>CV Index</td><td>#clusters</td><td>#noise pts</td><td>CV Index</td><td>#clusters</td><td>#noise pts</td></tr><tr><td>100</td><td>0.480</td><td>1,294</td><td>50,904</td><td>0.457</td><td>164</td><td>774,095</td></tr><tr><td>200</td><td>0.574</td><td>70</td><td>2,830</td><td>0.584</td><td>153</td><td>250,018</td></tr><tr><td>400</td><td>0.946</td><td>6</td><td>161</td><td>0.836</td><td>21</td><td>18,383</td></tr><tr><td>800</td><td>0.904</td><td>6</td><td>154</td><td>0.939</td><td>6</td><td>154</td></tr><tr><td>1,000</td><td>0.887</td><td>6</td><td>153</td><td>0.905</td><td>6</td><td>153</td></tr><tr><td colspan="7">(a) SS-varden-2D</td></tr><tr><td rowspan="2">$\epsilon$</td><td colspan="3">MinPts = 10</td><td colspan="3">${MinPts} = {100}$</td></tr><tr><td>CV Index</td><td>#clusters</td><td>#noise pts</td><td>CV Index</td><td>#clusters</td><td>#noise pts</td></tr><tr><td>100</td><td>0.321</td><td>1,031</td><td>577,830</td><td>0.055</td><td>114</td><td>1,358,330</td></tr><tr><td>200</td><td>0.698</td><td>1,989</td><td>317,759</td><td>0.403</td><td>100</td><td>600,273</td></tr><tr><td>400</td><td>0.864</td><td>573</td><td>23,860</td><td>0.751</td><td>91</td><td>383,122</td></tr><tr><td>800</td><td>0.917</td><td>11</td><td>195</td><td>0.908</td><td>91</td><td>5,0711</td></tr><tr><td>1,000</td><td>0.904</td><td>11</td><td>194</td><td>0.884</td><td>27</td><td>236</td></tr><tr><td colspan="7">(b) SS-varden-3D</td></tr><tr><td rowspan="2">$\epsilon$</td><td colspan="3">MinPts = 10</td><td colspan="3">MinPts = 100</td></tr><tr><td>CV Index</td><td>#clusters</td><td>#noise pts</td><td>CV Index</td><td>#clusters</td><td>#noise pts</td></tr><tr><td>400</td><td>0.244</td><td>5,880</td><td>267,914</td><td>0.523</td><td>10,160</td><td>568,393</td></tr><tr><td>800</td><td>0.755</td><td>286</td><td>200</td><td>0.858</td><td>4,540</td><td>432</td></tr><tr><td>1,000</td><td>0.952</td><td>12</td><td>200</td><td>0.903</td><td>1,667</td><td>357</td></tr><tr><td>2,000</td><td>0.980</td><td>8</td><td>200</td><td>0.980</td><td>8</td><td>200</td></tr><tr><td>3,000</td><td>0.980</td><td>8</td><td>200</td><td>0.980</td><td>8</td><td>200</td></tr><tr><td colspan="7">(c) SS-varden-5D</td></tr><tr><td rowspan="2">$\epsilon$</td><td colspan="3">MinPts = 10</td><td colspan="3">${MinPts} = {100}$</td></tr><tr><td>CV Index</td><td>#clusters</td><td>#noise pts</td><td>CV Index</td><td>#clusters</td><td>#noise pts</td></tr><tr><td>400</td><td>0.423</td><td>7,646</td><td>801,947</td><td>0.450</td><td>6,550</td><td>837,575</td></tr><tr><td>800</td><td>0.780</td><td>9,224</td><td>10,167</td><td>0.686</td><td>5,050</td><td>425,229</td></tr><tr><td>1,000</td><td>0.804</td><td>7,897</td><td>200</td><td>0.860</td><td>8,054</td><td>506</td></tr><tr><td>2,000</td><td>0.781</td><td>1,045</td><td>200</td><td>0.781</td><td>1,044</td><td>400</td></tr><tr><td>3,000</td><td>0.949</td><td>13</td><td>200</td><td>0.949</td><td>13</td><td>200</td></tr><tr><td>4,000</td><td>0.949</td><td>13</td><td>200 7 Th</td><td>0.949</td><td>13</td><td>200</td></tr></table>

<table><tbody><tr><td rowspan="2">$\epsilon$</td><td colspan="3">最小点数（MinPts） = 10</td><td colspan="3">${MinPts} = {100}$</td></tr><tr><td>凝聚度与分离度指数（CV Index）</td><td>聚类数量（#clusters）</td><td>噪声点数量（#noise pts）</td><td>凝聚度与分离度指数（CV Index）</td><td>聚类数量（#clusters）</td><td>噪声点数量（#noise pts）</td></tr><tr><td>100</td><td>0.480</td><td>1,294</td><td>50,904</td><td>0.457</td><td>164</td><td>774,095</td></tr><tr><td>200</td><td>0.574</td><td>70</td><td>2,830</td><td>0.584</td><td>153</td><td>250,018</td></tr><tr><td>400</td><td>0.946</td><td>6</td><td>161</td><td>0.836</td><td>21</td><td>18,383</td></tr><tr><td>800</td><td>0.904</td><td>6</td><td>154</td><td>0.939</td><td>6</td><td>154</td></tr><tr><td>1,000</td><td>0.887</td><td>6</td><td>153</td><td>0.905</td><td>6</td><td>153</td></tr><tr><td colspan="7">(a) 单样本值（SS-varden） - 二维</td></tr><tr><td rowspan="2">$\epsilon$</td><td colspan="3">最小点数（MinPts） = 10</td><td colspan="3">${MinPts} = {100}$</td></tr><tr><td>凝聚度与分离度指数（CV Index）</td><td>聚类数量（#clusters）</td><td>噪声点数量（#noise pts）</td><td>凝聚度与分离度指数（CV Index）</td><td>聚类数量（#clusters）</td><td>噪声点数量（#noise pts）</td></tr><tr><td>100</td><td>0.321</td><td>1,031</td><td>577,830</td><td>0.055</td><td>114</td><td>1,358,330</td></tr><tr><td>200</td><td>0.698</td><td>1,989</td><td>317,759</td><td>0.403</td><td>100</td><td>600,273</td></tr><tr><td>400</td><td>0.864</td><td>573</td><td>23,860</td><td>0.751</td><td>91</td><td>383,122</td></tr><tr><td>800</td><td>0.917</td><td>11</td><td>195</td><td>0.908</td><td>91</td><td>5,0711</td></tr><tr><td>1,000</td><td>0.904</td><td>11</td><td>194</td><td>0.884</td><td>27</td><td>236</td></tr><tr><td colspan="7">(b) 单样本值（SS-varden） - 三维</td></tr><tr><td rowspan="2">$\epsilon$</td><td colspan="3">最小点数（MinPts） = 10</td><td colspan="3">最小点数（MinPts） = 100</td></tr><tr><td>凝聚度与分离度指数（CV Index）</td><td>聚类数量（#clusters）</td><td>噪声点数量（#noise pts）</td><td>凝聚度与分离度指数（CV Index）</td><td>聚类数量（#clusters）</td><td>噪声点数量（#noise pts）</td></tr><tr><td>400</td><td>0.244</td><td>5,880</td><td>267,914</td><td>0.523</td><td>10,160</td><td>568,393</td></tr><tr><td>800</td><td>0.755</td><td>286</td><td>200</td><td>0.858</td><td>4,540</td><td>432</td></tr><tr><td>1,000</td><td>0.952</td><td>12</td><td>200</td><td>0.903</td><td>1,667</td><td>357</td></tr><tr><td>2,000</td><td>0.980</td><td>8</td><td>200</td><td>0.980</td><td>8</td><td>200</td></tr><tr><td>3,000</td><td>0.980</td><td>8</td><td>200</td><td>0.980</td><td>8</td><td>200</td></tr><tr><td colspan="7">(c) 单样本值（SS-varden） - 五维</td></tr><tr><td rowspan="2">$\epsilon$</td><td colspan="3">最小点数（MinPts） = 10</td><td colspan="3">${MinPts} = {100}$</td></tr><tr><td>凝聚度与分离度指数（CV Index）</td><td>聚类数量（#clusters）</td><td>噪声点数量（#noise pts）</td><td>凝聚度与分离度指数（CV Index）</td><td>聚类数量（#clusters）</td><td>噪声点数量（#noise pts）</td></tr><tr><td>400</td><td>0.423</td><td>7,646</td><td>801,947</td><td>0.450</td><td>6,550</td><td>837,575</td></tr><tr><td>800</td><td>0.780</td><td>9,224</td><td>10,167</td><td>0.686</td><td>5,050</td><td>425,229</td></tr><tr><td>1,000</td><td>0.804</td><td>7,897</td><td>200</td><td>0.860</td><td>8,054</td><td>506</td></tr><tr><td>2,000</td><td>0.781</td><td>1,045</td><td>200</td><td>0.781</td><td>1,044</td><td>400</td></tr><tr><td>3,000</td><td>0.949</td><td>13</td><td>200</td><td>0.949</td><td>13</td><td>200</td></tr><tr><td>4,000</td><td>0.949</td><td>13</td><td>200 7 日</td><td>0.949</td><td>13</td><td>200</td></tr></tbody></table>

(d) SS-varden-7D

(d) SS值 - 7D

<!-- Media -->

Figure 14(a) has 13 valleys,matching the 13 clusters found by $\epsilon  = {200}$ . Notice that the points in these valleys have roughly the same $f\left( x\right)$ values (i.e.,similar density). Figure 14(b),on the other hand,has eight valleys,namely,the eight clusters found by $\epsilon  = 2,{000}$ . Points in various valleys can have very different $f\left( x\right)$ values (i.e.,varying density). The OPTICS diagrams for the other synthetic datasets are omitted because they illustrate analogous observations about the composition of clusters.

图14(a)有13个谷，与$\epsilon  = {200}$所发现的13个簇相匹配。注意这些谷中的点大致具有相同的$f\left( x\right)$值（即，相似的密度）。另一方面，图14(b)有八个谷，即$\epsilon  = 2,{000}$所发现的八个簇。不同谷中的点可以有非常不同的$f\left( x\right)$值（即，不同的密度）。其他合成数据集的OPTICS图省略了，因为它们说明了关于簇组成的类似观察结果。

Next, we turned to the real datasets. Table 4 gives the statistics for PAMAP2, Farm, and Household. The CV indexes are much lower (than those of synthetic data), indicating that the clusters in these datasets are less obvious. For further analysis,we chose MinPts = 100 as the default (because it

接下来，我们转向真实数据集。表4给出了PAMAP2、农场和家庭数据集的统计信息。CV指数（比合成数据的）低得多，这表明这些数据集中的簇不太明显。为了进一步分析，我们选择MinPts = 100作为默认值（因为它

<!-- Media -->

<!-- figureText: 500 reachability distance 1m 1.5m 2m OPTICS ordering 1m 1.5m $2\mathrm{\;m}$ OPTICS ordering 300 100 0.5m (a) SS-simden-5D 3000 2500 1500 1000 500 0 0.5m (b) SS-varden-5D -->

<img src="https://cdn.noedgeai.com/0195c91b-4734-77b1-933f-3c7cb33c7bfe_29.jpg?x=148&y=260&w=1271&h=503&r=0"/>

Fig. 14. Optics diagrams for 5D synthetic data.

图14. 5D合成数据的OPTICS图。

<!-- figureText: 5000 reachability distance 2m 2.5m $3\mathrm{\;m}$ 3.5m OPTICS ordering ${2.5}\mathrm{\;m}$ 3.5m OPTICS ordering 1m 1.5m $2\mathrm{\;m}$ OPTICS ordering 4000 3000 2000 1000 0 0 0.5m 1m 1.5m (a) PAMAP2 5000 reachability distance 4000 3000 2000 1000 0.5m 1m 1.5m (b) Farm 10000 reachability distance 8000 6000 4000 2000 0 0.5m (c) Household -->

<img src="https://cdn.noedgeai.com/0195c91b-4734-77b1-933f-3c7cb33c7bfe_29.jpg?x=147&y=858&w=1272&h=760&r=0"/>

Fig. 15. Optics diagrams for real datasets.

图15. 真实数据集的OPTICS图。

<!-- Media -->

worked much better than MinPts $= {10}$ ),using which Figure 15 presents the OPTICS diagrams for the real datasets,while Table 5 details the sizes (unit: 1,000 ) of the 10 largest clusters under each $\epsilon$ value in Table 4. By combining all these data, we make the following observations:

比MinPts $= {10}$效果好得多），使用该值，图15展示了真实数据集的OPTICS图，而表5详细列出了表4中每个$\epsilon$值下10个最大簇的大小（单位：1000）。通过综合所有这些数据，我们有以下观察结果：

- PAMAP2: From Figure 15(a), we can see that this dataset contains numerous "tiny valleys," which explains the large number of clusters as shown in Table 4(a). An interesting $\epsilon$ value would be 500 , which discovers most of those valleys. Notice from Table 4(a) that the CV index is relatively high at $\epsilon  = {500}$ . It is worth mentioning that,although $\epsilon  = 4,{000}$ and 5,000 have even higher CV indexes,almost all the valleys disappear at these $\epsilon$ values,leaving only two major clusters,one of which contains over ${90}\%$ of the points.

- PAMAP2：从图15(a)中，我们可以看到这个数据集包含许多“小谷”，这解释了表4(a)中显示的大量簇。一个有趣的$\epsilon$值是500，它能发现大多数这些谷。从表4(a)中注意到，在$\epsilon  = {500}$处CV指数相对较高。值得一提的是，尽管$\epsilon  = 4,{000}$和5000的CV指数更高，但在这些$\epsilon$值下几乎所有的谷都消失了，只留下两个主要簇，其中一个簇包含超过${90}\%$的点。

<!-- Media -->

Table 4. Cluster Quality Under Different $\left( {\text{MinPts,}\epsilon }\right)$ : Real Data

表4. 不同$\left( {\text{MinPts,}\epsilon }\right)$下的簇质量：真实数据

<table><tr><td rowspan="2">$\epsilon$</td><td colspan="3">MinPts = 10</td><td colspan="3">${MinPts} = {100}$</td></tr><tr><td>CV Index</td><td>#clusters</td><td>#noise pts</td><td>CV Index</td><td>#clusters</td><td>#noise pts</td></tr><tr><td>100</td><td>0.174</td><td>6,585</td><td>2,578,125</td><td>0.103</td><td>478</td><td>3,369,657</td></tr><tr><td>200</td><td>0.222</td><td>17,622</td><td>1,890,108</td><td>0.210</td><td>818</td><td>2,800,524</td></tr><tr><td>400</td><td>0.092</td><td>11,408</td><td>620,932</td><td>0.226</td><td>1,129</td><td>2,396,808</td></tr><tr><td>500</td><td>0.059</td><td>5,907</td><td>406,247</td><td>0.233</td><td>1,238</td><td>2,097,941</td></tr><tr><td>800</td><td>0.037</td><td>3,121</td><td>215,925</td><td>0.099</td><td>756</td><td>949,167</td></tr><tr><td>1,000</td><td>0.032</td><td>2,530</td><td>159,570</td><td>0.078</td><td>483</td><td>594,075</td></tr><tr><td>2,000</td><td>0.033</td><td>549</td><td>28,901</td><td>0.126</td><td>237</td><td>209,236</td></tr><tr><td>3,000</td><td>0.237</td><td>110</td><td>5,840</td><td>0.302</td><td>100</td><td>75,723</td></tr><tr><td>4,000</td><td>0.106</td><td>30</td><td>1,673</td><td>0.492</td><td>31</td><td>24,595</td></tr><tr><td>5,000</td><td>0.490</td><td>9</td><td>673</td><td>0.506</td><td>12</td><td>9,060</td></tr></table>

<table><tbody><tr><td rowspan="2">$\epsilon$</td><td colspan="3">最小点数（MinPts） = 10</td><td colspan="3">${MinPts} = {100}$</td></tr><tr><td>凝聚度与分离度指数（CV Index）</td><td>簇的数量（#clusters）</td><td>噪声点的数量（#noise pts）</td><td>凝聚度与分离度指数（CV Index）</td><td>簇的数量（#clusters）</td><td>噪声点的数量（#noise pts）</td></tr><tr><td>100</td><td>0.174</td><td>6,585</td><td>2,578,125</td><td>0.103</td><td>478</td><td>3,369,657</td></tr><tr><td>200</td><td>0.222</td><td>17,622</td><td>1,890,108</td><td>0.210</td><td>818</td><td>2,800,524</td></tr><tr><td>400</td><td>0.092</td><td>11,408</td><td>620,932</td><td>0.226</td><td>1,129</td><td>2,396,808</td></tr><tr><td>500</td><td>0.059</td><td>5,907</td><td>406,247</td><td>0.233</td><td>1,238</td><td>2,097,941</td></tr><tr><td>800</td><td>0.037</td><td>3,121</td><td>215,925</td><td>0.099</td><td>756</td><td>949,167</td></tr><tr><td>1,000</td><td>0.032</td><td>2,530</td><td>159,570</td><td>0.078</td><td>483</td><td>594,075</td></tr><tr><td>2,000</td><td>0.033</td><td>549</td><td>28,901</td><td>0.126</td><td>237</td><td>209,236</td></tr><tr><td>3,000</td><td>0.237</td><td>110</td><td>5,840</td><td>0.302</td><td>100</td><td>75,723</td></tr><tr><td>4,000</td><td>0.106</td><td>30</td><td>1,673</td><td>0.492</td><td>31</td><td>24,595</td></tr><tr><td>5,000</td><td>0.490</td><td>9</td><td>673</td><td>0.506</td><td>12</td><td>9,060</td></tr></tbody></table>

(a) ${PAMAP2}$

<table><tr><td rowspan="2">$\epsilon$</td><td colspan="3">MinPts = 10</td><td colspan="3">MinPts = 100</td></tr><tr><td>CV Index</td><td>#clusters</td><td>#noise pts</td><td>CV Index</td><td>#clusters</td><td>#noise pts</td></tr><tr><td>100</td><td>0.002</td><td>925</td><td>3,542,419</td><td>0.001</td><td>3</td><td>3,621,494</td></tr><tr><td>200</td><td>0.005</td><td>3,296</td><td>2,473,933</td><td>0.008</td><td>21</td><td>3,404,402</td></tr><tr><td>400</td><td>0.006</td><td>1,420</td><td>1,153,340</td><td>0.191</td><td>13</td><td>1,840,989</td></tr><tr><td>700</td><td>0.004</td><td>962</td><td>514,949</td><td>0.364</td><td>28</td><td>1,039,114</td></tr><tr><td>800</td><td>0.004</td><td>994</td><td>410,432</td><td>0.198</td><td>18</td><td>859,002</td></tr><tr><td>1,000</td><td>0.005</td><td>689</td><td>273,723</td><td>0.295</td><td>15</td><td>594,462</td></tr><tr><td>2,000</td><td>0.002</td><td>217</td><td>46,616</td><td>0.120</td><td>13</td><td>181,628</td></tr><tr><td>3,000</td><td>0.001</td><td>55</td><td>15,096</td><td>0.131</td><td>6</td><td>62,746</td></tr><tr><td>4,000</td><td>0.058</td><td>35</td><td>8,100</td><td>0.764</td><td>3</td><td>24,791</td></tr><tr><td>5,000</td><td>0.024</td><td>27</td><td>5,298</td><td>0.157</td><td>6</td><td>12,890</td></tr></table>

<table><tbody><tr><td rowspan="2">$\epsilon$</td><td colspan="3">最小点数 = 10</td><td colspan="3">最小点数 = 100</td></tr><tr><td>凝聚度-分离度指数（CV Index）</td><td>簇的数量</td><td>噪声点的数量</td><td>凝聚度-分离度指数（CV Index）</td><td>簇的数量</td><td>噪声点的数量</td></tr><tr><td>100</td><td>0.002</td><td>925</td><td>3,542,419</td><td>0.001</td><td>3</td><td>3,621,494</td></tr><tr><td>200</td><td>0.005</td><td>3,296</td><td>2,473,933</td><td>0.008</td><td>21</td><td>3,404,402</td></tr><tr><td>400</td><td>0.006</td><td>1,420</td><td>1,153,340</td><td>0.191</td><td>13</td><td>1,840,989</td></tr><tr><td>700</td><td>0.004</td><td>962</td><td>514,949</td><td>0.364</td><td>28</td><td>1,039,114</td></tr><tr><td>800</td><td>0.004</td><td>994</td><td>410,432</td><td>0.198</td><td>18</td><td>859,002</td></tr><tr><td>1,000</td><td>0.005</td><td>689</td><td>273,723</td><td>0.295</td><td>15</td><td>594,462</td></tr><tr><td>2,000</td><td>0.002</td><td>217</td><td>46,616</td><td>0.120</td><td>13</td><td>181,628</td></tr><tr><td>3,000</td><td>0.001</td><td>55</td><td>15,096</td><td>0.131</td><td>6</td><td>62,746</td></tr><tr><td>4,000</td><td>0.058</td><td>35</td><td>8,100</td><td>0.764</td><td>3</td><td>24,791</td></tr><tr><td>5,000</td><td>0.024</td><td>27</td><td>5,298</td><td>0.157</td><td>6</td><td>12,890</td></tr></tbody></table>

(b) Farm

(b) 农场

MinPts $= {10}$ MinPts = 100

最小点数 $= {10}$ 最小点数 = 100

$\epsilon$ CV Index #clusters #noise pts CV Index #clusters #noise pts

$\epsilon$ 轮廓系数（CV Index） 簇数量 噪声点数量 轮廓系数（CV Index） 簇数量 噪声点数量

100 0.057 3,342 1,702,377 0.026 54 1,944,226

200 0.114 5,036 1,314,498 0.074 87 1,829,873

400 0.085 4,802 911,088 0.088 165 1,598,323

800 0.048 2,148 490,634 0.257 47 974,566

1,000 0.045 1,800 404,306 0.227 55 829,398

2,000 0.129 601 139,483 0.416 28 327,508

3,000 0.074 447 73,757 0.241 48 193,502

4,000 0.007 195 34,585 0.565 10 112,231

5,000 0.015 131 18,059 0.649 8 68,943

(c) Household

(c) 家庭

<!-- Media -->

-Farm: There are two clusters in the dataset. The first one is the valley between ${2.6}\mathrm{\;m}$ and ${2.8}\mathrm{\;m}$ on the $\mathrm{x}$ -axis of Figure 15(b),and the second one is the small dagger-shape valley at ${3.5}\mathrm{\;m}$ . The best value of $\epsilon$ that discovers both clusters lies around 700-they are the second and fourth largest clusters at the row of $\epsilon  = {700}$ in Table 5(b). Once again,there exist some large values $\epsilon$ such as 4,000 that give high CV indexes,but assign almost all the points (over ${99}\%$ for $\epsilon  = 4,{000})$ into one cluster.

-农场：数据集中有两个簇。第一个是图15(b)中 $\mathrm{x}$ 轴上 ${2.6}\mathrm{\;m}$ 和 ${2.8}\mathrm{\;m}$ 之间的谷值，第二个是 ${3.5}\mathrm{\;m}$ 处的小匕首形状的谷值。能发现这两个簇的 $\epsilon$ 的最佳值约为700——它们是表5(b)中 $\epsilon  = {700}$ 行的第二大和第四大的簇。再次说明，存在一些较大的 $\epsilon$ 值，如4000，这些值会给出较高的轮廓系数（CV Index），但会将几乎所有点（对于 $\epsilon  = 4,{000})$ 而言超过 ${99}\%$）分配到一个簇中。

<!-- Media -->

Table 5. Sizes of the 10 Largest Clusters: Real Data (Unit: ${10}^{3}$ )

表5. 10个最大簇的大小：真实数据（单位：${10}^{3}$ ）

<table><tr><td>$\epsilon$</td><td>1st</td><td>2nd</td><td>3rd</td><td>4th</td><td>5th</td><td>6th</td><td>7th</td><td>8th</td><td>9th</td><td>10th</td></tr><tr><td>100</td><td>18.8</td><td>16.9</td><td>11.9</td><td>10.2</td><td>9.72</td><td>8.10</td><td>7.00</td><td>5.57</td><td>5.54</td><td>5.54</td></tr><tr><td>200</td><td>25.9</td><td>21.6</td><td>20.0</td><td>18.9</td><td>18.8</td><td>18.2</td><td>18.2</td><td>17.4</td><td>16.9</td><td>15.5</td></tr><tr><td>400</td><td>66.9</td><td>54.9</td><td>39.1</td><td>35.2</td><td>29.1</td><td>28.1</td><td>23.8</td><td>21.7</td><td>20.0</td><td>19.4</td></tr><tr><td>500</td><td>124</td><td>114</td><td>55.5</td><td>53.4</td><td>47.0</td><td>42.9</td><td>41.2</td><td>29.3</td><td>29.2</td><td>20.0</td></tr><tr><td>800</td><td>2,219</td><td>41.5</td><td>37.3</td><td>26.7</td><td>20.5</td><td>19.2</td><td>19.0</td><td>17.4</td><td>15.3</td><td>13.9</td></tr><tr><td>1,000</td><td>2,794</td><td>116</td><td>20.5</td><td>16.4</td><td>13.1</td><td>9.12</td><td>9.08</td><td>8.69</td><td>7.65</td><td>7.10</td></tr><tr><td>2,000</td><td>3,409</td><td>78.0</td><td>18.2</td><td>13.3</td><td>9.65</td><td>9.57</td><td>6.60</td><td>6.60</td><td>5.11</td><td>5.04</td></tr><tr><td>3,000</td><td>3,470</td><td>239</td><td>18.5</td><td>11.8</td><td>2.03</td><td>1.90</td><td>1.83</td><td>1.21</td><td>1.20</td><td>1.09</td></tr><tr><td>4,000</td><td>3,495</td><td>315</td><td>2.03</td><td>1.86</td><td>1.84</td><td>0.965</td><td>0.786</td><td>0.735</td><td>0.698</td><td>0.687</td></tr><tr><td>5,000</td><td>3,497</td><td>339</td><td>1.85</td><td>0.977</td><td>0.553</td><td>0.551</td><td>0.328</td><td>0.328</td><td>0.217</td><td>0.216</td></tr><tr><td colspan="11">(a) PAMAP2</td></tr><tr><td>$\epsilon$</td><td>1st</td><td>2nd</td><td>3rd</td><td>4th</td><td>5th</td><td>6th</td><td>7th</td><td>8th</td><td>9th</td><td>10th</td></tr><tr><td>100</td><td>4.10</td><td>1.34</td><td>0.150</td><td/><td/><td/><td/><td/><td/><td/></tr><tr><td>200</td><td>195</td><td>18.0</td><td>3.93</td><td>0.868</td><td>0.717</td><td>0.647</td><td>0.529</td><td>0.393</td><td>0.391</td><td>0.303</td></tr><tr><td>400</td><td>1,604</td><td>129</td><td>37.3</td><td>11.1</td><td>1.75</td><td>0.713</td><td>0.408</td><td>0.327</td><td>0.265</td><td>0.226</td></tr><tr><td>700</td><td>2,282</td><td>218</td><td>44.1</td><td>17.0</td><td>10.4</td><td>4.27</td><td>3.59</td><td>1.12</td><td>1.09</td><td>0.863</td></tr><tr><td>800</td><td>2,358</td><td>381</td><td>17.4</td><td>6.28</td><td>1.34</td><td>0.921</td><td>0.859</td><td>0.545</td><td>0.528</td><td>0.414</td></tr><tr><td>1,000</td><td>3,009</td><td>18.0</td><td>1.47</td><td>0.740</td><td>0.718</td><td>0.446</td><td>0.422</td><td>0.332</td><td>0.287</td><td>0.214</td></tr><tr><td>2,000</td><td>3,418</td><td>18.8</td><td>2.79</td><td>1.88</td><td>1.45</td><td>0.386</td><td>0.374</td><td>0.230</td><td>0.186</td><td>0.165</td></tr><tr><td>3,000</td><td>3,562</td><td>0.951</td><td>0.681</td><td>0.350</td><td>0.190</td><td>0.177</td><td/><td/><td/><td/></tr><tr><td>4,000</td><td>3,600</td><td>1.08</td><td>0.470</td><td/><td/><td/><td/><td/><td/><td/></tr><tr><td>5,000</td><td>3,611</td><td>1.18</td><td>0.537</td><td>0.273</td><td>0.130</td><td>0.114</td><td/><td/><td/><td/></tr><tr><td colspan="11">(b) Farm</td></tr><tr><td>$\epsilon$</td><td>1st</td><td>2nd</td><td>3rd</td><td>4th</td><td>5th</td><td>6th</td><td>7th</td><td>8th</td><td>9th</td><td>10th</td></tr><tr><td>100</td><td>37.8</td><td>19.8</td><td>16.2</td><td>12.3</td><td>3.84</td><td>3.31</td><td>2.33</td><td>0.529</td><td>0.525</td><td>0.505</td></tr><tr><td>200</td><td>47.8</td><td>30.9</td><td>24.8</td><td>24.5</td><td>20.3</td><td>15.2</td><td>7.79</td><td>6.91</td><td>5.82</td><td>4.39</td></tr><tr><td>400</td><td>52.6</td><td>39.9</td><td>34.2</td><td>31.9</td><td>27.3</td><td>25.9</td><td>21.1</td><td>18.8</td><td>15.9</td><td>15.8</td></tr><tr><td>800</td><td>274</td><td>193</td><td>117</td><td>97.3</td><td>70.2</td><td>48.0</td><td>33.9</td><td>33.6</td><td>27.0</td><td>24.8</td></tr><tr><td>1,000</td><td>294</td><td>198</td><td>158</td><td>99.7</td><td>94.5</td><td>81.7</td><td>51.6</td><td>30.8</td><td>27.2</td><td>25.0</td></tr><tr><td>2,000</td><td>560</td><td>320</td><td>222</td><td>220</td><td>110</td><td>75.2</td><td>71.4</td><td>68.2</td><td>25.1</td><td>25.0</td></tr><tr><td>3,000</td><td>586</td><td>337</td><td>243</td><td>221</td><td>111</td><td>91.8</td><td>85.0</td><td>69.7</td><td>26.6</td><td>26.1</td></tr><tr><td>4,000</td><td>1,312</td><td>575</td><td>17.0</td><td>10.9</td><td>9.98</td><td>7.07</td><td>3.71</td><td>0.381</td><td>0.197</td><td>0.100</td></tr><tr><td>5,000</td><td>1,918</td><td>22.2</td><td>14.9</td><td>13.3</td><td>11.2 ...</td><td>0.299 7.7</td><td>0.101</td><td>0.100</td><td/><td/></tr></table>

<table><tbody><tr><td>$\epsilon$</td><td>第一</td><td>第二</td><td>第三</td><td>第四</td><td>第五</td><td>第六</td><td>第七</td><td>第八</td><td>第九</td><td>第十</td></tr><tr><td>100</td><td>18.8</td><td>16.9</td><td>11.9</td><td>10.2</td><td>9.72</td><td>8.10</td><td>7.00</td><td>5.57</td><td>5.54</td><td>5.54</td></tr><tr><td>200</td><td>25.9</td><td>21.6</td><td>20.0</td><td>18.9</td><td>18.8</td><td>18.2</td><td>18.2</td><td>17.4</td><td>16.9</td><td>15.5</td></tr><tr><td>400</td><td>66.9</td><td>54.9</td><td>39.1</td><td>35.2</td><td>29.1</td><td>28.1</td><td>23.8</td><td>21.7</td><td>20.0</td><td>19.4</td></tr><tr><td>500</td><td>124</td><td>114</td><td>55.5</td><td>53.4</td><td>47.0</td><td>42.9</td><td>41.2</td><td>29.3</td><td>29.2</td><td>20.0</td></tr><tr><td>800</td><td>2,219</td><td>41.5</td><td>37.3</td><td>26.7</td><td>20.5</td><td>19.2</td><td>19.0</td><td>17.4</td><td>15.3</td><td>13.9</td></tr><tr><td>1,000</td><td>2,794</td><td>116</td><td>20.5</td><td>16.4</td><td>13.1</td><td>9.12</td><td>9.08</td><td>8.69</td><td>7.65</td><td>7.10</td></tr><tr><td>2,000</td><td>3,409</td><td>78.0</td><td>18.2</td><td>13.3</td><td>9.65</td><td>9.57</td><td>6.60</td><td>6.60</td><td>5.11</td><td>5.04</td></tr><tr><td>3,000</td><td>3,470</td><td>239</td><td>18.5</td><td>11.8</td><td>2.03</td><td>1.90</td><td>1.83</td><td>1.21</td><td>1.20</td><td>1.09</td></tr><tr><td>4,000</td><td>3,495</td><td>315</td><td>2.03</td><td>1.86</td><td>1.84</td><td>0.965</td><td>0.786</td><td>0.735</td><td>0.698</td><td>0.687</td></tr><tr><td>5,000</td><td>3,497</td><td>339</td><td>1.85</td><td>0.977</td><td>0.553</td><td>0.551</td><td>0.328</td><td>0.328</td><td>0.217</td><td>0.216</td></tr><tr><td colspan="11">(a) PAMAP2数据集</td></tr><tr><td>$\epsilon$</td><td>第一</td><td>第二</td><td>第三</td><td>第四</td><td>第五</td><td>第六</td><td>第七</td><td>第八</td><td>第九</td><td>第十</td></tr><tr><td>100</td><td>4.10</td><td>1.34</td><td>0.150</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>200</td><td>195</td><td>18.0</td><td>3.93</td><td>0.868</td><td>0.717</td><td>0.647</td><td>0.529</td><td>0.393</td><td>0.391</td><td>0.303</td></tr><tr><td>400</td><td>1,604</td><td>129</td><td>37.3</td><td>11.1</td><td>1.75</td><td>0.713</td><td>0.408</td><td>0.327</td><td>0.265</td><td>0.226</td></tr><tr><td>700</td><td>2,282</td><td>218</td><td>44.1</td><td>17.0</td><td>10.4</td><td>4.27</td><td>3.59</td><td>1.12</td><td>1.09</td><td>0.863</td></tr><tr><td>800</td><td>2,358</td><td>381</td><td>17.4</td><td>6.28</td><td>1.34</td><td>0.921</td><td>0.859</td><td>0.545</td><td>0.528</td><td>0.414</td></tr><tr><td>1,000</td><td>3,009</td><td>18.0</td><td>1.47</td><td>0.740</td><td>0.718</td><td>0.446</td><td>0.422</td><td>0.332</td><td>0.287</td><td>0.214</td></tr><tr><td>2,000</td><td>3,418</td><td>18.8</td><td>2.79</td><td>1.88</td><td>1.45</td><td>0.386</td><td>0.374</td><td>0.230</td><td>0.186</td><td>0.165</td></tr><tr><td>3,000</td><td>3,562</td><td>0.951</td><td>0.681</td><td>0.350</td><td>0.190</td><td>0.177</td><td></td><td></td><td></td><td></td></tr><tr><td>4,000</td><td>3,600</td><td>1.08</td><td>0.470</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>5,000</td><td>3,611</td><td>1.18</td><td>0.537</td><td>0.273</td><td>0.130</td><td>0.114</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="11">(b) 农场数据集</td></tr><tr><td>$\epsilon$</td><td>第一</td><td>第二</td><td>第三</td><td>第四</td><td>第五</td><td>第六</td><td>第七</td><td>第八</td><td>第九</td><td>第十</td></tr><tr><td>100</td><td>37.8</td><td>19.8</td><td>16.2</td><td>12.3</td><td>3.84</td><td>3.31</td><td>2.33</td><td>0.529</td><td>0.525</td><td>0.505</td></tr><tr><td>200</td><td>47.8</td><td>30.9</td><td>24.8</td><td>24.5</td><td>20.3</td><td>15.2</td><td>7.79</td><td>6.91</td><td>5.82</td><td>4.39</td></tr><tr><td>400</td><td>52.6</td><td>39.9</td><td>34.2</td><td>31.9</td><td>27.3</td><td>25.9</td><td>21.1</td><td>18.8</td><td>15.9</td><td>15.8</td></tr><tr><td>800</td><td>274</td><td>193</td><td>117</td><td>97.3</td><td>70.2</td><td>48.0</td><td>33.9</td><td>33.6</td><td>27.0</td><td>24.8</td></tr><tr><td>1,000</td><td>294</td><td>198</td><td>158</td><td>99.7</td><td>94.5</td><td>81.7</td><td>51.6</td><td>30.8</td><td>27.2</td><td>25.0</td></tr><tr><td>2,000</td><td>560</td><td>320</td><td>222</td><td>220</td><td>110</td><td>75.2</td><td>71.4</td><td>68.2</td><td>25.1</td><td>25.0</td></tr><tr><td>3,000</td><td>586</td><td>337</td><td>243</td><td>221</td><td>111</td><td>91.8</td><td>85.0</td><td>69.7</td><td>26.6</td><td>26.1</td></tr><tr><td>4,000</td><td>1,312</td><td>575</td><td>17.0</td><td>10.9</td><td>9.98</td><td>7.07</td><td>3.71</td><td>0.381</td><td>0.197</td><td>0.100</td></tr><tr><td>5,000</td><td>1,918</td><td>22.2</td><td>14.9</td><td>13.3</td><td>11.2 ...</td><td>0.299 7.7</td><td>0.101</td><td>0.100</td><td></td><td></td></tr></tbody></table>

(c) Household

(c) 家庭（Household）

<!-- Media -->

- Household: This is the "most clustered" real dataset of the three. It is evident that $\epsilon  = 2,{000}$ is an interesting value: it has a relatively high CV index (see Table 4(c)), and discovers most of the important valleys in Figure 15, whose clusters are quite sizable as shown in Table 5(c).

- 家庭（Household）：这是三个真实数据集中“聚类程度最高”的数据集。显然，$\epsilon  = 2,{000}$ 是一个有趣的值：它具有相对较高的变异系数（CV）指数（见表 4(c)），并且能发现图 15 中大部分重要的低谷，其聚类规模如表 5(c) 所示相当可观。

Based on the above discussion,we set the default $\epsilon$ of each real dataset to the bold values in Table 5.

基于上述讨论，我们将每个真实数据集的默认 $\epsilon$ 设置为表 5 中的粗体值。

### 7.3 Approximation Quality

### 7.3 近似质量

In this subsection,we evaluate the quality of the clusters returned by the proposed $\rho$ -approximate DBSCAN algorithm.

在本小节中，我们评估所提出的 $\rho$ -近似 DBSCAN 算法返回的聚类质量。

<!-- Media -->

<!-- figureText: (a) Exact ( $\epsilon  = {5000}$ ) (b) $\rho  = {0.001},\epsilon  = {5000}$ (c) $\rho  = {0.01},\epsilon  = {5000}$ (d) $\rho  = {0.1},\epsilon  = {5000}$ (h) $\rho  = {0.1},\epsilon  = {11300}$ ract $\left( {\epsilon  = {12200}}\right)$ (j) $\rho  = {0.001},\epsilon  = {12200}$ (k) $\rho  = {0.01},\epsilon  = {12200}$ (l) $\rho  = {0.1},\epsilon  = {122}$ (e) Exact ( $\epsilon  = {11300}$ -->

<img src="https://cdn.noedgeai.com/0195c91b-4734-77b1-933f-3c7cb33c7bfe_32.jpg?x=147&y=261&w=1274&h=958&r=0"/>

Fig. 16. Comparison of the clusters found by exact DBSCAN and $\rho$ -approximate DBSCAN.

图 16. 精确 DBSCAN 和 $\rho$ -近似 DBSCAN 找到的聚类比较。

<!-- Media -->

2D Visualization. To show directly the effects of approximation, we take the 2D dataset in Figure 13 as the input (note that the cardinality was deliberately chosen to be small to facilitate visualization),and fixed MinPts $= {20}$ . Figure 16(a) demonstrates the four clusters found by exact DBSCAN with $\epsilon  = 5,{000}$ (which is the radius of the circle shown). The points of each cluster are depicted with the same color and marker. Figures 16(b), (c), and (d) present the clusters found by our $\rho$ -approximate DBSCAN when $\rho$ equals 0.001,0.01,and 0.1,respectively. In all cases, $\rho$ - approximate DBSCAN returned exactly the same clusters as DBSCAN.

二维可视化。为了直接展示近似的效果，我们以图 13 中的二维数据集作为输入（注意，特意选择较小的基数以方便可视化），并固定 MinPts $= {20}$。图 16(a) 展示了使用 $\epsilon  = 5,{000}$（即所示圆的半径）的精确 DBSCAN 找到的四个聚类。每个聚类的点用相同的颜色和标记表示。图 16(b)、(c) 和 (d) 分别展示了当 $\rho$ 等于 0.001、0.01 和 0.1 时，我们的 $\rho$ -近似 DBSCAN 找到的聚类。在所有情况下，$\rho$ -近似 DBSCAN 返回的聚类与 DBSCAN 完全相同。

Making things more interesting,in Figure 16(e),we increased $\epsilon$ to 11,300 (again, $\epsilon$ is the radius of the circle shown). This time, DBSCAN found three clusters (note that two clusters in Figure 16(a) have merged). Figures 16(f),(g),and (h) give the clusters of $\rho$ -approximate DBSCAN for $\rho  = {0.001}$ , 0.01,and 0.1,respectively. Once again,the clusters of $\rho  = {0.001}$ and 0.01 are exactly the same as DBSCAN. However, 0.1-approximate DBSCAN returned only two clusters. This can be understood by observing that the circle in Figure 16(e) almost touched a point from a different cluster. In fact, it will,once $\epsilon$ increases by ${10}\%$ ,which explains why 0.1-approximate DBSCAN produced different results.

更有趣的是，在图 16(e) 中，我们将 $\epsilon$ 增加到 11300（同样，$\epsilon$ 是所示圆的半径）。这次，DBSCAN 找到了三个聚类（注意，图 16(a) 中的两个聚类已经合并）。图 16(f)、(g) 和 (h) 分别给出了 $\rho$ -近似 DBSCAN 在 $\rho  = {0.001}$ 为 0.001、0.01 和 0.1 时的聚类。再次，$\rho  = {0.001}$ 为 0.001 和 0.01 时的聚类与 DBSCAN 完全相同。然而，0.1 -近似 DBSCAN 只返回了两个聚类。通过观察图 16(e) 中的圆几乎接触到来自不同聚类的一个点，可以理解这一点。实际上，一旦 $\epsilon$ 增加 ${10}\%$，就会出现这种情况，这解释了为什么 0.1 -近似 DBSCAN 产生了不同的结果。

Then we pushed $\epsilon$ even further to 12,200 so that DBSCAN yielded two clusters as shown in Figure 16(i). Figures 16(j),(k),and (l) illustrate the clusters of $\rho$ -approximate DBSCAN for $\rho  =$ 0.001,0.01,and 0.1,respectively. Here,both $\rho  = {0.01}$ and 0.1 had given up,but $\rho  = {0.001}$ still churned out exactly the same clusters as DBSCAN.

然后，我们进一步将 $\epsilon$ 增加到 12200，使得 DBSCAN 产生了两个聚类，如图 16(i) 所示。图 16(j)、(k) 和 (l) 分别展示了 $\rho$ -近似 DBSCAN 在 $\rho  =$ 为 0.001、0.01 和 0.1 时的聚类。在这里，$\rho  = {0.01}$ 和 0.1 都失败了，但 $\rho  = {0.001}$ 仍然产生了与 DBSCAN 完全相同的聚类。

Surprised by $\rho  = {0.01}$ not working,we examined the reason behind its failure. It turned out that 12,200 was extremely close to the "boundary $\epsilon$ " for DBSCAN to output two clusters. Specifically, as soon as $\epsilon$ grew up to 12,203,the exact DBSCAN would return only a single cluster. Actually, this can be seen from Figure 16(i)-note how close the circle is to the point from the right cluster! In other words,12,200 is in fact an "unstable" value for $\epsilon$ .

由于 $\rho  = {0.01}$ 不起作用而感到惊讶，我们研究了其失败背后的原因。结果发现，12200 非常接近 DBSCAN 输出两个聚类的“边界 $\epsilon$”。具体来说，一旦 $\epsilon$ 增长到 12203，精确 DBSCAN 将只返回一个聚类。实际上，从图 16(i) 可以看出这一点——注意圆与右侧聚类的点有多接近！换句话说，12200 实际上是 $\epsilon$ 的一个“不稳定”值。

<!-- Media -->

<!-- figureText: error free $\rho$ error free $\rho$ error free $\rho$ 0.1 0.01 0.001 $\varepsilon \left( {10}^{3}\right)$ 4 5 0.1 $\varepsilon \left( {10}^{3}\right)$ 5 (b) SS-simden-5D (c) SS-simden-7D error free $\rho$ 0.1 0.01 0.001 8 $\left( {10}^{3}\right)$ 4 5 0.1 1 ${}_{\epsilon \left( {10}^{3}\right) }^{2}$ 4 5 (e) SS-varden-5D (f) SS-varden-7D error free $\rho$ 0.1 0.01 0.001 4 5 0.1 1 4 5 $\varepsilon \left( {10}^{3}\right)$ $\varepsilon \left( {10}^{3}\right)$ (h) Farm (i) Household 0.1 0.1 0.01 0.01 0.001 0.001 0 0 0.1 4 5 0.1 $\varepsilon \left( {10}^{3}\right)$ (a) SS-simden-3D error free $\rho$ error free $\rho$ 1 0.1 0.1 0.01 0.01 0.001 0.001 0 0 0.1 $\varepsilon \left( {10}^{3}\right)$ 4 5 0.1 1 (d) SS-varden-3D error free $\rho$ error free $\rho$ 1 0.1 0.1 0.01 0.01 0.001 0.001 0 0.1 4 5 0.1 1 $\varepsilon \left( {10}^{3}\right)$ (g) ${PAMAP2}$ -->

<img src="https://cdn.noedgeai.com/0195c91b-4734-77b1-933f-3c7cb33c7bfe_33.jpg?x=135&y=256&w=1290&h=1250&r=0"/>

Fig. 17. Largest $\rho$ in $\{ {0.001},{0.01},{0.1},1\}$ for our $\rho$ -approximate DBSCAN algorithm to return the same results as precise DBSCAN.

图 17. 对于我们的 $\rho$ -近似 DBSCAN 算法，在 $\{ {0.001},{0.01},{0.1},1\}$ 中使结果与精确 DBSCAN 相同的最大 $\rho$。

<!-- Media -->

Dimensionalities $d \geq  3$ . We deployed the same methodology to study the approximation quality in higher dimensional space. Specifically,for a dataset and a value of $\epsilon$ ,we varied $\rho$ among 0.001,0.01,0.1,and 1 to identify the highest error-free $\rho$ under which our $\rho$ -approximate algorithm returned exactly the same result as precise DBSCAN. Figure 17 plots the highest error-free $\rho$ for various datasets when $\epsilon$ grew from 100 to 5,000 . For example,by the fact that in Figure 17(a) the (highest) error-free $\rho$ is 1 at $\epsilon  = {100}$ ,one should understand that our approximate algorithm also returned the exact clusters at $\rho  = {0.001},{0.01}$ ,and 0.1 at this $\epsilon$ . Notice that in nearly all the cases, 0.01-approximation already guaranteed the precise results.

维度 $d \geq  3$。我们采用相同的方法来研究高维空间中的近似质量。具体而言，对于一个数据集和一个 $\epsilon$ 值，我们在 0.001、0.01、0.1 和 1 之间改变 $\rho$，以确定在该 $\rho$ 下我们的 $\rho$ -近似算法能返回与精确 DBSCAN 完全相同结果的最高无误差 $\rho$。图 17 绘制了当 $\epsilon$ 从 100 增长到 5000 时，各种数据集的最高无误差 $\rho$。例如，从图 17(a) 中可知，在 $\epsilon  = {100}$ 处（最高）无误差 $\rho$ 为 1，这意味着我们的近似算法在 $\rho  = {0.001},{0.01}$ 处也返回了精确的聚类，并且在该 $\epsilon$ 下为 0.1。请注意，在几乎所有情况下，0.01 - 近似已经能保证得到精确结果。

As shown in the next subsection, our current implementation was fast enough on all the tested datasets even when $\rho$ was set to 0.001 . We therefore recommend this value for practical use, which was also the default $\rho$ in the following experiments. Recall that,by the sandwich theorem (Theorem 4.3), the result of 0.001-approximate DBSCAN must fall between the results of DBSCAN with $\epsilon$ and ${1.001\epsilon }$ ,respectively. Hence,if 0.001-approximate DBSCAN differs from DBSCAN in the outcome, it means that the (exact) DBSCAN clusters must have changed within the parameter range $\left\lbrack  {\epsilon ,{1.001\epsilon }}\right\rbrack$ .

如下一小节所示，即使将 $\rho$ 设置为 0.001，我们当前的实现对于所有测试数据集来说也足够快。因此，我们建议在实际应用中使用该值，这也是后续实验中的默认 $\rho$。回想一下，根据夹逼定理（定理 4.3），0.001 - 近似 DBSCAN 的结果必定介于分别使用 $\epsilon$ 和 ${1.001\epsilon }$ 的 DBSCAN 结果之间。因此，如果 0.001 - 近似 DBSCAN 的结果与 DBSCAN 不同，这意味着（精确）DBSCAN 聚类在参数范围 $\left\lbrack  {\epsilon ,{1.001\epsilon }}\right\rbrack$ 内必定发生了变化。

### 7.4 Computational Efficiency for $d \geq  3$

### 7.4 $d \geq  3$ 的计算效率

We now proceed to inspect the running time of DBSCAN clustering in dimensionality $d \geq  3$ using four algorithms:

我们现在开始使用四种算法来检查维度 $d \geq  3$ 下 DBSCAN 聚类的运行时间：

- KDD96 (Ester et al. 1996): the original DBSCAN algorithm in Ester et al. (1996), which deployed a memory R-tree whose leaf capacity was 12 and internal fanout was 4 (the same values were used in the R-trees deployed by the other methods as well).

- KDD96（Ester 等人，1996 年）：Ester 等人（1996 年）提出的原始 DBSCAN 算法，该算法使用了一个内存 R - 树，其叶子节点容量为 12，内部扇出为 4（其他方法使用的 R - 树也采用了相同的值）。

- CIT08 (Mahran and Mahar 2008): the state of the art of exact DBSCAN, namely, the fastest existing algorithm able to produce the same DBSCAN result as KDD96.

- CIT08（Mahran 和 Mahar，2008 年）：精确 DBSCAN 的当前最优算法，即能够产生与 KDD96 相同 DBSCAN 结果的现有最快算法。

- SkLearn (http://scikit-learn.org/stable): the DBSCAN implementation in the popular machine-learning tool-kit scikit-learn. One should note that, judging from its website, SkLearn was implemented in Cython with its wrapper in Python.

- SkLearn（http://scikit - learn.org/stable）：流行的机器学习工具包 scikit - learn 中的 DBSCAN 实现。需要注意的是，从其网站判断，SkLearn 是用 Cython 实现的，其包装器用 Python 编写。

- OurExact: the exact DBSCAN algorithm we developed in Theorem 3.3, except that we did not use the BCP algorithm in Lemma 2.5; instead, we indexed the core points of each cell with an R-tree, and solved the BCP problem between two cells by repetitive nearest neighbor search (Hjaltason and Samet 1999) using the R-tree.

- OurExact：我们在定理 3.3 中开发的精确 DBSCAN 算法，但我们没有使用引理 2.5 中的 BCP 算法；相反，我们使用 R - 树对每个单元格的核心点进行索引，并通过使用 R - 树进行重复最近邻搜索（Hjaltason 和 Samet，1999 年）来解决两个单元格之间的 BCP 问题。

- OurApprox: the $\rho$ -approximate DBSCAN algorithm we proposed in Theorem 4.6. Our implementation has improved the one in the short version (Gan and Tao 2015) by incorporating new heuristics (see Section 6). In some experiments, we will also include the results of the old implementation-referred to as OurApprox-SIG-to demonstrate the effectiveness of those heuristics.

- OurApprox：我们在定理 4.6 中提出的 $\rho$ -近似 DBSCAN 算法。我们的实现通过纳入新的启发式方法改进了简短版本（Gan 和 Tao，2015 年）中的实现（见第 6 节）。在一些实验中，我们还将包括旧实现（称为 OurApprox - SIG）的结果，以证明这些启发式方法的有效性。

Each parameter was set to its default value unless otherwise stated. Remember that the default values of MinPts and $\epsilon$ may be different for various datasets (see Section 7.2).

除非另有说明，每个参数都设置为其默认值。请记住，不同数据集的 MinPts 和 $\epsilon$ 的默认值可能不同（见第 7.2 节）。

Influence of $\epsilon$ . The first experiment aimed to understand the behavior of each method under the influence of $\epsilon$ . Figure 18 plots the running time as a function of $\epsilon$ ,when this parameter varied from 100 to 5,000 (we refer the reader to Gan and Tao (2015) for running time comparison under $\epsilon  > 5,{000})$ .

$\epsilon$ 的影响。第一个实验旨在了解每种方法在 $\epsilon$ 影响下的表现。图 18 绘制了运行时间随 $\epsilon$ 变化的函数关系，当该参数从 100 变化到 5000 时（关于 $\epsilon  > 5,{000})$ 下的运行时间比较，请参考 Gan 和 Tao（2015 年））。

KDD96 and CIT08 retrieve,for each data point $p$ ,all the points in $B\left( {p,\epsilon }\right)$ . As discussed in Section 6, these methods may be efficient when $\epsilon$ is small,but their performance deteriorates rapidly as $\epsilon$ increases. This can be observed from the results in Figure 18. OurExact and OurApprox (particularly the latter) offered either competitive or significantly better efficiency at a vast majority of $\epsilon$ values. Such a property is useful in tuning this crucial parameter in reality. Specifically, it enables a user to try out a large number of values in a wide spectrum, without having to worry about the possibly prohibitive cost-note that KDD96 and CIT08 demanded over 1,000 seconds at many values of $\epsilon$ that have been found to be interesting in Section 7.2.

KDD96和CIT08方法会为每个数据点$p$检索$B\left( {p,\epsilon }\right)$中的所有点。如第6节所述，当$\epsilon$较小时，这些方法可能效率较高，但随着$\epsilon$的增大，其性能会迅速下降。从图18的结果中可以观察到这一点。我们的精确算法（OurExact）和近似算法（OurApprox，尤其是后者）在绝大多数$\epsilon$取值下，要么具有相当的效率，要么效率显著更高。这一特性在实际调整这个关键参数时非常有用。具体来说，它使用户能够在广泛的范围内尝试大量取值，而不必担心可能过高的成本——注意，在第7.2节中发现的许多有趣的$\epsilon$取值下，KDD96和CIT08方法需要超过1000秒的运行时间。

The performance of OurApprox-SIG is reported in the first synthetic dataset SS-simden-3D and the first real dataset PAMAP2. There are two main observations here. First, the proposed heuristics allowed the new implementation to outperform the one at SIGMOD quite significantly. Second, the in the improvement diminished as $\epsilon$ increased. This happens because for a larger $\epsilon$ ,the side length of a cell (in the grid imposed by our algorithm) increases, which decreases the number of non-empty cells. In that scenario,the graph $G$ (see Section 4.4) has only a small number of edges,thus making even a less delicate implementation (such as OurApprox-SIG) reasonably fast. In other words,

OurApprox - SIG算法的性能在第一个合成数据集SS - simden - 3D和第一个真实数据集PAMAP2中进行了评估。这里有两个主要发现。首先，所提出的启发式方法使新的实现显著优于在SIGMOD会议上提出的实现。其次，随着$\epsilon$的增大，性能提升逐渐减小。这是因为对于较大的$\epsilon$，（我们算法所使用的网格中）单元格的边长会增大，从而减少了非空单元格的数量。在这种情况下，图$G$（见第4.4节）只有少量的边，因此即使是不太精细的实现（如OurApprox - SIG）也能有相当快的速度。换句话说，

<!-- Media -->

<!-- figureText: OurApprox OurExact OurApprox-SIG SkLearn ${10}^{4}$ time (sec) ${10}^{3}$ ${10}^{2}$ 10 0.1 0.2 0.4 4 5 $\varepsilon \left( {10}^{3}\right)$ $\varepsilon \left( {10}^{3}\right)$ (b) SS-simden-5D (c) SS-simden-7D ${10}^{3}$ time (sec) ${10}^{2}$ 10 4 5 0.1 0.2 0.4 45 $\varepsilon \left( {10}^{3}\right)$ $\varepsilon \left( {10}^{3}\right)$ (e) SS-varden-5D (f) SS-varden-7D 1000 time (sec) 800 600 400 200 0 0.7.1 3 4 5 0.1 0.2 0.4 2 $\varepsilon \left( {10}^{5}\right)$ $\varepsilon \left( {10}^{3}\right)$ (h) Farm (i) Household CIT08 KDD96 ${10}^{4}$ time (sec) ${10}^{4}$ time (sec) ${10}^{3}$ ${10}^{3}$ ${10}^{2}$ ${10}^{2}$ 10 10 0.1 0.1 0.2 0.4 2 4 5 0.1 0.2 0.4 $\varepsilon \left( {10}^{3}\right)$ (a) SS-simden-3D ${10}^{4}$ time (sec) ${10}^{4}$ time (sec) ${10}^{3}$ ${10}^{3}$ ${10}^{2}$ ${10}^{2}$ 10 10 0.1 0.1 0.2 0.4 45 0.1 0.2 0.4 $\varepsilon \left( {10}^{3}\right)$ (d) SS-varden-3D 200 time (sec) 2000 time (sec) 150 1500 100 1000 50 500 10 0.1 0.2 0.4 45 0.1 0.2 0.4 $\varepsilon \left( {10}^{3}\right)$ (g) ${PAMAP2}$ -->

<img src="https://cdn.noedgeai.com/0195c91b-4734-77b1-933f-3c7cb33c7bfe_35.jpg?x=141&y=260&w=1274&h=1313&r=0"/>

Fig. 18. Running time vs. $\epsilon \left( {d \geq  3}\right)$ .

图18. 运行时间与$\epsilon \left( {d \geq  3}\right)$的关系。

<!-- Media -->

the importance of the heuristics is reflected chiefly in small $\epsilon$ . To avoid clattering the diagrams, OurApprox-SIG is omitted from the other datasets, but similar patterns were observed.

启发式方法的重要性主要体现在较小的$\epsilon$值上。为避免图表过于杂乱，在其他数据集中省略了OurApprox - SIG，但观察到了类似的模式。

Scalability with $n$ . The next experiment examined how each method scales with the number $n$ objects. For this purpose,we used synthetic SS datasets by varying $n$ from ${100}\mathrm{k}$ to ${10}\mathrm{\;m}$ ,using the default $\epsilon$ and MinPts values in Tables 2 and 3. The results are presented in Figure 19-note that the $y$ -axis is in log scale. If ${SkLearn}$ does not have a result at a value of $n$ ,it ran out of memory on our machine (same convention adopted in the rest of the evaluation). KDD96 and CIT08 had competitive performance on similar-density datasets, but they were considerably slower (by a factor over an order of magnitude) than OurApprox and OurExact on varying-density data, confirming the analysis in Section 6.

随$n$的可扩展性。接下来的实验研究了每种方法随对象数量$n$的可扩展性。为此，我们使用合成的SS数据集，将$n$从${100}\mathrm{k}$变化到${10}\mathrm{\;m}$，使用表2和表3中的默认$\epsilon$和MinPts值。结果如图19所示——注意，$y$轴采用对数刻度。如果${SkLearn}$在某个$n$值下没有结果，说明它在我们的机器上内存不足（在其余评估中采用相同的约定）。KDD96和CIT08方法在密度相似的数据集上性能相当，但在密度变化的数据集上，它们比OurApprox和OurExact方法慢得多（慢一个数量级以上），这证实了第6节的分析。

Influence of $\rho$ . Figure 20 shows the running time of OurApprox as $\rho$ changed from 0.001 to 0.1 . The algorithm did not appear sensitive to this parameter. This, at first glance, may look surprising, because the theoretical analysis in Section 4.3 implies that the running time should contain a multiplicative term ${\left( 1/\rho \right) }^{d - 1}$ ,as is the worst-case cost of an approximate range count query,which is in turn for detecting whether two cells in $G$ (i.e.,the grid our algorithm imposes) have an edge. There are primarily two reasons why such a dramatic blow-up was not observed. First, the union-find heuristic explained in Section 6 significantly reduces the number of edges that need to be detected: once two cells are found to be in the same connected component, it is unnecessary to detect their edges. Second, even when an edge detection is indeed required, its cost is unlikely to reach ${\left( 1/\rho \right) }^{d - 1}$ because our algorithm for answering an approximate range count query often

$\rho$的影响。图20展示了OurApprox算法在$\rho$从0.001变化到0.1时的运行时间。该算法似乎对这个参数不敏感。乍一看，这可能令人惊讶，因为第4.3节的理论分析表明，运行时间应该包含一个乘法项${\left( 1/\rho \right) }^{d - 1}$，这是近似范围计数查询的最坏情况成本，而该查询又用于检测$G$（即我们算法所使用的网格）中的两个单元格是否有边相连。没有观察到这种显著的性能下降主要有两个原因。首先，第6节中解释的并查集启发式方法显著减少了需要检测的边的数量：一旦发现两个单元格属于同一个连通分量，就无需检测它们之间的边。其次，即使确实需要进行边检测，其成本也不太可能达到${\left( 1/\rho \right) }^{d - 1}$，因为我们用于回答近似范围计数查询的算法通常

<!-- Media -->

<!-- figureText: OurApprox OurExact CIT08 KDD96 SkLearn ${10}^{3}$ time (sec) ${10}^{2}$ 10 0.1 10 10 n (million) n (million) (c) SS-simden-7D ${10}^{3}$ time (sec) ${10}^{2}$ 10 1 0.1 10 0.1 1 2 10 n (million) n (million) (f) SS-varden-7D ${10}^{3}$ time (sec) ${10}^{3}$ time (sec) ${10}^{2}$ ${10}^{2}$ 10 10 0.1 0.1 10 n (million) (a) SS-simden-3D (b) SS-simden-5D ${10}^{3}$ time (sec) ${10}^{3}$ time (sec) ${10}^{2}$ ${10}^{2}$ 10 10 1 0.1 1 0.01 5 10 0.1 2 n (million) (d) SS-varden-3D (e) SS-varden-5D -->

<img src="https://cdn.noedgeai.com/0195c91b-4734-77b1-933f-3c7cb33c7bfe_36.jpg?x=148&y=260&w=1277&h=875&r=0"/>

Fig. 19. Running time vs. $n\left( {d \geq  3}\right)$ .

图19. 运行时间与$n\left( {d \geq  3}\right)$的关系。

<!-- figureText: ${10}^{2}$ time (sec) ${10}^{2}$ time (sec) SS-varden-3D ${10}^{5}$ time (sec) PAMAP2 SS-varden-5D -□ ${10}^{4}$ Farm Household SS-varden-7D $\rightarrow$ ${10}^{3}$ ${10}^{2}$ 10 0.06 0.08 0.1 0.001 0.02 0.04 0.06 0.08 0.1 p p (b) SS varying density data (c) Real datasets SS-simden-3D SS-simden-5D - ☐ SS-simden-7D $\rightarrow$ 10 10 0.001 0.02 0.04 0.06 0.08 0.1 0.001 0.02 0.04 (a) SS similar density data -->

<img src="https://cdn.noedgeai.com/0195c91b-4734-77b1-933f-3c7cb33c7bfe_36.jpg?x=149&y=1243&w=1269&h=434&r=0"/>

Fig. 20. Running time vs. $\rho \left( {d \geq  3}\right)$ .

图20. 运行时间与$\rho \left( {d \geq  3}\right)$的关系。

<!-- figureText: OurApprox OurExact CIT08 KDD96 SkLearn 120 time (sec) 100 80 60 40 20 60 80 100 0 20 40 60 80 100 minPts minPts (c) SS-simden-7D ${10}^{3}$ time (sec) ${10}^{2}$ 10 60 80 100 20 40 60 80 100 minPts minPts (f) SS-varden-7D ${10}^{3}$ time (sec) ${10}^{2}$ 10 60 80 100 20 40 60 80 100 minPts minPts (i) Household 60 time (sec) 80 time (sec) 45 60 30 40 15 20 0 20 40 60 80 100 10 20 40 minPts (a) SS-simden-3D (b) SS-simden-5D ${10}^{3}$ time (sec) ${10}^{3}$ time (sec) ${10}^{2}$ ${10}^{2}$ 10 10 1 10 20 40 60 80 100 10 20 40 minPts (d) SS-varden-3D (e) SS-varden-5D ${10}^{3}$ time (sec) ${10}^{4}$ time (sec) ${10}^{2}$ ${10}^{3}$ 10 ${10}^{2}$ 10 20 40 60 80 100 10 20 40 minPts (g) ${PAMAP2}$ (h) Farm -->

<img src="https://cdn.noedgeai.com/0195c91b-4734-77b1-933f-3c7cb33c7bfe_37.jpg?x=143&y=257&w=1275&h=1269&r=0"/>

Fig. 21. Running time vs. MinPts $\left( {d \geq  3}\right)$ .

图21. 运行时间与MinPts $\left( {d \geq  3}\right)$的关系。

<!-- Media -->

terminates without exploring the underlying quad-tree completely-recall that the algorithm can terminate as soon as it is certain whether the query answer is zero.

在未完全遍历底层四叉树时就会终止——请记住，一旦确定查询答案是否为零，算法就可以终止。

Influence of MinPts. The last set of experiments in this section measured the running time of each method when MinPts increased from 10 to 100 . The results are given in Figure 21. The impact of this parameter was limited, and did not change any of the observations made earlier.

最小点数（MinPts）的影响。本节的最后一组实验测量了最小点数从10增加到100时每种方法的运行时间。结果如图21所示。该参数的影响有限，且未改变之前的任何观察结果。

### 7.5 Computational Efficiency for $d = 2$

### 7.5 针对$d = 2$的计算效率

In this subsection, we will focus on exact DBSCAN in 2D space, and compare the following algorithms:

在本小节中，我们将专注于二维空间中的精确DBSCAN算法，并比较以下算法：

- KDD96 and SkLearn: As introduced at the beginning of Section 7.4.

- KDD96和SkLearn：如7.4节开头所介绍的。

- G13 (Gunawan 2013): The $O\left( {n\log n}\right)$ time algorithm by Gunawan,as reviewed in Section 2.2.

- G13（古纳万（Gunawan）2013年提出）：古纳万提出的$O\left( {n\log n}\right)$时间复杂度算法，如2.2节所述。

-Delaunay: Our algorithm as explained in Section 5.1,which runs in $O\left( {n\log n}\right)$ time.

- 德劳内三角剖分（Delaunay）：我们在5.1节中解释的算法，其运行时间为$O\left( {n\log n}\right)$。

<!-- Media -->

<!-- figureText: Wavefront Delaunay G13 KDD96 SkLearn ${10}^{3}$ time (sec) ${10}^{2}$ 10 1 0.1 0.01 2 10 n (million) (b) SS-varden-2D ${10}^{2}$ time (sec) 10 0.1 0.01 0.1 2 10 n (million) (a) SS-simden-2D -->

<img src="https://cdn.noedgeai.com/0195c91b-4734-77b1-933f-3c7cb33c7bfe_38.jpg?x=198&y=264&w=1174&h=486&r=0"/>

Fig. 22. Running time vs. $n\left( {d = 2}\right)$ .

图22. 运行时间与$n\left( {d = 2}\right)$的关系。

<!-- figureText: Wavefront Delaunay KDD96 SkLearn ${10}^{4}$ time (sec) ${10}^{3}$ ${10}^{2}$ 10 1 0.1 0.1 1 2 $\varepsilon \left( {10}^{3}\right)$ (b) SS-varden-2D ${10}^{3}$ time (sec) ${10}^{2}$ 10 1 0.1 100 200 400 800 E (a) SS-simden-2D -->

<img src="https://cdn.noedgeai.com/0195c91b-4734-77b1-933f-3c7cb33c7bfe_38.jpg?x=198&y=836&w=1171&h=488&r=0"/>

Fig. 23. Running time vs. $\epsilon \left( {d = 2}\right)$ .

图23. 运行时间与$\epsilon \left( {d = 2}\right)$的关系。

<!-- Media -->

- Wavefront: Our algorithm as in Theorem 5.2, assuming that the dataset has been bi-dimensionally sorted-recall that this is required to ensure the linear-time complexity of the algorithm.

- 波前算法（Wavefront）：如定理5.2所述的我们的算法，假设数据集已按二维排序——请回想一下，这是确保该算法线性时间复杂度所必需的。

Once again, each parameter was set to its default value (see Table 1 and Section 7.2) unless otherwise stated. All the experiments in this subsection were based on SS similar- and varying-density datasets.

再次强调，除非另有说明，每个参数都设置为其默认值（见表1和7.2节）。本小节的所有实验均基于SS相似和变密度数据集。

Results. In the experiment of Figure 22, we measured the running time of each algorithm as the cardinality $n$ escalated from ${100}\mathrm{k}$ to ${10}\mathrm{\;m}$ . Wavefront consistently outperformed all the other methods, while Delaunay was observed to be comparable to G13. It is worth pointing out the vast difference between the running time here and that shown in Figure 19 for $d \geq  3$ (one can feel the difficulty gap of the DBSCAN problem between $d = 2$ and $d \geq  3$ ).

结果。在图22的实验中，我们测量了随着基数$n$从${100}\mathrm{k}$增加到${10}\mathrm{\;m}$时每种算法的运行时间。波前算法始终优于所有其他方法，而观察到德劳内三角剖分算法与G13算法相当。值得指出的是，这里的运行时间与图19中针对$d \geq  3$的运行时间存在巨大差异（可以感受到在$d = 2$和$d \geq  3$情况下DBSCAN问题的难度差距）。

Next,we compared the running time of the five algorithms by varying $\epsilon$ . As shown in Figure 23, the cost of Wavefront,Delaunay,and G13 actually improved as $\epsilon$ grew,whereas KDD96 and SkLearn worsened. Wavefront was the overall winner by a wide margin.

接下来，我们通过改变$\epsilon$来比较这五种算法的运行时间。如图23所示，随着$\epsilon$的增加，波前算法、德劳内三角剖分算法和G13算法的成本实际上有所降低，而KDD96和SkLearn算法的成本则增加。波前算法以较大优势成为总体赢家。

Finally, we inspected the influence of MinPts on the running time. The results are presented in Figure 24. In general, for a larger MinPts, Wavefront, Delaunay, and G13 require a higher cost in labeling the data points as core or non-core points. The influence, however, is contained by the fact

最后，我们考察了最小点数对运行时间的影响。结果如图24所示。一般来说，对于较大的最小点数，波前算法、德劳内三角剖分算法和G13算法在将数据点标记为核心点或非核心点时需要更高的成本。然而，由于与数据集大小相比，该参数被设置为一个小常数，这种影响得到了控制。

<!-- Media -->

<!-- figureText: Wavefront Delaunay G13 KDD96 SkLearn ${10}^{3}$ time (sec) ${10}^{2}$ 10 1 0.1 20 40 60 80 100 minPts (b) SS-varden-2D ${10}^{2}$ time (sec) 10 1 0.1 20 40 60 80 100 minPts (a) SS-simden-2D -->

<img src="https://cdn.noedgeai.com/0195c91b-4734-77b1-933f-3c7cb33c7bfe_39.jpg?x=194&y=263&w=1176&h=487&r=0"/>

Fig. 24. Running time vs. MinPts $\left( {d = 2}\right)$ .

图24. 运行时间与最小点数$\left( {d = 2}\right)$的关系。

<!-- Media -->

that this parameter is set as a small constant compared to the dataset size. The relative superiority of all the methods remained the same.

所有方法的相对优势保持不变。

## 8 CONCLUSIONS

## 8 结论

DBSCAN is an effective technique for density-based clustering, which is very extensively applied in data mining, machine learning, and databases. However, currently there has not been clear understanding on its theoretical computational hardness. All the existing algorithms suffer from a time complexity that is quadratic to the dataset size $n$ when the dimensionality $d$ is at least 3 .

DBSCAN（基于密度的空间聚类应用程序）是一种有效的基于密度的聚类技术，在数据挖掘、机器学习和数据库领域应用广泛。然而，目前对其理论计算复杂度尚无清晰的认识。当数据维度 $d$ 至少为 3 时，现有的所有算法的时间复杂度与数据集大小 $n$ 呈二次方关系。

In this article, we show that, unless very significant breakthroughs (ones widely believed to be impossible) can be made in theoretical computer science,the DBSCAN problem requires $\Omega \left( {n}^{4/3}\right)$ time to solve for $d \geq  3$ under the Euclidean distance. This excludes the possibility of finding an algorithm of near-linear running time, thus motivating the idea of computing approximate clusters. Toward that direction,we propose $\rho$ -approximate DBSCAN,and prove both theoretical and experimentally that the new method has excellent guarantees both in the quality of cluster approximation and computational efficiency.

在本文中，我们证明了，除非理论计算机科学取得重大突破（人们普遍认为这是不可能的），否则在欧几里得距离下，求解维度为 $d \geq  3$ 的 DBSCAN 问题需要 $\Omega \left( {n}^{4/3}\right)$ 的时间。这排除了找到近线性运行时间算法的可能性，从而促使我们考虑计算近似聚类。为此，我们提出了 $\rho$ -近似 DBSCAN 算法，并从理论和实验两方面证明了该新方法在聚类近似质量和计算效率方面都有出色的表现。

The exact DBSCAN problem in dimensionality $d = 2$ is known to be solvable in $O\left( {n\log n}\right)$ time. This article further enhances that understanding by showing how to settle the problem in $O\left( n\right)$ time, provided that the data points have already been pre-sorted on each dimension. In other words, coordinating sorting is in fact the hardest component of the 2D DBSCAN problem. The result immediately implies that, when all the coordinates are integers, the problem can be solved in $O\left( {n\log \log n}\right)$ time deterministically,or $O\left( {n\sqrt{\log \log n}}\right)$ expected time randomly.

已知维度为 $d = 2$ 的精确 DBSCAN 问题可以在 $O\left( {n\log n}\right)$ 时间内求解。本文进一步深化了这一认识，表明在数据点已按每个维度预排序的情况下，该问题可以在 $O\left( n\right)$ 时间内解决。换句话说，坐标排序实际上是二维 DBSCAN 问题中最困难的部分。这一结果直接表明，当所有坐标都是整数时，该问题可以确定性地在 $O\left( {n\log \log n}\right)$ 时间内解决，或者以随机期望时间 $O\left( {n\sqrt{\log \log n}}\right)$ 解决。

We close the article with a respectful remark. The objective of the article, as well as its short version (Gan and Tao 2015), is to understand the computational complexity of DBSCAN and how to bring down the complexity with approximation. The intention has never, by any means, been to argue against the significance of DBSCAN-on the contrary, there is no doubt that DBSCAN has proved to be a highly successful technique. In fact, even though many algorithmic aspects about this technique have been resolved in this article, from the data mining perspective, how to choose between exact DBSCAN (even implemented just as in the KDD96 algorithm) and our approximate DBSCAN is far from being conclusive. There are,for sure,datasets where a small $\epsilon$ value suffices, in which case exact DBSCAN may finish even faster than the approximate version. However, selecting the right parameters is seldom trivial in reality, and often requires multiple iterations of "trial and error". The proposed approximate algorithm has the advantage of being reasonably fast regardless of the parameters. This allows users to inspect the clusters under numerous parameter values in a (much) more efficient manner. With this said, we feel that the exact and approximate versions can comfortably co-exist with each other: the approximate algorithm serves nicely as a "filtering step" for the exact algorithm.

在文章结尾，我们谨作如下说明。本文及其简短版本（Gan 和 Tao 2015）的目的是了解 DBSCAN 的计算复杂度，以及如何通过近似方法降低复杂度。我们绝无意否定 DBSCAN 的重要性——相反，毫无疑问，DBSCAN 已被证明是一种非常成功的技术。事实上，尽管本文已经解决了该技术的许多算法方面的问题，但从数据挖掘的角度来看，如何在精确 DBSCAN（即使只是按照 KDD96 算法实现）和我们的近似 DBSCAN 之间做出选择，远未得出定论。当然，有些数据集只需较小的 $\epsilon$ 值即可，在这种情况下，精确 DBSCAN 甚至可能比近似版本更快完成。然而，在现实中选择合适的参数很少是一件简单的事情，通常需要多次“试错”。我们提出的近似算法的优点是无论参数如何都能保持相当快的速度。这使得用户能够以（更）高效的方式检查众多参数值下的聚类情况。综上所述，我们认为精确版本和近似版本可以很好地共存：近似算法可以很好地作为精确算法的“过滤步骤”。

## A APPENDIX: SOLVING USEC WITH LINE SEPARATION (PROOF OF LEMMA 5.5)

## A 附录：用直线分隔解决 USEC 问题（引理 5.5 的证明）

We consider that all the disks in ${S}_{\text{ball }}$ intersect $\ell$ (any disk completely below $\ell$ can be safely discarded), and that all disks are distinct (otherwise, simply remove the redundant ones).

我们假设 ${S}_{\text{ball }}$ 中的所有圆盘都与 $\ell$ 相交（任何完全位于 $\ell$ 下方的圆盘都可以安全地舍弃），并且所有圆盘都是不同的（否则，只需移除冗余的圆盘）。

For each disk $s \in  {S}_{\text{ball }}$ ,we define its portion on or above $\ell$ as its active region (because only this region may contain points of ${S}_{pt}$ ). Also,we use the term upper arc to refer to the portion of the boundary of $s$ that is strictly above $\ell$ . See Figure 25 for an illustration of these notions (the upper arc is in bold). Note that,as the center of $s$ is on or below $\ell$ ,the active region and upper arc of $s$ are at most a semi-disk and a semi-circle, respectively. The following is a basic geometric fact:

对于每个圆盘 $s \in  {S}_{\text{ball }}$，我们将其位于 $\ell$ 上或上方的部分定义为其活动区域（因为只有该区域可能包含 ${S}_{pt}$ 中的点）。此外，我们用“上弧”来表示圆盘 $s$ 边界中严格位于 $\ell$ 上方的部分。有关这些概念的说明，请参见图 25（上弧用粗线表示）。请注意，由于圆盘 $s$ 的圆心位于 $\ell$ 上或下方，因此圆盘 $s$ 的活动区域和上弧分别最多为一个半圆和一个半圆弧。以下是一个基本的几何事实：

Proposition A.1. The upper arcs of any two disks in ${S}_{\text{ball }}$ can have at most one intersection point.

命题 A.1. ${S}_{\text{ball }}$ 中任意两个圆盘的上弧最多有一个交点。

Define the coverage region-denoted by $U$ -of ${S}_{\text{ball }}$ as the union of the active regions of all the disks in ${S}_{\text{ball }}$ . Figure 26(a) demonstrates $U$ for the example of Figure 10. Evidently,the answer of the USEC instance is yes if and only if ${S}_{pt}$ has at least a point falling in $U$ .

定义 ${S}_{\text{ball }}$ 的覆盖区域（用 $U$ 表示）为 ${S}_{\text{ball }}$ 中所有圆盘的活动区域的并集。图 26(a) 展示了图 10 示例中的 $U$。显然，USEC 实例的答案为“是”当且仅当 ${S}_{pt}$ 中至少有一个点落在 $U$ 内。

We use the term wavefront to refer to the part of the boundary of $U$ that is strictly above $\ell$ (see the solid curve in Figure 26(b)). A disk in ${S}_{\text{ball }}$ is said to be contributing,if it defines an arc on the wavefront. In Figure 26(b), for instance, the wavefront is defined by three contributing disks, which are shown in bold and labeled as ${s}_{1},{s}_{3},{s}_{6}$ in Figure 26(a).

我们使用术语“波前”（wavefront）来指代$U$边界中严格位于$\ell$上方的部分（见图26(b)中的实线曲线）。若${S}_{\text{ball }}$中的一个圆盘在波前上定义了一段弧，则称该圆盘为贡献圆盘。例如，在图26(b)中，波前由三个贡献圆盘定义，这些圆盘在图26(a)中用粗线表示并标记为${s}_{1},{s}_{3},{s}_{6}$。

It is rudimentary to verify the next three facts:

验证接下来的三个事实是很基础的：

Proposition A.2. U equals the union of the active regions of the contributing disks in ${S}_{\text{ball }}$ .

命题A.2：U等于${S}_{\text{ball }}$中贡献圆盘的活跃区域的并集。

Proposition A.3. Every contributing disk defines exactly one arc on the wavefront.

命题A.3：每个贡献圆盘在波前上恰好定义一段弧。

Proposition A.4. The wavefront is x-monotone, namely, no vertical line can intersect it at two points.

命题A.4：波前是x单调的，即没有垂直线可以与它相交于两点。

<!-- Media -->

<!-- figureText: upper arc of $s$ active region of $s$ -->

<img src="https://cdn.noedgeai.com/0195c91b-4734-77b1-933f-3c7cb33c7bfe_40.jpg?x=511&y=1509&w=551&h=168&r=0"/>

Fig. 25. Illustration of active region and upper arc.

图25. 活跃区域和上弧的图示。

<!-- figureText: ${S}_{3}$ (b) Wavefront (solid curve) ${s}_{1}$ ${s}_{6}$ ${s}_{5}$ ${s}_{2}$ ${s}_{4}$ (a) Coverage region $U$ (the shaded area) -->

<img src="https://cdn.noedgeai.com/0195c91b-4734-77b1-933f-3c7cb33c7bfe_40.jpg?x=165&y=1767&w=1249&h=302&r=0"/>

Fig. 26. Deciding the existence of an edge by USEC with line separation.

图26. 使用带直线分隔的USEC方法确定边的存在性。

<!-- figureText: ${W}_{1}$ l $p$ ${I}_{2}$ ${s}_{2}l$ -->

<img src="https://cdn.noedgeai.com/0195c91b-4734-77b1-933f-3c7cb33c7bfe_41.jpg?x=489&y=262&w=590&h=253&r=0"/>

Fig. 27. Illustration of the Step-1 algorithm in Section A.1.

图27. A.1节中步骤1算法的图示。

<!-- Media -->

### A.1 Computing the Wavefront in Linear Time

### A.1 线性时间计算波前

Utilizing the property that the centers of the disks in ${S}_{\text{ball }}$ have been sorted by x-dimension,next we explain how to compute the wavefront in $O\left( \left| {S}_{\text{ball }}\right| \right)$ time.

利用${S}_{\text{ball }}$中圆盘的中心已按x维度排序这一性质，接下来我们将解释如何在$O\left( \left| {S}_{\text{ball }}\right| \right)$时间内计算波前。

Label the disks in ${S}_{\text{ball }}$ as ${s}_{1},{s}_{2},{s}_{3},\ldots$ ,in ascending order of their centers’ $\mathrm{x}$ -coordinates. Let ${U}_{i}$ $\left( {1 \leq  i \leq  \left| {S}_{\text{ball }}\right| }\right)$ be the coverage region that unions the active regions of the first $i$ disks. Apparently, ${U}_{1} \subseteq  {U}_{2} \subseteq  {U}_{3} \subseteq  \ldots$ ,and ${U}_{\left| {S}_{\text{ball }}\right| }$ is exactly $U$ . Define ${W}_{i}$ to be the wavefront of ${U}_{i}$ ,namely,the portion of the boundary of ${U}_{i}$ strictly above $\ell$ . Our algorithm captures ${W}_{i}$ in a linked list $\mathcal{L}\left( {W}_{i}\right)$ , which arranges the defining disks (of ${W}_{i}$ ) in left-to-right order of the arcs (on ${W}_{i}$ ) they define (e.g.,in Figure 11, $\mathcal{L}\left( {W}_{6}\right)$ lists ${s}_{1},{s}_{3},{s}_{6}$ in this order). By Proposition A.3,every disk appears in $\mathcal{L}\left( {W}_{i}\right)$ at most once. Our goal is to compute ${\left. W\right| }_{{S}_{\text{ball }}}$ ,which is sufficient for deriving $U$ according to Proposition A.2.

将${S}_{\text{ball }}$中的圆盘按其中心的$\mathrm{x}$坐标升序标记为${s}_{1},{s}_{2},{s}_{3},\ldots$。设${U}_{i}$ $\left( {1 \leq  i \leq  \left| {S}_{\text{ball }}\right| }\right)$为前$i$个圆盘的活跃区域的并集所构成的覆盖区域。显然，${U}_{1} \subseteq  {U}_{2} \subseteq  {U}_{3} \subseteq  \ldots$，且${U}_{\left| {S}_{\text{ball }}\right| }$恰好就是$U$。定义${W}_{i}$为${U}_{i}$的波前，即${U}_{i}$边界中严格位于$\ell$上方的部分。我们的算法将${W}_{i}$存储在一个链表$\mathcal{L}\left( {W}_{i}\right)$中，该链表按照定义的弧（在${W}_{i}$上）从左到右的顺序排列定义圆盘（${W}_{i}$的）（例如，在图11中，$\mathcal{L}\left( {W}_{6}\right)$按此顺序列出${s}_{1},{s}_{3},{s}_{6}$）。根据命题A.3，每个圆盘在$\mathcal{L}\left( {W}_{i}\right)$中最多出现一次。我们的目标是计算${\left. W\right| }_{{S}_{\text{ball }}}$，根据命题A.2，这足以推导出$U$。

It is straightforward to obtain ${W}_{1}$ from ${s}_{1}$ in constant time. In general,provided that ${W}_{i - 1}\left( {i \geq  2}\right)$ is ready,we obtain ${W}_{i}$ in three steps:

在常数时间内从${s}_{1}$得到${W}_{1}$是很直接的。一般来说，假设${W}_{i - 1}\left( {i \geq  2}\right)$已准备好，我们通过三个步骤得到${W}_{i}$：

(1) Check if ${s}_{i}$ defines any arc on ${W}_{i}$ .

(1) 检查${s}_{i}$是否在${W}_{i}$上定义了任何弧。

(2) If the answer is no,set ${W}_{i} = {W}_{i - 1}$ .

(2) 如果答案是否定的，则令${W}_{i} = {W}_{i - 1}$。

(3) Otherwise,derive ${W}_{i}$ from ${W}_{i - 1}$ using ${s}_{i}$ .

(3) 否则，使用${s}_{i}$从${W}_{i - 1}$推导出${W}_{i}$。

Next, we describe how to implement Steps 1 and 3.

接下来，我们描述如何实现步骤1和步骤3。

Step 1. We perform this step in constant time as follows. Compute the intersection of ${s}_{i}$ and $\ell$ . The intersection is an interval on $\ell$ ,denoted as ${I}_{i}$ . Let ${s}_{\text{last }}$ be the rightmost defining disk of ${W}_{i - 1}$ (i.e., the last disk in $\mathcal{L}\left( {W}_{i - 1}\right)$ ). If the right endpoint of ${I}_{i}$ lies in ${s}_{\text{last }}$ ,return ${no}$ (that is, ${s}_{i}$ does not define any arc on ${W}_{i}$ ); otherwise,return yes.

步骤1. 我们按如下方式在常数时间内执行此步骤。计算${s}_{i}$和$\ell$的交集。该交集是$\ell$上的一个区间，记为${I}_{i}$。设${s}_{\text{last }}$为${W}_{i - 1}$最右侧的定义圆盘（即$\mathcal{L}\left( {W}_{i - 1}\right)$中的最后一个圆盘）。如果${I}_{i}$的右端点位于${s}_{\text{last }}$内，则返回${no}$（即${s}_{i}$在${W}_{i}$上不定义任何弧）；否则，返回“是”。

As an example,consider the processing of ${s}_{2}$ in Figure 26(a). At this moment, ${W}_{1}$ is as shown in Figure 27,and includes a single arc contributed by ${s}_{1}$ . Point $p$ is the right endpoint of ${I}_{2}$ . As $p$ falls in ${s}_{1}\left( { = {s}_{\text{last }}}\right)$ ,we declare that ${s}_{2}$ does not define any arc on ${W}_{2}$ (which therefore equals ${W}_{1}$ ).

例如，考虑图26(a)中${s}_{2}$的处理。此时，${W}_{1}$如图27所示，并且包含由${s}_{1}$贡献的单个弧。点$p$是${I}_{2}$的右端点。由于$p$落在${s}_{1}\left( { = {s}_{\text{last }}}\right)$内，我们声明${s}_{2}$在${W}_{2}$上不定义任何弧（因此${W}_{2}$等于${W}_{1}$）。

The lemma below proves the correctness of our strategy in general:

下面的引理证明了我们的策略总体上的正确性：

LEMMA A.5. Our Step-1 algorithm always makes the correct decision.

引理A.5. 我们的步骤1算法总是能做出正确的决策。

Proof. Consider first the case where the right endpoint $p$ of ${I}_{i}$ is covered by ${s}_{\text{last }}$ . Let ${I}_{\text{last }}$ be the intersection between ${s}_{\text{last }}$ and $\ell$ . By the facts that (i) the x-coordinate of the center of ${s}_{i}$ is larger than or equal to that of the center of ${s}_{\text{last }}$ ,and (ii) ${s}_{i}$ and ${s}_{\text{last }}$ have the same radius,it must hold that ${I}_{i}$ is contained in ${I}_{\text{last }}$ . This implies that the active region of ${s}_{i}$ must be contained in that of ${s}_{\text{last }}$ (otherwise,the upper arc of ${s}_{i}$ needs to go out of ${s}_{\text{last }}$ and then back in,and hence,must intersect the upper arc of ${s}_{\text{last }}$ at two points,violating Proposition A.1). This means that ${s}_{i}$ cannot define any arc on ${W}_{i}$ ; hence,our no decision in this case is correct.

证明。首先考虑${I}_{i}$的右端点$p$被${s}_{\text{last }}$覆盖的情况。设${I}_{\text{last }}$为${s}_{\text{last }}$和$\ell$的交集。根据以下事实：(i) ${s}_{i}$的圆心的x坐标大于或等于${s}_{\text{last }}$的圆心的x坐标，以及(ii) ${s}_{i}$和${s}_{\text{last }}$具有相同的半径，必然有${I}_{i}$包含于${I}_{\text{last }}$。这意味着${s}_{i}$的活动区域必然包含于${s}_{\text{last }}$的活动区域（否则，${s}_{i}$的上弧需要离开${s}_{\text{last }}$然后再进入，因此，必然会与${s}_{\text{last }}$的上弧在两点相交，这违反了命题A.1）。这意味着${s}_{i}$在${W}_{i}$上不能定义任何弧；因此，在这种情况下我们返回“否”的决策是正确的。

Now consider the case where $p$ is not covered by ${s}_{\text{last }}$ . This implies that $p$ is not covered by ${U}_{i - 1}$ , meaning that ${I}_{i}$ must define an arc on ${W}_{i}$ (because at least $p$ needs to appear in ${U}_{i}$ ). Our yes decision is therefore correct.

现在考虑$p$未被${s}_{\text{last }}$覆盖的情况。这意味着$p$未被${U}_{i - 1}$覆盖，即${I}_{i}$必须在${W}_{i}$上定义一个弧（因为至少$p$需要出现在${U}_{i}$中）。因此，我们返回“是”的决策是正确的。

<!-- Media -->

<!-- figureText: arc defined by ${s}_{1}$ arc defined by ${s}_{3}$ ${s}_{6}$ arc defined by ${s}_{5}$ -->

<img src="https://cdn.noedgeai.com/0195c91b-4734-77b1-933f-3c7cb33c7bfe_42.jpg?x=469&y=264&w=630&h=282&r=0"/>

Fig. 28. Illustration of the Step-3 algorithm in Section A.1.

图28. A.1节中步骤3算法的图示。

<!-- Media -->

Step 3. We derive $\mathcal{L}\left( {W}_{i}\right)$ by possibly removing several disks at the end of $\mathcal{L}\left( {W}_{i - 1}\right)$ ,and then eventually appending ${s}_{i}$ . Specifically:

步骤3. 我们通过可能移除$\mathcal{L}\left( {W}_{i - 1}\right)$末尾的几个圆盘，然后最终添加${s}_{i}$来得到$\mathcal{L}\left( {W}_{i}\right)$。具体如下：

(3.1) Set $\mathcal{L}\left( {W}_{i}\right)$ to $\mathcal{L}\left( {W}_{i - 1}\right)$ .

(3.1) 将$\mathcal{L}\left( {W}_{i}\right)$设为$\mathcal{L}\left( {W}_{i - 1}\right)$。

(3.2) Set ${s}_{\text{last }}$ to the last disk in $\mathcal{L}\left( {W}_{i}\right)$ .

(3.2) 将${s}_{\text{last }}$设为$\mathcal{L}\left( {W}_{i}\right)$中的最后一个圆盘。

(3.3) If the arc on ${W}_{i - 1}$ defined by ${s}_{\text{last }}$ is contained in ${s}_{i}$ ,remove ${s}_{\text{last }}$ from $\mathcal{L}\left( {W}_{i}\right)$ and repeat from Step 3.2.

(3.3) 如果由${s}_{\text{last }}$在${W}_{i - 1}$上定义的弧包含于${s}_{i}$，则从$\mathcal{L}\left( {W}_{i}\right)$中移除${s}_{\text{last }}$，并从步骤3.2开始重复。

(3.4) Otherwise,append ${s}_{i}$ to the end of $\mathcal{L}\left( {W}_{i}\right)$ and finish.

(3.4) 否则，将${s}_{i}$追加到$\mathcal{L}\left( {W}_{i}\right)$的末尾并结束。

To illustrate,consider the processing of ${s}_{6}$ in Figure 26(a). At this moment,the wavefront ${W}_{5}$ is as shown in Figure 28,where the arcs are defined by ${s}_{1},{s}_{3}$ ,and ${s}_{5}$ ,respectively. Our Step-3 algorithm starts by setting $\mathcal{L}\left( {W}_{6}\right)$ to $\mathcal{L}\left( {W}_{5}\right)$ ,which lists ${s}_{1},{s}_{3},{s}_{5}$ in this order. Currently, ${s}_{\text{last }} = {s}_{5}$ . As the arc on ${W}_{5}$ defined by ${s}_{5}$ is covered by ${s}_{6}$ (see Figure 28),we remove ${s}_{5}$ from $\mathcal{L}\left( {W}_{6}\right)$ ,after which ${s}_{\text{last }}$ becomes ${s}_{3}$ . As the arc on ${W}_{5}$ defined by ${s}_{3}$ is not contained in ${s}_{6}$ ,the algorithm terminates by adding ${s}_{6}$ to the end of $\mathcal{L}\left( {W}_{6}\right)$ ,which now lists ${s}_{1},{s}_{3},{s}_{6}$ in this order.

为了说明这一点，考虑图26(a)中${s}_{6}$的处理过程。此时，波前${W}_{5}$如图28所示，其中的弧分别由${s}_{1},{s}_{3}$和${s}_{5}$定义。我们的步骤3算法首先将$\mathcal{L}\left( {W}_{6}\right)$设置为$\mathcal{L}\left( {W}_{5}\right)$，它按此顺序列出了${s}_{1},{s}_{3},{s}_{5}$。当前，${s}_{\text{last }} = {s}_{5}$。由于${W}_{5}$上由${s}_{5}$定义的弧被${s}_{6}$覆盖（见图28），我们从$\mathcal{L}\left( {W}_{6}\right)$中移除${s}_{5}$，之后${s}_{\text{last }}$变为${s}_{3}$。由于${W}_{5}$上由${s}_{3}$定义的弧不包含在${s}_{6}$中，算法通过将${s}_{6}$添加到$\mathcal{L}\left( {W}_{6}\right)$的末尾而终止，此时$\mathcal{L}\left( {W}_{6}\right)$按此顺序列出了${s}_{1},{s}_{3},{s}_{6}$。

Now we prove the correctness of our algorithm:

现在我们证明我们算法的正确性：

LEMMA A.6. Our Step-3 algorithm always obtains the correct $\mathcal{L}\left( {W}_{i}\right)$ .

引理A.6. 我们的步骤3算法总是能得到正确的$\mathcal{L}\left( {W}_{i}\right)$。

Proof. If the arc on ${W}_{i - 1}$ defined by ${s}_{\text{last }}$ is covered by ${s}_{i}$ ,the upper arc of ${s}_{\text{last }}$ must be covered by the union of the disks in $\left\{  {{s}_{1},{s}_{2},\ldots ,{s}_{i}}\right\}   \smallsetminus  \left\{  {s}_{\text{last }}\right\}$ . Therefore, ${s}_{\text{last }}$ is not a defining disk of ${W}_{i}$ and should be removed.

证明。如果${W}_{i - 1}$上由${s}_{\text{last }}$定义的弧被${s}_{i}$覆盖，那么${s}_{\text{last }}$的上弧必定被$\left\{  {{s}_{1},{s}_{2},\ldots ,{s}_{i}}\right\}   \smallsetminus  \left\{  {s}_{\text{last }}\right\}$中的圆盘的并集覆盖。因此，${s}_{\text{last }}$不是${W}_{i}$的定义圆盘，应该被移除。

Otherwise, ${s}_{\text{last }}$ must be retained. Furthermore,in this case, ${s}_{i}$ cannot touch the arc on ${W}_{i - 1}$ defined by any of the disks that are before ${s}_{\text{last }}$ in $\mathcal{L}\left( {W}_{i}\right)$ . All those disks,therefore,should also be retained.

否则，必须保留${s}_{\text{last }}$。此外，在这种情况下，${s}_{i}$不能与${W}_{i - 1}$上由$\mathcal{L}\left( {W}_{i}\right)$中在${s}_{\text{last }}$之前的任何圆盘所定义的弧相接触。因此，所有这些圆盘也应该被保留。

Finally,by Lemma A. 5 and the fact that the execution is at Step 3,we know that ${s}_{i}$ defines an arc on ${W}_{i}$ ,and thus,should be added to $\mathcal{L}\left( {W}_{i}\right)$ .

最后，根据引理A.5以及执行到步骤3这一事实，我们知道${s}_{i}$在${W}_{i}$上定义了一条弧，因此应该将其添加到$\mathcal{L}\left( {W}_{i}\right)$中。

Running Time. It remains to bound the cost of our wavefront computation algorithm. Step 1 obviously takes $O\left( \left| {S}_{\text{ball }}\right| \right)$ time in total. Step 2 demands $\mathop{\sum }\limits_{{i = 1}}^{n}O\left( {1 + {x}_{i}}\right)$ time,where ${x}_{i}$ is the number of disks deleted at Step 3.3 when processing disk ${s}_{i}$ . The summation evaluates to $O\left( \left| {S}_{\text{ball }}\right| \right)$ ,noticing that $\mathop{\sum }\limits_{{i = 1}}^{n}{x}_{i} \leq  n$ because every disk can be deleted at most once.

运行时间。还需要确定我们的波前计算算法的成本上限。步骤1显然总共需要$O\left( \left| {S}_{\text{ball }}\right| \right)$的时间。步骤2需要$\mathop{\sum }\limits_{{i = 1}}^{n}O\left( {1 + {x}_{i}}\right)$的时间，其中${x}_{i}$是在处理圆盘${s}_{i}$时在步骤3.3中删除的圆盘数量。注意到$\mathop{\sum }\limits_{{i = 1}}^{n}{x}_{i} \leq  n$（因为每个圆盘最多只能被删除一次），该求和结果为$O\left( \left| {S}_{\text{ball }}\right| \right)$。

### A.2 Solving the USEC Problem

### A.2 解决USEC问题

Recall that the USEC instance has a yes answer if and only if a point of ${S}_{pt}$ is on or below the wavefront. Proposition A. 4 suggests a simple plane-sweep strategy to determine the answer. Specifically, imagine sweeping a vertical line from left to right, and at any moment, remember the (only) arc of the wavefront intersecting the sweeping line. Whenever a point $p \in  {S}_{pt}$ is swept by the line, check whether it falls below the arc mentioned earlier. Because (i) the arcs of the wavefront have been listed from left to right,and (ii) the points of ${S}_{pt}$ have been sorted on x-dimension,the plane sweep can be easily implemented in $O\left( {\left| {S}_{\text{ball }}\right|  + \left| {S}_{pt}\right| }\right)$ time,by scanning the arcs in the wavefront and the points of ${S}_{pt}$ synchronously in ascending order of x-coordinate. This concludes the proof of Lemma 5.5, and also that of Theorem 5.2.

回顾一下，当且仅当${S}_{pt}$中的一个点位于波前上或波前下方时，USEC（通用集覆盖，Universal Set Cover）实例的答案为“是”。命题A. 4提出了一种简单的平面扫描策略来确定答案。具体来说，想象从左到右扫描一条垂直线，并且在任何时刻，记住与扫描线相交的（唯一）波前弧。每当一个点$p \in  {S}_{pt}$被这条线扫过时，检查它是否落在前面提到的弧下方。由于（i）波前的弧已经从左到右列出，并且（ii）${S}_{pt}$中的点已经按x维度排序，通过同步地按x坐标升序扫描波前的弧和${S}_{pt}$中的点，平面扫描可以在$O\left( {\left| {S}_{\text{ball }}\right|  + \left| {S}_{pt}\right| }\right)$时间内轻松实现。这就完成了引理5.5的证明，同时也完成了定理5.2的证明。

## REFERENCES

## 参考文献

Pankaj K. Agarwal, Herbert Edelsbrunner, and Otfried Schwarzkopf. 1991. Euclidean minimum spanning trees and bichromatic closest pairs. Discrete & Computational Geometry 6 (1991), 407-422.

Arne Andersson, Torben Hagerup, Stefan Nilsson, and Rajeev Raman. 1998. Sorting in linear time?Journal of Computer and System Sciences (JCSS) 57, 1 (1998), 74-93.

Mihael Ankerst, Markus M. Breunig, Hans-Peter Kriegel, and Jörg Sander. 1999. OPTICS: Ordering points to identify the clustering structure. In Proceedings of ACM Management of Data (SIGMOD). 49-60.

Sunil Arya and David M. Mount. 2000. Approximate range searching. Computational Geometry 17, 3-4 (2000), 135-152. Sunil Arya and David M. Mount. 2016. A fast and simple algorithm for computing approximate Euclidean minimum spanning trees. In Proceedings of the Annual ACM-SIAM Symposium on Discrete Algorithms (SODA). 1220-1233.

K. Bache and M. Lichman. 2013. UCI Machine Learning Repository. Retrieved from http://archive.ics.uci.edu/ml.

Christian Böhm, Karin Kailing, Peer Kröger, and Arthur Zimek. 2004. Computing clusters of correlation connected objects. In Proceedings of ACM Management of Data (SIGMOD). 455-466.

B. Borah and D. K. Bhattacharyya. 2004. An improved sampling-based DBSCAN for large spatial databases. In Proceedings of Intelligent Sensing and Information Processing. 92-96.

Prosenjit Bose, Anil Maheshwari, Pat Morin, Jason Morrison, Michiel H. M. Smid, and Jan Vahrenhold. 2007. Space-efficient geometric divide-and-conquer algorithms. Computational Geometry 37, 3 (2007), 209-227.

Vineet Chaoji, Mohammad Al Hasan, Saeed Salem, and Mohammed J. Zaki. 2008. SPARCL: Efficient and effective shape-based clustering. In Proceedings of International Conference on Management of Data (ICDM). 93-102.

Mark de Berg, Otfried Cheong, Marc van Kreveld, and Mark Overmars. 2008. Computational Geometry: Algorithms and Applications (3rd ed.). Springer-Verlag.

Mark de Berg, Constantinos Tsirogiannis, and B. T. Wilkinson. 2015. Fast computation of categorical richness on raster data sets and related problems. 18:1-18:10.

Jeff Erickson. 1995. On the relative complexities of some geometric problems. In Proceedings of the Canadian Conference on Computational Geometry (CCCG). 85-90.

Jeff Erickson. 1996. New lower bounds for Hopcroft's problem. Discrete & Computational Geometry 16, 4 (1996), 389-418. Martin Ester. 2013. Density-based clustering. In Data Clustering: Algorithms and Applications. 111-126.

Martin Ester, Hans-Peter Kriegel, Jörg Sander, and Xiaowei Xu. 1996. A density-based algorithm for discovering clusters

in large spatial databases with noise. In Proceedings of ACM Knowledge Discovery and Data Mining (SIGKDD). 226-231. Junhao Gan and Yufei Tao. 2015. DBSCAN revisited: Mis-claim, un-fixability, and approximation. In Proceedings of ACM Management of Data (SIGMOD). 519-530.

Ade Gunawan. 2013. A Faster Algorithm for DBSCAN. Master's thesis. Technische University Eindhoven.

Jiawei Han, Micheline Kamber, and Jian Pei. 2012. Data Mining: Concepts and Techniques. Morgan Kaufmann.

Yijie Han and Mikkel Thorup. 2002. Integer sorting in $0\left( {\mathrm{n}\text{sqrt (log}\log \mathrm{n}}\right)$ ) expected time and linear space. In Proceedings of Annual IEEE Symposium on Foundations of Computer Science (FOCS). 135-144.

G. R. Hjaltason and H. Samet. 1999. Distance browsing in spatial databases. ACM Transactions on Database Systems (TODS) 24, 2 (1999), 265-318.

David G. Kirkpatrick and Stefan Reisch. 1984. Upper bounds for sorting integers on random access machines. Theoretical Computer Science 28 (1984), 263-276.

Matthias Klusch, Stefano Lodi, and Gianluca Moro. 2003. Distributed clustering based on sampling local density estimates. In Proceedings of the International Joint Conference of Artificial Intelligence (IJCAI). 485-490.

Zhenhui Li, Bolin Ding, Jiawei Han, and Roland Kays. 2010. Swarm: Mining relaxed temporal moving object clusters. Proceedings of the VLDB Endowment (PVLDB) 3, 1 (2010), 723-734.

Bing Liu. 2006. A fast density-based clustering algorithm for large databases. In Proceedings of International Conference on Machine Learning and Cybernetics. 996-1000.

Eric Hsueh-Chan Lu, Vincent S. Tseng, and Philip S. Yu. 2011. Mining cluster-based temporal mobile sequential patterns in location-based service environments. IEEE Transactions on Knowledge and Data Engineering (TKDE) 23, 6 (2011), 914-927.

Shaaban Mahran and Khaled Mahar. 2008. Using grid for accelerating density-based clustering. In Proceedings of IEEE International Conference on Computer and Information Technology (CIT). 35-40.

Jiri Matousek. 1993. Range searching with efficient hiearchical cutting. Discrete & Computational Geometry 10 (1993), 157- 182.

Boriana L. Milenova and Marcos M. Campos. 2002. O-Cluster: Scalable clustering of large high dimensional data sets. In Proceedings of International Conference on Management of Data (ICDM). 290-297.

Davoud Moulavi, Pablo A. Jaskowiak, Ricardo J. G. B. Campello, Arthur Zimek, and Jörg Sander. 2014. Density-based clustering validation. In International Conference on Data Mining. 839-847.

Md. Mostofa Ali Patwary, Diana Palsetia, Ankit Agrawal, Wei-keng Liao, Fredrik Manne, and Alok N. Choudhary. 2012. A new scalable parallel DBSCAN algorithm using the disjoint-set data structure. In Conference on High Performance Computing Networking, Storage and Analysis. 62.

Tao Pei, A-Xing Zhu, Chenghu Zhou, Baolin Li, and Chengzhi Qin. 2006. A new approach to the nearest-neighbour method to discover cluster features in overlaid spatial point processes. International Journal of Geographical Information Science 20,2 (2006), 153-168.

Attila Reiss and Didier Stricker. 2012. Introducing a new benchmarked dataset for activity monitoring. In International Symposium on Wearable Computers. 108-109.

S. Roy and D. K. Bhattacharyya. 2005. An approach to find embedded clusters using density based techniques. In Proceedings of Distributed Computing and Internet Technology. 523-535.

Gholamhosein Sheikholeslami, Surojit Chatterjee, and Aidong Zhang. 2000. WaveCluster: A wavelet based clustering approach for spatial data in very large databases. The VLDB Journal 8, 3-4 (2000), 289-304.

Pang-Ning Tan, Michael Steinbach, and Vipin Kumar. 2006. Introduction to Data Mining. Pearson.

Robert Endre Tarjan. 1979. A class of algorithms which require nonlinear time to maintain disjoint sets. Journal of Computer and System Sciences (JCSS) 18, 2 (1979), 110-127.

Cheng-Fa Tsai and Chien-Tsung Wu. 2009. GF-DBSCAN: A new efficient and effective data clustering technique for large databases. In Proceedings of International Conference on Multimedia Systems and Signal Processing. 231-236.

Manik Varma and Andrew Zisserman. 2003. Texture classification: Are filter banks necessary?. In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 691-698.

Wei Wang, Jiong Yang, and Richard R. Muntz. 1997. STING: A statistical information grid approach to spatial data mining. In Proceedings of Very Large Data Bases (VLDB). 186-195.

Ji-Rong Wen, Jian-Yun Nie, and HongJiang Zhang. 2002. Query clustering using user logs. ACM Transactions on Information Systems (TOIS) 20, 1 (2002), 59-81.