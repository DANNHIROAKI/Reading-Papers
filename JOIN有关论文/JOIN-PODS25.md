## Optimal (Multiway) Spatial Joins*

## 最优（多路）空间连接*

RU WANG and YUFEI TAO, The Chinese University of Hong Kong, China

王儒和陶宇飞，中国香港中文大学

In a spatial join,we are given a constant number $k \geq  2$ of sets - denoted as ${R}_{1},{R}_{2},\ldots ,{R}_{k}$ - containing axis-parallel rectangles in a $2\mathrm{D}$ space. The objective is to report all $k$ -tuples $\left( {{r}_{1},{r}_{2},\ldots ,{r}_{k}}\right)  \in  {R}_{1} \times  {R}_{2} \times  \ldots  \times  {R}_{k}$ where the rectangles ${r}_{1},{r}_{2},\ldots ,{r}_{k}$ have a non-empty intersection,i.e., ${r}_{1} \cap  {r}_{2} \cap  \ldots  \cap  {r}_{k} \neq  \varnothing$ . The problem holds significant importance in spatial databases and has been extensively studied in the database community. In this paper,we show how to settle the problem in $O\left( {n\log n + \mathrm{{OUT}}}\right)$ time - regardless of the constant $k -$ where $n = \mathop{\sum }\limits_{{i = 1}}^{k}\left| {R}_{i}\right|$ and OUT is the result size (i.e.,the total number of $k$ -tuples reported). The runtime is asymptotically optimal in the class of comparison-based algorithms, to which our solution belongs. Previously, the state of the art was an algorithm with running time $O\left( {n{\log }^{{2k} - 1}n + \mathrm{{OUT}}}\right)$ .

在空间连接中，我们给定固定数量 $k \geq  2$ 的集合（记为 ${R}_{1},{R}_{2},\ldots ,{R}_{k}$ ），这些集合包含 $2\mathrm{D}$ 空间中的轴平行矩形。目标是报告所有 $k$ 元组 $\left( {{r}_{1},{r}_{2},\ldots ,{r}_{k}}\right)  \in  {R}_{1} \times  {R}_{2} \times  \ldots  \times  {R}_{k}$ ，其中矩形 ${r}_{1},{r}_{2},\ldots ,{r}_{k}$ 有非空交集，即 ${r}_{1} \cap  {r}_{2} \cap  \ldots  \cap  {r}_{k} \neq  \varnothing$ 。该问题在空间数据库中具有重要意义，并且在数据库领域得到了广泛研究。在本文中，我们展示了如何在 $O\left( {n\log n + \mathrm{{OUT}}}\right)$ 时间内解决该问题，而与常数 $k -$ 无关，其中 $n = \mathop{\sum }\limits_{{i = 1}}^{k}\left| {R}_{i}\right|$ 且 OUT 是结果大小（即报告的 $k$ 元组的总数）。我们的解决方案属于基于比较的算法类别，该运行时间在该类别算法中是渐进最优的。此前，最先进的算法运行时间为 $O\left( {n{\log }^{{2k} - 1}n + \mathrm{{OUT}}}\right)$ 。

CCS Concepts: - Theory of computation $\rightarrow$ Design and analysis of algorithms.

计算机协会概念分类： - 计算理论 $\rightarrow$ 算法设计与分析。

Additional Key Words and Phrases: Multiway Spatial Joins; Computational Geometry; Theory

其他关键词和短语：多路空间连接；计算几何；理论

## ACM Reference Format:

## 美国计算机协会引用格式：

Ru Wang and Yufei Tao. 2024. Optimal (Multiway) Spatial Joins. Proc. ACM Manag. Data 2, 5 (PODS), Article 210 (November 2024), 25 pages. https://doi.org/10.1145/3695828

王儒和陶宇飞。2024 年。最优（多路）空间连接。《美国计算机协会数据管理会议论文集》2, 5（数据库系统原理研讨会），文章编号 210（2024 年 11 月），25 页。https://doi.org/10.1145/3695828

## 1 Introduction

## 1 引言

This paper studies the spatial join (SJ) problem formulated as follows. Let $k \geq  2$ be a constant integer. In the $k$ - $S\mathcal{J}$ problem,the input comprises $k$ sets - denoted as ${R}_{1},{R}_{2},\ldots ,{R}_{k}$ - of axis-parallel rectangles ${}^{1}$ in ${\mathbb{R}}^{2}$ . The goal is to find all $k$ -tuples $\left( {{r}_{1},{r}_{2},\ldots ,{r}_{k}}\right)$ where

本文研究如下表述的空间连接（SJ）问题。设 $k \geq  2$ 为一个固定整数。在 $k$ - $S\mathcal{J}$ 问题中，输入包含 $k$ 个集合（记为 ${R}_{1},{R}_{2},\ldots ,{R}_{k}$ ），这些集合是 ${\mathbb{R}}^{2}$ 中的轴平行矩形 ${}^{1}$ 。目标是找到所有 $k$ 元组 $\left( {{r}_{1},{r}_{2},\ldots ,{r}_{k}}\right)$ ，其中

- ${r}_{i} \in  {R}_{i}$ for each $i \in  \left\lbrack  {1,k}\right\rbrack$ ; and

- 对于每个 $i \in  \left\lbrack  {1,k}\right\rbrack$ ，有 ${r}_{i} \in  {R}_{i}$ ；并且

- ${r}_{1} \cap  {r}_{2} \cap  \ldots  \cap  {r}_{k} \neq  \varnothing$ ,namely,the $k$ rectangles ${r}_{1},{r}_{2},\ldots ,{r}_{k}$ have a non-empty intersection.

- ${r}_{1} \cap  {r}_{2} \cap  \ldots  \cap  {r}_{k} \neq  \varnothing$ ，即 $k$ 个矩形 ${r}_{1},{r}_{2},\ldots ,{r}_{k}$ 有非空交集。

We represent the set of $k$ -tuples described above as $\mathcal{J}\left( {{R}_{1},{R}_{2},\ldots ,{R}_{k}}\right)$ ,referred to as the join result. Set $n = \mathop{\sum }\limits_{{i = 1}}^{k}\left| {R}_{i}\right|$ ,i.e.,the input size,and OUT $= \left| {\mathcal{J}\left( {{R}_{1},{R}_{2},\ldots ,{R}_{k}}\right) }\right|$ ,i.e.,the output size.

我们将上述 $k$ 元组的集合表示为 $\mathcal{J}\left( {{R}_{1},{R}_{2},\ldots ,{R}_{k}}\right)$ ，称为连接结果。设 $n = \mathop{\sum }\limits_{{i = 1}}^{k}\left| {R}_{i}\right|$ ，即输入大小，以及 OUT $= \left| {\mathcal{J}\left( {{R}_{1},{R}_{2},\ldots ,{R}_{k}}\right) }\right|$ ，即输出大小。

SJ is a fundamental operation in spatial databases (SDB), which manage geometric entities such as land parcels, service areas, habitat zones, commercial districts, administrative boundaries, etc. The operation plays a crucial role in implementing the filter-refinement mechanism, which is the dominant approach for computing overlay information in an SDB. To explain this mechanism, first note that a geometric entity is typically modeled as a polygon. Determining whether two entities overlap amounts to deciding if two polygons intersect, which can be exceedingly expensive when the polygons have complex boundaries. To mitigate the issue,an SDB stores,for each polygon $\gamma$ ,its minimum bounding rectangle (MBR) defined as the smallest axis-parallel rectangle enclosing $\gamma$ ; this way,each set $\Gamma$ of geometric entities spawns a set $R$ of MBRs. Consider $k$ sets of geometric entities ${\Gamma }_{1},{\Gamma }_{2},\ldots ,{\Gamma }_{k}$ ,and the corresponding sets of MBRs ${R}_{1},{R}_{2},\ldots ,{R}_{k}$ . To compute overlays from ${\Gamma }_{1},{\Gamma }_{2},\ldots ,{\Gamma }_{k}$ ,

空间连接（SJ，Spatial Join）是空间数据库（SDB，Spatial Database）中的一项基本操作，空间数据库用于管理诸如地块、服务区、栖息地、商业区、行政边界等几何实体。该操作在实现过滤 - 细化机制方面起着至关重要的作用，而过滤 - 细化机制是计算空间数据库中叠加信息的主要方法。为了解释这一机制，首先要注意的是，几何实体通常被建模为多边形。确定两个实体是否重叠等同于判断两个多边形是否相交，当多边形的边界复杂时，这一计算成本可能极高。为缓解这一问题，空间数据库会为每个多边形 $\gamma$ 存储其最小边界矩形（MBR，Minimum Bounding Rectangle），即包围 $\gamma$ 的最小轴对齐矩形；这样，每组几何实体 $\Gamma$ 都会生成一组最小边界矩形 $R$。考虑 $k$ 组几何实体 ${\Gamma }_{1},{\Gamma }_{2},\ldots ,{\Gamma }_{k}$ 以及对应的最小边界矩形组 ${R}_{1},{R}_{2},\ldots ,{R}_{k}$。为了从 ${\Gamma }_{1},{\Gamma }_{2},\ldots ,{\Gamma }_{k}$ 中计算叠加信息

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

允许个人或课堂使用本作品的全部或部分内容制作数字或硬拷贝，无需付费，但前提是这些拷贝不得用于盈利或商业目的，并且拷贝必须带有此声明和首页的完整引用信息。必须尊重本作品中除作者之外其他人拥有的版权。允许进行带引用的摘要。如需以其他方式复制、重新发布、上传到服务器或分发给列表，则需要事先获得特定许可和/或支付费用。请向 permissions@acm.org 请求许可。

© 2024 Copyright held by the owner/author(s). Publication rights licensed to ACM.

© 2024 版权归所有者/作者所有。出版权授权给美国计算机协会（ACM，Association for Computing Machinery）。

ACM 2836-6573/2024/11-ART210

美国计算机协会 2836 - 6573/2024/11 - ART210

https://doi.org/10.1145/3695828 filter-refinement first executes (i) a "filter step",which performs an SJ to obtain $\mathcal{J}\left( {{R}_{1},{R}_{2},\ldots ,{R}_{k}}\right)$ , and (ii) a "refinement step",which,for each $\left( {{r}_{1},{r}_{2},\ldots ,{r}_{k}}\right)  \in  \mathcal{J}\left( {{R}_{1},{R}_{2},\ldots ,{R}_{k}}\right)$ ,examines if ${\gamma }_{1},{\gamma }_{2},\ldots ,{\gamma }_{k}$ indeed have a non-empty intersection,where ${\gamma }_{i}\left( {i \in  \left\lbrack  {1,k}\right\rbrack  }\right)$ is the entity in ${\Gamma }_{i}$ whose MBR is ${r}_{i}$ .

https://doi.org/10.1145/3695828 过滤 - 细化机制首先执行（i）“过滤步骤”，该步骤执行一次空间连接以获得 $\mathcal{J}\left( {{R}_{1},{R}_{2},\ldots ,{R}_{k}}\right)$，以及（ii）“细化步骤”，对于每个 $\left( {{r}_{1},{r}_{2},\ldots ,{r}_{k}}\right)  \in  \mathcal{J}\left( {{R}_{1},{R}_{2},\ldots ,{R}_{k}}\right)$，检查 ${\gamma }_{1},{\gamma }_{2},\ldots ,{\gamma }_{k}$ 是否确实存在非空交集，其中 ${\gamma }_{i}\left( {i \in  \left\lbrack  {1,k}\right\rbrack  }\right)$ 是 ${\Gamma }_{i}$ 中最小边界矩形为 ${r}_{i}$ 的实体。

---

<!-- Footnote -->

*This work was supported in part by GRF projects 14207820, 14203421, and 14222822 from HKRGC.

*本研究部分得到了香港研究资助局（HKRGC，Hong Kong Research Grants Council）的研资局研究基金（GRF，General Research Fund）项目 14207820、14203421 和 14222822 的支持。

${}^{1}$ A rectangle is axis-parallel if it has the form $r = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  \left\lbrack  {{y}_{1},{y}_{2}}\right\rbrack$ .

${}^{1}$ 如果一个矩形具有 $r = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  \left\lbrack  {{y}_{1},{y}_{2}}\right\rbrack$ 的形式，则称其为轴对齐矩形。

Authors' Contact Information: Ru Wang, rwang21@cse.cuhk.edu.hk; Yufei Tao, taoyf@cse.cuhk.edu.hk, The Chinese University of Hong Kong, Hong Kong, Shatin, China.

作者联系方式：王茹，rwang21@cse.cuhk.edu.hk；陶宇飞，taoyf@cse.cuhk.edu.hk，香港中文大学，中国香港沙田。

<!-- Footnote -->

---

Math Conventions. For any integer $x \geq  1$ ,we use $\left\lbrack  x\right\rbrack$ to represent the set $\{ 1,2,\ldots ,x\}$ . Given $k \geq  2$ sets ${S}_{1},{S}_{2},\ldots ,{S}_{k}$ (of arbitrary elements),we often treat a $k$ -tuple $\left( {{e}_{1},{e}_{2},\ldots ,{e}_{k}}\right)$ in the Cartesian product ${S}_{1} \times  {S}_{2} \times  \ldots  \times  {S}_{k}$ as a $k$ -dimensional vector $t$ with $t\left\lbrack  i\right\rbrack   = {e}_{i}$ for each $i \in  \left\lbrack  k\right\rbrack$ . Unless otherwise stated, every mention of the word "rectangle" henceforth will refer to an axis-parallel rectangle in ${\mathbb{R}}^{2}$ . All logarithms have base 2 by default.

数学约定。对于任意整数 $x \geq  1$ ，我们用 $\left\lbrack  x\right\rbrack$ 表示集合 $\{ 1,2,\ldots ,x\}$ 。给定 $k \geq  2$ 个集合 ${S}_{1},{S}_{2},\ldots ,{S}_{k}$ （元素任意），我们通常将笛卡尔积 ${S}_{1} \times  {S}_{2} \times  \ldots  \times  {S}_{k}$ 中的一个 $k$ -元组 $\left( {{e}_{1},{e}_{2},\ldots ,{e}_{k}}\right)$ 视为一个 $k$ 维向量 $t$ ，其中对于每个 $i \in  \left\lbrack  k\right\rbrack$ 都有 $t\left\lbrack  i\right\rbrack   = {e}_{i}$ 。除非另有说明，此后提到的“矩形”一词均指 ${\mathbb{R}}^{2}$ 中的轴平行矩形。默认情况下，所有对数的底数均为 2。

### 1.1 Previous Results

### 1.1 先前的研究成果

SJs have been extensively studied in the database-system community, leading to the development of numerous methods that, although lacking strong theoretical guarantees, exhibit good empirical performance in real-world applications. We refer interested readers to $\left\lbrack  {3,4,7,8,{10} - {15},{18},{19}}\right\rbrack$ as entry points into the literature.

半连接（Semi-joins，SJs）在数据库系统领域得到了广泛研究，催生了众多方法。尽管这些方法缺乏强有力的理论保证，但在实际应用中表现出了良好的实证性能。我们建议感兴趣的读者参考 $\left\lbrack  {3,4,7,8,{10} - {15},{18},{19}}\right\rbrack$ 作为该领域文献的切入点。

From the perspective of theory,SJs are best understood when $k = 2$ ,i.e.,the pairwise scenario, where it is folklore that the problem can be solved by a comparison-based algorithm in $O(n\log n +$ OUT) time (e.g., by planesweep [5]). However, the problem becomes much more challenging for $k \geq  3$ ,known as the multiway scenario. All the solutions developed before 2022 (see [7,13,14,18] and the references therein) suffer from a worst-case time complexity of $O\left( {n}^{k}\right)$ ,offering essentially no improvement over the naive method that enumerates the entire cartesian product ${R}_{1} \times  {R}_{2} \times  \ldots  \times  {R}_{k}$ .

从理论角度来看，当 $k = 2$ 时，即成对场景下，对半连接（SJs）的理解最为透彻。在这种情况下，众所周知，该问题可以通过基于比较的算法在 $O(n\log n +$ OUT) 时间内解决（例如，通过平面扫描算法 [5]）。然而，当 $k \geq  3$ 时，问题变得更具挑战性，这被称为多路场景。2022 年之前开发的所有解决方案（见 [7,13,14,18] 及其参考文献）的最坏情况时间复杂度为 $O\left( {n}^{k}\right)$ ，与枚举整个笛卡尔积 ${R}_{1} \times  {R}_{2} \times  \ldots  \times  {R}_{k}$ 的朴素方法相比，基本上没有改进。

Year 2022 witnessed two independent works [9,21] that,although not tackling $k$ -SJ directly, imply provably fast $k$ -SJ algorithms. Specifically,in [21],Tao and Yi studied several variants of "interval intersection joins" under updates. Most relevant to our context is the variant where the input includes,for each $i \in  \left\lbrack  k\right\rbrack$ ,a set ${I}_{i}$ of 1D intervals in $\mathbb{R}$ ,and the join result comprises all $k$ -tuples $\left( {{I}_{1},{I}_{2},\ldots ,{I}_{k}}\right)  \in  {\mathcal{I}}_{1} \times  {\mathcal{I}}_{2} \times  \ldots  \times  {\mathcal{I}}_{k}$ with $\mathop{\bigcap }\limits_{{i = 1}}^{k}{I}_{i} \neq  \varnothing$ . The objective is to design a data structure, which,given the insertion (resp.,deletion) of an interval in one of the $k$ sets,can identify all the newly-appearing (resp.,disappearing) $k$ -tuples in the join result in $O\left( {\left( {1 + \Delta }\right)  \cdot  \text{polylog}n}\right)$ time, where $n = \mathop{\sum }\limits_{{i = 1}}^{k}\left| {\mathcal{I}}_{i}\right|$ and $\Delta$ is the number of such $k$ -tuples. Tao and Yi [21] presented a structure of $O\left( {n\text{polylog }n}\right)$ space achieving the purpose. Combining their structure with planesweep,one can obtain an algorithm for solving the $k$ -SJ problem in $O\left( {\left( {n + \mathrm{{OUT}}}\right)  \cdot  \text{polylog}n}\right)$ time.

2022年出现了两篇独立的研究成果[9,21]，尽管它们没有直接处理$k$ - 半连接（$k$ -SJ）问题，但暗示了可证明的快速$k$ - 半连接算法。具体而言，在文献[21]中，Tao和Yi研究了更新操作下“区间交集连接”的几种变体。与我们的研究最相关的变体是：对于每个$i \in  \left\lbrack  k\right\rbrack$，输入包含一个在$\mathbb{R}$中的一维区间集合${I}_{i}$，连接结果包含所有满足$\mathop{\bigcap }\limits_{{i = 1}}^{k}{I}_{i} \neq  \varnothing$的$k$ - 元组$\left( {{I}_{1},{I}_{2},\ldots ,{I}_{k}}\right)  \in  {\mathcal{I}}_{1} \times  {\mathcal{I}}_{2} \times  \ldots  \times  {\mathcal{I}}_{k}$。目标是设计一种数据结构，当其中一个$k$集合中插入（或删除）一个区间时，该数据结构能够在$O\left( {\left( {1 + \Delta }\right)  \cdot  \text{polylog}n}\right)$时间内识别出连接结果中所有新出现（或消失）的$k$ - 元组，其中$n = \mathop{\sum }\limits_{{i = 1}}^{k}\left| {\mathcal{I}}_{i}\right|$且$\Delta$是此类$k$ - 元组的数量。Tao和Yi [21]提出了一种占用$O\left( {n\text{polylog }n}\right)$空间的数据结构来实现这一目标。将他们的数据结构与平面扫描算法相结合，可以得到一个在$O\left( {\left( {n + \mathrm{{OUT}}}\right)  \cdot  \text{polylog}n}\right)$时间内解决$k$ - 半连接问题的算法。

In [9], Khamis et al. investigated a type of joins that extends the conventional equi-join in two ways. First, each attribute value in a relation is an interval (rather than a real value); second, each equality predicate in an equi-join is replaced with a "non-empty intersection" predicate on the attributes involved. The $k$ -SJ problem can be converted to a join under the framework of [9] as defined next. For each $i \in  \left\lbrack  k\right\rbrack$ ,define ${R}_{i}$ as a relation over two attributes $X$ and $Y$ . For each tuple $t \in  {R}_{i}$ ,its values $t\left( X\right)$ and $t\left( Y\right)$ on the two attributes are both intervals (effectively defining a rectangle). The objective is to output all $k$ -tuples $\left( {{t}_{1},{t}_{2},\ldots ,{t}_{k}}\right)  \in  {R}_{1} \times  {R}_{2} \times  \ldots  \times  {R}_{k}$ satisfying $\mathop{\bigcap }\limits_{{i = 1}}^{k}{t}_{i}\left( X\right)  \neq  \varnothing$ and $\mathop{\bigcap }\limits_{{i = 1}}^{k}{t}_{i}\left( Y\right)  \neq  \varnothing$ . It is clear that there is one-one correspondence between the result of this join and that of k-SJ. Khamis et al. [9] developed an algorithm that can process the join in $O\left( {n{\log }^{{2k} - 1}n + \mathrm{{OUT}}}\right)$ time.

在文献[9]中，Khamis等人研究了一种以两种方式扩展传统等值连接的连接类型。首先，关系中的每个属性值是一个区间（而不是一个实数值）；其次，等值连接中的每个相等谓词被替换为所涉及属性上的“非空交集”谓词。$k$ - 半连接问题可以按照以下方式转换为文献[9]框架下的连接问题。对于每个$i \in  \left\lbrack  k\right\rbrack$，将${R}_{i}$定义为一个基于两个属性$X$和$Y$的关系。对于每个元组$t \in  {R}_{i}$，它在这两个属性上的值$t\left( X\right)$和$t\left( Y\right)$都是区间（实际上定义了一个矩形）。目标是输出所有满足$\mathop{\bigcap }\limits_{{i = 1}}^{k}{t}_{i}\left( X\right)  \neq  \varnothing$和$\mathop{\bigcap }\limits_{{i = 1}}^{k}{t}_{i}\left( Y\right)  \neq  \varnothing$的$k$ - 元组$\left( {{t}_{1},{t}_{2},\ldots ,{t}_{k}}\right)  \in  {R}_{1} \times  {R}_{2} \times  \ldots  \times  {R}_{k}$。显然，这种连接的结果与$k$ - 半连接（k - SJ）的结果之间存在一一对应关系。Khamis等人[9]开发了一种可以在$O\left( {n{\log }^{{2k} - 1}n + \mathrm{{OUT}}}\right)$时间内处理该连接的算法。

$\Omega \left( {n\log n}\right)$ is a lower bound on the runtime of any comparison-based $k$ -SJ algorithms even for $k = 2$ . This can be established via a reduction from the element distinctness problem; see [6].

即使对于$k = 2$，$\Omega \left( {n\log n}\right)$也是任何基于比较的$k$ - 半连接算法运行时间的下界。这可以通过从元素唯一性问题进行归约来证明；详见文献[6]。

### 1.2 Our Results

### 1.2 我们的研究成果

In this paper,we solve the $k$ -SJ problem with a comparison-based algorithm that runs in $O(n\log n +$ OUT) time regardless of the constant $k$ . The time complexity is asymptotically optimal in the class of comparison-based algorithms.

在本文中，我们使用一种基于比较的算法解决了$k$ -SJ问题，该算法的运行时间为$O(n\log n +$ OUT)，且与常数$k$无关。在基于比较的算法类别中，该时间复杂度是渐近最优的。

<!-- Media -->

<table><tr><td>$k$</td><td>method</td><td>runtime</td><td>remark</td></tr><tr><td>2</td><td>folklore</td><td>$O\left( {n\log n + \mathrm{{OUT}}}\right)$</td><td>optimal</td></tr><tr><td>≥ 3</td><td>before 2022</td><td>$O\left( {n}^{k}\right)$</td><td/></tr><tr><td>$\geq  3$</td><td>[21]</td><td>$O\left( {\left( {n + \mathrm{{OUT}}}\right)  \cdot  \operatorname{polylog}n}\right)$</td><td/></tr><tr><td>$\geq  3$</td><td>[9]</td><td>$O\left( {n{\log }^{{2k} - 1}n + \mathrm{{OUT}}}\right)$</td><td/></tr><tr><td>$\geq  3$</td><td>ours</td><td>$O\left( {n\log n + \mathrm{{OUT}}}\right)$</td><td>optimal</td></tr></table>

<table><tbody><tr><td>$k$</td><td>方法</td><td>运行时间</td><td>备注</td></tr><tr><td>2</td><td>民间传说</td><td>$O\left( {n\log n + \mathrm{{OUT}}}\right)$</td><td>最优的</td></tr><tr><td>≥ 3</td><td>2022年之前</td><td>$O\left( {n}^{k}\right)$</td><td></td></tr><tr><td>$\geq  3$</td><td>[21]</td><td>$O\left( {\left( {n + \mathrm{{OUT}}}\right)  \cdot  \operatorname{polylog}n}\right)$</td><td></td></tr><tr><td>$\geq  3$</td><td>[9]</td><td>$O\left( {n{\log }^{{2k} - 1}n + \mathrm{{OUT}}}\right)$</td><td></td></tr><tr><td>$\geq  3$</td><td>我们的</td><td>$O\left( {n\log n + \mathrm{{OUT}}}\right)$</td><td>最优的</td></tr></tbody></table>

Table 1. Result comparison on $k$ -SJ problem for a constant $k$

表1. 对于常数 $k$，在 $k$ -SJ问题上的结果比较

<!-- Media -->

Our primary technical contribution is the revelation of a new property on the problem's mathematical structure. Fix any $k \geq  3$ and an arbitrary algorithm $\mathcal{A}$ for the(k - 1)-SJ problem. Define function ${F}_{k - 1}\left( {n,\mathrm{{OUT}}}\right)$ to return the worst-case running time of $\mathcal{A}$ on any instance of the(k - 1)-SJ problem having input size at most $n$ and output size at most OUT. We will establish:

我们主要的技术贡献是揭示了该问题数学结构的一个新性质。固定任意 $k \geq  3$ 和针对(k - 1)-SJ问题的任意算法 $\mathcal{A}$。定义函数 ${F}_{k - 1}\left( {n,\mathrm{{OUT}}}\right)$ 以返回 $\mathcal{A}$ 在输入规模至多为 $n$ 且输出规模至多为OUT的(k - 1)-SJ问题的任何实例上的最坏情况运行时间。我们将证明：

THEOREM 1.1. Equipped with the algorithm $\mathcal{A}$ as described above,the $k$ -SJ problem with $k \geq  3$ can be solved in time

定理1.1. 配备如上所述的算法 $\mathcal{A}$，具有 $k \geq  3$ 的 $k$ -SJ问题可以在时间内求解

$$
O\left( {k}^{3}\right)  \cdot  \left( {{F}_{k - 1}\left( {n,\mathrm{{OUT}}}\right)  + n\log n + k \cdot  \mathrm{{OUT}}}\right)  \tag{1}
$$

where $n$ (resp.,OUT) is the input (resp.,output) size of the problem. Furthermore,if $\mathcal{A}$ is comparison-based, the obtained k-SJ algorithm is also comparison-based.

其中 $n$（分别地，OUT）是该问题的输入（分别地，输出）规模。此外，如果 $\mathcal{A}$ 是基于比较的，那么所得到的k - SJ算法也是基于比较的。

The theorem implies a recursive nature of $k$ -SJ. Indeed,we will see that an $k$ -SJ instance with input size $n$ and output size OUT can be converted to $O\left( {k}^{3}\right)$ instances of the(k - 1)-SJ problem - all having input size at most $n$ and output size at most OUT - plus an additional cost of $O\left( {k}^{3}\right)$ . $\left( {n\log n + k \cdot  \mathrm{{OUT}}}\right)$ . For 2-SJ,we can set $\mathcal{A}$ to the "folklore algorithm" mentioned in Section 1.1, which ensures ${F}_{2}\left( {n,\mathrm{{OUT}}}\right)  = O\left( {n\log n + \mathrm{{OUT}}}\right)$ . Combining this with (1) gives a recurrence that relates the time complexity of $k$ -SJ to that of(k - 1)-SJ. Solving the recurrence yields:

该定理暗示了 $k$ -SJ的递归性质。实际上，我们将看到，输入规模为 $n$ 且输出规模为OUT的 $k$ -SJ实例可以转换为 $O\left( {k}^{3}\right)$ 个(k - 1)-SJ问题的实例——所有实例的输入规模至多为 $n$ 且输出规模至多为OUT——再加上 $O\left( {k}^{3}\right)$ . $\left( {n\log n + k \cdot  \mathrm{{OUT}}}\right)$ 的额外成本。对于2 - SJ，我们可以将 $\mathcal{A}$ 设置为第1.1节中提到的“民间算法”，这确保了 ${F}_{2}\left( {n,\mathrm{{OUT}}}\right)  = O\left( {n\log n + \mathrm{{OUT}}}\right)$。将此与(1)相结合，得到一个将 $k$ -SJ的时间复杂度与(k - 1)-SJ的时间复杂度相关联的递推式。求解该递推式可得：

THEOREM 1.2. For $k \geq  3$ ,we can settle $k$ -SJ with a comparison-based algorithm in

定理1.2. 对于 $k \geq  3$，我们可以使用基于比较的算法在

$$
O\left( {{c}^{k} \cdot  {\left( k!\right) }^{3} \cdot  \left( {n\log n + k \cdot  \mathrm{{OUT}}}\right) }\right) 
$$

time,where $c > 1$ is a positive constant.

时间内解决 [latex1] -SJ问题，其中 $c > 1$ 是一个正常数。

When $k = O\left( 1\right)$ ,the time complexity becomes $O\left( {n\log n + \mathrm{{OUT}}}\right)$ ,as promised; the space consumption of our algorithm is $O\left( {n + \mathrm{{OUT}}}\right)$ . Now that Theorem 1.2 offers a satisfactory $k$ -SJ result for $k = O\left( 1\right)$ in 2D space,it is natural to wonder whether the constraint on dimensionality 2 is necessary. Interestingly,the answer is "yes" as far as $k \geq  3$ is concerned,subject to the absence of breakthroughs on a classical problem in graph theory. Specifically, if the 3D version of the 3-SJ problem (which we will formally define in Appendix E) could be solved in $O\left( {\left( {n + \mathrm{{OUT}}}\right)  \cdot  \text{polylog}n}\right)$ time,we would be able to detect the presence of a triangle (i.e.,3-clique) in a graph of $m$ edges in $O\left( {m\text{polylog}m}\right)$ time,which would make a remarkable breakthrough because the state of the art needs $O\left( {m}^{1.41}\right)$ time [2]. This reduction can be inferred from an argument in [9] used to prove a more generic result. We simplify the argument for 3D 3-SJ and present the full reduction in Appendix E.

当 $k = O\left( 1\right)$ 时，时间复杂度如预期变为 $O\left( {n\log n + \mathrm{{OUT}}}\right)$；我们算法的空间消耗为 $O\left( {n + \mathrm{{OUT}}}\right)$。既然定理1.2在二维空间中为 $k = O\left( 1\right)$ 提供了令人满意的 $k$ -SJ结果，自然会想知道二维的维度约束是否必要。有趣的是，就 $k \geq  3$ 而言，答案是“是”，前提是图论中的一个经典问题没有突破。具体来说，如果3 - SJ问题的三维版本（我们将在附录E中正式定义）可以在 $O\left( {\left( {n + \mathrm{{OUT}}}\right)  \cdot  \text{polylog}n}\right)$ 时间内求解，那么我们就能在 $O\left( {m\text{polylog}m}\right)$ 时间内检测出具有 $m$ 条边的图中是否存在三角形（即3 - 团），这将是一个显著的突破，因为目前的最优算法需要 $O\left( {m}^{1.41}\right)$ 时间 [2]。这个归约可以从 [9] 中用于证明更一般结果的一个论证中推导出来。我们为三维3 - SJ简化了该论证，并在附录E中给出完整的归约。

## 2 Preliminaries in Geometry

## 2 几何预备知识

This section will first introduce some definitions and notations to be frequently used in our presentation and then formulate several computational geometry problems, whose solutions will serve as building bricks for our $k$ -SJ algorithm.

本节将首先介绍一些在我们的阐述中会频繁使用的定义和符号，然后阐述几个计算几何问题，这些问题的解将作为我们 $k$ -SJ算法的构建模块。

Terminology. A horizontal segment is a segment of the form $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  y$ ,and a vertical segment is a segment of the form $x \times  \left\lbrack  {{y}_{1},{y}_{2}}\right\rbrack$ . We say that a horizontal segment ${h}_{1}$ is lower (resp.,higher)

术语。水平线段是形如 $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  y$ 的线段，垂直线段是形如 $x \times  \left\lbrack  {{y}_{1},{y}_{2}}\right\rbrack$ 的线段。我们称水平线段 ${h}_{1}$ 比另一条水平线段 [latex3] 更低（相应地，更高）

<!-- Media -->

<!-- figureText: ${r}_{2}$ ${r}_{1}$ ${r}_{4}$ -->

<img src="https://cdn.noedgeai.com/0195ccc5-d2d9-7daa-9177-3ae04293d71f_3.jpg?x=625&y=263&w=317&h=245&r=0"/>

<!-- Media -->

Fig. 1. For 4-tuple $t = \left\{  {{r}_{1},{r}_{2},{r}_{3},{r}_{4}}\right\}  ,{B}_{t}$ is the rectangle in gray,left-guard $\left( t\right)  = {r}_{3}$ and bot-guard $\left( t\right)  = {r}_{2}$ . than another horizontal segment ${h}_{2}$ if the y-coordinate of ${h}_{1}$ is smaller (resp.,larger) than that of ${h}_{2}$ . Similarly,a vertical segment ${v}_{1}$ is to the left (resp.,right) of another vertical segment ${v}_{2}$ if the $\mathrm{x}$ -coordinate of ${v}_{1}$ is smaller (resp.,larger) than that of ${v}_{2}$ .

图 1。对于四元组 $t = \left\{  {{r}_{1},{r}_{2},{r}_{3},{r}_{4}}\right\}  ,{B}_{t}$，灰色部分为矩形，左保护线为 $\left( t\right)  = {r}_{3}$，底保护线为 $\left( t\right)  = {r}_{2}$。如果 ${h}_{1}$ 的 y 坐标小于（相应地，大于）${h}_{2}$ 的 y 坐标。类似地，如果垂直线段 ${v}_{1}$ 的 $\mathrm{x}$ 坐标小于（相应地，大于）另一条垂直线段 ${v}_{2}$ 的 $\mathrm{x}$ 坐标，则称 ${v}_{1}$ 在 ${v}_{2}$ 的左侧（相应地，右侧）。

Given a horizontal segment $h = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  y$ ,we say that a rectangle $r$ is a left-end covering rectangle of $h$ if $r$ contains the left endpoint of $h$ (i.e., $\left( {{x}_{1},y}\right)  \in  r$ ). A horizontal/vertical segment $s$ crosses a rectangle $r$ if $s \cap  r \neq  \varnothing$ but $r$ covers neither of the two endpoints of $s$ . A rectangle $r$ contains a horizontal/vertical segment $s$ if $r$ covers both endpoints of $s$ .

给定一条水平线段 $h = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  y$，如果矩形 $r$ 包含 $h$ 的左端点（即 $\left( {{x}_{1},y}\right)  \in  r$），则称 $r$ 是 $h$ 的左端点覆盖矩形。如果水平/垂直线段 $s$ 满足 $s \cap  r \neq  \varnothing$，但 $r$ 不覆盖 $s$ 的两个端点，则称 $s$ 穿过矩形 $r$。如果矩形 $r$ 覆盖水平/垂直线段 $s$ 的两个端点，则称 $r$ 包含 $s$。

Let $S$ be a set of segments where either all segments are horizontal or all are vertical. Given a rectangle $r$ ,we define

设 $S$ 是一组线段，其中所有线段要么都是水平的，要么都是垂直的。给定一个矩形 $r$，我们定义

$$
{\operatorname{cross}}_{S}\left( r\right)  = \{ s \in  S \mid  s\text{ crosses }r\}  \tag{2}
$$

namely, ${\operatorname{cross}}_{S}\left( r\right)$ is the set of segments in $S$ crossing $r$ . Let $R$ be a set of rectangles. Given a horizontal segment $h$ ,we define

即，${\operatorname{cross}}_{S}\left( r\right)$ 是 $S$ 中穿过 $r$ 的线段的集合。设 $R$ 是一组矩形。给定一条水平线段 $h$，我们定义

$$
{\operatorname{contain}}_{R}\left( h\right)  = \{ r \in  R \mid  r\text{ contains }h\} ; \tag{3}
$$

namely, ${\operatorname{contain}}_{R}\left( h\right)$ is the set of rectangles in $R$ containing $h$ .

即，${\operatorname{contain}}_{R}\left( h\right)$ 是 $R$ 中包含 $h$ 的矩形的集合。

Given a rectangle $r = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  \left\lbrack  {{y}_{1},{y}_{2}}\right\rbrack$ ,we define left $\left( r\right)  = {x}_{1}$ ,right $\left( r\right)  = {x}_{2}$ ,bot $\left( r\right)  = {y}_{1}$ ,and $\operatorname{top}\left( r\right)  = {y}_{2}$ . Consider a $k$ -tuple $t = \left( {{r}_{1},{r}_{2},\ldots ,{r}_{k}}\right)$ where $k \geq  2$ and each $t\left\lbrack  i\right\rbrack   = {r}_{i}\left( {i \leq  \left\lbrack  k\right\rbrack  }\right)$ is a rectangle. We define

给定一个矩形 $r = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  \left\lbrack  {{y}_{1},{y}_{2}}\right\rbrack$，我们定义左 $\left( r\right)  = {x}_{1}$、右 $\left( r\right)  = {x}_{2}$、底 $\left( r\right)  = {y}_{1}$ 和 $\operatorname{top}\left( r\right)  = {y}_{2}$。考虑一个 $k$ 元组 $t = \left( {{r}_{1},{r}_{2},\ldots ,{r}_{k}}\right)$，其中 $k \geq  2$ 且每个 $t\left\lbrack  i\right\rbrack   = {r}_{i}\left( {i \leq  \left\lbrack  k\right\rbrack  }\right)$ 都是一个矩形。我们定义

$$
{B}_{t} = \mathop{\bigcap }\limits_{{i = 1}}^{k}{r}_{i} \tag{4}
$$

namely, ${B}_{t}$ is the intersection of the rectangles in $t$ (note: ${B}_{t}$ is a rectangle itself). Also,if ${B}_{t}$ is not empty, define:

即，${B}_{t}$ 是 $t$ 中矩形的交集（注意：${B}_{t}$ 本身也是一个矩形）。此外，如果 ${B}_{t}$ 不为空，则定义：

- left-guard(t)as the rectangle ${r}_{i},i \in  \left\lbrack  k\right\rbrack$ ,satisfying $\operatorname{left}\left( {r}_{i}\right)  = \operatorname{left}\left( {B}_{\mathbf{t}}\right)$ . In case multiple values in $\left\lbrack  k\right\rbrack$ fulfill the condition,let $i$ be the smallest of such values.

- 将左保护矩形（left-guard(t)）定义为矩形 ${r}_{i},i \in  \left\lbrack  k\right\rbrack$，满足 $\operatorname{left}\left( {r}_{i}\right)  = \operatorname{left}\left( {B}_{\mathbf{t}}\right)$。若 $\left\lbrack  k\right\rbrack$ 中有多个值满足该条件，则令 $i$ 为这些值中的最小值。

- bot-guard(t)as the rectangle ${r}_{i},i \in  \left\lbrack  k\right\rbrack$ ,satisfying $\operatorname{bot}\left( {r}_{i}\right)  = \operatorname{bot}\left( {B}_{t}\right)$ . In case multiple values in $\left\lbrack  k\right\rbrack$ fulfill the condition,let $i$ be the smallest of such values.

- 将下保护矩形（bot-guard(t)）定义为矩形 ${r}_{i},i \in  \left\lbrack  k\right\rbrack$，满足 $\operatorname{bot}\left( {r}_{i}\right)  = \operatorname{bot}\left( {B}_{t}\right)$。若 $\left\lbrack  k\right\rbrack$ 中有多个值满足该条件，则令 $i$ 为这些值中的最小值。

See Figure 1 for an illustration. It is worth mentioning that since a horizontal segment $h$ is a degenerated rectangle,notations such as left(h)and right(h)are well-defined.

具体示例见图 1。值得一提的是，由于水平线段 $h$ 是退化的矩形，因此诸如左端点（left(h)）和右端点（right(h)）等表示是明确定义的。

Problem $\mathcal{A}$ . The input involves a set $P$ of $2\mathrm{D}$ points and set $R$ of rectangles. In the detection version of Problem $\mathcal{A}$ ,the goal is to output,for each point $p \in  P$ ,whether it is covered by at least one rectangle in $R$ . Figure 2a gives an example where $P = \left\{  {{p}_{1},{p}_{2},{p}_{3}}\right\}$ and $R = \left\{  {{r}_{1},{r}_{2}}\right\}$ ; the output is "yes" for ${p}_{2}$ and ${p}_{3}$ and "no" for ${p}_{1}$ . The problem can be solved in $O\left( {n\log n}\right)$ time where $n = \left| P\right|  + \left| R\right|$ as shown in Appendix A.

问题 $\mathcal{A}$。输入包含一个由 $2\mathrm{D}$ 个点组成的集合 $P$ 和一个矩形集合 $R$。在问题 $\mathcal{A}$ 的检测版本中，目标是针对每个点 $p \in  P$，输出它是否至少被 $R$ 中的一个矩形覆盖。图 2a 给出了一个示例，其中 $P = \left\{  {{p}_{1},{p}_{2},{p}_{3}}\right\}$ 且 $R = \left\{  {{r}_{1},{r}_{2}}\right\}$；对于 ${p}_{2}$ 和 ${p}_{3}$，输出为“是”，对于 ${p}_{1}$，输出为“否”。如附录 A 所示，该问题可在 $O\left( {n\log n}\right)$ 时间内解决，其中 $n = \left| P\right|  + \left| R\right|$。

In the reporting version of Problem $\mathcal{A}$ ,the goal is to output,for each point $p \in  P$ ,all the rectangles $r \in  R$ containing $p$ ; if no such $r$ exists,report nothing for $p$ . In Figure 1a,for instance,the output is $\left\{  {\left( {{p}_{2} : {r}_{1},{r}_{2}}\right) ,\left( {{p}_{3} : {r}_{2}}\right) }\right\}$ . As shown in Appendix A,the problem can be solved in $O\left( {n\log n + \mathrm{{OUT}}}\right)$ time,where OUT is the number of pairs $\left( {p,r}\right)  \in  P \times  R$ such that $p \in  r$ .

在问题 $\mathcal{A}$ 的报告版本中，目标是针对每个点 $p \in  P$，输出包含 $p$ 的所有矩形 $r \in  R$；若不存在这样的 $r$，则不针对 $p$ 进行报告。例如，在图 1a 中，输出为 $\left\{  {\left( {{p}_{2} : {r}_{1},{r}_{2}}\right) ,\left( {{p}_{3} : {r}_{2}}\right) }\right\}$。如附录 A 所示，该问题可在 $O\left( {n\log n + \mathrm{{OUT}}}\right)$ 时间内解决，其中 OUT 是满足 $p \in  r$ 的对 $\left( {p,r}\right)  \in  P \times  R$ 的数量。

<!-- Media -->

<!-- figureText: ${p}_{1}$ ${h}_{1}$ ${p}_{2}$ ${h}_{3}$ (c) Problem C (d) Problem 9 (e) Problem & ${r}_{1}$ $\underline{{h}_{2}}$ ${p}_{2} \bullet$ ${p}_{3} \bullet$ (a) Problem $\mathcal{A}$ (b) Problem $\mathcal{B}$ -->

<img src="https://cdn.noedgeai.com/0195ccc5-d2d9-7daa-9177-3ae04293d71f_4.jpg?x=149&y=269&w=1269&h=310&r=0"/>

Fig. 2. Five geometric building brick problems

图 2. 五个几何构建块问题

<!-- Media -->

Problem $\mathcal{B}$ . The input involves a set $H$ of horizontal segments and a set $V$ of vertical segments. The goal is to report,for each segment $h \in  H$ ,the leftmost point $p$ on $h$ such that $p$ is on some vertical segment in $V$ . If $h$ does not intersect with any segment in $V$ ,report nothing for $h$ . Figure $2\mathrm{\;b}$ gives an example where $H = \left\{  {{h}_{1},{h}_{2},{h}_{3}}\right\}$ and $V = \left\{  {{v}_{1},{v}_{2}}\right\}$ ; the output is $\left\{  {\left( {{h}_{1},{p}_{1}}\right) ,\left( {{h}_{2},{p}_{2}}\right) }\right\}$ . The problem can be solved in $O\left( {n\log n}\right)$ time where $n = \left| H\right|  + \left| V\right|$ ,as shown in Appendix A.

问题 $\mathcal{B}$。输入包含一个水平线段集合 $H$ 和一个垂直线段集合 $V$。目标是针对每个线段 $h \in  H$，报告 $h$ 上最左边的点 $p$，使得 $p$ 位于 $V$ 中的某个垂直线段上。若 $h$ 与 $V$ 中的任何线段都不相交，则不针对 $h$ 进行报告。图 $2\mathrm{\;b}$ 给出了一个示例，其中 $H = \left\{  {{h}_{1},{h}_{2},{h}_{3}}\right\}$ 且 $V = \left\{  {{v}_{1},{v}_{2}}\right\}$；输出为 $\left\{  {\left( {{h}_{1},{p}_{1}}\right) ,\left( {{h}_{2},{p}_{2}}\right) }\right\}$。如附录 A 所示，该问题可在 $O\left( {n\log n}\right)$ 时间内解决，其中 $n = \left| H\right|  + \left| V\right|$。

Problem C. The input involves a set $H$ of horizontal segments and a set $R$ of rectangles. The goal is to report,for each segment $h \in  H$ ,the rightmost point $p$ on $h$ such that $p$ is covered by at least one left-end covering rectangle of $h$ in $R$ - formally,for $h = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  y$ ,we aim to find the maximum $x \in  \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$ such that at least one rectangle $r \in  R$ covers both the point $\left( {{x}_{1},y}\right)$ and the point(x,y). If the point $p$ exists (i.e., $h$ has at least one left-end covering rectangle in $R$ ),we should output a tuple(h,p); otherwise,output nothing for $h$ . Figure $2\mathrm{c}$ gives an example where $H = \left\{  {{h}_{1},{h}_{2},{h}_{3}}\right\}$ and $R$ includes the three rectangles shown; the output is $\left\{  {\left( {{h}_{1},{p}_{1}}\right) ,\left( {{h}_{2},{p}_{2}}\right) }\right\}$ . The problem can be solved in in $O\left( {n\log n}\right)$ time where $n = \left| H\right|  + \left| R\right|$ ,as shown in Appendix A.

问题C。输入包含一组水平线段$H$和一组矩形$R$。目标是为每条线段$h \in  H$报告$h$上最右侧的点$p$，使得$p$被$R$中$h$的至少一个左端点覆盖矩形所覆盖——形式上，对于$h = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  y$，我们的目标是找到最大的$x \in  \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$，使得至少有一个矩形$r \in  R$同时覆盖点$\left( {{x}_{1},y}\right)$和点(x,y)。如果点$p$存在（即$h$在$R$中至少有一个左端点覆盖矩形），我们应输出一个元组(h,p)；否则，不为$h$输出任何内容。图$2\mathrm{c}$给出了一个示例，其中$H = \left\{  {{h}_{1},{h}_{2},{h}_{3}}\right\}$且$R$包含所示的三个矩形；输出为$\left\{  {\left( {{h}_{1},{p}_{1}}\right) ,\left( {{h}_{2},{p}_{2}}\right) }\right\}$。如附录A所示，该问题可以在$O\left( {n\log n}\right)$时间内解决，其中$n = \left| H\right|  + \left| R\right|$。

Problem 2. The input involves a set $H$ of horizontal segments and a set $R$ of rectangles. In the find-lowest version of the problem,the goal is to report,for each rectangle $r \in  R$ ,the lowest segment in ${\operatorname{cross}}_{H}\left( r\right)$ ; see (2) for the definition of ${\operatorname{cross}}_{H}\left( r\right)$ . If no segment in $H$ crosses $r$ ,output nothing for $r$ . Figure 2d gives an example where $H = \left\{  {{h}_{1},{h}_{2},{h}_{3}}\right\}$ and $R = \left\{  {{r}_{1},{r}_{2}}\right\}$ ; the output is $\left\{  {\left( {{r}_{1},{h}_{3}}\right) ,\left( {{r}_{2},{h}_{2}}\right) }\right\}$ . The problem can be solved in $O\left( {n\log n}\right)$ time where $n = \left| H\right|  + \left| R\right|$ ,as shown in Appendix A.

问题2。输入包含一组水平线段$H$和一组矩形$R$。在该问题的“查找最低线段”版本中，目标是为每个矩形$r \in  R$报告${\operatorname{cross}}_{H}\left( r\right)$中最低的线段；关于${\operatorname{cross}}_{H}\left( r\right)$的定义见(2)。如果$H$中没有线段与$r$相交，则不为$r$输出任何内容。图2d给出了一个示例，其中$H = \left\{  {{h}_{1},{h}_{2},{h}_{3}}\right\}$且$R = \left\{  {{r}_{1},{r}_{2}}\right\}$；输出为$\left\{  {\left( {{r}_{1},{h}_{3}}\right) ,\left( {{r}_{2},{h}_{2}}\right) }\right\}$。如附录A所示，该问题可以在$O\left( {n\log n}\right)$时间内解决，其中$n = \left| H\right|  + \left| R\right|$。

In the find-all-sorted version of the problem,the goal is to report,for each rectangle $r \in  R$ ,the entire ${\operatorname{cross}}_{H}\left( r\right)$ sorted by y-coordinate. Formally,if ${\operatorname{cross}}_{H}\left( r\right)  = \left\{  {{h}_{1},{h}_{2},\ldots ,{h}_{z}}\right\}$ for some $z \geq  1$ , we output $\left( {r : {h}_{1},{h}_{2},\ldots ,{h}_{z}}\right)$ ,provided that ${y}_{i} \geq  {y}_{i - 1}$ for each $i \in  \left\lbrack  {2,z}\right\rbrack$ where ${y}_{i}$ (resp., ${y}_{i - 1}$ ) is the y-coordinate of ${h}_{i}$ (resp., ${h}_{i - 1}$ ). In the example of Figure 2d,the output is $\left\{  {\left( {{r}_{1} : {h}_{3},{h}_{2}}\right) ,\left( {{r}_{2} : {h}_{2},{h}_{1}}\right) }\right\}$ . In Appendix A,we explain how to solve the problem in $O\left( {n\log n + \mathrm{{OUT}}}\right)$ time where $\mathrm{{OUT}}$ is the number of pairs $\left( {h,r}\right)  \in  H \times  R$ such that $h$ crosses $r$ .

在该问题的全查找排序版本中，目标是针对每个矩形 $r \in  R$，报告按 y 坐标排序的整个 ${\operatorname{cross}}_{H}\left( r\right)$。形式上，如果对于某个 $z \geq  1$ 有 ${\operatorname{cross}}_{H}\left( r\right)  = \left\{  {{h}_{1},{h}_{2},\ldots ,{h}_{z}}\right\}$，我们输出 $\left( {r : {h}_{1},{h}_{2},\ldots ,{h}_{z}}\right)$，前提是对于每个 $i \in  \left\lbrack  {2,z}\right\rbrack$ 有 ${y}_{i} \geq  {y}_{i - 1}$，其中 ${y}_{i}$（分别地，${y}_{i - 1}$）是 ${h}_{i}$（分别地，${h}_{i - 1}$）的 y 坐标。在图 2d 的示例中，输出为 $\left\{  {\left( {{r}_{1} : {h}_{3},{h}_{2}}\right) ,\left( {{r}_{2} : {h}_{2},{h}_{1}}\right) }\right\}$。在附录 A 中，我们将解释如何在 $O\left( {n\log n + \mathrm{{OUT}}}\right)$ 时间内解决该问题，其中 $\mathrm{{OUT}}$ 是满足 $h$ 与 $r$ 相交的对 $\left( {h,r}\right)  \in  H \times  R$ 的数量。

Problem $\mathcal{E}$ . The input involves a set $H$ of horizontal segments and a set $R$ of rectangles. The goal is to report,for each segment $h \in  H$ ,the set ${\operatorname{contain}}_{R}\left( h\right)  -$ defined in (3) - where the rectangles are sorted by their right boundaries; if ${\operatorname{contain}}_{R}\left( h\right)$ is empty,output nothing for $h$ . Formally,if ${r}_{1},{r}_{2},\ldots ,{r}_{z}$ for some $z \geq  1$ are all the rectangles in ${\operatorname{contain}}_{R}\left( h\right)$ ,we output $\left( {h : {r}_{1},{r}_{2},\ldots ,{r}_{z}}\right)$ ,provided that right $\left( {r}_{i}\right)  \geq  \operatorname{right}\left( {r}_{i - 1}\right)$ for each $i \in  \left\lbrack  {2,z}\right\rbrack$ . Figure 2e gives an example where $H = \left\{  {{h}_{1},{h}_{2},{h}_{3}}\right\}$ and $R = \left\{  {{r}_{1},{r}_{2},{r}_{3}}\right\}$ ; the output is $\left\{  {\left( {{h}_{1} : {r}_{2},{r}_{1}}\right) ,\left( {{h}_{2} : {r}_{3},{r}_{2}}\right) }\right\}$ . In Appendix A,we explain how to solve the problem in $O\left( {n\log n + \mathrm{{OUT}}}\right)$ time where $n = \left| H\right|  + \left| R\right|$ and OUT is the number of pairs $\left( {h,r}\right)  \in  H \times  R$ such that $r$ contains $h$ .

问题 $\mathcal{E}$。输入包含一组水平线段 $H$ 和一组矩形 $R$。目标是针对每条线段 $h \in  H$，报告在 (3) 中定义的集合 ${\operatorname{contain}}_{R}\left( h\right)  -$ —— 其中矩形按其右边界排序；如果 ${\operatorname{contain}}_{R}\left( h\right)$ 为空，则不输出 $h$ 的任何信息。形式上，如果对于某个 $z \geq  1$，${r}_{1},{r}_{2},\ldots ,{r}_{z}$ 是 ${\operatorname{contain}}_{R}\left( h\right)$ 中的所有矩形，我们输出 $\left( {h : {r}_{1},{r}_{2},\ldots ,{r}_{z}}\right)$，前提是对于每个 $i \in  \left\lbrack  {2,z}\right\rbrack$ 有右 $\left( {r}_{i}\right)  \geq  \operatorname{right}\left( {r}_{i - 1}\right)$。图 2e 给出了一个示例，其中 $H = \left\{  {{h}_{1},{h}_{2},{h}_{3}}\right\}$ 且 $R = \left\{  {{r}_{1},{r}_{2},{r}_{3}}\right\}$；输出为 $\left\{  {\left( {{h}_{1} : {r}_{2},{r}_{1}}\right) ,\left( {{h}_{2} : {r}_{3},{r}_{2}}\right) }\right\}$。在附录 A 中，我们将解释如何在 $O\left( {n\log n + \mathrm{{OUT}}}\right)$ 时间内解决该问题，其中 $n = \left| H\right|  + \left| R\right|$ 且 OUT 是满足 $r$ 包含 $h$ 的对 $\left( {h,r}\right)  \in  H \times  R$ 的数量。

## 3 The Core: H-V Multiway Spatial Joins

## 3 核心：水平 - 垂直多路空间连接

Recall that the input of $k$ -SJ comprises $k$ sets of rectangles: ${R}_{1},{R}_{2},\ldots ,{R}_{k}$ . We now formulate a special version of $k$ -SJ,named the $H$ -V $k$ -SJ problem. The special nature is reflected in the introduction of three constraints: (i) $k \geq  3$ ,(ii) ${R}_{k - 1}$ should be a set of horizontal segments,and (iii) ${R}_{k}$ should be a set of vertical segments. For better clarity,we will represent the input sets as ${R}_{1},{R}_{2},\ldots ,{R}_{k - 2},H$ $\left( { = {R}_{k - 1}}\right)$ ,and $V\left( { = {R}_{k}}\right)$ . The goal is to output the join result $\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 2},H,V}\right)$ ,including every $k$ -tuple $\left( {{r}_{1},\ldots ,{r}_{k - 2},h,v}\right)  \in  {R}_{1} \times  \ldots  \times  {R}_{k - 2} \times  H \times  V$ such that $h \cap  v \cap  \mathop{\bigcap }\limits_{{i = 1}}^{{k - 2}}{r}_{i}$ is not empty.

回顾一下，$k$ -SJ的输入包含$k$个矩形集合：${R}_{1},{R}_{2},\ldots ,{R}_{k}$。我们现在来阐述$k$ -SJ的一个特殊版本，称为$H$ -V $k$ -SJ问题。其特殊性体现在引入了三个约束条件：（i）$k \geq  3$；（ii）${R}_{k - 1}$应为一组水平线段；（iii）${R}_{k}$应为一组垂直线段。为了更清晰起见，我们将输入集合表示为${R}_{1},{R}_{2},\ldots ,{R}_{k - 2},H$ $\left( { = {R}_{k - 1}}\right)$和$V\left( { = {R}_{k}}\right)$。目标是输出连接结果$\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 2},H,V}\right)$，其中包括每个满足$h \cap  v \cap  \mathop{\bigcap }\limits_{{i = 1}}^{{k - 2}}{r}_{i}$不为空的$k$ -元组$\left( {{r}_{1},\ldots ,{r}_{k - 2},h,v}\right)  \in  {R}_{1} \times  \ldots  \times  {R}_{k - 2} \times  H \times  V$。

<!-- Media -->

<!-- figureText: ${r}_{1}$ ${r}_{2}$ (b) Type 2 (a) Type 1 -->

<img src="https://cdn.noedgeai.com/0195ccc5-d2d9-7daa-9177-3ae04293d71f_5.jpg?x=532&y=264&w=506&h=312&r=0"/>

Fig. 3. Classifying H-V $k$ -SJ result tuples $\left( {k = 4}\right)$

图3. 对H - V $k$ -SJ结果元组$\left( {k = 4}\right)$进行分类

<!-- Media -->

Our objective is to prove that H-V $k$ -SJ can be efficiently reduced to(k - 1)-SJ (note: it is(k - 1)-SJ here,rather than $\mathrm{H} - \mathrm{V}\left( {k - 1}\right)  - \mathrm{{SJ}}$ ). To ensure the soundness of our notation system,let us formulate the "1-SJ" as the trivial problem where the input is a set $R$ of $n$ rectangles,and the goal is simply to enumerate each rectangle of $R$ ; the problem can obviously be "solved" in $O\left( n\right)$ time. We assume the existence of an algorithm $\mathcal{A}$ that can settle $\kappa$ -SJ for all $\kappa  \in  \left\lbrack  {1,k - 1}\right\rbrack$ . Denote by ${F}_{\kappa }\left( {n,\mathrm{{OUT}}}\right)$ the worst-case runtime of $\mathcal{A}$ on any instance of $\kappa$ -SJ that has input size $n$ and output size OUT. We consider that ${F}_{\kappa }\left( {n,\mathrm{{OUT}}}\right)  \leq  {F}_{\kappa  + 1}\left( {n,\mathrm{{OUT}}}\right)$ for any $\kappa  \geq  1$ ,that is,its overhead on $\kappa$ -SJ should not be larger than that on $\left( {\kappa  + 1}\right)$ -SJ.

我们的目标是证明H - V $k$ -SJ可以有效地归约为(k - 1) - SJ（注意：这里是(k - 1) - SJ，而不是$\mathrm{H} - \mathrm{V}\left( {k - 1}\right)  - \mathrm{{SJ}}$）。为了确保我们符号系统的合理性，让我们将“1 - SJ”定义为一个平凡问题，其输入是一个包含$n$个矩形的集合$R$，目标仅仅是枚举$R$中的每个矩形；显然，该问题可以在$O\left( n\right)$时间内“解决”。我们假设存在一个算法$\mathcal{A}$，它可以解决所有$\kappa  \in  \left\lbrack  {1,k - 1}\right\rbrack$的$\kappa$ - SJ问题。用${F}_{\kappa }\left( {n,\mathrm{{OUT}}}\right)$表示算法$\mathcal{A}$在任何输入规模为$n$、输出规模为OUT的$\kappa$ - SJ实例上的最坏情况运行时间。我们认为对于任何$\kappa  \geq  1$，有${F}_{\kappa }\left( {n,\mathrm{{OUT}}}\right)  \leq  {F}_{\kappa  + 1}\left( {n,\mathrm{{OUT}}}\right)$，即它在$\kappa$ - SJ上的开销不应大于在$\left( {\kappa  + 1}\right)$ - SJ上的开销。

We will establish:

我们将证明：

LEMMA 3.1. Equipped with the algorithm $\mathcal{A}$ described above,the H-Vk-SJ problem can be solved in

引理3.1. 配备上述算法$\mathcal{A}$，H - V k - SJ问题可以在

$$
O\left( k\right)  \cdot  \left( {{F}_{k - 1}\left( {n,\mathrm{{OUT}}}\right)  + n\log n + k \cdot  \mathrm{{OUT}}}\right) 
$$

time where $n$ (resp.,OUT) is the input (resp.,output) size of the problem. Furthermore,if $\mathcal{A}$ is comparison-based, the H-V k-SJ algorithm obtained is also comparison-based.

时间内解决，其中$n$（分别地，OUT）是该问题的输入（分别地，输出）规模。此外，如果$\mathcal{A}$是基于比较的，那么所得到的H - V k - SJ算法也是基于比较的。

The part of the paper from this point till the end of Section 5 will be devoted to proving the above lemma. This is the most challenging step in solving the general $k$ -SJ problem optimally,as will be discussed in Section 6, where we will prove Theorems 1.1 and 1.2 based on Lemma 3.1.

从本文的这一点到第5节结束的部分将致力于证明上述引理。这是最优解决一般$k$ - SJ问题中最具挑战性的一步，正如将在第6节中讨论的那样，我们将基于引理3.1证明定理1.1和定理1.2。

Consider any $k$ -tuple $\left( {{r}_{1},\ldots ,{r}_{k - 2},h,v}\right)$ in the join result $\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 2},H,V}\right)$ . We classify the tuple into one of the two types below:

考虑连接结果 $\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 2},H,V}\right)$ 中的任意 $k$ 元组 $\left( {{r}_{1},\ldots ,{r}_{k - 2},h,v}\right)$。我们将该元组分为以下两种类型之一：

- Type 1: $h$ crosses all of ${r}_{1},\ldots ,{r}_{k - 2}$ and,at the same time, $v$ crosses all of ${r}_{1},\ldots ,{r}_{k - 2}$ ;

- 类型 1：$h$ 与所有 ${r}_{1},\ldots ,{r}_{k - 2}$ 相交，同时，$v$ 也与所有 ${r}_{1},\ldots ,{r}_{k - 2}$ 相交；

- Type 2: either $h$ or $v$ fails to cross at least one rectangle in $\left\{  {{r}_{1},{r}_{2},\ldots ,{r}_{k - 2}}\right\}$ . Equivalently,at least a rectangle ${r}_{i}$ (for some $i \in  \left\lbrack  {k - 2}\right\rbrack$ ) covers an endpoint of either $h$ or $v$ or both.

- 类型 2：$h$ 或 $v$ 至少未能与 $\left\{  {{r}_{1},{r}_{2},\ldots ,{r}_{k - 2}}\right\}$ 中的一个矩形相交。等价地，至少有一个矩形 ${r}_{i}$（对于某个 $i \in  \left\lbrack  {k - 2}\right\rbrack$）覆盖了 $h$ 或 $v$ 或两者的一个端点。

Figure 3 illustrates a result tuple of each type,assuming $k = 4$ . In Section 4 (resp.,5),we will explain how to produce the result tuples of Type 1 (resp., 2) in the time complexity claimed in Lemma 3.1.

图 3 展示了每种类型的一个结果元组，假设 $k = 4$。在第 4 节（相应地，第 5 节）中，我们将解释如何在引理 3.1 所声称的时间复杂度内生成类型 1（相应地，类型 2）的结果元组。

Remark. In [20],Rahul et al. studied the problem of storing a set $H$ of horizontal segments and a set $V$ of vertical segments in a data structure such that,given a query rectangle $r$ ,all the pairs $\left( {h,v}\right)  \in  H \times  V$ satisfying $h \cap  v \cap  r \neq  \varnothing$ can be reported efficiently. They gave a structure of $O\left( {n\log n}\right)$ space that can be built in $O\left( {n\log n}\right)$ time and can be used to answer a query in $O\left( {\log n + K}\right)$ time, where $n = \left| H\right|  + \left| V\right|$ and $K$ is the number of pairs reported. Their structure can be utilized to solve H-V 3-SJ in $O\left( {n\log n + \text{OUT}}\right)$ time. Oh and Ahn [17] developed a structure for solving a problem more general than that of [20]; however, in the specific scenario of [20], the structure of [17] offers the same guarantees as [20]. We are unaware of a way to extend these solutions to handle H-V $k$ -SJ of $k > 3$ . Our method for proving Lemma 3.1 is based on drastically different ideas even for $k = 3$ .

备注。在文献 [20] 中，拉胡尔（Rahul）等人研究了将一组水平线段 $H$ 和一组垂直线段 $V$ 存储在一个数据结构中的问题，使得给定一个查询矩形 $r$，可以高效地报告所有满足 $h \cap  v \cap  r \neq  \varnothing$ 的对 $\left( {h,v}\right)  \in  H \times  V$。他们给出了一个空间复杂度为 $O\left( {n\log n}\right)$ 的结构，该结构可以在 $O\left( {n\log n}\right)$ 时间内构建，并且可以在 $O\left( {\log n + K}\right)$ 时间内回答一个查询，其中 $n = \left| H\right|  + \left| V\right|$ 且 $K$ 是报告的对的数量。他们的结构可用于在 $O\left( {n\log n + \text{OUT}}\right)$ 时间内解决水平 - 垂直 3 - 半连接（H - V 3 - SJ）问题。吴（Oh）和安（Ahn）[17] 开发了一个用于解决比 [20] 中问题更一般的问题的结构；然而，在 [20] 的特定场景下，[17] 中的结构提供了与 [20] 相同的保证。我们不知道如何扩展这些解决方案来处理 $k > 3$ 的水平 - 垂直 $k$ - 半连接（H - V $k$ - SJ）问题。即使对于 $k = 3$，我们证明引理 3.1 的方法也是基于截然不同的思路。

## 4 H-V $k$ -SJ: Result Tuples of Type 1

## 4 水平 - 垂直 $k$ - 半连接（H - V $k$ - SJ）：类型 1 的结果元组

As before,let ${R}_{1},\ldots ,{R}_{k - 2},H$ ,and $V$ be the input sets of the H-V $k$ -SJ problem. Denote by ${\mathcal{J}}_{1}$ the set of type-1 result tuples defined in Section 3. In this section,we aim to compute a set ${\mathcal{J}}^{ * }$ satisfying

和之前一样，设 ${R}_{1},\ldots ,{R}_{k - 2},H$ 和 $V$ 是水平 - 垂直 $k$ - 半连接问题的输入集合。用 ${\mathcal{J}}_{1}$ 表示第 3 节中定义的类型 1 结果元组的集合。在本节中，我们的目标是计算一个满足以下条件的集合 ${\mathcal{J}}^{ * }$：

$$
{\mathcal{J}}_{1} \subseteq  {\mathcal{J}}^{ * } \subseteq  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 2},H,V}\right)  \tag{5}
$$

where $\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 2},H,V}\right)$ ,let us recall,is the join result of the (whole) H-V $k$ -SJ. Remember that the output size OUT is defined as $\left| {\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 2},H,V}\right) }\right|$ . From ${\mathcal{J}}^{ * }$ ,we will report only those $k$ -tuples belonging to ${\mathcal{J}}_{1}$ and ignore the rest.

其中，我们回顾一下，$\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 2},H,V}\right)$ 是（整个）水平 - 垂直 $k$ - 半连接的连接结果。请记住，输出大小 OUT 定义为 $\left| {\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 2},H,V}\right) }\right|$。从 ${\mathcal{J}}^{ * }$ 中，我们将只报告那些属于 ${\mathcal{J}}_{1}$ 的 $k$ 元组，而忽略其余的。

Example 4.1. To illustrate our algorithm, we will utilize the running example in Figure 4a, where $k = 4$ ,and ${R}_{1} = \{ \alpha \}$ (the solid rectangle), ${R}_{2} = \left\{  {{\beta }_{1},{\beta }_{2}}\right\}$ (the dashed rectangles), $H = \left\{  {{h}_{1},{h}_{2}}\right.$ , $\left. {\ldots ,{h}_{6}}\right\}$ ,and $V = \left\{  {{v}_{1},{v}_{2},\ldots ,{v}_{5}}\right\}$ . The set ${\mathcal{J}}_{1}$ contains the following tuples: $\left( {\alpha ,{\beta }_{2},{h}_{2},{v}_{3}}\right) ,\left( {\alpha ,{\beta }_{2},{h}_{2},{v}_{5}}\right)$ , $\left( {\alpha ,{\beta }_{2},{h}_{5},{v}_{3}}\right)$ ,and $\left( {\alpha ,{\beta }_{2},{h}_{5},{v}_{5}}\right)$ .

示例4.1。为了说明我们的算法，我们将使用图4a中的运行示例，其中$k = 4$ ，并且${R}_{1} = \{ \alpha \}$ （实心矩形），${R}_{2} = \left\{  {{\beta }_{1},{\beta }_{2}}\right\}$ （虚线矩形），$H = \left\{  {{h}_{1},{h}_{2}}\right.$ ，$\left. {\ldots ,{h}_{6}}\right\}$ ，以及$V = \left\{  {{v}_{1},{v}_{2},\ldots ,{v}_{5}}\right\}$ 。集合${\mathcal{J}}_{1}$ 包含以下元组：$\left( {\alpha ,{\beta }_{2},{h}_{2},{v}_{3}}\right) ,\left( {\alpha ,{\beta }_{2},{h}_{2},{v}_{5}}\right)$ ，$\left( {\alpha ,{\beta }_{2},{h}_{5},{v}_{3}}\right)$ ，以及$\left( {\alpha ,{\beta }_{2},{h}_{5},{v}_{5}}\right)$ 。

Sets ${R}_{1}^{\prime },{R}_{2}^{\prime },\ldots ,{R}_{k - 2}^{\prime }$ . Fix any $i \in  \left\lbrack  {k - 2}\right\rbrack$ . For each rectangle $r \in  {R}_{i}$ ,we compute four segments:

集合${R}_{1}^{\prime },{R}_{2}^{\prime },\ldots ,{R}_{k - 2}^{\prime }$ 。固定任意$i \in  \left\lbrack  {k - 2}\right\rbrack$ 。对于每个矩形$r \in  {R}_{i}$ ，我们计算四个线段：

- ${h}_{ \bot  }$ (resp., ${h}_{\top }$ ): the lowest (resp.,highest) segment in $H$ that crosses $r$ ;

- ${h}_{ \bot  }$ （分别地，${h}_{\top }$ ）：$H$ 中穿过$r$ 的最低（分别地，最高）线段；

- ${v}_{ \vdash  }$ (resp., ${v}_{ \dashv  }$ ): the leftmost (resp.,rightmost) segment in $V$ that crosses $r$ .

- ${v}_{ \vdash  }$ （分别地，${v}_{ \dashv  }$ ）：$V$ 中穿过$r$ 的最左（分别地，最右）线段。

Define ${r}^{\prime } = \left\lbrack  {{x}_{ \vdash  },{x}_{ \dashv  }}\right\rbrack   \times  \left\lbrack  {{y}_{ \bot  },{y}_{\top }}\right\rbrack$ ,where ${x}_{ \vdash  }$ (resp., ${x}_{ \dashv  }$ ) is the x-coordinate of ${v}_{ \vdash  }$ (resp., ${v}_{ \dashv  }$ ),and ${y}_{ \bot  }$ (resp., ${y}_{\top }$ ) is the y-coordinate of ${h}_{ \bot  }$ (resp., ${h}_{\top }$ ). We say that ${r}^{\prime }$ is the trimmed rectangle of $r$ ,and conversely, $r$ is the full rectangle of ${r}^{\prime }$ . Note that ${r}^{\prime }$ exists if and only if $r$ is crossed by at least one horizontal segment in $H$ and by at least one vertical segment in $V$ .

定义${r}^{\prime } = \left\lbrack  {{x}_{ \vdash  },{x}_{ \dashv  }}\right\rbrack   \times  \left\lbrack  {{y}_{ \bot  },{y}_{\top }}\right\rbrack$ ，其中${x}_{ \vdash  }$ （分别地，${x}_{ \dashv  }$ ）是${v}_{ \vdash  }$ （分别地，${v}_{ \dashv  }$ ）的x坐标，并且${y}_{ \bot  }$ （分别地，${y}_{\top }$ ）是${h}_{ \bot  }$ （分别地，${h}_{\top }$ ）的y坐标。我们称${r}^{\prime }$ 是$r$ 的修剪矩形，反之，$r$ 是${r}^{\prime }$ 的完整矩形。注意，当且仅当$r$ 被$H$ 中的至少一个水平线段和$V$ 中的至少一个垂直线段穿过时，${r}^{\prime }$ 才存在。

Construct

构建

$$
{R}_{i}^{\prime } = \left\{  {{r}^{\prime } \mid  r \in  {R}_{i}}\right. \text{and its trimmed rectangle}\left. {{r}^{\prime }\text{exists}}\right\}  \text{.} \tag{6}
$$

Computing the "segment ${h}_{ \bot  }$ " for each $r \in  {R}_{i}$ is an instance of Problem 2 (the find-lowest version, with $H$ and ${R}_{i}$ as the input). By symmetry,so is computing the ${h}_{\top },{v}_{ \vdash  }$ ,and ${v}_{ \dashv  }$ segments for each $r \in  {R}_{i}$ . It thus follows from Section 2 that ${R}_{1}^{\prime },{R}_{2}^{\prime },\ldots ,{R}_{k - 2}^{\prime }$ can be produced in $O\left( {{kn}\log n}\right)$ total time.

为每个$r \in  {R}_{i}$ 计算“线段${h}_{ \bot  }$ ”是问题2的一个实例（最低查找版本，以$H$ 和${R}_{i}$ 作为输入）。通过对称性，为每个$r \in  {R}_{i}$ 计算${h}_{\top },{v}_{ \vdash  }$ 和${v}_{ \dashv  }$ 线段也是如此。因此，从第2节可知，可以在$O\left( {{kn}\log n}\right)$ 的总时间内生成${R}_{1}^{\prime },{R}_{2}^{\prime },\ldots ,{R}_{k - 2}^{\prime }$ 。

We now solve a(k - 2)-SJ problem on the input $\left\{  {{R}_{1}^{\prime },{R}_{2}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right\}$ using the algorithm $\mathcal{A}$ supplied (see Lemma 3.1). This(k - 2)-SJ clearly has an input size at most $n$ ,and let us represent its result as $\mathcal{J}\left( {{R}_{1}^{\prime },{R}_{2}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$ . We prove in Appendix B:

我们现在使用所提供的算法$\mathcal{A}$来解决输入$\left\{  {{R}_{1}^{\prime },{R}_{2}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right\}$上的(k - 2)-SJ问题（见引理3.1）。这个(k - 2)-SJ问题的输入规模显然至多为$n$，让我们将其结果表示为$\mathcal{J}\left( {{R}_{1}^{\prime },{R}_{2}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$。我们在附录B中证明：

LEMMA 4.1. $\left| {\mathcal{J}\left( {{R}_{1}^{\prime },{R}_{2}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right) }\right|  \leq$ OUT.

引理4.1. $\left| {\mathcal{J}\left( {{R}_{1}^{\prime },{R}_{2}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right) }\right|  \leq$ 输出。

As a corollary of Lemma 4.1,the(k - 2)-SJ can be settled in ${F}_{k - 2}\left( {n,\mathrm{{OUT}}}\right)$ time.

作为引理4.1的一个推论，(k - 2)-SJ问题可以在${F}_{k - 2}\left( {n,\mathrm{{OUT}}}\right)$时间内解决。

Example 4.2. Figure 4b shows the rectangles in ${R}_{1}^{\prime } = \left\{  {\alpha }^{\prime }\right\}$ and ${R}_{2}^{\prime } = \left\{  {{\beta }_{1}^{\prime },{\beta }_{2}^{\prime }}\right\}$ . For instance, ${\alpha }^{\prime }$ ,which is trimmed from rectangle $\alpha$ ,is decided by ${h}_{ \bot  } = {h}_{2},{h}_{\top } = {h}_{6},{v}_{ \vdash  } = {v}_{3}$ ,and ${v}_{ \dashv  } = {v}_{5}$ . The(k - 2)-SJ on ${R}_{1}^{\prime }$ and ${R}_{2}^{\prime }$ returns $\mathcal{J}\left( {{R}_{1}^{\prime },{R}_{2}^{\prime }}\right)  = \left\{  {\left( {{\alpha }^{\prime },{\beta }_{1}^{\prime }}\right) ,\left( {{\alpha }^{\prime },{\beta }_{2}^{\prime }}\right) }\right\}$ .

示例4.2。图4b展示了${R}_{1}^{\prime } = \left\{  {\alpha }^{\prime }\right\}$和${R}_{2}^{\prime } = \left\{  {{\beta }_{1}^{\prime },{\beta }_{2}^{\prime }}\right\}$中的矩形。例如，从矩形$\alpha$中裁剪得到的${\alpha }^{\prime }$由${h}_{ \bot  } = {h}_{2},{h}_{\top } = {h}_{6},{v}_{ \vdash  } = {v}_{3}$和${v}_{ \dashv  } = {v}_{5}$决定。${R}_{1}^{\prime }$和${R}_{2}^{\prime }$上的(k - 2)-SJ问题返回$\mathcal{J}\left( {{R}_{1}^{\prime },{R}_{2}^{\prime }}\right)  = \left\{  {\left( {{\alpha }^{\prime },{\beta }_{1}^{\prime }}\right) ,\left( {{\alpha }^{\prime },{\beta }_{2}^{\prime }}\right) }\right\}$。

Generating ${\mathcal{J}}^{ * }$ . Take any(k - 2)-tuple $t = \left( {{r}_{1}^{\prime },{r}_{2}^{\prime },\ldots ,{r}_{k - 2}^{\prime }}\right)  \in  \mathcal{J}\left( {{R}_{1}^{\prime },{R}_{2}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$ . The reader should recall from Section 2 that

生成${\mathcal{J}}^{ * }$。取任意的(k - 2)-元组$t = \left( {{r}_{1}^{\prime },{r}_{2}^{\prime },\ldots ,{r}_{k - 2}^{\prime }}\right)  \in  \mathcal{J}\left( {{R}_{1}^{\prime },{R}_{2}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$。读者应该从第2节中回忆起

- ${B}_{t}$ is $\mathop{\bigcap }\limits_{{i = 1}}^{{k - 2}}t\left\lbrack  i\right\rbrack   = \mathop{\bigcap }\limits_{{i = 1}}^{{k - 2}}{r}_{i}^{\prime }$ ;

- ${B}_{t}$ 是 $\mathop{\bigcap }\limits_{{i = 1}}^{{k - 2}}t\left\lbrack  i\right\rbrack   = \mathop{\bigcap }\limits_{{i = 1}}^{{k - 2}}{r}_{i}^{\prime }$；

- left-guard(t)is the ${r}_{i}^{\prime }\left( {1 \leq  i \leq  k - 2}\right)$ with $\operatorname{left}\left( {r}_{i}^{\prime }\right)  = \operatorname{left}\left( {B}_{\mathbf{t}}\right)$ ;

- 左守卫(t)是具有$\operatorname{left}\left( {r}_{i}^{\prime }\right)  = \operatorname{left}\left( {B}_{\mathbf{t}}\right)$的${r}_{i}^{\prime }\left( {1 \leq  i \leq  k - 2}\right)$；

- bot-guard(t)is the ${r}_{i}^{\prime }\left( {1 \leq  i \leq  k - 2}\right)$ with $\operatorname{bot}\left( {r}_{i}^{\prime }\right)  = \operatorname{bot}\left( {B}_{t}\right)$ .

- 底守卫(t)是具有$\operatorname{bot}\left( {r}_{i}^{\prime }\right)  = \operatorname{bot}\left( {B}_{t}\right)$的${r}_{i}^{\prime }\left( {1 \leq  i \leq  k - 2}\right)$。

We now introduce:

我们现在引入：

$$
d - {\operatorname{cross}}_{H}\left( \mathbf{t}\right)  = \left\{  {h \in  H \mid  h}\right. \text{crosses both}{B}_{t}\text{and bot-guard}\left( \mathbf{t}\right) \}  \tag{7}
$$

$$
d - {\operatorname{cross}}_{V}\left( \mathbf{t}\right)  = \left\{  {v \in  V \mid  v\text{ crosses both }{B}_{t}\text{ and left-guard }\left( \mathbf{t}\right) }\right\}  .
$$

The prefix "d-" stands for "double". These sets have important properties as stated in the next lemma, whose proof can be found in Appendix B:

前缀“d - ”代表“双重”。这些集合具有下一个引理中所述的重要性质，其证明可在附录B中找到：

<!-- Media -->

<!-- figureText: ${v}_{3}$ ${v}_{5}$ If ${\alpha }^{\prime }$ 10 7 4 3 2 5 7 12 (b) ${R}_{1}^{\prime }$ and ${R}_{2}^{\prime }$ 12 ${h}_{6}$ 11 G ${h}_{5}$ 9 ${h}_{4}$ 6 D `α* ${h}_{2}$ ${h}_{1}$ 3 2 2 5 6 7 1 12 (d) ${R}_{1}^{ * },{R}_{2}^{ * }\left( { = \varnothing }\right)$ ,and $H$ a 12 _ ${h}_{6}$ 11 10 ${h}_{5}$ 9 7 6 ${h}_{3}$ ${h}_{2}$ ${h}_{1}$ 4 3 3 5 8 10 (a) ${R}_{1},{R}_{2},H$ ,and $V$ ${v}_{3}$ ${v}_{5}$ 12 11 ` ${\alpha }^{\prime }$ G ${h}_{5}$ 9 8 ${h}_{4}$ 7 6 ${h}_{3}$ 5 ${\beta }_{1}^{\prime }$ ${h}_{2}$ E 3 2 1 7 8 (c) ${R}_{1}^{\prime },{R}_{2}^{\prime },H$ ,and $V$ -->

<img src="https://cdn.noedgeai.com/0195ccc5-d2d9-7daa-9177-3ae04293d71f_7.jpg?x=303&y=260&w=954&h=1021&r=0"/>

Fig. 4. Finding H-V $k$ -SJ result tuples of type $1\left( {k = 4}\right)$

图4. 寻找类型为$1\left( {k = 4}\right)$的H - V $k$ - SJ结果元组

<!-- Media -->

LEMMA 4.2. All the following statements are true:

引理4.2. 以下所有陈述均为真：

(1) Consider any(k - 2)-tuple $t \in  \mathcal{J}\left( {{R}_{1}^{\prime },{R}_{2}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$ . Let ${r}_{i}\left( {i \in  \left\lbrack  {k - 2}\right\rbrack  }\right)$ be the full rectangle of $t\left\lbrack  i\right\rbrack$ . Then,for any $h \in  \mathrm{d} - {\operatorname{cross}}_{H}\left( t\right)$ and any $v \in  \mathrm{d} - {\operatorname{cross}}_{V}\left( t\right)$ ,the $k$ -tuple $\left( {{r}_{1},{r}_{2},\ldots ,{r}_{k - 2},h,v}\right)$ must belong to $\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 2},H,V}\right)$ .

(1) 考虑任意的(k - 2)元组 $t \in  \mathcal{J}\left( {{R}_{1}^{\prime },{R}_{2}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$ 。设 ${r}_{i}\left( {i \in  \left\lbrack  {k - 2}\right\rbrack  }\right)$ 为 $t\left\lbrack  i\right\rbrack$ 的全矩形。那么，对于任意的 $h \in  \mathrm{d} - {\operatorname{cross}}_{H}\left( t\right)$ 和任意的 $v \in  \mathrm{d} - {\operatorname{cross}}_{V}\left( t\right)$ ， $k$ 元组 $\left( {{r}_{1},{r}_{2},\ldots ,{r}_{k - 2},h,v}\right)$ 必定属于 $\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 2},H,V}\right)$ 。

(2) Consider any $k$ -tuple $\left( {{r}_{1},{r}_{2},\ldots ,{r}_{k - 2},h,v}\right)  \in  {\mathcal{J}}_{1}$ . Let ${r}_{i}^{\prime }\left( {i \in  \left\lbrack  {k - 2}\right\rbrack  }\right)$ be the trimmed rectangle of ${r}_{i}$ ,and set $\mathbf{t} = \left( {{r}_{1}^{\prime },{r}_{2}^{\prime },\ldots ,{r}_{k - 2}^{\prime }}\right)$ . Then,we must have

(2) 考虑任意的 $k$ 元组 $\left( {{r}_{1},{r}_{2},\ldots ,{r}_{k - 2},h,v}\right)  \in  {\mathcal{J}}_{1}$ 。设 ${r}_{i}^{\prime }\left( {i \in  \left\lbrack  {k - 2}\right\rbrack  }\right)$ 为 ${r}_{i}$ 的修剪矩形，并设 $\mathbf{t} = \left( {{r}_{1}^{\prime },{r}_{2}^{\prime },\ldots ,{r}_{k - 2}^{\prime }}\right)$ 。那么，我们必定有

- $t \in  \mathcal{J}\left( {{R}_{1}^{\prime },{R}_{2}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$ ;

- $h \in  \mathrm{d} - {\operatorname{cross}}_{H}\left( \mathbf{t}\right)$ and $v \in  \mathrm{d} - {\operatorname{cross}}_{V}\left( \mathbf{t}\right)$ .

- $h \in  \mathrm{d} - {\operatorname{cross}}_{H}\left( \mathbf{t}\right)$ 且 $v \in  \mathrm{d} - {\operatorname{cross}}_{V}\left( \mathbf{t}\right)$ 。

(3) $\mathop{\sum }\limits_{t}\left| {\mathrm{d} - {\operatorname{cross}}_{H}\left( t\right) }\right|  \leq  \mathrm{{OUT}}$ and $\mathop{\sum }\limits_{t}\left| {\mathrm{d} - {\operatorname{cross}}_{V}\left( t\right) }\right|  \leq  \mathrm{{OUT}}$ ,where the two summations are over all $\mathbf{t} \in  \mathcal{J}\left( {{R}_{1}^{\prime },{R}_{2}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$ .

(3) $\mathop{\sum }\limits_{t}\left| {\mathrm{d} - {\operatorname{cross}}_{H}\left( t\right) }\right|  \leq  \mathrm{{OUT}}$ 且 $\mathop{\sum }\limits_{t}\left| {\mathrm{d} - {\operatorname{cross}}_{V}\left( t\right) }\right|  \leq  \mathrm{{OUT}}$ ，其中两个求和是对所有的 $\mathbf{t} \in  \mathcal{J}\left( {{R}_{1}^{\prime },{R}_{2}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$ 进行的。

Example 4.3. Let us examine,in turn,the two 2-tuples ${\mathbf{t}}_{1} = \left( {{\alpha }^{\prime },{\beta }_{1}^{\prime }}\right)$ and ${\mathbf{t}}_{2} = \left( {{\alpha }^{\prime },{\beta }_{2}^{\prime }}\right)$ in $\mathcal{J}\left( {{R}_{1}^{\prime },{R}_{2}^{\prime }}\right)$ . For ${\mathbf{t}}_{1},{B}_{{\mathbf{t}}_{1}}$ is the rectangle ABCD in Figure $4\mathrm{c}$ ,and left-guard $\left( {\mathbf{t}}_{1}\right)  =$ bot-guard $\left( {\mathbf{t}}_{1}\right)  = {\alpha }^{\prime }$ . Accordingly, d-cross ${s}_{H}\left( {\mathbf{t}}_{1}\right)  = \left\{  {h}_{2}\right\}$ and d-cross ${s}_{V}\left( {\mathbf{t}}_{1}\right)  = \left\{  {v}_{3}\right\}$ . For ${\mathbf{t}}_{2},{B}_{{\mathbf{t}}_{2}}$ is the rectangle AEFG,and left-guard $\left( {\mathbf{t}}_{2}\right)  =$ bot-guard $\left( {\mathbf{t}}_{2}\right)  = {\alpha }^{\prime }$ . Accordingly, $\mathrm{d} - {\operatorname{cross}}_{H}\left( {\mathbf{t}}_{2}\right)  = \left\{  {{h}_{2},{h}_{4},{h}_{5}}\right\}$ and $\mathrm{d} - {\operatorname{cross}}_{V}\left( {\mathbf{t}}_{2}\right)  = \left\{  {{v}_{3},{v}_{5}}\right\}$ .

示例4.3。让我们依次考察$\mathcal{J}\left( {{R}_{1}^{\prime },{R}_{2}^{\prime }}\right)$中的两个二元组${\mathbf{t}}_{1} = \left( {{\alpha }^{\prime },{\beta }_{1}^{\prime }}\right)$和${\mathbf{t}}_{2} = \left( {{\alpha }^{\prime },{\beta }_{2}^{\prime }}\right)$。对于${\mathbf{t}}_{1},{B}_{{\mathbf{t}}_{1}}$，它是图$4\mathrm{c}$中的矩形ABCD，并且左守卫为$\left( {\mathbf{t}}_{1}\right)  =$，底守卫为$\left( {\mathbf{t}}_{1}\right)  = {\alpha }^{\prime }$。因此，d交叉为${s}_{H}\left( {\mathbf{t}}_{1}\right)  = \left\{  {h}_{2}\right\}$且d交叉为${s}_{V}\left( {\mathbf{t}}_{1}\right)  = \left\{  {v}_{3}\right\}$。对于${\mathbf{t}}_{2},{B}_{{\mathbf{t}}_{2}}$，它是矩形AEFG，并且左守卫为$\left( {\mathbf{t}}_{2}\right)  =$，底守卫为$\left( {\mathbf{t}}_{2}\right)  = {\alpha }^{\prime }$。因此，$\mathrm{d} - {\operatorname{cross}}_{H}\left( {\mathbf{t}}_{2}\right)  = \left\{  {{h}_{2},{h}_{4},{h}_{5}}\right\}$且$\mathrm{d} - {\operatorname{cross}}_{V}\left( {\mathbf{t}}_{2}\right)  = \left\{  {{v}_{3},{v}_{5}}\right\}$。

---

Equipped with Lemma 4.2,we generate our target ${\mathcal{J}}^{ * }$ as follows:

借助引理4.2，我们按如下方式生成目标${\mathcal{J}}^{ * }$：

													algorithm generate- ${\mathcal{J}}^{ * }$

													算法generate- ${\mathcal{J}}^{ * }$

														1. ${\mathcal{J}}^{ * } = \varnothing$

															2. for each(k - 2)-tuple $\mathbf{t} \in  \mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$ do

															2. 对每个(k - 2)元组$\mathbf{t} \in  \mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$执行以下操作

														3. $\;{r}_{i} \leftarrow$ the full rectangle of $\mathbf{t}\left\lbrack  i\right\rbrack$ ,for each $i \in  \left\lbrack  {k - 2}\right\rbrack$

														3. $\;{r}_{i} \leftarrow$是$\mathbf{t}\left\lbrack  i\right\rbrack$的完整矩形，对每个$i \in  \left\lbrack  {k - 2}\right\rbrack$而言

---

4. for each $\left( {h,v}\right)  \in  \mathrm{d} - {\operatorname{cross}}_{H}\left( \mathbf{t}\right)  \times  \mathrm{d} - {\operatorname{cross}}_{V}\left( \mathbf{t}\right)$ do

4. 对每个$\left( {h,v}\right)  \in  \mathrm{d} - {\operatorname{cross}}_{H}\left( \mathbf{t}\right)  \times  \mathrm{d} - {\operatorname{cross}}_{V}\left( \mathbf{t}\right)$执行以下操作

5. add $\left( {{r}_{1},\ldots ,{r}_{k - 2},h,v}\right)$ to ${\mathcal{J}}^{ * }$

5. 将$\left( {{r}_{1},\ldots ,{r}_{k - 2},h,v}\right)$添加到${\mathcal{J}}^{ * }$中

By statements (1) and (2) of Lemma 4.2,the set ${\mathcal{J}}^{ * }$ thus computed indeed satisfies (5). Furthermore, if we are given $\mathrm{d} - {\operatorname{cross}}_{H}\left( \mathbf{t}\right)$ and $\mathrm{d} - {\operatorname{cross}}_{V}\left( \mathbf{t}\right)$ for each $\mathbf{t}$ ,the above algorithm runs in $O(1 + k$ . $\left. {\left| {\mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right) }\right|  + k \cdot  \left| {\mathcal{J}}^{ * }\right| }\right)  = O\left( {1 + k \cdot  \mathrm{{OUT}}}\right)$ time,where the derivation used (5) and Lemma 4.1.

根据引理4.2的陈述(1)和(2)，如此计算得到的集合${\mathcal{J}}^{ * }$确实满足(5)。此外，如果我们已知每个$\mathbf{t}$对应的$\mathrm{d} - {\operatorname{cross}}_{H}\left( \mathbf{t}\right)$和$\mathrm{d} - {\operatorname{cross}}_{V}\left( \mathbf{t}\right)$，上述算法的运行时间为$O(1 + k$。$\left. {\left| {\mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right) }\right|  + k \cdot  \left| {\mathcal{J}}^{ * }\right| }\right)  = O\left( {1 + k \cdot  \mathrm{{OUT}}}\right)$时间，其中推导过程使用了(5)和引理4.1。

The rest of the section will focus on how to prepare the sets $\mathrm{d} - {\operatorname{cross}}_{H}\left( \mathbf{t}\right)$ of all $\mathbf{t} \in  \mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$ in $O\left( {{kn}\log n + k \cdot  \mathrm{{OUT}}}\right)$ time. An analogous method can be used to compute the sets $\mathrm{d} - {\operatorname{cross}}_{V}\left( \mathbf{t}\right)$ of all $t$ within the same time complexity.

本节的其余部分将重点介绍如何在$O\left( {{kn}\log n + k \cdot  \mathrm{{OUT}}}\right)$时间内准备所有$\mathbf{t} \in  \mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$的集合$\mathrm{d} - {\operatorname{cross}}_{H}\left( \mathbf{t}\right)$。可以使用类似的方法在相同的时间复杂度内计算所有$t$的集合$\mathrm{d} - {\operatorname{cross}}_{V}\left( \mathbf{t}\right)$。

Example 4.4. In our running example,the ${\mathcal{J}}^{ * }$ computed includes 7 tuples: $\left( {\alpha ,{\beta }_{1},{h}_{2},{v}_{3}}\right)$ ,and $\{ \alpha \}  \times$ $\left. {\overline{\left\{  {\beta }_{2}\right\}  } \times  \left\{  {{h}_{2},{h}_{4},{h}_{5}}\right\}   \times  \left\{  {{v}_{3},{v}_{5}}\right\}  \text{. All these 7 tuples belong to}\mathcal{J}\left( {{R}_{1},{R}_{2},H,V}\right) \text{and include the 4 tuples}}\right\}$ in ${\mathcal{J}}_{1}$ (see Example 4.1).

示例4.4。在我们的运行示例中，计算得到的${\mathcal{J}}^{ * }$包含7个元组：$\left( {\alpha ,{\beta }_{1},{h}_{2},{v}_{3}}\right)$，以及$\{ \alpha \}  \times$ $\left. {\overline{\left\{  {\beta }_{2}\right\}  } \times  \left\{  {{h}_{2},{h}_{4},{h}_{5}}\right\}   \times  \left\{  {{v}_{3},{v}_{5}}\right\}  \text{. All these 7 tuples belong to}\mathcal{J}\left( {{R}_{1},{R}_{2},H,V}\right) \text{and include the 4 tuples}}\right\}$在${\mathcal{J}}_{1}$中（见示例4.1）。

Sets ${R}_{1}^{ * },{R}_{2}^{ * },\ldots ,{R}_{k - 2}^{ * }$ . Fix any $i \in  \left\lbrack  {k - 2}\right\rbrack$ . Define for each ${r}^{\prime } \in  {R}_{i}^{\prime }$ :

集合${R}_{1}^{ * },{R}_{2}^{ * },\ldots ,{R}_{k - 2}^{ * }$。固定任意$i \in  \left\lbrack  {k - 2}\right\rbrack$。为每个${r}^{\prime } \in  {R}_{i}^{\prime }$定义：

$$
\operatorname{maxtop}\left( {r}^{\prime }\right)  = \mathop{\max }\limits_{{\mathbf{t} \in  \mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)  : \text{ bot-guard }\left( \mathbf{t}\right)  = {r}^{\prime }}}\operatorname{top}\left( {B}_{\mathbf{t}}\right) . \tag{8}
$$

We set $\operatorname{maxtop}\left( {r}^{\prime }\right)$ to $- \infty$ if no $t \in  \mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$ has ${r}^{\prime }$ as the bot-guard(t). When $\operatorname{maxtop}\left( {r}^{\prime }\right)  \neq$ $- \infty$ ,assuming ${r}^{\prime } = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  \left\lbrack  {{y}_{1},{y}_{2}}\right\rbrack$ ,we introduce a rectangle

如果没有$t \in  \mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$以${r}^{\prime }$作为底部守卫（t），我们将$\operatorname{maxtop}\left( {r}^{\prime }\right)$设为$- \infty$。当$\operatorname{maxtop}\left( {r}^{\prime }\right)  \neq$ $- \infty$时，假设${r}^{\prime } = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  \left\lbrack  {{y}_{1},{y}_{2}}\right\rbrack$，我们引入一个矩形

$$
{r}^{ * } = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  \left\lbrack  {{y}_{1},\operatorname{maxtop}\left( {r}^{\prime }\right) }\right\rbrack  . \tag{9}
$$

and call it the top-sliced rectangle of ${r}^{\prime }$ .

并将其称为${r}^{\prime }$的顶部切片矩形。

Example 4.5. Recall from Example 4.3 that rectangle ${\alpha }^{\prime }$ is both bot-guard $\left( {\mathbf{t}}_{1}\right)$ and bot-guard $\left( {\mathbf{t}}_{2}\right)$ . Thus, $\operatorname{maxtop}\left( {\alpha }^{\prime }\right)  = \max \left\{  {\operatorname{top}\left( {B}_{{t}_{1}}\right) ,\operatorname{top}\left( {B}_{{t}_{2}}\right) }\right\}   = \max \{ 6,{9.5}\}  = {9.5}$ . The top-sliced rectangle of ${\alpha }^{\prime }$ is the rectangle ${\alpha }^{ * }$ in Figure 4d. Rectangles ${\beta }_{1}^{\prime }$ and ${\beta }_{2}^{\prime }$ do not have top-sliced rectangles.

示例4.5。从示例4.3中回忆起，矩形${\alpha }^{\prime }$既是底部守卫$\left( {\mathbf{t}}_{1}\right)$又是底部守卫$\left( {\mathbf{t}}_{2}\right)$。因此，$\operatorname{maxtop}\left( {\alpha }^{\prime }\right)  = \max \left\{  {\operatorname{top}\left( {B}_{{t}_{1}}\right) ,\operatorname{top}\left( {B}_{{t}_{2}}\right) }\right\}   = \max \{ 6,{9.5}\}  = {9.5}$。${\alpha }^{\prime }$的顶部切片矩形是图4d中的矩形${\alpha }^{ * }$。矩形${\beta }_{1}^{\prime }$和${\beta }_{2}^{\prime }$没有顶部切片矩形。

Next,we construct from ${R}_{i}^{\prime }$ a new set of rectangles:

接下来，我们从${R}_{i}^{\prime }$构造一组新的矩形：

$$
{R}_{i}^{ * } = \left\{  {{r}^{ * } \mid  {r}^{\prime } \in  {R}_{i}^{\prime }}\right. \text{and its top-sliced rectangle}\left. {{r}^{ * }\text{exists}}\right\}  \text{.} \tag{10}
$$

In Appendix B,we show how to compute ${R}_{1}^{ * },\ldots ,{R}_{k - 2}^{ * }$ altogether in $O\left( {n + k \cdot  \mathrm{{OUT}}}\right)$ total time.

在附录B中，我们展示了如何在总共$O\left( {n + k \cdot  \mathrm{{OUT}}}\right)$的时间内一起计算${R}_{1}^{ * },\ldots ,{R}_{k - 2}^{ * }$。

Our interest lies specifically in the sets ${\operatorname{cross}}_{H}\left( {r}^{ * }\right)$ of the rectangles ${r}^{ * }$ in ${R}_{i}^{ * }$ ,where ${\operatorname{cross}}_{H}\left( {r}^{ * }\right)  -$ defined in (2) - is the set of segments in $H$ crossing ${r}^{ * }$ . The following lemma,proven in Appendix B, presents some useful properties of these sets.

我们特别关注${R}_{i}^{ * }$中矩形${r}^{ * }$的集合${\operatorname{cross}}_{H}\left( {r}^{ * }\right)$，其中(2)中定义的${\operatorname{cross}}_{H}\left( {r}^{ * }\right)  -$是$H$中与${r}^{ * }$相交的线段集合。附录B中证明的以下引理给出了这些集合的一些有用性质。

LEMMA 4.3. Both statements below are true:

引理4.3. 以下两个陈述均为真：

(1) $\mathop{\sum }\limits_{{i = 1}}^{{k - 2}}\mathop{\sum }\limits_{{{r}^{ * } \in  {R}_{i}^{ * }}}\left| {{\operatorname{cross}}_{H}\left( {r}^{ * }\right) }\right|  \leq$ OUT.

(1) $\mathop{\sum }\limits_{{i = 1}}^{{k - 2}}\mathop{\sum }\limits_{{{r}^{ * } \in  {R}_{i}^{ * }}}\left| {{\operatorname{cross}}_{H}\left( {r}^{ * }\right) }\right|  \leq$ 外部。

(2) Consider any tuple $\mathbf{t} \in  \mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$ . Let ${r}^{\prime } =$ bot-guard(t)and ${r}^{ * }$ be the top-sliced rectangle of ${r}^{\prime }$ . Then,we have ${\operatorname{d-cross}}_{H}\left( t\right)  \subseteq  {\operatorname{cross}}_{H}\left( {r}^{ * }\right)$ . Furthermore,if the (horizontal) segments of ${\operatorname{cross}}_{H}\left( {r}^{ * }\right)$ are sorted in ascending order of their $y$ -coordinates,then $\mathrm{d} - {\operatorname{cross}}_{H}\left( t\right)$ includes a prefix of the sorted order.

(2) 考虑任意元组$\mathbf{t} \in  \mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$。设${r}^{\prime } =$为bot - guard(t)，${r}^{ * }$为${r}^{\prime }$的顶部切片矩形。那么，我们有${\operatorname{d-cross}}_{H}\left( t\right)  \subseteq  {\operatorname{cross}}_{H}\left( {r}^{ * }\right)$。此外，如果${\operatorname{cross}}_{H}\left( {r}^{ * }\right)$的（水平）线段按其$y$坐标升序排序，那么$\mathrm{d} - {\operatorname{cross}}_{H}\left( t\right)$包含排序顺序的一个前缀。

Example 4.6. It is clear from Figure 4d that ${\operatorname{cross}}_{H}\left( {\alpha }^{ * }\right)$ contains ${h}_{2},{h}_{4}$ ,and ${h}_{5}$ ,sorted in ascending order of their y-coordinates. Recall that bot-guard $\left( {\mathbf{t}}_{1}\right)  =$ bot-guard $\left( {\mathbf{t}}_{2}\right)  = {\alpha }^{\prime }$ . Both d-cross ${}_{H}\left( {\mathbf{t}}_{1}\right)  =$ $\left\{  {h}_{2}\right\}$ and $\mathrm{d} - {\operatorname{cross}}_{H}\left( {t}_{2}\right)  = \left\{  {{h}_{2},{h}_{4},{h}_{5}}\right\}$ are indeed prefixes of the sorted ${\operatorname{cross}}_{H}\left( {\alpha }^{ * }\right)$ ,as stated in Lemma 4.3.

示例4.6. 从图4d可以清楚地看出，${\operatorname{cross}}_{H}\left( {\alpha }^{ * }\right)$包含${h}_{2},{h}_{4}$和${h}_{5}$，它们按y坐标升序排序。回想一下，bot - guard $\left( {\mathbf{t}}_{1}\right)  =$，bot - guard $\left( {\mathbf{t}}_{2}\right)  = {\alpha }^{\prime }$。如引理4.3所述，d - cross ${}_{H}\left( {\mathbf{t}}_{1}\right)  =$ $\left\{  {h}_{2}\right\}$和$\mathrm{d} - {\operatorname{cross}}_{H}\left( {t}_{2}\right)  = \left\{  {{h}_{2},{h}_{4},{h}_{5}}\right\}$确实是排序后的${\operatorname{cross}}_{H}\left( {\alpha }^{ * }\right)$的前缀。

Finding the ${\operatorname{cross}}_{H}\left( {r}^{ * }\right)$ sets of all ${r}^{ * } \in  {R}_{i}^{ * }$ is an instance of the find-all-sorted version of Problem $\mathcal{D}$ (with $H$ and ${R}_{i}^{ * }$ as the input). Statement (1) of Lemma 4.3,as well as the discussion in Section 2, assures us that the total time to do so for all ${R}_{1}^{ * },\ldots ,{R}_{k - 2}^{ * }$ is bounded by $O\left( {{kn}\log n + \mathrm{{OUT}}}\right)$ . Note that, for each ${\operatorname{cross}}_{H}\left( {r}^{ * }\right)$ computed,the (horizontal) segments therein have been sorted in ascending order of y-coordinate.

找到所有${r}^{ * } \in  {R}_{i}^{ * }$的${\operatorname{cross}}_{H}\left( {r}^{ * }\right)$集合是问题$\mathcal{D}$的全排序查找版本的一个实例（输入为$H$和${R}_{i}^{ * }$）。引理4.3的陈述(1)以及第2节的讨论向我们保证，对所有${R}_{1}^{ * },\ldots ,{R}_{k - 2}^{ * }$进行此操作的总时间受$O\left( {{kn}\log n + \mathrm{{OUT}}}\right)$限制。请注意，对于计算出的每个${\operatorname{cross}}_{H}\left( {r}^{ * }\right)$，其中的（水平）线段已按y坐标升序排序。

<!-- Media -->

<!-- figureText: ${v}_{2}$ ${v}_{2}$ $\alpha$ 12 11 ${v}_{1}$ 10 B C ${h}_{2}^{\prime }$ 2 4 7 1 12 (b) ${R}_{1},{H}^{\prime }$ ,and $V$ 12 11 ${\beta }_{3}$ 10 9 ${\beta }_{1}$ ${h}_{3}^{ * }$ 6 ${h}_{2}^{ * }$ 2 4 12 (d) ${R}_{2}$ and ${H}^{ * }$ 12 11 ${\beta }_{3}$ ${v}_{1}$ 10 9 8 7 6 ${h}_{2}$ 5 4 3 1 3 6 8 9 12 (a) ${R}_{1},{R}_{2},H$ ,and $V$ 11 ${\beta }_{3}$ 10 9 8 7 6 5 3 2 1 2 3 5 8 (c) ${R}_{2},{H}^{\prime }$ ,and $V$ -->

<img src="https://cdn.noedgeai.com/0195ccc5-d2d9-7daa-9177-3ae04293d71f_9.jpg?x=299&y=259&w=961&h=1020&r=0"/>

Fig. 5. Finding H-V $k$ -SJ result tuples of type $2\left( {k = 4}\right)$

图5. 查找类型为$2\left( {k = 4}\right)$的H - V $k$ - SJ结果元组

<!-- Media -->

Computing the "d-cross" Sets. We are ready to compute $d - {\operatorname{cross}}_{H}\left( t\right)$ ,defined in (7),for any $t \in  \mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$ ,thanks to Statement (2) of Lemma 4.3. First,compute ${B}_{t}$ ,obtain the rectangle ${r}^{\prime } =$ bot-guard(t),and fetch the (already computed) top-sliced rectangle ${r}^{ * }$ of ${r}^{\prime }$ ; these steps require $O\left( k\right)$ time. Then,scan the segments in ${\operatorname{cross}}_{H}\left( {r}^{ * }\right)$ in ascending order of their y-coordinates. For each segment $h$ scanned,check whether $h$ belongs to $\mathrm{d} - {\operatorname{cross}}_{H}\left( t\right)$ ,namely,whether $h$ crosses ${B}_{t}$ (the reader can verify that $h$ must cross bot-guard(t)); this can be done in constant time. Abort the scan as soon as $h \notin  \mathrm{d} - {\operatorname{cross}}_{H}\left( t\right)$ . This way,we produce $\mathrm{d} - {\operatorname{cross}}_{H}\left( t\right)$ in $O\left( {k + \left| {\mathrm{d} - {\operatorname{cross}}_{H}\left( t\right) }\right| }\right)$ time. Doing so for all $t \in  \mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$ takes $O\left( {k \cdot  \left| \mathcal{J}\right|  + \mathop{\sum }\limits_{t}\left| {\mathrm{\;d} - {\operatorname{cross}}_{H}\left( t\right) }\right| }\right)  = O\left( {k \cdot  \mathrm{{OUT}}}\right)$ time,where the derivation used Lemma 4.1 and statement (3) of Lemma 4.2.

计算“d-交叉”集合。由于引理4.3的陈述(2)，我们准备为任意的$t \in  \mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$计算(7)中定义的$d - {\operatorname{cross}}_{H}\left( t\right)$。首先，计算${B}_{t}$，得到矩形${r}^{\prime } =$ bot-guard(t)，并获取${r}^{\prime }$的（已计算的）顶部切片矩形${r}^{ * }$；这些步骤需要$O\left( k\right)$的时间。然后，按照线段在${\operatorname{cross}}_{H}\left( {r}^{ * }\right)$中的y坐标升序扫描这些线段。对于扫描到的每个线段$h$，检查$h$是否属于$\mathrm{d} - {\operatorname{cross}}_{H}\left( t\right)$，即$h$是否与${B}_{t}$相交（读者可以验证$h$必定与bot-guard(t)相交）；这可以在常数时间内完成。一旦$h \notin  \mathrm{d} - {\operatorname{cross}}_{H}\left( t\right)$就中止扫描。通过这种方式，我们可以在$O\left( {k + \left| {\mathrm{d} - {\operatorname{cross}}_{H}\left( t\right) }\right| }\right)$的时间内生成$\mathrm{d} - {\operatorname{cross}}_{H}\left( t\right)$。对所有的$t \in  \mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$都这样做需要$O\left( {k \cdot  \left| \mathcal{J}\right|  + \mathop{\sum }\limits_{t}\left| {\mathrm{\;d} - {\operatorname{cross}}_{H}\left( t\right) }\right| }\right)  = O\left( {k \cdot  \mathrm{{OUT}}}\right)$的时间，其中推导过程使用了引理4.1和引理4.2的陈述(3)。

We conclude that ${\mathcal{J}}_{1}$ - the set of type-1 result tuples - can be computed in ${F}_{k - 2}\left( {n,\mathrm{{OUT}}}\right)  +$ $O\left( {{kn}\log n + k \cdot  \mathrm{{OUT}}}\right)$ time.

我们得出结论：类型1的结果元组集合${\mathcal{J}}_{1}$可以在${F}_{k - 2}\left( {n,\mathrm{{OUT}}}\right)  +$ $O\left( {{kn}\log n + k \cdot  \mathrm{{OUT}}}\right)$的时间内计算得出。

## 5 H-V $k$ -SJ: Result Tuples of Type 2

## 5 水平 - 垂直 $k$ - 半连接：类型2的结果元组

Still,denote by ${R}_{1},\ldots ,{R}_{k - 2},H$ ,and $V$ the input sets of the H-V $k$ -SJ problem. This section will explain how to find the result tuples of Type 2 as defined in Section 3.

仍然用${R}_{1},\ldots ,{R}_{k - 2},H$和$V$表示水平 - 垂直 $k$ - 半连接问题的输入集合。本节将解释如何找到第3节中定义的类型2的结果元组。

As mentioned before,for a result tuple $\left( {{r}_{1},\ldots ,{r}_{k - 2},h,v}\right)$ of this type,a rectangle ${r}_{i}$ ,for some $i \in  \left\lbrack  {k - 2}\right\rbrack$ ,covers an endpoint of $h$ or $v$ or both. As (i) there are $k - 2$ choices for $i$ and (ii) $h$ and $v$ together have four endpoints,we can divide Type 2 further into $4\left( {k - 2}\right)$ "sub-types": in subtype 1 (resp.,2), ${r}_{1}$ covers the left (resp.,right) endpoint of $h$ ,in subtype 3 (resp.,4), ${r}_{1}$ covers the bottom (resp.,top) endpoint of $v$ ,in subtype 5 (resp.,6), ${r}_{2}$ covers the left (resp.,right) endpoint of $h$ ,etc. It is possible for the result tuple to belong to multiple sub-types simultaneously. Next, we will focus on producing the result tuples of a particular sub-type:

如前所述，对于这种类型的结果元组$\left( {{r}_{1},\ldots ,{r}_{k - 2},h,v}\right)$，对于某个$i \in  \left\lbrack  {k - 2}\right\rbrack$，矩形${r}_{i}$覆盖$h$或$v$的一个端点，或者同时覆盖两者。由于(i) $i$有$k - 2$种选择，并且(ii) $h$和$v$总共具有四个端点，我们可以将类型2进一步划分为$4\left( {k - 2}\right)$个“子类型”：在子类型1（分别地，子类型2）中，${r}_{1}$覆盖$h$的左（分别地，右）端点；在子类型3（分别地，子类型4）中，${r}_{1}$覆盖$v$的下（分别地，上）端点；在子类型5（分别地，子类型6）中，${r}_{2}$覆盖$h$的左（分别地，右）端点，等等。结果元组有可能同时属于多个子类型。接下来，我们将专注于生成特定子类型的结果元组：

$$
{\mathcal{J}}_{2} = \left\{  {\left( {{r}_{1},\ldots ,{r}_{k - 2},h,v}\right)  \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 2},H,V}\right)  \mid  {r}_{k - 2}}\right. \text{covers the left endpoint of}\left. h\right\}  \text{.} \tag{11}
$$

The other sub-types can be found analogously.

其他子类型可以类似地找到。

A remark is in order about duplicate removal. By finding each sub-type separately, we may see the same result tuple multiple times (precisely,up to $4\left( {k - 2}\right)$ times) in the whole algorithm. However, this does not mean that the tuple needs to be reported multiple times. Whenever a type-2 result tuple is found,we can immediately decide in $O\left( k\right)$ time all the sub-types it belongs to. To avoid outputting the tuple more than once, we can enforce a policy to designate a specific sub-type for outputting. One such policy is the following: among all sub-types that the tuple belongs to, identify the one with the smallest sub-type number $t$ (an integer from 1 to $4\left( {k - 2}\right)$ ); report the tuple only when we are computing the particular sub-type $t$ .

有必要对去重问题做一个说明。通过分别查找每个子类型，在整个算法中我们可能会多次（准确地说，最多$4\left( {k - 2}\right)$次）看到相同的结果元组。然而，这并不意味着该元组需要被多次报告。每当找到一个2型结果元组时，我们可以在$O\left( k\right)$时间内立即确定它所属的所有子类型。为了避免多次输出该元组，我们可以实施一项策略，指定一个特定的子类型进行输出。其中一种策略如下：在该元组所属的所有子类型中，找出子类型编号$t$最小的那个（一个从1到$4\left( {k - 2}\right)$的整数）；仅当我们计算特定子类型$t$时才报告该元组。

Example 5.1. To illustrate our algorithm, we will utilize the running example in Figure 5a, where $k = 4$ ,and ${R}_{1} = \{ \alpha \} ,{R}_{2} = \left\{  {{\beta }_{1},{\beta }_{2},{\beta }_{3}}\right\}  ,H = \left\{  {{h}_{1},{h}_{2},{h}_{3}}\right\}$ ,and $V = \left\{  {{v}_{1},{v}_{2}}\right\}$ . The set ${\mathcal{J}}_{2}$ contains the following tuples: $\left( {\alpha ,{\beta }_{1},{h}_{2},{v}_{1}}\right) ,\left( {\alpha ,{\beta }_{1},{h}_{3},{v}_{1}}\right) ,\left( {\alpha ,{\beta }_{2},{h}_{3},{v}_{1}}\right) ,\left( {\alpha ,{\beta }_{3},{h}_{3},{v}_{1}}\right)$ ,and $\left( {\alpha ,{\beta }_{3},{h}_{3},{v}_{2}}\right)$ .

示例5.1。为了说明我们的算法，我们将使用图5a中的运行示例，其中$k = 4$，并且${R}_{1} = \{ \alpha \} ,{R}_{2} = \left\{  {{\beta }_{1},{\beta }_{2},{\beta }_{3}}\right\}  ,H = \left\{  {{h}_{1},{h}_{2},{h}_{3}}\right\}$，并且$V = \left\{  {{v}_{1},{v}_{2}}\right\}$。集合${\mathcal{J}}_{2}$包含以下元组：$\left( {\alpha ,{\beta }_{1},{h}_{2},{v}_{1}}\right) ,\left( {\alpha ,{\beta }_{1},{h}_{3},{v}_{1}}\right) ,\left( {\alpha ,{\beta }_{2},{h}_{3},{v}_{1}}\right) ,\left( {\alpha ,{\beta }_{3},{h}_{3},{v}_{1}}\right)$，以及$\left( {\alpha ,{\beta }_{3},{h}_{3},{v}_{2}}\right)$。

Set ${H}^{\prime }$ . Take any horizontal segment $h = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  y \in  H$ . Recall from Section 2 that a left-end covering rectangle of $h$ is a rectangle covering the left endpoint of $h$ . Let $p$ be the rightmost point on $h$ such that at least one left-end covering rectangle of $h$ in ${R}_{k - 2}$ covers $p$ . This $p$ exists if and only if $h$ has at least one left-end covering rectangle in ${R}_{k - 2}$ . If $p$ exists and has coordinates(x,y), we refer to the segment ${h}^{\prime } = \left\lbrack  {{x}_{1},x}\right\rbrack   \times  y$ as the trimmed segment of $h$ ; conversely,we call $h$ the full segment of ${h}^{\prime }$ .

设${H}^{\prime }$。取任意水平线段$h = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  y \in  H$。回顾第2节内容，$h$的左端点覆盖矩形是指覆盖$h$左端点的矩形。设$p$是$h$上最靠右的点，使得${R}_{k - 2}$中至少有一个$h$的左端点覆盖矩形覆盖$p$。当且仅当${R}_{k - 2}$中至少有一个$h$的左端点覆盖矩形时，这个$p$才存在。如果$p$存在且坐标为(x,y)，我们将线段${h}^{\prime } = \left\lbrack  {{x}_{1},x}\right\rbrack   \times  y$称为$h$的修剪线段；反之，我们称$h$为${h}^{\prime }$的完整线段。

Construct

构建

$$
{H}^{\prime } = \left\{  {{h}^{\prime } \mid  h \in  H}\right. \text{and its trimmed segment}\left. {{h}^{\prime }\text{exists}}\right\}  \text{.} \tag{12}
$$

The construction is an instance of Problem $\mathcal{C}$ (with $H$ and ${R}_{k - 2}$ as the input) and finishes in $O\left( {n\log n}\right)$ time based on the discussion in Section 2.

该构建是问题$\mathcal{C}$的一个实例（以$H$和${R}_{k - 2}$作为输入），根据第2节的讨论，它可以在$O\left( {n\log n}\right)$时间内完成。

Now,solve a(k - 1)-SJ problem on the input $\left\{  {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right\}$ using the algorithm $\mathcal{A}$ supplied (by Lemma 3.1). Let $\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right)$ represent the result of this(k - 1)-SJ,whose input size is at most $n$ . Given the lemma below (which is proved in Appendix C),we assert that $\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right)$ can be computed in ${F}_{k - 1}\left( {n,\mathrm{{OUT}}}\right)$ time.

现在，使用（由引理3.1提供的）算法$\mathcal{A}$对输入$\left\{  {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right\}$求解一个(k - 1)-SJ问题。设$\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right)$表示这个(k - 1)-SJ的结果，其输入规模至多为$n$。根据下面的引理（在附录C中证明），我们断言$\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right)$可以在${F}_{k - 1}\left( {n,\mathrm{{OUT}}}\right)$时间内计算出来。

LEMMA 5.1. $\left| {\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right) }\right|  \leq$ OUT.

引理5.1. $\left| {\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right) }\right|  \leq$ 结束。

Example 5.2. Segment ${h}_{1}$ has no left-covering rectangle in ${R}_{2}$ (see Figure 5a) and thus has no trimmed segment. Segment ${h}_{2}$ has one left-covering rectangle in ${R}_{2}$ ,which is ${\beta }_{1}$ . As the entire ${h}_{2}$ is covered by ${\beta }_{1}$ ,it is equivalent to its trimmed segment ${h}_{2}^{\prime }$ ; see Figure 5b. Segment ${h}_{3}$ has two left-covering rectangles in ${R}_{2}$ ,which are ${\beta }_{2}$ and ${\beta }_{3}$ . The right endpoint of its trimmed segment ${h}_{3}^{\prime }$ ,as shown in Figure 5b,is decided by the right edge of ${\beta }_{3}$ . Therefore, ${H}^{\prime } = \left\{  {{h}_{2}^{\prime },{h}_{3}^{\prime }}\right\}$ . It is clear from Figure 5b that $\mathcal{J}\left( {{R}_{1},{H}^{\prime },V}\right)$ has 3 tuples: ${\mathbf{t}}_{1} = \left( {\alpha ,{h}_{2}^{\prime },{v}_{1}}\right) ,{\mathbf{t}}_{2} = \left( {\alpha ,{h}_{3}^{\prime },{v}_{1}}\right)$ ,and ${\mathbf{t}}_{3} = \left( {\alpha ,{h}_{3}^{\prime },{v}_{2}}\right)$ .

示例5.2. 线段 ${h}_{1}$ 在 ${R}_{2}$ 中没有左覆盖矩形（见图5a），因此没有修剪后的线段。线段 ${h}_{2}$ 在 ${R}_{2}$ 中有一个左覆盖矩形，即 ${\beta }_{1}$ 。由于整个 ${h}_{2}$ 被 ${\beta }_{1}$ 覆盖，它等同于其修剪后的线段 ${h}_{2}^{\prime }$ ；见图5b。线段 ${h}_{3}$ 在 ${R}_{2}$ 中有两个左覆盖矩形，分别是 ${\beta }_{2}$ 和 ${\beta }_{3}$ 。其修剪后的线段 ${h}_{3}^{\prime }$ 的右端点，如图5b所示，由 ${\beta }_{3}$ 的右边缘决定。因此， ${H}^{\prime } = \left\{  {{h}_{2}^{\prime },{h}_{3}^{\prime }}\right\}$ 。从图5b可以清楚地看到， $\mathcal{J}\left( {{R}_{1},{H}^{\prime },V}\right)$ 有3个元组： ${\mathbf{t}}_{1} = \left( {\alpha ,{h}_{2}^{\prime },{v}_{1}}\right) ,{\mathbf{t}}_{2} = \left( {\alpha ,{h}_{3}^{\prime },{v}_{1}}\right)$ 和 ${\mathbf{t}}_{3} = \left( {\alpha ,{h}_{3}^{\prime },{v}_{2}}\right)$ 。

Generating ${\mathcal{J}}_{2}$ . Take any(k - 1)-tuple $t = \left( {{r}_{1},\ldots ,{r}_{k - 3},{h}^{\prime },v}\right)  \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right)$ . Note that ${B}_{t}$ - defined in (4) - is the point ${h}^{\prime } \cap  v$ (the intersection of ${h}^{\prime }$ and $v$ ). Suppose that ${h}^{\prime } = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  y$ and ${B}_{t} = \left( {x,y}\right)$ ; we define the effective horizontal segment of $t$ as the horizontal segment $\left\lbrack  {{x}_{1},x}\right\rbrack   \times  y$ . This allows us to define

生成 ${\mathcal{J}}_{2}$ 。任取一个(k - 1)元组 $t = \left( {{r}_{1},\ldots ,{r}_{k - 3},{h}^{\prime },v}\right)  \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right)$ 。注意，(4)中定义的 ${B}_{t}$ 是点 ${h}^{\prime } \cap  v$ （ ${h}^{\prime }$ 和 $v$ 的交点）。假设 ${h}^{\prime } = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  y$ 和 ${B}_{t} = \left( {x,y}\right)$ ；我们将 $t$ 的有效水平线段定义为水平线段 $\left\lbrack  {{x}_{1},x}\right\rbrack   \times  y$ 。这使我们能够定义

$$
{\operatorname{contain}}_{{R}_{k - 2}}\left( \mathbf{t}\right)  = \left\{  {r \in  {R}_{k - 2} \mid  r\text{ contains the effective horizontal segment of }\mathbf{t}}\right\}   \tag{13}
$$

The above should not be confused with (3), where the "contain" function takes a segment as the parameter, rather than a tuple. Example 5.3. Consider the tuples ${\mathbf{t}}_{1},{\mathbf{t}}_{2}$ ,and ${\mathbf{t}}_{3}$ of $\mathcal{J}\left( {{R}_{1},{H}^{\prime },V}\right)$ given in Example 5.2. For ${\mathbf{t}}_{1} =$ $\left( {\alpha ,{h}_{2}^{\prime },{v}_{1}}\right)$ ,its effective horizontal segment is DC (see Figure 5b). For ${t}_{2} = \left( {\alpha ,{h}_{3}^{\prime },{v}_{1}}\right)$ ,its effective horizontal segment is EB. For ${\mathbf{t}}_{3} = \left( {\alpha ,{h}_{3}^{\prime },{v}_{2}}\right)  \in  \mathcal{J}\left( {{R}_{1},{H}^{\prime },V}\right)$ ,its effective horizontal segment is EA. Accordingly,as can be seen from Figure 5c, ${\operatorname{contain}}_{{R}_{k - 2}}\left( {\mathbf{t}}_{1}\right)  = {\operatorname{contain}}_{{R}_{2}}\left( {\mathbf{t}}_{1}\right)  = \left\{  {\beta }_{1}\right\}  ,{\operatorname{contain}}_{{R}_{2}}\left( {\mathbf{t}}_{2}\right)  =$ $\left\{  {{\beta }_{1},{\beta }_{2},{\beta }_{3}}\right\}$ ,and ${\operatorname{contain}}_{{R}_{2}}\left( {\mathbf{t}}_{3}\right)  = \left\{  {\beta }_{3}\right\}$ .

上述内容不应与(3)混淆，在(3)中，“包含”函数的参数是一个线段，而非一个元组。示例5.3。考虑示例5.2中给出的$\mathcal{J}\left( {{R}_{1},{H}^{\prime },V}\right)$的元组${\mathbf{t}}_{1},{\mathbf{t}}_{2}$和${\mathbf{t}}_{3}$。对于${\mathbf{t}}_{1} =$ $\left( {\alpha ,{h}_{2}^{\prime },{v}_{1}}\right)$，其有效的水平线段是DC（见图5b）。对于${t}_{2} = \left( {\alpha ,{h}_{3}^{\prime },{v}_{1}}\right)$，其有效的水平线段是EB。对于${\mathbf{t}}_{3} = \left( {\alpha ,{h}_{3}^{\prime },{v}_{2}}\right)  \in  \mathcal{J}\left( {{R}_{1},{H}^{\prime },V}\right)$，其有效的水平线段是EA。因此，从图5c可以看出，${\operatorname{contain}}_{{R}_{k - 2}}\left( {\mathbf{t}}_{1}\right)  = {\operatorname{contain}}_{{R}_{2}}\left( {\mathbf{t}}_{1}\right)  = \left\{  {\beta }_{1}\right\}  ,{\operatorname{contain}}_{{R}_{2}}\left( {\mathbf{t}}_{2}\right)  =$ $\left\{  {{\beta }_{1},{\beta }_{2},{\beta }_{3}}\right\}$和${\operatorname{contain}}_{{R}_{2}}\left( {\mathbf{t}}_{3}\right)  = \left\{  {\beta }_{3}\right\}$。

We prove the next lemma in Appendix C (the reader may want to be reminded that,for each $t \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right) ,t\left\lbrack  {k - 2}\right\rbrack$ is a horizontal segment and $t\left\lbrack  {k - 1}\right\rbrack$ is a vertical segment).

我们在附录C中证明下一个引理（读者可能需要注意，对于每个$t \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right) ,t\left\lbrack  {k - 2}\right\rbrack$是水平线段，$t\left\lbrack  {k - 1}\right\rbrack$是垂直线段）。

LEMMA 5.2. All the following statements are true:

引理5.2。以下所有陈述均为真：

(1) Consider any(k - 1)-tuple $t \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right)$ . Denote by $h$ the full segment of $t\left\lbrack  {k - 2}\right\rbrack$ . Then,for any $r \in  {\operatorname{contain}}_{{R}_{k - 2}}\left( t\right)$ ,the $k$ -tuple $\left( {t\left\lbrack  1\right\rbrack  ,\ldots ,t\left\lbrack  {k - 3}\right\rbrack  ,r,h,t\left\lbrack  {k - 1}\right\rbrack  }\right)$ belongs to ${\mathcal{J}}_{2}$ .

(1) 考虑任意(k - 1)元组$t \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right)$。用$h$表示$t\left\lbrack  {k - 2}\right\rbrack$的完整线段。那么，对于任意$r \in  {\operatorname{contain}}_{{R}_{k - 2}}\left( t\right)$，$k$元组$\left( {t\left\lbrack  1\right\rbrack  ,\ldots ,t\left\lbrack  {k - 3}\right\rbrack  ,r,h,t\left\lbrack  {k - 1}\right\rbrack  }\right)$属于${\mathcal{J}}_{2}$。

(2) Consider any $k$ -tuple $\left( {{r}_{1},\ldots ,{r}_{k - 2},h,v}\right)  \in  {\mathcal{J}}_{2}$ . Let ${h}^{\prime }$ be the trimmed segment of $h$ and set $t =$ $\left( {{r}_{1},\ldots ,{r}_{k - 3},{h}^{\prime },v}\right)$ . Then, $t \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right)$ and ${r}_{k - 2} \in  {\operatorname{contain}}_{{R}_{k - 2}}\left( t\right)$ .

(2) 考虑任意$k$元组$\left( {{r}_{1},\ldots ,{r}_{k - 2},h,v}\right)  \in  {\mathcal{J}}_{2}$。设${h}^{\prime }$是$h$的修剪线段，并设$t =$ $\left( {{r}_{1},\ldots ,{r}_{k - 3},{h}^{\prime },v}\right)$。那么，$t \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right)$且${r}_{k - 2} \in  {\operatorname{contain}}_{{R}_{k - 2}}\left( t\right)$。

(3) $\mathop{\sum }\limits_{t}\left| {{\operatorname{contain}}_{{R}_{k - 2}}\left( t\right) }\right|  \leq  \mathrm{{OUT}}$ ,where the summation is over all $t \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right)$ .

(3) $\mathop{\sum }\limits_{t}\left| {{\operatorname{contain}}_{{R}_{k - 2}}\left( t\right) }\right|  \leq  \mathrm{{OUT}}$ ，其中求和是对所有 $t \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right)$ 进行的。

Equipped with Lemma 5.2,we generate our target ${\mathcal{J}}_{2}$ as follows:

借助引理5.2，我们按如下方式生成目标 ${\mathcal{J}}_{2}$ ：

algorithm generate- ${\mathcal{J}}_{2}$

算法生成 - ${\mathcal{J}}_{2}$

${\mathcal{J}}_{2} = \varnothing$

2. for each(k - 2)-tuple $\mathbf{t} \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right)$ do

2. 对每个(k - 2)元组 $\mathbf{t} \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right)$ 执行以下操作

3. $\;h \leftarrow$ the full segment of $\mathbf{t}\left\lbrack  {k - 2}\right\rbrack$

3. $\;h \leftarrow$ $\mathbf{t}\left\lbrack  {k - 2}\right\rbrack$ 的完整片段

4. for each $r \in  {\operatorname{contain}}_{{R}_{k - 2}}\left( t\right)$ do

4. 对每个 $r \in  {\operatorname{contain}}_{{R}_{k - 2}}\left( t\right)$ 执行以下操作

5. add $\left( {\mathbf{t}\left\lbrack  1\right\rbrack  ,\ldots ,\mathbf{t}\left\lbrack  {k - 3}\right\rbrack  ,r,h,\mathbf{t}\left\lbrack  {k - 1}\right\rbrack  }\right)$ to ${\mathcal{J}}_{2}$

5. 将 $\left( {\mathbf{t}\left\lbrack  1\right\rbrack  ,\ldots ,\mathbf{t}\left\lbrack  {k - 3}\right\rbrack  ,r,h,\mathbf{t}\left\lbrack  {k - 1}\right\rbrack  }\right)$ 添加到 ${\mathcal{J}}_{2}$ 中

The correctness of the algorithm follows from statements (1) and (2) of Lemma 5.2. Furthermore, if we are given ${\operatorname{contain}}_{{R}_{k - 2}}\left( \mathbf{t}\right)$ for each $\mathbf{t}$ ,statement (3) of Lemma 5.2 assures us that the algorithm runs in $O\left( {1 + k \cdot  \left| {\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right) }\right|  + k\mathop{\sum }\limits_{t}\left| {{\operatorname{contain}}_{{R}_{k - 2}}\left( t\right) }\right| }\right)  = O\left( {1 + k \cdot  \mathrm{{OUT}}}\right)$ time,where the derivation used Lemma 5.1 and statement (3) of Lemma 5.2.

该算法的正确性源自引理5.2的陈述(1)和(2)。此外，如果我们已知每个 $\mathbf{t}$ 对应的 ${\operatorname{contain}}_{{R}_{k - 2}}\left( \mathbf{t}\right)$ ，引理5.2的陈述(3)确保该算法的运行时间为 $O\left( {1 + k \cdot  \left| {\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right) }\right|  + k\mathop{\sum }\limits_{t}\left| {{\operatorname{contain}}_{{R}_{k - 2}}\left( t\right) }\right| }\right)  = O\left( {1 + k \cdot  \mathrm{{OUT}}}\right)$ ，其中推导过程使用了引理5.1和引理5.2的陈述(3)。

Example 5.4. For ${t}_{1} = \left( {\alpha ,{h}_{2}^{\prime },{v}_{1}}\right)$ ,the full segment of ${h}_{2}^{\prime }$ is ${h}_{2}$ . As ${\beta }_{1}$ is the only rectangle in $\overline{{\operatorname{contain}}_{{R}_{2}}\left( {t}_{1}\right) }$ ,Line 5 of the algorithm adds tuple $\left( {\alpha ,{\beta }_{1},{h}_{2},{v}_{1}}\right)$ to ${\mathcal{J}}_{2}$ . For ${t}_{2} = \left( {\alpha ,{h}_{3}^{\prime },{v}_{1}}\right)$ ,the full segment of ${h}_{3}^{\prime }$ is ${h}_{3}$ . As ${\operatorname{contain}}_{{R}_{2}}\left( {t}_{2}\right)  = \left\{  {{\beta }_{1},{\beta }_{2},{\beta }_{3}}\right\}$ ,Line 5 adds $\left( {\alpha ,{\beta }_{1},{h}_{3},{v}_{1}}\right) ,\left( {\alpha ,{\beta }_{2},{h}_{3},{v}_{1}}\right)$ ,and $\left( {\alpha ,{\beta }_{3},{h}_{3},{v}_{1}}\right)$ to ${\mathcal{J}}_{2}$ . Finally,the processing of ${\mathbf{t}}_{3} = \left( {\alpha ,{h}_{3}^{\prime },{v}_{2}}\right)$ adds $\left( {\alpha ,{\beta }_{3},{h}_{3},{v}_{2}}\right)$ to ${\mathcal{J}}_{2}$ .

示例5.4。对于${t}_{1} = \left( {\alpha ,{h}_{2}^{\prime },{v}_{1}}\right)$，${h}_{2}^{\prime }$的完整线段为${h}_{2}$。由于${\beta }_{1}$是$\overline{{\operatorname{contain}}_{{R}_{2}}\left( {t}_{1}\right) }$中唯一的矩形，该算法的第5行将元组$\left( {\alpha ,{\beta }_{1},{h}_{2},{v}_{1}}\right)$添加到${\mathcal{J}}_{2}$中。对于${t}_{2} = \left( {\alpha ,{h}_{3}^{\prime },{v}_{1}}\right)$，${h}_{3}^{\prime }$的完整线段为${h}_{3}$。由于${\operatorname{contain}}_{{R}_{2}}\left( {t}_{2}\right)  = \left\{  {{\beta }_{1},{\beta }_{2},{\beta }_{3}}\right\}$，第5行将$\left( {\alpha ,{\beta }_{1},{h}_{3},{v}_{1}}\right) ,\left( {\alpha ,{\beta }_{2},{h}_{3},{v}_{1}}\right)$和$\left( {\alpha ,{\beta }_{3},{h}_{3},{v}_{1}}\right)$添加到${\mathcal{J}}_{2}$中。最后，对${\mathbf{t}}_{3} = \left( {\alpha ,{h}_{3}^{\prime },{v}_{2}}\right)$的处理将$\left( {\alpha ,{\beta }_{3},{h}_{3},{v}_{2}}\right)$添加到${\mathcal{J}}_{2}$中。

Set ${H}^{ * }$ . For each segment ${h}^{\prime } \in  {H}^{\prime }$ ,define

设${H}^{ * }$。对于每个线段${h}^{\prime } \in  {H}^{\prime }$，定义

$$
\operatorname{minleft}\left( {h}^{\prime }\right)  = \mathop{\min }\limits_{\substack{{\mathbf{t} \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right)  : } \\  {\mathbf{t}\left\lbrack  {k - 2}\right\rbrack   = {h}^{\prime }} }}x\text{-coordinate of }\mathbf{t}\left\lbrack  {k - 1}\right\rbrack  . \tag{14}
$$

We set minleft $\left( {h}^{\prime }\right)$ to $\infty$ if no $t \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right)$ has ${h}^{\prime }$ in its field $t\left\lbrack  {k - 2}\right\rbrack$ . When minleft $\left( {h}^{\prime }\right)  \neq$ $\infty$ ,assuming ${h}^{\prime } = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  y$ ,we introduce a horizontal segment

如果没有$t \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right)$在其字段$t\left\lbrack  {k - 2}\right\rbrack$中包含${h}^{\prime }$，我们将minleft $\left( {h}^{\prime }\right)$设为$\infty$。当minleft $\left( {h}^{\prime }\right)  \neq$ $\infty$时，假设${h}^{\prime } = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  y$，我们引入一个水平线段

$$
{h}^{ * } = \left\lbrack  {{x}_{1},\operatorname{minleft}\left( {h}^{\prime }\right) }\right\rbrack   \times  y.
$$

and call it the minimal segment of ${h}^{\prime }$ .

并将其称为${h}^{\prime }$的最小线段。

Example 5.5. As mentioned, $\mathcal{J}\left( {{R}_{1},{H}^{\prime },V}\right)$ has 3 tuples ${\mathbf{t}}_{1},{\mathbf{t}}_{2}$ ,and ${\mathbf{t}}_{3}$ . Both ${\mathbf{t}}_{2} = \left( {\alpha ,{h}_{3}^{\prime },{v}_{1}}\right)$ and ${t}_{3} = \left( {\alpha ,{h}_{3}^{\prime },{v}_{2}}\right)$ have ${h}_{3}^{\prime }$ as the horizontal segment. Therefore,minleft $\left( {h}_{3}^{\prime }\right)$ equals 5,which is the smaller between the x-coordinate of ${v}_{1}$ and that of ${v}_{2}$ . The minimal segment ${h}_{3}^{ * }$ of ${h}_{3}^{\prime }$ is shown in Figure 5(c). On the other hand,it is easy to verify that minleft $\left( {h}_{2}^{\prime }\right)$ is the $\mathrm{x}$ -coordinate of ${v}_{1}$ . The minimal segment ${h}_{2}^{ * }$ of ${h}_{2}^{\prime }$ is also shown in Figure 5(c).

示例5.5。如前所述，$\mathcal{J}\left( {{R}_{1},{H}^{\prime },V}\right)$有3个元组${\mathbf{t}}_{1},{\mathbf{t}}_{2}$和${\mathbf{t}}_{3}$。${\mathbf{t}}_{2} = \left( {\alpha ,{h}_{3}^{\prime },{v}_{1}}\right)$和${t}_{3} = \left( {\alpha ,{h}_{3}^{\prime },{v}_{2}}\right)$都以${h}_{3}^{\prime }$作为水平线段。因此，minleft $\left( {h}_{3}^{\prime }\right)$等于5，这是${v}_{1}$和${v}_{2}$的x坐标中较小的那个。${h}_{3}^{\prime }$的最小线段${h}_{3}^{ * }$如图5(c)所示。另一方面，很容易验证minleft $\left( {h}_{2}^{\prime }\right)$是${v}_{1}$的$\mathrm{x}$坐标。${h}_{2}^{\prime }$的最小线段${h}_{2}^{ * }$也如图5(c)所示。

Next, we construct a new set of horizontal segments:

接下来，我们构建一组新的水平线段：

$$
{H}^{ * } = \left\{  {{h}^{ * } \mid  {h}^{\prime } \in  {H}^{\prime }\text{ and its minimal segment }{h}^{ * }\text{ exists }}\right\}  . \tag{15}
$$

This can be done in $O\left( {n + k \cdot  \mathrm{{OUT}}}\right)$ time,as shown in Appendix C.

如附录 C 所示，这可以在 $O\left( {n + k \cdot  \mathrm{{OUT}}}\right)$ 时间内完成。

We are interested in the sets ${\operatorname{contain}}_{{R}_{k - 2}}\left( {h}^{ * }\right)$ of the segments ${h}^{ * }$ in ${H}^{ * }$ ,where ${\operatorname{contain}}_{{R}_{k - 2}}\left( {h}^{ * }\right)  -$ defined in (3) - is the set of rectangles in ${R}_{k - 2}$ containing ${h}^{ * }$ . These sets have some useful properties:

我们关注 ${H}^{ * }$ 中线段 ${h}^{ * }$ 的集合 ${\operatorname{contain}}_{{R}_{k - 2}}\left( {h}^{ * }\right)$，其中 (3) 中定义的 ${\operatorname{contain}}_{{R}_{k - 2}}\left( {h}^{ * }\right)  -$ 是 ${R}_{k - 2}$ 中包含 ${h}^{ * }$ 的矩形集合。这些集合具有一些有用的性质：

LEMMA 5.3. Both statements below are true:

引理 5.3。以下两个陈述均为真：

(1) $\mathop{\sum }\limits_{{{h}^{ * } \in  {H}^{ * }}}\left| {{\operatorname{contain}}_{{R}_{k - 2}}\left( {h}^{ * }\right) }\right|  \leq$ OUT.

(1) $\mathop{\sum }\limits_{{{h}^{ * } \in  {H}^{ * }}}\left| {{\operatorname{contain}}_{{R}_{k - 2}}\left( {h}^{ * }\right) }\right|  \leq$ 输出。

(2) Consider any tuple $t \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right)$ . Set ${h}^{\prime } = t\left\lbrack  {k - 2}\right\rbrack$ and let ${h}^{ * }$ be the minimal segment of ${h}^{\prime }$ . Then,

(2) 考虑任意元组 $t \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right)$。设 ${h}^{\prime } = t\left\lbrack  {k - 2}\right\rbrack$，并令 ${h}^{ * }$ 为 ${h}^{\prime }$ 的最小线段。那么，

$$
{\operatorname{contain}}_{{R}_{k - 2}}\left( \mathbf{t}\right)  \subseteq  {\operatorname{contain}}_{{R}_{k - 2}}\left( {h}^{ * }\right) \text{.}
$$

Furthermore,if the rectangles $r$ in ${\operatorname{contain}}_{{R}_{k - 2}}\left( {h}^{ * }\right)$ are sorted in descending order of $\operatorname{right}\left( r\right)$ , then ${\operatorname{contain}}_{{R}_{k - 2}}\left( t\right)$ includes a prefix of the sorted order.

此外，如果 ${\operatorname{contain}}_{{R}_{k - 2}}\left( {h}^{ * }\right)$ 中的矩形按 $\operatorname{right}\left( r\right)$ 降序排序，那么 ${\operatorname{contain}}_{{R}_{k - 2}}\left( t\right)$ 包含排序顺序的一个前缀。

The proof can be found in Appendix C.

证明可在附录 C 中找到。

Example 5.6. It is clear from Figure 5(d) that ${\operatorname{contain}}_{{R}_{2}}\left( {h}_{3}^{ * }\right)$ has rectangles ${\beta }_{3},{\beta }_{2},{\beta }_{1}$ ,sorted in descending order of their right boundaries’ $\mathrm{x}$ -coordinates. Consider ${\mathbf{t}}_{2} = \left( {\alpha ,{h}_{3}^{\prime },{v}_{1}}\right)$ and ${\mathbf{t}}_{3} = \left( {\alpha ,{h}_{3}^{\prime },{v}_{2}}\right)$ . Segment ${h}_{3}^{ * }$ is the minimal segment of ${h}_{3}^{\prime }$ . As stated in Lemma 5.3,both ${\operatorname{contain}}_{{R}_{2}}\left( {t}_{2}\right)  = \left\{  {{\beta }_{1},{\beta }_{2},{\beta }_{3}}\right\}$ and ${\operatorname{contain}}_{{R}_{2}}\left( {\mathbf{t}}_{3}\right)  = \left\{  {\beta }_{1}\right\}$ are prefixes of the sorted order of ${\operatorname{contain}}_{{R}_{2}}\left( {h}_{3}^{ * }\right)$ . Regarding ${h}_{2}^{ * }$ ,it is the minimal segment of ${h}_{2}^{\prime }$ ,and contain ${R}_{{R}_{2}}\left( {h}_{2}^{ * }\right)$ contains only ${\beta }_{1}$ . For ${\mathbf{t}}_{1} = \left( {\alpha ,{h}_{2}^{\prime },{v}_{1}}\right) ,{\operatorname{contain}}_{{R}_{2}}\left( {\mathbf{t}}_{1}\right)  = \left\{  {\beta }_{1}\right\}$ is a (trivial) prefix of ${\operatorname{contain}}_{{R}_{2}}\left( {h}_{2}^{ * }\right)$ ,as is also consistent with the lemma.

示例 5.6。从图 5(d) 可以清楚地看出，${\operatorname{contain}}_{{R}_{2}}\left( {h}_{3}^{ * }\right)$ 有矩形 ${\beta }_{3},{\beta }_{2},{\beta }_{1}$，按其右边界的 $\mathrm{x}$ 坐标降序排序。考虑 ${\mathbf{t}}_{2} = \left( {\alpha ,{h}_{3}^{\prime },{v}_{1}}\right)$ 和 ${\mathbf{t}}_{3} = \left( {\alpha ,{h}_{3}^{\prime },{v}_{2}}\right)$。线段 ${h}_{3}^{ * }$ 是 ${h}_{3}^{\prime }$ 的最小线段。如引理 5.3 所述，${\operatorname{contain}}_{{R}_{2}}\left( {t}_{2}\right)  = \left\{  {{\beta }_{1},{\beta }_{2},{\beta }_{3}}\right\}$ 和 ${\operatorname{contain}}_{{R}_{2}}\left( {\mathbf{t}}_{3}\right)  = \left\{  {\beta }_{1}\right\}$ 都是 ${\operatorname{contain}}_{{R}_{2}}\left( {h}_{3}^{ * }\right)$ 排序顺序的前缀。关于 ${h}_{2}^{ * }$，它是 ${h}_{2}^{\prime }$ 的最小线段，并且集合 ${R}_{{R}_{2}}\left( {h}_{2}^{ * }\right)$ 仅包含 ${\beta }_{1}$。因为 ${\mathbf{t}}_{1} = \left( {\alpha ,{h}_{2}^{\prime },{v}_{1}}\right) ,{\operatorname{contain}}_{{R}_{2}}\left( {\mathbf{t}}_{1}\right)  = \left\{  {\beta }_{1}\right\}$ 是 ${\operatorname{contain}}_{{R}_{2}}\left( {h}_{2}^{ * }\right)$ 的（平凡）前缀，这也与引理一致。

Finding the contain ${R}_{k - 2}\left( {h}^{ * }\right)$ sets of all ${h}^{ * } \in  {H}^{ * }$ is an instance of Problem $\mathcal{E}$ (with ${H}^{ * }$ and ${R}_{k - 2}$ as the input). The cost is $O\left( {n\log n + \mathrm{{OUT}}}\right)$ according statement (1) of Lemma 4.3 and the discussion in Section 2. Note that,for each ${\operatorname{contain}}_{{R}_{k - 2}}\left( {h}^{ * }\right)$ computed,the rectangles $r$ therein have been sorted in descending order of right(r).

找出所有${h}^{ * } \in  {H}^{ * }$的包含${R}_{k - 2}\left( {h}^{ * }\right)$集合是问题$\mathcal{E}$的一个实例（以${H}^{ * }$和${R}_{k - 2}$作为输入）。根据引理4.3的陈述(1)和第2节的讨论，成本为$O\left( {n\log n + \mathrm{{OUT}}}\right)$。注意，对于计算出的每个${\operatorname{contain}}_{{R}_{k - 2}}\left( {h}^{ * }\right)$，其中的矩形$r$已按右边界r降序排序。

Computing the "contain ${R}_{k - 2}\left( t\right)$ " Sets. Statement (2) of Lemma 5.3 allows us to produce contain ${R}_{k - 2}\left( t\right)$ - defined in (13) - for each $t \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right)$ as follows. First,fetch the (already computed) minimal segment ${h}^{ * }$ of $t\left\lbrack  {k - 2}\right\rbrack$ in $O\left( 1\right)$ time. Then,scan the rectangles $r$ of contain ${R}_{k - 2}\left( {h}^{ * }\right)$ in descending order of right(r). For each $r$ scanned,check whether $r \in  {\operatorname{contain}}_{{R}_{k - 2}}\left( t\right)$ ,or equivalently, whether $r$ covers ${B}_{t}$ (recall that ${B}_{t}$ is a point); the cost of this inspection is $O\left( 1\right)$ . Abort the scan as soon as $r \notin  {\operatorname{contain}}_{{R}_{k - 2}}\left( t\right)$ . This way,contain ${R}_{k - 2}\left( t\right)$ can be decided in $O\left( {k + \left| {{\operatorname{contain}}_{{R}_{k - 2}}\left( t\right) }\right| }\right)$ time. Doing so for all $t \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right)$ takes $O\left( {k \cdot  \left| \mathcal{J}\right|  + \mathop{\sum }\limits_{t}\left| {{\operatorname{contain}}_{{R}_{k - 2}}\left( t\right) }\right| }\right)  = O\left( {k \cdot  \mathrm{{OUT}}}\right)$ time, where the derivation used Lemma 5.1 and statement (3) of Lemma 5.2.

计算“包含${R}_{k - 2}\left( t\right)$”集合。引理5.3的陈述(2)使我们能够按如下方式为每个$t \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right)$生成(13)中定义的包含${R}_{k - 2}\left( t\right)$集合。首先，在$O\left( 1\right)$时间内获取$t\left\lbrack  {k - 2}\right\rbrack$在$O\left( 1\right)$中的（已计算出的）最小线段${h}^{ * }$。然后，按右边界r降序扫描包含${R}_{k - 2}\left( {h}^{ * }\right)$的矩形$r$。对于扫描到的每个$r$，检查是否满足$r \in  {\operatorname{contain}}_{{R}_{k - 2}}\left( t\right)$，或者等价地，检查$r$是否覆盖${B}_{t}$（回想一下，${B}_{t}$是一个点）；此检查的成本为$O\left( 1\right)$。一旦满足$r \notin  {\operatorname{contain}}_{{R}_{k - 2}}\left( t\right)$，则停止扫描。这样，包含${R}_{k - 2}\left( t\right)$集合可以在$O\left( {k + \left| {{\operatorname{contain}}_{{R}_{k - 2}}\left( t\right) }\right| }\right)$时间内确定。对所有$t \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right)$执行此操作需要$O\left( {k \cdot  \left| \mathcal{J}\right|  + \mathop{\sum }\limits_{t}\left| {{\operatorname{contain}}_{{R}_{k - 2}}\left( t\right) }\right| }\right)  = O\left( {k \cdot  \mathrm{{OUT}}}\right)$时间，其中推导使用了引理5.1和引理5.2的陈述(3)。

We conclude that ${\mathcal{J}}_{2} -$ see (11)-can be computed in ${F}_{k - 1}\left( {n,\mathrm{{OUT}}}\right)  + O\left( {n\log n + k \cdot  \mathrm{{OUT}}}\right)$ time. Remember that,to generate the entire type-2 result,we need to repeat the algorithm $4\left( {k - 2}\right)$ times (one for each sub-type). The total running time is therefore $O\left( k\right)  \cdot  \left( {{F}_{k - 1}\left( {n,\mathrm{{OUT}}}\right)  + n\log n + k \cdot  \mathrm{{OUT}}}\right)$ , as claimed in Lemma 3.1.

我们得出结论，${\mathcal{J}}_{2} -$（见(11)）可以在${F}_{k - 1}\left( {n,\mathrm{{OUT}}}\right)  + O\left( {n\log n + k \cdot  \mathrm{{OUT}}}\right)$时间内计算得出。请记住，为了生成整个2型结果，我们需要将该算法重复执行$4\left( {k - 2}\right)$次（每个子类型执行一次）。因此，总运行时间为$O\left( k\right)  \cdot  \left( {{F}_{k - 1}\left( {n,\mathrm{{OUT}}}\right)  + n\log n + k \cdot  \mathrm{{OUT}}}\right)$，如引理3.1所述。

## 6 Settling $k$ -SJ

## 6 解决$k$ -SJ问题

This section will tackle the $k$ -SJ problem in its general form,where the input comprises $k \geq  3$ sets of rectangles ${R}_{1},{R}_{2},\ldots ,{R}_{k}$ . The join result $\mathcal{J}\left( {{R}_{1},{R}_{2},\ldots ,{R}_{k}}\right)$ is the set of $k$ -tuples $\mathbf{t} = \left( {{r}_{1},{r}_{2},\ldots ,{r}_{k}}\right)  \in$ ${R}_{1} \times  {R}_{2} \times  \ldots  \times  {R}_{k}$ satisfying the condition that ${B}_{t} -$ which is $\mathop{\bigcap }\limits_{{i = 1}}^{k}{r}_{i} \neq  \varnothing$ (see (4)) - is non-empty.

本节将处理一般形式的 $k$ -SJ 问题，其中输入包含 $k \geq  3$ 组矩形 ${R}_{1},{R}_{2},\ldots ,{R}_{k}$。连接结果 $\mathcal{J}\left( {{R}_{1},{R}_{2},\ldots ,{R}_{k}}\right)$ 是满足条件 ${B}_{t} -$（即 $\mathop{\bigcap }\limits_{{i = 1}}^{k}{r}_{i} \neq  \varnothing$（见 (4)））非空的 $k$ 元组 $\mathbf{t} = \left( {{r}_{1},{r}_{2},\ldots ,{r}_{k}}\right)  \in$ ${R}_{1} \times  {R}_{2} \times  \ldots  \times  {R}_{k}$ 的集合。

Consider any result tuple $t = \left( {{r}_{1},{r}_{2},\ldots ,{r}_{k}}\right)  \in  \mathcal{J}\left( {{R}_{1},{R}_{2},\ldots ,{R}_{k}}\right)$ ,and let $p$ be the top-left corner of ${B}_{t}$ . Depending on how $p$ is determined,we classify $t$ into one of the two categories below:

考虑任意结果元组 $t = \left( {{r}_{1},{r}_{2},\ldots ,{r}_{k}}\right)  \in  \mathcal{J}\left( {{R}_{1},{R}_{2},\ldots ,{R}_{k}}\right)$，并设 $p$ 为 ${B}_{t}$ 的左上角。根据 $p$ 的确定方式，我们将 $t$ 分为以下两类之一：

- Cat. 1: $p$ is the top-left corner of ${r}_{i}$ for some $i \in  \left\lbrack  k\right\rbrack$ .

- 类别 1：对于某个 $i \in  \left\lbrack  k\right\rbrack$，$p$ 是 ${r}_{i}$ 的左上角。

- Cat. 2: $p$ is not a corner of any of ${r}_{1},\ldots ,{r}_{k}$ . This means $p$ must be the intersection point between the top edge of some rectangle ${r}_{i}$ and the left edge of another rectangle ${r}_{j}$ ,where $i,j \in  \left\lbrack  k\right\rbrack$ and $i \neq  j$ .

- 类别 2：$p$ 不是任何 ${r}_{1},\ldots ,{r}_{k}$ 的角点。这意味着 $p$ 必须是某个矩形 ${r}_{i}$ 的上边缘与另一个矩形 ${r}_{j}$ 的左边缘的交点，其中 $i,j \in  \left\lbrack  k\right\rbrack$ 且 $i \neq  j$。

Figure 6 illustrates a tuple of each category,assuming $k = 4$ .

图 6 展示了假设 $k = 4$ 时每一类的一个元组。

<!-- Media -->

<!-- figureText: ${r}_{2}$ ${r}_{1}$ ${r}_{2}$ ${r}_{4}$ ${r}_{3}$ (b) Category 2 ${r}_{4}$ (a) Category 1 -->

<img src="https://cdn.noedgeai.com/0195ccc5-d2d9-7daa-9177-3ae04293d71f_13.jpg?x=428&y=262&w=710&h=315&r=0"/>

Fig. 6. Classifying $k$ -SJ result tuples $\left( {k = 4}\right)$

图 6. 对 $k$ -SJ 结果元组 $\left( {k = 4}\right)$ 进行分类

<!-- Media -->

The rest of this section serves as a proof of Theorem 1.1. Theorem 1.2 is a corollary of Theorem 1.1, as proved in Appendix D. As stated in Theorem 1.1,we are given an algorithm $\mathcal{A}$ that can solve any(k - 1)-SJ problem in ${F}_{k - 1}\left( {n,\mathrm{{OUT}}}\right)$ time,where $n$ and OUT are the input and output sizes, respectively. Equipped with $\mathcal{A}$ ,we will show how to find the result tuples of each category within the time complexity of (1).

本节的其余部分用于证明定理 1.1。定理 1.2 是定理 1.1 的推论，如附录 D 中所证明。如定理 1.1 所述，我们有一个算法 $\mathcal{A}$，它可以在 ${F}_{k - 1}\left( {n,\mathrm{{OUT}}}\right)$ 时间内解决任何 (k - 1) -SJ 问题，其中 $n$ 和 OUT 分别是输入和输出的规模。借助 $\mathcal{A}$，我们将展示如何在 (1) 的时间复杂度内找到每一类的结果元组。

Category 1. Given an $i \in  \left\lbrack  k\right\rbrack$ ,we denote by ${\mathcal{T}}_{i}^{\text{cat 1 }}$ the set of $k$ -tuples $\mathbf{t} = \left( {{r}_{1},\ldots ,{r}_{k}}\right)  \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k}}\right)$ such that the top-left corner of ${B}_{t}$ is the top-left corner of $t\left\lbrack  i\right\rbrack   = {r}_{i}$ . We will show how to compute ${\mathcal{J}}_{i}^{\text{cat1 }}$ for $i = k$ ; the set ${\mathcal{J}}_{i}^{\text{cat1 }}$ of every other $i$ can be produced in the same manner.

类别 1。给定一个 $i \in  \left\lbrack  k\right\rbrack$，我们用 ${\mathcal{T}}_{i}^{\text{cat 1 }}$ 表示 $k$ 元组 $\mathbf{t} = \left( {{r}_{1},\ldots ,{r}_{k}}\right)  \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k}}\right)$ 的集合，使得 ${B}_{t}$ 的左上角是 $t\left\lbrack  i\right\rbrack   = {r}_{i}$ 的左上角。我们将展示如何为 $i = k$ 计算 ${\mathcal{J}}_{i}^{\text{cat1 }}$；其他每个 $i$ 的集合 ${\mathcal{J}}_{i}^{\text{cat1 }}$ 可以用相同的方式生成。

For every $\mathbf{t} = \left( {{r}_{1},{r}_{2},\ldots ,{r}_{k}}\right)  \in  {\mathcal{J}}_{k}^{\text{cat 1 }}$ ,the top-left corner of ${r}_{k}$ must be covered by all of ${r}_{1},\ldots ,{r}_{k - 1}$ . This observation motivates us to find ${\mathcal{J}}_{k}^{\text{cat1 }}$ as follows. First,collect the set $P$ of top-left corners of all the rectangles in ${R}_{k}$ . Remove from $P$ every point $p$ with the property that,there exists at least one $j \in  \left\lbrack  {k - 1}\right\rbrack$ such that $p$ is covered by no rectangle in ${R}_{j}$ . This requires solving $k - 1$ instances of the detection version of Problem $\mathcal{A}$ (in each instance,the input includes $P$ together with a different ${R}_{j},j \in  \left\lbrack  {k - 1}\right\rbrack  )$ ; the cost is $O\left( {{kn}\log n}\right)$ by the discussion in Section 2.

对于每一个$\mathbf{t} = \left( {{r}_{1},{r}_{2},\ldots ,{r}_{k}}\right)  \in  {\mathcal{J}}_{k}^{\text{cat 1 }}$，${r}_{k}$的左上角必须被所有的${r}_{1},\ldots ,{r}_{k - 1}$覆盖。这一观察结果促使我们按如下方式找到${\mathcal{J}}_{k}^{\text{cat1 }}$。首先，收集${R}_{k}$中所有矩形的左上角集合$P$。从$P$中移除每一个点$p$，该点满足以下性质：存在至少一个$j \in  \left\lbrack  {k - 1}\right\rbrack$，使得$p$未被${R}_{j}$中的任何矩形覆盖。这需要求解问题$\mathcal{A}$检测版本的$k - 1$个实例（在每个实例中，输入包括$P$以及一个不同的${R}_{j},j \in  \left\lbrack  {k - 1}\right\rbrack  )$；根据第2节的讨论，成本为$O\left( {{kn}\log n}\right)$）。

Let ${P}^{\prime }$ be the set of remaining points in $P$ after the aforementioned removal. Next,for each $j \in  \left\lbrack  {k - 1}\right\rbrack$ ,solve the reporting version of Problem $\mathcal{A}$ by feeding ${P}^{\prime }$ and ${R}_{j}$ as the input. This produces the set ${\operatorname{contain}}_{{R}_{j}}\left( p\right)$ for each point $p \in  {P}^{\prime }$ ,where ${\operatorname{contain}}_{{R}_{j}}\left( p\right)$ is defined in (3) (treating $p$ as a degenerated "horizontal segment") and includes all rectangles of ${R}_{j}$ covering $p$ . By the discussion in Section 2, the total cost of this step is bounded by

设${P}^{\prime }$为经过上述移除操作后$P$中剩余点的集合。接下来，对于每个$j \in  \left\lbrack  {k - 1}\right\rbrack$，将${P}^{\prime }$和${R}_{j}$作为输入，求解问题$\mathcal{A}$的报告版本。这为每个点$p \in  {P}^{\prime }$生成集合${\operatorname{contain}}_{{R}_{j}}\left( p\right)$，其中${\operatorname{contain}}_{{R}_{j}}\left( p\right)$在(3)中定义（将$p$视为退化的“水平线段”），并且包含${R}_{j}$中覆盖$p$的所有矩形。根据第2节的讨论，此步骤的总成本有界为

$$
O\left( {{kn}\log n + \mathop{\sum }\limits_{{p \in  {P}^{\prime }}}\mathop{\sum }\limits_{{j \in  \left\lbrack  {k - 1}\right\rbrack  }}\left| {{\operatorname{contain}}_{{R}_{j}}\left( p\right) }\right| }\right) . \tag{16}
$$

We are ready to generate ${\mathcal{J}}_{k}^{\text{cat1 }}$ . Take any point $p \in  {P}^{\prime }$ ,and let $r \in  {R}_{k}$ be the rectangle with $p$ as the top-left corner. For every(k - 1)-tuple

我们准备生成${\mathcal{J}}_{k}^{\text{cat1 }}$。取任意一点$p \in  {P}^{\prime }$，并设$r \in  {R}_{k}$是以$p$为左上角的矩形。对于每一个(k - 1)元组

$$
\left( {{r}_{1},\ldots ,{r}_{k - 1}}\right)  \in  {\operatorname{contain}}_{{R}_{1}}\left( p\right)  \times  \ldots  \times  {\operatorname{contain}}_{{R}_{k - 1}}\left( p\right) 
$$

we add $\left( {{r}_{1},\ldots ,{r}_{k - 1},r}\right)$ to ${\mathcal{J}}_{k}^{\text{cat1 }}$ . Performing the above for all $p \in  {P}^{\prime }$ generates the whole ${\mathcal{J}}_{k}^{\text{cat1 }}$ in $O\left( {1 + k \cdot  \left| {\mathcal{S}}_{k}^{\text{cat1 }}\right| }\right)$ time. The way ${P}^{\prime }$ is computed ensures that ${\operatorname{contain}}_{{R}_{j}}\left( p\right)  \neq  \varnothing$ for each $j \in  \left\lbrack  {k - 1}\right\rbrack$ . Hence, $\mathop{\sum }\limits_{{j \in  \left\lbrack  {k - 1}\right\rbrack  }}\left| {{\operatorname{contain}}_{{R}_{j}}\left( p\right) }\right|  \leq  \mathop{\prod }\limits_{{j \in  \left\lbrack  {k - 1}\right\rbrack  }}\left| {{\operatorname{contain}}_{{R}_{j}}\left( p\right) }\right|$ ,which implies that (16) is bounded by $O\left( {{kn}\log n + k \cdot  \left| {\mathcal{J}}_{k}^{\text{cat1 }}\right| }\right)  = O\left( {{kn}\log n + k \cdot  \mathrm{{OUT}}}\right) .$

我们将$\left( {{r}_{1},\ldots ,{r}_{k - 1},r}\right)$加到${\mathcal{J}}_{k}^{\text{cat1 }}$上。对所有的$p \in  {P}^{\prime }$执行上述操作，能在$O\left( {1 + k \cdot  \left| {\mathcal{S}}_{k}^{\text{cat1 }}\right| }\right)$时间内生成整个${\mathcal{J}}_{k}^{\text{cat1 }}$。${P}^{\prime }$的计算方式确保了对于每个$j \in  \left\lbrack  {k - 1}\right\rbrack$都有${\operatorname{contain}}_{{R}_{j}}\left( p\right)  \neq  \varnothing$。因此，$\mathop{\sum }\limits_{{j \in  \left\lbrack  {k - 1}\right\rbrack  }}\left| {{\operatorname{contain}}_{{R}_{j}}\left( p\right) }\right|  \leq  \mathop{\prod }\limits_{{j \in  \left\lbrack  {k - 1}\right\rbrack  }}\left| {{\operatorname{contain}}_{{R}_{j}}\left( p\right) }\right|$，这意味着(16)以$O\left( {{kn}\log n + k \cdot  \left| {\mathcal{J}}_{k}^{\text{cat1 }}\right| }\right)  = O\left( {{kn}\log n + k \cdot  \mathrm{{OUT}}}\right) .$为界

Therefore,the total time of computing all of ${\mathcal{J}}_{1}^{\text{cat1 }},\ldots ,{\mathcal{J}}_{k}^{\text{cat1 }}$ is $O\left( k\right)  \cdot  \left( {{kn}\log n + k \cdot  \mathrm{{OUT}}}\right)$ . A category-1 result tuple $t$ may be seen more than once (this happens if the top-left corner of ${B}_{t}$ is the top-left corner of more than one rectangle in $t$ ). Duplicate removal can be implemented at no extra cost asymptotically, following the ideas explained in Section 5.

因此，计算所有${\mathcal{J}}_{1}^{\text{cat1 }},\ldots ,{\mathcal{J}}_{k}^{\text{cat1 }}$的总时间为$O\left( k\right)  \cdot  \left( {{kn}\log n + k \cdot  \mathrm{{OUT}}}\right)$。一个1类结果元组$t$可能会被多次看到（如果${B}_{t}$的左上角是$t$中多个矩形的左上角，就会发生这种情况）。根据第5节中解释的思路，可以在渐进意义上不增加额外成本地实现去重。

Category 2. Given $i,j \in  \left\lbrack  k\right\rbrack$ with $i \neq  j$ ,we denote by ${\mathcal{J}}_{i,j}$ the set of $k$ -tuples $t = \left( {{r}_{1},\ldots ,{r}_{k}}\right)$ $\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k}}\right)$ such that the top-left corner of ${B}_{t}$ is the intersection between the top edge of ${r}_{i}$ and the left edge of ${r}_{j}$ . The Category 2 of result tuples is the union of the ${\mathcal{T}}_{i,j}^{\text{cat }2}$ of all possible $i,j$ .

2类。给定具有$i \neq  j$的$i,j \in  \left\lbrack  k\right\rbrack$，我们用${\mathcal{J}}_{i,j}$表示$k$元组$t = \left( {{r}_{1},\ldots ,{r}_{k}}\right)$ $\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k}}\right)$的集合，使得${B}_{t}$的左上角是${r}_{i}$的上边缘和${r}_{j}$的左边缘的交点。结果元组的2类是所有可能的$i,j$的${\mathcal{T}}_{i,j}^{\text{cat }2}$的并集。

The computation of ${\mathcal{J}}_{i,j}^{\mathrm{{cat}}2}$ is an instance of the H-V $k$ -SJ problem. Specifically,collect the top-edges of all rectangles of ${R}_{i}$ into a set $H$ ,and collect the left-edges of all the rectangles of ${R}_{j}$ into a set $V$ . This yields an H-V $k$ -SJ instance whose input comprises all the ${R}_{z}$ with $z \in  \left\lbrack  k\right\rbrack   \smallsetminus  \{ i,j\} ,H$ , and $V$ . Each result tuple consists of a rectangle ${r}_{z} \in  {R}_{z}$ ,for $z \in  \left\lbrack  k\right\rbrack   \smallsetminus  \{ i,j\}$ ,a horizontal segment $h \in  H$ ,and a vertical segment $v \in  V$ such that $h \cap  v \cap  \mathop{\bigcap }\limits_{{z \in  \left\lbrack  k\right\rbrack  \smallsetminus \{ i,j\} }}{r}_{z} \neq  \varnothing$ . There is one-one correspondence between the output of the H-V $k$ -SJ and ${\mathcal{J}}_{i,j}^{\text{cat2 }}$ . Thus,by Lemma 3.1,the H-V $k$ -SJ can be solved in $O\left( k\right)  \cdot  \left( {{F}_{k - 1}\left( {n,\left| {\mathcal{J}}_{i,j}^{\text{cat2 }}\right| }\right)  + n\log n + k \cdot  \left| {\mathcal{J}}_{i,j}^{\text{cat2 }}\right| }\right)$ time. Converting the output into ${\mathcal{J}}_{i,j}^{\text{cat2 }}$ takes another $O\left( {k \cdot  \left| {\mathcal{J}}_{i,j}^{\mathrm{{cat}}2}\right| }\right)$ time. Applying $\left| {\mathcal{J}}_{i,j}^{\mathrm{{cat}}2}\right|  \leq  \mathrm{{OUT}}$ ,we know that ${\mathcal{J}}_{i,j}^{\mathrm{{cat}}2}$ can be produced in $O\left( k\right)  \cdot  \left( {{F}_{k - 1}\left( {n,\mathrm{{OUT}}}\right)  + n\log n + k \cdot  \mathrm{{OUT}}}\right)$ time

${\mathcal{J}}_{i,j}^{\mathrm{{cat}}2}$的计算是H-V $k$ -SJ问题的一个实例。具体来说，将${R}_{i}$中所有矩形的上边缘收集到集合$H$中，并将${R}_{j}$中所有矩形的左边缘收集到集合$V$中。这就得到了一个H-V $k$ -SJ实例，其输入包括所有带有$z \in  \left\lbrack  k\right\rbrack   \smallsetminus  \{ i,j\} ,H$的${R}_{z}$以及$V$。每个结果元组由一个矩形${r}_{z} \in  {R}_{z}$（对于$z \in  \left\lbrack  k\right\rbrack   \smallsetminus  \{ i,j\}$）、一个水平线段$h \in  H$和一个垂直线段$v \in  V$组成，使得$h \cap  v \cap  \mathop{\bigcap }\limits_{{z \in  \left\lbrack  k\right\rbrack  \smallsetminus \{ i,j\} }}{r}_{z} \neq  \varnothing$成立。H-V $k$ -SJ的输出与${\mathcal{J}}_{i,j}^{\text{cat2 }}$之间存在一一对应关系。因此，根据引理3.1，H-V $k$ -SJ问题可以在$O\left( k\right)  \cdot  \left( {{F}_{k - 1}\left( {n,\left| {\mathcal{J}}_{i,j}^{\text{cat2 }}\right| }\right)  + n\log n + k \cdot  \left| {\mathcal{J}}_{i,j}^{\text{cat2 }}\right| }\right)$时间内解决。将输出转换为${\mathcal{J}}_{i,j}^{\text{cat2 }}$还需要$O\left( {k \cdot  \left| {\mathcal{J}}_{i,j}^{\mathrm{{cat}}2}\right| }\right)$时间。应用$\left| {\mathcal{J}}_{i,j}^{\mathrm{{cat}}2}\right|  \leq  \mathrm{{OUT}}$，我们知道${\mathcal{J}}_{i,j}^{\mathrm{{cat}}2}$可以在$O\left( k\right)  \cdot  \left( {{F}_{k - 1}\left( {n,\mathrm{{OUT}}}\right)  + n\log n + k \cdot  \mathrm{{OUT}}}\right)$时间内生成

Performing the above for all $i,j \in  \left\lbrack  k\right\rbrack$ with $i \neq  j$ leads to a total time complexity of $O\left( {k}^{3}\right)$ . $\left( {{F}_{k - 1}\left( {n,\mathrm{{OUT}}}\right)  + n\log n + k \cdot  \mathrm{{OUT}})}\right.$ . A category-2 result tuple $t$ may be seen more than once (this can happen if,for example,more than one rectangle in $t$ has the same top-edge). Again,duplicate removal can be achieved at no extra cost asymptotically.

对所有满足$i \neq  j$的$i,j \in  \left\lbrack  k\right\rbrack$执行上述操作，总的时间复杂度为$O\left( {k}^{3}\right)$。$\left( {{F}_{k - 1}\left( {n,\mathrm{{OUT}}}\right)  + n\log n + k \cdot  \mathrm{{OUT}})}\right.$。类别2的结果元组$t$可能会被多次看到（例如，如果$t$中有多个矩形具有相同的上边缘，就会发生这种情况）。同样，去重操作在渐进意义上不会产生额外的成本。

We now complete the proof of Theorem 1.1.

我们现在完成定理1.1的证明。

## Appendix

## 附录

## A Building Brick Algorithms

## A 基础算法

Terminology. Each point(x,y)is said to define a y-coordinate $y$ ,a horizontal segment $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  y$ is said to define a y-coordinate $y$ ,and each rectangle $r = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  \left\lbrack  {{y}_{1},{y}_{2}}\right\rbrack$ is said to define two y-coordinates ${y}_{1}$ and ${y}_{2}$ . These definitions permit us to conveniently specify a set of y-coordinates using expressions like "the set of four y-coordinates defined by point $p$ ,horizontal segment $h$ ,and rectangle ${r}^{\prime \prime }$ .

术语。每个点(x, y)被认为定义了一个y坐标$y$，一条水平线段$\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  y$被认为定义了一个y坐标$y$，每个矩形$r = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  \left\lbrack  {{y}_{1},{y}_{2}}\right\rbrack$被认为定义了两个y坐标${y}_{1}$和${y}_{2}$。这些定义使我们能够方便地使用诸如“由点$p$、水平线段$h$和矩形${r}^{\prime \prime }$定义的四个y坐标的集合”这样的表达式来指定一组y坐标。

Fundamental Data Structures. The interval tree [5] stores a set $S$ of intervals in $\mathbb{R}$ using $O\left( \left| S\right| \right)$ space such that,given any real value $q$ ,the intervals of $S$ containing $q$ can be found in $O\left( {\log \left| S\right|  + K}\right)$ time,where $K$ is the number of intervals reported. It can also be used to detect whether $S$ has at least one interval containing $q$ in $O\left( {\log \left| S\right| }\right)$ time. The structure supports insertions and deletions on $S$ in $O\left( {\log \left| S\right| }\right)$ time.

基本数据结构。区间树[5]使用$O\left( \left| S\right| \right)$的空间存储$\mathbb{R}$中的一组区间$S$，使得给定任何实数值$q$，可以在$O\left( {\log \left| S\right|  + K}\right)$的时间内找到$S$中包含$q$的区间，其中$K$是所报告的区间数量。它还可以用于在$O\left( {\log \left| S\right| }\right)$的时间内检测$S$中是否至少有一个包含$q$的区间。该结构支持在$O\left( {\log \left| S\right| }\right)$的时间内对$S$进行插入和删除操作。

Now,let us assume that each interval of $S$ is associated with a real-valued weight. Given a real value $q$ ,a stabbing max query returns the maximum weight of all the intervals in $S$ covering $q$ (if no such intervals exist,the query returns $- \infty$ ). We can store $S$ in a structure of [1] using $O\left( \left| S\right| \right)$ space that can answer such a query in $O\left( {\log \left| S\right| }\right)$ time. The structure supports insertions and deletions on $S$ in $O\left( {\log \left| S\right| }\right)$ amortized time.

现在，让我们假设$S$中的每个区间都与一个实数值权重相关联。给定一个实数值$q$，刺探最大查询返回$S$中覆盖$q$的所有区间的最大权重（如果不存在这样的区间，查询返回$- \infty$）。我们可以使用$O\left( \left| S\right| \right)$的空间将$S$存储在文献[1]的一种结构中，该结构可以在$O\left( {\log \left| S\right| }\right)$的时间内回答这样的查询。该结构支持在$O\left( {\log \left| S\right| }\right)$的摊还时间内对$S$进行插入和删除操作。

The priority search tree (PST) [16] stores a set $P$ of points using $O\left( \left| P\right| \right)$ space such that,given a 3-sided rectangle $q = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  \lbrack y,\infty )$ ,the points of $S$ covered by $q$ can be found in $O\left( {\log \left| P\right|  + K}\right)$ time,where $K$ is the number of points reported. The structure supports insertions and deletions on $S$ in $O\left( {\log \left| S\right| }\right)$ time.

优先搜索树（PST）[16]使用$O\left( \left| P\right| \right)$的空间存储一组点$P$，使得给定一个三边矩形$q = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  \lbrack y,\infty )$，可以在$O\left( {\log \left| P\right|  + K}\right)$的时间内找到被$q$覆盖的$S$中的点，其中$K$是所报告的点的数量。该结构支持在$O\left( {\log \left| S\right| }\right)$的时间内对$S$进行插入和删除操作。

The PST can be deployed to answer queries on intervals. Let $S$ be a set of intervals in $\mathbb{R}$ . Given an interval $q = \left\lbrack  {{z}_{1},{z}_{2}}\right\rbrack$ ,a containment query reports all the intervals of $S$ that are contained by $q$ . We can store $S$ in a PST of $O\left( \left| S\right| \right)$ space that solves such a query in $O\left( {\log \left| S\right|  + K}\right)$ time,where $K$ is the number of intervals reported. To see why,observe that an interval $\left\lbrack  {x,y}\right\rbrack$ is contained by another $\left\lbrack  {{z}_{1},{z}_{2}}\right\rbrack$ if and only if the point(x,y)falls in the 3-sided rectangle $\left\lbrack  {{z}_{1},\infty }\right)  \times  \left( {-\infty ,{z}_{2}}\right\rbrack$ . Thus,we create from $S$ a point set $P = \{ \left( {x,y}\right)  \mid  \left\lbrack  {x,y}\right\rbrack   \in  S\}$ and store $P$ in a PST. Given an interval $q = \left\lbrack  {{z}_{1},{z}_{2}}\right\rbrack$ ,we can answer the containment query by using the PST to find all the points in $P$ covered by $\left\lbrack  {{z}_{1},\infty }\right)  \times  \left( {-\infty ,{z}_{2}}\right\rbrack$ and,for each such point(x,y),report $\left\lbrack  {x,y}\right\rbrack$ .

可以部署区间搜索树（PST）来回答关于区间的查询。设 $S$ 是 $\mathbb{R}$ 中的一组区间。给定一个区间 $q = \left\lbrack  {{z}_{1},{z}_{2}}\right\rbrack$，包含查询会报告 $S$ 中所有被 $q$ 包含的区间。我们可以将 $S$ 存储在一个空间复杂度为 $O\left( \left| S\right| \right)$ 的区间搜索树中，该树可以在 $O\left( {\log \left| S\right|  + K}\right)$ 时间内解决此类查询，其中 $K$ 是报告的区间数量。要理解原因，请注意，当且仅当点 (x, y) 落在三边矩形 $\left\lbrack  {{z}_{1},\infty }\right)  \times  \left( {-\infty ,{z}_{2}}\right\rbrack$ 内时，一个区间 $\left\lbrack  {x,y}\right\rbrack$ 才会被另一个区间 $\left\lbrack  {{z}_{1},{z}_{2}}\right\rbrack$ 包含。因此，我们从 $S$ 创建一个点集 $P = \{ \left( {x,y}\right)  \mid  \left\lbrack  {x,y}\right\rbrack   \in  S\}$，并将 $P$ 存储在区间搜索树中。给定一个区间 $q = \left\lbrack  {{z}_{1},{z}_{2}}\right\rbrack$，我们可以通过使用区间搜索树来找到 $P$ 中被 $\left\lbrack  {{z}_{1},\infty }\right)  \times  \left( {-\infty ,{z}_{2}}\right\rbrack$ 覆盖的所有点，并为每个这样的点 (x, y) 报告 $\left\lbrack  {x,y}\right\rbrack$。

Another closely related query is the reverse-containment query,which,given an interval $q =$ $\left\lbrack  {{z}_{1},{z}_{2}}\right\rbrack$ ,finds all the intervals of $S$ that contain $q$ (rather than "being contained by $q$ "). Again,we can store $S$ in a PST of $O\left( \left| S\right| \right)$ space that answers such a query in $O\left( {\log \left| S\right|  + K}\right)$ time,where $K$ is the number of intervals reported. In general,an interval $\left\lbrack  {x,y}\right\rbrack$ contains another $\left\lbrack  {{z}_{1},{z}_{2}}\right\rbrack$ if and only if the point(x,y)falls in the 3-sided rectangle $\left( {-\infty ,{z}_{1}\rbrack  \times  \left\lbrack  {{z}_{2},\infty }\right. }\right)$ . Thus,we create from $S$ a point set $P = \{ \left( {x,y}\right)  \mid  \left\lbrack  {x,y}\right\rbrack   \in  S\}$ and store $P$ in a PST. Given an interval $q = \left\lbrack  {{z}_{1},{z}_{2}}\right\rbrack$ ,we use the PST to find all the points in $P$ covered by $\left( {-\infty ,{z}_{1}\rbrack  \times  \left\lbrack  {{z}_{2},\infty }\right. }\right)$ and,for each such point(x,y),report $\left\lbrack  {x,y}\right\rbrack$ .

另一个密切相关的查询是反向包含查询，给定一个区间 $q =$ $\left\lbrack  {{z}_{1},{z}_{2}}\right\rbrack$，该查询会找到 $S$ 中所有包含 $q$ 的区间（而不是“被 $q$ 包含”）。同样，我们可以将 $S$ 存储在一个空间复杂度为 $O\left( \left| S\right| \right)$ 的区间搜索树中，该树可以在 $O\left( {\log \left| S\right|  + K}\right)$ 时间内回答此类查询，其中 $K$ 是报告的区间数量。一般来说，当且仅当点 (x, y) 落在三边矩形 $\left( {-\infty ,{z}_{1}\rbrack  \times  \left\lbrack  {{z}_{2},\infty }\right. }\right)$ 内时，一个区间 $\left\lbrack  {x,y}\right\rbrack$ 才会包含另一个区间 $\left\lbrack  {{z}_{1},{z}_{2}}\right\rbrack$。因此，我们从 $S$ 创建一个点集 $P = \{ \left( {x,y}\right)  \mid  \left\lbrack  {x,y}\right\rbrack   \in  S\}$，并将 $P$ 存储在区间搜索树中。给定一个区间 $q = \left\lbrack  {{z}_{1},{z}_{2}}\right\rbrack$，我们使用区间搜索树来找到 $P$ 中被 $\left( {-\infty ,{z}_{1}\rbrack  \times  \left\lbrack  {{z}_{2},\infty }\right. }\right)$ 覆盖的所有点，并为每个这样的点 (x, y) 报告 $\left\lbrack  {x,y}\right\rbrack$。

Algorithm for Problem $\mathcal{A}$ . Consider first the detection version. Sort the set of y-coordinates defined by the points of $P$ and the rectangles of $R$ . Next,sweep (conceptually) a horizontal line $\ell$ from $y =  - \infty$ to $y = \infty$ . At all times,maintain the set ${R}_{\ell }$ of rectangles in $R$ that intersect with $\ell$ . Let ${S}_{\ell }$ be the set of $\mathrm{x}$ -ranges of the rectangles in ${R}_{\ell }$ ; we store ${S}_{\ell }$ in an interval tree $\mathcal{T}$ . Specifically,when $\ell$ hits the bottom (resp.,top) edge of a rectangle $r = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  \left\lbrack  {{y}_{1},{y}_{2}}\right\rbrack$ of $R$ ,we insert (resp.,delete) $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$ into (resp.,from) $\mathcal{T}$ ,which can be done in $O\left( {\log n}\right)$ time. When $\ell$ hits a point $p = \left( {x,y}\right)$ of $P$ ,search $\mathcal{T}$ to determine if any interval in ${S}_{\ell }$ contains the value $x$ . If so,point $p$ is covered by at least one rectangle in $R$ ; otherwise,it is not. The overall running time is $O\left( {n\log n}\right)$ .

问题 $\mathcal{A}$ 的算法。首先考虑检测版本。对由 $P$ 中的点和 $R$ 中的矩形所定义的 y 坐标集合进行排序。接下来，（从概念上）将一条水平线 $\ell$ 从 $y =  - \infty$ 扫到 $y = \infty$。在任何时刻，维护 $R$ 中与 $\ell$ 相交的矩形集合 ${R}_{\ell }$。设 ${S}_{\ell }$ 为 ${R}_{\ell }$ 中矩形的 $\mathrm{x}$ 范围集合；我们将 ${S}_{\ell }$ 存储在一个区间树 $\mathcal{T}$ 中。具体来说，当 $\ell$ 碰到 $R$ 中矩形 $r = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  \left\lbrack  {{y}_{1},{y}_{2}}\right\rbrack$ 的底边（相应地，顶边）时，我们将 $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$ 插入（相应地，删除）到 $\mathcal{T}$ 中（相应地，从 $\mathcal{T}$ 中删除），这可以在 $O\left( {\log n}\right)$ 时间内完成。当 $\ell$ 碰到 $P$ 中的点 $p = \left( {x,y}\right)$ 时，搜索 $\mathcal{T}$ 以确定 ${S}_{\ell }$ 中是否有任何区间包含值 $x$。如果是，则点 $p$ 被 $R$ 中的至少一个矩形覆盖；否则，它没有被覆盖。总体运行时间为 $O\left( {n\log n}\right)$。

The algorithm for the reporting version of Problem $\mathcal{A}$ is similar. The only difference is that, when $\ell$ hits a point $p = \left( {x,y}\right)$ of $P$ ,we use $\mathcal{T}$ to report all the intervals in ${S}_{\ell }$ that contain $x$ ; the cost is $O\left( {n\log n + {K}_{p}}\right)$ ,where ${K}_{p}$ is the number of such intervals. Every interval corresponds to a rectangle in $R$ that contains $p$ . The total running time is $O\left( {n\log n + \mathop{\sum }\limits_{p}{K}_{p}}\right)  = O\left( {n\log n + \mathrm{{OUT}}}\right)$ .

问题 $\mathcal{A}$ 的报告版本的算法类似。唯一的区别是，当 $\ell$ 碰到 $P$ 中的点 $p = \left( {x,y}\right)$ 时，我们使用 $\mathcal{T}$ 来报告 ${S}_{\ell }$ 中包含 $x$ 的所有区间；成本为 $O\left( {n\log n + {K}_{p}}\right)$，其中 ${K}_{p}$ 是此类区间的数量。每个区间对应于 $R$ 中包含 $p$ 的一个矩形。总运行时间为 $O\left( {n\log n + \mathop{\sum }\limits_{p}{K}_{p}}\right)  = O\left( {n\log n + \mathrm{{OUT}}}\right)$。

Algorithm for Problem $\mathcal{B}$ . Sort the set of y-coordinates defined by all the segments of $H$ and $V$ . Next,sweep a horizontal line $\ell$ from $y =  - \infty$ to $y = \infty$ . At all times,maintain the set ${V}_{\ell }$ of segments in $V$ that intersect with $\ell$ . Let ${S}_{\ell }$ be the set of $\mathrm{x}$ -coordinates of the segments in ${V}_{\ell }$ ; we store ${S}_{\ell }$ in a binary search tree (BST) $\mathcal{T}$ . Specifically,when $\ell$ hits the lower (resp.,upper) endpoint of a vertical segment $v = x \times  \left\lbrack  {{y}_{1},{y}_{2}}\right\rbrack$ of $V$ ,we insert (resp.,delete) the value $x$ into (resp.,from) $\mathcal{T}$ ,which can be done in $O\left( {\log n}\right)$ amortized time. When $\ell$ hits a horizontal segment $h = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  y$ of $H$ ,search $\mathcal{T}$ to determine the successor ${x}^{\prime }$ of ${x}_{1}$ in ${S}_{\ell }$ . If ${x}^{\prime } \leq  {x}_{2}$ ,then we output a pair(h,p),where $p$ is the point $\left( {{x}^{\prime },y}\right)$ ; otherwise,output nothing for $h$ . The overall running time is $O\left( {n\log n}\right)$ .

问题 $\mathcal{B}$ 的算法。对由 $H$ 和 $V$ 的所有线段所定义的 y 坐标集合进行排序。接下来，从 $y =  - \infty$ 到 $y = \infty$ 扫描一条水平线 $\ell$。在任何时候，维护 $V$ 中与 $\ell$ 相交的线段集合 ${V}_{\ell }$。设 ${S}_{\ell }$ 为 ${V}_{\ell }$ 中线段的 $\mathrm{x}$ 坐标集合；我们将 ${S}_{\ell }$ 存储在一棵二叉搜索树（BST）$\mathcal{T}$ 中。具体来说，当 $\ell$ 碰到 $V$ 的一条垂直线段 $v = x \times  \left\lbrack  {{y}_{1},{y}_{2}}\right\rbrack$ 的下端点（相应地，上端点）时，我们将值 $x$ 插入（相应地，从……中删除）$\mathcal{T}$，这可以在 $O\left( {\log n}\right)$ 均摊时间内完成。当 $\ell$ 碰到 $H$ 的一条水平线段 $h = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  y$ 时，搜索 $\mathcal{T}$ 以确定 ${S}_{\ell }$ 中 ${x}_{1}$ 的后继 ${x}^{\prime }$。如果 ${x}^{\prime } \leq  {x}_{2}$，那么我们输出一个对 (h, p)，其中 $p$ 是点 $\left( {{x}^{\prime },y}\right)$；否则，对于 $h$ 不输出任何内容。总体运行时间为 $O\left( {n\log n}\right)$。

Algorithm for Problem %. Sort the set of y-coordinates defined by all the segments in $H$ and all the rectangles in $R$ . Next,sweep a horizontal line $\ell$ from $y =  - \infty$ to $y = \infty$ . At all times,maintain the set ${R}_{\ell }$ of rectangles in $R$ that intersect with $\ell$ . Let ${S}_{\ell }$ be the set of $\mathrm{x}$ -ranges of the rectangles in ${R}_{\ell }$ ; we store ${S}_{\ell }$ in a stabbing-max structure $\mathcal{T}$ of [1]. Specifically,when $\ell$ hits the bottom (resp.,top) edge of a rectangle $r = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  \left\lbrack  {{y}_{1},{y}_{2}}\right\rbrack$ of $R$ ,we insert (resp.,delete) $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$ with weight ${x}_{2}$ into (resp., from) $\mathcal{T}$ ,which can be done in $O\left( {\log n}\right)$ time. When $\ell$ hits a horizontal segment $h = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  y$ of $P$ ,search $\mathcal{T}$ to determine the maximum weight $w$ of all the intervals in ${S}_{\ell }$ containing the value ${x}_{1}$ . If $w \neq   - \infty$ ,we output(h,p)where the point $p$ is defined in a way depending on $w$ : if $w \leq  {x}_{2}$ ,then $p = \left( {w,y}\right)$ ; otherwise $p = \left( {{x}_{2},y}\right)$ . The overall running time is $O\left( {n\log n}\right)$ .

问题 % 的算法。对由 $H$ 中的所有线段和 $R$ 中的所有矩形所定义的 y 坐标集合进行排序。接下来，从 $y =  - \infty$ 到 $y = \infty$ 扫描一条水平线 $\ell$。在任何时候，维护 $R$ 中与 $\ell$ 相交的矩形集合 ${R}_{\ell }$。设 ${S}_{\ell }$ 为 ${R}_{\ell }$ 中矩形的 $\mathrm{x}$ 范围集合；我们将 ${S}_{\ell }$ 存储在文献 [1] 中的一个刺探 - 最大值结构 $\mathcal{T}$ 中。具体来说，当 $\ell$ 碰到 $R$ 中一个矩形 $r = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  \left\lbrack  {{y}_{1},{y}_{2}}\right\rbrack$ 的底边（相应地，顶边）时，我们将权重为 ${x}_{2}$ 的 $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$ 插入（相应地，从……中删除）$\mathcal{T}$，这可以在 $O\left( {\log n}\right)$ 时间内完成。当 $\ell$ 碰到 $P$ 的一条水平线段 $h = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  y$ 时，搜索 $\mathcal{T}$ 以确定 ${S}_{\ell }$ 中包含值 ${x}_{1}$ 的所有区间的最大权重 $w$。如果 $w \neq   - \infty$，我们输出 (h, p)，其中点 $p$ 的定义方式取决于 $w$：如果 $w \leq  {x}_{2}$，那么 $p = \left( {w,y}\right)$；否则 $p = \left( {{x}_{2},y}\right)$。总体运行时间为 $O\left( {n\log n}\right)$。

Algorithm for Problem 2. Consider first the find-lowest version. Sort the set of y-coordinates defined by all the horizontal segments and rectangles. In the outset,all rectangles of $R$ are marked as inactive. During our algorithm, the status of each rectangle will turn from inactive to active at some point, turn from active back to inactive at a later point, and then stay that way forever.

问题2的算法。首先考虑查找最低版本。对由所有水平线段和矩形定义的y坐标集合进行排序。一开始，$R$中的所有矩形都被标记为非活动状态。在我们的算法执行过程中，每个矩形的状态会在某个时刻从非活动变为活动，在稍后的某个时刻从活动变回非活动，然后永远保持该状态。

Sweep a horizontal line $\ell$ from $y =  - \infty$ to $y = \infty$ . At all times,we maintain the set ${R}_{\ell }$ of active rectangles in $R$ that intersect with $\ell$ . Let ${S}_{\ell }$ be the set of $\mathrm{x}$ -ranges of the rectangles in ${R}_{\ell }$ ; we store ${S}_{\ell }$ in a PST $\mathcal{T}$ . Specifically,when $\ell$ hits the top edge of a rectangle $r = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  \left\lbrack  {{y}_{1},{y}_{2}}\right\rbrack$ of $R$ ,we insert $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$ into $\mathcal{T}$ and mark $r$ as active,which can be done in $O\left( {\log n}\right)$ time. The rectangle $r$ will be referred to as the host of $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$ and is stored together with $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$ in $\mathcal{T}$ .

从$y =  - \infty$到$y = \infty$扫描一条水平线$\ell$。在任何时候，我们都维护$R$中与$\ell$相交的活动矩形集合${R}_{\ell }$。设${S}_{\ell }$为${R}_{\ell }$中矩形的$\mathrm{x}$范围集合；我们将${S}_{\ell }$存储在一个优先搜索树（Priority Search Tree，PST）$\mathcal{T}$中。具体来说，当$\ell$碰到$R$中一个矩形$r = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  \left\lbrack  {{y}_{1},{y}_{2}}\right\rbrack$的上边缘时，我们将$\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$插入到$\mathcal{T}$中，并将$r$标记为活动状态，这可以在$O\left( {\log n}\right)$时间内完成。矩形$r$将被称为$\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$的宿主矩形，并与$\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$一起存储在$\mathcal{T}$中。

When $\ell$ hits a horizontal segment $h = \left\lbrack  {{z}_{1},{z}_{2}}\right\rbrack   \times  y$ of $P$ ,perform a containment query on $\mathcal{T}$ to find all the intervals in ${S}_{\ell }$ that are contained by $\left\lbrack  {{z}_{1},{z}_{2}}\right\rbrack$ ; if ${K}_{h}$ is the number of such intervals, this retrieval takes $O\left( {\log n + {K}_{h}}\right)$ time. For each retrieved interval $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$ ,we also obtain its host rectangle $r$ (stored along with $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$ in $\mathcal{T}$ ). As can be verified shortly, $h$ is the lowest segment in $H$ that crosses $r$ ; we therefore output(r,h). After that, $r$ is marked as inactive,and accordingly, its $\mathrm{x}$ -range $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$ is deleted from $\mathcal{T}$ in $O\left( {\log n}\right)$ time. As $r$ will remain inactive in the rest of the execution, its x-range will not be retrieved again by another containment query in the future. This implies that $h$ is indeed the lowest segment in $H$ crossing $r$ .

当$\ell$碰到$P$的一条水平线段$h = \left\lbrack  {{z}_{1},{z}_{2}}\right\rbrack   \times  y$时，对$\mathcal{T}$执行包含查询，以找出${S}_{\ell }$中所有被$\left\lbrack  {{z}_{1},{z}_{2}}\right\rbrack$包含的区间；如果${K}_{h}$是此类区间的数量，那么这次检索需要$O\left( {\log n + {K}_{h}}\right)$时间。对于每个检索到的区间$\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$，我们还会获取其宿主矩形$r$（与$\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$一起存储在$\mathcal{T}$中）。很快可以验证，$h$是$H$中穿过$r$的最低线段；因此我们输出(r,h)。之后，$r$被标记为非活动状态，相应地，其$\mathrm{x}$范围$\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$会在$O\left( {\log n}\right)$时间内从$\mathcal{T}$中删除。由于$r$在后续执行过程中将保持非活动状态，其x范围在未来不会再被另一次包含查询检索到。这意味着$h$确实是$H$中穿过$r$的最低线段。

When $\ell$ hits the bottom edge of a rectangle $r = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  \left\lbrack  {{y}_{1},{y}_{2}}\right\rbrack$ of $R$ ,we check whether $r$ is active. If so,delete $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$ from $\mathcal{T}$ and mark $r$ as inactive; otherwise,do nothing.

当$\ell$碰到$R$中一个矩形$r = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  \left\lbrack  {{y}_{1},{y}_{2}}\right\rbrack$的下边缘时，我们检查$r$是否处于活动状态。如果是，则从$\mathcal{T}$中删除$\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$并将$r$标记为非活动状态；否则，不做任何操作。

Overall,each rectangle of $R$ necessitates one insertion and one deletion in $\mathcal{T}$ . All these insertions and deletions take $O\left( {n\log n}\right)$ time in total. Each segment $h$ of $H$ performs a containment query on $\mathcal{T}$ ,which has a cost of $O\left( {\log n + {K}_{h}}\right)$ . All these queries demand a total cost of $O\left( {n\log n + \mathop{\sum }\limits_{h}{K}_{h}}\right)$ . Recall that the x-range of a rectangle in $R$ can be retrieved by at most one containment query. Hence, $\mathop{\sum }\limits_{h}{K}_{h} \leq  \left| R\right|  \leq  n$ and the runtime of our algorithm is $O\left( {n\log n}\right)$ .

总体而言，$R$中的每个矩形都需要在$\mathcal{T}$中进行一次插入和一次删除操作。所有这些插入和删除操作总共需要$O\left( {n\log n}\right)$的时间。$H$中的每个线段$h$都会对$\mathcal{T}$执行一次包含查询，该查询的成本为$O\left( {\log n + {K}_{h}}\right)$。所有这些查询的总成本为$O\left( {n\log n + \mathop{\sum }\limits_{h}{K}_{h}}\right)$。请记住，$R$中矩形的x范围最多可以通过一次包含查询来获取。因此，$\mathop{\sum }\limits_{h}{K}_{h} \leq  \left| R\right|  \leq  n$，并且我们算法的运行时间为$O\left( {n\log n}\right)$。

Next,we consider the find-all-sorted version of Problem $\mathcal{D}$ . For each $r \in  R$ ,we keep a linked list,which at the end of our algorithm will store the horizontal segments of ${\operatorname{cross}}_{H}\left( r\right)$ in ascending order of their y-coordinates. In the outset, all linked lists are empty. Unlike the detection version, we will not need to keep the active status for the rectangles.

接下来，我们考虑问题$\mathcal{D}$的全排序查找版本。对于每个$r \in  R$，我们维护一个链表，在算法结束时，该链表将按y坐标升序存储${\operatorname{cross}}_{H}\left( r\right)$的水平线段。一开始，所有链表都是空的。与检测版本不同，我们不需要为矩形维护活动状态。

Again,sweep a horizontal line $\ell$ from $y =  - \infty$ to $y = \infty$ . At all times,we maintain the set ${R}_{\ell }$ of rectangles in $R$ that intersect with $\ell$ . Let ${S}_{\ell }$ be the set of $\mathrm{x}$ -ranges of the rectangles in ${R}_{\ell }$ ; we store ${S}_{\ell }$ in a PST $\mathcal{T}$ . Specifically,when $\ell$ hits the bottom (resp.,top) edge of a rectangle $r = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  \left\lbrack  {{y}_{1},{y}_{2}}\right\rbrack$ of $R$ ,we insert (resp.,delete) $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$ into (resp.,from) $\mathcal{T}$ ,which can be done in $O\left( {\log n}\right)$ time. Again, the rectangle $r$ - the host of $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   -$ is stored together with $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$ in $\mathcal{T}$ .

再次，从$y =  - \infty$到$y = \infty$扫描一条水平线$\ell$。在任何时候，我们都维护$R$中与$\ell$相交的矩形集合${R}_{\ell }$。设${S}_{\ell }$为${R}_{\ell }$中矩形的$\mathrm{x}$范围集合；我们将${S}_{\ell }$存储在一个持久线段树（Persistent Segment Tree，PST）$\mathcal{T}$中。具体来说，当$\ell$碰到$R$中矩形$r = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  \left\lbrack  {{y}_{1},{y}_{2}}\right\rbrack$的底边（分别地，顶边）时，我们将$\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$插入（分别地，删除）到$\mathcal{T}$中（分别地，从$\mathcal{T}$中删除），这可以在$O\left( {\log n}\right)$的时间内完成。同样，矩形$r$（即$\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   -$的宿主矩形）与$\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$一起存储在$\mathcal{T}$中。

When $\ell$ hits a horizontal segment $h = \left\lbrack  {{z}_{1},{z}_{2}}\right\rbrack   \times  y$ of $P$ ,perform a containment query on $\mathcal{T}$ to find all the intervals in ${S}_{\ell }$ that are contained by $\left\lbrack  {{z}_{1},{z}_{2}}\right\rbrack$ ; the query cost is $O\left( {\log n + {K}_{h}}\right)$ ,where ${K}_{h}$ is the number of intervals reported. For each retrieved interval $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$ ,we also obtain its host rectangle $r$ . It is clear that $h$ is a segment crossing $r$ and is thus appended to the linked list of $r$ . Note that $h$ is higher than all the segments already in that linked list.

当$\ell$碰到$P$的一条水平线段$h = \left\lbrack  {{z}_{1},{z}_{2}}\right\rbrack   \times  y$时，对$\mathcal{T}$执行一次包含查询，以找出${S}_{\ell }$中所有被$\left\lbrack  {{z}_{1},{z}_{2}}\right\rbrack$包含的区间；查询成本为$O\left( {\log n + {K}_{h}}\right)$，其中${K}_{h}$是报告的区间数量。对于每个检索到的区间$\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$，我们还会获取其宿主矩形$r$。显然，$h$是一条穿过$r$的线段，因此会被追加到$r$的链表中。请注意，$h$比该链表中已有的所有线段都高。

Overall,each rectangle of $R$ necessitates one insertion and one deletion in $\mathcal{T}$ . All these insertions and deletions take $O\left( {n\log n}\right)$ time in total. Each segment $h$ of $H$ performs a containment query on $\mathcal{T}$ ,which has a cost of $O\left( {\log n + {K}_{h}}\right)$ . All these queries demand a total cost of $O\left( {n\log n + \mathop{\sum }\limits_{h}{K}_{h}}\right)$ . However,unlike the detection version,the sum $\mathop{\sum }\limits_{h}{K}_{h}$ here is equal to the total size of ${\operatorname{cross}}_{H}\left( r\right)$ for all the $r \in  R$ . The total size is equivalent to OUT.

总体而言，$R$中的每个矩形都需要在$\mathcal{T}$中进行一次插入和一次删除操作。所有这些插入和删除操作总共需要$O\left( {n\log n}\right)$的时间。$H$中的每个线段$h$都会对$\mathcal{T}$执行一次包含查询，该查询的成本为$O\left( {\log n + {K}_{h}}\right)$。所有这些查询的总成本为$O\left( {n\log n + \mathop{\sum }\limits_{h}{K}_{h}}\right)$。然而，与检测版本不同的是，这里的总和$\mathop{\sum }\limits_{h}{K}_{h}$等于所有$r \in  R$对应的${\operatorname{cross}}_{H}\left( r\right)$的总大小。该总大小等同于输出（OUT）。

Algorithm for Problem $\mathcal{E}$ . Sort all the rectangles $r \in  R$ in ascending order of right(r)(namely, the x-coordinate of the right edge of $r$ ). To each $r \in  R$ ,we assign an ID $i \in  \left\lbrack  \left| R\right| \right\rbrack$ if $r$ is at the $i$ -th position of the sorted list.

问题$\mathcal{E}$的算法。将所有矩形$r \in  R$按照右边界（r）（即$r$右边缘的x坐标）的升序进行排序。如果$r$在排序后的列表中位于第$i$个位置，我们为每个$r \in  R$分配一个ID $i \in  \left\lbrack  \left| R\right| \right\rbrack$。

Next,we aim to produce,for each pair $\left( {h,r}\right)  \in  H \times  r$ such that $h$ crosses $r$ ,a pair $\left( {h,\lambda }\right)$ where $\lambda$ is the ID of $r$ . These pairs may be output in an arbitrary order. Sort the set of y-coordinates defined by all the horizontal segments and rectangles. Sweep a horizontal line $\ell$ from $y =  - \infty$ to $y = \infty$ . At all times,we maintain the set ${R}_{\ell }$ of rectangles in $R$ that intersect with $\ell$ . Let ${S}_{\ell }$ be the set of $\mathrm{x}$ -ranges of the rectangles in ${R}_{\ell }$ ; we store ${S}_{\ell }$ in a PST $\mathcal{T}$ . Specifically,when $\ell$ hits the bottom (resp., top) edge of a rectangle $r = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  \left\lbrack  {{y}_{1},{y}_{2}}\right\rbrack$ of $R$ ,we insert (resp.,delete) $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$ into (resp.,from) $\mathcal{T}$ ,which can be done in $O\left( {\log n}\right)$ time. We call the rectangle $r$ the host of $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$ and store its ID together with $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$ in $\mathcal{T}$ . When $\ell$ hits a horizontal segment $h = \left\lbrack  {{z}_{1},{z}_{2}}\right\rbrack   \times  y$ of $H$ ,perform a reverse-containment query on $\mathcal{T}$ to find all the intervals in ${S}_{\ell }$ that contain $\left\lbrack  {{z}_{1},{z}_{2}}\right\rbrack$ ; the query cost is $O\left( {\log n + {K}_{h}}\right)$ ,where ${K}_{h}$ is the number of intervals reported. For each retrieved interval $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$ , we also obtain the ID $\lambda$ of its host rectangle $r$ ,and output the pair $\left( {h,\lambda }\right)$ .

接下来，我们的目标是为每一对满足$h$与$r$相交的$\left( {h,r}\right)  \in  H \times  r$生成一对$\left( {h,\lambda }\right)$，其中$\lambda$是$r$的ID。这些对可以以任意顺序输出。对所有水平线段和矩形所定义的y坐标集合进行排序。从$y =  - \infty$到$y = \infty$扫描一条水平线$\ell$。在任何时候，我们都维护$R$中与$\ell$相交的矩形集合${R}_{\ell }$。设${S}_{\ell }$为${R}_{\ell }$中矩形的$\mathrm{x}$范围集合；我们将${S}_{\ell }$存储在一个持久化线段树（PST）$\mathcal{T}$中。具体来说，当$\ell$碰到$R$中某个矩形$r = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  \left\lbrack  {{y}_{1},{y}_{2}}\right\rbrack$的下（或上）边缘时，我们将$\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$插入（或删除）到$\mathcal{T}$中，这可以在$O\left( {\log n}\right)$的时间内完成。我们称矩形$r$为$\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$的宿主矩形，并将其ID与$\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$一起存储在$\mathcal{T}$中。当$\ell$碰到$H$的一条水平线段$h = \left\lbrack  {{z}_{1},{z}_{2}}\right\rbrack   \times  y$时，对$\mathcal{T}$执行一次反向包含查询，以找出${S}_{\ell }$中包含$\left\lbrack  {{z}_{1},{z}_{2}}\right\rbrack$的所有区间；查询成本为$O\left( {\log n + {K}_{h}}\right)$，其中${K}_{h}$是报告的区间数量。对于每个检索到的区间$\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$，我们还会获取其宿主矩形$r$的ID $\lambda$，并输出这一对$\left( {h,\lambda }\right)$。

Each rectangle of $R$ necessitates one insertion and one deletion in $\mathcal{T}$ . All these insertions and deletions take $O\left( {n\log n}\right)$ time in total. Each segment $h$ of $H$ performs a reverse-containment query on $\mathcal{T}$ ,which has a cost of $O\left( {\log n + {K}_{h}}\right)$ . All these queries demand a total cost of $O\left( {n\log n + \mathop{\sum }\limits_{h}{K}_{h}}\right)$ . The sum $\mathop{\sum }\limits_{h}{K}_{h}$ here is equal to the total size of ${\operatorname{contain}}_{R}\left( h\right)$ for all the $h \in  H$ . The total size is equivalent to OUT. The cost so far is therefore $O\left( {n\log n + \mathrm{{OUT}}}\right)$ .

$R$中的每个矩形都需要在$\mathcal{T}$中进行一次插入和一次删除操作。所有这些插入和删除操作总共需要$O\left( {n\log n}\right)$的时间。$H$中的每个线段$h$都会在$\mathcal{T}$上执行一次反向包含查询，其代价为$O\left( {\log n + {K}_{h}}\right)$。所有这些查询的总代价为$O\left( {n\log n + \mathop{\sum }\limits_{h}{K}_{h}}\right)$。这里的和$\mathop{\sum }\limits_{h}{K}_{h}$等于所有$h \in  H$对应的${\operatorname{contain}}_{R}\left( h\right)$的总大小。该总大小等同于输出规模（OUT）。因此，到目前为止的代价为$O\left( {n\log n + \mathrm{{OUT}}}\right)$。

Let $L$ be the list of $\left( {h,\lambda }\right)$ pairs produced (the size of $L$ is OUT). We now proceed to sort $L$ in ascending order of the $\lambda$ -field (which is a rectangle ID),breaking ties arbitrarily. Because the IDs are integers created by the algorithm,we are permitted to sort $L$ using counting sort without violating the comparison-based requirements. The counting sort finishes in $O\left( {\left| R\right|  + \left| L\right| }\right)  = O\left( {n + \mathrm{{OUT}}}\right)$ time, recalling that all the IDs are in $\left\lbrack  \left| R\right| \right\rbrack$ .

设$L$为生成的$\left( {h,\lambda }\right)$对的列表（$L$的大小为输出规模（OUT））。现在我们开始按照$\lambda$字段（即矩形ID）的升序对$L$进行排序，若有平局则任意打破。由于这些ID是由算法生成的整数，我们可以使用计数排序对$L$进行排序，而不会违反基于比较的要求。回想一下，所有的ID都在$\left\lbrack  \left| R\right| \right\rbrack$范围内，计数排序在$O\left( {\left| R\right|  + \left| L\right| }\right)  = O\left( {n + \mathrm{{OUT}}}\right)$时间内完成。

Finally,we generate,for each $h$ ,its set ${\operatorname{contain}}_{R}\left( h\right)$ (i.e.,the set of rectangles covering $h$ ),where the rectangles $r$ are sorted by right(r). To start with,initialize an empty linked list for every $h \in  H$ . Inspect the pairs $\left( {h,\lambda }\right)  \in  L$ in ascending order of the ID-field $\lambda$ . For each pair $\left( {h,\lambda }\right)$ examined, identify the rectangle $r$ whose ID is $\lambda$ and add $r$ to the linked list of $h$ . By the way the rectangle IDs were generated,it is clear that right(r)is larger than or equal to the x-coordinates of the right edges of the rectangles already in the linked list. The whole scan over $L$ finishes in $O\left( {1 + \mathrm{{OUT}}}\right)$ time,and produces the correct output for Problem $\mathcal{E}$ .

最后，对于每个$h$，我们生成其集合${\operatorname{contain}}_{R}\left( h\right)$（即覆盖$h$的矩形集合），其中矩形$r$按右边界（right(r)）排序。首先，为每个$h \in  H$初始化一个空的链表。按ID字段$\lambda$的升序检查这些对$\left( {h,\lambda }\right)  \in  L$。对于检查的每一对$\left( {h,\lambda }\right)$，找出ID为$\lambda$的矩形$r$，并将$r$添加到$h$的链表中。根据矩形ID的生成方式，显然右边界（right(r)）大于或等于链表中已有矩形右边缘的x坐标。对$L$的整个扫描在$O\left( {1 + \mathrm{{OUT}}}\right)$时间内完成，并为问题$\mathcal{E}$生成正确的输出。

## B Supplementary Proofs for Section 4

## B 第4节的补充证明

We start by presenting two properties underneath the procedures designed in Section 4. These properties will enable us to construct simpler proofs later.

我们首先介绍第4节所设计过程背后的两个性质。这些性质将使我们后续能够构建更简单的证明。

Proposition B.1. Consider any rectangle $r$ taken from ${R}_{1},{R}_{2},\ldots$ ,or ${R}_{k - 2}$ . Let ${r}^{\prime }$ be the trimmed rectangle of $r$ defined in Section 4.

命题B.1。考虑从${R}_{1},{R}_{2},\ldots$或${R}_{k - 2}$中选取的任意矩形$r$。设${r}^{\prime }$为第4节中定义的$r$的修剪矩形。

- If a horizontal segment $h \in  H$ crosses $r$ ,then $h$ must also cross ${r}^{\prime }$ .

- 如果一条水平线段$h \in  H$穿过$r$，那么$h$也必定穿过${r}^{\prime }$。

- If a vertical segment $v \in  V$ crosses $r$ ,then $v$ must also cross ${r}^{\prime }$ .

- 如果一条垂直线段$v \in  V$穿过$r$，那么$v$也必定穿过${r}^{\prime }$。

Proof. We will prove only the first bullet due to symmetry. Let us represent $h$ as $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  y$ . The fact of $h$ crossing $r$ indicates that $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$ contains the x-range of $r$ . Since the x-range of $r$ contains that of ${r}^{\prime }$ ,we know that $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$ must also contain the x-range of ${r}^{\prime }$ . To prove that $h$ crosses ${r}^{\prime }$ , we still need to show $y \in  \left\lbrack  {\operatorname{bot}\left( {r}^{\prime }\right) ,\operatorname{top}\left( {r}^{\prime }\right) }\right\rbrack$ . Recall that $\operatorname{bot}\left( {r}^{\prime }\right)$ is the y-coordinate of the lowest segment in $H$ crossing $r$ . This implies $y \geq  \operatorname{bot}\left( {r}^{\prime }\right)$ because $h$ itself is a segment in $H$ crossing $r$ . Analogously,it also holds that $y \leq  \operatorname{top}\left( {r}^{\prime }\right)$ . We can now conclude that $h$ crosses ${r}^{\prime }$ .

证明。由于对称性，我们仅证明第一个要点。我们将$h$表示为$\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  y$。$h$与$r$相交这一事实表明，$\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$包含$r$的x范围。由于$r$的x范围包含${r}^{\prime }$的x范围，我们知道$\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$也必定包含${r}^{\prime }$的x范围。为了证明$h$与${r}^{\prime }$相交，我们仍需证明$y \in  \left\lbrack  {\operatorname{bot}\left( {r}^{\prime }\right) ,\operatorname{top}\left( {r}^{\prime }\right) }\right\rbrack$。回想一下，$\operatorname{bot}\left( {r}^{\prime }\right)$是$H$中与$r$相交的最低线段的y坐标。这意味着$y \geq  \operatorname{bot}\left( {r}^{\prime }\right)$，因为$h$本身就是$H$中与$r$相交的一条线段。类似地，$y \leq  \operatorname{top}\left( {r}^{\prime }\right)$也成立。我们现在可以得出结论，$h$与${r}^{\prime }$相交。

Proposition B.2. Consider the sets ${R}_{1}^{\prime },{R}_{2}^{\prime },\ldots ,{R}_{k - 2}^{\prime }$ defined in (6). Let $t$ be any(k - 2)-tuple in $\mathcal{J}\left( {{R}_{1}^{\prime },{R}_{2}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$ . Then,neither $\mathrm{d} - {\operatorname{cross}}_{H}\left( \mathbf{t}\right)$ nor $\mathrm{d} - {\operatorname{cross}}_{V}\left( \mathbf{t}\right)$ can be empty.

命题B.2。考虑在(6)中定义的集合${R}_{1}^{\prime },{R}_{2}^{\prime },\ldots ,{R}_{k - 2}^{\prime }$。设$t$是$\mathcal{J}\left( {{R}_{1}^{\prime },{R}_{2}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$中的任意一个(k - 2)元组。那么，$\mathrm{d} - {\operatorname{cross}}_{H}\left( \mathbf{t}\right)$和$\mathrm{d} - {\operatorname{cross}}_{V}\left( \mathbf{t}\right)$都不可能为空。

Proof. Due to symmetry,we will give the proof only for $\mathrm{d} - {\operatorname{cross}}_{H}\left( \mathbf{t}\right)  \neq  \varnothing$ . Set ${r}^{\prime } = \operatorname{bot} - \operatorname{guard}\left( \mathbf{t}\right)$ , and let $r$ be the full rectangle of ${r}^{\prime }$ . Define $h$ as the lowest segment crossing $r$ (note that $h$ definitely exists because otherwise $r$ has no trimmed rectangle,contradicting the definition of ${r}^{\prime }$ ). We will show that $h \in  \mathrm{d} - {\operatorname{cross}}_{H}\left( \mathbf{t}\right)$ ,which indicates $\mathrm{d} - {\operatorname{cross}}_{H}\left( \mathbf{t}\right)  \neq  \varnothing$ . For this purpose,we should explain why $h$ crosses both ${B}_{t}$ and ${r}^{\prime }$ . However,by Proposition B.1, $h$ crossing $r$ directly implies $h$ crossing ${r}^{\prime }$ . It remains to prove that $h$ crosses ${B}_{t}$ .

证明。由于对称性，我们仅对$\mathrm{d} - {\operatorname{cross}}_{H}\left( \mathbf{t}\right)  \neq  \varnothing$给出证明。设${r}^{\prime } = \operatorname{bot} - \operatorname{guard}\left( \mathbf{t}\right)$，并设$r$是${r}^{\prime }$的完整矩形。将$h$定义为与$r$相交的最低线段（注意，$h$肯定存在，因为否则$r$就没有修剪后的矩形，这与${r}^{\prime }$的定义相矛盾）。我们将证明$h \in  \mathrm{d} - {\operatorname{cross}}_{H}\left( \mathbf{t}\right)$，这表明$\mathrm{d} - {\operatorname{cross}}_{H}\left( \mathbf{t}\right)  \neq  \varnothing$。为此，我们应该解释为什么$h$与${B}_{t}$和${r}^{\prime }$都相交。然而，根据命题B.1，$h$与$r$相交直接意味着$h$与${r}^{\prime }$相交。还需要证明$h$与${B}_{t}$相交。

By the definitions of ${r}^{\prime }$ and $h$ ,the bottom edge of ${r}^{\prime }$ must be contained in $h$ (note that $h$ is one of the segments used to trim $r$ into ${r}^{\prime }$ ). Because ${r}^{\prime } =$ bot-guard(t),the bottom edge of ${B}_{t}$ is contained in the bottom edge of ${r}^{\prime }$ and thus also contained in $h$ . This means that $h \cap  {B}_{t} \neq  \varnothing$ . On the other hand,the x-range of $h$ must cover that of ${r}^{\prime }$ (because $h$ crosses ${r}^{\prime }$ ),which in turn must cover that of ${B}_{t}$ (because ${r}^{\prime }$ covers ${B}_{t}$ ). Thus,the x-range of $h$ covers that of ${B}_{t}$ . This together with $h \cap  {B}_{t} \neq  \varnothing$ tells us that $h$ must cross ${B}_{t}$ .

根据${r}^{\prime }$和$h$的定义，${r}^{\prime }$的底边必定包含在$h$中（注意，$h$是用于将$r$修剪成${r}^{\prime }$的线段之一）。因为${r}^{\prime } =$ bot - guard(t)，${B}_{t}$的底边包含在${r}^{\prime }$的底边中，因此也包含在$h$中。这意味着$h \cap  {B}_{t} \neq  \varnothing$。另一方面，$h$的x范围必须覆盖${r}^{\prime }$的x范围（因为$h$与${r}^{\prime }$相交），而${r}^{\prime }$的x范围又必须覆盖${B}_{t}$的x范围（因为${r}^{\prime }$覆盖${B}_{t}$）。因此，$h$的x范围覆盖${B}_{t}$的x范围。结合$h \cap  {B}_{t} \neq  \varnothing$，我们可知$h$必定与${B}_{t}$相交。

We now proceed to elaborate the proofs postponed from Section 4. The order of the subsequent proofs will not strictly follow the sequence in which they are referenced in Section 4. In particular, we will prove Lemma 4.2 before Lemma 4.1, because the claims of the former lemma can be used to produce a succinct argument for the latter.

现在我们来详细阐述第4节中推迟的证明。后续证明的顺序不会严格遵循它们在第4节中被引用的顺序。特别地，我们将在引理4.1之前证明引理4.2，因为前者引理的命题可用于为后者提供简洁的论证。

## Proof of Lemma 4.2. We will prove each statement in turn.

## 引理4.2的证明。我们将依次证明每个命题。

Proof of Statement (1). Take any(k - 2)-tuple $t \in  \mathcal{J}\left( {{R}_{1}^{\prime },{R}_{2}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$ . Let ${r}_{i}\left( {i \in  \left\lbrack  {k - 2}\right\rbrack  }\right)$ be the full rectangle of $t\left\lbrack  i\right\rbrack$ . Fix any $h \in  \mathrm{d} - {\operatorname{cross}}_{H}\left( t\right)$ and any $v \in  \mathrm{d} - {\operatorname{cross}}_{V}\left( t\right)$ . The segments $h$ and $v$ both cross ${B}_{t}$ ,as can be seen directly from the definitions of $\mathrm{d} - {\operatorname{cross}}_{H}\left( t\right)$ and $\mathrm{d} - {\operatorname{cross}}_{V}\left( t\right)$ . Thus, $h \cap  v$ is a point in ${B}_{t} = \mathop{\bigcap }\limits_{{i = 1}}^{{k - 2}}t\left\lbrack  i\right\rbrack$ . As ${r}_{i}\left( {i \in  \left\lbrack  {k - 2}\right\rbrack  }\right)$ is the full rectangle of $t\left\lbrack  i\right\rbrack$ ,we know that $\mathop{\bigcap }\limits_{{i = 1}}^{{k - 2}}{r}_{i}$ covers ${B}_{t} = \mathop{\bigcap }\limits_{{i = 1}}^{{k - 2}}t\left\lbrack  i\right\rbrack$ . Hence,point $h \cap  v$ falls in $\mathop{\bigcap }\limits_{{i = 1}}^{{k - 2}}{r}_{i}$ ,indicating that $h \cap  v \cap  \mathop{\bigcap }\limits_{{i = 1}}^{{k - 2}}{r}_{i} \neq  \varnothing$ . It follows that $\left( {{r}_{1},\ldots ,{r}_{k - 2},h,v}\right)$ is a result tuple in $\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 2},H,V}\right)$ .

命题(1)的证明。任取一个(k - 2)元组$t \in  \mathcal{J}\left( {{R}_{1}^{\prime },{R}_{2}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$。设${r}_{i}\left( {i \in  \left\lbrack  {k - 2}\right\rbrack  }\right)$是$t\left\lbrack  i\right\rbrack$的完整矩形。固定任意$h \in  \mathrm{d} - {\operatorname{cross}}_{H}\left( t\right)$和任意$v \in  \mathrm{d} - {\operatorname{cross}}_{V}\left( t\right)$。线段$h$和$v$都与${B}_{t}$相交，这可直接从$\mathrm{d} - {\operatorname{cross}}_{H}\left( t\right)$和$\mathrm{d} - {\operatorname{cross}}_{V}\left( t\right)$的定义看出。因此，$h \cap  v$是${B}_{t} = \mathop{\bigcap }\limits_{{i = 1}}^{{k - 2}}t\left\lbrack  i\right\rbrack$中的一个点。由于${r}_{i}\left( {i \in  \left\lbrack  {k - 2}\right\rbrack  }\right)$是$t\left\lbrack  i\right\rbrack$的完整矩形，我们知道$\mathop{\bigcap }\limits_{{i = 1}}^{{k - 2}}{r}_{i}$覆盖${B}_{t} = \mathop{\bigcap }\limits_{{i = 1}}^{{k - 2}}t\left\lbrack  i\right\rbrack$。因此，点$h \cap  v$落在$\mathop{\bigcap }\limits_{{i = 1}}^{{k - 2}}{r}_{i}$内，这表明$h \cap  v \cap  \mathop{\bigcap }\limits_{{i = 1}}^{{k - 2}}{r}_{i} \neq  \varnothing$。由此可知，$\left( {{r}_{1},\ldots ,{r}_{k - 2},h,v}\right)$是$\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 2},H,V}\right)$中的一个结果元组。

Proof of Statement (2). Take any $k$ -tuple $\left( {{r}_{1},\ldots ,{r}_{k - 2},h,v}\right)  \in  {\mathcal{J}}_{1}$ . By definition of ${\mathcal{J}}_{1}$ ,segments $h$ and $v$ cross each of the rectangles ${r}_{1},\ldots ,{r}_{k - 2}$ . By Proposition B.1,segments $h$ and $v$ must also cross each of the trimmed rectangles ${r}_{1}^{\prime },\ldots ,{r}_{k - 2}^{\prime }$ . Hence, ${B}_{t} = \mathop{\bigcap }\limits_{{i = 1}}^{{k - 2}}{r}_{i}^{\prime }$ is non-empty as it contains the point $h \cap  v$ ,which proves the first claim $\mathbf{t} = \left( {{r}_{1}^{\prime },\ldots ,{r}_{k - 2}^{\prime }}\right)  \in  \mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$ .

命题 (2) 的证明。任取一个 $k$ -元组 $\left( {{r}_{1},\ldots ,{r}_{k - 2},h,v}\right)  \in  {\mathcal{J}}_{1}$ 。根据 ${\mathcal{J}}_{1}$ 的定义，线段 $h$ 和 $v$ 穿过每个矩形 ${r}_{1},\ldots ,{r}_{k - 2}$ 。根据命题 B.1，线段 $h$ 和 $v$ 也必定穿过每个修剪后的矩形 ${r}_{1}^{\prime },\ldots ,{r}_{k - 2}^{\prime }$ 。因此，${B}_{t} = \mathop{\bigcap }\limits_{{i = 1}}^{{k - 2}}{r}_{i}^{\prime }$ 非空，因为它包含点 $h \cap  v$ ，这就证明了第一个断言 $\mathbf{t} = \left( {{r}_{1}^{\prime },\ldots ,{r}_{k - 2}^{\prime }}\right)  \in  \mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$ 。

Next,we prove the second claim,i.e., $h \in  \mathrm{d} - {\operatorname{cross}}_{H}\left( t\right)$ and $v \in  \mathrm{d} - {\operatorname{cross}}_{V}\left( t\right)$ . It suffices to show only the former due to symmetry. For that purpose,we need to argue that $h$ crosses bot-guard(t) and ${B}_{t}$ . The first part, $h$ crossing bot-guard(t),is done because as mentioned $h$ crosses each of ${r}_{1}^{\prime },\ldots ,{r}_{k - 2}^{\prime }$ ,and bot-guard(t)is merely one of those rectangles. To prove that $h$ crosses ${B}_{t}$ ,first note that $h \cap  {B}_{t} \neq  \varnothing$ because as explained before ${B}_{t}$ contains $h \cap  v$ . On the other hand,as bot-guard(t) covers ${B}_{t}$ ,the fact of $h$ crossing bot-guard(t)indicates that the x-range of $h$ contains that of ${B}_{t}$ . Combining this with $h \cap  {B}_{t} \neq  \varnothing$ shows that $h$ crosses ${B}_{t}$ .

接下来，我们证明第二个断言，即 $h \in  \mathrm{d} - {\operatorname{cross}}_{H}\left( t\right)$ 和 $v \in  \mathrm{d} - {\operatorname{cross}}_{V}\left( t\right)$ 。由于对称性，只需证明前者即可。为此，我们需要论证 $h$ 穿过 bot - guard(t) 和 ${B}_{t}$ 。第一部分，$h$ 穿过 bot - guard(t)，这已经得到证明，因为如前所述，$h$ 穿过每个 ${r}_{1}^{\prime },\ldots ,{r}_{k - 2}^{\prime }$ ，而 bot - guard(t) 只是其中一个矩形。为了证明 $h$ 穿过 ${B}_{t}$ ，首先注意到 $h \cap  {B}_{t} \neq  \varnothing$ ，因为如前所述 ${B}_{t}$ 包含 $h \cap  v$ 。另一方面，由于 bot - guard(t) 覆盖 ${B}_{t}$ ，$h$ 穿过 bot - guard(t) 这一事实表明 $h$ 的 x 范围包含 ${B}_{t}$ 的 x 范围。将此与 $h \cap  {B}_{t} \neq  \varnothing$ 相结合表明 $h$ 穿过 ${B}_{t}$ 。

Proof of Statement (3). We will prove

命题 (3) 的证明。我们将证明

$$
\mathop{\sum }\limits_{{\mathbf{t} \in  \mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right) }}\left| {\mathrm{d} - {\operatorname{cross}}_{H}\left( \mathbf{t}\right) }\right|  \cdot  \left| {\mathrm{d} - {\operatorname{cross}}_{V}\left( \mathbf{t}\right) }\right|  \leq  \mathrm{{OUT}} \tag{17}
$$

which implies statement (3) because,by Proposition B.2, $\left| {\mathrm{d} - {\operatorname{cross}}_{H}\left( t\right) }\right|  \geq  1$ and $\left| {\mathrm{d} - {\operatorname{cross}}_{V}\left( t\right) }\right|  \geq  1$ for any $t$ in the summation. Our proof resorts to the algorithm generate- ${\mathcal{J}}^{ * }$ given in Section 4. This algorithm adds to ${\mathcal{J}}^{ * }$ as many tuples as calculated by the left hand side of (17). By statement (1) of Lemma 4.2,the ${\mathcal{J}}^{ * }$ produced must be a subset of $\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 2},H,V}\right)$ ,whereas OUT = $\left| {\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 2},H,V}\right) }\right|$ . This establishes the inequality in (17).

这意味着命题 (3) 成立，因为根据命题 B.2，对于求和中的任何 $t$ ，有 $\left| {\mathrm{d} - {\operatorname{cross}}_{H}\left( t\right) }\right|  \geq  1$ 和 $\left| {\mathrm{d} - {\operatorname{cross}}_{V}\left( t\right) }\right|  \geq  1$ 。我们的证明借助于第 4 节给出的算法 generate - ${\mathcal{J}}^{ * }$ 。该算法向 ${\mathcal{J}}^{ * }$ 中添加的元组数量与 (17) 式左边计算的数量相同。根据引理 4.2 的命题 (1)，生成的 ${\mathcal{J}}^{ * }$ 必定是 $\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 2},H,V}\right)$ 的一个子集，而 OUT = $\left| {\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 2},H,V}\right) }\right|$ 。这就确立了 (17) 式中的不等式。

Proof of Lemma 4.1. Our proof again resorts to the algorithm generate- ${\mathcal{J}}^{ * }$ in Section 4. By Proposition B.2,both $\mathrm{d} - {\operatorname{cross}}_{H}\left( t\right)$ and $\mathrm{d} - {\operatorname{cross}}_{V}\left( t\right)$ are non-empty for any $t \in  \mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$ . Therefore,in processing this $t$ ,the algorithm adds at least one new tuple to ${\mathcal{J}}^{ * }$ . Hence, $\left| {\mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right) }\right|  \leq  \left| {\mathcal{J}}^{ * }\right|  \leq$ $\left| {\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 2},H,V}\right) }\right|  =$ OUT,where the second inequality used Statement (1) of Lemma 4.2.

引理4.1的证明。我们的证明再次借助第4节中的算法generate - ${\mathcal{J}}^{ * }$。根据命题B.2，对于任意$t \in  \mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$，$\mathrm{d} - {\operatorname{cross}}_{H}\left( t\right)$和$\mathrm{d} - {\operatorname{cross}}_{V}\left( t\right)$均非空。因此，在处理这个$t$时，该算法至少会向${\mathcal{J}}^{ * }$添加一个新元组。因此，$\left| {\mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right) }\right|  \leq  \left| {\mathcal{J}}^{ * }\right|  \leq$ $\left| {\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 2},H,V}\right) }\right|  =$ OUT，其中第二个不等式使用了引理4.2的陈述(1)。

Computing ${R}_{1}^{ * },{R}_{2}^{ * },\ldots ,{R}_{k - 2}^{ * }$ . We consider,w.l.o.g.,that each rectangle in the input ${R}_{1} \cup  \ldots  \cup  {R}_{k - 2}$ is given a distinct integer ID in $\left\lbrack  n\right\rbrack$ . This allows us to create an array of size $n$ and allocate an array cell to each $r \in  {R}_{1} \cup  \ldots  \cup  {R}_{k - 2}$ . The cell can be accessed by the ID of $r$ in constant time.

计算${R}_{1}^{ * },{R}_{2}^{ * },\ldots ,{R}_{k - 2}^{ * }$。不失一般性，我们考虑为输入${R}_{1} \cup  \ldots  \cup  {R}_{k - 2}$中的每个矩形赋予一个在$\left\lbrack  n\right\rbrack$范围内的不同整数ID。这使我们能够创建一个大小为$n$的数组，并为每个$r \in  {R}_{1} \cup  \ldots  \cup  {R}_{k - 2}$分配一个数组单元。可以通过$r$的ID在常数时间内访问该单元。

To compute ${R}_{1}^{ * },{R}_{2}^{ * },\ldots ,{R}_{k - 2}^{ * }$ ,we start by deriving $\operatorname{maxtop}\left( {r}^{\prime }\right)$ for each rectangle ${r}^{\prime }$ in ${R}_{1}^{\prime } \cup  \ldots  \cup  {R}_{k - 2}^{\prime }$ . For this purpose,first initialize $\operatorname{maxtop}\left( {r}^{\prime }\right)  =  - \infty$ for each such ${r}^{\prime }$ . Recall that ${r}^{\prime }$ is the trimmed rectangle of some rectangle $r$ in ${R}_{1} \cup  \ldots  \cup  {R}_{k - 2}$ . We store maxtop $\left( {r}^{\prime }\right)$ in the array cell allocated to $r$ . Then,we scan $\mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$ . For each tuple $t$ therein,use $O\left( k\right)$ time to identify the rectangle ${r}^{\prime } =$ bot-guard(t),and then update in constant time maxtop $\left( {r}^{\prime }\right)$ to the maximum between its current value and $\operatorname{top}\left( {r}^{\prime }\right)$ . The scan requires $O\left( {n + k \cdot  \mathrm{{OUT}}}\right)$ time.

为了计算${R}_{1}^{ * },{R}_{2}^{ * },\ldots ,{R}_{k - 2}^{ * }$，我们首先为${R}_{1}^{\prime } \cup  \ldots  \cup  {R}_{k - 2}^{\prime }$中的每个矩形${r}^{\prime }$推导$\operatorname{maxtop}\left( {r}^{\prime }\right)$。为此，首先为每个这样的${r}^{\prime }$初始化$\operatorname{maxtop}\left( {r}^{\prime }\right)  =  - \infty$。回想一下，${r}^{\prime }$是${R}_{1} \cup  \ldots  \cup  {R}_{k - 2}$中某个矩形$r$的修剪后的矩形。我们将maxtop $\left( {r}^{\prime }\right)$存储在为$r$分配的数组单元中。然后，我们扫描$\mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$。对于其中的每个元组$t$，使用$O\left( k\right)$时间来识别矩形${r}^{\prime } =$ bot - guard(t)，然后在常数时间内将maxtop $\left( {r}^{\prime }\right)$更新为其当前值和$\operatorname{top}\left( {r}^{\prime }\right)$之间的最大值。扫描需要$O\left( {n + k \cdot  \mathrm{{OUT}}}\right)$时间。

Finally,for each $i \in  \left\lbrack  {k - 2}\right\rbrack$ ,we construct ${R}_{i}^{ * }$ by collecting the top-sliced rectangle (see definition in (9)) of every rectangle ${r}^{\prime } \in  {R}_{i}^{\prime }$ with $\operatorname{maxtop}\left( {r}^{\prime }\right)  \neq   - \infty$ . This step takes $O\left( \left| {R}_{i}^{\prime }\right| \right)$ time for each $i \in  \left\lbrack  {k - 2}\right\rbrack$ ,or $O\left( n\right)$ total time for all $i$ .

最后，对于每个 $i \in  \left\lbrack  {k - 2}\right\rbrack$，我们通过收集每个满足 $\operatorname{maxtop}\left( {r}^{\prime }\right)  \neq   - \infty$ 的矩形 ${r}^{\prime } \in  {R}_{i}^{\prime }$ 的顶部切片矩形（见 (9) 中的定义）来构造 ${R}_{i}^{ * }$。这一步对于每个 $i \in  \left\lbrack  {k - 2}\right\rbrack$ 需要 $O\left( \left| {R}_{i}^{\prime }\right| \right)$ 的时间，或者对于所有 $i$ 总共需要 $O\left( n\right)$ 的时间。

Proof of Lemma 4.3. We will prove each statement in turn.

引理 4.3 的证明。我们将依次证明每个陈述。

Proof of Statement (1). We will map each ${r}^{ * }$ of $\mathop{\bigcup }\limits_{{i = 1}}^{{k - 2}}{R}_{i}^{ * }$ to a unique tuple $t$ in $\mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$ satisfying ${\operatorname{cross}}_{H}\left( {r}^{ * }\right)  \subseteq  \mathrm{d} - {\operatorname{cross}}_{H}\left( t\right)$ . The mapping allows us to derive

陈述 (1) 的证明。我们将把 $\mathop{\bigcup }\limits_{{i = 1}}^{{k - 2}}{R}_{i}^{ * }$ 中的每个 ${r}^{ * }$ 映射到 $\mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$ 中满足 ${\operatorname{cross}}_{H}\left( {r}^{ * }\right)  \subseteq  \mathrm{d} - {\operatorname{cross}}_{H}\left( t\right)$ 的唯一元组 $t$。该映射使我们能够推导出

$$
\mathop{\sum }\limits_{{i \in  \left\lbrack  {k - 2}\right\rbrack  }}\mathop{\sum }\limits_{{{r}^{ * } \in  {R}_{i}^{ * }}}\left| {{\operatorname{cross}}_{H}\left( {r}^{ * }\right) }\right|  \leq  \mathop{\sum }\limits_{{\mathbf{t} \in  \mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right) }}\left| {\mathrm{d} - {\operatorname{cross}}_{H}\left( \mathbf{t}\right) }\right|  \leq  \mathrm{{OUT}}
$$

where the last step used statement (3) of Lemma 4.2.

其中最后一步使用了引理 4.2 的陈述 (3)。

The mapping is as follows. Consider an arbitrary ${r}^{ * } \in  \mathop{\bigcup }\limits_{{i = 1}}^{{k - 2}}{R}_{i}^{ * }$ . Recall that ${r}^{ * }$ is the top-sliced of some rectangle ${r}^{\prime }$ . Specifically,if ${r}^{\prime } = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  \left\lbrack  {{y}_{1},{y}_{2}}\right\rbrack$ ,then ${r}^{ * } = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  \left\lbrack  {{y}_{1},\operatorname{maxtop}\left( {r}^{\prime }\right) }\right\rbrack$ . By the definition of $\operatorname{maxtop}\left( {r}^{\prime }\right)$ in (9),there exists a tuple $t \in  \mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$ satisfying bot-guard $\left( t\right)  = {r}^{\prime }$ and $\operatorname{maxtop}\left( {r}^{\prime }\right)  = \operatorname{top}\left( {B}_{t}\right)$ . We map ${r}^{ * }$ to $t$ .

映射如下。考虑任意的 ${r}^{ * } \in  \mathop{\bigcup }\limits_{{i = 1}}^{{k - 2}}{R}_{i}^{ * }$。回想一下，${r}^{ * }$ 是某个矩形 ${r}^{\prime }$ 的顶部切片。具体来说，如果 ${r}^{\prime } = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  \left\lbrack  {{y}_{1},{y}_{2}}\right\rbrack$，那么 ${r}^{ * } = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  \left\lbrack  {{y}_{1},\operatorname{maxtop}\left( {r}^{\prime }\right) }\right\rbrack$。根据 (9) 中 $\operatorname{maxtop}\left( {r}^{\prime }\right)$ 的定义，存在一个元组 $t \in  \mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$ 满足底部防护 $\left( t\right)  = {r}^{\prime }$ 和 $\operatorname{maxtop}\left( {r}^{\prime }\right)  = \operatorname{top}\left( {B}_{t}\right)$。我们将 ${r}^{ * }$ 映射到 $t$。

Next, we will prove

接下来，我们将证明

Claim 1: If a segment $h \in  {\operatorname{cross}}_{H}\left( {r}^{ * }\right)$ ,then $h \in  \mathrm{d} - {\operatorname{cross}}_{H}\left( \mathbf{t}\right)$ .

断言 1：如果线段 $h \in  {\operatorname{cross}}_{H}\left( {r}^{ * }\right)$，那么 $h \in  \mathrm{d} - {\operatorname{cross}}_{H}\left( \mathbf{t}\right)$。

For this purpose,we need to show that $h$ crosses both bot-guard $\left( \mathbf{t}\right)  = {r}^{\prime }$ and ${B}_{\mathbf{t}}$ .

为此，我们需要证明 $h$ 同时穿过底部防护 $\left( \mathbf{t}\right)  = {r}^{\prime }$ 和 ${B}_{\mathbf{t}}$。

- First, $h$ crosses ${r}^{\prime }$ follows from the facts that (i) $h$ crosses ${r}^{ * }$ ,(ii) ${r}^{ * } \subseteq  {r}^{\prime }$ ,and (iii) ${r}^{ * }$ and ${r}^{\prime }$ share the same x-range.

- 首先，$h$ 穿过 ${r}^{\prime }$ 可由以下事实得出：(i) $h$ 穿过 ${r}^{ * }$，(ii) ${r}^{ * } \subseteq  {r}^{\prime }$，以及 (iii) ${r}^{ * }$ 和 ${r}^{\prime }$ 具有相同的 x 范围。

- Then,we explain why $h$ crosses ${B}_{t}$ . Because ${B}_{t} \subseteq  {r}^{\prime }$ and $h$ crosses ${r}^{\prime }$ (just proved),no endpoint of $h$ can fall in ${B}_{t}$ . Let us take the y-coordinate ${y}_{h}$ of $h$ . We will prove that the point $p = \left( {\operatorname{left}\left( {B}_{t}\right) ,{y}_{h}}\right)$ is on the segment $h$ and is also in ${B}_{t}$ ,suggesting $h \cap  {B}_{t} \neq  \varnothing$ . Once this is done,we can assert that $h$ crosses ${B}_{t}$ (as no endpoint of $h$ falls in ${B}_{t}$ ).

- 然后，我们解释为什么$h$与${B}_{t}$相交。因为${B}_{t} \subseteq  {r}^{\prime }$且$h$与${r}^{\prime }$相交（刚刚已证明），所以$h$的任何端点都不能落在${B}_{t}$内。我们取$h$的y坐标${y}_{h}$。我们将证明点$p = \left( {\operatorname{left}\left( {B}_{t}\right) ,{y}_{h}}\right)$在线段$h$上且也在${B}_{t}$内，这意味着$h \cap  {B}_{t} \neq  \varnothing$。一旦完成这一步，我们就可以断言$h$与${B}_{t}$相交（因为$h$的任何端点都不在${B}_{t}$内）。

- As bot-guard $\left( t\right)  = {r}^{\prime }$ ,the bottom edge of ${B}_{t}$ is contained in ${r}^{\prime }$ . Thus,left $\left( {B}_{t}\right)  \in  \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$ . The fact of $h$ crossing ${r}^{ * }$ tells us that $\operatorname{left}\left( h\right)  < {x}_{1} \leq  {x}_{2} < \operatorname{right}\left( h\right)$ ,which leads to $\operatorname{left}\left( {B}_{t}\right)  \in  \left\lbrack  {\operatorname{left}\left( h\right) ,\operatorname{right}\left( h\right) }\right\rbrack$ ,suggesting that $p \in  h$ .

- 作为底部边界$\left( t\right)  = {r}^{\prime }$，${B}_{t}$的底边包含在${r}^{\prime }$内。因此，左侧为$\left( {B}_{t}\right)  \in  \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$。$h$与${r}^{ * }$相交这一事实告诉我们$\operatorname{left}\left( h\right)  < {x}_{1} \leq  {x}_{2} < \operatorname{right}\left( h\right)$，这导致$\operatorname{left}\left( {B}_{t}\right)  \in  \left\lbrack  {\operatorname{left}\left( h\right) ,\operatorname{right}\left( h\right) }\right\rbrack$，意味着$p \in  h$。

- As bot-guard $\left( t\right)  = {r}^{\prime }$ and $\operatorname{maxtop}\left( {r}^{\prime }\right)  = \operatorname{top}\left( {B}_{t}\right)$ ,the y-range of ${B}_{t}$ is $\left\lbrack  {{y}_{1},\operatorname{maxtop}\left( {r}^{\prime }\right) }\right\rbrack$ . Because $h \cap  {r}^{ * } \neq  \varnothing$ (as $h$ crosses ${r}^{ * }$ ) and $\left\lbrack  {{y}_{1},\operatorname{maxtop}\left( {r}^{\prime }\right) }\right\rbrack$ is also the y-range of ${r}^{ * }$ ,we know that ${y}_{h} \in  \left\lbrack  {{y}_{1},\operatorname{maxtop}\left( {r}^{\prime }\right) }\right\rbrack$ ,suggesting that $p \in  {B}_{t}$ .

- 作为底部边界$\left( t\right)  = {r}^{\prime }$且$\operatorname{maxtop}\left( {r}^{\prime }\right)  = \operatorname{top}\left( {B}_{t}\right)$，${B}_{t}$的y范围是$\left\lbrack  {{y}_{1},\operatorname{maxtop}\left( {r}^{\prime }\right) }\right\rbrack$。因为$h \cap  {r}^{ * } \neq  \varnothing$（因为$h$与${r}^{ * }$相交）且$\left\lbrack  {{y}_{1},\operatorname{maxtop}\left( {r}^{\prime }\right) }\right\rbrack$也是${r}^{ * }$的y范围，我们知道${y}_{h} \in  \left\lbrack  {{y}_{1},\operatorname{maxtop}\left( {r}^{\prime }\right) }\right\rbrack$，这意味着$p \in  {B}_{t}$。

It remains to show that no two distinct rectangles ${r}_{1}^{ * },{r}_{2}^{ * } \in  \mathop{\bigcup }\limits_{{i = 1}}^{{k - 2}}{R}_{i}^{ * }$ can be mapped to the same tuple in $\mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$ . Assume,on the contrary,that ${r}_{1}^{ * }$ and ${r}_{2}^{ * }$ are mapped to the same tuple $t \in  \mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$ . Suppose that ${r}_{1}^{ * }$ (resp., ${r}_{2}^{ * }$ ) is the top-sliced rectangle of ${r}_{1}^{\prime }$ (resp., ${r}_{2}^{\prime }$ ). Under our mapping,it must be true that ${r}_{1}^{\prime } = {r}_{2}^{\prime } =$ bot-guard $\left( {B}_{t}\right)$ . However,the distinctness of ${r}_{1}^{ * }$ and ${r}_{2}^{ * }$ requires ${r}_{1}^{\prime } \neq  {r}_{2}^{\prime }$ ,thus giving a contradiction.

接下来需要证明，没有两个不同的矩形 ${r}_{1}^{ * },{r}_{2}^{ * } \in  \mathop{\bigcup }\limits_{{i = 1}}^{{k - 2}}{R}_{i}^{ * }$ 可以映射到 $\mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$ 中的同一个元组。相反，假设 ${r}_{1}^{ * }$ 和 ${r}_{2}^{ * }$ 被映射到同一个元组 $t \in  \mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$。假设 ${r}_{1}^{ * }$（分别地，${r}_{2}^{ * }$）是 ${r}_{1}^{\prime }$（分别地，${r}_{2}^{\prime }$）的顶部切片矩形。在我们的映射下，必然有 ${r}_{1}^{\prime } = {r}_{2}^{\prime } =$ 底部防护 $\left( {B}_{t}\right)$。然而，${r}_{1}^{ * }$ 和 ${r}_{2}^{ * }$ 的不同要求 ${r}_{1}^{\prime } \neq  {r}_{2}^{\prime }$，从而产生矛盾。

Proof of Statement (2). Take any tuple $t \in  \mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$ . Set ${r}^{\prime } =$ bot-guard(t),and let ${r}^{ * }$ be the top-sliced rectangle of ${r}^{\prime }$ . If we represent ${r}^{\prime }$ as $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  \left\lbrack  {{y}_{1},{y}_{2}}\right\rbrack$ ,then ${r}^{ * }$ can be written as $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  \left\lbrack  {{y}_{1},\operatorname{maxtop}\left( {r}^{\prime }\right) }\right\rbrack$ .

陈述 (2) 的证明。取任意元组 $t \in  \mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 2}^{\prime }}\right)$。设 ${r}^{\prime } =$ 为底部防护(t)，并设 ${r}^{ * }$ 为 ${r}^{\prime }$ 的顶部切片矩形。如果我们将 ${r}^{\prime }$ 表示为 $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  \left\lbrack  {{y}_{1},{y}_{2}}\right\rbrack$，那么 ${r}^{ * }$ 可以写成 $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  \left\lbrack  {{y}_{1},\operatorname{maxtop}\left( {r}^{\prime }\right) }\right\rbrack$。

We will first prove $d - {\operatorname{cross}}_{H}\left( t\right)  \subseteq  {\operatorname{cross}}_{H}\left( {r}^{ * }\right)$ or equivalently: if a segment $h \in  d - {\operatorname{cross}}_{H}\left( t\right)$ , then $h$ crosses ${r}^{ * }$ . To do so,we need to explain why $\operatorname{left}\left( h\right)  < {x}_{1} \leq  {x}_{2} < \operatorname{right}\left( h\right)$ and ${y}_{h} \in  \left\lbrack  {y}_{1}\right.$ , $\left. {\operatorname{maxtop}\left( {r}^{\prime }\right) }\right\rbrack$ .

我们首先将证明 $d - {\operatorname{cross}}_{H}\left( t\right)  \subseteq  {\operatorname{cross}}_{H}\left( {r}^{ * }\right)$，或者等价地：如果线段 $h \in  d - {\operatorname{cross}}_{H}\left( t\right)$，那么 $h$ 与 ${r}^{ * }$ 相交。为此，我们需要解释为什么 $\operatorname{left}\left( h\right)  < {x}_{1} \leq  {x}_{2} < \operatorname{right}\left( h\right)$ 以及 ${y}_{h} \in  \left\lbrack  {y}_{1}\right.$，$\left. {\operatorname{maxtop}\left( {r}^{\prime }\right) }\right\rbrack$。

- The fact of $h \in  \mathrm{d} - {\operatorname{cross}}_{H}\left( t\right)$ tells us that $h$ crosses bot-guard(t),which is ${r}^{\prime }$ . As the x-range of ${r}^{\prime }$ is $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$ ,it must hold true that $\operatorname{left}\left( h\right)  < {x}_{1} \leq  {x}_{2} < \operatorname{right}\left( h\right)$ .

- $h \in  \mathrm{d} - {\operatorname{cross}}_{H}\left( t\right)$ 这一事实告诉我们，$h$ 与底部防护(t)相交，即 ${r}^{\prime }$。由于 ${r}^{\prime }$ 的 x 范围是 $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$，必然有 $\operatorname{left}\left( h\right)  < {x}_{1} \leq  {x}_{2} < \operatorname{right}\left( h\right)$ 成立。

- The fact of $h \in  \mathrm{d} - {\operatorname{cross}}_{H}\left( \mathbf{t}\right)$ also tells us that $h$ crosses ${B}_{t}$ . Hence, ${y}_{h} \in  \left\lbrack  {\operatorname{bot}\left( {B}_{t}\right) ,\operatorname{top}\left( {B}_{t}\right) }\right\rbrack$ . From ${r}^{\prime } =$ bot-guard(t),we get $\operatorname{bot}\left( {B}_{t}\right)  = \operatorname{bot}\left( {r}^{\prime }\right)  = {y}_{1}$ . By the definition of maxtop $\left( {r}^{\prime }\right)$ in (8),we know $\operatorname{top}\left( {B}_{t}\right)  \leq  \operatorname{maxtop}\left( {r}^{\prime }\right)$ . It thus follows that ${y}_{h} \in  \left\lbrack  {{y}_{1},\operatorname{maxtop}\left( {r}^{\prime }\right) }\right\rbrack$ .

- $h \in  \mathrm{d} - {\operatorname{cross}}_{H}\left( \mathbf{t}\right)$这一事实也告诉我们，$h$与${B}_{t}$相交。因此，${y}_{h} \in  \left\lbrack  {\operatorname{bot}\left( {B}_{t}\right) ,\operatorname{top}\left( {B}_{t}\right) }\right\rbrack$。从${r}^{\prime } =$ bot - guard(t)，我们得到$\operatorname{bot}\left( {B}_{t}\right)  = \operatorname{bot}\left( {r}^{\prime }\right)  = {y}_{1}$。根据(8)中maxtop $\left( {r}^{\prime }\right)$的定义，我们知道$\operatorname{top}\left( {B}_{t}\right)  \leq  \operatorname{maxtop}\left( {r}^{\prime }\right)$。由此可得${y}_{h} \in  \left\lbrack  {{y}_{1},\operatorname{maxtop}\left( {r}^{\prime }\right) }\right\rbrack$。

Next,assuming that the segments of ${\operatorname{cross}}_{H}\left( {r}^{ * }\right)$ are sorted in ascending order of their y-coordinates, we will prove that $\mathrm{d} - {\operatorname{cross}}_{H}\left( t\right)$ includes a prefix of the sorted order. It suffices to establish the following equivalent claim:

接下来，假设${\operatorname{cross}}_{H}\left( {r}^{ * }\right)$的线段按其y坐标升序排列，我们将证明$\mathrm{d} - {\operatorname{cross}}_{H}\left( t\right)$包含排序顺序的一个前缀。只需证明以下等价命题即可：

Claim 2: If a segment $h \in  {\operatorname{cross}}_{H}\left( {r}^{ * }\right)$ has y-coordinate ${y}_{h} \leq  \operatorname{top}\left( {B}_{t}\right)$ ,then $h$ must be in $\mathrm{d} - {\operatorname{cross}}_{H}\left( \mathbf{t}\right)$ .

命题2：如果线段$h \in  {\operatorname{cross}}_{H}\left( {r}^{ * }\right)$的y坐标为${y}_{h} \leq  \operatorname{top}\left( {B}_{t}\right)$，那么$h$必定在$\mathrm{d} - {\operatorname{cross}}_{H}\left( \mathbf{t}\right)$中。

To prove the above,we must explain why $h$ crosses both ${r}^{\prime }$ and ${B}_{t}$ .

为了证明上述命题，我们必须解释为什么$h$与${r}^{\prime }$和${B}_{t}$都相交。

- We first show that $h$ crosses ${r}^{\prime }$ ,or equivalently: $\operatorname{left}\left( h\right)  < {x}_{1} \leq  {x}_{2} < \operatorname{right}\left( h\right)$ and ${y}_{h} \in  \left\lbrack  {{y}_{1},{y}_{2}}\right\rbrack$ . These conditions hold true because (i) $h$ crosses ${r}^{ * } = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  \left\lbrack  {{y}_{1},\operatorname{maxtop}\left( {r}^{\prime }\right) }\right\rbrack$ ,and (ii) the definition of a top-sliced rectangle in (9) tells us maxtop $\left( {r}^{\prime }\right)  \leq  {y}_{2}$ .

- 我们首先证明$h$与${r}^{\prime }$相交，或者等价地：$\operatorname{left}\left( h\right)  < {x}_{1} \leq  {x}_{2} < \operatorname{right}\left( h\right)$和${y}_{h} \in  \left\lbrack  {{y}_{1},{y}_{2}}\right\rbrack$。这些条件成立是因为：(i) $h$与${r}^{ * } = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  \left\lbrack  {{y}_{1},\operatorname{maxtop}\left( {r}^{\prime }\right) }\right\rbrack$相交，并且(ii) (9)中顶部切片矩形的定义告诉我们maxtop $\left( {r}^{\prime }\right)  \leq  {y}_{2}$。

- Next we show that $h$ crosses ${B}_{t}$ ,or equivalently: $\operatorname{left}\left( h\right)  < \operatorname{left}\left( {B}_{t}\right)  \leq  \operatorname{right}\left( {B}_{t}\right)  < \operatorname{right}\left( h\right)$ and ${y}_{h} \in  \left\lbrack  {\operatorname{bot}\left( {B}_{t}\right) ,\operatorname{top}\left( {B}_{t}\right) }\right\rbrack   = \left\lbrack  {{y}_{1},\operatorname{top}\left( {B}_{t}\right) }\right\rbrack$ .

- 接下来我们证明$h$与${B}_{t}$相交，或者等价地：$\operatorname{left}\left( h\right)  < \operatorname{left}\left( {B}_{t}\right)  \leq  \operatorname{right}\left( {B}_{t}\right)  < \operatorname{right}\left( h\right)$和${y}_{h} \in  \left\lbrack  {\operatorname{bot}\left( {B}_{t}\right) ,\operatorname{top}\left( {B}_{t}\right) }\right\rbrack   = \left\lbrack  {{y}_{1},\operatorname{top}\left( {B}_{t}\right) }\right\rbrack$。

- The fact of $h$ crossing ${r}^{\prime }$ tells us $\operatorname{left}\left( h\right)  < {x}_{1} \leq  {x}_{2} < \operatorname{right}\left( h\right)$ . As ${r}^{\prime } = \operatorname{bot-guard}\left( {B}_{t}\right)$ ,the $\mathrm{x}$ -range of ${B}_{t}$ must be contained in $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$ . Thus, $\operatorname{left}\left( h\right)  < \operatorname{left}\left( {B}_{t}\right)  \leq  \operatorname{right}\left( {B}_{t}\right)  < \operatorname{right}\left( h\right)$ .

- $h$ 与 ${r}^{\prime }$ 相交这一事实告诉我们 $\operatorname{left}\left( h\right)  < {x}_{1} \leq  {x}_{2} < \operatorname{right}\left( h\right)$。由于 ${r}^{\prime } = \operatorname{bot-guard}\left( {B}_{t}\right)$，${B}_{t}$ 的 $\mathrm{x}$ 范围必定包含在 $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$ 内。因此，$\operatorname{left}\left( h\right)  < \operatorname{left}\left( {B}_{t}\right)  \leq  \operatorname{right}\left( {B}_{t}\right)  < \operatorname{right}\left( h\right)$。

- The fact of $h$ crossing ${r}^{ * }$ also tells us that ${y}_{h} \in  \left\lbrack  {{y}_{1},\operatorname{maxtop}\left( {r}^{\prime }\right) }\right\rbrack$ . Moreover,Claim 2 explicitly gives us the condition ${y}_{h} \leq  \operatorname{top}\left( {B}_{t}\right)$ . It thus follows that ${y}_{h} \in  \left\lbrack  {{y}_{1},\operatorname{top}\left( {B}_{t}\right) }\right\rbrack$ .

- $h$ 与 ${r}^{ * }$ 相交这一事实还告诉我们 ${y}_{h} \in  \left\lbrack  {{y}_{1},\operatorname{maxtop}\left( {r}^{\prime }\right) }\right\rbrack$。此外，命题 2 明确给出了条件 ${y}_{h} \leq  \operatorname{top}\left( {B}_{t}\right)$。由此可得 ${y}_{h} \in  \left\lbrack  {{y}_{1},\operatorname{top}\left( {B}_{t}\right) }\right\rbrack$。

## C Supplementary Proofs for Section 5

## C 第 5 节的补充证明

We start by presenting two properties underneath the procedures designed in Section 5. These properties will enable us to construct simpler proofs later.

我们首先介绍第 5 节所设计过程背后的两个性质。这些性质将使我们后续能够构建更简单的证明。

Proposition C.1. Consider any $k$ -tuple $\left( {{r}_{1},\ldots ,{r}_{k - 2},h,v}\right)  \in  {\mathcal{J}}_{2}$ . Define ${h}^{\prime }$ as the trimmed segment of $h$ . It holds that $h \cap  v = {h}^{\prime } \cap  v$ .

命题 C.1。考虑任意 $k$ 元组 $\left( {{r}_{1},\ldots ,{r}_{k - 2},h,v}\right)  \in  {\mathcal{J}}_{2}$。将 ${h}^{\prime }$ 定义为 $h$ 的修剪段。有 $h \cap  v = {h}^{\prime } \cap  v$ 成立。

Proof. Set $p = h \cap  v$ . As ${h}^{\prime }$ is contained in $h$ ,for proving $p = {h}^{\prime } \cap  v$ ,it suffices to explain why $p$ is on ${h}^{\prime }$ . Let us represent $h$ as $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  y$ . Accordingly,the segment ${h}^{\prime }$ can be written as $\left\lbrack  {{x}_{1},{x}_{2}^{\prime }}\right\rbrack   \times  y$ for some ${x}_{2}^{\prime } \in  \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$ . By the definition of a trimmed segment, $\left( {{x}_{2}^{\prime },y}\right)$ is the rightmost point on $h$ that falls in a left-end covering rectangle of $h$ in ${R}_{k - 2}$ . As $\left( {{r}_{1},\ldots ,{r}_{k - 2},h,v}\right)  \in  {\mathcal{J}}_{2}$ ,we know that ${r}_{k - 2}$ is a left-end covering rectangle of $h$ ,and ${r}_{k - 2}$ covers $p$ (which is a point on $h$ ). Therefore,the $\mathrm{x}$ -coordinate of $p$ cannot exceed ${x}_{2}^{\prime }$ ,allowing us to assert that $p$ is on ${h}^{\prime }$ .

证明。设 $p = h \cap  v$。由于 ${h}^{\prime }$ 包含在 $h$ 中，为证明 $p = {h}^{\prime } \cap  v$，只需解释为什么 $p$ 在 ${h}^{\prime }$ 上。我们将 $h$ 表示为 $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  y$。相应地，线段 ${h}^{\prime }$ 可以写成 $\left\lbrack  {{x}_{1},{x}_{2}^{\prime }}\right\rbrack   \times  y$，其中 ${x}_{2}^{\prime } \in  \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$。根据修剪段的定义，$\left( {{x}_{2}^{\prime },y}\right)$ 是 $h$ 上落在 ${R}_{k - 2}$ 中 $h$ 的左端点覆盖矩形内的最右点。由于 $\left( {{r}_{1},\ldots ,{r}_{k - 2},h,v}\right)  \in  {\mathcal{J}}_{2}$，我们知道 ${r}_{k - 2}$ 是 $h$ 的左端点覆盖矩形，并且 ${r}_{k - 2}$ 覆盖 $p$（它是 $h$ 上的一个点）。因此，$p$ 的 $\mathrm{x}$ 坐标不能超过 ${x}_{2}^{\prime }$，从而我们可以断言 $p$ 在 ${h}^{\prime }$ 上。

Proposition C.2. For each tuple $t \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right)$ (where ${H}^{\prime }$ is defined in (12)),the set ${\operatorname{contain}}_{{R}_{k - 2}}\left( t\right)$ is non-empty.

命题 C.2。对于每个元组 $t \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right)$（其中 ${H}^{\prime }$ 在 (12) 中定义），集合 ${\operatorname{contain}}_{{R}_{k - 2}}\left( t\right)$ 非空。

Proof. Set ${h}^{\prime } = t\left\lbrack  {k - 2}\right\rbrack$ . By the definition of a trimmed segment,there exists a rectangle ${r}_{k - 2} \in  {R}_{k - 2}$ covering both endpoints of ${h}^{\prime }$ . Thus,the effective horizontal segment of $\mathbf{t}$ (defined in Section 5) - which must be contained in ${h}^{\prime } -$ must also be covered by ${r}_{k - 2}$ . Hence,contain ${R}_{k - 2}\left( \mathbf{t}\right)$ contains ${r}_{k - 2}$ and thus cannot be empty.

证明。设 ${h}^{\prime } = t\left\lbrack  {k - 2}\right\rbrack$ 。根据修剪线段（trimmed segment）的定义，存在一个矩形 ${r}_{k - 2} \in  {R}_{k - 2}$ 覆盖 ${h}^{\prime }$ 的两个端点。因此，$\mathbf{t}$ 的有效水平线段（在第 5 节中定义）——它必定包含在 ${h}^{\prime } -$ 中——也必定被 ${r}_{k - 2}$ 覆盖。因此，包含 ${R}_{k - 2}\left( \mathbf{t}\right)$ 包含 ${r}_{k - 2}$ ，从而不可能为空。

We now proceed to elaborate the proofs postponed from Section 5. The order of the proofs here will not strictly follow the sequence in which they are referenced in Section 5. In particular, we will prove Lemma 5.2 before Lemma 5.1, because the claims of the former lemma can be used to construct a simple argument for the latter.

现在我们来详细阐述第 5 节中推迟的证明。这里证明的顺序不会严格遵循它们在第 5 节中被引用的顺序。特别地，我们将在引理 5.1 之前证明引理 5.2，因为前者引理的命题可用于为后者构造一个简单的论证。

Proof of Lemma 5.2. We will prove each statement in turn.

引理 5.2 的证明。我们将依次证明每个命题。

Proof of Statement (1). Take any tuple $t = \left( {{r}_{1},\ldots ,{r}_{k - 3},{h}^{\prime },v}\right)  \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right)$ . The point ${h}^{\prime } \cap  v$ is covered by all of ${r}_{1},\ldots ,{r}_{k - 3}$ . Let $h$ be the full segment of ${h}^{\prime }$ . As $h$ contains ${h}^{\prime }$ ,we have $h \cap  v = {h}^{\prime } \cap  v$ .

命题 (1) 的证明。任取一个元组 $t = \left( {{r}_{1},\ldots ,{r}_{k - 3},{h}^{\prime },v}\right)  \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right)$ 。点 ${h}^{\prime } \cap  v$ 被所有的 ${r}_{1},\ldots ,{r}_{k - 3}$ 覆盖。设 $h$ 为 ${h}^{\prime }$ 的完整线段。由于 $h$ 包含 ${h}^{\prime }$ ，我们有 $h \cap  v = {h}^{\prime } \cap  v$ 。

Consider any rectangle ${r}_{k - 2} \in  {\operatorname{contain}}_{{R}_{k - 2}}\left( \mathbf{t}\right)$ . From the definition of ${\operatorname{contain}}_{{R}_{k - 2}}\left( \mathbf{t}\right)$ ,we know that ${r}_{k - 2}$ covers the effective horizontal segment of $t$ whose right endpoint is ${h}^{\prime } \cap  v$ . This means that $h \cap  v$ falls in ${r}_{k - 2}$ . We now have $h \cap  v \cap  \mathop{\bigcap }\limits_{{i = 1}}^{{k - 2}}{r}_{i}$ equals ${h}^{\prime } \cap  v$ and hence is non-empty,meaning that $\left( {{r}_{1},\ldots ,{r}_{k - 3},{r}_{k - 2},h,v}\right)  \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 2},H,V}\right)$ .

考虑任意矩形 ${r}_{k - 2} \in  {\operatorname{contain}}_{{R}_{k - 2}}\left( \mathbf{t}\right)$ 。根据 ${\operatorname{contain}}_{{R}_{k - 2}}\left( \mathbf{t}\right)$ 的定义，我们知道 ${r}_{k - 2}$ 覆盖 $t$ 的有效水平线段，其右端点为 ${h}^{\prime } \cap  v$ 。这意味着 $h \cap  v$ 落在 ${r}_{k - 2}$ 内。现在我们有 $h \cap  v \cap  \mathop{\bigcap }\limits_{{i = 1}}^{{k - 2}}{r}_{i}$ 等于 ${h}^{\prime } \cap  v$ ，因此非空，这意味着 $\left( {{r}_{1},\ldots ,{r}_{k - 3},{r}_{k - 2},h,v}\right)  \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 2},H,V}\right)$ 。

To show $\left( {{r}_{1},\ldots ,{r}_{k - 3},{r}_{k - 2},h,v}\right)  \in  {\mathcal{J}}_{2}$ ,we still need to explain why ${r}_{k - 2}$ covers the left endpoint of $h$ . This follows immediately from the fact that ${r}_{k - 2}$ covers the effective horizontal segment of ${h}^{\prime }$ (which shares the same left endpoint as $h$ ).

为了证明 $\left( {{r}_{1},\ldots ,{r}_{k - 3},{r}_{k - 2},h,v}\right)  \in  {\mathcal{J}}_{2}$ ，我们仍然需要解释为什么 ${r}_{k - 2}$ 覆盖 $h$ 的左端点。这直接源于 ${r}_{k - 2}$ 覆盖 ${h}^{\prime }$ 的有效水平线段（它与 $h$ 具有相同的左端点）这一事实。

Proof of Statement (2). Take any $k$ -tuple $\left( {{r}_{1},\ldots ,{r}_{k - 2},h,v}\right)  \in  {\mathcal{J}}_{2}$ . Set $p = h \cap  v$ ,and define ${h}^{\prime }$ as the trimmed segment of $h$ . We first argue that the(k - 1)-tuple $t = \left( {{r}_{1},\ldots ,{r}_{k - 3},{h}^{\prime },v}\right)$ belongs to $\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right)$ ,or equivalently,the point ${h}^{\prime } \cap  v$ falls in all of ${r}_{1},\ldots ,{r}_{k - 3}$ . By Proposition C.1, ${h}^{\prime } \cap  v$ is the same point as $p$ . From $\left( {{r}_{1},\ldots ,{r}_{k - 2},h,v}\right)  \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 2},H,V}\right)$ ,we know that $p = h \cap  v$ falls in all of ${r}_{1},\ldots ,{r}_{k - 2}$ . It thus follows that ${r}_{1},\ldots ,{r}_{k - 3}$ all cover ${h}^{\prime } \cap  v$ .

命题 (2) 的证明。任取一个 $k$ -元组 $\left( {{r}_{1},\ldots ,{r}_{k - 2},h,v}\right)  \in  {\mathcal{J}}_{2}$ 。设 $p = h \cap  v$ ，并将 ${h}^{\prime }$ 定义为 $h$ 的修剪段。我们首先证明 (k - 1) -元组 $t = \left( {{r}_{1},\ldots ,{r}_{k - 3},{h}^{\prime },v}\right)$ 属于 $\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right)$ ，或者等价地，点 ${h}^{\prime } \cap  v$ 落在所有的 ${r}_{1},\ldots ,{r}_{k - 3}$ 中。根据命题 C.1，${h}^{\prime } \cap  v$ 与 $p$ 是同一点。由 $\left( {{r}_{1},\ldots ,{r}_{k - 2},h,v}\right)  \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 2},H,V}\right)$ 可知，$p = h \cap  v$ 落在所有的 ${r}_{1},\ldots ,{r}_{k - 2}$ 中。因此，${r}_{1},\ldots ,{r}_{k - 3}$ 都覆盖 ${h}^{\prime } \cap  v$ 。

Next,we argue that ${r}_{k - 2} \in  {\operatorname{contain}}_{{R}_{k - 2}}\left( t\right)$ . Let ${p}^{\prime }$ be the left endpoint of $h$ . Because ${h}^{\prime } \cap  v = h \cap  v = p$ , the effective horizontal segment of $t$ is the segment ${p}^{\prime }p$ . To prove ${r}_{k - 2} \in  {\operatorname{contain}}_{{R}_{k - 2}}\left( t\right)$ ,we need to explain why ${r}_{k - 2}$ covers segment ${p}^{\prime }p$ . This is true because:

接下来，我们证明 ${r}_{k - 2} \in  {\operatorname{contain}}_{{R}_{k - 2}}\left( t\right)$ 。设 ${p}^{\prime }$ 为 $h$ 的左端点。因为 ${h}^{\prime } \cap  v = h \cap  v = p$ ，$t$ 的有效水平段是线段 ${p}^{\prime }p$ 。为了证明 ${r}_{k - 2} \in  {\operatorname{contain}}_{{R}_{k - 2}}\left( t\right)$ ，我们需要解释为什么 ${r}_{k - 2}$ 覆盖线段 ${p}^{\prime }p$ 。这是因为：

- ${r}_{k - 2}$ covers $p$ ,as we already know;

- 我们已经知道，${r}_{k - 2}$ 覆盖 $p$ ；

- ${r}_{k - 2}$ covers ${p}^{\prime }$ ,as it is a left-end covering rectangle of $h$ .

- 由于 ${r}_{k - 2}$ 是 $h$ 的左端点覆盖矩形，所以它覆盖 ${p}^{\prime }$ 。

Proof of Statement (3). Our proof resorts to the algorithm generate- ${\mathcal{J}}_{2}$ (see Section 5). This algorithm adds to ${\mathcal{J}}_{2}$ exactly

命题 (3) 的证明。我们的证明借助了生成 - ${\mathcal{J}}_{2}$ 算法（见第 5 节）。该算法恰好向 ${\mathcal{J}}_{2}$ 中添加

$$
\mathop{\sum }\limits_{{\mathbf{t} \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{3},{H}^{\prime },V}\right) }}\left| {{\operatorname{contain}}_{{R}_{k - 2}}\left( \mathbf{t}\right) }\right| 
$$

tuples. By statement (2),the ${\mathcal{J}}_{2}$ produced must be a subset of $\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 2},H,V}\right)$ ,whose size is OUT. Therefore, the above expression is at most OUT, as claimed in statement (3).

元组。根据命题 (2)，所生成的 ${\mathcal{J}}_{2}$ 必定是 $\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 2},H,V}\right)$ 的一个子集，其大小为 OUT。因此，上述表达式至多为 OUT，正如命题 (3) 所声称的那样。

Proof of Lemma 5.1. Again,we resort to the algorithm generate- ${\mathcal{J}}_{2}$ . For each tuple $t \in  \mathcal{J}\left( {{R}_{1},\ldots }\right.$ , ${R}_{k - 3},{H}^{\prime },V)$ ,we know by Proposition C. 2 that ${\operatorname{contain}}_{{R}_{k - 2}}\left( t\right)  \neq  \varnothing$ ; thus,the algorithm adds at least one tuple to ${\mathcal{J}}_{2}$ . Therefore, $\left| {\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right) }\right|  \leq  \left| {\mathcal{J}}_{2}\right|  \leq$ OUT,where the derivation used the definition of ${\mathcal{J}}_{2}$ ; see (11).

引理 5.1 的证明。同样，我们借助生成 - ${\mathcal{J}}_{2}$ 算法。对于每个元组 $t \in  \mathcal{J}\left( {{R}_{1},\ldots }\right.$ ，${R}_{k - 3},{H}^{\prime },V)$ ，根据命题 C.2 我们知道 ${\operatorname{contain}}_{{R}_{k - 2}}\left( t\right)  \neq  \varnothing$ ；因此，该算法至少向 ${\mathcal{J}}_{2}$ 中添加一个元组。所以，$\left| {\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right) }\right|  \leq  \left| {\mathcal{J}}_{2}\right|  \leq$ OUT，这里的推导使用了 ${\mathcal{J}}_{2}$ 的定义；见 (11)。

Computing ${H}^{ * }$ . We assume,w.l.o.g.,that each segment in the input $H$ is given a distinct integer ID in $\left\lbrack  \left| H\right| \right\rbrack$ . This allows us to create an array of size $\left| H\right|  < n$ and allocate an array cell to each $h \in  H$ . The cell can be accessed by the ID of $h$ in constant time.

计算 ${H}^{ * }$。不失一般性，我们假设输入 $H$ 中的每个线段都被赋予了一个在 $\left\lbrack  \left| H\right| \right\rbrack$ 范围内的不同整数 ID。这使我们能够创建一个大小为 $\left| H\right|  < n$ 的数组，并为每个 $h \in  H$ 分配一个数组单元格。可以通过 $h$ 的 ID 在常量时间内访问该单元格。

To compute ${H}^{ * }$ ,we start by deriving minleft $\left( {h}^{\prime }\right)$ for each segment ${h}^{\prime }$ in ${H}^{\prime }$ . For this purpose,first initialize minleft $\left( {h}^{\prime }\right)  = \infty$ for each such ${h}^{\prime }$ . Recall that ${h}^{\prime }$ is the trimmed segment of some segment $h$ in $H$ . We store minleft $\left( {h}^{\prime }\right)$ in the array cell allocated to $h$ . Then,we scan $\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right)$ . For each tuple $t$ therein,update in constant time minleft $\left( {h}^{\prime }\right)$ to the minimum between its current value and the $x$ -coordinate of $t\left\lbrack  {k - 1}\right\rbrack$ . The scan requires $O\left( {n + k \cdot  \mathrm{{OUT}}}\right)$ time.

为了计算 ${H}^{ * }$，我们首先为 ${H}^{\prime }$ 中的每个线段 ${h}^{\prime }$ 推导 minleft $\left( {h}^{\prime }\right)$。为此，首先为每个这样的 ${h}^{\prime }$ 初始化 minleft $\left( {h}^{\prime }\right)  = \infty$。回想一下，${h}^{\prime }$ 是 $H$ 中某个线段 $h$ 的修剪后的线段。我们将 minleft $\left( {h}^{\prime }\right)$ 存储在分配给 $h$ 的数组单元格中。然后，我们扫描 $\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right)$。对于其中的每个元组 $t$，在常量时间内将 minleft $\left( {h}^{\prime }\right)$ 更新为其当前值和 $t\left\lbrack  {k - 1}\right\rbrack$ 的 $x$ 坐标之间的最小值。扫描需要 $O\left( {n + k \cdot  \mathrm{{OUT}}}\right)$ 时间。

Finally,we construct ${H}^{ * }$ by collecting the minimal segment (defined in (15)) of every segment ${h}^{\prime } \in  {H}^{\prime }$ with $\operatorname{minleft}\left( {h}^{\prime }\right)  \neq  \infty$ . This step takes $O\left( \left| {H}^{\prime }\right| \right)  = O\left( n\right)$ time.

最后，我们通过收集每个满足 $\operatorname{minleft}\left( {h}^{\prime }\right)  \neq  \infty$ 的线段 ${h}^{\prime } \in  {H}^{\prime }$ 的最小线段（在 (15) 中定义）来构造 ${H}^{ * }$。这一步需要 $O\left( \left| {H}^{\prime }\right| \right)  = O\left( n\right)$ 时间。

Proof of Lemma 5.3. We will prove each statement in turn.

引理 5.3 的证明。我们将依次证明每个陈述。

Proof of Statement (1). We will map each segment ${h}^{ * } \in  {H}^{ * }$ to a unique tuple $t \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right)$ satisfying ${\operatorname{contain}}_{{R}_{k - 2}}\left( {h}^{ * }\right)  = {\operatorname{contain}}_{{R}_{k - 2}}\left( t\right)$ . The mapping allows us to derive

陈述 (1) 的证明。我们将把每个线段 ${h}^{ * } \in  {H}^{ * }$ 映射到一个满足 ${\operatorname{contain}}_{{R}_{k - 2}}\left( {h}^{ * }\right)  = {\operatorname{contain}}_{{R}_{k - 2}}\left( t\right)$ 的唯一元组 $t \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right)$。该映射使我们能够推导

$$
\mathop{\sum }\limits_{{{h}^{ * } \in  {H}^{ * }}}\left| {{\operatorname{contain}}_{{R}_{k - 2}}\left( {h}^{ * }\right) }\right|  \leq  \mathop{\sum }\limits_{{\mathbf{t} \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right) }}\left| {{\operatorname{contain}}_{{R}_{k - 2}}\left( \mathbf{t}\right) }\right|  \leq  \mathrm{{OUT}}
$$

where the last inequality used Statement (3) of Lemma 5.2.

其中最后一个不等式使用了引理 5.2 的陈述 (3)。

The mapping is as follows. Consider any ${h}^{ * } \in  {H}^{ * }$ . Recall that ${h}^{ * }$ is the minimal segment of some segment ${h}^{\prime } \in  {H}^{\prime }$ . Specifically,if ${h}^{\prime } = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  y$ ,then ${h}^{ * } = \left\lbrack  {{x}_{1},\operatorname{minleft}\left( {h}^{\prime }\right) }\right\rbrack   \times  y$ . By the definition of minleft $\left( {h}^{\prime }\right)$ in (15),there exists a tuple $t \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right)$ satisfying $t\left\lbrack  {k - 2}\right\rbrack   = {h}^{\prime }$ and $\operatorname{minleft}\left( {h}^{\prime }\right)  = x$ -coordinate of the vertical segment $\mathbf{t}\left\lbrack  {k - 1}\right\rbrack$ . We map ${h}^{ * }$ to $\mathbf{t}$ .

映射如下。考虑任意 ${h}^{ * } \in  {H}^{ * }$ 。回顾 ${h}^{ * }$ 是某个线段 ${h}^{\prime } \in  {H}^{\prime }$ 的最小线段。具体而言，如果 ${h}^{\prime } = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  y$ ，那么 ${h}^{ * } = \left\lbrack  {{x}_{1},\operatorname{minleft}\left( {h}^{\prime }\right) }\right\rbrack   \times  y$ 。根据 (15) 中 minleft $\left( {h}^{\prime }\right)$ 的定义，存在一个元组 $t \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right)$ 满足 $t\left\lbrack  {k - 2}\right\rbrack   = {h}^{\prime }$ 以及垂直线段 $\mathbf{t}\left\lbrack  {k - 1}\right\rbrack$ 的 $\operatorname{minleft}\left( {h}^{\prime }\right)  = x$ 坐标。我们将 ${h}^{ * }$ 映射到 $\mathbf{t}$ 。

Next, we will prove

接下来，我们将证明

Claim 1: a rectangle $r \in  {\operatorname{contain}}_{{R}_{k - 2}}\left( {h}^{ * }\right)$ if and only if $r \in  {\operatorname{contain}}_{{R}_{k - 2}}\left( t\right)$ .

命题 1：一个矩形 $r \in  {\operatorname{contain}}_{{R}_{k - 2}}\left( {h}^{ * }\right)$ 当且仅当 $r \in  {\operatorname{contain}}_{{R}_{k - 2}}\left( t\right)$ 。

According to the definitions of ${\operatorname{contain}}_{{R}_{k - 2}}\left( {h}^{ * }\right)$ and ${\operatorname{contain}}_{{R}_{k - 2}}\left( \mathbf{t}\right)  -$ see (3) and (13),respectively - it suffices to show that ${h}^{ * }$ is the same as the effective horizontal segment of $t$ .

根据 ${\operatorname{contain}}_{{R}_{k - 2}}\left( {h}^{ * }\right)$ 和 ${\operatorname{contain}}_{{R}_{k - 2}}\left( \mathbf{t}\right)  -$ 的定义（分别见 (3) 和 (13)），只需证明 ${h}^{ * }$ 与 $t$ 的有效水平线段相同即可。

Denote by $x$ the $\mathrm{x}$ -coordinate of $t\left\lbrack  {k - 1}\right\rbrack$ . Then,the point ${h}^{\prime } \cap  v$ can be written as(x,y), where as mentioned $y$ is the $y$ -coordinate of ${h}^{\prime }$ . The effective horizontal segment of $t$ is therefore $\left\lbrack  {{x}_{1},x}\right\rbrack   \times  y$ . Recall that ${h}^{ * } = \left\lbrack  {{x}_{1},\operatorname{minleft}\left( {h}^{\prime }\right) }\right\rbrack   \times  y$ ,where (by definition of $t$ ) minleft $\left( {h}^{\prime }\right)  =$ $x$ -coordinate of $\mathbf{t}\left\lbrack  {k - 1}\right\rbrack   = x$ . Therefore, ${h}^{ * }$ is the same as the effective horizontal segment of $\mathbf{t}$ , proving claim 1 .

用 $x$ 表示 $t\left\lbrack  {k - 1}\right\rbrack$ 的 $\mathrm{x}$ 坐标。那么，点 ${h}^{\prime } \cap  v$ 可以写成 (x, y) 的形式，其中如前所述，$y$ 是 ${h}^{\prime }$ 的 $y$ 坐标。因此，$t$ 的有效水平线段是 $\left\lbrack  {{x}_{1},x}\right\rbrack   \times  y$ 。回顾 ${h}^{ * } = \left\lbrack  {{x}_{1},\operatorname{minleft}\left( {h}^{\prime }\right) }\right\rbrack   \times  y$ ，其中（根据 $t$ 的定义）minleft $\left( {h}^{\prime }\right)  =$ 是 $\mathbf{t}\left\lbrack  {k - 1}\right\rbrack   = x$ 的 $x$ 坐标。因此，${h}^{ * }$ 与 $\mathbf{t}$ 的有效水平线段相同，从而证明了命题 1。

It remains to show that no two distinct ${h}_{1}^{ * },{h}_{2}^{ * } \in  {H}^{ * }$ can be mapped to the same tuple in $\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right)$ . Assume,on the contrary,that ${h}_{1}^{ * }$ and ${h}_{2}^{ * }$ are both mapped to $t \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3}}\right.$ , ${H}^{\prime },V)$ . Suppose that ${h}_{1}^{ * }$ (resp., ${h}_{2}^{ * }$ ) is the minimal segment of ${h}_{1}^{\prime }$ (resp., ${h}_{2}^{\prime }$ ). Under our mapping,it must hold that ${h}_{1}^{\prime } = {h}_{2}^{\prime } = t\left\lbrack  {k - 2}\right\rbrack$ . However,the distinctness of ${h}_{1}^{ * }$ and ${h}_{2}^{ * }$ requires ${h}_{1}^{\prime } \neq  {h}_{2}^{\prime }$ ,which yields a contradiction.

接下来需要证明，没有两个不同的${h}_{1}^{ * },{h}_{2}^{ * } \in  {H}^{ * }$会被映射到$\mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3},{H}^{\prime },V}\right)$中的同一个元组。相反，假设${h}_{1}^{ * }$和${h}_{2}^{ * }$都被映射到$t \in  \mathcal{J}\left( {{R}_{1},\ldots ,{R}_{k - 3}}\right.$，${H}^{\prime },V)$。假设${h}_{1}^{ * }$（分别地，${h}_{2}^{ * }$）是${h}_{1}^{\prime }$（分别地，${h}_{2}^{\prime }$）的最小线段。在我们的映射下，必然有${h}_{1}^{\prime } = {h}_{2}^{\prime } = t\left\lbrack  {k - 2}\right\rbrack$。然而，${h}_{1}^{ * }$和${h}_{2}^{ * }$的不同要求${h}_{1}^{\prime } \neq  {h}_{2}^{\prime }$，这就产生了矛盾。

Proof of Statement (2). Take any tuple $t \in  \mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 3}^{\prime },{H}^{\prime },V}\right)$ . Set ${h}^{\prime } = t\left\lbrack  {k - 2}\right\rbrack$ ,and define ${h}^{ * }$ as the minimal segment of ${h}^{\prime }$ . Let us write ${h}^{\prime }$ as $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  y$ ; accordingly, ${h}^{ * }$ can be written as $\left\lbrack  {{x}_{1},\operatorname{minleft}\left( {h}^{\prime }\right) }\right\rbrack   \times  y$ . Additionally,let $x$ be the $x$ -coordinate of the vertical segment $t\left\lbrack  {k - 1}\right\rbrack$ . The effective horizontal segment of $t$ can then be represented as $\left\lbrack  {{x}_{1},x}\right\rbrack   \times  y$ .

命题(2)的证明。任取一个元组$t \in  \mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 3}^{\prime },{H}^{\prime },V}\right)$。设${h}^{\prime } = t\left\lbrack  {k - 2}\right\rbrack$，并将${h}^{ * }$定义为${h}^{\prime }$的最小线段。我们将${h}^{\prime }$写为$\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  y$；相应地，${h}^{ * }$可以写为$\left\lbrack  {{x}_{1},\operatorname{minleft}\left( {h}^{\prime }\right) }\right\rbrack   \times  y$。此外，设$x$为垂直线段$t\left\lbrack  {k - 1}\right\rbrack$的$x$坐标。那么$t$的有效水平线段可以表示为$\left\lbrack  {{x}_{1},x}\right\rbrack   \times  y$。

We will first prove ${\operatorname{contain}}_{{R}_{k - 2}}\left( t\right)  \subseteq  {\operatorname{contain}}_{{R}_{k - 2}}\left( {h}^{ * }\right)$ ,or equivalently,every rectangle ${r}_{k - 2} \in$ ${\operatorname{contain}}_{{R}_{k - 2}}\left( t\right)$ covers ${h}^{ * }$ . It suffices to show that ${h}^{ * }$ is contained in the effective horizontal segment of $t$ ,or equivalently,minleft $\left( {h}^{\prime }\right)  \leq  x$ . By the definition in (14),minleft $\left( {h}^{\prime }\right)$ is the minimum $\mathrm{x}$ -coordinate of ${\mathbf{t}}^{\prime }\left\lbrack  {k - 1}\right\rbrack$ among all ${\mathbf{t}}^{\prime } \in  \mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 3}^{\prime },{H}^{\prime },V}\right)$ with ${\mathbf{t}}^{\prime }\left\lbrack  {k - 2}\right\rbrack   = {h}^{\prime }$ ,Thus,minleft $\left( {h}^{\prime }\right)  \leq  x$ holds because $t$ is merely one such ${t}^{\prime }$ .

我们首先证明${\operatorname{contain}}_{{R}_{k - 2}}\left( t\right)  \subseteq  {\operatorname{contain}}_{{R}_{k - 2}}\left( {h}^{ * }\right)$，或者等价地，每个矩形${r}_{k - 2} \in$ ${\operatorname{contain}}_{{R}_{k - 2}}\left( t\right)$覆盖${h}^{ * }$。只需证明${h}^{ * }$包含在$t$的有效水平线段中，或者等价地，minleft $\left( {h}^{\prime }\right)  \leq  x$。根据(14)中的定义，minleft $\left( {h}^{\prime }\right)$是所有满足${\mathbf{t}}^{\prime }\left\lbrack  {k - 2}\right\rbrack   = {h}^{\prime }$的${\mathbf{t}}^{\prime } \in  \mathcal{J}\left( {{R}_{1}^{\prime },\ldots ,{R}_{k - 3}^{\prime },{H}^{\prime },V}\right)$中${\mathbf{t}}^{\prime }\left\lbrack  {k - 1}\right\rbrack$的最小$\mathrm{x}$坐标。因此，minleft $\left( {h}^{\prime }\right)  \leq  x$成立，因为$t$只是这样的一个${t}^{\prime }$。

Next,assuming that the rectangles $r \in  {\operatorname{contain}}_{{R}_{k - 2}}\left( {h}^{ * }\right)$ have been sorted in descending order of right(r),we will prove that ${\operatorname{contain}}_{{R}_{k - 2}}\left( t\right)$ includes a prefix of the sorted order. It suffices to establish the following equivalent claim:

接下来，假设矩形 $r \in  {\operatorname{contain}}_{{R}_{k - 2}}\left( {h}^{ * }\right)$ 已按右边界 $r$ 降序排序，我们将证明 ${\operatorname{contain}}_{{R}_{k - 2}}\left( t\right)$ 包含排序顺序的一个前缀。只需证明以下等价命题即可：

Claim 2: If a rectangle $r \in  {\operatorname{contain}}_{{R}_{k - 2}}\left( {h}^{ * }\right)$ satisfies the condition that $\operatorname{right}\left( r\right)  \geq  x$ (where $x$ is the $\mathrm{x}$ -coordinate of $\mathbf{t}\left\lbrack  {k - 1}\right\rbrack$ ),then $r \in  {\operatorname{contain}}_{{R}_{k - 2}}\left( \mathbf{t}\right)$ .

命题 2：如果矩形 $r \in  {\operatorname{contain}}_{{R}_{k - 2}}\left( {h}^{ * }\right)$ 满足条件 $\operatorname{right}\left( r\right)  \geq  x$（其中 $x$ 是 $\mathbf{t}\left\lbrack  {k - 1}\right\rbrack$ 的 $\mathrm{x}$ 坐标），那么 $r \in  {\operatorname{contain}}_{{R}_{k - 2}}\left( \mathbf{t}\right)$。

To prove the above,we must explain why the rectangle $r$ in the claim contains the effective horizontal segment of $t$ . Since $r \in  {\operatorname{contain}}_{{R}_{k - 2}}\left( {h}^{ * }\right)$ ,we know that $r$ covers the segment $\left\lbrack  {{x}_{1},\operatorname{minleft}\left( {h}^{\prime }\right) }\right\rbrack   \times  y$ . Hence, $\left\lbrack  {{x}_{1},\operatorname{minleft}\left( {h}^{\prime }\right) }\right\rbrack   \subseteq  \left\lbrack  {\operatorname{left}\left( r\right) ,\operatorname{right}\left( r\right) }\right\rbrack$ and $y \in  \left\lbrack  {\operatorname{bot}\left( r\right) ,\operatorname{top}\left( r\right) }\right\rbrack$ . Using also the condition $\operatorname{right}\left( r\right)  \geq  x$ given in claim 2,we can derive $\left\lbrack  {{x}_{1},x}\right\rbrack   \subseteq  \left\lbrack  {\operatorname{left}\left( r\right) ,\operatorname{right}\left( r\right) }\right\rbrack$ . Therefore, $\left\lbrack  {{x}_{1},x}\right\rbrack   \times  y$ ,i.e. the effective horizontal segment of $t$ ,is contained in $r$ ,establishing claim 2 .

为证明上述命题，我们必须解释为什么命题中的矩形 $r$ 包含 $t$ 的有效水平线段。由于 $r \in  {\operatorname{contain}}_{{R}_{k - 2}}\left( {h}^{ * }\right)$，我们知道 $r$ 覆盖线段 $\left\lbrack  {{x}_{1},\operatorname{minleft}\left( {h}^{\prime }\right) }\right\rbrack   \times  y$。因此，$\left\lbrack  {{x}_{1},\operatorname{minleft}\left( {h}^{\prime }\right) }\right\rbrack   \subseteq  \left\lbrack  {\operatorname{left}\left( r\right) ,\operatorname{right}\left( r\right) }\right\rbrack$ 且 $y \in  \left\lbrack  {\operatorname{bot}\left( r\right) ,\operatorname{top}\left( r\right) }\right\rbrack$。再利用命题 2 中给出的条件 $\operatorname{right}\left( r\right)  \geq  x$，我们可以推导出 $\left\lbrack  {{x}_{1},x}\right\rbrack   \subseteq  \left\lbrack  {\operatorname{left}\left( r\right) ,\operatorname{right}\left( r\right) }\right\rbrack$。所以，$\left\lbrack  {{x}_{1},x}\right\rbrack   \times  y$，即 $t$ 的有效水平线段，包含在 $r$ 中，从而证明了命题 2。

## D Proof of Theorem 1.2

## D 定理 1.2 的证明

As mentioned in Section 1.1,2-SJ can be solved in ${F}_{2}\left( {n,\mathrm{{OUT}}}\right)  = O\left( {n\log n + \mathrm{{OUT}}}\right)$ time using a comparison-based algorithm. By Theorem 1.1,in general,a comparison-based(k - 1)-SJ algorithm with runtime ${F}_{k - 1}\left( {n,\mathrm{{OUT}}}\right)$ spawns a comparison-based $k$ -SJ algorithm whose running time ${F}_{k}\left( {n,\mathrm{{OUT}}}\right)$ obeys (1). Specifically,for $k \geq  3$ ,there is a constant $c \geq  2$ such that

如 1.1 节所述，使用基于比较的算法可以在 ${F}_{2}\left( {n,\mathrm{{OUT}}}\right)  = O\left( {n\log n + \mathrm{{OUT}}}\right)$ 时间内解决 2 - SJ 问题。根据定理 1.1，一般来说，运行时间为 ${F}_{k - 1}\left( {n,\mathrm{{OUT}}}\right)$ 的基于比较的 $(k - 1)$ - SJ 算法会衍生出一个基于比较的 $k$ - SJ 算法，其运行时间 ${F}_{k}\left( {n,\mathrm{{OUT}}}\right)$ 满足 (1)。具体来说，对于 $k \geq  3$，存在一个常数 $c \geq  2$ 使得

$$
{F}_{k}\left( {n,\mathrm{{OUT}}}\right) 
$$

$$
 \leq  \;c \cdot  {k}^{3} \cdot  \left( {{F}_{k - 1}\left( {n,\mathrm{{OUT}}}\right)  + n\log n + k \cdot  \mathrm{{OUT}}}\right) 
$$

$$
 = {c}^{2} \cdot  {k}^{3}{\left( k - 1\right) }^{3} \cdot  \left( {{F}_{k - 2}\left( {n,\mathrm{{OUT}}}\right)  + n\log n + k \cdot  \mathrm{{OUT}}}\right)  + c \cdot  {k}^{3} \cdot  \left( {n\log n + k \cdot  \mathrm{{OUT}}}\right) 
$$

$$
 < {c}^{2} \cdot  {k}^{3}{\left( k - 1\right) }^{3} \cdot  {F}_{k - 2}\left( {n,\mathrm{{OUT}}}\right)  + \left( {c + {c}^{2}}\right)  \cdot  {k}^{3}{\left( k - 1\right) }^{3} \cdot  \left( {n\log n + k \cdot  \mathrm{{OUT}}}\right) 
$$

$$
 \leq  {c}^{3} \cdot  {k}^{3}{\left( k - 1\right) }^{3}{\left( k - 2\right) }^{3} \cdot  {F}_{k - 3}\left( {n,\mathrm{{OUT}}}\right)  + \left( {c + {c}^{2} + {c}^{3}}\right)  \cdot  {k}^{3}{\left( k - 1\right) }^{3}{\left( k - 2\right) }^{3} \cdot  \left( {n\log n + k \cdot  \mathrm{{OUT}}}\right) 
$$

$$
 \leq  \ldots  \leq  {c}^{k - 2} \cdot  {\left( k!\right) }^{3} \cdot  {F}_{2}\left( {n,\mathrm{{OUT}}}\right)  + \left( {\mathop{\sum }\limits_{{i = 1}}^{{k - 2}}{c}^{i}}\right)  \cdot  {\left( k!\right) }^{3} \cdot  \left( {n\log n + k \cdot  \mathrm{{OUT}}}\right) 
$$

$$
 \leq  2{c}^{k - 1} \cdot  {\left( k!\right) }^{3} \cdot  O\left( {n\log n + k \cdot  \mathrm{{OUT}}}\right) 
$$

which completes the proof of Theorem 1.2.

这就完成了定理 1.2 的证明。

## E Hardness of 3-SJ in 3D Space

## E 三维空间中 3 - SJ 问题的难度

An axis-parallel rectangle in 3D space has the form $r = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  \left\lbrack  {{y}_{1},{y}_{2}}\right\rbrack   \times  \left\lbrack  {{z}_{1},{z}_{2}}\right\rbrack$ . We will refer to $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$ as the $x$ -projection of $r$ ,and define its $y$ - and $z$ -projections analogously.

三维空间中的轴平行矩形具有 $r = \left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack   \times  \left\lbrack  {{y}_{1},{y}_{2}}\right\rbrack   \times  \left\lbrack  {{z}_{1},{z}_{2}}\right\rbrack$ 的形式。我们将 $\left\lbrack  {{x}_{1},{x}_{2}}\right\rbrack$ 称为 $r$ 的 $x$ 投影，并类似地定义其 $y$ 投影和 $z$ 投影。

In the 3D 3-SJ problem,the input comprises three sets of axis-parallel rectangles in ${\mathbb{R}}^{3} : {R}_{1},{R}_{2}$ , and ${R}_{3}$ . The goal is to report all 3-tuples $\left( {{r}_{1},{r}_{2},{r}_{3}}\right)  \in  {R}_{1} \times  {R}_{2} \times  {R}_{3}$ such that ${r}_{1} \cap  {r}_{2} \cap  {r}_{3} \neq  \varnothing$ . Denote by $\mathcal{J}\left( {{R}_{1},{R}_{2},{R}_{3}}\right)$ the set of those 3-tuples. Define the input size as $n = \left| {R}_{1}\right|  + \left| {R}_{2}\right|  + \left| {R}_{3}\right|$ and the output size as $\mathrm{{OUT}} = \left| {\mathcal{J}\left( {{R}_{1},{R}_{2},{R}_{3}}\right) }\right|$ .

在三维三重矩形相交（3D 3-SJ）问题中，输入包含三组在${\mathbb{R}}^{3} : {R}_{1},{R}_{2}$和${R}_{3}$中与坐标轴平行的矩形。目标是报告所有满足${r}_{1} \cap  {r}_{2} \cap  {r}_{3} \neq  \varnothing$的三元组$\left( {{r}_{1},{r}_{2},{r}_{3}}\right)  \in  {R}_{1} \times  {R}_{2} \times  {R}_{3}$。用$\mathcal{J}\left( {{R}_{1},{R}_{2},{R}_{3}}\right)$表示这些三元组的集合。将输入规模定义为$n = \left| {R}_{1}\right|  + \left| {R}_{2}\right|  + \left| {R}_{3}\right|$，输出规模定义为$\mathrm{{OUT}} = \left| {\mathcal{J}\left( {{R}_{1},{R}_{2},{R}_{3}}\right) }\right|$。

In the triangle detection problem,we are given an undirected graph $G = \left( {V,E}\right)$ and need to determine whether $G$ has a triangle (a.k.a. 3-clique). Set $m = \left| E\right|$ . We consider that $G$ has no isolated vertices (namely,vertices with degree 0 ),and therefore $\left| V\right|  \leq  {2m}$ .

在三角形检测问题中，给定一个无向图$G = \left( {V,E}\right)$，需要确定$G$中是否存在一个三角形（也称为 3 - 团）。设$m = \left| E\right|$。我们假设$G$中没有孤立顶点（即度数为 0 的顶点），因此$\left| V\right|  \leq  {2m}$。

The subsequent discussion will show that if the 3D 3-SJ problem can be solved in $O(\left( {n + \mathrm{{OUT}}}\right)  \cdot$ polylog $n$ ) time,then the triangle detection problem can be solved in $O\left( {m\text{polylog}m}\right)$ time. This would be truly surprising because as mentioned in Section 1.2 the state-of-the-art algorithm for triangle detection runs in $O\left( {m}^{1.41}\right)$ time [2]. Thus,in the absence of such a breakthrough,no $O\left( {\left( {n + \mathrm{{OUT}}}\right)  \cdot  \text{polylog }n}\right)$ time algorithms can exist for the 3D 3-SJ problem.

后续讨论将表明，如果三维三重矩形相交问题可以在$O(\left( {n + \mathrm{{OUT}}}\right)  \cdot$多项式对数$n$）时间内解决，那么三角形检测问题可以在$O\left( {m\text{polylog}m}\right)$时间内解决。这将非常令人惊讶，因为如 1.2 节所述，目前最先进的三角形检测算法的运行时间为$O\left( {m}^{1.41}\right)$ [2]。因此，在没有此类突破的情况下，三维三重矩形相交问题不存在$O\left( {\left( {n + \mathrm{{OUT}}}\right)  \cdot  \text{polylog }n}\right)$时间的算法。

Given a graph $G = \left( {V,E}\right)$ for triangle detection,we will construct an instance of the 3D 3-SJ problem with input size ${3m}$ . W.l.o.g.,let us assume that each vertex of $V$ is represented as a unique integer in $\left\lbrack  \left| V\right| \right\rbrack$ . Initialize ${R}_{1},{R}_{2}$ ,and ${R}_{3}$ as 3 empty sets of rectangles. For each edge $\{ u,v\}  \in  E$ where $u < v$ ,we add

给定一个用于三角形检测的图$G = \left( {V,E}\right)$，我们将构造一个输入规模为${3m}$的三维三重矩形相交问题的实例。不失一般性，假设$V$的每个顶点都表示为$\left\lbrack  \left| V\right| \right\rbrack$中的一个唯一整数。将${R}_{1},{R}_{2}$和${R}_{3}$初始化为 3 个空的矩形集合。对于每条边$\{ u,v\}  \in  E$，其中$u < v$，我们添加

- a rectangle $\left( {-\infty ,\infty }\right)  \times  \left\lbrack  {u,u}\right\rbrack   \times  \left\lbrack  {v,v}\right\rbrack$ to ${R}_{1}$ ;

- 一个矩形$\left( {-\infty ,\infty }\right)  \times  \left\lbrack  {u,u}\right\rbrack   \times  \left\lbrack  {v,v}\right\rbrack$到${R}_{1}$；

- a rectangle $\left\lbrack  {v,v}\right\rbrack   \times  \left( {-\infty ,\infty }\right)  \times  \left\lbrack  {u,u}\right\rbrack$ to ${R}_{2}$ ;

- 一个矩形$\left\lbrack  {v,v}\right\rbrack   \times  \left( {-\infty ,\infty }\right)  \times  \left\lbrack  {u,u}\right\rbrack$到${R}_{2}$；

- a rectangle $\left\lbrack  {v,v}\right\rbrack   \times  \left\lbrack  {u,u}\right\rbrack   \times  \left( {-\infty ,\infty }\right)$ to ${R}_{3}$ .

- 一个矩形$\left\lbrack  {v,v}\right\rbrack   \times  \left\lbrack  {u,u}\right\rbrack   \times  \left( {-\infty ,\infty }\right)$到${R}_{3}$。

Note that every rectangle in ${R}_{1}$ (resp., ${R}_{2}$ and ${R}_{3}$ ) has $\left( {-\infty ,\infty }\right)$ as the x- (resp.,y- and z-) projection.

注意，${R}_{1}$（分别地，${R}_{2}$和${R}_{3}$）中的每个矩形的 x -（分别地，y - 和 z -）投影为$\left( {-\infty ,\infty }\right)$。

The construction has the property that $G$ has a triangle if and only if $\mathcal{J}\left( {{R}_{1},{R}_{2},{R}_{3}}\right)  \neq  \varnothing$ . We can prove this with the following argument.

该构造具有这样的性质：当且仅当$\mathcal{J}\left( {{R}_{1},{R}_{2},{R}_{3}}\right)  \neq  \varnothing$时，$G$中有一个三角形。我们可以用以下论证来证明这一点。

- Suppose that $G$ has a triangle with vertices $u,v$ ,and $w$ such that $u < v < w$ . Then,by our construction, ${r}_{1} = \left( {-\infty ,\infty }\right)  \times  \left\lbrack  {u,u}\right\rbrack   \times  \left\lbrack  {v,v}\right\rbrack   \in  {R}_{1},{r}_{2} = \left\lbrack  {w,w}\right\rbrack   \times  \left( {-\infty ,\infty }\right)  \times  \left\lbrack  {v,v}\right\rbrack   \in  {R}_{2}$ ,and ${r}_{3} = \left\lbrack  {w,w}\right\rbrack   \times  \left\lbrack  {u,u}\right\rbrack   \times  \left( {-\infty ,\infty }\right)  \in  {r}_{3}$ . It is clear that $\left( {{r}_{1},{r}_{2},{r}_{3}}\right)$ is a result tuple in $\mathcal{J}\left( {{R}_{1},{R}_{2},{R}_{3}}\right)$ .

- 假设$G$有一个以$u,v$、$w$为顶点的三角形，使得$u < v < w$。那么，根据我们的构造，${r}_{1} = \left( {-\infty ,\infty }\right)  \times  \left\lbrack  {u,u}\right\rbrack   \times  \left\lbrack  {v,v}\right\rbrack   \in  {R}_{1},{r}_{2} = \left\lbrack  {w,w}\right\rbrack   \times  \left( {-\infty ,\infty }\right)  \times  \left\lbrack  {v,v}\right\rbrack   \in  {R}_{2}$和${r}_{3} = \left\lbrack  {w,w}\right\rbrack   \times  \left\lbrack  {u,u}\right\rbrack   \times  \left( {-\infty ,\infty }\right)  \in  {r}_{3}$。显然，$\left( {{r}_{1},{r}_{2},{r}_{3}}\right)$是$\mathcal{J}\left( {{R}_{1},{R}_{2},{R}_{3}}\right)$中的一个结果元组。

- Conversely,consider that $\mathcal{J}\left( {{R}_{1},{R}_{2},{R}_{3}}\right)$ is non-empty. Consider an arbitrary result tuple $\left( {{r}_{1},{r}_{2},{r}_{3}}\right)  \in  \mathcal{J}\left( {{R}_{1},{R}_{2},{R}_{3}}\right)$ . Assume,w.l.o.g.,that ${r}_{1} = \left( {-\infty ,\infty }\right)  \times  \left\lbrack  {u,u}\right\rbrack   \times  \left\lbrack  {v,v}\right\rbrack$ ,where $u < v$ . Because the z-projection of ${r}_{2}$ must match that of ${r}_{1}$ ,we assert that ${r}_{2}$ must have the form $\left\lbrack  {w,w}\right\rbrack   \times  \left( {-\infty ,\infty }\right)  \times  \left\lbrack  {v,v}\right\rbrack$ for some $w > v$ . Because the $\mathrm{x}$ -projection of ${r}_{3}$ must match that of ${r}_{2}$ and the y-projection of ${r}_{3}$ must match that of ${r}_{1}$ ,it follows that ${r}_{3}$ must have the form $\left\lbrack  {w,w}\right\rbrack   \times  \left\lbrack  {u,u}\right\rbrack   \times  \left( {-\infty ,\infty }\right)$ . This means that the edges $\{ u,v\} ,\{ v,w\}$ ,and $\{ u,w\}$ must all exist in $G$ ,and thus form a triangle.

- 反之，假设$\mathcal{J}\left( {{R}_{1},{R}_{2},{R}_{3}}\right)$非空。考虑任意一个结果元组$\left( {{r}_{1},{r}_{2},{r}_{3}}\right)  \in  \mathcal{J}\left( {{R}_{1},{R}_{2},{R}_{3}}\right)$。不失一般性，假设${r}_{1} = \left( {-\infty ,\infty }\right)  \times  \left\lbrack  {u,u}\right\rbrack   \times  \left\lbrack  {v,v}\right\rbrack$，其中$u < v$。因为${r}_{2}$的z投影必须与${r}_{1}$的z投影匹配，我们断言${r}_{2}$对于某个$w > v$必定具有$\left\lbrack  {w,w}\right\rbrack   \times  \left( {-\infty ,\infty }\right)  \times  \left\lbrack  {v,v}\right\rbrack$的形式。因为${r}_{3}$的$\mathrm{x}$投影必须与${r}_{2}$的$\mathrm{x}$投影匹配，且${r}_{3}$的y投影必须与${r}_{1}$的y投影匹配，所以${r}_{3}$必定具有$\left\lbrack  {w,w}\right\rbrack   \times  \left\lbrack  {u,u}\right\rbrack   \times  \left( {-\infty ,\infty }\right)$的形式。这意味着边$\{ u,v\} ,\{ v,w\}$和$\{ u,w\}$必定都存在于$G$中，从而构成一个三角形。

Now,assume that there exists an algorithm $\mathcal{A}$ capable of solving the 3D 3-SJ problem in $O((n +$ OUT) - polylog $n$ ) time. For OUT $= 0$ ,this algorithm must perform at most $c \cdot  n$ polylog $n$ steps when the input size is $n$ . We run $\mathcal{A}$ on the 3D 3-SJ instance ${R}_{1},{R}_{2},{R}_{3}$ constructed earlier in a cost-monitoring manner:

现在，假设存在一个算法$\mathcal{A}$，它能够在$O((n +$ OUT) - 多项式对数$n$ )时间内解决三维3 - 半连接（3D 3 - SJ）问题。对于OUT $= 0$，当输入规模为$n$时，该算法最多必须执行$c \cdot  n$多项式对数$n$步。我们以成本监控的方式对前面构造的三维3 - 半连接实例${R}_{1},{R}_{2},{R}_{3}$运行$\mathcal{A}$：

- If $\mathcal{A}$ terminates within $c \cdot  \left( {3m}\right)$ polylog(3m)steps,we check whether it has output any result tuple in $\mathcal{J}\left( {{R}_{1},{R}_{2},{R}_{3}}\right)$ . If so,a triangle has been found in $G$ ; otherwise,we declare that $G$ has no triangles.

- 如果$\mathcal{A}$在$c \cdot  \left( {3m}\right)$ polylog(3m)步内终止，我们检查它是否在$\mathcal{J}\left( {{R}_{1},{R}_{2},{R}_{3}}\right)$中输出了任何结果元组。如果是，则在$G$中找到了一个三角形；否则，我们声明$G$中没有三角形。

- If $\mathcal{A}$ has performed $1 + c \cdot  \left( {3m}\right)$ polylog(3m)steps,we manually terminate the algorithm and declare that $\mathcal{J}\left( {{R}_{1},{R}_{2},{R}_{3}}\right)  \neq  \varnothing$ ,meaning that $G$ must have at least one triangle.

- 如果$\mathcal{A}$已经执行了$1 + c \cdot  \left( {3m}\right)$ polylog(3m)步，我们手动终止该算法并声明$\mathcal{J}\left( {{R}_{1},{R}_{2},{R}_{3}}\right)  \neq  \varnothing$，这意味着$G$中一定至少有一个三角形。

The above strategy thus settles the triangle detection problem in $O\left( {m\text{polylog}m}\right)$ time.

因此，上述策略在$O\left( {m\text{polylog}m}\right)$时间内解决了三角形检测问题。

## References

## 参考文献

[1] Pankaj K. Agarwal, Lars Arge, Haim Kaplan, Eyal Molad, Robert Endre Tarjan, and Ke Yi. An optimal dynamic data structure for stabbing-semigroup queries. SIAM Journal of Computing, 41(1):104-127, 2012.

[2] Noga Alon, Raphael Yuster, and Uri Zwick. Finding and counting given length cycles. Algorithmica, 17(3):209-223, 1997.

[3] Lars Arge, Octavian Procopiuc, Sridhar Ramaswamy, Torsten Suel, Jan Vahrenhold, and Jeffrey Scott Vitter. A unified approach for indexed and non-indexed spatial joins. In Proceedings of Extending Database Technology (EDBT), volume 1777 of Lecture Notes in Computer Science, pages 413-429, 2000.

[4] Thomas Brinkhoff, Hans-Peter Kriegel, and Bernhard Seeger. Efficient processing of spatial joins using R-trees. In Proceedings of ACM Management of Data (SIGMOD), pages 237-246, 1993.

[5] Mark de Berg, Otfried Cheong, Marc van Kreveld, and Mark Overmars. Computational Geometry: Algorithms and Applications. Springer-Verlag, 3rd edition, 2008.

[6] David P. Dobkin and Richard J. Lipton. On the complexity of computations under varying sets of primitives. Journal of Computer and System Sciences (JCSS), 18(1):86-91, 1979.

[7] Himanshu Gupta, Bhupesh Chawda, Sumit Negi, Tanveer A. Faruquie, L. Venkata Subramaniam, and Mukesh K. Mohania. Processing multi-way spatial joins on map-reduce. In Proceedings of Extending Database Technology (EDBT), pages 113-124, 2013.

[8] Edwin H. Jacox and Hanan Samet. Spatial join techniques. ACM Transactions on Database Systems (TODS), 32(1):7, 2007.

[9] Mahmoud Abo Khamis, George Chichirim, Antonia Kormpa, and Dan Olteanu. The complexity of boolean conjunctive queries with intersection joins. In Proceedings of ACM Symposium on Principles of Database Systems (PODS), pages 53-65, 2022.

[10] Nick Koudas and Kenneth C. Sevcik. Size separation spatial join. In Proceedings of ACM Management of Data (SIGMOD), pages 324-335, 1997.

[11] Ming-Ling Lo and Chinya V. Ravishankar. Spatial joins using seeded trees. In Proceedings of ACM Management of Data (SIGMOD), pages 209-220, 1994.

[12] Ming-Ling Lo and Chinya V. Ravishankar. Spatial hash-joins. In Proceedings of ACM Management of Data (SIGMOD), pages 247-258, 1996.

[13] Nikos Mamoulis and Dimitris Papadias. Constraint-based algorithms for computing clique intersection joins. In Robert Laurini, Kia Makki, and Niki Pissinou, editors, Proceedings of ACM Symposium on Advances in Geographic Information Systems (GIS), pages 118-123, 1998.

[14] Nikos Mamoulis and Dimitris Papadias. Multiway spatial joins. ACM Transactions on Database Systems (TODS), 26(4):424-475, 2001.

[15] Nikos Mamoulis and Dimitris Papadias. Slot index spatial join. IEEE Transactions on Knowledge and Data Engineering (TKDE), 15(1):211-231, 2003.

[16] Edward M. McCreight. Priority search trees. SIAM Journal of Computing, 14(2):257-276, 1985.

[17] Eunjin Oh and Hee-Kap Ahn. Finding pairwise intersections of rectangles in a query rectangle. Comput. Geom., 85, 2019.

[18] Dimitris Papadias, Nikos Mamoulis, and Yannis Theodoridis. Processing and optimization of multiway spatial joins using R-trees. In Proceedings of ACM Symposium on Principles of Database Systems (PODS), pages 44-55, 1999.

[19] Jignesh M. Patel and David J. DeWitt. Partition based spatial-merge join. In Proceedings of ACM Management of Data (SIGMOD), pages 259-270, 1996.

[20] Saladi Rahul, Ananda Swarup Das, Krishnan Sundara Rajan, and Kannan Srinathan. Range-aggregate queries involving geometric aggregation operations. In Proceedings of International Conference on WALCOM: Algorithms and Computation, volume 6552, pages 122-133, 2011.

[21] Yufei Tao and Ke Yi. Intersection joins under updates. Journal of Computer and System Sciences (JCSS), 124:41-64, 2022.