# A Density-Based Algorithm for Discovering Clusters A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise

# 一种基于密度的带噪声大空间数据库聚类发现算法

Martin Ester, Hans-Peter Kriegel, Jörg Sander, Xiaowei Xu

马丁·埃斯特（Martin Ester）、汉斯 - 彼得·克里格尔（Hans - Peter Kriegel）、约尔格·桑德（Jörg Sander）、徐晓伟（Xiaowei Xu）

Institute for Computer Science, University of Munich

慕尼黑大学计算机科学研究所

Oettingenstr. 67, D-80538 München, Germany

德国慕尼黑奥廷根街67号，邮编：D - 80538

\{ester | kriegel | sander | xwxu\}@informatik.uni-muenchen.de

\{ester | kriegel | sander | xwxu\}@informatik.uni - muenchen.de

## Abstract

## 摘要

Clustering algorithms are attractive for the task of class identification in spatial databases. However, the application to large spatial databases rises the following requirements for clustering algorithms: minimal requirements of domain knowledge to determine the input parameters, discovery of clusters with arbitrary shape and good efficiency on large databases. The well-known clustering algorithms offer no solution to the combination of these requirements. In this paper, we present the new clustering algorithm DBSCAN relying on a density-based notion of clusters which is designed to discover clusters of arbitrary shape. DBSCAN requires only one input parameter and supports the user in determining an appropriate value for it. We performed an experimental evaluation of the effectiveness and efficiency of DBSCAN using synthetic data and real data of the SEQUOIA 2000 benchmark. The results of our experiments demonstrate that (1) DBSCAN is significantly more effective in discovering clusters of arbitrary shape than the well-known algorithm CLAR-ANS, and that (2) DBSCAN outperforms CLARANS by a factor of more than 100 in terms of efficiency.

聚类算法在空间数据库的类别识别任务中具有吸引力。然而，将其应用于大型空间数据库对聚类算法提出了以下要求：确定输入参数所需的领域知识最少；能够发现任意形状的聚类；在大型数据库上具有良好的效率。著名的聚类算法无法同时满足这些要求。在本文中，我们提出了新的聚类算法DBSCAN，它基于一种基于密度的聚类概念，旨在发现任意形状的聚类。DBSCAN只需要一个输入参数，并帮助用户确定该参数的合适值。我们使用合成数据和红杉2000基准测试的真实数据对DBSCAN的有效性和效率进行了实验评估。实验结果表明：（1）与著名的CLAR - ANS算法相比，DBSCAN在发现任意形状的聚类方面明显更有效；（2）在效率方面，DBSCAN比CLARANS高出100多倍。

Keywords: Clustering Algorithms, Arbitrary Shape of Clusters, Efficiency on Large Spatial Databases, Handling Nlj4- 275oise.

关键词：聚类算法；任意形状的聚类；大型空间数据库的效率；处理噪声。

## 1. Introduction

## 1. 引言

Numerous applications require the management of spatial data, i.e. data related to space. Spatial Database Systems (SDBS) (Gueting 1994) are database systems for the management of spatial data. Increasingly large amounts of data are obtained from satellite images, X-ray crystallography or other automatic equipment. Therefore, automated knowledge discovery becomes more and more important in spatial databases.

许多应用需要管理空间数据，即与空间相关的数据。空间数据库系统（SDBS）（格廷，1994年）是用于管理空间数据的数据库系统。从卫星图像、X射线晶体学或其他自动设备中获取的数据量越来越大。因此，在空间数据库中，自动知识发现变得越来越重要。

Several tasks of knowledge discovery in databases (KDD) have been defined in the literature (Matheus, Chan & Pi-atetsky-Shapiro 1993). The task considered in this paper is class identification, i.e. the grouping of the objects of a database into meaningful subclasses. In an earth observation database, e.g., we might want to discover classes of houses along some river.

文献中已经定义了数据库知识发现（KDD）的几个任务（马西厄斯、陈和皮亚捷茨基 - 夏皮罗，1993年）。本文考虑的任务是类别识别，即把数据库中的对象分组为有意义的子类。例如，在一个地球观测数据库中，我们可能想发现某条河流沿岸的房屋类别。

Clustering algorithms are attractive for the task of class identification. However, the application to large spatial databases rises the following requirements for clustering algorithms:

聚类算法在类别识别任务中具有吸引力。然而，将其应用于大型空间数据库对聚类算法提出了以下要求：

(1) Minimal requirements of domain knowledge to determine the input parameters, because appropriate values are often not known in advance when dealing with large databases.

（1）确定输入参数所需的领域知识最少，因为在处理大型数据库时，合适的值通常事先并不知晓。

(2) Discovery of clusters with arbitrary shape, because the shape of clusters in spatial databases may be spherical, drawn-out, linear, elongated etc.

（2）能够发现任意形状的聚类，因为空间数据库中聚类的形状可能是球形、细长形、线性、狭长形等。

(3) Good efficiency on large databases, i.e. on databases of significantly more than just a few thousand objects.

（3）在大型数据库上具有良好的效率，即处理对象数量远超过几千个的数据库。

The well-known clustering algorithms offer no solution to the combination of these requirements. In this paper, we present the new clustering algorithm DBSCAN. It requires only one input parameter and supports the user in determining an appropriate value for it. It discovers clusters of arbitrary shape. Finally, DBSCAN is efficient even for large spatial databases. The rest of the paper is organized as follows. We discuss clustering algorithms in section 2 evaluating them according to the above requirements. In section 3, we present our notion of clusters which is based on the concept of density in the database. Section 4 introduces the algorithm DBSCAN which discovers such clusters in a spatial database. In section 5, we performed an experimental evaluation of the effectiveness and efficiency of DBSCAN using synthetic data and data of the SEQUOIA 2000 benchmark. Section 6 concludes with a summary and some directions for future research.

著名的聚类算法无法同时满足这些要求。在本文中，我们提出了新的聚类算法DBSCAN。它只需要一个输入参数，并帮助用户确定该参数的合适值。它能够发现任意形状的聚类。最后，即使对于大型空间数据库，DBSCAN也具有较高的效率。本文的其余部分组织如下。在第2节中，我们根据上述要求对聚类算法进行评估和讨论。在第3节中，我们介绍基于数据库密度概念的聚类概念。第4节介绍在空间数据库中发现此类聚类的DBSCAN算法。在第5节中，我们使用合成数据和红杉2000基准测试的数据对DBSCAN的有效性和效率进行了实验评估。第6节总结全文并指出未来研究的方向。

## 2. Clustering Algorithms

## 2. 聚类算法

There are two basic types of clustering algorithms (Kaufman & Rousseeuw 1990): partitioning and hierarchical algorithms. Partitioning algorithms construct a partition of a database $D$ of $n$ objects into a set of $k$ clusters. $k$ is an input parameter for these algorithms, i.e some domain knowledge is required which unfortunately is not available for many applications. The partitioning algorithm typically starts with an initial partition of $D$ and then uses an iterative control strategy to optimize an objective function. Each cluster is represented by the gravity center of the cluster ( $k$ -means algorithms) or by one of the objects of the cluster located near its center ( $k$ -medoid algorithms). Consequently,partitioning algorithms use a two-step procedure. First,determine $k$ representatives minimizing the objective function. Second, assign each object to the cluster with its representative "closest" to the considered object. The second step implies that a partition is equivalent to a voronoi diagram and each cluster is contained in one of the voronoi cells. Thus, the shape of all clusters found by a partitioning algorithm is convex which is very restrictive.

聚类算法有两种基本类型（考夫曼和鲁塞厄 1990 年）：划分算法和层次算法。划分算法将包含 $n$ 个对象的数据库 $D$ 划分为 $k$ 个簇的集合。$k$ 是这些算法的一个输入参数，即需要一些领域知识，不幸的是，许多应用中并没有这些知识。划分算法通常从 $D$ 的初始划分开始，然后使用迭代控制策略来优化目标函数。每个簇由簇的重心（$k$ -均值算法）或位于簇中心附近的一个对象（$k$ -中心点算法）来表示。因此，划分算法使用两步过程。首先，确定使目标函数最小化的 $k$ 个代表。其次，将每个对象分配到其代表与该对象“最接近”的簇中。第二步意味着划分等同于一个沃罗诺伊图（Voronoi diagram），并且每个簇都包含在一个沃罗诺伊单元中。因此，划分算法找到的所有簇的形状都是凸的，这是非常有局限性的。

$\mathrm{{Ng}}\& \mathrm{{Han}}$ (1994) explore partitioning algorithms for KDD in spatial databases. An algorithm called CLARANS (Clustering Large Applications based on RANdomized Search) is introduced which is an improved k -medoid method. Compared to former k-medoid algorithms, CLARANS is more effective and more efficient. An experimental evaluation indicates that CLARANS runs efficiently on databases of thousands of objects. Ng & Han (1994) also discuss methods to determine the "natural" number ${\mathrm{k}}_{\text{nat }}$ of clusters in a database. They propose to run CLARANS once for each $\mathrm{k}$ from 2 to $\mathrm{n}$ . For each of the discovered clusterings the silhouette coefficient (Kaufman & Rousseeuw 1990) is calculated, and finally, the clustering with the maximum silhouette coefficient is chosen as the "natural" clustering. Unfortunately, the run time of this approach is prohibitive for large $n$ ,because it implies $O\left( n\right)$ calls of CLARANS.

$\mathrm{{Ng}}\& \mathrm{{Han}}$（1994 年）探索了空间数据库中用于知识发现（KDD）的划分算法。引入了一种名为 CLARANS（基于随机搜索的大规模应用聚类）的算法，它是一种改进的 k -中心点方法。与以前的 k -中心点算法相比，CLARANS 更有效且更高效。实验评估表明，CLARANS 在包含数千个对象的数据库上能高效运行。吴和韩（1994 年）还讨论了确定数据库中簇的“自然”数量 ${\mathrm{k}}_{\text{nat }}$ 的方法。他们建议对从 2 到 $\mathrm{n}$ 的每个 $\mathrm{k}$ 运行一次 CLARANS。对于每个发现的聚类，计算轮廓系数（考夫曼和鲁塞厄 1990 年），最后，选择轮廓系数最大的聚类作为“自然”聚类。不幸的是，对于大型 $n$，这种方法的运行时间过长，因为这意味着要调用 $O\left( n\right)$ 次 CLARANS。

CLARANS assumes that all objects to be clustered can reside in main memory at the same time which does not hold for large databases. Furthermore, the run time of CLARANS is prohibitive on large databases. Therefore, Ester, Kriegel &Xu (1995) present several focusing techniques which address both of these problems by focusing the clustering process on the relevant parts of the database. First, the focus is small enough to be memory resident and second, the run time of CLARANS on the objects of the focus is significantly less than its run time on the whole database.

CLARANS 假设所有要聚类的对象可以同时驻留在主内存中，这对于大型数据库是不成立的。此外，CLARANS 在大型数据库上的运行时间过长。因此，埃斯特、克里格尔和徐（1995 年）提出了几种聚焦技术，通过将聚类过程聚焦在数据库的相关部分来解决这两个问题。首先，聚焦区域足够小，可以驻留在内存中；其次，CLARANS 在聚焦区域的对象上的运行时间明显少于在整个数据库上的运行时间。

Hierarchical algorithms create a hierarchical decomposition of $D$ . The hierarchical decomposition is represented by a dendrogram,a tree that iteratively splits $D$ into smaller subsets until each subset consists of only one object. In such a hierarchy,each node of the tree represents a cluster of $D$ . The dendrogram can either be created from the leaves up to the root (agglomerative approach) or from the root down to the leaves (divisive approach) by merging or dividing clusters at each step. In contrast to partitioning algorithms, hierarchical algorithms do not need $k$ as an input. However,a termination condition has to be defined indicating when the merge or division process should be terminated. One example of a termination condition in the agglomerative approach is the critical distance ${\mathrm{D}}_{\min }$ between all the clusters of $Q$ .

层次算法对 $D$ 进行层次分解。层次分解由一个树状图（dendrogram）表示，这是一种将 $D$ 迭代地划分为更小的子集，直到每个子集仅包含一个对象的树。在这样的层次结构中，树的每个节点代表 $D$ 的一个簇。树状图可以通过在每一步合并或划分簇，从叶子节点向上到根节点（凝聚法）或从根节点向下到叶子节点（分裂法）来创建。与划分算法不同，层次算法不需要将 $k$ 作为输入。然而，必须定义一个终止条件，指示合并或划分过程何时应该终止。凝聚法中终止条件的一个例子是 $Q$ 的所有簇之间的临界距离 ${\mathrm{D}}_{\min }$。

So far, the main problem with hierarchical clustering algorithms has been the difficulty of deriving appropriate parameters for the termination condition,e.g. a value of ${\mathrm{D}}_{\min }$ which is small enough to separate all "natural" clusters and, at the same time large enough such that no cluster is split into two parts. Recently, in the area of signal processing the hierarchical algorithm Ejcluster has been presented (García, Fdez-Valdivia, Cortijo & Molina 1994) automatically deriving a termination condition. Its key idea is that two points belong to the same cluster if you can walk from the first point to the second one by a "sufficiently small" step. Ejcluster follows the divisive approach. It does not require any input of domain knowledge. Furthermore, experiments show that it is very effective in discovering non-convex clusters. However,the computational cost of Ejcluster is $O\left( {n}^{2}\right)$ due to the distance calculation for each pair of points. This is acceptable for applications such as character recognition with moderate values for $n$ ,but it is prohibitive for applications on large databases.

到目前为止，层次聚类算法的主要问题在于难以推导出终止条件的合适参数，例如一个 ${\mathrm{D}}_{\min }$ 的值，它要足够小以分离所有“自然”的聚类，同时又要足够大，使得没有一个聚类被分割成两部分。最近，在信号处理领域提出了层次算法 Ejcluster（加西亚、费尔南德斯 - 瓦尔迪维亚、科尔蒂霍和莫利纳，1994 年），该算法能自动推导出终止条件。其核心思想是，如果可以通过“足够小”的步长从第一个点走到第二个点，那么这两个点就属于同一个聚类。Ejcluster 采用分裂式方法。它不需要任何领域知识的输入。此外，实验表明，它在发现非凸聚类方面非常有效。然而，由于要计算每对点之间的距离，Ejcluster 的计算成本为 $O\left( {n}^{2}\right)$。对于像字符识别这类 $n$ 值适中的应用来说，这是可以接受的，但对于大型数据库的应用来说，这是难以承受的。

Jain (1988) explores a density based approach to identify clusters in k-dimensional point sets. The data set is partitioned into a number of nonoverlapping cells and histograms are constructed. Cells with relatively high frequency counts of points are the potential cluster centers and the boundaries between clusters fall in the "valleys" of the histogram. This method has the capability of identifying clusters of any shape. However, the space and run-time requirements for storing and searching multidimensional histograms can be enormous. Even if the space and run-time requirements are optimized, the performance of such an approach crucially depends on the size of the cells.

贾因（1988 年）探索了一种基于密度的方法来识别 k 维点集中的聚类。数据集被划分为多个不重叠的单元格，并构建直方图。点的频率计数相对较高的单元格是潜在的聚类中心，而聚类之间的边界位于直方图的“谷”中。这种方法能够识别任何形状的聚类。然而，存储和搜索多维直方图的空间和运行时间要求可能非常巨大。即使对空间和运行时间要求进行了优化，这种方法的性能也关键取决于单元格的大小。

### 3.A Density Based Notion of Clusters

### 3. 基于密度的聚类概念

When looking at the sample sets of points depicted in figure 1, we can easily and unambiguously detect clusters of points and noise points not belonging to any of those clusters.

当查看图 1 中描绘的点样本集时，我们可以轻松且明确地检测出点的聚类以及不属于任何这些聚类的噪声点。

<!-- Media -->

<!-- figureText: ... database 3 database 1 database 2 -->

<img src="https://cdn.noedgeai.com/0195c912-ef4a-7c11-b8a2-dcec81361fe7_1.jpg?x=971&y=821&w=684&h=273&r=0"/>

figure 1: Sample databases

图 1：样本数据库

<!-- Media -->

The main reason why we recognize the clusters is that within each cluster we have a typical density of points which is considerably higher than outside of the cluster. Furthermore, the density within the areas of noise is lower than the density in any of the clusters.

我们能够识别出聚类的主要原因是，在每个聚类内部，我们有一个典型的点密度，该密度明显高于聚类外部。此外，噪声区域内的密度低于任何一个聚类内的密度。

In the following, we try to formalize this intuitive notion of "clusters" and "noise" in a database $D$ of points of some k-dimensional space $S$ . Note that both,our notion of clusters and our algorithm DBSCAN, apply as well to 2D or 3D Euclidean space as to some high dimensional feature space. The key idea is that for each point of a cluster the neighborhood of a given radius has to contain at least a minimum number of points, i.e. the density in the neighborhood has to exceed some threshold. The shape of a neighborhood is determined by the choice of a distance function for two points $p$ and $q$ ,denoted by $\operatorname{dist}\left( {p,q}\right)$ . For instance,when using the Manhattan distance in 2D space, the shape of the neighborhood is rectangular. Note, that our approach works with any distance function so that an appropriate function can be chosen for some given application. For the purpose of proper visualization,all examples will be in $2\mathrm{D}$ space using the Euclidean distance.

接下来，我们尝试在某个 k 维空间 $S$ 的点数据库 $D$ 中，将“聚类”和“噪声”的这种直观概念形式化。请注意，我们的聚类概念和我们的算法 DBSCAN 既适用于二维或三维欧几里得空间，也适用于某些高维特征空间。核心思想是，对于一个聚类中的每个点，给定半径的邻域必须至少包含最小数量的点，即邻域内的密度必须超过某个阈值。邻域的形状由两点 $p$ 和 $q$ 的距离函数的选择决定，用 $\operatorname{dist}\left( {p,q}\right)$ 表示。例如，在二维空间中使用曼哈顿距离时，邻域的形状是矩形。请注意，我们的方法适用于任何距离函数，因此可以为给定的应用选择合适的函数。为了便于正确可视化，所有示例都将使用欧几里得距离在 $2\mathrm{D}$ 空间中进行。

Definition 1: (Eps-neighborhood of a point) The Eps-neighborhood of a point $p$ ,denoted by ${N}_{\text{Eos }}\left( p\right)$ ,is defined by ${N}_{\mathrm{{Eps}}}\left( \mathrm{p}\right)  = \{ \mathrm{q} \in  D \mid  \mathrm{{dist}}\left( {\mathrm{p},\mathrm{q}}\right)  \leq  \mathrm{{Eps}}\} .$

定义 1：（点的 Eps 邻域）点 $p$ 的 Eps 邻域，用 ${N}_{\text{Eos }}\left( p\right)$ 表示，定义为 ${N}_{\mathrm{{Eps}}}\left( \mathrm{p}\right)  = \{ \mathrm{q} \in  D \mid  \mathrm{{dist}}\left( {\mathrm{p},\mathrm{q}}\right)  \leq  \mathrm{{Eps}}\} .$

A naive approach could require for each point in a cluster that there are at least a minimum number (MinPts) of points in an Eps-neighborhood of that point. However, this approach fails because there are two kinds of points in a cluster, points inside of the cluster (core points) and points on the border of the cluster (border points). In general, an Eps-neighborhood of a border point contains significantly less points than an Eps-neighborhood of a core point. Therefore, we would have to set the minimum number of points to a relatively low value in order to include all points belonging to the same cluster. This value, however, will not be characteristic for the respective cluster - particularly in the presence of noise. Therefore, we require that for every point $\mathrm{p}$ in a cluster $\mathrm{C}$ there is a point $\mathrm{q}$ in $\mathrm{C}$ so that $\mathrm{p}$ is inside of the Eps-neighborhood of $q$ and ${N}_{\mathrm{{Eps}}}\left( q\right)$ contains at least MinPts points. This definition is elaborated in the following.

一种简单的方法可能要求对于一个簇中的每个点，该点的Eps邻域内至少有最小数量（MinPts）的点。然而，这种方法行不通，因为一个簇中有两种类型的点，即簇内部的点（核心点）和簇边界上的点（边界点）。一般来说，边界点的Eps邻域包含的点明显少于核心点的Eps邻域。因此，为了包含属于同一簇的所有点，我们必须将最小点数设置为相对较低的值。然而，这个值对于相应的簇来说并不具有代表性，尤其是在存在噪声的情况下。因此，我们要求对于簇$\mathrm{C}$中的每个点$\mathrm{p}$，在$\mathrm{C}$中存在一个点$\mathrm{q}$，使得$\mathrm{p}$在$q$的Eps邻域内，并且${N}_{\mathrm{{Eps}}}\left( q\right)$至少包含MinPts个点。下面将详细阐述这个定义。

Definition 2: (directly density-reachable) A point $\mathrm{p}$ is ${di}$ - rectly density-reachable from a point q wrt. Eps, MinPts if

定义2：（直接密度可达）如果满足以下条件，则点$\mathrm{p}$相对于Eps和MinPts从点q是${di}$ - 直接密度可达的

1) $p \in  {N}_{\text{Eps }}\left( q\right)$ and

1) $p \in  {N}_{\text{Eps }}\left( q\right)$ 并且

2) $\left| {{N}_{\text{Eps }}\left( q\right) }\right|  \geq$ MinPts (core point condition).

2) $\left| {{N}_{\text{Eps }}\left( q\right) }\right|  \geq$ ≥ MinPts（核心点条件）。

Obviously, directly density-reachable is symmetric for pairs of core points. In general, however, it is not symmetric if one core point and one border point are involved. Figure 2 shows the asymmetric case.

显然，对于核心点对，直接密度可达性是对称的。然而，一般来说，如果涉及一个核心点和一个边界点，它就不是对称的。图2展示了非对称的情况。

<!-- Media -->

<!-- figureText: (a) (b) p directly density- reachable from $q$ ${qnotdirectlydensity}$ reachable from p p: border point $q :$ core point -->

<img src="https://cdn.noedgeai.com/0195c912-ef4a-7c11-b8a2-dcec81361fe7_2.jpg?x=163&y=857&w=701&h=164&r=0"/>

## figure 2: core points and border points

## 图2：核心点和边界点

<!-- Media -->

Definition 3: (density-reachable) A point $\mathrm{p}$ is density-reachable from a point $\mathrm{q}$ wrt. Eps and MinPts if there is a chain of points ${p}_{1},\ldots ,{p}_{n},{p}_{1} = q,{p}_{n} = p$ such that ${p}_{i + 1}$ is directly density-reachable from ${p}_{i}$ .

定义3：（密度可达）如果存在一个点链${p}_{1},\ldots ,{p}_{n},{p}_{1} = q,{p}_{n} = p$，使得${p}_{i + 1}$从${p}_{i}$直接密度可达，则点$\mathrm{p}$相对于Eps和MinPts从点$\mathrm{q}$是密度可达的。

Density-reachability is a canonical extension of direct density-reachability. This relation is transitive, but it is not symmetric. Figure 3 depicts the relations of some sample points and, in particular, the asymmetric case. Although not symmetric in general, it is obvious that density-reachability is symmetric for core points.

密度可达性是直接密度可达性的自然扩展。这种关系是传递的，但不是对称的。图3描绘了一些样本点之间的关系，特别是非对称的情况。虽然一般情况下不是对称的，但显然对于核心点来说，密度可达性是对称的。

Two border points of the same cluster $\mathrm{C}$ are possibly not density reachable from each other because the core point condition might not hold for both of them. However, there must be a core point in $\mathrm{C}$ from which both border points of $\mathrm{C}$ are density-reachable. Therefore, we introduce the notion of density-connectivity which covers this relation of border points.

同一簇$\mathrm{C}$的两个边界点可能彼此之间不是密度可达的，因为核心点条件可能对它们两者都不成立。然而，$\mathrm{C}$中必须存在一个核心点，从该核心点出发，$\mathrm{C}$的两个边界点都是密度可达的。因此，我们引入了密度连通性的概念，它涵盖了边界点之间的这种关系。

Definition 4: (density-connected) A point $\mathrm{p}$ is density-connected to a point q wrt. Eps and MinPts if there is a point o such that both, $\mathrm{p}$ and $\mathrm{q}$ are density-reachable from o wrt. Eps and MinPts.

定义4：（密度连通）如果存在一个点o，使得$\mathrm{p}$和$\mathrm{q}$相对于Eps和MinPts都从o是密度可达的，则点$\mathrm{p}$相对于Eps和MinPts与点q是密度连通的。

Density-connectivity is a symmetric relation. For density reachable points, the relation of density-connectivity is also reflexive (c.f. figure 3).

密度连通性是一种对称关系。对于密度可达的点，密度连通性关系也是自反的（参见图3）。

Now, we are able to define our density-based notion of a cluster. Intuitively, a cluster is defined to be a set of density-connected points which is maximal wrt. density-reachability. Noise will be defined relative to a given set of clusters. Noise is simply the set of points in $D$ not belonging to any of its clusters.

现在，我们能够定义基于密度的簇的概念。直观地说，簇被定义为一组密度连通的点，这些点相对于密度可达性是最大的。噪声将相对于给定的一组簇来定义。噪声简单来说就是$D$中不属于任何簇的点的集合。

<!-- Media -->

<!-- figureText: (a) (b) p and q density- connected to each other by 0 p density- reachable from q q not density- reachable from p -->

<img src="https://cdn.noedgeai.com/0195c912-ef4a-7c11-b8a2-dcec81361fe7_2.jpg?x=942&y=155&w=711&h=161&r=0"/>

## figure 3: density-reachability and density-connectivity

## 图3：密度可达性和密度连通性

<!-- Media -->

Definition 5: (cluster) Let $D$ be a database of points. A cluster $C$ wrt. Eps and MinPts is a non-empty subset of $D$ satisfying the following conditions:

定义5：（簇）设$D$是一个点的数据库。相对于Eps和MinPts的簇$C$是$D$的一个非空子集，满足以下条件：

1) $\forall \mathrm{p},\mathrm{q}$ : if $\mathrm{p} \in  \mathrm{C}$ and $\mathrm{q}$ is density-reachable from $\mathrm{p}$ wrt. Eps and MinPts,then $q \in  C$ . (Maximality)

1) $\forall \mathrm{p},\mathrm{q}$ ：如果$\mathrm{p} \in  \mathrm{C}$且$\mathrm{q}$相对于Eps和MinPts从$\mathrm{p}$是密度可达的，那么$q \in  C$ 。（最大性）

2) $\forall \mathrm{p},\mathrm{q} \in  \mathrm{C} : \mathrm{p}$ is density-connected to $\mathrm{q}$ wrt. EPS and MinPts. (Connectivity)

2) $\forall \mathrm{p},\mathrm{q} \in  \mathrm{C} : \mathrm{p}$ 相对于（wrt.）参数 EPS 和 MinPts 与 $\mathrm{q}$ 是密度相连的。（连通性）

Definition 6: (noise) Let ${C}_{1},\ldots ,{C}_{k}$ be the clusters of the database $D$ wrt. parameters ${\mathrm{{Eps}}}_{\mathrm{i}}$ and ${\mathrm{{MinPts}}}_{\mathrm{i}},\mathrm{i} = 1,\ldots ,\mathrm{k}$ . Then we define the noise as the set of points in the database $D$ not belonging to any cluster ${C}_{i}$ ,i.e. noise $= \{ \mathrm{p} \in  D \mid  \forall \mathrm{i} : \mathrm{p}$ $\notin  \left. {C}_{i}\right\}$ .

定义 6：（噪声）设 ${C}_{1},\ldots ,{C}_{k}$ 是数据库 $D$ 相对于参数 ${\mathrm{{Eps}}}_{\mathrm{i}}$ 和 ${\mathrm{{MinPts}}}_{\mathrm{i}},\mathrm{i} = 1,\ldots ,\mathrm{k}$ 的聚类。那么我们将噪声定义为数据库 $D$ 中不属于任何聚类 ${C}_{i}$ 的点的集合，即噪声 $= \{ \mathrm{p} \in  D \mid  \forall \mathrm{i} : \mathrm{p}$ $\notin  \left. {C}_{i}\right\}$ 。

Note that a cluster C wrt. Eps and MinPts contains at least MinPts points because of the following reasons. Since $\mathrm{C}$ contains at least one point $p,p$ must be density-connected to itself via some point o (which may be equal to p). Thus, at least o has to satisfy the core point condition and, consequently, the Eps-Neighborhood of o contains at least MinPts points.

请注意，相对于参数 Eps 和 MinPts 的聚类 C 至少包含 MinPts 个点，原因如下。由于 $\mathrm{C}$ 至少包含一个点 $p,p$ ，$p,p$ 必须通过某个点 o（可能等于 p）与自身密度相连。因此，至少点 o 必须满足核心点条件，并且因此，点 o 的 Eps 邻域至少包含 MinPts 个点。

The following lemmata are important for validating the correctness of our clustering algorithm. Intuitively, they state the following. Given the parameters Eps and MinPts, we can discover a cluster in a two-step approach. First, choose an arbitrary point from the database satisfying the core point condition as a seed. Second, retrieve all points that are density-reachable from the seed obtaining the cluster containing the seed.

以下引理对于验证我们的聚类算法的正确性非常重要。直观地说，它们表明了以下内容。给定参数 Eps 和 MinPts，我们可以通过两步方法发现一个聚类。首先，从数据库中选择一个满足核心点条件的任意点作为种子。其次，检索所有从该种子点密度可达的点，从而得到包含该种子点的聚类。

Lemma 1: Let $p$ be a point in $D$ and $\left| {{\mathrm{N}}_{\mathrm{{Eps}}}\left( \mathrm{p}\right) }\right|  \geq$ MinPts. Then the set $O = \{ \mathrm{o} \mid  \mathrm{o} \in  \mathrm{D}$ and $\mathrm{o}$ is density-reachable from p wrt. Eps and MinPts \} is a cluster wrt. Eps and MinPts.

引理 1：设 $p$ 是 $D$ 中的一个点，且 $\left| {{\mathrm{N}}_{\mathrm{{Eps}}}\left( \mathrm{p}\right) }\right|  \geq$ ≥ MinPts。那么集合 { $O = \{ \mathrm{o} \mid  \mathrm{o} \in  \mathrm{D}$ 且 $\mathrm{o}$ 相对于 Eps 和 MinPts 从 p 是密度可达的 } 是相对于 Eps 和 MinPts 的一个聚类。

It is not obvious that a cluster $C$ wrt. Eps and MinPts is uniquely determined by any of its core points. However, each point in $C$ is density-reachable from any of the core points of $C$ and,therefore,a cluster $C$ contains exactly the points which are density-reachable from an arbitrary core point of $C$ .

相对于 Eps 和 MinPts 的聚类 $C$ 由其任何一个核心点唯一确定这一点并不明显。然而，$C$ 中的每个点都可以从 $C$ 的任何一个核心点密度可达，因此，聚类 $C$ 恰好包含那些从 $C$ 的任意一个核心点密度可达的点。

Lemma 2: Let $C$ be a cluster wrt. Eps and MinPts and let $\mathrm{p}$ be any point in $C$ with $\left| {{\mathrm{N}}_{\mathrm{{Eps}}}\left( \mathrm{p}\right) }\right|  \geq$ MinPts. Then $C$ equals to the set $O = \{ \mathrm{o} \mid  \mathrm{o}$ is density-reachable from $\mathrm{p}$ wrt. Eps and MinPts $\}$ .

引理 2：设 $C$ 是相对于 Eps 和 MinPts 的一个聚类，并且设 $\mathrm{p}$ 是 $C$ 中的任意一个点，且 $\left| {{\mathrm{N}}_{\mathrm{{Eps}}}\left( \mathrm{p}\right) }\right|  \geq$ ≥ MinPts。那么 $C$ 等于集合 { $O = \{ \mathrm{o} \mid  \mathrm{o}$ 相对于 Eps 和 MinPts 从 $\mathrm{p}$ 是密度可达的 $\}$ }。

## 4. DBSCAN: Density Based Spatial Clustering of Applications with Noise

## 4. DBSCAN：基于密度的带噪声应用空间聚类

In this section, we present the algorithm DBSCAN (Density Based Spatial Clustering of Applications with Noise) which is designed to discover the clusters and the noise in a spatial database according to definitions 5 and 6 . Ideally, we would have to know the appropriate parameters Eps and MinPts of each cluster and at least one point from the respective cluster. Then, we could retrieve all points that are density-reachable from the given point using the correct parameters. But there is no easy way to get this information in advance for all clusters of the database. However, there is a simple and effective heuristic (presented in section section 4.2) to determine the parameters Eps and MinPts of the "thinnest", i.e. least dense, cluster in the database. Therefore, DBSCAN uses global values for Eps and MinPts, i.e. the same values for all clusters. The density parameters of the "thinnest" cluster are good candidates for these global parameter values specifying the lowest density which is not considered to be noise.

在本节中，我们介绍 DBSCAN 算法（基于密度的带噪声应用空间聚类），该算法旨在根据定义 5 和 6 发现空间数据库中的聚类和噪声。理想情况下，我们必须知道每个聚类的合适参数 Eps 和 MinPts，以及来自相应聚类的至少一个点。然后，我们可以使用正确的参数检索所有从给定点密度可达的点。但是，没有简单的方法预先为数据库中的所有聚类获取这些信息。然而，有一种简单有效的启发式方法（在 4.2 节中介绍）来确定数据库中“最稀疏”（即密度最小）的聚类的参数 Eps 和 MinPts。因此，DBSCAN 对 Eps 和 MinPts 使用全局值，即对所有聚类使用相同的值。“最稀疏”聚类的密度参数是这些全局参数值的合适候选，这些全局参数值指定了不被视为噪声的最低密度。

### 4.1 The Algorithm

### 4.1 算法

To find a cluster, DBSCAN starts with an arbitrary point p and retrieves all points density-reachable from $\mathrm{p}$ wrt. Eps and MinPts. If $p$ is a core point,this procedure yields a cluster wrt. Eps and MinPts (see Lemma 2). If $p$ is a border point, no points are density-reachable from $p$ and DBSCAN visits the next point of the database.

为了找到一个聚类，DBSCAN算法从任意一点p开始，检索所有在给定邻域半径Eps和最小点数MinPts条件下从$\mathrm{p}$出发密度可达的点。如果$p$是一个核心点，此过程将得到一个关于Eps和MinPts的聚类（见引理2）。如果$p$是一个边界点，那么从$p$出发没有密度可达的点，DBSCAN将访问数据库中的下一个点。

Since we use global values for Eps and MinPts, DBSCAN may merge two clusters according to definition 5 into one cluster, if two clusters of different density are "close" to each other. Let the distance between two sets of points ${\mathrm{S}}_{1}$ and ${\mathrm{S}}_{2}$ be defined as $\operatorname{dist}\left( {{S}_{1},{S}_{2}}\right)  = \min \left\{  {\operatorname{dist}\left( {p,q}\right)  \mid  p \in  {S}_{1},q \in  {S}_{2}}\right\}$ . Then, two sets of points having at least the density of the thinnest cluster will be separated from each other only if the distance between the two sets is larger than Eps. Consequently, a recursive call of DBSCAN may be necessary for the detected clusters with a higher value for MinPts. This is, however, no disadvantage because the recursive application of DBSCAN yields an elegant and very efficient basic algorithm. Furthermore, the recursive clustering of the points of a cluster is only necessary under conditions that can be easily detected.

由于我们对Eps和MinPts使用全局值，如果两个不同密度的聚类彼此“接近”，根据定义5，DBSCAN可能会将这两个聚类合并为一个聚类。设两组点${\mathrm{S}}_{1}$和${\mathrm{S}}_{2}$之间的距离定义为$\operatorname{dist}\left( {{S}_{1},{S}_{2}}\right)  = \min \left\{  {\operatorname{dist}\left( {p,q}\right)  \mid  p \in  {S}_{1},q \in  {S}_{2}}\right\}$。那么，只有当两组点之间的距离大于Eps时，至少具有最稀疏聚类密度的两组点才会彼此分离。因此，对于检测到的聚类，可能需要使用更高的MinPts值对DBSCAN进行递归调用。然而，这并非缺点，因为DBSCAN的递归应用产生了一个优雅且非常高效的基本算法。此外，只有在可以轻松检测到的条件下，才需要对聚类中的点进行递归聚类。

In the following, we present a basic version of DBSCAN omitting details of data types and generation of additional information about clusters:

下面，我们给出DBSCAN的一个基本版本，省略了数据类型的细节以及关于聚类的额外信息的生成：

---

DBSCAN (SetOfPoints, Eps, MinPts)

DBSCAN（点集，邻域半径Eps，最小点数MinPts）

// SetOfPoints is UNCLASSIFIED

// 点集未分类

	ClusterId := nextId(NOISE);

	  聚类编号 := 下一个编号（噪声点编号）;

	FOR i FROM 1 TO SetOfPoints.size DO

	  对于从1到点集大小的i执行循环

		Point := SetOfPoints.get(i);

		    点 := 点集获取(i);

		IF Point.ClId = UNCLASSIFIED THEN

		    如果点的聚类编号 = 未分类 则

			IF ExpandCluster (SetOfPoints, Point,

			      如果扩展聚类（点集，点，

							ClusterId, Eps, MinPts) THEN

							                      聚类编号，邻域半径Eps，最小点数MinPts）为真 则

				ClusterId := nextId(ClusterId)

				        聚类编号 := 下一个编号（聚类编号）

			END IF

		END IF

	END FOR

END; // DBSCAN

结束; // DBSCAN

---

SetOfPoints is either the whole database or a discovered cluster from a previous run. Eps and MinPts are the global density parameters determined either manually or according to the heuristics presented in section 4.2. The function SetOfPoints.get(i) returns the i-th element of SetOfPoints. The most important function

点集可以是整个数据库，也可以是上一次运行中发现的一个聚类。邻域半径Eps和最小点数MinPts是全局密度参数，可以手动确定，也可以根据4.2节中介绍的启发式方法确定。函数点集获取(i)返回点集中的第i个元素。DBSCAN使用的最重要的函数

<!-- Media -->

used by DBSCAN is ExpandCluster which is present-

是扩展聚类函数，其具体内容

---

ed below:

如下所示：

ExpandCluster (SetOfPoints, Point, ClId, Eps,

扩展聚类（点集，点，聚类编号，邻域半径，

												MinPts) : Boolean;

												最小点数）：布尔型;

		seeds:=SetOfPoints.regionQuery(Point,Eps);

		种子点集 := 点集的区域查询（点，邻域半径）;

		IF seeds.size<MinPts THEN // no core point

		如果种子点集的大小 < 最小点数 那么 // 不是核心点

			SetOfPoint.changeClId(Point,NOISE) ;

			点集更改该点的聚类编号为噪声;

			RETURN False;

			返回假;

		ELSE // all points in seeds are density-

		否则 // 种子点集中的所有点都可以从该点密度

							// reachable from Point

							 // 可达

			SetOfPoints.changeClIds(seeds,ClId);

			点集将种子点集的聚类编号更改为聚类编号;

			seeds.delete(Point);

			种子点集删除该点;

			WHILE seeds <> Empty DO

			当种子点集不为空时

					currentP := seeds.first(   );

					当前点 := 种子点集的第一个点;

					result := SetOfPoints.regionQuery(currentP,

					结果 := 点集的区域查询（当前点，

																																	Eps);

																																	邻域半径）;

					IF result.size >= MinPts THEN

					如果结果的大小 >= 最小点数 那么

						FOR i FROM 1 TO result.size DO

						对于从 1 到结果大小的 i 执行

								resultP := result.get(i);

								resultP := result.get(i);（结果P 赋值为 result 列表中索引为 i 的元素）

								IF resultP. ClId

								如果 结果P 的类别ID（ClId）

											IN \{UNCLASSIFIED, NOISE\} THEN

											属于 {未分类（UNCLASSIFIED）, 噪声（NOISE）} 则

										IF resultP.Clid = UNCLASSIFIED THEN

										如果 结果P 的类别ID（ClId）等于 未分类（UNCLASSIFIED）则

											seeds.append(resultP) ;

											将 结果P 添加到种子列表（seeds）中;

										END IF ;

										SetOfPoints.changeClId(resultP,ClId);

										点集（SetOfPoints）将 结果P 的类别ID（ClId）修改为 ClId;

								END IF; // UNCLASSIFIED or NOISE

								结束条件判断; // 未分类（UNCLASSIFIED）或 噪声（NOISE）

							END FOR;

					END IF; // result.size >= MinPts

					结束条件判断; // 结果列表（result）的大小 大于等于 最小点数（MinPts）

					seeds.delete(currentP);

					从种子列表（seeds）中删除 当前点（currentP）;

			END WHILE; // seeds <> Empty

			结束循环; // 种子列表（seeds）不为空

			RETURN True;

			返回真;

		END IF

---

<!-- Media -->

END; // ExpandCluster

结束; // 扩展聚类

A call of SetOfPoints.regionQue-ry (Point, Eps) returns the Eps-Neighborhood of Point in SetOfPoints as a list of points. Region queries can be supported efficiently by spatial access methods such as R*-trees (Beckmann et al. 1990) which are assumed to be available in a SDBS for efficient processing of several types of spatial queries (Brinkhoff et al. 1994). The height of an ${\mathrm{R}}^{ * }$ -tree is $\mathrm{O}\left( {\log \mathrm{n}}\right)$ for a database of $\mathrm{n}$ points in the worst case and a query with a "small" query region has to traverse only a limited number of paths in the ${\mathrm{R}}^{ * }$ -tree. Since the Eps-Neighborhoods are expected to be small compared to the size of the whole data space, the average run time complexity of a single region query is $\mathrm{O}\left( {\log \mathrm{n}}\right)$ . For each of the $\mathrm{n}$ points of the database, we have at most one region query. Thus, the average run time complexity of DBSCAN is $\mathrm{O}\left( {\mathrm{n} * \log \mathrm{n}}\right)$ .

调用SetOfPoints.regionQuery(Point, Eps)会返回点集SetOfPoints中该点的Eps邻域，以点列表的形式呈现。区域查询可以通过空间访问方法（如R*树（贝克曼等人，1990年））高效实现，假设空间数据库系统（SDBS）中具备这些方法，以便高效处理多种类型的空间查询（布林克霍夫等人，1994年）。在最坏情况下，对于包含$\mathrm{n}$个点的数据库，${\mathrm{R}}^{ * }$树的高度为$\mathrm{O}\left( {\log \mathrm{n}}\right)$，并且对于“小”查询区域的查询，只需遍历${\mathrm{R}}^{ * }$树中有限数量的路径。由于预计Eps邻域与整个数据空间的大小相比是较小的，因此单个区域查询的平均运行时间复杂度为$\mathrm{O}\left( {\log \mathrm{n}}\right)$。对于数据库中的$\mathrm{n}$个点，每个点最多进行一次区域查询。因此，DBSCAN算法的平均运行时间复杂度为$\mathrm{O}\left( {\mathrm{n} * \log \mathrm{n}}\right)$。

The ClId (clusterId) of points which have been marked to be NOISE may be changed later, if they are density-reachable from some other point of the database. This happens for border points of a cluster. Those points are not added to the seeds-list because we already know that a point with a CIId of NOISE is not a core point. Adding those points to seeds would only result in additional region queries which would yield no new answers.

已被标记为噪声点的点的聚类ID（ClId），如果它们可以从数据库中的其他某个点密度可达，之后可能会被更改。这种情况会发生在聚类的边界点上。这些点不会被添加到种子列表中，因为我们已经知道聚类ID为噪声的点不是核心点。将这些点添加到种子列表只会导致额外的区域查询，而不会产生新的结果。

If two clusters ${C}_{1}$ and ${C}_{2}$ are very close to each other,it might happen that some point $p$ belongs to both, ${C}_{1}$ and ${C}_{2}$ . Then $p$ must be a border point in both clusters because otherwise ${\mathrm{C}}_{1}$ would be equal to ${\mathrm{C}}_{2}$ since we use global parameters. In this case,point $p$ will be assigned to the cluster discovered first. Except from these rare situations, the result of DBSCAN is independent of the order in which the points of the database are visited due to Lemma 2.

如果两个聚类${C}_{1}$和${C}_{2}$彼此非常接近，可能会出现某个点$p$同时属于${C}_{1}$和${C}_{2}$的情况。那么$p$必定是这两个聚类的边界点，否则由于我们使用全局参数，${\mathrm{C}}_{1}$将等于${\mathrm{C}}_{2}$。在这种情况下，点$p$将被分配到最先发现的聚类中。除了这些罕见情况之外，根据引理2，DBSCAN算法的结果与数据库中点的访问顺序无关。

### 4.2 Determining the Parameters Eps and MinPts

### 4.2 确定参数Eps和MinPts

In this section, we develop a simple but effective heuristic to determine the parameters Eps and MinPts of the "thinnest" cluster in the database. This heuristic is based on the following observation. Let $d$ be the distance of a point $p$ to its $k$ -th nearest neighbor,then the d-neighborhood of p contains exactly $k + 1$ points for almost all points $p$ . The d-neighborhood of $p$ contains more than $k + 1$ points only if several points have exactly the same distance $d$ from $p$ which is quite unlikely. Furthermore,changing $k$ for a point in a cluster does not result in large changes of $d$ . This only happens if the $k$ -th nearest neighbors of $p$ for $k = 1,2,3,\ldots$ are located approximately on a straight line which is in general not true for a point in a cluster.

在本节中，我们将开发一种简单而有效的启发式方法，用于确定数据库中“最稀疏”聚类的参数Eps和MinPts。这种启发式方法基于以下观察。设$d$为点$p$到其第$k$近邻的距离，那么对于几乎所有的点$p$，其d邻域恰好包含$k + 1$个点。只有当多个点与$p$的距离恰好都为$d$时，$p$的d邻域才会包含超过$k + 1$个点，而这种情况是不太可能发生的。此外，对于聚类中的一个点，改变$k$的值不会导致$d$发生较大变化。只有当$k = 1,2,3,\ldots$时$p$的第$k$近邻大致位于一条直线上时，才会出现这种情况，而对于聚类中的点，通常并非如此。

For a given $k$ we define a function $k$ -dist from the database $D$ to the real numbers,mapping each point to the distance from its k -th nearest neighbor. When sorting the points of the database in descending order of their k - dist values,the graph of this function gives some hints concerning the density distribution in the database. We call this graph the sorted $k$ -dist graph. If we choose an arbitrary point $\mathrm{p}$ ,set the parameter Eps to k-dist(p) and set the parameter MinPts to k, all points with an equal or smaller $k$ -dist value will be core points. If we could find a threshold point with the maximal k -dist value in the "thinnest" cluster of $D$ we would have the desired parameter values. The threshold point is the first point in the first "valley" of the sorted k -dist graph (see figure 4). All points with a higher $k$ -dist value ( left of the threshold) are considered to be noise, all other points (right of the threshold) are assigned to some cluster.

对于给定的$k$，我们定义一个从数据库$D$到实数的函数$k$-dist，该函数将每个点映射到其第k近邻的距离。当按照数据库中点的k - dist值降序排列这些点时，该函数的图形会给出一些关于数据库中密度分布的提示。我们将这个图形称为排序后的$k$-dist图。如果我们任意选择一个点$\mathrm{p}$，将参数Eps设置为k - dist(p)，并将参数MinPts设置为k，那么所有k - dist值等于或小于该值的点都将是核心点。如果我们能在$D$的“最稀疏”聚类中找到具有最大k - dist值的阈值点，我们就能得到所需的参数值。阈值点是排序后的k - dist图中第一个“谷”中的第一个点（见图4）。所有k - dist值高于该阈值的点（阈值左侧的点）被视为噪声点，其他所有点（阈值右侧的点）被分配到某个聚类中。

<!-- Media -->

<!-- figureText: 4-dist threshold point points noise clusters -->

<img src="https://cdn.noedgeai.com/0195c912-ef4a-7c11-b8a2-dcec81361fe7_4.jpg?x=273&y=1308&w=600&h=273&r=0"/>

figure 4: sorted 4-dist graph for sample database 3

图4：样本数据库3的排序4-距离图

<!-- Media -->

In general, it is very difficult to detect the first "valley" automatically, but it is relatively simple for a user to see this valley in a graphical representation. Therefore, we propose to follow an interactive approach for determining the threshold point.

一般来说，自动检测第一个“谷值”非常困难，但用户在图形表示中查看这个谷值相对简单。因此，我们建议采用交互式方法来确定阈值点。

DBSCAN needs two parameters, Eps and MinPts. However,our experiments indicate that the $\mathrm{k}$ -dist graphs for $\mathrm{k} > 4$ do not significantly differ from the 4-dist graph and, furthermore, they need considerably more computation. Therefore, we eliminate the parameter MinPts by setting it to 4 for all databases (for 2-dimensional data). We propose the following interactive approach for determining the parameter Eps of DBSCAN:

DBSCAN算法需要两个参数，即邻域半径（Eps）和最小点数（MinPts）。然而，我们的实验表明，$\mathrm{k}$ -距离图与4-距离图相比，对于$\mathrm{k} > 4$并没有显著差异，而且它们需要更多的计算量。因此，对于所有数据库（二维数据），我们将参数MinPts设置为4，从而消除该参数。我们提出以下交互式方法来确定DBSCAN的参数Eps：

- The system computes and displays the 4-dist graph for the database.

- 系统计算并显示数据库的4-距离图。

- If the user can estimate the percentage of noise, this percentage is entered and the system derives a proposal for the threshold point from it.

- 如果用户能够估计噪声的百分比，则输入该百分比，系统会从中得出一个阈值点的建议值。

- The user either accepts the proposed threshold or selects another point as the threshold point. The 4-dist value of the threshold point is used as the Eps value for DBSCAN.

- 用户可以接受建议的阈值，也可以选择另一个点作为阈值点。阈值点的4-距离值将用作DBSCAN的Eps值。

## 5. Performance Evaluation

## 5. 性能评估

In this section, we evaluate the performance of DBSCAN. We compare it with the performance of CLARANS because this is the first and only clustering algorithm designed for the purpose of KDD. In our future research, we will perform a comparison with classical density based clustering algorithms. We have implemented DBSCAN in C++ based on an implementation of the R*-tree (Beckmann et al. 1990). All experiments have been run on HP 735 / 100 workstations. We have used both synthetic sample databases and the database of the SEQUOIA 2000 benchmark.

在本节中，我们评估DBSCAN的性能。我们将其与CLARANS的性能进行比较，因为CLARANS是第一个也是唯一一个专为知识发现与数据挖掘（KDD）目的而设计的聚类算法。在未来的研究中，我们将与经典的基于密度的聚类算法进行比较。我们基于R*树（Beckmann等人，1990年）的实现，用C++实现了DBSCAN。所有实验均在惠普735 / 100工作站上运行。我们使用了合成样本数据库和红杉2000基准测试数据库。

To compare DBSCAN with CLARANS in terms of effectivity (accuracy), we use the three synthetic sample databases which are depicted in figure 1. Since DBSCAN and CLARANS are clustering algorithms of different types, they have no common quantitative measure of the classification accuracy. Therefore, we evaluate the accuracy of both algorithms by visual inspection. In sample database 1, there are four ball-shaped clusters of significantly differing sizes. Sample database 2 contains four clusters of nonconvex shape. In sample database 3, there are four clusters of different shape and size with additional noise. To show the results of both clustering algorithms, we visualize each cluster by a different color (see www availability after section 6). To give CLARANS some advantage,we set the parameter $k$ to 4 for these sample databases. The clusterings discovered by CLARANS are depicted in figure 5.

为了在有效性（准确性）方面比较DBSCAN和CLARANS，我们使用图1中所示的三个合成样本数据库。由于DBSCAN和CLARANS是不同类型的聚类算法，它们没有共同的分类准确性定量度量。因此，我们通过目视检查来评估这两种算法的准确性。在样本数据库1中，有四个大小明显不同的球形聚类。样本数据库2包含四个非凸形状的聚类。在样本数据库3中，有四个不同形状和大小的聚类，还有额外的噪声。为了展示这两种聚类算法的结果，我们用不同的颜色可视化每个聚类（见第6节之后的网址可用性）。为了给CLARANS一些优势，我们将这些样本数据库的参数$k$设置为4。CLARANS发现的聚类如图5所示。

<!-- Media -->

<!-- figureText: database 1 database 2 database 3 -->

<img src="https://cdn.noedgeai.com/0195c912-ef4a-7c11-b8a2-dcec81361fe7_4.jpg?x=1000&y=1375&w=678&h=256&r=0"/>

figure 5: Clusterings discovered by CLARANS

图5：CLARANS发现的聚类

<!-- Media -->

For DBSCAN, we set the noise percentage to 0% for sample databases 1 and 2, and to 10% for sample database 3, respectively. The clusterings discovered by DBSCAN are depicted in figure 6 .

对于DBSCAN，我们分别将样本数据库1和2的噪声百分比设置为0%，将样本数据库3的噪声百分比设置为10%。DBSCAN发现的聚类如图6所示。

DBSCAN discovers all clusters (according to definition 5) and detects the noise points (according to definition 6) from all sample databases. CLARANS, however, splits clusters if they are relatively large or if they are close to some other cluster. Furthermore, CLARANS has no explicit notion of noise. Instead, all points are assigned to their closest medoid.

DBSCAN从所有样本数据库中发现了所有聚类（根据定义5）并检测到了噪声点（根据定义6）。然而，如果聚类相对较大或者它们靠近其他聚类，CLARANS会分割聚类。此外，CLARANS没有明确的噪声概念。相反，所有点都被分配到离它们最近的中心点。

<!-- Media -->

<!-- figureText: database 1 database 2 道 database 3 -->

<img src="https://cdn.noedgeai.com/0195c912-ef4a-7c11-b8a2-dcec81361fe7_5.jpg?x=180&y=151&w=675&h=259&r=0"/>

## figure 6: Clusterings discovered by DBSCAN

## 图6：DBSCAN发现的聚类

<!-- Media -->

To test the efficiency of DBSCAN and CLARANS, we use the SEQUOIA 2000 benchmark data. The SEQUOIA 2000 benchmark database (Stonebraker et al. 1993) uses real data sets that are representative of Earth Science tasks. There are four types of data in the database: raster data, point data, polygon data and directed graph data. The point data set contains 62,584 Californian names of landmarks, extracted from the US Geological Survey's Geographic Names Information System, together with their location. The point data set occupies about ${2.1}\mathrm{M}$ bytes. Since the run time of CLAR-ANS on the whole data set is very high, we have extracted a series of subsets of the SEQUIOA 2000 point data set containing from $2\%$ to ${20}\%$ representatives of the whole set. The run time comparison of DBSCAN and CLARANS on these databases is shown in table 1 .

为了测试DBSCAN和CLARANS的效率，我们使用红杉2000基准测试数据。红杉2000基准测试数据库（Stonebraker等人，1993年）使用代表地球科学任务的真实数据集。数据库中有四种类型的数据：栅格数据、点数据、多边形数据和有向图数据。点数据集包含从美国地质调查局的地理名称信息系统中提取的62,584个加利福尼亚地标的名称及其位置。点数据集约占${2.1}\mathrm{M}$字节。由于CLARANS在整个数据集上的运行时间非常长，我们从红杉2000点数据集中提取了一系列子集，包含从$2\%$到${20}\%$个整个集合的代表。表1显示了DBSCAN和CLARANS在这些数据库上的运行时间比较。

<!-- Media -->

Table 1: run time in seconds

表1：运行时间（秒）

<table><tr><td>number of points</td><td>1252</td><td>2503</td><td>3910</td><td>5213</td><td>6256</td></tr><tr><td>DBSCAN</td><td>3.1</td><td>6.7</td><td>11.3</td><td>16.0</td><td>17.8</td></tr><tr><td>CLAR- ANS</td><td>758</td><td>3026</td><td>6845</td><td>11745</td><td>18029</td></tr><tr><td>number of points</td><td>7820</td><td>8937</td><td>10426</td><td>12512</td><td/></tr><tr><td>DBSCAN</td><td>24.5</td><td>28.2</td><td>32.7</td><td>41.7</td><td/></tr><tr><td>CLAR- ANS</td><td>29826</td><td>39265</td><td>60540</td><td>80638</td><td/></tr></table>

<table><tbody><tr><td>点数</td><td>1252</td><td>2503</td><td>3910</td><td>5213</td><td>6256</td></tr><tr><td>基于密度的空间聚类应用程序（DBSCAN）</td><td>3.1</td><td>6.7</td><td>11.3</td><td>16.0</td><td>17.8</td></tr><tr><td>CLAR - ANS（原文未明确通用译法，保留英文）</td><td>758</td><td>3026</td><td>6845</td><td>11745</td><td>18029</td></tr><tr><td>点数</td><td>7820</td><td>8937</td><td>10426</td><td>12512</td><td></td></tr><tr><td>基于密度的空间聚类应用程序（DBSCAN）</td><td>24.5</td><td>28.2</td><td>32.7</td><td>41.7</td><td></td></tr><tr><td>CLAR - ANS（原文未明确通用译法，保留英文）</td><td>29826</td><td>39265</td><td>60540</td><td>80638</td><td></td></tr></tbody></table>

The results of our experiments show that the run time of DBSCAN is slightly higher than linear in the number of points. The run time of CLARANS, however, is close to quadratic in the number of points. The results show that DB-SCAN outperforms CLARANS by a factor of between 250 and 1900 which grows with increasing size of the database.

我们的实验结果表明，DBSCAN（基于密度的空间聚类应用程序）的运行时间与点数呈略高于线性的关系。然而，CLARANS（基于随机搜索的聚类算法）的运行时间与点数接近二次方关系。结果显示，DBSCAN的性能比CLARANS高出250到1900倍，且随着数据库规模的增大，这一倍数也会增加。

<!-- Media -->

## 6. Conclusions

## 6. 结论

Clustering algorithms are attractive for the task of class identification in spatial databases. However, the well-known algorithms suffer from severe drawbacks when applied to large spatial databases. In this paper, we presented the clustering algorithm DBSCAN which relies on a density-based notion of clusters. It requires only one input parameter and supports the user in determining an appropriate value for it. We performed a performance evaluation on synthetic data and on real data of the SEQUOIA 2000 benchmark. The results of these experiments demonstrate that DBSCAN is significantly more effective in discovering clusters of arbitrary shape than the well-known algorithm CLARANS. Furthermore, the experiments have shown that DBSCAN outperforms CLARANS by a factor of at least 100 in terms of efficiency.

聚类算法在空间数据库的类别识别任务中颇具吸引力。然而，当应用于大型空间数据库时，这些知名算法存在严重的缺陷。在本文中，我们提出了DBSCAN聚类算法，该算法基于基于密度的聚类概念。它只需要一个输入参数，并帮助用户确定该参数的合适值。我们对合成数据和SEQUOIA 2000基准测试的真实数据进行了性能评估。这些实验结果表明，与知名的CLARANS算法相比，DBSCAN在发现任意形状的聚类方面明显更有效。此外，实验还表明，在效率方面，DBSCAN的性能至少比CLARANS高出100倍。

Future research will have to consider the following issues. First, we have only considered point objects. Spatial databases, however, may also contain extended objects such as polygons. We have to develop a definition of the density in an Eps-neighborhood in polygon databases for generalizing DBSCAN. Second, applications of DBSCAN to high dimensional feature spaces should be investigated. In particular,the shape of the k -dist graph in such applications has to be explored.

未来的研究需要考虑以下问题。首先，我们仅考虑了点对象。然而，空间数据库可能还包含诸如多边形等扩展对象。我们必须为多边形数据库中的Eps邻域开发密度定义，以推广DBSCAN算法。其次，应研究DBSCAN在高维特征空间中的应用。特别是，必须探索此类应用中k -距离图的形状。

## WWW Availability

## 万维网获取方式

A version of this paper in larger font, with large figures and clusterings in color is available under the following URL: http://www.dbs.informatik.uni-muenchen.de/ dbs/project/publikationen/veroeffentlichun-gen. html.

本文的大字体版本，包含大尺寸图形和彩色聚类结果，可通过以下URL获取：http://www.dbs.informatik.uni-muenchen.de/ dbs/project/publikationen/veroeffentlichun-gen. html。

## References

## 参考文献

Beckmann N., Kriegel H.-P., Schneider R, and Seeger B. 1990. The

贝克曼（Beckmann）N.、克里格尔（Kriegel）H.-P.、施耐德（Schneider）R和塞格（Seeger）B. 1990年。《

R*-tree: An Efficient and Robust Access Method for Points and Rectangles, Proc. ACM SIGMOD Int. Conf. on Management of Data, Atlantic City, NJ, 1990, pp. 322-331.

Brinkhoff T., Kriegel H.-P., Schneider R., and Seeger B. 1994 Efficient Multi-Step Processing of Spatial Joins, Proc. ACM SIGMOD Int. Conf. on Management of Data, Minneapolis, MN, 1994, pp. 197-208.

Ester M., Kriegel H.-P., and Xu X. 1995. A Database Interface for Clustering in Large Spatial Databases, Proc. 1st Int. Conf. on Knowledge Discovery and Data Mining, Montreal, Canada, 1995, AAAI Press, 1995.

García J.A., Fdez-Valdivia J., Cortijo F. J., and Molina R. 1994. A Dynamic Approach for Clustering Data. Signal Processing, Vol. 44, No. 2, 1994, pp. 181-196.

Gueting R.H. 1994. An Introduction to Spatial Database Systems. The VLDB Journal 3(4): 357-399.

Jain Anil K. 1988. Algorithms for Clustering Data. Prentice Hall.

Kaufman L., and Rousseeuw P.J. 1990. Finding Groups in Data: an Introduction to Cluster Analysis. John Wiley & Sons.

Matheus C.J.; Chan P.K.; and Piatetsky-Shapiro G. 1993. Systems for Knowledge Discovery in Databases, IEEE Transactions on Knowledge and Data Engineering 5(6): 903-913.

Ng R.T., and Han J. 1994. Efficient and Effective Clustering Methods for Spatial Data Mining, Proc. 20th Int. Conf. on Very Large Data Bases, 144-155. Santiago, Chile.

Stonebraker M., Frew J., Gardels K., and Meredith J.1993. The SEQUOIA 2000 Storage Benchmark, Proc. ACM SIGMOD Int. Conf. on Management of Data, Washington, DC, 1993, pp. 2-11.