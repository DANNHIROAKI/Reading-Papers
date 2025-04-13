# FINGER: Fast Inference for Graph-based Approximate Nearest Neighbor Search

# FINGER：基于图的近似最近邻搜索的快速推理

Patrick H. Chen

帕特里克·H·陈

UCLA, Los Angeles, CA,, USA

美国加利福尼亚州洛杉矶市加州大学洛杉矶分校

patrickchen@g.ucla.edu

Wei-Cheng Chang

张维政

Amazon Search, Palo Alto, CA,, USA

美国加利福尼亚州帕洛阿尔托市亚马逊搜索部门

weicheng.cmu@gmail.com

Jyun-Yu Jiang

江俊宇

Amazon Search, Palo Alto, CA,, USA

美国加利福尼亚州帕洛阿尔托市亚马逊搜索部门

jyunyu.jiang@gmail.com

Hsiang-Fu Yu

余翔富

Amazon Search, Palo Alto, CA,, USA

美国加利福尼亚州帕洛阿尔托市亚马逊搜索部门

rofu.yu@gmail.com

Inderjit S. Dhillon

英德吉特·S·狄隆

Google and UT Austin, Austin, TX,

谷歌公司和美国得克萨斯州奥斯汀市得克萨斯大学奥斯汀分校

USA

inderjit@cs.utexas.edu

Cho-Jui Hsieh

谢卓睿

Amazon Search and UCLA, Los

亚马逊搜索部门和美国加利福尼亚州洛杉矶市加州大学洛杉矶分校

Angeles, CA,, USA

美国加利福尼亚州洛杉矶市

chohsieh@cs.ucla.edu

## ABSTRACT

## 摘要

Approximate K-Nearest Neighbor Search (AKNNS) has now become ubiquitous in modern applications, such as a fast search procedure with two-tower deep learning models. Graph-based methods for AKNNS in particular have received great attention due to their superior performance. These methods rely on greedy graph search to traverse the data points as embedding vectors in a database. Under this greedy search scheme, we make a key observation: many distance computations do not influence search updates so that these computations can be approximated without hurting performance. As a result, we propose FINGER, a fast inference method for efficient graph search in AKNNS. FINGER approximates the distance function by estimating angles between neighboring residual vectors. The approximated distance can be used to bypass unnecessary computations for faster searches. Empirically, when it comes to speeding up the inference of HNSW, which is one of the most popular graph-based AKNNS methods, FINGER significantly outperforms existing acceleration approaches and conventional libraries by ${20}\%$ to ${60}\%$ across different benchmark datasets.

近似K近邻搜索（Approximate K-Nearest Neighbor Search，AKNNS）如今在现代应用中已无处不在，例如双塔深度学习模型的快速搜索过程。特别是基于图的AKNNS方法，因其卓越的性能而备受关注。这些方法依靠贪心图搜索来遍历数据库中作为嵌入向量的数据点。在这种贪心搜索方案下，我们有一个关键发现：许多距离计算不会影响搜索更新，因此可以在不影响性能的情况下对这些计算进行近似处理。因此，我们提出了FINGER，这是一种用于AKNNS中高效图搜索的快速推理方法。FINGER通过估计相邻残差向量之间的角度来近似距离函数。近似后的距离可用于绕过不必要的计算，从而实现更快的搜索。根据经验，在加速HNSW（最流行的基于图的AKNNS方法之一）的推理时，FINGER在不同的基准数据集上比现有的加速方法和传统库的性能显著提高了${20}\%$至${60}\%$。

## KEYWORDS

## 关键词

Approximate K-Nearest Neighbor Search (AKNNS); Similarity Search; Graph-based Approximate K-Nearest Neighbor Search.

近似k近邻搜索（Approximate K-Nearest Neighbor Search，AKNNS）；相似度搜索；基于图的近似k近邻搜索

## ACM Reference Format:

## ACM引用格式：

Patrick H. Chen, Wei-Cheng Chang, Jyun-Yu Jiang, Hsiang-Fu Yu, Inderjit S. Dhillon, and Cho-Jui Hsieh. 2023. FINGER: Fast Inference for Graph-based Approximate Nearest Neighbor Search. In Proceedings of the ACM Web Conference 2023 (WWW '23), April 30-May 04, 2023, Austin, TX, USA. ACM, New York, NY, USA, 11 pages. https://doi.org/10.1145/3543507.358318

Patrick H. Chen、Wei-Cheng Chang、Jyun-Yu Jiang、Hsiang-Fu Yu、Inderjit S. Dhillon和Cho-Jui Hsieh。2023年。FINGER：基于图的近似近邻搜索的快速推理。收录于《2023年ACM网络会议论文集》（WWW '23），2023年4月30日至5月4日，美国德克萨斯州奥斯汀市。美国纽约州纽约市ACM，共11页。https://doi.org/10.1145/3543507.358318

## 1 INTRODUCTION

## 1 引言

$K$ -Nearest Neighbor Search (KNNS) is a fundamental problem in machine learning [6], and is applied in various real-world applications in computer vision, natural language processing, and data mining $\left\lbrack  {9,{38},{41}}\right\rbrack$ . Further,most of the neural embedding-based retrieval and recommendation algorithms require KNNS in the inference phase to find items that are nearest to a given query [50]. Formally,consider a dataset $D$ with $n$ data points $\left\{  {{d}_{1},{d}_{2},\ldots ,{d}_{n}}\right\}$ , where each data point has $m$ -dimensional features. Given a query $q \in  {\mathbb{R}}^{m}$ ,KNNS algorithms return the $K$ closest points in $D$ under a certain distance measure (e.g., ${L2}$ distance $\parallel  \cdot  {\parallel }_{2}$ ). Despite its simplicity, the cost of finding exact nearest neighbors is linear in the size of a dataset, which can be prohibitive for massive datasets in real-time applications. It is almost impossible to obtain exact $K$ -nearest neighbors without a linear scan of the whole dataset due to a well-known phenomenon called curse of dimensionality [26]. Thus, in practice, an exact KNNS becomes time-consuming or even infeasible for large-scale data. To overcome this problem, researchers resort to Approximate $K$ -Nearest Neighbor Search (AKNNS). An AKNNS method proposes a set of $K$ candidate neighbors $T = \left\{  {{t}_{1},\cdots ,{t}_{K}}\right\}$ to approximate the exact answer. Performance of AKNNS is usually measured by recall@K defined as $\frac{\left| T \cap  A\right| }{K}$ ,where $A$ is the set of ground-truth $K$ -nearest neighbors of the query $q$ in the dataset $D$ . Most AKNNS methods try to minimize the search time by leveraging pre-computed data structures while maintaining high recall [27]. Among voluminous AKNNS literature [7, 12, 38, 44, 48], most of the efficient AKNNS methods can be categorized into three categories: quantization methods, space partitioning methods, and graph-based methods. In particular, graph-based methods receive extensive attention from researchers due to their competitive performance. Many papers have reported that graph-based methods are among the most competitive AKNNS methods on various benchmark datasets $\left\lbrack  {3,7,{17},{44}}\right\rbrack$ .

$K$近邻搜索（K-Nearest Neighbor Search，KNNS）是机器学习中的一个基本问题[6]，并应用于计算机视觉、自然语言处理和数据挖掘等各种实际应用中$\left\lbrack  {9,{38},{41}}\right\rbrack$。此外，大多数基于神经嵌入的检索和推荐算法在推理阶段都需要使用KNNS来查找与给定查询最接近的项目[50]。形式上，考虑一个包含$n$个数据点$\left\{  {{d}_{1},{d}_{2},\ldots ,{d}_{n}}\right\}$的数据集$D$，其中每个数据点具有$m$维特征。给定一个查询$q \in  {\mathbb{R}}^{m}$，KNNS算法会在特定距离度量（例如${L2}$距离$\parallel  \cdot  {\parallel }_{2}$）下返回$D$中最接近的$K$个点。尽管KNNS算法看似简单，但查找精确近邻的成本与数据集的大小呈线性关系，这对于实时应用中的大规模数据集来说可能是难以承受的。由于著名的“维度灾难”现象[26]，如果不线性扫描整个数据集，几乎不可能获得精确的$K$近邻。因此，在实践中，对于大规模数据，精确的KNNS变得耗时甚至不可行。为了克服这个问题，研究人员采用了近似$K$近邻搜索（Approximate $K$-Nearest Neighbor Search，AKNNS）。AKNNS方法提出一组$K$个候选近邻$T = \left\{  {{t}_{1},\cdots ,{t}_{K}}\right\}$来近似精确答案。AKNNS的性能通常用recall@K来衡量，其定义为$\frac{\left| T \cap  A\right| }{K}$，其中$A$是数据集$D$中查询$q$的真实$K$近邻集合。大多数AKNNS方法试图在保持高召回率的同时，通过利用预计算的数据结构来最小化搜索时间[27]。在大量的AKNNS文献[7, 12, 38, 44, 48]中，大多数高效的AKNNS方法可以分为三类：量化方法、空间划分方法和基于图的方法。特别是，基于图的方法因其具有竞争力的性能而受到研究人员的广泛关注。许多论文都报道称，基于图的方法在各种基准数据集上是最具竞争力的AKNNS方法之一$\left\lbrack  {3,7,{17},{44}}\right\rbrack$。

Graph-based methods work by constructing an underlying search graph, where each node in the graph corresponds to a data point in $D$ . Given a query $q$ and a current search node $c$ ,at each step,an algorithm will only calculate distances between $q$ and all neighboring nodes of $c$ . Once the local search of $c$ is completed, the current search node will be replaced with an unexplored node whose distance is the closest to $q$ among all unexplored nodes. Thus, neighboring edge selection of a data point plays an important role in graph-based methods as it controls the complexity of the search space. Consequently, most recent research is focused on how to construct different search graphs or design heuristics to prune edges in a graph to achieve efficient searches $\left\lbrack  {{17},{27},{35},{43}}\right\rbrack$ . Despite different methods having their own advantages, there is no clear winner among these graph construction approaches on all datasets. Following a recent systematic evaluation protocol [3], we evaluate performance by comparing throughput versus recall@10 curves, where a larger area under the curve corresponds to a better method. As shown in Figure 1, many graph-based methods achieve similar performance on three benchmark datasets. A method (e.g., PyNNDescent [13]) can be competitive on a dataset (e.g., GIST-1M- 960) while another method (e.g., HNSW [35]) performs better on the other dataset (e.g., DEEP-10M-96). These results suggest there might not be a single graph construction method that works best, which motivates us to consider the research question: Other than improving an underlying search graph, is there any other strategy to improve search efficiency of all graph-based methods?.

基于图的方法通过构建一个底层搜索图来工作，图中的每个节点对应于$D$中的一个数据点。给定一个查询$q$和当前搜索节点$c$，在每一步，算法只会计算$q$与$c$的所有相邻节点之间的距离。一旦完成$c$的局部搜索，当前搜索节点将被一个未探索过的节点替换，该节点在所有未探索过的节点中与$q$的距离最近。因此，数据点的相邻边选择在基于图的方法中起着重要作用，因为它控制着搜索空间的复杂度。因此，最近的大多数研究都集中在如何构建不同的搜索图或设计启发式方法来修剪图中的边，以实现高效搜索$\left\lbrack  {{17},{27},{35},{43}}\right\rbrack$。尽管不同的方法各有优势，但在所有数据集上，这些图构建方法中并没有明显的赢家。按照最近的系统评估协议[3]，我们通过比较吞吐量与召回率@10曲线来评估性能，曲线下面积越大，方法越好。如图1所示，许多基于图的方法在三个基准数据集上取得了相似的性能。一种方法（例如，PyNNDescent [13]）在一个数据集（例如，GIST - 1M - 960）上可能具有竞争力，而另一种方法（例如，HNSW [35]）在另一个数据集（例如，DEEP - 10M - 96）上表现更好。这些结果表明，可能不存在一种单一的图构建方法能达到最佳效果，这促使我们思考一个研究问题：除了改进底层搜索图之外，是否还有其他策略可以提高所有基于图的方法的搜索效率？

<!-- Media -->

<!-- figureText: NYTIMES-290K-256-angular GIST-1M-960-euclidean DEEP-10M-96-angular HNSW(hnswlib) HNSW(hnswlib) HNSW(nmslib) HNSW(nmslib) HNSW(pecos) HNSW(pecos) HNSW(n2) 20000 HNSW(n2) PyNNDescen Throughoput (#queries/sec) PyNNDescent 15000 10000 Vamana 5000 0.8 0.9 1.0 0.8 0.9 1.0 Recall@10 Recall@10 HNSW(hnswlib) HNSW(nmslib) HNSW(pecos) 20000 HNSW(n2) 4000 throughoput (#queries/sec) PyNNDescent Throughoput (#queries/sec) 2000 Vamana 1500( 10000 1000 0.6 0.7 0.8 0.9 0.5 0.6 0.7 Recall@10 -->

<img src="https://cdn.noedgeai.com/0195c754-cd29-73d1-bc23-778cbb503ea4_1.jpg?x=256&y=238&w=1275&h=457&r=0"/>

Figure 1: Comparison of state-of-the-art graph-based libraries on three benchmark datasets. Throughput versus recall@10 curve is used as the metric, where a larger area under the curve corresponds to a better method. We can observe no single method outperforms the rest on all datasets. Best viewed in color.

图1：三种基准数据集上最先进的基于图的库的比较。使用吞吐量与召回率@10曲线作为指标，曲线下面积越大，方法越好。我们可以观察到，没有一种方法在所有数据集上都优于其他方法。彩色显示效果最佳。

<!-- Media -->

In this paper, instead of proposing yet another graph construction method, we show that for a given graph, part of the computations in the inference phase can be substantially reduced. Specifically, we observe that after a few node updates, most of the distance computations will not influence the search update. This suggests the complexity of distance calculation during an intermediate stage can be reduced without hurting performance. Based on this observation, we propose FINGER, Fast INference for Graph-based approximated nearest neighbor sEaRch, which reduces computational cost in a graph search while maintaining high recall. Our contribution are summarized as follows:

在本文中，我们没有提出另一种图构建方法，而是表明对于给定的图，可以大幅减少推理阶段的部分计算量。具体来说，我们观察到在进行几次节点更新后，大多数距离计算不会影响搜索更新。这表明在中间阶段可以降低距离计算的复杂度，而不会影响性能。基于这一观察，我们提出了FINGER（基于图的近似最近邻快速搜索，Fast INference for Graph - based approximated nearest neighbor sEaRch），它在保持高召回率的同时降低了图搜索的计算成本。我们的贡献总结如下：

- We provide an empirical observation that most of the distance computations in the prevalent best-first-search graph search scheme do not affect final search results. Thus, we can reduce the computational complexity of many distance functions.

- 我们通过实证观察发现，在流行的最佳优先搜索图搜索方案中，大多数距离计算不会影响最终搜索结果。因此，我们可以降低许多距离函数的计算复杂度。

- Leveraging this characteristic, we propose an approximated distance based on modeling angles between neighboring vectors. Unlike previous methods which directly approximate whole L2- distance or inner-product, we propose a simple yet effective decomposition of the distance function and reduce the approximation error by modeling only the angle between residual vectors. This decomposition yields a much smaller approximate error and thus a better search result.

- 利用这一特性，我们基于对相邻向量之间的角度进行建模，提出了一种近似距离。与之前直接近似整个L2距离或内积的方法不同，我们提出了一种简单而有效的距离函数分解方法，仅通过对残差向量之间的角度进行建模来减少近似误差。这种分解产生的近似误差要小得多，因此搜索结果更好。

- We provide an open source efficient $\mathrm{C} +  +$ implementation of the proposed algorithm FINGER on the popular HNSW graph-based method. HNSW-FINGER outperforms many popular graph-based AKNNS algorithms in wall-clock time across various benchmark datasets by ${20}\%$ to ${60}\%$ .

- 我们在流行的基于HNSW图的方法上提供了所提出的FINGER算法的高效$\mathrm{C} +  +$开源实现。在各种基准数据集上，HNSW - FINGER在实际运行时间上比许多流行的基于图的AKNNS算法快${20}\%$到${60}\%$。

## 2 RELATED WORK

## 2 相关工作

There are three major directions in developing efficient approximate K-Nearest-Neighbours Search (AKNNS) methods. The first direction traverses all elements in a database but reduce the complexity of each distance calculation; quantization methods represent this direction. The second direction partitions the search space into regions and only search data points falling into matched regions, including tree-based methods [42] and hashing-based methods [8]. The third direction is graph-based methods which construct a search graph and convert the search procedure into a graph traversal.

开发高效的近似K近邻搜索（AKNNS）方法主要有三个方向。第一个方向是遍历数据库中的所有元素，但降低每次距离计算的复杂度；量化方法代表了这一方向。第二个方向是将搜索空间划分为多个区域，只搜索落入匹配区域的数据点，包括基于树的方法[42]和基于哈希的方法[8]。第三个方向是基于图的方法，它构建一个搜索图，并将搜索过程转换为图遍历。

### 2.1 Quantization Methods

### 2.1 量化方法

Quantization methods compress data points and represent them as short codes. Compressed codes consume less storage and thus achieve more efficient memory bandwidth usage [22]. In addition, the complexity of distance computations can be reduced by computing approximate distances with the pre-computed lookup tables. Quantization can be done by random projections [34], or learned by exploiting structure in the data distribution [36, 39]. In particular, the seminal Product Quantization method [28] separates the data feature space into different parts and constructs a quantization codebook for each chunk. Product Quantization has become the cornerstone for most recent quantization methods $\left\lbrack  {{14},{22},{37},{46}}\right\rbrack$ . There is also work focusing on learning transformations in accordance with product quantization [19]. Most recent quantization methods achieve competitive results on various benchmarks [22, 30].

量化方法对数据点进行压缩，并将它们表示为短码。压缩后的代码占用更少的存储空间，从而更有效地利用内存带宽 [22]。此外，通过使用预计算的查找表计算近似距离，可以降低距离计算的复杂度。量化可以通过随机投影实现 [34]，也可以通过挖掘数据分布中的结构来学习得到 [36, 39]。特别地，具有开创性的乘积量化方法 [28] 将数据特征空间划分为不同部分，并为每个部分构建一个量化码本。乘积量化已成为大多数最新量化方法的基石 $\left\lbrack  {{14},{22},{37},{46}}\right\rbrack$。也有研究专注于根据乘积量化学习变换 [19]。大多数最新的量化方法在各种基准测试中取得了有竞争力的结果 [22, 30]。

### 2.2 Space Partition Methods

### 2.2 空间划分方法

Hashing-based Methods generate low-bit codes for high dimensional data and try to preserve the similarity among the original distance measure. Locality sensitive hashing [20] is a representative framework that enables users to design a set of hashing functions. Some data-dependent hashing functions have also been designed [25, 45]. Nevertheless, a recent review [7] reported the simplest random-projection hashing [8] actually achieves the best performance. According to this review, the advantage of hashing-based methods is simplicity and low memory usage; however, they are significantly outperformed by graph-based methods.

基于哈希的方法为高维数据生成低比特代码，并试图保留原始距离度量中的相似性。局部敏感哈希 [20] 是一个具有代表性的框架，它允许用户设计一组哈希函数。也有人设计了一些依赖于数据的哈希函数 [25, 45]。然而，最近的一篇综述 [7] 指出，最简单的随机投影哈希 [8] 实际上取得了最佳性能。根据这篇综述，基于哈希的方法的优点是简单且内存使用量低；然而，它们的性能明显不如基于图的方法。

Tree-based Methods learn a recursive space partition function as a tree following some criteria. When a new query comes, the learned partition tree is applied to the query and the distance computation is performed only on relevant elements falling in the same sub-tree. Representative methods are KD-tree forest with a unified search queue [42] and ${R}^{ * }$ -tree [5]. Previous studies observed that tree-based methods only work for very low-dimensional data and their performances drop significantly for high-dimensional problems [7].

基于树的方法按照一定的标准学习一个递归的空间划分函数，并将其表示为一棵树。当有新的查询到来时，将学习到的划分树应用于该查询，并仅对落在同一子树中的相关元素进行距离计算。具有代表性的方法有带有统一搜索队列的 KD 树森林 [42] 和 ${R}^{ * }$ 树 [5]。以往的研究发现，基于树的方法仅适用于非常低维的数据，对于高维问题，其性能会显著下降 [7]。

### 2.3 Graph-based Methods

### 2.3 基于图的方法

Graph-based methods date back to theoretical work in the graph theory of graph paradigm with proper theoretical properties [4, 11, 32]. However, these theoretical guarantees only work for low-dimensional data $\left\lbrack  {4,{32}}\right\rbrack$ or require expensive $\left( {O\left( {n}^{2}\right) }\right.$ or higher) index building complexity [11], which is not scalable to large-scale datasets. Recent studies are mostly geared toward approximations of different proximity graph structures in order to improve approximate nearest neighbor search. An early work showing the practical value of these methods could be found in [2], and is a series of works on approximating $K$ -nearest-neighbour graphs $\left\lbrack  {{16},{23},{24},{29}}\right\rbrack$ . Most recent studies approximate monotonic graphs [18] or relative neighbour graph $\left\lbrack  {2,{35}}\right\rbrack$ . In essence,these methods first construct an approximated $K$ -nearest-neighbour graph and prune redundant edges by different criteria inspired by different proximity graph structures. Some other works mixed the above criteria with other heuristics to prune the graph $\left\lbrack  {{17},{27}}\right\rbrack$ . Some pruning strategies can even work on randomly initialized dense graphs [27]. According to various empirical studies $\left\lbrack  {3,7,{24}}\right\rbrack$ ,graph-based methods achieve very competitive performance among all AKNNS methods. Despite concerns about scalability of graph-based methods due to their larger memory usage [14], it has been shown that graph-based methods can be deployed in billion scale commercial usage [18]. In addition, recent studies also demonstrated that graph-based AKNNS can scale quite well on billion-scale benchmarks when implemented on SSD hard-disks [10, 27].

基于图的方法可以追溯到具有适当理论性质的图范式图论的理论研究 [4, 11, 32]。然而，这些理论保证仅适用于低维数据 $\left\lbrack  {4,{32}}\right\rbrack$，或者需要高昂的 $\left( {O\left( {n}^{2}\right) }\right.$（或更高）的索引构建复杂度 [11]，这对于大规模数据集来说是不可扩展的。最近的研究大多致力于对不同的邻近图结构进行近似，以改进近似最近邻搜索。一篇展示这些方法实际价值的早期研究可以在 [2] 中找到，还有一系列关于近似 $K$ 近邻图 $\left\lbrack  {{16},{23},{24},{29}}\right\rbrack$ 的研究。最近的研究大多对单调图 [18] 或相对近邻图 $\left\lbrack  {2,{35}}\right\rbrack$ 进行近似。本质上，这些方法首先构建一个近似的 $K$ 近邻图，并根据不同邻近图结构的启发，通过不同的标准修剪冗余边。其他一些研究将上述标准与其他启发式方法相结合来修剪图 $\left\lbrack  {{17},{27}}\right\rbrack$。一些修剪策略甚至可以应用于随机初始化的稠密图 [27]。根据各种实证研究 $\left\lbrack  {3,7,{24}}\right\rbrack$，基于图的方法在所有近似 k 近邻搜索（AKNNS）方法中取得了非常有竞争力的性能。尽管由于基于图的方法内存使用量较大，人们担心其可扩展性 [14]，但研究表明，基于图的方法可以应用于十亿规模的商业应用 [18]。此外，最近的研究还表明，当在固态硬盘上实现时，基于图的 AKNNS 在十亿规模的基准测试中也能有很好的扩展性 [10, 27]。

In this work, we aim at demonstrating a generic method to accelerate the inference speed of graph-based methods so we will mainly focus on in-memory scenarios. There are also prior works working on better search schemes on graph-based methods by using KD-Tree [40] and clustering [47]. We will provide more details and compare to these baseline methods in Section 4.

在这项工作中，我们旨在展示一种通用的方法来加速基于图的方法的推理速度，因此我们将主要关注内存内场景。也有先前的研究通过使用 KD 树 [40] 和聚类 [47] 来改进基于图的方法的搜索方案。我们将在第 4 节中提供更多细节，并与这些基线方法进行比较。

## 3 FINGER: FAST INFERENCE FOR GRAPH-BASED AKNNS

## 3 FINGER：基于图的近似 k 近邻搜索的快速推理

In this section, we first provide a motivating observation suggesting that approximating distance computations can accelerate inference of graph-based methods. Next, we analyze the distance computation in a graph search and figure out that the key to approximate the distance is to estimate angles between neighboring residual vectors. We then propose FINGER, Fast INference for Graph-based approximated nearest neighbor sEaRch, a low-rank estimation method plus a distribution matching technique to improve the inference speed of general graph-based algorithms.

在本节中，我们首先给出一个有启发性的观察结果，表明近似距离计算可以加速基于图的方法的推理。接下来，我们分析图搜索中的距离计算，发现近似距离的关键在于估计相邻残差向量之间的角度。然后，我们提出了 FINGER（基于图的近似最近邻搜索的快速推理），这是一种低秩估计方法与分布匹配技术相结合的方法，用于提高一般基于图的算法的推理速度。

### 3.1 Observation: Most distance computations do not contribute to better search results

### 3.1 观察结果：大多数距离计算对改善搜索结果并无贡献

Once a search graph is built, graph-based methods use a greedy-search strategy (Algorithm 1) to find relevant elements of a query in a database. It maintains two priority queues: candidate queue that stores potential candidates to expand and top results queue that stores current most similar candidates (line 1). At each iteration, it finds the current nearest point in the candidate queue and explores its neighboring points. An upper-bound variable records the distance of the farthest element from the current top results queue to the query $q$ (line 4). The search will stop when the current nearest distance from the candidate queue is larger than the upper-bound (line 5), or there is no element left in the candidate queue (line 2). The upper-bound not only controls termination of the search but also determines if a point will be present in the candidate queue (line 11). An exploring point will not be added into the candidate queue if the distance from the point to the query is larger than the upper-bound.

一旦构建了搜索图，基于图的方法会使用贪心搜索策略（算法1）在数据库中查找查询的相关元素。它维护两个优先队列：一个是候选队列，用于存储待扩展的潜在候选元素；另一个是顶级结果队列，用于存储当前最相似的候选元素（第1行）。在每次迭代中，它会在候选队列中找到当前最近的点，并探索其相邻点。一个上界变量会记录当前顶级结果队列中最远元素到查询的距离 $q$（第4行）。当候选队列中当前最近距离大于上界时（第5行），或者候选队列中没有剩余元素时（第2行），搜索将停止。上界不仅控制搜索的终止，还决定一个点是否会出现在候选队列中（第11行）。如果一个探索点到查询的距离大于上界，则该点不会被添加到候选队列中。

The upper-bound plays an important role as we need to spend computational resources on distance calculation (dist(   ) function in line 11) but it might not influence search results if the distance is larger than the upper-bound. Empirically, as shown in Figure 2, we observe in two benchmark datasets that most explorations of graph search end up having larger distances than the upper-bound. Especially,starting from the mid-phase of a search,over $\mathbf{{80}}\%$ of distance calculations are larger than the upper-bound. Using greedy graph search will inevitably waste a significant amount of computing time on non-influential operations. [33] also found this phenomenon and proposed to learn an early termination criterion by an ML model. Instead of only focusing on the near-termination phase, we propose a more general framework by incorporating the idea of reducing the complexity of distance calculations into a graph search. The fact that most distance computations do not influence search results suggests that we don't need to have exact distance computations. A faster distance approximation can be applied in the search.

上界起着重要作用，因为我们需要在距离计算（第11行的dist( )函数）上花费计算资源，但如果距离大于上界，它可能不会影响搜索结果。根据经验，如图2所示，我们在两个基准数据集上观察到，图搜索的大多数探索最终得到的距离都大于上界。特别是，从搜索的中间阶段开始，超过 $\mathbf{{80}}\%$ 的距离计算结果都大于上界。使用贪心图搜索不可避免地会在无影响的操作上浪费大量的计算时间。文献[33]也发现了这一现象，并提出通过机器学习模型学习一个提前终止准则。我们没有只关注接近终止的阶段，而是提出了一个更通用的框架，将降低距离计算复杂度的思想融入到图搜索中。大多数距离计算不影响搜索结果这一事实表明，我们不需要进行精确的距离计算。在搜索中可以应用更快的距离近似方法。

### 3.2 Modeling Distributions of Neighboring Residual Angles

### 3.2 对相邻残差角的分布进行建模

In this section, we will derive an efficient method to approximate the distance calculations in the search. While most existing methods approximate the whole L2-distance or inner-product directly, instead, in this section we will show that by simple manipulations, we only need to model angles between neighboring residual pairs and thus a much small approximation error could be achieved. Given a query $q$ and the current point $c$ that is nearest to the query in the candidate queue, we will expand the search by exploring neighbors of $c$ in Line 7 of Algorithm 1. Consider a specific neighbor of $c$ called $d$ ,we have to compute distance between $q$ and $d$ in order to update the search results. Here,we will focus on the ${L2}$ distance (i.e.,Dist $= \parallel q - d{\parallel }_{2}$ ). The derivations of inner-product and angle distance are provided in Appendix D. As shown in the previous section, most distance computations will not contribute to the search in later stages, we aim at finding a fast approximation of ${L2}$ distance. A key idea is that we can leverage $c$ to represent $q$ (and $d$ ) as a vector along $c$ (i.e.,projection) and a vector orthogonal to $c$ (i.e.,residual):

在本节中，我们将推导一种有效的方法来近似搜索中的距离计算。虽然大多数现有方法直接对整个L2距离或内积进行近似，但在本节中，我们将展示通过简单的操作，我们只需要对相邻残差对之间的角度进行建模，从而可以实现更小的近似误差。给定一个查询 $q$ 和候选队列中当前距离查询最近的点 $c$，我们将在算法1的第7行中通过探索 $c$ 的邻居来扩展搜索。考虑 $c$ 的一个特定邻居 $d$，为了更新搜索结果，我们必须计算 $q$ 和 $d$ 之间的距离。这里，我们将关注 ${L2}$ 距离（即Dist $= \parallel q - d{\parallel }_{2}$）。内积和角度距离的推导见附录D。如前一节所示，大多数距离计算在后期阶段对搜索没有贡献，我们的目标是找到 ${L2}$ 距离的快速近似方法。一个关键的想法是，我们可以利用 $c$ 将 $q$（和 $d$）表示为沿着 $c$ 的向量（即投影）和与 $c$ 正交的向量（即残差）：

$$
q = {q}_{\text{proj }} + {q}_{\text{res }},\;{q}_{\text{proj }} = \frac{{c}^{T}q}{{c}^{T}c}c,\;{q}_{\text{res }} = q - {q}_{\text{proj }}. \tag{1}
$$

<!-- Media -->

<!-- figureText: io of Nodes with distance $>$ upper bound Skip Ratio of Running HNSW on FashionMNIST-60K-784 o of Nodes with distance $>$ upper bound Skip Ratio of Running HNSW on GLOVE-1.2M-100 0.9 0.8 0.6 0.3 0.0] Number of Visited Nodes from Candidate Set C (b) GLOVE-1.2M-100 0.9 0.6 0.3 0.0 12 14 16 18 Number of Visited Nodes from Candidate Set C (a) FashionMNIST-60K-784 -->

<img src="https://cdn.noedgeai.com/0195c754-cd29-73d1-bc23-778cbb503ea4_3.jpg?x=213&y=238&w=1365&h=411&r=0"/>

Figure 2: Empirical observation that distances between query and most points in a database are larger than the upper-bound. (a) shows results on FashionMNIST-60K-784 dataset and (b) shows results on Glove-1.2M-100 dataset. We observed that starting from the 5th step of greedy graph search (i.e., running line 2 in Algorithm 1 five times), both experiments show more than 80% of data points will be larger than the current upper-bound. These distance computations won’t affect search updates.

图2：经验观察表明，查询与数据库中大多数点之间的距离大于上界。(a)显示了在FashionMNIST - 60K - 784数据集上的结果，(b)显示了在Glove - 1.2M - 100数据集上的结果。我们观察到，从贪心图搜索的第5步开始（即算法1的第2行运行5次），两个实验都显示超过80%的数据点的距离将大于当前上界。这些距离计算不会影响搜索更新。

Algorithm 1: Greedy Graph Search

算法1：贪心图搜索

---

Input: graph $G$ ,query $q$ ,start point $p$ ,distance dist(   ),

输入：图 $G$，查询 $q$，起始点 $p$，距离函数dist( )，

			number of nearest points to return efs

			要返回的最近点的数量efs

Output: top results queue $T$

输出：顶级结果队列 $T$

candidate queue $C = \{ \mathrm{p}\}$ ,currently top results queue

候选队列 $C = \{ \mathrm{p}\}$，当前顶级结果队列

	$T = \{ \mathrm{p}\}$ ,visited set $V = \{ \mathrm{p}\}$ ;

	$T = \{ \mathrm{p}\}$，已访问集合 $V = \{ \mathrm{p}\}$；

while $C$ is not empty do

当 $C$ 不为空时执行以下操作

		cur $\leftarrow$ nearest element from $C$ to $q$ (i.e.,current nearest

		cur $\leftarrow$ 从 $C$ 到 $q$ 的最近元素（即当前要扩展的最近点）；

		point to expand);

		点进行扩展）；

		ub $\leftarrow$ distance of farthest element from $T$ to $q$ (i.e.,

		ub $\leftarrow$ 从 $T$ 到 $q$ 的最远元素的距离（即

		upper bound of the candidate search);

		候选搜索的上界）;

		if $\operatorname{dist}\left( {{cur},q}\right)  > {ub}$ then

		如果 $\operatorname{dist}\left( {{cur},q}\right)  > {ub}$ 则

			return $T$

			返回 $T$

		for point $n \in$ neighbour of cur in $G$ do

		对于 $G$ 中 cur 的邻居点 $n \in$ 执行

			if $n \in  V$ then

			如果 $n \in  V$ 则

					continue

					继续

			V.add(n)

			V 添加 (n)

			if $\operatorname{dist}\left( {n,q}\right)  \leq  {ub}$ or $\left| T\right|  \leq  {efs}$ then

			如果 $\operatorname{dist}\left( {n,q}\right)  \leq  {ub}$ 或 $\left| T\right|  \leq  {efs}$ 则

					C.add(n)

					C 添加 (n)

					$T$ .add(n)

					$T$ 添加 (n)

					if $\left| T\right|  >$ efs then

					如果 $\left| T\right|  >$ efs 则

						remove farthest point from $T$ to $q$

						移除从 $T$ 到 $q$ 的最远点

					ub $\leftarrow$ distance of farthest element from $T$ to $q$

					ub $\leftarrow$ 从 $T$ 到 $q$ 的最远元素的距离

					(i.e., update ub)

					（即，更新 ub）

return $T$

返回 $T$

---

<!-- Media -->

A schematic illustration of this decomposition is shown in Figure 3. In other words, we treat each center node as a basis and project the query and its neighboring points onto the center vector so query and data can be written as $q = {q}_{\text{proj }} + {q}_{\text{res }}$ and $d = {d}_{\text{proj }} + {d}_{\text{res }}$ respectively. To this end,the squared ${L2}$ distance can be written as:

图3展示了这种分解的示意图。换句话说，我们将每个中心节点视为一个基，并将查询点及其相邻点投影到中心向量上，这样查询点和数据点就可以分别写成$q = {q}_{\text{proj }} + {q}_{\text{res }}$和$d = {d}_{\text{proj }} + {d}_{\text{res }}$。为此，平方${L2}$距离可以写成：

$$
{\text{ Dist }}^{2} = \parallel q - d{\parallel }_{2}^{2} = {\begin{Vmatrix}{q}_{\text{proj }} + {q}_{\text{res }} - {d}_{\text{proj }} - {d}_{\text{res }}\end{Vmatrix}}_{2}^{2}
$$

$$
 = {\begin{Vmatrix}\left( {q}_{\text{proj }} - {d}_{\text{proj }}\right)  + \left( {q}_{\text{res }} - {d}_{\text{res }}\right) \end{Vmatrix}}_{2}^{2}
$$

$$
 = {\begin{Vmatrix}\left( {q}_{\text{proj }} - {d}_{\text{proj }}\right) \end{Vmatrix}}_{2}^{2} + {\begin{Vmatrix}\left( {q}_{\text{res }} - {d}_{\text{res }}\right) \end{Vmatrix}}_{2}^{2}
$$

$$
 + 2{\left( {q}_{\text{proj }} - {d}_{\text{proj }}\right) }^{T}\left( {{q}_{\text{res }} - {d}_{\text{res }}}\right) 
$$

$$
\overset{\left( a\right) }{ = }{\begin{Vmatrix}\left( {q}_{\text{proj }} - {d}_{\text{proj }}\right) \end{Vmatrix}}_{2}^{2} + {\begin{Vmatrix}\left( {q}_{\text{res }} - {d}_{\text{res }}\right) \end{Vmatrix}}_{2}^{2}
$$

$$
 = {\begin{Vmatrix}\left( {q}_{\text{proj }} - {d}_{\text{proj }}\right) \end{Vmatrix}}_{2}^{2} + {\begin{Vmatrix}{q}_{\text{res }}\end{Vmatrix}}_{2}^{2} + {\begin{Vmatrix}{d}_{\text{res }}\end{Vmatrix}}_{2}^{2} - 2{q}_{\text{res }}^{T}{d}_{\text{res }}\text{,} \tag{2}
$$

<!-- Media -->

<!-- figureText: q Gres ures Origin -->

<img src="https://cdn.noedgeai.com/0195c754-cd29-73d1-bc23-778cbb503ea4_3.jpg?x=998&y=821&w=563&h=356&r=0"/>

Figure 3: Decomposition by center point. Query and neighboring data point can be expressed by vectors parallel and orthogonal to the center vector. We named the parallel vector "proj" (projection) and the orthogonal vector "res" (residual).

图3：按中心点分解。查询点和相邻数据点可以用与中心向量平行和正交的向量表示。我们将平行向量命名为“proj”（投影），将正交向量命名为“res”（残差）。

<!-- Media -->

ubwhere (a) comes from the fact that projection vectors are orthogonal to residual vectors so the inner product vanishes. For ${d}_{proj}$ and ${d}_{res}$ ,we can pre-calculate these values after the search graph is constructed. For ${q}_{proj}$ ,notice that center node $c$ is extracted from the candidate queue (Line 3 of Algorithm 1). That means we must have already visited $c$ before. Thus, $\parallel q - c{\parallel }_{2}$ has been calculated and we can get ${q}^{T}c$ by a simple algebraic manipulation:

其中 (a) 源于投影向量与残差向量正交，因此内积为零这一事实。对于${d}_{proj}$和${d}_{res}$，我们可以在构建搜索图后预先计算这些值。对于${q}_{proj}$，注意到中心节点$c$是从候选队列中提取的（算法1的第3行）。这意味着我们之前一定已经访问过$c$。因此，$\parallel q - c{\parallel }_{2}$已经计算出来了，我们可以通过简单的代数运算得到${q}^{T}c$：

$$
{q}^{T}c = \frac{\parallel q{\parallel }_{2}^{2} + \parallel c{\parallel }_{2}^{2} - \parallel q - c{\parallel }_{2}^{2}}{2}.
$$

Since calculation of $\parallel q{\parallel }_{2}^{2}$ is a one-time task for a query,it’s not too costly when a dataset is moderately large. $\parallel c{\parallel }_{2}^{2}$ can again be pre-computed in advance so ${q}^{T}c$ and thus ${q}_{\text{proj }}$ can be obtained in just a few arithmetic operations. Also notice that $\parallel q{\parallel }_{2}^{2} = {\begin{Vmatrix}{q}_{\text{proj }}\end{Vmatrix}}_{2}^{2}$ $+ {\begin{Vmatrix}{q}_{res}\end{Vmatrix}}_{2}^{2}$ as ${q}_{proj}$ and ${q}_{res}$ are orthogonal,so we can get ${\begin{Vmatrix}{q}_{res}\end{Vmatrix}}_{2}^{2}$ by calculating $\parallel q{\parallel }_{2}^{2} - {\begin{Vmatrix}{q}_{\text{proj }}\end{Vmatrix}}_{2}^{2}$ in few operations too.

由于对一个查询而言，$\parallel q{\parallel }_{2}^{2}$的计算是一次性任务，因此当数据集规模适中时，成本并不高。$\parallel c{\parallel }_{2}^{2}$同样可以提前预计算，这样${q}^{T}c$进而${q}_{\text{proj }}$就可以通过几次算术运算得到。还要注意，由于${q}_{proj}$和${q}_{res}$正交，所以$\parallel q{\parallel }_{2}^{2} = {\begin{Vmatrix}{q}_{\text{proj }}\end{Vmatrix}}_{2}^{2}$ $+ {\begin{Vmatrix}{q}_{res}\end{Vmatrix}}_{2}^{2}$，因此我们也可以通过几次运算计算$\parallel q{\parallel }_{2}^{2} - {\begin{Vmatrix}{q}_{\text{proj }}\end{Vmatrix}}_{2}^{2}$来得到${\begin{Vmatrix}{q}_{res}\end{Vmatrix}}_{2}^{2}$。

After the above manipulation, the only uncertain term in Eq. (2) is ${q}_{res}^{T}{d}_{res}$ . If we can estimate this term with less computational resources,we can obtain a fast yet accurate approximation of ${L2}$ distance. Since we have no direct access to the distribution of $q$ and thus ${q}_{res}$ ,we hypothesize we can instead use the distribution of residual vectors between neighbors of $c$ to approximate the distribution of ${q}_{res}^{T}{d}_{res}$ term. The rationale behind this is as we only approximate ${q}_{res}^{T}{d}_{res}$ when $q$ and $c$ are close enough (i.e., $c$ is selected in Line 3 of Algorithm 1),both $q$ and $d$ could be treated as near points in our search graph and thus interaction between ${q}_{res}$ and ${d}_{res}$ might be well approximated by ${d}_{res}^{\prime }{}^{T}{d}_{res}$ ,where ${d}^{\prime }$ is another neighbouring point of $c$ and ${d}_{res}^{\prime }$ is its residual vector. Formally,given an existing search graph $G = \left( {D,E}\right)$ ,where $D$ are nodes in the graph corresponding to data points and $E$ are edges connecting data points, we collect all residual vectors into ${D}_{\text{res }} \in  {\mathbb{R}}^{m \times  \left| E\right| }$ ,where $\left| E\right|$ is total number of edges in $G$ . We assume ${D}_{\text{res }}$ spans the whole space which residual vectors lie in. Obtaining approximated distance of residual vectors can then be formulated as the following optimization problem:

经过上述操作后，方程 (2) 中唯一不确定的项是${q}_{res}^{T}{d}_{res}$。如果我们能够用较少的计算资源估计这项，我们就可以得到${L2}$距离的快速且准确的近似值。由于我们无法直接获取$q$的分布，进而也无法获取${q}_{res}$的分布，我们假设可以用$c$的邻居之间的残差向量的分布来近似${q}_{res}^{T}{d}_{res}$项的分布。其背后的原理是，只有当$q$和$c$足够接近时（即，$c$在算法1的第3行被选中），我们才对${q}_{res}^{T}{d}_{res}$进行近似，在我们的搜索图中，$q$和$d$都可以被视为近邻点，因此${q}_{res}$和${d}_{res}$之间的相互作用可以用${d}_{res}^{\prime }{}^{T}{d}_{res}$很好地近似，其中${d}^{\prime }$是$c$的另一个相邻点，${d}_{res}^{\prime }$是其残差向量。形式上，给定一个现有的搜索图$G = \left( {D,E}\right)$，其中$D$是图中对应于数据点的节点，$E$是连接数据点的边，我们将所有残差向量收集到${D}_{\text{res }} \in  {\mathbb{R}}^{m \times  \left| E\right| }$中，其中$\left| E\right|$是$G$中边的总数。我们假设${D}_{\text{res }}$张成了残差向量所在的整个空间。然后，获得残差向量的近似距离可以表述为以下优化问题：

$$
\underset{P \in  {\mathbb{R}}^{r \times  m}}{\arg \min }{\mathbb{E}}_{x,y \sim  {D}_{\text{res }}}{\begin{Vmatrix}\parallel Px - Py{\parallel }_{2}^{2} - \parallel x - y{\parallel }_{2}^{2}\end{Vmatrix}}_{2}, \tag{3}
$$

where we aim at finding an optimal projection matrix $P$ minimizing the approximating error over the residual pairs ${D}_{\text{res }}$ from training data. It is not hard to see that the Singular Value Decomposition (SVD) of ${D}_{res}$ will provide an answer to the above optimization problem. Nevertheless, low-rank approximation would not be practical as it consumes much more memory usage. Since we have to save low-rank coordinates for residual vector of each edge, total additional memory is $r \times  \left| E\right|  \times  4$ bytes,where 4 comes from using 32 bits floating points to save each coordinate. For a million scale dataset,with a small rank $r = {16}$ and a moderately complex graph (i.e., $\left| E\right|  \approx  5\mathrm{e}7$ ,it will still cost additional ${3.2}\mathrm{{GB}}$ to save and operate. This greatly inhibits the potential deployment on larger datasets and we need to seek a more memory-efficient approach to estimate ${q}_{\text{res }}^{T}{d}_{\text{res }}$

我们的目标是找到一个最优投影矩阵 $P$，使训练数据中残差对 ${D}_{\text{res }}$ 的近似误差最小化。不难看出，${D}_{res}$ 的奇异值分解（SVD，Singular Value Decomposition）将为上述优化问题提供一个答案。然而，低秩近似并不实用，因为它会消耗更多的内存。由于我们必须为每条边的残差向量保存低秩坐标，因此总共需要额外的 $r \times  \left| E\right|  \times  4$ 字节内存，其中 4 是因为使用 32 位浮点数来保存每个坐标。对于百万规模的数据集，当秩 $r = {16}$ 较小时，以及图的复杂度适中（即 $\left| E\right|  \approx  5\mathrm{e}7$），仍然需要额外的 ${3.2}\mathrm{{GB}}$ 来进行保存和操作。这极大地限制了其在更大数据集上的潜在应用，因此我们需要寻找一种更节省内存的方法来估计 ${q}_{\text{res }}^{T}{d}_{\text{res }}$

An intuitive idea to reduce the memory is not to save full floating point precision. We can use IEEE FP16 [1] or even self-defined precision [31] to reduce memory consumption. Following this idea, the extreme case is to just use 1 bit to store the sign of the result, and this connects to the canonical theory in Locality Sensitive Hashing (LSH) [8]. Specifically, Random Projection Locality Sensitive Hashing (RPLSH) samples $r$ random vectors from Normal distributions to form a hashing basis. The angles between vectors can be estimated by the following lemma.

减少内存的一个直观想法是不保存完整的浮点精度。我们可以使用 IEEE FP16 [1] 甚至自定义精度 [31] 来减少内存消耗。按照这个思路，极端情况是仅使用 1 位来存储结果的符号，这与局部敏感哈希（LSH，Locality Sensitive Hashing）的经典理论 [8] 相关。具体来说，随机投影局部敏感哈希（RPLSH，Random Projection Locality Sensitive Hashing）从正态分布中采样 $r$ 个随机向量来形成一个哈希基。向量之间的夹角可以通过以下引理来估计。

LEMMA 1 (LEMMA 3.2 IN [21]). Given $r$ random vectors $B = {\left\{  {v}_{i}\right\}  }_{i = 1}^{r}$ sampled from a Gaussian Distribution, the estimate for the angle between vectors $x$ and $y$ is given by

引理 1（文献 [21] 中的引理 3.2）。给定从高斯分布中采样的 $r$ 个随机向量 $B = {\left\{  {v}_{i}\right\}  }_{i = 1}^{r}$，向量 $x$ 和 $y$ 之间夹角的估计值由下式给出

$$
\frac{1}{\pi r}\mathop{\sum }\limits_{i}\operatorname{sgn}\left( {{x}^{T}{v}_{i}}\right)  \neq  \operatorname{sgn}\left( {{y}^{T}{v}_{i}}\right) .
$$

<!-- Media -->

---

Algorithm 2: FINGER Graph Search

算法 2：FINGER 图搜索

	Input: graph $G$ ,query $q$ ,starting point $p$ ,learend RPLSH

	输入：图 $G$，查询 $q$，起始点 $p$，学习到的 RPLSH

				basis $B$ ,distance function dist(   ),approximate

				基 $B$，距离函数 dist( )，近似

				distance appx(   ),pre-calculated information $S$ ,

				距离 appx( )，预计算信息 $S$，

				number of nearest points to return efs

				要返回的最近点数量 efs

	Output: top candidate set $T$

	输出：顶级候选集 $T$

	Query Projection result $Y = {q}^{T}B$

	查询投影结果 $Y = {q}^{T}B$

	candidate set $C = \{ \mathrm{p}\}$

	候选集 $C = \{ \mathrm{p}\}$

	dynamic list of currently best candidates $T = \{ \mathrm{p}\}$

	当前最佳候选的动态列表 $T = \{ \mathrm{p}\}$

	visited $V = \{ \mathrm{p}\}$

	已访问 $V = \{ \mathrm{p}\}$

	while $C$ is not empty do

	当 $C$ 不为空时

		cur $\leftarrow$ nearest element from $C$ to

		cur $\leftarrow$ 从 $C$ 到的最近元素

		ub $\leftarrow$ distance of the farthest element from $T$ to $q$ (i.e.,

		ub $\leftarrow$ 从 $T$ 到 $q$ 的最远元素的距离（即

			upper bound of the candidate search)

			候选搜索的上界

		if $\operatorname{dist}\left( {{cur},q}\right)  > {ub}$ then

		如果 $\operatorname{dist}\left( {{cur},q}\right)  > {ub}$ 成立，则

				return $\mathrm{T}$

				返回 $\mathrm{T}$

		for point $n \in$ neighbour of cur in $G$ do

		对于 $G$ 中当前点 cur 的邻居点 $n \in$ 执行

				if $n \in  V$ then

				如果 $n \in  V$ 则

					continue

					继续

				V.add(n)

				V 添加(n)

				if #updates of cur > 5 times then

				如果当前的更新次数 > 5 次，则

					$\mathrm{e} = \operatorname{appx}\left( {\mathrm{n},\mathrm{q},S,Y}\right) //$ Approximate Eq.(2)

					$\mathrm{e} = \operatorname{appx}\left( {\mathrm{n},\mathrm{q},S,Y}\right) //$ 近似公式(2)

				else

				否则

					$\mathrm{e} = \operatorname{dist}\left( {\mathrm{n},\mathrm{q}}\right) //$ exact distance $\mathrm{{Eq}}$ .(2)

					$\mathrm{e} = \operatorname{dist}\left( {\mathrm{n},\mathrm{q}}\right) //$ 精确距离 $\mathrm{{Eq}}$ .(2)

				if $e \leq  {ub}$ or $\left| T\right|  \leq  {efs}$ then

				如果 $e \leq  {ub}$ 或 $\left| T\right|  \leq  {efs}$ 则

					update distance to be dist(n,q)

					将距离更新为 dist(n,q)

					C.add(n)

					C 添加(n)

					T.add(n)

					T 添加(n)

					if $\left| T\right|  >$ efs then

					如果 $\left| T\right|  >$ efs 则

							remove farthest point to $q$ from $\mathrm{T}$

							从 $\mathrm{T}$ 中移除距离 $q$ 最远的点

					ub $\leftarrow$ distance of the farthest element from $T$ to

					上界 $\leftarrow$ 从 $T$ 到最远元素的距离

						$q$ (i.e.,update ub)

						$q$ （即，更新上界）

	return $\mathrm{T}$

	返回 $\mathrm{T}$

---

<!-- Media -->

Although this approximation cannot achieve optimal value of the above optimization problem, it uses much less memory as the low-rank results are now stored in binary representation instead of full 32 bits precision. For example,when $r = 8$ it only takes 1 byte to save the pre-computed results,which is much smaller than $8 \times  4 = {32}$ bytes used by low-rank based approximations. To leverage the idea of RPLSH in our approximation, we need to make two adjustments. First, notice that above lemma is used to estimate angles between vectors whereas we want to estimate inner-product. Thus, we have to further decompose ${q}_{res}^{T}{d}_{res}$ into $\begin{Vmatrix}{q}_{res}\end{Vmatrix}\begin{Vmatrix}{d}_{res}\end{Vmatrix}\cos \left( {{q}_{res},{d}_{res}}\right)$ and calculate hamming distance between signed binarized result to estimate only $\cos \left( {{q}_{res},{d}_{res}}\right)$ . Consequently, $\begin{Vmatrix}{d}_{res}\end{Vmatrix}$ needs to be precompute and stored. Second, vanilla random projection guarantees worst case performance [15] and it is oblivious of the data distribution. Since we can sample abundant neighboring residual vectors from the training database, we can leverage the data information to obtain a better approximation. Instead of generating random Gaussian basis, we used top eigenvectors learned from residual pairs ${D}_{\text{res }}$ which will better capture the span of residual vectors. We will show in Section 4 that this modification achieves better results than using random projections. By using signed LSH to store the low-precision low-rank result, we could greatly reduce the memory usage. Detailed analysis of memory usage and a case study is shown in Appendix C.

尽管这种近似方法无法达到上述优化问题的最优值，但它使用的内存要少得多，因为低秩结果现在以二进制表示形式存储，而不是完整的32位精度。例如，当$r = 8$时，只需1个字节来保存预计算的结果，这比基于低秩的近似方法使用的$8 \times  4 = {32}$字节小得多。为了在我们的近似方法中利用随机投影局部敏感哈希（RPLSH）的思想，我们需要进行两项调整。首先，注意到上述引理用于估计向量之间的角度，而我们想要估计内积。因此，我们必须进一步将${q}_{res}^{T}{d}_{res}$分解为$\begin{Vmatrix}{q}_{res}\end{Vmatrix}\begin{Vmatrix}{d}_{res}\end{Vmatrix}\cos \left( {{q}_{res},{d}_{res}}\right)$，并计算有符号二值化结果之间的汉明距离，以仅估计$\cos \left( {{q}_{res},{d}_{res}}\right)$。因此，需要预先计算并存储$\begin{Vmatrix}{d}_{res}\end{Vmatrix}$。其次，普通的随机投影保证了最坏情况下的性能[15]，并且它不考虑数据分布。由于我们可以从训练数据库中采样大量相邻的残差向量，我们可以利用数据信息来获得更好的近似。我们没有生成随机高斯基，而是使用从残差对${D}_{\text{res }}$中学到的前几个特征向量，这将更好地捕捉残差向量的张成空间。我们将在第4节中表明，这种修改比使用随机投影取得了更好的结果。通过使用有符号局部敏感哈希（LSH）来存储低精度低秩结果，我们可以大大减少内存使用。内存使用的详细分析和案例研究见附录C。

### 3.3 Overall Algorithm of FINGER

### 3.3 FINGER的总体算法

Algorithm 2 summarizes how FINGER works. Our aim is to provide a generic acceleration for all graph-based search. Thus, FINGER can applied on top of any existing graph $G$ . Given a query,FINGER firstly compute query basis multiplications (line 1 in Algorithm 2). This is a one-time computation so the cost is negligible when dataset is moderately large. As we mentioned in Section 3.1, most exact distance computation will not lead to an update of candidate set after expanding 5 times of candidate set; therefore, in Line 14 of Algorithm 2, FINGER uses exact distance when exploring first 5 candidates from the candidate set, and starting from the 6th iteration, FINGER uses approximation distance to scan. Approximation function takes neighboring node, query node, query projections and more pre-computed and stored information $S$ . We leave the details of how pre-computed information is used to obtain approximate distance and time complexity analysis in Appendix B.

算法2总结了FINGER的工作原理。我们的目标是为所有基于图的搜索提供通用加速。因此，FINGER可以应用于任何现有的图$G$之上。给定一个查询，FINGER首先计算查询基的乘法（算法2中的第1行）。这是一次性计算，因此当数据集适中时，成本可以忽略不计。正如我们在3.1节中提到的，在候选集扩展5次后，大多数精确距离计算不会导致候选集的更新；因此，在算法2的第14行，FINGER在从候选集中探索前5个候选对象时使用精确距离，从第6次迭代开始，FINGER使用近似距离进行扫描。近似函数需要相邻节点、查询节点、查询投影以及更多预计算和存储的信息$S$。我们将如何使用预计算信息来获得近似距离的细节和时间复杂度分析留在附录B中。

## 4 EXPERIMENTS

## 4 实验

Baseline Methods. We compare FINGER to the most competitive graph-based and quantization methods. We include different implementations of the popular HNSW methods, such as NM-SLIB [35], $\mathrm{n}{2}^{1}$ ,PECOS [49] and HNSWLIB [35]. We also compare other graph construction methods include NGT-PANNG [43] , VA-MANA(DiskANN) [27] and PyNNDescent [13]. Since our goal is to demonstrate FINGER can improve search efficiency of an underlying graph, we mainly include these competitive methods with good python interface and documentation. For quantization methods, we compare to the best performing ScaNN [22] and Faiss-IVFPQFS [30]. In experiments, we combine FINGER with HNSW as it is a simple and prevalent method. The implementation of HNSW-FINGER is based on a modification of PECOS as its codebase is easy to read and extend. Pre-processing cost is discussed in Appendix C.

基线方法。我们将FINGER与最具竞争力的基于图的方法和量化方法进行比较。我们纳入了流行的分层可导航小世界图（HNSW）方法的不同实现，如NM - SLIB [35]、$\mathrm{n}{2}^{1}$、PECOS [49]和HNSWLIB [35]。我们还比较了其他图构建方法，包括近邻图（NGT - PANNG）[43]、VA - MANA（DiskANN）[27]和PyNNDescent [13]。由于我们的目标是证明FINGER可以提高底层图的搜索效率，我们主要纳入了这些具有良好Python接口和文档的有竞争力的方法。对于量化方法，我们与性能最佳的ScaNN [22]和Faiss - IVFPQFS [30]进行比较。在实验中，我们将FINGER与HNSW结合使用，因为它是一种简单且流行的方法。HNSW - FINGER的实现基于对PECOS的修改，因为其代码库易于阅读和扩展。预处理成本在附录C中讨论。

Evaluation Protocol and Dataset. We follow the latest ANN-benchmark protocol [3] to conduct all experiments. Instead of using a single set of hyperparameter, the protocol searches over a predefined set of hyper-parameters ${}^{2}$ for each method,and reports the best performance over each recall regime. In other words, it allows methods to compete others with its own best hyper-parmameters within each recall regime. We follow this protocol to measure recall@10 values and report the best performance over 10 runs. Results will be presented as throughput versus recall@10 charts. A method is better if the area under curve is larger in the plot. All experiments are run on AWS r5dn.24xlarge instance with Intel(R) Xeon(R) Platinum 8259CL CPU @ 2.50GHz. We evaluate results over both ${L2}$ -based and angular-based metric. We represent a dataset with the following format: (dataset name)-(training data size)-(dimensionality of dataset). For ${L2}$ distance measure,we evaluate on FashionMNIST-60K-784, SIFT-1M-128, and GIST-1M-960. For cosine distance measure, we evaluate on NYTIMES-290K-256, GLOVE-1.2M-100 and DEEP-10M-96. More details of each dataset can be found in [3]. For search hyper-parameters, we follow the same set of search grid as hnmslib used in ann-benchmark repository. In addition,we search over $r = {64}$ and 128 number of basis.

评估协议和数据集。我们遵循最新的ANN基准测试协议[3]来进行所有实验。该协议并非使用单一的超参数集，而是针对每种方法在预定义的超参数集${}^{2}$中进行搜索，并报告每个召回率区间内的最佳性能。换句话说，它允许各种方法在每个召回率区间内以其自身的最佳超参数与其他方法竞争。我们遵循此协议来测量召回率@10的值，并报告10次运行中的最佳性能。结果将以吞吐量与召回率@10的图表形式呈现。在图表中，曲线下面积越大的方法越好。所有实验均在配备英特尔（Intel）至强（Xeon）铂金8259CL CPU（主频2.50GHz）的AWS r5dn.24xlarge实例上运行。我们基于${L2}$距离和角度距离这两种度量标准对结果进行评估。我们用以下格式表示数据集：（数据集名称） - （训练数据大小） - （数据集维度）。对于${L2}$距离度量，我们在FashionMNIST - 60K - 784、SIFT - 1M - 128和GIST - 1M - 960数据集上进行评估。对于余弦距离度量，我们在NYTIMES - 290K - 256、GLOVE - 1.2M - 100和DEEP - 10M - 96数据集上进行评估。每个数据集的更多详细信息可在[3]中找到。对于搜索超参数，我们采用与ann - benchmark仓库中hnmslib使用的相同的搜索网格。此外，我们还对$r = {64}$和128个基进行搜索。

### 4.1 Improvements of FINGER over HNSW

### 4.1 FINGER相对于HNSW的改进

In Figure 4, we demonstrate how FINGER accelerates the competitive HNSW algorithm on all datasets. Since FINGER is implemented on top of PECOS, it is important for us to check if PECOS provides any advantage over other HNSW libraries. Results verify that across all 6 datasets, the performance of PECOS does not give an edge over other HNSW implementations, so the performance difference between FINGER and other HNSW implementations could be mostly attributed to the proposed approximate distance search scheme. We observe that FINGER greatly boosts the performance over all different datasets and outperforms existing graph-based algorithms. FINGER works better not only on datasets with large dimensionality such as FashionMNIST-60K-784 and GIST-1M-960, but also works for dimensionality within range between 96 to 128 . This shows that FINGER can accelerate the distance computation across different dimensionalities. Results of comparison to most competitive graph-based methods are shown in Figure 7 of Appendix A. Briefly speaking, HNSW-FINGER outperforms most state-of-the-art graph-based methods except FashionMNIST-60K-784 where PyN-NDescent achieves the best and HNSW-FINGER is the runner-up. Notice that FINGER could also be implemented over other graph structures including PyNNDescent. We chose to build on top of HNSW algorithm only due to its simplicity and popularity. Studying which graph-based method benefits most from FINGER is an interesting future direction. Here, we aim at empirically demonstrating approximated distance function can be integrated into the greedy search for graph-based methods to achieve a better performance.

在图4中，我们展示了FINGER如何在所有数据集上加速具有竞争力的HNSW算法。由于FINGER是基于PECOS实现的，因此我们有必要检查PECOS相对于其他HNSW库是否具有优势。结果证实，在所有6个数据集上，PECOS的性能并不比其他HNSW实现更出色，因此FINGER与其他HNSW实现之间的性能差异主要可归因于所提出的近似距离搜索方案。我们观察到，FINGER在所有不同的数据集上都显著提升了性能，并且优于现有的基于图的算法。FINGER不仅在高维度数据集（如FashionMNIST - 60K - 784和GIST - 1M - 960）上表现更好，而且在维度范围为96至128的数据集上也能发挥作用。这表明FINGER可以在不同维度上加速距离计算。与最具竞争力的基于图的方法的比较结果见附录A的图7。简而言之，除了在FashionMNIST - 60K - 784数据集上PyN - NDescent表现最佳，HNSW - FINGER位居第二之外，HNSW - FINGER优于大多数最先进的基于图的方法。需要注意的是，FINGER也可以在包括PyNNDescent在内的其他图结构上实现。我们选择基于HNSW算法构建，仅仅是因为它简单且流行。研究哪种基于图的方法从FINGER中受益最大是一个有趣的未来研究方向。在这里，我们旨在通过实验证明，近似距离函数可以集成到基于图的方法的贪心搜索中，以实现更好的性能。

### 4.2 Comparison to Previous Search Methods

### 4.2 与以往搜索方法的比较

As noted in Section 2, Xu et al. [47] and Munoz et al. [40] also propose better search methods over vanilla greedy algorithms. In sum, TOGG-KMC [47] uses KD-Tree or clustering to select querying neighbor points, and add a fine-tuned step when searching points near the query. HCNNG [40] uses KD-tree to select points in the same direction as the query. Notice that both TOGG-KMC and HCNNG methods select a subset of points to query but still use full exact distance. Whereas, FINGER still explores all neighbors and use a faster yet accurate enough approximation to scan the distances. Since HCNNG did not release code, we could only use reported results as in Fig. 7 of [40]. Munoz et al. [40] only reported speedup-ratio over exact nearest neighbor search, so we cannot directly compare the throughput numbers. Instead, in this section we will report its speedup ratio over HNSW graph with greedy search algorithm. For TOGG-KMC,we run the released code ${}^{3}$ on greedy (GA) and proposed method (TOGG-KMC) setup and compute the speed-up ratio of TOGG-KMC over GA. Munoz et al. [40] only includes results on SIFT-1M-128, GIST-1M-960, and GLOVE-1.2M- 100 datasets so we could only compare speedup ratios on these datasets, and the result is shown in Figure 5. As we can observe that on lower recall regions, all methods perform similarly well. All methods could at least speedup vanilla greedy search algorithms ${1.25}\mathrm{x}$ . But when we look at recalls larger than .8,only FINGER could speedup HNSW over 50% whereas HCNNG or TOGG-KMC failed to accelerate HNSW graph much on SIFT-1M-128 (HCNNG) or GIST-1M-960 (TOGG-KMC). This shows that these previous method might be data sensitive that it only works on certain data distribution. In addition,FINGER remains steadily about $2\mathrm{x}$ speedup on GIST-1M-960 and other two baselines fall to 1.25x quickly. Overall, only FINGER achieves steady and significant speedup ratio over original HNSW method across all datasets. This part of experiments justify that FINGER is a better accelerating search algorithm compared to previous methods.

如第2节所述，Xu等人[47]和Munoz等人[40]也提出了比普通贪心算法更好的搜索方法。总之，TOGG - KMC[47]使用KD树或聚类来选择查询邻点，并在查询点附近搜索时增加一个微调步骤。HCNNG[40]使用KD树来选择与查询方向相同的点。请注意，TOGG - KMC和HCNNG方法都选择了一个点的子集进行查询，但仍然使用完整的精确距离。而FINGER仍然会探索所有邻点，并使用一种更快且足够准确的近似方法来扫描距离。由于HCNNG没有发布代码，我们只能使用[40]中图7所报告的结果。Munoz等人[40]只报告了相对于精确最近邻搜索的加速比，因此我们无法直接比较吞吐量数值。相反，在本节中，我们将报告它相对于使用贪心搜索算法的HNSW图的加速比。对于TOGG - KMC，我们在贪心算法（GA）和所提出的方法（TOGG - KMC）设置下运行发布的代码${}^{3}$，并计算TOGG - KMC相对于GA的加速比。Munoz等人[40]只包含了SIFT - 1M - 128、GIST - 1M - 960和GLOVE - 1.2M - 100数据集的结果，因此我们只能在这些数据集上比较加速比，结果如图5所示。我们可以观察到，在较低召回率区域，所有方法的表现都相似。所有方法至少可以将普通贪心搜索算法加速${1.25}\mathrm{x}$。但当我们查看召回率大于0.8的情况时，只有FINGER能使HNSW加速超过50%，而HCNNG或TOGG - KMC在SIFT - 1M - 128（HCNNG）或GIST - 1M - 960（TOGG - KMC）数据集上未能显著加速HNSW图。这表明这些先前的方法可能对数据敏感，仅在某些数据分布上有效。此外，FINGER在GIST - 1M - 960数据集上仍能稳定地实现约$2\mathrm{x}$的加速，而其他两个基线方法的加速比很快降至1.25倍。总体而言，在所有数据集上，只有FINGER相对于原始的HNSW方法实现了稳定且显著的加速比。这部分实验证明，与先前的方法相比，FINGER是一种更好的加速搜索算法。

---

<!-- Footnote -->

${}^{1}$ https://github.com/kakao/n2/tree/master

${}^{1}$ https://github.com/kakao/n2/tree/master

${}^{2}$ https://github.com/erikbern/ann-benchmarks/blob/master/algos.yaml

${}^{2}$ https://github.com/erikbern/ann-benchmarks/blob/master/algos.yaml

${}^{3}$ https://github.com/whenever5225/TOGG

${}^{3}$ https://github.com/whenever5225/TOGG

<!-- Footnote -->

---

<!-- Media -->

<!-- figureText: 30000 40000 5000 gist-960-euclidean HNSW-FINGER HNSW-FINGER HNSW(hnswlib) HNSW(hnswlib) HNSW(nmslib) HNSW(nmslib) HNSW(pecos) 4000 HNSW(pecos) HNSW(n2) Throughoput (#queries/sec) HNSW(n2) 2000 1000 0.9 0.5 0.6 0.8 0.9 1.0 Recall10@10 Recall10@10 glove-100-angular 25000 deep-image-96-angular HNSW-FINGER HNSW-FINGER HNSW(hnswlib) HNSW(hnswlib) HNSW(nmslib) HNSW(nmslib) HNSW(pecos) 20000 HNSW(pecos) HNSW(n2) Throughoput (#queries/sec) HNSW(n2) 15000 5000 1.0 0 Recall10@10 Recall10@10 35000 25000 Throughoput (#queries/sec) 20000 Throughoput (#queries/sec) 30000 25000 20000 15000 10000 15000 10000 HNSW-FINGER HNSW(hnswlib HNSW(nmslib) HNSW(pecos) 5000 HNSW(n2) 0.75 0.80 0.85 0.90 0.95 1.00 0.5 0.6 0.7 Recall10@10 20000 nytimes-256-angular 25000 HNSW-FINGER 17500 HNSW(hnswlib) HNSW(nmslib) HNSW(pecos) 20000 Throughoput (#queries/sec) 15000 HNSW(n2) Throughoput (#queries/sec) 15000 12500 7500 5000 5000 2500 0.6 Recall10@10 -->

<img src="https://cdn.noedgeai.com/0195c754-cd29-73d1-bc23-778cbb503ea4_6.jpg?x=154&y=246&w=1485&h=998&r=0"/>

Figure 4: Experimental results of HNSW graph-based methods. Throughput versus Recall@10 chart is plotted for all datasets. Top row presents datasets with ${L2}$ distance measure and bottom row presents datasets with angular distance measure. We can observe a significant performance gain of FINGER over all existing HNSW graph-based implementations. Best viewed in color.

图4：基于HNSW图的方法的实验结果。为所有数据集绘制了吞吐量与Recall@10的图表。第一行展示了使用${L2}$距离度量的数据集，第二行展示了使用角度距离度量的数据集。我们可以观察到，FINGER相对于所有现有的基于HNSW图的实现都有显著的性能提升。彩色显示效果最佳。

<!-- figureText: 2.50 sift-128-euclidean 2.50 gist-960-euclidean 2.50 glove-100-angular HNSW-FINGER 2.25 HCNNG TOGG-KMC Speed-up Ratio over HNSW 2.00 1.50 1.25 1.00 0.75 0.5( Recall10@10 Recall10@10 HNSW-FINGER HNSW-FINGER 2.25 HCNNG 2.25 HCNNG TOGG-KMC TOGG-KMC Speed-up Ratio over HNSW 2.00 Speed-up Ratio over HNSW 2.00 1.50 1.25 1.00 1.50 1.25 1.00 0.75 0.75 0.50 0.50 Recall10@10 -->

<img src="https://cdn.noedgeai.com/0195c754-cd29-73d1-bc23-778cbb503ea4_6.jpg?x=163&y=1422&w=1471&h=499&r=0"/>

Figure 5: Experimental results of comparisons to previous methods. X-axis denotes the recall@10 values. Y-axis denotes the speed-up of each algorithm over HNSW with greedy search algorithm. We can observe FINGER achieves significant speed-up over original HNSW graph on all three datasets. Best viewed in color.

图5：与先前方法比较的实验结果。X轴表示Recall@10的值。Y轴表示每种算法相对于使用贪心搜索算法的HNSW的加速比。我们可以观察到，在所有三个数据集上，FINGER相对于原始的HNSW图实现了显著的加速。彩色显示效果最佳。

<!-- figureText: 30000 25000 20000 nytimes-256-angular HNSW-FINGER HNSW-FINGER HNSW-FINGER-RF HNSW-FINGER-RP HNSW-RPLSH 17500 HNSW-RPLSH HNSW(pecos) HNSW(pecos) HNSW-SVD Throughoput (#queries/sec) 15000 HNSW-SVD 12500 10000 5000 2500 0.8 0.9 1.0 0.5 0.6 0.7 0.8 0.9 1.0 Recall10@10 Recall10@10 25000 20000 Throughoput (#queries/sec) 20000 Throughoput (#queries/sec) 10000 15000 10000 HNSW-FINGER HNSW-FINGER-RP 5000 HNSW(pecos) HNSW-SVD 0.75 0.80 0.85 0.90 0.95 1.00 0.5 0.6 0.7 Recall10@10 -->

<img src="https://cdn.noedgeai.com/0195c754-cd29-73d1-bc23-778cbb503ea4_7.jpg?x=159&y=246&w=1469&h=496&r=0"/>

Figure 6: Experimental results of ablation studies. Throughput versus Recall@10 chart is plotted for all datasets. HNSW(pecos) is the baseline graph of all other approximating methods. HNSW-FINGER-RP uses random Gaussian basis instead of top eigenvectors of residual vectors for hashing. HNSW-SVD and HNSW-RPLSH directly approximate the distance between vectors whereas HNSW-FINGER approximate the angles of residual vectors. We can observe a significant performance gain of FINGER over all other variants. Best viewed in color.

图6：消融实验的结果。为所有数据集绘制了吞吐量与Recall@10的图表。HNSW(pecos)是所有其他近似方法的基线图。HNSW - FINGER - RP使用随机高斯基而不是残差向量的前特征向量进行哈希。HNSW - SVD和HNSW - RPLSH直接近似向量之间的距离，而HNSW - FINGER近似残差向量的角度。我们可以观察到，FINGER相对于所有其他变体都有显著的性能提升。彩色显示效果最佳。

<!-- Media -->

### 4.3 Ablation Studies

### 4.3 消融实验

In this section, we will do two ablation studies to analyze the effectiveness of FINGER. First, we want to demonstrate the proposed basis sampled from top eigenvectors of residual vector matrix indeed performs better than random Gaussian vectors. To justify, we only need to change the way FINGER generates projection basis from learned residual eigenvectors to randomly sampled Gaussian vectors, and we call this method HNSW-FINGER-RP. Comparisons on selected datasets are shown in Figure 6. We can see that on the three selected datasets, HNSW-FINGER all performs 10%-15% better than HNSW-FINGER-RP. This results directly validates the proposed basis generation scheme is better than random Gaussian vectors. Also notice that HNSW-FINGER-RP actually performs better than HNSW(pecos) on all datasets. This further validates that the proposed approximation scheme is useful, and we could use different angle estimations methods to achieve good acceleration.

在本节中，我们将进行两项消融研究，以分析FINGER（快速图近邻搜索方法）的有效性。首先，我们想证明从残差向量矩阵的顶部特征向量采样得到的提议基确实比随机高斯向量表现更好。为了验证这一点，我们只需将FINGER生成投影基的方式从学习到的残差特征向量改为随机采样的高斯向量，我们将这种方法称为HNSW - FINGER - RP。选定数据集上的比较结果如图6所示。我们可以看到，在三个选定的数据集上，HNSW - FINGER的表现都比HNSW - FINGER - RP好10% - 15%。这一结果直接验证了所提议的基生成方案优于随机高斯向量。此外，还需注意的是，HNSW - FINGER - RP在所有数据集上的表现实际上都比HNSW(pecos)好。这进一步验证了所提议的近似方案是有用的，并且我们可以使用不同的角度估计方法来实现良好的加速。

Second, we want to compare the proposed approximating distance function to other canonical choices. Approximating distance function used in FINGER is based on the decomposition of exact distance and FINGER only estimates the angle of residual vectors. To demonstrate the effectiveness of this approach, we could substitute the approximating function in line 15 of Algorithm 2 with other approximating distance functions. A natural candidate is directly using RPLSH to estimate the angles of vectors. Another popular candidate is using low-rank SVD to approximate the distance between two vectors. We call these two approaches HNSW-RPLSH and HNSW-SVD. Results are also shown in Figure 6.

其次，我们想将所提议的近似距离函数与其他经典选择进行比较。FINGER中使用的近似距离函数基于精确距离的分解，并且FINGER仅估计残差向量的角度。为了证明这种方法的有效性，我们可以用其他近似距离函数替换算法2第15行中的近似函数。一个自然的候选方案是直接使用RPLSH（随机投影局部敏感哈希）来估计向量的角度。另一个流行的候选方案是使用低秩奇异值分解（SVD）来近似两个向量之间的距离。我们将这两种方法称为HNSW - RPLSH和HNSW - SVD。结果也如图6所示。

As we can see that HNSW-RPLSH and HNSW-SVD performs much worse than FINGER. In fact, these two methods even failed to accelerate HNSW on GLOVE-1.2M-100 dataset. Notice that a major difference between FINGER and these two methods is that HNSW-RPLSH and HNSW-SVD do not use the decomposition introduced in FINGER. HNSW-RPLSH and HNSW-SVD directly approximate the distance between original vectors instead of only the residual vectors part. This will lead to a much larger approximation error and consequently a worse throughput-recall@10 performance. In particular, HNSW-FINGER-RP and HNSW-RPLSH use the same approximation method and the difference is only the target term of approximation. Given the same amount of approximating capability provided by signed locality sensitive hashing, limiting the approximation to only a smaller portion of whole arithmetic would naturally lead to a smaller approximation error. And we can see that the performance difference between HNSW-RPLSH and HNSW-FINGER-RP is significant. This further validates the effectiveness of the approximation scheme proposed in Section 3.2.

正如我们所见，HNSW - RPLSH和HNSW - SVD的表现远不如FINGER。事实上，在GLOVE - 1.2M - 100数据集上，这两种方法甚至未能加速HNSW。需要注意的是，FINGER与这两种方法的一个主要区别在于，HNSW - RPLSH和HNSW - SVD没有使用FINGER中引入的分解方法。HNSW - RPLSH和HNSW - SVD直接近似原始向量之间的距离，而不是仅近似残差向量部分。这将导致更大的近似误差，从而导致吞吐量 - 召回率@10性能更差。特别是，HNSW - FINGER - RP和HNSW - RPLSH使用相同的近似方法，区别仅在于近似的目标项。在有符号局部敏感哈希提供相同近似能力的情况下，将近似限制在整个运算的较小部分自然会导致较小的近似误差。我们可以看到，HNSW - RPLSH和HNSW - FINGER - RP之间的性能差异显著。这进一步验证了第3.2节中提出的近似方案的有效性。

## 5 CONCLUSIONS

## 5 结论

In this work, we propose FINGER, a fast inference method for graph-based AKNNS. FINGER approximates distance function in graph-based methods by estimating angles between neighboring residual vectors. FINGER leveraged residual bases to perform memory-efficient hashing estimate of residual angles. The approximated distance can be used to bypass unnecessary distance evaluations, which translates into a faster searching. Empirically, FINGER on top of HNSW is shown to outperform all existing graph-based methods.

在这项工作中，我们提出了FINGER，一种用于基于图的近似k近邻搜索（AKNNS）的快速推理方法。FINGER通过估计相邻残差向量之间的角度来近似基于图的方法中的距离函数。FINGER利用残差基对残差角度进行内存高效的哈希估计。近似距离可用于绕过不必要的距离评估，从而实现更快的搜索。根据实验结果，基于HNSW（分层可导航小世界图）的FINGER表现优于所有现有的基于图的方法。

## ACKNOWLEDGMENTS

## 致谢

This work is supported in part by NSF under IIS-2008173 and IIS- 2048280.

这项工作部分得到了美国国家科学基金会（NSF）在IIS - 2008173和IIS - 2048280项目下的支持。

## REFERENCES

## 参考文献

[1] 2019. IEEE Standard for Floating-Point Arithmetic. IEEE Std 754-2019 (Revision of IEEE 754-2008) (2019), 1-84. https://doi.org/10.1109/IEEESTD.2019.8766229

[2] Sunil Arya and David M Mount. 1993. Approximate nearest neighbor queries in fixed dimensions.. In SODA, Vol. 93. Citeseer, 271-280.

[3] Martin Aumüller, Erik Bernhardsson, and Alexander Faithfull. 2020. ANN-Benchmarks: A benchmarking tool for approximate nearest neighbor algorithms. Information Systems 87 (2020), 101374.

[4] Franz Aurenhammer. 1991. Voronoi diagrams-a survey of a fundamental geometric data structure. ACM Computing Surveys (CSUR) 23, 3 (1991), 345-405.

[5] Norbert Beckmann, Hans-Peter Kriegel, Ralf Schneider, and Bernhard Seeger. 1990. The R*-tree: An efficient and robust access method for points and rectangles. In SIGMOD. 322-331.

[6] Christopher M Bishop. 2006. Pattern recognition. Machine learning 128, 9 (2006).

[7] Deng Cai. 2019. A revisit of hashing algorithms for approximate nearest neighbor search. IEEE Transactions on Knowledge and Data Engineering (2019).

[8] Moses S Charikar. 2002. Similarity estimation techniques from rounding algorithms. In STOC. 380-388.

[9] Patrick H Chen, Si Si, Sanjiv Kumar, Yang Li, and Cho-Jui Hsieh. 2019. Learning to screen for fast softmax inference on large vocabulary neural networks. In ${ICLR}$ .

[10] Qi Chen, Bing Zhao, Haidong Wang, Mingqin Li, Chuanjie Liu, Zhiyong Zheng, Mao Yang, and Jingdong Wang. 2021. SPANN: Highly-efficient Billion-scale Approximate Nearest Neighborhood Search. NeurIPS 34 (2021).

[11] DW Dearholt, N Gonzales, and G Kurup. 1988. Monotonic search networks for computer vision databases. In Twenty-Second Asilomar Conference on Signals, Systems and Computers, Vol. 2. IEEE, 548-553.

[12] Qin Ding, Hsiang-Fu Yu, and Cho-Jui Hsieh. 2019. A fast sampling algorithm for maximum inner product search. In The 22nd International Conference on Artificial Intelligence and Statistics. PMLR, 3004-3012.

[13] Wei Dong, Charikar Moses, and Kai Li. 2011. Efficient k-nearest neighbor graph construction for generic similarity measures. In WWW. 577-586.

[14] Matthijs Douze, Alexandre Sablayrolles, and Hervé Jégou. 2018. Link and code: Fast indexing with graphs and compact regression codes. In CVPR. 3646-3654.

[15] Casper Benjamin Freksen. 2021. An Introduction to Johnson-Lindenstrauss

Transforms. arXiv preprint arXiv:2103.00564 (2021).

[16] Cong Fu and Deng Cai. 2016. EFANNA: An extremely fast approximate nearest neighbor search algorithm based on knn graph. arXiv preprint arXiv:1609.07228 (2016).

[17] Cong Fu, Changxu Wang, and Deng Cai. 2021. High Dimensional Similarity Search with Satellite System Graph: Efficiency, Scalability, and Unindexed Query Compatibility. IEEE Trans. Pattern Anal. Mach. Intell. PP (March 2021).

[18] Cong Fu, Chao Xiang, Changxu Wang, and Deng Cai. 2017. Fast Approximate Nearest Neighbor Search With The Navigating Spreading-out Graph. (July 2017). arXiv:cs.LG/1707.00143

[19] Tiezheng Ge, Kaiming He, Qifa Ke, and Jian Sun. 2013. Optimized product quantization. IEEE TPAMI 36, 4 (2013), 744-755.

[20] Aristides Gionis, Piotr Indyk, Rajeev Motwani, et al. 1999. Similarity search in high dimensions via hashing. In VLDB, Vol. 99. 518-529.

[21] Michel X Goemans and David P Williamson. 1995. Improved approximation algorithms for maximum cut and satisfiability problems using semidefinite programming. Journal of the ACM (JACM) 42, 6 (1995), 1115-1145.

[22] Ruiqi Guo, Philip Sun, Erik Lindgren, Quan Geng, David Simcha, Felix Chern, and Sanjiv Kumar. 2020. Accelerating large-scale inference with anisotropic vector quantization. In ICML. PMLR, 3887-3896.

[23] Kiana Hajebi, Yasin Abbasi-Yadkori, Hossein Shahbazi, and Hong Zhang. 2011. Fast approximate nearest-neighbor search with k-nearest neighbor graph. In Twenty-Second International Joint Conference on Artificial Intelligence.

[24] Ben Harwood and Tom Drummond. 2016. FANNG: Fast approximate nearest neighbour graphs. In CVPR. 5713-5722.

[25] Kaiming He, Fang Wen, and Jian Sun. 2013. K-means hashing: An affinity-preserving quantization method for learning binary compact codes. In CVPR.

[26] Piotr Indyk and Rajeev Motwani. 1998. Approximate nearest neighbors: towards removing the curse of dimensionality. In STOC. 604-613.

[27] Suhas Jayaram Subramanya, Fnu Devvrit, Harsha Vardhan Simhadri, Ravishankar Krishnawamy, and Rohan Kadekodi. 2019. DiskANN: Fast accurate billion-point nearest neighbor search on a single node. NeurIPS 32 (2019).

[28] Herve Jegou, Matthijs Douze, and Cordelia Schmid. 2010. Product quantization for nearest neighbor search. IEEE TPAMI 33, 1 (2010), 117-128.

[29] Zhongming Jin, Debing Zhang, Yao Hu, Shiding Lin, Deng Cai, and Xiaofei He. 2014. Fast and accurate hashing via iterative nearest neighbors expansion. IEEE transactions on cybernetics 44, 11 (2014), 2167-2177.

[30] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-scale similarity search with gpus. IEEE Transactions on Big Data 7, 3 (2019), 535-547.

[31] Maximilian Lam. 2018. Word2Bits - Quantized Word Vectors. arXiv preprint arXiv:1803.05651 (2018).

[32] Der-Tsai Lee and Bruce J Schachter. 1980. Two algorithms for constructing a Delaunay triangulation. International Journal of Computer & Information Sciences 9,3 (1980), 219-242.

[33] Conglong Li, Minjia Zhang, David G Andersen, and Yuxiong He. 2020. Improving approximate nearest neighbor search through learned adaptive early termination. In SIGMOD. 2539-2554.

[34] Xiaoyun Li and Ping Li. 2019. Random projections with asymmetric quantization. NeurIPS 32 (2019).

[35] Yu A Malkov and Dmitry A Yashunin. 2018. Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. IEEE

TPAMI 42, 4 (2018), 824-836.

[36] Etienne Marcheret, Vaibhava Goel, and Peder A Olsen. 2009. Optimal quantization and bit allocation for compressing large discriminative feature space transforms. In 2009 IEEE Workshop on ASRU. IEEE, 64-69.

[37] Julieta Martinez, Shobhit Zakhmi, Holger H Hoos, and James J Little. 2018. LSQ++: Lower running time and higher recall in multi-codebook quantization. In ECCV.

[38] Yusuke Matsui, Yusuke Uchida, Hervé Jégou, and Shin'ichi Satoh. 2018. A survey of product quantization. ITE Transactions on Media Technology and Applications 6, 1 (2018), 2-10.

[39] Stanislav Morozov and Artem Babenko. 2019. Unsupervised neural quantization for compressed-domain similarity search. In ICCV. 3036-3045.

[40] Javier Vargas Munoz, Marcos A Gonçalves, Zanoni Dias, and Ricardo da S Torres. 2019. Hierarchical clustering-based graphs for large scale approximate nearest neighbor search. Pattern Recognition 96 (2019), 106970.

[41] Tobias Plötz and Stefan Roth. 2018. Neural nearest neighbors networks. arXiv preprint arXiv:1810.12575 (2018).

[42] Chanop Silpa-Anan and Richard Hartley. 2008. Optimised KD-trees for fast image descriptor matching. In CVPR. IEEE, 1-8.

[43] Kohei Sugawara, Hayato Kobayashi, and Masajiro Iwasaki. 2016. On approximately searching for similar word embeddings. In ${ACL}$ .

[44] Hongya Wang, Zhizheng Wang, Wei Wang, Yingyuan Xiao, Zeng Zhao, and

Kaixiang Yang. 2020. A Note on Graph-Based Nearest Neighbor Search. arXiv preprint arXiv:2012.11083 (2020).

[45] Jun Wang, Sanjiv Kumar, and Shih-Fu Chang. 2010. Sequential projection learning for hashing with compact codes. (2010).

[46] Xiang Wu, Ruiqi Guo, Ananda Theertha Suresh, Sanjiv Kumar, Daniel N Holtmann-Rice, David Simcha, and Felix Yu. 2017. Multiscale quantization for fast similarity search. NeurIPS 30 (2017), 5745-5755.

[47] Xiaoliang Xu, Mengzhao Wang, Yuxiang Wang, and Dingcheng Ma. 2021. Two-stage routing with optimized guided search and greedy algorithm on proximity graph. Knowledge-Based Systems 229 (2021), 107305.

[48] Hsiang-Fu Yu, Cho-Jui Hsieh, Qi Lei, and Inderjit S Dhillon. 2017. A greedy approach for budgeted maximum inner product search. Advances in neural information processing systems 30 (2017).

[49] Hsiang-Fu Yu, Kai Zhong, and Inderjit S Dhillon. 2020. PECOS: Prediction for Enormous and Correlated Output Spaces. arXiv preprint arXiv:2010.05878 (2020).

[50] Shuai Zhang, Lina Yao, Aixin Sun, and Yi Tay. 2019. Deep learning based recommender system: A survey and new perspectives. ACM Computing Surveys (CSUR) 52, 1 (2019), 1-38.

<!-- Media -->

<!-- figureText: 30000 fashion-mnist-784-euclidean 40000 sift-128-euclidean 5000 gist-960-euclidean HNSW-FINGER HNSW-FINGER HNSW(hnswlib) HNSW(hnswlib) HNSW(nmslib) HNSW(nmslib) HNSW(pecos) 4000 HNSW(pecos) HNSW(n2) Throughoput (#queries/sec) HNSW(n2) NGT-PANNG PyNNDescent Vamana 2000 NGT-PANNG PyNNDescent Vamana 1000 0.80 0.85 0.90 0.95 1.00 0.5 0.6 0.7 0.9 Recall10@10 Recall10@10 glove-100-angular 25000 deep-image-96-angular HNSW-FINGER HNSW-FINGER HNSW(hnswlib) HNSW(hnswlib) HNSW(nmslib) HNSW(nmslib) HNSW(pecos) 20000 HNSW(pecos) HNSW(n2) Throughoput (#queries/sec) HNSW(n2) NGT-PANNG PyNNDescent 15000 Vamana 10000 NGT-PANNG PyNNDescent Vamana 5000 1.0 0 0.6 Recall10@10 Recall10@10 35000 25000 Throughoput (#queries/sec) 20000 Throughoput (#queries/sec) 30000 25000 15000 10000 15000 HNSW-FINGER HNSW(hnswlib) 10000 HNSW(nmslib) HNSW(pecos) HNSW(n2) NGT-PANNG PyNNDescent 5000 Vamana 0.72 0.80 0.85 0.90 0.95 1.00 0.65 0.70 75 Recall10@10 20000 nytimes-256-angular 25000 HNSW-FINGER 17500 HNSW(hnswlib HNSW(nmslib) Throughoput (#queries/sec) HNSW(pecos) 20000 15000 HNSW(n2) Throughoput (#queries/sec) 15000 NGT-PANNG 12500 PyNNDescent Vamana 10000 7500 5000 5000 0.6 0.7 Recall10@10 -->

<img src="https://cdn.noedgeai.com/0195c754-cd29-73d1-bc23-778cbb503ea4_9.jpg?x=159&y=248&w=1487&h=1002&r=0"/>

Figure 7: Experimental results of all graph-based methods. Throughput versus Recall@10 chart is plotted for all datasets. Top row presents datasets with ${L2}$ distance measure and bottom row presents datasets with angular distance measure. We can observe a significant performance gain of HNSW-FINGER over existing graph-based methods. Best viewed in color.

图7：所有基于图的方法的实验结果。为所有数据集绘制了吞吐量与召回率@10的图表。顶行展示了使用${L2}$距离度量的数据集，底行展示了使用角度距离度量的数据集。我们可以观察到HNSW - FINGER相对于现有基于图的方法有显著的性能提升。彩色视图效果最佳。

<!-- Media -->

## A COMPLETE COMPARISON OF GRAPH-BASED METHODS

## 基于图的方法的完整比较

Complete results of all graph-based methods are shown in Figure 7. HNSW-FINGER basically outperforms all existing graph-based methods except on FashionMNIST-60K-784 where PyNNDescent performs extremely well. Results show that currently no graph-based methods completely exploits the training data distribution. This reflects the importance of the inference acceleration methods as FINGER that can create consistently faster inference on all underlying search graph. Making a search graph maximally suitable for applying FINGER is also an interesting future direction. In principle, FINGER could also be applied on PyNNDescent to further improve the result. For example,applying FINGER code on kNN graphs from PyNNDescent. At Recall@10 of 99%, on the Fashion-MNIST dataset, FINGER improved the throughput by 40% over the original PyNNDescent and HNSW. For the SIFT dataset, FINGER improved the throughput by ${20}\%$ and ${25}\%$ over the original PyNNDescent and HNSW, respectively.

所有基于图的方法的完整结果如图7所示。除了在FashionMNIST - 60K - 784数据集上PyNNDescent表现极其出色外，HNSW - FINGER基本上优于所有现有的基于图的方法。结果表明，目前没有一种基于图的方法能完全利用训练数据的分布。这反映了像FINGER这样的推理加速方法的重要性，它可以在所有底层搜索图上实现持续更快的推理。使搜索图最大程度地适合应用FINGER也是一个有趣的未来研究方向。原则上，FINGER也可以应用于PyNNDescent以进一步改善结果。例如，将FINGER代码应用于PyNNDescent生成的k近邻图。在Fashion - MNIST数据集上，当召回率@10为99%时，FINGER相对于原始的PyNNDescent和HNSW将吞吐量提高了40%。对于SIFT数据集，FINGER相对于原始的PyNNDescent和HNSW分别将吞吐量提高了${20}\%$和${25}\%$。

## B DETAILED STEPS OF FINGER APPROXIMATION

## B 手指近似法的详细步骤

As shown in Eq.(2), the L2 distance between q and d can be written as:

如式(2)所示，q 和 d 之间的 L2 距离可以写成：

$$
\parallel q - d{\parallel }_{2}^{2} = {\begin{Vmatrix}\left( {q}_{\text{proj }} - {d}_{\text{proj }}\right) \end{Vmatrix}}_{2}^{2} + {\begin{Vmatrix}{q}_{\text{res }}\end{Vmatrix}}_{2}^{2} + {\begin{Vmatrix}{d}_{\text{res }}\end{Vmatrix}}_{2}^{2} - 2{q}_{\text{res }}^{T}{d}_{\text{res }},
$$

where ${q}_{\text{proj }},{d}_{\text{proj }},{q}_{\text{res }},{q}_{\text{res }}$ are obtained by projecting onto vector of center node. We will explain each term individually and use bold text to denote information could be pre-computed and stored. We will also analyze number of arithmetic and memory read operations needed for the whole algorithm.

其中 ${q}_{\text{proj }},{d}_{\text{proj }},{q}_{\text{res }},{q}_{\text{res }}$ 是通过投影到中心节点的向量得到的。我们将分别解释每一项，并使用粗体文本表示可以预先计算和存储的信息。我们还将分析整个算法所需的算术运算和内存读取操作的数量。

- ${\begin{Vmatrix}\left( {q}_{proj} - {d}_{proj}\right) \end{Vmatrix}}_{2}^{2}$ : Since ${q}_{proj}$ and ${d}_{proj}$ are projections of $q$ and $d$ onto the vector of the center node. Without loss of generality, we can write it as ${q}_{proj} = {tc}$ and ${d}_{proj} = {bc}$ ,where $c$ is the center vector and $t,b$ are scalars. ${\begin{Vmatrix}\left( {q}_{\text{proj }} - {d}_{\text{proj }}\right) \end{Vmatrix}}_{2}^{2}$ then becomes ${\left( t - b\right) }^{2}\parallel c{\parallel }_{2}^{2}$ . We can pre-compute $\parallel c{\parallel }_{2}^{2}$ for each node and ${\left( t - b\right) }^{2}$ is just a subtraction plus a multiplication to itself. $b$ for each neighboring node can also be pre-calculated. To get t, recall the projection formula: $t = \frac{{q}^{T}c}{\parallel c{\parallel }_{2}^{2}}$ . The denominator $\parallel c{\parallel }_{2}^{2}$

- ${\begin{Vmatrix}\left( {q}_{proj} - {d}_{proj}\right) \end{Vmatrix}}_{2}^{2}$ ：由于 ${q}_{proj}$ 和 ${d}_{proj}$ 分别是 $q$ 和 $d$ 在中心节点向量上的投影。不失一般性，我们可以将其写成 ${q}_{proj} = {tc}$ 和 ${d}_{proj} = {bc}$ ，其中 $c$ 是中心向量，$t,b$ 是标量。那么 ${\begin{Vmatrix}\left( {q}_{\text{proj }} - {d}_{\text{proj }}\right) \end{Vmatrix}}_{2}^{2}$ 就变成了 ${\left( t - b\right) }^{2}\parallel c{\parallel }_{2}^{2}$ 。我们可以为每个节点预先计算 $\parallel c{\parallel }_{2}^{2}$ ，而 ${\left( t - b\right) }^{2}$ 只是一次减法再加上一次自乘。每个相邻节点的 $b$ 也可以预先计算。为了得到 t，回顾投影公式：$t = \frac{{q}^{T}c}{\parallel c{\parallel }_{2}^{2}}$ 。分母 $\parallel c{\parallel }_{2}^{2}$

is pre-computed so we only need to get the result of the inner-product between $\mathrm{q}$ and $\mathrm{c}$ . In Section 3.2,we explained when we explore neighbors of center node, we must have visited it before so the value $\parallel q - c{\parallel }_{2}^{2}$ is stored in candidate queue; therefore, we can get ${q}^{T}c = \frac{\parallel q{\parallel }^{2} + \parallel c{\parallel }^{2} - \parallel q - c{\parallel }^{2}}{2}$ ,and thus $t = \frac{{q}^{T}c}{\parallel c{\parallel }_{2}^{2}}$ with simple calculations. Notice that $\parallel q{\parallel }_{2}^{2}$ is a one time cost for all nodes the cost is negligible when dataset is moderately large. In total, we need 3 memory reads and 6 arithmetic to complete this step.

是预先计算好的，所以我们只需要得到 $\mathrm{q}$ 和 $\mathrm{c}$ 的内积结果。在 3.2 节中，我们解释过当我们探索中心节点的邻居时，我们之前一定已经访问过它，所以值 $\parallel q - c{\parallel }_{2}^{2}$ 存储在候选队列中；因此，我们可以通过简单的计算得到 ${q}^{T}c = \frac{\parallel q{\parallel }^{2} + \parallel c{\parallel }^{2} - \parallel q - c{\parallel }^{2}}{2}$ ，进而得到 $t = \frac{{q}^{T}c}{\parallel c{\parallel }_{2}^{2}}$ 。注意，$\parallel q{\parallel }_{2}^{2}$ 对于所有节点来说是一次性成本，当数据集适中时，这个成本可以忽略不计。总共，我们需要 3 次内存读取和 6 次算术运算来完成这一步。

- $\parallel$ dres ${\parallel }_{2}^{2} : \parallel$ dres ${\parallel }_{2}^{2}$ can be pre-computed and stored as a single floating point so no computation is needed here. It costs 1 memory read.

- $\parallel$ dres ${\parallel }_{2}^{2} : \parallel$ dres ${\parallel }_{2}^{2}$ 可以预先计算并存储为一个单精度浮点数，因此这里不需要进行计算。这需要 1 次内存读取。

- ${\begin{Vmatrix}{q}_{\text{res }}\end{Vmatrix}}_{2}^{2}$ : From above,we know that ${\begin{Vmatrix}{q}_{\text{proj }}\end{Vmatrix}}_{2}^{2} = {t}^{2}\parallel c{\parallel }_{2}^{2}$ ,and we have already loaded the pre-computed $\parallel c{\parallel }_{2}^{2}$ . Thus to get ${\begin{Vmatrix}{q}_{\text{proj }}\end{Vmatrix}}_{2}^{2}$ ,we need 2 multiplications. Since $\parallel q{\parallel }_{2}^{2} = {\begin{Vmatrix}{q}_{\text{proj }}\end{Vmatrix}}_{2}^{2} +$ ${\begin{Vmatrix}{q}_{res}\end{Vmatrix}}_{2}^{2}$ ,we can get ${\begin{Vmatrix}{q}_{res}\end{Vmatrix}}_{2}^{2} = {\begin{Vmatrix}{q}_{proj}\end{Vmatrix}}_{2}^{2} - \parallel q{\parallel }_{2}^{2}$ ,by an additional subtraction. In total, it costs 3 arithmetic.

- ${\begin{Vmatrix}{q}_{\text{res }}\end{Vmatrix}}_{2}^{2}$ ：从上面我们知道 ${\begin{Vmatrix}{q}_{\text{proj }}\end{Vmatrix}}_{2}^{2} = {t}^{2}\parallel c{\parallel }_{2}^{2}$ ，并且我们已经加载了预先计算好的 $\parallel c{\parallel }_{2}^{2}$ 。因此，为了得到 ${\begin{Vmatrix}{q}_{\text{proj }}\end{Vmatrix}}_{2}^{2}$ ，我们需要进行 2 次乘法运算。由于 $\parallel q{\parallel }_{2}^{2} = {\begin{Vmatrix}{q}_{\text{proj }}\end{Vmatrix}}_{2}^{2} +$ ${\begin{Vmatrix}{q}_{res}\end{Vmatrix}}_{2}^{2}$ ，我们可以通过额外的一次减法得到 ${\begin{Vmatrix}{q}_{res}\end{Vmatrix}}_{2}^{2} = {\begin{Vmatrix}{q}_{proj}\end{Vmatrix}}_{2}^{2} - \parallel q{\parallel }_{2}^{2}$ 。总共，这需要 3 次算术运算。

- ${q}_{res}^{T}{d}_{res}$ : we get this term by using ${\begin{Vmatrix}{q}_{res}\end{Vmatrix}}_{2}{\begin{Vmatrix}{d}_{res}\end{Vmatrix}}_{2}\cos \left( {{q}_{res},{d}_{res}}\right)$ . Given the LSH basis $B,\operatorname{sgn}\left( {{d}_{\text{res }}^{T}B}\right)$ can be pre-computed as saved as compact binary representations. To get ${q}_{\text{res }}^{T}B$ ,recall ${q}_{\text{res }} = q - {q}_{\text{proj }}$ ,so ${q}_{\text{res }}^{T}B = {q}^{T}B - {q}_{\text{proj }}^{T}B = {q}^{T}B - t{c}^{T}B =$ ${q}^{T}B - \frac{{q}^{T}c}{\parallel c{\parallel }_{2}^{2}}{c}^{T}B$ . results of ${c}^{T}B$ can be pre-computed,and we have already calculated $\frac{{q}^{T}c}{\parallel c{\parallel }_{2}^{2}}$ ,so we can get ${q}_{res}^{T}B$ by $r$ subtractions of ${q}_{\text{proj }}^{T}B$ from ${q}^{T}B$ ,where $r$ is the number of LSH basis used. Again,computing ${q}^{T}B$ is also a one time cost for all nodes in the search of a query, so the cost is negligible when dataset is moderately large. After getting ${q}_{\text{res }}^{T}B$ ,we can take its sign and we are ready to estimate angles. Notice that this whole process needs to be done only once for a center node exploration. When number of edges is moderately large (i.e., 32 or 64), this cost is almost negligible for each neighboring node. Without loss of generality, we assume calculating hamming distance between $\operatorname{sgn}\left( {{d}_{res}^{T}B}\right)$ and $\operatorname{sgn}\left( {{q}_{res}^{T}B}\right)$ costs $r$ arithmetic.

- ${q}_{res}^{T}{d}_{res}$ ：我们通过使用${\begin{Vmatrix}{q}_{res}\end{Vmatrix}}_{2}{\begin{Vmatrix}{d}_{res}\end{Vmatrix}}_{2}\cos \left( {{q}_{res},{d}_{res}}\right)$得到这一项。给定局部敏感哈希（LSH，Locality-Sensitive Hashing）基$B,\operatorname{sgn}\left( {{d}_{\text{res }}^{T}B}\right)$可以预先计算并保存为紧凑的二进制表示。为了得到${q}_{\text{res }}^{T}B$，回顾${q}_{\text{res }} = q - {q}_{\text{proj }}$，所以${q}_{\text{res }}^{T}B = {q}^{T}B - {q}_{\text{proj }}^{T}B = {q}^{T}B - t{c}^{T}B =$ ${q}^{T}B - \frac{{q}^{T}c}{\parallel c{\parallel }_{2}^{2}}{c}^{T}B$ 。${c}^{T}B$的结果可以预先计算，并且我们已经计算出$\frac{{q}^{T}c}{\parallel c{\parallel }_{2}^{2}}$，因此我们可以通过从${q}^{T}B$中减去${q}_{\text{proj }}^{T}B$ $r$次来得到${q}_{res}^{T}B$，其中$r$是所使用的局部敏感哈希基的数量。同样，对于一次查询搜索中的所有节点，计算${q}^{T}B$也只需进行一次，因此当数据集规模适中时，该计算成本可以忽略不计。得到${q}_{\text{res }}^{T}B$后，我们可以取其符号，然后就可以估计角度了。请注意，对于中心节点的探索，整个过程只需进行一次。当边的数量适中（即32或64）时，对于每个相邻节点，该成本几乎可以忽略不计。不失一般性，我们假设计算$\operatorname{sgn}\left( {{d}_{res}^{T}B}\right)$和$\operatorname{sgn}\left( {{q}_{res}^{T}B}\right)$之间的汉明距离需要$r$次算术运算。

We can pre-compute ${\begin{Vmatrix}{d}_{res}\end{Vmatrix}}_{2}$ and ${q}_{res}$ has been calculated above. So we just need 2 more multiplications to get ${q}_{res}^{T}{d}_{res}$ . In total,we need $r + \frac{r}{32} + 1$ memory reads, $r + 2$ arithmetic to complete this step.

我们可以预先计算${\begin{Vmatrix}{d}_{res}\end{Vmatrix}}_{2}$，并且上面已经计算出${q}_{res}$。因此，我们只需再进行2次乘法运算就可以得到${q}_{res}^{T}{d}_{res}$。总共，我们需要进行$r + \frac{r}{32} + 1$次内存读取和$r + 2$次算术运算来完成此步骤。

Since $\parallel q - d{\parallel }_{2}^{2} = {\begin{Vmatrix}\left( {q}_{\text{proj }} - {d}_{\text{proj }}\right) \end{Vmatrix}}_{2}^{2} + {\begin{Vmatrix}{q}_{\text{res }}\end{Vmatrix}}_{2}^{2} + {\begin{Vmatrix}{d}_{\text{res }}\end{Vmatrix}}_{2}^{2} - 2{q}_{\text{res }}^{T}{d}_{\text{res }}$ , we need 4 more arithmetic to combine all above terms. Thus in total it costs $r + \frac{r}{8} + 5$ memory reads and $r + {15}$ arithmetic to complete the computation. Consider a full dimensional L2 distance on SIFT- $1\mathrm{M} - {128}$ dataset. Recall the data dimension $m = {128}$ and we use $r = {64}$ . L2 distance requires 128 memory reads,128 subtractions, 128 multiplications and 127 additions. For approximation distance, in total we only need 71 memory reads and 80 arithmetic. We can observe that approximation distance used much less operations so it will be much faster.

由于$\parallel q - d{\parallel }_{2}^{2} = {\begin{Vmatrix}\left( {q}_{\text{proj }} - {d}_{\text{proj }}\right) \end{Vmatrix}}_{2}^{2} + {\begin{Vmatrix}{q}_{\text{res }}\end{Vmatrix}}_{2}^{2} + {\begin{Vmatrix}{d}_{\text{res }}\end{Vmatrix}}_{2}^{2} - 2{q}_{\text{res }}^{T}{d}_{\text{res }}$，我们需要再进行4次算术运算来合并上述所有项。因此，总共需要$r + \frac{r}{8} + 5$次内存读取和$r + {15}$次算术运算来完成计算。考虑在SIFT - $1\mathrm{M} - {128}$数据集上进行全维度的L2距离计算。回顾数据维度$m = {128}$，并且我们使用$r = {64}$。L2距离计算需要128次内存读取、128次减法、128次乘法和127次加法。对于近似距离计算，总共我们只需要71次内存读取和80次算术运算。我们可以观察到，近似距离计算使用的运算次数少得多，因此速度会快得多。

Time Complexity Analysis. In theory, the time complexity of graph search is linear to the number of visited nodes times the distance computation cost between query and each node. The former is query dependent (non analytic), while the latter is where the improvement is made in this paper. Specifically, the time complexity of distance computation is reduced from $\mathrm{O}\left( \mathrm{d}\right)$ to $\mathrm{O}\left( \mathrm{r}\right)$ ,where $\mathrm{r}$ is the low rank used in LSH (Lemma 1) and $\mathrm{d}$ is the original data dimension.

时间复杂度分析。理论上，图搜索的时间复杂度与访问节点的数量乘以查询与每个节点之间的距离计算成本成正比。前者依赖于查询（非解析的），而后者正是本文所做改进之处。具体而言，距离计算的时间复杂度从$\mathrm{O}\left( \mathrm{d}\right)$降低到$\mathrm{O}\left( \mathrm{r}\right)$，其中$\mathrm{r}$是局部敏感哈希（LSH）中使用的低秩（引理1），$\mathrm{d}$是原始数据维度。

## C MEMORY FOOTPRINT AND OVERHEAD OF HNSW-FINGER AND HNSW

## C HNSW - FINGER和HNSW的内存占用与开销

As illustrated in Section B,we pre-compute and store $r$ floating points in ${c}^{T}B$ ( $r \times  4$ bytes) and $\parallel c{\parallel }_{2}^{2}$ (1 byte) for each node. For each edge,we pre-compute and store $\frac{r}{8}$ bytes of signed code and projection coefficient $b$ ( 4 bytes) and residual norm ${d}_{res}$ ( 4 bytes). Thus in total for a graph $G = \left( {V,E}\right)$ ,we save additional $\left| V\right|  \times$ $\left( {{4r} + 1}\right)  + \left| E\right|  \times  \left( {\frac{r}{8} + 8}\right)$ bytes. Take GIST-1M-960 with maximal 96 edges per node for example,we use $r = {64},\left| E\right|$ is maximally $1 \times  {96} = {96}$ million edges. This translates into about additional $1{e}^{6} \times  \left( {4 * {64} + 1}\right)  + {96}{e}^{6}\left( {{64}/8 + 8}\right)$ Bytes $\left( { \approx  {1709}\mathrm{{MB}}}\right)$ ,which is about the half whole original $1\mathrm{M}$ training database ( ${3.6}\mathrm{{GB}}$ ). Compared to the original HNSW model (4.5GB), the additional cost of FINGER is acceptable. In particular, if we use full precision low-rank model, even for $r = {16}$ ,it will cost $\left| E\right|  \times  {16} \times  4$ Bytes $\left( { \approx  {5859}\mathrm{{MB}}}\right)$ . We can use much more basis in RPLSH setup with less memory footprint. Notice that this setup is perhaps the largest working search index for all the 6 datasets used in this paper. Best performing graph mostly will not need to have more than 96 edges and 64 basis. Thus, in practice, additional storage is about a constant of original training vector size. In terms of pre-processing time, with the same hardware configuration, on the GIST dataset, the time overhead of FINGER and TOGG-KMC over HNSW index building are 10.08% and ${11.05}\%$ ,respectively,which is not very time consuming compared to previous baseline methods.

如B节所述，我们为每个节点预先计算并存储$r$个浮点数，分别存储在${c}^{T}B$（$r \times  4$字节）和$\parallel c{\parallel }_{2}^{2}$（1字节）中。对于每条边，我们预先计算并存储$\frac{r}{8}$字节的有符号代码、投影系数$b$（4字节）和残差范数${d}_{res}$（4字节）。因此，对于图$G = \left( {V,E}\right)$，总共额外节省了$\left| V\right|  \times$$\left( {{4r} + 1}\right)  + \left| E\right|  \times  \left( {\frac{r}{8} + 8}\right)$字节。以每个节点最多有96条边的GIST - 1M - 960为例，我们使用的$r = {64},\left| E\right|$最多有$1 \times  {96} = {96}$百万条边。这相当于大约额外的$1{e}^{6} \times  \left( {4 * {64} + 1}\right)  + {96}{e}^{6}\left( {{64}/8 + 8}\right)$字节$\left( { \approx  {1709}\mathrm{{MB}}}\right)$，约为整个原始$1\mathrm{M}$训练数据库（${3.6}\mathrm{{GB}}$）的一半。与原始的HNSW模型（4.5GB）相比，FINGER的额外成本是可以接受的。特别是，如果我们使用全精度低秩模型，即使对于$r = {16}$，也将花费$\left| E\right|  \times  {16} \times  4$字节$\left( { \approx  {5859}\mathrm{{MB}}}\right)$。我们可以在随机投影局部敏感哈希（RPLSH）设置中使用更多的基，同时减少内存占用。请注意，此设置可能是本文使用的所有6个数据集的最大工作搜索索引。性能最佳的图大多不需要超过96条边和64个基。因此，在实践中，额外存储大约是原始训练向量大小的一个常数。在预处理时间方面，在相同的硬件配置下，在GIST数据集上，FINGER和TOGG - KMC相对于HNSW索引构建的时间开销分别为10.08%和${11.05}\%$，与之前的基线方法相比，这并不十分耗时。

## D FORMULATION OF INNER-PRODUCT

## D 内积的公式推导

In the main text,we presented derivation of ${L2}$ distance,and in this section we will derive the approximation for inner-product distance measure. Notice that angle measure can be obtained by firstly normalizing data vectors and then apply inner-product distance and thus the derivation is the same. For a query $q$ and data point $d$ ,inner-product distance measure is ${Dist} = {q}^{T}d$ . Similar to ${L2}$ distance,we can apply the same decomposition to write $q = {q}_{\text{proj }} + {q}_{\text{res }}$ and $d = {d}_{\text{proj }} + {d}_{\text{res }}$ . substituting the decomposition into distance definition, we have

在正文部分，我们给出了${L2}$距离的推导，在本节中，我们将推导内积距离度量的近似公式。请注意，角度度量可以通过先对数据向量进行归一化，然后应用内积距离来获得，因此推导过程是相同的。对于查询$q$和数据点$d$，内积距离度量为${Dist} = {q}^{T}d$。与${L2}$距离类似，我们可以应用相同的分解来表示$q = {q}_{\text{proj }} + {q}_{\text{res }}$和$d = {d}_{\text{proj }} + {d}_{\text{res }}$。将分解代入距离定义中，我们得到

$$
\text{ Dist } = {q}_{\text{proj }}^{T}{d}_{\text{proj }} + {q}_{\text{res }}^{T}{d}_{\text{res }}.
$$

As in ${L2}$ case ${q}_{proj}$ and ${d}_{proj}$ can be obtained by simple operations and the remaining uncertainy term is again ${q}_{res}^{T}{d}_{res}$ . Therefore,in inner-product case, angle between neighboring residual vectors is still the target to approximate.

与${L2}$的情况一样，${q}_{proj}$和${d}_{proj}$可以通过简单的运算得到，而剩余的不确定项仍然是${q}_{res}^{T}{d}_{res}$。因此，在内积的情况下，相邻残差向量之间的角度仍然是需要近似的目标。